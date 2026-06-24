#!/usr/bin/env python3
"""
Weaviate ingest script for Slack channel exports.

Mirrors weaviate_setup_des.py's checkpoint + micro-batch + retry pattern
(via ingest_utils.py) but consumes Slack channel export files instead of
PDFs. See slack_preprocessing.py for the parsing/cleaning/filtering logic.

Expects one JSON file per channel in SLACK_DATA_DIR, each a flat list of
Slack message objects (filename = channel name), e.g.:

  src/data/slack_export/3x2pt_harmonic.json
  src/data/slack_export/general.json
  ...

Typical run:
  WEAVIATE_SLACK_CLASS_NAME=DS_SLACK_DES python3 weaviate_setup_slack.py

Env overrides:
  SLACK_DATA_DIR              default: <script_dir>/data/slack_export
  WEAVIATE_SLACK_CLASS_NAME   default: DS_SLACK_DES
  SLACK_MIN_WORDS             default: 6   (low-value-chatter filter threshold)
  CHUNK_BATCH_SIZE            default: 120
  SLEEP_BETWEEN_BATCHES       default: 0.2
  MAX_RETRIES_PER_BATCH       default: 6
  RUN_POST_INGEST_TESTS       default: 0
"""

from pathlib import Path
import os
import sys
import time

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

import weaviate
from weaviate import AuthApiKey
from langchain_community.vectorstores import Weaviate

from embedding import get_embedding_function
from slack_preprocessing import parse_channel_file
from ingest_utils import iter_batches, retry_sleep, load_checkpoint, save_checkpoint


# ─── CONFIGURATION ──────────────────────────────────────────────────────────
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
CLASS_NAME = os.getenv("WEAVIATE_SLACK_CLASS_NAME", "DS_SLACK_DES")

SCRIPT_DIR = Path(__file__).resolve().parent
SLACK_DATA_DIR = Path(os.getenv("SLACK_DATA_DIR", SCRIPT_DIR / "data" / "slack_export"))

MIN_WORDS = int(os.getenv("SLACK_MIN_WORDS", "6"))
CHUNK_BATCH_SIZE = int(os.getenv("CHUNK_BATCH_SIZE", "120"))
SLEEP_SECONDS_BETWEEN_BATCHES = float(os.getenv("SLEEP_BETWEEN_BATCHES", "0.2"))
MAX_RETRIES_PER_BATCH = int(os.getenv("MAX_RETRIES_PER_BATCH", "6"))
RUN_POST_INGEST_TESTS = os.getenv("RUN_POST_INGEST_TESTS", "0") in ("1", "true", "TRUE", "yes", "YES")

CHECKPOINT_PATH = SCRIPT_DIR / f".ingest_checkpoint_{CLASS_NAME}.txt"

if not WEAVIATE_URL or not WEAVIATE_API_KEY:
    raise ValueError("Missing WEAVIATE_URL or WEAVIATE_API_KEY in .env file")

print(f"🔗 Connecting to Weaviate at {WEAVIATE_URL}")
print(f"🧱 Target class/collection: {CLASS_NAME}")
print(f"📁 Slack export dir: {SLACK_DATA_DIR}")
print(f"📦 Chunk batch size: {CHUNK_BATCH_SIZE}")
print(f"🔎 Min words per unit: {MIN_WORDS}")
print(f"🧾 Checkpoint: {CHECKPOINT_PATH}")


# ─── CONNECT TO WEAVIATE (v3 client) ────────────────────────────────────────
try:
    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=AuthApiKey(api_key=WEAVIATE_API_KEY),
        additional_headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", "")},
    )
    if client.is_ready():
        print("✅ Successfully connected to Weaviate")
    else:
        raise ConnectionError("Weaviate client is not ready")
except Exception as e:
    print(f"❌ Failed to connect to Weaviate: {e}")
    raise

try:
    client.batch.configure(batch_size=CHUNK_BATCH_SIZE, dynamic=True)
except Exception:
    pass


# ─── RESET SCHEMA ───────────────────────────────────────────────────────────
# WARNING: deletes/recreates the class each run (PoC behavior, matches the
# PDF script). Make sure CLASS_NAME is correct before running.
try:
    if client.schema.exists(CLASS_NAME):
        print(f"🔄 Deleting existing class `{CLASS_NAME}`")
        client.schema.delete_class(CLASS_NAME)

    print(f"➕ Creating class `{CLASS_NAME}`")
    client.schema.create_class({
        "class": CLASS_NAME,
        "vectorizer": "none",  # we provide our own vectors
        "properties": [
            {"name": "content",            "dataType": ["text"],    "description": "Chunk text (reconstructed Slack thread/message transcript)"},
            {"name": "channel",            "dataType": ["text"],    "description": "Slack channel name (source JSON filename)"},
            {"name": "source",             "dataType": ["text"],    "description": "Logical source tag, e.g. slack:3x2pt_harmonic"},
            {"name": "doc_id",             "dataType": ["text"],    "description": "Stable id for the unit (channel + thread-root timestamp)"},
            {"name": "chunk_id",           "dataType": ["text"],    "description": "Stable chunk id (uuid5)"},
            {"name": "chunk_index",        "dataType": ["int"],     "description": "Index of chunk within the unit"},
            {"name": "text_hash",          "dataType": ["text"],    "description": "SHA-256 of chunk text (debug/dedupe)"},
            {"name": "ts_start",           "dataType": ["text"],    "description": "Timestamp (UTC) of first message in unit"},
            {"name": "ts_end",             "dataType": ["text"],    "description": "Timestamp (UTC) of last message in unit"},
            {"name": "message_count",      "dataType": ["int"],     "description": "Number of Slack messages combined into this unit"},
            {"name": "participants",       "dataType": ["text"],    "description": "Comma-separated display names of participants"},
            {"name": "unit_type",          "dataType": ["text"],    "description": "single | thread_complete | thread_partial | thread_orphan_reply"},
            {"name": "reaction_count",     "dataType": ["int"],     "description": "Total emoji reactions across the unit"},
            {"name": "has_thanks_reaction","dataType": ["boolean"], "description": "True if a positive/confirming reaction was found"},
        ],
    })
    print(f"✅ Class `{CLASS_NAME}` created successfully")
except Exception as e:
    print(f"❌ Schema operation failed: {e}")
    raise


# ─── EMBEDDING FUNCTION ─────────────────────────────────────────────────────
print("🔧 Setting up embedding function and vector store...")
try:
    EMBED_FN = get_embedding_function()
    print("✅ Embedding function initialized")
except Exception as e:
    print(f"❌ Failed to initialize embedding function: {e}")
    raise

try:
    store = Weaviate(
        client=client,
        index_name=CLASS_NAME,
        text_key="content",
        attributes=[
            "channel", "source", "doc_id", "chunk_id", "chunk_index", "text_hash",
            "ts_start", "ts_end", "message_count", "participants", "unit_type",
            "reaction_count", "has_thanks_reaction",
        ],
        embedding=EMBED_FN,
        by_text=False,
    )
    print("✅ Vector store initialized")
except Exception as e:
    print(f"❌ Failed to initialize vector store: {e}")
    raise


# ─── INGEST SLACK CHANNEL FILES ─────────────────────────────────────────────
if not SLACK_DATA_DIR.exists():
    print(f"❌ Slack export directory not found: {SLACK_DATA_DIR}")
    sys.exit(1)

channel_files = sorted(SLACK_DATA_DIR.glob("*.json"))
if not channel_files:
    print(f"⚠️  No channel JSON files found in {SLACK_DATA_DIR}")
    sys.exit(1)

done = load_checkpoint(CHECKPOINT_PATH)
print(f"\n📂 Found {len(channel_files)} channel file(s) in {SLACK_DATA_DIR}")
print(f"✅ Checkpoint has {len(done)} channel(s) already ingested for class {CLASS_NAME}")
print("=" * 60)

total_chunks = 0
processed_channels = 0
skipped_low_value_total = 0
unit_type_totals = {}

for idx, channel_file in enumerate(channel_files, start=1):
    channel_name = channel_file.stem

    if channel_name in done:
        print(f"\n[{idx}/{len(channel_files)}] Skipping already ingested: {channel_name}")
        continue

    try:
        print(f"\n[{idx}/{len(channel_files)}] Processing channel: {channel_name}")

        documents = parse_channel_file(channel_file, min_words=MIN_WORDS)
        print(f"   ✂️  Produced {len(documents)} chunk(s)")

        for d in documents:
            ut = d.metadata.get("unit_type", "unknown")
            unit_type_totals[ut] = unit_type_totals.get(ut, 0) + 1

        ids = [d.metadata["chunk_id"] for d in documents]

        for b_i, (doc_batch, id_batch) in enumerate(
            zip(iter_batches(documents, CHUNK_BATCH_SIZE), iter_batches(ids, CHUNK_BATCH_SIZE)),
            start=1,
        ):
            attempt = 0
            while True:
                try:
                    store.add_documents(doc_batch, ids=id_batch)
                    break
                except Exception as e:
                    attempt += 1
                    if attempt > MAX_RETRIES_PER_BATCH:
                        raise
                    msg = str(e).lower()
                    print(f"   ⚠️ Batch {b_i}: error (attempt {attempt}/{MAX_RETRIES_PER_BATCH}): {e}")
                    if "rate" in msg or "timeout" in msg or "429" in msg or "temporar" in msg:
                        retry_sleep(attempt)
                    else:
                        time.sleep(2.0)

            if SLEEP_SECONDS_BETWEEN_BATCHES > 0:
                time.sleep(SLEEP_SECONDS_BETWEEN_BATCHES)

        total_chunks += len(documents)
        processed_channels += 1
        save_checkpoint(CHECKPOINT_PATH, channel_name)
        print(f"   ✅ Added {len(documents)} chunks (checkpoint saved)")

    except Exception as e:
        print(f"   ❌ Error processing {channel_name}: {e}")
        continue

print("\n" + "=" * 60)
print(f"✅ Ingestion complete: {total_chunks} total chunks from {processed_channels} channel(s) (class={CLASS_NAME})")
print(f"📊 Unit type breakdown across all ingested chunks: {unit_type_totals}")
print(f"🧾 Checkpoint file: {CHECKPOINT_PATH}")

if unit_type_totals.get("thread_partial"):
    print(
        f"\n⚠️  Note: {unit_type_totals['thread_partial']} chunk(s) are from threads where "
        f"replies were reported but not fully captured in the export "
        f"(unit_type='thread_partial'). Consider re-exporting these channels later "
        f"for more complete coverage — see `unit_type` field to find them."
    )


# ─── OPTIONAL POST-INGEST TESTS (OFF BY DEFAULT) ────────────────────────────
if RUN_POST_INGEST_TESTS:
    print("\n🔍 Testing similarity search…")
    print("=" * 60)
    test_query = "How do I access the latest DES catalogs?"
    print(f"Query: '{test_query}'\n")
    try:
        hits = store.similarity_search_with_score(test_query, k=4)
        if not hits:
            print("⚠️  No results found")
        else:
            for i, (doc, score) in enumerate(hits, start=1):
                snippet = doc.page_content.replace("\n", " ").strip()[:200]
                md = doc.metadata
                print(f"{i}. Score: {score:.4f}")
                print(f"   Channel: {md.get('channel')}  |  Type: {md.get('unit_type')}")
                print(f"   Content: {snippet}...\n")
    except Exception as e:
        print(f"❌ Search failed: {e}")

print("🎉 Script completed successfully!")

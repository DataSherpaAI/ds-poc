#!/usr/bin/env python3
"""
Weaviate ingest script (safe for large corpora).

Key features for big ingests (e.g., 1000+ PDFs):
- PDF-by-PDF streaming (does not load entire corpus in memory)
- Micro-batched inserts (prevents huge requests + reduces rate-limit pain)
- Retry with exponential backoff (helps with OpenAI 429/timeouts)
- Checkpointing (resume without starting from scratch)
- Deterministic chunk IDs + metadata for traceability (doc/page/chunk)

Typical runs (on the VM):
  # Create DS_POC from documents_10
  WEAVIATE_CLASS_NAME=DS_POC USE_LARGE_CORPUS=0 python3 weaviate_setup.py

  # Create DS_DES_FULL from documents_100 (or documents_500 / documents_1367 as you name it)
  WEAVIATE_CLASS_NAME=DS_DES_FULL USE_LARGE_CORPUS=1 python3 weaviate_setup.py

You can also override the directory directly:
  DATA_DIR=/home/fabs/ds-poc/src/data/documents_1367 WEAVIATE_CLASS_NAME=DS_DES_FULL python3 weaviate_setup.py
"""

from pathlib import Path
import os
import sys
import time
import hashlib
import uuid

# Load .env (expects repo root if script is under repo/src or repo/scripts)
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

import weaviate
from weaviate import AuthApiKey

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate

# OpenAI-based embedding
from embedding import get_embedding_function


# ─── CONFIGURATION ──────────────────────────────────────────────────────────────
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# Make class selectable at runtime (safer than editing the file)
CLASS_NAME = os.getenv("WEAVIATE_CLASS_NAME", "DS_POC")

# Corpus directory switching (quick PoC toggle) + direct override
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR_SMALL = SCRIPT_DIR / "data" / "documents_10"
DATA_DIR_LARGE = SCRIPT_DIR / "data" / "documents_100"
USE_LARGE_CORPUS = os.getenv("USE_LARGE_CORPUS", "0") in ("1", "true", "TRUE", "yes", "YES")

# Allow direct override (recommended for 1367 docs): export DATA_DIR=/path/to/docs
DATA_DIR_OVERRIDE = os.getenv("DATA_DIR")
if DATA_DIR_OVERRIDE:
    DATA_DIR = Path(DATA_DIR_OVERRIDE).expanduser().resolve()
else:
    DATA_DIR = DATA_DIR_LARGE if USE_LARGE_CORPUS else DATA_DIR_SMALL

# Big-corpus ingest controls
CHUNK_BATCH_SIZE = int(os.getenv("CHUNK_BATCH_SIZE", "120"))  # 80–200 good
SLEEP_SECONDS_BETWEEN_BATCHES = float(os.getenv("SLEEP_BETWEEN_BATCHES", "0.2"))
MAX_RETRIES_PER_BATCH = int(os.getenv("MAX_RETRIES_PER_BATCH", "6"))

# Disable expensive post-ingest tests for large runs
RUN_POST_INGEST_TESTS = os.getenv("RUN_POST_INGEST_TESTS", "0") in ("1", "true", "TRUE", "yes", "YES")

# Checkpoint file (resume ingestion)
CHECKPOINT_PATH = SCRIPT_DIR / f".ingest_checkpoint_{CLASS_NAME}.txt"

# Validate environment variables
if not WEAVIATE_URL or not WEAVIATE_API_KEY:
    raise ValueError("Missing WEAVIATE_URL or WEAVIATE_API_KEY in .env file")

print(f"🔗 Connecting to Weaviate at {WEAVIATE_URL}")
print(f"🧱 Target class/collection: {CLASS_NAME}")
print(f"📁 Data dir: {DATA_DIR}")
print(f"📦 Chunk batch size: {CHUNK_BATCH_SIZE}")
print(f"🧾 Checkpoint: {CHECKPOINT_PATH}")


# ─── STABLE ID HELPERS (chunk identification + dedupe) ──────────────────────────
CHUNK_UUID_NAMESPACE = uuid.UUID("6a2f2c7b-3a3f-4d8d-9b8d-2f6a0f5f2b1a")

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def make_doc_id(pdf_name: str) -> str:
    return sha256_hex(pdf_name)[:16]

def make_text_hash(chunk_text: str) -> str:
    return sha256_hex(chunk_text)

def make_chunk_id(doc_id: str, page: int, chunk_index: int, chunk_text: str) -> str:
    text_hash16 = sha256_hex(chunk_text)[:16]
    raw = f"{doc_id}:{page}:{chunk_index}:{text_hash16}"
    return str(uuid.uuid5(CHUNK_UUID_NAMESPACE, raw))


# ─── CHECKPOINT HELPERS ─────────────────────────────────────────────────────────
def iter_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def load_checkpoint() -> set[str]:
    if CHECKPOINT_PATH.exists():
        return set(CHECKPOINT_PATH.read_text(encoding="utf-8").splitlines())
    return set()

def save_checkpoint(pdf_name: str) -> None:
    with CHECKPOINT_PATH.open("a", encoding="utf-8") as f:
        f.write(pdf_name + "\n")

def retry_sleep(attempt: int) -> None:
    # exponential backoff with cap
    sleep_s = min(60.0, (2 ** attempt) * 1.0)
    time.sleep(sleep_s)


# ─── CONNECT TO WEAVIATE (v3 client) ────────────────────────────────────────────
try:
    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=AuthApiKey(api_key=WEAVIATE_API_KEY),
        additional_headers={
            # Some Weaviate setups use this header for modules; harmless if unused
            "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", "")
        },
    )

    if client.is_ready():
        print("✅ Successfully connected to Weaviate")
    else:
        raise ConnectionError("Weaviate client is not ready")

except Exception as e:
    print(f"❌ Failed to connect to Weaviate: {e}")
    raise

# Configure batching (best-effort; may vary by client version)
try:
    client.batch.configure(batch_size=CHUNK_BATCH_SIZE, dynamic=True)
except Exception:
    pass


# ─── RESET SCHEMA ──────────────────────────────────────────────────────────────
# WARNING: This deletes/recreates the class each run (PoC behavior).
# Make sure CLASS_NAME is correct before running.
try:
    if client.schema.exists(CLASS_NAME):
        print(f"🔄 Deleting existing class `{CLASS_NAME}`")
        client.schema.delete_class(CLASS_NAME)

    print(f"➕ Creating class `{CLASS_NAME}`")
    client.schema.create_class({
        "class": CLASS_NAME,
        "vectorizer": "none",  # we provide our own vectors
        "properties": [
            {"name": "content",     "dataType": ["text"], "description": "Chunk text"},
            {"name": "source",      "dataType": ["text"], "description": "Source PDF filename"},
            {"name": "doc_id",      "dataType": ["text"], "description": "Stable document id (hash of filename)"},
            {"name": "chunk_id",    "dataType": ["text"], "description": "Stable chunk id (uuid5)"},
            {"name": "page",        "dataType": ["int"],  "description": "Page number in source PDF"},
            {"name": "chunk_index", "dataType": ["int"],  "description": "Index of chunk within the PDF"},
            {"name": "text_hash",   "dataType": ["text"], "description": "SHA-256 of chunk text (debug/dedupe)"},
        ],
    })
    print(f"✅ Class `{CLASS_NAME}` created successfully")

except Exception as e:
    print(f"❌ Schema operation failed: {e}")
    raise


# ─── SETTING EMBEDDING FUNCTION ────────────────────────────────────────────────
print("🔧 Setting up embedding function and vector store...")
try:
    EMBED_FN = get_embedding_function()
    print("✅ Embedding function initialized")
except Exception as e:
    print(f"❌ Failed to initialize embedding function: {e}")
    raise


def test_embedding_function() -> bool:
    print("\n🧪 Testing embedding function...")
    try:
        test_text = "The Dark Energy Survey studies cosmic acceleration"
        embedding = EMBED_FN.embed_query(test_text)
        print(f"   ✅ Embedding generated: {len(embedding)} dimensions")
        print(f"   📊 Sample values: [{embedding[0]:.4f}, {embedding[1]:.4f}, ...]")
        return True
    except Exception as e:
        print(f"   ❌ Embedding test failed: {e}")
        return False


if not test_embedding_function():
    print("⚠️  Embedding function test failed; aborting.")
    sys.exit(1)


# ─── VECTOR STORE SETUP ────────────────────────────────────────────────────────
try:
    store = Weaviate(
        client=client,
        index_name=CLASS_NAME,
        text_key="content",
        attributes=["source", "doc_id", "chunk_id", "page", "chunk_index", "text_hash"],
        embedding=EMBED_FN,
        by_text=False,
    )
    print("✅ Vector store initialized")
except Exception as e:
    print(f"❌ Failed to initialize vector store: {e}")
    raise


# ─── INGEST PDFS ───────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)

if not DATA_DIR.exists():
    print(f"❌ Data directory not found: {DATA_DIR}")
    sys.exit(1)

pdfs = sorted(DATA_DIR.glob("*.pdf"))
if not pdfs:
    print(f"⚠️  No PDF files found in {DATA_DIR}")
    sys.exit(1)

done = load_checkpoint()
print(f"\n📂 Found {len(pdfs)} PDF(s) in {DATA_DIR}")
print(f"✅ Checkpoint has {len(done)} PDF(s) already ingested for class {CLASS_NAME}")
print("=" * 60)

total_chunks = 0
processed_pdfs = 0

for idx, pdf in enumerate(pdfs, start=1):
    if pdf.name in done:
        print(f"\n[{idx}/{len(pdfs)}] Skipping already ingested: {pdf.name}")
        continue

    try:
        print(f"\n[{idx}/{len(pdfs)}] Processing: {pdf.name}")

        # Load PDF: returns page-level documents with metadata like {"source": ..., "page": ...}
        docs = PyPDFLoader(str(pdf)).load()
        print(f"   📄 Loaded {len(docs)} page(s)")

        # Split into chunks
        chunks = splitter.split_documents(docs)
        print(f"   ✂️  Split into {len(chunks)} chunk(s)")

        doc_id = make_doc_id(pdf.name)

        # Prepare metadata + deterministic IDs
        ids = []
        for chunk_i, chunk in enumerate(chunks):
            text = chunk.page_content
            page = int(chunk.metadata.get("page", -1))

            chunk_id = make_chunk_id(doc_id=doc_id, page=page, chunk_index=chunk_i, chunk_text=text)
            text_hash = make_text_hash(text)

            chunk.metadata["source"] = pdf.name
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["page"] = page
            chunk.metadata["chunk_index"] = chunk_i
            chunk.metadata["text_hash"] = text_hash

            ids.append(chunk_id)

        # Micro-batch ingestion (critical for large corpora)
        for b_i, (chunk_batch, id_batch) in enumerate(
            zip(iter_batches(chunks, CHUNK_BATCH_SIZE), iter_batches(ids, CHUNK_BATCH_SIZE)),
            start=1,
        ):
            attempt = 0
            while True:
                try:
                    store.add_documents(chunk_batch, ids=id_batch)
                    break
                except Exception as e:
                    attempt += 1
                    if attempt > MAX_RETRIES_PER_BATCH:
                        raise

                    msg = str(e).lower()
                    print(f"   ⚠️ Batch {b_i}: error (attempt {attempt}/{MAX_RETRIES_PER_BATCH}): {e}")

                    # Backoff helps for OpenAI rate limits and transient network issues
                    if "rate" in msg or "timeout" in msg or "429" in msg or "temporar" in msg:
                        retry_sleep(attempt)
                    else:
                        time.sleep(2.0)

            if SLEEP_SECONDS_BETWEEN_BATCHES > 0:
                time.sleep(SLEEP_SECONDS_BETWEEN_BATCHES)

        total_chunks += len(chunks)
        processed_pdfs += 1
        save_checkpoint(pdf.name)
        print(f"   ✅ Added {len(chunks)} chunks (checkpoint saved)")

    except Exception as e:
        print(f"   ❌ Error processing {pdf.name}: {e}")
        continue

print("\n" + "=" * 60)
print(f"✅ Ingestion complete: {total_chunks} total chunks from {processed_pdfs} PDFs (class={CLASS_NAME})")
print(f"🧾 Checkpoint file: {CHECKPOINT_PATH}")


# ─── OPTIONAL POST-INGEST TESTS (OFF BY DEFAULT) ───────────────────────────────
if RUN_POST_INGEST_TESTS:
    print("\n📊 Chunk Quality Analysis:")
    print("=" * 60)

    sample_chunks = store.similarity_search("DES project overview", k=3)
    for i, chunk in enumerate(sample_chunks, 1):
        content = chunk.page_content
        md = chunk.metadata

        print(f"\nChunk {i}:")
        print(f"  Length: {len(content)} characters (~{len(content.split())} words)")
        print(f"  Source: {md.get('source', 'unknown')}")
        print(f"  Doc ID: {md.get('doc_id', 'unknown')}")
        print(f"  Chunk ID: {md.get('chunk_id', 'unknown')}")
        print(f"  Page: {md.get('page', 'unknown')}, Chunk Index: {md.get('chunk_index', 'unknown')}")
        print(f"  Text hash: {str(md.get('text_hash', ''))[:16]}...")
        print(f"  Preview: {content[:150]}...")

        if not content.strip().endswith((".", "!", "?", "\n")):
            print("  ⚠️  Warning: Chunk may be cut mid-sentence")

    print("\n🔍 Testing similarity search…")
    print("=" * 60)

    test_query = "How many documents do you currently have access to?"
    print(f"Query: '{test_query}'\n")

    try:
        hits = store.similarity_search_with_score(test_query, k=4)

        if not hits:
            print("⚠️  No results found")
        else:
            print(f"📊 Top {len(hits)} matches:\n")
            for i, (doc, score) in enumerate(hits, start=1):
                snippet = doc.page_content.replace("\n", " ").strip()[:200]
                md = doc.metadata

                print(f"{i}. Score: {score:.4f}")
                print(f"   Source: {md.get('source', 'unknown')}")
                print(f"   Doc ID: {md.get('doc_id', 'unknown')}")
                print(f"   Chunk ID: {md.get('chunk_id', 'unknown')}")
                print(f"   Page: {md.get('page', 'unknown')}, Chunk Index: {md.get('chunk_index', 'unknown')}")
                print(f"   Content: {snippet}...\n")

    except Exception as e:
        print(f"❌ Search failed: {e}")

print("🎉 Script completed successfully!")

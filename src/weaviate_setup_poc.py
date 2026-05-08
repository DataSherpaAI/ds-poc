#!/usr/bin/env python3
from pathlib import Path
import os
import sys
import hashlib
import uuid

# Load .env
from dotenv import load_dotenv

# NOTE: This expects your .env two levels above this file (repo root if this is in repo/src/ or repo/scripts/)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

import weaviate
from weaviate import AuthApiKey

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate

# OpenAI‐based embedding
from embedding import get_embedding_function


# ─── CONFIGURATION ────────────────────────────────────────────
WEAVIATE_URL     = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
CLASS_NAME       = "DS_POC"  # Match your chatbot's index_name

# Two corpora folders (quick PoC toggle)
DATA_DIR_SMALL = Path(__file__).parent / "data" / "documents_10"
DATA_DIR_LARGE = Path(__file__).parent / "data" / "documents_100"
USE_LARGE_CORPUS = True  # flip this to switch between 10 and 100 docs
DATA_DIR = DATA_DIR_LARGE if USE_LARGE_CORPUS else DATA_DIR_SMALL

# Validate environment variables
if not WEAVIATE_URL or not WEAVIATE_API_KEY:
    raise ValueError("Missing WEAVIATE_URL or WEAVIATE_API_KEY in .env file")

print(f"🔗 Connecting to Weaviate at {WEAVIATE_URL}")

# ─── STABLE ID HELPERS (chunk identification + dedupe) ─────────
# Stable namespace for deterministic UUIDs (constant UUID; you can keep this forever)
CHUNK_UUID_NAMESPACE = uuid.UUID("6a2f2c7b-3a3f-4d8d-9b8d-2f6a0f5f2b1a")

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def make_doc_id(pdf_name: str) -> str:
    # Stable per-document id; short is fine as a grouping key
    return sha256_hex(pdf_name)[:16]

def make_text_hash(chunk_text: str) -> str:
    return sha256_hex(chunk_text)

def make_chunk_id(doc_id: str, page: int, chunk_index: int, chunk_text: str) -> str:
    # Deterministic chunk id based on doc_id + position + content hash
    text_hash16 = sha256_hex(chunk_text)[:16]
    raw = f"{doc_id}:{page}:{chunk_index}:{text_hash16}"
    return str(uuid.uuid5(CHUNK_UUID_NAMESPACE, raw))


# ─── CONNECT TO WEAVIATE (v3 client) ───────────────────────────
try:
    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=AuthApiKey(api_key=WEAVIATE_API_KEY),
        additional_headers={
            "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", "")
        }
    )

    # Test connection
    if client.is_ready():
        print("✅ Successfully connected to Weaviate")
    else:
        raise ConnectionError("Weaviate client is not ready")

except Exception as e:
    print(f"❌ Failed to connect to Weaviate: {e}")
    raise


# ─── RESET SCHEMA ─────────────────────────────────────────────
# PoC behavior: drop and recreate class each run
# Later (incremental ingest) you should remove the delete_class block.
try:
    if client.schema.exists(CLASS_NAME):
        print(f"🔄 Deleting existing class `{CLASS_NAME}`")
        client.schema.delete_class(CLASS_NAME)

    print(f"➕ Creating class `{CLASS_NAME}`")
    client.schema.create_class({
        "class": CLASS_NAME,
        "vectorizer": "none",  # We provide our own vectors
        "properties": [
            {"name": "content",     "dataType": ["text"], "description": "Chunk text"},
            {"name": "source",      "dataType": ["text"], "description": "Source PDF filename"},
            {"name": "doc_id",      "dataType": ["text"], "description": "Stable document id (hash of filename)"},
            {"name": "chunk_id",    "dataType": ["text"], "description": "Stable chunk id (uuid5)"},
            {"name": "page",        "dataType": ["int"],  "description": "Page number in source PDF"},
            {"name": "chunk_index", "dataType": ["int"],  "description": "Index of chunk within the PDF"},
            {"name": "text_hash",   "dataType": ["text"], "description": "SHA-256 of chunk text (debug/dedupe)"},
        ]
    })
    print(f"✅ Class `{CLASS_NAME}` created successfully")

except Exception as e:
    print(f"❌ Schema operation failed: {e}")
    raise


# ─── SETTING EMBEDDING FUNCTION ───────────────────────────────
print("🔧 Setting up embedding function and vector store...")
try:
    EMBED_FN = get_embedding_function()
    print("✅ Embedding function initialized")
except Exception as e:
    print(f"❌ Failed to initialize embedding function: {e}")
    raise


# ─── TEST EMBEDDING FUNCTION ──────────────────────────────────
def test_embedding_function():
    """Quick test to verify embeddings are working"""
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

if test_embedding_function():
    print("   🎉 Embedding function is working correctly\n")
else:
    print("   ⚠️  Warning: Embedding function may have issues")
    sys.exit(1)


# ─── VECTOR STORE SETUP ───────────────────────────────────────
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


# ─── INGEST PDFS ──────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Check if data directory exists
if not DATA_DIR.exists():
    print(f"❌ Data directory not found: {DATA_DIR}")
    print("   Please create it and add PDF files")
    sys.exit(1)

pdfs = sorted(DATA_DIR.glob("*.pdf"))
if not pdfs:
    print(f"⚠️  No PDF files found in {DATA_DIR}")
    sys.exit(1)

print(f"\n📂 Found {len(pdfs)} PDF(s) in {DATA_DIR}")
print("=" * 60)

total_chunks = 0
for idx, pdf in enumerate(pdfs, start=1):
    try:
        print(f"\n[{idx}/{len(pdfs)}] Processing: {pdf.name}")

        # Load PDF (page-level docs w/ metadata like {'source': ..., 'page': ...})
        docs = PyPDFLoader(str(pdf)).load()
        print(f"   📄 Loaded {len(docs)} page(s)")

        # Split into chunks
        chunks = splitter.split_documents(docs)
        print(f"   ✂️  Split into {len(chunks)} chunk(s)")

        # Stable per-document ID
        doc_id = make_doc_id(pdf.name)

        # Add metadata + deterministic object IDs
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

        # Add to vector store (ids ensure stable identification for debugging / future upserts)
        store.add_documents(chunks, ids=ids)

        total_chunks += len(chunks)
        print(f"   ✅ Added to vector store")

    except Exception as e:
        print(f"   ❌ Error processing {pdf.name}: {e}")
        continue

print("\n" + "=" * 60)
print(f"✅ Ingestion complete: {total_chunks} total chunks from {len(pdfs)} PDFs")


# ─── CHUNK QUALITY ANALYSIS [COMMENT IT OUT LATER] ─────────────
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
    print(f"  Text hash: {md.get('text_hash', 'unknown')[:16]}...")
    print(f"  Preview: {content[:150]}...")

    if not content.strip().endswith(('.', '!', '?', '\n')):
        print("  ⚠️  Warning: Chunk may be cut mid-sentence")


# ─── SIMPLE QUERY TEST ─────────────────────────────────────────
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
            print(f"   Content: {snippet}...")
            print()

except Exception as e:
    print(f"❌ Search failed: {e}")

print("=" * 60)
print("🎉 Script completed successfully!")

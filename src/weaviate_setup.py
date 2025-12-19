#!/usr/bin/env python3
from pathlib import Path
import os
import sys

# Load .env
from dotenv import load_dotenv
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
DATA_DIR         = Path(__file__).parent / "data" / "documents"

# Validate environment variables
if not WEAVIATE_URL or not WEAVIATE_API_KEY:
    raise ValueError("Missing WEAVIATE_URL or WEAVIATE_API_KEY in .env file")

print(f"🔗 Connecting to Weaviate at {WEAVIATE_URL}")

# ─── CONNECT TO WEAVIATE CLOUD (v3 client) ───────────────────
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
try:
    if client.schema.exists(CLASS_NAME):
        print(f"🔄 Deleting existing class `{CLASS_NAME}`")
        client.schema.delete_class(CLASS_NAME)
    
    print(f"➕ Creating class `{CLASS_NAME}`")
    client.schema.create_class({
        "class": CLASS_NAME,
        "vectorizer": "none",  # We provide our own vectors
        "properties": [
            {
                "name": "content",
                "dataType": ["text"],
                "description": "The text content of the document chunk"
            },
            {
                "name": "source",
                "dataType": ["text"],  # Changed from "string" to "text"
                "description": "Source PDF filename"
            }
        ]
    })
    print(f"✅ Class `{CLASS_NAME}` created successfully")
    
except Exception as e:
    print(f"❌ Schema operation failed: {e}")
    raise

# ─── SETTING EMBEDDING FUNCTION────────────────────────────
print("🔧 Setting up embedding function and vector store...")
try:
    EMBED_FN = get_embedding_function()
    print("✅ Embedding function initialized")
except Exception as e:
    print(f"❌ Failed to initialize embedding function: {e}")
    raise

# ─── TEST EMBEDDING FUNCTION ────────────────────────────────
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

# Call the test
if test_embedding_function():
    print("   🎉 Embedding function is working correctly\n")
else:
    print("   ⚠️  Warning: Embedding function may have issues")
    exit(1)

# ─── VECTOR STORE SETUP ──────────────────────────────────────
try:
    store = Weaviate(
        client=client,
        index_name=CLASS_NAME,
        text_key="content",
        attributes=["source"],
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
    print(f"   Please create it and add PDF files")
    exit(1)

pdfs = sorted(DATA_DIR.glob("*.pdf"))
if not pdfs:
    print(f"⚠️  No PDF files found in {DATA_DIR}")
    exit(1)

print(f"\n📂 Found {len(pdfs)} PDF(s) in {DATA_DIR}")
print("="*60)

total_chunks = 0
for idx, pdf in enumerate(pdfs, start=1):
    try:
        print(f"\n[{idx}/{len(pdfs)}] Processing: {pdf.name}")
        
        # Load PDF
        docs = PyPDFLoader(str(pdf)).load()
        print(f"   📄 Loaded {len(docs)} page(s)")
        
        # Split into chunks
        chunks = splitter.split_documents(docs)
        print(f"   ✂️  Split into {len(chunks)} chunk(s)")
        
        # Add metadata
        for chunk in chunks:
            chunk.metadata["source"] = pdf.name
        
        # Add to vector store
        store.add_documents(chunks)
        total_chunks += len(chunks)
        print(f"   ✅ Added to vector store")
        
    except Exception as e:
        print(f"   ❌ Error processing {pdf.name}: {e}")
        continue

print("\n" + "="*60)
print(f"✅ Ingestion complete: {total_chunks} total chunks from {len(pdfs)} PDFs")

# ─── CHUNK QUALITY ANALYSIS [COMMENT IT OUT LATER]───────────────────────────────────
print("\n📊 Chunk Quality Analysis:")
print("="*60)

sample_chunks = store.similarity_search("DES project overview", k=3)
for i, chunk in enumerate(sample_chunks, 1):
    content = chunk.page_content
    print(f"\nChunk {i}:")
    print(f"  Length: {len(content)} characters (~{len(content.split())} words)")
    print(f"  Source: {chunk.metadata.get('source', 'unknown')}")
    print(f"  Preview: {content[:150]}...")
    
    # Check if chunks are complete thoughts
    if not content.strip().endswith(('.', '!', '?', '\n')):
        print(f"  ⚠️  Warning: Chunk may be cut mid-sentence")

# ─── SIMPLE QUERY TEST ────────────────────────────────────────
print("\n🔍 Testing similarity search…")
print("="*60)

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
            src = doc.metadata.get("source", "unknown")
            print(f"{i}. Score: {score:.4f}")
            print(f"   Source: {src}")
            print(f"   Content: {snippet}...")
            print()
            
except Exception as e:
    print(f"❌ Search failed: {e}")

print("="*60)
print("🎉 Script completed successfully!")

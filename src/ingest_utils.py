#!/usr/bin/env python3
"""
Shared helpers for Weaviate ingestion scripts (PDF + Slack).

Deliberately kept separate from weaviate_setup_des.py so the existing,
working PDF ingestion pipeline is never touched. New ingestion scripts
import from here; the PDF script is left exactly as it is.
"""

import hashlib
import time
import uuid
from pathlib import Path

CHUNK_UUID_NAMESPACE = uuid.UUID("6a2f2c7b-3a3f-4d8d-9b8d-2f6a0f5f2b1a")


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def make_doc_id(name: str) -> str:
    """Stable id for a 'document' (e.g., channel + thread-root timestamp)."""
    return sha256_hex(name)[:16]


def make_chunk_id(doc_id: str, chunk_index: int, chunk_text: str) -> str:
    text_hash16 = sha256_hex(chunk_text)[:16]
    raw = f"{doc_id}:{chunk_index}:{text_hash16}"
    return str(uuid.uuid5(CHUNK_UUID_NAMESPACE, raw))


def iter_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def retry_sleep(attempt: int) -> None:
    # exponential backoff with cap (same policy as the PDF script)
    sleep_s = min(60.0, (2 ** attempt) * 1.0)
    time.sleep(sleep_s)


def load_checkpoint(path: Path) -> set:
    if path.exists():
        return set(path.read_text(encoding="utf-8").splitlines())
    return set()


def save_checkpoint(path: Path, key: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(key + "\n")

#!/usr/bin/env python3
"""
Parses one Slack channel export file (a flat JSON list of message objects,
filename = channel name) into clean, retrieval-ready LangChain Documents.

Design (see conversation/analysis for rationale):
- Filter noise STRUCTURALLY (subtype, bot_id), not by guessing from text.
- Reconstruct threads (root + replies_full) into a single labeled transcript
  BEFORE filtering for information density. This avoids dropping a real
  Q&A exchange just because it contains a lone "thanks!".
- Apply the word-count filter to the whole reconstructed unit, not to
  individual messages.
- Track thread completeness explicitly (`unit_type`), since some exports
  report reply_count > 0 but never captured the replies themselves.
- Track reactions as a light-weight "was this helpful" signal.

This module has no network/Weaviate dependency; it can be tested standalone
against a sample export file.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest_utils import make_doc_id, make_chunk_id, sha256_hex

# ─── CONFIG ─────────────────────────────────────────────────────────────────
NOISE_SUBTYPES = {"channel_join", "channel_leave", "channel_topic", "channel_purpose"}

# Reactions that signal "this message was a helpful/confirmed answer"
POSITIVE_REACTIONS = {
    "thankyou", "thanks", "+1", "thumbsup", "raised_hands", "pray",
    "clap", "100", "heavy_check_mark", "white_check_mark", "tada", "fire",
}

MIN_WORDS_DEFAULT = 6  # below this, a unit is dropped as low-value chatter

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


# ─── TEXT CLEANING ──────────────────────────────────────────────────────────
def clean_slack_text(text: str, user_map: dict) -> str:
    """Resolve Slack markup (<@U..>, <#C..>, links) into plain readable text."""
    if not text:
        return ""

    # Labeled links: <https://url|label> -> "label (https://url)"
    text = re.sub(
        r"<(https?://[^|>]+)\|([^>]+)>",
        lambda m: f"{m.group(2)} ({m.group(1)})",
        text,
    )
    # Bare links: <https://url> -> "https://url"
    text = re.sub(r"<(https?://[^>]+)>", r"\1", text)

    # Channel mentions: <#C0123|> or <#C0123|general> -> "#general" or "#channel"
    text = re.sub(r"<#([A-Z0-9]+)\|([^>]*)>", lambda m: f"#{m.group(2)}" if m.group(2) else "#channel", text)

    # Special mentions: <!here>, <!channel>, <!everyone>
    text = re.sub(r"<!(here|channel|everyone)>", r"@\1", text)

    # User mentions: <@U0123> -> "@DisplayName" (fallback to id if unknown)
    text = re.sub(
        r"<@([A-Z0-9]+)>",
        lambda m: f"@{user_map.get(m.group(1), m.group(1))}",
        text,
    )

    return text.strip()


# ─── LOADING ────────────────────────────────────────────────────────────────
def load_channel_messages(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_user_map(messages: list) -> dict:
    """Map user_id -> display name, scanning both top-level msgs and replies."""
    user_map = {}

    def _record(m):
        uid = m.get("user")
        disp = m.get("user_display")
        if uid and disp:
            user_map[uid] = disp

    for m in messages:
        _record(m)
        for r in m.get("replies_full", []):
            _record(r)

    return user_map


# ─── NOISE FILTERING (structural, not text-based) ──────────────────────────
def is_noise_message(msg: dict) -> bool:
    if msg.get("subtype") in NOISE_SUBTYPES:
        return True
    if msg.get("bot_id"):
        return True
    return False


def summarize_reactions(msg: dict) -> tuple:
    reactions = msg.get("reactions", []) or []
    total = sum(r.get("count", 0) for r in reactions)
    has_positive = any(r.get("name") in POSITIVE_REACTIONS for r in reactions)
    return total, has_positive


def ts_to_iso(ts: str) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


# ─── THREAD/UNIT RECONSTRUCTION ────────────────────────────────────────────
def _render_line(msg: dict, user_map: dict) -> str:
    speaker = msg.get("user_display") or user_map.get(msg.get("user", ""), "unknown")
    when = ts_to_iso(msg["ts"])
    text = clean_slack_text(msg.get("text", ""), user_map)
    return f"[{when}] {speaker}: {text}"


def reconstruct_units(messages: list, user_map: dict) -> list:
    """
    Turn a flat list of Slack messages into "units": either a single
    standalone message, or a reconstructed thread (root + known replies).

    Returns a list of dicts: {
        text, participants, reaction_count, has_thanks_reaction,
        unit_type, message_count, ts_start, ts_end, root_ts,
        reply_count_reported, reply_count_captured
    }
    """
    units = []

    for msg in messages:
        is_root = msg.get("thread_ts") == msg.get("ts")
        is_reply_only_toplevel = msg.get("thread_ts") and not is_root

        if is_reply_only_toplevel:
            # A reply that happens to also appear standalone at the top
            # level (not seen in our sample data, but handled defensively
            # so nothing silently disappears).
            sub_messages = [msg]
            unit_type = "thread_orphan_reply"
            reply_count_reported = 0
            reply_count_captured = 0
        else:
            replies_full = msg.get("replies_full", []) or []
            if msg.get("reply_count", 0) > 0 and replies_full:
                # Skip index 0: replies_full's first entry duplicates the root.
                sub_messages = [msg] + replies_full[1:]
                reply_count_reported = msg.get("reply_count", 0)
                reply_count_captured = len(sub_messages) - 1
                unit_type = (
                    "thread_complete"
                    if reply_count_captured >= reply_count_reported
                    else "thread_partial"
                )
            elif msg.get("reply_count", 0) > 0:
                # Reported replies exist but were never captured in this export.
                sub_messages = [msg]
                reply_count_reported = msg.get("reply_count", 0)
                reply_count_captured = 0
                unit_type = "thread_partial"
            else:
                sub_messages = [msg]
                reply_count_reported = 0
                reply_count_captured = 0
                unit_type = "single"

        # Drop structurally-noisy lines (system/bot) from within the unit
        sub_messages = [m for m in sub_messages if not is_noise_message(m)]
        if not sub_messages:
            continue

        sub_messages.sort(key=lambda m: float(m["ts"]))

        lines = [_render_line(m, user_map) for m in sub_messages]
        raw_word_count = sum(len(m.get("text", "").split()) for m in sub_messages)

        participants = sorted({
            m.get("user_display") or user_map.get(m.get("user", ""), "unknown")
            for m in sub_messages
        })

        reaction_count = 0
        has_thanks = False
        for m in sub_messages:
            cnt, pos = summarize_reactions(m)
            reaction_count += cnt
            has_thanks = has_thanks or pos

        units.append({
            "text": "\n".join(lines),
            "raw_word_count": raw_word_count,
            "participants": participants,
            "reaction_count": reaction_count,
            "has_thanks_reaction": has_thanks,
            "unit_type": unit_type,
            "message_count": len(sub_messages),
            "ts_start": ts_to_iso(sub_messages[0]["ts"]),
            "ts_end": ts_to_iso(sub_messages[-1]["ts"]),
            "root_ts": msg["ts"],
            "reply_count_reported": reply_count_reported,
            "reply_count_captured": reply_count_captured,
        })

    return units


def filter_low_value(units: list, min_words: int = MIN_WORDS_DEFAULT) -> list:
    return [u for u in units if u["raw_word_count"] >= min_words]


# ─── CHUNKING INTO DOCUMENTS ────────────────────────────────────────────────
def units_to_documents(units: list, channel_name: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents = []
    for unit in units:
        doc_id = make_doc_id(f"{channel_name}:{unit['root_ts']}")
        pieces = splitter.split_text(unit["text"]) or [unit["text"]]

        for chunk_index, piece in enumerate(pieces):
            chunk_id = make_chunk_id(doc_id, chunk_index, piece)
            metadata = {
                "channel": channel_name,
                "source": f"slack:{channel_name}",
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "text_hash": sha256_hex(piece),
                "ts_start": unit["ts_start"],
                "ts_end": unit["ts_end"],
                "message_count": unit["message_count"],
                "participants": ", ".join(unit["participants"]),
                "unit_type": unit["unit_type"],
                "reaction_count": unit["reaction_count"],
                "has_thanks_reaction": unit["has_thanks_reaction"],
            }
            documents.append(Document(page_content=piece, metadata=metadata))

    return documents


# ─── TOP-LEVEL ENTRY POINT ──────────────────────────────────────────────────
def parse_channel_file(path: Path, min_words: int = MIN_WORDS_DEFAULT) -> list:
    """Parse one channel export file into chunked, filtered Documents."""
    channel_name = path.stem
    messages = load_channel_messages(path)
    user_map = build_user_map(messages)

    units = reconstruct_units(messages, user_map)
    units = filter_low_value(units, min_words=min_words)

    return units_to_documents(units, channel_name)

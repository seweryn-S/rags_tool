"""Shared store access helpers decoupled from retrieval logic.

This module exposes reusable functions for reading document metadata and
section/chunk contents from Qdrant. It is intentionally free of retrieval
scoring, MMR, or shaping concerns, so other components can import it without
pulling in search internals.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from qdrant_client.http import models as qm

from app.core.chunking import SECTION_HIERARCHY
from app.qdrant_utils import qdrant
from app.settings import get_settings

settings = get_settings()

SECTION_LEVEL_INDEX: Dict[str, int] = {level: idx for idx, level in enumerate(SECTION_HIERARCHY)}


def normalize_whitespace(text: Optional[str]) -> str:
    """Trim and collapse internal whitespace; return safe string."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().split())


def canonical_section_label(section_path: Optional[str], merge_level: str) -> Optional[str]:
    """Build a canonical section label up to the requested hierarchy level.

    Falls back to the full normalized path when the desired level cannot be
    detected in the provided `section_path` string.
    """
    if merge_level not in SECTION_LEVEL_INDEX:
        merge_level = "ust"

    if isinstance(section_path, str) and section_path.strip():
        path_segments = [seg.strip() for seg in section_path.split(">") if seg.strip()]
        if not path_segments:
            return None

        def is_level(seg: str, lvl: str) -> bool:
            s = seg.strip().lower()
            if lvl == "par":
                return s.startswith("§ ")
            if lvl == "ust":
                return s.startswith("ust.")
            if lvl == "pkt":
                return s.startswith("pkt ")
            if lvl == "lit":
                return s.startswith("lit.")
            if lvl == "chapter":
                return s.startswith("rozdział ")
            if lvl == "attachment":
                return s.startswith("załącznik")
            if lvl == "regulamin":
                return s == "regulamin"
            return False

        collected: List[str] = []
        cut_done = False
        for seg in path_segments:
            collected.append(normalize_whitespace(seg))
            if is_level(seg, merge_level):
                cut_done = True
                break
        if not cut_done:
            collected = [normalize_whitespace(seg) for seg in path_segments]
        joined = normalize_whitespace(" > ".join(collected))
        return joined or None

    if isinstance(section_path, str) and section_path.strip():
        normalized = normalize_whitespace(section_path)
        if normalized:
            return normalized
    return None


def fetch_doc_summaries(doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch summaries for given doc_ids from the summaries collection.

    Returns a map: doc_id -> { doc_id, doc_summary, doc_signature, doc_entities, doc_title, doc_date, is_active }.
    Best-effort: returns partial data if scroll fails midway.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not doc_ids:
        return out
    try:
        flt = qm.Filter(
            must=[
                qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids)),
                qm.FieldCondition(key="point_type", match=qm.MatchValue(value="summary")),
            ]
        )
        offset = None
        while True:
            res = qdrant.scroll(
                collection_name=settings.qdrant_summary_collection,
                scroll_filter=flt,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if isinstance(res, tuple):
                records, offset = res
            else:
                records = getattr(res, "points", None)
                offset = getattr(res, "next_page_offset", None)
                if records is None:
                    records = []
            if not records:
                break
            for rec in records:
                payload = rec.payload or {}
                did = payload.get("doc_id")
                if not did:
                    continue
                out[str(did)] = {
                    "doc_id": str(did),
                    "doc_summary": payload.get("summary"),
                    "doc_signature": payload.get("signature"),
                    "doc_entities": payload.get("entities"),
                    "doc_title": payload.get("title"),
                    "doc_date": payload.get("doc_date"),
                    "is_active": payload.get("is_active"),
                    "doc_url": payload.get("doc_url"),
                }
            if offset is None:
                break
    except Exception:
        return out
    return out


def fetch_sections_chunks_batch(doc_id: str, sections: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch chunks for many sections of a single document with one scroll.

    Returns a map: normalized_section -> list of chunk payloads (sorted). Sections
    include descendants via prefix matching.
    """
    result: Dict[str, List[Dict[str, Any]]] = {}
    if not doc_id or not sections:
        return result
    labels = [s for s in {normalize_whitespace(s or "") for s in sections} if s]
    if not labels:
        return result
    must = [
        qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
        qm.FieldCondition(key="point_type", match=qm.MatchValue(value="chunk")),
        qm.FieldCondition(key="section_path_prefixes", match=qm.MatchAny(any=labels)),
    ]
    flt = qm.Filter(must=must)
    offset = None
    try:
        while True:
            res = qdrant.scroll(
                collection_name=settings.qdrant_content_collection,
                scroll_filter=flt,
                limit=256,
                offset=offset,
                with_payload=["text", "chunk_id", "section_path_prefixes"],
                with_vectors=False,
            )
            if isinstance(res, tuple):
                records, offset = res
            else:
                records = getattr(res, "points", None)
                offset = getattr(res, "next_page_offset", None)
                if records is None:
                    records = []
            if not records:
                break
            for rec in records:
                payload = rec.payload or {}
                prefixes = payload.get("section_path_prefixes") or []
                if not isinstance(prefixes, list):
                    continue
                for lab in labels:
                    if lab in prefixes:
                        result.setdefault(lab, []).append(payload)
            if offset is None:
                break
    except Exception:
        return result
    for lab, lst in result.items():
        lst.sort(key=lambda p: int(p.get("chunk_id", 0)))
    return result


def truncate_head_tail(text: str, limit: int) -> str:
    """Truncate to `limit` chars keeping 70% head and 30% tail."""
    t = (text or "").strip()
    if len(t) <= max(1, int(limit)):
        return t
    head = int(limit * 0.7)
    tail = max(0, int(limit) - head)
    return (t[:head] + "\n...\n" + t[-tail:]).strip()

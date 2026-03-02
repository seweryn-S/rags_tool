"""FastAPI endpoints for rags_tool."""

from __future__ import annotations

import base64
import csv
import inspect
import io
import json
import logging
import pathlib
import re
import sys
import tempfile
import time
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import ValidationError

from app.core.chunking import chunk_text_by_sections
from qdrant_client.http import models as qm
from app.core.embedding import IterableCorpus, prepare_tfidf
from app.core.embedding import embed_query
from app.core.parsing import SUPPORTED_EXT, extract_text
from app.core.search import (
    _build_sparse_queries_for_query,
    _classify_mode,
    _stage1_select_documents,
    _stage2_select_chunks,
    _shape_results,
    _truncate_head_tail,
    _fetch_doc_chunks,
    _canonical_section_label,
    _fetch_doc_summaries,
)
from app.core.summary import llm_summary
from app.core.ranker_client import OpenAIReranker
from app.models import (
    About,
    CollectionsExportRequest,
    CollectionsImportRequest,
    IngestBuildRequest,
    InitCollectionsRequest,
    ScanRequest,
    ScanResponse,
    SearchQuery,
    SearchResponse,
    SummariesGenerateRequest,
    BrowseQuery,
    BrowseIdsResponse,
    DocIdsQuery,
    DocStatsResponse,
    QuotesFindRequest,
    QuotesFindResponse,
    QuoteItem,
)
from app.qdrant_utils import (
    build_and_upsert_points,
    derive_collection_names,
    ensure_collections,
    export_collections_bundle,
    import_collections_bundle,
    qdrant,
    sha1,
    find_path_by_content_sha256,
    swap_collection_aliases,
)
from app.settings import get_settings
from app.admin_routes import attach_admin_routes
from app.golden_routes import attach_golden_routes
from app.core.summary_cache import (
    compute_file_sha256,
    load_sidecar,
    save_sidecar,
    sidecar_path_for,
)
from app.core.embedding import embed_passage
from app.core import browse_service
from app.core.fts import fts_doc_counts, rebuild_fts_from_qdrant, fts_count


settings = get_settings()

logger = logging.getLogger("rags_tool")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)
logger.propagate = False


_collection_init_lock = Lock()
_initialized_collections: Set[str] = set()

# Przygotuj czytelny JSON body dla operacji ingest-build bez ryzykownego łączenia stringów
INGEST_BUILD_BODY = json.dumps(
    {
        "base_dir": "/app/data",
        "glob": "**/*",
        "recursive": True,
        "reindex": False,
        "chunk_tokens": settings.chunk_tokens,
        "chunk_overlap": settings.chunk_overlap,
        "collection_name": "rags_tool",
        "enable_sparse": True,
        "rebuild_tfidf": True,
        "force_regen_summary": False,
    },
    ensure_ascii=False,
    indent=2,
)


# Build a stable cache key for (summary, content) collections derived from base.
def _collection_cache_key(base: Optional[str]) -> str:
    """Stable cache key for a pair of (summary, content) collections."""
    summary_collection, content_collection = derive_collection_names(base)
    return f"{summary_collection}::{content_collection}"


# Mark collections for the given base as initialized in this process.
def _mark_collections_initialized(base: Optional[str]) -> None:
    """Remember that collections for `base` were ensured in this process."""
    key = _collection_cache_key(base)
    with _collection_init_lock:
        _initialized_collections.add(key)


# Clear the initialized-collections flag for the given base.
def _clear_collection_cache(base: Optional[str]) -> None:
    """Forget initialization status for collections derived from `base`."""
    key = _collection_cache_key(base)
    with _collection_init_lock:
        _initialized_collections.discard(key)


# Ensure collections exist once per process (guards repeated init).
def _ensure_collections_cached(base: Optional[str] = None) -> None:
    """Ensure Qdrant collections exist once per process using a local cache."""
    key = _collection_cache_key(base)
    with _collection_init_lock:
        if key in _initialized_collections:
            return
        ensure_collections(base)
        _initialized_collections.add(key)


# Qdrant preflight check with explicit logging instead of bubbling exceptions.
def _qdrant_available_or_log(context: str) -> bool:
    """Return True if Qdrant responds; otherwise log a clear error and return False.

    Intended to protect tool endpoints from returning 500 when Qdrant is down by
    allowing a graceful empty response. Affects only the endpoints that use it.
    """
    try:
        qdrant.get_collections()
        return True
    except Exception as exc:
        logger.error(
            "Qdrant unavailable | context=%s url=%s error=%s",
            context,
            getattr(settings, "qdrant_url", ""),
            exc,
        )
        return False


# --- Quotes/find helpers ---

def _decode_cursor(cur: Optional[str]) -> Optional[Dict[str, int]]:
    if not cur:
        return None
    try:
        raw = base64.urlsafe_b64decode(cur.encode("utf-8")).decode("utf-8")
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return None
        got = {k: int(obj.get(k, 0) or 0) for k in ("doc_index", "chunk_index", "occ_index")}
        return got
    except Exception:
        return None


def _encode_cursor(state: Dict[str, int]) -> str:
    payload = json.dumps(state, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("utf-8")


def _tokenize(s: str) -> List[str]:
    return re.findall(r"\w+", s or "")


def _find_all(text: str, needle: str, *, case_sensitive: bool) -> List[Tuple[int, int]]:
    if not needle:
        return []
    if case_sensitive:
        t = text
        n = needle
    else:
        t = text.casefold()
        n = needle.casefold()
    out: List[Tuple[int, int]] = []
    i = 0
    L = len(n)
    if L == 0:
        return []
    while True:
        i = t.find(n, i)
        if i == -1:
            break
        out.append((i, i + L))
        i = i + L
    return out


def _merge_and_sort_occurrences(items: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sort occurrences by start, then end; keep duplicates as separate entries."""
    return sorted(items, key=lambda p: (int(p[0]), int(p[1])))


# Scan filesystem for supported files matching pattern.
def _scan_files(base: pathlib.Path, pattern: str, recursive: bool) -> List[pathlib.Path]:
    """Scan `base` for files matching pattern and supported extensions."""
    iterator = base.rglob(pattern) if recursive else base.glob(pattern)
    return [p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]


def _normalize_filename_key(name: str) -> str:
    """Placeholder for future filename key normalization used in link mapping."""
    return name


def _load_doc_url_map(base_dir: pathlib.Path) -> Dict[str, str]:
    """Load mapping: normalized filename stem -> doc_url from CSV in corpus root.

    CSV is expected to reside in the corpus base directory under the fixed name
    'wikamp_normative_acts_map_doc.csv' and contain at least columns:
    - filename
    - posturl

    Selection rules per stem:
    - Prefer entries where filename ends with .doc or .docx (take first).
    - If none, but other entries exist:
      - Prefer .pdf (take first),
      - Otherwise take the first available entry.
    """
    mapping: Dict[str, str] = {}
    csv_path = base_dir / "wikamp_normative_acts_map_doc.csv"
    if not csv_path.exists():
        logger.info("Doc URL map CSV not found in corpus root | path=%s", csv_path)
        return mapping
    try:
        rows_per_key: Dict[str, List[Dict[str, str]]] = {}
        with csv_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=";")
            for row in reader:
                filename = (row.get("filename") or "").strip()
                posturl = (row.get("posturl") or "").strip()
                if not filename or not posturl:
                    continue
                name_obj = pathlib.Path(filename)
                stem = _normalize_filename_key(name_obj.stem)
                if not stem:
                    continue
                ext = name_obj.suffix.lower()
                rows_per_key.setdefault(stem, []).append(
                    {
                        "ext": ext,
                        "posturl": posturl,
                    }
                )
        total_rows = sum(len(v) for v in rows_per_key.values())
        for stem, rows in rows_per_key.items():
            doc_like = [r for r in rows if r["ext"] in {".doc", ".docx"}]
            chosen: Optional[Dict[str, str]]
            if doc_like:
                chosen = doc_like[0]
            else:
                pdf_like = [r for r in rows if r["ext"] == ".pdf"]
                if pdf_like:
                    chosen = pdf_like[0]
                else:
                    chosen = rows[0] if rows else None
            if chosen:
                mapping[stem] = chosen["posturl"]
        logger.info(
            "Doc URL map loaded | csv_path=%s rows=%d keys=%d",
            csv_path,
            int(total_rows),
            int(len(mapping)),
        )
    except Exception as exc:
        logger.warning("Doc URL map load failed | path=%s error=%s", csv_path, exc)
        return {}
    return mapping


# Iterate ingest records: chunks + LLM summary (+ cached vectors) per file.
def _iter_document_records(
    file_paths: List[pathlib.Path],
    chunk_tokens: int,
    chunk_overlap: int,
    *,
    force_regen_summary: bool,
    collection_base: Optional[str],
    dedupe_on_ingest: bool,
    doc_url_map: Optional[Dict[str, str]] = None,
    stats: Optional[Dict[str, int]] = None,
) -> Iterable[Dict[str, Any]]:
    """Yield ingest records for each document (chunks + summary + vectors).

    Uses sidecar cache unless `force_regen_summary` is set.
    """
    seen_by_hash: Dict[str, str] = {}
    for path in file_paths:
        doc_start = time.time()
        logger.debug("Processing document %s", path)
        # Try to attach external source URL based on filename stem
        doc_url: Optional[str] = None
        try:
            stem = _normalize_filename_key(path.stem)
            if doc_url_map is not None and stem in doc_url_map:
                doc_url = doc_url_map[stem]
                if stats is not None:
                    stats["doc_url_matched"] = int(stats.get("doc_url_matched", 0)) + 1
            else:
                if stats is not None:
                    stats["doc_url_missing"] = int(stats.get("doc_url_missing", 0)) + 1
        except Exception:
            if stats is not None:
                stats["doc_url_missing"] = int(stats.get("doc_url_missing", 0)) + 1
        # Compute content hash up front to enable early dedupe
        content_sha256 = compute_file_sha256(path)
        if dedupe_on_ingest:
            # in-run dedupe
            existing_local = seen_by_hash.get(content_sha256)
            if existing_local and str(path.resolve()) != existing_local:
                logger.info(
                    "Duplicate skipped | sha256=%s existing=%s duplicate=%s",
                    content_sha256,
                    existing_local,
                    str(path.resolve()),
                )
                if stats is not None:
                    stats["duplicates_skipped"] = int(stats.get("duplicates_skipped", 0)) + 1
                continue
            # cross-run dedupe via Qdrant (only for first occurrence in this run)
            existing_remote = find_path_by_content_sha256(content_sha256, collection_base)
            if existing_remote and str(path.resolve()) != str(existing_remote):
                logger.info(
                    "Duplicate skipped | sha256=%s existing=%s duplicate=%s",
                    content_sha256,
                    str(existing_remote),
                    str(path.resolve()),
                )
                if stats is not None:
                    stats["duplicates_skipped"] = int(stats.get("duplicates_skipped", 0)) + 1
                # Do not record in seen_by_hash to allow logging more duplicates against the same existing_remote
                continue

        # Always extract text and chunks (not cached here)
        raw = extract_text(path)
        chunk_items = chunk_text_by_sections(
            raw,
            target_tokens=chunk_tokens,
            overlap_tokens=chunk_overlap,
        )
        chunks = chunk_items
        if not chunks:
            logger.debug("Document %s produced no chunks; skipping", path)
            continue
        # Try sidecar cache for summary + vectors (unless forced to regenerate)
        sidecar = None
        sc_path = sidecar_path_for(path)
        sc_name = sc_path.name
        if force_regen_summary:
            if sc_path.exists():
                logger.debug("Force regen: ignoring sidecar | path=%s sidecar=%s", path, sc_name)
        else:
            if sc_path.exists():
                logger.debug("Sidecar present, validating | path=%s sidecar=%s", path, sc_name)
            else:
                logger.debug("Sidecar not found | path=%s expected=%s", path, sc_name)
            sidecar = load_sidecar(path, expected_sha256=content_sha256)
        if sidecar:
            logger.debug("Using sidecar cache | path=%s sidecar=%s", path, sc_name)
            summ_block = sidecar.get("summary", {})
            vectors_block = sidecar.get("vectors", {})
            doc_sum = {
                "title": str(summ_block.get("title") or ""),
                "subtitle": str(summ_block.get("subtitle") or "brak") or "brak",
                "summary": str(summ_block.get("summary") or ""),
                "signature": list(summ_block.get("signature") or []),
                "entities": [str(x).strip() for x in list(summ_block.get("entities") or []) if str(x).strip()],
                "replacement": str(summ_block.get("replacement") or "brak") or "brak",
                "doc_date": str(summ_block.get("doc_date") or "brak") or "brak",
                "is_active": bool(summ_block.get("is_active", True)),
            }
            summary_dense_vec = list(vectors_block.get("summary_dense") or [])
        else:
            # Generate with LLM and compute dense embedding for summary; then cache
            doc_sum = llm_summary(raw, path=str(path.resolve()))
            summary_text = doc_sum.get("summary", "") or ""
            summary_dense_vec = embed_passage([summary_text])[0]
            # Persist sidecar (atomic write); best-effort, ignore failures
            try:
                save_sidecar(
                    path,
                    content_sha256=content_sha256,
                    title=str(doc_sum.get("title", "") or ""),
                    subtitle=str(doc_sum.get("subtitle", "brak") or "brak"),
                    summary=summary_text,
                    signature=list(doc_sum.get("signature", []) or []),
                    entities=[str(x).strip() for x in list(doc_sum.get("entities", []) or []) if str(x).strip()],
                    replacement=str(doc_sum.get("replacement", "brak") or "brak"),
                    summary_dense=list(summary_dense_vec),
                    doc_date=str(doc_sum.get("doc_date", "brak") or "brak"),
                    is_active=bool(doc_sum.get("is_active", True)),
                )
                logger.debug("Sidecar saved | path=%s sidecar=%s", path, sidecar_path_for(path).name)
            except Exception as exc:
                logger.debug("Sidecar save skipped | path=%s error=%s", path, exc)
            if not force_regen_summary and sc_path.exists():
                logger.debug("Sidecar rejected or stale; regenerated | path=%s sidecar=%s", path, sc_name)
        # Mark as seen for subsequent in-run duplicates
        if dedupe_on_ingest:
            seen_by_hash[content_sha256] = str(path.resolve())
        doc_id = sha1(str(path.resolve()))
        doc_title = str(doc_sum.get("title", "") or "").strip() or path.stem
        summary_signature = doc_sum.get("signature", [])
        replacement_info = doc_sum.get("replacement", "brak") or "brak"
        if isinstance(replacement_info, str) and replacement_info.lower() == "brak":
            replacement_info = "brak"
        summary_sparse_parts = [
            doc_title,
            str(doc_sum.get("subtitle", "")),
            doc_sum.get("summary", ""),
            " ".join(summary_signature),
        ]
        # Include entities (as tokens) to improve Stage-1 sparse matching
        entities_list = [str(x).strip() for x in list(doc_sum.get("entities", []) or []) if str(x).strip()]
        if entities_list:
            summary_sparse_parts.append(" ".join(entities_list))
        # Include doc_date in sparse summary if available and not 'brak'
        doc_date_val = str(doc_sum.get("doc_date", "") or "").strip()
        if doc_date_val and doc_date_val.lower() != "brak":
            summary_sparse_parts.append(doc_date_val)
        if replacement_info.lower() != "brak":
            summary_sparse_parts.append(replacement_info)
        summary_sparse_text = " ".join(part for part in summary_sparse_parts if part).strip()
        rec = {
            "doc_id": doc_id,
            "path": str(path.resolve()),
            "doc_url": doc_url,
            "chunks": chunks,
            "doc_title": doc_title,
            "subtitle": str(doc_sum.get("subtitle", "brak") or "brak"),
            "doc_summary": doc_sum.get("summary", ""),
            "doc_signature": summary_signature,
            "doc_entities": entities_list,
            "replacement": replacement_info,
            "doc_date": doc_sum.get("doc_date", "brak"),
            "summary_sparse_text": summary_sparse_text,
            # Precomputed dense summary vector (used to skip embedding in upsert stage)
            "summary_dense_vec": list(summary_dense_vec) if (sidecar or summary_dense_vec is not None) else None,
            # Pass content hash for downstream payload persistence
            "content_sha256": content_sha256,
            # Initial active flag based on LLM (default True when absent)
            "is_active": bool(doc_sum.get("is_active", True)),
        }
        logger.debug(
            "Document %s parsed | chunks=%d summary_len=%d took_ms=%d",
            path,
            len(chunks),
            len(rec["doc_summary"] or ""),
            int((time.time() - doc_start) * 1000),
        )
        yield rec


# Iterate JSONL records saved during ingest.
def _iter_saved_records(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    """Iterate JSONL records previously written by ingest step."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# Iterate chunk texts from saved ingest JSONL.
def _iter_chunk_texts(path: pathlib.Path) -> Iterable[str]:
    """Iterate chunk texts from stored ingest records."""
    for rec in _iter_saved_records(path):
        for chunk in rec.get("chunks", []):
            if isinstance(chunk, dict):
                text = chunk.get("text", "")
            else:
                text = str(chunk)
            if text:
                yield text


# Iterate summary texts from saved ingest JSONL (for TF‑IDF fitting).
def _iter_summary_texts(path: pathlib.Path) -> Iterable[str]:
    """Iterate summary texts (for TF-IDF fitting) from stored records."""
    for rec in _iter_saved_records(path):
        summary = rec.get("summary_sparse_text")
        if summary:
            yield summary


ADMIN_OPERATION_SPECS: List[Dict[str, Any]] = [
    {"id": "search-debug-embed", "path": "/search/debug/embed", "method": "POST", "label": "Search Debug: 1) embed", "body": "{\"query\":\"Jak działa rags_tool?\",\"mode\":\"auto\",\"use_hybrid\":true}"},
    {"id": "search-debug-stage1", "path": "/search/debug/stage1", "method": "POST", "label": "Search Debug: 2) stage1", "body": "{\"q_text\":\"Jak działa rags_tool?\",\"q_vec\":[0.0],\"mode\":\"auto\",\"use_hybrid\":true,\"top_m\":100,\"score_norm\":\"minmax\",\"dense_weight\":0.6,\"sparse_weight\":0.4,\"mmr_stage1\":true,\"mmr_lambda\":0.3}"},
    {"id": "search-debug-stage2", "path": "/search/debug/stage2", "method": "POST", "label": "Search Debug: 3) stage2", "body": "{\"q_text\":\"Jak działa rags_tool?\",\"q_vec\":[0.0],\"cand_doc_ids\":[\"<doc_id>\"],\"doc_map\":{},\"top_k\":10,\"per_doc_limit\":2,\"score_norm\":\"minmax\",\"dense_weight\":0.6,\"sparse_weight\":0.4,\"mmr_lambda\":0.3}"},
    {"id": "search-debug-shape", "path": "/search/debug/shape", "method": "POST", "label": "Search Debug: 4) shape", "body": "{\"final_hits\":[{\"doc_id\":\"<doc_id>\",\"path\":\"/abs/path\",\"section\":null,\"chunk_id\":0,\"score\":0.5,\"snippet\":\"...\"}],\"result_format\":\"blocks\",\"summary_mode\":\"first\"}"},
    {"id": "about", "path": "/about", "method": "GET"},
    {"id": "health", "path": "/health", "method": "GET"},
    {
        "id": "collections-init",
        "path": "/collections/init",
        "method": "POST",
        "body": "{\n  \"collection_name\": \"rags_tool\",\n  \"force_dim_probe\": false\n}",
    },
    {
        "id": "collections-export",
        "path": "/collections/export",
        "method": "POST",
        "label": "Eksport kolekcji (plik .tar.gz)",
        "body": "{}",
    },
    {
        "id": "collections-import",
        "path": "/collections/import",
        "method": "POST",
        "label": "Import kolekcji z archiwum",
        "body": "{\n  \"archive_base64\": \"<wklej_archiwum_base64>\",\n  \"replace_existing\": true\n}",
        "accepts_file": True,
    },
    {
        "id": "ingest-scan",
        "path": "/ingest/scan",
        "method": "POST",
        "body": "{\n  \"base_dir\": \"/app/data\",\n  \"glob\": \"**/*\",\n  \"recursive\": true\n}",
    },
    {
        "id": "summaries-generate",
        "path": "/summaries/generate",
        "method": "POST",
        "body": "{\n  \"files\": [\n    \"/app/data/example.md\"\n  ]\n}",
    },
    {
        "id": "ingest-build",
        "path": "/ingest/build",
        "method": "POST",
        "body": INGEST_BUILD_BODY,
    },
    {
        "id": "search-query",
        "path": "/search/query",
        "method": "POST",
        "body": "{\n  \"query\": [\n    \"Jak działa rags_tool?\",\n    \"architektura rags_tool\"\n  ],\n  \"top_m\": 10,\n  \"top_k\": 5,\n  \"mode\": \"auto\",\n  \"use_hybrid\": true,\n  \"dense_weight\": 0.6,\n  \"sparse_weight\": 0.4,\n  \"mmr_lambda\": 0.3,\n  \"per_doc_limit\": 2,\n  \"score_norm\": \"minmax\",\n  \"rep_alpha\": 0.6,\n  \"mmr_stage1\": true,\n  \"summary_mode\": \"first\",\n  \"result_format\": \"blocks\"\n}",
    },
    {
        "id": "search-query-restricted",
        "path": "/search/query",
        "method": "POST",
        "body": "{\n  \"query\": [\n    \"cytaty dla skrótu\"\n  ],\n  \"top_m\": 1200,\n  \"top_k\": 100,\n  \"mode\": \"auto\",\n  \"use_hybrid\": true,\n  \"per_doc_limit\": 50,\n  \"result_format\": \"blocks\",\n  \"restrict_doc_ids\": [\n    \"<doc_id_1>\",\n    \"<doc_id_2>\"\n  ]\n}",
    },
]

app = FastAPI(title=f"{settings.app_name} OpenAPI Tool", version=settings.app_version)

# Attach Admin UI and debug endpoints from isolated module
attach_admin_routes(app)
attach_golden_routes(app)


# Ensure Qdrant collections and indexes at process startup
@app.on_event("startup")
def _startup_ensure_collections() -> None:
    """Best-effort ensure of collections and TF-IDF warm-up at startup."""
    try:
        _ensure_collections_cached()
        # Pre-warm TF-IDF vectorizers (content + summaries) to avoid cold-start latency
        try:
            # Load existing TF-IDF vectorizers if present (no fitting at startup)
            prepare_tfidf(
                all_chunks=None,
                summary_corpus=None,
                enable_sparse=True,
                rebuild_tfidf=False,
            )
            logger.info("TF-IDF vectorizers pre-warmed at startup")
        except Exception as exc:
            logger.warning("TF-IDF pre-warm skipped: %s", exc)
        # FTS index at startup: rebuild only if missing/empty; otherwise keep as-is
        try:
            cnt_before = int(fts_count())
            if cnt_before <= 0:
                inserted = int(rebuild_fts_from_qdrant())
                logger.info("FTS index rebuilt at startup | rows=%d", inserted)
            else:
                logger.info("FTS index present at startup | rows=%d", cnt_before)
        except Exception as exc:
            logger.warning("FTS init at startup failed: %s", exc)
        # Log key runtime switches for clarity
        logger.info(
            "Startup config | skip_stage1=%s dual_query_sparse=%s minimal_payload=%s batch_section_fetch=%s rrf_k=%d oversample=%d dense_for_mmr=%s",
            bool(settings.search_skip_stage1_default),
            bool(settings.search_dual_query_sparse),
            bool(settings.search_minimal_payload),
            bool(settings.batch_section_fetch),
            int(settings.dual_query_rrf_k),
            int(settings.dual_query_oversample),
            bool(settings.dual_query_dense_for_mmr),
        )
        logger.info("Collections ensured at startup")
    except Exception as exc:
        # Do not block startup on failures; health endpoint will reflect real status
        logger.warning("Startup ensure_collections failed: %s", exc)


# Admin UI and step-by-step debug endpoints are defined in app/admin_routes.py


@app.get(
    "/about",
    response_model=About,
    include_in_schema=False,
    summary="Informacje o aplikacji",
    description="Zwraca metadane serwisu rags_tool (nazwa, wersja, autor, opis).",
)
# Return static service metadata (name, version, author, description).
def about():
    """Return basic service metadata (name, version, author, description)."""
    return About()


@app.get(
    "/health",
    include_in_schema=False,
    summary="Stan usługi",
    description="Sprawdza połączenie z Qdrant i raportuje kondycję aplikacji.",
)
# Lightweight health probe with Qdrant connectivity check.
def health():
    """Return health status with a light Qdrant connectivity check."""
    try:
        qdrant.get_collections()
        return {"status": "ok", "qdrant": True}
    except Exception as e:
        return {"status": "degraded", "qdrant": False, "error": str(e)}


# --- Browse/analytics (LLM-friendly) ---




@app.post(
    "/browse/doc-ids",
    response_model=BrowseIdsResponse,
    summary="Lista doc_id i metadanych",
    description=(
        "Zwraca listę dokumentów (doc_id, tytuł, data, is_active, doc_kind) spełniających zapytanie. "
        "'candidates_total' to całkowita liczba kandydatów po filtrach (niezależna od 'limit'). "
        "Dla 'limit=0' zawsze zwracana jest dokładna wartość 'candidates_total'. Jeżeli podano zawężenie treścią (query) lub filtrem 'kinds', "
        "odpowiedź zawiera także próbkę do 15 dokumentów (ORDER BY doc_date DESC); 'approx=true' oznacza próbkę niepełną. "
        "Gdy brak takich zawężeń (pełny korpus), 'limit=0' zwraca wyłącznie 'candidates_total' (bez listy). "
        "Parametry: query, match=phrase|any|all (domyślnie phrase), status=active|inactive|all, kinds."
    ),
    operation_id="rags_tool_browse_doc_ids",
    tags=["tools"],
)
def browse_doc_ids(req: DocIdsQuery, limit: int = Query(200, ge=0, le=5000)) -> BrowseIdsResponse:
    t0 = time.time()
    if not _qdrant_available_or_log("/browse/doc-ids"):
        took_ms = int((time.time() - t0) * 1000)
        return BrowseIdsResponse(took_ms=took_ms, total=0, approx=False, docs=[])
    try:
        _ensure_collections_cached()
    except Exception as exc:
        logger.error(
            "Qdrant ensure_collections failed | context=/browse/doc-ids url=%s error=%s",
            getattr(settings, "qdrant_url", ""),
            exc,
        )
        took_ms = int((time.time() - t0) * 1000)
        return BrowseIdsResponse(took_ms=took_ms, total=0, approx=False, docs=[])
    # Normalize query to a list of non-empty strings; avoid implicit str(None)
    queries_raw = req.query
    if not isinstance(queries_raw, list):
        queries_raw = [queries_raw]
    queries = [str(q or "").strip() for q in queries_raw if str(q or "").strip()]
    docs, approx, candidates_total = browse_service.list_doc_ids_via_fts(
        queries,
        match=str(getattr(req, "match", "phrase")),
        status=str(getattr(req, "status", "active")),
        kinds=getattr(req, "kinds", None),
        limit=int(limit),
    )
    took_ms = int((time.time() - t0) * 1000)
    # For limit=0 we may return a small sample (when filters present) along with accurate candidates_total.
    return BrowseIdsResponse(
        took_ms=took_ms,
        total=len(docs),
        approx=bool(approx),
        candidates_total=int(candidates_total),
        docs=docs,
    )


@app.get(
    "/docs/stats",
    response_model=DocStatsResponse,
    include_in_schema=False,
    summary="Statystyka dokumentów (FTS)",
    description="Zwraca liczbę dokumentów w korpusie (aktywnych/nieaktywnych/w sumie) w oparciu o indeks FTS (distinct doc_id).",
)
def docs_stats() -> DocStatsResponse:
    t0 = time.time()
    if not _qdrant_available_or_log("/docs/stats"):
        return DocStatsResponse(total_docs=0, active_docs=0, inactive_docs=0)
    try:
        _ensure_collections_cached()
    except Exception:
        return DocStatsResponse(total_docs=0, active_docs=0, inactive_docs=0)
    total, active, inactive = fts_doc_counts()
    return DocStatsResponse(total_docs=int(total), active_docs=int(active), inactive_docs=int(inactive))


@app.post(
    "/quotes/find",
    response_model=QuotesFindResponse,
    summary="Quotes: znajdź wystąpienia w wybranych dokumentach",
    description=(
        "Enumeruje wystąpienia frazy/tokenów w treści dokumentów ograniczonych przez 'restrict_doc_ids'. "
        "Nie używa MMR ani top_k; paginuje deterministycznie po doc_id→chunk_id→pozycja. "
        "Wymaga wcześniejszego pozyskania listy doc_id (np. z POST /browse/doc-ids)."
    ),
    tags=["tools"],
)
def quotes_find(req: QuotesFindRequest) -> QuotesFindResponse:
    t0 = time.time()
    if not _qdrant_available_or_log("/quotes/find"):
        took_ms = int((time.time() - t0) * 1000)
        return QuotesFindResponse(took_ms=took_ms, total_quotes=0, returned=0, complete=True, next_cursor=None, quotes=[])
    # Validate doc_ids
    raw_ids = req.restrict_doc_ids or []
    seen: Set[str] = set()
    doc_ids: List[str] = []
    for v in raw_ids:
        s = str(v or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        doc_ids.append(s)
    if not doc_ids:
        raise HTTPException(status_code=422, detail="quotes/find wymaga niepustego 'restrict_doc_ids'.")

    # Normalize queries
    q_raw = req.query
    if not isinstance(q_raw, list):
        q_raw = [q_raw]
    queries = [str(x or "").strip() for x in q_raw if str(x or "").strip()]
    if not queries:
        raise HTTPException(status_code=422, detail="Parametr 'query' musi zawierać frazę lub listę fraz.")

    match = (req.match or "phrase").strip().lower()
    if match not in {"phrase", "any", "all", "regex"}:
        match = "phrase"
    case_sensitive = bool(req.case_sensitive)
    granularity = (req.granularity or "occurrence").strip().lower()
    if granularity not in {"occurrence", "chunk"}:
        granularity = "occurrence"
    ctx = int(max(0, min(400, int(getattr(req, "context_chars", 80) or 80))))
    limit = int(max(1, min(1000, int(getattr(req, "limit", 200) or 200))))

    # Prepare search objects
    tokens: List[str] = []
    pattern: Optional[re.Pattern[str]] = None
    if match == "phrase":
        phrases = queries
    elif match in {"any", "all"}:
        for q in queries:
            tokens.extend(_tokenize(q))
        # stabilize and dedupe preserving order
        seen_tok: Set[str] = set()
        toks: List[str] = []
        for t in tokens:
            if t and t not in seen_tok:
                seen_tok.add(t)
                toks.append(t)
        tokens = toks
        if not tokens:
            raise HTTPException(status_code=422, detail="Brak tokenów do wyszukania (match=any/all)")
        phrases = []
    else:  # regex
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile("|".join(f"({p})" for p in queries), flags)
        except re.error as exc:
            raise HTTPException(status_code=422, detail=f"Błędny regex: {exc}") from exc
        phrases = []

    # Optional: fetch doc metadata (titles/dates) once
    try:
        doc_meta = _fetch_doc_summaries(doc_ids)
    except Exception:
        doc_meta = {}

    # Cursor decode
    cur = _decode_cursor(req.cursor)
    start_doc_i = int(cur.get("doc_index", 0)) if cur else 0
    start_chunk_i = int(cur.get("chunk_index", 0)) if cur else 0
    start_occ_i = int(cur.get("occ_index", 0)) if cur else 0

    quotes: List[QuoteItem] = []
    produced = 0
    total_found = 0
    next_cursor: Optional[str] = None

    # Iterate documents deterministically in provided order
    for di, did in enumerate(doc_ids):
        if di < start_doc_i:
            continue
        chunks = _fetch_doc_chunks(did)
        if not chunks:
            continue
        # Sort by chunk_id for stable order
        try:
            chunks.sort(key=lambda p: int(p.get("chunk_id", 0)))
        except Exception:
            pass

        for ci, ch in enumerate(chunks):
            if di == start_doc_i and ci < start_chunk_i:
                continue
            text = str(ch.get("text") or "")
            if not text:
                continue

            occs: List[Tuple[int, int]] = []
            if match == "phrase":
                for ph in phrases:
                    occs.extend(_find_all(text, ph, case_sensitive=case_sensitive))
            elif match in {"any", "all"}:
                found_map: Dict[str, List[Tuple[int, int]]] = {}
                for tok in tokens:
                    mm = _find_all(text, tok, case_sensitive=case_sensitive)
                    if mm:
                        found_map[tok] = mm
                if match == "all" and (set(tokens) - set(found_map.keys())):
                    occs = []
                else:
                    for lst in found_map.values():
                        occs.extend(lst)
            else:  # regex
                assert pattern is not None
                try:
                    for m in pattern.finditer(text):
                        s, e = int(m.start()), int(m.end())
                        if e > s:
                            occs.append((s, e))
                except Exception:
                    occs = []

            occs = _merge_and_sort_occurrences(occs)
            if not occs:
                continue

            # Resume within chunk if cursor points here
            occ_start_index = start_occ_i if (di == start_doc_i and ci == start_chunk_i) else 0
            if granularity == "chunk":
                # One record per chunk: use first occurrence beyond cursor
                if occ_start_index >= len(occs):
                    continue
                s, e = occs[occ_start_index]
                total_found += len(occs)
                # Produce exactly one item for this chunk
                left = text[max(0, s - ctx):s]
                mid = text[s:e]
                right = text[e:min(len(text), e + ctx)]
                meta = doc_meta.get(did, {})
                section = _canonical_section_label(ch.get("section_path"), getattr(settings, "section_merge_level", "ust"))
                quotes.append(
                    QuoteItem(
                        doc_id=did,
                        path=str(ch.get("path") or ""),
                        title=meta.get("doc_title"),
                        doc_date=meta.get("doc_date"),
                        is_active=meta.get("is_active"),
                        section=section,
                        chunk_id=int(ch.get("chunk_id", 0)),
                        start=int(s),
                        end=int(e),
                        left_context=left,
                        text=mid,
                        right_context=right,
                    )
                )
                produced += 1
                if produced >= limit:
                    next_cursor = _encode_cursor({"doc_index": di, "chunk_index": ci + 1, "occ_index": 0})
                # We still continue scanning to compute total_found
            else:
                # occurrence-level items
                total_found += len(occs)
                for oi, (s, e) in enumerate(occs):
                    if oi < occ_start_index:
                        continue
                    if produced < limit:
                        left = text[max(0, s - ctx):s]
                        mid = text[s:e]
                        right = text[e:min(len(text), e + ctx)]
                        meta = doc_meta.get(did, {})
                        section = _canonical_section_label(ch.get("section_path"), getattr(settings, "section_merge_level", "ust"))
                        quotes.append(
                            QuoteItem(
                                doc_id=did,
                                path=str(ch.get("path") or ""),
                                title=meta.get("doc_title"),
                                doc_date=meta.get("doc_date"),
                                is_active=meta.get("is_active"),
                                section=section,
                                chunk_id=int(ch.get("chunk_id", 0)),
                                start=int(s),
                                end=int(e),
                                left_context=left,
                                text=mid,
                                right_context=right,
                            )
                        )
                        produced += 1
                        if produced >= limit:
                            next_cursor = _encode_cursor({"doc_index": di, "chunk_index": ci, "occ_index": oi + 1})
                    # continue to count total_found regardless of the limit

            # Reset within-chunk cursor after first applicable chunk
            if di == start_doc_i and ci == start_chunk_i:
                start_occ_i = 0

        # Reset start_chunk when moving to next doc
        if di == start_doc_i:
            start_chunk_i = 0

    took_ms = int((time.time() - t0) * 1000)
    complete = next_cursor is None
    return QuotesFindResponse(
        took_ms=took_ms,
        total_quotes=int(total_found),
        returned=len(quotes),
        complete=bool(complete),
        next_cursor=next_cursor,
        quotes=quotes,
    )


@app.post(
    "/fts/rebuild",
    include_in_schema=False,
    summary="Odbuduj lokalny indeks FTS",
    description="Przebudowuje lokalny indeks FTS (SQLite FTS5) na podstawie kolekcji chunków w Qdrant.",
)
def fts_rebuild():
    t0 = time.time()
    if not _qdrant_available_or_log("/fts/rebuild"):
        return {"ok": False, "inserted": 0, "took_ms": 0}
    try:
        _ensure_collections_cached()
    except Exception:
        pass
    before = 0
    try:
        before = int(fts_count())
    except Exception:
        before = 0
    inserted = int(rebuild_fts_from_qdrant())
    after = int(fts_count())
    took_ms = int((time.time() - t0) * 1000)
    return {"ok": True, "inserted": inserted, "before": before, "after": after, "took_ms": took_ms}


# NOTE: /browse/facets endpoint was removed in 2.43.0. Use /browse/doc-ids for
# listing and counts (via candidates_total) and aggregate on the client if needed.


# --- Search debug: step-by-step endpoints ---





@app.post(
    "/collections/init",
    include_in_schema=False,
    summary="Inicjalizacja kolekcji",
    description=(
        "Tworzy parę kolekcji (streszczenia + treść) dla wskazanej nazwy bazowej (jeśli nie istnieją) i opcjonalnie sondą sprawdza wymiar embeddingów przy użyciu force_dim_probe."
    ),
)
# Initialize or validate summary/content collections for a base name.
def collections_init(req: InitCollectionsRequest):
    """Create or validate the pair of collections for the given base name."""
    from app.core.embedding import get_embedding_dim

    dim = get_embedding_dim() if req.force_dim_probe else None
    ensure_collections(req.collection_name, dim)
    _mark_collections_initialized(req.collection_name)
    summary_collection, content_collection = derive_collection_names(req.collection_name)
    return {
        "ok": True,
        "collection": req.collection_name,
        "summary_collection": summary_collection,
        "content_collection": content_collection,
    }


@app.post(
    "/collections/export",
    include_in_schema=False,
    summary="Eksport kolekcji Qdrant",
    description="Eksportuje wszystkie kolekcje do archiwum .tar.gz zawierającego snapshoty Qdrant oraz lokalne artefakty TF-IDF.",
)
# Export all collections and TF‑IDF artifacts as a tar.gz bundle.
def collections_export(req: CollectionsExportRequest):
    """Export all Qdrant collections and local TF-IDF artifacts to tar.gz."""
    if req.collection_names:
        logger.info(
            "Parametr collection_names=%s został przesłany, ale eksport obejmuje wszystkie kolekcje.",
            req.collection_names,
        )
    bundle, meta = export_collections_bundle(req.collection_names)
    filename = meta.get("filename") or "qdrant-export.tar.gz"
    headers = {
        "Content-Disposition": f"attachment; filename={filename}",
        "X-Rags-Collections": ",".join(meta.get("collections", [])),
        "X-Rags-Vector-Store": ",".join(meta.get("vector_store_files", [])),
        "X-Rags-Snapshots": ",".join(meta.get("snapshots", [])),
    }
    return StreamingResponse(io.BytesIO(bundle), media_type="application/gzip", headers=headers)


@app.post(
    "/collections/import",
    include_in_schema=False,
    summary="Import kolekcji Qdrant",
    description=(
        "Przyjmuje archiwum .tar.gz wygenerowane przez /collections/export (plik lub base64) i odtwarza kolekcje ze snapshotów Qdrant oraz indeksy TF-IDF."
    ),
)
# Import collections from an uploaded tar.gz bundle (multipart or raw body).
async def collections_import(
    request: Request,
    archive_file: UploadFile | None = File(
        default=None,
        description="Archiwum .tar.gz wygenerowane przez /collections/export.",
    ),
    replace_existing_form: bool | None = Form(
        default=None,
        description="Czy nadpisać istniejące kolekcje (gdy wysyłasz formularz).",
    ),
    replace_existing_query: bool = Query(
        default=True,
        description="Czy nadpisać istniejące kolekcje (dla zapytań bez formularza).",
    ),
):
    binary_archive: bytes | None = None
    replace_existing = replace_existing_query

    if archive_file is not None:
        binary_archive = await archive_file.read()
        await archive_file.close()
        if replace_existing_form is not None:
            replace_existing = replace_existing_form
    else:
        content_type = request.headers.get("content-type", "").lower()
        if "application/json" in content_type:
            try:
                payload_data = await request.json()
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Nie udało się wczytać JSON: {exc}") from exc

            try:
                payload = CollectionsImportRequest(**payload_data)
            except ValidationError as exc:
                raise HTTPException(status_code=422, detail=exc.errors()) from exc

            try:
                binary_archive = base64.b64decode(payload.archive_base64)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Nie udało się zdekodować base64: {exc}") from exc
            replace_existing = payload.replace_existing
        elif "application/octet-stream" in content_type or "application/gzip" in content_type:
            binary_archive = await request.body()
        elif content_type.startswith("multipart/form-data"):
            form = await request.form()
            if "archive_base64" in form:
                try:
                    payload = CollectionsImportRequest(
                        archive_base64=str(form["archive_base64"]),
                        replace_existing=replace_existing_form if replace_existing_form is not None else replace_existing_query,
                    )
                except ValidationError as exc:
                    raise HTTPException(status_code=422, detail=exc.errors()) from exc
                try:
                    binary_archive = base64.b64decode(payload.archive_base64)
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"Nie udało się zdekodować base64: {exc}") from exc
                replace_existing = payload.replace_existing
            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Formularz multipart musi zawierać pole 'archive_file' z plikiem archiwum lub 'archive_base64'."
                    ),
                )
        else:
            body_bytes = await request.body()
            if body_bytes:
                binary_archive = body_bytes

    if not binary_archive:
        raise HTTPException(
            status_code=400,
            detail=(
                "Brak archiwum do importu. Wyślij JSON z polem 'archive_base64', formularz multipart z polem 'archive_file' lub surowe archiwum .tar.gz."
            ),
        )

    summary = import_collections_bundle(binary_archive, replace_existing=replace_existing)
    with _collection_init_lock:
        _initialized_collections.clear()
    return {"status": "ok", **summary}


@app.post(
    "/ingest/scan",
    response_model=ScanResponse,
    include_in_schema=False,
    summary="Skanowanie korpusu",
    description="Zwraca listę plików w katalogu bazowym, które kwalifikują się do ingestu.",
)
# List candidate files for ingest under base_dir.
def ingest_scan(req: ScanRequest):
    """Return files under base_dir that qualify for ingest (by extension)."""
    base = pathlib.Path(req.base_dir)
    if not base.exists():
        raise HTTPException(status_code=400, detail="base_dir nie istnieje")
    files = [str(p) for p in _scan_files(base, req.glob, req.recursive)]
    return ScanResponse(files=files)


@app.post(
    "/summaries/generate",
    include_in_schema=False,
    summary="Streszczenia wybranych plików",
    description="Generuje streszczenia oraz podpisy dla listy plików bez zapisu do Qdrant.",
)
# Generate summaries for the given files (no persistence).
def summaries_generate(req: SummariesGenerateRequest):
    """Generate summaries for the provided files without persisting results."""
    results = {}
    for f in req.files:
        p = pathlib.Path(f)
        if not p.exists():
            results[f] = {"error": "not found"}
            continue
        text = extract_text(p)
        summ = llm_summary(text, path=str(p))
        results[f] = summ
    return {"results": results}


@app.post(
    "/ingest/build",
    include_in_schema=False,
    summary="Pełny ingest korpusu",
    description=(
        "Buduje indeks rags_tool: parsuje dokumenty, tworzy streszczenia, embeddingi oraz zapisuje punkty (wraz z TF-IDF) do Qdrant."
    ),
)
# Full ingest: parse, summarize, embed and upsert into Qdrant.
def ingest_build(req: IngestBuildRequest):
    """Parse, summarize, embed and upsert corpus into Qdrant (full ingest)."""
    t0 = time.time()
    logger.debug(
        "Starting ingest build | base_dir=%s glob=%s recursive=%s reindex=%s collection_name=%s",
        req.base_dir,
        req.glob,
        req.recursive,
        req.reindex,
        req.collection_name,
    )

    logical_base = req.collection_name
    target_base = logical_base

    # For the default corpus, aliases (settings.qdrant_*_collection) are derived
    # from settings.collection_name with base=None. Keep this mapping so that
    # alias swaps affect the names used by search/browse.
    alias_base: Optional[str] = logical_base
    if logical_base == settings.collection_name:
        alias_base = None

    if req.reindex:
        # Full rebuild: create a fresh pair of collections for a new base and
        # populate them, then atomically repoint the aliases to this new pair.
        import datetime as _dt

        _clear_collection_cache(logical_base)
        stamp = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        target_base = f"{logical_base}_v{stamp}"
        logger.info(
            "Reindex requested; building into new versioned base '%s' (logical_base=%s)",
            target_base,
            logical_base,
        )
    else:
        logger.info(
            "Incremental ingest; using existing base '%s' (logical_base=%s)",
            target_base,
            logical_base,
        )

    # Ensure target collections (physical) exist; aliases are handled inside.
    ensure_collections(target_base)
    _mark_collections_initialized(target_base)

    base = pathlib.Path(req.base_dir)
    if not base.exists():
        raise HTTPException(status_code=400, detail="base_dir nie istnieje")

    file_paths = _scan_files(base, req.glob, req.recursive)
    # Pre-load external document URLs map from CSV in corpus root
    doc_url_map = _load_doc_url_map(base)
    logger.debug("Found %d files for ingest", len(file_paths))
    if not file_paths:
        return {"ok": True, "indexed": 0, "took_ms": int((time.time() - t0) * 1000)}

    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = pathlib.Path(tmpdir) / "doc_records.jsonl"
        doc_count = 0
        chunk_count = 0
        summary_count = 0
        stats: Dict[str, int] = {"duplicates_skipped": 0}

        with store_path.open("w", encoding="utf-8") as fh:
            for record in _iter_document_records(
                file_paths,
                req.chunk_tokens,
                req.chunk_overlap,
                force_regen_summary=req.force_regen_summary,
                collection_base=logical_base,
                dedupe_on_ingest=bool(settings.dedupe_on_ingest),
                doc_url_map=doc_url_map,
                stats=stats,
            ):
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                doc_count += 1
                chunk_count += len(record.get("chunks", []))
                if record.get("summary_sparse_text"):
                    summary_count += 1

        if doc_count == 0:
            took_ms = int((time.time() - t0) * 1000)
            return {"ok": True, "indexed": 0, "documents": 0, "duplicates_skipped": int(stats.get("duplicates_skipped", 0)), "took_ms": took_ms}

        if req.enable_sparse:
            chunk_corpus = IterableCorpus(
                size=chunk_count,
                factory=lambda: _iter_chunk_texts(store_path),
            )
            summary_corpus = IterableCorpus(
                size=summary_count,
                factory=lambda: _iter_summary_texts(store_path),
            )
        else:
            chunk_corpus = None
            summary_corpus = None

        content_vec, summary_vec = prepare_tfidf(
            chunk_corpus,
            summary_corpus,
            req.enable_sparse,
            req.rebuild_tfidf,
        )

        point_count = build_and_upsert_points(
            _iter_saved_records(store_path),
            content_vec,
            summary_vec,
            enable_sparse=req.enable_sparse,
            collection_base=target_base,
        )

    if req.reindex:
        # Atomically repoint aliases for this logical base so that the active
        # pair (used by search/browse) now references the freshly built version.
        try:
            swap_collection_aliases(alias_base, target_base)
            logger.info(
                "Aliases repointed after reindex | logical_base=%s new_base=%s",
                logical_base,
                target_base,
            )
        except HTTPException:
            # Bubble up HTTPException as-is so the caller sees clear status/diagnostic.
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Nie udało się przełączyć aliasów po reindex dla kolekcji '{logical_base}': {exc}",
            ) from exc

    took_ms = int((time.time() - t0) * 1000)
    logger.debug(
        "Ingest build finished | documents=%d points=%d duplicates_skipped=%d doc_url_matched=%d doc_url_missing=%d took_ms=%d",
        doc_count,
        point_count,
        int(stats.get("duplicates_skipped", 0)),
        int(stats.get("doc_url_matched", 0)),
        int(stats.get("doc_url_missing", 0)),
        took_ms,
    )
    return {
        "ok": True,
        "indexed": point_count,
        "documents": doc_count,
        "duplicates_skipped": int(stats.get("duplicates_skipped", 0)),
        "took_ms": took_ms,
    }


@app.post(
    "/search/query",
    response_model=SearchResponse,
    summary="rags_tool search (LLM tool)",
    operation_id="rags_tool_search",
    tags=["tools"],
    description=settings.search_tool_description,
)
# Two‑stage hybrid RAG retrieval returning evidence blocks for answers.
def search_query(req: SearchQuery):
    """
    Endpoint: /search/query (POST)

    Purpose
    -------
    Two‑stage hybrid RAG retrieval for LLM‑powered tools. Stage 1 ranks document
    summaries; Stage 2 ranks full‑text chunks within the selected documents using a
    combination of dense embeddings and TF‑IDF sparse vectors, with optional hybrid MMR
    diversification and a per‑document cap. The endpoint returns concise, merged
    evidence blocks suitable for direct citation in answers.

    Important
    ---------
    This endpoint is for answer retrieval only. Do not use it to list
    documents. For lists/counts use the browse endpoint: POST /browse/doc-ids.

    Default scope
    -------------
    When `mode` is "auto", prefer current (obowiązujące) documents by default. Switch
    to archival only when the query clearly points to historical context (keywords or
    explicit years), and to all only on explicit user intent (e.g., "wszystkie").

    Parameters (SearchQuery)
    ------------------------
    - **query** (List[str]): list of focused queries (3–12 words each; prefer titles/signatures/dates). All queries are executed and results are fused.
    - **top_m** (int): candidate documents for Stage 1 (default 100; typical 50–200).
    - **top_k** (int): global number of final results after Stage 2 (typical 5–10).
      Use `per_doc_limit` to prevent dominance of a single document.
    - **mode** (str): `"auto"` (detect current/archival), `"current"`, `"archival"` or `"all"`.
    - **use_hybrid** (bool): enable dense + sparse scoring.
    - **dense_weight** / **sparse_weight** (float): weighting of dense vs sparse scores.
    - **mmr_lambda** (float): trade‑off between relevance and diversity (default 0.3).
    - **per_doc_limit** (int): max chunks per document after MMR.
    - **score_norm** (str): `"minmax"`, `"zscore"` or `"none"` for score normalisation.
    - **rep_alpha** (float): weighting between dense and sparse similarity in MMR.
    - **mmr_stage1** (bool): apply MMR already at document selection.
    - **summary_mode** (str): `"none" | "first" | "all"` (summary duplication strategy).
    - **result_format** (str): `"flat" | "grouped" | "blocks"` (`"blocks"` is default and recommended for tools).
    - **entities** (List[str], optional): encje wydobyte z intencji (nazwy/ID/lata/cytowane frazy). Gdy pominięte i `AUTO_EXTRACT_QUERY_ENTITIES=true`, backend użyje heurystyk dla boostingu.
    - **restrict_doc_ids** (List[str], optional): lista doc_id zawężająca zakres wyszukiwania. UŻYWAJ TYLKO, gdy wcześniej pobrałeś kandydatów z POST /browse/doc-ids; w przeciwnym razie pozostaw puste.
    - **entity_strategy** (str): `"optional" | "auto" | "boost" | "must_any" | "must_all" | "exclude"` (domyślnie `optional`). Filtry `must_*`/`exclude` ograniczają pulę na Etapie 1 (i Etapie 2 w trybie skip Stage‑1); `optional`/`auto`/`boost` dodają miękki bonus do rankingów bez twardego filtra.

    Recommendations for LLM callers
    --------------------------------
    * Prefer `result_format="blocks"` to obtain concise evidence blocks; blocks are
      built by merging all chunks of a given section into a single text block.
    * Keep `top_k` between 5‑10 unless you need finer granularity.
    * `summary_mode="first"` returns a single document summary per hit, useful for
      citation.
    * The returned payload contains `blocks` where each block includes:
        - `text` – concatenated chunk text (evidence).
        - `path` – source file path.
        - `title` – document title.
        - `doc_date` – document date (YYYY, YYYY-MM, or YYYY-MM-DD) or 'brak'.
        - `is_active` – whether the document is current (true) or archival (false).
        - `first_chunk_id` / `last_chunk_id` – range of original chunk IDs.
        - `score` – relevance score.

    Returns
    -------
    SearchResponse containing timing, hits (if not using blocks), optional groups,
    and a list of `blocks` when `result_format="blocks"`.

    Example request (JSON):
    {
        "query": [
            "Jak działa rags_tool?",
            "architektura rags_tool"
        ],
        "top_m": 10,
        "top_k": 5,
        "mode": "auto",
        "use_hybrid": true,
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
        "mmr_lambda": 0.3,
        "per_doc_limit": 2,
        "score_norm": "minmax",
        "rep_alpha": 0.6,
        "mmr_stage1": true,
        "summary_mode": "first",
        "result_format": "blocks"
    }

    The endpoint returns a JSON with `blocks` ready for citation by the LLM.
    """
    t0 = time.time()
    # Preflight: log and degrade to an empty response if Qdrant is unavailable
    if not _qdrant_available_or_log("/search/query"):
        took_ms = int((time.time() - t0) * 1000)
        return SearchResponse(took_ms=took_ms, hits=[])
    # --- RERANKER: konfiguracja z .env ---
    # Włączony wtedy, gdy podano zarówno BASE_URL, jak i MODEL.
    ranker_enabled = bool(settings.ranker_base_url and settings.ranker_model)
    # Minimalne parametry sterowane z .env (LLM nie może ich nadpisać):
    RERANK_TOP_N_MAX = max(1, int(getattr(settings, "rerank_top_n_max", 50)))
    RETURN_TOP_K_MAX = max(1, int(getattr(settings, "return_top_k_max", 50)))
    RANKER_THRESHOLD = float(settings.ranker_score_threshold)
    RANKER_MAX_LEN = max(1, int(settings.ranker_max_length))
    # Internal fusion defaults (hidden from tool schema/LLM)
    RRF_K = 60
    OVERSAMPLE = 2
    DEDUPE_BY = "chunk"  # other option could be "doc", kept internal

    queries = [q.strip() for q in (req.query or []) if str(q or "").strip()]
    if not queries:
        raise HTTPException(status_code=422, detail="Field 'query' must contain at least one non-empty string")

    query_hash = sha1(json.dumps(queries, ensure_ascii=False))
    try:
        _ensure_collections_cached()
    except Exception as exc:
        logger.error(
            "Qdrant ensure_collections failed | context=/search/query url=%s error=%s",
            getattr(settings, "qdrant_url", ""),
            exc,
        )
        took_ms = int((time.time() - t0) * 1000)
        return SearchResponse(took_ms=took_ms, hits=[])

    # Determine unified mode when 'auto' is requested
    if req.mode != "auto":
        mode = req.mode
    else:
        modes = {_classify_mode(q, "auto") for q in queries}
        if modes == {"current"}:
            mode = "current"
        elif modes == {"archival"}:
            mode = "archival"
        else:
            mode = "all"

    flt = None
    if mode in ("current", "archival"):
        flt = qm.Filter(must=[qm.FieldCondition(key="is_active", match=qm.MatchValue(value=(mode == "current")))])

    # Entities filtering (only when LLM requests a filter strategy)
    try:
        strat = str(getattr(req, "entity_strategy", "auto") or "auto").strip().lower()
    except Exception:
        strat = "auto"
    if strat in {"must_any", "must_all", "exclude"}:
        raw_list = getattr(req, "entities", None) or []
        # Build union of raw and casefolded forms to mitigate case-sensitive KEYWORD matching.
        ordered_raw: List[str] = []
        seen_raw: Set[str] = set()
        for v in raw_list:
            s = str(v or "").strip()
            if not s or s in seen_raw:
                continue
            seen_raw.add(s)
            ordered_raw.append(s)
        ents_union: List[str] = []
        seen_all: Set[str] = set()
        for v in ordered_raw:
            for cand in (v, v.casefold()):
                if cand and cand not in seen_all:
                    seen_all.add(cand)
                    ents_union.append(cand)
        if ents_union:
            base_must = list(getattr(flt, "must", []) or []) if flt else []
            base_must_not = list(getattr(flt, "must_not", []) or []) if flt else []
            if strat == "must_any":
                # Any of raw/casefold may match
                base_must.append(qm.FieldCondition(key="entities", match=qm.MatchAny(any=ents_union)))
            elif strat == "must_all":
                # For each original token require (raw OR casefold) using MatchAny
                for v in ordered_raw:
                    variants = [x for x in (v, v.casefold()) if x]
                    base_must.append(qm.FieldCondition(key="entities", match=qm.MatchAny(any=variants)))
            elif strat == "exclude":
                base_must_not.append(qm.FieldCondition(key="entities", match=qm.MatchAny(any=ents_union)))
            flt = qm.Filter(
                must=base_must or None,
                must_not=base_must_not or None,
                should=(getattr(flt, "should", None) if flt else None),
            )

    # Optional restriction by explicit doc_id allowlist (use only with prior /browse/doc-ids)
    restrict_ids_raw = getattr(req, "restrict_doc_ids", None) or []
    if restrict_ids_raw:
        seen: Set[str] = set()
        restrict_ids: List[str] = []
        for v in restrict_ids_raw:
            s = str(v or "").strip()
            if not s or s in seen:
                continue
            seen.add(s)
            restrict_ids.append(s)
            if len(restrict_ids) >= 2000:  # safety cap to keep filter manageable
                break
        if restrict_ids:
            base_must = list(getattr(flt, "must", []) or []) if flt else []
            base_must.append(qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=restrict_ids)))
            flt = qm.Filter(
                must=base_must or None,
                must_not=(getattr(flt, "must_not", None) if flt else None),
                should=(getattr(flt, "should", None) if flt else None),
            )
            # Heuristics: when search is restricted to a small allowlist, raise limits generously.
            try:
                n_ids = len(restrict_ids)
                # Target limits scale with the size of the subset; clamp to safe caps.
                target_top_m = min(500, max(500, 50 * n_ids))
                target_top_k = min(50, max(50, 5 * n_ids))
                target_per_doc = max(15, int(getattr(req, "per_doc_limit", 0) or 0))
                upd: Dict[str, Any] = {}
                if int(getattr(req, "top_m", 0) or 0) < target_top_m:
                    upd["top_m"] = int(target_top_m)
                if int(getattr(req, "top_k", 0) or 0) < target_top_k:
                    upd["top_k"] = int(target_top_k)
                if int(getattr(req, "per_doc_limit", 0) or 0) < target_per_doc:
                    upd["per_doc_limit"] = int(target_per_doc)
                if upd:
                    req = req.model_copy(update=upd)
            except Exception:
                # Best-effort only; proceed with client-provided values on any error
                pass

    # Batch-embed queries (with model-specific query prefix)
    q_vecs = embed_query(queries)

    # Accumulators for fusion and neighbor expansion
    fused: Dict[tuple, Dict[str, Any]] = {}
    global_mmr_pool: List[Dict[str, Any]] = []
    global_rel2: List[float] = []

    # Oversampled per-query limit to keep enough candidates after dedup/fusion
    # Jeśli ranker jest włączony, budżetujemy kandydatów pod rerank zamiast polegać na top_k z API.
    # Final output size respects req.top_k but is capped by RETURN_TOP_K_MAX
    final_k = min(max(1, int(req.top_k)), int(RETURN_TOP_K_MAX))
    if ranker_enabled:
        # Keep enough candidates per query for fusion/rerank, but respect server caps
        per_query_limit = max(1, min(int(RERANK_TOP_N_MAX), max(final_k, int(req.top_k) * OVERSAMPLE)))
    else:
        per_query_limit = max(1, int(req.top_k) * OVERSAMPLE)

    any_docs = False
    skip_stage1 = bool(settings.search_skip_stage1_default)
    for qi, (q, q_vec) in enumerate(zip(queries, q_vecs)):
        content_sparse_query, summary_sparse_query = _build_sparse_queries_for_query(q, req.use_hybrid)
        # Clone request with per-query top_k oversampled
        req_i = req.model_copy(update={"top_k": per_query_limit})
        if skip_stage1:
            # Pełny korpus: pomiń Etap 1 i wyszukuj bezpośrednio w chunkach (z zachowaniem filtra trybu)
            final_hits, mmr_pool, rel2 = _stage2_select_chunks(None, q, q_vec, content_sparse_query, {}, req_i, flt)
            if not final_hits:
                continue
            any_docs = True
        else:
            cand_doc_ids, doc_map = _stage1_select_documents(q, q_vec, flt, summary_sparse_query, req)
            if not cand_doc_ids:
                continue
            any_docs = True
            final_hits, mmr_pool, rel2 = _stage2_select_chunks(cand_doc_ids, q, q_vec, content_sparse_query, doc_map, req_i)
        # Append to global neighbor pools (used later for optional neighbor expansion)
        global_mmr_pool.extend(mmr_pool)
        global_rel2.extend(rel2)
        # RRF fusion on chunk identity
        for rank, fh in enumerate(final_hits, start=1):
            payload = fh.get("payload") or {}
            did = payload.get("doc_id") or ""
            sec = payload.get("section_path")
            cid = payload.get("chunk_id")
            if not did or cid is None:
                continue
            key = (did, sec, int(cid)) if DEDUPE_BY == "chunk" else (did, None, -1)
            entry = fused.get(key)
            incr = 1.0 / (RRF_K + rank)
            if entry is None:
                fused[key] = {
                    "payload": payload,
                    "score": incr,
                }
            else:
                entry["score"] += incr

    if not any_docs or not fused:
        took_ms = int((time.time() - t0) * 1000)
        logger.info(
            "Search finished | took_ms=%d query_hash=%s stage=fusion hits=0",
            took_ms,
            query_hash,
        )
        return SearchResponse(took_ms=took_ms, hits=[])

    # Build final fused list (RRF over unique chunks)
    fused_list = [
        {"payload": v.get("payload"), "score": float(v.get("score", 0.0))}
        for v in fused.values()
    ]
    fused_list.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    # Shape results first; merged blocks are built AFTER fusion
    # Dalsze kroki (reranking) wykonujemy na już zmergowanych blokach
    results, groups_payload, blocks_payload = _shape_results(fused_list, {}, global_mmr_pool, global_rel2, req)

    if req.result_format == "blocks":
        # Opcjonalny reranker na zmergowanych blokach sekcyjnych
        if ranker_enabled and (blocks_payload or []):
            try:
                client = OpenAIReranker(settings.ranker_base_url or "", settings.ranker_api_key, settings.ranker_model or "")
                passages = [
                    _truncate_head_tail(str(b.get("text", "")), max(1, int(settings.ranker_max_length)))
                    for b in (blocks_payload or [])
                ]
                q_joined = " | ".join(queries)
                top_n = min(len(passages), max(1, final_k))
                rr = client.rerank(query=q_joined, documents=passages, top_n=top_n)
                # Zamapuj ranking rankera do bloków (po indeksie wejściowym)
                idx_to_block = {i: b for i, b in enumerate(blocks_payload or [])}
                rr_sorted = sorted(rr, key=lambda r: float(r.get("relevance_score", 0.0)), reverse=True)

                # Gating progów rankera: miękki (RANKER_SCORE_THRESHOLD) i twardy (RANKER_HARD_THRESHOLD)
                soft_thr = float(getattr(settings, "ranker_score_threshold", 0.0) or 0.0)
                hard_thr = float(getattr(settings, "ranker_hard_threshold", 0.0) or 0.0)

                def _score(r) -> float:
                    try:
                        return float(r.get("relevance_score", 0.0))
                    except Exception:
                        return 0.0

                above_soft = [r for r in rr_sorted if _score(r) >= soft_thr]
                mid_band = [r for r in rr_sorted if _score(r) >= hard_thr and _score(r) < soft_thr]

                counts: Dict[str, int] = {}
                selected_blocks: List[Dict[str, Any]] = []

                def add_from(results_list):
                    nonlocal selected_blocks
                    for r in results_list:
                        i = int(r.get("index", -1))
                        if i < 0 or i >= len(idx_to_block):
                            continue
                        b = dict(idx_to_block[i])
                        b["ranker_score"] = _score(r)
                        did = str(b.get("doc_id", ""))
                        if did:
                            if counts.get(did, 0) >= max(1, int(req.per_doc_limit)):
                                continue
                            counts[did] = counts.get(did, 0) + 1
                        selected_blocks.append(b)
                        if len(selected_blocks) >= top_n:
                            break

                # Najpierw elementy ≥ soft_thr; jeśli za mało, dopełnij tylko elementami ≥ hard_thr
                add_from(above_soft)
                if len(selected_blocks) < top_n and mid_band:
                    add_from(mid_band)
                # Nigdy nie dokładamy elementów < hard_thr — wynik może być krótszy niż top_n
                blocks_payload = selected_blocks
            except Exception as exc:
                logger.warning("Ranker failed on merged blocks: %s", exc)
                # Fallback: bez reranka, utnij do top_k po score i per_doc_limit
                counts: Dict[str, int] = {}
                trimmed = []
                for b in sorted(blocks_payload or [], key=lambda x: float(x.get("score", 0.0)), reverse=True):
                    did = str(b.get("doc_id", ""))
                    if did:
                        if counts.get(did, 0) >= max(1, int(req.per_doc_limit)):
                            continue
                        counts[did] = counts.get(did, 0) + 1
                    trimmed.append(b)
                    if len(trimmed) >= max(1, final_k):
                        break
                blocks_payload = trimmed
        else:
            # Bez rankera: egzekwuj per_doc_limit i utnij do top_k po score
            counts: Dict[str, int] = {}
            trimmed = []
            for b in sorted(blocks_payload or [], key=lambda x: float(x.get("score", 0.0)), reverse=True):
                did = str(b.get("doc_id", ""))
                if did:
                    if counts.get(did, 0) >= max(1, int(req.per_doc_limit)):
                        continue
                    counts[did] = counts.get(did, 0) + 1
                trimmed.append(b)
                if len(trimmed) >= max(1, final_k):
                    break
            blocks_payload = trimmed

        took_ms = int((time.time() - t0) * 1000)
        logger.info(
            "Search finished | took_ms=%d query_hash=%s mode=%s fmt=blocks blocks=%d",
            took_ms,
            query_hash,
            mode,
            len(blocks_payload or []),
        )
        return SearchResponse(took_ms=took_ms, hits=[], groups=None, blocks=blocks_payload)

    took_ms = int((time.time() - t0) * 1000)
    logger.info(
        "Search finished | took_ms=%d query_hash=%s mode=%s fmt=%s hits=%d groups=%d blocks=%d",
        took_ms,
        query_hash,
        mode,
        req.result_format,
        len(results or []),
        len(groups_payload or []),
        len(blocks_payload or []),
    )
    return SearchResponse(took_ms=took_ms, hits=results, groups=groups_payload, blocks=blocks_payload)

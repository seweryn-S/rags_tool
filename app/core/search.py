"""Hybrid search logic: stage-1 summaries, stage-2 chunks, MMR, shaping."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
from qdrant_client.http import models as qm

from app.core.embedding import (
    SUMMARY_VECTORIZER_PATH,
    load_vectorizer,
    tfidf_vector,
)
from app.core.constants import (
    CONTENT_SPARSE_NAME,
    CONTENT_VECTOR_NAME,
    SPARSE_ENABLED,
    SUMMARY_SPARSE_NAME,
    SUMMARY_VECTOR_NAME,
)
from app.core.chunking import SECTION_HIERARCHY
from app.qdrant_utils import qdrant
from app.settings import get_settings
from app.core.ranker_client import OpenAIReranker
from app.core.constants import RANKER_USE_STAGE1

settings = get_settings()

# Vector names and defaults
 # names imported from constants

DEFAULT_MMR_LAMBDA = 0.3
DEFAULT_PER_DOC_LIMIT = 2
DEFAULT_SCORE_NORM = "minmax"  # minmax|zscore|none

SECTION_LEVEL_INDEX: Dict[str, int] = {level: idx for idx, level in enumerate(SECTION_HIERARCHY)}

# Normalize whitespace: trim and collapse internal spaces.
def _normalize_whitespace(text: Optional[str]) -> str:
    """Trim, collapse internal whitespace and return a safe string."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().split())


# --- Entities helpers ---
def _normalize_entity(value: Optional[str]) -> str:
    """Normalize an entity string for matching: casefold + collapse spaces."""
    if not isinstance(value, str):
        return ""
    s = value.casefold().strip()
    s = " ".join(s.split())
    # strip common quotes
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1].strip()
    return s


def _norm_entities_list(values: Optional[List[str]]) -> List[str]:
    """Normalize a list of entities and drop empties/duplicates preserving order."""
    out: List[str] = []
    seen: Set[str] = set()
    for v in (values or []):
        n = _normalize_entity(v)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _extract_entities_from_text(text: str) -> List[str]:
    """Heuristic entity extraction from a single query string.

    Captures:
    - quoted phrases ("...") including Polish quotes „…”
    - uppercase acronyms (2+ letters, may include dots)
    - numbers with slashes (e.g., 12/2024) and years (19xx/20xx)
    """
    s = (text or "").strip()
    if not s:
        return []
    ents: List[str] = []
    # quoted phrases
    for m in re.finditer(r'"([^"\n]{2,})"', s):
        ents.append(m.group(1))
    for m in re.finditer(r"„([^”\n]{2,})”", s):
        ents.append(m.group(1))
    # uppercase acronyms/words (allow dots and hyphens inside)
    for m in re.finditer(r"\b[A-ZĄĆĘŁŃÓŚŹŻ]{2,}(?:[.-][A-ZĄĆĘŁŃÓŚŹŻ0-9]{2,})*\b", s):
        ents.append(m.group(0))
    # numbers with slash or dash (IDs)
    for m in re.finditer(r"\b\d{1,4}[/-]\d{2,4}\b", s):
        ents.append(m.group(0))
    # years
    for m in re.finditer(r"\b(19|20)\d{2}\b", s):
        ents.append(m.group(0))
    return _norm_entities_list(ents)


def _query_entities_for(q_text: str, req: Any) -> List[str]:
    """Return normalized entities for this query, using req.entities or heuristics."""
    # prefer explicit entities provided by caller
    explicit = getattr(req, "entities", None)
    if explicit:
        return _norm_entities_list(explicit)
    if getattr(settings, "auto_extract_query_entities", True):
        return _extract_entities_from_text(q_text)
    return []


# Build canonical section label up to the merge level.
def _canonical_section_label(
    section_path: Optional[str],
    merge_level: str,
) -> Optional[str]:
    """Build a canonical section label up to the requested hierarchy level.

    Falls back to the full normalized path when the desired level cannot be
    detected in the provided `section_path` string.
    """
    if merge_level not in SECTION_LEVEL_INDEX:
        merge_level = "ust"
    normalized_merge = merge_level

    if isinstance(section_path, str) and section_path.strip():
        # Heurystyczne cięcie po poziomie merge_level bez metadanych poziomów
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
            collected.append(_normalize_whitespace(seg))
            if is_level(seg, normalized_merge):
                cut_done = True
                break
        if not cut_done:
            # Nie znaleziono segmentu odpowiadającego merge_level – użyj całej ścieżki
            collected = [_normalize_whitespace(seg) for seg in path_segments]
        joined = _normalize_whitespace(" > ".join(collected))
        return joined or None

    if isinstance(section_path, str) and section_path.strip():
        normalized = _normalize_whitespace(section_path)
        if normalized:
            return normalized
    return None


# Check whether reranker is configured (base URL + model).
def _ranker_enabled() -> bool:
    """Return True when an external reranker is configured (BASE_URL + MODEL)."""
    return bool(settings.ranker_base_url and settings.ranker_model)


# Truncate text to limit, preserving head and tail.
def _truncate_head_tail(text: str, limit: int) -> str:
    """Truncate to `limit` chars keeping 70% head and 30% tail."""
    t = (text or "").strip()
    if len(t) <= max(1, int(limit)):
        return t
    head = int(limit * 0.7)
    tail = max(0, int(limit) - head)
    return (t[:head] + "\n...\n" + t[-tail:]).strip()


# --- Date helpers ---
def _doc_date_ord(value: Optional[str]) -> int:
    """Convert doc_date string (YYYY, YYYY-MM, YYYY-MM-DD, or 'brak') to sortable ordinal.

    Returns an integer YYYYMMDD where missing month/day are treated as 00.
    Unknown/invalid values return 0 (oldest).
    """
    if not isinstance(value, str):
        return 0
    s = value.strip()
    if not s or s.lower() == "brak":
        return 0
    try:
        parts = s.split("-")
        y = int(parts[0]) if len(parts) >= 1 and parts[0].isdigit() else 0
        m = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
        d = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 0
        if y <= 0:
            return 0
        m = max(0, min(12, m))
        d = max(0, min(31, d))
        return int(y) * 10000 + int(m) * 100 + int(d)
    except Exception:
        return 0


def _apply_date_tiebreak_by_score(
    ids_in_order: List[str],
    primary_scores: List[float],
    doc_map: Dict[str, Any],
    *,
    epsilon: float = 1e-9,
) -> List[str]:
    """Apply secondary ordering by doc_date (DESC) within near-equal score groups.

    - `ids_in_order` corresponds positionally to `primary_scores`.
    - Groups consecutive items where |score[i] - score[first]| <= epsilon and sorts
      each group by doc_date DESC, then doc_id ASC for stability.
    """
    if not ids_in_order:
        return ids_in_order
    out: List[str] = []
    i = 0
    while i < len(ids_in_order):
        j = i + 1
        base = float(primary_scores[i] if i < len(primary_scores) else 0.0)
        while j < len(ids_in_order):
            sj = float(primary_scores[j] if j < len(primary_scores) else 0.0)
            if abs(sj - base) <= float(epsilon):
                j += 1
            else:
                break
        group = ids_in_order[i:j]
        group.sort(
            key=lambda did: (
                _doc_date_ord((doc_map.get(did, {}) or {}).get("doc_date")),
                str(did),
            ),
            reverse=True,
        )
        out.extend(group)
        i = j
    return out


# Call external reranker and map input index to score.
def _rerank_indices(query: str, passages: List[str], top_n: int) -> Dict[int, float]:
    """Call the reranker and return a map index->score for scored items.

    Note: `top_n` limits the number of scored results returned by the endpoint
    (not the number of inputs). Unreturned items have no score.
    """
    client = OpenAIReranker(settings.ranker_base_url or "", settings.ranker_api_key, settings.ranker_model or "")
    rr = client.rerank(query=query, documents=passages, top_n=min(max(1, top_n), len(passages)))
    return {int(it.get("index")): float(it.get("relevance_score")) for it in rr}


# Normalize a list of scores according to `method`.
def _normalize(values: List[float], method: str = DEFAULT_SCORE_NORM) -> List[float]:
    """Normalize scores via minmax/zscore/none according to `method`."""
    if not values:
        return []
    if method == "none":
        return values
    arr = np.array(values, dtype=float)
    if method == "zscore":
        mean = float(arr.mean())
        std = float(arr.std())
        if std < 1e-12:
            return [0.0 for _ in values]
        return [float((v - mean) / std) for v in arr]
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax - vmin < 1e-12:
        return [0.0 for _ in values]
    return [float((v - vmin) / (vmax - vmin)) for v in arr]


# Sparse dot product between query map and (indices, values) payload.
def _sparse_dot(query_lookup: Dict[int, float], indices: List[int], values: List[float]) -> float:
    """Compute dot product between a sparse query map and (indices, values)."""
    if not query_lookup or not indices or not values:
        return 0.0
    acc = 0.0
    for i, v in zip(indices, values):
        q = query_lookup.get(int(i))
        if q is not None:
            acc += q * float(v)
    return float(acc)


# Sparse cosine-like similarity for two vectors (CSC form).
def _sparse_pair_cos(
    a_idx: List[int], a_val: List[float], b_idx: List[int], b_val: List[float]
) -> float:
    """Compute cosine-like similarity for two sparse vectors in CSC form."""
    if not a_idx or not b_idx:
        return 0.0
    i = j = 0
    sim = 0.0
    while i < len(a_idx) and j < len(b_idx):
        ai = int(a_idx[i])
        bj = int(b_idx[j])
        if ai == bj:
            sim += float(a_val[i]) * float(b_val[j])
            i += 1
            j += 1
        elif ai < bj:
            i += 1
        else:
            j += 1
    return float(sim)


# Dense cosine similarity between two vectors.
def _cosine_dense(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two dense vectors (lists)."""
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / denom)


# Ensure a field condition is present in a filter.
def _with_must_condition(
    flt: Optional[qm.Filter], condition: qm.FieldCondition
) -> qm.Filter:
    """Return a new filter that ensures the given field condition is included."""
    if flt is None:
        return qm.Filter(must=[condition])
    must_items = list(flt.must or []) + [condition]
    should_items = list(flt.should or []) if flt.should else None
    must_not_items = list(flt.must_not or []) if flt.must_not else None
    return qm.Filter(must=must_items, should=should_items, must_not=must_not_items)


# Fetch summaries payloads for provided doc_ids (best‑effort).
def _fetch_doc_summaries(doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch summaries for given doc_ids from the summaries collection.

    Returns a map: doc_id -> { doc_id, doc_summary, doc_signature, doc_entities }
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
        # Best-effort; if scroll fails, return what we have (possibly empty)
        return out
    return out


# Build TF‑IDF sparse queries (content and summary) for the raw query.
def _build_sparse_queries_for_query(query: str, use_hybrid: bool) -> Tuple[
    Optional[Tuple[List[int], List[float]]], Optional[Tuple[List[int], List[float]]]
]:
    """Build content and summary TF-IDF queries for a given raw query string."""
    if not (use_hybrid and SPARSE_ENABLED):
        return None, None
    content_sparse_query: Optional[Tuple[List[int], List[float]]] = None
    summary_sparse_query: Optional[Tuple[List[int], List[float]]] = None
    content_vec = load_vectorizer()
    if content_vec is not None:
        idx, val = tfidf_vector([query], content_vec)[0]
        if idx:
            content_sparse_query = (idx, val)
    summary_vec_model = load_vectorizer(path=SUMMARY_VECTORIZER_PATH)
    if summary_vec_model is not None:
        s_idx, s_val = tfidf_vector([query], summary_vec_model, path=SUMMARY_VECTORIZER_PATH)[0]
        if s_idx:
            summary_sparse_query = (s_idx, s_val)
    return content_sparse_query, summary_sparse_query


# Stage‑1: select candidate documents (summaries) with hybrid scoring.
def _stage1_select_documents(
    q_text: str,
    q_vec: List[float],
    flt: Optional[qm.Filter],
    summary_sparse_query: Optional[Tuple[List[int], List[float]]],
    req: Any,
) -> Tuple[List[str], Dict[str, Any]]:
    """Stage-1: retrieve and score document summaries; return ids and map.

    Supports single-query hybrid scoring and optional dual-query (dense+sparse)
    path when configured.
    """
    dual = bool(settings.search_dual_query_sparse and req.use_hybrid and SPARSE_ENABLED)
    point_type_filter = _with_must_condition(
        flt, qm.FieldCondition(key="point_type", match=qm.MatchValue(value="summary"))
    )
    if not dual:
        include_fields = [
            "doc_id",
            "path",
            "summary",
            "signature",
            "entities",
            "title",
            "doc_date",
            "is_active",
            "doc_url",
        ]
        if summary_sparse_query is not None and SPARSE_ENABLED:
            include_fields.extend(["summary_sparse_indices", "summary_sparse_values"])
        sum_search = qdrant.search(
            collection_name=settings.qdrant_summary_collection,
            query_vector=(SUMMARY_VECTOR_NAME, q_vec),
            query_filter=point_type_filter,
            limit=max(50, req.top_m),
            with_payload=include_fields if settings.search_minimal_payload else True,
            with_vectors=[SUMMARY_VECTOR_NAME] if req.mmr_stage1 else False,
            score_threshold=None,
            search_params=qm.SearchParams(exact=False, hnsw_ef=128),
        )
        summary_sparse_lookup: Dict[int, float] = {}
        if summary_sparse_query is not None:
            summary_sparse_lookup = dict(zip(summary_sparse_query[0], summary_sparse_query[1]))

        doc_map: Dict[str, Dict[str, Any]] = {}
        for r in sum_search:
            payload = r.payload or {}
            if payload.get("point_type") and payload.get("point_type") != "summary":
                continue
            did = payload.get("doc_id")
            if not did or did in doc_map:
                continue
            dense_score = float(r.score or 0.0)
            sparse_dot = 0.0
            if summary_sparse_lookup and payload.get("summary_sparse_indices") and payload.get("summary_sparse_values"):
                sparse_dot = _sparse_dot(summary_sparse_lookup, payload.get("summary_sparse_indices", []), payload.get("summary_sparse_values", []))
            vec_map = r.vector or {}
            dense_vec = vec_map.get(SUMMARY_VECTOR_NAME) or []
            doc_map[did] = {
                "doc_id": did,
                "dense_vec": dense_vec,
                "sparse_idx": payload.get("summary_sparse_indices", []) or [],
                "sparse_val": payload.get("summary_sparse_values", []) or [],
                "dense_score": dense_score,
                "sparse_score": sparse_dot,
                "path": payload.get("path"),
                "doc_summary": payload.get("summary"),
                "doc_signature": payload.get("signature"),
                "doc_entities": payload.get("entities"),
                "doc_title": payload.get("title"),
                "doc_date": payload.get("doc_date"),
                "is_active": payload.get("is_active"),
                "doc_url": payload.get("doc_url"),
            }

        if not doc_map:
            return [], {}

        doc_items = list(doc_map.values())
        dense_scores = [float(x["dense_score"]) for x in doc_items]
        sparse_scores = [float(x["sparse_score"]) for x in doc_items]
        dense_norm = _normalize(dense_scores, req.score_norm)
        sparse_norm = _normalize(sparse_scores, req.score_norm)
        hybrid_rel = [req.dense_weight * d + req.sparse_weight * s for d, s in zip(dense_norm, sparse_norm)]
    else:
        # Dual-query: dense + sparse, then fuse (no TF-IDF payload needed)
        include_fields = [
            "doc_id",
            "path",
            "summary",
            "signature",
            "entities",
            "title",
            "doc_date",
            "is_active",
            "doc_url",
        ]
        with_payload_dense = include_fields if settings.search_minimal_payload else True
        dense_hits = qdrant.search(
            collection_name=settings.qdrant_summary_collection,
            query_vector=(SUMMARY_VECTOR_NAME, q_vec),
            query_filter=point_type_filter,
            limit=max(50, req.top_m),
            with_payload=with_payload_dense,
            with_vectors=[SUMMARY_VECTOR_NAME] if req.mmr_stage1 else False,
            score_threshold=None,
            search_params=qm.SearchParams(exact=False, hnsw_ef=128),
        )
        sparse_hits = []
        if summary_sparse_query is not None and summary_sparse_query[0]:
            try:
                s_idx, s_val = summary_sparse_query
                with_payload_sparse = (["doc_id", "path"] if settings.search_minimal_payload else True)
                sparse_hits = qdrant.search(
                    collection_name=settings.qdrant_summary_collection,
                    query_vector=(SUMMARY_SPARSE_NAME, qm.SparseVector(indices=s_idx, values=s_val)),
                    query_filter=point_type_filter,
                    limit=max(50, req.top_m),
                    with_payload=with_payload_sparse,
                    with_vectors=False,
                    score_threshold=None,
                    search_params=qm.SearchParams(exact=False, hnsw_ef=128),
                )
            except Exception:
                sparse_hits = []
        doc_map = {}
        dense_scores_map: Dict[str, float] = {}
        sparse_scores_map: Dict[str, float] = {}
        for r in dense_hits:
            p = r.payload or {}
            did = p.get("doc_id")
            if not did:
                continue
            dense_scores_map[did] = float(r.score or 0.0)
            vec = (r.vector or {}).get(SUMMARY_VECTOR_NAME) or []
            doc_map[did] = {
                "doc_id": did,
                "dense_vec": vec,
                "path": p.get("path"),
                "doc_summary": p.get("summary"),
                "doc_signature": p.get("signature"),
                "doc_entities": p.get("entities"),
                "doc_title": p.get("title"),
                "doc_date": p.get("doc_date"),
                "is_active": p.get("is_active"),
                "doc_url": p.get("doc_url"),
            }
        for r in sparse_hits:
            p = r.payload or {}
            did = p.get("doc_id")
            if not did:
                continue
            sparse_scores_map[did] = float(r.score or 0.0)
            if did not in doc_map:
                doc_map[did] = {
                    "doc_id": did,
                    "dense_vec": [],
                    "path": p.get("path"),
                    "doc_summary": None,
                    "doc_signature": None,
                    "doc_entities": None,
                    "doc_title": None,
                    "doc_date": None,
                    "is_active": None,
                }
        if not doc_map:
            return [], {}
        doc_ids = list(doc_map.keys())
        dense_vals = [dense_scores_map.get(d, 0.0) for d in doc_ids]
        sparse_vals = [sparse_scores_map.get(d, 0.0) for d in doc_ids]
        dense_norm = _normalize(dense_vals, req.score_norm)
        sparse_norm = _normalize(sparse_vals, req.score_norm)
        hybrid_rel = [req.dense_weight * d + req.sparse_weight * s for d, s in zip(dense_norm, sparse_norm)]

    # --- Entities soft boost at Stage 1 (after fusion, before MMR/order) ---
    try:
        strat = str(getattr(req, "entity_strategy", "auto") or "auto").strip().lower()
    except Exception:
        strat = "auto"
    q_entities = set(_query_entities_for(q_text, req))
    if hybrid_rel and q_entities and strat != "exclude":
        ids_for_order = list(doc_map.keys())
        for i, did in enumerate(ids_for_order):
            ents_raw = doc_map.get(did, {}).get("doc_entities") or []
            ents = set(_norm_entities_list(ents_raw if isinstance(ents_raw, list) else []))
            if not ents:
                continue
            inter = ents.intersection(q_entities)
            if inter:
                # Cap contribution to prevent overpowering dense/sparse fusion
                boost = float(getattr(settings, "entity_boost_stage1", 0.15) or 0.0)
                scale = min(1.0, len(inter) / 3.0)
                hybrid_rel[i] = float(hybrid_rel[i]) + boost * scale

    if req.mmr_stage1 and (len(doc_map) > 1):
        rep_alpha = req.rep_alpha if req.rep_alpha is not None else req.dense_weight
        ids = list(doc_map.keys())
        dense_vecs = [doc_map[d]["dense_vec"] for d in ids]
        sparse_vecs = [([], []) for _ in ids]
        mmr_idx = mmr_diversify_hybrid(dense_vecs, sparse_vecs, hybrid_rel, min(req.top_m, len(ids)), req.mmr_lambda, rep_alpha)
        order_idx = mmr_idx
    else:
        ids = list(doc_map.keys())
        order_idx = sorted(range(len(ids)), key=lambda i: hybrid_rel[i], reverse=True)

    # Etap 1: opcjonalny rerank streszczeń (tylko jeśli ranker skonfigurowany i flaga włączona)
    if _ranker_enabled() and RANKER_USE_STAGE1 and order_idx:
        try:
            # Zbuduj krótkie passage na podstawie streszczenia (ew. podpisu)
            passages = []
            for i in order_idx:
                did = list(doc_map.keys())[i]
                item = doc_map.get(did, {})
                s = str(item.get("doc_summary") or "")
                sig = item.get("doc_signature") or []
                if isinstance(sig, list):
                    s = (s + "\n\n" + ", ".join(map(str, sig))) if s else ", ".join(map(str, sig))
                passages.append(_truncate_head_tail(s, settings.ranker_max_length))
            score_map = _rerank_indices(q_text, passages, getattr(settings, "rerank_top_n_max", 50))
            # Posortuj ocenione na początku; nieocenione zachowaj w oryginalnej kolejności za nimi
            scored = [(i, score_map[i2]) for i, i2 in enumerate(range(len(order_idx))) if i2 in score_map]
            # scored to (local_pos, score); przemapuj na globalne indeksy
            scored_global = [(order_idx[pos], sc) for (pos, sc) in scored]
            # Build base score map from hybrid_rel for tie-break of unscored items
            base_ids = list(doc_map.keys())
            base_scores_map: Dict[str, float] = {base_ids[i]: float(hybrid_rel[i]) for i in range(len(base_ids))}

            # Apply date tie-break within near-equal reranker scores
            # Map reranked items back to doc_ids and scores
            scored_pairs: List[Tuple[str, float]] = [
                (base_ids[idx], float(sc)) for (idx, sc) in scored_global
            ]
            # Original order for not-scored (fallback to hybrid order)
            not_scored_ids = [base_ids[i] for i in order_idx if i not in {idx for (idx, _) in scored_global}]

            # Prepare epsilon from settings or default
            try:
                eps = float(getattr(settings, "score_tie_epsilon", 1e-9) or 1e-9)
            except Exception:
                eps = 1e-9

            # Tie-break scored part
            scored_ids_in_order = [p[0] for p in sorted(scored_pairs, key=lambda x: x[1], reverse=True)]
            scored_scores = [p[1] for p in sorted(scored_pairs, key=lambda x: x[1], reverse=True)]
            scored_ids_tiebroken = _apply_date_tiebreak_by_score(scored_ids_in_order, scored_scores, doc_map, epsilon=eps)

            # Tie-break not scored part by base hybrid score
            not_scores = [base_scores_map.get(d, 0.0) for d in not_scored_ids]
            not_ids_tiebroken = _apply_date_tiebreak_by_score(not_scored_ids, not_scores, doc_map, epsilon=eps)

            cand_doc_ids = (scored_ids_tiebroken + not_ids_tiebroken)[: min(req.top_m, len(scored_ids_tiebroken) + len(not_ids_tiebroken))]
        except Exception:
            # W przypadku błędu rankera, fallback do bazowego porządku
            base_ids = list(doc_map.keys())
            # Apply tie-break by date on hybrid_rel within epsilon
            ids_in_order = [base_ids[i] for i in order_idx]
            scores_in_order = [float(hybrid_rel[i]) for i in order_idx]
            try:
                eps = float(getattr(settings, "score_tie_epsilon", 1e-9) or 1e-9)
            except Exception:
                eps = 1e-9
            ids_tiebroken = _apply_date_tiebreak_by_score(ids_in_order, scores_in_order, doc_map, epsilon=eps)
            cand_doc_ids = ids_tiebroken[: min(req.top_m, len(ids_tiebroken))]
    else:
        base_ids = list(doc_map.keys())
        ids_in_order = [base_ids[i] for i in order_idx]
        scores_in_order = [float(hybrid_rel[i]) for i in order_idx]
        try:
            eps = float(getattr(settings, "score_tie_epsilon", 1e-9) or 1e-9)
        except Exception:
            eps = 1e-9
        ids_tiebroken = _apply_date_tiebreak_by_score(ids_in_order, scores_in_order, doc_map, epsilon=eps)
        cand_doc_ids = ids_tiebroken[: min(req.top_m, len(ids_tiebroken))]

    return cand_doc_ids, doc_map


# Stage‑2: select candidate chunks and diversify with hybrid MMR.
def _stage2_select_chunks(
    cand_doc_ids: Optional[List[str]],
    q_text: str,
    q_vec: List[float],
    content_sparse_query: Optional[Tuple[List[int], List[float]]],
    doc_map: Dict[str, Any],
    req: Any,
    flt: Optional[qm.Filter] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[float]]:
    """Stage-2: retrieve candidate chunks and apply hybrid MMR diversification.

    Returns (final_hits, mmr_pool, rel2_scores).
    """
    # Build filter for Stage-2 search
    must = [
        qm.FieldCondition(key="point_type", match=qm.MatchValue(value="chunk")),
    ]
    if cand_doc_ids:
        must.insert(0, qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=cand_doc_ids)))
    if flt and getattr(flt, "must", None):
        must = list(flt.must) + must
    flt2 = qm.Filter(
        must=must,
        should=(flt.should if flt and getattr(flt, "should", None) else None),
        must_not=(flt.must_not if flt and getattr(flt, "must_not", None) else None),
    )
    include_fields2 = ["doc_id", "chunk_id", "path", "section_path", "is_active", "doc_url"]
    dense_hits = qdrant.search(
        collection_name=settings.qdrant_content_collection,
        query_vector=(CONTENT_VECTOR_NAME, q_vec),
        query_filter=flt2,
        limit=req.top_m,
        with_payload=(include_fields2 if settings.search_minimal_payload else True),
        with_vectors=[CONTENT_VECTOR_NAME],
        search_params=qm.SearchParams(exact=False, hnsw_ef=128),
    )

    dual = bool(settings.search_dual_query_sparse and req.use_hybrid and SPARSE_ENABLED)
    sparse_hits = []
    if dual and content_sparse_query is not None and content_sparse_query[0]:
        try:
            c_idx, c_val = content_sparse_query
            sparse_hits = qdrant.search(
                collection_name=settings.qdrant_content_collection,
                query_vector=(CONTENT_SPARSE_NAME, qm.SparseVector(indices=c_idx, values=c_val)),
                query_filter=flt2,
                limit=req.top_m,
                with_payload=(include_fields2 if settings.search_minimal_payload else True),
                with_vectors=False,
                search_params=qm.SearchParams(exact=False, hnsw_ef=128),
            )
        except Exception:
            sparse_hits = []

    # Build union pool with fused scores
    pool_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for hit in dense_hits:
        p = hit.payload or {}
        did = p.get("doc_id") or ""
        cid = p.get("chunk_id")
        if did and cid is not None:
            key = (did, int(cid))
            pool_map.setdefault(key, {"dense": 0.0, "sparse": 0.0, "hit": hit, "payload": p})
            pool_map[key]["dense"] = float(hit.score or 0.0)
    for hit in (sparse_hits or []):
        p = hit.payload or {}
        did = p.get("doc_id") or ""
        cid = p.get("chunk_id")
        if did and cid is not None:
            key = (did, int(cid))
            ent = pool_map.setdefault(key, {"dense": 0.0, "sparse": 0.0, "hit": hit, "payload": p})
            ent["sparse"] = float(hit.score or 0.0)

    mmr_pool: List[Dict[str, Any]] = []
    for key, ent in pool_map.items():
        hit = ent.get("hit")
        p = ent.get("payload") or {}
        # Enrich with doc summaries and document-level metadata
        info = doc_map.get(p.get("doc_id", ""), {})
        if info:
            p.setdefault("summary", info.get("doc_summary"))
            if info.get("doc_signature") is not None:
                p.setdefault("signature", info.get("doc_signature"))
            if info.get("doc_entities") is not None:
                p.setdefault("entities", info.get("doc_entities"))
            if info.get("doc_title") is not None:
                p.setdefault("title", info.get("doc_title"))
            if info.get("doc_date") is not None:
                p.setdefault("doc_date", info.get("doc_date"))
            if info.get("doc_url") is not None:
                p.setdefault("doc_url", info.get("doc_url"))
            if p.get("is_active") is None and info.get("is_active") is not None:
                p.setdefault("is_active", info.get("is_active"))
        mmr_pool.append(
            {
                "hit": hit,
                "doc_id": p.get("doc_id", ""),
                "dense_vec": (hit.vector or {}).get(CONTENT_VECTOR_NAME) if hit else [] or [],
                "dense_score": float(ent.get("dense", 0.0)),
                "sparse_score": float(ent.get("sparse", 0.0)),
                "payload": p,
            }
        )

    if not mmr_pool:
        return [], [], []

    dense_scores2 = [x["dense_score"] for x in mmr_pool]
    sparse_scores2 = [x["sparse_score"] for x in mmr_pool]
    dense_norm2 = _normalize(dense_scores2, req.score_norm)
    sparse_norm2 = _normalize(sparse_scores2, req.score_norm)
    rel2 = [req.dense_weight * d + req.sparse_weight * s for d, s in zip(dense_norm2, sparse_norm2)]

    # Entities soft boost at Stage 2
    try:
        strat2 = str(getattr(req, "entity_strategy", "auto") or "auto").strip().lower()
    except Exception:
        strat2 = "auto"
    q_entities2 = set(_query_entities_for(q_text, req))
    if rel2 and q_entities2 and strat2 != "exclude":
        boost2 = float(getattr(settings, "entity_boost_stage2", 0.10) or 0.0)
        for i, item in enumerate(mmr_pool):
            p = item.get("payload") or {}
            ents_raw = p.get("entities") or []
            ents = set(_norm_entities_list(ents_raw if isinstance(ents_raw, list) else []))
            if not ents:
                continue
            inter = ents.intersection(q_entities2)
            if inter:
                scale = min(1.0, len(inter) / 3.0)
                rel2[i] = float(rel2[i]) + boost2 * scale

    # Etap 2: brak reranku na poziomie chunków — reranking odbywa się
    # na poziomie scalonych sekcji w warstwie API po zbudowaniu bloków.

    rep_alpha = req.rep_alpha if req.rep_alpha is not None else req.dense_weight
    dense_vecs2 = [x["dense_vec"] for x in mmr_pool]
    # In dual-query path nie używamy sparse wektorów w MMR
    sparse_vecs2 = [([], []) for _ in mmr_pool]
    doc_ids2 = [x["doc_id"] for x in mmr_pool]

    sel_idx = mmr_diversify_hybrid(
        dense_vecs2,
        sparse_vecs2,
        rel2,
        min(req.top_k, len(mmr_pool)),
        req.mmr_lambda,
        rep_alpha,
        per_doc_ids=doc_ids2,
        per_doc_limit=max(1, int(req.per_doc_limit)) if req.per_doc_limit and req.per_doc_limit > 0 else None,
    )

    selected = [mmr_pool[i] for i in sel_idx]
    final_hits: List[Dict[str, Any]] = []
    # If Stage-1 was skipped, enrich with summaries so that summary_mode and shaping behave as before
    if not doc_map:
        try:
            missing_ids = sorted({x.get("doc_id", "") for x in selected if x.get("doc_id")})
            if missing_ids:
                doc_map.update(_fetch_doc_summaries(missing_ids))
        except Exception:
            pass
    for idx, item in zip(sel_idx, selected):
        hit = item["hit"]
        payload = hit.payload or {}
        doc_info = doc_map.get(payload.get("doc_id", ""), {})
        if doc_info:
            payload.setdefault("summary", doc_info.get("doc_summary"))
            if doc_info.get("doc_signature") is not None:
                payload.setdefault("signature", doc_info.get("doc_signature"))
            if doc_info.get("doc_entities") is not None:
                payload.setdefault("entities", doc_info.get("doc_entities"))
            if doc_info.get("doc_title") is not None:
                payload.setdefault("title", doc_info.get("doc_title"))
            if doc_info.get("doc_date") is not None:
                payload.setdefault("doc_date", doc_info.get("doc_date"))
            if doc_info.get("doc_url") is not None:
                payload.setdefault("doc_url", doc_info.get("doc_url"))
            if payload.get("is_active") is None and doc_info.get("is_active") is not None:
                payload.setdefault("is_active", doc_info.get("is_active"))
        final_score = float(rel2[idx])
        final_hits.append({"hit": hit, "score": float(final_score), "payload": payload})
    final_hits.sort(key=lambda x: x["score"], reverse=True)
    return final_hits, mmr_pool, rel2


# Heuristically classify mode if 'auto' (current/archival/all).
def _classify_mode(query: str, mode: str) -> str:
    """Heuristically classify mode when `mode` is 'auto'.

    Policy: prefer "current" by default. Switch to "archival" only when the query
    clearly points to historical/archival context (keywords like 'archiwal*',
    'stara', explicit years, 'wersja z ...'). Use "all" only on explicit cues
    ('wszystkie', 'cała historia', 'pełen zakres').
    """
    if mode != "auto":
        return mode
    q = (query or "").lower()
    # Archival cues: explicit year, 'archiwal', 'stara', 'wersja z'
    if re.search(r"archiw|stara|wersja\s+z|\b(19|20)\d{2}\b", q):
        return "archival"
    # All-history cues
    if re.search(r"\bwszystkie\b|cała\s+histori|pełen\s+zakres|od\s+zawsze", q):
        return "all"
    # Default: current
    if re.search(r"obowiązując|aktualn|teraz|bieżąc", q):
        return "current"
    return "current"


# Classic MMR over dense vectors: return selected indices.
def mmr_diversify(vectors: np.ndarray, scores: np.ndarray, k: int, lam: float = DEFAULT_MMR_LAMBDA) -> List[int]:
    """Classic MMR over dense vectors; returns indices of selected items."""
    selected: List[int] = []
    candidates = list(range(len(scores)))
    if len(candidates) <= k:
        return candidates
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    def sims(a, b) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / (denom + 1e-9))

    while len(selected) < k and candidates:
        best_i = None
        best_score = -1e9
        for i in candidates:
            rep = 0.0
            for j in selected:
                rep = max(rep, sims(vectors[i], vectors[j]))
            mmr = lam * float(scores[i]) - (1 - lam) * rep
            if mmr > best_score:
                best_score = mmr
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        candidates.remove(best_i)
    return selected


# Hybrid MMR over dense+sparse representations with optional per-doc limit.
def mmr_diversify_hybrid(
    dense_vecs: List[List[float]],
    sparse_vecs: List[Tuple[List[int], List[float]]],
    rel_scores: List[float],
    k: int,
    lam: float,
    rep_alpha: float,
    per_doc_ids: Optional[List[str]] = None,
    per_doc_limit: Optional[int] = None,
) -> List[int]:
    """MMR over hybrid representations with optional per-doc limit enforcement."""
    n = len(rel_scores)
    candidates = list(range(n))
    if n <= k:
        return candidates
    selected: List[int] = []
    counts: Dict[str, int] = {}

    def allowed(i: int) -> bool:
        if per_doc_ids is None or per_doc_limit is None:
            return True
        did = per_doc_ids[i]
        return counts.get(did, 0) < per_doc_limit

    while len(selected) < k and candidates:
        best_i = None
        best_score = -1e12
        for i in candidates:
            if per_doc_limit is not None and per_doc_ids is not None:
                if not allowed(i):
                    continue
            rep = 0.0
            for j in selected:
                d_sim = _cosine_dense(dense_vecs[i], dense_vecs[j])
                s_sim = _sparse_pair_cos(
                    sparse_vecs[i][0], sparse_vecs[i][1], sparse_vecs[j][0], sparse_vecs[j][1]
                )
                rep = max(rep, rep_alpha * d_sim + (1.0 - rep_alpha) * s_sim)
            mmr = lam * float(rel_scores[i]) - (1.0 - lam) * rep
            if mmr > best_score:
                best_score = mmr
                best_i = i
        if best_i is None:
            break
        if per_doc_limit is not None and per_doc_ids is not None and not allowed(best_i):
            alt = None
            alt_score = -1e12
            for i in candidates:
                if allowed(i):
                    rep = 0.0
                    for j in selected:
                        d_sim = _cosine_dense(dense_vecs[i], dense_vecs[j])
                        s_sim = _sparse_pair_cos(
                            sparse_vecs[i][0], sparse_vecs[i][1], sparse_vecs[j][0], sparse_vecs[j][1]
                        )
                        rep = max(rep, rep_alpha * d_sim + (1.0 - rep_alpha) * s_sim)
                    mmr = lam * float(rel_scores[i]) - (1.0 - lam) * rep
                    if mmr > alt_score:
                        alt_score = mmr
                        alt = i
            if alt is not None:
                best_i = alt
        selected.append(best_i)
        candidates.remove(best_i)
        if per_doc_limit is not None and per_doc_ids is not None:
            did = per_doc_ids[best_i]
            counts[did] = counts.get(did, 0) + 1
    return selected


def _fetch_doc_chunks(doc_id: str) -> List[Dict[str, Any]]:
    """Pobierz wszystkie chunki dokumentu."""
    if not doc_id:
        return []
    try:
        flt = qm.Filter(
            must=[
                qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
                qm.FieldCondition(key="point_type", match=qm.MatchValue(value="chunk")),
            ]
        )
        out: List[Dict[str, Any]] = []
        offset = None
        while True:
            res = qdrant.scroll(
                collection_name=settings.qdrant_content_collection,
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
                if payload.get("point_type") and payload.get("point_type") != "chunk":
                    continue
                out.append(payload)
            if offset is None:
                break
        out.sort(key=lambda p: int(p.get("chunk_id", 0)))
        return out
    except Exception:
        return []


# Fetch chunk payloads for a given section (optionally include descendants).
def _fetch_section_chunks(
    doc_id: str,
    section: Optional[str],
    *,
    include_descendants: bool = False,
    chunk_cache: Optional[Dict[Tuple[str, str], List[Dict[str, Any]]]] = None,
    merge_level: str = "ust",
) -> List[Dict[str, Any]]:
    """Fetch raw chunk payloads for a given section (optionally descendants)."""
    if not doc_id or not section:
        return []

    section_norm = _normalize_whitespace(section)
    if not section_norm:
        return []

    cache_key: Optional[Tuple[str, str]] = None
    if chunk_cache is not None:
        cache_key = (doc_id, section_norm)
        cached = chunk_cache.get(cache_key)
        if cached is not None:
            return cached

    must: List[qm.Condition] = [
        qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
        qm.FieldCondition(key="point_type", match=qm.MatchValue(value="chunk")),
    ]
    if include_descendants:
        # Każdy chunk przechowuje ścieżkę wraz z prefiksami, więc wystarczy dopasować sekcję.
        must.append(
            qm.FieldCondition(
                key="section_path_prefixes",
                match=qm.MatchValue(value=section_norm),
            )
        )
    else:
        must.append(
            qm.FieldCondition(
                key="section_path",
                match=qm.MatchValue(value=section_norm),
            )
        )

    flt = qm.Filter(must=must)
    out: List[Dict[str, Any]] = []
    offset = None
    try:
        while True:
            res = qdrant.scroll(
                collection_name=settings.qdrant_content_collection,
                scroll_filter=flt,
                limit=256,
                offset=offset,
                with_payload=["text", "chunk_id"],
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
                if payload.get("point_type") and payload.get("point_type") != "chunk":
                    continue
                out.append(payload)
            if offset is None:
                break
    except Exception:
        return []

    out.sort(key=lambda p: int(p.get("chunk_id", 0)))
    if chunk_cache is not None and cache_key is not None:
        chunk_cache[cache_key] = out
    return out


# Batch-fetch chunks for multiple sections of a document in one scroll.
def _fetch_sections_chunks_batch(
    doc_id: str,
    sections: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch chunks for many sections of a single document with one scroll.

    Returns a map: normalized_section -> list of chunk payloads (sorted).
    Sections include descendants via prefix matching.
    """
    result: Dict[str, List[Dict[str, Any]]] = {}
    if not doc_id or not sections:
        return result
    labels = [s for s in {(_normalize_whitespace(s) or "") for s in sections} if s]
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
                # Przydziel chunk do wszystkich sekcji, których jest prefiksem
                for lab in labels:
                    if lab in prefixes:
                        result.setdefault(lab, []).append(payload)
            if offset is None:
                break
    except Exception:
        return result
    # Posortuj listy
    for lab, lst in result.items():
        lst.sort(key=lambda p: int(p.get("chunk_id", 0)))
    return result


# Build merged section blocks from chunk-level hits.
def _build_blocks_from_hits(
    final_hits: List[Dict[str, Any]],
    summary_mode: str = "first",
) -> List[Dict[str, Any]]:
    """Build merged evidence blocks by full sections.

    - Group hits by (doc_id, section)
    - For named sections, fetch full section (and descendants) from Qdrant
      and concatenate text in order; otherwise use hit chunks only
    - Block score = max(score) among member hits
    - `summary_mode` controls document summary duplication
    """
    merge_level_raw = getattr(settings, "section_merge_level", "ust")
    merge_level = str(merge_level_raw).strip().lower() if merge_level_raw else "ust"
    if merge_level not in SECTION_LEVEL_INDEX:
        merge_level = "ust"

    by_key: Dict[Tuple[str, Optional[str]], List[Dict[str, Any]]] = {}
    for fh in final_hits:
        payload = fh.get("payload") or {}
        did = payload.get("doc_id")
        if not did:
            continue
        section_path = payload.get("section_path")
        canonical = _canonical_section_label(section_path, merge_level)
        fallback = _normalize_whitespace(section_path) if isinstance(section_path, str) else None
        group_label = canonical or (fallback if fallback else None)
        key = (str(did), group_label)
        by_key.setdefault(key, []).append(fh)

    blocks: List[Dict[str, Any]] = []
    seen_docs: set = set()
    doc_chunk_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    # Batchuj pobranie sekcji per dokument (gdy włączone)
    if settings.batch_section_fetch:
        labels_by_doc: Dict[str, List[str]] = {}
        for (did, section) in by_key.keys():
            if isinstance(section, str) and section:
                labels_by_doc.setdefault(did, []).append(section)
        for did, labels in labels_by_doc.items():
            mapping = _fetch_sections_chunks_batch(did, labels)
            for lab, chunks in mapping.items():
                doc_chunk_cache[(did, lab)] = chunks

    for (did, section), hits in by_key.items():
        # Bazowy payload do metadanych
        base_payload = (hits[0].get("payload") or {})
        path = base_payload.get("path", "")
        title = base_payload.get("title")
        doc_date = base_payload.get("doc_date")
        is_active = base_payload.get("is_active")
        doc_url = base_payload.get("doc_url")
        # Score sekcji = max score spośród jej trafień
        sect_score = max(float(h.get("score", 0.0)) for h in hits)

        # Dociągnij pełną sekcję (jeśli mamy etykietę) i zbuduj tekst + zakres id
        merged_text_parts: List[str] = []
        first_cid: Optional[int] = None
        last_cid: Optional[int] = None

        base_section_path = base_payload.get("section_path")
        canonical_label = _canonical_section_label(base_section_path, merge_level)
        fallback_label = section if isinstance(section, str) else None
        if fallback_label:
            fallback_label = _normalize_whitespace(fallback_label) or None
        normalized_label = canonical_label or fallback_label
        include_descendants = bool(normalized_label and str(normalized_label).strip())
        if normalized_label:
            # Użyj cache batchowego (gdy aktywny); fallback na pojedynczy fetch jeśli brak
            sec_chunks = doc_chunk_cache.get((did, normalized_label)) if settings.batch_section_fetch else None
            if sec_chunks is None:
                sec_chunks = _fetch_section_chunks(
                    did,
                    normalized_label,
                    include_descendants=include_descendants,
                    merge_level=merge_level,
                )
        else:
            sec_chunks = []
        if sec_chunks:
            for ch in sec_chunks:
                text = (ch.get("text") or "").strip()
                if text:
                    merged_text_parts.append(text)
                cid = int(ch.get("chunk_id", 0))
                first_cid = cid if first_cid is None else min(first_cid, cid)
                last_cid = cid if last_cid is None else max(last_cid, cid)
        else:
            # Brak sekcji lub nie udało się pobrać — użyj tylko trafionych chunków
            for h in sorted(hits, key=lambda x: int((x.get("payload") or {}).get("chunk_id", 0))):
                hp = h.get("payload") or {}
                text = (hp.get("text") or "").strip()
                if text:
                    merged_text_parts.append(text)
                cid = int(hp.get("chunk_id", 0))
                first_cid = cid if first_cid is None else min(first_cid, cid)
                last_cid = cid if last_cid is None else max(last_cid, cid)

        merged_text = "\n\n".join(merged_text_parts).strip()
        # token_estimate removed from the API (no longer returned)

        # Summary kontrolowane per dokument
        doc_summary_val = base_payload.get("summary")
        if summary_mode == "none":
            sum_val = None
        elif summary_mode == "first":
            if did in seen_docs:
                sum_val = None
            else:
                sum_val = doc_summary_val
                seen_docs.add(did)
        else:
            sum_val = doc_summary_val

        blocks.append(
            {
                "doc_id": did,
                "path": path,
                "title": title,
                "doc_date": doc_date,
                "is_active": is_active,
                "section": normalized_label,
                "first_chunk_id": int(first_cid if first_cid is not None else (base_payload.get("chunk_id", 0))),
                "last_chunk_id": int(last_cid if last_cid is not None else (base_payload.get("chunk_id", 0))),
                "score": float(sect_score),
                "summary": sum_val,
                "doc_url": doc_url,
                "text": merged_text,
            }
        )

    blocks.sort(key=lambda b: float(b.get("score", 0.0)), reverse=True)
    return blocks


# Shape final results into the requested format (flat/grouped/blocks).
def _shape_results(
    final_hits: List[Dict[str, Any]],
    doc_map: Dict[str, Any],
    mmr_pool: List[Dict[str, Any]],
    rel2: List[float],
    req: Any,
) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
    """Transform final hits into the requested response shape.

    Returns (hits, groups, blocks) where only one of groups/blocks is present
    depending on `req.result_format`.
    """
    if req.result_format == "blocks":
        blocks_payload = _build_blocks_from_hits(final_hits, summary_mode=req.summary_mode)
        return [], None, blocks_payload

    results: List[Dict[str, Any]] = []
    groups_payload: Optional[List[Dict[str, Any]]] = None
    # Ensure blocks_payload is always defined for non-"blocks" formats
    blocks_payload: Optional[List[Dict[str, Any]]] = None

    if req.result_format == "grouped":
        groups: Dict[str, Dict[str, Any]] = {}
        for fh in final_hits:
            payload = fh.get("payload") or {}
            did = payload.get("doc_id", "")
            if not did:
                continue
            grp = groups.get(did)
            if grp is None:
                grp = {
                    "doc_id": did,
                    "path": payload.get("path", ""),
                    "title": payload.get("title"),
                    "doc_date": payload.get("doc_date"),
                    "is_active": payload.get("is_active"),
                    "doc_url": payload.get("doc_url"),
                    "summary": None if req.summary_mode == "none" else payload.get("summary"),
                    "score": float(fh.get("score", 0.0)),
                    "chunks": [],
                }
                groups[did] = grp
            else:
                if float(fh.get("score", 0.0)) > float(grp.get("score", 0.0)):
                    grp["score"] = float(fh.get("score", 0.0))
            snippet = (payload.get("text") or "").strip()[:500]
            grp["chunks"].append({
                "chunk_id": payload.get("chunk_id", 0),
                "score": float(fh.get("score", 0.0)),
                "snippet": snippet,
            })
        groups_list = list(groups.values())
        for g in groups_list:
            g["chunks"].sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        groups_list.sort(key=lambda g: float(g.get("score", 0.0)), reverse=True)
        groups_payload = groups_list
        results = []
    else:
        seen_docs: set = set()
        for fh in final_hits:
            payload = fh.get("payload") or {}
            did = payload.get("doc_id", "")
            if req.summary_mode == "none":
                summary_val: Optional[str] = None
            elif req.summary_mode == "first":
                summary_val = None if did in seen_docs else payload.get("summary")
                if did not in seen_docs:
                    seen_docs.add(did)
            else:
                summary_val = payload.get("summary")
            results.append(
                {
                    "doc_id": did,
                    "path": payload.get("path", ""),
                    "title": payload.get("title"),
                    "doc_date": payload.get("doc_date"),
                    "is_active": payload.get("is_active"),
                    "doc_url": payload.get("doc_url"),
                    "section": payload.get("section_path"),
                    "chunk_id": payload.get("chunk_id", 0),
                    "score": float(fh.get("score", 0.0)),
                    "snippet": (payload.get("text") or "").strip()[:500] if payload.get("text") else (payload.get("summary", "")[:500]),
                    "summary": summary_val,
                }
            )

    return results, groups_payload, blocks_payload

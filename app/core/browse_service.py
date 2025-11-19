"""Lightweight browse/analytics over the corpus (LLM-friendly).

Provides simple operations without MMR/rerank/shaping, such as:
- count how many candidate documents match a query (content scope),
- list candidate doc ids and basic metadata.

Implementation note:
- Selection operates on document content (chunk-level points) and does not
  search through summaries. Summaries may be fetched only to enrich metadata
  (title/doc_date) after candidate selection.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from qdrant_client.http import models as qm

from app.core.embedding import embed_query
from app.core.search import (
    _build_sparse_queries_for_query,
    _classify_mode,
    _norm_entities_list,
)
from app.core.constants import CONTENT_VECTOR_NAME, CONTENT_SPARSE_NAME, SPARSE_ENABLED
from app.qdrant_utils import qdrant
from app.core.store_access import fetch_doc_summaries
from app.core.doc_kind import infer_doc_kind
from app.core.fts import fts_search_doc_ids, fts_search_doc_count, fts_search_doc_ids_all
from app.settings import get_settings

settings = get_settings()


class BrowseParams:
    """Input parameters for browse operations."""

    def __init__(
        self,
        queries: List[str],
        *,
        top_m: int = 100,
        use_hybrid: bool = True,
        mode: str = "auto",
        kinds: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        entity_strategy: str = "auto",  # auto|must_any|must_all|exclude|optional
        status: str = "active",  # active|inactive|all
        text_match: str = "none",  # none|phrase|any|all
    ) -> None:
        self.queries = [str(q).strip() for q in queries if str(q).strip()]
        self.top_m = int(top_m)
        self.use_hybrid = bool(use_hybrid)
        self.mode = str(mode or "auto")
        # Accept ASCII kind identifiers; normalize to lowercase, de-duplicate
        norm: List[str] = []
        seen: Set[str] = set()
        for k in (kinds or []):
            s = str(k).strip().lower()
            if s and s not in seen:
                seen.add(s)
                norm.append(s)
        self.kinds = norm or None
        # Entities: preserve raw tokens, but also include lowercase variants for matching.
        # Qdrant KEYWORD matching is case-sensitive, and stored payloads may keep original case.
        # To improve recall without schema changes, we match against both raw and casefolded forms.
        raw_list = [str(v or "").strip() for v in (entities or []) if str(v or "").strip()]
        # preserve order while deduping
        seen_raw: Set[str] = set()
        ordered_raw: List[str] = []
        for v in raw_list:
            if v not in seen_raw:
                seen_raw.add(v)
                ordered_raw.append(v)
        # build filter set = raw ∪ casefold(raw)
        seen_all: Set[str] = set()
        ents_all: List[str] = []
        for v in ordered_raw:
            for cand in (v, v.casefold()):
                if cand not in seen_all and cand:
                    seen_all.add(cand)
                    ents_all.append(cand)
        self.entities_raw = ordered_raw
        self.entities_filter = ents_all
        strat = str(entity_strategy or "auto").strip().lower()
        if strat not in {"auto", "must_any", "must_all", "exclude", "optional"}:
            strat = "auto"
        self.entity_strategy = strat
        s = str(status or "active").strip().lower()
        if s not in {"active", "inactive", "all"}:
            s = "active"
        self.status = s
        tm = str(text_match or "none").strip().lower()
        if tm not in {"none", "phrase", "any", "all"}:
            tm = "none"
        self.text_match = tm

    @staticmethod
    def _text_match_ok(q_text: str, chunk_text: Optional[str], mode: str) -> bool:
        if mode == "none":
            return True
        t = (chunk_text or "").casefold()
        q = (q_text or "").casefold().strip()
        if not q:
            return True
        if mode == "phrase":
            return q in t
        # Token modes
        import re
        def toks(s: str) -> List[str]:
            return re.findall(r"\w+", s)
        q_tokens = toks(q)
        if not q_tokens:
            return True
        t_set = set(toks(t))
        if mode == "any":
            return any(tok in t_set for tok in q_tokens)
        if mode == "all":
            return all(tok in t_set for tok in q_tokens)
        return True


def _unified_mode_and_filter(queries: List[str], mode: str) -> Tuple[str, Optional[qm.Filter]]:
    """Return unified mode and filter for Stage‑1 based on queries and requested mode."""
    if mode != "auto":
        m = mode
    else:
        modes = {_classify_mode(q, "auto") for q in queries}
        if modes == {"current"}:
            m = "current"
        elif modes == {"archival"}:
            m = "archival"
        else:
            m = "all"
    if m == "current":
        flt = qm.Filter(must=[qm.FieldCondition(key="is_active", match=qm.MatchValue(value=True))])
    elif m == "archival":
        flt = qm.Filter(must=[qm.FieldCondition(key="is_active", match=qm.MatchValue(value=False))])
    else:
        flt = None
    return m, flt


def stage1_candidates(params: BrowseParams) -> Tuple[Set[str], Dict[str, Dict[str, Any]], bool]:
    """Return union of candidate document ids across queries (content-based).

    Selection is performed over chunk-level content vectors (dense + optional
    TF-IDF). Summaries are not used for searching. Returns (doc_ids, doc_map,
    approx) where `approx` is True when any per-query unique doc_id list hit the
    `top_m` cap.
    """
    if not params.queries:
        return set(), {}, False

    # Status filter has priority over heuristic mode; default is 'active'
    if params.status == "active":
        flt = qm.Filter(must=[qm.FieldCondition(key="is_active", match=qm.MatchValue(value=True))])
    elif params.status == "inactive":
        flt = qm.Filter(must=[qm.FieldCondition(key="is_active", match=qm.MatchValue(value=False))])
    else:
        # Fallback to heuristic mode when 'all'
        _, flt = _unified_mode_and_filter(params.queries, params.mode)
    approx = False
    union: Set[str] = set()
    merged_map: Dict[str, Dict[str, Any]] = {}

    # Base filter: point_type == chunk plus optional is_active condition
    base_must = [qm.FieldCondition(key="point_type", match=qm.MatchValue(value="chunk"))]
    base_must_not: List[qm.Condition] = []
    if flt and getattr(flt, "must", None):
        base_must = list(flt.must) + base_must
    if flt and getattr(flt, "must_not", None):
        base_must_not = list(flt.must_not)

    # Build entity conditions separately
    ents = getattr(params, "entities_filter", None) or []
    ent_must: List[qm.Condition] = []
    ent_must_not: List[qm.Condition] = []
    if ents:
        if params.entity_strategy in {"auto", "must_any", "optional"}:
            ent_must.append(qm.FieldCondition(key="entities", match=qm.MatchAny(any=ents)))
        elif params.entity_strategy == "must_all":
            for e in ents:
                ent_must.append(qm.FieldCondition(key="entities", match=qm.MatchValue(value=e)))
        elif params.entity_strategy == "exclude":
            ent_must_not.append(qm.FieldCondition(key="entities", match=qm.MatchAny(any=ents)))

    # For vector search, include entity conditions only for strict strategies
    strict_strategies = {"auto", "must_any", "must_all", "exclude"}
    include_entities_in_vector = params.entity_strategy in strict_strategies and params.entity_strategy != "optional"

    flt_chunks_for_vector = qm.Filter(
        must=(base_must + (ent_must if include_entities_in_vector else [])),
        must_not=(base_must_not + ent_must_not) or None,
    )
    # Entities-only filter (used for scroll fallback or union mode)
    flt_entities_only = qm.Filter(
        must=(base_must + ent_must) or None,
        must_not=(base_must_not + ent_must_not) or None,
    )

    for q in params.queries:
        q_vec = embed_query([q])[0]
        content_sparse_query, _ = _build_sparse_queries_for_query(q, params.use_hybrid)

        per_query_doc_ids: List[str] = []
        seen_local: Set[str] = set()

        # Dense search over content
        # Dynamic search limit: oversample more when text_match is enabled
        factor = int(getattr(settings, "browse_search_oversample", 10) if params.text_match != "none" else 3)
        factor = max(1, factor)
        search_limit = max(50, int(params.top_m) * factor)
        max_cap = int(getattr(settings, "browse_search_limit_max", 4000))
        if max_cap > 0:
            search_limit = min(search_limit, max_cap)
        try:
            dense_hits = qdrant.search(
                collection_name=settings.qdrant_content_collection,
                query_vector=(CONTENT_VECTOR_NAME, q_vec),
                query_filter=flt_chunks_for_vector,
                limit=search_limit,
                with_payload=["doc_id", "is_active"] + (["text"] if params.text_match != "none" else []),
                with_vectors=False,
                search_params=qm.SearchParams(exact=False, hnsw_ef=128),
            )
        except Exception:
            dense_hits = []

        for hit in dense_hits:
            payload = hit.payload or {}
            did = str(payload.get("doc_id") or "")
            if not did or did in seen_local:
                continue
            if params.text_match != "none":
                if not BrowseParams._text_match_ok(q, payload.get("text"), params.text_match):
                    continue
            seen_local.add(did)
            per_query_doc_ids.append(did)
            # Record minimal metadata that we can derive without summaries
            merged_map.setdefault(did, {})
            if merged_map[did].get("is_active") is None and payload.get("is_active") is not None:
                merged_map[did]["is_active"] = bool(payload.get("is_active"))
            if len(per_query_doc_ids) >= params.top_m:
                break

        # Optional TF-IDF sparse over content to supplement recall
        if len(per_query_doc_ids) < params.top_m and params.use_hybrid and SPARSE_ENABLED and content_sparse_query is not None:
            try:
                c_idx, c_val = content_sparse_query
                sparse_hits = qdrant.search(
                    collection_name=settings.qdrant_content_collection,
                    query_vector=(CONTENT_SPARSE_NAME, qm.SparseVector(indices=c_idx, values=c_val)),
                    query_filter=flt_chunks_for_vector,
                    limit=search_limit,
                    with_payload=["doc_id", "is_active"] + (["text"] if params.text_match != "none" else []),
                    with_vectors=False,
                    search_params=qm.SearchParams(exact=False, hnsw_ef=128),
                )
            except Exception:
                sparse_hits = []
            for hit in sparse_hits:
                payload = hit.payload or {}
                did = str(payload.get("doc_id") or "")
                if not did or did in seen_local:
                    continue
                if params.text_match != "none":
                    if not BrowseParams._text_match_ok(q, payload.get("text"), params.text_match):
                        continue
                seen_local.add(did)
                per_query_doc_ids.append(did)
                merged_map.setdefault(did, {})
                if merged_map[did].get("is_active") is None and payload.get("is_active") is not None:
                    merged_map[did]["is_active"] = bool(payload.get("is_active"))
                if len(per_query_doc_ids) >= params.top_m:
                    break

        # Entities-only fallback: if entities filter is present and recall is low,
        # scan (scroll) matching chunks without relying on vector similarity.
        # Entities-only fallback / union: if strategy is 'optional' treat entities as OR condition
        if len(per_query_doc_ids) < params.top_m and (params.entities_filter or []):
            try:
                offset = None
                while len(per_query_doc_ids) < params.top_m:
                    res = qdrant.scroll(
                        collection_name=settings.qdrant_content_collection,
                        scroll_filter=flt_entities_only,
                        limit=256,
                        offset=offset,
                        with_payload=["doc_id", "is_active"],
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
                        did = str(payload.get("doc_id") or "")
                        if not did or did in seen_local:
                            continue
                        seen_local.add(did)
                        per_query_doc_ids.append(did)
                        merged_map.setdefault(did, {})
                        if merged_map[did].get("is_active") is None and payload.get("is_active") is not None:
                            merged_map[did]["is_active"] = bool(payload.get("is_active"))
                        if len(per_query_doc_ids) >= params.top_m:
                            break
                    if offset is None:
                        break
            except Exception:
                pass

        if len(per_query_doc_ids) >= params.top_m:
            approx = True

        for did in per_query_doc_ids:
            union.add(did)

    return union, merged_map, approx


def _ensure_doc_meta_with_kind(ids: Iterable[str], doc_map: Dict[str, Dict[str, Any]]) -> None:
    """Populate doc_map with doc_title/doc_date/is_active and inferred doc_kind for ids."""
    id_list = list(ids)
    if not id_list:
        return
    # Fetch summaries for any id that doesn't yet have summary-derived fields
    # Important: stage1_candidates() may have inserted a placeholder entry
    # containing only is_active from chunk payload. We must still fetch
    # summaries for such entries to obtain title/date/signature.
    missing = [
        d for d in id_list
        if not isinstance(doc_map.get(d), dict)
        or "doc_title" not in (doc_map.get(d) or {})
    ]
    if missing:
        doc_map.update(fetch_doc_summaries(missing))
    for did in id_list:
        meta = doc_map.get(did) or {}
        if meta.get("doc_kind"):
            continue
        title = (meta.get("doc_title") or "")
        sig = meta.get("doc_signature") or meta.get("signature") or []
        if not isinstance(sig, list):
            sig = []
        meta["doc_kind"] = infer_doc_kind(title, sig)
        doc_map[did] = meta


def _apply_kind_filter(ids: Iterable[str], doc_map: Dict[str, Dict[str, Any]], kinds: Optional[List[str]]) -> List[str]:
    """Filter ids by doc_kind; returns ordered list preserving input order."""
    if not kinds:
        return list(ids)
    allowed = {str(k).strip().lower() for k in kinds if str(k).strip()}
    out: List[str] = []
    for did in ids:
        kind = (doc_map.get(did, {}) or {}).get("doc_kind") or "other"
        if str(kind).lower() in allowed:
            out.append(did)
    return out


def list_document_minimal(params: BrowseParams, limit: int = 200) -> Tuple[List[Dict[str, Any]], bool, int]:
    """List up to `limit` candidate documents with minimal metadata.

    Returns (docs, approx, candidates_total). Ordering by doc_date DESC (then doc_id).
    """
    ids, doc_map, approx = stage1_candidates(params)
    # Enrich with summaries (title/date) and infer kind
    _ensure_doc_meta_with_kind(ids, doc_map)
    # Apply kind filter post-selection
    selected = _apply_kind_filter(sorted(ids), doc_map, params.kinds)
    # Order by doc_date DESC before applying limit
    def _date_ord_from_map(did: str) -> int:
        s = str((doc_map.get(did, {}) or {}).get("doc_date") or "").strip()
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
            return y * 10000 + m * 100 + d
        except Exception:
            return 0
    selected.sort(key=lambda did: (_date_ord_from_map(did), did), reverse=True)
    candidates_total = len(selected)
    if limit and len(selected) > limit:
        approx = True
        selected = selected[:limit]
    docs = [
        {
            "doc_id": did,
            "title": (doc_map.get(did, {}) or {}).get("doc_title"),
            "doc_date": (doc_map.get(did, {}) or {}).get("doc_date"),
            "is_active": (doc_map.get(did, {}) or {}).get("is_active"),
            "doc_kind": (doc_map.get(did, {}) or {}).get("doc_kind"),
            "doc_url": (doc_map.get(did, {}) or {}).get("doc_url"),
        }
        for did in selected
    ]
    # Docs already follow selected order by date desc; no further sort needed
    return docs, approx, candidates_total


"""Facets aggregation removed in 2.43.0. Use client-side grouping over /browse/doc-ids results."""


# --- Simplified FTS-based doc-ids (full corpus) ---

def list_doc_ids_via_fts(
    queries: List[str],
    *,
    match: str,
    status: str,
    kinds: Optional[List[str]],
    limit: int,
) -> Tuple[List[Dict[str, Any]], bool, int]:
    m = str(match or "phrase").lower()
    st = str(status or "active").lower()
    # When limit <= 0, compute candidates_total and optionally return a small sample.
    # Sample is returned only when any filter is set (query non-empty OR kinds specified OR status != 'active').
    if int(limit) <= 0:
        total = int(fts_search_doc_count(queries, match=m, status=st, kinds=kinds))
        # Treat sample-worthy narrowing as presence of query text or kinds filter.
        # Status alone (including 'all' or 'inactive') does not trigger sampling to avoid large payloads
        # for generic count questions.
        has_narrowing = bool(queries) or (kinds and len(kinds) > 0)
        if not has_narrowing or total <= 0:
            return [], False, total
        sample_n = min(15, total)
        ids = fts_search_doc_ids(
            queries,
            match=m,
            status=st,
            limit=sample_n,
            kinds=kinds,
            order="date_desc",
        )
        doc_map = fetch_doc_summaries(ids)
        # Infer kind if missing (should be rare now that FTS stores doc_kind, but summaries may not have it)
        for did in ids:
            meta = doc_map.get(did, {})
            if "doc_kind" not in meta:
                title = meta.get("doc_title") or ""
                meta["doc_kind"] = infer_doc_kind(title, [])
                doc_map[did] = meta
        docs = [
            {
                "doc_id": did,
                "title": (doc_map.get(did, {}) or {}).get("doc_title"),
                "doc_date": (doc_map.get(did, {}) or {}).get("doc_date"),
                "is_active": (doc_map.get(did, {}) or {}).get("is_active"),
                "doc_kind": (doc_map.get(did, {}) or {}).get("doc_kind"),
                "doc_url": (doc_map.get(did, {}) or {}).get("doc_url"),
            }
            for did in ids
        ]
        # approx=True when sample does not cover all candidates
        return docs, (total > sample_n), total

    ids = fts_search_doc_ids(
        queries,
        match=m,
        status=st,
        limit=max(1, int(limit)),
        kinds=kinds,
        order="date_desc",
    )
    # Enrich summaries and infer kind
    doc_map = fetch_doc_summaries(ids)
    # Attach inferred kind and filter
    for did in ids:
        meta = doc_map.get(did, {})
        if "doc_kind" not in meta:
            title = meta.get("doc_title") or ""
            meta["doc_kind"] = infer_doc_kind(title, [])
            doc_map[did] = meta
    # kinds filter already applied via FTS; no need to filter again
    candidates_total = len(ids)
    approx = False
    if limit and len(ids) > int(limit):
        approx = True
        ids = ids[: int(limit)]
    docs = [
        {
            "doc_id": did,
            "title": (doc_map.get(did, {}) or {}).get("doc_title"),
            "doc_date": (doc_map.get(did, {}) or {}).get("doc_date"),
            "is_active": (doc_map.get(did, {}) or {}).get("is_active"),
            "doc_kind": (doc_map.get(did, {}) or {}).get("doc_kind"),
            "doc_url": (doc_map.get(did, {}) or {}).get("doc_url"),
        }
        for did in ids
    ]
    # Preserve FTS order (date DESC). Do not resort by title.
    return docs, approx, candidates_total

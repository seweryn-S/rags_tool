"""Admin UI and step-by-step debug endpoints, isolated from core API.

This module attaches all non-functional (UI/debug) routes to a provided FastAPI app.
It keeps the core/business endpoints in app/api.py clean and separate.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin
import inspect

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel
from qdrant_client.http import models as qm

from app.core.embedding import embed_query
from app.core.search import (
    _build_sparse_queries_for_query,
    _classify_mode,
    _stage1_select_documents,
    _stage2_select_chunks,
    _shape_results,
)
from app.models import (
    DebugMultiEmbedRequest,
    DebugMultiStage1Request,
    DebugMultiStage2Request,
    DebugMultiShapeRequest,
)
from app.settings import get_settings


settings = get_settings()
logger = logging.getLogger("rags_tool")


ADMIN_OPERATION_SPECS: List[Dict[str, Any]] = [
    # Statystyki i indeks
    {"id": "docs-stats", "path": "/docs/stats", "method": "GET", "label": "Statystyka dokumentów (FTS)"},
    {"id": "fts-rebuild", "path": "/fts/rebuild", "method": "POST", "label": "Odbuduj FTS (chunki → SQLite)", "body": "{}"},
    # Przeglądanie korpusu (wybór dokumentów)
    {"id": "browse-doc-ids", "path": "/browse/doc-ids", "method": "POST", "label": "Browse: doc-ids (FTS, uproszczone)", "body": "{\n  \"query\": [\n    \"regulamin\"\n  ],\n  \"match\": \"phrase\",\n  \"status\": \"active\",\n  \"kinds\": [\"regulation\"]\n}"},
    # NOTE: /browse/facets removed in 2.43.0
    # Cytaty (enumeracja w wybranych dokumentach)
    {"id": "quotes-find", "path": "/quotes/find", "method": "POST", "label": "Quotes: znajdź cytaty (restrict_doc_ids)", "body": "{\n  \"restrict_doc_ids\": [\n    \"<doc_id_1>\",\n    \"<doc_id_2>\"\n  ],\n  \"query\": \"fraza do znalezienia\",\n  \"match\": \"phrase\",\n  \"granularity\": \"occurrence\",\n  \"limit\": 50\n}"},
    # Search (dowody do odpowiedzi)
    {"id": "search-query-restricted", "path": "/search/query", "method": "POST", "label": "Search (restricted by doc_ids)", "body": "{\n  \"query\": [\n    \"cytaty dla skrótu\"\n  ],\n  \"mode\": \"auto\",\n  \"use_hybrid\": true,\n  \"top_m\": 1200,\n  \"top_k\": 100,\n  \"per_doc_limit\": 50,\n  \"result_format\": \"blocks\",\n  \"restrict_doc_ids\": [\n    \"<doc_id_1>\",\n    \"<doc_id_2>\"\n  ]\n}"},
    # Multi-query flow mirroring /search/query
    {"id": "search-debug-embed-multi", "path": "/search/debug/embed_multi", "method": "POST", "label": "Search Debug (multi): 1) embed", "body": "{\n  \"query\": [\n    \"pierwsze zapytanie\",\n    \"drugie zapytanie\"\n  ],\n  \"mode\": \"auto\",\n  \"use_hybrid\": true,\n  \"top_m\": 100,\n  \"top_k\": 10,\n  \"per_doc_limit\": 2,\n  \"score_norm\": \"minmax\",\n  \"dense_weight\": 0.6,\n  \"sparse_weight\": 0.4,\n  \"mmr_lambda\": 0.3,\n  \"mmr_stage1\": true,\n  \"result_format\": \"blocks\",\n  \"summary_mode\": \"first\"\n}"},
    {"id": "search-debug-stage1-multi", "path": "/search/debug/stage1_multi", "method": "POST", "label": "Search Debug (multi): 2) stage1", "body": "{\n  \"queries\": [\n    \"pierwsze zapytanie\",\n    \"drugie zapytanie\"\n  ],\n  \"q_vecs\": [[0.0],[0.0]],\n  \"mode\": \"auto\",\n  \"use_hybrid\": true,\n  \"top_m\": 100,\n  \"score_norm\": \"minmax\",\n  \"dense_weight\": 0.6,\n  \"sparse_weight\": 0.4,\n  \"mmr_stage1\": true,\n  \"mmr_lambda\": 0.3\n}"},
    {"id": "search-debug-stage2-multi", "path": "/search/debug/stage2_multi", "method": "POST", "label": "Search Debug (multi): 3) stage2 + RRF", "body": "{\n  \"queries\": [\n    \"pierwsze zapytanie\",\n    \"drugie zapytanie\"\n  ],\n  \"q_vecs\": [[0.0],[0.0]],\n  \"mode\": \"auto\",\n  \"cand_doc_ids_list\": [[\"<doc_id>\"] , [\"<doc_id>\"]],\n  \"doc_maps\": [{}, {}],\n  \"top_m\": 100,\n  \"top_k\": 10,\n  \"per_doc_limit\": 2,\n  \"score_norm\": \"minmax\",\n  \"dense_weight\": 0.6,\n  \"sparse_weight\": 0.4,\n  \"mmr_lambda\": 0.3,\n  \"result_format\": \"blocks\",\n  \"summary_mode\": \"first\"\n}"},
    {"id": "search-debug-shape-multi", "path": "/search/debug/shape_multi", "method": "POST", "label": "Search Debug (multi): 4) shape fused", "body": "{\n  \"fused_hits\": [ { \"doc_id\": \"<doc_id>\", \"chunk_id\": 0, \"score\": 0.04 } ],\n  \"result_format\": \"blocks\",\n  \"summary_mode\": \"first\"\n}"},
    # Primary functional endpoints for convenience in Admin UI
    {"id": "about", "path": "/about", "method": "GET"},
    {"id": "health", "path": "/health", "method": "GET"},
    {"id": "collections-init", "path": "/collections/init", "method": "POST", "body": "{\n  \"collection_name\": \"rags_tool\",\n  \"force_dim_probe\": false\n}"},
    {"id": "collections-export", "path": "/collections/export", "method": "POST", "label": "Eksport kolekcji (plik .tar.gz)", "body": "{}"},
    {"id": "collections-import", "path": "/collections/import", "method": "POST", "label": "Import kolekcji z archiwum", "body": "{\n  \"archive_base64\": \"<wklej_archiwum_base64>\",\n  \"replace_existing\": true\n}", "accepts_file": True},
    {"id": "ingest-scan", "path": "/ingest/scan", "method": "POST", "body": "{\n  \"base_dir\": \"/app/data\",\n  \"glob\": \"**/*\",\n  \"recursive\": true\n}"},
    {"id": "summaries-generate", "path": "/summaries/generate", "method": "POST", "body": "{\n  \"files\": [\n    \"/app/data/example.md\"\n  ]\n}"},
    {"id": "ingest-build", "path": "/ingest/build", "method": "POST", "body": "{\n  \"base_dir\": \"/app/data\",\n  \"glob\": \"**/*\",\n  \"recursive\": true,\n  \"reindex\": false,\n  \"chunk_tokens\": 1200,\n  \"chunk_overlap\": 150,\n  \"collection_name\": \"rags_tool\",\n  \"enable_sparse\": true,\n  \"rebuild_tfidf\": true,\n  \"force_regen_summary\": false\n}"},
    {"id": "search-query", "path": "/search/query", "method": "POST", "body": "{\n  \"query\": [\n    \"Jak działa rags_tool?\",\n    \"architektura rags_tool\"\n  ],\n  \"top_m\": 10,\n  \"top_k\": 5,\n  \"mode\": \"auto\",\n  \"use_hybrid\": true,\n  \"dense_weight\": 0.6,\n  \"sparse_weight\": 0.4,\n  \"mmr_lambda\": 0.3,\n  \"per_doc_limit\": 2,\n  \"score_norm\": \"minmax\",\n  \"rep_alpha\": 0.6,\n  \"mmr_stage1\": true,\n  \"summary_mode\": \"first\",\n  \"result_format\": \"blocks\"\n}"},
]

ADMIN_UI_REQUEST_HEADER = "x-admin-ui"


# Pack a sparse (indices, values) tuple into a serializable dict.
def _sq_pack(sq: Optional[Tuple[List[int], List[float]]]) -> Optional[Dict[str, Any]]:
    """Pack a sparse vector tuple into a JSON-serializable dict."""
    if not sq:
        return None
    idx, val = sq
    return {"indices": [int(i) for i in idx], "values": [float(v) for v in val]}


# Inspect registered routes and assemble Admin UI operation specs.
def _build_admin_operations(app) -> List[Dict[str, Any]]:
    """Build Admin UI operation descriptors from registered routes.

    Enhancements:
    - Derive human‑readable parameter docs for request body models (Pydantic),
      so Admin UI shows a concise, up‑to‑date list of accepted fields with
      defaults and allowed values when detectable. This avoids duplication by
      sourcing directly from model `Field` descriptions.
    """
    def _short_type(tp: Any) -> str:
        """Return a short, user‑friendly type name for display in UI."""
        try:
            origin = get_origin(tp)
            args = get_args(tp)
            if str(origin).endswith("Union"):
                # Optional[X] is Union[X, NoneType]
                args_set = set(args)
                if type(None) in args_set and len(args) == 2:
                    other = next(a for a in args if a is not type(None))
                    return f"Optional[{_short_type(other)}]"
                return "Union[" + ", ".join(_short_type(a) for a in args) + "]"
            if origin is list or origin is List:
                inner = _short_type(args[0]) if args else "Any"
                return f"List[{inner}]"
            if origin is tuple or origin is Tuple:
                inner = ", ".join(_short_type(a) for a in args) if args else ""
                return f"Tuple[{inner}]"
            if origin is dict or origin is Dict:
                if len(args) == 2:
                    return f"Dict[{_short_type(args[0])}, {_short_type(args[1])}]"
                return "Dict"
            if origin is Optional:
                inner = _short_type(args[0]) if args else "Any"
                return f"Optional[{inner}]"
        except Exception:
            pass
        # Builtins and typing names
        try:
            if tp in (str, int, float, bool):
                return tp.__name__
        except Exception:
            pass
        return getattr(tp, "__name__", str(tp))

    def _extract_choices(desc: Optional[str]) -> Optional[str]:
        """Extract pipe‑separated choices like a|b|c from description text."""
        if not desc:
            return None
        import re

        # Find sequences like word|word|word (letters, digits, _ . -)
        candidates = re.findall(r"(?:[A-Za-z0-9_.-]+\|){1,}[A-Za-z0-9_.-]+", desc)
        if not candidates:
            return None
        # Choose the longest by number of variants
        best = max(candidates, key=lambda s: s.count("|"))
        # De‑duplicate and keep order
        parts: List[str] = []
        for p in best.split("|"):
            if p not in parts:
                parts.append(p)
        return " | ".join(parts)

    def _model_param_doc(model_cls: Any) -> Optional[str]:
        """Build a Polish parameter section for a Pydantic BaseModel class."""
        try:
            if not (isinstance(model_cls, type) and issubclass(model_cls, BaseModel)):
                return None
        except Exception:
            return None
        lines: List[str] = []
        # Header
        lines.append(f"Parametry ({getattr(model_cls, '__name__', 'model')}):")
        try:
            fields = getattr(model_cls, "model_fields", {})
            # Preserve declared order
            for name, fi in fields.items():
                ann = getattr(fi, "annotation", Any)
                typ = _short_type(ann)
                # Default handling (pydantic v2)
                default_val = getattr(fi, "default", ...)
                if default_val is ...:
                    # Try default factory (pydantic v2)
                    factory = getattr(fi, "default_factory", None)
                    if callable(factory):
                        try:
                            default_val = factory()
                        except Exception:
                            default_val = "<factory>"
                default_str: str
                if default_val is ...:
                    default_str = "wymagany"
                else:
                    try:
                        default_str = json.dumps(default_val, ensure_ascii=False)
                    except Exception:
                        default_str = str(default_val)
                desc = getattr(fi, "description", None) or ""
                choices = _extract_choices(desc)
                bullet = f"- {name} ({typ}) domyślnie: {default_str}. {desc}".strip()
                if choices:
                    bullet += f" Dozwolone wartości: {choices}."
                lines.append(bullet)
        except Exception:
            # Fallback without fields listing
            return None
        return "\n".join(lines)

    def _request_model_from_route(route: Optional[APIRoute]) -> Optional[Any]:
        if not route or not getattr(route, "endpoint", None):
            return None
        try:
            sig = inspect.signature(route.endpoint)
            for p in sig.parameters.values():
                ann = p.annotation
                try:
                    if isinstance(ann, type) and issubclass(ann, BaseModel):
                        return ann
                except Exception:
                    continue
        except Exception:
            return None
        # Try FastAPI dependant/body_params for request model (robust path)
        try:
            dep = getattr(route, "dependant", None)
            body_params = getattr(dep, "body_params", []) if dep else []
            for bp in body_params or []:
                # Try a few likely attribute names across FastAPI/Pydantic versions
                for attr in ("annotation", "type_annotation", "type_", "outer_type_", "field"):  # type: ignore[attr-defined]
                    ann = getattr(bp, attr, None)
                    if ann is None:
                        continue
                    try:
                        if isinstance(ann, type) and issubclass(ann, BaseModel):
                            return ann
                    except Exception:
                        # Some wrappers carry nested `.type_` / `.annotation`
                        inner = getattr(ann, "type_", None) or getattr(ann, "annotation", None)
                        try:
                            if isinstance(inner, type) and issubclass(inner, BaseModel):
                                return inner
                        except Exception:
                            pass
        except Exception:
            pass
        # Fallback for known route id/path
        try:
            if getattr(route, "path", "") == "/search/query" or getattr(route, "operation_id", "") == "rags_tool_search":
                from app.models import SearchQuery as _SQ  # lazy import to avoid cycles
                if isinstance(_SQ, type) and issubclass(_SQ, BaseModel):
                    return _SQ
        except Exception:
            pass
        return None
    route_lookup: Dict[Tuple[str, str], APIRoute] = {}
    for route in app.routes:
        if isinstance(route, APIRoute):
            for method in route.methods or []:
                route_lookup[(route.path, method.upper())] = route
    operations: List[Dict[str, Any]] = []
    for spec in ADMIN_OPERATION_SPECS:
        method = spec["method"].upper()
        path = spec["path"]
        route = route_lookup.get((path, method))
        summary = route.summary if route else None
        description = route.description if route else None
        doc_parts = [part for part in [summary, description] if part]
        # Add dynamic parameter list for request model (keeps docs in sync)
        model_cls = _request_model_from_route(route)
        extra = _model_param_doc(model_cls) if model_cls else None
        if extra:
            doc_parts.append(extra)
        handler_name = None
        handler_loc = None
        if route and getattr(route, "endpoint", None):
            try:
                handler_name = getattr(route.endpoint, "__name__", None)
                handler_loc = f"{getattr(route.endpoint, '__module__', 'app')}:{handler_name}"
            except Exception:
                handler_name = None
                handler_loc = None
        label = spec.get("label") or f"{method} {path}"
        if path == "/search/query":
            label = f"{label} ({'full-corpus' if settings.search_skip_stage1_default else 'two-stage'})"
        operations.append(
            {
                "id": spec["id"],
                "label": label,
                "method": method,
                "path": path,
                "doc": "\n\n".join(doc_parts),
                "body": spec.get("body"),
                "accepts_file": spec.get("accepts_file", False),
                "meta": {
                    "app_version": settings.app_version,
                    "skip_stage1": bool(settings.search_skip_stage1_default),
                },
                "handler": handler_name,
                "handler_loc": handler_loc,
            }
        )
    return operations


# Attach Admin UI and multi-step debug endpoints to the given FastAPI app.
def attach_admin_routes(app) -> None:
    """Attach Admin UI and multi-step debug endpoints to a FastAPI app."""
    # Collect existing routes to avoid duplicate registration when imported alongside legacy definitions
    existing = set()
    try:
        for route in app.routes:
            if isinstance(route, APIRoute):
                for method in route.methods or []:
                    existing.add((route.path, method.upper()))
    except Exception:
        existing = set()
    # Middleware (activate DEBUG logs from Admin UI)
    async def admin_ui_debug_middleware(request: Request, call_next):
        header_value = request.headers.get(ADMIN_UI_REQUEST_HEADER)
        if header_value and header_value.lower() in {"1", "true", "yes"}:
            logger.setLevel(logging.DEBUG)
            logger.info("Admin UI request detected — DEBUG logging enabled")
        response = await call_next(request)
        return response

    app.middleware("http")(admin_ui_debug_middleware)

    # /admin console (HTML)
    def admin_console():
        operations = _build_admin_operations(app)
        tpl_path = pathlib.Path(__file__).parent.parent / "templates" / "admin.html"
        try:
            html = tpl_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to load admin template: %s", exc)
            return HTMLResponse(content="<html><body><p>Admin UI unavailable.</p></body></html>")
        html = html.replace("__OPERATIONS__", json.dumps(operations, ensure_ascii=False))
        return HTMLResponse(content=html)

    if ("/admin", "GET") not in existing:
        app.get("/admin", include_in_schema=False, response_class=HTMLResponse, summary="Panel administracyjny", description="Statyczny panel HTML do testowania i debugowania endpointów rags_tool.")(admin_console)

    # --- Single debug removed (migrated to multi) ---

    # --- Multi-query debug endpoints (mirror /search/query pipeline) ---

    def _sq_unpack_list(items: Optional[List[Optional[object]]]) -> Optional[List[Optional[Tuple[List[int], List[float]]]]]:
        if items is None:
            return None
        out: List[Optional[Tuple[List[int], List[float]]]] = []
        for it in items:
            if not it:
                out.append(None)
                continue
            idx: Optional[List[int]] = None
            val: Optional[List[float]] = None
            if isinstance(it, dict):
                idx = it.get("indices", [])  # type: ignore[arg-type]
                val = it.get("values", [])   # type: ignore[arg-type]
            else:
                # Pydantic model (SparseQuery) or similar
                try:
                    idx = getattr(it, "indices", None)
                    val = getattr(it, "values", None)
                    if (idx is None or val is None) and hasattr(it, "dict"):
                        d = it.dict()  # type: ignore[attr-defined]
                        idx = d.get("indices", [])
                        val = d.get("values", [])
                except Exception:
                    idx = []
                    val = []
            out.append((list(map(int, idx or [])), list(map(float, val or []))))
        return out

    def search_debug_embed_multi(req: DebugMultiEmbedRequest):
        # Canonicalize queries
        if isinstance(req.query, list):
            queries = [str(x).strip() for x in req.query if str(x or "").strip()]
        else:
            qraw = str(req.query or "").strip()
            queries = [qraw] if qraw else []
        if not queries:
            raise HTTPException(status_code=422, detail="Field 'query' must contain at least one non-empty string")

        # Determine unified mode like /search/query
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

        # Embed all queries and build sparse queries per query
        q_vecs = embed_query(queries)
        content_sparse_queries: List[Optional[Tuple[List[int], List[float]]]] = []
        summary_sparse_queries: List[Optional[Tuple[List[int], List[float]]]] = []
        for q in queries:
            c_sq, s_sq = _build_sparse_queries_for_query(q, req.use_hybrid)
            content_sparse_queries.append(c_sq)
            summary_sparse_queries.append(s_sq)

        def pack_list(sqs):
            out = []
            for sq in sqs:
                out.append(_sq_pack(sq))
            return out

        next_payload = {
            "queries": queries,
            "q_vecs": q_vecs,
            "mode": mode,
            "use_hybrid": bool(req.use_hybrid),
            "top_m": int(req.top_m),
            "score_norm": str(req.score_norm),
            "dense_weight": float(req.dense_weight),
            "sparse_weight": float(req.sparse_weight),
            "mmr_stage1": bool(req.mmr_stage1),
            "mmr_lambda": float(req.mmr_lambda),
            "rep_alpha": float(req.rep_alpha) if req.rep_alpha is not None else None,
            "top_k": int(req.top_k),
            "per_doc_limit": int(req.per_doc_limit),
            "result_format": str(req.result_format),
            "summary_mode": str(req.summary_mode),
            "summary_sparse_queries": pack_list(summary_sparse_queries),
            "content_sparse_queries": pack_list(content_sparse_queries),
        }

        return {
            "step": "embed_multi",
            "queries": queries,
            "mode": mode,
            "q_vecs": q_vecs,
            "q_vec_lens": [len(v) for v in q_vecs],
            "content_sparse_queries": pack_list(content_sparse_queries),
            "summary_sparse_queries": pack_list(summary_sparse_queries),
            "_next": {"operation_id": "search-debug-stage1-multi", "payload": next_payload},
        }

    if ("/search/debug/embed_multi", "POST") not in existing:
        app.post("/search/debug/embed_multi", include_in_schema=False, summary="Search Debug (multi): Etap 1/4 — embed wszystkich zapytań")(search_debug_embed_multi)

    def search_debug_stage1_multi(req: DebugMultiStage1Request):
        # Build filter for mode
        flt = None
        if req.mode in ("current", "archival"):
            flt = qm.Filter(must=[qm.FieldCondition(key="is_active", match=qm.MatchValue(value=(req.mode == "current")))])

        # Unpack sparse per query
        s_list = _sq_unpack_list([sq.dict() if sq else None for sq in (req.summary_sparse_queries or [])]) if req.summary_sparse_queries is not None else None

        cand_doc_ids_list: List[List[str]] = []
        doc_maps: List[Dict[str, Any]] = []

        class _R:
            def __init__(self, src: DebugMultiStage1Request):
                self.top_m = src.top_m
                self.score_norm = src.score_norm
                self.dense_weight = src.dense_weight
                self.sparse_weight = src.sparse_weight
                self.mmr_stage1 = src.mmr_stage1
                self.mmr_lambda = src.mmr_lambda
                self.rep_alpha = src.rep_alpha

        for i, (q, qv) in enumerate(zip(req.queries, req.q_vecs)):
            s_sq = s_list[i] if s_list and i < len(s_list) else None
            cand_doc_ids, doc_map = _stage1_select_documents(q, qv, flt, s_sq, _R(req))
            cand_doc_ids_list.append(cand_doc_ids)
            doc_maps.append(doc_map)

        next_payload = {
            "queries": req.queries,
            "q_vecs": req.q_vecs,
            "mode": req.mode,
            "cand_doc_ids_list": cand_doc_ids_list,
            "doc_maps": doc_maps,
            "content_sparse_queries": [sq.dict() if sq else None for sq in (req.content_sparse_queries or [])] if req.content_sparse_queries is not None else None,
            "top_m": req.top_m,
            "top_k": req.top_k,
            "per_doc_limit": req.per_doc_limit,
            "score_norm": req.score_norm,
            "dense_weight": req.dense_weight,
            "sparse_weight": req.sparse_weight,
            "mmr_lambda": req.mmr_lambda,
            "rep_alpha": req.rep_alpha,
            "result_format": req.result_format,
            "summary_mode": req.summary_mode,
        }

        return {
            "step": "stage1_multi",
            "cand_doc_ids_list": cand_doc_ids_list,
            "doc_maps": doc_maps,
            "_next": {"operation_id": "search-debug-stage2-multi", "payload": next_payload},
        }

    if ("/search/debug/stage1_multi", "POST") not in existing:
        app.post("/search/debug/stage1_multi", include_in_schema=False, summary="Search Debug (multi): Etap 2/4 — dokumenty dla wszystkich zapytań")(search_debug_stage1_multi)

    def search_debug_stage2_multi(req: DebugMultiStage2Request):
        # Filter for mode
        flt = None
        if req.mode in ("current", "archival"):
            flt = qm.Filter(must=[qm.FieldCondition(key="is_active", match=qm.MatchValue(value=(req.mode == "current")))])

        # Settings parity with /search/query
        ranker_enabled = bool(settings.ranker_base_url and settings.ranker_model)
        RERANK_TOP_N_MAX = max(1, int(getattr(settings, "rerank_top_n_max", 50)))
        RRF_K = 60
        OVERSAMPLE = 2

        # Unpack content sparse per query
        c_list = _sq_unpack_list([sq.dict() if sq else None for sq in (req.content_sparse_queries or [])]) if req.content_sparse_queries is not None else None

        per_query_hits: List[List[Dict[str, Any]]] = []
        fused: Dict[tuple, Dict[str, Any]] = {}

        class _R2:
            def __init__(self, src: DebugMultiStage2Request, top_k_override: int):
                self.top_m = src.top_m
                self.top_k = top_k_override
                self.per_doc_limit = src.per_doc_limit
                self.score_norm = src.score_norm
                self.dense_weight = src.dense_weight
                self.sparse_weight = src.sparse_weight
                self.mmr_lambda = src.mmr_lambda
                self.rep_alpha = src.rep_alpha
                self.result_format = "flat"
                self.summary_mode = "first"

        for i, (q, qv) in enumerate(zip(req.queries, req.q_vecs)):
            per_query_limit = (max(1, min(int(RERANK_TOP_N_MAX), max(int(req.top_k), int(req.top_k) * OVERSAMPLE)))) if ranker_enabled else max(1, int(req.top_k) * OVERSAMPLE)
            cand_ids = None
            doc_map = {}
            if req.cand_doc_ids_list is not None and i < len(req.cand_doc_ids_list):
                cand_ids = req.cand_doc_ids_list[i]
            if req.doc_maps is not None and i < len(req.doc_maps):
                doc_map = req.doc_maps[i] or {}
            c_sq = c_list[i] if c_list and i < len(c_list) else None
            final_hits, mmr_pool, rel2 = _stage2_select_chunks(cand_ids if cand_ids else None, q, qv, c_sq, doc_map, _R2(req, per_query_limit), flt)
            # Convert to debug hits and accumulate RRF
            dbg_hits: List[Dict[str, Any]] = []
            for rank, fh in enumerate(final_hits, start=1):
                payload = fh.get("payload") or {}
                did = payload.get("doc_id", "")
                sec = payload.get("section_path")
                cid = int(payload.get("chunk_id", 0))
                snippet = (payload.get("text") or "").strip()[:400] if payload.get("text") else (payload.get("summary", "")[:400])
                dbg_hits.append({
                    "doc_id": did,
                    "path": payload.get("path"),
                    "section": sec,
                    "chunk_id": cid,
                    "score": float(fh.get("score", 0.0)),
                    "snippet": snippet,
                })
                key = (did, sec, cid)
                incr = 1.0 / (RRF_K + rank)
                entry = fused.get(key)
                if entry is None:
                    fused[key] = {"payload": payload, "score": incr}
                else:
                    entry["score"] += incr
            per_query_hits.append(dbg_hits)

        fused_list = [
            {"payload": v.get("payload"), "score": float(v.get("score", 0.0))}
            for v in fused.values()
        ]
        fused_list.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        final_fused = fused_list  # nie obcinamy tu; shaping może ograniczyć

        fused_hits_dbg = []
        for fh in final_fused:
            payload = fh.get("payload") or {}
            fused_hits_dbg.append({
                "doc_id": payload.get("doc_id", ""),
                "path": payload.get("path"),
                "section": payload.get("section_path"),
                "chunk_id": int(payload.get("chunk_id", 0)),
                "score": float(fh.get("score", 0.0)),
                "snippet": (payload.get("text") or "").strip()[:400] if payload.get("text") else (payload.get("summary", "")[:400]),
            })

        next_payload = {
            "fused_hits": fused_hits_dbg[: max(1, int(req.top_k))] if not ranker_enabled else fused_hits_dbg,
            "result_format": req.result_format,
            "summary_mode": req.summary_mode,
        }

        return {
            "step": "stage2_multi",
            "per_query_hits": per_query_hits,
            "fused_hits": fused_hits_dbg,
            "_next": {"operation_id": "search-debug-shape-multi", "payload": next_payload},
        }

    if ("/search/debug/stage2_multi", "POST") not in existing:
        app.post("/search/debug/stage2_multi", include_in_schema=False, summary="Search Debug (multi): Etap 3/4 — chunki + fuzja RRF")(search_debug_stage2_multi)

    def search_debug_shape_multi(req: DebugMultiShapeRequest):
        # Build final_hits acceptable by _shape_results
        final_hits = []
        for h in req.fused_hits:
            payload = {
                "doc_id": h.doc_id,
                "path": h.path,
                "section_path": h.section,
                "chunk_id": h.chunk_id,
                "text": h.snippet or "",
            }
            final_hits.append({"payload": payload, "score": float(h.score)})

        class _R4:
            def __init__(self, src: DebugMultiShapeRequest):
                self.result_format = src.result_format
                self.summary_mode = src.summary_mode

        results, groups, blocks = _shape_results(final_hits, {}, [], [], _R4(req))
        return {
            "step": "shape_multi",
            "results": results,
            "groups": groups,
            "blocks": blocks,
        }

    if ("/search/debug/shape_multi", "POST") not in existing:
        app.post("/search/debug/shape_multi", include_in_schema=False, summary="Search Debug (multi): Etap 4/4 — kształtowanie fuzji")(search_debug_shape_multi)

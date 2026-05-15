"""Admin UI routes for operational endpoints."""

from __future__ import annotations

import inspect
import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin

from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel

from app.models import SearchQuery
from app.settings import get_settings


settings = get_settings()


ADMIN_OPERATION_SPECS: List[Dict[str, Any]] = [
    {"id": "about", "path": "/about", "method": "GET"},
    {"id": "health", "path": "/health", "method": "GET"},
    {"id": "docs-stats", "path": "/docs/stats", "method": "GET", "label": "Statystyka dokumentow (FTS)"},
    {"id": "fts-rebuild", "path": "/fts/rebuild", "method": "POST", "label": "Odbuduj FTS (chunki -> SQLite)", "body": "{}"},
    {
        "id": "browse-doc-ids",
        "path": "/browse/doc-ids",
        "method": "POST",
        "label": "Browse: doc-ids",
        "body": "{\n  \"query\": [\n    \"regulamin\"\n  ],\n  \"match\": \"phrase\",\n  \"status\": \"active\",\n  \"kinds\": [\"regulation\"]\n}",
    },
    {
        "id": "quotes-find",
        "path": "/quotes/find",
        "method": "POST",
        "label": "Quotes: znajdz cytaty",
        "body": "{\n  \"restrict_doc_ids\": [\n    \"<doc_id_1>\",\n    \"<doc_id_2>\"\n  ],\n  \"query\": \"fraza do znalezienia\",\n  \"match\": \"phrase\",\n  \"granularity\": \"occurrence\",\n  \"limit\": 50\n}",
    },
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
        "label": "Eksport aktywnych kolekcji projektu (plik .tar.gz)",
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
        "body": "{\n  \"base_dir\": \"/app/data\",\n  \"glob\": \"**/*\",\n  \"recursive\": true,\n  \"reindex\": false,\n  \"chunk_tokens\": 1200,\n  \"chunk_overlap\": 150,\n  \"collection_name\": \"rags_tool\",\n  \"enable_sparse\": true,\n  \"rebuild_tfidf\": true,\n  \"force_regen_summary\": false\n}",
    },
    {
        "id": "search-query",
        "path": "/search/query",
        "method": "POST",
        "body": "{\n  \"query\": [\n    \"Jak dziala rags_tool?\",\n    \"architektura rags_tool\"\n  ],\n  \"top_m\": 10,\n  \"top_k\": 5,\n  \"mode\": \"auto\",\n  \"use_hybrid\": true,\n  \"dense_weight\": 0.6,\n  \"sparse_weight\": 0.4,\n  \"mmr_lambda\": 0.3,\n  \"per_doc_limit\": 2,\n  \"score_norm\": \"minmax\",\n  \"rep_alpha\": 0.6,\n  \"mmr_stage1\": true,\n  \"summary_mode\": \"first\",\n  \"result_format\": \"blocks\"\n}",
    },
    {
        "id": "search-query-restricted",
        "path": "/search/query",
        "method": "POST",
        "label": "Search (restricted by doc_ids)",
        "body": "{\n  \"query\": [\n    \"cytaty dla skrotu\"\n  ],\n  \"top_m\": 1200,\n  \"top_k\": 100,\n  \"mode\": \"auto\",\n  \"use_hybrid\": true,\n  \"per_doc_limit\": 50,\n  \"result_format\": \"blocks\",\n  \"restrict_doc_ids\": [\n    \"<doc_id_1>\",\n    \"<doc_id_2>\"\n  ]\n}",
    },
]


def _build_admin_operations(app) -> List[Dict[str, Any]]:
    """Build Admin UI operation descriptors from registered routes."""

    def _short_type(tp: Any) -> str:
        try:
            origin = get_origin(tp)
            args = get_args(tp)
            if str(origin).endswith("Union"):
                args_set = set(args)
                if type(None) in args_set and len(args) == 2:
                    other = next(a for a in args if a is not type(None))
                    return f"Optional[{_short_type(other)}]"
                return "Union[" + ", ".join(_short_type(a) for a in args) + "]"
            if origin in (list, List):
                inner = _short_type(args[0]) if args else "Any"
                return f"List[{inner}]"
            if origin in (tuple, Tuple):
                inner = ", ".join(_short_type(a) for a in args) if args else ""
                return f"Tuple[{inner}]"
            if origin in (dict, Dict):
                if len(args) == 2:
                    return f"Dict[{_short_type(args[0])}, {_short_type(args[1])}]"
                return "Dict"
            if origin is Optional:
                inner = _short_type(args[0]) if args else "Any"
                return f"Optional[{inner}]"
        except Exception:
            pass
        try:
            if tp in (str, int, float, bool):
                return tp.__name__
        except Exception:
            pass
        return getattr(tp, "__name__", str(tp))

    def _extract_choices(desc: Optional[str]) -> Optional[str]:
        if not desc:
            return None
        import re

        candidates = re.findall(r"(?:[A-Za-z0-9_.-]+\|){1,}[A-Za-z0-9_.-]+", desc)
        if not candidates:
            return None
        best = max(candidates, key=lambda s: s.count("|"))
        parts: List[str] = []
        for part in best.split("|"):
            if part not in parts:
                parts.append(part)
        return " | ".join(parts)

    def _model_param_doc(model_cls: Any) -> Optional[str]:
        try:
            if not (isinstance(model_cls, type) and issubclass(model_cls, BaseModel)):
                return None
        except Exception:
            return None

        lines: List[str] = [f"Parametry ({getattr(model_cls, '__name__', 'model')}):"]
        try:
            for name, fi in getattr(model_cls, "model_fields", {}).items():
                ann = getattr(fi, "annotation", Any)
                default_val = getattr(fi, "default", ...)
                if default_val is ...:
                    factory = getattr(fi, "default_factory", None)
                    if callable(factory):
                        try:
                            default_val = factory()
                        except Exception:
                            default_val = "<factory>"
                if default_val is ...:
                    default_str = "wymagany"
                else:
                    try:
                        default_str = json.dumps(default_val, ensure_ascii=False)
                    except Exception:
                        default_str = str(default_val)
                desc = getattr(fi, "description", None) or ""
                choices = _extract_choices(desc)
                line = f"- {name} ({_short_type(ann)}) domyslnie: {default_str}. {desc}".strip()
                if choices:
                    line += f" Dozwolone wartosci: {choices}."
                lines.append(line)
        except Exception:
            return None
        return "\n".join(lines)

    def _request_model_from_route(route: Optional[APIRoute]) -> Optional[Any]:
        if not route or not getattr(route, "endpoint", None):
            return None
        try:
            sig = inspect.signature(route.endpoint)
            for param in sig.parameters.values():
                ann = param.annotation
                try:
                    if isinstance(ann, type) and issubclass(ann, BaseModel):
                        return ann
                except Exception:
                    continue
        except Exception:
            return None

        try:
            dep = getattr(route, "dependant", None)
            for bp in getattr(dep, "body_params", []) if dep else []:
                for attr in ("annotation", "type_annotation", "type_", "outer_type_", "field"):
                    ann = getattr(bp, attr, None)
                    if ann is None:
                        continue
                    try:
                        if isinstance(ann, type) and issubclass(ann, BaseModel):
                            return ann
                    except Exception:
                        inner = getattr(ann, "type_", None) or getattr(ann, "annotation", None)
                        try:
                            if isinstance(inner, type) and issubclass(inner, BaseModel):
                                return inner
                        except Exception:
                            pass
        except Exception:
            pass

        if getattr(route, "path", "") == "/search/query":
            return SearchQuery
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
        model_cls = _request_model_from_route(route)
        param_doc = _model_param_doc(model_cls) if model_cls else None

        doc_parts = [part for part in (summary, description, param_doc) if part]
        handler_name = None
        handler_loc = None
        if route and getattr(route, "endpoint", None):
            handler_name = getattr(route.endpoint, "__name__", None)
            handler_loc = f"{getattr(route.endpoint, '__module__', 'app')}:{handler_name}"

        op = dict(spec)
        op["label"] = op.get("label") or summary or f"{method} {path}"
        op["doc"] = "\n\n".join(doc_parts) if doc_parts else "Brak dokumentacji dla tej operacji."
        op["handler"] = handler_name
        op["handler_loc"] = handler_loc

        meta: Dict[str, Any] = {"app_version": settings.app_version}
        if path == "/search/query":
            meta["skip_stage1"] = bool(getattr(settings, "search_skip_stage1_default", False))
        op["meta"] = meta
        operations.append(op)

    return operations


def attach_admin_routes(app) -> None:
    """Attach the operational Admin UI to a FastAPI app."""
    existing = set()
    try:
        for route in app.routes:
            if isinstance(route, APIRoute):
                for method in route.methods or []:
                    existing.add((route.path, method.upper()))
    except Exception:
        existing = set()

    def admin_console():
        operations = _build_admin_operations(app)
        tpl_path = pathlib.Path(__file__).parent.parent / "templates" / "admin.html"
        try:
            html = tpl_path.read_text(encoding="utf-8")
        except Exception:
            return HTMLResponse(content="<html><body><p>Admin UI unavailable.</p></body></html>")
        html = html.replace("__OPERATIONS__", json.dumps(operations, ensure_ascii=False))
        return HTMLResponse(content=html)

    if ("/admin", "GET") not in existing:
        app.get(
            "/admin",
            include_in_schema=False,
            response_class=HTMLResponse,
            summary="Panel administracyjny",
            description="Statyczny panel HTML do obslugi endpointow rags_tool.",
        )(admin_console)

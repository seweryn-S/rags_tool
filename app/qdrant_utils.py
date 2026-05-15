"""Qdrant client and collection utilities, point upsert helpers."""

from __future__ import annotations

import datetime
import hashlib
import io
import json
import logging
import os
import pathlib
import re
import shutil
import tarfile
import tempfile
import time
import uuid
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple
from urllib import request as _urlreq
from urllib import parse as _urlparse

from qdrant_client import QdrantClient
from openai import BadRequestError  # type: ignore
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.embedding import (
    SUMMARY_VECTORIZER_PATH,
    embed_passage,
    tfidf_vector,
)
from app.core.constants import (
    CONTENT_SPARSE_NAME,
    CONTENT_VECTOR_NAME,
    SPARSE_ENABLED,
    SUMMARY_SPARSE_NAME,
    SUMMARY_VECTOR_NAME,
)
from app.settings import get_settings
from fastapi import HTTPException

settings = get_settings()

logger = logging.getLogger("rags_tool")

qdrant = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
    timeout=settings.qdrant_request_timeout,
)


def _strip_diacritics(s: str) -> str:
    import unicodedata
    norm = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")


def _normalize_subtitle(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return "brak"
    text = _strip_diacritics(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "brak"
    if len(text) > 100:
        text = text[:100]
    return text


def _doc_date_ord(value: Optional[str]) -> int:
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


def find_path_by_content_sha256(content_sha256: str, collection_base: Optional[str]) -> Optional[str]:
    """Return existing document path for a given content SHA-256, if present.

    Looks up a summary point with payload "content_sha256" = value and returns its
    "path" payload. Returns None when not found or on failure.
    """
    try:
        if not content_sha256:
            return None
        # Use alias-backed collection when available so cross-run dedupe
        # always checks against the currently active corpus version.
        try:
            summary_collection, _ = derive_alias_names(collection_base)
        except Exception:
            summary_collection, _ = derive_collection_names(collection_base)
        flt = qm.Filter(
            must=[
                qm.FieldCondition(key="point_type", match=qm.MatchValue(value="summary")),
                qm.FieldCondition(key="content_sha256", match=qm.MatchValue(value=content_sha256)),
            ]
        )
        res = qdrant.scroll(
            collection_name=summary_collection,
            scroll_filter=flt,
            limit=1,
            with_payload=["path"],
            with_vectors=False,
        )
        if isinstance(res, tuple):
            records = res[0]
        else:
            records = getattr(res, "points", None) or []
        for rec in records:
            payload = getattr(rec, "payload", None) or {}
            path = payload.get("path")
            if isinstance(path, str) and path.strip():
                return path
        return None
    except Exception:
        return None


# Remove all entries within a directory without removing the directory itself.
def _clear_dir_contents(target: pathlib.Path) -> None:
    """Remove all contents of a directory without removing the directory itself.

    This is safe for cases where `target` is a mountpoint (e.g., Docker volume),
    where removing the directory would raise 'Device or resource busy' (EBUSY).
    """
    if not target.exists():
        return
    for entry in target.iterdir():
        try:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink(missing_ok=True)  # type: ignore[call-arg]
        except Exception as exc:
            # Attempt a best-effort rename to sidestep EBUSY/permission issues
            try:
                fallback = entry.with_name(f"{entry.name}.old-{uuid.uuid4().hex[:8]}")
                entry.rename(fallback)
            except Exception:
                logger.warning("Nie udało się usunąć/przenieść '%s': %s", entry, exc)


def sha1(s: str) -> str:
    """Return hex SHA1 of the given UTF-8 string."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _derive_summary_collection(base: Optional[str]) -> str:
    """Derive the summary collection name from a base name or settings."""
    if base:
        return f"{base}_summaries"
    if settings.summary_collection_name:
        return settings.summary_collection_name
    return f"{settings.collection_name}_summaries"


def _derive_content_collection(base: Optional[str]) -> str:
    """Derive the content collection name from a base name or settings."""
    if base:
        return f"{base}_content"
    if settings.content_collection_name:
        return settings.content_collection_name
    return f"{settings.collection_name}_content"


def derive_collection_names(base: Optional[str] = None) -> Tuple[str, str]:
    """Return (summary_collection, content_collection) for the provided base."""
    return _derive_summary_collection(base), _derive_content_collection(base)


def derive_alias_names(base: Optional[str] = None) -> Tuple[str, str]:
    """Return (summary_alias, content_alias) for the provided base.

    Aliases are derived from the physical collection names and suffixed with
    '_active' so they can be atomically repointed to a new version of the
    underlying collections during a full rebuild.
    """
    summary_collection, content_collection = derive_collection_names(base)
    return f"{summary_collection}_active", f"{content_collection}_active"


def _collection_names_from_response(listed: Any) -> Set[str]:
    """Extract collection names from mixed qdrant-client response shapes."""
    if isinstance(listed, dict):
        result = listed.get("result", listed)
        entries = result.get("collections", []) if isinstance(result, dict) else []
        return {
            str(item.get("name"))
            for item in entries
            if isinstance(item, dict) and item.get("name")
        }
    return {
        str(item.name)
        for item in getattr(listed, "collections", []) or []
        if getattr(item, "name", None)
    }


def _extract_alias_entries(raw: Any) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        result = raw.get("result", raw)
        if isinstance(result, dict):
            entries = result.get("aliases") or []
            return list(entries) if isinstance(entries, list) else []
        return []

    result = getattr(raw, "result", None)
    if result is not None:
        return _extract_alias_entries(result)

    entries = getattr(raw, "aliases", None)
    if entries is None:
        return []
    try:
        return list(entries)
    except TypeError:
        return []


def _alias_name_from_entry(entry: Any) -> Optional[str]:
    if isinstance(entry, dict):
        value = entry.get("alias_name") or entry.get("alias")
    else:
        value = getattr(entry, "alias_name", None) or getattr(entry, "alias", None)
    return str(value) if value else None


def _alias_collection_from_entry(entry: Any) -> Optional[str]:
    if isinstance(entry, dict):
        value = entry.get("collection_name") or entry.get("collection")
    else:
        value = getattr(entry, "collection_name", None) or getattr(entry, "collection", None)
    return str(value) if value else None


def _parse_alias_targets(raw: Any) -> Dict[str, str]:
    targets: Dict[str, str] = {}
    for entry in _extract_alias_entries(raw):
        alias_name = _alias_name_from_entry(entry)
        collection_name = _alias_collection_from_entry(entry)
        if alias_name and collection_name:
            targets[alias_name] = collection_name
    return targets


def _get_qdrant_alias_targets() -> Dict[str, str]:
    """Return Qdrant alias -> collection mappings using client or REST."""
    method = getattr(qdrant, "get_aliases", None)
    if method is not None:
        try:
            return _parse_alias_targets(method())
        except Exception as exc:
            logger.debug("Client get aliases failed, fallback to REST: %s", exc)

    base = settings.qdrant_url.rstrip("/")
    req = _urlreq.Request(f"{base}/aliases", method="GET", headers=_qdrant_headers_json())
    try:
        with _urlreq.urlopen(req, timeout=settings.qdrant_request_timeout) as resp:
            body = resp.read().decode("utf-8", "ignore")
        return _parse_alias_targets(json.loads(body) if body.strip() else {})
    except Exception as exc:
        logger.debug("REST get aliases failed: %s", exc)
        return {}


def _ensure_collection_available(
    collection_name: Optional[str],
    *,
    role: str,
    available_names: Optional[Set[str]],
) -> str:
    if not collection_name:
        raise HTTPException(
            status_code=404,
            detail=f"Brak aktywnej kolekcji projektu dla roli '{role}'",
        )
    if available_names is not None and collection_name not in available_names:
        raise HTTPException(
            status_code=404,
            detail=f"Aktywna kolekcja projektu '{collection_name}' dla roli '{role}' nie istnieje w Qdrant",
        )
    if available_names is None and not _collection_exists(collection_name):
        raise HTTPException(
            status_code=404,
            detail=f"Aktywna kolekcja projektu '{collection_name}' dla roli '{role}' nie istnieje w Qdrant",
        )
    return collection_name


def _collection_points_count(collection_name: str) -> Optional[int]:
    try:
        info = qdrant.get_collection(collection_name)
        value = getattr(info, "points_count", None)
        return int(value) if value is not None else None
    except Exception as exc:
        logger.debug(
            "Client get_collection failed for '%s', trying raw REST for point count: %s",
            collection_name,
            exc,
        )

    base = settings.qdrant_url.rstrip("/")
    url = f"{base}/collections/{_urlparse.quote(collection_name)}"
    req = _urlreq.Request(url, method="GET", headers=_qdrant_headers_json())
    try:
        with _urlreq.urlopen(req, timeout=settings.qdrant_request_timeout) as resp:
            body = resp.read().decode("utf-8", "ignore")
        parsed = json.loads(body) if body.strip() else {}
        result = parsed.get("result") if isinstance(parsed, dict) else {}
        value = result.get("points_count") if isinstance(result, dict) else None
        return int(value) if value is not None else None
    except Exception as exc:
        logger.debug("Raw REST get collection failed for '%s': %s", collection_name, exc)
        return None


def resolve_active_project_collections(
    available_names: Optional[Set[str]] = None,
) -> List[Dict[str, str]]:
    """Resolve active summary/content collections for the configured project."""
    summary_alias, content_alias = derive_alias_names(None)
    summary_base, content_base = derive_collection_names(None)
    aliases = _get_qdrant_alias_targets()

    summary_target = aliases.get(summary_alias)
    content_target = aliases.get(content_alias)

    if not summary_target and (
        available_names is None and _collection_exists(summary_base)
        or available_names is not None and summary_base in available_names
    ):
        summary_target = summary_base
    if not content_target and (
        available_names is None and _collection_exists(content_base)
        or available_names is not None and content_base in available_names
    ):
        content_target = content_base

    return [
        {
            "role": "summary",
            "alias": summary_alias,
            "name": _ensure_collection_available(
                summary_target,
                role="summary",
                available_names=available_names,
            ),
        },
        {
            "role": "content",
            "alias": content_alias,
            "name": _ensure_collection_available(
                content_target,
                role="content",
                available_names=available_names,
            ),
        },
    ]


DEFAULT_PAYLOAD_INDEXES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("doc_id", {"type": "keyword"}),
    ("point_type", {"type": "keyword"}),
    ("is_active", {"type": "bool"}),
    ("section_path", {"type": "keyword"}),
    ("section_path_prefixes", {"type": "keyword"}),
    ("doc_date", {"type": "keyword"}),
    ("doc_date_ord", {"type": "integer"}),
    ("content_sha256", {"type": "keyword"}),
    ("entities", {"type": "keyword"}),
    ("subtitle", {"type": "keyword"}),
    ("subtitle_norm", {"type": "keyword"}),
    ("ingested_ts", {"type": "integer"}),
)


def _ensure_payload_indexes(
    collection: str,
    specs: Optional[Iterable[Tuple[str, Dict[str, Any]]]],
) -> None:
    """Create payload indexes in a client-version tolerant way.

    Falls back through several schema representations depending on qdrant-client version:
    1) qm.PayloadIndexParams(type=...)
    2) qm.PayloadSchemaType.KEYWORD/BOOL enums
    3) plain dict: {"type": "keyword"}
    """
    if not specs:
        return

    def build_schema(params: Dict[str, Any]):  # type: ignore[override]
        # Try modern pydantic model
        schema_type = (params or {}).get("type", "keyword")
        try:
            PIP = getattr(qm, "PayloadIndexParams", None)
            if PIP is not None:
                # Map plain strings to enums if available
                tval = schema_type
                try:
                    PST = getattr(qm, "PayloadSchemaType", None)
                    if PST is not None:
                        mapping = {
                            "keyword": getattr(PST, "KEYWORD", None),
                            "bool": getattr(PST, "BOOL", None),
                            "integer": getattr(PST, "INTEGER", None),
                            "float": getattr(PST, "FLOAT", None),
                            "text": getattr(PST, "TEXT", None),
                        }
                        if schema_type in mapping and mapping[schema_type] is not None:
                            tval = mapping[schema_type]
                except Exception:
                    pass
                return PIP(type=tval)  # type: ignore[misc]
        except Exception:
            pass
        # Fallback for newer qdrant_client versions: pass plain string
        # schema name (e.g. "keyword", "integer", "bool", "text"). This is
        # compatible with CreateFieldIndex(field_schema: AnyPayloadFieldSchema).
        return str(schema_type)

    for field_name, params in specs:
        try:
            schema = build_schema(params or {})
            qdrant.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=schema,  # type: ignore[arg-type]
            )
            logger.debug("Created payload index '%s' on '%s'", field_name, collection)
        except UnexpectedResponse as exc:
            msg = str(exc).lower()
            if getattr(exc, "status_code", None) == 409 or "exists" in msg or "already" in msg:
                logger.debug("Payload index '%s' already exists on '%s'", field_name, collection)
                continue
            logger.warning(
                "Failed to create payload index '%s' on '%s': %s",
                field_name,
                collection,
                exc,
            )
        except Exception as exc:
            logger.warning(
                "Failed to create payload index '%s' on '%s': %s",
                field_name,
                collection,
                exc,
            )


# Create or validate a single collection; ensure vectors/sparse and indexes.
def _ensure_single_collection(
    collection: str,
    vectors_config: Dict[str, qm.VectorParams],
    sparse_config: Optional[Dict[str, qm.SparseVectorParams]] = None,
    payload_indexes: Optional[Iterable[Tuple[str, Dict[str, Any]]]] = None,
):
    """Create or validate a single Qdrant collection and payload indexes.

    Ensures named dense vectors (and optional sparse vectors) match the
    expected schema. Creates payload indexes in a client-version tolerant way.
    """
    try:
        info = qdrant.get_collection(collection)
        try:
            vecs = getattr(getattr(info, "config", None), "params", None)
            vecs = getattr(vecs, "vectors", None)
            names: set = set()
            if isinstance(vecs, dict):
                names = set(vecs.keys())
            else:
                for attr in ("configs", "map", "vectors", "items", "data"):
                    mapping = getattr(vecs, attr, None)
                    if isinstance(mapping, dict) and mapping:
                        names = set(mapping.keys())
                        break
            expected = set(vectors_config.keys())
            if names != expected:
                logger.error(
                    "Collection '%s' incompatible vectors config. Found names=%s, expected %s",
                    collection,
                    sorted(list(names)) if names else None,
                    sorted(list(expected)),
                )
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Kolekcja '{collection}' ma niekompatybilną konfigurację wektorów. "
                        f"Wymagane nazwane wektory: {sorted(list(expected))}. Usuń kolekcję lub uruchom ingest z reindex=true."
                    ),
                )
        except Exception as exc:
            logger.warning("Could not verify vectors config for '%s': %s", collection, exc)
        if SPARSE_ENABLED and sparse_config:
            try:
                qdrant.update_collection(
                    collection_name=collection,
                    sparse_vectors_config=sparse_config,
                )
                logger.debug("Ensured sparse vectors for '%s' are configured", collection)
            except UnexpectedResponse as exc:
                msg = str(exc).lower()
                if "already" in msg or "exist" in msg or "conflict" in msg:
                    logger.debug("Sparse vectors already present for '%s'", collection)
                else:
                    logger.error("Failed to ensure sparse vectors for '%s': %s", collection, exc)
                    raise
        logger.debug("Collection '%s' exists and is compatible", collection)
        _ensure_payload_indexes(collection, payload_indexes)
        return
    except HTTPException:
        raise
    except Exception:
        logger.debug("Collection '%s' not found; creating", collection)

    try:
        qdrant.create_collection(
            collection_name=collection,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
            optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=20000),
        )
        logger.debug("Collection '%s' created", collection)
    except UnexpectedResponse as exc:
        if getattr(exc, "status_code", None) == 409 or "already exists" in str(exc).lower():
            logger.debug("Collection '%s' creation returned conflict; treating as existing", collection)
            _ensure_payload_indexes(collection, payload_indexes)
            return
        raise
    _ensure_payload_indexes(collection, payload_indexes)


# Ensure summary/content collections exist with expected named vectors.
def ensure_collections(collection_base: Optional[str] = None, dim: Optional[int] = None):
    """Ensure the summary/content collections exist with expected schema."""
    if dim is None:
        dim = settings.embedding_dim
    summary_collection, content_collection = derive_collection_names(collection_base)
    summary_vectors = {
        SUMMARY_VECTOR_NAME: qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    }
    content_vectors = {
        CONTENT_VECTOR_NAME: qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    }
    summary_sparse = (
        {SUMMARY_SPARSE_NAME: qm.SparseVectorParams()}
        if SPARSE_ENABLED
        else None
    )
    content_sparse = (
        {CONTENT_SPARSE_NAME: qm.SparseVectorParams()}
        if SPARSE_ENABLED
        else None
    )
    _ensure_single_collection(
        summary_collection,
        summary_vectors,
        summary_sparse,
        payload_indexes=DEFAULT_PAYLOAD_INDEXES,
    )
    _ensure_single_collection(
        content_collection,
        content_vectors,
        content_sparse,
        payload_indexes=DEFAULT_PAYLOAD_INDEXES,
    )

    # Best-effort: ensure aliases for this pair exist and point to some collection.
    # Aliases are used as stable public names (e.g., *_active) that can be
    # atomically repointed to new physical collections during full rebuilds.
    summary_alias, content_alias = derive_alias_names(collection_base)
    existing_aliases = _get_qdrant_alias_targets()
    for alias_name, target in ((summary_alias, summary_collection), (content_alias, content_collection)):
        if existing_aliases.get(alias_name):
            continue
        try:
            base_url = settings.qdrant_url.rstrip("/")
            url = f"{base_url}/collections/aliases"
            payload = {
                "actions": [
                    {
                        "create_alias": {
                            "collection_name": target,
                            "alias_name": alias_name,
                        }
                    }
                ]
            }
            data = json.dumps(payload).encode("utf-8")
            req = _urlreq.Request(url, data=data, method="POST", headers=_qdrant_headers_json())
            with _urlreq.urlopen(req, timeout=settings.qdrant_request_timeout) as resp:
                _ = resp.read()
            logger.info("Ensured Qdrant alias '%s' -> '%s'", alias_name, target)
        except Exception as exc:
            logger.debug(
                "Ensure alias skipped or failed | alias=%s target=%s error=%s",
                alias_name,
                target,
                exc,
            )


# Export active project collections and local TF‑IDF artifacts as a tar.gz archive.
def export_collections_bundle(collection_names: Optional[Iterable[str]] = None) -> Tuple[bytes, Dict[str, Any]]:
    """Serialize active project Qdrant collections and local TF-IDF indices."""

    try:
        listed = qdrant.get_collections()
    except Exception as exc:  # pragma: no cover - network dependency
        raise HTTPException(status_code=502, detail=f"Nie udało się pobrać listy kolekcji: {exc}") from exc

    available_names = _collection_names_from_response(listed)
    if collection_names:
        logger.info(
            "Parametr collection_names=%s jest przestarzały i został zignorowany; eksport obejmuje aktywne kolekcje projektu '%s'.",
            list(collection_names),
            settings.collection_name,
        )

    active_collections = resolve_active_project_collections(available_names=available_names)
    collection_roles = {entry["name"]: entry for entry in active_collections}
    selected_names = [entry["name"] for entry in active_collections]
    alias_map = {entry["alias"]: entry["name"] for entry in active_collections}

    bundle_buffer = io.BytesIO()
    metadata: Dict[str, Any] = {
        "meta": {
            "generated_at": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "app_version": settings.app_version,
        },
        "project": {
            "collection_name": settings.collection_name,
            "summary_alias": active_collections[0]["alias"],
            "content_alias": active_collections[1]["alias"],
            "aliases": alias_map,
        },
        "collections": [],
    }

    vector_store_dir = settings.vector_store_dir
    vector_files: List[str] = []
    if vector_store_dir.exists():
        for path in vector_store_dir.rglob("*"):
            if path.is_file():
                vector_files.append(str(path.relative_to(vector_store_dir)))

    with tempfile.TemporaryDirectory(prefix="qdrant-export-") as scratch_dir:
        scratch_path = pathlib.Path(scratch_dir)
        with tarfile.open(fileobj=bundle_buffer, mode="w:gz") as tar:
            for name in selected_names:
                logger.debug("Eksport kolekcji '%s'", name)
                points_count = _collection_points_count(name)

                snapshot = _create_collection_snapshot(name)
                snapshot_name = _extract_snapshot_name(snapshot)
                if not snapshot_name:
                    raise HTTPException(status_code=502, detail=f"Snapshot dla kolekcji '{name}' nie zwrócił nazwy")

                snapshot_local = _download_collection_snapshot(
                    collection_name=name,
                    snapshot_name=snapshot_name,
                    scratch_dir=scratch_path / name,
                )

                snapshot_arcname = pathlib.Path("snapshots") / name / snapshot_local.name
                tar.add(snapshot_local, arcname=str(snapshot_arcname))

                try:
                    _delete_remote_snapshot(name, snapshot_name)
                except Exception:
                    logger.debug("Nie udało się usunąć snapshotu '%s' dla '%s' po pobraniu", snapshot_name, name)

                metadata_entry = {
                    "name": name,
                    "role": collection_roles[name]["role"],
                    "alias": collection_roles[name]["alias"],
                    "snapshot": str(snapshot_arcname),
                    "points_estimate": points_count,
                    "snapshot_name": snapshot_name,
                    "snapshot_size": _extract_snapshot_size(snapshot),
                }
                metadata["collections"].append(metadata_entry)

            for relative in vector_files:
                source = vector_store_dir / relative
                tar.add(source, arcname=str(pathlib.Path("vector_store") / relative))

            metadata["vector_store"] = {
                "base": str(vector_store_dir),
                "files": vector_files,
            }

            meta_bytes = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
            meta_info = tarfile.TarInfo(name="metadata.json")
            meta_info.size = len(meta_bytes)
            meta_info.mtime = int(time.time())
            tar.addfile(meta_info, io.BytesIO(meta_bytes))

    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    filename = f"qdrant-export-{stamp}.tar.gz"
    return bundle_buffer.getvalue(), {
        "filename": filename,
        "collections": selected_names,
        "count": len(selected_names),
        "vector_store_files": vector_files,
        "snapshots": [
            entry.get("snapshot_name")
            for entry in metadata.get("collections", [])
            if entry.get("snapshot_name")
        ],
    }


# Create a snapshot of a collection using client or REST fallback.
def _create_collection_snapshot(collection_name: str):
    # Prefer client method if available; otherwise use REST
    method = getattr(qdrant, "create_snapshot", None)
    if method is not None:
        try:
            try:
                return method(collection_name=collection_name, wait=True)
            except TypeError:
                return method(collection_name)
        except Exception as exc:
            logger.debug("Client snapshot create failed, fallback to REST: %s", exc)
    # REST fallback
    base = settings.qdrant_url.rstrip("/")
    url = f"{base}/collections/{_urlparse.quote(collection_name)}/snapshots"
    data = json.dumps({}).encode("utf-8")
    req = _urlreq.Request(url, data=data, method="POST", headers=_qdrant_headers_json())
    try:
        with _urlreq.urlopen(req, timeout=settings.qdrant_request_timeout) as resp:
            body = resp.read().decode("utf-8")
            try:
                return json.loads(body)
            except Exception:
                return {"name": None, "raw": body}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Nie udało się utworzyć snapshotu kolekcji '{collection_name}': {exc}") from exc


# Extract snapshot filename/id from mixed client/REST responses.
def _extract_snapshot_name(snapshot: Any) -> Optional[str]:
    if snapshot is None:
        return None
    if isinstance(snapshot, dict):
        result = snapshot.get("result")
        if isinstance(result, dict):
            return _extract_snapshot_name(result)
        for key in ("name", "snapshot_name", "id"):
            candidate = snapshot.get(key)
            if candidate:
                return candidate
        return None
    return next(
        (
            getattr(snapshot, attr)
            for attr in ("name", "snapshot_name", "id")
            if getattr(snapshot, attr, None)
        ),
        None,
    )


# Extract snapshot size in bytes from mixed client/REST responses.
def _extract_snapshot_size(snapshot: Any) -> Optional[int]:
    if snapshot is None:
        return None
    if isinstance(snapshot, dict):
        result = snapshot.get("result")
        if isinstance(result, dict):
            return _extract_snapshot_size(result)
        return snapshot.get("size") or snapshot.get("size_bytes")
    return getattr(snapshot, "size", None) or getattr(snapshot, "size_bytes", None)


# Download a snapshot file for a collection to a temporary/scratch dir.
def _download_collection_snapshot(
    *,
    collection_name: str,
    snapshot_name: str,
    scratch_dir: pathlib.Path,
) -> pathlib.Path:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    target_path = scratch_dir / snapshot_name
    # REST download with ?download=true
    base = settings.qdrant_url.rstrip("/")
    quoted = _urlparse.quote
    url = f"{base}/collections/{quoted(collection_name)}/snapshots/{quoted(snapshot_name)}?download=true"
    req = _urlreq.Request(url, method="GET", headers=_qdrant_headers_binary())
    try:
        with _urlreq.urlopen(req, timeout=settings.qdrant_request_timeout) as resp, target_path.open("wb") as out:
            shutil.copyfileobj(resp, out)
    except Exception as exc:
        # Try without query param
        url = f"{base}/collections/{quoted(collection_name)}/snapshots/{quoted(snapshot_name)}"
        req = _urlreq.Request(url, method="GET", headers=_qdrant_headers_binary())
        try:
            with _urlreq.urlopen(req, timeout=settings.qdrant_request_timeout) as resp, target_path.open("wb") as out:
                shutil.copyfileobj(resp, out)
        except Exception as exc2:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Pobranie snapshotu '{snapshot_name}' kolekcji '{collection_name}' nie powiodło się: {exc2}"
                ),
            ) from exc2
    if target_path.stat().st_size <= 0:
        raise HTTPException(status_code=502, detail=f"Snapshot '{snapshot_name}' jest pusty po pobraniu")
    return target_path


# Backwards-compatible helper to fetch a snapshot to destination.
def _attempt_download_snapshot(
    collection_name: str,
    snapshot_name: str,
    destination: pathlib.Path,
    *,
    expect_file: bool = True,
) -> bool:
    # Kept for backward compatibility; now delegates to REST
    try:
        tmp = _download_collection_snapshot(
            collection_name=collection_name,
            snapshot_name=snapshot_name,
            scratch_dir=destination if destination.is_dir() else destination.parent,
        )
        if destination.is_dir():
            return tmp.exists()
        if tmp != destination:
            shutil.move(str(tmp), str(destination))
        return destination.exists() and destination.stat().st_size > 0
    except Exception as exc:
        if expect_file:
            logger.debug(
                "REST download snapshot failed for '%s': %s",
                snapshot_name,
                exc,
            )
        return False


# Delete a remote snapshot using client or REST fallback.
def _delete_remote_snapshot(collection_name: str, snapshot_name: str) -> None:
    # Try client methods
    for name in ("delete_snapshot", "delete_collection_snapshot"):
        method = getattr(qdrant, name, None)
        if method is None:
            continue
        try:
            try:
                method(collection_name=collection_name, snapshot_name=snapshot_name)
            except TypeError:
                method(collection_name, snapshot_name)
            return
        except Exception as exc:
            logger.debug("Client delete snapshot failed, trying REST: %s", exc)
            break
    # REST fallback
    base = settings.qdrant_url.rstrip("/")
    url = f"{base}/collections/{_urlparse.quote(collection_name)}/snapshots/{_urlparse.quote(snapshot_name)}"
    req = _urlreq.Request(url, method="DELETE", headers=_qdrant_headers_json())
    try:
        with _urlreq.urlopen(req, timeout=settings.qdrant_request_timeout) as _:
            return
    except Exception as exc:
        logger.debug("REST delete snapshot failed: %s", exc)
        return


def _apply_collection_aliases(alias_targets: Dict[str, str]) -> None:
    actions = []
    current_aliases = _get_qdrant_alias_targets()
    for alias_name, target in alias_targets.items():
        current_target = current_aliases.get(alias_name)
        if current_target == target:
            continue
        if current_target:
            actions.append({"delete_alias": {"alias_name": alias_name}})
        else:
            try:
                qdrant.get_collection(alias_name)
                actions.append({"delete_alias": {"alias_name": alias_name}})
            except Exception:
                pass
        actions.append({"create_alias": {"collection_name": target, "alias_name": alias_name}})

    if not actions:
        return

    base_url = settings.qdrant_url.rstrip("/")
    url = f"{base_url}/collections/aliases"
    payload = {"actions": actions}
    data = json.dumps(payload).encode("utf-8")
    req = _urlreq.Request(url, data=data, method="POST", headers=_qdrant_headers_json())
    try:
        with _urlreq.urlopen(req, timeout=settings.qdrant_request_timeout) as resp:
            body = resp.read().decode("utf-8", "ignore")
            try:
                parsed = json.loads(body) if body.strip() else {}
            except Exception:
                parsed = {}
            status = parsed.get("status") or parsed.get("result", {}).get("status")
            if isinstance(status, str) and status.lower() not in {"ok", "acknowledged", "success"}:
                logger.warning("Qdrant alias swap returned status=%s body=%s", status, body[:500])
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Nie udało się przełączyć aliasów kolekcji: {exc}",
        ) from exc


def swap_collection_aliases(
    logical_base: Optional[str],
    new_collection_base: str,
) -> None:
    """Atomically repoint summary/content aliases to a new physical pair.

    logical_base:
        Base name used for deriving alias names. For the default corpus this
        should be None so that aliases align with settings.qdrant_*_collection.
    new_collection_base:
        Base name used for deriving the new physical collections that should
        become the active pair after the swap.
    """
    summary_alias, content_alias = derive_alias_names(logical_base)
    new_summary, new_content = derive_collection_names(new_collection_base)
    try:
        _apply_collection_aliases({summary_alias: new_summary, content_alias: new_content})
    except HTTPException as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=(
                f"Nie udało się przełączyć aliasów kolekcji "
                f"(base={logical_base}, new_base={new_collection_base}): {exc.detail}"
            ),
        ) from exc


def _collection_role_targets(collections: List[Any]) -> Dict[str, str]:
    roles: Dict[str, str] = {}
    for entry in collections:
        if not isinstance(entry, dict):
            raise HTTPException(status_code=400, detail="Zły format archiwum: wpis kolekcji musi być obiektem")
        role = entry.get("role")
        name = entry.get("name")
        if role in {"summary", "content"} and isinstance(name, str) and name:
            if role in roles and roles[role] != name:
                raise HTTPException(status_code=400, detail=f"Archiwum zawiera wiele kolekcji dla roli '{role}'")
            roles[role] = name
    return roles


def _validate_project_bundle_metadata(metadata: Dict[str, Any], collections: List[Any]) -> Dict[str, str]:
    summary_alias, content_alias = derive_alias_names(None)
    summary_base, content_base = derive_collection_names(None)
    names = [
        str((entry or {}).get("name"))
        for entry in collections
        if isinstance(entry, dict) and (entry or {}).get("name")
    ]
    unique_names = set(names)
    if len(names) != 2 or len(unique_names) != 2:
        raise HTTPException(
            status_code=400,
            detail="Archiwum musi zawierać dokładnie aktywną parę kolekcji projektu (summary/content)",
        )

    project = metadata.get("project")
    if isinstance(project, dict) and project:
        project_name = project.get("collection_name")
        if project_name != settings.collection_name:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Archiwum jest dla projektu '{project_name}', a bieżący projekt to "
                    f"'{settings.collection_name}'"
                ),
            )

        roles = _collection_role_targets(collections)
        if set(roles) != {"summary", "content"}:
            raise HTTPException(
                status_code=400,
                detail="Archiwum projektu musi zawierać role kolekcji 'summary' i 'content'",
            )

        aliases = project.get("aliases") if isinstance(project.get("aliases"), dict) else {}
        expected_alias_map = {
            summary_alias: roles["summary"],
            content_alias: roles["content"],
        }
        if aliases:
            for alias_name, target in expected_alias_map.items():
                if aliases.get(alias_name) != target:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Metadata archiwum nie zgadza się z kolekcją docelową aliasu '{alias_name}'",
                    )

        for entry in collections:
            alias_name = entry.get("alias") if isinstance(entry, dict) else None
            role = entry.get("role") if isinstance(entry, dict) else None
            if role == "summary" and alias_name and alias_name != summary_alias:
                raise HTTPException(status_code=400, detail=f"Nieoczekiwany alias kolekcji summary: {alias_name}")
            if role == "content" and alias_name and alias_name != content_alias:
                raise HTTPException(status_code=400, detail=f"Nieoczekiwany alias kolekcji content: {alias_name}")

        return expected_alias_map

    legacy_expected = {summary_base, content_base}
    if unique_names != legacy_expected:
        raise HTTPException(
            status_code=400,
            detail=(
                "Archiwum nie zawiera metadata projektu i nie pasuje do bazowej pary "
                f"kolekcji bieżącego projektu: {', '.join(sorted(legacy_expected))}"
            ),
        )
    return {summary_alias: summary_base, content_alias: content_base}


def import_collections_bundle(bundle: bytes, *, replace_existing: bool = True) -> Dict[str, Any]:
    # Restore collections and TF‑IDF indices from a tar.gz bundle.

    try:
        tar_context = tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz")
    except tarfile.TarError as exc:
        raise HTTPException(status_code=400, detail=f"Nie udało się odczytać archiwum: {exc}") from exc

    with tar_context as tar:
        restored = _extract_json_member(tar, "metadata.json")
        if not isinstance(restored, dict):
            raise HTTPException(status_code=400, detail="Plik metadata.json ma nieprawidłowy format")

        collections = restored.get("collections") or []
        if not isinstance(collections, list):
            raise HTTPException(status_code=400, detail="Zły format archiwum: pole 'collections' musi być listą")
        alias_targets = _validate_project_bundle_metadata(restored, collections)

        summary: Dict[str, Any] = {
            "restored": [],
            "skipped": [],
            "errors": [],
            "meta": restored.get("meta") or {},
            "project": restored.get("project") or {"collection_name": settings.collection_name},
            "aliases": {},
            "vector_store": {"restored": [], "base": str(settings.vector_store_dir)},
        }

        for entry in collections:
            name = (entry or {}).get("name")
            if not name:
                summary["errors"].append({"collection": None, "error": "Brak nazwy kolekcji w wpisie"})
                continue

            try:
                snapshot_ref = entry.get("snapshot")
                if snapshot_ref:
                    restored = _restore_collection_from_snapshot(
                        tar,
                        snapshot_ref,
                        collection_name=name,
                        replace_existing=replace_existing,
                    )
                    if restored:
                        summary["restored"].append(name)
                    else:
                        summary["skipped"].append(name)
                    continue

                config_ref = entry.get("config")
                indexes_ref = entry.get("indexes")
                points_ref = entry.get("points")

                if isinstance(config_ref, dict):
                    config = config_ref
                else:
                    config = _extract_json_member(tar, config_ref)
                if not isinstance(config, dict):
                    raise HTTPException(status_code=400, detail=f"Konfiguracja kolekcji '{name}' ma nieprawidłowy format")

                if isinstance(indexes_ref, list):
                    indexes = indexes_ref
                elif indexes_ref:
                    indexes = _extract_json_member(tar, indexes_ref)
                else:
                    indexes = []
                if indexes and not isinstance(indexes, list):
                    raise HTTPException(status_code=400, detail=f"Lista indeksów kolekcji '{name}' ma nieprawidłowy format")

                _prepare_collection(name, config, indexes, replace_existing=replace_existing)
                if isinstance(points_ref, list):
                    _restore_points_from_list(points_ref, collection_name=name)
                else:
                    _restore_points_from_tar(tar, points_ref, collection_name=name)
                summary["restored"].append(name)
            except HTTPException as exc:
                summary["errors"].append({"collection": name, "error": exc.detail})
            except Exception as exc:  # pragma: no cover - depends on remote behaviour
                logger.exception("Nie udało się odtworzyć kolekcji '%s'", name)
                summary["errors"].append({"collection": name, "error": str(exc)})

        vector_dir = settings.vector_store_dir
        vector_dir.mkdir(parents=True, exist_ok=True)
        if replace_existing:
            try:
                _clear_dir_contents(vector_dir)
            except Exception as exc:
                logger.warning("Nie udało się wyczyścić zawartości vector_store '%s': %s", vector_dir, exc)

        for member in tar.getmembers():
            if not member.isfile():
                continue
            if not member.name.startswith("vector_store/"):
                continue
            relative_path = pathlib.Path(member.name).relative_to("vector_store")
            destination = vector_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with tar.extractfile(member) as src, destination.open("wb") as dst:
                    if src is None:
                        raise ValueError(f"Brak danych dla pliku {member.name}")
                    shutil.copyfileobj(src, dst)
                summary["vector_store"]["restored"].append(str(relative_path))
            except Exception as exc:
                logger.error("Nie udało się odtworzyć pliku indeksu '%s': %s", member.name, exc)
                summary["errors"].append({"collection": None, "error": f"vector_store:{member.name}: {exc}"})

        if not summary["errors"]:
            try:
                _apply_collection_aliases(alias_targets)
                summary["aliases"] = alias_targets
            except HTTPException as exc:
                summary["errors"].append({"collection": None, "error": exc.detail})

    return summary


def _restore_collection_from_snapshot(
    tar: tarfile.TarFile,
    snapshot_member: str,
    *,
    collection_name: str,
    replace_existing: bool,
) -> bool:
    if not replace_existing and _collection_exists(collection_name):
        return False

    if replace_existing:
        _drop_collection_if_exists(collection_name)

    snapshot_path = _extract_tar_member_to_tempfile(tar, snapshot_member)
    snapshot_filename = pathlib.Path(snapshot_member).name

    try:
        _upload_snapshot_file(collection_name, snapshot_filename, snapshot_path)
        _recover_uploaded_snapshot(collection_name, snapshot_filename)
    finally:
        try:
            snapshot_path.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.debug(
                "Nie udało się usunąć pliku tymczasowego snapshotu '%s': %s",
                snapshot_path,
                exc,
            )

    return True


def _collection_exists(collection_name: str) -> bool:
    try:
        qdrant.get_collection(collection_name)
        return True
    except Exception:
        return False


def _drop_collection_if_exists(collection_name: str) -> None:
    try:
        qdrant.delete_collection(collection_name)
    except UnexpectedResponse as exc:
        if getattr(exc, "status_code", None) != 404 and "not found" not in str(exc).lower():
            raise HTTPException(
                status_code=502,
                detail=f"Usunięcie kolekcji '{collection_name}' nie powiodło się: {exc}",
            ) from exc
    except Exception as exc:
        err_text = str(exc).lower()
        if "not found" not in err_text:
            raise HTTPException(
                status_code=502,
                detail=f"Usunięcie kolekcji '{collection_name}' nie powiodło się: {exc}",
            ) from exc


def _extract_tar_member_to_tempfile(tar: tarfile.TarFile, member_name: str) -> pathlib.Path:
    try:
        member = tar.getmember(member_name)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Brak pliku '{member_name}' w archiwum") from exc

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_path = pathlib.Path(temp_file.name)
    with temp_file as dst:
        with tar.extractfile(member) as src:
            if src is None:
                raise HTTPException(status_code=400, detail=f"Brak danych snapshotu '{member_name}'")
            shutil.copyfileobj(src, dst)
    if temp_path.stat().st_size == 0:
        raise HTTPException(status_code=400, detail=f"Snapshot '{member_name}' jest pusty")
    return temp_path


# Upload a snapshot via client (if available) or REST multipart fallback.
def _upload_snapshot_file(collection_name: str, snapshot_name: str, path: pathlib.Path) -> None:
    # Try client method if present
    method = getattr(qdrant, "upload_snapshot", None)
    if method is not None:
        try:
            try:
                method(collection_name=collection_name, snapshot_name=snapshot_name, snapshot_path=str(path))
            except TypeError:
                try:
                    method(collection_name=collection_name, snapshot_path=str(path))
                except TypeError:
                    method(collection_name, str(path))
            return
        except Exception as exc:
            logger.debug("Client upload snapshot failed, fallback to REST: %s", exc)
    # REST multipart upload: POST /collections/{collection}/snapshots/upload?wait=true
    base = settings.qdrant_url.rstrip("/")
    url = f"{base}/collections/{_urlparse.quote(collection_name)}/snapshots/upload?wait=true"
    boundary = f"----qdrantBoundary{uuid.uuid4().hex}"
    body = _build_multipart(boundary, {
        "snapshot": (snapshot_name, path.read_bytes(), "application/octet-stream"),
    })
    headers = _qdrant_headers_multipart(boundary)
    req = _urlreq.Request(url, data=body, headers=headers, method="POST")
    try:
        with _urlreq.urlopen(req, timeout=settings.qdrant_request_timeout) as resp:
            _ = resp.read()
            return
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Upload snapshotu '{snapshot_name}' dla '{collection_name}' (REST) nie powiódł się: {exc}",
        ) from exc


def _recover_uploaded_snapshot(collection_name: str, snapshot_name: str) -> None:
    # For REST upload endpoint, recovery happens automatically; this is a no‑op.
    method = getattr(qdrant, "recover_snapshot", None)
    if method is None:
        return
    try:
        try:
            method(collection_name=collection_name, snapshot_name=snapshot_name)
        except TypeError:
            method(collection_name, snapshot_name)
    except Exception:
        # Ignore if unsupported; upload already performed recovery.
        return


def _qdrant_headers_json() -> Dict[str, str]:
    # Build JSON headers; include API key when configured.
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if settings.qdrant_api_key:
        headers["api-key"] = settings.qdrant_api_key
    return headers


def _qdrant_headers_binary() -> Dict[str, str]:
    # Build headers for binary transfer; include API key when configured.
    headers = {}
    if settings.qdrant_api_key:
        headers["api-key"] = settings.qdrant_api_key
    return headers


def _qdrant_headers_multipart(boundary: str) -> Dict[str, str]:
    # Build multipart/form-data headers; include API key when configured.
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if settings.qdrant_api_key:
        headers["api-key"] = settings.qdrant_api_key
    return headers


def _build_multipart(boundary: str, files: Dict[str, tuple]) -> bytes:
    # Construct a minimal multipart body for file uploads.
    # files: name -> (filename, content_bytes, content_type)
    lines: List[bytes] = []
    b = boundary
    for field, (filename, content, ctype) in files.items():
        lines.append((f"--{b}\r\n").encode("utf-8"))
        disp = f"Content-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\n"
        lines.append(disp.encode("utf-8"))
        lines.append((f"Content-Type: {ctype}\r\n\r\n").encode("utf-8"))
        lines.append(content)
        lines.append(b"\r\n")
    lines.append((f"--{b}--\r\n").encode("utf-8"))
    return b"".join(lines)


def _prepare_collection(
    name: str,
    config: Dict[str, Any],
    indexes: List[Dict[str, Any]],
    *,
    replace_existing: bool,
) -> None:
    if replace_existing:
        try:
            qdrant.delete_collection(name)
        except UnexpectedResponse as exc:
            if getattr(exc, "status_code", None) != 404 and "not found" not in str(exc).lower():
                raise HTTPException(status_code=502, detail=f"Usunięcie kolekcji '{name}' nie powiodło się: {exc}") from exc
        except Exception as exc:
            err_text = str(exc).lower()
            if "not found" not in err_text:
                raise HTTPException(status_code=502, detail=f"Usunięcie kolekcji '{name}' nie powiodło się: {exc}") from exc
    else:
        try:
            qdrant.get_collection(name)
            return  # Kolekcja istnieje — pomijamy, aby nie nadpisywać
        except Exception:
            pass

    params = (config or {}).get("params") or {}
    vectors_config = _inflate_vectors_config(params.get("vectors"))
    sparse_vectors_config = _inflate_sparse_vectors_config(params.get("sparse_vectors"))

    kwargs: Dict[str, Any] = {}
    if vectors_config is not None:
        kwargs["vectors_config"] = vectors_config
    if sparse_vectors_config is not None:
        kwargs["sparse_vectors_config"] = sparse_vectors_config

    for field in ("shard_number", "replication_factor", "write_consistency_factor", "on_disk_payload"):
        value = params.get(field)
        if value is not None:
            kwargs[field] = value

    if params.get("hnsw_config"):
        kwargs["hnsw_config"] = qm.HnswConfigDiff(**params["hnsw_config"])
    if params.get("optimizer_config"):
        kwargs["optimizers_config"] = qm.OptimizersConfigDiff(**params["optimizer_config"])
    if params.get("wal_config"):
        kwargs["wal_config"] = qm.WalConfigDiff(**params["wal_config"])

    try:
        qdrant.recreate_collection(collection_name=name, **kwargs)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Nie udało się utworzyć kolekcji '{name}': {exc}") from exc

    for index in indexes or []:
        if not isinstance(index, dict):
            continue
        field_name = index.get("field_name")
        params_dict = index.get("params") or index.get("field_schema") or {}
        if not field_name or not isinstance(params_dict, dict):
            continue
        try:
            qdrant.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=qm.PayloadIndexParams(**params_dict),
            )
        except Exception:
            logger.debug("Nie udało się odtworzyć indeksu pola '%s' w '%s'", field_name, name)


def _restore_points_from_tar(
    tar: tarfile.TarFile,
    member_name: Optional[str],
    *,
    collection_name: str,
    batch_size: int = 256,
) -> None:
    if not member_name:
        raise HTTPException(status_code=400, detail=f"Brak ścieżki do punktów dla kolekcji '{collection_name}'")
    try:
        member = tar.getmember(member_name)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Brak pliku '{member_name}' w archiwum") from exc

    with tar.extractfile(member) as fh:
        if fh is None:
            raise HTTPException(status_code=400, detail=f"Brak danych punktów dla '{collection_name}'")
        batch: List[qm.PointStruct] = []
        for raw_line in fh:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            try:
                point_data = json.loads(line)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Błędny JSON w '{member_name}': {exc}") from exc
            point_struct = _point_struct_from_dump(point_data)
            batch.append(point_struct)
            if len(batch) >= batch_size:
                _flush_points(collection_name, batch)
        if batch:
            _flush_points(collection_name, batch)


def _restore_points_from_list(
    points: Iterable[Dict[str, Any]],
    *,
    collection_name: str,
    batch_size: int = 256,
) -> None:
    batch: List[qm.PointStruct] = []
    for point in points:
        point_struct = _point_struct_from_dump(point)
        batch.append(point_struct)
        if len(batch) >= batch_size:
            _flush_points(collection_name, batch)
    if batch:
        _flush_points(collection_name, batch)


def _flush_points(collection_name: str, batch: List[qm.PointStruct]) -> None:
    try:
        qdrant.upsert(collection_name=collection_name, wait=True, points=batch)
    finally:
        batch.clear()


def _extract_json_member(tar: tarfile.TarFile, member_name: Optional[str]):
    if not member_name:
        raise HTTPException(status_code=400, detail="Brak referencji do pliku w archiwum")
    try:
        member = tar.getmember(member_name)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Brak pliku '{member_name}' w archiwum") from exc
    with tar.extractfile(member) as fh:
        if fh is None:
            raise HTTPException(status_code=400, detail=f"Brak danych dla '{member_name}'")
        raw = fh.read().decode("utf-8")
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Nie udało się zinterpretować JSON z '{member_name}': {exc}") from exc


def _inflate_vectors_config(raw: Optional[Dict[str, Any]]):
    if raw is None:
        return None
    if isinstance(raw, dict) and raw.get("size") is not None:
        return qm.VectorParams(**raw)
    mapping = raw
    if isinstance(raw, dict) and "map" in raw:
        mapping = raw.get("map") or {}
    result = {}
    for key, value in (mapping or {}).items():
        if isinstance(value, dict):
            result[key] = qm.VectorParams(**value)
    return result or None


def _inflate_sparse_vectors_config(raw: Optional[Dict[str, Any]]):
    if raw is None:
        return None
    mapping = raw
    if isinstance(raw, dict) and "map" in raw:
        mapping = raw.get("map") or {}
    result = {}
    for key, value in (mapping or {}).items():
        params = value or {}
        result[key] = qm.SparseVectorParams(**params)
    return result or None


def _point_struct_from_dump(point: Dict[str, Any]) -> qm.PointStruct:
    if not isinstance(point, dict):
        raise HTTPException(status_code=400, detail="Zły format wpisu punktu w archiwum")
    try:
        return qm.PointStruct.model_validate(point)
    except Exception:
        data: Dict[str, Any] = {
            "id": point.get("id"),
            "payload": point.get("payload"),
        }
        for key in ("vector", "vectors", "sparse_vector", "sparse_vectors"):
            if key in point:
                data[key] = point[key]
        return qm.PointStruct(**data)


def build_and_upsert_points(
    doc_records: Iterable[Dict[str, Any]],
    content_vec,
    summary_vec,
    *,
    enable_sparse: bool,
    collection_base: Optional[str],
) -> int:
    summary_collection, content_collection = derive_collection_names(collection_base)
    batch_limit = 256
    summary_points: List[qm.PointStruct] = []
    content_points: List[qm.PointStruct] = []
    point_count = 0
    replacement_relations: Dict[str, List[str]] = {}
    doc_paths: Dict[str, str] = {}
    doc_titles: Dict[str, str] = {}
    seen_subtitle_norms: Set[str] = set()
    now_ts = int(time.time())

    for rec in doc_records:
        doc_id = rec["doc_id"]
        path = rec["path"]
        doc_url = rec.get("doc_url")
        chunks = rec["chunks"]
        doc_summary = rec["doc_summary"]
        doc_signature = rec["doc_signature"]
        doc_entities_raw = rec.get("doc_entities", [])
        if isinstance(doc_entities_raw, (list, tuple)):
            doc_entities = [str(x).strip() for x in doc_entities_raw if str(x).strip()]
        elif isinstance(doc_entities_raw, str):
            # Backward-compat: accept legacy string, split to tokens on commas/semicolons/newlines
            parts = re.split(r",|;|\n", doc_entities_raw)
            doc_entities = [p.strip() for p in parts if p.strip()]
        else:
            doc_entities = []
        summary_sparse_text = rec["summary_sparse_text"]
        doc_date_val = str(rec.get("doc_date", "") or "").strip()
        replacement_info = str(rec.get("replacement", "") or "brak").strip() or "brak"
        if replacement_info.lower() == "brak":
            replacement_info = "brak"
        doc_title = str(rec.get("doc_title") or rec.get("title") or "").strip()
        subtitle_raw = str(rec.get("subtitle", "") or "").strip()
        subtitle = subtitle_raw if subtitle_raw else "brak"
        subtitle_norm = _normalize_subtitle(subtitle)
        if not doc_title:
            try:
                doc_title = pathlib.Path(path).stem
            except Exception:
                doc_title = ""

        doc_paths[doc_id] = path
        if doc_title:
            doc_titles[doc_id] = doc_title
        if subtitle_norm and subtitle_norm != "brak":
            seen_subtitle_norms.add(subtitle_norm)
        normalized_refs = _parse_replacement_list(replacement_info)
        if normalized_refs:
            replacement_relations[doc_id] = normalized_refs

        # Prefer precomputed dense summary vector from sidecar cache when available
        cached_vec = rec.get("summary_dense_vec") if isinstance(rec, dict) else None
        if isinstance(cached_vec, list) and cached_vec:
            summary_dense_vec = cached_vec
        else:
            summary_dense_vec = embed_passage([doc_summary])[0]
        content_texts = [c.get("text", c) if isinstance(c, dict) else str(c) for c in chunks]
        try:
            content_vecs = embed_passage(content_texts)
        except BadRequestError as exc:
            # Diagnostic: log which chunk candidates look suspicious
            try:
                stats = []
                for i, t in enumerate(content_texts):
                    s = str(t or "")
                    stats.append((len(s), i, s[:120].replace("\n", " ")))
                stats.sort(reverse=True)  # longest first
                sample = [
                    {"i": i, "char_len": ln, "head": head}
                    for (ln, i, head) in stats[:10]
                ]
                logger.error(
                    "Embedding content batch failed | doc_id=%s path=%s chunks=%d top_by_len=%s error=%s",
                    doc_id,
                    path,
                    len(content_texts),
                    json.dumps(sample, ensure_ascii=False),
                    exc,
                )
            except Exception:
                logger.error(
                    "Embedding content batch failed | doc_id=%s path=%s error=%s",
                    doc_id,
                    path,
                    exc,
                )
            raise

        if enable_sparse:
            sparse_chunks = tfidf_vector(content_texts, content_vec)
            if summary_vec is not None and summary_sparse_text:
                summary_sparse = tfidf_vector(
                    [summary_sparse_text], summary_vec, path=SUMMARY_VECTORIZER_PATH
                )[0]
            else:
                summary_sparse = ([], [])
        else:
            sparse_chunks = [([], []) for _ in chunks]
            summary_sparse = ([], [])

        is_active_flag = bool(rec.get("is_active", True))
        summary_payload: Dict[str, Any] = {
            "doc_id": doc_id,
            "path": path,
            "is_active": is_active_flag,
            "point_type": "summary",
            "title": doc_title,
            "subtitle": subtitle,
            "subtitle_norm": subtitle_norm,
            "summary": doc_summary,
            "signature": doc_signature,
            "entities": doc_entities,
        }
        if isinstance(doc_url, str) and doc_url.strip():
            summary_payload["doc_url"] = doc_url.strip()
        # propagate content hash if available in record (used for dedupe)
        try:
            csha = str(rec.get("content_sha256"))
            if csha:
                summary_payload["content_sha256"] = csha
        except Exception:
            pass
        summary_payload["replacement"] = replacement_info
        if doc_date_val:
            summary_payload["doc_date"] = doc_date_val
            summary_payload["doc_date_ord"] = _doc_date_ord(doc_date_val)
        summary_payload["ingested_ts"] = now_ts
        summary_vectors: Dict[str, Any] = {
            SUMMARY_VECTOR_NAME: summary_dense_vec,
        }
        if SPARSE_ENABLED and enable_sparse and summary_sparse[0]:
            summary_vectors[SUMMARY_SPARSE_NAME] = qm.SparseVector(
                indices=summary_sparse[0], values=summary_sparse[1]
            )
            summary_payload["summary_sparse_indices"] = summary_sparse[0]
            summary_payload["summary_sparse_values"] = summary_sparse[1]

        summary_pid = int(str(int(sha1(f"{doc_id}:summary")[0:12], 16))[:12])
        summary_points.append(
            qm.PointStruct(id=summary_pid, vector=summary_vectors, payload=summary_payload)
        )
        point_count += 1
        if len(summary_points) >= batch_limit:
            qdrant.upsert(collection_name=summary_collection, points=summary_points)
            summary_points = []

        for i, chunk_item in enumerate(chunks):
            if isinstance(chunk_item, dict):
                chunk_text_val = chunk_item.get("text", "")
                section_path = chunk_item.get("section_path")
                section_path_prefixes = chunk_item.get("section_path_prefixes")
            else:
                chunk_text_val = str(chunk_item)
                section_path = None
                section_path_prefixes = None
            pid = int(str(int(sha1(f"{doc_id}:{i}")[0:12], 16))[:12])
            payload: Dict[str, Any] = {
                "doc_id": doc_id,
                "path": path,
                "chunk_id": i,
                "is_active": is_active_flag,
                "point_type": "chunk",
                "text": chunk_text_val,
            }
            # Replicate document-level entities at chunk-level payload to enable entity filters
            if doc_entities:
                payload["entities"] = doc_entities
            if isinstance(section_path, str) and section_path.strip():
                payload["section_path"] = section_path.strip()
            if isinstance(section_path_prefixes, (list, tuple)) and section_path_prefixes:
                payload["section_path_prefixes"] = [
                    str(prefix).strip() for prefix in section_path_prefixes if str(prefix).strip()
                ]
            # No per-chunk section level stored (kept minimal)

            vectors: Dict[str, Any] = {
                CONTENT_VECTOR_NAME: content_vecs[i],
            }

            if SPARSE_ENABLED and enable_sparse:
                indices, values = sparse_chunks[i]
                if indices:
                    vectors[CONTENT_SPARSE_NAME] = qm.SparseVector(indices=indices, values=values)
                    payload["content_sparse_indices"] = indices
                    payload["content_sparse_values"] = values

            content_points.append(qm.PointStruct(id=pid, vector=vectors, payload=payload))
            point_count += 1
            if len(content_points) >= batch_limit:
                qdrant.upsert(collection_name=content_collection, points=content_points)
                content_points = []

    if summary_points:
        qdrant.upsert(collection_name=summary_collection, points=summary_points)
    if content_points:
        qdrant.upsert(collection_name=content_collection, points=content_points)

    # Aggregate deactivation decisions from all rules to ensure additivity (only 1->0 changes).
    deactivate_total: Set[str] = set()
    # Replacement-based deactivations
    rep_deactivate, rep_keep_active = _decisions_replacement(
        replacement_relations, doc_paths, doc_titles
    )
    deactivate_total |= rep_deactivate
    # Per-rule preview log (replacement)
    if rep_deactivate:
        try:
            curr_rep = _fetch_is_active_for_docs(summary_collection, list(rep_deactivate))
            rep_changed = sum(1 for did in rep_deactivate if bool(curr_rep.get(did)) is True)
        except Exception:
            rep_changed = 0
        logger.info(
            "Replacement decisions | deactivate=%d changed_if_applied=%d",
            len(rep_deactivate),
            rep_changed,
        )
    # Subtitle-conflict deactivations
    sub_deactivate, sub_keep_active, sub_groups = _decisions_subtitle(
        summary_collection, seen_subtitle_norms
    ) if seen_subtitle_norms else (set(), set(), 0)
    deactivate_total |= sub_deactivate
    # Per-rule preview log (subtitle)
    if sub_deactivate or sub_groups:
        try:
            curr_sub = _fetch_is_active_for_docs(summary_collection, list(sub_deactivate)) if sub_deactivate else {}
            sub_changed = sum(1 for did in sub_deactivate if bool(curr_sub.get(did)) is True)
        except Exception:
            sub_changed = 0
        logger.info(
            "Subtitle decisions | groups=%d deactivate=%d changed_if_applied=%d",
            int(sub_groups),
            len(sub_deactivate),
            sub_changed,
        )

    # Compute how many flags will actually change (1->0)
    changed = 0
    if deactivate_total:
        current = _fetch_is_active_for_docs(summary_collection, list(deactivate_total))
        changed = sum(1 for did in deactivate_total if bool(current.get(did)) is True)
        logger.info(
            "Aggregated is_active updates | deactivate=%d changed=%d (rep=%d, sub=%d, sub_groups=%d)",
            len(deactivate_total),
            changed,
            len(rep_deactivate),
            len(sub_deactivate),
            int(sub_groups),
        )
        for did in deactivate_total:
            _set_is_active_flag(summary_collection, content_collection, did, False)

    return point_count


def _parse_replacement_list(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    if text.lower() == "brak":
        return []
    parts = re.split(r"[,;\n]", text)
    items = [part.strip() for part in parts if part.strip()]
    return items


def _normalize_reference(value: str) -> str:
    norm = value.strip().lower()
    norm = norm.replace("\\", "/")
    norm = re.sub(r"\s+", " ", norm)
    return norm


def _apply_replacement_statuses(
    summary_collection: str,
    content_collection: str,
    replacements: Dict[str, List[str]],
    doc_paths: Dict[str, str],
    doc_titles: Optional[Dict[str, str]] = None,
):
    if not replacements:
        return

    name_to_doc: Dict[str, str] = {}
    titles = doc_titles or {}
    for doc_id, path in doc_paths.items():
        norm_path = _normalize_reference(path)
        name_to_doc[norm_path] = doc_id
        path_obj = pathlib.Path(path)
        name_to_doc[_normalize_reference(path_obj.name)] = doc_id
        name_to_doc[_normalize_reference(path_obj.stem)] = doc_id
        name_to_doc[doc_id] = doc_id
        title_val = titles.get(doc_id)
        if title_val:
            name_to_doc[_normalize_reference(title_val)] = doc_id

    deactivate: Set[str] = set()
    keep_active: Set[str] = set(replacements.keys())

    for replacer_id, refs in replacements.items():
        for ref in refs:
            norm_ref = _normalize_reference(ref)
            candidate = name_to_doc.get(norm_ref)
            if not candidate and norm_ref:
                # try unique substring match across known names
                matches = {
                    doc
                    for name, doc in name_to_doc.items()
                    if norm_ref in name and doc != replacer_id
                }
                if len(matches) == 1:
                    candidate = matches.pop()
            if candidate and candidate != replacer_id:
                deactivate.add(candidate)
            elif candidate is None:
                logger.debug(
                    "Replacement reference not matched | doc_id=%s reference=%s",
                    replacer_id,
                    ref,
                )

    # Summarize planned changes vs current state
    # Legacy method retained for potential callers; now only logs an intention summary.
    desired: Dict[str, bool] = {did: False for did in deactivate}
    for did in keep_active:
        if did not in deactivate:
            desired[did] = True
    if desired:
        current = _fetch_is_active_for_docs(summary_collection, list(desired.keys()))
        # Only count 1->0 as changes to keep semantics aligned with additivity
        changes = sum(1 for did, want in desired.items() if want is False and bool(current.get(did)) is True)
        logger.info(
            "Replacement decisions (preview) | deactivate=%d activate=%d changed_if_applied=%d",
            len(deactivate),
            len([d for d in keep_active if d not in deactivate]),
            changes,
        )
    # Do not mutate state here (aggregated application happens in build_and_upsert_points)


def _set_is_active_flag(
    summary_collection: str,
    content_collection: str,
    doc_id: str,
    is_active: bool,
):
    payload = {"is_active": is_active}
    for collection in (summary_collection, content_collection):
        filter_clause = qm.Filter(
            must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
        )
        point_ids = _collect_point_ids(collection, filter_clause)
        if not point_ids:
            continue
        try:
            qdrant.set_payload(
                collection_name=collection,
                payload=payload,
                points=point_ids,
            )
        except Exception as exc:
            logger.warning(
                "Failed to update is_active flag | collection=%s doc_id=%s error=%s",
                collection,
                doc_id,
                exc,
            )


def _collect_point_ids(collection: str, flt: qm.Filter) -> List[int]:
    ids: List[int] = []
    offset = None
    try:
        while True:
            res = qdrant.scroll(
                collection_name=collection,
                scroll_filter=flt,
                limit=256,
                offset=offset,
                with_payload=False,
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
            ids.extend([rec.id for rec in records if rec.id is not None])
            if offset is None:
                break
    except Exception as exc:
        logger.warning(
            "Failed to collect point ids | collection=%s error=%s",
            collection,
            exc,
        )
        return []
    return ids


def _fetch_is_active_for_docs(summary_collection: str, doc_ids: List[str]) -> Dict[str, Optional[bool]]:
    """Fetch current is_active flags for provided doc_ids from summaries collection.

    Returns a map doc_id -> is_active (or None when not found).
    """
    out: Dict[str, Optional[bool]] = {str(d): None for d in doc_ids}
    if not doc_ids:
        return out
    try:
        flt = qm.Filter(
            must=[
                qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=list(out.keys()))),
                qm.FieldCondition(key="point_type", match=qm.MatchValue(value="summary")),
            ]
        )
        offset = None
        while True:
            res = qdrant.scroll(
                collection_name=summary_collection,
                scroll_filter=flt,
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
                payload = getattr(rec, "payload", None) or {}
                did = str(payload.get("doc_id") or "")
                if did in out:
                    out[did] = bool(payload.get("is_active")) if payload.get("is_active") is not None else None
            if offset is None:
                break
    except Exception:
        return out
    return out


def _decisions_subtitle(
    summary_collection: str,
    subtitle_norms: Set[str],
) -> Tuple[Set[str], Set[str], int]:
    """Return (deactivate_set, keep_active_set, groups_count) for subtitle conflicts.

    Winner per group determined by doc_date_ord desc, then ingested_ts desc, then doc_id desc.
    No side‑effects here; application is aggregated by the caller.
    """
    if not subtitle_norms:
        return set(), set(), 0
    deactivate: Set[str] = set()
    keep_active: Set[str] = set()
    groups = 0
    for norm in subtitle_norms:
        if not norm or norm == "brak":
            continue
        groups += 1
        try:
            flt = qm.Filter(
                must=[
                    qm.FieldCondition(key="point_type", match=qm.MatchValue(value="summary")),
                    qm.FieldCondition(key="subtitle_norm", match=qm.MatchValue(value=norm)),
                ]
            )
            records: List[Any] = []
            offset = None
            while True:
                res = qdrant.scroll(
                    collection_name=summary_collection,
                    scroll_filter=flt,
                    limit=256,
                    offset=offset,
                    with_payload=["doc_id", "doc_date", "ingested_ts"],
                    with_vectors=False,
                )
                if isinstance(res, tuple):
                    page, offset = res
                else:
                    page = getattr(res, "points", None)
                    offset = getattr(res, "next_page_offset", None)
                    if page is None:
                        page = []
                if not page:
                    break
                records.extend(page)
                if offset is None:
                    break
            if not records:
                continue
            candidates: List[Tuple[int, int, str]] = []
            for rec in records:
                p = getattr(rec, "payload", None) or {}
                did = str(p.get("doc_id") or "")
                if not did:
                    continue
                dd = str(p.get("doc_date") or "").strip()
                dd_ord = _doc_date_ord(dd)
                its = 0
                try:
                    its = int(p.get("ingested_ts") or 0)
                except Exception:
                    its = 0
                candidates.append((dd_ord, its, did))
            if not candidates:
                continue
            candidates.sort(reverse=True)
            winner = candidates[0][2]
            keep_active.add(winner)
            for _, __, did in candidates[1:]:
                deactivate.add(did)
        except Exception as exc:
            logger.warning("Subtitle conflict scan failed | norm=%s error=%s", norm, exc)
    return deactivate, keep_active, groups


def _decisions_replacement(
    replacements: Dict[str, List[str]],
    doc_paths: Dict[str, str],
    doc_titles: Optional[Dict[str, str]] = None,
) -> Tuple[Set[str], Set[str]]:
    """Return (deactivate_set, keep_active_set) for replacement relations.

    No side‑effects here; application is aggregated by the caller.
    """
    if not replacements:
        return set(), set()
    name_to_doc: Dict[str, str] = {}
    titles = doc_titles or {}
    for doc_id, path in doc_paths.items():
        norm_path = _normalize_reference(path)
        name_to_doc[norm_path] = doc_id
        path_obj = pathlib.Path(path)
        name_to_doc[_normalize_reference(path_obj.name)] = doc_id
        name_to_doc[_normalize_reference(path_obj.stem)] = doc_id
        name_to_doc[doc_id] = doc_id
        title_val = titles.get(doc_id)
        if title_val:
            name_to_doc[_normalize_reference(title_val)] = doc_id

    deactivate: Set[str] = set()
    keep_active: Set[str] = set(replacements.keys())
    for replacer_id, refs in replacements.items():
        for ref in refs:
            norm_ref = _normalize_reference(ref)
            candidate = name_to_doc.get(norm_ref)
            if not candidate and norm_ref:
                matches = {
                    doc
                    for name, doc in name_to_doc.items()
                    if norm_ref in name and doc != replacer_id
                }
                if len(matches) == 1:
                    candidate = matches.pop()
            if candidate and candidate != replacer_id:
                deactivate.add(candidate)
    return deactivate, keep_active

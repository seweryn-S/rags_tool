import io
import json
import pathlib
import tarfile
import types
import unittest
from unittest.mock import patch

try:
    from fastapi import HTTPException
    from app import qdrant_utils as q
except ModuleNotFoundError as exc:  # pragma: no cover - local dev dependency guard
    HTTPException = None
    q = None
    MISSING_DEPENDENCY = f"Missing test dependency: {exc.name}"
else:
    MISSING_DEPENDENCY = ""


def _collections_response(names):
    return types.SimpleNamespace(
        collections=[types.SimpleNamespace(name=name) for name in names]
    )


@unittest.skipIf(q is None, MISSING_DEPENDENCY)
class ProjectCollectionExportImportTests(unittest.TestCase):
    def test_resolve_active_project_collections_uses_alias_targets(self):
        summary_alias, content_alias = q.derive_alias_names(None)
        summary_target = f"{q.settings.collection_name}_v20260515T120000_summaries"
        content_target = f"{q.settings.collection_name}_v20260515T120000_content"
        available = {
            summary_target,
            content_target,
            "other_project_summaries",
            "other_project_content",
        }

        with patch.object(
            q,
            "_get_qdrant_alias_targets",
            return_value={summary_alias: summary_target, content_alias: content_target},
        ):
            resolved = q.resolve_active_project_collections(available_names=available)

        self.assertEqual(
            resolved,
            [
                {"role": "summary", "alias": summary_alias, "name": summary_target},
                {"role": "content", "alias": content_alias, "name": content_target},
            ],
        )

    def test_resolve_active_project_collections_falls_back_to_base_pair(self):
        summary_base, content_base = q.derive_collection_names(None)

        with patch.object(q, "_get_qdrant_alias_targets", return_value={}):
            resolved = q.resolve_active_project_collections(
                available_names={summary_base, content_base, "foreign_content"}
            )

        self.assertEqual([item["name"] for item in resolved], [summary_base, content_base])

    def test_export_bundle_selects_only_active_project_pair(self):
        summary_alias, content_alias = q.derive_alias_names(None)
        summary_target = f"{q.settings.collection_name}_v20260515T120000_summaries"
        content_target = f"{q.settings.collection_name}_v20260515T120000_content"
        all_collections = [
            summary_target,
            content_target,
            "foreign_summaries",
            "foreign_content",
        ]

        def fake_download(*, collection_name, snapshot_name, scratch_dir):
            scratch = pathlib.Path(scratch_dir)
            scratch.mkdir(parents=True, exist_ok=True)
            path = scratch / snapshot_name
            path.write_bytes(f"snapshot:{collection_name}".encode("utf-8"))
            return path

        with patch.object(q.qdrant, "get_collections", return_value=_collections_response(all_collections)), patch.object(
            q.qdrant,
            "get_collection",
            return_value=types.SimpleNamespace(points_count=1),
        ), patch.object(
            q,
            "_get_qdrant_alias_targets",
            return_value={summary_alias: summary_target, content_alias: content_target},
        ), patch.object(
            q,
            "_create_collection_snapshot",
            side_effect=lambda name: {"name": f"{name}.snapshot", "size": 10},
        ), patch.object(
            q,
            "_download_collection_snapshot",
            side_effect=fake_download,
        ), patch.object(q, "_delete_remote_snapshot"):
            bundle, meta = q.export_collections_bundle(collection_names=["foreign_content"])

        self.assertEqual(meta["collections"], [summary_target, content_target])
        with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tar:
            metadata = json.loads(tar.extractfile("metadata.json").read().decode("utf-8"))

        self.assertEqual(metadata["project"]["collection_name"], q.settings.collection_name)
        self.assertEqual(
            metadata["project"]["aliases"],
            {summary_alias: summary_target, content_alias: content_target},
        )
        self.assertEqual(
            [entry["name"] for entry in metadata["collections"]],
            [summary_target, content_target],
        )

    def test_validate_project_bundle_rejects_other_project(self):
        summary_name, content_name = q.derive_collection_names(None)
        with self.assertRaises(HTTPException) as raised:
            q._validate_project_bundle_metadata(
                {"project": {"collection_name": "other_project"}},
                [{"name": summary_name}, {"name": content_name}],
            )

        self.assertEqual(raised.exception.status_code, 400)

    def test_validate_legacy_bundle_rejects_foreign_collections(self):
        summary_name, _ = q.derive_collection_names(None)
        with self.assertRaises(HTTPException) as raised:
            q._validate_project_bundle_metadata(
                {},
                [{"name": summary_name}, {"name": "foreign_content"}],
            )

        self.assertEqual(raised.exception.status_code, 400)

    def test_validate_project_bundle_returns_alias_targets(self):
        summary_alias, content_alias = q.derive_alias_names(None)
        summary_target = f"{q.settings.collection_name}_v20260515T120000_summaries"
        content_target = f"{q.settings.collection_name}_v20260515T120000_content"

        alias_targets = q._validate_project_bundle_metadata(
            {
                "project": {
                    "collection_name": q.settings.collection_name,
                    "aliases": {summary_alias: summary_target, content_alias: content_target},
                }
            },
            [
                {"name": summary_target, "role": "summary", "alias": summary_alias},
                {"name": content_target, "role": "content", "alias": content_alias},
            ],
        )

        self.assertEqual(alias_targets, {summary_alias: summary_target, content_alias: content_target})


if __name__ == "__main__":
    unittest.main()

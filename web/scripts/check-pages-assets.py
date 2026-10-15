#!/usr/bin/env python3
"""Small temporary-file tests; no web build, runtime download, or judge needed."""

import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("retention", Path(__file__).with_name("retain-pages-assets.py"))
retention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retention)
NOW = 1_800_000_000


class PagesAssetsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def build(self, label, extra=None):
        dist = self.root / label
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text(f'<script type="module" src="./assets/{label}.js"></script>')
        (dist / "assets" / f"{label}.js").write_text(label)
        for name, value in (extra or {}).items():
            path = dist / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
        return dist

    def archive(self, label, files):
        target = self.root / f"{label}.tar"
        with tarfile.open(target, "w") as archive:
            for name, value in files:
                member = tarfile.TarInfo(name)
                if isinstance(value, tuple):
                    member.type, member.linkname = value
                    archive.addfile(member)
                else:
                    value = value.encode()
                    member.size = len(value)
                    archive.addfile(member, io.BytesIO(value))
        return target

    def publish(self, dist, previous=None, now=NOW):
        archive = self.root / f"{dist.name}.tar"
        manifest = retention.retain_assets(dist, previous, now=now, archive_out=archive)
        return archive, manifest

    def test_first_publication_has_current_manifest_and_small_archive(self):
        dist = self.build("first", {"browser-runtime/huge.data": "must not be archived"})
        archive, manifest = self.publish(dist)
        self.assertEqual(manifest["generations"], [{"created_at": retention.iso_time(NOW), "assets": ["assets/first.js"]}])
        with tarfile.open(archive) as handle:
            self.assertEqual(set(handle.getnames()), {retention.MANIFEST, "assets/first.js"})

    def test_legacy_archive_retains_assets_but_never_old_html_or_runtime(self):
        previous = self.archive("legacy", [("./assets/old.js", "old"), ("./index.html", "old HTML"), ("./browser-runtime/data", "runtime")])
        dist = self.build("current")
        _, manifest = self.publish(dist, previous)
        self.assertEqual((dist / "assets/old.js").read_text(), "old")
        self.assertIn("current.js", (dist / "index.html").read_text())
        self.assertFalse((dist / "browser-runtime").exists())
        self.assertEqual(manifest["generations"][1]["retired_at"], retention.iso_time(NOW))

    def test_three_fast_releases_survive_then_expire_without_accumulation(self):
        first, _ = self.publish(self.build("first"))
        second, _ = self.publish(self.build("second"), first, NOW + 60)
        third_dist = self.build("third")
        third, third_manifest = self.publish(third_dist, second, NOW + 120)
        self.assertEqual(len(third_manifest["generations"]), 3)
        self.assertTrue((third_dist / "assets/first.js").is_file())
        fourth_dist = self.build("fourth")
        _, manifest = self.publish(fourth_dist, third, NOW + 1920)
        self.assertEqual(len(manifest["generations"]), 2)
        self.assertTrue((fourth_dist / "assets/third.js").is_file())
        self.assertFalse((fourth_dist / "assets/first.js").exists())
        self.assertFalse((fourth_dist / "assets/second.js").exists())

    def test_long_lived_current_version_gets_grace_when_replaced(self):
        first, _ = self.publish(self.build("first"))
        dist = self.build("current")
        _, manifest = self.publish(dist, first, NOW + 7 * 86400)
        self.assertTrue((dist / "assets/first.js").exists())
        self.assertEqual(manifest["generations"][1]["created_at"], retention.iso_time(NOW))
        self.assertEqual(manifest["generations"][1]["retired_at"], retention.iso_time(NOW + 7 * 86400))

    def test_current_file_wins_and_nested_old_assets_work(self):
        previous = self.archive("legacy", [("assets/shared.js", "old"), ("assets/fonts/font.woff", "font")])
        dist = self.build("current", {"assets/shared.js": "new"})
        _, manifest = self.publish(dist, previous)
        self.assertEqual((dist / "assets/shared.js").read_text(), "new")
        self.assertEqual((dist / "assets/fonts/font.woff").read_text(), "font")
        self.assertNotIn("assets/fonts/font.woff", manifest["generations"][0]["assets"])

    def test_unsafe_archive_paths_rejected(self):
        for number, name in enumerate(("../escape", "/absolute", "assets/../escape", "assets\\escape", "C:/escape", "assets//escape")):
            with self.subTest(name=name):
                previous = self.archive(f"bad{number}", [(name, "bad")])
                with self.assertRaises(ValueError):
                    self.publish(self.build(f"current{number}"), previous)
        self.assertFalse((self.root / "escape").exists())

    def test_links_and_special_archive_members_rejected(self):
        for number, kind in enumerate((tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE)):
            with self.subTest(kind=kind):
                previous = self.archive(f"bad{number}", [("assets/linked.js", (kind, "outside"))])
                with self.assertRaises(ValueError):
                    self.publish(self.build(f"current{number}"), previous)

    def test_current_symlinks_rejected(self):
        dist = self.build("current")
        (dist / "assets/link.js").symlink_to(dist / "assets/current.js")
        with self.assertRaises(ValueError):
            self.publish(dist)

    def test_manifest_missing_asset_rejected(self):
        value = {"version": 1, "generations": [{"created_at": retention.iso_time(NOW), "assets": ["assets/missing.js"]}]}
        previous = self.archive("missing", [(retention.MANIFEST, json.dumps(value))])
        with self.assertRaisesRegex(ValueError, "missing"):
            self.publish(self.build("current"), previous)

    def test_invalid_manifest_is_not_silently_treated_as_legacy(self):
        previous = self.archive("invalid", [(retention.MANIFEST, '{"version":99}'), ("assets/old.js", "old")])
        with self.assertRaises(ValueError):
            self.publish(self.build("current"), previous)

    def test_manifest_cannot_escape_assets_or_keep_multiple_active_generations(self):
        for number, generations in enumerate((
            [{"created_at": retention.iso_time(NOW), "assets": ["../escape"]}],
            [{"created_at": retention.iso_time(NOW), "assets": ["assets/old.js"]}] * 2,
        )):
            with self.subTest(number=number):
                previous = self.archive(f"invalid{number}", [(retention.MANIFEST, json.dumps({"version": 1, "generations": generations})), ("assets/old.js", "old")])
                with self.assertRaises(ValueError):
                    self.publish(self.build(f"current{number}"), previous)

    def test_duplicate_archive_file_rejected(self):
        previous = self.archive("duplicate", [("assets/old.js", "one"), ("./assets/old.js", "two")])
        with self.assertRaises(ValueError):
            self.publish(self.build("current"), previous)

    def test_current_entry_must_be_complete(self):
        dist = self.build("current")
        (dist / "index.html").write_text('<script src="./assets/missing.js"></script>')
        with self.assertRaises(ValueError):
            self.publish(dist)

    def test_size_limit_is_checked_before_copying_previous_assets(self):
        previous = self.archive("large", [("assets/old.js", "x" * 20)])
        dist = self.build("current")
        limit = retention.MAX_ASSET_BYTES
        try:
            retention.MAX_ASSET_BYTES = 10
            with self.assertRaises(ValueError):
                self.publish(dist, previous)
            self.assertFalse((dist / "assets/old.js").exists())
        finally:
            retention.MAX_ASSET_BYTES = limit

    def test_reprocessing_a_retained_build_is_rejected(self):
        dist = self.build("current")
        self.publish(dist)
        with self.assertRaises(ValueError):
            self.publish(dist)


if __name__ == "__main__":
    unittest.main()

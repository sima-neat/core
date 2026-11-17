#!/usr/bin/env python3
"""Host-only content, failure, concurrency and cache tests for accepted fixtures."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

import prepare_codec_perf_fixtures as preparation


def sha(data):
    return hashlib.sha256(data).hexdigest()


class FixturePreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "fixtures"
        self.archive = self.root / "original.tar.gz"
        self.cache = self.root / "cache"
        self.lock = self.root / "fixtures.lock.json"
        self.members = {"h264.h264": b"original h264", "h264.nv12": b"decoded h264",
                        "h265.h265": b"original h265", "h265.nv12": b"decoded h265"}
        self.data = {"schema_version": 1, "source": {"commit": "a" * 40},
                     "archive": {"url": "https://artifacts.neat.sima.ai/core/develop/aaaaaaaaaaaa/extras.tar.gz"},
                     "members": {name: {"path": "assets/" + name, "size": len(value),
                                        "sha256": sha(value)} for name, value in self.members.items()}}
        self.make_archive()

    def make_archive(self, *, missing=None, duplicate=None, link=None, corrupt=None):
        with tarfile.open(self.archive, "w:gz") as archive:
            for name, value in self.members.items():
                if name == missing:
                    continue
                entry = tarfile.TarInfo("./assets/" + name)
                entry.size = len(value)
                if name == corrupt:
                    value = b"X" * len(value)
                if name == link:
                    entry.type = tarfile.SYMTYPE
                    entry.linkname = "../../escape"
                    entry.size = 0
                archive.addfile(entry, None if name == link else io.BytesIO(value))
                if name == duplicate:
                    archive.addfile(entry, io.BytesIO(value))
            # Unrelated extras, including a path that must never be extracted.
            extra = tarfile.TarInfo("../unrelated")
            extra.size = 3
            archive.addfile(extra, io.BytesIO(b"BAD"))
        self.data["archive"].update(size=self.archive.stat().st_size,
                                    sha256=preparation.digest(self.archive))
        self.lock.write_text(json.dumps(self.data))

    def prepare(self, **kwargs):
        return preparation.prepare(self.output, self.lock, accuracy=False, **kwargs)

    def assert_members(self):
        for name, data in self.members.items():
            self.assertEqual((self.output / name).read_bytes(), data)

    def test_exact_pairs_only_and_existing_accuracy_untouched(self):
        self.output.mkdir()
        (self.output / "jpeg.jpg").write_bytes(b"unrelated accuracy fixture")
        report = self.prepare(archive=self.archive)
        self.assert_members()
        self.assertEqual((self.output / "jpeg.jpg").read_bytes(), b"unrelated accuracy fixture")
        self.assertEqual(set(report["members"]), set(self.members))
        self.assertFalse((self.root / "unrelated").exists())

    def test_warm_offline_has_no_cli_or_generator_dependency(self):
        self.prepare(archive=self.archive)
        self.archive.unlink()
        with mock.patch.object(preparation.subprocess, "run", side_effect=AssertionError("network")):
            self.prepare(offline=True, cli="not-installed")
        self.assert_members()

    def test_stale_recipe_stamp_does_not_redefine_locked_content(self):
        self.prepare(archive=self.archive)
        (self.output / "h264.recipe.sha256").write_text("obsolete generator")
        self.prepare(offline=True)
        self.assert_members()

    def test_missing_encoded_or_reference_restored_as_verified_pair(self):
        for name in self.members:
            with self.subTest(name=name):
                self.prepare(archive=self.archive)
                (self.output / name).unlink()
                self.prepare(archive=self.archive)
                self.assert_members()

    def test_same_size_corruption_never_passes_offline_without_archive(self):
        self.prepare(archive=self.archive)
        name = "h265.nv12"
        (self.output / name).write_bytes(b"X" * len(self.members[name]))
        with self.assertRaisesRegex(ValueError, "offline cache"):
            self.prepare(offline=True)
        self.prepare(archive=self.archive)
        self.assert_members()

    def test_verified_archive_cache_repairs_offline(self):
        self.cache.mkdir()
        target = self.cache / (self.data["archive"]["sha256"] + ".tar.gz")
        target.write_bytes(self.archive.read_bytes())
        self.prepare(cache=self.cache, offline=True)
        self.assert_members()

    def test_corrupt_archive_fails_before_replacing_members(self):
        self.output.mkdir()
        for name in self.members:
            (self.output / name).write_bytes(b"old pair")
        self.archive.write_bytes(b"wrong archive")
        with self.assertRaisesRegex(ValueError, "archive differs"):
            self.prepare(archive=self.archive)
        self.assertTrue(all((self.output / name).read_bytes() == b"old pair" for name in self.members))

    def test_missing_duplicate_link_or_wrong_reference_rejected_before_commit(self):
        for mutation in ("missing", "duplicate", "link", "corrupt"):
            with self.subTest(mutation=mutation):
                self.make_archive(**{mutation: "h264.nv12"})
                with self.assertRaises(ValueError):
                    self.prepare(archive=self.archive)
                self.assertFalse(any((self.output / name).exists() for name in self.members))

    def test_unsafe_lock_member_rejected(self):
        self.data["members"]["h264.h264"]["path"] = "../h264.h264"
        self.lock.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.prepare(archive=self.archive)

    def test_symlink_output_is_not_followed(self):
        outside = self.root / "outside"; outside.mkdir()
        self.output.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.prepare(archive=self.archive)
        self.assertEqual(list(outside.iterdir()), [])

    def test_cli_download_is_immutable_and_cached(self):
        payload = self.archive.read_bytes()
        def download(command, **kwargs):
            self.assertEqual(command[:3], ["sima-cli", "download", self.data["archive"]["url"]])
            self.assertEqual(command[3], "-d")
            (Path(command[4]) / "extras.tar.gz").write_bytes(payload)
        with mock.patch.object(preparation.subprocess, "run", side_effect=download) as call:
            self.prepare(cache=self.cache)
            (self.output / "h264.h264").unlink()
            self.prepare(cache=self.cache, offline=True)
            self.assertEqual(call.call_count, 1)
        self.assert_members()

    def test_wrong_download_does_not_install_or_poison_cache(self):
        def download(command, **kwargs):
            (Path(command[4]) / "extras.tar.gz").write_bytes(b"wrong")
        with mock.patch.object(preparation.subprocess, "run", side_effect=download):
            with self.assertRaisesRegex(ValueError, "Downloaded"):
                self.prepare(cache=self.cache)
        self.assertEqual(list(self.cache.glob("*.tar.gz")), [])
        self.assertFalse(any((self.output / name).exists() for name in self.members))

    def test_repacked_container_with_identical_members_still_fails_closed(self):
        expected = dict(self.data["archive"])
        # A publisher can repack identical workloads with different container
        # metadata. That requires an explicit lock refresh, never auto-acceptance.
        with tarfile.open(self.archive, "w:gz") as archive:
            for name, value in self.members.items():
                entry = tarfile.TarInfo("./assets/" + name)
                entry.size = len(value)
                entry.mtime = 123
                archive.addfile(entry, io.BytesIO(value))
        payload = self.archive.read_bytes()
        self.assertNotEqual(sha(payload), expected["sha256"])
        def download(command, **kwargs):
            (Path(command[4]) / "extras.tar.gz").write_bytes(payload)
        with mock.patch.object(preparation.subprocess, "run", side_effect=download):
            with self.assertRaises(ValueError) as failure:
                self.prepare(cache=self.cache)
        message = str(failure.exception)
        self.assertIn("expected size=" + str(expected["size"]), message)
        self.assertIn("sha256=" + expected["sha256"], message)
        self.assertIn("actual size=" + str(len(payload)), message)
        self.assertIn("sha256=" + sha(payload), message)
        self.assertEqual(list(self.cache.glob("*.tar.gz")), [])
        self.assertFalse(any((self.output / name).exists() for name in self.members))

    def test_interrupted_commit_is_not_success_and_next_prepare_repairs(self):
        replace = preparation.os.replace
        calls = 0
        def interrupted(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("interrupted cold preparation")
            replace(source, target)
        with mock.patch.object(preparation.os, "replace", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.prepare(archive=self.archive)
        self.prepare(archive=self.archive, offline=True)
        self.assert_members()

    def test_accuracy_recipe_is_called_but_no_b_generation_is_not(self):
        import make_codec_perf_fixtures as generator
        with mock.patch.object(generator, "generate", side_effect=AssertionError("no-B generation")), \
             mock.patch.object(generator, "make_accuracy_fixtures") as accuracy:
            preparation.prepare(self.output, self.lock, archive=self.archive)
        accuracy.assert_called_once_with(self.output.resolve(), False)
        self.assert_members()

    def test_generator_remains_explicit_new_workload_producer(self):
        import make_codec_perf_fixtures as generator
        with mock.patch.object(generator, "generate") as generate, \
             mock.patch.object(sys, "argv", ["make_codec_perf_fixtures.py", "--output-dir", str(self.output), "--force"]):
            self.assertEqual(generator.main(), 0)
        generate.assert_called_once_with(self.output, True, 1280, 720, 30)

    def test_two_cold_processes_serialize_and_leave_complete_pairs(self):
        program = ("import sys; from pathlib import Path; "
                   "import prepare_codec_perf_fixtures as p; "
                   "p.prepare(Path(sys.argv[1]), Path(sys.argv[2]), archive=Path(sys.argv[3]), accuracy=False)")
        command = [sys.executable, "-c", program, str(self.output), str(self.lock), str(self.archive)]
        processes = [subprocess.Popen(command, cwd=Path(__file__).parent, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True) for _ in range(2)]
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, stdout + stderr)
        self.assert_members()


if __name__ == "__main__":
    unittest.main()

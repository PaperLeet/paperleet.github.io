"""Historical evidence must survive renames without masking corruption."""
from pathlib import Path
import gzip
import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('qualification_verifier', Path(__file__).with_name('verify_qualification.py'))
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
digest = lambda data: hashlib.sha256(data).hexdigest()
legacy = 'packages/stridetorch/benchmarks/results/'
current = 'packages/torchwasm/benchmarks/results/'


class HistoricalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name)
        (self.repo/current).mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def archive(self, name, files, blobs, external=None):
        manifest = dict(format='sha256-deduplicated-v2' if external else 'sha256-deduplicated-v1', files=files)
        if external:
            manifest['external_blob_sources'] = external
        payload = json.dumps(manifest).encode()
        archive_path = self.repo/current/(name+'.tar.gz')
        with archive_path.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as zipped, tarfile.open(fileobj=zipped, mode='w') as tar:
            for path, data in [('manifest.json', payload), *[('blobs/'+digest(data), data) for data in blobs]]:
                info = tarfile.TarInfo(path)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        reference = dict(path=legacy+archive_path.name, sha256=digest(archive_path.read_bytes()), manifest_sha256=digest(payload))
        record_path = self.repo/current/(name+'.json')
        record_path.write_text(json.dumps(dict(raw_archive=reference)))
        return record_path, reference

    def simple(self):
        data = b'immutable evidence'
        item = dict(sha256=digest(data), bytes=len(data))
        path, ref = self.archive('parent', {'fixture.txt': item}, [data])
        return path, ref, data, item

    def test_legacy_missing_path_resolves_renamed_archive(self):
        path, _, _, _ = self.simple()
        result = verifier.verify(path, self.repo)
        self.assertEqual(result['status'], 'verified_evidence_integrity')
        self.assertEqual(result['local_blobs'], 1)

    def test_current_paths_remain_supported(self):
        path, ref, _, _ = self.simple()
        ref['path'] = ref['path'].replace(legacy, current)
        path.write_text(json.dumps(dict(raw_archive=ref)))
        self.assertEqual(verifier.verify(path, self.repo)['logical_files'], 1)

    def test_older_v1_record_is_bound_by_complete_archive_hash(self):
        path, ref, _, _ = self.simple()
        del ref['manifest_sha256']
        path.write_text(json.dumps(dict(raw_archive=ref)))
        self.assertEqual(verifier.verify(path, self.repo)['local_blobs'], 1)
        target = self.repo/current/Path(ref['path']).name
        target.write_bytes(target.read_bytes()+b'changed')
        with self.assertRaises(AssertionError):
            verifier.verify(path, self.repo)

    def test_legacy_inherited_record_and_archive_are_both_resolved(self):
        parent, ref, data, item = self.simple()
        provider = dict(kind='archive_blob', record=legacy+parent.name,
                        record_sha256=digest(parent.read_bytes()), archive=ref['path'],
                        archive_sha256=ref['sha256'], member='blobs/'+digest(data), bytes=len(data))
        child, _ = self.archive('child', {'inherited.txt': item}, [], {digest(data): provider})
        result = verifier.verify(child, self.repo)
        self.assertEqual(result['inherited_archive_blobs'], 1)
        self.assertEqual(result['local_blobs'], 0)

    def test_existing_corrupt_original_is_not_hidden_by_renamed_copy(self):
        path, ref, _, _ = self.simple()
        original = self.repo/ref['path']
        original.parent.mkdir(parents=True)
        original.write_bytes(b'corrupt original')
        with self.assertRaises(AssertionError):
            verifier.verify(path, self.repo)

    def test_escaping_reference_remains_invalid(self):
        path, ref, _, _ = self.simple()
        ref['path'] = legacy+'../../../../../outside.tar.gz'
        path.write_text(json.dumps(dict(raw_archive=ref)))
        with self.assertRaises(ValueError):
            verifier.verify(path, self.repo)

    def test_renamed_symlink_cannot_escape_repository(self):
        path, ref, _, _ = self.simple()
        target = self.repo/current/Path(ref['path']).name
        target.unlink()
        with tempfile.TemporaryDirectory() as external:
            outside = Path(external)/target.name
            outside.write_bytes(b'external evidence')
            target.symlink_to(outside)
            with self.assertRaises(ValueError):
                verifier.verify(path, self.repo)


if __name__ == '__main__':
    unittest.main()

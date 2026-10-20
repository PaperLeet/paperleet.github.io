"""Restore a self-contained qualification archive without repository history.

python restore.py RECORD.json ARCHIVE.tar.gz NEW_OUTPUT_DIRECTORY [PREFIX]
The optional prefix selects logical files, e.g. sdk/. All archive blobs are
still checked. The destination must not already exist.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import tarfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('record', type=Path)
parser.add_argument('archive', type=Path)
parser.add_argument('destination', type=Path)
parser.add_argument('prefix', nargs='?', default='')
options = parser.parse_args()
record_path, archive_path, destination = options.record, options.archive, options.destination
prefix = options.prefix
reference = json.loads(record_path.read_text())['raw_archive']
hasher = hashlib.sha256()
with archive_path.open('rb') as source:
    for block in iter(lambda: source.read(1048576), b''):
        hasher.update(block)
assert hasher.hexdigest() == reference['sha256']
with tarfile.open(archive_path, 'r|gz') as archive:
    iterator = iter(archive)
    first = next(iterator)
    assert first.name == 'manifest.json'
    payload = archive.extractfile(first).read()
    assert hashlib.sha256(payload).hexdigest() == reference['manifest_sha256']
    manifest = json.loads(payload)
    assert manifest['format'] == 'sha256-deduplicated-v1'
    assert not manifest.get('external_blob_sources')
    wanted = {}
    for name, item in manifest['files'].items():
        path = PurePosixPath(name)
        assert not path.is_absolute() and '..' not in path.parts
        if name.startswith(prefix):
            wanted.setdefault(item['sha256'], []).append((path, item['bytes']))
    assert wanted
    destination.mkdir(parents=True, exist_ok=False)
    seen = set()
    restored = 0
    for member in iterator:
        assert member.isfile()
        blob = archive.extractfile(member).read()
        digest = hashlib.sha256(blob).hexdigest()
        assert member.name == 'blobs/' + digest and digest not in seen
        seen.add(digest)
        for name, size in wanted.pop(digest, []):
            assert len(blob) == size
            target = destination.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
            restored += 1
    assert not wanted
print(json.dumps(dict(status='restored_verified_archive', files=restored,
                      unique_blobs=len(seen), destination=str(destination))))

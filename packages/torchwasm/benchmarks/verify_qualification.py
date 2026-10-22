"""Check evidence bytes, including inherited archives and pinned Git blobs.

Usage: python benchmarks/verify_qualification.py results/RECORD.json
This checks evidence integrity; a passing archive is not itself an API guarantee.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import re
import subprocess
import tarfile


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(record_path, repo, check_current_source=False):
    repo = repo.resolve()
    def inside(name):
        path = (repo / name).resolve()
        if not path.is_relative_to(repo):
            raise ValueError(f'Evidence path escapes repository: {name}')
        # Immutable pre-rename records retain their original package paths.
        # Resolve only missing evidence paths; an existing corrupt original
        # must still fail its recorded hash rather than be silently replaced.
        legacy = 'packages/stridetorch/benchmarks/results/'
        if not path.exists() and name.startswith(legacy):
            renamed = (repo / 'packages/torchwasm/benchmarks/results' / name[len(legacy):]).resolve()
            if not renamed.is_relative_to(repo):
                raise ValueError(f'Evidence path escapes repository: {name}')
            if renamed.is_file():
                return renamed
        return path

    record = json.loads(record_path.read_text())
    reference = record['raw_archive']
    archive = inside(reference['path'])
    assert file_hash(archive) == reference['sha256'], archive
    resolved = {}
    with tarfile.open(archive, 'r|gz') as tar:
        iterator = iter(tar)
        first = next(iterator)
        assert first.name == 'manifest.json'
        payload = tar.extractfile(first).read()
        # Some older v1 records pin the entire archive without recording a
        # second manifest digest. The checked archive SHA already binds it.
        if 'manifest_sha256' in reference:
            assert hashlib.sha256(payload).hexdigest() == reference['manifest_sha256']
        manifest = json.loads(payload)
        assert manifest['format'] in {'sha256-deduplicated-v1', 'sha256-deduplicated-v2', 'sha256-deduplicated-v3'}
        for member in iterator:
            data = tar.extractfile(member).read()
            digest = hashlib.sha256(data).hexdigest()
            assert member.name == 'blobs/' + digest
            assert digest not in resolved
            resolved[digest] = len(data)
    local_count = len(resolved)
    groups = collections.defaultdict(dict)
    git_count = 0
    for digest, provider in manifest.get('external_blob_sources', {}).items():
        assert re.fullmatch('[0-9a-f]{64}', digest)
        assert digest not in resolved
        if provider.get('kind') == 'git_blob':
            commit, oid, name = provider['commit'], provider['git_blob'], provider['path']
            assert re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}', commit)
            assert re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}', oid)
            actual = subprocess.check_output(['git', 'rev-parse', '--verify', commit + ':' + name], cwd=repo, text=True).strip()
            assert actual == oid
            data = subprocess.check_output(['git', 'cat-file', 'blob', oid], cwd=repo)
            assert hashlib.sha256(data).hexdigest() == digest == provider['sha256']
            assert len(data) == provider['bytes']
            resolved[digest] = len(data)
            git_count += 1
        else:
            assert provider.get('kind', 'archive_blob') == 'archive_blob'
            assert file_hash(inside(provider['record'])) == provider['record_sha256']
            assert provider['member'] == 'blobs/' + digest
            groups[(provider['archive'], provider['archive_sha256'])][digest] = provider
    archive_count = 0
    for (name, expected), wanted in groups.items():
        parent = inside(name)
        assert file_hash(parent) == expected
        pending = set(wanted)
        with tarfile.open(parent, 'r|gz') as tar:
            for member in tar:
                if not member.name.startswith('blobs/'):
                    continue
                digest = member.name[len('blobs/'):]
                if digest not in pending:
                    continue
                data = tar.extractfile(member).read()
                assert hashlib.sha256(data).hexdigest() == digest
                assert len(data) == wanted[digest]['bytes']
                resolved[digest] = len(data)
                pending.remove(digest)
                archive_count += 1
        assert not pending, (name, pending)
        assert file_hash(parent) == expected
    for name, item in manifest['files'].items():
        assert resolved[item['sha256']] == item['bytes'], name
    if check_current_source:
        source = repo / 'packages/torchwasm/src/torchwasm'
        actual = {str(path.relative_to(source)): file_hash(path) for path in source.rglob('*')
                  if path.is_file() and '__pycache__' not in path.parts}
        assert actual == record['public_files']
        tests = repo / 'packages/torchwasm/tests'
        assert {path.name: file_hash(path) for path in tests.glob('*.py')} == record['sdk_files']
    return dict(status='verified_evidence_integrity', format=manifest['format'],
                logical_files=len(manifest['files']), local_blobs=local_count,
                inherited_archive_blobs=archive_count, inherited_git_blobs=git_count,
                current_source_checked=check_current_source, external_capture_directories_used=False,
                archive_sha256=reference['sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path)
    parser.add_argument('--check-current-source', action='store_true')
    options = parser.parse_args()
    print(json.dumps(verify(options.record.resolve(), Path(__file__).resolve().parents[3],
                            options.check_current_source), indent=2))

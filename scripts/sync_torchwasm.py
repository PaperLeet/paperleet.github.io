"""Adopt a reviewed TorchWasm source snapshot without running upstream code."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile

REPOSITORY = "PaperLeet/stridetorch"
PACKAGE = "packages/torchwasm"
LOCK = PACKAGE + "/upstream.lock.json"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def managed(name):
    if not isinstance(name, str) or "\\" in name or any(ord(char) < 32 for char in name):
        return False
    parts = PurePosixPath(name).parts
    return (bool(parts) and not name.startswith("/")
            and all(part not in {"", ".", ".."} and not part.startswith(".") for part in name.split("/"))
            and (name in {"LICENSE", "NOTICE", "pyproject.toml"}
                 or name.startswith(("src/", "tests/"))))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(source, revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a full immutable TorchWasm commit SHA")
    if git(source, "rev-parse", "--verify", revision + "^{commit}").decode().strip() != revision:
        raise ValueError("Runtime revision is not a commit")
    files = {}
    size = 0
    for row in git(source, "ls-tree", "-rz", "--full-tree", revision).split(b"\0"):
        if not row:
            continue
        metadata, raw_name = row.split(b"\t", 1)
        name = raw_name.decode()
        if not managed(name):
            continue
        mode, kind, oid = metadata.decode().split()
        if kind != "blob" or mode != "100644":
            raise ValueError("Runtime sources must be regular non-executable files: " + name)
        length = int(git(source, "cat-file", "-s", oid))
        size += length
        if length > 8 * 1024 * 1024 or size > 32 * 1024 * 1024:
            raise ValueError("Runtime snapshot exceeds the source budget")
        files[name] = git(source, "cat-file", "blob", oid)
    if "src/stridetorch/__init__.py" in files:
        raise ValueError("Runtime package rename requires coordinated product integration; preserve the current runtime pin")
    if not {"LICENSE", "NOTICE", "pyproject.toml", "src/torchwasm/__init__.py"} <= files.keys():
        raise ValueError("Incomplete standalone runtime snapshot")
    if not any(name.startswith("tests/") for name in files):
        raise ValueError("Runtime snapshot has no independent tests")
    return files


def check(root):
    lock = json.loads((root / LOCK).read_text())
    if lock.get("version") != 1 or lock.get("repository") != REPOSITORY:
        raise ValueError("Unknown runtime provenance")
    if not re.fullmatch(r"[0-9a-f]{40}", lock.get("revision", "")):
        raise ValueError("Runtime provenance requires an immutable revision")
    for name, expected in lock["files"].items():
        if not managed(name):
            raise ValueError("Unsafe managed runtime path")
        path = root / PACKAGE / name
        if any(parent.is_symlink() for parent in [path, *path.parents] if parent != root.parent):
            raise ValueError("Runtime paths must not traverse symlinks")
        if not path.is_file() or digest(path.read_bytes()) != expected:
            raise ValueError("Local runtime changes must be preserved before syncing: " + name)
    return lock


def update(root, source, revision):
    previous = check(root)
    files = snapshot(source, revision)
    package = root / PACKAGE
    for name in files.keys() - previous["files"].keys():
        if (package / name).exists():
            raise ValueError("Refusing to overwrite an unmanaged runtime file: " + name)
    # Stage all bytes before replacing any managed source. Keep rollback bytes in
    # memory, including provenance, so a failed update cannot silently advance it.
    before = {name: (package / name).read_bytes() for name in previous["files"]}
    old_lock = (root / LOCK).read_bytes()
    touched = set()
    temporary_paths = set()
    try:
        for name, data in files.items():
            path = package / name
            if before.get(name) == data:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".runtime-update-", delete=False) as output:
                temporary = Path(output.name)
                temporary_paths.add(temporary)
                output.write(data)
            temporary.chmod(0o644)
            temporary.replace(path)
            touched.add(name)
        for name in previous["files"].keys() - files.keys():
            (package / name).unlink()
            touched.add(name)
        lock = {"version": 1, "repository": REPOSITORY, "revision": revision,
                "files": {name: digest(data) for name, data in sorted(files.items())}}
        (root / LOCK).write_text(json.dumps(lock, indent=2) + "\n")
    except Exception:
        for name in touched:
            if name in before:
                (package / name).write_bytes(before[name])
            else:
                (package / name).unlink(missing_ok=True)
        (root / LOCK).write_bytes(old_lock)
        raise
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return len(touched)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="Reviewed full standalone runtime commit SHA")
    parser.add_argument("--source", type=Path, help="Existing local standalone Git checkout")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not args.revision:
        print("Pinned TorchWasm source verified: " + check(root)["revision"])
        return
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("Use a full immutable TorchWasm commit SHA")
    with tempfile.TemporaryDirectory(prefix="torchwasm-sync-") as directory:
        source = args.source or Path(directory)
        if args.source is None:
            git(source, "init", "-q")
            git(source, "fetch", "--depth=1", "https://github.com/" + REPOSITORY + ".git", args.revision)
        changed = update(root, source, args.revision)
    print(f"Updated {changed} managed runtime files. Run native, wheel and WASM checks before merging.")


if __name__ == "__main__":
    main()

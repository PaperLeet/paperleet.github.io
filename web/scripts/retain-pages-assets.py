#!/usr/bin/env python3
"""Keep recently replaced Vite assets without retaining old HTML or runtimes."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import tarfile

MANIFEST = "pages-assets-manifest.json"
WINDOW_SECONDS = 7 * 24 * 60 * 60
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_ASSET_BYTES = 256 * 1024 * 1024
MAX_ASSET_FILES = 20_000


def safe_name(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or any(ord(c) < 32 for c in name):
        raise ValueError("Invalid artifact path")
    while name.startswith("./"):
        name = name[2:]
    name = name.removesuffix("/")
    if name in ("", "."):
        return ""
    if any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError(f"Unsafe artifact path: {name!r}")
    return name


def asset_names(value):
    if not isinstance(value, list) or not value or len(value) > MAX_ASSET_FILES:
        raise ValueError("Invalid generation asset list")
    names = []
    for item in value:
        name = safe_name(item)
        if name != item or not name.startswith("assets/"):
            raise ValueError("Generation paths must be relative assets/ files")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate generation asset path")
    return names


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Invalid generation timestamp")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Generation timestamps must include a timezone")
    return result.timestamp()


def iso_time(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


class EntryAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        value = attrs.get("src") if tag == "script" else attrs.get("href") if tag == "link" else None
        if value and value.removeprefix("./").startswith("assets/"):
            self.paths.add(safe_name(value))


def current_assets(dist):
    if dist.is_symlink() or not dist.is_dir():
        raise ValueError("The current build must be a real directory")
    for path in dist.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not allowed in the current build: {path}")
    if (dist / MANIFEST).exists():
        raise ValueError("Asset retention requires a fresh build, without an old generation manifest")
    assets = sorted(path.relative_to(dist).as_posix() for path in (dist / "assets").rglob("*") if path.is_file())
    asset_names(assets)
    parser = EntryAssets()
    parser.feed((dist / "index.html").read_text(encoding="utf-8"))
    if not parser.paths or not parser.paths.issubset(assets):
        raise ValueError("The current index.html references missing build assets")
    return assets


def previous_generations(archive, members, now):
    manifest_member = members.get(MANIFEST)
    if manifest_member is None:
        names = sorted(name for name, member in members.items() if name.startswith("assets/") and member.isfile())
        asset_names(names)
        print("Previous artifact has no generation manifest; migrating its assets for 7 days.")
        return [{"created_at": iso_time(now), "retired_at": iso_time(now), "assets": names}]
    if not manifest_member.isfile() or manifest_member.size > MAX_MANIFEST_BYTES:
        raise ValueError("Invalid previous generation manifest")
    with archive.extractfile(manifest_member) as source:
        value = json.load(source)
    if not isinstance(value, dict) or value.get("version") != 1 or not isinstance(value.get("generations"), list):
        raise ValueError("Unsupported previous generation manifest")
    retained = []
    active = 0
    for generation in value["generations"]:
        if not isinstance(generation, dict):
            raise ValueError("Invalid asset generation")
        created = timestamp(generation.get("created_at"))
        names = asset_names(generation.get("assets"))
        retired_at = generation.get("retired_at")
        active += retired_at is None
        retired = now if retired_at is None else timestamp(retired_at)
        if created > now + 60 or retired > now + 60 or retired < created:
            raise ValueError("Invalid generation lifetime")
        # A version may have been live for weeks. Its grace period starts when
        # replaced, not when it was built, because users can just have cached it.
        if now - retired < WINDOW_SECONDS:
            retained.append({"created_at": iso_time(created), "retired_at": iso_time(retired), "assets": names})
    if active != 1:
        raise ValueError("Expected exactly one current generation in the previous artifact")
    return retained


def retain_assets(dist, previous=None, *, seeds=(), now=None, archive_out=None):
    dist = Path(dist)
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    own = current_assets(dist)
    own_size = sum((dist / name).stat().st_size for name in own)
    if own_size > MAX_ASSET_BYTES:
        raise ValueError("Current assets exceed the 256 MiB size limit")
    generations = [{"created_at": iso_time(now), "assets": own}]
    retained_count = 0
    # Historical lightweight artifacts repair versions pruned by an older,
    # shorter grace period. All archives stay open until validated assets copy.
    archives = ([previous] if previous is not None else []) + list(seeds)
    with ExitStack() as stack:
        sources = {}
        older = {}
        for path in archives:
            archive = stack.enter_context(tarfile.open(path, "r:*"))
            members = {}
            for member in archive:
                name = safe_name(member.name)
                if not (member.isdir() or member.isfile()):
                    raise ValueError(f"Links and special files are not allowed: {member.name!r}")
                if not name:
                    if not member.isdir():
                        raise ValueError("Invalid archive root")
                    continue
                if name in members:
                    raise ValueError(f"Duplicate archive path: {name}")
                members[name] = member
            for generation in previous_generations(archive, members, now):
                key = (generation["created_at"], tuple(generation["assets"]))
                existing = older.get(key)
                # Reading a historical archive must not extend a retirement
                # date already recorded in a newer archive.
                if existing is None or timestamp(generation["retired_at"]) < timestamp(existing["retired_at"]):
                    older[key] = generation
                for name in generation["assets"]:
                    member = members.get(name)
                    if member is None or not member.isfile():
                        raise ValueError(f"Previous generation asset is missing: {name}")
                    sources.setdefault(name, (archive, member))
        retained = sorted(older.values(), key=lambda item: (timestamp(item["retired_at"]), timestamp(item["created_at"])), reverse=True)
        pruned = 0
        while True:
            wanted = sorted({name for generation in retained for name in generation["assets"]}.difference(own))
            size = own_size + sum(sources[name][1].size for name in wanted)
            if size <= MAX_ASSET_BYTES and len(own) + len(wanted) <= MAX_ASSET_FILES:
                break
            # Evict complete oldest versions, never individual chunks: keeping
            # only part of a lazy dependency graph would still break Submit.
            retained.pop()
            pruned += 1
        if pruned:
            print(f"Asset budget reached; dropped {pruned} oldest complete generations.")
        for name in wanted:
            archive, member = sources[name]
            destination = dist / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
            retained_count += 1
        generations.extend(retained)
    if not archives:
        print("No previous successful Pages artifact is available; publishing the validated current build only.")
    manifest = {"version": 1, "grace_seconds": WINDOW_SECONDS, "generations": generations}
    encoded = json.dumps(manifest, indent=2) + "\n"
    if len(encoded.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise ValueError("Generation manifest exceeds the 4 MiB size limit")
    (dist / MANIFEST).write_text(encoded, encoding="utf-8")
    if archive_out is not None:
        with tarfile.open(archive_out, "w", dereference=True) as archive:
            archive.add(dist / MANIFEST, arcname=MANIFEST, recursive=False)
            for name in sorted({name for generation in generations for name in generation["assets"]}):
                archive.add(dist / name, arcname=name, recursive=False)
    print(f"Pages assets: {len(own)} current, {retained_count} retained, {len(generations)} generations; grace period 7 days.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--archive-out", type=Path)
    parser.add_argument("--seed", type=Path, action="append", default=[])
    args = parser.parse_args()
    retain_assets(args.dist, args.previous, seeds=args.seed, archive_out=args.archive_out)


if __name__ == "__main__":
    main()

# Explicit releases

Ordinary branch pushes and pull requests do not start CI or publish. Use an explicit release when ready. Existing native/reference/mutation/determinism/budget, infrastructure/frontend and full WASM checks remain mandatory.

## Manual release

Run **PaperLeet quality gate** in Actions with branch `main`, or use:

```sh
gh workflow run ci.yml --repo PaperLeet/paperleet.github.io --ref main
```

After all four checks succeed, **Publish PaperLeet public site** automatically builds and publishes that exact tested commit. Its standalone manual entry is only a retry of publication after a complete explicit quality run for the same current main; it cannot bypass tests.

## Version tag

Tags use `vMAJOR.MINOR.PATCH`, for example `v1.0.0`. Lightweight and annotated tags work. No prerelease suffix is accepted. Both tag and checkout must resolve to the current `main` commit:

```sh
git switch main
git pull --ff-only
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin refs/tags/v1.0.0
```

Pushing the tag starts the same complete quality gate and subsequent Pages publication. Creating a draft GitHub Release alone is not the trigger. Do not move or reuse release tags. If main or the tag changes during validation/build, publication is rejected; release the new main explicitly instead. Historical tags are not rollback commands.

Repeated publication of an already successfully deployed commit skips its build/deploy, including concurrent attempts serialized by the existing Pages deployment group. A previous overall success with a skipped deploy does not count as publication.

## Backend coordination

Public and private repositories have independent tags and commits. A public tag does not update the backend. First finish and verify the public release. In the private backend, update `core` to that reviewed, published public SHA and commit the gitlink on backend main. Then explicitly release backend main through its own manual workflow or version tag. The private complete suite tests that exact pinned core before its existing Oracle deployment.

The private dispatcher still deploys only current main. These workflows do not enable historical-SHA promotion, force publication, change deployment credentials, or modify production data.

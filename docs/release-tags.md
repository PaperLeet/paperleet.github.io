# Local iteration and explicit tag releases

Normal pushes to main or feature branches and pull requests do not start GitHub
Actions. Continue running the maintained checks locally. No tag, release or
manual workflow dispatch is created by the ongoing StrideTorch iteration.

When the maintainer chooses to publish a tested main revision that includes this
workflow configuration, create and push a version tag, for example:

```sh
git tag -a v0.33.0 -m "Release v0.33.0"
git push origin v0.33.0
```

The example is an instruction for a future maintainer action, not a command run
by the agent. `vMAJOR.MINOR.PATCH` and prerelease names such as `v0.33.0-rc.1` are
accepted. Lightweight and annotated tags are supported. Deleted or moved tags,
branch pushes, PRs and manual CI runs do not publish. Do not put `[skip ci]` in a
release-target commit message: skip directives can also suppress its later tag
workflow. Event filters handle normal development without skip directives.

The release workflow first resolves the tag and requires its commit to be
contained in main. It then calls the repository's quality workflow at that exact
resolved SHA. All existing native/frontend, infrastructure, full WASM and pilot
checks must succeed, with their tests, budgets and three native repetitions
retained. Only then does Pages build and deploy. The tag is checked again before
publication; moving/deleting it or rewriting an annotation invalidates the run.
Main may advance while an explicitly chosen older release commit remains pinned.

Manual quality checks can still be requested explicitly from Actions. They run
checks only and have no automatic publication link. A newer explicit tag cancels
an older release workflow to avoid redundant runner use. No scheduled CI or
background release is configured.

Historical asset retention accepts prior main and tag deployments and requires
an actual successful Pages deployment before selecting the previous artifact.
The existing bounded history backfill and cached-page recovery remain in place.

The policy follows GitHub's [tag event filters](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#onpushbranchestagsbranches-ignoretags-ignore)
and [local reusable workflows](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows).

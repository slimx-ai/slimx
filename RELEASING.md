# Releasing SlimX

A short, repeatable checklist so every release is consistent. SlimX uses semantic
versioning under `0.x` (minor = features/behavior changes, patch = fixes).

## Version lives in two places

Keep these in sync on every release (the README shows the version via its PyPI badge, so
there's nothing to bump there):

1. `pyproject.toml` → `version = "X.Y.Z"`
2. `slimx/__init__.py` → `__version__ = "X.Y.Z"`

## Checklist

```bash
# 1. Bump the two version locations above to X.Y.Z, and add a dated CHANGELOG entry.

# 2. Refresh the lockfile so it pins the new version.
uv lock

# 3. Validate (everything must pass).
uv run ruff check .
uv run pyright            # runs over the whole repo, tests included
uv run pytest -q
uv run python -m build    # confirms the package builds

# 4. Commit the release.
git add -A
git commit -m "release: vX.Y.Z"

# 5. Tag it (annotated) and push commit + tag.
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin main --follow-tags

# 6. Publish to PyPI.
rm -rf dist
uv run python -m build
uv run twine upload dist/*
```

## Notes

- **Run `pyright` over the whole repo**, not just `slimx/` — CI checks the tests too.
- **Tag every release.** Tags are what make a "release" real; a commit message that
  merely says "release: vX.Y.Z" without a matching tag is just a label. Current tags:
  `v0.4.1`, `v0.5.0`, `v0.6.0`. The `0.7.x` line shipped without tags — consider
  back-tagging the relevant commits, or simply tag from the next release forward.
- **Docs site.** If the change touches `docs/`, publish with `uv run mkdocs gh-deploy`
  (or let the docs workflow handle it). Mermaid diagrams render via the
  `pymdownx.superfences` config in `mkdocs.yml`.
- **Don't rewrite pushed history.** If a past commit is mislabeled (e.g. a lockfile
  bump titled "release: v0.7.0"), leave it — a cosmetic label isn't worth a force-push.

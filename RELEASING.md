# Releasing Cytnx

This guide is for maintainers cutting a tagged release. It is intentionally
short: the release itself is a single action — publishing a GitHub Release,
which pushes a `vMAJOR.MINOR.PATCH` tag — but two metadata files have to be
correct **in the tagged commit** first, so they are prepared and merged
before the release is published.

## What a `v*` tag triggers

Pushing a `vMAJOR.MINOR.PATCH` tag (which happens when you publish a GitHub
Release) fans out to three workflows:

| Workflow                   | Effect                                                                     |
| -------------------------- | ------------------------------------------------------------------------- |
| `release_pypi.yml`         | Builds wheels and publishes `cytnx X.Y.Z` to PyPI                         |
| `conda_build_release.yml`  | Builds and uploads the conda package                                      |
| `docs.yml`                 | Publishes the Sphinx docs to `gh-pages/X.Y.Z/` and updates the `gh-pages/stable/` permalink |

Two of these read version metadata from the repository at the tagged commit,
so that metadata must already be right when the tag is created:

- **The package version comes from `version.cmake`, not from the git tag.**
  scikit-build-core stamps `MAJOR.MINOR.PATCH` from `version.cmake` onto the
  PyPI/conda packages and onto `cytnx.__version__`. If `version.cmake` says
  `1.1.0` but you tag `v1.2.0`, PyPI publishes `1.1.0`.
- **The docs slug must match the published directory.** The version switcher
  and the documentation landing page build URLs as `<site_root>/<slug>/`,
  taking `<slug>` from the `version` field of `docs/site_root/versions.json`.
  `docs.yml` deploys release docs to a directory named after the tag **with
  the leading `v` stripped** (`v1.1.0` → `gh-pages/1.1.0/`), so the
  `versions.json` slug must be the numeric `1.1.0`, never `v1.1.0`. A
  `v`-prefixed slug links to a directory that does not exist and serves a 404.
- **`CITATION.cff` is read by people, not by a workflow.** Its `version` and
  `date-released` are what GitHub's "Cite this repository" widget and every
  downstream citation tool show. Nothing in the build consumes them, so a
  stale `version` there is invisible until a user cites the wrong release --
  which is how it sat at `1.0.0` through three later releases (1.0.1, 1.1.0,
  1.1.1).

Everything else derives its version from `version.cmake` at build time and
needs no edit: the PyPI/conda package version and `cytnx.__version__`
(scikit-build-core), and both documentation builds. The Sphinx user guide
(`docs/source/conf.py`) and the Doxygen API reference
(`docs/build_api_docs.py`) read it through `docs/cytnx_version.py`, which
shows `X.Y.Z` on the release tag and `X.Y.Z.devN+g<sha>` on the `dev` docs
built from `master`.

## Steps

1. **Run the bump script** with the new `MAJOR.MINOR.PATCH` (no leading `v`):

   ```sh
   python3 tools/bump_version.py 1.2.0
   ```

   It rewrites the three files above in one go — `version.cmake`, the new
   `{ "name": "1.2.0", "version": "1.2.0" }` entry in
   `docs/site_root/versions.json`, and `version` plus `date-released` in
   `CITATION.cff` (today's date, or pass `--date YYYY-MM-DD`) — and then runs
   `tools/check_release_consistency.py`.

   Keep the `dev` entry in `versions.json`. There is no separate `stable`
   entry to maintain: the switcher automatically labels the highest-numbered
   release `(stable)` and the documentation root redirects to it.
   (`gh-pages/stable/` still exists as a permalink to the latest release
   docs, maintained by `docs.yml`.) The `preferred-citation` block in
   `CITATION.cff` is left alone: it describes the SciPost paper, which does
   not change when a release ships.

2. **Open the change as a release-prep pull request and merge it.** The
   `Release metadata consistency` workflow runs the same check as the script:
   `version.cmake`, `versions.json`, and `CITATION.cff` must agree, no slug
   may carry a leading `v`, and the documentation builds must not hard-code a
   version. If anything was edited by hand and drifted, the PR check fails
   before the release goes out.

3. **Draft and publish the GitHub Release.** On GitHub: *Releases → Draft a
   new release*, create the tag `vMAJOR.MINOR.PATCH` targeting the merged
   release-prep commit on `master`, click *Generate release notes*, review,
   and publish. Publishing pushes the tag and starts the release workflows
   above.

Doing step 1 in a merged PR first means the tagged commit already holds
the correct `version.cmake`, `versions.json`, and `CITATION.cff`, and the
consistency check has already passed — so publishing the release is the
last action, not the first.

## After publishing

- Once the `docs.yml` run finishes, confirm `https://cytnx-dev.github.io/Cytnx/X.Y.Z/`
  resolves and that the version switcher lists the new release, marked `(stable)`.
- Once `release_pypi.yml` finishes, confirm `pip install cytnx==X.Y.Z`
  resolves.

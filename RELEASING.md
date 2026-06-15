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

## Steps

1. **Bump `version.cmake`** to the new `MAJOR.MINOR.PATCH`.

2. **Add the docs slug to `docs/site_root/versions.json`.** Insert an entry
   whose `version` (the URL slug) and `name` (the switcher label) are both the
   numeric version with **no leading `v`**:

   ```json
   { "name": "1.2.0", "version": "1.2.0" }
   ```

   Keep the `dev` entry. There is no separate `stable` entry to maintain:
   the switcher automatically labels the highest-numbered release `(stable)`
   and the documentation root redirects to it, so adding the new release entry
   is all that is needed. (`gh-pages/stable/` still exists as a permalink to
   the latest release docs, maintained by `docs.yml`.)

3. **Open steps 1 and 2 as a release-prep pull request and merge it.** The
   `Release metadata consistency` workflow checks that `version.cmake` and
   `versions.json` agree; if you forget step 1 or 2, or write a `v`-prefixed
   slug, the PR check fails before the release goes out. Run the same check
   locally with:

   ```sh
   python3 tools/check_release_consistency.py
   ```

4. **Draft and publish the GitHub Release.** On GitHub: *Releases → Draft a
   new release*, create the tag `vMAJOR.MINOR.PATCH` targeting the merged
   release-prep commit on `master`, click *Generate release notes*, review,
   and publish. Publishing pushes the tag and starts the release workflows
   above.

Doing steps 1–2 in a merged PR first means the tagged commit already holds
the correct `version.cmake` and `versions.json` and the consistency check has
already passed — so publishing the release is the last action, not the first.

## After publishing

- Once the `docs.yml` run finishes, confirm `https://cytnx-dev.github.io/Cytnx/X.Y.Z/`
  resolves and that the version switcher lists the new release, marked `(stable)`.
- Once `release_pypi.yml` finishes, confirm `pip install cytnx==X.Y.Z`
  resolves.

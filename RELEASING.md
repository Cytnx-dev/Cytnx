# Releasing Cytnx

This guide is for maintainers cutting a tagged release. It is intentionally
short: the release itself is a single action — publishing a GitHub Release,
which pushes a `vMAJOR.MINOR.PATCH` tag — but three metadata files have to
be correct **in the tagged commit** first, so they are prepared and merged
before the release is published.

## What a `v*` tag triggers

Pushing a `vMAJOR.MINOR.PATCH` tag (which happens when you publish a GitHub
Release) fans out to four release workflows (the `Release metadata
consistency` check also runs on the tag, see step 2):

| Workflow                   | Effect                                                                     |
| -------------------------- | ------------------------------------------------------------------------- |
| `release_pypi.yml`         | Builds wheels and publishes `cytnx X.Y.Z` to PyPI                         |
| `release_pypi_cuda.yml`    | Builds the Linux x86_64 CUDA wheel and publishes `cytnx-cuda X.Y.Z` to PyPI |
| `conda_build_release.yml`  | Builds and uploads the conda package to the `kaihsinwu` anaconda.org channel |
| `docs.yml`                 | Publishes the Sphinx docs to `gh-pages/X.Y.Z/` and updates the `gh-pages/stable/` permalink |

All of them read version metadata from the repository at the tagged commit,
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

Between releases `version.cmake` holds a version *above* the last release
(step 4 below), so with 1.1.1 released the `dev` docs read, for example,
`1.1.2.dev581+g<sha>`: 581 commits past the `v1.1.1` tag, labelled with the
version in `version.cmake`. The nightly wheels carry the same base version
(`1.1.2.devYYYYMMDDHHMM`, see `tools/prepare_nightly_release.py`). PEP 440
orders a `.devN` version *before* its base release, which is why
`version.cmake` must never equal the last release between releases: a
`1.1.1.devN` build would rank below 1.1.1 (#1204).

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
   `CITATION.cff` must cite the newest `versions.json` slug, `version.cmake`
   must not be behind that slug (after step 1 all three name the new
   release), no slug may carry a leading `v`, and the documentation builds
   must not hard-code a version. On the tag push the check is strict: the
   tag, `version.cmake`, and `CITATION.cff` must all agree. If anything was
   edited by hand and drifted, the PR check fails before the release goes
   out.

3. **Draft and publish the GitHub Release.** On GitHub: *Releases → Draft a
   new release*, create the tag `vMAJOR.MINOR.PATCH` targeting the merged
   release-prep commit on `master`, click *Generate release notes*, review,
   and publish. Publishing pushes the tag and starts the release workflows
   above.

4. **Advance `version.cmake` past the release.** Right after publishing,
   open and merge a one-line pull request made with

   ```sh
   python3 tools/bump_version.py 1.2.1 --next
   ```

   It changes only `version.cmake`; `versions.json` and `CITATION.cff`
   keep describing 1.2.0, the release that exists. From then on the nightly
   wheels are `1.2.1.devYYYYMMDDHHMM` and the `dev` docs read
   `1.2.1.devN+g<sha>`, both of which outrank 1.2.0. Skipping this step
   leaves every nightly versioned *below* the release, and `pip install
   --pre … cytnx` would install the release instead of the nightly (#1204).
   The consistency check accepts `version.cmake` running ahead of the newest
   release, but never behind it.

   Use the patch bump by default. The label only has to be *above* the last
   release, not equal to the next one: whatever ships next (1.2.1, 1.3.0 or
   2.0.0) is at least 1.2.1, so no later release can be outranked by the
   nightlies that came before it. Choose a higher label only once the next
   release number is settled; a label above what actually ships next (say
   `2.0.0.dev…` nightlies followed by a 1.3.0 release) leaves those
   nightlies ranked above the release until `tools/prune_nightly_wheels.py`
   has cycled them out.

Doing step 1 in a merged PR first means the tagged commit already holds
the correct `version.cmake`, `versions.json`, and `CITATION.cff`, and the
consistency check has already passed — so publishing the release is the
last action before the bump, not the first.

## After publishing

- Once the `docs.yml` run finishes, confirm `https://cytnx-dev.github.io/Cytnx/X.Y.Z/`
  resolves and that the version switcher lists the new release, marked `(stable)`.
- Once `release_pypi.yml` finishes, confirm `pip install cytnx==X.Y.Z`
  resolves.
- Once `release_pypi_cuda.yml` finishes, confirm
  `pip install cytnx-cuda==X.Y.Z` resolves (Linux x86_64 only).
- Once `conda_build_release.yml` finishes, confirm the package is listed by
  `conda search -c kaihsinwu cytnx` at `X.Y.Z`.
- Merge the post-release bump from step 4, then check that the next push to
  `master` publishes nightlies versioned above `X.Y.Z` (see below).

## Nightly builds

Every push to `master` runs `release_pypi.yml` and `release_pypi_cuda.yml`
on their nightly channel. `tools/prepare_nightly_release.py` stamps
`MAJOR.MINOR.PATCH.devYYYYMMDDHHMM` from `version.cmake` into
`pyproject.toml`, the wheels are uploaded to the `cytnx-nightly-wheels`
channel on anaconda.org (index URL
`https://pypi.anaconda.org/cytnx-nightly-wheels/simple`), never to PyPI,
and `tools/prune_nightly_wheels.py` keeps the channel at five versions.

Because `version.cmake` is above the last release between releases (step 4),
a nightly outranks every published release, so

```sh
pip install --pre --extra-index-url https://pypi.anaconda.org/cytnx-nightly-wheels/simple cytnx
```

selects the nightly rather than the PyPI release. `pytests/release_tools_test.py`
checks this resolution against local indexes for both `cytnx` and
`cytnx-cuda`, and `tools/check_release_consistency.py` fails any pull request
that drops `version.cmake` below the newest release.

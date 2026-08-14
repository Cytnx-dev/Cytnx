#!/usr/bin/env bash
set -euo pipefail

# NOTE:
# We install native dependencies via conda-forge (through a self-contained
# micromamba environment) rather than Homebrew. Two reasons:
#
#  * Homebrew's bottled dylibs encode whatever macOS deployment target
#    Homebrew itself currently targets for that formula, which trends
#    upward release over release -- Cytnx's actual minimum supported macOS
#    version was rising as a side effect of Homebrew's own build policy,
#    not a decision made here (#963). conda-forge's clang_osx-64 /
#    clang_osx-arm64 toolchain targets a pinned, deliberately old SDK, so
#    its packages carry a controlled, low minos regardless of the runner's
#    own OS version.
#  * conda-forge ships an explicit "openmp" build variant of OpenBLAS,
#    replacing Homebrew's pthreads-only build (see
#    tools/cibuildwheel_before_all.sh for why that matters with HPTT's
#    OpenMP usage).
#
# The dylib-minos-probing logic below is unchanged from the Homebrew
# version of this script; only the package source and prefix changed, so
# MACOSX_DEPLOYMENT_TARGET keeps tracking whatever floor the actual
# vendored dylibs require instead of a value guessed in advance.

# CMake refuses to export an INTERFACE_INCLUDE_DIRECTORIES entry that lives
# inside the source directory (a hard error: "which is prefixed in the
# source directory"), which the cytnx target needs to do for downstream
# find_package(Cytnx) consumers (see ci-downstream-find-package.yml). So
# this prefix must live outside ${PWD} (the checkout). A fixed, hardcoded
# path (rather than one derived from $TMPDIR) means
# pyproject.toml's [tool.cibuildwheel.macos].environment can reference it
# as a plain literal -- no dynamic discovery needed, and no dependency on
# cibuildwheel's environment-table evaluator supporting shell parameter
# expansion (it doesn't: only plain $VAR references and $(command)
# substitutions are honored there, which is why MACOSX_DEPLOYMENT_TARGET
# below still goes through a file instead of a literal, since its value
# is only known after probing the installed dylibs).
deps_prefix="/tmp/cytnx-deps"
# ccache is a build-only tool, never linked into or vendored with the
# wheel, so it lives in its own prefix -- kept out of ${deps_prefix} so the
# minos probe below (which scans every dylib under ${deps_prefix}/lib)
# only ever sees the wheel's actual runtime dependency closure. Installing
# it in the same env as openblas/arpack/boost/llvm-openmp would let one of
# ccache's own transitive deps (built with a newer minos than the actual
# runtime libs) needlessly raise MACOSX_DEPLOYMENT_TARGET.
build_tools_prefix="/tmp/cytnx-build-tools"

arch="$(uname -m)"
if [[ "${arch}" == "arm64" ]]; then
  conda_subdir="osx-arm64"
else
  conda_subdir="osx-64"
fi

curl -fLs "https://micro.mamba.pm/api/micromamba/${conda_subdir}/latest" | tar -xj bin/micromamba
export MAMBA_ROOT_PREFIX="/tmp/cytnx-micromamba"
./bin/micromamba create -y -p "${deps_prefix}" -c conda-forge \
  "openblas=*=*openmp*" \
  liblapacke \
  "arpack=*=nompi*" \
  libboost-headers \
  llvm-openmp
./bin/micromamba create -y -p "${build_tools_prefix}" -c conda-forge ccache

report_file="${PWD}/.cibw_macos_condaforge_minos_report.txt"
target_file="${PWD}/.cibw_macos_deployment_target.txt"

# Truncate/create report file before appending package diagnostics.
: > "${report_file}"

max_minos="0.0"

# macOS ships bash 3.2 which lacks `mapfile`, so read into arrays with a
# while-loop instead.
while IFS= read -r dylib; do
  minos="$(vtool -show-build "${dylib}" 2>/dev/null | awk '/minos/ {print $2; exit}')"
  minos="${minos:-unknown}"
  echo "  ${minos}  $(basename "${dylib}")" | tee -a "${report_file}"

  if [[ "${minos}" != "unknown" ]]; then
    if [[ "$(printf '%s\n' "${max_minos}" "${minos}" | sort -V | tail -n1)" == "${minos}" ]]; then
      max_minos="${minos}"
    fi
  fi
# -L follows conda-forge's own symlinks so dylibs nested a level deeper are
# still picked up. Keep -maxdepth bounded to guard against symlink-induced
# recursion.
done < <(find -L "${deps_prefix}/lib" -maxdepth 4 -name "*.dylib" -type f)

# Use the highest minos observed to avoid building wheels that reference a
# vendored dylib requiring a newer macOS than the deployment target.
echo "${max_minos}" > "${target_file}"
echo "Computed MACOSX_DEPLOYMENT_TARGET=${max_minos}" | tee -a "${report_file}"
echo "Wrote minos report: ${report_file}"
echo "Wrote deployment target: ${target_file}"

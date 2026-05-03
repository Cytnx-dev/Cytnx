#!/usr/bin/env bash
set -euo pipefail

# NOTE:
# We intentionally install dependencies via Homebrew here. Their dylibs may
# encode a minimum supported macOS version (minos). We inspect that metadata
# and persist both a report and an effective deployment target for cibuildwheel.

probe_packages=(arpack boost openblas)
install_only_packages=(ccache libomp)
install_packages=("${probe_packages[@]}" "${install_only_packages[@]}")

brew update
brew install "${install_packages[@]}"

report_file="${PWD}/.cibw_macos_brew_minos_report.txt"
target_file="${PWD}/.cibw_macos_deployment_target.txt"

# Truncate/create report file before appending package diagnostics.
: > "${report_file}"

max_minos="0.0"

# Expand the probe set with the full transitive runtime dependency closure,
# since dylibs we link against may in turn load deeper brewed dylibs whose
# minos is higher than the direct dependencies'. macOS ships bash 3.2 which
# lacks `mapfile`, so read into arrays with a while-loop instead.
transitive_deps=()
while IFS= read -r dep; do
  transitive_deps+=("${dep}")
done < <(brew deps --union "${probe_packages[@]}")

all_probe_packages=()
while IFS= read -r pkg; do
  all_probe_packages+=("${pkg}")
done < <(printf '%s\n' "${probe_packages[@]}" "${transitive_deps[@]}" | awk 'NF && !seen[$0]++')

for pkg in "${all_probe_packages[@]}"; do
  echo "### ${pkg} ###" | tee -a "${report_file}"
  prefix="$(brew --prefix "${pkg}")"
  if [[ -d "${prefix}/lib" ]]; then
    while IFS= read -r dylib; do
      minos="$(vtool -show-build "${dylib}" 2>/dev/null | awk '/minos/ {print $2; exit}')"
      minos="${minos:-unknown}"
      echo "  ${minos}  $(basename "${dylib}")" | tee -a "${report_file}"

      if [[ "${minos}" != "unknown" ]]; then
        if [[ "$(printf '%s\n' "${max_minos}" "${minos}" | sort -V | tail -n1)" == "${minos}" ]]; then
          max_minos="${minos}"
        fi
      fi
    # -L follows brew's version symlinks (e.g. gcc's lib/gcc/current -> 15)
    # so dylibs nested under lib/<formula>/<version>/ such as libgfortran,
    # libquadmath and libgcc_s are picked up. Keep -maxdepth bounded to
    # guard against symlink-induced recursion.
    done < <(find -L "${prefix}/lib" -maxdepth 4 -name "*.dylib" -type f)
  fi
  echo | tee -a "${report_file}"
done

# Use the highest minos observed to avoid building wheels that reference
# brewed dylibs requiring a newer macOS than the deployment target.
echo "${max_minos}" > "${target_file}"
echo "Computed MACOSX_DEPLOYMENT_TARGET=${max_minos}" | tee -a "${report_file}"
echo "Wrote minos report: ${report_file}"
echo "Wrote deployment target: ${target_file}"

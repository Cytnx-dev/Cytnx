#!/usr/bin/env bash
set -euo pipefail

# NOTE:
# We intentionally install dependencies via Homebrew here. Their dylibs may
# encode a minimum supported macOS version (minos). We inspect that metadata
# and persist both a report and an effective deployment target for cibuildwheel.

brew_update_and_install_packages=(arpack boost ccache libomp openblas)
minos_probe_packages=(arpack boost libomp openblas)

brew update
brew install "${brew_update_and_install_packages[@]}"

report_file="${PWD}/.cibw_macos_brew_minos_report.txt"
target_file="${PWD}/.cibw_macos_deployment_target.txt"

# Truncate/create report file before appending package diagnostics.
: > "${report_file}"

max_minos="0.0"

for pkg in "${minos_probe_packages[@]}"; do
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
    done < <(find "${prefix}/lib" -maxdepth 2 -name "*.dylib" -type f)
  fi
  echo | tee -a "${report_file}"
done

# Use the highest minos observed to avoid building wheels that reference
# brewed dylibs requiring a newer macOS than the deployment target.
echo "${max_minos}" > "${target_file}"
echo "Computed MACOSX_DEPLOYMENT_TARGET=${max_minos}" | tee -a "${report_file}"
echo "Wrote minos report: ${report_file}"
echo "Wrote deployment target: ${target_file}"

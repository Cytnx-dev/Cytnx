set -xe
# Debug: find available package managers
echo "=== Checking for package managers ==="
ls -la /usr/bin/dnf /usr/bin/yum /usr/bin/microdnf /usr/bin/apt-get 2>/dev/null || true
echo "=== Checking PATH ==="
echo $PATH
echo "=== Looking for *dnf* or *yum* in /usr/bin ==="
ls /usr/bin/*dnf* /usr/bin/*yum* 2>/dev/null || echo "None found"
echo "=== Check if packages are pre-installed ==="
rpm -qa | grep -E "(boost|openblas|arpack)" || echo "Packages not found via rpm"
echo "=== Check /usr/include/openblas ==="
ls /usr/include/openblas 2>/dev/null || echo "openblas headers not found"

set -xe

# Install required packages for manylinux_2_28+ (AlmaLinux/RHEL).
if command -v dnf >/dev/null 2>&1; then
    dnf install -y boost-devel openblas-devel arpack-devel ccache
# musllinux_1_2 images are Alpine-based, so use apk there.
elif command -v apk >/dev/null 2>&1; then
    apk update
    apk add boost-dev openblas-dev arpack-dev ccache
else
    echo "Unsupported package manager: expected dnf or apk" >&2
    exit 1
fi

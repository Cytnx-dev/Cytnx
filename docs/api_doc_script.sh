#!/usr/bin/bash

doxygen_build()
{
	VTAG=$1
	echo $VTAG
	git checkout $VTAG || {
    echo "Failed to checkout tag $VTAG";
    exit 1;
	}
	doxygen docs.doxygen > /dev/null 2>&1
	if [ "$VTAG" = master ]; then
		mkdir -p docs/api_docs/versions
		VTAG=latest
	fi
	mv docs/html docs/api_docs/versions/$VTAG
}

# get current branch
branch=$(git rev-parse --abbrev-ref HEAD)

mkdir build
cd ../

# Build latest version
doxygen_build master

smallest_ver=0.7.3
# Get all version latest than smallest_ver
versions=($(git tag \
  | sed 's/^v//' \
  | sed 's/[A-Za-z].*$//' \
  | sort -V \
  | awk -F. -v cv="$smallest_ver" '
      BEGIN { split(cv,c,"."); min=c[1]*10000+c[2]*100+c[3] }
      { val=$1*10000+$2*100+$3; if(val>=min) print }
  '))

# Build older version latest from smallest_ver
for i in "${!versions[@]}"; do
  ver="v${versions[i]}"
  doxygen_build $ver
done

# checkout original version
git checkout $branch

# Create index.rst
cd ./docs/api_docs/home_source
cat > index.rst <<'EOF'
.. image:: Icon_small.png
    :width: 350

Cytnx API
=================================
    Cytnx is a library design for Quantum physics simulation using GPUs and CPUs.

* `Latest version <versions/latest/index.html>`__.

Older versions:

EOF

for ((i=${#versions[@]}-1; i>=0; i--)); do
  ver="v${versions[i]}"
	echo '	* `'$ver' <versions/'$ver'/index.html>`__.' >> index.rst
done

# build API documentation index.
cd ../
make html

# move the api_build under build
mv versions ./build_home/html/
mv build_home ../build/api_build

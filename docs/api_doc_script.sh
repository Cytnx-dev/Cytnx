#!/usr/bin/env bash

doxygen_build()
{
	VTAG=$1
	echo $VTAG
	git checkout $VTAG
	doxygen docs.doxygen > /dev/null 2>&1
	if [ "$VTAG" = master ]; then
		mkdir -p docs/api_docs/versions
		VTAG=latest
	fi
	mv docs/html docs/api_docs/versions/$VTAG
}

branch=$(git rev-parse --abbrev-ref HEAD)
echo "$branch"

rm -r api_docs/versions
cd ../
doxygen_build master
crit_ver=0.9.5
versions=($(git tag \
  | sed 's/^v//' \
  | sed 's/[A-Za-z].*$//' \
  | sort -V \
  | awk -F. -v cv="$crit_ver" '
      BEGIN { split(cv,c,"."); min=c[1]*10000+c[2]*100+c[3] }
      { val=$1*10000+$2*100+$3; if(val>=min) print }
  '))

for i in "${!versions[@]}"; do
  ver="v${versions[i]}"
  doxygen_build $ver
done

git checkout $branch

cd ./docs/api_docs/home_source
cat > index.rst <<'EOF'
.. image:: Icon_small.png
    :width: 350

Cytnx API
=================================
    Cytnx is a library design for Quantum physics simulation using GPUs and CPUs.

* `Latest version <../../versions/latest/index.html>`__.

Older versions:

EOF

for ((i=${#versions[@]}-1; i>=0; i--)); do
  ver="v${versions[i]}"
	echo '	* `'$ver' <../../versions/'$ver'/index.html>`__.' >> index.rst
done
cd ../
make html



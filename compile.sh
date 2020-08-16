export CYTNX_INC=$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_include__)\")")
export CYTNX_LIB=$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_lib__)\")")
export CYTNX_LINK="$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_linkflags__)\")")"

g++ -std=c++11 -I${CYTNX_INC} test.cpp ${CYTNX_LIB}/libcytnx.a ${CYTNX_LINK} -o test


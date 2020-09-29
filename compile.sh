export CYTNX_INC=$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_include__)\")")
export CYTNX_LIB=$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_lib__)\")")
export CYTNX_LINK="$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_linkflags__)\")")"
export CYTNX_CXXFLAGS="$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_flags__)\")")"

g++ -I${CYTNX_INC} ${CYTNX_CXXFLAGS} test.cpp ${CYTNX_LIB}/libcytnx.a ${CYTNX_LINK} -o test


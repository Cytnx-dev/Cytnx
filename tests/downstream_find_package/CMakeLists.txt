# Minimal downstream project that consumes cytnx purely through
# `find_package(Cytnx CONFIG)` and the exported `Cytnx::cytnx` target.
#
# Its only job is to prove that the package cytnx exports is self-contained:
# a consumer who does nothing but `find_package(Cytnx)` +
# `target_link_libraries(app Cytnx::cytnx)` can compile, link, and run
# against cytnx — without naming Boost, OpenMP, LAPACK, or HPTT itself.
# That transitively checks:
#   * CytnxConfig.cmake re-finds the imported-target dependencies
#     (Boost, and OpenMP when USE_HPTT=ON) via find_dependency.
#   * CytnxTargets.cmake carries cytnx's INTERFACE_LINK_LIBRARIES, including
#     the bundled `Cytnx::hptt_static` ($<LINK_ONLY:>) it needs at link time.
#
# The same project is driven against both the build-tree export and the
# install-tree export by the "Tests and Codecov" workflow
# (.github/workflows/ci-cmake_tests.yml).
cmake_minimum_required(VERSION 3.24)
project(cytnx_downstream_consumer LANGUAGES CXX)

find_package(Cytnx CONFIG REQUIRED)

add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE Cytnx::cytnx)

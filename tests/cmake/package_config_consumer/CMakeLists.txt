cmake_minimum_required(VERSION 3.20)
project(CytnxPackageConfigConsumer LANGUAGES CXX)

find_package(Cytnx REQUIRED)

add_executable(cytnx_package_config_consumer main.cpp)
target_link_libraries(cytnx_package_config_consumer PRIVATE ${Cytnx_Libraries})

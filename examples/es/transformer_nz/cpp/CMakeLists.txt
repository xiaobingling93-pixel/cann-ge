cmake_minimum_required(VERSION 3.16)

project(samples)
set(ASCEND_INSTALL_PATH $ENV{ASCEND_HOME_PATH})
set(ASCEND_LIB_DIR ${ASCEND_INSTALL_PATH}/lib64)
message(STATUS "ASCEND_LIB_DIR = ${ASCEND_LIB_DIR}")

set(CMAKE_CXX_STANDARD 17) # C++ 17 的项目
add_compile_definitions(
        _GLIBCXX_USE_CXX11_ABI=0
        google=ascend_private
)

include_directories(${ASCEND_INSTALL_PATH})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_PREFIX_PATH  ${ASCEND_LIB_DIR} ${CMAKE_PREFIX_PATH})

find_library(GE_COMPILE NAMES ge_compiler REQUIRED)
find_library(GE_RUNNER NAMES ge_runner REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${ASCEND_INSTALL_PATH}/include/ge/cmake")
find_package(GenerateEsPackage REQUIRED)
add_library(opgraph_all INTERFACE)
# 使用run包的so生成es的构图api，因当前版本没有集成es的api，所以需要手动调用cmake函数生成
set_target_properties(opgraph_all PROPERTIES
        INTERFACE_LIBRARY_OUTPUT_DIRECTORY  "${ASCEND_INSTALL_PATH}/opp/built-in/op_proto"
)
add_es_library(
        ES_LINKABLE_AND_ALL_TARGET es_all
        OPP_PROTO_TARGET  opgraph_all
        OUTPUT_PATH       ${CMAKE_BINARY_DIR}/output
)
add_subdirectory(src)
add_executable(sample main.cpp)
target_link_libraries(sample PRIVATE es_showcase ${GE_RUNNER} ${GE_COMPILE})

cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(ascendc_add_custom LANGUAGES ASC CXX)

set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
set(INCLUDE_DIR ${ASCEND_PATH}/include)

set(common_compile_options
    -std=c++17
    -Wall
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-deprecated-declarations")

set(common_compile_definitions
    _GLIBCXX_USE_CXX11_ABI=0
)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH
)
set(TORCH_NPU_INCLUDE_DIRS ${TORCH_NPU_PATH}/include)
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_PATH}/lib)

add_definitions(-D__ASCEND_C__)

add_library(${PROJECT_NAME} SHARED
    add_custom_kernel/add_custom.asc
    add_custom_kernel/custom_op.cpp
)

target_compile_definitions(${PROJECT_NAME} PRIVATE __ASCEND_C__)

# 固定.so名字
set_target_properties(${PROJECT_NAME} PROPERTIES
        OUTPUT_NAME cust_opapi
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
    ${INCLUDE_DIR}
    ${INCLUDE_DIR}/graph
    ${INCLUDE_DIR}/register
    ${INCLUDE_DIR}/external
    ${INCLUDE_DIR}/experiment
    ${INCLUDE_DIR}/experiment/platform
    ${INCLUDE_DIR}/experiment/runtime
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    -Wl,--no-as-needed
    torch_npu
    Python3::Python
    ascendcl
    ascendc_runtime
    acl_rt
    ge_compiler
    ge_common
    ge_executor
    ge_runner
    ge_runner_v2
    runtime
    pthread
    dl
    c10
    lowering
    -Wl,--as-needed
)

target_link_directories(${PROJECT_NAME} PRIVATE
    ${ASCEND_PATH}/lib64/
    ${TORCH_NPU_LIBRARIES}
)

target_compile_options(${PROJECT_NAME} PRIVATE
    ${TORCH_CXX_FLAGS}
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)


set(CUSTOM_INSTALL_PATH ${CMAKE_SOURCE_DIR}/build)
set(OPS_INFO_PATH ${CMAKE_SOURCE_DIR}/add_custom_kernel)


message(STATUS "CUSTOM_INSTALL_PATH: ${CUSTOM_INSTALL_PATH}")
message(STATUS "OPS_INFO_PATH: ${OPS_INFO_PATH}")

add_npu_support_target(
    OPS_INFO_DIR ${OPS_INFO_PATH}
    OUT_DIR ${CUSTOM_INSTALL_PATH}
)

install(
    TARGETS ${PROJECT_NAME}
    DESTINATION ${CUSTOM_INSTALL_PATH}
)

set(MY_PROJECT_NAME ${PROJECT_NAME})
set(MY_CUSTOM_INSTALL_PATH ${CUSTOM_INSTALL_PATH})

install(CODE "
    message(STATUS \"\")
    message(STATUS \"===================================================\")
    message(STATUS \"Installation of ${MY_PROJECT_NAME} completed!\")
    message(STATUS \"Generated files are:\")
    message(STATUS \"  ${MY_CUSTOM_INSTALL_PATH}/libcust_opapi.so\")
    message(STATUS \"Please set environment variables:\")
    message(STATUS \"  export ASCEND_CUSTOM_OPP_PATH=${MY_CUSTOM_INSTALL_PATH}:\$ASCEND_CUSTOM_OPP_PATH\")
")
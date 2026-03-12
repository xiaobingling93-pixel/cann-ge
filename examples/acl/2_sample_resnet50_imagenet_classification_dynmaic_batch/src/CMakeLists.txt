# Copyright (c) Huawei Technologies Co., Ltd. 2015. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(ACL_RESNET50_DYNAMIC_BATCH)

# Compile options
add_compile_options(-std=c++17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

set(INC_PATH $ENV{ASCEND_HOME_PATH})

if (NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(WARNING "please source the set_env.sh")
    set(INC_PATH "/usr/local/Ascend/cann")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "set INC_PATH: ${INC_PATH}")
endif ()

# Dynamic libraries in the stub directory can only be used for compilation
set(LIB_PATH "${INC_PATH}/devlib")
message(STATUS "set LIB_PATH: ${LIB_PATH}")

# Header path
include_directories(
    ${INC_PATH}/include/
    ../inc/
)

# add host lib path
link_directories(
    ${LIB_PATH}
)

add_executable(main
        ../../common/utils.cpp
        ../../common/sampleModel.cpp
        sample_resnet50_imagenet_classification_dynamic_batch.cpp)

target_link_libraries(main acl_rt acl_mdl stdc++)

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

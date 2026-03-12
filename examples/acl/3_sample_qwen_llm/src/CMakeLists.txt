# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(ACL_QWEN_LLM)

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
        sample_qwen_llm.cpp)

target_link_libraries(main acl_rt acl_mdl stdc++)

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

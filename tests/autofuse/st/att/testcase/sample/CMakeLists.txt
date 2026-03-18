# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)
project (SampleProject)
add_compile_options(-std=c++17)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -fsanitize=address -fsanitize=leak -fsanitize-recover=address")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address -fsanitize=leak -fsanitize-recover=address")
include(CMakePrintHelpers)
set(SAMPLE_PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(TOP_DIR ${ATT_PROJECT_PATH}/../)
message(STATUS "Using environment variable ATT_PROJECT_PATH: ${ATT_PROJECT_PATH}")

string(TOLOWER ${CMAKE_SYSTEM_NAME} system_name_lower)
if (NOT DEFINED CMAKE_INCLUDE_PATH)
    set(CMAKE_INCLUDE_PATH
            ${ASCEND_INSTALL_PATH}/${CMAKE_SYSTEM_PROCESSOR}-${system_name_lower}/include
            )
endif()
if (NOT DEFINED CMAKE_LIBRARY_PATH)
    set(CMAKE_LIBRARY_PATH
            ${ASCEND_INSTALL_PATH}/${CMAKE_SYSTEM_PROCESSOR}-${system_name_lower}/
            )
endif()

if (NOT DEFINED CMAKE_MODULE_PATH)
    set(CMAKE_MODULE_PATH
            ${ASCEND_3RD_LIB_PATH}/cmake/modules
            ${GE_METADEF_DIR}/cmake/modules
            ${CMAKE_CURRENT_LIST_DIR}/depends/cmake/modules
            ${TOP_DIR}/cmake/third_party/scripts/modules
            )
endif()
if (NOT DEFINED CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH
            ${ASCEND_3RD_LIB_PATH}/symengine/lib/cmake/symengine
            ${ASCEND_3RD_LIB_PATH}/protoc
            ${ASCEND_3RD_LIB_PATH}/protobuf_static
            ${ASCEND_3RD_LIB_PATH}/ascend_protobuf
            ${ASCEND_3RD_LIB_PATH}/ascend_protobuf_static
            )
endif()
message(STATUS "Using environment variable SAMPLE_PROJECT_PATH: ${SAMPLE_PROJECT_PATH}")
message(STATUS "Using environment variable LD_LIBRARY_PATH: $ENV{LD_LIBRARY_PATH}")
message(STATUS "Using environment variable ASCEND_INSTALL_PATH: ${ASCEND_INSTALL_PATH}")
message(STATUS "Using environment variable GEN_TARGET: ${GEN_TARGET}")
message(STATUS "Using environment variable ASCEND_3RD_LIB_PATH: ${ASCEND_3RD_LIB_PATH}")
message(STATUS "Using environment variable:")
cmake_print_variables(CMAKE_MODULE_PATH)
cmake_print_variables(CMAKE_PREFIX_PATH)
cmake_print_variables(CMAKE_INCLUDE_PATH)
cmake_print_variables(CMAKE_LIBRARY_PATH)
find_package(SymEngine CONFIG REQUIRED)
find_package(securec MODULE REQUIRED)
find_package(ascend_protobuf_shared MODULE REQUIRED)
find_package(ascend_protobuf_static MODULE REQUIRED)
find_package(unified_dlog MODULE REQUIRED)
find_package(mmpa MODULE REQUIRED)
find_package(atrace MODULE REQUIRED)
add_subdirectory("${TOP_DIR}/tests/depends/slog" "slog_build")
add_subdirectory("${TOP_DIR}/tests/depends/trace" "trace_build")

foreach(dir $ENV{LD_LIBRARY_PATH})
    file(TO_CMAKE_PATH "${dir}" dir)
    message(STATUS "Using environment variable enum: ${dir}")
    link_directories(${dir})
endforeach()

set(CMAKE_VERBOSE_MAKEFILE ON)

file(GLOB USER_SOURCES
        ${SAMPLE_PROJECT_PATH}/user_src/*.cpp
        )

file(GLOB SAMPLE_PROJECT_COMMON_SOURCE
        ${SAMPLE_PROJECT_PATH}/depends/common/*.cpp
        )

file(GLOB SAMPLE_PROJECT_CODE_GEN_SOURCES
        ${SAMPLE_PROJECT_PATH}/gen_code_st/*.cpp
        ${SAMPLE_PROJECT_PATH}/gen_code_st/utils/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/utils/*.cpp
        ${SAMPLE_PROJECT_COMMON_SOURCE}
        )

set(COMMON_COMPILE_OPTIONS
        -fPIC
        -Werror
        -fno-common
        -Wextra
        -Wfloat-equal
        -g -O0
        )
set(COMMON_LINK_OPTIONS
        -rdynamic
        -Wl,-Bsymbolic
        -Wl,--exclude-libs,-rpath
        -L ${ASCEND_3RD_LIB_PATH}/symengine/lib
        )

set(ops_header_dir ${TOP_DIR}/build/ascir_builtin_ops)

add_executable(code_gen
        ${USER_SOURCES}
        ${SAMPLE_PROJECT_CODE_GEN_SOURCES}
        )
target_include_directories(code_gen PRIVATE
        ${SAMPLE_PROJECT_PATH}
        ${SAMPLE_PROJECT_PATH}/user_inc
        ${SAMPLE_PROJECT_PATH}/depends/fake
        ${SAMPLE_PROJECT_PATH}/depends/common
        ${TOP_DIR}/ascir/meta/
        ${TOP_DIR}/
        ${TOP_DIR}/base
        ${TOP_DIR}/common
        ${TOP_DIR}/build/proto/metadef_protos
        ${TOP_DIR}/build/ascir_builtin_ops
        ${GE_METADEF_DIR}/inc/graph/utils/
        ${GE_METADEF_DIR}
        ${GE_METADEF_DIR}/inc
        ${GE_METADEF_DIR}/inc/graph
        ${GE_METADEF_DIR}/inc/graph/ascendc_ir
        ${GE_METADEF_DIR}/inc/external
        ${GE_METADEF_DIR}/graph/
        ${GE_METADEF_DIR}/graph/debug
        ${GE_METADEF_DIR}/graph/ascendc_ir
        ${ATT_PROJECT_PATH}
        ${ATT_PROJECT_PATH}/gen_model_info
        ${ATT_PROJECT_PATH}/gen_model_info/api_perf_register/
        ${ASCEND_INSTALL_PATH}/include
        ${ASCEND_INSTALL_PATH}/include/experiment/runtime
        ${ASCEND_INSTALL_PATH}/include/experiment/msprof
        ${ASCEND_3RD_LIB_PATH}/json/include
        )

target_compile_options(code_gen PRIVATE ${COMMON_COMPILE_OPTIONS})
target_link_options(code_gen PRIVATE ${COMMON_LINK_OPTIONS})
target_link_libraries(code_gen PRIVATE
        aihac_ir_register # depend on metadef
        c_sec
        unified_dlog
        graph_base
        graph
        metadef
        symengine # depend on metadef
        aihac_ir
        aihac_symbolizer
        error_manager
        ascend_protobuf
        atrace_stub
        )
target_link_libraries(code_gen PRIVATE ${TOP_DIR}/build/tests/libatt.so)

file(GLOB SAMPLE_PROJECT_DATA_GEN_SOURCE
        ${SAMPLE_PROJECT_PATH}/gen_data_st/*.cpp
        ${SAMPLE_PROJECT_PATH}/depends/fake/*.cpp
        ${TOP_DIR}/tests/autofuse/st/att/testcase/check_input/stub_node.cpp
        )

add_executable(gen_struct
        ${SAMPLE_PROJECT_PATH}/depends/binary/create_struct.cpp
        )
target_compile_options(gen_struct PRIVATE ${COMMON_COMPILE_OPTIONS})

# all st file copy to st_project dir
# file tiling_func_st.cpp
# file tiling_data.h(in st_project)
# file tiling_func.cpp
# file input_shapes.json
add_executable(data_gen
        ${SAMPLE_PROJECT_DATA_GEN_SOURCE}
        ${SAMPLE_PROJECT_COMMON_SOURCE}
        )
target_compile_options(data_gen PRIVATE -DDEBUG ${COMMON_COMPILE_OPTIONS})
target_link_options(data_gen PRIVATE ${COMMON_LINK_OPTIONS})
target_include_directories(data_gen PRIVATE
        ${SAMPLE_PROJECT_PATH}
        ${SAMPLE_PROJECT_PATH}/user_inc
        ${SAMPLE_PROJECT_PATH}/gen_data_st/stub
        ${SAMPLE_PROJECT_PATH}/depends/fake
        ${SAMPLE_PROJECT_PATH}/depends/common
        ${ASCEND_3RD_LIB_PATH}/json/include
        ${ASCEND_INSTALL_PATH}/include
        ${GE_METADEF_DIR}
        ${GE_METADEF_DIR}/inc
        ${GE_METADEF_DIR}/inc/external
        ${TOP_DIR}/tests/autofuse/st/att/testcase
        )
# tiling context depend on platform, need to mock platform and c_sec
target_link_libraries(data_gen PRIVATE
        $<BUILD_INTERFACE:mmpa_headers>
        graph_base
        graph
        metadef
        error_manager
        exe_graph
        lowering
        register
        unified_dlog
        mmpa
        c_sec
        )
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)
project(udf)

if(NOT DEFINED TOP_DIR)
    set(TOP_DIR ${CMAKE_SOURCE_DIR})
    message(STATUS "TOP_DIR is not exists, set it to ${CMAKE_SOURCE_DIR}.")
endif()

if (TARGET udf_executor)
    message(STATUS "Target udf_executor already exists, not use udf_executor in ge.")
    return()
endif ()

if (PRODUCT_SIDE STREQUAL "device")
    include(cmake/device/seccomp.cmake)
endif()

add_subdirectory(reader_writer)
add_subdirectory(flow_func)
add_subdirectory(built_in)
add_subdirectory(stubs)
add_subdirectory(toolchain)

add_library(udf_headers INTERFACE)
target_include_directories(udf_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc/external>
)

# To prevent conflict with air proto when udf call ge, udf add "ff_" for all proto file and add "ff." to proto package
set(proto_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/proto/ff_udf_attr.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/proto/ff_udf_def.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/proto/ff_deployer.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/proto/ff_dynamic_sched.proto
)

protobuf_generate(udf_exe PROTO_SRCS PROTO_HDRS ${proto_src_files})

set(udf_src_file
    ${CMAKE_CURRENT_SOURCE_DIR}/execute/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execute/flow_func_drv_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execute/flow_func_executor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execute/flow_func_thread_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execute/flow_model_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execute/memory_statistic_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execute/npu_sched_processor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/model/attr_value_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/model/flow_func_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/config/global_config.cpp
    ${PROTO_SRCS}
)

SET(udf_inc_path
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/external
    ${CMAKE_BINARY_DIR}/proto/udf_exe/proto
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../dflow/inc
)

add_executable(udf_executor
    ${udf_src_file}
)

target_include_directories(udf_executor PRIVATE
    ${udf_inc_path}
)

target_compile_definitions(udf_executor PRIVATE
    google=ascend_private
    $<$<STREQUAL:${PRODUCT_SIDE},device>:RUN_ON_DEVICE>
)

target_compile_options(udf_executor PRIVATE
    -O2
    -fno-strict-aliasing
    -fvisibility=hidden
    -Wno-unused-parameter   # protobuf has many unused-parameter alarms
    -Wno-array-bounds       # protobuf has array-bounds alarms on higher g++
    -Werror
    -Wextra
    -Wfloat-equal
    -fno-common
)

target_link_libraries(udf_executor PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:slog_headers>
    $<BUILD_INTERFACE:mmpa_headers>
    $<BUILD_INTERFACE:air_headers>
    $<BUILD_INTERFACE:ascend_hal_headers>
    PUBLIC c_sec ascend_protobuf_static
    $<IF:$<STREQUAL:${PRODUCT_SIDE},device>,alog,unified_dlog>
    reader_writer
    flow_func
    mmpa
    ascend_hal_stub
    $<$<STREQUAL:${PRODUCT_SIDE},device>:stackcore>
    udf_dump
    -ldl
    $<$<STREQUAL:${PRODUCT_SIDE},device>:seccomp_stub>
)

set(INSTALL_LIBRARY_DIR lib)
set(INSTALL_RUNTIME_DIR lib)

install(TARGETS udf_executor OPTIONAL
    EXPORT udf_executor-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}
)

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set(proto_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/../../proto/ff_dump_data.proto
)

protobuf_generate(dumpdata PROTO_SRCS PROTO_HDRS ${proto_src_files})

set(src_device
    udf_dump_task_device.cpp
)

set(src_host
    udf_dump_task_host.cpp
)

###for adump so
set(udf_dump_files
    $<$<STREQUAL:${PRODUCT_SIDE},device>:${src_device}>
    $<$<NOT:$<STREQUAL:${PRODUCT_SIDE},device>>:${src_host}>
    udf_dump_manager.cpp
    udf_dump.cpp
    ${PROTO_SRCS}
)

##for so inc path
set(dump_inc_path
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../../
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc/external
    ${CMAKE_BINARY_DIR}/proto/dumpdata/proto
)

add_library(udf_dump SHARED
    ${udf_dump_files}
)

target_include_directories(udf_dump PRIVATE
    ${dump_inc_path}
)

target_compile_definitions(udf_dump PRIVATE
    google=ascend_private
    $<$<STREQUAL:${PRODUCT_SIDE},device>:RUN_ON_DEVICE>
)

target_compile_options(udf_dump PRIVATE
    -ftrapv
    -O2
    -fvisibility-inlines-hidden
    -fvisibility=hidden
    -Wno-unused-parameter   # protobuf has many unused-parameter alarms
    -Wno-array-bounds       # protobuf has array-bounds alarms on higher g++
    -Werror
    -Wextra
    -Wfloat-equal
    -fno-common
)

target_link_libraries(udf_dump PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:slog_headers>
    $<BUILD_INTERFACE:mmpa_headers>
    $<BUILD_INTERFACE:ascend_hal_headers>
    $<BUILD_INTERFACE:c_sec_headers>
    $<BUILD_INTERFACE:datagw_headers>
    PUBLIC c_sec
    ascend_protobuf_static
    flow_func
    $<IF:$<STREQUAL:${PRODUCT_SIDE},device>,alog,unified_dlog>
)

set(INSTALL_LIBRARY_DIR lib)

install(TARGETS udf_dump OPTIONAL
    EXPORT udf_dump-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}
)

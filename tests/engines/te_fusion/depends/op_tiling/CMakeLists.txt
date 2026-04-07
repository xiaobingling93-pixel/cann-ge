# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

project(op_tiling_rt2)

file(GLOB_RECURSE PROTO_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "${METADEF_PROTO_DIR}/op_mapping.proto"
        "${METADEF_PROTO_DIR}/task.proto")

protobuf_generate(op_tiling_rt2_proto PROTO_SRCS PROTO_HDRS ${PROTO_LIST} TARGET)

set(FE_TILING_CPP
    op_tiling_rt2_stub.cc)

# compile for fe_st
add_library(op_tiling_rt2_stub SHARED 
            ${FE_TILING_CPP}
            ${PROTO_HDRS})

target_link_directories(op_tiling_rt2_stub PUBLIC
        ${ASCEND_INSTALL_PATH}/pkg_inc
        ${ASCEND_INSTALL_PATH}/pkg_inc/base
)

add_dependencies(op_tiling_rt2_stub op_tiling_rt2_proto)

target_include_directories(op_tiling_rt2_stub PRIVATE
        ${AIR_CODE_DIR}/runtime/v1
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/runtime/v2
        ${AIR_CODE_DIR}
        ${CMAKE_BINARY_DIR}/proto/op_tiling_rt2_proto
        ${METADEF_DIR}/pkg_inc
        ${ASCEND_INSTALL_PATH}/pkg_inc
        ${ASCEND_INSTALL_PATH}/pkg_inc/base
)

# add it for slog compile at 12.16
target_compile_definitions(op_tiling_rt2_stub PRIVATE
        LOG_CPP
)

target_compile_options(op_tiling_rt2_stub PRIVATE
        -D__OPTIMIZER_KT_TEST__
        -DDAVINCI_CLOUD
        -Dgoogle=ascend_private)

target_link_libraries(op_tiling_rt2_stub PRIVATE 
        intf_llt_pub
        slog_headers
        msprof_headers
        mmpa_headers
        runtime_headers
        metadef_headers
        cce_headers
        -Wl,--no-as-needed
        ascend_protobuf
        c_sec
        -Wl,--as-needed
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
)

set(INSTALL_BASE_DIR "")
set(INSTALL_INCLUDE_DIR include)
set(INSTALL_LIBRARY_DIR lib)
set(INSTALL_CONFIG_DIR  lib/cmake)

install(TARGETS op_tiling_rt2_stub OPTIONAL
        EXPORT op_tiling_rt2_stub-targets
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR})

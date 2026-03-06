# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE RUNTIME2_SRCS CONFIGURE_DEPENDS "${AIR_CODE_DIR}/runtime/v2/*.cc"  "${AIR_CODE_DIR}/runtime/v2/*.c")

set(GERT_INCLUDING_PATH
        ${ASCEND_INSTALL_PATH}/include
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/inc
        ${AIR_CODE_DIR}/runtime/v2
        ${AIR_CODE_DIR}/runtime/v2/engine
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/graph_metadef
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}/runtime/v1
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/compiler
        ${CMAKE_BINARY_DIR}
        ${CMAKE_BINARY_DIR}/proto/ge
        ${CMAKE_BINARY_DIR}/proto/ge/proto
        ${CMAKE_BINARY_DIR}/proto/graphengine_protos
        ${METADEF_DIR}/pkg_inc
        ${AIR_CODE_DIR}/inc/graph_metadef/exe_graph/runtime
        ${AIR_CODE_DIR}/base/metadef/pkg_inc/exe_graph/runtime
        ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/att/util/log
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/runtime/include/external/acl
        ${TOP_DIR}/runtime/include/external/acl/error_codes
)

set(GERT_COMPILE_DEF
        PROTOBUF_INLINE_NOT_IN_HEADERS=0
        google=ascend_private
        $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
        $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

set(GERT_DEPEND_PROTO
        "${METADEF_PROTO_DIR}/task.proto"
        "${METADEF_PROTO_DIR}/ge_ir.proto"
        "${METADEF_PROTO_DIR}/flow_model.proto"
        "${METADEF_PROTO_DIR}/op_mapping.proto"
        "${METADEF_PROTO_DIR}/var_manager.proto"
        )
protobuf_generate(gert_proto GERT_DEPEND_PROTO_SRCS GERT_DEPEND_PROTO_HDRS ${GERT_DEPEND_PROTO} TARGET)

add_library(gert_static STATIC ${RUNTIME2_SRCS})
target_include_directories(gert_static PUBLIC
        ${GERT_INCLUDING_PATH}
        ${CMAKE_BINARY_DIR}/proto/gert_proto
        ${METADEF_DIR}/pkg_inc
        ${AIR_CODE_DIR}/inc/graph_metadef
        )
target_compile_definitions(gert_static PRIVATE ${GERT_COMPILE_DEF})
target_link_libraries(gert_static PRIVATE
        intf_pub
        cce_headers
        slog_headers
        msprof_headers
        adump_headers
        mmpa_headers
        metadef_headers
        runtime_headers
        datagw_headers
        ascend_dump
        json
        ascend_protobuf_static
        c_sec
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
)
add_dependencies(gert_static gert_proto)

target_compile_options(gert_static PRIVATE  ${AIR_COMMON_COMPILE_OPTION})

add_library(gert SHARED ${RUNTIME2_SRCS})
target_include_directories(gert PRIVATE
        ${GERT_INCLUDING_PATH}
        ${METADEF_DIR}/pkg_inc
        ${AIR_CODE_DIR}/inc/graph_metadef
        ${TOP_DIR}/metadef/inc
)
target_compile_definitions(gert PRIVATE ${GERT_COMPILE_DEF})
target_link_libraries(gert PRIVATE
        intf_pub
        $<$<TARGET_EXISTS:cce_headers>:cce_headers>
        $<$<TARGET_EXISTS:aicpu_headers>:aicpu_headers>
        msprof_headers
        mmpa_headers
        metadef_headers
        adump_headers
        $<$<TARGET_EXISTS:datagw_headers>:datagw_headers>
        ${AIR_COMMON_LINK_OPTION}
        -Wl,--no-as-needed
        ge_common
        ge_common_base
        runtime
        unified_dlog
        graph
        register
        opp_registry
        error_manager
        ascend_protobuf
        c_sec
        exe_graph
        lowering
        ascend_dump
        ge_executor_shared
        davinci_executor
        hybrid_executor
        -Wl,--as-needed
        json
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
)

target_compile_options(gert PRIVATE ${AIR_COMMON_COMPILE_OPTION}
        $<$<STREQUAL:${CMAKE_CXX_COMPILER_ID},Clang>:-fblocks>
        -fno-strict-aliasing
        )

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE PROTO_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "${METADEF_PROTO_DIR}/task.proto"
        "${METADEF_PROTO_DIR}/ge_ir.proto"
        )

protobuf_generate(ffts_proto PROTO_SRCS PROTO_HDRS ${PROTO_LIST} TARGET)
file(GLOB FFTS_SRC_CPP RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "engine/*.cc"
        "common/*.cc"
        "utils/*.cc"
        "utils/param_calculate/*cc"
        "optimizer/graph_optimizer/*.cc"
        "optimizer/cache_optimizer/*.cc"
        "task_builder/*.cc"
        "task_builder/mode/*.cc"
        "task_builder/mode/auto/*.cc"
        "task_builder/mode/manual/*.cc"
        "task_builder/mode/mixl2/*.cc"
        "task_builder/data_ctx/*.cc"
        "task_builder/thread_ctx/*.cc"
        "task_builder/mixl2_ctx/*.cc")

#compiler for host
add_library(ffts SHARED
        ${FFTS_SRC_CPP}
        ${PROTO_HDRS})

add_dependencies(ffts ffts_proto)
target_include_directories(ffts PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${ENG_DIR}/nn_engine/inc
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/inc/framework
        ${CMAKE_BINARY_DIR}/proto/ffts_proto)

target_compile_options(ffts PRIVATE
        -Werror
        -fno-common
        -fno-strict-aliasing
        -Dgoogle=ascend_private
        -Wno-deprecated-declarations)

target_link_libraries(ffts PRIVATE
        intf_pub
        mmpa_headers
        msprof_headers
        runtime_headers
        cce_headers
        $<$<BOOL:${BUILD_OPEN_PROJECT}>:op_compile_adapter_headers>
        $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:$<BUILD_INTERFACE:tensor_engine_headers>>
        platform
        unified_dlog
        graph
#        graph_base
        register
        c_sec
        ascend_protobuf
        error_manager
        aicore_utils
        -O2
        -Wl,--no-as-needed
        json
        -ldl)

install(TARGETS ffts OPTIONAL
        EXPORT ffts-targets
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
        )

add_custom_target(
        ffts.ini ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ffts.ini
)
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ffts.ini
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/common/ffts_config/ffts.ini ${CMAKE_CURRENT_BINARY_DIR}/
)
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/ffts.ini
        DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
)

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
        "${METADEF_PROTO_DIR}/ge_ir.proto"
        )

protobuf_generate(fusion_proto PROTO_SRCS PROTO_HDRS ${PROTO_LIST} TARGET)

file(GLOB FUSION_SRC_CPP_1 RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "common/*.cc"
        "common/fusion_statistic/*.cc"
        "fusion_config_manager/*.cc"
        "fusion_rule_manager/*.cc"
        "fusion_rule_manager/fusion_rule_data/*.cc"
        "fusion_rule_manager/fusion_rule_parser/*.cc"
        "graph_optimizer/fusion_common/*.cc"
        "graph_optimizer/graph_fusion/*.cc"
        "graph_optimizer/graph_fusion/fusion_pass_manager/*.cc"
        "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/*.cc"
        "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/*.cc"
        "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/checker/*.cc"
        "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/common/*.cc"
        "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/quant_pass/*.cc"
        "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/quant_pass/bias_optimize_quant_rollback/*.cc"
        "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/quant_pass/requant_fusion_pass/*.cc"
        "graph_optimizer/node_optimizer/*.cc"
        "graph_optimizer/stream_graph_optimizer/*.cc"
        "graph_optimizer/stream_graph_optimizer/l2_optimizer/*.cc"
        "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_allocation/*.cc"
        "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_comm/*.cc"
        "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_handler/*.cc"
        "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_parser/*.cc"
        "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_rtl2ctrl/*.cc"
        "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_net_fusion/*.cc"
        "graph_optimizer/ub_fusion/*.cc"
        "graph_optimizer/ub_fusion/fusion_graph_merge/*.cc"
        "graph_optimizer/lx_fusion_optimizer/*.cc"
        "ub_pass_slice_info/*.cc")

add_library(fusion_pass SHARED
        ${FUSION_SRC_CPP_1}
        ${PROTO_HDRS})

add_dependencies(fusion_pass fusion_proto)

target_include_directories(fusion_pass PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${NN_ENG_DIR}/inc/
        ${NN_ENG_DIR}/utils/
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/base
        ${ENG_DIR}/ffts_engine
        ${TOP_DIR}/inc
        ${TOP_DIR}/metadef/inc
        ${CMAKE_BINARY_DIR}/proto/fusion_proto
        ${AIR_CODE_DIR}/compiler/graph/fusion/pass)

target_compile_options(fusion_pass PRIVATE
        -Wfloat-equal
        -Werror
        -fno-common
        -fno-strict-aliasing
        -Dgoogle=ascend_private
        -Wno-deprecated-declarations)

target_compile_definitions(fusion_pass PRIVATE
        $<$<STREQUAL:${BUILD_OPEN_PROJECT},True>:ONLY_COMPILE_BLUE>
        )

target_link_directories(fusion_pass PRIVATE
        ${ASCEND_INSTALL_PATH}/lib64
)

target_link_libraries(fusion_pass PRIVATE
        intf_pub
        msprof_headers
        $<$<BOOL:${BUILD_OPEN_PROJECT}>:op_compile_adapter_headers>
        $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:$<BUILD_INTERFACE:tensor_engine_headers>>
        $<BUILD_INTERFACE:atrace_headers>
        atrace_share
        unified_dlog
        exe_graph
        graph
        ge_common
        register
        c_sec
        runtime
        ascend_protobuf
        error_manager
        aicore_utils
        opskernel
        static_mmpa
        -O2
        -Wl,--no-as-needed
        json
        -ldl)

install(TARGETS fusion_pass OPTIONAL
        EXPORT fusion_pass-targets
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)


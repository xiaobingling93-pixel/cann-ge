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

protobuf_generate(fe_proto PROTO_SRCS PROTO_HDRS ${PROTO_LIST} TARGET)

file(GLOB FE_SRC_CPP_1 RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "adapter/adapter_itf/*.cc"
        "adapter/common/*.cc"
        "adapter/dsa_adapter/*.cc"
        "adapter/ffts_adapter/*.cc"
        "adapter/tbe_adapter/*.cc"
        "adapter/tbe_adapter/kernel_launch/*.cc"
        "adapter/tbe_adapter/tbe_info/*.cc"
        "adapter/tbe_adapter/tbe_info/estimator/*.cc"
        "common/*.cc"
        "format_selector/*.cc"
        "format_selector/builtin/*.cc"
        "format_selector/builtin/broadcast/*.cc"
        "format_selector/builtin/broadcast/format_process/*.cc"
        "format_selector/builtin/process/*.cc"
        "format_selector/builtin/reduce/*.cc"
        "format_selector/builtin/reduce/reduce_format_selector/*.cc"
        "format_selector/common/*.cc"
        "format_selector/manager/*.cc"
        "format_selector/op_customize/*.cc"
        "format_selector/op_kernel/*.cc"
        "fusion_manager/*.cc"
        "graph_optimizer/*.cc"
        "graph_optimizer/json_parser/*.cc"
        "graph_optimizer/dsa_optimizer/*.cc"
        "graph_optimizer/heavy_format_propagation/*.cc"
        "graph_optimizer/op_axis_update/*.cc"
        "graph_optimizer/op_compiler/*.cc"
        "graph_optimizer/op_judge/*.cc"
        "graph_optimizer/op_judge/format_and_dtype/*.cc"
        "graph_optimizer/op_judge/format_and_dtype/strategy/*.cc"
        "graph_optimizer/op_judge/format_and_dtype/strategy/dtype_strategy/*.cc"
        "graph_optimizer/op_judge/format_and_dtype/strategy/format_strategy/*.cc"
        "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/*.cc"
        "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/dtype/*.cc"
        "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/format/*.cc"
        "graph_optimizer/op_judge/format_and_dtype/update_desc/*.cc"
        "graph_optimizer/op_judge/format_and_dtype/update_desc/subgraph/*.cc"
        "graph_optimizer/op_judge/imply_type/*.cc"
        "graph_optimizer/op_setter/*.cc"
        "graph_optimizer/shape_format_transfer/*.cc"
        "graph_optimizer/shape_format_transfer/trans_node_implementation/*.cc"
        "graph_optimizer/shape_format_transfer/trans_node_manager/*.cc"
        "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_insertion/*.cc"
        "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_merging/*.cc"
        "graph_optimizer/spacesize_calculator/*.cc"
        "graph_optimizer/weight_compress_flag/*.cc"
        "graph_optimizer/weight_prefetch/*.cc"
        "graph_optimizer/dynamic_shape_optimizer/fuzzy_compiler/*.cc"
        "graph_optimizer/dynamic_shape_optimizer/model_binary_compiler/*.cc"
        "itf_handler/*.cc"
        "ops_kernel_store/*.cc"
        "ops_kernel_builder/*.cc"
        "ops_kernel_builder/task_builder/*.cc"
        "ops_kernel_builder/task_builder/cmo_task/*.cc"
        "cmo/*.cc"
        "trace_handle_manager/*.cc"
        "trace_handle_manager/trace_msg/*.cc"
        ${ENG_DIR}/ffts_engine/utils/param_calculate/ffts_param_calculator.cc)

add_library(fe SHARED
        ${FE_SRC_CPP_1}
        ${PROTO_HDRS})

add_dependencies(fe fe_proto)

target_include_directories(fe PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${NN_ENG_DIR}/inc/
        ${NN_ENG_DIR}/utils/
        ${NN_ENG_DIR}/fusion/
        ${AIR_CODE_DIR}/
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/base
        ${ENG_DIR}/ffts_engine/inc
        ${ENG_DIR}/ffts_engine
        ${TOP_DIR}/inc
        ${AIR_CODE_DIR}/compiler/opcompiler/inc
        ${TOP_DIR}/metadef/inc
        ${TOP_DIR}/metadef/inc/common/util
        ${CMAKE_BINARY_DIR}/proto/fe_proto
        ${TOP_DIR}/runtime/include/external/acl/error_codes
)

target_compile_options(fe PRIVATE
        -Wfloat-equal
        -Werror
        -fno-common
        -fno-strict-aliasing
        -Dgoogle=ascend_private
        -Wno-deprecated-declarations)

target_compile_definitions(fe PRIVATE
        $<$<STREQUAL:${BUILD_OPEN_PROJECT},True>:ONLY_COMPILE_BLUE>
        )

target_link_directories(fe PRIVATE
        ${ASCEND_INSTALL_PATH}/lib64
)

target_link_libraries(fe PRIVATE
        intf_pub
        msprof_headers
        cce_headers
        $<$<BOOL:${BUILD_OPEN_PROJECT}>:op_compile_adapter_headers>
        $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:$<BUILD_INTERFACE:tensor_engine_headers>>
        $<BUILD_INTERFACE:atrace_headers>
        atrace_share
        platform
        unified_dlog
        exe_graph
        lowering
        graph
        ge_common
#        graph_base
        register
        opp_registry
        c_sec
        runtime
        ascend_protobuf
        compress
        error_manager
        aicore_utils
        fusion_pass
        opskernel
        static_mmpa
        -O2
        -Wl,--no-as-needed
        json
        -ldl)

add_custom_target(
        fe.ini ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/fe.ini
)

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/fe.ini
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/fe_config/fe.ini ${CMAKE_CURRENT_BINARY_DIR}/
)

install(TARGETS fe OPTIONAL
        EXPORT fe-targets
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/fe.ini
        DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
)

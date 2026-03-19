# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(pne/cpu)
add_subdirectory(pne/udf)

set(DFLOW_SESSION_SRC_LIST
    "session/dflow_api.cc"
    "session/dflow_session_impl.cc"
    "session/dflow_session_manager.cc"
)

set(DFLOW_COMPILER_SRC_LIST
    "../compiler/pne/process_node_engine_manager.cc"
    "../compiler/pne/npu/npu_process_node_engine.cc"
    "../compiler/model/flow_model_builder.cc"
    "../compiler/model/flow_model_cache.cc"
    "../compiler/model/flow_model_helper.cc"
    "../compiler/model/dflow_graph_manager.cc"
    "../compiler/data_flow_graph/compile_config_json.cc"
    "../compiler/data_flow_graph/convert_batch_attr_to_udf_pass.cc"
    "../compiler/data_flow_graph/data_flow_graph_prune_pass.cc"
    "../compiler/data_flow_graph/data_flow_graph.cc"
    "../compiler/data_flow_graph/data_flow_graph_auto_deployer.cc"
    "../compiler/data_flow_graph/data_flow_graph_model_relation_builder.cc"
    "../compiler/data_flow_graph/data_flow_graph_utils.cc"
    "../compiler/data_flow_graph/function_compile.cc"
    "../compiler/data_flow_graph/process_point_loader.cc"
    "../compiler/data_flow_graph/inner_pp_loader.cc"
    "../compiler/data_flow_graph/data_flow_attr_utils.cc"
)

set(DFLOW_EXECUTOR_SRC_LIST
    "../executor/flow_msg.cc"
    "../executor/heterogeneous_model_executor.cc"
    "../executor/heterogeneous_model_io_helper.cc"
    "../executor/inner_process_msg_forwarding.cc"
    "../executor/data_flow_exception_handler.cc"
    "../executor/data_flow_data_aligner.cc"
    "../executor/data_flow_info_impl.cc"
    "../executor/flow_model_manager.cc"
    "../executor/data_flow_executor_utils.cc"
)

if (NOT ENABLE_D AND NOT ENABLE_ACL AND NOT ENABLE_MS_TESTCASES)
message("CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
add_library(dflow_runner SHARED
    ${DFLOW_SESSION_SRC_LIST}
    ${DFLOW_COMPILER_SRC_LIST}
    ${DFLOW_EXECUTOR_SRC_LIST}
)

target_compile_definitions(dflow_runner PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
)

target_compile_options(dflow_runner PRIVATE
    -O2
    -fno-common
    -Werror
    -Wextra
    -Wfloat-equal
    -fvisibility=default
)

target_include_directories(dflow_runner SYSTEM PRIVATE
    ${CMAKE_BINARY_DIR}/proto/data_flow_protos
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
)

target_include_directories(dflow_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}/dflow/inc/data_flow
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/inc
    ${AIR_CODE_DIR}/compiler
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${AIR_CODE_DIR}/api/session
    ${AIR_CODE_DIR}
    ${CMAKE_BINARY_DIR}
    ${METADEF_DIR}/pkg_inc
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/include/external/acl/error_codes
)

target_link_options(dflow_runner PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(dflow_runner
    PRIVATE
        intf_pub
        ge_compiler
        ge_runner_v2
        -Wl,--as-needed
        acl_rt
        graph
        graph_base
        ge_common
        ge_common_base
        data_flow_base
        runtime
        error_manager
        unified_dlog
        c_sec
        ascend_protobuf
        profapi_share
        static_mmpa
        -Wl,--no-as-needed
        json
        -lrt
        -ldl
        crypto_static
    PUBLIC
        air_headers
)

endif()

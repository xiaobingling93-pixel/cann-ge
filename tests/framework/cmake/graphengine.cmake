# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# ---- Test coverage ----

if (ENABLE_GE_COV)
    set(COVERAGE_COMPILER_FLAGS "-g --coverage -fprofile-arcs -fPIC -O0 -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${COVERAGE_COMPILER_FLAGS}")
endif()

# ----metadef Proto generate ----
set(PROTO_LIST
    "${METADEF_PROTO_DIR}/om.proto"
    "${METADEF_PROTO_DIR}/ge_ir.proto"
    "${METADEF_PROTO_DIR}/insert_op.proto"
    "${METADEF_PROTO_DIR}/task.proto"
    "${METADEF_PROTO_DIR}/dump_task.proto"
    "${METADEF_PROTO_DIR}/fwk_adapter.proto"
    "${METADEF_PROTO_DIR}/op_mapping.proto"
    "${METADEF_PROTO_DIR}/ge_api.proto"
    "${METADEF_PROTO_DIR}/optimizer_priority.proto"
    "${METADEF_PROTO_DIR}/onnx/ge_onnx.proto"
    "${METADEF_PROTO_DIR}/tensorflow/attr_value.proto"
    "${METADEF_PROTO_DIR}/tensorflow/function.proto"
    "${METADEF_PROTO_DIR}/tensorflow/graph.proto"
    "${METADEF_PROTO_DIR}/tensorflow/graph_library.proto"
    "${METADEF_PROTO_DIR}/tensorflow/node_def.proto"
    "${METADEF_PROTO_DIR}/tensorflow/op_def.proto"
    "${METADEF_PROTO_DIR}/tensorflow/resource_handle.proto"
    "${METADEF_PROTO_DIR}/tensorflow/tensor.proto"
    "${METADEF_PROTO_DIR}/tensorflow/tensor_shape.proto"
    "${METADEF_PROTO_DIR}/tensorflow/types.proto"
    "${METADEF_PROTO_DIR}/tensorflow/versions.proto"
    "${METADEF_PROTO_DIR}/var_manager.proto"
    "${METADEF_PROTO_DIR}/flow_model.proto"
    "${METADEF_PROTO_DIR}/attr_group_base.proto"
    "${METADEF_PROTO_DIR}/ascendc_ir.proto"
    "${METADEF_PROTO_DIR}/ge_ir_mobile.proto"
    "${METADEF_PROTO_DIR}/task_mobile.proto"
)

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST} "--proto_path=${METADEF_PROTO_DIR}" TARGET)

# First of all, released version of metadef (i.e. libgraph_base.so) does NOT include this.
# Not removing it here (from libmetadef_graph.so) will crash the test when linked together
# with libge_compiler.so (which statically link these proto), due to "File already exists
# in database" error of protobuf.
#
# Still, libgraphengine.a needs this as it includes too many stuffs.
set(COMPILER_LINKED_PROTO_SRCS
    "${CMAKE_BINARY_DIR}/proto/ge/proto/optimizer_priority.pb.cc"
)

add_library(graphengine_inc INTERFACE)
target_include_directories(graphengine_inc INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${AIR_CODE_DIR}/base"
    "${AIR_CODE_DIR}/compiler"
    "${AIR_CODE_DIR}/runtime/v1"
    "${AIR_CODE_DIR}/runtime/v2"
    "${AIR_CODE_DIR}/api/session/"
    "${AIR_CODE_DIR}/inc"
    "${AIR_CODE_DIR}"
    "${METADEF_DIR}"
    "${METADEF_DIR}/graph"
    "${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/inc"
    "${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/common"
    "${AIR_CODE_DIR}/inc/external"
    "${AIR_CODE_DIR}/inc/framework/common"
    "${AIR_CODE_DIR}/inc/framework"
    "${AIR_CODE_DIR}/inc/parser"
    "${AIR_CODE_DIR}/inc/parser/external"
    "${AIR_CODE_DIR}/inc/graph_metadef"
    "${METADEF_DIR}/inc/external"
    "${METADEF_DIR}/pkg_inc"
    "${METADEF_DIR}/inc/external/graph"
    "${METADEF_DIR}/inc/graph"
    "${AIR_CODE_DIR}/parser/"
    "${AIR_CODE_DIR}/parser/parser/"
    "${AIR_CODE_DIR}/inc/parser/"
    "${AIR_CODE_DIR}/tests/ge/ut/ge"
    "${AIR_CODE_DIR}/tests/ge/ut/common"
    "${CMAKE_BINARY_DIR}"
    "${CMAKE_BINARY_DIR}/proto/ge"
    "${CMAKE_BINARY_DIR}/proto/ge/proto"
    "${CMAKE_BINARY_DIR}/proto/data_flow_protos"
    "${CMAKE_BINARY_DIR}/proto/data_flow_base_proto"
    "${CMAKE_BINARY_DIR}/proto/data_flow_base_proto/proto"
    "${AIR_CODE_DIR}/dflow"
    "${AIR_CODE_DIR}/dflow/inc"
    "${AIR_CODE_DIR}/dflow/inc/data_flow"
)

add_library(ge_metadef_inc INTERFACE)
target_include_directories(ge_metadef_inc INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_BINARY_DIR}"
        "${CMAKE_BINARY_DIR}/proto/ge"
        "${CMAKE_BINARY_DIR}/proto/ge/proto"
        )

list(APPEND STUB_LIBS
    c_sec
    slog_stub
    runtime_stub
    mmpa_stub
    platform
    profiler_stub
    hccl_stub
    error_manager
    ascend_protobuf
    metadef
    opp_registry
    json
)

#   Targe: engine_conf
add_custom_target(
    engine_conf_json ALL
    DEPENDS ${CMAKE_BINARY_DIR}/engine_conf.json
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/engine_conf.json
    COMMAND cp ${AIR_CODE_DIR}/compiler/engines/manager/engine_manager/engine_conf.json ${CMAKE_BINARY_DIR}/
)

#   Targe: optimizer priority
add_custom_target(
    optimizer_priority_pbtxt ALL
    DEPENDS ${CMAKE_BINARY_DIR}/tests/ge/st/testcase/plugin/opskernel/optimizer_priority.pbtxt
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/tests/ge/st/testcase/plugin/opskernel/optimizer_priority.pbtxt
    COMMAND mkdir -p ${CMAKE_BINARY_DIR}/tests/ge/st/testcase/plugin/opskernel/
    COMMAND cp ${AIR_CODE_DIR}/compiler/engines/manager/opskernel_manager/optimizer_priority.pbtxt ${CMAKE_BINARY_DIR}/tests/ge/st/testcase/plugin/opskernel/
    COMMAND mkdir -p ${CMAKE_BINARY_DIR}/tests/framework/ge_running_env/tests/plugin/opskernel/
    COMMAND cp ${AIR_CODE_DIR}/compiler/engines/manager/opskernel_manager/optimizer_priority.pbtxt ${CMAKE_BINARY_DIR}/tests/framework/ge_running_env/tests/plugin/opskernel/
)

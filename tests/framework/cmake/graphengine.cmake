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

# ---- File glob by group ----
file(GLOB_RECURSE METADEF_SRCS CONFIGURE_DEPENDS
    "${GE_METADEF_DIR}/graph/*.cc"
    "${GE_METADEF_DIR}/graph/*.cpp"
    "${GE_METADEF_DIR}/exe_graph/*.cc"
    "${GE_METADEF_DIR}/register/*.cc"
    "${GE_METADEF_DIR}/register/op_tiling/*.cc"
    "${GE_METADEF_DIR}/register/*.cpp"
    "${GE_METADEF_DIR}/ops/*.cc"
    "${GE_METADEF_DIR}/third_party/transformer/src/*.cc"
    "${GE_METADEF_DIR}/ops/op_imp.cpp"
    "${GE_METADEF_DIR}/base/*.cc"
)

file(GLOB_RECURSE ASCIR_EXCLUDE_SOURCE CONFIGURE_DEPENDS
        "${GE_METADEF_DIR}/graph/ascendc_ir/utils/mem_utils.cc"
        )

list(REMOVE_ITEM METADEF_SRCS ${ASCIR_EXCLUDE_SOURCE})

file(GLOB_RECURSE METADEF_REGISTER_SRCS CONFIGURE_DEPENDS
    "${GE_METADEF_DIR}/register/*.cc"
    "${GE_METADEF_DIR}/register/op_tiling/*.cc"
    "${GE_METADEF_DIR}/register/*.cpp"
)

file(GLOB_RECURSE PARSER_SRCS CONFIGURE_DEPENDS
    "${PARSER_DIR}/parser/common/*.cc"
    "${PARSER_DIR}/parser/tensorflow/*.cc"
    "${PARSER_DIR}/parser/onnx/*.cc"
)

file(GLOB_RECURSE LOCAL_ENGINE_SRC CONFIGURE_DEPENDS
    "${AIR_CODE_DIR}/compiler/engines/local_engine/*.cc"
)

list(REMOVE_ITEM LOCAL_ENGINE_SRC
    "${AIR_CODE_DIR}/base/host_cpu_engine/host_cpu_engine.cc"
    "${AIR_CODE_DIR}/compiler/engines/local_engine/ops_kernel_store/ge_local_ops_kernel_builder.cc"
)

file(GLOB_RECURSE NN_ENGINE_SRC CONFIGURE_DEPENDS
    "${AIR_CODE_DIR}/compiler/engines/manager/engine/*.cc"
)

file(GLOB_RECURSE OFFLINE_SRC CONFIGURE_DEPENDS
    "${AIR_CODE_DIR}/api/atc/*.cc"
)

file(GLOB_RECURSE EAGER_STYLE_GRAPH_BUILDER_BASE_SRCS CONFIGURE_DEPENDS
    "${AIR_CODE_DIR}/compiler/graph/eager_style_graph_builder/es_generator/*.cc"
    "${AIR_CODE_DIR}/compiler/graph/eager_style_graph_builder/es_base_struct/*.cc"
)

file(GLOB_RECURSE GE_SRCS CONFIGURE_DEPENDS
    "${AIR_CODE_DIR}/base/common/*.cc"
    "${AIR_CODE_DIR}/base/slice/*.cc"
    "${AIR_CODE_DIR}/base/formats/*.cc"
    "${AIR_CODE_DIR}/base/graph/*.cc"
    "${AIR_CODE_DIR}/base/exec_runtime/*.cc"
    "${AIR_CODE_DIR}/base/host_cpu_engine/*.cc"
    "${AIR_CODE_DIR}/compiler/analyzer/*.cc"
    "${AIR_CODE_DIR}/compiler/engines/manager/engine_manager/*.cc"
    "${AIR_CODE_DIR}/compiler/engines/local_engine/*.cc"
    "${AIR_CODE_DIR}/compiler/opt_info/*.cc"
    "${AIR_CODE_DIR}/compiler/api/generator/*.cc"
    "${AIR_CODE_DIR}/compiler/graph/*.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/*.cc"
    "${AIR_CODE_DIR}/compiler/inc/*.cc"
    "${AIR_CODE_DIR}/compiler/api/gelib/*.cc"
    "${AIR_CODE_DIR}/compiler/api/aclgrph/*.cc"
    "${AIR_CODE_DIR}/api/atc/*.cc"
    "${AIR_CODE_DIR}/compiler/engines/manager/opskernel_manager/*.cc"
    "${AIR_CODE_DIR}/compiler/engines/custom_engine/*.cc"
    "${AIR_CODE_DIR}/compiler/plugin/*.cc"
    "${AIR_CODE_DIR}/dflow/compiler/model/*.cc"
    "${AIR_CODE_DIR}/dflow/compiler/pne/npu/*.cc"
    "${AIR_CODE_DIR}/dflow/compiler/pne/cpu/*.cc"
    "${AIR_CODE_DIR}/dflow/compiler/data_flow_graph/*.cc"
    "${AIR_CODE_DIR}/dflow/compiler/pne/*.cc"
    "${AIR_CODE_DIR}/dflow/compiler/pne/udf/*.cc"
    "${AIR_CODE_DIR}/dflow/compiler/session/*.cc"
    "${AIR_CODE_DIR}/dflow/executor/*.cc"
    "${AIR_CODE_DIR}/runtime/v1/*.cc"
    "${AIR_CODE_DIR}/api/session/*.cc"
)

file(GLOB_RECURSE GE_SUB_ENGINE_SRCS CONFIGURE_DEPENDS
    "${AIR_CODE_DIR}/base/host_cpu_engine/host_cpu_engine.cc"
)

list(REMOVE_ITEM GE_SRCS
    ${NN_ENGINE_SRC}
    ${EAGER_STYLE_GRAPH_BUILDER_BASE_SRCS}
    ${AIR_CODE_DIR}/api/atc/main.cc
)

list(APPEND GE_SRCS ${GE_SUB_ENGINE_SRCS})

# First of all, released version of metadef (i.e. libgraph_base.so) does NOT include this.
# Not removing it here (from libmetadef_graph.so) will crash the test when linked together
# with libge_compiler.so (which statically link these proto), due to "File already exists
# in database" error of protobuf.
#
# Still, libgraphengine.a needs this as it includes too many stuffs.
set(COMPILER_LINKED_PROTO_SRCS
    "${CMAKE_BINARY_DIR}/proto/ge/proto/optimizer_priority.pb.cc"
)
list(REMOVE_ITEM PROTO_SRCS ${COMPILER_LINKED_PROTO_SRCS})
list(APPEND GE_SRCS ${COMPILER_LINKED_PROTO_SRCS})
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

# ---- Target : metadef graph ----
add_library(metadef_graph SHARED
        ${METADEF_SRCS}
        ${PROTO_SRCS}
        )
add_dependencies(metadef_graph ge)

target_compile_definitions(metadef_graph PRIVATE
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
        FMK_SUPPORT_DUMP
        FUNC_VISIBILITY
        )

target_compile_definitions(metadef_graph PUBLIC
        google=ascend_private
        )

target_compile_options(metadef_graph PRIVATE ${AIR_COMMON_COMPILE_OPTION})

target_include_directories(metadef_graph PRIVATE
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/graph_metadef/external
    ${AIR_CODE_DIR}/inc/framework
    ${METADEF_DIR}/inc/external/exe_graph
    )

target_link_libraries(metadef_graph PUBLIC
        intf_pub
        metadef_headers
        -Wl,-z,muldefs
        runtime_headers
        cce_headers
        ${AIR_COMMON_LINK_OPTION}
        ${STUB_LIBS}
        ge_metadef_inc
        symengine
        Boost::boost
        ascendalog
        slog_stub
        error_manager
        -Wl,--no-as-needed unified_dlog -Wl,--as-needed
        )

add_library(local_engine SHARED
    ${LOCAL_ENGINE_SRC}
)
add_dependencies(local_engine ge)

target_include_directories(local_engine PUBLIC
    "${AIR_CODE_DIR}/compiler/engines/local_engine"
)

target_compile_definitions(local_engine PRIVATE
    google=ascend_private
)

target_compile_options(local_engine PRIVATE
    ${AIR_COV_COMPILE_OPTION}
    -Werror=format
)

target_link_libraries(local_engine PUBLIC
    intf_pub
    metadef_headers
    runtime_headers
    cce_headers
    ${STUB_LIBS}
    graphengine_inc
    ${AIR_COMMON_LINK_OPTION}
)

set_target_properties(local_engine PROPERTIES
    OUTPUT_NAME ge_local_engine
)

# ---- Target : engine plugin----
#
add_library(nnengine SHARED
    ${NN_ENGINE_SRC}
)

target_include_directories(nnengine PUBLIC
    "${AIR_CODE_DIR}/compiler/engines/manager/engine"
)

target_compile_definitions(nnengine PRIVATE
    google=ascend_private
)

target_compile_options(nnengine PRIVATE
    ${AIR_COV_COMPILE_OPTION}
    -Werror=format
)

target_link_libraries(nnengine PUBLIC
    intf_pub
    metadef_headers
    ${STUB_LIBS}
    graphengine_inc
    ${AIR_COMMON_LINK_OPTION}
)

target_link_options(nnengine PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
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


# ---- Target : Graph engine ----
add_library(graphengine STATIC
    ${PARSER_SRCS}
    ${GE_SRCS}
)
add_dependencies(graphengine ge)

target_compile_definitions(graphengine PRIVATE
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    google=ascend_private
    FMK_SUPPORT_DUMP
    FWK_SUPPORT_TRAINING_TRACE
)

target_compile_options(graphengine PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

target_link_libraries(graphengine PUBLIC
    intf_pub
    metadef_headers
    adump_headers
    datagw_headers
    graphengine_inc
    json
    gert
    ${STUB_LIBS}
    metadef_graph
    crypto_static
    data_flow_base
    aihacb_autofusion_stub -lstdc++fs
)

add_dependencies(graphengine local_engine nnengine engine_conf_json optimizer_priority_pbtxt)

stub_module(graph metadef_graph)
stub_module(graph_base metadef_graph)
stub_module(register metadef_graph)
stub_module(exe_graph metadef_graph)
stub_module(lowering metadef_graph)
stub_module(flow_graph metadef_graph)
stub_module(aihac_ir metadef_graph)
stub_module(aihac_symbolizer metadef_graph)
stub_module(aihac_ir_register metadef_graph)
stub_module(metadef_graph_protos_obj metadef_graph)

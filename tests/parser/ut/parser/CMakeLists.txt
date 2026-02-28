# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

################################################################################
set(PARSER_PROTO_LIST
    "${METADEF_PROTO_DIR}/om.proto"
    "${METADEF_PROTO_DIR}/ge_ir.proto"
    "${METADEF_PROTO_DIR}/task.proto"
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
    "${METADEF_PROTO_DIR}/caffe/caffe.proto"
    "${METADEF_PROTO_DIR}/onnx/ge_onnx.proto"
    "${METADEF_PROTO_DIR}/ge_ir_mobile.proto"
    "${METADEF_PROTO_DIR}/task_mobile.proto"
)

protobuf_generate(ge PARSER_PROTO_SRCS PARSER_PROTO_HDRS ${PARSER_PROTO_LIST})

############ libut_parser_proto.a ############
add_library(ut_parser_proto STATIC
    ${PARSER_PROTO_HDRS} ${PARSER_PROTO_SRCS}
)

target_compile_definitions(ut_parser_proto PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
)

target_compile_options(ut_parser_proto PRIVATE
    -O2 -g -fno-common
)

target_link_libraries(ut_parser_proto PRIVATE
    intf_pub
    ascend_protobuf
)


################################################################################
set(DUPLICATE_PROTO_LIST
    "${METADEF_PROTO_DIR}/onnx/ge_onnx.proto"
)

protobuf_generate(ge DUP_PROTO_SRCS DUP_PROTO_HDRS ${DUPLICATE_PROTO_LIST})

################################################################################
file(GLOB_RECURSE METADEF_BASE_FILES
    ${GE_METADEF_DIR}/register/op_lib_register.cc
    ${GE_METADEF_DIR}/graph/*.cc
)

file(GLOB_RECURSE TO_EXCLUDE
    ${GE_METADEF_DIR}/graph/ascendc_ir/*.*
    ${GE_METADEF_DIR}/graph/expression/*.*
    ${GE_METADEF_DIR}/graph/cache_policy/*.*
    ${GE_METADEF_DIR}/graph/debug/*.*
    ${GE_METADEF_DIR}/graph/fast_graph/*.*
    ${GE_METADEF_DIR}/graph/hcom/*.*
    ${GE_METADEF_DIR}/graph/stub/*.*
    ${METADEF_DIR}/base/context_builder/*.*
)

list(REMOVE_ITEM METADEF_BASE_FILES ${TO_EXCLUDE})

set(MATEDEF_SRC_FILES
    ${METADEF_BASE_FILES}
    "${GE_METADEF_DIR}/ops/op_imp.cpp"
    "${GE_METADEF_DIR}/third_party/transformer/src/axis_util.cc"
    "${GE_METADEF_DIR}/third_party/transformer/src/expand_dimension.cc"
    "${GE_METADEF_DIR}/third_party/transformer/src/transfer_shape_according_to_format.cc"
    "${GE_METADEF_DIR}/third_party/transformer/src/transfer_shape_utils.cc"
)

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${METADEF_DIR}/inc)
include_directories(${METADEF_DIR}/inc/graph)
include_directories(${GE_METADEF_INC_DIR})
include_directories(${GE_METADEF_INC_DIR}/graph)
include_directories(${GE_METADEF_INC_DIR}/external)
include_directories(${GE_METADEF_INC_DIR}/external/graph)
include_directories(${GE_METADEF_INC_DIR}/graph)
include_directories(${GE_METADEF_INC_DIR}/register)
include_directories(${GE_METADEF_DIR}/third_party/transformer/inc)
include_directories(${GE_METADEF_DIR}/third_party/transformer/src)
include_directories(${METADEF_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CMAKE_BINARY_DIR}/proto/ge/proto)

############ libut_parser_graph.a ############
add_library(ut_parser_graph STATIC
    ${MATEDEF_SRC_FILES} ${PARSER_PROTO_HDRS} ${DUP_PROTO_HDRS}
)

target_compile_definitions(ut_parser_graph PRIVATE
    google=ascend_private
)

target_compile_options(ut_parser_graph PRIVATE
    -O2 -g -fno-common
)

target_link_libraries(ut_parser_graph PRIVATE
    intf_pub
    metadef_headers
    -Wl,-z,muldefs
    air_headers
    slog_headers
    mmpa_headers
    json
    c_sec ascend_protobuf platform_stub_parser
)


################################################################################
set(REGISTER_SRC_FILES
    "${GE_METADEF_DIR}/register/auto_mapping_util.cpp"
    "${GE_METADEF_DIR}/register/graph_optimizer/buffer_fusion/buffer_fusion_pass_base.cc"
    "${GE_METADEF_DIR}/register/graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.cc"
    "${GE_METADEF_DIR}/register/graph_optimizer/buffer_fusion/buffer_fusion_pattern.cc"
    "${GE_METADEF_DIR}/register/graph_optimizer/fusion_statistic/fusion_statistic_recorder.cc"
    "${GE_METADEF_DIR}/register/graph_optimizer/graph_fusion/fusion_pass_registry.cc"
    "${GE_METADEF_DIR}/register/graph_optimizer/graph_fusion/fusion_pattern.cc"
    "${GE_METADEF_DIR}/register/graph_optimizer/graph_fusion/graph_fusion_pass_base.cc"
    "${GE_METADEF_DIR}/register/graph_optimizer/graph_fusion/pattern_fusion_base_pass.cc"
    "${GE_METADEF_DIR}/register/graph_optimizer/graph_fusion/pattern_fusion_base_pass_impl.cc"
    "${GE_METADEF_DIR}/register/host_cpu_context.cc"
    "${GE_METADEF_DIR}/register/infer_data_slice_registry.cc"
    "${GE_METADEF_DIR}/register/ops_kernel_builder_registry.cc"
    "${GE_METADEF_DIR}/register/op_kernel_registry.cpp"
    "${GE_METADEF_DIR}/register/op_tiling/op_tiling.cc"
    "${GE_METADEF_DIR}/register/op_tiling/op_tiling_registry.cc"
    "${GE_METADEF_DIR}/register/register.cpp"
    "${GE_METADEF_DIR}/register/register_custom_pass.cpp"
    "${GE_METADEF_DIR}/register/scope/scope_graph.cc"
    "${GE_METADEF_DIR}/register/scope/scope_pass.cc"
    "${GE_METADEF_DIR}/register/scope/scope_pass_registry.cc"
    "${GE_METADEF_DIR}/register/scope/scope_pattern.cc"
    "${GE_METADEF_DIR}/register/scope/scope_util.cc"
    "${GE_METADEF_DIR}/register/tensor_assign.cpp"
    "${GE_METADEF_DIR}/register/prototype_pass_registry.cc"
    "${GE_METADEF_DIR}/base/utils/type_utils_inner.cc"
)

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${GE_METADEF_DIR})
include_directories(${GE_METADEF_DIR}/graph)
include_directories(${GE_METADEF_DIR}/register)
include_directories(${GE_METADEF_INC_DIR})
include_directories(${GE_METADEF_INC_DIR}/external)
include_directories(${GE_METADEF_INC_DIR}/register)
include_directories(${GE_METADEF_INC_DIR}/graph/utils)
include_directories(${AIR_CODE_DIR}/inc/graph_metadef)

############ libut_parser_register.a ############
add_library(ut_parser_register STATIC
    ${REGISTER_SRC_FILES} ${PARSER_PROTO_HDRS}
)

target_compile_definitions(ut_parser_register PRIVATE
    google=ascend_private
)

target_compile_options(ut_parser_register PRIVATE
    -O2 -g -fno-common
)

target_link_libraries(ut_parser_register PRIVATE
    intf_pub
    air_headers
    slog_headers
    runtime_headers
    cce_headers
    msprof_headers
    metadef_headers
    c_sec ascend_protobuf json
)


################################################################################
set(PARSER_SRC_FILES
    "${PARSER_DIR}/parser/caffe/caffe_custom_parser_adapter.cc"
    "${PARSER_DIR}/parser/caffe/caffe_data_parser.cc"
    "${PARSER_DIR}/parser/caffe/caffe_op_parser.cc"
    "${PARSER_DIR}/parser/caffe/caffe_parser.cc"
    "${PARSER_DIR}/parser/caffe/caffe_reshape_parser.cc"
    "${PARSER_DIR}/parser/common/acl_graph_parser_util.cc"
    "${PARSER_DIR}/parser/common/convert/pb2json.cc"
    "${PARSER_DIR}/parser/common/convert/message2operator.cc"
    "${PARSER_DIR}/parser/common/data_op_parser.cc"
    "${PARSER_DIR}/parser/common/model_saver.cc"
    "${PARSER_DIR}/parser/common/op_def/arg_op_operator.cc"
    "${PARSER_DIR}/parser/common/op_def/constant_operator.cc"
    "${PARSER_DIR}/parser/common/op_def/fill_operator.cc"
    "${PARSER_DIR}/parser/common/op_def/framework_op_operator.cc"
    "${PARSER_DIR}/parser/common/op_def/ir_pb_converter.cc"
    "${PARSER_DIR}/parser/common/op_def/no_op_operator.cc"
    "${PARSER_DIR}/parser/common/op_def/operator.cc"
    "${PARSER_DIR}/parser/common/op_def/ref_switch_operator.cc"
    "${PARSER_DIR}/parser/common/op_def/shape_n_operator.cc"
    "${PARSER_DIR}/parser/common/op_def/variable_operator.cc"
    "${PARSER_DIR}/parser/common/op_def/var_is_initialized_op_operator.cc"
    "${PARSER_DIR}/parser/common/op_map.cc"
    "${PARSER_DIR}/parser/common/op_parser_factory.cc"
    "${PARSER_DIR}/parser/common/parser_api.cc"
    "${PARSER_DIR}/parser/common/parser_factory.cc"
    "${PARSER_DIR}/parser/common/parser_fp16_t.cc"
    "${PARSER_DIR}/parser/common/parser_inner_ctx.cc"
    "${PARSER_DIR}/parser/common/parser_types.cc"
    "${PARSER_DIR}/parser/common/parser_utils.cc"
    "${PARSER_DIR}/parser/common/pass_manager.cc"
    "${PARSER_DIR}/parser/common/pre_checker.cc"
    "${PARSER_DIR}/parser/common/proto_file_parser.cc"
    "${PARSER_DIR}/parser/common/prototype_pass_manager.cc"
    "${PARSER_DIR}/parser/common/op_registration_tbe.cc"
    "${PARSER_DIR}/parser/common/tbe_plugin_loader.cc"
    "${PARSER_DIR}/parser/common/thread_pool.cc"
    "${PARSER_DIR}/parser/common/auto_mapping_subgraph_io_index_func.cc"
    "${PARSER_DIR}/parser/onnx/onnx_constant_parser.cc"
    "${PARSER_DIR}/parser/onnx/onnx_file_constant_parser.cc"
    "${PARSER_DIR}/parser/onnx/onnx_custom_parser_adapter.cc"
    "${PARSER_DIR}/parser/onnx/onnx_data_parser.cc"
    "${PARSER_DIR}/parser/onnx/onnx_parser.cc"
    "${PARSER_DIR}/parser/onnx/onnx_util.cc"
    "${PARSER_DIR}/parser/onnx/subgraph_adapter/if_subgraph_adapter.cc"
    "${PARSER_DIR}/parser/onnx/subgraph_adapter/subgraph_adapter_factory.cc"
    "${PARSER_DIR}/parser/tensorflow/graph_to_function_def.cc"
    "${PARSER_DIR}/parser/tensorflow/parser_graph_optimizer.cc"
    "${PARSER_DIR}/parser/tensorflow/iterator_fusion_pass.cc"
    "${PARSER_DIR}/parser/tensorflow/scope/scope_pass_manager.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_arg_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_auto_mapping_parser_adapter.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_constant_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_custom_parser_adapter.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_data_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_enter_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_fill_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_frameworkop_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_fusion_custom_parser_adapter.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_fusion_op_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_identity_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_merge_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_no_op_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_custom_op_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_ref_switch_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_reshape_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_shape_n_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_squeeze_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_util.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_variable_v2_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_var_is_initialized_op_parser.cc"
)

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${PARSER_DIR})
include_directories(${AIR_CODE_DIR}/parser)
include_directories(${AIR_CODE_DIR}/parser/parser)
include_directories(${AIR_CODE_DIR}/parser/onnx)
include_directories(${AIR_CODE_DIR}/tests/parser)
include_directories(${AIR_CODE_DIR}/inc/parser)
include_directories(${AIR_CODE_DIR}/inc/parser/external)
include_directories(${AIR_CODE_DIR}/inc/graph_metadef)
include_directories(${METADEF_DIR}/inc)
include_directories(${METADEF_DIR}/inc/external)
include_directories(${METADEF_DIR}/inc/register)
include_directories(${METADEF_DIR}/pkg_inc)

set(PARSER_UT_FILES
    "parser_ut_utils.cc"
    "testcase/common/acl_graph_parser_unittest.cc"
    "testcase/onnx_parser_testcase/onnx_parser_unittest.cc"
    "testcase/caffe_parser_testcase/caffe_parser_unittest.cc"
    "testcase/onnx_parser_testcase/message2operator_unittest.cc"
    "testcase/tensorflow_parser_testcase/tensorflow_parser_unittest.cc"
    "testcase/tensorflow_parser_testcase/tensorflow_auto_mapping_parser_adapter_unittest.cc"
    "testcase/graph_optimizer_testcase/graph_optimizer_unittest.cc"
)

############ libut_parser_common.a ############
add_library(ut_parser_common STATIC
    ${PARSER_SRC_FILES} ${PARSER_PROTO_HDRS}
)

target_compile_definitions(ut_parser_common PRIVATE
    google=ascend_private
)

target_compile_options(ut_parser_common PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

target_link_libraries(ut_parser_common PRIVATE
    intf_pub
    air_headers
    slog_headers
    ut_parser_proto ut_parser_graph c_sec
    ascend_protobuf
    -lstdc++fs
    json
    mmpa_headers
    opp_registry
)


################################################################################
add_executable(ut_parser
    ${PARSER_UT_FILES} ${PARSER_PROTO_SRCS}
)

target_compile_options(ut_parser PRIVATE
    -g
    -fno-access-control
)

target_compile_definitions(ut_parser PRIVATE
    google=ascend_private
)

target_link_libraries(ut_parser
    intf_pub
    metadef_headers
    air_headers
    slog_headers
    cce_headers
    runtime_headers
    msprof_headers
    ut_parser_proto
    metadef -lstdc++fs
    -Wl,--whole-archive ut_parser_common -Wl,--no-whole-archive
    attr_util_stub ut_parser_graph ut_parser_register error_manager parser_mmpa_stub platform_stub_parser
    GTest::gtest GTest::gtest_main parser_slog_stub ascend_protobuf c_sec json -lrt -ldl -lgcov
    -Wl,--no-as-needed unified_dlog -Wl,--as-needed
)

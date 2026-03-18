# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

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
    #"${METADEF_PROTO_DIR}/proto_inner/ge_onnx.proto"
    "${METADEF_PROTO_DIR}/ge_ir_mobile.proto"
    "${METADEF_PROTO_DIR}/task_mobile.proto"
)

protobuf_generate(ge PARSER_PROTO_SRCS PARSER_PROTO_HDRS ${PARSER_PROTO_LIST})


file(GLOB_RECURSE SRCS RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "src/attr_util_stub.cc"
        "src/opp_so_manager_stub.cc"
)

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CMAKE_BINARY_DIR}/proto/ge/proto)
include_directories(${METADEF_DIR})

add_library(attr_util_stub STATIC
    ${SRCS} ${PARSER_PROTO_HDRS}
)

target_compile_definitions(attr_util_stub PRIVATE
    google=ascend_private
)

target_compile_options(attr_util_stub PRIVATE
    -O2 -g -fno-common
)

target_link_libraries(attr_util_stub PRIVATE
    intf_llt_pub
    metadef_headers
    air_headers
    slog_headers
    ascend_protobuf
    c_sec
)

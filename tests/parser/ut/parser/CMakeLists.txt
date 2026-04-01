# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ======================================================================================================================

################################################################################
set(PARSER_SRC_FILES
    "${PARSER_DIR}/parser/caffe/caffe_parser.cc"
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

################################################################################
add_executable(ut_parser
    ${PARSER_UT_FILES} ${PARSER_SRC_FILES}
)

target_compile_options(ut_parser PRIVATE
    -g
    -fno-access-control
)

target_compile_definitions(ut_parser PRIVATE
    google=ascend_private
    CMAKE_BINARY_DIR=\"${CMAKE_BINARY_DIR}\"
)

target_link_libraries(ut_parser
    intf_llt_pub
    metadef_headers
    air_headers
    slog_headers
    cce_headers
    runtime_headers
    msprof_headers
    metadef
    graph
    graph_base
    fmk_parser
    fmk_onnx_parser
    -Wl,--whole-archive attr_util_stub -Wl,--no-whole-archive
    error_manager parser_mmpa_stub platform_stub_parser
    GTest::gtest GTest::gtest_main ascend_protobuf_static parser_slog_stub  c_sec json -lrt -ldl
    -Wl,--no-as-needed unified_dlog -Wl,--as-needed
    # Workaround for undefined references
    ge_common_base
    ascendcl_stub
)

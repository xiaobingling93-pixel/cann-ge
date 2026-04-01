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
# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
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


set(PARSER_ST_FILES
    "parser_st_utils.cc"
    "testcase/test_main.cc"
    "testcase/test_onnx_parser.cc"
    "testcase/test_caffe_parser.cc"
    "testcase/test_tensorflow_parser.cc"
    "${PARSER_DIR}/parser/caffe/caffe_parser.cc"
)

################################################################################
add_executable(st_parser
    ${PARSER_ST_FILES}
)

target_compile_options(st_parser PRIVATE
    -g
    -fno-access-control
)

target_compile_definitions(st_parser PRIVATE
    google=ascend_private
    CMAKE_BINARY_DIR=\"${CMAKE_BINARY_DIR}\"
)

target_link_libraries(st_parser PRIVATE
    intf_llt_pub
    metadef_headers
    air_headers
    slog_headers
    cce_headers
    runtime_headers
    msprof_headers
    metadef
    fmk_parser
    fmk_onnx_parser
    graph
    graph_base
    parser_mmpa_stub
    json
    -Wl,--whole-archive -Wl,--no-whole-archive register error_manager parser_mmpa_stub attr_util_stub
    error_manager parser_mmpa_stub attr_util_stub
    GTest::gtest GTest::gtest_main parser_slog_stub ascend_protobuf c_sec  -lrt -ldl
    -Wl,--no-as-needed unified_dlog -Wl,--as-needed
    # Workaround for undefined references
    ge_common_base
    ascendcl_stub
)

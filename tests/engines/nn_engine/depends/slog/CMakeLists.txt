# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

SET(FE_ROOT_PATH ${AIR_CODE_DIR}/compiler/engines/nn_engine)
SET(FWK_TEST_PATH ${AIR_CODE_DIR}/tests)
SET(AIR_DIR ${AIR_CODE_DIR})

#-------------------------ge_graph_dsl------------------------------
add_subdirectory(${FWK_TEST_PATH}/framework/easy_graph easy_graph)
add_subdirectory(${FWK_TEST_PATH}/framework/ge_graph_dsl/include ge_graph_dsl_inc)
file(GLOB_RECURSE SOURCES
        ${AIR_DIR}/base/common/types.cc
        ${FWK_TEST_PATH}/framework/ge_graph_dsl/src/assert/*.cc
        ${FWK_TEST_PATH}/framework/ge_graph_dsl/src/op_desc/*.cc
        ${FWK_TEST_PATH}/framework/ge_graph_dsl/src/vistor/*.cc)

add_library(ge_graph_dsl STATIC ${SOURCES})

target_link_libraries(ge_graph_dsl PUBLIC slog_headers ge_graph_dsl_inc easy_graph graph graph_base intf_llt_pub)

target_include_directories(ge_graph_dsl PRIVATE
        ${ASCEND_INSTALL_PATH}/include
        ${FWK_TEST_PATH}/framework/ge_graph_dsl/include
        ${FWK_TEST_PATH}/framework/easy_graph/include
        ${FWK_TEST_PATH}/tests/framework/easy_graph/include
        ${AIR_DIR}/inc/external
        ${AIR_DIR}/inc
        ${AIR_DIR}/inc/framework
        ${AIR_DIR}/inc/framework/common
        )
#-------------------------------------------------------------------
file(GLOB_RECURSE SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "slog_stub.cc"
        )

add_library(slog_stub SHARED
        ${SRC_FILES}
        )

target_link_libraries(slog_stub
        PRIVATE
                intf_llt_pub
        PUBLIC
                unified_dlog_headers
        )
stub_module(unified_dlog slog_stub)

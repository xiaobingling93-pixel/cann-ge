# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cc" "*.CC" "*.cpp" "*.CPP" "*.c++")

add_library(ge_runtime_stub STATIC ${SOURCES})

target_compile_options(ge_runtime_stub PRIVATE ${AIR_COMMON_COMPILE_OPTION})

target_compile_definitions(ge_runtime_stub PRIVATE
        CMAKE_BINARY_DIR=\"${CMAKE_BINARY_DIR}\"
        AIR_CODE_DIR=\"${AIR_CODE_DIR}\"
        ASCEND_INSTALL_PATH=\"${ASCEND_INSTALL_PATH}\")

target_include_directories(ge_runtime_stub PUBLIC
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/runtime/v1
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/runtime/v2
        ${AIR_CODE_DIR}/tests
        ${OP_PROTO_DIR}
        ${AIR_CODE_DIR}/tests/depends/runtime/src
        ${AIR_CODE_DIR}/tests/framework/ge_runtime_stub/include
)

target_link_libraries(ge_runtime_stub PRIVATE
        intf_pub
        msprof_headers
        runtime_headers
        opp_registry
        json
        datagw_headers
        ${AIR_COMMON_LINK_OPTION}
        ge_runtime_stub_inc
        ge_graph_dsl
        ge_common
        ge_common_base
        es_ge_test
        )

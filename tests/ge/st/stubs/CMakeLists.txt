# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE UTILS_SRCS CONFIGURE_DEPENDS *.cc)

add_library(st_stubs STATIC ${UTILS_SRCS})

target_include_directories(st_stubs PRIVATE
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/graph_metadef
        ${METADEF_DIR}/inc
        ${METADEF_DIR}/pkg_inc
)

target_include_directories(st_stubs PUBLIC
        ./)

target_compile_definitions(st_stubs PRIVATE
        google=ascend_private
)

target_compile_options(st_stubs PRIVATE
    ${AIR_COMMON_COMPILE_OPTION}
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
)

target_link_libraries(st_stubs PUBLIC intf_pub ge_graph_dsl ge_running_env GTest::gtest GTest::gmock GTest::gmock_main datagw_headers)

add_library(memset_stubs STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/memset.c)

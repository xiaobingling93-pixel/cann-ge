# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

message(STATUS "Setting up graph construction test in: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Binary directory: ${CMAKE_CURRENT_BINARY_DIR}")

# 1. 包含 add_es_library 函数
include(${AIR_CODE_DIR}/cmake/generate_es_package.cmake)

# 2. 创建 stub_geir_ops 原型库（测试用）
add_library(stub_geir_ops SHARED
        ${AIR_CODE_DIR}/tests/ge/ut/ge/graph/eager_style_graph_builder/es_generator/stub_geir_ops.cc
        )

target_compile_options(stub_geir_ops PRIVATE ${AIR_COMMON_COMPILE_OPTION})
target_link_libraries(stub_geir_ops
        PRIVATE
        intf_pub
        graph
        graph_base
        error_manager
        c_sec
        )

# 3. 使用 add_es_library_and_whl 函数生成 ES API 包
add_es_library_and_whl(
        ES_LINKABLE_AND_ALL_TARGET es_ut_test
        OPP_PROTO_TARGET stub_geir_ops
        OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/output
)

# GCOV 模式下，动态库也需要链接 gcov 库，避免 hidden symbol 错误
if (CMAKE_BUILD_TYPE STREQUAL "GCOV")
    target_link_options(es_ut_test_so PRIVATE --coverage)
endif ()

# 4. 创建测试可执行文件（链接到 add_es_library_and_whl 生成的库）
add_executable(ut_graph_construction
        graph_construction_unittest.cc
        )

# build_es_ut_test 已自动依赖 gen_esb_impl（在 generate_es_package.cmake 中）
# 链接 es_ut_test 会自动触发 build_es_ut_test（通过 INTERFACE 库的 add_dependencies）
# 此处无需显式添加依赖

target_compile_options(ut_graph_construction PRIVATE ${AIR_COMMON_COMPILE_OPTION})
target_link_libraries(ut_graph_construction
        PRIVATE
        GTest::gtest GTest::gtest_main
        es_ut_test
        graph_base
        graph
        intf_pub
        )
set_target_properties(ut_graph_construction PROPERTIES CXX_STANDARD 17)

add_test(NAME ut_graph_construction COMMAND ut_graph_construction --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/ut_graph_construction.xml)
set_tests_properties(ut_graph_construction PROPERTIES LABELS "st;st_hetero;ut_graph_construction")
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${METADEF_DIR}/inc/common/util/trace_manager)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos/proto)
include_directories(${AIR_CODE_DIR}/graph_metadef/graph)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CMAKE_BINARY_DIR}/proto/ge/proto)
include_directories(${AIR_CODE_DIR}/tests)
include_directories(${AIR_CODE_DIR}/tests/framework/ge_runtime_stub/include/)
include_directories(${AIR_CODE_DIR}/inc/framework)
include_directories(${AIR_CODE_DIR}/inc/external)

file(GLOB_RECURSE UT_FILES CONFIGURE_DEPENDS "${GE_METADEF_TEST_DIR}/ut/graph/testcase/*.cc")
file(GLOB_RECURSE FAKE_FILES CONFIGURE_DEPENDS "${GE_METADEF_TEST_DIR}/depends/cache_desc_stub/runtime_cache_desc.cc")
file(GLOB_RECURSE UTILS_FILES CONFIGURE_DEPENDS "${GE_METADEF_TEST_DIR}/ut/graph/common/*.cc")
file(GLOB_RECURSE FAKER_SRCS CONFIGURE_DEPENDS "${GE_METADEF_TEST_DIR}/depends/faker/kernel_run*.cc")
add_executable(ut_graph ${UT_FILES} ${FAKE_FILES} ${UTILS_FILES} ${FAKER_SRCS})

target_compile_options(ut_graph PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
    -fno-access-control
)

target_compile_definitions(ut_graph PRIVATE
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    google=ascend_private
    FUNC_VISIBILITY
)

target_link_libraries(ut_graph PRIVATE
    intf_pub -lgcov
    -Wl,--no-as-needed
    platform
    slog_headers
    ge_runtime_stub eager_style_graph_builder_base
    metadef_headers mmpa_stub graph graph_base error_manager aihac_symbolizer runtime_headers
    msprof_headers GTest::gtest GTest::gtest_main ascend_protobuf slog_stub c_sec json -lrt -ldl
    ascendcl_stub
)

target_include_directories(ut_graph PRIVATE
    ${GE_METADEF_TEST_DIR}/depends
    ${GE_METADEF_TEST_DIR}/ut/graph
    ${GE_METADEF_TEST_DIR}/ut/graph/common
)

target_link_options(ut_graph PRIVATE
        -Wl,--disable-new-dtags
        -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
        )
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${METADEF_DIR}/inc/common/util/trace_manager)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos/proto)
include_directories(${METADEF_DIR})
include_directories(${METADEF_DIR}/graph)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CMAKE_BINARY_DIR}/proto/ge/proto)

include_directories(${ASCEND_INSTALL_PATH}/include/experiment)
include_directories(${ASCEND_INSTALL_PATH}/include/experiment/runtime)
include_directories(${ASCEND_INSTALL_PATH}/include/experiment/msprof)
include_directories(${AIR_CODE_DIR}/tests)
include_directories(${CMAKE_BINARY_DIR}/tests/depends/aihacb_autofusion)

add_compile_definitions(CMAKE_BINARY_DIR=\"${CMAKE_BINARY_DIR}\")
file(GLOB_RECURSE UT_FILES CONFIGURE_DEPENDS "${GE_METADEF_TEST_DIR}/ut/ascendc_ir/testcase/*.cc")
file(GLOB_RECURSE FAKE_FILES CONFIGURE_DEPENDS "${GE_METADEF_TEST_DIR}/depends/cache_desc_stub/runtime_cache_desc.cc")
file(GLOB_RECURSE UTILS_FILES CONFIGURE_DEPENDS "${GE_METADEF_DIR}/graph/ascendc_ir/utils/mem_utils.cc")
add_executable(ut_ascendc_ir ${UT_FILES} ${UTILS_FILES} ${FAKE_FILES})

target_compile_options(ut_ascendc_ir PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Wno-deprecated-declarations
    -Wno-error=unused-variable
    -Wall -Wfloat-equal -Werror
    -D_GLIBCXX_USE_CXX11_ABI=0
    -fno-access-control
)

target_compile_definitions(ut_ascendc_ir PRIVATE
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    google=ascend_private
    FUNC_VISIBILITY
)

add_dependencies(ut_ascendc_ir aihacb_autofusion_stub)

# intf_pub包含stdc++11 当前需要按照更高版本来编译ut
target_link_libraries(ut_ascendc_ir PRIVATE
    -lgcov
    -Wl,--no-as-needed
    platform
    runtime_stub
    slog
    slog_headers
    metadef_headers
    ascir_builtin_ops_stub
    #ascir_generate #正常不应该依赖这个，当前因为ut要校验generate能力
    aihac_ir aihac_symbolizer aihac_ir_register graph graph_base error_manager mmpa
    GTest::gtest GTest::gtest_main ascend_protobuf slog_stub slog c_sec json mmpa_stub -lrt -ldl
)

target_include_directories(ut_ascendc_ir PRIVATE
    ${GE_METADEF_TEST_DIR}/depends
    ${GE_METADEF_TEST_DIR}/ut/ascendc_ir
)

add_test(NAME ut_ascendc_ir COMMAND ut_ascendc_ir --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_ascendc_ir.xml)
set_tests_properties(ut_ascendc_ir PROPERTIES LABELS "ut;ge_rt;ut_ascendc_ir")
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
include_directories(${CMAKE_SOURCE_DIR})
add_compile_definitions(TOP_DIR=\"${CMAKE_SOURCE_DIR}\")
file(GLOB_RECURSE UT_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/tests/ge/ut/sc_check/testcase/*.cc")
add_executable(ut_sc_check ${UT_FILES})

target_compile_options(ut_sc_check PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
    -D_GLIBCXX_USE_CXX11_ABI=0
)

target_compile_definitions(ut_sc_check PRIVATE
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    google=ascend_private
    FUNC_VISIBILITY
)

target_link_libraries(ut_sc_check PRIVATE
    -lgcov
    -Wl,--no-as-needed
    GTest::gtest GTest::gtest_main -lrt -ldl
)

add_test(NAME ut_sc_check COMMAND ut_sc_check --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_sc_check.xml)
set_tests_properties(ut_sc_check PROPERTIES LABELS "ut;ge_common;ut_sc_check")
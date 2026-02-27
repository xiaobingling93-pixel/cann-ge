# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE DVPP_TEST_SRCS CONFIGURE_DEPENDS "*.cc")

add_compile_definitions(DVPP_LLT)
add_compile_definitions(RUN_TEST)

add_executable(ut_dvpp_runtime2_test
         ${DVPP_TEST_SRCS}
         ${AIR_CODE_DIR}/runtime/v2/engine/dvpp/kernel/dvpp_rt_kernel.cc)

target_compile_options(ut_dvpp_runtime2_test PRIVATE
        ${AIR_COMMON_COMPILE_OPTION}
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal -Werror
        --coverage -fprofile-arcs -ftest-coverage
        )

target_compile_definitions(ut_dvpp_runtime2_test PRIVATE
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
        CMAKE_BINARY_DIR=\"${CMAKE_BINARY_DIR}\"
        )

target_include_directories(ut_dvpp_runtime2_test PRIVATE
        ${ASCEND_INSTALL_PATH}/include
        ${AIR_CODE_DIR}/runtime/v2/engine/dvpp/kernel
        ${AIR_CODE_DIR}/tests/ge/ut/ge/runtime/fast_v2/engine/dvpp/dvpp_rtkernel
        )

target_link_libraries(ut_dvpp_runtime2_test PRIVATE
        intf_pub
        ${AIR_COMMON_LINK_OPTION}
        -Wl,--whole-archive
        -Wl,--no-whole-archive
        GTest::gtest
        GTest::gmock
        GTest::gtest_main
        c_sec
        mockcpp_static
        )

add_test(NAME ut_dvpp_runtime2_test COMMAND ut_dvpp_runtime2_test --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_dvpp_runtime2_test.xml)
set_tests_properties(ut_dvpp_runtime2_test PROPERTIES LABELS "ut;ge_rt;ut_dvpp_runtime2_test")

include(CTest)
enable_testing()
add_test(NAME test COMMAND ut_dvpp_runtime2_test)

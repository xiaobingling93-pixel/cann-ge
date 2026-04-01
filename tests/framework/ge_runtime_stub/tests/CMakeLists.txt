# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cc" "*.CC" "*.cpp" "*.CPP")

add_executable(ge_runtime_stub_test ${SOURCES})

target_include_directories(ge_runtime_stub_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${AIR_CODE_DIR}/inc ${AIR_CODE_DIR}/inc/framework ${AIR_CODE_DIR}/inc/external
        )

target_compile_options(ge_runtime_stub_test PRIVATE
        -g
        )

target_link_libraries(ge_runtime_stub_test PUBLIC intf_llt_pub GTest::gtest GTest::gtest_main ge_runtime_stub gert)

include(CTest)
enable_testing()
add_test(NAME test COMMAND ge_runtime_stub_test)
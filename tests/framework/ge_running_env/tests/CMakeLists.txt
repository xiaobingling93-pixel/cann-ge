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

add_executable(ge_running_env_test ${SOURCES})

target_include_directories(ge_running_env_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(ge_running_env_test PRIVATE
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(ge_running_env_test PRIVATE
    -g
)

target_link_libraries(ge_running_env_test PUBLIC intf_pub GTest::gtest ge_running_env graphengine flow_graph ascendcl_stub
-Wl,--no-as-needed)

add_test(NAME ge_running_env_test COMMAND ge_running_env_test --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/ge_running_env_test.xml)
set_tests_properties(ge_running_env_test PROPERTIES LABELS "st;st_hetero;ge_running_env_test")

include(CTest)
enable_testing()
add_test(NAME test COMMAND ge_running_env_test)
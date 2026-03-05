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

add_executable(hybrid_model_async_exec_test ${SOURCES})

target_include_directories(hybrid_model_async_exec_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/tests/depends/op_stub
        )

target_compile_definitions(hybrid_model_async_exec_test PRIVATE
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
        )
target_compile_options(hybrid_model_async_exec_test PUBLIC -g
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal -Werror
        -fno-access-control
        )
target_link_libraries(hybrid_model_async_exec_test PUBLIC 
        intf_pub
        flow_graph
        -Wl,-z,muldefs
        -Wl,--whole-archive
        gert
        gert_op_impl_static
        ascendcl_stub
        -Wl,--no-whole-archive
        GTest::gtest GTest::gtest_main st_stubs ge_graph_dsl ge_running_env es_ge_test ge_runtime_stub gert_op_impl)

include(CTest)
enable_testing()
add_test(NAME test COMMAND hybrid_model_async_exec_test)
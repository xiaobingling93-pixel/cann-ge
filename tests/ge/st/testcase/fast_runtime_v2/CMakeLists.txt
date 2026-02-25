# Copyright 2022 Huawei Technologies Co., Ltd
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE RUNTIME2_TEST_SRCS CONFIGURE_DEPENDS "*.cc" )

list(REMOVE_ITEM RUNTIME2_TEST_SRCS
        "${AIR_CODE_DIR}/tests/ge/st/testcase/fast_runtime_v2/dvpp/dvpp_rtkernel/dvpp_rtkernel_unittest.cc"
        "${AIR_CODE_DIR}/tests/ge/st/testcase/fast_runtime_v2/dvpp/dvpp_rtkernel/stub_dvpp_context.cc"
)

add_executable(st_fast_runtime2_test
        "${AIR_CODE_DIR}/tests/depends/graph_tuner/rt2_src/graph_tunner_rt2_stub.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/ffts_plus_proto_tools.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/test_tools_task_info.cc"
        "${AIR_CODE_DIR}/tests/ge/st/testcase/dump/dynamic_shape/dynamic_shape_dump_test.cc"
        ${RUNTIME2_TEST_SRCS}
        "${AIR_CODE_DIR}/tests/graph_metadef/depends/faker/plugin_manager_stub.cc")

target_link_libraries(st_fast_runtime2_test PRIVATE intf_pub ${AIR_COMMON_LINK_OPTION})
target_compile_options(st_fast_runtime2_test PRIVATE ${AIR_COMMON_COMPILE_OPTION}
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal -Werror
        -Wno-subobject-linkage
        )
target_compile_definitions(st_fast_runtime2_test PRIVATE
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
        CMAKE_BINARY_DIR=\"${CMAKE_BINARY_DIR}\"
        )

target_include_directories(st_fast_runtime2_test PRIVATE
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/runtime/v2
        ${AIR_CODE_DIR}/runtime/v2/engine
        ${AIR_CODE_DIR}/runtime/v1
        ${AIR_CODE_DIR}/compiler
        ${AIR_CODE_DIR}/tests/depends/
        ${AIR_CODE_DIR}/tests/
        ${AIR_CODE_DIR}/tests/ge/st/testcase/
        ${AIR_CODE_DIR}/tests/ge/st/stubs/
        ${AIR_CODE_DIR}/tests/framework/ge_running_env/include/
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/graph_metadef
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/tests/depends/op_stub
        ${AIR_CODE_DIR}/tests/ge/ut/ge/runtime/fast_v2/kernel/memory_profiling_log_matcher.h
        )

target_link_options(st_fast_runtime2_test PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        )

target_link_libraries(st_fast_runtime2_test PRIVATE 
        intf_pub
        GTest::gtest
        GTest::gtest_main
        adump_headers
        gert
        ge_graph_dsl
        ge_executor_shared
        davinci_executor
        hybrid_executor
        ge_common
        ge_common_base
        metadef_headers
        -Wl,--no-as-needed
        profiler_stub
        ge_runner
        ge_running_env
        ge_compiler
        st_stubs
        es_ge_test
        -Wl,--whole-archive
        rt2_registry_static
        gert_op_impl_static
        ge_runtime_stub
        ascendcl_stub
        -Wl,--no-whole-archive
      )

include(CTest)
enable_testing()
add_test(NAME test COMMAND st_fast_runtime2_test)

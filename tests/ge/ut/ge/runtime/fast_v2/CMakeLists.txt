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
        "${AIR_CODE_DIR}/tests/ge/ut/ge/runtime/fast_v2/engine/dvpp/dvpp_rtkernel/dvpp_rtkernel_unittest.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/runtime/fast_v2/engine/dvpp/dvpp_rtkernel/stub_dvpp_context.cc"
)

add_executable(ut_fast_runtime2_test
        "${AIR_CODE_DIR}/tests/depends/graph_tuner/rt2_src/graph_tunner_rt2_stub.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/ffts_plus_proto_tools.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/test_tools_task_info.cc"
        "${AIR_CODE_DIR}/tests/depends/op_stub/op_impl/autotiling/norm.cc"
        "${AIR_CODE_DIR}/tests/depends/op_stub/less_important_op_impl.cc"
        "${AIR_CODE_DIR}/tests/depends/op_stub/op_impl/dynamicatomicaddrclean/dynamic_atomic_addr_clean_impl.cc"
        "${AIR_CODE_DIR}/tests/depends/op_stub/op_impl/maxpoolv3/max_pool_v3_impl.cc"
        "${AIR_CODE_DIR}/tests/depends/op_stub/op_impl/dynamic_rnn_impl.cc"
        "${AIR_CODE_DIR}/tests/depends/op_stub/op_impl/reshape_op_impl.cc"
        "${AIR_CODE_DIR}/tests/depends/op_stub/op_impl/transdata/transdata_op_impl.cc"
        "${AIR_CODE_DIR}/tests/depends/op_stub/op_impl/transdata/trans_data_negative_target_tc_201.cc"
        "${AIR_CODE_DIR}/tests/depends/op_stub/op_impl/transdata/trans_data_positive_source_tc_1010.cc"
        "${AIR_CODE_DIR}/tests/depends/op_stub/data_flow_op_impl.cc"
         ${RUNTIME2_TEST_SRCS})

target_compile_options(ut_fast_runtime2_test PRIVATE
        ${AIR_COMMON_COMPILE_OPTION}
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal -Werror
        -fno-sanitize-address-use-after-scope
        )

target_compile_definitions(ut_fast_runtime2_test PRIVATE
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
        CMAKE_BINARY_DIR=\"${CMAKE_BINARY_DIR}\"
        )

target_include_directories(ut_fast_runtime2_test PRIVATE
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/inc
        ${AIR_CODE_DIR}/compiler
        ${AIR_CODE_DIR}/tests/depends/
        ${AIR_CODE_DIR}/runtime/v2
        ${AIR_CODE_DIR}/runtime/v2/engine
        ${AIR_CODE_DIR}/runtime/v1
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/graph_metadef
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}/tests/depends/op_stub
        ${AIR_CODE_DIR}/tests/ge/ut/ge/runtime/fast_v2
        ${AIR_CODE_DIR}/tests/framework/macro_utils
        ${METADEF_DIR}/pkg_inc
        )

target_link_options(ut_fast_runtime2_test PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        )

target_link_libraries(ut_fast_runtime2_test PRIVATE
        intf_pub
        ${AIR_COMMON_LINK_OPTION}
        -Wl,--whole-archive
        rt2_registry_static
        -Wl,--no-whole-archive
        metadef_headers
        datagw_headers
        GTest::gtest
        GTest::gmock
        GTest::gtest_main
        gert
        ge_graph_dsl
        ge_running_env
        ge_executor_shared 
        davinci_executor 
        hybrid_executor
        ge_common
        ge_common_base
        ge_runtime_stub
        ascendcl_stub
        )

add_test(NAME ut_fast_runtime2_test COMMAND ut_fast_runtime2_test --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_fast_runtime2_test.xml)
set_tests_properties(ut_fast_runtime2_test PROPERTIES LABELS "ut;ge_rt;ut_fast_runtime2_test")

include(CTest)
enable_testing()
add_test(NAME test COMMAND ut_fast_runtime2_test)

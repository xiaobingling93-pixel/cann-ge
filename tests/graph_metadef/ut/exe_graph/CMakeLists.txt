# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE TEST_SRCS CONFIGURE_DEPENDS "${GE_METADEF_TEST_DIR}/ut/exe_graph/*.cc")
file(GLOB_RECURSE FAKER_SRCS CONFIGURE_DEPENDS "${GE_METADEF_TEST_DIR}/depends/faker/*.cc")

add_executable(ut_exe_graph ${TEST_SRCS} ${FAKER_SRCS} ${GE_RUNTIME_STUB_SRCS})

target_include_directories(ut_exe_graph PRIVATE
        ${GE_METADEF_DIR}
        ${GE_METADEF_TEST_DIR}/depends
		${AIR_CODE_DIR}/tests/framework/ge_runtime_stub/include
        ${GE_METADEF_DIR}/exe_graph
        ${CMAKE_BINARY_DIR}
        ${CMAKE_BINARY_DIR}/proto/metadef_protos
        ${METADEF_DIR}/inc/external/base
		${AIR_CODE_DIR}/inc/framework
        )

target_compile_options(ut_exe_graph PRIVATE
        -g --coverage -fprofile-arcs -ftest-coverage
	    -Wno-deprecated-declarations
	    -Wall -Wfloat-equal -Werror
        -fno-access-control
        )

target_compile_definitions(ut_exe_graph PRIVATE
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
        google=ascend_private
        )

target_link_options(ut_exe_graph PRIVATE
		-rdynamic
		-Wl,-Bsymbolic
		)

target_link_libraries(ut_exe_graph PRIVATE
        intf_pub
        -Wl,--no-as-needed
        slog_headers
        platform
        msprof_headers
        runtime_headers
        -Wl,--whole-archive
        rt2_registry_static
        -Wl,--no-whole-archive
        ge_runtime_stub exe_graph lowering register opp_registry error_manager
        mmpa
        GTest::gtest GTest::gtest_main unified_dlog ascend_protobuf c_sec mmpa_stub -lrt -ldl -lgcov
        metadef_headers
        graph
        graph_base
		aihac_symbolizer
        -Wl,--as-needed
        )

add_test(NAME ut_exe_graph COMMAND ut_exe_graph --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_exe_graph.xml)
set_tests_properties(ut_exe_graph PROPERTIES LABELS "ut;ge_rt;ut_exe_graph")
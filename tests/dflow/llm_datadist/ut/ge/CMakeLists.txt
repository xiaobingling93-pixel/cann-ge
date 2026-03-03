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
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/data_flow_protos)
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CMAKE_BINARY_DIR}/proto/ge/proto)
include_directories(${CMAKE_BINARY_DIR}/proto_grpc/deployer_proto)

include_directories(${AIR_CODE_DIR}/base)
include_directories(${AIR_CODE_DIR}/compiler)
include_directories(${AIR_CODE_DIR}/runtime/v1)
include_directories(${AIR_CODE_DIR}/api/session)
include_directories(${AIR_CODE_DIR}/dflow/compiler/session)
include_directories(${AIR_CODE_DIR}/inc)
include_directories(${AIR_CODE_DIR}/inc/graph_metadef)
include_directories(${AIR_CODE_DIR}/inc/external)
include_directories(${AIR_CODE_DIR}/inc/framework)
include_directories(${AIR_CODE_DIR}/runtime/v2)

include_directories(${PARSER_DIR})
include_directories(${AIR_CODE_DIR}/parser)
include_directories(${AIR_CODE_DIR}/parser/parser)
include_directories(${AIR_CODE_DIR}/inc/parser)
include_directories(${AIR_CODE_DIR}/inc/parser/external)

include_directories(${AIR_CODE_DIR}/tests/)
include_directories(${AIR_CODE_DIR}/tests/dflow/runner/ut/ge)
include_directories(${AIR_CODE_DIR}/tests/dflow/runner/ut/ge/runtime/stub)
include_directories(${AIR_CODE_DIR}/tests/framework/ge_graph_dsl/include)
include_directories(${AIR_CODE_DIR}/tests/framework/ge_runtime_stub/include/)
include_directories(${AIR_CODE_DIR}/tests/framework/easy_graph/include)
include_directories(${AIR_CODE_DIR}/tests/framework/macro_utils)
include_directories(${METADEF_DIR}/inc)
include_directories(${METADEF_DIR}/pkg_inc)
include_directories(${METADEF_DIR})
include_directories(${ASCGEN_DIR}/inc)

# test files
set(LLM_ENGINE_TEST_FILES
        "dflow/llm_datadist/llm_datadist_v2_unittest.cc"
        "dflow/llm_datadist/llm_utils_unittest.cc"
        "dflow/llm_datadist/llm_datadist_v1_api_unittest.cc"
)

# ut_llm_engine_utest
add_executable(ut_llm_engine_utest
        ${LLM_ENGINE_TEST_FILES}
        )

target_compile_options(ut_llm_engine_utest PRIVATE ${AIR_COMMON_COMPILE_OPTION}
        -Werror=format
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal -Werror
        )

target_compile_definitions(ut_llm_engine_utest PRIVATE
        google=ascend_private
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
        )

target_include_directories(ut_llm_engine_utest PRIVATE
        ${DFLOW_CODE_DIR}/inc
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}/tests/depends/llm_datadist
        ${AIR_CODE_DIR}/dflow/llm_datadist/v1
        ${AIR_CODE_DIR}/dflow/llm_datadist/v1/common
        ${AIR_CODE_DIR}/tests/depends/llm_datadist/src
        ${ASCEND_3RD_LIB_PATH}/json/include
        ${ASCGEN_DIR}/inc
        #for session_util.h
        ${AIR_CODE_DIR}/api/session
        )

target_link_options(ut_llm_engine_utest PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(ut_llm_engine_utest
        c_sec_headers
        mmpa_headers
        runtime_headers
        ascendcl_headers
        msprof_headers
        air_headers
        slog_headers
        metadef_headers
        c_sec
        slog_stub
        error_manager
        json
        runtime_stub
        mmpa_stub2
        intf_pub
        ascendcl_stub
        llm_engine_base
        ${AIR_COMMON_LINK_OPTION}
        -Wl,--whole-archive
        -Wl,--no-whole-archive
        GTest::gtest GTest::gtest_main GTest::gmock_main
        -lrt -ldl -lpthread -lgcov
        -Wl,--no-as-needed unified_dlog GTest::gmock -Wl,--as-needed
        )

add_test(NAME ut_llm_engine_utest COMMAND ut_llm_engine_utest --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_llm_engine_utest.xml)
set_tests_properties(ut_llm_engine_utest PROPERTIES LABELS "ut;ge_rt;ut_llm_engine_utest")

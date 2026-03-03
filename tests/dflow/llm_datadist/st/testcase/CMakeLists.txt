# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_directories(${AIR_CODE_DIR}/tests/)
include_directories(${AIR_CODE_DIR}/tests/ge/ut)
include_directories(${AIR_CODE_DIR}/tests/ge/ut/ge)

################################################################################
# build llm_engine_test
set(LLM_ENGINE_TEST_FILES
        llm_datadist/test_llm_engine_v2.cc
        llm_datadist/test_llm_datadist_v1_api.cc
)

add_executable(llm_engine_test
        ${LLM_ENGINE_TEST_FILES}
)

target_compile_definitions(llm_engine_test PRIVATE
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
        google=ascend_private
)
target_compile_options(llm_engine_test PRIVATE
        ${AIR_COMMON_COMPILE_OPTION}
)

target_include_directories(llm_engine_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/dflow/llm_datadist/v1
        ${AIR_CODE_DIR}/dflow/llm_datadist/v1/flow_func
        ${AIR_CODE_DIR}/dflow/llm_datadist/v1/common
        ${AIR_CODE_DIR}/tests/depends/llm_datadist/src
        ${ASCEND_3RD_LIB_PATH}/json/include
        ${ASCGEN_DIR}/inc
        ${AIR_CODE_DIR}/api/session
)

target_link_libraries(llm_engine_test
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
        runtime_stub
        mmpa_stub2
        ascendcl_stub
        intf_pub
        llm_engine_base
        ${AIR_COMMON_LINK_OPTION}
        GTest::gtest GTest::gtest_main GTest::gmock
        -Wl,--no-as-needed
        unified_dlog
)

add_test(NAME llm_engine_test COMMAND llm_engine_test --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/llm_engine_test.xml)
set_tests_properties(llm_engine_test PROPERTIES LABELS "st;st_hetero;llm_engine_test")

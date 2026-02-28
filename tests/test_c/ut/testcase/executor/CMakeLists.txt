# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# ge executor c ut
set(GE_EXECUTOR_C_TESTED_FILES
    ${AIR_CODE_DIR}/runtime/c/executor/ge_executor.c
    ${AIR_CODE_DIR}/runtime/c/executor/model_loader.c
    ${AIR_CODE_DIR}/runtime/c/executor/model_full_load.c
    ${AIR_CODE_DIR}/runtime/c/executor/model_executor.c
    ${AIR_CODE_DIR}/runtime/c/executor/model_desc.c
    ${AIR_CODE_DIR}/runtime/c/executor/model_parse.c
    ${AIR_CODE_DIR}/runtime/c/executor/maintain_manager_dynamic.c
    ${AIR_CODE_DIR}/runtime/c/executor/model_manager.c
)

set(GE_EXECUTOR_C_DBG_TESTED_FILES
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/dump_config.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/dump.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/dump_thread_manager_dynamic.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/parse_dbg_file.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/parse_json_file.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/tlv_parse.c
    ${AIR_CODE_DIR}/runtime/c/dbg/profiling/profiling_dynamic.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dbg_main.c
)

set(GE_EXECUTOR_C_TESTED_STUB
    ${AIR_CODE_DIR}/tests/test_c/ut/testcase/executor/gmock_stub/ascend_rt_stub.cc
    ${AIR_CODE_DIR}/tests/test_c/ut/testcase/executor/gmock_stub/mmpa_api_stub.cc
)


set(GE_EXECUTOR_C_TEST_CASE_FILES
    "main.cc"
    "profiling_test.cc"
    "json_parser_test.cc"
    "ge_executor_test.cc"
    "ge_dbg_test.cc"
)

set(C_BASE_JSON_FILES
    ${AIR_CODE_DIR}/tests/test_c/c_base/src/json_parser.c
    ${AIR_CODE_DIR}/tests/test_c/c_base/src/mem_pool.c
    ${AIR_CODE_DIR}/tests/test_c/c_base/src/ref_obj.c
    ${AIR_CODE_DIR}/tests/test_c/c_base/src/sort_vector.c
    ${AIR_CODE_DIR}/tests/test_c/c_base/src/vector.c
)

add_executable(ut_ge_executor_c_utest
    ${C_BASE_JSON_FILES}
    ${GE_EXECUTOR_C_TESTED_STUB}
    ${GE_EXECUTOR_C_TESTED_FILES}
    ${GE_EXECUTOR_C_DBG_TESTED_FILES}
    ${GE_EXECUTOR_C_TEST_CASE_FILES}
)

target_include_directories(ut_ge_executor_c_utest PRIVATE
    ${AIR_CODE_DIR}/runtime/c/dbg/profiling
    ${AIR_CODE_DIR}/runtime/c/dbg/
    ${AIR_CODE_DIR}/runtime/c/executor/
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/framework/executor_c/
    ${AIR_CODE_DIR}/inc/framework/executor_c/c_base
    ${AIR_CODE_DIR}/inc/framework/
    ${AIR_CODE_DIR}/tests/test_c/ut/testcase/executor
    ${AIR_CODE_DIR}/tests/test_c/ut/testcase/executor/gmock_stub
    ${ASCEND_INSTALL_PATH}/pkg_inc/base
)

target_compile_options(ut_ge_executor_c_utest PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Wextra
    -Wall
    # -Werror
    -Wfloat-equal
)

target_compile_definitions(ut_ge_executor_c_utest PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    RUN_TEST
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},LiteOS>,NANO_OS_TYPE=1,NANO_OS_TYPE=0>
    UT_DATA_DIR="${AIR_CODE_DIR}/tests/test_c/ut/testcase/"
)

target_link_libraries(ut_ge_executor_c_utest
    c_sec
    msprof_headers
    runtime_headers
    -Wl,--whole-archive
    -Wl,--no-whole-archive
    GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main ${COMMON_SHARED_LIBRARIES} -lrt -lpthread -lgcov
)

# LiteOS
set(GE_EXECUTOR_C_LITEOS_TESTED_FILES
    ${AIR_CODE_DIR}/runtime/c/executor/ge_executor.c
    ${AIR_CODE_DIR}/runtime/c/executor/model_loader.c
    ${AIR_CODE_DIR}/runtime/c/executor/model_executor.c
    ${AIR_CODE_DIR}/runtime/c/executor/model_desc.c
    ${AIR_CODE_DIR}/runtime/c/executor/model_parse.c
    ${AIR_CODE_DIR}/runtime/c/executor/model_manager.c
    ${AIR_CODE_DIR}/runtime/c/executor/model_segment_load.c
    ${AIR_CODE_DIR}/runtime/c/executor/maintain_manager_static.c
)

set(GE_EXECUTOR_C_DBG_LITEOS_TESTED_FILES
    ${AIR_CODE_DIR}/runtime/c/dbg/dbg_stub_liteos/dump/dump.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dbg_stub_liteos/profiling/profiling.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dbg_main.c
)


set(GE_EXECUTOR_C_LITEOS_TEST_CASE_FILES
    "main.cc"
    "ge_executor_liteos_test.cc"
)

add_executable(ut_ge_executor_c_liteos_utest
    ${C_BASE_JSON_FILES}
    ${GE_EXECUTOR_C_TESTED_STUB}
    ${GE_EXECUTOR_C_LITEOS_TESTED_FILES}
    ${GE_EXECUTOR_C_DBG_LITEOS_TESTED_FILES}
    ${GE_EXECUTOR_C_LITEOS_TEST_CASE_FILES}
)

target_include_directories(ut_ge_executor_c_liteos_utest PRIVATE
    ${AIR_CODE_DIR}/runtime/c/dbg/profiling
    ${AIR_CODE_DIR}/runtime/c/dbg/
    ${AIR_CODE_DIR}/runtime/c/executor/
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/framework/executor_c/
    ${AIR_CODE_DIR}/inc/framework/executor_c/c_base
    ${AIR_CODE_DIR}/inc/framework/
    ${AIR_CODE_DIR}/tests/test_c/ut/testcase/executor
    ${AIR_CODE_DIR}/tests/test_c/ut/testcase/executor/gmock_stub
    ${ASCEND_INSTALL_PATH}/pkg_inc/base
)

target_compile_options(ut_ge_executor_c_liteos_utest PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Wextra
    -Wall
    # -Werror
    -Wfloat-equal
)

target_compile_definitions(ut_ge_executor_c_liteos_utest PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    RUN_TEST
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},LiteOS>,NANO_OS_TYPE=1,NANO_OS_TYPE=0>
    UT_DATA_DIR="${AIR_CODE_DIR}/tests/test_c/ut/testcase/"
)

target_link_libraries(ut_ge_executor_c_liteos_utest
    c_sec
    msprof_headers
    runtime_headers
    -Wl,--whole-archive
    -Wl,--no-whole-archive
    GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main ${COMMON_SHARED_LIBRARIES} -lrt -lpthread -lgcov
)

# liteos static
set(DBG_TESTED_STATIC_STUB
    ${AIR_CODE_DIR}/tests/test_c/ut/testcase/executor/gmock_stub/dbg_stub.cc
    ${AIR_CODE_DIR}/tests/test_c/ut/testcase/executor/gmock_stub/mmpa_api_stub.cc
)

set(DBG_LITEOS_STATIC_FILES
    ${AIR_CODE_DIR}/runtime/c/dbg/profiling/profiling_static.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/dump_thread_manager_static.c
)

set(DBG_LITEOS_TESTCASE_FILES
    "main.cc"
    "dbg_liteos_test.cc"
)

add_executable(ut_dbg_liteos_static_utest
    ${C_BASE_JSON_FILES}
    ${DBG_TESTED_STATIC_STUB}
    ${DBG_LITEOS_STATIC_FILES}
    ${DBG_LITEOS_TESTCASE_FILES}
)

target_include_directories(ut_dbg_liteos_static_utest PRIVATE
    ${AIR_CODE_DIR}/runtime/c/dbg/profiling
    ${AIR_CODE_DIR}/runtime/c/dbg/
    ${AIR_CODE_DIR}/runtime/c/executor/
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/framework/executor_c/
    ${AIR_CODE_DIR}/inc/framework/executor_c/c_base
    ${AIR_CODE_DIR}/tests/test_c/ut/testcase/executor
    ${AIR_CODE_DIR}/tests/test_c/ut/testcase/executor/gmock_stub
    ${ASCEND_INSTALL_PATH}/pkg_inc/base
)

target_compile_options(ut_dbg_liteos_static_utest PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Wextra
    -Wall
    # -Werror
    -Wfloat-equal
)

target_compile_definitions(ut_dbg_liteos_static_utest PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    RUN_TEST
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},LiteOS>,NANO_OS_TYPE=1,NANO_OS_TYPE=0>
)

target_link_libraries(ut_dbg_liteos_static_utest
    c_sec
    msprof_headers
    runtime_headers
    -Wl,--whole-archive
    -Wl,--no-whole-archive
    GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main ${COMMON_SHARED_LIBRARIES} -lrt -lpthread -lgcov
)

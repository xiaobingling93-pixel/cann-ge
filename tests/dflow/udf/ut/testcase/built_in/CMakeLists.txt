# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(BUILT_IN_LOCAL_STUB_FILES
        ../../../builtin_stub/hccl_test_stub.cpp
        ../../../builtin_stub/hal_stub.cpp
)

# built in flow func ut
set(LOCAL_BUILT_IN_FLOW_FUNC_SRC_FILES
        ${LOCAL_UDF_EXECUTOR_SRC_FILES}
)
file(GLOB_RECURSE LLM_FLOW_FUNC_SRC_FILES
        ${UDF_SRC_PATH}/built_in/*.cpp
)

set(LOCAL_BUILT_IN_FLOW_FUNC_TEST_CASE_FILES
        ../../main.cpp
        time_batch_flow_func_utest.cpp
        count_batch_flow_func_utest.cpp
        llm_service_flow_func_utest.cpp
)

set(proto_src_files
        ${UDF_SRC_PATH}/proto/ff_udf_attr.proto
        ${UDF_SRC_PATH}/proto/ff_udf_def.proto
        ${UDF_SRC_PATH}/proto/ff_deployer.proto
        ${UDF_SRC_PATH}/proto/ff_dynamic_sched.proto
        ${UDF_SRC_PATH}/proto/ff_dump_data.proto
)

protobuf_generate(udf_exe PROTO_SRCS PROTO_HDRS ${proto_src_files})

add_executable(ut_built_in_flow_func
        ${PROTO_SRCS}
        ${LOCAL_STUB_FILES}
        ${BUILT_IN_LOCAL_STUB_FILES}
        ${LOCAL_BUILT_IN_FLOW_FUNC_SRC_FILES}
        ${LLM_FLOW_FUNC_SRC_FILES}
        ${LOCAL_BUILT_IN_FLOW_FUNC_TEST_CASE_FILES}
)

target_include_directories(ut_built_in_flow_func PRIVATE
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/inc/graph_metadef
        ${AIR_CODE_DIR}/dflow/inc
        ${ASCEND_INSTALL_PATH}/include
        ${UDF_SRC_PATH}/inc/external
        ${UDF_SRC_PATH}/inc/external/flow_func
        ${UDF_SRC_PATH}/
        ${UDF_SRC_PATH}/built_in
        ${CMAKE_BINARY_DIR}/proto/udf_exe/proto
        ../../../stub
        ../../../
        ${SRC_PATH}/built_in
)

target_link_libraries(ut_built_in_flow_func PRIVATE ${AIR_COMMON_LINK_OPTION}
        c_sec
        ascend_protobuf
        -rdynamic
        $<BUILD_INTERFACE:intf_llt_pub>
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:ascend_hal_headers>
        $<BUILD_INTERFACE:datagw_headers>
        GTest::gtest GTest::gmock GTest::gmock_main
)

target_compile_options(ut_built_in_flow_func PRIVATE ${AIR_COMMON_COMPILE_OPTION}
        -rdynamic
        -fpermissive
        -Werror=format
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal -Werror
)

target_compile_definitions(ut_built_in_flow_func PRIVATE
        STATISTIC
        google=ascend_private
        UDF_TEST
)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        add_test(NAME ut_built_in_flow_func COMMAND ut_built_in_flow_func --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_built_in_flow_func.xml)
        set_tests_properties(ut_built_in_flow_func PROPERTIES LABELS "ut;ut_dflow;ut_built_in_flow_func")
else ()
        message(STATUS "Detected architecture: ${CMAKE_SYSTEM_PROCESSOR}, mock-cpp is only supported on x86_64, no ut_built_in_flow_func test on this architecture")
endif ()
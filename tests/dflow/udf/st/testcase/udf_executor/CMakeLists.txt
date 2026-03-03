# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(LOCAL_TEST_CASE_FILES
        ../../main.cpp
        main_stest.cpp
        flow_func_model_stest.cpp
        attr_value_impl_stest.cpp
        flow_func_thread_pool_stest.cpp
        flow_func_manager_stest.cpp
        flow_func_drv_manager_stest.cpp
        flow_func_executor_stest.cpp
        flow_func_executor_with_proxy_stest.cpp
        flow_func_executor_with_npu_sched_model.cpp
        muti_in_out_stest.cpp
        empty_input_stest.cpp
        balance_align_stest.cpp
        run_flow_model_stest.cpp
        run_flow_model_with_proxy_stest.cpp
        ascend_string_stest.cpp
        multi_flow_func_stest.cpp
        flow_func_logger_stest.cpp
        flow_func_profiling_stest.cpp
        flow_func_dump_host_stest.cpp
        flow_func_dump_device_stest.cpp
        memory_statistic_manager_stest.cpp
)

set(proto_src_files
        ${UDF_SRC_PATH}/proto/ff_udf_attr.proto
        ${UDF_SRC_PATH}/proto/ff_udf_def.proto
        ${UDF_SRC_PATH}/proto/ff_deployer.proto
        ${UDF_SRC_PATH}/proto/ff_dynamic_sched.proto
        ${UDF_SRC_PATH}/proto/ff_dump_data.proto
)

protobuf_generate(udf_exe PROTO_SRCS PROTO_HDRS ${proto_src_files})

add_executable(st_flow_func_executor
        ${PROTO_SRCS}
        ${LOCAL_SRC_FILES}
        ${LOCAL_STUB_FILES}
        ${LOCAL_TEST_CASE_FILES}
)

target_include_directories(st_flow_func_executor PRIVATE
        ../../
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/inc/graph_metadef
        ${AIR_CODE_DIR}/dflow/inc
        ${ASCEND_INSTALL_PATH}/include
        ${UDF_SRC_PATH}/inc/external
        ${UDF_SRC_PATH}/
        ${CMAKE_BINARY_DIR}/proto/udf_exe/proto
        ../../../stub
        ../../..
)

target_link_libraries(st_flow_func_executor PRIVATE ${AIR_COMMON_LINK_OPTION}
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

target_compile_options(st_flow_func_executor PRIVATE ${AIR_COMMON_COMPILE_OPTION}
        -rdynamic
        -fpermissive
        -Werror=format
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal -Werror
)

target_compile_definitions(st_flow_func_executor PRIVATE
        STATISTIC
        google=ascend_private
        UDF_TEST
)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        add_test(NAME st_flow_func_executor COMMAND st_flow_func_executor --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/st_flow_func_executor.xml)
        set_tests_properties(st_flow_func_executor PROPERTIES LABELS "st;st_dflow;st_flow_func_executor")
else ()
        message(STATUS "Detected architecture: ${CMAKE_SYSTEM_PROCESSOR}, mock-cpp is only supported on x86_64, no st_flow_func_executor test on this architecture")
endif ()
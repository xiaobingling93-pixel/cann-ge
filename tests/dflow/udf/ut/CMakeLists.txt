# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(UDF_SRC_PATH ${DFLOW_CODE_DIR}/udf)

set(LOCAL_UDF_EXECUTOR_SRC_FILES
        ${UDF_SRC_PATH}/flow_func/flow_func_manager.cpp
        ${UDF_SRC_PATH}/flow_func/flow_func_config_manager.cpp
        ${UDF_SRC_PATH}/flow_func/flow_func_context.cpp
        ${UDF_SRC_PATH}/flow_func/flow_func_run_context.cpp
        ${UDF_SRC_PATH}/flow_func/flow_func_processor.cpp
        ${UDF_SRC_PATH}/flow_func/async_executor.cpp
        ${UDF_SRC_PATH}/flow_func/flow_func_dumper.cpp
        ${UDF_SRC_PATH}/flow_func/flow_func_params.cpp
        ${UDF_SRC_PATH}/flow_func/single_func_wrapper.cpp
        ${UDF_SRC_PATH}/flow_func/multi_func_wrapper.cpp
        ${UDF_SRC_PATH}/flow_func/mbuf_flow_msg.cpp
        ${UDF_SRC_PATH}/flow_func/mbuf_flow_msg_queue.cpp
        ${UDF_SRC_PATH}/flow_func/flow_func_helper.cpp
        ${UDF_SRC_PATH}/flow_func/out_options.cpp
        ${UDF_SRC_PATH}/flow_func/flow_func_statistic.cpp
        ${UDF_SRC_PATH}/model/attr_value_impl.cpp
        ${UDF_SRC_PATH}/model/flow_func_model.cpp
        ${UDF_SRC_PATH}/common/data_utils.cpp
        ${UDF_SRC_PATH}/config/global_config.cpp
        ${UDF_SRC_PATH}/execute/flow_func_thread_pool.cpp
        ${UDF_SRC_PATH}/execute/flow_func_executor.cpp
        ${UDF_SRC_PATH}/execute/flow_func_drv_manager.cpp
        ${UDF_SRC_PATH}/execute/flow_model_impl.cpp
        ${UDF_SRC_PATH}/execute/memory_statistic_manager.cpp
        ${UDF_SRC_PATH}/execute/npu_sched_processor.cpp
        ${UDF_SRC_PATH}/execute/main.cpp
        ${UDF_SRC_PATH}/reader_writer/mbuf_reader.cpp
        ${UDF_SRC_PATH}/reader_writer/mbuf_writer.cpp
        ${UDF_SRC_PATH}/reader_writer/queue_wrapper.cpp
        ${UDF_SRC_PATH}/reader_writer/proxy_queue_wrapper.cpp
        ${UDF_SRC_PATH}/reader_writer/data_aligner.cpp
        ${UDF_SRC_PATH}/flow_func/flow_func_timer.cpp
        ${UDF_SRC_PATH}/flow_func/ascend_string.cpp
        ${UDF_SRC_PATH}/flow_func/logger/flow_func_logger_impl.cpp
        ${UDF_SRC_PATH}/flow_func/logger/flow_func_logger_manager.cpp
        ${UDF_SRC_PATH}/flow_func/logger/flow_func_logger_flow_control.cpp
        ${UDF_SRC_PATH}/toolchain/dump/udf_dump_manager.cpp
        ${UDF_SRC_PATH}/toolchain/dump/udf_dump_task_host.cpp
        ${UDF_SRC_PATH}/toolchain/dump/udf_dump_task_device.cpp
        ${UDF_SRC_PATH}/toolchain/dump/udf_dump.cpp
        ${UDF_SRC_PATH}/toolchain/profiling/udf_profiling_manager.cpp
)

set(LOCAL_STUB_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/../stub/slog_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../stub/mmpa_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/stub/drv_common_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/stub/drv_buf_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/stub/drv_queue_stub.cpp
)

add_subdirectory(testcase/built_in/)
add_subdirectory(testcase/udf_executor/)
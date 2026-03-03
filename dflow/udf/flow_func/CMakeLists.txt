# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (TARGET flow_func)
    message(STATUS "Target flow_func already exists, not use flow_func in ge.")
    return()
endif ()

###for flow_func so
set(flow_func_so_files
    flow_func_context.cpp
    flow_func_params.cpp
    flow_func_processor.cpp
    flow_func_statistic.cpp
    async_executor.cpp
    flow_func_run_context.cpp
    single_func_wrapper.cpp
    multi_func_wrapper.cpp
    flow_func_manager.cpp
    flow_func_config_manager.cpp
    flow_func_dumper.cpp
    mbuf_flow_msg.cpp
    mbuf_flow_msg_queue.cpp
    flow_func_timer.cpp
    ascend_string.cpp
    out_options.cpp
    flow_func_helper.cpp
    ../common/data_utils.cpp
    logger/flow_func_logger_impl.cpp
    logger/flow_func_logger_manager.cpp
    logger/flow_func_logger_flow_control.cpp
)
##for so inc path
set(flow_func_so_inc_path
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc/external
)

add_library(flow_func SHARED
    ${flow_func_so_files}
)
target_include_directories(flow_func PRIVATE
    ${flow_func_so_inc_path}
)

target_compile_options(flow_func PRIVATE
    -ftrapv
    -O2
    -fvisibility-inlines-hidden
    -fvisibility=hidden
    -Werror
    -Wextra
    -Wfloat-equal
    -fno-common
)

target_link_libraries(flow_func PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:slog_headers>
    $<BUILD_INTERFACE:mmpa_headers>
    $<BUILD_INTERFACE:ascend_hal_headers>
    PUBLIC c_sec
    reader_writer
    ascend_hal_stub
    $<IF:$<STREQUAL:${PRODUCT_SIDE},device>,alog,unified_dlog>
)
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS flow_func OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    RUNTIME DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(TARGETS flow_func OPTIONAL
    EXPORT flow_func-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/${CMAKE_SYSTEM_PROCESSOR}
    RUNTIME DESTINATION ${INSTALL_LIBRARY_DIR}/${CMAKE_SYSTEM_PROCESSOR}
)
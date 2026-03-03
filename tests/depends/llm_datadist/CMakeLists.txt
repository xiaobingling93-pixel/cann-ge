# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(llm_mmpa_stub SHARED
        "${AIR_CODE_DIR}/tests/depends/llm_datadist/src/mmpa_stub.cc"
)

target_include_directories(llm_mmpa_stub PRIVATE
        ${METADEF_DIR}/inc/external
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/dflow/llm_datadist/v1
        ${AIR_CODE_DIR}/dflow/llm_datadist/v1/common
        ${AIR_CODE_DIR}/tests
        ${AIR_CODE_DIR}/tests/depends/llm_datadist/src
)

target_link_libraries(llm_mmpa_stub PRIVATE
        c_sec_headers
        mmpa_headers
        runtime_headers
        ascendcl_headers
        msprof_headers
        air_headers
        slog_headers
        metadef_headers
        slog_stub
        -Wl,-as-needed
        ${AIR_COMMON_LINK_OPTION}
)

target_compile_options(llm_mmpa_stub PRIVATE
        ${AIR_COMMON_COMPILE_OPTION}
)

set(LLM_ENGINE_BASE_SRCS
        "${AIR_CODE_DIR}/tests/depends/llm_datadist/src/ascend_string_stub.cc"
        "${AIR_CODE_DIR}/tests/depends/llm_datadist/src/aligned_ptr.cc"
        "${AIR_CODE_DIR}/tests/depends/llm_datadist/src/tensor_stub.cc"
        "${AIR_CODE_DIR}/tests/depends/llm_datadist/src/df_stubs.cc"
        "${AIR_CODE_DIR}/tests/depends/llm_datadist/src/session_stub.cc"
        "${AIR_CODE_DIR}/tests/depends/llm_datadist/src/type_utils_stub.cc"
        "${AIR_CODE_DIR}/tests/depends/llm_datadist/src/cache_engine_stubs.cc"
        "${AIR_CODE_DIR}/tests/depends/llm_datadist/src/llm_test_helper.cc"
        "${AIR_CODE_DIR}/tests/depends/llm_datadist/src/llm_engine_test_helper.cc"
        "${AIR_CODE_DIR}/dflow/llm_datadist/v1/common/cache_engine.cc"
        "${AIR_CODE_DIR}/dflow/llm_datadist/v1/common/cache_manager.cc"
        "${AIR_CODE_DIR}/dflow/llm_datadist/v1/common/swap_impl.cc"
        "${AIR_CODE_DIR}/dflow/llm_datadist/v1/common/llm_ge_api.cc"
        "${AIR_CODE_DIR}/dflow/llm_datadist/v1/common/llm_utils.cc"
        "${AIR_CODE_DIR}/dflow/llm_datadist/v1/common/llm_common.cc"
        "${AIR_CODE_DIR}/dflow/llm_datadist/v1/common/link_manager.cc"
        "${AIR_CODE_DIR}/dflow/llm_datadist/v1/common/flow_graph_manager.cc"
        "${AIR_CODE_DIR}/dflow/llm_datadist/v1/common/llm_flow_service.cc"
        "${AIR_CODE_DIR}/dflow/llm_datadist/v1/llm_datadist.cc"
        "${AIR_CODE_DIR}/dflow/llm_datadist/v1/llm_error_codes.cc"
        "${AIR_CODE_DIR}/dflow/llm_datadist/v1/common/llm_file_saver.cc"
        "${AIR_CODE_DIR}/dflow/llm_datadist/v1/common/llm_thread_pool.cc"
        "${AIR_CODE_DIR}/dflow/llm_datadist/v1/api/llm_datadist_impl.cc"
)

add_library(llm_engine_base SHARED
        ${LLM_ENGINE_BASE_SRCS}
)

target_include_directories(llm_engine_base PRIVATE
        ${METADEF_DIR}/inc/external
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/dflow/llm_datadist/v1
        ${AIR_CODE_DIR}/dflow/llm_datadist/v1/common
        ${AIR_CODE_DIR}/tests/depends/llm_datadist/src
        ${ASCEND_3RD_LIB_PATH}/json/include
        #for session_util.h
        ${AIR_CODE_DIR}/api/session
)

target_link_libraries(llm_engine_base PRIVATE
        c_sec_headers
        mmpa_headers
        runtime_headers
        ascendcl_headers
        msprof_headers
        air_headers
        slog_headers
        metadef_headers
        intf_pub
        llm_mmpa_stub
        slog_stub
        runtime_stub
        ascendcl_stub
        -Wl,--no-as-needed
        unified_dlog
        -Wl,-as-needed
        ${AIR_COMMON_LINK_OPTION}
)

target_compile_options(llm_engine_base PRIVATE
        ${AIR_COMMON_COMPILE_OPTION}
)
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(LLM_ENGINE_WRAPPER_SRC_FILES
        "${AIR_CODE_DIR}/api/python/llm_wrapper/llm_wrapper.cc"
        "${AIR_CODE_DIR}/api/python/llm_wrapper/llm_tensor.cc"
        "${AIR_CODE_DIR}/api/python/llm_wrapper/llm_datadist_wrapper.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/llm_ge_api.cc")

add_library(llm_wrapper MODULE
        ${LLM_ENGINE_WRAPPER_SRC_FILES}
)

target_include_directories(llm_wrapper PRIVATE
        ${HI_PYTHON_INC}
        ${pybind11_INCLUDE_DIR}
        ${METADEF_DIR}/inc
        ${METADEF_DIR}/inc/external
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/dflow/llm_datadist/v1
        ${AIR_CODE_DIR}/dflow/llm_datadist/v1/common
        ${AIR_CODE_DIR}/tests/depends/llm_datadist/src
        ${ASCEND_3RD_LIB_PATH}/json/include
        ${AIR_CODE_DIR}/api/session
)

target_link_libraries(llm_wrapper PRIVATE
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
        intf_llt_pub
        ascendcl_stub
        llm_engine_base
        -Wl,--no-as-needed
        unified_dlog
        -Wl,-as-needed
        ${AIR_COMMON_LINK_OPTION}
)

set_target_properties(llm_wrapper
        PROPERTIES
        PREFIX ""
)

target_compile_options(llm_wrapper PRIVATE
        ${AIR_COMMON_COMPILE_OPTION}
        -Xlinker -export-dynamic
)

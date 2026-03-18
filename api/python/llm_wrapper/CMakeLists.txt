# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (NOT ENABLE_TEST)
    add_library(llm_wrapper MODULE
            llm_wrapper.cc
            llm_tensor.cc
            llm_datadist_wrapper.cc)

    target_include_directories(llm_wrapper PRIVATE
            ${HI_PYTHON_INC}
            ${pybind11_INCLUDE_DIR}
            ${METADEF_DIR}/inc/external
            ${METADEF_DIR}/pkg_inc
            ${AIR_CODE_DIR}/inc/external
            ${AIR_CODE_DIR}/dflow/llm_datadist/v1
    )

    target_link_libraries(llm_wrapper PRIVATE
            $<BUILD_INTERFACE:intf_pub>
            $<BUILD_INTERFACE:slog_headers>
            $<BUILD_INTERFACE:mmpa_headers>
            $<BUILD_INTERFACE:hccl_headers>
            $<BUILD_INTERFACE:pybind_options>
            alog
            flow_graph
            llm_engine
    )

    set_target_properties(llm_wrapper
            PROPERTIES
            PREFIX ""
    )

    target_compile_definitions(llm_wrapper PRIVATE
            PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF
    )

    target_compile_options(llm_wrapper PRIVATE
            -O2
            -Xlinker -export-dynamic
    )

    target_link_options(llm_wrapper PRIVATE
            -s
    )
endif ()

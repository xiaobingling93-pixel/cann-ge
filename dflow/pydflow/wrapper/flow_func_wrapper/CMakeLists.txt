# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

############ libflowfunc_wrapper.so ############
add_library(flowfunc_wrapper MODULE flow_func_wrapper.cpp)

target_include_directories(flowfunc_wrapper PRIVATE
        ${HI_PYTHON_INC}
        ${pybind11_INCLUDE_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/../
)

target_link_libraries(flowfunc_wrapper PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:udf_headers>
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:metadef_headers>
        $<BUILD_INTERFACE:pybind_options>
        flow_func
        unified_dlog
)

set_target_properties(flowfunc_wrapper
        PROPERTIES
        PREFIX ""
)

target_compile_definitions(flowfunc_wrapper PRIVATE
        PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF
)

target_compile_options(flowfunc_wrapper PRIVATE
        -Xlinker -export-dynamic
)

target_link_options(flowfunc_wrapper PRIVATE
        $<$<CONFIG:Release>:-s>
)
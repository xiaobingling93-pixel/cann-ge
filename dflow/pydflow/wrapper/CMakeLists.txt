# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

############ libdflow_wrapper.so ############
add_library(dflow_wrapper MODULE dflow_wrapper.cpp)

target_include_directories(dflow_wrapper PRIVATE
        ${HI_PYTHON_INC}
        ${pybind11_INCLUDE_DIR}
        ${AIR_CODE_DIR}
)

target_link_libraries(dflow_wrapper PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:udf_headers>
        $<BUILD_INTERFACE:metadef_headers>
        $<BUILD_INTERFACE:parser_headers>
        $<BUILD_INTERFACE:air_headers>
        $<BUILD_INTERFACE:pybind_options>
        unified_dlog
        flow_graph
        ge_runner_v2
        dflow_runner
        fmk_parser
        fmk_onnx_parser
)

set_target_properties(dflow_wrapper
        PROPERTIES
        PREFIX ""
)

target_compile_definitions(dflow_wrapper PRIVATE
        PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF
)

target_compile_options(dflow_wrapper PRIVATE
        -Xlinker -export-dynamic
)

target_link_options(dflow_wrapper PRIVATE
        $<$<CONFIG:Release>:-s>
)


############ libdata_wrapper.so ############
add_library(data_wrapper MODULE data_wrapper.cpp)

target_include_directories(data_wrapper PRIVATE
        ${HI_PYTHON_INC}
        ${pybind11_INCLUDE_DIR}
)

target_compile_definitions(data_wrapper PRIVATE
        PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF
)

target_link_libraries(data_wrapper PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:udf_headers>
        $<BUILD_INTERFACE:metadef_headers>
        $<BUILD_INTERFACE:pybind_options>
)

set_target_properties(data_wrapper
        PROPERTIES
        PREFIX ""
)

target_compile_options(data_wrapper PRIVATE
        -Xlinker -export-dynamic
)

target_link_options(data_wrapper PRIVATE
        $<$<CONFIG:Release>:-s>
)

add_subdirectory(flow_func_wrapper)

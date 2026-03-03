# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(eager_style_graph_builder_base_headers INTERFACE)
target_include_directories(eager_style_graph_builder_base_headers INTERFACE
        ${AIR_CODE_DIR}/inc/external/ge/eager_style_graph_builder/c
        ${AIR_CODE_DIR}/inc/external/ge/eager_style_graph_builder/cpp
        )

add_library(eager_style_graph_builder_base_inner_headers INTERFACE)
target_include_directories(eager_style_graph_builder_base_inner_headers INTERFACE
        ${AIR_CODE_DIR}/inc/framework/graph/eager_style_graph_builder
        )

file(GLOB_RECURSE EAGER_STYLE_GRAPH_BUILDER CONFIGURE_DEPENDS "*.cc")

add_library(eager_style_graph_builder_base SHARED ${EAGER_STYLE_GRAPH_BUILDER})

target_compile_options(eager_style_graph_builder_base PRIVATE ${AIR_COMMON_DYNAMIC_COMPILE_OPTION})
target_link_libraries(eager_style_graph_builder_base
        PRIVATE
        intf_pub
        slog_headers
        unified_dlog
        air_headers
        error_manager
        mmpa_headers
        graph
        graph_base
        aihac_symbolizer
        c_sec
        PUBLIC
        eager_style_graph_builder_base_headers
        eager_style_graph_builder_base_inner_headers
        )

add_library(eager_style_graph_builder_base_static STATIC ${EAGER_STYLE_GRAPH_BUILDER})

target_compile_options(eager_style_graph_builder_base_static PRIVATE ${AIR_COMMON_COMPILE_OPTION})
target_link_libraries(eager_style_graph_builder_base_static
        PRIVATE
        intf_pub
        slog_headers
        unified_dlog
        air_headers
        error_manager
        mmpa_headers
        graph
        graph_base
        aihac_symbolizer
        c_sec
        PUBLIC
        eager_style_graph_builder_base_headers
        eager_style_graph_builder_base_inner_headers
        )
install(TARGETS eager_style_graph_builder_base_static OPTIONAL
        ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR})
install(TARGETS eager_style_graph_builder_base OPTIONAL
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR})

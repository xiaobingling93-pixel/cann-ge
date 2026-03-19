# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE EAGER_GRAPH_BUILDER CONFIGURE_DEPENDS "*.cpp")

# Use different target name to avoid conflict with tests/framework/eager_style_graph_builder
add_library(autofuse_eager_graph_builder STATIC ${EAGER_GRAPH_BUILDER})

target_compile_options(autofuse_eager_graph_builder
    PRIVATE
        -Werror=format
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal
        #        -Werror
        -Wno-subobject-linkage
        )

target_link_libraries(autofuse_eager_graph_builder PRIVATE
        intf_llt_pub
        slog_headers
        mmpa_headers
        metadef_headers
        c_sec
        -ldl
)

add_library(autofuse_eager_graph_builder_lib INTERFACE)
target_link_libraries(autofuse_eager_graph_builder_lib INTERFACE autofuse_eager_graph_builder)
target_include_directories(autofuse_eager_graph_builder_lib INTERFACE ${CMAKE_CURRENT_LIST_DIR})

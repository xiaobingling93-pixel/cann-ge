# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE EASY_GRAPH_SRCS  src/*.cc)

add_library(easy_graph_b STATIC ${EASY_GRAPH_SRCS})

target_include_directories(easy_graph_b
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include  # 公共包含目录
    PRIVATE src                                # 私有包含目录
)
# 添加其他依赖库（如果有）
target_link_libraries(easy_graph_b PUBLIC intf_llt_pub)
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cc" "*.CC" "*.cpp" "*.CPP")

add_library(share_graph STATIC ${SOURCES})

target_include_directories(share_graph
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include ${AIR_CODE_DIR}/inc
)

target_link_libraries(share_graph PUBLIC intf_llt_pub)

set_target_properties(share_graph PROPERTIES CXX_STANDARD 17)
target_link_libraries(share_graph PUBLIC 
        graph
        graph_base
        ascir
        aihac_ir
        aihac_ir_register
        ascgen_common
        asc_slog_stub
        autofuse_runtime_stub
        error_manager)

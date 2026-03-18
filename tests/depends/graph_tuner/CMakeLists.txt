# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

FILE(GLOB_RECURSE STUB_SOURCE "${CMAKE_CURRENT_LIST_DIR}/src/" *.cc)
add_library(graphtuner_executor SHARED ${STUB_SOURCE})
target_compile_options(graphtuner_executor PRIVATE -g -O0)
target_include_directories(graphtuner_executor PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/src
    ${AIR_CODE_DIR}
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/external/ge
    ${CMAKE_BINARY_DIR}
    ${AIR_CODE_DIR}/runtime/v2
)
target_link_libraries(graphtuner_executor
    PRIVATE
        intf_llt_pub
        metadef_headers
        c_sec_headers
        slog_headers
)

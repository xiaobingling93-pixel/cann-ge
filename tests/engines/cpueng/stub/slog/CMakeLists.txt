# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(LOCAL_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/slog_stub.cpp
)

add_library(slog_aicpu_stub SHARED
    ${LOCAL_SRC_FILES}
)

target_compile_options(slog_aicpu_stub PRIVATE
    -fPIC
)

target_include_directories(slog_aicpu_stub PRIVATE
    ${ASCEND_INSTALL_PATH}/pkg_inc
    ${ASCEND_INSTALL_PATH}/pkg_inc/base
)

target_link_libraries(slog_aicpu_stub
    PRIVATE
        intf_llt_pub
    PUBLIC
        unified_dlog_headers
)

stub_module(unified_dlog slog_aicpu_stub)

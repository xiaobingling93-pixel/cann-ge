# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set(TRACE_SRC_CPP
        atrace_api_stub.cc)

add_library(atrace_stub SHARED
        ${TRACE_SRC_CPP})

target_include_directories(atrace_stub PRIVATE
        ${ASCEND_INSTALL_PATH}/x86_64-linux/pkg_inc)

target_compile_options(atrace_stub PRIVATE
        -Dgoogle=ascend_private)

target_link_libraries(atrace_stub
    PRIVATE
        intf_llt_pub
        c_sec
    PUBLIC
        atrace_headers
)

stub_module(atrace_share atrace_stub)
stub_module(atrace atrace_stub)

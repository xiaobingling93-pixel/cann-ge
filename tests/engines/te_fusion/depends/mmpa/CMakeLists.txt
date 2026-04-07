# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE SRCS RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "src/mmpa_stub.cc"
)

include_directories(${AIR_CODE_DIR}/inc)
include_directories(${AIR_CODE_DIR}/inc/external)
include_directories(${AIR_CODE_DIR}/inc/framework)

add_library(mmpa_stub SHARED ${SRCS})

target_link_libraries(mmpa_stub
        PRIVATE
                intf_llt_pub
                msprof_headers
                runtime_headers
                metadef_headers
                c_sec -ldl
        PUBLIC
                mmpa_headers
)

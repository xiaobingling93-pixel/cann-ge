# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(hccl_stub SHARED src/hccl_stub.cc)

target_include_directories(hccl_stub PRIVATE
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/inc
)

target_link_options(hccl_stub PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        )

target_link_libraries(hccl_stub
    PRIVATE
        intf_pub
        slog_headers
        cce_headers
        runtime_headers
        msprof_headers
        ${AIR_COMMON_LINK_OPTION}
    PUBLIC
        hccl_headers
)

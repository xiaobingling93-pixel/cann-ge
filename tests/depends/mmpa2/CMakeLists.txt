# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(mmpa_stub2 SHARED src/mmpa_stub.cc)
target_compile_options(mmpa_stub PRIVATE -g)
target_link_options(mmpa_stub2 PRIVATE
        -Wl,-Bsymbolic
        )

target_include_directories(mmpa_stub2 PRIVATE
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${AIR_CODE_DIR}/tests/depends/mmpa/src
)
target_link_libraries(mmpa_stub2
    PRIVATE
        c_sec_headers
        mmpa_headers
        runtime_headers
        ascendcl_headers
        msprof_headers
        air_headers
        slog_headers
        metadef_headers
        intf_pub
        slog_stub
        c_sec
        -ldl
    PUBLIC
        mmpa_headers
)

stub_module(static_mmpa2 mmpa_stub2)

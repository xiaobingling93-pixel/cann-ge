# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE BIN_SRCS CONFIGURE_DEPENDS "kernel/*.cc")

add_executable(ge_runtime_bin ${BIN_SRCS} ${FAKER_SRCS})

target_link_libraries(ge_runtime_bin PUBLIC intf_llt_pub)

target_link_libraries(ge_runtime_bin PUBLIC ge_runtime_v2 ge_graph_dsl ge_runtime_stub)

target_compile_options(ge_runtime_bin PRIVATE -O2)

target_include_directories(ge_runtime_bin PUBLIC
        ${AIR_CODE_DIR}/tests/ge/ut/ge/runtime/fast_v2
        )




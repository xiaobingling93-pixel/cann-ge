# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cc" "*.CC" "*.cpp" "*.CPP" "*.c++")
# ---- Target : stub Host engine ----
add_library(ge_running_env STATIC ${SOURCES})
add_dependencies(ge_running_env nnengine local_engine)

target_include_directories(ge_running_env
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/inc/graph_metadef
        )

target_compile_definitions(ge_running_env PRIVATE
        google=ascend_private
        FMK_SUPPORT_DUMP
        AIR_CODE_DIR="${AIR_CODE_DIR}"
        ENGINE_PATH="$<TARGET_FILE:nnengine>"
        OPS_KERNEL_LIB_PATH="$<TARGET_FILE:local_engine>"
        FUNC_VISIBILITY
        )

target_compile_options(ge_running_env PRIVATE
        -g --coverage -fprofile-arcs -ftest-coverage
        -Werror=format
        )

target_link_libraries(ge_running_env PUBLIC
        intf_pub
        metadef_headers
        runtime_headers
        cce_headers
        slog_headers
        ge_running_env_inc graphengine_inc ${STUB_LIBS}
        )
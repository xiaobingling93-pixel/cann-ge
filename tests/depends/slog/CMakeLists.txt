# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
file(GLOB_RECURSE SLOG_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "src/slog_stub.cc")
file(GLOB_RECURSE UNIFIED_DLOG_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "src/unified_dlog_stub.cc")
file(GLOB_RECURSE ASCENDALOG_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "src/ascendalog_stub.cc")

add_library(slog_common SHARED ${SLOG_SRC_FILES})
target_compile_options(slog_common PRIVATE -g -O0)
target_link_libraries(slog_common PRIVATE intf_pub
        PUBLIC slog_headers
        )

add_library(ascendalog SHARED ${ASCENDALOG_SRC_FILES})
target_compile_options(ascendalog PRIVATE -g -O0)
target_link_libraries(ascendalog PRIVATE intf_pub
        PUBLIC slog_headers
        slog_common
        )

# Only create unified_dlog target if it doesn't already exist (e.g., from CANN package)
if(NOT TARGET unified_dlog)
add_library(unified_dlog SHARED ${UNIFIED_DLOG_SRC_FILES})
target_compile_options(unified_dlog PRIVATE -g -O0)
target_link_libraries(unified_dlog PRIVATE
        PUBLIC slog_headers
        slog_common
        )
endif()

stub_module(slog ascendalog)
stub_module(slog_stub ascendalog)
stub_module(parser_slog_stub ascendalog)

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (ENABLE_GE_UT OR ENABLE_GE_ST OR ENABLE_METADEF_UT OR ENABLE_METADEF_ST OR ENABLE_METADEF_COV)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -fsanitize=address -fsanitize=leak -fsanitize-recover=address")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address -fsanitize=leak -fsanitize-recover=address")
endif()

stub_module(slog slog_stub)
stub_module(static_mmpa mmpa_stub)
stub_module(mmpa mmpa_stub)
stub_module(platform platform_stub)
stub_module(runtime runtime_stub)

add_subdirectory(ut)

if (ENABLE_BENCHMARK)
    find_package(benchmark CONFIG REQUIRED)
    add_subdirectory(benchmark)
endif()


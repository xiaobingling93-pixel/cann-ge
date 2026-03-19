# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

find_package(benchmark CONFIG REQUIRED)

file(GLOB_RECURSE BENCHMARK_SRCS CONFIGURE_DEPENDS "*.cc")

add_executable(ge_runtime_benchmark ${BENCHMARK_SRCS} ${FAKER_SRCS})

target_link_libraries(ge_runtime_benchmark PUBLIC intf_llt_pub)

target_include_directories(ge_runtime_benchmark PRIVATE
        ${AIR_CODE_DIR}/tests/ge/ut/ge/runtime/fast_v2
        ${AIR_CODE_DIR}/runtime/v2
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}/inc/external
        ./runtime/inc
        )

target_link_libraries(ge_runtime_benchmark PUBLIC
        benchmark::benchmark gert gert_op_impl ge_graph_dsl ge_runtime_stub
        -Wl,--no-as-needed
        profiler_stub
        )

target_compile_options(ge_runtime_benchmark PRIVATE ${AIR_COMMON_COMPILE_OPTION})

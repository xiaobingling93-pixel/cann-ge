# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(BENCHMARK_TEST  "compute_graph_base_unittest.cc")

add_executable(fast_graph_benchmark ${BENCHMARK_TEST})

target_include_directories(fast_graph_benchmark PRIVATE
        ${AIR_CODE_DIR}/tests/ge/ut/ge/runtime/fast_v2
        ${AIR_CODE_DIR}/runtime/v2
        ${AIR_CODE_DIR}/inc/framework
        ${METADEF_DIR}/inc
        ${METADEF_DIR}/inc/external
        ${METADEF_DIR}/third_party/inc
        ${METADEF_DIR}/third_party/inc/external
        ${CMAKE_BINARY_DIR}
        ${CMAKE_BINARY_DIR}/proto/metadef_protos
        ${CMAKE_BINARY_DIR}/proto/ge
        ${CMAKE_BINARY_DIR}/proto/ge/proto
        )

target_compile_definitions(fast_graph_benchmark PRIVATE
        google=ascend_private
        FUNC_VISIBILITY
        )

set_target_properties(fast_graph_benchmark PROPERTIES CXX_STANDARD 17)

target_compile_options(fast_graph_benchmark PRIVATE -O2 -std=c++17)

target_link_libraries(fast_graph_benchmark  PRIVATE benchmark::benchmark
        intf_llt_pub
        metadef_headers
        -Wl,--no-as-needed
        graph
        graph_base
        c_sec
        error_manager
        slog
        ascend_protobuf
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
)

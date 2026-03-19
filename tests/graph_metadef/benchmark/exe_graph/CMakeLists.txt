# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE EXE_GRAPH_SRCS CONFIGURE_DEPENDS "*.cc")

add_executable(exec_graph_benchmark ${EXE_GRAPH_SRCS})

target_link_libraries(exec_graph_benchmark PRIVATE intf_llt_pub)

target_include_directories(exec_graph_benchmark PRIVATE
        ${AIR_CODE_DIR}/tests/ge/ut/ge/runtime/fast_v2
        ${AIR_CODE_DIR}/runtime/v2
        ${AIR_CODE_DIR}/inc/framework
        ${METADEF_DIR}/exe_graph
        ${CMAKE_BINARY_DIR}
        ${CMAKE_BINARY_DIR}/proto/metadef_protos
        )

target_link_libraries(exec_graph_benchmark PRIVATE benchmark::benchmark exe_graph lowering error_manager
        slog_stub ascend_protobuf c_sec mmpa_stub -lrt -ldl
        metadef_headers
        graph
        )

set_target_properties(exec_graph_benchmark PROPERTIES CXX_STANDARD 17)
target_compile_options(exec_graph_benchmark PRIVATE -O2)

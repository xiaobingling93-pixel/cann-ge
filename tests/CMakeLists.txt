# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(METADEF_DIR ${AIR_CODE_DIR}/base/metadef)
set(GE_METADEF_TEST_DIR ${AIR_CODE_DIR}/tests/graph_metadef)
set(AIR_COMMON_COMPILE_OPTION ${AIR_COMMON_COMPILE_OPTION} -fno-access-control)

include(cmake/intf_llt_pub.cmake)
include(dflow/udf/cmake/third_party/boost.cmake)
include(dflow/udf/cmake/third_party/mockcpp.cmake)
add_subdirectory(depends/slog)
add_subdirectory(depends/mmpa)
add_subdirectory(depends/mmpa2)
add_subdirectory(depends/runtime)
add_subdirectory(depends/hccl)
add_subdirectory(depends/profiler)
add_subdirectory(depends/op_stub)
add_subdirectory(depends/graph_tuner)
add_subdirectory(depends/aoe)
add_subdirectory(depends/platform)
add_subdirectory(depends/llm_datadist)
add_subdirectory(depends/python)
add_subdirectory(depends/aihacb_autofusion)
add_subdirectory(depends/ascendcl)
add_subdirectory(framework)
add_subdirectory(benchmark)
add_subdirectory(ge)
add_subdirectory(bin)
add_subdirectory(parser)
add_subdirectory(graph_metadef)
add_subdirectory(dflow)
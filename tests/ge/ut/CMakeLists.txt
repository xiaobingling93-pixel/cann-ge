# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (ENABLE_GE_COV)
    set(COVERAGE_COMPILER_FLAGS "-g --coverage -fprofile-arcs -fPIC -O0 -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${COVERAGE_COMPILER_FLAGS}")
endif()

add_subdirectory(common/graph)
add_subdirectory(ge)
add_subdirectory(ge/runtime/fast_v2)
add_subdirectory(ge/runtime/fast_v2/engine/dvpp/dvpp_rtkernel)
add_subdirectory(sc_check)


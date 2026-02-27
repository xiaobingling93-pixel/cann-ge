# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (ENABLE_LLT_COV)
    set(COVERAGE_COMPILER_FLAGS "-g --coverage -fprofile-arcs -fPIC -O0 -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${COVERAGE_COMPILER_FLAGS}")
endif()

add_subdirectory(stub)

if (ENABLE_UT)
    add_subdirectory(ut)
    add_custom_target(aicpu_ut DEPENDS tf_engine_utest cpu_engine_utest host_engine_utest)
    add_custom_command(TARGET aicpu_ut
      COMMENT "=====run cpu test end====="
    )
endif()

if (ENABLE_ST)
    add_subdirectory(st)
    add_custom_target(aicpu_st DEPENDS tf_engine_stest cpu_engine_stest host_engine_stest)
    add_custom_command(TARGET aicpu_st
      COMMENT "=====run cpu test end====="
    )
endif()

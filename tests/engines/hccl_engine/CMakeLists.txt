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

set(ASCEND_SDK_PACKAGE_PATH "${ASCEND_INSTALL_PATH}")

add_custom_target(hcce_ut DEPENDS hccl_engine_utest)
add_custom_command(TARGET hcce_ut
  COMMENT "=====run hccl engine test end====="
)

add_subdirectory(ut)

add_custom_target(hcce_st DEPENDS hccl_engine_stest)
add_custom_command(TARGET hcce_st
  COMMENT "=====run hccl engine test end====="
)

add_subdirectory(st)
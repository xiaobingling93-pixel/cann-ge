# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(UDF_SRC_PATH ${DFLOW_CODE_DIR}/udf)

set(LOCAL_STUB_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/../stub/slog_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../stub/mmpa_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/stub/udf_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/udf_utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/stub/flow_func_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/stub/flow_func_multi_in_out.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/stub/flow_func_call_nn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/stub/multi_flow_func_stub.cpp
)

set(BUILT_IN_LOCAL_STUB_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/../builtin_stub/hal_stub.cpp
)

add_subdirectory(testcase/built_in/)
add_subdirectory(testcase/udf_executor/)

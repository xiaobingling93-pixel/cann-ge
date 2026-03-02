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
        ${CMAKE_CURRENT_SOURCE_DIR}/stub/drv_common_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/stub/drv_buf_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/stub/drv_queue_stub.cpp
)

add_subdirectory(testcase/built_in/)
add_subdirectory(testcase/udf_executor/)
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(v1)
add_subdirectory(v2)
add_subdirectory(stub)
if(NOT DEFINED BUILD_COMPONENT AND NOT ENABLE_PKG AND NOT ENABLE_LLT_PKG AND NOT CMAKE_BUILD_TYPE MATCHES GCOV AND NOT ENABLE_GE_DT AND NOT ENABLE_GE_C_LLT)
    add_subdirectory(c)
endif()
add_subdirectory(ops)
add_subdirectory(om2)

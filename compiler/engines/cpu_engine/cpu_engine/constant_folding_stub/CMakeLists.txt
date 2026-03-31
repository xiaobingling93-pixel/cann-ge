# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
 
add_library(constant_folding_ops_stub SHARED
    constant_folding_ops_stub.cpp
)
 
target_include_directories(constant_folding_ops_stub PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
 
set_target_properties(constant_folding_ops_stub PROPERTIES
    OUTPUT_NAME "constant_folding_ops"
)

target_link_libraries(constant_folding_ops_stub PRIVATE $<BUILD_INTERFACE:intf_pub>)
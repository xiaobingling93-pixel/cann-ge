# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Use different target name to avoid conflict with tests/depends/runtime
add_library(autofuse_runtime_stub SHARED src/runtime_stub.cc)
target_compile_options(autofuse_runtime_stub PRIVATE -g)
target_compile_definitions(autofuse_runtime_stub PRIVATE
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)
target_include_directories(autofuse_runtime_stub PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src)
target_link_libraries(autofuse_runtime_stub
        PRIVATE
        intf_llt_pub
        c_sec
        PUBLIC
        runtime_headers
)

stub_module(runtime autofuse_runtime_stub)
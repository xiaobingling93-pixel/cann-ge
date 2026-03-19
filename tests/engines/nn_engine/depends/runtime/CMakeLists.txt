# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(FE_SRC_CPP
        src/runtime_stub.cc)

# compile for fe_st
add_library(runtime_stub_fe SHARED
        ${FE_SRC_CPP})

#target_include_directories(runtime_stub_fe PRIVATE
#        )

# add it for slog compile at 12.16
target_compile_definitions(runtime_stub_fe PRIVATE
        LOG_CPP
        )

target_compile_options(runtime_stub_fe PRIVATE
        -Dgoogle=ascend_private)

target_link_libraries(runtime_stub_fe 
        PRIVATE
                intf_llt_pub
                msprof_headers
                c_sec
        PUBLIC
                runtime_headers)

install(TARGETS runtime_stub_fe OPTIONAL
        EXPORT runtime_stub_fe-targets
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR})

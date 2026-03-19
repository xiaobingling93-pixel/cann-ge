# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

project(cann_kb_stub)

set(FE_SRC_CPP
        src/cann_kb_api_stub.cc)

# compile for fe_st
add_library(cann_kb_stub SHARED
        ${FE_SRC_CPP})

target_include_directories(cann_kb_stub PRIVATE
        ${ASCEND_INSTALL_PATH}/include/experiment/opat
        ${FE_ROOT_PATH}/utils
        ${AIR_CODE_DIR}/inc
        ${ASCEND_INSTALL_PATH}/pkg_inc
        ${ASCEND_INSTALL_PATH}/pkg_inc/base
        )

# add it for slog compile at 12.16
target_compile_definitions(cann_kb_stub PRIVATE
        LOG_CPP
        )

target_compile_options(cann_kb_stub PRIVATE
        -Dgoogle=ascend_private)

target_link_libraries(cann_kb_stub
        PRIVATE
                intf_llt_pub
                c_sec
                air_headers
                metadef_headers)

set(INSTALL_BASE_DIR "")
set(INSTALL_INCLUDE_DIR include)
set(INSTALL_LIBRARY_DIR lib)
set(INSTALL_CONFIG_DIR  lib/cmake)

install(TARGETS cann_kb_stub OPTIONAL
        EXPORT cann_kb_stub-targets
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR})

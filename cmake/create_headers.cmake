# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(hccl_headers INTERFACE)
target_include_directories(hccl_headers INTERFACE
        ${ASCEND_INSTALL_PATH}/include
        ${ASCEND_INSTALL_PATH}/include/hccl
        ${ASCEND_INSTALL_PATH}/pkg_inc
        ${ASCEND_INSTALL_PATH}/pkg_inc/hccl
)

add_library(datagw_headers INTERFACE)
target_include_directories(datagw_headers INTERFACE
        ${ASCEND_INSTALL_PATH}/pkg_inc
        ${ASCEND_INSTALL_PATH}/pkg_inc/aicpu
        ${ASCEND_INSTALL_PATH}/pkg_inc/aicpu/queue_schedule
        ${ASCEND_INSTALL_PATH}/pkg_inc/aicpu/aicpu_schedule
        ${ASCEND_INSTALL_PATH}/pkg_inc/aicpu/tsd
        ${ASCEND_INSTALL_PATH}/pkg_inc/aicpu/common
)

add_library(slog_headers INTERFACE)
target_include_directories(slog_headers INTERFACE
        ${ASCEND_INSTALL_PATH}/pkg_inc
        ${ASCEND_INSTALL_PATH}/pkg_inc/base
)

add_library(runtime_headers INTERFACE)
target_include_directories(runtime_headers INTERFACE
        ${ASCEND_INSTALL_PATH}/pkg_inc
        ${ASCEND_INSTALL_PATH}/pkg_inc/runtime
        ${ASCEND_INSTALL_PATH}/pkg_inc/runtime/runtime
        ${ASCEND_INSTALL_PATH}/include
        ${ASCEND_INSTALL_PATH}/include/acl
        ${ASCEND_INSTALL_PATH}/include/acl/error_codes
)

add_library(adump_headers INTERFACE)
target_include_directories(adump_headers INTERFACE
        ${ASCEND_INSTALL_PATH}/pkg_inc
        ${ASCEND_INSTALL_PATH}/pkg_inc/dump
)

add_library(msprof_headers INTERFACE)
target_include_directories(msprof_headers INTERFACE
        ${ASCEND_INSTALL_PATH}/pkg_inc
        ${ASCEND_INSTALL_PATH}/pkg_inc/profiling
        ${ASCEND_INSTALL_PATH}/pkg_inc/toolchain
)

add_library(mmpa_headers INTERFACE)
target_include_directories(mmpa_headers INTERFACE
        ${ASCEND_INSTALL_PATH}/include/
        ${ASCEND_INSTALL_PATH}/include/mmpa
        ${ASCEND_INSTALL_PATH}/include/mmpa/sub_inc
)

add_library(ascendcl_headers INTERFACE)
target_include_directories(ascendcl_headers INTERFACE
        ${ASCEND_INSTALL_PATH}/include
        ${ASCEND_INSTALL_PATH}/include/acl
)

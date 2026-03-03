# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# todo 整改后，不再因为XX_DT导致重新configure cmake工程，测试工程的打桩策略保持一致
# todo 整改后，gcov模式下不需要打桩的package,挪到`find_common_cann_packages`中
if (GE_DT)
    message(STATUS "GE DT mode")
    include(cmake/create_headers.cmake)
    find_package(cce MODULE REQUIRED)
    find_package(ascend_hal MODULE REQUIRED)

    set(METADEF_DIR ${CMAKE_CURRENT_LIST_DIR}/../base/metadef)
    if (NOT BUILD_METADEF)
        find_package(metadef MODULE REQUIRED)
        set(GE_METADEF_DIR ${CMAKE_CURRENT_LIST_DIR}/../graph_metadef)
        set(GE_METADEF_INC_DIR ${CMAKE_CURRENT_LIST_DIR}/../inc/graph_metadef)
    endif ()
    set(PARSER_DIR ${CMAKE_CURRENT_LIST_DIR}/parser)
elseif (ENGINE_DT OR GE_C_DT)
    message(STATUS "ENGINE OR GE_C DT mode")
    add_library(licctrl_headers INTERFACE)
    target_include_directories(licctrl_headers INTERFACE
        ${ASCEND_INSTALL_PATH}/include/experiment
        ${ASCEND_INSTALL_PATH}/include/experiment/licctrl
    )

    include(cmake/function.cmake)
    find_package_if_target_not_exists(slog MODULE REQUIRED)
    find_package_if_target_not_exists(unified_dlog MODULE REQUIRED)
    find_package_if_target_not_exists(atrace MODULE REQUIRED)
    find_package_if_target_not_exists(platform MODULE REQUIRED)
    find_package_if_target_not_exists(runtime MODULE REQUIRED)
    find_package_if_target_not_exists(datagw MODULE REQUIRED)
    find_package_if_target_not_exists(mmpa MODULE REQUIRED)
    find_package_if_target_not_exists(msprof MODULE REQUIRED)
    if (NOT BUILD_OPEN_PROJECT AND NOT ENABLE_OPEN_SRC)
        find_package_if_target_not_exists(opcompiler MODULE REQUIRED)
        find_package_if_target_not_exists(graph_tuner MODULE REQUIRED)
        find_package_if_target_not_exists(opat MODULE REQUIRED)
        find_package_if_target_not_exists(hccl MODULE REQUIRED)
        find_package_if_target_not_exists(runtime_static MODULE REQUIRED)
    endif ()

    find_package_if_target_not_exists(ascend_hal MODULE REQUIRED)
    find_package_if_target_not_exists(adump MODULE REQUIRED)
    find_package_if_target_not_exists(cce MODULE REQUIRED)
    find_package_if_target_not_exists(aicpu MODULE REQUIRED)
    find_package_if_target_not_exists(ascendcl MODULE REQUIRED)

    # 使用medadef发布包编译
    find_package(metadef MODULE REQUIRED)

    set(METADEF_DIR ${CMAKE_CURRENT_LIST_DIR}/../base/metadef)
    set(PARSER_DIR ${CMAKE_CURRENT_LIST_DIR}/parser)
elseif (ENABLE_OPEN_SRC AND RUN_TEST)
    find_package_if_target_not_exists(mmpa MODULE REQUIRED)
    find_package(metadef MODULE REQUIRED)
    find_package_if_target_not_exists(platform MODULE REQUIRED)
else ()
    message(WARNING "GCOV general mode")
endif ()

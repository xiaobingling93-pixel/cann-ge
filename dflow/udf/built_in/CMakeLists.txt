# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

###for built_in_flowfunc so
file(GLOB_RECURSE SRC_LIST
    *.cpp
)
##for so inc path
set(built_in_flowfunc_so_inc_path
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc/external
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc/external/flow_func
)

add_library(built_in_flowfunc SHARED
    ${SRC_LIST}
)
target_include_directories(built_in_flowfunc PRIVATE
    ${built_in_flowfunc_so_inc_path}
)

target_compile_options(built_in_flowfunc PRIVATE
    -ftrapv
    -O2
    -fvisibility-inlines-hidden
    -fvisibility=hidden
    -Werror
    -Wextra
    -fstack-protector-all
    -fPIC
    -Wfloat-equal
    -fno-common
)

target_link_libraries(built_in_flowfunc PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:slog_headers>
    $<BUILD_INTERFACE:mmpa_headers>
    $<BUILD_INTERFACE:ascend_hal_headers>
    $<BUILD_INTERFACE:hccl_headers>
    PUBLIC c_sec
    flow_func
    ascend_hal_stub
)

set(INSTALL_LIBRARY_DIR lib)

install(TARGETS built_in_flowfunc OPTIONAL
    EXPORT built_in_flowfunc-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    RUNTIME DESTINATION ${INSTALL_LIBRARY_DIR}
)

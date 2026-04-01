# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

############ libacl_op_compiler.so ############
add_library(acl_op_compiler SHARED
    single_op/op_compiler.cpp
    single_op/compile/local_compiler.cpp
    single_op/compile/op_compile_processor.cpp
    single_op/compile/init_callback_register.cpp
    ../common/log_inner.cpp
    ../common/json_parser.cpp
    ../common/prof_api_reg.cpp
    ../utils/attr_utils.cpp
    ../utils/array_utils.cpp
    ../utils/hash_utils.cpp
)

target_include_directories(acl_op_compiler PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${CMAKE_CURRENT_LIST_DIR}/../common
    ${CMAKE_CURRENT_LIST_DIR}/../utils
    ${CMAKE_CURRENT_LIST_DIR}/../acl_model
    ${CMAKE_CURRENT_LIST_DIR}/../acl_op_executor
    ${CMAKE_CURRENT_LIST_DIR}/compile
    ${CMAKE_CURRENT_LIST_DIR}/utils
    ${CMAKE_CURRENT_LIST_DIR}/single_op
    ${CMAKE_CURRENT_LIST_DIR}/../acl_op_executor/single_op
    ${CMAKE_CURRENT_LIST_DIR}/../acl_op_executor/single_op/compile

    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/external/acl
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${AIR_CODE_DIR}/inc/graph_metadef/external
    ${AIR_CODE_DIR}/inc/framework
    ${AIR_CODE_DIR}/base/metadef/pkg_inc

    ${ASCEND_INSTALL_PATH}/pkg_inc/base
    ${ASCEND_INSTALL_PATH}/include/prof
    ${ASCEND_INSTALL_PATH}/include/external

    ${TOP_DIR}/air/inc/external
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/include/external/acl
    ${TOP_DIR}/runtime/include/external/acl/error_codes
)

target_compile_options(acl_op_compiler PRIVATE
    -ftrapv
    -O2
    -Werror
    -Wno-deprecated-declarations
    -fvisibility=hidden
    -fno-common
)

target_compile_definitions(acl_op_compiler PRIVATE
    OS_TYPE=0
    FUNC_VISIBILITY
    google=ascend_private
    $<$<STREQUAL:${ENABLE_ACL_UT},ON>:ONLY_ENABLE_ACL_UT>
)

target_link_options(acl_op_compiler PRIVATE
    -rdynamic
    -Wl,-Bsymbolic
    -Wl,--exclude-libs,ALL
)

target_link_libraries(acl_op_compiler PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:slog_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:msprof_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:mmpa_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:runtime_headers>>
    -Wl,--no-as-needed
    acl_op_executor
    acl_op_executor_impl
    acl_rt
    c_sec
    unified_dlog
    runtime
    mmpa
    profapi_share
    gert
    graph_base
    graph
    ge_compiler
    ge_runner
    ge_common_base
    error_manager
    platform
    -Wl,--as-needed
    -ldl
     $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    json
)

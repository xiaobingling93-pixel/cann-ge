# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

############ libacl_op_executor_impl.so ############
add_library(acl_op_executor_impl SHARED
    single_op/op.cpp
    single_op/op_executor.cpp
    single_op/op_model_cache.cpp
    single_op/ge_tensor_cache.cpp
    single_op/acl_op_resource_manager.cpp
    single_op/executor/init_callback_register.cpp
    single_op/shape_range_utils.cpp
    single_op/builtin/cast_op.cpp
    single_op/op_model_parser.cpp
    single_op/compile/op_kernel_selector.cpp
    single_op/compile/op_kernel_registry.cpp
    single_op/compile/op_compiler.cpp
    single_op/compile/op_compile_service.cpp
    single_op/executor/stream_executor.cpp
    single_op/executor/resource_manager.cpp
    single_op/executor/op_task.cpp
    types/op_attr.cpp
    types/op_model.cpp
    types/acl_op_inner.cpp
    ../common/common_inner.cpp
    ../common/log_inner.cpp
    ../common/json_parser.cpp
    ../utils/attr_utils.cpp
    ../utils/array_utils.cpp
    ../utils/hash_utils.cpp
    ../utils/string_utils.cpp
    ../utils/file_utils.cpp
    ../common/prof_api_reg.cpp
)

target_include_directories(acl_op_executor_impl PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${CMAKE_CURRENT_LIST_DIR}/../common
    ${CMAKE_CURRENT_LIST_DIR}/../util

    ${CMAKE_CURRENT_LIST_DIR}/../acl_model

    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/external/acl
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${AIR_CODE_DIR}/inc/graph_metadef/external
    ${AIR_CODE_DIR}/inc/framework
    ${AIR_CODE_DIR}/base/metadef/pkg_inc
    ${ASCEND_INSTALL_PATH}/pkg_inc/runtime
    ${ASCEND_INSTALL_PATH}/include/external
    ${ASCEND_INSTALL_PATH}/pkg_inc/base
    ${ASCEND_INSTALL_PATH}/include/prof


    ${TOP_DIR}/air/inc/external
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/include/external/acl
    ${TOP_DIR}/runtime/include/external/acl/error_codes
)

target_compile_options(acl_op_executor_impl PRIVATE
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-fvisibility=hidden -O2 -fno-common>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:/utf-8 /Od>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

target_compile_definitions(acl_op_executor_impl PRIVATE
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    FUNC_VISIBILITY
    google=ascend_private
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    $<$<STREQUAL:${ENABLE_ACL_UT},ON>:ONLY_ENABLE_ACL_UT>    
)

target_link_options(acl_op_executor_impl PRIVATE
    -rdynamic
    -Wl,-Bsymbolic
    -Wl,--exclude-libs,ALL
)

target_link_directories(acl_op_executor_impl PRIVATE
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Android>:${TOP_DIR}/prebuilts/clang/linux-x86/aarch64/android-ndk-r21/sysroot/usr/lib/aarch64-linux-android/29>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Android>:${TOP_DIR}/build/prebuilts/clang/linux-x86/aarch64/android-ndk-r21/sysroot/usr/lib/aarch64-linux-android/29>
)

target_link_libraries(acl_op_executor_impl PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:slog_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:msprof_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:mmpa_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:runtime_headers>>
    -Wl,--whole-archive
    -Wl,--no-whole-archive
    -Wl,--no-as-needed
    acl_rt
    acl_mdl
    acl_mdl_impl
    c_sec
    unified_dlog
    mmpa
    profapi_share
    runtime
    error_manager
    awatchdog_headers
    awatchdog_share
    json
    
    gert
    ge_executor_shared
    ge_common_base
    graph_base
    graph

    -Wl,--as-needed
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Android>:-llog>
)

############ libacl_op_executor.so ############
add_library(acl_op_executor SHARED
    single_op/acl_op_executor.cpp
)

target_include_directories(acl_op_executor PRIVATE
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/external/acl

    ${ASCEND_INSTALL_PATH}/include
    ${ASCEND_INSTALL_PATH}/include/acl

    ${TOP_DIR}/air/inc/external
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/include/external/acl
)

target_compile_options(acl_op_executor PRIVATE
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-fvisibility=hidden -O2 -fno-common>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:/utf-8 /Od>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

target_compile_definitions(acl_op_executor PRIVATE
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    FUNC_VISIBILITY
    google=ascend_private
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    $<$<STREQUAL:${ENABLE_ACL_UT},ON>:ONLY_ENABLE_ACL_UT>
)

target_link_options(acl_op_executor PRIVATE
    -rdynamic
    -Wl,-Bsymbolic
    -Wl,--exclude-libs,ALL
)

target_link_directories(acl_op_executor PRIVATE
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Android>:${TOP_DIR}/prebuilts/clang/linux-x86/aarch64/android-ndk-r21/sysroot/usr/lib/aarch64-linux-android/29>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Android>:${TOP_DIR}/build/prebuilts/clang/linux-x86/aarch64/android-ndk-r21/sysroot/usr/lib/aarch64-linux-android/29>
)

target_link_libraries(acl_op_executor PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:slog_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:msprof_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:mmpa_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:runtime_headers>>
    -Wl,--whole-archive
    -Wl,--no-whole-archive
    -Wl,--no-as-needed
    acl_op_executor_impl

    -Wl,--as-needed
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Android>:-llog>
)

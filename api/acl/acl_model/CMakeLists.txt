# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

############ libacl_mdl_impl.so ############
add_library(acl_mdl_impl SHARED
    model/model.cpp
    model/acl_aipp.cpp
    model/aipp_param_check.cpp
    model/model_config.cpp
    model/acl_resource_manager.cpp
    model/init_callback_register.cpp
    types/tensor_desc_internal.cpp
    ../common/json_parser.cpp
    ../common/log_inner.cpp
    ../utils/string_utils.cpp
    ../common/prof_api_reg.cpp
)

target_include_directories(acl_mdl_impl PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${CMAKE_CURRENT_LIST_DIR}/../common
    ${CMAKE_CURRENT_LIST_DIR}/../utils

    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/external/acl
    ${AIR_CODE_DIR}/inc/graph_metadef

    ${ASCEND_INSTALL_PATH}/pkg_inc/base
    ${ASCEND_INSTALL_PATH}/include/prof

    ${AIR_CODE_DIR}/inc/framework
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${AIR_CODE_DIR}/inc/graph_metadef/external
    ${ASCEND_INSTALL_PATH}/include
    ${ASCEND_INSTALL_PATH}/include/acl
    ${AIR_CODE_DIR}/base/metadef/pkg_inc
    ${ASCEND_INSTALL_PATH}/pkg_inc
    ${ASCEND_INSTALL_PATH}/pkg_inc/runtime
    ${ASCEND_INSTALL_PATH}/pkg_inc/runtime/runtime

    ${TOP_DIR}/air/inc/external
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/include/external/acl
)

target_compile_options(acl_mdl_impl PRIVATE
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-fvisibility=hidden -O2 -fno-common>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:/utf-8 /Od>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

target_compile_definitions(acl_mdl_impl PRIVATE
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    FUNC_VISIBILITY
    google=ascend_private
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    $<$<STREQUAL:${ENABLE_ACL_UT},ON>:ONLY_ENABLE_ACL_UT>
)

target_link_options(acl_mdl_impl PRIVATE
    -rdynamic
    -Wl,-Bsymbolic
    -Wl,--exclude-libs,ALL
)

target_link_directories(acl_mdl_impl PRIVATE
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Android>:${TOP_DIR}/prebuilts/clang/linux-x86/aarch64/android-ndk-r21/sysroot/usr/lib/aarch64-linux-android/29>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Android>:${TOP_DIR}/build/prebuilts/clang/linux-x86/aarch64/android-ndk-r21/sysroot/usr/lib/aarch64-linux-android/29>
)

target_link_libraries(acl_mdl_impl PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:slog_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:msprof_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:mmpa_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:runtime_headers>>
    -Wl,--whole-archive
    -Wl,--no-whole-archive
    -Wl,--no-as-needed
    acl_rt # needed by dump reg func
    c_sec
    unified_dlog
    mmpa
    profapi_share
    runtime
    error_manager

    gert
    ge_executor_shared
    ge_common_base
    graph_base
    graph
    om2_executor

    -Wl,--as-needed
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Android>:-llog>
    json
)


############ libacl_mdl.so ############
add_library(acl_mdl SHARED
    model/acl_model.cpp
)

target_include_directories(acl_mdl PRIVATE
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/external/acl

    ${ASCEND_INSTALL_PATH}/include
    ${ASCEND_INSTALL_PATH}/include/acl

    ${TOP_DIR}/air/inc/external
    ${TOP_DIR}/air/inc/external/acl
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/include/external/acl
)

target_compile_options(acl_mdl PRIVATE
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-fvisibility=hidden -O2 -fno-common>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:/utf-8 /Od>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

target_compile_definitions(acl_mdl PRIVATE
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    FUNC_VISIBILITY
    google=ascend_private
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    $<$<STREQUAL:${ENABLE_ACL_UT},ON>:ONLY_ENABLE_ACL_UT>
)

target_link_options(acl_mdl PRIVATE
    -rdynamic
    -Wl,-Bsymbolic
    -Wl,--exclude-libs,ALL
)

target_link_directories(acl_mdl PRIVATE
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Android>:${TOP_DIR}/prebuilts/clang/linux-x86/aarch64/android-ndk-r21/sysroot/usr/lib/aarch64-linux-android/29>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Android>:${TOP_DIR}/build/prebuilts/clang/linux-x86/aarch64/android-ndk-r21/sysroot/usr/lib/aarch64-linux-android/29>
)

target_link_libraries(acl_mdl PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:slog_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:msprof_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:mmpa_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:runtime_headers>>
    -Wl,--whole-archive
    -Wl,--no-whole-archive
    -Wl,--no-as-needed

    acl_mdl_impl

    -Wl,--as-needed
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Android>:-llog>
)

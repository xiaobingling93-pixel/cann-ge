# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.14)
if (ENABLE_OPEN_SRC)
    find_package(c_base MODULE REQUIRED)
endif()

set(INSTALL_LIBRARY_DIR lib)
set(EXECUTOR_SRC_LIST
    ${AIR_CODE_DIR}/runtime/c/executor/ge_executor.c
    ${AIR_CODE_DIR}/runtime/c/executor/model_executor.c
    ${AIR_CODE_DIR}/runtime/c/executor/model_loader.c
    ${AIR_CODE_DIR}/runtime/c/executor/model_manager.c
    ${AIR_CODE_DIR}/runtime/c/executor/model_desc.c
    ${AIR_CODE_DIR}/runtime/c/executor/model_parse.c
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:${AIR_CODE_DIR}/runtime/c/executor/maintain_manager_dynamic.c>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},LiteOS>:${AIR_CODE_DIR}/runtime/c/executor/maintain_manager_static.c>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:${AIR_CODE_DIR}/runtime/c/executor/model_full_load.c>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},LiteOS>:${AIR_CODE_DIR}/runtime/c/executor/model_segment_load.c>
)

add_library(ge_executor_c_static STATIC
    ${EXECUTOR_SRC_LIST}
)

target_include_directories(ge_executor_c_static PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/framework/executor_c/c_base
    ${AIR_CODE_DIR}/inc/framework/executor_c/c_mmpa
    ${AIR_CODE_DIR}/runtime/c/executor
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},LiteOS>:${AIR_CODE_DIR}/runtime/c/dbg/profiling>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},LiteOS>:${AIR_CODE_DIR}/runtime/c/dbg/dump>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},LiteOS>:${AIR_CODE_DIR}/runtime/c/dbg/>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},LiteOS>:${AIR_CODE_DIR}/inc>
)

set(COMMON_COMPILE_OPTIONS
    -Os
    -fvisibility=hidden
    -fno-common
    -Wextra
    -Wfloat-equal
)
if(${TARGET_SYSTEM_NAME} STREQUAL "LiteOS")
    target_compile_definitions(ge_executor_c_static PRIVATE
        FUNC_VISIBILITY
        NANO_OS_TYPE=1
    )

    target_compile_options(ge_executor_c_static PRIVATE
        ${COMMON_COMPILE_OPTIONS}
        ${CMAKE_EXTRA_COMPILE_OPTIONS}
    )
elseif(${TARGET_SYSTEM_NAME} STREQUAL "Linux")
    target_compile_definitions(ge_executor_c_static PRIVATE
        FUNC_VISIBILITY
        NANO_OS_TYPE=0
    )
    target_compile_options(ge_executor_c_static PRIVATE
        ${COMMON_COMPILE_OPTIONS}
    )
endif()

target_link_libraries(ge_executor_c_static PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    c_sec_headers
    msprof_headers
    -Wl,--whole-archive
    c_base_static
    runtime_static
    -Wl,--no-whole-archive
    -s
PUBLIC
    runtime_headers
    air_headers
    slog_headers
)

if (NOT ENABLE_OPEN_SRC)
    target_link_libraries(ge_executor_c_static PRIVATE
        c_mmpa_headers
    )
endif()

target_link_options(ge_executor_c_static PRIVATE
    -Wl,-Bsymbolic
)

set_target_properties(ge_executor_c_static PROPERTIES
    OUTPUT_NAME ge_executor
)

install(TARGETS  ge_executor_c_static OPTIONAL
    ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR}/c
)

set(PKG_NAME air_c_exec)
if (NOT ENABLE_OPEN_SRC)
    install(TARGETS ge_executor_c_static
            EXPORT ${PKG_NAME}-targets
            ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR}/c/ge_exec OPTIONAL COMPONENT opensdk
            )
    if (PACKAGE STREQUAL "opensdk")
        install(EXPORT ${PKG_NAME}-targets DESTINATION ${INSTALL_CONFIG_DIR}
            FILE ${PKG_NAME}-targets.cmake COMPONENT opensdk EXCLUDE_FROM_ALL
        )
        configure_package_config_file(${TOP_DIR}/cmake/config/pkg_config_template.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/${PKG_NAME}-config.cmake
            INSTALL_DESTINATION ${INSTALL_CONFIG_DIR}
            PATH_VARS INSTALL_BASE_DIR INSTALL_INCLUDE_DIR INSTALL_LIBRARY_DIR INSTALL_RUNTIME_DIR INSTALL_CONFIG_DIR
            INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
        )
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PKG_NAME}-config.cmake
            DESTINATION ${INSTALL_CONFIG_DIR} COMPONENT opensdk EXCLUDE_FROM_ALL
        )
    endif()
endif()
unset(PKG_NAME)

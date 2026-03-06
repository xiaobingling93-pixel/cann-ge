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
set(MAINTAIN_SRC_LIST
    ${AIR_CODE_DIR}/runtime/c/dbg/profiling/profiling_dynamic.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dbg_main.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/dump_config.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/dump_thread_manager_dynamic.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/dump.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/parse_dbg_file.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/parse_json_file.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/tlv_parse.c
)

set(MAINTAIN_SRC_LITEOS_STUB_LIST
    ${AIR_CODE_DIR}/runtime/c/dbg/dbg_stub_liteos/profiling/profiling.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dbg_main.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dbg_stub_liteos/dump/dump.c
)

set(MAINTAIN_SRC_LITEOS_LIST
    ${AIR_CODE_DIR}/runtime/c/dbg/profiling/profiling_static.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dbg_main.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/dump_config.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/dump_thread_manager_static.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/dump.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/parse_dbg_file.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/parse_json_file.c
    ${AIR_CODE_DIR}/runtime/c/dbg/dump/tlv_parse.c
)
if(${TARGET_SYSTEM_NAME} STREQUAL "Linux")
    add_library(ge_dbg_c SHARED
        ${MAINTAIN_SRC_LIST}
    )

    target_include_directories(ge_dbg_c PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AIR_CODE_DIR}/runtime/c/dbg/profiling
        ${AIR_CODE_DIR}/runtime/c/dbg/dump
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/inc/framework/executor_c/c_base
        ${AIR_CODE_DIR}/inc/framework/executor_c/c_mmpa
    )

    target_compile_options(ge_dbg_c PRIVATE
        -Os
        -fno-common
        -Wextra
        -Wfloat-equal
    )

    target_compile_definitions(ge_dbg_c PRIVATE
        FUNC_VISIBILITY
        $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},LiteOS>,NANO_OS_TYPE=1,NANO_OS_TYPE=0>
    )

    target_link_libraries(ge_dbg_c PRIVATE
        intf_pub
        c_sec_headers
        msprof_headers
        runtime_headers
        -Wl,--whole-archive
        c_json_static
        -Wl,--no-whole-archive
        slog_headers
        -s
    )

    target_link_options(ge_dbg_c PRIVATE
        -Wl,-Bsymbolic
    )

    set_target_properties(ge_dbg_c PROPERTIES
        OUTPUT_NAME ge_dbg
    )

    install(TARGETS  ge_dbg_c OPTIONAL
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/c
    )
elseif(${TARGET_SYSTEM_NAME} STREQUAL "LiteOS")
    add_library(ge_dbg_c_static_stub STATIC
        ${MAINTAIN_SRC_LITEOS_STUB_LIST}
    )

    target_include_directories(ge_dbg_c_static_stub PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AIR_CODE_DIR}/runtime/c/dbg/profiling
        ${AIR_CODE_DIR}/runtime/c/dbg/dump
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/inc/framework/executor_c/c_base
        ${AIR_CODE_DIR}/inc/framework/executor_c/c_mmpa
    )

    target_compile_options(ge_dbg_c_static_stub PRIVATE
        ${CMAKE_EXTRA_COMPILE_OPTIONS}
    )

    target_compile_definitions(ge_dbg_c_static_stub PRIVATE
        FUNC_VISIBILITY
        $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},LiteOS>,NANO_OS_TYPE=1,NANO_OS_TYPE=0>
    )

    target_link_libraries(ge_dbg_c_static_stub PRIVATE
        intf_pub
        c_sec_headers
        msprof_headers
        c_mmpa_headers
        runtime_headers
        slog_headers
        -s
    )

    target_link_options(ge_dbg_c_static_stub PRIVATE
        -Wl,-Bsymbolic
    )

    set_target_properties(ge_dbg_c_static_stub PROPERTIES
        OUTPUT_NAME ge_dbg_stub
    )

    install(TARGETS  ge_dbg_c_static_stub OPTIONAL
        ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR}/c
    )
    # begin
    add_library(ge_dbg_c_static_debug STATIC
        ${MAINTAIN_SRC_LITEOS_LIST}
    )

    target_include_directories(ge_dbg_c_static_debug PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AIR_CODE_DIR}/runtime/c/dbg/profiling
        ${AIR_CODE_DIR}/runtime/c/dbg/dump
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/inc/framework/executor_c/c_base
        ${AIR_CODE_DIR}/inc/framework/executor_c/c_mmpa
    )

    target_compile_options(ge_dbg_c_static_debug PRIVATE
        ${CMAKE_EXTRA_COMPILE_OPTIONS}
    )

    target_compile_definitions(ge_dbg_c_static_debug PRIVATE
        FUNC_VISIBILITY
        $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},LiteOS>,NANO_OS_TYPE=1,NANO_OS_TYPE=0>
    )

    target_link_libraries(ge_dbg_c_static_debug PRIVATE
        intf_pub
        c_sec_headers
        msprof_headers
        runtime_headers
        c_mmpa_headers
        -Wl,--whole-archive
        c_json_static
        -Wl,--no-whole-archive
        slog_headers
        -s
    )

    target_link_options(ge_dbg_c_static_debug PRIVATE
        -Wl,-Bsymbolic
    )

    set_target_properties(ge_dbg_c_static_debug PROPERTIES
        OUTPUT_NAME ge_dbg_debug # lib name
    )

    # LiteOS打包
    set(DBG_STATIC_LIB_NAME "libge_dbg.a")
    set(GE_DBG_C_STATIC_LIB_MRI "${CMAKE_CURRENT_BINARY_DIR}/ge_dbg_c_static.mri")
    if(NOT ENABLE_OPEN_SRC)
        add_custom_command(
            OUTPUT ${GE_DBG_C_STATIC_LIB_MRI}
            COMMAND ${CMAKE_COMMAND} -E touch ${GE_DBG_C_STATIC_LIB_MRI}
            COMMAND ${CMAKE_COMMAND} -E echo "create ${CMAKE_CURRENT_BINARY_DIR}/${DBG_STATIC_LIB_NAME}" >> ${GE_DBG_C_STATIC_LIB_MRI}
            COMMAND ${CMAKE_COMMAND} -E echo "addlib $<TARGET_FILE:ge_dbg_c_static_debug>" >> ${GE_DBG_C_STATIC_LIB_MRI}
            COMMAND ${CMAKE_COMMAND} -E echo "addlib $<TARGET_FILE:c_json_static>" >> ${GE_DBG_C_STATIC_LIB_MRI}
            COMMAND ${CMAKE_COMMAND} -E echo "save" >> ${GE_DBG_C_STATIC_LIB_MRI}
            COMMAND ${CMAKE_COMMAND} -E echo "end" >> ${GE_DBG_C_STATIC_LIB_MRI}
            DEPENDS ge_dbg_c_static_debug c_json_static
        )
    else()
        file(WRITE ${GE_DBG_C_STATIC_LIB_MRI}
        "create ${CMAKE_CURRENT_BINARY_DIR}/${DBG_STATIC_LIB_NAME}
            addlib ${CMAKE_CURRENT_BINARY_DIR}/libge_dbg_debug.a
            addlib ${ASCEND_INSTALL_PATH}/lib64/c/libc_json.a
            save
            end")
    endif()

    add_custom_target(
        ge_dbg_c_static_lib_mri ALL
        DEPENDS ${GE_DBG_C_STATIC_LIB_MRI})

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${DBG_STATIC_LIB_NAME}
        COMMAND ar -M <${GE_DBG_C_STATIC_LIB_MRI}
        DEPENDS ge_dbg_c_static_lib_mri
    )

    add_custom_target(ge_dbg_c_static DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${DBG_STATIC_LIB_NAME})

    install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/${DBG_STATIC_LIB_NAME}
        DESTINATION ${INSTALL_LIBRARY_DIR}/c OPTIONAL
    )
    # LiteOS end
    # end
endif()

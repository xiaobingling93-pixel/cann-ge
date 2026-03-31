# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (ascend_hal_FOUND)
    message(STATUS "Package ascend_hal has been found.")
    return()
endif()

set(_cmake_targets_defined "")
set(_cmake_targets_not_defined "")
set(_cmake_expected_targets "")
foreach(_cmake_expected_target IN ITEMS ascend_hal_stub ascend_hal_headers)
    list(APPEND _cmake_expected_targets "${_cmake_expected_target}")
    if(TARGET "${_cmake_expected_target}")
        list(APPEND _cmake_targets_defined "${_cmake_expected_target}")
    else()
        list(APPEND _cmake_targets_not_defined "${_cmake_expected_target}")
    endif()
endforeach()
unset(_cmake_expected_target)

if(_cmake_targets_defined STREQUAL _cmake_expected_targets)
    unset(_cmake_targets_defined)
    unset(_cmake_targets_not_defined)
    unset(_cmake_expected_targets)
    unset(CMAKE_IMPORT_FILE_VERSION)
    cmake_policy(POP)
    return()
endif()

if(NOT _cmake_targets_defined STREQUAL "")
    string(REPLACE ";" ", " _cmake_targets_defined_text "${_cmake_targets_defined}")
    string(REPLACE ";" ", " _cmake_targets_not_defined_text "${_cmake_targets_not_defined}")
    message(FATAL_ERROR "Some (but not all) targets in this export set were already defined.\nTargets Defined: ${_cmake_targets_defined_text}\nTargets not yet defined: ${_cmake_targets_not_defined_text}\n")
endif()
unset(_cmake_targets_defined)
unset(_cmake_targets_not_defined)
unset(_cmake_expected_targets)

find_path(ascend_hal_INCLUDE_DIR
    NAMES driver/ascend_hal.h
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

set(HAL_STUB_PATH_SUFFIX "devlib")

if(PRODUCT_SIDE STREQUAL "device")
    set(HAL_STUB_PATH_SUFFIX "devlib/device")
endif()

find_library(ascend_hal_stub_SHARED_LIBRARY
    NAMES libascend_hal.so
    PATH_SUFFIXES ${HAL_STUB_PATH_SUFFIX}
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ascend_hal
    FOUND_VAR
        ascend_hal_FOUND
    REQUIRED_VARS
        ascend_hal_INCLUDE_DIR
        ascend_hal_stub_SHARED_LIBRARY
)

if(ascend_hal_FOUND)
    include(CMakePrintHelpers)
    message(STATUS "Variables in ascend_hal module:")
    cmake_print_variables(ascend_hal_INCLUDE_DIR)
    cmake_print_variables(ascend_hal_stub_SHARED_LIBRARY)
    # 创建 stub_lib 目录（仅包含需要的桩库）
    set(ASCEND_HAL_STUB_LIB_DIR "${CMAKE_BINARY_DIR}/stub_lib")
    file(MAKE_DIRECTORY "${ASCEND_HAL_STUB_LIB_DIR}")

    # 创建 libascend_hal.so 的符号链接到 stub_lib 目录
    get_filename_component(ASCEND_HAL_STUB_REALPATH "${ascend_hal_stub_SHARED_LIBRARY}" REALPATH)
    set(ASCEND_HAL_STUB_LINK "${ASCEND_HAL_STUB_LIB_DIR}/libascend_hal.so")

    # 如果链接已存在，先删除
    if(EXISTS "${ASCEND_HAL_STUB_LINK}")
        file(REMOVE "${ASCEND_HAL_STUB_LINK}")
    endif()

    # 创建符号链接（使用 COPY_ON_ERROR 作为后备）
    file(CREATE_LINK "${ASCEND_HAL_STUB_REALPATH}" "${ASCEND_HAL_STUB_LINK}" COPY_ON_ERROR)

    message(STATUS "Created stub library symlink: ${ASCEND_HAL_STUB_LINK} -> ${ASCEND_HAL_STUB_REALPATH}")
    # 使用 SHARED IMPORTED 方式
    add_library(ascend_hal_stub SHARED IMPORTED)
    set_target_properties(ascend_hal_stub PROPERTIES
        INTERFACE_LINK_LIBRARIES "ascend_hal_headers"
        IMPORTED_LOCATION "${ASCEND_HAL_STUB_LINK}"
    )
    add_library(ascend_hal_headers INTERFACE IMPORTED)
    set_target_properties(ascend_hal_headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ascend_hal_INCLUDE_DIR};${ascend_hal_INCLUDE_DIR}/driver"
    )

    cmake_print_properties(TARGETS ascend_hal_stub
        PROPERTIES INTERFACE_LINK_LIBRARIES IMPORTED_LOCATION
    )
    cmake_print_properties(TARGETS ascend_hal_headers
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
    )
endif()

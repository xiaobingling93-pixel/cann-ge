# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (cce_FOUND)
    message(STATUS "Package cce has been found.")
    return()
endif()

find_path(_INCLUDE_DIR
    NAMES experiment/cce/cce_def.hpp
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cce
    FOUND_VAR
        cce_FOUND
    REQUIRED_VARS
        _INCLUDE_DIR
)

if(cce_FOUND)
    set(cce_INCLUDE_DIR "${_INCLUDE_DIR}/experiment")
    include(CMakePrintHelpers)
    message(STATUS "Variables in cce module:")
    cmake_print_variables(cce_INCLUDE_DIR)

    add_library(cce_headers INTERFACE IMPORTED)
    set_target_properties(cce_headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${cce_INCLUDE_DIR};${cce_INCLUDE_DIR}/cce"
    )

    include(CMakePrintHelpers)
    cmake_print_properties(TARGETS cce_headers
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
    )
endif()

# Cleanup temporary variables.
set(_INCLUDE_DIR)

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (zlib_FOUND)
    message(STATUS "Package zlib has been found.")
    return()
endif()

find_path(ZLIB_INCLUDE
    NAMES zlib.h
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
)
find_library(ZLIB_LIBRARY
    NAMES libz.a
    PATH_SUFFIXES lib lib64
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
)
find_path(MINIZIP_INCLUDE
    NAMES minizip/zip.h minizip/unzip.h minizip/ioapi.h
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
)

find_library(MINIZIP_LIBRARY
    NAMES libminizip.a
    PATH_SUFFIXES lib lib64
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(zlib
    FOUND_VAR
        zlib_FOUND
    REQUIRED_VARS
        ZLIB_INCLUDE
        ZLIB_LIBRARY
        MINIZIP_INCLUDE
        MINIZIP_LIBRARY
)

if(zlib_FOUND)
    set(ZLIB_INCLUDE_DIR ${ZLIB_INCLUDE})

    add_library(zlib_static STATIC IMPORTED)
    set_target_properties(zlib_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ZLIB_INCLUDE}"
        IMPORTED_LOCATION             "${ZLIB_LIBRARY}"
    )

    add_library(minizip_static STATIC IMPORTED)
    set_target_properties(minizip_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${MINIZIP_INCLUDE}"
        IMPORTED_LOCATION             "${MINIZIP_LIBRARY}"
        # 自动添加libminizip.a对libz.a的依赖
        INTERFACE_LINK_LIBRARIES ${ZLIB_LIBRARY}
    )
endif()

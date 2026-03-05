# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (TARGET zlib_bin_build)
    return()
endif()

include(ExternalProject)

find_path(ZLIB_INCLUDE
    NAMES zlib.h
    PATH_SUFFIXES include
    PATHS ${CMAKE_THIRD_PARTY_LIB_DIR}/zlib
    NO_DEFAULT_PATH)
find_library(ZLIB_LIBRARY
    NAMES libz.a
    PATH_SUFFIXES lib lib64
    PATHS ${CMAKE_THIRD_PARTY_LIB_DIR}/zlib
    NO_DEFAULT_PATH)
find_path(MINIZIP_INCLUDE
        NAMES minizip/zip.h minizip/unzip.h minizip/ioapi.h
        PATH_SUFFIXES include
        PATHS ${CMAKE_THIRD_PARTY_LIB_DIR}/zlib
        NO_DEFAULT_PATH)
find_library(MINIZIP_LIBRARY
        NAMES libminizip.a
        PATH_SUFFIXES lib lib64
        PATHS ${CMAKE_THIRD_PARTY_LIB_DIR}/zlib
        NO_DEFAULT_PATH)

if(ZLIB_INCLUDE AND ZLIB_LIBRARY AND MINIZIP_INCLUDE AND MINIZIP_LIBRARY)
    set(zlib_FOUND TRUE)
else()
    set(zlib_FOUND FALSE)
endif()

if(zlib_FOUND AND NOT FORCE_REBUILD_CANN_3RD)
    message(STATUS "[zlib] zlib inc found in ${ZLIB_INCLUDE}, zlib lib found in ${ZLIB_LIBRARY}.")
else()
    message(STATUS "[zlib] zlib_FOUND:${zlib_FOUND}, FORCE_REBUILD_CANN_3RD:${FORCE_REBUILD_CANN_3RD}")
    if(zlib_FOUND)
        message(STATUS "[zlib] zlib inc found in ${ZLIB_INCLUDE}, zlib lib found in ${ZLIB_LIBRARY}.")
    endif()
    set(ZLIB_INSTALL_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/zlib)
    set(REQ_URL "${CMAKE_THIRD_PARTY_LIB_DIR}/zlib/zlib-1.2.13.tar.xz")
    set(REQ_URL_BACK "${CMAKE_THIRD_PARTY_LIB_DIR}/zlib/zlib-1.2.13.tar.gz")
    if(EXISTS ${REQ_URL})
        message(STATUS "[zlib] ${REQ_URL} found.")
    elseif(EXISTS ${REQ_URL_BACK})
        message(STATUS "[zlib] ${REQ_URL_BACK} found.")
        set(REQ_URL ${REQ_URL_BACK})
    else()
        message(STATUS "[zlib] ${REQ_URL} not found, need download.")
        set(REQ_URL "https://gitcode.com/cann-src-third-party/zlib/releases/download/v1.2.13/zlib-1.2.13.tar.gz")
    endif()

    set(ZLIB_C_FLAGS "-fPIC -fexceptions -O2")
    ExternalProject_Add(zlib_bin_build
                        URL ${REQ_URL}
                        TLS_VERIFY OFF
                        PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/patch/zlib_add_minizip_static_lib.patch
                        CONFIGURE_COMMAND ${CMAKE_COMMAND}
                            -DCMAKE_INSTALL_PREFIX=${ZLIB_INSTALL_DIR}
                            -DCMAKE_C_FLAGS=${ZLIB_C_FLAGS}
                            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
                            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                            -DLLVM_PATH=${LLVM_PATH}
                            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
                            # -DUNZ_MAXFILENAMEINZIP 4096配置ZIP中文件名长度，默认值256偏短
                            -DUNZ_MAXFILENAMEINZIP=4096
                            <SOURCE_DIR>
                        BUILD_COMMAND $(MAKE)
                        EXCLUDE_FROM_ALL TRUE
    )
    message(STATUS "zlib and minizip will be installed to: ${ZLIB_INSTALL_DIR}")
endif()

# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

include_guard(GLOBAL)

if (POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif ()

unset(mockcpp_FOUND CACHE)
unset(mockcpp_FOUND CACHE)

set(MOCKCPP_DOWNLOAD_PATH ${CMAKE_INSTALL_PREFIX}/third_party/mockcpp_src)
set(MOCKCPP_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/third_party/mockcpp)

find_path(MOCKCPP_INCLUDE
        NAMES mockcpp/mockcpp.h
        PATHS ${MOCKCPP_INSTALL_PATH}/include
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH)

find_path(BOOST_INCLUDE
        NAMES boost/config.hpp
        PATHS
        ${ASCEND_3RD_LIB_PATH}/boost/include
        ${CMAKE_INSTALL_PREFIX}/boost_src
        NO_DEFAULT_PATH
        NO_CMAKE_FIND_ROOT_PATH)

find_library(MOCKCPP_STATIC_LIBRARY
        NAMES libmockcpp.a
        PATHS ${MOCKCPP_INSTALL_PATH}/lib
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(mockcpp
        FOUND_VAR
        mockcpp_FOUND
        REQUIRED_VARS MOCKCPP_INCLUDE BOOST_INCLUDE MOCKCPP_STATIC_LIBRARY
)

if (mockcpp_FOUND)
    message("mockcpp found in ${MOCKCPP_INSTALL_PATH}")
    set(MOCKCPP_INCLUDE_DIR ${MOCKCPP_INCLUDE} ${BOOST_INCLUDE})
    add_library(mockcpp_static STATIC IMPORTED)
    set_target_properties(mockcpp_static PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${MOCKCPP_INCLUDE_DIR}"
            IMPORTED_LOCATION "${MOCKCPP_STATIC_LIBRARY}")
else ()
    message("can not find mockcpp found in ${MOCKCPP_INSTALL_PATH}, begin to download")
    if (CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(mockcpp_CXXFLAGS "-fPIC -D_GLIBCXX_USE_CXX11_ABI=0")
    else ()
        set(mockcpp_CXXFLAGS "-fPIC -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0")
    endif ()
    set(mockcpp_FLAGS "-fPIC -D_GLIBCXX_USE_CXX11_ABI=0")
    set(mockcpp_LINKER_FLAGS "")

    if (BOOST_INCLUDE)
        message(STATUS "find boost include path：${BOOST_INCLUDE}")
    else ()
        message(FATAL_ERROR "can not find boost include path, please install boost first!")
    endif ()

    set(PATCH_FILE ${CMAKE_CURRENT_LIST_DIR}/patch/mockcpp-2.7_py3.patch)
    include(ExternalProject)
    ExternalProject_Add(mockcpp
            URL "https://gitcode.com/cann-src-third-party/mockcpp/releases/download/v2.7-h1/mockcpp-2.7.tar.gz"
            URL_HASH SHA256=73ab0a8b6d1052361c2cebd85e022c0396f928d2e077bf132790ae3be766f603
            DOWNLOAD_DIR ${MOCKCPP_DOWNLOAD_PATH}/src
            SOURCE_DIR ${MOCKCPP_DOWNLOAD_PATH}/src
            PATCH_COMMAND git init && git apply ${PATCH_FILE}

            CONFIGURE_COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR}
            -DCMAKE_CXX_FLAGS=${mockcpp_CXXFLAGS}
            -DCMAKE_C_FLAGS=${mockcpp_FLAGS}
            -DBOOST_INCLUDE_DIRS=${BOOST_INCLUDE}
            -DCMAKE_SHARED_LINKER_FLAGS=${mockcpp_LINKER_FLAGS}
            -DCMAKE_EXE_LINKER_FLAGS=${mockcpp_LINKER_FLAGS}
            -DCMAKE_INSTALL_PREFIX=${MOCKCPP_INSTALL_PATH}
            <SOURCE_DIR>
            BUILD_COMMAND $(MAKE)
    )
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${MOCKCPP_INSTALL_PATH}/include)
    set(MOCKCPP_INCLUDE_DIR ${MOCKCPP_INSTALL_PATH}/include ${BOOST_INCLUDE})
    add_library(mockcpp_static STATIC IMPORTED)
    set_target_properties(mockcpp_static PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${MOCKCPP_INCLUDE_DIR}"
            IMPORTED_LOCATION "${MOCKCPP_INSTALL_PATH}/lib/libmockcpp.a")
    add_dependencies(mockcpp_static mockcpp)
endif ()
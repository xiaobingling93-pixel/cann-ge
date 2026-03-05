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
project (ThirdPartyPkg[CXX])

set(CMAKE_VERBOSE_MAKEFILE True)
set(CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_LIST_DIR}/build/modules
)

include(CMakePrintHelpers)
include(CMakePackageConfigHelpers)

message(STATUS "[Third part build] Variables for third_party packages:")
cmake_print_variables(CMAKE_THIRD_PARTY_LIB_DIR)
cmake_print_variables(CMAKE_BUILD_COMPONENT)
cmake_print_variables(ENABLE_BUILD_DEVICE)
cmake_print_variables(USE_CXX11_ABI)
cmake_print_variables(CMAKE_TOOLCHAIN_FILE)
cmake_print_variables(LLVM_PATH)
cmake_print_variables(CMAKE_MODULE_PATH)

if(ENABLE_BUILD_DEVICE)
    # 非MDC编译流程
    set(USE_CXX11_ABI 0)
else()
    string(FIND "${CMAKE_TOOLCHAIN_FILE}" "ubuntu18.04-x86_64-llvm-toolchain.cmake" SUBSTRING_FOUND)
    if(SUBSTRING_FOUND GREATER_EQUAL 0)
        set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY false)
        # MDC编译运行态
        message("[MDC compile] Thirdparty build, MDC compile runtime state.")
        add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})
    else()
        # MDC编译开发态
        message("[MDC compile] Thirdparty build, MDC compile develop state.")
        set(USE_CXX11_ABI 0)
    endif()
endif()

set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)
set(INSTALL_RUNTIME_DIR bin)
set(INSTALL_INCLUDE_DIR include)
set(INSTALL_CONFIG_DIR cmake)
set(CMAKE_INSTALL_LIBDIR lib)
set(FORCE_REBUILD_CANN_3RD false)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif(CCACHE_PROGRAM)

if (NOT DEFINED CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH
        ${CMAKE_THIRD_PARTY_LIB_DIR}/json
        ${CMAKE_THIRD_PARTY_LIB_DIR}/openssl
        ${CMAKE_THIRD_PARTY_LIB_DIR}/protoc
        ${CMAKE_THIRD_PARTY_LIB_DIR}/protoc_grpc
        ${CMAKE_THIRD_PARTY_LIB_DIR}/grpc
        ${CMAKE_THIRD_PARTY_LIB_DIR}/protobuf_static
        ${CMAKE_THIRD_PARTY_LIB_DIR}/ascend_protobuf
        ${CMAKE_THIRD_PARTY_LIB_DIR}/ascend_protobuf_static
        ${CMAKE_THIRD_PARTY_LIB_DIR}/symengine/lib/cmake/symengine
        ${CMAKE_THIRD_PARTY_LIB_DIR}/boost/lib/cmake/Boost-1.87.0
        ${CMAKE_THIRD_PARTY_LIB_DIR}/zlib
    )
endif()

message(STATUS "[${CMAKE_BUILD_COMPONENT}] third-party process")
if(CMAKE_BUILD_COMPONENT STREQUAL "ge-compiler")
    find_package(json MODULE REQUIRED)
    find_package(openssl MODULE REQUIRED)
    find_package(abseil-cpp MODULE REQUIRED)
    find_package(protoc MODULE REQUIRED)
    find_package(ascend_protobuf_shared MODULE REQUIRED)
    find_package(boost MODULE REQUIRED)
    find_package(symengine MODULE REQUIRED)
    find_package(zlib_bin MODULE REQUIRED)
    set(selected_targets
            ascend_protobuf_build
            json_build
            openssl_build
            protoc_build
            boost_build
            symengine_build
            zlib_bin_build
    )
elseif(CMAKE_BUILD_COMPONENT STREQUAL "ge-executor")
    find_package(grpc MODULE REQUIRED)
    find_package(protoc_grpc MODULE REQUIRED)
    # protoc_grpc_build或grpc_build需要编译则强制重新编译abseil-cpp,protobuf_static,cares,re2
    if(TARGET protoc_grpc_build OR TARGET grpc_build)
        set(FORCE_REBUILD_CANN_3RD true)
    endif()
    find_package(abseil-cpp MODULE REQUIRED)
    find_package(protobuf_static MODULE REQUIRED)
    find_package(cares MODULE REQUIRED)
    find_package(re2 MODULE REQUIRED)
    # grpc_build需要编译则强制重新编译zlib,openssl
    if(TARGET grpc_build)
        set(FORCE_REBUILD_CANN_3RD true)
    else()
        set(FORCE_REBUILD_CANN_3RD false)
    endif()
    find_package(zlib_bin MODULE REQUIRED)
    find_package(openssl MODULE REQUIRED)
    # 以下独立判断是否重新编译
    find_package(ascend_protobuf_shared MODULE REQUIRED)
    find_package(ascend_protobuf_static MODULE REQUIRED)
    find_package(json MODULE REQUIRED)
    find_package(protoc MODULE REQUIRED)
    find_package(boost MODULE REQUIRED)
    find_package(symengine MODULE REQUIRED)
    set(selected_targets
            ascend_protobuf_build
            ascend_protobuf_static_build
            grpc_build
            json_build
            protobuf_static_build
            protoc_build
            protoc_grpc_build
            zlib_bin_build
            boost_build
            symengine_build
    )
else()
    find_package(grpc MODULE REQUIRED)
    find_package(protoc_grpc MODULE REQUIRED)
    # protoc_grpc_build或grpc_build需要编译则强制重新编译abseil-cpp,protobuf_static,cares,re2
    if(TARGET protoc_grpc_build OR TARGET grpc_build)
        set(FORCE_REBUILD_CANN_3RD true)
    endif()
    find_package(abseil-cpp MODULE REQUIRED)
    find_package(protobuf_static MODULE REQUIRED)
    find_package(cares MODULE REQUIRED)
    find_package(re2 MODULE REQUIRED)
    # grpc_build需要编译则强制重新编译zlib,openssl
    if(TARGET grpc_build)
        set(FORCE_REBUILD_CANN_3RD true)
    else()
        set(FORCE_REBUILD_CANN_3RD false)
    endif()
    find_package(zlib_bin MODULE REQUIRED)
    find_package(openssl MODULE REQUIRED)
    # 以下独立判断是否重新编译
    find_package(boost MODULE REQUIRED)
    find_package(symengine MODULE REQUIRED)
    find_package(ascend_protobuf_shared MODULE REQUIRED)
    find_package(ascend_protobuf_static MODULE REQUIRED)
    find_package(json MODULE REQUIRED)
    find_package(protoc MODULE REQUIRED)

    # abseil_build,cares_build,re2_build只在被依赖时编译,不主动编译
    set(selected_targets
            ascend_protobuf_build
            ascend_protobuf_static_build
            grpc_build
            json_build
            openssl_build
            protobuf_static_build
            protoc_build
            protoc_grpc_build
            zlib_bin_build
            boost_build
            symengine_build
    )
endif()

if(CMAKE_BUILD_COMPONENT STREQUAL "LLT")
    find_package(gbenchmark MODULE REQUIRED)
    find_package(gtest_shared MODULE REQUIRED)
    set(selected_targets
        ${selected_targets}
        benchmark_build
        gtest_shared_build
    )
endif()

message(STATUS "[Third part build] Building default targets")
add_custom_target(select_targets)
foreach(dep ${selected_targets})
    if(TARGET ${dep})
        add_dependencies(select_targets ${dep})
        message(STATUS "[Third part build] add dependence:${dep} success.")
    else()
        message(STATUS "[Third part build] add dependence:${dep} ignored.")
    endif()
endforeach()

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/scripts/modules
        DESTINATION ${CMAKE_THIRD_PARTY_LIB_DIR}/cmake
)
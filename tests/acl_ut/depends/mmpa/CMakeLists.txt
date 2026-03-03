# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

project(STUB_MMPA)

file(GLOB_RECURSE SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "src/mmpa_stub.cpp"
)

add_library(mmpa_stub SHARED ${SRC_FILES})

target_include_directories(mmpa_stub PRIVATE
    ${BASE_DIR}/inc
    ${BASE_DIR}/inc/graph_metadef
    ${BASE_DIR}/inc/graph_metadef/external
    ${BASE_DIR}/inc/external
    ${BASE_DIR}/inc/external/ge
    ${BASE_DIR}/inc/framework
    ${BASE_DIR}/tests/acl_ut/depends
    ${METADEF_DIR}/pkg_inc
    ${ASCEND_INSTALL_PATH}
    ${ASCEND_INSTALL_PATH}/pkg_inc/base
    ${ASCEND_INSTALL_PATH}/include
    ${ASCEND_INSTALL_PATH}/include/acl
    ${ASCEND_INSTALL_PATH}/include/external
    ${ASCEND_INSTALL_PATH}/include/graph
    ${ASCEND_INSTALL_PATH}/pkg_inc
    ${ASCEND_INSTALL_PATH}/pkg_inc/runtime
    ${ASCEND_INSTALL_PATH}/pkg_inc/runtime/runtime
    ${ASCEND_INSTALL_PATH}/pkg_inc/profiling
    ${ASCEND_INSTALL_PATH}/pkg_inc/dump
)

target_link_libraries(mmpa_stub PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        -Wl,--no-as-needed
        c_sec
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        GTest::gmock_main
)

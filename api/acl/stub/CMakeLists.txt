# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

##################################################################################################################
add_custom_target(
    stub_acl_src ALL
    DEPENDS  ${CMAKE_CURRENT_BINARY_DIR}/cblas_stub.cpp
             ${CMAKE_CURRENT_BINARY_DIR}/op_compiler_stub.cpp
             ${CMAKE_CURRENT_BINARY_DIR}/op_executor_stub.cpp
             ${CMAKE_CURRENT_BINARY_DIR}/mdl_stub.cpp
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cblas_stub.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/op_compiler_stub.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/op_executor_stub.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/mdl_stub.cpp
    COMMAND echo "Generating stub files:cblas_stub.cpp, op_compiler_stub.cpp, op_executor_stub.cpp, mdl_stub.cpp."
            && ${HI_PYTHON} ${CMAKE_CURRENT_LIST_DIR}/gen_stubapi.py
            ${CMAKE_CURRENT_LIST_DIR}/../../../inc/external/acl/ops
            ${CMAKE_CURRENT_LIST_DIR}/../../../inc/external/acl
            ${CMAKE_CURRENT_LIST_DIR}/../../../inc/external/acl
            ${CMAKE_CURRENT_LIST_DIR}/../../../inc/external/acl
            ${CMAKE_CURRENT_BINARY_DIR}/cblas_stub.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/op_compiler_stub.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/op_executor_stub.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/mdl_stub.cpp
            && echo "Generating stub files end."
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS gen_stubapi.py
    ${CMAKE_CURRENT_LIST_DIR}/../../../inc/external/acl/ops
    ${CMAKE_CURRENT_LIST_DIR}/../../../inc/external/acl
    ${CMAKE_CURRENT_LIST_DIR}/../../../inc/external/acl
    ${CMAKE_CURRENT_LIST_DIR}/../../../inc/external/acl
)

#----------------------------------------------------------------------------------------------------------------#
############ stub/libacl_cblas.so ############
add_library(stub_acl_cblas SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/cblas_stub.cpp
)

target_include_directories(stub_acl_cblas PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../inc/external
    ${CMAKE_CURRENT_LIST_DIR}/../../../inc/external/acl
    ${CMAKE_CURRENT_LIST_DIR}/../../../inc/external/acl/ops
    ${ASCEND_INSTALL_PATH}/include
    ${ASCEND_INSTALL_PATH}/include/acl
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/include/external/acl
)

target_compile_definitions(stub_acl_cblas PRIVATE
    _FORTIFY_SOURCE=2
)

target_compile_options(stub_acl_cblas PRIVATE
    -Wno-unused-parameter
)

set_target_properties(stub_acl_cblas
    PROPERTIES
    OUTPUT_NAME acl_cblas
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/stub
)

target_link_libraries(stub_acl_cblas PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
)
#----------------------------------------------------------------------------------------------------------------#
############ stub/libacl_op_compiler.so ############
add_library(stub_acl_op_compiler SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/op_compiler_stub.cpp
)

target_include_directories(stub_acl_op_compiler PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../inc/external
    ${CMAKE_CURRENT_LIST_DIR}/../../../inc/external/acl
    ${ASCEND_INSTALL_PATH}/include
    ${ASCEND_INSTALL_PATH}/include/acl
)

target_compile_definitions(stub_acl_op_compiler PRIVATE
    _FORTIFY_SOURCE=2
)
set_target_properties(stub_acl_op_compiler
    PROPERTIES
    OUTPUT_NAME acl_op_compiler
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/stub
)

target_compile_options(stub_acl_op_compiler PRIVATE
    -Wno-unused-parameter
)

target_link_libraries(stub_acl_op_compiler PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
)
##################################################################################################################

#----------------------------------------------------------------------------------------------------------------#
############ stub/libacl_op_executor.so ############
add_library(stub_acl_op_executor SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/op_executor_stub.cpp
)

target_include_directories(stub_acl_op_executor PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../inc/external
    ${CMAKE_CURRENT_LIST_DIR}/../../../inc/external/acl
    ${ASCEND_INSTALL_PATH}/include
    ${ASCEND_INSTALL_PATH}/include/acl
)

target_compile_definitions(stub_acl_op_executor PRIVATE
    _FORTIFY_SOURCE=2
)

target_compile_options(stub_acl_op_executor PRIVATE
    -Wno-unused-parameter
)

set_target_properties(stub_acl_op_executor
    PROPERTIES
    OUTPUT_NAME acl_op_executor
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/stub
)

target_link_libraries(stub_acl_op_executor PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
)
##################################################################################################################

#----------------------------------------------------------------------------------------------------------------#
############ stub/libacl_mdl.so ############
add_library(stub_acl_mdl SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/mdl_stub.cpp
)

target_include_directories(stub_acl_mdl PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../inc/external
    ${CMAKE_CURRENT_LIST_DIR}/../../../inc/external/acl
    ${ASCEND_INSTALL_PATH}/include
    ${ASCEND_INSTALL_PATH}/include/acl
)

target_compile_definitions(stub_acl_mdl PRIVATE
    _FORTIFY_SOURCE=2
)

target_compile_options(stub_acl_mdl PRIVATE
    -Wno-unused-parameter
)

set_target_properties(stub_acl_mdl
    PROPERTIES
    OUTPUT_NAME acl_mdl
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/stub
)

target_link_libraries(stub_acl_mdl PRIVATE
    acl_rt
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
)
##################################################################################################################

add_dependencies(stub_acl_cblas stub_acl_src)
add_dependencies(stub_acl_op_compiler stub_acl_src)
add_dependencies(stub_acl_op_executor stub_acl_src)
add_dependencies(stub_acl_mdl stub_acl_src)
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(GERT_INCLUDING_PATH
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/runtime/v2
    ${AIR_CODE_DIR}/runtime/v2/engine
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${AIR_CODE_DIR}/runtime/v1
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/ge
    ${CMAKE_BINARY_DIR}/proto/ge/proto
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/include/external/acl
    ${TOP_DIR}/runtime/include/external/acl/error_codes
)

set(GERT_COMPILE_DEF
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    LOG_CPP
)

#######################################################
set(STUB_HEADER_LIST
    ${AIR_CODE_DIR}/inc/framework/runtime/exe_graph_resource_guard.h
    ${AIR_CODE_DIR}/inc/framework/runtime/gert_api.h
    ${AIR_CODE_DIR}/inc/framework/runtime/mem_allocator.h
    ${AIR_CODE_DIR}/inc/framework/runtime/model_v2_executor.h
    ${AIR_CODE_DIR}/inc/framework/runtime/model_desc.h
    ${AIR_CODE_DIR}/inc/framework/runtime/stream_executor.h
    ${AIR_CODE_DIR}/inc/framework/runtime/subscriber/executor_subscribers_scheduler.h
)

list(TRANSFORM STUB_HEADER_LIST
    REPLACE "^.*/([^/]+)\\.h$" "${CMAKE_CURRENT_BINARY_DIR}/stub_\\1.cc"
    OUTPUT_VARIABLE STUB_SRC_LIST
)


if(NOT BUILD_PKG_COMPONENT)
    # TODO:待整包下线后删除
    add_custom_command(
        OUTPUT ${STUB_SRC_LIST}
        COMMAND echo "Generating gert_stub files."
                && ${HI_PYTHON} ${METADEF_DIR}/tests/stub/gen_stubapi.py ${CMAKE_CURRENT_BINARY_DIR} ${STUB_HEADER_LIST}
                && echo "Generating stub files end."
    )
else()
    add_custom_command(
        OUTPUT ${STUB_SRC_LIST}
        COMMAND echo "Generating gert_stub files."
                && ${HI_PYTHON} ${GE_METADEF_DIR}/../tests/graph_metadef/stub/gen_stubapi.py ${CMAKE_CURRENT_BINARY_DIR} ${STUB_HEADER_LIST}
                && echo "Generating stub files end."
    )
endif()

add_custom_target(stub_gert DEPENDS ${STUB_SRC_LIST})

############### stub/libgert.so ###############
add_library(gert_stub SHARED ${STUB_SRC_LIST})
add_dependencies(gert_stub stub_gert)
target_include_directories(gert_stub PRIVATE ${GERT_INCLUDING_PATH})
target_compile_definitions(gert_stub PRIVATE ${GERT_COMPILE_DEF})
target_link_libraries(gert_stub PRIVATE
    intf_pub
    slog_headers
    metadef_headers
    mmpa_headers
    msprof_headers
    runtime_headers
    c_sec_headers
    ${AIR_COMMON_LINK_OPTION}
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
)

target_compile_options(gert_stub PRIVATE
    -fvisibility=hidden -Os -fno-common -Wfloat-equal
    -Werror=return-type
    -Wno-unused-parameter
)

set_target_properties(gert_stub PROPERTIES
    OUTPUT_NAME gert
    LIBRARY_OUTPUT_DIRECTORY stub
)

############### stub/libgert.a ###############
if ("${PRODUCT}" STREQUAL "ascend031")
    target_clone(gert_stub gert_static_stub STATIC)
    
    add_dependencies(gert_static_stub stub_gert)
    
    target_compile_options(gert_static_stub PRIVATE
        -ffunction-sections
        -fdata-sections
    )
    
    set_target_properties(gert_static_stub PROPERTIES
        OUTPUT_NAME gert
        ARCHIVE_OUTPUT_DIRECTORY stub
    )
endif ()

install(TARGETS gert_stub OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/${CMAKE_SYSTEM_PROCESSOR}/stub
)

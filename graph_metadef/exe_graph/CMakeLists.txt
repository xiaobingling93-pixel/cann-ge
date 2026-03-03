# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include(${GE_METADEF_DIR}/cmake/build_type.cmake)

set(LOWERING_SRCS
    lowering/data_dependent_interpreter.cc
    lowering/device_tiling_context_builder.cc
    lowering/getcdim.cc
    lowering/kernel_run_context_builder.cc
    lowering/shape_utils.cc
    lowering/tiling_context_builder.cc
    lowering/tiling_parse_context_builder.cc
    lowering/bg_ir_attrs.cc
    lowering/bg_kernel_context_extend.cc
    lowering/buffer_pool.cc
    lowering/dev_mem_value_holder.cc
    lowering/frame_selector.cc
    lowering/generate_exe_graph.cc
    lowering/lowering_global_data.cc
    lowering/value_holder.cc
    lowering/value_holder_utils.cc
    runtime/eager_op_execution_context.cc #todo temp
)

######### liblowering.so #############
add_library(lowering SHARED ${LOWERING_SRCS})

target_include_directories(lowering PRIVATE
    ${METADEF_DIR}
)

target_include_directories(lowering PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
    ${AIR_CODE_DIR}/inc/graph_metadef
)

target_link_libraries(lowering PRIVATE
    intf_pub
    mmpa_headers
    metadef_headers
    c_sec
    slog
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
)

target_compile_definitions(lowering PRIVATE
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
)

target_compile_options(lowering PRIVATE
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal -fno-common>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

target_link_libraries(lowering PRIVATE
    ascend_protobuf
    graph_base
    register
)

if (NOT ENABLE_OPEN_SRC)
    ######### liblowering_static.a #############
    target_clone(lowering lowering_static STATIC)
    target_link_libraries(lowering_static PRIVATE
        ascend_protobuf_static
        graph_static
        register_static
    )
    target_compile_options(lowering_static PRIVATE $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>)
    set_target_properties(lowering_static PROPERTIES OUTPUT_NAME lowering)

    ############ install ############
    install(TARGETS lowering_static OPTIONAL
        ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR}
    )
endif()

##############################################################
set(STUB_HEADER_LIST_LOWERING
    ${GE_METADEF_INC_DIR}/exe_graph/lowering/bg_kernel_context_extend.h
    ${GE_METADEF_INC_DIR}/exe_graph/lowering/device_tiling_context_builder.h
)

list(TRANSFORM STUB_HEADER_LIST_LOWERING
    REPLACE "^.*/([^/]+)\\.h$" "${CMAKE_CURRENT_BINARY_DIR}/stub_\\1.cc"
    OUTPUT_VARIABLE STUB_SRC_LIST_LOWERING
)

add_custom_command(
    OUTPUT ${STUB_SRC_LIST_LOWERING}
    COMMAND echo "Generating stub files."
            && ${HI_PYTHON} ${GE_METADEF_DIR}/../tests/graph_metadef/stub/gen_stubapi.py ${CMAKE_CURRENT_BINARY_DIR} ${STUB_HEADER_LIST_LOWERING}
            && echo "Generating stub files end."
)

add_custom_target(lowering_stub DEPENDS ${STUB_SRC_LIST_LOWERING})

############ stub/liblowering.so ############
add_library(stub_lowering SHARED ${STUB_SRC_LIST_LOWERING})

add_dependencies(stub_lowering lowering_stub)

target_include_directories(stub_lowering PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
    ${AIR_CODE_DIR}/inc
)

target_compile_options(stub_lowering PRIVATE
    -Wfloat-equal
    -fno-common
    -Os
    -Werror=return-type
)

target_link_libraries(stub_lowering
    PRIVATE
        intf_pub
        c_sec_headers
    PUBLIC
        metadef_headers
)

set_target_properties(stub_lowering PROPERTIES
    OUTPUT_NAME lowering
    LIBRARY_OUTPUT_DIRECTORY stub
)

############ stub/liblowering.a ############
if (NOT ENABLE_OPEN_SRC)
    target_clone(stub_lowering stub_lowering_static STATIC)

    add_dependencies(stub_lowering_static lowering_stub)

    target_compile_options(stub_lowering_static PRIVATE
        -ffunction-sections
        -fdata-sections
    )
    set_target_properties(stub_lowering_static PROPERTIES
        OUTPUT_NAME lowering
        ARCHIVE_OUTPUT_DIRECTORY stub
    )

    ############ install ############
    install(TARGETS stub_lowering_static OPTIONAL
        ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR}/${CMAKE_SYSTEM_PROCESSOR}/stub
    )
endif ()

############ install ############
install(TARGETS stub_lowering OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/${CMAKE_SYSTEM_PROCESSOR}/stub
)

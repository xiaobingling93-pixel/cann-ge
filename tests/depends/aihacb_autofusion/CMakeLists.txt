# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include(${GE_METADEF_DIR}/graph/ascendc_ir/generator/external_generator.cmake)
file(GLOB_RECURSE ASC_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "src/*.cc"
        "${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/autofuse/*.cpp"
        "${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/autofuse/lowering/*.cpp"
        "${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/autofuse/can_fuse/*.cpp"
        "${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/autofuse/post_process/*.cpp"
        "${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/autofuse/utils/*.cpp"
        "${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/common/common_utils.cpp"
        "${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/common/platform_context.cpp"
        "${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/common/ascgraph_info_complete.cpp"
        "${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/common/code_printer.cpp"
        "${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/common/autofuse_config/*.cpp"
)

# Only create ascir_ops_header_generator if it doesn't already exist
if(NOT TARGET ascir_ops_header_generator)
add_executable(ascir_ops_header_generator
        ${GE_METADEF_DIR}/graph/ascendc_ir/generator/ascir_ops_generator_main.cc
        ${GE_METADEF_DIR}/graph/ascendc_ir/generator/generator.cc
        )
target_include_directories(ascir_ops_header_generator PUBLIC
        ${ASCEND_INSTALL_PATH}/include/experiment
        ${METADEF_DIR}/inc/external
        ${C_SEC_INCLUDE}
        ${GE_METADEF_DIR}/graph
        ${GE_METADEF_INC_DIR}
        ${GE_METADEF_INC_DIR}/graph
        )
target_link_libraries(ascir_ops_header_generator PRIVATE
        -Wl,--no-as-needed
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
        aihac_ir_register
        PUBLIC
        metadef_headers)
endif()

file(GLOB ASCIR_SOURCES
        "${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/ascir/generator/ascir_builtin_ops.cpp"
        "${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/ascir/reg_func/*.cpp"
        "${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/v35/ascir/generator/ascir_builtin_ops.cpp"
        "${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/v35/ascir/reg_func/*.cpp"
        "${AIR_CODE_DIR}/tests/graph_metadef/ut/ascendc_ir/stub/*.cc")

add_library(ascir_builtin_ops_stub SHARED ${ASCIR_SOURCES})
target_include_directories(ascir_builtin_ops_stub PUBLIC
        ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/ascir
        ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/v35/ascir
        ${METADEF_DIR}/inc
        ${METADEF_DIR}/inc/graph
        ${GE_METADEF_DIR}/graph/ascendc_ir/generator
        ${GE_METADEF_INC_DIR}
        ${GE_METADEF_INC_DIR}/graph
        ${GE_METADEF_INC_DIR}/graph/attribute_group
        ${GE_METADEF_INC_DIR}/graph/ascendc_ir
        ${GE_METADEF_INC_DIR}/graph/ascendc_ir/ascendc_ir_core
        ${GE_METADEF_INC_DIR}/external
        ${GE_METADEF_DIR}/graph
        ${C_SEC_INCLUDE}
        ${CMAKE_BINARY_DIR}/proto/ge/
        ${CMAKE_BINARY_DIR}/proto/graphengine_protos
        ${ASCEND_3RD_LIB_PATH}/protoc/include
        )
add_dependencies(ascir_builtin_ops_stub metadef_graph_protos_obj ascir_ops_header_generator)

target_link_options(ascir_builtin_ops_stub PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        -Wl,--exclude-libs,All
        )

target_link_libraries(ascir_builtin_ops_stub PUBLIC aihac_ir_register ascend_protobuf)

set(ops_header_dir ${CMAKE_CURRENT_BINARY_DIR})

message("ops_header_dir is " ${ops_header_dir}/ascir_ops.h)
ascir_generate(ascir_builtin_ops_stub
        ${ops_header_dir}
        ${CMAKE_BINARY_DIR}/tests/depends/aihacb_autofusion/libascir_builtin_ops_stub.so
        ${ops_header_dir}/ascir_ops.h)

add_library(aihacb_autofusion_stub SHARED
        ${ASC_SRC_FILES}
        ${ops_header_dir}/ascir_ops.h
        )

add_dependencies(aihacb_autofusion_stub metadef_graph ascir_builtin_ops_stub)

target_compile_definitions(aihacb_autofusion_stub PRIVATE
        PROTOBUF_INLINE_NOT_IN_HEADERS=0
        google=ascend_private
        )

target_compile_options(aihacb_autofusion_stub PRIVATE ${AIR_COMMON_COMPILE_OPTION})

target_include_directories(aihacb_autofusion_stub PRIVATE
        "src/"
        ${AIR_CODE_DIR}/
        ${AIR_CODE_DIR}/compiler
        ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/common
        ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/autofuse
        ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/autofuse/lowering
        ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/autofuse/can_fuse
        ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/autofuse/post_process
        ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/autofuse/utils
        ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/inc
        ${METADEF_DIR}/inc/
        ${GE_METADEF_INC_DIR}/
        ${GE_METADEF_INC_DIR}/graph/ascendc_ir
        ${GE_METADEF_INC_DIR}/graph/ascendc_ir/ascendc_ir_core
        ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/att/
        ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/ascir/meta
        ${CMAKE_BINARY_DIR}/proto/metadef_protos
        ${CMAKE_BINARY_DIR}/proto/graphengine_protos
        ${ops_header_dir}
        ${ASCEND_INSTALL_PATH}/include/experiment/runtime
        ${ASCEND_INSTALL_PATH}/include/experiment/msprof
        ${ASCEND_INSTALL_PATH}/include/toolchain
        )

target_link_libraries(aihacb_autofusion_stub PRIVATE
        intf_pub
        metadef_headers
        slog_headers
        mmpa_headers
        msprof_headers
        runtime_headers
        json
        ascend_protobuf
        c_sec
        ascend_protobuf
        ${AIR_COMMON_LINK_OPTION}
        )

#stub_module(aihacb_autofusion aihacb_autofusion_stub)
stub_module(aihac_autofusion aihacb_autofusion_stub)
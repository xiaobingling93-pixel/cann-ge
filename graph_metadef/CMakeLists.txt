# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include(CMakePrintHelpers)

if (NOT ENABLE_OPEN_SRC)
    set(METADEF_DIR ${TOP_DIR}/metadef)
    set(GE_METADEF_DIR ${TOP_DIR}/air/graph_metadef)
    set(GE_METADEF_INC_DIR ${TOP_DIR}/air/inc/graph_metadef)
    set(METADEF_PROTO_DIR ${TOP_DIR}/air/graph_metadef/proto)
    set(AIR_CODE_DIR ${TOP_DIR}/air)
endif()

message(STATUS "Variables in graph metadef project:")
include(cmake/common_funcs.cmake)
add_subdirectory(proto)
add_subdirectory(graph)
add_subdirectory(exe_graph)
add_subdirectory(register)
if (ENABLE_METADEF_UT OR ENABLE_METADEF_ST OR ENABLE_BENCHMARK)
    find_package(benchmark CONFIG REQUIRED)
    add_subdirectory(tests)
endif()

install(TARGETS lowering graph graph_base register metadef_headers
        aihac_ir aihac_ir_register aihac_symbolizer
    EXPORT graph_metadef-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL COMPONENT opensdk
    ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL COMPONENT opensdk
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR} OPTIONAL COMPONENT opensdk
)

if (NOT ENABLE_OPEN_SRC)
    install(TARGETS graph_base_static metadef_static
        ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
    )
    install(TARGETS atc_stub_graph_static stub_exe_graph_static stub_register_static
        ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR}/${CMAKE_SYSTEM_PROCESSOR}/stub OPTIONAL
    )
endif ()

# 下列头文件发布是非法的，需要在后续整改中删掉
# --------------------start------------------------
install(FILES   ${GE_METADEF_DIR}/third_party/transformer/inc/axis_util.h
                ${GE_METADEF_DIR}/third_party/transformer/inc/expand_dimension.h
                ${GE_METADEF_DIR}/third_party/transformer/inc/transfer_shape_utils.h
                ${GE_METADEF_DIR}/third_party/transformer/inc/transfer_range_according_to_format.h
                ${GE_METADEF_DIR}/third_party/transformer/inc/transfer_shape_according_to_format.h
                ${GE_METADEF_DIR}/third_party/transformer/inc/transfer_shape_according_to_format_ext.h
                ${GE_METADEF_DIR}/third_party/transformer/inc/transfer_def.h
        DESTINATION ${INSTALL_INCLUDE_DIR}/metadef/transformer COMPONENT opensdk EXCLUDE_FROM_ALL
)
install(FILES   ${GE_METADEF_DIR}/register/op_tiling/op_tiling_constants.h
                ${GE_METADEF_DIR}/register/op_tiling/op_compile_info_manager.h
                ${GE_METADEF_DIR}/register/op_tiling/op_tiling_utils.h
        DESTINATION ${INSTALL_INCLUDE_DIR}/metadef/register/op_tiling COMPONENT opensdk EXCLUDE_FROM_ALL
)
install(FILES   ${GE_METADEF_DIR}/graph/normal_graph/operator_impl.h
                ${GE_METADEF_DIR}/graph/normal_graph/op_io.h
        DESTINATION ${INSTALL_INCLUDE_DIR}/metadef/graph COMPONENT opensdk EXCLUDE_FROM_ALL
)

install(FILES   ${GE_METADEF_DIR}/graph/utils/dumper/ge_graph_dumper.h
        DESTINATION ${INSTALL_INCLUDE_DIR}/metadef/graph/utils/dumper COMPONENT opensdk EXCLUDE_FROM_ALL
)
# ---------------------end-------------------------

install(DIRECTORY ${GE_METADEF_INC_DIR}/ DESTINATION ${INSTALL_INCLUDE_DIR}/metadef
    COMPONENT opensdk EXCLUDE_FROM_ALL FILES_MATCHING PATTERN "*.h"
)
install(DIRECTORY ${METADEF_PROTO_DIR}/ DESTINATION proto
    COMPONENT opensdk EXCLUDE_FROM_ALL FILES_MATCHING PATTERN "*.proto"
)

if (PACKAGE STREQUAL "opensdk")
    install(EXPORT graph_metadef-targets DESTINATION ${INSTALL_CONFIG_DIR}
        FILE graph_metadef-targets.cmake COMPONENT opensdk EXCLUDE_FROM_ALL
    )
    set(PKG_NAME graph_metadef)
    configure_package_config_file(${TOP_DIR}/cmake/config/pkg_config_template.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/graph_metadef-config.cmake
        INSTALL_DESTINATION ${INSTALL_CONFIG_DIR}
        PATH_VARS INSTALL_BASE_DIR INSTALL_INCLUDE_DIR INSTALL_LIBRARY_DIR INSTALL_RUNTIME_DIR INSTALL_CONFIG_DIR
        INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
    )
    unset(PKG_NAME)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/graph_metadef-config.cmake
        DESTINATION ${INSTALL_CONFIG_DIR} COMPONENT opensdk EXCLUDE_FROM_ALL
    )
endif()

if (ENABLE_OPEN_SRC)
    set(ARCH_LINX_PATH "${CMAKE_SYSTEM_PROCESSOR}-linux")

    install(TARGETS
                    lowering
                    graph
                    graph_base
                    register
                    aihac_ir
                    aihac_ir_register
                    aihac_symbolizer
        LIBRARY DESTINATION ${ARCH_LINX_PATH}/lib64 OPTIONAL COMPONENT packages EXCLUDE_FROM_ALL
        ARCHIVE DESTINATION ${ARCH_LINX_PATH}/lib64 OPTIONAL COMPONENT packages EXCLUDE_FROM_ALL
        RUNTIME DESTINATION ${ARCH_LINX_PATH}/bin OPTIONAL COMPONENT packages EXCLUDE_FROM_ALL
    )
endif()

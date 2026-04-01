# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (NOT DEFINED GE_METADEF_INC_DIR)
    # 开发环境编译兼容处理
    set(METADEF_DIR ${TOP_DIR}/metadef)
    set(GE_METADEF_DIR ${TOP_DIR}/air/graph_metadef)
    set(GE_METADEF_INC_DIR ${TOP_DIR}/air/inc/graph_metadef)
    set(AIR_CODE_DIR ${TOP_DIR}/air)
    set(RUNTIME_CODE_DIR ${TOP_DIR}/runtime)
else ()
    add_library(metadef_headers INTERFACE)
endif ()

# 影响sdk包导出cmake，增加时必须包含BUILD_INTERFACE
target_include_directories(metadef_headers INTERFACE
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/external>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/external/base>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/external/exe_graph>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/external/exe_graph/runtime>
    $<BUILD_INTERFACE:${GE_METADEF_INC_DIR}>
    $<BUILD_INTERFACE:${GE_METADEF_INC_DIR}/graph>
    $<BUILD_INTERFACE:${ASCEND_INSTALL_PATH}/pkg_inc>
    $<BUILD_INTERFACE:${ASCEND_INSTALL_PATH}/pkg_inc/base>
    $<BUILD_INTERFACE:${ASCEND_INSTALL_PATH}/pkg_inc/graph>
    $<BUILD_INTERFACE:${ASCEND_INSTALL_PATH}/pkg_inc/common>
    $<BUILD_INTERFACE:${ASCEND_INSTALL_PATH}/pkg_inc/graph/utils>
    $<BUILD_INTERFACE:${ASCEND_INSTALL_PATH}/pkg_inc/common/ge_common>
    $<BUILD_INTERFACE:${ASCEND_INSTALL_PATH}>
    $<BUILD_INTERFACE:${ASCEND_INSTALL_PATH}/include>
    $<BUILD_INTERFACE:${ASCEND_INSTALL_PATH}/include/external>
    $<BUILD_INTERFACE:${ASCEND_INSTALL_PATH}/include/base>
    $<BUILD_INTERFACE:${ASCEND_INSTALL_PATH}/include/graph>
    $<BUILD_INTERFACE:${ASCEND_INSTALL_PATH}/include/exe_graph>
    $<BUILD_INTERFACE:${ASCEND_INSTALL_PATH}/include/experiment/metadef>
    $<BUILD_INTERFACE:${ASCEND_INSTALL_PATH}/include/register>
    $<BUILD_INTERFACE:${AIR_CODE_DIR}/inc>
    $<BUILD_INTERFACE:${GE_METADEF_INC_DIR}/common>
    $<BUILD_INTERFACE:${GE_METADEF_INC_DIR}/exe_graph>
    $<BUILD_INTERFACE:${GE_METADEF_INC_DIR}/external>
    $<BUILD_INTERFACE:${GE_METADEF_INC_DIR}/external/graph>
    $<BUILD_INTERFACE:${GE_METADEF_INC_DIR}/external/exe_graph>
    $<BUILD_INTERFACE:${GE_METADEF_INC_DIR}/external/op_common>
    $<BUILD_INTERFACE:${GE_METADEF_INC_DIR}/graph/utils>
    $<BUILD_INTERFACE:${GE_METADEF_INC_DIR}/register>
    $<BUILD_INTERFACE:${RUNTIME_CODE_DIR}/include/external>

    # --------------------start------------------------
    $<BUILD_INTERFACE:${GE_METADEF_DIR}>
    $<BUILD_INTERFACE:${GE_METADEF_DIR}/third_party/transformer>
    $<BUILD_INTERFACE:${GE_METADEF_DIR}/third_party/transformer/inc>
    $<BUILD_INTERFACE:${GE_METADEF_DIR}/third_party/transformer/src>
    $<BUILD_INTERFACE:${GE_METADEF_DIR}/exe_graph>
    $<BUILD_INTERFACE:${GE_METADEF_DIR}/graph>
    $<BUILD_INTERFACE:${GE_METADEF_DIR}/register>
    $<BUILD_INTERFACE:${GE_METADEF_DIR}/register/op_tiling>
    # ---------------------end-----------------------
)

message(STATUS "inc graph_metadef")
set(SRC_GRAPH_DIR ${ASCEND_INSTALL_PATH}/include/graph)

if (EXISTS "${ASCEND_INSTALL_PATH}/include/graph")
    set(SRC_GRAPH_DIR ${ASCEND_INSTALL_PATH}/include/graph)
else ()
    set(SRC_GRAPH_DIR ${METADEF_DIR}/inc/external/graph)
endif()

message(STATUS "copy head files from ${SRC_GRAPH_DIR} to ${GE_METADEF_INC_DIR}/external/graph")

# 兼容以./方式包含其他头文件，如#include "./operator.h"
file(GLOB METADEF_HEAD_FILES "${SRC_GRAPH_DIR}/*.h")

# 安装包下是软连接，需要转换为真实路径
foreach(head_file IN LISTS METADEF_HEAD_FILES)
    get_filename_component(ABS_PATH_NAME ${head_file} REALPATH)
    get_filename_component(file_name ${ABS_PATH_NAME} NAME)
    if (NOT EXISTS "${GE_METADEF_INC_DIR}/external/graph/${file_name}")
        file(COPY ${ABS_PATH_NAME} DESTINATION ${GE_METADEF_INC_DIR}/external/graph)
    endif()
endforeach()

get_target_property(INCLUDE_PATHS metadef_headers INTERFACE_INCLUDE_DIRECTORIES)
foreach(line IN LISTS INCLUDE_PATHS)
    message(STATUS "metadef_headers ${line}")
endforeach()

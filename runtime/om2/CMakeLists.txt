# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE OM2_EXECUTOR_SRCS CONFIGURE_DEPENDS "${AIR_CODE_DIR}/runtime/om2/*.cc")

add_library(om2_executor SHARED
        ${OM2_EXECUTOR_SRCS}
        # 此处仅添加执行时依赖的源文件
        ${AIR_CODE_DIR}/base/common/helper/om2/zip_archive.cc
        ${AIR_CODE_DIR}/base/common/helper/om2/zip_mem_ioapi.cc
        ${AIR_CODE_DIR}/base/common/helper/om2/om2_utils.cc
)

target_compile_options(om2_executor PRIVATE ${AIR_COMMON_COMPILE_OPTION}
        $<$<STREQUAL:${CMAKE_CXX_COMPILER_ID},Clang>:-fblocks>
        -fno-strict-aliasing
)

target_include_directories(om2_executor PRIVATE
        ${ASCEND_INSTALL_PATH}/include/
        ${ASCEND_INSTALL_PATH}/pkg_inc/runtime/
        ${AIR_CODE_DIR}/runtime/om2/
        ${AIR_CODE_DIR}/base/
        ${AIR_CODE_DIR}/inc/
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}/inc/external
)

target_link_libraries(om2_executor PRIVATE
        ${AIR_COMMON_LINK_OPTION}
        intf_pub
        slog_headers
        msprof_headers
        adump_headers
        mmpa_headers
        metadef_headers
        runtime_headers
        error_manager
        c_sec
        runtime
        json
        minizip_static
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
)


# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB OPSKERNEL_SRC_CPP_1 RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "ops_store/*.cc")

add_library(opskernel SHARED
        ${OPSKERNEL_SRC_CPP_1})

target_include_directories(opskernel PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${NN_ENG_DIR}/inc/
        ${AIR_CODE_DIR}/inc)

target_compile_options(opskernel PRIVATE
        -Wfloat-equal
        -Werror
        -fexceptions
        -fno-common
        -fno-strict-aliasing)

target_compile_definitions(opskernel PRIVATE
        google=ascend_private
        $<$<STREQUAL:${BUILD_OPEN_PROJECT},True>:ONLY_COMPILE_BLUE>
        )

target_link_directories(opskernel PRIVATE
 	${ASCEND_INSTALL_PATH}/lib64
)

target_link_libraries(opskernel PRIVATE
        intf_pub
        mmpa_headers
        msprof_headers
        runtime_headers
        -O2
        c_sec
        unified_dlog
        graph
#        graph_base
        error_manager
        aicore_utils
        -Wl,--no-as-needed
        json
        -ldl)

install(TARGETS opskernel OPTIONAL
        EXPORT opskernel-targets
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
        )

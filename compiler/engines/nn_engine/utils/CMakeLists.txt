# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB COMM_SRC_CPP_1 RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "common/config_parser/*.cc"
        "common/format/*.cc"
        "common/graph/*.cc"
        "common/range_format_transfer/*.cc"
        "common/util/*.cc"
        "common/*.cc"
        "param_calculate/*.cc")

add_library(aicore_utils SHARED
        ${COMM_SRC_CPP_1}
        ${PROTO_HDRS})

target_include_directories(aicore_utils PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${NN_ENG_DIR}/inc/
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${CMAKE_BINARY_DIR}/proto/metadef_protos)

target_compile_options(aicore_utils PRIVATE
        -Werror
        -Wfloat-equal
        -fexceptions
        -fno-common
        -fno-strict-aliasing
        -Dgoogle=ascend_private)

target_compile_definitions(aicore_utils PRIVATE
        $<$<STREQUAL:${BUILD_OPEN_PROJECT},True>:ONLY_COMPILE_BLUE>
        )

target_link_directories(aicore_utils PRIVATE
 	${ASCEND_INSTALL_PATH}/lib64
)

target_link_libraries(aicore_utils
        PRIVATE
                intf_pub
                msprof_headers
                runtime_headers
                -O2
                -Wl,--no-as-needed
                c_sec
                static_mmpa
                unified_dlog
                exe_graph
                graph
#                graph_base
                ascend_protobuf
                error_manager
                register
                opp_registry
                runtime
                -Wl,--no-as-needed
                json
                -ldl
        PUBLIC
                air_headers
)

if (BUILD_OPEN_PROJECT OR ENABLE_OPEN_SRC)
    install(TARGETS aicore_utils OPTIONAL
            EXPORT aicore_utils-targets
            LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
            )
endif()
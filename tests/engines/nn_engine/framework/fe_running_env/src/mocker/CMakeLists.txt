# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE PROTO_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "${METADEF_PROTO_DIR}/task.proto"
        )
protobuf_generate(mocker_proto PROTO_SRCS PROTO_HDRS ${PROTO_LIST} TARGET)

set(FE_ENV_MOCKER_SRC_CPP
    mock_plugin_manager.cc
    mock_ffts_plugin_manager.cc
    mock_fftsplus_ops_kernel_builder.cc
)

add_library(fe_env_mocker SHARED
        ${FE_ENV_MOCKER_SRC_CPP})

add_dependencies(fe_env_mocker mocker_proto)

target_include_directories(fe_env_mocker PRIVATE
        ${AIR_CODE_DIR}/tests/engines/nn_engine
        ${AIR_CODE_DIR}/compiler/engines/nn_engine/inc
        ${AIR_CODE_DIR}/compiler/engines/nn_engine/optimizer
        ${AIR_CODE_DIR}/compiler
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/compiler/engines/ffts_engine
        ${CMAKE_BINARY_DIR}/proto/mocker_proto)

target_link_directories(fe_env_mocker PRIVATE
    ${ASCEND_INSTALL_PATH}/lib64
)

target_link_libraries(fe_env_mocker PUBLIC
        intf_llt_pub
        msprof_headers
        runtime_headers
        c_sec
        graph
#        graph_base
        register
        opp_registry
        slog_stub
        json
        ascend_protobuf
        )

target_compile_definitions(fe_env_mocker PRIVATE
    google=ascend_private
)
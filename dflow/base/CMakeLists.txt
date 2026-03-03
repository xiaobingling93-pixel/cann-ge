# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(proto_src_files
        proto/udf_attr.proto
        proto/udf_def.proto
)

protobuf_generate(data_flow_base_proto PROTO_SRCS PROTO_HDRS ${proto_src_files} TARGET)

set(DATA_FLOW_BASE_SRC_LIST
    ${PROTO_SRCS}
    utils/data_flow_utils.cc
    utils/process_utils.cc
    model/flow_model.cc
    model/pne_model.cc
    model/flow_model_om_loader.cc
    model/flow_model_om_saver.cc
    model/graph_model.cc
    model/model_relation.cc
    model/endpoint.cc
    deploy/deploy_planner.cc
    exec_runtime/execution_runtime.cc
)

############ libdata_flow_base.so ############
add_library(data_flow_base SHARED
    ${DATA_FLOW_BASE_SRC_LIST}
)

add_dependencies(data_flow_base data_flow_base_proto graphengine_protos)

target_compile_definitions(data_flow_base PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
)

target_compile_options(data_flow_base PRIVATE
    -O2
    -fno-common
    -Werror
    -Wextra
    -Wfloat-equal
    -fvisibility=default
)

target_include_directories(data_flow_base SYSTEM PRIVATE
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/data_flow_base_proto
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    ${AIR_CODE_DIR}/
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/dflow
    ${AIR_CODE_DIR}/dflow/inc
    ${AIR_CODE_DIR}/dflow/inc/data_flow
)

target_link_options(data_flow_base PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(data_flow_base PRIVATE
    intf_pub
    -Wl,--no-as-needed
    ascend_protobuf
    static_mmpa
    metadef_headers
    msprof_headers
    runtime_headers
    unified_dlog
    c_sec
    ge_common
    -Wl,--as-needed
    -lrt
    -ldl
    json
)

install(TARGETS data_flow_base OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)

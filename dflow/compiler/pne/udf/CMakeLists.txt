# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(UDF_COMPILER_SRC_LIST
    "udf_attr_utils.cc"
    "udf_model_builder.cc"
    "udf_model.cc"
    "udf_process_node_engine.cc"
)

############ libudf_compiler.so ############
add_library(udf_compiler SHARED
    ${UDF_COMPILER_SRC_LIST}
)

add_dependencies(udf_compiler
    graphengine_protos
    data_flow_base_proto
)

target_compile_definitions(udf_compiler PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
)

target_compile_options(udf_compiler PRIVATE
    -O2
    -fno-common
    -Werror
    -Wextra
    -Wfloat-equal
    -fvisibility=default
)

target_include_directories(udf_compiler SYSTEM PRIVATE
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    ${CMAKE_BINARY_DIR}/proto/data_flow_base_proto
)

target_include_directories(udf_compiler PRIVATE
        .
    ${AIR_CODE_DIR}
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/compiler
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}
)

target_link_options(udf_compiler PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(udf_compiler PRIVATE
    intf_pub
    msprof_headers
    static_mmpa
    ge_proto_common
    -Wl,--no-as-needed
    graph
    graph_base
    ge_common
    ge_common_base
    ge_compiler
    dflow_runner
    data_flow_base
    ascend_protobuf
    register
    c_sec
    error_manager
    unified_dlog
    runtime
    -Wl,--as-needed
    json
    -lrt
    -ldl
    crypto_static
)

install(TARGETS udf_compiler OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)

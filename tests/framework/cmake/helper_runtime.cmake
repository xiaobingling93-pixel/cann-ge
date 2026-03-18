# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# ---- Test coverage ----

if (NOT DEFINED PRODUCT_SIDE)
    set(PRODUCT_SIDE "host")
endif ()

set(GRPC_STUB_SRC
        ${AIR_CODE_DIR}/tests/depends/helper_runtime/src/deployer_client_stub.cc
        )

list(APPEND STUB_LIBS
    c_sec
    slog_stub
    runtime_stub
    mmpa_stub
    profiler_stub
    hccl_stub
    error_manager
    ascend_protobuf
    json
)

# ---- Target : helper runtime ----
set(GRPC_PROTO_LIST
    "${AIR_CODE_DIR}/dflow/deployer/proto/deployer.proto"
)

protobuf_generate_grpc(deployer GRPC_PROTO_SRCS GRPC_PROTO_HDRS ${GRPC_PROTO_LIST} "--proto_path=${METADEF_PROTO_DIR}")

add_library(helper_runtime SHARED
    ${GRPC_PROTO_SRCS}
    ${GRPC_PROTO_HDRS}
)

target_compile_definitions(helper_runtime PRIVATE
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    google=ascend_private
)

target_include_directories(helper_runtime PUBLIC
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}/proto_grpc/deployer
    ${AIR_CODE_DIR}
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/runtime/v1
    ${AIR_CODE_DIR}/compiler
    ${AIR_CODE_DIR}/runtime
    ${AIR_CODE_DIR}/runtime/v2
    ${AIR_CODE_DIR}/dflow/inc
    ${AIR_CODE_DIR}/dflow/inc/data_flow
    ${AIR_CODE_DIR}/dflow/deployer
    ${CMAKE_BINARY_DIR}/proto
    ${CMAKE_BINARY_DIR}/opensrc/json/include
    ${CMAKE_BINARY_DIR}/grpc_build-prefix/src/grpc_build/include
    ${CMAKE_BINARY_DIR}/protoc_grpc_build-prefix/src/protoc_grpc_build/include
    ${CMAKE_BINARY_DIR}/proto/ge
    ${CMAKE_BINARY_DIR}/proto/ge/proto
    ${CMAKE_BINARY_DIR}/proto/data_flow_protos
    ${CMAKE_BINARY_DIR}/proto/data_flow_base_proto
    ${CMAKE_BINARY_DIR}/proto/data_flow_base_proto/proto
    ${AIR_CODE_DIR}/dflow
    ${AIR_CODE_DIR}/dflow/inc
    ${AIR_CODE_DIR}/dflow/inc/data_flow
)

target_compile_options(helper_runtime PRIVATE
    -g
    -Werror=format
)

target_link_libraries(helper_runtime PUBLIC
    intf_llt_pub
    mmpa_headers
    slog_headers
    metadef_headers
    runtime_headers
    cce_headers
    datagw_headers
    ascendcl_headers
    ascend_hal_headers
    adump_headers
    flow_graph_protos_obj
    gRPC::grpc++
    gRPC::grpc
    -Wl,--no-as-needed
    ${STUB_LIBS}
    memset_stubs
    dl
    -Wl,--as-needed
    -lrt -ldl -lpthread
    data_flow_base
)

# ---- Target : helper runtime no_grpc ----
set(PROTO_LIST
        "${AIR_CODE_DIR}/runtime/proto/deployer.proto"
        )

protobuf_generate(deployer PROTO_SRCS PROTO_HDRS ${GRPC_PROTO_LIST} "--proto_path=${METADEF_PROTO_DIR}")

add_library(helper_runtime_no_grpc SHARED
        ${GRPC_STUB_SRC}
        ${PROTO_SRCS}
        ${PROTO_HDRS}
        )

target_compile_definitions(helper_runtime_no_grpc PRIVATE
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
        google=ascend_private
        )

target_include_directories(helper_runtime_no_grpc PUBLIC
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/inc/framework
        ${CMAKE_BINARY_DIR}/proto_grpc/deployer
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/runtime/v1
        ${AIR_CODE_DIR}/runtime
        ${AIR_CODE_DIR}/runtime/v2
        ${AIR_CODE_DIR}/dflow/deployer
        ${CMAKE_BINARY_DIR}/proto
        ${CMAKE_BINARY_DIR}/opensrc/json/include
        ${CMAKE_BINARY_DIR}/grpc_build-prefix/src/grpc_build/include
        ${CMAKE_BINARY_DIR}/protoc_grpc_build-prefix/src/protoc_grpc_build/include
        ${CMAKE_BINARY_DIR}/proto/ge
        ${CMAKE_BINARY_DIR}/proto/ge/proto
        ${CMAKE_BINARY_DIR}/proto/data_flow_protos
        ${CMAKE_BINARY_DIR}/proto/data_flow_base_proto
        ${CMAKE_BINARY_DIR}/proto/data_flow_base_proto/proto
        ${AIR_CODE_DIR}/dflow
        ${AIR_CODE_DIR}/dflow/inc
        ${AIR_CODE_DIR}/dflow/inc/data_flow
        )

target_compile_options(helper_runtime_no_grpc PRIVATE
        -g
        -Werror=format
        )

target_link_libraries(helper_runtime_no_grpc PUBLIC
        intf_llt_pub
        mmpa_headers
        slog_headers
        metadef_headers
        -Wl,--no-as-needed
        ${STUB_LIBS}
        dl
        -Wl,--as-needed
        -lrt -ldl -lpthread
        data_flow_base
        )

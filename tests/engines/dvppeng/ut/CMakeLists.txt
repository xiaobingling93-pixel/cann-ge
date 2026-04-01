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
    ${METADEF_PROTO_DIR}/task.proto
)

set(proto_dvpp_src_files
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/proto/dvpp_attr.proto
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/proto/dvpp_node_def.proto
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/proto/dvpp_tensor.proto
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/proto/dvpp_tensor_shape.proto
)

protobuf_generate(dvpp PROTO_SRCS PROTO_HDRS ${proto_src_files})
protobuf_generate(dvpp PROTO_DVPP_SRCS PROTO_DVPP_HDRS ${proto_dvpp_src_files})

set(dvpp_engine_src_file
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/adapter/dvpp_engine.cc
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/adapter/dvpp_graph_optimizer.cc
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/adapter/dvpp_ops_kernel_builder.cc
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/adapter/dvpp_ops_kernel_info_store.cc
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/ascend910b/dvpp_builder_910b.cc
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/ascend910b/dvpp_info_store_910b.cc
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/ascend910b/dvpp_optimizer_910b.cc
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/common/dvpp_builder.cc
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/common/dvpp_chip_capability.cc
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/common/dvpp_info_store.cc
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/common/dvpp_ops_checker.cc
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/common/dvpp_ops_lib.cc
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/common/dvpp_optimizer.cc
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/util/util.cc
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/util/json_file_operator.cc
    ${PROTO_HDRS}
    ${PROTO_DVPP_SRCS}
)

set(local_lib_inc_path
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/adapter
    ${CMAKE_CURRENT_SOURCE_DIR}/ascend910b
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/util
    ${CMAKE_CURRENT_SOURCE_DIR}/proto
    ${CMAKE_BINARY_DIR}/proto/dvpp
    ${CMAKE_BINARY_DIR}/proto/dvpp/proto
)

set(dvpp_engine_src_test_file
    ./ut_main.cpp
    ./dvpp_engine_ut.cpp
    ./dvpp_graph_optimizer_ut.cpp
    ./dvpp_ops_kernel_builder_ut.cpp
    ./dvpp_ops_kernel_info_store_ut.cpp
    ../stub/stub.cpp
    )

add_executable(dvpp_engine_utest
    ${dvpp_engine_src_file}
    ${dvpp_engine_src_test_file}
)

target_include_directories(dvpp_engine_utest PRIVATE
    ${AIR_CODE_DIR}/compiler/graphcompiler/engines/
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/adapter
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/common
    ${AIR_CODE_DIR}/compiler/engines/dvpp_engine/util
    ${AIR_CODE_DIR}/tests/engines/dvppeng/stub
    ${AIR_CODE_DIR}/tests/engines/dvppeng
    ${AIR_CODE_DIR}
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/framework
    ${local_lib_inc_path}
    ${CMAKE_BINARY_DIR}/proto/dvpp/
)

target_compile_definitions(dvpp_engine_utest PRIVATE
    RUN_TEST
    google=ascend_private
)

target_link_options(dvpp_engine_utest PRIVATE
    -Wl,-rpath,${ASCEND_COMPILER_DIR}
)

target_link_directories(dvpp_engine_utest PRIVATE
    ${ASCEND_INSTALL_PATH}/lib64
)

target_link_libraries(dvpp_engine_utest PRIVATE
    intf_llt_pub
    runtime_headers
    msprof_headers
    cce_headers
    air_headers
    -Wl,--as-needed
    GTest::gtest
    error_manager
    graph
    metadef
    graph_base
    ascend_protobuf
    register
    opp_registry
    c_sec
    mmpa
    json
    -Wl,--no-as-needed
    unified_dlog
    cce_headers
    -ldl
    -lrt
    -lpthread
    -lgcov
)

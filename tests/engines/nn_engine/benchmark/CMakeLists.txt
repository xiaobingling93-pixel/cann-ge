# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

SET(FE_ROOT_PATH ${AIR_CODE_DIR}/compiler/engines/nn_engine)
SET(BENCHMARK_ROOT_PATH ${AIR_CODE_DIR}/tests/engines/nn_engine/benchmark)

file(GLOB_RECURSE PROTO_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "${METADEF_PROTO_DIR}/task.proto"
        "${METADEF_PROTO_DIR}/ge_ir.proto"
        "${METADEF_PROTO_DIR}/om.proto"
        "${METADEF_PROTO_DIR}/insert_op.proto"
        "${METADEF_PROTO_DIR}/ge_ir_mobile.proto"
        "${METADEF_PROTO_DIR}/task_mobile.proto"
        )


protobuf_generate(fe_benchmark PROTO_SRCS PROTO_HDRS ${PROTO_LIST})
file(GLOB_RECURSE FE_SOURCE_CODE_SRCS CONFIGURE_DEPENDS "${FE_ROOT_PATH}/*.cc")
file(GLOB_RECURSE FE_BENCHMARK_SRCS CONFIGURE_DEPENDS "${BENCHMARK_ROOT_PATH}/*.cc")
# compile for fe_benchmark
add_executable(fe_benchmark
        ${PROTO_HDRS}
        ${FE_SOURCE_CODE_SRCS}
        ${FE_BENCHMARK_SRCS}
        )

target_include_directories(fe_benchmark PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${FE_ROOT_PATH}/executor
        ${FE_ROOT_PATH}/inc
        ${FE_ROOT_PATH}/optimizer
        ${FE_ROOT_PATH}/utils
        ${FE_ROOT_PATH}/fusion
        ${FE_ROOT_PATH}/opskernel
        ${AIR_CODE_DIR}/compiler/engines/ffts_engine
        ${AIR_CODE_DIR}/compiler/engines/ffts_engine/inc
        ${AIR_CODE_DIR}/tests/engines/nn_engine/depends
        ${AIR_CODE_DIR}/tests/engines/nn_engine/graph_constructor
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/inc/framework
        ${GTEST_INCLUDE}
        ${CMAKE_BINARY_DIR}/proto/fe_benchmark)

target_compile_options(fe_benchmark PRIVATE
        -D__OPTIMIZER_KT_TEST__
        -DDAVINCI_CLOUD
        -Dgoogle=ascend_private
        -O2)

target_compile_definitions(fe_benchmark PRIVATE
        $<$<STREQUAL:${BUILD_OPEN_PROJECT},True>:ONLY_COMPILE_BLUE>)

target_link_directories(fe_benchmark PRIVATE
    ${ASCEND_INSTALL_PATH}/lib64
)

target_link_libraries(fe_benchmark PRIVATE
        intf_llt_pub
        benchmark::benchmark
        msprof_headers
        runtime_headers
        -Wl,--no-as-needed
        -lrt -ldl -lpthread -lgcov
        graph
        graph_base
        register
        opp_registry
        runtime_stub_fe
        ascend_protobuf
        c_sec
        platform
        compress
        error_manager
        json
        c_sec_static
        mmpa_stub
        slog_stub
        )

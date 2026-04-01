# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

execute_process(
    COMMAND cp -rf ${CMAKE_CURRENT_SOURCE_DIR}/../stub/config ${CMAKE_CURRENT_BINARY_DIR}/../stub
    COMMAND cp -rf ${CMAKE_CURRENT_SOURCE_DIR}/../stub/config ${CMAKE_CURRENT_BINARY_DIR}/../ut
)

set(proto_src_files
    ${METADEF_PROTO_DIR}/tensorflow/types.proto
    ${METADEF_PROTO_DIR}/tensorflow/graph.proto
    ${METADEF_PROTO_DIR}/tensorflow/node_def.proto
    ${METADEF_PROTO_DIR}/tensorflow/attr_value.proto
    ${METADEF_PROTO_DIR}/tensorflow/tensor.proto
    ${METADEF_PROTO_DIR}/tensorflow/resource_handle.proto
    ${METADEF_PROTO_DIR}/tensorflow/tensor_shape.proto
    ${METADEF_PROTO_DIR}/tensorflow/function.proto
    ${METADEF_PROTO_DIR}/tensorflow/op_def.proto
    ${METADEF_PROTO_DIR}/tensorflow/versions.proto
    ${METADEF_PROTO_DIR}/task.proto
    ${AIR_CODE_DIR}/compiler/engines/cpu_engine/tf_engine/proto/fwk_adapter.proto
)

set(proto_context_src_files
    ${METADEF_PROTO_DIR}/aicpu/cpu_attr.proto
    ${METADEF_PROTO_DIR}/aicpu/cpu_node_def.proto
    ${METADEF_PROTO_DIR}/aicpu/cpu_tensor.proto
    ${METADEF_PROTO_DIR}/aicpu/cpu_tensor_shape.proto
)

protobuf_generate(aicpu PROTO_SRCS PROTO_HDRS ${proto_src_files})
protobuf_generate(aicpu PROTO_CONTEXT_SRCS PROTO_CONTEXT_HDRS ${proto_context_src_files})

set(engine_common_src_path
    ${AIR_CODE_DIR}/compiler/engines/cpu_engine/common
)

set(engine_common_src_files
    ${engine_common_src_path}/aicpu_ops_kernel_info_store/aicpu_ops_kernel_info_store.cc
    ${engine_common_src_path}/aicpu_ops_kernel_info_store/kernel_info.cc
    ${engine_common_src_path}/aicpu_graph_optimizer/aicpu_graph_optimizer.cc
    ${engine_common_src_path}/pass/concat_from_sequence_pass.cc
    ${engine_common_src_path}/aicpu_graph_optimizer/graph_optimizer_utils.cc
    ${engine_common_src_path}/aicpu_graph_optimizer/optimizer.cc
    ${engine_common_src_path}/aicpu_ops_kernel_builder/aicpu_ops_kernel_builder.cc
    ${engine_common_src_path}/aicpu_ops_kernel_builder/kernel_builder.cc
    ${engine_common_src_path}/config/config_file.cc
    ${engine_common_src_path}/config/ops_json_file.cc
    ${engine_common_src_path}/util/util.cc
    ${engine_common_src_path}/error_code/error_code.cc
    ${engine_common_src_path}/engine/base_engine.cc
    ${AIR_CODE_DIR}/compiler/engines/cpu_engine/cpu_engine/common/util/cpu_engine_util.cc
    ${engine_common_src_path}/config/ops_parallel_rule_json_file.cpp
    ${PROTO_HDRS}
    ${PROTO_CONTEXT_HDRS}
)

set(engine_common_stub_src_path "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(engine_common_stub_src_files
    ${engine_common_stub_src_path}/stub/platform_info_stub.cpp
    ${engine_common_stub_src_path}/stub/ops_kernel_builder_registry_stub.cpp
)

set(engine_common_src_test_files
    ut_main.cpp
)

set(engine_common_includes
    ${AIR_CODE_DIR}/tests/engines/cpueng/stub/
    ${AIR_CODE_DIR}/compiler/engines/cpu_engine
    ${AIR_CODE_DIR}/compiler/engines/cpu_engine/common
    ${AIR_CODE_DIR}/compiler/engines/nn_engine/inc
    ${AIR_CODE_DIR}/compiler/engines/cpu_engine/cpu_engine/common/util
    ${CMAKE_BINARY_DIR}/proto/aicpu
    ${CMAKE_BINARY_DIR}/proto/aicpu/proto
    ${OP_PROTO_DIR}
    ${ASCEND_INSTALL_PATH}/pkg_inc
    ${ASCEND_INSTALL_PATH}/pkg_inc/base
)

set(engine_src_path ${AIR_CODE_DIR}/compiler/engines/cpu_engine)

set(tf_engine_src_files
    ${engine_src_path}/tf_engine/engine/tf_engine.cc
    ${engine_src_path}/tf_engine/tf_kernel_info/tf_kernel_info.cc
    ${engine_src_path}/tf_engine/tf_optimizer/tf_optimizer.cc
    ${engine_src_path}/tf_engine/tf_optimizer/tf_optimizer_utils.cc
    ${engine_src_path}/tf_engine/tf_optimizer/tensorflow_util.cc
    ${engine_src_path}/tf_engine/tf_optimizer/tf_function_builder.cc
    ${engine_src_path}/tf_engine/util/tf_util.cc
    ${engine_src_path}/tf_engine/tf_kernel_builder/tf_kernel_builder.cc
    ${engine_src_path}/tf_engine/ir2tf/ir2tf_parser_factory.cc
    ${engine_src_path}/tf_engine/ir2tf/ir2tf_base_parser.cc
    ${engine_src_path}/tf_engine/config/ir2tf_json_file.cc
    ${engine_src_path}/tf_engine/tf_kernel_builder/tf_ops_kernel_builder.cc
)

set(tf_engine_src_test_files
    tf_engine/tf_engine_ut.cpp
    tf_engine/tf_kernel_info_ut.cpp
    tf_engine/ir2tf_ut.cpp
    tf_engine/tf_optimizer_ut.cpp
    tf_engine/tf_kernel_builder_ut.cpp
    tf_engine/tf_ops_kernel_builder_ut.cpp
)

set(tf_engine_includes
    ${AIR_CODE_DIR}/compiler/engines/cpu_engine/tf_engine
    ${AIR_CODE_DIR}/compiler/engines/nn_engine/inc
    ${CMAKE_BINARY_DIR}/proto/aicpu/proto
    ${CMAKE_BINARY_DIR}/proto/aicpu/
)

set(cpu_engine_src_files
    ${engine_src_path}/cpu_engine/aicpu_engine/engine/aicpu_engine.cpp
    ${engine_src_path}/cpu_engine/aicpu_engine/kernel_info/aicpu_kernel_info.cpp
    ${engine_src_path}/cpu_engine/aicpu_engine/kernel_info/aicpu_cust_kernel_info.cpp
    ${engine_src_path}/cpu_engine/aicpu_engine/optimizer/aicpu_optimizer.cpp
    ${engine_src_path}/cpu_engine/aicpu_engine/kernel_builder/aicpu_kernel_builder.cpp
    ${engine_src_path}/cpu_engine/common/kernel_builder/cpu_kernel_builder.cpp
    ${engine_src_path}/cpu_engine/common/optimizer/cpu_optimizer.cpp
    ${engine_src_path}/cpu_engine/aicpu_const_folding/folding.cc
)

if (BUILD_OPEN_PROJECT OR ENABLE_OPEN_SRC)
    set(cpu_engine_includes
        ${AIR_CODE_DIR}/tests/engines/cpueng/stub/
        ${AIR_CODE_DIR}/compiler/engines/cpu_engine/cpu_engine/
        ${AIR_CODE_DIR}/compiler/engines/cpu_engine
        ${AIR_CODE_DIR}/compiler/engines/cpu_engine/common
        ${AIR_CODE_DIR}/compiler/engines/nn_engine/inc
        ${AIR_CODE_DIR}/compiler/engines/cpu_engine/cpu_engine/common/util
        ${CMAKE_BINARY_DIR}/proto/aicpu
        ${CMAKE_BINARY_DIR}/proto/aicpu/proto
        ${METADEF_DIR}/inc
    )
else ()
    set(cpu_engine_includes
        ${AIR_CODE_DIR}/tests/engines/cpueng/stub/
        ${AIR_CODE_DIR}/compiler/engines/cpu_engine/cpu_engine/
        ${AIR_CODE_DIR}/compiler/engines/cpu_engine
        ${AIR_CODE_DIR}/compiler/engines/cpu_engine/common
        ${AIR_CODE_DIR}/compiler/engines/nn_engine/inc
        ${AIR_CODE_DIR}/compiler/engines/cpu_engine/cpu_engine/common/util
        ${CMAKE_BINARY_DIR}/proto/aicpu
        ${CMAKE_BINARY_DIR}/proto/aicpu/proto
        ${METADEF_DIR}/inc
        ${TOP_DIR}/inc/aicpu/cpu_kernels
        ${TOP_DIR}/inc/external/aicpu
    )
endif ()

set(cpu_engine_src_test_files
    cpu_engine/aicpu_engine_ut.cpp
    cpu_engine/aicpu_kernel_info_ut.cpp
    cpu_engine/aicpu_cust_kernel_builder_ut.cpp
    cpu_engine/aicpu_optimizer_ut.cpp
    cpu_engine/aicpu_kernel_builder_ut.cpp
    cpu_engine/aicpu_pass_ut.cpp
    cpu_engine/folding_ut.cpp
)

set(host_engine_src_files
    ${engine_src_path}/cpu_engine/hostcpu_engine/engine/hostcpu_engine.cpp
    ${engine_src_path}/cpu_engine/hostcpu_engine/kernel_info/hostcpu_kernel_info.cpp
    ${engine_src_path}/cpu_engine/hostcpu_engine/optimizer/hostcpu_optimizer.cpp
    ${engine_src_path}/cpu_engine/hostcpu_engine/kernel_builder/hostcpu_kernel_builder.cpp
    ${engine_src_path}/cpu_engine/hostcpu_engine/kernel_builder/hostcpu_ops_kernel_builder.cpp
    ${engine_src_path}/cpu_engine/common/kernel_builder/cpu_kernel_builder.cpp
    ${engine_src_path}/cpu_engine/common/optimizer/cpu_optimizer.cpp
)

set(host_engine_src_test_files
    cpu_engine/hostcpu_engine_ut.cpp
    cpu_engine/hostcpu_kernel_info_ut.cpp
    cpu_engine/hostcpu_kernel_builder_ut.cpp

)

add_executable(tf_engine_utest
    ${engine_common_src_files}
    ${engine_common_src_test_files}
    ${tf_engine_src_files}
    ${tf_engine_src_test_files}
    ${engine_common_stub_src_files}
)

add_executable(cpu_engine_utest
    ${engine_common_src_files}
    ${engine_common_src_test_files}
    ${cpu_engine_src_files}
    ${cpu_engine_src_test_files}
    ${engine_common_stub_src_files}
    ${engine_common_stub_src_path}/stub/aicpu/aicpu_stub.cpp
)

add_executable(host_engine_utest
    ${engine_common_src_files}
    ${engine_common_src_test_files}
    ${host_engine_src_files}
    ${host_engine_src_test_files}
    ${engine_common_stub_src_files}
)

target_include_directories(tf_engine_utest PRIVATE
    ${engine_common_includes}
    ${tf_engine_includes}
    ${cpu_engine_includes}
)

target_include_directories(cpu_engine_utest PRIVATE
    ${engine_common_includes}
    ${cpu_engine_includes}
)

target_include_directories(host_engine_utest PRIVATE
    ${engine_common_includes}
    ${cpu_engine_includes}
)

target_compile_definitions(tf_engine_utest PRIVATE
    RUN_TEST
    google=ascend_private
)

target_compile_definitions(cpu_engine_utest PRIVATE
    RUN_TEST
    google=ascend_private
)

target_compile_definitions(host_engine_utest PRIVATE
    RUN_TEST
    google=ascend_private
)

target_link_options(tf_engine_utest PRIVATE
    -Wl,-rpath,${ASCEND_COMPILER_DIR}
)

target_link_options(cpu_engine_utest PRIVATE
    -Wl,-rpath,${ASCEND_COMPILER_DIR}
)

target_link_options(host_engine_utest PRIVATE
    -Wl,-rpath,${ASCEND_COMPILER_DIR}
)

target_link_directories(tf_engine_utest PRIVATE
    ${ASCEND_INSTALL_PATH}/lib64
)

target_link_directories(cpu_engine_utest PRIVATE
    ${ASCEND_INSTALL_PATH}/lib64
)

target_link_directories(host_engine_utest PRIVATE
    ${ASCEND_INSTALL_PATH}/lib64
)

target_link_libraries(tf_engine_utest PRIVATE
    intf_llt_pub
    msprof_headers
    air_headers
    datagw_headers
    -Wl,--as-needed
    GTest::gtest
    slog_aicpu_stub
    stub_for_aicpu
    graph
    metadef
    graph_base
    error_manager
    register
    opp_registry
    ascend_protobuf
    platform_headers
    c_sec
    json
    runtime_aicpu_stub
    -Wl,--no-as-needed
    unified_dlog
    -ldl
    -lrt
    -lpthread
    -lgcov
    cce_headers
    platform
    ascendalog
    mmpa
)

if (BUILD_OPEN_PROJECT OR ENABLE_OPEN_SRC)
    target_link_libraries(cpu_engine_utest PRIVATE
        intf_llt_pub
        msprof_headers
        air_headers
        datagw_headers
        aicpu_headers
        -Wl,--as-needed
        stub_for_aicpu
        error_manager
        graph
        metadef
        graph_base
        register
        opp_registry
        GTest::gtest
        ascend_protobuf
        platform_headers
        c_sec
        json
        runtime_aicpu_stub
        -Wl,--no-as-needed
        unified_dlog
        -ldl
        -lrt
        -lpthread
        -lgcov
        cce_headers
        platform
        ascendalog
        mmpa
    )
else ()
    target_link_libraries(cpu_engine_utest PRIVATE
        intf_llt_pub
        msprof_headers
        air_headers
        datagw_headers
        -Wl,--as-needed
        stub_for_aicpu
        error_manager
        graph
        metadef
        graph_base
        register
        opp_registry
        GTest::gtest
        ascend_protobuf
        platform_headers
        c_sec
        json
        runtime_aicpu_stub
        -Wl,--no-as-needed
        unified_dlog
        -ldl
        -lrt
        -lpthread
        -lgcov
        cce_headers
        platform
        ascendalog
        mmpa
    )
endif ()

target_link_libraries(host_engine_utest PRIVATE
    intf_llt_pub
    msprof_headers
    air_headers
    datagw_headers
    -Wl,--as-needed
    graph
    metadef
    graph_base
    slog_aicpu_stub
    stub_for_aicpu
    c_sec
    error_manager
    register
    opp_registry
    GTest::gtest
    ascend_protobuf
    platform_headers
    json
    runtime_aicpu_stub
    -Wl,--no-as-needed
    unified_dlog
    -ldl
    -lrt
    -lpthread
    -lgcov
    cce_headers
    platform
    ascendalog
    mmpa
)
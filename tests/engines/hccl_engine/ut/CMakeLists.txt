# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CMAKE_BUILD_TYPE Debug)

set(proto_src_files
    ${METADEF_PROTO_DIR}/task.proto
)

protobuf_generate(hcce PROTO_SRCS PROTO_HDRS ${proto_src_files})

set(hccl_engine_src_file
    #/hccl/hcom_graph_adaptor/ge_plugin
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/hcom_plugin.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/plugin_manager.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/common/hcom_op_utils.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/common/op_fusion_base.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/common/ops_kernel_info_store_base.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/common/ops_kernel_builder_base.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/common/hcom_base_funs.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/common/hcom_acl_adapter.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/common/hcom_log.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/common/dlhccl_function.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/common/adapter_dlhcclfunc.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/hcom/hcom_all_reduce_fusion.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/hcom/hcom_alltoallvc_fusion.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/hcom/hcom_allgather_fusion.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/hcom/hcom_reducescatter_fusion.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/hcom/hcom_broadcast_fusion.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/hcom/hcom_reduce_fusion.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/hcom/hcom_ops_kernel_builder.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/hcom/hcom_fusion_optimizer.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/hcom/hcom_graph_optimizer.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/hcom/hcom_ops_kernel_info_store.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/hcom/offline_build_config_parse.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/hcom/hcom_graph_mc2.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/hcom/hcom_graph_superkernel.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/hcom/lowering/hcom_build_graph.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/hcom/lowering/hcom_launch_kernel.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/hcom/lowering/hcom_node_converter.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/auto_tuning/auto_tuning_hcom_all_reduce_fusion.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/auto_tuning/auto_tuning_hcom_graph_optimizer.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/auto_tuning/auto_tuning_hcom_ops_kernel_builder.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/auto_tuning/auto_tuning_hcom_ops_kernel_info_store.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/auto_tuning/auto_tuning_plugin.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/calculation_tool/communication.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/calculation_tool/cluster.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/calculation_tool/evaluator.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/calculation_tool/layers.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/calculation_tool/model.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/calculation_tool/topology.cc

    #/hccl/hcom_graph_adaptor/hcom_executor
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/hcom_executor/hcom_executor.cc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/hcom_executor/executor_message.cc

    #/hccl/hcom_gradient_split_tune
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_gradient_split_tune/src/gradient_split_tune.cc
    ${PROTO_HDRS}
)

add_library(hcce_ut_llt SHARED ${hccl_engine_src_file})

set(hcce_ut_llt_stub
    ${CMAKE_CURRENT_SOURCE_DIR}/../stub/hccl_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../stub/llt_hccl_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../stub/llt_hccl_stub_gert.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../stub/llt_hccl_stub_ge.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../stub/llt_hccl_stub_sal.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../stub/llt_hccl_stub_hccl.cc
)

set(hccl_engine_src_test_file
    ${CMAKE_CURRENT_SOURCE_DIR}/ut_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ut_hcom_plugin.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ut_hcom_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ut_hcom_launch_kernel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ut_hcom_lowering.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ut_hccl_kernel_builder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ut_hccl_graph_optimizer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ut_auto_tuning_hcom_all_reduce_fusion.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ut_autotuning_hccl_kernel_builder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ut_gradient_split_tune.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ut_gradient_split.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ut_hccl_kernel_info.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ut_superFastKernel.cc
    ${hcce_ut_llt_stub}
)

add_executable(hccl_engine_utest
    ${hccl_engine_src_test_file}
)

set(src_path
    ${AIR_CODE_DIR}
    ${AIR_CODE_DIR}/base/metadef/pkg_inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/framework
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${AIR_CODE_DIR}/inc/graph_metadef/register
    ${AIR_CODE_DIR}/inc/graph_metadef/graph
    ${AIR_CODE_DIR}/inc/graph_metadef/exe_graph
    ${AIR_CODE_DIR}/inc/graph_metadef/exe_graph/lowering
    ${AIR_CODE_DIR}/inc/graph_metadef/external/hcom
    ${AIR_CODE_DIR}/inc/graph_metadef/external
    ${CMAKE_BINARY_DIR}/proto/hcom_graph_adaptor_proto

    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/inc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_gradient_split_tune/inc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/inc
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/hcom_executor
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/common
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/hcom
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/hcom/lowering
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/auto_tuning
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin
    ${AIR_CODE_DIR}/compiler/engines/hccl_engine/hcom_graph_adaptor/ge_plugin/calculation_tool

    ${ASCEND_INSTALL_PATH}
    ${ASCEND_INSTALL_PATH}/include
    ${ASCEND_INSTALL_PATH}/include/external
    ${ASCEND_INSTALL_PATH}/include/mmpa
    ${ASCEND_INSTALL_PATH}/include/hccl
    ${ASCEND_INSTALL_PATH}/pkg_inc
    ${ASCEND_INSTALL_PATH}/pkg_inc/hccl
    ${ASCEND_INSTALL_PATH}/pkg_inc/base
    ${ASCEND_INSTALL_PATH}/pkg_inc/profiling
    ${ASCEND_INSTALL_PATH}/pkg_inc/aicpu
    ${ASCEND_INSTALL_PATH}/pkg_inc/dump
    ${ASCEND_INSTALL_PATH}/pkg_inc/runtime
    ${ASCEND_INSTALL_PATH}/include/msprof
    ${ASCEND_INSTALL_PATH}/include/adump/external
    ${ASCEND_INSTALL_PATH}/include/slog/toolchain

    ${ASCEND_3RD_LIB_PATH}/json/include

    ${ASCEND_3RD_LIB_PATH}/protoc/include

    ${CMAKE_BINARY_DIR}/proto/hcce

    ${CMAKE_CURRENT_SOURCE_DIR}/../stub
    ${AIR_CODE_DIR}/tests/engines/hccl_engine/third_party/mockcpp/mockcpp-2.7/3rdparty
    ${AIR_CODE_DIR}/tests/engines/hccl_engine/third_party/mockcpp/mockcpp-2.7/include

    ${CMAKE_BINARY_DIR}/proto/metadef_protos
)

target_include_directories(hcce_ut_llt PRIVATE ${src_path})

target_compile_definitions(hcce_ut_llt PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    HALF_T=float
    RUN_TEST
    google=ascend_private
    -DCCL_LLT
)

target_include_directories(hccl_engine_utest PRIVATE ${src_path})

target_compile_definitions(hccl_engine_utest PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    HALF_T=float
    RUN_TEST
    google=ascend_private
    -DCCL_LLT
)

target_link_options(hccl_engine_utest PRIVATE
    -Wl,-rpath,${ASCEND_COMPILER_DIR}
    -Wl,-rpath,${ASCEND_3RD_LIB_PATH}/gtest_shared/lib64
    -Wl,-rpath,${AIR_CODE_DIR}/build/graph_metadef/register
    -Wl,-rpath,${AIR_CODE_DIR}/build/graph_metadef/graph
)

target_link_directories(hcce_ut_llt PRIVATE
	${ASCEND_INSTALL_PATH}/lib64
)

target_link_directories(hccl_engine_utest PRIVATE
    ${ASCEND_INSTALL_PATH}/lib64
)

find_library(REGISTER_LIB NAMES libregister.so PATH_SUFFIXES lib64)

find_library(LIBRARY_MOCKCPP libmockcpp.a ${AIR_CODE_DIR}/tests/engines/hccl_engine/third_party/mockcpp/mockcpp-2.7/build)

target_link_libraries(hcce_ut_llt PRIVATE
    intf_llt_pub
    runtime
    msprof_headers
    -Wl,--as-needed
    GTest::gtest GTest::gtest_main GTest::gmock
    ${LIBRARY_MOCKCPP}
    error_manager
    graph
    graph_base
    metadef
    ascend_protobuf
    hcomm
 	${REGISTER_LIB}
    c_sec
    platform
    mmpa
    json
    -Wl,--no-as-needed
    -ldl
    -lrt
    -lpthread
    -lgcov
)

target_link_libraries(hccl_engine_utest PRIVATE
    intf_llt_pub
    runtime
    msprof_headers
    -Wl,--as-needed
    GTest::gtest GTest::gtest_main GTest::gmock
    ${LIBRARY_MOCKCPP}
    error_manager
    graph
    graph_base
    metadef
    ascend_protobuf
    hcomm
 	${REGISTER_LIB}
    hcce_ut_llt
    c_sec
    platform
    ascendcl
    mmpa
    json
    -Wl,--no-as-needed
    -ldl
    -lrt
    -lpthread
    -lgcov
)

add_dependencies(hccl_engine_utest hcce_ut_llt)
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
protobuf_generate(rts_engine_proto PROTO_SRCS PROTO_HDRS ${proto_src_files})

set(rts_engine_src_file
    ${CMAKE_CURRENT_SOURCE_DIR}/../stub/platform_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../stub/runtime_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../stub/rts_engine_utest_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rts_engine_utest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rts_engine_utest_op.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rts_engine_utest_graph_optimizer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rts_engine_utest_ops_kernel_builder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rts_engine_utest_ops_kernel_info.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/common/util.cpp
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/common/rt_log.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/engine/rts_engine.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/graph_optimizer/rts_graph_optimizer.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/rts_ops_kernel_builder.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/rts_ops_kernel_info.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/graph_optimizer/rts_ffts_plus_graph_optimizer.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/rts_ffts_plus_ops_kernel_builder.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/rts_ffts_plus_ops_kernel_info.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/end_graph_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/label_goto_ex_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/label_goto_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/label_set_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/label_switch_by_index_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/label_switch_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/memcpy_addr_async_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/cmo_addr_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/memcpy_async_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/op_factory.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/op_ffts_plus_factory.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/model_exit_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/recv_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/send_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/stream_active_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/stream_switchN_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/stream_switch_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/npu_get_float_status_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/npu_clear_float_status_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/npu_get_float_debug_status_op.cpp
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/npu_clear_float_debug_status_op.cpp
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/send_notify_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/recv_notify_op.cc
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/stream_merge_op.cc
    ${PROTO_HDRS}
)

add_executable(rts_engine_utest
    ${rts_engine_src_file}
)

target_include_directories(rts_engine_utest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../compiler/engines/rts_engine
    ${CMAKE_BINARY_DIR}/proto/rts_engine_proto
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}
    ${AIR_CODE_DIR}/base/metadef/pkg_inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${AIR_CODE_DIR}/inc/graph_metadef/register
    ${AIR_CODE_DIR}/inc/graph_metadef/exe_graph
    ${AIR_CODE_DIR}/inc/graph_metadef/exe_graph/lowering
    ${AIR_CODE_DIR}/inc/graph_metadef/external/hcom
    ${AIR_CODE_DIR}/inc/graph_metadef/external
    ${CMAKE_BINARY_DIR}/proto/rts_engine_proto
    ${CMAKE_CURRENT_SOURCE_DIR}/../stub
    ${CMAKE_CURRENT_SOURCE_DIR}/test
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/graph_metadef/common/ge_common/
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/graph_metadef/common/opskernel/
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/graph_metadef/common/optimizer/
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/graph_metadef/common/ge_common
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/graph_metadef/common/opskernel
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/graph_metadef/common/optimizer
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/graph_metadef/util/
    #${AIR_CODE_DIR}/inc/graph_metadef/register
    ${AIR_CODE_DIR}/inc/graph_metadef/common/ge_common/
    ${AIR_CODE_DIR}/inc/graph_metadef/common/opskernel/
    ${AIR_CODE_DIR}/inc/graph_metadef/common/optimizer/
    ${METADEF_DIR}/inc/external/register
    ${METADEF_DIR}/inc/graph
    ${AIR_CODE_DIR}/inc/external/register
    ${AIR_CODE_DIR}/inc/external/register/
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/common
    ${AIR_CODE_DIR}/../inc/
    ${ASCEND_3RD_LIB_PATH}/json/include
    ${ASCEND_3RD_LIB_PATH}/protoc/include
    ${ASCEND_INSTALL_PATH}/
    ${ASCEND_INSTALL_PATH}
    ${ASCEND_INSTALL_PATH}/lib64
    ${ASCEND_INSTALL_PATH}/lib64/
    ${ASCEND_INSTALL_PATH}/include
    ${ASCEND_INSTALL_PATH}/include/external
    ${ASCEND_INSTALL_PATH}/include/mmpa
    ${ASCEND_INSTALL_PATH}/include/hccl
    ${ASCEND_INSTALL_PATH}/pkg_inc
    ${ASCEND_INSTALL_PATH}/pkg_inc/hccl
    ${ASCEND_INSTALL_PATH}/pkg_inc/base
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/engine/
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/common/constant/
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/graph_optimizer/
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/
    ${AIR_CODE_DIR}/compiler/engines/rts_engine/ops_kernel_store/op/
    ${CMAKE_BINARY_DIR}/proto/rts_engine_proto
    ${AIR_CODE_DIR}/inc/graph_metadef/common/ge_common
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/external/acl
    ${AIR_CODE_DIR}/base/metadef/pkg_inc
    ${AIR_CODE_DIR}/base/metadef/pkg_inc/common
    ${AIR_CODE_DIR}/inc/external/ge
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    ${ASCEND_3RD_LIB_PATH}/protoc/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../stub
)

target_compile_definitions(rts_engine_utest PRIVATE
    RUN_TEST
    DRIVER_NEW_API
    FEATURE_NOTIFY
    FEATURE_PCTRACE
    FEATURE_DEBUG
    FEATURE_GE_API
    CFG_DEV_PLATFORM_PC
    google=ascend_private
)

target_compile_options(rts_engine_utest PRIVATE
    -std=c++11
    -Wno-deprecated-declarations
    -fno-common
    -fno-strict-aliasing
    -Wextra
    -Wfloat-equal
)

target_link_options(rts_engine_utest PRIVATE
    -rdynamic
    -Wl,-Bsymbolic
    -Wl,--exclude-libs,ALL
    -Wl,-rpath,${ASCEND_COMPILER_DIR}
    -Wl,-rpath,${ASCEND_3RD_LIB_PATH}/gtest_shared/lib64
    -Wl,-rpath,${AIR_CODE_DIR}/build/graph_metadef/register
    -Wl,-rpath,${AIR_CODE_DIR}/build/graph_metadef/graph
)

target_link_libraries(rts_engine_utest PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:mmpa_headers>
    $<BUILD_INTERFACE:msprof_headers>
    $<BUILD_INTERFACE:slog_headers>
    $<BUILD_INTERFACE:runtime_headers>
    $<BUILD_INTERFACE:cce_headers>
    $<BUILD_INTERFACE:atrace_headers>
    metadef_headers
    c_sec
    alog
    graph
    runtime
    -Wl,--as-needed
    cce_headers
    intf_llt_pub
    runtime_headers
    msprof_headers
    GTest::gtest
    error_manager
    graph_base
    air_headers
    ascend_protobuf
    register
    mmpa
    platform
    json
    -Wl,--no-as-needed
    slog
    -ldl
    -lrt
    -lpthread
    -lgcov
)


# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

SET(FFTS_ROOT_PATH ${AIR_CODE_DIR}/compiler/engines/ffts_engine)
SET(FE_ROOT_PATH ${AIR_CODE_DIR}/compiler/engines/nn_engine)
file(GLOB_RECURSE PROTO_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "${METADEF_PROTO_DIR}/task.proto"
        "${METADEF_PROTO_DIR}/ge_ir.proto"
        "${METADEF_PROTO_DIR}/om.proto"
        "${METADEF_PROTO_DIR}/insert_op.proto"
        "${METADEF_PROTO_DIR}/ge_ir_mobile.proto"
        "${METADEF_PROTO_DIR}/task_mobile.proto")

protobuf_generate(ffts_st PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

# compile for ffts_st
add_executable(ffts_st
        ${PROTO_HDRS}
        ${FE_ROOT_PATH}/utils/common/scope_allocator.cc
        ${FE_ROOT_PATH}/utils/common/graph_comm.cc
        ${FE_ROOT_PATH}/utils/common/fe_type_utils.cc
        ${FE_ROOT_PATH}/utils/common/config_parser/op_debug_config_parser.cc
        ${FE_ROOT_PATH}/utils/common/platform_utils.cc
        ${FE_ROOT_PATH}/utils/common/fe_report_error.cc
        ${FE_ROOT_PATH}/utils/common/fe_gentask_utils.cc
        ${FE_ROOT_PATH}/utils/common/config_parser/op_impl_mode_config_parser.cc
        ${FE_ROOT_PATH}/utils/common/config_parser/op_cust_dtypes_config_parser.cc
        ${FE_ROOT_PATH}/utils/common/config_parser/modify_mixlist_config_parser.cc
        ${FE_ROOT_PATH}/utils/common/fe_context_utils.cc
        ${FE_ROOT_PATH}/utils/common/configuration.cc
        ${FE_ROOT_PATH}/utils/common/fe_graph_common.cc
        ${FE_ROOT_PATH}/utils/common/graph_comm_impl.cc
        ${FE_ROOT_PATH}/utils/common/aicore_util_attr_define.cc
        ${FE_ROOT_PATH}/utils/common/thread_slice_info_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/main.cc
        ${FFTS_ROOT_PATH}/utils/ffts_utils.cc
        ${FFTS_ROOT_PATH}/common/ffts_plugin_manager.cc
        ${FFTS_ROOT_PATH}/common/ffts_constants.cc
        ${FFTS_ROOT_PATH}/common/ffts_configuration.cc
        ${FFTS_ROOT_PATH}/engine/engine_manager.cc
        ${FFTS_ROOT_PATH}/engine/fftsplus_engine.cc
        ${FFTS_ROOT_PATH}/task_builder/data_ctx/cache_persistent_manual_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/data_ctx/out_auto_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/data_ctx/out_dynamic_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/data_ctx/out_manual_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/data_ctx/prefetch_auto_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/data_ctx/prefetch_dynamic_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/data_ctx/prefetch_manual_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/fftsplus_ops_kernel_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/fftsplus_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/mode/auto/auto_thread_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/mode/data_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/mode/manual/manual_thread_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/thread_ctx/dsa_manual_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/mode/mixl2/mixl2_mode_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/mode/memory_slice.cc
        ${FFTS_ROOT_PATH}/task_builder/mode/thread_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/thread_ctx/aicpu_manual_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/thread_ctx/aicpu_auto_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/thread_ctx/aic_aiv_auto_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/thread_ctx/aic_aiv_manual_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/thread_ctx/collection_ops_manual_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/thread_ctx/mix_aic_aiv_auto_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/thread_ctx/mix_aic_aiv_dynamic_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/thread_ctx/mix_aic_aiv_manual_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/thread_ctx/runtime_ops_manual_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/thread_ctx/runtime_ops_auto_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/mixl2_ctx/mix_l2_task_builder.cc
        ${FFTS_ROOT_PATH}/task_builder/thread_ctx/aic_aiv_dynamic_task_builder.cc
        ${FFTS_ROOT_PATH}/utils/param_calculate/ffts_param_calculator.cc
        ${CMAKE_CURRENT_LIST_DIR}/../graph_constructor/graph_builder_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/../graph_constructor/graph_constructor.cc
        ${CMAKE_CURRENT_LIST_DIR}/../graph_constructor/ffts_llt_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/../stub/slog_stub.cc
        ${CMAKE_CURRENT_LIST_DIR}/../stub/runtime_stub.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/engine/engine_manager_stest.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/main.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/task_builder/aic_aiv_task_builder_stest.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/task_builder/aicpu_auto_task_builder_stest.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/task_builder/data_context_stest.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/task_builder/dsa_manual_task_builder_stest.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/task_builder/mix_aic_aiv_auto_task_builder_stest.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/task_builder/fftsplus_ops_kernel_builder_stest.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/task_builder/fftsplus_task_builder_stest.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/task_builder/fftsplus_dynamic_task_builder_stest.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/common/ffts_configuration_stest.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcase/param_calculator/ffts_param_calculator_stest.cc
        )

target_include_directories(ffts_st PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${FFTS_ROOT_PATH}
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}/compiler/engines
        ${AIR_CODE_DIR}/compiler/graphcompiler/engines
        ${AIR_CODE_DIR}/compiler/engines/nn_engine/inc
        ${AIR_CODE_DIR}/compiler/engines/nn_engine/utils
        ${AIR_CODE_DIR}/tests/engines/ffts_engine/graph_constructor
        ${AIR_CODE_DIR}/tests/engines/ffts_engine/stub
        ${OP_PROTO_DIR}
        ${GTEST_INCLUDE}
        ${CMAKE_BINARY_DIR}/proto/ffts_st)

target_compile_definitions(ffts_st PRIVATE
        $<$<STREQUAL:${BUILD_OPEN_PROJECT},True>:ONLY_COMPILE_BLUE>
        )

target_compile_options(ffts_st PRIVATE
        -D__OPTIMIZER_KT_TEST__
        -DDAVINCI_CLOUD
        -Dgoogle=ascend_private)

target_link_directories(ffts_st PRIVATE
    ${ASCEND_INSTALL_PATH}/lib64
)

target_link_libraries(ffts_st PRIVATE
        intf_llt_pub
        slog_headers
        msprof_headers
        runtime_headers
        air_headers
        -Wl,--no-as-needed
        -lrt -ldl -lpthread -lgcov
        exe_graph
        graph
        metadef
        graph_base
        register
        opp_registry
        ascend_protobuf
        c_sec
        platform
        error_manager
        json
        GTest::gtest
        GTest::gtest_main
        c_sec_static
        cce_headers
        unified_dlog
        ascendalog
        mmpa)

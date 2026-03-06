# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(engines)
add_subdirectory(graph/eager_style_graph_builder)
if (BUILD_OPEN_PROJECT OR ENABLE_OPEN_SRC)
    add_subdirectory(graph/common/compress)
    add_subdirectory(opcompiler/op_compile_adapter/source)
else()
endif()
if (BUILD_METADEF)
    add_subdirectory(graph/optimize/autofuse)
endif()
add_subdirectory(graphcompiler)

set(GRAPHENGINE_PROTO_LIST
    "${METADEF_PROTO_DIR}/om.proto"
    "${METADEF_PROTO_DIR}/task.proto"
    "${METADEF_PROTO_DIR}/insert_op.proto"
    "${METADEF_PROTO_DIR}/ge_ir.proto"
    "${METADEF_PROTO_DIR}/ascendc_ir.proto"
    "${METADEF_PROTO_DIR}/ge_api.proto"
    "${METADEF_PROTO_DIR}/fusion_model.proto"
    "${METADEF_PROTO_DIR}/optimizer_priority.proto"
    "${METADEF_PROTO_DIR}/fwk_adapter.proto"
    "${METADEF_PROTO_DIR}/op_mapping.proto"
    "${METADEF_PROTO_DIR}/dump_task.proto"
    "${METADEF_PROTO_DIR}/tensorflow/graph_library.proto"
    "${METADEF_PROTO_DIR}/tensorflow/graph.proto"
    "${METADEF_PROTO_DIR}/tensorflow/node_def.proto"
    "${METADEF_PROTO_DIR}/tensorflow/function.proto"
    "${METADEF_PROTO_DIR}/tensorflow/versions.proto"
    "${METADEF_PROTO_DIR}/tensorflow/attr_value.proto"
    "${METADEF_PROTO_DIR}/tensorflow/op_def.proto"
    "${METADEF_PROTO_DIR}/tensorflow/tensor.proto"
    "${METADEF_PROTO_DIR}/tensorflow/tensor_shape.proto"
    "${METADEF_PROTO_DIR}/tensorflow/types.proto"
    "${METADEF_PROTO_DIR}/tensorflow/resource_handle.proto"
    "${METADEF_PROTO_DIR}/var_manager.proto"
    "${AIR_CODE_DIR}/dflow/proto/dflow.proto"
    "${METADEF_PROTO_DIR}/flow_model.proto"
    "${METADEF_PROTO_DIR}/ge_ir_mobile.proto"
    "${METADEF_PROTO_DIR}/task_mobile.proto"
)

protobuf_generate(graphengine_protos GRAPHENGINE_PROTO_SRCS GRAPHENGINE_PROTO_HDRS ${GRAPHENGINE_PROTO_LIST} "--proto_path=${METADEF_PROTO_DIR}" TARGET)

set(GE_FUSION_MODEL_PROTO_SRCS
    "${CMAKE_BINARY_DIR}/proto/graphengine_protos/proto/fusion_model.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/graphengine_protos/proto/optimizer_priority.pb.cc"
)

add_library(ge_fusion_model_protos_obj OBJECT ${GE_FUSION_MODEL_PROTO_SRCS})

add_dependencies(ge_fusion_model_protos_obj graphengine_protos)

target_compile_definitions(ge_fusion_model_protos_obj PRIVATE
    google=ascend_private
)

target_link_libraries(ge_fusion_model_protos_obj PRIVATE ascend_protobuf intf_pub)

target_compile_options(ge_fusion_model_protos_obj PRIVATE
    -Werror
    -Wextra
    -Wfloat-equal
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC>
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

if (NOT ENABLE_D AND NOT ENABLE_ACL AND NOT ENABLE_MS_TESTCASES)
############ libge_proto_common.a ############
add_library(ge_proto_common STATIC
    $<TARGET_OBJECTS:ge_fusion_model_protos_obj>
)

target_compile_definitions(ge_proto_common PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
)

target_compile_options(ge_proto_common PRIVATE
    -O2
    -fno-common
    -Wextra
    -Wfloat-equal
)

target_link_libraries(ge_proto_common PRIVATE
    intf_pub
    ascend_protobuf
)
endif ()

##################################################################
file(GLOB_RECURSE SYMBOL_SHAPE_INFER_FUNCS_SRCS CONFIGURE_DEPENDS
    "graph/optimize/symbolic/*.cc"
)
file(GLOB_RECURSE MEM_CHECK_FUNCS_SRCS CONFIGURE_DEPENDS "graph/optimize/mem_layout_conflict_optimize/checker/funcs/*.cc" )
set(COMPILER_SRC_LIST
    "analyzer/analyzer.cc"
    "engines/manager/engine_manager/dnnengine_manager.cc"
    "opt_info/ge_opt_info.cc"
    "api/generator/ge_generator.cc"
    "graph/build/graph_builder.cc"
    "graph/build/label_allocator.cc"
    "graph/build/stream/assign_attached_stream_pass.cc"
    "graph/build/stream/assign_attached_notify_pass.cc"
    "graph/build/stream/assign_attached_event_pass.cc"
    "graph/build/stream/attach_resource_assign_helper.cc"
    "graph/build/stream/logical_stream_allocator.cc"
    "graph/build/stream/stream_allocator.cc"
    "graph/build/stream/stream_info.cc"
    "graph/build/stream/dynamic_stream_allocator.cc"
    "graph/build/stream/stream_utils.cc"
    "graph/build/memory/binary_block_mem_assigner.cc"
    "graph/build/memory/continuous_mem.cc"
    "graph/build/memory/block_mem_assigner.cc"
    "graph/build/memory/block_type_list.cc"
    "graph/build/memory/checker/dependency_analyzer.cc"
    "graph/build/memory/checker/reuse_checker.cc"
    "graph/build/memory/checker/node_checker_register.cc"
    "graph/build/memory/checker/node_checker_utils.cc"
    "graph/build/memory/checker/special_node_checker.cc"
    "graph/build/memory/checker/atomic_clean_checker.cc"
    "graph/build/memory/checker/node_checker/continuous_input_node_checker.cc"
    "graph/build/memory/checker/node_checker/nopadding_continuous_input_node_checker.cc"
    "graph/build/memory/checker/node_checker/continuous_output_node_checker.cc"
    "graph/build/memory/checker/node_checker/nopadding_continuous_output_node_checker.cc"
    "graph/build/memory/buffer_pool_mem_assigner.cc"
    "graph/build/memory/graph_mem_assigner.cc"
    "graph/build/memory/hybrid_mem_assigner.cc"
    "graph/build/memory/mem_inplace.cc"
    "graph/build/memory/max_block_mem_assigner.cc"
    "graph/build/memory/memory_assigner.cc"
    "graph/build/memory/graph_mem_splitter.cc"
    "graph/build/memory/dynamic_batch_mem_assigner.cc"
    "graph/build/memory/mem_reuse_strategy.cc"
    "graph/build/model_builder.cc"
    "graph/build/run_context_util.cc"
    "graph/build/stream_graph_optimizer.cc"
    "graph/build/task_generator.cc"
    "graph/build/task_generator_utils.cc"
    "graph/build/graph_compile_summary_impl.cc"
    "graph/build/profiling_task_utils.cc"
    "graph/build/model_cache.cc"
    "graph/common/trans_op_creator.cc"
    "graph/common/ge_utils.cc"
    "graph/fusion/pattern.cc"
    "graph/fusion/match_result.cc"
    "graph/fusion/node_matcher.cc"
    "graph/fusion/pattern_matcher.cc"
    "graph/fusion/pattern_matcher_config.cc"
    "graph/fusion/subgraph_boundary.cc"
    "graph/fusion/graph_rewriter.cc"
    "graph/fusion/fusion_utils.cc"
    "graph/fusion/pass/decompose_pass.cc"
    "graph/fusion/pass/pattern_fusion_pass.cc"
    "graph/fusion/pass/fusion_pass_executor.cc"
    "graph/fuzz_compile/fuzz_compile_bin_selector.cc"
    "graph/label/case_op_label_maker.cc"
    "graph/label/if_op_label_maker.cc"
    "graph/label/label_maker.cc"
    "graph/label/partitioned_call_op_label_maker.cc"
    "graph/label/while_op_label_maker.cc"
    "graph/manager/graph_context.cc"
    "graph/manager/graph_manager.cc"
    "graph/manager/util/rt_context_util.cc"
    "graph/manager/util/graph_rebuild_state_ctrl.cc"
    "graph/manager/util/graph_optimize_utility.cc"
    "graph/optimize/graph_optimize.cc"
    "graph/optimize/mem_rw_conflict_optimize.cc"
    "graph/optimize/autofuse/autofuse_optimize.cc"
    "graph/optimize/summary_optimize.cc"
    "graph/partition/base_partitioner.cc"
    "graph/partition/dynamic_shape_partition.cc"
    "graph/partition/base_cluster.cc"
    "graph/partition/engine_place.cc"
    "graph/partition/engine_partitioner.cc"
    "graph/partition/stage_partitioner.cc"
    "graph/partition/optimizer/dynamic_data_flow_engine_reassign_pass.cc"
    "graph/partition/optimizer/dynamic_data_flow_partitioner_pass.cc"
    "graph/partition/optimizer/hostcpu_engine_update_pass.cc"
    "graph/passes/feature/set_ffts_plus_attr_pass.cc"
    "graph/passes/standard_optimize/addn_pass.cc"
    "graph/passes/standard_optimize/remove_unsupported_op/assert_pass.cc"
    "graph/passes/variable_optimize/assign_remove_pass.cc"
    "graph/passes/memory_conflict/atomic_addr_clean_pass.cc"
    "graph/passes/control_flow_and_stream/attach_stream_label_pass.cc"
    "graph/passes/feature/attached_resource_pass.cc"
    "graph/passes/base_pass.cc"
    "graph/passes/feature/buffer_pool_memory_pass.cc"
    "graph/passes/format_optimize/cast_remove_pass.cc"
    "graph/passes/standard_optimize/common_subexpression_elimination_pass.cc"
    "graph/passes/feature/compile_nodes_pass.cc"
    "graph/passes/control_flow_and_stream/cond_pass.cc"
    "graph/passes/control_flow_and_stream/cond_remove_pass.cc"
    "graph/passes/symbolic/symbolic_cond_remove_pass.cc"
    "graph/passes/standard_optimize/constant_folding/constant_folding_pass.cc"
    "graph/passes/feature/constant_clip_pass.cc"
    "graph/passes/standard_optimize/constant_fuse_same_pass.cc"
    "graph/passes/control_flow_and_stream/control_trigger_pass.cc"
    "graph/passes/multi_batch/create_subgraph_with_scope_pass.cc"
    "graph/passes/standard_optimize/ctrl_edge_transfer_pass.cc"
    "graph/passes/control_flow_and_stream/data_pass.cc"
    "graph/passes/shape_optimize/split_shape_n_pass.cc"
    "graph/passes/shape_optimize/merge_unknown_shape_n_pass.cc"
    "graph/passes/standard_optimize/constant_folding/dimension_adjust_pass.cc"
    "graph/passes/standard_optimize/constant_folding/dimension_compute_pass.cc"
    "graph/passes/standard_optimize/remove_unsupported_op/dropout_pass.cc"
    "graph/passes/standard_optimize/end_of_sequence_add_control_pass.cc"
    "graph/passes/control_flow_and_stream/enter_pass.cc"
    "graph/passes/control_flow_and_stream/flow_ctrl_pass.cc"
    "graph/passes/standard_optimize/constant_folding/folding_pass.cc"
    "graph/passes/control_flow_and_stream/for_pass.cc"
    "graph/passes/standard_optimize/fuse_data_nodes_with_common_input_pass.cc"
    "graph/passes/feature/get_original_format_pass.cc"
    "graph/passes/feature/global_step_insert_pass.cc"
    "graph/passes/standard_optimize/remove_unsupported_op/guarantee_const_pass.cc"
    "graph/passes/memory_conflict/hccl_continuous_memcpy_pass.cc"
    "graph/passes/feature/hccl_group_pass.cc"
    "graph/passes/memory_conflict/hccl_memcpy_pass.cc"
    "graph/passes/feature/hccl_sequence_adjust_pass.cc"
    "graph/passes/feature/hccl_tailing_optimization_pass.cc"
    "graph/passes/memory_conflict/identity_pass.cc"
    "graph/passes/shape_optimize/infer_base_pass.cc"
    "graph/passes/shape_optimize/infer_value_range_pass.cc"
    "graph/passes/shape_optimize/infershape_pass.cc"
    "graph/passes/memory_conflict/inplace_support_check_pass.cc"
    "graph/passes/feature/input_output_connection_identify_pass.cc"
    "graph/passes/feature/iterator_op_pass.cc"
    "graph/passes/feature/super_kernel_pass.cc"
    "graph/passes/feature/link_gen_mask_nodes_pass.cc"
    "graph/passes/control_flow_and_stream/mark_agnostic_pass.cc"
    "graph/passes/shape_optimize/mark_force_unknown_for_cond_pass.cc"
    "graph/passes/shape_optimize/mark_graph_unknown_status_pass.cc"
    "graph/passes/memory_conflict/mark_same_addr_pass.cc"
    "graph/passes/memory_conflict/memcpy_addr_async_pass.cc"
    "graph/passes/control_flow_and_stream/merge_input_memcpy_pass.cc"
    "graph/passes/control_flow_and_stream/merge_pass.cc"
    "graph/passes/control_flow_and_stream/merge_to_stream_merge_pass.cc"
    "graph/passes/multi_batch/multi_batch_clone_pass.cc"
    "graph/passes/multi_batch/multi_batch_pass.cc"
    "graph/passes/feature/net_output_pass.cc"
    "graph/passes/feature/inner_tensor_move_add_pass.cc"
    "graph/passes/feature/inner_tensor_move_delete_pass.cc"
    "graph/passes/control_flow_and_stream/next_iteration_pass.cc"
    "graph/passes/shape_optimize/no_use_reshape_remove_pass.cc"
    "graph/passes/feature/parallel_concat_start_op_pass.cc"
    "graph/passes/feature/parallel_group_pass.cc"
    "graph/passes/pass_manager.cc"
    "graph/passes/pass_utils.cc"
    "graph/passes/standard_optimize/permute_pass.cc"
    "graph/passes/standard_optimize/placeholder_with_default_pass.cc"
    "graph/passes/standard_optimize/remove_unsupported_op/prevent_gradient_pass.cc"
    "graph/passes/standard_optimize/remove_unsupported_op/print_op_pass.cc"
    "graph/passes/standard_optimize/prune_pass.cc"
    "graph/passes/variable_optimize/ref_identity_delete_op_pass.cc"
    "graph/passes/standard_optimize/remove_same_const_pass.cc"
    "graph/passes/shape_optimize/replace_transshape_pass.cc"
    "graph/passes/standard_optimize/constant_folding/replace_with_empty_const_pass.cc"
    "graph/passes/shape_optimize/reshape_recovery_pass.cc"
    "graph/passes/shape_optimize/reshape_remove_pass.cc"
    "graph/passes/standard_optimize/same_transdata_breadth_fusion_pass.cc"
    "graph/passes/standard_optimize/tensor_move_delete_pass.cc"
    "graph/passes/standard_optimize/save_pass.cc"
    "graph/passes/memory_conflict/set_input_output_offset_pass.cc"
    "graph/passes/shape_optimize/shape_operate_op_remove_pass.cc"
    "graph/passes/standard_optimize/remove_unsupported_op/snapshot_pass.cc"
    "graph/passes/standard_optimize/remove_unsupported_op/stop_gradient_pass.cc"
    "graph/passes/multi_batch/subexpression_migration_pass.cc"
    "graph/passes/multi_batch/subgraph_const_migration_pass.cc"
    "graph/passes/multi_batch/subgraph_multi_dims_pass.cc"
    "graph/passes/multi_batch/subgraph_multi_dims_clone_pass.cc"
    "graph/passes/memory_conflict/subgraph_pass.cc"
    "graph/passes/control_flow_and_stream/switch_data_edges_bypass.cc"
    "graph/passes/control_flow_and_stream/switch_dead_branch_elimination.cc"
    "graph/passes/control_flow_and_stream/switch_logic_remove_pass.cc"
    "graph/passes/control_flow_and_stream/switch_to_stream_switch_pass.cc"
    "graph/passes/standard_optimize/start_of_sequence_pass.cc"
    "graph/passes/format_optimize/transop_breadth_fusion_pass.cc"
    "graph/passes/format_optimize/transop_nearby_allreduce_fusion_pass.cc"
    "graph/passes/format_optimize/transop_symmetry_elimination_pass.cc"
    "graph/passes/format_optimize/transop_without_reshape_fusion_pass.cc"
    "graph/passes/format_optimize/transpose_transdata_pass.cc"
    "graph/passes/format_optimize/unchanged_transpose_remove_pass.cc"
    "graph/passes/format_optimize/dim1_transpose_to_squeeze_pass.cc"
    "graph/passes/standard_optimize/unused_args_clean_pass.cc"
    "graph/passes/standard_optimize/unused_const_pass.cc"
    "graph/passes/standard_optimize/useless_control_out_remove_pass.cc"
    "graph/passes/variable_optimize/var_is_initialized_op_pass.cc"
    "graph/passes/variable_optimize/variable_op_pass.cc"
    "graph/passes/variable_optimize/variable_prepare_op_pass.cc"
    "graph/passes/variable_optimize/variable_ref_delete_op_pass.cc"
    "graph/passes/variable_optimize/variable_ref_useless_control_out_delete_pass.cc"
    "graph/passes/memory_optimize/swap_space_pass.cc"
    "graph/passes/feature/data_flow_prepare_pass.cc"
    "graph/passes/standard_optimize/constant_folding/potential_folding_pass.cc"
    "graph/passes/standard_optimize/constant_folding/potential_const_taken_effect_pass.cc"
    "graph/passes/feature/recompute_pass.cc"
    "graph/passes/feature/auto_fuse_pass.cc"
    "graph/passes/feature/algebraic_simplification_pass.cc"
    "graph/passes/variable_optimize/split_variable_into_subgraph_pass.cc"
    "graph/passes/memory_optimize/concat_notask_pass.cc"
    "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.cc"
    "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_optimizer.cc"
    "graph/optimize/mem_layout_conflict_optimize/checker/checker.cc"
    "graph/optimize/mem_layout_conflict_optimize/checker/check_register.cc"
    "graph/optimize/mem_layout_conflict_optimize/checker/checker_log.cc"
    "graph/optimize/mem_layout_conflict_optimize/checker/control_node_subgraph_conflict_solve.cc"
    "${SYMBOL_SHAPE_INFER_FUNCS_SRCS}"
    "${MEM_CHECK_FUNCS_SRCS}"
    "graph/preprocess/graph_prepare.cc"
    "graph/preprocess/insert_op/aipp_op.cc"
    "graph/preprocess/insert_op/insert_aipp_op_util.cc"
    "graph/preprocess/multi_batch_copy_graph.cc"
    "graph/preprocess/multi_batch_options.cc" 
    "graph/preprocess/checker/graph_lint.cc"
    "host_kernels/kernel_utils.cc"
    "host_kernels/array_ops/gathershapes_kernel.cc"
    "host_kernels/array_ops/broadcast_args_kernel.cc"
    "host_kernels/array_ops/broadcast_gradient_args_kernel.cc"
    "host_kernels/array_ops/empty_kernel.cc"
    "host_kernels/array_ops/expanddims_kernel.cc"
    "host_kernels/array_ops/identity_kernel.cc"
    "host_kernels/array_ops/rank_kernel.cc"
    "host_kernels/array_ops/shape_kernel.cc"
    "host_kernels/array_ops/shape_n_kernel.cc"
    "host_kernels/array_ops/size_kernel.cc"
    "host_kernels/array_ops/squeeze_kernel.cc"
    "host_kernels/array_ops/squeezev3_kernel.cc"
    "host_kernels/array_ops/reshape_kernel.cc"
    "host_kernels/array_ops/unsqueeze_kernel.cc"
    "host_kernels/array_ops/unsqueezev3_kernel.cc"
    "host_kernels/custom_ops/ssd_prior_box_kernel.cc"
    "host_kernels/custom_ops/reformat_kernel.cc"
    "host_kernels/data_flow_ops/dynamic_stitch_kernel.cc"
    "host_kernels/elewise_calculation_ops/add_kernel.cc"
    "host_kernels/elewise_calculation_ops/cast_kernel.cc"
    "host_kernels/elewise_calculation_ops/floordiv_kernel.cc"
    "host_kernels/elewise_calculation_ops/floormod_kernel.cc"
    "host_kernels/elewise_calculation_ops/greater_kernel.cc"
    "host_kernels/elewise_calculation_ops/maximum_kernel.cc"
    "host_kernels/elewise_calculation_ops/mul_kernel.cc"
    "host_kernels/elewise_calculation_ops/rsqrt_kernel.cc"
    "host_kernels/elewise_calculation_ops/sub_kernel.cc"
    "host_kernels/pad_ops/fill_kernel.cc"
    "host_kernels/reduce_ops/reduce_prod_kernel.cc"
    "host_kernels/selection_ops/range_kernel.cc"
    "host_kernels/selection_ops/gather_v2_kernel.cc"
    "host_kernels/selection_ops/slice_d_kernel.cc"
    "host_kernels/selection_ops/slice_kernel.cc"
    "host_kernels/selection_ops/strided_slice_kernel.cc"
    "host_kernels/split_combination_ops/concat_offset_kernel.cc"
    "host_kernels/split_combination_ops/concat_v2_kernel.cc"
    "host_kernels/split_combination_ops/pack_kernel.cc"
    "host_kernels/split_combination_ops/unpack_kernel.cc"
    "host_kernels/transformation_ops/flattenv2_kernel.cc"
    "host_kernels/transformation_ops/permute_kernel.cc"
    "host_kernels/transformation_ops/transdata_kernel.cc"
    "host_kernels/transformation_ops/transpose_kernel.cc"
    "api/gelib/gelib.cc"
    "api/aclgrph/attr_options/keep_dtype_option.cc"
    "api/aclgrph/attr_options/utils.cc"
    "api/aclgrph/attr_options/weight_compress_option.cc"
    "api/aclgrph/ge_ir_build.cc"
    "api/aclgrph/option_utils.cc"
    "engines/manager/opskernel_manager/ops_kernel_builder_manager.cc"
    "engines/manager/opskernel_manager/ops_kernel_manager.cc"
    "engines/custom_engine/custom_graph_optimizer.cc"
    "engines/custom_engine/custom_ops_kernel_info_store.cc"
    "engines/custom_engine/custom_ops_kernel_builder.cc"
    "graph/build/model_data_info.cc"
)

if (NOT ENABLE_D AND NOT ENABLE_ACL AND NOT ENABLE_MS_TESTCASES)
message("CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
############ libge_compiler.so ############
add_library(ge_compiler SHARED
    ${COMPILER_SRC_LIST}
)

add_dependencies(ge_compiler
    graphengine_protos
)

target_compile_definitions(ge_compiler PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    REUSE_MEMORY=1
    FMK_SUPPORT_DUMP
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    $<$<STREQUAL:${ENABLE_TEST},True>:FUNC_VISIBILITY>
)

target_compile_options(ge_compiler PRIVATE
    ${AIR_COMMON_DYNAMIC_COMPILE_OPTION}
)

target_include_directories(ge_compiler  PRIVATE
    ${AIR_CODE_DIR}/inc/graph_metadef
)

target_include_directories(ge_compiler SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/inc
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/inc/framework
    ${ASCGEN_DIR}/inc
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    ${ASCGEN_DIR}/inc
    ${METADEF_DIR}/base/registry
    ${METADEF_DIR}/register
    ${METADEF_DIR}/pkg_inc
    ${TOP_DIR}/metadef/inc
    ${TOP_DIR}/runtime/include/external/
    ${TOP_DIR}/runtime/include/external/acl/error_codes
)

target_link_options(ge_compiler PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_compiler
    PRIVATE
        intf_pub
        cce_headers
        msprof_headers
        static_mmpa
        ge_proto_common
        -Wl,--no-as-needed
        graph
        graph_base
        flow_graph
        ge_common_base
        ascend_protobuf
        aihac_autofusion
        register
        opp_registry
        c_sec
        error_manager
        unified_dlog
        runtime_headers
        aihac_symbolizer
        -Wl,--as-needed
        json
        -lrt
        -ldl
        ${AIR_COMMON_LINK_OPTION}
        crypto_static
    PUBLIC
        ge_common
        air_headers
        metadef_headers
)

    ###############################################################
add_custom_target(
    engine_conf.json ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/engine_conf.json
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/engine_conf.json
    COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/engines/manager/engine_manager/engine_conf.json ${CMAKE_CURRENT_BINARY_DIR}/
)

###############################################################
add_custom_target(
    optimizer_priority.pbtxt ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/optimizer_priority.pbtxt
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/optimizer_priority.pbtxt
    COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/engines/manager/opskernel_manager/optimizer_priority.pbtxt ${CMAKE_CURRENT_BINARY_DIR}/
)

###############################################################

############ install ############
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/engine_conf.json
    ${CMAKE_CURRENT_BINARY_DIR}/optimizer_priority.pbtxt OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)
endif()

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(PROTO_LIST
    "${METADEF_PROTO_DIR}/om.proto"
    "${METADEF_PROTO_DIR}/ge_ir.proto"
    "${METADEF_PROTO_DIR}/task.proto"
    "${METADEF_PROTO_DIR}/ge_api.proto"
    "${METADEF_PROTO_DIR}/insert_op.proto"
    "${METADEF_PROTO_DIR}/dump_task.proto"
    "${METADEF_PROTO_DIR}/fwk_adapter.proto"
    "${METADEF_PROTO_DIR}/op_mapping.proto"
    "${METADEF_PROTO_DIR}/optimizer_priority.proto"
    "${METADEF_PROTO_DIR}/ge_api.proto"
    "${METADEF_PROTO_DIR}/tensorflow/attr_value.proto"
    "${METADEF_PROTO_DIR}/tensorflow/tensor.proto"
    "${METADEF_PROTO_DIR}/tensorflow/resource_handle.proto"
    "${METADEF_PROTO_DIR}/tensorflow/tensor_shape.proto"
    "${METADEF_PROTO_DIR}/tensorflow/graph_library.proto"
    "${METADEF_PROTO_DIR}/tensorflow/types.proto"
    "${METADEF_PROTO_DIR}/tensorflow/node_def.proto"
    "${METADEF_PROTO_DIR}/onnx/ge_onnx.proto"
    "${METADEF_PROTO_DIR}/var_manager.proto"
    "${METADEF_PROTO_DIR}/ge_ir_mobile.proto"
    "${METADEF_PROTO_DIR}/task_mobile.proto"
)

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST} "--proto_path=${METADEF_PROTO_DIR}")

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/data_flow_protos)
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CMAKE_BINARY_DIR}/proto/ge/proto)
include_directories(${CMAKE_BINARY_DIR}/proto_grpc/deployer_proto)

include_directories(${AIR_CODE_DIR}/base)
include_directories(${AIR_CODE_DIR}/compiler)
include_directories(${AIR_CODE_DIR}/runtime/v1)
include_directories(${AIR_CODE_DIR}/api/session)
include_directories(${AIR_CODE_DIR}/inc)
include_directories(${AIR_CODE_DIR}/inc/graph_metadef)
include_directories(${AIR_CODE_DIR}/inc/external)
include_directories(${AIR_CODE_DIR}/inc/framework)
include_directories(${AIR_CODE_DIR}/runtime/v2)
include_directories(${AIR_CODE_DIR}/runtime/om2)

include_directories(${PARSER_DIR})
include_directories(${AIR_CODE_DIR}/parser)
include_directories(${AIR_CODE_DIR}/parser/parser)
include_directories(${AIR_CODE_DIR}/inc/parser)
include_directories(${AIR_CODE_DIR}/inc/parser/external)

include_directories(${AIR_CODE_DIR}/tests/)
include_directories(${AIR_CODE_DIR}/tests/ge/ut/ge)
include_directories(${AIR_CODE_DIR}/tests/ge/ut/ge/runtime/stub)
include_directories(${AIR_CODE_DIR}/tests/ge/ut/common)
include_directories(${AIR_CODE_DIR}/tests/framework/ge_runtime_stub/include/)
include_directories(${AIR_CODE_DIR}/tests/framework/macro_utils)
include_directories(${AIR_CODE_DIR}/tests/ge/ut/ge/graph/optimize/symbolic)
include_directories(${METADEF_DIR}/inc)
include_directories(${METADEF_DIR}/pkg_inc)
include_directories(${METADEF_DIR})
include_directories(${ASCGEN_DIR}/inc)

# libge_ut_common.a resembles a testing-version of libge_compiler.so (and many more?),
# hence should also include what libge_compiler.so needs.
# See also tests/framework/cmake/graphengine.cmake for more info.
set(COMPILER_LINKED_PROTO_SRCS
    "${CMAKE_BINARY_DIR}/proto/ge/proto/optimizer_priority.pb.cc"
)

# test files
set(COMMON_TEST_FILES
    "graph/passes/graph_builder_utils.cc"
    "graph/utils/buffer_pool_graph_builder.cc"
    "test.cc"
    "graph/bin_cache/node_compile_cache_module_unittest.cc"
    "${AIR_CODE_DIR}/tests/graph_metadef/depends/faker/plugin_manager_stub.cc"
)

set(DISTINCT_GRAPH_LOAD_TEST_FILES
    "${AIR_CODE_DIR}/tests/ge/ut/ge/test_tools_task_info.cc"
    "graph/load/aipp_utils_unittest.cc"
    "graph/load/data_dumper_unittest.cc"
    "graph/load/model_manager_unittest.cc"
    "graph/load/new_model_manager_model_manager_aicpu_unittest.cc"
    "graph/load/end_graph_task_unittest.cc"
    "graph/load/event_record_task_unittest.cc"
    "graph/load/event_mem_record_task_unittest.cc"
    "graph/load/event_wait_task_unittest.cc"
    "graph/load/event_mem_wait_task_unittest.cc"
    "graph/load/notify_record_task_unittest.cc"
    "graph/load/notify_wait_task_unittest.cc"
    "graph/load/davinci_model_unittest.cc"
    "graph/load/memory_block_manager_unittest.cc"
    "graph/load/kernel_handle_unittest.cc"
    "graph/load/tbe_handle_store_unittest.cc"
    "graph/load/task_info_unittest.cc"
    "graph/load/hccl_task_info_unittest.cc"
    "graph/load/kernel_ex_task_info_unittest.cc"
    "graph/load/kernel_task_info_unittest.cc"
    "graph/load/kernel_manager_unittest.cc"
    "graph/load/cmo_task_info_unittest.cc"
    "graph/load/cmo_addr_task_info_unittest.cc"
    "graph/load/update_pc_task_info_unittest.cc"
    "graph/load/nop_task_info_unittest.cc"
    "graph/load/cmo_barrier_task_info_unittest.cc"
    "graph/load/ffts_plus_task_info_unittest.cc"
    "graph/load/dsa_task_info_unittest.cc"
    "graph/load/dvpp_task_info_unittest.cc"
    "graph/load/npu_get_float_status_task_info_unittest.cc"
    "graph/load/npu_clear_float_status_task_info_unittest.cc"
    "graph/load/memcpy_addr_async_task_info_unittest.cc"
    "graph/load/memcpy_async_task_info_unittest.cc"
    "graph/load/cpu_queue_schedule_unittest.cc"
    "graph/load/model_parser_base_unittest.cc"
    "executor/ge_executor_unittest.cc"
    "executor/om2_model_executor_unittest.cc"
    "executor/executor_utils_unittest.cc"
    "executor/runtime/model_rt_var_manager_unittest.cc"
    "graph/load/model_helper_unittest.cc"
    "graph/load/om2_package_helper_unittest.cc"
    "graph/load/om_file_helper_unittest.cc"
    "hybrid/hybrid_davinci_model_unittest.cc"
    "graph/load/model_utils_unittest.cc"
    "graph/load/profiler_trace_task_info_unittest.cc"
    "graph/load/graph_loader_unittest.cc"
    "graph/load/stream_switch_task_info_unittest.cc"
    "graph/load/scattered_collection_unittest.cc"
    "graph/load/sink_only_allocator_unittest.cc"
    "graph/load/memory_app_type_classifier_unittest.cc"
    "graph/load/task_args_refresh_type_classifier_unittest.cc"
    "graph/load/task_node_map_unittest.cc"
    "graph/load/model_args_layout_planner_unittest.cc"
    "graph/load/device_memory_ptr_unittest.cc"
    "graph/load/model_args_manager_unittest.cc"
)

file(GLOB_RECURSE MEM_LAYOUT_CONFLICT_OPTIMIZE_TEST_SRCS CONFIGURE_DEPENDS "graph/optimize/mem_layout_conflict/*.cc" )
set(PASS_TEST_FILES
    "${MEM_LAYOUT_CONFLICT_OPTIMIZE_TEST_SRCS}"
    "graph/passes/addn_pass_unittest.cc"
    "graph/passes/assign_remove_pass_unittest.cc"
    "graph/passes/atomic_addr_clean_pass_unittest.cc"
    "graph/passes/attach_stream_label_pass_unittest.cc"
    "graph/passes/attached_resource_pass_unittest.cc"
    "graph/passes/compile_nodes_pass_unittest.cc"
    "graph/passes/cond_pass_unittest.cc"
    "graph/passes/cond_remove_pass_unittest.cc"
    "graph/passes/control_trigger_pass_unittest.cc"
    "graph/passes/transop_without_reshape_fusion_pass_unittest.cc"
    "graph/passes/infer_value_range_pass_unittest.cc"
    "graph/passes/infer_base_pass_unittest.cc"
    "graph/passes/prune_pass_unittest.cc"
    "graph/passes/enter_pass_unittest.cc"
    "graph/passes/for_pass_unittest.cc"
    "graph/passes/switch_op_pass_unittest.cc"
    "graph/passes/get_original_format_pass_unittest.cc"
    "graph/passes/pass_manager_unittest.cc"
    "graph/passes/permute_pass_unittest.cc"
    "graph/passes/print_op_pass_unittest.cc"
    "graph/passes/shape_operate_op_remove_pass_unittest.cc"
    "graph/passes/variable_op_pass_unittest.cc"
    "graph/passes/base_pass_unittest.cc"
    "graph/passes/save_pass_unittest.cc"
    "graph/passes/merge_pass_unittest.cc"
    "graph/passes/multi_batch_pass_unittest.cc"
    "graph/passes/switch_logic_remove_pass_unittest.cc"
    "graph/passes/cond_branch_v1_unittest.cc"
    "graph/passes/loop_branch_v1_unittest.cc"
    "graph/passes/switch_dead_branch_elimination_unittest.cc"
    "graph/passes/subexpression_migration_pass_unittest.cc"
    "graph/passes/subgraph_pass_unittest.cc"
    "graph/passes/switch_to_stream_switch_pass_unittest.cc"
    "graph/passes/assert_pass_unittest.cc"
    "graph/passes/dropout_pass_unittest.cc"
    "graph/passes/end_of_sequence_add_control_pass_unittest.cc"
    "graph/passes/ctrl_edge_transfer_pass_unittest.cc"
    "graph/passes/common_subexpression_elimination_pass_unittest.cc"
    "graph/passes/unused_const_pass_unittest.cc"
    "graph/passes/reshape_remove_pass_unittest.cc"
    "graph/passes/ref_identity_delete_op_pass_unittest.cc"
    "graph/passes/trans_op_breadth_fusion_pass_unittest.cc"
    "graph/passes/transop_nearby_allreduce_fusion_pass_unittest.cc"
    "graph/passes/constant_folding_pass_unittest.cc"
    "graph/passes/constant_clip_pass_unittest.cc"
    "graph/passes/potential_const_taken_effect_pass_unittest.cc"
    "graph/passes/fuse_data_nodes_with_common_input_pass_unittest.cc"
    "graph/passes/stop_gradient_pass_unittest.cc"
    "graph/passes/prevent_gradient_pass_unittest.cc"
    "graph/passes/identity_pass_unittest.cc"
    "graph/passes/tensor_move_delete_pass_unittest.cc"
    "graph/passes/inner_tensor_move_add_pass_unittest.cc"
    "graph/passes/inner_tensor_move_delete_pass_unittest.cc"
    "graph/passes/global_step_insert_pass_unittest.cc"
    "graph/passes/placeholder_with_default_pass_unittest.cc"
    "graph/passes/snapshot_pass_unittest.cc"
    "graph/passes/guarantee_const_pass_unittest.cc"
    "graph/passes/flow_ctrl_pass_unittest.cc"
    "graph/passes/dimension_compute_pass_unittest.cc"
    "graph/passes/variable_prepare_pass_unittest.cc"
    "graph/passes/hccl_tailing_optimization_pass_unittest.cc"
    "graph/passes/inplace_support_check_pass_unittest.cc"
    "graph/passes/next_iteration_pass_unittest.cc"
    "graph/passes/variable_ref_delete_pass_unittest.cc"
    "graph/passes/dimension_adjust_pass_unittest.cc"
    "graph/passes/pass_utils_unittest.cc"
    "graph/passes/net_output_pass_unittest.cc"
    "graph/passes/no_use_reshape_remove_pass_unittest.cc"
    "graph/passes/infershape_pass_unittest.cc"
    "graph/passes/mark_force_unknown_for_cond_pass_unittest.cc"
    "graph/passes/multi_batch_clone_pass_unittest.cc"
    "graph/passes/constant_fuse_same_pass_unittest.cc"
    "graph/passes/subgraph_multi_dims_clone_pass_unittest.cc"
    "graph/passes/create_subgraph_with_scope_pass_unittest.cc"
    "graph/passes/subgraph_const_migration_pass_unittest.cc"
    "graph/passes/start_of_sequence_pass_unittest.cc"
    "graph/passes/replace_with_empty_const_pass_unittest.cc"
    "graph/passes/link_gen_mask_nodes_pass_unittest.cc"
    "graph/passes/transpose_transdata_pass_unittest.cc"
    "graph/passes/parallel_group_pass_unittest.cc"
    "graph/passes/buffer_pool_memory_pass_unittest.cc"
    "graph/passes/reshape_recovery_pass_unittest.cc"
    "graph/passes/cast_remove_pass_unittest.cc"
    "graph/passes/memcpy_addr_async_unittest.cc"
    "graph/passes/hccl_continuous_pass_unittest.cc"
    "graph/passes/hccl_memcpy_pass_unittest.cc"
    "graph/passes/same_transdata_breadth_fusion_pass_unittest.cc"
    "graph/passes/set_ffts_plus_attr_pass_unittest.cc"
    "graph/passes/transop_symmetry_elimination_pass_unittest.cc"
    "graph/passes/data_flow_prepare_pass_unittest.cc"
    "graph/passes/data_pass_unittest.cc"
    "graph/passes/merge_shape_n_pass_unittest.cc"
    "graph/passes/parallel_concat_start_op_pass_unittest.cc"
    "graph/passes/switch_data_edges_bypass_unittest.cc"
    "graph/passes/set_input_output_offset_pass_unittest.cc"
    "graph/passes/hccl_group_pass_unittest.cc"
    "graph/passes/hccl_sequence_adjust_pass_unittest.cc"
    "graph/passes/mark_same_addr_pass_unittest.cc"
    "graph/passes/mark_graph_unknown_status_pass_unittest.cc"
    "graph/passes/mark_agnostic_pass_unittest.cc"
    "graph/passes/iterator_op_pass_unittest.cc"
    "graph/passes/super_kernel_pass_unittest.cc"
    "graph/passes/unused_args_clean_pass_unittest.cc"
    "graph/passes/var_is_initialized_op_pass_unittest.cc"
    "graph/passes/replace_transshape_pass_unittest.cc"
    "graph/passes/remove_same_const_pass_unittest.cc"
    "graph/passes/input_output_connection_identify_pass_unittest.cc"
    "graph/passes/variable_ref_useless_control_out_delete_pass_unittest.cc"
    "graph/passes/split_shape_n_pass_unittest.cc"
    "graph/passes/swap_space_pass_unittest.cc"
    "graph/passes/unchanged_transpose_remove_pass_unittest.cc"
    "graph/passes/dim1_transpose_to_squeeze_pass_unittest.cc"
    "graph/passes/recompute_pass_unittest.cc"
    "graph/passes/split_variable_into_subgraph_pass_unittest.cc"
    "graph/passes/concat_notask_pass_unittest.cc"
    "graph/passes/auto_fuse_pass_unittest.cc"
    "graph/passes/algebraic_simplification_pass_unittest.cc"
    "fusion/pattern_unittest.cc"
    "fusion/fusion_utils_unittest.cc"
    "fusion/subgraph_boundary_unittest.cc"
    "fusion/graph_rewriter_unittest.cc"
    "fusion/node_matcher/constant_node_matcher_unittest.cc"
    "fusion/node_matcher/data_node_matcher_unittest.cc"
    "fusion/node_matcher/normal_node_matcher_unittest.cc"
    "fusion/pattern_matcher_unittest.cc"
    "fusion/pattern_matcher_config_unittest.cc"
    "fusion/match_result_unittest.cc"
    "fusion/fusion_component_test.cc"
    "fusion/pass/pattern_fusion_pass_unittest.cc"
    "fusion/pass/decompose_pass_unittest.cc"
    "fusion/pass/reg_pass_unittest.cc"
    "fusion/pass/fusion_pass_executor_unittest.cc"
)

set(KERNEL_TEST_FILES
    "graph/passes/folding_kernel/greater_kernel_unittest.cc"
    "graph/passes/folding_kernel/maximum_kernel_unittest.cc"
    "graph/passes/folding_kernel/floormod_kernel_unittest.cc"
    "graph/passes/folding_kernel/floordiv_kernel_unittest.cc"
    "graph/passes/folding_kernel/range_kernel_unittest.cc"
    "graph/passes/folding_kernel/mul_kernel_unittest.cc"
    "graph/passes/folding_kernel/shape_kernel_unittest.cc"
    "graph/passes/folding_kernel/shape_n_kernel_unittest.cc"
    "graph/passes/folding_kernel/size_kernel_unittest.cc"
    "graph/passes/folding_kernel/broadcast_args_kernel_unittest.cc"
    "graph/passes/folding_kernel/broadcast_gradient_args_kernel_unittest.cc"
    "graph/passes/folding_kernel/fill_kernel_unittest.cc"
    "graph/passes/folding_kernel/empty_kernel_unittest.cc"
    "graph/passes/folding_kernel/rank_kernel_unittest.cc"
    "graph/passes/folding_kernel/expanddims_kernel_unittest.cc"
    "graph/passes/folding_kernel/reshape_kernel_unittest.cc"
    "graph/passes/folding_kernel/reformat_kernel_unittest.cc"
    "graph/passes/folding_kernel/squeeze_kernel_unittest.cc"
    "graph/passes/folding_kernel/squeezev3_kernel_unittest.cc"
    "graph/passes/folding_kernel/transdata_kernel_unittest.cc"
    "graph/passes/folding_kernel/permute_kernel_unittest.cc"
    "graph/passes/folding_kernel/cast_kernel_unittest.cc"
    "graph/passes/folding_kernel/ssd_prior_box_kernel_unittest.cc"
    "graph/passes/folding_kernel/strided_slice_kernel_unittest.cc"
    "graph/passes/folding_kernel/pack_kernel_unittest.cc"
    "graph/passes/folding_kernel/concat_v2_kernel_unittest.cc"
    "graph/passes/folding_kernel/add_kernel_unittest.cc"
    "graph/passes/folding_kernel/sub_kernel_unittest.cc"
    "graph/passes/folding_kernel/reduce_prod_kernel_unittest.cc"
    "graph/passes/folding_kernel/rsqrt_kernel_unittest.cc"
    "graph/passes/folding_kernel/concat_offset_kernel_unittest.cc"
    "graph/passes/folding_kernel/gather_v2_kernel_unittest.cc"
    "graph/passes/folding_kernel/slice_kernel_unittest.cc"
    "graph/passes/folding_kernel/dynamic_stitch_kernel_unittest.cc"
    "graph/passes/folding_kernel/gathershapes_kernel_unittest.cc"
    "graph/passes/folding_kernel/flattenv2_kernel_unittest.cc"
    "graph/passes/folding_kernel/identity_kernel_unittest.cc"
    "graph/passes/folding_kernel/slice_d_kernel_unittest.cc"
    "graph/passes/folding_kernel/transpose_kernel_unittest.cc"
    "graph/passes/folding_kernel/unpack_kernel_unittest.cc"
    "graph/passes/folding_kernel/unsqueeze_kernel_unittest.cc"
    "graph/passes/folding_kernel/unsqueezev3_kernel_unittest.cc"
    "graph/passes/folding_kernel/kernel_utils_unittest.cc"
)

set(FORMAT_TRANSFERS_TEST_FILES
    "common/datatype_transfer_unittest.cc"
    "common/format_transfer_unittest.cc"
    "common/format_transfer_transpose_unittest.cc"
    "common/format_transfer_nchw_5d_unittest.cc"
    "common/format_transfer_nchw_fractalz_unittest.cc"
    "common/format_transfer_hwcn_fractalz_unittest.cc"
    "common/format_transfer_nhwc_fractalz_unittest.cc"
    "common/format_transfer_fractal_nz_unittest.cc"
    "common/format_transfer_fractal_zz_unittest.cc"
    "common/format_transfer_fractal_z_tbe_unittest.cc"
    "common/format_transfer_nhwc_5d_unittest.cc"
    "common/format_transfer_5d_nchw_unittest.cc"
    "common/format_transfer_5d_nhwc_unittest.cc"
    "common/format_transfer_hwcn_c1hwncoc0_unittest.cc"
    "common/format_transfer_c1hwncoc0_hwcn_unittest.cc"
    "common/format_transfer_fracz_nchw_unittest.cc"
    "common/format_transfer_fracz_nhwc_unittest.cc"
    "common/format_transfer_fracz_hwcn_unittest.cc"
    "common/format_transfer_dhwnc_fracz3D_transpose_unittest.cc"
    "common/format_transfer_dhwnc_fracz3D_unittest.cc"
    "common/format_transfer_fzc04_unittest.cc"
    "common/ge_format_util_unittest.cc"
    "common/register_format_transfer_unittest.cc"
)

set(BASE_COMMON_TEST_FILES
    "common/cust_aicpu_kernel_store_unittest.cc"
    "common/diagnose_switch_unittest.cc"
    "common/util_unittest.cc"
    "common/fp16_unittest.cc"
    "common/hif8_unittest.cc"
    "common/hccl_unittest.cc"
    "common/dump_manager_unittest.cc"
    "common/dump_properties_unittest.cc"
    "common/memory_dumper_unittest.cc"
    "common/mem_type_utils_unittest.cc"
    "common/tensor_trans_utils_unittest.cc"
    "${AIR_CODE_DIR}/tests/graph_metadef/depends/checker/tensor_check_utils.cc"
    "common/external_allocator_unittest.cc"
    "common/properties_manager_unittest.cc"
    "common/kernel_store_unittest.cc"
    "common/ge_auth_file_saver_unittest.cc"
    "common/tbe_plugin_manager_unittest.cc"
    "common/file_constant_unittest.cc"
    "common/graph_compile_summary_impl_unittest.cc"
    "common/profiling_definitions_unittest.cc"
    "common/dataslice_unittest.cc"
    "common/data_dependent_interpreter_unittest.cc"
    "common/ge_op_utils_unittest.cc"
    "common/ge_root_model_unittest.cc"
    "common/thread_pool_unittest.cc"
    "common/nano_dbg_data_unittest.cc"
    "common/nano_model_save_helper_unittest.cc"
    "common/pre_model_save_helper_unittest.cc"
    "common/op_tiling_rt2_unittest.cc"
    "common/tiling_context_builder_unittest.cc"
    "common/global_dumper_unittest.cc"
    "common/model_introduction_unittest.cc"
    "common/model_saver_unittest.cc"
    "common/model_compress_manager_unittest.cc"
    "common/preload/pre_task_status_unittest.cc"
    "common/preload/pre_model_utils_unittest.cc"
    "common/preload/pre_davinci_model_unittest.cc"
    "common/preload/pre_model_partition_utils_unittest.cc"
    "common/tiling_resource_manager_unittest.cc"
    "common/weight_manager_unittest.cc"
    "common/mobile_model_helper_unittest.cc"
    "common/dump_callback_unittest.cc"
    "common/error_registry_unittest.cc"
    "common/zip_archive_unittest.cc"
    "common/om2_codegen_unittest.cc"
    "common/program_generator_unittest.cc"
)

file(GLOB_RECURSE SYMBOL_SHAPE_INFER_FUNCS_TEST_SRCS CONFIGURE_DEPENDS "graph/optimize/symbolic/*.cc")
file(GLOB_RECURSE ES_GRAPH_BUILDER_TEST_SRCS CONFIGURE_DEPENDS "graph/eager_style_graph_builder/es_base_struct/*.cc" "graph/eager_style_graph_builder/es_generator/*.cc")

set(MULTI_PARTS_TEST_FILES
    "${AIR_CODE_DIR}/tests/ge/ut/ge/test_tools_task_info.cc"
    "${AIR_CODE_DIR}/tests/ge/ut/ge/ffts_plus_proto_tools.cc"
    "graph/common/trans_op_creator_unittest.cc"
    "graph/common/ge_utils_unittest.cc"
    "graph/build/test_memory_shared_graph.cc"
    "graph/build/var_mem_assign_util_unittest.cc"
    "graph/build/buffer_pool_mem_assigner_unittest.cc"
    "graph/build/graph_mem_assigner_unittest.cc"
    "graph/build/block_mem_assigner_unittest.cc"
    "graph/build/memory/checker/dependency_analyzer_unittest.cc"
    "graph/build/memory/checker/reuse_checker_unittest.cc"
    "graph/build/memory/checker/node_checker_utils_unittest.cc"
    "graph/build/memory/checker/special_node_checker_unittest.cc"
    "graph/build/memory/checker/atomic_clean_checker_unittest.cc"
    "graph/build/hybrid_mem_assigner_unittest.cc"
    "graph/build/memory_assigner_unittest.cc"
    "common/analyzer_unittest.cc"
    "common/dnnengine_manager_unittest.cc"
    "graph/optimize/graph_optimize_unittest.cc"
    "graph/optimize/graph_lint_unittest.cc"
    "graph/optimize/mem_rw_conflict_optimize_unittest.cc"
    "graph/preprocess/insert_op/ge_aipp_op_unittest.cc"
    "graph/preprocess/insert_op/util_insert_aipp_op_unittest.cc"
    "common/gelib_unittest.cc"
    "common/ge_local_engine_unittest.cc"
    "common/host_cpu_engine_unittest.cc"
    "common/ge_local_ops_kernel_builder_unittest.cc"
    "common/ge_local_ops_kernel_info_unittest.cc"
    "common/op_unittest.cc"
    "common/op_factory_unittest.cc"
    "common/no_op_unittest.cc"
    "common/ge_deleted_op_unittest.cc"
    "common/ops_kernel_builder_manager_unittest.cc"
    "common/ops_kernel_manager_unittest.cc"
    "common/platform_info_util_unittest.cc"
    "common/dnnengines_unittest.cc"
    "common/engine_manage_unittest.cc"
    "graph_ir/ge_custom_op_factory_unittest.cc"
    "graph_ir/ge_operator_factory_unittest.cc"
    "graph_ir/ge_ir_build_unittest.cc"
    "graph/transop_util_unittest.cc"
    "common/dump_op_unittest.cc"
    "common/dump_exception_unittest.cc"
    "common/opdebug_register_unittest.cc"
    "graph/graph_rebuild_state_ctrl_unittest.cc"
    "graph/build/logical_stream_allocator_unittest.cc"
    "graph/build/stream/stream_utils_unittest.cc"
    "graph/build/assign_attached_stream_pass_unittest.cc"
    "graph/build/assign_attached_notify_pass_unittest.cc"
    "graph/build/assign_attached_event_pass_unittest.cc"
    "graph/build/stream_allocator_unittest.cc"
    "graph/build/dynamic_stream_allocator_unittest.cc"
    "graph/build/stream/custom_stream_pass_unittest.cc"
    "graph/build/model_builder_unittest.cc"
    "graph/build/mem_assigner_unittest.cc"
    "graph/build/memory/block_type_list_unittest.cc"
    "graph/build/graph_mem_assigner_unittest.cc"
    "graph/build/memory/continuous_mem_unittest.cc"
    "graph/build/task_generator_unittest.cc"
    "graph/build/buffer_pool_mem_assigner_unittest.cc"
    "graph/build/graph_builder_unittest.cc"
    "graph/build/label_allocator_unittest.cc"
    "graph/build/run_context_unittest.cc"
    "graph/build/model_cache_unittest.cc"
    "graph/execute/graph_execute_unittest.cc"
    "graph/execute/model_executor_unittest.cc"
    "graph/preprocess/graph_preprocess_unittest.cc"
    "graph/preprocess/multi_batch_copy_graph_unittest.cc"
    "graph/preprocess/insert_op/ge_aipp_op_unittest.cc"
    "graph/manager/hcom_util_unittest.cc"
    "graph/manager/active_memory_allocator_unittest.cc"
    "graph/manager/graph_caching_allocator_unittest.cc"
    "graph/manager/host_mem_allocator_unittest.cc"
    "graph/manager/host_mem_manager_unittest.cc"
    "graph/manager/memory_api_unittest.cc"
    "graph/manager/rdma_pool_allocator_unittest.cc"
    "graph/manager/trans_var_data_utils_unittest.cc"
    "graph/manager/session_scope_mem_allocator_unittest.cc"
    "graph/manager/run_graph_unittest.cc"
    "graph/partition/dynamic_shape_partition_unittest.cc"
    "graph/partition/engine_place_unittest.cc"
    "graph/partition/graph_partition_unittest.cc"
    "graph/partition/stage_partition_unittest.cc"
    "graph/partition/optimizer/dynamic_data_flow_engine_reassign_unittest.cc"
    "graph/partition/optimizer/hostcpu_engine_update_pass_unittest.cc"
    "graph/manager/graph_manager_unittest.cc"
    "graph/manager/graph_var_manager_unittest.cc"
    "graph/optimize/mem_rw_conflict_optimize_unittest.cc"
    "graph/optimize/graph_optimize_unittest.cc"
    "graph/optimize/summary_optimize_unittest.cc"
    "graph/fuzz_compile/fuzz_compile_bin_selector_unittest.cc"
    "graph/fuzz_compile/one_node_single_bin_selector_unittest.cc"
    "session/main_unittest.cc"
    "session/omg_omg_unittest.cc"
    "session/single_op_parser_unittest.cc"
    "session/ge_api_unittest.cc"
    "session/inner_session_unittest.cc"
    "session/user_hybrid_graph_manager_unittest.cc"
    "session/session_manager_unittest.cc"
    "session_v2/ge_api_v2_unittest.cc"
    "${AIR_CODE_DIR}/tests/graph_metadef/depends/checker/tensor_check_utils.cc"
    "common/error_tracking_unittest.cc")

set(GE_OPT_INFO_TEST_FILES
    "opt_info/opt_info_unittest.cc"
)

set(GENERATOR_TEST_FILES
    "generator/ge_generator_unittest.cc"
)

set(SINGLE_OP_TEST_FILES
    "single_op/single_op_model_unittest.cc"
    "single_op/single_op_manager_unittest.cc"
    "single_op/stream_resource_unittest.cc"
    "single_op/single_op_task_unittest.cc"
    "single_op/single_op_unittest.cc"
    "single_op/task/op_task_unittest.cc"
    "single_op/task/tbe_task_builder_unittest.cc"
)

set(PROFILING_MNG_TEST_FILES
    "profiling/ge_profiling_manager_unittest.cc"
    "profiling/profiling_properties_unittest.cc"
    "profiling/profiling_init_unittest.cc"
    "profiling/global_profiling_unittest.cc"
)

set(HYBRID_TEST_FILES
    "hybrid/ge_hybrid_unittest.cc"
    "hybrid/executor/node_state_unittest.cc"
    "hybrid/executor/subgraph_executor_unittest.cc"
    "hybrid/executor/scalable_allocator_manager_unittest.cc"
    "hybrid/executor/worker/execution_engine_unittest.cc"
    "hybrid/model/hybrid_model_builder_unittest.cc"
    "hybrid/model/test_infer_shape_output.cc"
    "hybrid/node_executor/node_executor_unittest.cc"
    "hybrid/node_executor/task_context_unittest.cc"
    "hybrid/node_executor/rts/rts_node_task_unittest.cc"
    "hybrid/node_executor/host_cpu/host_cpu_node_task_unittest.cc"
    "hybrid/node_executor/ge_local/ge_local_node_executor_unittest.cc"
    "hybrid/node_executor/controlop/control_op_executor_unittest.cc"
    "hybrid/node_executor/compiledsubgraph/known_node_executor_unittest.cc"
    "hybrid/node_executor/hccl/hccl_node_executor_unittest.cc"
    "hybrid/node_executor/aicpu/aicpu_node_executor_unittest.cc"
    "hybrid/executor/hybrid_model_async_executor_unittest.cc"
    "hybrid/executor/hybrid_profiler_unittest.cc"
    "hybrid/executor/cann_trace_profiler_unittest.cc"
    "hybrid/executor/hybrid_model_rt1_executor_unittest.cc"
    "hybrid/executor/hybrid_model_pipeline_executor_unittest.cc"
    "hybrid/executor/hybrid_model_rt2_executor_unittest.cc"
    "${AIR_CODE_DIR}/tests/graph_metadef/depends/checker/tensor_check_utils.cc"
    "hybrid/executor/rt_v2_pipeline_executor_unittest.cc"
    "hybrid/executor/runtime_v2/rt_v2_utils_unittests.cc"
    "hybrid/node_executor/aicore/aicore_op_task_unittest.cc"
    "hybrid/node_executor/aicore/aicore_node_executor_unittest.cc"
    "hybrid/node_executor/aicore/aicore_task_builder_unittest.cc"
)

set(OTHERS_TEST_FILES
    "plugin_manager/ge_util_unittest.cc"
    "plugin_manager/plugin_manager_unittest.cc"
    "engine_manager/engine_manager_unittest.cc"
)

set(LABEL_MAKER_TEST_FILES
    "label_maker/label_maker_unittest.cc"
)

set(MANUAL_TEST_FILES
    "graph/build/memory_assigner_manual_test.cc"
    "${AIR_CODE_DIR}/tests/autofuse/ut/autofuse/fusion_strategy_solver_manual_test.cpp"
)

set(SYMBOL_SHAPE_INFER_FILE
    "${SYMBOL_SHAPE_INFER_FUNCS_TEST_SRCS}"
)

list(APPEND COMMON_SHARED_LIBRARIES
    c_sec
    slog_stub
    runtime_stub
    #tunner_stub
    profiler_stub
    mmpa_stub
    platform
    hccl_stub
    error_manager
    ascend_protobuf
    json
)

# build common
add_library(ge_ut_common STATIC ${COMMON_SRC_FILES} ${COMPILER_COMMON_SRC_FILES}
    ${EXECUTOR_COMMON_SRC_FILES} ${PROTO_HDRS})

target_include_directories(ge_ut_common PRIVATE
    ${AIR_CODE_DIR}
    ${ASCEND_INSTALL_PATH}/include
    ${ASCGEN_DIR}/inc
    ${AIR_CODE_DIR}/inc/graph_metadef
)

target_compile_definitions(ge_ut_common PRIVATE
    google=ascend_private
)

target_compile_options(ge_ut_common PRIVATE -g ${AIR_COV_COMPILE_OPTION}
    -Werror=format
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
    -Wno-subobject-linkage
)

target_link_options(ge_ut_common PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(ge_ut_common PRIVATE
    intf_llt_pub
    metadef_headers
    datagw_headers
    adump_headers
    c_sec
    ascend_protobuf
    json
    Boost::boost
    runtime_stub
    profiler_stub
    crypto_static
    ascendcl_stub
    minizip_static
)

target_link_libraries(ge_ut_common PUBLIC
    ge_graph_dsl
    ascendcl_stub)

# ut binary
file(GLOB_RECURSE JIT_EXECUTION_SRC_FILES CONFIGURE_DEPENDS "${AIR_CODE_DIR}/api/session/jit_execution/*.cc")
# libge_mutiparts_utest
add_executable(ut_libge_multiparts_utest
    ${COMMON_TEST_FILES}
    ${MULTI_PARTS_TEST_FILES}
    ${AIR_CODE_DIR}/tests/ge/st/stubs/src/mock_ops_kernel_builder.cc
)

target_compile_options(ut_libge_multiparts_utest PRIVATE
    -Werror=format
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
    -Wno-write-strings
    -fno-access-control
)

target_compile_definitions(ut_libge_multiparts_utest PRIVATE
    google=ascend_private
    FUNC_VISIBILITY
)

target_include_directories(ut_libge_multiparts_utest PRIVATE
    ${AIR_CODE_DIR}/tests/st/stubs
    ${AIR_CODE_DIR}/tests/framework/ge_runtime_stub/include
    ${ASCEND_INSTALL_PATH}/include
    ${ASCGEN_DIR}/inc
    ${CMAKE_BINARY_DIR}/proto/data_flow_protos
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${AIR_CODE_DIR}/tests/ge/st/stubs/
)

target_link_options(ut_libge_multiparts_utest PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(ut_libge_multiparts_utest
    intf_llt_pub
    metadef_headers
    ascendcl_stub
    -Wl,-z,muldefs
    symengine
    unified_dlog
    graph
    graph_base
    ge_executor_shared
    hybrid_executor
    davinci_executor
    lowering
    register
    ge_local_engine
    ge_local_opskernel_builder
    ge_runner
    udf_compiler
    ge_runner_v2
    engine
    json
    ascend_protobuf
    ge_graph_dsl
    ge_compiler
    ascendcl_stub
    -Wl,--whole-archive
    atc_static gert gert_op_impl es_ge_test
    -Wl,--no-whole-archive
    ge_running_env ge_runtime_stub GTest::gtest GTest::gtest_main GTest::gmock_main ${COMMON_SHARED_LIBRARIES} -lrt
    -ldl
    -Wl,--no-as-needed GTest::gmock -Wl,--as-needed
)

set_target_properties(ut_libge_multiparts_utest PROPERTIES CXX_STANDARD 17)

add_test(NAME ut_libge_multiparts_utest COMMAND ut_libge_multiparts_utest --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_libge_multiparts_utest.xml)
set_tests_properties(ut_libge_multiparts_utest PROPERTIES LABELS "ut;ge_common;ut_libge_multiparts_utest" DEPENDS ut_libge_others_utest)

# ut_libge_symbol_infer_utest
add_executable(ut_libge_symbol_infer_utest
        ${SYMBOL_SHAPE_INFER_FILE}
        ${JIT_EXECUTION_SRC_FILES}
)

target_compile_options(ut_libge_symbol_infer_utest PRIVATE
        -Werror=format
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal -Werror
        -Wno-write-strings
)

target_compile_definitions(ut_libge_symbol_infer_utest PRIVATE
        google=ascend_private
)

target_include_directories(ut_libge_symbol_infer_utest PRIVATE
        ${AIR_CODE_DIR}/tests/framework/ge_runtime_stub/include
        ${ASCEND_INSTALL_PATH}/include
        ${ASCGEN_DIR}/inc
        ${METADEF_DIR}/base/registry
        ${METADEF_DIR}/register
)

target_link_options(ut_libge_symbol_infer_utest PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        -Wl,--disable-new-dtags
        -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
        ${AIR_COMMON_LINK_OPTION}
        )

target_link_libraries(ut_libge_symbol_infer_utest
        intf_llt_pub
        -Wl,-z,muldefs
        symengine
        lowering
        register
        graph
        graph_base
        ge_compiler
        gert
        ascendcl
        ge_executor_shared
        hybrid_executor
        davinci_executor
        unified_dlog
        -Wl,--whole-archive
        rt2_registry_static
        es_ge_test
        -Wl,--no-whole-archive
        ge_runtime_stub st_stubs GTest::gtest_main GTest::gmock_main -lrt -ldl
        -Wl,--no-as-needed GTest::gmock -Wl,--as-needed
)

add_test(NAME ut_libge_symbol_infer_utest COMMAND ut_libge_symbol_infer_utest --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_libge_symbol_infer_utest.xml)
set_tests_properties(ut_libge_symbol_infer_utest PROPERTIES LABELS "ut;ge_common;ut_libge_symbol_infer_utest" DEPENDS ut_libge_multiparts_utest)

# libge_common_utest
add_executable(ut_libge_common_utest
    ${COMMON_TEST_FILES}
    ${BASE_COMMON_TEST_FILES}
    ${AIR_CODE_DIR}/tests/ge/ut/ge/ffts_plus_proto_tools.cc
    ${CMAKE_BINARY_DIR}/proto/ge/proto/tensorflow/graph_library.pb.cc
)

add_dependencies(ut_libge_common_utest ge)

target_compile_options(ut_libge_common_utest PRIVATE -g -O0
    -Werror=format
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
    -fno-access-control
)

target_compile_definitions(ut_libge_common_utest PRIVATE
    google=ascend_private
    AIR_CODE_DIR=\"${AIR_CODE_DIR}\"
    CMAKE_BINARY_DIR=\"${CMAKE_BINARY_DIR}\"
)

target_link_options(ut_libge_common_utest PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(ut_libge_common_utest
    intf_llt_pub
    metadef_headers
    ascendcl_stub
    symengine
    slice
    ge_graph_dsl
    ge_compiler
    ge_runner
    ge_executor_shared
    hybrid_executor
    davinci_executor
    lowering
    register
    graph
    graph_base
    gert
    unified_dlog
    -Wl,-z,muldefs
    -Wl,--whole-archive
    ge_runtime_stub
    es_ge_test
    gert_op_impl_static rt2_registry_static
    -Wl,--no-whole-archive
    minizip_static
    ge_running_env GTest::gtest GTest::gtest_main GTest::gmock_main ${COMMON_SHARED_LIBRARIES} -lrt -ldl
    -Wl,--no-as-needed GTest::gmock -Wl,--as-needed
)

target_link_options(ut_libge_common_utest PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        )

add_test(NAME ut_libge_common_utest COMMAND ut_libge_common_utest --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_libge_common_utest.xml)
set_tests_properties(ut_libge_common_utest PROPERTIES LABELS "ut;ge_common")
add_test(NAME ut_libge_common_utest_st COMMAND ut_libge_common_utest --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/ut_libge_common_utest.xml)
set_tests_properties(ut_libge_common_utest_st PROPERTIES LABELS "st;st_hetero")

# libge_others_utest
add_executable(ut_libge_others_utest
    ${COMMON_TEST_FILES}
    ${PASS_TEST_FILES}
    ${EXECUTE_TEST_FILES}
    ${OTHERS_TEST_FILES}
    ${GE_OPT_INFO_TEST_FILES}
    ${JIT_EXECUTION_SRC_FILES}
    ${DEPLOYER_PROTO_SRCS}
)

add_dependencies(ut_libge_others_utest mdat)

target_compile_options(ut_libge_others_utest PRIVATE ${AIR_COMMON_COMPILE_OPTION}
    -Werror=format
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
    -Wno-subobject-linkage
)

target_include_directories(ut_libge_others_utest PRIVATE
    ${AIR_CODE_DIR}/tests/framework/ge_runtime_stub/include
    ${ASCEND_INSTALL_PATH}/include
    ${ASCGEN_DIR}/inc
)

target_compile_definitions(ut_libge_others_utest PRIVATE
    google=ascend_private
)

target_link_options(ut_libge_others_utest PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(ut_libge_others_utest
    intf_llt_pub
    metadef_headers
    ascendcl_stub
    ${AIR_COMMON_LINK_OPTION}
    ge_compiler
    gert
    ge_runner
    ge_runner_v2
    gRPC::grpc++
    gRPC::grpc
    -Wl,--whole-archive
    es_ge_test
    rt2_registry_static
    -Wl,--no-whole-archive
    ge_running_env ge_runtime_stub GTest::gtest GTest::gtest_main GTest::gmock_main ${COMMON_SHARED_LIBRARIES} -lrt -ldl ${AIR_COMMON_LINK_OPTION}
    -Wl,--no-as-needed GTest::gmock -Wl,--as-needed
)

add_test(NAME ut_libge_others_utest COMMAND ut_libge_others_utest --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_libge_others_utest.xml)
set_tests_properties(ut_libge_others_utest PROPERTIES LABELS "ut;ge_common;ut_libge_others_utest" DEPENDS ut_libge_distinct_load_utest)

# libge_label_maker_utest
add_executable(ut_libge_label_maker_utest
    ${COMMON_TEST_FILES}
    ${LABEL_MAKER_SRC_FILES}
    ${LABEL_MAKER_TEST_FILES}
    ${JIT_EXECUTION_SRC_FILES}
)

target_compile_definitions(ut_libge_label_maker_utest PRIVATE
    google=ascend_private
)

target_compile_options(ut_libge_label_maker_utest PRIVATE -g -O0
    -Werror=format
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
    -fno-access-control
)

target_include_directories(ut_libge_label_maker_utest PRIVATE
    ${ASCEND_INSTALL_PATH}/include)

target_link_options(ut_libge_label_maker_utest PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

# TODO:ge_ut_common need shrink; it has too many depends so
target_link_libraries(ut_libge_label_maker_utest
    intf_llt_pub
    metadef_headers
    ascendcl_stub
    ge_common
    ge_common_base
    gert
    ge_compiler
    gRPC::grpc++
    gRPC::grpc
    -Wl,--whole-archive
    es_ge_test
    -Wl,--no-whole-archive
    ge_running_env GTest::gtest GTest::gtest_main GTest::gmock_main ${COMMON_SHARED_LIBRARIES} -lrt -ldl
    -Wl,--no-as-needed unified_dlog GTest::gmock -Wl,--as-needed
)

add_test(NAME ut_libge_label_maker_utest COMMAND ut_libge_label_maker_utest --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_libge_label_maker_utest.xml)
set_tests_properties(ut_libge_label_maker_utest PROPERTIES LABELS "ut;ge_common;ut_libge_label_maker_utest")

# libge_kernel_utest
add_executable(ut_libge_kernel_utest
        ${COMMON_TEST_FILES}
        ${KERNEL_TEST_FILES}
        ${FORMAT_TRANSFERS_TEST_FILES}
        ${KERNEL_SRC_FILES}
        ${JIT_EXECUTION_SRC_FILES}
        ${DEPLOYER_PROTO_SRCS}
)

target_compile_definitions(ut_libge_kernel_utest PRIVATE
    google=ascend_private
    FUNC_VISIBILITY
)

target_compile_options(ut_libge_kernel_utest PRIVATE -g
        -Werror=format
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal -Werror
        -fno-access-control
)

target_link_options(ut_libge_kernel_utest PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_include_directories(ut_libge_kernel_utest PRIVATE
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
)

target_link_libraries(ut_libge_kernel_utest
        intf_llt_pub
        metadef_headers
        -Wl,--whole-archive
        ascendcl_stub
        es_ge_test
        ge_common
        ge_common_base
        gert
        ge_compiler
        -Wl,--no-whole-archive
        ge_running_env GTest::gtest GTest::gtest_main GTest::gmock_main ${COMMON_SHARED_LIBRARIES} -lrt -ldl
        -Wl,--no-as-needed GTest::gmock -Wl,--as-needed
)

add_test(NAME ut_libge_kernel_utest COMMAND ut_libge_kernel_utest --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_libge_kernel_utest.xml)
set_tests_properties(ut_libge_kernel_utest PROPERTIES LABELS "ut;ge_common")
add_test(NAME ut_libge_kernel_utest_st COMMAND ut_libge_kernel_utest --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/ut_libge_kernel_utest.xml)
set_tests_properties(ut_libge_kernel_utest_st PROPERTIES LABELS "st;st_hetero")

# libge_distinct_load_utest
add_executable(ut_libge_distinct_load_utest
        ${COMMON_TEST_FILES}
        ${GENERATOR_TEST_FILES}
        ${DISTINCT_GRAPH_LOAD_TEST_FILES}
        ${SINGLE_OP_TEST_FILES}
        ${PROFILING_MNG_TEST_FILES}
        ${HYBRID_TEST_FILES}
        ${JIT_EXECUTION_SRC_FILES}
        "${AIR_CODE_DIR}/tests/ge/ut/ge/ffts_plus_proto_tools.cc"
)

target_compile_options(ut_libge_distinct_load_utest PRIVATE
    -Werror=format
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
    -fno-access-control
    ${AIR_COMMON_COMPILE_OPTION}
)

target_compile_definitions(ut_libge_distinct_load_utest PRIVATE
    google=ascend_private
)

target_include_directories(ut_libge_distinct_load_utest PRIVATE
    ${AIR_CODE_DIR}/base
    ${ASCGEN_DIR}/inc
    ${AIR_CODE_DIR}/runtime
    ${AIR_CODE_DIR}/runtime/device/remote
    ${AIR_CODE_DIR}/runtime/communication/cluster
    ${AIR_CODE_DIR}/runtime/communication
    ${AIR_CODE_DIR}/tests/ge/ut/ge/runtime/fast_v2
    ${AIR_CODE_DIR}/tests/depends/op_stub
    ${AIR_CODE_DIR}/tests/framework/ge_runtime_stub/include
    ${ASCEND_INSTALL_PATH}/include)

target_link_options(ut_libge_distinct_load_utest PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(ut_libge_distinct_load_utest
    intf_llt_pub
    metadef_headers
    ascendcl_stub
    datagw_headers
    msprof_headers
    symengine
    gert
    ge_compiler
    om2_executor
    -Wl,-z,muldefs
    -Wl,--whole-archive
    ge_graph_dsl ge_runtime_stub
    davinci_executor
    profiler_stub es_ge_test gert_op_impl_static rt2_registry_static
    -Wl,--no-whole-archive
    json
    minizip_static
    ge_running_env GTest::gtest GTest::gtest_main GTest::gmock_main ${COMMON_SHARED_LIBRARIES} -lrt -ldl ${AIR_COMMON_LINK_OPTION} -lpthread
    -Wl,--no-as-needed GTest::gmock -Wl,--as-needed
)

target_link_options(ut_libge_distinct_load_utest PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        )

add_test(NAME ut_libge_distinct_load_utest COMMAND ut_libge_distinct_load_utest --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_libge_distinct_load_utest.xml)
set_tests_properties(ut_libge_distinct_load_utest PROPERTIES LABELS "ut;ge_common;ut_libge_distinct_load_utest")

# ge_manual_test
add_executable(ge_manual_test
        ${MANUAL_TEST_FILES}
        )

target_compile_definitions(ge_manual_test PRIVATE
        google=ascend_private
        )

target_include_directories(ge_manual_test PRIVATE ${AIR_CODE_DIR}/tests/framework/ge_runtime_stub/include
        ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/autofuse
        ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/ascir/meta
        ${CMAKE_BINARY_DIR}/tests/depends/aihacb_autofusion
        ${CMAKE_BINARY_DIR}/graph_metadef/graph/ascendc_ir/generator/
        ${ASCEND_INSTALL_PATH}/include
        ${ASCGEN_DIR}/inc
)

target_link_options(ge_manual_test PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(ge_manual_test
        intf_llt_pub
        metadef_headers
        ascendcl_stub
        ge_compiler
        graph_base
        gert
        aihac_ir
        aihac_autofusion
        ge_running_env ge_runtime_stub GTest::gtest GTest::gtest_main GTest::gmock_main
        ${COMMON_SHARED_LIBRARIES} -lrt -ldl
        -Wl,--no-as-needed GTest::gmock -Wl,--as-needed
        )


file(GLOB_RECURSE JIT_EXECUTION_TEST_FILES CONFIGURE_DEPENDS "${AIR_CODE_DIR}/tests/ge/ut/ge/jit_execution/*.cc")
file(GLOB_RECURSE JIT_EXECUTION_UTIL_SRC_FILES CONFIGURE_DEPENDS "graph/passes/graph_builder_utils.cc")

# libjit_execution_utest
add_executable(ut_jit_execution
        ${JIT_EXECUTION_SRC_FILES}
        ${JIT_EXECUTION_TEST_FILES}
        ${JIT_EXECUTION_UTIL_SRC_FILES}
        "${SYMBOL_SHAPE_INFER_FUNCS_SRCS}"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/test.cc"
        "${AIR_CODE_DIR}/tests/graph_metadef/depends/checker/tensor_check_utils.cc"
        )

target_compile_options(ut_jit_execution PRIVATE -g -O0
        -Werror=format
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal -Werror
        -Wno-write-strings
        -fno-access-control
)

target_compile_definitions(ut_jit_execution PRIVATE
        google=ascend_private
)

target_include_directories(ut_jit_execution PRIVATE
        ${AIR_CODE_DIR}/tests/framework/ge_runtime_stub/include
        ${AIR_CODE_DIR}/tests/ge/st/stubs/utils
        ${ASCEND_INSTALL_PATH}/include
        ${ASCGEN_DIR}/inc
)

target_link_options(ut_jit_execution PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(ut_jit_execution
        intf_llt_pub
        metadef_headers
        ascendcl_stub
        ${AIR_COMMON_LINK_OPTION}
        -Wl,--whole-archive
        ge_common_base
        ge_graph_dsl
        ge_compiler
        ge_runner
        ge_runner_v2
        symengine
        aihacb_autofusion_stub
        gert
        rt2_registry_static
        es_ge_test
        -Wl,--no-whole-archive
        ge_ut_common
        ge_running_env ge_runtime_stub st_stubs GTest::gtest GTest::gtest_main GTest::gmock_main -lrt -ldl
        -Wl,--no-as-needed GTest::gmock -Wl,--as-needed
)

target_link_options(ut_jit_execution PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        )

add_test(NAME ut_jit_execution COMMAND ut_jit_execution --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_jit_execution.xml)
set_tests_properties(ut_jit_execution PROPERTIES LABELS "ut;ge_common" DEPENDS ut_libge_multiparts_utest)
add_test(NAME ut_jit_execution_st COMMAND ut_jit_execution --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/ut_jit_execution.xml)
set_tests_properties(ut_jit_execution_st PROPERTIES LABELS "st;st_hetero")

set(ES_HISTORY_REGISTRY_FIXTURE_SRC_DIR
        ${AIR_CODE_DIR}/tests/ge/ut/ge/graph/eager_style_graph_builder/es_generator/history/fixtures/history_registry)
file(COPY ${ES_HISTORY_REGISTRY_FIXTURE_SRC_DIR} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/fixtures)

add_executable(ut_eager_style_builder
        ${ES_GRAPH_BUILDER_TEST_SRCS}
        )
target_compile_options(ut_eager_style_builder PRIVATE -g -O0
        -Werror=format
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal -Werror
        -Wno-write-strings
        )

target_compile_definitions(ut_eager_style_builder PRIVATE
        google=ascend_private
        )

target_link_options(ut_eager_style_builder PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(ut_eager_style_builder
        intf_llt_pub
        -Wl,-z,muldefs
        metadef_headers
        es_generator_inner_headers
        json
        symengine
        Boost::boost
        -Wl,--whole-archive
        ge_common_base
        gen_esb_impl
        eager_style_graph_builder_base
        aihac_symbolizer
        ge_running_env
        ge_compiler
        GTest::gtest GTest::gtest_main GTest::gmock_main ${COMMON_SHARED_LIBRARIES} -lrt -ldl ${AIR_COMMON_LINK_OPTION}
        -Wl,--no-as-needed GTest::gmock -Wl,--as-needed
        ascendcl_stub
        )

add_test(NAME ut_eager_style_builder COMMAND ut_eager_style_builder --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/ut/ut_eager_style_builder.xml)
set_tests_properties(ut_eager_style_builder PROPERTIES LABELS "ut;ge_common" DEPENDS ut_libge_multiparts_utest)
add_test(NAME ut_eager_style_builder_st COMMAND ut_eager_style_builder --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/ut_eager_style_builder.xml)
set_tests_properties(ut_eager_style_builder_st PROPERTIES LABELS "st;st_hetero")

add_subdirectory(graph/eager_style_graph_builder/graph_construction_test)
add_subdirectory(graph/pyge_tests)

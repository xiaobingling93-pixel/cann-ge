/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_op_execute_kernel.h"
#include "common/checker.h"
#include "engine/aicore/fe_rt2_common.h"
#include "register/kernel_registry.h"
#include "exe_graph/runtime/op_execute_context.h"
#include "kernel/memory/multi_stream_mem_block.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"
#include "core/utils/tensor_utils.h"
#include "core/debug/kernel_tracing.h"
#include "acl/acl_rt.h"
#include "rts/rts_stream.h"
#include "kernel/common_kernel_impl/platform.h"

namespace gert {
namespace kernel {
const OpImplKernelRegistry::OpImplFunctionsV2 *GetOpImplFunctions(KernelContext *context) {
  auto node_type = context->GetInputValue<char *>(0);
  if (node_type == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Failed to find op execute func, node type is nullptr");
    return nullptr;
  }
  auto space_registry = context->GetInputValue<gert::OpImplSpaceRegistryV2 *>(1);
  if (space_registry == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Failed to find op execute func, space registry is nullptr");
    return nullptr;
  }
  return space_registry->GetOpImpl(node_type);
}

ge::graphStatus SetStreamCoreNumLimit(const rtStream stream, const int64_t op_aicore_num, const int64_t op_vec_core_num,
                                      bool &need_set_stream_aicore_num, bool &need_set_stream_vec_core_num) {
  need_set_stream_aicore_num = false;
  need_set_stream_vec_core_num = false;
  if (op_aicore_num > 0) {
    GE_CHK_RT_RET(aclrtSetStreamResLimit(stream, ACL_RT_DEV_RES_CUBE_CORE, static_cast<uint32_t>(op_aicore_num)));
    need_set_stream_aicore_num = true;
  }
  if (op_vec_core_num > 0) {
    GE_CHK_RT_RET(aclrtSetStreamResLimit(stream, ACL_RT_DEV_RES_VECTOR_CORE, static_cast<uint32_t>(op_vec_core_num)));
    need_set_stream_vec_core_num = true;
  }
  if (need_set_stream_aicore_num || need_set_stream_vec_core_num) {
    GE_CHK_RT_RET(aclrtUseStreamResInCurrentThread(stream));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResetStreamCoreNumLimit(const rtStream stream, const int64_t global_aicore_num, const int64_t global_vec_core_num,
                                        const bool need_set_stream_aicore_num, const bool need_set_stream_vec_core_num) {
  if (need_set_stream_aicore_num) {
    GE_ASSERT_TRUE(global_aicore_num >= 0);
    GE_CHK_RT_RET(aclrtSetStreamResLimit(stream, ACL_RT_DEV_RES_CUBE_CORE, static_cast<uint32_t>(global_aicore_num)));
  }
  if (need_set_stream_vec_core_num) {
    GE_ASSERT_TRUE(global_vec_core_num >= 0);
    GE_CHK_RT_RET(aclrtSetStreamResLimit(stream, ACL_RT_DEV_RES_VECTOR_CORE, static_cast<uint32_t>(global_vec_core_num)));
  }
  return ge::GRAPH_SUCCESS;
}


ge::graphStatus FindOpExeFunc(KernelContext *context) {
  auto op_fun_ptr = context->GetOutputPointer<OpImplKernelRegistry::OpExecuteFunc>(0);
  auto functions = GetOpImplFunctions(context);
  if (functions == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (functions->op_execute_func == nullptr || op_fun_ptr == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Failed to find op execute func, op impl functions is nullptr");
    return ge::GRAPH_FAILED;
  }
  *op_fun_ptr = functions->op_execute_func;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FindOpExe2PhaseFunc(KernelContext *context) {
  auto op_prepare_fun_ptr = context->GetOutputPointer<OpImplRegisterV2::OpExecPrepareFunc>(0);
  auto op_launch_fun_ptr = context->GetOutputPointer<OpImplRegisterV2::OpExecLaunchFunc>(1);
  auto functions = GetOpImplFunctions(context);
  if (functions == nullptr || functions->op_execute_prepare_func == nullptr ||
      functions->op_execute_launch_func == nullptr || op_prepare_fun_ptr == nullptr || op_launch_fun_ptr == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Failed to find op execute func, op impl functions is nullptr");
    return ge::GRAPH_FAILED;
  }
  *op_prepare_fun_ptr = functions->op_execute_prepare_func;
  *op_launch_fun_ptr = functions->op_execute_launch_func;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ExecuteOpFunc(KernelContext *context) {
  OpExecuteContext *op_execute_context = reinterpret_cast<OpExecuteContext *>(context);
  OpImplKernelRegistry::OpExecuteFunc op_execute_func =
      reinterpret_cast<OpImplKernelRegistry::OpExecuteFunc>(op_execute_context->GetOpExecuteFunc());

  const size_t input_num = op_execute_context->GetComputeNodeInputNum();
  const size_t output_num = op_execute_context->GetComputeNodeOutputNum();
  auto single_stage_aclnn_op_fwk_data = reinterpret_cast<gert::KernelContext *>(context)->GetInputPointer<SingleStageAclnnOpFwkData>(
      input_num + output_num + static_cast<size_t>(OpExecuteInputExtendIndex::kFwkData));
  GE_ASSERT_NOTNULL(single_stage_aclnn_op_fwk_data);
  auto core_num_infos = single_stage_aclnn_op_fwk_data->core_num_infos;
  GE_ASSERT_NOTNULL(core_num_infos);
  const auto stream = op_execute_context->GetStream();
  bool need_set_stream_aicore_num = false;
  bool need_set_stream_vec_core_num = false;
  GE_ASSERT_SUCCESS(SetStreamCoreNumLimit(stream, core_num_infos->op_aicore_num, core_num_infos->op_vec_core_num,
                                          need_set_stream_aicore_num, need_set_stream_vec_core_num));
  FE_ASSERT_NOTNULL(op_execute_func);
  FE_ASSERT_SUCCESS(op_execute_func(op_execute_context));
  GE_ASSERT_SUCCESS(ResetStreamCoreNumLimit(stream, core_num_infos->global_aicore_num, core_num_infos->global_vec_core_num,
                                            need_set_stream_aicore_num, need_set_stream_vec_core_num));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildSingleStageAclnnOpFwkData(KernelContext *context) {
  const auto core_num_infos = context->GetInputValue<CoreNumInfos *>(static_cast<size_t>(SingleStageAclnnOpFwkDataIndex::kCoreNumInfos));
  GE_ASSERT_NOTNULL(core_num_infos);
  const auto fwk_data = context->GetOutputPointer<SingleStageAclnnOpFwkData>(0UL);
  GE_ASSERT_NOTNULL(fwk_data);
  fwk_data->core_num_infos = core_num_infos;
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CreateBlockMemoryOutput(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto block_memory = context->GetOutput(static_cast<size_t>(OpExecuteOutputIndex::kBlockMemory));
  FE_ASSERT_NOTNULL(block_memory);
  auto mem_block_holder = new (std::nothrow) std::vector<memory::MultiStreamMemBlock *>();
  FE_ASSERT_NOTNULL(mem_block_holder);
  // 节省MallocWorkspace时vector添加元素时动态扩容的开销
  mem_block_holder->reserve(1U);
  block_memory->SetWithDefaultDeleter(mem_block_holder);
  return ge::GRAPH_SUCCESS;
}

static void *GetInputByIndex(KernelContext *context, size_t index) {
  auto *extended_kernel_context = reinterpret_cast<ExtendedKernelContext *>(context);
  const size_t input_num = extended_kernel_context->GetComputeNodeInputNum();
  const size_t output_num = extended_kernel_context->GetComputeNodeOutputNum();
  auto inner_data = context->GetInputPointer<void *>(input_num + output_num + index);
  if (inner_data == nullptr) {
    return nullptr;
  }
  return *inner_data;
}

ge::graphStatus ExecuteOpPrepare(KernelContext *context) {
  auto *op_prepare_context = reinterpret_cast<OpExecutePrepareContext *>(context);
  auto dual_stage_aclnn_op_fwk_data = reinterpret_cast<DualStageAclnnOpFwkData *>(
      GetInputByIndex(context, static_cast<size_t>(OpExecutePrepareInputExtendIndex::kFwkData)));
  GE_ASSERT_NOTNULL(dual_stage_aclnn_op_fwk_data);

  const auto stream = reinterpret_cast<rtStream>(
      GetInputByIndex(context, static_cast<size_t>(OpExecutePrepareInputExtendIndex::kStream)));

  bool need_set_stream_aicore_num = false;
  bool need_set_stream_vec_core_num = false;
  auto core_num_infos = dual_stage_aclnn_op_fwk_data->core_num_infos;
  GE_ASSERT_NOTNULL(core_num_infos);
  GE_ASSERT_SUCCESS(SetStreamCoreNumLimit(stream, core_num_infos->op_aicore_num, core_num_infos->op_vec_core_num,
                                          need_set_stream_aicore_num, need_set_stream_vec_core_num));

  const auto op_execute_prepare_func =
      reinterpret_cast<OpImplRegisterV2::OpExecPrepareFunc>(dual_stage_aclnn_op_fwk_data->op_execute_prepare_func);
  GE_ASSERT_NOTNULL(op_execute_prepare_func);
  GE_ASSERT_SUCCESS(op_execute_prepare_func(op_prepare_context));

  GE_ASSERT_SUCCESS(ResetStreamCoreNumLimit(stream, core_num_infos->global_aicore_num, core_num_infos->global_vec_core_num,
                                            need_set_stream_aicore_num, need_set_stream_vec_core_num));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ExecuteOpLaunch(KernelContext *context) {
  auto *op_launch_context = reinterpret_cast<OpExecuteLaunchContext *>(context);
  auto aclnn_op_fwk_data = reinterpret_cast<DualStageAclnnOpFwkData *>(
      GetInputByIndex(context, static_cast<size_t>(OpExecuteLaunchInputIndex::kFwkData)));
  GE_ASSERT_NOTNULL(aclnn_op_fwk_data);
  const auto op_execute_launch_func =
      reinterpret_cast<OpImplRegisterV2::OpExecLaunchFunc>(aclnn_op_fwk_data->op_execute_launch_func);
  GE_ASSERT_NOTNULL(op_execute_launch_func);
  GE_ASSERT_SUCCESS(op_execute_launch_func(op_launch_context));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildDualStageAclnnOpFwkData(KernelContext *context) {
  const auto execute_op_prepare_func =
      context->GetInputValue<void *>(static_cast<size_t>(DualStageAclnnOpFwkDataIndex::kExecutePrepareFunc));
  GE_ASSERT_NOTNULL(execute_op_prepare_func);
  const auto execute_op_launch_func =
      context->GetInputValue<void *>(static_cast<size_t>(DualStageAclnnOpFwkDataIndex::kExecuteLaunchFunc));
  GE_ASSERT_NOTNULL(execute_op_launch_func);
  const auto platform_info =
      context->GetInputValue<fe::PlatFormInfos *>(static_cast<size_t>(DualStageAclnnOpFwkDataIndex::kPlatformInfo));
  GE_ASSERT_NOTNULL(platform_info);
  const auto core_num_infos = context->GetInputValue<CoreNumInfos *>(static_cast<size_t>(DualStageAclnnOpFwkDataIndex::kCoreNumInfos));
  GE_ASSERT_NOTNULL(core_num_infos);
  const auto fwk_data = context->GetOutputPointer<DualStageAclnnOpFwkData>(0UL);
  GE_ASSERT_NOTNULL(fwk_data);
  fwk_data->op_execute_prepare_func = execute_op_prepare_func;
  fwk_data->op_execute_launch_func = execute_op_launch_func;
  fwk_data->platform_info = platform_info;
  fwk_data->core_num_infos = core_num_infos;
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CreateSingleStageAclnnOpFwkDataOutput(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto fwk_data_av = context->GetOutput(0UL);
  GE_ASSERT_NOTNULL(fwk_data_av);
  auto fwk_data_ptr = new (std::nothrow) SingleStageAclnnOpFwkData();
  fwk_data_av->SetWithDefaultDeleter(fwk_data_ptr);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CreateDualStageAclnnOpFwkDataOutput(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto fwk_data_av = context->GetOutput(0UL);
  GE_ASSERT_NOTNULL(fwk_data_av);
  auto fwk_data_ptr = new (std::nothrow) DualStageAclnnOpFwkData();
  fwk_data_av->SetWithDefaultDeleter(fwk_data_ptr);
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(FindOpExeFunc).RunFunc(FindOpExeFunc);
REGISTER_KERNEL(FindOpExe2PhaseFunc).RunFunc(FindOpExe2PhaseFunc);
REGISTER_KERNEL(BuildSingleStageAclnnOpFwkData).RunFunc(BuildSingleStageAclnnOpFwkData).OutputsCreator(CreateSingleStageAclnnOpFwkDataOutput);
REGISTER_KERNEL(ExecuteOpFunc)
    .RunFunc(ExecuteOpFunc)
    .OutputsCreator(CreateBlockMemoryOutput)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);
REGISTER_KERNEL(ExecuteOpPrepare).RunFunc(ExecuteOpPrepare);
REGISTER_KERNEL(ExecuteOpLaunch).RunFunc(ExecuteOpLaunch);
REGISTER_KERNEL(BuildDualStageAclnnOpFwkData).RunFunc(BuildDualStageAclnnOpFwkData).OutputsCreator(CreateDualStageAclnnOpFwkDataOutput);
}  // namespace kernel
}  // namespace gert

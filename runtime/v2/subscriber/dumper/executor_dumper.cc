/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor_dumper.h"

#include <utility>
#include <aicore/launch_kernel/ai_core_launch_kernel.h>
#include <tuning_utils.h>
#include <dlog_pub.h>
#include "common/checker.h"
#include "common/ge_inner_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "common/global_variables/diagnose_switch.h"
#include "exe_graph/lowering/lowering_definitions.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "framework/common/ge_types.h"
#include "graph/def_types.h"
#include "graph/ge_context.h"
#include "graph/utils/op_desc_utils.h"
#include "aicore/launch_kernel/rt_kernel_launch_args_ex.h"
#include "lowering/placement/placed_lowering_result.h"
#include "runtime/model_v2_executor.h"
#include "rt_error_codes.h"
#include "engine/aicpu/kernel/aicpu_ext_info_handle.h"
#include "engine/aicpu/kernel/aicpu_args_handler.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "graph/utils/type_utils.h"
#include "core/builder/node_types.h"
#include "framework/common/types.h"
#include "subscriber/subscriber_utils.h"
#include "runtime/gert_const_types.h"
#include "graph/load/model_manager/davinci_model.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"
#include "register/ffts_node_calculater_registry.h"
#include "graph/utils/tensor_utils.h"
#include "utils/utils.h"
#include "engine/aicore/kernel/mixl2_update_kernel.h" // todo: to be deleted
#include "engine/aicpu/kernel/ffts_plus/aicpu_update_kernel.h" // todo: to be deleted
#include "core/utils/executor_utils.h"
#include "kernel/known_subgraph/davinci_model_kernel.h"
#include "graph/utils/attr_utils.h"
#include "exe_graph/lowering/value_holder_utils.h"
#include "runtime/rts/rts_stream.h"
#include "runtime/rts/rts_kernel.h"

namespace gert {
namespace {
constexpr const char *kInvalidGraphName = "Invalid";
constexpr const char *kWhileNodeType = "While";
constexpr const char *kIfNodeType = "If";
constexpr const char *kCaseNodeType = "Case";
constexpr const char *kWaitAnyMoreType = "WaitAnyone";
constexpr const char *kExitType = "Exit";
constexpr size_t kControlFrameIdx = 0UL;
constexpr size_t kNodeMemParamInput = 1UL;
constexpr int64_t kScalarShapeSize = 1;
const ge::char_t *const kDumpOutput = "output";
const ge::char_t *const kDumpInput = "input";
const ge::char_t *const kDumpModeAll = "all";
const ge::char_t *const kRtsFftsPlusOpKernelName = "DNN_VM_RTS_FFTS_PLUS_OP_STORE";
const std::string kDumpStatusOpen = "on";

// kernel types no nedd add dependency in InitOrderHoldersFromExeGraph
const std::set<std::string> kKernelTypesNoNeedWait = {"SendEvents", "WaitEvents"};

ge::Status CopyH2D(const void *host_addr, const ge::GeTensorDesc &td, const size_t builtin_tensor_data_size,
                   std::vector<void *> &allocated_mem, std::vector<uintptr_t> &dump_addrs) {
  void *device_addr = nullptr;
  // scalar tensor dump with shape size 1
  const auto shape_size = std::max(td.GetShape().GetShapeSize(), kScalarShapeSize);
  if (ge::GetSizeByDataType(td.GetDataType()) < 0) {
    GELOGW("Calc][TensorSizeByShape] Get data type[%s] size less than zero.",
           ge::TypeUtils::DataTypeToSerialString(td.GetDataType()).c_str());
    return ge::GRAPH_FAILED;
  }
  const auto tensor_size = ge::GetSizeInBytes(shape_size, td.GetDataType());
  if (tensor_size < 0) {
    GELOGW("[Calc][TensorSizeByShape] shape_size[%" PRId64 "], data_type[%s]", shape_size,
           ge::TypeUtils::DataTypeToSerialString(td.GetDataType()).c_str());
    return ge::GRAPH_FAILED;
  }
  const bool is_tensor_size_valid =
      (builtin_tensor_data_size == 0UL) || (builtin_tensor_data_size >= static_cast<uint64_t>(tensor_size));
  GE_ASSERT_TRUE(is_tensor_size_valid, "Built in tensor data size %zu < calc_tensor_size %zu, do nothing.",
                 builtin_tensor_data_size, tensor_size);
  GE_ASSERT_RT_OK(rtMalloc(&device_addr, tensor_size, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  allocated_mem.emplace_back(device_addr);
  dump_addrs.emplace_back(reinterpret_cast<uintptr_t>(device_addr));
  GE_ASSERT_RT_OK(rtMemcpy(device_addr, tensor_size, host_addr, tensor_size, RT_MEMCPY_HOST_TO_DEVICE));
  return ge::SUCCESS;
}

ge::Status GetDumpAddrFromChainAddr(NodeDumpUnit &dump_unit, bool is_input, std::vector<void *> &allocated_mem,
                                    std::vector<uintptr_t> &dump_addrs) {
  const auto &chain_addrs = is_input ? dump_unit.input_addrs : dump_unit.output_addrs;
  const auto op_desc = dump_unit.node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const ge::char_t *in_or_out = is_input ? kDumpInput : kDumpOutput;

  for (size_t i = 0UL; i < chain_addrs.size(); ++i) {
    if (chain_addrs[i] == nullptr) {
      dump_addrs.emplace_back(reinterpret_cast<uintptr_t>(chain_addrs[i]));
      GELOGI("[Dumper] node %s %s[%zu] is null.", dump_unit.node->GetNamePtr(), in_or_out, i);
      continue;
    }
    const auto tensor_data = chain_addrs[i]->GetPointer<TensorData>();
    GE_ASSERT_NOTNULL(tensor_data);
    if (TensorPlacementUtils::IsOnHost(tensor_data->GetPlacement())) {
      GELOGD("[Dumper] The address of node %s is not device addr. Start copy h2d.", dump_unit.node->GetNamePtr());
      auto host_addr = tensor_data->GetAddr() == nullptr ? tensor_data : tensor_data->GetAddr();
      const auto &td = is_input ? op_desc->GetInputDesc(i) : op_desc->GetOutputDesc(i);
      if (CopyH2D(host_addr, td, tensor_data->GetSize(), allocated_mem, dump_addrs) != ge::SUCCESS) {
        GELOGW("[Dumper] Copy h2d failed, skip dump node %s", dump_unit.node->GetName().c_str());
        return ge::FAILED;
      }
    } else {
      dump_addrs.emplace_back(reinterpret_cast<uintptr_t>(tensor_data->GetAddr()));
      GELOGI("[Dumper] node %s %s[%zu] is %p.", dump_unit.node->GetNamePtr(), in_or_out, i, tensor_data->GetAddr());
    }
  }
  return ge::SUCCESS;
}

void UpdateShape(const Shape &storage_shape, ge::GeTensorDescPtr &tensor_desc) {
  // dump op get dim by range based
  std::vector<int64_t> dims(storage_shape.GetDimNum());
  for (size_t j = 0UL; j < storage_shape.GetDimNum(); ++j) {
    dims[j] = storage_shape.GetDim(j);
  }
  tensor_desc->SetShape(ge::GeShape(std::move(dims)));
}

void UpdateOriginalShape(const Shape &original_shape, ge::GeTensorDescPtr &tensor_desc) {
  // dump op get dim by range based
  std::vector<int64_t> dims(original_shape.GetDimNum());
  for (size_t j = 0UL; j < original_shape.GetDimNum(); ++j) {
    dims[j] = original_shape.GetDim(j);
  }
  tensor_desc->SetOriginShape(ge::GeShape(std::move(dims)));
}

ge::Status UpdateAddrsForExceptionDump(const NodeDumpUnit &dump_unit, const bool is_input,
                                       std::vector<void *> &extra_op_addrs) {
  const auto &chain_addrs = is_input ? dump_unit.input_addrs : dump_unit.output_addrs;
  for (size_t i = 0UL; i < chain_addrs.size(); ++i) {
    if (is_input && (dump_unit.node->GetOpDescBarePtr()->MutableInputDesc(static_cast<uint32_t>(i)) == nullptr)) {
      continue;
    }
    if (chain_addrs[i] == nullptr) {
      extra_op_addrs.emplace_back(nullptr);
      continue;
    }
    const auto tensor_data = chain_addrs[i]->GetPointer<TensorData>();
    if (tensor_data == nullptr) {
      extra_op_addrs.emplace_back(nullptr);
      continue;
    }
    extra_op_addrs.emplace_back(tensor_data->GetAddr());
  }
  return ge::SUCCESS;
}

bool IsDavinciModelExecute(const char *const kernel_type) {
  return (strcmp(kernel_type, "DavinciModelExecute") == 0);
}
bool IsNeedCheckOverflowNode(const char *const node_type) {
  return IsAiCoreLaunchNode(node_type) || IsAiCpuLaunchNode(node_type) || IsLaunchFFTSPlusTaskNode(node_type) ||
         IsExecuteOpFuncNode(node_type) || IsExecuteOplaunchNode(node_type) || IsDavinciModelExecute(node_type) ||
         IsCustomOpFuncNode(node_type);
}
ge::Status CheckOverflow(const Node &node, const rtStream_t stream, bool &is_overflow) {
  auto timeout = ge::GetContext().StreamSyncTimeout();
  const auto rt_ret = rtStreamSynchronizeWithTimeout(stream, timeout);
  if ((rt_ret == ACL_ERROR_RT_OVER_FLOW) || (rt_ret == ACL_ERROR_RT_AICORE_OVER_FLOW) ||
      (rt_ret == ACL_ERROR_RT_AIVEC_OVER_FLOW)) {
    is_overflow = true;
    const auto compute_node_name = static_cast<const ComputeNodeInfo *>(node.context.compute_node_info)->GetNodeName();
    const auto compute_node_type = static_cast<const ComputeNodeInfo *>(node.context.compute_node_info)->GetNodeType();
    GELOGW("[Overflow][Dumper]Dynamic shape op overflow has been detected, node[%s], type[%s].", compute_node_name,
           compute_node_type);
    return ge::SUCCESS;
  } else if (rt_ret == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
    GELOGE(rt_ret, "[Invoke][rtStreamSynchronizeWithTimeout] failed, stream synchronize timeout:%d, ret:%d.", timeout,
           rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "rtStreamSynchronizeWithTimeout failed, stream synchronize timeout:%d, ret:%d.",
                      timeout, rt_ret);
    return ge::FAILED;
  } else {
    GE_ASSERT_RT_OK(rt_ret);
    return ge::SUCCESS;
  }
}

bool GetKeyAndCallbackByKernel(const std::string &kernel_type, const KernelContext *context,
                               const rtFftsPlusTaskInfo_t **task_info, const gert::ContinuousVector **ctx_info,
                               void (*&set_ctx_dump_flag_callback)(rtFftsPlusComCtx_t *)) {
  const auto aicore_callback = [] (rtFftsPlusComCtx_t *addr) {
    reinterpret_cast<rtFftsPlusAicAivCtx_t *>(addr)->dumpSwitch = true;
  };
  const auto aicpu_callback = [] (rtFftsPlusComCtx_t *addr) {
    reinterpret_cast<rtFftsPlusAiCpuCtx_t *>(addr)->dumpSwitch = true;
  };
  if (IsAICoreUpdateContextNode(kernel_type.c_str())) {
    *task_info = context->GetInputPointer<rtFftsPlusTaskInfo_t>(static_cast<size_t>(kernel::UpdateKey::TASK_INFO));
    *ctx_info = context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(kernel::UpdateKey::AICORE_CTX));
    set_ctx_dump_flag_callback = aicore_callback;
  } else if (IsStaAutoUpdateContext(kernel_type.c_str())) {
    *task_info = context->GetInputPointer<rtFftsPlusTaskInfo_t>(static_cast<size_t>(kernel::AutoUpdateKey::TASK_INFO));
    *ctx_info =
        context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(kernel::AutoUpdateKey::AICORE_CTX));
    set_ctx_dump_flag_callback = aicore_callback;
  } else if (IsAICpuUpdateContextNode(kernel_type.c_str())) {
    *task_info = context->GetOutputPointer<rtFftsPlusTaskInfo_t>(0);
    *ctx_info =
        context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(kernel::UpdateContextInputIndex::kCtxIds));
    set_ctx_dump_flag_callback = aicpu_callback;
  } else {
    return false;
  }

  return true;
}

ge::Status NormalProcessor(const ge::OpDescPtr &op_desc, ge::ExceptionDumper *dumper, NodeDumpUnit &dump_unit,
                           ge::ExtraOpInfo &extra_dump_unit, rtStream_t &stream) {
  if (UpdateAddrsForExceptionDump(dump_unit, true, extra_dump_unit.input_addrs) == ge::SUCCESS &&
      UpdateAddrsForExceptionDump(dump_unit, false, extra_dump_unit.output_addrs) == ge::SUCCESS) {
    uint32_t task_id = 0U;
    uint32_t stream_id = 0U;
    int32_t device_id = 0;
    GE_CHK_RT_RET(rtsGetThreadLastTaskId(&task_id));
    GE_CHK_RT_RET(rtsStreamGetId(stream, reinterpret_cast<int32_t*>(&stream_id)));
    GE_CHK_RT_RET(rtGetDevice(&device_id));
    ge::OpDescInfoId id(task_id, stream_id, device_id);
    dumper->SaveDumpOpInfo(op_desc, extra_dump_unit, id, true);
  }
  return ge::SUCCESS;
}

ge::Status FftsPlusProcessor(const ge::OpDescPtr &op_desc, ge::ExceptionDumper *dumper, NodeDumpUnit &dump_unit,
                             ge::ExtraOpInfo &extra_dump_unit, const rtStream_t &stream) {
  (void) stream;
  int32_t device_id = 0;
  GE_CHK_RT_RET(rtGetDevice(&device_id));
  ge::OpDescInfoId id(UINT32_MAX, UINT32_MAX, device_id);
  for (const auto &ctx : dump_unit.context_list) {
    id.context_id = ctx.context_id;
    id.thread_id = ctx.thread_id;
    extra_dump_unit.input_addrs.clear();
    extra_dump_unit.input_sizes.clear();
    for (const auto &addr : ctx.input) {
      extra_dump_unit.input_addrs.emplace_back(reinterpret_cast<void *>(addr.address));
      extra_dump_unit.input_sizes.emplace_back(addr.size);
    }
    extra_dump_unit.output_addrs.clear();
    extra_dump_unit.output_sizes.clear();
    for (const auto &addr : ctx.output) {
      extra_dump_unit.output_addrs.emplace_back(reinterpret_cast<void *>(addr.address));
      extra_dump_unit.output_sizes.emplace_back(addr.size);
    }
    dumper->SaveDumpOpInfo(op_desc, extra_dump_unit, id, true);
  }
  return ge::SUCCESS;
}

enum class ProcessorType { kNormal = 0U, kFftsPlus, kEnd };
const std::array<
    std::function<ge::Status(const ge::OpDescPtr &, ge::ExceptionDumper *, NodeDumpUnit &, ge::ExtraOpInfo &, rtStream_t &)>,
    static_cast<uint32_t>(ProcessorType::kEnd)>
    processors = {NormalProcessor, FftsPlusProcessor};


ge::Status FindNodeNameFromSubGraph(const bg::ValueHolderPtr &order_holder, const std::string &node_type,
                                    std::string &node_name) {
  const auto cond_graph = ge::FastNodeUtils::GetSubgraphFromNode(order_holder->GetFastNode(), kControlFrameIdx);
  GE_ASSERT_NOTNULL(cond_graph);
  const auto holder_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(cond_graph, node_type.c_str());
  GE_ASSERT_NOTNULL(holder_node);
  node_name = holder_node->GetName();
  return ge::SUCCESS;
}
}  // namespace

/**
 * kernel_a ---- out_a                     out_b
 *                    \                  /
 *                      -> new_kernel_c
 *                   /                  \
 * kernel_b ---- out_b                     out_a
 * PassChangedKernels{{kernel_a, 0}, {new_kernel_c, 1}}
 * PassChangedKernels{{kernel_b, 0}, {new_kernel_c, 0}}
 **/
// replace old kernel name&idx to new kernel name&idx which is changed by pass.
KernelNameAndIdx ExecutorDumper::GetKernelNameAndIdxAfterPass(const ge::OpDesc *op_desc,
                                                              const KernelNameAndIdx &kernel_name_and_idx,
                                                              const NodeDumpUnit *dump_unit) const {
  auto pass_changed_info = op_desc->TryGetExtAttr(kPassChangedInfo, PassChangedKernels{});
  for (auto &pass_changed_kernel : pass_changed_info.pass_changed_kernels) {
    // 映射关系里记录了launch name时才去比较，目前只有CopyFlowLaunch场景记录了launch name
    if (!pass_changed_kernel.first.launch_name.empty()) {
      if (kernel_name_and_idx == pass_changed_kernel.first) {
        const auto iter = compute_node_name_to_launch_kernel_name_.find(dump_unit->node->GetName());
        if ((iter != compute_node_name_to_launch_kernel_name_.end()) &&
            (pass_changed_kernel.first.launch_name == iter->second)) {
          GELOGI("[Dumper] Pass changed %s[%d] %s to %s[%d]", kernel_name_and_idx.kernel_name.c_str(),
                 kernel_name_and_idx.idx, kernel_name_and_idx.launch_name.c_str(),
                 pass_changed_kernel.second.kernel_name.c_str(), pass_changed_kernel.second.idx);
          return pass_changed_kernel.second;
        }
      }
    } else if (kernel_name_and_idx == pass_changed_kernel.first) {
      GELOGI("[Dumper] Pass changed %s[%d] to %s[%d]", kernel_name_and_idx.kernel_name.c_str(), kernel_name_and_idx.idx,
             pass_changed_kernel.second.kernel_name.c_str(), pass_changed_kernel.second.idx);
      return pass_changed_kernel.second;
    }
  }
  return kernel_name_and_idx;
}

ge::Status ExecutorDumper::InitOutputChainFromEquivalentDataEdges(const KernelNameAndIdx &kernel_name_and_idx,
                                                                  std::vector<Chain *> &output_chain) {
  const auto exe_graph = extend_info_->exe_graph;
  const auto node = SubscriberUtils::FindNodeFromExeGraph(exe_graph.get(), kernel_name_and_idx.kernel_name);
  if (node == nullptr) {
    GELOGW("[Dumper] Can not find kernel %s", kernel_name_and_idx.kernel_name.c_str());
    output_chain.emplace_back(nullptr);
    return ge::SUCCESS;
  }
  GE_ASSERT_NOTNULL(node->GetExtendInfo());
  const auto symbol = node->GetExtendInfo()->GetOutputSymbol(kernel_name_and_idx.idx);
  GE_ASSERT_TRUE(symbol != ge::kInvalidSymbol, "[Dumper] Can not find out kernel [%s][%d] from equivalent data edges.",
                 node->GetNamePtr(), kernel_name_and_idx.idx);
  const auto value_iter = extend_info_->symbols_to_value.find(symbol);
  GE_ASSERT_TRUE(value_iter != extend_info_->symbols_to_value.cend(),
                 "[Dumper] Can not find out kernel [%s] from symbol to values.", node->GetNamePtr());
  output_chain.emplace_back(reinterpret_cast<Chain *>(value_iter->second));
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::InitOutputShapes(const PlacedLoweringResult &placed_lower_result, NodeDumpUnit *dump_unit) {
  auto &out_shapes = placed_lower_result.GetResult()->out_shapes;
  for (const auto &out_shape : out_shapes) {
    if (out_shape == nullptr) {
      continue;
    }
    auto name_and_idx = GetKernelNameAndIdxAfterPass(
        bg::ValueHolderUtils::GetNodeOpDescBarePtr(out_shape),
        {bg::ValueHolderUtils::GetNodeName(out_shape), out_shape->GetOutIndex()}, dump_unit);
    if ((InitOutputChainFromMainGraph(name_and_idx, dump_unit, dump_unit->output_shapes, false) == ge::SUCCESS) ||
        (InitOutputChainFromInitGraph(name_and_idx, dump_unit, dump_unit->output_shapes, false) == ge::SUCCESS)) {
      continue;
    } else {
      GELOGI("[Dumper] Find out shape kernel [%s] of node [%s] from equivalent data edges.",
             bg::ValueHolderUtils::GetNodeNameBarePtr(out_shape), dump_unit->node->GetNamePtr());
      GE_ASSERT_SUCCESS(InitOutputChainFromEquivalentDataEdges(name_and_idx, dump_unit->output_shapes),
                        "[Dumper] Can not find out shape kernel [%s] of node [%s] from equivalent data edges.",
                        bg::ValueHolderUtils::GetNodeNameBarePtr(out_shape), dump_unit->node->GetNamePtr());
    }
  }
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::InitOutputChainFromMainGraph(const KernelNameAndIdx &kernel_name_and_idx,
                                                        NodeDumpUnit *dump_unit, vector<Chain *> &output_chain,
                                                        bool is_input) {
  const std::string &kernel_name = kernel_name_and_idx.kernel_name;
  const auto iter = kernel_names_to_exe_nodes_.find(kernel_name);
  if (iter == kernel_names_to_exe_nodes_.cend()) {
    return ge::FAILED;  // kernel is not in main graph
  }

  const auto exe_node = iter->second;
  GE_ASSERT_NOTNULL(exe_node);
  const uint32_t idx = kernel_name_and_idx.idx;
  const auto curr_idx = output_chain.size();
  output_chain.emplace_back(reinterpret_cast<KernelContext *>(&exe_node->context)->GetOutput(idx));
  GELOGI("[Dumper][Init%s] kernel %s[%u] to node %s[%zu]", (is_input ? "Input" : "Output"), kernel_name.c_str(), idx,
         dump_unit->node->GetNamePtr(), curr_idx);

  if (is_input) {
    return ge::SUCCESS;
  }

  auto &dump_units = kernel_idxes_to_dump_units_[exe_node->node_id];
  const auto unit_iter = std::find(dump_units.cbegin(), dump_units.cend(), dump_unit);
  if (unit_iter == dump_units.cend()) {
    ++dump_unit->total_update_count;
    dump_units.emplace_back(dump_unit);
    GELOGI("[Dumper] Op %s add dependency to kernel %s, count: %zu", dump_unit->node->GetNamePtr(), kernel_name.c_str(),
           dump_unit->total_update_count);
  }
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::InitOutputChainFromInitGraph(const KernelNameAndIdx &kernel_name_and_idx,
                                                        const NodeDumpUnit *dump_unit,
                                                        std::vector<Chain *> &output_chain, bool is_input) {
  const std::string &kernel_name = kernel_name_and_idx.kernel_name;
  const uint32_t idx = kernel_name_and_idx.idx;
  const auto iter = init_kernel_names_to_exe_nodes_.find(kernel_name);
  const auto curr_idx = output_chain.size();
  if (iter != init_kernel_names_to_exe_nodes_.cend()) {
    output_chain.emplace_back(reinterpret_cast<KernelContext *>(&iter->second->context)->GetOutput(idx));
    GELOGI("[Dumper][Init%s] kernel %s[%u] to node %s[%zu]", (is_input ? "Input" : "Output"), kernel_name.c_str(), idx,
           dump_unit->node->GetNamePtr(), curr_idx);
    return ge::SUCCESS;
  }
  return ge::FAILED;
}

ge::Status ExecutorDumper::InitInputShapes(const LowerInputInfo &input_info, NodeDumpUnit *dump_unit) {
  auto &out_shapes = input_info.input_shapes;  // lower input is peer node's lower output
  const auto op_desc = dump_unit->node->GetOpDescBarePtr();
  GELOGI("[Dumper] node %s input size %zu, input shape num %zu.", dump_unit->node->GetNamePtr(),
         op_desc->GetAllInputsSize(), out_shapes.size());
  std::vector<Chain *> input_shapes;
  for (const auto &out_shape : out_shapes) {
    if (out_shape == nullptr) {
      continue;
    }

    const auto peer_op_desc = bg::ValueHolderUtils::GetNodeOpDescBarePtr(out_shape);
    const auto output_index = out_shape->GetOutIndex();
    const auto peer_op_name = peer_op_desc->GetName();
    auto name_and_idx = GetKernelNameAndIdxAfterPass(peer_op_desc, {peer_op_name, output_index}, dump_unit);
    if ((InitOutputChainFromMainGraph(name_and_idx, dump_unit, input_shapes, true) == ge::SUCCESS) ||
        (InitOutputChainFromInitGraph(name_and_idx, dump_unit, input_shapes, true) == ge::SUCCESS)) {
      continue;
    } else {
      GE_ASSERT_SUCCESS(InitOutputChainFromEquivalentDataEdges(name_and_idx, input_shapes),
                        "[Dumper] Can not find out shape kernel [%s][%d] for node [%s] from equivalent data edges.",
                        peer_op_name.c_str(), output_index, dump_unit->node->GetNamePtr());
      GELOGI("[Dumper][InitInput] shape for node [%s][%zu] from kernel[%s][%d] from equivalent data edges.",
             dump_unit->node->GetNamePtr(), input_shapes.size() - 1U, name_and_idx.kernel_name.c_str(),
             name_and_idx.idx);
    }
  }

  size_t idx = 0UL;
  for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); ++i) {
    if ((op_desc->MutableInputDesc(static_cast<uint32_t>(i)) == nullptr) || idx >= input_shapes.size()) {
      dump_unit->input_shapes.emplace_back(nullptr);
      continue;
    }
    dump_unit->input_shapes.emplace_back(input_shapes[idx++]);
  }
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::InitOutputAddrs(const PlacedLoweringResult &placed_lower_result, NodeDumpUnit *dump_unit) {
  auto &out_addrs = placed_lower_result.GetResult()->out_addrs;
  for (const auto &out_addr : out_addrs) {
    if (out_addr == nullptr) {
      continue;
    }
    auto name_and_idx =
        GetKernelNameAndIdxAfterPass(bg::ValueHolderUtils::GetNodeOpDescBarePtr(out_addr),
                                     {bg::ValueHolderUtils::GetNodeName(out_addr), out_addr->GetOutIndex()}, dump_unit);
    if ((InitOutputChainFromMainGraph(name_and_idx, dump_unit, dump_unit->output_addrs, false) == ge::SUCCESS) ||
        (InitOutputChainFromInitGraph(name_and_idx, dump_unit, dump_unit->output_addrs, false) == ge::SUCCESS)) {
      continue;
    } else {
      GELOGI("[Dumper] Find out addr kernel [%s] of node [%s] from equivalent data edges.",
             name_and_idx.kernel_name.c_str(), dump_unit->node->GetNamePtr());
      GE_ASSERT_SUCCESS(InitOutputChainFromEquivalentDataEdges(name_and_idx, dump_unit->output_addrs),
                        "[Dumper] Can not find out addr kernel [%s] of node [%s] from equivalent data edges.",
                        name_and_idx.kernel_name.c_str(), dump_unit->node->GetNamePtr());
    }
  }
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::InitInputAddrs(const LowerInputInfo &input_info, NodeDumpUnit *dump_unit) {
  auto &out_addrs = input_info.input_addrs;  // lower input is peer node's lower output
  std::vector<Chain *> input_addrs;
  const auto op_desc = dump_unit->node->GetOpDescBarePtr();
  GELOGI("[Dumper] node %s inpu size %zu, input addr num %zu.", dump_unit->node->GetNamePtr(),
         op_desc->GetAllInputsSize(), out_addrs.size());
  for (const auto &out_addr : out_addrs) {
    if (out_addr == nullptr) {
      continue;
    }

    const auto peer_op_desc = bg::ValueHolderUtils::GetNodeOpDescBarePtr(out_addr);
    const auto output_index = out_addr->GetOutIndex();
    const auto peer_op_name = peer_op_desc->GetName();
    auto name_and_idx = GetKernelNameAndIdxAfterPass(peer_op_desc, {peer_op_name, output_index}, dump_unit);
    if ((InitOutputChainFromMainGraph(name_and_idx, dump_unit, input_addrs, true) == ge::SUCCESS) ||
        (InitOutputChainFromInitGraph(name_and_idx, dump_unit, input_addrs, true) == ge::SUCCESS)) {
      continue;
    } else {
      GE_ASSERT_SUCCESS(InitOutputChainFromEquivalentDataEdges(name_and_idx, input_addrs),
                        "[Dumper] Can not find out addr kernel [%s][%d] for node [%s] from equivalent data edges.",
                        peer_op_name.c_str(), output_index, dump_unit->node->GetNamePtr());
      GELOGI("[Dumper][InitInput] addr for node [%s][%zu] from kernel[%s][%d] from equivalent data edges.",
             dump_unit->node->GetNamePtr(), input_addrs.size() - 1U, name_and_idx.kernel_name.c_str(),
             name_and_idx.idx);
    }
  }

  size_t idx = 0UL;
  for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); ++i) {
    if ((op_desc->MutableInputDesc(static_cast<uint32_t>(i)) == nullptr) || idx >= input_addrs.size()) {
      dump_unit->input_addrs.emplace_back(nullptr);
      continue;
    }
    dump_unit->input_addrs.emplace_back(input_addrs[idx++]);
  }
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::InitOrderHoldersFromExeGraph(const std::string &name, NodeDumpUnit *dump_unit) {
  const auto iter = kernel_names_to_exe_nodes_.find(name);
  if (iter != kernel_names_to_exe_nodes_.cend()) {
    const auto exe_node = iter->second;
    GE_ASSERT_NOTNULL(exe_node);
    const auto kernel_extend_info = reinterpret_cast<const KernelExtendInfo *>(exe_node->context.kernel_extend_info);
    GE_ASSERT_NOTNULL(kernel_extend_info);
    const auto kernel_type = kernel_extend_info->GetKernelType();
    if (kKernelTypesNoNeedWait.find(kernel_type) != kKernelTypesNoNeedWait.cend()) {
      return ge::SUCCESS;
    }
    auto &dump_units = kernel_idxes_to_dump_units_[exe_node->node_id];
    const auto dump_unit_iter = std::find(dump_units.cbegin(), dump_units.cend(), dump_unit);
    if (dump_unit_iter == dump_units.cend()) {
      ++dump_unit->total_update_count;
      dump_units.emplace_back(dump_unit);
      GELOGI("[Dumper] Op %s add dependency to kernel %s, count: %zu", dump_unit->node->GetNamePtr(), name.c_str(),
             dump_unit->total_update_count);
    }
    return ge::SUCCESS;
  }

  if (init_kernel_names_to_exe_nodes_.find(name) != init_kernel_names_to_exe_nodes_.cend()) {
    // order holder in init graph, do nothing
    return ge::SUCCESS;
  }

  return ge::FAILED;
}

ge::Status ExecutorDumper::InitOrderHoldersFromControlNodes(const bg::ValueHolderPtr &order_holder,
                                                            NodeDumpUnit *dump_unit) {
  const auto &node_type = bg::ValueHolderUtils::GetNodeType(order_holder);
  const auto &node_name = bg::ValueHolderUtils::GetNodeName(order_holder);
  if (node_type == kIfNodeType || node_type == kCaseNodeType) {
    std::string name;
    GE_ASSERT_SUCCESS(FindNodeNameFromSubGraph(order_holder, kWaitAnyMoreType, name));
    GE_ASSERT_SUCCESS(InitOrderHoldersFromExeGraph(name, dump_unit),
                      "[Dumper]Can not find order holder [%s] of node [%s] from subgraph of control nodes.",
                      node_name.c_str(), dump_unit->node->GetNamePtr());
  } else if (node_type == kWhileNodeType) {
    std::string name;
    GE_ASSERT_SUCCESS(FindNodeNameFromSubGraph(order_holder, kExitType, name));
    GE_ASSERT_SUCCESS(InitOrderHoldersFromExeGraph(name, dump_unit),
                      "[Dumper]Can not find order holder [%s] of node [%s] from subgraph of control nodes.",
                      node_name.c_str(), dump_unit->node->GetNamePtr());
  } else {
    GELOGW("[Dumper]Can not find order holder [%s] of node [%s] from subgraph of control nodes.", node_name.c_str(),
           dump_unit->node->GetNamePtr());
  }
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::InitOrderHolders(const PlacedLoweringResult &placed_lower_result, NodeDumpUnit *dump_unit) {
  const auto &order_holders = placed_lower_result.GetResult()->order_holders;
  for (const auto &order_holder : order_holders) {
    if (InitOrderHoldersFromExeGraph(bg::ValueHolderUtils::GetNodeName(order_holder), dump_unit) == ge::SUCCESS) {
      continue;
    } else {
      GELOGI("[Dumper] Find order holder [%s] of node [%s] from subgraph in control nodes.",
             bg::ValueHolderUtils::GetNodeNameBarePtr(order_holder), dump_unit->node->GetNamePtr());
      GE_ASSERT_SUCCESS(InitOrderHoldersFromControlNodes(order_holder, dump_unit));
    }
  }
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::InitDumpUnits() {
  const auto &compute_graph =
      extend_info_->exe_graph->TryGetExtAttr(kComputeGraph, std::make_shared<ge::ComputeGraph>(kInvalidGraphName));
  GE_ASSERT_NOTNULL(compute_graph);
  GE_ASSERT_TRUE(compute_graph->GetName() != kInvalidGraphName);
  for (const auto &node : compute_graph->GetAllNodes()) {
    if (IsOutputType(node->GetTypePtr())) {
      continue;
    }

    // init exception dump unit for computer node
    auto &extra_dump_unit = node_names_to_extra_units_[node->GetName()];
    (void)extra_dump_unit;

    auto dump_unit = &node_names_to_dump_units_[node->GetName()];
    dump_unit->node = node;
    node_names_to_dump_unit_wrappers_[node->GetName()] = ExecutorDataDumpInfoWrapper(dump_unit);

    auto placed_lower_result =
        node->GetOpDescBarePtr()->TryGetExtAttr(kLoweringResult, PlacedLoweringResult(nullptr, LowerResult()));
    if (InitOutputShapes(placed_lower_result, dump_unit) != ge::SUCCESS) {
      continue;
    }
    if (InitOutputAddrs(placed_lower_result, dump_unit) != ge::SUCCESS) {
      continue;
    }
    if (InitOrderHolders(placed_lower_result, dump_unit) != ge::SUCCESS) {
      continue;
    }

    auto input_info = node->GetOpDescBarePtr()->TryGetExtAttr(kLoweringInputInfo, LowerInputInfo());
    (void)InitInputShapes(input_info, dump_unit);
    (void)InitInputAddrs(input_info, dump_unit);
  }
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::InitByGraphType(SubExeGraphType type) {
  const auto exe_graph_executor = extend_info_->executor->GetExeGraphExecutor(type);
  GE_ASSERT_NOTNULL(exe_graph_executor);
  const auto execution_data = static_cast<const ExecutionData *>(exe_graph_executor->GetExecutionData());
  GE_ASSERT_NOTNULL(execution_data);
  const auto node_num = execution_data->base_ed.node_num;

  if (type == kMainExeGraph) {
    kernel_idxes_to_dump_units_.resize(node_num);
    exe_node_id_to_data_dump_filler_.resize(node_num);
    exe_node_id_to_exception_dump_filler_.resize(node_num);
  }

  for (size_t i = 0UL; i < node_num; ++i) {
    const auto kernel_extend_info = static_cast<const KernelExtendInfo *>(
        execution_data->base_ed.nodes[i]->context.kernel_extend_info);
    if (type == kMainExeGraph) {
      kernel_names_to_exe_nodes_[kernel_extend_info->GetKernelName()] = execution_data->base_ed.nodes[i];
      const auto kernel_funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel_extend_info->GetKernelType());
      if (kernel_funcs != nullptr) {
        exe_node_id_to_data_dump_filler_[i] = kernel_funcs->data_dump_info_filler;
        exe_node_id_to_exception_dump_filler_[i] = kernel_funcs->exception_dump_info_filler;
      }
    } else if (type == kInitExeGraph) {
      init_kernel_names_to_exe_nodes_[kernel_extend_info->GetKernelName()] = execution_data->base_ed.nodes[i];
    }
  }

  return ge::SUCCESS;
}

void ExecutorDumper::LoadDumpTaskForDavinciModels(const bool dump_enable) const {
  const auto execution_data = static_cast<const ExecutionData *>(
      extend_info_->executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData());
  if (!has_davinci_models_ || execution_data == nullptr) {
    return;
  }

  const auto &dump_properties = ge::DumpManager::GetInstance().GetDumpProperties(session_id_);
  for (size_t i = 0UL; i < execution_data->base_ed.node_num; ++i) {
    auto kernel_context = reinterpret_cast<KernelContext *>(&execution_data->base_ed.nodes[i]->context);
    const auto kernel_extend_info = static_cast<const KernelExtendInfo *>(kernel_context->GetKernelExtend());
    if (!IsDavinciModelExecute(kernel_extend_info->GetKernelType())) {
      continue;
    }

    GELOGI("[Dumper] Recognize davinci model in rt2, dump_enable %d.", dump_enable);
    auto davinci_model = kernel_context->MutableInputPointer<ge::DavinciModel>(
        static_cast<int32_t>(gert::kernel::InputsCommon::kDavinciModel));
    if (davinci_model == nullptr) {
      continue;
    }
    davinci_model->SetDumpProperties(dump_properties);
    dump_enable ? davinci_model->ReLoadDumpInfo() : davinci_model->UnloadDumpInfo();
  }
}

bool ExecutorDumper::IsDavinciModelExist() const {
  const auto execution_data = static_cast<const ExecutionData *>(
      extend_info_->executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData());
  if (execution_data == nullptr) {
    return false;
  }
  for (size_t i = 0UL; i < execution_data->base_ed.node_num; ++i) {
    auto kernel_context = reinterpret_cast<KernelContext *>(&execution_data->base_ed.nodes[i]->context);
    const auto kernel_extend_info = static_cast<const KernelExtendInfo *>(kernel_context->GetKernelExtend());
    if (IsDavinciModelExecute(kernel_extend_info->GetKernelType())) {
      GELOGI("[Dumper] Recognize davinci model in rt2.");
      return true;
    }
  }
  return false;
}

bool ExecutorDumper::IsSingleOpScene() const {
  if (extend_info_->root_graph != nullptr) {
    return ge::GraphUtils::IsSingleOpScene(extend_info_->root_graph);
  }
  return false;
}

ge::Status ExecutorDumper::Init() {
  if (is_inited_) {
    return ge::SUCCESS;
  }
  GELOGD("[Dumper] Start to init dumper.");
  GE_ASSERT_SUCCESS(InitByGraphType(kMainExeGraph));
  GE_ASSERT_SUCCESS(InitByGraphType(kInitExeGraph));
  GE_ASSERT_SUCCESS(CollectLaunchKernelName());
  GE_ASSERT_SUCCESS(InitDumpUnits());
  is_inited_ = true;
  GELOGD("[Dumper] Init dumper successfully.");
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::CollectLaunchKernelName() {
  const auto execution_data = static_cast<const ExecutionData *>(
      extend_info_->executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData());
  for (size_t i = 0U; i < execution_data->base_ed.node_num; i++) {
    const auto node = execution_data->base_ed.nodes[i];
    const auto ctx = &node->context;
    const auto kernel_extend_info = reinterpret_cast<const KernelExtendInfo *>(ctx->kernel_extend_info);
    GE_ASSERT_NOTNULL(kernel_extend_info);
    const auto kernel_type = kernel_extend_info->GetKernelType();
    const auto compute_node_info = reinterpret_cast<const ComputeNodeInfo *>(ctx->compute_node_info);
    if ((compute_node_info != nullptr) && IsLaunchNode(kernel_type)) {
      compute_node_name_to_launch_kernel_name_[compute_node_info->GetNodeName()] = kernel_extend_info->GetKernelName();
      GELOGD("compute node name: %s, launch kernel name: %s", compute_node_info->GetNodeName(),
             kernel_extend_info->GetKernelName());
    }
  }
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::OnExecutorDumperSwitch(void *ins, uint64_t enable_flags) {
  auto ess = static_cast<ExecutorDumper *>(ins);
  GE_ASSERT_NOTNULL(ess, "The instance is nullptr when switch dumper, ignore the event");
  uint64_t dynamic_switch_bits = (gert::BuiltInSubscriberUtil::EnableBit<gert::DumpType>(gert::DumpType::kDataDump) |
    gert::BuiltInSubscriberUtil::EnableBit<gert::DumpType>(gert::DumpType::kOverflowDump));
  const bool switch_enable = ((enable_flags & dynamic_switch_bits) != 0UL);
  ess->LoadDumpTaskForDavinciModels(switch_enable);
  return ge::SUCCESS;
}

ExecutorDumper::ExecutorDumper(const std::shared_ptr<const SubscriberExtendInfo> &extend_info)
    : extend_info_(extend_info),
      global_dumper_(GlobalDumper::GetInstance()),
      streams_(nullptr),
      is_op_debug_reg_(false) {
  if ((extend_info_ == nullptr) || (extend_info_->exe_graph == nullptr) || (extend_info_->executor == nullptr)) {
    GELOGE(ge::FAILED, "Exe graph is nullptr, dumper will do nothing.");
    return;
  }

  has_davinci_models_ = IsDavinciModelExist();
  if (has_davinci_models_) {  // only davinci_model need to do op_mapping_info load/unload
    GlobalDumper::GetInstance()->RegisterHandler(this, {this, ExecutorDumper::OnExecutorDumperSwitch});
    is_dumper_switch_reg_ = true;
  }

  if (global_dumper_->IsEnableSubscribeDump()) {
    if (Init() != ge::SUCCESS) {
      GELOGE(ge::FAILED, "Init dumper failed, dumper does nothing.");
      return;
    }
  }
}

ExecutorDumper::~ExecutorDumper() {
  if (is_dumper_switch_reg_) {
    GlobalDumper::GetInstance()->UnregisterHandler(this);
  }

  if (global_step_addr_ != 0U) {
    GE_CHK_RT(rtFree(ge::ValueToPtr(static_cast<uint64_t>(global_step_addr_))));
    global_step_addr_ = 0U;
  }
}

ge::Status ExecutorDumper::SaveWorkSpaceAddrForAiCpuLaunchCCNode(const Node &node) {
  const auto ctx = &node.context;
  const auto kernel_extend_info = reinterpret_cast<const KernelExtendInfo *>(ctx->kernel_extend_info);
  GE_ASSERT_NOTNULL(kernel_extend_info);
  const auto kernel_type = kernel_extend_info->GetKernelType();
  if (!IsAiCpuLaunchCCNode(kernel_type)) {
    return ge::SUCCESS;
  }

  GELOGI("[Dumper] workspace_info kernel_name:%s", kernel_type);
  const auto compute_node_info = reinterpret_cast<const ComputeNodeInfo *>(ctx->compute_node_info);
  GE_ASSERT_NOTNULL(compute_node_info);
  const auto node_name = compute_node_info->GetNodeName();
  std::vector<int64_t> space_type;
  if ((!ge::AttrUtils::GetListInt(node_names_to_dump_units_[node_name].node->GetOpDescBarePtr(),
                                  ge::ATTR_NAME_AICPU_WORKSPACE_TYPE, space_type)) ||
      (std::find(space_type.begin(), space_type.end(), ge::AicpuWorkSpaceType::CUST_LOG) == space_type.end())) {
    return ge::SUCCESS;
  }

  auto aicpu_ext = reinterpret_cast<const KernelContext *>(ctx)->GetInputPointer<AicpuExtInfoHandler>(4U);
  if ((aicpu_ext != nullptr) && (aicpu_ext->GetWorkSpaceInfo() != nullptr)) {
    auto space_info = aicpu_ext->GetWorkSpaceInfo();
    GE_ASSERT_NOTNULL(space_info);
    GELOGI("[Dumper] workspace_info aicpu_addr&size:%lu %lu", space_info->addr, space_info->size);
    node_names_to_dump_unit_wrappers_[node_name].AddWorkspace(space_info->addr, space_info->size);
  }
  return ge::SUCCESS;
}

void ExecutorDumper::GetLastKernelDumpUnits(const Node &node, std::vector<NodeDumpUnit *> &dump_nodes) {
  for (auto &dump_unit : kernel_idxes_to_dump_units_[node.node_id]) {
    if (++dump_unit->cur_update_count != dump_unit->total_update_count) {
      continue;
    }
    dump_nodes.emplace_back(dump_unit);
  }
}

void NodeDumpUnit::UpdateInputShapes(ge::OpDescPtr &op_desc) {
  for (size_t i = 0UL; i < input_shapes.size(); ++i) {
    auto input_desc = op_desc->MutableInputDesc(i);
    if ((input_desc == nullptr) || (input_shapes[i] == nullptr) ||
        (input_shapes[i]->GetPointer<StorageShape>() == nullptr)) {
      continue;
    }
    GELOGD("Op[%s, %s] input desc[shape:%s, original_shape:%s, dtype:%d, format:%d]", op_desc->GetNamePtr(),
           op_desc->GetTypePtr(), input_desc->GetShape().ToString().c_str(),
           input_desc->GetOriginShape().ToString().c_str(), input_desc->GetDataType(), input_desc->GetFormat());
    const auto &storage_shape = input_shapes[i]->GetPointer<StorageShape>()->GetStorageShape();
    const auto &original_shape = input_shapes[i]->GetPointer<StorageShape>()->GetOriginShape();
    UpdateShape(storage_shape, input_desc);
    UpdateOriginalShape(original_shape, input_desc);
    GELOGD("Update op[%s:%s] input shape, storage shape[%s] original shape[%s]", op_desc->GetNamePtr(),
           op_desc->GetTypePtr(), input_desc->GetShape().ToString().c_str(),
           input_desc->GetOriginShape().ToString().c_str());
  }
}

void NodeDumpUnit::UpdateOutputShapes(ge::OpDescPtr &op_desc) {
  for (size_t i = 0UL; i < output_shapes.size(); ++i) {
    auto output_desc = op_desc->MutableOutputDesc(i);
    if ((output_desc == nullptr) || (output_shapes[i] == nullptr) ||
        (output_shapes[i]->GetPointer<StorageShape>() == nullptr)) {
      continue;
    }
    GELOGD("Op[%s, %s] output desc[shape:%s, original_shape:%s, dtype:%d, format:%d]", op_desc->GetNamePtr(),
           op_desc->GetTypePtr(), output_desc->GetShape().ToString().c_str(),
           output_desc->GetOriginShape().ToString().c_str(), output_desc->GetDataType(), output_desc->GetFormat());
    auto &storage_shape = output_shapes[i]->GetPointer<StorageShape>()->GetStorageShape();
    auto &original_shape = output_shapes[i]->GetPointer<StorageShape>()->GetOriginShape();
    UpdateShape(storage_shape, output_desc);
    UpdateOriginalShape(original_shape, output_desc);
    GELOGD("Update op[%s:%s] output shape, storage shape[%s] original shape[%s]", op_desc->GetNamePtr(),
           op_desc->GetTypePtr(), output_desc->GetShape().ToString().c_str(),
           output_desc->GetOriginShape().ToString().c_str());
  }
}

bool ExecutorDumper::IsOpInDumpList(const ge::DumpProperties &dump_properties, const std::string &op_name) const {
  if (dump_properties.IsOpDebugOpen() || (IsSingleOpScene() && dump_properties.IsSingleOpNeedDump())) {
    return true;
  }
  if (dump_properties.IsLayerNeedDump(
    extend_info_->model_name, extend_info_->model_data.om_name, op_name) || IsInDumpOpRange(op_name)) {
    return true;
  }
  std::vector<std::string> original_names;
  const auto iter = node_names_to_dump_units_.find(op_name);
  if (iter == node_names_to_dump_units_.end()) {
    GELOGW("dump unit find op name: %s failed", op_name.c_str());
    return false;
  }
  if (ge::AttrUtils::GetListStr(iter->second.node->GetOpDescBarePtr(),
    ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names) && !original_names.empty()) {
    for (const auto &name : original_names) {
      if (dump_properties.IsLayerNeedDump(extend_info_->model_name, extend_info_->model_data.om_name, name)) {
        return true;
      }
    }
  }
  return false;
}

ge::Status ExecutorDumper::UpdateFftsplusLaunchTask(const Node *node) {
  const auto kernel_type =
      reinterpret_cast<const KernelExtendInfo *>(node->context.kernel_extend_info)->GetKernelType();
  if ((!ffts_dump_op_.IsFftsDumpInfoEmpty()) && (IsLaunchFFTSPlusTaskNode(kernel_type))) {
    ffts_dump_op_.SetDynamicModelInfo(extend_info_->model_name, extend_info_->model_data.om_name,
                                      extend_info_->model_id);
    const auto dump_properties = ge::DumpManager::GetInstance().GetDumpProperties(ge::kInferSessionId);
    if (!IsSingleOpScene()) {
      ffts_dump_op_.SetLoopAddr(global_step_addr_, 0U, 0U);
    }
    void *load_dump_info = nullptr;
    uint32_t load_dump_len = 0U;
    void *unload_dump_info = nullptr;
    uint32_t unload_dump_len = 0U;
    ffts_dump_op_.GenerateFftsDump(dump_properties, load_dump_info, load_dump_len,
                                   unload_dump_info, unload_dump_len, IsSingleOpScene());
    const auto context = reinterpret_cast<const KernelContext *>(&node->context);
    auto kernel_context = const_cast<KernelContext *>(context);
    auto task_info_para = kernel_context->GetInputValue<NodeMemPara *>(kNodeMemParamInput);
    GE_CHECK_NOTNULL(task_info_para);
    auto task_info = reinterpret_cast<TransTaskInfo *>(task_info_para->host_addr);
    GE_CHECK_NOTNULL(task_info);
    task_info->rt_task_info.fftsPlusDumpInfo.loadDumpInfo = load_dump_info;
    task_info->rt_task_info.fftsPlusDumpInfo.unloadDumpInfo = unload_dump_info;
    task_info->rt_task_info.fftsPlusDumpInfo.loadDumpInfolen = load_dump_len;
    task_info->rt_task_info.fftsPlusDumpInfo.unloadDumpInfolen = unload_dump_len;
  }

  return ge::SUCCESS;
}

ge::Status ExecutorDumper::DoDataDump(NodeDumpUnit &dump_unit, const ge::DumpProperties &dump_properties,
                                      const Node *exe_node) {
  const auto &name = dump_unit.node->GetName();
  const auto op_desc = dump_unit.node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  if (!IsOpInDumpList(dump_properties, name) || (op_desc->GetOpKernelLibName() == kRtsFftsPlusOpKernelName)) {
    GELOGI("[Dumper] [%s] is not in dump list, no need to dump", name.c_str());
    dump_unit.Clear();
    return ge::SUCCESS;
  }

  const auto &type = dump_unit.node->GetType();
  GELOGI("[Dumper] Start to dump, op name: %s, type: %s", name.c_str(), type.c_str());

  std::vector<void *> allocated_input_mem{};
  std::vector<void *> allocated_output_mem{};
  const auto callback = [&allocated_input_mem, &allocated_output_mem, &dump_unit]() {
    for (auto &mem : allocated_input_mem) {
      GE_CHK_RT(rtFree(mem));
    }
    for (auto &mem : allocated_output_mem) {
      GE_CHK_RT(rtFree(mem));
    }
    dump_unit.Clear();
  };
  GE_MAKE_GUARD(dump_release, callback);

  ge::DumpOp dump_op;
  dump_op.SetDynamicModelInfo(extend_info_->model_name, extend_info_->model_data.om_name, extend_info_->model_id);

  ge::OpDescPtr op_desc_dump = nullptr;
  GE_MAKE_SHARED(op_desc_dump = std::make_shared<ge::OpDesc>(*op_desc), return ge::FAILED);
  dump_unit.UpdateInputShapes(op_desc_dump);
  dump_unit.UpdateOutputShapes(op_desc_dump);

  if (!dump_unit.context_list.empty()) {
    ffts_dump_op_.SaveFftsSubOpInfo(op_desc_dump, dump_unit.context_list);
    GELOGI("Save ffts dump op:%s Successfully", name.c_str());
    return ge::SUCCESS;
  }

  std::vector<uintptr_t> input_addrs;
  if (GetDumpAddrFromChainAddr(dump_unit, true, allocated_input_mem, input_addrs) != ge::SUCCESS) {
    // skip and continue dump other nodes
    return ge::SUCCESS;
  }
  std::vector<uintptr_t> output_addrs;
  if (GetDumpAddrFromChainAddr(dump_unit, false, allocated_output_mem, output_addrs) != ge::SUCCESS) {
    // skip and continue dump other nodes
    return ge::SUCCESS;
  }

  if ((input_addrs.size() != op_desc->GetAllInputsSize()) ||
      (output_addrs.size() != op_desc->GetAllOutputsDescSize())) {
    GELOGW(
        "[Dumper] Node %s input addr or output addr size is invalid, input addr size is %zu, output addr size is %zu, "
        "op desc input size is %zu, output size is %zu.",
        name.c_str(), input_addrs.size(), output_addrs.size(), op_desc->GetInputsSize(), op_desc->GetOutputsSize());
    // skip and continue dump
    return ge::SUCCESS;
  }

  const auto execution_data = static_cast<const ExecutionData *>(
      extend_info_->executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData());
  const auto stream_idx = CalcArgIndex(execution_data->base_ed.input_num, ExecuteArgIndex::kStream);
  auto rt_streams =
      reinterpret_cast<Chain *>(execution_data->base_ed.input_values[stream_idx])->GetValue<ContinuousVector *>();
  GE_ASSERT_NOTNULL(rt_streams);
  auto stream = *(reinterpret_cast<rtStream_t *>(rt_streams->MutableData()) + 0U);
  GE_ASSERT_SUCCESS(GetKernelStream(exe_node, stream));
  dump_op.SetDumpInfo(dump_properties, op_desc_dump, input_addrs, output_addrs, stream);
  dump_op.SetWorkspaceAddrs(dump_unit.workspace_info);
  // single_op does not have step, 0U is default value which is different from graph
  if (!IsSingleOpScene()) {
    dump_op.SetLoopAddr(global_step_addr_, 0U, 0U);
  }
  GELOGD("[Dumper] Is single op %d", static_cast<int32_t>(IsSingleOpScene()));
  GE_ASSERT_SUCCESS(dump_op.LaunchDumpOp(IsSingleOpScene()),
                    "[Dumper] Launch DumpOp failed in hybrid model.");
  GE_ASSERT_SUCCESS(DoRtStreamSyncWithTimeout(stream));
  GELOGI("[Dumper] Launch dump op:%s Successfully", name.c_str());
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::InsertHcclDumpOp(const KernelRunContext &context, ExecutorEvent event) {
  const auto compute_node_info = static_cast<const ComputeNodeInfo *>(context.compute_node_info);
  GE_ASSERT_NOTNULL(compute_node_info);
  const auto mode = (event == ExecutorEvent::kExecuteStart ? kDumpInput : kDumpOutput);
  auto hccl_dump_properties = ge::DumpManager::GetInstance().GetDumpProperties(session_id_);
  const auto cur_mode = hccl_dump_properties.GetDumpMode();
  if (cur_mode == mode || cur_mode == kDumpModeAll) {
    hccl_dump_properties.ClearOpDebugFlag();
    hccl_dump_properties.SetDumpMode(mode);
    GE_ASSERT_SUCCESS(
        DoDataDump(node_names_to_dump_units_[compute_node_info->GetNodeName()], hccl_dump_properties));
  }
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::SetDumpFlagForMixl2(const Node *node) const {
  const auto compute_node_info = static_cast<const ComputeNodeInfo *>(node->context.compute_node_info);
  GE_ASSERT_NOTNULL(compute_node_info);
  const auto dump_properties = ge::DumpManager::GetInstance().GetDumpProperties(ge::kInferSessionId);
  if (!IsOpInDumpList(dump_properties, compute_node_info->GetNodeName())) {
    return ge::SUCCESS;
  }

  const auto context = reinterpret_cast<const KernelContext *>(&node->context);
  const auto task_info =
      context->GetInputValue<rtFftsPlusTaskInfo_t*>(static_cast<size_t>(kernel::MixL2UpdateKey::TASK_INFO));
  GE_ASSERT_NOTNULL(task_info);
  const auto ctx_id = context->GetInputValue<uint32_t>(static_cast<size_t>(kernel::MixL2UpdateKey::CTX_ID));
  GE_ASSERT_TRUE(ctx_id < task_info->fftsPlusSqe->totalContextNum);
  const auto context_head = static_cast<rtFftsPlusComCtx_t *>(const_cast<void *>(task_info->descBuf));
  GE_ASSERT_NOTNULL(context_head);
  reinterpret_cast<rtFftsPlusMixAicAivCtx_t*>(context_head + ctx_id)->dumpSwitch = true;
  GELOGI("Set dump flag for mixl2 node %s, context id %u.", compute_node_info->GetNodeName(), ctx_id);
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::SetDumpFlagForFfts(const std::string &kernel_type, const Node *node) const {
  if (IsMixL2UpdateContext(kernel_type.c_str())) {
    return SetDumpFlagForMixl2(node);
  }

  const auto context = reinterpret_cast<const KernelContext *>(&node->context);
  const rtFftsPlusTaskInfo_t *task_info = nullptr;
  const gert::ContinuousVector *ctx_ids = nullptr;
  void (*set_ctx_dump_flag_callback)(rtFftsPlusComCtx_t *) = nullptr;
  if (!GetKeyAndCallbackByKernel(kernel_type, context, &task_info, &ctx_ids, set_ctx_dump_flag_callback)) {
    return ge::SUCCESS;
  }
  GE_ASSERT_NOTNULL(task_info);
  GE_ASSERT_NOTNULL(ctx_ids);
  GE_ASSERT_NOTNULL(set_ctx_dump_flag_callback);

  const auto compute_node_info = static_cast<const ComputeNodeInfo *>(node->context.compute_node_info);
  GE_ASSERT_NOTNULL(compute_node_info);
  const auto dump_properties = ge::DumpManager::GetInstance().GetDumpProperties(ge::kInferSessionId);
  if (!IsOpInDumpList(dump_properties, compute_node_info->GetNodeName())) {
    return ge::SUCCESS;
  }

  auto *context_head = static_cast<rtFftsPlusComCtx_t *>(const_cast<void *>(task_info->descBuf));
  GE_ASSERT_NOTNULL(context_head);
  const auto ctx_id_vec = static_cast<const int32_t *>(ctx_ids->GetData());
  const auto ctx_num = ctx_ids->GetSize();
  const auto total_num = task_info->fftsPlusSqe->totalContextNum;
  std::stringstream ctx_ss;
  for (size_t idx = 0U; idx < ctx_num; ++idx) {
    GE_ASSERT_TRUE(ctx_id_vec[idx] < total_num);
    set_ctx_dump_flag_callback(context_head + ctx_id_vec[idx]);
    ctx_ss << ctx_id_vec[idx] << ",";
  }
  GELOGI("Set dump flag for ffts+ node %s, context list [%s].", compute_node_info->GetNodeName(), ctx_ss.str().c_str());
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::FillExtraDumpInfo(const Node &node) {
  const auto ctx = &node.context;
  const auto compute_node_info = static_cast<const ComputeNodeInfo *>(ctx->compute_node_info);
  if (compute_node_info == nullptr) {
    return ge::SUCCESS;
  }

  const auto filler = exe_node_id_to_exception_dump_filler_[node.node_id];
  if (filler == nullptr) {
    return ge::SUCCESS;
  }

  const auto node_name = compute_node_info->GetNodeName();
  const auto kernel_extend_info = static_cast<const KernelExtendInfo *>(node.context.kernel_extend_info);
  GE_ASSERT_NOTNULL(kernel_extend_info);
  const auto kernel_type = kernel_extend_info->GetKernelType();
  ge::Status ret = ge::SUCCESS;
  ExecutorExceptionDumpInfoWrapper warpper(&node_names_to_extra_units_[node_name]);
  ret = filler(reinterpret_cast<const KernelContext *>(ctx), static_cast<ExceptionDumpInfoWrapper &>(warpper));
  GE_ASSERT_SUCCESS(ret, "Dump filler failed, node %s, kernel %s.", node_name, kernel_type);
  GELOGI("Exception dump filler, node %s, kernel %s.", node_name, kernel_type);
  return ret;
}

ge::Status ExecutorDumper::FillDumpInfoByKernel(const Node &node) {
  const auto ctx = &node.context;
  const auto compute_node_info = static_cast<const ComputeNodeInfo *>(ctx->compute_node_info);
  if (compute_node_info == nullptr) {
    return ge::SUCCESS;
  }

  const auto filler = exe_node_id_to_data_dump_filler_[node.node_id];
  if (filler == nullptr) {
    return ge::SUCCESS;
  }

  const auto node_name = compute_node_info->GetNodeName();
  const auto kernel_extend_info = static_cast<const KernelExtendInfo *>(node.context.kernel_extend_info);
  GE_ASSERT_NOTNULL(kernel_extend_info);
  const auto kernel_type = kernel_extend_info->GetKernelType();
  ge::Status ret = ge::SUCCESS;
  ret = filler(reinterpret_cast<const KernelContext *>(ctx),
               static_cast<DataDumpInfoWrapper &>(node_names_to_dump_unit_wrappers_[node_name]));
  GE_ASSERT_SUCCESS(ret, "Dump filler failed, node %s, kernel %s.", node_name, kernel_type);
  GELOGI("Dump filler, node %s, kernel %s.", node_name, kernel_type);
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::UpdateStepNum() {
  void *step_id = nullptr;
  if (global_step_addr_ == 0U) {
    GE_CHK_RT_RET(rtMalloc(&step_id, sizeof(uint64_t), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
    global_step_addr_ = static_cast<uintptr_t>(ge::PtrToValue(step_id));
  }
  step_id = ge::ValueToPtr(static_cast<uint64_t>(global_step_addr_));
  GELOGI("Update step, addr:%p, iteration_num:%zu", step_id, iteration_num_);
  GE_ASSERT_RT_OK(rtMemcpy(step_id, sizeof(uint64_t), &iteration_num_, sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE));
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::DoStreamSyncAfterFftsTask(const Node *node) {
  const auto kernel_type =
    reinterpret_cast<const KernelExtendInfo *>(node->context.kernel_extend_info)->GetKernelType();
  if (!IsLaunchFFTSPlusTaskNode(kernel_type)) {
    return ge::SUCCESS;
  }

  const auto execution_data = static_cast<const ExecutionData *>(
      extend_info_->executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData());
  const auto stream_idx = CalcArgIndex(execution_data->base_ed.input_num, ExecuteArgIndex::kStream);
  auto rt_streams =
      reinterpret_cast<Chain *>(execution_data->base_ed.input_values[stream_idx])->GetValue<ContinuousVector *>();
  GE_ASSERT_NOTNULL(rt_streams);
  auto stream = *(reinterpret_cast<rtStream_t *>(rt_streams->MutableData()) + 0U);
  GE_ASSERT_SUCCESS(GetKernelStream(node, stream));
  GE_ASSERT_SUCCESS(DoRtStreamSyncWithTimeout(stream));
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::ResetDumpFsmState() {
  dump_fsm_state_.clear();
  GE_ASSERT_NOTNULL(extend_info_);
  auto dump_properties = ge::DumpManager::GetInstance().GetDumpProperties(session_id_);
  size_t num = dump_properties.GetDumpOpRangeSize(extend_info_->model_name, extend_info_->model_data.om_name);
  if (num > 0U) {
    GELOGI("Model[%s] om name[%s] opname range size[%zu].",
      extend_info_->model_name.c_str(), extend_info_->model_data.om_name.c_str(), num);
    dump_fsm_state_.resize(num, ge::DumpProcState::kInit);
  }
  dump_op_in_range_.clear();
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::SetDumpFsmState(const Node *node, const char *const node_type) {
  if (dump_fsm_state_.empty()) {
    return ge::SUCCESS;
  }

  const auto compute_node_info = static_cast<const ComputeNodeInfo *>(node->context.compute_node_info);
  if(compute_node_info == nullptr) {
    return ge::SUCCESS;
  }

  auto dump_properties = ge::DumpManager::GetInstance().GetDumpProperties(session_id_);
  GE_ASSERT_NOTNULL(extend_info_);

  bool is_update_dump_op_range = true;
  if (IsLaunchFFTSPlusTaskNode(node_type)) {
    is_update_dump_op_range = false;
    GELOGW("op[%s] node type[%s] no support dump with opname range", compute_node_info->GetNodeName(), node_type);
  }
  GE_ASSERT_SUCCESS(dump_properties.SetDumpFsmState(extend_info_->model_name, extend_info_->model_data.om_name,
    compute_node_info->GetNodeName(), dump_fsm_state_, dump_op_in_range_, is_update_dump_op_range));

  return ge::SUCCESS;
}

ge::Status ExecutorDumper::DataDump(const Node *node, ExecutorEvent event) {
  if (event == ExecutorEvent::kModelStart) {
    SaveSessionId();
    UpdateStepNum();
    GE_ASSERT_SUCCESS(ResetDumpFsmState());
    return Init();
  } else if (event == ExecutorEvent::kModelEnd) {
    CountIterNum();
    return ge::SUCCESS;
  }

  const auto kernel_extend_info = static_cast<const KernelExtendInfo *>(node->context.kernel_extend_info);
  GE_ASSERT_NOTNULL(kernel_extend_info);
  const auto kernel_type = kernel_extend_info->GetKernelType();
  if ((event == ExecutorEvent::kExecuteStart) && IsLaunchNode(kernel_type)) {
    GE_ASSERT_SUCCESS(SetDumpFsmState(node, kernel_type));
  }

  if (IsHcomLaunchNode(kernel_type)) {
    return InsertHcclDumpOp(node->context, event);
  }

  if (event == ExecutorEvent::kExecuteEnd) {
    GE_ASSERT_SUCCESS(SetDumpFlagForFfts(kernel_type, node));
    GE_ASSERT_SUCCESS(FillDumpInfoByKernel(*node));
    GE_ASSERT_SUCCESS(DoStreamSyncAfterFftsTask(node));
  }
  GE_ASSERT_SUCCESS(OnUpdateDumpUnit(event, *node));
  if (event == ExecutorEvent::kExecuteStart) {
    GE_ASSERT_SUCCESS(UpdateFftsplusLaunchTask(node));
  }
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::FillExceptionDumpInfoByKernel(const Node &node) {
  const ComputeNodeInfo *compute_node_info = reinterpret_cast<const ComputeNodeInfo *>(node.context.compute_node_info);
  if (compute_node_info == nullptr) {
    return ge::SUCCESS;
  }

  const auto node_name = compute_node_info->GetNodeName();
  const auto iter = node_names_to_dump_units_.find(node_name);
  if (iter == node_names_to_dump_units_.end()) {
    return ge::SUCCESS;
  }

  const KernelExtendInfo *ext_info = reinterpret_cast<const KernelExtendInfo *>(node.context.kernel_extend_info);
  GE_ASSERT_NOTNULL(ext_info);
  GE_ASSERT_SUCCESS(FillDumpInfoByKernel(node));
  GE_ASSERT_SUCCESS(FillExtraDumpInfo(node));
  return PrepareExceptionDump(node, ext_info->GetKernelType(), iter->second);
}

ge::Status ExecutorDumper::PrepareExceptionDump(const Node &node, const char *kernel_type, NodeDumpUnit &dump_unit) {
  if (!IsLaunchWithHandleNode(kernel_type) && !IsLaunchWithFlagNode(kernel_type) &&
      !IsAiCpuLaunchNode(kernel_type) && !IsUpdateContext(kernel_type)) {
    return ge::SUCCESS;
  }
  auto compute_node = dump_unit.node;
  if (compute_node == nullptr) {
    return ge::SUCCESS;
  }
  const auto &op_desc = compute_node->GetOpDesc();
  if (op_desc == nullptr) {
    return ge::SUCCESS;
  }
  auto name = compute_node->GetName();
  GELOGD("[Dumper] Begin exception dump preparation for node %s", name.c_str());
  if ((dump_unit.input_addrs.size() != op_desc->GetAllInputsSize()) ||
      (dump_unit.output_addrs.size() != op_desc->GetAllOutputsDescSize())) {
    GELOGW("Input addr or output addr size is invalid.");
    return ge::SUCCESS;
  }
  ge::OpDescPtr op_desc_dump = nullptr;
  GE_MAKE_SHARED(op_desc_dump = std::make_shared<ge::OpDesc>(*op_desc), return ge::FAILED);
  dump_unit.UpdateInputShapes(op_desc_dump);
  dump_unit.UpdateOutputShapes(op_desc_dump);

  auto dumper = global_dumper_->MutableExceptionDumper();
  GE_ASSERT_NOTNULL(dumper);

  auto& extra_dump_unit = node_names_to_extra_units_[name];
  ge::ExtraOpInfo extra_op_info;
  extra_op_info.tiling_data = std::move(extra_dump_unit.tiling_data);
  extra_op_info.args_before_execute = std::move(extra_dump_unit.args_before_execute);
  extra_op_info.args = extra_dump_unit.args;
  extra_op_info.args_size = extra_dump_unit.args_size;
  extra_op_info.tiling_key = extra_dump_unit.tiling_key;
  extra_op_info.workspace_info = std::move(extra_dump_unit.workspace_info);
  extra_op_info.is_host_args = extra_dump_unit.is_host_args;

  const auto execution_data = static_cast<const ExecutionData *>(
      extend_info_->executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData());
  const auto stream_idx = CalcArgIndex(execution_data->base_ed.input_num, ExecuteArgIndex::kStream);
  auto rt_streams =
      reinterpret_cast<Chain *>(execution_data->base_ed.input_values[stream_idx])->GetValue<ContinuousVector *>();
  GE_ASSERT_NOTNULL(rt_streams);
  auto stream = *(reinterpret_cast<rtStream_t *>(rt_streams->MutableData()) + 0U);
  GE_ASSERT_SUCCESS(GetKernelStream(&node, stream));
  const auto processor_type = dump_unit.context_list.empty() ? ProcessorType::kNormal : ProcessorType::kFftsPlus;
  GE_ASSERT_SUCCESS(processors[static_cast<uint32_t>(processor_type)](op_desc_dump, dumper, dump_unit,
                    extra_op_info, stream));
  extra_op_info.DebugLogString();

  // reset dump info
  extra_dump_unit.Clear();
  dump_unit.Clear();
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::ExceptionDump(const Node *node, ExecutorEvent event) {
  if (event == ExecutorEvent::kModelStart) {
    GE_ASSERT_SUCCESS(Init());
    SaveSessionId();
    UpdateStepNum();
  }

  if (event == ExecutorEvent::kExecuteEnd && is_inited_) {
    (void)FillExceptionDumpInfoByKernel(*node);
  }

  if (event == ExecutorEvent::kModelEnd) {
    CountIterNum();
  }
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::SaveSessionId() {
  const auto execution_data = static_cast<const ExecutionData *>(
      extend_info_->executor->GetExeGraphExecutor(kInitExeGraph)->GetExecutionData());
  GE_ASSERT_NOTNULL(execution_data);

  auto session =
      reinterpret_cast<Chain *>(execution_data->base_ed.input_values[static_cast<uint64_t>(ConstDataType::kRtSession)])
          ->GetValue<void *>();
  GE_ASSERT_NOTNULL(session);
  session_id_ = reinterpret_cast<RtSession *>(session)->GetSessionId();
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::SaveRtStream() {
  const auto execution_data = static_cast<const ExecutionData *>(
      extend_info_->executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData());
  GE_ASSERT_NOTNULL(execution_data);

  const auto stream_idx = CalcArgIndex(execution_data->base_ed.input_num, ExecuteArgIndex::kStream);
  auto rt_streams =
      reinterpret_cast<Chain *>(execution_data->base_ed.input_values[stream_idx])->GetValue<ContinuousVector *>();
  GE_ASSERT_NOTNULL(rt_streams);
  GE_ASSERT_TRUE(rt_streams->GetSize() > 0);
  streams_ = rt_streams;
  return ge::SUCCESS;
}

void ExecutorDumper::ClearDumpUnits() {
  for (auto &node_name_and_dump_unit : node_names_to_dump_units_) {
    node_name_and_dump_unit.second.Clear();
  }
}

ge::Status ExecutorDumper::SetOverflowDumpFlag(ExecutorEvent event, const Node &node) {
  GE_ASSERT_NOTNULL(node.context.kernel_extend_info);
  const auto kernel_type = reinterpret_cast<const KernelExtendInfo *>(node.context.kernel_extend_info)->GetKernelType();
  // 当前并非所有的执行节点都有kComputeNodeIndex这个属性，因此可能获取不到对应的ComputeNodeInfo,
  // 经过排查，没有kComputeNodeIndex属性的节点在dump中为非关键节点，因此此处直接返回SUCCESS。
  if (node.context.compute_node_info == nullptr) {
    GELOGD("[Overflow][Dumper]Current kernel has no compute node info. Kernel name[%s], kernel type[%s]",
           reinterpret_cast<const KernelExtendInfo *>(node.context.kernel_extend_info)->GetKernelName(), kernel_type);
    return ge::SUCCESS;
  }
  const auto compute_node_name = static_cast<const ComputeNodeInfo *>(node.context.compute_node_info)->GetNodeName();
  auto overflow_dump_unit = &node_names_to_dump_units_[compute_node_name];

  // UpdateContext kernel should enter DoDataDump to get io addrs for dump unit, and assemble ffts_op_list.
  // Thus, here we set its need_overflow_dump flag true.
  if ((event == kExecuteStart) && (IsUpdateContext(kernel_type))) {
    overflow_dump_unit->need_overflow_dump = true;
    return ge::SUCCESS;
  }
  if ((event == kExecuteEnd) && (IsNeedCheckOverflowNode(kernel_type))) {
    bool is_overflow = false;
    rtStream_t cur_stream = *(reinterpret_cast<rtStream_t *>(streams_->MutableData()));
    GE_ASSERT_SUCCESS(GetKernelStream(&node, cur_stream));
    GE_ASSERT_SUCCESS(CheckOverflow(node, cur_stream, is_overflow));
    if (is_overflow) {
      overflow_dump_unit->need_overflow_dump = true;
    }
    GELOGI("[Overflow][Dumper]Set need_overflow_dump is %s. Kernel name[%s], kernel type[%s]",
            overflow_dump_unit->need_overflow_dump ? "true" : "false",
            reinterpret_cast<const KernelExtendInfo *>(node.context.kernel_extend_info)->GetKernelName(), kernel_type);
  }
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::OnUpdateDumpUnit(ExecutorEvent event, const Node &node, bool overflow_flag) {
  if ((event != kExecuteStart) && (event != kExecuteEnd)) {
    return ge::SUCCESS;
  }

  if (overflow_flag) {
    GE_ASSERT_SUCCESS(SetOverflowDumpFlag(event, node));
  }
  const auto kernel_extend_info = static_cast<const KernelExtendInfo *>(node.context.kernel_extend_info);
  GE_ASSERT_NOTNULL(kernel_extend_info);
  const auto compute_node_info = static_cast<const ComputeNodeInfo *>(node.context.compute_node_info);
  const auto kernel_type = kernel_extend_info->GetKernelType();
  std::vector<NodeDumpUnit *> dump_nodes{};
  if (event == kExecuteEnd) {
    GetLastKernelDumpUnits(node, dump_nodes);
  } else if (IsLaunchFFTSPlusTaskNode(kernel_type)) {
    GE_ASSERT_NOTNULL(compute_node_info);
    dump_nodes.emplace_back(&node_names_to_dump_units_[compute_node_info->GetNodeName()]);
  }
  const auto dump_properties = ge::DumpManager::GetInstance().GetDumpProperties(session_id_);
  if (dump_properties.IsDumpWatcherModelEnable()) {
    GELOGW("[Dumper] Dynamic shape op %s does not support watcher mode.", kernel_extend_info->GetKernelName());
    return ge::SUCCESS;
  }
  for (auto &dump_unit : dump_nodes) {
    if (dump_unit == nullptr || dump_unit->node == nullptr) {
      continue;
    }
    if (!overflow_flag || dump_unit->need_overflow_dump) {
      GELOGI("[Dumper] %s dump for op:%s on kernel %s", overflow_flag ? "overflow" : "data",
             dump_unit->node->GetNamePtr(), kernel_extend_info->GetKernelName());
      GE_ASSERT_SUCCESS(DoDataDump(*dump_unit, dump_properties, &node));
    }
  }
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::DumpOpDebug() {
  const auto &dump_properties = ge::DumpManager::GetInstance().GetDumpProperties(session_id_);
  if (!dump_properties.IsOpDebugOpen()) {
    return ge::SUCCESS;
  }

  GELOGD("OpDebug is open in RT2.0");
  LoadDumpTaskForDavinciModels(true);
  const uint32_t op_debug_mode = dump_properties.GetOpDebugMode();
  for (size_t i = 0U; i < streams_->GetSize(); ++i) {
    auto rt_stream = *(reinterpret_cast<const rtStream_t *>(streams_->GetData()) + i);
    auto data_dumper = std::make_shared<ge::DataDumper>(nullptr);
    data_dumpers_.emplace_back(data_dumper);
    auto op_debug_register = std::make_shared<ge::OpdebugRegister>();
    op_debug_registers_.emplace_back(op_debug_register);
    GE_CHK_STATUS_RET(op_debug_register->RegisterDebugForStream(rt_stream, op_debug_mode, *data_dumper));

    data_dumper->SetDumpProperties(dump_properties);
    data_dumper->SetModelName(extend_info_->model_name);
    data_dumper->SetModelId(extend_info_->model_id);
    GELOGD("[Overflow][Dumper]model name[%s], model id[%u].", extend_info_->model_name.c_str(), extend_info_->model_id);
    int32_t device_id = 0;
    GE_CHK_RT_RET(rtGetDevice(&device_id));
    GE_ASSERT_TRUE(device_id >= 0);
    data_dumper->SetDeviceId(static_cast<uint32_t>(device_id));

    if (IsSingleOpScene()) {
      data_dumper->SetSingleOpDebug();
    } else {
      data_dumper->SetLoopAddr(global_step_addr_, 0U, 0U);
    }

    GE_CHK_STATUS_RET(data_dumper->LoadDumpInfo(), "[Invoke][LoadDumpInfo] failed in hybrid engine, model_id = %u.",
                      extend_info_->model_id);
  }
  is_op_debug_reg_ = true;
  GE_ASSERT_TRUE(data_dumpers_.size() == streams_->GetSize());
  GE_ASSERT_TRUE(op_debug_registers_.size() == streams_->GetSize());
  GELOGD("Dump op debug SUCCESS in RT2.0");
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::ClearDumpOpDebug() {
  if (!is_op_debug_reg_) {
    return ge::SUCCESS;
  }

  LoadDumpTaskForDavinciModels(false);
  GE_ASSERT_TRUE(streams_->GetSize() == data_dumpers_.size());
  GE_ASSERT_TRUE(op_debug_registers_.size() == streams_->GetSize());
  for (size_t i = 0U; i < streams_->GetSize(); ++i) {
    auto rt_stream = *(reinterpret_cast<const rtStream_t *>(streams_->GetData()) + i);
    op_debug_registers_[i]->UnregisterDebugForStream(rt_stream);
    // Unload dump info by model_id when there is no static subgraph in model
    GE_ASSERT_SUCCESS(data_dumpers_[i]->UnloadDumpInfoByModel(extend_info_->model_id));
  }
  data_dumpers_.clear();
  op_debug_registers_.clear();
  return ge::SUCCESS;
}

ge::Status ExecutorDumper::OverflowDump(const Node *node, ExecutorEvent event) {
  if (event == ExecutorEvent::kModelStart) {
    GE_ASSERT_SUCCESS(Init());
    GE_ASSERT_SUCCESS(SaveSessionId());
    GE_ASSERT_SUCCESS(UpdateStepNum());
    GE_ASSERT_SUCCESS(SaveRtStream());
    GE_ASSERT_SUCCESS(DumpOpDebug());
  }
  const auto &dump_properties = ge::DumpManager::GetInstance().GetDumpProperties(session_id_);
  if (!dump_properties.IsOpDebugOpen()) {
    REPORT_INNER_ERR_MSG("E19999",
                       "[Overflow][Dumper]Processing overflow dump, while op debug status is not open, please check.");                                                                \
    GELOGE(ge::FAILED, "[Overflow][Dumper]Processing overflow dump, while op debug status is not open, please check.");
    return ge::FAILED;
  }

  GE_ASSERT_SUCCESS(OnUpdateDumpUnit(event, *node, true));
  if (event == ExecutorEvent::kExecuteStart) {
    GE_ASSERT_SUCCESS(UpdateFftsplusLaunchTask(node));
  }
  if (event == ExecutorEvent::kExecuteEnd) {
    GE_ASSERT_SUCCESS(FillDumpInfoByKernel(*node));
  }
  if (event == ExecutorEvent::kModelEnd) {
    ClearDumpUnits();
    CountIterNum();
    GE_ASSERT_SUCCESS(ClearDumpOpDebug());
  }
  return ge::SUCCESS;
}

void ExecutorDumper::OnExecuteEvent(int32_t sub_exe_graph_type, ExecutorDumper *dumper, ExecutorEvent event,
                                    const void *node, KernelStatus result) {
  (void)result;
  if (dumper == nullptr) {
    return;
  }
  if (dumper->IsEnable(DumpType::kExceptionDump)) {
    (void)dumper->ExceptionDump(static_cast<const Node *>(node), event);
    return;
  }
  if (dumper->IsEnable(DumpType::kDataDump)) {
    (void)dumper->DataDump(static_cast<const Node *>(node), event);
    return;
  }
  if (dumper->IsEnable(DumpType::kOverflowDump) && sub_exe_graph_type == kMainExeGraph) {
    (void)dumper->OverflowDump(static_cast<const Node *>(node), event);
    return;
  }
}
}  // namespace gert

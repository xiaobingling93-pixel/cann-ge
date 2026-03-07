/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_node_converter.h"
#include "aicpu_callback.h"
#include "engine/node_converter_utils.h"
#include "graph_builder/bg_infer_shape.h"
#include "graph_builder/bg_memory.h"
#include "graph_builder/bg_identity.h"
#include "engine/aicpu/graph_builder/bg_launch.h"
#include "framework/common/ge_types.h"
#include "framework/common/types.h"
#include "common/hyper_status.h"
#include "graph/debug/ge_attr_define.h"
#include "aicpu_engine_struct.h"
#include "engine/aicpu/graph_builder/bg_aicpu_arg.h"
#include "engine/aicpu/graph_builder/bg_ext_info.h"
#include "graph/utils/node_utils.h"
#include "graph_builder/converter_checker.h"
#include "common/omg_util/omg_util.h"
#include "register/kernel_registry.h"
#include "graph_builder/bg_rt_session.h"
#include "engine/aicpu/kernel/aicpu_resource_manager.h"
#include "graph/utils/graph_utils.h"
#include "runtime/mem.h"
#include "exe_graph/lowering/frame_selector.h"

namespace gert {
namespace {
const std::set<std::string> kResourceOp = {"TensorListPushBack", "TensorListPopBack"};

void SetSingleOpScene(const ge::NodePtr &node) {
  const auto root_graph = ge::GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());
  if (root_graph != nullptr) {
    bool is_single_op = false;
    (void)ge::AttrUtils::GetBool(root_graph, ge::ATTR_SINGLE_OP_SCENE, is_single_op);
    AicpuResourceManager::GetInstance().SetSingleOp(is_single_op);
  }
}

bg::ValueHolderPtr UpdateWorkSpaceSizeAndAddr(const ge::NodePtr &node, const LowerInput &lower_input,
                                              const bg::ValueHolderPtr &ext_info_handler,
                                              bg::ValueHolderPtr &update_workspace_holder) {
  const auto &op_desc = node->GetOpDesc();
  std::vector<bg::ValueHolderPtr> workspace_info;
  int64_t workspace_size = 0;
  std::vector<int64_t> workspace_bytes = op_desc->GetWorkspaceBytes();
  std::vector<uint32_t> aicpu_workspace_type;
  bool has_aicpu_workspace_type_attr =
      ge::AttrUtils::GetListInt(op_desc, ge::ATTR_NAME_AICPU_WORKSPACE_TYPE, aicpu_workspace_type);
  if (has_aicpu_workspace_type_attr) {
    if (aicpu_workspace_type.size() != workspace_bytes.size()) {
      GELOGE(ge::PARAM_INVALID,
             "Op[%s] aicpu_workspace_type size and workspace_bytes size should be equal, but now aicpu_workspace_type "
             "size "
             "is [%zu], workspace_bytes is [%zu].",
             node->GetName().c_str(), aicpu_workspace_type.size(), workspace_bytes.size());
      return nullptr;
    }

    for (size_t temp_index = 0; temp_index < aicpu_workspace_type.size(); temp_index++) {
      if (aicpu_workspace_type[temp_index] == static_cast<uint32_t>(ge::AicpuWorkSpaceType::CUST_LOG)) {
        // workspace type 与 workspace size应该是一一对应关系
        workspace_size = workspace_bytes[temp_index];
        GELOGD("Op[%s] workspace size for CUST_LOG is [%ld].", node->GetName().c_str(), workspace_size);
        break;
      }
    }
  }

  if (workspace_size > 0) {
    auto workspace_size_holder = bg::ValueHolder::CreateConst(&workspace_size, sizeof(workspace_size));
    workspace_info.emplace_back(workspace_size_holder);
    auto workspace_addr_holder =
        bg::AllocMem(kOnDeviceHbm, workspace_size_holder, *(lower_input.global_data), op_desc->GetStreamId());
    workspace_info.emplace_back(workspace_addr_holder);
    workspace_info.emplace_back(ext_info_handler);
    update_workspace_holder = bg::ValueHolder::CreateSingleDataOutput("UpdateExtWorkSpaceInfo", workspace_info);
    bg::ValueHolder::AddDependency(workspace_addr_holder, update_workspace_holder);
    return workspace_addr_holder;
  } else {
    GELOGD("Op[%s] workspace size is zero for CUST_LOG.", node->GetName().c_str());
    return nullptr;
  }
}

NodeOutput UpdateOutputShapeAndAddr(const ge::NodePtr &node, const LowerInput &lower_input,
                                    const bg::AicpuArgs &aicpu_args, bg::ValueHolderPtr &update_holder,
                                    bg::ValueHolderPtr &workspace_addr_holder) {
  auto node_output = GetOutputShapeAndAddr(node, lower_input.input_shapes, lower_input.input_addrs,
                                           *(lower_input.global_data));

  const auto &op_desc = node->GetOpDesc();

  // get workspace size and addr, update them in ext_info
  bg::ValueHolderPtr update_workspace_holder = nullptr;
  workspace_addr_holder =
      UpdateWorkSpaceSizeAndAddr(node, lower_input, aicpu_args.ext_info_handler, update_workspace_holder);

  auto update_ext = bg::UpdateExtInfo(op_desc, {lower_input.input_shapes, node_output.shapes},
                                      aicpu_args.ext_info_handler, lower_input.global_data->GetStream());
  update_holder = bg::UpdateAicpuIoAddr(aicpu_args.args_handler, lower_input.input_addrs, node_output.addrs);

  if (update_ext != nullptr) {
    if (update_workspace_holder != nullptr) {
      bg::ValueHolder::AddDependency(update_workspace_holder, update_ext);
    }
    bg::ValueHolder::AddDependency(update_ext, update_holder);
  }
  return node_output;
}
} // namespace

LowerResult LoweringAiCpuTfNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  auto compile_result = lower_input.global_data->FindCompiledResult(node);
  const domi::TaskDef *task_def = GetTaskDef(node, compile_result, TaskDefType::kAICpu);
  if (task_def == nullptr) {
    return {HyperStatus::ErrorStatus(static_cast<const char*>("Not find AI cpu Tf taskdef.")), {}, {}, {}};
  }
  auto &kernel_ex_def = task_def->kernel_ex();
  auto session_id = bg::GetSessionId(*lower_input.global_data);

  // gen function handle
  auto rts_args = bg::BuildTfArgsBinHandle(node);

  // alloc args
  auto step_id = GetStepId(*lower_input.global_data);
  auto io_num = node->GetInDataNodesAndAnchors().size() + node->GetAllOutDataAnchorsSize();
  auto aicpu_args = bg::BuildTfAicpuArg(node, {kernel_ex_def, io_num, session_id, step_id}, false);

  // get output shape & addr, update ext_info & io_addr
  bg::ValueHolderPtr update_holder = nullptr;
  bg::ValueHolderPtr workspace_addr_holder = nullptr;
  auto node_output = UpdateOutputShapeAndAddr(node, lower_input, aicpu_args, update_holder, workspace_addr_holder);

  // launch
  auto launch_holder = bg::AicpuTfLaunchKernel(aicpu_args.args_handler, lower_input.global_data->GetStream(), rts_args.bin_handle, node);
  bg::ValueHolder::AddDependency(update_holder, launch_holder);

  SetReleaseAfter(lower_input.input_addrs, launch_holder);
  SetReleaseAfter(node_output.addrs, launch_holder);

  std::vector<bg::ValueHolderPtr> ordered_holders;
  ordered_holders.emplace_back(launch_holder);
  AicpuCallback(node, aicpu_args.ext_info_handler, launch_holder, *(lower_input.global_data), node_output);
  ordered_holders.emplace_back(launch_holder);
  if (kResourceOp.count(node->GetType()) > 0U) {
    std::vector<bg::ValueHolderPtr> inputs;
    inputs.emplace_back(lower_input.global_data->GetStream());
    inputs.insert(inputs.cend(), lower_input.input_addrs.cbegin(), lower_input.input_addrs.cend());
    inputs.insert(inputs.cend(), node_output.addrs.cbegin(), node_output.addrs.cend());
    auto resource_op = bg::ValueHolder::CreateSingleDataOutput("TensorListOp", inputs);
    bg::ValueHolder::AddDependency(launch_holder, resource_op);
    ordered_holders.emplace_back(resource_op);
  }
  std::vector<bg::DevMemValueHolderPtr> out_addrs;
  for (const auto &addrs : node_output.addrs) {
    out_addrs.emplace_back(std::dynamic_pointer_cast<bg::DevMemValueHolder>(addrs));
  }

  return {HyperStatus::Success(), ordered_holders, node_output.shapes, out_addrs};
}

LowerResult LoweringAiCpuCCNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  auto compile_result = lower_input.global_data->FindCompiledResult(node);
  const domi::TaskDef *task_def = GetTaskDef(node, compile_result, TaskDefType::kAICpu);
  if (task_def == nullptr) {
    return {HyperStatus::ErrorStatus(static_cast<const char*>("Not find AI cpu CC taskdef.")), {}, {}, {}};
  }
  auto &kernel_def = task_def->kernel();
  const auto &op_desc = node->GetOpDesc();
  const auto &stream = lower_input.global_data->GetStream();
  auto session_id = bg::GetSessionId(*lower_input.global_data);

  // gen function handle
  auto rts_args = bg::BuildCCArgsBinHandle(node);

  // alloc args
  auto io_num = node->GetInDataNodesAndAnchors().size() + node->GetAllOutDataAnchorsSize();
  auto aicpu_args = bg::BuildCCAicpuArg(node, kernel_def, io_num, session_id, false);

  // get output shape & addr, update ext_info & io_addr
  bg::ValueHolderPtr update_holder = nullptr;
  bg::ValueHolderPtr workspace_addr_holder = nullptr;
  auto node_output = UpdateOutputShapeAndAddr(node, lower_input, aicpu_args, update_holder, workspace_addr_holder);

  // launch
  auto block_dim = bg::CalcBlockDim(op_desc, lower_input.input_shapes);
  auto launch_holder = bg::AicpuCCLaunchKernel(aicpu_args.args_handler, stream, block_dim, kernel_def, op_desc,
                                               aicpu_args.ext_info_handler, rts_args.bin_handle, node);

  bg::ValueHolder::AddDependency(update_holder, launch_holder);
  SetReleaseAfter(lower_input.input_addrs, launch_holder);
  SetReleaseAfter(node_output.addrs, launch_holder);
  if (workspace_addr_holder != nullptr) {
    SetReleaseAfter({workspace_addr_holder}, launch_holder);
  }

  auto cc_launch_holder = launch_holder;
  AicpuCallback(node, aicpu_args.ext_info_handler, launch_holder, *(lower_input.global_data), node_output);
  std::vector<bg::DevMemValueHolderPtr> out_addrs;
  for (const auto &addrs : node_output.addrs) {
    out_addrs.emplace_back(std::dynamic_pointer_cast<bg::DevMemValueHolder>(addrs));
  }
  return {HyperStatus::Success(), {cc_launch_holder, launch_holder}, node_output.shapes, out_addrs};
}

LowerResult LoweringHostAiCpuNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  auto compile_result = lower_input.global_data->FindCompiledResult(node);
  const domi::TaskDef *task_def = GetTaskDef(node, compile_result, TaskDefType::kAICpu);
  if (task_def == nullptr) {
    return {HyperStatus::ErrorStatus(static_cast<const char*>("Not find host AI cpu taskdef.")), {}, {}, {}};
  }
  auto &kernel_def = task_def->kernel();
  auto session_id = bg::GetSessionId(*lower_input.global_data);

  // alloc args
  auto io_num = node->GetInDataNodesAndAnchors().size() + node->GetAllOutDataAnchorsSize();
  auto aicpu_args = bg::BuildHostCCAicpuArg(node, kernel_def, io_num, session_id);

  // get output shape and addr
  auto output_shapes = bg::GetMemAllocShape(node, lower_input.input_shapes, *(lower_input.global_data));
  auto output_sizes = bg::CalcTensorSize(node, output_shapes);

  // compute
  std::vector<bg::DevMemValueHolderPtr> output_addrs;
  auto compute_holder = bg::AicpuHostCompute(node, aicpu_args, {lower_input.input_addrs, lower_input.input_shapes,
      output_sizes, output_shapes}, *lower_input.global_data, output_addrs);

  auto after_compute_addrs = IdentityAddr(output_addrs, node->GetOpDescBarePtr()->GetStreamId());
  for (auto addr : after_compute_addrs) {
    bg::ValueHolder::AddDependency(compute_holder, addr);
  }
  SetReleaseAfter(lower_input.input_addrs, compute_holder);
  return {HyperStatus::Success(), {}, output_shapes, after_compute_addrs};
}

LowerResult LoweringAiCpuNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  if ((node == nullptr) || (node->GetOpDescBarePtr() == nullptr)) {
    GELOGE(ge::PARAM_INVALID, "[Check][Op]Can not find op.");
    REPORT_INNER_ERR_MSG("E19999", "Can not find op.");
    return {HyperStatus::ErrorStatus(static_cast<const char*>("Can not find op")), {}, {}, {}};
  }
  auto ret = CheckLowerInput(lower_input);
  if (!ret.IsSuccess()) {
    GELOGE(ge::PARAM_INVALID, "[Check][LowerInput]Op %s type %s lower_input is invalid.", node->GetName().c_str(),
           ge::NodeUtils::GetNodeType(node).c_str());
    REPORT_INNER_ERR_MSG("E19999", "Op %s type %s lower_input is invalid.", node->GetName().c_str(),
                       ge::NodeUtils::GetNodeType(node).c_str());
    return {ret, {}, {}, {}};
  }
  auto compile_result = lower_input.global_data->FindCompiledResult(node);
  if (compile_result == nullptr) {
    GELOGE(ge::PARAM_INVALID, "[Check][CompileResult]Can not find compile result for node %s type %s",
           node->GetName().c_str(), ge::NodeUtils::GetNodeType(node).c_str());
    REPORT_INNER_ERR_MSG("E19999", "[Check][CompileResult]Can not find compile result for node %s type %s",
                       node->GetName().c_str(), ge::NodeUtils::GetNodeType(node).c_str());
    return {HyperStatus::ErrorStatus(static_cast<const char*>("Can not find compile result")), {}, {}, {}};
  }
  if (compile_result->task_defs.empty()) {
    GELOGE(ge::PARAM_INVALID, "[Check][TaskDef]Unexpected task defs count %zu", compile_result->task_defs.size());
    REPORT_INNER_ERR_MSG("E19999", "Unexpected task defs count %zu", compile_result->task_defs.size());
    return {HyperStatus::ErrorStatus(static_cast<const char*>("Unexpected task defs count")), {}, {}, {}};
  }
  int32_t unknown_shape_type_val = 0;
  (void) ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  if ((bg::IsAicpuUnknownShape(node)) &&
      (unknown_shape_type_val == static_cast<int32_t>(ge::DEPEND_COMPUTE))) {
    // when the operator is the fourth type, and corresponding node is unknown, then 2 tasks are required.
    if (compile_result->task_defs.size() != 2U) {
      GELOGE(ge::PARAM_INVALID, "[Check][TaskDef]Op %s type %s is 4th op, unexpected task defs count %zu",
             node->GetName().c_str(), ge::NodeUtils::GetNodeType(node).c_str(), compile_result->task_defs.size());
      REPORT_INNER_ERR_MSG("E19999", "Op %s type %s is 4th op, unexpected task defs count %zu", node->GetName().c_str(),
                         ge::NodeUtils::GetNodeType(node).c_str(), compile_result->task_defs.size());
      return {HyperStatus::ErrorStatus(static_cast<const char*>("Unexpected task defs count")), {}, {}, {}};
    }
  }

  SetSingleOpScene(node);
  if (node->GetOpDescBarePtr()->GetOpKernelLibName() == ge::kEngineNameAiCpuTf) {
    GELOGI("Op %s type %s in tf_aicpu lowering.", node->GetName().c_str(), ge::NodeUtils::GetNodeType(node).c_str());
    return LoweringAiCpuTfNode(node, lower_input);
  } else if (node->GetOpDescBarePtr()->GetOpKernelLibName() == ge::kEngineNameAiCpu) {
    GELOGI("Op %s type %s in cc_aicpu lowering.", node->GetName().c_str(), ge::NodeUtils::GetNodeType(node).c_str());
    return LoweringAiCpuCCNode(node, lower_input);
  } else {
    GELOGI("Op %s type %s in host_cpu lowering.", node->GetName().c_str(), ge::NodeUtils::GetNodeType(node).c_str());
    return LoweringHostAiCpuNode(node, lower_input);
  }
}

REGISTER_NODE_CONVERTER_PLACEMENT(ge::kEngineNameAiCpuTf.c_str(), kOnDeviceHbm, LoweringAiCpuNode);
REGISTER_NODE_CONVERTER_PLACEMENT(ge::kEngineNameAiCpu.c_str(), kOnDeviceHbm, LoweringAiCpuNode);
REGISTER_NODE_CONVERTER_PLACEMENT(ge::kEngineNameHostCpu.c_str(), kOnHost, LoweringAiCpuNode);
}  // namespace gert

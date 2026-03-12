/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>

#include "bg_platform.h"
#include "bg_tiling.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder/bg_infer_shape.h"
#include "engine/aicore/converter/bg_kernel_launch.h"
#include "kernel/common_kernel_impl/op_tiling_kernel.h"
#include "kernel/common_kernel_impl/tiling.h"
#include "aicore/launch_kernel/rt_kernel_launch_args_ex.h"
#include "register/node_converter_registry.h"

namespace gert {
namespace {
constexpr const char *kTilingNodeName = "tiling_node";
}

//       +--------------------------------------------bg:Tiling-------------------------------------------+
//       |                   |                  |                 |                  |                    |
// kOutputTilingKey    kOutputBlockDim kOutputAtomicCleanFlag  kOutputTilingData  kOutputWorkspace  kOutputTilingCond
//      output1           output2                                    output0                               output4
//         |                  |                                         |                                     |
//         |       scalar     |     scalar     (size of tilling data)   |                 scalar              |
//         v          |       v        |                  |             v                    |                V
//       addr1     shape1   addr2    shape2          shape0({size})   addr0                 shape3          addr3
// ---------------------------------------------------------------------------------------------------------------
LowerResult LoweringOpTiling(const ge::NodePtr &node, const LowerInput &lower_input) {
  LowerResult lower_ret;

  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  // find real tilling node by search whole graph according to attribute node_name
  std::string tiling_node_name;
  (void)ge::AttrUtils::GetStr(op_desc, kTilingNodeName, tiling_node_name);
  if (tiling_node_name.empty()) {
    GELOGE(ge::PARAM_INVALID, "Optiling lowering failed result of empty tiling node name.");
    return {HyperStatus::ErrorStatus("Optiling is without attribute tiling_node_name"), {}, {}, {}};
  }
  auto sub_graph = ge::GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());
  GE_ASSERT_NOTNULL(sub_graph);

  const auto tiling_node = ge::GraphUtils::FindNodeFromAllNodes(sub_graph, tiling_node_name);
  if (tiling_node == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Can not find real tilling node by node:%s sub graph:%s.",
           node->GetName().c_str(), sub_graph->GetName().c_str());
    return {HyperStatus::ErrorStatus("Can not find real tilling node by name"), {}, {}, {}};
  }

  if (op_desc->GetOutputsSize() > static_cast<size_t>(kOpTilingOutputSize)) {
    GELOGE(ge::PARAM_INVALID, "OpTiling Outputsize:%zu of node:%s is larger than expected:%u.",
           node->GetOpDesc()->GetOutputsSize(), node->GetName().c_str(), kOpTilingOutputSize);
    return {HyperStatus::ErrorStatus("Optiling node's outputsize is larger than expected"), {}, {}, {}};
  }
  auto compile_result = lower_input.global_data->FindCompiledResult(tiling_node);
  GE_ASSERT_NOTNULL(compile_result);
  if (compile_result->task_defs.empty()) {
    GELOGE(ge::PARAM_INVALID, "Unexpected task defs count %zu", compile_result->task_defs.size());
    return {HyperStatus::ErrorStatus("Unexpected task defs count"), {}, {}, {}};
  }
  auto &task_def = compile_result->task_defs.back();
  auto launch_arg = bg::AllocRtArg(tiling_node, task_def.kernel_with_handle(), bg::kMaxTilingSize);
  GE_ASSERT_NOTNULL(bg::ValueHolder::GetCurrentFrame());
  auto current_node = bg::ValueHolder::GetCurrentFrame()->GetCurrentComputeNode();
  bg::ValueHolder::GetCurrentFrame()->SetCurrentComputeNode(tiling_node);
  auto output_shapes = bg::InferStorageShape(tiling_node, lower_input.input_shapes, *(lower_input.global_data));
  auto assembled_platform_info_holders = bg::AppendCoreTypeToPlatform(node, lower_input.global_data);
  GE_ASSERT_TRUE(assembled_platform_info_holders.size() == static_cast<size_t>(bg::AssemblePlatformInfoIndex::kNums));
  auto platform_info =
      assembled_platform_info_holders[static_cast<size_t>(bg::AssemblePlatformInfoIndex::kPlatformInfo)];
  auto tiling_ret = bg::Tiling(
      tiling_node, lower_input.input_shapes, output_shapes,
      {platform_info, *(lower_input.global_data), launch_arg[static_cast<size_t>(AllocLaunchArgOutputs::kRtArg)]});
  bg::ValueHolder::GetCurrentFrame()->SetCurrentComputeNode(current_node);

  lower_ret.out_shapes = bg::ValueHolder::CreateDataOutput("BuildOpTilingOutputShape", tiling_ret, kOpTilingOutputSize);
  const int64_t logic_stream_id = op_desc->GetStreamId();
  auto stream_id_holder = bg::ValueHolder::CreateConst(&logic_stream_id, sizeof(logic_stream_id));
  GE_ASSERT_TRUE(IsValidHolder(stream_id_holder));
  // 此处保证tiling_ret最后一个元素是stream_id
  tiling_ret.emplace_back(stream_id_holder);
  lower_ret.out_addrs = bg::DevMemValueHolder::CreateDataOutput("BuildOpTilingUnmanagedTensorData", tiling_ret,
                                                                kOpTilingOutputSize, logic_stream_id);

  lower_ret.result = HyperStatus::Success();
  return lower_ret;
}

REGISTER_NODE_CONVERTER("OpTiling", LoweringOpTiling);
}  // namespace gert

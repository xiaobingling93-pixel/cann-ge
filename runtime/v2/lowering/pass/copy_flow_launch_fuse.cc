/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "copy_flow_launch_fuse.h"

#include <queue>
#include <stack>
#include "common/checker.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "common/util/mem_utils.h"
#include "graph/utils/fast_node_utils.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "runtime/model_v2_executor.h"
#include "kernel/common_kernel_impl/memory_copy.h"
#include "kernel/common_kernel_impl/copy_flow_launch.h"
#include "aicore/launch_kernel/ai_core_launch_kernel.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "core/builder/node_types.h"
#include "kernel/common_kernel_impl/tiling.h"
#include "lowering/pass_changed_kernels_info.h"

namespace gert {
namespace bg {
namespace {
const char *kLaunchKernelTypes[] = {"LaunchKernelWithFlag", "LaunchKernelWithHandle", "AtomicLaunchKernelWithFlag", "AtomicLaunchKernelWithHandle",
                                    "LaunchMixKernelWithHandle", "LaunchMixKernelWithFlag"};

bool IsTargetLaunchNode(const ge::FastNode *const node) {
  const auto node_type = node->GetTypePtr();
  for (const auto target_type : kLaunchKernelTypes) {
    if (strcmp(node_type, target_type) == 0) {
      return true;
    }
  }
  return false;
}

// CopyFlowLaunchFuse里对于新生成的Free guarder节点是继承原来guarder全部的控制输入，这会导致很多冗余的控制边
// 新生成的Free guarder只接受来自于自己launch的控制输入以及其他非launch节点的控制输入
ge::graphStatus FilterAndCopyInCtrlEdges(const ge::FastNode *launch_node, const ge::FastNode *origin_guarder_node,
                                         ge::FastNode *new_guarder_node) {
  GE_ASSERT_NOTNULL(origin_guarder_node);
  GE_ASSERT_NOTNULL(new_guarder_node);

  const auto &src_ctrl_in_nodes = origin_guarder_node->GetInControlNodes();
  if (src_ctrl_in_nodes.empty()) {
    return ge::GRAPH_SUCCESS;
  }
  std::unordered_set<ge::FastNode *> exist_in_ctrl_nodes_set;
  const auto &exist_in_ctrl_nodes = new_guarder_node->GetInControlNodes();
  exist_in_ctrl_nodes_set.insert(exist_in_ctrl_nodes.begin(), exist_in_ctrl_nodes.end());

  const auto src_extend_info = origin_guarder_node->GetExtendInfo();
  GE_ASSERT_NOTNULL(src_extend_info, "The extend info of src node:% is null", origin_guarder_node->GetNamePtr());
  const auto graph = src_extend_info->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph, "The graph of src node:% is null", origin_guarder_node->GetNamePtr());
  for (const auto in_node : src_ctrl_in_nodes) {
    GE_ASSERT_NOTNULL(in_node);
    if (IsTargetLaunchNode(in_node) && (in_node != launch_node)) {
      continue;
    }
    if (exist_in_ctrl_nodes_set.count(in_node) == 0U) {
      exist_in_ctrl_nodes_set.insert(in_node);
      GE_ASSERT_NOTNULL(graph->AddEdge(in_node, ge::kControlEdgeIndex, new_guarder_node, ge::kControlEdgeIndex),
                        "Add ctrl edge %s->%s failed.", in_node->GetNamePtr(), new_guarder_node->GetNamePtr());
    }
  }
  return ge::GRAPH_SUCCESS;
}

const std::unordered_map<std::string, int32_t> kLaunchKernelNamesToIoAddrIndexes = {
    {"LaunchKernelWithFlag", static_cast<int32_t>(kernel::WithArgs::kIoAddrs)},
    {"AtomicLaunchKernelWithHandle", static_cast<int32_t>(kernel::WithAtomicHandle::kIoAddrs)},
    {"AtomicLaunchKernelWithFlag", static_cast<int32_t>(kernel::WithAtomic::kIoAddrs)},
    {"LaunchKernelWithHandle", static_cast<int32_t>(kernel::WithHandle::kIoAddrs)},
    {"LaunchMixKernelWithHandle", static_cast<int32_t>(kernel::WithHandle::kIoAddrs)},
    {"LaunchMixKernelWithFlag", static_cast<int32_t>(kernel::WithArgs::kIoAddrs)}};

struct CopyNode {
  ge::FastNode *copy_node;
  ge::FastNode *consumer_launch_node;
  std::vector<int32_t> input_index_of_launch;  // which input of launch node, offset from io_start
  std::unordered_map<size_t, size_t> out_idxs_2_copy_flow_out_indexes;
  std::unordered_map<size_t, ge::FastNode *> out_idxs_2_guarders;
};

void DebugInfoForCopyNode(const std::vector<CopyNode> &copy_nodes, int32_t io_addr_start_index) {
  if (GlobalTracer::GetInstance()->GetEnableFlags() == 0U) {
    return;
  }
  std::stringstream ss;
  for (const auto &copy_node : copy_nodes) {
    ss << "\nsrc node[" << copy_node.copy_node->GetNamePtr() << "], kernel launch node["
       << copy_node.consumer_launch_node->GetNamePtr() << "].";
    for (const auto &index : copy_node.input_index_of_launch) {
      ss << "\nkernel launch input data idx[" << index << "],[" << index + io_addr_start_index << " - "
         << io_addr_start_index << "].";
    }
  }
  GELOGD("Copy nodes info:%s", ss.str().c_str());
}

std::unique_ptr<uint8_t[]> GetContinuousVector2DByVector2D(const std::vector<std::vector<int32_t>> &vector_2d,
                                                           size_t &total_size) {
  total_size = ContinuousVectorVector::GetOverHeadLength(vector_2d.size());
  for (const auto &inner_vec : vector_2d) {
    size_t inner_vec_length = 0U;
    GE_ASSERT_TRUE(!ge::MulOverflow(inner_vec.size(), sizeof(int32_t), inner_vec_length));
    GE_ASSERT_TRUE(!ge::AddOverflow(inner_vec_length, sizeof(ContinuousVector), inner_vec_length));
    GE_ASSERT_TRUE(!ge::AddOverflow(total_size, inner_vec_length, total_size));
  }
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  auto cvv = new (holder.get()) ContinuousVectorVector();
  GE_ASSERT_NOTNULL(cvv);
  cvv->Init(vector_2d.size());

  for (const auto &inner_vec : vector_2d) {
    auto cv = cvv->Add<int32_t>(inner_vec.size());
    GE_ASSERT_NOTNULL(cv);
    if (!inner_vec.empty()) {
      const size_t copy_size = inner_vec.size() * sizeof(int32_t);
      GE_ASSERT_EOK(memcpy_s(cv->MutableData(), cv->GetCapacity() * sizeof(int32_t), inner_vec.data(), copy_size));
    }
  }
  return holder;
}

ge::graphStatus FindCopyNodes(ge::FastNode *const kernel_launch_node, std::vector<CopyNode> &copy_nodes) {
  GELOGD("find launch kernel node name %s, node type %s", kernel_launch_node->GetNamePtr(),
         kernel_launch_node->GetTypePtr());
  const auto iter = kLaunchKernelNamesToIoAddrIndexes.find(kernel_launch_node->GetType());
  if (iter == kLaunchKernelNamesToIoAddrIndexes.cend()) {
    GELOGE(ge::GRAPH_FAILED, "can't find io addr, node type: %s", kernel_launch_node->GetType().c_str());
    return ge::GRAPH_FAILED;
  }

  auto io_addr_start = iter->second;
  std::vector<ge::FastNode *> src_nodes;
  std::set<ge::FastNode *> unique_src_nodes;
  for (const auto src_node : kernel_launch_node->GetInDataNodes()) {
    GE_ASSERT_NOTNULL(src_node);
    if ((src_node->GetType() != kernel::kMakeSureTensorAtDevice) && (src_node->GetType() != kernel::kCopyH2D)) {
      continue;
    }
    if (unique_src_nodes.emplace(src_node).second) {
      src_nodes.emplace_back(src_node);
    }
  }

  for (const auto src_node : src_nodes) {
    bool need_optimize = true;
    std::vector<int32_t> launch_index = {};
    for (const auto &out_data_edges : src_node->GetAllOutDataEdgesRef()) {
      for (const auto out_data_edge : out_data_edges) {
        if (out_data_edge == nullptr) {
          continue;
        }
        const auto it = kLaunchKernelNamesToIoAddrIndexes.find(out_data_edge->dst->GetType());
        if ((it == kLaunchKernelNamesToIoAddrIndexes.cend()) && out_data_edge->dst->GetType() != "FreeMemory") {
          GELOGD("no need to optimize host input, src node name %s, dst node name %s, dst node type %s",
                 src_node->GetNamePtr(), out_data_edge->dst->GetNamePtr(), out_data_edge->dst->GetTypePtr());
          need_optimize = false;
          break;
        }
        if (out_data_edge->dst->GetName() == kernel_launch_node->GetName()) {
          GE_ASSERT_TRUE(out_data_edge->dst_input >= io_addr_start,
                         "[Param][Invalid] src node[%s], dst node[%s:%d], expect greater than or equal to[%d].",
                         src_node->GetNamePtr(), kernel_launch_node->GetNamePtr(), out_data_edge->dst_input,
                         io_addr_start);
          launch_index.emplace_back(out_data_edge->dst_input - io_addr_start);
        }
      }
    }

    if (need_optimize) {
      CopyNode copy_node{src_node, kernel_launch_node, launch_index, {}, {}};
      copy_nodes.emplace_back(std::move(copy_node));
    }
  }
  DebugInfoForCopyNode(copy_nodes, io_addr_start);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddInputDesc(const ge::OpDescPtr &op_desc, const std::vector<CopyNode> &copy_nodes) {
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(ge::GeTensorDesc()));  // CopyFlowLaunchInputs::kInputsNum
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(ge::GeTensorDesc()));  // CopyFlowLaunchInputs::kInputsIndex
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(ge::GeTensorDesc()));  // CopyFlowLaunchInputs::kRtArg from Tiling

  // CopyFlowLaunchInputs::kStream, CopyFlowLaunchInputs::kAllocator from MakeSureTensorAtDevice
  const auto src_op_desc = copy_nodes[0].copy_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(src_op_desc);
  const auto input_addr_start = static_cast<int32_t>(kernel::MakeSureTensorAtDeviceInputs::kAddrAndLengthStart);
  for (int32_t i = 0; i < input_addr_start; ++i) {
    auto in_data_edge = copy_nodes[0].copy_node->GetInDataEdgeByIndex(i);
    GE_ASSERT_NOTNULL(in_data_edge);
    GE_ASSERT_SUCCESS(op_desc->AddInputDesc(src_op_desc->GetInputDesc(in_data_edge->dst_input)));
  }

  // CopyFlowLaunchInputs::kAddrAndLengthStart
  for (const auto &proc_node : copy_nodes) {
    auto copy_node = proc_node.copy_node;
    const auto copy_op_desc = copy_node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(copy_op_desc);
    const uint32_t copy_node_in_data_size = copy_node->GetDataInNum();
    if (copy_node_in_data_size < static_cast<uint32_t>(input_addr_start)) {
      GELOGE(ge::GRAPH_FAILED, "node name %s, input data num is %u, less than %u", copy_node->GetName().c_str(),
             copy_node_in_data_size, static_cast<uint32_t>(kernel::MakeSureTensorAtDeviceInputs::kAddrAndLengthStart));
      return ge::GRAPH_FAILED;
    }
    for (uint32_t i = input_addr_start; i < copy_node_in_data_size; ++i) {
      auto in_data_edge = copy_node->GetInDataEdgeByIndex(static_cast<int32_t>(i));
      GE_ASSERT_NOTNULL(in_data_edge);
      GE_ASSERT_SUCCESS(op_desc->AddInputDesc(copy_op_desc->GetInputDesc(in_data_edge->dst_input)));
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddOutputDesc(const ge::OpDescPtr &op_desc, const std::vector<CopyNode> &copy_nodes)  {
  // CopyFlowLaunchOutputs::kAddress
  for (const auto &proc_node : copy_nodes) {
    auto copy_node = proc_node.copy_node;
    const auto copy_op_desc = copy_node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(copy_op_desc);
    for (size_t out_index = 0U; out_index < copy_node->GetDataOutNum(); ++out_index) {
      GE_ASSERT_SUCCESS(op_desc->AddOutputDesc(copy_op_desc->GetOutputDesc(out_index)));
    }
  }
  return ge::GRAPH_SUCCESS;
}
ge::FastNode *CreateCopyFlowLaunchNode(ge::ExecuteGraph *const graph, const std::vector<CopyNode> &copy_nodes) {
  size_t input_addr_start = static_cast<size_t>(kernel::MakeSureTensorAtDeviceInputs::kAddrAndLengthStart);
  const size_t in_data_size = copy_nodes[0].copy_node->GetDataInNum();
  if (in_data_size < static_cast<uint32_t>(input_addr_start)) {
    GELOGE(ge::GRAPH_FAILED, "copy node name %s, input data size is %u, less than %u",
           copy_nodes[0].copy_node->GetNamePtr(), in_data_size, static_cast<uint32_t>(input_addr_start));
    return nullptr;
  }

  // a copy node may have multiple launch node consumer
  // copy flow launch node should flow launch node, so name it by launch node name
  std::string fused_node_name = "CopyFlowLaunch_To_" + copy_nodes[0].consumer_launch_node->GetName();
  auto dst_op_desc = ge::MakeShared<ge::OpDesc>(fused_node_name, kernel::kCopyFlowLaunch);
  GE_ASSERT_NOTNULL(dst_op_desc);
  GE_ASSERT_SUCCESS(AddInputDesc(dst_op_desc, copy_nodes));
  GE_ASSERT_SUCCESS(AddOutputDesc(dst_op_desc, copy_nodes));
  return graph->AddNode(dst_op_desc);
}

ge::FastNode *CreateConstNode(ge::ExecuteGraph *const graph, const std::string node_name, const void *data, size_t size,
                              bool is_string) {
  auto const_op_desc = ge::MakeShared<ge::OpDesc>(node_name, "Const");
  GE_ASSERT_NOTNULL(const_op_desc);
  GE_ASSERT_SUCCESS(const_op_desc->AddOutputDesc(ge::GeTensorDesc()));
  auto node = graph->AddNode(const_op_desc);
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_SUCCESS(op_desc->SetAttr("is_string", ge::AnyValue::CreateFrom(is_string)));
  GE_ASSERT_TRUE(ge::AttrUtils::SetZeroCopyBytes(op_desc, kConstValue,
                                                 ge::Buffer::CopyFrom(ge::PtrToPtr<void, uint8_t>(data), size)));
  return node;
}

bool IsCopyNodeHasOtherConsumer(const CopyNode &copy_node) {
  // with guarder
  for (const auto out_node : copy_node.copy_node->GetOutDataNodes()) {
    if (!IsFreeNode(out_node->GetTypePtr())) {
      return true;
    }
  }
  return false;
}

ge::graphStatus RemoveCopyNodeAndGuarderIfNeed(CopyNode &copy_node_info) {
  if (IsCopyNodeHasOtherConsumer(copy_node_info)) {
    GELOGD("Copy node %s has other consumer, should keep.", copy_node_info.copy_node->GetNamePtr());
    return ge::GRAPH_SUCCESS;
  }

  auto copy_node = copy_node_info.copy_node;
  auto graph = copy_node_info.copy_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph);
  for (const auto guarder : copy_node_info.copy_node->GetOutDataNodes()) {
    GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::IsolateNode(guarder, {}));
    GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(graph, guarder));
  }
  GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::IsolateNode(copy_node, {}));
  GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(graph, copy_node));
  return ge::GRAPH_SUCCESS;
}

ge::FastNode *CreateGuarder(ge::FastNode *const origin_guarder, const std::string &node_name) {
  auto op_desc = ge::MakeShared<ge::OpDesc>(*(origin_guarder->GetOpDescBarePtr()));
  GE_ASSERT_NOTNULL(op_desc);
  op_desc->SetName(node_name);
  auto owner_graph = origin_guarder->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(owner_graph);
  return owner_graph->AddNode(op_desc);
}

ge::graphStatus CopyGuarderNodes(const ge::FastNode *launch_node, CopyNode &src_copy_node,
                                 ge::FastNode *const copy_flow_launch_node) {
  for (const auto &out_idx_2_guarder : src_copy_node.out_idxs_2_guarders) {
    auto origin_guarder = out_idx_2_guarder.second;
    auto copy_flow_out_idx = src_copy_node.out_idxs_2_copy_flow_out_indexes[out_idx_2_guarder.first];
    std::string new_guarder_name =
        origin_guarder->GetType() + "_" + copy_flow_launch_node->GetName() + std::to_string(copy_flow_out_idx);
    auto new_guarder = CreateGuarder(origin_guarder, new_guarder_name);
    GE_ASSERT_NOTNULL(new_guarder);
    auto copy_flow_launch_graph = copy_flow_launch_node->GetExtendInfo()->GetOwnerGraphBarePtr();  // to confirm
    GE_ASSERT_NOTNULL(copy_flow_launch_graph);
    GE_ASSERT_NOTNULL(copy_flow_launch_graph->AddEdge(copy_flow_launch_node, copy_flow_out_idx, new_guarder, 0));
    GE_ASSERT_SUCCESS(FilterAndCopyInCtrlEdges(launch_node, origin_guarder, new_guarder));
    GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::CopyOutCtrlEdges(origin_guarder, new_guarder));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReplaceCopyFlowLaunchNode(const ge::FastNode *launch_node, CopyNode &src_copy_node,
                                          ge::FastNode *copy_flow_launch_node, int32_t &input_index,
                                          int32_t &output_index) {
  const auto copy_node = src_copy_node.copy_node;
  auto graph = copy_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph);
  GELOGD("copy node name %s, type %s, input index %d, output index %d", copy_node->GetNamePtr(),
         copy_node->GetTypePtr(), input_index, output_index);
  // 1.copy all input edge
  uint32_t copy_node_index = 0;
  for (auto in_data_edge : copy_node->GetAllInDataEdgesRef()) {
    // LaunchKernel 多个in data 来自同一个MakeSureTensorAtDevice 的同一个out
    if (in_data_edge == nullptr) {
      continue;
    }
    GE_ASSERT_TRUE(static_cast<size_t>(input_index) < copy_flow_launch_node->GetDataInNum(),
                   "input index %d is invalid, valid range [0, %u).", input_index,
                   copy_flow_launch_node->GetDataInNum());
    if ((input_index < static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kAddrAndLengthStart)) ||
        (copy_node_index >= static_cast<int32_t>(kernel::MakeSureTensorAtDeviceInputs::kAddrAndLengthStart))) {
      GE_ASSERT_NOTNULL(
          graph->AddEdge(in_data_edge->src, in_data_edge->src_output, copy_flow_launch_node, input_index++));
    }
    ++copy_node_index;
  }
  GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::CopyInCtrlEdges(copy_node, copy_flow_launch_node));

  // 2. move output data edge to new node, record PassChangedInfo to copy node
  // MakeSureTensorAtDevice可能对应多个CopyFlowLaunch，存在一对多的映射关系，通过加入launch name以区分
  auto pass_changed_info =
      src_copy_node.copy_node->GetOpDescBarePtr()->TryGetExtAttr(kPassChangedInfo, PassChangedKernels{});
  for (const auto &out_data_edges : copy_node->GetAllOutDataEdgesRef()) {
    for (const auto out_data_edge : out_data_edges) {
      if (out_data_edge == nullptr) {
        continue;
      }
      if (out_data_edge->dst->GetType() == "FreeMemory") {
        src_copy_node.out_idxs_2_guarders[out_data_edge->src_output] = out_data_edge->dst;
      }
      if (out_data_edge->dst->GetName() != launch_node->GetName()) {
        continue;
      }
      auto dst_endpoint = ge::FastNodeUtils::GetDstEndpoint(out_data_edge);
      const auto src_index = out_data_edge->src_output;
      GE_ASSERT_GRAPH_SUCCESS(graph->RemoveEdge(out_data_edge));
      GE_ASSERT_TRUE(output_index < static_cast<int32_t>(copy_flow_launch_node->GetDataOutNum()),
                     "output index %d is invalid, valid range [0, %u).", output_index,
                     copy_flow_launch_node->GetDataOutNum());

      GE_ASSERT_NOTNULL(graph->AddEdge(copy_flow_launch_node, output_index, dst_endpoint.node, dst_endpoint.index));
      src_copy_node.out_idxs_2_copy_flow_out_indexes[src_index] = output_index;
      pass_changed_info.pass_changed_kernels.emplace_back(std::pair<KernelNameAndIdx, KernelNameAndIdx>{
          {src_copy_node.copy_node->GetName(), src_index, launch_node->GetName()},
          {copy_flow_launch_node->GetName(), output_index}});
      GELOGD("src copy node %s, copy flow node %s, launch node %s", src_copy_node.copy_node->GetNamePtr(),
             copy_flow_launch_node->GetNamePtr(), launch_node->GetNamePtr());
    }
    ++output_index;
  }
  (void)src_copy_node.copy_node->GetOpDescBarePtr()->SetExtAttr(kPassChangedInfo, pass_changed_info);
  GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::CopyOutCtrlEdges(copy_node, copy_flow_launch_node));

  // 3. handle guarder node
  GE_ASSERT_SUCCESS(CopyGuarderNodes(launch_node, src_copy_node, copy_flow_launch_node));

  // 4 remove copy node and its guarder
  GE_ASSERT_SUCCESS(RemoveCopyNodeAndGuarderIfNeed(src_copy_node));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddEdgeFromTilingNode(ge::FastNode *const copy_flow_launch_node,
                                          const ge::FastNode *const kernel_launch_node) {
  std::set<ge::FastNode *> memcheck_nodes;
  for (const auto src_node : kernel_launch_node->GetInDataNodes()) {
    GELOGD("src node name %s, src node type %s", src_node->GetNamePtr(), src_node->GetTypePtr());
    if (IsTilingNode(src_node->GetTypePtr())) {
      for (const auto dst_node : src_node->GetAllOutNodes()) {
        // 一个tiling只有一个memcheck
        // 当开启oom工具时，TilingAppendDfxInfo会向tilingData中拼一块memcheck内存
        // 需要保证随路拷贝在TilingAppendDfxInfo之后执行
        if (dst_node->GetType() == "TilingAppendDfxInfo") {
          memcheck_nodes.emplace(dst_node);
        }
      }
    }
  }
  const auto graph = copy_flow_launch_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph);
  const auto launch_arg_in_edge =
      kernel_launch_node->GetInDataEdgeByIndex(static_cast<size_t>(kernel::InputCommon::kRtArg));
  GE_ASSERT_NOTNULL(launch_arg_in_edge);
  // Tiling has one data edge to CopyFlowLaunch
  GE_ASSERT_NOTNULL(graph->AddEdge(launch_arg_in_edge->src, launch_arg_in_edge->src_output, copy_flow_launch_node,
                                   static_cast<size_t>(kernel::CopyFlowLaunchInputs::kRtArg)));

  // TilingAppendDfxInfo needs additional control edge
  for (const auto memcheck_node : memcheck_nodes) {
    GELOGD("link from src node name %s, src node type %s", memcheck_node->GetNamePtr(), memcheck_node->GetTypePtr());
    GE_ASSERT_NOTNULL(
        graph->AddEdge(memcheck_node, ge::kControlEdgeIndex, copy_flow_launch_node, ge::kControlEdgeIndex));
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddConstInputNode(ge::ExecuteGraph *const graph, ge::FastNode *const copy_flow_launch_node,
                                  std::vector<std::vector<int32_t>> &host_inputs_addr_index) {
  size_t host_inputs_num = host_inputs_addr_index.size();
  GELOGD("host inputs num %zu", host_inputs_num);
  std::string host_inputs_num_name = "Const_" + copy_flow_launch_node->GetName() + "_Num";
  auto host_inputs_num_node =
      CreateConstNode(graph, host_inputs_num_name, &host_inputs_num, sizeof(host_inputs_num), false);
  GE_ASSERT_NOTNULL(host_inputs_num_node);

  std::string host_inputs_index_name = "Const_" + copy_flow_launch_node->GetName() + "_Index";
  size_t cvv_total_size = 0U;
  auto holder = GetContinuousVector2DByVector2D(host_inputs_addr_index, cvv_total_size);
  GE_ASSERT_NOTNULL(holder);
  auto host_inputs_index_node = CreateConstNode(graph, host_inputs_index_name, holder.get(), cvv_total_size, true);
  GE_ASSERT_NOTNULL(host_inputs_index_node);

  GE_ASSERT_NOTNULL(graph->AddEdge(host_inputs_num_node, 0, copy_flow_launch_node,
                                   static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kInputsNum)));

  GE_ASSERT_NOTNULL(graph->AddEdge(host_inputs_index_node, 0, copy_flow_launch_node,
                                   static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kInputsIndex)));
  return ge::GRAPH_SUCCESS;
}

// all out ctrl edge of copy_node will copy to copy_flow_launch_node
// copy_flow_launch_node follow position of launch
// RedundanceCtrl from copy_node to launch_node will cause circle in graph.
ge::graphStatus RemoveRedundanceCtrlFromCopyToConsumer(ge::ExecuteGraph *const graph,
                                                       const std::vector<CopyNode> &copy_nodes) {
  GE_ASSERT_NOTNULL(graph);
  for (const auto &copy_node : copy_nodes) {
    const auto &copy_out_ctrl_edges = copy_node.copy_node->GetAllOutControlEdgesRef();
    std::unordered_set<ge::FastNode *> consumer_launch_node;
    for (const auto peer_in_launch : copy_node.copy_node->GetOutDataNodes()) {
      if (IsTargetLaunchNode(peer_in_launch)) {
        consumer_launch_node.emplace(peer_in_launch);
      }
    }
    for (auto copy_out_ctrl_edge : copy_out_ctrl_edges) {
      if ((copy_out_ctrl_edge != nullptr) && (copy_out_ctrl_edge->dst != nullptr) &&
          (consumer_launch_node.count(copy_out_ctrl_edge->dst) > 0u)) {
        GE_ASSERT_GRAPH_SUCCESS(graph->RemoveEdge(copy_out_ctrl_edge));
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FuseCopyNodes(ge::FastNode *const launch_node, std::vector<CopyNode> &copy_nodes, bool &changed) {
  const auto graph = launch_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph);
  if (copy_nodes.empty()) {
    GELOGD("no need fuse host inputs node");
    return ge::GRAPH_SUCCESS;
  }

  auto copy_flow_launch_node = CreateCopyFlowLaunchNode(graph, copy_nodes);
  GE_ASSERT_NOTNULL(copy_flow_launch_node);

  int64_t compute_node_index;
  if (ge::AttrUtils::GetInt(copy_nodes[0].consumer_launch_node->GetOpDescBarePtr(), kComputeNodeIndex,
                            compute_node_index)) {
    GE_ASSERT_TRUE(
        ge::AttrUtils::SetInt(copy_flow_launch_node->GetOpDescBarePtr(), kComputeNodeIndex, compute_node_index));
  }

  std::vector<std::vector<int32_t>> host_inputs_addr_index(copy_nodes.size());
  for (size_t idx = 0U; idx < copy_nodes.size(); ++idx) {
    host_inputs_addr_index[idx] = copy_nodes[idx].input_index_of_launch;
  }
  GE_ASSERT_SUCCESS(AddConstInputNode(graph, copy_flow_launch_node, host_inputs_addr_index));

  int32_t input_index = static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kStream);
  int32_t output_index = static_cast<int32_t>(kernel::CopyFlowLaunchOutputs::kAddress);
  for (auto &copy_node : copy_nodes) {
    GE_ASSERT_SUCCESS(
        ReplaceCopyFlowLaunchNode(launch_node, copy_node, copy_flow_launch_node, input_index, output_index));
  }
  GE_ASSERT_SUCCESS(AddEdgeFromTilingNode(copy_flow_launch_node, copy_nodes[0].consumer_launch_node));
  changed = true;
  return ge::GRAPH_SUCCESS;
}
}  // namespace

ge::graphStatus CopyFlowLaunchFuse::Run(ge::ExecuteGraph *const graph, bool &changed) {
  GE_TIMESTAMP_START(CopyFlowLaunchFuse);
  const auto kernel_launch_nodes = graph->GetAllNodes(IsTargetLaunchNode);
  for (const auto node : kernel_launch_nodes) {
    std::vector<CopyNode> copy_nodes = {};
    GE_ASSERT_SUCCESS(FindCopyNodes(node, copy_nodes));
    GE_ASSERT_SUCCESS(RemoveRedundanceCtrlFromCopyToConsumer(node->GetExtendInfo()->GetOwnerGraphBarePtr(), copy_nodes));
    GE_ASSERT_SUCCESS(FuseCopyNodes(node, copy_nodes, changed));
  }
  if (changed) {
    ge::DumpGraph(graph, "AfterCopyFlowLaunch");
  }
  GE_TIMESTAMP_EVENT_END(CopyFlowLaunchFuse, "Pass::CopyFlowLaunchFuse");
  return ge::GRAPH_SUCCESS;
}
}  // namespace bg
}  // namespace gert

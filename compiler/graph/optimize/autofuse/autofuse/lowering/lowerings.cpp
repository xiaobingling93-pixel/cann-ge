/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/checker.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/ir_definitions_recover.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "utils/autofuse_attrs.h"
#include "utils/autofuse_utils.h"
#include "utils/auto_fuse_config.h"
#include "asc_lowerer/asc_overrides.h"
#include "asc_lowerer/loop_common.h"
#include "autofuser.h"
#include "lowerings.h"

#include "ascir_ops_utils.h"
#include "backend/backend_spec.h"

namespace ge {
using namespace autofuse;
namespace {
graphStatus FallbackLowering(const NodePtr &node) {
  for (auto &anchor : node->GetAllInDataAnchors()) {
    if (anchor == nullptr || anchor->GetPeerOutAnchor() == nullptr) {
      continue;
    }
    loop::GetKernelBox(anchor->GetPeerOutAnchor()).Realize();
  }
  for (auto &anchor : node->GetAllOutDataAnchors()) {
    if (anchor == nullptr) {
      continue;
    }
    loop::StoreExtern(anchor);
  }
  return GRAPH_SUCCESS;
}

graphStatus RealizeInputsAndLowering(const NodePtr &node) {
  for (auto &anchor : node->GetAllInDataAnchors()) {
    if (anchor != nullptr && anchor->GetPeerOutAnchor() != nullptr) {
      loop::GetKernelBox(anchor->GetPeerOutAnchor()).Realize();
    }
  }
  return LoweringManager::Lowering(node);
}

bool IsNodeHasControlEdges(const NodePtr &node) {
  if (node->GetOutControlNodes().empty() && node->GetInControlNodes().empty()) {
    return false;
  }
  return true;
}

std::vector<loop::KernelBox> GetNodeKernelBoxes(const NodePtr &node) {
  std::vector<loop::KernelBox> kernel_boxes;
  for (auto &anchor : node->GetAllOutDataAnchors()) {
    kernel_boxes.push_back(loop::GetKernelBox(anchor));
  }
  return kernel_boxes;
}

bool KernelBoxHasSliceAndReduce(const NodePtr &node) {
  bool kernelbox_has_slice_ops = false;
  std::vector<loop::KernelBox> kernel_boxes = GetNodeKernelBoxes(node);
  for (auto &kernel_box : kernel_boxes) { 
    if (kernel_box.NumSlices() != 0U) {
      kernelbox_has_slice_ops = true;
      GELOGI("kernelbox has slice node: %s", kernel_box.DebugString().c_str());      
      break;
    }
  }
  if (kernelbox_has_slice_ops == false) {
    return false;
  }  
  bool out_node_is_reduce = false;
  for (auto out_node : node->GetOutNodes()) {  
    GE_ASSERT_NOTNULL(out_node);
    if (find(reduce_types.begin(), reduce_types.end(), out_node->GetType()) != reduce_types.end()) {
      out_node_is_reduce = true;
      GELOGI("output nodes is reduce type: %s", node->GetName().c_str());
      break;
    }
  }
  return out_node_is_reduce;
}

std::string WhyRealizeByNodeCategory(const ge::NodePtr &node) {
  if (IsNodeHasControlEdges(node)) {
    return "has control edges";
  }
  const static std::set<std::string> kHeavyOps = {"Exp"};
  if (kHeavyOps.count(node->GetType()) > 0U) {
    return "is heavy op";
  }
  if (KernelBoxHasSliceAndReduce(node)) {
    return "slice can not fuse reduce at lowering";
  }
  return "";
}

std::string WhyRealizeByKernelBoxCategory(loop::KernelBox &kernel_box, const LoweringConfig &config, size_t kernelbox_num) {
  if (kernel_box.NumOps() == config.max_loop_ops) {
    return "num loop ops reach limited " + std::to_string(config.max_loop_ops);
  }
  if (kernel_box.NumLoads() >= config.max_loop_loads) {
    return "num loads " + std::to_string(kernel_box.NumLoads()) + " reach limited " +
           std::to_string(config.max_loop_loads);
  }
  // kernelbox_num为node输出个数，根据输出个数判断不同Realize条件
  if (kernelbox_num > 1U && kernel_box.TargetBuffer()->GetPeerInDataNodesSize() > config.max_buffer_readers) {
    return "num readers " + std::to_string(kernel_box.TargetBuffer()->GetPeerInDataNodesSize()) + " exceed limited " +
           std::to_string(config.max_buffer_readers);
  }
  // 重计算阈值，忽略Slice类型融合
  if (kernelbox_num == 1U &&
      kernel_box.TargetBuffer()->GetPeerInDataNodesSize() > AutoFuseConfig::LoweringConfig().recomputation_threshold &&
      !kernel_box.IsSliceOnly()) {
    return "single anchor readers" + std::to_string(kernel_box.TargetBuffer()->GetPeerInDataNodesSize()) +
           " exceed limited recomputation_threshold " +
           std::to_string(AutoFuseConfig::LoweringConfig().recomputation_threshold) + ", anchor size " +
           std::to_string(kernelbox_num);
  }
  if (kernel_box.StreamLabel().empty()) {
    return "";
  }
  for (auto &anchor : kernel_box.TargetBuffer()->GetPeerInDataAnchors()) {
    std::string stream_label;
    if (AttrUtils::GetStr(anchor->GetOwnerNode()->GetOpDesc(), public_attr::USER_STREAM_LABEL, stream_label) &&
        !stream_label.empty() && stream_label != kernel_box.StreamLabel()) {
      return "stream label " + kernel_box.StreamLabel() + " != " + stream_label + " of user node " +
             anchor->GetOwnerNode()->GetName();
    }
  }
  return "";
}

bool IsInputAnchorEmptyTensor(const ge::InDataAnchorPtr& in_anchor) {
  std::vector<Expression> dims;
  GE_WARN_ASSERT(loop::GetBufferShape(in_anchor, dims) == GRAPH_SUCCESS);
  const auto zero = ge::Symbol(0);
  return std::any_of(dims.begin(), dims.end(), [&](const Expression &dim) {
    return dim.Simplify() == zero;
  });
}

bool IsOutputAnchorEmptyTensor(const ge::OutDataAnchor* out_anchor) {
  std::vector<Expression> dims;
  GE_WARN_ASSERT(loop::GetBufferShape(out_anchor, dims) == GRAPH_SUCCESS);
  const auto zero = ge::Symbol(0);
  return std::any_of(dims.begin(), dims.end(), [&](const Expression &dim) {
    return dim.Simplify() == zero;
  });
}

bool IsViewNodeShouldLowering(vector<const ge::Node *> origin_nodes) {
  if (origin_nodes.size() != 1) {
    GELOGI("View node num exceed one, Fall back lowering.");
    return false;
  }
  auto node = origin_nodes.at(0);
  if (node->GetType() != "Reshape") {
    GELOGI("Now only support single reshape node lowering, Fall back lowering.");
    return false;
  }
  if (!node->GetOutControlNodes().empty() || !node->GetInControlNodes().empty() ||
      (node->GetOutDataNodesSize() != 1)) {
    GELOGI("View node has control edge, or node has multi output anchor, Fall back lowering.");
  }
  return true;
}

std::vector<loop::KernelBox> GetRealizedKernelBoxes(const ge::NodePtr &node, const AscBackendFuseConfig &config) {
  (void)config;
  std::vector<loop::KernelBox> realized_kernel_boxes;
  if (node->GetAllOutDataAnchorsSize() == 0) {
    GELOGI("Node %s has no kernel box.", node->GetName().c_str());
    return {};
  }
  for (auto &anchor : node->GetAllOutDataAnchors()) {
    GE_ASSERT_NOTNULL(anchor);
    auto kernel_box = loop::GetKernelBox(anchor);
    if (kernel_box.IsExternKernel()) {
      GELOGI("Kernel box %s is external.", node->GetName().c_str(), kernel_box.Name().c_str());
      return {};
    }
    auto nodes = kernel_box.GetAscendIrNodes();
    vector<const ge::Node *> compute_ops = AutofuseUtils::GetComputeOps(nodes);
    if (kernel_box.IsRealized()) {
      if (!compute_ops.empty()) {
        GELOGI("Lowering for node scope: %s. As has Compute node", kernel_box.DebugString().c_str());
        realized_kernel_boxes.emplace_back(kernel_box);
      } else if (IsViewNodeShouldLowering(nodes)) {
        GELOGI("Lowering for single view node scope: %s", kernel_box.DebugString().c_str());
        realized_kernel_boxes.emplace_back(kernel_box);
      } else {
        GELOGI("Fall back lowering for node scope: %s", kernel_box.DebugString().c_str());
      }
    }
  }
  if (realized_kernel_boxes.empty()) {
    GELOGI("Node %s has no realized kernel box.", node->GetName().c_str());
    return {};
  }
  if (realized_kernel_boxes.size() != node->GetAllOutDataAnchorsSize()) {
    if (IsNodeHasControlEdges(node)) {
      GELOGI("Node %s has control edge but has non-realized kernel box.", node->GetName().c_str());
      return {};
    }
    return realized_kernel_boxes;
  }
  // view op also lowering to ascbc
  return realized_kernel_boxes;
}

string CreateAscbackendName(loop::KernelBox &kernel_box, CounterPtr counter) {
  auto nodes = kernel_box.GetAscendIrNodes();
  string ascbackend_name = "autofuse_" + FuseTypeToString(kernel_box.Type()) + "_";
  GE_ASSERT_NOTNULL(counter);
  ascbackend_name += std::to_string(counter->NextId());
  for (const auto node : nodes) {
    GE_ASSERT_NOTNULL(node);
    ascbackend_name += "_" + node->GetType();
  }
  if (ascbackend_name.size() > AutoFuseConfig::FusionStrategySolverConfig().max_op_name_len) {
    ascbackend_name = ascbackend_name.substr(0, AutoFuseConfig::FusionStrategySolverConfig().max_op_name_len);
  }
  return ascbackend_name;
}

graphStatus BuildOpForKernelBox(loop::KernelBox &kernel_box, CounterPtr counter, shared_ptr<loop::AscOverrides> asc_graph, Operator &asc_op) {
  std::string asc_op_name = CreateAscbackendName(kernel_box, counter);
  GE_WARN_ASSERT(!asc_op_name.empty(), "CreateAscbackendName failed, asc_op_name is empty.");
  GE_ASSERT_NOTNULL(asc_graph->GetOutput());
  if (IsOutputAnchorEmptyTensor(asc_graph->GetOutput())) {
    asc_op = OperatorFactory::CreateOperator(asc_op_name.c_str(), kAscBackendNoKernelOp.c_str());
  } else {
    asc_op = OperatorFactory::CreateOperator(asc_op_name.c_str(), kAscBackend.c_str());
  }
  asc_op.BreakConnect();
  asc_op.DynamicInputRegister("inputs", asc_graph->GetInputs().size());
  asc_op.DynamicOutputRegister("outputs", 1);
  GELOGI("Create fused asc backend op %s for kernel box %s", asc_op_name.c_str(), kernel_box.Name().c_str());
  return GRAPH_SUCCESS;
}

void RealizeUnusedBuffers(loop::KernelBox &kernel_box) {
  auto &optimized_buffers = kernel_box.GetOptimizedInputAscendBuffers();
  for (auto &buffer : optimized_buffers) {
    GELOGI("Realize unused buffer %s after lowering %s", loop::BufferName(buffer).c_str(),
           loop::BufferName(kernel_box.TargetBuffer()).c_str());
    loop::GetKernelBox(
        Anchor::DynamicAnchorCast<OutDataAnchor>(const_cast<OutDataAnchor *>(buffer)->shared_from_this()))
        .Realize();
  }
}

bool IsNodeShouldLowering(const NodePtr &node) {
  std::string super_kernel_scope;
  if (AttrUtils::GetStr(node->GetOpDesc(), "_super_kernel_scope", super_kernel_scope)) {
    GraphFusionReasonStore::CountNodeFuseFailReason(node->GetName(), "it is in super kernel scope" + super_kernel_scope,
                                                    GraphFusionReasonStore::FailReasonCategory::NODE_INFO_ERROR);
    return false;
  }
  bool disable_autofuse_scope;
  if ((AttrUtils::GetBool(node->GetOpDesc(), "_disable_autofuse_scope", disable_autofuse_scope))
      && disable_autofuse_scope) {
    GraphFusionReasonStore::CountNodeFuseFailReason(node->GetName(), "it is in disable autofuse scope",
                                                    GraphFusionReasonStore::FailReasonCategory::NODE_INFO_ERROR);
    return false;
  }

  const auto &skip_node_types = AutoFuseConfig::LoweringConfig().skip_node_types;
  const auto &skip_node_names = AutoFuseConfig::LoweringConfig().skip_node_names;
  
  if (!skip_node_types.empty() && skip_node_types.find(node->GetType()) != skip_node_types.end()) {
    GraphFusionReasonStore::CountNodeFuseFailReason(node->GetName(), node->GetType() + " is in skip list",
                                                    GraphFusionReasonStore::FailReasonCategory::TEMPORARILY_NOT_SUPPORTED);
    return false;
  }
  
  if (!skip_node_names.empty() && skip_node_names.find(node->GetName()) != skip_node_names.end()) {
    GraphFusionReasonStore::CountNodeFuseFailReason(node->GetName(), node->GetName() + " is in skip list",
                                                    GraphFusionReasonStore::FailReasonCategory::TEMPORARILY_NOT_SUPPORTED);
    return false;
  }

  auto in_anchors = node->GetAllInDataAnchors();
  auto out_anchors = node->GetAllOutDataAnchors();
  bool is_indata_empty = std::any_of(in_anchors.begin(), in_anchors.end(), [](const InDataAnchorPtr& in_anchor) -> bool {
    return (in_anchor != nullptr && IsInputAnchorEmptyTensor(in_anchor));
  });
  bool is_outdata_empty = std::any_of(out_anchors.begin(), out_anchors.end(), [](const OutDataAnchorPtr& out_anchor) -> bool {
    return (out_anchor != nullptr && IsOutputAnchorEmptyTensor(out_anchor.get()));
  });
  // 空 -> 非空 不lowering
  if (is_indata_empty && !is_outdata_empty) {
    GraphFusionReasonStore::CountNodeFuseFailReason(node->GetName(), "it is in empty tensor to nonempty tensor",
                                                    GraphFusionReasonStore::FailReasonCategory::NODE_INFO_ERROR);
    return false;
  }
  if ((node->GetType() != NETOUTPUT) && !CheckIrSpec(node->GetOpDesc())) {
    GraphFusionReasonStore::CountNodeFuseFailReason(node->GetName(), "failed to check IR compatibility",
                                                    GraphFusionReasonStore::FailReasonCategory::NODE_INFO_ERROR);
    return false;
  }
  return true;
}

bool IsAnyKernelBoxIsExtern(const std::vector<loop::KernelBox> &kernel_boxes) {
  return std::any_of(kernel_boxes.begin(), kernel_boxes.end(),
                     [](const loop::KernelBox &box) { return box.IsExternKernel(); });
}

bool IsAllKernelBoxIsSupport(const std::vector<loop::KernelBox> &kernel_boxes) {
  return std::all_of(kernel_boxes.begin(), kernel_boxes.end(),
                     [](const loop::KernelBox &box) { return box.IsSupport(); });
}

void RealizeKernelBoxesByCategory(const NodePtr &node, std::vector<loop::KernelBox> &kernel_boxes,
                                  const LoweringConfig &config) {
  auto node_realize_reason = WhyRealizeByNodeCategory(node);
  if (!node_realize_reason.empty()) {
    for (auto &kernel_box : kernel_boxes) {
      GELOGI("Realize persistent kernel box %s because node %s %s.", kernel_box.Name().c_str(), node->GetName().c_str(),
             node_realize_reason.c_str());
      kernel_box.Realize();
    }
    return;
  }
  for (auto &kernel_box : kernel_boxes) {
    if (kernel_box.IsRealizedPersistent()) {
      continue;
    }
    auto realize_reason = WhyRealizeByKernelBoxCategory(kernel_box, config, kernel_boxes.size());
    if (!realize_reason.empty()) {
      GELOGI("Realize persistent kernel box %s because %s.", kernel_box.Name().c_str(), realize_reason.c_str());
      kernel_box.Realize();
    }
  }
}
}  // namespace

graphStatus LoweringManager::Lowering(const NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  GELOGD("Start lowering node %s(%s).", node->GetTypePtr(), node->GetNamePtr());
  return Instance().LowerImpl(node);
}

graphStatus LoweringManager::GetFusedOriginComputeGraph(const AutoFuseAttrs &attrs, const NodePtr &node) {
  GE_ASSERT_NOTNULL(attrs.GetAscGraph());
  std::string name = attrs.GetAscGraph()->GetName() + "_origin";
  GELOGI("Cut origin compute graph %s for asc graph %s", name.c_str(), attrs.GetAscGraph()->GetName().c_str());
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(name);
  GE_ASSERT_NOTNULL(graph);
  std::map<const OutDataAnchor *, OutDataAnchorPtr> origin_to_replaced;
  std::vector<const ge::Node *> origin_ge_nodes = attrs.GetOriginNodes();
  std::sort(origin_ge_nodes.begin(), origin_ge_nodes.end(), [](const ge::Node *a, const ge::Node *b) {
    return a->GetOpDesc()->GetId() < b->GetOpDesc()->GetId();
  });
  for (auto &node : origin_ge_nodes) {
    GE_ASSERT(LoweringUtils::GetOriginToReplaced(node, graph, origin_to_replaced) == GRAPH_SUCCESS);
  }
  for (auto target_buffer : attrs.GetOriginOutputBuffers()) {
    auto iter = origin_to_replaced.find(target_buffer);
    GE_ASSERT(iter != origin_to_replaced.end());
    auto desc = std::make_shared<OpDesc>(loop::BufferName(target_buffer), NETOUTPUT);
    desc->AddInputDesc(GeTensorDesc());
    const auto net_output = graph->AddNode(desc);
    GE_ASSERT_NOTNULL(net_output);
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(iter->second, net_output->GetInDataAnchor(0)));
  }
  std::string dump_graph_name = LoweringUtils::GetConstructDumpGraphName(node);
  AutofuseUtils::DumpGEGraph(graph, kLoweringDir, dump_graph_name + "_original_graph");
  AutofuseUtils::DumpGraphToOnnx(*graph, kLoweringDir, dump_graph_name + "_original_graph");
  return GRAPH_SUCCESS;
}

OpDescPtr LoweringManager::BuildOpDescForKernelBox(
  loop::KernelBox &kernel_box, std::vector<const ge::OutDataAnchor *> &origin_inputs, CounterPtr counter) {
  auto *anchor = const_cast<ge::OutDataAnchor *>(kernel_box.TargetBuffer());
  GE_ASSERT_NOTNULL(anchor);
  std::string graph_name = loop::BufferName(anchor) + "_graph";
  GELOGI("Realize AscendC IR graph %s for kernel box %s, loop graph:\n%s", graph_name.c_str(),
         kernel_box.DebugString().c_str(), kernel_box.Readable().c_str());
  auto asc_graph = kernel_box.Realize<loop::AscOverrides>(graph_name, true /*do cse*/);
  GE_WARN_ASSERT(asc_graph != nullptr,
                 "Fall back lowering for node scope: %s. As Realize AscendC IR graph for kernel box %s failed",
                 kernel_box.DebugString().c_str(), kernel_box.Name().c_str());
  
  GE_WARN_ASSERT_GRAPH_SUCCESS(LoweringUtils::CheckSpecialFuseType(kernel_box, asc_graph),
                               "Fall back lowering, reason is showed above.");
  LoweringUtils::PrintReadableAscGraph(*asc_graph->SharedGraph());
  origin_inputs = asc_graph->GetInputs();
  Operator asc_op;
  GE_WARN_ASSERT_GRAPH_SUCCESS(BuildOpForKernelBox(kernel_box, counter, asc_graph, asc_op));
  auto asc_desc = OpDescUtils::GetOpDescFromOperator(asc_op);
  GE_ASSERT_NOTNULL(asc_desc);
  GE_ASSERT_SUCCESS(AutofuseUtils::AddOperatorPrototypeAttrs(asc_desc));
  GE_WARN_ASSERT_GRAPH_SUCCESS(LoweringUtils::SetStreamLabelForOpDesc(kernel_box, asc_desc),
                               "Fall back lowering, reason is showed above.");

  auto fuse_attrs = asc_desc->GetOrCreateAttrsGroup<AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(fuse_attrs);
  GE_ASSERT_NOTNULL(asc_graph->SharedGraph());
  fuse_attrs->SetAscGraph(asc_graph->SharedGraph(), kernel_box.Type());
  fuse_attrs->SetOriginOutputBuffers({anchor});
  fuse_attrs->SetOriginNodes(kernel_box.GetAscendIrNodes());
  fuse_attrs->SetOptimizedInputBuffers(kernel_box.GetOptimizedInputAscendBuffers());
  GE_ASSERT_SUCCESS(fuse_attrs->SetAndPrintOriginNames(asc_desc, graph_name, origin_inputs, anchor));
  GE_ASSERT_SUCCESS(LoweringUtils::SetAttrCoreNum(asc_desc, kernel_box));
  GetInterAttrs(fuse_attrs).is_fuse_from_lowering = true;
  auto buffer_desc = loop::GetBufferDesc(anchor);
  GE_ASSERT_NOTNULL(buffer_desc);
  GE_ASSERT_GRAPH_SUCCESS(asc_desc->UpdateOutputDesc(0, *buffer_desc));
  for (size_t i = 0U; i < origin_inputs.size(); ++i) {
    buffer_desc = loop::GetBufferDesc(origin_inputs[i]);
    GE_ASSERT_NOTNULL(buffer_desc);
    GE_ASSERT_GRAPH_SUCCESS(asc_desc->UpdateInputDesc(i, *buffer_desc));
  }
  GE_ASSERT_NOTNULL(asc_desc->MutableOutputDesc(0));
  const auto sym_attr = asc_desc->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>();
  GE_ASSERT_NOTNULL(sym_attr);
  GE_ASSERT_GRAPH_SUCCESS(
      loop::GetBufferShape(anchor, sym_attr->symbolic_tensor.MutableOriginSymbolShape().MutableDims()));

  GE_WARN_ASSERT_GRAPH_SUCCESS(LoweringUtils::AssembleConcreteEdges(kernel_box, *fuse_attrs, origin_inputs),
                               "Fall back lowering for node scope: %s. As failed to assemble concrete edges.",
                               kernel_box.DebugString().c_str());
  return asc_desc;
}

graphStatus LoweringManager::FusedSubgraphLoopToAscBackendOp(
    const ComputeGraphPtr &graph, const AscBackendFuseConfig &config,
    std::map<const ge::OutDataAnchor *, ge::OutDataAnchor *> &ascend_out_to_asc_out, CounterPtr counter) {
  for (auto &node : graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    std::vector<loop::KernelBox> kernel_boxes = GetRealizedKernelBoxes(node, config);
    NodePtr expect_position = node;  // Start position for Asc op
    for (auto &kernel_box : kernel_boxes) {
      auto *target_buffer = const_cast<ge::OutDataAnchor *>(kernel_box.TargetBuffer());
      GE_ASSERT_NOTNULL(target_buffer);
      std::vector<const OutDataAnchor *> inputs;
      auto op_desc = BuildOpDescForKernelBox(kernel_box, inputs, counter);
      if (op_desc == nullptr) {  // Maybe trigger by unsupported asc dtype, we never failed lowering
        GELOGW("Fall back lowering for node scope: %s. As failed to build AscendC IR node,"
               "we need to drop kernel box %s for buffer %s", kernel_box.DebugString().c_str(),
               kernel_box.Name().c_str(), loop::BufferName(target_buffer).c_str());
        continue;
      }
      auto asc_node = graph->InsertNode(expect_position, op_desc);
      GE_ASSERT_NOTNULL(asc_node);
      bool disable_lifting = false;
      if (AttrUtils::GetBool(node->GetOpDesc(), "_disable_lifting", disable_lifting) && disable_lifting) {
        (void)AttrUtils::SetBool(asc_node->GetOpDesc(), "_disable_lifting", disable_lifting);
        GELOGI("Success to set disable lifting flag to new ascbackend node: %s", asc_node->GetName().c_str());
      }
      expect_position = asc_node;
      ascend_out_to_asc_out[target_buffer] = asc_node->GetOutDataAnchor(0).get();
      for (auto &input : inputs) {
        auto iter = ascend_out_to_asc_out.find(input);
        if (iter != ascend_out_to_asc_out.end()) {
          input = iter->second;
        }
      }
      std::set<const ge::Node *> used_in_nodes;
      GE_ASSERT_GRAPH_SUCCESS(LoweringUtils::AddDataEdgesForAscNode(asc_node, inputs, target_buffer, used_in_nodes));

      std::set<NodePtr> unused_in_nodes;
      GE_ASSERT_GRAPH_SUCCESS(LoweringUtils::GetUnusedInNodes(kernel_box, used_in_nodes, unused_in_nodes));
      for (auto &ctl_node : unused_in_nodes) {
        GELOGI("Unused input %s(\"%s\") after lowering %s, add control edge to asc node %s", ctl_node->GetTypePtr(),
               ctl_node->GetNamePtr(), kernel_box.Name().c_str(), asc_node->GetName().c_str());
        GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(ctl_node->GetOutControlAnchor(), asc_node->GetInControlAnchor()));
      }
      GE_ASSERT_GRAPH_SUCCESS(LoweringUtils::MoveControlEdges(node, asc_node));
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus LoweringManager::FusedLoopToAscBackendOp(const ComputeGraphPtr &graph, const AscBackendFuseConfig &config, CounterPtr counter) {
  GE_ASSERT_NOTNULL(graph);
  GELOGI("Start fuse lowered graph %s to AscendC IR", graph->GetName().c_str());
  std::map<const ge::OutDataAnchor *, ge::OutDataAnchor *> ascend_out_to_asc_out;
  auto graphs = graph->GetAllSubgraphs();
  if (std::find(graphs.begin(), graphs.end(), graph) == graphs.end()) {
    graphs.insert(graphs.begin(), graph);
  }
  auto default_counter = std::make_unique<DefaultCounter>();
  for (const auto &subgraph : graphs) {
    if (counter != nullptr) {
      GE_ASSERT_SUCCESS(FusedSubgraphLoopToAscBackendOp(subgraph, config, ascend_out_to_asc_out, counter));
    } else {
      GELOGD("Used default counter.");
      GE_ASSERT_SUCCESS(FusedSubgraphLoopToAscBackendOp(subgraph, config, ascend_out_to_asc_out, default_counter.get()));
    }
  }
  return GRAPH_SUCCESS;
}

bool LoweringManager::IsLoweringRegistered(const std::string &op_type) const {
  return lowerings_.find(op_type) != lowerings_.end();
}

void LoweringManager::Register(const std::string &op_type, const std::function<graphStatus(const NodePtr &)> &lower) {
  Instance().RegisterImpl(op_type, lower);
}

LoweringManager &LoweringManager::Instance() {
  static LoweringManager instance;
  return instance;
}

void LoweringManager::RegisterImpl(const std::string &op_type,
                                   const std::function<graphStatus(const NodePtr &)> &lower) {
  lowerings_[op_type] = lower;
}

graphStatus LoweringManager::LowerImpl(const NodePtr &node) {
  auto op_type = node->GetType();
  auto iter = lowerings_.find(op_type);
  if (iter == lowerings_.end()) {
    if (!OpTypeUtils::IsConstNode(node->GetType()) && !OpTypeUtils::IsDataNode(node->GetType())) {
      GraphFusionReasonStore::CountNodeFuseFailReason(node->GetName(), op_type + "No lowering registered",
                                                      GraphFusionReasonStore::FailReasonCategory::TEMPORARILY_NOT_SUPPORTED);
    }
    return FallbackLowering(node);
  }
  return iter->second(node);
}

graphStatus LoweringManager::LoweringGraph(const ComputeGraphPtr &graph, const LoweringConfig &config) {
  GE_ASSERT_NOTNULL(graph);
  GELOGI("Start lowering graph %s", graph->GetName().c_str());
  GraphFusionReasonStore::StartProcessGraph(graph->GetName());
  for (auto &node : graph->GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    GE_ASSERT_NOTNULL(node->GetOpDesc());
    GraphFusionReasonStore::AddCurrentGraphNode(node->GetName(), node->GetType());
    if (!IsNodeShouldLowering(node) || Lowering(node) != GRAPH_SUCCESS) {
      GELOGD("Fallback lowering for node %s, type %s, as: This node should not lowering, "
             "or not register lowering func, or unable to imply lowering",
             node->GetName().c_str(), node->GetType().c_str());
      (void)FallbackLowering(node);
      continue;
    }
    GE_ASSERT_GRAPH_SUCCESS(PostPrecessAfterLoweringNode(node, config));
  }
  return GRAPH_SUCCESS;
}

graphStatus LoweringManager::PostPrecessAfterLoweringNode(const NodePtr &node, const LoweringConfig &config) {
  std::vector<loop::KernelBox> kernel_boxes = GetNodeKernelBoxes(node);
  // Fallback just like realize all output kernel box persistent if any kernel box is invalid
  if (IsAnyKernelBoxIsExtern(kernel_boxes)) {
    GELOGI("Fallback lowering for node %s, type %s as has external kernel box",
           node->GetName().c_str(), node->GetType().c_str());
    FallbackLowering(node);
    return GRAPH_SUCCESS;
  }

  if (!IsAllKernelBoxIsSupport(kernel_boxes)) {
    GraphFusionReasonStore::CountNodeFuseFailReason(node->GetName(), "Fallback lowering, has dtype unsupported kernel box",
                                                    GraphFusionReasonStore::FailReasonCategory::BACKEND_NOT_SUPPORTED);
    FallbackLowering(node);
    return GRAPH_SUCCESS;
  }

  if (LoweringUtils::IsAnyKernelBoxOversize(kernel_boxes, config) || LoweringUtils::IsNodeCoreNumDif(node)) {
    GELOGI("Try re-lowering for node %s, type %s after realize inputs as kernel box is oversize, or this node"
           "different core num scope with after nodes.",
           node->GetName().c_str(), node->GetType().c_str());
    if (RealizeInputsAndLowering(node) != GRAPH_SUCCESS) {
      GELOGI("Fallback lowering for node %s, type %s as lowered failed after realize inputs", node->GetName().c_str(),
             node->GetType().c_str());
      GraphFusionReasonStore::CountNodeFuseFailReason(node->GetName(), "lowered failed after realize inputs, "
                                                      "maybe kernel box is oversize, or this node has "
                                                      "different core num scope with after nodes",
                                                      GraphFusionReasonStore::FailReasonCategory::TEMPORARILY_NOT_SUPPORTED);
      (void)FallbackLowering(node);
      return GRAPH_SUCCESS;
    }
    kernel_boxes = GetNodeKernelBoxes(node);
    if (LoweringUtils::IsAnyKernelBoxOversize(kernel_boxes, config)) {
      GraphFusionReasonStore::CountNodeFuseFailReason(node->GetName(), "Fallbacl lowering, lowered failed after realize inputs, "
                                                      "as kernel box still oversize after origin kernel box is oversize",
                                                      GraphFusionReasonStore::FailReasonCategory::TEMPORARILY_NOT_SUPPORTED);
      (void)FallbackLowering(node);
      return GRAPH_SUCCESS;
    }
  }
  for (auto &kernel_box : kernel_boxes) {
    RealizeUnusedBuffers(kernel_box);  // Realize unused buffers, such as input buffer of zero_like
  }
  RealizeKernelBoxesByCategory(node, kernel_boxes, config);
  return GRAPH_SUCCESS;
}
}  // namespace ge

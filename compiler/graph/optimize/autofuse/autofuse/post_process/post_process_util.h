/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_POST_PROCESS_UTIL_H
#define AUTOFUSE_POST_PROCESS_UTIL_H
#include "common/checker.h"
#include "graph/utils/node_utils.h"
#include "utils/autofuse_utils.h"
#include "utils/autofuse_attrs.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "can_fuse/backend/backend_utils.h"
#include "ascir_ops.h"
#include "utils/auto_fuse_config.h"

namespace ge {
namespace asc_adapt {
struct TensorInfo {
  std::vector<int64_t> axis;
  std::vector<ge::Expression> repeats;
  std::vector<ge::Expression> strides;
  std::vector<int64_t> broadcast_info;
  unsigned long current_topo_id;
  DataType dtype;
  std::vector<int64_t> sched_axis;
};

using AdapterFunc = std::function<Status(AscGraph &, const NodePtr &)>;
const std::set<std::string> kReduceNodeTypes = {"Sum", "Mean", "Max", "Min", "Prod", "Any", "All"};

inline bool IsGeType() {
  const auto &config = autofuse::AutoFuseConfig::Config().GetFusionStrategySolver();
  auto fwk_type = config.fwk_type;
  return ge::AutoFuseFwkType::kGe == fwk_type;
}

inline bool IsTorchType() {
  const auto &config = autofuse::AutoFuseConfig::Config().GetFusionStrategySolver();
  auto fwk_type = config.fwk_type;
  return ge::AutoFuseFwkType::kTorch == fwk_type;
}

inline bool IsTorchDataType(const NodePtr &node) {
  return IsTorchType() && (node->GetType() == kDataType);
}

inline bool IsGatherDataType(const NodePtr &node) {
  return node->GetType() == kGatherType;
}

inline bool IsNextGatherNode(const NodePtr &node) {
  const auto out_anchor = node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(out_anchor);
  for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
    GE_ASSERT_NOTNULL(peer_in_anchor);
    const auto peer_in_node = peer_in_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(peer_in_node);
    if (IsGatherDataType(peer_in_node)) {
      return true;
    }
  }
  return false;
}

inline bool IsSatisfyAscBackendNoKernelFunc(const NodePtr &node, const std::string &proc_name) {
  // 空Tensor场景后端需要后处理补属性及序列化
  return (node->GetType() == kAscBackendNoKernelType) &&
         ((proc_name == "complete_attrs") || (proc_name == "complete_asc_io_index"));
}

inline std::vector<std::string> ReadListStrEnv(const char *env_name, const char sep = ',') {
  std::vector<std::string> result;
  std::unordered_set<std::string> single_result;
  const char *env_value = std::getenv(env_name);

  if (env_value == nullptr || env_value[0] == '\0') {
    return result;
  }

  std::string env_str(env_value);

  // 移除末尾的分号（如果存在）
  if (!env_str.empty() && env_str.back() == ';') {
    env_str.pop_back();
  }

  std::istringstream iss(env_str);
  std::string token;

  while (std::getline(iss, token, sep)) {
    if (!token.empty()) {
    if (single_result.insert(token).second) {
      result.push_back(token);
    }
    }
  }

  return result;
}

// 后处理异常流程dump正在处理的dump图
inline Status CacheGraphBeforePostProcess(const NodePtr &node, const std::string &proc_name, const ComputeGraphPtr &graph) {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    return SUCCESS;
  }

  GE_ASSERT_SUCCESS(BackendUtils::CacheGraph(node->GetName() + "_" + proc_name + "_before", graph));
  return SUCCESS;
}

// 后处理异常流程dump正在处理的dump图
inline Status DumpCacheGraphForExceptionPostProcess(const NodePtr &node, const std::string &proc_name) {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    return SUCCESS;
  }

  GE_ASSERT_SUCCESS(BackendUtils::DumpGraph(node->GetName() + "_" + proc_name + "_before", kPostProcessDir, ""));
  GE_ASSERT_SUCCESS(BackendUtils::DumpGraphAndSubgraphs({node->GetName()}, kPostProcessDir));
  return SUCCESS;
}

// 后处理异常流程dump正在处理的dump图
inline Status DumpFusedCacheGraphForExceptionPostProcess(const std::string &fused_graph_name) {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    return SUCCESS;
  }

  GE_ASSERT_SUCCESS(BackendUtils::DumpGraphAndSubgraphs({fused_graph_name}, kPostProcessDir));
  return SUCCESS;
}

// Original function with the replacement function as a parameter
inline Status ProcessAscBackendNodes(const ComputeGraphPtr &ge_or_fused_asc_backend_graph, AdapterFunc adatper_func,
                                     const std::string &proc_name) {
  for (const auto &node : ge_or_fused_asc_backend_graph->GetAllNodes()) {
    if (!BackendUtils::IsBackendFuseNode(node)) {
      continue;
    }
    if ((node->GetType() == kAscBackendType) || IsSatisfyAscBackendNoKernelFunc(node, proc_name)) {
      GELOGI("node: %s(%s) start to run the process(%s).", node->GetName().c_str(), node->GetType().c_str(),
             proc_name.c_str());
      const auto &op_desc = node->GetOpDesc();
      GE_ASSERT_NOTNULL(op_desc);
      const auto attr = op_desc->GetAttrsGroup<AutoFuseAttrs>();
      GE_ASSERT_NOTNULL(attr);
      if (attr->HasFuseType(loop::FuseType::kConcat)) {
        GELOGI("node %s(Concat) don't post process.", node->GetNamePtr());
        continue;
      }
      GE_ASSERT_NOTNULL(attr->GetAscGraph());
      const auto fused_graph = AscGraphUtils::GetComputeGraph(*(attr->GetAscGraph()));
      GE_ASSERT_SUCCESS(BackendUtils::AddInputOutputNodesForAscGraph(fused_graph));
      GE_ASSERT_SUCCESS(CacheGraphBeforePostProcess(node, proc_name, fused_graph));
      auto ret = adatper_func(*(attr->GetAscGraph()), node);
      if (ret != SUCCESS) {
        GELOGE(FAILED, "AscBackend node: %s(%s), post process(%s) failed, start to dump cache graphs;",
               node->GetName().c_str(), node->GetType().c_str(), proc_name.c_str());
        GE_ASSERT_SUCCESS(DumpCacheGraphForExceptionPostProcess(node, proc_name));
        return ret;
      }
      GELOGI("AscBackendPostProcessor: End to run the process(%s) on the graph, graph: %s, parent node: %s(%s).",
             proc_name.c_str(), fused_graph->GetName().c_str(), node->GetNamePtr(), node->GetType().c_str());
      GELOGD("dump node:%s(%s) asc graph info(with tensor attr info):", node->GetNamePtr(), node->GetType().c_str());
      BackendUtils::DumpAscGraph(node);
    } else if (node->GetType() == kFusedAscBackendType) {
      GELOGI("FusedAscbackend node: %s(%s) start to run the process(%s).", node->GetName().c_str(),
             node->GetType().c_str(), proc_name.c_str());
      GE_ASSERT_NOTNULL(node->GetOpDescBarePtr());
      const auto attr = node->GetOpDescBarePtr()->GetAttrsGroup<AutoFuseAttrs>();
      GE_ASSERT_NOTNULL(attr);
      GE_ASSERT_NOTNULL(attr->GetFuseComputeGraph());
      auto ret = ProcessAscBackendNodes(attr->GetFuseComputeGraph(), adatper_func, proc_name);
      if (ret != SUCCESS) {
        GELOGE(FAILED, "FusedAscBackend node: %s(%s), post process(%s) failed, start to dump cache graphs;",
               node->GetName().c_str(), node->GetType().c_str(), proc_name.c_str());
        GE_ASSERT_SUCCESS(
            DumpFusedCacheGraphForExceptionPostProcess((attr->GetFuseComputeGraph())->GetName()));
        return ret;
      }
    }
  }
  return SUCCESS;
}

inline Status UpdateTopoId(const AscGraph &asc_graph, const int64_t topo_id, int64_t topo_id_increment) {
  auto compute_graph = AscGraphUtils::GetComputeGraph(asc_graph);
  GE_ASSERT_NOTNULL(compute_graph);
  for (const auto &node : compute_graph->GetAllNodes()) {
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    if (op_desc->GetId() > topo_id) {
      op_desc->SetId(op_desc->GetId() + topo_id_increment);
    }
  }
  return SUCCESS;
}

inline Status UpdateTopoId(const AscGraph &asc_graph, const NodePtr &node, int64_t topo_id_increment) {
  const auto &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  auto topo_id = op_desc->GetId();
  GE_ASSERT_SUCCESS(UpdateTopoId(asc_graph, topo_id, topo_id_increment));
  return SUCCESS;
}

inline Status DelNode(const AscGraph &asc_graph, const NodePtr &node) {
  // 把node前后的节点连接起来
  const auto in_data_anchor = node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(in_data_anchor);
  const auto out_data_anchor = node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(out_data_anchor);
  const auto src_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src_anchor);
  GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::RemoveEdge(src_anchor, in_data_anchor));
  for (const auto &dst_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    GE_ASSERT_NOTNULL(dst_anchor);
    GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::RemoveEdge(out_data_anchor, dst_anchor));
    GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::AddEdge(src_anchor, dst_anchor));
  }

  // 删除del node和边
  GELOGD("Remove node: %s(%s) from asc_graph:%s.", node->GetName().c_str(), node->GetType().c_str(),
         asc_graph.GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveJustNode(AscGraphUtils::GetComputeGraph(asc_graph), node),
                          "[Remove][JustNode] failed, graph:%s, node:%s.", asc_graph.GetName().c_str(),
                          node->GetName().c_str());
  NodeUtils::UnlinkAll(*node);
  return SUCCESS;
}

inline Status FromDtypeToOtherDtype(const NodePtr &node, DataType s_dtype, DataType d_dtype) {
  const auto node_opdesc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(node_opdesc);
  const auto node_output_desc_size = node_opdesc->GetAllOutputsDescSize();
  for (auto i = 0U; i < node_output_desc_size; i++) {
    const auto output_tensor_desc = node_opdesc->MutableOutputDesc(i);
    GE_ASSERT_NOTNULL(output_tensor_desc);
    if (output_tensor_desc->GetDataType() == s_dtype) {
      output_tensor_desc->SetDataType(d_dtype);
      GELOGI("node: %s(%s) output_idx(%d) dtype change from %s to DT_FLOAT.", node->GetName().c_str(),
             node->GetType().c_str(), i, TypeUtils::DataTypeToSerialString(s_dtype).c_str(),
             TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str());
    }
  }
  return SUCCESS;
}

// 获取所有前驱节点
inline Status GetPeerOutNodes(const NodePtr &node, std::vector<NodePtr> &peer_out_nodes) {
  auto size = static_cast<int32_t>(node->GetAllInDataAnchorsSize());
  for (auto i = 0; i < size; i++) {
    const auto in_anchor = node->GetInDataAnchor(i);
    GE_ASSERT_NOTNULL(in_anchor);
    const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(peer_out_anchor);
    auto peer_out_node = peer_out_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(peer_out_node);
    peer_out_nodes.push_back(peer_out_node);
  }
  return SUCCESS;
}

// 获取前驱节点
inline Status GetPeerOutNode(const NodePtr &node, NodePtr &peer_out_node, const int32_t idx) {
  const auto in_anchor = node->GetInDataAnchor(idx);
  GE_ASSERT_NOTNULL(in_anchor);
  const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(peer_out_anchor);
  peer_out_node = peer_out_anchor->GetOwnerNode();
  GE_ASSERT_NOTNULL(peer_out_node);
  return SUCCESS;
}

// 获取后驱节点
inline Status GetPeerInNodes(const NodePtr &node, std::vector<NodePtr> &peer_in_nodes, const int32_t out_data_idx) {
  const auto out_anchor = node->GetOutDataAnchor(out_data_idx);
  GE_ASSERT_NOTNULL(out_anchor);
  for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
    GE_ASSERT_NOTNULL(peer_in_anchor);
    const auto peer_in_node = peer_in_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(peer_in_node);
    peer_in_nodes.push_back(peer_in_node);
  }
  return SUCCESS;
}

// 获取当前节点的输出描述符
inline Status GetOutputTensorDesc(const NodePtr &node, GeTensorDescPtr &output_tensor_desc) {
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  output_tensor_desc = op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(output_tensor_desc);
  return SUCCESS;
}

// 判断此节点是不是此Type
inline bool CheckNodeType(const NodePtr &node, const std::string &node_type) {
  if (node->GetType() == node_type) {
    return true;
  }
  return false;
}

// 获取当前节点的输出属性
inline Status GetOutputTensorAttr(const NodePtr &node, AscTensorAttr *&output_tensor_attr) {
  GeTensorDescPtr peer_output_tensor_desc;
  GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorDesc(node, peer_output_tensor_desc));
  output_tensor_attr = peer_output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(output_tensor_attr);
  return SUCCESS;
}

inline void TopologicalSorting(const ComputeGraphPtr graph) {
  graph->TopologicalSorting(
      [](const ge::NodePtr &a, const ge::NodePtr &b) { return a->GetOpDesc()->GetId() < b->GetOpDesc()->GetId(); });
}

// 去重模板函数
template <typename Container>
inline void RemoveDuplicates(Container &container) {
  std::unordered_set<typename Container::value_type> seen;
  auto write = container.begin();
  for (auto read = container.begin(); read != container.end(); ++read) {
    if (seen.find(*read) == seen.end()) {
      *write = *read;
      seen.insert(*read);
      ++write;
    }
  }
  container.erase(write, container.end());
}

// 条件去重函数模板
template <typename Container, typename Compare>
void RemoveDuplicatesConditional(Container &container, Compare compare) {
  std::unordered_set<typename Container::value_type> seen;
  auto write = container.begin();
  for (auto read = container.begin(); read != container.end(); ++read) {
    bool is_unique = true;
    for (const auto &elem : seen) {
      if (compare(*read, elem)) {
        is_unique = false;
        break;
      }
    }
    if (is_unique) {
      *write = *read;
      seen.insert(*read);
      ++write;
    }
  }
  container.erase(write, container.end());
}

// 更新子图的fused_subgraph_outputs属性，保证cse融合后的节点输出和ascgraph的输出节点对应，简化对子图输出的管理
inline Status UpdateSubgraphOutputAttr(const std::unordered_set<NodePtr> &removed_output_nodes, const NodePtr &node) {
  auto autofuse_attr = BackendUtils::GetNodeAutoFuseAttr(node);
  GE_ASSERT_NOTNULL(autofuse_attr);
  auto &outputs = GetInterAttrs(autofuse_attr).fused_subgraph_outputs;

  // 遍历 outputs
  for (auto output_node = outputs.begin(); output_node != outputs.end();) {
    // 检查当前 node 是否在 removed_output_nodes 中
    if (removed_output_nodes.find(*output_node) != removed_output_nodes.end()) {
      output_node = outputs.erase(output_node);  // 删除并更新迭代器
      GELOGI("node %s %s del from fused_subgraph_outputs.", (*output_node)->GetName().c_str(),
             (*output_node)->GetType().c_str());
    } else {
      ++output_node;  // 如果未删除，继续遍历下一个元素
    }
  }
  return SUCCESS;
}

inline bool CheckCastDtype(DataType input_dtype, DataType output_dtype) {
  std::vector<DataType> input_dtypes;
  std::vector<DataType> expect_output_dtypes;
  input_dtypes.push_back(input_dtype);
  expect_output_dtypes.push_back(output_dtype);
  return AutofuseUtils::CallAscirInferDataType<ascir_op::Cast>(input_dtypes, expect_output_dtypes) == SUCCESS;
}

inline bool CheckTransposeDtype(DataType dtype) {
  std::vector<DataType> input_dtypes;
  std::vector<DataType> expect_output_dtypes;
  input_dtypes.push_back(dtype);
  expect_output_dtypes.push_back(dtype);
  return AutofuseUtils::CallAscirInferDataType<ascir_op::Transpose>(input_dtypes, expect_output_dtypes) == SUCCESS;
}

inline Status GetTensorInfoFromAscgraph(TensorInfo &tensor_info, const AscGraph &asc_graph) {
  const auto graph_attr = AscGraphUtils::GetComputeGraph(asc_graph)->GetAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  tensor_info.axis.clear();
  tensor_info.repeats.clear();
  tensor_info.strides.clear();
  for (const auto &axis_info : graph_attr->axis) {
    tensor_info.axis.push_back(axis_info->id);
    tensor_info.repeats.push_back(axis_info->size);
  }
  GE_ASSERT_TRUE(tensor_info.axis.size() == tensor_info.repeats.size());
  ge::Expression temp_stride = kSymbolOne;
  for (size_t i = tensor_info.axis.size(); i > 0U; --i) {
    if (BackendUtils::IsEqOne(tensor_info.repeats[i - 1U])) {
      tensor_info.strides.insert(tensor_info.strides.begin(), kSymbolZero);
    } else {
      tensor_info.strides.insert(tensor_info.strides.begin(), temp_stride);
      temp_stride = tensor_info.repeats[i - 1U] * temp_stride;
    }
  }
  GELOGI("asc_graph %s, get tensor info: axis:%s, repeats:%s, strides:%s.", asc_graph.GetName().c_str(),
         AutofuseUtils::VectorToStr(tensor_info.axis).c_str(),
         AutofuseUtils::VectorToStr(tensor_info.repeats).c_str(),
         AutofuseUtils::VectorToStr(tensor_info.strides).c_str());
  return SUCCESS;
}

inline Status GetGatherAxis(const NodePtr &node, int64_t &axis) {
  const auto &op = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op);
  const auto &attr = op->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(attr);
  auto gather_ir_attr = dynamic_cast<ascir_op::Gather::AscGatherIrAttrDef *>(attr->ir_attr.get());
  GE_ASSERT_NOTNULL(gather_ir_attr);
  if (ge::GRAPH_SUCCESS == (gather_ir_attr->GetAxis(axis))) {
    GELOGI("get gather axis index from node: %s(%s) is %" PRId64 ".", node->GetName().c_str(), node->GetType().c_str(),
           axis);
  } else {
    GELOGW("get gather axis index from node: %s(%s) is null.", node->GetName().c_str(), node->GetType().c_str());
  }
  return SUCCESS;
}

inline Status SetGatherAxis(const NodePtr &node, const int64_t &axis) {
  const auto &op = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op);
  const auto &attr = op->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(attr);
  auto gather_ir_attr = dynamic_cast<ascir_op::Gather::AscGatherIrAttrDef *>(attr->ir_attr.get());
  GE_ASSERT_NOTNULL(gather_ir_attr);
  if (ge::GRAPH_SUCCESS == (gather_ir_attr->SetAxis(axis))) {
    GELOGI("set gather axis index to node: %s(%s) %" PRId64 ".", node->GetName().c_str(), node->GetType().c_str(), axis);
  } else {
    GELOGW("set gather axis index to node: %s(%s) null.", node->GetName().c_str(), node->GetType().c_str());
  }
  return SUCCESS;
}

inline bool IsGatherData(const NodePtr &node) {
  return (node->GetType() == kDataType) && IsNextGatherNode(node);
}

inline bool IsSingleInNode(const NodePtr &node) {
  // 判断单输入
  return node->GetAllInDataAnchorsSize() == 1U;
}

inline bool IsSingleOutNode(const NodePtr &node) {
  // 判断单输出
  auto out_data_anchor_size = node->GetAllOutDataAnchorsSize();
  if (out_data_anchor_size != 1U) {
    return false;
  }
  // 判断输出单引用
  std::vector<NodePtr> peer_in_nodes;
  if (asc_adapt::GetPeerInNodes(node, peer_in_nodes, 0) != SUCCESS) {
    return false;
  }
  return peer_in_nodes.size() == 1U;
}

inline bool IsSingleInAndOutNode(const NodePtr &node) {
  // 判断单输入且单输出且输出单引用
  return IsSingleInNode(node) && IsSingleOutNode(node);
}

inline bool IsNextCubeNode(const NodePtr &node) {
  const auto out_anchor = node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(out_anchor);
  for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
    GE_ASSERT_NOTNULL(peer_in_anchor);
    const auto peer_in_node = peer_in_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(peer_in_node);
    if (AutofuseUtils::IsCubeNodeType(peer_in_node)) {
      return true;
    }
  }
  return false;
}

inline bool IsCubeRelatedAscNode(const NodePtr &node) {
  if (AutofuseUtils::IsCubeNodeType(node)) {
    return true;
  }
  if ((node->GetType() == kLoadType) && IsNextCubeNode(node)) {
    return true;
  }
  if (node->GetType() == kDataType) {
    const auto out_anchor = node->GetOutDataAnchor(0);
    GE_ASSERT_NOTNULL(out_anchor);

    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_ASSERT_NOTNULL(peer_in_anchor);
      const auto peer_in_node = peer_in_anchor->GetOwnerNode();
      GE_ASSERT_NOTNULL(peer_in_node);
      if ((peer_in_node->GetType() == kLoadType) && IsNextCubeNode(peer_in_node)) {
        return true;
      }
    }
  }
  return false;
}

inline bool IsReduceNode(const NodePtr &node) {
  return kReduceNodeTypes.find(node->GetType()) != kReduceNodeTypes.end();
}

inline Status SaveReduceOriginalAxisToFuseAttrPro(AscGraph &asc_graph, [[maybe_unused]] const NodePtr &asc_node) {
  auto autofuse_attr = BackendUtils::GetNodeAutoFuseAttr(asc_node);
  GE_ASSERT_NOTNULL(autofuse_attr);
  if (!autofuse_attr->HasFuseType(loop::FuseType::kReduction)) {
    GELOGI("graph %s fuse type is not reduce, don't need to save axis between broadcast reduce .",
           asc_graph.GetName().c_str());
    return SUCCESS;
  }
  for (const auto &node : asc_graph.GetAllNodes()) {
    if (IsReduceNode(node)) {
      NodePtr reduce_input_node;
      GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNode(node, reduce_input_node, 0));
      TensorAttrInfo reduce_input_node_attr;
      GE_ASSERT_SUCCESS(BackendUtils::GetNodeTensorAttrInfo(reduce_input_node, reduce_input_node_attr));
      autofuse_attr->SetReduceOriginalAxis(reduce_input_node_attr.axis);
      autofuse_attr->SetReduceOriginalRepeats(reduce_input_node_attr.repeats);
      GELOGI("graph %s has broadcast linkto reduce, save axis(axis:%s repeat:%s) between broadcast reduce .",
             asc_graph.GetName().c_str(), AutofuseUtils::VectorToStr(autofuse_attr->GetReduceOriginalAxis()).c_str(),
             AutofuseUtils::VectorToStr(autofuse_attr->GetReduceOriginalRepeats()).c_str());
    }
  }
  return SUCCESS;
}

// 如果broadcast直连reduce，需要保存reduce前的轴信息，用于后处理反推broadcast
inline Status SaveReduceOriginalAxisToFuseAttr(const ComputeGraphPtr &graph) {
  GE_ASSERT_SUCCESS(
      ProcessAscBackendNodes(graph, SaveReduceOriginalAxisToFuseAttrPro, "save_axis_between_broadcast_reduce"));
  return SUCCESS;
}
}  // namespace asc_adapt
}  // namespace ge
#endif  // AUTOFUSE_POST_PROCESS_UTIL_H

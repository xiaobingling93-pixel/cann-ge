/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adaption_improve_precision.h"
#include "common/checker.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "utils/autofuse_utils.h"
#include "utils/autofuse_attrs.h"
#include "post_process/post_process_util.h"
#include "common/platform_context.h"

namespace ge {
namespace {
static const std::unordered_map<std::string, std::string> kTypeToGroup = {
    {kCastType, kCastType}, {kLoadType, kLoadType}, {kGatherType, kGatherType}, {kScalarType, kScalarType}, {kStoreType, kStoreType}};

static const std::unordered_map<std::string, std::string> kBlackList1 = {
    {kDataType, kDataType},           {kLoadType, kLoadType},     {kScalarType, kScalarType},
    {kStoreType, kStoreType},         {kOutputType, kOutputType}, {kBroadcastType, kBroadcastType},
    {kTransposeType, kTransposeType}, {kConcatType, kConcatType}, {kGatherType, kGatherType},
    {kSliceType, kSliceType}};

bool IsUltraLowPrecisionDataType(DataType dtype) {
  return dtype == DT_INT4 || dtype == DT_INT8 || dtype == DT_UINT8;
}

bool IsLowPrecisionDataType(DataType dtype) {
  return dtype == DT_FLOAT16 || dtype == DT_BF16;
}

bool IsHighPrecisionDataType(DataType dtype) {
  return dtype == DT_FLOAT;
}

bool IsFloatDataType(DataType dtype) {
  return (IsLowPrecisionDataType(dtype) || IsHighPrecisionDataType(dtype));
}

bool IsUltraLowToLowPrecision(DataType peer_output_dtype, DataType output_dtype) {
  return IsUltraLowPrecisionDataType(peer_output_dtype) && IsLowPrecisionDataType(output_dtype);
}

[[maybe_unused]]bool IsLowToUltraLowPrecision(DataType peer_output_dtype, DataType output_dtype) {
  return IsLowPrecisionDataType(peer_output_dtype) && IsUltraLowPrecisionDataType(output_dtype);
}

bool IsFloatToUltraLowPrecision(DataType peer_output_dtype, DataType output_dtype) {
  return IsFloatDataType(peer_output_dtype) && IsUltraLowPrecisionDataType(output_dtype);
}

// 判断输出多引用是不是有store
bool IsNodeTypeInPeerInNodes(const std::string &node_type, const std::vector<NodePtr> &peer_in_nodes) {
  for (const auto &peer_in_node : peer_in_nodes) {
    if (peer_in_node->GetType() == node_type) {
      return true;
    }
  }
  return false;
}

// 判断是否需要删除节点
bool ShouldDeleteCastNode(const NodePtr &node, const std::vector<NodePtr> &peer_in_nodes, const NodePtr &peer_out_node,
                          DataType peer_output_dtype, DataType output_dtype) {
  (void)node;
  (void)peer_in_nodes;
  (void)peer_out_node;
  // 如果此cast实现的是fp32和fp16或者bf16之间的转换，则需要删除
  return IsFloatDataType(output_dtype) && IsFloatDataType(peer_output_dtype);
}

// 判断是否需要修改节点数据类型
bool ShouldChangeDataType(const NodePtr &node, const std::vector<NodePtr> &peer_in_nodes, DataType peer_output_dtype,
                          DataType output_dtype) {
  if (IsNodeTypeInPeerInNodes(kStoreType, peer_in_nodes)) {
    GELOGI("Node %s is before Store node, do not change dtype.", node->GetName().c_str());
    return false;
  }
  if (IsUltraLowToLowPrecision(peer_output_dtype, output_dtype) &&
      !asc_adapt::CheckCastDtype(peer_output_dtype, DT_FLOAT)) {
    GELOGI("Node %s is change INT4/INT8/UINT8 to FLOAT16/BF16, do not change dtype.", node->GetName().c_str());
    return false;
  }
  return (output_dtype == DT_FLOAT16 || output_dtype == DT_BF16);
}

Status IsNeedInsertCastBeforeOther(const AscGraph &asc_graph, const NodePtr &other_node, bool &need_insert,
                                   std::vector<int32_t> &input_idxs) {
  // 获取前驱节点
  std::vector<NodePtr> peer_out_nodes;
  GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNodes(other_node, peer_out_nodes));
  GeTensorDescPtr peer_output_tensor_desc;
  for (auto idx = 0U ; idx < peer_out_nodes.size(); idx++) {
    auto peer_out_node = peer_out_nodes[idx];
    GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorDesc(peer_out_node, peer_output_tensor_desc));
    if (peer_out_node->GetType() == kCastType || peer_out_node->GetType() == kLoadType ||
        peer_out_node->GetType() == kGatherType) {
      if (IsLowPrecisionDataType(peer_output_tensor_desc->GetDataType())) {
        need_insert = true;
        input_idxs.push_back(idx);
        GELOGD("node %s(%s) in graph %s, has node %s(%s)(dtype %s) in front of it, need to insert cast",
               other_node->GetNamePtr(), other_node->GetTypePtr(), asc_graph.GetName().c_str(),
               peer_out_node->GetNamePtr(), peer_out_node->GetTypePtr(),
               TypeUtils::DataTypeToSerialString(peer_output_tensor_desc->GetDataType()).c_str());
      };
    }
  }
  return SUCCESS;
}

Status IsNeedInsertCastBeforeStore(const AscGraph &asc_graph, const NodePtr &store_node, bool &need_insert,
                                   bool &is_increase_precision) {
  // 获取前驱节点
  NodePtr peer_out_node;
  GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNode(store_node, peer_out_node, 0));
  GeTensorDescPtr peer_output_tensor_desc;
  GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorDesc(peer_out_node, peer_output_tensor_desc));
  GeTensorDescPtr store_output_tensor_desc;
  GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorDesc(store_node, store_output_tensor_desc));
  is_increase_precision = IsHighPrecisionDataType(store_output_tensor_desc->GetDataType());
  // store和前面节点dtype一致，不做处理
  if (peer_output_tensor_desc->GetDataType() == store_output_tensor_desc->GetDataType()) {
    GELOGI("node %s(%s) in graph %s, has same dtype(%s) node in front of it, does not need to insert cast",
           store_node->GetNamePtr(), store_node->GetTypePtr(),
           TypeUtils::DataTypeToSerialString(store_output_tensor_desc->GetDataType()).c_str(),
           asc_graph.GetName().c_str());
    need_insert = false;
    return SUCCESS;
  }

  need_insert = true;
  GELOGI("node %s(%s) output dtype %s, need insert cast.", store_node->GetName().c_str(), store_node->GetType().c_str(),
         TypeUtils::DataTypeToSerialString(store_output_tensor_desc->GetDataType()).c_str());
  return SUCCESS;
}

NodePtr CreateCastNode(AscGraph &asc_graph, const NodePtr &node) {
  OpDescBuilder cast_op_desc_builder("Cast_" + node->GetName() + "_" + std::to_string(AutofuseUtils::GenUniqueNumber()),
                                     kCastType);
  cast_op_desc_builder.AddInput("x");
  cast_op_desc_builder.AddOutput("y");
  auto cast_op_desc = cast_op_desc_builder.Build();
  GE_ASSERT_NOTNULL(cast_op_desc);
  cast_op_desc->AppendIrInput("x", ge::kIrInputRequired);
  cast_op_desc->AppendIrOutput("y", ge::kIrOutputRequired);
  auto op = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(cast_op_desc));
  GE_ASSERT_NOTNULL(op);
  return asc_graph.AddNode(*op);
}

Status ConstructAndInsertCastNode(AscGraph &asc_graph, const NodePtr &peer_in_node, NodePtr &c_node, int32_t input_idx) {
  c_node = CreateCastNode(asc_graph, peer_in_node);
  GE_ASSERT_NOTNULL(c_node);
  GE_ASSERT_SUCCESS(c_node->SetOwnerComputeGraph(AscGraphUtils::GetComputeGraph(asc_graph)));

  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::ReplaceNodeDataAnchors(c_node, peer_in_node, {input_idx}, {}));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(c_node->GetOutDataAnchor(0), peer_in_node->GetInDataAnchor(input_idx)));
  return SUCCESS;
}

Status UpdateNodeInfo(const AscGraph &asc_graph, const NodePtr &c_node, const NodePtr &peer_in_node) {
  GE_ASSERT_NOTNULL(c_node->GetOpDesc());
  const auto c_node_attr = c_node->GetOpDesc()->GetOrCreateAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(c_node_attr);

  GE_ASSERT_NOTNULL(peer_in_node->GetOpDesc());
  const auto peer_in_node_attr = peer_in_node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(peer_in_node_attr);

  c_node_attr->sched.axis = peer_in_node_attr->sched.axis;
  c_node->GetOpDesc()->SetId(peer_in_node->GetOpDesc()->GetId());
  peer_in_node->GetOpDesc()->SetId(peer_in_node->GetOpDesc()->GetId() + 1);

  GELOGI("cast node %s(%s) sched axis %s topo id %ld in graph %s.", c_node->GetName().c_str(),
         c_node->GetType().c_str(), AutofuseUtils::VectorToStr(c_node_attr->sched.axis).c_str(),
         c_node->GetOpDesc()->GetId(), asc_graph.GetName().c_str());

  return SUCCESS;
}

Status UpdateCastNodeTensorInfo(const NodePtr &c_node, const NodePtr &node, const NodePtr &next_node,
                                bool is_increase_precision) {
  const auto node_opdesc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(node_opdesc);
  const auto output_tensor_desc = node_opdesc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(output_tensor_desc);
  const auto c_opdesc = c_node->GetOpDesc();
  GE_ASSERT_NOTNULL(c_opdesc);

  const auto c_output_tensor_desc = c_opdesc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(c_output_tensor_desc);
  const auto c_o_attr = c_output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(c_o_attr);
  if (is_increase_precision) {
    c_output_tensor_desc->SetDataType(DT_FLOAT);
  } else {
    const auto next_node_opdesc = next_node->GetOpDesc();
    GE_ASSERT_NOTNULL(next_node_opdesc);
    const auto next_output_tensor_desc = next_node_opdesc->MutableOutputDesc(0);
    GE_ASSERT_NOTNULL(next_output_tensor_desc);
    c_output_tensor_desc->SetDataType(IsLowPrecisionDataType(next_output_tensor_desc->GetDataType())
                                      ? next_output_tensor_desc->GetDataType()
                                      : DT_FLOAT16);
  }

  const auto output_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(output_attr);
  c_o_attr->axis = output_attr->axis;
  c_o_attr->repeats = output_attr->repeats;
  c_o_attr->strides = output_attr->strides;

  GELOGI("Add cast node %s(%s), out tensor dtype:%s, axis:%s, repeats:%s, stride:%s.", c_node->GetName().c_str(),
         c_node->GetType().c_str(), TypeUtils::DataTypeToSerialString(c_output_tensor_desc->GetDataType()).c_str(),
         AutofuseUtils::VectorToStr(c_o_attr->axis).c_str(), AutofuseUtils::VectorToStr(c_o_attr->repeats).c_str(),
         AutofuseUtils::VectorToStr(c_o_attr->strides).c_str());
  return SUCCESS;
}

Status InsertCastToDecreasePrecision(AscGraph &asc_graph, const NodePtr &store_node, bool need_insert) {
  if (!need_insert) {
    GELOGI(
        "Node %s %s in graph %s has cast before it, or it is DT_FLOAT,"
        "it doesn't need to insert cast",
        store_node->GetName().c_str(), store_node->GetName().c_str(), asc_graph.GetName().c_str());
    return SUCCESS;
  }

  GE_ASSERT_SUCCESS(asc_adapt::UpdateTopoId(asc_graph, store_node, 1));
  NodePtr c_node = nullptr;
  GE_ASSERT_SUCCESS(ConstructAndInsertCastNode(asc_graph, store_node, c_node, 0));
  // 给cast补tensor info要根据前驱节点来补，如果根据后面的store节点来补，后驱节点是slice场景会出错
  NodePtr peer_out_node;
  GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNode(c_node, peer_out_node, 0));
  GE_ASSERT_SUCCESS(UpdateCastNodeTensorInfo(c_node, peer_out_node, store_node, false));
  GE_ASSERT_SUCCESS(UpdateNodeInfo(asc_graph, c_node, store_node));
  return SUCCESS;
}

Status InsertNodeAfterLoadNode(const NodePtr &load_node, const NodePtr &node) {
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::ReplaceNodeDataAnchors(node, load_node, {}, {0}));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(load_node->GetOutDataAnchor(0), node->GetInDataAnchor(0)));
  return SUCCESS;
}

Status UpdateCastNodeInfo(const NodePtr &c_node, const NodePtr &load_node) {
  const auto c_opdesc = c_node->GetOpDesc();
  GE_ASSERT_NOTNULL(c_opdesc);
  const auto c_node_attr = c_opdesc->GetOrCreateAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(c_node_attr);
  const auto load_node_attr = load_node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(load_node_attr);
  c_node_attr->sched.axis = load_node_attr->sched.axis;
  c_opdesc->SetId(load_node->GetOpDesc()->GetId() + 1);

  GELOGI("Cast node %s(%s) sched axis %s topo id %ld.", c_node->GetName().c_str(), c_node->GetType().c_str(),
         AutofuseUtils::VectorToStr(c_node_attr->sched.axis).c_str(), c_opdesc->GetId());
  return SUCCESS;
}

Status InsertCastBeforeNode(AscGraph &asc_graph, const NodePtr &other_node, bool is_need_insert_cast,
                            bool is_increase_precision, const std::vector<int32_t> &input_idxs) {
  if (!is_need_insert_cast) {
    GELOGD("Node %s(%s) in graph %s, doesn't need to insert cast before it", other_node->GetName().c_str(),
           other_node->GetName().c_str(), asc_graph.GetName().c_str());
    return SUCCESS;
  }
  for (auto input_idx : input_idxs) {
    GE_ASSERT_SUCCESS(asc_adapt::UpdateTopoId(asc_graph, other_node, 1));
    NodePtr c_node = nullptr;
    GE_ASSERT_SUCCESS(ConstructAndInsertCastNode(asc_graph, other_node, c_node, input_idx));
    // 给cast补tensor info要根据前驱节点来补，如果根据后面的节点来补，后驱节点是减少轴的view算子会出错
    NodePtr peer_out_node;
    GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNode(c_node, peer_out_node, 0));
    GE_ASSERT_SUCCESS(UpdateCastNodeTensorInfo(c_node, peer_out_node, other_node, is_increase_precision));
    GE_ASSERT_SUCCESS(UpdateNodeInfo(asc_graph, c_node, other_node));
  }
  return SUCCESS;
}

Status InsertCastToIncreasePrecision(AscGraph &asc_graph, const NodePtr &load_node, bool is_need_insert_cast) {
  if (!is_need_insert_cast) {
    GELOGI(
        "Node %s(%s) in graph %s has cast after it, or it is not DT_FLOAT16 or DT_BF16,"
        "it doesn't need to insert cast",
        load_node->GetName().c_str(), load_node->GetName().c_str(), asc_graph.GetName().c_str());
    return SUCCESS;
  }

  GE_ASSERT_SUCCESS(asc_adapt::UpdateTopoId(asc_graph, load_node, 1));
  auto c_node = CreateCastNode(asc_graph, load_node);
  GE_ASSERT_NOTNULL(c_node);
  GE_ASSERT_SUCCESS(InsertNodeAfterLoadNode(load_node, c_node));
  GE_ASSERT_SUCCESS(UpdateCastNodeTensorInfo(c_node, load_node, nullptr, true));
  GE_ASSERT_SUCCESS(UpdateCastNodeInfo(c_node, load_node));
  return SUCCESS;
}

Status IsNeedInsertCastAfterLoad(const NodePtr &node, bool &is_need_insert_cast) {
  const auto node_opdesc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(node_opdesc);
  const auto output_tensor_desc = node_opdesc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(output_tensor_desc);
  const auto attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(attr);
  // 获取后驱节点，考虑多引用的情况，在load后面输出多引用有cast的场景在各自的引用分支处理，无cast的场景(说明各引用分支dtype都一样)统一处理
  // load直连store场景不插入cast
  std::vector<NodePtr> peer_in_nodes;
  GE_ASSERT_SUCCESS(asc_adapt::GetPeerInNodes(node, peer_in_nodes, 0));
  if (IsNodeTypeInPeerInNodes(kCastType, peer_in_nodes) || IsNodeTypeInPeerInNodes(kStoreType, peer_in_nodes) ||
      !(output_tensor_desc->GetDataType() == DT_FLOAT16 || output_tensor_desc->GetDataType() == DT_BF16)) {
    GELOGI("node %s(%s) output dtype %s, doesn't need change from DT_FLOAT16 or DT_BF16 to DT_FLOAT.",
           node->GetName().c_str(), node->GetType().c_str(),
           TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str());
    return SUCCESS;
  }
  is_need_insert_cast = true;
  GELOGI("node %s(%s) output dtype %s, need insert cast.", node->GetName().c_str(), node->GetType().c_str(),
         TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str());
  return SUCCESS;
}

bool IsFloatToUltraLowNeedInsertCast(const NodePtr &peer_out_node, DataType peer_output_dtype, DataType output_dtype) {
  // 非降精度到INT4、INT8、UINT8跨精度场景返回false
  if (!IsFloatToUltraLowPrecision(peer_output_dtype, output_dtype)) {
    return false;
  }
  // 支持了跨精度cast返回false
  if (asc_adapt::CheckCastDtype(DT_FLOAT, output_dtype)) {
    return false;
  }
  // load或者gather或者cast 是fp16或bf16则不需要插cast
  if (((peer_out_node->GetType() == kLoadType) || (peer_out_node->GetType() == kGatherType) ||
       (peer_out_node->GetType() == kCastType)) &&
      (IsLowPrecisionDataType(peer_output_dtype))) {
    return false;
  }
  return true;
}

Status CastNodeProc(AscGraph &asc_graph, const NodePtr &node) {
  // 获取前驱节点
  NodePtr peer_out_node;
  GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNode(node, peer_out_node, 0));
  // 获取后驱节点，考虑多引用的情况
  std::vector<NodePtr> peer_in_nodes;
  GE_ASSERT_SUCCESS(asc_adapt::GetPeerInNodes(node, peer_in_nodes, 0));

  // 获取前驱节点的输出描述符
  GeTensorDescPtr peer_output_tensor_desc;
  GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorDesc(peer_out_node, peer_output_tensor_desc));

  // 获取当前节点的输出描述符
  GeTensorDescPtr output_tensor_desc;
  GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc));
  const auto peer_output_dtype = peer_output_tensor_desc->GetDataType();
  const auto output_dtype = output_tensor_desc->GetDataType();
  // 判断是否需要删除节点
  if (ShouldDeleteCastNode(node, peer_in_nodes, peer_out_node, peer_output_dtype, output_dtype)) {
    GE_ASSERT_SUCCESS(asc_adapt::DelNode(asc_graph, node));
    return SUCCESS;
  }

  // 1、判断是否 fp16/bf16非load节点(非load节点后续流程会升为p32)或fp32的load 降 int4/int8/uint8，需要在前面插入 fp32 降 fp16/bf16 的cast节点
  // 2、预留判断是否已支持跨精度cast（fp32不支持直接降 int4/int8/uint8），支持了跨精度则不需要再多插一个cast
  // 3、降精度cast前面是load或cast，且load或cast是fp16/bf16，不需要插cast处理
  if (IsFloatToUltraLowNeedInsertCast(peer_out_node, peer_output_dtype, output_dtype)) {
    GE_ASSERT_SUCCESS(InsertCastToDecreasePrecision(asc_graph, node, true));
    GELOGI("Node %s(%s) changed dtype from %s to %s, need insert cast to changed dtype from DT_FLOAT to %s.",
           node->GetName().c_str(), node->GetType().c_str(),
           TypeUtils::DataTypeToSerialString(peer_output_dtype).c_str(),
           TypeUtils::DataTypeToSerialString(output_dtype).c_str(),
           TypeUtils::DataTypeToSerialString(peer_output_dtype).c_str());
    return SUCCESS;
  }

  // 判断是否需要修改节点数据类型
  if (ShouldChangeDataType(node, peer_in_nodes, peer_output_dtype, output_dtype)) {
    output_tensor_desc->SetDataType(DT_FLOAT);
    GELOGI("Node %s(%s) changed dtype from %s to %s.", node->GetName().c_str(), node->GetType().c_str(),
           TypeUtils::DataTypeToSerialString(output_dtype).c_str(),
           TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str());
    return SUCCESS;
  }

  // 保持节点不变
  GELOGI("Node %s(%s) keeps dtype %s (peer out node %s(%s) dtype is %s).", node->GetName().c_str(),
         node->GetType().c_str(), TypeUtils::DataTypeToSerialString(output_dtype).c_str(),
         peer_out_node->GetName().c_str(), peer_out_node->GetType().c_str(),
         TypeUtils::DataTypeToSerialString(peer_output_dtype).c_str());

  return SUCCESS;
}

// 临时接口，使用node type来判断是否非ComputeElewise类型,ascir支持构建node的时候配置node_attr->api.compute_type后改用compute_type
bool IsInBlackList1(const NodePtr &node) {
  auto node_type = node->GetType();
  auto it = kBlackList1.find(node_type);
  return (it != kBlackList1.end());
}

// 外部配置的黑名单
bool IsInBlackList2(const NodePtr &node) {
  auto node_type = node->GetType();
  auto &blacklist2 = autofuse::AutoFuseConfig::Config().GetFusionStrategySolver().improve_precision_blacklist;
  return (blacklist2.find(node_type) != blacklist2.end());
}

// 如果在黑名单，不做升精度但是dtype又不是后端支持的类型，报错提示检查升精度黑名单
Status CheckNodeDtype(const NodePtr &node) {
  std::vector<NodePtr> peer_out_nodes;
  GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNodes(node, peer_out_nodes));
  std::vector<DataType> input_dtypes;
  std::vector<DataType> expect_output_dtypes;
  for (auto &peer_out_node : peer_out_nodes) {
    GeTensorDescPtr peer_output_tensor_desc;
    GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorDesc(peer_out_node, peer_output_tensor_desc));
    const auto peer_output_dtype = peer_output_tensor_desc->GetDataType();
    input_dtypes.push_back(peer_output_dtype);
  }
  GeTensorDescPtr output_tensor_desc;
  GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc));
  const auto output_dtype = output_tensor_desc->GetDataType();
  expect_output_dtypes.push_back(output_dtype);
  if (AutofuseUtils::CallAscirCommonInferDtype(node->GetType(), input_dtypes, expect_output_dtypes) != SUCCESS) {
    GELOGE(FAILED,
           "Node %s(%s) with dtype(%s) is not supported. Do not configure it in autofuse_enhance_precision_blacklist",
           node->GetName().c_str(), node->GetType().c_str(), TypeUtils::DataTypeToSerialString(output_dtype).c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status IsAllNodesInBlacklist(const AscGraph &asc_graph, bool &is_in_blacklist) {
  auto &blacklist2 = autofuse::AutoFuseConfig::Config().GetFusionStrategySolver().improve_precision_blacklist;
  bool has_all = (blacklist2.find(kAllNodesType) != blacklist2.end());
  // 如果有"all"，则对所有节点检查数据类型
  if (has_all) {
    GELOGI("Graph %s: 'all' improve precision blacklist flag is set, checking dtype for all nodes",
           asc_graph.GetName().c_str());
    for (const auto &node : AscGraphUtils::GetComputeGraph(asc_graph)->GetAllNodes()) {
      // 跳过输入输出节点
      if (BackendUtils::IsOutputNode(node) || BackendUtils::IsInputNode(node)) {
        continue;
      }
      // 检查节点数据类型
      auto status = CheckNodeDtype(node);
      if (status != SUCCESS) {
        // 节点必须升精度
        GELOGI("Graph %s: node %s(%s) must improve precision, 'all' flag ignored for this graph",
               asc_graph.GetName().c_str(), node->GetName().c_str(), node->GetType().c_str());
        is_in_blacklist = false;
        return SUCCESS;
      }
    }
    // 所有节点都通过了数据类型检查，不升精度
    GELOGI("Graph %s: all nodes passed dtype check, skip precision improvement", asc_graph.GetName().c_str());
    return SUCCESS;
  }
  // 没有"all"，走按ascir配置黑名单处理的流程
  for (const auto &node : AscGraphUtils::GetComputeGraph(asc_graph)->GetAllNodes()) {
    if (IsInBlackList1(node)) {
      // 在内部写死升精度黑名单, doing nothing
    } else if (IsInBlackList2(node)) {
      // 在外部配置升精度黑名单, 如果dtype不是后端支持的类型则报错
      GE_ASSERT_SUCCESS(CheckNodeDtype(node));
    } else {
      // 既不在内部写死升精度黑名单，又不在外部配置升精度黑名单，则走默认升精度
      is_in_blacklist = false;
      return SUCCESS;
    }
  }
  return SUCCESS;
}

Status GetTypeToNodesFromGraph(AscGraph &asc_graph,
                               std::unordered_map<std::string, std::vector<NodePtr>> &type_to_nodes) {
  for (const auto &node : AscGraphUtils::GetComputeGraph(asc_graph)->GetAllNodes()) {
    // 输入和Output不需要升精度处理
    auto node_type = node->GetType();
    if (BackendUtils::IsOutputNode(node) || BackendUtils::IsInputNode(node)) {
      continue;
    }
    // 获取当前节点的输出描述符
    GeTensorDescPtr output_tensor_desc;
    GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc));

    GELOGI("To find node to change precision, current node(%s), type:%s, dtype:%s in graph %s.",
           node->GetName().c_str(), node->GetType().c_str(),
           TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str(), asc_graph.GetName().c_str());
    auto it = kTypeToGroup.find(node_type);
    const std::string &group_name = (it != kTypeToGroup.end()) ? it->second : "Other";
    type_to_nodes[group_name].push_back(node);
  }
  return SUCCESS;
}

Status ImprovePrecision(AscGraph &asc_graph, [[maybe_unused]] const NodePtr &asc_node) {
  // 后端不支持concat后有cast的情况，当前分析不出这是concat特例还是搬运类算子的普遍情况，暂时先跳过concat，如果是普遍情况还需要和后端Scheduler讨论处理方案
  const auto asc_node_attr = BackendUtils::GetNodeAutoFuseAttr(asc_node);
  GE_ASSERT_NOTNULL(asc_node_attr);
  if (asc_node_attr->HasFuseType(loop::FuseType::kConcat)||asc_node_attr->HasFuseType(loop::FuseType::kSplit)) {
    GELOGI("graph %s fuse type is concat, don't improve precision.", asc_graph.GetName().c_str());
    return SUCCESS;
  }
  if (BackendUtils::IsCubeAscNode(asc_node)) {
    GELOGI("graph %s fuse type is cube, don't improve precision.", asc_graph.GetName().c_str());
    return SUCCESS;
  }

  bool is_in_blacklist = true;
  GE_ASSERT_SUCCESS(IsAllNodesInBlacklist(asc_graph, is_in_blacklist));
  if(is_in_blacklist) {
    GELOGI("graph %s all nodes are in blacklist, don't improve precision.", asc_graph.GetName().c_str());
    return SUCCESS;
  }

  std::unordered_map<std::string, std::vector<NodePtr>> type_to_nodes;
  GE_ASSERT_SUCCESS(GetTypeToNodesFromGraph(asc_graph, type_to_nodes));
  for (const auto &node : type_to_nodes[kCastType]) {
    GE_ASSERT_SUCCESS(CastNodeProc(asc_graph, node));
  }
  for (const auto &node : type_to_nodes[kLoadType]) {
    bool is_need_insert_cast = false;
    GE_ASSERT_SUCCESS(IsNeedInsertCastAfterLoad(node, is_need_insert_cast));
    GE_ASSERT_SUCCESS(InsertCastToIncreasePrecision(asc_graph, node, is_need_insert_cast));
  }
  for (const auto &node : type_to_nodes[kGatherType]) {
    bool is_need_insert_cast = false;
    GE_ASSERT_SUCCESS(IsNeedInsertCastAfterLoad(node, is_need_insert_cast));
    GE_ASSERT_SUCCESS(InsertCastToIncreasePrecision(asc_graph, node, is_need_insert_cast));
  }
  for (const auto &node : type_to_nodes[kScalarType]) {
    // Scalar是个常量，前面没有节点,后面也没有load节点，导致不会插入cast，就直接改dtype从fp16/bf16升精度到fp32
    auto out_nodes = node->GetOutNodes();
    // scalar直接链接store不需要修改类型,在内部写死升精度黑名单会判断scalar直连store场景不做升精度处理
    GE_ASSERT_SUCCESS(asc_adapt::FromDtypeToOtherDtype(node, DT_BF16, DT_FLOAT));
    GE_ASSERT_SUCCESS(asc_adapt::FromDtypeToOtherDtype(node, DT_FLOAT16, DT_FLOAT));
  }
  for (const auto &node : type_to_nodes["Other"]) {
    bool is_need_insert_cast = false;
    std::vector<int32_t> input_idxs;
    GE_ASSERT_SUCCESS(IsNeedInsertCastBeforeOther(asc_graph, node, is_need_insert_cast, input_idxs));
    bool is_increase_precision = true;
    GE_ASSERT_SUCCESS(InsertCastBeforeNode(asc_graph, node, is_need_insert_cast, is_increase_precision, input_idxs));
    GE_ASSERT_SUCCESS(asc_adapt::FromDtypeToOtherDtype(node, DT_BF16, DT_FLOAT));
    GE_ASSERT_SUCCESS(asc_adapt::FromDtypeToOtherDtype(node, DT_FLOAT16, DT_FLOAT));
  }
  for (const auto &node : type_to_nodes[kStoreType]) {
    bool is_need_insert_cast = false;
    bool is_increase_precision = true;
    GE_ASSERT_SUCCESS(IsNeedInsertCastBeforeStore(asc_graph, node, is_need_insert_cast, is_increase_precision));
    GE_ASSERT_SUCCESS(InsertCastBeforeNode(asc_graph, node, is_need_insert_cast, is_increase_precision, {0}));
  }
  // 给ascGraph的节点按照topo id排序，后端依赖排序后的节点顺序
  asc_adapt::TopologicalSorting(AscGraphUtils::GetComputeGraph(asc_graph));
  return SUCCESS;
}

Status PrintImprovePrecisionBlacklist2() {
  const auto &blacklist2 = autofuse::AutoFuseConfig::Config().GetFusionStrategySolver().improve_precision_blacklist;
  if (blacklist2.empty()) {
    GELOGI("improve precision blacklist is empty");
    return SUCCESS;
  }
  std::stringstream ss;
  for (const auto &str : blacklist2) {
    ss << str << " ";
  }
  GELOGI("improve precision blacklist: %s", ss.str().c_str());
  return SUCCESS;
}
}  // namespace

Status PrecisionImprover::ImprovePrecisionToFp32(const ComputeGraphPtr &ge_or_fused_asc_backend_graph) {
  GE_ASSERT_SUCCESS(PrintImprovePrecisionBlacklist2());
  GE_ASSERT_SUCCESS(
      asc_adapt::ProcessAscBackendNodes(ge_or_fused_asc_backend_graph, ImprovePrecision, "improve_precision"));
  return SUCCESS;
}
}  // namespace ge

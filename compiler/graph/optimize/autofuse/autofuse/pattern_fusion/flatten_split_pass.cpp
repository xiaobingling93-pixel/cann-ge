/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flatten_split_pass.h"

#include "ascir.h"
#include "backend/backend_spec.h"
#include "common/checker.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/debug/ge_op_types.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "register/shape_inference.h"
#include "utils/cg_utils.h"
#include "utils/auto_fuse_config.h"
#include "utils/autofuse_utils.h"
#include "lowering/asc_lowerer/loop_common.h"
#include "base/err_msg.h"

namespace ge {
constexpr int32_t kSplitVDataInputIndex = 0U;
constexpr int32_t kSplitVSizeSplitsInputIndex = 1U;
constexpr int32_t kSplitVSplitDimInputIndex = 2U;
namespace {
ge::NodePtr CreateNewSplit(const ComputeGraphPtr &graph, uint32_t output_cnt_per_split, NodePtr &split_end_node) {
  const std::string new_split_node_name = "Fusion_" + split_end_node->GetName();
  const std::string node_type = AF_SPLITV;
  auto op = ge::OperatorFactory::CreateOperator(new_split_node_name.c_str(), node_type.c_str());
  op.DynamicOutputRegister("y", output_cnt_per_split);
  op.SetAttr("num_split", output_cnt_per_split);
  op.BreakConnect();
  const auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto new_node = graph->AddNode(op_desc);
  GELOGD("create new split node: output count num %u; split node name is %s end",
         output_cnt_per_split, new_split_node_name.c_str());
  return new_node;
}

bool GetSplitNodeSplitDim(const NodePtr &split_node, int64_t &node_split_dim_value) {
  int64_t split_dim = 0;
  auto op_desc = split_node->GetOpDesc();
  auto input_desc_ptr = op_desc->GetInputDescPtr(0);
  if (input_desc_ptr->GetShape().IsUnknownDimNum()) {
    return false;
  }
  auto input_dim_num = static_cast<int64_t>(input_desc_ptr->GetShape().GetDimNum());
  GELOGD("shape of input dim_num of node:%s(%s) is [%ld]", split_node->GetName().c_str(), split_node->GetType().c_str(),
         input_dim_num);
  if ((AF_SPLIT == split_node->GetType()) || (AF_SPLITV == split_node->GetType())) {
    int32_t index = op_desc->GetInputIndexByName("split_dim");
    auto op = ge::OpDescUtils::CreateOperatorFromNode(split_node);
    const GeTensor *perm_tensor = ge::OpDescUtils::GetInputConstData(op, index);
    GE_WARN_ASSERT(perm_tensor != nullptr, "split dim input is not const data");

    const auto &tensor_desc = perm_tensor->GetTensorDesc();
    GE_ASSERT_TRUE((tensor_desc.GetShape().GetShapeSize() == 1) || (tensor_desc.GetShape().IsScalar()));

    if (tensor_desc.GetDataType() == DT_INT32) {
      const auto *perm_data = reinterpret_cast<const int32_t *>(perm_tensor->GetData().data());
      split_dim = static_cast<int64_t>(perm_data[0]);
    } else if (tensor_desc.GetDataType() == DT_INT64) {
      const auto *perm_data = reinterpret_cast<const int64_t *>(perm_tensor->GetData().data());
      split_dim = static_cast<int64_t>(perm_data[0]);
    } else {
      REPORT_INNER_ERR_MSG("E19999", "Remove node:%s(%s)", split_node->GetName().c_str(), split_node->GetType().c_str());
      return false;
    }
  } else {
    (void)ge::AttrUtils::GetInt(split_node->GetOpDesc(), "split_dim", split_dim);
  }

  if (split_dim < 0) {
    split_dim += (input_dim_num);  // 如果拼接轴是负数，表示是倒数第N个轴，将其转变为正数的轴。
  }
  node_split_dim_value = split_dim;
  return true;
}

graphStatus GetSplitOutputBufferShape(const NodePtr &split_node, std::vector<Expression> &dims) {
  const auto split_op_desc = split_node->GetOpDesc();
  GE_ASSERT_NOTNULL(split_op_desc);
  GE_ASSERT_TRUE(split_op_desc->GetAllOutputsDescSize() > 0U);
  const auto desc = split_op_desc->GetOutputDescPtr(0U);;
  GE_ASSERT_NOTNULL(desc);
  const auto sym_attr = desc->GetAttrsGroup<SymbolicDescAttr>();
  GE_ASSERT_NOTNULL(sym_attr);
  dims = sym_attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  return GRAPH_SUCCESS;
}

bool SplitNodeSplitSizeIsConst(const NodePtr &split_node, const int64_t original_split_dim) {
  const auto split_type = split_node->GetType();
  uint32_t node_split_num = 1U;
  bool is_split_size_const = true;
  if ((split_type == AF_SPLITD) || (split_type == AF_SPLIT)) {
    (void)ge::AttrUtils::GetInt(split_node->GetOpDesc(), "num_split", node_split_num);
    std::vector<ge::Expression> x_dims;
    GE_ASSERT_GRAPH_SUCCESS(GetSplitOutputBufferShape(split_node, x_dims));
    GE_ASSERT_TRUE((original_split_dim >= 0L) && (x_dims.size() > static_cast<size_t>(original_split_dim)));
    is_split_size_const = x_dims[original_split_dim].IsConstExpr();
    GELOGD("split node name is %s, size of split dim %s.",
           split_node->GetName().c_str(), is_split_size_const ? "is const" : "is not const");
  }
  return is_split_size_const;
}

graphStatus GetSplitSize(const NodePtr &split_node, std::vector<int64_t> &split_size, const int64_t original_split_dim) {
  const auto split_type = split_node->GetType();
  if ((split_type == AF_SPLITD) || (split_type == AF_SPLIT)) {
    uint32_t node_split_num = 1U;
    (void)ge::AttrUtils::GetInt(split_node->GetOpDesc(), "num_split", node_split_num);
    std::vector<ge::Expression> x_dims;
    GE_ASSERT_GRAPH_SUCCESS(GetSplitOutputBufferShape(split_node, x_dims));
    GE_ASSERT_TRUE((original_split_dim >= 0L) && (x_dims.size() > static_cast<size_t>(original_split_dim)));
    GE_WARN_ASSERT(x_dims[original_split_dim].IsConstExpr(), "split dim of output[%zu] is non-const", original_split_dim);
    int64_t dim = -1L;
    GE_ASSERT_TRUE(x_dims[original_split_dim].GetConstValue(dim), "Failed to get int value, expr = %s",
                   x_dims[original_split_dim].Str().get());
    GELOGD("split node split size is: peer in node name is %s, split size is %ld, node_split_num is %d",
        split_node->GetName().c_str(), dim, node_split_num);
    for (uint32_t index = 0U; index < node_split_num; index++) {
      split_size.push_back(dim);
    }
  } else {
    GE_ASSERT_GRAPH_SUCCESS(AutofuseUtils::GetListIntByInputOrAttr(split_node, split_size, "size_splits", "size_splits"));
  }
  return GRAPH_SUCCESS;
}

uint32_t SplitPeerInNodeFuseParaCal(const NodePtr &split_node, const NodePtr &peer_in_node, int64_t original_split_dim,
                                    uint32_t &split_node_flag, std::vector<uint32_t> &split_node_flagvec) {
  uint32_t peer_in_node_split_num = 1U;
  GE_ASSERT_NOTNULL(peer_in_node);
  if (AutofuseUtils::IsSplitType(peer_in_node->GetType())) {
    int64_t split_dim_value = 0L;
    auto split_dim_can_get_flag = GetSplitNodeSplitDim(peer_in_node, split_dim_value);
    auto split_size_const_flag = SplitNodeSplitSizeIsConst(split_node, original_split_dim);
    GELOGD("fuse split node info:original split dim is %d, fuse split dim is %d", original_split_dim,
           split_dim_value);
    if ((original_split_dim != split_dim_value) || (split_dim_can_get_flag != true) || (split_size_const_flag == false)) {
      split_node_flag = static_cast<uint32_t>(0);
      peer_in_node_split_num = static_cast<uint32_t>(1);
    } else {
      split_node_flag = static_cast<uint32_t>(1);
      (void)ge::AttrUtils::GetInt(peer_in_node->GetOpDesc(), "num_split", peer_in_node_split_num);
    }
    GELOGD("fuse node %s(%s) info: split dim can get flag[%d], final split flag[%d]",
           split_node->GetName().c_str(), split_node->GetType().c_str(),
           split_dim_can_get_flag, split_node_flag);
  }
  split_node_flagvec.push_back(split_node_flag);
  return peer_in_node_split_num;
}

graphStatus SplitNodeNeedCombineProcess(const NodePtr &out_ower_node, uint32_t &fusion_new_node_anchor_idx,
                                        std::vector<int32_t> &outputs_map, uint32_t out_anchor_index_outputs_map,
                                        std::vector<OutDataAnchorPtr> &new_split_out_data_anchors) {
  const auto out_data_anchor_size = out_ower_node->GetAllOutDataAnchorsSize();
  for (uint32_t index = 0; index < out_data_anchor_size; index++) {
    const auto output_index_temp = out_ower_node->GetOpDesc()->GetOutputIndexByName("y" + std::to_string(index));
    if (output_index_temp == (-1)) {
      break;
    }
    outputs_map.push_back(static_cast<int32_t>(out_anchor_index_outputs_map) + output_index_temp);
    new_split_out_data_anchors.push_back(out_ower_node->GetOutDataAnchor(output_index_temp));
    fusion_new_node_anchor_idx++;
  }
  return GRAPH_SUCCESS;
}

graphStatus NewSplitNodeUpdateDesc(NodePtr &split_node, NodePtr &new_split_node,
                                   std::vector<OutDataAnchorPtr> &new_split_out_data_anchors) {
  uint32_t new_node_output_index = 0U;
  auto new_split_op_desc = new_split_node->GetOpDesc();
  GE_ASSERT_NOTNULL(new_split_op_desc);
  for (const auto &out_data_anchor : new_split_out_data_anchors) {
    const auto owner_node = out_data_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(owner_node);
    const auto owner_node_op_desc = owner_node->GetOpDesc();
    GE_ASSERT_NOTNULL(owner_node_op_desc);
    const auto output_anchor_idx = out_data_anchor->GetIdx();
    GE_ASSERT_TRUE(output_anchor_idx >= 0);
    GE_ASSERT_TRUE(owner_node_op_desc->GetAllOutputsDescSize() > static_cast<uint32_t>(output_anchor_idx));
    const auto output_desc_y = owner_node_op_desc->GetOutputDesc(output_anchor_idx);
    GE_ASSERT_GRAPH_SUCCESS(new_split_op_desc->UpdateOutputDesc(new_node_output_index, output_desc_y));
    new_node_output_index++;
  }
  GE_ASSERT_TRUE(new_split_node->GetAllInDataAnchorsSize() > kSplitVSplitDimInputIndex);
  auto old_split_op_desc = split_node->GetOpDesc();
  GE_ASSERT_NOTNULL(old_split_op_desc);
  const auto split_node_input_index = old_split_op_desc->GetInputIndexByName("x");
  GE_ASSERT_TRUE(split_node_input_index >= 0);
  const auto input_desc_x = old_split_op_desc->GetInputDesc(split_node_input_index);
  GE_ASSERT_GRAPH_SUCCESS(new_split_op_desc->UpdateInputDesc(kSplitVDataInputIndex, input_desc_x));
  GELOGD("peer in node name is %s, fusion new node anchor Idx is %zu",
         new_split_node->GetName().c_str(), split_node_input_index);
  const auto old_split_node_type = split_node->GetType();
  if (old_split_node_type == AF_SPLITV || old_split_node_type == AF_SPLIT) {
    const auto split_dim_input_index = old_split_op_desc->GetInputIndexByName("split_dim");
    GE_ASSERT_TRUE(split_dim_input_index >= 0);
    const auto input_desc_split_dim = old_split_op_desc->GetInputDesc(split_dim_input_index);
    GE_ASSERT_GRAPH_SUCCESS(new_split_op_desc->UpdateInputDesc(kSplitVSplitDimInputIndex, input_desc_split_dim));
  } else {
    auto split_dim_td = new_split_op_desc->MutableInputDesc(kSplitVSplitDimInputIndex);
    GE_ASSERT_NOTNULL(split_dim_td);
    split_dim_td->SetDataType(DT_INT64);
    split_dim_td->SetFormat(FORMAT_ND);
    split_dim_td->SetShape(GeShape(std::vector<int64_t>{1L}));
  }
  new_split_op_desc->SetIsInputConst({false, true, true});
  return GRAPH_SUCCESS;
}

NodePtr CreateNewConstantNode(const ComputeGraphPtr &graph, const std::string &name, std::vector<int64_t> &data,
                              const ge::Format format = FORMAT_ND, const ge::DataType dtype = DT_INT32) {
  const auto const_op = ge::OperatorFactory::CreateOperator(name.c_str(), CONSTANT);
  const_op.BreakConnect();
  const auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(const_op);
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(data.size() > 0U);
  GE_ASSERT_TRUE(op_desc->GetAllOutputsDescSize() > 0U);
  const auto tensor_desc = op_desc->MutableOutputDesc(0U);
  GE_ASSERT_NOTNULL(tensor_desc);
  tensor_desc->SetShape(GeShape(std::vector{static_cast<int64_t>(data.size())}));
  tensor_desc->SetOriginShape(GeShape(std::vector{static_cast<int64_t>(data.size())}));
  tensor_desc->SetDataType(dtype);
  tensor_desc->SetOriginDataType(dtype);
  tensor_desc->SetFormat(format);
  tensor_desc->SetOriginFormat(format);
  GeTensorPtr tensor = nullptr;
  if (dtype == DT_INT32) {
    std::vector<int32_t> reformated_data(data.begin(), data.end());
    tensor = ComGraphMakeShared<GeTensor>(*tensor_desc, reinterpret_cast<uint8_t *>(reformated_data.data()), sizeof(int32_t) * data.size());
  } else if (dtype == DT_INT64) {
    tensor = ComGraphMakeShared<GeTensor>(*tensor_desc, reinterpret_cast<uint8_t *>(data.data()), sizeof(int64_t) * data.size());
  }
  GE_ASSERT_NOTNULL(tensor);
  AttrUtils::SetTensor(op_desc, "value", tensor);
  auto new_const_node = graph->AddNode(op_desc);
  GE_ASSERT_NOTNULL(new_const_node);
  auto new_const_node_op_desc = new_const_node->GetOpDesc();
  GE_ASSERT_NOTNULL(new_const_node_op_desc);
  GE_ASSERT_TRUE(new_const_node_op_desc->GetAllOutputsDescSize() > 0U);
  GE_ASSERT_GRAPH_SUCCESS(new_const_node_op_desc->UpdateOutputDesc(0U, *tensor_desc));
  return new_const_node;
}

// 提取常量节点数据格式信息
graphStatus ExtractConstantNodeFormatInfo(const NodePtr &old_split_node, Format &format, DataType &dtype) {
  if (old_split_node->GetType() == AF_SPLITV) {
    const auto op = ge::OpDescUtils::CreateOperatorFromNode(old_split_node);
    ge::Tensor val_tensor;
    GE_ASSERT_SUCCESS(op.GetInputConstData("size_splits", val_tensor));
    const auto tensor_desc = val_tensor.GetTensorDesc();
    format = tensor_desc.GetFormat();
    dtype = tensor_desc.GetDataType();
  } else {
    format = FORMAT_ND;
    dtype = DT_INT32;
  }
  return GRAPH_SUCCESS;
}

// 创建并连接size_splits常量节点
graphStatus CreateAndConnectSizeSplitsNode(const ComputeGraphPtr &graph, std::vector<int64_t> &list_size_splits,
                                           const NodePtr &new_split_node, const Format format, const DataType dtype) {
  const std::string size_splits_node_name = "SizeSplitsOf" + new_split_node->GetName();
  const auto new_size_split_node = CreateNewConstantNode(graph, size_splits_node_name, list_size_splits, format, dtype);
  GE_ASSERT_NOTNULL(new_size_split_node);
  GE_ASSERT_TRUE(new_size_split_node->GetAllOutDataAnchorsSize() > 0U);

  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(new_size_split_node->GetOutDataAnchor(0U),
                                              new_split_node->GetInDataAnchor(kSplitVSizeSplitsInputIndex)));

  GELOGD("add edge for new size split const data node %s and new split node %s, in data anchor is %d",
         new_size_split_node->GetName().c_str(), new_split_node->GetName().c_str(), kSplitVSizeSplitsInputIndex);

  const auto size_split_op_desc = new_size_split_node->GetOpDesc();
  GE_ASSERT_NOTNULL(size_split_op_desc);
  const auto tensor_desc = size_split_op_desc->GetOutputDesc(0U);
  const auto splitv_op_desc = new_split_node->GetOpDesc();
  GE_ASSERT_NOTNULL(splitv_op_desc);
  GE_ASSERT_GRAPH_SUCCESS(splitv_op_desc->UpdateInputDesc(kSplitVSizeSplitsInputIndex, tensor_desc));

  return GRAPH_SUCCESS;
}

// 处理主输入数据连接
graphStatus HandleMainInputConnection(const NodePtr &old_split_node, const NodePtr &new_split_node) {
  const auto old_split_op_desc = old_split_node->GetOpDesc();
  const auto input_data_index = old_split_op_desc->GetInputIndexByName("x");
  GE_ASSERT_TRUE(input_data_index >= 0);

  auto in_data_anchor = old_split_node->GetInDataAnchor(input_data_index);
  auto in_data_anchor_peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(in_data_anchor_peer_out_anchor);

  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(in_data_anchor_peer_out_anchor,
                                             new_split_node->GetInDataAnchor(kSplitVDataInputIndex)));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(in_data_anchor_peer_out_anchor, in_data_anchor));

  return GRAPH_SUCCESS;
}

// 处理split_dim输入连接（针对SplitV/Split节点）
graphStatus HandleSplitDimConnectionForSplitV(const NodePtr &old_split_node, const NodePtr &new_split_node) {
  auto old_split_op_desc = old_split_node->GetOpDesc();
  const auto split_dim_input_index = old_split_op_desc->GetInputIndexByName("split_dim");
  GE_ASSERT_TRUE(split_dim_input_index >= 0);

  auto in_data_anchor_split_dim = old_split_node->GetInDataAnchor(split_dim_input_index);
  auto split_dim_peer_out_anchor = in_data_anchor_split_dim->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(split_dim_peer_out_anchor);

  auto peer_out_split_dim_const_data_node = split_dim_peer_out_anchor->GetOwnerNodeBarePtr();
  GE_ASSERT_NOTNULL(peer_out_split_dim_const_data_node);

  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(split_dim_peer_out_anchor,
                                             new_split_node->GetInDataAnchor(kSplitVSplitDimInputIndex)));

  auto split_dim_op_desc = peer_out_split_dim_const_data_node->GetOpDesc();
  GE_ASSERT_NOTNULL(split_dim_op_desc);
  GE_ASSERT_TRUE(split_dim_op_desc->GetAllOutputsDescSize() > 0U);

  auto tensor_desc_dim = split_dim_op_desc->GetOutputDesc(0U);
  auto new_split_node_op_desc = new_split_node->GetOpDesc();
  GE_ASSERT_NOTNULL(new_split_node_op_desc);
  GE_ASSERT_GRAPH_SUCCESS(new_split_node_op_desc->UpdateInputDesc(kSplitVSplitDimInputIndex, tensor_desc_dim));

  GELOGD("add edge for split dim const data node and new split node %s, in data anchor is %d",
         new_split_node->GetName().c_str(), kSplitVSplitDimInputIndex);

  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(split_dim_peer_out_anchor, in_data_anchor_split_dim));

  return GRAPH_SUCCESS;
}

// 处理split_dim输入连接（针对SplitD节点）
graphStatus HandleSplitDimConnectionForSplitD(const NodePtr &old_split_node,
                                              const ComputeGraphPtr &graph,
                                              const NodePtr &new_split_node) {
  auto old_split_op_desc = old_split_node->GetOpDesc();
  AnyValue split_dim_data;
  old_split_op_desc->GetAttr("split_dim", split_dim_data);
  int64_t split_dim_raw_data = -1L;
  GE_ASSERT_GRAPH_SUCCESS(split_dim_data.GetValue<int64_t>(split_dim_raw_data));
  GE_ASSERT_TRUE(split_dim_raw_data >= 0L);

  std::vector<int64_t> list_split_dim = {split_dim_raw_data};
  const std::string split_num_node_name = "SplitNumOf" + new_split_node->GetName();
  auto new_split_dim_node = CreateNewConstantNode(graph, split_num_node_name, list_split_dim);
  GE_ASSERT_NOTNULL(new_split_dim_node);
  GE_ASSERT_TRUE(new_split_dim_node->GetAllOutDataAnchorsSize() > 0U);

  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(new_split_dim_node->GetOutDataAnchor(0U),
                                             new_split_node->GetInDataAnchor(kSplitVSplitDimInputIndex)));

  return GRAPH_SUCCESS;
}

// 主函数重构
graphStatus CreateConstDataNodesWithOldSplitNode(const ComputeGraphPtr &graph,
                                                 std::vector<int64_t> &list_size_splits,
                                                 const NodePtr &old_split_node,
                                                 const NodePtr &new_split_node) {
  GE_ASSERT_TRUE(new_split_node->GetAllInDataAnchorsSize() > kSplitVSplitDimInputIndex);
  auto splitv_op_desc = new_split_node->GetOpDesc();
  GE_ASSERT_NOTNULL(splitv_op_desc);
  GE_ASSERT_NOTNULL(old_split_node);
  auto old_split_op_desc = old_split_node->GetOpDesc();
  GE_ASSERT_NOTNULL(old_split_op_desc);

  // 提取格式信息
  Format format;
  DataType dtype;
  GE_ASSERT_SUCCESS(ExtractConstantNodeFormatInfo(old_split_node, format, dtype));

  // 创建并连接size_splits节点
  GE_ASSERT_SUCCESS(CreateAndConnectSizeSplitsNode(graph, list_size_splits, new_split_node, format, dtype));

  // 处理主输入连接
  GE_ASSERT_SUCCESS(HandleMainInputConnection(old_split_node, new_split_node));

  // 根据节点类型处理split_dim连接
  if (old_split_node->GetType() == AF_SPLITV || old_split_node->GetType() == AF_SPLIT) {
    GE_ASSERT_SUCCESS(HandleSplitDimConnectionForSplitV(old_split_node, new_split_node));
  } else {
    // SplitD模板
    GE_ASSERT_SUCCESS(HandleSplitDimConnectionForSplitD(old_split_node, graph, new_split_node));
  }

  return GRAPH_SUCCESS;
}

graphStatus SplitNodeRealizeCombine(const ComputeGraphPtr &graph, NodePtr &split_node, NodePtr &new_split_node,
                                    const std::vector<uint32_t> &split_node_flagVec, const std::vector<int32_t> &output_indices,
                                    const int64_t original_split_dim) {
  uint32_t flag_index = 0;
  uint32_t fusion_new_node_anchor_idx = 0;
  std::vector<NodePtr> new_nodes_lists;
  std::vector<NodePtr> old_nodes_lists;
  std::vector<int32_t> inputs_map;
  std::vector<int32_t> outputs_map;
  int32_t out_anchor_index_outputs_map = split_node->GetAllOutDataAnchorsSize();
  std::vector<OutDataAnchorPtr> new_split_out_data_anchors;

  old_nodes_lists.push_back(split_node);
  new_nodes_lists.push_back(new_split_node);
  std::vector<int64_t> new_node_size_splits;
  std::vector<int64_t> split_node_size_splits;
  GE_ASSERT_GRAPH_SUCCESS(GetSplitSize(split_node, split_node_size_splits, original_split_dim));
  for (auto out_data_anchor_index : output_indices) {
    auto out_anchor_owner_node_pair = NodeUtils::GetOutDataNodesWithAnchorByIndex(*split_node, out_data_anchor_index);
    GE_ASSERT_TRUE(out_anchor_owner_node_pair.size() == 1U);
    auto out_owner_node = out_anchor_owner_node_pair[0U].second;
    GE_ASSERT_NOTNULL(out_owner_node);
    GELOGD("split node output anchor info: peer in node name is %s, fusion new node anchor Idx is %d",
           out_owner_node->GetName().c_str(), fusion_new_node_anchor_idx);

    if (split_node_flagVec[flag_index] == 0U) {
      outputs_map.push_back(out_data_anchor_index);
      new_split_out_data_anchors.push_back(split_node->GetOutDataAnchor(out_data_anchor_index));
      fusion_new_node_anchor_idx++;
      GE_ASSERT_TRUE((out_data_anchor_index >= 0) && (split_node_size_splits.size() > static_cast<size_t>(out_data_anchor_index)));
      new_node_size_splits.push_back(split_node_size_splits[out_data_anchor_index]);
    } else {
      std::vector<int64_t> curr_node_size_splits;
      GE_ASSERT_SUCCESS(SplitNodeNeedCombineProcess(out_owner_node, fusion_new_node_anchor_idx, outputs_map,
                                                     out_anchor_index_outputs_map, new_split_out_data_anchors));
      old_nodes_lists.push_back(out_owner_node);
      out_anchor_index_outputs_map += out_owner_node->GetAllOutDataAnchorsSize();
      GE_ASSERT_GRAPH_SUCCESS(GetSplitSize(out_owner_node, curr_node_size_splits, original_split_dim));
      for (auto size_split : curr_node_size_splits) {
        new_node_size_splits.push_back(size_split);
      }
    }
    flag_index++;
  }
  GE_CHK_STATUS(GraphUtils::ReplaceNodesDataAnchors(new_nodes_lists, old_nodes_lists, inputs_map, outputs_map));
  GE_CHK_STATUS(GraphUtils::InheritExecutionOrder(new_nodes_lists, old_nodes_lists, graph));

  GE_CHK_STATUS(CreateConstDataNodesWithOldSplitNode(graph, new_node_size_splits, split_node, new_split_node));
  GE_CHK_STATUS(NewSplitNodeUpdateDesc(split_node, new_split_node, new_split_out_data_anchors));

  for (auto node : old_nodes_lists) {
    if (graph->RemoveNode(node) != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Remove node:%s(%s) from graph:%s failed", new_split_node->GetName().c_str(),
                        node->GetType().c_str(), graph->GetName().c_str());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus SplitNodeCombine(const ComputeGraphPtr &graph, NodePtr &split_node) {
  uint32_t fusion_out_count_total = 0;
  uint32_t cur_fuse_split_cnt = 0;
  NodePtr out_owner_node = nullptr;
  uint32_t split_node_flag = 0;
  uint32_t fuse_flag = 0;
  int64_t original_split_dim = 0;
  std::vector<int32_t> output_indices;
  std::vector<uint32_t> split_node_flag_vec;
  const uint32_t out_data_anchor_size = split_node->GetAllOutDataAnchorsSize();
  std::vector<NodePtr> cycle_nodes_lists;
  const auto split_op_desc = split_node->GetOpDesc();
  GE_ASSERT_NOTNULL(split_op_desc);
  if (!GetSplitNodeSplitDim(split_node, original_split_dim)) {
    return GRAPH_SUCCESS;
  }

  for (uint32_t index = 0; index < out_data_anchor_size; index++) {
    int32_t output_index_temp = split_op_desc->GetOutputIndexByName("y" + std::to_string(index));
    GE_ASSERT_TRUE(output_index_temp >= 0);
    output_indices.push_back(output_index_temp);
  }

  for (auto out_data_anchor_index : output_indices) {
    split_node_flag = 0;
    auto out_data_anchor = split_node->GetOutDataAnchor(out_data_anchor_index);
    for (auto peer_in_data_anchor: out_data_anchor->GetPeerInDataAnchorsPtr()) {
      out_owner_node = peer_in_data_anchor->GetOwnerNode();
      GE_ASSERT_NOTNULL(out_owner_node);
      cur_fuse_split_cnt =
          SplitPeerInNodeFuseParaCal(split_node, out_owner_node, original_split_dim, split_node_flag, split_node_flag_vec);
      GELOGD("out_owner_node: %s(%s), cur_fuse_split_cnt %d", out_owner_node->GetType().c_str(),
             out_owner_node->GetName().c_str(), cur_fuse_split_cnt);
      fusion_out_count_total += cur_fuse_split_cnt;
      fuse_flag += split_node_flag;
      if (split_node_flag != 0) {
        cycle_nodes_lists.push_back(out_owner_node);
      }
    }
  }

  if (static_cast<int>(fuse_flag) == 0 ||
      FlattenSplitPass::CanFlatten(split_node, original_split_dim, fusion_out_count_total) != ge::GRAPH_SUCCESS) {
    return GRAPH_SUCCESS;
  }
  /** 判断融合节点是否成环**/
  cycle_nodes_lists.push_back(split_node);
  const CycleDetectorSharedPtr cycle_detector = GraphUtils::CreateSharedCycleDetector(graph);
  const bool has_cycle_flag = cycle_detector->HasDetectedCycle({cycle_nodes_lists});
  GELOGD("Cycle flag of node %s(%s), Graph %s is : %u;", split_node->GetName().c_str(), split_node->GetType().c_str(),
         graph->GetName().c_str(), has_cycle_flag);
  if (has_cycle_flag == true) {
    return GRAPH_SUCCESS;
  }

  GELOGD("can fuse split node info:split_node_name is %s;", split_node->GetName().c_str());
  auto new_split_node = CreateNewSplit(graph, fusion_out_count_total, split_node);
  GE_ASSERT_NOTNULL(new_split_node);
  GE_ASSERT_GRAPH_SUCCESS(SplitNodeRealizeCombine(graph, split_node, new_split_node, split_node_flag_vec, output_indices, original_split_dim));
  return GRAPH_SUCCESS;
}
}  // namespace

graphStatus FlattenSplitPass::Run(const ComputeGraphPtr &graph) {
  if (!autofuse::AutoFuseConfig::LoweringConfig().experimental_lowering_split) {
    GELOGI("you can enable split by setting AUTOFUSE_FLAGS=\"--autofuse_enable_pass=split\""
      "and unsetting AUTOFUSE_FLAGS=\"--autofuse_disable_pass=split\"");
    return ge::GRAPH_SUCCESS;
  }
  const auto backend_spec = optimize::BackendSpec::GetInstance();
  if (!backend_spec->slice_split_spec.enable_split_flatten) {
    GELOGI("Skip split flatten as split flatten is disabled.");
    return ge::GRAPH_SUCCESS;
  }
  GE_CHECK_NOTNULL(graph);
  GELOGD("SplitProBeforeAutoFuse:main func begin");

  for (auto &node : graph->GetDirectNode()) {
    if (AutofuseUtils::IsSplitType(node->GetType())) {
      GELOGD("split node info: node name is %s, out data anchors size is %d", node->GetName().c_str(),
             node->GetAllOutDataAnchorsSize());
      auto ret = SplitNodeCombine(graph, node);
      if (ret != SUCCESS) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

graphStatus FlattenSplitPass::CanFlatten(const NodePtr &node, const size_t split_dim, const size_t num_outputs) {
  constexpr size_t kMaxSingleOpOutputNum = 63;
  constexpr size_t kMaxFreeSymbols = 16; // 过多的symbol会导致TilingData大小膨胀，导致编译失败。在有需求削减前，先限制
  Expression split_dim_size;
  for (auto anchor: node->GetAllOutDataAnchors()) {
    std::vector<Expression> output_shape;
    GE_WARN_ASSERT(ge::loop::GetBufferShape(anchor, output_shape) == ge::GRAPH_SUCCESS);
    GE_WARN_ASSERT(split_dim < output_shape.size());
    split_dim_size = output_shape[split_dim] + split_dim_size;
    GE_CHK_BOOL_RET_SPECIAL_STATUS(anchor->GetPeerInDataNodesSize() != 1U,
                                   ge::GRAPH_FAILED,
                                   "number of peer in data anchor of output(%s_out_%zu) is %zu != 1, "
                                   "mismatch typical split flatten case, do not flatten split",
                                   node->GetName().c_str(), anchor->GetIdx(), anchor->GetPeerInDataAnchors().size());
  }
  // 如果FreeSymbols多，但融合完不超过单算子能处理的上限，也可以融，只是后续不Lowering
  GE_WARN_ASSERT((num_outputs <= kMaxSingleOpOutputNum) ||
      (split_dim_size.FreeSymbols().size() <= kMaxFreeSymbols),
                 "flatten split will cause too many free symbols, do not flatten split, input split dim size = %s",
                 split_dim_size.Str().get());
  return ge::GRAPH_SUCCESS;
}
}  // namespace ge
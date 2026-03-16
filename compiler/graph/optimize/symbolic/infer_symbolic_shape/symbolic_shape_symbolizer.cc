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
#include "common/checker.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/context/local_context.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_type_utils.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "framework/common/types.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "symbolic_infer_util.h"
#include "symbolic_shape_symbolizer.h"
#include "graph/optimize/symbolic/shape_env_guarder.h"
#include "graph/symbolizer/guard_dfx_context.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
namespace {
const std::vector<int64_t> kDummyShape = {-3};

struct DataSymbolizeInfo {
  int32_t dataIndex;
  GeShape dataShape;
  GeShape inputShape;
};

Status BuildDataSymbolizeInfo(const NodePtr &data_node, const std::vector<GeTensor> &graph_inputs,
  DataSymbolizeInfo &info) {
  auto op_desc = data_node->GetOpDescBarePtr();
  GE_ASSERT_TRUE(AttrUtils::GetInt(op_desc, "index", info.dataIndex), "get data node %s index failed",
    op_desc->GetName().c_str());
  GE_ASSERT_TRUE(static_cast<size_t>(info.dataIndex) < graph_inputs.size(),
    "Invalid data index %d, graph inputs size %zu", info.dataIndex, graph_inputs.size());

  auto td = op_desc->GetOutputDescPtr(0);
  GE_ASSERT_NOTNULL(td);
  info.dataShape = td->GetOriginShape();
  if (!(info.dataShape == td->GetShape())) {
    GELOGW("The origin/storage shape are different, not support symbolize yet, data node %s", op_desc->GetNamePtr());
    return ge::UNSUPPORTED;
  }

  const auto &tensor = graph_inputs.at(info.dataIndex);
  const auto &ge_tensor_desc = tensor.GetTensorDesc();
  if (ge_tensor_desc.IsOriginShapeInitialized()) {
    info.inputShape = ge_tensor_desc.GetOriginShape();
  } else {
    info.inputShape = ge_tensor_desc.GetShape();
  }
  // atc + acl场景动态shape下开启自动融合会产生[-3]的shape，在这里拦截报错
  GE_ASSERT_TRUE(info.inputShape.GetDims() != kDummyShape,
    "Node[%s] is not supported symbolize, input origin shape is [-3].", data_node->GetNamePtr());
  return SUCCESS;
}

std::map<ge::DataType, std::string> kGeDType2CppDtype = {
    {ge::DT_INT32, "int32_t"},
    {ge::DT_INT64, "int64_t"},
    {ge::DT_UINT32, "uint32_t"},
    {ge::DT_UINT64, "uint64_t"},
};

// 泛化value的类型，可扩展为：只泛化value、泛化value并且求和，泛化value并且求平均
const char_t *const kSymbolizeValueType = "_symbolize_value_type";
enum SymbolizeValueType {
  SYMBOLIZE_VALUE_TYPE_NONE = 0,
  SYMBOLIZE_VALUE_TYPE_ONLY,
  SYMBOLIZE_VALUE_TYPE_SUM,
  SYMBOLIZE_VALUE_TYPE_END
};

Status MarkSymbolizeRepeatInputValue(const ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() != "Repeat" || node->GetAllInDataAnchorsSize() < 2U) {
      continue;
    }
    auto in_data_anchor = node->GetInDataAnchor(1);
    GE_ASSERT_NOTNULL(in_data_anchor);
    auto peer_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(peer_anchor);
    auto peer_in_node = peer_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(peer_in_node);
    auto op_desc = peer_in_node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    if (peer_in_node->GetType() == ge::DATA) {
      GE_ASSERT_TRUE(AttrUtils::SetInt(op_desc, kSymbolizeValueType, SYMBOLIZE_VALUE_TYPE_SUM));
    }
  }
  return GRAPH_SUCCESS;
}

// 使用tensor里面的值求和，造一个symbol
template <typename T>
typename std::enable_if<std::is_integral<T>::value, std::unique_ptr<std::vector<Expression>>>::type CreateSymbolValueSum(
    ShapeEnvAttr *shape_env_attr, const GeTensor &tensor, int32_t data_index) {
  GE_ASSERT_NOTNULL(shape_env_attr);
  std::unique_ptr<std::vector<Expression>> result = MakeUnique<std::vector<Expression>>();
  GE_ASSERT_NOTNULL(result);
  const T *const data = reinterpret_cast<const T *>(tensor.GetData().GetData());
  GE_ASSERT_NOTNULL(data);
  const size_t dims_num = tensor.GetData().size() / sizeof(T);
  T sum = 0;
  for (size_t i = 0UL; i < dims_num; i++) {
    GE_ASSERT_TRUE(!ge::AddOverflow(sum, data[i], sum));
  }
  auto value_source = MakeShared<InputValueSumSource>(data_index, tensor.GetTensorDesc().GetDataType());
  GE_ASSERT_NOTNULL(value_source);
  auto symbol = shape_env_attr->CreateSymbol<T>(sum, value_source);
  result->emplace_back(symbol);
  GELOGI("data_index %d, symbolize value %s success, hint %lld, source %s", data_index, symbol.Str().get(), static_cast<int64_t>(sum),
         value_source->GetSourceStr().c_str());
  return result;
}

Status SymbolizeInputValueForRepeat(const GeTensor &tensor, SymbolicDescAttr *attr, ShapeEnvAttr *shape_env_attr,
                                    int32_t data_index) {
  GE_ASSERT_NOTNULL(attr);
  switch (tensor.GetTensorDesc().GetDataType()) {
    case DT_INT32:
      attr->symbolic_tensor.SetSymbolicValue(CreateSymbolValueSum<int32_t>(shape_env_attr, tensor, data_index));
      break;
    case DT_INT64:
      attr->symbolic_tensor.SetSymbolicValue(CreateSymbolValueSum<int64_t>(shape_env_attr, tensor, data_index));
      break;
    case DT_UINT32:
      attr->symbolic_tensor.SetSymbolicValue(CreateSymbolValueSum<uint32_t>(shape_env_attr, tensor, data_index));
      break;
    case DT_UINT64:
      attr->symbolic_tensor.SetSymbolicValue(CreateSymbolValueSum<uint64_t>(shape_env_attr, tensor, data_index));
      break;
    default:
      GELOGE(ge::PARAM_INVALID, "symbolic value generalize and compute not support data type %s",
             TypeUtils::DataTypeToSerialString(tensor.GetTensorDesc().GetDataType()).c_str());
      return FAILED;
  }
  GELOGI("Symbolize value success, %s",
         SymbolicInferUtil::VectorExpressionToStr(*attr->symbolic_tensor.GetSymbolicValue()).c_str());
  return GRAPH_SUCCESS;
}

bool SupportSymbolizeValueSum(const GeTensor &ge_tensor) {
 const auto& tensor_desc = ge_tensor.GetTensorDesc();
 if (tensor_desc.GetPlacement() != kPlacementHost) {
    GELOGI("tensor data is on %d, Current we do not support symbolize tensor data value which is not on host",
           static_cast<int32_t>(tensor_desc.GetPlacement()));
    return false;
  }
  if (!ge_tensor.GetData().IsTensorDataValid()) {
    GELOGI("tensor data is invalid, will not symbolize");
    return false;
  }
  if (kGeDType2CppDtype.find(tensor_desc.GetDataType()) == kGeDType2CppDtype.end()) {
    GELOGI("symbolic value generalize and compute not support data type %s",
           TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str());
    return false;
  }
  return true;
}

bool IsAippInput(const NodePtr &data_node) {
  auto output_nodes = NodeUtils::GetOutDataNodes(*data_node, nullptr);
  return output_nodes.size() == 1 && output_nodes[0]->GetType() == AIPP;
}

bool IsSupportSymbolize(const NodePtr &data_node) {
  // data是aipp算子的输入暂不支持泛化
  GE_WARN_ASSERT(!IsAippInput(data_node), "Data[%s] not support symbolize, output is aipp", data_node->GetNamePtr());
  return true;
}

/**
 * 获取编译期间用户输入的data节点
 *
 * 以下几种情况的data由ge构造需要排除：
 * 1、动态分档的data，由MultiBatchClonePass构造
 * 2、AippData类型的data，当图中有aipp算子的时候在Prepare阶段构造
 * 3、带有ref_var_src_var_name标记的RefData，由VariablePrepareOpPass构造
 * 4、aipp的输入由于会被ge修改原始shape，暂不支持泛化
 *
 * @param compute_graph 需要泛化的图
 * @param support_input_nodes 出参，保存支持的泛化的node节点
 * @param input_size 用户输入的Tensor个数，用来校验跟data的数量是否一致
 * @return 成功返回 SUCCESS，失败返回对应错误码
 */
Status GetSupportSymbolizeInputDataNodes(const ComputeGraphPtr &compute_graph,
  std::vector<NodePtr> &support_input_nodes, const size_t input_size) {
  // GetUserInputDataNodes接口已过滤动态分档插入的data
  std::vector<NodePtr> user_input_nodes;
  for (const auto &node : GraphUtilsEx::GetUserInputDataNodes(compute_graph)) {
    if (node->GetType() == AIPPDATA) {
      GELOGI("Node[%s] is aippdata, skip it.", node->GetNamePtr());
      continue;
    }
    if (AttrUtils::HasAttr(node->GetOpDesc(), REF_VAR_SRC_VAR_NAME)) {
      GELOGI("Node[%s] is copy refdata, skip it.", node->GetNamePtr());
      continue;
    }
    user_input_nodes.emplace_back(node);
  }
  GE_ASSERT_TRUE(user_input_nodes.size() == input_size, "data node number %zu not equal graph_inputs.size() %zu",
    user_input_nodes.size(), input_size);

  for (const auto &user_input_node : user_input_nodes) {
    if (IsSupportSymbolize(user_input_node)) {
      support_input_nodes.emplace_back(user_input_node);
    }
  }
  return SUCCESS;
}

// 动态分档子图泛化
Status SymbolizeMultiBatchSubGraph(const ComputeGraphPtr &graph) {
  // 1、判断是否是动态分档图
  bool enable_dynamic_batch = false;
  (void)AttrUtils::GetBool(graph, "_enable_dynamic_batch", enable_dynamic_batch);
  if (!GetLocalOmgContext().need_multi_batch || !enable_dynamic_batch) {
    return SUCCESS;
  }
  GELOGD("Start to symbolize multi batch graph[%s]", graph->GetName().c_str());
  auto case_node = graph->FindFirstNodeMatchType(CASE);
  GE_ASSERT_NOTNULL(case_node, "Graph[%s] is multi batch, but not have case node.", graph->GetName().c_str());
  bool is_mbatch_inserted_node = false;
  (void)AttrUtils::GetBool(case_node->GetOpDesc(), ATTR_INSERT_BY_MBATCH, is_mbatch_inserted_node);
  GE_ASSERT_TRUE(is_mbatch_inserted_node, "Graph[%s] is multi batch, but case_node[%s] not create by MultiBatchPass.",
    graph->GetName().c_str(), case_node->GetNamePtr());

  std::vector<ComputeGraphPtr> sub_graphs;
  NodeUtils::GetDirectSubgraphs(case_node, sub_graphs);
  for (const auto &sub_graph : sub_graphs) {
    for (const auto &node : sub_graph->GetDirectNode()) {
      if (!OpTypeUtils::IsDataNode(node->GetType())) {
        continue;
      }
      auto op_desc = node->GetOpDesc();
      GE_ASSERT_NOTNULL(op_desc);
      auto output_desc = op_desc->GetOutputDescPtr(0L);
      GE_ASSERT_NOTNULL(output_desc);
      auto &shape = output_desc->GetOriginShape();
      GE_ASSERT_TRUE(!shape.IsUnknownShape(), "SubGraph[%s] DataNode[%s] is dynamic shape.",
        sub_graph->GetName().c_str(), node->GetNamePtr());
      auto symbol_attr = op_desc->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>();
      GE_ASSERT_NOTNULL(symbol_attr);
      auto &origin_symbol_shape = symbol_attr->symbolic_tensor.MutableOriginSymbolShape().MutableDims();
      origin_symbol_shape.clear();
      for (size_t i = 0; i < shape.GetDimNum(); i++) {
        GuardDfxContext dfx_context("node name:" + op_desc->GetName() + " index:" + to_string(i));
        origin_symbol_shape.push_back(Symbol(shape.GetDim(i)));
      }
    }
  }
  return SUCCESS;
}

Status HandleUnknownDimNum(const GeShape& input_origin_shape, const OpDesc *op_desc, ShapeEnvAttr *shape_env_attr,
                           int32_t data_index, GeShape& ge_shape) {
  GELOGI("Start symbolize unknow rank, data node %s, index %d", op_desc->GetName().c_str(), data_index);
  GE_ASSERT_NOTNULL(shape_env_attr);
  const auto input_rank_source = MakeShared<InputRankSource>(data_index);
  GE_ASSERT_NOTNULL(input_rank_source);
  const auto rank = input_origin_shape.GetDimNum();
  const auto rank_symbol = shape_env_attr->CreateSymbol(rank, input_rank_source);
  EXPECT_SYMBOL_EQ(rank_symbol, Symbol(rank));     // 维度是否变化
  ge_shape.SetDimNum(rank); // SetDimNum可以初始化rank个-1
  GELOGI("Symbolize data node %s, index %d, symbol name %s, rank source str is %s.",
    op_desc->GetName().c_str(), data_index, SymbolicUtils::ToString(rank_symbol).c_str(),
    input_rank_source->GetSourceStr().c_str());
  return SUCCESS;
}

Status SymbolizeShape(const DataSymbolizeInfo &info, const OpDesc *op_desc, ShapeEnvAttr *shape_env_attr,
  SymbolicDescAttr *symbolic_desc_attr, GeShape &ge_shape) {
  const auto &input_origin_shape = info.inputShape;
  const auto &data_index = info.dataIndex;

  GE_ASSERT_TRUE(ge_shape.GetDimNum() == input_origin_shape.GetDimNum(),
    "The index %d shape dim num between Data node(%s)(%zu) and input tensor(%zu) are different",
    data_index, op_desc->GetName().c_str(), ge_shape.GetDimNum(), input_origin_shape.GetDimNum());

  GE_ASSERT_NOTNULL(symbolic_desc_attr);
  auto &origin_symbol_shape = symbolic_desc_attr->symbolic_tensor.MutableOriginSymbolShape().MutableDims();
  origin_symbol_shape.clear();
  for (size_t i = 0UL; i < ge_shape.GetDimNum(); ++i) {
    GuardDfxContext dfx_context("node name:" + op_desc->GetName() + " index:" + to_string(i));
    auto dim_value = ge_shape.GetDim(i);
    if (dim_value >= 0) {
      origin_symbol_shape.push_back(Symbol(dim_value));
      continue;
    }
    dim_value = input_origin_shape.GetDim(i);
    GE_ASSERT_TRUE(dim_value >= 0, "input origin dim value %lld is negative", dim_value);
    auto input_source = MakeShared<InputShapeSource>(data_index, i);
    Symbol symbol = shape_env_attr->CreateSymbol(dim_value, input_source);
    // 需要生成符号是否是0的guard，判断输入是否是空tensor
    EXPECT_SYMBOL_EQ(symbol, Symbol(0));
    origin_symbol_shape.emplace_back(symbol);
    GELOGI("Symbolize data node %s, index %zu, value %lld, symbol name %s, source str is %s",
      op_desc->GetName().c_str(), i, dim_value, symbol.GetName().get(), input_source->GetSourceStr().c_str());
  }
  return SUCCESS;
}

Status SymbolizeRootGraph(const ComputeGraphPtr &graph, const std::vector<GeTensor> &graph_inputs) {
  std::vector<NodePtr> data_nodes;
  GE_ASSERT_SUCCESS(GetSupportSymbolizeInputDataNodes(graph, data_nodes, graph_inputs.size()));
  auto shape_env_attr = graph->GetAttrsGroup<ShapeEnvAttr>();
  GE_ASSERT_NOTNULL(shape_env_attr);
  for (auto &data_node : data_nodes) {
    auto op_desc = data_node->GetOpDescBarePtr();
    DataSymbolizeInfo info;
    auto ret = BuildDataSymbolizeInfo(data_node, graph_inputs, info);
    if (ret == ge::UNSUPPORTED) {
      continue;
    }
    GE_ASSERT_SUCCESS(ret);
    const auto &data_index = info.dataIndex;
    const auto &input_origin_shape = info.inputShape;
    // 如果shape是[-2], 即不知道dims
    GeShape ge_shape;    
    if (info.dataShape.IsUnknownDimNum()) {
      GE_ASSERT_SUCCESS(HandleUnknownDimNum(input_origin_shape , op_desc, shape_env_attr,
        data_index, ge_shape), "symbolize unknown rank node %s failed", op_desc->GetName().c_str());
    } else {
      ge_shape = info.dataShape;
    }
    const auto symbolic_desc_attr = op_desc->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>();
    GE_ASSERT_SUCCESS(SymbolizeShape(info, op_desc, shape_env_attr, symbolic_desc_attr, ge_shape));

    int64_t symbolize_value_type = SYMBOLIZE_VALUE_TYPE_NONE;
    const auto &tensor = graph_inputs.at(data_index);
    if (AttrUtils::GetInt(op_desc, kSymbolizeValueType, symbolize_value_type) &&
      symbolize_value_type == static_cast<ino64_t>(SYMBOLIZE_VALUE_TYPE_SUM) &&
      SupportSymbolizeValueSum(tensor)) {
      GELOGI("Symbolize value sum for node %s[%s]", op_desc->GetNamePtr(), op_desc->GetTypePtr());
      GE_ASSERT_SUCCESS(SymbolizeInputValueForRepeat(tensor, symbolic_desc_attr, shape_env_attr, data_index));
    }
  }
  return SUCCESS;
}
}
std::string InputShapeSource::GetSourceStr() const {
  return R"([&]() -> int64_t {
      const auto *tensor = context->GetGraphInputTensor()" + std::to_string(input_data_idx_) + R"();
      if (tensor == nullptr) {
        return -1;
      }
      return tensor->GetOriginShape().GetDim()" + std::to_string(dim_idx_) + R"();
    }())";
}

std::string InputValueSumSource::GetSourceStr() const {
  return R"([&]() -> int64_t {
              const auto* tensor = context->GetGraphInputTensor()" + std::to_string(input_data_idx_) + R"();
                if (tensor == nullptr) {
                  return -1;
                }
                const auto* data = tensor->GetData<)" + kGeDType2CppDtype[dtype_] + R"(>();
                int64_t sum = 0;
                for (size_t i = 0; i < tensor->GetSize() / sizeof()" + kGeDType2CppDtype[dtype_] + R"(); ++i) {
                  sum += data[i];
                }
                return sum;
            }()
        )";
}

std::string InputRankSource::GetSourceStr() const {
  return R"([&]() -> size_t {
      const auto *tensor = context->GetGraphInputTensor()" + std::to_string(input_data_idx_) + R"();
      if (tensor == nullptr) {
        return -1;
      }
      return tensor->GetOriginShape().GetDimNum();
    }())";
}

Status SymbolicShapeSymbolizer::Symbolize(const ComputeGraphPtr &graph, const std::vector<GeTensor> &graph_inputs) {
  GELOGD("Start symbolize graph: %s", graph->GetName().c_str());
  // 对repeat算子特殊处理，Repeat算子需要对value的sum做symbolize处理，给repeat的data节点打上标签
  MarkSymbolizeRepeatInputValue(graph);
  if (graph->DeleteAttrsGroup<ShapeEnvAttr>()) {
    GELOGI("graph [%s] has ShapeEnv, do reset for symbolic shape symbolizer!", graph->GetName().c_str());
  }
  auto shape_env_attr = graph->CreateAttrsGroup<ShapeEnvAttr>();
  GE_ASSERT_NOTNULL(shape_env_attr);
  ShapeEnvGuarder guarder(shape_env_attr);
  GE_ASSERT_SUCCESS(SymbolizeRootGraph(graph, graph_inputs));
  GE_ASSERT_SUCCESS(SymbolizeMultiBatchSubGraph(graph));

  GELOGD("Graph: %s finish symbolize.", graph->GetName().c_str());
  return SUCCESS;
}
}  // namespace ge

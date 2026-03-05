/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowering_impl.h"
#include "common/checker.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "utils/autofuse_attrs.h"
#include "utils/autofuse_utils.h"
#include "utils/auto_fuse_config.h"
#include "lowering/asc_lowerer/asc_overrides.h"
#include "lowering/asc_lowerer/loop_common.h"
#include "lowering/op_helper/stridedslice.h"
#include "backend/backend_spec.h"
#include "can_fuse/backend/backend_utils.h"
#include "lowering/op_helper/cube.h"

namespace ge {
using namespace autofuse;
namespace {
constexpr int gather_mode_two = 2;
constexpr int gather_data_num_two = 2;
constexpr int gather_data_num_three = 3;

// 检查是否因为尾轴太小而跳过lowering
// 返回true表示应该跳过，false表示继续lowering
static bool ShouldSkipLoweringForSmallLastAxis(const NodePtr &node,
                                              const std::vector<Expression> &src_dims,
                                              const std::vector<size_t> &reduce_axes_u,
                                              const InDataAnchorPtr &reduce_anchor) {
  // 输入tensor的dim>1
  if (src_dims.size() <= 1) {
    return false;
  }

  const size_t last_axis_idx = src_dims.size() - 1;
  const size_t second_last_axis_idx = src_dims.size() - 2;

  // 检查尾轴和倒数第二个轴是否不同（即是否在reduce_axes中只有一个）
  bool last_axis_in_reduce = std::find(reduce_axes_u.begin(), reduce_axes_u.end(), last_axis_idx) != reduce_axes_u.end();
  bool second_last_axis_in_reduce = std::find(reduce_axes_u.begin(), reduce_axes_u.end(), second_last_axis_idx) != reduce_axes_u.end();
  // 当尾轴和倒数第二个轴不同时
  if (last_axis_in_reduce != second_last_axis_in_reduce) {
    // 获取尾轴的大小
    const auto &last_axis_expr = src_dims[last_axis_idx];
    // 判断是否为常量表达式
    if (!last_axis_expr.IsConstExpr()) {
      // 如果不是常量，无法判断，不跳过
      return false;
    }

    // 获取输入数据类型大小
    ge::DataType input_dtype = ge::DT_FLOAT;
    const auto input_op_desc = reduce_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc();
    if (input_op_desc != nullptr && input_op_desc->GetOutputDescPtr(0) != nullptr) {
      input_dtype = input_op_desc->GetOutputDescPtr(0)->GetDataType();
    }
    size_t dtype_size = ge::GetSizeByDataType(input_dtype);

    // 先用IsConstExpr判断是否为常量，再用GetConstValue获取值
    int64_t last_axis_size = 0;
    (void)last_axis_expr.GetConstValue(last_axis_size);

    // 判断尾轴大小乘以dtypesize < 64时，返回true（跳过lowering）
    if (last_axis_size > 0 && (static_cast<size_t>(last_axis_size) * dtype_size < 64ULL)) {
      GELOGI("Skip lowering node %s, as: Last axis size[%ld] * dtype_size[%zu] < 64.",
               node->GetNamePtr(), last_axis_size, dtype_size);
      return true;
    }
  }

  return false;
}

constexpr int transpose_two_perms = 2;
constexpr int transpose_three_perms = 3;
constexpr int transpose_four_perms = 4;
constexpr int transpose_five_perms = 5;

constexpr int fp32_help_min_exponent = -126;
constexpr int fp32_help_rec_one_exponent = 38;
constexpr int fp32_help_rec_sec_exponent = 44;
constexpr int fp16_help_min_exponent = -24;
constexpr int fp16_help_rec_one_exponent = 12;
constexpr int help_min_precision = 50;
constexpr double pow_base_two = 2.0;
constexpr size_t kMaxConcatDynInputNum = 32UL;
constexpr size_t kTileToConcatMaxMultiple = 6UL;
using Permutation = std::vector<int64_t>;
const std::set<Permutation> SUPPORTED_TWO_PERMS = {
    {1,0}
};
const std::set<Permutation> SUPPORTED_THREE_PERMS = {
    {0,2,1},
    {1,0,2},
    {1,2,0},
    {2,0,1},
    {2,1,0}
};
const std::set<Permutation> SUPPORTED_FOUR_PERMS = {
    {0,1,3,2},
    {0,2,1,3},
    {0,2,3,1},
    {0,3,1,2},
    {0,3,2,1},
    {1,0,2,3},
    {1,2,0,3},
    {1,2,3,0},
    {2,0,1,3},
    {2,1,0,3},
    {2,3,0,1},
    {2,3,1,0},
    {3,0,1,2},
    {3,1,2,0},
    {3,2,0,1}
};

graphStatus CollectConcatInputs(const NodePtr &node, int64_t concat_dim_tensor_index, int64_t concat_dim,
                                std::vector<InDataAnchorPtr> &inputs, size_t &dyn_input_num) {
  int64_t input_index = 0;
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    GE_ASSERT_NOTNULL(in_anchor);
    if ((concat_dim_tensor_index != -1L) && (input_index++ == concat_dim_tensor_index)) {
      continue;
    }
    std::vector<Expression> input_dims;
    GE_WARN_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(in_anchor, input_dims));
    for (int64_t i = concat_dim; i < static_cast<int64_t>(input_dims.size()); ++i) {
      if (!input_dims[i].IsConstExpr()) {
        GELOGD("%s inputs[%d] is dynamic: %s", node->GetNamePtr(), in_anchor->GetIdx(),
               ge::ToString(input_dims).c_str());
        ++dyn_input_num;
        break;
      }
    }
    inputs.emplace_back(in_anchor);
  }
  return GRAPH_SUCCESS;
}

bool ConcatCanBeConvertedToBrc(const std::vector<InDataAnchorPtr> &inputs, int64_t concat_dim) {
  std::set<const OutDataAnchor *> distinct_src_anchors;
  for (const auto &input : inputs) {
    distinct_src_anchors.emplace(input->GetPeerOutAnchor().get());
  }
  bool can_convert = false;
  std::vector<Expression> input_dims;
  if ((inputs.size() > 1UL) && (distinct_src_anchors.size() == 1UL)) {
    GE_WARN_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(inputs.front(), input_dims));
    const auto &concat_dim_expr = input_dims[static_cast<size_t>(concat_dim)];
    can_convert = (SymbolicUtils::StaticCheckEq(concat_dim_expr, Symbol(1)) == TriBool::kTrue);
  }
  GELOGD("distinct_src num = %zu, input_dims = %s, concat_dim = %ld, can_convert = %d", distinct_src_anchors.size(),
         ToString(input_dims).c_str(), concat_dim, static_cast<int32_t>(can_convert));
  return can_convert;
}

graphStatus ConcatToBroadcast(const NodePtr &node) {
  std::vector<loop::Index> indices;
  const auto x_anchor = node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(x_anchor);
  auto x = loop::Load(x_anchor);
  std::vector<ge::Expression> x_dims;
  GE_WARN_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(node->GetInDataAnchor(0), x_dims));
  std::vector<Expression> output_dims;
  GE_WARN_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(node->GetOutDataAnchor(0), output_dims));
  indices.emplace_back(output_dims);
  indices.emplace_back(x_dims);
  loop::Index broadcast;
  GE_ASSERT_GRAPH_SUCCESS(Broadcast(indices, broadcast));
  loop::Store(node->GetOutDataAnchor(0), loop::Broadcast(x, x_dims, broadcast));
  GELOGD("concat node: %s lowered to broadcast", node->GetNamePtr());
  (void) AttrUtils::SetBool(node->GetOpDesc(), "_disable_lifting", true);
  return GRAPH_SUCCESS;
}

graphStatus ParseConcatDim(const NodePtr &node, int64_t &concat_dim, int64_t &concat_dim_tensor_index) {
  if (!AttrUtils::GetInt(node->GetOpDesc(), "concat_dim", concat_dim)) {
    concat_dim_tensor_index = node->GetOpDesc()->GetInputIndexByName("concat_dim");
    GE_WARN_ASSERT(concat_dim_tensor_index >= 0,
                   "Skip lowering node %s, as: Concat_dim_tensor_index < 0.",
                   node->GetNamePtr());
    const auto op = ge::OpDescUtils::CreateOperatorFromNode(node);
    ge::Tensor concat_dim_tensor;
    GE_WARN_ASSERT(op.GetInputConstData("concat_dim", concat_dim_tensor) == ge::SUCCESS,
                   "Skip lowering node %s, as: Failed to get concat dim", node->GetNamePtr());

    const std::vector<int64_t> dims = concat_dim_tensor.GetTensorDesc().GetShape().GetDims();
    GE_WARN_ASSERT(dims.empty() || (dims.size() == 1U && dims[0] == 1U),
                   "Skip lowering node %s, as: Concat dim must be a scalar or [1]", node->GetNamePtr());
    GE_WARN_ASSERT(concat_dim_tensor.GetData() != nullptr,
                   "Skip lowering node %s, as: Concat_dim_tensor is null", node->GetNamePtr());

    const ge::DataType tensor_dtype = concat_dim_tensor.GetTensorDesc().GetDataType();
    if (tensor_dtype == ge::DT_INT32) {
      concat_dim = *reinterpret_cast<const int32_t *>(concat_dim_tensor.GetData());
    } else {
      GE_WARN_ASSERT(tensor_dtype == ge::DT_INT64,
                     "Skip lowering node %s, as: Concat dim tensor type must be int32 or int64",
                     node->GetNamePtr());
      concat_dim = *reinterpret_cast<const int64_t *>(concat_dim_tensor.GetData());
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus LowerConcat(const NodePtr &node) {
  constexpr size_t kMaxFreeSymbols = 16;
  int64_t concat_dim = 0;
  int64_t concat_dim_tensor_index = -1L;
  GE_WARN_ASSERT(ParseConcatDim(node, concat_dim, concat_dim_tensor_index) == GRAPH_SUCCESS);
  std::vector<Expression> output_dims;
  GE_WARN_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(node->GetOutDataAnchor(0), output_dims));
  concat_dim = concat_dim < 0 ? concat_dim + static_cast<int64_t>(output_dims.size()) : concat_dim;
  GE_WARN_ASSERT(concat_dim >= 0 && concat_dim < static_cast<int64_t>(output_dims.size()),
                 "Skip lowering node %s, as: Concat dim %ld must in dim range [0, %zu)",
                 node->GetNamePtr(), concat_dim, output_dims.size());
  GE_WARN_ASSERT(output_dims[concat_dim].FreeSymbols().size() <= kMaxFreeSymbols,
                 "Skip lowering node %s, as: Output concat dim has too many free symbols: %zu, exceeds max value: %zu",
                 node->GetNamePtr(), output_dims[concat_dim].FreeSymbols().size(), kMaxFreeSymbols);

  std::vector<InDataAnchorPtr> inputs;
  size_t dyn_input_num = 0;
  GE_WARN_ASSERT_GRAPH_SUCCESS(CollectConcatInputs(node, concat_dim_tensor_index, concat_dim, inputs, dyn_input_num));
  if (ConcatCanBeConvertedToBrc(inputs, concat_dim)) {
    GE_WARN_ASSERT_GRAPH_SUCCESS(ConcatToBroadcast(node));
    return GRAPH_SUCCESS;
  }
  const auto backend_spec = optimize::BackendSpec::GetInstance();
  GE_WARN_ASSERT(backend_spec != nullptr);
  // 每个动态shape的输入会单独分组，太多会导致编译时间过长，函数栈过大等问题，性能大概率也不好
  GE_WARN_ASSERT((dyn_input_num <= kMaxConcatDynInputNum) || (inputs.size() > backend_spec->concat_max_input_num),
                 "Skip lowering node %s, as too many dynamic inputs: %zu, exceeds max value: %zu", node->GetNamePtr(),
                 dyn_input_num, kMaxConcatDynInputNum);
  loop::StoreConcat(node->GetOutDataAnchor(0), inputs, static_cast<size_t>(concat_dim));
  return GRAPH_SUCCESS;
}

graphStatus GetMultiples(const NodePtr &node, std::vector<int64_t> &multiples) {
  if (AttrUtils::GetListInt(node->GetOpDesc(), "multiples", multiples)) {
    return GRAPH_SUCCESS;
  }
  const auto op = ge::OpDescUtils::CreateOperatorFromNode(node);
  ge::Tensor multiples_tensor;
  GE_WARN_ASSERT(op.GetInputConstData("multiples", multiples_tensor) == ge::SUCCESS,
                 "Skip lowering node %s, as: Failed to get multiples dim", node->GetNamePtr());

  const std::vector<int64_t> dims = multiples_tensor.GetTensorDesc().GetShape().GetDims();
  GE_WARN_ASSERT(dims.size() == 1U, "Skip lowering node %s, as: Multiples dims(%zu) is not 1.", node->GetNamePtr(),
                 dims.size());
  GE_WARN_ASSERT(multiples_tensor.GetData() != nullptr, "Skip lowering node %s, as: Multiples_tensor is null.",
                 node->GetNamePtr());

  const ge::DataType tensor_dtype = multiples_tensor.GetTensorDesc().GetDataType();
  for (int64_t i = 0L; i < dims[0]; ++i) {
    if (tensor_dtype == ge::DT_INT32) {
      auto casted_axis = reinterpret_cast<const int32_t *>(multiples_tensor.GetData() + i * sizeof(int32_t));
      GE_ASSERT_NOTNULL(casted_axis);
      int32_t axis = *casted_axis;
      multiples.emplace_back(axis);
    } else if (tensor_dtype == ge::DT_INT64) {
      auto casted_axis = reinterpret_cast<const int64_t *>(multiples_tensor.GetData() + i * sizeof(int64_t));
      GE_ASSERT_NOTNULL(casted_axis);
      int64_t axis = *casted_axis;
      multiples.emplace_back(axis);
    } else {
      GE_WARN_ASSERT(false, "Skip lowering node %s, as: Multiples must be int32 or int64", node->GetNamePtr());
    }
  }
  return GRAPH_SUCCESS;
}

bool CanLoweringToBrc(const loop::Index &in_dims, const std::vector<int64_t> &multiples, size_t new_axis_count,
                         std::vector<loop::BroadcastOp::DimKind> &brc_status, std::deque<size_t> &concat_dims) {
  bool can_lowering_to_brc = true;
  for (auto i = 0U; i < in_dims.size(); i++) {
    auto m_index = i + new_axis_count;
    GE_WARN_ASSERT(m_index < multiples.size(),
                   "Skip lowering Tile node, as: multiples index (%zu) out of range (size=%zu)", m_index,
                   multiples.size());
    if (multiples[m_index] != 1) {
      concat_dims.emplace_front(i);
      can_lowering_to_brc =
          can_lowering_to_brc && (SymbolicUtils::StaticCheckEq(in_dims[i], Symbol(1)) == TriBool::kTrue);
    }
    if (can_lowering_to_brc) {
      brc_status.emplace_back(multiples[m_index] == 1L ? loop::BroadcastOp::DimKind::NORMAL
                                                       : loop::BroadcastOp::DimKind::BROADCAST);
    }
  }
  return can_lowering_to_brc;
}

/**
 *  Tile/TileD算子Lowering函数
 *  1. 获取复制次数 -> multiples
 *  2. 获取输入Shape，若长度 ＜ multiples长度，则从0位置开始补1 -> in_dims
 *  3. 若in_dims长度小，则标记为补轴
 *  4.1 支持可以等效成Broadcast的场景，判断逻辑如下：
 *    统计每根轴是否需要广播：其余轴判断是否需要广播。支持的场景是：in_dims[i]或multiples[i]至少有1个为1。
 *    将Tile/TileD 算子 Lowering成Broadcast算子
 *  4.2  支持等效单个concat的场景
 */
graphStatus LowerTile(const NodePtr &node) {
  // 1. 获取 multiples
  std::vector<int64_t> multiples;
  GE_WARN_ASSERT(GetMultiples(node, multiples) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Get multiples param failed.", node->GetNamePtr());

  // 2. 获取输入Shape
  loop::Index in_dims;
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(0));
  auto input_src = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(input_src);
  GE_WARN_ASSERT(loop::GetBufferShape(input_src, in_dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Get input shape failed.", node->GetNamePtr());

  // 3. 如果in_dims长度小，则需要从前往后补轴
  GE_WARN_ASSERT(multiples.size() >= in_dims.size(),
                 "Skip lowering node %s, as: Not support multiples.size=%zu < in_dims.size=%zu", node->GetNamePtr(),
                 multiples.size(), in_dims.size());
  auto new_axis_count = multiples.size() - in_dims.size();
  std::vector<loop::BroadcastOp::DimKind> brc_status{new_axis_count, loop::BroadcastOp::DimKind::NEW_AXIS};
  std::deque<size_t> concat_dims;
  for (unsigned long i = 0; i < new_axis_count; ++i) {
    concat_dims.emplace_front(i);
  }

  // 获取输出Shape
  auto output_anchor = node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(output_anchor);
  // 4. 判断是否支持等效为广播或concat
  bool can_lowering_to_brc = CanLoweringToBrc(in_dims, multiples, new_axis_count, brc_status, concat_dims);
  // 支持Lowering成broadcast
  if (can_lowering_to_brc) {
    auto brc = loop::Broadcast(loop::Load(node->GetInDataAnchor(0)), brc_status);
    (void)loop::Store(output_anchor, brc);
    return GRAPH_SUCCESS;
  }

  // 支持Lowering成单个concat
  GE_WARN_ASSERT(concat_dims.size() == 1U, "Skip lowering node %s, as: Not support more than 1 axis to be tiled",
                 node->GetNamePtr());
  auto in_anchor = node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(in_anchor);
  auto dim = concat_dims[0];
  GE_ASSERT(dim + new_axis_count < multiples.size(),
            "Skip lowering node %s, as: multiples index (%zu) out of range (size=%zu)", node->GetNamePtr(),
            dim + new_axis_count, multiples.size());
  auto multiple = multiples[dim + new_axis_count];
  auto has_dyn_dim = std::any_of(in_dims.cbegin() + dim, in_dims.cend(),
                                 [](const ge::Expression &dim) -> bool { return (!dim.IsConstExpr()); });
  GE_WARN_ASSERT(((!has_dyn_dim) || (static_cast<size_t>(multiple) <= kTileToConcatMaxMultiple)),
           "Skip lowering node %s, as too many dynamic inputs: %ld, exceeds max value: %zu", node->GetNamePtr(),
           multiple, kTileToConcatMaxMultiple);
  vector<InDataAnchorPtr> inputs(multiple, in_anchor);
  loop::StoreConcat(output_anchor, inputs, dim);
  return GRAPH_SUCCESS;
}

graphStatus BroadCastByInDataAnchors(const vector<InDataAnchorPtr> &indata_anchors, std::vector<loop::Index> &indices, loop::Index &broadcasted) {
  for (auto &in_anchor : indata_anchors) {
    GE_ASSERT_NOTNULL(in_anchor);
    auto src = in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(src);
    auto desc = src->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(src->GetIdx());
    GE_ASSERT_NOTNULL(desc);
    auto sym_attr = desc->GetAttrsGroup<SymbolicDescAttr>();
    GE_WARN_ASSERT(sym_attr != nullptr, "No symbolic desc attr for node %s", in_anchor->GetOwnerNode()->GetName().c_str());
    indices.emplace_back(sym_attr->symbolic_tensor.GetOriginSymbolShape().GetDims());
  }
  GE_WARN_ASSERT_GRAPH_SUCCESS(Broadcast(indices, broadcasted));
  return GRAPH_SUCCESS;
}

loop::LoopVar ScalarBroadcast2Size(const std::string &face, ge::DataType dtype, size_t new_size) {
  auto rst = loop::Scalar(face, dtype);
  std::vector<loop::BroadcastOp::DimKind> status(new_size, loop::BroadcastOp::DimKind::NEW_AXIS);
  status.resize(new_size, loop::BroadcastOp::DimKind::NEW_AXIS);
  rst = loop::Broadcast(rst, status);
  return rst;
}

using TensorValue = std::variant<float, int8_t, int32_t, uint8_t, int16_t, uint16_t, uint32_t>;
template <typename Func>
string ProcessScalarTensor(const vector<ge::Tensor> &tensors, Func&& func) {
  GE_ASSERT(!tensors.empty());
  vector<TensorValue> scalar_tensors;
  const ge::DataType dtype = tensors[0].GetDataType();
  for (auto &tensor : tensors) {
    GE_ASSERT(tensor.GetDataType() == dtype);
    const void* data = tensor.GetData();
    switch (dtype) {
      case ge::DT_FLOAT:   scalar_tensors.emplace_back(*reinterpret_cast<const float*>(data)); break;
      case ge::DT_INT8:    scalar_tensors.emplace_back(*reinterpret_cast<const int8_t*>(data)); break;
      case ge::DT_INT32:   scalar_tensors.emplace_back(*reinterpret_cast<const int32_t*>(data)); break;
      case ge::DT_UINT8:   scalar_tensors.emplace_back(*reinterpret_cast<const uint8_t*>(data)); break;
      case ge::DT_INT16:   scalar_tensors.emplace_back(*reinterpret_cast<const int16_t*>(data)); break;
      case ge::DT_UINT16:  scalar_tensors.emplace_back(*reinterpret_cast<const uint16_t*>(data)); break;
      case ge::DT_UINT32:  scalar_tensors.emplace_back(*reinterpret_cast<const uint32_t*>(data)); break;
      default: return "";
    }
  }
  return func(scalar_tensors);
}

bool IsClose(float a, float b, float rel_tol = 1e-08, float abs_tol = 0.0) {
  return std::abs(a - b) <= std::max(rel_tol * std::max(std::abs(a), std::abs(b)), abs_tol);
}

template <typename T>
std::string ToPrecString(T x, int prec = 3) {
  std::ostringstream oss;
  oss.precision(prec);            // 设置小数点后位数
  oss  << std::fixed << x;
  return oss.str();
}

//  x <= 0 输出 0; x > 0 输出 1;
loop::LoopVar CalculateOneOrZero(const loop::LoopVar &input, DataType dtype, const std::vector<loop::BroadcastOp::DimKind> &status) {
  auto help_min = input;
  auto help_rec_one = input;
  auto help_rec_sec = input;
  if (dtype == DT_FLOAT) {
    help_min = loop::Scalar(
        ToPrecString(std::pow(pow_base_two, fp32_help_min_exponent), help_min_precision), dtype);
    help_rec_one = loop::Scalar(ToPrecString(std::pow(pow_base_two, fp32_help_rec_one_exponent)), dtype);
    help_rec_sec = loop::Scalar(ToPrecString(std::pow(pow_base_two, fp32_help_rec_sec_exponent)), dtype);
  } else if (dtype == DT_FLOAT16) {
    help_min = loop::Scalar(
        ToPrecString(std::pow(pow_base_two, fp16_help_min_exponent), help_min_precision), dtype);
    help_rec_one = loop::Scalar(ToPrecString(std::pow(pow_base_two, fp16_help_rec_one_exponent)), dtype);
    help_rec_sec = help_rec_one;
  } else if (dtype == DT_INT32) {
    help_min = loop::Scalar("1", dtype);
    help_rec_one = help_min;
    help_rec_sec = help_min;
  } else {
    GE_WARN_ASSERT(false, "Dtype is not supported to calculate");
  }
  help_min = loop::Broadcast(help_min, status);
  help_rec_one = loop::Broadcast(help_rec_one, status);
  help_rec_sec = loop::Broadcast(help_rec_sec, status);
  auto zero = loop::Broadcast(loop::Scalar("0", dtype), status);
  auto min_y = loop::Minimum(input, help_min);
  auto max_y = loop::Maximum(min_y, zero);
  auto result = loop::Mul(max_y, help_rec_one);
  if (dtype == DT_FLOAT) {
    result = loop::Mul(result, help_rec_sec);
  }
  result = loop::Mul(result, help_rec_sec);
  return result;
}

graphStatus ParseSplitNodeAndValidate(const NodePtr &split_node, InDataAnchorPtr &in_data_anchor, int64_t &split_dim, vector<ge::Expression> &x_dims) {
  const auto desc = split_node->GetOpDesc();
  GE_ASSERT_NOTNULL(desc);
  int32_t input_idx = desc->GetInputIndexByName("x");
  GE_ASSERT_TRUE(input_idx != -1);
  in_data_anchor = split_node->GetInDataAnchor(input_idx);
  GE_ASSERT_NOTNULL(in_data_anchor);
  GE_WARN_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(in_data_anchor, x_dims));
  GE_ASSERT_TRUE(!x_dims.empty());
  vector<int64_t> split_dim_list = {};
  GE_WARN_ASSERT_GRAPH_SUCCESS(AutofuseUtils::GetListIntByInputOrAttr(split_node, split_dim_list, "split_dim", "split_dim"),
                          "Skip lowering node %s, as: Failed to get split_dim.", split_node->GetNamePtr());
  GE_ASSERT_TRUE(!split_dim_list.empty());
  split_dim = split_dim_list[0];
  GE_ASSERT_TRUE((split_dim < static_cast<int64_t>(x_dims.size())) && (split_dim >= -static_cast<int64_t>(x_dims.size())),
                 "Split dim %ld must in rank range[-%zu, %zu).", split_dim, x_dims.size(), x_dims.size());
  if (split_dim < 0) {
    split_dim += static_cast<int64_t>(x_dims.size());
  }
  GE_ASSERT_NOTNULL(in_data_anchor);
  GE_ASSERT_TRUE((split_dim >= 0L) && (!x_dims.empty()) && (x_dims.size() > static_cast<size_t>(split_dim)));
  return GRAPH_SUCCESS;
}

graphStatus ComputeSplitSplits(const NodePtr &node, const Expression &x_dim, vector<Expression> &size_splits) {
  vector<int64_t> num_split_list = {};
  GE_ASSERT_GRAPH_SUCCESS(AutofuseUtils::GetListIntByInputOrAttr(node, num_split_list, "num_split", "num_split"),
                 "Skip lowering node %s, as: Failed to get num_split.", node->GetNamePtr());
  GE_ASSERT(!num_split_list.empty());

  vector<int64_t> size_splits_list = {};
  if (node->GetType() == AF_SPLITVD || node->GetType() == AF_SPLITV) {
    GE_ASSERT_GRAPH_SUCCESS(AutofuseUtils::GetListIntByInputOrAttr(node, size_splits_list, "size_splits", "size_splits"),
                   "Skip lowering node %s, as: Failed to get size_splits .", node->GetNamePtr());
    GE_ASSERT(!size_splits_list.empty());
    for (size_t i = 0U; i < size_splits_list.size(); ++i) {
      size_splits.push_back(Symbol(size_splits_list[i]));
    }
  } else {
    GE_ASSERT_TRUE(num_split_list[0] > 0L);
    size_splits.assign(num_split_list[0], x_dim / Symbol(num_split_list[0]));
  }
  return GRAPH_SUCCESS;
}

bool CheckEndDimHasOne(const NodePtr &node) {
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    std::vector<Expression> output_dims;
    GE_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(out_anchor, output_dims));
    if (!output_dims.empty()) {
      auto index = output_dims.size() - 1;
      if (SymbolicUtils::StaticCheckEq(output_dims[index], Symbol(1)) == TriBool::kTrue) {
        GELOGI("output end dim has 1, not lowering.");
        return true;
      }
    }
  }
  return false;
}

graphStatus LowerSplitToStridedSlices(const NodePtr &node, const vector<Expression> &x_dims,
                                     int64_t &split_dim, const InDataAnchorPtr &x_anchor) {
  GELOGI("lowering split node %s as StridedSlices", node->GetName().c_str());
  vector<Expression> start;
  vector<Expression> stride;
  vector<Expression> size_splits;
  for (size_t i = 0; i < x_dims.size(); ++i) {
    stride.push_back(Symbol(1));
    start.push_back(Symbol(0));
  }
  if (split_dim < 0) {
    split_dim += static_cast<int64_t>(x_dims.size());
  }
  GE_ASSERT_TRUE(split_dim >= 0L);
  GE_WARN_ASSERT_GRAPH_SUCCESS(ComputeSplitSplits(node, x_dims[split_dim], size_splits));

  if (CheckEndDimHasOne(node)) {
    return GRAPH_FAILED;
  }
  Expression offset = Symbol(0);
  int32_t index = 0;
  string not_lowering_reason;
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    start[split_dim] = offset;
    auto split_lowering = loop::StoreStridedSlice(out_anchor, x_anchor, start, stride, x_dims, not_lowering_reason);
    if (!not_lowering_reason.empty()) {
      GELOGI("Skip lowering node %s, as: %s", node->GetNamePtr(), not_lowering_reason.c_str());
      loop::StoreExtern(out_anchor);
      continue;
    }
    loop::Store(out_anchor, split_lowering);
    offset = offset + size_splits[index++];
  }
  return GRAPH_SUCCESS;
}

graphStatus LowerSplitToNormalSplit(const NodePtr &node, std::vector<ge::Expression> &x_dims, int64_t split_dim, const InDataAnchorPtr &x_anchor) {
  GELOGI("split node %s do normal split lowering.", node->GetName().c_str());

  if ((node->GetAllOutDataAnchorsSize() > loop::StoreSplitOp::kSplitCanFuseMaxOutput)
      || ((x_dims.size() != 1U) && (static_cast<uint32_t>(split_dim) == (x_dims.size() - 1))
          && (CheckEndDimHasOne(node) == true) && (node->GetAllOutDataAnchorsSize() > loop::StoreSplitOp::kSplitCanLowerEndDimMaxOutput))) {
    GELOGI("Skip lowering node %s(%s), as: split node %s has output size %zu > %zu",
           node->GetName().c_str(), node->GetType().c_str(), node->GetName().c_str(), node->GetAllOutDataAnchorsSize(),
           loop::StoreSplitOp::kSplitCanFuseMaxOutput);
    return GRAPH_FAILED;
  }

  vector<OutDataAnchorPtr> outputs;
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    outputs.emplace_back(out_anchor);
  }

  string not_lowering_reason;
  auto split = loop::StoreSplit(outputs, x_anchor, static_cast<size_t>(split_dim), not_lowering_reason);
  if (!not_lowering_reason.empty()) {
    GELOGI("Skip lowering node %s, as: %s", node->GetNamePtr(), not_lowering_reason.c_str());
    loop::StoreExtern(node->GetOutDataAnchor(0));
  }

  return GRAPH_SUCCESS;
}
}  // namespace


graphStatus Broadcast(const std::vector<loop::Index> &indices, loop::Index &broadcasted) {
  GE_ASSERT(!indices.empty());
  size_t max_rank = indices[0].size();
  for (size_t i = 1U; i < indices.size(); ++i) {
    max_rank = std::max(max_rank, indices[i].size());
  }
  broadcasted.clear();
  broadcasted.resize(max_rank, Symbol(1));
  for (size_t current_dim = max_rank; current_dim > 0U; --current_dim) {
    size_t idx = current_dim - 1U;
    for (const auto &index : indices) {
      if (idx + index.size() < max_rank) {
        continue;
      }
      size_t current = idx - (max_rank - index.size());
      if (index.size() <= current || index[current] == broadcasted[idx]) {
        continue;
      }
      broadcasted[idx] = index[current];
      break;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus LowerPointwise(const NodePtr &node,
                           const std::function<loop::LoopVar(const std::vector<loop::LoopVar> &)> &kernel) {
  std::vector<loop::Index> indices;
  std::vector<ge::DataType> dtypes;
  for (auto &in_anchor : node->GetAllInDataAnchors()) {
    auto src = in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(src);
    auto desc = src->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(src->GetIdx());
    GE_ASSERT_NOTNULL(desc);
    auto sym_attr = desc->GetAttrsGroup<SymbolicDescAttr>();
    GE_WARN_ASSERT(sym_attr != nullptr, "Skip lowering node %s, as: No symbolic desc attr.",
                   node->GetName().c_str());
    indices.emplace_back(sym_attr->symbolic_tensor.GetOriginSymbolShape().GetDims());
    dtypes.emplace_back(desc->GetDataType());
  }

  std::vector<loop::LoopVar> vars;
  if (dtypes.size() > 1U) {
    loop::Index broadcasted;
    GE_ASSERT_GRAPH_SUCCESS(Broadcast(indices, broadcasted));
    for (int32_t i = 0; i < static_cast<int32_t>(dtypes.size()); ++i) {
      vars.emplace_back(loop::Broadcast(loop::Load(node->GetInDataAnchor(i)), indices[i], broadcasted));
    }
  } else if (dtypes.size() == 1U) {
    vars.emplace_back(loop::Load(node->GetInDataAnchor(0)));
  }

  auto out_anchor = node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(out_anchor);

  (void)loop::Store(out_anchor, kernel(vars));

  return GRAPH_SUCCESS;
}

graphStatus GetReduceDims(const NodePtr &node, std::vector<int64_t> &reduce_axes, const std::string &input = "axes",
                          const std::string &attr = "axes") {
  size_t num_inputs = node->GetAllInDataAnchorsSize();
  if (num_inputs == 1U) {
    GE_WARN_ASSERT(AttrUtils::GetListInt(node->GetOpDesc(), attr, reduce_axes));
    return GRAPH_SUCCESS;
  }
  GE_ASSERT(num_inputs == 2U);
  const auto op = ge::OpDescUtils::CreateOperatorFromNode(node);
  ge::Tensor axes_tensor;
  if (op.GetInputConstData(input.c_str(), axes_tensor) != ge::SUCCESS) {
    GELOGI("Force skip lowering node %s %s as dynamic axes", node->GetNamePtr(), node->GetTypePtr());
    return GRAPH_FAILED;
  }

  const std::vector<int64_t> dims = axes_tensor.GetTensorDesc().GetShape().GetDims();
  GE_WARN_ASSERT(dims.size() <= 1U, "Reduce axis must be a scalar or vector");
  const int64_t num_axes = dims.empty() ? 1 : dims[0];
  GE_WARN_ASSERT(num_axes >= 0, "Reduce axis must be positive");
  GE_WARN_ASSERT(axes_tensor.GetData() != nullptr);

  ge::DataType tensor_dtype = axes_tensor.GetTensorDesc().GetDataType();
  for (int64_t i = 0; i < num_axes; ++i) {
    if (tensor_dtype == ge::DT_INT32) {
      int32_t axis = *reinterpret_cast<const int32_t *>(axes_tensor.GetData() + i * sizeof(int32_t));
      reduce_axes.emplace_back(axis);
    } else if (tensor_dtype == ge::DT_INT64) {
      int64_t axis = *reinterpret_cast<const int64_t *>(axes_tensor.GetData() + i * sizeof(int64_t));
      reduce_axes.emplace_back(axis);
    } else {
      GE_WARN_ASSERT(false, "Reduce axis must be int32 or int64");
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus LowerReduction(const NodePtr &node, loop::ReduceType reduce_type) {
  std::vector<int64_t> reduce_axes;
  string input = "axes";
  if (node->GetAllInDataAnchorsSize() == 2U) {
    auto op_desc = node->GetOpDesc();
    GE_ASSERT(op_desc != nullptr);
    input = op_desc->GetInputNameByIndex(1U);
  }
  GE_WARN_ASSERT(GetReduceDims(node, reduce_axes, input) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get reduce_axes.", node->GetName().c_str());
  const auto reduce_anchor = node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(reduce_anchor);
  std::vector<Expression> src_dims;
  GE_WARN_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(reduce_anchor, src_dims),
                               "Skip lowering node %s, as: Input %s has no sym shape", node->GetName().c_str(),
                               loop::BufferName(reduce_anchor).c_str());
  const auto out_anchor = node->GetOutDataAnchor(0);
  std::vector<Expression> dst_dims;
  GE_WARN_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(out_anchor, dst_dims),
                               "Skip lowering node %s, as: Output %s no sym shape",
                               node->GetName().c_str(), loop::BufferName(out_anchor).c_str());
  vector<size_t> reduce_axes_u;
  for (auto reduce_dim : reduce_axes) {
    if (reduce_dim < 0) {
      GE_WARN_ASSERT(reduce_dim + static_cast<int64_t>(src_dims.size()) >= 0,
                     "Skip lowering node %s, as: Reduce axis %ld over rank %zu",
                     node->GetNamePtr(), reduce_dim, src_dims.size());
      reduce_dim += static_cast<int64_t>(src_dims.size());
    }
    GE_WARN_ASSERT(reduce_dim >= 0 && static_cast<size_t>(reduce_dim) < src_dims.size(),
                   "Skip lowering node %s, as: Reduce axis %ld must >=0 and within rank %zu",
                   node->GetNamePtr(), reduce_dim, src_dims.size());
    reduce_axes_u.emplace_back(static_cast<size_t>(reduce_dim));
  }

  // 检查是否需要跳过lowering：当输入tensor的dim>1，并且尾轴和倒数第二个轴不同时为A轴或者R轴
  // 判断尾轴大小乘以dtypesize < 64时，返回true（跳过lowering）
  GE_WARN_ASSERT(!ShouldSkipLoweringForSmallLastAxis(node, src_dims, reduce_axes_u, reduce_anchor),
                 "Skip lowering node %s, as: Last axis is too small.",
                 node->GetNamePtr());

  (void)loop::StoreReduction(reduce_type, out_anchor, loop::Load(reduce_anchor), src_dims, reduce_axes_u);
  return GRAPH_SUCCESS;
}

graphStatus LowerSlice(const NodePtr &node) {
  const auto x_anchor = node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(x_anchor);
  std::vector<ge::Expression> x_dims;
  GE_WARN_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(x_anchor, x_dims));
  GE_ASSERT(!x_dims.empty());
  vector<int64_t> offsets = {};
  GE_WARN_ASSERT(AutofuseUtils::GetListIntByInputOrAttr(node, offsets, "offsets", "offsets") == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get offsets.", node->GetNamePtr());
  GE_ASSERT(!offsets.empty());
  vector<Expression> start;
  vector<Expression> stride;
  for (const auto &offset : offsets) {
    start.push_back(Symbol(offset));
    stride.push_back(Symbol(1));
  }

  string not_lowering_reason;
  auto slice = loop::StoreStridedSlice(node->GetOutDataAnchor(0), x_anchor, start, stride, x_dims, not_lowering_reason);
  if (!not_lowering_reason.empty()) {
    GELOGI("Skip lowering node %s, as: %s", node->GetNamePtr(), not_lowering_reason.c_str());
    loop::StoreExtern(node->GetOutDataAnchor(0));
    return GRAPH_SUCCESS;
  }
  loop::Store(node->GetOutDataAnchor(0), slice);
  return GRAPH_SUCCESS;
}

graphStatus LowerStridedSlice(const NodePtr &node) {
  const auto x_anchor = node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(x_anchor);
  std::vector<ge::Expression> x_dims;
  GE_WARN_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(x_anchor, x_dims));
  GE_ASSERT(!x_dims.empty());
  vector<int64_t> begin_list = {};
  vector<int64_t> stride_list = {};
  GE_WARN_ASSERT(AutofuseUtils::GetListIntByInputOrAttr(node, begin_list, "begin", "begin") == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get begin.", node->GetNamePtr());
  GE_ASSERT(!begin_list.empty());
  GE_WARN_ASSERT(AutofuseUtils::GetListIntByInputOrAttr(node, stride_list, "strides", "strides") == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get strides.", node->GetNamePtr());
  GE_ASSERT(!stride_list.empty());
  vector<Expression> start;
  vector<Expression> stride;
  StridedSliceMaskAttr mask_attr;
  StrdedSliceIndexInputs index_input;
  GE_ASSERT_TRUE(AttrUtils::GetInt(node->GetOpDesc(), "begin_mask", mask_attr.begin_mask));
  GE_ASSERT_TRUE(AttrUtils::GetInt(node->GetOpDesc(), "end_mask", mask_attr.end_mask));
  GE_ASSERT_TRUE(AttrUtils::GetInt(node->GetOpDesc(), "ellipsis_mask", mask_attr.ellipsis_mask));
  GE_ASSERT_TRUE(AttrUtils::GetInt(node->GetOpDesc(), "new_axis_mask", mask_attr.new_axis_mask));
  GE_ASSERT_TRUE(AttrUtils::GetInt(node->GetOpDesc(), "shrink_axis_mask", mask_attr.shrink_axis_mask));

  auto stride_list_end_index = stride_list.size() - 1;
  GE_WARN_ASSERT(stride_list[stride_list_end_index] == 1,
                 "Skip lowering node %s, as: The end stride is %ld, it must be equal to one.",
                 node->GetName().c_str(), stride_list[stride_list_end_index]);

  for (size_t i = 0; i < begin_list.size(); ++i) {
    GE_WARN_ASSERT(stride_list[i] >= 0,
                   "Skip lowering node %s, as: The value stride is %ld, it must be greater than or equal to zero.",
                   node->GetName().c_str(), stride_list[i]);
    index_input.start_indexes.push_back(begin_list[i] >= 0 ? Symbol(begin_list[i]) : Symbol(begin_list[i]) + x_dims[i]);
    index_input.strides_indexes.push_back(Symbol(stride_list[i]));
  }
  GE_WARN_ASSERT_GRAPH_SUCCESS(InferShapeStridedSlice(x_dims, mask_attr, index_input));
  string not_lowering_reason;
  auto stride_slice = loop::StoreStridedSlice(node->GetOutDataAnchor(0), x_anchor, index_input.start_indexes,
                                              index_input.strides_indexes, x_dims, not_lowering_reason);
  if (!not_lowering_reason.empty()) {
    GELOGI("Skip lowering node %s, as: %s", node->GetNamePtr(), not_lowering_reason.c_str());
    loop::StoreExtern(node->GetOutDataAnchor(0));
    return GRAPH_SUCCESS;
  }
  loop::Store(node->GetOutDataAnchor(0), stride_slice);
  return GRAPH_SUCCESS;
}

graphStatus LowerSplit(const NodePtr &node) {
  GELOGI("LowerSplit start lowering node %s.", node->GetNamePtr());
  string not_lowering_reason;
  int64_t split_dim = 0L;
  InDataAnchorPtr x_anchor;
  vector<ge::Expression> x_dims;
  GE_WARN_ASSERT_GRAPH_SUCCESS(ParseSplitNodeAndValidate(node, x_anchor, split_dim, x_dims));
  const auto backend_spec = optimize::BackendSpec::GetInstance();
  if (split_dim < 0) {
    split_dim += static_cast<int64_t>(x_dims.size());
  }

  if (!backend_spec->slice_split_spec.split_lowered_to_split) {
    return LowerSplitToStridedSlices(node, x_dims, split_dim, x_anchor);
  }
  return LowerSplitToNormalSplit(node, x_dims, split_dim, x_anchor);
}

graphStatus LowerGather(const NodePtr &node) {
  const ge::OpDescPtr gather_op_desc = node->GetOpDesc();
  int64_t op_impl_mode = -1L;
  int64_t batch_dims = -1L;
  int64_t input_nums = 0L;
  ge::AttrUtils::GetInt(gather_op_desc, "_op_impl_mode_enum", op_impl_mode);
  ge::AttrUtils::GetInt(gather_op_desc, "batch_dims", batch_dims);
  GE_WARN_ASSERT(op_impl_mode != gather_mode_two, "Skip lowering node %s, as: Gather is in high_performance mode",
                 node->GetNamePtr());
  GE_WARN_ASSERT(batch_dims == 0, "Skip lowering node %s, as: Batch dims is not 0", node->GetNamePtr());
  std::vector<Expression> dims;
  for (auto &anchor : node->GetAllInDataAnchors()) {
    if (anchor == nullptr || anchor->GetPeerOutAnchor() == nullptr) {
      continue;
    }
    input_nums++;
    loop::GetKernelBox(anchor->GetPeerOutAnchor()).Realize();
  }

  if (input_nums == gather_data_num_two) {  // lowering for Gather
    auto loop_var = loop::GatherLoad(node->GetOutDataAnchor(0), node->GetInDataAnchor(0), node->GetInDataAnchor(1), 0);
    (void)loop::Store(node->GetOutDataAnchor(0), loop_var).Realize();
  } else if (input_nums == gather_data_num_three) {  // lowering for GatherV2
    auto node_axis = node->GetInDataAnchor(2)->GetPeerOutAnchor();
    auto node_type = node_axis->GetOwnerNode()->GetOpDesc()->GetType();
    GE_WARN_ASSERT(((node_axis != nullptr) && ((node_type == CONSTANT || node_type == CONSTANTOP))),
                   "Skip lowering node %s, as: Axis is not a CONSTANT input", node->GetNamePtr());

    vector<int64_t> axis = {};
    GE_WARN_ASSERT(AutofuseUtils::GetListIntByInputOrAttr(node, axis, "axis", "axis") == GRAPH_SUCCESS,
                   "Skip lowering node %s, as: Get axis failed", node->GetNamePtr());
    GE_ASSERT(!axis.empty());

    std::vector<Expression> params_dims;
    GE_WARN_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(node->GetInDataAnchor(0), params_dims));
    axis[0] = axis[0] < 0 ? axis[0] + static_cast<int64_t>(params_dims.size()) : axis[0];
    GE_WARN_ASSERT(axis[0] >= 0 && axis[0] < static_cast<int64_t>(params_dims.size()),
                   "Skip lowering node %s, as: Axis %ld must in dim range [0, %zu)", node->GetNamePtr(),
                   axis[0], params_dims.size());
    const auto backend_spec = optimize::BackendSpec::GetInstance();
    GE_CHECK_NOTNULL(backend_spec);
    GE_WARN_ASSERT(backend_spec->gather_spec.enable_non_tail_gather ||
      axis[0] + 1 == static_cast<int64_t>(params_dims.size()),
             "Skip lowering node %s, as: Is not last dim gather", node->GetNamePtr());
    auto loop_var =
        loop::GatherLoad(node->GetOutDataAnchor(0), node->GetInDataAnchor(0), node->GetInDataAnchor(1), axis[0]);
    (void)loop::Store(node->GetOutDataAnchor(0), loop_var).Realize();
  }
  return GRAPH_SUCCESS;
}

bool IsPermSupported(const Permutation &perm, uint32_t transpose_mode) {
  if (transpose_mode == static_cast<uint32_t>(optimize::TransposeMode::TRANSPOSE_MODE_UNNORMAL)) { // 1:非normal模式
    // 先直接判断是否小于等于5维，后续添加合轴后的维度判断
    return perm.size() <= transpose_five_perms;
  }
  switch (perm.size()) {
    case transpose_two_perms:
      return SUPPORTED_TWO_PERMS.find(perm) != SUPPORTED_TWO_PERMS.end();
    case transpose_three_perms:
      return SUPPORTED_THREE_PERMS.find(perm) != SUPPORTED_THREE_PERMS.end();
    case transpose_four_perms:
      return SUPPORTED_FOUR_PERMS.find(perm) != SUPPORTED_FOUR_PERMS.end();
    default:
      return false;
  }
}

graphStatus GetTransposePerms(const NodePtr &node, std::vector<int64_t> &perm) {
  if (!AttrUtils::GetListInt(node->GetOpDesc(), "perm", perm)) {
    const auto op = ge::OpDescUtils::CreateOperatorFromNode(node);
    ge::Tensor perm_tensor;
    GE_WARN_ASSERT(op.GetInputConstData("perm", perm_tensor) == ge::SUCCESS,
                   "Skip lowering node %s, as: Failed to get perm dim", node->GetNamePtr());

    const std::vector<int64_t> dims = perm_tensor.GetTensorDesc().GetShape().GetDims();
    GE_WARN_ASSERT(dims.size() == 1U,
                   "Skip lowering node %s, as: Perm dims(%zu) is not 1.", node->GetNamePtr(), dims.size());
    GE_WARN_ASSERT(perm_tensor.GetData() != nullptr,
                   "Skip lowering node %s, as: Perm dims value is null.", node->GetNamePtr());

    const ge::DataType tensor_dtype = perm_tensor.GetTensorDesc().GetDataType();
    for (int64_t i = 0L; i < dims[0]; ++i) {
      if (tensor_dtype == ge::DT_INT32) {
        int32_t axis = *reinterpret_cast<const int32_t *>(perm_tensor.GetData() + i * sizeof(int32_t));
        perm.emplace_back(axis);
      } else if (tensor_dtype == ge::DT_INT64) {
        int64_t axis = *reinterpret_cast<const int64_t *>(perm_tensor.GetData() + i * sizeof(int64_t));
        perm.emplace_back(axis);
      } else {
        GE_WARN_ASSERT(false, "Skip lowering node %s, as: Multiples must be int32 or int64", node->GetNamePtr());
      }
    }
  }
  return GRAPH_SUCCESS;
}


graphStatus LowerTranspose(const NodePtr &node) {
  //暂时添加环境变量控制是否启用
  if (!ge::AutoFuseConfig::LoweringConfig().experimental_lowering_transpose) {
    GELOGI(
        "Skip lowering node %s, you can enable it by setting AUTOFUSE_FLAGS=\"--autofuse_enable_pass=transpose\""
        " and unsetting AUTOFUSE_FLAGS=\"--autofuse_disable_pass=transpose\"",
        node->GetNamePtr());
    return GRAPH_FAILED;
  }

  const auto x_anchor = node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(x_anchor);
  auto src = x_anchor->GetPeerOutAnchor();
  if (src != nullptr) {
    loop::GetKernelBox(src).Realize();
  }
  std::vector<ge::Expression> dims;
  GE_WARN_ASSERT(loop::GetBufferShape(src, dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input shape.", node->GetNamePtr());
  auto desc = src->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(src->GetIdx());
  const ge::DataType input_dtype = desc->GetDataType();
  if (input_dtype != DT_FLOAT16 && input_dtype != DT_FLOAT) {
    GELOGI("Skip lowering node %s, as input dtype is not float32 or float16, not lowering.", node->GetNamePtr());
    return GRAPH_FAILED;
  }

  auto x = loop::Load(node->GetInDataAnchor(0));
  std::vector<int64_t> perm;
  auto ret = GetTransposePerms(node, perm);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }

  const auto backend_spec = optimize::BackendSpec::GetInstance();
  GE_CHECK_NOTNULL(backend_spec);
  uint32_t transpose_mode = backend_spec->transpose_mode;
  auto transpose_can_lower = IsPermSupported(perm, transpose_mode);
  if (transpose_can_lower != true) {
    GELOGI("dims or perm is not supported, not lowering.");
    return GRAPH_FAILED;
  }
  x = loop::Transpose(x, dims, perm);
  loop::Store(node->GetOutDataAnchor(0), x).Realize();
  return GRAPH_SUCCESS;
}

graphStatus TryLowerBatchMatMulToVector(const NodePtr &node, const MatMulAttr &matmul_attr) {
  const auto x_anchor = node->GetInDataAnchor(0);
  const auto y_anchor = node->GetInDataAnchor(1);
  GE_ASSERT_NOTNULL(x_anchor);
  GE_ASSERT_NOTNULL(y_anchor);
  std::vector<ge::Expression> x_dims;
  GE_WARN_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(x_anchor, x_dims));
  GE_WARN_ASSERT(x_dims.size() == 3U, "Skip lowering node %s, as: BatchMatMulV2 input x must be 3D",
                 node->GetName().c_str());
  std::vector<ge::Expression> y_dims;
  GE_WARN_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(y_anchor, y_dims));
  GE_WARN_ASSERT(y_dims.size() == 3U, "Skip lowering node %s, as: BatchMatMulV2 input y must be 3D",
                 node->GetName().c_str());
  GE_WARN_ASSERT(node->GetInDataNodesAndAnchors().size() == 2U,
                 "Skip lowering node %s, as: Has optional inputs",
                 node->GetName().c_str());

  bool transpose_a = matmul_attr.adj_x1 || matmul_attr.transpose_x1;
  bool transpose_b = matmul_attr.adj_x2 || matmul_attr.transpose_x2;
  GE_WARN_ASSERT(!transpose_a, "Skip lowering node %s, as: Input x1 transposed", node->GetName().c_str());
  int32_t offset = static_cast<int32_t>(matmul_attr.offset_x);
  GE_WARN_ASSERT(offset == 0, "Skip lowering node %s, as: Offset %d not 0", node->GetName().c_str(), offset);
  y_dims = transpose_b ? std::vector<ge::Expression>{y_dims[0], y_dims[2], y_dims[1]} : y_dims;
  const auto &batch_size = x_dims[0];
  const auto &m_size = x_dims[1];
  const auto &k_size = y_dims[1];
  const auto &n_size = y_dims[2];
  GE_WARN_ASSERT(EXPECT_SYMBOL_EQ(Symbol(1), n_size), "Skip lowering node %s, as: n %s is not 1",
                 node->GetName().c_str(), n_size.Str().get());
  int64_t const_k = 0U;
  GE_WARN_ASSERT(k_size.GetConstValue(const_k), "Skip lowering node %s, as: k %s not const",
                 node->GetName().c_str(), k_size.Str().get());
  const int64_t max_k = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_WARN_ASSERT(const_k <= max_k, "Skip lowering node %s, as: k %ld > %ld",
                 node->GetName().c_str(), const_k, max_k);

  // [BS, M, K] * [BS, K, 1] -> [BS, M, 1]
  auto x = loop::Load(x_anchor);
  auto y = loop::Load(y_anchor);
  y = transpose_b ? y : loop::Permute(y, {0, 2, 1});  // [BS, K, 1] -> [BS, 1, K]
  // [BS, 1, K] -> [BS, M, K]
  y = loop::Broadcast(y, {batch_size, Symbol(1), k_size}, {batch_size, m_size, k_size});
  const auto z = loop::Mul(x, y);  // [BS, M, K] * [BS, M, K] -> [BS, M, K]
  if (SymbolicUtils::StaticCheckEq(k_size, Symbol(1)) == TriBool::kTrue) {
    loop::Store(node->GetOutDataAnchor(0), z);
    return GRAPH_SUCCESS;
  }
  loop::StoreReduction(loop::ReduceType::SUM, node->GetOutDataAnchor(0), z, {batch_size, m_size, k_size},
                       {2});  // [BS, M, K] -> [BS, M, 1]
  return GRAPH_SUCCESS;
}

graphStatus TryLowerMatMulToVector(const NodePtr &node, const MatMulAttr &matmul_attr) {
  const auto x_anchor = node->GetInDataAnchor(0);
  const auto y_anchor = node->GetInDataAnchor(1);
  GE_ASSERT_NOTNULL(x_anchor);
  GE_ASSERT_NOTNULL(y_anchor);
  std::vector<ge::Expression> x_dims;
  GE_WARN_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(x_anchor, x_dims));
  GE_WARN_ASSERT(x_dims.size() == 2U, "Skip lowering node %s, as: BatchMatMulV2 input x must be 3D",
                 node->GetName().c_str());
  std::vector<ge::Expression> y_dims;
  GE_WARN_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(y_anchor, y_dims));
  GE_WARN_ASSERT(y_dims.size() == 2U, "Skip lowering node %s, as: BatchMatMulV2 input y must be 3D",
                 node->GetName().c_str());
  GE_WARN_ASSERT(node->GetInDataNodesAndAnchors().size() == 2U,
                 "Skip lowering node %s, as: Has optional inputs",
                 node->GetName().c_str());

  bool transpose_a = matmul_attr.adj_x1 || matmul_attr.transpose_x1;
  bool transpose_b = matmul_attr.adj_x2 || matmul_attr.transpose_x2;
  GE_WARN_ASSERT(!transpose_a, "Skip lowering node %s, as: Input x1 transposed", node->GetName().c_str());
  int32_t offset = static_cast<int32_t>(matmul_attr.offset_x);
  GE_WARN_ASSERT(offset == 0, "Skip lowering node %s, as: Offset %d not 0", node->GetName().c_str(), offset);
  y_dims = transpose_b ? std::vector<ge::Expression>{y_dims[1], y_dims[0]} : y_dims;
  const auto &m_size = x_dims[0];
  const auto &k_size = y_dims[0];
  const auto &n_size = y_dims[1];
  GE_WARN_ASSERT(EXPECT_SYMBOL_EQ(Symbol(1), n_size), "Skip lowering node %s, as: n %s is not 1",
                 node->GetName().c_str(), n_size.Str().get());
  int64_t const_k = 0U;
  GE_WARN_ASSERT(k_size.GetConstValue(const_k), "Skip lowering node %s, as: k %s not const",
                 node->GetName().c_str(), k_size.Str().get());
  const int64_t max_k = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_WARN_ASSERT(const_k <= max_k, "Skip lowering node %s, as: k %ld > %ld",
                 node->GetName().c_str(), const_k, max_k);

  // [M, K] * [K, 1] -> [M, 1]
  auto x = loop::Load(x_anchor);
  auto y = loop::Load(y_anchor);
  y = transpose_b ? y : loop::Permute(y, {1, 0});  // [K, 1] -> [1, K]
  // [1, K] -> [M, K]
  y = loop::Broadcast(y, {Symbol(1), k_size}, {m_size, k_size});
  const auto z = loop::Mul(x, y);  // [M, K] * [M, K] -> [M, K]
  if (SymbolicUtils::StaticCheckEq(k_size, Symbol(1)) == TriBool::kTrue) {
    loop::Store(node->GetOutDataAnchor(0), z);
    return GRAPH_SUCCESS;
  }
  loop::StoreReduction(loop::ReduceType::SUM, node->GetOutDataAnchor(0), z, {m_size, k_size},
                       {1});  // [M, K] -> [M, 1]
  return GRAPH_SUCCESS;
}

graphStatus InnerLowerMatMul(const NodePtr &node, bool is_batch) {
  MatMulAttr matmul_attr;
  matmul_attr.is_batch = is_batch;
  (void)AttrUtils::GetBool(node->GetOpDesc(), "transpose_x1", matmul_attr.transpose_x1);
  (void)AttrUtils::GetBool(node->GetOpDesc(), "transpose_x2", matmul_attr.transpose_x2);
  (void)AttrUtils::GetInt(node->GetOpDesc(), "offset_x", matmul_attr.offset_x);
  (void)AttrUtils::GetBool(node->GetOpDesc(), "adj_x1", matmul_attr.adj_x1);
  (void)AttrUtils::GetBool(node->GetOpDesc(), "adj_x2", matmul_attr.adj_x2);

  if (is_batch) {
    bool enable_hf32 = false;
    (void)AttrUtils::GetBool(node->GetOpDesc(), "enable_hf32", enable_hf32);
    matmul_attr.enable_hf32 = static_cast<int64_t>(enable_hf32);
    if (TryLowerBatchMatMulToVector(node, matmul_attr) == GRAPH_SUCCESS) {
      GELOGI("batch matmul lowrring to elemwise.");
      return GRAPH_SUCCESS;
    }
  } else {
    (void)AttrUtils::GetInt(node->GetOpDesc(), "opImplMode", matmul_attr.enable_hf32);
    if (TryLowerMatMulToVector(node, matmul_attr) == GRAPH_SUCCESS) {
      GELOGI("matmul lowrring to elemwise.");
      return GRAPH_SUCCESS;
    }
  }

  auto desc = node->GetOpDesc()->GetOutputDescPtr(0);
  GE_ASSERT_NOTNULL(desc);
  matmul_attr.output_dtype = desc->GetDataType();

  ge::Tensor bias_tensor;
  ge::Tensor offset_w_tensor;
  const auto op = ge::OpDescUtils::CreateOperatorFromNode(node);
  auto bias_index = node->GetOpDesc()->GetInputIndexByName("bias");
  if (bias_index >= 0) {
    const auto bias_anchor = node->GetInDataAnchor(bias_index);
    GE_ASSERT_NOTNULL(bias_anchor);
    if (bias_anchor->GetPeerOutAnchor() != nullptr) {
      matmul_attr.has_bias = true;
    }
  }
  auto offset_w_index = node->GetOpDesc()->GetInputIndexByName("offset_w");
  if (offset_w_index >= 0) {
    const auto offset_w_anchor = node->GetInDataAnchor(offset_w_index);
    GE_ASSERT_NOTNULL(offset_w_anchor);
    if (offset_w_anchor->GetPeerOutAnchor() != nullptr) {
      matmul_attr.has_offset_w = true;
    }
  }

  std::vector<ge::InDataAnchorPtr> inputs;
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    GE_ASSERT_NOTNULL(in_anchor);
    if (in_anchor->GetPeerOutAnchor() == nullptr) {
      continue;
    }
    inputs.emplace_back(in_anchor);
  }
  GELOGI(
      "matmul input info, is_batch=%d, transpose_x1=%d, transpose_x2=%d, offset_x=%ld, enable_hf32=%ld, adj_x1=%d, "
      "adj_x2=%d, input num=%zu, has_bias=%d, "
      "has_offset_w=%d.",
      matmul_attr.is_batch, matmul_attr.transpose_x1, matmul_attr.transpose_x2, matmul_attr.offset_x,
      matmul_attr.enable_hf32, matmul_attr.adj_x1, matmul_attr.adj_x2, inputs.size(), matmul_attr.has_bias,
      matmul_attr.has_offset_w);
  loop::StoreMatMul(node->GetOutDataAnchor(0), inputs, matmul_attr);
  return GRAPH_SUCCESS;
}

graphStatus LowerMatMul(const NodePtr &node) {
  return InnerLowerMatMul(node, false);
}

graphStatus BatchLowerMatMul(const NodePtr &node) {
  return InnerLowerMatMul(node, true);
}

REGISTER_POINTWISE_LOWER(Abs, loop::Abs);
REGISTER_POINTWISE_LOWER(Add, loop::Add);
REGISTER_POINTWISE_LOWER(AddV2, loop::Add);
REGISTER_POINTWISE_LOWER(BitwiseAnd, loop::BitwiseAnd);
REGISTER_POINTWISE_LOWER(Div, loop::Div);
REGISTER_POINTWISE_LOWER(Equal, loop::Eq);
REGISTER_POINTWISE_LOWER(Erf, loop::Erf);
REGISTER_POINTWISE_LOWER(Exp, loop::Exp);
REGISTER_POINTWISE_LOWER(GreaterEqual, loop::Ge);
REGISTER_POINTWISE_LOWER(Greater, loop::Gt);
REGISTER_POINTWISE_LOWER(Less, loop::Lt);
REGISTER_POINTWISE_LOWER(IsNan, loop::IsNan);
REGISTER_POINTWISE_LOWER(IsFinite, loop::IsFinite);
REGISTER_POINTWISE_LOWER(LessEqual, loop::Le);
REGISTER_POINTWISE_LOWER(LogicalAnd, loop::LogicalAnd);
REGISTER_POINTWISE_LOWER(LogicalOr, loop::LogicalOr);
REGISTER_POINTWISE_LOWER(LogicalNot, loop::LogicalNot);
REGISTER_POINTWISE_LOWER(Maximum, loop::Maximum);
REGISTER_POINTWISE_LOWER(Minimum, loop::Minimum);
REGISTER_POINTWISE_LOWER(Mul, loop::Mul);
REGISTER_POINTWISE_LOWER(NotEqual, loop::Ne);
REGISTER_POINTWISE_LOWER(Neg, loop::Neg);
REGISTER_POINTWISE_LOWER(Pow, loop::Pow);
REGISTER_POINTWISE_LOWER(RealDiv, loop::TrueDiv);
REGISTER_POINTWISE_LOWER(Reciprocal, loop::Reciprocal);
REGISTER_POINTWISE_LOWER(Relu, loop::Relu);
REGISTER_POINTWISE_LOWER(Rsqrt, loop::Rsqrt);
REGISTER_POINTWISE_LOWER(Sigmoid, loop::Sigmoid);
REGISTER_POINTWISE_LOWER(Sign, loop::Sign);
REGISTER_POINTWISE_LOWER(Sqrt, loop::Sqrt);
REGISTER_POINTWISE_LOWER(Sub, loop::Sub);
REGISTER_POINTWISE_LOWER(Tanh, loop::Tanh);
REGISTER_POINTWISE_LOWER(SelectV2, loop::Where);
REGISTER_POINTWISE_LOWER(FloorDiv, loop::FloorDiv);
REGISTER_POINTWISE_LOWER(Gelu, loop::Gelu);

REGISTER_REDUCTION_LOWER(ReduceSum, loop::ReduceType::SUM);
REGISTER_REDUCTION_LOWER(ReduceSumD, loop::ReduceType::SUM);
REGISTER_REDUCTION_LOWER(ReduceMean, loop::ReduceType::MEAN);
REGISTER_REDUCTION_LOWER(ReduceMeanD, loop::ReduceType::MEAN);
REGISTER_REDUCTION_LOWER(ReduceMax, loop::ReduceType::MAX);
REGISTER_REDUCTION_LOWER(ReduceMaxD, loop::ReduceType::MAX);
REGISTER_REDUCTION_LOWER(ReduceMin, loop::ReduceType::MIN);
REGISTER_REDUCTION_LOWER(ReduceMinD, loop::ReduceType::MIN);
REGISTER_REDUCTION_LOWER(ReduceProd, loop::ReduceType::PROD);
REGISTER_REDUCTION_LOWER(ReduceProdD, loop::ReduceType::PROD);

REGISTER_LOWERING_WITH_EXISTED(Concat, LowerConcat);
REGISTER_LOWERING_WITH_EXISTED(ConcatD, LowerConcat);
REGISTER_LOWERING_WITH_EXISTED(ConcatV2, LowerConcat);
REGISTER_LOWERING_WITH_EXISTED(ConcatV2D, LowerConcat);

REGISTER_LOWERING_WITH_EXISTED(GatherV2, LowerGather);
REGISTER_LOWERING_WITH_EXISTED(Gather, LowerGather);

REGISTER_LOWERING_WITH_EXISTED(Tile, LowerTile);
REGISTER_LOWERING_WITH_EXISTED(TileD, LowerTile);

REGISTER_LOWERING_WITH_EXISTED(Slice, LowerSlice);
REGISTER_LOWERING_WITH_EXISTED(SliceD, LowerSlice);
REGISTER_LOWERING_WITH_EXISTED(StridedSlice, LowerStridedSlice);
REGISTER_LOWERING_WITH_EXISTED(StridedSliceD, LowerStridedSlice);
REGISTER_LOWERING_WITH_EXISTED(Split, LowerSplit);
REGISTER_LOWERING_WITH_EXISTED(SplitD, LowerSplit);
REGISTER_LOWERING_WITH_EXISTED(SplitV, LowerSplit);
REGISTER_LOWERING_WITH_EXISTED(SplitVD, LowerSplit);
REGISTER_LOWERING_WITH_EXISTED(Transpose, LowerTranspose);
REGISTER_LOWERING_WITH_EXISTED(TransposeD, LowerTranspose);

REGISTER_LOWERING_WITH_EXISTED(MatMul, LowerMatMul);
REGISTER_LOWERING_WITH_EXISTED(MatMulV2, LowerMatMul);
REGISTER_LOWERING_WITH_EXISTED(MatMulV3, LowerMatMul);
REGISTER_LOWERING_WITH_EXISTED(BatchMatMul, BatchLowerMatMul);
REGISTER_LOWERING_WITH_EXISTED(BatchMatMulV2, BatchLowerMatMul);
REGISTER_LOWERING_WITH_EXISTED(BatchMatMulV3, BatchLowerMatMul);

REGISTER_LOWERING(ClipByValue) {
  loop::Index broadcasted;
  std::vector<loop::Index> indices;

  auto x = node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(x);
  auto y = node->GetInDataAnchor(1);
  GE_ASSERT_NOTNULL(y);
  auto z = node->GetInDataAnchor(2);
  GE_ASSERT_NOTNULL(z);

  auto input_tensor = x->GetPeerOutAnchor();
  auto clip_value_min = y->GetPeerOutAnchor();
  auto clip_value_max = z->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(input_tensor);
  GE_ASSERT_NOTNULL(clip_value_min);
  GE_ASSERT_NOTNULL(clip_value_max);
  std::vector<Expression> input_tensor_dims;
  std::vector<Expression> clip_value_min_dims;
  std::vector<Expression> clip_value_max_dims;
  GE_WARN_ASSERT(loop::GetBufferShape(input_tensor, input_tensor_dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input_tensor shape.", node->GetNamePtr());
  GE_WARN_ASSERT(loop::GetBufferShape(clip_value_min, clip_value_min_dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get clip_value_min shape.", node->GetNamePtr());
  GE_WARN_ASSERT(loop::GetBufferShape(clip_value_max, clip_value_max_dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get clip_value_max shape.", node->GetNamePtr());

  auto UnsqueezeToMatchRank = [&](loop::LoopVar tensor, std::vector<Expression> &dims) {
    if (dims.size() < input_tensor_dims.size()) {
      const size_t num_unsqueeze = input_tensor_dims.size() - dims.size();
      for (size_t i = 0U; i < num_unsqueeze; i++) {
        tensor = loop::Unsqueeze(tensor, static_cast<int64_t>(i));
        dims.emplace(dims.begin(), Symbol(1));
      }
    }
    return tensor;
  };

  auto clip_value_min_tensor = UnsqueezeToMatchRank(loop::Load(y), clip_value_min_dims);
  auto clip_value_max_tensor = UnsqueezeToMatchRank(loop::Load(z), clip_value_max_dims);

  indices.emplace_back(input_tensor_dims);
  indices.emplace_back(clip_value_min_dims);
  indices.emplace_back(clip_value_max_dims);

  GE_WARN_ASSERT_GRAPH_SUCCESS(Broadcast(indices, broadcasted));
  constexpr size_t clip_value_min_idx = 1U;
  constexpr size_t clip_value_max_idx = 2U;
  auto clip_value_min_var = loop::Broadcast(clip_value_min_tensor, indices[clip_value_min_idx], broadcasted);
  auto clip_value_max_var = loop::Broadcast(clip_value_max_tensor, indices[clip_value_max_idx], broadcasted);

  auto minimum = loop::Minimum(loop::Load(x), clip_value_max_var);
  (void)loop::Store(node->GetOutDataAnchor(0), loop::Maximum(minimum, clip_value_min_var));
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(BiasAdd) {
  std::vector<Expression> dims;
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(0));
  auto src = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src);
  GE_WARN_ASSERT(loop::GetBufferShape(src, dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input shape.", node->GetNamePtr());
  std::string data_format;
  GE_ASSERT(AttrUtils::GetStr(node->GetOpDesc(), "data_format", data_format));

  int32_t offset = -1;
  if (data_format == "NHWC") {
    offset = dims.size() - 1;
  } else if (data_format == "NCHW") {
    offset = dims.size() - 3;
  } else {
    GELOGI("biasAdd data format can only be NHWC or NCHW.");
    return GRAPH_FAILED;
  }

  std::vector<loop::BroadcastOp::DimKind> status;
  status.resize(dims.size(), loop::BroadcastOp::DimKind::NEW_AXIS);
  status[offset] = loop::BroadcastOp::DimKind::NORMAL;
  std::vector<loop::LoopVar> vars = {loop::Load(node->GetInDataAnchor(0)),
                                     loop::Broadcast(loop::Load(node->GetInDataAnchor(1)), status)};
  auto out_anchor = node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(out_anchor);
  (void)loop::Store(out_anchor, loop::Add(vars));
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(ZerosLike) {
  GE_WARN_ASSERT(node->GetOutNodesSize() == 1U,
                 "Skip lowering node:%s, as: node single out anchor has multi-reference",
                 node->GetName().c_str());
  auto after_node = node->GetOutNodes().at(0);
  GE_ASSERT_NOTNULL(after_node);
  GE_WARN_ASSERT(find(reduce_types.begin(), reduce_types.end(), after_node->GetType()) == reduce_types.end(),
                 "Skip lowering node:%s, as: After node is reduce type, fuse them is meaningless",
                 node->GetName().c_str());
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(0));
  auto src = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src);
  GE_ASSERT_NOTNULL(src->GetOwnerNode());
  GE_ASSERT_NOTNULL(src->GetOwnerNode()->GetOpDesc());
  auto desc = src->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(src->GetIdx());
  GE_ASSERT_NOTNULL(desc);
  auto dtype = desc->GetDataType();
  std::vector<Expression> dims;
  GE_WARN_ASSERT(loop::GetBufferShape(src, dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input shape.", node->GetNamePtr());

  loop::LoopVar scalar_var = loop::Scalar("0", dtype);
  std::vector<loop::BroadcastOp::DimKind> status(dims.size(), loop::BroadcastOp::DimKind::NEW_AXIS);
  status.resize(dims.size(), loop::BroadcastOp::DimKind::NEW_AXIS);
  (void)loop::Store(node->GetOutDataAnchor(0), loop::Broadcast(scalar_var, status));
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(SigmoidGrad) {
  auto x = node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(x);
  auto y = node->GetInDataAnchor(1);
  GE_ASSERT_NOTNULL(y);
  auto src = x->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src);
  GE_ASSERT_NOTNULL(src->GetOwnerNode());
  GE_ASSERT_NOTNULL(src->GetOwnerNode()->GetOpDesc());
  auto desc = src->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(src->GetIdx());
  GE_ASSERT_NOTNULL(desc);
  auto dtype = desc->GetDataType();
  auto sub_x = loop::Sub(loop::Scalar("1", dtype), loop::Load(x));
  auto mul_x = loop::Mul(loop::Load(x), sub_x);
  (void)loop::Store(node->GetOutDataAnchor(0), loop::Mul(mul_x, loop::Load(y)));
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(Fill) {
  std::vector<loop::Index> indices;
  std::vector<Expression> src_dims;
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(1));
  auto src = node->GetInDataAnchor(1)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src);
  GE_WARN_ASSERT(loop::GetBufferShape(src, src_dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input shape.", node->GetNamePtr());
  indices.emplace_back(src_dims);

  std::vector<Expression> dims;
  GE_WARN_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(node->GetOutDataAnchor(0), dims));
  indices.emplace_back(dims);
  loop::Index broadcast;
  GE_ASSERT_GRAPH_SUCCESS(Broadcast(indices, broadcast));
  (void)loop::Store(node->GetOutDataAnchor(0),
                    loop::Broadcast(loop::Load(node->GetInDataAnchor(1)), src_dims, broadcast));
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(AddN) {
  loop::Index broadcasted;
  std::vector<loop::Index> indices;
  for (auto &in_anchor : node->GetAllInDataAnchors()) {
    auto src = in_anchor->GetPeerOutAnchor();
    GE_WARN_ASSERT(src != nullptr,
                   "Skip lowering node %s, as: Input anchor is nullptr", node->GetName().c_str());
    auto desc = src->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(src->GetIdx());
    GE_ASSERT_NOTNULL(desc);
    auto sym_attr = desc->GetAttrsGroup<SymbolicDescAttr>();
    GE_WARN_ASSERT(sym_attr != nullptr,
                   "Skip lowering node %s, as: No symbolic desc attr.", node->GetName().c_str());
    indices.emplace_back(sym_attr->symbolic_tensor.GetOriginSymbolShape().GetDims());
  }
  GE_ASSERT_GRAPH_SUCCESS(Broadcast(indices, broadcasted));

  auto sum = loop::Broadcast(loop::Load(node->GetInDataAnchor(0)), indices[0], broadcasted);
  for (size_t i = 1U; i < node->GetAllInDataAnchorsSize(); i++) {
    sum = loop::Add(
        sum, loop::Broadcast(loop::Load(node->GetInDataAnchor(static_cast<int32_t>(i))), indices[i], broadcasted));
  }
  loop::Store(node->GetOutDataAnchor(0), sum);
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(Square) {
  auto x = loop::Load(node->GetInDataAnchor(0));
  loop::Store(node->GetOutDataAnchor(0), loop::Mul(x, x));
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(SquaredDifference) {
  auto kernel = [](const std::vector<loop::LoopVar> &vars) -> loop::LoopVar {
    auto sub = loop::Sub(vars[0], vars[1]);
    return loop::Mul(sub, sub);
  };
  return LowerPointwise(node, kernel);
}

REGISTER_LOWERING(Cast) {
  auto x = loop::Load(node->GetInDataAnchor(0));
  int32_t data_type = 0;
  GE_ASSERT_TRUE(ge::AttrUtils::GetInt(node->GetOpDesc(), "dst_type", data_type));
  GE_WARN_ASSERT(static_cast<ge::DataType>(data_type) != ge::DT_BOOL,
                 "Skip lowering node %s, as: Dst type no support bool.", node->GetName().c_str());
  loop::Store(node->GetOutDataAnchor(0), loop::Cast(x, static_cast<ge::DataType>(data_type)));
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(BiasAddGrad) {
  std::vector<Expression> dims;
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(0));
  auto src = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src);
  GE_WARN_ASSERT(loop::GetBufferShape(src, dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input shape.", node->GetNamePtr());
  std::string data_format;
  GE_ASSERT(AttrUtils::GetStr(node->GetOpDesc(), "data_format", data_format));
  size_t offset = 0U;
  if (data_format == "NCHW") {
    GE_ASSERT(dims.size() >= 3U);
    offset = dims.size() - 3U;
  } else if (data_format == "NHWC") {
    GE_ASSERT(!dims.empty());
    offset = dims.size() - 1U;
  } else {
    GELOGI("BiasAddGrad data format can only be NHWC or NCHW.");
    return GRAPH_FAILED;
  }
  std::vector<size_t> reduced_axis;
  for (size_t i = 0U; i < dims.size(); i++) {
    if (i != offset) {
      reduced_axis.emplace_back(i);
    }
  }
  auto x = loop::Load(node->GetInDataAnchor(0));
  loop::StoreReduction(loop::ReduceType::SUM, node->GetOutDataAnchor(0), x, dims, reduced_axis);
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(BroadcastTo) {
  std::vector<loop::Index> indices;
  const auto x_anchor = node->GetInDataAnchor(0);
  const auto shape_anchor = node->GetInDataAnchor(1);
  GE_ASSERT_NOTNULL(x_anchor);
  GE_ASSERT_NOTNULL(shape_anchor);
  auto x = loop::Load(x_anchor);
  auto shape = loop::Load(shape_anchor);
  std::vector<ge::Expression> x_dims;
  GE_WARN_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(node->GetInDataAnchor(0), x_dims));

  std::vector<Expression> output_dims;
  GE_WARN_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(node->GetOutDataAnchor(0), output_dims));
  indices.emplace_back(output_dims);
  indices.emplace_back(x_dims);

  loop::Index broadcast;
  GE_ASSERT_GRAPH_SUCCESS(Broadcast(indices, broadcast));
  loop::Store(node->GetOutDataAnchor(0), loop::Broadcast(x, x_dims, broadcast));
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(LayerNormBetaGammaBackpropV2) {
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(0));
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(1));
  auto dy = loop::Load(node->GetInDataAnchor(0));
  auto res_for_gamma = loop::Load(node->GetInDataAnchor(1));
  auto mul = loop::Mul(dy, res_for_gamma);
  std::vector<loop::Index> indices;
  std::vector<ge::Expression> dy_dims;
  std::vector<ge::Expression> res_for_gamma_dims;
  auto src_dy = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src_dy);
  GE_WARN_ASSERT(loop::GetBufferShape(src_dy, dy_dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input0 shape.", node->GetNamePtr());
  auto src_res_for_gamma = node->GetInDataAnchor(1)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src_res_for_gamma);
  GE_WARN_ASSERT(loop::GetBufferShape(src_res_for_gamma, res_for_gamma_dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input1 shape.", node->GetNamePtr());
  indices.emplace_back(dy_dims);
  indices.emplace_back(res_for_gamma_dims);
  loop::Index mul_shape;
  Broadcast(indices, mul_shape);

  // dy_shape compare with output_shape for get the reduce_axis. dy_shape [s0, s1, s2]  output_shape[s0, s1]
  // reduce_axis[2]
  std::vector<size_t> reduce_axis;
  std::vector<Expression> output_dims;
  vector<int64_t> shape_gamma;
  AttrUtils::GetListInt(node->GetOpDesc(), "shape_gamma", shape_gamma);
  auto it = find(shape_gamma.begin(), shape_gamma.end(), -1);
  if (it == shape_gamma.end()) {
    for (size_t i = 0; i < shape_gamma.size(); i++) {
      reduce_axis.emplace_back(i);
    }
  } else {
    for (size_t i = 0; i < shape_gamma.size(); i++) {
      if (shape_gamma[i] == -1) {
        reduce_axis.emplace_back(i);
      }
    }
  }

  auto pd_gamma = loop::StoreReduction(loop::ReduceType::SUM, node->GetOutDataAnchor(0), mul, mul_shape, reduce_axis);
  auto pd_beta = loop::StoreReduction(loop::ReduceType::SUM, node->GetOutDataAnchor(1), dy, dy_dims, reduce_axis);
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(ReluGrad) {
  auto gradients = loop::Load(node->GetInDataAnchor(0));
  auto features = loop::Load(node->GetInDataAnchor(1));
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(0));
  auto src = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  GE_WARN_ASSERT((src != nullptr) && (src->GetOwnerNode() != nullptr) && (src->GetOwnerNode()->GetOpDesc() != nullptr));
  auto desc = src->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(src->GetIdx());
  GE_ASSERT_NOTNULL(desc);
  std::vector<Expression> dims;
  GE_WARN_ASSERT(loop::GetBufferShape(src, dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input shape.", node->GetNamePtr());
  GE_ASSERT_NOTNULL(desc);
  auto dtype = desc->GetDataType();
  std::vector<loop::BroadcastOp::DimKind> status(dims.size(), loop::BroadcastOp::DimKind::NEW_AXIS);
  status.resize(dims.size(), loop::BroadcastOp::DimKind::NEW_AXIS);
  auto zero = loop::Broadcast(loop::Scalar("0", dtype), status);
  DataType transdtype = dtype;

  if ((dtype == DT_INT8) || (dtype == DT_UINT8)) {
    features = loop::Cast(features, DT_FLOAT16);
    transdtype = DT_FLOAT16;
  }
  if (dtype == DT_BF16) {
    features = loop::Cast(features, DT_FLOAT);
    transdtype = DT_FLOAT;
  }

  if ((transdtype != DT_FLOAT) && (transdtype != DT_FLOAT16) && (transdtype != DT_INT32)) {
    loop::Store(node->GetOutDataAnchor(0), loop::Where(loop::Le(features, zero), zero, gradients));
    return GRAPH_SUCCESS;
  }

  auto derivative = CalculateOneOrZero(features, transdtype, status);
  GE_WARN_ASSERT(derivative.IsValid(), "Skip lowering node %s, as derivative is invalid.", node->GetNamePtr());
  if ((dtype == DT_INT8) || (dtype == DT_UINT8) || (dtype == DT_BF16)) {
    derivative = loop::Cast(derivative, dtype);
  }

  loop::Store(node->GetOutDataAnchor(0), loop::Mul(gradients, derivative));
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(RsqrtGrad) {
  auto y = loop::Load(node->GetInDataAnchor(0));
  auto dy = loop::Load(node->GetInDataAnchor(1));
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(0));
  auto src = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  vector<Expression> dims;
  GE_WARN_ASSERT(loop::GetBufferShape(src, dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input shape.", node->GetNamePtr());
  GE_ASSERT_NOTNULL(src);
  GE_ASSERT_NOTNULL(src->GetOwnerNode());
  GE_ASSERT_NOTNULL(src->GetOwnerNode()->GetOpDesc());
  auto desc = src->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(src->GetIdx());
  GE_WARN_ASSERT(desc != nullptr,
               "Skip lowering node %s, as: Input opdesc is nullptr.", node->GetNamePtr());
  auto dtype = desc->GetDataType();

  auto scalar_num = loop::Scalar("-0.5", dtype);
  std::vector<loop::BroadcastOp::DimKind> status(dims.size(), loop::BroadcastOp::DimKind::NEW_AXIS);
  status.resize(dims.size(), loop::BroadcastOp::DimKind::NEW_AXIS);
  scalar_num = loop::Broadcast(scalar_num, status);
  auto z = loop::Mul(scalar_num, loop::Mul(dy, loop::Mul(y, loop::Mul(y, y))));
  loop::Store(node->GetOutDataAnchor(0), z);
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(Muls) {
  auto x = loop::Load(node->GetInDataAnchor(0));
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(0));
  auto src = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src);
  vector<Expression> dims;
  GE_WARN_ASSERT(loop::GetBufferShape(src, dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input shape.", node->GetNamePtr());
  auto desc = src->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(src->GetIdx());
  GE_WARN_ASSERT(desc != nullptr,
                 "Skip lowering node %s, as: Input opdesc is nullptr.", node->GetNamePtr());
  auto dtype = desc->GetDataType();

  float32_t value = 0;
  GE_WARN_ASSERT(AttrUtils::GetFloat(node->GetOpDesc(), "value", value),
                 "Skip lowering node %s, as: Failed to get value from opdesc.", node->GetNamePtr());

  std::ostringstream oss;
  oss.precision(7);            // 设置小数点后位数
  oss << std::fixed << value;  // 固定小数格式
  std::string slope_str = oss.str();
  auto scalar_value = loop::Scalar(slope_str, dtype);
  std::vector<loop::BroadcastOp::DimKind> status(dims.size(), loop::BroadcastOp::DimKind::NEW_AXIS);
  status.resize(dims.size(), loop::BroadcastOp::DimKind::NEW_AXIS);
  scalar_value = loop::Broadcast(scalar_value, status);

  loop::Store(node->GetOutDataAnchor(0), loop::Mul(x, scalar_value));
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(SquareSumV1) {
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(0));
  auto x_anchor = loop::Load(node->GetInDataAnchor(0));
  auto src = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  std::vector<int64_t> axis;
  GE_ASSERT(AttrUtils::GetListInt(node->GetOpDesc(), "axis", axis));

  auto pow = loop::Mul(x_anchor, x_anchor);
  std::vector<ge::Expression> dims;
  GE_WARN_ASSERT(loop::GetBufferShape(src, dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input shape.", node->GetNamePtr());
  int32_t dim_num = dims.size();
  std::vector<size_t> reduce_axis;

  for (size_t i = 0; i < axis.size(); i++) {
    int64_t dim = axis[i] < 0 ? static_cast<int64_t>(dim_num) + axis[i] : axis[i];
    reduce_axis.emplace_back(static_cast<size_t>(dim));
  }

  loop::StoreReduction(loop::ReduceType::SUM, node->GetOutDataAnchor(0), pow, dims, reduce_axis);
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(Unsqueeze) {
  auto x = loop::Load(node->GetInDataAnchor(0));
  std::vector<int64_t> vec_axes;
  GE_ASSERT_TRUE(AttrUtils::GetListInt(node->GetOpDesc(), "axes", vec_axes));
  for (const auto vec_axe : vec_axes) {
    x = loop::Unsqueeze(x, vec_axe);
  }
  loop::Store(node->GetOutDataAnchor(0), x);
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(Squeeze) {
  auto x = loop::Load(node->GetInDataAnchor(0));
  std::vector<int64_t> vec_axes;
  GE_ASSERT_TRUE(AttrUtils::GetListInt(node->GetOpDesc(), "axis", vec_axes));
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(0));
  auto src = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  std::vector<ge::Expression> dims;
  GE_WARN_ASSERT(loop::GetBufferShape(src, dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input shape.", node->GetNamePtr());
  if (vec_axes.empty()) {
    for (size_t i = 0UL; i < dims.size(); i++) {
      if (EXPECT_SYMBOL_EQ(Symbol(1), dims[i])) {
        vec_axes.emplace_back(i);
      }
    }
  }
  for (auto &vec_axe : vec_axes) {
    if (vec_axe < 0) {
      vec_axe += static_cast<int64_t>(dims.size());
      GE_WARN_ASSERT(vec_axe >= 0 && static_cast<size_t>(vec_axe) < dims.size(),
                     "Skip lowering node %s, as: Squeeze axis %ld must >= 0 and within rank %zu",
                     node->GetNamePtr(), vec_axe, dims.size());
    }
  }
  sort(vec_axes.begin(), vec_axes.end());
  for (size_t i = 0; i < vec_axes.size(); i++) {
    x = loop::Squeeze(x, vec_axes[i] - static_cast<int64_t>(i));
  }
  loop::Store(node->GetOutDataAnchor(0), x);
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(DivNoNan) {
  auto x1 = loop::Load(node->GetInDataAnchor(0));
  auto x2 = loop::Load(node->GetInDataAnchor(1));
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(1));
  auto src = node->GetInDataAnchor(1)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src);
  GE_ASSERT_NOTNULL(src->GetOwnerNode());
  GE_ASSERT_NOTNULL(src->GetOwnerNode()->GetOpDesc());
  auto desc = src->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(src->GetIdx());
  GE_ASSERT_NOTNULL(desc);
  auto dtype = desc->GetDataType();
  loop::LoopVar scalar_var = loop::Scalar("0", dtype);
  vector<Expression> dims;
  GE_WARN_ASSERT(loop::GetBufferShape(src, dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input shape.", node->GetNamePtr());
  std::vector<loop::BroadcastOp::DimKind> status(dims.size(), loop::BroadcastOp::DimKind::NEW_AXIS);
  status.resize(dims.size(), loop::BroadcastOp::DimKind::NEW_AXIS);
  auto scalar_zero = loop::Broadcast(scalar_var, status);
  auto is_zero = loop::Eq(x2, scalar_zero);
  auto div_result = loop::Div(x1, x2);
  auto result = loop::Where(is_zero, scalar_zero, div_result);
  loop::Store(node->GetOutDataAnchor(0), result);
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(LeakyRelu) {
  float32_t negative_slope = 0.0;
  GE_ASSERT_TRUE(ge::AttrUtils::GetFloat(node->GetOpDesc(), "negative_slope", negative_slope));
  auto x = loop::Load(node->GetInDataAnchor(0));
  auto leakyrelu = loop::LeakyRelu(x, negative_slope);
  loop::Store(node->GetOutDataAnchor(0), leakyrelu);
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(LeakyReluGrad) {
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(0));

  loop::Index broadcasted_var;
  std::vector<loop::Index> indices_var;
  vector<InDataAnchorPtr> indata_anchors_var;
  GE_WARN_ASSERT(node->GetAllInDataAnchorsSize() == 2U,
               "Skip lowering node %s, as: Input size is not 2.", node->GetNamePtr());
  for (auto &in_anchor : node->GetAllInDataAnchors()) {
    indata_anchors_var.emplace_back(in_anchor);
  }
  GE_WARN_ASSERT_GRAPH_SUCCESS(BroadCastByInDataAnchors(indata_anchors_var, indices_var, broadcasted_var));
  auto gradients = loop::Broadcast(loop::Load(node->GetInDataAnchor(0)), indices_var[0], broadcasted_var);
  auto features = loop::Broadcast(loop::Load(node->GetInDataAnchor(1)), indices_var[1], broadcasted_var);;

  auto src = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  GE_WARN_ASSERT(src != nullptr,
                 "Skip lowering node %s, as: Input anchor is nullptr.", node->GetNamePtr());
  auto desc = src->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(src->GetIdx());
  GE_WARN_ASSERT(desc != nullptr,
                 "Skip lowering node %s, as: Input opdesc is nullptr.", node->GetNamePtr());
  auto dtype = desc->GetDataType();
  auto zero = ScalarBroadcast2Size("0.0", dtype, broadcasted_var.size());
  float32_t negative_slope = 0.0f;
  GE_ASSERT_TRUE(ge::AttrUtils::GetFloat(node->GetOpDesc(), "negative_slope", negative_slope));
  std::ostringstream oss;
  oss.precision(7);
  oss << std::fixed << negative_slope;
  std::string slop_str = oss.str();
  auto negative_slop_var = ScalarBroadcast2Size(slop_str, dtype, broadcasted_var.size());

  auto mul_zero = loop::Mul(gradients, negative_slop_var);
  auto gt = loop::Gt(features, zero);
  auto select = loop::Where(gt, gradients, mul_zero);
  loop::Store(node->GetOutDataAnchor(0), select);
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(Log) {
  auto x = loop::Load(node->GetInDataAnchor(0));
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(0));
  auto src = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src);
  auto desc = src->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(src->GetIdx());
  GE_ASSERT_NOTNULL(desc);
  auto dtype = desc->GetDataType();
  std::vector<Expression> dims;
  GE_WARN_ASSERT(loop::GetBufferShape(src, dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input shape.", node->GetNamePtr());
  std::vector<loop::BroadcastOp::DimKind> status(dims.size(), loop::BroadcastOp::DimKind::NEW_AXIS);
  status.resize(dims.size(), loop::BroadcastOp::DimKind::NEW_AXIS);

  float base = -1.0f;
  float scale = 1.0f;
  float shift = 0.0f;
  constexpr double epsilon = 1e-8;
  (void)AttrUtils::GetFloat(node->GetOpDesc(), "base", base);
  GE_ASSERT(base > 0 || std::abs(base - (-1.0)) < epsilon, "Log base should be greater than 0");
  (void)AttrUtils::GetFloat(node->GetOpDesc(), "scale", scale);
  (void)AttrUtils::GetFloat(node->GetOpDesc(), "shift", shift);

  auto x_log = x;
  if (IsClose(scale, 1.0f) && IsClose(shift, 0.0f)) {
    x_log = loop::Ln(x);
  } else {
    auto x_scale_and_shift = x;
    if (!IsClose(scale, 1.0f)) {
      auto scalar_scale = loop::Broadcast(loop::Scalar(ToPrecString(scale, 7), dtype), status);
      x_scale_and_shift = loop::Mul(x, scalar_scale);
    }
    if (!IsClose(shift, 0.0f)) {
      auto scalar_shift = loop::Broadcast(loop::Scalar(ToPrecString(shift, 7), dtype), status);
      x_scale_and_shift = loop::Add(x_scale_and_shift, scalar_shift);
    }
    x_log = loop::Ln(x_scale_and_shift);
  }

  auto log_base = IsClose(base, -1.0f) ? 1.0f : std::log(base);
  auto base_scale = 1.0f / log_base;
  if (!IsClose(base_scale, 1.0f)) {
    auto res = loop::Mul(x_log, loop::Broadcast(loop::Scalar(ToPrecString(base_scale, 20), dtype), status));
    loop::Store(node->GetOutDataAnchor(0), res);
  } else {
    loop::Store(node->GetOutDataAnchor(0), x_log);
  }
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(Pack) {
  constexpr uint32_t kMaxInputForPackTailDim = 16;
  const auto &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  int64_t axis = 0;
  GE_ASSERT_TRUE(AttrUtils::GetInt(op_desc, "axis", axis), "Failed to get attr axis");
  vector<Expression> dims;
  GE_WARN_ASSERT(loop::GetBufferShape(node->GetInDataAnchor(0), dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input shape.", node->GetNamePtr());
  const auto dim_num = static_cast<int64_t>(dims.size());
  // 1. dim_num = 0, 实际为首轴concat，支持
  // 2. 非尾轴concat，支持
  // 3. 尾轴concat，输入个数<=kMaxInputForPackTailDim可以走优化后的kernel，支持，大于则不支持
  GE_WARN_ASSERT((dim_num == 0) || ((axis != -1) && (axis < dim_num)) ||
                     (node->GetAllInDataAnchorsSize() <= kMaxInputForPackTailDim),
                 "Skip lowering node %s, as: Pack on the last axis and num_inputs(%u) > %u is not supported yet,"
                 "input_shape = %s, axis = %ld", node->GetNamePtr(),
                 node->GetAllInDataAnchorsSize(), kMaxInputForPackTailDim,
                 op_desc->GetInputDesc(0U).GetShape().ToString().c_str(), axis);
  std::vector<InDataAnchorPtr> inputs;
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    GE_ASSERT_NOTNULL(in_anchor);
    inputs.emplace_back(in_anchor);
  }
  loop::StorePack(node->GetOutDataAnchor(0), inputs, axis);
  return GRAPH_SUCCESS;
}

bool CheckAndGetDims(const vector<Expression> &long_dims, const vector<Expression> &short_dims, vector<int64_t> &dims) {
  vector<bool> is_exist(long_dims.size(), false);
  size_t dims_idx = 0U;
  for (const auto &short_dim : short_dims) {
    bool is_find = false;
    string longdim_str;
    for (size_t i = dims_idx; i < long_dims.size(); i++) {
      longdim_str += " " + SymbolicUtils::ToString(long_dims[i]);
      if (long_dims[i] == short_dim && !is_exist[i]) {
        is_find = true;
        is_exist[i] = true;
        dims_idx = i;
        break;
      }
    }
    GE_WARN_ASSERT(
        is_find,
        "There are some axes in the short axis that do not exist in the long axis, short axes: %s, long dims: %s",
        SymbolicUtils::ToString(short_dim).c_str(), longdim_str.c_str());
  }

  for (size_t i = 0U; i < is_exist.size(); i++) {
    if (!is_exist[i] && long_dims[i] != Symbol(1)) {
      GELOGW("Axes that do not exist must be 1, Long axes: %zu, not equal than 1", i);
      return false;
    }
    if (!is_exist[i]) {
      dims.push_back(static_cast<int64_t>(i));
    }
  }
  return !dims.empty();
}

REGISTER_LOWERING(Reshape) {
  // Reshape只做attr为默认参数，且不进行轴转换，能进行unsqueeze/squeeze的情况。[3,4]->[1,3,4]/[2,1,3]->[2,3]
  const auto &op_desc = node->GetOpDesc();
  int64_t axis = 0;
  int64_t num_axes = -1L;
  GE_WARN_ASSERT(AttrUtils::GetInt(op_desc, "axis", axis),
                 "Skip lowering node %s, as: Failed to get attr axis", node->GetNamePtr());
  GE_WARN_ASSERT(AttrUtils::GetInt(op_desc, "num_axes", num_axes),
                 "Skip lowering node %s, as: Failed to get attr num_axes", node->GetNamePtr());
  GE_WARN_ASSERT(axis == 0 && num_axes == -1L,
                 "Skip lowering node %s, as: Attr axis must be 0 and num_axes must be -1", node->GetNamePtr());
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(0));
  auto src = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src);
  auto desc = src->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(src->GetIdx());
  GE_ASSERT_NOTNULL(desc);
  std::vector<Expression> dims;
  GE_WARN_ASSERT(loop::GetBufferShape(src, dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input shape", node->GetNamePtr());
  std::vector<Expression> output_dims;
  GE_WARN_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(node->GetOutDataAnchor(0), output_dims));
  size_t short_idx = 0U;
  std::vector<size_t> mul_idx;
  bool need_realize = false;
  // Shape为a,b <-> a*b 需要realize前面节点及该节点，当前功能后端不支持，直接返回。
  need_realize = (dims.size() > output_dims.size())
                     ? AutofuseUtils::CheckAndMulDetect(dims, output_dims, short_idx, mul_idx)
                     : AutofuseUtils::CheckAndMulDetect(output_dims, dims, short_idx, mul_idx);
  if (need_realize) {
    return GRAPH_SUCCESS;
  }

  auto x = loop::Load(node->GetInDataAnchor(0));
  auto reshape = loop::Reshape(x, dims, output_dims);
  GE_WARN_ASSERT(reshape.IsValid(), "skip lowering node %s, as: no specific reshape pattern matched", node->GetNamePtr());
  loop::Store(node->GetOutDataAnchor(0), reshape);
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(ExpandDims) {
  auto x = loop::Load(node->GetInDataAnchor(0));
  const auto op = ge::OpDescUtils::CreateOperatorFromNode(node);
  ge::Tensor axis_tensor;
  GE_WARN_ASSERT(op.GetInputConstData("axis", axis_tensor) == ge::SUCCESS,
                 "Skip lowering node %s, as: Dynamic axes", node->GetNamePtr());

  int64_t dim = 0;
  const std::vector<int64_t> dims = axis_tensor.GetTensorDesc().GetShape().GetDims();
  GE_WARN_ASSERT(dims.size() <= 1U, "Skip lowering node %s, as: ExpandDims axis must be a scalar or vector",
                 node->GetNamePtr());
  const int64_t num_axes = dims.empty() ? 1 : dims[0];
  GE_WARN_ASSERT(num_axes > 0, "Skip lowering node %s, as: Number of axes must be positive",
                 node->GetNamePtr());
  GE_WARN_ASSERT(axis_tensor.GetData() != nullptr,
                 "Skip lowering node %s, as: Axis tensor is nullptr", node->GetNamePtr());
  ge::DataType tensor_dtype = axis_tensor.GetTensorDesc().GetDataType();
  for (int64_t i = 0; i < num_axes; i++) {
    if (tensor_dtype == ge::DT_INT32) {
      int32_t axis = *reinterpret_cast<const int32_t *>(axis_tensor.GetData() + i * sizeof(int32_t));
      dim = static_cast<int64_t>(axis);
    } else if (tensor_dtype == ge::DT_INT64) {
      dim = *reinterpret_cast<const int64_t *>(axis_tensor.GetData() + i * sizeof(int64_t));
    } else {
      GE_WARN_ASSERT(false, "Skip lowering node %s, as: ExpandDims must be int32 or int64", node->GetNamePtr());
    }
  }

  x = loop::Unsqueeze(x, dim);
  loop::Store(node->GetOutDataAnchor(0), x);
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(Select) {
  // 如果cond shape 只有一维且和input0的shape[0]相同时，先进行unsqueeze再和其他输入进行broadcast
  loop::Index broadcasted;
  std::vector<loop::Index> indices;

  GE_ASSERT_NOTNULL(node->GetInDataAnchor(0));
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(1));
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(2));
  auto cond = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  auto input0 = node->GetInDataAnchor(1)->GetPeerOutAnchor();
  auto input1 = node->GetInDataAnchor(2)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(cond);
  GE_ASSERT_NOTNULL(input0);
  GE_ASSERT_NOTNULL(input1);
  auto cond_desc = cond->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(cond->GetIdx());
  auto input0_desc = input0->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(input0->GetIdx());
  auto input1_desc = input1->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(input1->GetIdx());
  GE_ASSERT_NOTNULL(cond_desc);
  GE_ASSERT_NOTNULL(input0_desc);
  GE_ASSERT_NOTNULL(input1_desc);
  std::vector<Expression> cond_dims;
  std::vector<Expression> input0_dims;
  std::vector<Expression> input1_dims;
  GE_WARN_ASSERT(loop::GetBufferShape(cond, cond_dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input0 shape.", node->GetNamePtr());
  GE_WARN_ASSERT(loop::GetBufferShape(input0, input0_dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input1 shape.", node->GetNamePtr());
  GE_WARN_ASSERT(loop::GetBufferShape(input1, input1_dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input2 shape.", node->GetNamePtr());

  auto cond_tensor = loop::Load(node->GetInDataAnchor(0));
  if (cond_dims.size() == 1 && input0_dims.size() > 1 && cond_dims[0] == input0_dims[0]) {
    for (size_t i = cond_dims.size(); i < input0_dims.size(); i++) {
      cond_tensor = loop::Unsqueeze(cond_tensor, static_cast<int64_t>(i));
      cond_dims.emplace_back(Symbol(1));
    }
  }
  indices.emplace_back(cond_dims);
  indices.emplace_back(input0_dims);
  indices.emplace_back(input1_dims);

  GE_ASSERT_GRAPH_SUCCESS(Broadcast(indices, broadcasted));
  constexpr size_t input1_idx = 2U;
  auto cond_var = loop::Broadcast(cond_tensor, indices[0], broadcasted);
  auto input1_var = loop::Broadcast(loop::Load(node->GetInDataAnchor(1)), indices[1], broadcasted);
  auto input2_var = loop::Broadcast(loop::Load(node->GetInDataAnchor(input1_idx)), indices[input1_idx], broadcasted);
  loop::Store(node->GetOutDataAnchor(0), loop::Where(cond_var, input1_var, input2_var));
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(ApplyAdagradD) {
  loop::Index broadcasted_var;
  std::vector<loop::Index> indices_var;
  vector<InDataAnchorPtr> indata_anchors_var;
  GE_WARN_ASSERT(node->GetAllInDataAnchorsSize() == 4U,
                 "Skip lowering node %s, as: Input size is not 4.", node->GetNamePtr());
  for (auto &in_anchor : node->GetAllInDataAnchors()) {
    indata_anchors_var.emplace_back(in_anchor);
  }

  GE_WARN_ASSERT_GRAPH_SUCCESS(BroadCastByInDataAnchors(indata_anchors_var, indices_var, broadcasted_var));
  auto accum = loop::Load(node->GetInDataAnchor(1));
  auto grad = loop::Load(node->GetInDataAnchor(3));
  bool update_slots = false;
  (void)AttrUtils::GetBool(node->GetOpDesc(), "update_slots", update_slots);
  if (update_slots) {
    loop::Index broadcasted_accum;
    std::vector<loop::Index> indices_accum;
    vector<InDataAnchorPtr> indata_anchors_accum;
    indata_anchors_accum.emplace_back(node->GetInDataAnchor(1));
    indata_anchors_accum.emplace_back(node->GetInDataAnchor(3));
    GE_WARN_ASSERT_GRAPH_SUCCESS(BroadCastByInDataAnchors(indata_anchors_accum, indices_accum, broadcasted_accum));
    accum = loop::Add(loop::Broadcast(accum, indices_accum[0], broadcasted_accum),
                      loop::Broadcast(loop::Mul(grad, grad), indices_accum[1], broadcasted_accum));
    loop::Store(node->GetOutDataAnchor(1), accum);
    accum = loop::Broadcast(accum, broadcasted_accum, broadcasted_var);
  } else {
    loop::Store(node->GetOutDataAnchor(1), accum);
    accum = loop::Broadcast(accum, indices_var[1], broadcasted_var);
  }
  auto var = loop::Broadcast(loop::Load(node->GetInDataAnchor(0)), indices_var[0], broadcasted_var);
  auto lr = loop::Broadcast(loop::Load(node->GetInDataAnchor(2)), indices_var[2], broadcasted_var);
  auto update_step =
      loop::Div(loop::Mul(lr, loop::Broadcast(grad, indices_var[3], broadcasted_var)), loop::Sqrt(accum));
  var = loop::Sub(var, update_step);
  loop::Store(node->GetOutDataAnchor(0), var);
  return GRAPH_SUCCESS;
}

graphStatus CalculateApplyAdamDScalarStr(const NodePtr &node, string &sub_beta1, string &sub_beta2, string &lr_str) {
  const auto op = ge::OpDescUtils::CreateOperatorFromNode(node);
  ge::Tensor beta1_power, beta2_power, beta1, beta2, lr;
  GE_WARN_ASSERT(op.GetInputConstData("beta1_power", beta1_power) == ge::SUCCESS,
                 "Skip lowering node %s %s as failed to get tensor", node->GetNamePtr(), node->GetTypePtr());
  GE_WARN_ASSERT(op.GetInputConstData("beta2_power", beta2_power) == ge::SUCCESS,
                 "Skip lowering node %s %s as failed to get tensor", node->GetNamePtr(), node->GetTypePtr());
  GE_WARN_ASSERT(op.GetInputConstData("beta1", beta1) == ge::SUCCESS,
                 "Skip lowering node %s %s as failed to get tensor", node->GetNamePtr(), node->GetTypePtr());
  GE_WARN_ASSERT(op.GetInputConstData("beta2", beta2) == ge::SUCCESS,
                 "Skip lowering node %s %s as failed to get tensor", node->GetNamePtr(), node->GetTypePtr());
  GE_WARN_ASSERT(op.GetInputConstData("lr", lr) == ge::SUCCESS, "Skip lowering node %s %s as failed to get tensor",
                 node->GetNamePtr(), node->GetTypePtr());

  vector<ge::Tensor> inputs_sub1 = {beta1};
  vector<ge::Tensor> inputs_sub2 = {beta2};
  vector<ge::Tensor> inputs_lr = {lr, beta1_power, beta2_power};
  sub_beta1 = ProcessScalarTensor(inputs_sub1, [](const vector<TensorValue> &scalar_tensors) -> string {
    return std::visit(
        [](auto &&beta1) -> string {
          using tensor_type = std::decay_t<decltype(beta1)>;
          tensor_type result = static_cast<tensor_type>(1) - beta1;
          return std::to_string(result);
        },
        scalar_tensors[0]);
  });
  sub_beta2 = ProcessScalarTensor(inputs_sub2, [](const vector<TensorValue> &scalar_tensors) -> string {
    return std::visit(
        [](auto &&beta2) -> string {
          using tensor_type = std::decay_t<decltype(beta2)>;
          tensor_type result = static_cast<tensor_type>(1) - beta2;
          return std::to_string(result);
        },
        scalar_tensors[0]);
  });
  lr_str = ProcessScalarTensor(inputs_lr, [](const vector<TensorValue> &scalar_tensors) -> string {
    constexpr size_t beta2_power_idx = 2;
    return std::visit(
        [](auto &&lr, auto &&beta1_power, auto &&beta2_power) -> string {
          using tensor_type = std::decay_t<decltype(lr)>;
          if (static_cast<tensor_type>(beta1_power) - static_cast<tensor_type>(1) < 1e-7 &&
              static_cast<tensor_type>(beta1_power) - static_cast<tensor_type>(1) > -1e-7) {
            GELOGW("ApplyAdamD beta1_power must not be equal to 1");
            return "";
          }
          tensor_type result =
              static_cast<tensor_type>(lr * (std::sqrt(static_cast<tensor_type>(1) - beta2_power)) / (static_cast<tensor_type>(1) - beta1_power));
          return std::to_string(result);
        },
        scalar_tensors[0], scalar_tensors[1], scalar_tensors[beta2_power_idx]);
  });
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(ApplyAdamD) {
  auto src = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  GE_WARN_ASSERT(src != nullptr,
                 "Skip lowering node %s, as: Input anchor is nullptr.", node->GetNamePtr());
  auto desc = src->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(src->GetIdx());
  GE_WARN_ASSERT(desc != nullptr,
                 "Skip lowering node %s, as: Input opdesc is nullptr.", node->GetNamePtr());
  auto dtype = desc->GetDataType();
  std::vector<ge::Expression> dims;
  GE_WARN_ASSERT(loop::GetBufferShape(src, dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input shape.", node->GetNamePtr());
  string sub_beta1_str, sub_beta2_str, lr_str;
  GE_WARN_ASSERT(CalculateApplyAdamDScalarStr(node, sub_beta1_str, sub_beta2_str, lr_str) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to calculate applyAdamD scalar str.", node->GetNamePtr());
  ;
  GE_WARN_ASSERT(!sub_beta1_str.empty() && !sub_beta2_str.empty() && !lr_str.empty(),
                 "Skip lowering node %s, as: Some scalar input is empty.", node->GetNamePtr());

  std::vector<loop::BroadcastOp::DimKind> status(dims.size(), loop::BroadcastOp::DimKind::NEW_AXIS);
  status.resize(dims.size(), loop::BroadcastOp::DimKind::NEW_AXIS);
  auto var = loop::Load(node->GetInDataAnchor(0));
  auto m = loop::Load(node->GetInDataAnchor(1));
  auto v = loop::Load(node->GetInDataAnchor(2));
  auto beta1 = loop::Broadcast(loop::Load(node->GetInDataAnchor(6)), status);
  auto sub_beta1 = ScalarBroadcast2Size(sub_beta1_str, dtype, dims.size());
  auto sub_beta2 = ScalarBroadcast2Size(sub_beta2_str, dtype, dims.size());
  auto epsilon = loop::Broadcast(loop::Load(node->GetInDataAnchor(8)), status);
  auto grad = loop::Load(node->GetInDataAnchor(9));
  bool use_nesterov = false;
  (void)AttrUtils::GetBool(node->GetOpDesc(), "use_nesterov", use_nesterov);

  m = loop::Add(m, loop::Mul(sub_beta1, loop::Sub(grad, m)));
  auto lr = ScalarBroadcast2Size(lr_str, dtype, dims.size());
  v = loop::Add(v, loop::Mul(sub_beta2, loop::Sub(loop::Mul(grad, grad), v)));
  if (use_nesterov) {
    var = loop::Sub(var, loop::Mul(lr, loop::Div(loop::Add(loop::Mul(m, beta1), loop::Mul(sub_beta1, grad)),
                                                 loop::Add(epsilon, loop::Sqrt(v)))));
  } else {
    var = loop::Sub(var, loop::Mul(lr, loop::Div(m, loop::Add(epsilon, loop::Sqrt(v)))));
  }

  loop::Store(node->GetOutDataAnchor(0), var);
  loop::Store(node->GetOutDataAnchor(1), m);
  loop::Store(node->GetOutDataAnchor(2), v);
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(ApplyGradientDescent) {
  loop::Index broadcasted;
  std::vector<loop::Index> indices;
  vector<InDataAnchorPtr> indata_anchors;
  for (auto &in_anchor : node->GetAllInDataAnchors()) {
    indata_anchors.emplace_back(in_anchor);
  }
  GE_WARN_ASSERT_GRAPH_SUCCESS(BroadCastByInDataAnchors(indata_anchors, indices, broadcasted));
  GE_WARN_ASSERT(indices.size() == 3,
                 "Skip lowering node %s, as: Input size is not 3.", node->GetNamePtr());
  auto var = loop::Load(node->GetInDataAnchor(0));
  auto alpha = loop::Broadcast(loop::Load(node->GetInDataAnchor(1)), indices[1], broadcasted);
  auto delta = loop::Broadcast(loop::Load(node->GetInDataAnchor(2)), indices[2], broadcasted);

  var = loop::Sub(var, loop::Mul(delta, alpha));
  loop::Store(node->GetOutDataAnchor(0), var);
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(Elu) {
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(0));
  const auto src = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src);
  GE_ASSERT_NOTNULL(src->GetOwnerNode());
  GE_ASSERT_NOTNULL(src->GetOwnerNode()->GetOpDesc());
  const auto desc = src->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(src->GetIdx());
  GE_ASSERT_NOTNULL(desc);
  const auto dtype = desc->GetDataType();
  std::vector<Expression> dims;
  GE_WARN_ASSERT(loop::GetBufferShape(src, dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input0 shape.", node->GetNamePtr());
  const auto x = loop::Load(node->GetInDataAnchor(0));
  float alpha = 1.0f;
  float scale = 1.0f;
  float input_scale = 1.0f;
  (void)AttrUtils::GetFloat(node->GetOpDesc(), "alpha", alpha);
  (void)AttrUtils::GetFloat(node->GetOpDesc(), "scale", scale);
  (void)AttrUtils::GetFloat(node->GetOpDesc(), "input_scale", input_scale);

  std::ostringstream oss;
  oss.precision(7);            // 设置小数点后位数
  oss << std::fixed << alpha;  // 固定小数格式
  std::string alpha_str = oss.str();

  std::ostringstream oss1;
  oss1.precision(7);            // 设置小数点后位数
  oss1 << std::fixed << scale;  // 固定小数格式
  std::string scale_str = oss1.str();

  std::ostringstream oss2;
  oss2.precision(7);                  // 设置小数点后位数
  oss2 << std::fixed << input_scale;  // 固定小数格式
  std::string input_scale_str = oss2.str();

  const loop::LoopVar scalar_zero_var = loop::Scalar("0", dtype);
  const loop::LoopVar scalar_one_var = loop::Scalar("1", dtype);
  const auto scalar_alpha_var = loop::Scalar(alpha_str, dtype);
  const auto scalar_scale_var = loop::Scalar(scale_str, dtype);
  const auto scalar_input_scale_var = loop::Scalar(input_scale_str, dtype);

  std::vector<loop::BroadcastOp::DimKind> status(dims.size(), loop::BroadcastOp::DimKind::NEW_AXIS);
  status.resize(dims.size(), loop::BroadcastOp::DimKind::NEW_AXIS);
  auto zero = loop::Broadcast(scalar_zero_var, status);
  auto one = loop::Broadcast(scalar_one_var, status);
  auto alpha_var = loop::Broadcast(scalar_alpha_var, status);
  auto scale_var = loop::Broadcast(scalar_scale_var, status);
  auto input_scale_var = loop::Broadcast(scalar_input_scale_var, status);

  auto negRes = loop::Sub(loop::Exp(loop::Mul(loop::Minimum(x, zero), input_scale_var)), one);
  const auto y = loop::Mul(loop::Add(loop::Mul(negRes, alpha_var), loop::Maximum(x, zero)), scale_var);
  loop::Store(node->GetOutDataAnchor(0), y);
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(EluGrad) {
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(0));
  const auto grads = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(grads);
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(1));
  const auto activate = node->GetInDataAnchor(1)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(activate);
  GE_ASSERT_NOTNULL(activate->GetOwnerNode());
  GE_ASSERT_NOTNULL(activate->GetOwnerNode()->GetOpDesc());
  const auto desc = activate->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(activate->GetIdx());
  GE_ASSERT_NOTNULL(desc);
  const auto dtype = desc->GetDataType();

  std::vector<Expression> grads_dims;
  std::vector<Expression> activate_dims;
  GE_WARN_ASSERT(loop::GetBufferShape(grads, grads_dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input0 shape.", node->GetNamePtr());
  GE_WARN_ASSERT(loop::GetBufferShape(activate, activate_dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input1 shape.", node->GetNamePtr());
  loop::Index broadcast;
  std::vector<loop::Index> indices;
  indices.emplace_back(grads_dims);
  indices.emplace_back(activate_dims);
  GE_ASSERT_GRAPH_SUCCESS(Broadcast(indices, broadcast));
  const auto dims_size = std::max(grads_dims.size(), activate_dims.size());
  std::vector<loop::BroadcastOp::DimKind> status(dims_size, loop::BroadcastOp::DimKind::NEW_AXIS);
  status.resize(dims_size, loop::BroadcastOp::DimKind::NEW_AXIS);
  const auto grads_var = loop::Broadcast(loop::Load(node->GetInDataAnchor(0)), indices[0], broadcast);
  const auto activate_var = loop::Broadcast(loop::Load(node->GetInDataAnchor(1)), indices[1], broadcast);
  const loop::LoopVar scalar_zero_var = loop::Scalar("0", dtype);
  const auto zero = loop::Broadcast(scalar_zero_var, status);
  const loop::LoopVar scalar_one_var = loop::Scalar("1", dtype);
  const auto one = loop::Broadcast(scalar_one_var, status);

  auto min_res = loop::Minimum(activate_var, zero);
  auto add_res = loop::Add(min_res, one);
  loop::Store(node->GetOutDataAnchor(0), loop::Mul(add_res, grads_var));
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(TanhGrad) {
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(0));
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(1));
  const auto input_y = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  const auto input_dy = node->GetInDataAnchor(1)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(input_y);
  GE_ASSERT_NOTNULL(input_dy);
  GE_ASSERT_NOTNULL(input_y->GetOwnerNode());
  GE_ASSERT_NOTNULL(input_dy->GetOwnerNode());
  GE_ASSERT_NOTNULL(input_y->GetOwnerNode()->GetOpDesc());
  GE_ASSERT_NOTNULL(input_dy->GetOwnerNode()->GetOpDesc());
  const auto desc0 = input_y->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(input_y->GetIdx());
  const auto desc1 = input_dy->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(input_dy->GetIdx());
  GE_ASSERT_NOTNULL(desc0);
  GE_ASSERT_NOTNULL(desc1);
  const auto dtype0 = desc0->GetDataType();
  GE_ASSERT(dtype0 != DT_COMPLEX32 && dtype0 != DT_COMPLEX64 && dtype0 != DT_COMPLEX128);
  const auto dtype1 = desc1->GetDataType();
  GE_ASSERT(dtype1 != DT_COMPLEX32 && dtype1 != DT_COMPLEX64 && dtype1 != DT_COMPLEX128);
  GE_WARN_ASSERT(dtype0 == dtype1,
                 "Skip lowering node %s, as: The shape of y and dy must the same.", node->GetNamePtr());

  std::vector<Expression> y_dims;
  std::vector<Expression> dy_dims;
  GE_WARN_ASSERT(loop::GetBufferShape(input_y, y_dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input_y shape", node->GetNamePtr());
  GE_WARN_ASSERT(loop::GetBufferShape(input_dy, dy_dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input_dy shape", node->GetNamePtr());
  loop::Index broadcast;
  std::vector<loop::Index> indices;
  indices.emplace_back(y_dims);
  indices.emplace_back(dy_dims);
  GE_ASSERT_GRAPH_SUCCESS(Broadcast(indices, broadcast));
  const auto y = loop::Broadcast(loop::Load(node->GetInDataAnchor(0)), indices[0], broadcast);
  const auto dy = loop::Broadcast(loop::Load(node->GetInDataAnchor(1)), indices[1], broadcast);
  const auto dims_size = std::max(y_dims.size(), dy_dims.size());
  std::vector<loop::BroadcastOp::DimKind> status(dims_size, loop::BroadcastOp::DimKind::NEW_AXIS);
  status.resize(dims_size, loop::BroadcastOp::DimKind::NEW_AXIS);

  const loop::LoopVar scalar_one_var = loop::Scalar("1", dtype0);
  const auto one = loop::Broadcast(scalar_one_var, status);

  loop::Store(node->GetOutDataAnchor(0), loop::Mul(dy, loop::Sub(one, loop::Mul(y, y))));
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(FusedMulAddN) {
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(0));
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(1));
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(2));
  auto input0 = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  auto input1 = node->GetInDataAnchor(1)->GetPeerOutAnchor();
  auto input2 = node->GetInDataAnchor(2)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(input0);
  GE_ASSERT_NOTNULL(input1);
  GE_ASSERT_NOTNULL(input2);
  std::vector<Expression> input0_exp;
  std::vector<Expression> input1_exp;
  std::vector<Expression> input2_exp;
  GE_WARN_ASSERT(loop::GetBufferShape(input0, input0_exp) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input0 shape", node->GetNamePtr());
  GE_WARN_ASSERT(loop::GetBufferShape(input1, input1_exp) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input1 shape", node->GetNamePtr());
  GE_WARN_ASSERT(loop::GetBufferShape(input2, input2_exp) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input2 shape", node->GetNamePtr());
  loop::Index broadcasted;
  std::vector<loop::Index> indices;
  indices.emplace_back(input0_exp);
  indices.emplace_back(input1_exp);
  indices.emplace_back(input2_exp);
  GE_ASSERT_GRAPH_SUCCESS(Broadcast(indices, broadcasted));
  auto x1 = loop::Broadcast(loop::Load(node->GetInDataAnchor(0)), indices[0], broadcasted);
  auto x2 = loop::Broadcast(loop::Load(node->GetInDataAnchor(1)), indices[1], broadcasted);
  auto x3 = loop::Broadcast(loop::Load(node->GetInDataAnchor(2)), indices[2], broadcasted);
  auto mul_res = loop::Mul(x1, x3);
  auto add_res = loop::Add(mul_res, x2);
  loop::Store(node->GetOutDataAnchor(0), add_res);
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(L2Loss) {
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(0));
  auto src = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src);
  GE_ASSERT_NOTNULL(src->GetOwnerNode());
  GE_ASSERT_NOTNULL(src->GetOwnerNode()->GetOpDesc());
  auto desc = src->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(src->GetIdx());
  GE_ASSERT_NOTNULL(desc);
  auto dtype = desc->GetDataType();
  auto x1 = loop::Load(node->GetInDataAnchor(0));
  auto square_res = loop::Mul(x1, x1);
  vector<Expression> dims;
  GE_WARN_ASSERT(loop::GetBufferShape(src, dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input shape", node->GetNamePtr());
  std::vector<loop::BroadcastOp::DimKind> status(dims.size(), loop::BroadcastOp::DimKind::NEW_AXIS);
  status.resize(dims.size(), loop::BroadcastOp::DimKind::NEW_AXIS);
  loop::LoopVar scalar_value = loop::Scalar("0.5", dtype);
  auto scalar_value_brc = loop::Broadcast(scalar_value, status);
  auto mul_res = loop::Mul(square_res, scalar_value_brc);
  std::vector<size_t> reduced_axis;
  for (size_t i = 0U; i < dims.size(); i++) {
    reduced_axis.emplace_back(i);
  }
  loop::StoreReduction(loop::ReduceType::SUM, node->GetOutDataAnchor(0), mul_res, dims, reduced_axis);
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(BNInferenceD) {
  GE_WARN_ASSERT(node->GetInDataAnchor(0) != nullptr && node->GetInDataAnchor(1) != nullptr &&
                 node->GetInDataAnchor(2) != nullptr);
  bool exist_scale = node->GetInDataAnchor(3) != nullptr && node->GetInDataAnchor(3)->GetPeerOutAnchor() != nullptr;
  bool exist_b = node->GetInDataAnchor(4) != nullptr && node->GetInDataAnchor(4)->GetPeerOutAnchor() != nullptr;

  std::vector<loop::Index> indices_var;
  loop::Index broadcasted_var;
  auto x = loop::Load(node->GetInDataAnchor(0));
  if (exist_scale && exist_b) {
    vector<InDataAnchorPtr> indata_anchors_var = {node->GetInDataAnchor(0), node->GetInDataAnchor(1),
                                                  node->GetInDataAnchor(2), node->GetInDataAnchor(3),
                                                  node->GetInDataAnchor(4)};
    GE_WARN_ASSERT_GRAPH_SUCCESS(BroadCastByInDataAnchors(indata_anchors_var, indices_var, broadcasted_var));
    auto mean = loop::Broadcast(loop::Load(node->GetInDataAnchor(1)), indices_var[1], broadcasted_var);
    auto variance = loop::Broadcast(loop::Load(node->GetInDataAnchor(2)), indices_var[2], broadcasted_var);
    auto scale = loop::Broadcast(loop::Load(node->GetInDataAnchor(3)), indices_var[3], broadcasted_var);
    auto b = loop::Broadcast(loop::Load(node->GetInDataAnchor(4)), indices_var[4], broadcasted_var);
    loop::Store(node->GetOutDataAnchor(0), loop::Add(loop::Mul(loop::Mul(variance, loop::Add(x, mean)), scale), b));
  } else if (exist_scale) {
    vector<InDataAnchorPtr> indata_anchors_var = {node->GetInDataAnchor(0), node->GetInDataAnchor(1),
                                                  node->GetInDataAnchor(2), node->GetInDataAnchor(3)};
    GE_WARN_ASSERT_GRAPH_SUCCESS(BroadCastByInDataAnchors(indata_anchors_var, indices_var, broadcasted_var));
    GE_WARN_ASSERT(indices_var.size() == 4);
    auto mean = loop::Broadcast(loop::Load(node->GetInDataAnchor(1)), indices_var[1], broadcasted_var);
    auto variance = loop::Broadcast(loop::Load(node->GetInDataAnchor(2)), indices_var[2], broadcasted_var);
    auto scale = loop::Broadcast(loop::Load(node->GetInDataAnchor(3)), indices_var[3], broadcasted_var);
    loop::Store(node->GetOutDataAnchor(0), loop::Mul(loop::Mul(variance, loop::Add(x, mean)), scale));
  } else {
    vector<InDataAnchorPtr> indata_anchors_var = {node->GetInDataAnchor(0), node->GetInDataAnchor(1),
                                                  node->GetInDataAnchor(2)};
    GE_WARN_ASSERT_GRAPH_SUCCESS(BroadCastByInDataAnchors(indata_anchors_var, indices_var, broadcasted_var));
    GE_WARN_ASSERT(indices_var.size() == 3);
    auto mean = loop::Broadcast(loop::Load(node->GetInDataAnchor(1)), indices_var[1], broadcasted_var);
    auto variance = loop::Broadcast(loop::Load(node->GetInDataAnchor(2)), indices_var[2], broadcasted_var);
    loop::Store(node->GetOutDataAnchor(0), loop::Mul(variance, loop::Add(x, mean)));
  }
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(Axpy) {
  loop::Index broadcasted;
  std::vector<loop::Index> indices;
  vector<InDataAnchorPtr> indata_anchors;
  for (auto &in_anchor : node->GetAllInDataAnchors()) {
    indata_anchors.emplace_back(in_anchor);
  }

  GE_WARN_ASSERT_GRAPH_SUCCESS(BroadCastByInDataAnchors(indata_anchors, indices, broadcasted));
  GE_WARN_ASSERT(indices.size() == 2,
                 "Skip lowering node %s, as: Input size is not 2.", node->GetNamePtr());
  auto x1 = loop::Broadcast(loop::Load(node->GetInDataAnchor(0)), indices[0], broadcasted);
  auto x2 = loop::Broadcast(loop::Load(node->GetInDataAnchor(1)), indices[1], broadcasted);
  float32_t alpha = 0.8;
  const auto &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  AttrUtils::GetFloat(op_desc, "alpha", alpha);
  loop::Store(node->GetOutDataAnchor(0), loop::Axpy(x1, x2, alpha));
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(Log1p) {
  GE_ASSERT_NOTNULL(node->GetInDataAnchor(0));
  auto src = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src);
  GE_ASSERT_NOTNULL(src->GetOwnerNode());
  GE_ASSERT_NOTNULL(src->GetOwnerNode()->GetOpDesc());
  auto desc = src->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(src->GetIdx());
  GE_ASSERT_NOTNULL(desc);
  auto dtype = desc->GetDataType();
  GE_WARN_ASSERT((dtype == DT_FLOAT) || (dtype == DT_FLOAT16),
                 "Skip lowering node %s, as: dtype only support FLOAT16 and FLOAT.", node->GetNamePtr());
  std::vector<Expression> dims;
  GE_WARN_ASSERT(loop::GetBufferShape(src, dims) == GRAPH_SUCCESS,
                 "Skip lowering node %s, as: Failed to get input shape.", node->GetNamePtr());
  auto x = loop::Load(node->GetInDataAnchor(0));
  auto one = ScalarBroadcast2Size("1.0", dtype, dims.size());
  auto neg_one = ScalarBroadcast2Size("-1.0", dtype, dims.size());
  auto inf = ScalarBroadcast2Size("inf", dtype, dims.size());
  auto input_add_one = loop::Add(x, one);
  auto input_mid = loop::Add(input_add_one, neg_one);
  input_mid = loop::Div(x, input_mid);
  auto output = loop::Ln(input_add_one);
  output = loop::Mul(output, input_mid);
  output = loop::Where(loop::Ne(input_add_one, one), output, x);
  output = loop::Where(loop::Ne(input_add_one, inf), output, input_mid);
  loop::Store(node->GetOutDataAnchor(0), output);
  return GRAPH_SUCCESS;
}
}

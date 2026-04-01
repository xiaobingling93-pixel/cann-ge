/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include "common/util/mem_utils.h"
#include "common/util.h"
#include "common/checker.h"
#include "framework/common/types.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/compute_graph.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"

namespace ge {
namespace {
constexpr size_t kXInputIndex = 0UL;
constexpr size_t kOutputIndex = 0UL;
constexpr size_t kAttrStartInputIndex = 0UL;
constexpr size_t kAttrEndInputIndex = 1UL;
constexpr size_t kAttrStridesInputIndex = 2UL;
constexpr size_t kAttrDBeginMaskIndex = 3UL;
constexpr size_t kAttrDEndMaskIndex = 4UL;
constexpr size_t kAttrDEllipsisMaskIndex = 5UL;
constexpr size_t kAttrDNewAxisMaskIndex = 6UL;
constexpr size_t kAttrDShrinkAxisMaskIndex = 7UL;

constexpr size_t kStartInputIndex = 1UL;
constexpr size_t kEndInputIndex = 2UL;
constexpr size_t kStridesInputIndex = 3UL;
constexpr size_t kAttrBeginMaskIndex = 0UL;
constexpr size_t kAttrEndMaskIndex = 1UL;
constexpr size_t kAttrEllipsisMaskIndex = 2UL;
constexpr size_t kAttrNewAxisMaskIndex = 3UL;
constexpr size_t kAttrShrinkAxisMaskIndex = 4UL;

constexpr size_t kAxesV2InputIndex = 3UL;
constexpr size_t kStridesV2InputIndex = 4UL;

struct StridedSliceAttr {
  int64_t begin_mask{0};
  int64_t end_mask{0};
  int64_t ellipsis_mask{0};
  int64_t new_axis_mask{0};
  int64_t shrink_axis_mask{0};
};

struct StrdedSliceIndexInputs {
  std::vector<Expression> start_indexes;
  std::vector<Expression> end_indexes;
  std::vector<Expression> strides_indexes;
  std::vector<bool> is_new_axis;
};

// 如果begin和end是负数的时候，需要调整为正数
Status NormalizeInput(std::vector<Expression> &input_indexes, const std::vector<Expression> &input_dims) {
  GE_ASSERT_TRUE(input_indexes.size() <= input_dims.size(),
                 "input indexes size: %zu should not more than input shape size: %zu", input_indexes.size(),
                 input_dims.size());
  for (size_t i = 0UL; i < input_indexes.size(); i++) {
    const bool lt_zero = EXPECT_SYMBOL_LT(input_indexes[i], kSymbolZero);
    input_indexes[i] = (lt_zero == true) ? input_indexes[i] + input_dims[i] : input_indexes[i];
  }
  return SUCCESS;
}

// 如果new_axis_mask和shrink_axis_mask的bit位与ellipsis_mask冲突，则不生效
void HandleMaskConflict(StridedSliceAttr &strided_slice_attr) {
  strided_slice_attr.new_axis_mask = ((static_cast<uint64_t>(strided_slice_attr.new_axis_mask) &
                                       static_cast<uint64_t>(strided_slice_attr.ellipsis_mask)) ^
                                      static_cast<uint64_t>(strided_slice_attr.new_axis_mask));
  strided_slice_attr.shrink_axis_mask = ((static_cast<uint64_t>(strided_slice_attr.shrink_axis_mask) &
                                          static_cast<uint64_t>(strided_slice_attr.ellipsis_mask)) ^
                                         static_cast<uint64_t>(strided_slice_attr.shrink_axis_mask));
  strided_slice_attr.shrink_axis_mask = ((static_cast<uint64_t>(strided_slice_attr.shrink_axis_mask) &
                                          static_cast<uint64_t>(strided_slice_attr.new_axis_mask)) ^
                                         static_cast<uint64_t>(strided_slice_attr.shrink_axis_mask));
  GELOGI("handle mask conflict, new_axis_mask: %lld, shrink_axis_mask: %lld", strided_slice_attr.new_axis_mask,
         strided_slice_attr.shrink_axis_mask);
}

int64_t CountBitNum(const int64_t num) {
  int64_t count = 0L;
  if (num <= 0) {
    return count;
  }
  for (uint64_t n = num; n > 0; n >>= 1) {
    count += (n & 1L);
  }
  return count;
}

bool IsInEllipsisMaskRange(const std::pair<int64_t, int64_t> &ellipsis_mask_range, const int64_t pos) {
  return ((pos >= ellipsis_mask_range.first) && (pos < ellipsis_mask_range.second));
}

Status AppendNewAxis(const std::pair<int64_t, int64_t> &ellipsis_mask_range, const int64_t new_axis_mask,
                     const std::vector<Expression> &input_dims, std::vector<Expression> &input_append_axis_shape,
                     StrdedSliceIndexInputs &index_input) {
  const size_t begin_len = index_input.start_indexes.size();
  int64_t new_axis_num = 0L;
  for (size_t i = 0UL; i < begin_len; ++i) {
    if ((static_cast<size_t>(new_axis_mask) & (1 << i)) > 0) {
      new_axis_num++;
    }
  }
  int64_t mask_pos = 0L;
  for (size_t i = 0UL; i < input_dims.size();) {
    if ((static_cast<size_t>(new_axis_mask) & (1 << mask_pos)) > 0) {
      if (IsInEllipsisMaskRange(ellipsis_mask_range, static_cast<int64_t>(input_append_axis_shape.size()))) {
        input_append_axis_shape.emplace_back(input_dims[i++]);
        index_input.is_new_axis.emplace_back(false);
      } else {
        new_axis_num--;
        input_append_axis_shape.emplace_back(Symbol(1));
        index_input.is_new_axis.emplace_back(true);
        mask_pos++;
      }
    } else {
      input_append_axis_shape.emplace_back(input_dims[i++]);
      index_input.is_new_axis.emplace_back(false);
      mask_pos++;
    }
  }
  while (0L != new_axis_num) {
    input_append_axis_shape.emplace_back(Symbol(1));
    index_input.is_new_axis.emplace_back(true);
    new_axis_num--;
  }
  GELOGI("Input shape after insert new axis: %s",
         SymbolicInferUtil::VectorExpressionToStr(input_append_axis_shape).c_str());
  return SUCCESS;
}

std::pair<int64_t, int64_t> GetEllipsisMaskRange(const StridedSliceAttr &strided_slice_attr,
                                                 const int64_t slice_dim_num, const int64_t input_size) {
  const int64_t bit_count = CountBitNum(strided_slice_attr.new_axis_mask);
  const int64_t ellipsis_mask_num = input_size + bit_count - slice_dim_num + 1;
  int64_t pos = 0L;
  for (; pos < slice_dim_num; pos++) {
    if ((static_cast<uint64_t>(strided_slice_attr.ellipsis_mask) &
        (1UL << static_cast<uint64_t>(pos))) > 0) {
      break;
    }
  }
  // 左开右闭
  if (pos == slice_dim_num) {
    // 未设置ellipsis_mask
    return std::make_pair(-1, -1);
  }
  GELOGI("ellipsis_mask_range: [%lld, %lld)", pos, pos + ellipsis_mask_num);
  return std::make_pair(pos, pos + ellipsis_mask_num);
}

Status HandleEllipsisMask(const int64_t ellipsis_mask_index, const std::vector<Expression> &input_dims,
                          StrdedSliceIndexInputs &index_input) {
  GE_ASSERT_TRUE(index_input.start_indexes.size() == index_input.end_indexes.size(),
                 "start_index size: %zu should equal to end_index size:%zu", index_input.start_indexes.size(),
                 index_input.end_indexes.size());
  GE_ASSERT_TRUE(index_input.start_indexes.size() == index_input.strides_indexes.size(),
                 "start_index size: %zu should equal to strides_index size:%zu", index_input.start_indexes.size(),
                 index_input.strides_indexes.size());
  for (size_t i = 0UL; i < index_input.start_indexes.size(); i++) {
    if (static_cast<int64_t>(i) == ellipsis_mask_index) {
      index_input.start_indexes[i] = Symbol(0);
      index_input.end_indexes[i] = input_dims[i];
      index_input.strides_indexes[i] = Symbol(1);
      break;
    }
  }
  GELOGD("start index after insert handle ellipsis_mask: %s",
         SymbolicInferUtil::VectorExpressionToStr(index_input.start_indexes).c_str());
  GELOGD("end index after insert handle ellipsis_mask: %s",
         SymbolicInferUtil::VectorExpressionToStr(index_input.end_indexes).c_str());
  GELOGD("strides index after insert handle ellipsis_mask: %s",
         SymbolicInferUtil::VectorExpressionToStr(index_input.strides_indexes).c_str());
  return SUCCESS;
}

void GetShrinkAxisIndex(const int64_t shrink_axis_mask, const std::pair<int64_t, int64_t> &ellipsis_mask_range,
                        const int64_t index_size, std::set<int64_t> &shrink_axis_indexes) {
  int64_t bit_pos = 0L;
  for (int64_t i = 0L; i < index_size; i++) {
    if ((static_cast<size_t>(shrink_axis_mask) & (1 << static_cast<size_t>(bit_pos))) > 0) {
      if (IsInEllipsisMaskRange(ellipsis_mask_range, i)) {
        continue;
      }
      shrink_axis_indexes.insert(i);
    }
    bit_pos++;
  }
}

Status HandleShrinkAxisShape(const std::set<int64_t> &shrink_axis_indexes, StrdedSliceIndexInputs &index_input) {
  for (const auto &shrink_axis_id : shrink_axis_indexes) {
    GE_ASSERT_TRUE((shrink_axis_id < static_cast<int64_t>(index_input.start_indexes.size())) && (shrink_axis_id >= 0));
    index_input.end_indexes[shrink_axis_id] = index_input.start_indexes[shrink_axis_id] + Symbol(1);
    index_input.strides_indexes[shrink_axis_id] = Symbol(1);
  }
  return SUCCESS;
}

Status FillMissionIndex(const std::pair<int64_t, int64_t> &ellipsis_mask_range,
                        const std::vector<Expression> &input_dims, StrdedSliceIndexInputs &index_input) {
  std::vector<Expression> origin_start_indexes = index_input.start_indexes;
  std::vector<Expression> origin_end_indexes = index_input.end_indexes;
  std::vector<Expression> origin_strides_indexes = index_input.strides_indexes;
  GELOGD("origin_start_indexes before insert fill missing: %s",
         SymbolicInferUtil::VectorExpressionToStr(origin_start_indexes).c_str());
  GELOGD("origin_end_indexes before insert fill missing: %s",
         SymbolicInferUtil::VectorExpressionToStr(origin_end_indexes).c_str());
  GELOGD("origin_strides_indexes before insert fill missing: %s",
         SymbolicInferUtil::VectorExpressionToStr(origin_strides_indexes).c_str());
  auto ori_start_size = origin_start_indexes.size();
  for (size_t i = ori_start_size; i < input_dims.size(); i++) {
    origin_start_indexes.emplace_back(Symbol(0));
    origin_end_indexes.emplace_back(input_dims[i]);
    origin_strides_indexes.emplace_back(Symbol(1));
  }
  GE_ASSERT_SUCCESS(NormalizeInput(origin_start_indexes, input_dims));
  GE_ASSERT_SUCCESS(NormalizeInput(origin_end_indexes, input_dims));
  index_input.start_indexes.clear();
  index_input.end_indexes.clear();
  index_input.strides_indexes.clear();
  int64_t start_index_pos = 0L;
  for (size_t i = 0UL; i < input_dims.size(); i++) {
    if (IsInEllipsisMaskRange(ellipsis_mask_range, static_cast<int64_t>(i))) {
      index_input.start_indexes.emplace_back(Symbol(0));
      index_input.end_indexes.emplace_back(input_dims[i]);
      index_input.strides_indexes.emplace_back(Symbol(1));
      if (static_cast<int64_t>(i) == ellipsis_mask_range.first) {
        start_index_pos++;
      }
      continue;
    }
    if (EXPECT_SYMBOL_LT(origin_start_indexes[start_index_pos], input_dims[i])) {
      index_input.start_indexes.emplace_back(origin_start_indexes[start_index_pos]);
    } else {
      index_input.start_indexes.emplace_back(input_dims[i]);
    }
    if (EXPECT_SYMBOL_LT(origin_end_indexes[start_index_pos], input_dims[i])) {
      index_input.end_indexes.emplace_back(origin_end_indexes[start_index_pos]);
    } else {
      index_input.end_indexes.emplace_back(input_dims[i]);
    }
    index_input.strides_indexes.emplace_back(origin_strides_indexes[start_index_pos]);
    start_index_pos++;
  }
  GELOGD("start index after insert fill missing: %s",
         SymbolicInferUtil::VectorExpressionToStr(index_input.start_indexes).c_str());
  GELOGD("end index after insert handle fill missing: %s",
         SymbolicInferUtil::VectorExpressionToStr(index_input.end_indexes).c_str());
  GELOGD("strides index after insert handle fill missing: %s",
         SymbolicInferUtil::VectorExpressionToStr(index_input.strides_indexes).c_str());
  return ge::SUCCESS;
}

Status HandleBeginEndMask(const StridedSliceAttr &strided_slice_attr, const std::vector<Expression> &input_dims,
                          const std::pair<int64_t, int64_t> &ellipsis_mask_range, StrdedSliceIndexInputs &index_input) {
  int64_t mask_pos = 0L;
  for (size_t i = 0UL; i < index_input.start_indexes.size(); i++) {
    if (IsInEllipsisMaskRange(ellipsis_mask_range, static_cast<int64_t>(i))) {
      if (static_cast<int64_t>(i) == ellipsis_mask_range.first) {
        mask_pos++;
      }
      continue;
    }
    int64_t strides_value = 0L;
    GE_ASSERT_TRUE(index_input.strides_indexes[i].GetConstValue(strides_value));
    if ((static_cast<uint64_t>(strided_slice_attr.begin_mask) & (1 << mask_pos)) > 0) {
      index_input.start_indexes[i] = (strides_value > 0) ? Symbol(0) : input_dims[i] - Symbol(1);
    }
    if ((static_cast<uint64_t>(strided_slice_attr.end_mask) & (1 << mask_pos)) > 0) {
      index_input.end_indexes[i] = (strides_value > 0) ? input_dims[i] : Symbol(0);
    }
    mask_pos++;
  }
  GELOGI("start index after insert handle begin end mask: %s",
         SymbolicInferUtil::VectorExpressionToStr(index_input.start_indexes).c_str());
  GELOGI("end index after insert handle begin end mask: %s",
         SymbolicInferUtil::VectorExpressionToStr(index_input.end_indexes).c_str());
  GELOGI("strides index after insert handle begin end mask: %s",
         SymbolicInferUtil::VectorExpressionToStr(index_input.strides_indexes).c_str());
  return SUCCESS;
}

Status CalcOutputShape(const int64_t shrink_axis_mask, const std::pair<int64_t, int64_t> &ellipsis_mask_range,
                       StrdedSliceIndexInputs &index_input, std::vector<Expression> &output_symbols_shape) {
  std::set<int64_t> shrink_axis_indexes;
  GetShrinkAxisIndex(shrink_axis_mask, ellipsis_mask_range, static_cast<int64_t>(index_input.start_indexes.size()),
                     shrink_axis_indexes);
  // 处理ShrinkAxisIndex，将shrink axis的维度设置成[start, start + 1, 1]
  GE_ASSERT_SUCCESS(HandleShrinkAxisShape(shrink_axis_indexes, index_input));
  for (size_t i = 0UL; i < index_input.start_indexes.size(); i++) {
    if (shrink_axis_indexes.count(static_cast<int64_t>(i)) > 0) {
      continue;
    }
    GE_ASSERT_TRUE(index_input.strides_indexes[i] != kSymbolZero);
    Expression result_dim;
    if (EXPECT_SYMBOL_EQ(index_input.strides_indexes[i], kSymbolOne)) {
      result_dim = (index_input.end_indexes[i] - index_input.start_indexes[i]);
    } else {
      result_dim =
          sym::Ceiling((index_input.end_indexes[i] - index_input.start_indexes[i]) / index_input.strides_indexes[i]);
    }
    ASSERT_SYMBOL_GE(result_dim, kSymbolZero);
    auto output_dim = (index_input.is_new_axis[i] == true) ? Symbol(1) : result_dim;
    output_symbols_shape.emplace_back(output_dim);
  }
  return SUCCESS;
}

Status GetStridedSliceDIndexInput(const gert::InferSymbolShapeContext *context, StrdedSliceIndexInputs &index_input) {
  GE_CHECK_NOTNULL(context);
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);

  GE_ASSERT_NOTNULL(attrs->GetListInt(kAttrStartInputIndex));
  for (size_t i = 0UL; i < attrs->GetListInt(kAttrStartInputIndex)->GetSize(); i++) {
    const auto start_vec_ptr = attrs->GetListInt(kAttrStartInputIndex);
    GE_ASSERT_NOTNULL(start_vec_ptr);
    const int64_t start_value = start_vec_ptr->GetData()[i];
    index_input.start_indexes.push_back(Symbol(start_value));
  }

  GE_ASSERT_NOTNULL(attrs->GetListInt(kAttrEndInputIndex));
  for (size_t i = 0UL; i < attrs->GetListInt(kAttrEndInputIndex)->GetSize(); i++) {
    const auto end_vec_ptr = attrs->GetListInt(kAttrEndInputIndex);
    GE_ASSERT_NOTNULL(end_vec_ptr);
    const int64_t end_value = end_vec_ptr->GetData()[i];
    index_input.end_indexes.push_back(Symbol(end_value));
  }

  GE_ASSERT_NOTNULL(attrs->GetListInt(kAttrStridesInputIndex));
  for (size_t i = 0UL; i < attrs->GetListInt(kAttrStridesInputIndex)->GetSize(); i++) {
    const auto strides_vec_ptr = attrs->GetListInt(kAttrStridesInputIndex);
    GE_ASSERT_NOTNULL(strides_vec_ptr);
    const int64_t strides_value = strides_vec_ptr->GetData()[i];
    index_input.strides_indexes.push_back(Symbol(strides_value));
  }
  return SUCCESS;
}

graphStatus GetValueFromInputConstData(const gert::InferSymbolShapeContext *context, const size_t index,
                                       std::vector<Expression> &dims) {
  GE_ASSERT_NOTNULL(context);
  const auto input_tensor = context->GetInputSymbolTensor(index);
  GE_UNSUPPORTED_IF_NULL(input_tensor);
  const auto symbols = input_tensor->GetSymbolicValue();
  if (symbols == nullptr) {
    GELOGW("Symbolic infer shape unsupported, reason: get symbolic value failed, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  for (const auto &symbol : *symbols) {
    int64_t dim = 0L;
    // 确定值有效
    if (symbol.GetConstValue(dim) == false) {
      return UNSUPPORTED;
    }
    dims.emplace_back(Symbol(dim));
  }
  return SUCCESS;
}

Status GetStridedSliceIndexInput(const gert::InferSymbolShapeContext *context, StrdedSliceIndexInputs &index_input,
                                 size_t stride_index, bool is_stride_optional = false) {
  auto ret = GetValueFromInputConstData(context, kStartInputIndex, index_input.start_indexes);
  if (ret != SUCCESS) {
    return ret;
  }
  ret = GetValueFromInputConstData(context, kEndInputIndex, index_input.end_indexes);
  if (ret != SUCCESS) {
    return ret;
  }
  return (is_stride_optional && context->GetInputSymbolTensor(stride_index) == nullptr) ?
    SUCCESS : GetValueFromInputConstData(context, stride_index, index_input.strides_indexes);
}

Status ConstructAxis(const gert::InferSymbolShapeContext *context, int64_t input_dim_num, std::vector<int64_t> &axes) {
  const auto axes_tensor = context->GetInputSymbolTensor(kAxesV2InputIndex);
  if (axes_tensor == nullptr) {
    GELOGI("Set axes to default for node %s.", context->GetNodeName());
    return SUCCESS;
  }

  const auto symbols = axes_tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(symbols);
  for (const auto &symbol : *symbols) {
    int64_t value = 0;
    if (!symbol.GetConstValue(value)) {
      GELOGI("Axis value for node %s is not const.", context->GetNodeName());
      return UNSUPPORTED;
    }
    GE_ASSERT_TRUE(value < input_dim_num && value >= -input_dim_num, "Invalid axis value %lld for node %s.",
                   value, context->GetNodeName());
    axes.push_back(value >= 0 ? value : value + input_dim_num);
    GELOGD("Get const value %lld and add new axes value %lld for node %s.", value, axes.back(), context->GetNodeName());
  }
  return SUCCESS;
}

Status ConstructBeginList(const gert::InferSymbolShapeContext *context, const std::vector<ge::Expression> &x_dims,
                          const std::vector<int64_t> &axes, std::vector<Expression> &start_indexes) {
  start_indexes.resize(x_dims.size(), Symbol(0));
  std::vector<Expression> begin_values;
  graphStatus ret = GetValueFromInputConstData(context, kStartInputIndex, begin_values);
  if (ret != SUCCESS) {
    GELOGI("Begin list is not const for node %s, error code %u.", context->GetNodeName(), static_cast<uint32_t>(ret));
    return ret;
  }
  for (size_t i = 0UL; i < axes.size() && i < x_dims.size(); ++i) {
    start_indexes[axes[i]] = begin_values[i];
    GELOGD("Index %zu axe %lld begin %s", i, axes[i], begin_values[i].Serialize().get());
  }

  return SUCCESS;
}

Status ConstructEndList(const gert::InferSymbolShapeContext *context, const std::vector<ge::Expression> &x_dims,
                        const std::vector<int64_t> &axes, std::vector<Expression> &end_indexes) {
  for (size_t i = 0UL; i < x_dims.size(); i++) {
    end_indexes.push_back(x_dims[i]);
  }
  std::vector<Expression> end_values;
  graphStatus ret = GetValueFromInputConstData(context, kEndInputIndex, end_values);
  if (ret != SUCCESS) {
    GELOGI("End list is not const for node %s, error code %u.", context->GetNodeName(), static_cast<uint32_t>(ret));
    return ret;
  }
  for (size_t i = 0UL; i < axes.size() && i < x_dims.size(); i++) {
    end_indexes[axes[i]] = end_values[i];
    GELOGD("Index %zu axe %lld end %s", i, axes[i], end_values[i].Serialize().get());
  }

  return SUCCESS;
}

Status ConstructStrideList(const gert::InferSymbolShapeContext *context, const std::vector<ge::Expression> &x_dims,
                           const std::vector<int64_t> &axes, std::vector<Expression> &strides_indexes) {
  strides_indexes.resize(x_dims.size(), Symbol(1));
  const auto stride_tensor = context->GetInputSymbolTensor(kStridesV2InputIndex);
  if (stride_tensor == nullptr) {
    GELOGD("Apply default stride for node %s.", context->GetNodeName());
    return SUCCESS;
  }

  std::vector<Expression> stride_values;
  if (GetValueFromInputConstData(context, kStridesV2InputIndex, stride_values) != SUCCESS) {
    GELOGW("Failed to get const stride values for node %s.", context->GetNodeName());
    return UNSUPPORTED;
  }
  for (size_t i = 0UL; i < axes.size() && i < x_dims.size(); i++) {
    strides_indexes[axes[i]] = stride_values[i];
    GELOGD("Index %zu axe %lld stride %s", i, axes[i], strides_indexes[i].Serialize().get());
  }

  return SUCCESS;
}

Status GetStridedSliceV2IndexInput(const gert::InferSymbolShapeContext *context, StrdedSliceIndexInputs &index_input) {
  const auto x_shape = context->GetInputSymbolShape(kXInputIndex);
  GE_ASSERT_NOTNULL(x_shape);
  const std::vector<ge::Expression> x_dims = x_shape->GetDims();
  std::vector<int64_t> axes{};
  Status ret = ConstructAxis(context, x_dims.size(), axes);
  if (ret != SUCCESS) {
    return ret;
  }
  ret = ConstructBeginList(context, x_dims, axes, index_input.start_indexes);
  if (ret != SUCCESS) {
    return ret;
  }
  ret = ConstructEndList(context, x_dims, axes, index_input.end_indexes);
  if (ret != SUCCESS) {
    return ret;
  }
  ret = ConstructStrideList(context, x_dims, axes, index_input.strides_indexes);
  if (ret != SUCCESS) {
    return ret;
  }
  return SUCCESS;
}

Status GetStridedSliceMaskAttr(const gert::InferSymbolShapeContext *context, StridedSliceAttr &strided_slice_attr) {
  GE_ASSERT_NOTNULL(context);
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto begin_ptr = attrs->GetInt(kAttrBeginMaskIndex);
  GE_ASSERT_NOTNULL(begin_ptr);
  strided_slice_attr.begin_mask = *begin_ptr;
  const auto end_ptr = attrs->GetInt(kAttrEndMaskIndex);
  GE_ASSERT_NOTNULL(end_ptr);
  strided_slice_attr.end_mask = *end_ptr;
  const auto ellipsis_ptr = attrs->GetInt(kAttrEllipsisMaskIndex);
  GE_ASSERT_NOTNULL(ellipsis_ptr);
  strided_slice_attr.ellipsis_mask = *ellipsis_ptr;
  const auto new_axis_ptr = attrs->GetInt(kAttrNewAxisMaskIndex);
  GE_ASSERT_NOTNULL(new_axis_ptr);
  strided_slice_attr.new_axis_mask = *new_axis_ptr;
  const auto shrink_axis_ptr = attrs->GetInt(kAttrShrinkAxisMaskIndex);
  GE_ASSERT_NOTNULL(shrink_axis_ptr);
  strided_slice_attr.shrink_axis_mask = *shrink_axis_ptr;
  return SUCCESS;
}

Status GetStridedSliceDMaskAttr(const gert::InferSymbolShapeContext *context, StridedSliceAttr &strided_slice_attr) {
  GE_ASSERT_NOTNULL(context);
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto begin_ptr = attrs->GetInt(kAttrDBeginMaskIndex);
  GE_ASSERT_NOTNULL(begin_ptr);
  strided_slice_attr.begin_mask = *begin_ptr;
  const auto end_ptr = attrs->GetInt(kAttrDEndMaskIndex);
  GE_ASSERT_NOTNULL(end_ptr);
  strided_slice_attr.end_mask = *end_ptr;
  const auto ellipsis_ptr = attrs->GetInt(kAttrDEllipsisMaskIndex);
  GE_ASSERT_NOTNULL(ellipsis_ptr);
  strided_slice_attr.ellipsis_mask = *ellipsis_ptr;
  const auto new_axis_ptr = attrs->GetInt(kAttrDNewAxisMaskIndex);
  GE_ASSERT_NOTNULL(new_axis_ptr);
  strided_slice_attr.new_axis_mask = *new_axis_ptr;
  const auto shrink_axis_ptr = attrs->GetInt(kAttrDShrinkAxisMaskIndex);
  GE_ASSERT_NOTNULL(shrink_axis_ptr);
  strided_slice_attr.shrink_axis_mask = *shrink_axis_ptr;
  return SUCCESS;
}

Status HandleMaskAttr(const std::pair<int64_t, int64_t> &ellipsis_mask_range,
                      const std::vector<Expression> &input_append_axis_shape,
                      const StridedSliceAttr &strided_slice_attr, StrdedSliceIndexInputs &index_input) {
  // 处理ellipsis_mask
  GE_ASSERT_SUCCESS(HandleEllipsisMask(ellipsis_mask_range.first, input_append_axis_shape, index_input));
  // 补充缺省的index维度
  GE_ASSERT_SUCCESS(FillMissionIndex(ellipsis_mask_range, input_append_axis_shape, index_input));
  // 处理begin_mask和end_mask
  HandleBeginEndMask(strided_slice_attr, input_append_axis_shape, ellipsis_mask_range, index_input);
  return SUCCESS;
}

graphStatus InferShape4StridedSlice(gert::InferSymbolShapeContext *context) {
  StrdedSliceIndexInputs index_input;
  StridedSliceAttr strided_slice_attr;
  Status retinput = PARAM_INVALID;
  Status retattr = PARAM_INVALID;
  GE_ASSERT_NOTNULL(context);
  if (strcmp(context->GetNodeType(), "StridedSliceD") == 0) {
    retinput = GetStridedSliceDIndexInput(context, index_input);
    retattr = GetStridedSliceDMaskAttr(context, strided_slice_attr);
  } else if (strcmp(context->GetNodeType(), "StridedSliceV2") == 0) {
    retinput = GetStridedSliceV2IndexInput(context, index_input);
    retattr = GetStridedSliceMaskAttr(context, strided_slice_attr);
  } else {
    retinput = GetStridedSliceIndexInput(context, index_input, kStridesInputIndex);
    retattr = GetStridedSliceMaskAttr(context, strided_slice_attr);
  }
  if (retattr != SUCCESS || retinput != SUCCESS) {
    const auto ret = (retattr != SUCCESS) ? retattr : retinput;
    return ret;
  }
  std::vector<Expression> input_x_dims;
  const auto x_shape = context->GetInputSymbolShape(kXInputIndex);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  for (const auto &s : x_shape->GetDims()) {
    input_x_dims.push_back(s);
  }
  HandleMaskConflict(strided_slice_attr);
  const std::pair<int64_t, int64_t> ellipsis_mask_range =
      GetEllipsisMaskRange(strided_slice_attr, static_cast<int64_t>(index_input.start_indexes.size()),
                           static_cast<int64_t>(input_x_dims.size()));
  std::vector<Expression> input_append_axis_shape;
  GE_ASSERT_SUCCESS(AppendNewAxis(ellipsis_mask_range, strided_slice_attr.new_axis_mask, input_x_dims,
                                  input_append_axis_shape, index_input));
  GE_ASSERT_SUCCESS(HandleMaskAttr(ellipsis_mask_range, input_append_axis_shape, strided_slice_attr, index_input));

  const auto shape_output = context->GetOutputSymbolShape(kOutputIndex);
  GE_ASSERT_NOTNULL(shape_output);
  std::vector<Expression> output_symbols_shape;
  GE_ASSERT_SUCCESS(
      CalcOutputShape(strided_slice_attr.shrink_axis_mask, ellipsis_mask_range, index_input, output_symbols_shape));
  shape_output->MutableDims() = output_symbols_shape;
  return SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(StridedSliceD).InferSymbolShape(InferShape4StridedSlice);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(StridedSlice).InferSymbolShape(InferShape4StridedSlice);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(StridedSliceV2).InferSymbolShape(InferShape4StridedSlice);

Expression CalculateBeginValue(const Expression &begin_input, const Expression &cur_axis_input_size,
                               const Expression &step_value) {
  int64_t step_const = 0;
  const auto clip_upper = (step_value.GetConstValue(step_const) && step_const < 0) ?
    cur_axis_input_size - Symbol(1) : cur_axis_input_size;
  Expression normalized_begin = (EXPECT_SYMBOL_LT(begin_input, kSymbolZero)) ?
      (begin_input + cur_axis_input_size) : begin_input;
  return (EXPECT_SYMBOL_LT(normalized_begin, kSymbolZero)) ? kSymbolZero :
      (EXPECT_SYMBOL_LT(clip_upper, normalized_begin)) ? clip_upper : normalized_begin;
}

Expression CalculateEndValue(const Expression &end_input, const Expression &cur_axis_input_size,
                             const Expression &step_value) {
  int64_t step_const = 0;
  const auto clip_lower = (step_value.GetConstValue(step_const) && step_const < 0) ? Symbol(-1) : kSymbolZero;
  Expression normalized_end = (EXPECT_SYMBOL_LT(end_input, kSymbolZero)) ?
      (end_input + cur_axis_input_size) : end_input;
  return (EXPECT_SYMBOL_LT(normalized_end, clip_lower)) ? clip_lower :
      (EXPECT_SYMBOL_LT(cur_axis_input_size, normalized_end)) ? cur_axis_input_size : normalized_end;
}

void CalculateOutputDimsForV3(const std::vector<int64_t> &axes, const std::vector<Expression> &input_x_dims,
                              const StrdedSliceIndexInputs &index_input, std::vector<Expression> &output_dims) {
  for (size_t i = 0UL; i < axes.size(); ++i) {
    const int64_t axis_value = axes[i];
    const Expression step_value = i < index_input.strides_indexes.size() ? index_input.strides_indexes[i] : Symbol(1);
    const Expression begin_value =
        i < index_input.start_indexes.size() ?
        CalculateBeginValue(index_input.start_indexes[i], input_x_dims[axis_value], step_value) : Symbol(0);
    const Expression end_value =
        i < index_input.end_indexes.size() ?
        CalculateEndValue(index_input.end_indexes[i], input_x_dims[axis_value], step_value) : input_x_dims[axis_value];
    Expression cur_out_size = sym::Ceiling((end_value - begin_value) / step_value);
    cur_out_size = (EXPECT_SYMBOL_LT(cur_out_size, kSymbolZero)) ? kSymbolZero : cur_out_size;
    GELOGD("Axe index %zu, begin symbol %s, end symbol %s, step symbol %s, outdim symbol %s", i,
           begin_value.Serialize().get(), end_value.Serialize().get(),
           step_value.Serialize().get(), cur_out_size.Serialize().get());
    output_dims[axis_value] = cur_out_size;
  }
}

Status InferShape4StridedSliceV3(gert::InferSymbolShapeContext *context) {
  GE_ASSERT_NOTNULL(context);
  const auto shape_output = context->GetOutputSymbolShape(kOutputIndex);
  GE_ASSERT_NOTNULL(shape_output);
  std::vector<Expression> input_x_dims;
  const auto x_shape = context->GetInputSymbolShape(kXInputIndex);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  for (const auto &s : x_shape->GetDims()) {
    input_x_dims.push_back(s);
  }

  StrdedSliceIndexInputs index_input;
  Status ret = GetStridedSliceIndexInput(context, index_input, kStridesV2InputIndex, true);
  if (ret != SUCCESS) {
    return ret;
  }

  std::vector<int64_t> axes;
  ret = ConstructAxis(context, input_x_dims.size(), axes);
  if (ret != SUCCESS) {
    return ret;
  }
  if (axes.empty()) {
    axes.resize(input_x_dims.size());
    std::iota(axes.begin(), axes.end(), 0);
  }
  shape_output->MutableDims() = input_x_dims;
  CalculateOutputDimsForV3(axes, input_x_dims, index_input, shape_output->MutableDims());
  return SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(StridedSliceV3).InferSymbolShape(InferShape4StridedSliceV3);
}  // namespace
}  // namespace ge

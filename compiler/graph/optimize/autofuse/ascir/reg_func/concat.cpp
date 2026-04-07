/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_ir.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "defalut_reg_func.h"

namespace ge {
namespace ascir {
namespace {
constexpr int32_t TWO = 2;
constexpr int32_t FOUR = 4;

constexpr int32_t TMP_SIZE_UNIT = 1024;
constexpr int32_t BASIC_TMP_SIZE = 16384;
constexpr int32_t MAX_TMP_SIZE = 65536;
constexpr int32_t MAX_TMP_SIZE_FOR_SMALL_TAIL = 96 * 1024;

constexpr int32_t TYPESIZEEQ8 = 8;
constexpr int32_t TYPESIZEEQ4 = 4;
constexpr int32_t TYPESIZEEQ2 = 2;
constexpr int32_t TYPESIZEEQ1 = 1;

constexpr int32_t ALIGNSIZE8 = 8;
constexpr int32_t ALIGNSIZE16 = 16;
constexpr int32_t ALIGNSIZE32 = 32;

constexpr int32_t ALIGNPAD_8 = 29;
constexpr int32_t ALIGNPAD_4 = 29;
constexpr int32_t ALIGNPAD_2 = 45;
constexpr int32_t ALIGNPAD_1 = 93;

constexpr int32_t TMPSIZEOF8_4 = 128;
constexpr int32_t TMPSIZEOF2 = 64;
constexpr int32_t TMPSIZEOF1 = 48;

Expression CalcForSmallTailKernel(AscNodeOutputs &node_outputs, uint32_t concat_dim) {
  auto dst_col_size_expr = node_outputs[0U].attr.repeats[concat_dim];
  for (uint32_t i = concat_dim + 1; i < node_outputs[0U].attr.repeats.size(); ++i) {
    dst_col_size_expr = dst_col_size_expr * node_outputs[0U].attr.repeats[i];
  }
  int64_t dst_col_size = -1;
  if (!dst_col_size_expr.GetConstValue(dst_col_size)) {
    return ge::sym::Align(dst_col_size_expr, ALIGNSIZE16) * ge::Symbol(TMP_SIZE_UNIT);
  }
  auto scale = std::max((dst_col_size + ALIGNSIZE16 - 1) / ALIGNSIZE16, 2L);
  auto buf_size = BASIC_TMP_SIZE * scale;
  if (buf_size * TWO <= MAX_TMP_SIZE_FOR_SMALL_TAIL) {
    buf_size *= TWO;
  }
  return ge::Symbol(buf_size);
}

bool IsAllStaticAligned(AscNodeInputs &node_inputs, uint32_t concat_dim, int32_t align_size) {
  for (uint32_t i = 0; i < node_inputs.Size(); ++i) {
    auto axis = node_inputs[i].attr.repeats[concat_dim];
    for (uint32_t j = concat_dim + 1; j < node_inputs[i].attr.repeats.size(); ++j) {
      axis = sym::Mul(axis, node_inputs[i].attr.repeats[j]);
    }

    if (SymbolicUtils::StaticCheckEq(ge::sym::Mod(axis, ge::Symbol(align_size)), sym::kSymbolZero) != TriBool::kTrue) {
      GELOGD("The product of dims after concat_dim is %s, not aligned.",
              ge::SymbolicUtils::ToString(ge::sym::Mod(axis, ge::Symbol(align_size))).c_str());
      return false;
    }
  }
  return true;
}

Expression CalcForDefaultKernel(AscNodeInputs &node_inputs, uint32_t concat_dim, bool flag) {
  Expression max_axis_size = ge::Symbol(0);
  if (flag) {
    for (uint32_t i = 1; i < node_inputs.Size(); ++i) {
      Expression axis = node_inputs[i].attr.repeats[concat_dim];
      for (uint32_t j = concat_dim + 1; j < node_inputs[i].attr.repeats.size(); ++j) {
        axis = sym::Mul(axis, node_inputs[i].attr.repeats[j]);
      }
      max_axis_size = sym::Max(max_axis_size, axis);
    }
  } else {
    for (uint32_t i = 1; i < node_inputs.Size(); ++i) {
      max_axis_size = sym::Max(max_axis_size, node_inputs[i].attr.repeats[node_inputs[i].attr.repeats.size() - 1]);
    }
  }

  auto type_size = GetSizeByDataType(node_inputs[0].attr.dtype);
  GE_ASSERT_TRUE(type_size != 0, "Invalid node inputs dtype, sizeof(T) = 0.");
  Expression min_tmp_buf_size = ge::Symbol(0);
  bool is_aligned = IsAllStaticAligned(node_inputs, concat_dim, ALIGNSIZE32 / type_size);
  if (type_size == TYPESIZEEQ8) {
    min_tmp_buf_size = is_aligned ? ge::Symbol(0) : (sym::Align(ge::Symbol(FOUR) * max_axis_size, ALIGNSIZE8) +
                                                              ge::Symbol(ALIGNPAD_8)) * ge::Symbol(TMPSIZEOF8_4);
  } else if (type_size == TYPESIZEEQ4) {
    min_tmp_buf_size = is_aligned ? ge::Symbol(0) : (sym::Align(ge::Symbol(TWO) * max_axis_size, ALIGNSIZE8) +
                                                              ge::Symbol(ALIGNPAD_4)) * ge::Symbol(TMPSIZEOF8_4);
  } else if (type_size == TYPESIZEEQ2) {
    min_tmp_buf_size = is_aligned ? ge::Symbol(0) : (sym::Align(ge::Symbol(TWO) * max_axis_size, ALIGNSIZE16) +
                                                              ge::Symbol(ALIGNPAD_2)) * ge::Symbol(TMPSIZEOF2);
  } else if (type_size == TYPESIZEEQ1) {
    min_tmp_buf_size = is_aligned ? ge::Symbol(0) : (sym::Align(ge::Symbol(TWO) * max_axis_size, ALIGNSIZE32) +
                                                              ge::Symbol(ALIGNPAD_1)) * ge::Symbol(TMPSIZEOF1);
  }
  return min_tmp_buf_size;
}
}  // namespace

std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcConcatTmpSize(const ge::AscNode &node) {
  std::vector<std::unique_ptr<ge::TmpBufDesc>> tmp_buf_desc;
  AscNodeInputs node_inputs = node.inputs;
  AscNodeOutputs node_outputs = node.outputs;
  if (node_inputs.Size() <= 0) {
    return tmp_buf_desc;
  }

  bool flag = false;  // 是否尾轴合并
  uint32_t concat_dim = 0;
  for (uint32_t i = 0; i < node_outputs[0].attr.repeats.size(); ++i) {
    if (SymbolicUtils::StaticCheckEq(node_outputs[0].attr.repeats[i], node_inputs[0].attr.repeats[i]) != TriBool::kTrue) {
      concat_dim = i;
      if (i != node_outputs[0].attr.repeats.size() - 1) {
        flag = true;
      }
    }
  }
  bool concat_small_tail = false;
  (void) ge::AttrUtils::GetBool(node.GetOpDesc(), "_concat_small_tail", concat_small_tail);
  const auto tmp_buf_size = concat_small_tail ? CalcForSmallTailKernel(node_outputs, concat_dim) :
                            CalcForDefaultKernel(node_inputs, concat_dim, flag);
  if (SymbolicUtils::StaticCheckEq(tmp_buf_size, sym::kSymbolZero) == TriBool::kTrue) {
    GELOGI("%s does not require tmp buf", node.GetNamePtr());
    return {};
  }

  auto min_tmp_buf_size = sym::Max(ge::Symbol(BASIC_TMP_SIZE), tmp_buf_size);
  auto max_tmp_buf_size = concat_small_tail ? MAX_TMP_SIZE_FOR_SMALL_TAIL : MAX_TMP_SIZE;
  min_tmp_buf_size = sym::Min(ge::Symbol(max_tmp_buf_size), min_tmp_buf_size);
  ge::TmpBufDesc desc = {min_tmp_buf_size, -1};
  tmp_buf_desc.emplace_back(std::make_unique<ge::TmpBufDesc>(desc));
  GELOGD("%s is_small_tail = %d, calc_buf_size = %s, min_tmp_buf_size = %s", node.GetNamePtr(),
         static_cast<int32_t>(concat_small_tail), ge::SymbolicUtils::ToString(tmp_buf_size).c_str(),
         ge::SymbolicUtils::ToString(min_tmp_buf_size).c_str());
  return tmp_buf_desc;
}

std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcConcatTmpSizeV2(const ge::AscNode &node) {
  AscNodeInputs node_inputs = node.inputs;
  AscNodeOutputs node_outputs = node.outputs;
  GE_ASSERT_TRUE(node_inputs.Size() > 0);
  uint32_t concat_dim = 0;
  const auto num_dims = node_outputs[0].attr.repeats.size();
  for (uint32_t idx = 0; idx < num_dims; ++idx) {
    const auto i = num_dims - idx - 1;
    if (node_outputs[0].attr.repeats[i] != node_inputs[0].attr.repeats[i]) {
      concat_dim = i;
      break;
    }
  }
  auto type_size = GetSizeByDataType(node_inputs[0].attr.dtype);
  GE_ASSERT_TRUE(type_size > 0,
                 "%s Invalid node inputs dtype: %d",
                 node.GetNamePtr(), static_cast<int32_t>(node_inputs[0].attr.dtype));
  Expression min_tmp_buf_size = ge::Symbol(0);
  bool is_aligned = IsAllStaticAligned(node_inputs, concat_dim, ALIGNSIZE32 / type_size);
  if (is_aligned) {
    GELOGD("%s is all aligned", node.GetNamePtr());
    return {};
  }
  constexpr int64_t kTmpBufSizeForConcatByScatter = 1024L;
  return GetTmpBuffer(ge::Symbol(kTmpBufSizeForConcatByScatter));
}
}  // namespace ascir
}  // namespace ge
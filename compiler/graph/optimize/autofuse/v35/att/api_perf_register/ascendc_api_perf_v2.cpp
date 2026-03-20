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
#include <numeric>
#include "common/checker.h"
#include "perf_param_v2.h"
#include "api_perf_register/ascendc_api_perf.h"
#include "api_perf_register/api_perf_factory.h"
namespace att {
namespace {
// 获取 CacheLine 大小，与 TilingScheduleConfigTable 保持一致
uint32_t GetCacheLineSize() {
  return TilingScheduleConfigTableV2().GetCacheLineSize();
}

// 初始化block_len扩展所需的参数
ge::Status InitBlockLenExpandParams(const NodeDetail &node_info, const std::vector<Expr> &dims,
                                     Expr &block_len, Expr &cache_line_ele_num) {
  const size_t dim_size = dims.size();
  block_len = dims[dim_size - 1UL];
  const int32_t kCacheLineSize = static_cast<int32_t>(GetCacheLineSize());
  const auto &data_type_size = kDataTypeSizeMap.find(node_info.input_dtype[0]);
  GE_ASSERT_TRUE(data_type_size != kDataTypeSizeMap.cend(), "Check data type size failed, node[%s]",
                 node_info.ToString().c_str());
  cache_line_ele_num = CreateExpr(kCacheLineSize) / data_type_size->second;
  return ge::SUCCESS;
}

ge::Status GetBlockCount(const NodeDetail &node_info, vector<CacheLineConfig> *cache_line_config) {
  if (cache_line_config != nullptr) {
    auto dims = node_info.input_dims;
    Expr dim_product = accumulate(dims.begin(), dims.end(), CreateExpr(1), [](Expr a, Expr b) { return a * b; });
    auto iter1 = kDataTypeSizeMap.find(node_info.input_dtype[0]);
    GE_ASSERT_TRUE(iter1 != kDataTypeSizeMap.end());
    Expr dim_product_byte = ge::sym::Mul(dim_product, iter1->second);
    if (!dim_product_byte.IsConstExpr()) {
      cache_line_config->push_back({node_info.name, dim_product_byte, GetCacheLineSize()});
    }
  }
  return ge::SUCCESS;
}

ge::Status GetLoadCase(const NodeDetail &node_info, Expr &blk, int32_t &use_case) {
  size_t dim_size = node_info.input_dims.size();
  GE_ASSERT_TRUE(!node_info.input_dims.empty(), "Check node input dims failed, node[%s]", node_info.ToString().c_str());
  const auto blk_len = node_info.input_dims[dim_size - 1UL];
  if (blk_len.IsConstExpr()) {
    int32_t blk_len_val;
    constexpr int32_t blk_threshold = 256U;
    blk_len.GetConstValue(blk_len_val);
    if (blk_len_val > blk_threshold) {
      use_case = kCaseOne;
    } else {
      use_case = kCaseTwo;
    }
  } else {
    const auto blk_thr = CreateExpr(256);
    blk = blk_len - blk_thr;
    use_case = kCaseDefault;
  }
  GELOGD("input dtype is %s, dim_size[%zu], block_len[%s], use_case[%d], node[%s]", node_info.input_dtype[0].c_str(),
         dim_size, blk_len.Str().get(), use_case, node_info.ToString().c_str());
  return ge::SUCCESS;
}

ge::Status LoadPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res_normal;
  Expr res_small_blk;
  Expr blk;
  Expr res_stride;
  int32_t use_case;
  GELOGD("Dma with Load: %s", node_info.ToString().c_str());
  GE_ASSERT_SUCCESS(GetPerf({kMoveGmToUb, node_info.input_dtype[0],node_info.output_dtype[0],
                             node_info.input_dims, CreateExpr(0)}, res_normal));
  GE_ASSERT_SUCCESS(GetPerf({kMoveGmToUb + "SmallBlk", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, CreateExpr(0)}, res_small_blk));
  GE_ASSERT_SUCCESS(GetPerf({kMoveGmToUb + "Stride", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride, node_info.block_count_idx}, res_stride));
  GE_ASSERT_SUCCESS(GetLoadCase(node_info, blk, use_case));
  GE_ASSERT_SUCCESS(GetBlockCount(node_info, perf.cache_line_config));
  if (use_case == kCaseOne) {
    perf.pipe_res[PipeType::AIV_MTE2] = res_normal + res_stride;
  } else if (use_case == kCaseTwo) {
    perf.pipe_res[PipeType::AIV_MTE2] = res_small_blk + res_stride;
  } else {
    Expr res = CreateExpr("load_node");
    std::shared_ptr<IfCase> branch_a = std::make_shared<IfCase>(res_normal + res_stride);
    GE_ASSERT_NOTNULL(branch_a);
    std::shared_ptr<IfCase> branch_b = std::make_shared<IfCase>(res_small_blk + res_stride);
    GE_ASSERT_NOTNULL(branch_b);
    // blocklen < 512B时走branch_b；否则走branch_a
    TernaryOp ternary_op = TernaryOp(CondType::K_LT, blk, CreateExpr(0), std::move(branch_b), std::move(branch_a));
    ternary_op.SetVariable(res);
    perf.ternary_ops[res] = ternary_op;
    perf.pipe_res[PipeType::AIV_MTE2] = res;
  }
  return ge::SUCCESS;
}

ge::Status ExpandMTE3BlockLen(const NodeDetail &node_info, PerfOutputInfo &perf, std::vector<Expr> &dims);

ge::Status StorePerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res_normal;
  Expr res_stride;
  GELOGD("Dma with Store: %s", node_info.ToString().c_str());
  std::vector<Expr> padded_dims{node_info.input_dims};
  // 应用MTE3场景的block_len padding（cache line大小128字节）
  GE_ASSERT_SUCCESS(ExpandMTE3BlockLen(node_info, perf, padded_dims),
                    "Expand MTE3 block len failed, node[%s]", node_info.ToString().c_str());
  GE_ASSERT_SUCCESS(GetPerf({kMoveUbToGm, node_info.input_dtype[0],node_info.output_dtype[0],
                             padded_dims, CreateExpr(0)}, res_normal));
  GE_ASSERT_SUCCESS(GetPerf({kMoveUbToGm + "Stride", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.repeats, node_info.gm_stride, node_info.block_count_idx}, res_stride));
  GE_ASSERT_SUCCESS(GetBlockCount(node_info, perf.cache_line_config));
  perf.pipe_res[PipeType::AIV_MTE3] = res_normal + res_stride;
  return ge::SUCCESS;
}

namespace {
ge::Status InitBlockLenExpand(const NodeDetail &node_info, std::vector<Expr> &dims,
                              Expr &block_len, Expr &cache_line_ele_num,
                              Expr &block_len_ref, int32_t &kCacheLineSize,
                              int32_t &blk_len_val, int32_t &stride_val) {
  GE_ASSERT_SUCCESS(InitBlockLenExpandParams(node_info, dims, block_len, cache_line_ele_num));
  block_len_ref = const_cast<std::vector<Expr> &>(dims)[dims.size() - 1UL];
  kCacheLineSize = static_cast<int32_t>(GetCacheLineSize());
  if (block_len_ref.IsConstExpr() && node_info.gm_stride.IsConstExpr()) {
    block_len_ref.GetConstValue(blk_len_val);
    node_info.gm_stride.GetConstValue(stride_val);
    return ge::SUCCESS;
  }
  return ge::FAILED;
}
}

ge::Status ExpandBlockLen(const NodeDetail &node_info, PerfOutputInfo &perf, std::vector<Expr> &dims) {
  // 满足gm非连续且搬运小于cache line，扩展每次搬运数据量到cache line
  Expr block_len;
  Expr cache_line_ele_num;
  Expr block_len_ref;
  int32_t kCacheLineSize = 0;
  int32_t blk_len_val = 1;
  int32_t stride_val = 0;

  if (InitBlockLenExpand(node_info, dims, block_len, cache_line_ele_num,
                         block_len_ref, kCacheLineSize, blk_len_val, stride_val) == ge::SUCCESS) {
    // 存在非连续并且block_len较小，无法并包，考虑将block_len对齐到cache line大小
    if ((blk_len_val < kCacheLineSize) && (stride_val > 0)) {
      block_len_ref = CreateExpr(kCacheLineSize);
    }
  } else {
    auto &block_len_ref_actual = const_cast<std::vector<Expr> &>(dims)[dims.size() - 1UL];
    auto is_small_block_len_checker =
        ge::sym::LogicalAnd({ge::sym::Gt(node_info.gm_stride, CreateExpr(0)), ge::sym::Lt(block_len_ref_actual, cache_line_ele_num)});
    bool is_small_block_len{false};
    if (is_small_block_len_checker.IsConstExpr()) {
      is_small_block_len_checker.GetConstValue(is_small_block_len);
      block_len_ref_actual = is_small_block_len ? std::move(cache_line_ele_num) : std::move(block_len_ref_actual);
    } else {
      Expr res = CreateExpr("block_len");
      auto normal_case = std::make_shared<IfCase>(block_len_ref_actual);
      GE_ASSERT_NOTNULL(normal_case);
      auto small_block_len_case = std::make_shared<IfCase>(cache_line_ele_num);
      GE_ASSERT_NOTNULL(small_block_len_case);
      TernaryOp ternary_op = TernaryOp(CondType::K_EQ, is_small_block_len_checker, CreateExpr(false),
                                    std::move(normal_case), std::move(small_block_len_case));
      ternary_op.SetVariable(res);
      perf.ternary_ops[res] = ternary_op;
      block_len_ref_actual = res;
    }
    GELOGD("Block len checker[%s], is_small_block_len[%d], block_len[%s]", is_small_block_len_checker.Serialize().get(),
           is_small_block_len, block_len_ref_actual.Serialize().get());
  }
  return ge::SUCCESS;
}

ge::Status ExpandMTE3BlockLen(const NodeDetail &node_info, PerfOutputInfo &perf, std::vector<Expr> &dims) {
  // 满足GM非连续且搬运小于cache line，扩展每次搬运数据量到cache line大小
  Expr block_len;
  Expr cache_line_ele_num;
  Expr block_len_ref;
  int32_t kCacheLineSize = 0;
  int32_t blk_len_val = 1;
  int32_t stride_val = 0;

  if (InitBlockLenExpand(node_info, dims, block_len, cache_line_ele_num,
                         block_len_ref, kCacheLineSize, blk_len_val, stride_val) == ge::SUCCESS) {
    // 计算实际字节数
    const auto &data_type_size = kDataTypeSizeMap.find(node_info.input_dtype[0]);
    int32_t data_type_size_val = 0;
    data_type_size->second.GetConstValue(data_type_size_val);
    int32_t block_len_bytes = blk_len_val * data_type_size_val;
    // 仅在有stride且block_len_bytes小于cache line时padding
    if ((block_len_bytes < kCacheLineSize) && (stride_val > 0)) {
      block_len_ref = cache_line_ele_num;
      GELOGD("MTE3 Store: block_len padded from %d to %d elements (%d bytes)", blk_len_val,
             kCacheLineSize / data_type_size_val, kCacheLineSize);
    }
  } else {
    // 动态shape处理 - 创建三元表达式
    auto &block_len_ref_actual = const_cast<std::vector<Expr> &>(dims)[dims.size() - 1UL];
    const auto &data_type_size = kDataTypeSizeMap.find(node_info.input_dtype[0]);
    Expr block_len_bytes = ge::sym::Mul(block_len_ref_actual, data_type_size->second);
    auto need_padding_checker = ge::sym::LogicalAnd(
        {ge::sym::Gt(node_info.gm_stride, CreateExpr(0)), ge::sym::Lt(block_len_bytes, CreateExpr(kCacheLineSize))});
    bool need_padding{false};
    if (need_padding_checker.IsConstExpr()) {
      need_padding_checker.GetConstValue(need_padding);
      block_len_ref_actual = need_padding ? std::move(cache_line_ele_num) : std::move(block_len_ref_actual);
    } else {
      // 动态分支：生成三元表达式
      Expr res = CreateExpr("mte3_block_len");
      auto normal_case = std::make_shared<IfCase>(block_len_ref_actual);
      GE_ASSERT_NOTNULL(normal_case);
      auto padding_case = std::make_shared<IfCase>(cache_line_ele_num);
      GE_ASSERT_NOTNULL(padding_case);
      TernaryOp ternary_op = TernaryOp(CondType::K_EQ, need_padding_checker, CreateExpr(false),
                                     std::move(normal_case), std::move(padding_case));
      ternary_op.SetVariable(res);
      perf.ternary_ops[res] = ternary_op;
      block_len_ref_actual = res;
    }
    GELOGD("MTE3 Block len padding checker[%s], need_padding[%d], block_len[%s]",
           need_padding_checker.Serialize().get(), need_padding, block_len_ref.Serialize().get());
  }
  return ge::SUCCESS;
}

ge::Status NddmaPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res_normal;
  Expr res_stride;
  GELOGD("Dma with nddma: %s", node_info.ToString().c_str());
  std::vector<Expr> dims{node_info.input_dims};
  // 将block_len先扩展到cache line再计算性能
  GE_ASSERT_TRUE(!dims.empty(), "Check node input dims failed, node[%s]", node_info.ToString().c_str());
  GE_ASSERT_SUCCESS(ExpandBlockLen(node_info, perf, dims), "Expand block len failed, node[%s]",
                    node_info.ToString().c_str());
  GE_ASSERT_SUCCESS(
      GetPerf({kNddma, node_info.input_dtype[0], node_info.output_dtype[0], dims, node_info.gm_stride}, res_normal));
  GE_ASSERT_SUCCESS(GetPerf({kNddma + "Stride", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.repeats, node_info.gm_stride, node_info.block_count_idx}, res_stride));
  GE_ASSERT_SUCCESS(GetBlockCount(node_info, perf.cache_line_config));
  perf.pipe_res[PipeType::AIV_MTE2] = res_normal + res_stride;
  return ge::SUCCESS;
}

REGISTER_ASCENDC_EVAL_FUNC_TAG(kLoad, V2, LoadPerf);
REGISTER_ASCENDC_EVAL_FUNC_TAG(kStore, V2, StorePerf);
REGISTER_ASCENDC_EVAL_FUNC_TAG(kNddma, V2, NddmaPerf);
}
}
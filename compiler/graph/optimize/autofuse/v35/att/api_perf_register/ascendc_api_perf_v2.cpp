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

ge::Status StorePerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res_normal;
  Expr res_stride;
  GELOGD("Dma with Load: %s", node_info.ToString().c_str());
  GE_ASSERT_SUCCESS(GetPerf({kMoveUbToGm, node_info.input_dtype[0],node_info.output_dtype[0],
                             node_info.input_dims, CreateExpr(0)}, res_normal));
  GE_ASSERT_SUCCESS(GetPerf({kMoveUbToGm + "Stride", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.repeats, node_info.gm_stride, node_info.block_count_idx}, res_stride));
  GE_ASSERT_SUCCESS(GetBlockCount(node_info, perf.cache_line_config));
  perf.pipe_res[PipeType::AIV_MTE3] = res_normal + res_stride;
  return ge::SUCCESS;
}

ge::Status ExpandBlockLen(const NodeDetail &node_info, PerfOutputInfo &perf, std::vector<Expr> &dims) {
  // 满足gm非连续且搬运小于cache line，扩展每次搬运数据量到cache line
  const size_t dim_size = dims.size();
  auto &block_len = dims[dim_size - 1UL];
  const int32_t kCacheLineSize = static_cast<int32_t>(GetCacheLineSize());
  const auto &data_type_size = kDataTypeSizeMap.find(node_info.input_dtype[0]);
  GE_ASSERT_TRUE(data_type_size != kDataTypeSizeMap.cend(), "Check data type size failed, node[%s]",
                 node_info.ToString().c_str());
  const auto kCacheLineEleNum = CreateExpr(kCacheLineSize) / data_type_size->second;
  if (block_len.IsConstExpr() && node_info.gm_stride.IsConstExpr()) {
    int32_t blk_len_val = 1;
    int32_t stride_val = 0;
    block_len.GetConstValue(blk_len_val);
    node_info.gm_stride.GetConstValue(stride_val);
    // 存在非连续并且block_len较小，无法并包，考虑将block_len对齐到cache line大小
    if ((blk_len_val < kCacheLineSize) && (stride_val > 0)) {
      block_len = CreateExpr(kCacheLineSize);
    }
  } else {
    auto is_small_block_len_checker =
        ge::sym::LogicalAnd({ge::sym::Gt(node_info.gm_stride, CreateExpr(0)), ge::sym::Lt(block_len, kCacheLineEleNum)});
    bool is_small_block_len{false};
    if (is_small_block_len_checker.IsConstExpr()) {
      is_small_block_len_checker.GetConstValue(is_small_block_len);
      block_len = is_small_block_len ? std::move(kCacheLineEleNum) : std::move(block_len);
    } else {
      Expr res = CreateExpr("block_len");
      auto normal_case = std::make_shared<IfCase>(block_len);
      GE_ASSERT_NOTNULL(normal_case);
      auto small_block_len_case = std::make_shared<IfCase>(kCacheLineEleNum);
      GE_ASSERT_NOTNULL(small_block_len_case);
      TernaryOp ternary_op = TernaryOp(CondType::K_EQ, is_small_block_len_checker, CreateExpr(false),
                                    std::move(normal_case), std::move(small_block_len_case));
      ternary_op.SetVariable(res);
      perf.ternary_ops[res] = ternary_op;
      block_len = res;
    }
    GELOGD("Block len checker[%s], is_small_block_len[%d], block_len[%s]", is_small_block_len_checker.Serialize().get(),
           is_small_block_len, block_len.Serialize().get());
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
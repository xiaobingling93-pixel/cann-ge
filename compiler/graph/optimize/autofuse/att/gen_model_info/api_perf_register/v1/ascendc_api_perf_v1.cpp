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
#include "base/att_const_values.h"
#include "api_perf_register/ascendc_api_perf.h"
#include "api_perf_register/api_perf_factory.h"
#include "perf_param_v1.h"
namespace att {
namespace {
using namespace ge::sym;
ge::Status GetLoadCase(const NodeDetail &node_info, Expr &data_size, int32_t &use_case) {
  constexpr uint64_t kLoadSizeThres = 25000U;
  auto dims = node_info.input_dims;
  Expr dim_product = accumulate(dims.begin(), dims.end(), CreateExpr(1), [](Expr a, Expr b) { return a * b; });
  auto iter1 = kBlkEleMap.find(node_info.input_dtype[0]);
  GE_ASSERT_TRUE(iter1 != kBlkEleMap.end());
  data_size = dim_product * (kBlkSize / iter1->second);
  if (data_size.IsConstExpr()) {
    uint64_t datasize;
    data_size.GetConstValue(datasize);
    if (datasize >= kLoadSizeThres) {
      use_case = kCaseOne;
    } else if (HasSmallBlockLenWithUbStride(node_info)) {
      use_case = kCaseThree;
    } else {
      use_case = kCaseTwo;
    }
  } else {
    use_case = kCaseDefault;
  }
  GELOGD("Data type is [%s], dim_product[%s], data_size[%s], use_case[%d].", node_info.input_dtype[0].c_str(),
         dim_product.Str().get(), data_size.Str().get(), use_case);
  return ge::SUCCESS;
}

ge::Status GetStoreCase(const NodeDetail &node_info, Expr &case1, Expr &case2, Expr &case3, int32_t &use_case) {
  size_t dim_size = node_info.input_dims.size();
  auto iter1 = kBlkEleMap.find(node_info.input_dtype[0]);
  GE_ASSERT_TRUE(iter1 != kBlkEleMap.end());
  Expr blocklen = node_info.input_dims[dim_size - 1UL];
  Expr blkelem = iter1->second;
  Expr cachelen = CreateExpr(512);   // 待匹配各芯片的cacheline大小
  if (blocklen.IsConstExpr()) {
    int32_t blklen;
    int32_t elelen;
    int32_t cacheline = 512U;   // 待匹配各芯片的cacheline大小
    blocklen.GetConstValue(blklen);
    blkelem.GetConstValue(elelen);
    if (blklen % elelen == 0) {   // 32B对齐
      use_case = kCaseOne;
    } else if (blklen > cacheline) {  // 大于512B非对齐
      use_case = kCaseTwo;
    } else if (blklen > elelen) {   // 大于32B且小于512B非对齐
      use_case = kCaseThree;
    } else {    // 小于32B非对齐
      use_case = kCaseFour;
    }
  } else {
    use_case = kCaseDefault;
    case1 = ge::sym::Mod(blocklen, blkelem);
    case2 = blocklen - blkelem;
    case3 = blocklen - cachelen;
  }
  GELOGD("input dtype is %s, dim_size[%zu], blocklen[%s], blkelem[%s], use_case[%d]", node_info.input_dtype[0].c_str(),
         dim_size, blocklen.Str().get(), blkelem.Str().get(), use_case);
  return ge::SUCCESS;
}

ge::Status GetStorePerf(const NodeDetail &node_info, Expr &res_normal, Expr &res_large_blk, Expr &res_middle_blk,
                        Expr &res_small_blk) {
  Expr res_stride;
  Expr res_continuous;
  GE_ASSERT_SUCCESS(GetPerf(
      {kMoveUbToGm, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride,
       node_info.block_count_idx},
      res_continuous));
  GE_ASSERT_SUCCESS(GetPerf({kMoveUbToGm + "Stride", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride, node_info.block_count_idx},
                            res_stride));
  GE_ASSERT_SUCCESS(GetPerf({kMoveUbToGm + "LargeBlk", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride},
                            res_large_blk));
  GE_ASSERT_SUCCESS(GetPerf({kMoveUbToGm + "MiddleBlk", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride},
                            res_middle_blk));
  GE_ASSERT_SUCCESS(GetPerf({kMoveUbToGm + "SmallBlk", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride},
                            res_small_blk));
  res_normal = res_continuous + res_stride;
  GELOGD("continuous[%s], stride[%s], normal[%s]", res_continuous.Str().get(), res_stride.Str().get(),
         res_normal.Str().get());
  return ge::SUCCESS;
}
}

ge::Status LoadPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res_normal;
  Expr res_stride;
  Expr res_continuous;
  Expr res_small_blk;
  Expr res_ub_stride;
  Expr data_size;
  int32_t use_case = 0;
  GELOGD("Dma with Load: %s", node_info.ToString().c_str());
  GE_ASSERT_SUCCESS(GetPerf({kMoveGmToUb, node_info.input_dtype[0],node_info.output_dtype[0],
                             node_info.input_dims, CreateExpr(0)}, res_continuous));
  GE_ASSERT_SUCCESS(GetPerf({kMoveGmToUb + "Stride", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride, node_info.block_count_idx}, res_stride));
  GE_ASSERT_SUCCESS(GetPerf({kMoveGmToUb + "UbStride", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, CreateExpr(0), node_info.block_count_idx}, res_ub_stride));
  GE_ASSERT_SUCCESS(GetPerf({kMoveGmToUb + "SmallBlk", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, CreateExpr(0)}, res_small_blk));
  GE_ASSERT_SUCCESS(GetLoadCase(node_info, data_size, use_case));
  res_normal = res_continuous + res_stride;
  res_small_blk = res_small_blk + res_stride;
  GELOGD("%s: continuous[%s], stride[%s], ub_stride[%s], normal[%s], small_blk[%s]", node_info.ToString().c_str(),
         res_continuous.Str().get(), res_stride.Str().get(), res_ub_stride.Str().get(), res_normal.Str().get(),
         res_small_blk.Str().get());
  if (use_case == kCaseOne) {
    perf.pipe_res[PipeType::AIV_MTE2] = res_normal;
  } else if (use_case == kCaseTwo) {
    perf.pipe_res[PipeType::AIV_MTE2] = res_small_blk;
  } else if (use_case == kCaseThree) {
    perf.pipe_res[PipeType::AIV_MTE2] = res_ub_stride;
  } else {
    Expr res = CreateExpr("load_node");
    auto data_type_size = kDataTypeSizeMap.find(node_info.input_dtype[0]);
    GE_ASSERT_TRUE(data_type_size != kDataTypeSizeMap.end());
    Expr block_len = Mul(node_info.input_dims[node_info.input_dims.size() - 1UL], data_type_size->second);
    Expr block_len_checker = LogicalAnd({Gt(node_info.ub_stride, CreateExpr(0)), Lt(block_len, kBlkSize)});
    auto normal_case = std::make_shared<IfCase>(res_normal);
    GE_ASSERT_NOTNULL(normal_case);
    auto ub_stride_case = std::make_shared<IfCase>(res_ub_stride);
    GE_ASSERT_NOTNULL(ub_stride_case);
    auto small_blk_case = std::make_shared<IfCase>(res_small_blk);
    GE_ASSERT_NOTNULL(small_blk_case);
    auto res_case = std::make_shared<IfCase>(CondType::K_EQ, block_len_checker, CreateExpr(false),
                                             std::move(small_blk_case), std::move(ub_stride_case));
    GE_ASSERT_NOTNULL(res_case);
    TernaryOp ternary_op = TernaryOp(CondType::K_LT, data_size, kLoadExprThres, std::move(res_case), std::move(normal_case));
    ternary_op.SetVariable(res);
    perf.ternary_ops[res] = ternary_op;
    perf.pipe_res[PipeType::AIV_MTE2] = res;
  }
  return ge::SUCCESS;
}

/*
StorePerf(DataCopy from UB to GM)的性能公式：
  1. 单次MTE3 = S(数据量Byte)/T + h(指令头开销)
  2. 总MTE3 = 单次MTE3 * 调用次数 + H(pipe启动头开销)
  3. blocklen对齐32B时，mte3 = S / (9.96 + 3.79/blockdim)
+ 12.09，针对非连续搬运场景会增加stride建模值(k*(stride%(256)*block_count))
  4. blocklen非对齐32B时，根据对齐值和cacheline划分为三个区域分别建模
  5. blocklen<32B 且 非对齐时，mte3 = S / (9.96 + 3.79/blockdim) + 12.09
  6. 32B<blocklen<512B 且 非对齐时，mte3 = S / 2 + 1.3
  7. blocklen>512B 且 非对齐时，mte3 = S / (9.96 + 3.79/blockdim) + H(非对齐惩罚项) + 12.09
  8. H(非对齐惩罚项) = 512 / 2 - (12.09 + 512 / (9.96 + 3.79/blockdim))
*/
ge::Status StorePerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res_normal;
  Expr res_large_blk;
  Expr res_middle_blk;
  Expr res_small_blk;
  Expr case1;
  Expr case2;
  Expr case3;
  int32_t use_case;

  GELOGD("Dma with store: %s", node_info.ToString().c_str());
  GE_ASSERT_SUCCESS(GetStorePerf(node_info, res_normal, res_large_blk, res_middle_blk, res_small_blk));
  GE_ASSERT_SUCCESS(GetStoreCase(node_info, case1, case2, case3, use_case));
  if (use_case == kCaseOne) {
    perf.pipe_res[PipeType::AIV_MTE3] = res_normal;
  } else if (use_case == kCaseTwo) {
    perf.pipe_res[PipeType::AIV_MTE3] = res_large_blk;
  } else if (use_case == kCaseThree) {
    perf.pipe_res[PipeType::AIV_MTE3] = res_middle_blk;
  } else if (use_case == kCaseFour) {
    perf.pipe_res[PipeType::AIV_MTE3] = res_small_blk;
  } else {
    Expr res = CreateExpr("store_node");
    std::shared_ptr<IfCase> branch_a = std::make_shared<IfCase>(res_normal);
    GE_ASSERT_NOTNULL(branch_a);
    std::shared_ptr<IfCase> branch_b_1 = std::make_shared<IfCase>(res_small_blk);
    std::shared_ptr<IfCase> branch_b_2 = std::make_shared<IfCase>(res_middle_blk);
    std::shared_ptr<IfCase> branch_b_3 = std::make_shared<IfCase>(res_large_blk);
    GE_ASSERT_NOTNULL(branch_b_1);
    GE_ASSERT_NOTNULL(branch_b_2);
    GE_ASSERT_NOTNULL(branch_b_3);
    // blocklen < 32B时走branch_b_1；否则走branch_b_2
    std::shared_ptr<IfCase> branch_c =
        std::make_shared<IfCase>(CondType::K_LT, case2, CreateExpr(0), std::move(branch_b_1), std::move(branch_b_2));
    GE_ASSERT_NOTNULL(branch_c);
    // blocklen < 512B时走branch_c；否则走branch_b_3
    std::shared_ptr<IfCase> branch_b =
        std::make_shared<IfCase>(CondType::K_LT, case3, CreateExpr(0), std::move(branch_c), std::move(branch_b_3));
    GE_ASSERT_NOTNULL(branch_b);
    // blocklen对齐32B时走branch_a；否则走branch_b
    TernaryOp ternary_op = TernaryOp(CondType::K_EQ, case1, CreateExpr(0), std::move(branch_a), std::move(branch_b));
    ternary_op.SetVariable(res);
    perf.ternary_ops[res] = ternary_op;
    perf.pipe_res[PipeType::AIV_MTE3] = res;
  }
  return ge::SUCCESS;
}
REGISTER_ASCENDC_EVAL_FUNC(kLoad, LoadPerf);
REGISTER_ASCENDC_EVAL_FUNC(kStore, StorePerf);
}
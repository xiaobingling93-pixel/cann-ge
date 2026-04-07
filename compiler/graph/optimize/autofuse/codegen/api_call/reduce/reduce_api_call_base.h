/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __AUTOFUSE_REDUCE_API_CALL_BASE_H__
#define __AUTOFUSE_REDUCE_API_CALL_BASE_H__

#include <sstream>
#include "codegen_kernel.h"

namespace reduce_base {
using namespace codegen;

static std::map<std::string, std::pair<int, std::string>> reduce_type_map = {
  {"Min", {ReduceOpType::kMin, "Min"}},  {"Max", {ReduceOpType::kMax, "Max"}},
  {"ArgMax", {ReduceOpType::kMax, "Max"}},
  {"ArgMaxMultiRPhase1", {ReduceOpType::kMax, "Max"}},
  {"ArgMaxMultiRPhase2", {ReduceOpType::kMax, "Max"}},
  {"Any", {ReduceOpType::kAny, "Max"}},  {"All", {ReduceOpType::kAll, "Min"}},
  {"Sum", {ReduceOpType::kSum, "Add"}},  {"Prod", {ReduceOpType::kProd, "Mul"}},
  {"Mean", {ReduceOpType::kMean, "Add"}}
};

void GetIsArAndPattern(const Tensor &y, bool &isAr, std::string &reduce_pattern);
void ReduceMergedSizeCodeGen(const TPipe &tpipe, std::stringstream &ss, const Tensor &src, const Tensor &dst,
                             bool is_tail = false);
bool IsNeedMultiReduce(const Tiler &tiler, const Tensor &input, const Tensor &output, ascir::AxisId axis_id);
void ReduceMeanCodeGen(std::string &dtype_name, const TPipe &tpipe, const Tensor &dst,
                       std::stringstream &ss);
void ReduceInitCodeGen(const Tensor &x, const Tensor &y, const int &type_value,
                       std::stringstream &ss, const TPipe &tpipe, const std::string &dtype_name);
void ReduceDimACodeGen(const Tensor &x, const std::string &apiName, std::stringstream &ss);
Status GetDtypeNameForReduce(const std::string &api_name, const Tensor &x, const Tensor &y, std::string &dtype_name);
void GenAccumulatedOffsetDeclForArgMax(const std::string &api_name, const Tensor &x, const Tensor &y,
                              const TPipe &tpipe, std::stringstream &ss);

/**
 * @brief 生成获取最后两个R轴大小乘积的代码字符串
 * @param x 输入张量
 * @param y 输出张量
 * @param tpipe Tiler对象
 * @param ss 输出字符串流
 */
void GenLastTwoRAxisSizeProductCode(const Tensor &x, const Tensor &y,
                                    const TPipe &tpipe, std::stringstream &ss);
}  // namespace reduce_base
#endif // __AUTOFUSE_REDUCE_API_CALL_BASE_H__
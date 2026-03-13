/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_UTIL_TOOL_FUNC_H_
#define ATT_UTIL_TOOL_FUNC_H_

#include <nlohmann/json.hpp>
#include <fstream>
#include "common_utils.h"
#include "base/att_const_values.h"
#include "api_perf_register/api_perf.h"
#include "base/model_info.h"
#include "gen_model_info/api_perf_register/perf_param.h"
#include "gen_model_info/parser/tuning_space.h"

using Json = nlohmann::json;
namespace att {
// CalculateStride 返回值结构体，包含 stride 表达式和 block_count_idx
struct StrideResult {
  Expr stride;
  int32_t block_count_idx;
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;  // 动态shape的三元表达式

  // 默认构造函数
  StrideResult() : stride(CreateExpr(0)), block_count_idx(0) {}
  // 参数化构造函数
  StrideResult(const Expr &s, int32_t idx) : stride(s), block_count_idx(idx) {}
};
std::unique_ptr<ApiPerf> GetApiPerf(const std::string &node_type);
ge::Status GetPerf(const NodePerfInfo &node_perf_info, Expr &res);
ge::Status SetNodeDetail(const std::vector<TensorShapeInfo> &input_shapes,
                         const std::vector<TensorShapeInfo> &output_shapes, NodeDetail &node_info);
ge::Status SetStride(const TensorShapeInfo &shape_info, NodeDetail &node_info, const int32_t supported_max_dma_len,
                     bool need_swap = false);
ge::Status SetDims(const TensorShapeInfo &tensor_shape_info, NodeDetail &node_info);
ge::Status SetDims(const std::vector<Expr> &dims, NodeDetail &node_info);
ge::Status SetDims(const std::vector<Expr> &input_dims, const std::vector<Expr> &output_dims, NodeDetail &node_info);
NodeDetail GenNodeDetail(const std::string &input_dtype, const std::string &output_dtype,
                         const std::vector<Expr> &dims);
// 获取PerfOutputInfo类型对应pipe的性能开销
Expr GetPipeCost(const PerfOutputInfo &perf_res, const PipeType &pipe_type);
// 获取pipe头开销的函数
PipeHeadPerfFunc GetPipeHeadPerfFunc(PipeType pipe_type);
ge::Status GetApiRegisterVerName(std::string &registered_key_name);
// 判断节点是否有小block len且ub stride大于0
bool HasSmallBlockLenWithUbStride(const NodeDetail &node_info);
// 根据转置信息重排 gm_strides（Load/Store 节点）
ge::Status ReorderGmStrideByTranspose(const ge::AscNodePtr &node, TensorShapeInfo &tensor);
// 计算 stride 的函数，用于性能建模
StrideResult CalculateStride(const TensorShapeInfo &shape_info, const bool is_ub_stride, const NodeDetail &node_info,
                             const int32_t supported_max_dma_len, bool need_swap);
// 合并tensor的连续dim(reduce轴和broadcast轴不会合并)
ge::Status MergeTensorContinuousDims(const ge::AscNodePtr &node, const std::string &tensor_name,
                                     TensorShapeInfo &tensor);
// 指定支持的循环数，获取外派循环轴/使用的轴
ge::Status GetOuterParams(const vector<Expr> &dims, Expr &outer_repeat, vector<Expr> &used_dims,
                          const uint32_t dma_max_len = kDmaMaxLen);
// 获取dma参数，获取外派循环轴/使用的轴，支持根据内轴大小比较进行交换
ge::Status GetDmaParams(const vector<Expr> &dims, Expr &outer_repeat, vector<Expr> &used_dims,
                        int32_t supported_max_dma_len, bool need_swap = false);
// 更新三元表达式的结果到output_res
ge::Status UpdateTenary(PerfOutputInfo &perf_res, PerfOutputInfo &output_res);
ge::Status GetDmaPerf(const TensorShapeInfo &tensor_info, NodeDetail &node_info, PerfOutputInfo &perf_res,
                      int32_t supported_max_dma_len, bool need_swap = false);
inline std::string GetNodeOutTensorName(const ge::AscNodePtr &node, const uint32_t tensor_index) {
  return (node != nullptr) ? (node->GetName() + "_output_" + std::to_string(tensor_index)) : "nil_out";
}
// 获取融合算子头开销
[[nodiscard]] ge::Status GetOpHeadCost(Expr &head_cost);
}
#endif  // ATT_UTIL_TOOL_FUNC_H_


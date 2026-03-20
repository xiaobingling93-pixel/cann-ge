/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_LOOP_OP_OVERRIDES_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_LOOP_OP_OVERRIDES_H_

#include <memory>

#include "graph/anchor.h"
#include "graph/node.h"
#include "loop_common.h"
#include "../op_helper/cube.h"

namespace ge {
namespace loop {
class OpOverrides {
 public:
  virtual ~OpOverrides() = default;

  virtual CseVar Load(const std::string &buffer, const TensorLoopDesc &loop_desc, const Expression &offset) = 0;
  virtual CseVar Load(const std::string &buffer, const TensorLoopDesc &loop_desc,
                      const TensorLoopDesc &loaded_loop_desc, const Expression &offset) = 0;
  virtual CseVar GatherLoad(const std::string &params, const std::string &indices,
              const TensorLoopDesc &loop_desc_params, const TensorLoopDesc &loop_desc_indices, int64_t axis, bool negative_index_support) = 0;
  virtual CseVar Store(const std::string &buffer, const CseVar &src, const TensorLoopDesc &loop_desc,
                       const Expression &offset) = 0;
  virtual CseVar StoreReduction(const std::string &buffer, const CseVar &src, ReduceType reduce_type,
                                const TensorLoopDesc &loop_desc, const Expression &offset) = 0;
  virtual CseVar StoreConcat(const std::string &buffer, const std::vector<CseVar> &inputs,
                             const TensorLoopDesc &loop_desc, const Expression &offset) = 0;

  virtual CseVar StoreSplit(const std::string &buffer, const CseVar &src,
                            const TensorLoopDesc &output_loop_descs,
                            const Expression &offset, size_t output_idx, size_t global_id) = 0;
  virtual CseVar Scalar(const std::string &face, ge::DataType dtype) = 0;
  virtual CseVar LoadSeed(const std::string &name, const CseVar &offset) = 0;
  virtual CseVar ReduceThenBroadcast(const CseVar &x, ReduceType reduce_type, int64_t reduce_dim) = 0;
  virtual CseVar ToDtypeBitcast(const CseVar &x, ge::DataType dst_type, ge::DataType src_type) = 0;
  virtual CseVar LeakyRelu(const CseVar &x, float32_t negative_slope) = 0;
  virtual CseVar Axpy(const CseVar &x1, const CseVar &x2, float32_t alpha) = 0;
  virtual CseVar StoreMatMul(const std::string &buffer, const std::vector<CseVar> &inputs,
                             const TensorLoopDesc &loop_desc, const MatMulAttr &matmul_attr,
                             const std::vector<DataType> &explicit_output_dtypes) = 0;

  virtual CseVar Abs(const CseVar &x) = 0;
  virtual CseVar Acos(const CseVar &x) = 0;
  virtual CseVar Acosh(const CseVar &x) = 0;
  virtual CseVar Asin(const CseVar &x) = 0;
  virtual CseVar Asinh(const CseVar &x) = 0;
  virtual CseVar Atan(const CseVar &x) = 0;
  virtual CseVar Atanh(const CseVar &x) = 0;
  virtual CseVar BesselJ0(const CseVar &x) = 0;
  virtual CseVar BesselJ1(const CseVar &x) = 0;
  virtual CseVar BesselY0(const CseVar &x) = 0;
  virtual CseVar BesselY1(const CseVar &x) = 0;
  virtual CseVar BitwiseNot(const CseVar &x) = 0;
  virtual CseVar Ceil(const CseVar &x) = 0;
  virtual CseVar Cos(const CseVar &x) = 0;
  virtual CseVar Cosh(const CseVar &x) = 0;
  virtual CseVar Erf(const CseVar &x) = 0;
  virtual CseVar Erfc(const CseVar &x) = 0;
  virtual CseVar Erfcx(const CseVar &x) = 0;
  virtual CseVar Erfinv(const CseVar &x) = 0;
  virtual CseVar Exp(const CseVar &x) = 0;
  virtual CseVar Exp2(const CseVar &x) = 0;
  virtual CseVar Expm1(const CseVar &x) = 0;
  virtual CseVar Floor(const CseVar &x) = 0;
  virtual CseVar Frexp(const CseVar &x) = 0;
  virtual CseVar I0(const CseVar &x) = 0;
  virtual CseVar I1(const CseVar &x) = 0;
  virtual CseVar Identity(const CseVar &x) = 0;
  virtual CseVar IsFinite(const CseVar &x) = 0;
  virtual CseVar IsInf(const CseVar &x) = 0;
  virtual CseVar IsNan(const CseVar &x) = 0;
  virtual CseVar Lgamma(const CseVar &x) = 0;
  virtual CseVar LibdeviceAbs(const CseVar &x) = 0;
  virtual CseVar LibdeviceCos(const CseVar &x) = 0;
  virtual CseVar LibdeviceExp(const CseVar &x) = 0;
  virtual CseVar LibdeviceLog(const CseVar &x) = 0;
  virtual CseVar LibdeviceSigmoid(const CseVar &x) = 0;
  virtual CseVar LibdeviceSin(const CseVar &x) = 0;
  virtual CseVar LibdeviceSqrt(const CseVar &x) = 0;
  virtual CseVar Ln(const CseVar &x) = 0;
  virtual CseVar Log(const CseVar &x) = 0;
  virtual CseVar Log10(const CseVar &x) = 0;
  virtual CseVar Log1p(const CseVar &x) = 0;
  virtual CseVar Log2(const CseVar &x) = 0;
  virtual CseVar LogicalNot(const CseVar &x) = 0;
  virtual CseVar ModifiedBesselI0(const CseVar &x) = 0;
  virtual CseVar ModifiedBesselI1(const CseVar &x) = 0;
  virtual CseVar Neg(const CseVar &x) = 0;
  virtual CseVar Reciprocal(const CseVar &x) = 0;
  virtual CseVar Relu(const CseVar &x) = 0;
  virtual CseVar Round(const CseVar &x) = 0;
  virtual CseVar Rsqrt(const CseVar &x) = 0;
  virtual CseVar Sigmoid(const CseVar &x) = 0;
  virtual CseVar Sign(const CseVar &x) = 0;
  virtual CseVar SignBit(const CseVar &x) = 0;
  virtual CseVar Sin(const CseVar &x) = 0;
  virtual CseVar Sinh(const CseVar &x) = 0;
  virtual CseVar SpecialErf(const CseVar &x) = 0;
  virtual CseVar Sqrt(const CseVar &x) = 0;
  virtual CseVar Square(const CseVar &x) = 0;
  virtual CseVar Tan(const CseVar &x) = 0;
  virtual CseVar Tanh(const CseVar &x) = 0;
  virtual CseVar Trunc(const CseVar &x) = 0;
  virtual CseVar Gelu(const CseVar &x) = 0;

  virtual CseVar Add(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar Atan2(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar BitwiseAnd(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar BitwiseLeftshift(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar BitwiseRightshift(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar BitwiseOr(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar BitwiseXor(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar ClampMax(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar ClampMin(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar CopySign(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar Div(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar Eq(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar FloorDiv(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar Fmod(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar Ge(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar Gt(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar Hypot(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar InlineAsmElementwise(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar Le(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar LogicalAnd(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar LogicalOr(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar LogicalXor(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar Lt(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar Maximum(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar Minimum(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar Mul(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar Ne(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar NextAfter(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar Pow(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar Rand(const CseVar &seed, const CseVar &offset) = 0;
  virtual CseVar RandN(const CseVar &seed, const CseVar &offset) = 0;
  virtual CseVar Remainder(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar Sub(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar TrueDiv(const CseVar &x1, const CseVar &x2) = 0;
  virtual CseVar TruncDiv(const CseVar &x1, const CseVar &x2) = 0;

  virtual CseVar ClipByValue(const CseVar &x1, const CseVar &clip_value_min, const CseVar &clip_value_max) = 0;
  virtual CseVar Fma(const CseVar &x1, const CseVar &x2, const CseVar &x3) = 0;
  virtual CseVar Masked(const CseVar &mask, const CseVar &body, const CseVar &other) = 0;
  virtual CseVar Where(const CseVar &x1, const CseVar &x2, const CseVar &x3) = 0;

  virtual CseVar RandInt64(const CseVar &seed, const CseVar &offset, const CseVar &low, const CseVar &high) = 0;

  virtual CseVar Cast(const CseVar &x, ge::DataType dst_type) = 0;
  virtual CseVar Ceil2Int(const CseVar &x, ge::DataType dst_type) = 0;
  virtual CseVar Floor2Int(const CseVar &x, ge::DataType dst_type) = 0;
  virtual CseVar IndexExpr(const CseVar &x, ge::DataType dst_type) = 0;
  virtual CseVar Round2Int(const CseVar &x, ge::DataType dst_type) = 0;
  virtual CseVar Trunc2Int(const CseVar &x, ge::DataType dst_type) = 0;

  // 不同的后端可能实现不同的Cse能力，因此需要从OpOverrides中获取Cse后的Kernel输入与输出
  virtual std::vector<const ge::OutDataAnchor *> GetInputs() const = 0;
  virtual const ge::OutDataAnchor *GetOutput() const = 0;
  // 设置整个Loop的循序轴
  virtual void SetLoopAxis(const LoopAxis &loop_axis) = 0;
  // 每个Loop的输入和输出buffer均有唯一的名字，通过SetBufferSrc设置buffer对应的输入
  void SetBufferSrc(const std::string &name, const ge::OutDataAnchor *src) {
    buffer_src_[name] = src;
  }

 protected:
  const ge::OutDataAnchor *GetBufferSrc(const std::string &name) const {
    const auto iter = buffer_src_.find(name);
    if (iter == buffer_src_.end()) {
      return nullptr;
    }
    return iter->second;
  }

  ge::DataType GetBufferDtype(const std::string &name) const {
    const auto buffer = GetBufferSrc(name);
    if (buffer == nullptr || buffer->GetOwnerNode() == nullptr || buffer->GetOwnerNode()->GetOpDesc() == nullptr) {
      return ge::DT_UNDEFINED;
    }
    auto desc = buffer->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(buffer->GetIdx());
    if (desc == nullptr) {
      return ge::DT_UNDEFINED;
    }
    return desc->GetDataType();
  }

 private:
  std::map<std::string, const ge::OutDataAnchor *> buffer_src_;
};

using OpOverridesPtr = std::shared_ptr<OpOverrides>;

OpOverridesPtr Ops();

class OpsGuard {
 public:
  explicit OpsGuard(OpOverridesPtr overrides);

  ~OpsGuard();

 private:
  OpOverridesPtr prior_;
};
}  // namespace loop
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_LOOP_OP_OVERRIDES_H_

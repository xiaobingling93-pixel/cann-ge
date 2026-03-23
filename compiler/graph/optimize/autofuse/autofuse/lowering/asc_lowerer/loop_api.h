/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_LOOP_API_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_LOOP_API_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "common/checker.h"
#include "graph/node.h"
#include "graph/symbolizer/symbolic.h"
#include "kernel_box.h"
#include "loop_common.h"
#include "loop_ops.h"
#include "utils/autofuse_utils.h"

namespace ge {
namespace loop {
inline bool UnimplementInferDatatype(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  if (!inputs.empty()) {
    outputs.emplace_back(inputs[0]);
  }
  return true;
}

#define ASCIR_INFERDTYPE_2_OP_INFERDTYPE(OP)                                                                   \
  [](const std::vector<DataType> &input_dtypes, std::vector<DataType> &expect_output_dtypes) -> bool {         \
    return AutofuseUtils::CallAscirInferDataType<ascir_op::OP>(input_dtypes, expect_output_dtypes) == SUCCESS; \
  }

#define MAKE_POINTWISE_BASE_INST1(OP, INFERDTYPEKERNEL)                                                               \
  inline LoopVar OP(const LoopVar &x) {                                                                               \
    GE_WARN_ASSERT(x.IsValid());                                                                                      \
    const static LoopKernel kKernel = [](const std::vector<CseVar> &vars) -> CseVar { return Ops()->OP(vars[0]); };   \
    std::vector<LoopOpPtr> inputs = {x.Op()};                                                                         \
    return LoopVar(std::make_shared<PointwiseOp>(kKernel, std::move(inputs), "ops." #OP, nullptr, INFERDTYPEKERNEL)); \
  }                                                                                                                   \
  inline LoopVar OP(const std::vector<loop::LoopVar> &vars) {                                                         \
    GE_WARN_ASSERT(vars.size() == 1);                                                                                 \
    return OP(vars[0]);                                                                                               \
  }

#define MAKE_POINTWISE_INST1(OP) MAKE_POINTWISE_BASE_INST1(OP, ASCIR_INFERDTYPE_2_OP_INFERDTYPE(OP))
#define MAKE_POINTWISE_UNIMPLEMENT_INST1(OP) MAKE_POINTWISE_BASE_INST1(OP, UnimplementInferDatatype)

#define MAKE_POINTWISE_BASE_INST2(OP, INFERDTYPEKERNEL)                                                               \
  inline LoopVar OP(const LoopVar &x1, const LoopVar &x2) {                                                           \
    GE_WARN_ASSERT(x1.IsValid());                                                                                     \
    GE_WARN_ASSERT(x2.IsValid());                                                                                     \
    const static LoopKernel kKernel = [](const std::vector<CseVar> &vars) -> CseVar {                                 \
      return Ops()->OP(vars[0], vars[1]);                                                                             \
    };                                                                                                                \
    std::vector<LoopOpPtr> inputs = {x1.Op(), x2.Op()};                                                               \
    return LoopVar(std::make_shared<PointwiseOp>(kKernel, std::move(inputs), "ops." #OP, nullptr, INFERDTYPEKERNEL)); \
  }                                                                                                                   \
  inline LoopVar OP(const std::vector<loop::LoopVar> &vars) {                                                         \
    GE_WARN_ASSERT(vars.size() == 2);                                                                                 \
    return OP(vars[0], vars[1]);                                                                                      \
  }

#define MAKE_POINTWISE_INST2(OP) MAKE_POINTWISE_BASE_INST2(OP, ASCIR_INFERDTYPE_2_OP_INFERDTYPE(OP))
#define MAKE_POINTWISE_UNIMPLEMENT_INST2(OP) MAKE_POINTWISE_BASE_INST2(OP, UnimplementInferDatatype)

#define MAKE_POINTWISE_BASE_INST3(OP, INFERDTYPEKERNEL)                                                               \
  inline LoopVar OP(const LoopVar &x1, const LoopVar &x2, const LoopVar &x3) {                                        \
    GE_WARN_ASSERT(x1.IsValid());                                                                                     \
    GE_WARN_ASSERT(x2.IsValid());                                                                                     \
    GE_WARN_ASSERT(x3.IsValid());                                                                                     \
    const static LoopKernel kKernel = [](const std::vector<CseVar> &vars) -> CseVar {                                 \
      return Ops()->OP(vars[0], vars[1], vars[2]);                                                                    \
    };                                                                                                                \
    std::vector<LoopOpPtr> inputs = {x1.Op(), x2.Op(), x3.Op()};                                                      \
    return LoopVar(std::make_shared<PointwiseOp>(kKernel, std::move(inputs), "ops." #OP, nullptr, INFERDTYPEKERNEL)); \
  }                                                                                                                   \
  inline LoopVar OP(const std::vector<loop::LoopVar> &vars) {                                                         \
    GE_WARN_ASSERT(vars.size() == 3);                                                                                 \
    return OP(vars[0], vars[1], vars[2]);                                                                             \
  }

#define MAKE_POINTWISE_INST3(OP) MAKE_POINTWISE_BASE_INST3(OP, ASCIR_INFERDTYPE_2_OP_INFERDTYPE(OP))
#define MAKE_POINTWISE_UNIMPLEMENT_INST3(OP) MAKE_POINTWISE_BASE_INST3(OP, UnimplementInferDatatype)

#define MAKE_POINTWISE_BASE_INST4(OP, INFERDTYPEKERNEL)                                                               \
  inline LoopVar OP(const LoopVar &x1, const LoopVar &x2, const LoopVar &x3, const LoopVar &x4) {                     \
    GE_WARN_ASSERT(x1.IsValid());                                                                                     \
    GE_WARN_ASSERT(x2.IsValid());                                                                                     \
    GE_WARN_ASSERT(x3.IsValid());                                                                                     \
    GE_WARN_ASSERT(x4.IsValid());                                                                                     \
    const static LoopKernel kKernel = [](const std::vector<CseVar> &vars) -> CseVar {                                 \
      return Ops()->OP(vars[0], vars[1], vars[2], vars[3]);                                                           \
    };                                                                                                                \
    std::vector<LoopOpPtr> inputs = {x1.Op(), x2.Op(), x3.Op(), x4.Op()};                                             \
    return LoopVar(std::make_shared<PointwiseOp>(kKernel, std::move(inputs), "ops." #OP, nullptr, INFERDTYPEKERNEL)); \
  }                                                                                                                   \
  inline LoopVar OP(const std::vector<loop::LoopVar> &vars) {                                                         \
    GE_WARN_ASSERT(vars.size() == 4);                                                                                 \
    return OP(vars[0], vars[1], vars[2], vars[3]);                                                                    \
  }

#define MAKE_POINTWISE_UNIMPLEMENT_INST4(OP) MAKE_POINTWISE_BASE_INST4(OP, UnimplementInferDatatype)

#define MAKE_POINTWISE_DTYPE_BASE_INST1(OP, INFERDTYPEKERNEL)                                                      \
  inline LoopVar OP(const LoopVar &x, ge::DataType data_type) {                                                    \
    GE_WARN_ASSERT(x.IsValid());                                                                                   \
    std::string op_name = "ops." #OP;                                                                              \
    LoopKernel kernel = [data_type](const std::vector<CseVar> &vars) -> CseVar {                                   \
      return Ops()->OP(vars[0], data_type);                                                                        \
    };                                                                                                             \
    PrintKernel readable = [data_type, op_name](const std::vector<std::string> &var_names) -> std::string {        \
      std::stringstream ss;                                                                                        \
      ss << op_name.c_str() << "(" << var_names[0] << ", " << TypeUtils::DataTypeToSerialString(data_type) << ")"; \
      return ss.str();                                                                                             \
    };                                                                                                             \
    std::vector<LoopOpPtr> inputs = {x.Op()};                                                                      \
    InferDtypeKernel inferdtype = [data_type](const std::vector<DataType> &inputs,                                 \
                                              std::vector<DataType> &outputs) -> Status {                          \
      outputs.emplace_back(data_type);                                                                             \
      return INFERDTYPEKERNEL(inputs, outputs);                                                                    \
    };                                                                                                             \
    return LoopVar(std::make_shared<PointwiseOp>(kernel, inputs, op_name.c_str(), readable, inferdtype));          \
  }

#define MAKE_POINTWISE_DTYPE_INST1(OP) MAKE_POINTWISE_DTYPE_BASE_INST1(OP, ASCIR_INFERDTYPE_2_OP_INFERDTYPE(OP))
#define MAKE_POINTWISE_DTYPE_UNIMPLEMENT_INST1(OP) MAKE_POINTWISE_DTYPE_BASE_INST1(OP, UnimplementInferDatatype)

#define LOWERING_ASSERT_NOTNULL(PRT)                 \
  if ((PRT) == nullptr) {                                               \
    GELOGW("Create default kernelbox, because the " #PRT " is null"); \
    return KernelBox(nullptr, std::make_shared<KernelBoxMeta>());      \
  }

LoopVar Load(const ge::InDataAnchorPtr &src);
LoopVar GatherLoad(const ge::OutDataAnchorPtr &dst, const ge::InDataAnchorPtr &params,
    const ge::InDataAnchorPtr &indices, int64_t axis, bool negative_index_support);
KernelBox Store(const ge::OutDataAnchorPtr &dst, const LoopVar &src);
KernelBox StoreReduction(ReduceType type, const ge::OutDataAnchorPtr &dst, const LoopVar &src,
                         const std::vector<Expression> &src_dims, const std::vector<size_t> &reduced_axis);
KernelBox StoreConcat(const ge::OutDataAnchorPtr &dst, const std::vector<ge::InDataAnchorPtr> &inputs,
                      size_t concat_dim);
KernelBox StorePack(const ge::OutDataAnchorPtr &dst, const std::vector<ge::InDataAnchorPtr> &inputs,
                    int64_t packed_dim);
KernelBox StoreExtern(const ge::OutDataAnchorPtr &dst);
KernelBox GetKernelBox(const ge::OutDataAnchorPtr &dst);

LoopVar Scalar(std::string face, ge::DataType dtype);
LoopVar Broadcast(const LoopVar &op, std::vector<BroadcastOp::DimKind> status);
LoopVar Broadcast(const LoopVar &op, std::vector<Expression> src, std::vector<Expression> dst);
LoopVar Permute(const LoopVar &op, std::vector<size_t> order);
LoopVar Transpose(const LoopVar &op, const std::vector<ge::Expression> &dims, const std::vector<int64_t> &perm);
LoopVar Squeeze(const LoopVar &op, int64_t dim);
LoopVar Unsqueeze(const LoopVar &op, int64_t dim);
LoopVar Reshape(const LoopVar &op, const std::vector<Expression> &src_dims, const std::vector<Expression> &dst_dims);
LoopVar LoadSeed(const std::string &name, const LoopVar &offset);
LoopVar ReduceThenBroadcast(ReduceType type, const LoopVar &op, int64_t dim);
LoopVar ToDtypeBitcast(const LoopVar &x, ge::DataType dst_type, ge::DataType src_type);
LoopVar LeakyRelu(const LoopVar &x, float32_t negative_slope);
LoopVar Axpy(const LoopVar &x1, const LoopVar &x2, float32_t alpha);
LoopVar StoreStridedSlice(const ge::OutDataAnchorPtr &dst, const InDataAnchorPtr &src,
                          const std::vector<Expression> &start, const std::vector<Expression> &stride,
                          const std::vector<Expression> &input_dims, string &not_lowering_reason);
std::vector<LoopVar> StoreSplit(const std::vector<OutDataAnchorPtr> &outputs,
                                const InDataAnchorPtr &src, size_t split_dim,
                                string &not_lowering_reason);
KernelBox StoreMatMul(const ge::OutDataAnchorPtr &dst, const std::vector<ge::InDataAnchorPtr> &inputs,
                      const MatMulAttr &matmul_attr);

MAKE_POINTWISE_INST1(Abs);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Acos);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Acosh);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Asin);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Asinh);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Atan);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Atanh);
MAKE_POINTWISE_UNIMPLEMENT_INST1(BesselJ0);
MAKE_POINTWISE_UNIMPLEMENT_INST1(BesselJ1);
MAKE_POINTWISE_UNIMPLEMENT_INST1(BesselY0);
MAKE_POINTWISE_UNIMPLEMENT_INST1(BesselY1);
MAKE_POINTWISE_UNIMPLEMENT_INST1(BitwiseNot);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Ceil);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Cos);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Cosh);
MAKE_POINTWISE_INST1(Erf);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Erfc);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Erfcx);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Erfinv);
MAKE_POINTWISE_INST1(Exp);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Exp2);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Expm1);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Floor);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Frexp);
MAKE_POINTWISE_UNIMPLEMENT_INST1(I0);
MAKE_POINTWISE_UNIMPLEMENT_INST1(I1);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Identity);
MAKE_POINTWISE_INST1(IsFinite);
MAKE_POINTWISE_BASE_INST1(IsNan, ASCIR_INFERDTYPE_2_OP_INFERDTYPE(Isnan));
MAKE_POINTWISE_UNIMPLEMENT_INST1(IsInf);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Lgamma);
MAKE_POINTWISE_UNIMPLEMENT_INST1(LibdeviceAbs);
MAKE_POINTWISE_UNIMPLEMENT_INST1(LibdeviceCos);
MAKE_POINTWISE_UNIMPLEMENT_INST1(LibdeviceExp);
MAKE_POINTWISE_UNIMPLEMENT_INST1(LibdeviceLog);
MAKE_POINTWISE_UNIMPLEMENT_INST1(LibdeviceSigmoid);
MAKE_POINTWISE_UNIMPLEMENT_INST1(LibdeviceSin);
MAKE_POINTWISE_UNIMPLEMENT_INST1(LibdeviceSqrt);
MAKE_POINTWISE_INST1(Ln);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Log);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Log10);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Log1p);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Log2);
MAKE_POINTWISE_INST1(LogicalNot);
MAKE_POINTWISE_UNIMPLEMENT_INST1(ModifiedBesselI0);
MAKE_POINTWISE_UNIMPLEMENT_INST1(ModifiedBesselI1);
MAKE_POINTWISE_INST1(Neg);
MAKE_POINTWISE_INST1(Reciprocal);
MAKE_POINTWISE_INST1(Relu);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Round);
MAKE_POINTWISE_INST1(Rsqrt);
MAKE_POINTWISE_INST1(Sigmoid);
MAKE_POINTWISE_INST1(Sign);
MAKE_POINTWISE_UNIMPLEMENT_INST1(SignBit);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Sin);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Sinh);
MAKE_POINTWISE_UNIMPLEMENT_INST1(SpecialErf);
MAKE_POINTWISE_INST1(Sqrt);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Square);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Tan);
MAKE_POINTWISE_INST1(Tanh);
MAKE_POINTWISE_UNIMPLEMENT_INST1(Trunc);
MAKE_POINTWISE_INST1(Gelu);

MAKE_POINTWISE_INST2(Add);
MAKE_POINTWISE_UNIMPLEMENT_INST2(Atan2);
MAKE_POINTWISE_INST2(BitwiseAnd);
MAKE_POINTWISE_UNIMPLEMENT_INST2(BitwiseLeftshift);
MAKE_POINTWISE_UNIMPLEMENT_INST2(BitwiseRightshift);
MAKE_POINTWISE_UNIMPLEMENT_INST2(BitwiseOr);
MAKE_POINTWISE_UNIMPLEMENT_INST2(BitwiseXor);
MAKE_POINTWISE_UNIMPLEMENT_INST2(ClampMax);
MAKE_POINTWISE_UNIMPLEMENT_INST2(ClampMin);
MAKE_POINTWISE_UNIMPLEMENT_INST2(CopySign);
MAKE_POINTWISE_INST2(Div);
MAKE_POINTWISE_INST2(Eq);
MAKE_POINTWISE_INST2(FloorDiv);
MAKE_POINTWISE_UNIMPLEMENT_INST2(Fmod);
MAKE_POINTWISE_INST2(Ge);
MAKE_POINTWISE_INST2(Gt);
MAKE_POINTWISE_UNIMPLEMENT_INST2(Hypot);
MAKE_POINTWISE_UNIMPLEMENT_INST2(InlineAsmElementwise);
MAKE_POINTWISE_INST2(Le);
MAKE_POINTWISE_INST2(LogicalAnd);
MAKE_POINTWISE_INST2(LogicalOr);
MAKE_POINTWISE_UNIMPLEMENT_INST2(LogicalXor);
MAKE_POINTWISE_INST2(Lt);
MAKE_POINTWISE_INST2(Maximum);
MAKE_POINTWISE_INST2(Minimum);
MAKE_POINTWISE_INST2(Mul);
MAKE_POINTWISE_INST2(Ne);
MAKE_POINTWISE_UNIMPLEMENT_INST2(NextAfter);
MAKE_POINTWISE_INST2(Pow);
MAKE_POINTWISE_UNIMPLEMENT_INST2(Rand);
MAKE_POINTWISE_UNIMPLEMENT_INST2(RandN);
MAKE_POINTWISE_UNIMPLEMENT_INST2(Remainder);
MAKE_POINTWISE_INST2(Sub);
MAKE_POINTWISE_INST2(TrueDiv);
MAKE_POINTWISE_UNIMPLEMENT_INST2(TruncDiv);

MAKE_POINTWISE_INST3(ClipByValue);
MAKE_POINTWISE_UNIMPLEMENT_INST3(Fma);
MAKE_POINTWISE_UNIMPLEMENT_INST3(Masked);
MAKE_POINTWISE_INST3(Where);

MAKE_POINTWISE_UNIMPLEMENT_INST4(RandInt64);

MAKE_POINTWISE_DTYPE_INST1(Cast);
MAKE_POINTWISE_DTYPE_UNIMPLEMENT_INST1(Ceil2Int);
MAKE_POINTWISE_DTYPE_UNIMPLEMENT_INST1(Floor2Int);
MAKE_POINTWISE_DTYPE_UNIMPLEMENT_INST1(IndexExpr);
MAKE_POINTWISE_DTYPE_UNIMPLEMENT_INST1(Round2Int);
MAKE_POINTWISE_DTYPE_UNIMPLEMENT_INST1(Trunc2Int);
}  // namespace loop
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_LOOP_API_H_

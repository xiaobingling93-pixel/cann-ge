/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_ASC_OVERRIDES_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_ASC_OVERRIDES_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>
#include <thread>

#include "ascendc_ir.h"
#include "common/checker.h"
#include "graph/ascendc_ir/ascendc_ir_check.h"
#include "graph/expression/const_values.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "loop_common.h"
#include "loop_op_overrides.h"
#include "ascir_ops.h"
#include "utils/autofuse_utils.h"

namespace ge {
namespace loop {
#define GET_IR_ATTR_UNIQUE_(C, T, obj)                                                        \
  auto ir_attr_##C = dynamic_cast<ascir_op::T::Asc##T##IrAttrDef *>(obj->attr.ir_attr.get()); \
  GE_WARN_ASSERT(ir_attr_##C != nullptr);                                                     \
  ir_attr_##C
#define GET_IR_ATTR_UNIQUE(C, T, obj) GET_IR_ATTR_UNIQUE_(C, T, obj)
#define GET_IR_ATTR(T, obj) GET_IR_ATTR_UNIQUE(__COUNTER__, T, obj)

#define SET_BATCH_MATMUL_ATTRS(OpType, Obj, Attr) \
    GE_WARN_ASSERT(Obj.Src() != nullptr); \
    GET_IR_ATTR(OpType, Obj.Src())->SetAdj_x1(static_cast<int64_t>(Attr.adj_x1)); \
    GET_IR_ATTR(OpType, Obj.Src())->SetAdj_x2(static_cast<int64_t>(Attr.adj_x2)); \
    GET_IR_ATTR(OpType, Obj.Src())->SetOffset_x(Attr.offset_x); \
    GET_IR_ATTR(OpType, Obj.Src())->SetEnable_hf32(Attr.enable_hf32); \
    GET_IR_ATTR(OpType, Obj.Src())->SetHas_relu(0);

#define SET_MATMUL_ATTRS(OpType, Obj, Attr) \
    GE_WARN_ASSERT(Obj.Src() != nullptr); \
    GET_IR_ATTR(OpType, (Obj).Src())->SetTranspose_x1(static_cast<int64_t>((Attr).transpose_x1)); \
    GET_IR_ATTR(OpType, (Obj).Src())->SetTranspose_x2(static_cast<int64_t>((Attr).transpose_x2)); \
    GET_IR_ATTR(OpType, Obj.Src())->SetOffset_x(Attr.offset_x); \
    GET_IR_ATTR(OpType, Obj.Src())->SetEnable_hf32(Attr.enable_hf32); \
    GET_IR_ATTR(OpType, Obj.Src())->SetHas_relu(0);

template <typename T>
bool InferAscirDataType(const std::vector<DataType> &input_dtypes, std::vector<DataType> &output_dtypes) {
  if (ge::AutofuseUtils::CallAscirInferDataType<T>(input_dtypes, output_dtypes)== SUCCESS) {
    return true;
  }
  // fp16,bf16可通过精度提升转成fp32,可以跳过校验推导输出datatype；ascir部分op不支持fp32，仅让部分op进行nocheck推导
  const std::set<string> reinfer_op = {"Sum", "Max", "Mean", "Min", "Prod"};
  if (reinfer_op.find(T::Type) != reinfer_op.end() &&
      std::any_of(input_dtypes.begin(), input_dtypes.end(),
                  [](const ge::DataType &dtype) { return (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16); })) {
    GELOGI("Asc ir %s Inputs dtype is fp16 or bf16, infer dtype with nocheck", T::Type);
    if (T::InferDataTypeWithNoCheck(input_dtypes, output_dtypes) == SUCCESS) {
      return true;
    }
  }
  return false;
}

/**
 * AscVar是在asc的loop overrides执行过程中，用于表达Ascend IR的输出值的变量，
 * 它实际代表着一个AscIR Node的输出值
 */
class AscVar : public Var {
 public:
  explicit AscVar(AscNodePtr node, const size_t index = 0) : node_(std::move(node)), index_(index) {
    try {
      if (node_ != nullptr) {
        tensor_ = &node_->outputs[index_];
      }
    } catch (const AscIRException &) {
      tensor_ = nullptr;
    }
  }
  AscVar(const AscVar &other) = default;
  AscVar &operator=(const AscVar &) = default;
  AscVar() : node_(nullptr), index_(0), tensor_(nullptr) {};
  ~AscVar() override = default;

  std::string Name() override {
    if (node_ == nullptr) {
      return "Asc(NULL)";
    }
    return "Asc(" + node_->GetName() + ":" + std::to_string(index_) + ")";
  }

  [[nodiscard]] bool IsValid() const override {
    return tensor_ != nullptr && tensor_->attr.dtype != ge::DT_UNDEFINED;
  }

  void SetTensorLoop(std::vector<int64_t> axis, std::vector<Expression> repeats, std::vector<Expression> strides) const {
    if (!IsValid()) {
      return;
    }
    tensor_->attr.axis = std::move(axis);
    tensor_->attr.repeats = std::move(repeats);
    tensor_->attr.strides = std::move(strides);
  }

  [[nodiscard]] ge::DataType Dtype() const {
    if (!IsValid()) {
      return ge::DT_UNDEFINED;
    }
    return tensor_->attr.dtype;
  }

  [[nodiscard]] const AscNodePtr &Src() const {
    return node_;
  }

  [[nodiscard]] OutDataAnchorPtr Buffer() const {
    if (!IsValid()) {
      return nullptr;
    }
    return node_->GetOutDataAnchor(static_cast<int32_t>(index_));
  }

  [[nodiscard]] std::shared_ptr<AscVar> Shared() const {
    return std::make_shared<AscVar>(*this);
  }

 private:
  AscNodePtr node_;
  size_t index_;
  AscTensor *tensor_;
};

class AscOut : public Var {
 public:
  explicit AscOut(AscNodePtr node) : node_(std::move(node)) {}
  AscOut(const AscOut &other) = default;
  AscOut &operator=(const AscOut &) = default;
  AscOut() : node_(nullptr) {}
  ~AscOut() override = default;
  bool IsValid() const override {
    return node_ != nullptr;
  }

  std::string Name() override {
    if (node_ == nullptr) {
      return "AscOut(NULL)";
    }
    return "AscOut(" + node_->GetName() + ")";
  }

  [[nodiscard]] std::shared_ptr<AscOut> Shared() const {
    return std::make_shared<AscOut>(node_);
  }

 private:
  AscNodePtr node_;
};
/**
 * UnimplementOp是在asc的loop overrides执行过程中，用于获取未实现的Ascend IR的CseKey值。
 */
class UnimplementOp {
 public:
  static constexpr const char *Type = "Unimplement";
};
/**
 * AscOverrides是为Asc codegen后端提供的Loop overrides，完成Loop表达到AscGraph的转换。
 */
class AscOverrides final : public OpOverrides {
 public:
  explicit AscOverrides(const char *name, bool auto_cse = true)
      : graph_(std::make_shared<AscGraph>(name)), auto_cse_(auto_cse) {}
  ~AscOverrides() override = default;

  [[nodiscard]] const AscGraph &Graph() const {
    return *graph_;
  }

  [[nodiscard]] const std::shared_ptr<AscGraph> &SharedGraph() const {
    return graph_;
  }

  void SetLoopAxis(const LoopAxis &loop_axis) override {
    asc_axis_.reserve(loop_axis.axis.size());
    asc_axis_repeats_.reserve(loop_axis.axis.size());
    for (size_t i = 0U; i < loop_axis.axis.size(); i++) {
      const auto &asc_axis = graph_->CreateAxis(loop_axis.names[i], loop_axis.repeats[i]);
      asc_axis_.push_back(asc_axis.id);
      asc_axis_repeats_.push_back(loop_axis.repeats[i]);
    }
  }

  [[nodiscard]] bool IsScalarGraph() const {
    return asc_axis_.empty() ||
           (!asc_axis_.empty() &&
            std::all_of(asc_axis_repeats_.begin(), asc_axis_repeats_.end(), [](const Expression &exp) -> bool {
              return (SymbolicUtils::StaticCheckEq(exp, sym::kSymbolOne) == TriBool::kTrue);
            }));
  }

  [[nodiscard]] bool IsAscAxisEmpty() const {
    return asc_axis_.empty();
  }

  [[nodiscard]] std::vector<const ge::OutDataAnchor *> GetInputs() const override {
    return inputs_;
  }

  [[nodiscard]] const ge::OutDataAnchor *GetOutput() const override {
    return output_;
  }

  CseVar Load(const std::string &buffer, const TensorLoopDesc &loop_desc, const TensorLoopDesc &loaded_loop_desc,
              const Expression &offset) override {
    const auto cse_key = CseKey<ascir_op::Load>(buffer, loop_desc, offset);
    auto cached = LookUp(cse_key);
    if (cached.IsValid()) {
      return cached;
    }
    auto data = AddInput(buffer);
    GE_WARN_ASSERT(loop_desc.repeats.size() == asc_axis_.size());
    GE_WARN_ASSERT(loop_desc.strides.size() == asc_axis_.size());
    data.SetTensorLoop(asc_axis_, loop_desc.repeats, loop_desc.strides);

    auto load = MakeAscVar<ascir_op::Load>(data);
    load.SetTensorLoop(asc_axis_, loaded_loop_desc.repeats, loaded_loop_desc.strides);
    GE_WARN_ASSERT(load.Src() != nullptr);
    GET_IR_ATTR(Load, load.Src())->SetOffset(offset);
    return MakeCse(cse_key, load);
  }

  CseVar Load(const std::string &buffer, const TensorLoopDesc &loop_desc, const Expression &offset) override {
    return Load(buffer, loop_desc, TensorLoopDesc(asc_axis_repeats_, ContiguousStrides(asc_axis_repeats_)), offset);
  }

  CseVar GatherLoad(const std::string &params, const std::string &indices,
            const TensorLoopDesc &loop_desc_params, const TensorLoopDesc &loop_desc_indices, int64_t axis, bool negative_index_support) override {
    const auto cse_key = CseKey<ascir_op::Gather>(params, indices, loop_desc_params, loop_desc_indices, axis, negative_index_support); // 所有入参
    auto cached = LookUp(cse_key);
    if (cached.IsValid()) {
      return cached;
    }

    // add data
    std::vector<int64_t> params_axis(loop_desc_params.repeats.size());
    for (size_t i = 0; i < loop_desc_params.repeats.size(); i ++ ) {
      params_axis[i] = i;
    }
    auto data_params = AddInput(params);
    data_params.SetTensorLoop(params_axis, loop_desc_params.repeats, loop_desc_params.strides);

    std::vector<int64_t> indices_axis(loop_desc_indices.repeats.size());
    for (size_t i = 0; i < loop_desc_indices.repeats.size(); i ++ ) {
      indices_axis[i] = i;
    }
    auto data_indices = AddInput(indices);
    data_indices.SetTensorLoop(indices_axis, loop_desc_indices.repeats, loop_desc_indices.strides);

    auto gatherload = MakeAscVar<ascir_op::Gather>(data_params, data_indices);

    GE_WARN_ASSERT(gatherload.Src() != nullptr);
    GET_IR_ATTR(Gather, gatherload.Src())->SetAxis(axis);
    GET_IR_ATTR(Gather, gatherload.Src())->SetNegative_index_support(negative_index_support);

    gatherload.SetTensorLoop(asc_axis_, asc_axis_repeats_, ContiguousStrides(asc_axis_repeats_));
    return MakeCse(cse_key, gatherload);
  }

  CseVar Store(const std::string &buffer, const CseVar &src, const TensorLoopDesc &loop_desc,
               const Expression &offset) override {
    static_cast<void>(offset);
    GE_WARN_ASSERT(loop_desc.repeats.size() == asc_axis_.size());
    GE_WARN_ASSERT(loop_desc.strides.size() == asc_axis_.size());
    GE_WARN_ASSERT(src.IsValid());
    return CseVar(AddOutput(buffer, *src.Get<AscVar>(), loop_desc.repeats, loop_desc.strides).Shared());
  }

  CseVar StoreReduction(const std::string &buffer, const CseVar &src, ReduceType reduce_type,
                        const TensorLoopDesc &loop_desc, const Expression &offset) override {
    static_cast<void>(offset);
    GE_WARN_ASSERT(loop_desc.repeats.size() == asc_axis_.size());
    GE_WARN_ASSERT(loop_desc.strides.size() == asc_axis_.size());
    AscVar reduce_op = MakeReduceVar(src, reduce_type);
    GE_WARN_ASSERT(reduce_op.IsValid());

    reduce_op.SetTensorLoop(asc_axis_, loop_desc.repeats, loop_desc.strides);
    return CseVar(AddOutput(buffer, reduce_op, loop_desc.repeats, loop_desc.strides).Shared());
  }

  CseVar StoreConcat(const std::string &buffer, const std::vector<CseVar> &inputs, const TensorLoopDesc &loop_desc,
                     const Expression &offset) override {
    static_cast<void>(offset);
    GE_WARN_ASSERT(loop_desc.repeats.size() == asc_axis_.size());
    GE_WARN_ASSERT(loop_desc.strides.size() == asc_axis_.size());
    auto concat = MakeAscVar<ascir_op::Concat>(inputs);
    concat.SetTensorLoop(asc_axis_, loop_desc.repeats, loop_desc.strides);
    return CseVar(AddOutput(buffer, concat, loop_desc.repeats, loop_desc.strides).Shared());
  }
  CseVar StoreSplit(const std::string &buffer, const CseVar &src,
                    const TensorLoopDesc &output_loop_desc,
                    const Expression &offset, size_t output_idx, size_t global_id) override {
    static_cast<void>(offset);
    CseVar var;
    GE_WARN_ASSERT(src.Get<AscVar>() != nullptr);
    const auto node = MakeRawNode<ascir_op::Split>(ToVector(src), {},{},{1});
    GET_IR_ATTR(Split, node)->SetIndex(output_idx);
    GET_IR_ATTR(Split, node)->SetGid(global_id);
    GELOGI("ascir node: %s(%s), output index: %zu, global id: %zu", node->GetType().c_str(),node->GetName().c_str(), output_idx, global_id);
    GE_WARN_ASSERT(node != nullptr);
    CseVar ret;

    auto split_out = AscVar(node);
    split_out.SetTensorLoop(asc_axis_, output_loop_desc.repeats, output_loop_desc.strides);
    const auto cse_key = CseKey<ascir_op::Split>(src, output_loop_desc, "Output_idx = " + std::to_string(output_idx), offset);
    auto cached = LookUp(cse_key);
    CseVar tmp;
    if (cached.IsValid()) {
      tmp = cached;
    }
    else {
      tmp = MakeCse(cse_key,split_out);
    }
    ret = CseVar(AddOutput(buffer, split_out, output_loop_desc.repeats, output_loop_desc.strides).Shared());

    return ret;
  }

  CseVar StoreMatMul(const std::string &buffer, const std::vector<CseVar> &inputs, const TensorLoopDesc &loop_desc,
                     const MatMulAttr &matmul_attr, const std::vector<DataType> &explicit_output_dtypes) override {
    GE_WARN_ASSERT(loop_desc.repeats.size() == asc_axis_.size());
    GE_WARN_ASSERT(loop_desc.strides.size() == asc_axis_.size());
    GE_WARN_ASSERT((inputs.size() > 1U) && (inputs.size() < 5U));
    AscVar mm;
    if (inputs.size() == 2U) {
      if (matmul_attr.is_batch) {
        mm = MakeAscVar<ascir_op::BatchMatMul>(explicit_output_dtypes, inputs[0], inputs[1]);
        SET_BATCH_MATMUL_ATTRS(BatchMatMul, mm, matmul_attr);
      } else {
        mm = MakeAscVar<ascir_op::MatMul>(explicit_output_dtypes, inputs[0], inputs[1]);
        SET_MATMUL_ATTRS(MatMul, mm, matmul_attr);
      }
    } else if (inputs.size() == 3U) {
      if (matmul_attr.has_bias) {
        if (matmul_attr.is_batch) {
          mm = MakeAscVar<ascir_op::BatchMatMulBias>(explicit_output_dtypes, inputs[0], inputs[1], inputs[2]);
          SET_BATCH_MATMUL_ATTRS(BatchMatMulBias, mm, matmul_attr);
        } else {
          mm = MakeAscVar<ascir_op::MatMulBias>(explicit_output_dtypes, inputs[0], inputs[1], inputs[2]);
          SET_MATMUL_ATTRS(MatMulBias, mm, matmul_attr);
        }
      } else if (matmul_attr.has_offset_w) {
        if (matmul_attr.is_batch) {
          mm = MakeAscVar<ascir_op::BatchMatMulOffset>(explicit_output_dtypes, inputs[0], inputs[1], inputs[2]);
          SET_BATCH_MATMUL_ATTRS(BatchMatMulOffset, mm, matmul_attr);
        } else {
          mm = MakeAscVar<ascir_op::MatMulOffset>(explicit_output_dtypes, inputs[0], inputs[1], inputs[2]);
          SET_MATMUL_ATTRS(MatMulOffset, mm, matmul_attr);
        }
      } else {
        GE_WARN_ASSERT(false, "Matmul attr info not match, input=3, bias=false, offset=false.");
      }
    } else if (inputs.size() == 4U) {
      if (matmul_attr.is_batch) {
        mm = MakeAscVar<ascir_op::BatchMatMulOffsetBias>(explicit_output_dtypes, inputs[0], inputs[1], inputs[2],
                                                         inputs[3]);
        SET_BATCH_MATMUL_ATTRS(BatchMatMulOffsetBias, mm, matmul_attr);
      } else {
        mm = MakeAscVar<ascir_op::MatMulOffsetBias>(explicit_output_dtypes, inputs[0], inputs[1], inputs[2], inputs[3]);
        SET_MATMUL_ATTRS(MatMulOffsetBias, mm, matmul_attr);
      }
    }
    mm.SetTensorLoop(asc_axis_, loop_desc.repeats, loop_desc.strides);
    return CseVar(AddOutput(buffer, mm, loop_desc.repeats, loop_desc.strides).Shared());
  }

  CseVar Scalar(const std::string &face, ge::DataType dtype) override {
    if (dtype == ge::DT_BOOL) {
      dtype = ge::DT_UINT8;
    }
    const auto cse_key = CseKey<ascir_op::Scalar>(face, dtype);
    auto cached = LookUp(cse_key);
    if (cached.IsValid()) {
      return cached;
    }
    auto op = MakeAscVar<ascir_op::Scalar>(dtype);
    GE_WARN_ASSERT(op.Src() != nullptr);
    GET_IR_ATTR(Scalar, op.Src())->SetValue(face);
    return MakeCse(cse_key, op);
  }

  CseVar ReduceThenBroadcast(const CseVar &src, ReduceType reduce_type, int64_t reduce_dim) override {
    const auto cse_key = CseKey<ascir_op::Scalar>(src, reduce_type, reduce_dim);
    auto cached = LookUp(cse_key);
    if (cached.IsValid()) {
      return cached;
    }
    AscVar reduce_op = MakeReduceVar(src, reduce_type);
    GE_WARN_ASSERT(reduce_op.IsValid());

    auto repeats = asc_axis_repeats_;
    reduce_dim = reduce_dim < 0 ? reduce_dim + static_cast<int64_t>(repeats.size()) : reduce_dim;
    GE_WARN_ASSERT(reduce_dim >= 0 && static_cast<size_t>(reduce_dim) < repeats.size());
    repeats[reduce_dim] = Symbol(1);
    reduce_op.SetTensorLoop(asc_axis_, repeats, ContiguousStrides(repeats));

    auto broadcast = MakeAscVar<ascir_op::Broadcast>(reduce_op);
    broadcast.SetTensorLoop(asc_axis_, asc_axis_repeats_, ContiguousStrides(asc_axis_repeats_));
    return MakeCse(cse_key, broadcast);
  }

  CseVar LoadSeed(const std::string &name, const CseVar &x) override {
    GE_WARN_ASSERT(false, "LoadSeed %s", CseKey<UnimplementOp>(name, x).c_str());
  }

  CseVar ToDtypeBitcast(const CseVar &x, ge::DataType dst_type, ge::DataType src_type) override {
    GE_WARN_ASSERT(false, "ToDtypeBitcast %s", CseKey<UnimplementOp>(x, dst_type, src_type).c_str());
  }

  CseVar Abs(const CseVar &x) override {
    return AscOp<ascir_op::Abs>(x);
  }

  CseVar Acos(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Acos %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Acosh(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Acosh %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Asin(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Asin %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Asinh(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Asinh %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Atan(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Atan %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Atanh(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Atanh %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar BesselJ0(const CseVar &x) override {
    GE_WARN_ASSERT(false, "BesselJ0 %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar BesselJ1(const CseVar &x) override {
    GE_WARN_ASSERT(false, "BesselJ1 %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar BesselY0(const CseVar &x) override {
    GE_WARN_ASSERT(false, "BesselY0 %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar BesselY1(const CseVar &x) override {
    GE_WARN_ASSERT(false, "BesselY1 %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar BitwiseNot(const CseVar &x) override {
    GE_WARN_ASSERT(false, "BitwiseNot %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Ceil(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Ceil %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Cos(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Cos %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Cosh(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Cosh %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Erf(const CseVar &x1) override {
    return AscOp<ascir_op::Erf>(x1);
  }

  CseVar Erfc(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Erfc %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Erfcx(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Erfcx %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Erfinv(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Erfinv %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Exp(const CseVar &x) override {
    return AscOp<ascir_op::Exp>(x);
  }

  CseVar Exp2(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Exp2 %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Expm1(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Expm1 %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Floor(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Floor %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Frexp(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Frexp %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar I0(const CseVar &x) override {
    GE_WARN_ASSERT(false, "I0 %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar I1(const CseVar &x) override {
    GE_WARN_ASSERT(false, "I1 %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Identity(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Identity %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar IsFinite(const CseVar &x) override {
    return AscOp<ascir_op::IsFinite>(x);
  }

  CseVar IsInf(const CseVar &x) override {
    GE_WARN_ASSERT(false, "IsInf %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar IsNan(const CseVar &x) override {
    return AscOp<ascir_op::Isnan>(x);
  }

  CseVar Lgamma(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Lgamma %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar LibdeviceAbs(const CseVar &x) override {
    GE_WARN_ASSERT(false, "LibdeviceAbs %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar LibdeviceCos(const CseVar &x) override {
    GE_WARN_ASSERT(false, "LibdeviceCos %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar LibdeviceExp(const CseVar &x) override {
    GE_WARN_ASSERT(false, "LibdeviceExp %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar LibdeviceLog(const CseVar &x) override {
    GE_WARN_ASSERT(false, "LibdeviceLog %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar LibdeviceSigmoid(const CseVar &x) override {
    GE_WARN_ASSERT(false, "LibdeviceSigmoid %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar LibdeviceSin(const CseVar &x) override {
    GE_WARN_ASSERT(false, "LibdeviceSin %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar LibdeviceSqrt(const CseVar &x) override {
    GE_WARN_ASSERT(false, "LibdeviceSqrt %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Ln(const CseVar &x) override {
    return AscOp<ascir_op::Ln>(x);
  }

  CseVar Log(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Log %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Log10(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Log10 %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Log1p(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Log1p %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Log2(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Log2 %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar LogicalNot(const CseVar &x) override {
    return AscOp<ascir_op::LogicalNot>(x);
  }

  CseVar ModifiedBesselI0(const CseVar &x) override {
    GE_WARN_ASSERT(false, "ModifiedBesselI0 %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar ModifiedBesselI1(const CseVar &x) override {
    GE_WARN_ASSERT(false, "ModifiedBesselI1 %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Neg(const CseVar &x) override {
    return AscOp<ascir_op::Neg>(x);
  }

  CseVar Reciprocal(const CseVar &x) override {
    return AscOp<ascir_op::Reciprocal>(x);
  }

  CseVar Relu(const CseVar &x) override {
    return AscOp<ascir_op::Relu>(x);
  }

  CseVar Round(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Round %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Rsqrt(const CseVar &x) override {
    return AscOp<ascir_op::Rsqrt>(x);
  }

  CseVar Sigmoid(const CseVar &x) override {
    return AscOp<ascir_op::Sigmoid>(x);
  }

  CseVar Sign(const CseVar &x) override {
    return AscOp<ascir_op::Sign>(x);
  }

  CseVar SignBit(const CseVar &x) override {
    GE_WARN_ASSERT(false, "SignBit %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Sin(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Sin %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Sinh(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Sinh %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar SpecialErf(const CseVar &x) override {
    GE_WARN_ASSERT(false, "SpecialErf %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Sqrt(const CseVar &x) override {
    return AscOp<ascir_op::Sqrt>(x);
  }

  CseVar Square(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Square %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Tan(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Tan %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Tanh(const CseVar &x) override {
    return AscOp<ascir_op::Tanh>(x);
  }

  CseVar Gelu(const CseVar &x) override {
    return AscOp<ascir_op::Gelu>(x);
  }

  CseVar Trunc(const CseVar &x) override {
    GE_WARN_ASSERT(false, "Trunc %s", CseKey<UnimplementOp>(x).c_str());
  }

  CseVar Add(const CseVar &x1, const CseVar &x2) override {
    return AscOp<ascir_op::Add>(x1, x2);
  }

  CseVar Atan2(const CseVar &x1, const CseVar &x2) override {
    GE_WARN_ASSERT(false, "Atan2 %s", CseKey<UnimplementOp>(x1, x2).c_str());
  }

  CseVar BitwiseAnd(const CseVar &x1, const CseVar &x2) override {
    return AscOp<ascir_op::BitwiseAnd>(x1, x2);
  }

  CseVar BitwiseLeftshift(const CseVar &x1, const CseVar &x2) override {
    GE_WARN_ASSERT(false, "BitwiseLeftshift %s", CseKey<UnimplementOp>(x1, x2).c_str());
  }

  CseVar BitwiseRightshift(const CseVar &x1, const CseVar &x2) override {
    GE_WARN_ASSERT(false, "BitwiseRightshift %s", CseKey<UnimplementOp>(x1, x2).c_str());
  }

  CseVar BitwiseOr(const CseVar &x1, const CseVar &x2) override {
    GE_WARN_ASSERT(false, "BitwiseOr %s", CseKey<UnimplementOp>(x1, x2).c_str());
  }

  CseVar BitwiseXor(const CseVar &x1, const CseVar &x2) override {
    GE_WARN_ASSERT(false, "BitwiseXor %s", CseKey<UnimplementOp>(x1, x2).c_str());
  }

  CseVar ClampMax(const CseVar &x1, const CseVar &x2) override {
    GE_WARN_ASSERT(false, "ClampMax %s", CseKey<UnimplementOp>(x1, x2).c_str());
  }

  CseVar ClampMin(const CseVar &x1, const CseVar &x2) override {
    GE_WARN_ASSERT(false, "ClampMin %s", CseKey<UnimplementOp>(x1, x2).c_str());
  }

  CseVar CopySign(const CseVar &x1, const CseVar &x2) override {
    GE_WARN_ASSERT(false, "CopySign %s", CseKey<UnimplementOp>(x1, x2).c_str());
  }

  CseVar Div(const CseVar &x1, const CseVar &x2) override {
    return AscOp<ascir_op::Div>(x1, x2);
  }

  CseVar Eq(const CseVar &x1, const CseVar &x2) override {
    return AscOp<ascir_op::Eq>(x1, x2);
  }

  CseVar FloorDiv(const CseVar &x1, const CseVar &x2) override {
    return AscOp<ascir_op::FloorDiv>(x1, x2);
  }

  CseVar Fmod(const CseVar &x1, const CseVar &x2) override {
    GE_WARN_ASSERT(false, "Fmod %s", CseKey<UnimplementOp>(x1, x2).c_str());
  }

  CseVar Ge(const CseVar &x1, const CseVar &x2) override {
    return AscOp<ascir_op::Ge>(x1, x2);
  }

  CseVar Gt(const CseVar &x1, const CseVar &x2) override {
    return AscOp<ascir_op::Gt>(x1, x2);
  }

  CseVar Hypot(const CseVar &x1, const CseVar &x2) override {
    GE_WARN_ASSERT(false, "Hypot %s", CseKey<UnimplementOp>(x1, x2).c_str());
  }

  CseVar InlineAsmElementwise(const CseVar &x1, const CseVar &x2) override {
    GE_WARN_ASSERT(false, "InlineAsmElementwise %s", CseKey<UnimplementOp>(x1, x2).c_str());
  }

  CseVar Le(const CseVar &x1, const CseVar &x2) override {
    return AscOp<ascir_op::Le>(x1, x2);
  }

  CseVar LeakyRelu(const CseVar &x, float32_t negative_slope) override {
    const auto cse_key = CseKey<ascir_op::LeakyRelu>(x, negative_slope);
    auto cached_var = LookUp(cse_key);
    if (cached_var.IsValid()) {
      return cached_var;
    }
    auto op = MakeAscVar<ascir_op::LeakyRelu>(x);
    GE_WARN_ASSERT(op.Src() != nullptr);
    GET_IR_ATTR(LeakyRelu, op.Src())->SetNegative_slope(negative_slope);
    return MakeCse(cse_key, op);
  }

  CseVar Axpy(const CseVar &x1, const CseVar &x2, float32_t alpha) override {
    const auto cse_key = CseKey<ascir_op::Axpy>(x1, x2, alpha);
    auto cached_var = LookUp(cse_key);
    if (cached_var.IsValid()) {
      return cached_var;
    }
    auto op = MakeAscVar<ascir_op::Axpy>(x1, x2);
    GE_WARN_ASSERT(op.Src() != nullptr);
    GET_IR_ATTR(Axpy, op.Src())->SetAlpha(alpha);
    return MakeCse(cse_key, op);
  }

  CseVar LogicalAnd(const CseVar &x1, const CseVar &x2) override {
    return AscOp<ascir_op::LogicalAnd>(x1, x2);
  }

  CseVar LogicalOr(const CseVar &x1, const CseVar &x2) override {
    return AscOp<ascir_op::LogicalOr>(x1, x2);
  }

  CseVar LogicalXor(const CseVar &x1, const CseVar &x2) override {
    GE_WARN_ASSERT(false, "LogicalXor %s", CseKey<UnimplementOp>(x1, x2).c_str());
  }

  CseVar Lt(const CseVar &x1, const CseVar &x2) override {
    return AscOp<ascir_op::Lt>(x1, x2);
  }

  CseVar Maximum(const CseVar &x1, const CseVar &x2) override {
    return AscOp<ascir_op::Maximum>(x1, x2);
  }

  CseVar Minimum(const CseVar &x1, const CseVar &x2) override {
    return AscOp<ascir_op::Minimum>(x1, x2);
  }

  CseVar Mul(const CseVar &x1, const CseVar &x2) override {
    return AscOp<ascir_op::Mul>(x1, x2);
  }

  CseVar Ne(const CseVar &x1, const CseVar &x2) override {
    return AscOp<ascir_op::Ne>(x1, x2);
  }

  CseVar NextAfter(const CseVar &x1, const CseVar &x2) override {
    GE_WARN_ASSERT(false, "NextAfter %s", CseKey<UnimplementOp>(x1, x2).c_str());
  }

  CseVar Pow(const CseVar &x1, const CseVar &x2) override {
    return AscOp<ascir_op::Pow>(x1, x2);
  }

  CseVar Rand(const CseVar &x1, const CseVar &x2) override {
    GE_WARN_ASSERT(false, "Rand %s", CseKey<UnimplementOp>(x1, x2).c_str());
  }

  CseVar RandN(const CseVar &x1, const CseVar &x2) override {
    GE_WARN_ASSERT(false, "RandN %s", CseKey<UnimplementOp>(x1, x2).c_str());
  }

  CseVar Remainder(const CseVar &x1, const CseVar &x2) override {
    GE_WARN_ASSERT(false, "Remainder %s", CseKey<UnimplementOp>(x1, x2).c_str());
  }

  CseVar Sub(const CseVar &x1, const CseVar &x2) override {
    return AscOp<ascir_op::Sub>(x1, x2);
  }

  CseVar TrueDiv(const CseVar &x1, const CseVar &x2) override {
    return AscOp<ascir_op::TrueDiv>(x1, x2);
  }

  CseVar TruncDiv(const CseVar &x1, const CseVar &x2) override {
    GE_WARN_ASSERT(false, "TruncDiv %s", CseKey<UnimplementOp>(x1, x2).c_str());
  }

  CseVar ClipByValue(const CseVar &x1, const CseVar &x2, const CseVar &x3) override {
    return AscOp<ascir_op::ClipByValue>(x1, x2, x3);
  }

  CseVar Fma(const CseVar &x1, const CseVar &x2, const CseVar &x3) override {
    GE_WARN_ASSERT(false, "Fma %s", CseKey<UnimplementOp>(x1, x2, x3).c_str());
  }

  CseVar Masked(const CseVar &x1, const CseVar &x2, const CseVar &x3) override {
    GE_WARN_ASSERT(false, "Masked %s", CseKey<UnimplementOp>(x1, x2, x3).c_str());
  }

  CseVar Where(const CseVar &x1, const CseVar &x2, const CseVar &x3) override {
    return AscOp<ascir_op::Where>(x1, x2, x3);
  }

  CseVar RandInt64(const CseVar &x1, const CseVar &x2, const CseVar &x3, const CseVar &x4) override {
    GE_WARN_ASSERT(false, "Masked %s", CseKey<UnimplementOp>(x1, x2, x3, x4).c_str());
  }

  CseVar Cast(const CseVar &x, ge::DataType dst_type) override {
    const auto cse_key = CseKey<ascir_op::Cast>(x, dst_type);
    auto cached_var = LookUp(cse_key);
    if (cached_var.IsValid()) {
      return cached_var;
    }
    auto op = MakeAscVar<ascir_op::Cast>(dst_type, x);
    return MakeCse(cse_key, op);
  }

  CseVar Ceil2Int(const CseVar &x, ge::DataType dst_type) override {
    GE_WARN_ASSERT(false, "Ceil2Int %s", CseKey<UnimplementOp>(x, dst_type).c_str());
  }

  CseVar Floor2Int(const CseVar &x, ge::DataType dst_type) override {
    GE_WARN_ASSERT(false, "Floor2Int %s", CseKey<UnimplementOp>(x, dst_type).c_str());
  }

  CseVar IndexExpr(const CseVar &x, ge::DataType dst_type) override {
    GE_WARN_ASSERT(false, "IndexExpr %s", CseKey<UnimplementOp>(x, dst_type).c_str());
  }

  CseVar Round2Int(const CseVar &x, ge::DataType dst_type) override {
    GE_WARN_ASSERT(false, "Round2Int %s", CseKey<UnimplementOp>(x, dst_type).c_str());
  }

  CseVar Trunc2Int(const CseVar &x, ge::DataType dst_type) override {
    GE_WARN_ASSERT(false, "Trunc2Int %s", CseKey<UnimplementOp>(x, dst_type).c_str());
  }

 private:
  AscVar AddInput(const std::string &buffer) {
    inputs_.emplace_back(GetBufferSrc(buffer));
    GE_WARN_ASSERT(inputs_.back() != nullptr, "Load from unknown buffer %s", buffer.c_str());
    auto dtype = GetBufferDtype(buffer);
    if (dtype == DT_BOOL) {
      dtype = DT_UINT8;
    }
    auto data = MakeAscVar<ascir_op::Data>(dtype);
    GE_WARN_ASSERT(data.Src() != nullptr);
    GET_IR_ATTR(Data, data.Src())->SetIndex(static_cast<int64_t>(inputs_.size() - 1U));
    return data;
  }

  AscOut AddOutput(const std::string &buffer, const AscVar &asc_var, const Index &repeats, const Index &strides) {
    auto ge_dtype = GetBufferDtype(buffer);
    ge_dtype = ge_dtype == DT_BOOL ? DT_UINT8 : ge_dtype;
    auto var = ge_dtype != asc_var.Dtype() ? MakeAscVar<ascir_op::Cast>(ge_dtype, asc_var) : asc_var;
    auto src = MakeAscVar<ascir_op::Store>(var);
    src.SetTensorLoop(asc_axis_, repeats, strides);
    GE_WARN_ASSERT(output_ == nullptr, "Store to graph %s output twice", graph_->GetName().c_str());
    output_ = GetBufferSrc(buffer);
    GE_WARN_ASSERT(output_ != nullptr, "Store to unknown buffer %s", buffer.c_str());
    GE_WARN_ASSERT(stored_buffers_.insert(buffer).second, "Store buffer %s twice", buffer.c_str());
    auto output = MakeRawNode<ascir_op::Output>({src});
    GE_WARN_ASSERT(output != nullptr);
    GET_IR_ATTR(Output, output)->SetIndex(0);
    return AscOut(output);
  }

  template <typename T, typename... Args>
  CseVar AscOp(Args... args) {
    auto cse_key = CseKey<T>(args...);
    auto cached_var = LookUp(cse_key);
    if (cached_var.IsValid()) {
      return cached_var;
    }
    auto op = MakeAscVar<T>(args...);
    return MakeCse(cse_key, op);
  }

  template <typename T>
  CseVar AscOp(const std::vector<CseVar> &vars) const {
    auto cse_key = CseKey<T>(vars);
    auto cached_var = LookUp(cse_key);
    if (cached_var.IsValid()) {
      return cached_var;
    }
    auto op = MakeAscVar<T>(vars);
    return MakeCse(cse_key, op);
  }

  template <typename T, size_t NUM_OUTPUTS, typename... Args>
  std::vector<CseVar> MultiOutAscOp(Args... args) const {
    auto cse_key = CseKey<T>(args...);
    auto cached_pack_var = LookUpMultiOut<NUM_OUTPUTS>(cse_key);
    if (!cached_pack_var.empty()) {
      return cached_pack_var;
    }
    std::vector<AscVar> vars = ToVector(args...);
    auto node = MakeRawNode<T>(vars, {}, {}, {NUM_OUTPUTS});

    std::vector<AscVar> asc_vars;
    asc_vars.reserve(NUM_OUTPUTS);
    for (size_t i = 0U; i < NUM_OUTPUTS; i++) {
      asc_vars.push_back(AscVar(node, i));
    }
    return MakeCse(cse_key, asc_vars);
  }

  [[nodiscard]] const std::vector<int64_t> &GetLoopAxisIds() const {
    return asc_axis_;
  }

  template <typename T>
  std::string CseKeyArgs(const T &head) const {
    std::stringstream ss;
    ss << head;
    return ss.str();
  }

  template <typename T>
  std::string CseKeyArgs(const std::vector<T> &head) const {
    std::stringstream ss;
    if (head.empty()) {
      return "[]";
    }
    ss << "[";
    for (size_t i = 0U; i + 1U < head.size(); i++) {
      ss << head << ", ";
    }
    ss << head.back() << "]";
    return ss.str();
  }

  template <typename T, typename... Args>
  std::string CseKeyArgs(const T &head, Args... rest) const {
    std::stringstream ss;
    ss << head << ",";
    return ss.str() + CseKeyArgs(rest...);
  }

  template <typename T, typename... Args>
  std::string CseKey(Args... rest) const {
    return std::string(T::Type) + "(" + CseKeyArgs(rest...) + ")";
  }

  CseVar LookUp(const std::string &cse_key) const {
    auto it = cse_cache_.find(cse_key);
    if (it != cse_cache_.end()) {
      GELOGI("Found cached %s for key %s", it->second.Name().c_str(), cse_key.c_str());
      return it->second;
    }
    return {};
  }

  template <size_t NUM_OUTPUTS>
  std::vector<CseVar> LookUpMultiOut(const std::string &cse_key) {
    auto it = cse_pack_cache_.find(cse_key);
    if (it != cse_pack_cache_.end()) {
      GELOGI("Found cached for key %s", cse_key.c_str());
      if (it->second.size() != NUM_OUTPUTS) {
        return std::vector<CseVar>(NUM_OUTPUTS);
      }
      return it->second;
    }
    return {};
  }

  static graphStatus SetAscNodeOutputDtypes(const AscNodePtr &node, const std::vector<ge::DataType> &output_dtypes) {
    // 推导出的dtype数量应该和输出anchor数量相等
    GE_WARN_ASSERT(node->GetAllOutDataAnchorsSize() == output_dtypes.size(),
                   "Output dtype size mismatch for %s, expect %zu, got %zu", node->GetName().c_str(),
                   node->GetAllOutDataAnchorsSize(), output_dtypes.size());
    try {
      for (size_t i = 0U; i < output_dtypes.size(); i++) {
        node->outputs[i].attr.dtype = output_dtypes[i];
      }
    } catch (const AscIRException &) {
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }

  template <typename T>
  AscNodePtr MakeRawNode(const std::vector<AscVar> &vars, const std::vector<ge::DataType> &explicit_output_dtypes = {},
                         const std::vector<size_t> &dynamic_input_size = {},
                         const std::vector<size_t> &dynamic_output_size = {}) {
    std::shared_ptr<T> op = std::make_shared<T>(NextName(T::Type).c_str());
    GE_WARN_ASSERT(op != nullptr);
    GE_WARN_ASSERT(dynamic_input_size.size() <= 1U);
    GE_WARN_ASSERT(dynamic_output_size.size() <= 1U);
    std::vector<ge::DataType> input_dtypes;
    if (dynamic_input_size.size() == 1U) {
      GE_WARN_ASSERT(vars.size() == dynamic_input_size[0]);
      op->DynamicInputRegister("x", dynamic_input_size[0]);
      input_dtypes.push_back(vars[0].Dtype());
    } else {
      for (const auto &var : vars) {
        input_dtypes.push_back(var.Dtype());
      }
    }
    if (dynamic_output_size.size() == 1U) {
      op->DynamicOutputRegister("y", dynamic_output_size[0]);
    }
    AscNodePtr node = graph_->AddNode(*op);
    GE_WARN_ASSERT(node != nullptr);
    node->attr.sched.exec_order = NextOrder();
    node->attr.sched.axis = GetLoopAxisIds();
    GELOGI("Asc graph <%s> add no.%ld node <%s, %s>", graph_->GetName().c_str(), node->attr.sched.exec_order,
           node->GetName().c_str(), node->GetType().c_str());
    GE_WARN_ASSERT(node->GetAllInDataAnchorsSize() == vars.size());
    for (int32_t i = 0U; i < static_cast<int32_t>(vars.size()); i++) {
      GE_WARN_ASSERT(vars[i].Buffer() != nullptr);
      GE_WARN_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(vars[i].Buffer(), node->GetInDataAnchor(i)));
    }
    if (node->GetAllOutDataAnchorsSize() == 0U) {
      return node;
    }
    // 除了Cast节点外，其他节点的dtype都是由输入推导出来的，特别地，对于Data和Scalar这种叶子节点，dtype是手动设置的
    if (std::is_same<T, ascir_op::Cast>::value || vars.empty()) {
      GE_WARN_ASSERT(!explicit_output_dtypes.empty(), "Explicit output dtype is required for %s", T::Type);
    }
    std::vector<ge::DataType> output_dtypes = explicit_output_dtypes;
    GE_ASSERT_TRUE(InferAscirDataType<T>(input_dtypes, output_dtypes),
                   "Failed to infer dtype for %s with inputs %s and outputs %s", T::Type,
                   loop::StrJoin(input_dtypes).c_str(), loop::StrJoin(output_dtypes).c_str());
    GELOGI("Infer asc ir %s with input dtypes %s got outputs %s", T::Type, loop::StrJoin(input_dtypes).c_str(),
           loop::StrJoin(output_dtypes).c_str());
    if (!dynamic_output_size.empty()) {
      GE_WARN_ASSERT(node->GetAllOutDataAnchorsSize() == dynamic_output_size[0]);
      GE_WARN_ASSERT(output_dtypes.size() == 1U);
      output_dtypes.resize(dynamic_output_size[0], output_dtypes[0]);
    }
    GE_WARN_ASSERT(SetAscNodeOutputDtypes(node, output_dtypes) == ge::SUCCESS);
    return node;
  }

  template <typename T, typename... Args>
  AscVar MakeAscVar(Args... args) {
    auto vars = ToVector(args...);
    auto node = MakeRawNode<T>(vars);
    GE_WARN_ASSERT(node != nullptr);
    GE_WARN_ASSERT(node->GetAllOutDataAnchorsSize() == 1U);
    return AscVar(node);
  }

  template <typename T, typename... Args>
  AscVar MakeAscVar(const std::vector<DataType> &explicit_output_dtypes, Args... args) {
    auto vars = ToVector(args...);
    auto node = MakeRawNode<T>(vars, explicit_output_dtypes);
    GE_WARN_ASSERT(node != nullptr);
    GE_WARN_ASSERT(node->GetAllOutDataAnchorsSize() == 1U);
    return AscVar(node);
  }

  template <typename T, typename... Args>
  AscVar MakeAscVar(const DataType &explicit_output_dtype, Args... args) {
    return MakeAscVar<T>(std::vector<ge::DataType>{explicit_output_dtype}, args...);
  }

  template <typename T>
  AscVar MakeAscVar(const std::vector<CseVar> &cse_vars) {
    std::vector<AscVar> vars;
    for (auto &var : cse_vars) {
      GE_WARN_ASSERT(var.Get<AscVar>() != nullptr);
      vars.push_back(*var.Get<AscVar>());
    }
    auto node = MakeRawNode<T>(vars, {}, {vars.size()});
    GE_WARN_ASSERT(node != nullptr);
    GE_WARN_ASSERT(node->GetAllOutDataAnchorsSize() == 1U);
    return AscVar(node);
  }

  static void AddToVector(const std::vector<AscVar> &vec) {
    static_cast<void>(vec);
  }

  template <typename... Args>
  void AddToVector(std::vector<AscVar> &vec, const CseVar &first, const Args &...rest) {
    if (first.IsValid()) {
      vec.push_back(*first.Get<AscVar>());
    }
    AddToVector(vec, rest...);
  }

  template <typename... Args>
  void AddToVector(std::vector<AscVar> &vec, const AscVar &first, const Args &...rest) {
    if (first.IsValid()) {
      vec.push_back(first);
    }
    AddToVector(vec, rest...);
  }

  template <typename... Args>
  std::vector<AscVar> ToVector(const Args &...args) {
    std::vector<AscVar> result;
    AddToVector(result, args...);
    return result;
  }

  CseVar MakeCse(const std::string &cse_key, const AscVar &asc_var) {
    GE_WARN_ASSERT(asc_var.IsValid());
    auto var = CseVar(asc_var.Shared());
    if (!cse_key.empty() && auto_cse_) {
      GELOGI("Cached %s for key %s", var.Name().c_str(), cse_key.c_str());
      cse_cache_[cse_key] = var;
    }
    return var;
  }

  std::vector<CseVar> MakeCse(const std::string &cse_key, const std::vector<AscVar> &vars) {
    std::vector<CseVar> pack;
    pack.reserve(vars.size());
    for (auto &var : vars) {
      pack.emplace_back(var.Shared());
    }
    if (!cse_key.empty() && auto_cse_) {
      cse_pack_cache_[cse_key] = pack;
    }
    return pack;
  }

  std::string NextName(const std::string &op_type) {
    return graph_->GetName() + "/" + op_type + "_" + std::to_string(typed_op_nums_[op_type]++);
  }

  AscVar MakeReduceVar(const CseVar &src, ReduceType reduce_type) {
    if (reduce_type == ReduceType::SUM) {
      return MakeAscVar<ascir_op::Sum>(src);
    }
    if (reduce_type == ReduceType::MAX) {
      return MakeAscVar<ascir_op::Max>(src);
    }
    if (reduce_type == ReduceType::MEAN) {
      return MakeAscVar<ascir_op::Mean>(src);
    }
    if (reduce_type == ReduceType::MIN) {
      return MakeAscVar<ascir_op::Min>(src);
    }
    if (reduce_type == ReduceType::PROD) {
      return MakeAscVar<ascir_op::Prod>(src);
    }
    if (reduce_type == ReduceType::ANY) {
      return MakeAscVar<ascir_op::Any>(src);
    }
    if (reduce_type == ReduceType::ALL) {
      return MakeAscVar<ascir_op::All>(src);
    }
    GE_WARN_ASSERT(false, "Unsupported reduce type %d", reduce_type);
  }

  int64_t NextOrder() {
    return exec_order_++;
  }
  int64_t exec_order_ = 0;
  std::shared_ptr<AscGraph> graph_;
  std::vector<int64_t> asc_axis_;
  std::vector<Expression> asc_axis_repeats_;
  std::set<std::string> stored_buffers_;
  bool auto_cse_;
  std::map<std::string, CseVar> cse_cache_;
  std::map<std::string, std::vector<CseVar>> cse_pack_cache_;
  std::map<std::string, size_t> typed_op_nums_;

  std::vector<const ge::OutDataAnchor *> inputs_;
  const ge::OutDataAnchor *output_ = nullptr;
};
}  // namespace loop
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_ASC_OVERRIDES_H_

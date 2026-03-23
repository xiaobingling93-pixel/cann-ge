/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_LOOP_OPS_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_LOOP_OPS_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include <stack>

#include "common/checker.h"
#include "graph/anchor.h"
#include "graph/graph.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/utils/type_utils.h"
#include "loop_common.h"
#include "loop_op_overrides.h"
#include "../op_helper/cube.h"


namespace ge {
namespace loop {
struct LoopCtx {
  struct GodLoadSpec {
    GodLoadSpec(TensorLoopDesc data, TensorLoopDesc load, Expression offset = Symbol(0))
        : data_loop(std::move(data)), load_loop(std::move(load)), load_offset(std::move(offset)) {}
    TensorLoopDesc data_loop;
    TensorLoopDesc load_loop;
    Expression load_offset;
  };
  explicit LoopCtx(const std::vector<Expression> &dims) : loop_axis(LoopAxis::FromDims(dims)) {}
  std::map<const LoopOp *, std::shared_ptr<Index>> load_2_index;
  std::map<const LoopOp *, GodLoadSpec> load_2_god_desc;
  std::map<const LoopOp *, CseVar> op_2_var;
  LoopAxis loop_axis;
  CseVar Set(const LoopOp *op, const CseVar &var) {
    return op_2_var[op] = var;
  }
  CseVar Get(const LoopOp *op) const {
    auto iter = op_2_var.find(op);
    return iter == op_2_var.end() ? CseVar() : iter->second;
  }
  [[nodiscard]] CseVar Get(const LoopOpPtr &op) const {
    return Get(op.get());
  }
  [[nodiscard]] std::vector<CseVar> Get(const std::vector<LoopOpPtr> &ops) const {
    std::vector<CseVar> vars;
    vars.reserve(ops.size());
    for (const auto &op : ops) {
      vars.push_back(Get(op));
    }
    return vars;
  }
};

class LoopOp {
  friend class KernelBox;

 public:
  LoopOp() = default;
  virtual ~LoopOp() = default;
  explicit LoopOp(LoopOpPtr input) : inputs_({std::move(input)}) {};
  explicit LoopOp(std::vector<LoopOpPtr> inputs) : inputs_(std::move(inputs)) {};

  static graphStatus Dfs(const LoopOp *start, const std::function<graphStatus(const LoopOp *)> &func);
  static graphStatus StrictTopoExecute(const LoopOp *start, const std::function<graphStatus(const LoopOp *)> &func);
  static graphStatus InferLoadIndex(const Index &index, const LoopOp *start, LoopCtx &loop_ctx);
  static graphStatus GetLoopOpsDataType(const std::vector<LoopOpPtr> &ops,
                                        std::map<const LoopOp *, DataType> &op_2_dtype,
                                        std::vector<DataType> &input_dtypes);
  [[nodiscard]] std::string Readable() const;
  [[nodiscard]] int64_t Id() const {
    return id_;
  }
  [[nodiscard]] virtual CseVar Compute(const LoopCtx &ctx) const = 0;
  [[nodiscard]] virtual graphStatus ReIndex(const Index &index, Index &reindex) const {
    static_cast<void>(index);
    static_cast<void>(reindex);
    GE_WARN_ASSERT(false, "LoopOp %s can not reindex", Type().c_str());
  }
  [[nodiscard]] virtual const std::string &Type() const = 0;
  [[nodiscard]] virtual LoopOpPtr CloneImpl() const = 0;
  virtual graphStatus RealizeImpl() {
    GE_WARN_ASSERT(false, "LoopOp %s can not realize", Type().c_str());
  }
  [[nodiscard]] virtual bool IsReduction() const {
    return false;
  }

  [[nodiscard]] virtual bool IsGather() const {
    return false;
  }

  [[nodiscard]] virtual bool InferDataType(const std::vector<DataType> &input_dtypes,
                                             std::vector<DataType> &expect_output_dtypes) const = 0;

  [[nodiscard]] virtual ge::NodePtr GetAscendIrNode() const {
    return nullptr;
  }

  void CountTypedOps(std::map<std::string, size_t> &typed_op_nums) const;
  void GetAscendIrNodes(std::vector<ge::NodePtr> &ordered_nodes) const;

  [[nodiscard]] const std::vector<LoopOpPtr> &Inputs() const {
    return inputs_;
  }

  [[nodiscard]] LoopOpPtr Clone() const {
    auto op = CloneImpl();
    for (size_t i = 0U; i < inputs_.size(); ++i) {
      op->inputs_[i] = op->inputs_[i]->Clone();
    }
    return op;
  }
 protected:
  graphStatus Realize() {
    return RealizeImpl();
  }
  [[nodiscard]] virtual std::string ReadableLine(const std::vector<std::string> &var_names) const;
  std::vector<LoopOpPtr> inputs_;
  int64_t id_ = global_id_++;

 private:
  static std::atomic<int64_t> global_id_;
};

class LoopVar {
 public:
  explicit LoopVar(LoopOpPtr op = nullptr) : op_(std::move(op)) {}
  [[nodiscard]] LoopOpPtr Op() const {
    return op_;
  }

  [[nodiscard]] std::string Readable() const {
    return op_ ? op_->Readable() : "Invalid";
  }

  [[nodiscard]] bool IsValid() const {
    return op_ != nullptr;
  }

  LoopVar Clone() const {
    return op_ ? LoopVar(op_->Clone()) : LoopVar();
  }
 private:
  LoopOpPtr op_;
};

class PointwiseOp : public LoopOp {
 public:
  explicit PointwiseOp(LoopKernel kernel, std::vector<LoopOpPtr> inputs, const char *type,
                       PrintKernel readable = nullptr, InferDtypeKernel inferdtype = nullptr)
      : LoopOp(std::move(inputs)),
        kernel_(std::move(kernel)),
        readable_(std::move(readable)),
        infer_dtype_((std::move(inferdtype))) {
    if (type != nullptr) {
      type_ = type;
    }
  }

  graphStatus ReIndex(const Index &index, Index &reindex) const override {
    reindex = index;
    return GRAPH_SUCCESS;
  }

  CseVar Compute(const LoopCtx &ctx) const override {
    return kernel_(ctx.Get(inputs_));
  }

  const std::string &Type() const override {
    return type_;
  }

  [[nodiscard]] LoopOpPtr CloneImpl() const override {
    return std::make_shared<PointwiseOp>(*this);
  };

  [[nodiscard]] bool InferDataType(const std::vector<DataType> &input_dtypes,
                                     std::vector<DataType> &expect_output_dtypes) const override;

 private:
  std::string ReadableLine(const std::vector<std::string> &var_names) const override {
    if (readable_) {
      return readable_(var_names);
    }
    return LoopOp::ReadableLine(var_names);
  }
  std::string type_ = "UnknownPointwise";
  LoopKernel kernel_;
  PrintKernel readable_;
  InferDtypeKernel infer_dtype_;
};

class LoadOp : public LoopOp {
 public:
  explicit LoadOp(const ge::OutDataAnchor *src, const ge::InDataAnchor *dst) : src_(src), dst_(dst) {}
  explicit LoadOp(const ge::OutDataAnchorPtr &src, const ge::InDataAnchorPtr &dst) : LoadOp(src.get(), dst.get()) {}

  [[nodiscard]] CseVar Compute(const LoopCtx &ctx) const override;

  [[nodiscard]] const std::string &Type() const override {
    const static std::string kType = "ops.Load";
    return kType;
  }

  [[nodiscard]] std::string ReadableLine(const std::vector<std::string> &var_names) const override {
    static_cast<void>(var_names);
    std::stringstream ss;
    ss << Type();
    ss << "(\"" << src_->GetOwnerNode()->GetName() << ":";
    ss << src_->GetIdx();
    ss << "\")";
    return ss.str();
  }

  graphStatus RealizeImpl() override {
    // 对于已经是Load的节点，Realize无需做任何事情
    GELOGW("Realize LoadOp %s has not effect", src_->GetOwnerNode()->GetName().c_str());
    return GRAPH_SUCCESS;
  }

  [[nodiscard]] const ge::OutDataAnchor *Buffer() const {
    return src_;
  }

  [[nodiscard]] std::pair<const ge::OutDataAnchor *, const ge::InDataAnchor *> Edge() const {
    return {src_, dst_};
  }

  [[nodiscard]] LoopOpPtr CloneImpl() const override {
    return std::make_shared<LoadOp>(*this);
  };

  [[nodiscard]] bool InferDataType(const std::vector<DataType> &input_dtypes,
                                         std::vector<DataType> &expect_output_dtypes) const override;
 private:
  const ge::OutDataAnchor *src_;
  const ge::InDataAnchor *dst_;
};

struct GatherInput {
  InDataAnchorPtr input_anchor;
  std::vector<Expression> input_dim;
};

class LoadGatherOp : public LoopOp {
 public:
  explicit LoadGatherOp(const ge::OutDataAnchor *dst, const std::vector<ge::OutDataAnchorPtr> &outputs,
                        std::vector<GatherInput> inputs, std::vector<Expression> dims, int64_t axis, bool negative_index_support)
      : dst_(dst), outputs_(outputs), ginputs_(inputs), dims_(dims), axis_(axis), negative_index_support_(negative_index_support) {}

  [[nodiscard]] CseVar Compute(const LoopCtx &ctx) const override;

  [[nodiscard]] bool IsGather() const override {
    return true;
  }

  [[nodiscard]] const std::string &Type() const override {
    const static std::string kType = "ops.LoadGather";
    return kType;
  }

  [[nodiscard]] LoopOpPtr CloneImpl() const override {
    return std::make_shared<LoadGatherOp>(*this);
  };

  [[nodiscard]] std::string ReadableLine(const std::vector<std::string> &var_names) const override {
    std::stringstream ss;
    ss << Type() << "(\"";
    ss << dst_->GetOwnerNode()->GetName() << ":" << dst_->GetIdx() << "\", ";
    ss << loop::StrJoin(var_names) << ")";
    return ss.str();
  }

  graphStatus RealizeImpl() override {
    // 对于已经是Load的节点，Realize无需做任何事情
    GELOGW("Realize LoadGatherOp %s has not effect", dst_->GetOwnerNode()->GetName().c_str());
    return GRAPH_SUCCESS;
  }

  [[nodiscard]] ge::NodePtr GetAscendIrNode() const override {
    return dst_ == nullptr ? nullptr : dst_->GetOwnerNode();
  }

  [[nodiscard]] bool InferDataType(const std::vector<DataType> &input_dtypes,
                                     std::vector<DataType> &expect_output_dtypes) const override;

  [[nodiscard]] vector<const ge::OutDataAnchor*> Buffers() const {
    vector<const OutDataAnchor*> buffers;
    for (const auto &output : outputs_) {
      buffers.push_back(output.get());
    }
    return buffers;
  }

  [[nodiscard]] vector<pair<const ge::OutDataAnchor *, const ge::InDataAnchor *>> Edges() const {
    vector<pair<const ge::OutDataAnchor *, const ge::InDataAnchor *>> edges;
    for (size_t i = 0U; i < outputs_.size(); ++i) {
      edges.emplace_back(outputs_[i].get(), ginputs_[i].input_anchor.get());
    }
    return edges;
  }

private:
  [[nodiscard]] std::string Buffer() const {
    if (!dst_ || !dst_->GetOwnerNode()) {
      return "Invalid dst_";
    }
    return dst_->GetOwnerNode()->GetName() + ":" + std::to_string(dst_->GetIdx());
  }

  const ge::OutDataAnchor *dst_;
  std::vector<ge::OutDataAnchorPtr> outputs_;
  std::vector<GatherInput> ginputs_;
  std::vector<Expression> dims_;
  int64_t axis_;
  bool negative_index_support_;
};

class StoreOp : public LoopOp {
 public:
  explicit StoreOp(const ge::OutDataAnchorPtr &dst, LoopOpPtr src, std::vector<Expression> dims)
      : LoopOp(std::move(src)), dst_(dst.get()), dims_(std::move(dims)) {}

  [[nodiscard]] graphStatus ReIndex(const Index &index, Index &reindex) const override {
    reindex = index;
    return GRAPH_SUCCESS;
  }

  [[nodiscard]] CseVar Compute(const LoopCtx &ctx) const override {
    return ctx.Get(inputs_[0]);
  }

  [[nodiscard]] const std::string &Type() const override {
    const static std::string kType = "ops.Store";
    return kType;
  }

  [[nodiscard]] LoopOpPtr CloneImpl() const override {
    return std::make_shared<StoreOp>(*this);
  };

  [[nodiscard]] ge::NodePtr GetAscendIrNode() const override {
    return dst_ == nullptr ? nullptr : dst_->GetOwnerNode();
  }

  [[nodiscard]] std::string ReadableLine(const std::vector<std::string> &var_names) const override {
    std::stringstream ss;
    ss << Type() << "(\"";
    ss << dst_->GetOwnerNode()->GetName() << ":" << dst_->GetIdx() << "\"";
    ss << ", " << var_names[0] << ")";
    return ss.str();
  }

  graphStatus RealizeImpl() override;

  [[nodiscard]] bool InferDataType(const std::vector<DataType> &input_dtypes,
                                     std::vector<DataType> &expect_output_dtypes) const override;

 private:
  ge::OutDataAnchor *dst_;
  std::vector<Expression> dims_;
};

class ReductionBaseOp : public LoopOp {
 public:
  explicit ReductionBaseOp(LoopOpPtr src) : LoopOp(std::move(src)) {}
  [[nodiscard]] bool IsReduction() const override {
    return true;
  }
};

class StoreReductionOp : public ReductionBaseOp {
 public:
  enum class DimKind : int32_t { NORM, REMOVE, REDUCE, END };
  explicit StoreReductionOp(ReduceType reduce_type, const ge::OutDataAnchorPtr &dst, LoopOpPtr src,
                            std::vector<Expression> src_dims, std::vector<DimKind> reduce_status)
      : ReductionBaseOp(std::move(src)),
        reduce_type_(reduce_type),
        src_dims_(std::move(src_dims)),
        reduce_status_(std::move(reduce_status)),
        dst_(dst.get()) {}

  [[nodiscard]] graphStatus ReIndex(const Index &index, Index &reindex) const override {
    reindex = index;
    return GRAPH_SUCCESS;
  }

  [[nodiscard]] CseVar Compute(const LoopCtx &ctx) const override {
    std::vector<Expression> dst_dims;
    for (size_t i = 0U; i < reduce_status_.size(); i++) {
      const auto kind = reduce_status_[i];
      if (kind == DimKind::NORM) {
        dst_dims.push_back(src_dims_[i]);
      } else {
        dst_dims.emplace_back(Symbol(1));
      }
    }
    std::string buffer = BufferName(dst_);
    Ops()->SetBufferSrc(buffer, dst_);
    const TensorLoopDesc loop_desc(dst_dims, ContiguousStrides(dst_dims));
    return Ops()->StoreReduction(buffer, ctx.Get(inputs_[0]), reduce_type_, loop_desc, Symbol(0));
  }

  [[nodiscard]] const std::string &Type() const override {
    const static std::string kType = "ops.StoreReduction";
    return kType;
  }

  [[nodiscard]] LoopOpPtr CloneImpl() const override {
    return std::make_shared<StoreReductionOp>(*this);
  };

  [[nodiscard]] ge::NodePtr GetAscendIrNode() const override {
    return dst_ == nullptr ? nullptr : dst_->GetOwnerNode();
  }

  [[nodiscard]] std::string ReadableLine(const std::vector<std::string> &var_names) const override {
    std::stringstream ss;
    ss << Type() << "(\"";
    ss << dst_->GetOwnerNode()->GetName() << ":" << dst_->GetIdx() << "\", ";
    ss << "ops." << ReduceTypeToString(reduce_type_) << "(" << var_names[0] << ", \"";
    std::vector<std::string> src_dims;
    std::vector<std::string> dst_dims;
    for (size_t i = 0U; i < reduce_status_.size(); i++) {
      auto kind = reduce_status_[i];
      src_dims.push_back("d" + std::to_string(i));
      if (kind == DimKind::NORM) {
        dst_dims.push_back(src_dims.back());
      } else if (kind == DimKind::REDUCE) {
        dst_dims.emplace_back("1");
      }
    }
    ss << StrJoin(src_dims) << "->" << StrJoin(dst_dims) << "\"))";
    return ss.str();
  }

  graphStatus RealizeImpl() override;

  [[nodiscard]] bool InferDataType(const std::vector<DataType> &input_dtypes,
                                     std::vector<DataType> &expect_output_dtypes) const override;

 private:
  [[nodiscard]] std::string Buffer() const {
    return dst_->GetOwnerNode()->GetName() + ":" + std::to_string(dst_->GetIdx());
  }

  ReduceType reduce_type_;
  std::vector<Expression> src_dims_;
  std::vector<DimKind> reduce_status_;
  const ge::OutDataAnchor *dst_;
};

class StoreConcatOp : public LoopOp {
 public:
  enum class DimKind : int32_t { NORM, REMOVE, REDUCE, END };
  explicit StoreConcatOp(const ge::OutDataAnchor *dst, std::vector<LoopOpPtr> inputs,
                         std::vector<std::vector<Expression>> input_dims, size_t concat_dim,
                         std::vector<Expression> dims)
      : LoopOp(std::move(inputs)),
        input_dims_(std::move(input_dims)),
        concat_dim_(concat_dim),
        dst_(dst),
        dims_(std::move(dims)) {}

  [[nodiscard]] CseVar Compute(const LoopCtx &ctx) const override {
    std::vector<CseVar> concat_inputs;
    concat_inputs.reserve(inputs_.size());
    for (const auto &input : inputs_) {
      concat_inputs.push_back(ctx.Get(input));
    }
    const std::string target = BufferName(dst_);
    Ops()->SetBufferSrc(target, dst_);
    return Ops()->StoreConcat(target, concat_inputs, TensorLoopDesc(dims_, ContiguousStrides(dims_)), Symbol(0));
  }

  [[nodiscard]] const std::string &Type() const override {
    const static std::string kType = "ops.StoreConcat";
    return kType;
  }

  [[nodiscard]] LoopOpPtr CloneImpl() const override {
    return std::make_shared<StoreConcatOp>(*this);
  };

  [[nodiscard]] ge::NodePtr GetAscendIrNode() const override {
    return dst_ == nullptr ? nullptr : dst_->GetOwnerNode();
  }

  [[nodiscard]] std::string ReadableLine(const std::vector<std::string> &var_names) const override {
    std::stringstream ss;
    ss << Type() << "(\"";
    ss << dst_->GetOwnerNode()->GetName() << ":" << dst_->GetIdx() << "\", ";
    ss << loop::StrJoin(var_names) << ", concat_dim=" << concat_dim_ << ")";
    return ss.str();
  }

  graphStatus RealizeImpl() override;

  [[nodiscard]] bool InferDataType(const std::vector<DataType> &input_dtypes,
                                     std::vector<DataType> &expect_output_dtypes) const override;

 private:
  [[nodiscard]] std::string Buffer() const {
    return dst_->GetOwnerNode()->GetName() + ":" + std::to_string(dst_->GetIdx());
  }

  std::vector<std::vector<Expression>> input_dims_;
  size_t concat_dim_;
  const ge::OutDataAnchor *dst_;
  std::vector<Expression> dims_;
};

class ReduceThenBroadcastOp : public ReductionBaseOp {
 public:
  explicit ReduceThenBroadcastOp(ReduceType type, LoopOpPtr input, int64_t dim)
      : ReductionBaseOp(std::move(input)), reduce_type_(type), reduce_dim_(dim) {}
  graphStatus ReIndex(const Index &index, Index &reindex) const override {
    reindex = index;
    return GRAPH_SUCCESS;
  }

  [[nodiscard]] CseVar Compute(const LoopCtx &ctx) const override {
    return Ops()->ReduceThenBroadcast(ctx.Get(inputs_[0]), reduce_type_, reduce_dim_);
  }

  [[nodiscard]] const std::string &Type() const override {
    const static std::string kType = "ops.ReduceThenBroadcast";
    return kType;
  }

  [[nodiscard]] LoopOpPtr CloneImpl() const override {
    return std::make_shared<ReduceThenBroadcastOp>(*this);
  };

  [[nodiscard]] std::string ReadableLine(const std::vector<std::string> &var_names) const override {
    std::stringstream ss;
    ss << "ops." << ReduceTypeToString(reduce_type_) << "Broadcast(" << var_names[0] << ", dim=" << reduce_dim_ << ")";
    return ss.str();
  }

  [[nodiscard]] bool InferDataType(const std::vector<DataType> &input_dtypes,
                                     std::vector<DataType> &expect_output_dtypes) const override;

 private:
  ReduceType reduce_type_;
  int64_t reduce_dim_;
};

class ScalarOp : public LoopOp {
 public:
  explicit ScalarOp(std::string face, ge::DataType dtype) : face_(std::move(face)), dtype_(dtype) {}
  CseVar Compute(const LoopCtx &ctx) const override {
    static_cast<void>(ctx);
    return Ops()->Scalar(face_, dtype_);
  }

  [[nodiscard]] const std::string &Type() const override {
    const static std::string kType = "ops.Scalar";
    return kType;
  }

  [[nodiscard]] LoopOpPtr CloneImpl() const override {
    return std::make_shared<ScalarOp>(*this);
  };

  [[nodiscard]] std::string ReadableLine(const std::vector<std::string> &var_names) const override {
    static_cast<void>(var_names);
    std::stringstream ss;
    ss << Type() << "(\"" << TypeUtils::DataTypeToSerialString(dtype_) << "(" << face_ << ")\")";
    return ss.str();
  }

  [[nodiscard]] bool InferDataType(const std::vector<DataType> &input_dtypes,
                                     std::vector<DataType> &expect_output_dtypes) const override;

 private:
  std::string face_;
  ge::DataType dtype_;
};

class BroadcastOp : public LoopOp {
 public:
  enum class DimKind : int32_t { NORMAL, BROADCAST, NEW_AXIS };
  explicit BroadcastOp(LoopOpPtr input, std::vector<DimKind> broadcast)
      : LoopOp(std::move(input)), broadcast_(std::move(broadcast)) {}
  graphStatus ReIndex(const Index &index, Index &reindex) const override {
    const static ge::Symbol kZero(0);
    GE_WARN_ASSERT(index.size() == broadcast_.size());
    reindex.clear();
    reindex.reserve(index.size());
    for (size_t i = 0U; i < index.size(); i++) {
      if (broadcast_[i] == DimKind::NEW_AXIS) {
        continue;
      }
      reindex.push_back((broadcast_[i] == DimKind::BROADCAST) ? kZero : index[i]);
    }
    GELOGD("BroadcastOp reindex %s -> %s", loop::StrJoin(index).c_str(), loop::StrJoin(reindex).c_str());
    return GRAPH_SUCCESS;
  }

  [[nodiscard]] CseVar Compute(const LoopCtx &ctx) const override {
    return ctx.Get(inputs_[0]);
  }

  [[nodiscard]] const std::string &Type() const override {
    const static std::string kType = "ops.Broadcast";
    return kType;
  }

  [[nodiscard]] LoopOpPtr CloneImpl() const override {
    return std::make_shared<BroadcastOp>(*this);
  };

  [[nodiscard]] std::string ReadableLine(const std::vector<std::string> &var_names) const override;

  [[nodiscard]] bool InferDataType(const std::vector<DataType> &input_dtypes,
                                     std::vector<DataType> &expect_output_dtypes) const override;

 private:
  std::vector<DimKind> broadcast_;
};

class PermuteOp : public LoopOp {
 public:
  explicit PermuteOp(LoopOpPtr input, std::vector<size_t> order) : LoopOp(std::move(input)), order_(std::move(order)) {}
  graphStatus ReIndex(const Index &index, Index &reindex) const override {
    GE_WARN_ASSERT(index.size() == order_.size());
    std::vector<size_t> neworder = order_;
    for (size_t i = 0; i < neworder.size(); i++) {
      for (size_t j = 0; j < order_.size(); j++) {
        if(order_[j] == i) {
          neworder[i] = j;
        }
      }
    }
    reindex.clear();
    reindex.reserve(index.size());
    for (size_t const i : neworder) {
      GE_WARN_ASSERT(i < index.size());
      reindex.push_back(index[i]);
    }
    GELOGD("PermuteOp reindex %s -> %s", loop::StrJoin(index).c_str(), loop::StrJoin(reindex).c_str());
    return GRAPH_SUCCESS;
  }

  [[nodiscard]] CseVar Compute(const LoopCtx &ctx) const override {
    return ctx.Get(inputs_[0]);
  }

  [[nodiscard]] const std::string &Type() const override {
    const static std::string kType = "ops.Permute";
    return kType;
  }

  [[nodiscard]] LoopOpPtr CloneImpl() const override {
    return std::make_shared<PermuteOp>(*this);
  };

  [[nodiscard]] std::string ReadableLine(const std::vector<std::string> &var_names) const override {
    std::stringstream ss;
    ss << Type() << "(" << var_names[0] << ", \"";
    std::vector<std::string> src_dims;
    std::vector<std::string> dst_dims;
    for (size_t i = 0U; i < order_.size(); i++) {
      src_dims.push_back("d" + std::to_string(i));
      dst_dims.push_back("d" + std::to_string(order_[i]));
    }
    ss << StrJoin(src_dims) << "->" << StrJoin(dst_dims) << "\")";
    return ss.str();
  }

  [[nodiscard]] bool InferDataType(const std::vector<DataType> &input_dtypes,
                                     std::vector<DataType> &expect_output_dtypes) const override {
    if (!input_dtypes.empty()) {
      expect_output_dtypes.emplace_back(input_dtypes[0]);
    }
    return true;
  }

 private:
  std::vector<size_t> order_;
};

class SqueezeOp : public LoopOp {
 public:
  explicit SqueezeOp(LoopOpPtr input, const int64_t removed_dim)
      : LoopOp(std::move(input)), removed_dim_(removed_dim) {}
  graphStatus ReIndex(const Index &index, Index &reindex) const override {
    reindex = index;
    const auto origin_rank = static_cast<int64_t>(index.size()) + 1U;
    GE_WARN_ASSERT(removed_dim_ < origin_rank && removed_dim_ >= -origin_rank,
                   "Squeeze dim %ld out of range [%ld, %ld)", removed_dim_, -origin_rank, origin_rank);
    const int64_t pos = removed_dim_ < 0 ? removed_dim_ + origin_rank : removed_dim_;
    reindex.insert(reindex.begin() + pos, Symbol(0));
    GELOGD("SqueezeOp reindex %s -> %s", loop::StrJoin(index).c_str(), loop::StrJoin(reindex).c_str());
    return GRAPH_SUCCESS;
  }

  [[nodiscard]] CseVar Compute(const LoopCtx &ctx) const override {
    return ctx.Get(inputs_[0]);
  }

  [[nodiscard]] const std::string &Type() const override {
    const static std::string kType = "ops.Squeeze";
    return kType;
  }

  [[nodiscard]] bool InferDataType(const std::vector<DataType> &input_dtypes,
                                     std::vector<DataType> &expect_output_dtypes) const override {
    if (!input_dtypes.empty()) {
      expect_output_dtypes.emplace_back(input_dtypes[0]);
    }
    return true;
  }

  [[nodiscard]] LoopOpPtr CloneImpl() const override {
    return std::make_shared<SqueezeOp>(*this);
  };

  [[nodiscard]] std::string ReadableLine(const std::vector<std::string> &var_names) const override {
    std::stringstream ss;
    ss << Type() << "(" << var_names[0] << ", " << removed_dim_ << ")";
    return ss.str();
  }

 private:
  int64_t removed_dim_;
};

class UnsqueezeOp : public LoopOp {
 public:
  explicit UnsqueezeOp(LoopOpPtr input, const int64_t added_dim) : LoopOp(std::move(input)), added_dim_(added_dim) {}
  graphStatus ReIndex(const Index &index, Index &reindex) const override {
    reindex = index;
    const auto un_squeezed_rank = static_cast<int64_t>(index.size());
    GE_WARN_ASSERT(added_dim_ < un_squeezed_rank && added_dim_ >= -un_squeezed_rank,
                   "UnSqueeze dim %ld out of range [%ld, %ld)", added_dim_, -un_squeezed_rank, un_squeezed_rank);
    const int64_t pos = added_dim_ < 0 ? added_dim_ + un_squeezed_rank : added_dim_;
    reindex.erase(reindex.begin() + pos);
    GELOGD("UnsqueezeOp reindex %s -> %s", loop::StrJoin(index).c_str(), loop::StrJoin(reindex).c_str());
    return GRAPH_SUCCESS;
  }

  [[nodiscard]] CseVar Compute(const LoopCtx &ctx) const override {
    return ctx.Get(inputs_[0]);
  }

  [[nodiscard]] const std::string &Type() const override {
    const static std::string kType = "ops.Unsqueeze";
    return kType;
  }

  [[nodiscard]] LoopOpPtr CloneImpl() const override {
    return std::make_shared<UnsqueezeOp>(*this);
  };

  [[nodiscard]] bool InferDataType(const std::vector<DataType> &input_dtypes,
                                     std::vector<DataType> &expect_output_dtypes) const override {
    if (!input_dtypes.empty()) {
      expect_output_dtypes.emplace_back(input_dtypes[0]);
    }
    return true;
  }

  [[nodiscard]] std::string ReadableLine(const std::vector<std::string> &var_names) const override {
    std::stringstream ss;
    ss << Type() << "(" << var_names[0] << ", " << added_dim_ << ")";
    return ss.str();
  }

 private:
  int64_t added_dim_;
};

struct StrideSliceParam {
  StrideSliceParam(std::vector<Expression> starts, std::vector<Expression> strides) :
  begin(std::move(starts)), strides(std::move(strides)) {};
  std::vector<Expression> begin;
  std::vector<Expression> strides;
};

class StoreStridedSliceOp : public LoopOp {
public:
  explicit StoreStridedSliceOp(const ge::OutDataAnchorPtr &dst, LoopOpPtr src, StrideSliceParam param,
                               std::vector<Expression> dims, std::vector<Expression> input_dims)
      : LoopOp(std::move(src)),
        param_(std::move(param)),
        input_dims_(std::move(input_dims)),
        dims_(std::move(dims)),
        dst_(dst.get()) {}

  [[nodiscard]] graphStatus ReIndex(const Index &index, Index &reindex) const override {
    reindex.clear();
    reindex.reserve(index.size());
    for (size_t i = 0U; i < index.size(); i++) {
      reindex.push_back(index[i] * param_.strides[i] + param_.begin[i]);
    }
    return GRAPH_SUCCESS;
  }

  [[nodiscard]] CseVar Compute(const LoopCtx &ctx) const override {
    return ctx.Get(inputs_[0]);
  }

  [[nodiscard]] const std::string &Type() const override {
    const static std::string kType = "ops.StoreStridedSlice";
    return kType;
  }

  [[nodiscard]] LoopOpPtr CloneImpl() const override {
    return std::make_shared<StoreStridedSliceOp>(*this);
  };

  [[nodiscard]] std::string ReadableLine(const std::vector<std::string> &var_names) const override {
    std::stringstream ss;
    ss << Type() << "(" << var_names[0] << ")";
    return ss.str();
  }

  [[nodiscard]] ge::NodePtr GetAscendIrNode() const override {
    return dst_ == nullptr ? nullptr : dst_->GetOwnerNode();
  }

  [[nodiscard]] bool InferDataType(const std::vector<DataType> &input_dtypes,
                                     std::vector<DataType> &expect_output_dtypes) const override {
    if (!input_dtypes.empty()) {
      expect_output_dtypes.emplace_back(input_dtypes[0]);
    }
    return true;
  }

private:
  [[nodiscard]] std::string Buffer() const {
    return dst_->GetOwnerNode()->GetName() + ":" + std::to_string(dst_->GetIdx());
  }

  StrideSliceParam param_;
  std::vector<Expression> input_dims_;
  std::vector<Expression> dims_;
  const ge::OutDataAnchor *dst_;
};

class StoreSplitOp : public LoopOp {
public:
  constexpr static size_t kSplitCanFuseMaxOutput = 512U;
  constexpr static size_t kSplitCanLowerEndDimMaxOutput = 48U;
  struct StoreSplitDesc {
    // 成员变量
    OutDataAnchorPtr output_;
    LoopOpPtr src_op_;
    std::vector<std::vector<Expression>> output_dims_;
    std::vector<Expression> input_dims_;
    size_t split_dim_;
    uint32_t index_;   // lowering后的StoreSplitOp对应的OutDataAnchor ID
    inline static uint32_t id_ = 0;  // lowering前的SplitOp的唯一索引号

    // 私有构造函数，禁止直接构造
    StoreSplitDesc() = default;
    friend class StoreSplitDescBuilder;
  };
    // Builder 类
    class StoreSplitDescBuilder {
    private:
      StoreSplitDesc desc_;

    public:
      StoreSplitDescBuilder& Output(const ge::OutDataAnchorPtr& output) {
        desc_.output_ = output;
        return *this;
      }

      StoreSplitDescBuilder& SrcOp(const LoopOpPtr& src_op) {
        desc_.src_op_ = src_op;
        return *this;
      }

      StoreSplitDescBuilder& OutputDims(const std::vector<std::vector<Expression>>& output_dims) {
        desc_.output_dims_ = output_dims;
        return *this;
      }

      StoreSplitDescBuilder& InputDims(const std::vector<Expression>& input_dims) {
        desc_.input_dims_ = input_dims;
        return *this;
      }

      StoreSplitDescBuilder& SplitDim(size_t split_dim) {
        desc_.split_dim_ = split_dim;
        return *this;
      }

      StoreSplitDescBuilder& Index(uint32_t index) {
        desc_.index_ = index;
        return *this;
      }

      // 构建并返回 StoreSplitDesc 实例
      StoreSplitDesc Build() {
        // 构造时自增 id，仅在 idx_ == 0 时
        if (desc_.index_ == 0) {
          desc_.id_++;
        }
        std::stringstream ss;
        ss << "StoreSplitDesc(original split global id: " << desc_.id_
           << ", index: " << desc_.index_
           << ", split_dim: " << desc_.split_dim_
           << ")";
        GELOGD("StoreSplitDesc: %s",ss.str().c_str());
        return desc_;
      }
    };
  explicit StoreSplitOp(const StoreSplitDesc &desc)
      : LoopOp(std::move(desc.src_op_)),
        input_dims_(std::move(desc.input_dims_)),
        dims_(std::move(desc.output_dims_)),
        output_(desc.output_),
        split_dim_(desc.split_dim_),
        idx_(desc.index_),
        id_(desc.id_){}

  [[nodiscard]] CseVar Compute(const LoopCtx &ctx) const override {
    CseVar split_input = ctx.Get(inputs_[0]);
    std::string target = BufferName(output_);
    Ops()->SetBufferSrc(target, output_.get());
    TensorLoopDesc output_desc(dims_[idx_],ContiguousStrides(dims_[idx_]));
    Ops()->StoreSplit(target,split_input, output_desc,Symbol(0),idx_,id_);
    return ctx.Get(inputs_[0]);
  }

  [[nodiscard]] const std::string &Type() const override {
    const static std::string kType = "ops.StoreSplit";
    return kType;
  }

  [[nodiscard]] graphStatus ReIndex(const Index &index, Index &reindex) const override {
    reindex=index;
    return GRAPH_SUCCESS;
  }

  [[nodiscard]] LoopOpPtr CloneImpl() const override {
    return std::make_shared<StoreSplitOp>(*this);
  };
  graphStatus RealizeImpl() override;
  void SetIndex(uint32_t index) {
    idx_ = index;
  }
  [[nodiscard]] std::string ReadableLine(const std::vector<std::string> &var_names) const override {
    std::stringstream ss;
    std::vector<std::string> wrapped_outputs;
    ss << Type() << "(" ;
    std::ostringstream oss;
    oss << "\"" << output_->GetOwnerNode()->GetName()
        << ":" << output_->GetIdx() << "\"";
    wrapped_outputs.push_back(oss.str());
    ss << loop::StrJoin(wrapped_outputs) <<", " ;
    ss << loop::StrJoin(var_names) << ", split_dim=" << split_dim_ << ")";
    return ss.str();
  }

  [[nodiscard]] ge::NodePtr GetAscendIrNode() const override {
    return output_ == nullptr ? nullptr : output_->GetOwnerNode();
  }

  [[nodiscard]] bool InferDataType(const std::vector<DataType> &input_dtypes,
                                     std::vector<DataType> &expect_output_dtypes) const override {
    if (!input_dtypes.empty()) {
      constexpr uint32_t ref_index=0;
      expect_output_dtypes.emplace_back(input_dtypes[ref_index]);
    }
    return true;
  }

private:
  [[nodiscard]] std::string Buffer() const {
    auto dst_=output_;
    return dst_->GetOwnerNode()->GetName() + ":" + std::to_string(dst_->GetIdx());
  }

  std::vector<Expression> input_dims_;
  std::vector<std::vector<Expression>> dims_;
  OutDataAnchorPtr output_;
  size_t split_dim_;
  size_t idx_;
  size_t id_;
  mutable std::vector<CseVar> vars_;
};

class StoreMatMulOp : public LoopOp {
 public:
  explicit StoreMatMulOp(const ge::OutDataAnchor *dst, std::vector<LoopOpPtr> inputs, MatMulAttr matmul_attr,
                         std::vector<Expression> dims, std::vector<std::vector<Expression>> input_dims)
      : LoopOp(std::move(inputs)),
        attrs_(std::move(matmul_attr)),
        dims_(std::move(dims)),
        dst_(dst),
        input_dims_(input_dims) {}

  [[nodiscard]] graphStatus ReIndex(const Index &index, Index &reindex) const override {
    reindex = index;
    return GRAPH_SUCCESS;
  }

  [[nodiscard]] CseVar Compute(const LoopCtx &ctx) const override {
    std::vector<CseVar> mm_inputs;
    mm_inputs.reserve(inputs_.size());
    for (const auto &input : inputs_) {
      mm_inputs.push_back(ctx.Get(input));
    }
    const std::string target = BufferName(dst_);
    Ops()->SetBufferSrc(target, dst_);
    return Ops()->StoreMatMul(target, mm_inputs, TensorLoopDesc(dims_, ContiguousStrides(dims_)), attrs_,
                              {attrs_.output_dtype});
  }

  [[nodiscard]] const std::string &Type() const override {
    const static std::string kType = "ops.StoreMatMul";
    return kType;
  }

  [[nodiscard]] LoopOpPtr CloneImpl() const override {
    return std::make_shared<StoreMatMulOp>(*this);
  };

  [[nodiscard]] ge::NodePtr GetAscendIrNode() const override {
    return dst_ == nullptr ? nullptr : dst_->GetOwnerNode();
  }

  [[nodiscard]] std::string ReadableLine(const std::vector<std::string> &var_names) const override {
    std::stringstream ss;
    ss << Type() << "(\"";
    ss << dst_->GetOwnerNode()->GetName() << ":" << dst_->GetIdx() << "\", ";
    ss << loop::StrJoin(var_names) << ", transpose_x1=" << attrs_.transpose_x1
       << ", transpose_x2=" << attrs_.transpose_x2 << ", offset_x=" << attrs_.offset_x
       << ", enable_hf32=" << attrs_.enable_hf32 << ", has_bias=" << attrs_.has_bias
       << ", has_offset_w=" << attrs_.has_offset_w << ")";
    return ss.str();
  }

  graphStatus RealizeImpl() override;

  [[nodiscard]] bool InferDataType(const std::vector<DataType> &input_dtypes,
                                   std::vector<DataType> &expect_output_dtypes) const override {
    (void)input_dtypes;
    expect_output_dtypes.emplace_back(attrs_.output_dtype);
    return true;
  }

 private:
  [[nodiscard]] std::string Buffer() const {
    return dst_->GetOwnerNode()->GetName() + ":" + std::to_string(dst_->GetIdx());
  }

  MatMulAttr attrs_;
  std::vector<Expression> dims_;
  const ge::OutDataAnchor *dst_;
  std::vector<std::vector<Expression>> input_dims_;
};

class ReshapeOp : public LoopOp {
 public:
  explicit ReshapeOp(LoopOpPtr input, const std::vector<Expression> &src_dims, const std::vector<Expression> &dst_dims,
                     size_t short_idx, const std::vector<size_t> &mul_idx)
      : LoopOp(std::move(input)), src_dims_(src_dims), dst_dims_(dst_dims), short_idx_(short_idx), mul_idx_(mul_idx) {}
  graphStatus ReIndex(const Index &index, Index &reindex) const override {
    GE_WARN_ASSERT(index.size() == dst_dims_.size(), "reshape index size %zu not equal dst_dims %zu", index.size(),
                   dst_dims_.size());
    reindex = index;
    if (src_dims_.size() > dst_dims_.size()) {
      // [A * B] -> [A, B]
      for (size_t i = 0U; mul_idx_.size() > 0U && i < mul_idx_.size() - 1; i++) {
        reindex.insert(reindex.begin() + short_idx_, Symbol(0));
      }
    } else {
      // [A, B] -> [A * B]
      size_t offset = 0U;
      Expression erase_index = Symbol(0);
      for (size_t i = 0U; i < mul_idx_.size(); i++) {
        Expression index_i = index[mul_idx_[i]];
        for (size_t j = i + 1U; j < mul_idx_.size(); j++) {
          index_i = index_i * dst_dims_[mul_idx_[j]];
        }
        erase_index = erase_index + index_i;
        reindex.erase(reindex.begin() + mul_idx_[i] - offset);
        offset++;
      }
      reindex.insert(reindex.begin() + short_idx_, erase_index);
    }

    return GRAPH_SUCCESS;
  }

  [[nodiscard]] CseVar Compute(const LoopCtx &ctx) const override {
    return ctx.Get(inputs_[0]);
  }

  [[nodiscard]] const std::string &Type() const override {
    const static std::string kType = "ops.Reshape";
    return kType;
  }

  [[nodiscard]] LoopOpPtr CloneImpl() const override {
    return std::make_shared<ReshapeOp>(*this);
  };

  [[nodiscard]] bool InferDataType(const std::vector<DataType> &input_dtypes,
                                   std::vector<DataType> &expect_output_dtypes) const override {
    if (!input_dtypes.empty()) {
      expect_output_dtypes.emplace_back(input_dtypes[0]);
    }
    return true;
  }

  [[nodiscard]] std::string ReadableLine(const std::vector<std::string> &var_names) const override {
    std::stringstream ss;
    ss << Type() << "(" << var_names[0] << ", " << loop::StrJoin(src_dims_).c_str() << " -> " << loop::StrJoin(dst_dims_).c_str() << ")";
    return ss.str();
  }

private:
  std::vector<Expression> src_dims_;
  std::vector<Expression> dst_dims_;
  size_t short_idx_;
  std::vector<size_t> mul_idx_;
};
}  // namespace loop
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_LOOP_OPS_H_

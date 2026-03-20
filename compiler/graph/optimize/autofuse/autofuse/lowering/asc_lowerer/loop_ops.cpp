/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "common/checker.h"
#include "graph/node.h"
#include "graph/symbolizer/symbolic.h"
#include "loop_common.h"
#include "loop_ops.h"
#include "asc_overrides.h"
#include "utils/autofuse_utils.h"

namespace ge {
namespace {
bool ReduceInferDataType(loop::ReduceType reduce_type, const std::vector<DataType> &input_dtypes,
                         std::vector<DataType> &expect_output_dtypes) {
  if (reduce_type == loop::ReduceType::SUM) {
    return ge::loop::InferAscirDataType<ascir_op::Sum>(input_dtypes, expect_output_dtypes);
  }
  if (reduce_type == loop::ReduceType::MAX) {
    return ge::loop::InferAscirDataType<ascir_op::Max>(input_dtypes, expect_output_dtypes);
  }
  if (reduce_type == loop::ReduceType::MEAN) {
    return ge::loop::InferAscirDataType<ascir_op::Mean>(input_dtypes, expect_output_dtypes);
  }
  if (reduce_type == loop::ReduceType::MIN) {
    return ge::loop::InferAscirDataType<ascir_op::Min>(input_dtypes, expect_output_dtypes);
  }
  if (reduce_type == loop::ReduceType::PROD) {
    return ge::loop::InferAscirDataType<ascir_op::Prod>(input_dtypes, expect_output_dtypes);
  }
  GELOGW("Unsupported reduce type %d", reduce_type);
  return false;
}
}
namespace loop {
std::atomic<int64_t> LoopOp::global_id_(0);
graphStatus LoopOp::Dfs(const LoopOp *start, const std::function<graphStatus(const LoopOp *)> &func) {
  std::stack<const LoopOp *> stack;
  stack.push(start);
  while (!stack.empty()) {
    auto *current = stack.top();
    stack.pop();
    GE_WARN_ASSERT_GRAPH_SUCCESS(func(current));
    for (auto it = current->inputs_.rbegin(); it != current->inputs_.rend(); ++it) {
      stack.push(it->get());
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus LoopOp::StrictTopoExecute(const LoopOp *start, const std::function<graphStatus(const LoopOp *)> &func) {
  std::map<int64_t, vector<const LoopOp *>> loop_ops;
  Dfs(start, [&loop_ops](const LoopOp *op) -> graphStatus {
    loop_ops[op->Id()].emplace_back(op);
    return GRAPH_SUCCESS;
  });
  for (const auto &item : loop_ops) {
    for (const auto &op : item.second) {
      GE_WARN_ASSERT_GRAPH_SUCCESS(func(op));
    }
  }
  return GRAPH_SUCCESS;
}

[[nodiscard]] std::string LoopOp::Readable() const {
  std::stringstream ss;
  std::map<const LoopOp *, std::string> op_var_names;
  size_t tmp_num = 0U;
  StrictTopoExecute(this, [&ss, &op_var_names, &tmp_num](const LoopOp *op) -> graphStatus {
    const std::string var_name = "tmp" + std::to_string(tmp_num++);
    std::vector<std::string> var_names;
    var_names.reserve(op->inputs_.size());
    for (const auto &input : op->inputs_) {
      var_names.push_back(op_var_names[input.get()]);
    }
    ss << var_name << " = " << op->ReadableLine(var_names) << std::endl;
    op_var_names[op] = var_name;
    return GRAPH_SUCCESS;
  });

  return ss.str();
}

void LoopOp::CountTypedOps(std::map<std::string, size_t> &typed_op_nums) const {
  Dfs(this, [&typed_op_nums](const LoopOp *op) {
    typed_op_nums[op->Type()]++;
    return GRAPH_SUCCESS;
  });
}

void LoopOp::GetAscendIrNodes(std::vector<ge::NodePtr> &ordered_nodes) const {
  std::set<ge::NodePtr> seen_nodes;
  StrictTopoExecute(this, [&ordered_nodes, &seen_nodes](const LoopOp *op) {
    auto node = op->GetAscendIrNode();
    if (node != nullptr && seen_nodes.insert(node).second) {
      ordered_nodes.push_back(node);
    }
    return GRAPH_SUCCESS;
  });
}

graphStatus LoopOp::InferLoadIndex(const Index &index, const LoopOp *start, LoopCtx &loop_ctx) {
  std::map<const LoopOp *, std::shared_ptr<Index>> id2index = {{start, std::make_shared<Index>(index)}};
  return Dfs(start, [&id2index, &loop_ctx](const LoopOp *op) -> graphStatus {
    if (op->Inputs().empty()) {
      if (op->Type() == "ops.Load") {
        loop_ctx.load_2_index[op] = id2index[op];
      }
      return GRAPH_SUCCESS;
    }
    auto reindex = std::make_shared<Index>();
    for (const auto &in : op->Inputs()) {
      id2index[in.get()] = reindex;
    }
    return op->ReIndex(*id2index[op], *reindex);
  });
}

graphStatus LoopOp::GetLoopOpsDataType(const std::vector<LoopOpPtr> &ops,
                                       std::map<const LoopOp *, DataType> &op_2_dtype,
                                       std::vector<DataType> &input_dtypes) {
  for (const auto &input : ops) {
    if (op_2_dtype.find(input.get()) == op_2_dtype.end()) {
      return GRAPH_FAILED;
    }
    input_dtypes.emplace_back(op_2_dtype[input.get()]);
  }
  return GRAPH_SUCCESS;
}

[[nodiscard]] std::string LoopOp::ReadableLine(const std::vector<std::string> &var_names) const {
  std::stringstream ss;
  ss << Type() << "(";
  for (size_t i = 0U; i < inputs_.size(); ++i) {
    ss << var_names[i];
    if (i + 1 < inputs_.size()) {
      ss << ", ";
    }
  }
  ss << ")";
  return ss.str();
}

[[nodiscard]] std::string BroadcastOp::ReadableLine(const std::vector<std::string> &var_names) const {
  std::stringstream ss;
  ss << Type() << "(" << var_names[0] << ", \"";
  std::vector<std::string> src_dims;
  std::vector<std::string> dst_dims;
  for (size_t i = 0U; i < broadcast_.size(); i++) {
    auto kind = broadcast_[i];
    if (kind == DimKind::NEW_AXIS) {
      dst_dims.push_back("d" + std::to_string(i));
    } else if (kind == DimKind::BROADCAST) {
      src_dims.emplace_back("1");
      dst_dims.push_back("d" + std::to_string(i));
    } else {
      src_dims.push_back("d" + std::to_string(i));
      dst_dims.push_back("d" + std::to_string(i));
    }
  }
  ss << StrJoin(src_dims) << "->" << StrJoin(dst_dims) << "\")";
  return ss.str();
}

bool PointwiseOp::InferDataType(const std::vector<DataType> &input_dtypes,
                                std::vector<DataType> &expect_output_dtypes) const {
  return (infer_dtype_ && infer_dtype_(input_dtypes, expect_output_dtypes));
}

CseVar LoadOp::Compute(const LoopCtx &ctx) const {
  const auto god_spec_iter = ctx.load_2_god_desc.find(this);
  if (god_spec_iter != ctx.load_2_god_desc.end()) {
    std::string buffer = BufferName(src_);
    Ops()->SetBufferSrc(buffer, src_);
    auto &god_spec = god_spec_iter->second;
    GELOGD("god_spec.load_offset: %s", god_spec.load_offset.Serialize().get());
    return Ops()->Load(buffer, god_spec.data_loop, god_spec.load_loop, god_spec.load_offset);
  }

  const auto iter = ctx.load_2_index.find(this);
  GE_WARN_ASSERT(iter != ctx.load_2_index.end(), "LoadOp %s has no index", src_->GetOwnerNode()->GetName().c_str());
  const auto &index = *iter->second;
  auto &loop_axis = ctx.loop_axis;
  const auto desc = src_->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(src_->GetIdx());
  const auto sym_attr = desc->GetAttrsGroup<SymbolicDescAttr>();
  GE_WARN_ASSERT(sym_attr != nullptr);
  const auto &dims = sym_attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  GE_WARN_ASSERT(dims.size() == index.size());

  std::vector<Expression> strides = ContiguousStrides(dims);
  Expression load_index = Symbol(0);
  for (size_t i = 0U; i < index.size(); ++i) {
    load_index = load_index + index[i] * strides[i];
  }
  Expression offset;
  std::vector<Expression> axis_strides;
  GetStrideAndOffset(load_index, loop_axis.axis, axis_strides, offset);

  std::vector<Expression> axis_repeats = loop_axis.repeats;
  for (size_t i = 0U; i < axis_strides.size(); ++i) {
    if (SymbolicUtils::StaticCheckEq(axis_strides[i], Symbol(0)) == TriBool::kTrue) {
      axis_repeats[i] = Symbol(1);
    }
  }

  std::string buffer = BufferName(src_);
  Ops()->SetBufferSrc(buffer, src_);
  return Ops()->Load(buffer, TensorLoopDesc(axis_repeats, axis_strides), offset);
}

bool LoadOp::InferDataType(const std::vector<DataType> &input_dtypes,
                           std::vector<DataType> &expect_output_dtypes) const {
  (void)input_dtypes;
  DataType data_type;
  if (GetBufferDataType(dst_, data_type) != GRAPH_SUCCESS) {
    return false;
  }
  std::vector<DataType> load_input_dtypes = {data_type};
  if (AutofuseUtils::CallAscirInferDataType<ascir_op::Load>(load_input_dtypes, expect_output_dtypes) != SUCCESS) {
    GELOGI("Infer Op ops.Load with input dtypes %s got outputs %s", loop::StrJoin(load_input_dtypes).c_str(),
           loop::StrJoin(expect_output_dtypes).c_str());
    return false;
  }
  return true;
}

CseVar LoadGatherOp::Compute(const LoopCtx &ctx) const {
  (void)ctx;
  std::vector<Expression> params_repeats = ginputs_[0].input_dim;
  std::vector<Expression> params_strides = ContiguousStrides(params_repeats);

  std::vector<Expression> indices_repeats = ginputs_[1].input_dim;
  std::vector<Expression> indices_strides = ContiguousStrides(indices_repeats);

  std::string buffer_params = BufferName(outputs_[0]);
  Ops()->SetBufferSrc(buffer_params, outputs_[0].get());
  std::string buffer_indices = BufferName(outputs_[1]);
  Ops()->SetBufferSrc(buffer_indices, outputs_[1].get());

  const TensorLoopDesc loop_desc_params(params_repeats, params_strides);
  const TensorLoopDesc loop_desc_indices(indices_repeats, indices_strides);

  return Ops()->GatherLoad(buffer_params, buffer_indices, loop_desc_params, loop_desc_indices, axis_, negative_index_support_);
}

bool LoadGatherOp::InferDataType(const std::vector<DataType> &input_dtypes,
                                 std::vector<DataType> &expect_output_dtypes) const {
  (void)input_dtypes;
  vector<DataType> data_types;
  for (const auto &input : ginputs_) {
    DataType data_type;
    GetBufferDataType(input.input_anchor.get(), data_type);
    data_types.emplace_back(data_type);
  }
  if (AutofuseUtils::CallAscirInferDataType<ascir_op::Gather>(data_types, expect_output_dtypes) != SUCCESS) {
    GELOGI("Infer Op ops.LoadGather with input dtypes %s got outputs %s", loop::StrJoin(data_types).c_str(),
           loop::StrJoin(expect_output_dtypes).c_str());
    return false;
  }
  return true;
}

graphStatus StoreOp::RealizeImpl() {
  for (size_t i = 0U; i < dims_.size(); ++i) {
    dims_[i] = dims_[i].Simplify();
  }
  LoopCtx loop_ctx(dims_);
  Ops()->SetLoopAxis(loop_ctx.loop_axis);
  const auto &loop_axis = loop_ctx.loop_axis;

  GE_WARN_ASSERT_GRAPH_SUCCESS(InferLoadIndex(loop_axis.axis, this, loop_ctx));

  auto status = StrictTopoExecute(this, [&loop_ctx](const LoopOp *op) -> graphStatus {
    const auto result = op->Compute(loop_ctx);
    GE_WARN_ASSERT(result.IsValid(), "Loop kernel for %s got %s", op->Type().c_str(), result.Name().c_str());
    loop_ctx.Set(op, result);
    return GRAPH_SUCCESS;
  });
  GE_WARN_ASSERT_GRAPH_SUCCESS(status, "Failed build asc graph for %s", BufferName(dst_).c_str());

  std::string buffer = BufferName(dst_);
  Ops()->SetBufferSrc(buffer, dst_);
  const TensorLoopDesc loop_desc(loop_axis.repeats, ContiguousStrides(loop_axis.repeats));
  auto var = Ops()->Store(buffer, loop_ctx.Get(this), loop_desc, Symbol(0));
  GE_WARN_ASSERT(var.IsValid());
  return GRAPH_SUCCESS;
}

bool StoreOp::InferDataType(const std::vector<DataType> &input_dtypes,
                            std::vector<DataType> &expect_output_dtypes) const {
  return AutofuseUtils::CallAscirInferDataType<ascir_op::Store>(input_dtypes, expect_output_dtypes) == SUCCESS;
}

graphStatus StoreReductionOp::RealizeImpl() {
  LoopCtx loop_ctx(src_dims_);
  Ops()->SetLoopAxis(loop_ctx.loop_axis);

  GE_WARN_ASSERT_GRAPH_SUCCESS(InferLoadIndex(loop_ctx.loop_axis.axis, this, loop_ctx));

  auto status = StrictTopoExecute(this, [&loop_ctx](const LoopOp *op) -> graphStatus {
    const auto result = op->Compute(loop_ctx);
    GE_WARN_ASSERT(result.IsValid(), "Loop kernel for %s got %s", op->Type().c_str(), result.Name().c_str());
    loop_ctx.Set(op, result);
    return GRAPH_SUCCESS;
  });
  GE_WARN_ASSERT_GRAPH_SUCCESS(status, "Failed build asc graph for %s", BufferName(dst_).c_str());
  GE_WARN_ASSERT(loop_ctx.Get(this).IsValid());
  return GRAPH_SUCCESS;
}

bool StoreReductionOp::InferDataType(const std::vector<DataType> &input_dtypes,
                                     std::vector<DataType> &expect_output_dtypes) const {
  return ReduceInferDataType(reduce_type_, input_dtypes, expect_output_dtypes);
}

graphStatus StoreConcatOp::RealizeImpl() {
  LoopCtx loop_ctx(dims_);
  auto load_index = std::make_shared<Index>(loop_ctx.loop_axis.axis);
  for (size_t i = 0U; i < inputs_.size(); i++) {
    auto input = inputs_[i].get();
    loop_ctx.load_2_index[input] = load_index;
    auto desc = TensorLoopDesc(input_dims_[i], ContiguousStrides(input_dims_[i]));
    loop_ctx.load_2_god_desc.emplace(input, LoopCtx::GodLoadSpec(desc, desc));
  }
  Ops()->SetLoopAxis(loop_ctx.loop_axis);
  const auto status = StrictTopoExecute(this, [&loop_ctx](const LoopOp *op) -> graphStatus {
    const auto result = op->Compute(loop_ctx);
    GE_WARN_ASSERT(result.IsValid(), "Loop kernel for %s got %s", op->Type().c_str(), result.Name().c_str());
    loop_ctx.Set(op, result);
    return GRAPH_SUCCESS;
  });
  GE_WARN_ASSERT_GRAPH_SUCCESS(status, "Failed build asc graph for %s", BufferName(dst_).c_str());
  GE_WARN_ASSERT(loop_ctx.Get(this).IsValid());
  return GRAPH_SUCCESS;
}

graphStatus StoreSplitOp:: RealizeImpl() {
  LoopCtx loop_ctx(input_dims_);
  auto load_index = std::make_shared<Index>(loop_ctx.loop_axis.axis);

  auto input = inputs_[0].get();
  loop_ctx.load_2_index[input] = load_index;
  auto desc = TensorLoopDesc(input_dims_, ContiguousStrides(input_dims_));
  loop_ctx.load_2_god_desc.emplace(input, LoopCtx::GodLoadSpec(desc, desc));
  GE_WARN_ASSERT_GRAPH_SUCCESS(InferLoadIndex(loop_ctx.loop_axis.axis, this, loop_ctx));
  Ops()->SetLoopAxis(loop_ctx.loop_axis);
  const auto status = StrictTopoExecute(this, [&loop_ctx](const LoopOp *op) -> graphStatus {
    const auto result = op->Compute(loop_ctx);
    GE_WARN_ASSERT(result.IsValid(), "Loop kernel for %s got %s", op->Type().c_str(), result.Name().c_str());
    loop_ctx.Set(op, result);
    return GRAPH_SUCCESS;
  });

  GE_WARN_ASSERT_GRAPH_SUCCESS(status, "Failed build asc graph for %s", BufferName(output_).c_str());

  GE_WARN_ASSERT(loop_ctx.Get(this).IsValid());
  return GRAPH_SUCCESS;
}

bool StoreConcatOp::InferDataType(const std::vector<DataType> &input_dtypes,
                                  std::vector<DataType> &expect_output_dtypes) const {
  std::vector<DataType> concat_input_dtypes;
  if (!input_dtypes.empty()) {
    concat_input_dtypes.emplace_back(input_dtypes[0]);
  }
  return AutofuseUtils::CallAscirInferDataType<ascir_op::Concat>(concat_input_dtypes, expect_output_dtypes) == SUCCESS;
}

bool ReduceThenBroadcastOp::InferDataType(const std::vector<DataType> &input_dtypes,
                                          std::vector<DataType> &expect_output_dtypes) const {
  return ReduceInferDataType(reduce_type_, input_dtypes, expect_output_dtypes);
}

bool ScalarOp::InferDataType(const std::vector<DataType> &input_dtypes,
                             std::vector<DataType> &expect_output_dtypes) const {
  DataType data_type = dtype_ == DT_BOOL ? DT_UINT8 : dtype_;
  expect_output_dtypes = {data_type};
  return AutofuseUtils::CallAscirInferDataType<ascir_op::Scalar>(input_dtypes, expect_output_dtypes) == SUCCESS;
}

bool BroadcastOp::InferDataType(const std::vector<DataType> &input_dtypes,
                                std::vector<DataType> &expect_output_dtypes) const {
  return AutofuseUtils::CallAscirInferDataType<ascir_op::Broadcast>(input_dtypes, expect_output_dtypes) == SUCCESS;
}

graphStatus StoreMatMulOp::RealizeImpl() {
  LoopCtx loop_ctx(dims_);
  // mm不考虑前融合，此处简化处理
  Ops()->SetLoopAxis(loop_ctx.loop_axis);

  auto load_index = std::make_shared<Index>(loop_ctx.loop_axis.axis);
  GE_ASSERT_NOTNULL(load_index);
  GE_WARN_ASSERT(inputs_.size() == input_dims_.size());
  for (size_t i = 0U; i < inputs_.size(); i++) {
    auto input = inputs_[i].get();
    loop_ctx.load_2_index[input] = load_index;
    auto desc = TensorLoopDesc(input_dims_[i], ContiguousStrides(input_dims_[i]));
    loop_ctx.load_2_god_desc.emplace(input, LoopCtx::GodLoadSpec(desc, desc));
  }

  auto status = StrictTopoExecute(this, [&loop_ctx](const LoopOp *op) -> graphStatus {
    const auto result = op->Compute(loop_ctx);
    GE_WARN_ASSERT(result.IsValid(), "Loop kernel for %s got %s", op->Type().c_str(), result.Name().c_str());
    loop_ctx.Set(op, result);
    return GRAPH_SUCCESS;
  });
  GE_WARN_ASSERT_GRAPH_SUCCESS(status, "Failed build asc graph for %s", BufferName(dst_).c_str());
  GE_WARN_ASSERT(loop_ctx.Get(this).IsValid());
  return GRAPH_SUCCESS;
}
}  // namespace loop
}  // namespace ge

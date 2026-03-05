/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "asc_graph_builder.h"

#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace ge::testing {
namespace {
void ComputeStrides(const std::vector<Expression> &repeats,
                    std::vector<Expression> &strides) {
  strides.clear();
  Expression stride = sym::kSymbolOne;
  for (auto iter = repeats.rbegin(); iter != repeats.rend(); ++iter) {
    if (SymbolicUtils::StaticCheckEq(*iter, ge::sym::kSymbolOne) == ge::TriBool::kTrue) {
      strides.push_back(ge::sym::kSymbolZero);
    } else {
      strides.push_back(stride);
      stride = stride * (*iter);
    }
  }
  std::reverse(strides.begin(), strides.end());
}
}

AscGraphBuilder::AscGraphBuilder(const std::string &name)
  : impl_(std::make_unique<Impl>(name)) {
}

AscGraphBuilder::~AscGraphBuilder() = default;

AscGraphBuilder &AscGraphBuilder::Loops(std::initializer_list<int64_t> sizes) {
  std::vector<Expression> expr_sizes;
  for (auto s: sizes) {
    expr_sizes.push_back(Symbol(s));
  }
  return Loops(expr_sizes);
}

AscGraphBuilder &AscGraphBuilder::Loops(std::initializer_list<Expression> sizes) {
  return Loops(std::vector<Expression>(sizes));
}

AscGraphBuilder &AscGraphBuilder::Loops(const std::vector<Expression> &sizes) {
  for (size_t i = 0; i < sizes.size(); ++i) {
    auto axis = impl_->graph_.CreateAxis("z" + std::to_string(i), sizes[i]);
    impl_->axis_ids_.push_back(axis.id);
    impl_->loop_repeats_.push_back(sizes[i]);
  }
  return *this;
}

AscGraphBuilder &AscGraphBuilder::Data(const std::string &name, int64_t index, DataType dtype) {
  ascir_op::Data data_op(name.c_str(), impl_->graph_);
  auto node = impl_->graph_.FindNode(name.c_str());
  assert(node != nullptr);
  data_op.ir_attr.SetIndex(index);
  data_op.y.dtype = dtype;
  impl_->nodes_[name] = node;
  return *this;
}

AscGraphBuilder &AscGraphBuilder::Data(const std::string &name, int64_t index,
                                       const std::vector<Expression> &shape,
                                       const std::vector<Expression> &strides,
                                       DataType dtype) {
  ascir_op::Data data_op(name.c_str(), impl_->graph_);
  auto node = impl_->graph_.FindNode(name.c_str());
  assert(node != nullptr);
  data_op.ir_attr.SetIndex(index);
  data_op.y.dtype = dtype;
  *data_op.y.axis = impl_->axis_ids_;
  *data_op.y.repeats = shape;
  *data_op.y.strides = strides;
  impl_->nodes_[name] = node;
  return *this;
}

AscGraphBuilder &AscGraphBuilder::Scalar(const std::string &name, const std::string &value, DataType dtype) {
  ascir_op::Scalar scalar_op(name.c_str(), impl_->graph_);
  scalar_op.ir_attr.SetValue(value);
  scalar_op.y.dtype = dtype;
  std::vector<Expression> scalar_repeats(impl_->loop_repeats_.size(), sym::kSymbolOne);
  *scalar_op.y.repeats = scalar_repeats;
  auto node = impl_->graph_.FindNode(name.c_str());
  assert(node != nullptr);
  impl_->nodes_[name] = node;
  return *this;
}

AscGraphBuilder &AscGraphBuilder::Output(const std::string &name, const std::string &input, int64_t index,
                                         DataType dtype) {
  ascir_op::Output output_op(name.c_str());
  auto node = impl_->graph_.AddNode(output_op);
  output_op.ir_attr.SetIndex(index);
  output_op.y.dtype = dtype;

  impl_->nodes_[name] = node;
  auto it = impl_->nodes_.find(input);
  assert(it != impl_->nodes_.end());
  GraphUtils::AddEdge(it->second->GetOutDataAnchor(0),
                      node->GetInDataAnchor(0));
  return *this;
}

AscGraphBuilder &AscGraphBuilder::Workspace(const std::string &name, const std::string &input, DataType dtype) {
  ascir_op::Workspace workspace_op(name.c_str());
  auto node = impl_->graph_.AddNode(workspace_op);
  workspace_op.y.dtype = dtype;

  impl_->nodes_[name] = node;
  if (!input.empty()) {
    auto it = impl_->nodes_.find(input);
    assert(it != impl_->nodes_.end());
    GraphUtils::AddEdge(it->second->GetOutDataAnchor(0),
                        node->GetInDataAnchor(0));
  }
  return *this;
}

// Load 通用实现
AscGraphBuilder &AscGraphBuilder::LoadImpl(const std::string &name, const std::string &input,
                                           const std::vector<Expression> *shape,
                                           const std::vector<Expression> *strides) {
  ascir_op::Load load_op(name.c_str());
  load_op.attr.sched.axis = impl_->axis_ids_;
  *load_op.y.axis = impl_->axis_ids_;

  auto &input_tensor = GetInputOutputTensor(input);
  load_op.y.dtype = input_tensor.attr.dtype;

  if (shape != nullptr && !shape->empty()) {
    *load_op.y.repeats = *shape;
  } else {
    *load_op.y.repeats = impl_->loop_repeats_;
  }
  if (strides != nullptr) {
    *load_op.y.strides = *strides;
  }

  CreateNodeAndConnect(name, load_op, input);
  return *this;
}

AscGraphBuilder &AscGraphBuilder::Load(const std::string &name, const std::string &input) {
  return LoadImpl(name, input, nullptr, nullptr);
}

AscGraphBuilder &AscGraphBuilder::Load(const std::string &name, const std::string &input,
                                       const std::vector<Expression> &shape,
                                       const std::vector<Expression> &strides) {
  return LoadImpl(name, input, &shape, &strides);
}

// Store 通用实现
AscGraphBuilder &AscGraphBuilder::StoreImpl(const std::string &name, const std::string &input,
                                            const std::vector<Expression> *shape,
                                            const std::vector<Expression> *strides) {
  ascir_op::Store store_op(name.c_str());
  store_op.attr.sched.axis = impl_->axis_ids_;

  auto &input_tensor = GetInputOutputTensor(input);
  *store_op.y.axis = input_tensor.attr.axis;
  store_op.y.dtype = input_tensor.attr.dtype;
  if (shape != nullptr) {
    *store_op.y.repeats = *shape;
  } else {
    *store_op.y.repeats = input_tensor.attr.repeats;
  }
  if (strides != nullptr) {
    *store_op.y.strides = *strides;
  }

  CreateNodeAndConnect(name, store_op, input);
  return *this;
}

AscGraphBuilder &AscGraphBuilder::Store(const std::string &name, const std::string &input) {
  return StoreImpl(name, input, nullptr, nullptr);
}

AscGraphBuilder &AscGraphBuilder::Store(const std::string &name, const std::string &input,
                                        const std::vector<Expression> &shape,
                                        const std::vector<Expression> &strides) {
  return StoreImpl(name, input, &shape, &strides);
}

AscGraphBuilder &AscGraphBuilder::BroadcastImpl(const std::string &name, const std::string &input,
                                                const std::vector<Expression> &output_shape) {
  auto &input_tensor = GetInputOutputTensor(input);

  ascir_op::Broadcast brc_op(name.c_str());
  brc_op.attr.sched.axis = impl_->axis_ids_;
  *brc_op.y.axis = input_tensor.attr.axis;
  brc_op.y.dtype = input_tensor.attr.dtype;
  *brc_op.y.repeats = output_shape;

  CreateNodeAndConnect(name, brc_op, input);
  return *this;
}

AscGraphBuilder &AscGraphBuilder::Broadcast(const std::string &name, const std::string &input,
                                            const std::vector<int64_t> &brc_axes) {
  auto &input_tensor = GetInputOutputTensor(input);

  std::vector<Expression> output_shape = input_tensor.attr.repeats;
  for (int64_t axis: brc_axes) {
    if (axis >= 0 && axis < static_cast<int64_t>(output_shape.size()) &&
        axis < static_cast<int64_t>(impl_->loop_repeats_.size())) {
      output_shape[axis] = impl_->loop_repeats_[axis];
    }
  }

  return BroadcastImpl(name, input, output_shape);
}

AscGraphBuilder &AscGraphBuilder::Broadcast(const std::string &name, const std::string &input,
                                            std::initializer_list<int64_t> brc_axes) {
  std::vector<int64_t> axes_vec(brc_axes);
  return Broadcast(name, input, axes_vec);
}

AscGraphBuilder &AscGraphBuilder::Broadcast(const std::string &name, const std::string &input,
                                            const std::vector<Expression> &shape) {
  return BroadcastImpl(name, input, shape);
}

AscGraphBuilder &AscGraphBuilder::Transpose(const std::string &name, const std::string &input,
                                            const std::vector<int64_t> &axes) {
  auto &input_tensor = GetInputOutputTensor(input);
  const auto &input_shape = input_tensor.attr.repeats;

  std::vector<Expression> output_shape;
  std::vector<AxisId> output_axis;
  for (int64_t axis_idx: axes) {
    if (axis_idx >= 0 && axis_idx < static_cast<int64_t>(input_shape.size())) {
      output_shape.push_back(input_shape[axis_idx]);
    }
    if (axis_idx >= 0 && axis_idx < static_cast<int64_t>(impl_->axis_ids_.size())) {
      output_axis.push_back(impl_->axis_ids_[axis_idx]);
    }
  }

  ascir_op::Transpose transpose_op(name.c_str());
  transpose_op.attr.sched.axis = output_axis;
  *transpose_op.y.repeats = output_shape;
  *transpose_op.y.axis = output_axis;
  transpose_op.y.dtype = input_tensor.attr.dtype;

  CreateNodeAndConnect(name, transpose_op, input);
  return *this;
}

AscGraphBuilder &AscGraphBuilder::Concat(const std::string &name, const std::vector<std::string> &inputs) {
  impl_->dynamic_input_ops_.emplace_back();
  auto &ops = impl_->dynamic_input_ops_.back();
  std::vector<ge::AscOpOutput> outputs;
  ops.reserve(inputs.size());
  outputs.reserve(inputs.size());

  for (const auto &input: inputs) {
    auto it = impl_->nodes_.find(input);
    assert(it != impl_->nodes_.end());
    ops.push_back(ge::OpDescUtils::CreateOperatorFromNode(it->second));
    outputs.emplace_back(&ops.back(), 0);
  }

  ascir_op::Concat concat_op(name.c_str());

  // 设置动态输入 - 这会自动创建节点并添加到图中
  concat_op.x = outputs;

  auto const_node = ge::NodeUtilsEx::GetNodeFromOperator(concat_op);
  assert(const_node != nullptr);
  auto node_ptr = std::const_pointer_cast<ge::Node>(const_node);
  auto asc_node = std::dynamic_pointer_cast<ge::AscNode>(node_ptr);
  assert(asc_node != nullptr);

  asc_node->attr.sched.axis = impl_->axis_ids_;
  auto &output = asc_node->outputs[0];
  auto &input_tensor = GetInputOutputTensor(inputs.empty() ? "" : inputs[0]);
  output.attr.axis = input_tensor.attr.axis;
  output.attr.dtype = input_tensor.attr.dtype;
  output.attr.repeats = impl_->loop_repeats_;

  impl_->nodes_[name] = asc_node;

  return *this;
}

// 通用 Reduce 模板实现
template<typename ReduceOp>
AscGraphBuilder &AscGraphBuilder::Reduce(const std::string &name, const std::string &input,
                                         const std::vector<size_t> &reduce_axes) {
  auto &input_tensor = GetInputOutputTensor(input);

  std::vector<Expression> output_shape = input_tensor.attr.repeats;
  for (size_t axis: reduce_axes) {
    if (axis < output_shape.size()) {
      output_shape[axis] = sym::kSymbolOne;
    }
  }

  ReduceOp reduce_op(name.c_str());
  reduce_op.attr.sched.axis = impl_->axis_ids_;
  *reduce_op.y.axis = input_tensor.attr.axis;
  reduce_op.y.dtype = input_tensor.attr.dtype;
  *reduce_op.y.repeats = output_shape;

  CreateNodeAndConnect(name, reduce_op, input);
  return *this;
}

template AscGraphBuilder &AscGraphBuilder::Reduce<ascir_op::Max>(
  const std::string &, const std::string &, const std::vector<size_t> &);

template AscGraphBuilder &AscGraphBuilder::Reduce<ascir_op::Sum>(
  const std::string &, const std::string &, const std::vector<size_t> &);

AscGraphBuilder &AscGraphBuilder::Cast(const std::string &name, const std::string &input, DataType dtype) {
  auto &input_tensor = GetInputOutputTensor(input);

  ascir_op::Cast cast_op(name.c_str());
  cast_op.attr.sched.axis = impl_->axis_ids_;
  cast_op.y.dtype = dtype;
  *cast_op.y.axis = input_tensor.attr.axis;
  *cast_op.y.repeats = input_tensor.attr.repeats;

  CreateNodeAndConnect(name, cast_op, input);
  return *this;
}

void AscGraphBuilder::ConnectEdge(const std::string &src_name, AscNodePtr dst_node, size_t dst_index) {
  auto it = impl_->nodes_.find(src_name);
  assert(it != impl_->nodes_.end());
  GraphUtils::AddEdge(it->second->GetOutDataAnchor(0), dst_node->GetInDataAnchor(dst_index));
}

AscGraph AscGraphBuilder::Build() {
  for (const auto &[name, node]: impl_->nodes_) {
    if (node->attr.api.type == ge::ApiType::kAPITypeBuffer) {
      continue;
    }
    node->attr.sched.axis = impl_->axis_ids_;

    for (auto &output: node->outputs()) {
      if (output->attr.axis.empty()) {
        output->attr.axis = impl_->axis_ids_;
      }

      if (output->attr.repeats.empty()) {
        output->attr.repeats = impl_->loop_repeats_;
      }

      if (output->attr.strides.empty()) {
        ComputeStrides(output->attr.repeats, output->attr.strides);
      }
    }
  }

  return impl_->graph_;
}
} // namespace ge::testing

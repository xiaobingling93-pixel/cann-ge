/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "loop_api.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include <regex>

#include "common/checker.h"
#include "graph/node.h"
#include "graph/symbolizer/symbolic.h"
#include "utils/auto_fuse_config.h"
#include "utils/autofuse_utils.h"
#include "asc_overrides.h"
#include "kernel_box.h"
#include "loop_common.h"
#include "loop_ops.h"

#include "backend/backend_spec.h"

namespace ge {
using namespace autofuse;
namespace loop {
class LoweringResultAttrs : public ge::AttrGroupsBase {
 public:
  KernelBox Set(const ge::OutDataAnchorPtr &dst, const std::shared_ptr<KernelBoxMeta> &meta) {
    lowering_results_[dst] = meta;
    return KernelBox(dst, meta);
  }

  KernelBox Get(const ge::OutDataAnchorPtr &dst) {
    auto iter = lowering_results_.find(dst);
    if (iter != lowering_results_.end()) {
      return KernelBox(dst, iter->second);
    }
    return Set(dst, std::make_shared<KernelBoxMeta>());
  }

  std::unique_ptr<AttrGroupsBase> Clone() override {
    // do nothing as can not serialize/deserialize
    return std::make_unique<LoweringResultAttrs>();
  }

 private:
  std::map<ge::OutDataAnchorPtr, std::shared_ptr<KernelBoxMeta>> lowering_results_;
};

KernelBox GetKernelBox(const ge::OutDataAnchorPtr &dst) {
  LOWERING_ASSERT_NOTNULL(dst);
  LOWERING_ASSERT_NOTNULL(dst->GetOwnerNode());
  LOWERING_ASSERT_NOTNULL(dst->GetOwnerNode()->GetOpDesc());
  auto lowered = dst->GetOwnerNode()->GetOpDesc()->GetOrCreateAttrsGroup<LoweringResultAttrs>();
  LOWERING_ASSERT_NOTNULL(lowered);
  return lowered->Get(dst);
}

KernelBox StoreExtern(const ge::OutDataAnchorPtr &dst) {
  LOWERING_ASSERT_NOTNULL(dst);
  LOWERING_ASSERT_NOTNULL(dst->GetOwnerNode());
  LOWERING_ASSERT_NOTNULL(dst->GetOwnerNode()->GetOpDesc());
  const auto lowered = dst->GetOwnerNode()->GetOpDesc()->GetOrCreateAttrsGroup<LoweringResultAttrs>();
  LOWERING_ASSERT_NOTNULL(lowered);
  return lowered->Set(dst, std::make_shared<KernelBoxMeta>());
}

namespace {
void DropLowerResultIfNeeded(const std::shared_ptr<KernelBoxMeta> &meta, const ge::OutDataAnchorPtr &dst,
                             const LoopVar &result) {
  // 暂时添加环境变量控制是否允许lowering成matmul
  if ((meta->type == FuseType::kCube) && !ge::AutoFuseConfig::LoweringConfig().experimental_lowering_matmul) {
    GELOGI(
        "Drop lower result %s of %s as disabled, you can enable it by setting "
        "AUTOFUSE_FLAGS=\"--autofuse_enable_pass=matmul\""
        " and unsetting AUTOFUSE_FLAGS=\"--autofuse_disable_pass=matmul\"",
        result.Readable().c_str(), BufferName(dst).c_str());
    meta->type = FuseType::kExtern;
  }
}

KernelBox SetLoopKernel(const ge::OutDataAnchorPtr &dst, const LoopVar &result) {
  LOWERING_ASSERT_NOTNULL(dst);
  LOWERING_ASSERT_NOTNULL(dst->GetOwnerNode());
  LOWERING_ASSERT_NOTNULL(dst->GetOwnerNode()->GetOpDesc());
  const auto lowered = dst->GetOwnerNode()->GetOpDesc()->GetOrCreateAttrsGroup<LoweringResultAttrs>();
  LOWERING_ASSERT_NOTNULL(lowered);
  auto meta = std::make_shared<KernelBoxMeta>(result);
  if (meta->type == FuseType::kReduction && !ge::AutoFuseConfig::LoweringConfig().experimental_lowering_reduce) {
    GELOGI(
        "Drop lower result %s of %s as disabled, you can enable it by setting "
        "AUTOFUSE_FLAGS=\"--autofuse_enable_pass=reduce\""
        " and unsetting AUTOFUSE_FLAGS=\"--autofuse_disable_pass=reduce\"",
        result.Readable().c_str(), BufferName(dst).c_str());
    meta->type = FuseType::kExtern;
  }
  if (meta->type == FuseType::kConcat && !ge::AutoFuseConfig::LoweringConfig().experimental_lowering_concat) {
    GELOGI(
        "Drop lower result %s of %s as disabled, you can enable it by setting "
        "AUTOFUSE_FLAGS=\"--autofuse_enable_pass=concat\""
        " and unsetting AUTOFUSE_FLAGS=\"--autofuse_disable_pass=concat\"",
        result.Readable().c_str(), BufferName(dst).c_str());
    meta->type = FuseType::kExtern;
  }
  if (meta->type == FuseType::kSplit && !ge::AutoFuseConfig::LoweringConfig().experimental_lowering_split) {
    GELOGI(
      "Drop lower result %s of %s as disabled, you can enable it by setting "
      "AUTOFUSE_FLAGS=\"--autofuse_enable_pass=split\""
      "and unsetting AUTOFUSE_FLAGS=\"--autofuse_disable_pass=split\"",
           result.Readable().c_str(), BufferName(dst).c_str());
    meta->type = FuseType::kExtern;
  }
  if (meta->type == FuseType::kSliceSplit && !ge::AutoFuseConfig::LoweringConfig().experimental_lowering_slice) {
    GELOGI(
        "Drop lower result %s of %s as disabled, you can enable it by setting "
        "AUTOFUSE_FLAGS=\"--autofuse_enable_pass=slice\""
        " and unsetting AUTOFUSE_FLAGS=\"--autofuse_disable_pass=slice\"",
        result.Readable().c_str(), BufferName(dst).c_str());
    meta->type = FuseType::kExtern;
  }
  if (meta->type == FuseType::kGather && !ge::AutoFuseConfig::LoweringConfig().experimental_lowering_gather) {
    GELOGI(
        "Drop lower result %s of %s as disabled, you can enable it by setting "
        "AUTOFUSE_FLAGS=\"--autofuse_enable_pass=gather\""
        " and unsetting AUTOFUSE_FLAGS=\"--autofuse_disable_pass=gather\"",
        result.Readable().c_str(), BufferName(dst).c_str());
    meta->type = FuseType::kExtern;
  }
  DropLowerResultIfNeeded(meta, dst, result);
  return lowered->Set(dst, meta);
}
}  // namespace

string ToStringSetPrecesion(float value, int32_t precision = 20) {
  std::ostringstream out;
  out << std::scientific << std::setprecision(precision) << value;
  return out.str();
}

graphStatus GetScalarFromInput(const ge::InDataAnchorPtr &src, ge::loop::LoopVar &value) {
  auto out = src->GetPeerOutAnchor();
  GE_WARN_ASSERT(out != nullptr);
  auto src_node = out->GetOwnerNodeBarePtr();
  GE_WARN_ASSERT(src_node != nullptr);
  std::vector<Expression> dims;
  GE_WARN_ASSERT(loop::GetBufferShape(out, dims) == GRAPH_SUCCESS);
  if (!(dims.empty() && (src_node->GetType() == "Const" || src_node->GetType() == "Constant"))) {
    return GRAPH_FAILED;
  }
  auto node = src->GetOwnerNode();
  GE_WARN_ASSERT(node != nullptr);
  auto desc = src->GetOwnerNode()->GetOpDesc();
  GE_WARN_ASSERT(desc != nullptr);
  const auto op = ge::OpDescUtils::CreateOperatorFromNode(node);
  ge::Tensor val_tensor;
  GE_WARN_ASSERT(op.GetInputConstData(desc->GetInputNameByIndex(src->GetIdx()).c_str(), val_tensor) == ge::SUCCESS,
                 "%s %s as failed to get tensor %s", node->GetNamePtr(), node->GetTypePtr(),
                 desc->GetInputNameByIndex(src->GetIdx()).c_str());
  GE_WARN_ASSERT(val_tensor.GetData() != nullptr);
  const ge::DataType dtype = val_tensor.GetTensorDesc().GetDataType();
  if (dtype == ge::DT_FLOAT) {
    float tensor_val = *reinterpret_cast<const float *>(val_tensor.GetData());
    value = loop::Scalar(ToStringSetPrecesion(tensor_val), dtype);
  } else if (dtype == ge::DT_INT8) {
    int8_t tensor_val = *reinterpret_cast<const int8_t *>(val_tensor.GetData());
    value = loop::Scalar(to_string(tensor_val), dtype);
  } else if (dtype == ge::DT_INT32) {
    int32_t tensor_val = *reinterpret_cast<const int32_t *>(val_tensor.GetData());
    value = loop::Scalar(to_string(tensor_val), dtype);
  } else if (dtype == ge::DT_UINT8 || dtype == ge::DT_BOOL) {
    uint8_t tensor_val = *reinterpret_cast<const uint8_t *>(val_tensor.GetData());
    value = loop::Scalar(to_string(tensor_val), ge::DT_UINT8);
  } else if (dtype == ge::DT_INT16) {
    int16_t tensor_val = *reinterpret_cast<const int16_t *>(val_tensor.GetData());
    value = loop::Scalar(to_string(tensor_val), dtype);
  } else if (dtype == ge::DT_UINT16) {
    uint16_t tensor_val = *reinterpret_cast<const uint16_t *>(val_tensor.GetData());
    value = loop::Scalar(to_string(tensor_val), dtype);
  }else if (dtype == ge::DT_UINT32) {
    uint32_t tensor_val = *reinterpret_cast<const uint32_t *>(val_tensor.GetData());
    value = loop::Scalar(to_string(tensor_val), dtype);
  } else {
    GE_WARN_ASSERT(dtype == ge::DT_FLOAT16, "Const Scalar only support {DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_INT16, DT_UINT16, DT_UINT32}");
    GELOGW("Unable to tran not support DT_FLOAT16");
    return GRAPH_FAILED;
  }
  GELOGD("Node %s create scalar ascir without load ascir", src_node->GetName().c_str());
  return GRAPH_SUCCESS;
}

LoopVar Load(const ge::InDataAnchorPtr &src) {
  GE_WARN_ASSERT(src != nullptr);
  auto out = src->GetPeerOutAnchor();
  GE_WARN_ASSERT(out != nullptr);
  LoopVar value;
  if (GetScalarFromInput(src, value) == GRAPH_SUCCESS && value.IsValid()) {
    return value;
  }

  auto kernel_box = GetKernelBox(out);
  return kernel_box.Load(src);
}

LoopVar GatherLoad(const ge::OutDataAnchorPtr &dst, const ge::InDataAnchorPtr &params,
    const ge::InDataAnchorPtr &indices, int64_t axis, bool negative_index_support) {
  std::vector<Expression> dims;
  GE_WARN_ASSERT(GetBufferShape(dst, dims) == GRAPH_SUCCESS,
    "Drop lower result of %s as it has no sym shape", BufferName(dst).c_str());
  
  std::vector<GatherInput> inputs;
  std::vector<std::vector<Expression>> input_dims;
  inputs.emplace_back(GatherInput{params, {}});
  inputs.emplace_back(GatherInput{indices, {}});
  input_dims.resize(inputs.size());

  std::vector<ge::OutDataAnchorPtr> outputs;
  for (size_t i = 0U; i < inputs.size(); i++) {
    GE_WARN_ASSERT(inputs[i].input_anchor != nullptr && inputs[i].input_anchor->GetPeerOutAnchor() != nullptr,
      "Drop lower result of %s as input %zu is nullptr", BufferName(dst).c_str(), i);

    outputs.emplace_back(inputs[i].input_anchor->GetPeerOutAnchor());

    const auto buffer = inputs[i].input_anchor->GetPeerOutAnchor().get();
    GE_WARN_ASSERT(GetBufferShape(buffer, inputs[i].input_dim) == GRAPH_SUCCESS,
      "Drop lower result of %s as input %s has no sym shape", BufferName(dst).c_str(),
      BufferName(buffer).c_str());
  }
  GE_WARN_ASSERT(outputs.size() == inputs.size());
  auto loop_var = LoopVar(std::make_shared<LoadGatherOp>(dst.get(), outputs, inputs, dims, axis, negative_index_support));
  return loop_var;
}

KernelBox Store(const ge::OutDataAnchorPtr &dst, const LoopVar &src) {
  std::vector<Expression> dims;
  if (loop::GetBufferShape(dst, dims) != GRAPH_SUCCESS) {
    GELOGI("Drop lower result %s of %s as buffer has no sym shape", src.Readable().c_str(), BufferName(dst).c_str());
    return StoreExtern(dst);
  }
  if (!src.IsValid()) {
    GELOGI("Drop lower result %s of %s as loop var is invalid", src.Readable().c_str(), BufferName(dst).c_str());
    return StoreExtern(dst);
  }

  return SetLoopKernel(dst, LoopVar(std::make_shared<StoreOp>(dst, src.Op(), dims)));
}

KernelBox StoreReduction(ReduceType type, const ge::OutDataAnchorPtr &dst, const LoopVar &src,
                         const std::vector<Expression> &src_dims, const std::vector<size_t> &reduced_axis) {
  std::vector<Expression> reduced_dims;
  if (GetBufferShape(dst, reduced_dims) != GRAPH_SUCCESS) {
    GELOGI("Drop lower result %s of %s as it has no sym shape", src.Readable().c_str(), BufferName(dst).c_str());
    return StoreExtern(dst);
  }
  if (!src.IsValid()) {
    GELOGI("Drop lower result %s of %s as loop var is invalid", src.Readable().c_str(), BufferName(dst).c_str());
    return StoreExtern(dst);
  }
  const bool is_keep_dim = reduced_dims.size() == src_dims.size();
  const bool all_one = std::all_of(reduced_axis.begin(), reduced_axis.end(), [src_dims](const size_t &dim) {
    return (SymbolicUtils::StaticCheckEq(src_dims[dim], Symbol(1)) == TriBool::kTrue);
  });
  // reduce的所有轴都为1，可由squeeze进行减轴
  if (all_one) {
    const auto node = dst->GetOwnerNodeBarePtr();
    if (is_keep_dim) {
      GELOGI("Optimize reduce %s to store as keep dim", node->GetName().c_str());
      return loop::Store(dst, src);
    }
    GELOGI("Optimize reduce %s to squeeze as not keep dim", node->GetName().c_str());
    auto ordered_reduced_axis = reduced_axis;
    sort( ordered_reduced_axis.begin(),  ordered_reduced_axis.end());
    auto x = src;
    for (size_t i = 0; i <  ordered_reduced_axis.size(); ++i) {
      x = loop::Squeeze(x, static_cast<int64_t>( ordered_reduced_axis[i] - i));
    }
    return loop::Store(dst, x);
  }
  std::vector<StoreReductionOp::DimKind> reduce_status;
  for (size_t i = 0U; i < src_dims.size(); i++) {
    if (std::find(reduced_axis.begin(), reduced_axis.end(), i) == reduced_axis.end()) {
      reduce_status.emplace_back(StoreReductionOp::DimKind::NORM);
    } else {
      reduce_status.emplace_back(is_keep_dim ? StoreReductionOp::DimKind::REDUCE : StoreReductionOp::DimKind::REMOVE);
    }
  }
  auto loop_var = LoopVar(std::make_shared<StoreReductionOp>(type, dst, src.Op(), src_dims, reduce_status));
  return SetLoopKernel(dst, loop_var).Realize();
}

KernelBox StoreConcat(const ge::OutDataAnchorPtr &dst, const std::vector<ge::InDataAnchorPtr> &inputs,
                      size_t concat_dim) {
  std::vector<Expression> dims;
  if (GetBufferShape(dst, dims) != GRAPH_SUCCESS) {
    GELOGI("Drop lower result of %s as it has no sym shape", BufferName(dst).c_str());
    return StoreExtern(dst);
  }
  std::vector<std::vector<Expression>> input_dims;
  std::vector<LoopOpPtr> input_buffers;
  input_dims.resize(inputs.size());
  for (size_t i = 0U; i < inputs.size(); i++) {
    if (inputs[i] == nullptr || inputs[i]->GetPeerOutAnchor() == nullptr) {
      GELOGI("Drop lower result of %s as input %zu is nullptr", BufferName(dst).c_str(), i);
      return StoreExtern(dst);
    }
    loop::GetKernelBox(inputs[i]->GetPeerOutAnchor()).Realize();
    input_buffers.emplace_back(loop::Load(inputs[i]).Op());
    const auto buffer = inputs[i]->GetPeerOutAnchor().get();
    if (GetBufferShape(buffer, input_dims[i]) != GRAPH_SUCCESS) {
      GELOGI("Drop lower result of %s as input %s has no sym shape", BufferName(dst).c_str(),
             BufferName(buffer).c_str());
      return StoreExtern(dst);
    }
  }
  auto loop_var = LoopVar(std::make_shared<StoreConcatOp>(dst.get(), input_buffers, input_dims, concat_dim, dims));
  return SetLoopKernel(dst, loop_var).Realize();
}

KernelBox StorePack(const ge::OutDataAnchorPtr &dst, const std::vector<ge::InDataAnchorPtr> &inputs,
                    int64_t packed_dim) {
  std::vector<Expression> dims;
  if (GetBufferShape(dst, dims) != GRAPH_SUCCESS) {
    GELOGI("Drop lower result of %s as it has no sym shape", BufferName(dst).c_str());
    return StoreExtern(dst);
  }
  packed_dim = packed_dim < 0 ? packed_dim + static_cast<int64_t>(dims.size()) : packed_dim;
  if (packed_dim < 0 || static_cast<size_t>(packed_dim) >= dims.size()) {
    GELOGW("Drop lower result of %s as packed dim %ld is out of range", BufferName(dst).c_str(), packed_dim);
    return StoreExtern(dst);
  }
  std::vector<Expression> input_dim = dims;
  input_dim[packed_dim] = Symbol(1);
  std::vector<std::vector<Expression>> input_dims;
  input_dims.resize(inputs.size(), input_dim);
  std::vector<LoopOpPtr> input_buffers;
  for (size_t i = 0U; i < inputs.size(); i++) {
    if (inputs[i] == nullptr || inputs[i]->GetPeerOutAnchor() == nullptr) {
      GELOGI("Drop lower result of %s as input %zu is nullptr", BufferName(dst).c_str(), i);
      return StoreExtern(dst);
    }
    loop::GetKernelBox(inputs[i]->GetPeerOutAnchor()).Realize();
    input_buffers.emplace_back(loop::Load(inputs[i]).Op());
  }
  auto loop_var = LoopVar(std::make_shared<StoreConcatOp>(dst.get(), input_buffers, input_dims, packed_dim, dims));
  return SetLoopKernel(dst, loop_var).Realize();
}

bool IsNumber(const std::string& s) {
  std::regex pattern(
      "^("
      // 十进制数字
      "[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?"
      "|"
      // 十六进制（0x前缀）
      "0[xX][0-9a-fA-F]+"
      "|"
      // 二进制（0b前缀）
      "0[bB][01]+"
      "|"
      // 八进制（0前缀）
      "0[0-7]*"
      "|"
      // 纯十六进制（无前缀）
      "[0-9a-fA-F]+"
      ")$"
  );
  return std::regex_match(s, pattern);
}

LoopVar Scalar(std::string face, ge::DataType dtype) {
  GE_WARN_ASSERT(IsNumber(face) || face == "inf", "Scalar face %s is not a number or inf", face.c_str());
  return LoopVar(std::make_shared<ScalarOp>(std::move(face), dtype));
}

LoopVar Broadcast(const LoopVar &op, std::vector<BroadcastOp::DimKind> status) {
  GE_WARN_ASSERT(op.IsValid());
  if (std::all_of(status.begin(), status.end(),
                  [](const BroadcastOp::DimKind &kind) { return kind == BroadcastOp::DimKind::NORMAL; })) {
    return op;
  }
  return LoopVar(std::make_shared<BroadcastOp>(op.Op(), std::move(status)));
}

LoopVar Broadcast(const LoopVar &op, std::vector<Expression> src, std::vector<Expression> dst) {
  GE_WARN_ASSERT(op.IsValid());
  GE_WARN_ASSERT(dst.size() >= src.size());
  size_t num_new_axis = dst.size() - src.size();
  std::vector<BroadcastOp::DimKind> status;
  status.resize(num_new_axis, BroadcastOp::DimKind::NEW_AXIS);
  auto *src_p = src.data();
  auto *dst_p = dst.data() + num_new_axis;
  for (size_t i = 0U; i < src.size(); i++) {
    // 依赖符号化ShapeEnv的能力增强，例如符号化推导阶段，ge.Add([s0], [s1]) ->
    // [s0]，s1和s0会guard相等，此时判断s0和s1相等要返回true
    if (EXPECT_SYMBOL_EQ(src_p[i], dst_p[i])) {
      status.push_back(BroadcastOp::DimKind::NORMAL);
    } else if (EXPECT_SYMBOL_EQ(src_p[i], Symbol(1))){
      status.push_back(BroadcastOp::DimKind::BROADCAST);
    } else {
      GELOGW("Failed broadcast %s to %s", src_p[i].Str().get(), dst_p[i].Str().get());
      return LoopVar(nullptr);
    }
  }
  return LoopVar(Broadcast(op, std::move(status)));
}

LoopVar Permute(const LoopVar &op, std::vector<size_t> order) {
  GE_WARN_ASSERT(op.IsValid());
  return LoopVar(std::make_shared<PermuteOp>(op.Op(), std::move(order)));
}

bool IsStaticShape(std::vector<Expression> output_dims) {
  for (const auto &exp : output_dims) {
    if (!exp.IsConstExpr()) {
      return false;
    }
  }
  return true;
}

bool IsSameBatchSize(const std::vector<std::vector<Expression>> &input_dims) {
  if (input_dims.size() < 2U) {
    return false;
  }
  // 前两个是matmul的输入矩阵
  const auto &input1 = input_dims[0];
  const auto &input2 = input_dims[1];
  Expression batch_size1 = Symbol(1);
  Expression batch_size2 = Symbol(1);
  // 两个矩阵的 batch size 必须相同才lowering
  for (size_t i = 0U; i < input1.size() - 2U; i++) {
    batch_size1 = batch_size1 * input1[i];
  }
  for (size_t i = 0U; i < input2.size() - 2U; ++i) {
    batch_size2 = batch_size2 * input2[i];
  }
  if (ge::SymbolicUtils::StaticCheckEq(batch_size1, batch_size2) != ge::TriBool::kTrue) {
    return false;
  }
  return true;
}

KernelBox StoreMatMul(const ge::OutDataAnchorPtr &dst, const std::vector<ge::InDataAnchorPtr> &inputs,
                      const MatMulAttr &matmul_attr) {
  if ((inputs.size() <= 1U) || (inputs.size() > 4U)) {
    GELOGI("Drop lower result of %s as it has err inputs num=%zu", BufferName(dst).c_str(), inputs.size());
    return StoreExtern(dst);
  }
  std::vector<Expression> dims;
  if (GetBufferShape(dst, dims) != GRAPH_SUCCESS) {
    GELOGI("Drop lower result of %s as it has no sym shape", BufferName(dst).c_str());
    return StoreExtern(dst);
  }
  std::vector<LoopOpPtr> input_buffers;
  std::vector<std::vector<Expression>> input_dims;
  input_dims.resize(inputs.size());
  size_t max_dim = 0U;
  for (size_t i = 0U; i < inputs.size(); i++) {
    if ((inputs[i] == nullptr) || (inputs[i]->GetPeerOutAnchor() == nullptr)) {
      GELOGI("Drop lower result of %s as input %zu is nullptr", BufferName(dst).c_str(), i);
      return StoreExtern(dst);
    }
    loop::GetKernelBox(inputs[i]->GetPeerOutAnchor()).Realize();
    input_buffers.emplace_back(loop::Load(inputs[i]).Op());
    const auto buffer = inputs[i]->GetPeerOutAnchor().get();
    if (GetBufferShape(buffer, input_dims[i]) != GRAPH_SUCCESS) {
      GELOGI("Drop lower result of %s as input %s has no sym shape", BufferName(dst).c_str(),
             BufferName(buffer).c_str());
      return StoreExtern(dst);
    }
    if (input_dims[i].size() > max_dim) {
      max_dim = input_dims[i].size();
    }
  }

  for (size_t i = 0U; i < inputs.size(); i++) {
    if (input_dims[i].size() < max_dim) {
      size_t diff = max_dim - input_dims[i].size();
      for (size_t j = 0U; j < diff; j++) {
        input_dims[i].insert(input_dims[i].begin(), ge::Symbol(1));
      }
    }
  }
  if (!IsSameBatchSize(input_dims)) {
    GELOGI("Drop lower result of %s as it has different batch size", BufferName(dst).c_str());
    return StoreExtern(dst);
  }

  if (!IsStaticShape(dims)) {
    GELOGI("Drop lower result of %s as it is not static shape", BufferName(dst).c_str());
    return StoreExtern(dst);
  }

  auto loop_var = LoopVar(std::make_shared<StoreMatMulOp>(dst.get(), input_buffers, matmul_attr, dims, input_dims));
  return SetLoopKernel(dst, loop_var).Realize();
}

LoopVar Squeeze(const LoopVar &op, int64_t dim) {
  GE_WARN_ASSERT(op.IsValid());
  return LoopVar(std::make_shared<SqueezeOp>(op.Op(), dim));
}

LoopVar Unsqueeze(const LoopVar &op, int64_t dim) {
  GE_WARN_ASSERT(op.IsValid());
  return LoopVar(std::make_shared<UnsqueezeOp>(op.Op(), dim));
}

bool CheckAndGetDims(const std::vector<Expression> &long_dims, const std::vector<Expression> &short_dims, std::vector<int64_t> &dims) {
  std::vector<bool> is_exist(long_dims.size(), false);
  size_t dims_idx = 0U;
  for (const auto &short_dim : short_dims) {
    bool is_find = false;
    string longdim_str;
    for (size_t i = dims_idx; i < long_dims.size(); i++) {
      longdim_str += " " + SymbolicUtils::ToString(long_dims[i]);
      if (SymbolicUtils::StaticCheckEq(long_dims[i], short_dim) == ge::TriBool::kTrue && !is_exist[i]) {
        is_find = true;
        is_exist[i] = true;
        dims_idx = i;
        break;
      }
    }
    GE_WARN_ASSERT(
        is_find,
        "There are some axes in the short axis that do not exist in the long axis, short axes: %s, long dims: %s",
        SymbolicUtils::ToString(short_dim).c_str(), longdim_str.c_str());
  }

  for (size_t i = 0U; i < is_exist.size(); i++) {
    if ((!is_exist[i]) && (long_dims[i] != Symbol(1))) {
      GELOGW("Axes that do not exist must be 1, Long axes: %zu, not equal than 1", i);
      return false;
    }
    if (!is_exist[i]) {
      dims.push_back(static_cast<int64_t>(i));
    }
  }
  return !dims.empty();
}

// Reshape只做attr为默认参数，且不进行轴转换，能进行unsqueeze/squeeze的情况。[3,4]->[1,3,4]/[2,1,3]->[2,3]
// 新增支持 [A*B, C]->[A,B,C]/[A,B,C]->[A*B,C]
LoopVar Reshape(const LoopVar &op, const std::vector<Expression> &src_dims, const std::vector<Expression> &dst_dims) {
  std::vector<int64_t> dims_new;
  GE_WARN_ASSERT(src_dims.size() != dst_dims.size(),
               "Input dims size equal than output dims");
  LoopVar reshape = op;
  size_t short_idx = 0U;
  std::vector<size_t> mul_idx;
  if (src_dims.size() > dst_dims.size()) {
    if (CheckAndGetDims(src_dims, dst_dims, dims_new)) {
      for (size_t i = 0U; i < dims_new.size(); i++) {
        // -i为后续减轴，index维度有变化需要更新
        reshape = loop::Squeeze(reshape, dims_new[i] - static_cast<int64_t>(i));
      }
      return reshape;
    }
    if (AutofuseUtils::CheckAndMulDetect(src_dims, dst_dims, short_idx, mul_idx)) {
      reshape = LoopVar(std::make_shared<ReshapeOp>(op.Op(), src_dims, dst_dims, short_idx, mul_idx));
      return reshape;
    }
  } else {
    if (CheckAndGetDims(dst_dims, src_dims, dims_new)) {
      for (const auto &new_dim : dims_new) {
        reshape = loop::Unsqueeze(reshape, new_dim);
      }
      return reshape;
    }
    if (AutofuseUtils::CheckAndMulDetect(dst_dims, src_dims, short_idx, mul_idx)) {
      reshape = LoopVar(std::make_shared<ReshapeOp>(op.Op(), src_dims, dst_dims, short_idx, mul_idx));
      return reshape;
    }
  }

  return LoopVar();
}

bool IsInvalidTranspose(const std::vector<ge::Expression> &dims,
                        const std::vector<int64_t> &perm,
                        std::vector<int64_t> &squeeze_axis) {
  GE_ASSERT_TRUE(dims.size() == perm.size(), "dims and perm dimensions are not equal.");
  // 1.过滤掉dim为1的轴
  std::vector<size_t> filtered_input_axis;
  std::vector<size_t> filtered_output_axis;
  for (size_t i = 0U; i < perm.size(); ++i) {
    auto in_dim_hint = -1;
    GE_ASSERT_TRUE(dims[i].GetHint(in_dim_hint), "Failed to get int value, expr = %s",
                   ge::SymbolicUtils::ToString(dims[i]).c_str());
    if (in_dim_hint != 1){
      filtered_input_axis.push_back(i);
    } else {
      squeeze_axis.push_back(static_cast<int64_t>(i));
    }
    GE_ASSERT_TRUE(perm[i] < static_cast<int64_t>(dims.size()), "Invalid perm index");
    auto out_dim_hint = -1;
    GE_ASSERT_TRUE(dims[perm[i]].GetHint(out_dim_hint), "Failed to get int value, expr = %s",
                   ge::SymbolicUtils::ToString(dims[perm[i]]).c_str());
    if (out_dim_hint != 1) {
      filtered_output_axis.push_back(perm[i]);
    }
  }
  // 比较删除交换的dim1轴后输入输出轴序是否相等
  GE_ASSERT_TRUE(filtered_input_axis.size() == filtered_output_axis.size(),
                 "After filtering out dim1 axes, input and output dimensions are not equal.");
  for (size_t i = 0U; i < filtered_input_axis.size(); ++i) {
    if (filtered_input_axis[i] != filtered_output_axis[i]) {
      return false; // 有效的转置
    }
  }
  return true; // 无效的转置，可以用squeeze优化
}

LoopVar Transpose(const LoopVar &op, const std::vector<ge::Expression> &dims, const std::vector<int64_t> &perm) {
  LoopVar x = op;
  std::vector<int64_t> squeeze_axis;
  if (IsInvalidTranspose(dims, perm, squeeze_axis)) {
    GELOGI("Invalid transpose, can optimize with squeeze.");
    for (auto it = squeeze_axis.rbegin(); it != squeeze_axis.rend(); ++it) {
      x = loop::Squeeze(x, *it);
    }
    return x;
  }
  std::vector<size_t> perm_tmp;
  for (int64_t i : perm) {
    perm_tmp.emplace_back(static_cast<size_t>(i));
  }
  x = loop::Permute(x, perm_tmp);
  return x;
}

LoopVar LoadSeed(const std::string &name, const LoopVar &offset) {
  GE_WARN_ASSERT(offset.IsValid());
  LoopKernel kernel = [name](const std::vector<CseVar> &vars) -> CseVar { return Ops()->LoadSeed(name, vars[0]); };
  PrintKernel readable = [name](const std::vector<std::string> &var_names) -> std::string {
    std::stringstream ss;
    ss << "ops.LoadSeed(" << name.c_str() << ", " << var_names[0] << ")";
    return ss.str();
  };
  std::vector<LoopOpPtr> inputs = {offset.Op()};
  return LoopVar(std::make_shared<PointwiseOp>(kernel, std::move(inputs), "ops.LoadSeed", std::move(readable), UnimplementInferDatatype));
}

LoopVar ReduceThenBroadcast(ReduceType type, const LoopVar &op, int64_t dim) {
  GE_WARN_ASSERT(op.IsValid());
  return LoopVar(std::make_shared<ReduceThenBroadcastOp>(type, op.Op(), dim));
}

LoopVar ToDtypeBitcast(const LoopVar &x, ge::DataType dst_type, ge::DataType src_type) {
  GE_WARN_ASSERT(x.IsValid());
  LoopKernel kernel = [dst_type, src_type](const std::vector<CseVar> &vars) -> CseVar {
    return Ops()->ToDtypeBitcast(vars[0], dst_type, src_type);
  };
  PrintKernel readable = [dst_type, src_type](const std::vector<std::string> &var_names) -> std::string {
    std::stringstream ss;
    ss << "ops.ToDtypeBitcast(" << var_names[0] << ", " << TypeUtils::DataTypeToSerialString(dst_type) << ", "
       << TypeUtils::DataTypeToSerialString(src_type) << ")";
    return ss.str();
  };
  std::vector<LoopOpPtr> inputs = {x.Op()};
  return LoopVar(std::make_shared<PointwiseOp>(kernel, std::move(inputs), "ops.ToDtypeBitcast", std::move(readable),
                                               UnimplementInferDatatype));
}

LoopVar LeakyRelu(const LoopVar &x, float32_t negative_slope) {
  GE_WARN_ASSERT(x.IsValid());
  LoopKernel kernel = [negative_slope](const std::vector<CseVar> &vars) -> CseVar {
    return Ops()->LeakyRelu(vars[0], negative_slope);
  };
  PrintKernel readable = [negative_slope](const std::vector<std::string> &var_names) -> std::string {
    std::stringstream ss;
    ss << "ops.LeakyRelu(" << var_names[0] << ", " << negative_slope << ")";
    return ss.str();
  };

  std::vector<LoopOpPtr> inputs = {x.Op()};
  return LoopVar(std::make_shared<PointwiseOp>(kernel, std::move(inputs), "ops.LeakyRelu", std::move(readable),
                                               ASCIR_INFERDTYPE_2_OP_INFERDTYPE(LeakyRelu)));
}

LoopVar Axpy(const LoopVar &x1, const LoopVar &x2, float32_t alpha) {
  GE_WARN_ASSERT(x1.IsValid());
  GE_WARN_ASSERT(x2.IsValid());
  LoopKernel kernel = [alpha](const std::vector<CseVar> &vars) -> CseVar {
    return Ops()->Axpy(vars[0], vars[1], alpha);
  };
  PrintKernel readable = [alpha](const std::vector<std::string> &var_names) -> std::string {
    std::stringstream ss;
    ss << "ops.Axpy(" << var_names[0] << ", " << var_names[1] << ", " << alpha << ")";
    return ss.str();
  };

  std::vector<LoopOpPtr> inputs = {x1.Op(), x2.Op()};
  return LoopVar(std::make_shared<PointwiseOp>(kernel, std::move(inputs), "ops.Axpy", std::move(readable),
                                               ASCIR_INFERDTYPE_2_OP_INFERDTYPE(Axpy)));
}

LoopVar StoreStridedSlice(const OutDataAnchorPtr &dst, const InDataAnchorPtr &src, const std::vector<Expression> &start,
                          const std::vector<Expression> &stride, const std::vector<Expression> &input_dims,
                          string &not_lowering_reason) {
  if (src->GetPeerOutAnchor() != nullptr) {
    const auto &peer_node = src->GetPeerOutAnchor()->GetOwnerNodeBarePtr();
    const auto &cur_node = src->GetOwnerNodeBarePtr();
    GE_WARN_ASSERT(peer_node, nullptr);
    GE_WARN_ASSERT(cur_node, nullptr);
    GELOGI("peer node name: %s, cur node name: %s", peer_node->GetName().c_str(), cur_node->GetName().c_str());
    GELOGI("peer node type: %s, cur node type: %s", peer_node->GetType().c_str(), cur_node->GetType().c_str());
    int64_t index = peer_node->GetType().find("Slice");
    GELOGI("peer node type find 'Slice' index: %d", index);
    if (index < 0) {
      auto kernelBox = loop::GetKernelBox(src->GetPeerOutAnchor());
      GELOGI("kernelBox name: %s", kernelBox.Name().c_str());
      kernelBox.Realize();
    }
  }
  std::vector<Expression> output_dims;
  std::stringstream ss;
  if (GetBufferShape(dst, output_dims) != GRAPH_SUCCESS) {
    ss << BufferName(dst).c_str() << " has no sym shape";
    not_lowering_reason = ss.str();
    return LoopVar(nullptr);
  }
  if (output_dims.size() != input_dims.size()) {
    ss << "since the input dim " << input_dims.size() << " does not match the output dim " << output_dims.size();
    not_lowering_reason = ss.str();
    return LoopVar(nullptr);
  }
  const auto backend_spec = optimize::BackendSpec::GetInstance();
  if (!output_dims.empty() && !backend_spec->slice_split_spec.slice_fuse_with_end_dim_1) {
    auto index = output_dims.size() - 1;
    if (SymbolicUtils::StaticCheckEq(output_dims[index], Symbol(1)) == TriBool::kTrue) {
      not_lowering_reason = "output end dim is 1, not lowering";
      return LoopVar(nullptr);
    }
  }

  const auto config = StrideSliceParam(start, stride);
  return LoopVar(std::make_shared<StoreStridedSliceOp>(dst, Load(src).Op(), config, output_dims, input_dims));
}

std::vector<LoopVar> StoreSplit(const std::vector<OutDataAnchorPtr> &outputs, const InDataAnchorPtr &src, size_t split_dim, string &not_lowering_reason) {
  if (src->GetPeerOutAnchor() != nullptr) {
    auto kernel_box = loop::GetKernelBox(src->GetPeerOutAnchor());
    GELOGI("kernel box name: %s", kernel_box.Name().c_str());
    kernel_box.Realize();
  }
  vector<ge::Expression> input_dims;
  GE_WARN_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(src, input_dims));
  GE_ASSERT(!input_dims.empty());
  std::vector<std::vector<Expression>> output_dims;
  std::stringstream ss;
  for (auto &out_anchor : outputs) {
    std::vector<Expression> outputx_dims;
    GE_WARN_ASSERT_GRAPH_SUCCESS(loop::GetBufferShape(out_anchor, outputx_dims));
    if (GetBufferShape(out_anchor, outputx_dims) != GRAPH_SUCCESS) {
      ss << BufferName(out_anchor).c_str() << " has no sym shape";
      not_lowering_reason = ss.str();
      return {};
    }
    if (outputx_dims.size() != input_dims.size()) {
      ss << "since the input dim " << input_dims.size() << " does not match the output dim " << output_dims.size();
      not_lowering_reason = ss.str();
      return {};
    }

    if (outputx_dims.size() != input_dims.size()) {
      ss << "since the input dim " << input_dims.size() << " does not match the output dim " << output_dims.size();
      not_lowering_reason = ss.str();
      return {};
    }
    output_dims.emplace_back(outputx_dims);
  }
  char soc_version[128] = {};
  GELOGI("soc_version: %s", soc_version);
  size_t idx = 0U;
  std::vector<LoopVar> ret;
  auto src_op=Load(src).Op();
  for (auto output: outputs) {
    StoreSplitOp::StoreSplitDescBuilder builder;
    auto desc = builder.Output(output).SrcOp(src_op).InputDims(input_dims).OutputDims(output_dims).SplitDim(split_dim).Index(idx++).Build();
    auto op = std::make_shared<StoreSplitOp>(desc);
    auto loop_var = LoopVar(op);
    ret.emplace_back(loop_var);
    SetLoopKernel(output, loop_var).Realize();
  }
  return ret;
}
}  // namespace loop
}  // namespace ge

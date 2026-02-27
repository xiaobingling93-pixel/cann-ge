/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_KERNEL_BOX_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_KERNEL_BOX_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "common/checker.h"
#include "graph/node.h"
#include "loop_common.h"
#include "loop_ops.h"

namespace ge {
namespace loop {
using Edge = std::pair<const ge::OutDataAnchor *, const ge::InDataAnchor *>;
// lazy init extra data, never hold any sharedptr of the node
struct ExtraKernelBoxMeta {
  size_t num_ops = 0U;
  size_t num_loads = 0U;
  size_t num_slices = 0U;
  std::vector<const ge::Node *> ascend_ir_nodes;
  std::set<const ge::OutDataAnchor *> used_ascend_buffers;
  std::set<const ge::OutDataAnchor *> optimized_ascend_buffers;
  std::set<loop::Edge> concrete_edges;  // edges consumed by this fused kernel
  std::string stream_label;
  std::string stream_priority;
  static ExtraKernelBoxMeta &Default() {
    static ExtraKernelBoxMeta kDefaultExtra;
    return kDefaultExtra;
  }
};

struct KernelBoxMeta {
  explicit KernelBoxMeta(LoopVar result) : var(std::move(result)) {
    std::map<const LoopOp *, DataType> op_2_dtype;
    LoopOp::StrictTopoExecute(var.Op().get(), [this, &op_2_dtype](const LoopOp *op) -> graphStatus {
      is_support = this->InferDataType(op, op_2_dtype);
      return GRAPH_SUCCESS;
    });
    if (!var.IsValid()) {
      type = FuseType::kExtern;
    } else {
      if (var.Op()->Type() == "ops.StoreReduction") {
        type = FuseType::kReduction;
      } else if (var.Op()->Type() == "ops.StoreConcat") {
        type = FuseType::kConcat;
      } else if (var.Op()->Type() == "ops.StoreSplit") {
        type = FuseType::kSplit;
      }else if (var.Op()->Type() == "ops.StoreStridedSlice") {
        type = FuseType::kSliceSplit;
      } else if (var.Op()->Type() == "ops.StoreMatMul") {
        type = FuseType::kCube;
      } else {
        type = FuseType::kPointwise;
        LoopOp::Dfs(var.Op().get(), [this](const LoopOp *op) -> graphStatus {
          if (op != nullptr && op->Type() == "ops.StoreStridedSlice") {
            type = FuseType::kSliceSplit;
          } else if (op != nullptr && op->IsReduction()) {
            type = FuseType::kReduction;
          } else if (op != nullptr && op->IsGather()) {
            type = FuseType::kGather;
          } 
          return GRAPH_SUCCESS;
        });
      }
    }
  }
  KernelBoxMeta() : type(FuseType::kExtern) {}
  ExtraKernelBoxMeta &Extra() {
    if (extra != nullptr) {
      return *extra;
    }
    if (!var.IsValid() || type == FuseType::kExtern) {
      return ExtraKernelBoxMeta::Default();
    }
    extra = std::make_shared<ExtraKernelBoxMeta>();
    std::set<ge::NodePtr> seen_nodes;
    LoopOp::StrictTopoExecute(var.Op().get(), [this, &seen_nodes](const LoopOp *op) -> graphStatus {
      if (op->Type() == "ops.Load") {
        const auto load = dynamic_cast<const LoadOp *>(op);
        extra->concrete_edges.insert(load->Edge());
        extra->used_ascend_buffers.insert(load->Buffer());
        extra->num_loads++;
      } else if (op->Type() == "ops.LoadGather") {
        const auto gather = dynamic_cast<const LoadGatherOp *>(op);
        GE_ASSERT_NOTNULL(gather);
        vector<const ge::OutDataAnchor *> gather_buffers = gather->Buffers();
        vector<pair<const ge::OutDataAnchor *, const ge::InDataAnchor *>> gather_edges = gather->Edges();
        extra->concrete_edges.insert(gather_edges.begin(), gather_edges.end());
        extra->used_ascend_buffers.insert(gather_buffers.begin(), gather_buffers.end());
        extra->num_loads++;
      } else if (op->Type() != "ops.Store") {
        extra->num_ops++;
      }
      if (op->Type() == "ops.StoreStridedSlice") {
        extra->num_slices++;
      }
      const auto node = op->GetAscendIrNode();
      if (node != nullptr && seen_nodes.insert(node).second) {
        if (extra->stream_label.empty()) {
          (void)AttrUtils::GetStr(node->GetOpDesc(), public_attr::USER_STREAM_LABEL, extra->stream_label);
          (void)AttrUtils::GetStr(node->GetOpDesc(), public_attr::USER_STREAM_PRIORITY, extra->stream_priority);
        }
        extra->ascend_ir_nodes.push_back(node.get());
      }
      return GRAPH_SUCCESS;
    });

    auto &used_buffers = extra->used_ascend_buffers;
    auto &fused_nodes = extra->ascend_ir_nodes;
    const auto rend = fused_nodes.rend();
    for (auto iter = fused_nodes.rbegin(); iter != rend; ++iter) {
      for (const auto &anchor : (*iter)->GetAllInDataAnchors()) {
        auto anchor_src = anchor->GetPeerOutAnchor();
        if (anchor_src == nullptr) {
          continue;  // optional input with no feed
        }
        if (std::find(iter, rend, anchor_src->GetOwnerNode().get()) != rend) {
          continue;  // anchor src is from inner fused nodes
        }
        if (used_buffers.find(anchor_src.get()) != used_buffers.end()) {
          continue;  // anchor src is used by kernel box
        }
        extra->optimized_ascend_buffers.insert(anchor_src.get());
      }
    }

    return *extra;
  }

  FuseType type;
  LoopVar var;
  bool is_support = true;
  bool realize = false;
  bool realize_persistent = false;

 private:
  std::shared_ptr<ExtraKernelBoxMeta> extra;
  bool InferDataType(const LoopOp *op, std::map<const LoopOp *, DataType> &op_2_dtype) const {
    std::vector<DataType> input_dtypes;
    std::vector<DataType> expect_output_dtypes;
    if (LoopOp::GetLoopOpsDataType(op->Inputs(), op_2_dtype, input_dtypes) != SUCCESS) {
      return false;
    }
    if (!op->InferDataType(input_dtypes, expect_output_dtypes)) {
      GELOGI("Infer Op %s with input dtypes %s got outputs %s", op->Type().c_str(),
             loop::StrJoin(input_dtypes).c_str(), loop::StrJoin(expect_output_dtypes).c_str());
      return false;
    }
    if (!expect_output_dtypes.empty()) {
      op_2_dtype[op] = expect_output_dtypes[0];
    }
    return true;
  }
};

class KernelBox {
 public:
  explicit KernelBox(const ge::OutDataAnchorPtr &target, std::shared_ptr<KernelBoxMeta> meta)
      : target_(target.get()), meta_(std::move(meta)) {
    if (IsReduction()) {
      Realize();
    }
  }
  std::string Name() const {
    if (target_ == nullptr) {
      return "Invalid";
    }
    std::stringstream ss;
    ss << target_->GetOwnerNode()->GetType() << "(\"";
    ss << target_->GetOwnerNode()->GetName() << ":" << target_->GetIdx() << "\")";
    return ss.str();
  }

  std::string Readable() const {
    if (IsExternKernel()) {
      if (target_ == nullptr) {
        return "Invalid";
      }
      std::stringstream ss;
      ss << "ops.ExternKernel." << Name();
      return ss.str();
    }
    return Var().Readable();
  }

  std::string DebugString() {
    std::stringstream ss;
    ss << "Fused(" << Name() << ")";
    std::vector<std::string> node_names;
    for (const auto &node : GetAscendIrNodes()) {
      node_names.push_back(node->GetName() + "(" + node->GetType() + ")");
    }
    ss << StrJoin(node_names);
    return ss.str();
  }

  bool IsReduction() const {
    return meta_ != nullptr && meta_->type == FuseType::kReduction;
  }

  bool IsGather() const {
    return meta_ != nullptr && meta_->type == FuseType::kGather;
  }

  bool IsExternKernel() const {
    return meta_ == nullptr || meta_->type == FuseType::kExtern;
  }

  bool IsPointwise() const {
    return meta_ != nullptr && meta_->type == FuseType::kPointwise;
  }

  bool IsCube() const {
    return (meta_ != nullptr) && (meta_->type == FuseType::kCube);
  }

  bool IsSlice() const {
    return (meta_ != nullptr) && (meta_->type == FuseType::kSliceSplit);
  }

  bool IsSliceOnly() const {
    if (!IsSlice()) {
      return false;
    }
    bool is_slice_only = true;
    static std::set<std::string> slice_ascir_ops = {"ops.Load", "ops.Store", "ops.StoreStridedSlice"};
    LoopOp::StrictTopoExecute(meta_->var.Op().get(), [&is_slice_only](const LoopOp *op) -> graphStatus {
      if (slice_ascir_ops.count(op->Type()) == 0) {
        is_slice_only = false;
        GELOGD("this kernelbox is not slice only, contains non-slice ascir op type: %s", op->Type().c_str());
      }
      return GRAPH_SUCCESS;
    });
    return is_slice_only;
  }

  bool IsSupport() const {
    return meta_ != nullptr && meta_->is_support;
  }
  FuseType Type() const {
    if (meta_ == nullptr) {
      return FuseType::kExtern;
    }
    return meta_->type;
  }

  KernelBox &Realize(bool persistent = true) {
    if (IsExternKernel()) {  // Do nothing if realize on an extern kernel
      return *this;
    }
    if (meta_ != nullptr) {
      meta_->realize = true;
      meta_->realize_persistent = persistent;
    }
    return *this;
  }

  bool IsRealized() const {
    return meta_ != nullptr && meta_->realize;
  }

  bool IsRealizedPersistent() const {
    return meta_ != nullptr && meta_->realize_persistent;
  }

  LoopVar Load(const ge::InDataAnchorPtr &dst) const {
    if (IsRealizedPersistent() || IsExternKernel()) {
      return LoopVar(std::make_shared<LoadOp>(target_, dst.get()));
    }
    return Var().Clone();
  }

  const ge::OutDataAnchor *TargetBuffer() const {
    return target_;
  }

  const std::set<loop::Edge> &GetConcreteEdges() {
    return GetExtraMeta().concrete_edges;
  }

  const std::vector<const ge::Node *> &GetAscendIrNodes() {
    return GetExtraMeta().ascend_ir_nodes;
  }

  const std::set<const ge::OutDataAnchor *> &GetInputAscendBuffers() {
    return GetExtraMeta().used_ascend_buffers;
  }

  const std::set<const ge::OutDataAnchor *> &GetOptimizedInputAscendBuffers() {
    return GetExtraMeta().optimized_ascend_buffers;
  }

  size_t NumOps() {
    return GetExtraMeta().num_ops;
  }

  size_t NumAscendNodes() {
    return GetAscendIrNodes().size();
  }

  size_t NumLoads() {
    return GetExtraMeta().num_loads;
  }

  size_t NumSlices() {
    return GetExtraMeta().num_slices;
  }  

  std::string StreamLabel() {
    return GetExtraMeta().stream_label;
  }

  std::string StreamPriority() {
    return GetExtraMeta().stream_priority;
  }

  template <typename T, typename... Args>
  auto Realize(const std::string &graph_name = "graph", Args... rest) -> std::shared_ptr<T> {
    GE_WARN_ASSERT(Var().IsValid());
    auto graph = std::make_shared<T>(graph_name.c_str(), rest...);
    OpsGuard guarder(graph);
    GE_WARN_ASSERT_GRAPH_SUCCESS(Var().Op()->Realize());
    return graph;
  }

 private:
  LoopVar Var() const {
    if (meta_ == nullptr) {
      return LoopVar();
    }
    return meta_->var;
  }

  const ExtraKernelBoxMeta &GetExtraMeta() {
    if (meta_ == nullptr) {
      return ExtraKernelBoxMeta::Default();
    }
    return meta_->Extra();
  }

  const ge::OutDataAnchor *target_;
  std::shared_ptr<KernelBoxMeta> meta_;
};
}  // namespace loop
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_KERNEL_BOX_H_

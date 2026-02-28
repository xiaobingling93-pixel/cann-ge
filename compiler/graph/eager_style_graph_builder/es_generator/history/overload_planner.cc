/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "overload_planner.h"

#include <algorithm>
#include <list>
#include <sstream>
#include <vector>

#include "attr_type_traits.h"
#include "ambiguity_checker.h"
#include "warning_formatter.h"
#include "../utils.h"

namespace ge {
namespace es {
namespace history {
namespace {
bool ContainsName(const std::vector<std::string> *names, const std::string &name) {
  if (names == nullptr) {
    return false;
  }
  for (const auto &candidate: *names) {
    if (candidate == name) {
      return true;
    }
  }
  return false;
}

bool IsInputParamKind(const ParamCxxKind kind) {
  return kind == ParamCxxKind::kEsTensorLikeRef || kind == ParamCxxKind::kTensorHolderRef ||
      kind == ParamCxxKind::kTensorHoldersVecRef;
}

bool AreAllInputsOptional(const std::vector<IrInput> &inputs) {
  if (inputs.empty()) {
    return false;
  }
  return std::all_of(inputs.begin(),
                     inputs.end(),
                     [](const IrInput &input) {
                       return input.type == kIrInputOptional;
                     });
}

std::vector<std::string> CollectInputNames(const IrOpProto &proto) {
  std::vector<std::string> names;
  names.reserve(proto.inputs.size());
  for (const auto &input: proto.inputs) {
    names.emplace_back(input.name);
  }
  return names;
}

bool IsSameInput(const IrInput &lhs, const IrInput &rhs) {
  return lhs.name == rhs.name && lhs.type == rhs.type && lhs.dtype == rhs.dtype;
}

bool IsSameOutput(const IrOutput &lhs, const IrOutput &rhs) {
  return lhs.name == rhs.name && lhs.type == rhs.type && lhs.dtype == rhs.dtype;
}

bool IsSameAttr(const IrAttr &lhs, const IrAttr &rhs) {
  return lhs.name == rhs.name && lhs.av_type == rhs.av_type &&
      lhs.required == rhs.required && lhs.default_value == rhs.default_value;
}

bool IsSameSubgraph(const IrSubgraph &lhs, const IrSubgraph &rhs) {
  return lhs.name == rhs.name && lhs.type == rhs.type;
}

template <typename T, typename EqualFn>
bool IsSameVector(const std::vector<T> &lhs, const std::vector<T> &rhs, EqualFn equal_fn) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0U; i < lhs.size(); ++i) {
    if (!equal_fn(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}

bool IsSameProto(const IrOpProto &lhs, const IrOpProto &rhs) {
  return lhs.op_type == rhs.op_type &&
      IsSameVector(lhs.inputs, rhs.inputs, IsSameInput) &&
      IsSameVector(lhs.outputs, rhs.outputs, IsSameOutput) &&
      IsSameVector(lhs.attrs, rhs.attrs, IsSameAttr) &&
      IsSameVector(lhs.subgraphs, rhs.subgraphs, IsSameSubgraph);
}

void AppendUniqueStrings(std::vector<std::string> &dst, const std::vector<std::string> &src) {
  for (const auto &item: src) {
    if (std::find(dst.begin(), dst.end(), item) != dst.end()) {
      continue;
    }
    dst.emplace_back(item);
  }
}

std::vector<std::string> CollectRemovedInputs(const IrOpProto &current, const IrOpProto &baseline) {
  std::vector<std::string> removed_inputs;
  if (baseline.inputs.size() >= current.inputs.size()) {
    return removed_inputs;
  }
  removed_inputs.reserve(current.inputs.size() - baseline.inputs.size());
  for (size_t i = baseline.inputs.size(); i < current.inputs.size(); ++i) {
    removed_inputs.emplace_back(current.inputs[i].name);
  }
  return removed_inputs;
}

void EraseInputParamsByName(Signature &sig, const std::vector<std::string> &input_names) {
  sig.params.erase(std::remove_if(sig.params.begin(),
                                  sig.params.end(),
                                  [&input_names](const Param &param) {
                                    if (!IsInputParamKind(param.kind)) {
                                      return false;
                                    }
                                    for (const auto &name: input_names) {
                                      if (param.ir_name == name) {
                                        return true;
                                      }
                                    }
                                    return false;
                                  }),
                   sig.params.end());
}

std::string JoinStrings(const std::vector<std::string> &items) {
  std::stringstream ss;
  bool first = true;
  for (const auto &item: items) {
    if (first) {
      first = false;
    } else {
      ss << ", ";
    }
    ss << item;
  }
  return ss.str();
}

std::string BuildWarningContext(const IrOpProto &current, const std::vector<std::string> &new_inputs) {
  std::string context;
  if (!current.op_type.empty()) {
    context = "op " + current.op_type;
  }
  if (new_inputs.empty()) {
    return context;
  }
  const std::string inputs = "new optional inputs [" + JoinStrings(new_inputs) + "]";
  if (context.empty()) {
    return inputs;
  }
  return context + ", " + inputs;
}

std::string BuildWarningDetail(const std::string &context, const std::string &reason) {
  if (reason.empty()) {
    return context;
  }
  if (context.empty()) {
    return reason;
  }
  return context + "; " + reason;
}

void AppendPlanWarning(std::vector<Warning> *warnings,
                       const WarningCode code,
                       const std::string &context,
                       const std::string &reason = "") {
  warnings->emplace_back(Warning{code, BuildWarningDetail(context, reason)});
}

void AppendWarningsUnique(std::vector<Warning> &dst, const std::vector<Warning> &src) {
  for (const auto &warning : src) {
    const bool exists = std::any_of(dst.begin(),
                                    dst.end(),
                                    [&warning](const Warning &item) {
                                      return item.code == warning.code && item.detail == warning.detail;
                                    });
    if (!exists) {
      dst.emplace_back(warning);
    }
  }
}
} // namespace

/**
 *规划总入口：
  1. 无历史时直接生成 A0（仅最新签名）；
  2. 有历史时先识别“需要保留旧函数形态”的版本分界点（即重载边界）；
  3. 对可重载场景按 Try0 -> A1 -> A2 逐级尝试；
  4. 若仍有二义性，则回退 A0 并输出告警。
*/

OverloadPlan OverloadPlanner::Plan(const IrOpProto &current, const HistoryContext &history) const {
  OverloadPlan plan;
  if (history.proto_chain.empty()) {
    plan.signatures.emplace_back(BuildA0Signature(current, &plan.warnings));
    return plan;
  }

  const auto versions = BuildVersionChain(current, history);
  const auto boundary_result = CollectBoundaries(versions);
  if (!boundary_result.success) {
    const std::string warning_context = BuildWarningContext(current, boundary_result.all_new_inputs);
    AppendPlanWarning(&plan.warnings, WarningCode::kFallbackToA0, warning_context, boundary_result.fail_reason);
    plan.signatures.emplace_back(BuildA0Signature(current, &plan.warnings));
    return plan;
  }
  if (boundary_result.boundaries.empty()) {
    plan.signatures.emplace_back(BuildA0Signature(current, &plan.warnings));
    return plan;
  }

  const std::string warning_context = BuildWarningContext(current, boundary_result.all_new_inputs);
  if (TryResolveModeEscalation(current, versions, boundary_result.boundaries, warning_context, plan)) {
    return plan;
  }

  AppendPlanWarning(&plan.warnings, WarningCode::kFallbackToA0, warning_context, "ambiguity remains after A2");
  plan.signatures.emplace_back(BuildA0Signature(current, &plan.warnings));
  return plan;
}

OverloadPlan PlanCppOverloads(const IrOpProto &current,
                              const HistoryContext &history,
                              std::vector<std::string> &warnings) {
  OverloadPlanner planner;
  auto plan = planner.Plan(current, history);
  for (const auto &warning : plan.warnings) {
    warnings.emplace_back(FormatWarning(warning));
  }
  return plan;
}

std::vector<const IrOpProto *> OverloadPlanner::BuildVersionChain(const IrOpProto &current,
                                                                  const HistoryContext &history) const {
  std::vector<const IrOpProto *> raw_versions;
  raw_versions.reserve(history.proto_chain.size() + 1U);
  for (const auto &proto: history.proto_chain) {
    raw_versions.push_back(&proto);
  }
  raw_versions.push_back(&current);

  // 历史库按“全量算子快照”归档时，同一算子在多个版本里可能完全一致。
  // 为避免后续重载规划把相邻重复版本当作独立 baseline 参与组合，导致重复签名/误判二义，
  // 在规划入口仅折叠“相邻等价原型”，保留真实演进拐点。
  std::vector<const IrOpProto *> versions;
  versions.reserve(raw_versions.size());
  for (const auto *proto : raw_versions) {
    if (!versions.empty() && IsSameProto(*versions.back(), *proto)) {
      continue;
    }
    versions.push_back(proto);
  }
  return versions;
}

// 从版本链中收集“需要保留旧重载”的分界点（重载边界）。
// 一旦出现不可兼容变化（例如非尾部追加、required attr 新增等），直接失败返回。
OverloadPlanner::BoundaryCollectResult OverloadPlanner::CollectBoundaries(
  const std::vector<const IrOpProto *> &versions) const {
  BoundaryCollectResult result;
  for (size_t i = 1U; i < versions.size(); ++i) {
    const auto diff = AnalyzeDiff(*versions[i], *versions[i - 1U]);
    if (!diff.compatible) {
      result.success = false;
      result.fail_reason = "incompatible schema change in history chain at step " + std::to_string(i) +
          (diff.detail.empty() ? "" : (": " + diff.detail));
      return result;
    }
    if (diff.can_safe_merge) {
      continue;
    }
    if (diff.new_inputs.empty()) {
      result.success = false;
      result.fail_reason = "incompatible schema change without new optional inputs at step " + std::to_string(i) +
          (diff.detail.empty() ? "" : (": " + diff.detail));
      return result;
    }
    result.boundaries.emplace_back(BoundaryInfo{i, diff.new_inputs, diff.baseline_all_inputs_optional});
    AppendUniqueStrings(result.all_new_inputs, diff.new_inputs);
  }
  return result;
}

std::vector<size_t> OverloadPlanner::CollectBaselineIndices(const size_t version_count,
                                                            const std::vector<BoundaryInfo> &boundaries) const {
  // baseline 下标表示“要保留哪一代函数外形”：
  // - 先放入最新版本（current），确保最新接口一定会生成；
  // - 对每个边界，同时保留边界两侧版本，避免某一代调用形态丢失。
  std::vector<size_t> indices;
  indices.reserve(boundaries.size() * 2U + 1U);
  indices.emplace_back(version_count - 1U);
  for (const auto &boundary: boundaries) {
    indices.emplace_back(boundary.step_index - 1U);
    indices.emplace_back(boundary.step_index);
  }
  std::sort(indices.begin(), indices.end());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
  return indices;
}

bool OverloadPlanner::TryAdoptModeSignatures(const IrOpProto &current,
                                             const std::vector<const IrOpProto *> &versions,
                                             const std::vector<BoundaryInfo> &boundaries,
                                             const MultiBaselineMode mode,
                                             std::vector<Warning> *warnings,
                                             std::vector<Signature> &accepted_signatures) const {
  std::vector<Warning> mode_warnings;
  auto mode_signatures = BuildMultiBaselineSignatures(current, versions, boundaries, mode, &mode_warnings);
  if (HasPotentialAmbiguity(mode_signatures)) {
    return false;
  }
  accepted_signatures = std::move(mode_signatures);
  if (warnings != nullptr) {
    AppendWarningsUnique(*warnings, mode_warnings);
  }
  return true;
}

// 模式升级决策：
// - Try0：最小约束，优先可读性；
// - A1：Try0 二义时，强制 required + guard；
// - A2：A1 仍二义时，将新增输入切为 TensorHolder + guard。
// 该函数只做“升级尝试”，最终回退由 Plan 统一处理。
bool OverloadPlanner::TryResolveModeEscalation(const IrOpProto &current,
                                               const std::vector<const IrOpProto *> &versions,
                                               const std::vector<BoundaryInfo> &boundaries,
                                               const std::string &warning_context,
                                               OverloadPlan &plan) const {
  if (TryAdoptModeSignatures(current, versions, boundaries, MultiBaselineMode::kTry0, &plan.warnings, plan.signatures)) {
    return true;
  }
  AppendPlanWarning(&plan.warnings, WarningCode::kUpgradeToA1, warning_context);
  if (TryAdoptModeSignatures(current, versions, boundaries, MultiBaselineMode::kA1, &plan.warnings, plan.signatures)) {
    return true;
  }
  AppendPlanWarning(&plan.warnings, WarningCode::kUpgradeToA2, warning_context);
  return TryAdoptModeSignatures(current, versions, boundaries, MultiBaselineMode::kA2, &plan.warnings,
                                plan.signatures);
}

// 批量生成多 baseline 的签名集合：
// 1. 每个 baseline 先生成一条主体签名；
// 2. 再按 boundary 为新增输入追加 guard 删除签名（A1/A2 才需要）。
std::vector<Signature> OverloadPlanner::BuildMultiBaselineSignatures(const IrOpProto &current,
                                                                     const std::vector<const IrOpProto *> &versions,
                                                                     const std::vector<BoundaryInfo> &boundaries,
                                                                     const MultiBaselineMode mode,
                                                                     std::vector<Warning> *warnings) const {
  const auto baseline_indices = CollectBaselineIndices(versions.size(), boundaries);
  size_t guard_count = 0U;
  for (const auto &boundary: boundaries) {
    guard_count += boundary.new_inputs.size();
  }

  std::vector<Signature> signatures;
  signatures.reserve(baseline_indices.size() + (mode == MultiBaselineMode::kTry0 ? 0U : guard_count));
  std::vector<Signature> baseline_signatures(versions.size());
  std::vector<bool> has_baseline_signature(versions.size(), false);

  for (const auto baseline_index: baseline_indices) {
    const auto &baseline = *versions[baseline_index];
    const auto plan_input = BuildBaselinePlanInput(current, baseline, boundaries, baseline_index, mode);
    const auto *required_ptr = plan_input.required_inputs.empty() ? nullptr : &plan_input.required_inputs;
    const auto *tensor_holder_ptr =
        plan_input.tensor_holder_inputs.empty() ? nullptr : &plan_input.tensor_holder_inputs;
    const auto *removed_ptr = plan_input.removed_inputs.empty() ? nullptr : &plan_input.removed_inputs;
    auto signature = BuildSignature(current,
                                    required_ptr,
                                    tensor_holder_ptr,
                                    removed_ptr,
                                    warnings);
    EraseInputParamsByName(signature, plan_input.removed_inputs);

    signatures.emplace_back(signature);
    baseline_signatures[baseline_index] = signature;
    has_baseline_signature[baseline_index] = true;
  }

  AppendBoundaryGuards(signatures, baseline_signatures, has_baseline_signature, boundaries, mode);
  return signatures;
}

// 计算某个 baseline 在当前 mode 下的参数策略：
// - baseline：决定“目标函数外形”停在哪个历史版本（通过 removed_inputs 裁剪 current）。
// - mode：决定“同一个外形”的消歧强度（Try0/A1/A2）。
//   例：baseline=v2、current=v4，若 v3/v4 新增了可选输入：
//   1) Try0: 仅裁剪到 v2 外形，不强制 required；
//   2) A1:   在 v2 外形上将这些新增输入强制 required；
//   3) A2:   在 A1 基础上把这些新增输入改为 TensorHolder。
// - removed_inputs: current 相对 baseline 多出来、需要从 legacy 形态裁剪掉的输入。
// - required_inputs: A1/A2 场景下需要被强制 required 的输入集合。
// - tensor_holder_inputs: A2 场景下需要从 TensorLike 切到 TensorHolder 的新增输入。
// 注意：当历史输入全可选时，为稳定 owner_builder 位置，会把 baseline 全输入强制 required。
OverloadPlanner::BaselinePlanInput OverloadPlanner::BuildBaselinePlanInput(
  const IrOpProto &current,
  const IrOpProto &baseline,
  const std::vector<BoundaryInfo> &boundaries,
  const size_t baseline_index,
  const MultiBaselineMode mode) const {
  BaselinePlanInput plan_input;
  plan_input.removed_inputs = CollectRemovedInputs(current, baseline);
  if (mode == MultiBaselineMode::kTry0) {
    // Try0 只做“外形还原”，不引入额外约束。
    return plan_input;
  }

  std::vector<std::string> upgraded_inputs;
  bool force_all_inputs_required = false;
  for (const auto &boundary: boundaries) {
    // 仅吸收“位于该 baseline 及其之前”的边界信息；
    // baseline 之后才出现的新增输入，不应影响该 baseline 的约束策略。
    if (boundary.step_index > baseline_index) {
      continue;
    }
    AppendUniqueStrings(upgraded_inputs, boundary.new_inputs);
    if (boundary.prev_all_inputs_optional) {
      // 当旧版本输入全可选时，位置调用可能写成 (..., owner_builder)。
      // 新增可选输入会改变 owner_builder 的位置语义，因此需升级为“全输入 required”。
      force_all_inputs_required = true;
    }
  }

  if (force_all_inputs_required) {
    plan_input.required_inputs = CollectInputNames(baseline);
  } else {
    plan_input.required_inputs = upgraded_inputs;
  }
  if (mode == MultiBaselineMode::kA2) {
    plan_input.tensor_holder_inputs = std::move(upgraded_inputs);
  }
  return plan_input;
}

void OverloadPlanner::AppendBoundaryGuards(std::vector<Signature> &signatures,
                                           const std::vector<Signature> &baseline_signatures,
                                           const std::vector<bool> &has_baseline_signature,
                                           const std::vector<BoundaryInfo> &boundaries,
                                           const MultiBaselineMode mode) const {
  if (mode == MultiBaselineMode::kTry0) {
    return;
  }
  const bool tensor_holder_mode = (mode == MultiBaselineMode::kA2);
  for (const auto &boundary: boundaries) {
    if (!has_baseline_signature[boundary.step_index]) {
      continue;
    }
    const auto &guard_base = baseline_signatures[boundary.step_index];
    for (const auto &input_name: boundary.new_inputs) {
      signatures.emplace_back(BuildNullptrGuardSignature(guard_base, input_name, tensor_holder_mode));
    }
  }
}

bool OverloadPlanner::DiffInputs(const IrOpProto &current,
                                 const IrOpProto &baseline,
                                 DiffResult &diff) const {
  if (current.inputs.size() < baseline.inputs.size()) {
    diff.detail = "inputs shrink from " + std::to_string(baseline.inputs.size()) +
        " to " + std::to_string(current.inputs.size());
    return false;
  }
  for (size_t i = 0U; i < baseline.inputs.size(); ++i) {
    // 当前策略：兼容窗口内要求输入类型严格一致（含 required/optional 标记一致）。
    // TTODO(es_compat): 若后续需要支持“required -> optional”兼容，
    // 需要在这里放宽 input.type 比较，并同步调整：
    // 1) EvaluateMergeSafety（避免 owner_builder 位置/默认值带来的语义漂移）
    // 2) BuildBaselinePlanInput（required 强制策略）
    // 3) 相关 UT（含二义性与历史链多版本场景）
    if (current.inputs[i].name != baseline.inputs[i].name || current.inputs[i].type != baseline.inputs[i].type) {
      diff.detail = "input mismatch at index " + std::to_string(i);
      return false;
    }
  }
  for (size_t i = baseline.inputs.size(); i < current.inputs.size(); ++i) {
    if (current.inputs[i].type != kIrInputOptional) {
      diff.detail = "new input '" + current.inputs[i].name + "' is not optional";
      return false;
    }
    diff.new_inputs.emplace_back(current.inputs[i].name);
  }
  return true;
}

bool OverloadPlanner::DiffAttrs(const IrOpProto &current,
                                const IrOpProto &baseline,
                                DiffResult &diff) const {
  if (current.attrs.size() < baseline.attrs.size()) {
    diff.detail = "attrs shrink from " + std::to_string(baseline.attrs.size()) +
        " to " + std::to_string(current.attrs.size());
    return false;
  }
  for (size_t i = 0U; i < baseline.attrs.size(); ++i) {
    // 当前策略：历史属性 required 标记需保持不变。
    // TTODO(es_compat): 若后续需要支持“required -> optional”兼容，
    // 需要在这里放宽 required 比较，并同步验证默认值/调用二义性影响。
    if (current.attrs[i].name != baseline.attrs[i].name ||
        current.attrs[i].av_type != baseline.attrs[i].av_type ||
        current.attrs[i].required != baseline.attrs[i].required) {
      diff.detail = "attr mismatch at index " + std::to_string(i);
      return false;
    }
  }
  for (size_t i = baseline.attrs.size(); i < current.attrs.size(); ++i) {
    if (current.attrs[i].required) {
      diff.detail = "new attr '" + current.attrs[i].name + "' is required";
      return false;
    }
    const auto parsed = AttrTypeTraits::ParseDefaultExpr(current.attrs[i].av_type, current.attrs[i].default_value);
    if (!parsed.success) {
      diff.detail = "new optional attr '" + current.attrs[i].name + "' has invalid default_value: " + parsed.error;
      return false;
    }
  }
  return true;
}

bool OverloadPlanner::DiffOutputs(const IrOpProto &current,
                                  const IrOpProto &baseline,
                                  DiffResult &diff) const {
  if (current.outputs.size() != baseline.outputs.size()) {
    diff.detail = "outputs changed";
    return false;
  }
  for (size_t i = 0U; i < current.outputs.size(); ++i) {
    if (current.outputs[i].name != baseline.outputs[i].name || current.outputs[i].type != baseline.outputs[i].type) {
      diff.detail = "output mismatch at index " + std::to_string(i);
      return false;
    }
  }
  return true;
}

bool OverloadPlanner::DiffSubgraphs(const IrOpProto &current,
                                    const IrOpProto &baseline,
                                    DiffResult &diff) const {
  if (current.subgraphs.size() != baseline.subgraphs.size()) {
    diff.detail = "subgraphs changed";
    return false;
  }
  for (size_t i = 0U; i < current.subgraphs.size(); ++i) {
    if (current.subgraphs[i].name != baseline.subgraphs[i].name ||
        current.subgraphs[i].type != baseline.subgraphs[i].type) {
      diff.detail = "subgraph mismatch at index " + std::to_string(i);
      return false;
    }
  }
  return true;
}

void OverloadPlanner::EvaluateMergeSafety(const IrOpProto &current,
                                          const IrOpProto &baseline,
                                          DiffResult &diff) const {
  diff.can_safe_merge = true;
  if (!diff.new_inputs.empty() && diff.baseline_all_inputs_optional) {
    diff.can_safe_merge = false;
    diff.detail = "all historical inputs are optional, owner_builder position would shift";
    return;
  }
  if (!diff.new_inputs.empty() && !baseline.attrs.empty()) {
    diff.can_safe_merge = false;
    diff.detail = "new optional inputs are appended after historical attrs, keep legacy overload";
    return;
  }
  if (!diff.new_inputs.empty() && HasScalarLiteralAttrRisk(current)) {
    diff.can_safe_merge = false;
    diff.detail = "scalar-literal attr risk requires overload disambiguation";
  }
}

OverloadPlanner::DiffResult OverloadPlanner::AnalyzeDiff(const IrOpProto &current,
                                                         const IrOpProto &baseline) const {
  DiffResult diff;
  diff.baseline_all_inputs_optional = AreAllInputsOptional(baseline.inputs);
  if (!DiffInputs(current, baseline, diff)) {
    return diff;
  }
  if (!DiffAttrs(current, baseline, diff)) {
    return diff;
  }
  if (!DiffOutputs(current, baseline, diff)) {
    return diff;
  }
  if (!DiffSubgraphs(current, baseline, diff)) {
    return diff;
  }
  diff.compatible = true;
  EvaluateMergeSafety(current, baseline, diff);
  return diff;
}

bool OverloadPlanner::HasPotentialAmbiguity(const std::vector<Signature> &signatures) const {
  for (size_t i = 0U; i < signatures.size(); ++i) {
    if (signatures[i].is_deleted) {
      continue;
    }
    for (size_t j = i + 1U; j < signatures.size(); ++j) {
      if (signatures[j].is_deleted) {
        continue;
      }
      if (AmbiguityChecker::HasPotentialAmbiguityByTypicalArgs(signatures[i], signatures[j])) {
        return true;
      }
    }
  }
  return false;
}

bool OverloadPlanner::HasScalarLiteralAttrRisk(const IrOpProto &proto) const {
  for (const auto &attr: proto.attrs) {
    if (attr.required) {
      continue;
    }
    if (attr.av_type == "Int" || attr.av_type == "Float" || attr.av_type == "Bool") {
      return true;
    }
  }
  return false;
}

Signature OverloadPlanner::BuildA0Signature(const IrOpProto &current, std::vector<Warning> *warnings) const {
  return BuildSignature(current, nullptr, nullptr, nullptr, warnings);
}

Signature OverloadPlanner::BuildSignature(const IrOpProto &current,
                                          const std::vector<std::string> *force_required_inputs,
                                          const std::vector<std::string> *force_tensor_holder_inputs,
                                          const std::vector<std::string> *removed_inputs,
                                          std::vector<Warning> *warnings) const {
  Signature sig;
  const auto policy = BuildDefaultValuePolicy(current, force_required_inputs);
  AppendInputParams(current, policy, force_tensor_holder_inputs, sig);
  AppendOwnerBuilderParam(current, policy, removed_inputs, sig);
  AppendDynamicOutputParams(current, sig);
  AppendSubgraphParams(current, sig);
  AppendAttrParams(current, sig, warnings);
  return sig;
}

Signature OverloadPlanner::BuildNullptrGuardSignature(const Signature &sig,
                                                      const std::string &input_name,
                                                      bool tensor_holder_mode) const {
  Signature guard = sig;
  guard.is_deleted = true;
  guard.is_deprecated = true;
  if (tensor_holder_mode) {
    guard.deprecate_msg =
        "New input " + input_name + " uses TensorHolder for disambiguation. "
        "Use legacy overload when disconnected, or use CreateScalar/CreateVector/CreateConst for constants.";
  } else {
    guard.deprecate_msg =
        "New input " + input_name + " cannot use nullptr in this overload. "
        "Use the legacy overload when this input is disconnected.";
  }

  const int index = FindInputParamIndex(guard, input_name);
  if (index >= 0 && static_cast<size_t>(index) < guard.params.size()) {
    guard.params[index].kind = ParamCxxKind::kNullptrT;
    guard.params[index].role = ParamRole::kInput;
    guard.params[index].ir_name = input_name;
    guard.params[index].name = InName(input_name);
    guard.params[index].has_default = false;
    guard.params[index].default_expr.clear();
  } else {
    guard.params.clear();
    Param param;
    param.kind = ParamCxxKind::kNullptrT;
    param.role = ParamRole::kInput;
    param.ir_name = input_name;
    param.name = InName(input_name);
    guard.params.push_back(std::move(param));
  }
  return guard;
}

int OverloadPlanner::FindInputParamIndex(const Signature &sig, const std::string &input_name) const {
  for (size_t i = 0U; i < sig.params.size(); ++i) {
    if (sig.params[i].ir_name == input_name && IsInputParamKind(sig.params[i].kind)) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

void OverloadPlanner::AppendInputParams(const IrOpProto &current,
                                        const DefaultValuePolicy &policy,
                                        const std::vector<std::string> *force_tensor_holder_inputs,
                                        Signature &sig) const {
  const bool use_tensor_like = UseTensorLikeInputs(current);
  for (const auto &input: current.inputs) {
    Param param;
    if (input.type == kIrInputDynamic) {
      param.kind = ParamCxxKind::kTensorHoldersVecRef;
      param.role = ParamRole::kInput;
      param.ir_name = input.name;
      param.name = InName(input.name);
    } else {
      const bool use_tensor_holder = ContainsName(force_tensor_holder_inputs, input.name);
      param.kind = use_tensor_holder
                     ? ParamCxxKind::kTensorHolderRef
                     : (use_tensor_like ? ParamCxxKind::kEsTensorLikeRef : ParamCxxKind::kTensorHolderRef);
      param.role = ParamRole::kInput;
      param.ir_name = input.name;
      param.name = InName(input.name);
      if (input.type == kIrInputOptional && policy.HasInputDefault(input.name)) {
        param.has_default = true;
        param.default_expr = "=nullptr";
      }
    }
    sig.params.emplace_back(std::move(param));
  }
}

void OverloadPlanner::AppendDynamicOutputParams(const IrOpProto &current, Signature &sig) const {
  for (const auto &output: current.outputs) {
    if (output.type != kIrOutputDynamic) {
      continue;
    }
    Param param;
    param.kind = ParamCxxKind::kInt64;
    param.role = ParamRole::kDynamicOutputNum;
    param.ir_name = output.name;
    param.name = DynamicOutputParamName(current, output.name);
    sig.params.emplace_back(std::move(param));
  }
}

void OverloadPlanner::AppendSubgraphParams(const IrOpProto &current, Signature &sig) const {
  for (const auto &subgraph: current.subgraphs) {
    Param param;
    if (subgraph.type == kStatic) {
      param.kind = ParamCxxKind::kGraphUniquePtr;
    } else {
      param.kind = ParamCxxKind::kGraphsVec;
    }
    param.role = ParamRole::kSubgraph;
    param.ir_name = subgraph.name;
    param.name = SubgraphName(subgraph.name);
    sig.params.emplace_back(std::move(param));
  }
}

void OverloadPlanner::AppendOwnerBuilderParam(const IrOpProto &current,
                                              const DefaultValuePolicy &policy,
                                              const std::vector<std::string> *removed_inputs,
                                              Signature &sig) const {
  if (current.inputs.empty()) {
    Param param;
    param.kind = ParamCxxKind::kGraphBuilderRef;
    param.role = ParamRole::kOwnerBuilder;
    param.ir_name = "owner_builder";
    param.name = "owner_builder";
    sig.params.emplace_back(std::move(param));
    return;
  }
  bool all_optional = true;
  for (const auto &in: current.inputs) {
    if (in.type == kIrInputRequired || in.type == kIrInputDynamic) {
      all_optional = false;
      break;
    }
  }
  if (!all_optional) {
    return;
  }
  bool has_effectively_optional_input = false;
  for (const auto &in: current.inputs) {
    if (ContainsName(removed_inputs, in.name)) {
      continue;
    }
    if (policy.IsInputEffectivelyOptional(in)) {
      has_effectively_optional_input = true;
      break;
    }
  }
  if (!has_effectively_optional_input) {
    return;
  }
  Param param;
  param.kind = ParamCxxKind::kGraphBuilderPtr;
  param.role = ParamRole::kOwnerBuilder;
  param.ir_name = "owner_builder";
  param.name = "owner_builder";
  if (policy.HasAnyInputDefault()) {
    param.has_default = true;
    param.default_expr = "=nullptr";
  }
  sig.params.emplace_back(std::move(param));
}

void OverloadPlanner::AppendAttrParams(const IrOpProto &current,
                                       Signature &sig,
                                       std::vector<Warning> *warnings) const {
  std::list<Param> required_params;
  std::list<Param> optional_params;
  for (const auto &attr: current.attrs) {
    Param param;
    bool supported = AttrTypeTraits::TryGetParamKindByHistoryType(attr.av_type, param.kind);
    if (!supported) {
      param.kind = ParamCxxKind::kNullptrT;
    }
    if (!supported && warnings != nullptr) {
      std::string detail = current.op_type.empty() ? "" : ("op " + current.op_type + ", ");
      detail += "attr '" + attr.name + "' type '" + attr.av_type + "'";
      warnings->emplace_back(Warning{WarningCode::kUnsupportedAttrType, detail});
    }
    param.role = ParamRole::kAttr;
    param.ir_name = attr.name;
    param.name = AttrName(attr.name, current.inputs);
    if (!attr.required) {
      const auto parsed = AttrTypeTraits::ParseDefaultExpr(attr.av_type, attr.default_value);
      if (parsed.success) {
        param.has_default = true;
        param.default_expr = parsed.default_expr;
      } else if (!attr.default_value.empty() && warnings != nullptr) {
        std::string detail = current.op_type.empty() ? "" : ("op " + current.op_type + ", ");
        detail += "attr '" + attr.name + "' type '" + attr.av_type + "'; " + parsed.error;
        warnings->emplace_back(Warning{WarningCode::kInvalidAttrDefaultValue, detail});
      }
      optional_params.emplace_back(std::move(param));
      continue;
    }
    required_params.emplace_back(std::move(param));
  }
  sig.params.insert(sig.params.end(), required_params.begin(), required_params.end());
  sig.params.insert(sig.params.end(), optional_params.begin(), optional_params.end());
}

// 支持input类型为TensorLike场景: 1、IrInputs都是可选 2、IrInputs个数大于1且不都是向量
bool OverloadPlanner::UseTensorLikeInputs(const IrOpProto &current) const {
  if (current.inputs.empty()) {
    return false;
  }
  if (std::all_of(current.inputs.begin(),
                  current.inputs.end(),
                  [](const auto &in) { return in.type == kIrInputOptional; })) {
    return true;
  }
  if (current.inputs.size() <= 1) {
    return false;
  }
  if (std::any_of(current.inputs.begin(),
                  current.inputs.end(),
                  [](const auto &in) { return in.type == kIrInputRequired || in.type == kIrInputOptional; })) {
    return true;
  }
  return false;
}

bool OverloadPlanner::IsInputNameDuplicated(const IrOpProto &current, const std::string &name) const {
  for (const auto &in: current.inputs) {
    if (in.name == name) {
      return true;
    }
  }
  return false;
}

std::string OverloadPlanner::DynamicOutputParamName(const IrOpProto &current, const std::string &output_name) const {
  std::string base = output_name;
  if (IsInputNameDuplicated(current, output_name)) {
    base = "ref_" + InName(output_name);
  } else if (IsKeyword(output_name)) {
    base = "out_" + output_name;
  }
  return base + "_num";
}
} // namespace history
} // namespace es
} // namespace ge

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_OVERLOAD_PLANNER_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_OVERLOAD_PLANNER_H_

#include <string>
#include <vector>

#include "default_value_policy.h"
#include "history_registry_types.h"
#include "overload_planner_types.h"
#include "history_registry_interface.h"

namespace ge {
namespace es {
namespace history {
// 负责根据 current proto 与兼容窗口内的历史 proto 生成 C++ 重载签名计划。
// 设计目标：
// 1. 尽量保持旧调用形态可用（保留旧重载）。
// 2. 在避免二义性的前提下优先使用更简单的签名策略。
// 3. 保留 A0 回退兜底（防御式路径）：
//    理想情况下兼容链应避免触发，但在历史数据异常、联调阶段 schema 不一致、
//    或 Try0/A1/A2 仍存在二义性时，必须保证生成结果可编译可用。
class OverloadPlanner {
  public:
    // 入口：输出用于代码生成的签名列表与告警信息。
    OverloadPlan Plan(const IrOpProto &current, const HistoryContext &history) const;

  private:
    // current 相对 baseline 的差异分析结果。
    // - compatible: schema 是否满足“兼容窗口内可处理”的基本约束。
    // - can_safe_merge: 是否可直接合并为单签名（无需保留旧重载）。
    // - baseline_all_inputs_optional: 用于 owner_builder 位置迁移风险判断。
    // - new_inputs: baseline 之后新增的可选输入（按尾部追加规则）。
    // - detail: 失败/降级原因，供 warning 输出。
    struct DiffResult {
      bool compatible = false;
      bool can_safe_merge = false;
      bool baseline_all_inputs_optional = false;
      std::vector<std::string> new_inputs;
      std::string detail;
    };

    // 历史链中的“重载边界”（版本分界点）：
    // 含义：相邻两个版本“仍兼容”，但“不能直接合并为一个最新签名”，
    //      因此必须保留旧函数形态（legacy overload）来兼容旧调用。
    // step_index 表示 versions[step_index - 1] -> versions[step_index] 这一跳是分界点。
    struct BoundaryInfo {
      size_t step_index = 0U;
      std::vector<std::string> new_inputs;
      bool prev_all_inputs_optional = false;
    };

    // CollectBoundaries 的返回值：
    // - success=false 表示历史链出现不可处理变化，Plan 需回退 A0。
    // - boundaries 保存需要保留旧形态的所有边界。
    // - all_new_inputs 用于聚合 warning 上下文。
    struct BoundaryCollectResult {
      bool success = true;
      std::string fail_reason;
      std::vector<BoundaryInfo> boundaries;
      std::vector<std::string> all_new_inputs;
    };

    // 某个 baseline 版本在特定 mode 下生成签名时需要的“强制参数策略”。
    // baseline 决定“要还原到哪个历史版本的函数外形”；
    // mode 决定“在这个外形上新增输入采用多强的约束来消歧义”。
    struct BaselinePlanInput {
      std::vector<std::string> removed_inputs;
      std::vector<std::string> required_inputs;
      std::vector<std::string> tensor_holder_inputs;
    };

    // 多重载模式（从保守到强约束逐级升级）：
    // kTry0: 不强制新增输入 required，不加 guard，先尝试最轻量方案。
    // kA1:  强制新增输入 required（特殊场景会扩大到 baseline 全输入），并追加 nullptr guard。
    // kA2:  在 A1 基础上将新增输入切到 TensorHolder，进一步压缩隐式转换空间，再追加 guard。
    // 关系说明：同一个 baseline 会在不同 mode 下生成不同“约束强度”的签名形态。
    enum class MultiBaselineMode {
      kTry0,
      kA1,
      kA2
    };

    // 组装版本链：history.proto_chain + current。
    std::vector<const IrOpProto *> BuildVersionChain(const IrOpProto &current,
                                                     const HistoryContext &history) const;
    // 遍历版本链并收集重载边界；若出现不可处理变化则直接返回失败原因。
    BoundaryCollectResult CollectBoundaries(const std::vector<const IrOpProto *> &versions) const;
    // 根据边界推导需要生成签名的 baseline 下标集合。
    std::vector<size_t> CollectBaselineIndices(size_t version_count,
                                               const std::vector<BoundaryInfo> &boundaries) const;
    // 在指定模式下构建一组签名并做二义性判定，成功则写入 accepted_signatures。
    bool TryAdoptModeSignatures(const IrOpProto &current,
                                const std::vector<const IrOpProto *> &versions,
                                const std::vector<BoundaryInfo> &boundaries,
                                MultiBaselineMode mode,
                                std::vector<Warning> *warnings,
                                std::vector<Signature> &accepted_signatures) const;
    // 执行 Try0 -> A1 -> A2 的升级链；任一模式可用即返回 true。
    bool TryResolveModeEscalation(const IrOpProto &current,
                                  const std::vector<const IrOpProto *> &versions,
                                  const std::vector<BoundaryInfo> &boundaries,
                                  const std::string &warning_context,
                                  OverloadPlan &plan) const;
    // 按 mode 为所有 baseline 构造签名，并按边界追加 guard 签名。
    std::vector<Signature> BuildMultiBaselineSignatures(const IrOpProto &current,
                                                        const std::vector<const IrOpProto *> &versions,
                                                        const std::vector<BoundaryInfo> &boundaries,
                                                        MultiBaselineMode mode,
                                                        std::vector<Warning> *warnings) const;
    // 计算单个 baseline 在 mode 下的强制 required/TensorHolder 策略与需裁剪输入。
    //baseline 负责保留哪一代调用外形, mode 负责这代外形里新增输入如何约束（Try0/A1/A2)
    BaselinePlanInput BuildBaselinePlanInput(const IrOpProto &current,
                                             const IrOpProto &baseline,
                                             const std::vector<BoundaryInfo> &boundaries,
                                             size_t baseline_index,
                                             MultiBaselineMode mode) const;
    // 基于边界新增输入追加 guard 删除签名（用于编译期阻断高风险调用形态）。
    void AppendBoundaryGuards(std::vector<Signature> &signatures,
                              const std::vector<Signature> &baseline_signatures,
                              const std::vector<bool> &has_baseline_signature,
                              const std::vector<BoundaryInfo> &boundaries,
                              MultiBaselineMode mode) const;
    // 以下 Diff* 函数用于分段比较 schema，便于维护与定位不兼容原因。
    bool DiffInputs(const IrOpProto &current, const IrOpProto &baseline, DiffResult &diff) const;
    bool DiffAttrs(const IrOpProto &current, const IrOpProto &baseline, DiffResult &diff) const;
    bool DiffOutputs(const IrOpProto &current, const IrOpProto &baseline, DiffResult &diff) const;
    bool DiffSubgraphs(const IrOpProto &current, const IrOpProto &baseline, DiffResult &diff) const;
    // 在 schema 基础兼容后，进一步评估是否可 safe-merge 到单签名。
    void EvaluateMergeSafety(const IrOpProto &current, const IrOpProto &baseline, DiffResult &diff) const;
    DiffResult AnalyzeDiff(const IrOpProto &current, const IrOpProto &baseline) const;
    // 对候选签名集做两两二义性检测。
    bool HasPotentialAmbiguity(const std::vector<Signature> &signatures) const;
    // 标量字面量属性会放大重载匹配范围，需参与 A1/A2 升级判断。
    bool HasScalarLiteralAttrRisk(const IrOpProto &proto) const;
    // A0：仅生成最新版本单签名。
    Signature BuildA0Signature(const IrOpProto &current, std::vector<Warning> *warnings) const;
    // 按既定参数顺序构造一个完整签名：
    // input -> owner_builder -> dynamic_output -> subgraph -> attrs
    Signature BuildSignature(const IrOpProto &current,
                             const std::vector<std::string> *force_required_inputs,
                             const std::vector<std::string> *force_tensor_holder_inputs,
                             const std::vector<std::string> *removed_inputs,
                             std::vector<Warning> *warnings) const;
    // 构造“删除签名 + deprecate 消息”的 guard，用于限制危险调用。
    Signature BuildNullptrGuardSignature(const Signature &sig,
                                         const std::string &input_name,
                                         bool tensor_holder_mode) const;
    int FindInputParamIndex(const Signature &sig, const std::string &input_name) const;
    void AppendInputParams(const IrOpProto &current,
                           const DefaultValuePolicy &policy,
                           const std::vector<std::string> *force_tensor_holder_inputs,
                           Signature &sig) const;
    void AppendDynamicOutputParams(const IrOpProto &current, Signature &sig) const;
    void AppendSubgraphParams(const IrOpProto &current, Signature &sig) const;
    void AppendOwnerBuilderParam(const IrOpProto &current,
                                 const DefaultValuePolicy &policy,
                                 const std::vector<std::string> *removed_inputs,
                                 Signature &sig) const;
    void AppendAttrParams(const IrOpProto &current, Signature &sig, std::vector<Warning> *warnings) const;
    bool UseTensorLikeInputs(const IrOpProto &current) const;
    bool IsInputNameDuplicated(const IrOpProto &current, const std::string &name) const;
    std::string DynamicOutputParamName(const IrOpProto &current, const std::string &output_name) const;
};

/**
 * 计划 C++ 算子重载（便捷封装）
 * @param current 当前算子原型
 * @param history 历史算子原型链
 * @param warnings 记录计划失败原因
 * @return 重载计划
 */
OverloadPlan PlanCppOverloads(const IrOpProto &current,
                              const HistoryContext &history,
                              std::vector<std::string> &warnings);
} // namespace history
} // namespace es
} // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_OVERLOAD_PLANNER_H_

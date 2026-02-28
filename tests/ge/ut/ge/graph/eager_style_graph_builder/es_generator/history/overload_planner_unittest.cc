/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "history/overload_planner.h"
#include "history/warning_formatter.h"

using namespace ge::es::history;

namespace {
bool HasDefaultBeforeRequired(const Signature &sig) {
  bool seen_default = false;
  for (const auto &param: sig.params) {
    if (param.has_default) {
      seen_default = true;
      continue;
    }
    if (seen_default) {
      return true;
    }
  }
  return false;
}

bool HasOwnerBuilderParam(const Signature &sig) {
  for (const auto &param: sig.params) {
    if (param.name == "owner_builder") {
      return true;
    }
  }
  return false;
}

const Param *FindParamByName(const Signature &sig, const std::string &name) {
  for (const auto &param : sig.params) {
    if (param.name == name) {
      return &param;
    }
  }
  return nullptr;
}

bool HasDeletedNullptrGuardForInput(const OverloadPlan &plan, const std::string &input_name) {
  for (const auto &sig : plan.signatures) {
    if (!sig.is_deleted) {
      continue;
    }
    for (const auto &param : sig.params) {
      if (param.ir_name == input_name && param.kind == ParamCxxKind::kNullptrT) {
        return true;
      }
    }
  }
  return false;
}

bool HasWarningCode(const OverloadPlan &plan, WarningCode code) {
  for (const auto &warning : plan.warnings) {
    if (warning.code == code) {
      return true;
    }
  }
  return false;
}

bool HasNonDeletedInputWithKind(const OverloadPlan &plan, const std::string &input_name, ParamCxxKind kind) {
  for (const auto &sig : plan.signatures) {
    if (sig.is_deleted) {
      continue;
    }
    const auto *param = FindParamByName(sig, input_name);
    if (param != nullptr && param->kind == kind) {
      return true;
    }
  }
  return false;
}
} // namespace

class OverloadPlannerUT : public ::testing::Test {
  protected:
    IrOpProto BuildProto(const std::vector<IrInput> &inputs,
                         const std::vector<IrAttr> &attrs = {},
                         const std::vector<IrOutput> &outputs = {},
                         const std::vector<IrSubgraph> &subgraphs = {}) const {
      IrOpProto proto;
      proto.op_type = "Foo";
      proto.inputs = inputs;
      proto.attrs = attrs;
      proto.outputs = outputs;
      proto.subgraphs = subgraphs;
      return proto;
    }

    HistoryContext BuildHistory(const std::vector<IrOpProto> &chain) const {
      HistoryContext context;
      context.proto_chain = chain;
      return context;
    }

    OverloadPlanner planner_;
};

TEST_F(OverloadPlannerUT, PlanWithoutHistoryReturnsSingleA0Signature) {
  const auto current = BuildProto({{"x", ge::kIrInputRequired, {}}, {"xo", ge::kIrInputOptional, {}}});
  const auto plan = planner_.Plan(current, HistoryContext{});
  ASSERT_EQ(plan.signatures.size(), 1U);
  ASSERT_EQ(plan.signatures[0].params.size(), 2U);
  EXPECT_EQ(plan.signatures[0].params[0].kind, ParamCxxKind::kEsTensorLikeRef);
  EXPECT_EQ(plan.signatures[0].params[1].kind, ParamCxxKind::kEsTensorLikeRef);
  EXPECT_TRUE(plan.signatures[0].params[1].has_default);
  EXPECT_EQ(plan.signatures[0].params[1].default_expr, "=nullptr");
}

TEST_F(OverloadPlannerUT, PlanA0WhenTailOptionalInputIsSafeToMerge) {
  // 场景：仅在尾部新增可选输入（无历史属性干扰），判定为可直接合并到单签名 A0。
  const auto v1 = BuildProto({{"x", ge::kIrInputRequired, {}}});
  const auto current = BuildProto({{"x", ge::kIrInputRequired, {}}, {"xo2", ge::kIrInputOptional, {}}});
  const auto plan = planner_.Plan(current, BuildHistory({v1}));
  ASSERT_EQ(plan.signatures.size(), 1U);
  ASSERT_EQ(plan.signatures[0].params.size(), 2U);
  EXPECT_EQ(plan.signatures[0].params[1].name, "xo2");
  EXPECT_TRUE(plan.signatures[0].params[1].has_default);
  EXPECT_EQ(plan.signatures[0].params[1].default_expr, "=nullptr");
  EXPECT_TRUE(plan.warnings.empty());
}

TEST_F(OverloadPlannerUT, PlanTry0WhenNewInputWithHistoricalRequiredAttrHasNoAmbiguity) {
  // 场景：新增可选输入 + 历史里已有必选属性。
  // 由于必选属性存在，新增可选输入不会拿到默认值（避免 default-before-required），Try0 两个签名可共存且无二义性。
  const IrAttr keep = {"keep", "Int", true, ""};
  const auto v1 = BuildProto({{"x", ge::kIrInputRequired, {}}}, {keep});
  const auto current = BuildProto({{"x", ge::kIrInputRequired, {}}, {"xo2", ge::kIrInputOptional, {}}}, {keep});
  const auto plan = planner_.Plan(current, BuildHistory({v1}));

  ASSERT_EQ(plan.signatures.size(), 2U);
  ASSERT_TRUE(plan.warnings.empty());

  // v1 legacy：x + keep
  ASSERT_EQ(plan.signatures[0].params.size(), 2U);
  EXPECT_EQ(plan.signatures[0].params[0].name, "x");
  EXPECT_EQ(plan.signatures[0].params[1].name, "keep");

  // v2 latest：x + xo2(无默认) + keep
  ASSERT_EQ(plan.signatures[1].params.size(), 3U);
  EXPECT_EQ(plan.signatures[1].params[1].name, "xo2");
  EXPECT_FALSE(plan.signatures[1].params[1].has_default);
  EXPECT_EQ(plan.signatures[1].params[2].name, "keep");
}

TEST_F(OverloadPlannerUT, PlanUpgradeToA1WhenTry0IsAmbiguous) {
  const IrAttr mode = {"mode", "String", false, "\"xx\""};
  const auto v1 = BuildProto({{"x", ge::kIrInputRequired, {}}, {"xo1", ge::kIrInputRequired, {}}}, {mode});
  const auto current = BuildProto({
                                    {"x", ge::kIrInputRequired, {}},
                                    {"xo1", ge::kIrInputRequired, {}},
                                    {"xo2", ge::kIrInputOptional, {}}
                                  },
                                  {mode});
  const auto plan = planner_.Plan(current, BuildHistory({v1}));
  ASSERT_EQ(plan.signatures.size(), 3U);

  // v1 legacy signature: no xo2.
  EXPECT_EQ(plan.signatures[0].params.size(), 3U);
  EXPECT_EQ(plan.signatures[0].params[0].name, "x");
  EXPECT_EQ(plan.signatures[0].params[1].name, "xo1");
  EXPECT_EQ(plan.signatures[0].params[2].name, "mode");

  // v2 A1 signature: xo2 is required EsTensorLike.
  EXPECT_EQ(plan.signatures[1].params[2].name, "xo2");
  EXPECT_EQ(plan.signatures[1].params[2].kind, ParamCxxKind::kEsTensorLikeRef);
  EXPECT_FALSE(plan.signatures[1].params[2].has_default);

  // nullptr guard.
  EXPECT_TRUE(plan.signatures[2].is_deleted);
  EXPECT_TRUE(plan.signatures[2].is_deprecated);
  EXPECT_EQ(plan.signatures[2].params[2].kind, ParamCxxKind::kNullptrT);

  ASSERT_FALSE(plan.warnings.empty());
  const auto &warning = plan.warnings.back();
  EXPECT_EQ(warning.code, WarningCode::kUpgradeToA1);
  EXPECT_NE(warning.detail.find("op Foo"), std::string::npos);
  EXPECT_NE(warning.detail.find("new optional inputs [xo2]"), std::string::npos);
  EXPECT_EQ(FormatWarning(warning),
            "Context: op Foo, new optional inputs [xo2]. "
            "Action: Try0 overloads (legacy + full signature) are ambiguous, so switch to A1: "
            "force new inputs required and add nullptr-guard overloads.");
}

TEST_F(OverloadPlannerUT, PlanUpgradeToA2WhenA1StillAmbiguousWithScalarLiteral) {
  const IrAttr a = {"a", "Int", false, "0"};
  const IrAttr b = {"b", "Int", false, "0"};
  const auto v1 = BuildProto({{"x", ge::kIrInputRequired, {}}, {"xo1", ge::kIrInputRequired, {}}}, {a});
  const auto current = BuildProto({
                                    {"x", ge::kIrInputRequired, {}},
                                    {"xo1", ge::kIrInputRequired, {}},
                                    {"xo2", ge::kIrInputOptional, {}}
                                  },
                                  {a, b});
  const auto plan = planner_.Plan(current, BuildHistory({v1}));
  ASSERT_EQ(plan.signatures.size(), 3U);
  EXPECT_EQ(plan.signatures[1].params[2].name, "xo2");
  EXPECT_EQ(plan.signatures[1].params[2].kind, ParamCxxKind::kTensorHolderRef);
  EXPECT_FALSE(plan.signatures[1].params[2].has_default);
  EXPECT_TRUE(plan.signatures[2].is_deleted);
  EXPECT_EQ(plan.signatures[2].params[2].kind, ParamCxxKind::kNullptrT);
  EXPECT_NE(plan.signatures[2].deprecate_msg.find("CreateScalar"), std::string::npos);

  bool has_upgrade_to_a1 = false;
  bool has_upgrade_to_a2 = false;
  std::string formatted_upgrade_to_a1;
  std::string formatted_upgrade_to_a2;
  for (const auto &warning : plan.warnings) {
    if (warning.code == WarningCode::kUpgradeToA1) {
      has_upgrade_to_a1 = true;
      EXPECT_NE(warning.detail.find("op Foo"), std::string::npos);
      EXPECT_NE(warning.detail.find("new optional inputs [xo2]"), std::string::npos);
      formatted_upgrade_to_a1 = FormatWarning(warning);
    }
    if (warning.code == WarningCode::kUpgradeToA2) {
      has_upgrade_to_a2 = true;
      EXPECT_NE(warning.detail.find("op Foo"), std::string::npos);
      EXPECT_NE(warning.detail.find("new optional inputs [xo2]"), std::string::npos);
      formatted_upgrade_to_a2 = FormatWarning(warning);
    }
  }
  EXPECT_TRUE(has_upgrade_to_a1);
  EXPECT_TRUE(has_upgrade_to_a2);
  EXPECT_EQ(formatted_upgrade_to_a1,
            "Context: op Foo, new optional inputs [xo2]. "
            "Action: Try0 overloads (legacy + full signature) are ambiguous, so switch to A1: "
            "force new inputs required and add nullptr-guard overloads.");
  EXPECT_EQ(formatted_upgrade_to_a2,
            "Context: op Foo, new optional inputs [xo2]. "
            "Action: A1 overloads are still ambiguous, so switch to A2: "
            "force new inputs as TensorHolder and add nullptr-guard overloads.");
}

TEST_F(OverloadPlannerUT, PlanRejectedModesShouldNotLeakInternalWarnings) {
  // 场景：新增可选属性中包含不支持类型且 default_value 不可解析。
  // 期望：在进入模式升级前就按不兼容回退 A0，同时仍保留不支持类型告警。
  const IrAttr a = {"a", "Int", false, "0"};
  const IrAttr b = {"b", "Int", false, "0"};
  const IrAttr unsupported = {"bad_attr", "UnknownType", false, ""};
  const auto v1 = BuildProto({{"x", ge::kIrInputRequired, {}}, {"xo1", ge::kIrInputRequired, {}}}, {a});
  const auto current = BuildProto({
                                    {"x", ge::kIrInputRequired, {}},
                                    {"xo1", ge::kIrInputRequired, {}},
                                    {"xo2", ge::kIrInputOptional, {}}
                                  },
                                  {a, b, unsupported});
  const auto plan = planner_.Plan(current, BuildHistory({v1}));

  bool has_upgrade_to_a1 = false;
  bool has_upgrade_to_a2 = false;
  bool has_fallback_to_a0 = false;
  int unsupported_warning_count = 0;
  for (const auto &warning : plan.warnings) {
    if (warning.code == WarningCode::kUpgradeToA1) {
      has_upgrade_to_a1 = true;
    }
    if (warning.code == WarningCode::kUpgradeToA2) {
      has_upgrade_to_a2 = true;
    }
    if (warning.code == WarningCode::kFallbackToA0) {
      has_fallback_to_a0 = true;
      EXPECT_NE(warning.detail.find("bad_attr"), std::string::npos);
      EXPECT_NE(warning.detail.find("invalid default_value"), std::string::npos);
    }
    if (warning.code == WarningCode::kUnsupportedAttrType) {
      ++unsupported_warning_count;
      EXPECT_NE(warning.detail.find("bad_attr"), std::string::npos);
    }
  }
  EXPECT_FALSE(has_upgrade_to_a1);
  EXPECT_FALSE(has_upgrade_to_a2);
  EXPECT_TRUE(has_fallback_to_a0);
  EXPECT_EQ(unsupported_warning_count, 1);
}

TEST_F(OverloadPlannerUT, PlanOnlyNewOptionalAttrKeepsSingleLatestSignature) {
  // 场景：仅新增可选属性（带默认值），应直接合并为 A0，不生成重载。
  const auto v1 = BuildProto({{"x", ge::kIrInputRequired, {}}});
  const IrAttr axis = {"axis", "Int", false, "7"};
  const auto current = BuildProto({{"x", ge::kIrInputRequired, {}}}, {axis});
  const auto plan = planner_.Plan(current, BuildHistory({v1}));
  ASSERT_EQ(plan.signatures.size(), 1U);
  ASSERT_EQ(plan.signatures[0].params.size(), 2U);
  const auto &param = plan.signatures[0].params[1];
  EXPECT_EQ(param.kind, ParamCxxKind::kInt64);
  EXPECT_TRUE(param.has_default);
  EXPECT_EQ(param.default_expr, "=7");
}

TEST_F(OverloadPlannerUT, PlanFallbackToA0WhenNewOptionalAttrMissingDefaultValue) {
  // 场景：新增可选属性但 default_value 缺失，判定为不兼容，回退 A0。
  const auto v1 = BuildProto({{"x", ge::kIrInputRequired, {}}});
  const IrAttr axis = {"axis", "Int", false, ""};
  const auto current = BuildProto({{"x", ge::kIrInputRequired, {}}}, {axis});
  const auto plan = planner_.Plan(current, BuildHistory({v1}));

  ASSERT_EQ(plan.signatures.size(), 1U);
  ASSERT_FALSE(plan.warnings.empty());
  bool has_fallback = false;
  for (const auto &warning : plan.warnings) {
    if (warning.code != WarningCode::kFallbackToA0) {
      continue;
    }
    has_fallback = true;
    EXPECT_NE(warning.detail.find("new optional attr 'axis' has invalid default_value"), std::string::npos);
    EXPECT_NE(warning.detail.find("missing default_value"), std::string::npos);
    EXPECT_EQ(FormatWarning(warning),
              "Context: op Foo. "
              "Cause: incompatible schema change in history chain at step 1: "
              "new optional attr 'axis' has invalid default_value: missing default_value. "
              "Action: fall back to A0 single-signature plan to avoid ambiguous or incompatible overloads.");
  }
  EXPECT_TRUE(has_fallback);
}

TEST_F(OverloadPlannerUT, PlanEmitsStructuredWarningWhenOptionalAttrDefaultTypeMismatch) {
  // 场景：可选属性 default_value 可解析 JSON 但类型不匹配。
  // 期望：默认值不写入签名，且产出结构化 warning。
  const IrAttr axis = {"axis", "Int", false, "\"not_int\""};
  const auto current = BuildProto({{"x", ge::kIrInputRequired, {}}}, {axis});
  const auto plan = planner_.Plan(current, HistoryContext{});

  ASSERT_EQ(plan.signatures.size(), 1U);
  ASSERT_EQ(plan.signatures[0].params.size(), 2U);
  EXPECT_EQ(plan.signatures[0].params[1].name, "axis");
  EXPECT_FALSE(plan.signatures[0].params[1].has_default);

  ASSERT_FALSE(plan.warnings.empty());
  bool has_invalid_default_warning = false;
  for (const auto &warning : plan.warnings) {
    if (warning.code != WarningCode::kInvalidAttrDefaultValue) {
      continue;
    }
    has_invalid_default_warning = true;
    EXPECT_NE(warning.detail.find("attr 'axis' type 'Int'"), std::string::npos);
    EXPECT_NE(warning.detail.find("default_value type mismatch for Int"), std::string::npos);
    EXPECT_EQ(FormatWarning(warning),
              "Context: op Foo, attr 'axis' type 'Int'. "
              "Cause: default_value type mismatch for Int. "
              "Action: attr default_value cannot be parsed; emit this attr without default in C++ signature.");
  }
  EXPECT_TRUE(has_invalid_default_warning);
}

TEST_F(OverloadPlannerUT, PlanA0WhenOptionalInputAndOptionalAttrAreBothAppended) {
  // 场景：同时新增“可选输入 + 可选属性（非标量字面量风险类型）”，仍可直接 merge 到 A0。
  const auto v1 = BuildProto({{"x", ge::kIrInputRequired, {}}});
  const IrAttr mode = {"mode", "String", false, "\"new_mode\""};
  const auto current = BuildProto({{"x", ge::kIrInputRequired, {}}, {"xo2", ge::kIrInputOptional, {}}}, {mode});
  const auto plan = planner_.Plan(current, BuildHistory({v1}));

  ASSERT_EQ(plan.signatures.size(), 1U);
  ASSERT_TRUE(plan.warnings.empty());
  ASSERT_EQ(plan.signatures[0].params.size(), 3U);
  EXPECT_EQ(plan.signatures[0].params[0].name, "x");
  EXPECT_EQ(plan.signatures[0].params[1].name, "xo2");
  EXPECT_TRUE(plan.signatures[0].params[1].has_default);
  EXPECT_EQ(plan.signatures[0].params[1].default_expr, "=nullptr");
  EXPECT_EQ(plan.signatures[0].params[2].name, "mode");
  EXPECT_TRUE(plan.signatures[0].params[2].has_default);
  EXPECT_EQ(plan.signatures[0].params[2].default_expr, "=\"new_mode\"");
}

TEST_F(OverloadPlannerUT, PlanA1WithTwoNewInputsBuildsTwoNullptrGuards) {
  const IrAttr mode = {"mode", "String", false, "\"xx\""};
  const auto v1 = BuildProto({{"x", ge::kIrInputRequired, {}}, {"xo1", ge::kIrInputRequired, {}}}, {mode});
  const auto current = BuildProto({
                                    {"x", ge::kIrInputRequired, {}},
                                    {"xo1", ge::kIrInputRequired, {}},
                                    {"xo2", ge::kIrInputOptional, {}},
                                    {"xo3", ge::kIrInputOptional, {}}
                                  },
                                  {mode});
  const auto plan = planner_.Plan(current, BuildHistory({v1}));
  ASSERT_EQ(plan.signatures.size(), 4U);

  const auto &v2 = plan.signatures[1];
  ASSERT_GE(v2.params.size(), 5U);
  EXPECT_EQ(v2.params[2].name, "xo2");
  EXPECT_EQ(v2.params[2].kind, ParamCxxKind::kEsTensorLikeRef);
  EXPECT_FALSE(v2.params[2].has_default);
  EXPECT_EQ(v2.params[3].name, "xo3");
  EXPECT_EQ(v2.params[3].kind, ParamCxxKind::kEsTensorLikeRef);
  EXPECT_FALSE(v2.params[3].has_default);

  const auto &guard_xo2 = plan.signatures[2];
  const auto &guard_xo3 = plan.signatures[3];
  EXPECT_TRUE(guard_xo2.is_deleted);
  EXPECT_TRUE(guard_xo3.is_deleted);
  EXPECT_EQ(guard_xo2.params[2].kind, ParamCxxKind::kNullptrT);
  EXPECT_EQ(guard_xo3.params[3].kind, ParamCxxKind::kNullptrT);
}

TEST_F(OverloadPlannerUT, PlanAllOptionalInputsWithNewInputKeepsLegacyByA1) {
  // 场景：全部输入可选 + 历史已有属性时新增可选输入。
  // 为避免 owner_builder 位置后移破坏旧调用，不允许 safe-merge，且通过 A1 保留旧重载。
  const IrAttr mode = {"mode", "String", false, "\"xx\""};
  const auto v1 = BuildProto({{"x", ge::kIrInputOptional, {}}}, {mode});
  const auto current = BuildProto({{"x", ge::kIrInputOptional, {}}, {"xo2", ge::kIrInputOptional, {}}}, {mode});
  const auto plan = planner_.Plan(current, BuildHistory({v1}));
  ASSERT_EQ(plan.signatures.size(), 3U);

  // legacy: x + owner_builder + mode
  ASSERT_EQ(plan.signatures[0].params.size(), 3U);
  EXPECT_EQ(plan.signatures[0].params[0].name, "x");
  EXPECT_EQ(plan.signatures[0].params[1].name, "owner_builder");
  EXPECT_EQ(plan.signatures[0].params[2].name, "mode");

  // A1: x / xo2 均强制 required，latest full 签名不再保留 owner_builder。
  ASSERT_EQ(plan.signatures[1].params.size(), 3U);
  EXPECT_EQ(plan.signatures[1].params[0].name, "x");
  EXPECT_FALSE(plan.signatures[1].params[0].has_default);
  EXPECT_EQ(plan.signatures[1].params[1].name, "xo2");
  EXPECT_FALSE(plan.signatures[1].params[1].has_default);
  EXPECT_EQ(plan.signatures[1].params[2].name, "mode");
  EXPECT_TRUE(plan.signatures[1].params[2].has_default);

  // guard: xo2 的 nullptr 保护签名
  EXPECT_TRUE(plan.signatures[2].is_deleted);
  EXPECT_TRUE(plan.signatures[2].is_deprecated);
  EXPECT_EQ(plan.signatures[2].params[1].kind, ParamCxxKind::kNullptrT);

  bool has_upgrade_to_a1 = false;
  bool has_fallback_a0 = false;
  for (const auto &warning : plan.warnings) {
    if (warning.code == WarningCode::kUpgradeToA1) {
      has_upgrade_to_a1 = true;
      EXPECT_NE(warning.detail.find("op Foo"), std::string::npos);
      EXPECT_NE(warning.detail.find("new optional inputs [xo2]"), std::string::npos);
      EXPECT_EQ(FormatWarning(warning),
                "Context: op Foo, new optional inputs [xo2]. "
                "Action: Try0 overloads (legacy + full signature) are ambiguous, so switch to A1: "
                "force new inputs required and add nullptr-guard overloads.");
    }
    if (warning.code == WarningCode::kFallbackToA0) {
      has_fallback_a0 = true;
    }
  }
  EXPECT_TRUE(has_upgrade_to_a1);
  EXPECT_FALSE(has_fallback_a0);
}

TEST_F(OverloadPlannerUT, PlanKeepsLegacyOverloadWhenAllOptionalInputsAppendNewOptionalInput) {
  // 场景：历史输入全部可选，且会生成 owner_builder。
  // 仅追加可选输入时，如果直接 merge 会导致 owner_builder 位置后移，破坏旧调用形态，
  // 因此这里不允许 safe-merge，必须保留旧重载。
  const auto v1 = BuildProto({{"x", ge::kIrInputOptional, {}}});
  const auto current = BuildProto({{"x", ge::kIrInputOptional, {}}, {"xo2", ge::kIrInputOptional, {}}});
  const auto plan = planner_.Plan(current, BuildHistory({v1}));

  ASSERT_GT(plan.signatures.size(), 1U);
  // 旧签名应作为第一个重载保留：x + owner_builder（无 xo2）
  ASSERT_EQ(plan.signatures[0].params.size(), 2U);
  EXPECT_EQ(plan.signatures[0].params[0].name, "x");
  EXPECT_EQ(plan.signatures[0].params[1].name, "owner_builder");

  bool has_fallback_a0 = false;
  bool has_upgrade_to_a1 = false;
  for (const auto &warning : plan.warnings) {
    if (warning.code == WarningCode::kFallbackToA0) {
      has_fallback_a0 = true;
      break;
    }
    if (warning.code == WarningCode::kUpgradeToA1) {
      has_upgrade_to_a1 = true;
      EXPECT_NE(warning.detail.find("op Foo"), std::string::npos);
      EXPECT_NE(warning.detail.find("new optional inputs [xo2]"), std::string::npos);
      EXPECT_EQ(FormatWarning(warning),
                "Context: op Foo, new optional inputs [xo2]. "
                "Action: Try0 overloads (legacy + full signature) are ambiguous, so switch to A1: "
                "force new inputs required and add nullptr-guard overloads.");
    }
  }
  EXPECT_FALSE(has_fallback_a0);
  EXPECT_TRUE(has_upgrade_to_a1);
}

TEST_F(OverloadPlannerUT, PlanBuildsMultiVersionSignaturesAcrossHistoryChain) {
  // 场景：历史链中连续两次新增可选输入（全部输入可选，都会影响 owner_builder 位置）。
  // 期望：Plan 需要覆盖多版本形态，而不是只围绕 latest 生成两条签名。
  const auto v1 = BuildProto({{"x", ge::kIrInputOptional, {}}});
  const auto v2 = BuildProto({{"x", ge::kIrInputOptional, {}}, {"xo2", ge::kIrInputOptional, {}}});
  const auto current = BuildProto({
                                    {"x", ge::kIrInputOptional, {}},
                                    {"xo2", ge::kIrInputOptional, {}},
                                    {"xo3", ge::kIrInputOptional, {}}
                                  });
  const auto plan = planner_.Plan(current, BuildHistory({v1, v2}));

  ASSERT_EQ(plan.signatures.size(), 5U);
  ASSERT_EQ(plan.signatures[0].params.size(), 2U);  // x + owner_builder
  EXPECT_EQ(plan.signatures[0].params[0].name, "x");
  EXPECT_EQ(plan.signatures[0].params[1].name, "owner_builder");

  ASSERT_EQ(plan.signatures[1].params.size(), 2U);  // x + xo2（中间版本也不再保留 owner_builder）
  EXPECT_EQ(plan.signatures[1].params[0].name, "x");
  EXPECT_EQ(plan.signatures[1].params[1].name, "xo2");
  EXPECT_FALSE(plan.signatures[1].params[0].has_default);
  EXPECT_FALSE(plan.signatures[1].params[1].has_default);

  ASSERT_EQ(plan.signatures[2].params.size(), 3U);  // x + xo2 + xo3（latest 不含 owner_builder）
  EXPECT_EQ(plan.signatures[2].params[0].name, "x");
  EXPECT_EQ(plan.signatures[2].params[1].name, "xo2");
  EXPECT_EQ(plan.signatures[2].params[2].name, "xo3");
  EXPECT_FALSE(plan.signatures[2].params[0].has_default);
  EXPECT_FALSE(plan.signatures[2].params[1].has_default);
  EXPECT_FALSE(plan.signatures[2].params[2].has_default);

  EXPECT_TRUE(plan.signatures[3].is_deleted);
  EXPECT_TRUE(plan.signatures[3].is_deprecated);
  EXPECT_EQ(plan.signatures[3].params[1].kind, ParamCxxKind::kNullptrT);  // xo2 guard

  EXPECT_TRUE(plan.signatures[4].is_deleted);
  EXPECT_TRUE(plan.signatures[4].is_deprecated);
  EXPECT_EQ(plan.signatures[4].params[2].kind, ParamCxxKind::kNullptrT);  // xo3 guard

  bool has_upgrade_to_a1 = false;
  for (const auto &warning : plan.warnings) {
    if (warning.code == WarningCode::kUpgradeToA1) {
      has_upgrade_to_a1 = true;
      EXPECT_NE(warning.detail.find("new optional inputs [xo2, xo3]"), std::string::npos);
      break;
    }
  }
  EXPECT_TRUE(has_upgrade_to_a1);
}

TEST_F(OverloadPlannerUT, PlanIgnoresDuplicateLatestProtoInHistoryChain) {
  // 场景：history 的最新原型与 current 完全一致（常见于全量快照归档）。
  // 期望：重复 latest 不应参与重载组合，行为与“仅一份 latest”一致。
  const IrAttr a = {"a", "Int", false, "0"};
  const IrAttr b = {"b", "Int", false, "0"};
  const auto v1 = BuildProto({{"x", ge::kIrInputRequired, {}}, {"xo1", ge::kIrInputRequired, {}}}, {a});
  const auto current = BuildProto({
                                    {"x", ge::kIrInputRequired, {}},
                                    {"xo1", ge::kIrInputRequired, {}},
                                    {"xo2", ge::kIrInputOptional, {}}
                                  },
                                  {a, b});
  const auto plan = planner_.Plan(current, BuildHistory({v1, current}));

  ASSERT_EQ(plan.signatures.size(), 3U);
  EXPECT_TRUE(HasWarningCode(plan, WarningCode::kUpgradeToA1));
  EXPECT_TRUE(HasWarningCode(plan, WarningCode::kUpgradeToA2));
  EXPECT_FALSE(HasWarningCode(plan, WarningCode::kFallbackToA0));
  EXPECT_TRUE(HasNonDeletedInputWithKind(plan, "xo2", ParamCxxKind::kTensorHolderRef));
  EXPECT_TRUE(HasDeletedNullptrGuardForInput(plan, "xo2"));
}

TEST_F(OverloadPlannerUT, PlanIgnoresDuplicateMiddleProtoInHistoryChain) {
  // 场景：历史链中间版本重复（v2、v2），真实演进是 v1 -> v2 -> current。
  // 期望：重复节点被折叠，不应扩大 baseline 集合，也不应影响 A1 结果。
  const auto v1 = BuildProto({{"x", ge::kIrInputOptional, {}}});
  const auto v2 = BuildProto({{"x", ge::kIrInputOptional, {}}, {"xo2", ge::kIrInputOptional, {}}});
  const auto current = BuildProto({
                                    {"x", ge::kIrInputOptional, {}},
                                    {"xo2", ge::kIrInputOptional, {}},
                                    {"xo3", ge::kIrInputOptional, {}}
                                  });
  const auto plan = planner_.Plan(current, BuildHistory({v1, v2, v2}));

  ASSERT_EQ(plan.signatures.size(), 5U);
  EXPECT_TRUE(HasWarningCode(plan, WarningCode::kUpgradeToA1));
  EXPECT_FALSE(HasWarningCode(plan, WarningCode::kFallbackToA0));
  EXPECT_TRUE(plan.signatures[3].is_deleted);
  EXPECT_TRUE(plan.signatures[4].is_deleted);
}

TEST_F(OverloadPlannerUT, PlanCollapsesAllDuplicateHistoryProtosToSingleA0) {
  // 场景：history 中所有版本都与 current 完全一致。
  // 期望：等价于“无有效历史差异”，最终仅生成一个 A0 签名。
  const auto current = BuildProto({{"x", ge::kIrInputRequired, {}}, {"xo", ge::kIrInputOptional, {}}});
  const auto plan = planner_.Plan(current, BuildHistory({current, current}));

  ASSERT_EQ(plan.signatures.size(), 1U);
  EXPECT_TRUE(plan.warnings.empty());
  ASSERT_EQ(plan.signatures[0].params.size(), 2U);
  EXPECT_EQ(plan.signatures[0].params[0].name, "x");
  EXPECT_EQ(plan.signatures[0].params[1].name, "xo");
  EXPECT_TRUE(plan.signatures[0].params[1].has_default);
}

TEST_F(OverloadPlannerUT, PlanA0AcrossMultiVersionTailOptionalInputsWhenNoAttrRisk) {
  // 场景：多版本历史链中连续追加可选输入，且无属性风险、无 owner_builder 位置风险。
  // 期望：所有版本可安全 merge，最终仅生成单个 A0 签名。
  const auto v1 = BuildProto({{"x", ge::kIrInputRequired, {}}});
  const auto v2 = BuildProto({{"x", ge::kIrInputRequired, {}}, {"xo2", ge::kIrInputOptional, {}}});
  const auto current = BuildProto({
                                    {"x", ge::kIrInputRequired, {}},
                                    {"xo2", ge::kIrInputOptional, {}},
                                    {"xo3", ge::kIrInputOptional, {}}
                                  });
  const auto plan = planner_.Plan(current, BuildHistory({v1, v2}));

  ASSERT_EQ(plan.signatures.size(), 1U);
  ASSERT_EQ(plan.signatures[0].params.size(), 3U);
  EXPECT_EQ(plan.signatures[0].params[0].name, "x");
  EXPECT_EQ(plan.signatures[0].params[1].name, "xo2");
  EXPECT_TRUE(plan.signatures[0].params[1].has_default);
  EXPECT_EQ(plan.signatures[0].params[2].name, "xo3");
  EXPECT_TRUE(plan.signatures[0].params[2].has_default);
  EXPECT_TRUE(plan.warnings.empty());
}

TEST_F(OverloadPlannerUT, PlanUpgradeToA2AcrossMultiVersionBoundaries) {
  // 场景：历史链中两次追加可选输入，且存在 optional scalar attr（会放大二义性风险）。
  // 期望：Try0/A1 失败后升级到 A2，新增输入在可见签名中应切到 TensorHolder，并生成对应 guard。
  const IrAttr a = {"a", "Int", false, "0"};
  const auto v1 = BuildProto({{"x", ge::kIrInputRequired, {}}, {"xo1", ge::kIrInputRequired, {}}}, {a});
  const auto v2 = BuildProto({
                               {"x", ge::kIrInputRequired, {}},
                               {"xo1", ge::kIrInputRequired, {}},
                               {"xo2", ge::kIrInputOptional, {}}
                             },
                             {a});
  const auto current = BuildProto({
                                    {"x", ge::kIrInputRequired, {}},
                                    {"xo1", ge::kIrInputRequired, {}},
                                    {"xo2", ge::kIrInputOptional, {}},
                                    {"xo3", ge::kIrInputOptional, {}}
                                  },
                                  {a});
  const auto plan = planner_.Plan(current, BuildHistory({v1, v2}));

  EXPECT_TRUE(HasWarningCode(plan, WarningCode::kUpgradeToA1));
  EXPECT_TRUE(HasWarningCode(plan, WarningCode::kUpgradeToA2));
  EXPECT_FALSE(HasWarningCode(plan, WarningCode::kFallbackToA0));
  EXPECT_TRUE(HasNonDeletedInputWithKind(plan, "xo2", ParamCxxKind::kTensorHolderRef));
  EXPECT_TRUE(HasNonDeletedInputWithKind(plan, "xo3", ParamCxxKind::kTensorHolderRef));
  EXPECT_TRUE(HasDeletedNullptrGuardForInput(plan, "xo2"));
  EXPECT_TRUE(HasDeletedNullptrGuardForInput(plan, "xo3"));
}

TEST_F(OverloadPlannerUT, PlanFallbackToA0WhenHistoryChainHasIncompatibleMiddleVersion) {
  // 场景：多版本历史链中间发生不兼容变化（输入名变化），后续版本即使再兼容也应整体回退 A0。
  const auto v1 = BuildProto({{"x", ge::kIrInputRequired, {}}});
  const auto v2 = BuildProto({{"y", ge::kIrInputRequired, {}}});
  const auto current = BuildProto({{"y", ge::kIrInputRequired, {}}, {"yo2", ge::kIrInputOptional, {}}});
  const auto plan = planner_.Plan(current, BuildHistory({v1, v2}));

  ASSERT_EQ(plan.signatures.size(), 1U);
  ASSERT_EQ(plan.signatures[0].params.size(), 2U);
  EXPECT_EQ(plan.signatures[0].params[0].name, "y");
  EXPECT_EQ(plan.signatures[0].params[1].name, "yo2");
  EXPECT_TRUE(plan.signatures[0].params[1].has_default);

  ASSERT_FALSE(plan.warnings.empty());
  const auto &warning = plan.warnings.back();
  EXPECT_EQ(warning.code, WarningCode::kFallbackToA0);
  EXPECT_NE(warning.detail.find("step 1"), std::string::npos);
  EXPECT_NE(warning.detail.find("input mismatch at index 0"), std::string::npos);
}

TEST_F(OverloadPlannerUT, PlanA0ContainsDynamicOutputAndSubgraphParamsInOrder) {
  const auto current = BuildProto({{"x", ge::kIrInputRequired, {}}},
                                  {},
                                  {{"dy", ge::kIrOutputDynamic, {}}},
                                  {{"then_branch", ge::kStatic}, {"branches", ge::kDynamic}});
  const auto plan = planner_.Plan(current, HistoryContext{});
  ASSERT_EQ(plan.signatures.size(), 1U);
  ASSERT_EQ(plan.signatures[0].params.size(), 4U);
  EXPECT_EQ(plan.signatures[0].params[0].name, "x");
  EXPECT_EQ(plan.signatures[0].params[0].kind, ParamCxxKind::kTensorHolderRef);
  EXPECT_EQ(plan.signatures[0].params[1].name, "dy_num");
  EXPECT_EQ(plan.signatures[0].params[1].kind, ParamCxxKind::kInt64);
  EXPECT_EQ(plan.signatures[0].params[2].name, "then_branch");
  EXPECT_EQ(plan.signatures[0].params[2].kind, ParamCxxKind::kGraphUniquePtr);
  EXPECT_EQ(plan.signatures[0].params[3].name, "branches");
  EXPECT_EQ(plan.signatures[0].params[3].kind, ParamCxxKind::kGraphsVec);
}

TEST_F(OverloadPlannerUT, PlanA0AllOptionalPlacesOwnerBuilderBeforeDynamicOutputAndSubgraph) {
  const auto current = BuildProto({{"x", ge::kIrInputOptional, {}}},
                                  {},
                                  {{"dy", ge::kIrOutputDynamic, {}}},
                                  {{"then_branch", ge::kStatic}, {"branches", ge::kDynamic}});
  const auto plan = planner_.Plan(current, HistoryContext{});
  ASSERT_EQ(plan.signatures.size(), 1U);
  ASSERT_EQ(plan.signatures[0].params.size(), 5U);
  EXPECT_EQ(plan.signatures[0].params[0].name, "x");
  EXPECT_EQ(plan.signatures[0].params[0].kind, ParamCxxKind::kEsTensorLikeRef);
  EXPECT_EQ(plan.signatures[0].params[1].name, "owner_builder");
  EXPECT_EQ(plan.signatures[0].params[1].kind, ParamCxxKind::kGraphBuilderPtr);
  EXPECT_EQ(plan.signatures[0].params[2].name, "dy_num");
  EXPECT_EQ(plan.signatures[0].params[2].kind, ParamCxxKind::kInt64);
  EXPECT_EQ(plan.signatures[0].params[3].name, "then_branch");
  EXPECT_EQ(plan.signatures[0].params[3].kind, ParamCxxKind::kGraphUniquePtr);
  EXPECT_EQ(plan.signatures[0].params[4].name, "branches");
  EXPECT_EQ(plan.signatures[0].params[4].kind, ParamCxxKind::kGraphsVec);
}

TEST_F(OverloadPlannerUT, PlanHistoryIncompatibleWithoutNewInputsFallsBackToA0) {
  const auto v1 = BuildProto({{"x", ge::kIrInputRequired, {}}});
  const auto current = BuildProto({{"y", ge::kIrInputRequired, {}}});
  const auto plan = planner_.Plan(current, BuildHistory({v1}));
  ASSERT_EQ(plan.signatures.size(), 1U);
  ASSERT_FALSE(plan.warnings.empty());
  const auto &warning = plan.warnings.back();
  EXPECT_EQ(warning.code, WarningCode::kFallbackToA0);
  EXPECT_NE(warning.detail.find("op Foo"), std::string::npos);
  EXPECT_NE(warning.detail.find("incompatible schema change"), std::string::npos);
  EXPECT_NE(warning.detail.find("step 1"), std::string::npos);
  EXPECT_NE(warning.detail.find("input mismatch at index 0"), std::string::npos);
  const auto formatted = FormatWarning(warning);
  EXPECT_NE(formatted.find("Context: op Foo."), std::string::npos);
  EXPECT_NE(formatted.find("Cause: incompatible schema change"), std::string::npos);
  EXPECT_NE(formatted.find("Action: fall back to A0 single-signature plan"), std::string::npos);
}

TEST_F(OverloadPlannerUT, ValidationShouldRejectDefaultBeforeRequiredOrder) {
  const IrAttr keep = {"keep", "Int", true, ""};
  const auto current = BuildProto({
                                    {"x", ge::kIrInputRequired, {}},
                                    {"y", ge::kIrInputOptional, {}},
                                    {"z", ge::kIrInputOptional, {}}
                                  },
                                  {keep});
  const auto v1 = BuildProto({
                               {"x", ge::kIrInputRequired, {}},
                               {"y", ge::kIrInputOptional, {}}
                             },
                             {keep});
  const auto plan = planner_.Plan(current, BuildHistory({v1}));

  bool has_invalid_signature = false;
  for (const auto &sig: plan.signatures) {
    if (sig.is_deleted) {
      continue;
    }
    if (HasDefaultBeforeRequired(sig)) {
      has_invalid_signature = true;
      break;
    }
  }
  EXPECT_FALSE(has_invalid_signature);
}

TEST_F(OverloadPlannerUT, ValidationShouldRejectTensorLikeSignatureWithoutOwnerBuilderForAllOptionalInputs) {
  const IrAttr mode = {"mode", "String", false, "\"xx\""};
  const auto current = BuildProto({
                                    {"x", ge::kIrInputOptional, {}},
                                    {"y", ge::kIrInputOptional, {}}
                                  },
                                  {mode},
                                  {},
                                  {{"branches", ge::kDynamic}});
  const auto v1 = BuildProto({{"x", ge::kIrInputOptional, {}}},
                             {mode},
                             {},
                             {{"branches", ge::kDynamic}});
  const auto plan = planner_.Plan(current, BuildHistory({v1}));

  bool has_invalid_signature = false;
  for (const auto &sig: plan.signatures) {
    if (sig.is_deleted) {
      continue;
    }
    bool has_effective_optional_tensor_input = false;
    for (const auto &param : sig.params) {
      if (param.role == ParamRole::kInput &&
          param.kind == ParamCxxKind::kEsTensorLikeRef &&
          param.has_default) {
        has_effective_optional_tensor_input = true;
        break;
      }
    }
    if (has_effective_optional_tensor_input && !HasOwnerBuilderParam(sig)) {
      has_invalid_signature = true;
      break;
    }
  }
  EXPECT_FALSE(has_invalid_signature);
}

TEST_F(OverloadPlannerUT, ValidationShouldNormalizeKeywordNamesForDynamicOutputAndSubgraph) {
  const auto current = BuildProto({{"x", ge::kIrInputRequired, {}}},
                                  {},
                                  {{"class", ge::kIrOutputDynamic, {}}},
                                  {{"class", ge::kStatic}});
  const auto plan = planner_.Plan(current, HistoryContext{});
  ASSERT_EQ(plan.signatures.size(), 1U);
  const auto &signature = plan.signatures[0];

  bool has_normalized_dynamic_output_name = false;
  bool has_normalized_subgraph_name = false;
  for (const auto &param: signature.params) {
    if (param.name == "out_class_num") {
      has_normalized_dynamic_output_name = true;
    }
    if (param.name == "subgraph_class") {
      has_normalized_subgraph_name = true;
    }
  }
  EXPECT_TRUE(has_normalized_dynamic_output_name);
  EXPECT_TRUE(has_normalized_subgraph_name);
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "history/history_registry_types.h"
#include "history/history_registry_interface.h"
#include "history/overload_planner_types.h"

#include <gtest/gtest.h>

using namespace ge::es::history;

TEST(HistoryRegistryTypesUT, ParamAggregateInit) {
  Param param{ParamCxxKind::kInt64, "axis", true, "=0"};
  EXPECT_EQ(param.kind, ParamCxxKind::kInt64);
  EXPECT_EQ(param.name, "axis");
  EXPECT_TRUE(param.has_default);
  EXPECT_EQ(param.default_expr, "=0");
}

TEST(HistoryRegistryTypesUT, SignatureDefaults) {
  Signature sig;
  EXPECT_TRUE(sig.params.empty());
  EXPECT_FALSE(sig.is_deleted);
  EXPECT_FALSE(sig.is_deprecated);
  EXPECT_TRUE(sig.deprecate_msg.empty());
}

TEST(HistoryRegistryTypesUT, OverloadPlanDefaults) {
  OverloadPlan plan;
  EXPECT_TRUE(plan.signatures.empty());
  EXPECT_TRUE(plan.warnings.empty());
}

TEST(HistoryRegistryTypesUT, HistoryContextDefaults) {
  HistoryContext context;
  EXPECT_TRUE(context.versions.empty());
  EXPECT_TRUE(context.proto_chain.empty());
}

TEST(HistoryRegistryTypesUT, HistoryContextBasicBuild) {
  VersionMeta meta;
  meta.release_version = "v1";
  meta.release_date = "2026-01-01";
  meta.branch_name = "dev";

  IrOpProto proto;
  proto.op_type = "Foo";

  HistoryContext context;
  context.versions.push_back(meta);
  context.proto_chain.push_back(proto);

  ASSERT_EQ(context.versions.size(), 1U);
  ASSERT_EQ(context.proto_chain.size(), 1U);
  EXPECT_EQ(context.versions[0].release_version, "v1");
  EXPECT_EQ(context.proto_chain[0].op_type, "Foo");
}

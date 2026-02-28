/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "history/ambiguity_checker.h"

#include <gtest/gtest.h>

using namespace ge::es::history;

class AmbiguityCheckerUT : public ::testing::Test {
  protected:
    Signature BuildSig(std::initializer_list<Param> params) const {
      Signature sig;
      sig.params.assign(params.begin(), params.end());
      return sig;
    }
};

TEST_F(AmbiguityCheckerUT, CallRangeBasic) {
  const auto sig = BuildSig({
    {ParamCxxKind::kInt64, "axis", false, ""},
    {ParamCxxKind::kBool, "flag", true, "=false"}
  });
  const auto range = AmbiguityChecker::CallRange(sig);
  EXPECT_EQ(range.first, 1);
  EXPECT_EQ(range.second, 2);
}

TEST_F(AmbiguityCheckerUT, CallRangeOverlap) {
  const auto a = BuildSig({
    {ParamCxxKind::kInt64, "a", false, ""},
    {ParamCxxKind::kBool, "b", true, "=false"}
  });
  const auto b = BuildSig({{ParamCxxKind::kInt64, "a", false, ""}});
  EXPECT_TRUE(AmbiguityChecker::HasCallRangeOverlap(a, b));
}

TEST_F(AmbiguityCheckerUT, CallRangeNoOverlap) {
  const auto a = BuildSig({{ParamCxxKind::kInt64, "a", false, ""}});
  const auto b = BuildSig({
    {ParamCxxKind::kInt64, "a", false, ""},
    {ParamCxxKind::kInt64, "b", false, ""},
    {ParamCxxKind::kInt64, "c", false, ""}
  });
  EXPECT_FALSE(AmbiguityChecker::HasCallRangeOverlap(a, b));
}

TEST_F(AmbiguityCheckerUT, Gate1NoOverlapIsSafe) {
  const auto v1 = BuildSig({{ParamCxxKind::kEsTensorLikeRef, "x", false, ""}});
  const auto v2 = BuildSig({
    {ParamCxxKind::kEsTensorLikeRef, "x", false, ""},
    {ParamCxxKind::kEsTensorLikeRef, "xo1", false, ""},
    {ParamCxxKind::kEsTensorLikeRef, "xo2", false, ""}
  });
  EXPECT_FALSE(AmbiguityChecker::HasPotentialAmbiguityByTypicalArgs(v1, v2));
}

TEST_F(AmbiguityCheckerUT, Gate2ConflictDetected) {
  const auto v1 = BuildSig({
    {ParamCxxKind::kEsTensorLikeRef, "x", false, ""},
    {ParamCxxKind::kEsTensorLikeRef, "xo1", false, ""},
    {ParamCxxKind::kInt64, "a", true, "=0"}
  });
  const auto v2 = BuildSig({
    {ParamCxxKind::kEsTensorLikeRef, "x", false, ""},
    {ParamCxxKind::kEsTensorLikeRef, "xo1", false, ""},
    {ParamCxxKind::kEsTensorLikeRef, "xo2", false, ""}
  });
  EXPECT_TRUE(AmbiguityChecker::HasPotentialAmbiguityByTypicalArgs(v1, v2));
}

TEST_F(AmbiguityCheckerUT, Gate2NoConflictIsSafe) {
  const auto v1 = BuildSig({
    {ParamCxxKind::kEsTensorLikeRef, "x", false, ""},
    {ParamCxxKind::kEsTensorLikeRef, "xo1", false, ""},
    {ParamCxxKind::kCString, "mode", true, "=\"xx\""}
  });
  const auto v2 = BuildSig({
    {ParamCxxKind::kEsTensorLikeRef, "x", false, ""},
    {ParamCxxKind::kEsTensorLikeRef, "xo1", false, ""},
    {ParamCxxKind::kTensorHolderRef, "xo2", false, ""}
  });
  EXPECT_FALSE(AmbiguityChecker::HasPotentialAmbiguityByTypicalArgs(v1, v2));
}

TEST_F(AmbiguityCheckerUT, Gate2NoConflictWhenRequiredSecondArgTypesAreDisjoint) {
  const auto a = BuildSig({
    {ParamCxxKind::kEsTensorLikeRef, "x", false, ""},
    {ParamCxxKind::kInt64, "axis", false, ""}
  });
  const auto b = BuildSig({
    {ParamCxxKind::kEsTensorLikeRef, "x", false, ""},
    {ParamCxxKind::kCString, "mode", false, ""}
  });
  EXPECT_FALSE(AmbiguityChecker::HasPotentialAmbiguityByTypicalArgs(a, b));
}

TEST_F(AmbiguityCheckerUT, Gate2DetectsAmbiguityOnOverlapArityOnly) {
  const auto a = BuildSig({
    {ParamCxxKind::kEsTensorLikeRef, "x", false, ""},
    {ParamCxxKind::kInt64, "axis", true, "=0"}
  });
  const auto b = BuildSig({
    {ParamCxxKind::kEsTensorLikeRef, "x", false, ""},
    {ParamCxxKind::kEsTensorLikeRef, "xo2", false, ""}
  });
  EXPECT_TRUE(AmbiguityChecker::HasPotentialAmbiguityByTypicalArgs(a, b));
}

TEST_F(AmbiguityCheckerUT, TypicalTokensTensorLikeIncludesNullptrAndScalarLiterals) {
  const auto tokens = AmbiguityChecker::TypicalTokens(ParamCxxKind::kEsTensorLikeRef);
  EXPECT_TRUE(tokens.count("tensor") > 0U);
  EXPECT_TRUE(tokens.count("nullptr") > 0U);
  EXPECT_TRUE(tokens.count("0") > 0U);
  EXPECT_TRUE(tokens.count("0.0") > 0U);
}

TEST_F(AmbiguityCheckerUT, TypicalTokensTensorHolderExcludesNullptrAndScalars) {
  const auto tokens = AmbiguityChecker::TypicalTokens(ParamCxxKind::kTensorHolderRef);
  EXPECT_TRUE(tokens.count("tensor") > 0U);
  EXPECT_TRUE(tokens.count("nullptr") == 0U);
  EXPECT_TRUE(tokens.count("0") == 0U);
}

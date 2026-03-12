/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include "util/ternary_op.h"

namespace att {

class TernaryOpsUtilsUnitTest : public testing::Test {
 public:
  void SetUp() override {}
  void TearDown() override {}

  void SetupCommonTernaryOps(std::map<Expr, TernaryOp, ExprCmp> &ternary_ops,
                              Expr &res1, Expr &res2, Expr &res3,
                              Expr &expr1, Expr &expr2, Expr &expr3,
                              Expr &expr4, Expr &expr5) const {
    res1 = CreateExpr("res1");
    res2 = CreateExpr("res2");
    res3 = CreateExpr("res3");
    expr1 = CreateExpr("expr1");
    expr2 = CreateExpr("expr2");
    expr3 = CreateExpr("expr3");
    expr4 = CreateExpr("expr4");
    expr5 = CreateExpr("expr5");
    TernaryOp ternary_op1 = TernaryOp(res3 + res2);
    ternary_op1.SetVariable(res1);
    ternary_ops[res1] = ternary_op1;
    TernaryOp ternary_op2 = TernaryOp(CondType::K_EQ, expr2, res3, expr4, expr3);
    ternary_op2.SetVariable(res2);
    ternary_ops[res2] = ternary_op2;
    TernaryOp ternary_op3 = TernaryOp(CondType::K_LE, expr4, expr5, expr1, expr3);
    ternary_op3.SetVariable(res3);
    ternary_ops[res3] = ternary_op3;
  }
};

TEST_F(TernaryOpsUtilsUnitTest, TestConcursiveReplaceVars) {
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  Expr res1 = CreateExpr("res1");
  Expr res2 = CreateExpr("res2");
  Expr res3 = CreateExpr("res3");
  Expr expr1 = CreateExpr("expr1");
  Expr expr2 = CreateExpr("expr2");
  Expr expr3 = CreateExpr("expr3");
  TernaryOp ternary_op1 = TernaryOp(expr1 + res2);
  ternary_op1.SetVariable(res1);
  ternary_ops[res1] = ternary_op1;
  TernaryOp ternary_op2 = TernaryOp(CondType::K_EQ, expr2, CreateExpr(2), res3, expr3);
  ternary_op2.SetVariable(res2);
  ternary_ops[res2] = ternary_op2;
  TernaryOp ternary_op3 = TernaryOp(CreateExpr(3));
  ternary_op3.SetVariable(res3);
  ternary_ops[res3] = ternary_op3;
  auto res = ConcursiveReplaceVars(ternary_ops);
  EXPECT_TRUE(!res.empty());
  EXPECT_EQ(Str(res1.Replace(res)), "(TernaryOp(IsEqual(expr2, 2), 3, expr3) + expr1)");
  EXPECT_EQ(Str(res2.Replace(res)), "TernaryOp(IsEqual(expr2, 2), 3, expr3)");
  EXPECT_EQ(Str(res3.Replace(res)), "3");
}

TEST_F(TernaryOpsUtilsUnitTest, TestConcursiveReplaceVars2) {
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  Expr res1, res2, res3, expr1, expr2, expr3, expr4, expr5;
  SetupCommonTernaryOps(ternary_ops, res1, res2, res3, expr1, expr2, expr3, expr4, expr5);
  Expr res4 = CreateExpr("res4");
  TernaryOp ternary_op4 = TernaryOp(CondType::K_GE, expr4, expr5, expr1, expr3);
  ternary_op4.SetVariable(res4);
  ternary_ops[res4] = ternary_op4;
  auto res = ConcursiveReplaceVars(ternary_ops);
  EXPECT_TRUE(!res.empty());
  EXPECT_EQ(Str(res1.Replace(res)), "(TernaryOp(IsEqual(expr2, TernaryOp(expr4 <= expr5, expr1, expr3)), expr4, expr3) + TernaryOp(expr4 <= expr5, expr1, expr3))");
  EXPECT_EQ(Str(res2.Replace(res)), "TernaryOp(IsEqual(expr2, TernaryOp(expr4 <= expr5, expr1, expr3)), expr4, expr3)");
  EXPECT_EQ(Str(res3.Replace(res)), "TernaryOp(expr4 <= expr5, expr1, expr3)");
  EXPECT_EQ(Str(res4.Replace(res)), "TernaryOp(expr4 >= expr5, expr1, expr3)");
}

TEST_F(TernaryOpsUtilsUnitTest, TestConcursiveRelatedVars) {
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  Expr res1, res2, res3, expr1, expr2, expr3, expr4, expr5;
  SetupCommonTernaryOps(ternary_ops, res1, res2, res3, expr1, expr2, expr3, expr4, expr5);
  auto res = ConcursiveRelatedVars(ternary_ops);
  EXPECT_TRUE(!res.empty());
  EXPECT_EQ(GetVecString(res[res1]), "expr4,expr5,expr1,expr3,expr2,expr4,expr5,expr1,expr3,expr4,expr3,");
  EXPECT_EQ(GetVecString(res[res2]), "expr2,expr4,expr5,expr1,expr3,expr4,expr3,");
  EXPECT_EQ(GetVecString(res[res3]), "expr4,expr5,expr1,expr3,");
}

TEST_F(TernaryOpsUtilsUnitTest, TestUpdateReplaceVars) {
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  Expr rec = CreateExpr("rec");
  Expr res1 = CreateExpr("res1");
  Expr expr1 = CreateExpr("expr1");
  TernaryOp ternary_op1 = TernaryOp(expr1 + res1);
  std::vector<std::pair<Expr, Expr>> expr_map;
  expr_map.emplace_back(std::make_pair(res1, rec));
  ternary_op1.SetVariable(res1);
  ternary_ops[res1] = ternary_op1;
  ternary_ops[res1].UpdateRelatedVars(expr_map);
  auto related_maps = ternary_ops[res1].GetRelatedVars();
  EXPECT_TRUE(find(related_maps.begin(), related_maps.end(), rec) != related_maps.end());
}
} //namespace
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_TENARY_OP_H_
#define ATT_TENARY_OP_H_

#include <stack>
#include "base/base_types.h"
#include "framework/common/debug/ge_log.h"
#include "tenary_op.h"

namespace att {
enum class CondType : int32_t {
  K_EQ = 0,
  K_LT,
  K_GT,
  K_LE,
  K_GE,
};

class IfCase {
 public:
  explicit IfCase(const Expr &exp) : expr_(exp) {}
  IfCase(const CondType &type_cond, const Expr &left_cond,
        const Expr &right_cond, std::shared_ptr<IfCase> &&if_case_a,
        std::shared_ptr<IfCase> &&if_case_b) : cond_type_(type_cond), cond_left_(left_cond),
        cond_right_(right_cond) {
    choice_a_ = std::move(if_case_a);
    choice_b_ = std::move(if_case_b);
  }
  std::string GetStr() const;
  std::shared_ptr<IfCase> DeepCopy() const;
  void Replace(const std::vector<std::pair<Expr, Expr>> &replace_vars);
  void GetUsedArgs(std::vector<Expr> &used_args) const;
  bool IsLeaf() const { return choice_b_ == nullptr; }
  const Expr &GetExpr() const { return expr_; }
  const Expr &GetCondLeft() const { return cond_left_; }
  const Expr &GetCondRight() const { return cond_right_; }
  CondType GetCondType() const { return cond_type_; }
  std::shared_ptr<IfCase> GetChoiceA() const { return choice_a_; }
  std::shared_ptr<IfCase> GetChoiceB() const { return choice_b_; }
 private:
  CondType cond_type_{};
  Expr cond_left_;
  Expr cond_right_;
  Expr expr_;
  std::shared_ptr<IfCase> choice_a_;
  std::shared_ptr<IfCase> choice_b_;
};

class TenaryOp {
public:
  TenaryOp() = default;
  explicit TenaryOp(const Expr &expr);
  TenaryOp(const CondType &cond_type, const Expr &cond_left, 
        const Expr &cond_right, const Expr &choice_a, const Expr &choice_b);
  TenaryOp(const Expr &var, std::shared_ptr<IfCase> &&op, 
          const std::vector<Expr> &related);
  TenaryOp(const CondType &cond_type, const Expr &cond_left, const Expr &cond_right, 
          std::shared_ptr<IfCase> &&if_case_a, std::shared_ptr<IfCase> &&if_case_b);
  TenaryOp(const CondType &cond_type, const Expr &cond_left, const Expr &cond_right, 
          const TenaryOp &tenary_op_a, const TenaryOp &tenary_op_b);
  Expr GetVariable() const;
  std::string GetTenaryOpStr() const;
  std::vector<Expr> GetRelatedVars() const;
  TenaryOp DeepCopy() const;
  std::shared_ptr<IfCase> DeepCopyIfCase() const;
  void SetVariable(const Expr &expr);
  void SetDescription(const std::string &desc);
  std::string GetDescription() const;
  void UpdateRelatedVars(const std::vector<std::pair<Expr, Expr>> &replace_vars);
  void Replace(const std::vector<std::pair<Expr, Expr>> &replace_vars);
  void DecomposeNamedVars(const std::string &var_prefix, std::string &preamble, std::string &tenary_expr) const;
private:
  Expr variable_;
  std::string description_;  // 描述信息（包含形状），用于注释显示
  std::shared_ptr<IfCase> tenary_op_;
  std::vector<Expr> related_vars_;
};

std::vector<std::pair<Expr, Expr>> ConcursiveReplaceVars(const std::map<Expr, TenaryOp, ExprCmp> &tenary_ops); 
std::map<Expr, std::vector<Expr>, ExprCmp> ConcursiveRelatedVars(const std::map<Expr, TenaryOp, ExprCmp> &tenary_ops);
void GetPerfVar(const std::string &prefix, Expr &res, const std::map<Expr, TenaryOp, ExprCmp> &tenary_ops);
}  // namespace att
#endif  // ATT_TENARY_OP_H_



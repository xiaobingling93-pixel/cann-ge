/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_COMMON_OM2_CODEGEN_AST_AST_NODES_H
#define BASE_COMMON_OM2_CODEGEN_AST_AST_NODES_H

#include <string>
#include <cstring>

#include "ast_context.h"
#include "common/checker.h"
#include "common/om2/codegen/emitter/code_emitter.h"
#include "ge_common/ge_api_types.h"

namespace ge {
#define RAW_CODE_STMT(ctx, cpp_code) \
  RawCodeStmt::Create((ctx), {{RawCodeStmt::LANG_CPP, (cpp_code)}})

class CodeEmitter;

class AstNode {
 public:
  virtual ~AstNode() = default;
  virtual Status Accept(CodeEmitter &emitter, std::string &output) const = 0;
};

class Stmt : public AstNode {};
class Expr : public AstNode {};

class RawCodeStmt final : public Stmt {
 public:
  enum RawCodeLangType : uint8_t {
    LANG_CPP = 0,
    LANG_PYTHON = 1,
    LANG_INVALID,
    LANG_COUNT = LANG_INVALID,
  };

  static RawCodeStmt *Create(AstContext &ctx, const std::map<RawCodeLangType, std::string> &codes);

  StringRef GetCode(RawCodeLangType lang) const;

  Status Accept(CodeEmitter &emitter, std::string &output) const override {
    return emitter.Emit(*this, output);
  }

 private:
  explicit RawCodeStmt(const ArrayRef<StringRef> &codes) : codes_(codes) {}

 private:
  ArrayRef<StringRef> codes_;
};

}  // namespace ge
#endif  // BASE_COMMON_OM2_CODEGEN_AST_AST_NODES_H

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ast_nodes.h"

namespace ge {

RawCodeStmt *RawCodeStmt::Create(AstContext &ctx, const std::map<RawCodeLangType, std::string> &codes) {
  const auto code_array = ctx.AllocateMutableArray<StringRef>(LANG_COUNT);
  GE_ASSERT_TRUE((code_array.Data() != nullptr) && (!code_array.Empty()));
  for (size_t lang = LANG_CPP; lang < LANG_COUNT; ++lang) {
    const auto it = codes.find(static_cast<RawCodeLangType>(lang));
    if (it != codes.end()) {
      code_array[lang] = ctx.CopyString(it->second.c_str());
    }
  }
  void *mem = ctx.Allocate(sizeof(RawCodeStmt));
  GE_ASSERT_NOTNULL(mem);
  return new (mem) RawCodeStmt(code_array);
}

StringRef RawCodeStmt::GetCode(const RawCodeLangType lang) const {
  if (lang >= LANG_COUNT) {
    return {};
  }
  return codes_[static_cast<size_t>(lang)];
}

}  // namespace ge
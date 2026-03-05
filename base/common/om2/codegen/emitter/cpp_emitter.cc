/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cpp_emitter.h"

#include "common/om2/codegen/ast/ast_nodes.h"

namespace ge {
Status CppEmitter::Emit(const RawCodeStmt &node, std::string &output) {
  const auto &raw_code = node.GetCode(RawCodeStmt::LANG_CPP);
  GE_ASSERT_TRUE(!raw_code.Empty(), "The code of RawCodeStmt is empty.");
  output.assign(raw_code.Data(), raw_code.Length());
  return SUCCESS;
}

}  // namespace ge
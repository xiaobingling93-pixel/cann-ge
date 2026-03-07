/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stub/converter_stub.h"
#include "graph/node.h"
#include "common/omg_util/omg_util.h"

namespace gert {
ConverterStub::~ConverterStub() {
  Clear();
}
ConverterStub &ConverterStub::Register(const string &key, NodeConverterRegistry::NodeConverter func) {
  return Register(key, func, -1);
}
ConverterStub &ConverterStub::Register(const string &key, NodeConverterRegistry::NodeConverter func,
                                       int32_t placement) {
  if (backup_.count(key) == 0) {
    backup_.emplace(key, NodeConverterRegistry::GetInstance().FindRegisterData(key));
  }

  NodeConverterRegistry::GetInstance().Register(key, {func, placement});
  return *this;
}

void ConverterStub::Clear() {
  for (const auto &key_and_data : backup_) {
    if (key_and_data.second.exists) {
      NodeConverterRegistry::GetInstance().Register(key_and_data.first, key_and_data.second.backup);
    } else {
      // todo unregister
    }
  }
  backup_.clear();
}
}  // namespace gert

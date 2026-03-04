/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "es_tensor_holder.h"
#include "es_c_graph_builder.h"
#include "es_graph_builder.h"
#include "graph/gnode.h"
#include "common/checker.h"

#ifdef _MSC_VER
#define WEAK_SYMBOL __declspec(selectany)
#else
#define WEAK_SYMBOL __attribute__((weak))
#endif

#define DEFINE_WEAK_OP_FUNCTION(op_name)                                                      \
  WEAK_SYMBOL EsCTensorHolder *Es##op_name(const EsCTensorHolder *x1, const EsCTensorHolder *x2) {        \
    (void)x1;                                                                                 \
    (void)x2;                                                                                 \
    GE_ASSERT(false,                                                                          \
              "Weak symbol implementation called. This usually means:\n"                      \
              "1. The generated es codegen library is not linked at all\n"                    \
              "2. The generated es codegen library is not linked with --whole-archive or --no-as-needed\n"      \
              "3. The compile package and opp package versions are not compatible\n"          \
              "Please ensure:\n"                                                              \
              "- Link with the generated es library\n"                                \
              "- Use: -Wl,--whole-archive libes_generated.a (name of the library, may be different from the codegen library) -Wl,--no-whole-archive\n" \
              "- Use: -Wl,--no-as-needed libes_generated.so (name of the library, may be different from the codegen library) -Wl,--as-needed\n" \
              "- Install matching versions of compile package and opp package");              \
  }

#ifdef __cplusplus
extern "C" {
#endif

namespace {
// 使用宏定义生成所有操作符的弱符号函数
DEFINE_WEAK_OP_FUNCTION(Add)
DEFINE_WEAK_OP_FUNCTION(Sub)
DEFINE_WEAK_OP_FUNCTION(Mul)
DEFINE_WEAK_OP_FUNCTION(Div)
} // namespace

#ifdef __cplusplus
}
#endif


namespace ge::es {
int32_t EsTensorHolder::GetProducerOutIndex() const {
  GE_ASSERT_NOTNULL(tensor_holder_);
  return tensor_holder_->GetOutIndex();
}

GNode *EsTensorHolder::GetProducer() const {
  GE_ASSERT_NOTNULL(tensor_holder_);
  return &(tensor_holder_->GetProducer());
}

EsTensorHolder EsTensorHolder::operator+(const EsTensorHolder &other) const {
  return EsAdd(tensor_holder_, other.tensor_holder_);
}

EsTensorHolder EsTensorHolder::operator-(const EsTensorHolder &other) const {
  return EsSub(tensor_holder_, other.tensor_holder_);
}

EsTensorHolder EsTensorHolder::operator*(const EsTensorHolder &other) const {
  return EsMul(tensor_holder_, other.tensor_holder_);
}

EsTensorHolder EsTensorHolder::operator/(const EsTensorHolder &other) const {
  return EsDiv(tensor_holder_, other.tensor_holder_);
}

Status EsTensorHolder::AddControlEdge(const std::vector<EsTensorHolder> &ctrl_ins) const {
  GE_ASSERT_NOTNULL(tensor_holder_);
  return EsAddControlEdge(tensor_holder_, TensorsToEsCTensorHolders(ctrl_ins).data(), static_cast<int64_t>(ctrl_ins.size()));
}
}  // namespace ge::es
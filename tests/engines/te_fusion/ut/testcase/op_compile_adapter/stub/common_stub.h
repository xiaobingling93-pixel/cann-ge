/* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef _COMMON_STUB_H
#define _COMMON_STUB_H

#include <string>
#include "graph/ascend_string.h"
#include "graph/operator.h"
#include "base/registry/op_impl_space_registry_v2.h"

void GetAddGeneralizeFuncReturn(char **generalizeResult);

void GetMulGeneralizeFuncReturn(char **generalizeResult);

void GetMul1GeneralizeFuncReturn(char **generalizeResult);

void GetDivGeneralizeFuncReturn(char **generalizeResult);

void GetAscendQuantGeneralizeFuncReturn(char **generalizeResult);

void GetOpSpecificInfoReturn(char **opSpecificInfo);

ge::graphStatus CheckOpSupportedStub(ge::Operator &op, ge::AscendString &result);

ge::graphStatus CheckOpSupportedStubFalse(ge::Operator &op, ge::AscendString &result);

ge::graphStatus CheckOpSupportedStubUnknown(ge::Operator &op, ge::AscendString &result);

ge::graphStatus CheckOpSupportedStubInvalid(ge::Operator &op, ge::AscendString &result);

ge::graphStatus OpSelectTbeFormatStub(ge::Operator &op, ge::AscendString &result);

ge::graphStatus OpGetSpecificInfoStub(ge::Operator &op, ge::AscendString &result);

ge::graphStatus CheckOpSupportedV2Stub(const gert::OpCheckContext *context, ge::AscendString &result);

ge::graphStatus OpSelectTbeFormatV2Stub(const gert::OpCheckContext *context, ge::AscendString &result);

ge::graphStatus OpGetSupportInfoV2Stub(const gert::OpCheckContext *context, ge::AscendString &result);

ge::graphStatus OpGetSpecificInfoV2Stub(const gert::OpCheckContext *context, ge::AscendString &result);

ge::graphStatus CheckOpSupportedV2StubFail(const gert::OpCheckContext *context, ge::AscendString &result);

ge::graphStatus OpSelectTbeFormatV2StubFail(const gert::OpCheckContext *context, ge::AscendString &result);

ge::graphStatus OpGetSupportInfoV2StubFail(const gert::OpCheckContext *context, ge::AscendString &result);

ge::graphStatus OpGetSpecificInfoV2StubFail(const gert::OpCheckContext *context, ge::AscendString &result);

bool GenSimplifiedkeyStub(const ge::Operator &op, ge::AscendString &result);
#endif

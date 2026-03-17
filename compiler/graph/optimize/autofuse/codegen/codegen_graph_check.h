/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __CODEGEN_GRAPH_CHECK_H__
#define __CODEGEN_GRAPH_CHECK_H__

#include "common/checker.h"
#include "ascir.h"

namespace codegen {

ge::Status IsDataTypeSupported(const ::ascir::ImplGraph &graph);
ge::Status IsRepeatStrideValid(const ::ascir::ImplGraph &graph);
ge::Status IsGraphNodeValid(const ::ascir::ImplGraph &graph);
bool CheckGraphValidity(const ::ascir::ImplGraph &graph);

}  // namespace codegen

#endif  // __CODEGEN_GRAPH_CHECK_H__
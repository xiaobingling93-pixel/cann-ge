/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DFLOW_EXECUTOR_DATA_FLOW_EXECUTOR_UTILS_H_
#define DFLOW_EXECUTOR_DATA_FLOW_EXECUTOR_UTILS_H_

#include "ge/ge_api_types.h"
#include "graph/ge_tensor.h"
#include "framework/common/runtime_tensor_desc.h"

namespace ge {
class DataFlowExecutorUtils {
 public:
  static Status FillRuntimeTensorDesc(const GeTensorDesc &tensor_desc,
                                      RuntimeTensorDesc &runtime_tensor_desc,
                                      bool calc_size = true);
};
}

#endif  // DFLOW_EXECUTOR_DATA_FLOW_EXECUTOR_UTILS_H_

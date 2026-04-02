/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "notify.h"
#include "register/kernel_registry.h"
#include "utils/utils.h"
#include "core/debug/kernel_tracing.h"
#include "common/checker.h"

namespace gert {
/**
 * Get rts notify from logic notify id.
 *
 * input 0: TypedContinuousVector<rtNotifies>
 * input 1: notify_num
 * output 0: rtNotify_t
 * @return ge::GRAPH_SUCCESS when success, otherwise failed.
 */
ge::graphStatus CreateNotifies(KernelContext *context) {
  const auto logic_notifies_to_rts_notifies =
      context->GetInputPointer<TypedContinuousVector<aclrtNotify>>(EnToIdx(SplitRtNotifiesInput::kNotifies));
  const auto notify_num = context->GetInputPointer<int64_t>(EnToIdx(SplitRtNotifiesInput::kNotifyNum));
  GE_ASSERT_NOTNULL(notify_num);
  if (SECUREC_UNLIKELY((logic_notifies_to_rts_notifies == nullptr))) {
    return ge::GRAPH_FAILED;
  }
  if (SECUREC_UNLIKELY((*notify_num <= 0) ||
                       (logic_notifies_to_rts_notifies->GetSize() < static_cast<size_t>(*notify_num)))) {
    GELOGE(ge::PARAM_INVALID, "Failed to get rts notify from notify_num %ld, rts notify num is not enough(%zu)",
           *notify_num, logic_notifies_to_rts_notifies->GetSize());
    return ge::PARAM_INVALID;
  }

  for (size_t i = 0U; i < static_cast<size_t>(*notify_num); ++i) {
    auto rts_notify = context->GetOutputPointer<aclrtNotify>(i);
    if (SECUREC_UNLIKELY(rts_notify == nullptr)) {
      return ge::GRAPH_FAILED;
    }
    *rts_notify = logic_notifies_to_rts_notifies->GetData()[i];
    KERNEL_TRACE("Get rts notify %p from logical notify %lld", *rts_notify, i);
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(CreateNotifies).RunFunc(CreateNotifies);
}  // namespace gert

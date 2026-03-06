/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stream.h"
#include "register/kernel_registry.h"
#include "utils/utils.h"
#include "core/debug/kernel_tracing.h"
#include "common/checker.h"
#include "runtime/stream.h"

namespace gert {
ge::graphStatus SyncStream(KernelContext *context) {
  auto stream = context->GetInputValue<rtStream_t>(0);
  return DoRtStreamSyncWithTimeout(stream);
}
REGISTER_KERNEL(SyncStream).RunFunc(SyncStream);
/**
 * Get rts stream from logic stream id.
 *
 * input 0: TypedContinuousVector<rtStream_t>
 * input 1: stream_num
 * output 0: rtStream_t
 * @return ge::GRAPH_SUCCESS when success, otherwise failed.
 */
ge::graphStatus SplitRtStreams(KernelContext *context) {
  auto logic_streams_to_rts_stream =
      context->GetInputPointer<TypedContinuousVector<rtStream_t>>(EnToIdx(SplitRtStreamsInput::kStreams));
  auto stream_num = context->GetInputPointer<int64_t>(EnToIdx(SplitRtStreamsInput::kStreamNum));
  GE_ASSERT_NOTNULL(stream_num);
  if (SECUREC_UNLIKELY((logic_streams_to_rts_stream == nullptr))) {
    return ge::GRAPH_FAILED;
  }
  if (SECUREC_UNLIKELY((*stream_num <= 0) ||
                       (logic_streams_to_rts_stream->GetSize() < static_cast<size_t>(*stream_num)))) {
    GELOGE(ge::PARAM_INVALID, "Failed to get rts stream from stream_num %ld, rts stream num is not enough(%zu)",
           *stream_num, logic_streams_to_rts_stream->GetSize());
    return ge::PARAM_INVALID;
  }

  for (size_t i = 0U; i < static_cast<size_t>(*stream_num); ++i) {
    auto rts_stream = context->GetOutputPointer<rtStream_t>(i);
    if (SECUREC_UNLIKELY(rts_stream == nullptr)) {
      return ge::GRAPH_FAILED;
    }
    *rts_stream = logic_streams_to_rts_stream->GetData()[i];
    int32_t rt_stream_id = -1;
    (void)rtGetStreamId(*rts_stream, &rt_stream_id);
    KERNEL_TRACE("Get rts stream %p from logical stream %lld, rts stream_id: %d", *rts_stream, i, rt_stream_id);
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(SplitRtStreams).RunFunc(SplitRtStreams);
}  // namespace gert

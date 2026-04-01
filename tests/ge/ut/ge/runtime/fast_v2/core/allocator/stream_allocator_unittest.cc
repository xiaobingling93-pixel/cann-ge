/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "framework/runtime/stream_allocator.h"
#include "depends/runtime/src/runtime_stub.h"
#include "depends/ascendcl/src/ascendcl_stub.h"
#include "framework/common/ge_inner_error_codes.h"

namespace gert {
class StreamAllocatorUT : public testing::Test {};

TEST_F(StreamAllocatorUT, AcquireStreams_Success_StreamPoolEmpty) {
  StreamAllocator sa;
  size_t num = 22;
  auto streams = sa.AcquireStreams(num);
  ASSERT_NE(streams, nullptr);
  ASSERT_EQ(streams->GetSize(), num);
  for (size_t i = 1U; i < num; ++i) {
    ASSERT_NE(streams->GetData()[i], nullptr);
  }
}
TEST_F(StreamAllocatorUT, Acquire_Success_Expand) {
  StreamAllocator sa;
  ASSERT_NE(sa.AcquireStreams(11), nullptr);

  size_t num = 22;
  auto streams = sa.AcquireStreams(num);
  ASSERT_NE(streams, nullptr);
  ASSERT_EQ(streams->GetSize(), num);
  for (size_t i = 1U; i < num; ++i) {
    ASSERT_NE(streams->GetData()[i], nullptr);
  }
}
TEST_F(StreamAllocatorUT, Acquire_Success_Shrink) {
  StreamAllocator sa;
  ASSERT_NE(sa.AcquireStreams(33), nullptr);

  size_t num = 22;
  auto streams = sa.AcquireStreams(num);
  ASSERT_NE(streams, nullptr);
  ASSERT_EQ(streams->GetSize(), 33);
  for (size_t i = 1U; i < num; ++i) {
    ASSERT_NE(streams->GetData()[i], nullptr);
  }
}
TEST_F(StreamAllocatorUT, AcquireStreams_FirstOneIsReserved) {
  StreamAllocator sa;
  size_t num = 22;
  auto streams = sa.AcquireStreams(num);
  ASSERT_NE(streams, nullptr);
  ASSERT_EQ(streams->GetSize(), num);
  ASSERT_EQ(streams->GetData()[0], nullptr);
  streams->MutableData()[0] = (rtStream_t)(1);
  // 析构时不会释放第0个，所以即使是个非法stream也不会报错
}

TEST_F(StreamAllocatorUT, AcquireStreams_Fail_NoEnoughStreamResource) {
  uint32_t system_stream_cap = 0;
  ASSERT_EQ(aclrtGetStreamAvailableNum(&system_stream_cap), RT_ERROR_NONE);
  StreamAllocator sa;
  EXPECT_EQ(sa.AcquireStreams(system_stream_cap + 8U), nullptr);
}
}  // namespace gert
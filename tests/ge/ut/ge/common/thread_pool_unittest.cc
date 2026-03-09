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
#include "common/thread_pool/thread_pool.h"
#include "base/err_mgr.h"

namespace ge {
class UtestThreadPool : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};

TEST_F(UtestThreadPool, WithContextSuccess) {
  auto options_bk = GetThreadLocalContext().GetAllGraphOptions();
  std::map<std::string, std::string> options;
  options.emplace("test_pool", "1");
  GetThreadLocalContext().SetGraphOption(options);

  auto &omg_context = GetLocalOmgContext();
  auto work_stream_id_bk = error_message::GetErrMgrContext().work_stream_id;

  error_message::SetErrMgrContext({6});

  ThreadPool pool("test_pool", 1);
  auto fut = pool.commit([&omg_context]() -> Status {
    auto &omg_context_inner = GetLocalOmgContext();
    if (&omg_context_inner == &omg_context) {
      return FAILED;
    }
    auto options_inner = GetThreadLocalContext().GetAllGraphOptions();
    if (options_inner["test_pool"] != "1") {
      return FAILED;
    }
    if (error_message::GetErrMgrContext().work_stream_id != 6) {
      return FAILED;
    }
    return SUCCESS;
  });
  EXPECT_EQ(fut.get(), SUCCESS);

  // recover
  GetThreadLocalContext().SetGraphOption(options_bk);
  error_message::SetErrMgrContext({work_stream_id_bk});
}
}

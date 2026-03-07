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
#include <gmock/gmock.h>
#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "graph/build/run_context_util.h"

#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "common/omg_util/omg_util.h"

#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;

namespace ge {

class UtestRunContext : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};

TEST_F(UtestRunContext, InitMemInfo) {
    RunContextUtil run_util;
    uint8_t *data_mem_base = nullptr;
    uint64_t data_mem_size = 1024;
    std::map<int64_t, uint8_t *> mem_type_to_data_mem_base;
    std::map<int64_t, uint64_t> mem_type_to_data_mem_size;
    uint8_t *weight_mem_base = nullptr;
    uint64_t weight_mem_size = 256;
    EXPECT_EQ(run_util.InitMemInfo(data_mem_base, data_mem_size, mem_type_to_data_mem_base, mem_type_to_data_mem_size, weight_mem_base, weight_mem_size), PARAM_INVALID);
    uint8_t buf1[1024] = {0};
    data_mem_base = buf1;
    EXPECT_EQ(run_util.InitMemInfo(data_mem_base, data_mem_size, mem_type_to_data_mem_base, mem_type_to_data_mem_size, weight_mem_base, weight_mem_size), PARAM_INVALID);
    uint8_t buf2[1024] = {0};
    weight_mem_base = buf2;
    EXPECT_EQ(run_util.InitMemInfo(data_mem_base, data_mem_size, mem_type_to_data_mem_base, mem_type_to_data_mem_size, weight_mem_base, weight_mem_size), PARAM_INVALID);
}

TEST_F(UtestRunContext, CreateRunContext) {
    RunContextUtil run_util;
    Model model("name", "1.1");
    ComputeGraphPtr graph = nullptr;
    Buffer buffer(1024);
    uint64_t session_id = 0;
    AttrUtils::SetInt(&model, ATTR_MODEL_STREAM_NUM, 10);
    EXPECT_EQ(run_util.CreateRunContext(model, graph, buffer, session_id), PARAM_INVALID);
}

TEST_F(UtestRunContext, CreateRunContextWithNotifySuccess) {
  RunContextUtil run_util;
  Model model("name", "1.1");

  AttrUtils::SetInt(&model, ATTR_MODEL_LABEL_NUM, 1);
  AttrUtils::SetInt(&model, ATTR_MODEL_NOTIFY_NUM, 1);
  AttrUtils::SetInt(&model, ATTR_MODEL_EVENT_NUM, 1);
  AttrUtils::SetInt(&model, ATTR_MODEL_STREAM_NUM, 1);
  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("");
  Buffer buffer(1024);
  uint64_t session_id = 0;
  EXPECT_EQ(run_util.CreateRunContext(model, graph, buffer, session_id), SUCCESS);
}
} // namespace ge

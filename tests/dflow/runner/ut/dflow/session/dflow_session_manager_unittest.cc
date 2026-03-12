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
#include "dflow/compiler/session/dflow_session_manager.h"
#include "dflow/compiler/session/dflow_session_impl.h"

namespace ge {
namespace dflow {
class DFlowSessionManagerTest : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(DFlowSessionManagerTest, OperateGraphWithoutInit) {
  DFlowSessionManager flow_session_manger;
  std::map<std::string, std::string> options;
  uint64_t session_id = 0;
  EXPECT_EQ(flow_session_manger.CreateSession(options, session_id), nullptr);
  EXPECT_EQ(flow_session_manger.DestroySession(session_id), SUCCESS);
  EXPECT_EQ(flow_session_manger.GetSession(session_id), nullptr);
}

TEST_F(DFlowSessionManagerTest, GetSessionNotExist) {
  DFlowSessionManager flow_session_manger;
  std::map<std::string, std::string> options;
  options["ge.runFlag"] = "0";
  flow_session_manger.Initialize();
  uint64_t session_id = 0;
  EXPECT_NE(flow_session_manger.CreateSession(options, session_id), nullptr);
  EXPECT_EQ(flow_session_manger.GetSession(100), nullptr);
  EXPECT_EQ(flow_session_manger.DestroySession(100), GE_SESSION_NOT_EXIST);
}

class DFlowSessionImplTest : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(DFlowSessionImplTest, InitialFinalizeBasicTest) {
  std::map<std::string, std::string> options;
  ge::DFlowSessionImpl inner_session(0, options);
  EXPECT_EQ(inner_session.Initialize({}), SUCCESS);
  EXPECT_EQ(inner_session.Initialize({}), SUCCESS);
  EXPECT_EQ(inner_session.GetFlowModel(0), nullptr);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}
}
}
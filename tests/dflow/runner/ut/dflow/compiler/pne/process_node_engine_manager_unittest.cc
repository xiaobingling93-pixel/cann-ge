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

#include "api/gelib/gelib.h"
#include "framework/omg/ge_init.h"
#include "dflow/compiler/pne/process_node_engine_manager.h"
#include "graph/debug/ge_attr_define.h"
#include "dflow/compiler/pne/npu/npu_process_node_engine.h"
#include "dflow/compiler/pne/cpu/cpu_process_node_engine.h"

namespace ge {
class UtestProcessNodeEngineManager : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

class TestProcessNodeEngine : public NPUProcessNodeEngine {
  Status Initialize(const std::map<std::string, std::string> &options) override {
    (void) options;
    (void) engine_id_;
    return FAILED;
  }
};

TEST_F(UtestProcessNodeEngineManager, process_node_engine_init) {
  std::map<std::string, std::string> options;
  auto cpu_engine = std::make_shared<CPUProcessNodeEngine>();
  EXPECT_NE(cpu_engine, nullptr);
  auto npu_engine = std::make_shared<NPUProcessNodeEngine>();
  EXPECT_NE(npu_engine, nullptr);
  ProcessNodeEngineManager::GetInstance().init_flag_ = false;
  EXPECT_EQ(ProcessNodeEngineManager::GetInstance().RegisterEngine("HOST_CPU", cpu_engine, nullptr), SUCCESS);
  EXPECT_EQ(ProcessNodeEngineManager::GetInstance().RegisterEngine("NPU", npu_engine, nullptr), SUCCESS);
  EXPECT_EQ(ProcessNodeEngineManager::GetInstance().Initialize(options), SUCCESS);
  EXPECT_NE(ProcessNodeEngineManager::GetInstance().GetEngine(PNE_ID_CPU), nullptr);
  EXPECT_NE(ProcessNodeEngineManager::GetInstance().GetEngine(PNE_ID_NPU), nullptr);
}

TEST_F(UtestProcessNodeEngineManager, process_node_engine_init_fail) {
  auto engine = std::make_shared<TestProcessNodeEngine>();
  EXPECT_NE(engine, nullptr);
  ProcessNodeEngineManager::GetInstance().init_flag_ = false;
  EXPECT_EQ(ProcessNodeEngineManager::GetInstance().RegisterEngine("TEST", engine, nullptr), SUCCESS);
  std::map<std::string, std::string> options;
  EXPECT_EQ(ProcessNodeEngineManager::GetInstance().Initialize(options), FAILED);
}
}  // namespace ge

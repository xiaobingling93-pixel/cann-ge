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
#include "ge/ge_api.h"
#include "ge_graph_dsl/assert/check_utils.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "easy_graph/layout/graph_layout.h"
#include "easy_graph/layout/engines/graph_easy/graph_easy_executor.h"
#include "ge_running_env/dir_env.h"
#include "faker/space_registry_faker.h"
#include "init_ge.h"
using namespace std;
using namespace ge;

int main(int argc, char **argv) {
  setenv("GE_PROFILING_TO_STD_OUT", "1", true);
  // Init running dir env
  DirEnv::GetInstance().InitDir();

  InitGe();

  EG_NS::GraphEasyExecutor executor;
  EG_NS::GraphLayout::GetInstance().Config(executor, nullptr);

  GeRunningEnvFaker::BackupEnv();
  CheckUtils::init();
  gert::LoadDefaultSpaceRegistry();
  (void)gert::CreateSceneInfo();
  gert::CreateVersionInfo();
  testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  gert::DestroyVersionInfo();
  return ret;
}

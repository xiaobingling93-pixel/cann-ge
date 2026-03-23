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
#include <nlohmann/json.hpp>
#include <iostream>
#include <list>
#define protected public
#define private public
#include "platform/platform_info.h"
#include "common/platform_utils.h"
#undef private
#undef protected
#include "fusion_manager/fusion_manager.h"

using namespace std;
using namespace fe;

class fuison_manager_stest : public testing::Test
{
 protected:
  static void SetUpTestCase() {}
};

TEST_F(fuison_manager_stest, dsa_instance)
{
  map<string, string> options;
  options.emplace(ge::SOC_VERSION, "Ascend910B1");
  PlatformUtils::Instance().is_init_ = false;
  FusionManager &fm = FusionManager::Instance(kDsaCoreName);
  EXPECT_EQ(fm.Initialize(kDsaCoreName, options), SUCCESS);
  map<string, GraphOptimizerPtr> graph_optimizers;
  fm.GetGraphOptimizerObjs(graph_optimizers, kDsaCoreName);
  EXPECT_EQ(graph_optimizers.size(), 1);
}


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
#include "fe_llt_utils.h"
#define protected public
#define private public
#include "itf_handler/itf_handler.h"
#include "fusion_manager/fusion_manager.h"
#undef private
#undef protected

using namespace std;
using namespace fe;

class itfhandler_st : public testing::Test
{
protected:
  static void TearDownTestCase()
  {
    Finalize();
    InitWithSocVersion("Ascend910B1", "");
  }
// AUTO GEN PLEASE DO NOT MODIFY IT
};

TEST_F(itfhandler_st, initialize_success)
{
  std:: map<string, string> options;
  options.emplace("ge.socVersion", "Ascend910B1");
  Status ret = Initialize(options);
  EXPECT_EQ(ret, SUCCESS);
  ret = Finalize();
  EXPECT_EQ(ret, SUCCESS);
  fe::InitPlatformInfo("Ascend910B1", true);
  options["ge.bufferOptimize"] = "lx_optimize";
  ret = Initialize(options);
  EXPECT_EQ(ret, fe::SUCCESS);
  options["ge.bufferOptimize"] = "l2_optimize";
  ret = Initialize(options);
  EXPECT_EQ(ret, fe::SUCCESS);
  map<string, OpsKernelInfoStorePtr> op_kern_infos;
  GetOpsKernelInfoStores(op_kern_infos);
  EXPECT_EQ(op_kern_infos.size(), 2);
  map<string, GraphOptimizerPtr> graph_optimizers;
  GetGraphOptimizerObjs(graph_optimizers);
  EXPECT_EQ(graph_optimizers.size(), 2);
  EXPECT_EQ(ret, SUCCESS);
}


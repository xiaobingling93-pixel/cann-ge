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

#include "fe_llt_utils.h"
#include "fusion_manager/fusion_manager.h"
#include "itf_handler/itf_handler.h"

using namespace std;
using namespace fe;
using namespace ge;

class itfhandler_unittest : public testing::Test
{
protected:
  static void TearDownTestCase() {}
// AUTO GEN PLEASE DO NOT MODIFY IT
};

TEST_F(itfhandler_unittest, initialize_and_finalize) {
  Status ret = Finalize();
  EXPECT_EQ(ret, fe::SUCCESS);
  map<string, string> options;
  ret = Initialize(options);
  EXPECT_NE(ret, fe::SUCCESS);
  options.emplace("ge.socVersion", "Ascend910B1");
  EXPECT_EQ(fe::InitPlatformInfo("Ascend910B1", true), 0);
  options["ge.bufferOptimize"] = "lx_optimize";
  ret = Initialize(options);
  EXPECT_EQ(ret, fe::SUCCESS);
  options["ge.bufferOptimize"] = "l2_optimize";
  ret = Initialize(options);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(itfhandler_unittest, GetOpsKernelInfoStores_suc) {
  map<string, string> options;
  Status ret = Initialize(options);
  EXPECT_EQ(ret, fe::SUCCESS);
  map<string, OpsKernelInfoStorePtr> op_kern_infos;
  GetOpsKernelInfoStores(op_kern_infos);
  EXPECT_EQ(op_kern_infos.size(), 2);
}

TEST_F(itfhandler_unittest, get_graph_optimizer_objs_success)
{
  map<string, string> options;
  Status ret = Initialize(options);
  EXPECT_EQ(ret, fe::SUCCESS);
  map<string, GraphOptimizerPtr> graph_optimizers;
  GetGraphOptimizerObjs(graph_optimizers);
  EXPECT_EQ(graph_optimizers.size(), 2);
}

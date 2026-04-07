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
#include <cstdlib>

#include "fe_llt_utils.h"
#include "itf_handler/itf_handler.h"
#include "common/fe_log.h"
#include "common/util/json_util.h"
#include "platform/platform_info.h"
#include "common/configuration.h"
#include "common/platform_utils.h"
#include "platform_info.h"
#include "ge/ge_api_types.h"
#include "te_llt_utils.h"
#include "mmpa/mmpa_api.h"
#undef private
#undef protected

using namespace std;



int main(int argc, char **argv)
{
  string stub_cann_path = fe::GetCodeDir() + "/tests/engines/nn_engine/depends/CANN_910b_stub/cann";
  fe::EnvVarGuard cann_guard(MM_ENV_ASCEND_HOME_PATH, stub_cann_path.c_str());
  string stub_opp_path = fe::GetCodeDir() + "/tests/engines/nn_engine/depends/CANN_910b_stub/cann/opp";
  fe::EnvVarGuard opp_guard(MM_ENV_ASCEND_OPP_PATH, stub_opp_path.c_str());
  te::fusion::InitTbe();
  fe::InitPlatformInfo("Ascend910B1");
  testing::InitGoogleTest(&argc,argv);
  EXPECT_EQ(fe::InitPlatformInfo("Ascend910B1"), 0);
  cann_guard.Restore();
  opp_guard.Restore();
  int ret = RUN_ALL_TESTS();
  return ret;
}


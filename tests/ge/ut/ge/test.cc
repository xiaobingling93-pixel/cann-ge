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

#include "common/debug/log.h"
#include "ge_graph_dsl/assert/check_utils.h"
#include "faker/space_registry_faker.h"
#include "graph/ge_local_context.h"
#include "register/optimization_option_registry.h"

using namespace std;
using namespace ge;

int main(int argc, char **argv) {
  // init the logging
  testing::InitGoogleTest(&argc, argv);
  setenv("GE_PROFILING_TO_STD_OUT", "1", true);
  GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
  CheckUtils::init();
  int ret = RUN_ALL_TESTS();

  printf("finish ge ut\n");

  return ret;
}

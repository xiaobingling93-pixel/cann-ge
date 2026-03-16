/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_ST_TESTCASE_INIT_GE_H_
#define AIR_CXX_TESTS_ST_TESTCASE_INIT_GE_H_
#include <map>
#include <iostream>
#include "ge/ge_api.h"
#include "ge/ge_api_v2.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "compiler/session/dflow_api.h"
namespace ge {

inline void InitGe() {
  std::map<AscendString, AscendString> options;
  options[ge::OPTION_HOST_ENV_OS] = "linux";
  options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  auto init_status = ge::GEInitialize(options);
  if (init_status != SUCCESS) {
    std::cout << "ge init failed , ret code:" << init_status << std::endl;
  }

  const_cast<std::map<std::string, OpsKernelInfoStorePtr>&>(
      OpsKernelManager::GetInstance().GetAllOpsKernelInfoStores()).clear();

  GeRunningEnvFaker ge_env;
  ge_env.InstallDefault();
}

inline void ReInitGe() {
  // init the logging
  ge::GEFinalizeV2();
  std::map<AscendString, AscendString> options;
  options[ge::OPTION_HOST_ENV_OS] = "linux";
  options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  auto init_status = ge::GEInitializeV2(options);
  if (init_status != SUCCESS) {
    std::cout << "ge init failed , ret code:" << init_status << std::endl;
  }

  const_cast<std::map<std::string, OpsKernelInfoStorePtr>&>(
      OpsKernelManager::GetInstance().GetAllOpsKernelInfoStores()).clear();

  GeRunningEnvFaker ge_env;
  ge_env.InstallDefault();
}

inline void ReInitDFlow() {
  dflow::DFlowFinalize();
  std::map<AscendString, AscendString> options;
  auto init_status = dflow::DFlowInitialize(options);
  if (init_status != SUCCESS) {
    std::cout << "dflow init failed , ret code:" << init_status << std::endl;
  }
}
}
#endif //AIR_CXX_TESTS_ST_TESTCASE_INIT_GE_H_

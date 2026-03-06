/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTCASE_DUMP_UTILS_DUMP_TEST_FIXTURE_H
#define TESTCASE_DUMP_UTILS_DUMP_TEST_FIXTURE_H

// Headers whose private fields are needed.
#include "macro_utils/dt_public_scope.h"
#include "graph/load/model_manager/model_manager.h"                     // ModelManager::GetInstance().max_model_id_
#include "macro_utils/dt_public_unscope.h"

// Headers for setting up test framework.
#include <gtest/gtest.h>
#include "framework/ge_runtime_stub/include/common/dump_checker.h"      // DumpCheckRuntimeStub
#include "ge_running_env/ge_running_env_faker.h"                        // GeRunningEnvFaker
#include "ge/st/stubs/utils/mock_ops_kernel_builder.h"                     // MockForGenerateTask
#include "ge/st/stubs/utils/taskdef_builder.h"                             // AiCoreTaskDefBuilder
#include "common/opskernel/ops_kernel_info_types.h"
// Public interfaces.
#include "ge/ge_api.h"

namespace ge {
namespace {
void MockGenerateTask() {
  auto aicore_func = [](const ge::Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    auto op_desc = node.GetOpDesc();
    op_desc->SetOpKernelLibName("AIcoreEngine");
    size_t arg_size = 100;
    std::vector<uint8_t> args(arg_size, 0);
    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    auto kernel_info = task_def.mutable_kernel();
    kernel_info->set_args(args.data(), args.size());
    kernel_info->set_args_size(arg_size);
    kernel_info->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    kernel_info->set_kernel_name(node.GetName());
    kernel_info->set_block_dim(1);
    uint16_t args_offset[2] = {0};
    kernel_info->mutable_context()->set_args_offset(args_offset, 2 * sizeof(uint16_t));
    kernel_info->mutable_context()->set_op_index(node.GetOpDesc()->GetId());

    tasks.emplace_back(task_def);
    return SUCCESS;
  };

  MockForGenerateTask("AIcoreEngine", aicore_func);
}
}
template <bool dynamic>
class DumpST : public ::testing::Test {
public:
  static void SetUpTestSuite() {
    const std::map<AscendString, AscendString> options = {
      { OPTION_HOST_ENV_OS, "linux" },
      { OPTION_HOST_ENV_CPU, "x86_64" },
    };
    ASSERT_EQ(GEInitialize(options), SUCCESS);
    GeRunningEnvFaker().InstallDefault();
    MockForGenerateTask("AiCoreLib", [](const Node &node, RunContext &, std::vector<domi::TaskDef> &tasks) {
      bool is_ffts = false;
      if (AttrUtils::GetBool(node.GetOpDesc(), "_ffts_plus", is_ffts) && is_ffts) {
        return SUCCESS;
      }
      tasks.emplace_back(AiCoreTaskDefBuilder(node).BuildTask(dynamic));
      return SUCCESS;
    });
  }
  static void TearDownTestSuite() {
    GeRunningEnvFaker().Reset();
    ASSERT_EQ(GEFinalize(), SUCCESS);
  }
protected:
  void SetUp() override {
    auto dump_checker_stub = std::make_shared<DumpCheckRuntimeStub>();
    RuntimeStub::SetInstance(dump_checker_stub);
    checker_ = &dump_checker_stub->GetDumpChecker();

    // Reset at each iteration for the convenience of ModelId checking.
    ModelManager::GetInstance().max_model_id_ = 1;
    MockGenerateTask();
  }
  void TearDown() override {
    OpsKernelBuilderRegistry::GetInstance().Unregister("AIcoreEngine");
    RuntimeStub::Reset();
  }
  DumpChecker *checker_;
};
} // namespace ge

#endif

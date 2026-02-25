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
#include "ge_graph_dsl/graph_dsl.h"

#include "macro_utils/dt_public_scope.h"
#include "framework/common/types.h"
#include "common/preload/model/pre_davinci_model.h"
#include "common/preload/model/pre_model_types.h"
#include "graph/buffer/buffer_impl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_local_context.h"
#include "common/model/ge_model.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "common/profiling/profiling_manager.h"
#include "graph/manager/graph_var_manager.h"
#include "framework/common/taskdown_common.h"
#include "common/preload/task_info/pre_generate_task_registry.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
class PreDavinciModelUnittest : public testing::Test {};

TEST_F(PreDavinciModelUnittest, GetEngineNameAndOpDescFail) {
  PreDavinciModel pre_davinci;
  std::string name;
  OpDescPtr op_desc;
  pre_davinci.Assign(nullptr);
  EXPECT_NE(pre_davinci.GetEngineNameAndOpDesc(static_cast<EngineType>(2U),
      domi::TaskDef(), name, op_desc), SUCCESS);
}

TEST_F(PreDavinciModelUnittest, GetHostFuncEngine) {
  PreDavinciModel pre_davinci;
  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType:: MODEL_TASK_KERNEL));

  domi::KernelDef kernel_def;
  domi::KernelContext context;
  context.set_kernel_type(static_cast<uint32_t>(ccKernelType::CCE_AI_CPU));

  *kernel_def.mutable_context() = context;
  *task_def.mutable_kernel() = kernel_def;

  std::string engine_name;
  OpDescPtr op_desc;

  EXPECT_NE(pre_davinci.GetEngineNameAndOpDesc(
      static_cast<EngineType>(EngineType::kNanoEngine),
      task_def,
      engine_name,
      op_desc),
      SUCCESS);

  EXPECT_TRUE(engine_name.empty());
  const auto func = PreGenerateTaskRegistry::GetInstance().FindPreGenerateTask(engine_name);
  ASSERT_EQ(func, nullptr);
}

TEST_F(PreDavinciModelUnittest, DoTaskSinkDefaultTaskSuccess) {
  uint32_t mem_offset = 0U;
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", "Data")->EDGE(0, 0)->NODE("add_n", "AddN"));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);
  AttrUtils::SetInt(graph, "globalworkspace_type", 1);
  AttrUtils::SetInt(graph, "globalworkspace_size", 1);
  SetKnownOpKernel(graph, mem_offset);

  ProfilingProperties::Instance().is_load_profiling_ = true;

  std::vector<uint64_t> weights_value(64, 1024);
  size_t weight_size = weights_value.size() * sizeof(uint64_t);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);

  ge_model->SetWeight(Buffer::CopyFrom((uint8_t *)weights_value.data(), weight_size));
  const size_t logic_var_base = VarManager::Instance(graph->GetSessionID())->GetVarMemLogicBase();
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 3));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_TASK_GEN_VAR_ADDR, logic_var_base));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, logic_var_base));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120));

  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  {
    const auto &node = graph->FindNode("add_n");
    const auto &op_desc = node->GetOpDesc();

    TBEKernelStore tbe_kernel_store;
    const auto kernel = MakeShared<OpKernelBin>("test", std::vector<char>(64, 0));
    tbe_kernel_store.AddTBEKernel(kernel);
    ge_model->SetTBEKernelStore(tbe_kernel_store);

    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_EVENT_RECORD));
    task_def->set_stream_id(op_desc->GetStreamId());
    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_op_index(op_desc->GetId());
  }

  PreDavinciModel pre_davinci;
  pre_davinci.Assign(ge_model);
  pre_davinci.InitNodes(graph);
  EXPECT_EQ(pre_davinci.DoTaskSink(ge::EngineType::kDefaultEngine), SUCCESS);
}

TEST_F(PreDavinciModelUnittest, GenerateTaskDesc) {
  PreDavinciModel pre_davinci;
  std::string engine_name;
  OpDescPtr op_desc;
  EXPECT_NE(pre_davinci.GetEngineNameAndOpDesc(static_cast<EngineType>(EngineType::kNanoEngine),
      domi::TaskDef(), engine_name, op_desc), SUCCESS);
  EXPECT_TRUE(engine_name.empty());
  const auto func = PreGenerateTaskRegistry::GetInstance().FindPreGenerateTask(engine_name);
  ASSERT_EQ(func, nullptr);
}
}

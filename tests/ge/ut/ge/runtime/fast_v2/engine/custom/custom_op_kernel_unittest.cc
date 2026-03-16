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
#include "engine/custom/converter/custom_node_converter.h"
#include "common/share_graph.h"
#include "check/executor_statistician.h"
#include "framework/runtime/model_v2_executor.h"
#include "faker/ge_model_builder.h"
#include "lowering/model_converter.h"
#include "faker/fake_value.h"
#include "framework/runtime/executor_option/multi_thread_executor_option.h"
#include "graph/custom_op_factory.h"
#include "graph/custom_op.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/task_producer_factory.h"
#include "core/executor/multi_thread_topological/executor/schedule/config/task_scheduler_config.h"
#include "core/executor/multi_thread_topological/execution_data/multi_thread_execution_data.h"
#include "operator_reg.h"
#include "common/executor_tracer_on.h"
#include "common/global_variables/diagnose_switch.h"

using namespace ge;
using namespace gert::bg;

namespace gert {
namespace kernel {
class CustomNodeKernelUT : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};
class TestBaseCustomOp : public EagerExecuteOp {
 public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    auto input_tensor0 = ctx->GetInputTensor(0);
    GE_ASSERT_NOTNULL(input_tensor0);
    auto input_shape0 = input_tensor0->GetShape().GetStorageShape();
    GE_ASSERT_TRUE(input_shape0.GetDimNum() == 1);
    GE_ASSERT_TRUE(input_shape0.GetDim(0) == 2048);
    auto input_tensor1 = ctx->GetInputTensor(1);
    GE_ASSERT_NOTNULL(input_tensor1);
    auto input_shape1 = input_tensor1->GetShape().GetStorageShape();
    GE_ASSERT_TRUE(input_shape1.GetDimNum() == 1);
    GE_ASSERT_TRUE(input_shape1.GetDim(0) == 2048);
    auto input_tensor2 = ctx->GetInputTensor(2);
    GE_ASSERT_NOTNULL(input_tensor2);
    auto input_shape2 = input_tensor2->GetShape().GetStorageShape();
    GE_ASSERT_TRUE(input_shape2.GetDimNum() == 1);
    GE_ASSERT_TRUE(input_shape2.GetDim(0) == 2048);
    auto workspaces = ctx->MallocWorkSpace(1024);
    GE_ASSERT_NOTNULL(workspaces);
    auto output_tensor = ctx->MallocOutputTensor(0, StorageShape({2048}, {2048}),
        StorageFormat(FORMAT_ND, FORMAT_ND, ExpandDimsType()), DT_FLOAT);
    GE_ASSERT_NOTNULL(output_tensor);
    return SUCCESS;
  }
};

REG_OP(CustomOp)
  .INPUT(x1, TensorType::BasicType())
  .INPUT(x2, TensorType::BasicType())
  .INPUT(x3, TensorType::BasicType())
  .OUTPUT(y, TensorType::BasicType())
  .OP_END_FACTORY_REG(CustomOp)

TEST_F(CustomNodeKernelUT, custom_op_kernel_execute_test) {
  auto graph = ShareGraph::BuildCustomOpGraph();
  graph->TopologicalSorting();
  CustomOpFactory::RegisterCustomOpCreator("CustomOp", []()->std::unique_ptr<BaseCustomOp> {
    return std::make_unique<TestBaseCustomOp>();
  });
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();
  bg::ValueHolder::PopGraphFrame();
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, {});
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  TaskProducerFactory::GetInstance().SetProducerType(TaskProducerType::KERNEL);
  ASSERT_EQ(TaskProducerFactory::GetInstance().GetProducerType(), TaskProducerType::KERNEL);
  auto model_executor = ModelV2Executor::Create(exe_graph,
      ExecutorOption(ExecutorType::kTopologicalPriority), ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Load(), GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 3);
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  auto ess = StartExecutorStatistician(model_executor);
  // 第一次执行，无缓存，全部算子调用tiling_func
  ess->Clear();
  // 打开info日志验证traceprinter
  ExecutorTracerOn executor_tracer_on;  // 开启trace
  ge::diagnoseSwitch::EnableProfiling({gert::ProfilingType::kTaskTime, gert::ProfilingType::kDevice});
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            GRAPH_SUCCESS);
  ge::diagnoseSwitch::DisableProfiling();
  rtStreamDestroy(stream);
}
}
}
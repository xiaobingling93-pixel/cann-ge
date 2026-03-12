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
#include <gmock/gmock.h>
#include <string.h>

#include "dflow/executor/flow_model_manager.h"
#include "common/profiling/profiling_manager.h"
#include "common/helper/om_file_helper.h"
#include "common/op/ge_op_utils.h"
#include "graph/ops_stub.h"
#include "graph/manager/graph_manager.h"
#include "graph/passes/graph_builder_utils.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "depends/runtime/src/runtime_stub.h"
#include "dflow/inc/data_flow/model/flow_model_helper.h"
#include "framework/common/helper/model_helper.h"

using namespace std;
using namespace testing;

namespace ge {
class UtestFlowModelManager : public testing::Test {
 protected:
  void SetUp() {
    RTS_STUB_SETUP();
  }
  void TearDown() override {
    RTS_STUB_TEARDOWN();
  }
};

namespace {
class MockModelDeployer : public ModelDeployer {
 public:
  Status DeployModel(const FlowModelPtr &flow_model,
                     DeployResult &deploy_result) override {
    deploy_result.model_id = 1;
    return SUCCESS;
  }

  MOCK_METHOD1(Undeploy, Status(uint32_t));
};

class MockExchangeService : public ExchangeService {
 public:
  Status CreateQueue(int32_t device_id,
                     const string &name,
                     const MemQueueAttr &mem_queue_attr,
                     uint32_t &queue_id) override {
    return SUCCESS;
  }
  Status DestroyQueue(int32_t device_id, uint32_t queue_id) override {
    return SUCCESS;
  }
  Status Enqueue(int32_t device_id, uint32_t queue_id, const void *data, size_t size,
                 const ControlInfo &control_info) override {
    return SUCCESS;
  }
  Status Enqueue(int32_t device_id, uint32_t queue_id, size_t size, rtMbufPtr_t m_buf,
                 const ControlInfo &control_info) override {
    return SUCCESS;
  }
  Status Enqueue(int32_t device_id, uint32_t queue_id, size_t size, const FillFunc &fill_func,
                 const ControlInfo &control_info) override {
    return SUCCESS;
  }
  Status Enqueue(const int32_t device_id, const uint32_t queue_id, const std::vector<BuffInfo> &buffs,
                 const ControlInfo &control_info) override {
    return SUCCESS;
  }
  Status EnqueueMbuf(int32_t device_id, uint32_t queue_id, rtMbufPtr_t m_buf, int32_t timeout) override {
    return SUCCESS;
  }
  Status Dequeue(int32_t device_id, uint32_t queue_id, void *data, size_t size, ControlInfo &control_info) override {
    return SUCCESS;
  }
  Status DequeueMbufTensor(const int32_t device_id, const uint32_t queue_id, std::shared_ptr<AlignedPtr> &aligned_ptr,
                           const size_t size, ControlInfo &control_info) override {
    return SUCCESS;
  }
  Status DequeueTensor(const int32_t device_id, const uint32_t queue_id, GeTensor &tensor,
                       ControlInfo &control_info) override {
    return 0;
  }
  Status DequeueMbuf(int32_t device_id, uint32_t queue_id, rtMbufPtr_t *m_buf, int32_t timeout) override {
    return 0;
  }
  void ResetQueueInfo(const int32_t device_id, const uint32_t queue_id) override {
    return;
  }
};

class MockExecutionRuntime : public ExecutionRuntime {
 public:
  Status Initialize(const map<std::string, std::string> &options) override {
    return SUCCESS;
  }
  Status Finalize() override {
    return SUCCESS;
  }
  ModelDeployer &GetModelDeployer() override {
    return model_deployer_;
  }
  ExchangeService &GetExchangeService() override {
    return exchange_service_;
  }

 private:
  MockModelDeployer model_deployer_;
  MockExchangeService exchange_service_;
};
} // namespace

TEST_F(UtestFlowModelManager, TestLoadFlowModel) {
  ExecutionRuntime::SetExecutionRuntime(make_shared<MockExecutionRuntime>());
  auto graph = std::make_shared<ComputeGraph>("root-graph");
  auto root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(root_model->Initialize(graph), SUCCESS);
  auto ge_model = MakeShared<ge::GeModel>();
  auto model_task_def = MakeShared<domi::ModelTaskDef>();
  model_task_def->set_version("test_v100_r001");
  ge_model->SetModelTaskDef(model_task_def);
  ge_model->SetName(graph->GetName());
  ge_model->SetGraph(graph);
  root_model->SetModelName(graph->GetName());	
  root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);	
  auto flow_root_model = ge::MakeShared<ge::FlowModel>();
  EXPECT_NE(flow_root_model, nullptr);
  flow_root_model->root_graph_ = std::make_shared<ComputeGraph>("root-graph");
  flow_root_model->model_relation_.reset(new ModelRelation());
  bool is_unknown_shape = false;
  auto ret = root_model->CheckIsUnknownShape(is_unknown_shape);
  EXPECT_EQ(ret, SUCCESS);
  ModelBufferData model_buffer_data{};
  const auto model_save_helper =
    ModelSaveHelperFactory::Instance().Create(OfflineModelFormat::OM_FORMAT_DEFAULT);
  EXPECT_NE(model_save_helper, nullptr);
  model_save_helper->SetSaveMode(false);
  ret = model_save_helper->SaveToOmRootModel(root_model, "NoUse", model_buffer_data, is_unknown_shape);
  EXPECT_EQ(ret, SUCCESS);
  ModelData model_data{};
  model_data.model_data = model_buffer_data.data.get();
	model_data.model_len = model_buffer_data.length;
  (void)flow_root_model->AddSubModel(FlowModelHelper::ToPneModel(model_data, graph));

  FlowModelManager model_manager;

  uint32_t model_id = 666;
  ASSERT_EQ(model_manager.Unload(model_id), SUCCESS);
  ASSERT_EQ(model_manager.LoadFlowModel(model_id, flow_root_model), SUCCESS);
  ASSERT_NE(model_manager.GetHeterogeneousModelExecutor(model_id), nullptr);

  ASSERT_EQ(model_manager.Unload(model_id), SUCCESS);
  ASSERT_TRUE(model_manager.GetHeterogeneousModelExecutor(model_id) == nullptr);
  ExecutionRuntime::SetExecutionRuntime(nullptr);
}

TEST_F(UtestFlowModelManager, TestUnloadHeterogeneousModel) {
  ExecutionRuntime::SetExecutionRuntime(nullptr);
  FlowModelManager model_manager;
  model_manager.heterogeneous_model_map_[666] =
      make_shared<HeterogeneousModelExecutor>(std::make_shared<FlowModel>(), DeployResult{});
  uint32_t model_id = 666;
  ASSERT_NE(model_manager.Unload(model_id), SUCCESS);  // execution runtime not set

  ExecutionRuntime::SetExecutionRuntime(make_shared<MockExecutionRuntime>());
  auto &model_deploy = (MockModelDeployer &) ExecutionRuntime::GetInstance()->GetModelDeployer();
  EXPECT_CALL(model_deploy, Undeploy).Times(1).WillOnce(testing::Return(SUCCESS));
  EXPECT_EQ(model_manager.Unload(model_id), SUCCESS);
  EXPECT_EQ(model_manager.GetHeterogeneousModelExecutor(model_id), nullptr);
  ExecutionRuntime::SetExecutionRuntime(nullptr);
}


TEST_F(UtestFlowModelManager, TestExecuteHeterogeneousModel_InvalidModelId) {
  FlowModelManager model_manager;
  std::vector<GeTensor> output_tensors;
  ASSERT_EQ(model_manager.ExecuteFlowModel(555, {}, output_tensors), PARAM_INVALID);
}

TEST_F(UtestFlowModelManager, GetRootGraphAndFlowModelNullptr) {
  auto flow_model = std::make_shared<FlowModel>();
  FlowModelManager mm;
  mm.heterogeneous_model_map_[100] = nullptr;
  EXPECT_EQ(mm.GetFlowModelByModelId(100), nullptr);
  DeployResult deploy_result;
  auto executor = MakeShared<HeterogeneousModelExecutor>(flow_model, deploy_result);
  mm.heterogeneous_model_map_[100] = executor;
  EXPECT_EQ(mm.GetFlowModelByModelId(100), flow_model);
}
}  // namespace ge

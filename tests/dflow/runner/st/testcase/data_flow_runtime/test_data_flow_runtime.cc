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
#include <fstream>
#include "nlohmann/json.hpp"
#include "depends/mmpa/src/mmpa_stub.h"
#include "utils/mock_execution_runtime.h"
#include "dflow/compiler/pne/process_node_engine_manager.h"
#include "dflow/compiler/pne/udf/udf_process_node_engine.h"
#include "dflow/executor/flow_msg.h"
#include "ge/ge_api.h"
#include "flow_graph/data_flow.h"

using namespace testing;

namespace ge {
class MockMmpaDeployer : public ge::MmpaStubApiGe {
  public:
  void *DlOpen(const char *file_name, int32_t mode) {
    if (std::string(file_name) == "libmodel_deployer.so") {
      return (void *) 0x8888;
    }
    return dlopen(file_name, mode);
  }

  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "InitializeHeterogeneousRuntime") {
      return (void *) &InitializeExecutionRuntimeStub;
    }
    return dlsym(handle, func_name);
  }

  int32_t DlClose(void *handle) override {
    if (handle == (void *) 0x8888) {
      return 0;
    }
    return dlclose(handle);
  }

  int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
    (void)strncpy_s(realPath, realPathLen, path, strlen(path));
    return 0;
  }
};

class DataFlowRuntimeTest : public testing::Test {
  void SetUp() {
    PrepareForBuiltInUdf();
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaDeployer>());
    GEFinalize();
    std::map<AscendString, AscendString> init_options = {
      {"ge.resourceConfigPath", "some_path"}
    };
    EXPECT_EQ(ge::GEInitialize(init_options), SUCCESS);
  }

  void TearDown() {
    remove("./builtin_udf_config.json");
    MmpaStub::GetInstance().Reset();
    GEFinalize();
  }

  static void PrepareForBuiltInUdf() {
    std::string builtin_udf_config_file = "./builtin_udf_config.json";
    {
      nlohmann::json builtin_udf_cfg_json = {
                                          {"input_num", 2},
                                          {"output_num", 2},
                                          {"built_in_flow_func", true},
                                          {"heavy_load", false},
                                          {"visible_device_enable", false},
                                          {"func_list", {{{"func_name", "_BuiltIn_func1"}}}}};
      std::ofstream json_file(builtin_udf_config_file);
      json_file << builtin_udf_cfg_json << std::endl;
    }
  }
};

namespace {
Graph BuildBuiltinDataFlowGraph() {
  auto data0 = dflow::FlowData("Data0", 0);
  auto data1 = dflow::FlowData("Data1", 1);
  auto node0 = dflow::FlowNode("node0", 2, 2).SetInput(0,data0).SetInput(1, data1);

  // function pp
  auto builtin_udf_pp = dflow::FunctionPp("builtin_udf_pp").SetCompileConfig("./builtin_udf_config.json");
  node0.AddPp(builtin_udf_pp);

  std::vector<dflow::FlowOperator> inputsOperator{data0, data1};
  std::vector<dflow::FlowOperator> outputsOperator{node0};

  dflow::FlowGraph flow_graph("graph");
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  return flow_graph.ToGeGraph();
}

Status MockDeployModel(const FlowModelPtr &flow_model, DeployResult &deploy_result) {
  deploy_result.input_queue_attrs = {{1, 0, 0}, {2, 0, 0}};
  deploy_result.output_queue_attrs = {{5, 0, 0}};
  deploy_result.dev_abnormal_callback = []() { return SUCCESS; };
  return SUCCESS;
}
}

TEST_F(DataFlowRuntimeTest, TestTensorFlowMsg) {
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto graph = BuildBuiltinDataFlowGraph();
  // mock deploy model
  auto execution_runtime = (ExecutionRuntimeStub *)ExecutionRuntime::GetInstance();
  EXPECT_CALL(execution_runtime->GetModelDeployerStub(), DeployModel).WillRepeatedly(
      testing::Invoke(MockDeployModel));

  std::map<AscendString, AscendString> session_options;
  Session session(session_options);
  std::map<AscendString, AscendString> graph_options = {};
  ASSERT_EQ(session.AddGraph(1, graph, graph_options), SUCCESS);

  std::vector<FlowMsgPtr> inputs;
  auto input0 = FlowBufferFactory::AllocTensorMsg({1, 1}, DT_INT32);
  // test set flow msg
  input0->SetFlowFlags(1U);
  input0->SetRetCode(2);
  input0->SetStartTime(3UL);
  input0->SetEndTime(4UL);
  input0->SetTransactionId(5UL);
  input0->SetUserData("test_user_data", 14);
  auto input0_tensor = input0->GetTensor();
  std::vector<int32_t> data0 = {static_cast<int32_t>(1)};
  input0_tensor->SetData(reinterpret_cast<const uint8_t *>(data0.data()), data0.size() * sizeof(int32_t));
  inputs.emplace_back(input0);

  std::vector<int32_t> data1 = {static_cast<int32_t>(2)};
  Tensor input1_tensor(TensorDesc(Shape({1, 1}), FORMAT_ND, DT_INT32),
                       reinterpret_cast<const uint8_t *>(data1.data()), data1.size() * sizeof(int32_t));
  auto input1 = FlowBufferFactory::ToFlowMsg(input1_tensor);
  inputs.emplace_back(input1);

  // mock enqueue and dequeue
  EXPECT_CALL(execution_runtime->GetExchangeServiceStub(), EnqueueMbuf).WillRepeatedly(Return(SUCCESS));
  auto mock_dequeue_mbuf =
    [input0](int32_t device_id, uint32_t queue_id, rtMbufPtr_t *m_buf, int32_t timeout) -> Status {
    auto input_msg = std::dynamic_pointer_cast<FlowMsgBase>(input0);
    *m_buf = input_msg->MbufCopyRef();
    return SUCCESS;
  };
  EXPECT_CALL(execution_runtime->GetExchangeServiceStub(), DequeueMbuf).WillRepeatedly(
      testing::Invoke(mock_dequeue_mbuf));

  ASSERT_EQ(session.FeedDataFlowGraph(1, inputs, 2000), SUCCESS);
  std::vector<FlowMsgPtr> outputs;
  ASSERT_EQ(session.FetchDataFlowGraph(1, outputs, 2000), SUCCESS);

  ASSERT_EQ(outputs.size(), 1U);
  // test get flow msg
  ASSERT_EQ(outputs[0]->GetMsgType(), MsgType::MSG_TYPE_TENSOR_DATA);
  ASSERT_EQ(outputs[0]->GetFlowFlags(), 1U);
  ASSERT_EQ(outputs[0]->GetRetCode(), 2);
  ASSERT_EQ(outputs[0]->GetStartTime(), 3UL);
  ASSERT_EQ(outputs[0]->GetEndTime(), 4UL);
  ASSERT_EQ(outputs[0]->GetTransactionId(), 5UL);
  char user_data[15] = {};
  ASSERT_EQ(outputs[0]->GetUserData(user_data, 14), SUCCESS);
  ASSERT_EQ(std::string(user_data), "test_user_data");
  auto output_tensor = outputs[0]->GetTensor();
  ASSERT_EQ(output_tensor->GetSize(), data0.size() * sizeof(int32_t));
  ASSERT_EQ(reinterpret_cast<const int32_t *>(output_tensor->GetData())[0], 1);

  // mock undeploy model
  EXPECT_CALL(execution_runtime->GetModelDeployerStub(), Undeploy).WillRepeatedly(Return(SUCCESS));
}

TEST_F(DataFlowRuntimeTest, TestRawDataFlowMsg) {
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto graph = BuildBuiltinDataFlowGraph();
  // mock deploy model
  auto execution_runtime = (ExecutionRuntimeStub *)ExecutionRuntime::GetInstance();
  EXPECT_CALL(execution_runtime->GetModelDeployerStub(), DeployModel).WillRepeatedly(
      testing::Invoke(MockDeployModel));

  std::map<AscendString, AscendString> session_options;
  Session session(session_options);
  std::map<AscendString, AscendString> graph_options = {};
  ASSERT_EQ(session.AddGraph(1, graph, graph_options), SUCCESS);

  std::vector<FlowMsgPtr> inputs;
  auto input0 = FlowBufferFactory::AllocRawDataMsg(sizeof(int32_t) * 2);
  void *input0_data_ptr = nullptr;
  uint64_t input0_data_size = 0UL;
  ASSERT_EQ(input0->GetRawData(input0_data_ptr, input0_data_size), SUCCESS);
  ASSERT_NE(input0_data_ptr, nullptr);
  ASSERT_EQ(input0_data_size, sizeof(int32_t) * 2);
  // init input0 flow msg
  reinterpret_cast<int32_t *>(input0_data_ptr)[0] = 1;
  reinterpret_cast<int32_t *>(input0_data_ptr)[1] = 2;
  input0->SetMsgType(static_cast<MsgType>(1065));
  inputs.emplace_back(input0);

  std::vector<int32_t> data1 = {3, 4};
  RawData input1_raw_data = {};
  input1_raw_data.addr = &data1[0];
  input1_raw_data.len = data1.size() * sizeof(int32_t);
  auto input1 = FlowBufferFactory::ToFlowMsg(input1_raw_data);
  inputs.emplace_back(input1);

  // mock enqueue and dequeue
  EXPECT_CALL(execution_runtime->GetExchangeServiceStub(), EnqueueMbuf).WillRepeatedly(Return(SUCCESS));
  auto mock_dequeue_mbuf =
    [input0](int32_t device_id, uint32_t queue_id, rtMbufPtr_t *m_buf, int32_t timeout) -> Status {
    auto input_msg = std::dynamic_pointer_cast<FlowMsgBase>(input0);
    *m_buf = input_msg->MbufCopyRef();
    return SUCCESS;
  };
  EXPECT_CALL(execution_runtime->GetExchangeServiceStub(), DequeueMbuf).WillRepeatedly(
      testing::Invoke(mock_dequeue_mbuf));

  ASSERT_EQ(session.FeedDataFlowGraph(1, inputs, 2000), SUCCESS);
  std::vector<FlowMsgPtr> outputs;
  ASSERT_EQ(session.FetchDataFlowGraph(1, outputs, 2000), SUCCESS);

  ASSERT_EQ(outputs.size(), 1U);
  // test get flow msg
  ASSERT_EQ(outputs[0]->GetMsgType(), static_cast<MsgType>(1065));
  void *output_data_ptr = nullptr;
  uint64_t output_data_size = 0UL;
  ASSERT_EQ(outputs[0]->GetRawData(output_data_ptr, output_data_size), SUCCESS);
  ASSERT_NE(output_data_ptr, nullptr);
  ASSERT_EQ(output_data_size, sizeof(int32_t) * 2);
  ASSERT_EQ(reinterpret_cast<int32_t *>(output_data_ptr)[0], 1);
  ASSERT_EQ(reinterpret_cast<int32_t *>(output_data_ptr)[1], 2);

  // mock undeploy model
  EXPECT_CALL(execution_runtime->GetModelDeployerStub(), Undeploy).WillRepeatedly(Return(SUCCESS));
}

TEST_F(DataFlowRuntimeTest, TestEmptyFlowMsg) {
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto graph = BuildBuiltinDataFlowGraph();
  // mock deploy model
  auto execution_runtime = (ExecutionRuntimeStub *)ExecutionRuntime::GetInstance();
  EXPECT_CALL(execution_runtime->GetModelDeployerStub(), DeployModel).WillRepeatedly(
      testing::Invoke(MockDeployModel));

  std::map<AscendString, AscendString> session_options;
  Session session(session_options);
  std::map<AscendString, AscendString> graph_options = {};
  ASSERT_EQ(session.AddGraph(1, graph, graph_options), SUCCESS);

  std::vector<FlowMsgPtr> inputs;
  auto input0 = FlowBufferFactory::AllocEmptyDataMsg(MsgType::MSG_TYPE_TENSOR_DATA);
  inputs.emplace_back(input0);

  std::vector<int32_t> data1 = {3, 4};
  RawData input1_raw_data = {};
  input1_raw_data.addr = &data1[0];
  input1_raw_data.len = data1.size() * sizeof(int32_t);
  auto input1 = FlowBufferFactory::ToFlowMsg(input1_raw_data);
  inputs.emplace_back(input1);

  // mock enqueue and dequeue
  EXPECT_CALL(execution_runtime->GetExchangeServiceStub(), EnqueueMbuf).WillRepeatedly(Return(SUCCESS));
  auto mock_dequeue_mbuf =
    [input0](int32_t device_id, uint32_t queue_id, rtMbufPtr_t *m_buf, int32_t timeout) -> Status {
    auto input_msg = std::dynamic_pointer_cast<FlowMsgBase>(input0);
    *m_buf = input_msg->MbufCopyRef();
    return SUCCESS;
  };
  EXPECT_CALL(execution_runtime->GetExchangeServiceStub(), DequeueMbuf).WillRepeatedly(
      testing::Invoke(mock_dequeue_mbuf));

  ASSERT_EQ(session.FeedDataFlowGraph(1, inputs, 2000), SUCCESS);
  std::vector<FlowMsgPtr> outputs;
  ASSERT_EQ(session.FetchDataFlowGraph(1, outputs, 2000), SUCCESS);

  ASSERT_EQ(outputs.size(), 1U);
  // test get flow msg
  ASSERT_EQ(outputs[0]->GetMsgType(), MsgType::MSG_TYPE_TENSOR_DATA);
  ASSERT_EQ(outputs[0]->GetTensor(), nullptr);

  // mock undeploy model
  EXPECT_CALL(execution_runtime->GetModelDeployerStub(), Undeploy).WillRepeatedly(Return(SUCCESS));
}
}  // namespace ge

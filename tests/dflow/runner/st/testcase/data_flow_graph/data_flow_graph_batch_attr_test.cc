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
#include <fstream>
#include "flow_graph/data_flow.h"
#include "ge/ge_api.h"
#include "depends/runtime/src/runtime_stub.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/ge_global_options.h"
#include "framework/common/ge_types.h"
#include "dflow/compiler/pne/process_node_engine_manager.h"
#include "dflow/compiler/pne/udf/udf_process_node_engine.h"
#include "init_ge.h"
#include "macro_utils/dt_public_scope.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "macro_utils/dt_public_unscope.h"
#include "common/env_path.h"
#include "compiler/session/dflow_api.h"

using namespace testing;

namespace ge {
namespace {
class MockExchangeService : public ExchangeService {
 public:
  Status CreateQueue(const int32_t device_id,
                     const string &name,
                     const MemQueueAttr &mem_queue_attr,
                     uint32_t &queue_id) override {
    return 0;
  }
  Status DestroyQueue(const int32_t device_id, const uint32_t queue_id) override {
    return 0;
  }
  Status Enqueue(const int32_t device_id,
                 const uint32_t queue_id,
                 const void *const data,
                 const size_t size,
                 const ControlInfo &control_info) override {
    return 0;
  }
  Status Enqueue(int32_t device_id, uint32_t queue_id, size_t size, rtMbufPtr_t m_buf,
                 const ControlInfo &control_info) override {
    return 0;
  }
  Status Enqueue(const int32_t device_id,
                 const uint32_t queue_id,
                 const size_t size,
                 const FillFunc &fill_func,
                 const ControlInfo &control_info) override {
    return 0;
  }
  Status Enqueue(const int32_t device_id, const uint32_t queue_id, const std::vector<BuffInfo> &buffs,
                 const ControlInfo &control_info) override {
    return 0;
  }
  Status EnqueueMbuf(int32_t device_id, uint32_t queue_id, rtMbufPtr_t m_buf, int32_t timeout) override {
    return 0;
  }
  Status Dequeue(const int32_t device_id,
                 const uint32_t queue_id,
                 void *const data,
                 const size_t size,
                 ControlInfo &control_info) override {
    return 0;
  }
  Status DequeueMbufTensor(const int32_t device_id, const uint32_t queue_id, std::shared_ptr<AlignedPtr> &aligned_ptr,
                           const size_t size, ControlInfo &control_info) override {
    return 0;
  }
  Status DequeueTensor(const int32_t device_id,
                       const uint32_t queue_id,
                       GeTensor &tensor,
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

class MockModelDeployer : public ModelDeployer {
 public:
  Status DeployModel(const FlowModelPtr &flow_model,
                     DeployResult &deploy_result) override {
    return SUCCESS;
  }
  Status Undeploy(uint32_t model_id) override {
    return SUCCESS;
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
}

class ConvertBatchAttrToUdfPassTest : public testing::Test {
 protected:
  void SetUp() {
    ExecutionRuntime::instance_ = ge::MakeShared<MockExecutionRuntime>();
    ExecutionRuntime::handle_ = (void *)0xffffffff;
    {
      auto &global_options_mutex = GetGlobalOptionsMutex();
      const std::lock_guard<std::mutex> lock(global_options_mutex);
      auto &global_options = GetMutableGlobalOptions();
      global_options[OPTION_NUMA_CONFIG] =
          R"({"cluster":[{"cluster_nodes":[{"is_local":true, "item_list":[{"item_id":0}], "node_id":0, "node_type":"TestNodeType1"}]}],"item_def":[{"aic_type":"[DAVINCI_V100:10]","item_type":"","memory":"[DDR:80GB]","resource_type":"Ascend"}],"node_def":[{"item_type":"","links_mode":"TCP:128Gb","node_type":"TestNodeType1","resource_type":"X86","support_links":"[ROCE]"}]})";
    }
    std::string st_dir_path = ge::PathUtils::Join({ge::EnvPath().GetAirBasePath(), "/tests/dflow/runner/st/"});
    auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config.json";
    setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
    dflow::DFlowInitialize({});
  }
  void TearDown() {
    ExecutionRuntime::instance_ = nullptr;
    ExecutionRuntime::handle_ = nullptr;
    dflow::DFlowFinalize();
    unsetenv("RESOURCE_CONFIG_PATH");
  }
};

namespace {
Graph BuildGeGraph() {
  DEF_GRAPH(graph_def) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3}).Build("add");
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE(add));
    ADD_OUTPUT(add, 0);
  };
  return ToGeGraph(graph_def);
}
}  // namespace

TEST_F(ConvertBatchAttrToUdfPassTest, TimeBatch_CountBatch_Run_Success) {
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto data0 = dflow::FlowData("Data0", 0);
  auto data1 = dflow::FlowData("Data1", 1);
  auto node0 = dflow::FlowNode("Node0", 2, 1).SetInput(0, data0).SetInput(1, data1);
  auto graph_pp = dflow::GraphPp("graph_pp", BuildGeGraph);
  node0.AddPp(graph_pp);
  dflow::TimeBatch time_batch = {0};
  time_batch.time_window = -1;
  dflow::DataFlowInputAttr input0_attr = {dflow::DataFlowAttrType::TIME_BATCH, &time_batch};
  node0.MapInput(0, graph_pp, 0, {input0_attr});
  dflow::CountBatch count_batch = {0};
  count_batch.batch_size = 1;
  dflow::DataFlowInputAttr input1_attr = {dflow::DataFlowAttrType::COUNT_BATCH, &count_batch};
  node0.MapInput(1, graph_pp, 1, {input1_attr});

  std::vector<dflow::FlowOperator> inputsOperator{data0, data1};
  std::vector<dflow::FlowOperator> outputsOperator{node0};

  dflow::FlowGraph flow_graph("FlowGraph");
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  auto graph = GraphUtilsEx::GetComputeGraph(flow_graph.ToGeGraph());

  EXPECT_EQ(graph->GetDirectNode().size(), 4);
  std::map<AscendString, AscendString> options = {{"ge.runFlag", "0"}};
  dflow::DFlowSession session(options);
  session.AddGraph(1, flow_graph);
  std::vector<Tensor> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNode().size(), 6);
  auto node_node0_time_batch = graph->FindNode("Node0__BuiltIn_TimeBatch_0");
  ASSERT_NE(node_node0_time_batch, nullptr);
  auto node_data0 = graph->FindNode("Data0");
  auto node_node0 = graph->FindNode("Node0");
  ASSERT_EQ(node_data0->GetOutDataAnchor(0)->IsLinkedWith(node_node0_time_batch->GetInDataAnchor(0)), true);
  ASSERT_EQ(node_node0_time_batch->GetOutDataAnchor(0)->IsLinkedWith(node_node0->GetInDataAnchor(0)), true);
  auto node_node0_count_batch = graph->FindNode("Node0__BuiltIn_CountBatch_1");
  ASSERT_NE(node_node0_count_batch, nullptr);
  auto node_data1 = graph->FindNode("Data1");
  ASSERT_EQ(node_data1->GetOutDataAnchor(0)->IsLinkedWith(node_node0_count_batch->GetInDataAnchor(0)), true);
  ASSERT_EQ(node_node0_count_batch->GetOutDataAnchor(0)->IsLinkedWith(node_node0->GetInDataAnchor(1)), true);
}

TEST_F(ConvertBatchAttrToUdfPassTest, TimeBatch_CountBatch_with_catch_exception) {
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto data0 = dflow::FlowData("Data0", 0);
  auto data1 = dflow::FlowData("Data1", 1);
  auto node0 = dflow::FlowNode("Node0", 2, 1).SetInput(0, data0).SetInput(1, data1);
  auto graph_pp = dflow::GraphPp("graph_pp", BuildGeGraph);
  node0.AddPp(graph_pp);
  dflow::TimeBatch time_batch = {0};
  time_batch.time_window = -1;
  dflow::DataFlowInputAttr input0_attr = {dflow::DataFlowAttrType::TIME_BATCH, &time_batch};
  node0.MapInput(0, graph_pp, 0, {input0_attr});
  dflow::CountBatch count_batch = {0};
  count_batch.batch_size = 1;
  dflow::DataFlowInputAttr input1_attr = {dflow::DataFlowAttrType::COUNT_BATCH, &count_batch};
  node0.MapInput(1, graph_pp, 1, {input1_attr});

  std::vector<dflow::FlowOperator> inputsOperator{data0, data1};
  std::vector<dflow::FlowOperator> outputsOperator{node0};

  dflow::FlowGraph flow_graph("FlowGraph");
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  auto graph = GraphUtilsEx::GetComputeGraph(flow_graph.ToGeGraph());
  (void)AttrUtils::SetBool(graph, "_enable_exception_catch", true);
  (void)AttrUtils::SetInt(graph, "_inputs_align_max_cache_num", 1024);
  (void)AttrUtils::SetInt(graph, "_inputs_align_timeout", 30 * 1000);
  (void)AttrUtils::SetBool(graph, "_inputs_align_dropout", true);
  EXPECT_EQ(graph->GetDirectNode().size(), 4);
  std::map<AscendString, AscendString> options = {{"ge.runFlag", "0"}};
  dflow::DFlowSession session(options);
  session.AddGraph(1, flow_graph);
  std::vector<Tensor> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, FAILED);
}

// batch deploy with attr node
TEST_F(ConvertBatchAttrToUdfPassTest, TimeBatch_CountBatch_with_deploy_info) {
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto data0 = dflow::FlowData("Data0", 0);
  auto data1 = dflow::FlowData("Data1", 1);
  auto node0 = dflow::FlowNode("Node0", 2, 1).SetInput(0, data0).SetInput(1, data1);
  auto graph_pp = dflow::GraphPp("graph_pp", BuildGeGraph);
  node0.AddPp(graph_pp);
  dflow::TimeBatch time_batch = {0};
  time_batch.time_window = -1;
  dflow::DataFlowInputAttr input0_attr = {dflow::DataFlowAttrType::TIME_BATCH, &time_batch};
  node0.MapInput(0, graph_pp, 0, {input0_attr});
  dflow::CountBatch count_batch = {0};
  count_batch.batch_size = 1;
  dflow::DataFlowInputAttr input1_attr = {dflow::DataFlowAttrType::COUNT_BATCH, &count_batch};
  node0.MapInput(1, graph_pp, 1, {input1_attr});

  std::vector<dflow::FlowOperator> inputsOperator{data0, data1};
  std::vector<dflow::FlowOperator> outputsOperator{node0};

  dflow::FlowGraph flow_graph("FlowGraph");
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  auto graph = GraphUtilsEx::GetComputeGraph(flow_graph.ToGeGraph());

  EXPECT_EQ(graph->GetDirectNode().size(), 4);

  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["Node0"],
            "logic_device_list":"0:0:1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  std::map<AscendString, AscendString> options = {{"ge.runFlag", "0"}};
  dflow::DFlowSession session(options);
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  session.AddGraph(1, flow_graph, graph_options);
  std::vector<Tensor> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNode().size(), 6);
  auto node_node0_time_batch = graph->FindNode("Node0__BuiltIn_TimeBatch_0");
  ASSERT_NE(node_node0_time_batch, nullptr);
  
  std::string logic_device_id;
  EXPECT_TRUE(AttrUtils::GetStr(node_node0_time_batch->GetOpDesc(), ATTR_NAME_LOGIC_DEV_ID, logic_device_id));
  EXPECT_EQ(logic_device_id, "0:0:1");

  auto node_node0_count_batch = graph->FindNode("Node0__BuiltIn_CountBatch_1");
  ASSERT_NE(node_node0_count_batch, nullptr);
  logic_device_id.clear();
  EXPECT_TRUE(AttrUtils::GetStr(node_node0_count_batch->GetOpDesc(), ATTR_NAME_LOGIC_DEV_ID, logic_device_id));
  EXPECT_EQ(logic_device_id, "0:0:1");
}
}  // namespace ge

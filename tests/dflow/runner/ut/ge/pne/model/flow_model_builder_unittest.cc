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
#include "ge_graph_dsl/graph_dsl.h"
#include "api/gelib/gelib.h"
#include "common/util/mem_utils.h"
#include "common/share_graph.h"
#include "dflow/compiler/pne/process_node_engine.h"
#include "dflow/compiler/model/flow_model_builder.h"
#include "dflow/compiler/pne/cpu/cpu_process_node_engine.h"
#include "dflow/compiler/pne/npu/npu_process_node_engine.h"
#include "dflow/compiler/pne/udf/udf_process_node_engine.h"
#include "dflow/compiler/pne/process_node_engine_manager.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/operator_factory_impl.h"
#include "proto/dflow.pb.h"
#include "dflow/inc/data_flow/flow_graph/model_pp.h"
#include "dflow/flow_graph/data_flow_attr_define.h"
#include "flow_graph/data_flow.h"
#include "graph/ge_context.h"
#include "graph/ge_global_options.h"
#include "dflow/compiler/model/flow_model_cache.h"
#include "common/env_path.h"
#include "dflow/inc/data_flow/model/flow_model_helper.h"
#include "framework/common/helper/model_save_helper.h"
#include "graph/manager/graph_var_manager.h"

using namespace testing;

namespace ge {
namespace {
ComputeGraphPtr FakeComputeGraph(const string &graph_name) {
  DEF_GRAPH(graph1) {
    auto data_0 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("FakeOpNpu").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("_arg_0", data_0)->NODE("fused_op1", fake_type2_op1)->NODE("Node_Output", net_output));
  };

  auto root_graph = ToComputeGraph(graph1);
  auto fused_op1 = root_graph->FindNode("fused_op1");
  (void) root_graph->SetGraphOutNodesInfo({{fused_op1, 0}});
  root_graph->SetName(graph_name);
  root_graph->SetSessionID(0);
  AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0_1");

  auto op_desc = root_graph->FindNode("Node_Output")->GetOpDesc();
  std::vector<std::string> src_name{"out"};
  op_desc->SetSrcName(src_name);
  std::vector<int64_t> src_index{0};
  op_desc->SetSrcIndex(src_index);
  return root_graph;
}

PneModelPtr BuildPneModel(const string &name, ComputeGraphPtr graph) {
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  GE_ASSERT_SUCCESS(ge_root_model->Initialize(graph));
  auto ge_model = MakeShared<ge::GeModel>();
  auto model_task_def = MakeShared<domi::ModelTaskDef>();
  model_task_def->set_version("test_v100_r001");
  ge_model->SetModelTaskDef(model_task_def);
  ge_model->SetName(name);
  ge_model->SetGraph(graph);
  ge_root_model->SetModelName(name);
  ge_root_model->SetSubgraphInstanceNameToModel(name, ge_model);
  bool is_unknown_shape = false;
  GE_ASSERT_SUCCESS(ge_root_model->CheckIsUnknownShape(is_unknown_shape));
  ModelBufferData model_buffer_data{};
  const auto model_save_helper =
    ModelSaveHelperFactory::Instance().Create(OfflineModelFormat::OM_FORMAT_DEFAULT);
  model_save_helper->SetSaveMode(false);
  GE_ASSERT_SUCCESS(model_save_helper->SaveToOmRootModel(ge_root_model, name, model_buffer_data, is_unknown_shape));
  ModelData model_data{};
  model_data.model_data = model_buffer_data.data.get();
	model_data.model_len = model_buffer_data.length;
  auto graph_model = FlowModelHelper::ToPneModel(model_data, graph, PNE_ID_NPU);
  graph_model->SetLogicDeviceId("0:0:1");
  graph_model->SetModelName(name);
  return graph_model;
}
}

class FlowModelBuilderTest : public testing::Test {
 protected:
  void SetUp() {
    OperatorFactoryImpl::RegisterInferShapeFunc("Data", [](Operator &op) {return GRAPH_SUCCESS;});
    OperatorFactoryImpl::RegisterInferShapeFunc("PartitionedCall", [](Operator &op) {return GRAPH_SUCCESS;});
    OperatorFactoryImpl::RegisterInferShapeFunc("FakeOpNpu", [](Operator &op) {return GRAPH_SUCCESS;});
    OperatorFactoryImpl::RegisterInferShapeFunc("Add", [](Operator &op) {return GRAPH_SUCCESS;});
    OperatorFactoryImpl::RegisterInferShapeFunc("FakeOpHostCpu", [](Operator &op) {return GRAPH_SUCCESS;});
    OperatorFactoryImpl::RegisterInferShapeFunc("StaticFoo", [](Operator &op) {return GRAPH_SUCCESS;});
    OperatorFactoryImpl::RegisterInferShapeFunc("NetOutput", [](Operator &op) {return GRAPH_SUCCESS;});

    SetLocalOmgContext(domi::GetContext());
    std::string cmd = "mkdir -p temp; cd temp; touch libtest.so";
    (void)system(cmd.c_str());
    std::ofstream cmakefile("./temp/CMakeLists.txt");
    {
      cmakefile << "cmake_minimum_required(VERSION 3.5)\n";
      // Prevent cmake from testing the toolchain
      cmakefile << "set(CMAKE_C_COMPILER_FORCED TRUE)\n";
      cmakefile << "set(CMAKE_CXX_COMPILER_FORCED TRUE)\n";
      cmakefile << "project(test)\n";
      cmakefile << "set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})\n";
      cmakefile << "execute_process(\n";
      cmakefile << "\tCOMMAND cp ${BASE_DIR}/libtest.so ${RELEASE_DIR}\n";
      cmakefile << ")\n";
      cmakefile << "unset(CMAKE_C_COMPILER_FORCED)\n";
      cmakefile << "unset(CMAKE_CXX_COMPILER_FORCED)\n";
    }
  }
  void TearDown() {
    OperatorFactoryImpl::operator_infershape_funcs_->erase("Data");
    OperatorFactoryImpl::operator_infershape_funcs_->erase("PartitionedCall");
    OperatorFactoryImpl::operator_infershape_funcs_->erase("FakeOpNpu");
    OperatorFactoryImpl::operator_infershape_funcs_->erase("Add");
    OperatorFactoryImpl::operator_infershape_funcs_->erase("FakeOpHostCpu");
    OperatorFactoryImpl::operator_infershape_funcs_->erase("StaticFoo");
    OperatorFactoryImpl::operator_infershape_funcs_->erase("NetOutput");
    MmpaStub::GetInstance().Reset();
    std::string cmd = "rm -rf temp";
    (void)system(cmd.c_str());
  }
  const std::string run_data_path = PathUtils::Join({EnvPath().GetAirBasePath(), "tests/ge/st/st_run_data/"});
};

namespace {
std::function<Status(const std::map<std::string, std::string> &options,
                     ComputeGraphPtr &compute_graph)> mock_gen_deploy_plan;

class MockNpuEngineImpl : public ProcessNodeEngineImpl {
 public:
  Status BuildGraph(uint32_t graph_id, ComputeGraphPtr &compute_graph,
                    const std::map<std::string, std::string> &options, const std::vector<GeTensor> &inputs,
                    PneModelPtr &model) override {
    model = BuildPneModel(compute_graph->GetName(), compute_graph);
    return SUCCESS;
  }
};

Status InitializeHeterogeneousRuntime(const std::map<std::string, std::string> &options) {
  return SUCCESS;
}
int32_t g_so_addr = 0;
class MockMmpa : public MmpaStubApiGe {
 public:
  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "InitializeHeterogeneousRuntime") {
      return (void *) &InitializeHeterogeneousRuntime;
    }
    return MmpaStubApiGe::DlSym(handle, func_name);
  }

  void *DlOpen(const char *file_name, int32_t mode) override {
    if (string("libmodel_deployer.so") == file_name) {
      return (void *) &g_so_addr;
    }
    return MmpaStubApiGe::DlOpen(file_name, mode);
  }
  int32_t DlClose(void *handle) override {
    if (handle == &g_so_addr) {
      return 0;
    }
    return MmpaStubApiGe::DlClose(handle);
  }
};

/**
 * @brief check flow model is flattened.
 * if flattened, submodel has no submodel.
 * @param flow_model flow model
 * @return int32_t is flattened
 */
bool CheckFlowModelIsFlattened(FlowModelPtr flow_model) {
  const auto &submodels = flow_model->GetSubmodels();
  for (const auto &submodel : submodels) {
    if (!submodel.second->GetSubmodels().empty()) {
      return false;
    }
  }
  return true;
}
}  // namespace

TEST_F(FlowModelBuilderTest, BuildHeterogeneousModel_EnginePartitioned) {
  DEF_GRAPH(dynamic_graph) {
    auto data_0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .Attr(ge::ATTR_NAME_PROCESS_NODE_ENGINE_ID, "NPU")
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op3 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .Attr(ge::ATTR_NAME_PROCESS_NODE_ENGINE_ID, "NPU")
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("_arg_0", data_0)->NODE("fused_op1", fake_type2_op1)
              ->NODE("fused_op3", fake_type2_op3)->NODE("Node_Output", net_output));
  };

  auto root_graph = ToComputeGraph(dynamic_graph);
  (void) AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  uint32_t graph_id = 1U;
  root_graph->SetGraphID(graph_id);

  auto npu_engine = std::make_shared<NPUProcessNodeEngine>();
  npu_engine->SetImpl(std::make_shared<MockNpuEngineImpl>());
  FlowModelBuilder flow_model_builder;
  flow_model_builder.process_node_engines_[PNE_ID_NPU] = npu_engine;

  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  FlowModelPtr flow_model;
  EXPECT_EQ(flow_model_builder.BuildModel(graph, {}, {}, flow_model), SUCCESS);
  EXPECT_TRUE(CheckFlowModelIsFlattened(flow_model));
}

TEST_F(FlowModelBuilderTest, BuildHeterogeneousModel_ParallelPartitioned) {
  DEF_GRAPH(dynamic_graph) {
    auto data_0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .Attr(ge::ATTR_NAME_PROCESS_NODE_ENGINE_ID, "NPU")
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op3 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .Attr(ge::ATTR_NAME_PROCESS_NODE_ENGINE_ID, "NPU")
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("_arg_0", data_0)->NODE("fused_op1", fake_type2_op1)
              ->NODE("fused_op3", fake_type2_op3)->NODE("Node_Output", net_output));
  };

  auto root_graph = ToComputeGraph(dynamic_graph);
  (void) AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  uint32_t graph_id = 1U;
  root_graph->SetGraphID(graph_id);
  auto npu_engine = std::make_shared<NPUProcessNodeEngine>();
  npu_engine->SetImpl(std::make_shared<MockNpuEngineImpl>());
  FlowModelBuilder flow_model_builder;
  flow_model_builder.process_node_engines_[PNE_ID_NPU] = npu_engine;

  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  FlowModelPtr flow_model;
  EXPECT_EQ(flow_model_builder.BuildModel(graph, {}, {}, flow_model), SUCCESS);
  EXPECT_TRUE(CheckFlowModelIsFlattened(flow_model));
}

// flow model no need flatten.
TEST_F(FlowModelBuilderTest, BuildHeterogeneousModel_NoPartition) {
  DEF_GRAPH(dynamic_graph) {
    auto data_0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op2 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op3 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("_arg_0", data_0)->NODE("fused_op1", fake_type2_op1)->NODE("fused_op2", fake_type2_op2)
              ->NODE("fused_op3", fake_type2_op3)->NODE("Node_Output", net_output));
  };
  auto root_graph = ToComputeGraph(dynamic_graph);
  auto output_node = root_graph->FindNode("Node_Output");
  output_node->GetOpDesc()->SetSrcIndex({0});
  output_node->GetOpDesc()->SetSrcName({"FakeOpNpu"});
  (void) AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  std::string logic_device_id = "0:0:0:0";
  std::string redundant_logic_device_id = "0:0:1:0";
  (void) AttrUtils::SetStr(*root_graph, ATTR_NAME_LOGIC_DEV_ID, logic_device_id);
  (void) AttrUtils::SetStr(*root_graph, ATTR_NAME_REDUNDANT_LOGIC_DEV_ID, redundant_logic_device_id);
  uint32_t graph_id = 1U;
  root_graph->SetGraphID(graph_id);

  auto npu_engine = std::make_shared<NPUProcessNodeEngine>();
  npu_engine->SetImpl(std::make_shared<MockNpuEngineImpl>());
  FlowModelBuilder flow_model_builder;
  flow_model_builder.process_node_engines_[PNE_ID_NPU] = npu_engine;

  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  FlowModelPtr flow_model;
  EXPECT_EQ(flow_model_builder.BuildModel(graph, {}, {}, flow_model), SUCCESS);
  EXPECT_TRUE(CheckFlowModelIsFlattened(flow_model));
  ASSERT_EQ(flow_model->GetSubmodels().size(), 1);
  EXPECT_EQ(flow_model->GetSubmodels().cbegin()->second->GetLogicDeviceId(), logic_device_id);
  EXPECT_EQ(flow_model->GetSubmodels().cbegin()->second->GetRedundantLogicDeviceId(), redundant_logic_device_id);
}

TEST_F(FlowModelBuilderTest, FlowModelBuildWithSubgraph) {
  auto static_graph = gert::ShareGraph::SimpleStaticGraph();
  static_graph->SetName("static");
  AttrUtils::SetInt(static_graph->FindNode("static_data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(static_graph->FindNode("NetOutput")->GetOpDesc()->MutableInputDesc(0U),
                    ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  DEF_GRAPH(dynamic_graph) {
    auto data_0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("FakeOpHostCpu")
        .InCnt(1)
        .OutCnt(1)
        .Attr(ge::ATTR_NAME_PROCESS_NODE_ENGINE_ID, "HOST_CPU")
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto pcall = OP_CFG("PartitionedCall")
        .InCnt(1)
        .OutCnt(1)
        .Attr(ge::ATTR_NAME_PROCESS_NODE_ENGINE_ID, "NPU")
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op3 = OP_CFG("FakeOpHostCpu")
        .InCnt(1)
        .OutCnt(1)
        .Attr(ge::ATTR_NAME_PROCESS_NODE_ENGINE_ID, "HOST_CPU")
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("_arg_0", data_0)->NODE("fused_op1", fake_type2_op1)->NODE("pcall", pcall)
              ->NODE("fused_op3", fake_type2_op3)->NODE("Node_Output", net_output));
  };

  auto root_graph = ToComputeGraph(dynamic_graph);
  AttrUtils::SetInt(root_graph->FindNode("_arg_0")->GetOpDesc(), "index", 0);
  auto pcall = root_graph->FindNode("pcall");
  pcall->GetOpDesc()->AddSubgraphName("f");
  pcall->GetOpDesc()->SetSubgraphInstanceName(0, static_graph->GetName());
  static_graph->SetParentNode(pcall);
  static_graph->SetParentGraph(root_graph);
  root_graph->AddSubgraph(static_graph);
  (void) AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);

  FlowModelBuilder flow_model_builder;

  auto npu_engine = std::make_shared<NPUProcessNodeEngine>();
  npu_engine->SetImpl(std::make_shared<MockNpuEngineImpl>());
  auto cpu_engine = std::make_shared<CPUProcessNodeEngine>();
  cpu_engine->SetImpl(std::make_shared<MockNpuEngineImpl>());
  flow_model_builder.process_node_engines_[PNE_ID_NPU] = npu_engine;
  flow_model_builder.process_node_engines_[PNE_ID_CPU] = cpu_engine;
  auto fn_cpu = []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::CPUProcessNodeEngine(); };
  auto fn_npu = []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::NPUProcessNodeEngine(); };
  ProcessNodeEngineManager::GetInstance().RegisterEngine("NPU", flow_model_builder.process_node_engines_[PNE_ID_NPU],
                                                         fn_npu);
  ProcessNodeEngineManager::GetInstance().RegisterEngine("HOST_CPU",
                                                         flow_model_builder.process_node_engines_[PNE_ID_CPU], fn_cpu);
  auto ge_graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  FlowModelPtr flow_model;
  EXPECT_EQ(flow_model_builder.BuildModel(ge_graph, {}, {}, flow_model), SUCCESS);
  EXPECT_TRUE(CheckFlowModelIsFlattened(flow_model));
}

TEST_F(FlowModelBuilderTest, FlowModelBuild) {
  DEF_GRAPH(dynamic_graph) {
    auto data_0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("FakeOpHostCpu")
        .InCnt(1)
        .OutCnt(1)
        .Attr(ge::ATTR_NAME_PROCESS_NODE_ENGINE_ID, "HOST_CPU")
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op2 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .Attr(ge::ATTR_NAME_PROCESS_NODE_ENGINE_ID, "NPU")
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op3 = OP_CFG("FakeOpHostCpu")
        .InCnt(1)
        .OutCnt(1)
        .Attr(ge::ATTR_NAME_PROCESS_NODE_ENGINE_ID, "HOST_CPU")
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("_arg_0", data_0)->NODE("fused_op1", fake_type2_op1)->NODE("fused_op2", fake_type2_op2)
              ->NODE("fused_op3", fake_type2_op3)->NODE("Node_Output", net_output));
  };

  auto root_graph = ToComputeGraph(dynamic_graph);
  (void) AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);

  FlowModelBuilder flow_model_builder;
  auto npu_engine = std::make_shared<NPUProcessNodeEngine>();
  npu_engine->SetImpl(std::make_shared<MockNpuEngineImpl>());
  auto cpu_engine = std::make_shared<CPUProcessNodeEngine>();
  cpu_engine->SetImpl(std::make_shared<MockNpuEngineImpl>());
  flow_model_builder.process_node_engines_[PNE_ID_NPU] = npu_engine;
  flow_model_builder.process_node_engines_[PNE_ID_CPU] = cpu_engine;
  auto fn_cpu = []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::CPUProcessNodeEngine(); };
  auto fn_npu = []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::NPUProcessNodeEngine(); };
  ProcessNodeEngineManager::GetInstance().RegisterEngine("NPU", flow_model_builder.process_node_engines_[PNE_ID_NPU],
                                                         fn_npu);
  ProcessNodeEngineManager::GetInstance().RegisterEngine("HOST_CPU",
                                                         flow_model_builder.process_node_engines_[PNE_ID_CPU], fn_cpu);
  auto ge_graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  FlowModelPtr flow_model;
  EXPECT_EQ(flow_model_builder.BuildModel(ge_graph, {}, {}, flow_model), SUCCESS);
  EXPECT_TRUE(CheckFlowModelIsFlattened(flow_model));
}

TEST_F(FlowModelBuilderTest, BuildModel_DataFlowGraph_SUCCESS) {

  {
    auto &global_options_mutex = GetGlobalOptionsMutex();
    const std::lock_guard<std::mutex> lock(global_options_mutex);
    auto &global_options = GetMutableGlobalOptions();
    global_options[OPTION_NUMA_CONFIG] =
        R"({"cluster":[{"cluster_nodes":[{"is_local":true, "item_list":[{"item_id":0}], "node_id":0, "node_type":"TestNodeType1"}]}],"item_def":[{"aic_type":"[DAVINCI_V100:10]","item_type":"","memory":"[DDR:80GB]","resource_type":"Ascend"}],"node_def":[{"item_type":"","links_mode":"TCP:128Gb","node_type":"TestNodeType1","resource_type":"X86","support_links":"[ROCE]"}]})";
  }
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node2 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto net_output = OP_CFG("NetOutput").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0)->EDGE(1, 1)->NODE("node1", node1));
    CHAIN(NODE("node1", node1)->EDGE(0, 0)->NODE("node2", node2));
    CHAIN(NODE("node1", node1)->EDGE(0, 1)->NODE("node2", node2));
    CHAIN(NODE("node2", node2)->EDGE(0, 0)->NODE("net_output", net_output));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  auto output_node = root_graph->FindNode("net_output");
  output_node->GetOpDesc()->SetSrcIndex({0});
  output_node->GetOpDesc()->SetSrcName({"node2"});
  EXPECT_EQ(AttrUtils::SetBool(root_graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true), true);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp0 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp0_config_file = "./pp0_config.json";
  std::string target_bin_path = "./libxxx.so";
  {
    nlohmann::json pp0_cfg_json = {
        {"workspace", "./temp"}, {"target_bin", "libxxx.so"},          {"input_num", 2},
        {"output_num", 2},       {"cmakelist_path", "CMakeLists.txt"}, {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
    std::ofstream target_bin(target_bin_path);
    target_bin << target_bin_path;
  }
  pp0.set_compile_cfg_file(pp0_config_file);
  std::string pp0_str;
  pp0.SerializeToString(&pp0_str);
  std::vector<std::string> pp0_attr{pp0_str};
  auto node0 = root_graph->FindNode("node0");
  auto op_desc0 = node0->GetOpDesc();
  AttrUtils::SetListStr(op_desc0, dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp0_attr);
  ASSERT_TRUE(AttrUtils::SetInt(op_desc0, ATTR_NAME_ESCHED_EVENT_PRIORITY, 0));
  ASSERT_TRUE(AttrUtils::SetInt(op_desc0, ATTR_NAME_ESCHED_EVENT_PRIORITY, 0));
  auto pp1 = dataflow::ProcessPoint();
  pp1.set_name("pp1");
  pp1.set_type(dataflow::ProcessPoint_ProcessPointType_GRAPH);
  std::string pp1_config_file = "./pp1_config.json";
  {
    nlohmann::json pp1_cfg_json = {
        {"inputs_tensor_desc",
         {{{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}}, {{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}}}}};
    std::ofstream json_file(pp1_config_file);
    json_file << pp1_cfg_json << std::endl;
  }
  pp1.set_compile_cfg_file(pp1_config_file);
  pp1.add_graphs("pp1");
  auto pp1_input0 = pp1.add_in_edges();
  pp1_input0->set_node_name("node1");
  pp1_input0->set_index(0);
  auto pp1_input1 = pp1.add_in_edges();
  pp1_input1->set_node_name("node1");
  pp1_input1->set_index(1);
  auto pp1_output0 = pp1.add_out_edges();
  pp1_output0->set_node_name("node1");
  pp1_output0->set_index(0);
  std::string pp1_str;
  pp1.SerializeToString(&pp1_str);
  std::vector<std::string> pp1_attr{pp1_str};
  auto node1 = root_graph->FindNode("node1");
  auto op_desc1 = node1->GetOpDesc();
  AttrUtils::SetListStr(op_desc1, dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp1_attr);
  ASSERT_TRUE(AttrUtils::SetInt(op_desc1, ATTR_NAME_ESCHED_EVENT_PRIORITY, 0));
  ASSERT_TRUE(AttrUtils::SetInt(op_desc1, ATTR_NAME_ESCHED_EVENT_PRIORITY, 0));
  ASSERT_TRUE(AttrUtils::SetBool(op_desc1, ATTR_NAME_FLOW_ATTR, true));
  ASSERT_TRUE(AttrUtils::SetInt(op_desc1, ATTR_NAME_FLOW_ATTR_DEPTH, 99));
  ASSERT_TRUE(AttrUtils::SetStr(op_desc1, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, "FIFO"));
  DEF_GRAPH(sub_graph_def) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3}).Build("add");
    auto net_output = OP_CFG("NetOutput").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE(add));
    CHAIN(NODE(add)->EDGE(0, 0)->NODE("Node_Output", net_output));
  };
  auto sub_graph = GraphUtilsEx::GetComputeGraph(ToGeGraph(sub_graph_def));
  auto graph_output_node = sub_graph->FindNode("Node_Output");
  graph_output_node->GetOpDesc()->SetSrcIndex({0});
  graph_output_node->GetOpDesc()->SetSrcName({"add"});
  (void)AttrUtils::SetStr(sub_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  sub_graph->SetParentNode(node1);
  sub_graph->SetParentGraph(root_graph);
  sub_graph->SetName("pp1");
  root_graph->AddSubgraph("pp1", sub_graph);

  dataflow::ProcessPoint pp2;
  pp2.set_name("func_invoke_graph_pp");
  pp2.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp2_config_file = "./pp2_config.json";
  {
    nlohmann::json pp2_cfg_json = {{"workspace", "./temp"},
                                   {"target_bin", "libxxx.so"},
                                   {"input_num", 2},
                                   {"output_num", 1},
                                   {"cmakelist_path", "CMakeLists.txt"},
                                   {"func_list",
                                    {{{"func_name", "func1"}, {"inputs_index", {0}}, {"outputs_index", {0}}},
                                     {{"func_name", "func2"}, {"inputs_index", {1}}, {"outputs_index", {0}}}}}};
    std::ofstream json_file(pp2_config_file);
    json_file << pp2_cfg_json << std::endl;
  }
  pp2.set_compile_cfg_file(pp2_config_file);

  auto pp3 = dataflow::ProcessPoint();
  pp3.set_name("pp3");
  pp3.set_type(dataflow::ProcessPoint_ProcessPointType_FLOW_GRAPH);

  pp1.set_name("invoked_graph_pp");
  pp1.set_graphs(0 ,"invoked_graph_pp");
  pp3.set_name("invoked_flow_graph_pp");
  pp3.add_graphs("invoked_flow_graph_pp");
  auto invoke_pps = pp2.mutable_invoke_pps();
  (*invoke_pps)["invoked_graph_pp"] = pp1;
  (*invoke_pps)["invoked_flow_graph_pp"] = pp3;
  std::string pp2_str;
  pp2.SerializeToString(&pp2_str);
  std::vector<std::string> pp2_attr{pp2_str};
  auto node2 = root_graph->FindNode("node2");
  auto op_desc2 = node2->GetOpDesc();
  AttrUtils::SetListStr(op_desc2, dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp2_attr);
  ASSERT_TRUE(AttrUtils::SetInt(op_desc2, ATTR_NAME_ESCHED_EVENT_PRIORITY, 0));
  ASSERT_TRUE(AttrUtils::SetInt(op_desc2, ATTR_NAME_ESCHED_EVENT_PRIORITY, 0));
  DEF_GRAPH(invoked_graph_def) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto net_output = OP_CFG("NetOutput").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0));
    CHAIN(NODE("node0", node0)->EDGE(0, 0)->NODE("Node_Output", net_output));
  };
  auto invoked_graph = GraphUtilsEx::GetComputeGraph(ToGeGraph(invoked_graph_def));
  auto invoked_output_node = invoked_graph->FindNode("Node_Output");
  invoked_output_node->GetOpDesc()->SetSrcIndex({0});
  invoked_output_node->GetOpDesc()->SetSrcName({"node0"});
  (void)AttrUtils::SetStr(invoked_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  (void)AttrUtils::SetBool(invoked_graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true);
  auto pp4 = dataflow::ProcessPoint();
  pp4.set_name("pp0");
  pp4.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp4_config_file = "./pp4_config.json";
  std::string target_bin_path1 = "./libxxx.so";
  {
    nlohmann::json pp4_cfg_json = {
        {"workspace", "./temp"}, {"target_bin", "libxxx.so"},          {"input_num", 2},
        {"output_num", 1},       {"cmakelist_path", "CMakeLists.txt"}, {"func_list", {{{"func_name", "func2"}}}}};
    std::ofstream json_file(pp4_config_file);
    json_file << pp4_cfg_json << std::endl;
    std::ofstream target_bin1(target_bin_path1);
    target_bin1 << target_bin_path1;
  }
  pp4.set_compile_cfg_file(pp4_config_file);
  std::string pp4_str;
  pp4.SerializeToString(&pp4_str);
  std::vector<std::string> pp4_attr{pp4_str};

  auto func_node0 = invoked_graph->FindNode("node0");
  AttrUtils::SetListStr(func_node0->GetOpDesc(), dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp4_attr);
  DEF_GRAPH(sub_graph_def1) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3}).Build("add");
    auto net_output = OP_CFG("NetOutput").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE(add));
    CHAIN(NODE(add)->EDGE(0, 0)->NODE("Node_Output", net_output));
  };
  auto sub_graph1 = GraphUtilsEx::GetComputeGraph(ToGeGraph(sub_graph_def1));
  auto graph1_output_node = sub_graph1->FindNode("Node_Output");
  graph1_output_node->GetOpDesc()->SetSrcIndex({0});
  graph1_output_node->GetOpDesc()->SetSrcName({"add"});
  (void)AttrUtils::SetStr(sub_graph1, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  sub_graph1->SetParentNode(node2);
  sub_graph1->SetParentGraph(root_graph);
  sub_graph1->SetName("invoked_graph_pp");
  invoked_graph->SetParentNode(node2);
  invoked_graph->SetParentGraph(root_graph);
  invoked_graph->SetName("invoked_flow_graph_pp");
  root_graph->AddSubgraph("invoked_flow_graph_pp", invoked_graph);
  root_graph->AddSubgraph("invoked_graph_pp", sub_graph1);

  uint32_t graph_id = 1U;
  root_graph->SetGraphID(graph_id);
  auto npu_engine = std::make_shared<NPUProcessNodeEngine>();
  npu_engine->SetImpl(std::make_shared<MockNpuEngineImpl>());
  FlowModelBuilder flow_model_builder;
  flow_model_builder.process_node_engines_[PNE_ID_NPU] = npu_engine;
  flow_model_builder.process_node_engines_[PNE_ID_UDF] = std::make_shared<UdfProcessNodeEngine>();

  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  FlowModelPtr flow_model;
  EXPECT_EQ(flow_model_builder.BuildModel(graph, {}, {}, flow_model), SUCCESS);
  EXPECT_TRUE(CheckFlowModelIsFlattened(flow_model));
  remove(pp0_config_file.c_str());
  remove(pp1_config_file.c_str());
  remove(pp2_config_file.c_str());
  remove(pp4_config_file.c_str());
  remove(target_bin_path.c_str());
  remove(target_bin_path1.c_str());
}

TEST_F(FlowModelBuilderTest, BuildModel_DataFlowGraph_FAILED) {

  {
    auto &global_options_mutex = GetGlobalOptionsMutex();
    const std::lock_guard<std::mutex> lock(global_options_mutex);
    auto &global_options = GetMutableGlobalOptions();
    global_options[OPTION_NUMA_CONFIG] =
        R"({"cluster":[{"cluster_nodes":[{"is_local":true, "item_list":[{"item_id":0}], "node_id":0, "node_type":"TestNodeType1"}]}],"item_def":[{"aic_type":"[DAVINCI_V100:10]","item_type":"","memory":"[DDR:80GB]","resource_type":"Ascend"}],"node_def":[{"item_type":"","links_mode":"TCP:128Gb","node_type":"TestNodeType1","resource_type":"X86","support_links":"[ROCE]"}]})";
  }
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto net_output = OP_CFG("NetOutput").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("net_output", net_output));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  EXPECT_EQ(AttrUtils::SetBool(root_graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true), true);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp0 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp0_config_file = "./pp0_config.json";
  std::string target_bin_path = "./libxxx.so";
  {
    nlohmann::json pp0_cfg_json = {
        {"workspace", "./temp"}, {"target_bin", "libxxx.so"},          {"input_num", 1},
        {"output_num", 1},       {"cmakelist_path", "CMakeLists.txt"},
        {"func_list", {{{"func_name", "func1"}, {"inputs_index", {0}}, {"outputs_index", {0}}}}}};
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
    std::ofstream target_bin(target_bin_path);
    target_bin << target_bin_path;
  }
  pp0.set_compile_cfg_file(pp0_config_file);
  auto pp1 = dataflow::ProcessPoint();
  pp1.set_name("pp1");
  pp1.set_type(dataflow::ProcessPoint_ProcessPointType_FLOW_GRAPH);

  pp1.set_name("invoked_flow_graph_pp");
  pp1.add_graphs("invoked_flow_graph_pp");
  auto invoke_pps = pp0.mutable_invoke_pps();
  (*invoke_pps)["invoked_flow_graph_pp"] = pp1;
  std::string pp0_str;
  pp0.SerializeToString(&pp0_str);
  std::vector<std::string> pp0_attr{pp0_str};
  auto node0 = root_graph->FindNode("node0");
  auto op_desc0 = node0->GetOpDesc();
  AttrUtils::SetListStr(op_desc0, dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp0_attr);
  ASSERT_TRUE(AttrUtils::SetInt(op_desc0, ATTR_NAME_ESCHED_EVENT_PRIORITY, 0));
  ASSERT_TRUE(AttrUtils::SetInt(op_desc0, ATTR_NAME_ESCHED_EVENT_PRIORITY, 0));

  auto sub_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(sub_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  (void)AttrUtils::SetBool(sub_graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true);
  auto dep1_node0 = sub_graph->FindNode("node0");
  AttrUtils::SetListStr(dep1_node0->GetOpDesc(), dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp0_attr);

  auto sub_graph1 = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(sub_graph1, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  (void)AttrUtils::SetBool(sub_graph1, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true);
  auto dep2_node0 = sub_graph1->FindNode("node0");
  AttrUtils::SetListStr(dep2_node0->GetOpDesc(), dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp0_attr);

  auto sub_graph2 = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(sub_graph2, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  (void)AttrUtils::SetBool(sub_graph2, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true);
  auto dep3_node0 = sub_graph2->FindNode("node0");
  AttrUtils::SetListStr(dep3_node0->GetOpDesc(), dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp0_attr);

  auto sub_graph3 = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(sub_graph3, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  (void)AttrUtils::SetBool(sub_graph3, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true);
  auto dep4_node0 = sub_graph3->FindNode("node0");
  AttrUtils::SetListStr(dep4_node0->GetOpDesc(), dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp0_attr);

  auto sub_graph4 = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(sub_graph4, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  (void)AttrUtils::SetBool(sub_graph4, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true);
  auto dep5_node0 = sub_graph4->FindNode("node0");
  AttrUtils::SetListStr(dep5_node0->GetOpDesc(), dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp0_attr);

  sub_graph4->SetParentNode(dep4_node0);
  sub_graph4->SetParentGraph(sub_graph3);
  sub_graph4->SetName("invoked_flow_graph_pp");
  sub_graph3->AddSubgraph("invoked_flow_graph_pp", sub_graph4);

  sub_graph3->SetParentNode(dep3_node0);
  sub_graph3->SetParentGraph(sub_graph2);
  sub_graph3->SetName("invoked_flow_graph_pp");
  sub_graph2->AddSubgraph("invoked_flow_graph_pp", sub_graph3);

  sub_graph2->SetParentNode(dep2_node0);
  sub_graph2->SetParentGraph(sub_graph1);
  sub_graph2->SetName("invoked_flow_graph_pp");
  sub_graph1->AddSubgraph("invoked_flow_graph_pp", sub_graph2);

  sub_graph1->SetParentNode(dep1_node0);
  sub_graph1->SetParentGraph(sub_graph);
  sub_graph1->SetName("invoked_flow_graph_pp");
  sub_graph->AddSubgraph("invoked_flow_graph_pp", sub_graph1);

  sub_graph->SetParentNode(node0);
  sub_graph->SetParentGraph(root_graph);
  sub_graph->SetName("invoked_flow_graph_pp");
  root_graph->AddSubgraph("invoked_flow_graph_pp", sub_graph);

  uint32_t graph_id = 1U;
  root_graph->SetGraphID(graph_id);

  auto npu_engine = std::make_shared<NPUProcessNodeEngine>();
  npu_engine->SetImpl(std::make_shared<MockNpuEngineImpl>());
  FlowModelBuilder flow_model_builder;
  flow_model_builder.process_node_engines_[PNE_ID_NPU] = npu_engine;
  flow_model_builder.process_node_engines_[PNE_ID_UDF] = std::make_shared<UdfProcessNodeEngine>();

  auto ge_graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  FlowModelPtr flow_model;
  EXPECT_NE(flow_model_builder.BuildModel(ge_graph, {}, {}, flow_model), SUCCESS);
  remove(pp0_config_file.c_str());
  remove(target_bin_path.c_str());
}

TEST_F(FlowModelBuilderTest, BuildModel_Invoke_modelpp_SUCCESS) {

  {
    auto &global_options_mutex = GetGlobalOptionsMutex();
    const std::lock_guard<std::mutex> lock(global_options_mutex);
    auto &global_options = GetMutableGlobalOptions();
    global_options[OPTION_NUMA_CONFIG] =
        R"({"cluster":[{"cluster_nodes":[{"is_local":true, "item_list":[{"item_id":0}], "node_id":0, "node_type":"TestNodeType1"}]}],"item_def":[{"aic_type":"[DAVINCI_V100:10]","item_type":"","memory":"[DDR:80GB]","resource_type":"Ascend"}],"node_def":[{"item_type":"","links_mode":"TCP:128Gb","node_type":"TestNodeType1","resource_type":"X86","support_links":"[ROCE]"}]})";
  }
  constexpr const char *compiler_config = "./temp/compiler_config.json";
  {
    nlohmann::json cpu_compiler_json = {
        {"compiler",
         {
             {
                 {"resource_type", "X86"},
                 {"toolchain", "/usr/bin/g++"},
             },
             {
                 {"resource_type", "Ascend"},
                 {"toolchain", "/usr/local/Ascend/hcc"},
             },
         }},
    };
    std::ofstream json_file(compiler_config);
    json_file << cpu_compiler_json << std::endl;
  }
  constexpr const char *udf_pp_config_file = "./temp/udf_pp_config.json";
  {
    nlohmann::json pp0_cfg_json = {{"workspace", "./temp"},
                                   {"target_bin", "libxxx.so"},
                                   {"input_num", 1},
                                   {"output_num", 1},
                                   {"cmakelist_path", "CMakeLists.txt"},
                                   {"compiler", compiler_config},
                                   {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(udf_pp_config_file);
    json_file << pp0_cfg_json << std::endl;
  }
  constexpr const char *deploy_info = "./temp/ut_deploy_info.json";
  {
    std::ofstream json_file(deploy_info);
    std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:0:0"
          }
        ]
      })";
    json_file << content << std::endl;
  }
  auto &ge_context = GetThreadLocalContext();
  const auto old_graph_options = ge_context.GetAllGraphOptions();
  std::map<std::string, std::string> graph_options = old_graph_options;
  graph_options["ge.experiment.data_flow_deploy_info_path"] = deploy_info;
  ge_context.SetGraphOption(graph_options);

  auto data0 = dflow::FlowData("Data0", 0);
  auto node0 = dflow::FlowNode("node0", 1, 1).SetInput(0, data0);

  auto invoked_model_pp0 =
      dflow::ModelPp("invoked_graph_pp0", PathUtils::Join({run_data_path, "origin_model/root_model.om"}).c_str());
  std::string pp_config_file = "./config.json";
  {
    nlohmann::json pp1_cfg_json = {{"invoke_model_fusion_inputs","0"}};
    std::ofstream json_file(pp_config_file);
    json_file << pp1_cfg_json << std::endl;
  }
  invoked_model_pp0.SetCompileConfig(pp_config_file.c_str());
  // function pp
  auto udf_pp = dflow::FunctionPp("udf_pp")
                    .SetCompileConfig(udf_pp_config_file)
                    .AddInvokedClosure("invoke_model_pp0", invoked_model_pp0);
  node0.AddPp(udf_pp);

  std::vector<dflow::FlowOperator> inputsOperator{data0};
  std::vector<dflow::FlowOperator> outputsOperator{node0};

  dflow::FlowGraph flow_graph("test_model_pp");
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  const auto &graph = flow_graph.ToGeGraph();
  const auto &comput_graph = GraphUtilsEx::GetComputeGraph(graph);
  (void) AttrUtils::SetStr(comput_graph, ATTR_NAME_SESSION_GRAPH_ID, "0");

  auto npu_engine = std::make_shared<NPUProcessNodeEngine>();
  npu_engine->SetImpl(std::make_shared<MockNpuEngineImpl>());
  FlowModelBuilder flow_model_builder;
  flow_model_builder.process_node_engines_[PNE_ID_NPU] = npu_engine;
  flow_model_builder.process_node_engines_[PNE_ID_UDF] = std::make_shared<UdfProcessNodeEngine>();

  auto ge_graph = GraphUtilsEx::CreateGraphFromComputeGraph(comput_graph);
  FlowModelPtr flow_model;
  EXPECT_EQ(flow_model_builder.BuildModel(ge_graph, {}, {}, flow_model), SUCCESS);
  ge_context.SetGraphOption(old_graph_options);
  (void)system("rm -rf pp_config_file");
}

TEST_F(FlowModelBuilderTest, UpdateDeployInfo_SUCCESS) {
  DEF_GRAPH(graph_def) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3}).Build("add");
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE(add));
    ADD_OUTPUT(add, 0);
  };
  auto graph = GraphUtilsEx::GetComputeGraph(ToGeGraph(graph_def));
  std::string logic_device_id0 = "0:0:0";
  auto graph_model = BuildPneModel("root_graph", graph);
  ASSERT_EQ(graph_model->SetLogicDeviceId(logic_device_id0), SUCCESS);
  ASSERT_EQ(graph_model->GetLogicDeviceId(), logic_device_id0);
  FlowModelPtr flow_model = MakeShared<FlowModel>(graph);
  flow_model->AddSubModel(graph_model, PNE_ID_NPU);
  std::string logic_device_id1 = "0:0:1";
  ASSERT_EQ(AttrUtils::SetStr(graph, ATTR_NAME_LOGIC_DEV_ID, logic_device_id1), true);
  FlowModelBuilder builder;
  ASSERT_EQ(builder.UpdateDeployInfo(graph, flow_model), SUCCESS);
  ASSERT_EQ(graph_model->GetLogicDeviceId(), logic_device_id1);
}

TEST_F(FlowModelBuilderTest, MakeInputTensors_by_inputshape_range_success) {
  DEF_GRAPH(graph_def) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {-1, 2, -1});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {-1, -1, 3});
    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3}).Build("add");
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE(add));
    ADD_OUTPUT(add, 0);
  };
  auto graph = GraphUtilsEx::GetComputeGraph(ToGeGraph(graph_def));
  FlowModelBuilder builder;
  std::map<std::string, std::string> options;
  options["ge.exec.dynamicGraphExecuteMode"] = "dynamic_execute";
  std::vector<GeTensor> input_tensors;
  ASSERT_EQ(builder.MakeInputTensors(graph, options, input_tensors), SUCCESS);
  input_tensors.clear();
  options["ge.exec.dataInputsShapeRange"] = "[1~3,2,-1],[2~3,-1,3]";
  input_tensors.clear();
  ASSERT_EQ(builder.MakeInputTensors(graph, options, input_tensors), SUCCESS);
  ASSERT_EQ(input_tensors.size(), 2);
  std::vector<int64_t> v1 = {1, 2, 0};
  std::vector<int64_t> v2 = {2, 0, 3};
  ASSERT_EQ(input_tensors[0].GetTensorDesc().GetShape().GetDims(), v1);
  ASSERT_EQ(input_tensors[1].GetTensorDesc().GetShape().GetDims(), v2);
}

TEST_F(FlowModelBuilderTest, BuildModel_Failed) {

  {
    auto &global_options_mutex = GetGlobalOptionsMutex();
    const std::lock_guard<std::mutex> lock(global_options_mutex);
    auto &global_options = GetMutableGlobalOptions();
    global_options[OPTION_NUMA_CONFIG] =
        R"({"cluster":[{"cluster_nodes":[{"is_local":true, "item_list":[{"item_id":0}], "node_id":0, "node_type":"TestNodeType1"}]}],"item_def":[{"aic_type":"[DAVINCI_V100:10]","item_type":"","memory":"[DDR:80GB]","resource_type":"Ascend"}],"node_def":[{"item_type":"","links_mode":"TCP:128Gb","node_type":"TestNodeType1","resource_type":"X86","support_links":"[ROCE]"}]})";
  }
  class MockMmpaOpen : public MockMmpa {
   public:
    MOCK_METHOD(INT32, Open2, (const CHAR *path_name, INT32 flags, MODE mode));
  };
  auto data0 = dflow::FlowData("Data0", 0);
  auto node0 = dflow::FlowNode("node0", 1, 1).SetInput(0, data0);
  // function pp
  auto pp0 = dflow::FunctionPp("func_pp0").SetCompileConfig("./pp0_config.json");
  {
    nlohmann::json pp0_cfg_json = {
        {"workspace", "./temp"}, {"target_bin", "libxxx.so"},          {"input_num", 1},
        {"output_num", 1},       {"cmakelist_path", "CMakeLists.txt"}, {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file("./pp0_config.json");
    json_file << pp0_cfg_json << std::endl;
  }
  node0.AddPp(pp0);
  std::vector<dflow::FlowOperator> inputsOperator{data0};
  std::vector<dflow::FlowOperator> outputsOperator{node0};
  dflow::FlowGraph flow_graph("flow_graph");
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  auto root_graph = GraphUtilsEx::GetComputeGraph(flow_graph.ToGeGraph());
  (void)AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "00");
  uint32_t graph_id = 1U;
  root_graph->SetGraphID(graph_id);

  auto npu_engine = std::make_shared<NPUProcessNodeEngine>();
  npu_engine->SetImpl(std::make_shared<MockNpuEngineImpl>());
  FlowModelBuilder flow_model_builder;
  flow_model_builder.process_node_engines_[PNE_ID_NPU] = npu_engine;
  flow_model_builder.process_node_engines_[PNE_ID_UDF] = std::make_shared<UdfProcessNodeEngine>();

  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  FlowModelPtr flow_model;
  auto mock = std::make_shared<MockMmpaOpen>();
  MmpaStub::GetInstance().SetImpl(mock);
  using ::testing::_;
  auto fd0 = open("./temp/CMakeLists.txt", O_WRONLY);
  EXPECT_CALL(*mock, Open2(_, _, _)).WillOnce(Return(fd0)).WillRepeatedly(Return(100));
  EXPECT_EQ(flow_model_builder.BuildModel(graph, {}, {}, flow_model), FAILED);
  remove("./pp0_config.json");
}

TEST_F(FlowModelBuilderTest, FlowModelBuildWithCache) {
  std::map<std::string, std::string> graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  GE_MAKE_GUARD(recover_graph_cfg, [&graph_options](){
    GetThreadLocalContext().SetGraphOption(graph_options);
  });
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  std::map<std::string, std::string> graph_option_new;
  graph_option_new["ge.graph_compiler_cache_dir"] = "./ut_cache_dir";
  graph_option_new["ge.ExternalWeightDir"] = "./ut_cache_dir";
  graph_option_new["ge.graph_key"] = "graph_key_xxx";
  GetThreadLocalContext().SetGraphOption(graph_option_new);
  (void)system("mkdir ./ut_cache_dir");
  auto graph = FakeComputeGraph("root_graph");

  auto graph1 = FakeComputeGraph("root_graph");
  auto ge_root_model1 = BuildPneModel("graph1", graph1);
  FlowModelPtr flow_model = MakeShared<ge::FlowModel>(graph);
  flow_model->AddSubModel(ge_root_model1, PNE_ID_NPU);

  {
    FlowModelCache flow_model_cache;
    auto ret = flow_model_cache.Init(graph);
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    EXPECT_EQ(ret, SUCCESS);
  }
  auto ge_graph = GraphUtilsEx::CreateGraphFromComputeGraph(graph);
  FlowModelPtr flow_model1;
  FlowModelBuilder flow_model_builder;
  EXPECT_EQ(flow_model_builder.BuildModel(ge_graph, {}, {}, flow_model1), SUCCESS);
  VarManager::Instance(0)->Destory();
}

TEST_F(FlowModelBuilderTest, FlowModelBuildWithCacheIOChkSucc) {
  std::map<std::string, std::string> graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  GE_MAKE_GUARD(recover_graph_cfg, [&graph_options](){
    GetThreadLocalContext().SetGraphOption(graph_options);
  });
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  std::map<std::string, std::string> graph_option_new;
  graph_option_new["ge.graph_compiler_cache_dir"] = "./ut_cache_dir";
  graph_option_new["ge.graph_key"] = "graph_key_xxx";
  GetThreadLocalContext().SetGraphOption(graph_option_new);
  (void)system("mkdir ./ut_cache_dir");
  auto graph = FakeComputeGraph("root_graph");
  EXPECT_EQ(AttrUtils::SetBool(graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true), true);

  auto graph1 = FakeComputeGraph("root_graph");
  auto ge_root_model1 = BuildPneModel("graph1", graph1);
  FlowModelPtr flow_model = MakeShared<ge::FlowModel>(graph);
  flow_model->AddSubModel(ge_root_model1, PNE_ID_NPU);

  {
    FlowModelCache flow_model_cache;
    auto ret = flow_model_cache.Init(graph);
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    EXPECT_EQ(ret, SUCCESS);
  }
  EXPECT_EQ(AttrUtils::SetBool(graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true), true);
  auto ge_graph = GraphUtilsEx::CreateGraphFromComputeGraph(graph);
  FlowModelPtr flow_model1;
  FlowModelBuilder flow_model_builder;
  EXPECT_EQ(flow_model_builder.BuildModel(ge_graph, {}, {}, flow_model1), SUCCESS);
  VarManager::Instance(0)->Destory();
}

TEST_F(FlowModelBuilderTest, FlowModelBuildWithCacheIOChkOutputFailed) {
  std::map<std::string, std::string> graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  GE_MAKE_GUARD(recover_graph_cfg, [&graph_options](){
    GetThreadLocalContext().SetGraphOption(graph_options);
  });
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  std::map<std::string, std::string> graph_option_new;
  graph_option_new["ge.graph_compiler_cache_dir"] = "./ut_cache_dir";
  graph_option_new["ge.graph_key"] = "graph_key_xxx";
  GetThreadLocalContext().SetGraphOption(graph_option_new);
  (void)system("mkdir ./ut_cache_dir");
  auto graph = FakeComputeGraph("root_graph");
  EXPECT_EQ(AttrUtils::SetBool(graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true), true);

  auto graph1 = FakeComputeGraph("root_graph");
  auto ge_root_model1 = BuildPneModel("graph1", graph1);
  FlowModelPtr flow_model = MakeShared<ge::FlowModel>(graph);
  flow_model->AddSubModel(ge_root_model1, PNE_ID_NPU);

  {
    FlowModelCache flow_model_cache;
    auto ret = flow_model_cache.Init(graph);
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    EXPECT_EQ(ret, SUCCESS);
  }
  {
    auto data0 = dflow::FlowData("Data0", 0);
    auto node0 = dflow::FlowNode("node0", 1, 1).SetInput(0, data0);
    // function pp
    auto pp0 = dflow::FunctionPp("func_pp0").SetCompileConfig("./pp0_config.json");
    {
      nlohmann::json pp0_cfg_json = {
          {"workspace", "./temp"}, {"target_bin", "libxxx.so"},          {"input_num", 1},
          {"output_num", 1},       {"cmakelist_path", "CMakeLists.txt"}, {"func_list", {{{"func_name", "func1"}}}}};
      std::ofstream json_file("./pp0_config.json");
      json_file << pp0_cfg_json << std::endl;
    }
    node0.AddPp(pp0);
    std::vector<dflow::FlowOperator> inputsOperator{data0};
    std::vector<dflow::FlowOperator> outputsOperator{node0, node0};
    dflow::FlowGraph flow_graph("flow_graph");
    flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
    graph = GraphUtilsEx::GetComputeGraph(flow_graph.ToGeGraph());
    AttrUtils::SetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, "0_1");
  }
  EXPECT_EQ(AttrUtils::SetBool(graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true), true);
  auto ge_graph = GraphUtilsEx::CreateGraphFromComputeGraph(graph);
  FlowModelPtr flow_model1;
  FlowModelBuilder flow_model_builder;
  EXPECT_EQ(flow_model_builder.BuildModel(ge_graph, {}, {}, flow_model1), FAILED);
  VarManager::Instance(0)->Destory();
}

TEST_F(FlowModelBuilderTest, FlowModelBuildWithCacheIOChkInputNumFailed) {
  std::map<std::string, std::string> graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  GE_MAKE_GUARD(recover_graph_cfg, [&graph_options](){
    GetThreadLocalContext().SetGraphOption(graph_options);
  });
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  std::map<std::string, std::string> graph_option_new;
  graph_option_new["ge.graph_compiler_cache_dir"] = "./ut_cache_dir";
  graph_option_new["ge.graph_key"] = "graph_key_xxx";
  GetThreadLocalContext().SetGraphOption(graph_option_new);
  (void)system("mkdir ./ut_cache_dir");
  auto graph = FakeComputeGraph("root_graph");
  EXPECT_EQ(AttrUtils::SetBool(graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true), true);

  auto graph1 = FakeComputeGraph("root_graph");
  auto ge_root_model1 = BuildPneModel("graph1", graph1);
  FlowModelPtr flow_model = MakeShared<ge::FlowModel>(graph);
  flow_model->AddSubModel(ge_root_model1, PNE_ID_NPU);

  {
    FlowModelCache flow_model_cache;
    auto ret = flow_model_cache.Init(graph);
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    EXPECT_EQ(ret, SUCCESS);
  }
  {
    DEF_GRAPH(graph1) {
      auto data_0 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {16});
      auto data_1 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {16});
      auto fake_type2_op1 = OP_CFG("FakeOpNpu").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {16});
      auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {-1});
      CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("fused_op1", fake_type2_op1));
      CHAIN(NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("fused_op1", fake_type2_op1)->EDGE(0, 0)->NODE("Node_Output", net_output));
    };
    graph = ToComputeGraph(graph1);
    graph->SetName("Name");
    graph->SetSessionID(0);
    AttrUtils::SetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, "0_1");
  }
  EXPECT_EQ(AttrUtils::SetBool(graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true), true);
  auto ge_graph = GraphUtilsEx::CreateGraphFromComputeGraph(graph);
  FlowModelPtr flow_model1;
  FlowModelBuilder flow_model_builder;
  EXPECT_EQ(flow_model_builder.BuildModel(ge_graph, {}, {}, flow_model1), FAILED);
  VarManager::Instance(0)->Destory();
}

TEST_F(FlowModelBuilderTest, FlowModelBuildWithCacheIOChkInputNameFailed) {
  std::map<std::string, std::string> graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  GE_MAKE_GUARD(recover_graph_cfg, [&graph_options](){
    GetThreadLocalContext().SetGraphOption(graph_options);
  });
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  std::map<std::string, std::string> graph_option_new;
  graph_option_new["ge.graph_compiler_cache_dir"] = "./ut_cache_dir";
  graph_option_new["ge.graph_key"] = "graph_key_xxx";
  GetThreadLocalContext().SetGraphOption(graph_option_new);
  (void)system("mkdir ./ut_cache_dir");
  auto graph = FakeComputeGraph("root_graph");
  EXPECT_EQ(AttrUtils::SetBool(graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true), true);

  auto graph1 = FakeComputeGraph("root_graph");
  auto ge_root_model1 = BuildPneModel("graph1", graph1);
  FlowModelPtr flow_model = MakeShared<ge::FlowModel>(graph);
  flow_model->AddSubModel(ge_root_model1, PNE_ID_NPU);

  {
    FlowModelCache flow_model_cache;
    auto ret = flow_model_cache.Init(graph);
    EXPECT_EQ(ret, SUCCESS);
    ret = flow_model_cache.TryCacheFlowModel(flow_model);
    EXPECT_EQ(ret, SUCCESS);
  }
  {
    DEF_GRAPH(graph1) {
      auto data_0 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {16});
      auto data_1 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {16});
      auto fake_type2_op1 = OP_CFG("FakeOpNpu").InCnt(1).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {16});
      auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {-1});
      CHAIN(NODE("data0", data_0)->EDGE(0, 0)->NODE("fused_op1", fake_type2_op1)->EDGE(0, 0)->NODE("Node_Output", net_output));
    };
    graph = ToComputeGraph(graph1);
    graph->SetName("Name");
    graph->SetSessionID(0);
    AttrUtils::SetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, "0_1");
  }
  EXPECT_EQ(AttrUtils::SetBool(graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true), true);
  auto ge_graph = GraphUtilsEx::CreateGraphFromComputeGraph(graph);
  FlowModelPtr flow_model1;
  FlowModelBuilder flow_model_builder;
  EXPECT_EQ(flow_model_builder.BuildModel(ge_graph, {}, {}, flow_model1), FAILED);
  VarManager::Instance(0)->Destory();
}

TEST_F(FlowModelBuilderTest, FlowModelBuildWithCacheRemoveSubgraphs) {
  auto static_graph = gert::ShareGraph::SimpleStaticGraph();
  static_graph->SetName("static");
  AttrUtils::SetInt(static_graph->FindNode("static_data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(static_graph->FindNode("NetOutput")->GetOpDesc()->MutableInputDesc(0U),
                    ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  auto graph1 = FakeComputeGraph("root_graph");
  graph1->SetAllSubgraphs({static_graph});
  auto ge_root_model1 = BuildPneModel("root_graph", graph1);
  FlowModelPtr flow_model = MakeShared<ge::FlowModel>(graph1);
  flow_model->AddSubModel(ge_root_model1, PNE_ID_NPU);

  FlowModelBuilder flow_model_builder;
  FlowModelBuilder::CacheParam cache_param = {true, true, true};
  EXPECT_EQ(flow_model_builder.RemoveDataFlowSubgraphs(flow_model, cache_param), SUCCESS);
}

TEST_F(FlowModelBuilderTest, ProcessNetOutput_failed) {
  FlowModelBuilder flow_model_builder;
  ComputeGraphPtr compute_graph;
  EXPECT_NE(flow_model_builder.ProcessNetOutput(compute_graph), SUCCESS);
}
}  // namespace ge

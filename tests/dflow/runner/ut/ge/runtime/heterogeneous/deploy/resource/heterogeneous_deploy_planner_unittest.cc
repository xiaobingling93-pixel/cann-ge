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
#include "dflow/base/model/endpoint.h"

#include <gmock/gmock.h>
#include "macro_utils/dt_public_scope.h"
#include "deploy/resource/resource_manager.h"
#include "deploy/resource/heterogeneous_deploy_planner.h"
#include "deploy/flowrm/flow_route_planner.h"
#include "dflow/base/model/model_deploy_resource.h"
#include "common/config/configurations.h"
#include "macro_utils/dt_public_unscope.h"

#include "graph/passes/graph_builder_utils.h"
#include "graph/build/graph_builder.h"
#include "stub_models.h"
#include "graph/ge_local_context.h"
#include "stub/heterogeneous_stub_env.h"
#include "depends/runtime/src/runtime_stub.h"
#include "dflow/inc/data_flow/model/flow_model_helper.h"
#include "depends/mmpa/src/mmpa_stub.h"

using namespace std;

namespace ge {
namespace {
class MockMmpa : public MmpaStubApiGe {
public:
  int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
    (void)strncpy_s(realPath, realPathLen, path, strlen(path));
    return EN_OK;
  }

  int32_t Sleep(UINT32 microSecond) override {
    return 0;
  }
};
}
class HeterogeneousDeployPlannerTest : public testing::Test {
 protected:
  void SetUp() override {
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  }
  void TearDown() override {
    MmpaStub::GetInstance().Reset();
    RuntimeStub::Reset();
  }

  std::vector<DeployPlan::DeviceInfo>
      double_node_list{DeployPlan::DeviceInfo{1, 0, 0},
                                  DeployPlan::DeviceInfo{0, 1, 0},
                                  DeployPlan::DeviceInfo{0, 1, 1}};
  std::vector<DeployPlan::DeviceInfo>
      double_device_list{DeployPlan::DeviceInfo{0, 1, 0}, DeployPlan::DeviceInfo{0, 1, 1}};
  std::vector<DeployPlan::DeviceInfo> single_device_with_cpu_list{DeployPlan::DeviceInfo{1, 0, 0},
                                                                  DeployPlan::DeviceInfo{0, 0, 0}};
  std::vector<DeployPlan::DeviceInfo> double_device_with_cpu_list{DeployPlan::DeviceInfo{1, 0, 0},
                                                                  DeployPlan::DeviceInfo{0, 0, 0},
                                                                  DeployPlan::DeviceInfo{0, 0, 1}};
  std::vector<DeployPlan::DeviceInfo> four_device_with_cpu_list{DeployPlan::DeviceInfo{1, 0, 0},
                                                                DeployPlan::DeviceInfo{0, 0, 0},
                                                                DeployPlan::DeviceInfo{0, 0, 1},
                                                                DeployPlan::DeviceInfo{0, 1, 0},
                                                                DeployPlan::DeviceInfo{0, 1, 1}};
  std::vector<DeployPlan::DeviceInfo> single_device_list{DeployPlan::DeviceInfo{0, 1, 0}};
  std::vector<DeployPlan::DeviceInfo> single_host_list{DeployPlan::DeviceInfo{1, 0, 0}};
  std::vector<DeployPlan::DeviceInfo> device_list_2_dev
      {DeployPlan::DeviceInfo{0, 1, 0}, DeployPlan::DeviceInfo{0, 2, 0}};

  void BuildDeviceInfos(const std::vector<DeployPlan::DeviceInfo> &device_list,
                        bool is_multi_cluster = false,
                        bool has_host_flowgw = false) {
    ResourceManager::GetInstance().device_info_map_.clear();
    mock_device_list_.clear();
    mock_device_list_.resize(device_list.size());
    int32_t device_index = 0;    
    std::map<int32_t, std::vector<int32_t>> npu_node_id_to_mesh;
    std::map<int32_t, std::vector<int32_t>> cpu_node_id_to_mesh;
    std::set<int32_t> npu_node_ids;
    for (const auto &device : device_list) {
      auto node_id = device.GetNodeId();
      auto type = static_cast<DeviceType>(device.GetType());
      if (type == NPU) {
        npu_node_ids.emplace(node_id);
        npu_node_id_to_mesh[node_id] = {0, static_cast<int>(npu_node_ids.size() - 1)};
        if (!is_multi_cluster) {
          npu_node_id_to_mesh[node_id].insert(npu_node_id_to_mesh[node_id].begin(), 0);
        }
      } else {
        cpu_node_id_to_mesh[node_id] = {0, node_id};
        if (!is_multi_cluster) {
          npu_node_id_to_mesh[node_id].insert(npu_node_id_to_mesh[node_id].begin(), 0);
        }
      }
    }
    for (const auto &device : device_list) {
      auto type = static_cast<DeviceType>(device.GetType()); 
      auto node_id = device.GetNodeId();
      auto device_id = device.GetDeviceId();
      mock_device_list_[device_index].SetNodeId(node_id);
      mock_device_list_[device_index].SetDeviceId(device_id);
      mock_device_list_[device_index].SetHcomDeviceId(device_id);
      mock_device_list_[device_index].SetPhyDeviceId(device_id);
      mock_device_list_[device_index].SetDeviceType(type);
      if (type == NPU) {
        mock_device_list_[device_index].SetDeviceIndex(device_id);
        mock_device_list_[device_index].SetNodeMeshIndex(npu_node_id_to_mesh[node_id]);
      } else {
        mock_device_list_[device_index].SetDeviceIndex(-1);
        mock_device_list_[device_index].SetNodeMeshIndex(cpu_node_id_to_mesh[node_id]);
      }
      if (is_multi_cluster) {
        mock_device_list_[device_index].SetSupportFlowgw(has_host_flowgw);
      }
      std::cout << "device mesh = " << mock_device_list_[device_index].ToIndex().c_str()
                << ", type = " << type
                << ", node_id = " << node_id
                << ", device_id = " << device_id
                << ", device_index = " << mock_device_list_[device_index].GetDeviceIndex()
                << ", index = " << device_index << std::endl;
      ResourceManager::GetInstance().device_info_map_[node_id][device_id][type] = &(mock_device_list_[device_index]);
      device_index++;
    }
  }

 private:
  std::vector<DeviceInfo> mock_device_list_;
};

namespace {
class MockHeterogeneousExchangeDeployer : public HeterogeneousExchangeDeployer {
 public:
  MockHeterogeneousExchangeDeployer(ExchangeService &exchange_service,
                             deployer::FlowRoutePlan route_plan,
                             FlowGwClientManager &client_manager)
      : HeterogeneousExchangeDeployer(exchange_service, std::move(route_plan), client_manager) {}
  MOCK_METHOD1(BindRoute, Status(const std::vector<std::pair<const ExchangeEndpoint *,
                                                             const ExchangeEndpoint *>> &));
};
}  // namespace

TEST_F(HeterogeneousDeployPlannerTest, TestBuildTransferInfo) {
  DeployPlan deploy_plan;
  DeployPlan::DeviceInfo local_device(CPU, 0, 0);
  ASSERT_EQ(HeterogeneousDeployPlanner().BuildTransferPlan(local_device, {}, deploy_plan), SUCCESS);
  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 0);

  ASSERT_EQ(HeterogeneousDeployPlanner().BuildTransferPlan(local_device, {local_device}, deploy_plan), SUCCESS);
  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 1);
}

TEST_F(HeterogeneousDeployPlannerTest, TestBuildDeployPlanWithProxy) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildGraphWithQueueBindings());
  ASSERT_TRUE(flow_model != nullptr);
  EXPECT_EQ(flow_model->GetSubmodels().size(), 2);
  auto model_relation = flow_model->GetModelRelation();
  ASSERT_TRUE(model_relation != nullptr);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.output_endpoint_names.size(), 1);

  DeployPlan deploy_plan;
  std::vector<GeRootModelPtr> models;
  BuildDeviceInfos(single_device_with_cpu_list, true);
  auto ret = HeterogeneousDeployPlanner(flow_model, single_device_with_cpu_list).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 8);
  ASSERT_EQ(deploy_plan.GetInputQueueIndices().size(), 2);
  ASSERT_EQ(deploy_plan.GetOutputQueueIndices().size(), 1);
}

TEST_F(HeterogeneousDeployPlannerTest, TestBuildServerDynamicSchedDeployPlanWithProxyAndMultipleDevice) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildGraphWithQueueBindings());
  ASSERT_TRUE(flow_model != nullptr);
  EXPECT_EQ(flow_model->GetSubmodels().size(), 2);
  auto model_relation = flow_model->GetModelRelation();
  ASSERT_TRUE(model_relation != nullptr);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.output_endpoint_names.size(), 1);

  DeployPlan deploy_plan;
  deploy_plan.SetIsDynamicSched(true);
  std::vector<GeRootModelPtr> models;
  std::vector<DeployPlan::DeviceInfo> node_list = {DeployPlan::DeviceInfo{1, 0, 0},
                                                   DeployPlan::DeviceInfo{0, 0, 0},
                                                   DeployPlan::DeviceInfo{0, 1, 0},
                                                   DeployPlan::DeviceInfo{0, 1, 1}};
  BuildDeviceInfos(node_list, true);
  auto ret = HeterogeneousDeployPlanner(flow_model, node_list).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  std::cout << "GetQueueInfoList" << deploy_plan.GetQueueInfoList().size() <<
    "GetQueueBindings" << deploy_plan.GetQueueBindings().size();
  std::cout << "GetInputQueueIndices" << deploy_plan.GetInputQueueIndices().size() <<
    "GetOutputQueueIndices" << deploy_plan.GetOutputQueueIndices().size() <<
    "GetStatusOutputQueueIndices" << deploy_plan.GetDynamicSchedPlan().GetStatusOutputQueueIndices().size() <<
    "GetSchedInputQueueIndices" << deploy_plan.GetDynamicSchedPlan().GetSchedInputQueueIndices().size() <<
    "GetSchedOutputQueueIndices" << deploy_plan.GetDynamicSchedPlan().GetSchedOutputQueueIndices().size() <<
    "GetDatagwRequestBindings" << deploy_plan.GetDynamicSchedPlan().GetDatagwRequestBindings().size() <<
    "GetEntryBindings" << deploy_plan.GetDynamicSchedPlan().GetEntryBindings().size() <<
    "GetModelIndexInfo" << deploy_plan.GetDynamicSchedPlan().GetModelIndexInfo().size() <<
    "GetModelInstanceNum" << deploy_plan.GetDynamicSchedPlan().GetModelInstanceNum().size();
  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 64);
  ASSERT_EQ(deploy_plan.GetInputQueueIndices().size(), 2);
  ASSERT_EQ(deploy_plan.GetOutputQueueIndices().size(), 1);
}

TEST_F(HeterogeneousDeployPlannerTest, TestBuildDeployPlanWithProxyAndMultipleDevice) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildGraphWithQueueBindings());
  ASSERT_TRUE(flow_model != nullptr);
  EXPECT_EQ(flow_model->GetSubmodels().size(), 2);
  auto model_relation = flow_model->GetModelRelation();
  ASSERT_TRUE(model_relation != nullptr);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.output_endpoint_names.size(), 1);

  DeployPlan deploy_plan;
  std::vector<GeRootModelPtr> models;
  std::vector<DeployPlan::DeviceInfo> node_list = {DeployPlan::DeviceInfo{1, 0, 0},
                                                   DeployPlan::DeviceInfo{0, 0, 0},
                                                   DeployPlan::DeviceInfo{0, 1, 0},
                                                   DeployPlan::DeviceInfo{0, 1, 1}};
  BuildDeviceInfos(node_list, true);
  auto ret = HeterogeneousDeployPlanner(flow_model, node_list).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 30);
  const auto &input_indices = deploy_plan.GetInputQueueIndices();
  ASSERT_EQ(input_indices.size(), 2);
  ASSERT_EQ(deploy_plan.GetOutputQueueIndices().size(), 1);
  // test invalid index
  ASSERT_EQ(deploy_plan.GetBroadcastIndices(100000).size(), 0);
  // test model multi-deployed
  ASSERT_EQ(deploy_plan.GetBroadcastIndices(input_indices[0]).size(), 0);
}
/**
 *      NetOutput
 *         |
 *         |
 *        PC_2
 *        |  \
 *       PC_1 |
 *     /    \
 *    |      |
 *  data1  data2
 */
TEST_F(HeterogeneousDeployPlannerTest, TestBuildDeployPlan) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildGraphWithQueueBindings());
  ASSERT_TRUE(flow_model != nullptr);
  EXPECT_EQ(flow_model->GetSubmodels().size(), 2);
  auto model_relation = flow_model->GetModelRelation();
  ASSERT_TRUE(model_relation != nullptr);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.output_endpoint_names.size(), 1);

  DeployPlan deploy_plan;
  std::vector<GeRootModelPtr> models;
  BuildDeviceInfos(single_device_list);
  auto ret = HeterogeneousDeployPlanner(flow_model, single_device_list).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  // data2 -> PC_1, data2 -> PC_2 ==== relation: data2(q) -> tagxx && tagxx -> pc1(q) && tagxx -> pc2(q)
  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 14);  // 8(queue) + 3(in group) + 3(out group)
  ASSERT_EQ(deploy_plan.GetGroupEntryInfoList().size(), 6);  
  ASSERT_EQ(deploy_plan.GetQueueBindings().size(), 7);
  ASSERT_EQ(deploy_plan.GetInputQueueIndices().size(), 2);
  ASSERT_EQ(deploy_plan.GetOutputQueueIndices().size(), 1);
}

/**
 *      NetOutput
 *         |
 *         |
 *        PC_2
 *        |  \
 *       PC_1 |
 *     /    \
 *    |      |
 *  data1  data2
 */
TEST_F(HeterogeneousDeployPlannerTest, TestDynamicSchedBuildDeployPlan) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildGraphWithQueueBindings());
  ASSERT_TRUE(flow_model != nullptr);
  EXPECT_EQ(flow_model->GetSubmodels().size(), 2);
  auto model_relation = flow_model->GetModelRelation();
  ASSERT_TRUE(model_relation != nullptr);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.output_endpoint_names.size(), 1);

  DeployPlan deploy_plan;
  deploy_plan.SetIsDynamicSched(true);
  std::vector<GeRootModelPtr> models;
  BuildDeviceInfos(single_device_list);
  auto ret = HeterogeneousDeployPlanner(flow_model, single_device_list).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  std::cout << "GetQueueInfoList" << deploy_plan.GetQueueInfoList().size() <<
    "GetQueueBindings" << deploy_plan.GetQueueBindings().size() <<
    "GetGroupEntryInfoList" << deploy_plan.GetGroupEntryInfoList().size();
  std::cout << "GetInputQueueIndices" << deploy_plan.GetInputQueueIndices().size() <<
    "GetOutputQueueIndices" << deploy_plan.GetOutputQueueIndices().size() <<
    "GetStatusOutputQueueIndices" << deploy_plan.GetDynamicSchedPlan().GetStatusOutputQueueIndices().size() <<
    "GetSchedInputQueueIndices" << deploy_plan.GetDynamicSchedPlan().GetSchedInputQueueIndices().size() <<
    "GetSchedOutputQueueIndices" << deploy_plan.GetDynamicSchedPlan().GetSchedOutputQueueIndices().size() <<
    "GetDatagwRequestBindings" << deploy_plan.GetDynamicSchedPlan().GetDatagwRequestBindings().size() <<
    "GetEntryBindings" << deploy_plan.GetDynamicSchedPlan().GetEntryBindings().size() <<
    "GetModelIndexInfo" << deploy_plan.GetDynamicSchedPlan().GetModelIndexInfo().size() <<
    "GetModelInstanceNum" << deploy_plan.GetDynamicSchedPlan().GetModelInstanceNum().size();
  // data2 -> PC_1, data2 -> PC_2 ==== relation: data2(q) -> tagxx && tagxx -> pc1(q) && tagxx -> pc2(q)
  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 28);  // 8(queue) + 3(in group) + 3(out group)
  ASSERT_EQ(deploy_plan.GetGroupEntryInfoList().size(), 12);
  ASSERT_EQ(deploy_plan.GetQueueBindings().size(), 15);
  ASSERT_EQ(deploy_plan.GetInputQueueIndices().size(), 2);
  ASSERT_EQ(deploy_plan.GetOutputQueueIndices().size(), 1);
}

/**
 *      NetOutput
 *         |
 *         |
 *        PC_2
 *        |  \
 *       PC_1 |
 *     /    \
 *    |      |
 *  data1  data2
 */
TEST_F(HeterogeneousDeployPlannerTest, TestBuildDeployPlanInputsFusion) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildGraphWithQueueBindings());
  ASSERT_TRUE(flow_model != nullptr);
  EXPECT_EQ(flow_model->GetSubmodels().size(), 2);
  auto model_relation = flow_model->GetModelRelation();
  ASSERT_TRUE(model_relation != nullptr);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.output_endpoint_names.size(), 1);

  std::map<std::string, std::string> options;
  options.insert(std::pair<std::string, std::string>(OPTION_EXEC_ENABLE_FUSION, "true"));
  ge::GetThreadLocalContext().SetGraphOption(options);

  DeployPlan deploy_plan;
  std::vector<GeRootModelPtr> models;
  BuildDeviceInfos(single_device_list);
  auto ret = HeterogeneousDeployPlanner(flow_model, single_device_list).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  // data2 -> PC_1, data2 -> PC_2
  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 12);
  ASSERT_EQ(deploy_plan.GetGroupEntryInfoList().size(), 4);
  ASSERT_EQ(deploy_plan.GetQueueBindings().size(), 5);
  ASSERT_EQ(deploy_plan.GetInputQueueIndices().size(), 2);
  ASSERT_EQ(deploy_plan.GetOutputQueueIndices().size(), 1);
  std::map<std::string, std::string> empty_options;
  ge::GetThreadLocalContext().SetGraphOption(empty_options);
}

/**
 *  NetOutput
 *     |
 *    PC_2
 *     |  
 *    PC_1
 *     |  
 *   data1
 */
TEST_F(HeterogeneousDeployPlannerTest, TestBuildDeployPlanWithSeriesModel) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildSeriesPartitionedCallGraph());
  ASSERT_TRUE(flow_model != nullptr);
  EXPECT_EQ(flow_model->GetSubmodels().size(), 2);
  auto model_relation = flow_model->GetModelRelation();
  ASSERT_TRUE(model_relation != nullptr);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.input_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.input_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.output_endpoint_names.size(), 1);

  auto model_iter = flow_model->GetSubmodels().begin();
  auto submodel1 = model_iter->second;
  submodel1->SetLogicDeviceId("0:0:0,0:0:1");
  ++model_iter;
  auto submodel2 = model_iter->second;
  submodel2->SetLogicDeviceId("0:0:0");

  DeployPlan deploy_plan;
  BuildDeviceInfos(double_device_list);
  auto ret = HeterogeneousDeployPlanner(flow_model, double_device_list).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);

  // data2 -> PC_1, data2 -> PC_2
  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 15);
  ASSERT_EQ(deploy_plan.GetGroupEntryInfoList().size(), 9);
  ASSERT_EQ(deploy_plan.GetQueueBindings().size(), 7);
  ASSERT_EQ(deploy_plan.GetInputQueueIndices().size(), 1);
  ASSERT_EQ(deploy_plan.GetOutputQueueIndices().size(), 1);
}

TEST_F(HeterogeneousDeployPlannerTest, TestBuildDeployPlanWithLogicDeviceId1) {
    Configurations::GetInstance().information_.has_cluster_define = true;
    auto flow_model = StubModels::BuildFlowModel(StubModels::BuildSinglePartitionedCallGraph());
    ASSERT_TRUE(flow_model != nullptr);
    EXPECT_EQ(flow_model->GetSubmodels().size(), 1);
    auto model_relation = flow_model->GetModelRelation();
    ASSERT_TRUE(model_relation != nullptr);
    ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 1);

    auto model_iter = flow_model->GetSubmodels().begin();
    auto submodel1 = model_iter->second.get();
    submodel1->SetLogicDeviceId("0:0:0,0:0:1");
    DeployPlan deploy_plan;
    BuildDeviceInfos(double_device_list, true);
    auto ret = HeterogeneousDeployPlanner(flow_model, double_device_list).BuildPlan(deploy_plan);
    ASSERT_EQ(ret, SUCCESS);
}

TEST_F(HeterogeneousDeployPlannerTest, TestBuildDeployPlanWithLogicDeviceId2) {
    auto flow_model = StubModels::BuildFlowModel(StubModels::BuildSinglePartitionedCallGraph());
    ASSERT_TRUE(flow_model != nullptr);
    EXPECT_EQ(flow_model->GetSubmodels().size(), 1);
    auto model_relation = flow_model->GetModelRelation();
    ASSERT_TRUE(model_relation != nullptr);
    ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 1);

    auto model_iter = flow_model->GetSubmodels().begin();
    auto submodel1 = model_iter->second.get();
    submodel1->SetModelType(PNE_ID_CPU);
    submodel1->SetLogicDeviceId("0:0:-1");
    DeployPlan deploy_plan;
    BuildDeviceInfos(single_host_list, true);
    auto ret = HeterogeneousDeployPlanner(flow_model, single_host_list).BuildPlan(deploy_plan);
    ASSERT_EQ(ret, SUCCESS);
}

/**
 *     NetOutput
 *         |
 *       PC_3
 *      /   \
 *    PC_1  PC2
 *    |      |
 *  data1  data2
 */
TEST_F(HeterogeneousDeployPlannerTest, TestBuildDeployPlan_2) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildGraphWithoutNeedForBindingQueues());
  ASSERT_TRUE(flow_model != nullptr);
  EXPECT_EQ(flow_model->GetSubmodels().size(), 3);
  auto model_relation = flow_model->GetModelRelation();
  ASSERT_TRUE(model_relation != nullptr);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 3);
  ASSERT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.input_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.input_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-3")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-3")->second.output_endpoint_names.size(), 1);

  DeployPlan deploy_plan;
  BuildDeviceInfos(single_device_list);
  auto ret = HeterogeneousDeployPlanner(flow_model, single_device_list).BuildPlan(deploy_plan);

  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 14);
  ASSERT_EQ(deploy_plan.GetGroupEntryInfoList().size(), 6);
  ASSERT_EQ(deploy_plan.GetQueueBindings().size(), 6);
}

/**
 *      NetOutput
 *         |
 *         |
 *       XXXX
 *     /     \
 *    |      |
 *  data1  data2
 */
TEST_F(HeterogeneousDeployPlannerTest, TestBuildPlanForSingleModel) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildSinglePartitionedCallGraph());
  ASSERT_TRUE(flow_model != nullptr);
  HeterogeneousDeployPlanner planner(flow_model, single_device_list);
  DeployPlan deploy_plan;
  BuildDeviceInfos(single_device_list);
  auto ret = planner.BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 12);
  ASSERT_EQ(deploy_plan.GetGroupEntryInfoList().size(), 6);
  ASSERT_EQ(deploy_plan.GetInputQueueIndices().size(), 2);
  ASSERT_EQ(deploy_plan.GetOutputQueueIndices().size(), 1);

  ASSERT_EQ(deploy_plan.GetQueueBindings().size(), 6);
  ASSERT_EQ(deploy_plan.GetSubmodels().size(), 1);
  auto iter = deploy_plan.submodels_.begin();
  auto &submodel_info = deploy_plan.submodels_[iter->first];
  ASSERT_EQ(submodel_info.input_queue_indices.size(), deploy_plan.GetInputQueueIndices().size());
  ASSERT_EQ(submodel_info.output_queue_indices.size(), deploy_plan.GetOutputQueueIndices().size());
}

/**
 *      NetOutput
 *         |
 *         |
 *       XXXX
 *     /     \
 *    |      |
 *  data1  data2
 */
TEST_F(HeterogeneousDeployPlannerTest, TestBuildPlanForSingleModel_host) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildSinglePartitionedCallGraph());
  ASSERT_TRUE(flow_model != nullptr);
  flow_model->GetSubmodels().begin()->second->SetModelType(PNE_ID_CPU);
  HeterogeneousDeployPlanner planner(flow_model, single_host_list);
  DeployPlan deploy_plan;
  BuildDeviceInfos(single_host_list);
  auto ret = planner.BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 3);
  ASSERT_EQ(deploy_plan.GetInputQueueIndices().size(), 2);
  ASSERT_EQ(deploy_plan.GetOutputQueueIndices().size(), 1);

  ASSERT_EQ(deploy_plan.GetQueueBindings().size(), 0);
  ASSERT_EQ(deploy_plan.GetSubmodels().size(), 1);
  auto iter = deploy_plan.submodels_.begin();
  auto &submodel_info = deploy_plan.submodels_[iter->first];
  ASSERT_EQ(submodel_info.input_queue_indices.size(), deploy_plan.GetInputQueueIndices().size());
  ASSERT_EQ(submodel_info.output_queue_indices.size(), deploy_plan.GetOutputQueueIndices().size());
  ASSERT_EQ(submodel_info.is_head, true);
}


TEST_F(HeterogeneousDeployPlannerTest, TestFailedDueToTheAbsenceOfModel) {
  auto flow_model = std::make_shared<FlowModel>();
  HeterogeneousDeployPlanner planner(flow_model, single_device_list);
  DeployPlan deploy_plan;
  auto ret = planner.BuildPlan(deploy_plan);
  ASSERT_EQ(ret, PARAM_INVALID);
}

void AddQueueDef(ModelRelation &model_relation, const std::string &name) {
  Endpoint queue_def(name, EndpointType::kQueue);
  QueueNodeUtils(queue_def).SetDepth(128L).SetEnqueuePolicy("FIFO");
  model_relation.endpoints.emplace_back(queue_def);
}

/**
 *   NetOutput      NetOutput
 *      |             |  \
 *    PC_2          PC_2  PC_2
 *     |             |  x  |
 *    PC_1          PC_1  PC_1
 *     |             |    /
 *    data1          data1
 */
TEST_F(HeterogeneousDeployPlannerTest, TestBuildDeployPlan_2_dev) {
  auto graph_1 = ge::MakeShared<ComputeGraph>("submodel-1");
  auto graph_2 = ge::MakeShared<ComputeGraph>("submodel-2");
  auto submodel_1 = StubModels::BuildRootModel(graph_1, false);
  auto submodel_2 = StubModels::BuildRootModel(graph_2, false);
  ASSERT_TRUE(submodel_1 != nullptr);
  ASSERT_TRUE(submodel_2 != nullptr);

  auto model_relation_ptr = std::make_shared<ModelRelation>();
  ASSERT_TRUE(model_relation_ptr != nullptr);
  ModelRelation &model_relation = *model_relation_ptr;
  model_relation.root_model_endpoint_info.input_endpoint_names = {"in-queue-1",};
  model_relation.root_model_endpoint_info.output_endpoint_names = {"out-queue-1"};
  model_relation.submodel_endpoint_infos["submodel-1"].model_name = "submodel-1";
  model_relation.submodel_endpoint_infos["submodel-1"].input_endpoint_names = {"in-queue-1"};
  model_relation.submodel_endpoint_infos["submodel-1"].output_endpoint_names = {"inner-queue-1"};
  model_relation.submodel_endpoint_infos["submodel-2"].model_name = "submodel-2";
  model_relation.submodel_endpoint_infos["submodel-2"].input_endpoint_names = {"inner-queue-1"};
  model_relation.submodel_endpoint_infos["submodel-2"].output_endpoint_names = {"out-queue-1"};
  AddQueueDef(model_relation, "in-queue-1");
  AddQueueDef(model_relation, "out-queue-1");
  AddQueueDef(model_relation, "inner-queue-1");

  DeployPlan deploy_plan;
  auto flow_model = std::make_shared<FlowModel>();

  ASSERT_TRUE(flow_model != nullptr);
  flow_model->AddSubModel(submodel_1, PNE_ID_NPU);
  flow_model->AddSubModel(submodel_2, PNE_ID_NPU);
  flow_model->SetModelRelation(model_relation_ptr);

  {
    HeterogeneousDeployPlanner planner(flow_model, device_list_2_dev);
    BuildDeviceInfos(device_list_2_dev);
    auto ret = planner.BuildPlan(deploy_plan);
    ASSERT_EQ(ret, SUCCESS);
    // (1 + 2 * 2)(output) + (2 * 2 + 1)(input) + 5(output group) + 5(input group) + 2(transit queue)
    // with remove relation in diffirent device
    ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 14);
    // 2 + 2 + 2 + 1 + 1(output group) + 1 + 1 + 1 + 1 + 2(input group)
    // PC_1->PC_2 contains 2 queue->queue
    // in PC_1's output group, entries are [tag, queue]
    // but in PC_2's input group, entries are [tag]
    ASSERT_EQ(deploy_plan.GetGroupEntryInfoList().size(), 8);
    // 5(output) + 5(input)
    ASSERT_EQ(deploy_plan.GetQueueBindings().size(), 6);
  }
}

/**
 *         NetOutput
 *            |
 *        Submodel-3
 *       /         \
 *    Submodel-1  Submodel-2
 *     |     |     |   \
 *    D1    D2    D3   D4
 */
TEST_F(HeterogeneousDeployPlannerTest, TestBuildDeployPlan_AllRawModel) {
  auto graph_1 = ge::MakeShared<ComputeGraph>("submodel-1");
  auto graph_2 = ge::MakeShared<ComputeGraph>("submodel-2");
  auto graph_3 = ge::MakeShared<ComputeGraph>("submodel-3");
  auto submodel_1 = StubModels::BuildRootModel(graph_1, false);
  auto submodel_2 = StubModels::BuildRootModel(graph_2, false);
  auto submodel_3 = StubModels::BuildRootModel(graph_3, false);
  ASSERT_TRUE(submodel_1 != nullptr);
  ASSERT_TRUE(submodel_2 != nullptr);
  ASSERT_TRUE(submodel_3 != nullptr);

  auto model_relation_ptr = std::make_shared<ModelRelation>();
  ASSERT_TRUE(model_relation_ptr != nullptr);
  ModelRelation &model_relation = *model_relation_ptr;
  model_relation.root_model_endpoint_info.input_endpoint_names = {"in-queue-1", "in-queue-2", "in-queue-3", "in-queue-4"};
  model_relation.root_model_endpoint_info.output_endpoint_names = {"out-queue-1"};
  model_relation.submodel_endpoint_infos["submodel-1"].model_name = "submodel-1";
  model_relation.submodel_endpoint_infos["submodel-1"].input_endpoint_names = {"in-queue-1", "in-queue-2"};
  model_relation.submodel_endpoint_infos["submodel-1"].output_endpoint_names = {"inner-queue-1"};
  model_relation.submodel_endpoint_infos["submodel-2"].model_name = "submodel-2";
  model_relation.submodel_endpoint_infos["submodel-2"].input_endpoint_names = {"in-queue-3", "in-queue-4"};
  model_relation.submodel_endpoint_infos["submodel-2"].output_endpoint_names = {"inner-queue-2"};
  model_relation.submodel_endpoint_infos["submodel-3"].model_name = "submodel-3";
  model_relation.submodel_endpoint_infos["submodel-3"].input_endpoint_names = {"inner-queue-1", "inner-queue-2"};
  model_relation.submodel_endpoint_infos["submodel-3"].output_endpoint_names = {"out-queue-1"};
  AddQueueDef(model_relation, "in-queue-1");
  AddQueueDef(model_relation, "in-queue-2");
  AddQueueDef(model_relation, "in-queue-3");
  AddQueueDef(model_relation, "in-queue-4");
  AddQueueDef(model_relation, "out-queue-1");
  AddQueueDef(model_relation, "inner-queue-1");
  AddQueueDef(model_relation, "inner-queue-2");

  DeployPlan deploy_plan;
  auto flow_model = std::make_shared<FlowModel>();
  ASSERT_TRUE(flow_model != nullptr);
  flow_model->AddSubModel(submodel_1, PNE_ID_NPU);
  flow_model->AddSubModel(submodel_2, PNE_ID_NPU);
  flow_model->AddSubModel(submodel_3, PNE_ID_NPU);
  flow_model->SetModelRelation(model_relation_ptr);
  HeterogeneousDeployPlanner planner(flow_model, single_device_list);
  BuildDeviceInfos(single_device_list);
  auto ret = planner.BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 22);
  ASSERT_EQ(deploy_plan.GetGroupEntryInfoList().size(), 10);
  ASSERT_EQ(deploy_plan.GetQueueBindings().size(), 10);
}

/**
 *     NetOutput
 *         |
 *       PC_3
 *      /   \         -> PC_4 2-inputs, 1-output
 *    PC_1  PC2
 *    |      |
 *  data1  data2
 *
 *
 *       NetOutput
 *          |
 *        PC_4_3
 *       /     \
 *    PC_4_1  PC_4_2
 *     |  |    |  \
 *    D1 D2   D3  D4
 */
TEST_F(HeterogeneousDeployPlannerTest, TestBuildDeployPlan_AllWrappedModels) {
  auto submodel_1 = StubModels::BuildFlowModel(StubModels::BuildGraphWithoutNeedForBindingQueues());
  auto submodel_2 = StubModels::BuildFlowModel(StubModels::BuildGraphWithoutNeedForBindingQueues());
  auto submodel_3 = StubModels::BuildFlowModel(StubModels::BuildGraphWithoutNeedForBindingQueues());
  ASSERT_TRUE(submodel_1 != nullptr);
  ASSERT_TRUE(submodel_2 != nullptr);
  ASSERT_TRUE(submodel_3 != nullptr);
  submodel_1->SetModelName("submodel-1");
  submodel_2->SetModelName("submodel-2");
  submodel_3->SetModelName("submodel-3");
  EXPECT_EQ(submodel_1->GetSubmodels().size(), 3);

  auto model_relation_ptr = std::make_shared<ModelRelation>();
  ASSERT_TRUE(model_relation_ptr != nullptr);
  ModelRelation &model_relation = *model_relation_ptr;
  model_relation.root_model_endpoint_info.input_endpoint_names = {"in-queue-1", "in-queue-2", "in-queue-3", "in-queue-4"};
  model_relation.root_model_endpoint_info.output_endpoint_names = {"out-queue-1"};
  model_relation.submodel_endpoint_infos["submodel-1"].input_endpoint_names = {"in-queue-1", "in-queue-2"};
  model_relation.submodel_endpoint_infos["submodel-1"].output_endpoint_names = {"inner-queue-1"};
  model_relation.submodel_endpoint_infos["submodel-2"].input_endpoint_names = {"in-queue-3", "in-queue-4"};
  model_relation.submodel_endpoint_infos["submodel-2"].output_endpoint_names = {"inner-queue-2"};
  model_relation.submodel_endpoint_infos["submodel-3"].input_endpoint_names = {"inner-queue-1", "inner-queue-2"};
  model_relation.submodel_endpoint_infos["submodel-3"].output_endpoint_names = {"out-queue-1"};
  AddQueueDef(model_relation, "in-queue-1");
  AddQueueDef(model_relation, "in-queue-2");
  AddQueueDef(model_relation, "in-queue-3");
  AddQueueDef(model_relation, "in-queue-4");
  AddQueueDef(model_relation, "out-queue-1");
  AddQueueDef(model_relation, "inner-queue-1");
  AddQueueDef(model_relation, "inner-queue-2");

  DeployPlan deploy_plan;
  HeterogeneousDeployPlanner planner({submodel_1, submodel_2, submodel_3}, &model_relation,single_device_list);
  BuildDeviceInfos(single_device_list);
  auto ret = planner.BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 28);
  ASSERT_EQ(deploy_plan.GetGroupEntryInfoList().size(), 10);
  ASSERT_EQ(deploy_plan.GetQueueBindings().size(), 10);
  ASSERT_EQ(deploy_plan.GetSubmodels().size(), 9);
}

/**
 *     NetOutput
 *         |
 *       PC_3
 *      /   \         -> PC_4 2-inputs, 1-output
 *    PC_1  PC2
 *    |      |
 *  data1  data2
 *
 *
 *       NetOutput
 *          |
 *        PC_4_3
 *       /     \
 *    PC_4_1  Submodel-2
 *     |  |    |  \
 *    D1 D2   D3  D4
 */
TEST_F(HeterogeneousDeployPlannerTest, TestBuildDeployPlan_Mixed) {
  auto submodel_1 = StubModels::BuildFlowModel(StubModels::BuildGraphWithoutNeedForBindingQueues());
  auto submodel_2 = std::make_shared<FlowModel>();  // single model
  auto submodel_3 = StubModels::BuildFlowModel(StubModels::BuildGraphWithoutNeedForBindingQueues());
  ASSERT_TRUE(submodel_1 != nullptr);
  ASSERT_TRUE(submodel_2 != nullptr);
  ASSERT_TRUE(submodel_3 != nullptr);
  submodel_1->SetModelName("submodel-1");
  submodel_2->SetModelName("submodel-2");
  submodel_3->SetModelName("submodel-3");
  EXPECT_EQ(submodel_1->GetSubmodels().size(), 3);

  auto model_relation_ptr = std::make_shared<ModelRelation>();
  ASSERT_TRUE(model_relation_ptr != nullptr);
  ModelRelation &model_relation = *model_relation_ptr;
  model_relation.root_model_endpoint_info.input_endpoint_names = {"in-queue-1", "in-queue-2", "in-queue-3", "in-queue-4"};
  model_relation.root_model_endpoint_info.output_endpoint_names = {"out-queue-1"};
  model_relation.submodel_endpoint_infos["submodel-1"].input_endpoint_names = {"in-queue-1", "in-queue-2"};
  model_relation.submodel_endpoint_infos["submodel-1"].output_endpoint_names = {"inner-queue-1"};
  model_relation.submodel_endpoint_infos["submodel-2"].input_endpoint_names = {"in-queue-3", "in-queue-4"};
  model_relation.submodel_endpoint_infos["submodel-2"].output_endpoint_names = {"inner-queue-2"};
  model_relation.submodel_endpoint_infos["submodel-3"].input_endpoint_names = {"inner-queue-1", "inner-queue-2"};
  model_relation.submodel_endpoint_infos["submodel-3"].output_endpoint_names = {"out-queue-1"};
  AddQueueDef(model_relation, "in-queue-1");
  AddQueueDef(model_relation, "in-queue-2");
  AddQueueDef(model_relation, "in-queue-3");
  AddQueueDef(model_relation, "in-queue-4");
  AddQueueDef(model_relation, "out-queue-1");
  AddQueueDef(model_relation, "inner-queue-1");
  AddQueueDef(model_relation, "inner-queue-2");

  DeployPlan deploy_plan;
  HeterogeneousDeployPlanner planner({submodel_1, submodel_2, submodel_3}, &model_relation, single_device_list);
  BuildDeviceInfos(single_device_list);
  auto ret = planner.BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 26);
  ASSERT_EQ(deploy_plan.GetGroupEntryInfoList().size(), 10);
  ASSERT_EQ(deploy_plan.GetQueueBindings().size(), 10);
  ASSERT_EQ(deploy_plan.GetSubmodels().size(), 7);
}

TEST_F(HeterogeneousDeployPlannerTest, TestBuildDeployPlan_NoModel) {
  DeployPlan deploy_plan;
  ASSERT_EQ(HeterogeneousDeployPlanner(nullptr, {DeployPlan::DeviceInfo{0, 0, 0}}).BuildPlan(deploy_plan), PARAM_INVALID);
}

TEST_F(HeterogeneousDeployPlannerTest, TestBuildDeployPlan_MultiplyModelWithNoRelation) {
  auto graph_1 = ge::MakeShared<ComputeGraph>("subgraph-1");
  auto graph_2 = ge::MakeShared<ComputeGraph>("subgraph-2");
  auto submodel_1 = StubModels::BuildRootModel(graph_1, false);
  auto submodel_2 = StubModels::BuildRootModel(graph_2, false);
  ASSERT_TRUE(submodel_1 != nullptr);
  ASSERT_TRUE(submodel_2 != nullptr);
  DeployPlan deploy_plan;
  auto flow_model = std::make_shared<FlowModel>();
  ASSERT_TRUE(flow_model != nullptr);
  flow_model->AddSubModel(submodel_1, PNE_ID_NPU);
  flow_model->AddSubModel(submodel_2, PNE_ID_NPU);
  ASSERT_EQ(HeterogeneousDeployPlanner(flow_model, {DeployPlan::DeviceInfo{0, 0, 0}}).BuildPlan(deploy_plan), PARAM_INVALID);
}

/**
 *  NetOutput
 *      |
 *      |
 *     XXXX
 *     /  \
 *    |    |
 *  data1  data2
 */
TEST_F(HeterogeneousDeployPlannerTest, TestBuildPlanForSingleModel_2PG) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildSinglePartitionedCallGraph());
  ASSERT_TRUE(flow_model != nullptr);
  HeterogeneousDeployPlanner planner(flow_model, double_device_list);
  DeployPlan deploy_plan;
  BuildDeviceInfos(double_device_list);
  auto ret = planner.BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  // 5(input queue) + 4(output queue) + 5(input group) + 4(output group)
  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 18);
  ASSERT_EQ(deploy_plan.GetGroupEntryInfoList().size(), 12);
  ASSERT_EQ(deploy_plan.GetInputQueueIndices().size(), 2);
  ASSERT_EQ(deploy_plan.GetOutputQueueIndices().size(), 1);

  ASSERT_EQ(deploy_plan.GetQueueBindings().size(), 9);
  ASSERT_EQ(deploy_plan.GetSubmodels().size(), 2);
  ASSERT_EQ(deploy_plan.GetGroups().size(), 9);
}

TEST_F(HeterogeneousDeployPlannerTest, TestBuildPlanForSingleModel_DeployResource_not_soc) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildSinglePartitionedCallGraph());
  ASSERT_TRUE(flow_model != nullptr);
  auto deploy_resource_ptr = std::make_shared<ModelDeployResource>();
  ASSERT_TRUE(deploy_resource_ptr != nullptr);
  ModelDeployResource &deploy_resource = *deploy_resource_ptr;
  deploy_resource.resource_type = "X86_64";
  const auto &root_model = flow_model->GetSubmodels().begin()->second;
  root_model->SetModelType(PNE_ID_UDF);
  root_model->SetDeployResource(deploy_resource_ptr);

  DeviceInfo local_device(0, CPU, 0);
  local_device.SetResourceType("X86_64");

  DeviceInfo remote_device_0(0, NPU, 0);
  remote_device_0.SetResourceType("Aarch64");

  DeviceInfo remote_device_1(0, NPU, 1);
  remote_device_1.SetResourceType("Aarch64");
  std::vector<DeviceInfo> device_info_list = std::move(ResourceManager::GetInstance().device_info_list_);
  std::map<int32_t, std::map<int32_t, std::map<DeviceType, const DeviceInfo *>>> device_info_map = std::move(
    ResourceManager::GetInstance().device_info_map_);
  ResourceManager::GetInstance().device_info_list_.emplace_back(local_device);
  ResourceManager::GetInstance().device_info_list_.emplace_back(remote_device_0);
  ResourceManager::GetInstance().device_info_list_.emplace_back(remote_device_1);
  ResourceManager::GetInstance().device_info_map_[0][0][CPU] = &local_device;
  ResourceManager::GetInstance().device_info_map_[1][0][NPU] = &remote_device_0;
  ResourceManager::GetInstance().device_info_map_[1][1][NPU] = &remote_device_1;

  std::vector<DeployPlan::DeviceInfo> device_list{DeployPlan::DeviceInfo{1, 0, 0},
                                                  DeployPlan::DeviceInfo{0, 1, 0},
                                                  DeployPlan::DeviceInfo{0, 1, 1}};
  HeterogeneousDeployPlanner planner(flow_model, device_list);
  DeployPlan deploy_plan;
  BuildDeviceInfos(device_list);

  auto ret = planner.BuildPlan(deploy_plan);
  ASSERT_NE(ret, SUCCESS);
  ResourceManager::GetInstance().device_info_list_ = std::move(device_info_list);
  ResourceManager::GetInstance().device_info_map_ = std::move(device_info_map);
}

/**
 *  NetOutput
 *      |
 *      |
 *     XXXX
 *     /  \
 *    |    |
 *  data1  data2
 */
TEST_F(HeterogeneousDeployPlannerTest, TestBuildPlanForSingleModel_2PG_UseNode0) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildSinglePartitionedCallGraph());
  ASSERT_TRUE(flow_model != nullptr);
  std::map<std::string, std::string> options;
  options.insert(std::pair<std::string, std::string>("ge.exec.logicalDeviceClusterDeployMode", "SINGLE"));
  options.insert(std::pair<std::string, std::string>("ge.exec.logicalDeviceId", "[0:0]"));
  ge::GetThreadLocalContext().SetGraphOption(options);
  std::vector<DeployPlan::DeviceInfo> device_list{DeployPlan::DeviceInfo{1, 0, 0},
                                                  DeployPlan::DeviceInfo{0, 1, 0},
                                                  DeployPlan::DeviceInfo{0, 1, 1}};
  HeterogeneousDeployPlanner planner(flow_model, device_list);
  DeployPlan deploy_plan;
  BuildDeviceInfos(device_list);
  auto ret = planner.BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(planner.model_relation_.submodel_endpoint_infos.size(), 1);
  std::map<std::string, std::string> empty_options;
  ge::GetThreadLocalContext().SetGraphOption(empty_options);
}

/**
 *      NetOutput
 *         |
 *         |
 *        PC_2
 *        |  \
 *       PC_1 |
 *     /    \
 *    |      |
 *  data1  data2
 */
TEST_F(HeterogeneousDeployPlannerTest, TestBuildPlanFor2Models_2PG) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildGraphWithQueueBindings());
  ASSERT_TRUE(flow_model != nullptr);
  EXPECT_EQ(flow_model->GetSubmodels().size(), 2);
  auto model_relation = flow_model->GetModelRelation();
  ASSERT_TRUE(model_relation != nullptr);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.output_endpoint_names.size(), 1);

  DeployPlan deploy_plan;
  BuildDeviceInfos(double_device_list);
  auto ret = HeterogeneousDeployPlanner(flow_model, double_device_list).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  // data2 -> PC_1, data2 -> PC_2
  std::cout << "GetQueueInfoList().size():" << deploy_plan.GetQueueInfoList().size() << std::endl;
  std::cout << "GetGroupEntryInfoList().size():" << deploy_plan.GetGroupEntryInfoList().size() << std::endl;
  std::cout << "GetQueueBindings().size():" << deploy_plan.GetQueueBindings().size() << std::endl;
  std::cout << "GetInputQueueIndices().size():" << deploy_plan.GetInputQueueIndices().size() << std::endl;
  std::cout << "GetOutputQueueIndices().size():" << deploy_plan.GetOutputQueueIndices().size() << std::endl;
  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 25);
  ASSERT_EQ(deploy_plan.GetGroupEntryInfoList().size(), 16);
  ASSERT_EQ(deploy_plan.GetQueueBindings().size(), 12);
  ASSERT_EQ(deploy_plan.GetInputQueueIndices().size(), 2);
  ASSERT_EQ(deploy_plan.GetOutputQueueIndices().size(), 1);

  DeviceInfo local_device(0, CPU, 0);
  local_device.SetHcomDeviceId(0);
  local_device.SetHostIp("127.0.0.1");
  local_device.SetDeviceIp("127.0.0.1");
  local_device.SetDgwPort(1);

  DeviceInfo remote_device_0(1, NPU, 0);
  remote_device_0.SetHcomDeviceId(0);
  remote_device_0.SetHostIp("127.0.0.2");
  remote_device_0.SetDeviceIp("127.0.0.2");
  remote_device_0.SetDgwPort(1);

  DeviceInfo remote_device_1(1, NPU, 1);
  remote_device_1.SetHcomDeviceId(1);
  remote_device_1.SetHostIp("127.0.0.2");
  remote_device_1.SetDeviceIp("127.0.0.2");
  remote_device_1.SetDgwPort(2);
  ResourceManager::GetInstance().device_info_list_.emplace_back(local_device);
  ResourceManager::GetInstance().device_info_list_.emplace_back(remote_device_0);
  ResourceManager::GetInstance().device_info_list_.emplace_back(remote_device_1);
  ResourceManager::GetInstance().device_info_map_[0][0][CPU] = &local_device;
  ResourceManager::GetInstance().device_info_map_[1][0][NPU] = &remote_device_0;
  ResourceManager::GetInstance().device_info_map_[1][1][NPU] = &remote_device_1;

  Configurations::GetInstance().information_.host_info.data_panel.ipaddr = "127.0.0.1";

  HcomRank hcom_rank;
  hcom_rank.rank_id = "0";
  hcom_rank.device_id = "0";
  HcomRank hcom_rank_1;
  hcom_rank_1.rank_id = "1";
  hcom_rank_1.device_id = "0";
  HcomRank hcom_rank_2;
  hcom_rank_2.rank_id = "2";
  hcom_rank_2.device_id = "1";
  DeployContext::LocalContext().GetRankTableBuilder().ip_rank_map_["127.0.0.1:0"] = &hcom_rank;
  DeployContext::LocalContext().GetRankTableBuilder().ip_rank_map_["127.0.0.2:0"] = &hcom_rank_1;
  DeployContext::LocalContext().GetRankTableBuilder().ip_rank_map_["127.0.0.2:1"] = &hcom_rank_2;

  auto CountEndpointByType = [](const deployer::FlowRoutePlan &exchange_plan, int32_t type) {
    return std::count_if(exchange_plan.endpoints().begin(), exchange_plan.endpoints().end(),
                         [type](const deployer::EndpointDesc &endpoint_desc) -> bool {
                           return endpoint_desc.type() == type;
                         });
  };
  {
    deployer::FlowRoutePlan exchange_plan{};
    DeployPlan::DeviceInfo target_device_info(1, 0, 0);
    std::map<std::string, DeployPlan::DeviceInfo> key_2_device;
    EXPECT_EQ(FlowRoutePlanner::ResolveFlowRoutePlan(deploy_plan, 0, exchange_plan), SUCCESS);
    EXPECT_EQ(CountEndpointByType(exchange_plan, static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue)), 3);
    EXPECT_EQ(CountEndpointByType(exchange_plan, static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeTag)), 8);
    EXPECT_EQ(CountEndpointByType(exchange_plan, static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeGroup)), 4);
    FlowRoutePlanner::PrintFlowRoutePlan(exchange_plan);
  }
  {
    deployer::FlowRoutePlan exchange_plan{};
    EXPECT_EQ(FlowRoutePlanner::ResolveFlowRoutePlan(deploy_plan, 0, exchange_plan), SUCCESS);
    EXPECT_EQ(CountEndpointByType(exchange_plan, static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue)), 3);
    EXPECT_EQ(CountEndpointByType(exchange_plan, static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeTag)), 8);
    EXPECT_EQ(CountEndpointByType(exchange_plan, static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeGroup)), 4);

    FlowGwClientManager client_manager;
    client_manager.GetOrCreateClient(0, 0, {0}, false);
    client_manager.GetOrCreateClient(1, 0, {0}, false);
    client_manager.GetOrCreateClient(0, 1, {0}, false);
    stub::MockExchangeService exchange_service;
    MockHeterogeneousExchangeDeployer route_deployer(exchange_service, exchange_plan, client_manager);
    ExchangeRoute flow_route;

    auto check_func = [](const std::vector<std::pair<const ExchangeEndpoint *,
                                                     const ExchangeEndpoint *>> &routes) -> Status {
      for (auto &route : routes) {
        auto &src_endpoint = route.first;
        auto &dst_endpoint = route.second;
        cout << "src endpoint = [" << src_endpoint->DebugString().c_str() << "]" << endl;
        cout << "dst endpoint = [" << dst_endpoint->DebugString().c_str() << "]" << endl;
      }
      return SUCCESS;
    };
    EXPECT_CALL(route_deployer, BindRoute)
        .WillRepeatedly(testing::Invoke(check_func));
    EXPECT_EQ(route_deployer.Deploy(flow_route), SUCCESS);
    client_manager.Finalize();
  }
}

/**
 *      NetOutput
 *         |
 *         |
 *        PC_2
 *        |  \
 *       PC_1 |
 *     /    \
 *    |      |
 *  data1  data2
 */
TEST_F(HeterogeneousDeployPlannerTest, TestBuildPlanFor2Models_Host) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildGraphWithQueueBindings());
  ASSERT_TRUE(flow_model != nullptr);
  EXPECT_EQ(flow_model->GetSubmodels().size(), 2);
  auto model_relation = flow_model->GetModelRelation();
  for (auto &model : flow_model->GetSubmodels()) {
    model.second->SetModelType(PNE_ID_CPU);
  }

  ASSERT_TRUE(model_relation != nullptr);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.output_endpoint_names.size(), 1);

  DeployPlan deploy_plan;
  BuildDeviceInfos(single_host_list);
  auto ret = HeterogeneousDeployPlanner(flow_model, single_host_list).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  // data2 -> PC_1, data2 -> PC_2
  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 8);  // 6 + 2(group)
  ASSERT_EQ(deploy_plan.GetGroupEntryInfoList().size(), 2);
  ASSERT_NE(deploy_plan.GetGroupEntryInfoList()[0].ref_index, -1);
  ASSERT_NE(deploy_plan.GetGroupEntryInfoList()[1].ref_index, -1);
  ASSERT_EQ(deploy_plan.GetQueueBindings().size(), 2);
  ASSERT_EQ(deploy_plan.GetInputQueueIndices().size(), 2);
  ASSERT_EQ(deploy_plan.GetOutputQueueIndices().size(), 1);

  DeviceInfo local_device(0, CPU, 0);
  local_device.SetHostIp("127.0.0.1");
  local_device.SetDgwPort(0);
  ResourceManager::GetInstance().device_info_list_.emplace_back(local_device);
  ResourceManager::GetInstance().device_info_map_[0][0][CPU] = &local_device;
  Configurations::GetInstance().information_.host_info.data_panel.ipaddr = "127.0.0.1";
  HcomRank hcom_rank;
  hcom_rank.rank_id = "0";
  hcom_rank.device_id = "0";
  DeployContext::LocalContext().GetRankTableBuilder().ip_rank_map_["127.0.0.1:0"] = &hcom_rank;

  auto CountEndpointByType = [](const deployer::FlowRoutePlan &exchange_plan, int32_t type) {
    return std::count_if(exchange_plan.endpoints().begin(), exchange_plan.endpoints().end(),
                         [type](const deployer::EndpointDesc &endpoint_desc) -> bool {
                           return endpoint_desc.type() == type;
                         });
  };
  deployer::FlowRoutePlan exchange_plan{};
  DeployPlan::DeviceInfo target_device_info(1, 0, 0);
  EXPECT_EQ(FlowRoutePlanner::ResolveFlowRoutePlan(deploy_plan, 0, exchange_plan), SUCCESS);
  EXPECT_EQ(CountEndpointByType(exchange_plan, static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue)), 6);
  EXPECT_EQ(CountEndpointByType(exchange_plan, static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeTag)), 0);
  EXPECT_EQ(CountEndpointByType(exchange_plan, static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeGroup)), 2);
  FlowRoutePlanner::PrintFlowRoutePlan(exchange_plan);

  FlowGwClientManager client_manager;
  client_manager.GetOrCreateClient(0, 0, {0}, false);
  client_manager.GetOrCreateClient(1, 0, {0}, false);
  client_manager.GetOrCreateClient(0, 1, {0}, false);
  stub::MockExchangeService exchange_service;
  MockHeterogeneousExchangeDeployer route_deployer(exchange_service, exchange_plan, client_manager);
  ExchangeRoute flow_route;

  auto check_func = [](const std::vector<std::pair<const ExchangeEndpoint *,
                                                   const ExchangeEndpoint *>> &routes) -> Status {
    for (auto &route : routes) {
      auto &src_endpoint = route.first;
      auto &dst_endpoint = route.second;
      cout << "src endpoint = [" << src_endpoint->DebugString().c_str() << "]" << endl;
      cout << "dst endpoint = [" << dst_endpoint->DebugString().c_str() << "]" << endl;
    }
    return SUCCESS;
  };
  EXPECT_CALL(route_deployer, BindRoute)
      .WillRepeatedly(testing::Invoke(check_func));
  EXPECT_EQ(route_deployer.Deploy(flow_route), SUCCESS);
  client_manager.Finalize();
}

/**
 *     NetOutput
 *         |
 *       PC_3
 *      /   \         -> PC_4 2-inputs, 1-output
 *    PC_1  PC2
 *    |      |
 *  data1  data2
 *
 *
 *       NetOutput
 *          |
 *        PC_4_3
 *       /     \
 *    PC_4_1  PC_4_2
 *     |  |    |  \
 *    D1 D2   D3  D4
 */
TEST_F(HeterogeneousDeployPlannerTest, TestFlattenModelRelation_2_level) {
  auto submodel_1 = StubModels::BuildFlowModel(StubModels::BuildGraphWithoutNeedForBindingQueues("sub1_"));
  auto submodel_2 = StubModels::BuildFlowModel(StubModels::BuildGraphWithoutNeedForBindingQueues("sub2_"));
  auto submodel_3 = StubModels::BuildFlowModel(StubModels::BuildGraphWithoutNeedForBindingQueues("sub3_"));
  ASSERT_TRUE(submodel_1 != nullptr);
  ASSERT_TRUE(submodel_2 != nullptr);
  ASSERT_TRUE(submodel_3 != nullptr);
  submodel_1->SetModelName("submodel-1");
  submodel_2->SetModelName("submodel-2");
  submodel_3->SetModelName("submodel-3");

  auto model_relation_ptr = std::make_shared<ModelRelation>();
  ASSERT_TRUE(model_relation_ptr != nullptr);
  ModelRelation &model_relation = *model_relation_ptr;
  model_relation.root_model_endpoint_info.input_endpoint_names = {"in-queue-1", "in-queue-2", "in-queue-3", "in-queue-4"};
  model_relation.root_model_endpoint_info.output_endpoint_names = {"out-queue-1"};
  model_relation.submodel_endpoint_infos["submodel-1"].model_name = "submodel-1";
  model_relation.submodel_endpoint_infos["submodel-1"].input_endpoint_names = {"in-queue-1", "in-queue-2"};
  model_relation.submodel_endpoint_infos["submodel-1"].output_endpoint_names = {"inner-queue-1"};
  model_relation.submodel_endpoint_infos["submodel-2"].model_name = "submodel-2";
  model_relation.submodel_endpoint_infos["submodel-2"].input_endpoint_names = {"in-queue-3", "in-queue-4"};
  model_relation.submodel_endpoint_infos["submodel-2"].output_endpoint_names = {"inner-queue-2"};
  model_relation.submodel_endpoint_infos["submodel-3"].model_name = "submodel-3";
  model_relation.submodel_endpoint_infos["submodel-3"].input_endpoint_names = {"inner-queue-1", "inner-queue-2"};
  model_relation.submodel_endpoint_infos["submodel-3"].output_endpoint_names = {"out-queue-1"};
  AddQueueDef(model_relation, "in-queue-1");
  AddQueueDef(model_relation, "in-queue-2");
  AddQueueDef(model_relation, "in-queue-3");
  AddQueueDef(model_relation, "in-queue-4");
  AddQueueDef(model_relation, "out-queue-1");
  AddQueueDef(model_relation, "inner-queue-1");
  AddQueueDef(model_relation, "inner-queue-2");

  auto flow_model = std::make_shared<FlowModel>(nullptr);
  flow_model->SetModelRelation(model_relation_ptr);
  flow_model->AddSubModel(submodel_1);
  flow_model->AddSubModel(submodel_2);
  flow_model->AddSubModel(submodel_3);
  ModelRelationFlattener flattener(flow_model);
  ModelRelation flattened_model_relation;
  std::map<std::string, PneModelPtr> models;
  EXPECT_EQ(flattener.Flatten(flattened_model_relation, models), SUCCESS);
  EXPECT_EQ(models.size(), 9);
  EXPECT_EQ(flattened_model_relation.submodel_endpoint_infos.size(), 9);
  std::set<std::string> expected_queue_names = {
      "in-queue-1", "in-queue-2", "in-queue-3", "in-queue-4", "out-queue-1", "inner-queue-1", "inner-queue-2",
      "sub1_PartitionedCall1:0", "sub1_PartitionedCall2:0",
      "sub2_PartitionedCall1:0", "sub2_PartitionedCall2:0",
      "sub3_PartitionedCall1:0", "sub3_PartitionedCall2:0"};
  std::set<std::string> queue_names;
  for (const auto &queue_def : flattened_model_relation.endpoints) {
    queue_names.emplace(queue_def.GetName());
  }
  EXPECT_EQ(queue_names, expected_queue_names);
}

/**
 *     NetOutput
 *         |
 *       PC_3
 *      /   \         -> PC_4 2-inputs, 1-output
 *    PC_1  PC2
 *    |      |
 *  data1  data2
 *
 *
 *       NetOutput
 *          |
 *        PC_4_3
 *       /     \
 *    PC_4_1  Submodel-2
 *     |  |    |  \
 *    D1 D2   D3  D4
 */
TEST_F(HeterogeneousDeployPlannerTest, TestFlattenModelRelation_Mixed) {
  auto submodel_1 = StubModels::BuildFlowModel(StubModels::BuildGraphWithoutNeedForBindingQueues("sub1_"));
  auto submodel_2 = std::make_shared<FlowModel>();  // single model
  auto submodel_3 = StubModels::BuildFlowModel(StubModels::BuildGraphWithoutNeedForBindingQueues("sub3_"));
  ASSERT_TRUE(submodel_1 != nullptr);
  ASSERT_TRUE(submodel_3 != nullptr);
  submodel_1->SetModelName("submodel-1");
  submodel_2->SetModelName("submodel-2");
  submodel_3->SetModelName("submodel-3");
  EXPECT_EQ(submodel_1->GetSubmodels().size(), 3);

  auto model_relation_ptr = std::make_shared<ModelRelation>();
  ASSERT_TRUE(model_relation_ptr != nullptr);
  ModelRelation &model_relation = *model_relation_ptr;
  model_relation.root_model_endpoint_info.input_endpoint_names = {"in-queue-1", "in-queue-2", "in-queue-3", "in-queue-4"};
  model_relation.root_model_endpoint_info.output_endpoint_names = {"out-queue-1"};
  model_relation.submodel_endpoint_infos["submodel-1"].model_name = "submodel-1";
  model_relation.submodel_endpoint_infos["submodel-1"].input_endpoint_names = {"in-queue-1", "in-queue-2"};
  model_relation.submodel_endpoint_infos["submodel-1"].output_endpoint_names = {"inner-queue-1"};
  model_relation.submodel_endpoint_infos["submodel-2"].model_name = "submodel-2";
  model_relation.submodel_endpoint_infos["submodel-2"].input_endpoint_names = {"in-queue-3", "in-queue-4"};
  model_relation.submodel_endpoint_infos["submodel-2"].output_endpoint_names = {"inner-queue-2"};
  model_relation.submodel_endpoint_infos["submodel-3"].model_name = "submodel-3";
  model_relation.submodel_endpoint_infos["submodel-3"].input_endpoint_names = {"inner-queue-1", "inner-queue-2"};
  model_relation.submodel_endpoint_infos["submodel-3"].output_endpoint_names = {"out-queue-1"};
  AddQueueDef(model_relation, "in-queue-1");
  AddQueueDef(model_relation, "in-queue-2");
  AddQueueDef(model_relation, "in-queue-3");
  AddQueueDef(model_relation, "in-queue-4");
  AddQueueDef(model_relation, "out-queue-1");
  AddQueueDef(model_relation, "inner-queue-1");
  AddQueueDef(model_relation, "inner-queue-2");

  auto flow_model = std::make_shared<FlowModel>(nullptr);
  flow_model->SetModelRelation(model_relation_ptr);
  flow_model->AddSubModel(submodel_1);
  flow_model->AddSubModel(submodel_2);
  flow_model->AddSubModel(submodel_3);
  ModelRelationFlattener flattener(flow_model);
  ModelRelation flattened_model_relation;
  std::map<std::string, PneModelPtr> models;
  EXPECT_EQ(flattener.Flatten(flattened_model_relation, models), SUCCESS);
  EXPECT_EQ(models.size(), 7);
  EXPECT_EQ(flattened_model_relation.submodel_endpoint_infos.size(), 7);
  std::set<std::string> expected_queue_names = {
      "in-queue-1", "in-queue-2", "in-queue-3", "in-queue-4", "out-queue-1", "inner-queue-1", "inner-queue-2",
      "sub1_PartitionedCall1:0", "sub1_PartitionedCall2:0",
      "sub3_PartitionedCall1:0", "sub3_PartitionedCall2:0"};
  std::set<std::string> queue_names;
  for (const auto &queue_def : flattened_model_relation.endpoints) {
    queue_names.emplace(queue_def.GetName());
  }
  EXPECT_EQ(queue_names, expected_queue_names);
}

/**
 *     NetOutput
 *         |
 *       PC_3
 *      /   \              root_graph
 *    PC_1  PC2
 *    |      |
 *  data1  data2
 */
TEST_F(HeterogeneousDeployPlannerTest, TestFlattenModelRelation_1_level) {
  auto submodel_1 = std::make_shared<FlowModel>();  // single model
  auto submodel_2 = std::make_shared<FlowModel>();  // single model
  auto submodel_3 = std::make_shared<FlowModel>();  // single model
  submodel_1->SetModelName("submodel-1");
  submodel_2->SetModelName("submodel-2");
  submodel_3->SetModelName("submodel-3");

  auto model_relation_ptr = std::make_shared<ModelRelation>();
  ASSERT_TRUE(model_relation_ptr != nullptr);
  ModelRelation &model_relation = *model_relation_ptr;
  model_relation.root_model_endpoint_info.input_endpoint_names = {"in-queue-1", "in-queue-2"};
  model_relation.root_model_endpoint_info.output_endpoint_names = {"out-queue-1"};
  model_relation.submodel_endpoint_infos["submodel-1"].model_name = "submodel-1";
  model_relation.submodel_endpoint_infos["submodel-1"].input_endpoint_names = {"in-queue-1"};
  model_relation.submodel_endpoint_infos["submodel-1"].output_endpoint_names = {"inner-queue-1"};
  model_relation.submodel_endpoint_infos["submodel-2"].model_name = "submodel-2";
  model_relation.submodel_endpoint_infos["submodel-2"].input_endpoint_names = {"in-queue-2"};
  model_relation.submodel_endpoint_infos["submodel-2"].output_endpoint_names = {"inner-queue-2"};
  model_relation.submodel_endpoint_infos["submodel-3"].model_name = "submodel-3";
  model_relation.submodel_endpoint_infos["submodel-3"].input_endpoint_names = {"inner-queue-1", "inner-queue-2"};
  model_relation.submodel_endpoint_infos["submodel-3"].output_endpoint_names = {"out-queue-1"};
  AddQueueDef(model_relation, "in-queue-1");
  AddQueueDef(model_relation, "in-queue-2");
  AddQueueDef(model_relation, "inner-queue-1");
  AddQueueDef(model_relation, "inner-queue-2");
  AddQueueDef(model_relation, "out-queue-1");

  auto flow_model = std::make_shared<FlowModel>(nullptr);
  flow_model->SetModelRelation(model_relation_ptr);
  flow_model->AddSubModel(submodel_1);
  flow_model->AddSubModel(submodel_2);
  flow_model->AddSubModel(submodel_3);
  ModelRelationFlattener flattener(flow_model);
  ModelRelation flattened_model_relation;
  std::map<std::string, PneModelPtr> models;
  EXPECT_EQ(flattener.Flatten(flattened_model_relation, models), SUCCESS);
  EXPECT_EQ(models.size(), 3);
  EXPECT_EQ(flattened_model_relation.submodel_endpoint_infos.size(), 3);
  std::set<std::string>
      expected_queue_names = {"in-queue-1", "in-queue-2", "out-queue-1", "inner-queue-1", "inner-queue-2"};
  std::set<std::string> queue_names;
  for (const auto &queue_def : flattened_model_relation.endpoints) {
    queue_names.emplace(queue_def.GetName());
  }
  EXPECT_EQ(queue_names, expected_queue_names);
}

/**
 *   NetOutput
 *      |
 *    PC_3     -> submodel_2
 *      |
 *    data1
 *
 *   NetOutput
 *      |
 *    PC_2     -> submodel_1
 *      |
 *    data1
 *
 *   NetOutput
 *      |
 *    PC_1     -> root model
 *     |
 *   Data
 */
TEST_F(HeterogeneousDeployPlannerTest, TestFlattenModelRelation_3_level) {
  auto submodel_3 = std::make_shared<FlowModel>();  // leaf
  submodel_3->SetModelName("submodel_3");

  auto submodel_2 = std::make_shared<FlowModel>();  // mid-2
  {
    submodel_2->SetModelName("submodel_2");
    submodel_2->AddSubModel(submodel_3);
    auto model_relation = std::make_shared<ModelRelation>();
    submodel_2->SetModelRelation(model_relation);
    AddQueueDef(*model_relation, "submodel_2_input_0");
    AddQueueDef(*model_relation, "submodel_2_output_0");
    model_relation->root_model_endpoint_info.input_endpoint_names = {"submodel_2_input_0"};
    model_relation->root_model_endpoint_info.output_endpoint_names = {"submodel_2_output_0"};
    model_relation->submodel_endpoint_infos["submodel_3"].model_name = "submodel_3";
    model_relation->submodel_endpoint_infos["submodel_3"].input_endpoint_names = {"submodel_2_input_0"};
    model_relation->submodel_endpoint_infos["submodel_3"].output_endpoint_names = {"submodel_2_output_0"};
  }

  auto submodel_1 = std::make_shared<FlowModel>();  // mid-1
  {
    submodel_1->SetModelName("submodel_1");
    submodel_1->AddSubModel(submodel_2);
    auto model_relation = std::make_shared<ModelRelation>();
    submodel_1->SetModelRelation(model_relation);
    AddQueueDef(*model_relation, "submodel_1_input_0");
    AddQueueDef(*model_relation, "submodel_1_output_0");
    model_relation->root_model_endpoint_info.input_endpoint_names = {"submodel_1_input_0"};
    model_relation->root_model_endpoint_info.output_endpoint_names = {"submodel_1_output_0"};
    model_relation->submodel_endpoint_infos["submodel_2"].model_name = "submodel_2";
    model_relation->submodel_endpoint_infos["submodel_2"].input_endpoint_names = {"submodel_1_input_0"};
    model_relation->submodel_endpoint_infos["submodel_2"].output_endpoint_names = {"submodel_1_output_0"};
  }

  auto root_model = std::make_shared<FlowModel>();  // root
  {
    root_model->SetModelName("root_model");
    root_model->AddSubModel(submodel_1);
    auto model_relation = std::make_shared<ModelRelation>();
    root_model->SetModelRelation(model_relation);
    AddQueueDef(*model_relation, "input_0");
    AddQueueDef(*model_relation, "output_0");
    model_relation->root_model_endpoint_info.input_endpoint_names = {"input_0"};
    model_relation->root_model_endpoint_info.output_endpoint_names = {"output_0"};
    model_relation->submodel_endpoint_infos["submodel_1"].model_name = "submodel_1";
    model_relation->submodel_endpoint_infos["submodel_1"].input_endpoint_names = {"input_0"};
    model_relation->submodel_endpoint_infos["submodel_1"].output_endpoint_names = {"output_0"};
  }

  ModelRelationFlattener flattener(root_model);
  ModelRelation flattened_model_relation;
  std::map<std::string, PneModelPtr> models;
  EXPECT_EQ(flattener.Flatten(flattened_model_relation, models), SUCCESS);
  EXPECT_EQ(models.size(), 1);
  EXPECT_EQ(flattened_model_relation.submodel_endpoint_infos.size(), 1);
  std::set<std::string>
      expected_queue_names = {"input_0", "output_0"};
  std::set<std::string> queue_names;
  for (const auto &queue_def : flattened_model_relation.endpoints) {
    queue_names.emplace(queue_def.GetName());
  }
  EXPECT_EQ(queue_names, expected_queue_names);

  ModelRelationFlattener shallow_flattener(root_model);
  shallow_flattener.max_depth_ = 1;
  EXPECT_EQ(shallow_flattener.Flatten(flattened_model_relation, models), UNSUPPORTED);
}

TEST_F(HeterogeneousDeployPlannerTest, ParseLogicalDeviceId_single) {
  std::vector<std::string> logical_device_ids;
  EXPECT_EQ(HeterogeneousDeployPlanner::ParseLogicalDeviceIds("[1:1]", logical_device_ids), SUCCESS);
  EXPECT_EQ(logical_device_ids, std::vector<std::string>{"1:1"});
}

TEST_F(HeterogeneousDeployPlannerTest, ParseLogicalDeviceId_multi) {
  {
    std::vector<std::string> logical_device_ids;
    EXPECT_EQ(HeterogeneousDeployPlanner::ParseLogicalDeviceIds("[1:1 , 2:2]", logical_device_ids), SUCCESS);
    EXPECT_EQ(logical_device_ids, (std::vector<std::string>{"1:1", "2:2"}));
  }
  {
    std::vector<std::string> logical_device_ids;
    EXPECT_EQ(HeterogeneousDeployPlanner::ParseLogicalDeviceIds("[1:1,2:2]", logical_device_ids), SUCCESS);
    EXPECT_EQ(logical_device_ids, (std::vector<std::string>{"1:1", "2:2"}));
  }
  {
    std::vector<std::string> logical_device_ids;
    EXPECT_EQ(HeterogeneousDeployPlanner::ParseLogicalDeviceIds("[ 1:1,2:2 ]", logical_device_ids), SUCCESS);
    EXPECT_EQ(logical_device_ids, (std::vector<std::string>{"1:1", "2:2"}));
  }
}

TEST_F(HeterogeneousDeployPlannerTest, ParseLogicalDeviceId_invalid) {
  std::vector<std::string> logical_device_ids;
  EXPECT_EQ(HeterogeneousDeployPlanner::ParseLogicalDeviceIds("[a:b , 2:2]", logical_device_ids), PARAM_INVALID);
  EXPECT_EQ(HeterogeneousDeployPlanner::ParseLogicalDeviceIds("[]", logical_device_ids), PARAM_INVALID);
}

TEST_F(HeterogeneousDeployPlannerTest, ReindexNpuDevices) {
  std::vector<DeployPlan::DeviceInfo> device_list;
  device_list.emplace_back(DeployPlan::DeviceInfo(NPU, 1, 0));
  device_list.emplace_back(DeployPlan::DeviceInfo(NPU, 1, 1));
  device_list.emplace_back(DeployPlan::DeviceInfo(NPU, 2, 0));
  device_list.emplace_back(DeployPlan::DeviceInfo(NPU, 2, 1));
  BuildDeviceInfos(device_list);
  HeterogeneousDeployPlanner planner(nullptr, device_list);
  auto ret = planner.ReindexDevices();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(planner.index_to_devices_.size(), 4);
  std::set<std::string> keys;
  std::set<std::string> expected_keys {"0:0:0:0", "0:0:0:1", "0:0:1:0", "0:0:1:1"};
  for (auto &it : planner.index_to_devices_) {
    keys.emplace(it.first);
  }
  EXPECT_EQ(keys, expected_keys);
}

TEST_F(HeterogeneousDeployPlannerTest, TestBuildPlanForSingleModel_Without_CompileRes) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildSinglePartitionedCallGraph());
  ASSERT_TRUE(flow_model != nullptr);
  std::map<std::string, std::string> options;
  ge::GetThreadLocalContext().SetGraphOption(options);
  std::vector<DeployPlan::DeviceInfo> device_list{DeployPlan::DeviceInfo{0, 0, 0},
                                                  DeployPlan::DeviceInfo{0, 1, 0},
                                                  DeployPlan::DeviceInfo{0, 1, 1}};
  HeterogeneousDeployPlanner planner(flow_model, device_list);
  DeployPlan deploy_plan;
  BuildDeviceInfos(device_list);
  // without compile resource json
  auto ret = planner.BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(deploy_plan.submodels_.size(), device_list.size());
  std::map<std::string, std::string> empty_options;
  ge::GetThreadLocalContext().SetGraphOption(empty_options);
}

TEST_F(HeterogeneousDeployPlannerTest, TestBuildPlanForHeavyLoadUdf) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildSinglePartitionedCallGraph());
  ASSERT_TRUE(flow_model != nullptr);
  auto deploy_resource_ptr = std::make_shared<ModelDeployResource>();
  ASSERT_TRUE(deploy_resource_ptr != nullptr);
  ModelDeployResource &deploy_resource = *deploy_resource_ptr;
  deploy_resource.resource_type = "X86";
  deploy_resource.is_heavy_load = true;
  const auto &udf_model = flow_model->GetSubmodels().begin()->second;
  udf_model->SetModelType(PNE_ID_UDF);
  udf_model->SetDeployResource(deploy_resource_ptr);
  udf_model->SetLogicDeviceId("0:0:1:0");

  DeviceInfo local_device(0, CPU, 0);
  local_device.SetResourceType("X86");

  DeviceInfo remote_device_0(0, NPU, 0);
  remote_device_0.SetResourceType("Ascend");

  DeviceInfo remote_device_1(0, NPU, 1);
  remote_device_1.SetResourceType("Ascend");
  // backup
  std::vector<DeviceInfo> device_info_list = std::move(ResourceManager::GetInstance().device_info_list_);
  std::map<int32_t, std::map<int32_t, std::map<DeviceType, const DeviceInfo *>>> device_info_map = std::move(
    ResourceManager::GetInstance().device_info_map_);

  ResourceManager::GetInstance().device_info_list_.emplace_back(local_device);
  ResourceManager::GetInstance().device_info_list_.emplace_back(remote_device_0);
  ResourceManager::GetInstance().device_info_list_.emplace_back(remote_device_1);
  ResourceManager::GetInstance().device_info_map_[0][0][CPU] = &local_device;
  ResourceManager::GetInstance().device_info_map_[0][0][NPU] = &remote_device_0;
  ResourceManager::GetInstance().device_info_map_[0][1][NPU] = &remote_device_1;

  std::vector<DeployPlan::DeviceInfo> device_list{DeployPlan::DeviceInfo{1, 0, 0},
                                                  DeployPlan::DeviceInfo{0, 0, 0},
                                                  DeployPlan::DeviceInfo{0, 0, 1}};
  HeterogeneousDeployPlanner planner(flow_model, device_list);
  DeployPlan deploy_plan;
  BuildDeviceInfos(device_list, true);
  auto ret = planner.BuildPlan(deploy_plan);
  EXPECT_EQ(ret, SUCCESS);
  const auto &submodels = deploy_plan.GetSubmodels();
  EXPECT_EQ(submodels.size(), 1);
  DeployPlan::DeviceInfo expect_device_info(CPU, 0, 0, 1);
  DeployPlan::DeviceInfo expect_proxy_device_info(NPU, 0, 1, 1);
  const auto &submodel_info = submodels.cbegin()->second;
  EXPECT_EQ(submodel_info.device_info.GetKey(), expect_device_info.GetKey());
  EXPECT_EQ(submodel_info.queue_device_info.GetProxyDeviceId(), 1);
  EXPECT_EQ(submodel_info.queue_device_info.WithProxy(), true);
  EXPECT_EQ(submodel_info.queue_device_info.ProxyDevice().GetKey(), expect_proxy_device_info.GetKey());
  // recover
  ResourceManager::GetInstance().device_info_list_ = std::move(device_info_list);
  ResourceManager::GetInstance().device_info_map_ = std::move(device_info_map);
}

TEST_F(HeterogeneousDeployPlannerTest, TestBuildPlanForHeavyLoadUdf_cannot_assign_host) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildSinglePartitionedCallGraph());
  ASSERT_TRUE(flow_model != nullptr);
  auto deploy_resource_ptr = std::make_shared<ModelDeployResource>();
  ASSERT_TRUE(deploy_resource_ptr != nullptr);
  ModelDeployResource &deploy_resource = *deploy_resource_ptr;
  deploy_resource.resource_type = "X86";
  deploy_resource.is_heavy_load = true;
  const auto &udf_model = flow_model->GetSubmodels().begin()->second;
  udf_model->SetModelType(PNE_ID_UDF);
  udf_model->SetDeployResource(deploy_resource_ptr);
  udf_model->SetLogicDeviceId("0:0:-1:0");

  DeviceInfo local_device(0, CPU, 0);
  local_device.SetResourceType("X86");

  DeviceInfo remote_device_0(0, NPU, 0);
  remote_device_0.SetResourceType("Ascend");

  DeviceInfo remote_device_1(0, NPU, 1);
  remote_device_1.SetResourceType("Ascend");
  // backup
  std::vector<DeviceInfo> device_info_list = std::move(ResourceManager::GetInstance().device_info_list_);
  std::map<int32_t, std::map<int32_t, std::map<DeviceType, const DeviceInfo *>>> device_info_map = std::move(
    ResourceManager::GetInstance().device_info_map_);

  ResourceManager::GetInstance().device_info_list_.emplace_back(local_device);
  ResourceManager::GetInstance().device_info_list_.emplace_back(remote_device_0);
  ResourceManager::GetInstance().device_info_list_.emplace_back(remote_device_1);
  ResourceManager::GetInstance().device_info_map_[0][0][CPU] = &local_device;
  ResourceManager::GetInstance().device_info_map_[0][0][NPU] = &remote_device_0;
  ResourceManager::GetInstance().device_info_map_[0][1][NPU] = &remote_device_1;

  std::vector<DeployPlan::DeviceInfo> device_list{DeployPlan::DeviceInfo{1, 0, 0},
                                                  DeployPlan::DeviceInfo{0, 0, 0},
                                                  DeployPlan::DeviceInfo{0, 0, 1}};
  HeterogeneousDeployPlanner planner(flow_model, device_list);
  DeployPlan deploy_plan;
  BuildDeviceInfos(device_list, true);
  auto ret = planner.BuildPlan(deploy_plan);
  EXPECT_NE(ret, SUCCESS);
  // recover
  ResourceManager::GetInstance().device_info_list_ = std::move(device_info_list);
  ResourceManager::GetInstance().device_info_map_ = std::move(device_info_map);
}

TEST_F(HeterogeneousDeployPlannerTest, TestBuildPlanForHeavyLoadUdf_without_logic_deviceId) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildSinglePartitionedCallGraph());
  ASSERT_TRUE(flow_model != nullptr);
  auto deploy_resource_ptr = std::make_shared<ModelDeployResource>();
  ASSERT_TRUE(deploy_resource_ptr != nullptr);
  ModelDeployResource &deploy_resource = *deploy_resource_ptr;
  deploy_resource.resource_type = "X86";
  deploy_resource.is_heavy_load = true;
  const auto &udf_model = flow_model->GetSubmodels().begin()->second;
  udf_model->SetModelType(PNE_ID_UDF);
  udf_model->SetDeployResource(deploy_resource_ptr);

  DeviceInfo local_device(0, CPU, 0);
  local_device.SetResourceType("X86");

  DeviceInfo remote_device_0(0, NPU, 0);
  remote_device_0.SetResourceType("Ascend");

  DeviceInfo remote_device_1(0, NPU, 1);
  remote_device_1.SetResourceType("Ascend");
  // backup
  std::vector<DeviceInfo> device_info_list = std::move(ResourceManager::GetInstance().device_info_list_);
  std::map<int32_t, std::map<int32_t, std::map<DeviceType, const DeviceInfo *>>> device_info_map = std::move(
    ResourceManager::GetInstance().device_info_map_);

  ResourceManager::GetInstance().device_info_list_.emplace_back(local_device);
  ResourceManager::GetInstance().device_info_list_.emplace_back(remote_device_0);
  ResourceManager::GetInstance().device_info_list_.emplace_back(remote_device_1);
  ResourceManager::GetInstance().device_info_map_[0][0][CPU] = &local_device;
  ResourceManager::GetInstance().device_info_map_[0][0][NPU] = &remote_device_0;
  ResourceManager::GetInstance().device_info_map_[0][1][NPU] = &remote_device_1;

  std::vector<DeployPlan::DeviceInfo> device_list{DeployPlan::DeviceInfo{1, 0, 0},
                                                  DeployPlan::DeviceInfo{0, 0, 0},
                                                  DeployPlan::DeviceInfo{0, 0, 1}};
  HeterogeneousDeployPlanner planner(flow_model, device_list);
  DeployPlan deploy_plan;
  BuildDeviceInfos(device_list, true);
  auto ret = planner.BuildPlan(deploy_plan);
  EXPECT_NE(ret, SUCCESS);
  // recover
  ResourceManager::GetInstance().device_info_list_ = std::move(device_info_list);
  ResourceManager::GetInstance().device_info_map_ = std::move(device_info_map);
}

TEST_F(HeterogeneousDeployPlannerTest, TestBuildDynamicSchedDeployPlanWithProxy) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildGraphWithQueueBindings());
  ASSERT_TRUE(flow_model != nullptr);
  EXPECT_EQ(flow_model->GetSubmodels().size(), 2);
  auto model_relation = flow_model->GetModelRelation();
  ASSERT_TRUE(model_relation != nullptr);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.output_endpoint_names.size(), 1);

  DeployPlan deploy_plan;
  deploy_plan.SetIsDynamicSched(true);
  std::vector<GeRootModelPtr> models;
  BuildDeviceInfos(single_device_with_cpu_list, true);
  auto ret = HeterogeneousDeployPlanner(flow_model, single_device_with_cpu_list).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);

  cout << deploy_plan.GetQueueInfoList().size() << endl;
  cout << deploy_plan.GetQueueBindings().size() << endl;
  cout << deploy_plan.GetInputQueueIndices().size() << endl;
  cout << deploy_plan.GetOutputQueueIndices().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetStatusOutputQueueIndices().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetSchedInputQueueIndices().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetSchedOutputQueueIndices().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetDatagwRequestBindings().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetEntryBindings().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetModelIndexInfo().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetModelInstanceNum().size() << endl;

  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 16);
  ASSERT_EQ(deploy_plan.GetQueueBindings().size(), 7);
  ASSERT_EQ(deploy_plan.GetInputQueueIndices().size(), 2);
  ASSERT_EQ(deploy_plan.GetOutputQueueIndices().size(), 1);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetStatusOutputQueueIndices().size(), 3);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetSchedInputQueueIndices().size(), 1);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetSchedOutputQueueIndices().size(), 1);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetDatagwRequestBindings().size(), 2);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetEntryBindings().size(), 2);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetModelIndexInfo().size(), 1);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetModelInstanceNum().size(), 3);
}

TEST_F(HeterogeneousDeployPlannerTest, TestBuildDynamicSchedDeployPlanWithProxyAndMultipleDevice) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildGraphWithQueueBindings());
  ASSERT_TRUE(flow_model != nullptr);
  EXPECT_EQ(flow_model->GetSubmodels().size(), 2);
  auto model_relation = flow_model->GetModelRelation();
  ASSERT_TRUE(model_relation != nullptr);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.input_endpoint_names.size(), 2);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.output_endpoint_names.size(), 1);

  DeployPlan deploy_plan;
  deploy_plan.SetIsDynamicSched(true);
  std::vector<GeRootModelPtr> models;
  std::vector<DeployPlan::DeviceInfo> node_list = {DeployPlan::DeviceInfo{1, 0, 0},
                                                   DeployPlan::DeviceInfo{0, 0, 0},
                                                   DeployPlan::DeviceInfo{0, 1, 0},
                                                   DeployPlan::DeviceInfo{0, 1, 1}};
  BuildDeviceInfos(node_list, true);
  auto ret = HeterogeneousDeployPlanner(flow_model, node_list).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  
  cout << deploy_plan.GetQueueInfoList().size() << endl;
  cout << deploy_plan.GetQueueBindings().size() << endl;
  cout << deploy_plan.GetInputQueueIndices().size() << endl;
  cout << deploy_plan.GetOutputQueueIndices().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetStatusOutputQueueIndices().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetSchedInputQueueIndices().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetSchedOutputQueueIndices().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetDatagwRequestBindings().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetEntryBindings().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetModelIndexInfo().size() << endl;
  cout << deploy_plan.GetDynamicSchedPlan().GetModelInstanceNum().size() << endl;

  ASSERT_EQ(deploy_plan.GetQueueInfoList().size(), 64);
  ASSERT_EQ(deploy_plan.GetQueueBindings().size(), 33);
  ASSERT_EQ(deploy_plan.GetInputQueueIndices().size(), 2);
  ASSERT_EQ(deploy_plan.GetOutputQueueIndices().size(), 1);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetStatusOutputQueueIndices().size(), 9);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetSchedInputQueueIndices().size(), 3);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetSchedOutputQueueIndices().size(), 3);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetDatagwRequestBindings().size(), 6);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetEntryBindings().size(), 11);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetModelIndexInfo().size(), 3);
  ASSERT_EQ(deploy_plan.GetDynamicSchedPlan().GetModelInstanceNum().size(), 9);
}

TEST_F(HeterogeneousDeployPlannerTest, TestBuildDeployPlanWithHostFlowgw) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildSinglePartitionedCallGraph());
  ASSERT_TRUE(flow_model != nullptr);
  EXPECT_EQ(flow_model->GetSubmodels().size(), 1);
  auto model_relation = flow_model->GetModelRelation();
  ASSERT_TRUE(model_relation != nullptr);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 1);

  DeployPlan deploy_plan;
  BuildDeviceInfos(single_device_with_cpu_list, true, true);
  auto ret = HeterogeneousDeployPlanner(flow_model, single_device_with_cpu_list).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(HeterogeneousDeployPlannerTest, TestBuildDeployPlanWithHostFlowgw2) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildSeriesPartitionedCallGraph());
  ASSERT_TRUE(flow_model != nullptr);
  EXPECT_EQ(flow_model->GetSubmodels().size(), 2);
  auto model_relation = flow_model->GetModelRelation();
  ASSERT_TRUE(model_relation != nullptr);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.input_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.input_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.output_endpoint_names.size(), 1);

  DeviceInfo local_device(0, CPU, 0);
  local_device.SetResourceType("X86");

  DeviceInfo remote_device_0(0, NPU, 0);
  remote_device_0.SetResourceType("Ascend");

  DeviceInfo remote_device_1(0, NPU, 1);
  remote_device_1.SetResourceType("Ascend");
  // backup
  std::vector<DeviceInfo> backup_device_info_list = std::move(ResourceManager::GetInstance().device_info_list_);
  std::map<int32_t, std::map<int32_t, std::map<DeviceType, const DeviceInfo *>>> backup_device_info_map =
      std::move(ResourceManager::GetInstance().device_info_map_);

  ResourceManager::GetInstance().device_info_list_.emplace_back(local_device);
  ResourceManager::GetInstance().device_info_list_.emplace_back(remote_device_0);
  ResourceManager::GetInstance().device_info_list_.emplace_back(remote_device_1);
  ResourceManager::GetInstance().device_info_map_[0][0][CPU] = &local_device;
  ResourceManager::GetInstance().device_info_map_[0][0][NPU] = &remote_device_0;
  ResourceManager::GetInstance().device_info_map_[0][1][NPU] = &remote_device_1;

  auto deploy_resource_ptr = std::make_shared<ModelDeployResource>();
  ASSERT_TRUE(deploy_resource_ptr != nullptr);
  ModelDeployResource &deploy_resource = *deploy_resource_ptr;
  deploy_resource.is_heavy_load = true;
  deploy_resource.resource_type = "X86";

  auto model_iter = flow_model->GetSubmodels().begin();
  auto &udf_model = model_iter->second;
  udf_model->SetModelType(PNE_ID_UDF);
  udf_model->SetDeployResource(deploy_resource_ptr);
  udf_model->SetLogicDeviceId("0:0:0:0");

  ++model_iter;
  auto submodel2 = model_iter->second.get();
  submodel2->SetLogicDeviceId("0:0:0:0,0:0:1:0");

  DeployPlan deploy_plan;
  deploy_plan.SetIsDynamicSched(true);
  BuildDeviceInfos(double_device_with_cpu_list, true, true);
  auto ret = HeterogeneousDeployPlanner(flow_model, double_device_with_cpu_list).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);

  ResourceManager::GetInstance().device_info_list_ = std::move(backup_device_info_list);
  ResourceManager::GetInstance().device_info_map_ = std::move(backup_device_info_map);
}

TEST_F(HeterogeneousDeployPlannerTest, TestBuildDeployPlanWithHostFlowgw3) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildSeriesPartitionedCallGraph());
  ASSERT_TRUE(flow_model != nullptr);
  EXPECT_EQ(flow_model->GetSubmodels().size(), 2);
  auto model_relation = flow_model->GetModelRelation();
  ASSERT_TRUE(model_relation != nullptr);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.input_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.input_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.output_endpoint_names.size(), 1);

  DeviceInfo local_device(0, CPU, 0);
  local_device.SetResourceType("X86");

  DeviceInfo remote_device_0(0, NPU, 0);
  remote_device_0.SetResourceType("Ascend");

  DeviceInfo remote_device_1(0, NPU, 1);
  remote_device_1.SetResourceType("Ascend");
  // backup
  std::vector<DeviceInfo> backup_device_info_list = std::move(ResourceManager::GetInstance().device_info_list_);
  std::map<int32_t, std::map<int32_t, std::map<DeviceType, const DeviceInfo *>>> backup_device_info_map =
      std::move(ResourceManager::GetInstance().device_info_map_);

  ResourceManager::GetInstance().device_info_list_.emplace_back(local_device);
  ResourceManager::GetInstance().device_info_list_.emplace_back(remote_device_0);
  ResourceManager::GetInstance().device_info_list_.emplace_back(remote_device_1);
  ResourceManager::GetInstance().device_info_map_[0][0][CPU] = &local_device;
  ResourceManager::GetInstance().device_info_map_[0][0][NPU] = &remote_device_0;
  ResourceManager::GetInstance().device_info_map_[0][1][NPU] = &remote_device_1;

  auto deploy_resource_ptr = std::make_shared<ModelDeployResource>();
  ASSERT_TRUE(deploy_resource_ptr != nullptr);
  ModelDeployResource &deploy_resource = *deploy_resource_ptr;
  deploy_resource.is_heavy_load = true;
  deploy_resource.resource_type = "X86";

  auto model_iter = flow_model->GetSubmodels().begin();
  auto &udf_model = model_iter->second;
  udf_model->SetModelType(PNE_ID_UDF);
  udf_model->SetDeployResource(deploy_resource_ptr);
  udf_model->SetLogicDeviceId("0:0:0:0,0:0:1:0");

  ++model_iter;
  auto submodel2 = model_iter->second.get();
  submodel2->SetLogicDeviceId("0:0:0:0,0:1:1:0");

  DeployPlan deploy_plan;
  BuildDeviceInfos(four_device_with_cpu_list, true, true);
  auto ret = HeterogeneousDeployPlanner(flow_model, four_device_with_cpu_list).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  ResourceManager::GetInstance().device_info_list_ = std::move(backup_device_info_list);
  ResourceManager::GetInstance().device_info_map_ = std::move(backup_device_info_map);
}

TEST_F(HeterogeneousDeployPlannerTest, TestBuildDeployPlanWithHostFlowgwBothHeavy) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildSeriesPartitionedCallGraph());
  ASSERT_TRUE(flow_model != nullptr);
  EXPECT_EQ(flow_model->GetSubmodels().size(), 2);
  auto model_relation = flow_model->GetModelRelation();
  ASSERT_TRUE(model_relation != nullptr);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.input_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.input_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.output_endpoint_names.size(), 1);

  DeviceInfo local_device(0, CPU, 0);
  local_device.SetResourceType("X86");

  DeviceInfo remote_device_0(0, NPU, 0);
  remote_device_0.SetResourceType("Ascend");

  DeviceInfo remote_device_1(0, NPU, 1);
  remote_device_1.SetResourceType("Ascend");
  // backup
  std::vector<DeviceInfo> backup_device_info_list = std::move(ResourceManager::GetInstance().device_info_list_);
  std::map<int32_t, std::map<int32_t, std::map<DeviceType, const DeviceInfo *>>> backup_device_info_map =
      std::move(ResourceManager::GetInstance().device_info_map_);

  ResourceManager::GetInstance().device_info_list_.emplace_back(local_device);
  ResourceManager::GetInstance().device_info_list_.emplace_back(remote_device_0);
  ResourceManager::GetInstance().device_info_list_.emplace_back(remote_device_1);
  ResourceManager::GetInstance().device_info_map_[0][0][CPU] = &local_device;
  ResourceManager::GetInstance().device_info_map_[0][0][NPU] = &remote_device_0;
  ResourceManager::GetInstance().device_info_map_[0][1][NPU] = &remote_device_1;

  auto deploy_resource_ptr = std::make_shared<ModelDeployResource>();
  ASSERT_TRUE(deploy_resource_ptr != nullptr);
  ModelDeployResource &deploy_resource = *deploy_resource_ptr;
  deploy_resource.is_heavy_load = true;
  deploy_resource.resource_type = "X86";

  auto model_iter = flow_model->GetSubmodels().begin();
  auto &udf_model = model_iter->second;
  udf_model->SetModelType(PNE_ID_UDF);
  udf_model->SetDeployResource(deploy_resource_ptr);
  udf_model->SetLogicDeviceId("0:0:0:0,0:0:1:0");

  ++model_iter;
  auto &udf_model2 = model_iter->second;
  udf_model2->SetModelType(PNE_ID_UDF);
  udf_model2->SetDeployResource(deploy_resource_ptr);
  udf_model2->SetLogicDeviceId("0:0:0:0,0:0:1:0");

  DeployPlan deploy_plan;
  BuildDeviceInfos(four_device_with_cpu_list, true, true);
  auto ret = HeterogeneousDeployPlanner(flow_model, four_device_with_cpu_list).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
  ResourceManager::GetInstance().device_info_list_ = std::move(backup_device_info_list);
  ResourceManager::GetInstance().device_info_map_ = std::move(backup_device_info_map);
}

TEST_F(HeterogeneousDeployPlannerTest, TestBuildDeployPlanWithHostFlowgwReuseQueue) {
  auto flow_model = StubModels::BuildFlowModel(StubModels::BuildParallelPartitionedCallGraph());
  ASSERT_TRUE(flow_model != nullptr);
  EXPECT_EQ(flow_model->GetSubmodels().size(), 2);
  auto model_relation = flow_model->GetModelRelation();
  ASSERT_TRUE(model_relation != nullptr);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.size(), 2);
  ASSERT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.input_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-1")->second.output_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.input_endpoint_names.size(), 1);
  ASSERT_EQ(model_relation->submodel_endpoint_infos.find("subgraph-2")->second.output_endpoint_names.size(), 0);

  auto model_iter = flow_model->GetSubmodels().begin();
  auto &model_ins = model_iter->second;
  model_ins->SetLogicDeviceId("0:0:0:0");

  ++model_iter;
  auto &model_ins2 = model_iter->second;
  model_ins2->SetLogicDeviceId("0:0:0:0");

  DeployPlan deploy_plan;
  BuildDeviceInfos(single_device_with_cpu_list, true, true);
  auto ret = HeterogeneousDeployPlanner(flow_model, single_device_with_cpu_list).BuildPlan(deploy_plan);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(HeterogeneousDeployPlannerTest, TestCreateDynamicSchedTags_DiffNode) {
  DeployPlan deploy_plan;
  HeterogeneousDeployPlanner base_plan;
  base_plan.BuildPlan(deploy_plan);
  DeployPlan::QueueInfo src_queue;
  src_queue.device_info = DeployPlan::DeviceInfo(static_cast<int32_t>(CPU), 0, 1, 1);
  DeployPlan::QueueInfo dst_queue;
  dst_queue.device_info = DeployPlan::DeviceInfo(static_cast<int32_t>(CPU), 1, 1, 1);
  base_plan.deploy_plan_.queues_.push_back(src_queue);
  base_plan.deploy_plan_.queues_.push_back(dst_queue);
  ASSERT_EQ(base_plan.CreateDynamicSchedTags(0, 1, dst_queue), SUCCESS);
}

TEST_F(HeterogeneousDeployPlannerTest, TestCreateDynamicSchedTags_SameDevice) {
  DeployPlan deploy_plan;
  HeterogeneousDeployPlanner base_plan;
  base_plan.BuildPlan(deploy_plan);
  DeployPlan::QueueInfo src_queue;
  src_queue.device_info = DeployPlan::DeviceInfo(static_cast<int32_t>(CPU), 0, 1);
  DeployPlan::QueueInfo dst_queue;
  dst_queue.device_info = DeployPlan::DeviceInfo(static_cast<int32_t>(CPU), 0, 1);
  base_plan.deploy_plan_.queues_.push_back(src_queue);
  base_plan.deploy_plan_.queues_.push_back(dst_queue);
  ASSERT_EQ(base_plan.CreateDynamicSchedTags(0, 1, dst_queue), SUCCESS);
}
}  // namespace ge

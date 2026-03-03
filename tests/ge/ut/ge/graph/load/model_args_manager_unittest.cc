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
#include <memory>
#include "graph/load/model_manager/model_args_manager.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/task_info/task_info.h"

#include "common/share_graph.h"
#include "faker/active_mem_base_faker.h"
#include "faker/aicore_taskdef_faker.h"
#include "faker/dsacore_taskdef_faker.h"
#include "faker/ge_model_builder.h"
#include "faker/label_switch_task_def_faker.h"
#include "faker/rts_task_def_faker.h"
#include "faker/event_record_task_def_faker.h"
#include "fixed_addr_bulk_checker.h"
#include "stub/gert_runtime_stub.h"
#include "task_args_refresh_type_classifier_common_header.h"
#include "task_info_init_checker.h"
#include "task_info_stubs.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace ge {
namespace {
void InsertStubAllocator(DavinciModel *model) {
  model->mem_type_to_allocator_[RT_MEMORY_HBM] = MakeShared<StubMemoryBlockManager>(RT_MEMORY_HBM);
  model->mem_type_to_allocator_[RT_MEMORY_TS] = MakeShared<StubMemoryBlockManager>(RT_MEMORY_TS);
  model->mem_type_to_allocator_[RT_MEMORY_DDR] = MakeShared<StubMemoryBlockManager>(RT_MEMORY_DDR);
  model->mem_type_to_allocator_[RT_MEMORY_DEFAULT] = MakeShared<StubMemoryBlockManager>(RT_MEMORY_DEFAULT);
}

StubMemoryBlockManager *GetStubAllocator(DavinciModel *model, ArgsPlacement placement) {
  auto rt_mem_type = ge::GetRtsMemoryType(placement, 0);
  auto iter = model->mem_type_to_allocator_.find(rt_mem_type);
  if (iter != model->mem_type_to_allocator_.end()) {
    std::shared_ptr<MemoryBlockManager> allocator = iter->second;
    return dynamic_cast<StubMemoryBlockManager *>(allocator.get());
  } else {
    return nullptr;
  }
}

const gert::RuntimeStubImpl::MemoryInfo *FindMemoryInfo(const gert::GertRuntimeStub &runtime_stub,
                                                        StubMemoryBlockManager *allocator,
                                                        const std::vector<ModelArgsManager::ModelArgs> &model_args,
                                                        ArgsPlacement placement) {
  for (const auto &arg : model_args) {
    if (arg.placement == placement) {
      if (allocator != nullptr) {
        auto &addrs_to_allocated_mem = allocator->GetAllocatedRtsMemory();
        auto iter = addrs_to_allocated_mem.find(ValueToPtr(arg.model_args_device_addr));
        if (iter == addrs_to_allocated_mem.end()) {
          std::cerr << "find memory info failed, addr: " << arg.model_args_device_addr << std::endl;
          return nullptr;
        }
        return &iter->second;
      } else {
        auto &addrs_to_allocated_mem = runtime_stub.GetRtsRuntimeStub().GetAllocatedRtsMemory();
        auto iter = addrs_to_allocated_mem.find(ValueToPtr(arg.model_args_device_addr));
        return &iter->second;
      }
    }
  }
  return nullptr;
}
bool ModelArgsHasPlacement(const gert::GertRuntimeStub &runtime_stub, StubMemoryBlockManager *allocator,
                           const std::vector<ModelArgsManager::ModelArgs> &model_args, ArgsPlacement placement,
                           std::string &ret) {
  std::array<rtMemType_t, static_cast<int32_t>(ArgsPlacement::kEnd)> expect_placements = {
      RT_MEMORY_HBM,  // hbm
      RT_MEMORY_TS,   // ts
      RT_MEMORY_HBM,  // sqe
      RT_MEMORY_HOST_SVM
  };
  std::stringstream ss;
  for (const auto &arg : model_args) {
    gert::RuntimeStubImpl::MemoryInfo mem_info;
    if (arg.placement == placement) {
      if (allocator != nullptr) {
        auto &addrs_to_allocated_mem = allocator->GetAllocatedRtsMemory();
        auto iter = addrs_to_allocated_mem.find(ValueToPtr(arg.model_args_device_addr));
        if (iter == addrs_to_allocated_mem.end()) {
          ss << "The device addr " << std::hex << arg.model_args_device_addr << " of placement "
             << GetArgsPlacementStr(placement) << " does not exists in allocator" << std::endl;
          ret = ss.str();
          return false;
        }
        mem_info = iter->second;
      } else {
        auto &addrs_to_allocated_mem = runtime_stub.GetRtsRuntimeStub().GetAllocatedRtsMemory();
        auto iter = addrs_to_allocated_mem.find(ValueToPtr(arg.model_args_device_addr));
        if (iter == addrs_to_allocated_mem.end()) {
          ss << "The device addr " << std::hex << arg.model_args_device_addr << " of placement "
             << GetArgsPlacementStr(placement) << " does not exists" << std::endl;
          ret = ss.str();
          return false;
        }
        mem_info = iter->second;
      }
      if (mem_info.rts_mem_type != expect_placements[static_cast<int32_t>(arg.placement)]) {
        ss << "Invalid rts memory type " << std::hex << mem_info.rts_mem_type << ", expect " << std::hex
           << expect_placements[static_cast<int32_t>(arg.placement)] << ", placement "
           << GetArgsPlacementStr(arg.placement) << std::endl;
        ret = ss.str();
        return false;
      }
      return true;
    }
  }
  ss << "Placement " << GetArgsPlacementStr(placement) << " does not in model args" << std::endl;
  ret = ss.str();
  return false;
}
std::unordered_map<std::string, std::vector<size_t>> ConvertToNoeKey(
    const ComputeGraphPtr &graph, const std::unordered_map<size_t, int64_t> &task_indexes_to_node_id) {
  std::unordered_map<int64_t, Node *> node_ids_to_node;
  for (const auto &node : graph->GetAllNodes()) {
    node_ids_to_node[node->GetOpDesc()->GetId()] = node.get();
  }

  std::unordered_map<std::string, std::vector<size_t>> node_names_to_task_indexes;
  for (const auto &task_index_and_node_id : task_indexes_to_node_id) {
    std::string node_name = "-1";
    if (task_index_and_node_id.second != -1) {
      node_name = node_ids_to_node.at(task_index_and_node_id.second)->GetName();
    }
    node_names_to_task_indexes[node_name].push_back(task_index_and_node_id.first);
  }
  return node_names_to_task_indexes;
}

ge::NodePtr BuildTestNode() {
  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  return graph->AddNode(op_desc);
}

#define GET_TASK_INFO(node_name) \
  reinterpret_cast<StubTaskInfo *>(task_list[node_names_to_task_indexes.at(node_name).at(0)].get())
#define GET_UPDATE_CALL_TIMES(node_name) GET_TASK_INFO(node_name)->GetUpdateDumpInfosCalls().size()
#define CLEAR_ALL_TASK_INFO(tl)                                   \
  do {                                                            \
    for (auto &task_info : tl) {                                  \
      reinterpret_cast<StubTaskInfo *>(task_info.get())->Clear(); \
    }                                                             \
  } while (0)

}  // namespace
class ModelArgsManagerUT : public testing::Test {};
/**
 * 预置条件：
 * 1. feature map refreshable: true
 */
TEST_F(ModelArgsManagerUT, InitV2_TaskInitParametersCorrect_AllRefreshable) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub().StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL);
  auto graph = gert::ShareGraph::BuildTwoAddNodeKnownShapeGraph();
  graph->TopologicalSorting();
  auto model = gert::GeModelBuilder(graph)
                   .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                   .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                   .Build();

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();

  InsertStubAllocator(davinci_model.get());
  auto allocator = GetStubAllocator(davinci_model.get(), ArgsPlacement::kArgsPlacementHbm);
  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);

  ASSERT_EQ(mam.GetFixedAddrBulk().device_addr, nullptr);

  ASSERT_EQ(task_list.size(), 2UL);

  auto checker = TASK_INFO_CHECKER(task_list, 0)
                     .ArgsMemory(FindMemoryInfo(runtime_stub, allocator, mam.GetModelArgs(),
                                                ArgsPlacement::kArgsPlacementHbm));
  std::string ret;
  ASSERT_TRUE(checker.GeneralCheck(ret)) << ret;
  ASSERT_TRUE(checker.CheckInputAddrs(
      ret, {{MemoryAppType::kMemoryTypeModelIo, true}, {MemoryAppType::kMemoryTypeModelIo, true}}))
      << ret;
  ASSERT_TRUE(checker.CheckOutputAddrs(ret, {{MemoryAppType::kMemoryTypeFeatureMap, true}})) << ret;
  ASSERT_TRUE(checker.CheckWsAddrs(ret, {})) << ret;

  checker = TASK_INFO_CHECKER(task_list, 1)
                .ArgsMemory(FindMemoryInfo(runtime_stub, allocator, mam.GetModelArgs(),
                                           ArgsPlacement::kArgsPlacementHbm));
  ASSERT_TRUE(checker.GeneralCheck(ret)) << ret;
  ASSERT_TRUE(checker.CheckInputAddrs(
      ret, {{MemoryAppType::kMemoryTypeModelIo, true}, {MemoryAppType::kMemoryTypeFeatureMap, true}}))
      << ret;
  ASSERT_TRUE(checker.CheckOutputAddrs(ret, {{MemoryAppType::kMemoryTypeModelIo, true}})) << ret;
  ASSERT_TRUE(checker.CheckWsAddrs(ret, {{MemoryAppType::kMemoryTypeFeatureMap, true}})) << ret;
}
/**
 * 预置条件：
 * 1. feature map refreshable: false
 */
TEST_F(ModelArgsManagerUT, InitV2_TaskInitParametersCorrect_AllRefreshableAndFmNotRfreshable) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub().StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL);
  auto graph = gert::ShareGraph::BuildTwoAddNodeKnownShapeGraph();
  graph->TopologicalSorting();
  auto model = gert::GeModelBuilder(graph)
                   .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                   .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                   .Build();

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(false)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();
  InsertStubAllocator(davinci_model.get());
  auto allocator = GetStubAllocator(davinci_model.get(), ArgsPlacement::kArgsPlacementHbm);
  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);

  ASSERT_EQ(mam.GetFixedAddrBulk().device_addr, nullptr);

  ASSERT_EQ(task_list.size(), 2UL);

  auto checker = TASK_INFO_CHECKER(task_list, 0)
                     .ArgsMemory(FindMemoryInfo(runtime_stub, allocator, mam.GetModelArgs(),
                                                ArgsPlacement::kArgsPlacementHbm));
  std::string ret;
  ASSERT_TRUE(checker.GeneralCheck(ret)) << ret;
  ASSERT_TRUE(checker.CheckInputAddrs(
      ret, {{MemoryAppType::kMemoryTypeModelIo, true}, {MemoryAppType::kMemoryTypeModelIo, true}}))
      << ret;
  ASSERT_TRUE(checker.CheckOutputAddrs(ret, {{MemoryAppType::kMemoryTypeFeatureMap, false}})) << ret;
  ASSERT_TRUE(checker.CheckWsAddrs(ret, {})) << ret;

  checker = TASK_INFO_CHECKER(task_list, 1)
                .ArgsMemory(FindMemoryInfo(runtime_stub, allocator, mam.GetModelArgs(),
                                           ArgsPlacement::kArgsPlacementHbm));
  ASSERT_TRUE(checker.GeneralCheck(ret)) << ret;
  ASSERT_TRUE(checker.CheckInputAddrs(
      ret, {{MemoryAppType::kMemoryTypeModelIo, true}, {MemoryAppType::kMemoryTypeFeatureMap, false}}))
      << ret;
  ASSERT_TRUE(checker.CheckOutputAddrs(ret, {{MemoryAppType::kMemoryTypeModelIo, true}})) << ret;
  ASSERT_TRUE(checker.CheckWsAddrs(ret, {{MemoryAppType::kMemoryTypeFeatureMap, false}})) << ret;
}
/**
 * 预置条件：
 * 1. feature map refreshable: false
 */
TEST_F(ModelArgsManagerUT, InitV2_TaskInitParametersCorrect_HbmTsSqeArgs) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelDebug();
  runtime_stub.GetTaskInfoFactoryStub()
      .StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL)
      .StubTaskInfo<DsaStubTaskInfo>(ModelTaskType::MODEL_TASK_DSA)
      .StubTaskInfo<RtsStubTaskInfo>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC);
  auto graph = gert::ShareGraph::BuildAiCoreRtsDsaNodeKnownShapeGraph();
  graph->TopologicalSorting();
  auto model_detail = gert::GeModelBuilder(graph)
                          .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                          .AddTaskDef("dsa1", gert::DsaCoreTaskDefFaker())
                          .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                          .AddTaskDef("id1", gert::RtsTaskDefFaker())
                          .BuildDetail();
  auto model = model_detail.model;
  auto node_names_to_task_indexes = ConvertToNoeKey(graph, model_detail.task_indexes_to_node_id);

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(false)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();
  InsertStubAllocator(davinci_model.get());
  auto allocator = GetStubAllocator(davinci_model.get(), ArgsPlacement::kArgsPlacementHbm);
  auto ts_allocator = GetStubAllocator(davinci_model.get(), ArgsPlacement::kArgsPlacementTs);
  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);

  auto &model_args = mam.GetModelArgs();
  ASSERT_EQ(model_args.size(), 2UL);
  std::string ret;
  EXPECT_TRUE(ModelArgsHasPlacement(runtime_stub, allocator, model_args, ArgsPlacement::kArgsPlacementHbm, ret)) << ret;
  EXPECT_TRUE(ModelArgsHasPlacement(runtime_stub, ts_allocator,
                                    model_args, ArgsPlacement::kArgsPlacementTs, ret)) << ret;

  ASSERT_EQ(mam.GetFixedAddrBulk().device_addr, nullptr);

  ASSERT_EQ(task_list.size(), 4UL);
  auto checker = TASK_INFO_CHECKER(task_list, node_names_to_task_indexes.at("add1").at(0))
                     .ArgsMemory(FindMemoryInfo(runtime_stub, allocator, mam.GetModelArgs(),
                                                ArgsPlacement::kArgsPlacementHbm));
  ASSERT_TRUE(checker.GeneralCheck(ret)) << ret;
  ASSERT_TRUE(checker.CheckInputAddrs(
      ret, {{MemoryAppType::kMemoryTypeModelIo, true}, {MemoryAppType::kMemoryTypeModelIo, true}}))
      << ret;
  ASSERT_TRUE(checker.CheckOutputAddrs(ret, {{MemoryAppType::kMemoryTypeFeatureMap, false}})) << ret;
  ASSERT_TRUE(checker.CheckWsAddrs(ret, {})) << ret;

  checker = TASK_INFO_CHECKER(task_list, node_names_to_task_indexes.at("add2").at(0))
                .ArgsMemory(FindMemoryInfo(runtime_stub, allocator,
                                           mam.GetModelArgs(), ArgsPlacement::kArgsPlacementHbm));
  ASSERT_TRUE(checker.GeneralCheck(ret)) << ret;
  ASSERT_TRUE(checker.CheckInputAddrs(
      ret, {{MemoryAppType::kMemoryTypeFeatureMap, false}, {MemoryAppType::kMemoryTypeFeatureMap, false}}))
      << ret;
  ASSERT_TRUE(checker.CheckOutputAddrs(ret, {{MemoryAppType::kMemoryTypeFeatureMap, false}})) << ret;
  ASSERT_TRUE(checker.CheckWsAddrs(ret, {{MemoryAppType::kMemoryTypeFeatureMap, false}})) << ret;

  checker =
      TASK_INFO_CHECKER(task_list, node_names_to_task_indexes.at("dsa1").at(0))
          .ArgsMemory(FindMemoryInfo(runtime_stub, allocator, mam.GetModelArgs(), ArgsPlacement::kArgsPlacementHbm))
          // 虽然返回的有一块sqe内存，不过sqe内存在申请的时候是合并到hbm中一起申请的，因此从rts视角，看到的仍然是hbm内存
          .ArgsMemory(FindMemoryInfo(runtime_stub, allocator, mam.GetModelArgs(), ArgsPlacement::kArgsPlacementHbm));
  ASSERT_TRUE(checker.GeneralCheck(ret)) << ret;
  ASSERT_TRUE(checker.CheckInputAddrs(ret, {})) << ret;
  ASSERT_TRUE(checker.CheckOutputAddrs(ret, {{MemoryAppType::kMemoryTypeFeatureMap, false}})) << ret;
  ASSERT_TRUE(checker.CheckWsAddrs(ret, {})) << ret;


  checker = TASK_INFO_CHECKER(task_list, node_names_to_task_indexes.at("id1").at(0))
                .ArgsMemory(FindMemoryInfo(runtime_stub, ts_allocator, mam.GetModelArgs(),
                                           ArgsPlacement::kArgsPlacementTs));
  ASSERT_TRUE(checker.GeneralCheck(ret)) << ret;
  ASSERT_TRUE(checker.CheckInputAddrs(ret, {{MemoryAppType::kMemoryTypeFeatureMap, false}})) << ret;
  ASSERT_TRUE(checker.CheckOutputAddrs(ret, {{MemoryAppType::kMemoryTypeModelIo, true}})) << ret;
  ASSERT_TRUE(checker.CheckWsAddrs(ret, {})) << ret;
}

/**
 * 预置条件：
 * 1. feature map refreshable: true
 */
TEST_F(ModelArgsManagerUT, InitV2_FixedAddrCorrect_NotSupportRefreshConnectToRefreshableFm) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub()
      .StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL)
      .StubTaskInfo<DsaStubTaskInfo>(ModelTaskType::MODEL_TASK_DSA)
      .StubTaskInfo<RtsStubTaskInfo>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC);
  auto graph = gert::ShareGraph::BuildAiCoreRtsDsaToIdentityKnownShapeGraph();
  graph->TopologicalSorting();
  auto model_detail = gert::GeModelBuilder(graph)
                          .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                          .AddTaskDef("dsa1", gert::DsaCoreTaskDefFaker())
                          .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                          .AddTaskDef("id1", gert::RtsTaskDefFaker())
                          .BuildDetail();
  auto model = model_detail.model;
  auto node_names_to_task_indexes = ConvertToNoeKey(graph, model_detail.task_indexes_to_node_id);

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();
  InsertStubAllocator(davinci_model.get());
  auto allocator = GetStubAllocator(davinci_model.get(), ArgsPlacement::kArgsPlacementHbm);
  auto ts_allocator = GetStubAllocator(davinci_model.get(), ArgsPlacement::kArgsPlacementTs);
  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);

  auto &model_args = mam.GetModelArgs();
  ASSERT_EQ(model_args.size(), 2UL);
  std::string ret;
  EXPECT_TRUE(ModelArgsHasPlacement(runtime_stub, allocator, model_args, ArgsPlacement::kArgsPlacementHbm, ret)) << ret;
  EXPECT_TRUE(ModelArgsHasPlacement(runtime_stub, ts_allocator,
                                    model_args, ArgsPlacement::kArgsPlacementTs, ret)) << ret;

  auto &fixed_addr = mam.GetFixedAddrBulk();
  ASSERT_TRUE(FixedAddrBulkChecker(runtime_stub, ts_allocator, fixed_addr, node_names_to_task_indexes)
                  .FixedMemory({{"dsa1", TaskArgsRefreshTypeClassifier::kOutput, 0},
                                {"id1", TaskArgsRefreshTypeClassifier::kInput, 0}})
                  .Check(ret))
      << ret;

  ASSERT_EQ(task_list.size(), 4UL);
  auto checker = TASK_INFO_CHECKER(task_list, node_names_to_task_indexes.at("add1").at(0))
                     .ArgsMemory(FindMemoryInfo(runtime_stub, allocator,
                                                mam.GetModelArgs(), ArgsPlacement::kArgsPlacementHbm))
                     .FixedAddrBulk(fixed_addr);
  ASSERT_TRUE(checker.GeneralCheck(ret)) << ret;
  ASSERT_TRUE(checker.CheckInputAddrs(
      ret, {{MemoryAppType::kMemoryTypeModelIo, true}, {MemoryAppType::kMemoryTypeModelIo, true}}))
      << ret;
  ASSERT_TRUE(checker.CheckOutputAddrs(ret, {{MemoryAppType::kMemoryTypeFeatureMap, true}})) << ret;
  ASSERT_TRUE(checker.CheckWsAddrs(ret, {})) << ret;

  checker = TASK_INFO_CHECKER(task_list, node_names_to_task_indexes.at("add2").at(0))
                .ArgsMemory(FindMemoryInfo(runtime_stub, allocator,
                                           mam.GetModelArgs(), ArgsPlacement::kArgsPlacementHbm))
                .FixedAddrBulk(fixed_addr);
  ASSERT_TRUE(checker.GeneralCheck(ret)) << ret;
  ASSERT_TRUE(checker.CheckInputAddrs(
      ret, {{MemoryAppType::kMemoryTypeFeatureMap, true}, {MemoryAppType::kMemoryTypeFeatureMap, true}}))
      << ret;
  ASSERT_TRUE(checker.CheckOutputAddrs(ret, {{MemoryAppType::kMemoryTypeModelIo, true}})) << ret;
  ASSERT_TRUE(checker.CheckWsAddrs(ret, {})) << ret;

  checker =
      TASK_INFO_CHECKER(task_list, node_names_to_task_indexes.at("dsa1").at(0))
          .ArgsMemory(FindMemoryInfo(runtime_stub, allocator,
                                     mam.GetModelArgs(), ArgsPlacement::kArgsPlacementHbm))
          // 虽然返回的有一块sqe内存，不过sqe内存在申请的时候是合并到hbm中一起申请的，因此从rts视角，看到的仍然是hbm内存
          .ArgsMemory(FindMemoryInfo(runtime_stub, allocator,
                                     mam.GetModelArgs(), ArgsPlacement::kArgsPlacementHbm))
          .FixedAddrBulk(fixed_addr);
  ASSERT_TRUE(checker.GeneralCheck(ret)) << ret;
  ASSERT_TRUE(checker.CheckInputAddrs(ret, {})) << ret;
  ASSERT_TRUE(checker.CheckOutputAddrs(ret, {{MemoryAppType::kMemoryTypeFix, false}})) << ret;
  ASSERT_TRUE(checker.CheckWsAddrs(ret, {})) << ret;
  checker = TASK_INFO_CHECKER(task_list, node_names_to_task_indexes.at("id1").at(0))
                .ArgsMemory(FindMemoryInfo(runtime_stub, ts_allocator,
                                           mam.GetModelArgs(), ArgsPlacement::kArgsPlacementTs))
                .FixedAddrBulk(fixed_addr);
  ASSERT_TRUE(checker.GeneralCheck(ret)) << ret;
  ASSERT_TRUE(checker.CheckInputAddrs(ret, {{MemoryAppType::kMemoryTypeFix, false}})) << ret;
  ASSERT_TRUE(checker.CheckOutputAddrs(ret, {{MemoryAppType::kMemoryTypeFeatureMap, true}})) << ret;
  ASSERT_TRUE(checker.CheckWsAddrs(ret, {})) << ret;
}
TEST_F(ModelArgsManagerUT, InitV2_FixedAddrReplacedCorrectly_LabelSwitchTaskHasNoArgsAndHasFixedAddr) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub()
      .StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL)
      .StubTaskInfo<RtsLabelSwitchStubTaskInfo>(ModelTaskType::MODEL_TASK_STREAM_LABEL_SWITCH_BY_INDEX)
      .StubTaskInfo<RtsStubTaskInfo>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC);
  auto graph = gert::ShareGraph::BuildGraphHasLabelSwitch();
  graph->TopologicalSorting();
  auto model_detail = gert::GeModelBuilder(graph)
                          .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                          .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                          .AddTaskDef("id1", gert::RtsTaskDefFaker())
                          .AddTaskDef("ls1", gert::LabelSwitchTaskDefFaker())
                          .BuildDetail();
  auto model = model_detail.model;
  auto node_names_to_task_indexes = ConvertToNoeKey(graph, model_detail.task_indexes_to_node_id);

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();
  InsertStubAllocator(davinci_model.get());
  auto allocator = GetStubAllocator(davinci_model.get(), ArgsPlacement::kArgsPlacementHbm);
  auto ts_allocator = GetStubAllocator(davinci_model.get(), ArgsPlacement::kArgsPlacementTs);
  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);

  auto &model_args = mam.GetModelArgs();
  ASSERT_EQ(model_args.size(), 2UL);
  std::string ret;
  EXPECT_TRUE(ModelArgsHasPlacement(runtime_stub, allocator,
                                    model_args, ArgsPlacement::kArgsPlacementHbm, ret)) << ret;
  EXPECT_TRUE(ModelArgsHasPlacement(runtime_stub, ts_allocator,
                                    model_args, ArgsPlacement::kArgsPlacementTs, ret)) << ret;

  auto &fixed_addr = mam.GetFixedAddrBulk();
  ASSERT_TRUE(FixedAddrBulkChecker(runtime_stub, ts_allocator, fixed_addr, node_names_to_task_indexes)
                  .FixedMemory({{"id1", TaskArgsRefreshTypeClassifier::kOutput, 0},
                                {"ls1", TaskArgsRefreshTypeClassifier::kInput, 0}})
                  .FixedMemory({{"ls1", TaskArgsRefreshTypeClassifier::kWorkspace, 0}})
                  .Check(ret))
      << ret;

  ASSERT_EQ(task_list.size(), 4UL);
  auto checker = TASK_INFO_CHECKER(task_list, node_names_to_task_indexes.at("add1").at(0))
                     .ArgsMemory(FindMemoryInfo(runtime_stub, allocator,
                                                mam.GetModelArgs(), ArgsPlacement::kArgsPlacementHbm))
                     .FixedAddrBulk(fixed_addr);
  ASSERT_TRUE(checker.GeneralCheck(ret)) << ret;
  ASSERT_TRUE(checker.CheckInputAddrs(
      ret, {{MemoryAppType::kMemoryTypeModelIo, true}, {MemoryAppType::kMemoryTypeModelIo, true}}))
      << ret;
  ASSERT_TRUE(checker.CheckOutputAddrs(ret, {{MemoryAppType::kMemoryTypeFeatureMap, true}})) << ret;
  ASSERT_TRUE(checker.CheckWsAddrs(ret, {})) << ret;

  checker = TASK_INFO_CHECKER(task_list, node_names_to_task_indexes.at("add2").at(0))
                .ArgsMemory(FindMemoryInfo(runtime_stub, allocator,
                                           mam.GetModelArgs(), ArgsPlacement::kArgsPlacementHbm))
                .FixedAddrBulk(fixed_addr);
  ASSERT_TRUE(checker.GeneralCheck(ret)) << ret;
  ASSERT_TRUE(checker.CheckInputAddrs(
      ret, {{MemoryAppType::kMemoryTypeModelIo, true}, {MemoryAppType::kMemoryTypeFeatureMap, true}}))
      << ret;
  ASSERT_TRUE(checker.CheckOutputAddrs(ret, {{MemoryAppType::kMemoryTypeModelIo, true}})) << ret;
  ASSERT_TRUE(checker.CheckWsAddrs(ret, {})) << ret;

  checker = TASK_INFO_CHECKER(task_list, node_names_to_task_indexes.at("ls1").at(0)).FixedAddrBulk(fixed_addr);
  ASSERT_TRUE(checker.GeneralCheck(ret)) << ret;
  ASSERT_TRUE(checker.CheckInputAddrs(ret, {{MemoryAppType::kMemoryTypeFix, false}})) << ret;
  ASSERT_TRUE(checker.CheckOutputAddrs(ret, {})) << ret;
  ASSERT_TRUE(checker.CheckWsAddrs(ret, {{MemoryAppType::kMemoryTypeFix, false}})) << ret;

  checker = TASK_INFO_CHECKER(task_list, node_names_to_task_indexes.at("id1").at(0))
                .ArgsMemory(FindMemoryInfo(runtime_stub, ts_allocator,
                                           mam.GetModelArgs(), ArgsPlacement::kArgsPlacementTs))
                .FixedAddrBulk(fixed_addr);
  ASSERT_TRUE(checker.GeneralCheck(ret)) << ret;
  ASSERT_TRUE(checker.CheckInputAddrs(ret, {{MemoryAppType::kMemoryTypeFeatureMap, true}})) << ret;
  ASSERT_TRUE(checker.CheckOutputAddrs(ret, {{MemoryAppType::kMemoryTypeFix, false}})) << ret;
  ASSERT_TRUE(checker.CheckWsAddrs(ret, {})) << ret;
}
TEST_F(ModelArgsManagerUT, InitV2_AllOthersCorrect_EventTaskHasNoCorrospondingNode) {
  gert::GertRuntimeStub runtime_stub;
  dlog_setlevel(GE_MODULE_NAME, 0, 0);
  runtime_stub.GetTaskInfoFactoryStub()
      .StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL)
      .StubTaskInfo<DsaStubTaskInfo>(ModelTaskType::MODEL_TASK_DSA)
      .StubTaskInfo<EventStubTaskInfo>(ModelTaskType::MODEL_TASK_EVENT_RECORD)
      .StubTaskInfo<RtsStubTaskInfo>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC);
  auto graph = gert::ShareGraph::BuildAiCoreRtsDsaNodeKnownShapeGraph();
  graph->TopologicalSorting();
  auto model_detail = gert::GeModelBuilder(graph)
                          .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                          .AddTaskDef("dsa1", gert::DsaCoreTaskDefFaker())
                          .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                          .AddTaskDef("id1", gert::RtsTaskDefFaker())
                          .AppendTaskDef(gert::EventRecordTaskDefFaker())
                          .BuildDetail();
  auto model = model_detail.model;
  auto node_names_to_task_indexes = ConvertToNoeKey(graph, model_detail.task_indexes_to_node_id);

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(false)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();
  InsertStubAllocator(davinci_model.get());
  auto allocator = GetStubAllocator(davinci_model.get(), ArgsPlacement::kArgsPlacementHbm);
  auto ts_allocator = GetStubAllocator(davinci_model.get(), ArgsPlacement::kArgsPlacementTs);
  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);

  auto &model_args = mam.GetModelArgs();
  ASSERT_EQ(model_args.size(), 2UL);
  std::string ret;
  EXPECT_TRUE(ModelArgsHasPlacement(runtime_stub, allocator, model_args, ArgsPlacement::kArgsPlacementHbm, ret)) << ret;
  EXPECT_TRUE(ModelArgsHasPlacement(runtime_stub, ts_allocator,
                                    model_args, ArgsPlacement::kArgsPlacementTs, ret)) << ret;

  ASSERT_EQ(mam.GetFixedAddrBulk().device_addr, nullptr);

  ASSERT_EQ(task_list.size(), 5UL);
  auto checker = TASK_INFO_CHECKER(task_list, node_names_to_task_indexes.at("add1").at(0))
                     .ArgsMemory(FindMemoryInfo(runtime_stub, allocator,
                                                mam.GetModelArgs(), ArgsPlacement::kArgsPlacementHbm));
  ASSERT_TRUE(checker.GeneralCheck(ret)) << ret;
  ASSERT_TRUE(checker.CheckInputAddrs(
      ret, {{MemoryAppType::kMemoryTypeModelIo, true}, {MemoryAppType::kMemoryTypeModelIo, true}}))
      << ret;
  ASSERT_TRUE(checker.CheckOutputAddrs(ret, {{MemoryAppType::kMemoryTypeFeatureMap, false}})) << ret;
  ASSERT_TRUE(checker.CheckWsAddrs(ret, {})) << ret;

  checker = TASK_INFO_CHECKER(task_list, node_names_to_task_indexes.at("add2").at(0))
                .ArgsMemory(FindMemoryInfo(runtime_stub, allocator,
                                           mam.GetModelArgs(), ArgsPlacement::kArgsPlacementHbm));
  ASSERT_TRUE(checker.GeneralCheck(ret)) << ret;
  ASSERT_TRUE(checker.CheckInputAddrs(
      ret, {{MemoryAppType::kMemoryTypeFeatureMap, false}, {MemoryAppType::kMemoryTypeFeatureMap, false}}))
      << ret;
  ASSERT_TRUE(checker.CheckOutputAddrs(ret, {{MemoryAppType::kMemoryTypeFeatureMap, false}})) << ret;
  ASSERT_TRUE(checker.CheckWsAddrs(ret, {{MemoryAppType::kMemoryTypeFeatureMap, false}})) << ret;

  checker =
      TASK_INFO_CHECKER(task_list, node_names_to_task_indexes.at("dsa1").at(0))
          .ArgsMemory(FindMemoryInfo(runtime_stub, allocator,
                                     mam.GetModelArgs(), ArgsPlacement::kArgsPlacementHbm))
          // 虽然返回的有一块sqe内存，不过sqe内存在申请的时候是合并到hbm中一起申请的，因此从rts视角，看到的仍然是hbm内存
          .ArgsMemory(FindMemoryInfo(runtime_stub, allocator,
                                     mam.GetModelArgs(), ArgsPlacement::kArgsPlacementHbm));
  ASSERT_TRUE(checker.GeneralCheck(ret)) << ret;
  ASSERT_TRUE(checker.CheckInputAddrs(ret, {})) << ret;
  ASSERT_TRUE(checker.CheckOutputAddrs(ret, {{MemoryAppType::kMemoryTypeFeatureMap, false}})) << ret;
  ASSERT_TRUE(checker.CheckWsAddrs(ret, {})) << ret;

  checker = TASK_INFO_CHECKER(task_list, node_names_to_task_indexes.at("id1").at(0))
                .ArgsMemory(FindMemoryInfo(runtime_stub, ts_allocator,
                                           mam.GetModelArgs(), ArgsPlacement::kArgsPlacementTs));
  ASSERT_TRUE(checker.GeneralCheck(ret)) << ret;
  ASSERT_TRUE(checker.CheckInputAddrs(ret, {{MemoryAppType::kMemoryTypeFeatureMap, false}})) << ret;
  ASSERT_TRUE(checker.CheckOutputAddrs(ret, {{MemoryAppType::kMemoryTypeModelIo, true}})) << ret;
  ASSERT_TRUE(checker.CheckWsAddrs(ret, {})) << ret;
}
TEST_F(ModelArgsManagerUT, InitV2_Suceess_NoTaskInModelDef) {
  domi::ModelTaskDef model_task_def;
  ModelArgsManager mam(nullptr);
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(model_task_def, &task_list), SUCCESS);
}
/**
 * 预置条件：
 * 1. feature map refreshable: false
 */
TEST_F(ModelArgsManagerUT, UpdateForExecute_AllTaskInfoCalled_FirstTime) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub()
      .StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL)
      .StubTaskInfo<DsaStubTaskInfo>(ModelTaskType::MODEL_TASK_DSA)
      .StubTaskInfo<RtsStubTaskInfo>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC);
  auto graph = gert::ShareGraph::BuildAiCoreRtsDsaToIdentityKnownShapeGraph();
  graph->TopologicalSorting();
  auto model_detail = gert::GeModelBuilder(graph)
                          .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                          .AddTaskDef("dsa1", gert::DsaCoreTaskDefFaker())
                          .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                          .AddTaskDef("id1", gert::RtsTaskDefFaker())
                          .BuildDetail();
  auto model = model_detail.model;
  auto node_names_to_task_indexes = ConvertToNoeKey(graph, model_detail.task_indexes_to_node_id);

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(false)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();
  std::set<std::string> temp;
  davinci_model.get()->data_dumper_.dump_properties_.model_dump_properties_map_.emplace(DUMP_ALL_MODEL, temp);

  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  mam.SetAllocationHitCount(1U, 1U);
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);

  rtStream_t stream = (rtStream_t)1;
  std::vector<uint64_t> active_mem_base_addr = ActiveMemBaseFaker(2, 1).FmBaseIndex(0).ModelIoBaseIndex(1).Build();
  mam.AllocKernelLaunchArgsHostMem(active_mem_base_addr.size());
  uint64_t* active_mem_base_addr_temp = mam.GetActivateMemBaseAddrs();
  for (size_t i = 0; i < active_mem_base_addr.size(); i++) {
    active_mem_base_addr_temp[i] = active_mem_base_addr[i];
  }

  uint32_t up = 4;
  ASSERT_EQ(mam.UpdateForExecute(up, stream), SUCCESS);

  for (const auto &task_info : task_list) {
    ASSERT_EQ(reinterpret_cast<StubTaskInfo *>(task_info.get())->GetUpdateDumpInfosCalls().size(), 1UL);
  }
  // todo 所有H2D的args拷贝地址、长度正确
}

TEST_F(ModelArgsManagerUT, UpdateForExecute_AllTaskInfoCalled_FirstTime_With_UpdateModelParamOp) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub()
      .StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL)
      .StubTaskInfo<DsaStubTaskInfo>(ModelTaskType::MODEL_TASK_DSA)
      .StubTaskInfo<RtsStubTaskInfo>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC);
  auto graph = gert::ShareGraph::BuildAiCoreRtsDsaToIdentityKnownShapeGraph();
  graph->TopologicalSorting();
  auto model_detail = gert::GeModelBuilder(graph)
                          .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                          .AddTaskDef("dsa1", gert::DsaCoreTaskDefFaker())
                          .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                          .AddTaskDef("id1", gert::RtsTaskDefFaker())
                          .BuildDetail();
  auto model = model_detail.model;
  auto node_names_to_task_indexes = ConvertToNoeKey(graph, model_detail.task_indexes_to_node_id);

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(false)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();
  std::set<std::string> temp;
  davinci_model.get()->data_dumper_.dump_properties_.model_dump_properties_map_.emplace(DUMP_ALL_MODEL, temp);

  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  mam.SetAllocationHitCount(1U, 1U);
  std::string stub_func_str = "func_test";
  mam.SetFuncHandle(static_cast<void*>(const_cast<char*>(stub_func_str.c_str())));
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);

  rtStream_t stream = (rtStream_t)1;
  std::vector<uint64_t> active_mem_base_addr = ActiveMemBaseFaker(2, 1).FmBaseIndex(0).ModelIoBaseIndex(1).Build();
  mam.AllocKernelLaunchArgsHostMem(active_mem_base_addr.size());
  uint64_t* active_mem_base_addr_temp = mam.GetActivateMemBaseAddrs();
  for (size_t i = 0; i < active_mem_base_addr.size(); i++) {
    active_mem_base_addr_temp[i] = active_mem_base_addr[i];
  }

  uint32_t up = 4;
  ASSERT_EQ(mam.UpdateForExecute(up, stream), SUCCESS);
  // todo 所有H2D的args拷贝地址、长度正确
}

TEST_F(ModelArgsManagerUT, UpdateForExecute_AllTaskInfoCalled_FirstTime_Version_1) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub()
      .StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL)
      .StubTaskInfo<DsaStubTaskInfo>(ModelTaskType::MODEL_TASK_DSA)
      .StubTaskInfo<RtsStubTaskInfo>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC);
  auto graph = gert::ShareGraph::BuildAiCoreRtsDsaToIdentityKnownShapeGraph();
  graph->TopologicalSorting();
  auto model_detail = gert::GeModelBuilder(graph)
                          .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                          .AddTaskDef("dsa1", gert::DsaCoreTaskDefFaker())
                          .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                          .AddTaskDef("id1", gert::RtsTaskDefFaker())
                          .BuildDetail();
  auto model = model_detail.model;
  auto node_names_to_task_indexes = ConvertToNoeKey(graph, model_detail.task_indexes_to_node_id);

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(false)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();
  std::set<std::string> temp;
  davinci_model.get()->data_dumper_.dump_properties_.model_dump_properties_map_.emplace(DUMP_ALL_MODEL, temp);

  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  mam.SetAllocationHitCount(1U, 1U);
  mam.update_version_ = 1U;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);

  rtStream_t stream = (rtStream_t)1;
  std::vector<uint64_t> active_mem_base_addr = ActiveMemBaseFaker(2, 1).FmBaseIndex(0).ModelIoBaseIndex(1).Build();
  mam.AllocKernelLaunchArgsHostMem(active_mem_base_addr.size());
  uint64_t* active_mem_base_addr_temp = mam.GetActivateMemBaseAddrs();
  for (size_t i = 0; i < active_mem_base_addr.size(); i++) {
    active_mem_base_addr_temp[i] = active_mem_base_addr[i];
  }
  mam.update_version_ = 1;
  uint32_t up = 1;
  ASSERT_EQ(mam.UpdateForExecute(up, stream), SUCCESS);

  for (const auto &task_info : task_list) {
    ASSERT_EQ(reinterpret_cast<StubTaskInfo *>(task_info.get())->GetUpdateHostArgsCalls().size(), 1UL);
  }
  // todo 所有H2D的args拷贝地址、长度正确
}

// fusion段更新testcase
TEST_F(ModelArgsManagerUT, UpdateForExecute_AllTaskInfoCalled_FusionChanged) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub()
      .StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL)
      .StubTaskInfo<DsaStubTaskInfo>(ModelTaskType::MODEL_TASK_DSA)
      .StubTaskInfo<RtsStubTaskInfo>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC);
  auto graph = gert::ShareGraph::BuildAiCoreRtsDsaToIdentityKnownShapeGraph();
  graph->TopologicalSorting();
  auto model_detail = gert::GeModelBuilder(graph)
                          .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                          .AddTaskDef("dsa1", gert::DsaCoreTaskDefFaker())
                          .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                          .AddTaskDef("id1", gert::RtsTaskDefFaker())
                          .BuildDetail();
  auto model = model_detail.model;
  auto node_names_to_task_indexes = ConvertToNoeKey(graph, model_detail.task_indexes_to_node_id);
  std::vector<int64_t> fusion_lengths = {0x100};
  std::vector<int64_t> fm_lengths = {0x100};
  auto davinci_model = DavinciModelFaker()
                           .ModelFusionLengths(fusion_lengths)
                           .ModelFmLengths(fm_lengths)
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();
  std::set<std::string> temp;
  davinci_model.get()->data_dumper_.dump_properties_.model_dump_properties_map_.emplace(DUMP_ALL_MODEL, temp);

  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);
  mam.SetAllocationHitCount(1U, 1U);
  rtStream_t stream = (rtStream_t)1;
  std::vector<uint64_t> active_mem_base_addr = ActiveMemBaseFaker(fusion_lengths.size(), fm_lengths.size(), 2, 1)
                                 .FusionBaseIndex(0).FmBaseIndex(0).ModelIoBaseIndex(1).Build();
  mam.AllocKernelLaunchArgsHostMem(active_mem_base_addr.size());
  uint64_t* active_mem_base_addr_temp = mam.GetActivateMemBaseAddrs();
  for (size_t i = 0; i < active_mem_base_addr.size(); i++) {
    active_mem_base_addr_temp[i] = active_mem_base_addr[i];
  }

  uint32_t ret_up = 4;
  ASSERT_EQ(mam.UpdateForExecute(ret_up, stream),
            SUCCESS);

  for (const auto &task_info : task_list) {
    ASSERT_EQ(reinterpret_cast<StubTaskInfo *>(task_info.get())->GetUpdateDumpInfosCalls().size(), 1UL);
  }
  // fm base changed
  ret_up = 2;
  ASSERT_EQ(mam.UpdateForExecute(ret_up, stream),
            SUCCESS);

  // add1 will be updated in policies model-io,fm-and-model-io,all-one-time
  // add2 will be updated in policies model-io,fm-and-model-io,all-one-time
  // id1 will be updated in policies model-io,fm-and-model-io,all-one-time
  // dsa1 will be updated in policies all-one-time,
  EXPECT_EQ(GET_UPDATE_CALL_TIMES("add1"), 2UL);
  EXPECT_EQ(GET_UPDATE_CALL_TIMES("add2"), 2UL);
  EXPECT_EQ(GET_UPDATE_CALL_TIMES("id1"), 2UL);
  EXPECT_EQ(GET_UPDATE_CALL_TIMES("dsa1"), 1UL);
}

// FM分段申请更新testcase
TEST_F(ModelArgsManagerUT, UpdateForExecute_AllTaskInfoCalled_SubFmChanged) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub()
      .StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL)
      .StubTaskInfo<DsaStubTaskInfo>(ModelTaskType::MODEL_TASK_DSA)
      .StubTaskInfo<RtsStubTaskInfo>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC);
  auto graph = gert::ShareGraph::BuildAiCoreRtsDsaToIdentityKnownShapeGraph();
  graph->TopologicalSorting();
  auto model_detail = gert::GeModelBuilder(graph)
                          .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                          .AddTaskDef("dsa1", gert::DsaCoreTaskDefFaker())
                          .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                          .AddTaskDef("id1", gert::RtsTaskDefFaker())
                          .BuildDetail();
  auto model = model_detail.model;
  auto node_names_to_task_indexes = ConvertToNoeKey(graph, model_detail.task_indexes_to_node_id);
  std::vector<int64_t> fm_lengths = {0x100, 0x100};  // FM分段申请
  auto davinci_model = DavinciModelFaker()
                           .ModelFmLengths(fm_lengths)
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();
  std::set<std::string> temp;
  davinci_model.get()->data_dumper_.dump_properties_.model_dump_properties_map_.emplace(DUMP_ALL_MODEL, temp);

  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);

  rtStream_t stream = (rtStream_t)1;
  std::vector<uint64_t> active_mem_base_addr1 = ActiveMemBaseFaker(fm_lengths.size(), 2, 1).FmBaseIndex(0).ModelIoBaseIndex(1).Build();
  mam.AllocKernelLaunchArgsHostMem(active_mem_base_addr1.size());
  uint64_t* active_mem_base_addr_temp1 = mam.GetActivateMemBaseAddrs();
  for (size_t i = 0; i < active_mem_base_addr1.size(); i++) {
    active_mem_base_addr_temp1[i] = active_mem_base_addr1[i];
  }

  uint32_t up = 4;
  ASSERT_EQ(mam.UpdateForExecute(up, stream), SUCCESS);

  for (const auto &task_info : task_list) {
    ASSERT_EQ(reinterpret_cast<StubTaskInfo *>(task_info.get())->GetUpdateDumpInfosCalls().size(), 1UL);
  }
  // fm base changed
  std::vector<uint64_t> active_mem_base_addr2 = ActiveMemBaseFaker(fm_lengths.size(), 2, 1).FmBaseIndex(1).ModelIoBaseIndex(1).Build();
  mam.AllocKernelLaunchArgsHostMem(active_mem_base_addr2.size());
  uint64_t* active_mem_base_addr_temp2 = mam.GetActivateMemBaseAddrs();
  for (size_t i = 0; i < active_mem_base_addr2.size(); i++) {
    active_mem_base_addr_temp2[i] = active_mem_base_addr2[i];
  }

  up = 3;
  ASSERT_EQ(mam.UpdateForExecute(up, stream), SUCCESS);
  // 2 意味着被刷新了两次， 也就是这个算子的地址含有fm的地址
  EXPECT_EQ(GET_UPDATE_CALL_TIMES("add1"), 2UL);
  EXPECT_EQ(GET_UPDATE_CALL_TIMES("add2"), 2UL);
  EXPECT_EQ(GET_UPDATE_CALL_TIMES("id1"), 2UL);
  EXPECT_EQ(GET_UPDATE_CALL_TIMES("dsa1"), 1UL);
}

TEST_F(ModelArgsManagerUT, UpdateForExecute_OnlyModelIoCalled_IoChanged) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub()
      .StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL)
      .StubTaskInfo<DsaStubTaskInfo>(ModelTaskType::MODEL_TASK_DSA)
      .StubTaskInfo<RtsStubTaskInfo>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC);
  auto graph = gert::ShareGraph::BuildAiCoreRtsDsaToIdentityKnownShapeGraph();
  graph->TopologicalSorting();
  auto model_detail = gert::GeModelBuilder(graph)
                          .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                          .AddTaskDef("dsa1", gert::DsaCoreTaskDefFaker())
                          .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                          .AddTaskDef("id1", gert::RtsTaskDefFaker())
                          .BuildDetail();
  auto model = model_detail.model;
  auto node_names_to_task_indexes = ConvertToNoeKey(graph, model_detail.task_indexes_to_node_id);

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();
  std::set<std::string> temp;
  davinci_model.get()->data_dumper_.dump_properties_.model_dump_properties_map_.emplace(DUMP_ALL_MODEL, temp);

  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  dlog_setlevel(0,0,0);
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);
  mam.SetAllocationHitCount(1U, 1U);
  rtStream_t stream = (rtStream_t)1;
  std::vector<uint64_t> active_mem_base_addr = ActiveMemBaseFaker(2, 1).FmBaseIndex(0).ModelIoBaseIndex(1).Build();
  mam.AllocKernelLaunchArgsHostMem(active_mem_base_addr.size());
  uint64_t* active_mem_base_addr_temp = mam.GetActivateMemBaseAddrs();
  for (size_t i = 0; i < active_mem_base_addr.size(); i++) {
    active_mem_base_addr_temp[i] = active_mem_base_addr[i];
  }

  mam.SetFuncHandle((void*)100);

  uint32_t up = 2;
  ASSERT_EQ(mam.UpdateForExecute(up, stream), SUCCESS);

  // model io addr changed
  CLEAR_ALL_TASK_INFO(task_list);
  ASSERT_EQ(mam.UpdateForExecute(up, stream), SUCCESS);
  EXPECT_EQ(GET_UPDATE_CALL_TIMES("add1"), 1UL);
  EXPECT_EQ(GET_UPDATE_CALL_TIMES("add2"), 1UL);
  EXPECT_EQ(GET_UPDATE_CALL_TIMES("id1"), 0UL);
  EXPECT_EQ(GET_UPDATE_CALL_TIMES("dsa1"), 0UL);

  // model io and fm does not change
  CLEAR_ALL_TASK_INFO(task_list);

  up = 0;
  ASSERT_EQ(mam.UpdateForExecute(up, stream), SUCCESS);
  EXPECT_EQ(GET_UPDATE_CALL_TIMES("add1"), 0UL);
  EXPECT_EQ(GET_UPDATE_CALL_TIMES("add2"), 0UL);
  EXPECT_EQ(GET_UPDATE_CALL_TIMES("id1"), 0UL);
  EXPECT_EQ(GET_UPDATE_CALL_TIMES("dsa1"), 0UL);

  // version == 1
  mam.update_version_ = 1U;
  ASSERT_EQ(mam.UpdateForExecute(up, stream), SUCCESS);

  // version == 3
  mam.update_version_ = 3U;
  up = 2;
  gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::ProfilingType>(
          {gert::ProfilingType::kTaskTime}));
  ASSERT_EQ(mam.UpdateForExecute(up, stream), SUCCESS);

  gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::ProfilingType>(
          {gert::ProfilingType::kTaskTime, gert::ProfilingType::kDevice}));
  ASSERT_EQ(mam.UpdateForExecute(up, stream), SUCCESS);
  gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(0);

  // print dfx info
  mam.dfx_info_.get_model_args_device_table_flag = true;
  ASSERT_EQ(mam.UpdateForExecute(up, stream), SUCCESS);

  up = 0;
  ASSERT_EQ(mam.UpdateForExecute(up, stream), SUCCESS);
  dlog_setlevel(0,3,0);
}


TEST_F(ModelArgsManagerUT, UpdateForExecute_OnlyModelIoCalled_IoChanged_NoModelIoHit) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub()
      .StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL)
      .StubTaskInfo<DsaStubTaskInfo>(ModelTaskType::MODEL_TASK_DSA)
      .StubTaskInfo<RtsStubTaskInfo>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC);
  auto graph = gert::ShareGraph::BuildAiCoreRtsDsaToIdentityKnownShapeGraph();
  graph->TopologicalSorting();
  auto model_detail = gert::GeModelBuilder(graph)
                          .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                          .AddTaskDef("dsa1", gert::DsaCoreTaskDefFaker())
                          .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                          .AddTaskDef("id1", gert::RtsTaskDefFaker())
                          .BuildDetail();
  auto model = model_detail.model;
  auto node_names_to_task_indexes = ConvertToNoeKey(graph, model_detail.task_indexes_to_node_id);

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();
  std::set<std::string> temp;
  davinci_model.get()->data_dumper_.dump_properties_.model_dump_properties_map_.emplace(DUMP_ALL_MODEL, temp);

  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);
  mam.SetAllocationHitCount(1U, 0U);
  rtStream_t stream = (rtStream_t)1;
  std::vector<uint64_t> active_mem_base_addr = ActiveMemBaseFaker(2, 1).FmBaseIndex(0).ModelIoBaseIndex(1).Build();
  mam.AllocKernelLaunchArgsHostMem(active_mem_base_addr.size());
  uint64_t* active_mem_base_addr_temp = mam.GetActivateMemBaseAddrs();
  for (size_t i = 0; i < active_mem_base_addr.size(); i++) {
    active_mem_base_addr_temp[i] = active_mem_base_addr[i];
  }

  uint32_t up = 0;
  ASSERT_EQ(mam.UpdateForExecute(up, stream), SUCCESS);

  // model io addr changed and no model io hit
  CLEAR_ALL_TASK_INFO(task_list);
  std::vector<uint64_t> active_mem_base_addr1 = ActiveMemBaseFaker(2, 1).FmBaseIndex(0).ModelIoBaseIndex(2).Build();
  mam.AllocKernelLaunchArgsHostMem(active_mem_base_addr1.size());
  uint64_t* active_mem_base_addr_temp1 = mam.GetActivateMemBaseAddrs();
  for (size_t i = 0; i < active_mem_base_addr1.size(); i++) {
    active_mem_base_addr_temp1[i] = active_mem_base_addr1[i];
  }
  ASSERT_EQ(mam.UpdateForExecute(up, stream), SUCCESS);
  EXPECT_EQ(GET_UPDATE_CALL_TIMES("add1"), 0UL);
  EXPECT_EQ(GET_UPDATE_CALL_TIMES("add2"), 0UL);
  EXPECT_EQ(GET_UPDATE_CALL_TIMES("id1"), 0UL);
  EXPECT_EQ(GET_UPDATE_CALL_TIMES("dsa1"), 0UL);
}

/**
 * todo 存在不支持刷新的model io时，这部分model io的更新函数不会被调用
 * todo 当前构造不出完全不需要刷新的task来
 */

TEST_F(ModelArgsManagerUT, UpdateForExecute_SqeLaunchedCorrect_SqePlacementExists) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub()
      .StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL)
      .StubTaskInfo<DsaStubTaskInfo>(ModelTaskType::MODEL_TASK_DSA)
      .StubTaskInfo<RtsStubTaskInfo>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC);
  auto graph = gert::ShareGraph::BuildAiCoreRtsDsaToIdentityKnownShapeGraph();
  graph->TopologicalSorting();
  auto model_detail = gert::GeModelBuilder(graph)
                          .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                          .AddTaskDef("dsa1", gert::DsaCoreTaskDefFaker())
                          .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                          .AddTaskDef("id1", gert::RtsTaskDefFaker())
                          .BuildDetail();
  auto model = model_detail.model;
  auto node_names_to_task_indexes = ConvertToNoeKey(graph, model_detail.task_indexes_to_node_id);

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(false)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();

  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);
  for (size_t i = 0UL; i < task_list.size(); ++i) {
    mam.OnTaskDistributed(i, task_list[i].get());
  }

  rtStream_t stream = (rtStream_t)1;
  std::vector<uint64_t> active_mem_base_addr = ActiveMemBaseFaker(2, 1).FmBaseIndex(0).ModelIoBaseIndex(1).Build();
  mam.AllocKernelLaunchArgsHostMem(active_mem_base_addr.size());
  uint64_t* active_mem_base_addr_temp = mam.GetActivateMemBaseAddrs();
  for (size_t i = 0; i < active_mem_base_addr.size(); i++) {
    active_mem_base_addr_temp[i] = active_mem_base_addr[i];
  }

  uint32_t up = 4;
  ASSERT_EQ(mam.UpdateForExecute(up, stream), SUCCESS);

  ASSERT_EQ(runtime_stub.GetRtsRuntimeStub().GetLaunchSqeUpdateTaskArgs().size(), 1);

  auto &arg = *(runtime_stub.GetRtsRuntimeStub().GetLaunchSqeUpdateTaskArgs().begin());
  auto task_info = GET_TASK_INFO("dsa1");

  EXPECT_EQ(arg.stream_id, task_info->GetStreamId());
  EXPECT_EQ(arg.task_id, task_info->GetTaskID());
  EXPECT_EQ(
      arg.src,
      ValueToPtr(
          task_info->GetInitCalls().at(0).args.at(static_cast<int32_t>(ArgsPlacement::kArgsPlacementSqe)).dev_addr));
  EXPECT_EQ(arg.cnt, task_info->GetGenTaskArgsLen());
  EXPECT_EQ(arg.stm, (rtStream_t)1);
}
TEST_F(ModelArgsManagerUT, Compatibility_PrintWarningMessage_WhenHistoryTaskInfoExists) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevel(DLOG_INFO);
  runtime_stub.GetTaskInfoFactoryStub().StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL);
  auto graph = gert::ShareGraph::BuildTwoAddNodeKnownShapeGraph();
  graph->TopologicalSorting();
  auto model = gert::GeModelBuilder(graph)
                   .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                   .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                   .Build();

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();

  ModelArgsManager mam(davinci_model.get());

  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);
  runtime_stub.GetSlogStub().FindWarnLogEndsWith(
      "CompatibleAllocation:There are still normal args allocation, not support in new version, known args len hbm/ts "
      "104/0");
  runtime_stub.GetSlogStub().FindInfoLogEndsWith("Begin to allocate fixed addr.");
  rtStream_t stream = (rtStream_t)1;
  std::vector<uint64_t> active_mem_base_addr = ActiveMemBaseFaker(2, 1).FmBaseIndex(0).ModelIoBaseIndex(1).Build();
  mam.AllocKernelLaunchArgsHostMem(active_mem_base_addr.size());
  uint64_t* active_mem_base_addr_temp = mam.GetActivateMemBaseAddrs();
  for (size_t i = 0; i < active_mem_base_addr.size(); i++) {
    active_mem_base_addr_temp[i] = active_mem_base_addr[i];
  }

  uint32_t up = 0;
  ASSERT_EQ(mam.UpdateForExecute(up, stream), SUCCESS);
}

TEST_F(ModelArgsManagerUT, GetRtsMemoryType_HostSvm) {
  ASSERT_EQ(GetRtsMemoryType(ArgsPlacement::kArgsPlacementHostSvm, 1), RT_MEMORY_HOST_SVM);
  ASSERT_EQ(GetRtsMemoryType(ArgsPlacement::kArgsPlacementHbm, 1), RT_MEMORY_HBM);
}

TEST_F(ModelArgsManagerUT, GenModelArgsRefreshInfosForTask) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub().StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL);
  auto graph = gert::ShareGraph::BuildTwoAddNodeKnownShapeGraph();
  graph->TopologicalSorting();
  auto model = gert::GeModelBuilder(graph)
                   .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                   .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                   .Build();

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();

  ModelArgsManager mam(davinci_model.get());

  TaskArgsRefreshInfo info_high_32_bit = {
      0,
      0x32,
      0,
      0x64,
      ArgsPlacement::kArgsPlacementHbm,
      ArgsFormatPolicy::kAddrHigh32Bit
  };

  TaskArgsRefreshInfo info_low_32_bit = {
      0,
      0x32,
      0,
      0x68,
      ArgsPlacement::kArgsPlacementHbm,
      ArgsFormatPolicy::kAddrLow32Bit
  };
  std::vector<TaskArgsRefreshInfo> infos;
  infos.emplace_back(std::move(info_high_32_bit));
  infos.emplace_back(std::move(info_low_32_bit));
  mam.allocation_ids_to_model_args_refresh_infos_addr_high_32bit.resize(infos.size());
  mam.allocation_ids_to_model_args_refresh_infos_addr_low_32bit.resize(infos.size());
  PisToArgs args;
  args[0].dev_addr = PtrToValue(malloc(1024));
  void *host = nullptr;
  host = malloc(1024);
  args[0U].host_addr = host;
  args[0].len = 1024;

  auto node = BuildTestNode();
  ASSERT_EQ(mam.GenModelArgsRefreshInfosForTask(infos, args, node), SUCCESS);

  free(ValueToPtr(args[0].dev_addr));
  free(host);
}

TEST_F(ModelArgsManagerUT, GenAddrRefreshOpKernelLaunchArgsInfo_Test) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub().StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL);
  auto graph = gert::ShareGraph::BuildTwoAddNodeKnownShapeGraph();
  graph->TopologicalSorting();
  auto model = gert::GeModelBuilder(graph)
                   .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                   .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                   .Build();

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();
  //davinci_model.init();
  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  dlog_setlevel(0,0,0);
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);

  TaskArgsRefreshInfo info_high_32_bit = {
      0,
      0,
      0,
      0,
      ArgsPlacement::kArgsPlacementHbm,
      ArgsFormatPolicy::kAddrHigh32Bit
  };

  TaskArgsRefreshInfo info_low_32_bit = {
      0,
      0x20,
      0,
      0x20,
      ArgsPlacement::kArgsPlacementHbm,
      ArgsFormatPolicy::kAddrLow32Bit
  };

  TaskArgsRefreshInfo info_all = {
      1,
      0x8,
      1,
      0x8,
      ArgsPlacement::kArgsPlacementHbm,
      ArgsFormatPolicy::kAddrAll
  };

  std::vector<TaskArgsRefreshInfo> infos;
  infos.emplace_back(std::move(info_high_32_bit));
  infos.emplace_back(std::move(info_low_32_bit));
  infos.emplace_back(std::move(info_all));
  mam.allocation_ids_to_model_args_refresh_infos_addr_all.resize(infos.size());
  mam.allocation_ids_to_model_args_refresh_infos_addr_high_32bit.resize(infos.size());
  mam.allocation_ids_to_model_args_refresh_infos_addr_low_32bit.resize(infos.size());
  PisToArgs args;
  args[0].dev_addr = mam.model_args_[0].model_args_device_addr;
  args[0U].host_addr = mam.model_args_[0].model_args_host_addr.get();
  args[0].len = 1024;

  auto node = BuildTestNode();
  uint64_t offset_num = 5;
  ASSERT_EQ(mam.GenModelArgsRefreshInfosForTask(infos, args, node), SUCCESS);
  mam.model_args_len_[0] = 40;
  //ASSERT_EQ(mam.GenAddrRefreshOpKernelLaunchArgsInfo(pls), SUCCESS);
  mam.davinci_model_->logical_mem_allocations_.resize(40);
  ASSERT_EQ(mam.AllocKernelLaunchArgsHostMem(mam.davinci_model_->logical_mem_allocations_.size()), SUCCESS);
  ASSERT_EQ(mam.GenKernelLaunchArgs(offset_num), SUCCESS);
  ASSERT_EQ(mam.GenAddrRefreshIndexAndOffset(offset_num), SUCCESS);
  mam.launched_args_unique_ptr_ = nullptr;
  ASSERT_NE(mam.GenKernelLaunchArgs(offset_num), SUCCESS);
  dlog_setlevel(0,3,0);
}

TEST_F(ModelArgsManagerUT, GenKernelLaunchArgs_with_300_modelio_Test) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub().StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL);
  auto graph = gert::ShareGraph::BuildTwoAddNodeKnownShapeGraph();
  graph->TopologicalSorting();
  auto model = gert::GeModelBuilder(graph)
                   .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                   .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                   .Build();

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();
  //davinci_model.init();
  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);

  TaskArgsRefreshInfo info_high_32_bit = {
      0,
      0,
      0,
      0,
      ArgsPlacement::kArgsPlacementHbm,
      ArgsFormatPolicy::kAddrHigh32Bit
  };

  TaskArgsRefreshInfo info_low_32_bit = {
      0,
      0x4,
      0,
      0x4,
      ArgsPlacement::kArgsPlacementHbm,
      ArgsFormatPolicy::kAddrLow32Bit
  };

  TaskArgsRefreshInfo info_all = {
      1,
      0x8,
      1,
      0x8,
      ArgsPlacement::kArgsPlacementHbm,
      ArgsFormatPolicy::kAddrAll
  };

  std::vector<TaskArgsRefreshInfo> infos;
  infos.emplace_back(std::move(info_high_32_bit));
  infos.emplace_back(std::move(info_low_32_bit));
  infos.emplace_back(std::move(info_all));
  mam.allocation_ids_to_model_args_refresh_infos_addr_all.resize(infos.size());
  mam.allocation_ids_to_model_args_refresh_infos_addr_high_32bit.resize(infos.size());
  mam.allocation_ids_to_model_args_refresh_infos_addr_low_32bit.resize(infos.size());
  PisToArgs args;
  args[0].dev_addr = mam.model_args_[0].model_args_device_addr;
  args[0U].host_addr = mam.model_args_[0].model_args_host_addr.get();
  args[0].len = 1024;

  auto node = BuildTestNode();
  uint64_t offset_num = 300;
  mam.davinci_model_->logical_mem_allocations_.resize(300);
  //ArgsPlacement pls = ArgsPlacement::kArgsPlacementHbm;
  ASSERT_EQ(mam.AllocKernelLaunchArgsHostMem(mam.davinci_model_->logical_mem_allocations_.size()), SUCCESS);
  ASSERT_EQ(mam.GenKernelLaunchArgs(offset_num), SUCCESS);
}

TEST_F(ModelArgsManagerUT, GenAllocationToIowPaRemapInfos_TaskNoSupportPaRemap) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub().StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL);
  auto graph = gert::ShareGraph::BuildTwoAddNodeKnownShapeGraph();
  graph->TopologicalSorting();
  auto model = gert::GeModelBuilder(graph)
                   .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                   .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                   .Build();

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();

  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);
  ASSERT_EQ(mam.GetFixedAddrBulk().device_addr, nullptr);
  ASSERT_EQ(task_list.size(), 2UL);

  ASSERT_EQ(mam.allocation_ids_to_iow_pa_remap_infos_.size(), 5UL);
  ASSERT_EQ(mam.allocation_ids_to_iow_pa_remap_infos_[0].size(), 3UL); // fm
  ASSERT_EQ(mam.allocation_ids_to_iow_pa_remap_infos_[1].size(), 2UL); // input1
  ASSERT_EQ(mam.allocation_ids_to_iow_pa_remap_infos_[2].size(), 1UL); // input2
  ASSERT_EQ(mam.allocation_ids_to_iow_pa_remap_infos_[3].size(), 1UL); // output
  ASSERT_EQ(mam.allocation_ids_to_iow_pa_remap_infos_[4].size(), 0UL); // absolute

  rtStream_t stream = (rtStream_t)1;
  std::vector<uint64_t> active_mem_base_addr = ActiveMemBaseFaker(2, 1).FmBaseIndex(0).ModelIoBaseIndex(1).Build();
  mam.AllocKernelLaunchArgsHostMem(active_mem_base_addr.size());
  uint64_t* active_mem_base_addr_temp = mam.GetActivateMemBaseAddrs();
  for (size_t i = 0; i < active_mem_base_addr.size(); i++) {
    active_mem_base_addr_temp[i] = active_mem_base_addr[i];
  }

  uint32_t up = 3;
  ASSERT_EQ(mam.UpdateForExecute(up, stream), SUCCESS);

  uint64_t va = mam.last_bases_[0]; //fm allocation
  uint64_t va_len = mam.id_to_len_[0];
  std::vector<std::pair<uint64_t, uint64_t>> overlap_range;
  ASSERT_EQ(mam.PaRemapped(va, 0, va_len, overlap_range), FAILED);
  ASSERT_EQ(mam.pa_remap_match_nosupport_num_, 3);
  ASSERT_EQ(overlap_range.size(), 1);
  ASSERT_EQ(overlap_range[0].first, mam.last_bases_[0]);
  ASSERT_EQ(overlap_range[0].second, mam.last_bases_[0] + mam.id_to_len_[0] - 1);
}

TEST_F(ModelArgsManagerUT, PaRemapped_NoVaCrossOver) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub().StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL);
  auto graph = gert::ShareGraph::BuildTwoAddNodeKnownShapeGraph();
  graph->TopologicalSorting();
  auto model = gert::GeModelBuilder(graph)
                   .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                   .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                   .Build();

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();

  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);
  ASSERT_EQ(mam.GetFixedAddrBulk().device_addr, nullptr);
  ASSERT_EQ(task_list.size(), 2UL);
  mam.allocation_ids_to_iow_pa_remap_infos_.clear();
  mam.allocation_ids_to_iow_pa_remap_infos_.resize(2);
  mam.last_bases_.clear();
  mam.id_to_len_.clear();

  // fm[0] [200，300）
  mam.last_bases_.emplace_back(200UL);
  mam.id_to_len_.emplace_back(100UL);

  // absolute
  mam.last_bases_.emplace_back(0UL);
  mam.id_to_len_.emplace_back(0xFFFFFFFFFFFFFFFFUL);

  // tensor [210, 220)
  struct IowPaRemapInfo iow_pa_remap_info = {nullptr, 0U, 10UL, 10UL, PaRemapPolicy::KNoSupport};
  mam.allocation_ids_to_iow_pa_remap_infos_[0].insert(std::move(iow_pa_remap_info));

  mam.AllocKernelLaunchArgsHostMem(mam.davinci_model_->GetLogicalMemAllocation().size());
  auto active_mem_base_ptr = mam.GetActivateMemBaseAddrs();
  for (size_t i = 0; i < mam.last_bases_.size(); i++) {
    active_mem_base_ptr[i] =  mam.last_bases_[i];
  }
  // va [100, 200)
  std::vector<std::pair<uint64_t, uint64_t>> overlap_range;
  ASSERT_EQ(mam.PaRemapped(100, 0, 100, overlap_range), PARAM_INVALID);
  ASSERT_EQ(overlap_range.size(), 0);
};

TEST_F(ModelArgsManagerUT, PaRemapped_VaCrossOverWithNoTensor) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub().StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL);
  auto graph = gert::ShareGraph::BuildTwoAddNodeKnownShapeGraph();
  graph->TopologicalSorting();
  auto model = gert::GeModelBuilder(graph)
                   .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                   .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                   .Build();

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();

  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);
  ASSERT_EQ(mam.GetFixedAddrBulk().device_addr, nullptr);
  ASSERT_EQ(task_list.size(), 2UL);
  mam.allocation_ids_to_iow_pa_remap_infos_.clear();
  mam.allocation_ids_to_iow_pa_remap_infos_.resize(2);
  mam.last_bases_.clear();
  mam.id_to_len_.clear();

  // fm[0] [200，300）
  mam.last_bases_.emplace_back(200UL);
  mam.id_to_len_.emplace_back(100UL);

  // absolute
  mam.last_bases_.emplace_back(0UL);
  mam.id_to_len_.emplace_back(0xFFFFFFFFFFFFFFFFUL);
  mam.AllocKernelLaunchArgsHostMem(mam.davinci_model_->GetLogicalMemAllocation().size());
  auto active_mem_base_ptr = mam.GetActivateMemBaseAddrs();
  for (size_t i = 0; i < mam.last_bases_.size(); i++) {
    active_mem_base_ptr[i] =  mam.last_bases_[i];
  }
  // va [200, 300)
  std::vector<std::pair<uint64_t, uint64_t>> overlap_range;
  ASSERT_EQ(mam.PaRemapped(200, 0, 100, overlap_range), SUCCESS);
  ASSERT_EQ(mam.pa_remap_match_support_num_, 1);
  ASSERT_EQ(overlap_range.size(), 1);
  ASSERT_EQ(overlap_range[0].first, 200);
  ASSERT_EQ(overlap_range[0].second, 299);
};


//|-------va-----------|
//    |------fm------------------|
//    |tensor1|
//       |tensor2|
//    |tensor3---|
//              |tensor4--|
TEST_F(ModelArgsManagerUT, PaRemapped_VaRightCrossOver) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub().StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL);
  auto graph = gert::ShareGraph::BuildTwoAddNodeKnownShapeGraph();
  graph->TopologicalSorting();
  auto model = gert::GeModelBuilder(graph)
                   .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                   .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                   .Build();

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();

  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);
  ASSERT_EQ(mam.GetFixedAddrBulk().device_addr, nullptr);
  ASSERT_EQ(task_list.size(), 2UL);
  mam.allocation_ids_to_iow_pa_remap_infos_.clear();
  mam.allocation_ids_to_iow_pa_remap_infos_.resize(2);
  mam.last_bases_.clear();
  mam.id_to_len_.clear();

  // fm[0] [200，300）
  mam.last_bases_.emplace_back(200UL);
  mam.id_to_len_.emplace_back(100UL);

  // absolute
  mam.last_bases_.emplace_back(0UL);
  mam.id_to_len_.emplace_back(0xFFFFFFFFFFFFFFFFUL);

  // tensor [200, 220)
  mam.allocation_ids_to_iow_pa_remap_infos_[0].insert({nullptr, 0U, 0UL, 20UL, PaRemapPolicy::KNoSupport});
  // tensor [210, 230)
  mam.allocation_ids_to_iow_pa_remap_infos_[0].insert({nullptr, 0U, 10UL, 20UL, PaRemapPolicy::KNoSupport});
  // tensor [200, 230)
  mam.allocation_ids_to_iow_pa_remap_infos_[0].insert({nullptr, 0U, 0UL, 30UL, PaRemapPolicy::KNoSupport});
  // tensor [230, 260)
  mam.allocation_ids_to_iow_pa_remap_infos_[0].insert({nullptr, 0U, 30UL, 30UL, PaRemapPolicy::KNoSupport});
  mam.AllocKernelLaunchArgsHostMem(mam.davinci_model_->GetLogicalMemAllocation().size());
  auto active_mem_base_ptr = mam.GetActivateMemBaseAddrs();
  for (size_t i = 0; i < mam.last_bases_.size(); i++) {
    active_mem_base_ptr[i] =  mam.last_bases_[i];
  }
  // va [100, 250)
  std::vector<std::pair<uint64_t, uint64_t>> overlap_range;
  ASSERT_EQ(mam.PaRemapped(100, 0, 150, overlap_range), FAILED);
  ASSERT_EQ(mam.pa_remap_match_nosupport_num_, 4);
  ASSERT_EQ(overlap_range.size(), 1);
  ASSERT_EQ(overlap_range[0].first, 200);
  ASSERT_EQ(overlap_range[0].second, 249);
};


//    |----------va-----------|
//|--------------fm--------------------|
//  |tensor1|  |tensor2|  |tensor3|
TEST_F(ModelArgsManagerUT, PaRemapped_VaAllCrossOver) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub().StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL);
  auto graph = gert::ShareGraph::BuildTwoAddNodeKnownShapeGraph();
  graph->TopologicalSorting();
  auto model = gert::GeModelBuilder(graph)
                   .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                   .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                   .Build();

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();

  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);
  ASSERT_EQ(mam.GetFixedAddrBulk().device_addr, nullptr);
  ASSERT_EQ(task_list.size(), 2UL);
  mam.allocation_ids_to_iow_pa_remap_infos_.clear();
  mam.allocation_ids_to_iow_pa_remap_infos_.resize(2);
  mam.last_bases_.clear();
  mam.id_to_len_.clear();

  // fm[0] [100，300）
  mam.last_bases_.emplace_back(100UL);
  mam.id_to_len_.emplace_back(200UL);

  // absolute
  mam.last_bases_.emplace_back(0UL);
  mam.id_to_len_.emplace_back(0xFFFFFFFFFFFFFFFFUL);

  // tensor [150, 180)
  mam.allocation_ids_to_iow_pa_remap_infos_[0].insert({nullptr, 0U, 50UL, 30UL, PaRemapPolicy::KNoSupport});
  // tensor [190, 210)
  mam.allocation_ids_to_iow_pa_remap_infos_[0].insert({nullptr, 0U, 90UL, 20UL, PaRemapPolicy::KNoSupport});
  // tensor [220, 260)
  mam.allocation_ids_to_iow_pa_remap_infos_[0].insert({nullptr, 0U, 120UL, 40UL, PaRemapPolicy::KNoSupport});
  // tensor [250, 290)
  mam.allocation_ids_to_iow_pa_remap_infos_[0].insert({nullptr, 0U, 250UL, 40UL, PaRemapPolicy::KNoSupport});
  mam.AllocKernelLaunchArgsHostMem(mam.davinci_model_->GetLogicalMemAllocation().size());
  auto active_mem_base_ptr = mam.GetActivateMemBaseAddrs();
  for (size_t i = 0; i < mam.last_bases_.size(); i++) {
    active_mem_base_ptr[i] =  mam.last_bases_[i];
  }

  // va [160, 250)
  std::vector<std::pair<uint64_t, uint64_t>> overlap_range;
  ASSERT_EQ(mam.PaRemapped(160, 0, 90, overlap_range), FAILED);
  ASSERT_EQ(mam.pa_remap_match_nosupport_num_, 3);
  ASSERT_EQ(overlap_range.size(), 1);
  ASSERT_EQ(overlap_range[0].first, 160);
  ASSERT_EQ(overlap_range[0].second, 249);
};


//           |---va----------|
//|---fm-----------------|
//   |tensor1| |tenosr2|
TEST_F(ModelArgsManagerUT, PaRemapped_VaLeftCrossOverWithHalfOpen) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub().StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL);
  auto graph = gert::ShareGraph::BuildTwoAddNodeKnownShapeGraph();
  graph->TopologicalSorting();
  auto model = gert::GeModelBuilder(graph)
                   .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                   .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                   .Build();

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();

  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);
  ASSERT_EQ(mam.GetFixedAddrBulk().device_addr, nullptr);
  ASSERT_EQ(task_list.size(), 2UL);
  mam.allocation_ids_to_iow_pa_remap_infos_.clear();
  mam.allocation_ids_to_iow_pa_remap_infos_.resize(2);
  mam.last_bases_.clear();
  mam.id_to_len_.clear();

  // fm[0] [100，300）
  mam.last_bases_.emplace_back(100UL);
  mam.id_to_len_.emplace_back(200UL);

  // absolute
  mam.last_bases_.emplace_back(0UL);
  mam.id_to_len_.emplace_back(0xFFFFFFFFFFFFFFFFUL);

  // tensor [150, 180)
  mam.allocation_ids_to_iow_pa_remap_infos_[0].insert({nullptr, 0U, 50UL, 30UL, PaRemapPolicy::KNoSupport});
  // tensor [190, 210)
  mam.allocation_ids_to_iow_pa_remap_infos_[0].insert({nullptr, 0U, 90UL, 20UL, PaRemapPolicy::KNoSupport});
  mam.AllocKernelLaunchArgsHostMem(mam.davinci_model_->GetLogicalMemAllocation().size());
  auto active_mem_base_ptr = mam.GetActivateMemBaseAddrs();
  for (size_t i = 0; i < mam.last_bases_.size(); i++) {
    active_mem_base_ptr[i] =  mam.last_bases_[i];
  }
  // va [180, 320)
  std::vector<std::pair<uint64_t, uint64_t>> overlap_range;
  ASSERT_EQ(mam.PaRemapped(180, 0, 140, overlap_range), FAILED);
  ASSERT_EQ(mam.pa_remap_match_nosupport_num_, 1);
  ASSERT_EQ(overlap_range.size(), 1);
  ASSERT_EQ(overlap_range[0].first, 180);
  ASSERT_EQ(overlap_range[0].second, 299);
};


//           |---va----------|
//|---fm-----------------|
//     |tensor1| |tenosr2|
TEST_F(ModelArgsManagerUT, PaRemapped_VaLeftCrossOver) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub().StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL);
  auto graph = gert::ShareGraph::BuildTwoAddNodeKnownShapeGraph();
  graph->TopologicalSorting();
  auto model = gert::GeModelBuilder(graph)
                   .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                   .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                   .Build();

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();

  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);
  ASSERT_EQ(mam.GetFixedAddrBulk().device_addr, nullptr);
  ASSERT_EQ(task_list.size(), 2UL);
  mam.allocation_ids_to_iow_pa_remap_infos_.clear();
  mam.allocation_ids_to_iow_pa_remap_infos_.resize(2);
  mam.last_bases_.clear();
  mam.id_to_len_.clear();

  // fm[0] [100，300）
  mam.last_bases_.emplace_back(100UL);
  mam.id_to_len_.emplace_back(200UL);

  // absolute
  mam.last_bases_.emplace_back(0UL);
  mam.id_to_len_.emplace_back(0xFFFFFFFFFFFFFFFFUL);

  // tensor [150, 180)
  mam.allocation_ids_to_iow_pa_remap_infos_[0].insert({nullptr, 0U, 50UL, 30UL, PaRemapPolicy::KNoSupport});
  // tensor [190, 210)
  mam.allocation_ids_to_iow_pa_remap_infos_[0].insert({nullptr, 0U, 90UL, 20UL, PaRemapPolicy::KNoSupport});
  mam.AllocKernelLaunchArgsHostMem(mam.davinci_model_->GetLogicalMemAllocation().size());
  auto active_mem_base_ptr = mam.GetActivateMemBaseAddrs();
  for (size_t i = 0; i < mam.last_bases_.size(); i++) {
    active_mem_base_ptr[i] =  mam.last_bases_[i];
  }
  // va [170, 320)
  std::vector<std::pair<uint64_t, uint64_t>> overlap_range;
  ASSERT_EQ(mam.PaRemapped(170, 0, 150, overlap_range), FAILED);
  ASSERT_EQ(mam.pa_remap_match_nosupport_num_, 2);
  ASSERT_EQ(overlap_range.size(), 1);
  ASSERT_EQ(overlap_range[0].first, 170);
  ASSERT_EQ(overlap_range[0].second, 299);
};


//      |----------va------------------|
//|-----fm1--------|    |---fm2-----------|
//   |tensor1|            |tensor2|
TEST_F(ModelArgsManagerUT, PaRemapped_VaCrossOverWithMultiFm) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub().StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL);
  auto graph = gert::ShareGraph::BuildTwoAddNodeKnownShapeGraph();
  graph->TopologicalSorting();
  auto model = gert::GeModelBuilder(graph)
                   .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                   .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                   .Build();

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();

  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);
  ASSERT_EQ(mam.GetFixedAddrBulk().device_addr, nullptr);
  ASSERT_EQ(task_list.size(), 2UL);
  mam.allocation_ids_to_iow_pa_remap_infos_.clear();
  mam.allocation_ids_to_iow_pa_remap_infos_.resize(3);
  mam.last_bases_.clear();
  mam.id_to_len_.clear();

  // fm[0] [100，300）
  mam.last_bases_.emplace_back(100UL);
  mam.id_to_len_.emplace_back(200UL);

  // fm[1] [400，600）
  mam.last_bases_.emplace_back(400UL);
  mam.id_to_len_.emplace_back(200UL);

  // absolute
  mam.last_bases_.emplace_back(0UL);
  mam.id_to_len_.emplace_back(0xFFFFFFFFFFFFFFFFUL);

  // tensor [150, 180)
  mam.allocation_ids_to_iow_pa_remap_infos_[0].insert({nullptr, 0U, 50UL, 30UL, PaRemapPolicy::KNoSupport});
  // tensor [420, 500)
  mam.allocation_ids_to_iow_pa_remap_infos_[1].insert({nullptr, 0U, 20UL, 80UL, PaRemapPolicy::KNoSupport});
  mam.AllocKernelLaunchArgsHostMem(mam.davinci_model_->GetLogicalMemAllocation().size());
  auto active_mem_base_ptr = mam.GetActivateMemBaseAddrs();
  for (size_t i = 0; i < mam.last_bases_.size(); i++) {
    active_mem_base_ptr[i] =  mam.last_bases_[i];
  }
  // va [170, 500)
  std::vector<std::pair<uint64_t, uint64_t>> overlap_range;
  ASSERT_EQ(mam.PaRemapped(170, 0, 330, overlap_range), FAILED);
  ASSERT_EQ(mam.pa_remap_match_nosupport_num_, 2);
  ASSERT_EQ(overlap_range.size(), 2);
  ASSERT_EQ(overlap_range[0].first, 170);
  ASSERT_EQ(overlap_range[0].second, 299);
  ASSERT_EQ(overlap_range[1].first, 400);
  ASSERT_EQ(overlap_range[1].second, 499);
};

//      |----------va------------------|
//|-----absolute----------------------------|
//          |tensor1|
TEST_F(ModelArgsManagerUT, PaRemapped_VaCrossOverWithAbsolute) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub().StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL);
  auto graph = gert::ShareGraph::BuildTwoAddNodeKnownShapeGraph();
  graph->TopologicalSorting();
  auto model = gert::GeModelBuilder(graph)
                   .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                   .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                   .Build();

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();

  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);
  ASSERT_EQ(mam.GetFixedAddrBulk().device_addr, nullptr);
  ASSERT_EQ(task_list.size(), 2UL);
  mam.allocation_ids_to_iow_pa_remap_infos_.clear();
  mam.allocation_ids_to_iow_pa_remap_infos_.resize(2);
  mam.last_bases_.clear();
  mam.id_to_len_.clear();

  // fm[0] [100，300）
  mam.last_bases_.emplace_back(100UL);
  mam.id_to_len_.emplace_back(200UL);

  // absolute
  mam.last_bases_.emplace_back(0UL);
  mam.id_to_len_.emplace_back(0xFFFFFFFFFFFFFFFFUL);

  // tensor [400, 450)
  mam.allocation_ids_to_iow_pa_remap_infos_[1].insert({nullptr, 1U, 400UL, 50UL, PaRemapPolicy::KNoSupport});

  // tensor [500, 550)
  mam.allocation_ids_to_iow_pa_remap_infos_[1].insert({nullptr, 1U, 500UL, 50UL, PaRemapPolicy::KNoSupport});
  mam.AllocKernelLaunchArgsHostMem(mam.davinci_model_->GetLogicalMemAllocation().size());
  auto active_mem_base_ptr = mam.GetActivateMemBaseAddrs();
  for (size_t i = 0; i < mam.last_bases_.size(); i++) {
    active_mem_base_ptr[i] =  mam.last_bases_[i];
  }
  // va [350, 600)
  std::vector<std::pair<uint64_t, uint64_t>> overlap_range;
  ASSERT_EQ(mam.PaRemapped(350, 0, 300, overlap_range), FAILED);
  ASSERT_EQ(mam.pa_remap_match_nosupport_num_, 2);
  ASSERT_EQ(overlap_range.size(), 2);
  ASSERT_EQ(overlap_range[0].first, 400);
  ASSERT_EQ(overlap_range[0].second, 449);
  ASSERT_EQ(overlap_range[1].first, 500);
  ASSERT_EQ(overlap_range[1].second, 549);
};

//    |----------va-----------|
//|--------------fm--------------------|
//        |tensor1 empty|
TEST_F(ModelArgsManagerUT, PaRemapped_VaAllCrossOverWithEmptyTensor) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub().StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL);
  auto graph = gert::ShareGraph::BuildTwoAddNodeKnownShapeGraph();
  graph->TopologicalSorting();
  auto model = gert::GeModelBuilder(graph)
                   .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                   .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                   .Build();

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();

  ModelArgsManager mam(davinci_model.get());
  std::vector<TaskInfoPtr> task_list;
  ASSERT_EQ(mam.Init(*(model->GetModelTaskDefPtr()), &task_list), SUCCESS);
  ASSERT_EQ(mam.GetFixedAddrBulk().device_addr, nullptr);
  ASSERT_EQ(task_list.size(), 2UL);
  mam.allocation_ids_to_iow_pa_remap_infos_.clear();
  mam.allocation_ids_to_iow_pa_remap_infos_.resize(2);
  mam.last_bases_.clear();
  mam.id_to_len_.clear();

  // fm[0] [100，300）
  mam.last_bases_.emplace_back(100UL);
  mam.id_to_len_.emplace_back(200UL);

  // absolute
  mam.last_bases_.emplace_back(0UL);
  mam.id_to_len_.emplace_back(0xFFFFFFFFFFFFFFFFUL);

  // tensor [160, 160)
  mam.allocation_ids_to_iow_pa_remap_infos_[0].insert({nullptr, 0U, 160UL, 0UL, PaRemapPolicy::KNoSupport});

  // va [160, 250)
  std::vector<std::pair<uint64_t, uint64_t>> overlap_range;
  mam.AllocKernelLaunchArgsHostMem(mam.davinci_model_->GetLogicalMemAllocation().size());
  auto active_mem_base_ptr = mam.GetActivateMemBaseAddrs();
  for (size_t i = 0; i < mam.last_bases_.size(); i++) {
    active_mem_base_ptr[i] =  mam.last_bases_[i];
  }
  ASSERT_EQ(mam.PaRemapped(160, 0, 90, overlap_range), SUCCESS);
  ASSERT_EQ(overlap_range.size(), 1);
};

TEST_F(ModelArgsManagerUT, CalculateUpdateModelParamTiling_test) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetTaskInfoFactoryStub().StubTaskInfo<AicoreStubTaskInfo>(ModelTaskType::MODEL_TASK_KERNEL);
  auto graph = gert::ShareGraph::BuildTwoAddNodeKnownShapeGraph();
  graph->TopologicalSorting();
  auto model = gert::GeModelBuilder(graph)
                   .AddTaskDef("add1", gert::AiCoreTaskDefFaker("add1"))
                   .AddTaskDef("add2", gert::AiCoreTaskDefFaker("add2"))
                   .Build();

  auto davinci_model = DavinciModelFaker()
                           .SetFmRefreshable(true)
                           .GeModel(model)
                           .GenerateSymbolForTaskInfoFaker(&(runtime_stub.GetTaskInfoFactoryStub()))
                           .Build();

  ModelArgsManager mam(davinci_model.get());
  UpdateModelParamTilingData tiling{0};
  uint32_t active_base_len = 512 * 8;
  uint32_t index_len = 32 * 1024;
  uint32_t block_dim{0};
  mam.CalculateUpdateModelParamTiling(active_base_len, index_len, block_dim, tiling);
  EXPECT_EQ(tiling.totalActiveBaseTblCnt, 1024);
  EXPECT_EQ(tiling.blockCnt, 2048);
  EXPECT_EQ(tiling.tileCnt, 2048);
  EXPECT_EQ(tiling.tailCnt, 2048);
  EXPECT_EQ(tiling.lastTailCnt, 2048);
  EXPECT_EQ(tiling.tileNum, 1);
  EXPECT_EQ(tiling.lastTileNum, 1);
  EXPECT_EQ(block_dim, 4);
  index_len = 128 * 1024;
  mam.CalculateUpdateModelParamTiling(active_base_len, index_len, block_dim, tiling);
  EXPECT_EQ(tiling.totalActiveBaseTblCnt, 1024);
  EXPECT_EQ(tiling.blockCnt, 2368);
  EXPECT_EQ(tiling.tileCnt, 2368);
  EXPECT_EQ(tiling.tailCnt, 2368);
  EXPECT_EQ(tiling.lastTailCnt, 1984);
  EXPECT_EQ(tiling.tileNum, 1);
  EXPECT_EQ(tiling.lastTileNum, 1);
  EXPECT_EQ(block_dim, 14);
};

}  // namespace ge

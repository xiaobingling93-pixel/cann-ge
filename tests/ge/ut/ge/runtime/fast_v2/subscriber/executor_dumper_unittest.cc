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
#include <mmpa/mmpa_api.h>
#include "core/execution_data.h"
#include "common/bg_test.h"
#include "core/model_v2_executor_unittest.h"
#include "stub/gert_runtime_stub.h"
#include "core/builder/model_v2_executor_builder.h"
#include "common/fake_node_helper.h"
#include "common/dump/dump_manager.h"
#include "common/dump/dump_properties.h"
#include "framework/runtime/model_v2_executor.h"
#include "framework/common/types.h"
#include "kernel/tensor_attr.h"
#include "graph/utils/graph_utils.h"
#include "common/share_graph.h"
#include "faker/fake_value.h"
#include <faker/aicpu_taskdef_faker.h>
#include "lowering/model_converter.h"
#include "core/executor_error_code.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"
#include "register/ffts_node_calculater_registry.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"
#include "macro_utils/dt_public_scope.h"
#include "graph/load/model_manager/davinci_model.h"
#include "subscriber/dumper/executor_dumper.h"
#include "common/global_variables/diagnose_switch.h"
#include "macro_utils/dt_public_unscope.h"
#include "common/model/ge_model.h"
#include "engine/aicore/kernel/mixl2_update_kernel.h"  //todo: to be deleted
#include "engine/aicpu/kernel/ffts_plus/aicpu_update_kernel.h" // todo: to be deleted
#include "depends/profiler/src/dump_stub.h"
#include "exe_graph/lowering/lowering_definitions.h"

namespace gert {
namespace {
uint64_t FindFirstNonEmptyId(ExecutorDumper *dumper) {
  for (uint64_t i = 0U; i < dumper->kernel_idxes_to_dump_units_.size(); ++i) {
    if (!dumper->kernel_idxes_to_dump_units_[i].empty()) {
      return i;
    }
  }
  return std::numeric_limits<uint64_t>::max();
}
void ExecutorRun(ModelV2Executor *executor) {
  ASSERT_EQ(executor->Load(), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);

  rtStream_t rt_stream;
  ASSERT_EQ(rtStreamCreate(&rt_stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(rt_stream));

  ASSERT_EQ(
      executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(), outputs.size()),
      ge::GRAPH_SUCCESS);
  ASSERT_EQ(executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(rt_stream);
}
}  // namespace
class ExecutorDumperUT : public bg::BgTest {
 protected:
  void TearDown() {
    GlobalDumper::GetInstance()->SetEnableFlags(0UL);
    ge::DumpManager::GetInstance().RemoveDumpProperties(ge::kInferSessionId);
  }
};
namespace {
ge::Status UpdateAddrs(const std::vector<Chain *> &chain_addrs, std::vector<uintptr_t> &dump_addrs) {
  for (auto &addr : chain_addrs) {
    GE_ASSERT_NOTNULL(addr);
    const auto tensor_data = addr->GetPointer<TensorData>();
    if ((tensor_data->GetPlacement() == kOnHost) || (tensor_data->GetPlacement() == kFollowing)) {
      GELOGW("Not device addr, do not dump.");
      return ge::FAILED;
    } else {
      dump_addrs.emplace_back(reinterpret_cast<uintptr_t>(tensor_data->GetAddr()));
    }
  }
  return ge::SUCCESS;
}

ge::ExecuteGraphPtr GetExecuteGraph(ge::ExecuteGraph *const root_graph, SubExeGraphType eg_type) {
  auto graph_type_str = GetSubExeGraphTypeStr(eg_type);
  auto graph_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(root_graph, graph_type_str);
  GE_ASSERT_NOTNULL(graph_node, "Failed to find node %s from root graph, execute graph type %d", graph_type_str,
                    static_cast<int32_t>(eg_type));

  return ge::FastNodeUtils::GetSubgraphFromNode(graph_node, 0)->shared_from_this();
}

std::string FindKernelNameByKeyWord(ge::ExecuteGraph *const exe_graph, const std::string &key_word) {
  for (auto &node : GetExecuteGraph(exe_graph, kMainExeGraph)->GetDirectNode()) {
    if (node->GetName().find(key_word) != std::string::npos) {
      return node->GetName();
    }
  }
  return "";
}

std::string FindKernelNameByStartKeyWord(ge::ExecuteGraph *const exe_graph, const std::string &key_word) {
  for (auto &node : GetExecuteGraph(exe_graph, kMainExeGraph)->GetDirectNode()) {
    if (node->GetName().find(key_word, 0) == 0) {
      return node->GetName();
    }
  }
  return "";
}

std::string FindKernelNameByStartKeyWord(const std::string kernel_name, const std::string &key_word) {
  if (kernel_name.find(key_word, 0) == 0) {
    return kernel_name;
  }
  return "";
}
}  // namespace

TEST_F(ExecutorDumperUT, EnableDump_Ok) {
  auto exe_graph = std::make_shared<ge::ExecuteGraph>("");
  const auto &extend_info = ge::MakeShared<const SubscriberExtendInfo>(nullptr, exe_graph, nullptr, ge::ModelData{},
                                                                       nullptr, SymbolsToValue{}, 0, "", nullptr,
                                                                       std::unordered_map<std::string, TraceAttr>{});
  ExecutorDumper dumper(extend_info);
  GlobalDumper::GetInstance()->SetEnableFlags(BuiltInSubscriberUtil::BuildEnableFlags<DumpType>({DumpType::kDataDump}));
  EXPECT_TRUE(dumper.IsEnable(DumpType::kDataDump));
  GlobalDumper::GetInstance()->SetEnableFlags(
      BuiltInSubscriberUtil::BuildEnableFlags<DumpType>({DumpType::kExceptionDump}));
  EXPECT_TRUE(dumper.IsEnable(DumpType::kExceptionDump));
  EXPECT_FALSE(dumper.IsEnable(DumpType::kDataDump));
}

TEST_F(ExecutorDumperUT, EnableAllDump_Ok) {
  auto exe_graph = std::make_shared<ge::ExecuteGraph>("");
  const auto &extend_info = ge::MakeShared<const SubscriberExtendInfo>(nullptr, exe_graph, nullptr, ge::ModelData{},
                                                                       nullptr, SymbolsToValue{}, 0, "", nullptr,
                                                                       std::unordered_map<std::string, TraceAttr>{});
  ExecutorDumper dumper(extend_info);
  GlobalDumper::GetInstance()->SetEnableFlags(BuiltInSubscriberUtil::BuildEnableFlags<DumpType>({DumpType::kAll}));
  EXPECT_TRUE(dumper.IsEnable(DumpType::kDataDump));
  EXPECT_TRUE(dumper.IsEnable(DumpType::kExceptionDump));
}

TEST_F(ExecutorDumperUT, DumperConstruct_Ok) {
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto executor_2_exe_graph = BuildExecutorFromSingleNode();
  auto executor = std::move(executor_2_exe_graph.executor);
  auto exe_graph = executor_2_exe_graph.exe_graph;
  ASSERT_NE(executor, nullptr);
  ASSERT_NE(exe_graph, nullptr);
  const auto &extend_info = ge::MakeShared<const SubscriberExtendInfo>(
      executor.get(), exe_graph, nullptr, ge::ModelData{}, nullptr, SymbolsToValue{}, 0, "", nullptr,
      std::unordered_map<std::string, TraceAttr>{});
  ExecutorDumper dumper(extend_info);
  EXPECT_NE(dumper.extend_info_->executor, nullptr);
  EXPECT_NE(dumper.extend_info_->exe_graph, nullptr);
}

TEST_F(ExecutorDumperUT, DumperInit_InitKernelNamesToExeNodes_WithSingleNodeGraph) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  EXPECT_FALSE(dumper->kernel_names_to_exe_nodes_.empty());
  EXPECT_EQ(static_cast<const ExecutionData *>(
                dumper->extend_info_->executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData())
                ->base_ed.node_num,
            dumper->kernel_names_to_exe_nodes_.size());
  EXPECT_EQ(static_cast<const ExecutionData *>(
                dumper->extend_info_->executor->GetExeGraphExecutor(kInitExeGraph)->GetExecutionData())
                ->base_ed.node_num,
            dumper->init_kernel_names_to_exe_nodes_.size());
}

TEST_F(ExecutorDumperUT, DumperInit_InitNodeNameToDumpUnits_WithSingleNodeGraph) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  EXPECT_FALSE(dumper->node_names_to_dump_units_.empty());
  auto dump_unit = dumper->node_names_to_dump_units_["add1"];
  EXPECT_EQ(dump_unit.total_update_count, 3);
  auto exe_graph = dumper->extend_info_->exe_graph;
  auto infer_shape_kernel_name = FindKernelNameByKeyWord(exe_graph.get(), "InferShape");
  auto infer_shape_exe_node = dumper->kernel_names_to_exe_nodes_[infer_shape_kernel_name];
  EXPECT_EQ(reinterpret_cast<KernelContext *>(&infer_shape_exe_node->context)->GetOutput(0),
            dump_unit.output_shapes[0]);
  EXPECT_EQ(reinterpret_cast<KernelContext *>(&infer_shape_exe_node->context)->GetInput(0), dump_unit.input_shapes[0]);
  EXPECT_EQ(reinterpret_cast<KernelContext *>(&infer_shape_exe_node->context)->GetInput(1), dump_unit.input_shapes[1]);
  auto alloc_kernel_name = FindKernelNameByKeyWord(exe_graph.get(), "AllocMemHbm_add");
  auto alloc_exe_node = dumper->kernel_names_to_exe_nodes_[alloc_kernel_name];
  ASSERT_NE(alloc_exe_node, nullptr);
  EXPECT_EQ(reinterpret_cast<KernelContext *>(&alloc_exe_node->context)->GetOutput(0), dump_unit.output_addrs[0]);
  EXPECT_NE(dump_unit.input_addrs[0], nullptr);
  EXPECT_NE(dump_unit.input_addrs[1], nullptr);
}

TEST_F(ExecutorDumperUT, DumperInit_InitInputTest) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  EXPECT_FALSE(dumper->node_names_to_dump_units_.empty());
  auto &dump_unit = dumper->node_names_to_dump_units_["add1"];
  EXPECT_EQ(dump_unit.total_update_count, 3);

  auto node = dump_unit.node;
  auto input_info = node->GetOpDescBarePtr()->TryGetExtAttr(kLoweringInputInfo, LowerInputInfo());
  EXPECT_NE(input_info.input_shapes.size(), 0);
  EXPECT_NE(input_info.input_addrs.size(), 0);

  // add a invalid input to node
  ge::GeShape shape({3, 2, 3});
  ge::GeTensorDesc input_desc(shape, ge::FORMAT_RESERVED, ge::DT_UNDEFINED);
  node->GetOpDescBarePtr()->AddInputDesc(input_desc);

  dump_unit.input_shapes.clear();
  dump_unit.input_addrs.clear();

  auto res = dumper->InitInputShapes(input_info, &dump_unit);
  EXPECT_EQ(res, ge::GRAPH_SUCCESS);
  EXPECT_EQ(dump_unit.input_shapes.size(), node->GetOpDescBarePtr()->GetAllInputsSize());
  EXPECT_EQ(dump_unit.input_shapes.back(), nullptr);  // last input must be null as the last input is invalid

  res = dumper->InitInputAddrs(input_info, &dump_unit);
  EXPECT_EQ(res, ge::GRAPH_SUCCESS);
  EXPECT_EQ(dump_unit.input_addrs.size(), node->GetOpDescBarePtr()->GetAllInputsSize());
  EXPECT_EQ(dump_unit.input_addrs.back(), nullptr);  // last input must be null as the last input is invalid
}

TEST_F(ExecutorDumperUT, IncreaseUpdataCount_Ok_AfterKernelDone) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  uint64_t i = FindFirstNonEmptyId(dumper);
  auto fake_node = FakeNodeHelper::FakeNode("add1", "test", i);
  auto dump_unit = dumper->kernel_idxes_to_dump_units_[fake_node.node.node_id][0];
  dump_unit->total_update_count = 2;
  EXPECT_EQ(dump_unit->cur_update_count, 0);
  dumper->OnUpdateDumpUnit(kExecuteEnd, fake_node.node);
  EXPECT_EQ(dump_unit->cur_update_count, 1);
  dump_unit->Clear();
  EXPECT_EQ(dump_unit->cur_update_count, 0);
}

TEST_F(ExecutorDumperUT, DumperInit_InitKerelIdxesToDumpUnits_WithSingleNodeGraph) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  ASSERT_NE(executor, nullptr);
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  auto exe_graph = dumper->extend_info_->exe_graph;
  auto infer_shape_kernel_name = FindKernelNameByKeyWord(exe_graph.get(), "InferShape");
  auto infer_shape_exe_node = dumper->kernel_names_to_exe_nodes_[infer_shape_kernel_name];
  EXPECT_EQ(dumper->kernel_idxes_to_dump_units_[infer_shape_exe_node->node_id][0],
            &dumper->node_names_to_dump_units_["add1"]);
  auto launch_kernel_name = FindKernelNameByStartKeyWord(exe_graph.get(), "LaunchKernelWithHandle_add1");
  auto launch_exe_node = dumper->kernel_names_to_exe_nodes_[launch_kernel_name];
  ASSERT_NE(launch_exe_node, nullptr);
  EXPECT_EQ(dumper->kernel_idxes_to_dump_units_[launch_exe_node->node_id][0],
            &dumper->node_names_to_dump_units_["add1"]);
}

TEST_F(ExecutorDumperUT, UpdateShapeInfo_Ok_AfterInfershapeKernelExecution) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  ASSERT_NE(executor, nullptr);
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  auto exe_graph = dumper->extend_info_->exe_graph;
  auto name = FindKernelNameByKeyWord(exe_graph.get(), "InferShape");
  ;
  auto exe_node = dumper->kernel_names_to_exe_nodes_[name];
  reinterpret_cast<KernelContext *>(&exe_node->context)
      ->GetOutput(0)
      ->SetWithDefaultDeleter(new StorageShape{{256}, {256}});
  auto op_desc = dumper->kernel_idxes_to_dump_units_[exe_node->node_id][0]->node->GetOpDescBarePtr();
  ge::OpDescPtr op_desc_dump = std::make_shared<ge::OpDesc>(*op_desc);
  dumper->kernel_idxes_to_dump_units_[exe_node->node_id][0]->UpdateOutputShapes(op_desc_dump);
  EXPECT_EQ(op_desc_dump->GetOutputDesc(0).GetShape().GetDim(0), 256);
  EXPECT_EQ(op_desc_dump->GetOutputDesc(0).GetOriginShape().GetDim(0), 256);
}

TEST_F(ExecutorDumperUT, Dumper_spaceaddr_empty_ok) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  auto fake_node = FakeNodeHelper::FakeNode("add1", "AicpuLaunchCCKernel", FindFirstNonEmptyId(dumper));
  auto dump_unit = dumper->kernel_idxes_to_dump_units_[fake_node.node.node_id][0];
  EXPECT_EQ(dumper->SaveWorkSpaceAddrForAiCpuLaunchCCNode(fake_node.node), ge::SUCCESS);
  dump_unit->Clear();
}

TEST_F(ExecutorDumperUT, Dumper_spaceaddr_ok) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump(true);
  ASSERT_NE(executor, nullptr);
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  auto fake_node = FakeNodeHelper::FakeNode("add1", "AicpuLaunchCCKernel", FindFirstNonEmptyId(dumper));
  auto dump_unit = dumper->kernel_idxes_to_dump_units_[fake_node.node.node_id][0];
  EXPECT_EQ(dumper->SaveWorkSpaceAddrForAiCpuLaunchCCNode(fake_node.node), ge::SUCCESS);
  dump_unit->Clear();
}

TEST_F(ExecutorDumperUT, UpdateAddrInfo_Ok_AfterAllocKernelExecution) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  ASSERT_NE(executor, nullptr);
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  auto exe_graph = dumper->extend_info_->exe_graph;
  auto name = FindKernelNameByKeyWord(exe_graph.get(), "AllocMemHbm_add");
  auto exe_node = dumper->kernel_names_to_exe_nodes_[name];
  ASSERT_NE(exe_node, nullptr);
  reinterpret_cast<KernelContext *>(&exe_node->context)
      ->GetOutput(0)
      ->SetWithDefaultDeleter(new TensorData{(void *)1024, nullptr, 0, kOnDeviceHbm});
  std::vector<uintptr_t> actual_output_addrs;
  EXPECT_EQ(UpdateAddrs(dumper->kernel_idxes_to_dump_units_[exe_node->node_id][0]->output_addrs, actual_output_addrs),
            ge::SUCCESS);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(actual_output_addrs[0]), reinterpret_cast<uintptr_t>((void *)1024));
}

TEST_F(ExecutorDumperUT, DoDataDump_Ok) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  ExecutorRun(executor.get());
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  StorageShape storage_shape{{4}, {4, 1}};
  kernel::BuildTensorAttr attr{kOnDeviceHbm, ge::DT_FLOAT16, {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ExpandDimsType()}};
  Tensor tensor_holder{storage_shape, attr.storage_format, attr.placement, attr.data_type, nullptr};
  tensor_holder.MutableTensorData() = TensorData{(void *)1024, nullptr, 0, kOnDeviceHbm};
  auto context_holder_1 =
      KernelRunContextFaker().KernelIONum(1, 2).Outputs({&tensor_holder, &tensor_holder.MutableTensorData()}).Build();
  auto &node_add_dump_unit = dumper->node_names_to_dump_units_["add1"];
  ASSERT_NE(context_holder_1.GetContext<KernelContext>(), nullptr);
  ASSERT_FALSE(node_add_dump_unit.output_addrs.empty());
  ASSERT_FALSE(node_add_dump_unit.input_addrs.empty());
  node_add_dump_unit.output_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  node_add_dump_unit.input_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  node_add_dump_unit.input_addrs[1] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  gert::StorageShape x2_shape = {{4, 8, 16, 32, 64}, {4, 8, 16, 4, 2, 16, 16}};
  gert::StorageShape x1_shape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
  auto context_holder_2 = KernelRunContextFaker()
                              .Inputs({&x1_shape})
                              .Outputs({&x2_shape})
                              .KernelIONum(1, 1)
                              .NodeIoNum(1, 1)
                              .IrInstanceNum({1})
                              .Build();
  node_add_dump_unit.output_shapes[0] = context_holder_2.GetContext<KernelContext>()->GetOutput(0);
  node_add_dump_unit.input_shapes[0] = const_cast<Chain *>(context_holder_2.GetContext<KernelContext>()->GetInput(0));
  node_add_dump_unit.input_shapes[1] = const_cast<Chain *>(context_holder_2.GetContext<KernelContext>()->GetInput(0));
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  auto op_desc = dumper->node_names_to_dump_units_["add1"].node->GetOpDesc();
  std::vector<int64_t> tvm_workspace_memory_type = {ge::AicpuWorkSpaceType::CUST_LOG};
  ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_AICPU_WORKSPACE_TYPE, tvm_workspace_memory_type);
  op_desc->SetWorkspaceBytes(vector<int64_t>{32});
  const auto properties = ge::DumpManager::GetInstance().GetDumpProperties(ge::kInferSessionId);
  auto compute_graph = dumper->extend_info_->root_graph;
  ASSERT_NE(compute_graph, nullptr);
  auto node_add = compute_graph->FindFirstNodeMatchType("Add");
  ge::GeShape output0_shape(node_add->GetOpDesc()->GetOutputDesc(0).GetShape());
  ge::GeShape input0_shape(node_add->GetOpDesc()->GetInputDesc(0).GetShape());
  ge::GeShape input1_shape(node_add->GetOpDesc()->GetInputDesc(1).GetShape());
  ge::GeShape input0_origin_shape(node_add->GetOpDesc()->GetInputDesc(0).GetOriginShape());
  ge::GeShape input1_origin_shape(node_add->GetOpDesc()->GetInputDesc(1).GetOriginShape());
  EXPECT_EQ(dumper->DoDataDump(dumper->node_names_to_dump_units_["add1"], properties), ge::SUCCESS);
  EXPECT_EQ(node_add->GetOpDesc()->GetOutputDesc(0).GetShape(), output0_shape);
  EXPECT_EQ(node_add->GetOpDesc()->GetInputDesc(0).GetShape(), input0_shape);
  EXPECT_EQ(node_add->GetOpDesc()->GetInputDesc(1).GetShape(), input1_shape);
  EXPECT_EQ(node_add->GetOpDesc()->GetInputDesc(0).GetOriginShape(), input0_origin_shape);
  EXPECT_EQ(node_add->GetOpDesc()->GetInputDesc(1).GetOriginShape(), input1_origin_shape);
  std::vector<uintptr_t> actual_output_addrs{};
  EXPECT_EQ(UpdateAddrs(dumper->node_names_to_dump_units_["add1"].output_addrs, actual_output_addrs), ge::SUCCESS);
  EXPECT_EQ(actual_output_addrs[0], reinterpret_cast<uintptr_t>((void *)1024));
  std::vector<uintptr_t> actual_input_addrs{};
  EXPECT_EQ(UpdateAddrs(dumper->node_names_to_dump_units_["add1"].input_addrs, actual_input_addrs), ge::SUCCESS);
  EXPECT_EQ(actual_input_addrs[0], reinterpret_cast<uintptr_t>((void *)1024));
  EXPECT_EQ(actual_input_addrs[1], reinterpret_cast<uintptr_t>((void *)1024));
}

TEST_F(ExecutorDumperUT, DoDataDumpByOriginalName_Ok) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  ExecutorRun(executor.get());
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  StorageShape storage_shape{{4}, {4, 1}};
  kernel::BuildTensorAttr attr{kOnDeviceHbm, ge::DT_FLOAT16, {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ExpandDimsType()}};
  Tensor tensor_holder{storage_shape, attr.storage_format, attr.placement, attr.data_type, nullptr};
  tensor_holder.MutableTensorData() = TensorData{(void *)1024, nullptr, 0, kOnDeviceHbm};
  auto context_holder_1 =
      KernelRunContextFaker().KernelIONum(1, 2).Outputs({&tensor_holder, &tensor_holder.MutableTensorData()}).Build();
  auto &node_add_dump_unit = dumper->node_names_to_dump_units_["add1"];
  ASSERT_NE(context_holder_1.GetContext<KernelContext>(), nullptr);
  ASSERT_FALSE(node_add_dump_unit.output_addrs.empty());
  ASSERT_FALSE(node_add_dump_unit.input_addrs.empty());
  node_add_dump_unit.output_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  node_add_dump_unit.input_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  node_add_dump_unit.input_addrs[1] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  gert::StorageShape x2_shape = {{4, 8, 16, 32, 64}, {4, 8, 16, 4, 2, 16, 16}};
  gert::StorageShape x1_shape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
  auto context_holder_2 = KernelRunContextFaker()
                              .Inputs({&x1_shape})
                              .Outputs({&x2_shape})
                              .KernelIONum(1, 1)
                              .NodeIoNum(1, 1)
                              .IrInstanceNum({1})
                              .Build();
  node_add_dump_unit.output_shapes[0] = context_holder_2.GetContext<KernelContext>()->GetOutput(0);
  node_add_dump_unit.input_shapes[0] = const_cast<Chain *>(context_holder_2.GetContext<KernelContext>()->GetInput(0));
  node_add_dump_unit.input_shapes[1] = const_cast<Chain *>(context_holder_2.GetContext<KernelContext>()->GetInput(0));
  ge::DumpProperties dump_properties;
  std::vector<std::string> original_adds = {"add_ori1", "add_ori2"};
  dump_properties.AddPropertyValue("LAYER_OP_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"add_ori1", "add_ori2"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  auto op_desc = dumper->node_names_to_dump_units_["add1"].node->GetOpDesc();
  ge::AttrUtils::SetListStr(op_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_adds);
  std::vector<int64_t> tvm_workspace_memory_type = {ge::AicpuWorkSpaceType::CUST_LOG};
  ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_AICPU_WORKSPACE_TYPE, tvm_workspace_memory_type);
  op_desc->SetWorkspaceBytes(vector<int64_t>{32});
  const auto properties = ge::DumpManager::GetInstance().GetDumpProperties(ge::kInferSessionId);
  auto compute_graph = dumper->extend_info_->root_graph;
  ASSERT_NE(compute_graph, nullptr);
  auto node_add = compute_graph->FindFirstNodeMatchType("Add");
  ge::GeShape output0_shape(node_add->GetOpDesc()->GetOutputDesc(0).GetShape());
  ge::GeShape input0_shape(node_add->GetOpDesc()->GetInputDesc(0).GetShape());
  ge::GeShape input1_shape(node_add->GetOpDesc()->GetInputDesc(1).GetShape());
  ge::GeShape input0_origin_shape(node_add->GetOpDesc()->GetInputDesc(0).GetOriginShape());
  ge::GeShape input1_origin_shape(node_add->GetOpDesc()->GetInputDesc(1).GetOriginShape());
  EXPECT_EQ(dumper->DoDataDump(dumper->node_names_to_dump_units_["add1"], properties), ge::SUCCESS);
  EXPECT_FALSE(dumper->IsOpInDumpList(properties, "miss"));
  EXPECT_EQ(node_add->GetOpDesc()->GetOutputDesc(0).GetShape(), output0_shape);
  EXPECT_EQ(node_add->GetOpDesc()->GetInputDesc(0).GetShape(), input0_shape);
  EXPECT_EQ(node_add->GetOpDesc()->GetInputDesc(1).GetShape(), input1_shape);
  EXPECT_EQ(node_add->GetOpDesc()->GetInputDesc(0).GetOriginShape(), input0_origin_shape);
  EXPECT_EQ(node_add->GetOpDesc()->GetInputDesc(1).GetOriginShape(), input1_origin_shape);
  std::vector<uintptr_t> actual_output_addrs{};
  EXPECT_EQ(UpdateAddrs(dumper->node_names_to_dump_units_["add1"].output_addrs, actual_output_addrs), ge::SUCCESS);
  EXPECT_EQ(actual_output_addrs[0], reinterpret_cast<uintptr_t>((void *)1024));
  std::vector<uintptr_t> actual_input_addrs{};
  EXPECT_EQ(UpdateAddrs(dumper->node_names_to_dump_units_["add1"].input_addrs, actual_input_addrs), ge::SUCCESS);
  EXPECT_EQ(actual_input_addrs[0], reinterpret_cast<uintptr_t>((void *)1024));
  EXPECT_EQ(actual_input_addrs[1], reinterpret_cast<uintptr_t>((void *)1024));
}

TEST_F(ExecutorDumperUT, DoDataDump_Ok_WithHostAddr) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  ExecutorRun(executor.get());
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  StorageShape storage_shape{{2048}, {2048, 1}};
  auto total_size = 0UL;
  auto tensor_holder =
      Tensor::CreateFollowing(storage_shape.GetStorageShape().GetShapeSize(), ge::DT_FLOAT16, total_size);
  auto context_holder_1 =
      KernelRunContextFaker()
          .KernelIONum(1, 2)
          .Outputs({tensor_holder.get(), &reinterpret_cast<Tensor *>(tensor_holder.get())->MutableTensorData()})
          .Build();
  auto &node_add_dump_unit = dumper->node_names_to_dump_units_["add1"];
  node_add_dump_unit.output_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  node_add_dump_unit.input_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  node_add_dump_unit.input_addrs[1] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  gert::StorageShape x2_shape = {{4, 8, 16, 1, 1}, {4, 8, 16, 1, 1, 1, 1}};
  gert::StorageShape x1_shape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
  auto context_holder_2 = KernelRunContextFaker()
                              .Inputs({&x1_shape})
                              .Outputs({&x2_shape})
                              .KernelIONum(1, 1)
                              .NodeIoNum(1, 1)
                              .IrInstanceNum({1})
                              .Build();
  node_add_dump_unit.output_shapes[0] = context_holder_2.GetContext<KernelContext>()->GetOutput(0);
  node_add_dump_unit.input_shapes[0] = const_cast<Chain *>(context_holder_2.GetContext<KernelContext>()->GetInput(0));
  node_add_dump_unit.input_shapes[1] = const_cast<Chain *>(context_holder_2.GetContext<KernelContext>()->GetInput(0));
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  GertRuntimeStub stub;
  const auto properties = ge::DumpManager::GetInstance().GetDumpProperties(ge::kInferSessionId);
  EXPECT_EQ(dumper->DoDataDump(dumper->node_names_to_dump_units_["add1"], properties), ge::SUCCESS);
  // check dump_op
  auto cpu_args = stub.GetRtsRuntimeStub().PopCpuLaunchArgsByKernelName("DumpDataInfo");
  EXPECT_EQ(cpu_args->GetKernelName(), "DumpDataInfo");
  toolkit::aicpu::dump::OpMappingInfo op_mapping_info;
  op_mapping_info.ParseFromString(cpu_args->GetSerializeDumpInfo());
  EXPECT_EQ(op_mapping_info.task_size(), 1UL);
  toolkit::aicpu::dump::Task task = op_mapping_info.task(0);
  EXPECT_EQ(task.input().at(0).shape().dim(0), 1);
  EXPECT_EQ(task.input().at(0).shape().dim().size(), 5);
  EXPECT_EQ(task.input().at(1).shape().dim(0), 1);
  EXPECT_EQ(task.input().at(1).shape().dim().size(), 5);
  EXPECT_EQ(task.output().at(0).shape().dim(0), 4);
  EXPECT_EQ(task.output().at(0).shape().dim().size(), 7);
}
TEST_F(ExecutorDumperUT, ExceptionDump_Ok_PrepareExceptionDump) {
  SpaceRegistryFaker::SetefaultSpaceRegistryNull();
  ge::diagnoseSwitch::EnableExceptionDump();
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto graph = ShareGraph::BuildSingleNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub fakeRuntime;
  fakeRuntime.GetKernelStub().StubTiling();
  ge::GeRootModelPtr root_model = std::make_shared<ge::GeRootModel>();
  root_model->SetRootGraph(graph);
  ge::ModelData model_data{};
  model_data.om_name = "test";
  auto model_executor = ModelV2Executor::Create(exe_graph, model_data, root_model);
  auto dumper = model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  EXPECT_TRUE(dumper->is_inited_);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto mem_block = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors{{2048}, 2, mem_block.get()};

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  //  check turn off dumper
  ge::diagnoseSwitch::DisableDumper();
  EXPECT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper),
            nullptr);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
  ge::diagnoseSwitch::DisableDumper();
}

TEST_F(ExecutorDumperUT, ExceptionDump_Ok_CUST_AICPU) {
  const auto SetUnknownOpKernelForNoTiling = [](const ge::ComputeGraph::Vistor<ge::NodePtr> &all_nodes) {
    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      auto type = op_desc->GetType();
      if (op_desc->GetType() == ge::ADD) {
        op_desc->SetOpKernelLibName("aicpu_ascend_kernel");
        op_desc->SetWorkspace(vector<int64_t>{32});
        std::vector<int64_t> tvm_workspace_memory_type = {ge::AicpuWorkSpaceType::CUST_LOG};
        ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_AICPU_WORKSPACE_TYPE, tvm_workspace_memory_type);
        op_desc->SetWorkspaceBytes(vector<int64_t>{32});
      }
    }
  };
  ge::diagnoseSwitch::EnableExceptionDump();
  auto graph = ShareGraph::BuildSingleNodeGraph();
  graph->TopologicalSorting();
  SetUnknownOpKernelForNoTiling(graph->GetDirectNode());
  GeModelBuilder builder(graph);
  AiCpuCCTaskDefFaker aicpu_task_def_faker;
  auto ge_root_model = builder.AddTaskDef("Add", aicpu_task_def_faker.SetNeedMemcpy(true)).BuildGeRootModel();

  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub fakeRuntime;
  fakeRuntime.GetKernelStub().StubTiling();
  ge::GeRootModelPtr root_model = std::make_shared<ge::GeRootModel>();
  root_model->SetRootGraph(graph);
  ge::ModelData model_data{};
  model_data.om_name = "test";
  auto model_executor = ModelV2Executor::Create(exe_graph, model_data, root_model);
  auto dumper =
      model_executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  EXPECT_TRUE(dumper->is_inited_);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto mem_block = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors{{2048}, 2, mem_block.get()};

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  //  check turn off dumper
  ge::diagnoseSwitch::DisableDumper();
  EXPECT_NE(model_executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper),
            nullptr);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
  ge::diagnoseSwitch::DisableDumper();
}

TEST_F(ExecutorDumperUT, ExceptionDump_HOSTADDR_CUST_AICPU) {
  const auto SetUnknownOpKernelForNoTiling = [](const ge::ComputeGraph::Vistor<ge::NodePtr> &all_nodes) {
    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      auto type = op_desc->GetType();
      if (op_desc->GetType() == ge::ADD) {
        op_desc->SetOpKernelLibName("aicpu_ascend_kernel");
        op_desc->SetWorkspace(vector<int64_t>{32});
        std::vector<int64_t> tvm_workspace_memory_type = {ge::AicpuWorkSpaceType::CUST_LOG};
        ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_AICPU_WORKSPACE_TYPE, tvm_workspace_memory_type);
        op_desc->SetWorkspaceBytes(vector<int64_t>{32});
      }
    }
  };
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  ge::diagnoseSwitch::EnableExceptionDump();
  auto graph = ShareGraph::BuildSingleNodeGraph();
  graph->TopologicalSorting();
  SetUnknownOpKernelForNoTiling(graph->GetDirectNode());
  GeModelBuilder builder(graph);
  AiCpuCCTaskDefFaker aicpu_task_def_faker;
  auto ge_root_model = builder.AddTaskDef("Add", aicpu_task_def_faker.SetNeedMemcpy(true)).BuildGeRootModel();

  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub fakeRuntime;
  fakeRuntime.GetKernelStub().StubTiling();
  ge::GeRootModelPtr root_model = std::make_shared<ge::GeRootModel>();
  root_model->SetRootGraph(graph);
  ge::ModelData model_data{};
  model_data.om_name = "test";
  auto model_executor = ModelV2Executor::Create(exe_graph, model_data, root_model);
  auto dumper =
      model_executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  EXPECT_TRUE(dumper->is_inited_);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto mem_block = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto outputs = FakeTensors({2048}, 1, nullptr, kOnHost);
  auto inputs = FakeTensors{{2048}, 2, mem_block.get(), kOnHost};

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  //  check turn off dumper
  ge::diagnoseSwitch::DisableDumper();
  EXPECT_NE(model_executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper),
            nullptr);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
  ge::diagnoseSwitch::DisableDumper();
}

TEST_F(ExecutorDumperUT, InitDumper_Ok_WithControlOp) {
  auto compute_graph = ShareGraph::IfGraph2();
  ASSERT_NE(compute_graph, nullptr);
  compute_graph->TopologicalSorting();
  GeModelBuilder builder(compute_graph);
  auto ge_root_model = builder.BuildGeRootModel();
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  auto init_dumper =
      model_executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(model_executor, nullptr);
  const auto &subscriber_extend_info = ge::MakeShared<const SubscriberExtendInfo>(
      model_executor.get(), exe_graph, ge_root_model->GetRootGraph(), ge::ModelData(), ge_root_model,
      init_dumper->extend_info_->symbols_to_value, 0, "", nullptr, std::unordered_map<std::string, TraceAttr>{});
  auto dumper = ge::MakeUnique<ExecutorDumper>(subscriber_extend_info);
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(dumper->Init(), ge::SUCCESS);
  ASSERT_NE(dumper->extend_info_, nullptr);
  const auto if_node = compute_graph->FindFirstNodeMatchType("If");
  const auto iter = dumper->node_names_to_dump_units_.find(if_node->GetName());
  ASSERT_NE(iter, dumper->node_names_to_dump_units_.end());
  auto actual_out_addr_chain = iter->second.output_addrs[0];
  const auto main_graph = GetExecuteGraph(exe_graph.get(), kMainExeGraph);
  const auto exe_node_name = FindKernelNameByStartKeyWord(exe_graph.get(), "If");
  auto if_exe_node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(main_graph.get(), exe_node_name.c_str());
  ASSERT_NE(if_exe_node->GetExtendInfo(), nullptr);
  const auto out_symbol = if_exe_node->GetExtendInfo()->GetOutputSymbol(1);
  ASSERT_NE(out_symbol, ge::kInvalidSymbol);
  EXPECT_EQ(actual_out_addr_chain,
            reinterpret_cast<Chain *>(dumper->extend_info_->symbols_to_value.find(out_symbol)->second));
}

TEST_F(ExecutorDumperUT, DoDataDump_NoCoredDump_WithEmptyAddr) {
  auto graph = std::make_shared<ge::ComputeGraph>("test");
  auto op_data = ge::OpDescBuilder("fake", "Fake").AddInput("x_0").AddInput("x_1").AddOutput("y").Build();
  auto fake_node = graph->AddNode(op_data);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ExecutorRun(executor.get());
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  NodeDumpUnit dump_unit{};
  dump_unit.node = fake_node;
  dump_unit.input_addrs.resize(2, nullptr);
  dump_unit.output_addrs.resize(1, nullptr);
  dump_unit.input_shapes.resize(2, nullptr);
  dump_unit.output_shapes.resize(1, nullptr);
  const auto properties = ge::DumpManager::GetInstance().GetDumpProperties(ge::kInferSessionId);
  EXPECT_EQ(dumper->DoDataDump(dump_unit, properties), ge::SUCCESS);
}

TEST_F(ExecutorDumperUT, SaveSessionId_Ok_LoadWithSession) {
  auto executor = BuildExecutorFromSingleNodeForDump();
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  RtSession rt_session(1);
  ModelLoadArg load_arg{&rt_session, {nullptr, 0}};
  EXPECT_EQ(executor->Load(nullptr, load_arg), ge::SUCCESS);
  EXPECT_EQ(dumper->GetSessionId(), 0);
  GlobalDumper::GetInstance()->SetEnableFlags(BuiltInSubscriberUtil::EnableBit<DumpType>(DumpType::kDataDump));
  ExecutorDumper::OnExecuteEvent(0, dumper, kModelStart, nullptr, kStatusSuccess);
  ExecutorDumper::OnExecuteEvent(0, dumper, kModelEnd, nullptr, kStatusSuccess);
  GlobalDumper::GetInstance()->SetEnableFlags(0);
  EXPECT_EQ(dumper->GetSessionId(), 1);
}

TEST_F(ExecutorDumperUT, OverflowDumpModelStart_SaveSessionIdAndRtStreamSuccess_OnlyOverflowDumpEnabled) {
  auto executor = BuildExecutorFromSingleNodeForDump();
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  RtSession rt_session(1);
  ModelLoadArg load_arg{&rt_session, {nullptr, 0}};
  ModelExecuteArg execute_arg(reinterpret_cast<rtStream_t>(0x11));
  auto outputs = FakeTensors({2048, 1, 1, 1}, 1);
  auto inputs = FakeTensors({2048, 1, 1, 1}, 2);
  ASSERT_EQ(executor->Load(execute_arg, load_arg), ge::SUCCESS);
  ASSERT_EQ(
      executor->Execute(execute_arg, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(), outputs.size()),
      ge::GRAPH_SUCCESS);
  EXPECT_EQ(dumper->GetSessionId(), 0);
  EXPECT_EQ(dumper->streams_, nullptr);

  ge::diagnoseSwitch::EnableOverflowDump();
  ExecutorDumper::OnExecuteEvent(SubExeGraphType::kMainExeGraph, dumper, kModelStart, nullptr, kStatusSuccess);
  EXPECT_EQ(dumper->GetSessionId(), 1);
  EXPECT_EQ(reinterpret_cast<const rtStream_t *>(dumper->streams_->GetData())[0], reinterpret_cast<rtStream_t>(0x11));
  ExecutorDumper::OnExecuteEvent(SubExeGraphType::kMainExeGraph, dumper, kModelEnd, nullptr, kStatusSuccess);
  ge::diagnoseSwitch::DisableDumper();
}

TEST_F(ExecutorDumperUT, OverflowDumpModelStart_DumpOpDebug_OnlyOverflowDumpEnabled) {
  auto executor = BuildExecutorFromSingleNodeForDump();
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  uint64_t session_id = 1U;
  RtSession rt_session(session_id);
  ModelLoadArg load_arg{&rt_session, {nullptr, 0}};
  const rtStream_t stream = reinterpret_cast<rtStream_t>(0x11);
  ModelExecuteArg execute_arg(stream);
  auto outputs = FakeTensors({2048, 1, 1, 1}, 1);
  auto inputs = FakeTensors({2048, 1, 1, 1}, 2);
  ASSERT_EQ(executor->Load(execute_arg, load_arg), ge::SUCCESS);
  ASSERT_EQ(
      executor->Execute(execute_arg, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(), outputs.size()),
      ge::GRAPH_SUCCESS);

  ge::DumpProperties dump_property;
  dump_property.InitInferOpDebug();
  ge::DumpManager::GetInstance().AddDumpProperties(session_id, dump_property);

  ge::diagnoseSwitch::EnableOverflowDump();
  ExecutorDumper::OnExecuteEvent(SubExeGraphType::kMainExeGraph, dumper, kModelStart, nullptr, kStatusSuccess);
  EXPECT_EQ(dumper->GetSessionId(), session_id);
  auto stream_vec = reinterpret_cast<ContinuousVector *>(dumper->streams_);
  auto dump_stream = *(reinterpret_cast<rtStream_t *>(stream_vec->MutableData()) + 0U);
  EXPECT_EQ(dump_stream, stream);
  ExecutorDumper::OnExecuteEvent(SubExeGraphType::kInitExeGraph, dumper, kModelEnd, nullptr, kStatusSuccess);
  ASSERT_EQ(dumper->op_debug_registers_.size(), 1U);
  EXPECT_EQ(dumper->op_debug_registers_[0]->stream_ref_count_[stream], 1);
  ASSERT_NE(dumper->op_debug_registers_[0]->op_debug_tasks_[stream], nullptr);
//  EXPECT_NE(dumper->op_debug_registers_[0]->op_debug_tasks_[stream]->op_debug_addr_, nullptr);

  EXPECT_EQ(dumper->data_dumpers_.size(), 1U);
  EXPECT_EQ(dumper->data_dumpers_[0]->is_op_debug_, true);
  EXPECT_EQ(*(reinterpret_cast<void *const *>(dumper->data_dumpers_[0]->op_debug_addr_)),
            dumper->op_debug_registers_[0]->op_debug_tasks_[stream]->op_debug_addr_);
  EXPECT_EQ(dumper->data_dumpers_[0]->op_debug_task_id_, dumper->op_debug_registers_[0]->op_debug_tasks_[stream]->debug_task_id_);
  EXPECT_EQ(dumper->data_dumpers_[0]->op_debug_stream_id_,
            dumper->op_debug_registers_[0]->op_debug_tasks_[stream]->debug_stream_id_);
  EXPECT_NE(dumper->data_dumpers_[0]->dev_mem_load_, nullptr);
  EXPECT_EQ(dumper->data_dumpers_[0]->load_flag_, true);

  EXPECT_EQ(dumper->is_op_debug_reg_, true);
  ExecutorDumper::OnExecuteEvent(SubExeGraphType::kMainExeGraph, dumper, kModelEnd, nullptr,
                                 kStatusSuccess);  // release mem
  ge::diagnoseSwitch::DisableDumper();
}

TEST_F(ExecutorDumperUT, MultiStream_TwoStream_OverflowDumpModelStart_DataDumperNumIsTwo) {
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamTwoNodeGraph(stream_num, event_num);
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin"))
                           .AddTaskDef("Relu", AiCoreTaskDefFaker("ReluStubBin"))
                           .SetRootModelStreamNum(stream_num)
                           .SetRootModelEventNum(event_num)
                           .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(executor, nullptr);

  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  uint64_t session_id = 1U;
  RtSession rt_session(session_id);
  ModelLoadArg load_arg{&rt_session, {nullptr, 0}};
  const rtStream_t new_stream = reinterpret_cast<rtStream_t>(0x11);
  ModelExecuteArg execute_arg(new_stream);
  auto outputs = FakeTensors({2048, 1, 1, 1}, 1);
  auto inputs = FakeTensors({2048, 1, 1, 1}, 2);
  ASSERT_EQ(executor->Load(execute_arg, load_arg), ge::SUCCESS);
  ASSERT_EQ(
      executor->Execute(execute_arg, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(), outputs.size()),
      ge::GRAPH_SUCCESS);

  ge::DumpProperties dump_property;
  dump_property.InitInferOpDebug();
  ge::DumpManager::GetInstance().AddDumpProperties(session_id, dump_property);

  ge::diagnoseSwitch::EnableOverflowDump();
  ExecutorDumper::OnExecuteEvent(SubExeGraphType::kMainExeGraph, dumper, kModelStart, nullptr, kStatusSuccess);
  EXPECT_EQ(dumper->GetSessionId(), session_id);
  auto stream_vec = reinterpret_cast<ContinuousVector *>(dumper->streams_);
  EXPECT_EQ(stream_vec->GetSize(), 2U);
  EXPECT_EQ(dumper->data_dumpers_.size(), 2U);
  for (size_t i = 0U; i < stream_vec->GetSize(); i++) {
    auto stream = *(reinterpret_cast<rtStream_t *>(stream_vec->MutableData()) + i);
    EXPECT_EQ(dumper->op_debug_registers_[i]->stream_ref_count_[stream], 1);
    ASSERT_NE(dumper->op_debug_registers_[i]->op_debug_tasks_[stream], nullptr);
//    EXPECT_NE(dumper->op_debug_registers_[i]->op_debug_tasks_[stream]->op_debug_addr_, nullptr);
    EXPECT_EQ(dumper->data_dumpers_[i]->is_op_debug_, true);
    EXPECT_EQ(*(reinterpret_cast<void *const *>(dumper->data_dumpers_[i]->op_debug_addr_)),
              dumper->op_debug_registers_[i]->op_debug_tasks_[stream]->op_debug_addr_);
    EXPECT_EQ(dumper->data_dumpers_[i]->op_debug_task_id_,
              dumper->op_debug_registers_[i]->op_debug_tasks_[stream]->debug_task_id_);
    EXPECT_EQ(dumper->data_dumpers_[i]->op_debug_stream_id_,
              dumper->op_debug_registers_[i]->op_debug_tasks_[stream]->debug_stream_id_);
    EXPECT_NE(dumper->data_dumpers_[i]->dev_mem_load_, nullptr);
    EXPECT_EQ(dumper->data_dumpers_[i]->load_flag_, true);
  }
  EXPECT_EQ(dumper->is_op_debug_reg_, true);

  ExecutorDumper::OnExecuteEvent(SubExeGraphType::kMainExeGraph, dumper, kModelEnd, nullptr,
                                 kStatusSuccess);  // release mem
  ge::diagnoseSwitch::DisableDumper();
}

TEST_F(ExecutorDumperUT, OverflowDump_OverflowDetected_StreamSyncReturnOverflow) {
  ge::diagnoseSwitch::EnableOverflowDump();

  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  ExecutorRun(executor.get());
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  StorageShape storage_shape{{2048}, {2048, 1}};
  kernel::BuildTensorAttr attr{kOnDeviceHbm, ge::DT_FLOAT16, {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ExpandDimsType()}};
  Tensor tensor_holder{storage_shape, attr.storage_format, attr.placement, attr.data_type, nullptr};
  tensor_holder.MutableTensorData() = TensorData{(void *)1024, nullptr, 0, kOnDeviceHbm};
  auto context_holder_1 =
      KernelRunContextFaker().KernelIONum(1, 2).Outputs({&tensor_holder, &tensor_holder.MutableTensorData()}).Build();
  auto &node_add_dump_unit = dumper->node_names_to_dump_units_["add1"];
  node_add_dump_unit.output_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  node_add_dump_unit.input_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  node_add_dump_unit.input_addrs[1] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  gert::StorageShape x2_shape = {{4, 8, 16, 1, 1}, {4, 8, 16, 1, 1, 1, 1}};
  gert::StorageShape x1_shape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
  auto context_holder_2 = KernelRunContextFaker()
                              .Inputs({&x1_shape})
                              .Outputs({&x2_shape})
                              .KernelIONum(1, 1)
                              .NodeIoNum(1, 1)
                              .IrInstanceNum({1})
                              .Build();
  node_add_dump_unit.output_shapes[0] = context_holder_2.GetContext<KernelContext>()->GetOutput(0);
  node_add_dump_unit.input_shapes[0] = const_cast<Chain *>(context_holder_2.GetContext<KernelContext>()->GetInput(0));
  node_add_dump_unit.input_shapes[1] = const_cast<Chain *>(context_holder_2.GetContext<KernelContext>()->GetInput(0));

  auto kernel_launch_context_holder = KernelRunContextFaker()
                                          .NodeName("add1")
                                          .KernelIONum(2, 1)
                                          .KernelType("LaunchKernelWithHandle")
                                          .KernelName("Add_LaunchKernelWithHandle")
                                          .Build();
  Node add_kernel_launch_node{"", 0, nullptr, *kernel_launch_context_holder.GetContext<KernelRunContext>()};
  for (auto &kernel_name_and_exe_node : dumper->kernel_names_to_exe_nodes_) {
    if (!FindKernelNameByStartKeyWord(kernel_name_and_exe_node.first, "LaunchKernelWithHandle").empty()) {
      add_kernel_launch_node.node_id = kernel_name_and_exe_node.second->node_id;
    }
  }
  // OverflowDumpLoadStage for saving stream and session id
  GertRuntimeStub stub;
  uint64_t session_id = 1U;
  RtSession rt_session(session_id);
  ModelLoadArg load_arg{&rt_session, {nullptr, 0}};
  const auto stream = reinterpret_cast<rtStream_t>(0x11);
  ModelExecuteArg arg(stream);
  EXPECT_EQ(executor->Load(arg, load_arg), ge::SUCCESS);
  ge::DumpConfig dump_config;
  dump_config.dump_debug = "on";
  dump_config.dump_path = "./";
  ge::DumpManager::GetInstance().SetDumpConf(dump_config);

  ASSERT_EQ(dumper->OverflowDump(&add_kernel_launch_node, kModelStart), ge::SUCCESS);
  auto debug_dump_model_id = dumper->data_dumpers_[0]->model_id_;

  // Do overflow dump
  auto &kernel_launch_dump_units = dumper->kernel_idxes_to_dump_units_[add_kernel_launch_node.node_id];
  kernel_launch_dump_units.resize(1);
  *kernel_launch_dump_units[0] = node_add_dump_unit;
  kernel_launch_dump_units[0]->cur_update_count = 0;
  kernel_launch_dump_units[0]->total_update_count = 1;
  EXPECT_EQ(kernel_launch_dump_units[0]->need_overflow_dump, false);
  ge::AttrUtils::SetBool(kernel_launch_dump_units[0]->node->GetOpDesc(), ge::GLOBALWORKSPACE_TYPE, true);
  mmSetEnv("SYNCSTREAM_OVERFLOW_RET", "aicore", 1);  // for rtStreamSynchronizeWithTimeout stub
  ExecutorDumper::OnExecuteEvent(SubExeGraphType::kMainExeGraph, dumper, kExecuteStart, &add_kernel_launch_node,
                                 kStatusSuccess);
  ExecutorDumper::OnExecuteEvent(SubExeGraphType::kMainExeGraph, dumper, kExecuteEnd, &add_kernel_launch_node,
                                 kStatusSuccess);
  ExecutorDumper::OnExecuteEvent(SubExeGraphType::kMainExeGraph, dumper, kModelEnd, &add_kernel_launch_node,
                                 kStatusSuccess);
  unsetenv("SYNCSTREAM_OVERFLOW_RET");
  EXPECT_EQ(kernel_launch_dump_units[0]->need_overflow_dump, false);  // been clear after dump

  // check dump_op
  auto cpu_args = stub.GetRtsRuntimeStub().PopCpuLaunchArgsByKernelName("DumpDataInfo");
  ASSERT_NE(cpu_args, nullptr);
  EXPECT_EQ(cpu_args->GetKernelName(), "DumpDataInfo");
  toolkit::aicpu::dump::OpMappingInfo op_mapping_info;
  op_mapping_info.ParseFromString(cpu_args->GetSerializeDumpInfo());
  EXPECT_EQ(op_mapping_info.task_size(), 1UL);
  toolkit::aicpu::dump::Task task = op_mapping_info.task(0);
  EXPECT_EQ(task.input().at(0).shape().dim(0), 1);
  EXPECT_EQ(task.input().at(0).shape().dim().size(), 5);
  EXPECT_EQ(task.input().at(1).shape().dim(0), 1);
  EXPECT_EQ(task.input().at(1).shape().dim().size(), 5);
  EXPECT_EQ(task.output().at(0).shape().dim(0), 4);
  EXPECT_EQ(task.output().at(0).shape().dim().size(), 7);
  EXPECT_EQ(task.input().at(0).address(), 1024);
  EXPECT_EQ(task.input().at(1).address(), 1024);
  EXPECT_EQ(task.output().at(0).address(), 1024);
  // check DumpOpDebug and DoDataDump is the same model-id.
  auto data_dump_model_id = op_mapping_info.model_id();
  EXPECT_EQ(debug_dump_model_id, data_dump_model_id);

  ge::diagnoseSwitch::DisableDumper();
}
TEST_F(ExecutorDumperUT, OverflowDump_NoDataDump_ComputeNodeInfoIsNullptr) {
  ge::diagnoseSwitch::EnableOverflowDump();

  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  ExecutorRun(executor.get());
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  StorageShape storage_shape{{2048}, {2048, 1}};
  kernel::BuildTensorAttr attr{kOnDeviceHbm, ge::DT_FLOAT16, {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ExpandDimsType()}};
  Tensor tensor_holder{storage_shape, attr.storage_format, attr.placement, attr.data_type, nullptr};
  tensor_holder.MutableTensorData() = TensorData{(void *)1024, nullptr, 0, kOnDeviceHbm};
  auto context_holder_1 =
      KernelRunContextFaker().KernelIONum(1, 2).Outputs({&tensor_holder, &tensor_holder.MutableTensorData()}).Build();
  auto &node_add_dump_unit = dumper->node_names_to_dump_units_["add1"];
  node_add_dump_unit.output_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  node_add_dump_unit.input_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  node_add_dump_unit.input_addrs[1] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  gert::StorageShape x2_shape = {{4, 8, 16, 1, 1}, {4, 8, 16, 1, 1, 1, 1}};
  gert::StorageShape x1_shape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
  auto context_holder_2 = KernelRunContextFaker()
                              .Inputs({&x1_shape})
                              .Outputs({&x2_shape})
                              .KernelIONum(1, 1)
                              .NodeIoNum(1, 1)
                              .IrInstanceNum({1})
                              .Build();
  node_add_dump_unit.output_shapes[0] = context_holder_2.GetContext<KernelContext>()->GetOutput(0);
  node_add_dump_unit.input_shapes[0] = const_cast<Chain *>(context_holder_2.GetContext<KernelContext>()->GetInput(0));
  node_add_dump_unit.input_shapes[1] = const_cast<Chain *>(context_holder_2.GetContext<KernelContext>()->GetInput(0));

  auto kernel_launch_context_holder = KernelRunContextFaker()
                                          .NodeName("add1")
                                          .KernelIONum(2, 1)
                                          .KernelType("LaunchKernelWithHandle")
                                          .KernelName("Add_LaunchKernelWithHandle")
                                          .Build();
  Node add_kernel_launch_node{"", 0, nullptr, *kernel_launch_context_holder.GetContext<KernelRunContext>()};
  for (auto &kernel_name_and_exe_node : dumper->kernel_names_to_exe_nodes_) {
    if (!FindKernelNameByStartKeyWord(kernel_name_and_exe_node.first, "LaunchKernelWithHandle").empty()) {
      add_kernel_launch_node.node_id = kernel_name_and_exe_node.second->node_id;
    }
  }
  // OverflowDumpLoadStage for saving stream and session id
  GertRuntimeStub stub;
  uint64_t session_id = 1U;
  RtSession rt_session(session_id);
  ModelLoadArg load_arg{&rt_session, {nullptr, 0}};
  const auto stream = reinterpret_cast<rtStream_t>(0x11);
  ModelExecuteArg arg(stream);
  EXPECT_EQ(executor->Load(arg, load_arg), ge::SUCCESS);
  ge::DumpConfig dump_config;
  dump_config.dump_debug = "on";
  dump_config.dump_path = "./";
  ge::DumpManager::GetInstance().SetDumpConf(dump_config);

  ASSERT_EQ(dumper->OverflowDump(&add_kernel_launch_node, kModelStart), ge::SUCCESS);

  // Do overflow dump
  auto &kernel_launch_dump_units = dumper->kernel_idxes_to_dump_units_[add_kernel_launch_node.node_id];
  kernel_launch_dump_units.resize(1);
  *kernel_launch_dump_units[0] = node_add_dump_unit;
  kernel_launch_dump_units[0]->cur_update_count = 0;
  kernel_launch_dump_units[0]->total_update_count = 1;
  EXPECT_EQ(kernel_launch_dump_units[0]->need_overflow_dump, false);
  ge::AttrUtils::SetBool(kernel_launch_dump_units[0]->node->GetOpDesc(), ge::GLOBALWORKSPACE_TYPE, true);
  mmSetEnv("SYNCSTREAM_OVERFLOW_RET", "aicore", 1);  // for rtStreamSynchronizeWithTimeout stub
  add_kernel_launch_node.context.compute_node_info = nullptr;
  ExecutorDumper::OnExecuteEvent(SubExeGraphType::kMainExeGraph, dumper, kExecuteStart, &add_kernel_launch_node,
                                 kStatusSuccess);
  ExecutorDumper::OnExecuteEvent(SubExeGraphType::kMainExeGraph, dumper, kExecuteEnd, &add_kernel_launch_node,
                                 kStatusSuccess);
  ExecutorDumper::OnExecuteEvent(SubExeGraphType::kMainExeGraph, dumper, kModelEnd, &add_kernel_launch_node,
                                 kStatusSuccess);
  unsetenv("SYNCSTREAM_OVERFLOW_RET");
  EXPECT_EQ(kernel_launch_dump_units[0]->need_overflow_dump, false);  // been clear after dump

  // check dump_op
  auto cpu_args = stub.GetRtsRuntimeStub().PopCpuLaunchArgsByKernelName("DumpDataInfo");
  EXPECT_EQ(cpu_args, nullptr);

  ge::diagnoseSwitch::DisableDumper();
}
TEST_F(ExecutorDumperUT, OverflowDump_NoOverflowDetected_StreamSyncOK) {
  ge::diagnoseSwitch::EnableOverflowDump();

  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  ExecutorRun(executor.get());
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  StorageShape storage_shape{{2048}, {2048, 1}};
  kernel::BuildTensorAttr attr{kOnDeviceHbm, ge::DT_FLOAT16, {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ExpandDimsType()}};
  Tensor tensor_holder{storage_shape, attr.storage_format, attr.placement, attr.data_type, nullptr};
  tensor_holder.MutableTensorData() = TensorData{(void *)1024, nullptr, 0, kOnDeviceHbm};
  auto context_holder_1 =
      KernelRunContextFaker().KernelIONum(1, 2).Outputs({&tensor_holder, &tensor_holder.MutableTensorData()}).Build();
  auto &node_add_dump_unit = dumper->node_names_to_dump_units_["add1"];
  node_add_dump_unit.output_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  node_add_dump_unit.input_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  node_add_dump_unit.input_addrs[1] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  gert::StorageShape x2_shape = {{4, 8, 16, 1, 1}, {4, 8, 16, 1, 1, 1, 1}};
  gert::StorageShape x1_shape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
  auto context_holder_2 = KernelRunContextFaker()
                              .Inputs({&x1_shape})
                              .Outputs({&x2_shape})
                              .KernelIONum(1, 1)
                              .NodeIoNum(1, 1)
                              .IrInstanceNum({1})
                              .Build();
  node_add_dump_unit.output_shapes[0] = context_holder_2.GetContext<KernelContext>()->GetOutput(0);
  node_add_dump_unit.input_shapes[0] = const_cast<Chain *>(context_holder_2.GetContext<KernelContext>()->GetInput(0));
  node_add_dump_unit.input_shapes[1] = const_cast<Chain *>(context_holder_2.GetContext<KernelContext>()->GetInput(0));

  auto kernel_launch_context_holder = KernelRunContextFaker()
                                          .NodeName("add1")
                                          .KernelIONum(2, 1)
                                          .KernelType("LaunchKernelWithHandle")
                                          .KernelName("Add_LaunchKernelWithHandle")
                                          .Build();
  Node add_kernel_launch_node{"", 0, nullptr, *kernel_launch_context_holder.GetContext<KernelRunContext>()};
  for (auto &kernel_name_and_exe_node : dumper->kernel_names_to_exe_nodes_) {
    if (!FindKernelNameByStartKeyWord(kernel_name_and_exe_node.first, "LaunchKernelWithHandle").empty()) {
      add_kernel_launch_node.node_id = kernel_name_and_exe_node.second->node_id;
    }
  }
  // OverflowDumpLoadStage for saving stream and session id
  GertRuntimeStub stub;
  uint64_t session_id = 1U;
  RtSession rt_session(session_id);
  ModelLoadArg load_arg{&rt_session, {nullptr, 0}};
  const auto stream = reinterpret_cast<rtStream_t>(0x11);
  ModelExecuteArg arg(stream);
  EXPECT_EQ(executor->Load(arg, load_arg), ge::SUCCESS);
  ge::DumpProperties dump_property;
  dump_property.InitInferOpDebug();  // open debug mode
  ge::DumpManager::GetInstance().AddDumpProperties(session_id, dump_property);
  ASSERT_EQ(dumper->OverflowDump(&add_kernel_launch_node, kModelStart), ge::SUCCESS);

  // Do overflow dump
  auto &kernel_launch_dump_units = dumper->kernel_idxes_to_dump_units_[add_kernel_launch_node.node_id];
  kernel_launch_dump_units.resize(1);
  *kernel_launch_dump_units[0] = node_add_dump_unit;
  kernel_launch_dump_units[0]->cur_update_count = 0;
  kernel_launch_dump_units[0]->total_update_count = 1;
  EXPECT_EQ(kernel_launch_dump_units[0]->need_overflow_dump, false);
  ge::AttrUtils::SetBool(kernel_launch_dump_units[0]->node->GetOpDesc(), ge::GLOBALWORKSPACE_TYPE, true);

  ExecutorDumper::OnExecuteEvent(SubExeGraphType::kMainExeGraph, dumper, kExecuteStart, &add_kernel_launch_node,
                                 kStatusSuccess);
  ExecutorDumper::OnExecuteEvent(SubExeGraphType::kMainExeGraph, dumper, kExecuteEnd, &add_kernel_launch_node,
                                 kStatusSuccess);
  ExecutorDumper::OnExecuteEvent(SubExeGraphType::kMainExeGraph, dumper, kModelEnd, &add_kernel_launch_node,
                                 kStatusSuccess);

  EXPECT_EQ(kernel_launch_dump_units[0]->need_overflow_dump, false);
  // check dump_op
  auto cpu_args = stub.GetRtsRuntimeStub().PopCpuLaunchArgsByKernelName("DumpDataInfo");
  EXPECT_EQ(cpu_args, nullptr);

  ge::diagnoseSwitch::DisableDumper();
}

TEST_F(ExecutorDumperUT, OverflowDump_NoOverflowDetected_StreamSyncTimeOut) {
  ge::diagnoseSwitch::EnableOverflowDump();

  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  ExecutorRun(executor.get());
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  StorageShape storage_shape{{2048}, {2048, 1}};
  kernel::BuildTensorAttr attr{kOnDeviceHbm, ge::DT_FLOAT16, {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ExpandDimsType()}};
  Tensor tensor_holder{storage_shape, attr.storage_format, attr.placement, attr.data_type, nullptr};
  tensor_holder.MutableTensorData() = TensorData{(void *)1024, nullptr, 0, kOnDeviceHbm};
  auto context_holder_1 =
      KernelRunContextFaker().KernelIONum(1, 2).Outputs({&tensor_holder, &tensor_holder.MutableTensorData()}).Build();
  auto &node_add_dump_unit = dumper->node_names_to_dump_units_["add1"];
  node_add_dump_unit.output_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  node_add_dump_unit.input_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  node_add_dump_unit.input_addrs[1] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  gert::StorageShape x2_shape = {{4, 8, 16, 1, 1}, {4, 8, 16, 1, 1, 1, 1}};
  gert::StorageShape x1_shape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
  auto context_holder_2 = KernelRunContextFaker()
                              .Inputs({&x1_shape})
                              .Outputs({&x2_shape})
                              .KernelIONum(1, 1)
                              .NodeIoNum(1, 1)
                              .IrInstanceNum({1})
                              .Build();
  node_add_dump_unit.output_shapes[0] = context_holder_2.GetContext<KernelContext>()->GetOutput(0);
  node_add_dump_unit.input_shapes[0] = const_cast<Chain *>(context_holder_2.GetContext<KernelContext>()->GetInput(0));
  node_add_dump_unit.input_shapes[1] = const_cast<Chain *>(context_holder_2.GetContext<KernelContext>()->GetInput(0));

  auto kernel_launch_context_holder = KernelRunContextFaker()
                                          .NodeName("add1")
                                          .KernelIONum(2, 1)
                                          .KernelType("LaunchKernelWithHandle")
                                          .KernelName("Add_LaunchKernelWithHandle")
                                          .Build();
  Node add_kernel_launch_node{"", 0, nullptr, *kernel_launch_context_holder.GetContext<KernelRunContext>()};
  for (auto &kernel_name_and_exe_node : dumper->kernel_names_to_exe_nodes_) {
    if (!FindKernelNameByStartKeyWord(kernel_name_and_exe_node.first, "LaunchKernelWithHandle").empty()) {
      add_kernel_launch_node.node_id = kernel_name_and_exe_node.second->node_id;
    }
  }
  // OverflowDumpLoadStage for saving stream and session id
  GertRuntimeStub stub;
  uint64_t session_id = 1U;
  RtSession rt_session(session_id);
  ModelLoadArg load_arg{&rt_session, {nullptr, 0}};
  const auto stream = reinterpret_cast<rtStream_t>(0x11);
  ModelExecuteArg arg(stream);
  EXPECT_EQ(executor->Load(arg, load_arg), ge::SUCCESS);
  ge::DumpProperties dump_property;
  dump_property.InitInferOpDebug();  // open debug mode
  ge::DumpManager::GetInstance().AddDumpProperties(session_id, dump_property);
  ASSERT_EQ(dumper->OverflowDump(&add_kernel_launch_node, kModelStart), ge::SUCCESS);

  // Do overflow dump
  auto &kernel_launch_dump_units = dumper->kernel_idxes_to_dump_units_[add_kernel_launch_node.node_id];
  kernel_launch_dump_units.resize(1);
  *kernel_launch_dump_units[0] = node_add_dump_unit;
  kernel_launch_dump_units[0]->cur_update_count = 0;
  kernel_launch_dump_units[0]->total_update_count = 1;
  EXPECT_EQ(kernel_launch_dump_units[0]->need_overflow_dump, false);
  ge::AttrUtils::SetBool(kernel_launch_dump_units[0]->node->GetOpDesc(), ge::GLOBALWORKSPACE_TYPE, true);
  mmSetEnv("TIMEOUT", "timeout", 1);  // for rtStreamSynchronizeWithTimeout stub
  ExecutorDumper::OnExecuteEvent(SubExeGraphType::kMainExeGraph, dumper, kExecuteStart, &add_kernel_launch_node,
                                 kStatusSuccess);
  ExecutorDumper::OnExecuteEvent(SubExeGraphType::kMainExeGraph, dumper, kExecuteEnd, &add_kernel_launch_node,
                                 kStatusSuccess);
  ExecutorDumper::OnExecuteEvent(SubExeGraphType::kMainExeGraph, dumper, kModelEnd, &add_kernel_launch_node,
                                 kStatusSuccess);
  unsetenv("TIMEOUT");

  EXPECT_EQ(kernel_launch_dump_units[0]->need_overflow_dump, false);
  // check dump_op
  auto cpu_args = stub.GetRtsRuntimeStub().PopCpuLaunchArgsByKernelName("DumpDataInfo");
  EXPECT_EQ(cpu_args, nullptr);

  ge::diagnoseSwitch::DisableDumper();
}
/**
 * 用例描述：ffts+算子的溢出dump
 *
 * 预置条件：
 * 1. ffts+算子的dumpunit
 *
 * 测试步骤：
 * 1. 构造ffts+算子的dumpunit
 * 2. 构造executorDump类
 * 3. 使能溢出dump
 *
 * 预期结果：
 * 1. 执行成功
 */
TEST_F(ExecutorDumperUT, OverflowDump_FFTSPlus_Ok) {
  ge::DumpConfig dump_config;
  dump_config.dump_debug = "on";
  dump_config.dump_path = "./";
  ge::DumpManager::GetInstance().SetDumpConf(dump_config);

  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  ExecutorRun(executor.get());
  uint64_t session_id = 1U;
  RtSession rt_session(session_id);
  ModelLoadArg load_arg{&rt_session, {nullptr, 0}};
  const auto stream = reinterpret_cast<rtStream_t>(0x11);
  ModelExecuteArg arg(stream);
  EXPECT_EQ(executor->Load(arg, load_arg), ge::SUCCESS);

  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  uint64_t id = FindFirstNonEmptyId(dumper);

  kernel::AICoreThreadParam thread_param;
  uint32_t thread_dim = 2U;
  auto out_type = ContinuousVector::Create<uint32_t>(1);
  auto out_type_vec = reinterpret_cast<ContinuousVector *>(out_type.get());
  out_type_vec->SetSize(1);
  auto out_type_ptr = reinterpret_cast<uint32_t *>(out_type_vec->MutableData());
  out_type_ptr[0] = 0;
  auto context_holder_1 = KernelRunContextFaker()
                              .NodeName(std::move("add1"))
                              .KernelIONum(static_cast<size_t>(kernel::ArgsInKey::kNUM), 0)
                              .KernelType("FFTSUpdateAICoreArgs")
                              .KernelName("FFTSUpdateAICoreArgs")
                              .Build();
  context_holder_1.value_holder[static_cast<size_t>(kernel::ArgsInKey::THREAD_PARAM)].Set(&thread_param, nullptr);
  context_holder_1.value_holder[static_cast<size_t>(kernel::ArgsInKey::IN_MEM_TYPE)].Set(out_type_vec, nullptr);
  context_holder_1.value_holder[static_cast<size_t>(kernel::ArgsInKey::OUT_MEM_TYPE)].Set(out_type_vec, nullptr);
  context_holder_1.value_holder[static_cast<size_t>(kernel::ArgsInKey::THREAD_DIM)].Set(reinterpret_cast<void *>(thread_dim), nullptr);
  size_t size = sizeof(Node) + sizeof(AsyncAnyValue *) * 6;
  Node *launch_node_1 = (Node *)malloc(size);
  launch_node_1->node_id = id;
  memcpy(&launch_node_1->context, context_holder_1.context, sizeof(KernelRunContext) + 6 * sizeof(AsyncAnyValue *));
  launch_node_1->func = nullptr;

  rtStream_t stream_ = reinterpret_cast<void *>(0x12);
  NodeMemPara node_para;
  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(10);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t);
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *pre_data_ptr = reinterpret_cast<TransTaskInfo *>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  pre_data_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  pre_data_ptr->rt_task_info.descBufLen = descBufLen;
  pre_data_ptr->rt_task_info.descBuf = holder.get() + sizeof(TransTaskInfo);
  node_para.host_addr = pre_data_ptr;
  node_para.dev_addr = pre_data_ptr;
  auto context_holder = KernelRunContextFaker()
                            .NodeName(std::move("add1"))
                            .KernelIONum(2, 0)
                            .Inputs({reinterpret_cast<void *>(stream_), &node_para})
                            .KernelType("LaunchFFTSPlusTask")
                            .KernelName("LaunchFFTSPlusTask")
                            .Build();
  size = sizeof(Node) + sizeof(AsyncAnyValue *) * 2;
  Node *launch_node = (Node *)malloc(size);
  launch_node->node_id = id;
  memcpy(&launch_node->context, context_holder.context, sizeof(KernelRunContext) + 2 * sizeof(AsyncAnyValue *));
  launch_node->func = nullptr;

  // dynamic dump ffts_plus
  EXPECT_EQ(dumper->UpdateFftsplusLaunchTask(launch_node), ge::SUCCESS);
  ASSERT_EQ(dumper->OnUpdateDumpUnit(kExecuteStart, *launch_node, true), ge::SUCCESS);

  auto ctx_ids = ContinuousVector::Create<int32_t>(4);
  auto ctx_ids_vec = reinterpret_cast<ContinuousVector *>(ctx_ids.get());
  ctx_ids_vec->SetSize(4);
  auto ctx_ids_ptr = reinterpret_cast<int32_t *>(ctx_ids_vec->MutableData());
  for (size_t i = 0U; i < ctx_ids_vec->GetSize(); i++) {
    ctx_ids_ptr[i] = i;
  }
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo *>(holder.get());
  // buf_offset = sizeof(rtFftsPlusComCtx_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  const size_t args_size = sizeof(rtFftsPlusComCtx_t);
  auto *buff_ptr = &task_info_ptr->args[args_size];
  for (int i = 0; i < 4; ++i) {
    auto context = reinterpret_cast<rtFftsPlusAicAivCtx_t *>(buff_ptr);
    context->contextType = RT_CTX_TYPE_AICORE;
    context->threadDim = 1U;
    buff_ptr += sizeof(rtFftsPlusComCtx_t);
  }
  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[args_size];
  ffts_plus_sqe->totalContextNum = 16;

  auto context_holder_2 = KernelRunContextFaker()
                              .NodeName(std::move("add1"))
                              .KernelIONum(15, 0)
                              .KernelType("AICoreUpdateContext")
                              .KernelName("AICoreUpdateContext")
                              .Build();
  context_holder_2.value_holder[1].Set(ctx_ids_vec, nullptr);
  context_holder_2.value_holder[14].Set(&task_inf, nullptr);
  size = sizeof(Node) + sizeof(AsyncAnyValue *) * 15;
  Node *launch_node_2 = (Node *)malloc(size);
  launch_node_2->node_id = id;
  memcpy(&launch_node_2->context, context_holder_2.context, sizeof(KernelRunContext) + 15 * sizeof(AsyncAnyValue *));
  launch_node_2->func = nullptr;

  // dynamic dump ffts_plus
  ASSERT_EQ(dumper->OverflowDump(launch_node_1, kExecuteStart), ge::SUCCESS);
  ASSERT_EQ(dumper->OverflowDump(launch_node_2, kModelStart), ge::SUCCESS);
  ASSERT_EQ(dumper->OverflowDump(launch_node_2, kExecuteEnd), ge::SUCCESS);
  ASSERT_EQ(dumper->OverflowDump(launch_node_2, kModelEnd), ge::SUCCESS);
  // buf_offset = sizeof(rtFftsPlusComCtx_t);
  auto *buff_ptr_1 = &task_info_ptr->args[args_size];
  for (int i = 0; i < 4; ++i) {
    auto context = reinterpret_cast<rtFftsPlusAiCpuCtx_t *>(buff_ptr_1);
    context->contextType = RT_CTX_TYPE_AICPU;
    context->threadDim = 1U;
    buff_ptr_1 += sizeof(rtFftsPlusComCtx_t);
  }
  task_inf.descBuf = &task_info_ptr->args[args_size];

  auto context_holder_3 = KernelRunContextFaker()
                              .NodeName(std::move("add1"))
                              .KernelIONum(15, 0)
                              .KernelType("AICpuUpdateContext")
                              .KernelName("AICpuUpdateContext")
                              .Build();
  context_holder_3.value_holder[1].Set(ctx_ids_vec, nullptr);
  context_holder_3.value_holder[14].Set(&task_inf, nullptr);
  size = sizeof(Node) + sizeof(AsyncAnyValue *) * 15;
  Node *launch_node_3 = (Node *)malloc(size);
  launch_node_3->node_id = id;
  memcpy(&launch_node_3->context, context_holder_3.context, sizeof(KernelRunContext) + 15 * sizeof(AsyncAnyValue *));
  launch_node_3->func = nullptr;

  auto &dump_unit = dumper->node_names_to_dump_units_["add1"];
  StorageShape storage_shape{{4}, {4, 1}};
  kernel::BuildTensorAttr attr{kOnDeviceHbm, ge::DT_FLOAT16, {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ExpandDimsType()}};
  Tensor tensor_holder{storage_shape, attr.storage_format, attr.placement, attr.data_type, nullptr};
  tensor_holder.MutableTensorData() = TensorData{(void *)1024, nullptr, 0, kOnDeviceHbm};
  auto context_holder_4 =
      KernelRunContextFaker().KernelIONum(1, 2).Outputs({&tensor_holder, &tensor_holder.MutableTensorData()}).Build();
  dump_unit.output_addrs[0] = context_holder_4.GetContext<KernelContext>()->GetOutput(1);
  dump_unit.input_addrs[0] = context_holder_4.GetContext<KernelContext>()->GetOutput(1);
  dump_unit.input_addrs[1] = context_holder_4.GetContext<KernelContext>()->GetOutput(1);
  gert::StorageShape x2_shape = {{4, 8, 16, 32, 64}, {4, 8, 16, 4, 2, 16, 16}};
  gert::StorageShape x1_shape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
  auto context_holder_5 = KernelRunContextFaker()
                              .Inputs({&x1_shape})
                              .Outputs({&x2_shape})
                              .KernelIONum(1, 1)
                              .NodeIoNum(1, 1)
                              .IrInstanceNum({1})
                              .Build();
  dump_unit.output_shapes[0] = context_holder_5.GetContext<KernelContext>()->GetOutput(0);
  dump_unit.input_shapes[0] = const_cast<Chain *>(context_holder_5.GetContext<KernelContext>()->GetInput(0));
  dump_unit.input_shapes[1] = const_cast<Chain *>(context_holder_5.GetContext<KernelContext>()->GetInput(0));

  ge::Context dump_context;
  dump_context.context_id = 0;
  dump_context.thread_id = 0;
  dump_unit.context_list.emplace_back(dump_context);

  auto properties = ge::DumpManager::GetInstance().GetDumpProperties(ge::kInferSessionId);
  EXPECT_EQ(dumper->DoDataDump(dump_unit, properties), ge::SUCCESS);

  // dynamic dump ffts_plus
  EXPECT_EQ(dumper->UpdateFftsplusLaunchTask(launch_node), ge::SUCCESS);

  free(launch_node);
  free(launch_node_1);
  free(launch_node_2);
  free(launch_node_3);
  ge::diagnoseSwitch::DisableDumper();

  properties.ClearOpDebugFlag();
  EXPECT_EQ(dumper->DoDataDump(dump_unit, properties), ge::SUCCESS);
  ge::DumpManager::GetInstance().RemoveDumpProperties(0);
}

TEST_F(ExecutorDumperUT, DoHcclDataDump_Ok) {
  StorageShape storage_shape{{4}, {4, 1}};
  kernel::BuildTensorAttr attr{kOnDeviceHbm, ge::DT_FLOAT16, {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ExpandDimsType()}};
  Tensor tensor_holder{storage_shape, attr.storage_format, attr.placement, attr.data_type, nullptr};
  tensor_holder.MutableTensorData() = TensorData{(void *)1024, nullptr, 0, kOnDeviceHbm};
  auto context_holder_1 =
      KernelRunContextFaker().KernelIONum(1, 2).Outputs({&tensor_holder, &tensor_holder.MutableTensorData()}).Build();
  const auto &sub_extend_info = ge::MakeShared<const SubscriberExtendInfo>();
  GlobalDumper::GetInstance()->SetEnableFlags(20);
  ExecutorDumper dumper(sub_extend_info);
  auto &node_add_dump_unit = dumper.node_names_to_dump_units_["hcom_reduce"];
  ASSERT_NE(context_holder_1.GetContext<KernelContext>(), nullptr);

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("HcomReduce", "HcomReduce");
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node_info = graph->AddNode(op_desc);
  node_add_dump_unit.node = node_info;
  node_add_dump_unit.output_addrs.push_back(nullptr);
  node_add_dump_unit.input_addrs.push_back(nullptr);
  node_add_dump_unit.output_shapes.push_back(nullptr);
  node_add_dump_unit.input_shapes.push_back(nullptr);
  node_add_dump_unit.output_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  node_add_dump_unit.input_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  gert::StorageShape x2_shape = {{4, 8, 16, 32, 64}, {4, 8, 16, 4, 2, 16, 16}};
  gert::StorageShape x1_shape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
  auto context_holder_2 = KernelRunContextFaker()
                              .Inputs({&x1_shape})
                              .Outputs({&x2_shape})
                              .KernelIONum(1, 1)
                              .NodeIoNum(1, 1)
                              .IrInstanceNum({1})
                              .Build();
  node_add_dump_unit.output_shapes[0] = context_holder_2.GetContext<KernelContext>()->GetOutput(0);
  node_add_dump_unit.input_shapes[0] = const_cast<Chain *>(context_holder_2.GetContext<KernelContext>()->GetInput(0));
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);

  Node node;
  size_t total_size = 0UL;
  ComputeNodeInfo::CalcSize(1, 1, 1, total_size);
  auto compute_node_info_holder = ge::ComGraphMakeUnique<uint8_t[]>(total_size);
  ComputeNodeInfo *compute_node_info = ge::PtrToPtr<uint8_t, ComputeNodeInfo>(compute_node_info_holder.get());
  compute_node_info->Init(1, 1, 1, "hcom_reduce", "HcomReduce");
  uint8_t holder[sizeof(KernelExtendInfo)];
  KernelExtendInfo *extend_kernel_info = reinterpret_cast<KernelExtendInfo *>(holder);
  extend_kernel_info->SetKernelName("hcom_kernel_launch");
  extend_kernel_info->SetKernelType("LaunchHcomKernel");
  KernelRunContext kernel_run_context;
  kernel_run_context.input_size = 1;
  kernel_run_context.output_size = 1;
  kernel_run_context.compute_node_info = static_cast<void *>(compute_node_info);
  kernel_run_context.kernel_extend_info = static_cast<void *>(extend_kernel_info);
  node.context = kernel_run_context;

  const auto properties = ge::DumpManager::GetInstance().GetDumpProperties(ge::kInferSessionId);
  // dynamic dump hccl input
  EXPECT_EQ(dumper.DataDump(&node, kExecuteStart), ge::SUCCESS);
  // dynamic dump hccl output
  EXPECT_EQ(dumper.DataDump(&node, kExecuteEnd), ge::SUCCESS);
}

TEST_F(ExecutorDumperUT, DoFftsplusDataDump_Ok) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  uint64_t id = FindFirstNonEmptyId(dumper);

  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);

  rtStream_t stream_ = reinterpret_cast<void *>(0x12);
  NodeMemPara node_para;
  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(10);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t);
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *pre_data_ptr = reinterpret_cast<TransTaskInfo *>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  pre_data_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  pre_data_ptr->rt_task_info.descBufLen = descBufLen;
  pre_data_ptr->rt_task_info.descBuf = holder.get() + sizeof(TransTaskInfo);
  node_para.host_addr = pre_data_ptr;
  node_para.dev_addr = pre_data_ptr;
  auto context_holder = KernelRunContextFaker()
                            .NodeName(std::move("add1"))
                            .KernelIONum(2, 0)
                            .Inputs({reinterpret_cast<void *>(stream_), &node_para})
                            .KernelType("LaunchFFTSPlusTask")
                            .KernelName("LaunchFFTSPlusTask")
                            .Build();
  size_t size = sizeof(Node) + sizeof(AsyncAnyValue *) * 2;
  Node *launch_node = (Node *)malloc(size);
  launch_node->node_id = id;
  memcpy(&launch_node->context, context_holder.context, sizeof(KernelRunContext) + 2 * sizeof(AsyncAnyValue *));
  launch_node->func = nullptr;

  // dynamic dump ffts_plus
  EXPECT_EQ(dumper->UpdateFftsplusLaunchTask(launch_node), ge::SUCCESS);
  free(launch_node);

  auto ctx_ids = ContinuousVector::Create<int32_t>(4);
  auto ctx_ids_vec = reinterpret_cast<ContinuousVector *>(ctx_ids.get());
  ctx_ids_vec->SetSize(4);
  auto ctx_ids_ptr = reinterpret_cast<int32_t *>(ctx_ids_vec->MutableData());
  for (size_t i = 0U; i < ctx_ids_vec->GetSize(); i++) {
    ctx_ids_ptr[i] = i;
  }
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo *>(holder.get());
  // buf_offset = sizeof(rtFftsPlusComCtx_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  const size_t args_size = sizeof(rtFftsPlusComCtx_t);
  auto *buff_ptr = &task_info_ptr->args[args_size];
  for (int i = 0; i < 4; ++i) {
    auto context = reinterpret_cast<rtFftsPlusAicAivCtx_t *>(buff_ptr);
    context->contextType = RT_CTX_TYPE_AICORE;
    context->threadDim = 1U;
    buff_ptr += sizeof(rtFftsPlusComCtx_t);
  }

  auto &dump_unit = dumper->node_names_to_dump_units_["add1"];
  StorageShape storage_shape{{4}, {4, 1}};
  kernel::BuildTensorAttr attr{kOnDeviceHbm, ge::DT_FLOAT16, {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ExpandDimsType()}};
  Tensor tensor_holder{storage_shape, attr.storage_format, attr.placement, attr.data_type, nullptr};
  tensor_holder.MutableTensorData() = TensorData{(void *)1024, nullptr, 0, kOnDeviceHbm};
  auto context_holder_4 =
      KernelRunContextFaker().KernelIONum(1, 2).Outputs({&tensor_holder, &tensor_holder.MutableTensorData()}).Build();
  dump_unit.output_addrs[0] = context_holder_4.GetContext<KernelContext>()->GetOutput(1);
  dump_unit.input_addrs[0] = context_holder_4.GetContext<KernelContext>()->GetOutput(1);
  dump_unit.input_addrs[1] = context_holder_4.GetContext<KernelContext>()->GetOutput(1);
  gert::StorageShape x2_shape = {{4, 8, 16, 32, 64}, {4, 8, 16, 4, 2, 16, 16}};
  gert::StorageShape x1_shape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
  auto context_holder_5 = KernelRunContextFaker()
                              .Inputs({&x1_shape})
                              .Outputs({&x2_shape})
                              .KernelIONum(1, 1)
                              .NodeIoNum(1, 1)
                              .IrInstanceNum({1})
                              .Build();
  dump_unit.output_shapes[0] = context_holder_5.GetContext<KernelContext>()->GetOutput(0);
  dump_unit.input_shapes[0] = const_cast<Chain *>(context_holder_5.GetContext<KernelContext>()->GetInput(0));
  dump_unit.input_shapes[1] = const_cast<Chain *>(context_holder_5.GetContext<KernelContext>()->GetInput(0));

  ge::Context dump_context;
  dump_context.context_id = 0;
  dump_context.thread_id = 0;
  dump_unit.context_list.emplace_back(dump_context);

  const auto properties = ge::DumpManager::GetInstance().GetDumpProperties(ge::kInferSessionId);
  EXPECT_EQ(dumper->DoDataDump(dump_unit, properties), ge::SUCCESS);
}

TEST_F(ExecutorDumperUT, DataDump_DoNotReload_WhenDavinciModelNull) {
  ge::diagnoseSwitch::DisableDumper();
  ge::DumpConfig dump_config;
  dump_config.dump_path = "/test";
  dump_config.dump_mode = "all";
  dump_config.dump_status = "on";
  dump_config.dump_op_switch = "on";
  (void)ge::DumpManager::GetInstance().SetDumpConf(dump_config);

  auto graph = ShareGraph::BuildWithKnownSubgraph();
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  GertRuntimeStub fakeRuntime;
  auto global_data = faker.FakeWithoutHandleAiCore("Conv2d", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  ge::diagnoseSwitch::EnableDataDump();
  auto dumper =
      model_executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  dumper->LoadDumpTaskForDavinciModels(true);
  dumper->LoadDumpTaskForDavinciModels(false);
  ge::DumpManager::GetInstance().RemoveDumpProperties(0);
  ge::diagnoseSwitch::DisableDumper();
}

TEST_F(ExecutorDumperUT, DataDump_ReloadTask_WhenDavinciModelNotNull) {
  ge::diagnoseSwitch::DisableDumper();
  ge::DumpConfig dump_config;
  dump_config.dump_path = "/test";
  dump_config.dump_mode = "all";
  dump_config.dump_status = "on";
  dump_config.dump_op_switch = "on";
  (void)ge::DumpManager::GetInstance().SetDumpConf(dump_config);

  auto graph = ShareGraph::BuildWithKnownSubgraph();
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  GertRuntimeStub fakeRuntime;
  auto global_data = faker.FakeWithoutHandleAiCore("Conv2d", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  ge::diagnoseSwitch::EnableDataDump();
  auto dumper =
      model_executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  auto model = ge::MakeUnique<ge::DavinciModel>(0, nullptr);
  ge::GeModelPtr ge_model = ge::MakeShared<ge::GeModel>();
  model->Assign(ge_model);
  auto execution_data =
      reinterpret_cast<const ExecutionData *>(model_executor->GetExeGraphExecutor(kInitExeGraph)->GetExecutionData());
  for (size_t i = 0; i < execution_data->base_ed.node_num; ++i) {
    auto node = execution_data->base_ed.nodes[i];
    auto kernel_context = reinterpret_cast<KernelContext *>(&node->context);
    const auto kernel_extend_info = static_cast<const KernelExtendInfo *>(kernel_context->GetKernelExtend());
    std::string kernel_type = kernel_extend_info->GetKernelType();
    if (kernel_type == "DavinciModelCreate") {
      kernel_context->GetOutput(0)->Set(model.get(), nullptr);
    }
  }
  dumper->LoadDumpTaskForDavinciModels(true);
  dumper->LoadDumpTaskForDavinciModels(false);
  ge::DumpManager::GetInstance().RemoveDumpProperties(0);
  ge::diagnoseSwitch::DisableDumper();
}

TEST_F(ExecutorDumperUT, ExceptionDump_PrepareExceptionDumpSuccess_WithEmptyAddrs) {
  SpaceRegistryFaker::SetefaultSpaceRegistryNull();
  ge::diagnoseSwitch::EnableExceptionDump();
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto graph = ShareGraph::BuildSingleNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub fakeRuntime;
  fakeRuntime.GetKernelStub().StubTiling();
  ge::GeRootModelPtr root_model = std::make_shared<ge::GeRootModel>();
  root_model->SetRootGraph(graph);
  ge::ModelData model_data{};
  model_data.om_name = "test";
  auto model_executor = ModelV2Executor::Create(exe_graph, model_data, root_model);
  auto dumper = model_executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  EXPECT_TRUE(dumper->is_inited_);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto mem_block = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors{{2048}, 2, mem_block.get()};

  ge::GeShape shape({1, 2, 3});
  ge::GeTensorDesc tensor_desc(shape);
  ge::GeTensorDesc tensor_desc_invalid(shape, ge::FORMAT_RESERVED, ge::DT_UNDEFINED);
  // 2 valid input, 1 invalid input, 1 output
  auto op_data = ge::OpDescBuilder("fake", "Fake")
                     .AddInput("x_0", tensor_desc)
                     .AddInput("x_1", tensor_desc)
                     .AddInput("x_2", tensor_desc_invalid)
                     .AddOutput("y", tensor_desc)
                     .Build();
  auto fake_node = graph->AddNode(op_data);
  auto executor = BuildExecutorFromSingleNodeForDump();
  auto &exception_dump_unit = dumper->node_names_to_extra_units_["fake"];
  exception_dump_unit.tiling_data = "-- tiling data --";
  exception_dump_unit.args = 100U;
  exception_dump_unit.args_before_execute = "args before execute: 0x000";
  exception_dump_unit.is_host_args = true;
  exception_dump_unit.tiling_key = 199U;
  exception_dump_unit.workspace_info.emplace_back(std::pair(20U, 24));
  exception_dump_unit.args_size = 1U;

  NodeDumpUnit dump_unit{};
  dump_unit.node = fake_node;
  dump_unit.input_addrs.resize(3, nullptr);
  Chain chain{};
  chain.Set(nullptr, nullptr);
  dump_unit.input_addrs[0] = &chain;
  dump_unit.output_addrs.resize(1, nullptr);
  dump_unit.input_shapes.resize(3, nullptr);
  dump_unit.output_shapes.resize(1, nullptr);
  ge::DumpStub::GetInstance().ClearOpInfos();

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  auto fake_node_1 = FakeNodeHelper::FakeNode("add1", "AicpuLaunchCCKernel", FindFirstNonEmptyId(dumper));
  EXPECT_EQ(dumper->PrepareExceptionDump(fake_node_1.node, "LaunchKernelWithHandle", dump_unit), ge::SUCCESS);
  const auto &dump_op_infos = ge::DumpStub::GetInstance().GetOpInfos();
  EXPECT_EQ(dump_op_infos.size(), 2U);
  EXPECT_EQ(dump_op_infos[0].tensorInfos.size(), 19U);
  EXPECT_EQ(dump_op_infos[0].tensorInfos[0].argsOffSet, std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(dump_op_infos[0].tensorInfos[2].argsOffSet, std::numeric_limits<uint32_t>::max());
  ge::DumpStub::GetInstance().ClearOpInfos();

  // all field should be cleared
  EXPECT_EQ(exception_dump_unit.tiling_data.empty(), true);
  EXPECT_EQ(exception_dump_unit.args_before_execute.empty(), true);
  EXPECT_EQ(exception_dump_unit.is_host_args, false);
  EXPECT_EQ(exception_dump_unit.workspace_info.empty(), true);
  EXPECT_EQ(exception_dump_unit.tiling_key, 0U);
  EXPECT_EQ(exception_dump_unit.args, 0U);
  EXPECT_EQ(exception_dump_unit.args_size, 0U);
  ge::diagnoseSwitch::DisableDumper();
  EXPECT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper),
            nullptr);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
  ge::diagnoseSwitch::DisableDumper();
}

TEST_F(ExecutorDumperUT, ExceptionDump_FillExtraInfoNoCore_WithEmptyComputeNodeInfo) {
  auto node = FakeNodeHelper::FakeNode("test", "test");
  node.context.context->compute_node_info = nullptr;
  ExecutorDumper dumper{nullptr};
  EXPECT_NO_THROW(dumper.FillExceptionDumpInfoByKernel(node.node));
}

TEST_F(ExecutorDumperUT, DoFftsplusExceptionDump_Ok) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  uint64_t id = FindFirstNonEmptyId(dumper);

  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);

  kernel::AICoreThreadParam thread_param;
  uint32_t thread_dim = 2U;
  auto out_type = ContinuousVector::Create<uint32_t>(1);
  auto out_type_vec = reinterpret_cast<ContinuousVector *>(out_type.get());
  out_type_vec->SetSize(1);
  auto out_type_ptr = reinterpret_cast<uint32_t *>(out_type_vec->MutableData());
  out_type_ptr[0] = 0;
  auto context_holder_1 = KernelRunContextFaker()
                              .NodeName(std::move("add1"))
                              .KernelIONum(static_cast<size_t>(kernel::ArgsInKey::kNUM), 0)
                              .KernelType("FFTSUpdateAICoreArgs")
                              .KernelName("FFTSUpdateAICoreArgs")
                              .Build();
  context_holder_1.value_holder[static_cast<size_t>(kernel::ArgsInKey::THREAD_PARAM)].Set(&thread_param, nullptr);
  context_holder_1.value_holder[static_cast<size_t>(kernel::ArgsInKey::IN_MEM_TYPE)].Set(out_type_vec, nullptr);
  context_holder_1.value_holder[static_cast<size_t>(kernel::ArgsInKey::OUT_MEM_TYPE)].Set(out_type_vec, nullptr);
  context_holder_1.value_holder[static_cast<size_t>(kernel::ArgsInKey::THREAD_DIM)].Set(reinterpret_cast<void *>(thread_dim), nullptr);
  size_t size = sizeof(Node) + sizeof(AsyncAnyValue *) * 6;
  Node *launch_node_1 = (Node *)malloc(size);
  launch_node_1->node_id = id;
  memcpy(&launch_node_1->context, context_holder_1.context, sizeof(KernelRunContext) + 6 * sizeof(AsyncAnyValue *));
  launch_node_1->func = nullptr;

  rtStream_t stream_ = reinterpret_cast<void *>(0x12);
  NodeMemPara node_para;
  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(10);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t);
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *pre_data_ptr = reinterpret_cast<TransTaskInfo *>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  pre_data_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  pre_data_ptr->rt_task_info.descBufLen = descBufLen;
  pre_data_ptr->rt_task_info.descBuf = holder.get() + sizeof(TransTaskInfo);
  node_para.host_addr = pre_data_ptr;
  node_para.dev_addr = pre_data_ptr;
  auto context_holder = KernelRunContextFaker()
                            .NodeName(std::move("add1"))
                            .KernelIONum(2, 0)
                            .Inputs({reinterpret_cast<void *>(stream_), &node_para})
                            .KernelType("LaunchFFTSPlusTask")
                            .KernelName("LaunchFFTSPlusTask")
                            .Build();
  size = sizeof(Node) + sizeof(AsyncAnyValue *) * 2;
  Node *launch_node = (Node *)malloc(size);
  launch_node->node_id = id;
  memcpy(&launch_node->context, context_holder.context, sizeof(KernelRunContext) + 2 * sizeof(AsyncAnyValue *));
  launch_node->func = nullptr;

  // dynamic dump ffts_plus
  EXPECT_EQ(dumper->UpdateFftsplusLaunchTask(launch_node), ge::SUCCESS);

  auto ctx_ids = ContinuousVector::Create<int32_t>(4);
  auto ctx_ids_vec = reinterpret_cast<ContinuousVector *>(ctx_ids.get());
  ctx_ids_vec->SetSize(4);
  auto ctx_ids_ptr = reinterpret_cast<int32_t *>(ctx_ids_vec->MutableData());
  for (size_t i = 0U; i < ctx_ids_vec->GetSize(); i++) {
    ctx_ids_ptr[i] = i;
  }
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo *>(holder.get());
  // buf_offset = sizeof(rtFftsPlusComCtx_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  const size_t args_size = sizeof(rtFftsPlusComCtx_t);
  auto *buff_ptr = &task_info_ptr->args[args_size];
  for (int i = 0; i < 4; ++i) {
    auto context = reinterpret_cast<rtFftsPlusAicAivCtx_t *>(buff_ptr);
    context->contextType = RT_CTX_TYPE_AICORE;
    context->threadDim = 1U;
    buff_ptr += sizeof(rtFftsPlusComCtx_t);
  }
  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[args_size];
  ffts_plus_sqe->totalContextNum = 16;

  auto context_holder_2 = KernelRunContextFaker()
                              .NodeName(std::move("add1"))
                              .KernelIONum(15, 0)
                              .KernelType("AICoreUpdateContext")
                              .KernelName("AICoreUpdateContext")
                              .Build();
  context_holder_2.value_holder[1].Set(ctx_ids_vec, nullptr);
  context_holder_2.value_holder[14].Set(&task_inf, nullptr);
  size = sizeof(Node) + sizeof(AsyncAnyValue *) * 15;
  Node *launch_node_2 = (Node *)malloc(size);
  launch_node_2->node_id = id;
  memcpy(&launch_node_2->context, context_holder_2.context, sizeof(KernelRunContext) + 15 * sizeof(AsyncAnyValue *));
  launch_node_2->func = nullptr;

  // buf_offset = sizeof(rtFftsPlusComCtx_t);
  auto *buff_ptr_1 = &task_info_ptr->args[args_size];
  for (int i = 0; i < 4; ++i) {
    auto context = reinterpret_cast<rtFftsPlusAiCpuCtx_t *>(buff_ptr_1);
    context->contextType = RT_CTX_TYPE_AICPU;
    context->threadDim = 1U;
    buff_ptr_1 += sizeof(rtFftsPlusComCtx_t);
  }
  task_inf.descBuf = &task_info_ptr->args[args_size];

  auto context_holder_3 = KernelRunContextFaker()
                              .NodeName(std::move("add1"))
                              .KernelIONum(15, 0)
                              .KernelType("FFTSUpdateAICoreArgs")
                              .KernelName("AICpuUpdateContext")
                              .Build();
  context_holder_3.value_holder[1].Set(ctx_ids_vec, nullptr);
  context_holder_3.value_holder[14].Set(&task_inf, nullptr);
  size = sizeof(Node) + sizeof(AsyncAnyValue *) * 15;
  Node *launch_node_3 = (Node *)malloc(size);
  launch_node_3->node_id = id;
  memcpy(&launch_node_3->context, context_holder_3.context, sizeof(KernelRunContext) + 15 * sizeof(AsyncAnyValue *));
  launch_node_3->func = nullptr;

  auto &dump_unit = dumper->node_names_to_dump_units_["add1"];
  StorageShape storage_shape{{4}, {4, 1}};
  kernel::BuildTensorAttr attr{kOnDeviceHbm, ge::DT_FLOAT16, {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ExpandDimsType()}};
  Tensor tensor_holder{storage_shape, attr.storage_format, attr.placement, attr.data_type, nullptr};
  tensor_holder.MutableTensorData() = TensorData{(void *)1024, nullptr, 0, kOnDeviceHbm};
  auto context_holder_4 =
      KernelRunContextFaker().KernelIONum(1, 2).Outputs({&tensor_holder, &tensor_holder.MutableTensorData()}).Build();
  dump_unit.output_addrs[0] = context_holder_4.GetContext<KernelContext>()->GetOutput(1);
  dump_unit.input_addrs[0] = context_holder_4.GetContext<KernelContext>()->GetOutput(1);
  dump_unit.input_addrs[1] = context_holder_4.GetContext<KernelContext>()->GetOutput(1);
  gert::StorageShape x2_shape = {{4, 8, 16, 32, 64}, {4, 8, 16, 4, 2, 16, 16}};
  gert::StorageShape x1_shape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
  auto context_holder_5 = KernelRunContextFaker()
                              .Inputs({&x1_shape})
                              .Outputs({&x2_shape})
                              .KernelIONum(1, 1)
                              .NodeIoNum(1, 1)
                              .IrInstanceNum({1})
                              .Build();
  dump_unit.output_shapes[0] = context_holder_5.GetContext<KernelContext>()->GetOutput(0);
  dump_unit.input_shapes[0] = const_cast<Chain *>(context_holder_5.GetContext<KernelContext>()->GetInput(0));
  dump_unit.input_shapes[1] = const_cast<Chain *>(context_holder_5.GetContext<KernelContext>()->GetInput(0));
  ge::Context dump_context;
  dump_context.context_id = 0;
  dump_context.thread_id = 0;
  ge::RealAddressAndSize addr = {0, 0};
  dump_context.input.emplace_back(addr);
  dump_context.input.emplace_back(addr);
  dump_context.output.emplace_back(addr);
  dump_unit.context_list.emplace_back(dump_context);

  vector<int64_t> vec;
  ge::AttrUtils::SetListInt(dump_unit.node->GetOpDesc(), "_context_id_list", vec);
  dumper->exe_node_id_to_data_dump_filler_[launch_node->node_id] = [](const KernelContext *, DataDumpInfoWrapper &) {
    return ge::SUCCESS;
  };
  dumper->FillExceptionDumpInfoByKernel(*launch_node);
  dumper->FillExceptionDumpInfoByKernel(*launch_node_1);
  dumper->FillExceptionDumpInfoByKernel(*launch_node_2);
  dumper->FillExceptionDumpInfoByKernel(*launch_node_3);

  free(launch_node);
  free(launch_node_1);
  free(launch_node_2);
  free(launch_node_3);
}

TEST_F(ExecutorDumperUT, DoFftsplusDataDump_new) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  uint64_t id = FindFirstNonEmptyId(dumper);

  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);

  uint64_t offset_vec[5] = {0};
  uint32_t thread_dim = 2U;
  auto thread_offset = ContinuousVector::Create<uint64_t>(2);
  auto thread_offset_vec = reinterpret_cast<ContinuousVector *>(thread_offset.get());
  thread_offset_vec->SetSize(2);
  auto thread_offset_vec_ptr = reinterpret_cast<uint64_t *>(thread_offset_vec->MutableData());
  thread_offset_vec_ptr[0] = reinterpret_cast<uint64_t>(&offset_vec[0]);
  thread_offset_vec_ptr[1] = reinterpret_cast<uint64_t>(&offset_vec[1]);
  auto context_holder_1 = KernelRunContextFaker()
                              .NodeName(std::move("add1"))
                              .KernelIONum(6, 0)
                              .KernelType("FFTSUpdateAutoAICoreArgs")
                              .KernelName("FFTSUpdateAutoAICoreArgs")
                              .Build();
  context_holder_1.value_holder[5].Set(thread_offset_vec, nullptr);
  context_holder_1.value_holder[3].Set(reinterpret_cast<void *>(thread_dim), nullptr);
  size_t size = sizeof(Node) + sizeof(AsyncAnyValue *) * 6;
  Node *launch_node_1 = (Node *)malloc(size);
  launch_node_1->node_id = id;
  memcpy(&launch_node_1->context, context_holder_1.context, sizeof(KernelRunContext) + 6 * sizeof(AsyncAnyValue *));
  launch_node_1->func = nullptr;

  rtStream_t stream_ = reinterpret_cast<void *>(0x12);
  NodeMemPara node_para;
  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(10);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t);
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *pre_data_ptr = reinterpret_cast<TransTaskInfo *>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  pre_data_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  pre_data_ptr->rt_task_info.descBufLen = descBufLen;
  pre_data_ptr->rt_task_info.descBuf = holder.get() + sizeof(TransTaskInfo);
  node_para.host_addr = pre_data_ptr;
  node_para.dev_addr = pre_data_ptr;
  auto context_holder = KernelRunContextFaker()
                            .NodeName(std::move("add1"))
                            .KernelIONum(2, 0)
                            .Inputs({reinterpret_cast<void *>(stream_), &node_para})
                            .KernelType("LaunchFFTSPlusTask")
                            .KernelName("LaunchFFTSPlusTask")
                            .Build();
  size = sizeof(Node) + sizeof(AsyncAnyValue *) * 2;
  Node *launch_node = (Node *)malloc(size);
  launch_node->node_id = id;
  memcpy(&launch_node->context, context_holder.context, sizeof(KernelRunContext) + 2 * sizeof(AsyncAnyValue *));
  launch_node->func = nullptr;

  // dynamic dump ffts_plus
  EXPECT_EQ(dumper->UpdateFftsplusLaunchTask(launch_node), ge::SUCCESS);

  auto ctx_ids = ContinuousVector::Create<int32_t>(4);
  auto ctx_ids_vec = reinterpret_cast<ContinuousVector *>(ctx_ids.get());
  ctx_ids_vec->SetSize(4);
  auto ctx_ids_ptr = reinterpret_cast<int32_t *>(ctx_ids_vec->MutableData());
  for (size_t i = 0U; i < ctx_ids_vec->GetSize(); i++) {
    ctx_ids_ptr[i] = i;
  }
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo *>(holder.get());
  // buf_offset = sizeof(rtFftsPlusComCtx_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  const size_t args_size = sizeof(rtFftsPlusComCtx_t);
  auto *buff_ptr = &task_info_ptr->args[args_size];
  for (int i = 0; i < 4; ++i) {
    auto context = reinterpret_cast<rtFftsPlusAicAivCtx_t *>(buff_ptr);
    context->contextType = RT_CTX_TYPE_AICORE;
    context->threadDim = 1U;
    buff_ptr += sizeof(rtFftsPlusComCtx_t);
  }
  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[args_size];
  ffts_plus_sqe->totalContextNum = 16;

  auto context_holder_2 = KernelRunContextFaker()
                              .NodeName(std::move("add1"))
                              .KernelIONum(15, 0)
                              .KernelType("StaAutoUpdateContext")
                              .KernelName("StaAutoUpdateContext")
                              .Build();
  context_holder_2.value_holder[3].Set(ctx_ids_vec, nullptr);
  context_holder_2.value_holder[1].Set(&task_inf, nullptr);
  size = sizeof(Node) + sizeof(AsyncAnyValue *) * 15;
  Node *launch_node_2 = (Node *)malloc(size);
  launch_node_2->node_id = id;
  memcpy(&launch_node_2->context, context_holder_2.context, sizeof(KernelRunContext) + 15 * sizeof(AsyncAnyValue *));
  launch_node_2->func = nullptr;

  // dynamic dump ffts_plus
  EXPECT_EQ(dumper->DataDump(launch_node_2, kModelStart), ge::SUCCESS);
  EXPECT_EQ(dumper->DataDump(launch_node_2, kExecuteEnd), ge::SUCCESS);
  EXPECT_EQ(dumper->DataDump(launch_node_2, kModelEnd), ge::SUCCESS);

  auto context_holder_3 = KernelRunContextFaker()
                              .KernelIONum(static_cast<size_t>(kernel::UpdateKey::RESERVED), 1)
                              .NodeName(std::move("add1"))
                              .NodeIoNum(2, 2)
                              .IrInputNum(2)
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                              .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                              .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                              .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                              .KernelType("AICoreUpdateContext")
                              .KernelName("AICoreUpdateContext")
                              .Build();
  Shape shape({800, 800, 1000});
  auto in_slice = ContinuousVector::Create<Shape>(12);
  auto in_slice_vec = reinterpret_cast<ContinuousVector *>(in_slice.get());
  in_slice_vec->SetSize(2);
  auto in_slice_ptr = reinterpret_cast<Shape *>(in_slice_vec->MutableData());
  for (size_t i = 0; i < in_slice_vec->GetSize(); i++) {
    in_slice_ptr[i] = shape;
  }

  auto out_slice = ContinuousVector::Create<Shape>(12);
  auto out_slice_vec = reinterpret_cast<ContinuousVector *>(out_slice.get());
  out_slice_vec->SetSize(2);
  auto out_slice_ptr = reinterpret_cast<Shape *>(out_slice_vec->MutableData());
  for (size_t i = 0; i < out_slice_vec->GetSize(); i++) {
    out_slice_ptr[i] = shape;
  }
  context_holder_3.value_holder[static_cast<size_t>(kernel::UpdateKey::LAST_IN_SLICE)].Set(in_slice_vec, nullptr);
  context_holder_3.value_holder[static_cast<size_t>(kernel::UpdateKey::LAST_OUT_SLICE)].Set(out_slice_vec, nullptr);
  size = sizeof(Node) + sizeof(AsyncAnyValue *) * 6;
  Node *launch_node_3 = (Node *)malloc(size);
  launch_node_3->node_id = id;
  memcpy(&launch_node_3->context, context_holder_3.context, sizeof(KernelRunContext) + 6 * sizeof(AsyncAnyValue *));
  launch_node_3->func = nullptr;

  free(launch_node);
  free(launch_node_1);
  free(launch_node_2);
  free(launch_node_3);
}

TEST_F(ExecutorDumperUT, DoFftsplusDataDumpWithOpRange_new) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  uint64_t id = FindFirstNonEmptyId(dumper);

  ge::DumpProperties dump_properties;
  std::vector<std::pair<std::string, std::string>> op_ranges = {{"add1", "add1"},};
  dump_properties.SetOpDumpRange("test", op_ranges);
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);

  uint64_t offset_vec[5] = {0};
  uint32_t thread_dim = 2U;
  auto thread_offset = ContinuousVector::Create<uint64_t>(2);
  auto thread_offset_vec = reinterpret_cast<ContinuousVector *>(thread_offset.get());
  thread_offset_vec->SetSize(2);
  auto thread_offset_vec_ptr = reinterpret_cast<uint64_t *>(thread_offset_vec->MutableData());
  thread_offset_vec_ptr[0] = reinterpret_cast<uint64_t>(&offset_vec[0]);
  thread_offset_vec_ptr[1] = reinterpret_cast<uint64_t>(&offset_vec[1]);
  auto context_holder_1 = KernelRunContextFaker()
                              .NodeName(std::move("add1"))
                              .KernelIONum(6, 0)
                              .KernelType("FFTSUpdateAutoAICoreArgs")
                              .KernelName("FFTSUpdateAutoAICoreArgs")
                              .Build();
  context_holder_1.value_holder[5].Set(thread_offset_vec, nullptr);
  context_holder_1.value_holder[3].Set(reinterpret_cast<void *>(thread_dim), nullptr);
  size_t size = sizeof(Node) + sizeof(AsyncAnyValue *) * 6;
  Node *launch_node_1 = (Node *)malloc(size);
  launch_node_1->node_id = id;
  memcpy(&launch_node_1->context, context_holder_1.context, sizeof(KernelRunContext) + 6 * sizeof(AsyncAnyValue *));
  launch_node_1->func = nullptr;

  rtStream_t stream_ = reinterpret_cast<void *>(0x12);
  NodeMemPara node_para;
  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(10);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t);
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *pre_data_ptr = reinterpret_cast<TransTaskInfo *>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  pre_data_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  pre_data_ptr->rt_task_info.descBufLen = descBufLen;
  pre_data_ptr->rt_task_info.descBuf = holder.get() + sizeof(TransTaskInfo);
  node_para.host_addr = pre_data_ptr;
  node_para.dev_addr = pre_data_ptr;
  auto context_holder = KernelRunContextFaker()
                            .NodeName(std::move("add1"))
                            .KernelIONum(2, 0)
                            .Inputs({reinterpret_cast<void *>(stream_), &node_para})
                            .KernelType("LaunchFFTSPlusTask")
                            .KernelName("LaunchFFTSPlusTask")
                            .Build();
  size = sizeof(Node) + sizeof(AsyncAnyValue *) * 2;
  Node *launch_node = (Node *)malloc(size);
  launch_node->node_id = id;
  memcpy(&launch_node->context, context_holder.context, sizeof(KernelRunContext) + 2 * sizeof(AsyncAnyValue *));
  launch_node->func = nullptr;

  // dynamic dump ffts_plus
  EXPECT_EQ(dumper->UpdateFftsplusLaunchTask(launch_node), ge::SUCCESS);

  // dynamic dump ffts_plus
  EXPECT_EQ(dumper->DataDump(launch_node, kModelStart), ge::SUCCESS);
  EXPECT_EQ(dumper->DataDump(launch_node, kExecuteStart), ge::SUCCESS);

  auto ctx_ids = ContinuousVector::Create<int32_t>(4);
  auto ctx_ids_vec = reinterpret_cast<ContinuousVector *>(ctx_ids.get());
  ctx_ids_vec->SetSize(4);
  auto ctx_ids_ptr = reinterpret_cast<int32_t *>(ctx_ids_vec->MutableData());
  for (size_t i = 0U; i < ctx_ids_vec->GetSize(); i++) {
    ctx_ids_ptr[i] = i;
  }
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo *>(holder.get());
  // buf_offset = sizeof(rtFftsPlusComCtx_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  const size_t args_size = sizeof(rtFftsPlusComCtx_t);
  auto *buff_ptr = &task_info_ptr->args[args_size];
  for (int i = 0; i < 4; ++i) {
    auto context = reinterpret_cast<rtFftsPlusAicAivCtx_t *>(buff_ptr);
    context->contextType = RT_CTX_TYPE_AICORE;
    context->threadDim = 1U;
    buff_ptr += sizeof(rtFftsPlusComCtx_t);
  }
  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[args_size];
  ffts_plus_sqe->totalContextNum = 16;

  auto context_holder_2 = KernelRunContextFaker()
                              .NodeName(std::move("add1"))
                              .KernelIONum(15, 0)
                              .KernelType("StaAutoUpdateContext")
                              .KernelName("StaAutoUpdateContext")
                              .Build();
  context_holder_2.value_holder[3].Set(ctx_ids_vec, nullptr);
  context_holder_2.value_holder[1].Set(&task_inf, nullptr);
  size = sizeof(Node) + sizeof(AsyncAnyValue *) * 15;
  Node *launch_node_2 = (Node *)malloc(size);
  launch_node_2->node_id = id;
  memcpy(&launch_node_2->context, context_holder_2.context, sizeof(KernelRunContext) + 15 * sizeof(AsyncAnyValue *));
  launch_node_2->func = nullptr;

  // dynamic dump ffts_plus
  EXPECT_EQ(dumper->DataDump(launch_node_2, kModelStart), ge::SUCCESS);
  EXPECT_EQ(dumper->DataDump(launch_node_2, kExecuteStart), ge::SUCCESS);
  EXPECT_EQ(dumper->DataDump(launch_node_2, kExecuteEnd), ge::SUCCESS);
  EXPECT_EQ(dumper->DataDump(launch_node_2, kModelEnd), ge::SUCCESS);

  auto context_holder_3 = KernelRunContextFaker()
                              .KernelIONum(static_cast<size_t>(kernel::UpdateKey::RESERVED), 1)
                              .NodeName(std::move("add1"))
                              .NodeIoNum(2, 2)
                              .IrInputNum(2)
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                              .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                              .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                              .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                              .KernelType("AICoreUpdateContext")
                              .KernelName("AICoreUpdateContext")
                              .Build();
  Shape shape({800, 800, 1000});
  auto in_slice = ContinuousVector::Create<Shape>(12);
  auto in_slice_vec = reinterpret_cast<ContinuousVector *>(in_slice.get());
  in_slice_vec->SetSize(2);
  auto in_slice_ptr = reinterpret_cast<Shape *>(in_slice_vec->MutableData());
  for (size_t i = 0; i < in_slice_vec->GetSize(); i++) {
    in_slice_ptr[i] = shape;
  }

  auto out_slice = ContinuousVector::Create<Shape>(12);
  auto out_slice_vec = reinterpret_cast<ContinuousVector *>(out_slice.get());
  out_slice_vec->SetSize(2);
  auto out_slice_ptr = reinterpret_cast<Shape *>(out_slice_vec->MutableData());
  for (size_t i = 0; i < out_slice_vec->GetSize(); i++) {
    out_slice_ptr[i] = shape;
  }
  context_holder_3.value_holder[static_cast<size_t>(kernel::UpdateKey::LAST_IN_SLICE)].Set(in_slice_vec, nullptr);
  context_holder_3.value_holder[static_cast<size_t>(kernel::UpdateKey::LAST_OUT_SLICE)].Set(out_slice_vec, nullptr);
  size = sizeof(Node) + sizeof(AsyncAnyValue *) * 6;
  Node *launch_node_3 = (Node *)malloc(size);
  launch_node_3->node_id = id;
  memcpy(&launch_node_3->context, context_holder_3.context, sizeof(KernelRunContext) + 6 * sizeof(AsyncAnyValue *));
  launch_node_3->func = nullptr;

  free(launch_node);
  free(launch_node_1);
  free(launch_node_2);
  free(launch_node_3);
}

TEST_F(ExecutorDumperUT, SetAicoreFftsplusDataDumpFlag) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  uint64_t id = FindFirstNonEmptyId(dumper);

  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);

  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(10);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t);
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  auto ctx_ids = ContinuousVector::Create<int32_t>(4);
  auto ctx_ids_vec = reinterpret_cast<ContinuousVector *>(ctx_ids.get());
  ctx_ids_vec->SetSize(4);
  auto ctx_ids_ptr = reinterpret_cast<int32_t *>(ctx_ids_vec->MutableData());
  for (size_t i = 0U; i < ctx_ids_vec->GetSize(); i++) {
    ctx_ids_ptr[i] = i;
  }
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo *>(holder.get());
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  const size_t args_size = sizeof(rtFftsPlusComCtx_t);
  auto *buff_ptr = &task_info_ptr->args[args_size];
  for (int i = 0; i < 4; ++i) {
    buff_ptr += sizeof(rtFftsPlusComCtx_t);
  }
  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[args_size];
  ffts_plus_sqe->totalContextNum = 16;

  auto context_holder = KernelRunContextFaker()
                            .NodeName(std::move("add1"))
                            .KernelIONum(15, 0)
                            .KernelType("AICoreUpdateContext")
                            .KernelName("AICoreUpdateContext")
                            .Build();
  context_holder.value_holder[static_cast<size_t>(kernel::UpdateKey::AICORE_CTX)].Set(ctx_ids_vec, nullptr);
  context_holder.value_holder[static_cast<size_t>(kernel::UpdateKey::TASK_INFO)].Set(&task_inf, nullptr);
  size_t size = sizeof(Node) + sizeof(AsyncAnyValue *) * 15;
  Node *launch_node = (Node *)malloc(size);
  launch_node->node_id = id;
  memcpy(&launch_node->context, context_holder.context, sizeof(KernelRunContext) + 15 * sizeof(AsyncAnyValue *));
  launch_node->func = nullptr;
  EXPECT_EQ(dumper->DataDump(launch_node, kExecuteEnd), ge::SUCCESS);
  free(launch_node);
}

TEST_F(ExecutorDumperUT, SetAicpuFftsplusDataDumpFlag) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  uint64_t id = FindFirstNonEmptyId(dumper);

  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);

  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(10);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t);
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  auto ctx_ids = ContinuousVector::Create<int32_t>(4);
  auto ctx_ids_vec = reinterpret_cast<ContinuousVector *>(ctx_ids.get());
  ctx_ids_vec->SetSize(4);
  auto ctx_ids_ptr = reinterpret_cast<int32_t *>(ctx_ids_vec->MutableData());
  for (size_t i = 0U; i < ctx_ids_vec->GetSize(); i++) {
    ctx_ids_ptr[i] = i;
  }
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo *>(holder.get());
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  const size_t args_size = sizeof(rtFftsPlusComCtx_t);
  auto *buff_ptr = &task_info_ptr->args[args_size];
  for (int i = 0; i < 4; ++i) {
    buff_ptr += sizeof(rtFftsPlusComCtx_t);
  }
  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[args_size];
  ffts_plus_sqe->totalContextNum = 16;

  auto context_holder = KernelRunContextFaker()
                            .NodeName(std::move("add1"))
                            .KernelIONum(15, 1)
                            .KernelType("AICpuUpdateContext")
                            .KernelName("AICpuUpdateContext")
                            .Build();
  context_holder.value_holder[static_cast<size_t>(kernel::UpdateContextInputIndex::kCtxIds)].Set(ctx_ids_vec, nullptr);
  context_holder.value_holder[15U].Set(&task_inf, nullptr);
  size_t size = sizeof(Node) + sizeof(AsyncAnyValue *) * 15;
  Node *launch_node = (Node *)malloc(size);
  launch_node->node_id = id;
  memcpy(&launch_node->context, context_holder.context, sizeof(KernelRunContext) + 15 * sizeof(AsyncAnyValue *));
  launch_node->func = nullptr;
  EXPECT_EQ(dumper->DataDump(launch_node, kExecuteEnd), ge::SUCCESS);
  EXPECT_EQ(dumper->SetDumpFlagForFfts("", nullptr), ge::SUCCESS);
  free(launch_node);
}

TEST_F(ExecutorDumperUT, SetMixl2DataDumpFlag) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  uint64_t id = FindFirstNonEmptyId(dumper);

  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);

  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(10);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t);
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo *>(holder.get());
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  rtFftsPlusTaskInfo_t task_inf;
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[sizeof(rtFftsPlusComCtx_t)];
  ffts_plus_sqe->totalContextNum = 16;

  auto context_holder = KernelRunContextFaker()
                            .NodeName(std::move("add1"))
                            .KernelIONum(15, 0)
                            .KernelType("MixL2UpdateContext")
                            .KernelName("MixL2UpdateContext")
                            .Build();
  context_holder.value_holder[static_cast<size_t>(kernel::MixL2UpdateKey::CTX_ID)].Set(0, nullptr);
  context_holder.value_holder[static_cast<size_t>(kernel::MixL2UpdateKey::TASK_INFO)].Set(&task_inf, nullptr);
  size_t size = sizeof(Node) + sizeof(AsyncAnyValue *) * 15;
  Node *launch_node = (Node *)malloc(size);
  launch_node->node_id = id;
  memcpy(&launch_node->context, context_holder.context, sizeof(KernelRunContext) + 15 * sizeof(AsyncAnyValue *));
  launch_node->func = nullptr;
  EXPECT_EQ(dumper->DataDump(launch_node, kExecuteEnd), ge::SUCCESS);
  free(launch_node);
}

TEST_F(ExecutorDumperUT, CheckFftsplusDataDump_PostSync) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  ExecutorRun(executor.get());
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  uint64_t id = FindFirstNonEmptyId(dumper);

  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);

  rtStream_t stream_ = reinterpret_cast<void *>(0x12);
  NodeMemPara node_para;
  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(10);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t);
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *pre_data_ptr = reinterpret_cast<TransTaskInfo *>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  pre_data_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  pre_data_ptr->rt_task_info.descBufLen = descBufLen;
  pre_data_ptr->rt_task_info.descBuf = holder.get() + sizeof(TransTaskInfo);
  node_para.host_addr = pre_data_ptr;
  node_para.dev_addr = pre_data_ptr;
  auto context_holder = KernelRunContextFaker()
                            .NodeName(std::move("add1"))
                            .KernelIONum(2, 0)
                            .Inputs({reinterpret_cast<void *>(stream_), &node_para})
                            .KernelType("LaunchFFTSPlusTask")
                            .KernelName("LaunchFFTSPlusTask")
                            .Build();
  size_t size = sizeof(Node) + sizeof(AsyncAnyValue *) * 2;
  Node *launch_node = (Node *)malloc(size);
  launch_node->node_id = id;
  memcpy(&launch_node->context, context_holder.context, sizeof(KernelRunContext) + 2 * sizeof(AsyncAnyValue *));
  launch_node->func = nullptr;

  // dynamic dump ffts_plus
  EXPECT_EQ(dumper->DataDump(launch_node, kModelStart), ge::SUCCESS);
  EXPECT_EQ(dumper->DataDump(launch_node, kExecuteStart), ge::SUCCESS);
  EXPECT_EQ(dumper->DataDump(launch_node, kExecuteEnd), ge::SUCCESS);
  EXPECT_EQ(dumper->DataDump(launch_node, kModelEnd), ge::SUCCESS);
  free(launch_node);
}

TEST_F(ExecutorDumperUT, OverflowDump_FFTSPlus_Ok_new) {
  ge::DumpConfig dump_config;
  dump_config.dump_debug = "on";
  dump_config.dump_path = "./";
  ge::DumpManager::GetInstance().SetDumpConf(dump_config);

  auto executor = BuildExecutorTraningTrace();
  ASSERT_NE(executor, nullptr);
  ExecutorRun(executor.get());
  uint64_t session_id = 1U;
  RtSession rt_session(session_id);
  ModelLoadArg load_arg{&rt_session, {nullptr, 0}};
  const auto stream = reinterpret_cast<rtStream_t>(0x11);
  ModelExecuteArg arg(stream);
  EXPECT_EQ(executor->Load(arg, load_arg), ge::SUCCESS);

  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  uint64_t id = FindFirstNonEmptyId(dumper);

  kernel::AICoreThreadParam thread_param;
  uint32_t thread_dim = 2U;
  auto out_type = ContinuousVector::Create<uint32_t>(1);
  auto out_type_vec = reinterpret_cast<ContinuousVector *>(out_type.get());
  out_type_vec->SetSize(1);
  auto out_type_ptr = reinterpret_cast<uint32_t *>(out_type_vec->MutableData());
  out_type_ptr[0] = 0;
  auto context_holder_1 = KernelRunContextFaker()
                              .NodeName(std::move("add1"))
                              .KernelIONum(static_cast<size_t>(kernel::ArgsInKey::kNUM), 0)
                              .KernelType("FFTSUpdateAICoreArgs")
                              .KernelName("FFTSUpdateAICoreArgs")
                              .Build();
  context_holder_1.value_holder[static_cast<size_t>(kernel::ArgsInKey::THREAD_PARAM)].Set(&thread_param, nullptr);
  context_holder_1.value_holder[static_cast<size_t>(kernel::ArgsInKey::IN_MEM_TYPE)].Set(out_type_vec, nullptr);
  context_holder_1.value_holder[static_cast<size_t>(kernel::ArgsInKey::OUT_MEM_TYPE)].Set(out_type_vec, nullptr);
  context_holder_1.value_holder[static_cast<size_t>(kernel::ArgsInKey::THREAD_DIM)].Set(reinterpret_cast<void *>(thread_dim), nullptr);
  size_t size = sizeof(Node) + sizeof(AsyncAnyValue *) * 6;
  Node *launch_node_1 = (Node *)malloc(size);
  launch_node_1->node_id = id;
  memcpy(&launch_node_1->context, context_holder_1.context, sizeof(KernelRunContext) + 6 * sizeof(AsyncAnyValue *));
  launch_node_1->func = nullptr;

  rtStream_t stream_ = reinterpret_cast<void *>(0x12);
  NodeMemPara node_para;
  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(10);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t);
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *pre_data_ptr = reinterpret_cast<TransTaskInfo *>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  pre_data_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  pre_data_ptr->rt_task_info.descBufLen = descBufLen;
  pre_data_ptr->rt_task_info.descBuf = holder.get() + sizeof(TransTaskInfo);
  node_para.host_addr = pre_data_ptr;
  node_para.dev_addr = pre_data_ptr;
  auto context_holder = KernelRunContextFaker()
                            .NodeName(std::move("add1"))
                            .KernelIONum(2, 0)
                            .Inputs({reinterpret_cast<void *>(stream_), &node_para})
                            .KernelType("LaunchFFTSPlusTask")
                            .KernelName("LaunchFFTSPlusTask")
                            .Build();
  size = sizeof(Node) + sizeof(AsyncAnyValue *) * 2;
  Node *launch_node = (Node *)malloc(size);
  launch_node->node_id = id;
  memcpy(&launch_node->context, context_holder.context, sizeof(KernelRunContext) + 2 * sizeof(AsyncAnyValue *));
  launch_node->func = nullptr;

  // dynamic dump ffts_plus
  EXPECT_EQ(dumper->UpdateFftsplusLaunchTask(launch_node), ge::SUCCESS);
  ASSERT_EQ(dumper->OnUpdateDumpUnit(kExecuteStart, *launch_node, true), ge::SUCCESS);

  auto ctx_ids = ContinuousVector::Create<int32_t>(4);
  auto ctx_ids_vec = reinterpret_cast<ContinuousVector *>(ctx_ids.get());
  ctx_ids_vec->SetSize(4);
  auto ctx_ids_ptr = reinterpret_cast<int32_t *>(ctx_ids_vec->MutableData());
  for (size_t i = 0U; i < ctx_ids_vec->GetSize(); i++) {
    ctx_ids_ptr[i] = i;
  }
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo *>(holder.get());
  // buf_offset = sizeof(rtFftsPlusComCtx_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  const size_t args_size = sizeof(rtFftsPlusComCtx_t);
  auto *buff_ptr = &task_info_ptr->args[args_size];
  for (int i = 0; i < 4; ++i) {
    auto context = reinterpret_cast<rtFftsPlusAicAivCtx_t *>(buff_ptr);
    context->contextType = RT_CTX_TYPE_AICORE;
    context->threadDim = 1U;
    buff_ptr += sizeof(rtFftsPlusComCtx_t);
  }
  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[args_size];
  ffts_plus_sqe->totalContextNum = 16;

  auto context_holder_2 = KernelRunContextFaker()
                              .NodeName(std::move("add1"))
                              .KernelIONum(15, 0)
                              .KernelType("AICoreUpdateContext")
                              .KernelName("AICoreUpdateContext")
                              .Build();
  context_holder_2.value_holder[1].Set(ctx_ids_vec, nullptr);
  context_holder_2.value_holder[14].Set(&task_inf, nullptr);
  size = sizeof(Node) + sizeof(AsyncAnyValue *) * 15;
  Node *launch_node_2 = (Node *)malloc(size);
  launch_node_2->node_id = id;
  memcpy(&launch_node_2->context, context_holder_2.context, sizeof(KernelRunContext) + 15 * sizeof(AsyncAnyValue *));
  launch_node_2->func = nullptr;

  // dynamic dump ffts_plus
  ASSERT_EQ(dumper->OverflowDump(launch_node_1, kExecuteStart), ge::SUCCESS);
  ASSERT_EQ(dumper->OverflowDump(launch_node_2, kModelStart), ge::SUCCESS);
  ASSERT_EQ(dumper->OverflowDump(launch_node_2, kExecuteEnd), ge::SUCCESS);
  ASSERT_EQ(dumper->OverflowDump(launch_node_2, kModelEnd), ge::SUCCESS);
  // buf_offset = sizeof(rtFftsPlusComCtx_t);
  auto *buff_ptr_1 = &task_info_ptr->args[args_size];
  for (int i = 0; i < 4; ++i) {
    auto context = reinterpret_cast<rtFftsPlusAiCpuCtx_t *>(buff_ptr_1);
    context->contextType = RT_CTX_TYPE_AICPU;
    context->threadDim = 1U;
    buff_ptr_1 += sizeof(rtFftsPlusComCtx_t);
  }
  task_inf.descBuf = &task_info_ptr->args[args_size];

  auto context_holder_3 = KernelRunContextFaker()
                              .NodeName(std::move("add1"))
                              .KernelIONum(15, 0)
                              .KernelType("AICpuUpdateContext")
                              .KernelName("AICpuUpdateContext")
                              .Build();
  context_holder_3.value_holder[1].Set(ctx_ids_vec, nullptr);
  context_holder_3.value_holder[14].Set(&task_inf, nullptr);
  size = sizeof(Node) + sizeof(AsyncAnyValue *) * 15;
  Node *launch_node_3 = (Node *)malloc(size);
  launch_node_3->node_id = id;
  memcpy(&launch_node_3->context, context_holder_3.context, sizeof(KernelRunContext) + 15 * sizeof(AsyncAnyValue *));
  launch_node_3->func = nullptr;

  auto &dump_unit = dumper->node_names_to_dump_units_["add1"];
  StorageShape storage_shape{{4}, {4, 1}};
  kernel::BuildTensorAttr attr{kOnDeviceHbm, ge::DT_FLOAT16, {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ExpandDimsType()}};
  Tensor tensor_holder{storage_shape, attr.storage_format, attr.placement, attr.data_type, nullptr};
  tensor_holder.MutableTensorData() = TensorData{(void *)1024, nullptr, 0, kOnDeviceHbm};
  auto context_holder_4 =
      KernelRunContextFaker().KernelIONum(1, 2).Outputs({&tensor_holder, &tensor_holder.MutableTensorData()}).Build();
  dump_unit.output_addrs[0] = context_holder_4.GetContext<KernelContext>()->GetOutput(1);
  dump_unit.input_addrs[0] = context_holder_4.GetContext<KernelContext>()->GetOutput(1);
  dump_unit.input_addrs[1] = context_holder_4.GetContext<KernelContext>()->GetOutput(1);
  gert::StorageShape x2_shape = {{4, 8, 16, 32, 64}, {4, 8, 16, 4, 2, 16, 16}};
  gert::StorageShape x1_shape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
  auto context_holder_5 = KernelRunContextFaker()
                              .Inputs({&x1_shape})
                              .Outputs({&x2_shape})
                              .KernelIONum(1, 1)
                              .NodeIoNum(1, 1)
                              .IrInstanceNum({1})
                              .Build();
  dump_unit.output_shapes[0] = context_holder_5.GetContext<KernelContext>()->GetOutput(0);
  dump_unit.input_shapes[0] = const_cast<Chain *>(context_holder_5.GetContext<KernelContext>()->GetInput(0));
  dump_unit.input_shapes[1] = const_cast<Chain *>(context_holder_5.GetContext<KernelContext>()->GetInput(0));

  ge::Context dump_context;
  dump_context.context_id = 0;
  dump_context.thread_id = 0;
  dump_unit.context_list.emplace_back(dump_context);

  auto properties = ge::DumpManager::GetInstance().GetDumpProperties(ge::kInferSessionId);
  EXPECT_EQ(dumper->DoDataDump(dump_unit, properties), ge::SUCCESS);

  // dynamic dump ffts_plus
  EXPECT_EQ(dumper->UpdateFftsplusLaunchTask(launch_node), ge::SUCCESS);

  free(launch_node);
  free(launch_node_1);
  free(launch_node_2);
  free(launch_node_3);
  ge::diagnoseSwitch::DisableDumper();

  properties.ClearOpDebugFlag();
  EXPECT_EQ(dumper->DoDataDump(dump_unit, properties), ge::SUCCESS);
  ge::DumpManager::GetInstance().RemoveDumpProperties(0);
}

TEST_F(ExecutorDumperUT, DoDataDump_WatchModeFailed) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  uint64_t i = FindFirstNonEmptyId(dumper);
  auto fake_node = FakeNodeHelper::FakeNode("add1", "test", i);
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("LAYER_OP_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.AddPropertyValue("WATCHER_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  EXPECT_EQ(dumper->OnUpdateDumpUnit(kExecuteEnd, fake_node.node), ge::SUCCESS);
}

TEST_F(ExecutorDumperUT, ExecutorExceptionDumpInfoWrapperUT) {
  ExceptionDumpUint unit;
  ExecutorExceptionDumpInfoWrapper wrapper(&unit);
  uint64_t tiling_data = 100;
  uint32_t tiling_key = 101;
  uint64_t host_args = 102;
  uint64_t workspace = 103;
  int64_t workspace_size = static_cast<int64_t>(sizeof(workspace));
  wrapper.SetTilingData(reinterpret_cast<uintptr_t>(&tiling_data), sizeof(tiling_data));
  wrapper.SetTilingKey(tiling_key);
  wrapper.SetHostArgs(reinterpret_cast<uintptr_t>(&host_args), sizeof(host_args));
  wrapper.AddWorkspace(reinterpret_cast<uintptr_t>(&workspace), workspace_size);

  EXPECT_EQ(unit.is_host_args, true);
  EXPECT_EQ(unit.args, reinterpret_cast<uintptr_t>(&host_args));
  EXPECT_EQ(unit.args_size, sizeof(host_args));
  EXPECT_EQ(unit.workspace_info.size(), 1U);
  EXPECT_EQ(unit.workspace_info[0].first, reinterpret_cast<uintptr_t>(&workspace));
  EXPECT_EQ(unit.workspace_info[0].second, workspace_size);
  EXPECT_EQ(unit.tiling_key, tiling_key);

  unit.Clear();
  EXPECT_EQ(unit.tiling_data.empty(), true);
  EXPECT_EQ(unit.args_before_execute.empty(), true);
  EXPECT_EQ(unit.is_host_args, false);
  EXPECT_EQ(unit.workspace_info.empty(), true);
  EXPECT_EQ(unit.tiling_key, 0U);
  EXPECT_EQ(unit.args, 0U);
  EXPECT_EQ(unit.args_size, 0U);
}

TEST_F(ExecutorDumperUT, InitOrderHoldersFromExeGraph_UT) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  dumper->kernel_idxes_to_dump_units_.clear();
  dumper->kernel_idxes_to_dump_units_.resize(3);

  // normal exec node, add dependency
  NodeDumpUnit dump_unit;
  Node node;
  node.node_id = 0;
  size_t total_size = 0UL;
  ComputeNodeInfo::CalcSize(1, 1, 1, total_size);
  auto compute_node_info_holder = ge::ComGraphMakeUnique<uint8_t[]>(total_size);
  ComputeNodeInfo *compute_node_info = ge::PtrToPtr<uint8_t, ComputeNodeInfo>(compute_node_info_holder.get());
  compute_node_info->Init(1, 1, 1, "Test_1025", "Launch");
  uint8_t holder[sizeof(KernelExtendInfo)];
  KernelExtendInfo *extend_kernel_info = reinterpret_cast<KernelExtendInfo *>(holder);
  extend_kernel_info->SetKernelName("Test_1025");
  extend_kernel_info->SetKernelType("AicpuLaunchCCKernel");
  KernelRunContext kernel_run_context;
  kernel_run_context.input_size = 1;
  kernel_run_context.output_size = 1;
  kernel_run_context.compute_node_info = static_cast<void *>(compute_node_info);
  kernel_run_context.kernel_extend_info = static_cast<void *>(extend_kernel_info);
  node.context = kernel_run_context;

  dumper->kernel_names_to_exe_nodes_.emplace("Test_1025", &node);
  dumper->kernel_idxes_to_dump_units_.clear();
  dumper->kernel_idxes_to_dump_units_.resize(2);
  dumper->InitOrderHoldersFromExeGraph("Test_1025", &dump_unit);
  EXPECT_EQ(dumper->kernel_idxes_to_dump_units_[0].size(), 1);
  EXPECT_EQ(dumper->kernel_idxes_to_dump_units_[0][0], &dump_unit);

  // SendEvents node, no dependency
  Node node1;
  node1.node_id = 1;
  total_size = 0UL;
  ComputeNodeInfo::CalcSize(1, 1, 1, total_size);
  auto compute_node_info_holder1 = ge::ComGraphMakeUnique<uint8_t[]>(total_size);
  ComputeNodeInfo *compute_node_info1 = ge::PtrToPtr<uint8_t, ComputeNodeInfo>(compute_node_info_holder1.get());
  compute_node_info1->Init(1, 1, 1, "SendEvent_1025", "SendEvents");
  uint8_t holder1[sizeof(KernelExtendInfo)];
  KernelExtendInfo *extend_kernel_info1 = reinterpret_cast<KernelExtendInfo *>(holder1);
  extend_kernel_info1->SetKernelName("SendEvent_1025");
  extend_kernel_info1->SetKernelType("SendEvents");
  KernelRunContext kernel_run_context1;
  kernel_run_context1.input_size = 1;
  kernel_run_context1.output_size = 1;
  kernel_run_context1.compute_node_info = static_cast<void *>(compute_node_info1);
  kernel_run_context1.kernel_extend_info = static_cast<void *>(extend_kernel_info1);
  node1.context = kernel_run_context1;

  dumper->kernel_names_to_exe_nodes_.emplace("SendEvent_1025", &node1);
  dumper->InitOrderHoldersFromExeGraph("SendEvent_1025", &dump_unit);
  EXPECT_EQ(dumper->kernel_idxes_to_dump_units_[1].size(), 0);

  // WaitEvents node, no dependency
  Node node2;
  node2.node_id = 2;
  total_size = 0UL;
  ComputeNodeInfo::CalcSize(1, 1, 1, total_size);
  auto compute_node_info_holder2 = ge::ComGraphMakeUnique<uint8_t[]>(total_size);
  ComputeNodeInfo *compute_node_info2 = ge::PtrToPtr<uint8_t, ComputeNodeInfo>(compute_node_info_holder2.get());
  compute_node_info2->Init(1, 1, 1, "WaitEvents_1025", "WaitEvents");
  uint8_t holder2[sizeof(KernelExtendInfo)];
  KernelExtendInfo *extend_kernel_info2 = reinterpret_cast<KernelExtendInfo *>(holder2);
  extend_kernel_info2->SetKernelName("WaitEvents_1025");
  extend_kernel_info2->SetKernelType("WaitEvents");
  KernelRunContext kernel_run_context2;
  kernel_run_context2.input_size = 1;
  kernel_run_context2.output_size = 1;
  kernel_run_context2.compute_node_info = static_cast<void *>(compute_node_info2);
  kernel_run_context2.kernel_extend_info = static_cast<void *>(extend_kernel_info2);
  node2.context = kernel_run_context2;

  dumper->kernel_names_to_exe_nodes_.emplace("WaitEvents_1025", &node2);
  dumper->InitOrderHoldersFromExeGraph("WaitEvents_1025", &dump_unit);
  EXPECT_EQ(dumper->kernel_idxes_to_dump_units_[2].size(), 0);
}
}  // namespace gert

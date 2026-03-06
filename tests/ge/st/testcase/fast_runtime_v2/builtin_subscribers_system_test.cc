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
#define private public
#include "base/registry/op_impl_space_registry_v2.h"
#undef private
#include "common/bg_test.h"
#include "common/share_graph.h"
#include "faker/aicore_taskdef_faker.h"
#include "faker/hccl_taskdef_faker.h"
#include "faker/aicpu_taskdef_faker.h"
#include "faker/fake_value.h"
#include "faker/ge_model_builder.h"
#include "stub/gert_runtime_stub.h"
#include "framework/common/types.h"
#include "lowering/model_converter.h"
#include "aicpu_task_struct.h"
#include "depends/profiler/src/profiling_test_util.h"
#include "faker/global_data_faker.h"
#include "lowering/graph_converter.h"
#include "runtime/model_v2_executor.h"
#include "subscriber/profiler/cann_host_profiler.h"
#include "subscriber/profiler/ge_host_profiler.h"
#include "common/global_variables/diagnose_switch.h"
#include "op_tiling/op_tiling_constants.h"
#include "register/op_impl_registry.h"
#include "register/op_tiling_registry.h"
#include "check/executor_statistician.h"
#include "graph/debug/ge_attr_define.h"
#include "exe_graph/lowering/value_holder.h"
#include "ge/ut/ge/runtime/fast_v2/core/model_v2_executor_unittest.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include "kernel/tensor_attr.h"
#include "faker/kernel_run_context_facker.h"
#include "core/executor_error_code.h"

#include "macro_utils/dt_public_scope.h"
#include "subscriber/dumper/executor_dumper.h"
#include "subscriber/dumper/host_executor_dumper.h"
#include "ge/ut/ge/test_tools_task_info.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "engine/aicore/fe_rt2_common.h"
#include "graph_builder/bg_tiling.h"
#include "graph_tuner/rt2_src/graph_tunner_rt2_stub.h"
#include "macro_utils/dt_public_unscope.h"

#include "kernel/memory/caching_mem_allocator.h"
#include "common/sgt_slice_type.h"
#include "proto/dump_task.pb.h"
#include "graph/operator_factory_impl.h"
#include "depends/profiler/src/dump_stub.h"
#include "core/utils/tensor_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "framework/ge_runtime_stub/include/common/dump_checker.h"
#include "framework/runtime/gert_const_types.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace gert {
namespace {
std::string FindKernelNameByStartKeyWord(const std::string kernel_name, const std::string &key_word) {
  if (kernel_name.find(key_word, 0) == 0) {
    return kernel_name;
  }
  return "";
}
bool IsFile(const std::string &filename) {
  struct stat buffer;
  return (stat(filename.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

bool IsDirectory(const std::string &file_folder) {
  struct stat buffer;
  return (stat(file_folder.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

// remove dir and files inside
int RemoveFileAndDirectory(const std::string &path) {
  int result = 0;
  DIR *p_dir;
  struct dirent *p_dirent;
  if (IsDirectory(path)) {
    if ((p_dir = opendir(path.c_str())) == nullptr) {
      return -1;
    }

    while ((p_dirent = readdir(p_dir)) != nullptr) {
      std::string file_name = path + "/" + p_dirent->d_name;
      // It is a directory
      if (IsDirectory(file_name) && (0 != strcmp(p_dirent->d_name, ".")) && (0 != strcmp(p_dirent->d_name, ".."))) {
        result = RemoveFileAndDirectory(file_name);
        if (result < 0) {
          return result;
        }
      }
      // It is a file
      else if ((0 != strcmp(p_dirent->d_name, ".")) && (0 != strcmp(p_dirent->d_name, ".."))) {
        result = remove(file_name.c_str());
        if (result < 0) {
          return result;
        }
      }
    }
    closedir(p_dir);
    result = rmdir(path.c_str());
  } else if (IsFile(path)) {
    result = remove(path.c_str());
  }
  return result;
}

int GetFirstFileInDirectory(const std::string &path, std::string &file) {
  int result = 0;
  DIR *p_dir;
  struct dirent *p_dirent;
  if ((p_dir = opendir(path.c_str())) == nullptr) {
    return -1;
  }

  while ((p_dirent = readdir(p_dir)) != nullptr) {
    std::string file_name = path + "/" + p_dirent->d_name;
    // It is a directory
    if (IsDirectory(file_name) && (0 != strcmp(p_dirent->d_name, ".")) && (0 != strcmp(p_dirent->d_name, ".."))) {
      result = GetFirstFileInDirectory(file_name, file);
      if (result < 0) {
        return result;
      }
    } else if ((0 != strcmp(p_dirent->d_name, ".")) && (0 != strcmp(p_dirent->d_name, ".."))) {
      file = file_name;
      break;
    }
  }
  closedir(p_dir);
  return result;
}

ge::ComputeGraphPtr BuildFallibleTilingNodeComputeGraph() {
  auto graph = ShareGraph::BuildSingleNodeGraph("Add", {{-1, 2, 3, 4}, {1, -1, 3, 4}, {-1}, {-1, -1, 3, 4}},
                                                {{1, 2, 3, 4}, {1, 1, 3, 4}, {1}, {1, 1, 3, 4}},
                                                {{100, 2, 3, 4}, {1, 100, 3, 4}, {100}, {100, 100, 3, 4}});
  GE_ASSERT_NOTNULL(graph);
  graph->TopologicalSorting();
  auto add_node = graph->FindFirstNodeMatchType("Add");
  GE_ASSERT_TRUE(ge::AttrUtils::SetBool(add_node->GetOpDesc(), ge::kPartiallySupported, true));
  GE_ASSERT_TRUE(ge::AttrUtils::SetStr(add_node->GetOpDesc(), ge::kAICpuKernelLibName, "aicpu_tf_kernel"));
  GE_ASSERT_TRUE(ge::AttrUtils::SetStr(add_node->GetOpDesc(), optiling::COMPILE_INFO_KEY, "HelloWorld"));
  return graph;
}

ge::graphStatus TilingAddFail(TilingContext *tiling_context) {
  return ge::GRAPH_FAILED;
}

ge::graphStatus TilingAddSuccess(TilingContext *tiling_context) {
  return ge::GRAPH_SUCCESS;
}

void TestFallibleTiling(bool rollback) {
  auto graph = BuildFallibleTilingNodeComputeGraph();
  auto ge_root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(ge_root_model).FakeWithoutHandleAiCore("Add", false, true).Build();
  ASSERT_NE(graph, nullptr);

  GertRuntimeStub stub;
  // todo 当前没有Op的打桩方法，暂时强改注册中心里面的Tiling函数了
  auto op_impl = const_cast<OpImplKernelRegistry::OpImplFunctionsV2 *>(
      DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Add"));
  auto tiling_fun = op_impl->tiling;
  if (rollback) {
    op_impl->tiling = TilingAddFail;
  } else {
    op_impl->tiling = TilingAddSuccess;
  }
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  std::hash<std::string> hs;
  auto check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len) -> int32_t {
    EXPECT_EQ(moduleId, static_cast<uint32_t>(MsprofReporterModuleId::MSPROF_MODULE_FRAMEWORK));
    auto report_data = reinterpret_cast<ReporterData *>(data);
    std::string tag_name(report_data->tag);
    if (type == static_cast<uint32_t>(MSPROF_REPORTER_REPORT)) {
      if (tag_name == "task_desc_info") {
        auto task_data_ptr = reinterpret_cast<MsprofGeProfTaskData *>(report_data->data);
        if (rollback) {
          EXPECT_EQ(MSPROF_GE_TASK_TYPE_AI_CPU, task_data_ptr->taskType);
        } else {
          EXPECT_EQ(MSPROF_GE_TASK_TYPE_AI_CORE, task_data_ptr->taskType);
        }
      }
    }
    if (type == static_cast<uint32_t>(MSPROF_REPORTER_HASH)) {
      auto *hash_data = reinterpret_cast<MsprofHashData *>(data);
      std::string name((char *)hash_data->data, hash_data->dataLen);
      hash_data->hashId = hs(name);
    }
    return 0;
  };
  ge::ProfilingTestUtil::Instance().SetProfFunc(check_func);
  RtSession session{ge::kInferSessionId};
  ModelLoadArg load_arg(&session, {nullptr, 0});
  EXPECT_EQ(model_executor->Load({}, load_arg), ge::GRAPH_SUCCESS);

  auto inputs = FakeTensors({1, 2, 3, 4}, 2);
  auto outputs = FakeTensors({1, 2, 3, 4}, 1);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ess->Clear();
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kDevice});
  ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(), inputs.size(),
                                    outputs.GetTensorList(), outputs.size()),
            ge::GRAPH_SUCCESS);
  ess->PrintExecutionSummary();
  if (rollback) {
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "AicpuLaunchTfKernel"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "LaunchKernelWithFlag"), 0);

  } else {
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "AicpuLaunchTfKernel"), 0);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "LaunchKernelWithFlag"), 1);
  }
  Shape expect_out_shape{1, 2, 3, 4};
  EXPECT_EQ(outputs.GetTensorList()[0]->GetShape().GetStorageShape(), expect_out_shape);

  op_impl->tiling = tiling_fun;

  auto registry = const_cast<std::unordered_map<std::string, optiling::OpTilingFuncInfo> *>(
      &optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo());
  registry->erase("Add");
  ge::diagnoseSwitch::DisableProfiling();
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

void RunIfGraphWithDataDump(TensorHolder &pred_tensor, bool expect_branch) {
  auto compute_graph = ShareGraph::IfGraph2();
  ASSERT_NE(compute_graph, nullptr);
  compute_graph->TopologicalSorting();
  GeModelBuilder builder(compute_graph);
  auto ge_root_model = builder.BuildGeRootModel();

  GertRuntimeStub rtstub;
  rtstub.GetRtsRuntimeStub().Clear();
  rtstub.StubByNodeTypes({"Shape", "Rank"});

  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  auto ess = StartExecutorStatistician(model_executor);
  RtSession rt_session(2);
  ModelLoadArg load_arg{&rt_session, {nullptr, 0}};
  EXPECT_EQ(model_executor->Load(nullptr, load_arg), ge::GRAPH_SUCCESS);

  auto input_holder = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_FLOAT).Shape({8, 3, 224, 224}).Build();
  std::vector<Tensor *> inputs{pred_tensor.GetTensor(), input_holder.GetTensor()};
  auto output_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT64).Shape({8}).Build();
  std::vector<Tensor *> outputs{output_holder.GetTensor()};
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ess->Clear();
  ge::diagnoseSwitch::EnableDataDump();
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(2, dump_properties);
  ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.data(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);

  auto cpu_args = rtstub.GetRtsRuntimeStub().PopCpuLaunchArgsByKernelName("DumpDataInfo");
  EXPECT_EQ(cpu_args->GetStream(), stream);
  EXPECT_EQ(cpu_args->GetKernelName(), "DumpDataInfo");
  EXPECT_EQ(cpu_args->GetArgSize(), sizeof(aicpu::AicpuParamHead) + (2UL * sizeof(uint64_t)));
  toolkit::aicpu::dump::OpMappingInfo op_mapping_info;
  op_mapping_info.ParseFromString(cpu_args->GetSerializeDumpInfo());
  EXPECT_EQ(op_mapping_info.task_size(), 1UL);
  toolkit::aicpu::dump::Task task = op_mapping_info.task(0);
  EXPECT_EQ(task.input().at(1).address(), reinterpret_cast<uintptr_t>(input_holder.GetTensor()->GetAddr()));
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

std::unique_ptr<ModelV2Executor> BuildModelV2ExecutorWithStaticSubGraph(ge::GeRootModelPtr &root_model, GlobalDataFaker &faker,
                                                                        ge::ComputeGraphPtr &graph) {
  GertRuntimeStub fakeRuntime;
  auto global_data = faker.FakeWithoutHandleAiCore("Conv2d", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  GE_ASSERT_NOTNULL(exe_graph);
  return ModelV2Executor::Create(exe_graph, root_model);
}

bool CheckLogExpected(std::vector<OneLog> &logs, const std::string &expect_log) {
  for (auto &onelog : logs) {
    std::string content = onelog.content;
    if (content.find(expect_log) != std::string::npos) {
      return true;
    }
  }
  return false;
}
class ThirdAicpuLaunchStub : public ge::RuntimeStub {
 public:
  rtError_t rtAicpuKernelLaunchExWithArgs(uint32_t kernelType, const char *opName, uint32_t blockDim,
                                          const rtAicpuArgsEx_t *argsInfo, rtSmDesc_t *smDesc,
                                          rtStream_t stream, uint32_t flags) override {
    if (opName == string("nonzero")) {
      EXPECT_EQ(argsInfo->kernelOffsetInfoNum, 2);
      EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[0].addrOffset, 80);
      EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[0].dataOffset, 112);
      EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[1].addrOffset, 88);
      EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[1].dataOffset, 350);
    }
    return RT_ERROR_NONE;
  }
};
}  // namespace
class BuiltinSubscribersST : public bg::BgTest {
  void SetUp() override {
  }
 public:
  void BuildExecutorInner(std::unique_ptr<ModelV2Executor> &model_executor, const LoweringOption &option) {
    while (bg::ValueHolder::PopGraphFrame() != nullptr) {}
    auto graph = ShareGraph::BuildSingleNodeGraph();
    graph->TopologicalSorting();
    GeModelBuilder builder(graph);
    auto add_node = graph->FindFirstNodeMatchType("Add");
    ge::AttrUtils::SetBool(add_node->GetOpDesc(), ge::GLOBALWORKSPACE_TYPE, true);
    ge::AttrUtils::SetInt(add_node->GetOpDesc(), ge::ATTR_NAME_IMPLY_TYPE, 1);
    auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();
    ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
    ASSERT_NE(exe_graph, nullptr);
    ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
    GertRuntimeStub fakeRuntime;
    fakeRuntime.GetKernelStub().StubTiling();
    ge::GeRootModelPtr root_model = std::make_shared<ge::GeRootModel>();
    ge::AttrUtils::SetBool(graph,  ge::ATTR_SINGLE_OP_SCENE, true);
    root_model->SetRootGraph(graph);
    ge::ModelData model_data{};
    root_model->SetModelName("test_model");
    model_data.om_name = "test";
    model_executor = ModelV2Executor::Create(exe_graph, model_data, ge_root_model);
    ASSERT_NE(model_executor, nullptr);
  }

    void BuildAclnnExecutorInner(std::unique_ptr<ModelV2Executor> &model_executor, const LoweringOption &option) {
    while (bg::ValueHolder::PopGraphFrame() != nullptr) {}
    auto graph = ShareGraph::BuildSingleNodeGraph();
    graph->TopologicalSorting();
    GeModelBuilder builder(graph);
    auto add_node = graph->FindFirstNodeMatchType("Add");
    ge::AttrUtils::SetBool(add_node->GetOpDesc(), ge::GLOBALWORKSPACE_TYPE, true);
    ge::AttrUtils::SetInt(add_node->GetOpDesc(), ge::ATTR_NAME_IMPLY_TYPE, 1);
    auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();
    ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
    ASSERT_NE(exe_graph, nullptr);
    ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
    GertRuntimeStub fakeRuntime;
    fakeRuntime.GetKernelStub().StubTiling();
    ge::GeRootModelPtr root_model = std::make_shared<ge::GeRootModel>();
    ge::AttrUtils::SetBool(graph,  ge::ATTR_SINGLE_OP_SCENE, true);
    root_model->SetRootGraph(graph);
    ge::ModelData model_data{};
    root_model->SetModelName("test_model");
    model_data.om_name = "test";
    model_executor = ModelV2Executor::Create(exe_graph, model_data, ge_root_model);
    auto execution_data = reinterpret_cast<const ExecutionData *>(model_executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData());
    for (size_t i = 0UL; i < execution_data->base_ed.node_num; ++i) {
      const_cast<KernelExtendInfo *>(reinterpret_cast<const KernelExtendInfo *>(
        execution_data->base_ed.nodes[i]->context.kernel_extend_info))->SetKernelType("ExecuteOpFunc");
    }
    ASSERT_NE(model_executor, nullptr);
  }

  void BuildHcclExecutorInner(std::unique_ptr<ModelV2Executor> &model_executor, const LoweringOption &option) {
    while (bg::ValueHolder::PopGraphFrame() != nullptr) {}
    auto graph = ShareGraph::BuildSingleHcclNodeGraph();
    graph->TopologicalSorting();
    GeModelBuilder builder(graph);
    auto ge_root_model = builder.AddTaskDef("HcomReduce", HcclTaskDefFaker()).BuildGeRootModel();
    ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
    ASSERT_NE(exe_graph, nullptr);
    ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
    GertRuntimeStub fakeRuntime;
    fakeRuntime.GetKernelStub().StubTiling();
    ge::GeRootModelPtr root_model = std::make_shared<ge::GeRootModel>();
    ge::AttrUtils::SetBool(graph,  ge::ATTR_SINGLE_OP_SCENE, true);
    root_model->SetRootGraph(graph);
    ge::ModelData model_data{};
    model_data.om_name = "test";
    model_executor = ModelV2Executor::Create(exe_graph, model_data, ge_root_model);
    ASSERT_NE(model_executor, nullptr);
  }

  std::unique_ptr<ModelV2Executor> BuildExecutor(LoweringOption option={}) {
    std::unique_ptr<ModelV2Executor> model_executor;
    BuildExecutorInner(model_executor, option);
    // turn off profiling to prevent other cases from being affected
    ge::diagnoseSwitch::DisableProfiling();
    return model_executor;
  }

  std::unique_ptr<ModelV2Executor> BuildHcclExecutor(LoweringOption option={}) {
    std::unique_ptr<ModelV2Executor> model_executor;
    BuildHcclExecutorInner(model_executor, option);
    // turn off profiling to prevent other cases from being affected
    ge::diagnoseSwitch::DisableProfiling();
    return model_executor;
  }

    std::unique_ptr<ModelV2Executor> BuildAclnnExecutor(LoweringOption option={}) {
    std::unique_ptr<ModelV2Executor> model_executor;
    BuildAclnnExecutorInner(model_executor, option);
    // turn off profiling to prevent other cases from being affected
    ge::diagnoseSwitch::DisableProfiling();
    return model_executor;
  }

  void BuildAicpuExecutorInner(std::unique_ptr<ModelV2Executor> &model_executor, const LoweringOption &option,
                               int32_t k_class_shape) {
    while (bg::ValueHolder::PopGraphFrame() != nullptr) {}
    auto graph = ShareGraph::BuildBlockGraph();
    graph->TopologicalSorting();
    auto add_node = graph->FindFirstNodeMatchType("Add");
    ge::AttrUtils::SetBool(add_node->GetOpDesc(), ge::GLOBALWORKSPACE_TYPE, true);
    ge::AttrUtils::SetInt(add_node->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, k_class_shape);
    GeModelBuilder builder(graph);
    AiCpuTfTaskDefFaker aicpu_task_def_faker;
    auto ge_root_model = builder.AddTaskDef("Add", aicpu_task_def_faker.SetNeedMemcpy(true)).BuildGeRootModel();
    ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
    ASSERT_NE(exe_graph, nullptr);
    ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
    ge::DumpGraph(exe_graph.get(), "E2EBlockAddOpGraph");

    GertRuntimeStub fakeRuntime;
    fakeRuntime.GetKernelStub().StubTiling();
    ge::GeRootModelPtr root_model = std::make_shared<ge::GeRootModel>();
    ge::AttrUtils::SetBool(graph,  ge::ATTR_SINGLE_OP_SCENE, true);
    root_model->SetRootGraph(graph);
    ge::ModelData model_data{};
    root_model->SetModelName("test_model");
    model_data.om_name = "test";
    model_executor = ModelV2Executor::Create(exe_graph, model_data, ge_root_model);
    ASSERT_NE(model_executor, nullptr);
  }
  std::unique_ptr<ModelV2Executor> BuildAicpuExecutor(int32_t k_class_shape, LoweringOption option={}) {
    std::unique_ptr<ModelV2Executor> model_executor;
    BuildAicpuExecutorInner(model_executor, option, k_class_shape);
    // turn off profiling to prevent other cases from being affected
    ge::diagnoseSwitch::DisableProfiling();
    return model_executor;
  }

  void BuildExecutorInnerSpace(std::unique_ptr<ModelV2Executor> &model_executor) {
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

    auto graph = ShareGraph::BuildSingleNodeGraph();
    graph->TopologicalSorting();
    SetUnknownOpKernelForNoTiling(graph->GetAllNodes());
    GeModelBuilder builder(graph);
    AiCpuCCTaskDefFaker aicpu_task_def_faker;
    auto ge_root_model = builder.AddTaskDef("Add", aicpu_task_def_faker.SetNeedMemcpy(true)).BuildGeRootModel();

    bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
    ASSERT_NE(exe_graph, nullptr);
    ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
    ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

    model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
    ASSERT_NE(model_executor, nullptr);
  }

  std::unique_ptr<ModelV2Executor> BuildExecutorSpace() {
    std::unique_ptr<ModelV2Executor> model_executor;
    BuildExecutorInnerSpace(model_executor);
    // turn off profiling to prevent other cases from being affected
    ge::diagnoseSwitch::DisableProfiling();
    return model_executor;
  }

  void TestNormalExceptionDump(){
    setenv("NPU_COLLECT_PATH_EXE", "dump", true);
    setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", true);
    ge::diagnoseSwitch::EnableExceptionDump();
    auto model_executor = BuildExecutor();
    RtSession session{ge::kInferSessionId};
    ModelLoadArg load_arg(&session, {nullptr, 0});
    EXPECT_EQ(model_executor->Load({}, load_arg), ge::GRAPH_SUCCESS);
    auto mem_block = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors{{2048}, 2, mem_block.get()};
    GertRuntimeStub fakeRuntime;
    fakeRuntime.GetSlogStub().Clear();
    fakeRuntime.GetSlogStub().SetLevel(DLOG_INFO);
    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
    ASSERT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                      reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
              ge::GRAPH_SUCCESS);

    Adx::OperatorInfoV2 info;
    EXPECT_TRUE(ge::DumpStub::GetInstance().GetOpInfo(0, 1, 0, info));  // deviceId 0, streamId 1, taskId 0

    //  check turn off dumper
    ge::diagnoseSwitch::DisableDumper();
    EXPECT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
    unsetenv("NPU_COLLECT_PATH_EXE");
    unsetenv("ASCEND_SLOG_PRINT_TO_STDOUT");
    ge::DumpStub::GetInstance().ClearOpInfos();
  }
  memory::CachingMemAllocator caching_mem_allocator_{0};
  memory::SingleStreamL2Allocator single_stream_l2_allocator_{&caching_mem_allocator_};

 protected:
  void TearDown() {
    ge::diagnoseSwitch::MutableProfiling().SetEnableFlag(0);
    ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0);
    GlobalDumper::GetInstance()->ClearInnerExceptionDumpers();
    GlobalProfilingWrapper::GetInstance()->Free();
    ge::DumpManager::GetInstance().RemoveDumpProperties(ge::kInferSessionId); // need clear otherwise can not emplace
  }
};

TEST_F(BuiltinSubscribersST, ProfilingOk) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry();
  auto model_executor = BuildExecutor();
  RtSession session{ge::kInferSessionId};
  ModelLoadArg load_arg(&session, {nullptr, 0});
  EXPECT_EQ(model_executor->Load({}, load_arg), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling),
            nullptr);
  // turn off prof and execute->record nothing
  ge::diagnoseSwitch::MutableProfiling().SetEnableFlag(0UL);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);

  EXPECT_EQ(GlobalProfilingWrapper::GetInstance()->GetGlobalProfiler(), nullptr);
  EXPECT_EQ(GlobalProfilingWrapper::GetInstance()->GetRecordCount(), 0UL);

  // turn on prof and execute
  ge::diagnoseSwitch::MutableProfiling().SetEnableFlag(
      BuiltInSubscriberUtil::BuildEnableFlags<ProfilingType>({ProfilingType::kGeHost}));
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_NE(GlobalProfilingWrapper::GetInstance()->GetGlobalProfiler(), nullptr);
  EXPECT_NE(GlobalProfilingWrapper::GetInstance()->GetRecordCount(), 0);

  // turn off prof to dump data and free global profiler
  ge::diagnoseSwitch::MutableProfiling().SetEnableFlag(0);
  EXPECT_EQ(GlobalProfilingWrapper::GetInstance()->GetGlobalProfiler(), nullptr);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(BuiltinSubscribersST, SetAscendWorkPath_ProfilingOk) {
  ge::char_t current_path[MMPA_MAX_PATH] = {'\0'};
  getcwd(current_path, MMPA_MAX_PATH);
  mmSetEnv("ASCEND_WORK_PATH", current_path, 1);
  auto model_executor = BuildExecutor();
  RtSession session{ge::kInferSessionId};
  ModelLoadArg load_arg(&session, {nullptr, 0});
  EXPECT_EQ(model_executor->Load({}, load_arg), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling),
            nullptr);
  // turn off prof and execute->record nothing
  ge::diagnoseSwitch::MutableProfiling().SetEnableFlag(0UL);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);

  EXPECT_EQ(GlobalProfilingWrapper::GetInstance()->GetGlobalProfiler(), nullptr);
  EXPECT_EQ(GlobalProfilingWrapper::GetInstance()->GetRecordCount(), 0UL);

  // turn on prof and execute
  ge::diagnoseSwitch::MutableProfiling().SetEnableFlag(
      BuiltInSubscriberUtil::BuildEnableFlags<ProfilingType>({ProfilingType::kGeHost}));
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_NE(GlobalProfilingWrapper::GetInstance()->GetGlobalProfiler(), nullptr);
  EXPECT_NE(GlobalProfilingWrapper::GetInstance()->GetRecordCount(), 0);

  // turn off prof to dump data and free global profiler
  ge::diagnoseSwitch::MutableProfiling().SetEnableFlag(0);
  EXPECT_EQ(GlobalProfilingWrapper::GetInstance()->GetGlobalProfiler(), nullptr);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);

  std::string ge_profiling_path = current_path;
  ge_profiling_path += "/ge_profiling_" + std::to_string(mmGetPid()) + ".txt";
  EXPECT_EQ(mmAccess(ge_profiling_path.c_str()), EN_OK);
  unsetenv("ASCEND_WORK_PATH");
}

TEST_F(BuiltinSubscribersST, DataDump_Ok)  {
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0);
  auto model_executor = BuildExecutor();
  RtSession session;
  session.SetSessionId(9);
  ModelExecuteArg exec_args;
  gert::OuterWeightMem weight = {nullptr, 0};
  ModelLoadArg load_arg(&session, weight);
  EXPECT_EQ(model_executor->Load(exec_args, load_arg), ge::GRAPH_SUCCESS);
  auto mem_block_1 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto mem_block_2 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto outputs = FakeTensors({2048}, 1, (void *)mem_block_1.get());
  auto inputs = FakeTensors({2048}, 2, (void *)mem_block_2.get());

  GertRuntimeStub stub;
  stub.GetRtsRuntimeStub().Clear();
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
  ge::diagnoseSwitch::EnableDataDump();
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(session.GetSessionId(), dump_properties);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // check dump_op
  auto cpu_args = stub.GetRtsRuntimeStub().PopCpuLaunchArgsByKernelName("DumpDataInfo");
  ASSERT_NE(cpu_args, nullptr);
  EXPECT_EQ(cpu_args->GetStream(), stream);
  EXPECT_EQ(cpu_args->GetKernelName(), "DumpDataInfo");
  EXPECT_EQ(cpu_args->GetArgSize(), sizeof(aicpu::AicpuParamHead) + (2UL * sizeof(uint64_t)));
  toolkit::aicpu::dump::OpMappingInfo op_mapping_info;
  op_mapping_info.ParseFromString(cpu_args->GetSerializeDumpInfo());
  EXPECT_EQ(op_mapping_info.task_size(), 1UL);
  toolkit::aicpu::dump::Task task = op_mapping_info.task(0);
  EXPECT_EQ(task.input().at(0).address(), reinterpret_cast<uintptr_t>((void *)mem_block_2.get()));
  EXPECT_EQ(task.input().at(1).address(), reinterpret_cast<uintptr_t>((void *)mem_block_2.get()));
  EXPECT_EQ(task.output().at(0).address(), reinterpret_cast<uintptr_t>((void *)mem_block_1.get()));
  EXPECT_EQ(task.input().at(0).shape().dim(0), 2048UL);
  EXPECT_EQ(task.input().at(1).shape().dim(0), 2048UL);
  EXPECT_EQ(task.output().at(0).shape().dim(0), 2048UL);
  //  check turn off dumper
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0);
  EXPECT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(BuiltinSubscribersST, DataDumpWithOpRange_Ok)  {
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0);
  auto model_executor = BuildExecutor();
  RtSession session;
  session.SetSessionId(9);
  ModelExecuteArg exec_args;
  gert::OuterWeightMem weight = {nullptr, 0};
  ModelLoadArg load_arg(&session, weight);
  EXPECT_EQ(model_executor->Load(exec_args, load_arg), ge::GRAPH_SUCCESS);
  auto mem_block_1 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto mem_block_2 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto outputs = FakeTensors({2048}, 1, (void *)mem_block_1.get());
  auto inputs = FakeTensors({2048}, 2, (void *)mem_block_2.get());

  GertRuntimeStub stub;
  stub.GetRtsRuntimeStub().Clear();
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
  ge::diagnoseSwitch::EnableDataDump();
  ge::DumpProperties dump_properties;
  std::vector<std::pair<std::string, std::string>> op_ranges = {{"add1", "add1"}};
  dump_properties.SetOpDumpRange("g1", op_ranges);
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().RemoveDumpProperties(ge::kInferSessionId);
  ge::DumpManager::GetInstance().RemoveDumpProperties(session.GetSessionId());
  ge::DumpManager::GetInstance().AddDumpProperties(session.GetSessionId(), dump_properties);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
                                    ge::GRAPH_SUCCESS);
  ge::DumpManager::GetInstance().RemoveDumpProperties(ge::kInferSessionId);
  ge::DumpManager::GetInstance().RemoveDumpProperties(session.GetSessionId());
}

TEST_F(BuiltinSubscribersST, DataDumpByOriginalName_Ok) {
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(1UL);
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  StorageShape storage_shape{{2048}, {2048, 1}};
  kernel::BuildTensorAttr attr{kOnDeviceHbm, ge::DT_FLOAT16, {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ExpandDimsType()}};
  Tensor tensor_holder{storage_shape, attr.storage_format, attr.placement, attr.data_type, nullptr};
  tensor_holder.MutableTensorData() = TensorData{(void *)1024, nullptr, 0, kOnDeviceHbm};
  GertTensorData tensor_data;
  TensorUtils::RefTdToGtd(tensor_holder.MutableTensorData(), -1, tensor_data);
  auto context_holder_1 =
      KernelRunContextFaker().KernelIONum(1, 2).Outputs({&tensor_holder, &tensor_data}).Build();
  std::string node_name = "add1";
  const auto iter = dumper->node_names_to_dump_units_.find(node_name);
  EXPECT_NE(iter, dumper->node_names_to_dump_units_.end());
  auto op_desc = iter->second.node->GetOpDescBarePtr();

  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("LAYER_OP_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"add_ori1", "add_ori2"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);

  std::vector<std::string> original_adds = {"add_ori1", "add_ori2"};
  ge::AttrUtils::SetListStr(op_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_adds);
  const auto properties = ge::DumpManager::GetInstance().GetDumpProperties(ge::kInferSessionId);
  const auto &name = dumper->node_names_to_dump_units_["add1"].node->GetName();
  EXPECT_TRUE(dumper->IsOpInDumpList(properties, name));
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);

  // check turn off dumper
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0);
}

/**
 * 用例描述：开启溢出检测+Aicore溢出
 *
 * 预置条件：
 * 1. 包含fake Add算子的v2执行器
 * 2. 流同步接口打桩返回溢出码
 *
 * 测试步骤：
 * 1. 构造单算子Add的计算图
 * 2. lowering、加载计算图
 * 3. 使能overflow dump
 * 4. 构造输入Tensor，shape为[2048]，输出Tensor的shape为[2048]，执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 校验传入aicpu launch接口的地址与add算子的输入输出地址一致
 * 3. 校验传入aicpu launch接口的shape与add算子的shape一致
 */
TEST_F(BuiltinSubscribersST, OverflowDump_AicoreOp) {
  ge::diagnoseSwitch::EnableOverflowDump();
  mmSetEnv("SYNCSTREAM_OVERFLOW_RET", "aicore", 1);  // for rtStreamSynchronizeWithTimeout stub
  ge::DumpProperties dump_properties;
  dump_properties.InitInferOpDebug(); // open debug dump
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  ASSERT_EQ(ge::DumpManager::GetInstance().GetDumpProperties(0).IsOpDebugOpen(), true);

  auto model_executor = BuildExecutor();
  RtSession rt_session(ge::kInferSessionId);
  GertRuntimeStub stub;
  stub.GetRtsRuntimeStub().Clear();
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  EXPECT_EQ(model_executor->Load({i3.value}, {&rt_session}), ge::GRAPH_SUCCESS);
  auto mem_block_1 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto mem_block_2 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto outputs = FakeTensors({2048}, 1, (void *)mem_block_1.get());
  auto inputs = FakeTensors({2048}, 2, (void *)mem_block_2.get());

  ASSERT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // check dump_op
  auto cpu_args = stub.GetRtsRuntimeStub().PopCpuLaunchArgsByKernelName("DumpDataInfo");
  ASSERT_NE(cpu_args, nullptr);
  EXPECT_EQ(cpu_args->GetStream(), stream);
  EXPECT_EQ(cpu_args->GetKernelName(), "DumpDataInfo");
  EXPECT_EQ(cpu_args->GetArgSize(), sizeof(aicpu::AicpuParamHead) + (2UL * sizeof(uint64_t)));
  toolkit::aicpu::dump::OpMappingInfo op_mapping_info;
  op_mapping_info.ParseFromString(cpu_args->GetSerializeDumpInfo());
  EXPECT_EQ(op_mapping_info.task_size(), 1UL);
  toolkit::aicpu::dump::Task task = op_mapping_info.task(0);
  EXPECT_EQ(task.input().at(0).address(), reinterpret_cast<uintptr_t>((void *)mem_block_2.get()));
  EXPECT_EQ(task.input().at(1).address(), reinterpret_cast<uintptr_t>((void *)mem_block_2.get()));
  EXPECT_EQ(task.output().at(0).address(), reinterpret_cast<uintptr_t>((void *)mem_block_1.get()));
  EXPECT_EQ(task.input().at(0).shape().dim(0), 2048UL);
  EXPECT_EQ(task.input().at(1).shape().dim(0), 2048UL);
  EXPECT_EQ(task.output().at(0).shape().dim(0), 2048UL);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  ge::diagnoseSwitch::DisableDumper();
  rtStreamDestroy(stream);
  unsetenv("SYNCSTREAM_OVERFLOW_RET");
}

/**
 * 用例描述：开启溢出检测+Aicore溢出
 *
 * 预置条件：
 * 1. 包含fake Add算子的v2执行器
 * 2. 流同步接口打桩返回溢出码
 *
 * 测试步骤：
 * 1. 构造单算子Add的计算图
 * 2. lowering、加载计算图
 * 3. 使能aclnn 算子的overflow dump
 * 4. 构造输入Tensor，shape为[2048]，输出Tensor的shape为[2048]，执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 校验传入aicpu launch接口的地址与add算子的输入输出地址一致
 * 3. 校验传入aicpu launch接口的shape与add算子的shape一致
 */
TEST_F(BuiltinSubscribersST, OverflowDump_ExecuteFuncOp) {
  ge::diagnoseSwitch::EnableOverflowDump();
  mmSetEnv("SYNCSTREAM_OVERFLOW_RET", "aicore", 1);  // for rtStreamSynchronizeWithTimeout stub
  ge::DumpProperties dump_properties;
  dump_properties.InitInferOpDebug(); // open debug dump
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  ASSERT_EQ(ge::DumpManager::GetInstance().GetDumpProperties(0).IsOpDebugOpen(), true);

  auto model_executor = BuildAclnnExecutor();
  RtSession rt_session(ge::kInferSessionId);
  GertRuntimeStub stub;
  stub.GetRtsRuntimeStub().Clear();
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  EXPECT_EQ(model_executor->Load({i3.value}, {&rt_session}), ge::GRAPH_SUCCESS);
  auto mem_block_1 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto mem_block_2 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto outputs = FakeTensors({2048}, 1, (void *)mem_block_1.get());
  auto inputs = FakeTensors({2048}, 2, (void *)mem_block_2.get());
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);

  ASSERT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // check dump_op
  auto cpu_args = stub.GetRtsRuntimeStub().PopCpuLaunchArgsByKernelName("DumpDataInfo");
  ASSERT_NE(cpu_args, nullptr);
  EXPECT_EQ(cpu_args->GetStream(), stream);
  EXPECT_EQ(cpu_args->GetKernelName(), "DumpDataInfo");
  EXPECT_EQ(cpu_args->GetArgSize(), sizeof(aicpu::AicpuParamHead) + (2UL * sizeof(uint64_t)));
  toolkit::aicpu::dump::OpMappingInfo op_mapping_info;
  op_mapping_info.ParseFromString(cpu_args->GetSerializeDumpInfo());
  EXPECT_EQ(op_mapping_info.task_size(), 1UL);
  toolkit::aicpu::dump::Task task = op_mapping_info.task(0);
  EXPECT_EQ(task.input().at(0).address(), reinterpret_cast<uintptr_t>((void *)mem_block_2.get()));
  EXPECT_EQ(task.input().at(1).address(), reinterpret_cast<uintptr_t>((void *)mem_block_2.get()));
  EXPECT_EQ(task.output().at(0).address(), reinterpret_cast<uintptr_t>((void *)mem_block_1.get()));
  EXPECT_EQ(task.input().at(0).shape().dim(0), 2048UL);
  EXPECT_EQ(task.input().at(1).shape().dim(0), 2048UL);
  EXPECT_EQ(task.output().at(0).shape().dim(0), 2048UL);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  ge::diagnoseSwitch::DisableDumper();
  rtStreamDestroy(stream);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  unsetenv("SYNCSTREAM_OVERFLOW_RET");
}

/**
 * 用例描述：开启溢出检测+Aicpu不溢出
 *
 * 预置条件：
 * 1. 包含4类Aicpu Add算子的v2执行器
 * 2. 流同步接口打桩返回正常值
 *
 * 测试步骤：
 * 1. 构造单算子Add的计算图
 * 2. lowering、加载计算图
 * 3. 使能overflow dump
 * 4. 构造输入Tensor，shape为[2048]，输出Tensor的shape为[2048]，执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 无DumpDataInfo
 */
TEST_F(BuiltinSubscribersST, OverflowDump_NoDumpDataInfo_AicpuOpNotOverflow) {
  ge::diagnoseSwitch::EnableOverflowDump();
  ge::DumpProperties dump_properties;
  dump_properties.InitInferOpDebug(); // open debug dump
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  ASSERT_EQ(ge::DumpManager::GetInstance().GetDumpProperties(0).IsOpDebugOpen(), true);

  int32_t k_class_shape = 4;
  auto model_executor = BuildAicpuExecutor(k_class_shape);
  RtSession rt_session(ge::kInferSessionId);
  GertRuntimeStub stub;
  stub.GetRtsRuntimeStub().Clear();
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  EXPECT_EQ(model_executor->Load({i3.value}, {&rt_session}), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2}, 1);
  auto inputs = FakeTensors({2048, 2, 3, 4}, 2);

  ASSERT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // check dump_op
  auto cpu_args = stub.GetRtsRuntimeStub().PopCpuLaunchArgsByKernelName("DumpDataInfo");
  ASSERT_EQ(cpu_args, nullptr);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  ge::diagnoseSwitch::DisableDumper();
  rtStreamDestroy(stream);
}

/**
 * 用例描述：开启溢出检测+Aicpu溢出
 *
 * 预置条件：
 * 1. 包含1类Aicpu Add算子的v2执行器
 * 2. 流同步接口打桩返回溢出码
 *
 * 测试步骤：
 * 1. 构造单算子Add的计算图
 * 2. lowering、加载计算图
 * 3. 使能overflow dump
 * 4. 构造输入Tensor，shape为[2048]，输出Tensor的shape为[2048]，执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 校验传入aicpu launch接口的地址与add算子的输入输出地址一致
 * 3. 校验传入aicpu launch接口的shape与add算子的shape一致
 */
TEST_F(BuiltinSubscribersST, OverflowDump_AicpuOverflow) {
  mmSetEnv("SYNCSTREAM_OVERFLOW_RET", "aicpu", 1);  // for rtStreamSynchronizeWithTimeout stub
  ge::diagnoseSwitch::EnableOverflowDump();
  ge::DumpProperties dump_properties;
  dump_properties.InitInferOpDebug(); // open debug dump
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  ASSERT_EQ(ge::DumpManager::GetInstance().GetDumpProperties(0).IsOpDebugOpen(), true);

  int32_t k_class_shape = 1;
  auto model_executor = BuildAicpuExecutor(k_class_shape);
  RtSession rt_session(ge::kInferSessionId);
  GertRuntimeStub stub;
  stub.GetRtsRuntimeStub().Clear();
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  EXPECT_EQ(model_executor->Load({i3.value}, {&rt_session}), ge::GRAPH_SUCCESS);
  auto mem_block_1 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto mem_block_2 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto outputs = FakeTensors({2048}, 1, (void *)mem_block_1.get());
  auto inputs = FakeTensors({2048}, 2, (void *)mem_block_2.get());

  ASSERT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // check dump_op
  auto cpu_args = stub.GetRtsRuntimeStub().PopCpuLaunchArgsByKernelName("DumpDataInfo");
  ASSERT_NE(cpu_args, nullptr);
  EXPECT_EQ(cpu_args->GetStream(), stream);
  EXPECT_EQ(cpu_args->GetKernelName(), "DumpDataInfo");
  EXPECT_EQ(cpu_args->GetArgSize(), sizeof(aicpu::AicpuParamHead) + (2UL * sizeof(uint64_t)));
  toolkit::aicpu::dump::OpMappingInfo op_mapping_info;
  op_mapping_info.ParseFromString(cpu_args->GetSerializeDumpInfo());
  EXPECT_EQ(op_mapping_info.task_size(), 1UL);
  toolkit::aicpu::dump::Task task = op_mapping_info.task(0);
  EXPECT_EQ(task.input().at(0).address(), reinterpret_cast<uintptr_t>((void *)mem_block_2.get()));
  EXPECT_EQ(task.input().at(1).address(), reinterpret_cast<uintptr_t>((void *)mem_block_2.get()));
  EXPECT_EQ(task.output().at(0).address(), reinterpret_cast<uintptr_t>((void *)mem_block_1.get()));
  EXPECT_EQ(task.input().at(0).shape().dim(0), 2048UL);
  EXPECT_EQ(task.input().at(1).shape().dim(0), 2048UL);
  EXPECT_EQ(task.output().at(0).shape().dim(0), 2048UL);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  ge::diagnoseSwitch::DisableDumper();
  rtStreamDestroy(stream);
  unsetenv("SYNCSTREAM_OVERFLOW_RET");
}
TEST_F(BuiltinSubscribersST, OverflowDump_NoDataDump_ComputeNodeInfoIsNullptr) {
  ge::diagnoseSwitch::EnableOverflowDump();

  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  ASSERT_NE(dumper, nullptr);
  StorageShape storage_shape{{2048}, {2048, 1}};
  kernel::BuildTensorAttr attr{kOnDeviceHbm, ge::DT_FLOAT16, {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ExpandDimsType()}};
  Tensor tensor_holder{storage_shape, attr.storage_format, attr.placement, attr.data_type, nullptr};
  tensor_holder.MutableTensorData() = TensorData{(void *)1024, nullptr, 0, kOnDeviceHbm};
  GertTensorData tensor_data;
  TensorUtils::RefTdToGtd(tensor_holder.MutableTensorData(), -1, tensor_data);
  auto context_holder_1 =
      KernelRunContextFaker().KernelIONum(1, 2).Outputs({&tensor_holder, &tensor_data}).Build();
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

  // todo here save rt stream on main exe graph, after load to do OverflowDump is not right
  ASSERT_NE(dumper->OverflowDump(&add_kernel_launch_node, kModelStart), ge::SUCCESS);

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

/**
* 用例描述：使能总是零拷贝情况下dump
*
* 预置条件：
* 1. 包含fake Add算子的v2执行器
*
* 测试步骤：
* 1. 构造单算子Add的计算图
* 2. lowering、加载计算图
* 3. 使能data dump
* 4. 构造输入Tensor，shape为[2048]，输出Tensor的shape为[2048]，执行
*
* 预期结果：
* 1. 执行成功
* 2. 校验传入aicpu launch接口的地址与add算子的输入输出地址一致
* 3. 校验传入aicpu launch接口的shape与add算子的shape一致
*/
TEST_F(BuiltinSubscribersST, DataDump_Ok_EnableAlwaysZeroCopy) {
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0);
  auto model_executor = BuildExecutor({.trust_shape_on_out_tensor = true, .always_zero_copy = true});

  RtSession session;
  session.SetSessionId(9);
  ModelExecuteArg exec_args;
  gert::OuterWeightMem weight = {nullptr, 0};
  ModelLoadArg load_arg(&session, weight);
  EXPECT_EQ(model_executor->Load(exec_args, load_arg), ge::GRAPH_SUCCESS);
  auto mem_block_1 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto mem_block_2 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto mem_block_3 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto outputs = FakeTensors({2048}, 1, (void *)mem_block_1.get());
  auto inputs_0 = FakeTensors({2048}, 1, (void *)mem_block_2.get());
  auto inputs_1 = FakeTensors({2048}, 1, (void *)mem_block_3.get());
  auto inputs = std::vector{&inputs_0.at(0), &inputs_1.at(0)};

  GertRuntimeStub stub;
  stub.GetRtsRuntimeStub().Clear();
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
  ge::diagnoseSwitch::EnableDataDump();
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(session.GetSessionId(), dump_properties);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // check dump_op
  auto cpu_args = stub.GetRtsRuntimeStub().PopCpuLaunchArgsByKernelName("DumpDataInfo");
  ASSERT_NE(cpu_args, nullptr);
  EXPECT_EQ(cpu_args->GetStream(), stream);
  EXPECT_EQ(cpu_args->GetKernelName(), "DumpDataInfo");
  EXPECT_EQ(cpu_args->GetArgSize(), sizeof(aicpu::AicpuParamHead) + (2UL * sizeof(uint64_t)));
  toolkit::aicpu::dump::OpMappingInfo op_mapping_info;
  op_mapping_info.ParseFromString(cpu_args->GetSerializeDumpInfo());
  EXPECT_EQ(op_mapping_info.task_size(), 1UL);

  auto src = stub.GetRtsRuntimeStub().FindSrcAddrCpyToDst(op_mapping_info.step_id_addr());
  EXPECT_EQ(src, 0);

  toolkit::aicpu::dump::Task task = op_mapping_info.task(0);
  EXPECT_EQ(task.input().at(0).address(), reinterpret_cast<uintptr_t>((void *)mem_block_2.get()));
  EXPECT_EQ(task.input().at(1).address(), reinterpret_cast<uintptr_t>((void *)mem_block_3.get()));
  EXPECT_EQ(task.output().at(0).address(), reinterpret_cast<uintptr_t>((void *)mem_block_1.get()));
  EXPECT_EQ(task.input().at(0).shape().dim(0), 2048UL);
  EXPECT_EQ(task.input().at(1).shape().dim(0), 2048UL);
  EXPECT_EQ(task.output().at(0).shape().dim(0), 2048UL);

  //  check turn off dumper
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0);
  EXPECT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(BuiltinSubscribersST, DataDump_Ok_WithHostAddr) {
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0);
  auto model_executor = BuildExecutor();
  RtSession session;
  session.SetSessionId(9);
  ModelExecuteArg exec_args;
  gert::OuterWeightMem weight = {nullptr, 0};
  ModelLoadArg load_arg(&session, weight);
  EXPECT_EQ(model_executor->Load(exec_args, load_arg), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2048}, 1);
  auto mem_block2 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto inputs = FakeTensors({2048}, 2, mem_block2.get(), kOnHost);

  GertRuntimeStub stub;
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
  ge::diagnoseSwitch::EnableDataDump();
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(session.GetSessionId(), dump_properties);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // check dump_op
  auto cpu_args = stub.GetRtsRuntimeStub().PopCpuLaunchArgsByKernelName("DumpDataInfo");
  ASSERT_NE(cpu_args, nullptr);
  EXPECT_EQ(cpu_args->GetStream(), stream);
  EXPECT_EQ(cpu_args->GetKernelName(), "DumpDataInfo");
  EXPECT_EQ(cpu_args->GetArgSize(), sizeof(aicpu::AicpuParamHead) + (2UL * sizeof(uint64_t)));
  toolkit::aicpu::dump::OpMappingInfo op_mapping_info;
  op_mapping_info.ParseFromString(cpu_args->GetSerializeDumpInfo());
  EXPECT_EQ(op_mapping_info.task_size(), 1UL);
  toolkit::aicpu::dump::Task task = op_mapping_info.task(0);
  EXPECT_EQ(reinterpret_cast<uint8_t *>(task.output().at(0).address())[0], 'F');
  EXPECT_EQ(reinterpret_cast<uint8_t *>(task.output().at(0).address())[1], 'A');
  EXPECT_EQ(reinterpret_cast<uint8_t *>(task.output().at(0).address())[2], 'K');
  EXPECT_EQ(reinterpret_cast<uint8_t *>(task.output().at(0).address())[3], 'E');
  EXPECT_EQ(task.input().at(0).shape().dim(0), 2048UL);
  EXPECT_EQ(task.input().at(1).shape().dim(0), 2048UL);
  EXPECT_EQ(task.output().at(0).shape().dim(0), 2048UL);
  //  check turn off dumper
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0);
  EXPECT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(BuiltinSubscribersST, ExceptionDump_Ok) {
  TestNormalExceptionDump();
}

TEST_F(BuiltinSubscribersST, DataDump_Workspace_Ok) {
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0);
  auto model_executor = BuildExecutorSpace();
  RtSession session;
  session.SetSessionId(9);
  ModelExecuteArg exec_args;
  gert::OuterWeightMem weight = {nullptr, 0};
  ModelLoadArg load_arg(&session, weight);
  EXPECT_EQ(model_executor->Load(exec_args, load_arg), ge::GRAPH_SUCCESS);
  auto mem_block_1 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto mem_block_2 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto outputs = FakeTensors({2048}, 1, (void *)mem_block_1.get());
  auto inputs = FakeTensors({2048}, 2, (void *)mem_block_2.get());

  GertRuntimeStub stub;
  stub.GetRtsRuntimeStub().Clear();
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
  ge::diagnoseSwitch::EnableDataDump();
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(session.GetSessionId(), dump_properties);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // check dump_op
  auto cpu_args = stub.GetRtsRuntimeStub().PopCpuLaunchArgsByKernelName("DumpDataInfo");
  EXPECT_EQ(cpu_args->GetStream(), stream);
  EXPECT_EQ(cpu_args->GetKernelName(), "DumpDataInfo");
  EXPECT_EQ(cpu_args->GetArgSize(), sizeof(aicpu::AicpuParamHead) + (2UL * sizeof(uint64_t)));
  toolkit::aicpu::dump::OpMappingInfo op_mapping_info;
  op_mapping_info.ParseFromString(cpu_args->GetSerializeDumpInfo());
  EXPECT_EQ(op_mapping_info.task_size(), 1UL);
  toolkit::aicpu::dump::Task task = op_mapping_info.task(0);
  EXPECT_EQ(task.input().at(0).address(), reinterpret_cast<uintptr_t>((void *)mem_block_2.get()));
  EXPECT_EQ(task.input().at(1).address(), reinterpret_cast<uintptr_t>((void *)mem_block_2.get()));
  EXPECT_EQ(task.output().at(0).address(), reinterpret_cast<uintptr_t>((void *)mem_block_1.get()));
  EXPECT_EQ(task.input().at(0).shape().dim(0), 2048UL);
  EXPECT_EQ(task.input().at(1).shape().dim(0), 2048UL);
  EXPECT_EQ(task.output().at(0).shape().dim(0), 2048UL);
  if (!task.space().empty()) {
    EXPECT_EQ(task.space().at(0).size(), 32UL);
  }
  //  check turn off dumper
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0);
  EXPECT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(BuiltinSubscribersST, ExceptionDump_Workspace_Ok) {
  setenv("NPU_COLLECT_PATH_EXE", "dump", true);
  ge::diagnoseSwitch::DisableDumper();
  auto model_executor = BuildExecutorSpace();
  RtSession session{ge::kInferSessionId};
  ModelLoadArg load_arg(&session, {nullptr, 0});
  EXPECT_EQ(model_executor->Load({}, load_arg), ge::GRAPH_SUCCESS);
  auto mem_block = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors{{2048}, 2, mem_block.get()};

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
  ge::diagnoseSwitch::EnableExceptionDump();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  Adx::OperatorInfoV2 info;
  EXPECT_TRUE(ge::DumpStub::GetInstance().GetOpInfo(0, 0, 0, info));  // deviceId 0, streamId 0, taskId 0

  //  check turn off dumper
  ge::diagnoseSwitch::DisableDumper();
  EXPECT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
  ge::DumpStub::GetInstance().ClearOpInfos();
  unsetenv("NPU_COLLECT_PATH_EXE");
}

TEST_F(BuiltinSubscribersST, ExceptionDump_Host_Ok) {
  setenv("NPU_COLLECT_PATH_EXE", "dump", true);
  ge::diagnoseSwitch::DisableDumper();
  auto model_executor = BuildExecutorSpace();
  RtSession session{ge::kInferSessionId};
  ModelLoadArg load_arg(&session, {nullptr, 0});
  EXPECT_EQ(model_executor->Load({}, load_arg), ge::GRAPH_SUCCESS);
  auto mem_block = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);

  auto outputs = FakeTensors({2048}, 1, nullptr, kOnHost);
  auto inputs = FakeTensors{{2048}, 2, mem_block.get(), kOnHost};

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
  ge::diagnoseSwitch::EnableExceptionDump();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
  reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
  ge::GRAPH_SUCCESS);

  Adx::OperatorInfoV2 info;
  EXPECT_TRUE(ge::DumpStub::GetInstance().GetOpInfo(0, 0, 0, info));  // deviceId 0, streamId 0, taskId 0
  ge::diagnoseSwitch::DisableDumper();
  rtStreamDestroy(stream);
  ge::DumpStub::GetInstance().ClearOpInfos();
  unsetenv("NPU_COLLECT_PATH_EXE");
}

TEST_F(BuiltinSubscribersST, CannProfiling_Ok_RollbackToAiCpu) {
  TestFallibleTiling(true);
  TestFallibleTiling(false);
}

TEST_F(BuiltinSubscribersST, DataDump_Ok_WithControlNodeInExegraph) {
  auto pred_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Value<int32_t>({1}).Build();
  RunIfGraphWithDataDump(pred_holder, true);
}


/**
 * 用例描述：静态子图中的算子发生aic error，能够进行exception dump
 *
 * 预置条件：
 * 1. 构造包含静态子图的计算图
 *
 * 测试步骤：
 * 1. 使能exception dump
 * 2. 构造包含静态子图的计算图
 * 3. 转换为执行图，构造执行器
 * 4. 构造输入Tensor执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 校验在预设路径下有dump文件生成
 */
TEST_F(BuiltinSubscribersST, ExceptionDump_DumpStaticNode_WithStaticSubGraph) {
  setenv("NPU_COLLECT_PATH_EXE", "dump", true);
  ge::diagnoseSwitch::DisableDumper();
  auto graph = ShareGraph::BuildWithKnownSubgraph();
  graph->TopologicalSorting();
  auto ge_root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(ge_root_model);
  auto model_executor = BuildModelV2ExecutorWithStaticSubGraph(ge_root_model, faker, graph);
  ASSERT_NE(model_executor, nullptr);

  ge::DumpStub::GetInstance().ClearOpInfos();
  ge::diagnoseSwitch::EnableExceptionDump();
  RtSession session{ge::kInferSessionId};
  ModelLoadArg load_arg(&session, {nullptr, 0});
  EXPECT_EQ(model_executor->Load({}, load_arg), ge::GRAPH_SUCCESS);

  auto mem_block = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto outputs = FakeTensors({2, 2}, 3);
  auto inputs = FakeTensors({2, 2}, 1, mem_block.get());
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  EXPECT_FALSE(ge::DumpStub::GetInstance().GetOpInfos().empty());
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
  ge::DumpStub::GetInstance().ClearOpInfos();
  ge::diagnoseSwitch::DisableDumper();
  unsetenv("NPU_COLLECT_PATH_EXE");
}

static void BuildMixL2NodeGraph(ge::ComputeGraphPtr &root_graph, ge::NodePtr &node) {
  DEF_GRAPH(fused_graph) {
    auto data_0 = OP_CFG(ge::DATA).Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto data_1 = OP_CFG(ge::DATA).Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto ret_val_0 = OP_CFG(ge::FRAMEWORKOP).Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0)
                         .Attr(ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "_RetVal");
    auto conv = OP_CFG("CONV2D_T")
                    .Attr(ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                    .Attr(ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
                    .Attr(ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    auto sqrt = OP_CFG("SQRT_T")
                    .Attr(ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                    .Attr(ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
                    .Attr(ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    CHAIN(NODE("_arg_in_0", data_0)
              ->EDGE(0, 0)
              ->NODE("conv2d", conv)
              ->EDGE(0, 0)
              ->NODE("sqrt", sqrt)
              ->EDGE(0, 0)
              ->NODE("retVal", ret_val_0));
    CHAIN(NODE("_arg_in_1", data_1)
              ->EDGE(0, 1)
              ->NODE("conv2d", conv));
  };
  auto origin_fused_graph = ge::ToComputeGraph(fused_graph);

  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(ge::DATA).Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto data_1 = OP_CFG(ge::DATA).Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto fused_conv = OP_CFG("CONV2D_T")
                          .Attr(ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                          .Attr(ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
                          .Attr(ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    CHAIN(NODE("_arg_0", data_0)
              ->EDGE(0, 0)
              ->NODE("Conv2D_Sqrt", fused_conv)
              ->EDGE(0, 0)
              ->NODE("Node_Output", ge::NETOUTPUT));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("Conv2D_Sqrt", fused_conv));
  };
  uint32_t mem_offset = 0U;
  root_graph = ge::ToComputeGraph(g1);
  root_graph->SetGraphUnknownFlag(true);
  SetUnknownOpKernel(root_graph, mem_offset, true);
  SetGraphOutShapeRange(root_graph);
  ge::AttrUtils::SetInt(root_graph->FindNode("_arg_0")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(root_graph->FindNode("_arg_1")->GetOpDesc(), "index", 1);
  node = root_graph->FindNode("Conv2D_Sqrt");
  auto conv2d_desc = node->GetOpDesc();
  ge::AttrUtils::SetGraph(conv2d_desc, "_original_fusion_graph", origin_fused_graph);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::kAttrLowingFunc, ge::kFFTSMixL2LowerFunc);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::kAttrCalcArgsSizeFunc, ge::kFFTSMixL2CalcFunc);
  (void)ge::AttrUtils::SetInt(conv2d_desc, bg::kMaxTilingSize, 50);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::ATTR_NAME_ALIAS_ENGINE_NAME, "mix_l2");

  vector<int64_t> workspace_bytes = { 200, 300, 400};
  conv2d_desc->SetWorkspaceBytes(workspace_bytes);

  string compile_info_key = "compile_info_key";
  string compile_info_json = "{\"_workspace_size_list\":[]}";
  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = ge::MakeShared<ge::OpKernelBin>("s_mix_aictbeKernel", std::move(test_bin));
  conv2d_desc->SetExtAttr(std::string("_mix_aic") + ge::OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);
  conv2d_desc->SetExtAttr(std::string("_mix_aiv") + ge::OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::TVM_ATTR_NAME_MAGIC, "FFTS_BINARY_MAGIC_ELF_MIX_AIC");
  ge::AttrUtils::SetBool(conv2d_desc, "support_dynamicshape", true);
  ge::AttrUtils::SetInt(conv2d_desc, "op_para_size", 512);
  (void)ge::AttrUtils::SetStr(conv2d_desc, optiling::COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(conv2d_desc, optiling::COMPILE_INFO_JSON, compile_info_json);

  std::vector<std::string> names_prefix;
  names_prefix.emplace_back("_mix_aic");
  (void)ge::AttrUtils::SetListStr(node->GetOpDesc(), ge::ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
}

static ge::graphStatus TilingTestSuccess(TilingContext *tiling_context) {
  tiling_context->SetTilingKey(0);
  tiling_context->SetBlockDim(32);
  TilingData *tiling_data = tiling_context->GetRawTilingData();
  if (tiling_data == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto workspaces = tiling_context->GetWorkspaceSizes(2);
  workspaces[0] = 4096;
  workspaces[1] = 6904;
  int64_t data = 100;
  tiling_data->Append<int64_t>(data);
  tiling_data->SetDataSize(sizeof(int64_t));
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeTest1(InferShapeContext *context) {
  auto input_shape_0 = *context->GetInputShape(0);
  auto input_shape_1 = *context->GetInputShape(1);
  auto output_shape = context->GetOutputShape(0);
  if (input_shape_0.GetDimNum() != input_shape_1.GetDimNum()) {
    GELOGE(ge::PARAM_INVALID, "Add param invalid, node:[%s], input_shape_0.GetDimNum() is %zu,  input_shape_1.GetDimNum() is %zu",
           context->GetNodeName(), input_shape_0.GetDimNum(), input_shape_1.GetDimNum());
  }
  output_shape->SetDimNum(input_shape_0.GetDimNum());
  for (size_t i = 0; i < input_shape_0.GetDimNum(); ++i) {
    output_shape->SetDim(i, std::max(input_shape_0.GetDim(i), input_shape_1.GetDim(i)));
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseTEST(gert::KernelContext *kernel_context) {
  (void)kernel_context;
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeTest2(InferShapeContext *context) {
  auto input_shape_0 = *context->GetInputShape(0);
  auto output_shape = context->GetOutputShape(0);
  output_shape->SetDimNum(input_shape_0.GetDimNum());
  for (size_t i = 0; i < input_shape_0.GetDimNum(); ++i) {
    output_shape->SetDim(i, input_shape_0.GetDim(i));
  }
  return ge::GRAPH_SUCCESS;
}

static void BuildMixL2GraphAndTaskDef(ge::GeRootModelPtr &root_model, LoweringGlobalData &global_data) {
  ge::NodePtr node = nullptr;
  ge::ComputeGraphPtr root_graph;
  BuildMixL2NodeGraph(root_graph, node);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = ge::MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ge::ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));
  (void)ctx_def_0.mutable_label_ctx();
  uint32_t need_mode = 1U;
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kNeedModeAddr, need_mode);

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(1);

  root_model = GeModelBuilder(root_graph).BuildGeRootModel();
  global_data = GlobalDataFaker(root_model).Build();
  global_data.AddCompiledResult(node, {{task_def}});
  OpImplSpaceRegistryV2Array space_registry_v2_array;
  space_registry_v2_array[static_cast<size_t>(OppImplVersionTag::kOpp)] = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  global_data.SetSpaceRegistriesV2(space_registry_v2_array);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);
  auto op_impl_func = space_registry_v2_array[static_cast<size_t>(OppImplVersionTag::kOpp)]->CreateOrGetOpImpl("CONV2D_T");
  op_impl_func->infer_shape = InferShapeTest1;
  op_impl_func->tiling = TilingTestSuccess;
  op_impl_func->tiling_parse = TilingParseTEST;
  space_registry_v2_array[static_cast<size_t>(OppImplVersionTag::kOpp)]->CreateOrGetOpImpl("SQRT_T")->infer_shape = InferShapeTest2;

  auto infer_fun = [](ge::Operator &op) -> ge::graphStatus {
    const char_t *name = "__output0";
    op.UpdateOutputDesc(name, op.GetInputDesc(0));
    return ge::GRAPH_SUCCESS;
  };
  ge::OperatorFactoryImpl::RegisterInferShapeFunc("CONV2D_T", infer_fun);
  ge::OperatorFactoryImpl::RegisterInferShapeFunc("SQRT_T", infer_fun);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);
}

void TestFFTSAutoStaticLowering(ge::ComputeGraphPtr &graph, LoweringGlobalData &global_data) {
  ASSERT_NE(graph, nullptr);
  graph->TopologicalSorting();
  auto ge_root_model = GeModelBuilder(graph).BuildGeRootModel();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "LoweringFFTSAutoStaticGraph");

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  auto ess = StartExecutorStatistician(model_executor);

  RtSession session{ge::kInferSessionId};
  ModelLoadArg load_arg(&session, {nullptr, 0});
  EXPECT_EQ(model_executor->Load({}, load_arg), ge::GRAPH_SUCCESS);
  FakeTensors inputs = FakeTensors({6, 4, 4, 4}, 2);
  auto output = TensorFaker().Shape({3,4,4,4}).DataType(ge::DT_INT64).Build();
  std::vector<Tensor *> outputs{output.GetTensor()};

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ess->Clear();
  ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(), inputs.size(),
                                    outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  ess->PrintExecutionSummary();

  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("CONV2D_T", "StaAutoUpdateContext"), 1);

  Shape expect_out_shape = {3, 4, 4, 4};
  EXPECT_EQ(outputs.data()[0]->GetShape().GetStorageShape(), expect_out_shape);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(BuiltinSubscribersST, OverflowDumpForMixl2) {
  ge::diagnoseSwitch::EnableOverflowDump();
  mmSetEnv("SYNCSTREAM_OVERFLOW_RET", "aicpu", 1);
  ge::GeRootModelPtr ge_root_model;
  LoweringGlobalData global_data;
  BuildMixL2GraphAndTaskDef(ge_root_model, global_data);

  auto root_graph = ge_root_model->GetRootGraph();
  root_graph->TopologicalSorting();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto exe_graph = GraphConverter().SetModelDescHolder(&model_desc_holder).
                   ConvertComputeGraphToExecuteGraph(root_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  RtSession session{ge::kInferSessionId};
  ModelLoadArg load_arg(&session, {nullptr, 0});
  EXPECT_EQ(model_executor->Load({}, load_arg), ge::GRAPH_SUCCESS);
  FakeTensors inputs = FakeTensors({4, 4, 4, 4}, 2);
  FakeTensors outputs = FakeTensors({4, 4, 4, 4}, 1);

  GertRuntimeStub stub;
  stub.GetRtsRuntimeStub().Clear();
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper),
            nullptr);
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.InitInferOpDebug();
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(), inputs.size(),
                                    outputs.GetTensorList(), outputs.size()), ge::GRAPH_SUCCESS);

  ge::diagnoseSwitch::DisableDumper();
  unsetenv("SYNCSTREAM_OVERFLOW_RET");
  auto cpu_args = stub.GetRtsRuntimeStub().PopCpuLaunchArgsByKernelName("test_haha");
  ASSERT_NE(cpu_args, nullptr);
  toolkit::aicpu::dump::OpMappingInfo op_mapping_info;
  op_mapping_info.ParseFromString(cpu_args->GetSerializeDumpInfo());
  const auto task_size = op_mapping_info.task_size();
  ASSERT_EQ(task_size, 1);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(BuiltinSubscribersST, DataDumpForMixl2) {
  ge::diagnoseSwitch::EnableDataDump();
  ge::GeRootModelPtr ge_root_model;
  LoweringGlobalData global_data;
  BuildMixL2GraphAndTaskDef(ge_root_model, global_data);

  auto root_graph = ge_root_model->GetRootGraph();
  root_graph->TopologicalSorting();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto exe_graph = GraphConverter().SetModelDescHolder(&model_desc_holder).
      ConvertComputeGraphToExecuteGraph(root_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  RtSession session{ge::kInferSessionId};
  ModelLoadArg load_arg(&session, {nullptr, 0});
  EXPECT_EQ(model_executor->Load({}, load_arg), ge::GRAPH_SUCCESS);
  FakeTensors inputs = FakeTensors({4, 4, 4, 4}, 2);
  FakeTensors outputs = FakeTensors({4, 4, 4, 4}, 1);

  GertRuntimeStub stub;
  stub.GetRtsRuntimeStub().Clear();
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper),
            nullptr);
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(), inputs.size(),
                                    outputs.GetTensorList(), outputs.size()), ge::GRAPH_SUCCESS);

  ge::diagnoseSwitch::DisableDumper();
  auto cpu_args = stub.GetRtsRuntimeStub().PopCpuLaunchArgsByKernelName("test_haha");
  ASSERT_NE(cpu_args, nullptr);
  toolkit::aicpu::dump::OpMappingInfo op_mapping_info;
  op_mapping_info.ParseFromString(cpu_args->GetSerializeDumpInfo());
  const auto task_size = op_mapping_info.task_size();
  ASSERT_EQ(task_size, 1);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(BuiltinSubscribersST, HostDump_Ok) {
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0);
  auto model_executor = BuildExecutor();
  RtSession session{ge::kInferSessionId};
  ModelLoadArg load_arg(&session, {nullptr, 0});
  EXPECT_EQ(model_executor->Load({}, load_arg), ge::GRAPH_SUCCESS);
  auto mem_block_1 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto mem_block_2 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto outputs = FakeTensors({2048}, 1, (void *)mem_block_1.get());
  auto inputs = FakeTensors({2048}, 2, (void *)mem_block_2.get());

  GertRuntimeStub stub;
  stub.GetRtsRuntimeStub().Clear();
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper), nullptr);
  ge::diagnoseSwitch::EnableHostDump();
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  dump_properties.SetDumpStep("0|2-5");
  dump_properties.SetDumpPath("/");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // check dump_op
  auto event_log = stub.GetSlogStub().GetLogs();
  std::string expectd_log = "not in the dump step";
  EXPECT_FALSE(CheckLogExpected(event_log, expectd_log));

  //  check turn off dumper
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0);
  EXPECT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper), nullptr);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(BuiltinSubscribersST, HostDump_NotInStep ) {
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0);
  auto model_executor = BuildExecutor();
  RtSession session{ge::kInferSessionId};
  ModelLoadArg load_arg(&session, {nullptr, 0});
  EXPECT_EQ(model_executor->Load({}, load_arg), ge::GRAPH_SUCCESS);
  auto mem_block_1 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto mem_block_2 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto outputs = FakeTensors({2048}, 1, (void *)mem_block_1.get());
  auto inputs = FakeTensors({2048}, 2, (void *)mem_block_2.get());

  GertRuntimeStub stub;
  stub.GetSlogStub().SetLevel(DLOG_WARN);
  stub.GetRtsRuntimeStub().Clear();
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper), nullptr);
  ge::diagnoseSwitch::EnableHostDump();
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  dump_properties.SetDumpStep("1-10|20");
  dump_properties.SetDumpPath("/");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // check dump_op
  auto event_log = stub.GetSlogStub().GetLogs();
  std::string expectd_log = "not in the dump step";
  EXPECT_TRUE(CheckLogExpected(event_log, expectd_log));

  //  check turn off dumper
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0);
  EXPECT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper), nullptr);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(BuiltinSubscribersST, HostDump_Workspace_Ok) {
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0);
  auto model_executor = BuildExecutorSpace();
  RtSession session{ge::kInferSessionId};
  ModelLoadArg load_arg(&session, {nullptr, 0});
  EXPECT_EQ(model_executor->Load({}, load_arg), ge::GRAPH_SUCCESS);
  auto mem_block_1 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto mem_block_2 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto outputs = FakeTensors({2048}, 1, (void *)mem_block_1.get());
  auto inputs = FakeTensors({2048}, 2, (void *)mem_block_2.get());

  GertRuntimeStub stub;
  stub.GetRtsRuntimeStub().Clear();
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  auto dumper =
      model_executor->GetSubscribers().MutableBuiltInSubscriber<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper);
  ASSERT_NE(dumper, nullptr);
  ge::diagnoseSwitch::EnableHostDump();
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  HostExecutorDumper::OnExecuteEvent(0, dumper, kModelStart, nullptr, kStatusSuccess);
  HostExecutorDumper::OnExecuteEvent(0, dumper, kModelStart, nullptr, kStatusSuccess);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // check dump_op
  auto event_log = stub.GetSlogStub().GetLogs();
  std::string expectd_log = "not in the dump step";
  EXPECT_FALSE(CheckLogExpected(event_log, expectd_log));
  //  check turn off dumper
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0);
  EXPECT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper), nullptr);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(BuiltinSubscribersST, ExceptionDump_L0_Ok) {
  ge::diagnoseSwitch::EnableLiteExceptionDump();
  ge::DumpStub::GetInstance().Clear();
  auto model_executor = BuildExecutor();
  GertRuntimeStub runtime_stub;
  RtSession session{ge::kInferSessionId};
  ModelLoadArg load_arg(&session, {nullptr, 0});
  EXPECT_EQ(model_executor->Load({}, load_arg), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);
  auto units = ge::DumpStub::GetInstance().GetDynamicUnits();
  ASSERT_EQ(units.size(), 1);
  ASSERT_EQ(units[0].size(), 24);
  EXPECT_EQ(units[0][0], 8224); // input 1 size
  EXPECT_EQ(units[0][1], 8224); // input 2 size
  EXPECT_EQ(units[0][2], 8224); // output 1 size
  EXPECT_EQ(units[0][3], 4096); // workspace size
  // 中间knownworkspace, 不校验
  EXPECT_EQ(units[0][19], 1); // input 1 dim num
  EXPECT_EQ(units[0][20], 2048); // input 1 dim
  EXPECT_EQ(units[0][21], 1); // input 2 dim nume
  EXPECT_EQ(units[0][22], 2048); // input 2 dim
  EXPECT_EQ(units[0][23], 0); // output 1 dim num
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(BuiltinSubscribersST, DataDump_Ok_ByAclSet) {
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0);
  auto model_executor = BuildExecutor();
  RtSession session{ge::kInferSessionId};
  ModelLoadArg load_arg(&session, {nullptr, 0});
  EXPECT_EQ(model_executor->Load({}, load_arg), ge::GRAPH_SUCCESS);
  auto mem_block_1 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto mem_block_2 = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
  auto outputs = FakeTensors({2048}, 1, (void *)mem_block_1.get());
  auto inputs = FakeTensors({2048}, 2, (void *)mem_block_2.get());

  GertRuntimeStub stub;
  stub.GetRtsRuntimeStub().Clear();
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
  ge::DumpConfig dump_config;
  dump_config.dump_path = "/test";
  dump_config.dump_mode = "all";
  dump_config.dump_status = "on";
  dump_config.dump_op_switch = "on";
  (void)ge::DumpManager::GetInstance().SetDumpConf(dump_config);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // check dump_op
  auto cpu_args = stub.GetRtsRuntimeStub().PopCpuLaunchArgsByKernelName("DumpDataInfo");
  EXPECT_EQ(cpu_args->GetStream(), stream);
  EXPECT_EQ(cpu_args->GetKernelName(), "DumpDataInfo");
  EXPECT_EQ(cpu_args->GetArgSize(), sizeof(aicpu::AicpuParamHead) + (2UL * sizeof(uint64_t)));
  toolkit::aicpu::dump::OpMappingInfo op_mapping_info;
  op_mapping_info.ParseFromString(cpu_args->GetSerializeDumpInfo());
  EXPECT_EQ(op_mapping_info.task_size(), 1UL);
  toolkit::aicpu::dump::Task task = op_mapping_info.task(0);
  EXPECT_EQ(task.input().at(0).address(), reinterpret_cast<uintptr_t>((void *)mem_block_2.get()));
  EXPECT_EQ(task.input().at(1).address(), reinterpret_cast<uintptr_t>((void *)mem_block_2.get()));
  EXPECT_EQ(task.output().at(0).address(), reinterpret_cast<uintptr_t>((void *)mem_block_1.get()));
  EXPECT_EQ(task.input().at(0).shape().dim(0), 2048UL);
  EXPECT_EQ(task.input().at(1).shape().dim(0), 2048UL);
  EXPECT_EQ(task.output().at(0).shape().dim(0), 2048UL);
  //  check turn off dumper
  ge::diagnoseSwitch::MutableDumper().SetEnableFlag(0);
  EXPECT_NE(model_executor->GetSubscribers().GetBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper), nullptr);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
  ge::DumpManager::GetInstance().RemoveDumpProperties(0);
}

TEST_F(BuiltinSubscribersST, DataDump_ReloadTask) {
  ge::diagnoseSwitch::DisableDumper();
  ge::DumpConfig dump_config;
  dump_config.dump_path = "/test";
  dump_config.dump_mode = "all";
  dump_config.dump_status = "on";
  dump_config.dump_op_switch = "on";
  (void)ge::DumpManager::GetInstance().SetDumpConf(dump_config);

  auto graph = ShareGraph::BuildWithKnownSubgraph();
  graph->TopologicalSorting();
  auto ge_root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(ge_root_model);
  GertRuntimeStub fakeRuntime;
  auto global_data = faker.FakeWithoutHandleAiCore("Conv2d", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  auto dumper =
      model_executor->GetSubscribers().MutableBuiltInSubscriber<ExecutorDumper>(BuiltInSubscriberType::kDumper);
  dumper->LoadDumpTaskForDavinciModels(true);
  dumper->LoadDumpTaskForDavinciModels(false);
  ge::DumpManager::GetInstance().RemoveDumpProperties(0);
  ge::diagnoseSwitch::DisableDumper();
}
}  // namespace gert

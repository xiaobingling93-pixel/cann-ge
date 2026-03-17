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
#include <iostream>
#include "faker/fake_value.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "common/share_graph.h"
#include "lowering/graph_converter.h"
#include "faker/global_data_faker.h"
#include "faker/aicpu_taskdef_faker.h"
#include "runtime/model_v2_executor.h"
#include "common/bg_test.h"
#include "runtime/dev.h"

#include "stub/gert_runtime_stub.h"
#include "op_impl/less_important_op_impl.h"
#include "op_impl/transdata/trans_data_positive_source_tc_1010.h"
#include "op_impl/dynamicatomicaddrclean/dynamic_atomic_addr_clean_impl.h"
#include "graph/operator_reg.h"
#include "graph/utils/op_desc_utils.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "check/executor_statistician.h"
#include "graph/ge_local_context.h"
#include "macro_utils/dt_public_scope.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "kernel/common_kernel_impl/tiling.h"

using namespace ge;
namespace gert {
namespace {
UINT32 FakeTilingWithBigWorkspace(gert::KernelContext *context) {
  const auto input_num = context->GetInputNum();
  GE_ASSERT(input_num > 2);
  auto fwk_data = context->GetInputPointer<kernel::TilingFwkData>(input_num - 3);
  GE_ASSERT_NOTNULL(fwk_data);
  auto launch_arg = fwk_data->launch_arg;
  GE_ASSERT_NOTNULL(launch_arg);
  auto tiling_data_av = context->GetOutput(TilingContext::kOutputTilingData);
  GE_ASSERT_NOTNULL(tiling_data_av);
  auto launch_arg_av = context->GetOutput(static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg));
  GE_ASSERT_NOTNULL(launch_arg_av);
  tiling_data_av->Set(&launch_arg->GetTilingData(), nullptr);
  launch_arg_av->Set(launch_arg, nullptr);
  auto tiling_context = reinterpret_cast<gert::TilingContext *>(context);
  auto tiling_data = tiling_context->GetTilingData<uint64_t>();
  tiling_context->SetTilingKey(0);
  tiling_context->SetBlockDim(32);
  auto workspaces = tiling_context->GetWorkspaceSizes(1);
  workspaces[0] = 26850624;
  *tiling_data = 100;
  return 0;
}

std::unique_ptr<gert::Allocators> CreateDefaultAllocators() {
  std::shared_ptr<ge::Allocator> device_allocator(AllocatorFactory::Create(kOnDeviceHbm).release());
  std::shared_ptr<ge::Allocator> host_allocator(AllocatorFactory::Create(kOnHost).release());
  if ((device_allocator == nullptr) || (host_allocator == nullptr)) {
    GELOGE(ge::PARAM_INVALID, "device_allocator is nullptr or host_allocator is nullptr");
  }
  std::unique_ptr<Allocators> allocators = std::make_unique<Allocators>();
  for (size_t i = 0U; i < static_cast<size_t>(kTensorPlacementEnd); ++i) {
    for (size_t j = 0U; j < static_cast<size_t>(AllocatorUsage::kEnd); ++j) {
      if (i == static_cast<size_t>(kOnDeviceHbm)) {
        allocators->SetAllocator(static_cast<TensorPlacement>(i), j, device_allocator);
      } else if (i == static_cast<size_t>(kOnHost) || i == static_cast<size_t>(kFollowing)) {
        allocators->SetAllocator(static_cast<TensorPlacement>(i), j, host_allocator);
      } else {
        GELOGE(ge::PARAM_INVALID, "Unsupported placement %zu to set allocator", i);
      }
    }
  }
  return allocators;
}
}
class Runtime2AllocatorSystemTest : public bg::BgTest {
  void SetUp() {
    memory::RtsCachingMemAllocator::GetAllocator(0, RT_MEMORY_HBM)->Recycle();
    memory::RtsCachingMemAllocator::device_id_to_allocators_.clear();
  }
};

TEST_F(Runtime2AllocatorSystemTest, ExternalAllocator_SingleNodeAiCpuTf_ExecuteSuccess) {
  auto graph = ShareGraph::Aicpu4thGraph();
  graph->FindNode("add1")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpuTf.c_str());
  graph->FindNode("add2")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpuTf.c_str());
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  AiCpuTfTaskDefFaker aicpu_task_def_faker;
  auto ge_root_model = builder.AddTaskDef("Add", aicpu_task_def_faker.SetNeedMemcpy(true)).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  GertRuntimeStub fakeRuntime;
  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048, 1, 1, 1}, 1);
  auto inputs = FakeTensors({2048, 1, 1, 1}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  std::unique_ptr<Allocators> allocators(CreateDefaultAllocators().release());

  ASSERT_EQ(model_executor->Execute({i3.value, allocators.get()}, inputs.GetTensorList(), inputs.size(),
                                    outputs.GetTensorList(), outputs.size()),
            ge::GRAPH_SUCCESS);

  ASSERT_EQ(model_executor->Execute({i3.value, allocators.get()}, inputs.GetTensorList(), inputs.size(),
                                    outputs.GetTensorList(), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(Runtime2AllocatorSystemTest, ExternalAllocator_SingleNodeDataDependency_ExecuteSuccess) {
  auto graph = ShareGraph::BuildDataDependencySingleOpNodeGraph();
  GertRuntimeStub fakeRuntime;

  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  fakeRuntime.GetKernelStub().StubTiling();

  std::unique_ptr<Allocators> allocators(CreateDefaultAllocators().release());
  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  int32_t shape_value = 24;
  std::unique_ptr<uint8_t[]> host_address(new (std::nothrow) uint8_t[512]);
  memcpy_s((void *)(host_address.get()), 512, (void *)(&shape_value), 4);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs0 = FakeTensors({1, 2, 3, 4}, 1);
  auto inputs1 = FakeTensors({1}, 1, (void *)(host_address.get()));
  auto inputs = std::vector<Tensor *>({inputs0.GetTensorList()[0], inputs1.GetTensorList()[0]});

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value, allocators.get()}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  ASSERT_EQ(model_executor->Execute({i3.value, allocators.get()}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(Runtime2AllocatorSystemTest, test_allocate_mem_block_try_recycle_then_malloc_when_mem_cannot_alloc) {
  static uint64_t total_size = 0;
  static uint8_t malloc_count = 0;
  struct FakeRuntime : RuntimeStubImpl {
    rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) {
      malloc_count++;
      total_size += size;
      if (total_size > 32 * 1024UL * 1024UL * 1024UL){
        total_size = (malloc_count == 3) ? 0 : total_size;
        *dev_ptr = nullptr;
        return -1;
      }
      *dev_ptr = new uint8_t [1];
      return RT_ERROR_NONE;
    }
    rtError_t rtFree(void *dev_ptr) {
      delete[](uint8_t *) dev_ptr;
      return RT_ERROR_NONE;
    }
    rtError_t rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t *free, size_t *total) override {
      *free = 60UL * 1024UL * 1024UL * 1024UL;
      *total = 60UL * 1024UL * 1024UL * 1024UL;
      return RT_ERROR_NONE;
    }

  };

  struct FakeAclRuntime : AclRuntimeStubImpl {
    aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total) {
      *free = 60UL * 1024UL * 1024UL * 1024UL;
      *total = 60UL * 1024UL * 1024UL * 1024UL;
      return ACL_SUCCESS;
    }
  };

  GertRuntimeStub stub(std::unique_ptr<RuntimeStubImpl>(new FakeRuntime()), true, std::unique_ptr<AclRuntimeStubImpl>(new FakeAclRuntime()));
  memory::CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  memory::CachingMemAllocator allocator1(0, RT_MEMORY_HBM);
  allocator1.SetStream((void *)1);
  auto mem_block = allocator.Malloc(20 * 1024UL * 1024UL * 1024UL);
  ASSERT_NE(mem_block->GetAddr(), nullptr);
  auto mem_block1 = allocator1.Malloc(15 * 1024UL * 1024UL *1024UL);
  ASSERT_NE(mem_block1->GetAddr(), nullptr);
  mem_block->Free();
  ASSERT_EQ(ge::GRAPH_SUCCESS, allocator.Finalize());
  mem_block1->Free();

  auto alloc_size = PAGE_MEM_SIZE_THRESHOLD_DEFAULT[1U]/2;
  auto span1 = allocator1.Malloc(1024U);
  span1->Free();
  ASSERT_EQ(2, allocator1.GetScalableAllocator()->GetIdleSpanCount());
  span1 = allocator1.Malloc(alloc_size);
  ASSERT_TRUE(span1 == nullptr);
  ASSERT_EQ(ge::GRAPH_SUCCESS, allocator1.Finalize());
  ASSERT_EQ(0, allocator1.GetScalableAllocator()->GetIdleSpanCount());
}

TEST_F(Runtime2AllocatorSystemTest, test_alloc_total_exceed_thresold) {
  gert::memory::RtsCachingMemAllocator allocator(0U, RT_MEMORY_HBM);
  auto span1 = allocator.Malloc(1024UL * 1024UL + 1024UL);
  ASSERT_TRUE(span1 != nullptr);
  const ScalableConfig &cfg = ScalableConfig();
  auto alloc_size = cfg.page_mem_size_total_threshold - 4 * 1024UL*1024UL;
  auto span2 = allocator.Malloc(alloc_size);
  ASSERT_TRUE(span2 != nullptr);
  auto span3 = allocator.Malloc(1024UL * 1024UL);
  ASSERT_TRUE(span3 == nullptr);
  span2->Free();
  span1->Free();
  ASSERT_EQ(ge::GRAPH_SUCCESS, allocator.Finalize());
}

TEST_F(Runtime2AllocatorSystemTest, test_set_memory_pool_threshold) {
  struct FakeRuntime : RuntimeStubImpl {
    rtError_t rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t *free, size_t *total) override {
      *free = 60UL * 1024UL * 1024UL * 1024UL;
      *total = 60UL * 1024UL * 1024UL * 1024UL;
      return RT_ERROR_NONE;
    }
  };

  struct FakeAclRuntime : AclRuntimeStubImpl {
    aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total) override {
      *free = 60UL * 1024UL * 1024UL * 1024UL;
      *total = 60UL * 1024UL * 1024UL * 1024UL;
      return RT_ERROR_NONE;
    }
  };
  GertRuntimeStub fakeRuntime(std::unique_ptr<RuntimeStubImpl>(new FakeRuntime()), true, std::unique_ptr<AclRuntimeStubImpl>(new FakeAclRuntime()));
  ScalableConfig default_cfg;
  constexpr const char *kOptionDisableMemoryPoolThreshold = "ge.experiment.memory_pool_threshold";
  const auto back_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  auto new_options = back_options;
  new_options[kOptionDisableMemoryPoolThreshold] = "4";
  ge::GetThreadLocalContext().SetGlobalOption(new_options);
  ScalableConfig cfg1;
  EXPECT_EQ(cfg1.page_mem_size_total_threshold, 4 * MEM_SIZE_GB);

  new_options[kOptionDisableMemoryPoolThreshold] = "abc";
  ge::GetThreadLocalContext().SetGlobalOption(new_options);
  ScalableConfig cfg2;
  EXPECT_EQ(cfg2.page_mem_size_total_threshold, default_cfg.page_mem_size_total_threshold);

  new_options[kOptionDisableMemoryPoolThreshold] = "0";
  ge::GetThreadLocalContext().SetGlobalOption(new_options);
  ScalableConfig cfg3;
  EXPECT_EQ(cfg3.page_mem_size_total_threshold, default_cfg.page_mem_size_total_threshold);
  ge::GetThreadLocalContext().SetGlobalOption(back_options);
}

TEST_F(Runtime2AllocatorSystemTest, expandable_memory_allocator_fail) {
  class MockRuntime : public ge::RuntimeStub {
   public:
    rtError_t rtMallocPhysical(rtDrvMemHandle* handle, size_t size, rtDrvMemProp_t* prop, uint64_t flags) {
      static size_t cnt = 0U;
      ++cnt;
      if (cnt >= 3U) {
        return -1;
      }
      return 0;
    }
  };
  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 10U * 2U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();

  auto span1 = caching_allocator->Malloc(alloc_size);
  ASSERT_EQ(span1, nullptr);
  caching_allocator.reset(nullptr);
  ge::RuntimeStub::Reset();
}

/**
 * 用例描述：外置allocator场景下，workspace与nodeoutput使用同一个allocator
 *
 * 预置条件：
 * 1. fake 一个单算子子图场景1，需要GE申请中间阶段output内存，完成lowering、tiling等整套实现,graph1使用的是默认的StubTiling函数，
 *    nodeoutput需要内存268500992B, workspace需要4096B
 * 2. fake 一个单算子子图场景2，需要GE申请workspace内存，完成lowering、tiling等整套实现，graph2使用的是自定义的StubTiling函数，
 *    需要workspace需要268500992B
 *
 * 测试步骤：
 * 1. 按照预制条件构造好两张图
 * 2. lowering、加载计算图
 * 3. 先执行单算子子图场景1下发，再执行单算子子图场景2下发
 *
 * 预期结果：
 * 1. 两张图都执行成功
 * 2. allocator申请的内存只有一块，说明workspace与nodeoutput能复用成功执行图中就存在一个大块内存，由于allocator
 *    未暴露出count跟size接口，当前只能从日志中去校验,校验使用内存的size。
 * 3. 后续若你的修改会影响size大小，那可以修改st中校验的size，但是如果影响了内存块数，那说明需要check下你的修改
 */

TEST_F(Runtime2AllocatorSystemTest, ExternalAllocator_NodeoutputAndWorkspaceUseSameAllocator) {

  struct FakeRuntime : RuntimeStubImpl {
    rtError_t rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t *free, size_t *total) override {
      *free = 60UL * 1024UL * 1024UL * 1024UL;
      *total = 60UL * 1024UL * 1024UL * 1024UL;
      return RT_ERROR_NONE;
    }
  };

  struct FakeAclRuntime : AclRuntimeStubImpl {
    aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total) override {
      *free = 60UL * 1024UL * 1024UL * 1024UL;
      *total = 60UL * 1024UL * 1024UL * 1024UL;
      return RT_ERROR_NONE;
    }
  };

  GertRuntimeStub fakeRuntime(std::unique_ptr<RuntimeStubImpl>(new FakeRuntime()), true, std::unique_ptr<AclRuntimeStubImpl>(new FakeAclRuntime()));
  fakeRuntime.GetSlogStub().SetLevel(DLOG_INFO);
  fakeRuntime.GetSlogStub().Clear();
  auto graph1 = ShareGraph::BuildTwoAddNodeGraph();
  auto add1 = graph1->FindNode("add1");
  add1->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);

  graph1->TopologicalSorting();
  GeModelBuilder builder(graph1);
  auto root_model = GeModelBuilder(graph1).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph1, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "AddGraph");

  fakeRuntime.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({64, 64, 128, 128}, 1);
  auto inputs0 = FakeTensors({64, 64, 128, 128}, 1);
  auto inputs1 = FakeTensors({64, 64, 128, 128}, 1);
  auto inputs = std::vector<Tensor *>({inputs0.GetTensorList()[0], inputs1.GetTensorList()[0]});

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  std::unique_ptr<Allocators> allocators(CreateDefaultAllocators().release());

  ASSERT_EQ(model_executor->Execute({i3.value, allocators.get()}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  auto graph2 =
      ShareGraph::BuildSingleNodeGraph("Mul", {{-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}},
                                       {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}},
                                       {{-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}});
  auto mul_node = graph1->FindNode("add1");
  mul_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);

  graph2->TopologicalSorting();
  GeModelBuilder builder2(graph2);
  auto root_model2 = GeModelBuilder(graph2).BuildGeRootModel();
  auto global_data2 = GlobalDataFaker(root_model2).FakeWithHandleAiCore("Mul", false).Build();
  ModelDescHolder model_desc_holder2 = ModelDescHolderFaker().Build();
  model_desc_holder2.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert2 = GraphConverter().SetModelDescHolder(&model_desc_holder2);
  auto exe_graph2 = graph_convert2.ConvertComputeGraphToExecuteGraph(graph2, global_data2);

  // 重新打桩tiling函数，申请一块较大的worspace，与上一张图的nodeoutput复用
  fakeRuntime.GetKernelStub().StubTilingWithCustomTiling(FakeTilingWithBigWorkspace);

  auto model_executor2 = ModelV2Executor::Create(exe_graph2, root_model2);
  ASSERT_NE(model_executor2, nullptr);

  EXPECT_EQ(model_executor2->Load(), ge::GRAPH_SUCCESS);
  auto outputs2 = FakeTensors({64, 64, 128, 128}, 1);
  outputs2.GetTensorList()[0]->SetSize(18014398509481984);
  ASSERT_EQ(model_executor2->Execute({i3.value, allocators.get()}, inputs.data(), inputs.size(),
                                     reinterpret_cast<Tensor **>(outputs2.GetAddrList()), outputs2.size()),
            ge::GRAPH_SUCCESS);

  // 校验日志中存在该行日志，并且行号大于0,如果没有该行日志，行号小于0
  ASSERT_GE(fakeRuntime.GetSlogStub().FindLogEndswith(DLOG_INFO, "[span count:2 page count:4098 total size:268566528]"), 0);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor2->UnLoad(), ge::GRAPH_SUCCESS);

  rtStreamDestroy(stream);
}

/**
 * 用例描述：测试always_external_allocator功能
 *
 * 预置条件：
 * 1. 构造一个小图，需要在GE申请内存，需要用到allocator
 *
 * 测试步骤：
 * 1. 构造一张需要申请包含中间结果内存的小图
 * 2. lowering、加载这张图
 * 3. 将option trust_output_tensor设置为true，执行图
 *
 * 预期结果：
 * 1. 加载完的图只有GetAllocator节点，不会有SelectAllocator与CreateAllocator节点
 * 2. 执行成功
 */
TEST_F(Runtime2AllocatorSystemTest, ExternalAllocator_AllwaysExternalAllocator) {
  auto graph = ShareGraph::BuildTwoAddNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .BuildGeRootModel();

  LoweringOption option;
  option.always_external_allocator = true;
  ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "AlwaysExternalAllocatorGraph");

  std::unique_ptr<Allocators> allocators(CreateDefaultAllocators().release());
  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  ess->Clear();

  auto output = TensorFaker().StorageShape({8, 1, 224, 224, 16}).OriginShape({8, 3, 224, 224}).Build();
  std::vector<Tensor *> outputs = {output.GetTensor()};
  auto inputs = FakeTensors({1, 2, 3, 4}, 2);
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Load({i3.value, allocators.get()}), ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->Execute({i3.value, allocators.get()}, inputs.GetTensorList(), inputs.size(), outputs.data(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Data", "GetExternalL1Allocator"), 1);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Data", "CreateL1Allocator"), 0);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Data", "SelectL1Allocator"), 1);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
  ge::RuntimeStub::Reset();
}

TEST_F(Runtime2AllocatorSystemTest, expandable_memory_allocator_hole) {
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 12U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();

  auto span1 = caching_allocator->Malloc(alloc_size);
  // va |--11|1111|
  ASSERT_NE(span1, nullptr);
  auto span2 = caching_allocator->Malloc(4U * 1024U *1024U);
  // va |2211|1111|
  ASSERT_NE(span2, nullptr);

  span1->Free();
  // va |22--|----|
  auto span3 = caching_allocator->Malloc(24U * 1024U * 1024U);
  // new_va |3333|3333|3333|
  // va |3333|3333|22--|3333|
  auto addr3 = span3->GetAddr();
  auto span4 = caching_allocator->Malloc(4U * 1024U * 1024U);
  // new_va |3333|3333|3333|
  // va |3333|3333|2244|3333|
  span3->Free();
  // new_va |----|----|----|
  // va |----|----|2244|----|
  auto span5 = caching_allocator->Malloc(16U * 1024U * 1024U);
  // new_va |5555|5555|----|
  // va |5555|5555|2244|----|
  auto span6 = caching_allocator->Malloc(24U * 1024U * 1024U);
  // new_va |5555|5555|----|
  // new_va |6666|6666|6666|
  // va |6666|6666|5555|5555|2244|6666|
  auto addr6 = span5->GetAddr();
  ASSERT_NE(addr3, addr6);

  ASSERT_EQ(scalable_allocator->GetReachTheoryRate(), static_cast<float>(ge::kRatioBase));
  span2->Free();
  span4->Free();
  span5->Free();
  span6->Free();
  ASSERT_EQ(scalable_allocator->GetStatics().empty(), false);
  scalable_allocator->Recycle();
  scalable_allocator->Finalize(false);
  caching_allocator.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(Runtime2AllocatorSystemTest, expandable_memory_allocator_hole_12M) {
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 12U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();

  auto span1 = caching_allocator->Malloc(alloc_size);
  // va |--11|1111|
  ASSERT_NE(span1, nullptr);

  auto span2 = caching_allocator->Malloc(10U * 1024U *1024U);
  // va |-222|2211|1111|
  ASSERT_NE(span2, nullptr);

  auto addr2 = span2->GetAddr();
  span1->Free(); // 12M
  // va |-222|22--|----|

  auto span3 = caching_allocator->Malloc(alloc_size * 2); // 36M
  // new_va |3333|3333|3333|
  // va |3333|3333|-222|22--|3333|

  auto addr3 = span3->GetAddr();
  ASSERT_NE(addr2, addr3);
  span2->Free(); // 24M
  // new_va |3333|3333|3333|
  // va |3333|3333|----|----|3333|

  auto span4 = caching_allocator->Malloc(32U * 1024U * 1024U); // 56M
  // new_va |3333|3333|3333|
  // va |4444|4444|3333|3333|4444|4444|3333|

  auto addr4 = span4->GetAddr();
  ASSERT_NE(addr4, addr2);
  span3->Free();// 32M
  // new_va |----|----|----|
  // va |4444|4444|----|----|4444|4444|----|

  auto span5 = caching_allocator->Malloc(alloc_size * 2); // 56M
  // new_va |5555|5555|5555|
  // va |4444|4444|5555|5555|4444|4444|5555|

  auto addr5 = span5->GetAddr();
  ASSERT_EQ(addr5, addr3);
  ASSERT_NE(static_cast<const uint8_t *>(span5->GetAddr()),
            static_cast<const uint8_t *>(span4->GetAddr()) + 16 * 1024U *1024U);
  ASSERT_EQ(scalable_allocator->GetReachTheoryRate(), static_cast<float>(ge::kRatioBase));
  span5->Free();
  span4->Free();
  ASSERT_EQ(scalable_allocator->GetStatics().empty(), false);
  scalable_allocator->Finalize(false);
  caching_allocator.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(Runtime2AllocatorSystemTest, expandable_memory_allocator_add_page_record_failed) {
  dlog_setlevel(GE_MODULE_NAME, 1, 0); // 为了能走到pa va检查
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 8U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator(10086);
  auto scalable_allocator = caching_allocator->GetScalableAllocator();

  // step1 malloc 3 pa, free
  /*
   *     ┌──────────────────┐
   * va  │span1 free        │
   *     └──▲─────▲──────▲──┘
   *        │     │      │
   *        │     │      │
   *     ┌──┼─┐ ┌─┼──┐ ┌─┼──┐
   * pa  │  2 │ │ 1  │ │ 0  │
   *     └────┘ └────┘ └────┘
   */
  GELOGI("==========================span1 malloc 24M begin");
  auto span1 = caching_allocator->Malloc(alloc_size * 3);
  GELOGI("==========================span1 malloc 24M success");
  GELOGI("==========================span1 free 24M begin");
  span1->Free();
  GELOGI("==========================span1 free 24M success");

  // step2 Injection error
  auto &physical_allocator = scalable_allocator->device_allocator_.GetExpandableAllocator()
      .GetPhysicalMemoryAllocator();
  ge::PageRecord error_page_record{(const uint8_t *)0x123, 0U, alloc_size, (const uint8_t *)0x123, alloc_size};
  physical_allocator.AddPageRecord(0, error_page_record);

  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().Clear();
  auto new_span1 = caching_allocator->Malloc(alloc_size * 3);
  ASSERT_EQ(new_span1, nullptr);
  runtime_stub.GetSlogStub().FindErrorLogEndsWith("ProcPageRecord: ErrorNo: 4294967295(failed) virtual and physical page mapping check failed");

  // release
  scalable_allocator->Recycle();
  scalable_allocator->Finalize(false);
  caching_allocator.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  dlog_setlevel(GE_MODULE_NAME, 3, 0); // 为了能走到pa va检查
}

TEST_F(Runtime2AllocatorSystemTest, expandable_memory_allocator_ProcPageRecordByPaList_failed) {
  dlog_setlevel(GE_MODULE_NAME, 1, 0); // 为了能走到pa va检查
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 8U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator(10086);
  auto scalable_allocator = caching_allocator->GetScalableAllocator();

  auto span1 = caching_allocator->Malloc(alloc_size);
  // va |1111|
  ASSERT_NE(span1, nullptr);

  auto span2 = caching_allocator->Malloc(alloc_size);
  // va |2222|1111|
  ASSERT_NE(span2, nullptr);

  auto span3 = caching_allocator->Malloc(alloc_size);
  ASSERT_NE(span3, nullptr);
  // va |3333|2222|1111|

  span3->Free();
  span1->Free();
  // va |----|2222|----|

  // Injection error
  auto &physical_allocator = scalable_allocator->device_allocator_.GetExpandableAllocator()
      .GetPhysicalMemoryAllocator();
  ge::PageRecord error_page_record{(const uint8_t *)0x123, 0U, alloc_size, (const uint8_t *)0x123, alloc_size};
  physical_allocator.AddPageRecord(0, error_page_record);

  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().Clear();
  auto span4 = caching_allocator->Malloc(alloc_size * 2U);
  ASSERT_EQ(span4, nullptr);
  runtime_stub.GetSlogStub().FindErrorLogEndsWith("ProcPageRecordByPaList: ErrorNo: 4294967295(failed) virtual and physical page mapping check failed");

  scalable_allocator->Recycle();
  span2->Free();
  scalable_allocator->Finalize(false);
  caching_allocator.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  dlog_setlevel(GE_MODULE_NAME, 3, 0); // 为了能走到pa va检查
}

}  // namespace gert
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
#include "macro_utils/dt_public_scope.h"
#include "faker/space_registry_faker.h"
#include "common/model/external_allocator_manager.h"
#include "hybrid/executor/hybrid_model_async_executor.h"
#include "hybrid/executor/hybrid_model_rt_v2_executor.h"
#include "graph_builder/converter_checker.h"
#include "graph_builder/bg_memory.h"
#include "graph/ge_context.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/manager/host_mem_manager.h"
#include "graph/types.h"
#include "graph/manager/mem_manager.h"
#include "graph/manager/graph_external_weight_manager.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "common/share_graph.h"
#include "register/node_converter_registry.h"
#include "op_impl/less_important_op_impl.h"
#include "faker/ge_model_builder.h"
#include "faker/magic_ops.h"
#include "depends/profiler/src/profiling_auto_checker.h"
#include "common/global_variables/diagnose_switch.h"
#include "graph_metadef/depends/checker/tensor_check_utils.h"
#include "runtime/subscriber/built_in_subscriber_definitions.h"
#include "runtime/subscriber/global_dumper.h"
#include "rt_error_codes.h"

using namespace ge;
using namespace hybrid;
using namespace gert;
namespace {
LowerResult LoweringFooWithStreamSync(const ge::NodePtr &node, const LowerInput &lower_input) {
  size_t output_size = 512U;
  gert::StorageShape shape;
  auto size_holder = bg::ValueHolder::CreateConst(&output_size, sizeof(output_size));
  auto output_addrs = bg::AllocOutputMemory(kOnDeviceHbm, node, {size_holder}, *(lower_input.global_data));
  auto compute_holder = bg::ValueHolder::CreateVoid<bg::ValueHolder>("SyncStream", {lower_input.global_data->GetStream()});

  return {HyperStatus::Success(), {compute_holder}, {lower_input.input_shapes[0]}, output_addrs};
}

std::vector<gert::Tensor> InputData2GertTensors(const InputData &input_data) {
  std::vector<gert::Tensor> input_tensors;
  for (size_t i = 0U; i < input_data.blobs.size(); ++i) {
    gert::Tensor tensor;
    tensor.MutableTensorData().SetSize(input_data.blobs[i].length);
    tensor.MutableTensorData().SetAddr(input_data.blobs[i].data, nullptr);
    tensor.MutableStorageShape().SetDimNum(input_data.shapes[i].size());
    for (size_t j = 0U; j < input_data.shapes[i].size(); ++j) {
      tensor.MutableStorageShape().SetDim(0, input_data.shapes[i][j]);
    }
    input_tensors.emplace_back(std::move(tensor));
  }
  return input_tensors;
}
void TestHybridModelExecuteWithIterationLoop() {
  auto graph = ShareGraph::SimpleFooGraph();
  graph->SetNeedIteration(true);
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Foo") {
      MockLessImportantNodeKernel(node);
    }
  }
  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput->GetOpDesc()->SetSrcName({"foo"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  const size_t kPipeLoop = 10;
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  std::map<std::string, std::string> options;
  options[ge::ATTR_NAME_ITERATORS_PER_LOOP] = std::to_string(kPipeLoop);
  ge::GetThreadLocalContext().SetGraphOption(options);

  rtStream_t stream = nullptr;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);

  std::string iterations_loop;
  ge::GetThreadLocalContext().GetOption(ge::ATTR_NAME_ITERATORS_PER_LOOP, iterations_loop);
  EXPECT_EQ(iterations_loop, std::to_string(kPipeLoop));

  InputData inputs;
  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[3072]);
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 3072, false));
  inputs.shapes.push_back({1, 16, 16, 3});
  HybridModelExecutor::ExecuteArgs args;

  auto desc = MakeShared<GeTensorDesc>();
  GeShape geshape({2, 2, 2, 2});
  desc->SetShape(geshape);

  auto allocator = NpuMemoryAllocator::GetAllocator();
  auto tensor_buffer = TensorBuffer::Create(allocator, 100);
  auto output_tensor = TensorValue(shared_ptr<TensorBuffer>(tensor_buffer.release()));
  args.outputs.emplace_back(output_tensor);
  args.output_desc.emplace_back(desc);

  EXPECT_EQ(executor_rt_v2.Execute(inputs, args), SUCCESS);
  EXPECT_EQ(executor_rt_v2.run_ctx_.iterations_per_loop_, kPipeLoop);
}

REGISTER_NODE_CONVERTER("_lower_foo_with_stream_sync", LoweringFooWithStreamSync);

void SetDefaultOutputTensorDesc(const ge::NodePtr &file_constant) {
  file_constant->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_ND);
  file_constant->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_ND);
  file_constant->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({5, 5}));
  file_constant->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({1, 5, 5}));
  file_constant->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_INT32);
  file_constant->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_INT32);
}
void SetDefaultAttr(const ge::NodePtr &file_constant, const std::string &file_path_config = "test_weight_convert.bin") {
  // set attr
  std::vector<int64_t> shape = {5, 5};
  std::vector<int64_t> original_shape = {1, 5, 5};
  ge::AttrUtils::SetInt(file_constant->GetOpDesc(), "offset", 0);
  ge::AttrUtils::SetInt(file_constant->GetOpDesc(), "length", 0);
  ge::AttrUtils::SetStr(file_constant->GetOpDesc(), "location", file_path_config);
  ge::AttrUtils::SetStr(file_constant->GetOpDesc(), "file_path", "");
  ge::AttrUtils::SetStr(file_constant->GetOpDesc(), "file_id", "");
  ge::AttrUtils::SetDataType(file_constant->GetOpDesc(), "dtype", DT_INT32);
  ge::AttrUtils::SetListInt(file_constant->GetOpDesc(), "shape", shape);
  ge::AttrUtils::SetListInt(file_constant->GetOpDesc(), "original_shape", original_shape);
}

/*
 *            netoutput
 *            /      \
 * file_constant_0  file_constant_1
 */
ComputeGraphPtr Build2FileConstantGraph(const std::vector<std::string> &file_path_config) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("file_constant_0", "FileConstant")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("file_constant_1", "FileConstant")->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g1);
  GE_ASSERT_TRUE(file_path_config.size() == 2U);
  auto file_constant_0 = graph->FindNode("file_constant_0");
  SetDefaultOutputTensorDesc(file_constant_0);
  SetDefaultAttr(file_constant_0, file_path_config[0]);
  auto file_constant_1 = graph->FindNode("file_constant_1");
  SetDefaultOutputTensorDesc(file_constant_1);
  SetDefaultAttr(file_constant_1, file_path_config[1]);

  auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"file_constant_0", "file_constant_1"});
  net_output->GetOpDesc()->SetSrcIndex({0, 1});
  return graph;
}

ge::Status CreateFileConstantFile(const std::string &file_dir, const std::string &file_name, const size_t output_size) {
  if (!file_dir.empty()) {
    GE_ASSERT_TRUE(ge::CreateDirectory(file_dir) == 0);
  }
  std::unique_ptr<int32_t[]> int32_t_buf(new int32_t[output_size / sizeof(int32_t)]);
  for (size_t i = 0U; i < output_size / sizeof(int32_t); ++i) {
    int32_t_buf[i] = i;
  }
  const auto file_constant_file = file_dir + file_name;
  std::ofstream out1(file_constant_file, std::ios::binary);
  GE_ASSERT_TRUE(out1.is_open());
  out1.write((char *)int32_t_buf.get(), output_size);
  out1.close();
  return ge::SUCCESS;
}

std::shared_ptr<HybridModel> FakeHybridModel(const bool is_exec_on_host,
                                             const std::vector<std::string> &location_config,
                                             ge::ComputeGraphPtr &compute_graph, const uint64_t session_id) {
  std::map<std::string, std::string> options;
  if (is_exec_on_host) {
    options["ge.exec.placement"] = "HOST";
  } else {
    options["ge.exec.placement"] = "DEVICE";
  }
  ge::GetThreadLocalContext().SetGraphOption(options);

  compute_graph = Build2FileConstantGraph(location_config);
  GE_ASSERT_NOTNULL(compute_graph);
  auto file_constant_0 = compute_graph->FindNode("file_constant_0");
  auto file_constant_1 = compute_graph->FindNode("file_constant_1");
  GE_ASSERT_NOTNULL(file_constant_0);
  GE_ASSERT_NOTNULL(file_constant_1);
  int64_t aligned_mem_size = 0U;
  ge::TensorUtils::GetTensorMemorySizeInBytes(file_constant_0->GetOpDesc()->GetOutputDesc(0U), aligned_mem_size);
  ge::TensorUtils::SetSize(*file_constant_0->GetOpDesc()->MutableOutputDesc(0U), aligned_mem_size);
  ge::TensorUtils::GetTensorMemorySizeInBytes(file_constant_1->GetOpDesc()->GetOutputDesc(0U), aligned_mem_size);
  ge::TensorUtils::SetSize(*file_constant_1->GetOpDesc()->MutableOutputDesc(0U), aligned_mem_size);
  for (const auto &location : location_config) {
    GE_ASSERT_SUCCESS(CreateFileConstantFile("", location, 5 * 5 * sizeof(int32_t)));
  }

  compute_graph->TopologicalSorting();
  GeModelBuilder builder(compute_graph);
  auto ge_root_model = builder.BuildGeRootModel();
  for (const auto &it : ge_root_model->GetSubgraphInstanceNameToModel()) {
    AttrUtils::SetInt(it.second, ATTR_MODEL_VAR_SIZE, 2048);
  }

  auto hybrid_model = std::make_shared<HybridModel>(ge_root_model);
  GE_ASSERT_NOTNULL(hybrid_model);
  hybrid_model->root_graph_item_.reset(new GraphItem);
  hybrid_model->root_graph_ = ge_root_model->GetRootGraph();
  GE_ASSERT_SUCCESS(hybrid_model->Init());
  GE_ASSERT_TRUE(hybrid_model->execute_by_rt_v2_);

  GE_ASSERT_NOTNULL(hybrid_model->ge_root_model_);
  AttrUtils::SetInt(hybrid_model->ge_root_model_->GetSubgraphInstanceNameToModel().begin()->second,
                    MODEL_ATTR_SESSION_ID, session_id);
  hybrid_model->root_runtime_param_.session_id = session_id;
  return hybrid_model;
}
class ExternalAllocatorUtStub : public Allocator {
 public:
  MemBlock *Malloc(size_t size) override {
    void *mem = nullptr;
    (void)rtMalloc(&mem, size, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
    malloc_cnt++;
    return new (std::nothrow) MemBlock(*this, mem, size);
  }
  MemBlock *MallocAdvise(size_t size, void *addr) override {
    malloc_advise_cnt++;
    return Malloc(size);
  }
  void Free(MemBlock *block) override {
    if (block != nullptr) {
      rtFree(block->GetAddr());
      free_cnt++;
      delete block;
    }
  }
  uint32_t GetMallocCnt() {
    return malloc_cnt;
  }
  uint32_t GetMallocAdviseCnt() {
    return malloc_advise_cnt;
  }
  uint32_t GetFreeCnt() {
    return free_cnt;
  }
 private:
  uint32_t malloc_cnt = 0;
  uint32_t malloc_advise_cnt = 0;
  uint32_t free_cnt = 0;

};
}
class UtestHybridRt2Executor : public testing::Test {
 protected:
  void SetUp() {
    RTS_STUB_SETUP();
  }
  void TearDown() {
    RTS_STUB_TEARDOWN();
  }
 public:
  static void TestRunCtxInitWithTwoFileConstants(const std::vector<std::string> &location_config,
                                                 const bool is_exec_on_host, const uint64_t session_id = 0U,
                                                 const bool is_test_two_session = false) {
    ge::ComputeGraphPtr compute_graph = nullptr;
    auto hybrid_model = FakeHybridModel(is_exec_on_host, location_config, compute_graph, session_id);

    // Prepare global var manager resource
    VarManagerPool::Instance().Destory();
    ExternalWeightManagerPool::Instance().Destroy();
    auto var_manager = VarManager::Instance(hybrid_model->GetSessionId());
    auto weight_manager = ExternalWeightManagerPool::Instance().GetManager(hybrid_model->GetSessionId());
    ge::ScopeGuard guarder([&var_manager, &weight_manager, &is_test_two_session]() {
      var_manager->FreeVarMemory();
      // 如果想测试session间相同权重文件会再次加载，需要保留第1个session的weight_manager信息，清除掉第1个session的var_manager信息
      // 如果第2个session没有再次加载，则权重值将为随机数
      if (!is_test_two_session) {
        weight_manager->Finalize();
      }
    });
    ASSERT_EQ(var_manager->Init(0, hybrid_model->GetSessionId(), 0, 0), SUCCESS);

    HybridModelRtV2Executor::RunCtx run_ctx;
    ASSERT_EQ(run_ctx.Init(hybrid_model.get()), SUCCESS);
    gert::StorageShape shape_0, shape_1;
    gert::TensorData memory_0, memory_1;
    auto file_constant_0 = compute_graph->FindNode("file_constant_0");
    auto file_constant_1 = compute_graph->FindNode("file_constant_1");
    ASSERT_NE(file_constant_0, nullptr);
    ASSERT_NE(file_constant_1, nullptr);
    ASSERT_EQ(run_ctx.graph_var_visitor_.GetVarShapeAndMemory(file_constant_0->GetName(), shape_0, memory_0), SUCCESS);
    ASSERT_EQ(run_ctx.graph_var_visitor_.GetVarShapeAndMemory(file_constant_1->GetName(), shape_1, memory_1), SUCCESS);
    ASSERT_EQ(shape_0.GetOriginShape().GetDimNum(), 3U);
    EXPECT_EQ(shape_0.GetOriginShape().GetDim(0), 1U);
    EXPECT_EQ(shape_0.GetOriginShape().GetDim(1), 5U);
    EXPECT_EQ(shape_0.GetOriginShape().GetDim(2), 5U);
    ASSERT_EQ(shape_0.GetStorageShape().GetDimNum(), 2U);
    EXPECT_EQ(shape_0.GetStorageShape().GetDim(0), 5U);
    EXPECT_EQ(shape_0.GetStorageShape().GetDim(1), 5U);
    EXPECT_EQ(shape_0, shape_1);

    if (location_config[0] == location_config[1]) {
      // fileconstant节点location相同, 内存共享
      EXPECT_TRUE(memory_0.IsSharedWith(memory_1));
      ASSERT_TRUE(memory_0.GetSize() >= 5 * 5 * sizeof(int32_t));
      for (size_t idx = 0U; idx < 5 * 5; ++idx) {
        EXPECT_EQ(static_cast<int32_t *>(memory_0.GetAddr())[idx], idx);
      }
      system(std::string("rm -rf ").append(location_config[0]).c_str());
    } else {
      // fileconstant节点location不相同
      EXPECT_FALSE(memory_0.IsSharedWith(memory_1));
      ASSERT_TRUE(memory_0.GetSize() >= 5 * 5 * sizeof(int32_t));
      ASSERT_TRUE(memory_1.GetSize() >= 5 * 5 * sizeof(int32_t));
      for (size_t idx = 0U; idx < 5 * 5; ++idx) {
        EXPECT_EQ(static_cast<int32_t *>(memory_0.GetAddr())[idx], idx);
      }
      for (size_t idx = 0U; idx < 5 * 5; ++idx) {
        EXPECT_EQ(static_cast<int32_t *>(memory_1.GetAddr())[idx], idx);
      }
      system(std::string("rm -rf ").append(location_config[0]).c_str());
      system(std::string("rm -rf ").append(location_config[1]).c_str());
    }
  }
};

class Listener : public ModelListener {
 public:
  Listener(std::function<void()> done) : done_(done) {}
  Status OnComputeDone(uint32_t model_id, uint32_t data_index, uint32_t result_code,
                       std::vector<gert::Tensor> &outputs) override {
    done_();
    return SUCCESS;
  }
  std::function<void()> done_;
};

void HybridRt2ExecutorRunImpl() {
  auto graph = ShareGraph::SimpleFooGraph();
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Foo") {
      MockLessImportantNodeKernel(node);
    }
  }
  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput->GetOpDesc()->SetSrcName({"foo"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  HybridModelAsyncExecutor executor(&hybrid_model);
  EXPECT_EQ(executor.Init(), SUCCESS);
  EXPECT_NE(executor.executor_, nullptr);

  bool reached = false;
  std::shared_ptr<ModelListener> listener = std::make_shared<Listener>([&reached]() { reached = true; });
  executor.Start(listener);
  HybridModelRtV2Executor *executor_rt_v2 = reinterpret_cast<HybridModelRtV2Executor *>(executor.executor_.get());

  InputData inputs;
  inputs.blobs.resize(executor_rt_v2->num_inputs_);
  inputs.shapes.resize(executor_rt_v2->num_inputs_);
  auto data = std::make_shared<RunArgs>();
  data->input_tensor = std::move(InputData2GertTensors(inputs));
  executor.EnqueueData(data);

  size_t kMaxWaitSeconds = 5U;
  for (size_t seconds_wait = 0U; seconds_wait < kMaxWaitSeconds; seconds_wait++) {
    if (reached) {
      break;
    }
    sleep(1);
  }
  executor.Stop();
}

TEST_F(UtestHybridRt2Executor, run_success) {
  HybridRt2ExecutorRunImpl();
}

namespace {
template <typename T>
struct GeType {};

template <>
struct GeType<float> {
  static ge::DataType type;
};
ge::DataType GeType<float >::type = ge::DT_FLOAT;

template <>
struct GeType<int32_t> {
  static ge::DataType type;
};

template <typename T>
GeTensorPtr MakeScalarTensor(const T &v) {
  auto desc = ge::GeTensorDesc(GeShape(), ge::FORMAT_ND, GeType<T>::type);
  return std::make_shared<GeTensor>(desc, reinterpret_cast<const uint8_t*>(&v), sizeof(T));
}
}

TEST_F(UtestHybridRt2Executor, context_init_with_device_variable_success) {
  auto graph = ShareGraph::SimpleVariableGraph();

  GeTensorPtr weight = MakeScalarTensor(2.0f);
  ASSERT_NE(weight, nullptr);

  auto constant = graph->FindFirstNodeMatchType("Constant");
  auto variable = graph->FindFirstNodeMatchType("Variable");
  ASSERT_NE(constant, nullptr);
  ASSERT_NE(variable, nullptr);

  auto constant_desc = constant->GetOpDesc()->MutableOutputDesc(0);
  auto variable_desc = variable->GetOpDesc()->MutableOutputDesc(0);

  constant_desc->SetShape(GeShape(std::vector<int64_t>{8, 3, 64, 64}));
  constant_desc->SetOriginShape(GeShape(std::vector<int64_t>{8, 3, 64, 64}));
  constant_desc->SetDataType(ge::DT_FLOAT);
  variable_desc->SetShape(GeShape(std::vector<int64_t>{8, 1, 64, 64, 16}));
  variable_desc->SetOriginShape(GeShape(std::vector<int64_t>{8, 3, 64, 64}));
  variable_desc->SetDataType(ge::DT_FLOAT);
  variable_desc->SetPlacement(ge::Placement::kPlacementDevice);

  ge::AttrUtils::SetTensor(constant->GetOpDesc(), ATTR_NAME_WEIGHTS, weight);

  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput->GetOpDesc()->SetSrcName({"foo"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();
  for (const auto &it : ge_root_model->GetSubgraphInstanceNameToModel()) {
    AttrUtils::SetInt(it.second, ATTR_MODEL_VAR_SIZE, 2048);
  }
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();

  // Prepare global var manager resource
  auto var_manager = VarManager::Instance(hybrid_model.GetSessionId());
  ASSERT_EQ(var_manager->Init(0, hybrid_model.GetSessionId(), 0, 0), SUCCESS);

  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  HybridModelRtV2Executor::RunCtx run_ctx;
  ASSERT_EQ(run_ctx.Init(&hybrid_model), SUCCESS);
  ge::ScopeGuard guarder([&var_manager]() { var_manager->FreeVarMemory(); });
  gert::StorageShape shape;
  gert::TensorData memory;
  ASSERT_EQ(run_ctx.graph_var_visitor_.GetVarShapeAndMemory(variable->GetName(), shape, memory), SUCCESS);

  ASSERT_EQ(memory.GetPlacement(), gert::TensorPlacement::kOnDeviceHbm);
  ASSERT_EQ(shape.GetStorageShape().GetDimNum(), 5U);
  ASSERT_EQ(shape.GetStorageShape().GetDim(0), 8U);
  ASSERT_EQ(shape.GetStorageShape().GetDim(1), 1U);
  ASSERT_EQ(shape.GetStorageShape().GetDim(2), 64U);
  ASSERT_EQ(shape.GetStorageShape().GetDim(3), 64U);
  ASSERT_EQ(shape.GetStorageShape().GetDim(4), 16U);

  ASSERT_EQ(shape.GetOriginShape().GetDimNum(), 4U);
  ASSERT_EQ(shape.GetOriginShape().GetDim(0), 8U);
  ASSERT_EQ(shape.GetOriginShape().GetDim(1), 3U);
  ASSERT_EQ(shape.GetOriginShape().GetDim(2), 64U);
  ASSERT_EQ(shape.GetOriginShape().GetDim(3), 64U);

  ASSERT_EQ(memory.GetPlacement(), gert::TensorPlacement::kOnDeviceHbm);
  ASSERT_NE(memory.GetAddr(), nullptr);
}

TEST_F(UtestHybridRt2Executor, context_init_with_host_constant_share_in_session_success) {
  // make option as host exec
  std::map<std::string, std::string> options;
  options["ge.exec.placement"] = "HOST";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ASSERT_TRUE(ge::GetContext().GetHostExecFlag());

  // prepare graph
  auto graph = ShareGraph::SimpleVariableGraph();
  GeTensorPtr weight = MakeScalarTensor(2.0f);
  ASSERT_NE(weight, nullptr);

  auto constant = graph->FindFirstNodeMatchType("Constant");
  auto variable = graph->FindFirstNodeMatchType("Variable");
  ASSERT_NE(constant, nullptr);
  ASSERT_NE(variable, nullptr);
  ge::AttrUtils::SetStr(variable->GetOpDesc(), ge::ATTR_VARIABLE_PLACEMENT, "host");

  auto constant_desc = constant->GetOpDesc()->MutableOutputDesc(0);
  ge::TensorUtils::SetSize(*(constant_desc), weight->GetData().GetSize());
  auto variable_desc = variable->GetOpDesc()->MutableOutputDesc(0);

  constant_desc->SetShape(GeShape(std::vector<int64_t>{8, 3, 64, 64}));
  constant_desc->SetOriginShape(GeShape(std::vector<int64_t>{8, 3, 64, 64}));
  constant_desc->SetDataType(ge::DT_FLOAT);
  variable_desc->SetShape(GeShape(std::vector<int64_t>{8, 1, 64, 64, 16}));
  variable_desc->SetOriginShape(GeShape(std::vector<int64_t>{8, 3, 64, 64}));
  variable_desc->SetDataType(ge::DT_FLOAT);
  variable_desc->SetPlacement(ge::Placement::kPlacementDevice);
  int64_t real_dim_size = 0L;
  ASSERT_EQ(ge::TensorUtils::GetTensorSizeInBytes(*variable_desc, real_dim_size), GRAPH_SUCCESS);
  ge::TensorUtils::SetSize(*variable_desc, real_dim_size);

  ge::AttrUtils::SetTensor(constant->GetOpDesc(), ATTR_NAME_WEIGHTS, weight);

  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput->GetOpDesc()->SetSrcName({"foo"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  graph->TopologicalSorting();

  // build model
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();
  for (const auto &it : ge_root_model->GetSubgraphInstanceNameToModel()) {
    AttrUtils::SetInt(it.second, ATTR_MODEL_VAR_SIZE, 2048);
  }
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();

  // Prepare global var manager resource
  VarManagerPool::Instance().Destory();
  auto var_manager = VarManager::Instance(hybrid_model.GetSessionId());
  ASSERT_EQ(var_manager->Init(0, hybrid_model.GetSessionId(), 0, 0), SUCCESS);

  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  HybridModelRtV2Executor::RunCtx run_ctx;
  ASSERT_EQ(run_ctx.Init(&hybrid_model), SUCCESS);
  ge::ScopeGuard guarder([&var_manager]() { var_manager->FreeVarMemory(); });
  gert::StorageShape shape;
  gert::TensorData memory;
  ASSERT_EQ(run_ctx.graph_var_visitor_.GetVarShapeAndMemory(constant->GetName(), shape, memory), SUCCESS);

  ASSERT_EQ(shape.GetStorageShape().GetDimNum(), 4U);
  ASSERT_EQ(shape.GetStorageShape().GetDim(0), 8U);
  ASSERT_EQ(shape.GetStorageShape().GetDim(1), 3U);
  ASSERT_EQ(shape.GetStorageShape().GetDim(2), 64U);
  ASSERT_EQ(shape.GetStorageShape().GetDim(3), 64U);

  ASSERT_EQ(shape.GetOriginShape().GetDimNum(), 4U);
  ASSERT_EQ(shape.GetOriginShape().GetDim(0), 8U);
  ASSERT_EQ(shape.GetOriginShape().GetDim(1), 3U);
  ASSERT_EQ(shape.GetOriginShape().GetDim(2), 64U);
  ASSERT_EQ(shape.GetOriginShape().GetDim(3), 64U);

  ASSERT_EQ(memory.GetPlacement(), gert::TensorPlacement::kOnHost);
  ASSERT_NE(memory.GetAddr(), nullptr);
  auto constant_addr = memory.GetAddr();

   // build model2
  GeModelBuilder builder2(graph);
  auto ge_root_model2 = builder2.BuildGeRootModel();
  for (const auto &it : ge_root_model2->GetSubgraphInstanceNameToModel()) {
    AttrUtils::SetInt(it.second, ATTR_MODEL_VAR_SIZE, 2048);
  }
  HybridModel hybrid_model2(ge_root_model2);
  hybrid_model2.root_graph_item_.reset(new GraphItem);
  hybrid_model2.root_graph_ = ge_root_model2->GetRootGraph();

  EXPECT_EQ(hybrid_model2.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model2.execute_by_rt_v2_);

  HybridModelRtV2Executor::RunCtx run_ctx2;
  ASSERT_EQ(run_ctx2.Init(&hybrid_model2), SUCCESS);
  gert::StorageShape shape2;
  gert::TensorData memory2;
  ASSERT_EQ(run_ctx2.graph_var_visitor_.GetVarShapeAndMemory(constant->GetName(), shape2, memory2), SUCCESS);
  // constant in two graph share same host addr
  ASSERT_EQ(memory2.GetAddr(), constant_addr);

  std::string placement;
  (void)AttrUtils::GetStr(constant->GetOpDescBarePtr(), ge::ATTR_VARIABLE_PLACEMENT, placement);
  EXPECT_EQ(placement, "host");
  ge::GetThreadLocalContext().SetGraphOption({});
}


TEST_F(UtestHybridRt2Executor, context_init_with_empty_host_constant_success) {
  // make option as host exec
  std::map<std::string, std::string> options;
  options["ge.exec.placement"] = "HOST";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ASSERT_TRUE(ge::GetContext().GetHostExecFlag());

  // prepare graph
  auto graph = ShareGraph::SimpleVariableGraph();
  GeTensorPtr weight = MakeScalarTensor(2.0f);
  ASSERT_NE(weight, nullptr);
  weight->ClearData(); // make weight size is 0

  auto constant = graph->FindFirstNodeMatchType("Constant");
  auto variable = graph->FindFirstNodeMatchType("Variable");
  ASSERT_NE(constant, nullptr);
  ASSERT_NE(variable, nullptr);

  auto constant_desc = constant->GetOpDesc()->MutableOutputDesc(0);
  auto variable_desc = variable->GetOpDesc()->MutableOutputDesc(0);

  constant_desc->SetShape(GeShape(std::vector<int64_t>{8, 0, 64, 64}));
  constant_desc->SetOriginShape(GeShape(std::vector<int64_t>{8, 0, 64, 64}));
  constant_desc->SetDataType(ge::DT_FLOAT);
  variable_desc->SetShape(GeShape(std::vector<int64_t>{8, 1, 64, 64, 16}));
  variable_desc->SetOriginShape(GeShape(std::vector<int64_t>{8, 3, 64, 64}));
  variable_desc->SetDataType(ge::DT_FLOAT);
  variable_desc->SetPlacement(ge::Placement::kPlacementDevice);
  int64_t real_dim_size = 0L;
  ASSERT_EQ(ge::TensorUtils::GetTensorSizeInBytes(*variable_desc, real_dim_size), GRAPH_SUCCESS);
  ge::TensorUtils::SetSize(*variable_desc, real_dim_size);

  ge::AttrUtils::SetTensor(constant->GetOpDesc(), ATTR_NAME_WEIGHTS, weight);

  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput->GetOpDesc()->SetSrcName({"foo"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  graph->TopologicalSorting();

  // build model
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();
  for (const auto &it : ge_root_model->GetSubgraphInstanceNameToModel()) {
    AttrUtils::SetInt(it.second, ATTR_MODEL_VAR_SIZE, 2048);
  }
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();

  // Prepare global var manager resource
  VarManagerPool::Instance().Destory();
  auto var_manager = VarManager::Instance(hybrid_model.GetSessionId());
  ASSERT_EQ(var_manager->Init(0, hybrid_model.GetSessionId(), 0, 0), SUCCESS);

  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  HybridModelRtV2Executor::RunCtx run_ctx;
  ASSERT_EQ(run_ctx.Init(&hybrid_model), SUCCESS);
  ge::ScopeGuard guarder([&var_manager]() { var_manager->FreeVarMemory(); });
  gert::StorageShape shape;
  gert::TensorData memory;
  EXPECT_EQ(run_ctx.graph_var_visitor_.GetVarShapeAndMemory(constant->GetName(), shape, memory), SUCCESS);
  EXPECT_EQ(memory.GetSize(), 0U);
  ge::GetThreadLocalContext().SetGraphOption({});
}

TEST_F(UtestHybridRt2Executor, context_init_with_host_variable_success) {
  auto graph = ShareGraph::SimpleVariableGraph();

  GeTensorPtr weight = MakeScalarTensor(2.0f);
  ASSERT_NE(weight, nullptr);

  auto constant = graph->FindFirstNodeMatchType("Constant");
  auto variable = graph->FindFirstNodeMatchType("Variable");
  ASSERT_NE(constant, nullptr);
  ASSERT_NE(variable, nullptr);
  ge::AttrUtils::SetStr(variable->GetOpDesc(), ge::ATTR_VARIABLE_PLACEMENT, "host");

  auto constant_desc = constant->GetOpDesc()->MutableOutputDesc(0);
  auto variable_desc = variable->GetOpDesc()->MutableOutputDesc(0);

  constant_desc->SetShape(GeShape(std::vector<int64_t>{8, 3, 64, 64}));
  constant_desc->SetOriginShape(GeShape(std::vector<int64_t>{8, 3, 64, 64}));
  constant_desc->SetDataType(ge::DT_FLOAT);
  variable_desc->SetShape(GeShape(std::vector<int64_t>{8, 1, 64, 64, 16}));
  variable_desc->SetOriginShape(GeShape(std::vector<int64_t>{8, 3, 64, 64}));
  variable_desc->SetDataType(ge::DT_FLOAT);
  variable_desc->SetPlacement(ge::Placement::kPlacementDevice);
  int64_t real_dim_size = 0L;
  ASSERT_EQ(ge::TensorUtils::GetTensorSizeInBytes(*variable_desc, real_dim_size), GRAPH_SUCCESS);
  ge::TensorUtils::SetSize(*variable_desc, real_dim_size);

  ge::AttrUtils::SetTensor(constant->GetOpDesc(), ATTR_NAME_WEIGHTS, weight);

  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput->GetOpDesc()->SetSrcName({"foo"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();
  for (const auto &it : ge_root_model->GetSubgraphInstanceNameToModel()) {
    AttrUtils::SetInt(it.second, ATTR_MODEL_VAR_SIZE, 2048);
  }
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();

  // Prepare global var manager resource
  VarManagerPool::Instance().Destory();
  auto var_manager = VarManager::Instance(hybrid_model.GetSessionId());
  ASSERT_EQ(var_manager->Init(0, hybrid_model.GetSessionId(), 0, 0), SUCCESS);

  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  HybridModelRtV2Executor::RunCtx run_ctx;
  ASSERT_EQ(run_ctx.Init(&hybrid_model), SUCCESS);
  ge::ScopeGuard guarder([&var_manager]() { var_manager->FreeVarMemory(); });
  gert::StorageShape shape;
  gert::TensorData memory;
  ASSERT_EQ(run_ctx.graph_var_visitor_.GetVarShapeAndMemory(variable->GetName(), shape, memory), SUCCESS);

  ASSERT_EQ(shape.GetStorageShape().GetDimNum(), 5U);
  ASSERT_EQ(shape.GetStorageShape().GetDim(0), 8U);
  ASSERT_EQ(shape.GetStorageShape().GetDim(1), 1U);
  ASSERT_EQ(shape.GetStorageShape().GetDim(2), 64U);
  ASSERT_EQ(shape.GetStorageShape().GetDim(3), 64U);
  ASSERT_EQ(shape.GetStorageShape().GetDim(4), 16U);

  ASSERT_EQ(shape.GetOriginShape().GetDimNum(), 4U);
  ASSERT_EQ(shape.GetOriginShape().GetDim(0), 8U);
  ASSERT_EQ(shape.GetOriginShape().GetDim(1), 3U);
  ASSERT_EQ(shape.GetOriginShape().GetDim(2), 64U);
  ASSERT_EQ(shape.GetOriginShape().GetDim(3), 64U);

  ASSERT_EQ(memory.GetPlacement(), gert::TensorPlacement::kOnHost);
  ASSERT_NE(memory.GetAddr(), nullptr);
}

TEST_F(UtestHybridRt2Executor, context_init_with_host_shm_variable_success) {
  std::string shm_var_name = "variable_host_shm";
  auto graph = ShareGraph::SimpleVariableGraph(shm_var_name);
  std::vector<int64_t> shape_dims = {8, 3, 64, 64};
  std::vector<int64_t> storage_dims = {8, 1, 64, 64, 16};
  size_t mem_size = 8 * 1 * 64 * 64 * 16 * 4 + 512;  // Malloc for storage, 512 padding size
  SharedMemInfo shm_info(shm_var_name, mem_size);
  ASSERT_EQ(HostMemManager::Instance().Initialize(), SUCCESS);
  ASSERT_EQ(HostMemManager::Instance().MallocHostSharedMemory(shm_info), SUCCESS);
  SharedMemInfo shm_info_malloced;
  ASSERT_TRUE(HostMemManager::Instance().QueryVarMemInfo(shm_var_name, shm_info_malloced));
  ASSERT_EQ(shm_info_malloced.op_name, shm_var_name);
  ASSERT_GE(shm_info_malloced.mem_size, mem_size);
  ASSERT_NE(shm_info_malloced.host_aligned_ptr->Get(), nullptr);

  GeTensorPtr weight = MakeScalarTensor(2.0f);
  ASSERT_NE(weight, nullptr);

  auto constant = graph->FindFirstNodeMatchType("Constant");
  auto variable = graph->FindFirstNodeMatchType("Variable");
  ASSERT_NE(constant, nullptr);
  ASSERT_NE(variable, nullptr);
  ge::AttrUtils::SetStr(variable->GetOpDesc(), ge::ATTR_VARIABLE_PLACEMENT, "host");
  std::map<std::string, std::string> options;
  options["ge.exec.placement"] = "HOST";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ASSERT_TRUE(ge::GetContext().GetHostExecFlag());

  auto constant_desc = constant->GetOpDesc()->MutableOutputDesc(0);
  auto variable_desc = variable->GetOpDesc()->MutableOutputDesc(0);

  constant_desc->SetShape(GeShape(shape_dims));
  constant_desc->SetOriginShape(GeShape(shape_dims));
  constant_desc->SetDataType(ge::DT_FLOAT);
  variable_desc->SetShape(GeShape(storage_dims));
  variable_desc->SetOriginShape(GeShape(shape_dims));
  variable_desc->SetDataType(ge::DT_FLOAT);
  variable_desc->SetPlacement(ge::Placement::kPlacementDevice);

  ge::AttrUtils::SetTensor(constant->GetOpDesc(), ATTR_NAME_WEIGHTS, weight);

  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput->GetOpDesc()->SetSrcName({"foo"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();
  for (const auto &it : ge_root_model->GetSubgraphInstanceNameToModel()) {
    AttrUtils::SetInt(it.second, ATTR_MODEL_VAR_SIZE, 2048);
  }
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();

  // Prepare global var manager resource
  auto var_manager = VarManager::Instance(hybrid_model.GetSessionId());
  ASSERT_EQ(var_manager->Init(0, hybrid_model.GetSessionId(), 0, 0), SUCCESS);

  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  HybridModelRtV2Executor::RunCtx run_ctx;
  ASSERT_EQ(run_ctx.Init(&hybrid_model), SUCCESS);
  ge::ScopeGuard guarder([&var_manager]() { var_manager->FreeVarMemory(); });
  gert::StorageShape shape;
  gert::TensorData memory;
  ASSERT_EQ(run_ctx.graph_var_visitor_.GetVarShapeAndMemory(variable->GetName(), shape, memory), SUCCESS);

  ASSERT_EQ(shape.GetStorageShape().GetDimNum(), 5U);
  ASSERT_EQ(shape.GetStorageShape().GetDim(0), 8U);
  ASSERT_EQ(shape.GetStorageShape().GetDim(1), 1U);
  ASSERT_EQ(shape.GetStorageShape().GetDim(2), 64U);
  ASSERT_EQ(shape.GetStorageShape().GetDim(3), 64U);
  ASSERT_EQ(shape.GetStorageShape().GetDim(4), 16U);

  ASSERT_EQ(shape.GetOriginShape().GetDimNum(), 4U);
  ASSERT_EQ(shape.GetOriginShape().GetDim(0), 8U);
  ASSERT_EQ(shape.GetOriginShape().GetDim(1), 3U);
  ASSERT_EQ(shape.GetOriginShape().GetDim(2), 64U);
  ASSERT_EQ(shape.GetOriginShape().GetDim(3), 64U);

  ASSERT_EQ(memory.GetPlacement(), gert::TensorPlacement::kOnHost);
  ASSERT_EQ(memory.GetAddr(), shm_info_malloced.host_aligned_ptr->Get());
  ge::GetThreadLocalContext().SetGraphOption({});
}

TEST_F(UtestHybridRt2Executor, Test_execute_on_host) {
  auto graph = ShareGraph::SimpleFooGraph();
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Foo") {
      MockLessImportantNodeKernel(node);
    }
  }
  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput->GetOpDesc()->SetSrcName({"foo"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  std::map<std::string, std::string> options;
  options["ge.exec.placement"] = "HOST";
  ge::GetThreadLocalContext().SetGraphOption(options);

  rtStream_t stream = nullptr;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(executor_rt_v2.run_ctx_.host_exec_flag_, true);

  InputData inputs;
  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[3072]);
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 3072, false));
  inputs.shapes.push_back({1, 16, 16, 3});
  HybridModelExecutor::ExecuteArgs args;

  auto desc = MakeShared<GeTensorDesc>();
  GeShape geshape({2, 2, 2, 2});
  desc->SetShape(geshape);

  auto allocator = NpuMemoryAllocator::GetAllocator();
  auto tensor_buffer = TensorBuffer::Create(allocator, 100);
  auto output_tensor = TensorValue(shared_ptr<TensorBuffer>(tensor_buffer.release()));
  args.outputs.emplace_back(output_tensor);
  args.output_desc.emplace_back(desc);

  EXPECT_EQ(executor_rt_v2.Execute(inputs, args), SUCCESS);
  EXPECT_EQ(executor_rt_v2.rt_inputs_.size(), 1);
  EXPECT_EQ(executor_rt_v2.rt_inputs_[0]->GetPlacement(), gert::kOnHost);
}

TEST_F(UtestHybridRt2Executor, SyncExecute_GertTensor) {
  auto graph = ShareGraph::SimpleFooGraph();
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Foo") {
      MockLessImportantNodeKernel(node);
    }
  }
  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput->GetOpDesc()->SetSrcName({"foo"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  std::map<std::string, std::string> options;
  options["ge.exec.placement"] = "HOST";
  ge::GetThreadLocalContext().SetGraphOption(options);

  rtStream_t stream = nullptr;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(executor_rt_v2.run_ctx_.host_exec_flag_, true);

  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[3072]);
  std::vector<gert::Tensor> inputs(1);
  inputs[0] = {{{1, 16, 16, 3}, {1, 16, 16, 3}},    // shape
                     {ge::FORMAT_ND, ge::FORMAT_ND, {}},  // format
                     gert::kOnHost,                       // placement
                     ge::DT_FLOAT16,                      // data type
                     (void *) data_buf.get()};

  std::vector<gert::Tensor> outputs;
  HybridModelExecutor::CtrlArgs ctrl_args;
  EXPECT_EQ(executor_rt_v2.Execute(inputs, outputs, ctrl_args), SUCCESS);
  EXPECT_EQ(executor_rt_v2.rt_inputs_.size(), 1);
  EXPECT_EQ(executor_rt_v2.rt_inputs_[0]->GetPlacement(), gert::kOnHost);
}

TEST_F(UtestHybridRt2Executor, prepare_inputdata_failed) {
  auto graph = ShareGraph::SimpleFooGraph();
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Foo") {
      MockLessImportantNodeKernel(node);
    }
  }
  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput->GetOpDesc()->SetSrcName({"foo"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = nullptr;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  InputData inputs;
  HybridModelExecutor::ExecuteArgs args;
  args.input_desc.resize(executor_rt_v2.num_inputs_);
  args.inputs.resize(executor_rt_v2.num_inputs_ + 1);
  EXPECT_EQ(executor_rt_v2.PrepareInputData(inputs, args), PARAM_INVALID);
}

TEST_F(UtestHybridRt2Executor, execute_model_online_return_eos) {
  const char_t *const kEnvRecordPath = "END_OF_SEQUENCE";
  char_t record_path[MMPA_MAX_PATH] = "end";
  mmSetEnv(kEnvRecordPath, &record_path[0U], MMPA_MAX_PATH);

  auto graph = ShareGraph::SimpleFooGraph();
  auto foo = graph->FindFirstNodeMatchType("Foo");
  ge::AttrUtils::SetStr(foo->GetOpDesc(), "_ge_attr_lowering_func", "_lower_foo_with_stream_sync");
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = nullptr;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);
  InputData inputs;
  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[3072]);
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 3072, false));
  inputs.shapes.push_back({1, 16, 16, 3});
  OutputData outputs;
  HybridModelExecutor::ExecuteArgs args;
  ret = executor_rt_v2.Execute(inputs, args);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(args.ctrl_args.is_eos, true);
  unsetenv(kEnvRecordPath);
}
namespace {
class MyMockRuntime : public RuntimeStub {
 public:
  MOCK_METHOD5(rtMemcpy, int32_t(void *, uint64_t, const void *, uint64_t, rtMemcpyKind_t));
};

rtError_t MockRtMemcpy(void *dst, uint64_t dest_max, const void *src, uint64_t count, rtMemcpyKind_t kind) {
  if (dest_max != 544) {
    return -1;
  }
  return 0;
}

size_t g_rt_memcpy_call_time = 0;
void *g_src_addr = nullptr;
class MyMockRuntime2 : public RuntimeStub {
 public:
  MOCK_METHOD5(rtMemcpy, int32_t(void *, uint64_t, const void *, uint64_t, rtMemcpyKind_t));
};

rtError_t MockRtMemcpy2(void *dst, uint64_t dest_max, const void *src, uint64_t count, rtMemcpyKind_t kind) {
  if (src == g_src_addr) {
    g_rt_memcpy_call_time++;
  }
  return 0;
}
}

TEST_F(UtestHybridRt2Executor, Execute_Success_CheckInputMallocSize) {
  auto graph = ShareGraph::SimpleFooGraph();
  auto foo = graph->FindFirstNodeMatchType("Foo");
  ge::AttrUtils::SetStr(foo->GetOpDesc(), "_ge_attr_lowering_func", "_lower_foo_with_stream_sync");
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = nullptr;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);
  InputData inputs;
  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[512]);
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 512, false));
  inputs.shapes.push_back({1, 1, 1, 128});
  OutputData outputs;
  HybridModelExecutor::ExecuteArgs args;

  // 对拷贝mock，校验内存长度为((512 + (32 * 2U) - 1U) / 32) * 32 = 544
  auto runtime_stub = std::make_shared<MyMockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtMemcpy).WillRepeatedly(testing::Invoke(MockRtMemcpy));

  ret = executor_rt_v2.Execute(inputs, args);
  EXPECT_EQ(ret, SUCCESS);
  RuntimeStub::Reset();
}

TEST_F(UtestHybridRt2Executor, Execute_Success_SkipRtMemcpyWhenSizeIsZero) {
  auto graph = ShareGraph::SimpleFooGraph();
  auto foo = graph->FindFirstNodeMatchType("Foo");
  ge::AttrUtils::SetStr(foo->GetOpDesc(), "_ge_attr_lowering_func", "_lower_foo_with_stream_sync");
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = nullptr;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);
  InputData inputs;
  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[0]);
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 0, false));
  g_src_addr = inputs.blobs.front().data;
  inputs.shapes.push_back({1, 1, 1, 128});
  OutputData outputs;
  HybridModelExecutor::ExecuteArgs args;

  // HybridModelRtV2Executor::Execute中输入长度如果为0，不发生拷贝
  auto runtime_stub = std::make_shared<MyMockRuntime2>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_EQ(g_rt_memcpy_call_time, 0);
  EXPECT_CALL(*runtime_stub, rtMemcpy).WillRepeatedly(testing::Invoke(MockRtMemcpy2));
  EXPECT_EQ(g_rt_memcpy_call_time, 0);

  ret = executor_rt_v2.Execute(inputs, args);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(g_rt_memcpy_call_time, 0);
  RuntimeStub::Reset();
}

TEST_F(UtestHybridRt2Executor, lazy_recompile_with_device_output_directly) {
  // set device output directly
  std::map<std::string, std::string> options;
  options[OPTION_EXEC_DYNAMIC_EXECUTE_MODE] = kLazyRecompile;
  options[OPTION_EXEC_ENABLE_COPY_OUTPUT_ADDR] = kIsCopyOuputAddr;
  ge::GetThreadLocalContext().SetGraphOption(options);
  HybridRt2ExecutorRunImpl();
  options.clear();
  ge::GetThreadLocalContext().SetGraphOption(options);
}

TEST_F(UtestHybridRt2Executor, Test_execute_with_iteration_loop_success) {
  TestHybridModelExecuteWithIterationLoop();
}

TEST_F(UtestHybridRt2Executor, Test_execute_on_host_with_multi_thread_success) {
  char_t max_runtime_core_path[MMPA_MAX_PATH] = "4";
  mmSetEnv("MAX_RUNTIME_CORE_NUMBER", &max_runtime_core_path[0U], MMPA_MAX_PATH);
  auto graph = ShareGraph::SimpleFooGraph();
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Foo") {
      MockLessImportantNodeKernel(node);
    }
  }
  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput->GetOpDesc()->SetSrcName({"foo"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  GlobalDumper::GetInstance()->SetEnableFlags(BuiltInSubscriberUtil::EnableBit<DumpType>(DumpType::kDataDump));
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  std::map<std::string, std::string> options;
  options["ge.exec.placement"] = "HOST";
  ge::GetThreadLocalContext().SetGraphOption(options);

  rtStream_t stream = nullptr;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(executor_rt_v2.run_ctx_.host_exec_flag_, true);

  InputData inputs;
  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[3072]);
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 3072, false));
  inputs.shapes.push_back({1, 16, 16, 3});
  HybridModelExecutor::ExecuteArgs args;

  auto desc = MakeShared<GeTensorDesc>();
  GeShape geshape({2, 2, 2, 2});
  desc->SetShape(geshape);

  auto allocator = NpuMemoryAllocator::GetAllocator();
  auto tensor_buffer = TensorBuffer::Create(allocator, 100);
  auto output_tensor = TensorValue(shared_ptr<TensorBuffer>(tensor_buffer.release()));
  args.outputs.emplace_back(output_tensor);
  args.output_desc.emplace_back(desc);

  EXPECT_EQ(executor_rt_v2.Execute(inputs, args), SUCCESS);
  unsetenv("MAX_RUNTIME_CORE_NUMBER");
  diagnoseSwitch::DisableDumper();
}

TEST_F(UtestHybridRt2Executor, Test_multi_thread_executor_with_invalid_value) {
  char_t max_runtime_core_path[MMPA_MAX_PATH] = "aa";
  mmSetEnv("MAX_RUNTIME_CORE_NUMBER", &max_runtime_core_path[0U], MMPA_MAX_PATH);
  auto graph = ShareGraph::SimpleFooGraph();
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Foo") {
      MockLessImportantNodeKernel(node);
    }
  }
  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput->GetOpDesc()->SetSrcName({"foo"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  GlobalDumper::GetInstance()->SetEnableFlags(BuiltInSubscriberUtil::EnableBit<DumpType>({DumpType::kDataDump}));
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  std::map<std::string, std::string> options;
  options["ge.exec.placement"] = "HOST";
  ge::GetThreadLocalContext().SetGraphOption(options);

  rtStream_t stream = nullptr;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_NE(ret, SUCCESS);
  unsetenv("MAX_RUNTIME_CORE_NUMBER");
  diagnoseSwitch::DisableDumper();
}

TEST_F(UtestHybridRt2Executor, Test_execute_rtStreamSynchronize_timeout_failed) {
  ge::diagnoseSwitch::EnableTrainingTrace();
  const char_t *const kTimeoutEnvPath = "TIMEOUT";
  char_t record_path[MMPA_MAX_PATH] = "timeout";
  mmSetEnv(kTimeoutEnvPath, &record_path[0U], MMPA_MAX_PATH);

  auto graph = ShareGraph::SimpleFooGraph();
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Foo") {
      MockLessImportantNodeKernel(node);
    }
  }
  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput->GetOpDesc()->SetSrcName({"foo"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  ge::AttrUtils::SetBool(netoutput->GetOpDesc(), ge::ATTR_NAME_INSERT_FP_PROFILILNG_TASK, true);
  ge::AttrUtils::SetInt(netoutput->GetOpDesc(), ge::ATTR_NAME_INSERT_PROFILILNG_TASK_LOG_ID, 20000);
  ge::AttrUtils::SetBool(netoutput->GetOpDesc(), ge::ATTR_NAME_INSERT_BP_PROFILILNG_TASK, true);
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  ge::GetContext().SetStreamSyncTimeout(15000);

  rtStream_t stream = nullptr;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);

  InputData inputs;
  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[3072]);
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 3072, false));
  inputs.shapes.push_back({1, 16, 16, 3});
  HybridModelExecutor::ExecuteArgs args;

  auto desc = MakeShared<GeTensorDesc>();
  GeShape geshape({2, 2, 2, 2});
  desc->SetShape(geshape);

  auto allocator = NpuMemoryAllocator::GetAllocator();
  auto tensor_buffer = TensorBuffer::Create(allocator, 100);
  auto output_tensor = TensorValue(shared_ptr<TensorBuffer>(tensor_buffer.release()));
  args.outputs.emplace_back(output_tensor);
  args.output_desc.emplace_back(desc);

  EXPECT_EQ(executor_rt_v2.Execute(inputs, args), ge::PARAM_INVALID);
  unsetenv(kTimeoutEnvPath);
  ge::diagnoseSwitch::DisableProfiling();
}

TEST_F(UtestHybridRt2Executor, HybridModelExecute_RecordProfiler_WithIterationLoop) {
  ge::diagnoseSwitch::EnableDeviceProfiling();
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(TestHybridModelExecuteWithIterationLoop, 20, 10, 20, 0);
  ge::diagnoseSwitch::DisableProfiling();
}

// 用例描述: 计算图上存在2个location相同的FileConstant节点，指定device执行。预期结果：二者TensorData相同。
TEST_F(UtestHybridRt2Executor, RunCtxInitWithDeviceFileConstants_TensorDataIsShared_FileConstantHasSameLocation) {
  const std::vector<std::string> location_config = {"xj_test_file_constant.bin", "xj_test_file_constant.bin"};
  const bool is_exec_on_host = false;
  TestRunCtxInitWithTwoFileConstants(location_config, is_exec_on_host);
}

// 用例描述: 计算图上存在2个location不同的FileConstant节点，指定device执行。预期结果：二者TensorData不同。
TEST_F(UtestHybridRt2Executor, RunCtxInitWithDeviceFileConstants_TensorDataNotShared_FileConstantHasDiffLocation) {
  const std::vector<std::string> location_config = {"xj_test_file_constant1.bin", "xj_test_file_constant2.bin"};
  const bool is_exec_on_host = false;
  TestRunCtxInitWithTwoFileConstants(location_config, is_exec_on_host);
}

// 用例描述: 计算图上存在2个location相同的FileConstant节点，指定host执行。预期结果：二者TensorData相同。
TEST_F(UtestHybridRt2Executor, RunCtxInitWithHostFileConstants_TensorDataIsShared_FileConstantHasSameLocation) {
  const std::vector<std::string> location_config = {"xj_test_file_constant.bin", "xj_test_file_constant.bin"};
  const bool is_exec_on_host = true;
  TestRunCtxInitWithTwoFileConstants(location_config, is_exec_on_host);
}

// 用例描述: 计算图上存在2个location不同的FileConstant节点，指定host执行。预期结果：二者TensorData不同。
TEST_F(UtestHybridRt2Executor, RunCtxInitWithHostFileConstants_TensorDataNotShared_FileConstantHasDiffLocation) {
  const std::vector<std::string> location_config = {"xj_test_file_constant1.bin", "xj_test_file_constant2.bin"};
  const bool is_exec_on_host = true;
  TestRunCtxInitWithTwoFileConstants(location_config, is_exec_on_host);
}

// 用例描述: 1个进程，2个session，session间权重文件相同，加载2次
TEST_F(UtestHybridRt2Executor, RunCtxInitWithDeviceFileConstants_TensorNotSharedBetweenSession_FileConstantHasSameLocation) {
  // session 0
  {
    const std::vector<std::string> location_config = {"tmp_weight_xj_fileconstant1.bin", "tmp_weight_xj_fileconstant1.bin"};
    const bool is_exec_on_host = false;
    uint64_t session_id = 0U;
    bool is_test_two_session = true;
    TestRunCtxInitWithTwoFileConstants(location_config, is_exec_on_host, session_id, is_test_two_session);
  }
  // session 1
  {
    const std::vector<std::string> location_config = {"tmp_weight_xj_fileconstant1.bin", "tmp_weight_xj_fileconstant1.bin"};
    const bool is_exec_on_host = false;
    uint64_t session_id = 1U;
    bool is_test_two_session = false;
    TestRunCtxInitWithTwoFileConstants(location_config, is_exec_on_host, session_id, is_test_two_session);
  }
}

/*
 * 用例描述: 加载执行动态shape模型，依赖执行器推导输出shape

 * 预置条件：
 * 1. 构造动态shape模型,走RT2执行器，输出构造为空，依赖RT2执行其返回
 *
 * 测试步骤：
 * 1. 构造一个包含Add节点的动态shape图
 * 2. 模型编译
 * 3. 将输入填好，输出构造为空指针
 * 4. 模型执行
 *
 * 预期结果：
 * 1. 模型执行成功
 * 2. 输出的shape与输入shape一致
 * 3. 输出的地址不为空指针
 */
TEST_F(UtestHybridRt2Executor, ExecuteWithStreamAsync_execute_model_online_dynamic_shape) {
  LoadDefaultSpaceRegistry();

  auto graph = ShareGraph::AicoreGraph();
  graph->TopologicalSorting();
  ge::AttrUtils::SetBool(graph, "need_set_stream_core_limits", true);
  std::map<std::string, std::string> options;
  options["ge.aicoreNum"] = "1";
  options["ge.vectorcoreNum"] = "1";
  ge::GetThreadLocalContext().SetSessionOption(options);
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = (void*)0x01;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_TRUE(executor_rt_v2.run_ctx_.aicore_num_str_ == "1");
  EXPECT_TRUE(executor_rt_v2.run_ctx_.vectorcore_num_str_ == "1");
  EXPECT_EQ(ret, SUCCESS);

  unique_ptr<uint8_t[]> data_buf(new(std::nothrow) uint8_t[512]);
  std::vector<GeTensor> input_tensors(2U);
  std::vector<GeTensor> output_tensors(1U);

  for (size_t i = 0U; i < input_tensors.size(); ++i) {
    input_tensors[i].MutableTensorDesc() = GeTensorDesc(GeShape({1, 1, 1, 128}), FORMAT_NCHW, DT_FLOAT);
    input_tensors[i].MutableTensorDesc().SetOriginShape(GeShape({1, 1, 1, 128}));
    input_tensors[i].MutableTensorDesc().SetPlacement(Placement::kPlacementDevice);
    input_tensors[i].SetData(data_buf.get(), 512);
  }
  for (size_t i = 0U; i < output_tensors.size(); ++i) {
    output_tensors[i].SetData(nullptr, 0U);
  }
  executor_rt_v2.run_ctx_.host_exec_flag_ = false;
  ret = executor_rt_v2.ExecuteWithStreamAsync(input_tensors, output_tensors, stream);
  EXPECT_EQ(ret, SUCCESS);

  for (size_t i = 0; i < output_tensors.size(); ++i) {
    EXPECT_NE(output_tensors[i].GetData().GetData(), nullptr);
    EXPECT_EQ(output_tensors[i].GetData().GetSize(), 512);
  }
  std::vector<int64_t>shape({1, 1, 1, 128});
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    EXPECT_EQ(output_tensors[i].MutableTensorDesc().GetOriginShape().GetDims(), shape);
    EXPECT_EQ(output_tensors[i].MutableTensorDesc().GetShape().GetDims(), shape);
  }

  std::vector<GeTensor> outputs;
  ret = executor_rt_v2.ExecuteWithStreamAsync(input_tensors, outputs, stream);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_FALSE(outputs.empty());

  RuntimeStub::Reset();
  std::map<std::string, std::string> options_empty;
  ge::GetThreadLocalContext().SetSessionOption(options_empty);
  ge::AttrUtils::SetBool(graph, "need_set_stream_core_limits", false);
}

TEST_F(UtestHybridRt2Executor, TfExecuteDynamicShapeWithBatchH2dDeviceId0) {
  RTS_STUB_RETURN_VALUE(rtsMemcpyBatch, rtError_t, RT_ERROR_NONE);

  auto graph = ShareGraph::AicoreGraph();
  graph->TopologicalSorting();
  ge::AttrUtils::SetBool(graph, "need_set_stream_core_limits", true);
  std::map<std::string, std::string> options_map;
  const uint64_t input_fusion_size = 0U;
  options_map[OPTION_EXEC_INPUT_FUSION_SIZE] = std::to_string(input_fusion_size);
  const string input_batch_cpy_str = "1";
  options_map[configure_option::INPUT_BATCH_CPY] = input_batch_cpy_str;
  ge::GetThreadLocalContext().SetSessionOption(options_map);
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = (void*)0x01;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);

  executor_rt_v2.run_ctx_.host_exec_flag_ = false;
  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[3072]);
  std::vector<gert::Tensor> inputs(2);
  inputs[0] = {{{1, 16, 16, 3}, {1, 16, 16, 3}},    // shape
                     {ge::FORMAT_ND, ge::FORMAT_ND, {}},  // format
                     gert::kOnHost,                       // placement
                     ge::DT_FLOAT16,                      // data type
                     (void *) data_buf.get()};

  inputs[1] = {{{1, 16, 16, 3}, {1, 16, 16, 3}},    // shape
                     {ge::FORMAT_ND, ge::FORMAT_ND, {}},  // format
                     gert::kOnHost,                       // placement
                     ge::DT_FLOAT16,                      // data type
                     (void *) data_buf.get()};

  std::vector<gert::Tensor> outputs;
  HybridModelExecutor::CtrlArgs ctrl_args;
  ret = executor_rt_v2.Execute(inputs, outputs, ctrl_args);
  EXPECT_EQ(ret, SUCCESS);

  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 2);
  EXPECT_EQ(RuntimeStub::GetInstance()->batch_memcpy_device_id, 0);

  RuntimeStub::Reset();
  std::map<std::string, std::string> options_empty;
  ge::GetThreadLocalContext().SetSessionOption(options_empty);
  ge::AttrUtils::SetBool(graph, "need_set_stream_core_limits", false);

  RuntimeStub::GetInstance()->cur_device_id = 0;
  RuntimeStub::GetInstance()->batch_memcpy_device_id = 0;
}

TEST_F(UtestHybridRt2Executor, TfExecuteDynamicShapeWithBatchH2dDeviceId1) {
  RTS_STUB_RETURN_VALUE(rtsMemcpyBatch, rtError_t, RT_ERROR_NONE);
  RuntimeStub::GetInstance()->cur_device_id = 1;
  int32_t cur_rtGetDevice_is_mock_new_way = GetMockRtGetDeviceWay();
  SetMockRtGetDeviceWay(1);

  auto graph = ShareGraph::AicoreGraph();
  graph->TopologicalSorting();
  // ge::AttrUtils::SetBool(graph, "need_set_stream_core_limits", true);
  std::map<std::string, std::string> options_map;
  const uint64_t input_fusion_size = 0U;
  options_map[OPTION_EXEC_INPUT_FUSION_SIZE] = std::to_string(input_fusion_size);

  const string input_batch_cpy_str = "1";
  options_map[configure_option::INPUT_BATCH_CPY] = input_batch_cpy_str;
  ge::GetThreadLocalContext().SetSessionOption(options_map);
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = (void*)0x01;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);

  executor_rt_v2.run_ctx_.host_exec_flag_ = false;
  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[3072]);
  std::vector<gert::Tensor> inputs(2);
  inputs[0] = {{{1, 16, 16, 3}, {1, 16, 16, 3}},    // shape
                     {ge::FORMAT_ND, ge::FORMAT_ND, {}},  // format
                     gert::kOnHost,                       // placement
                     ge::DT_FLOAT16,                      // data type
                     (void *) data_buf.get()};

  inputs[1] = {{{1, 16, 16, 3}, {1, 16, 16, 3}},    // shape
                     {ge::FORMAT_ND, ge::FORMAT_ND, {}},  // format
                     gert::kOnHost,                       // placement
                     ge::DT_FLOAT16,                      // data type
                     (void *) data_buf.get()};

  std::vector<gert::Tensor> outputs;
  HybridModelExecutor::CtrlArgs ctrl_args;
  ret = executor_rt_v2.Execute(inputs, outputs, ctrl_args);
  EXPECT_EQ(ret, SUCCESS);

  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 2);
  EXPECT_EQ(RuntimeStub::GetInstance()->batch_memcpy_device_id, 1);

  RuntimeStub::Reset();
  // ge::AttrUtils::SetBool(graph, "need_set_stream_core_limits", false);

  std::map<std::string, std::string> options_empty;
  ge::GetThreadLocalContext().SetSessionOption(options_empty);
  RuntimeStub::GetInstance()->cur_device_id = 0;
  RuntimeStub::GetInstance()->batch_memcpy_device_id = 0;
  SetMockRtGetDeviceWay(cur_rtGetDevice_is_mock_new_way);
}

/*
 * 用例描述: 加载执行动态shape模型，注册外置allocator

 * 预置条件：
 * 1. 构造动态shape模型,走RT2执行器，输出构造为空，依赖RT2执行其返回
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 模型编译
 * 3. 注册external allocator
 * 4. 将输入填好，输出构造为空指针
 * 5. 模型执行（通过allocator申请内存）
 *
 * 预期结果：
 * 1. 模型执行成功
 * 2. 使用外置allocator进行内存申请成功
 * 3. 使用外置allocator进行内存释放成功
 */
TEST_F(UtestHybridRt2Executor, ExecuteWithStreamAsync_execute_model_online_dynamic_shape_with_external_allocator) {
  auto graph = ShareGraph::AicoreGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = (void*)0x01;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  ExternalAllocatorManager::SetExternalAllocator(stream, external_allocator);

  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);
  std::vector<GeTensor> input_tensors(2U);
  std::vector<GeTensor> output_tensors(1U);
  unique_ptr<uint8_t[]> data_buf(new(std::nothrow) uint8_t[512]);
  {
    for (size_t i = 0U; i < input_tensors.size(); ++i) {
      input_tensors[i].MutableTensorDesc() = GeTensorDesc(GeShape({1, 1, 1, 128}), FORMAT_NCHW, DT_FLOAT);
      input_tensors[i].MutableTensorDesc().SetOriginShape(GeShape({1, 1, 1, 128}));
      input_tensors[i].MutableTensorDesc().SetPlacement(Placement::kPlacementDevice);
      input_tensors[i].SetData(data_buf.get(), 512);
    }
    for (size_t i = 0U; i < output_tensors.size(); ++i) {
      output_tensors[i].SetData(nullptr, 0U);
    }
    executor_rt_v2.run_ctx_.host_exec_flag_ = false;
    ret = executor_rt_v2.ExecuteWithStreamAsync(input_tensors, output_tensors, stream);
    EXPECT_EQ(ret, SUCCESS);

    std::vector<gert::Tensor> input_tensor;
    std::vector<gert::Tensor> output_tensor;
    input_tensor.resize(2);
    output_tensor.resize(1);
    input_tensor[0] = {{{1, 1, 1, 128}, {1, 1, 1, 128}},                // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              gert::kOnDeviceHbm,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *) data_buf.get()};
    input_tensor[1] = {{{1, 1, 1, 128}, {1, 1, 1, 128}},                // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              gert::kOnDeviceHbm,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *) data_buf.get()};
    ret = executor_rt_v2.ExecuteWithStreamAsync(input_tensor, output_tensor, stream);
    EXPECT_EQ(ret, SUCCESS);

    std::vector<gert::Tensor> outputs;
    ret = executor_rt_v2.ExecuteWithStreamAsync(input_tensor, outputs, stream);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_FALSE(outputs.empty());

    executor_rt_v2.run_ctx_.host_exec_flag_ = true;
    input_tensor[0] = {{{1, 1, 1, 128}, {1, 1, 1, 128}},                // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              gert::kOnHost,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *) data_buf.get()};
    input_tensor[1] = {{{1, 1, 1, 128}, {1, 1, 1, 128}},                // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              gert::kOnHost,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *) data_buf.get()};
    ret = executor_rt_v2.ExecuteWithStreamAsync(input_tensor, output_tensor, stream);
    EXPECT_EQ(ret, SUCCESS);
    
    executor_rt_v2.run_ctx_.host_exec_flag_ = false;
    unique_ptr<uint8_t[]> data_buf1(new(std::nothrow) uint8_t[4]);

    input_tensor[0] = {{{1}, {1}},                // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              gert::kOnHost,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *) data_buf1.get()};
    input_tensor[1] = {{{1}, {1}},                // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              gert::kOnHost,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *) data_buf1.get()};
    gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(1);
    ret = executor_rt_v2.ExecuteWithStreamAsync(input_tensor, output_tensor, stream);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 0);
    EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);
  }
  // 转移所有权后，由外部释放内存
  using RawGeDataPtr = std::unique_ptr<uint8_t[], ge::Tensor::DeleteFunc>;
  for (size_t i = 0U; i < output_tensors.size(); ++i) {
    auto alging_ptr = output_tensors[i].GetAlignedPtr();
    if (alging_ptr != nullptr) {
      RawGeDataPtr ge_data_ptr = alging_ptr->Reset();
    }
  }
  // 内存合理释放
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetFreeCnt(), 0);
  RuntimeStub::Reset();
  ExternalAllocatorManager::DeleteExternalAllocator(stream);
}
/*
 * 用例描述: 加载执行动态shape模型，注册外置allocator

 * 预置条件：
 * 1. 构造动态shape模型,走RT2执行器，输出构造为非空
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 模型编译
 * 3. 注册external allocator
 * 4. 将输入填好，同时构造好输出
 * 5. 模型执行
 *
 * 预期结果：
 * 1. 模型执行成功
 * 2. 不会使用外置Allocator申请内存
 */
TEST_F(UtestHybridRt2Executor, ExecuteWithStreamAsync_execute_model_online_dynamic_shape_with_external_allocator_gived_output) {
  auto graph = ShareGraph::AicoreGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = (void*)0x01;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  ExternalAllocatorManager::SetExternalAllocator(stream, external_allocator);

  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);
  std::vector<GeTensor> input_tensors(2U);
  std::vector<GeTensor> output_tensors(1U);
  unique_ptr<uint8_t[]> data_buf(new(std::nothrow) uint8_t[512]);
  {
    for (size_t i = 0U; i < input_tensors.size(); ++i) {
      input_tensors[i].MutableTensorDesc() = GeTensorDesc(GeShape({1, 1, 1, 128}), FORMAT_NCHW, DT_FLOAT);
      input_tensors[i].MutableTensorDesc().SetOriginShape(GeShape({1, 1, 1, 128}));
      input_tensors[i].MutableTensorDesc().SetPlacement(Placement::kPlacementDevice);
      input_tensors[i].SetData(data_buf.get(), 512);
    }
    for (size_t i = 0U; i < output_tensors.size(); ++i) {
      output_tensors[i].MutableTensorDesc() = GeTensorDesc(GeShape({1, 1, 1, 128}), FORMAT_NCHW, DT_FLOAT);
      output_tensors[i].MutableTensorDesc().SetOriginShape(GeShape({1, 1, 1, 128}));
      output_tensors[i].MutableTensorDesc().SetPlacement(Placement::kPlacementDevice);
      output_tensors[i].SetData(data_buf.get(), 512);
    }
    executor_rt_v2.run_ctx_.host_exec_flag_ = false;
    ret = executor_rt_v2.ExecuteWithStreamAsync(input_tensors, output_tensors, stream);
    EXPECT_EQ(ret, SUCCESS);

    // 外置Allocator不会申请内存
    EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 16);
    EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);
  }

  RuntimeStub::Reset();
  ExternalAllocatorManager::DeleteExternalAllocator(stream);
}

TEST_F(UtestHybridRt2Executor, ExecuteWithStreamAsync_execute_model_online_host_exec) {
  auto graph = ShareGraph::AicoreGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = (void*)0x01;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);
  unique_ptr<uint8_t[]> data_buf(new(std::nothrow) uint8_t[512]);
  std::vector<GeTensor> input_tensors(2U);
  std::vector<GeTensor> output_tensors(1U);

  for (size_t i = 0U; i < input_tensors.size(); ++i) {
    input_tensors[i].MutableTensorDesc() = GeTensorDesc(GeShape({1, 1, 1, 128}), FORMAT_NCHW, DT_FLOAT);
    input_tensors[i].MutableTensorDesc().SetOriginShape(GeShape({1, 1, 1, 128}));
    input_tensors[i].MutableTensorDesc().SetPlacement(Placement::kPlacementHost);
    input_tensors[i].SetData(data_buf.get(), 512);
  }
  for (size_t i = 0U; i < output_tensors.size(); ++i) {
    output_tensors[i].SetData(nullptr, 0U);
  }
  executor_rt_v2.run_ctx_.host_exec_flag_ = true;
  ret = executor_rt_v2.ExecuteWithStreamAsync(input_tensors, output_tensors, stream);
  EXPECT_EQ(ret, SUCCESS);
  executor_rt_v2.run_ctx_.host_exec_flag_ = false;

  RuntimeStub::Reset();
}

TEST_F(UtestHybridRt2Executor, ExecuteWithStreamAsync_execute_model_online_host_input) {
  auto graph = ShareGraph::AicoreGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = (void *)0x01;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);
  executor_rt_v2.run_ctx_.host_exec_flag_ = false;

  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[512]);
  std::vector<GeTensor> input_tensors;
  std::vector<GeTensor> output_tensors(1U);

  auto scalar_host_tensor = GeTensor(GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 4);
  scalar_host_tensor.MutableTensorDesc().SetOriginShape(GeShape());
  scalar_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  auto list_scalar_host_tensor = GeTensor(GeTensorDesc(GeShape({128}), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 512);
  list_scalar_host_tensor.MutableTensorDesc().SetOriginShape(GeShape({128}));
  list_scalar_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  auto nd_host_tensor = GeTensor(GeTensorDesc(GeShape({1, 1, 1, 128}), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 512);
  nd_host_tensor.MutableTensorDesc().SetOriginShape(GeShape({1, 1, 1, 128}));
  nd_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);


  input_tensors.resize(2U, scalar_host_tensor);
  output_tensors[0].SetData(nullptr, 0U);
  ret = executor_rt_v2.ExecuteWithStreamAsync(input_tensors, output_tensors, stream);
  EXPECT_EQ(ret, SUCCESS);

  input_tensors.clear();
  input_tensors.push_back(scalar_host_tensor);
  input_tensors.push_back(list_scalar_host_tensor);
  output_tensors[0].SetData(nullptr, 0U);
  ret = executor_rt_v2.ExecuteWithStreamAsync(input_tensors, output_tensors, stream);
  EXPECT_EQ(ret, SUCCESS);

  input_tensors.clear();
  input_tensors.resize(2U, nd_host_tensor);  // not support
  output_tensors[0].SetData(nullptr, 0U);
  ret = executor_rt_v2.ExecuteWithStreamAsync(input_tensors, output_tensors, stream);
  EXPECT_NE(ret, SUCCESS);

  input_tensors.clear();
  input_tensors.push_back(scalar_host_tensor);
  input_tensors.push_back(nd_host_tensor);  // not support
  output_tensors[0].SetData(nullptr, 0U);
  ret = executor_rt_v2.ExecuteWithStreamAsync(input_tensors, output_tensors, stream);
  EXPECT_NE(ret, SUCCESS);

  RuntimeStub::Reset();
}

TEST_F(UtestHybridRt2Executor, ExecuteOnlineModel_RecycleAfterExecute_GertTensor) {
  auto graph = ShareGraph::AicoreGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = (void *)0x01;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  ASSERT_EQ(ret, SUCCESS);
  executor_rt_v2.run_ctx_.host_exec_flag_ = false;
  std::vector<gert::Tensor> inputs(2);
  TensorCheckUtils::ConstructGertTensor(inputs[0]);
  TensorCheckUtils::ConstructGertTensor(inputs[1]);

  GertRuntimeStub runtime_stub;
  // 通过info日志检查执行过程的正确性
  dlog_setlevel(GE_MODULE_NAME, 1, 0);
  runtime_stub.GetSlogStub().Clear();
  ret = executor_rt_v2.ExecuteOnlineModel(inputs, nullptr);
  auto find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("rts_allocator_.* Free:Free block device_id:0 theory_size_:0 theory_min_size_");
  EXPECT_TRUE(find_log > 0);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
  EXPECT_EQ(ret, SUCCESS);
  RuntimeStub::Reset();
}

TEST_F(UtestHybridRt2Executor, Execute_Success_SkipBatchMemcpyWhenSizeIsZero) {
  auto graph = ShareGraph::AicoreGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  std::map<std::string, std::string> options;
  options["ge.inputBatchCpy"] = "1";
  ge::GetThreadLocalContext().SetSessionOption(options);
  ge::GetThreadLocalContext().SetGraphOption(options);

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = (void *)0x01;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);
  executor_rt_v2.run_ctx_.host_exec_flag_ = false;

  // 空数据 buffer
  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[0]);

  // 空 GeTensor
  auto empty_tensor = GeTensor(GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT), nullptr, 0);
  empty_tensor.MutableTensorDesc().SetOriginShape(GeShape());
  empty_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  InputData inputs;
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 0, false));
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 0, false));
  inputs.shapes.push_back({});
  inputs.shapes.push_back({});

  auto gert_inputs = InputData2GertTensors(inputs);
  ret = executor_rt_v2.ExecuteOnlineModel(gert_inputs, nullptr);
  EXPECT_EQ(ret, SUCCESS);
  RuntimeStub::Reset();
  ge::GetThreadLocalContext().SetSessionOption(options);
  ge::GetThreadLocalContext().SetGraphOption(options);
}

TEST_F(UtestHybridRt2Executor, HandleResult_RecycleWhenEOS) {
  auto graph = ShareGraph::AicoreGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = (void *)0x01;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);
  executor_rt_v2.run_ctx_.host_exec_flag_ = false;

  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[512]);
  std::vector<GeTensor> input_tensors;
  std::vector<GeTensor> output_tensors(1U);

  auto scalar_host_tensor = GeTensor(GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 4);
  scalar_host_tensor.MutableTensorDesc().SetOriginShape(GeShape());
  scalar_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  auto list_scalar_host_tensor = GeTensor(GeTensorDesc(GeShape({128}), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 512);
  list_scalar_host_tensor.MutableTensorDesc().SetOriginShape(GeShape({128}));
  list_scalar_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  auto nd_host_tensor = GeTensor(GeTensorDesc(GeShape({1, 1, 1, 128}), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 512);
  nd_host_tensor.MutableTensorDesc().SetOriginShape(GeShape({1, 1, 1, 128}));
  nd_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  InputData inputs;
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 512, false));
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 512, false));
  inputs.shapes.push_back({});
  inputs.shapes.push_back({});

  input_tensors.resize(2U, scalar_host_tensor);
  output_tensors[0].SetData(nullptr, 0U);
  auto gert_inputs = InputData2GertTensors(inputs);
  ret = executor_rt_v2.ExecuteOnlineModel(gert_inputs, nullptr);
  EXPECT_EQ(ret, SUCCESS);
  HybridModelExecutor::CtrlArgs args;
  args.stream = stream;
  args.is_eos = true;
  std::vector<gert::Tensor> gert_outputs;
  ret = executor_rt_v2.HandleResult(SUCCESS, 0, args, gert_outputs, nullptr);
  EXPECT_EQ(ret, END_OF_SEQUENCE);
  RuntimeStub::Reset();
}

TEST_F(UtestHybridRt2Executor, HandleResult_BatchH2d_RecycleWhenEOS) {
  auto graph = ShareGraph::AicoreGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  std::map<std::string, std::string> options;
  options["ge.inputBatchCpy"] = "1";
  options["ge.inputPlacement"] = "DeviceHbm";
  ge::GetThreadLocalContext().SetSessionOption(options);
  ge::GetThreadLocalContext().SetGraphOption(options);

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = (void *)0x01;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);
  executor_rt_v2.run_ctx_.host_exec_flag_ = false;

  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[512]);
  std::vector<GeTensor> input_tensors;
  std::vector<GeTensor> output_tensors(1U);

  auto scalar_host_tensor = GeTensor(GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 4);
  scalar_host_tensor.MutableTensorDesc().SetOriginShape(GeShape());
  scalar_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  auto list_scalar_host_tensor = GeTensor(GeTensorDesc(GeShape({128}), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 512);
  list_scalar_host_tensor.MutableTensorDesc().SetOriginShape(GeShape({128}));
  list_scalar_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  auto nd_host_tensor = GeTensor(GeTensorDesc(GeShape({1, 1, 1, 128}), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 512);
  nd_host_tensor.MutableTensorDesc().SetOriginShape(GeShape({1, 1, 1, 128}));
  nd_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  InputData inputs;
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 512, false));
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 512, false));
  inputs.shapes.push_back({});
  inputs.shapes.push_back({});

  auto gert_inputs = InputData2GertTensors(inputs);
  ret = executor_rt_v2.ExecuteOnlineModel(gert_inputs, nullptr);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 2);
  HybridModelExecutor::CtrlArgs args;
  args.stream = stream;
  args.is_eos = true;
  std::vector<gert::Tensor> gert_outputs;
  ret = executor_rt_v2.HandleResult(SUCCESS, 0, args, gert_outputs, nullptr);
  EXPECT_EQ(ret, END_OF_SEQUENCE);
  RuntimeStub::Reset();

  options["ge.inputBatchCpy"] = "0";
  ge::GetThreadLocalContext().SetSessionOption(options);
  ge::GetThreadLocalContext().SetGraphOption(options);
}

TEST_F(UtestHybridRt2Executor, HandleResult_BatchH2dButNotSupport_RecycleWhenEOS) {
  auto graph = ShareGraph::AicoreGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  std::map<std::string, std::string> options;
  options["ge.inputBatchCpy"] = "1";
  ge::GetThreadLocalContext().SetSessionOption(options);
  ge::GetThreadLocalContext().SetGraphOption(options);

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = (void *)0x01;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);
  executor_rt_v2.run_ctx_.host_exec_flag_ = false;

  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[512]);
  std::vector<GeTensor> input_tensors;
  std::vector<GeTensor> output_tensors(1U);

  auto scalar_host_tensor = GeTensor(GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 4);
  scalar_host_tensor.MutableTensorDesc().SetOriginShape(GeShape());
  scalar_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  auto list_scalar_host_tensor = GeTensor(GeTensorDesc(GeShape({128}), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 512);
  list_scalar_host_tensor.MutableTensorDesc().SetOriginShape(GeShape({128}));
  list_scalar_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  auto nd_host_tensor = GeTensor(GeTensorDesc(GeShape({1, 1, 1, 128}), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 512);
  nd_host_tensor.MutableTensorDesc().SetOriginShape(GeShape({1, 1, 1, 128}));
  nd_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  InputData inputs;
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 512, false));
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 512, false));
  inputs.shapes.push_back({});
  inputs.shapes.push_back({});

  input_tensors.resize(2U, scalar_host_tensor);
  output_tensors[0].SetData(nullptr, 0U);
  RTS_STUB_RETURN_VALUE(rtsMemcpyBatch, rtError_t, ACL_ERROR_RT_FEATURE_NOT_SUPPORT);
  auto gert_inputs = InputData2GertTensors(inputs);
  ret = executor_rt_v2.ExecuteOnlineModel(gert_inputs, nullptr);
  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 0);
  EXPECT_EQ(ret, SUCCESS);
  HybridModelExecutor::CtrlArgs args;
  args.stream = stream;
  args.is_eos = true;
  std::vector<gert::Tensor> gert_outputs;
  ret = executor_rt_v2.HandleResult(SUCCESS, 0, args, gert_outputs, nullptr);
  EXPECT_EQ(ret, END_OF_SEQUENCE);
  RuntimeStub::Reset();

  options["ge.inputBatchCpy"] = "0";
  ge::GetThreadLocalContext().SetSessionOption(options);
  ge::GetThreadLocalContext().SetGraphOption(options);
}

TEST_F(UtestHybridRt2Executor, HandleResult_BatchH2dButFailed_RecycleWhenEOS) {
  auto graph = ShareGraph::AicoreGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  std::map<std::string, std::string> options;
  options["ge.inputBatchCpy"] = "1";
  ge::GetThreadLocalContext().SetSessionOption(options);
  ge::GetThreadLocalContext().SetGraphOption(options);

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = (void *)0x01;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);
  executor_rt_v2.run_ctx_.host_exec_flag_ = false;

  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[512]);
  std::vector<GeTensor> input_tensors;
  std::vector<GeTensor> output_tensors(1U);

  auto scalar_host_tensor = GeTensor(GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 4);
  scalar_host_tensor.MutableTensorDesc().SetOriginShape(GeShape());
  scalar_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  auto list_scalar_host_tensor = GeTensor(GeTensorDesc(GeShape({128}), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 512);
  list_scalar_host_tensor.MutableTensorDesc().SetOriginShape(GeShape({128}));
  list_scalar_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  auto nd_host_tensor = GeTensor(GeTensorDesc(GeShape({1, 1, 1, 128}), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 512);
  nd_host_tensor.MutableTensorDesc().SetOriginShape(GeShape({1, 1, 1, 128}));
  nd_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  InputData inputs;
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 512, false));
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 512, false));
  inputs.shapes.push_back({});
  inputs.shapes.push_back({});

  input_tensors.resize(2U, scalar_host_tensor);
  output_tensors[0].SetData(nullptr, 0U);
  RTS_STUB_RETURN_VALUE(rtsMemcpyBatch, rtError_t, -1);
  auto gert_inputs = InputData2GertTensors(inputs);
  ret = executor_rt_v2.ExecuteOnlineModel(gert_inputs, nullptr);
  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 0);
  EXPECT_NE(ret, SUCCESS);

  RuntimeStub::Reset();
  options["ge.inputBatchCpy"] = "0";
  ge::GetThreadLocalContext().SetSessionOption(options);
  ge::GetThreadLocalContext().SetGraphOption(options);
}

TEST_F(UtestHybridRt2Executor, HandleResult_BatchH2dFallbackButFailed_RecycleWhenEOS) {
  auto graph = ShareGraph::AicoreGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  std::map<std::string, std::string> options;
  options["ge.inputBatchCpy"] = "1";
  ge::GetThreadLocalContext().SetSessionOption(options);
  ge::GetThreadLocalContext().SetGraphOption(options);

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = (void *)0x01;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);
  executor_rt_v2.run_ctx_.host_exec_flag_ = false;

  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[512]);
  std::vector<GeTensor> input_tensors;
  std::vector<GeTensor> output_tensors(1U);

  auto scalar_host_tensor = GeTensor(GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 4);
  scalar_host_tensor.MutableTensorDesc().SetOriginShape(GeShape());
  scalar_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  auto list_scalar_host_tensor = GeTensor(GeTensorDesc(GeShape({128}), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 512);
  list_scalar_host_tensor.MutableTensorDesc().SetOriginShape(GeShape({128}));
  list_scalar_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  auto nd_host_tensor = GeTensor(GeTensorDesc(GeShape({1, 1, 1, 128}), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 512);
  nd_host_tensor.MutableTensorDesc().SetOriginShape(GeShape({1, 1, 1, 128}));
  nd_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  InputData inputs;
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 512, false));
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 512, false));
  inputs.shapes.push_back({});
  inputs.shapes.push_back({});

  input_tensors.resize(2U, scalar_host_tensor);
  output_tensors[0].SetData(nullptr, 0U);
  RTS_STUB_RETURN_VALUE(rtsMemcpyBatch, rtError_t, ACL_ERROR_RT_FEATURE_NOT_SUPPORT);
  RTS_STUB_RETURN_VALUE(rtMemcpy, rtError_t, -1);
  auto gert_inputs = InputData2GertTensors(inputs);
  ret = executor_rt_v2.ExecuteOnlineModel(gert_inputs, nullptr);
  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 0);
  EXPECT_NE(ret, SUCCESS);

  RuntimeStub::Reset();
  options["ge.inputBatchCpy"] = "0";
  ge::GetThreadLocalContext().SetSessionOption(options);
  ge::GetThreadLocalContext().SetGraphOption(options);
}

TEST_F(UtestHybridRt2Executor, HandleResult_BatchH2dOneInputFallback_RecycleWhenEOS) {
  auto graph = ShareGraph::SimpleFooGraph();
  auto foo = graph->FindFirstNodeMatchType("Foo");
  ge::AttrUtils::SetStr(foo->GetOpDesc(), "_ge_attr_lowering_func", "_lower_foo_with_stream_sync");
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  std::map<std::string, std::string> options;
  options["ge.inputBatchCpy"] = "1";
  ge::GetThreadLocalContext().SetSessionOption(options);
  ge::GetThreadLocalContext().SetGraphOption(options);

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = nullptr;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);
  InputData inputs;
  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[512]);
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 512, false));
  inputs.shapes.push_back({1, 1, 1, 128});
  OutputData outputs;
  HybridModelExecutor::ExecuteArgs args;

  ret = executor_rt_v2.Execute(inputs, args);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 0);
  RuntimeStub::Reset();
  options["ge.inputBatchCpy"] = "0";
  ge::GetThreadLocalContext().SetSessionOption(options);
  ge::GetThreadLocalContext().SetGraphOption(options);
}

TEST_F(UtestHybridRt2Executor, HandleResult_BatchH2dOneInputFallbackFailed_RecycleWhenEOS) {
  auto graph = ShareGraph::SimpleFooGraph();
  auto foo = graph->FindFirstNodeMatchType("Foo");
  ge::AttrUtils::SetStr(foo->GetOpDesc(), "_ge_attr_lowering_func", "_lower_foo_with_stream_sync");
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  std::map<std::string, std::string> options;
  options["ge.inputBatchCpy"] = "1";
  ge::GetThreadLocalContext().SetSessionOption(options);
  ge::GetThreadLocalContext().SetGraphOption(options);

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = nullptr;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);
  InputData inputs;
  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[512]);
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 512, false));
  inputs.shapes.push_back({1, 1, 1, 128});
  OutputData outputs;
  HybridModelExecutor::ExecuteArgs args;

  RTS_STUB_RETURN_VALUE(rtMemcpy, rtError_t, -1);
  ret = executor_rt_v2.Execute(inputs, args);
  EXPECT_NE(ret, SUCCESS);
  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 0);
  RuntimeStub::Reset();
  options["ge.inputBatchCpy"] = "0";
  ge::GetThreadLocalContext().SetSessionOption(options);
  ge::GetThreadLocalContext().SetGraphOption(options);
}

TEST_F(UtestHybridRt2Executor, HandleResult_RecycleWhenError) {
  auto graph = ShareGraph::AicoreGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = (void *)0x01;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);
  executor_rt_v2.run_ctx_.host_exec_flag_ = false;

  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[512]);
  std::vector<GeTensor> input_tensors;
  std::vector<GeTensor> output_tensors(1U);

  auto scalar_host_tensor = GeTensor(GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 4);
  scalar_host_tensor.MutableTensorDesc().SetOriginShape(GeShape());
  scalar_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  auto list_scalar_host_tensor = GeTensor(GeTensorDesc(GeShape({128}), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 512);
  list_scalar_host_tensor.MutableTensorDesc().SetOriginShape(GeShape({128}));
  list_scalar_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  auto nd_host_tensor = GeTensor(GeTensorDesc(GeShape({1, 1, 1, 128}), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 512);
  nd_host_tensor.MutableTensorDesc().SetOriginShape(GeShape({1, 1, 1, 128}));
  nd_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  InputData inputs;
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 512, false));
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 512, false));
  inputs.shapes.push_back({});
  inputs.shapes.push_back({});

  input_tensors.resize(2U, scalar_host_tensor);
  output_tensors[0].SetData(nullptr, 0U);
  auto gert_inputs = InputData2GertTensors(inputs);
  ret = executor_rt_v2.ExecuteOnlineModel(gert_inputs, nullptr);
  EXPECT_EQ(ret, SUCCESS);
  HybridModelExecutor::CtrlArgs args;
  args.stream = stream;
  std::vector<gert::Tensor> gert_outputs;
  ret = executor_rt_v2.HandleResult(FAILED, 0, args, gert_outputs, nullptr);
  EXPECT_EQ(ret, INTERNAL_ERROR);
  RuntimeStub::Reset();
}

TEST_F(UtestHybridRt2Executor, HandleResult_RecycleWhenCopyOutputsError) {
  auto graph = ShareGraph::AicoreGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);
  rtStream_t stream = (void *)0x01;
  HybridModelRtV2Executor executor_rt_v2(&hybrid_model, 0, stream);
  auto ret = executor_rt_v2.Init();
  EXPECT_EQ(ret, SUCCESS);
  executor_rt_v2.run_ctx_.host_exec_flag_ = false;

  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[512]);
  std::vector<GeTensor> input_tensors;
  std::vector<GeTensor> output_tensors(1U);

  auto scalar_host_tensor = GeTensor(GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 4);
  scalar_host_tensor.MutableTensorDesc().SetOriginShape(GeShape());
  scalar_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  auto list_scalar_host_tensor = GeTensor(GeTensorDesc(GeShape({128}), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 512);
  list_scalar_host_tensor.MutableTensorDesc().SetOriginShape(GeShape({128}));
  list_scalar_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  auto nd_host_tensor = GeTensor(GeTensorDesc(GeShape({1, 1, 1, 128}), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 512);
  nd_host_tensor.MutableTensorDesc().SetOriginShape(GeShape({1, 1, 1, 128}));
  nd_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  InputData inputs;
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 512, false));
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 512, false));
  inputs.shapes.push_back({});
  inputs.shapes.push_back({});

  input_tensors.resize(2U, scalar_host_tensor);
  output_tensors[0].SetData(nullptr, 0U);
  auto gert_inputs = InputData2GertTensors(inputs);
  ret = executor_rt_v2.ExecuteOnlineModel(gert_inputs, nullptr);
  EXPECT_EQ(ret, SUCCESS);
  HybridModelExecutor::CtrlArgs args;
  args.stream = stream;
  std::vector<gert::Tensor> gert_outputs(1);
  ret = executor_rt_v2.HandleResult(SUCCESS, 0, args, gert_outputs, nullptr);
  EXPECT_EQ(ret, INTERNAL_ERROR);
  RuntimeStub::Reset();
}
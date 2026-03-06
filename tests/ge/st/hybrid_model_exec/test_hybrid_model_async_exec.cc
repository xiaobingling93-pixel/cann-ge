/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <tuple>
#include <gtest/gtest.h>
#include "runtime/rt.h"
#include "macro_utils/dt_public_scope.h"
#include "framework/executor/ge_executor.h"
#include "graph_builder/bg_memory.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/ge_context.h"
#include "graph/manager/host_mem_manager.h"
#include "single_op/single_op.h"
#include "single_op/single_op_manager.h"
#include "utils/model_data_builder.h"
#include "single_op/task/build_task_utils.h"
#include "single_op/task/tbe_task_builder.h"
#include "utils/tensor_descs.h"
#include "utils/data_buffers.h"
#include "register/op_tiling_registry.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/optimize/graph_optimize.h"
#include "hybrid/node_executor/aicore/aicore_node_executor.h"
#include "hybrid/node_executor/ge_local/ge_local_node_executor.h"
#include "graph/manager/mem_manager.h"
#include "graph/manager/graph_external_weight_manager.h"
#include "utils/bench_env.h"
#include "utils/graph_factory.h"
#include "utils/mock_runtime.h"
#include "common/dump/dump_manager.h"
#include "common/profiling/profiling_manager.h"
#include "common/profiling/profiling_properties.h"
#include "common/profiling/profiling_init.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "hybrid/executor/hybrid_model_pipeline_executor.h"
#include "hybrid/executor/hybrid_model_async_executor.h"
#include "hybrid/model/hybrid_model_builder.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/operator_factory_impl.h"
#include "common/share_graph.h"
#include "op_impl/less_important_op_impl.h"
#include "faker/aicore_taskdef_faker.h"
#include "faker/ge_model_builder.h"
#include "faker/magic_ops.h"
#include "graph/operator_reg.h"
#include "graph/ge_attr_value.h"
#include "common/global_variables/diagnose_switch.h"
#include "stub/gert_runtime_stub.h"
#include "src/ascendcl_stub.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "macro_utils/dt_public_unscope.h"
#include "common/model/external_allocator_manager.h"
#include "hybrid/executor/runtime_v2/rt_v2_utils.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_inference.h"
#include "graph/optimize/symbolic/codegen/guard_codegen.h"
#include "common/env_path.h"
#include "rt_error_codes.h"
#include "graph_metadef/depends/checker/tensor_check_utils.h"

using namespace ge;
using namespace gert;
using namespace std;
using namespace optiling::utils;
using namespace ge::hybrid;

namespace {
class MockStreamSync : public ge::RuntimeStub {
 public:
  MOCK_METHOD2(rtStreamSynchronizeWithTimeout, int32_t(rtStream_t stm, int32_t timeout));
  MOCK_METHOD4(rtMalloc, int32_t(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId));
  MOCK_METHOD1(rtFree, int32_t(void *dev_ptr));
};
  REG_OP(ReduceSum)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceSum);
constexpr int64_t kMemtypeHostCompileIndependent = 2;
uint8_t kBufferAddr[1024] = {};
using SingleOpArgsTuple = std::tuple<std::vector<GeTensorDesc>,
    std::vector<DataBuffer>,
    std::vector<GeTensorDesc>,
    std::vector<DataBuffer>>;
SingleOpArgsTuple CreateSingleOpArgsForHostMemInput() {
  std::vector<GeTensorDesc> inputs;
  std::vector<GeTensorDesc> outputs;
  GeShape shape0({4});
  GeTensorDesc tensor_desc0(shape0, FORMAT_ND, DT_UINT64);
  AttrUtils::SetInt(tensor_desc0, ge::ATTR_NAME_PLACEMENT, kMemtypeHostCompileIndependent);
  inputs.emplace_back(tensor_desc0);

  GeShape shape1({1});
  GeTensorDesc tensor_desc1(shape1, FORMAT_ND, DT_UINT64);
  AttrUtils::SetInt(tensor_desc1, ge::ATTR_NAME_PLACEMENT, kMemtypeHostCompileIndependent);
  inputs.emplace_back(tensor_desc1);

  GeShape shape2({4});
  GeTensorDesc tensor_desc2(shape2, FORMAT_ND, DT_UINT64);
  outputs.emplace_back(tensor_desc2);

  std::vector<DataBuffer> input_buffers;
  std::vector<DataBuffer> output_buffers;
  const size_t input0_size = 4 * sizeof(uint64_t);
  const size_t input1_size = 1 * sizeof(uint64_t);
  const size_t output_size = 4 * sizeof(uint64_t);
  uint64_t *addr = PtrToPtr<uint8_t, uint64_t>(kBufferAddr);
  for (size_t i = 0; i < (input0_size + input1_size); i++) {
    addr[i] = kHostMemInputValue;
  }
  input_buffers.emplace_back(DataBuffer(kBufferAddr, input0_size, false, 1));
  input_buffers.emplace_back(DataBuffer(kBufferAddr + input0_size, input1_size, false, 1));
  output_buffers.emplace_back(DataBuffer(kBufferAddr + input0_size + input1_size, output_size));
  return {inputs, input_buffers, outputs, output_buffers};
}

ge::Status CreateFileConstantFile(const std::string &file_location, const ge::float32_t scalar) {
  std::unique_ptr<ge::float32_t[]> float32_t_buf(new ge::float32_t[1]);
  float32_t_buf[0] = scalar;
  std::ofstream out1(file_location, std::ios::binary);
  GE_ASSERT_TRUE(out1.is_open());
  out1.write((char *)float32_t_buf.get(), 1 * sizeof(ge::float32_t));
  out1.close();
  return ge::SUCCESS;
}

class MockMallocFailed : public RuntimeStub {
 public:
  rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) override {
    return -1;
  }
};
class MockAclrtMemcpy : public AclRuntimeStub {
public:
  aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) override {
    return -1;
  }
};
std::vector<gert::Tensor> InputData2GertTensors(const InputData &input_data) {
  std::vector<gert::Tensor> input_tensors;
  for (size_t i = 0U; i < input_data.blobs.size(); ++i) {
    gert::Tensor tensor;
    tensor.MutableTensorData().SetAddr(input_data.blobs[i].data, nullptr);
    tensor.MutableTensorData().SetSize(input_data.blobs[i].length);
    tensor.MutableStorageShape().SetDimNum(input_data.shapes[i].size());
    for (size_t j = 0; j < input_data.shapes[i].size(); ++j) {
      tensor.MutableStorageShape().SetDim(0, input_data.shapes[i][j]);
    }
    input_tensors.emplace_back(std::move(tensor));
  }
  return input_tensors;
}
}
namespace ge {
class HybridModelAsyncTest : public testing::Test {
protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(HybridModelAsyncTest, test_hybrid_model_malloc_failed) {
  ProfilingProperties::Instance().SetLoadProfiling(true);
  BenchEnv::Init();
  uint8_t model_data[8192];
  ge::ModelData modelData{.model_data = model_data};
  ModelDataBuilder(modelData).AddGraph(GraphFactory::HybridSingeOpGraph()).AddTask(2, 2)
  .AddTask(2, 4)
  .AddTask(2, 4)
  .AddTask(2, 5)
  .AddTask(2, 5)
  .Build();

  auto data_buffers = DataBuffers(3);
  auto buffers = data_buffers.Value();
  auto input_desc = TensorDescs(2).Value();
  std::vector<DataBuffer> input_buffers{buffers[0], buffers[1]};
  auto output_desc = TensorDescs(1).Value();
  std::vector<DataBuffer> output_buffers{buffers[2]};

  ge::DynamicSingleOp *singleOp = nullptr;
  EXPECT_EQ(ge::GeExecutor::LoadDynamicSingleOpV2("dynamic_op_fail", modelData, nullptr, &singleOp, 6), SUCCESS);
  auto malloc_mock = std::make_shared<MockMallocFailed>();
  RuntimeStub::SetInstance(malloc_mock);
  EXPECT_EQ(singleOp->ExecuteAsync(input_desc, input_buffers, output_desc, output_buffers), ACL_ERROR_GE_DEVICE_MEMORY_OPERATE_FAILED);
  RuntimeStub::SetInstance(nullptr);
}

TEST_F(HybridModelAsyncTest, test_hybrid_model_dynamic_shape_success) {
  ProfilingProperties::Instance().SetLoadProfiling(true);
  BenchEnv::Init();
  uint8_t model_data[8192];
  ge::ModelData modelData{.model_data = model_data};
  ModelDataBuilder(modelData).AddGraph(GraphFactory::HybridSingeOpGraph()).AddTask(2, 2)
  .AddTask(2, 4)
  .AddTask(2, 4)
  .AddTask(2, 5)
  .AddTask(2, 5)
  .Build();

  auto data_buffers = DataBuffers(3);
  auto buffers = data_buffers.Value();
  auto input_desc = TensorDescs(2).Value();
  std::vector<DataBuffer> input_buffers{buffers[0], buffers[1]};
  auto output_desc = TensorDescs(1).Value();
  std::vector<DataBuffer> output_buffers{buffers[2]};

  ge::DynamicSingleOp *singleOp = nullptr;
  EXPECT_EQ(ge::GeExecutor::LoadDynamicSingleOpV2("dynamic_op", modelData, nullptr, &singleOp, 4), SUCCESS);
  EXPECT_EQ(singleOp->ExecuteAsync(input_desc, input_buffers, output_desc, output_buffers), SUCCESS);
}

TEST_F(HybridModelAsyncTest, test_singleop_dynamic_shape_success) {
  ProfilingProperties::Instance().SetLoadProfiling(true);
  BenchEnv::Init();
  uint8_t model_data[8192];
  ge::ModelData modelData{.model_data = model_data};
  ModelDataBuilder(modelData).AddGraph(GraphFactory::SingeOpGraph()).AddTask(2, 2).Build();

  auto data_buffers = DataBuffers(2);
  auto buffers = data_buffers.Value();
  auto input_desc = TensorDescs(1).Value();
  std::vector<DataBuffer> input_buffers{buffers[0]};
  auto output_desc = TensorDescs(1).Value();
  std::vector<DataBuffer> output_buffers{buffers[1]};

  ge::DynamicSingleOp *single_op = nullptr;
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/MyTransdata", std::move(kernelBin));
  auto holder = std::unique_ptr<KernelHolder>(new (std::nothrow) KernelHolder("0/_tvmbin", tbe_kernel));
  KernelBinRegistry::GetInstance().AddKernel("0/_tvmbin", std::move(holder));
  EXPECT_EQ(ge::GeExecutor::LoadDynamicSingleOpV2("dynamic_op2", modelData, nullptr, &single_op, 0), SUCCESS);
  EXPECT_EQ(single_op->ExecuteAsync(input_desc, input_buffers, output_desc, output_buffers), SUCCESS);
}

// dynamic hybrid , aicore with host mem input
TEST_F(HybridModelAsyncTest, test_singleop_with_hostmem_success) {
  auto runtime_stub = MockForKernelLaunchWithHostMemInput();
  BenchEnv::Init();
  uint8_t model_data[8192];
  ge::ModelData modelData{.model_data = model_data};
  ModelDataBuilder(modelData).AddGraph(GraphFactory::HybridSingeOpGraphForHostMemInput()).AddTask(2, 2)
      .AddTask(2, 4)
      .AddTask(2, 4)
      .AddTask(2, 5)
      .AddTask(2, 5)
      .Build();
  SingleOpArgsTuple arg = CreateSingleOpArgsForHostMemInput();

  ge::DynamicSingleOp *single_op = nullptr;
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/transdata", std::move(kernelBin));
  auto holder = std::unique_ptr<KernelHolder>(new (std::nothrow) KernelHolder("0/_tvmbin", tbe_kernel));
  KernelBinRegistry::GetInstance().AddKernel("0/_tvmbin", std::move(holder));
  EXPECT_EQ(ge::GeExecutor::LoadDynamicSingleOpV2("dynamic_op2", modelData, nullptr, &single_op, 110), SUCCESS);
  EXPECT_EQ(single_op->ExecuteAsync(get<0>(arg), get<1>(arg), get<2>(arg), get<3>(arg)), SUCCESS);
}

TEST_F(HybridModelAsyncTest, test_singleop_aicpu_load_success) {
  BenchEnv::Init();
  uint8_t model_data[8192];
  ge::ModelData modelData{.model_data = model_data};
  auto graph = GraphFactory::SingeOpGraph2();
  ComputeGraphPtr compute_graph_ = ge::GraphUtilsEx::GetComputeGraph(graph);
  AttrUtils::SetInt(compute_graph_, "globalworkspace_type", 1);
  AttrUtils::SetInt(compute_graph_, "globalworkspace_size", 512);
  graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph_);
  ModelDataBuilder(modelData).AddGraph(graph).AddAicpuTask(2).Build();

  auto data_buffers = DataBuffers(2);
  auto buffers = data_buffers.Value();
  auto input_desc = TensorDescs(1).Value();
  std::vector<DataBuffer> input_buffers{buffers[0]};
  auto output_desc = TensorDescs(1).Value();
  std::vector<DataBuffer> output_buffers{buffers[1]};

  ge::DynamicSingleOp *single_op = nullptr;
  EXPECT_EQ(ge::GeExecutor::LoadDynamicSingleOpV2("dynamic_op2", modelData, nullptr, &single_op, 112), SUCCESS);
}

TEST_F(HybridModelAsyncTest, test_hybird_with_hostmem_success) {
  BenchEnv::Init();
  uint8_t model_data[8192];
  ge::ModelData modelData{.model_data = model_data};
  ModelDataBuilder(modelData).AddGraph(GraphFactory::SingeOpGraph3()).AddTask(2, 3).Build();

  ge::DynamicSingleOp *single_op = nullptr;
  EXPECT_EQ(ge::GeExecutor::LoadDynamicSingleOpV2("dynamic_op2", modelData, nullptr, &single_op, 3), SUCCESS);
}

TEST_F(HybridModelAsyncTest, test_pipeline_execute_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder builder(hybrid_model);
  EXPECT_EQ(builder.Build(), SUCCESS);

  HybridModelPipelineExecutor pipeline_executor(&hybrid_model, 1, nullptr);
  EXPECT_EQ(pipeline_executor.Init(), SUCCESS);

  StageSubject stage_subject;
  stage_subject.Release(1);
  EXPECT_EQ(stage_subject.Await(1), SUCCESS);
  InputData input_data;
  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[3072]);
  input_data.blobs.push_back(DataBuffer(data_buf.get(), 3072, false));
  input_data.shapes.push_back({1, 16, 16, 3});
  std::vector<gert::Tensor> inputs = InputData2GertTensors(input_data);
  EXPECT_EQ(pipeline_executor.ExecuteOnlineModel(inputs, nullptr), SUCCESS);
}

TEST_F(HybridModelAsyncTest, test_pipeline_stage_execute_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_ = std::unique_ptr<GraphItem>(new(std::nothrow)GraphItem());

  PipeExecutionConfig config;
  config.device_id = 1;
  config.num_executors = 2;
  config.num_stages = 1;
  config.iteration_end = 2;
  rtCtxGetCurrent(&config.rt_context);

  HybridModelPipelineExecutor pip_executor(&hybrid_model, 1, nullptr);
  HybridModelExecutor::ExecuteArgs args;
  ASSERT_EQ(pip_executor.config_.iteration_end, 0);
  ASSERT_EQ(pip_executor.Init(), SUCCESS);
  pip_executor.Execute(args);

  StageSubject stage_subject;
  StageExecutor executor(0, &hybrid_model, &config, &stage_subject);
  StageExecutor next_executor(1, &hybrid_model, &config, &stage_subject);
  EXPECT_EQ(stage_subject.Await(1), SUCCESS);
  executor.SetNext(&next_executor);
  EXPECT_EQ(executor.Init(), SUCCESS);

  auto allocator = NpuMemoryAllocator::GetAllocator();
  EXPECT_NE(allocator, nullptr);
  StageExecutor::StageTask task_info_1;
  task_info_1.stage = 0;
  task_info_1.iteration = 0;
  EXPECT_EQ(rtEventCreate(&task_info_1.event), RT_ERROR_NONE);
  EXPECT_EQ(executor.ExecuteAsync(task_info_1), SUCCESS);
  EXPECT_EQ(executor.Start({}, {}, 2), SUCCESS);

  StageExecutor::StageTask task_info_2;
  task_info_2.stage = 0;
  task_info_2.iteration = 1;
  const char_t *const kEnvRecordPath = "CONSTANT_FOLDING_PASS_9";
  char_t record_path[MMPA_MAX_PATH] = "mock_fail";
  mmSetEnv(kEnvRecordPath, &record_path[0U], MMPA_MAX_PATH);
  EXPECT_EQ(rtEventCreate(&task_info_2.event), RT_ERROR_NONE);
  EXPECT_EQ(executor.ExecuteAsync(task_info_2), SUCCESS);
  EXPECT_NE(executor.Start({}, {}, 2), SUCCESS);

  StageExecutor::StageTask task_info_3;
  task_info_3.stage = 0;
  task_info_3.iteration = 2;
  EXPECT_EQ(rtEventCreate(&task_info_3.event), RT_ERROR_NONE);
  EXPECT_EQ(executor.ExecuteAsync(task_info_3), SUCCESS);
  EXPECT_NE(executor.Start({}, {}, 2), SUCCESS);

  // release the memory held by task_info.event
  // task_info_2 and task_info_3 need to be released
  EXPECT_EQ(rtEventDestroy(task_info_2.event), RT_ERROR_NONE);
  EXPECT_EQ(rtEventDestroy(task_info_3.event), RT_ERROR_NONE);

  executor.ExecuteEndTaskAndReleae();
  executor.Reset();
  unsetenv(kEnvRecordPath);
}

TEST_F(HybridModelAsyncTest, test_ascend_aicpu_load_success) {
  ProfilingProperties::Instance().SetLoadProfiling(true);
  ge::DumpConfig dump_cfg;
  dump_cfg.dump_path = "./dump/";
  dump_cfg.dump_mode = "all";
  dump_cfg.dump_debug = "on";
  DumpManager::GetInstance().SetDumpConf(dump_cfg);
  BenchEnv::Init();
  uint8_t model_data[8192];
  ge::ModelData modelData{.model_data = model_data};
  auto graph = GraphFactory::BuildAicpuSingeOpGraph();
  auto computeGraph = GraphUtilsEx::GetComputeGraph(graph);
  auto data1 = computeGraph->FindNode("data1");
  auto data2 = computeGraph->FindNode("data2");
  auto transdata1 = computeGraph->FindNode("transdata1");
  auto transdata2 = computeGraph->FindNode("transdata2");
  data1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1,1,2,3}));
  data2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1,1,2,3}));

  transdata1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({1,1,2,3}));
  transdata1->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({1,1,2,3}));
  transdata2->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({1,1,2,3}));
  transdata2->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({1,1,2,3}));
  auto infer_fun = [](Operator &op) -> graphStatus {
    return GRAPH_SUCCESS;
  };
  OperatorFactoryImpl::RegisterInferShapeFunc(DATA, infer_fun);

  ModelDataBuilder(modelData).AddGraph(graph).AddTask(2, 2)
  .AddAicpuTask(4)
  .AddAicpuTask(5)
  .Build();

  auto data_buffers = DataBuffers(3);
  auto buffers = data_buffers.Value();
  auto input_desc = TensorDescs(2).Value();
  std::vector<DataBuffer> input_buffers{buffers[0], buffers[1]};
  auto output_desc = TensorDescs(1).Value();
  std::vector<DataBuffer> output_buffers{buffers[2]};

  ge::DynamicSingleOp *singleOp = nullptr;
  EXPECT_EQ(ge::GeExecutor::LoadDynamicSingleOpV2("dynamic_op3", modelData, nullptr, &singleOp, 5), SUCCESS);
  EXPECT_EQ(singleOp->ExecuteAsync(input_desc, input_buffers, output_desc, output_buffers), SUCCESS);
  ge::diagnoseSwitch::DisableDumper();
}

TEST_F(HybridModelAsyncTest, test_dynamic_shape_with_squeezev3_success) {
  ProfilingProperties::Instance().SetLoadProfiling(false);
  DEF_GRAPH(dynamic_op) {
    auto op_ptr = OP_CFG(DATA)
                      .InCnt(1)
                      .OutCnt(1)
                      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 3, 4})
                      .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                      .Attr("compile_info_key", "ddd")
                      .Attr("compile_info_json", "cccc")
                      .Build("data1");

    auto op_ptr2 = OP_CFG(DATA)
                       .InCnt(1)
                       .OutCnt(1)
                       .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1})
                       .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                       .Attr("compile_info_key", "ddd")
                       .Attr("compile_info_json", "cccc")
                       .Attr("_force_unknown_shape", true)
                       .Build("axes");

    auto squeezev3 = OP_CFG(SQUEEZEV3)
                         .InCnt(2)
                         .OutCnt(1)
                         .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                         .Attr("_force_infershape_when_running", true)
                         .Attr("_force_unknown_shape", true)
                         .Build("SqueezeV3");

    auto net_output = OP_CFG(NETOUTPUT)
                          .InCnt(1)
                          .OutCnt(1)
                          .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                          .Build("net_output");

    CHAIN(NODE(op_ptr)->EDGE(0, 0)->NODE(squeezev3)->EDGE(0, 0)->NODE(net_output));
    CHAIN(NODE(op_ptr2)->EDGE(0, 1)->NODE(squeezev3));
  };

  const auto ShapeInfer = [](Operator &op) {
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    const auto input_desc_x = op_desc->MutableInputDesc(0);
    const auto &input_shape = input_desc_x->GetShape();
    const auto x_shape_dim = input_shape.GetDims();
    auto output_desc = op_desc->MutableOutputDesc(0);

    output_desc->SetDataType(input_desc_x->GetDataType());
    output_desc->SetOriginDataType(input_desc_x->GetDataType());

    const std::vector<string> dep_inputs = {"axes"};
    op_desc->SetOpInferDepends(dep_inputs);
    std::vector<int64_t> axes_val;
    const auto axes_idx = static_cast<uint32_t>(op_desc->GetInputIndexByName("axes"));
    const GeTensor *tensor = OpDescUtils::GetInputConstData(op, axes_idx);
    if (tensor != nullptr) {
      auto pbuff = tensor->GetData().GetData();
      if (pbuff == nullptr) {
        GELOGE(FAILED, "[InferShape] Get data from axis input failed, as data buff is null");
        return GRAPH_FAILED;
      }
      const auto axes_len = tensor->GetData().GetSize();
      auto axes_pbuff = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(pbuff));
      for (size_t i = 0UL; i < (axes_len / sizeof(int64_t)); ++i) {
        axes_val.emplace_back(axes_pbuff[i]);
      }
    } else {
      GELOGW("Op get input const data of axes failed");
    }
    GeShape &output_shape = output_desc->MutableShape();
    const auto dim_size = x_shape_dim.size();
    output_shape.SetDimNum(dim_size);

    std::vector<int64_t> dim_idx(dim_size);
    std::vector<std::pair<int64_t, int64_t>> output_range;
    std::for_each(axes_val.begin(), axes_val.end(), [&dim_idx](const int64_t axis) { dim_idx[axis] = -1; });

    uint64_t idx = 0UL;
    for (size_t i = 0UL; i < dim_size; ++i) {
      if (dim_idx[i] != -1) {
        output_shape.SetDim(idx, x_shape_dim[i]);
        ++idx;
      }
    }
    output_shape.SetDimNum(idx);
    output_desc->SetShapeRange(output_range);
    output_desc->SetOriginShape(output_shape);
    return GRAPH_SUCCESS;
  };
  auto graph = ToGeGraph(dynamic_op);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto squeezev3 = compute_graph->FindNode("SqueezeV3");
  std::map<string, uint32_t> names;
  names["input0"] = 0;
  names["axes"] = 1;
  squeezev3->GetOpDesc()->UpdateInputName(names);
  auto axes_desc = squeezev3->GetOpDesc()->MutableInputDesc(1);
  GeTensor weight;
  int64_t data = 0;
  weight.SetData(reinterpret_cast<uint8_t *>(&data), sizeof(int64_t));
  GeTensorDesc weight_desc;
  weight.SetTensorDesc(weight_desc);
  GeTensorPtr tensor = MakeShared<GeTensor>(weight);
  AttrUtils::SetTensor(axes_desc, ATTR_NAME_VALUE, tensor);
  auto graph_2 = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  BenchEnv::Init();
  uint8_t model_data[8192];
  ge::ModelData modelData{.model_data = model_data};
  ModelDataBuilder(modelData).AddGraph(graph_2).AddTask(2, 3).Build();

  // avoid stack-use-after-scope since buffer is DataBuffers's private member
  auto data_buffers = DataBuffers(3, true);
  auto buffers = data_buffers.Value();
  auto input_desc = TensorDescs(2).Value();
  std::vector<DataBuffer> input_buffers{buffers[0], buffers[1]};
  auto output_desc = TensorDescs(1).Value();
  std::vector<DataBuffer> output_buffers{buffers[2]};

  ge::DynamicSingleOp *singleOp = nullptr;
  EXPECT_EQ(ge::GeExecutor::LoadDynamicSingleOpV2("dynamic_op", modelData, nullptr, &singleOp, 1), SUCCESS);

  OperatorFactoryImpl::operator_infershape_funcs_->emplace("SqueezeV3", ShapeInfer);
  EXPECT_EQ(singleOp->ExecuteAsync(input_desc, input_buffers, output_desc, output_buffers), SUCCESS);
}

TEST_F(HybridModelAsyncTest, test_dynamic_shape_with_unsqueezev3_success) {
  DEF_GRAPH(dynamic_op) {
    auto op_ptr = OP_CFG(DATA)
                      .InCnt(1)
                      .OutCnt(1)
                      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 3, 4})
                      .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                      .Attr("compile_info_key", "ddd")
                      .Attr("compile_info_json", "cccc")
                      .Build("data1");

    auto op_ptr2 = OP_CFG(DATA)
                       .InCnt(1)
                       .OutCnt(1)
                       .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1})
                       .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                       .Attr("compile_info_key", "ddd")
                       .Attr("compile_info_json", "cccc")
                       .Attr("_force_unknown_shape", true)
                       .Build("axes");

    auto unsqueezev3 = OP_CFG(UNSQUEEZEV3)
                         .InCnt(2)
                         .OutCnt(1)
                         .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                         .Attr("_force_infershape_when_running", true)
                         .Attr("_force_unknown_shape", true)
                         .Build("UnsqueezeV3");

    auto net_output = OP_CFG(NETOUTPUT)
                          .InCnt(1)
                          .OutCnt(1)
                          .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                          .Build("net_output");

    CHAIN(NODE(op_ptr)->EDGE(0, 0)->NODE(unsqueezev3)->EDGE(0, 0)->NODE(net_output));
    CHAIN(NODE(op_ptr2)->EDGE(0, 1)->NODE(unsqueezev3));
  };

  const auto ShapeInfer = [](Operator &op) {
    return GRAPH_SUCCESS;
  };

  auto graph = ToGeGraph(dynamic_op);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto unsqueezev3 = compute_graph->FindNode("UnsqueezeV3");
  std::map<string, uint32_t> names;
  names["input0"] = 0;
  names["axes"] = 1;
  unsqueezev3->GetOpDesc()->UpdateInputName(names);
  auto axes_desc = unsqueezev3->GetOpDesc()->MutableInputDesc(1);
  GeTensor weight;
  int64_t data = 0;
  weight.SetData(reinterpret_cast<uint8_t *>(&data), sizeof(int64_t));
  GeTensorDesc weight_desc;
  weight.SetTensorDesc(weight_desc);
  GeTensorPtr tensor = MakeShared<GeTensor>(weight);
  AttrUtils::SetTensor(axes_desc, ATTR_NAME_VALUE, tensor);
  auto graph_2 = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  BenchEnv::Init();
  uint8_t model_data[8192];
  ge::ModelData modelData{.model_data = model_data};
  ModelDataBuilder(modelData).AddGraph(graph_2).AddTask(2, 3).Build();

  // avoid stack-use-after-scope since buffer is DataBuffers's private member
  auto data_buffers = DataBuffers(3, true);
  auto buffers = data_buffers.Value();
  auto input_desc = TensorDescs(2).Value();
  std::vector<DataBuffer> input_buffers;
  input_buffers.emplace_back(buffers[0]);
  input_buffers.emplace_back(buffers[1]);
  auto output_desc = TensorDescs(1).Value();
  std::vector<DataBuffer> output_buffers;
  output_buffers.emplace_back(buffers[2]);

  ge::DynamicSingleOp *singleOp = nullptr;
  EXPECT_EQ(ge::GeExecutor::LoadDynamicSingleOpV2("dynamic_op", modelData, nullptr, &singleOp, 1), SUCCESS);

  OperatorFactoryImpl::operator_infershape_funcs_->emplace("UnsqueezeV3", ShapeInfer);
  EXPECT_EQ(singleOp->ExecuteAsync(input_desc, input_buffers, output_desc, output_buffers), SUCCESS);
}

TEST_F(HybridModelAsyncTest, test_hybrid_model_dynamic_shape_failed) {
  ProfilingProperties::Instance().SetLoadProfiling(false);
  BenchEnv::Init();
  uint8_t model_data[8192];
  ge::ModelData modelData{.model_data = model_data};
  ModelDataBuilder(modelData).AddGraph(GraphFactory::HybridSingeOpGraph2()).AddTask(2, 2)
  .AddTask(2, 5)
  .AddTask(2, 6)
  .Build();

  auto data_buffers = DataBuffers(3);
  auto buffers = data_buffers.Value();
  auto input_desc = TensorDescs(2).Value();
  std::vector<DataBuffer> input_buffers{buffers[0], buffers[1]};
  auto output_desc = TensorDescs(1).Value();
  // create invalid output buffer whose length < expected_size
  // this will return GRAPH_PARAM_INVALID
  buffers[2].length = 1;
  std::vector<DataBuffer> output_buffers{buffers[2]};

  ge::DynamicSingleOp *singleOp = nullptr;
  EXPECT_EQ(ge::GeExecutor::LoadDynamicSingleOpV2("dynamic_op", modelData, nullptr, &singleOp, 1), SUCCESS);
  EXPECT_EQ(singleOp->ExecuteAsync(input_desc, input_buffers, output_desc, output_buffers), GRAPH_PARAM_INVALID);
}

TEST_F(HybridModelAsyncTest, Test_execute_with_deivce_placement_success) {
  auto graph = ShareGraph::SimpleFooGraph();
  graph->TopologicalSorting();
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Foo") {
      MockLessImportantNodeKernel(node);
    }
  }
  GertRuntimeStub runtime_stub;
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
  std::vector<DataBuffer> inputs;
  std::vector<GeTensorDesc> input_desc;
  std::vector<DataBuffer> outputs;
  std::vector<GeTensorDesc> output_desc;

  unique_ptr<uint8_t[]> data_buf = std::make_unique<uint8_t[]>(3072);
  DataBuffer buffer(data_buf.get(), 3072, false, static_cast<uint32_t>(Placement::kPlacementDevice));
  GeTensorDescPtr tensor_desc = make_shared<GeTensorDesc>(GeShape({-1, 16, 16, 3}));
  tensor_desc->SetShapeRange({{1, 256}, {16, 16}, {16, 16}, {3, 3}});
  input_desc.push_back(*tensor_desc);
  inputs.emplace_back(buffer);
  runtime_stub.Clear();
  ASSERT_EQ(executor.Execute(inputs, input_desc, outputs, output_desc), SUCCESS);
  ASSERT_TRUE(runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords().empty());
}

TEST_F(HybridModelAsyncTest, Test_init_with_stream_success) {
  auto graph = ShareGraph::SimpleFooGraph();
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Foo") {
      MockLessImportantNodeKernel(node);
    }
  }
  GertRuntimeStub runtime_stub;
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
  rtStream_t stream = (void*)0x01;
  EXPECT_EQ(executor.Init(stream), SUCCESS);
  EXPECT_NE(executor.executor_, nullptr);
  std::vector<DataBuffer> inputs;
  std::vector<GeTensorDesc> input_desc;
  std::vector<DataBuffer> outputs;
  std::vector<GeTensorDesc> output_desc;

  unique_ptr<uint8_t[]> data_buf = std::make_unique<uint8_t[]>(3072);
  DataBuffer buffer(data_buf.get(), 3072, false, static_cast<uint32_t>(Placement::kPlacementDevice));
  GeTensorDescPtr tensor_desc = make_shared<GeTensorDesc>(GeShape({-1, 16, 16, 3}));
  tensor_desc->SetShapeRange({{1, 256}, {16, 16}, {16, 16}, {3, 3}});
  input_desc.push_back(*tensor_desc);
  inputs.emplace_back(buffer);
  runtime_stub.Clear();
  ASSERT_EQ(executor.Execute(inputs, input_desc, outputs, output_desc), SUCCESS);
}

/*
 * 用例描述: 加载执行动态shape模型，依赖执行器推导输出shape

 * 预置条件：
 * 1. 构造动态shape模型,走RT2执行器，输出构造为空，依赖RT2执行其返回
 *
 * 测试步骤：
 * 1. 构造一个包含Add节点的动态shape图，其中输入输出的dtype构造成int4类型
 * 2. 模型编译
 * 3. 将输入填好，输出构造为空指针
 * 4. 模型执行
 *
 * 预期结果：
 * 1. 模型执行成功
 * 2. 输出的shape的size大小为1536, size is (4 * 16 * 16 * 3) * 4 / 8 = 1536
 * 3. 输出的地址不为空指针
 */
TEST_F(HybridModelAsyncTest, Test_run_with_stream_async_with_int4) {
  auto graph = ShareGraph::AicoreGraph();
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelInfo();
  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  ASSERT_NE(netoutput, nullptr);
  auto desc = netoutput->GetOpDesc()->MutableInputDesc(0);
  ASSERT_NE(desc, nullptr);
  desc->SetDataType(ge::DT_INT4);

  auto data = graph->FindFirstNodeMatchType("Data");
  data->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_INT4);
  data->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_INT4);

  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  HybridModelAsyncExecutor executor(&hybrid_model);
  rtStream_t stream = (void*)0x01;
  EXPECT_EQ(executor.Init(stream), SUCCESS);
  EXPECT_NE(executor.executor_, nullptr);
  unique_ptr<uint8_t[]> data_buf = std::make_unique<uint8_t[]>(3072);
  std::vector<GeTensor> input_tensors(2U);
  std::vector<GeTensor> output_tensors(1U);

  for (size_t i = 0U; i < input_tensors.size(); ++i) {
    input_tensors[i].MutableTensorDesc() = GeTensorDesc(GeShape({4, 16, 16, 3}), FORMAT_NCHW, DT_INT4);
    input_tensors[i].MutableTensorDesc().SetOriginShape(GeShape({4, 16, 16, 3}));
    input_tensors[i].MutableTensorDesc().SetPlacement(Placement::kPlacementDevice);
    input_tensors[i].SetData(data_buf.get(), 3072);
  }
  for (size_t i = 0U; i < output_tensors.size(); ++i) {
    output_tensors[i].SetData(nullptr, 0U);
  }

  runtime_stub.Clear();
  ASSERT_EQ(executor.ExecuteWithStreamAsync(input_tensors, output_tensors, stream), SUCCESS);

  // check outputsize for int4, size is (4 * 16 * 16 * 3) * 4 / 8 = 1536
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    EXPECT_NE(output_tensors[i].GetData().GetData(), nullptr);
    EXPECT_EQ(output_tensors[i].GetData().GetSize(), 1536);
  }
  runtime_stub.GetSlogStub().SetLevel(DLOG_ERROR);
}

/*
 * 用例描述: 加载执行动态shape模型，依赖执行器推导输出shape

 * 预置条件：
 * 1. 构造动态shape模型,走RT2执行器，输出构造为空，依赖RT2执行其返回
 *
 * 测试步骤：
 * 1. 构造一个包含Add节点的动态shape图，其中输入输出的dtype构造成int4类型
 * 2. 模型编译
 * 3. 将输入填好，输出构造为空指针
 * 4. 模型执行
 *
 * 预期结果：
 * 1. 模型执行成功
 * 2. 输出的shape的size大小为1536, size is (4 * 16 * 16 * 3) * 4 / 8 = 1536
 * 3. 输出的地址不为空指针
 */
TEST_F(HybridModelAsyncTest, Test_execute_with_stream_async_with_int4) {
  auto graph = ShareGraph::AicoreGraph();
  ge::AttrUtils::SetBool(graph, "need_set_stream_core_limits", true);
  std::map<std::string, std::string> options;
  options["ge.aicoreNum"] = "1";
  options["ge.vectorcoreNum"] = "1";
  ge::GetThreadLocalContext().SetSessionOption(options);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevel(DLOG_INFO);
  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  ASSERT_NE(netoutput, nullptr);
  auto desc = netoutput->GetOpDesc()->MutableInputDesc(0);
  ASSERT_NE(desc, nullptr);
  desc->SetDataType(ge::DT_INT4);

  auto data = graph->FindFirstNodeMatchType("Data");
  data->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_INT4);
  data->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_INT4);

  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  HybridModelAsyncExecutor executor(&hybrid_model);
  rtStream_t stream = (void*)0x01;
  EXPECT_EQ(executor.Init(stream), SUCCESS);
  EXPECT_NE(executor.executor_, nullptr);
  unique_ptr<uint8_t[]> data_buf = std::make_unique<uint8_t[]>(3072);
  std::vector<gert::Tensor> input_tensors(2U);
  std::vector<gert::Tensor> output_tensors(1U);
  std::vector<int64_t> shape = {4, 16, 16, 3};

  for (size_t i = 0U; i < input_tensors.size(); ++i) {
    DimsAsShape(shape, input_tensors[i].MutableStorageShape());
    input_tensors[i].SetStorageFormat(FORMAT_NCHW);
    input_tensors[i].SetDataType(DT_INT4);
    DimsAsShape(shape, input_tensors[i].MutableOriginShape());

    input_tensors[i].SetPlacement(gert::kOnDeviceHbm);
    input_tensors[i].MutableTensorData().SetAddr((void*)data_buf.get(), nullptr);
    input_tensors[i].SetSize(3072);
  }
  for (size_t i = 0U; i < output_tensors.size(); ++i) {
    output_tensors[i].MutableTensorData().SetAddr(nullptr, nullptr);
    output_tensors[i].SetSize(0);
  }

  runtime_stub.Clear();
  //ASSERT_EQ(executor.ExecuteWithStreamAsync(input_tensors, output_tensors, stream), SUCCESS);
  executor.ExecuteWithStreamAsync(input_tensors, output_tensors, stream);
  // check outputsize for int4, size is (4 * 16 * 16 * 3) * 4 / 8 = 1536
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    EXPECT_NE(output_tensors[i].GetAddr(), nullptr);
    EXPECT_EQ(output_tensors[i].GetSize(), 1536);
  }
  runtime_stub.GetSlogStub().SetLevel(DLOG_ERROR);
  std::map<std::string, std::string> options_empty;
  ge::GetThreadLocalContext().SetSessionOption(options_empty);
}

TEST_F(HybridModelAsyncTest, Test_multi_thread_executor_with_invalid_value) {
  char_t max_runtime_core_path[MMPA_MAX_PATH] = "aa";
  mmSetEnv("MAX_RUNTIME_CORE_NUMBER", &max_runtime_core_path[0U], MMPA_MAX_PATH);
  auto graph = ShareGraph::AicoreGraph();
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelInfo();
  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  ASSERT_NE(netoutput, nullptr);
  auto desc = netoutput->GetOpDesc()->MutableInputDesc(0);
  ASSERT_NE(desc, nullptr);
  desc->SetDataType(ge::DT_INT4);

  auto data = graph->FindFirstNodeMatchType("Data");
  data->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_INT4);
  data->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_INT4);

  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  HybridModelAsyncExecutor executor(&hybrid_model);
  rtStream_t stream = (void *)0x01;
  EXPECT_NE(executor.Init(stream), SUCCESS);
  unsetenv("MAX_RUNTIME_CORE_NUMBER");
}

TEST_F(HybridModelAsyncTest, Test_multi_thread_executor_success) {
  char_t max_runtime_core_path[MMPA_MAX_PATH] = "2";
  mmSetEnv("MAX_RUNTIME_CORE_NUMBER", &max_runtime_core_path[0U], MMPA_MAX_PATH);
  auto graph = ShareGraph::AicoreGraph();
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelInfo();
  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  ASSERT_NE(netoutput, nullptr);
  auto desc = netoutput->GetOpDesc()->MutableInputDesc(0);
  ASSERT_NE(desc, nullptr);
  desc->SetDataType(ge::DT_INT4);

  auto data = graph->FindFirstNodeMatchType("Data");
  data->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_INT4);
  data->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_INT4);

  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  HybridModelAsyncExecutor executor(&hybrid_model);
  rtStream_t stream = (void *)0x01;
  EXPECT_EQ(executor.Init(stream), SUCCESS);
  unsetenv("MAX_RUNTIME_CORE_NUMBER");
}

TEST_F(HybridModelAsyncTest, ExecuteWithStreamAsync_execute_model_online_host_input) {
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
  std::map<std::string, std::string> options_empty;
  ge::GetThreadLocalContext().SetSessionOption(options_empty);
}

TEST_F(HybridModelAsyncTest, ExecuteWithStreamAsync_execute_model_online_BatchH2d) {
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

  input_tensors.resize(2U, scalar_host_tensor);
  output_tensors[0].SetData(nullptr, 0U);
  std::vector<gert::Tensor> gert_inputs = InputData2GertTensors(inputs);
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


TEST_F(HybridModelAsyncTest, Execute_Success_SkipBatchMemcpyWhenSizeIsZero) {
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

  std::vector<GeTensor> input_tensors(2U, empty_tensor);
  std::vector<GeTensor> output_tensors(1U);
  output_tensors[0].SetData(nullptr, 0U);
  std::vector<gert::Tensor> gert_inputs = InputData2GertTensors(inputs);
  ret = executor_rt_v2.ExecuteOnlineModel(gert_inputs, nullptr);

  EXPECT_EQ(ret, SUCCESS);
  RuntimeStub::Reset();
  ge::GetThreadLocalContext().SetSessionOption(options);
  ge::GetThreadLocalContext().SetGraphOption(options);
}

TEST_F(HybridModelAsyncTest, ExecuteWithStreamAsync_execute_model_online_BatchH2d_NotSupport) {
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
  std::vector<gert::Tensor> gert_inputs = InputData2GertTensors(inputs);
  ret = executor_rt_v2.ExecuteOnlineModel(gert_inputs, nullptr);
  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 0);
  EXPECT_EQ(ret, SUCCESS);

  RuntimeStub::Reset();
  options["ge.inputBatchCpy"] = "0";
  ge::GetThreadLocalContext().SetSessionOption(options);
  ge::GetThreadLocalContext().SetGraphOption(options);
}

TEST_F(HybridModelAsyncTest, ExecuteWithStreamAsync_execute_model_online_BatchH2d_Failed) {
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
  std::vector<gert::Tensor> gert_inputs = InputData2GertTensors(inputs);
  ret = executor_rt_v2.ExecuteOnlineModel(gert_inputs, nullptr);
  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 0);
  EXPECT_NE(ret, SUCCESS);

  RuntimeStub::Reset();
  options["ge.inputBatchCpy"] = "0";
  ge::GetThreadLocalContext().SetSessionOption(options);
  ge::GetThreadLocalContext().SetGraphOption(options);
}

TEST_F(HybridModelAsyncTest, ExecuteWithStreamAsync_execute_model_online_BatchH2d_FallbackFailed) {
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
  MockAclrtMemcpy mock_aclrt_memcpy;
  AclRuntimeStub::Install(&mock_aclrt_memcpy);
  std::vector<gert::Tensor> gert_inputs = InputData2GertTensors(inputs);
  ret = executor_rt_v2.ExecuteOnlineModel(gert_inputs, nullptr);
  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 0);
  EXPECT_NE(ret, SUCCESS);

  AclRuntimeStub::UnInstall(&mock_aclrt_memcpy);
  RuntimeStub::Reset();
  options["ge.inputBatchCpy"] = "0";
  ge::GetThreadLocalContext().SetSessionOption(options);
  ge::GetThreadLocalContext().SetGraphOption(options);
}

TEST_F(HybridModelAsyncTest, ExecuteWithStreamAsync_execute_model_online_BatchH2dOneInputFallback) {
  auto graph = ShareGraph::SimpleFooGraph();
  ge::AttrUtils::SetBool(graph, "need_set_stream_core_limits", true);
  auto foo = graph->FindFirstNodeMatchType("Foo");
  ge::AttrUtils::SetStr(foo->GetOpDesc(), "_ge_attr_lowering_func", "_lower_foo_with_stream_sync");
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  std::map<std::string, std::string> options;
  options["ge.inputBatchCpy"] = "1";
  options["ge.aicoreNum"] = "1";
  options["ge.vectorcoreNum"] = "1";
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
  std::map<std::string, std::string> options_empty;
  ge::GetThreadLocalContext().SetSessionOption(options_empty);
  ge::GetThreadLocalContext().SetGraphOption(options_empty);
}

TEST_F(HybridModelAsyncTest, ExecuteWithStreamAsync_execute_model_online_BatchH2dOneInputFallbackFailed) {
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
  MockAclrtMemcpy mock_aclrt_memcpy;
  AclRuntimeStub::Install(&mock_aclrt_memcpy);
  ret = executor_rt_v2.Execute(inputs, args);
  EXPECT_NE(ret, SUCCESS);
  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 0);
  AclRuntimeStub::UnInstall(&mock_aclrt_memcpy);
  RuntimeStub::Reset();
  options["ge.inputBatchCpy"] = "0";
  ge::GetThreadLocalContext().SetSessionOption(options);
  ge::GetThreadLocalContext().SetGraphOption(options);
}

TEST_F(HybridModelAsyncTest, ExecuteWithStreamAsync_execute_with_preassigned_output) {
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

  std::vector<GeTensor> input_tensors;
  std::vector<GeTensor> output_tensors;

  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[512]);
  auto scalar_host_tensor = GeTensor(GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 4);
  scalar_host_tensor.MutableTensorDesc().SetOriginShape(GeShape());
  scalar_host_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

  unique_ptr<uint8_t[]> output_buf(new (std::nothrow) uint8_t[512]);
  auto output_tensor = GeTensor(GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT), data_buf.get(), 4);
  output_tensor.MutableTensorDesc().SetOriginShape(GeShape());
  output_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementDevice);

  input_tensors.resize(2U, scalar_host_tensor);
  output_tensors.push_back(output_tensor);
  ret = executor_rt_v2.ExecuteWithStreamAsync(input_tensors, output_tensors, stream);
  EXPECT_EQ(ret, SUCCESS);

  RuntimeStub::Reset();
}
}  // namespace ge

class Listener : public ModelListener {
  using Done = std::function<void(uint32_t, uint32_t, uint32_t, std::vector<gert::Tensor> &)>;

 public:
  explicit Listener(Done done) : done_(done) {}
  Status OnComputeDone(uint32_t model_id, uint32_t data_index, uint32_t result_code,
                       std::vector<gert::Tensor> &outputs) override {
    done_(model_id, data_index, result_code, outputs);
    return SUCCESS;
  }
  Done done_;
};
class StestHybridRt2Executor : public testing::Test {
 protected:
  void SetUp() {
    setenv("ENABLE_RUNTIME_V2", "1", 0);
    const std::vector<rtMemType_t> mem_type{RT_MEMORY_HBM, RT_MEMORY_P2P_DDR};
    (void) MemManager::Instance().Initialize(mem_type);
    RTS_STUB_SETUP();
  }
  void TearDown() {
    unsetenv("ENABLE_RUNTIME_V2");
    hybrid::NpuMemoryAllocator::Finalize();
    ge::MemManager::Instance().Finalize();
    RTS_STUB_TEARDOWN();
  }
};
TEST_F(StestHybridRt2Executor, run_success) {
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

  EXPECT_EQ(hybrid_model.Init(), SUCCESS);

  HybridModelAsyncExecutor executor(&hybrid_model);
  EXPECT_EQ(executor.Init(), SUCCESS);

  bool reached = false;
  std::shared_ptr<ModelListener> listener =
      std::make_shared<Listener>([&reached](uint32_t model_id, uint32_t data_index, uint32_t result_code,
                                            std::vector<gert::Tensor> &outputs) { reached = true; });

  executor.Start(listener);

  InputData inputs;
  inputs.blobs.emplace_back(ge::DataBuffer());
  inputs.shapes.emplace_back(std::vector<int64_t>{});
  OutputData outputs;
  auto wrapper = std::make_shared<RunArgs>();
  wrapper->input_tensor = std::move(InputData2GertTensors(inputs));
  executor.EnqueueData(wrapper);

  size_t kMaxWaitSeconds = 5U;
  for (size_t seconds_wait = 0U; seconds_wait < kMaxWaitSeconds; seconds_wait++) {
    if (reached) {
      break;
    }
    sleep(1);
  }
  executor.Stop();
}

namespace {
template <typename T>
struct GeType {};

template <>
struct GeType<float> {
  static ge::DataType type;
};
ge::DataType GeType<float >::type = ge::DT_FLOAT;

template <typename T>
GeTensorPtr MakeScalarTensor(const T &v) {
  auto desc = ge::GeTensorDesc(GeShape(), ge::FORMAT_ND, GeType<T>::type);
  return std::make_shared<GeTensor>(desc, reinterpret_cast<const uint8_t*>(&v), sizeof(T));
}
}

namespace gert {
namespace {
LowerResult LoweringFoo(const ge::NodePtr &node, const LowerInput &lower_input) {
  size_t output_size = 512U;
  gert::StorageShape shape;
  auto size_holder = bg::ValueHolder::CreateConst(&output_size, sizeof(output_size));
  auto output_addrs = bg::AllocOutputMemory(kOnDeviceHbm, node, {size_holder}, *(lower_input.global_data));
  auto compute_holder = bg::ValueHolder::CreateVoid<bg::ValueHolder>("LaunchFooAssignAdd", {lower_input.input_addrs[1], output_addrs[0], lower_input.global_data->GetStream()});

  return {HyperStatus::Success(), {compute_holder}, {lower_input.input_shapes[0]}, output_addrs};
}
REGISTER_NODE_CONVERTER("_lower_foo", LoweringFoo);

LowerResult LoweringFooWithStreamSync(const ge::NodePtr &node, const LowerInput &lower_input) {

  size_t output_size = 512U;
  gert::StorageShape shape;
  auto size_holder = bg::ValueHolder::CreateConst(&output_size, sizeof(output_size));
  auto output_addrs = bg::AllocOutputMemory(kOnDeviceHbm, node, {size_holder}, *(lower_input.global_data));
  auto compute_holder = bg::ValueHolder::CreateVoid<bg::ValueHolder>("SyncStream", {lower_input.global_data->GetStream()});

  return {HyperStatus::Success(), {compute_holder}, {lower_input.input_shapes[0]}, output_addrs};
}
REGISTER_NODE_CONVERTER("_lower_foo_with_stream_sync", LoweringFooWithStreamSync);

// This kernel is something like aicore AssignAdd
ge::graphStatus LaunchFooAssignAdd(KernelContext *context) {
  auto value = context->GetInputPointer<gert::GertTensorData>(0U);
  auto variable = context->GetInputPointer<gert::GertTensorData>(1U);
  GE_ASSERT_NOTNULL(value);
  GE_ASSERT_NOTNULL(value->GetAddr());
  GE_ASSERT_NOTNULL(variable);
  GE_ASSERT_NOTNULL(variable->GetAddr());

  *static_cast<float*>(variable->GetAddr()) += *static_cast<float*>(value->GetAddr());

  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(LaunchFooAssignAdd).RunFunc(LaunchFooAssignAdd);
}  // namespace
}  // namespace gert

TEST_F(StestHybridRt2Executor, run_graph_with_ref_variable_success) {
  auto graph = ShareGraph::SimpleVariableGraph("variable_another");

  GeTensorPtr weight = MakeScalarTensor(2.0f);
  ASSERT_NE(weight, nullptr);

  auto constant = graph->FindFirstNodeMatchType("Constant");
  auto variable = graph->FindFirstNodeMatchType("Variable");
  auto foo = graph->FindFirstNodeMatchType("Foo");
  ASSERT_NE(constant, nullptr);
  ASSERT_NE(variable, nullptr);
  ASSERT_NE(foo, nullptr);

  ge::TensorUtils::SetSize(*variable->GetOpDescBarePtr()->MutableOutputDesc(0), 64);
  ge::AttrUtils::SetStr(foo->GetOpDesc(), "_ge_attr_lowering_func", "_lower_foo");
  ge::AttrUtils::SetStr(foo->GetOpDesc()->MutableOutputDesc(0), ASSIGN_VAR_NAME, variable->GetName());

  auto constant_desc = constant->GetOpDesc()->MutableOutputDesc(0);
  auto variable_desc = variable->GetOpDesc()->MutableOutputDesc(0);

  constant_desc->SetShape(GeShape(std::vector<int64_t>{}));
  constant_desc->SetOriginShape(GeShape(std::vector<int64_t>{}));
  constant_desc->SetDataType(ge::DT_FLOAT);
  ge::TensorUtils::SetSize(*constant_desc, 4);

  variable_desc->SetShape(GeShape(std::vector<int64_t>{1}));
  variable_desc->SetOriginShape(GeShape(std::vector<int64_t>{1}));
  variable_desc->SetDataType(ge::DT_FLOAT);
  ge::TensorUtils::SetSize(*variable_desc, 4);

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

  EXPECT_EQ(hybrid_model.Init(), SUCCESS);

  HybridModelAsyncExecutor executor(&hybrid_model);

  auto var_manager = VarManager::Instance(hybrid_model.GetSessionId());
  ASSERT_EQ(var_manager->Init(0, hybrid_model.GetSessionId(), 0, 0), SUCCESS);
  
  ge::ScopeGuard guarder([&var_manager]() { var_manager->FreeVarMemory(); });

  EXPECT_EQ(executor.Init(), SUCCESS);

  InputData inputs;
  OutputData outputs;
  std::shared_ptr<RunArgs> wrapper;

  bool reached = false;
  float result = 0.0f;
  std::shared_ptr<ModelListener> listener = std::make_shared<Listener>(
      [&](uint32_t model_id, uint32_t data_index, uint32_t result_code, std::vector<gert::Tensor> &outputs) {
        if (!outputs.empty()) {
          result = *static_cast<float *>(outputs[0].GetAddr());
        }
        reached = true;
      });

  executor.Start(listener);
  for (size_t i = 0U; i < 2U; i++) {
    wrapper = std::make_shared<RunArgs>();
    wrapper->input_tensor = std::move(InputData2GertTensors(inputs));
    executor.EnqueueData(wrapper);

    size_t kMaxWaitSeconds = 5U;
    for (size_t seconds_wait = 0U; seconds_wait < kMaxWaitSeconds; seconds_wait++) {
      if (reached) {
        break;
      }
      sleep(1);
    }
    reached = false;
    // We run variable assign add 2.0 twice, expect result == 2.0f at first and then expect 4.0f at second time
    EXPECT_LT(abs(result - 2.0f * float(i + 1)), 1.0e-5);
  }
  executor.Stop();

  EXPECT_LT(abs(result - 4.0f), 1.0e-5);
}

TEST_F(StestHybridRt2Executor, run_graph_with_host_ref_variable_success) {
  auto graph = ShareGraph::SimpleVariableGraph("variable_host1");

  GeTensorPtr weight = MakeScalarTensor(2.0f);
  ASSERT_NE(weight, nullptr);

  auto constant = graph->FindFirstNodeMatchType("Constant");
  auto variable = graph->FindFirstNodeMatchType("Variable");
  auto foo = graph->FindFirstNodeMatchType("Foo");
  ASSERT_NE(constant, nullptr);
  ge::TensorUtils::SetSize(*(constant->GetOpDescBarePtr()->MutableOutputDesc(0)), weight->GetData().GetSize());
  ASSERT_NE(variable, nullptr);
  ASSERT_NE(foo, nullptr);


  ge::AttrUtils::SetStr(variable->GetOpDesc(), ge::ATTR_VARIABLE_PLACEMENT, "host");
  std::map<std::string, std::string> options;
  options["ge.exec.placement"] = "HOST";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ASSERT_TRUE(ge::GetContext().GetHostExecFlag());

  ge::AttrUtils::SetStr(foo->GetOpDesc(), "_ge_attr_lowering_func", "_lower_foo");
  ge::AttrUtils::SetStr(foo->GetOpDesc()->MutableOutputDesc(0), ASSIGN_VAR_NAME, variable->GetName());

  auto constant_desc = constant->GetOpDesc()->MutableOutputDesc(0);
  auto variable_desc = variable->GetOpDesc()->MutableOutputDesc(0);

  constant_desc->SetShape(GeShape(std::vector<int64_t>{}));
  constant_desc->SetOriginShape(GeShape(std::vector<int64_t>{}));
  constant_desc->SetDataType(ge::DT_FLOAT);
  variable_desc->SetShape(GeShape(std::vector<int64_t>{}));
  variable_desc->SetOriginShape(GeShape(std::vector<int64_t>{}));
  variable_desc->SetDataType(ge::DT_FLOAT);
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
  gert::GertRuntimeStub runtime_stub;
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);

  HybridModelAsyncExecutor executor(&hybrid_model);

  auto var_manager = VarManager::Instance(hybrid_model.GetSessionId());
  ASSERT_EQ(var_manager->Init(0, hybrid_model.GetSessionId(), 0, 0), SUCCESS);
  
  ge::ScopeGuard guarder([&var_manager]() { var_manager->FreeVarMemory(); });

  EXPECT_EQ(executor.Init(), SUCCESS);

  InputData inputs;
  OutputData outputs;
  std::shared_ptr<RunArgs> wrapper;

  bool reached = false;
  float result = 0.0f;
  std::shared_ptr<ModelListener> listener = std::make_shared<Listener>(
      [&](uint32_t model_id, uint32_t data_index, uint32_t result_code, std::vector<gert::Tensor> &outputs) {
        if (!outputs.empty()) {
          result = *static_cast<float *>(outputs[0].GetAddr());
        }
        reached = true;
      });

  executor.Start(listener);
  for (size_t i = 0U; i < 2U; i++) {
    wrapper = std::make_shared<RunArgs>();
    wrapper->input_tensor = std::move(InputData2GertTensors(inputs));
    executor.EnqueueData(wrapper);

    size_t kMaxWaitSeconds = 5U;
    for (size_t seconds_wait = 0U; seconds_wait < kMaxWaitSeconds; seconds_wait++) {
      if (reached) {
        break;
      }
      sleep(1);
    }
    reached = false;
    // We run variable assign add 2.0 twice, expect result == 2.0f at first and then expect 4.0f at second time
    EXPECT_LT(abs(result - 2.0f * float(i + 1)), 1.0e-5);
  }
  executor.Stop();

  EXPECT_LT(abs(result - 4.0f), 1.0e-5);
}

TEST_F(StestHybridRt2Executor, run_graph_with_host_shm_ref_variable_success) {
  std::string shm_var_name = "variable_host_shm";
  auto graph = ShareGraph::SimpleVariableGraph(shm_var_name);
  SharedMemInfo shm_info(shm_var_name, 64);
  ASSERT_EQ(HostMemManager::Instance().Initialize(), SUCCESS);
  ASSERT_EQ(HostMemManager::Instance().MallocHostSharedMemory(shm_info), SUCCESS);
  SharedMemInfo shm_info_malloced;
  ASSERT_TRUE(HostMemManager::Instance().QueryVarMemInfo(shm_var_name, shm_info_malloced));
  ASSERT_EQ(shm_info_malloced.op_name, shm_var_name);
  ASSERT_GE(shm_info_malloced.mem_size, 64);
  ASSERT_NE(shm_info_malloced.host_aligned_ptr->Get(), nullptr);
  memset(shm_info_malloced.host_aligned_ptr->MutableGet(), 0, shm_info_malloced.mem_size);
  float base = 1.0;
  *(reinterpret_cast<float *>(shm_info_malloced.host_aligned_ptr->MutableGet())) = base;

  GeTensorPtr weight = MakeScalarTensor(2.0f);
  ASSERT_NE(weight, nullptr);

  auto constant = graph->FindFirstNodeMatchType("Constant");
  auto variable = graph->FindFirstNodeMatchType("Variable");
  auto foo = graph->FindFirstNodeMatchType("Foo");
  ASSERT_NE(constant, nullptr);
  ASSERT_NE(variable, nullptr);
  ASSERT_NE(foo, nullptr);


  ge::AttrUtils::SetStr(variable->GetOpDesc(), ge::ATTR_VARIABLE_PLACEMENT, "host");
  std::map<std::string, std::string> options;
  options["ge.exec.placement"] = "HOST";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ASSERT_TRUE(ge::GetContext().GetHostExecFlag());

  ge::AttrUtils::SetStr(foo->GetOpDesc(), "_ge_attr_lowering_func", "_lower_foo");
  ge::AttrUtils::SetStr(foo->GetOpDesc()->MutableOutputDesc(0), ASSIGN_VAR_NAME, variable->GetName());

  auto constant_desc = constant->GetOpDesc()->MutableOutputDesc(0);
  auto variable_desc = variable->GetOpDesc()->MutableOutputDesc(0);

  constant_desc->SetShape(GeShape(std::vector<int64_t>{}));
  constant_desc->SetOriginShape(GeShape(std::vector<int64_t>{}));
  constant_desc->SetDataType(ge::DT_FLOAT);
  variable_desc->SetShape(GeShape(std::vector<int64_t>{}));
  variable_desc->SetOriginShape(GeShape(std::vector<int64_t>{}));
  variable_desc->SetDataType(ge::DT_FLOAT);

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

  gert::GertRuntimeStub runtime_stub;
  HybridModel hybrid_model(ge_root_model);

  EXPECT_EQ(hybrid_model.Init(), SUCCESS);

  HybridModelAsyncExecutor executor(&hybrid_model);

  auto var_manager = VarManager::Instance(hybrid_model.GetSessionId());
  ASSERT_EQ(var_manager->Init(0, hybrid_model.GetSessionId(), 0, 0), SUCCESS);
  ge::ScopeGuard guarder([&var_manager]() { var_manager->FreeVarMemory(); });

  EXPECT_EQ(executor.Init(), SUCCESS);

  InputData inputs;
  OutputData outputs;
  std::shared_ptr<RunArgs> wrapper;

  bool reached = false;
  float result = 0.0f;
  std::shared_ptr<ModelListener> listener = std::make_shared<Listener>(
      [&](uint32_t model_id, uint32_t data_index, uint32_t result_code, std::vector<gert::Tensor> &outputs) {
        if (!outputs.empty()) {
         result = *static_cast<float *>(outputs[0].GetAddr());
       }
        reached = true;
      });

  executor.Start(listener);
  for (size_t i = 0U; i < 2U; i++) {
    wrapper = std::make_shared<RunArgs>();
    wrapper->input_tensor = std::move(InputData2GertTensors(inputs));
    executor.EnqueueData(wrapper);

    size_t kMaxWaitSeconds = 5U;
    for (size_t seconds_wait = 0U; seconds_wait < kMaxWaitSeconds; seconds_wait++) {
      if (reached) {
        break;
      }
      sleep(1);
    }
    reached = false;
    // We run variable assign add 2.0 twice, expect result == 2.0f + base at first and then expect 4.0f + base at second time
    EXPECT_LT(abs(result - 2.0f * float(i + 1) - base), 1.0e-5);
  }
  executor.Stop();
  HostMemManager::Instance().Finalize();
}
/*
  测试若sync stream 返回END_OF_SEQUENCE，则终止取输入数据下发

  测试准备：
       1、构造只有一个foo节点的计算图，该节点lowering出一个SyncStream节点
       2、设置环境变量，令rts桩接口返回EOS状态码
  测试步骤：
      设置循环取数据迭代次数为2，执行该模型
  预期结果：
       iter num为1.
*/
TEST_F(StestHybridRt2Executor, run_graph_with_end_of_senquence) {
  const char_t *const kEnvRecordPath = "END_OF_SEQUENCE";
  char_t record_path[MMPA_MAX_PATH] = "end";
  mmSetEnv(kEnvRecordPath, &record_path[0U], MMPA_MAX_PATH);

  auto graph = ShareGraph::SimpleFooGraph();
  graph->TopologicalSorting();
  auto foo = graph->FindFirstNodeMatchType("Foo");
  ASSERT_NE(foo, nullptr);
  auto data = graph->FindFirstNodeMatchType("RefData");
  auto data_desc = data->GetOpDesc()->MutableOutputDesc(0);
  data_desc->SetShape(GeShape(std::vector<int64_t>{}));
  data_desc->SetOriginShape(GeShape(std::vector<int64_t>{}));
  data_desc->SetDataType(ge::DT_FLOAT);
  ge::AttrUtils::SetStr(foo->GetOpDesc(), "_ge_attr_lowering_func", "_lower_foo_with_stream_sync");

  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();
  HybridModel hybrid_model(ge_root_model);
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);

  HybridModelAsyncExecutor executor(&hybrid_model);
  EXPECT_EQ(executor.Init(), SUCCESS);

  InputData inputs;
  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[3072]);
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 3072, false));
  inputs.shapes.push_back({1, 16, 16, 3});
  OutputData outputs;
  std::shared_ptr<RunArgs> wrapper;

  bool reached = false;
  bool is_eos = false;
  float result = 0.0f;
  std::shared_ptr<ModelListener> listener = std::make_shared<Listener>(
      [&](uint32_t model_id, uint32_t data_index, uint32_t result_code, std::vector<gert::Tensor> &outputs) {
        if (result_code == ge::END_OF_SEQUENCE) {
          is_eos = true;
        }
        if (!outputs.empty()) {
         result = *static_cast<float *>(outputs[0].GetAddr());
       }
        reached = true;
      });

  executor.Start(listener);
  size_t iter_num = 0;
  for (size_t i = 0U; i < 2U; i++) {
    if (is_eos) {
      break;
    }
    wrapper = std::make_shared<RunArgs>();
    auto gert_inputs = InputData2GertTensors(inputs);
    gert_inputs[0].SetPlacement(TensorPlacement::kOnHost);
    wrapper->input_tensor = std::move(gert_inputs);
    executor.EnqueueData(wrapper);

    size_t kMaxWaitSeconds = 5U;
    for (size_t seconds_wait = 0U; seconds_wait < kMaxWaitSeconds; seconds_wait++) {
      if (reached) {
        break;
      }
      sleep(1);
    }
    reached = false;
    ++iter_num;
  }
  EXPECT_EQ(iter_num, 1);
  executor.Stop();
  unsetenv(kEnvRecordPath);
}

TEST_F(StestHybridRt2Executor, run_graph_with_iterations_loop_success) {
  ge::diagnoseSwitch::EnableDeviceProfiling();
  auto graph = ShareGraph::SimpleVariableGraph("variable_host2");
  graph->SetNeedIteration(true);

  GeTensorPtr weight = MakeScalarTensor(2.0f);
  ASSERT_NE(weight, nullptr);

  auto constant = graph->FindFirstNodeMatchType("Constant");
  auto variable = graph->FindFirstNodeMatchType("Variable");
  auto foo = graph->FindFirstNodeMatchType("Foo");
  ASSERT_NE(constant, nullptr);
  ASSERT_NE(variable, nullptr);
  ASSERT_NE(foo, nullptr);

  auto origin_option = GetThreadLocalContext().GetAllGraphOptions();
  ge::AttrUtils::SetStr(variable->GetOpDesc(), ge::ATTR_VARIABLE_PLACEMENT, "host");
  std::map<std::string, std::string> options;
  options["ge.exec.placement"] = "HOST";
  const size_t kPipeLoop = 10;
  options[ge::ATTR_NAME_ITERATORS_PER_LOOP] = std::to_string(kPipeLoop);
  ge::GetThreadLocalContext().SetGraphOption(options);
  ASSERT_TRUE(ge::GetContext().GetHostExecFlag());

  ge::AttrUtils::SetStr(foo->GetOpDesc(), "_ge_attr_lowering_func", "_lower_foo");
  ge::AttrUtils::SetStr(foo->GetOpDesc()->MutableOutputDesc(0), ASSIGN_VAR_NAME, variable->GetName());
  ge::AttrUtils::SetBool(foo->GetOpDesc(), ge::ATTR_NAME_INSERT_FP_PROFILILNG_TASK, true);
  ge::AttrUtils::SetInt(foo->GetOpDesc(), ge::ATTR_NAME_INSERT_PROFILILNG_TASK_LOG_ID, 20000);
  ge::AttrUtils::SetBool(foo->GetOpDesc(), ge::ATTR_NAME_INSERT_BP_PROFILILNG_TASK, true);

  auto constant_desc = constant->GetOpDesc()->MutableOutputDesc(0);
  auto variable_desc = variable->GetOpDesc()->MutableOutputDesc(0);

  constant_desc->SetShape(GeShape(std::vector<int64_t>{}));
  constant_desc->SetOriginShape(GeShape(std::vector<int64_t>{}));
  constant_desc->SetDataType(ge::DT_FLOAT);
  variable_desc->SetShape(GeShape(std::vector<int64_t>{}));
  variable_desc->SetOriginShape(GeShape(std::vector<int64_t>{}));
  variable_desc->SetDataType(ge::DT_FLOAT);
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

  gert::GertRuntimeStub runtime_stub;
  HybridModel hybrid_model(ge_root_model);

  EXPECT_EQ(hybrid_model.Init(), SUCCESS);

  HybridModelAsyncExecutor executor(&hybrid_model);

  auto var_manager = VarManager::Instance(hybrid_model.GetSessionId());
  ASSERT_EQ(var_manager->Init(0, hybrid_model.GetSessionId(), 0, 0), SUCCESS);
  
  ge::ScopeGuard guarder([&var_manager]() { var_manager->FreeVarMemory(); });

  EXPECT_EQ(executor.Init(), SUCCESS);

  std::string iterations_loop;
  ge::GetThreadLocalContext().GetOption(ge::ATTR_NAME_ITERATORS_PER_LOOP, iterations_loop);
  EXPECT_EQ(iterations_loop, std::to_string(kPipeLoop));

  InputData inputs;
  OutputData outputs;
  std::shared_ptr<RunArgs> wrapper;

  bool reached = false;
  float result = 0.0f;
  std::shared_ptr<ModelListener> listener = std::make_shared<Listener>(
      [&](uint32_t model_id, uint32_t data_index, uint32_t result_code, std::vector<gert::Tensor> &outputs) {
        if (!outputs.empty()) {
          result = *static_cast<float *>(outputs[0].GetAddr());
        }
        reached = true;
      });

  executor.Start(listener);
  for (size_t i = 0U; i < 2U; i++) {
    wrapper = std::make_shared<RunArgs>();
    wrapper->input_tensor = std::move(InputData2GertTensors(inputs));
    executor.EnqueueData(wrapper);

    size_t kMaxWaitSeconds = 5U;
    for (size_t seconds_wait = 0U; seconds_wait < kMaxWaitSeconds; seconds_wait++) {
      if (reached) {
        break;
      }
      sleep(1);
    }
    reached = false;
    // We run variable assign add 20.0 twice(each iteration loop 10 times and each time add 2.0), expect result == 20.0f
    // at first and then expect 40.0f at second time
    EXPECT_LT(abs(result - (2.0f * 10) * float(i + 1)), 1.0e-5);
  }
  executor.Stop();

  EXPECT_LT(abs(result - (2.0f * 10 * 2)), 1.0e-5);
  ge::GetThreadLocalContext().SetGraphOption(origin_option);
  ge::diagnoseSwitch::DisableProfiling();
}

TEST_F(StestHybridRt2Executor, test_hybrid_v2_execute_with_rtStreamSync_timeout) {
  const char_t *const kTimeoutEnvPath = "TIMEOUT";
  char_t record_path[MMPA_MAX_PATH] = "timeout";
  mmSetEnv(kTimeoutEnvPath, &record_path[0U], MMPA_MAX_PATH);
  ge::GetContext().SetStreamSyncTimeout(15000);

  auto graph = ShareGraph::SimpleVariableGraph("variable_host");

  GeTensorPtr weight = MakeScalarTensor(2.0f);
  ASSERT_NE(weight, nullptr);

  auto constant = graph->FindFirstNodeMatchType("Constant");
  auto variable = graph->FindFirstNodeMatchType("Variable");
  auto foo = graph->FindFirstNodeMatchType("Foo");
  ASSERT_NE(constant, nullptr);
  ASSERT_NE(variable, nullptr);
  ASSERT_NE(foo, nullptr);


  ge::AttrUtils::SetStr(variable->GetOpDesc(), ge::ATTR_VARIABLE_PLACEMENT, "host");
  std::map<std::string, std::string> options;
  options["ge.exec.placement"] = "HOST";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ASSERT_TRUE(ge::GetContext().GetHostExecFlag());

  ge::AttrUtils::SetStr(foo->GetOpDesc(), "_ge_attr_lowering_func", "_lower_foo");
  ge::AttrUtils::SetStr(foo->GetOpDesc()->MutableOutputDesc(0), ASSIGN_VAR_NAME, variable->GetName());

  auto constant_desc = constant->GetOpDesc()->MutableOutputDesc(0);
  auto variable_desc = variable->GetOpDesc()->MutableOutputDesc(0);

  constant_desc->SetShape(GeShape(std::vector<int64_t>{}));
  constant_desc->SetOriginShape(GeShape(std::vector<int64_t>{}));
  constant_desc->SetDataType(ge::DT_FLOAT);
  variable_desc->SetShape(GeShape(std::vector<int64_t>{}));
  variable_desc->SetOriginShape(GeShape(std::vector<int64_t>{}));
  variable_desc->SetDataType(ge::DT_FLOAT);

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

  gert::GertRuntimeStub runtime_stub;
  HybridModel hybrid_model(ge_root_model);

  EXPECT_EQ(hybrid_model.Init(), SUCCESS);

  HybridModelAsyncExecutor executor(&hybrid_model);

  auto var_manager = VarManager::Instance(hybrid_model.GetSessionId());
  ASSERT_EQ(var_manager->Init(0, hybrid_model.GetSessionId(), 0, 0), SUCCESS);
  
  ge::ScopeGuard guarder([&var_manager]() { var_manager->FreeVarMemory(); });

  EXPECT_EQ(executor.Init(), SUCCESS);

  InputData inputs;
  OutputData outputs;
  std::shared_ptr<RunArgs> wrapper;

  bool reached = false;
  float result = 0.0f;
  std::shared_ptr<ModelListener> listener = std::make_shared<Listener>(
      [&](uint32_t model_id, uint32_t data_index, uint32_t result_code, std::vector<gert::Tensor> &outputs) {
        if (!outputs.empty()) {
         result = *static_cast<float *>(outputs[0].GetAddr());
       }
        reached = true;
        ge::GetContext().SetStreamSyncTimeout(15000);
      });

  executor.Start(listener);
  for (size_t i = 0U; i < 2U; i++) {
    wrapper = std::make_shared<RunArgs>();
    wrapper->input_tensor = std::move(InputData2GertTensors(inputs));
    executor.EnqueueData(wrapper);

    size_t kMaxWaitSeconds = 5U;
    for (size_t seconds_wait = 0U; seconds_wait < kMaxWaitSeconds; seconds_wait++) {
      if (reached) {
        break;
      }
      sleep(1);
    }
    reached = false;
    // We run variable assign add 2.0 twice,
    // but because rtaStreamSynchronize timeout, expect result = 1.0e-5 each time
    EXPECT_LT(abs(result), 1.0e-5);
  }
  executor.Stop();

  EXPECT_LT(abs(result), 1.0e-5);

  unsetenv(kTimeoutEnvPath);
}

TEST_F(StestHybridRt2Executor, run_graph_with_ref_fileconstant_success) {
  std::map<std::string, std::string> options;
  options["ge.exec.placement"] = "DEVICE";
  ge::GetThreadLocalContext().SetGraphOption(options);
  const std::string location = "xj_test_file_constant.bin";
  const ge::float32_t scalar = 2.0f;
  const std::string var_name = "variable_100";
  const std::string file_constant_name = "file_constant_100";

  ASSERT_EQ(CreateFileConstantFile(location, scalar), ge::SUCCESS);
  auto graph = ShareGraph::SimpleFileConstantGraph(var_name, file_constant_name, location);
  ASSERT_NE(graph, nullptr);
  auto file_constant = graph->FindFirstNodeMatchType("FileConstant");
  auto variable = graph->FindFirstNodeMatchType("Variable");
  auto foo = graph->FindFirstNodeMatchType("Foo");
  ASSERT_NE(file_constant, nullptr);
  ASSERT_NE(variable, nullptr);
  ASSERT_NE(foo, nullptr);

  ge::TensorUtils::SetSize(*variable->GetOpDescBarePtr()->MutableOutputDesc(0), 64);
  ge::AttrUtils::SetStr(foo->GetOpDesc(), "_ge_attr_lowering_func", "_lower_foo");
  ge::AttrUtils::SetStr(foo->GetOpDesc()->MutableOutputDesc(0), ASSIGN_VAR_NAME, variable->GetName());

  auto file_constant_desc = file_constant->GetOpDesc()->MutableOutputDesc(0);
  auto variable_desc = variable->GetOpDesc()->MutableOutputDesc(0);
  variable->GetOpDesc()->SetOutputOffset({137438953472U});

  file_constant_desc->SetShape(GeShape(std::vector<int64_t>{}));
  file_constant_desc->SetOriginShape(GeShape(std::vector<int64_t>{}));
  file_constant_desc->SetDataType(ge::DT_FLOAT);
  int64_t aligned_mem_size = 0U;
  ge::TensorUtils::GetTensorMemorySizeInBytes(file_constant->GetOpDesc()->GetOutputDesc(0U), aligned_mem_size);
  ge::TensorUtils::SetSize(*file_constant->GetOpDesc()->MutableOutputDesc(0U), aligned_mem_size);

  variable_desc->SetShape(GeShape(std::vector<int64_t>{}));
  variable_desc->SetOriginShape(GeShape(std::vector<int64_t>{}));
  variable_desc->SetDataType(ge::DT_FLOAT);


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

  EXPECT_EQ(hybrid_model.Init(), SUCCESS);

  HybridModelAsyncExecutor executor(&hybrid_model);

  VarManagerPool::Instance().Destory();
  auto var_manager = VarManager::Instance(hybrid_model.GetSessionId());
  auto weight_manager = ExternalWeightManagerPool::Instance().GetManager(hybrid_model.GetSessionId());
  ASSERT_EQ(var_manager->Init(0, hybrid_model.GetSessionId(), 0, 0), SUCCESS);
  
  ge::ScopeGuard guarder([&var_manager, &weight_manager]() {
    var_manager->FreeVarMemory();
    weight_manager->Finalize();
  });

  EXPECT_EQ(executor.Init(), SUCCESS);

  InputData inputs;
  OutputData outputs;
  std::shared_ptr<RunArgs> wrapper;

  bool reached = false;
  float result = 0.0f;
  std::shared_ptr<ModelListener> listener = std::make_shared<Listener>(
      [&](uint32_t model_id, uint32_t data_index, uint32_t result_code, std::vector<gert::Tensor> &outputs) {
        if (!outputs.empty()) {
         result = *static_cast<float *>(outputs[0].GetAddr());
       }
        reached = true;
      });

  executor.Start(listener);
  for (size_t i = 0U; i < 2U; i++) {
    wrapper = std::make_shared<RunArgs>();
    wrapper->input_tensor = std::move(InputData2GertTensors(inputs));
    executor.EnqueueData(wrapper);

    size_t kMaxWaitSeconds = 5U;
    for (size_t seconds_wait = 0U; seconds_wait < kMaxWaitSeconds; seconds_wait++) {
      if (reached) {
        break;
      }
      sleep(1);
    }
    reached = false;
    // We run variable assign add 2.0 twice, expect result == 2.0f at first and then expect 4.0f at second time
    EXPECT_LT(abs(result - 2.0f * float(i + 1)), 1.0e-5);
  }
  executor.Stop();

  EXPECT_LT(abs(result - 4.0f), 1.0e-5);
  system(std::string("rm -rf ").append(location).c_str());
}

/*
 * 用例场景：tf动态图执行场景，在HybridModelRtV2Executor会创建allocator（实际内部是CachdingMemAllocator实例）传给rt2.0执行器，
 * 这个allocator在SelectAllocator中会设置stream，并且当rtMalloc失败时触发流同步和内存回收，同步设置的流
 * 步骤：
 * step 1. 构造简单的动态图，并对对rtMalloc和rtStreamSynchronize打桩
 * 期望：由于执行过程中会多次调用到rtMalloc申请内存，而本用例期望当图内节点申请内存时失败一次，所以在rtMalloc桩函数中判断调用次数，只失败一次。
 * step 2. 执行图
 * 期望： 由于 rtStreamSynchronize 校验stream非空，否则会返回失败，执行图成功表示
 * step 3. 校验
 * 期望： 校验日志，确认触发了内存回收
 */
TEST_F(StestHybridRt2Executor, run_graph_with_recycle_memory_success) {
  auto graph = ShareGraph::AicoreGraphTwoAdd();
  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  ASSERT_NE(netoutput, nullptr);
  auto desc = netoutput->GetOpDesc()->MutableInputDesc(0);
  ASSERT_NE(desc, nullptr);
  desc->SetDataType(ge::DT_INT4);

  auto data = graph->FindFirstNodeMatchType("Data");
  data->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_INT4);
  data->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_INT4);

  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  HybridModelAsyncExecutor executor(&hybrid_model);
  rtStream_t stream = (void*)0x01;
  EXPECT_EQ(executor.Init(stream), SUCCESS);
  EXPECT_NE(executor.executor_, nullptr);
  unique_ptr<uint8_t[]> data_buf = std::make_unique<uint8_t[]>(3072);
  std::vector<GeTensor> input_tensors(2U);
  std::vector<GeTensor> output_tensors(1U);

  for (size_t i = 0U; i < input_tensors.size(); ++i) {
    input_tensors[i].MutableTensorDesc() = GeTensorDesc(GeShape({4, 16, 16, 3}), FORMAT_NCHW, DT_INT4);
    input_tensors[i].MutableTensorDesc().SetOriginShape(GeShape({4, 16, 16, 3}));
    input_tensors[i].MutableTensorDesc().SetPlacement(Placement::kPlacementDevice);
    input_tensors[i].SetData(data_buf.get(), 3072);
  }
  for (size_t i = 0U; i < output_tensors.size(); ++i) {
    output_tensors[i].SetData(nullptr, 0U);
  }

  auto sync_check = [](rtStream_t stm, int32_t timeout) -> int {
    if (stm != nullptr) {
      return 0;
    }
    std::cerr << " stream not correct! " << std::endl;
    return -1;
  };

  size_t malloc_cnt = 0U;
  auto malloc_failed = [&malloc_cnt](void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) -> int {
    ++malloc_cnt;
    if (malloc_cnt == 1) {
      *dev_ptr = nullptr;
      return -1;
    } else {
      *dev_ptr = new uint8_t[size];
      return 0;
    }
  };

  auto rts_stub = std::make_shared<MockStreamSync>();
  ge::RuntimeStub::SetInstance(rts_stub);
  EXPECT_CALL(*rts_stub, rtStreamSynchronizeWithTimeout).WillRepeatedly(sync_check);
  EXPECT_CALL(*rts_stub, rtMalloc).WillRepeatedly(malloc_failed);

  auto slog_sub = std::make_shared<SlogStubImpl>();
  ge::SlogStub::SetInstance(slog_sub);
  slog_sub->Clear();
  dlog_setlevel(GE_MODULE_NAME, 2, 0);

  ASSERT_EQ(executor.ExecuteWithStreamAsync(input_tensors, output_tensors, stream), SUCCESS);
  // check outputsize for int4, size is (4 * 16 * 16 * 3) * 4 / 8 = 1536
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    EXPECT_NE(output_tensors[i].GetData().GetData(), nullptr);
    EXPECT_EQ(output_tensors[i].GetData().GetSize(), 1536);
  }
  EXPECT_TRUE(slog_sub->FindWarnLogEndsWith("Failed to apply for memory. We will try to free memory from memory pool, the above warning log can be ignored. Try to free cached memory...") >= 0);
  ge::RuntimeStub::Reset();
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
}
TEST_F(StestHybridRt2Executor, Test_multiStream_execute_by_runGraph_with_rtv2) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamTwoNodeGraph(stream_num, event_num);
  EXPECT_TRUE(stream_num > 1);
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Add" || node->GetType() == "Relu") {
      MockLessImportantNodeKernel(node);
    }
  }

  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin"))
      .AddTaskDef("Relu", AiCoreTaskDefFaker("ReluStubBin"))
      .SetRootModelStreamNum(stream_num)
      .SetRootModelEventNum(event_num)
      .BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();
  std::vector<GeTensor> inputs;
  const std::vector<uint8_t> tensor_data{1, 212, 32, 32};
  GeTensorDescPtr tensor_desc = make_shared<GeTensorDesc>(GeShape({-1, 16, 16, 3}));
  tensor_desc->SetShapeRange({{1, 256}, {16, 16}, {16, 16}, {3, 3}});
  auto ge_tensor = GeTensor(*tensor_desc, tensor_data.data(), sizeof(uint8_t) * tensor_data.size());
  ge_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementDevice);
  inputs.push_back(ge_tensor);
  inputs.push_back(ge_tensor);
  {
    HybridModelAsyncExecutor executor(&hybrid_model);
    rtStream_t stream = (void *)0x01;
    EXPECT_EQ(executor.Init(stream), SUCCESS);
    EXPECT_NE(executor.executor_, nullptr);

    std::vector<gert::Tensor> gert_inputs;
    TensorTransUtils::GeTensors2GertTensors(inputs, gert_inputs);
    std::vector<gert::Tensor> gert_outputs;
    std::vector<GeTensor> outputs;
    ASSERT_EQ(executor.Execute(gert_inputs, gert_outputs), SUCCESS);
    auto all_rt_streams = runtime_stub.GetRtsRuntimeStub().GetAllRtStreams();
    ASSERT_EQ(all_rt_streams.size(), 1);
  }
  {
    HybridModelAsyncExecutor executor(&hybrid_model);
    rtStream_t stream = (void *)0x01;
    EXPECT_EQ(executor.Init(stream), SUCCESS);
    EXPECT_NE(executor.executor_, nullptr);

    std::vector<GeTensor> outputs;
    ASSERT_EQ(executor.ExecuteWithStreamAsync(inputs, outputs, stream), SUCCESS);
  }
  unsetenv("ENABLE_RUNTIME_V2");
  runtime_stub.Clear();
}

TEST_F(StestHybridRt2Executor, Test_multiStream_execute_by_runGraph_with_rtv2_rollback_singleStream) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  setenv("MOCK_AVAIL_STREAM_NUM", "1", 0); // only has 1 stream
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamTwoNodeGraph(stream_num, event_num);
  EXPECT_TRUE(stream_num > 1);
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Add" || node->GetType() == "Relu") {
      MockLessImportantNodeKernel(node);
    }
  }

  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin"))
      .AddTaskDef("Relu", AiCoreTaskDefFaker("ReluStubBin"))
      .SetRootModelStreamNum(stream_num)
      .SetRootModelEventNum(event_num)
      .BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();
  {
    HybridModelAsyncExecutor executor(&hybrid_model);
    rtStream_t stream = (void *)0x01;
    EXPECT_EQ(executor.Init(stream), SUCCESS);
    EXPECT_NE(executor.executor_, nullptr);

    std::vector<GeTensor> inputs;
    const std::vector<uint8_t> tensor_data{1, 212, 32, 32};
    GeTensorDescPtr tensor_desc = make_shared<GeTensorDesc>(GeShape({-1, 16, 16, 3}));
    tensor_desc->SetShapeRange({{1, 256}, {16, 16}, {16, 16}, {3, 3}});
    auto ge_tensor = GeTensor(*tensor_desc, tensor_data.data(), sizeof(uint8_t) * tensor_data.size());
    ge_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementDevice);
    inputs.push_back(ge_tensor);
    inputs.push_back(ge_tensor);
    std::vector<gert::Tensor> gert_inputs;
    TensorTransUtils::GeTensors2GertTensors(inputs, gert_inputs);
    std::vector<gert::Tensor> gert_outputs;
    std::vector<GeTensor> outputs;
    ASSERT_EQ(executor.Execute(gert_inputs, gert_outputs), SUCCESS);
    auto all_rt_streams = runtime_stub.GetRtsRuntimeStub().GetAllRtStreams();
    ASSERT_EQ(all_rt_streams.size(), 0); // execute on 1 streams, use external stream, no need create streams

    EXPECT_EQ(executor.Init(stream), SUCCESS);
    ASSERT_EQ(executor.ExecuteWithStreamAsync(inputs, outputs, stream), SUCCESS);
  }
  unsetenv("ENABLE_RUNTIME_V2");
  unsetenv("MOCK_AVAIL_STREAM_NUM");
  runtime_stub.Clear();
}

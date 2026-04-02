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
#include "ge_graph_dsl/graph_dsl.h"
#include "es_ge_test_ops.h"
#include "graph/utils/graph_utils_ex.h"
#include "jit_execution/jit_executor.h"
#include <vector>
#include "jit_share_graph.h"
#include "common/model/external_allocator_manager.h"
#include "ge/ge_api.h"
#include "stub/gert_runtime_stub.h"
#include "stub/runtime_stub_impl.h"
#include "ge_running_env/dir_env.h"
#include "common_setup.h"
#include "jit_execution/utils/partitioner/binary_partitioner.h"
#include "graph/execute/model_executor.h"
#include "utils/taskdef_builder.h"
#include "api/aclgrph/option_utils.h"
#include "common/mem_conflict_share_graph.h"
#include "common/memory/tensor_trans_utils.h"
#include "error_codes/rt_error_codes.h"

using namespace testing;
using namespace ge;

class RuntimeMock : public gert::RuntimeStubImpl {
public:
  rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen) override {
    (void)label;
    (void)key;
    (void)strcpy_s(val, maxLen, "fake"); // 用例不应该走自动融合
    return RT_ERROR_NONE;
  }
};

bool test_callback_called = false;
class JitExecutorUT : public testing::Test {
 protected:
  void SetUp() override {
    const auto env_ptr = getenv("LD_PRELOAD");
    if (env_ptr != nullptr) {
      env = env_ptr;
      unsetenv("LD_PRELOAD");
    }

    CommonSetupUtil::CommonSetup();
    auto &rts_stub = gert_stub_.GetAclRuntimeStub();
    gert_stub_.GetKernelStub().StubTiling();
    AclRuntimeStub::Install(&rts_stub);
    gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  }
  void TearDown() override {
    CommonSetupUtil::CommonTearDown();
    auto &rts_stub = gert_stub_.GetAclRuntimeStub();
    rts_stub.Clear();
    AclRuntimeStub::UnInstall(&rts_stub);
    if (!env.empty()) {
      setenv("LD_PRELOAD", env.c_str(), 1);
    }
  }
  gert::GertRuntimeStub gert_stub_;
  std::string env;
};

TEST_F(JitExecutorUT, CreateJitExecutor_Success) {
  auto &rts_stub = gert_stub_.GetAclRuntimeStub();
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphExecutionQueue task_queue;
  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  ExecutionOrder order({user_graph_id, compute_graph});
  CompileContext compile_context(graph_manager);
  CompiledModelCache cmc(user_graph_id, compile_context, graph_manager);
  std::mutex tmp_mutex;

  auto jit_executor = JitExecutor::Create(graph_manager, task_queue, order, compile_context, cmc, tmp_mutex);
  EXPECT_NE(jit_executor, nullptr);
  // 校验本次创建jit executor申请了1个stream，注册了1个device allocator
  EXPECT_EQ(rts_stub.GetAllRtStreams().size(), 1); // required 1 stream
  auto stream = rts_stub.GetAllRtStreams().at(0);
  auto allocator = ExternalAllocatorManager::GetExternalAllocator(stream);
  EXPECT_NE(allocator, nullptr); // required 1 allocator

  // 校验卸载资源成功
  EXPECT_EQ(jit_executor->Finalize(), SUCCESS);
  EXPECT_EQ(rts_stub.GetAllRtStreams().size(), 1);
  auto stream_after = rts_stub.GetAllRtStreams().at(0);
  EXPECT_EQ(stream_after, nullptr);
  auto allocator_after = ExternalAllocatorManager::GetExternalAllocator(stream);
  EXPECT_EQ(allocator_after, nullptr);

  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

// to stub rts return fail when create stream
TEST_F(JitExecutorUT, CreateJitExecutor_Failed) {
  class MockBrokenRTS : public gert::AclRuntimeStubImpl {
    aclError aclrtCreateStream(aclrtStream *stream) {
      return -1;
    }
  };
  auto mock_runtime = std::make_shared<MockBrokenRTS>();
  AclRuntimeStub::Install(mock_runtime.get());

  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphExecutionQueue task_queue;
  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  ExecutionOrder order({user_graph_id, compute_graph});
  CompileContext compile_context(graph_manager);
  CompiledModelCache cmc(user_graph_id, compile_context, graph_manager);
  std::mutex tmp_mutex;

  auto jit_executor = JitExecutor::Create(graph_manager, task_queue, order, compile_context, cmc, tmp_mutex);
  EXPECT_EQ(jit_executor, nullptr);

  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
  mock_runtime->Clear();
}

/*
 * 构造一个最简单的图，该图无切分机会
 *       data
 *        |
 *       relu
 *        |
 *       relu1
 *        |
 *    netoutput
 */
TEST_F(JitExecutorUT, Run_DynamicShape_NoSlice_Success) {
  // 1 准备构造jit executor的入参
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphExecutionQueue task_queue;
  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  ExecutionOrder order({user_graph_id, compute_graph});
  CompileContext compile_context(graph_manager);
  CompiledModelCache cmc(user_graph_id, compile_context, graph_manager);
  std::mutex tmp_mutex;

  // 3 构造jit executor
  auto jit_executor = JitExecutor::Create(graph_manager, task_queue, order, compile_context, cmc, tmp_mutex);
  EXPECT_NE(jit_executor, nullptr);
  // 校验本次创建jit executor申请了1个stream，注册了1个device allocator
  auto &rts_stub = gert_stub_.GetAclRuntimeStub();
  EXPECT_EQ(rts_stub.GetAllRtStreams().size(), 1); // required 1 stream
  auto stream = rts_stub.GetAllRtStreams().at(0);
  auto allocator = ExternalAllocatorManager::GetExternalAllocator(stream);
  EXPECT_NE(allocator, nullptr); // required 1 allocator

  // 4 准备执行接口的入参并触发执行
  std::vector<int64_t> shape_dim = {2, 3, 3, 2};
  TensorDesc td(Shape(shape_dim), FORMAT_NCHW, DT_FLOAT);
  td.SetOriginShape(Shape(shape_dim)); // todo check tfa set origin shape?
  Tensor tensor(td);
  std::vector<Tensor> inputs{tensor};
  std::vector<Tensor> outputs;
  const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(outputs.size(), 1);
    auto cur_dims = TensorTransUtils::GetDimsFromGertShape(outputs[0].GetStorageShape());
    EXPECT_EQ(cur_dims, shape_dim);
    return SUCCESS;
  };
  std::vector<gert::Tensor> gert_inputs;
  TensorTransUtils::Tensors2GertTensors(inputs, gert_inputs);
  UserGraphExecution task(user_graph_id, gert_inputs, callback, 0);
  EXPECT_EQ(jit_executor->RunWithCallback(std::move(task)), SUCCESS);

  // 5 清理本用例相关资源
  EXPECT_EQ(jit_executor->Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

TEST_F(JitExecutorUT, RunGraphAsyncAutoFuse) {
  EXPECT_EQ(GEInitialize(map<AscendString, AscendString>{}), SUCCESS);
  std::map<AscendString, AscendString> options;
  options["ge.inputBatchCpy"] = "1";
  const auto session_ptr = new Session(options);
  GraphId graph_id = 1;
  auto graph = JitShareGraph::OneAddNode();
  EXPECT_EQ(session_ptr->AddGraph(graph_id, *graph.get()), SUCCESS);

  std::vector<ge::Tensor> outputs;

  // invalid graph id
  // RunGraphAsync submit failed
  test_callback_called = false;
  auto callback = [](Status status, std::vector<ge::Tensor> &outputs) {
    EXPECT_EQ(status, SUCCESS);
    test_callback_called = true;
  };

  std::vector<int64_t> shape_dim = {2, 3, 3, 2};
  TensorDesc td(Shape(shape_dim), FORMAT_NCHW, DT_FLOAT);
  td.SetOriginShape(Shape(shape_dim));
  Tensor tensor(td);
  std::vector<int64_t> input_data_1(36, 0);
  tensor.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), 36 * sizeof(int64_t));
  std::vector<Tensor> inputs{tensor, tensor};

  // get graph_node fail
  EXPECT_NE(session_ptr->RunGraphAsync(10, inputs, nullptr), SUCCESS);
  // after RunGraphAsync run failed before, RunGraphAsync submit success
  RTS_STUB_RETURN_VALUE(rtsMemcpyBatch, rtError_t, RT_ERROR_NONE);
  EXPECT_EQ(session_ptr->RunGraphAsync(graph_id, inputs, callback), SUCCESS);
  size_t sleep_time_max = 5U;
  size_t sleep_time = 0U;
  while (!test_callback_called) {
    sleep(1);  // wait callback
    if (++sleep_time >= sleep_time_max) {
      break;
    }
  }
  EXPECT_EQ(test_callback_called, true);
  RuntimeStub::UnInstall(nullptr);
  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 2);
  RuntimeStub::GetInstance()->input_mem_copy_batch_count_ = 0;
  auto &rts_stub = gert_stub_.GetRtsRuntimeStub();
  RuntimeStub::Install(&rts_stub);
  delete session_ptr;
  EXPECT_EQ(GEFinalize(), SUCCESS);
  RTS_STUB_SETUP();
}

TEST_F(JitExecutorUT, RunGraphAsyncAutoFuseFallback) {
  EXPECT_EQ(GEInitialize(map<AscendString, AscendString>{}), SUCCESS);
  std::map<string, string> options;
  options["ge.inputBatchCpy"] = "1";
  OptionSetter batch_cpy_option(options);
  const auto session_ptr = new Session(map<AscendString, AscendString>{});
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  EXPECT_EQ(session_ptr->AddGraph(graph_id, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph)), SUCCESS);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;

  // invalid graph id
  // RunGraphAsync submit failed
  test_callback_called = false;
  auto callback = [](Status status, std::vector<ge::Tensor> &outputs) {
    EXPECT_NE(status, SUCCESS);
    test_callback_called = true;
  };

  std::vector<int64_t> input_data_1(2 * 2, 0);
  TensorDesc desc_1(Shape({2, 2}));
  desc_1.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int64_t));
  inputs.emplace_back(input_tensor_1);

  std::vector<int64_t> input_data_2(2 * 2, 0);
  TensorDesc desc_2(Shape({2, 2}));
  desc_2.SetPlacement(Placement::kPlacementDevice);
  ge::Tensor input_tensor_2{desc_2};
  input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int64_t));
  inputs.emplace_back(input_tensor_2);

  // get graph_node fail
  EXPECT_NE(session_ptr->RunGraphAsync(10, inputs, nullptr), SUCCESS);
  sleep(1);  // wait callback
  // after RunGraphAsync run failed before, RunGraphAsync submit success
  RTS_STUB_RETURN_VALUE(rtsMemcpyBatch, rtError_t, ACL_ERROR_RT_FEATURE_NOT_SUPPORT);
  EXPECT_EQ(session_ptr->RunGraphAsync(graph_id, inputs, callback), SUCCESS);
  sleep(1);  // wait callback
  EXPECT_EQ(test_callback_called, true);
  RuntimeStub::UnInstall(nullptr);
  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 0);
  RuntimeStub::GetInstance()->input_mem_copy_batch_count_ = 0;
  auto &rts_stub = gert_stub_.GetRtsRuntimeStub();
  RuntimeStub::Install(&rts_stub);
  delete session_ptr;
  EXPECT_EQ(GEFinalize(), SUCCESS);
  RTS_STUB_SETUP();
}

/*
 * 构造一个最简单的图，该图无切分机会
 *       data
 *        |
 *       relu
 *        |
 *       relu1
 *        |
 *    netoutput
 */
TEST_F(JitExecutorUT, Run_StaticShape_NoSlice_Success) {
  // 1 准备构造jit executor的入参
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphExecutionQueue task_queue;
  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodesStaticShape();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  
  ExecutionOrder order({user_graph_id, compute_graph});
  CompileContext compile_context(graph_manager);
  CompiledModelCache cmc(user_graph_id, compile_context, graph_manager);
  std::mutex tmp_mutex;

  // 3 构造jit executor
  auto jit_executor = JitExecutor::Create(graph_manager, task_queue, order, compile_context, cmc, tmp_mutex);
  EXPECT_NE(jit_executor, nullptr);
  // 校验本次创建jit executor申请了1个stream，注册了1个device allocator
  auto &rts_stub = gert_stub_.GetAclRuntimeStub();
  EXPECT_EQ(rts_stub.GetAllRtStreams().size(), 1); // required 1 stream
  auto stream = rts_stub.GetAllRtStreams().at(0);
  auto allocator = ExternalAllocatorManager::GetExternalAllocator(stream);
  EXPECT_NE(allocator, nullptr); // required 1 allocator

  // 4 准备执行接口的入参并触发执行
  std::vector<int64_t> shape_dim = {2, 3, 3, 2};
  TensorDesc td(Shape(shape_dim), FORMAT_NCHW, DT_FLOAT);
  td.SetOriginShape(Shape(shape_dim)); // todo check tfa set origin shape?
  Tensor tensor(td);
  std::vector<Tensor> inputs{tensor};
  std::vector<Tensor> outputs;
  const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(outputs.size(), 1);
    auto cur_dims = TensorTransUtils::GetDimsFromGertShape(outputs[0].GetStorageShape());
    EXPECT_EQ(cur_dims, shape_dim);
    return SUCCESS;
  };

  std::vector<gert::Tensor> gert_inputs;
  TensorTransUtils::Tensors2GertTensors(inputs, gert_inputs);
  UserGraphExecution task(user_graph_id, gert_inputs, callback, 0);
  EXPECT_EQ(jit_executor->RunWithCallback(std::move(task)), SUCCESS);

  // 5 清理本用例相关资源
  EXPECT_EQ(jit_executor->Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

/**
 *      data0
 *        |
 *       relu data1
 *        |     |
 *        reshape
 *        |
 *       relu1
 *        |
 *    netoutput
 **/
TEST_F(JitExecutorUT, run_success_when_input_graph_contain_one_reshape_node) {
  // 1 准备构造jit executor的入参
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphExecutionQueue task_queue;
  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::OneReshapeNode();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  ExecutionOrder order({user_graph_id, compute_graph});
  CompileContext compile_context(graph_manager);
  CompiledModelCache cmc(user_graph_id, compile_context, graph_manager);
  std::mutex tmp_mutex;

  // 2 构造jit executor
  auto jit_executor = JitExecutor::Create(graph_manager, task_queue, order, compile_context, cmc, tmp_mutex);
  EXPECT_NE(jit_executor, nullptr);
  // 校验本次创建jit executor申请了1个stream，注册了1个device allocator
  auto &rts_stub = gert_stub_.GetAclRuntimeStub();
  EXPECT_EQ(rts_stub.GetAllRtStreams().size(), 1); // required 1 stream
  auto stream = rts_stub.GetAllRtStreams().at(0);
  auto allocator = ExternalAllocatorManager::GetExternalAllocator(stream);
  EXPECT_NE(allocator, nullptr); // required 1 allocator

  // 3 准备执行接口的入参并触发执行
  std::vector<int64_t> shape_dim = {2, 3, 3, 2};
  TensorDesc td(Shape(shape_dim), FORMAT_NCHW, DT_FLOAT);
  td.SetOriginShape(Shape(shape_dim));
  Tensor tensor(td);
  std::vector<int64_t> input_data_2{2, 3, 3, 2};
  TensorDesc desc_2(Shape({4}), FORMAT_NCHW, DT_INT32);
  desc_2.SetOriginShape(Shape({4}));
  Tensor input_tensor_2{desc_2};
  input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int64_t));
  std::vector<Tensor> inputs{tensor, input_tensor_2};
  std::vector<Tensor> outputs;
  const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(outputs.size(), 1);
    auto cur_dims = TensorTransUtils::GetDimsFromGertShape(outputs[0].GetStorageShape());
    EXPECT_EQ(cur_dims, shape_dim);
    return SUCCESS;
  };

  std::vector<gert::Tensor> gert_inputs;
  TensorTransUtils::Tensors2GertTensors(inputs, gert_inputs);
  UserGraphExecution task(user_graph_id, gert_inputs, callback, 0);
  EXPECT_EQ(jit_executor->RunWithCallback(std::move(task)), SUCCESS);

  // 4 清理本用例相关资源
  EXPECT_EQ(jit_executor->Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

TEST_F(JitExecutorUT, run_success_when_input_graph_contain_one_reshape_node_with_host_cpu) {
  auto fe_optimizer = MakeShared<FakeAiCoreEngineOptimizer>();
  auto host_cpu_ops_kernel_builder = MakeShared<FakeHostcpuOpsKernelBuilder>("DNN_VM_HOST_CPU");
  auto fe_ops_kernel_builder = MakeShared<FakeAiCoreOpsKernelBuilder>("AIcoreEngine");
  GeRunningEnvFaker()
      .Reset()
      .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("DNN_VM_RTS_OP_STORE"))
      .Install(FakeEngine("DNN_VM_HOST_CPU")
                   .KernelInfoStore("DNN_VM_HOST_CPU_OP_STORE")
                   .GraphOptimizer("fe", fe_optimizer)
                   .KernelBuilder(host_cpu_ops_kernel_builder))
      .Install(FakeEngine("AIcoreEngine")
                   .KernelInfoStore("AIcoreEngine")
                   .GraphOptimizer("fe", fe_optimizer)
                   .KernelBuilder(fe_ops_kernel_builder))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp("Relu")
                   .Inputs({"x"})
                   .Outputs({"y"})
                   .InfoStoreAndBuilder("AIcoreEngine")
                   .InferShape(SingleIOForwardInfer))
      .Install(FakeOp(RESHAPE)
                   .Inputs({"x", "shape"})
                   .Outputs({"y"})
                   .AttrsDef("axis", 0)
                   .AttrsDef("num_axes", -1)
                   .InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE")
                   .InferShape(ReshapeInferFun))
      .Install(FakeOp(ADD)
                   .Inputs({"x1", "x2"})
                   .Outputs({"y"})
                   .InfoStoreAndBuilder("DNN_VM_HOST_CPU")
                   .InferShape(SingleIOForwardInfer));

  // 1 准备构造jit executor的入参
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphExecutionQueue task_queue;
  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::OneReshapeNodeWithHostInput();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  ExecutionOrder order({user_graph_id, compute_graph});
  CompileContext compile_context(graph_manager);
  CompiledModelCache cmc(user_graph_id, compile_context, graph_manager);
  std::mutex tmp_mutex;

  auto add = compute_graph->FindFirstNodeMatchType(ADD);
  ASSERT_NE(add, nullptr);
  add->GetOpDesc()->SetOpKernelLibName(kEngineNameHostCpu);

  // 2 构造jit executor
  auto jit_executor = JitExecutor::Create(graph_manager, task_queue, order, compile_context, cmc, tmp_mutex);
  EXPECT_NE(jit_executor, nullptr);
  // 校验本次创建jit executor申请了1个stream，注册了1个device allocator
  auto &rts_stub = gert_stub_.GetAclRuntimeStub();
  EXPECT_EQ(rts_stub.GetAllRtStreams().size(), 1);  // required 1 stream
  auto stream = rts_stub.GetAllRtStreams().at(0);
  auto allocator = ExternalAllocatorManager::GetExternalAllocator(stream);
  EXPECT_NE(allocator, nullptr);  // required 1 allocator

  // 3 准备执行接口的入参并触发执行
  std::vector<int64_t> shape_dim = {2, 3, 3, 2};
  TensorDesc td(Shape(shape_dim), FORMAT_NCHW, DT_FLOAT);
  td.SetOriginShape(Shape(shape_dim));
  td.SetPlacement(Placement::kPlacementHost);
  Tensor tensor(td);
  std::vector<int64_t> input_data_1(36, 0);
  tensor.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), 36 * sizeof(int64_t));
  std::vector<int64_t> input_data_2{2, 3, 3, 2};
  TensorDesc desc_2(Shape({4}), FORMAT_NCHW, DT_INT64);
  desc_2.SetOriginShape(Shape({4}));
  desc_2.SetPlacement(Placement::kPlacementHost);
  Tensor input_tensor_2{desc_2};
  input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int64_t));
  std::vector<Tensor> inputs{tensor, tensor, input_tensor_2};
  std::vector<Tensor> outputs;
  const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
    ASSERT_EQ(status, SUCCESS);
    ASSERT_EQ(outputs.size(), 1);
    auto cur_dims = TensorTransUtils::GetDimsFromGertShape(outputs[0].GetStorageShape());
    EXPECT_EQ(cur_dims, shape_dim);
  };

  std::vector<gert::Tensor> gert_inputs;
  TensorTransUtils::Tensors2GertTensors(inputs, gert_inputs);
  UserGraphExecution task(user_graph_id, gert_inputs, callback, 0);
  EXPECT_EQ(jit_executor->RunWithCallback(std::move(task)), SUCCESS);

  // 4 清理本用例相关资源
  EXPECT_EQ(jit_executor->Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

TEST_F(JitExecutorUT, run_success_when_input_graph_contain_one_reshape_two_relu_node) {
  GTEST_SKIP() << "GE修改桩存导致用例走到自动融合，下一个pr修复";
  // 1 准备构造jit executor的入参
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphExecutionQueue task_queue;
  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::OneReshapeNodeTwoRelu();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  ExecutionOrder order({user_graph_id, compute_graph});
  CompileContext compile_context(graph_manager);
  CompiledModelCache cmc(user_graph_id, compile_context, graph_manager);
  std::mutex tmp_mutex;

  // 2 构造jit executor
  auto jit_executor = JitExecutor::Create(graph_manager, task_queue, order, compile_context, cmc, tmp_mutex);
  EXPECT_NE(jit_executor, nullptr);
  // 校验本次创建jit executor申请了1个stream，注册了1个device allocator
  auto &rts_stub = gert_stub_.GetAclRuntimeStub();
  EXPECT_EQ(rts_stub.GetAllRtStreams().size(), 1); // required 1 stream
  auto stream = rts_stub.GetAllRtStreams().at(0);
  auto allocator = ExternalAllocatorManager::GetExternalAllocator(stream);
  EXPECT_NE(allocator, nullptr); // required 1 allocator

  // 3 准备执行接口的入参并触发执行
  std::vector<int64_t> shape_dim = {2, 3, 3, 2};
  TensorDesc td(Shape(shape_dim), FORMAT_NCHW, DT_FLOAT);
  td.SetOriginShape(Shape(shape_dim));
  Tensor tensor(td);
  std::vector<int64_t> input_data_1(36, 0);
  tensor.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), 36 * sizeof(int64_t));
  std::vector<int64_t> input_data_2{2, 3, 3, 2};
  TensorDesc desc_2(Shape({4}), FORMAT_NCHW, DT_INT32);
  desc_2.SetOriginShape(Shape({4}));
  Tensor input_tensor_2{desc_2};
  input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int64_t));
  std::vector<int64_t> data3_shape_dim = {2, 3, 3};
  TensorDesc td3(Shape(data3_shape_dim), FORMAT_NCHW, DT_FLOAT);
  td3.SetOriginShape(Shape(data3_shape_dim));
  Tensor tensor3(td3);
  std::vector<Tensor> inputs{tensor, input_tensor_2, tensor3};
  const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(outputs.size(), 2);
    auto cur_dims = TensorTransUtils::GetDimsFromGertShape(outputs[0].GetStorageShape());
    EXPECT_EQ(cur_dims, shape_dim);
    return SUCCESS;
  };

  std::vector<gert::Tensor> gert_inputs;
  TensorTransUtils::Tensors2GertTensors(inputs, gert_inputs);
  UserGraphExecution task(user_graph_id, gert_inputs, callback, 0);
  EXPECT_EQ(jit_executor->RunWithCallback(std::move(task)), SUCCESS);
  // 4 清理本用例相关资源
  EXPECT_EQ(jit_executor->Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

// 该用例放开需要解决：1、reshape算子原型中默认的attr定义；2、netout节点配置了ATTR_NAME_PARENT_NODE_INDEX属性
TEST_F(JitExecutorUT, run_success_when_input_graph_contain_two_reshape_node) {
  // 1 准备构造jit executor的入参
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphExecutionQueue task_queue;
  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::TwoReshapeNodeTwoRelu();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  ExecutionOrder order({user_graph_id, compute_graph});
  CompileContext compile_context(graph_manager);
  CompiledModelCache cmc(user_graph_id, compile_context, graph_manager);
  std::mutex tmp_mutex;

  // 2 构造jit executor
  auto jit_executor = JitExecutor::Create(graph_manager, task_queue, order, compile_context, cmc, tmp_mutex);
  EXPECT_NE(jit_executor, nullptr);
  // 校验本次创建jit executor申请了1个stream，注册了1个device allocator
  auto &rts_stub = gert_stub_.GetAclRuntimeStub();
  EXPECT_EQ(rts_stub.GetAllRtStreams().size(), 1); // required 1 stream
  auto stream = rts_stub.GetAllRtStreams().at(0);
  auto allocator = ExternalAllocatorManager::GetExternalAllocator(stream);
  EXPECT_NE(allocator, nullptr); // required 1 allocator

  // 3 准备执行接口的入参并触发执行
  std::vector<int64_t> shape_dim = {2, 3, 3, 2};
  TensorDesc td(Shape(shape_dim), FORMAT_NCHW, DT_FLOAT);
  td.SetOriginShape(Shape(shape_dim));
  Tensor tensor(td);
  std::vector<int64_t> input_data_2{2, 3, 3, 2};
  TensorDesc desc_2(Shape({4}), FORMAT_NCHW, DT_INT32);
  desc_2.SetOriginShape(Shape({4}));
  Tensor input_tensor_2{desc_2};
  input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int64_t));
  std::vector<Tensor> inputs{tensor, input_tensor_2};
  std::vector<Tensor> outputs;
  const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(outputs.size(), 1);
    auto cur_dims = TensorTransUtils::GetDimsFromGertShape(outputs[0].GetStorageShape());
    EXPECT_EQ(cur_dims, shape_dim);
    return SUCCESS;
  };

  std::vector<gert::Tensor> gert_inputs;
  TensorTransUtils::Tensors2GertTensors(inputs, gert_inputs);
  UserGraphExecution task(user_graph_id, gert_inputs, callback, 0);
  EXPECT_EQ(jit_executor->RunWithCallback(std::move(task)), SUCCESS);

  // 4 清理本用例相关资源
  EXPECT_EQ(jit_executor->Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

TEST_F(JitExecutorUT, run_success_when_input_graph_contain_two_reshape_one_const_node) {
  // 1 准备构造jit executor的入参
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphExecutionQueue task_queue;
  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::OneConstTwoReshapeNodeTwoRelu();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  ExecutionOrder order({user_graph_id, compute_graph});
  CompileContext compile_context(graph_manager);
  CompiledModelCache cmc(user_graph_id, compile_context, graph_manager);
  std::mutex tmp_mutex;

  // 2 构造jit executor
  auto jit_executor = JitExecutor::Create(graph_manager, task_queue, order, compile_context, cmc, tmp_mutex);
  EXPECT_NE(jit_executor, nullptr);
  // 校验本次创建jit executor申请了1个stream，注册了1个device allocator
  auto &rts_stub = gert_stub_.GetAclRuntimeStub();
  EXPECT_EQ(rts_stub.GetAllRtStreams().size(), 1); // required 1 stream
  auto stream = rts_stub.GetAllRtStreams().at(0);
  auto allocator = ExternalAllocatorManager::GetExternalAllocator(stream);
  EXPECT_NE(allocator, nullptr); // required 1 allocator

  // 3 准备执行接口的入参并触发执行
  std::vector<int64_t> shape_dim = {2, 3, 3, 2};
  TensorDesc td(Shape(shape_dim), FORMAT_NCHW, DT_FLOAT);
  td.SetOriginShape(Shape(shape_dim));
  Tensor tensor(td);
  std::vector<int64_t> input_data_2 = {2, 3, 3, 2};
  TensorDesc desc_2(Shape({4}), FORMAT_NCHW, DT_INT32);
  desc_2.SetOriginShape(Shape({4}));
  Tensor input_tensor_2{desc_2};
  input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int64_t));
  std::vector<Tensor> inputs{tensor, input_tensor_2};
  std::vector<Tensor> outputs;
  const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(outputs.size(), 1);
    auto cur_dims = TensorTransUtils::GetDimsFromGertShape(outputs[0].GetStorageShape());
    EXPECT_EQ(cur_dims, shape_dim);
    return SUCCESS;
  };

  std::vector<gert::Tensor> gert_inputs;
  TensorTransUtils::Tensors2GertTensors(inputs, gert_inputs);
  UserGraphExecution task(user_graph_id, gert_inputs, callback, 0);
  EXPECT_EQ(jit_executor->RunWithCallback(std::move(task)), SUCCESS);

  // 4 清理本用例相关资源
  EXPECT_EQ(jit_executor->Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

// 该用例放开需要解决：1、unique算子原型中默认的attr定义；2、unique三类算子执行报错；3、切图前未作infershape
//TEST_F(JitExecutorUT, run_success_add_unique) {
//  // 1 准备构造jit executor的入参
//  uint64_t session_id = 0;
//  InnerSession inner_session(session_id, {});
//  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
//  UserGraphExecutionQueue task_queue;
//  uint32_t user_graph_id = 0u;
//  auto graph = JitShareGraph::AddUniqueNode();
//  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
//  for(auto node : compute_graph->GetDirectNode()) {
//    std::cout << node->GetName() << std::endl;
//  }
//  auto unique_node1 = compute_graph->FindNode("Unique_1");
//  EXPECT_NE(unique_node1, nullptr);
//  unique_node1->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
//  unique_node1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
//  ExecutionOrder order({user_graph_id, compute_graph});
//  CompileContext compile_context(inner_session);
//  CompiledModelCache cmc(user_graph_id, compile_context, inner_session);
//  std::mutex tmp_mutex;
//
//  // 3 构造jit executor
//  auto jit_executor = JitExecutor::Create(inner_session, task_queue, order, compile_context, cmc, tmp_mutex);
//  EXPECT_NE(jit_executor, nullptr);
//  // 校验本次创建jit executor申请了1个stream，注册了1个device allocator
//  auto &rts_stub = gert_stub_.GetRtsRuntimeStub();
//  EXPECT_EQ(rts_stub.GetAllRtStreams().size(), 1); // required 1 stream
//  auto stream = rts_stub.GetAllRtStreams().at(0);
//  auto allocator = ExternalAllocatorManager::GetExternalAllocator(stream);
//  EXPECT_NE(allocator, nullptr); // required 1 allocator
//
//  // 4 准备执行接口的入参并触发执行
//  std::vector<int64_t> output_shape_dim = {100};
//  std::vector<int64_t> shape_dim = {2, 3, 3, 2};
//  TensorDesc td(Shape(shape_dim), FORMAT_NCHW, DT_FLOAT);
//  td.SetOriginShape(Shape(shape_dim)); // todo check tfa set origin shape?
//  Tensor tensor(td);
//  std::vector<Tensor> inputs{tensor, tensor};
//  std::vector<Tensor> outputs;
//  const RunAsyncCallback callback = [&](Status status, std::vector<Tensor> &outputs) {
//    EXPECT_EQ(status, SUCCESS);
//    EXPECT_EQ(outputs.size(), 1);
//    EXPECT_EQ(outputs[0].GetTensorDesc().GetShape().GetDims(), output_shape_dim);
//    return SUCCESS;
//  };
//
//  UserGraphExecution task(inputs, callback);
//  EXPECT_EQ(jit_executor->RunWithCallback(std::move(task)), SUCCESS);
//
//  // 5 清理本用例相关资源
//  EXPECT_EQ(jit_executor->Finalize(), SUCCESS);
//  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
//}

TEST_F(JitExecutorUT, guard_hit_success_and_miss_success_when_input_graph_contain_add_node) {
  // 1 准备构造jit executor的入参
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphExecutionQueue task_queue;
  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::OneAddNode();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  ExecutionOrder order({user_graph_id, compute_graph});
  CompileContext compile_context(graph_manager);
  CompiledModelCache cmc(user_graph_id, compile_context, graph_manager);
  std::mutex tmp_mutex;

  // 3 构造jit executor
  auto jit_executor = JitExecutor::Create(graph_manager, task_queue, order, compile_context, cmc, tmp_mutex);
  EXPECT_NE(jit_executor, nullptr);
  // 校验本次创建jit executor申请了1个stream，注册了1个device allocator
  auto &rts_stub = gert_stub_.GetAclRuntimeStub();
  EXPECT_EQ(rts_stub.GetAllRtStreams().size(), 1); // required 1 stream
  auto stream = rts_stub.GetAllRtStreams().at(0);
  auto allocator = ExternalAllocatorManager::GetExternalAllocator(stream);
  EXPECT_NE(allocator, nullptr); // required 1 allocator

  // 4 准备执行接口的入参并触发执行
  std::vector<int64_t> shape_dim = {2, 3, 3, 2}; // 第一次出发guard命中场景
  TensorDesc td(Shape(shape_dim), FORMAT_NCHW, DT_FLOAT);
  td.SetOriginShape(Shape(shape_dim));
  Tensor tensor(td);
  std::vector<Tensor> inputs{tensor, tensor};
  std::vector<Tensor> outputs;
  const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(outputs.size(), 1);
    auto cur_dims = TensorTransUtils::GetDimsFromGertShape(outputs[0].GetStorageShape());
    EXPECT_EQ(cur_dims, shape_dim);
    return SUCCESS;
  };

  std::vector<gert::Tensor> gert_inputs;
  TensorTransUtils::Tensors2GertTensors(inputs, gert_inputs);
  UserGraphExecution task(user_graph_id, gert_inputs, callback, 10086);
  EXPECT_EQ(jit_executor->RunWithCallback(std::move(task)), SUCCESS);
  EXPECT_TRUE(!jit_executor->IsUserGraphNeedRebuild());

  std::vector<int64_t> shape_dim2 = {2, 3, 3, 1};// 第二次触发guard miss场景
  TensorDesc td2(Shape(shape_dim2), FORMAT_NCHW, DT_FLOAT);
  td2.SetOriginShape(Shape(shape_dim2));
  Tensor tensor2(td2);
  std::vector<Tensor> inputs_boardcast{tensor, tensor2};
  std::vector<gert::Tensor> gert_inputs_boardcast;
  TensorTransUtils::Tensors2GertTensors(inputs_boardcast, gert_inputs_boardcast);
  UserGraphExecution task_boardcast(user_graph_id, gert_inputs_boardcast, callback, 0);
  EXPECT_EQ(jit_executor->RunWithCallback(std::move(task_boardcast)), SUCCESS);
  // 5 清理本用例相关资源
  EXPECT_EQ(jit_executor->Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

TEST_F(JitExecutorUT, test_autofuse_flag_slice_schedule_open) {
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true;--experimental_enable_jit_executor_v2=true", 1);
  const bool enable_slice_schedule = (ge::GetAutofuseFlagValue("--enable_autofuse") == "true") &&
      (ge::GetAutofuseFlagValue("--experimental_enable_jit_executor_v2") == "true");
  EXPECT_EQ(enable_slice_schedule, true);
  unsetenv("AUTOFUSE_FLAGS");
}

TEST_F(JitExecutorUT, test_autofuse_flag_with_error_option_name) {
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true;--experimental_enable_jit_executor=true", 1);
  const bool enable_slice_schedule = (ge::GetAutofuseFlagValue("--enable_autofuse") == "true") &&
      (ge::GetAutofuseFlagValue("--experimental_enable_jit_executor_v2") == "true");
  EXPECT_EQ(enable_slice_schedule, false);
  unsetenv("AUTOFUSE_FLAGS");
}

TEST_F(JitExecutorUT, test_autofuse_flag_with_error_option_value) {
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=1;--experimental_enable_jit_executor_v2=true", 1);
  const bool enable_slice_schedule = (ge::GetAutofuseFlagValue("--enable_autofuse") == "true") &&
      (ge::GetAutofuseFlagValue("--experimental_enable_jit_executor_v2") == "true");
  EXPECT_EQ(enable_slice_schedule, false);
  unsetenv("AUTOFUSE_FLAGS");
}

TEST_F(JitExecutorUT, test_autofuse_flag_with_long_option) {
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true;--experimental_enable_jit_executor_v23=true", 1);
  const bool enable_slice_schedule = (ge::GetAutofuseFlagValue("--enable_autofuse") == "true") &&
      (ge::GetAutofuseFlagValue("--experimental_enable_jit_executor_v2") == "true");
  EXPECT_EQ(enable_slice_schedule, false);
  unsetenv("AUTOFUSE_FLAGS");
}

TEST_F(JitExecutorUT, test_autofuse_flag_with_slice_schedule_close) {
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true;--experimental_enable_jit_executor_v2=false", 1);
  const bool enable_slice_schedule = (ge::GetAutofuseFlagValue("--enable_autofuse") == "true") &&
      (ge::GetAutofuseFlagValue("--experimental_enable_jit_executor_v2") == "true");
  EXPECT_EQ(enable_slice_schedule, false);
  unsetenv("AUTOFUSE_FLAGS");
}

TEST_F(JitExecutorUT, test_autofuse_flag_with_autofuse_close) {
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=false;--experimental_enable_jit_executor_v2=true", 1);
  const bool enable_slice_schedule = (ge::GetAutofuseFlagValue("--enable_autofuse") == "true") &&
      (ge::GetAutofuseFlagValue("--experimental_enable_jit_executor_v2") == "true");
  EXPECT_EQ(enable_slice_schedule, false);
  unsetenv("AUTOFUSE_FLAGS");
}

TEST_F(JitExecutorUT, test_autofuse_flag_with_not_set_slice_schedule) {
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  const bool enable_slice_schedule = (ge::GetAutofuseFlagValue("--enable_autofuse") == "true") &&
      (ge::GetAutofuseFlagValue("--experimental_enable_jit_executor_v2") == "true");
  EXPECT_EQ(enable_slice_schedule, false);
  unsetenv("AUTOFUSE_FLAGS");
}

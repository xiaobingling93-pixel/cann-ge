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
#include <condition_variable>
#include <thread>
#include <mutex>
#include <future>
#include "graph/ge_context.h"
#include "ge_running_env/fake_ops_kernel_builder.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "register/op_tiling_registry.h"
#include "stub/gert_runtime_stub.h"
#include "api/gelib/gelib.h"

#include "graph/operator_reg.h"
#include "graph/ge_attr_value.h"
#include "common/dump/dump_manager.h"
#include "framework/executor/ge_executor.h"
#include "ge_running_env/fake_op.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge/ge_api.h"
#include "graph/utils/tensor_adapter.h"
#include "init_ge.h"
#include "utils/mock_ops_kernel_builder.h"
#include "utils/taskdef_builder.h"
#include "utils/graph_factory.h"
#include "ge_graph_dsl/assert/check_utils.h"

#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/debug/ge_attr_define.h"
#include "depends/op_stub/op_impl/less_important_op_impl.h"
#include "faker/global_data_faker.h"
#include "depends/profiler/src/profiling_test_util.h"
#include "register/register_custom_pass.h"


namespace ge {
namespace {

class SliceSchedulerMultiInstanceTest : public testing::Test {
 protected:
  void SetUp() {
    auto infer_fun = [](Operator &op) -> graphStatus {
      auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
      *op_desc->MutableOutputDesc(0) = *op_desc->GetInputDescPtr(0);
      return GRAPH_SUCCESS;
    };

    auto unique_infer_fun = [](Operator &op) -> graphStatus {
      auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
      const auto& input_desc = op_desc->GetInputDesc(0);
      const auto input_shape = input_desc.GetShape().GetDims();
      auto output0_desc = op_desc->MutableOutputDesc(0);
      output0_desc->SetShape(GeShape({-1}));
      output0_desc->SetShapeRange({
          {1, input_shape[0]}
      });
      op_desc->MutableOutputDesc(1)->SetDataType(DT_INT32);
      op_desc->MutableOutputDesc(1)->SetShape({});
      return GRAPH_SUCCESS;
    };

    auto shape_infer_fun = [](Operator &op) -> graphStatus {
      auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
      auto x_desc = op_desc->MutableInputDesc("x");
      const auto x_dim_num = x_desc->GetShape().GetDimNum();

      auto td = op_desc->MutableOutputDesc("y");
      td->SetShape(ge::GeShape({static_cast<int64_t>(x_dim_num)}));
      td->SetOriginShape(ge::GeShape({static_cast<int64_t>(x_dim_num)}));
      td->SetDataType(DT_INT64);
      td->SetOriginDataType(DT_INT64);
      return GRAPH_SUCCESS;
    };

    auto reshape_infer_fun = [](Operator &op) -> graphStatus {
      auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
      op_desc->SetOpInferDepends({"shape"});
      auto x_desc = op_desc->MutableInputDesc("x");
      auto y_desc = op_desc->MutableOutputDesc("y");
      auto x_shape = vector<int64_t>(x_desc->GetShape().GetDims());
      std::string shape_input_name = "shape";

      ge::Tensor shape_tensor;
      op.GetInputConstData(shape_input_name.c_str(), shape_tensor);

      ge::GeShape output_shape;
      auto shape_tensor_desc = op_desc->MutableInputDesc("shape");
      auto &shape_ref = shape_tensor_desc->MutableShape();
      auto shape_dims = shape_ref.GetDims();

      int64_t dim_num = shape_dims[0];
      const int64_t *shape_data = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(shape_tensor.GetData()));
      std::vector<int64_t> out_dims;
      int64_t product = 1;
      for (int64_t i = 0; i < dim_num; i++) {
        auto dim = shape_data[i];
        if (dim != 0 && product > (INT64_MAX / dim)) {
          return ge::GRAPH_PARAM_INVALID;
        }
        out_dims.push_back(dim);
        product *= dim;
      }

      auto td = op_desc->MutableOutputDesc("y");
      td->SetShape(ge::GeShape(out_dims));
      td->SetOriginShape(ge::GeShape(out_dims));
      td->SetDataType(op_desc->MutableInputDesc("x")->GetDataType());
      td->SetOriginDataType(op_desc->MutableInputDesc("x")->GetDataType());
      return ge::GRAPH_SUCCESS;
    };

    auto ge_dev = GeRunningEnvFaker();
    ge_dev.Reset()
        .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
        .Install(FakeEngine(kEngineNameAiCpu).KernelInfoStore(kEngineNameAiCpu))
        .Install(FakeEngine(kEngineNameAiCpuTf).KernelInfoStore(kEngineNameAiCpuTf))
        .Install(FakeEngine("DNN_VM_HOST_CPU").KernelInfoStore("DNN_VM_HOST_CPU_OP_STORE"))
        .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("DNN_VM_RTS_OP_STORE"))
        .Install(FakeEngine("DNN_VM_AICPU").KernelInfoStore("DNN_VM_AICPU"))
        .Install(FakeOp("Unique").InfoStoreAndBuilder("aicpu_ascend_kernel").InferShape(unique_infer_fun))
        .Install(FakeOp(ADD).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_fun))
        .Install(FakeOp(MUL).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_fun))
        .Install(FakeOp("Abs").InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_fun))
        .Install(FakeOp(RELU).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_fun))
        .Install(FakeOp(NEG).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_fun))
        .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(VARIABLE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(SHAPE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(shape_infer_fun))
        .Install(FakeOp(RESHAPE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(reshape_infer_fun));
  }

  void TearDown() {
    GEFinalize();
    ReInitGe();
  }
};

template <typename Func, typename... Args>
using invoke_result = typename std::result_of<Func(Args...)>::type;

template <typename Func, typename... Args>
std::map<int, invoke_result<Func, Args...>> ConcurrentProcess(Func&& func, int thread_nums, Args&&... args) {
  typedef invoke_result<Func, Args...> ResultType;
  std::map<int, std::future<ResultType>> futureMap;

  for (int i = 0; i < thread_nums; ++i) {
    futureMap.insert(std::make_pair(i,std::async(std::launch::async, std::bind(func, std::forward<Args>(args)...))));
  }
  std::map<int, ResultType> resultMap;
  for (auto& entry : futureMap) {
    ResultType result = entry.second.get();
    resultMap.insert(std::make_pair(entry.first, result));
  }
  return resultMap;
}

std::map<AscendString, AscendString> DefaultOptions() {
  std::map<AscendString, AscendString> options;
  options[OPTION_EXEC_ENABLE_DUMP_DEBUG] = "1";
  options[OPTION_EXEC_DUMP_PATH] = "./";
  options[OPTION_EXEC_DUMP_DEBUG_MODE] = "aicore_overflow"; // OP_DEBUG_ATOMIC /  OP_DEBUG_ALL
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  options[ge::OPTION_CONST_LIFECYCLE] = "graph";
  options[JIT_COMPILE.c_str()] = "1";
  return options;
}

std::string PrintShape(const Shape &shape) {
  if (shape.GetDims().size() == 0U) {
    return "()";
  }
  std::stringstream ss;
  ss << "(";
  for (size_t i = 0U; i < shape.GetDimNum(); i++) {
    ss << shape.GetDim(i) << ((i + 1) == shape.GetDimNum() ? ")" : ",");
  }
  return ss.str();
}

Status GetDump(Graph &graph, const std::string &graph_name) {
  ComputeGraphPtr compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  setenv("DUMP_GE_GRAPH", "2", 1);
  setenv("DUMP_GRAPH_LEVEL", "1", 1);
  GE_DUMP(compute_graph, graph_name);
  unsetenv("DUMP_GE_GRAPH");
  unsetenv("DUMP_GRAPH_LEVEL");
  return SUCCESS;
}

Tensor CreateTensor(const TensorDesc &tensor_desc) {
  int64_t tensor_size = -1;
  TensorUtils::GetTensorSizeInBytes(TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc), tensor_size);
  std::vector<uint8_t> tensor_buffer(tensor_size);
  Tensor tensor(tensor_desc);
  tensor.SetData(std::move(tensor_buffer));
  return tensor;
}

Tensor CreateTensor(const std::vector<int64_t> &dims, Format format = FORMAT_ND, DataType data_type = DT_FLOAT) {
  return CreateTensor(TensorDesc(Shape(dims), format, data_type));
}

Status EXPECT_AddGraph(std::mutex &mutex, Session &session, GraphId graph_id, Graph &recover_ir_graph, std::map<AscendString, AscendString> &graph_options) {
  std::lock_guard<std::mutex> lock(mutex);
  GELOGT(TRACE_RUNNING, "Thread %d AddGraph start session addr is %d", std::this_thread::get_id(), &session);
  Status ret = session.AddGraph(graph_id, recover_ir_graph, graph_options);
  EXPECT_EQ(ret, SUCCESS);
  GELOGT(TRACE_RUNNING, "Thread %d AddGraph end", std::this_thread::get_id());
  return ret;
}

std::vector<Tensor> EXPECT_RunGraphAsync_withStatus(std::mutex &mutex, Session &session, GraphId graph_id, const std::vector<Tensor> &inputs, Status expectStatus) {
  std::lock_guard<std::mutex> lock(mutex);
  GELOGT(TRACE_RUNNING, "Thread %d RunGraphAsync start session addr is %d", std::this_thread::get_id(), &session);
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  Status ret = SUCCESS;
  std::vector<Tensor> outputs;
  RunAsyncCallback callback = [&](Status status, std::vector<ge::Tensor> &output_tensors) {
    std::unique_lock<std::mutex> lk(mu);
    ret = status;
    outputs = output_tensors;
    done = true;
    cv.notify_all();
  };

  ret = session.RunGraphAsync(graph_id, inputs, callback);
  if (ret == SUCCESS) {
    std::unique_lock<std::mutex> lk(mu);
    if (!cv.wait_for(lk, std::chrono::seconds(15), [&]() { return done; })) {
      ret = FAILED;
    }
  }
  EXPECT_EQ(ret, expectStatus);
  GELOGT(TRACE_RUNNING, "Thread %d RunGraphAsync end", std::this_thread::get_id());
  return outputs;
}

std::vector<Tensor> EXPECT_RunGraphAsync(std::mutex &mutex, Session &session, GraphId graph_id, const std::vector<Tensor> &inputs) {
  return EXPECT_RunGraphAsync_withStatus(mutex, session, graph_id, inputs, SUCCESS);
}

Status GenerateTaskForAiCore(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) {
  tasks.emplace_back(AiCoreTaskDefBuilder(node).BuildAtomicAddrCleanTask());
  tasks.emplace_back(AiCoreTaskDefBuilder(node).BuildTask(true));
  return SUCCESS;
}

Status GenerateTaskForAicpuDependRange(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) {
  tasks.emplace_back(AicpuTaskDefBuilder(node).BuildAicpuTask(3));
  return SUCCESS;
}

Graph BuildStaticInputGraph() {
  DEF_GRAPH(slice_scheduler_static_graph) {
    auto data1 = OP_CFG(DATA)
                     .InCnt(1)
                     .OutCnt(1)
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 24, 24});
    auto data2 = OP_CFG(DATA)
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 24, 24});
    auto add1 = OP_CFG(ADD)
                    .InCnt(2)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 24, 24});
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
  };
  return ToGeGraph(slice_scheduler_static_graph);
}

/**
 *
┌────────┐  (0,0)   ┌────────┐  (0,0)   ┌────────┐  (0,0)   ┌─────────┐  (0,0)   ┌─────────────┐
│ _arg_0 │ ───────> │  add   │ ───────> │ unique │ ───────> │  add1   │ ───────> │ Node_Output │
└────────┘          └────────┘          └────────┘          └─────────┘          └─────────────┘
                      ∧                                       ∧
                      │ (0,1)                                 │ (0,1)
                      │                                       │
                    ┌────────┐                              ┌─────────┐
                    │ _arg_1 │                              │ const_0 │
                    └────────┘                              └─────────┘
 */
Graph BuildDynamicInputGraph() {
  DEF_GRAPH(slice_scheduler_dynamic_graph) {
    GeTensor tensor(GeTensorDesc(GeShape({1}), FORMAT_ND, DT_FLOAT));
    float32_t value = 6.3f;
    tensor.SetData((uint8_t *)&value, sizeof(value));

    auto data_0 = OP_CFG(DATA)
                      .InCnt(1)
                      .OutCnt(1)
                      .Attr(ATTR_NAME_INDEX, 0)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto data_1 = OP_CFG(DATA).InCnt(1)
                      .OutCnt(1)
                      .Attr(ATTR_NAME_INDEX, 1)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto add = OP_CFG(ADD)
                   .InCnt(2)
                   .OutCnt(1)
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto unique_op = OP_CFG("Unique")
                         .InCnt(1)
                         .OutCnt(2)
                         .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto add_1 = OP_CFG(ADD)
                     .InCnt(2)
                     .OutCnt(1)
                     .TensorDesc(FORMAT_ND, DT_FLOAT, {-1})
                     .Build("add1");

    auto const_0 = OP_CFG(CONSTANTOP)
                       .OutCnt(1)
                       .Attr(ATTR_NAME_WEIGHTS, tensor)
                       .TensorDesc(FORMAT_ND, DT_FLOAT, {1});

    auto net_output = OP_CFG(NETOUTPUT)
                          .InCnt(1)
                          .OutCnt(1)
                          .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    CHAIN(NODE("_arg_0", data_0)
              ->NODE("add", add)
              ->NODE("unique", unique_op)
              ->NODE(add_1)
              ->NODE("Node_Output", net_output));
    CHAIN(NODE("_arg_1", data_1)->NODE("add", add));
    CHAIN(NODE("const_0", const_0)->NODE(add_1));
  };

  return ToGeGraph(slice_scheduler_dynamic_graph);
}

Graph BuildLargeDynamicInputGraph() {
  DEF_GRAPH(dynamic_graph_reshape) {
    vector<int64_t> dims_vec_0 = {2, 3, 4};
    vector<int32_t> data_vec_0 = {2, 1, 4, 1, 2, -2, 3, 9, 1, 2, 0, 7, 5};
    GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_ND, DT_FLOAT);
    GeTensorPtr tensor = std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

    auto data_0 = OP_CFG(DATA)
                      .InCnt(1)
                      .OutCnt(1)
                      .Attr(ATTR_NAME_INDEX, 0)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1})
                      .Build("_arg_0");
    auto data_1 = OP_CFG(DATA)
                      .InCnt(1)
                      .OutCnt(1)
                      .Attr(ATTR_NAME_INDEX, 0)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1})
                      .Build("_arg_1");
    auto data_2 = OP_CFG(DATA)
                      .InCnt(1)
                      .OutCnt(1)
                      .Attr(ATTR_NAME_INDEX, 0)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1})
                      .Build("_arg_2");

    auto const_0 = OP_CFG(CONSTANTOP)
                       .OutCnt(1)
                       .Weight(tensor)
                       .TensorDesc(FORMAT_ND, DT_FLOAT, {})
                       .Build("const_0");

    auto add_0 = OP_CFG(ADD)
                     .InCnt(2)
                     .OutCnt(1)
                     .TensorDesc(FORMAT_ND, DT_INT32, {-1, -1, -1})
                     .Build("add_0");

    auto add_1 = OP_CFG(ADD)
                     .InCnt(2)
                     .OutCnt(1)
                     .TensorDesc(FORMAT_ND, DT_INT32, {-1, -1, -1})
                     .Build("add_1");

    auto abs_0 = OP_CFG("Abs")
                     .InCnt(1)
                     .OutCnt(1)
                     .TensorDesc(FORMAT_ND, DT_INT32, {-1, -1, -1})
                     .Build("abs_0");

    auto abs_1 = OP_CFG("Abs")
                     .InCnt(1)
                     .OutCnt(1)
                     .TensorDesc(FORMAT_ND, DT_INT32, {-1, -1, -1})
                     .Build("abs_1");

    auto abs_2 = OP_CFG("Abs")
                     .InCnt(1)
                     .OutCnt(1)
                     .TensorDesc(FORMAT_ND, DT_INT32, {-1, -1, -1})
                     .Build("abs_2");

    auto relu_0 = OP_CFG(RELU)
                     .InCnt(1)
                     .OutCnt(1)
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, -1, -1})
                      .Build("relu_0");

    auto relu_1 = OP_CFG(RELU)
                      .InCnt(1)
                      .OutCnt(1)
                      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, -1, -1})
                      .Build("relu_1");

    auto shape_0 = OP_CFG(SHAPE)
                     .InCnt(1)
                     .OutCnt(1)
                     .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                     .Attr("_force_infershape_when_running", true)
                     .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1})
                     .Build("shape_0");

    auto shape_1 = OP_CFG(SHAPE)
                       .InCnt(1)
                       .OutCnt(1)
                       .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                       .Attr("_force_infershape_when_running", true)
                       .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1})
                       .Build("shape_1");

    auto shape_2 = OP_CFG(SHAPE)
                       .InCnt(1)
                       .OutCnt(1)
                       .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                       .Attr("_force_infershape_when_running", true)
                       .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1})
                       .Build("shape_2");

    auto reshape_0 = OP_CFG(RESHAPE)
                       .InCnt(2)
                       .OutCnt(1)
                       .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                       .Attr("_force_infershape_when_running", true)
                       .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1})
                       .Build("reshape_0");

    auto reshape_1 = OP_CFG(RESHAPE)
                         .InCnt(2)
                         .OutCnt(1)
                         .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                         .Attr("_force_infershape_when_running", true)
                         .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1})
                         .Build("reshape_1");

    auto reshape_2 = OP_CFG(RESHAPE)
                         .InCnt(2)
                         .OutCnt(1)
                         .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                         .Attr("_force_infershape_when_running", true)
                         .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1})
                         .Build("reshape_2");

    auto net_output = OP_CFG(NETOUTPUT)
                          .InCnt(1)
                          .OutCnt(1)
                          .TensorDesc(FORMAT_ND, DT_INT32, {-1, -1, -1})
                          .Build("net_output");

    CHAIN(NODE(data_0) -> EDGE(0, 0) -> NODE(add_0) -> NODE(abs_0) -> EDGE(0, 0) -> NODE(reshape_0)
              -> NODE(abs_1) -> NODE(relu_0) -> EDGE(0, 0) -> NODE(reshape_1)
              -> EDGE(0, 0) -> NODE(add_1) -> NODE(relu_1) -> EDGE(0, 0) -> NODE(reshape_2)
              -> NODE(abs_2)-> NODE(net_output));
    CHAIN(NODE(const_0) -> EDGE(0, 1) -> NODE(add_0));
    CHAIN(NODE(const_0) -> EDGE(0, 1) -> NODE(add_1));
    CHAIN(NODE(data_1) -> EDGE(0, 0) -> NODE(shape_1) -> EDGE(0, 1) -> NODE(reshape_0));
    CHAIN(NODE(data_0) -> EDGE(0, 0) -> NODE(shape_0) -> EDGE(0, 1) -> NODE(reshape_1));
    CHAIN(NODE(data_2) -> EDGE(0, 0) -> NODE(shape_2) -> EDGE(0, 1) -> NODE(reshape_2));
  };
  return ToGeGraph(dynamic_graph_reshape);
}

Graph BuildDynamicInputGraphWithVarAndConst() {
  DEF_GRAPH(slice_scheduler_dynamic_graph_deep2) {
    GeTensorDesc tensor_desc(GeShape({1}), FORMAT_ND, DT_INT32);
    GeTensor tensor(tensor_desc);
    int32_t value = 2;
    tensor.SetData((uint8_t *) &value, sizeof(int32_t));

    auto data_0 = OP_CFG(DATA)
                      .InCnt(1)
                      .OutCnt(1)
                      .Attr(ATTR_NAME_INDEX, 0)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {-1})
                      .Build("_arg_0");

    auto data_1 = OP_CFG(DATA)
                      .InCnt(1)
                      .OutCnt(1)
                      .Attr(ATTR_NAME_INDEX, 1)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {-1})
                      .Build("_arg_1");

    auto unique_2 = OP_CFG("Unique")
                        .InCnt(1)
                        .OutCnt(2)
                        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1})
                        .Build("unique_2");

    auto unique_3 = OP_CFG("Unique")
                        .InCnt(1)
                        .OutCnt(2)
                        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1})
                        .Build("unique_3");

    auto unique_4 = OP_CFG("Unique")
                        .InCnt(1)
                        .OutCnt(2)
                        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1})
                        .Build("unique_4");

    auto add_bridge1 = OP_CFG(ADD)
                           .InCnt(2)
                           .OutCnt(1)
                           .TensorDesc(FORMAT_ND, DT_FLOAT, {-1})
                           .Build("add_bridge1");

    auto add_bridge2 = OP_CFG(ADD)
                           .InCnt(2)
                           .OutCnt(1)
                           .TensorDesc(FORMAT_ND, DT_FLOAT, {-1})
                           .Build("add_bridge2");

    auto add_bridge3 = OP_CFG(ADD)
                           .InCnt(2)
                           .OutCnt(1)
                           .TensorDesc(FORMAT_ND, DT_FLOAT, {-1})
                           .Build("add_bridge3");

    auto const_0 = OP_CFG(CONSTANTOP)
                       .OutCnt(1)
                       .Attr(ATTR_NAME_WEIGHTS, tensor)
                       .TensorDesc(FORMAT_ND, DT_FLOAT, {1})
                       .Build("const_0");  // 添加Build()

    auto net_output = OP_CFG(NETOUTPUT)
                          .InCnt(1)
                          .OutCnt(1)
                          .TensorDesc(FORMAT_ND, DT_FLOAT, {-1})
                          .Build("net_output");

    CHAIN(NODE(data_0)
              -> NODE(add_bridge1)
              -> NODE(unique_2) -> EDGE(0, 0)
              -> NODE(add_bridge2)
              -> NODE(unique_3) -> EDGE(0, 0)
              -> NODE(add_bridge3)
              -> NODE(unique_4) -> EDGE(0, 0)
              -> NODE(net_output));

    CHAIN(NODE(data_1) -> EDGE(0, 1) -> NODE(add_bridge1));
    CHAIN(NODE(const_0) -> EDGE(0, 1) -> NODE(add_bridge2));
    CHAIN(NODE(const_0) -> EDGE(0, 1)-> NODE(add_bridge3));
  };
  return ToGeGraph(slice_scheduler_dynamic_graph_deep2);
}

} // namespace


TEST_F(SliceSchedulerMultiInstanceTest, TestSliceScheduler_ForDynamicGraph_DisableJIT) {
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);

  auto ir_graph = BuildDynamicInputGraph();
  auto slogStub = std::make_shared<gert::SlogStubImpl>();
  slogStub->SetLevel(DLOG_DEBUG);
  slogStub->Clear();
  ge::SlogStub::SetInstance(slogStub);

  auto options = DefaultOptions();
  options[JIT_COMPILE.c_str()] = "0";
  std::map<AscendString, AscendString> graph_options;
  Session session(options);
  GraphId graph_id = 1;
  std::vector<Tensor> inputs;
  inputs.emplace_back(CreateTensor({16}));
  inputs.emplace_back(CreateTensor({16}));
  // add graph
  std::mutex mtx;
  std::map<int, Status> addResultMap = ConcurrentProcess(EXPECT_AddGraph, 3, std::ref(mtx), std::ref(session),
                                                         graph_id, std::ref(ir_graph), std::ref(graph_options));
  // build graph
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
  // run graph
  std::map<int, std::vector<Tensor>> resultMap = ConcurrentProcess(EXPECT_RunGraphAsync, 3, std::ref(mtx), std::ref(session), graph_id, inputs);
  // 检测未走断图流程
  EXPECT_EQ(slogStub->FindLog(DLOG_DEBUG, "Start to commit user graph execution task"), -1);
  EXPECT_EQ(slogStub->FindLog(DLOG_DEBUG, "GetRemainingNodes completed. uninfer nodes size"), -1);

  session.RemoveGraph(graph_id);

  ge::SlogStub::SetInstance(nullptr);
}

TEST_F(SliceSchedulerMultiInstanceTest, TestSliceScheduler_ForDynamicGraph_EnableDynamicBatch) {
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);

  auto ir_graph = BuildDynamicInputGraph();
  auto slogStub = std::make_shared<gert::SlogStubImpl>();
  slogStub->SetLevel(DLOG_DEBUG);
  slogStub->Clear();
  ge::SlogStub::SetInstance(slogStub);
  EXPECT_EQ(setenv("OPTION_DYNAMIC_BATCH_SIZE", "1,2,4,8,16", 1), SUCCESS);

  auto options = DefaultOptions();
  std::map<AscendString, AscendString> graph_options;
  Session session(options);
  GraphId graph_id = 37;
  std::vector<Tensor> inputs;
  inputs.emplace_back(CreateTensor({16}));
  inputs.emplace_back(CreateTensor({16}));

  // add graph
  std::mutex mtx;
  std::map<int, Status> addResultMap = ConcurrentProcess(EXPECT_AddGraph, 3, std::ref(mtx), std::ref(session),
                                                         graph_id, std::ref(ir_graph), std::ref(graph_options));
  // build graph
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
  // run graph
  std::map<int, std::vector<Tensor>> resultMap = ConcurrentProcess(EXPECT_RunGraphAsync, 3, std::ref(mtx), std::ref(session), graph_id, inputs);
  // 检测未走断图流程
  EXPECT_EQ(slogStub->FindLog(DLOG_DEBUG, "Start to commit user graph execution task"), -1);
  EXPECT_EQ(slogStub->FindLog(DLOG_DEBUG, "GetRemainingNodes completed. uninfer nodes size"), -1);

  session.RemoveGraph(graph_id);

  ge::SlogStub::SetInstance(nullptr);
  unsetenv("OPTION_DYNAMIC_BATCH_SIZE");
}

TEST_F(SliceSchedulerMultiInstanceTest, TestSliceScheduler_ForStaticGraph) {
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);

  auto ir_graph = BuildStaticInputGraph();
  auto slogStub = std::make_shared<gert::SlogStubImpl>();
  slogStub->SetLevel(DLOG_DEBUG);
  slogStub->Clear();
  ge::SlogStub::SetInstance(slogStub);

  auto options = DefaultOptions();
  std::map<AscendString, AscendString> graph_options;
  Session session(options);
  GraphId graph_id = 707;
  std::vector<Tensor> inputs;
  inputs.emplace_back(CreateTensor({16}));
  inputs.emplace_back(CreateTensor({16}));

  // add graph
  std::mutex mtx;
  std::map<int, Status> addResultMap = ConcurrentProcess(EXPECT_AddGraph, 3, std::ref(mtx), std::ref(session),
                                                         graph_id, std::ref(ir_graph), std::ref(graph_options));
  // build graph
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
  // run graph
  std::map<int, std::vector<Tensor>> resultMap = ConcurrentProcess(EXPECT_RunGraphAsync, 3, std::ref(mtx), std::ref(session), graph_id, inputs);
  // 检测未走断图流程
  EXPECT_EQ(slogStub->FindLog(DLOG_DEBUG, "Start to commit user graph execution task"), -1);
  EXPECT_EQ(slogStub->FindLog(DLOG_DEBUG, "GetRemainingNodes completed. uninfer nodes size"), -1);

  session.RemoveGraph(graph_id);

  ge::SlogStub::SetInstance(nullptr);
}

TEST_F(SliceSchedulerMultiInstanceTest, TestSliceScheduler_ForDynamicGraph) {
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);

  auto ir_graph = BuildDynamicInputGraph();
  GetDump(ir_graph, "DynamicGraph");
  auto slogStub = std::make_shared<gert::SlogStubImpl>();
  slogStub->SetLevel(DLOG_DEBUG);
  slogStub->Clear();
  ge::SlogStub::SetInstance(slogStub);

  auto options = DefaultOptions();
  std::map<AscendString, AscendString> graph_options;
  Session session(options);
  GraphId graph_id = 9;
  std::vector<Tensor> inputs;
  inputs.emplace_back(CreateTensor({16}));
  inputs.emplace_back(CreateTensor({16}));
  // add graph
  std::mutex mtx;
  std::map<int, Status> addResultMap = ConcurrentProcess(EXPECT_AddGraph, 3, std::ref(mtx), std::ref(session),
                                                         graph_id, std::ref(ir_graph), std::ref(graph_options));
  // build graph
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
  // run graph
  std::map<int, std::vector<Tensor>> resultMap = ConcurrentProcess(EXPECT_RunGraphAsync, 3, std::ref(mtx),
                                                                   std::ref(session), graph_id, inputs);
  // 检测进入断图
  EXPECT_TRUE(slogStub->FindLog(DLOG_DEBUG, "Start to commit user graph execution task") >= 0);
  // 校验切图
  EXPECT_TRUE(slogStub->FindLog(DLOG_DEBUG, "GetRemainingNodes completed. uninfer nodes size") >= 0);

  session.RemoveGraph(graph_id);

  ge::SlogStub::SetInstance(nullptr);
}

TEST_F(SliceSchedulerMultiInstanceTest, TestSliceScheduler_ForLargeDynamicGraph) {
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);

  auto ir_graph = BuildLargeDynamicInputGraph();
  GetDump(ir_graph, "LargeDynamicGraph");
  auto slogStub = std::make_shared<gert::SlogStubImpl>();
  slogStub->SetLevel(DLOG_DEBUG);
  slogStub->Clear();
  ge::SlogStub::SetInstance(slogStub);

  std::map<AscendString, AscendString> options = DefaultOptions();
  std::map<AscendString, AscendString> graph_options;
  Session session(options);
  GraphId graph_id = 56;
  std::vector<Tensor> inputs;
  inputs.emplace_back(CreateTensor({2, 3, 4}));
  inputs.emplace_back(CreateTensor({1, 4, 6}));
  inputs.emplace_back(CreateTensor({4, 3, 2}));
  uint16_t thread_num = 3;

  // add graph
  std::mutex mtx;
  std::map<int, Status> addResultMap = ConcurrentProcess(EXPECT_AddGraph, thread_num, std::ref(mtx), std::ref(session),
                                                         graph_id, std::ref(ir_graph), std::ref(graph_options));
  // build graph
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
  // run graph
  std::map<int, std::vector<Tensor>> resultMap = ConcurrentProcess(EXPECT_RunGraphAsync, thread_num, std::ref(mtx), std::ref(session), graph_id, inputs);
  // 检测进入断图
  EXPECT_TRUE(slogStub->FindLog(DLOG_DEBUG, "Start to commit user graph execution task") >= 0);
  // 校验切图
  EXPECT_TRUE(slogStub->FindLog(DLOG_DEBUG, "GetRemainingNodes completed. uninfer nodes size") >= 0);
  // 校验 output
  EXPECT_EQ(resultMap.size() == thread_num, true);
  for (auto iter = resultMap.cbegin(); iter != resultMap.cend(); ++iter) {
    auto tensor = iter->second.at(0);
    auto shape = tensor.GetTensorDesc().GetShape();
    std::string shape_str = PrintShape(shape);
    GELOGT(TRACE_RUNNING, "ForLargeDynamicGraph thread %d output shape %s", iter->first, shape_str.c_str());
    EXPECT_EQ(shape_str == "(4,3,2)", true);
  }

  session.RemoveGraph(graph_id);

  ge::SlogStub::SetInstance(nullptr);
}

TEST_F(SliceSchedulerMultiInstanceTest, TestSliceScheduler_ForRemovingGraph) {
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);

  auto slogStub = std::make_shared<gert::SlogStubImpl>();
  slogStub->SetLevel(DLOG_DEBUG);
  slogStub->Clear();
  ge::SlogStub::SetInstance(slogStub);

  std::map<AscendString, AscendString> options = DefaultOptions();
  std::map<AscendString, AscendString> graph_options;
  Session session(options);
  GraphId graph_id_1 = 11;
  GraphId graph_id_2 = 12;
  std::vector<Tensor> inputs1;
  inputs1.emplace_back(CreateTensor({1, 3, 4}));
  inputs1.emplace_back(CreateTensor({1, 3, 4}));
  inputs1.emplace_back(CreateTensor({2, 1, 6}));

  std::mutex mtx;
  // process graph 1
  auto ir_graph_1 = BuildLargeDynamicInputGraph();
  ConcurrentProcess(EXPECT_AddGraph, 3, std::ref(mtx), std::ref(session), graph_id_1,
                    std::ref(ir_graph_1), std::ref(graph_options));
  EXPECT_EQ(session.BuildGraph(graph_id_1, inputs1), SUCCESS);
  std::map<int, std::vector<Tensor>> result_map_1 = ConcurrentProcess(EXPECT_RunGraphAsync, 3, std::ref(mtx),
                                                                      std::ref(session), graph_id_1, inputs1);
  // 检测进入断图
  EXPECT_TRUE(slogStub->FindLog(DLOG_DEBUG, "Start to commit user graph execution task") >= 0);
  // 校验切图
  EXPECT_TRUE(slogStub->FindLog(DLOG_DEBUG, "GetRemainingNodes completed. uninfer nodes size") >= 0);

  // process graph 2
  slogStub->Clear();
  std::vector<Tensor> inputs2;
  inputs2.emplace_back(CreateTensor({16}));
  inputs2.emplace_back(CreateTensor({16}));
  auto ir_graph_2 = BuildDynamicInputGraphWithVarAndConst();
  ConcurrentProcess(EXPECT_AddGraph, 3, std::ref(mtx), std::ref(session),graph_id_2,
                    std::ref(ir_graph_2), std::ref(graph_options));
  EXPECT_EQ(session.BuildGraph(graph_id_2, inputs2), SUCCESS);
  std::map<int, std::vector<Tensor>> result_map_2 = ConcurrentProcess(EXPECT_RunGraphAsync, 3,std::ref(mtx),
                                                                      std::ref(session), graph_id_2, inputs2);
  // 检测进入断图
  EXPECT_TRUE(slogStub->FindLog(DLOG_DEBUG, "Start to commit user graph execution task") >= 0);
  // 校验切图
  EXPECT_TRUE(slogStub->FindLog(DLOG_DEBUG, "GetRemainingNodes completed. uninfer nodes size") >= 0);

  session.RemoveGraph(graph_id_1);
  std::map<int, std::vector<Tensor>> result_map_3 = ConcurrentProcess(EXPECT_RunGraphAsync_withStatus, 3, std::ref(mtx),
                                                                      std::ref(session), graph_id_1, inputs1, FAILED);
  session.RemoveGraph(graph_id_2);

  ge::SlogStub::SetInstance(nullptr);
}

TEST_F(SliceSchedulerMultiInstanceTest, TestSliceScheduler_ForDynamicGraph_WithException) {
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);

  auto slogStub = std::make_shared<gert::SlogStubImpl>();
  slogStub->SetLevel(DLOG_DEBUG);
  slogStub->Clear();
  ge::SlogStub::SetInstance(slogStub);

  std::map<AscendString, AscendString> options = DefaultOptions();
  std::map<AscendString, AscendString> graph_options;

  Session session1(options);
  std::mutex mtx;
  auto ir_graph_1 = BuildDynamicInputGraph();
  GraphId graph_id_1 = 101;
  ConcurrentProcess(EXPECT_AddGraph, 3, std::ref(mtx), std::ref(session1), graph_id_1,
                    std::ref(ir_graph_1), std::ref(graph_options));

  // incorrect build
  std::vector<Tensor> build_inputs;
  build_inputs.emplace_back(CreateTensor({16}));
  build_inputs.emplace_back(CreateTensor({16}));
  // 空 graph
  EXPECT_NE(SUCCESS, session1.BuildGraph(102, build_inputs));
  session1.RemoveGraph(graph_id_1);

  // incorrect run
  Session session2(options);
  auto ir_graph_2 = BuildDynamicInputGraph();
  GraphId graph_id = 103;
  ConcurrentProcess(EXPECT_AddGraph, 3, std::ref(mtx), std::ref(session2), graph_id,
                    std::ref(ir_graph_2), std::ref(graph_options));
  EXPECT_EQ(SUCCESS, session2.BuildGraph(graph_id, build_inputs));
  std::vector<ge::Tensor> inputs;
  // incorrect input
  ConcurrentProcess(EXPECT_RunGraphAsync_withStatus, 3, std::ref(mtx), std::ref(session2), graph_id_1, inputs, FAILED);

  ge::SlogStub::SetInstance(nullptr);
}

}  // namespace ge

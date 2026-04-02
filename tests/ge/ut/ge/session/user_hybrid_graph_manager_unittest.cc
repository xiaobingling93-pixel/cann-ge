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
#include "session/user_hybrid_graph_manager.h"
#undef private
#include "ge_graph_dsl/graph_dsl.h"
#include "common/share_graph.h"
#include "common/memory/tensor_trans_utils.h"
#include "es_ge_test_ops.h"
#include "compliant_node_builder.h"
#include "graph/utils/graph_utils_ex.h"
#include "stub/gert_runtime_stub.h"
#include "graph/operator_factory_impl.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "ge_running_env/fake_graph_optimizer.h"
#include "ge_running_env/fake_engine.h"
#include <vector>
#include "common/model/external_allocator_manager.h"
#include "ge/st/stubs/utils/mock_ops_kernel_builder.h"
#include "ge_running_env/dir_env.h"
#include "faker/space_registry_faker.h"
#include "ge/ge_api.h"
#include "api/aclgrph/option_utils.h"
#include "compiler/api/gelib/gelib.h"
#include "session/session_manager.h"

using namespace testing;

namespace gert {
LowerResult LoweringAdd(const ge::NodePtr &node, const LowerInput &lower_input) {
  std::vector<bg::ValueHolderPtr> inputs;
  inputs.push_back(lower_input.global_data->GetStreamById(0));
  inputs.insert(inputs.cend(), lower_input.input_shapes.begin(), lower_input.input_shapes.end());
  inputs.insert(inputs.cend(), lower_input.input_addrs.begin(), lower_input.input_addrs.end());
  std::string name = "Execute_";
  auto out_count = node->GetAllOutDataAnchorsSize();
  auto outputs =
      bg::DevMemValueHolder::CreateDataOutput("Execute_", inputs, out_count * 2, node->GetOpDesc()->GetStreamId());
  return {HyperStatus::Success(),
          {*outputs.begin()},
          {outputs.begin(), outputs.begin() + out_count},
          {outputs.begin() + out_count, outputs.end()}};
}
REGISTER_NODE_CONVERTER("Add", LoweringAdd);

ge::graphStatus StubKernel(KernelContext *context) {
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(Execute_).RunFunc(StubKernel);
}

namespace ge {
bool EnableSliceSchedule() { // 桩函数
  return ((ge::GetAutofuseFlagValue(kAutoFuseEnableOption) == "true") &&
          (ge::GetAutofuseFlagValue(kSliceScheduleOption) == "true"));;
}
ge::graphStatus StubInferShape(ge::Operator &op) {
  auto x_input_desc = op.GetInputDesc(0);
  auto x_shape = x_input_desc.GetShape().GetDims();
  auto x_type = x_input_desc.GetDataType();
  std::vector<std::pair<int64_t, int64_t>> x_shape_range;
  (void)x_input_desc.GetShapeRange(x_shape_range);
  TensorDesc op_output_desc = op.GetOutputDesc(0);
  op_output_desc.SetShape(ge::Shape(x_shape));
  op_output_desc.SetOriginShape(ge::Shape(x_shape));
  op_output_desc.SetDataType(x_type);
  if (!x_shape_range.empty()) {
    op_output_desc.SetShapeRange(x_shape_range);
  }
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  return op_desc->UpdateOutputDesc(0, TensorAdapter::TensorDesc2GeTensorDesc(op_output_desc));
}

ge::graphStatus GetShapeInferShape(ge::Operator &op) {
  std::cout << "Enter infershape getshape" << std::endl;
  std::vector<std::string> tiling_inline_engine;
  tiling_inline_engine.push_back("AIcoreEngine");
  vector<std::string> export_shape_engine;
  export_shape_engine.push_back("AIcoreEngine");
  op.SetAttr("_op_tiling_inline_engine", tiling_inline_engine);
  op.SetAttr("_op_export_shape_engine", export_shape_engine);
  return ge::GRAPH_SUCCESS;
}

class UserHybridGraphManagerlUT : public testing::Test {
 protected:
  void SetUp() override {
    gert_stub_.GetKernelStub().StubTiling();
    RuntimeStub::Install(nullptr); // gert的rts stub不能在多线程环境下工作，因此使用默认rts stub
    AclRuntimeStub::Install(nullptr);
  }
  void TearDown() override {
    gert_stub_.Clear();
  }
  gert::GertRuntimeStub gert_stub_;
};

TEST_F(UserHybridGraphManagerlUT, AddGraph_RemoveGraph_Success) {
  uint64_t session_id = 0;
  ModelExecutor model_executor;
  model_executor.Initialize({}, session_id);
  GraphManager graph_manager;
  graph_manager.Initialize({}, &model_executor);
  UserGraphsManager user_graph_manager(graph_manager);
  UserHybridGraphManager user_hybrid_graph_manager(user_graph_manager);

  uint32_t user_graph_id = 0u;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::map<std::string, std::string> options;
  options["ge.inputShape"] = "data1:-1,-1,-1;data2:-1,-1,-1";
  options["ge.dynamicDims"] = "1,1,1,1,1,1;3,3,3,3,3,3;5,5,5,5,5,5";
  options["ge.dynamicNodeType"] = "1";
  options["ge.compileHybridMode"] = "1";
  options[OPTION_GRAPH_KEY] = "./cache";
  setenv("AUTOFUSE_FLAGS", "--enable_autofuse=true;--experimental_enable_jit_executor_v2=true", 1);
  EXPECT_EQ(user_hybrid_graph_manager.AddGraph(user_graph_id, graph, options), PARAM_INVALID);
  unsetenv("AUTOFUSE_FLAGS");
  EXPECT_EQ(user_hybrid_graph_manager.AddGraph(user_graph_id, graph, options), SUCCESS);
  EXPECT_EQ(user_hybrid_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_hybrid_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

TEST_F(UserHybridGraphManagerlUT, SelectExecuteGraph_MatchDynamicGear) {
  uint64_t session_id = 0;
  ModelExecutor model_executor;
  model_executor.Initialize({}, session_id);
  GraphManager graph_manager;
  graph_manager.Initialize({}, &model_executor);
  UserGraphsManager user_graph_manager(graph_manager);
  UserHybridGraphManager user_hybrid_graph_manager(user_graph_manager);

  const uint32_t graph_id = 0;
  HybridDynamicDimsInfo dynamic_dims_info;
  std::vector<std::vector<int64_t>> dynamic_shape_dims{{4,4,4}, {1,1,1}, {2,2,2}};
  std::vector<std::pair<std::string, std::vector<int64_t>>> user_input_dims;
  user_input_dims.push_back({"data1", {-1, 3, -1, 224}});
  user_input_dims.push_back({"data2", {1, 3, 224, 224}});
  user_input_dims.push_back({"data3", {3, 3, -1, 224}});
  dynamic_dims_info.dynamic_shape_dims = dynamic_shape_dims;
  dynamic_dims_info.user_input_dims = user_input_dims;
  uint32_t dynamic_gear_graph_id = 99;
  uint32_t dynamic_shape_graph_id = 9999;
  user_hybrid_graph_manager.SetDynamicGraphId(graph_id);
  EXPECT_TRUE(user_hybrid_graph_manager.TryGetDynamicGraphId(graph_id, dynamic_gear_graph_id, dynamic_shape_graph_id));
  user_hybrid_graph_manager.SetDynamicGearInfo(dynamic_gear_graph_id, dynamic_dims_info);

  std::vector<gert::Tensor> inputs;
  uint32_t selected_graph_id = 0U;
  ASSERT_NE(user_hybrid_graph_manager.SelectExecuteGraph(dynamic_gear_graph_id, dynamic_shape_graph_id,
                                                         inputs, selected_graph_id), SUCCESS);
  inputs.resize(3);
  ASSERT_NE(user_hybrid_graph_manager.SelectExecuteGraph(dynamic_gear_graph_id, dynamic_shape_graph_id,
                                                         inputs, selected_graph_id), SUCCESS);

  inputs.clear();
  std::vector<gert::Tensor> ge_inputs(3);
  ge_inputs[0].MutableStorageShape() = {4,3,4,224};
  ge_inputs[1].MutableStorageShape() = {1, 3, 224, 224};
  ge_inputs[2].MutableStorageShape() = {3,3,4,224};
  ASSERT_EQ(user_hybrid_graph_manager.SelectExecuteGraph(dynamic_gear_graph_id, dynamic_shape_graph_id,
                                                         ge_inputs, selected_graph_id), SUCCESS);
  ASSERT_TRUE(selected_graph_id == dynamic_gear_graph_id);
  graph_manager.Finalize();
}

TEST_F(UserHybridGraphManagerlUT, SelectExecuteGraph_MatchDynamicShape) {
  uint64_t session_id = 0;
  ModelExecutor model_executor;
  model_executor.Initialize({}, session_id);
  GraphManager graph_manager;
  graph_manager.Initialize({}, &model_executor);
  UserGraphsManager user_graph_manager(graph_manager);
  UserHybridGraphManager user_hybrid_graph_manager(user_graph_manager);

  const uint32_t graph_id = 0;
  HybridDynamicDimsInfo dynamic_dims_info;
  std::vector<std::vector<int64_t>> dynamic_shape_dims{{4,4,4}, {1,1,1}, {2,2,2}};
  std::vector<std::pair<std::string, std::vector<int64_t>>> user_input_dims;
  user_input_dims.push_back({"data1", {-1, 3, -1, 224}});
  user_input_dims.push_back({"data2", {1, 3, 224, 224}});
  user_input_dims.push_back({"data3", {3, 3, -1, 224}});
  dynamic_dims_info.dynamic_shape_dims = dynamic_shape_dims;
  dynamic_dims_info.user_input_dims = user_input_dims;
  user_hybrid_graph_manager.SetDynamicGraphId(graph_id);
  uint32_t dynamic_gear_graph_id = 99;
  uint32_t dynamic_shape_graph_id = 9999;
  EXPECT_TRUE(user_hybrid_graph_manager.TryGetDynamicGraphId(graph_id, dynamic_gear_graph_id, dynamic_shape_graph_id));
  user_hybrid_graph_manager.SetDynamicGearInfo(dynamic_gear_graph_id, dynamic_dims_info);

  std::vector<gert::Tensor> inputs(3);
  inputs[0].MutableStorageShape() = {3,3,3,224};
  inputs[1].MutableStorageShape() = {1, 3, 224, 224};
  inputs[2].MutableStorageShape() = {3,3,3,224};
  uint32_t selected_graph_id = 0U;
  ASSERT_EQ(user_hybrid_graph_manager.SelectExecuteGraph(dynamic_gear_graph_id, dynamic_shape_graph_id,
                                                         inputs, selected_graph_id), SUCCESS);
  ASSERT_TRUE(selected_graph_id == dynamic_shape_graph_id);
  graph_manager.Finalize();
}

TEST_F(UserHybridGraphManagerlUT, RunGraphAsyncTest) {
  vector<Tensor> inputs;
  vector<Tensor> outputs;
  ge::Tensor tensor1;
  TensorDesc tensor_desc1(Shape({3, 3, 3}), FORMAT_ND, DT_FLOAT);
  tensor1.SetTensorDesc(tensor_desc1);
  std::vector<uint8_t> data(27 * 4, 1);
  tensor1.SetData(data);
  inputs.emplace_back(tensor1);
  inputs.emplace_back(tensor1);

  ge::Tensor tensor2;
  TensorDesc tensor_desc2(Shape({3, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  tensor2.SetTensorDesc(tensor_desc2);
  std::vector<uint8_t> data2(27 * 4, 1);
  tensor2.SetData(data2);
  outputs.emplace_back(tensor2);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> empty_options;
  options["ge.inputShape"] = "data1:-1,-1,-1;data2:-1,-1,-1";
  options["ge.dynamicDims"] = "1,1,1,1,1,1;3,3,3,3,3,3;5,5,5,5,5,5";
  options["ge.dynamicNodeType"] = "1";
  options["ge.compileHybridMode"] = "1";
  EXPECT_EQ(GEInitialize(empty_options), SUCCESS);
  Session session(empty_options);
  ComputeGraphPtr com_graph = gert::ShareGraph::AicoreGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(com_graph);

  GraphId graph_id = 4;
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Data");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Add");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("NetOutput");
  auto instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);
  //  SchedulerConf conf;
  SchedulerConf scheduler_conf;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->name = "DNN_VM_GE_LOCAL";
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->id = "DNN_VM_GE_LOCAL";
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->independent = false;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->attach = true;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->skip_assign_stream = true;

  scheduler_conf.cal_engines["AIcoreEngine"] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines["AIcoreEngine"]->name = "AIcoreEngine";
  scheduler_conf.cal_engines["AIcoreEngine"]->id = "AIcoreEngine";
  scheduler_conf.cal_engines["AIcoreEngine"]->independent = false;
  scheduler_conf.cal_engines["AIcoreEngine"]->attach = false;
  scheduler_conf.cal_engines["AIcoreEngine"]->skip_assign_stream = false;

  scheduler_conf.cal_engines["DNN_VM_AICPU"] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->name = "DNN_VM_AICPU";
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->id = "DNN_VM_AICPU";
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->independent = false;
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->attach = true;
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->skip_assign_stream = false;

  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"]->name = "DNN_VM_GE_LOCAL_OP_STORE";
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"]->id = "DNN_VM_GE_LOCAL_OP_STORE";
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"]->independent = false;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"]->attach = true;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"]->skip_assign_stream = true;

  instance_ptr->DNNEngineManagerObj().schedulers_["multi_batch"] = scheduler_conf;

  GeRunningEnvFaker ge_env;
  auto multi_dims = MakeShared<FakeMultiDimsOptimizer>();
  ge_env.Install(FakeEngine("AIcoreEngine").KernelInfoStore("AiCoreLib").GraphOptimizer("AIcoreEngine").Priority(PriorityEnum::COST_0));
  ge_env.Install(FakeEngine("VectorEngine").KernelInfoStore("VectorLib").GraphOptimizer("VectorEngine").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_AICPU").KernelInfoStore("AicpuLib").GraphOptimizer("aicpu_tf_optimizer").Priority(PriorityEnum::COST_3));
  ge_env.Install(FakeEngine("DNN_VM_AICPU_ASCEND").KernelInfoStore("AicpuAscendLib").GraphOptimizer("aicpu_ascend_optimizer").Priority(PriorityEnum::COST_2));
  ge_env.Install(FakeEngine("DNN_HCCL").KernelInfoStore("ops_kernel_info_hccl").GraphOptimizer("hccl_graph_optimizer").GraphOptimizer("hvd_graph_optimizer").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("RTSLib").GraphOptimizer("DNN_VM_RTS_GRAPH_OPTIMIZER_STORE").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_9));
  ge_env.Install(FakeEngine("DNN_VM_HOST_CPU").KernelInfoStore("DNN_VM_HOST_CPU_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_10));
  ge_env.Install(FakeEngine("DSAEngine").KernelInfoStore("DSAEngine").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("AIcoreEngine").GraphOptimizer("MultiDims", multi_dims));
  ge_env.Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(CASE).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(STREAMACTIVE).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(EXIT).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(SEND).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(SENDNOTIFY).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(RECV).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("MapIndex").InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("UpdateTensorDesc").InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("LabelSet").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("LabelSwitchByIndex").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("LabelGotoEx").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(MUL).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(DATA).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(ADD).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("GetShape").InferShape(GetShapeInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(CONCAT).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(CONCATV2).InfoStoreAndBuilder("AiCoreLib"));
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
  EXPECT_FALSE(session.IsGraphNeedRebuild(graph_id));
  EXPECT_EQ(session.RunGraphAsync(graph_id, inputs, nullptr), SUCCESS);
  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  EXPECT_TRUE(session.IsGraphNeedRebuild(graph_id));
  EXPECT_EQ(GEFinalize(), SUCCESS);
  RuntimeStub::Reset();
  AclRuntimeStub::Reset();
  ge_env.Reset();
}

}  // namespace ge
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model_v2_executor_unittest.h"
#include <gtest/gtest.h>
#include "graph/operator_factory_impl.h"
#include "graph/types.h"
#include "core/execution_data.h"
#include "register/kernel_registry.h"
#include "exe_graph/runtime/tensor.h"
#include "faker/fake_value.h"
#include "faker/model_desc_holder_faker.h"

#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "common/bg_test.h"
#include "faker/magic_ops.h"
#include "faker/model_data_faker.h"
#include "proto/insert_op.pb.h"
#include "graph/utils/tensor_utils.h"
#include "framework/runtime/executor_option/multi_thread_executor_option.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/task_producer_factory.h"
#include "graph/ge_context.h"
#include "graph/utils/graph_dump_utils.h"
#include "framework/runtime/model_rt_var_manager.h"
#include "graph/load/model_manager/model_manager.h"
#include "common/opskernel/ops_kernel_info_types.h"

// using namespace ge;
namespace gert {
namespace {
const char_t *const kEnvName = "ASCEND_OPP_PATH";
const char_t *const kBuiltIn = "built-in";
const char_t *const kVendors = "vendors";
const char_t *const kOpMasterDeviceLib = "/op_impl/ai_core/tbe/op_master_device/lib/";
LowerResult LoweringFoo(const ge::NodePtr &node, const LowerInput &lower_input) {
  auto rt_session = bg::GetRtSession(*lower_input.global_data);
  auto ret = bg::DevMemValueHolder::CreateSingleDataOutput("GetTestSessionId", {rt_session},
                                                           node->GetOpDesc()->GetStreamId());
  LowerResult result;
  result.out_shapes.push_back(lower_input.input_shapes[0]);
  result.out_addrs.push_back(ret);
  return result;
}

ge::graphStatus GetTestSessionId(KernelContext *context) {
  auto rt_session = context->GetInputValue<RtSession *>(0);
  //auto session_id = context->GetOutputPointer<uint64_t>(0);
  //*session_id = rt_session->GetSessionId();

  auto session_id_holder = context->GetOutputPointer<GertTensorData>(0);
  auto session_id_value = rt_session->GetSessionId();
  memcpy_s(session_id_holder->GetTensorData().GetAddr(), sizeof(uint64_t), &session_id_value, sizeof(uint64_t));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateGetSessionIdTensorDataAtHost(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto session_id_chain = context->GetOutput(0);
  const auto data_type_size = ge::GetSizeByDataType(ge::DT_UINT64);
  auto malloc_buffer_size = data_type_size + sizeof(GertTensorData);
  auto out_data = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[malloc_buffer_size]);
  GE_ASSERT_NOTNULL(out_data);
  new (out_data.get())
    GertTensorData(out_data.get() + sizeof(GertTensorData), malloc_buffer_size - sizeof(TensorData), kOnHost, -1);
  //auto tensor = new (std::nothrow) TensorData();
  //tensor->SetPlacement(kOnHost);
  session_id_chain->SetWithDefaultDeleter<uint8_t[]>(out_data.release());
  return ge::GRAPH_SUCCESS;
}

static ge::GeRootModelPtr ConstructGeRootModel(
    const std::vector<std::pair<ge::ccKernelType, const std::string>> &kernel_type_so_names) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  ge::GeRootModelPtr ge_root_model = std::make_shared<ge::GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), ge::SUCCESS);
  ge::GeModelPtr ge_model = std::make_shared<ge::GeModel>();
  ge_root_model->SetSubgraphInstanceNameToModel("graph", ge_model);
  ge_model->SetGraph(graph);

  domi::ModelTaskDef model_task_def;
  std::shared_ptr<domi::ModelTaskDef> model_task_def_ptr = make_shared<domi::ModelTaskDef>(model_task_def);
  for (const auto &item : kernel_type_so_names) {
    domi::TaskDef *task_def = model_task_def_ptr->add_task();
    ge_model->SetModelTaskDef(model_task_def_ptr);
    task_def->set_type(static_cast<uint32_t>(ge::ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL));

    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_so_name(item.second);
    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_kernel_type(static_cast<uint32_t>(item.first));
    context->set_op_index(1);
  }
  return ge_root_model;
}

static void ConstructOpMasterDeviceSo(
    const std::string &opp_path, const size_t built_in_num, const size_t cust_num, const bool &is_cust_same,
    std::vector<std::pair<ge::ccKernelType, const std::string>> &kernel_type_so_names) {
  for (size_t i = 0UL; i < built_in_num; ++i) {
    std::string inner_op_master = opp_path + kBuiltIn + kOpMasterDeviceLib;
    system(("mkdir -p " + inner_op_master).c_str());
    inner_op_master += std::to_string(i) + "-Ascend-V7.6-libopmaster.so";
    system(("touch " + inner_op_master).c_str());
    system(("echo 'Ascend-V7.6-libopmaster' > " + inner_op_master).c_str());
    kernel_type_so_names.emplace_back(ge::ccKernelType::AI_CPU, inner_op_master);
  }

  std::string vendor_names = "vendor=";
  for (size_t i = 0UL; i < cust_num; ++i) {
    std::string vendor_name = "cust-" + std::to_string(i);
    std::string inner_op_master = opp_path + kVendors + "/" + vendor_name + kOpMasterDeviceLib;
    system(("mkdir -p " + inner_op_master).c_str());
    inner_op_master += "libcust_opmaster.so";
    system(("touch " + inner_op_master).c_str());
    if (is_cust_same) {
      system(("echo 'Ascend-V7.6-libopmaster' > " + inner_op_master).c_str());
    } else {
      system(("echo " + std::to_string(i) + " > " + inner_op_master).c_str());
    }
    vendor_names.append(vendor_name + ",");
    kernel_type_so_names.emplace_back(ge::ccKernelType::CUST_AI_CPU, inner_op_master);
  }

  std::string vendor_config = opp_path + kVendors + "/config.ini";
  system(("touch " + vendor_config).c_str());
  system(("echo " + vendor_names + " > " + vendor_config).c_str());
}
} // namespace


class ExecutorUnitTest : public bg::BgTest {
  void SetUp() override {
    bg::BgTest::SetUp();
    std::string opp_version = "./version.info";
    system(("touch " + opp_version).c_str());
    system(("echo 'Version=6.4.T5.0.B121' > " + opp_version).c_str());
    setenv("ASCEND_OPP_PATH", "./", 1);
  }

  void TearDown() override {
    bg::BgTest::TearDown();
    unsetenv("ASCEND_OPP_PATH");
   }
};

TEST_F(ExecutorUnitTest, CheckParam_Failed_WhenIoNumsError) {
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto model_executor = BuildExecutorFromSingleNode().executor;
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2}, 1);
  auto inputs = FakeTensors({256}, 1);
  ASSERT_NE(model_executor->Execute({nullptr}, reinterpret_cast<Tensor **>(inputs.GetAddrList()), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  outputs = FakeTensors({2}, 1);
  inputs = FakeTensors({256}, 3);
  ASSERT_NE(model_executor->Execute({nullptr}, reinterpret_cast<Tensor **>(inputs.GetAddrList()), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  outputs = FakeTensors({2}, 1);
  inputs = FakeTensors({256}, 3);
  ASSERT_NE(model_executor->Execute({nullptr}, reinterpret_cast<Tensor **>(inputs.GetAddrList()), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), 0),
            ge::GRAPH_SUCCESS);

  outputs = FakeTensors({2}, 2);
  inputs = FakeTensors({256}, 3);
  ASSERT_NE(model_executor->Execute({nullptr}, reinterpret_cast<Tensor **>(inputs.GetAddrList()), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
}

TEST_F(ExecutorUnitTest, CheckParam_Failed_WhenNullIoTensor) {
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto model_executor = BuildExecutorFromSingleNode().executor;
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2}, 1);
  auto tensors = FakeTensors({256}, 1);
  std::vector<Tensor *> inputs = {tensors.data(), nullptr};
  ASSERT_NE(model_executor->Execute({nullptr}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  auto input_tensors = FakeTensors({256}, 2);
  std::vector<Tensor *> output_tensors = {nullptr};
  ASSERT_NE(model_executor->Execute({nullptr}, reinterpret_cast<Tensor **>(input_tensors.GetAddrList()), inputs.size(),
                                    output_tensors.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
}
TEST_F(ExecutorUnitTest, test_graph_executor_for_add_graph_run_success) {
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto model_executor = BuildExecutorFromSingleNode().executor;
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2}, 1);
  auto input0 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input1 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input2 = FakeValue<uint64_t>(0);

  ASSERT_EQ(
      model_executor->Execute({input2.value}, std::vector<Tensor *>({input0.holder.get(), input1.holder.get()}).data(),
                              2, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
      ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}
TEST_F(ExecutorUnitTest, State_Failed_ExecuteWithoutLoad) {
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto model_executor = BuildExecutorFromSingleNode().executor;
  ASSERT_NE(model_executor, nullptr);

  auto outputs = FakeTensors({2}, 1);
  auto input0 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input1 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input2 = FakeValue<uint64_t>(0);

  ASSERT_NE(
      model_executor->Execute({input2.value}, std::vector<Tensor *>({input0.holder.get(), input1.holder.get()}).data(),
                              2, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
      ge::GRAPH_SUCCESS);
}
TEST_F(ExecutorUnitTest, State_Failed_LoadDuplicated) {
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto model_executor = BuildExecutorFromSingleNode().executor;
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  ASSERT_NE(model_executor->Load(), ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}
TEST_F(ExecutorUnitTest, State_Failed_UnloadWithoutLoad) {
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto model_executor = BuildExecutorFromSingleNode().executor;
  ASSERT_NE(model_executor, nullptr);
  ASSERT_NE(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}
TEST_F(ExecutorUnitTest, State_Failed_ExecuteWithInputShapeSizeBiggerThanRangeSize) {
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto model_executor = BuildExecutorFromSingleNode().executor;
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2}, 1);
  auto inputs = FakeTensors({3, 4}, 2);
  auto input2 = FakeValue<uint64_t>(0);

  for (size_t i = 0UL; i < inputs.size(); ++i) {
    const auto model_input_desc = model_executor->GetModelDesc().GetInputDesc(i);
    EXPECT_FALSE(model_input_desc->IsOriginShapeInRange(inputs.GetTensorList()[i]->GetOriginShape()));
    EXPECT_FALSE(model_input_desc->IsShapeInRange(inputs.GetTensorList()[i]->GetStorageShape()));
  }
  EXPECT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}

TEST_F(ExecutorUnitTest, State_Failed_ExecuteWithInvalidInputShape) {
  auto exe_graph = BuildExeGraphFromSingleNodeWithShapeAndRange({{-1}, {-1}, {-1}, {-1}}, {{1}, {1}, {1}, {1}},
                                                                {{99}, {99}, {99}, {99}});
  ASSERT_NE(exe_graph, nullptr);

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  // fake compute graph
  auto compute_graph = std::make_shared<ge::ComputeGraph>("tests");
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2}, 1);
  auto inputs = FakeTensors({1, 100, 100, 100}, 2);
  auto input2 = FakeValue<uint64_t>(0);

  for (size_t i = 0UL; i < inputs.size(); ++i) {
    const auto model_input_desc = model_executor->GetModelDesc().GetInputDesc(i);
    EXPECT_FALSE(model_input_desc->IsOriginShapeInRange(inputs.GetTensorList()[i]->GetOriginShape()));
    EXPECT_FALSE(model_input_desc->IsShapeInRange(inputs.GetTensorList()[i]->GetStorageShape()));
  }
  EXPECT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}

TEST_F(ExecutorUnitTest, test_load_const_data_run_success) {
  // 1.为Foo算子的converter和相关kernel打桩
  // 注册Foo算子的node converter
  // 注册GetSessionId kernel的实现
  GertRuntimeStub stub;
  stub.GetConverterStub().Register("Foo", LoweringFoo);
  KernelRegistry::KernelFuncs funcs = {};
  funcs.run_func = GetTestSessionId;
  funcs.outputs_creator = CreateGetSessionIdTensorDataAtHost;
  stub.GetKernelStub().SetUp("GetTestSessionId", funcs);

  // 2.构造Foo graph
  auto compute_graph = ShareGraph::SimpleFooGraph();
  // 设置输出数据类型为UINT64_T,与session id数据类型一致
  auto foo_node = compute_graph->FindFirstNodeMatchType("Foo");
  foo_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_UINT64);
  auto netoutput_node = compute_graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_UINT64);
  compute_graph->TopologicalSorting();

  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  // 3.将计算图lowering成执行图
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "ExecutorBuilder_FooGetSessionExeGraph");

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  // 4.构造rt session输入
  uint64_t session_id = 100U;
  auto session = gert::RtSession(session_id);
  gert::OuterWeightMem weight = {nullptr, 0U};
  ModelLoadArg load_arg(&session, weight);
  ModelExecuteArg arg = {(void *)2};
  // 5. 执行加载流程
  ASSERT_EQ(model_executor->Load(&arg, load_arg), ge::GRAPH_SUCCESS);

  // 6. 执行main图
  auto mem_block = std::unique_ptr<uint8_t[]>(new uint8_t[8]);
  auto outputs = FakeTensors({2}, 1, mem_block.get(), kOnHost);  // fake value内部写死了tensor size为4，不好
  auto input0 =
      FakeValue<Tensor>(Tensor{{{1}, {1}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_UINT64, 0});
  ASSERT_EQ(model_executor->Execute(&arg, std::vector<Tensor *>({input0.holder.get()}).data(), 1,
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  // 7. 校验main图输出value为GetSessionId，即100U.
  auto ret_session_id = outputs.at(0).GetData<uint64_t>();
  EXPECT_EQ(*ret_session_id, session_id);
}

TEST_F(ExecutorUnitTest, TestAippSetAndGetSuccess) {
  auto graph = ShareGraph::BuildAippDataGraph();
  graph->TopologicalSorting();

  ge::NamedAttrs aipp_attr;
  aipp_attr.SetAttr("aipp_mode", ge::GeAttrValue::CreateFrom<int64_t>(domi::AippOpParams_AippMode_dynamic));
  aipp_attr.SetAttr("related_input_rank", ge::GeAttrValue::CreateFrom<int64_t>(0));
  aipp_attr.SetAttr("max_src_image_size", ge::GeAttrValue::CreateFrom<int64_t>(2048));
  aipp_attr.SetAttr("support_rotation", ge::GeAttrValue::CreateFrom<int64_t>(1));

  auto aippData1 = graph->FindNode("aippData1");
  auto op_desc = aippData1->GetOpDesc();
  ge::GeTensorDesc tensor_desc(ge::GeShape(), ge::FORMAT_NHWC, ge::DT_FLOAT);
  ge::TensorUtils::SetSize(tensor_desc, 512);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->UpdateOutputDesc(0, tensor_desc);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  op_desc->SetOpKernelLibName("GeLocal");
  ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_INDEX, 0);
  ge::AttrUtils::SetNamedAttrs(op_desc, ge::ATTR_NAME_AIPP, aipp_attr);
  ge::AttrUtils::SetStr(op_desc, ge::ATTR_DATA_RELATED_AIPP_MODE, "dynamic_aipp");
  ge::AttrUtils::SetStr(op_desc, ge::ATTR_DATA_AIPP_DATA_NAME_MAP, "aippData1");

  std::vector<string> inputs = { "NCHW:DT_FLOAT:TensorName:100:3:1,2,8" };
  ge::AttrUtils::SetListStr(op_desc, ge::ATTR_NAME_AIPP_INPUTS, inputs);
  std::vector<string> outputs = { "NCHW:DT_FLOAT:TensorName:100:3:1,2,8" };
  ge::AttrUtils::SetListStr(op_desc, ge::ATTR_NAME_AIPP_OUTPUTS, outputs);

  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDefForAll(AiCoreTaskDefFaker("stub_func")).AddWeight().BuildGeRootModel();
  auto model_data = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  ge::graphStatus error_code = ge::GRAPH_FAILED;
  auto stream_executor = gert::LoadExecutorFromModelData(model_data.Get(), {}, nullptr, nullptr, nullptr, error_code);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);
  ASSERT_NE(stream_executor, nullptr);

  ge::AippConfigInfo aipp_info;
  ge::InputAippType aipp_type;
  size_t aipp_index = 0;

  // Has been set
  ge::Status ret = stream_executor->GetAippInfo(0, aipp_info);
  ASSERT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(aipp_info.aipp_mode, domi::AippOpParams_AippMode_dynamic);
  ret = stream_executor->GetAippType(0, aipp_type, aipp_index);
  ASSERT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(aipp_type, ge::DATA_WITH_DYNAMIC_AIPP);
  EXPECT_TRUE(aipp_index == 0);

  ge::OriginInputInfo orig_input_info;
  std::vector<ge::InputOutputDims> input_dims;
  std::vector<ge::InputOutputDims> output_dims;
  ASSERT_EQ(ge::SUCCESS, stream_executor->GetOriginAippInputInfo(0, orig_input_info));
  ASSERT_EQ(orig_input_info.format, ge::FORMAT_NCHW);
  ASSERT_EQ(orig_input_info.data_type, ge::DT_FLOAT);
  ASSERT_EQ(orig_input_info.dim_num, 3);

  ASSERT_EQ(ge::SUCCESS, stream_executor->GetAllAippInputOutputDims(0, input_dims, output_dims));
  ASSERT_EQ(input_dims.size(), 1);
  ASSERT_EQ(input_dims[0].name, "TensorName");
  ASSERT_EQ(input_dims[0].size, 100);
  ASSERT_EQ(input_dims[0].dim_num, 3);
  std::vector<int64_t> verify_vec{1,2,8};
  ASSERT_EQ(input_dims[0].dims, verify_vec);

  ASSERT_EQ(output_dims.size(), 1);
  ASSERT_EQ(output_dims[0].name, "TensorName");
  ASSERT_EQ(output_dims[0].size, 100);
  ASSERT_EQ(output_dims[0].dim_num, 3);
  ASSERT_EQ(output_dims[0].dims, verify_vec);
}

TEST_F(ExecutorUnitTest, TestAippGetFailed) {
  auto graph = ShareGraph::BuildAippDataGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDefForAll(AiCoreTaskDefFaker("stub_func")).AddWeight().BuildGeRootModel();
  auto model_data = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  ge::graphStatus error_code;
  auto stream_executor = gert::LoadExecutorFromModelData(model_data.Get(), error_code);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);
  ASSERT_NE(stream_executor, nullptr);

  ge::AippConfigInfo aipp_info;
  ge::InputAippType aipp_type;
  size_t aipp_index = 0;
  ge::Status ret;
  // Has been set
  ret = stream_executor->GetAippInfo(0, aipp_info);
  EXPECT_EQ(ret, ACL_ERROR_GE_AIPP_NOT_EXIST);
  ret = stream_executor->GetAippType(0, aipp_type, aipp_index);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(aipp_type, ge::DATA_WITHOUT_AIPP);
  EXPECT_EQ(aipp_index, 0xFFFFFFFF);
}

/*
 * 设置了ATTR_NAME_INPUT_DIMS属性时，aipp shape为从属性上拿到的shape，无range
 * */
TEST_F(ExecutorUnitTest, TestAippShapeV2SetAndGetSuccess_When_Set_ATTR_NAME_INPUT_DIMS) {
  auto graph = ShareGraph::BuildAippDataGraph();
  graph->TopologicalSorting();
  auto aippData1 = graph->FindNode("aippData1");
  auto op_desc = aippData1->GetOpDesc();

  const std::vector<int64_t> input_dims {1, 2, 3, 4};
  ge::GeTensorDesc tensor_desc(ge::GeShape(input_dims), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorUtils::SetSize(tensor_desc, 24);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->UpdateOutputDesc(0, tensor_desc);
  op_desc->SetOpKernelLibName("GeLocal");
  ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_INDEX, 0);

  ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_INPUT_DIMS, input_dims);

  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDefForAll(AiCoreTaskDefFaker("stub_func")).AddWeight().BuildGeRootModel();
  auto model_data = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  ge::graphStatus error_code;
  auto stream_executor = gert::LoadExecutorFromModelData(model_data.Get(), error_code);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);
  ASSERT_NE(stream_executor, nullptr);
  Shape tmp_shape({1, 2, 3, 4});
  EXPECT_EQ(stream_executor->GetModelDesc().GetInputDesc(0)->GetAippShape(), tmp_shape);

  RtSession session(999);
  auto executor2 = gert::LoadExecutorFromModelDataWithRtSession(model_data.Get(), &session, error_code);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);
}

/*
 * 设置了ATTR_DYNAMIC_AIPP_INPUT_DIMS属性时，aipp shape为从属性上拿到的shape，range为input_desc上拿到的range
 * */
TEST_F(ExecutorUnitTest, TestAippShapeV2SetAndGetSuccess_When_Set_ATTR_DYNAMIC_AIPP_INPUT_DIMS) {
  auto graph = ShareGraph::BuildAippDataGraph();
  graph->TopologicalSorting();
  auto aippData1 = graph->FindNode("aippData1");
  auto op_desc = aippData1->GetOpDesc();

  const std::vector<int64_t> input_dims {-1, -1, -1, -1};
  const std::vector<std::pair<int64_t, int64_t>> input_ranges {{1, -1}, {1, -1}, {1, -1}, {1, -1}};

  ge::GeTensorDesc tensor_desc(ge::GeShape(input_dims), ge::FORMAT_NCHW, ge::DT_FLOAT);
  tensor_desc.SetShapeRange(input_ranges);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->UpdateOutputDesc(0, tensor_desc);
  op_desc->SetOpKernelLibName("GeLocal");
  ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_INDEX, 0);

  ge::AttrUtils::SetListInt(op_desc, ge::ATTR_DYNAMIC_AIPP_INPUT_DIMS, input_dims);

  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDefForAll(AiCoreTaskDefFaker("stub_func")).AddWeight().BuildGeRootModel();
  auto model_data = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  ge::graphStatus error_code;
  auto stream_executor = gert::LoadExecutorFromModelData(model_data.Get(), error_code);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);
  ASSERT_NE(stream_executor, nullptr);
  Shape tmp_shape({-1, -1, -1, -1});

  EXPECT_EQ(stream_executor->GetModelDesc().GetInputDesc(0)->GetAippShape(), tmp_shape);
}

/*
 * 未设置任何属性时，shape与range均从input_desc上获取
 * */
TEST_F(ExecutorUnitTest, TestAippShapeV2SetAndGetSuccess_When_No_Set_Any_ATTR) {
  auto graph = ShareGraph::BuildAippDataGraph();
  graph->TopologicalSorting();
  auto aippData1 = graph->FindNode("aippData1");
  auto op_desc = aippData1->GetOpDesc();

  const std::vector<int64_t> input_dims {1, 2, 3, 4};

  ge::GeTensorDesc tensor_desc(ge::GeShape(input_dims), ge::FORMAT_NCHW, ge::DT_FLOAT);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->UpdateOutputDesc(0, tensor_desc);
  op_desc->SetOpKernelLibName("GeLocal");
  ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_INDEX, 0);

  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDefForAll(AiCoreTaskDefFaker("stub_func")).AddWeight().BuildGeRootModel();
  auto model_data = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  ge::graphStatus error_code;
  auto stream_executor = gert::LoadExecutorFromModelData(model_data.Get(), error_code);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);
  ASSERT_NE(stream_executor, nullptr);
  Shape tmp_shape({1, 2, 3, 4});
  ShapeRange tmp_shape_range;

  EXPECT_EQ(stream_executor->GetModelDesc().GetInputDesc(0)->GetAippShape(), tmp_shape);
}

TEST_F(ExecutorUnitTest, MultiThreadExecutor_TaskSchedulerByKernel_Success){
  auto exe_graph_with_root_model = BuildExeGraphFromSingleNode();
  auto exe_graph = exe_graph_with_root_model.exe_graph;

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  TaskProducerFactory::GetInstance().SetProducerType(TaskProducerType::KERNEL);
  ASSERT_EQ(TaskProducerFactory::GetInstance().GetProducerType(), TaskProducerType::KERNEL);
  auto option = MultiThreadExecutorOption(kLeastThreadNumber);
  auto compute_graph = std::make_shared<ge::ComputeGraph>("tests");
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto model_executor = ModelV2Executor::Create(exe_graph, option, root_model);

  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2}, 1);
  auto input0 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input1 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input2 = FakeValue<uint64_t>(0);

  ASSERT_EQ(
      model_executor->Execute({input2.value}, std::vector<Tensor *>({input0.holder.get(), input1.holder.get()}).data(),
                              2, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
      ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}

TEST_F(ExecutorUnitTest, MultiThreadExecutor_LessCoreNum_Failed) {
  auto exe_graph_with_root_model = BuildExeGraphFromSingleNode();
  auto exe_graph = exe_graph_with_root_model.exe_graph;

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  TaskProducerFactory::GetInstance().SetProducerType(TaskProducerType::KERNEL);
  ASSERT_EQ(TaskProducerFactory::GetInstance().GetProducerType(), TaskProducerType::KERNEL);
  auto option = MultiThreadExecutorOption(kLeastThreadNumber - 1U);
  auto compute_graph = std::make_shared<ge::ComputeGraph>("tests");
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto model_executor = ModelV2Executor::Create(exe_graph, option, root_model);
  ASSERT_EQ(model_executor, nullptr);
}

TEST_F(ExecutorUnitTest, test_ExecuteSync_with_rtStreamSynchronize_timeout_run_failed) {
  const char_t *const kTimeoutEnvPath = "TIMEOUT";
  char_t record_path[MMPA_MAX_PATH] = "timeout";
  mmSetEnv(kTimeoutEnvPath, &record_path[0U], MMPA_MAX_PATH);
  ge::GetContext().SetStreamSyncTimeout(15000);

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto model_executor = BuildExecutorFromSingleNode().executor;
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2}, 1);
  auto input0 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input1 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input2 = FakeValue<uint64_t>(0);

  ASSERT_NE(
      model_executor->ExecuteSync(std::vector<Tensor *>({input0.holder.get(), input1.holder.get()}).data(),
                              2, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
      ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  unsetenv(kTimeoutEnvPath);
}

TEST_F(ExecutorUnitTest, ExecutorArg_IndexCorrect) {
  ASSERT_EQ(static_cast<int32_t>(ExecuteArgIndex::kNum), 4);
  ASSERT_EQ(static_cast<int32_t>(ExecuteArgIndex::kNotifies), -4);
  ASSERT_EQ(static_cast<int32_t>(ExecuteArgIndex::kRtEvents), -3);
  ASSERT_EQ(static_cast<int32_t>(ExecuteArgIndex::kExternalAllocator), -2);
  ASSERT_EQ(static_cast<int32_t>(ExecuteArgIndex::kStream), -1);
  ASSERT_EQ(static_cast<int32_t>(ExecuteArgIndex::kEnd), 0);
}

TEST_F(ExecutorUnitTest, LoadExecutorFromModelDataWithExternalStreamAllocator) {
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamTwoNodeGraph(stream_num, event_num);
  graph->TopologicalSorting();
  ASSERT_EQ(stream_num, 2);
  ASSERT_EQ(event_num, 2);

  GeModelBuilder builder(graph);
  auto ge_root_model = builder.SetRootModelStreamNum(stream_num)
                           .SetRootModelEventNum(event_num)
                           .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .FakeTbeBin({"Add"})
                           .AddTaskDef("Relu", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .FakeTbeBin({"Relu"})
                           .BuildGeRootModel();

  GertRuntimeStub rts_stub;
  {
    ASSERT_EQ(rts_stub.GetAclRuntimeStub().GetAllRtStreams().size(), 0);
    // load model v2 executor
    auto model_data = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
    ge::graphStatus error_code;
    StreamAllocator stream_allocator;
    EventAllocator event_allocator;
    NotifyAllocator notify_allocator;
    auto executor = gert::LoadExecutorFromModelData(model_data.Get(), {}, &stream_allocator, &event_allocator,
                                                    &notify_allocator, error_code);
    ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);
    ASSERT_NE(executor, nullptr);

    ASSERT_EQ(executor->GetModelDesc().GetReusableStreamNum(), 2);
    ASSERT_EQ(executor->GetModelDesc().GetReusableEventNum(), 2 + 1);     // origin 2 event, last sync 1 event
    ASSERT_EQ(rts_stub.GetRtsRuntimeStub().GetAllRtStreams().size(), 1);  // require 1 sub stream

    rtStream_t stream = 0;
    // execute
    ModelExecuteArg arg;
    arg.stream = stream;
    arg.external_stream_allocator = &stream_allocator;
    arg.external_event_allocator = &event_allocator;

    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    executor->Execute(arg, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(), outputs.size());
    // no more stream acquired during executing with same stream allocator
    ASSERT_EQ(rts_stub.GetRtsRuntimeStub().GetAllRtStreams().size(), 1);
  }
}

TEST_F(ExecutorUnitTest, LoadExecutorFromModelDataWithOnlyExternalStreamAllocator) {
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamTwoNodeGraph(stream_num, event_num);
  graph->TopologicalSorting();
  ASSERT_EQ(stream_num, 2);
  ASSERT_EQ(event_num, 2);

  GeModelBuilder builder(graph);
  auto ge_root_model = builder.SetRootModelStreamNum(stream_num)
      .SetRootModelEventNum(event_num)
      .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
      .FakeTbeBin({"Add"})
      .AddTaskDef("Relu", AiCoreTaskDefFaker("AddStubBin").WithHandle())
      .FakeTbeBin({"Relu"})
      .BuildGeRootModel();

  GertRuntimeStub rts_stub;
  {
    // load model v2 executor
    auto model_data = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
    ge::graphStatus error_code;
    StreamAllocator stream_allocator;
    EventAllocator event_allocator;
    NotifyAllocator notify_allocator;
    auto executor = gert::LoadExecutorFromModelData(model_data.Get(), {}, &stream_allocator, &event_allocator,
                                                    &notify_allocator, error_code);
    ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);
    ASSERT_NE(executor, nullptr);

    ASSERT_EQ(executor->GetModelDesc().GetReusableStreamNum(), 2);
    ASSERT_EQ(executor->GetModelDesc().GetReusableEventNum(), 2 + 1);     // origin 2 event, last sync 1 event
    ASSERT_EQ(rts_stub.GetRtsRuntimeStub().GetAllRtStreams().size(), 1);  // require 1 sub stream

    rtStream_t stream = 0;
    // execute
    ModelExecuteArg arg;
    arg.stream = stream;
    arg.external_stream_allocator = &stream_allocator;
    arg.external_event_allocator = nullptr; // event allocator not come with stream allocator

    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    ASSERT_EQ(executor->Execute(arg, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(), outputs.size()),
              ge::PARAM_INVALID);
  }
}

TEST_F(ExecutorUnitTest, LoadExecutorFromModelDataWithVariableGraph_session_mismatch) {
  setenv("MOCK_AVAIL_STREAM_NUM", "1", 0); // only has 1 stream
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::GraphDynamicAndStaticGraphWithVariables(stream_num, event_num);
  graph->TopologicalSorting();
  ASSERT_EQ(stream_num, 1);

  auto relu = graph->FindFirstNodeMatchType("Relu");
  ASSERT_NE(relu, nullptr);

  ge::OperatorFactoryImpl::RegisterInferShapeFunc("Relu", [](ge::Operator &op) {return ge::GRAPH_SUCCESS;});
  AddCompileResult(relu, false);
  relu->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
  relu->GetOpDesc()->SetOpEngineName(ge::kEngineNameAiCore);
  relu->GetOpDescBarePtr()->SetStreamId(0);

  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("TransData", AiCoreTaskDefFaker("TransDataStubName"))
      .FakeTbeBin({"TransData"})
      .AddTaskDef("Relu", AiCoreTaskDefFaker("ReluStubName"))
      .FakeTbeBin({"Relu"})
      .SetRootModelStreamNum(stream_num)
      .SetRootModelEventNum(event_num)
      .BuildGeRootModel();

  auto model_data = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  GertStreamStub rts_stub;
  ge::graphStatus error_code;
  RtSession session(999);
  LoadExecutorArgs args{.rt_session = &session, .file_constant_mems = {}};
  auto executor = gert::LoadExecutorFromModelData(model_data.Get(), args, error_code);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);
  ASSERT_NE(executor, nullptr);
  StorageShape shape;
  TensorData memory;
  ASSERT_EQ(ModelRtVarManager::Instance(999)->GetVarShapeAndMemory("var2", shape, memory), ge::GRAPH_SUCCESS);

  // session id mis match
  RtSession session1(888);
  gert::ModelLoadArg load_arg1(&session1);
  EXPECT_NE(executor->Load({}, load_arg1), ge::GRAPH_SUCCESS);

  unsetenv("MOCK_AVAIL_STREAM_NUM");
  ASSERT_EQ(rts_stub.GetAllRtStreams().size(), 0);
}

TEST_F(ExecutorUnitTest, LoadExecutorFromModelDataWithRollbackSingleStream) {
  setenv("MOCK_AVAIL_STREAM_NUM", "1", 0); // only has 1 stream
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamTwoNodeGraph(stream_num, event_num);
  graph->TopologicalSorting();
  ASSERT_EQ(stream_num, 2);

  GeModelBuilder builder(graph);
  auto ge_root_model = builder.SetRootModelStreamNum(stream_num)
                           .SetRootModelEventNum(event_num)
                           .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .FakeTbeBin({"Add"})
                           .AddTaskDef("Relu", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .FakeTbeBin({"Relu"})
                           .BuildGeRootModel();

  auto model_data = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  GertStreamStub rts_stub;
  ge::graphStatus error_code;
  StreamAllocator stream_allocator;
  EventAllocator event_allocator;
  NotifyAllocator notify_allocator;
  auto executor = gert::LoadExecutorFromModelData(model_data.Get(), {}, &stream_allocator, &event_allocator,
                                                  &notify_allocator, error_code);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);
  ASSERT_NE(executor, nullptr);

  ASSERT_EQ(executor->GetModelDesc().GetReusableStreamNum(), 1);
  ASSERT_EQ(executor->GetModelDesc().GetReusableEventNum(), 0);
  unsetenv("MOCK_AVAIL_STREAM_NUM");
  ASSERT_EQ(rts_stub.GetAllRtStreams().size(), 0);
}

TEST_F(ExecutorUnitTest, LoadExecutorFromModelData_InitOpMasterDeviceSo_Success) {
  std::string opp_path = __FILE__;
  std::vector<std::pair<ge::ccKernelType, const std::string>> kernel_type_so_names;

  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "/test_tmp/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  ConstructOpMasterDeviceSo(opp_path, 1, 1, true, kernel_type_so_names);

  ge::ModelBufferData model;
  ge::GeRootModelPtr ge_root_model = ConstructGeRootModel(kernel_type_so_names);

  EXPECT_EQ(ge_root_model->CheckAndSetNeedSoInOM(), ge::SUCCESS);
  EXPECT_EQ(ge_root_model->GetSoInOmFlag(), 0x4000);

  std::string output_file = opp_path + "/output.om";
  ge::ModelHelper model_helper;
  model_helper.SetSaveMode(false);
  EXPECT_EQ(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, false), ge::SUCCESS);

  ge::ModelData model_data;
  model_data.model_data = reinterpret_cast<void *>(model.data.get());
  model_data.model_len = model.length;
  ge::graphStatus error_code;
  auto executor = gert::LoadExecutorFromModelData(model_data, error_code);
  EXPECT_EQ(executor, nullptr);

  std::string so_name = kernel_type_so_names.back().second;
  EXPECT_NE(ge::ModelManager::GetInstance().LoadCustAicpuSoAndUpdateSoName(0, so_name), ge::SUCCESS);
  EXPECT_EQ(so_name, "");
  system(("rm -rf " + opp_path).c_str());
}
}  // namespace gert

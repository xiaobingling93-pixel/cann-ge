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
#include <numeric>
#include <string>
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "faker/fake_value.h"
#include "runtime/base.h"
#include "ge/ge_api.h"
#include "ge/ge_api_error_codes.h"
#include "ge/ge_graph_compile_summary.h"
#include "graph/execute/model_executor.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/load/model_manager/model_utils.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "utils/mock_ops_kernel_builder.h"
#include "utils/taskdef_builder.h"
#include "stub/gert_runtime_stub.h"
#include "easy_graph/builder/graph_dsl.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg_box.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "utils/taskdef_builder.h"
#include "common/args_checker.h"
#include "graph/load/model_manager/model_manager.h"
#include "init_ge.h"
#include "common/opskernel/ops_kernel_info_store.h"
#include "graph/ge_local_context.h"
#include "graph/ge_global_options.h"
#include "utils/synchronizer.h"
#include "common/global_variables/diagnose_switch.h"
#include "hcom/hcom_topo_info.h"
#include "dflow/inc/data_flow/model/graph_model.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph/custom_op_factory.h"
#include "graph/custom_op.h"

namespace ge {
using namespace gert;
namespace {
Status GenerateTaskForCustomOp(const Node &node,
                               RunContext &run_context,
                               std::vector<domi::TaskDef> &tasks) {
  (void)node;
  (void)run_context;
  domi::TaskDef task_def = {};
  task_def.set_stream_id(node.GetOpDesc()->GetStreamId());
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  task_def.set_sqe_num(5);

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->mutable_context()->set_op_index(node.GetOpDesc()->GetId());
  tasks.push_back(task_def);
  return SUCCESS;
}

Status GenerateTaskForMemCopyAync(const Node &node,
                                  RunContext &run_context,
                                  std::vector<domi::TaskDef> &tasks) {
  if ((node.GetType() != MEMCPYASYNC) && (node.GetType() != IDENTITY)) {
    return SUCCESS;
  }
  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
  auto kernel_def = task_def.mutable_memcpy_async();
  kernel_def->set_op_index(node.GetOpDesc()->GetId());
  kernel_def->set_kind(RT_MEMCPY_ADDR_DEVICE_TO_DEVICE);
  uint8_t *membase =  run_context.dataMemBase;
  kernel_def->set_src((uintptr_t)membase + node.GetOpDesc()->GetInputOffset()[0]);
  kernel_def->set_dst((uintptr_t)membase + node.GetOpDesc()->GetOutputOffset()[0]);
  tasks.emplace_back(task_def);
  return SUCCESS;
}
void ConstructCustomInputOutputTensor(size_t input_num, size_t output_num,
                                      std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs) {
  for (size_t i = 0; i < input_num; i++) {
    std::vector<float32_t> input_data(2 * 2 * 2 , 0);
    TensorDesc desc(Shape({2, 2, 2}));
    ge::Tensor input_tensor{desc};
    input_tensor.SetData(reinterpret_cast<uint8_t *>(input_data.data()), input_data.size() * sizeof(float32_t));
    inputs.emplace_back(input_tensor);
  }

  for (size_t i = 0; i < output_num; ++i) {
    std::vector<uint8_t> output_data_1(32, 0xff);
    TensorDesc output_desc_1(Shape({2, 2, 2}));
    ge::Tensor output_tensor_1{output_desc_1};
    output_tensor_1.SetData(output_data_1.data(), output_data_1.size());
    outputs.emplace_back(output_tensor_1);
  }
  return;
}
void MockGenerateTask() {
  auto aicore_func = [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    if (node.GetType() == CONSTANT) {
      return SUCCESS;
    }

    auto op_desc = node.GetOpDesc();
    op_desc->SetOpKernelLibName("AiCoreLib");
    ge::AttrUtils::SetStr(op_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_KERNEL_BIN_ID, op_desc->GetName() + "_fake_id");
    const char kernel_bin[] = "kernel_bin";
    vector<char> buffer(kernel_bin, kernel_bin + strlen(kernel_bin));
    ge::OpKernelBinPtr kernel_bin_ptr = std::make_shared<ge::OpKernelBin>("test", std::move(buffer));
    op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, kernel_bin_ptr);
    size_t arg_size = 100;
    std::vector<uint8_t> args(arg_size, 0);
    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    auto kernel_info = task_def.mutable_kernel();
    kernel_info->set_args(args.data(), args.size());
    kernel_info->set_args_size(arg_size);
    kernel_info->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    kernel_info->set_kernel_name(node.GetName());
    kernel_info->set_block_dim(1);
    uint16_t args_offset[2] = {0};
    kernel_info->mutable_context()->set_args_offset(args_offset, 2 * sizeof(uint16_t));
    kernel_info->mutable_context()->set_op_index(node.GetOpDesc()->GetId());

    tasks.emplace_back(task_def);
    return SUCCESS;
  };

  auto rts_func = [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    return SUCCESS;
  };

  MockForGenerateTask("AiCoreLib", aicore_func);
  MockForGenerateTask("RTSLib", rts_func);
}
void *output_addr = nullptr;
void **args_table = nullptr;
}

class CustomOpRefreshTest : public testing::Test {
protected:
  void SetUp() {
    ModelManager::GetInstance().ClearAicpuSo();
    MockGenerateTask();
  }
  void TearDown() {
    OpsKernelBuilderRegistry::GetInstance().Unregister("AiCoreLib");
    OpsKernelBuilderRegistry::GetInstance().Unregister("RTSLib");
  }
};

class TestBaseCustomOp : public EagerExecuteOp {
public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    auto input_tensor0 = ctx->GetInputTensor(0);
    GE_ASSERT_NOTNULL(input_tensor0);
    auto input_shape0 = input_tensor0->GetShape().GetStorageShape();
    std::cout << "intput shape dimnum " << input_shape0.GetDimNum() << std::endl;
    GE_ASSERT_TRUE(input_shape0.GetDimNum() == 3);
    GE_ASSERT_TRUE(input_shape0.GetDim(0) == 2);
    auto input_tensor1 = ctx->GetInputTensor(1);
    GE_ASSERT_NOTNULL(input_tensor1);
    auto input_shape1 = input_tensor1->GetShape().GetStorageShape();
    GE_ASSERT_TRUE(input_shape1.GetDimNum() == 3);
    GE_ASSERT_TRUE(input_shape1.GetDim(0) == 2);
    auto input_tensor2 = ctx->GetInputTensor(2);
    GE_ASSERT_NOTNULL(input_tensor2);
    auto input_shape2 = input_tensor2->GetShape().GetStorageShape();
    GE_ASSERT_TRUE(input_shape2.GetDimNum() == 3);
    GE_ASSERT_TRUE(input_shape2.GetDim(0) == 2);

    // allocator 申请workspace有问题，taskinfo传入的是MemoryBlockManager但是在eager_op_execution_context里是按照GertAllocator来使用的
    auto workspaces = ctx->MallocWorkSpace(1024);
    GE_ASSERT_NOTNULL(workspaces);

    auto output_tensor = ctx->MallocOutputTensor(0, gert::StorageShape({2, 2 ,2}, {2, 2, 2}),
        gert::StorageFormat(FORMAT_ND, FORMAT_ND, ExpandDimsType()), DT_FLOAT);
    GE_ASSERT_NOTNULL(output_tensor);
    auto output_shape = output_tensor->GetShape().GetStorageShape();
    GE_ASSERT_TRUE(output_shape.GetDimNum() == 3);
    GE_ASSERT_TRUE(output_shape.GetDim(0) == 2);
    output_addr = output_tensor->GetAddr();
    GE_ASSERT_NOTNULL(output_addr);

    rtSetTaskTag("custom_op");
    void *input_0 = const_cast<void*>(ctx->GetInputTensor(0)->GetAddr());
    void *input_1 = const_cast<void*>(ctx->GetInputTensor(1)->GetAddr());
    void *input_2 = const_cast<void*>(ctx->GetInputTensor(2)->GetAddr());
    void *output_0 = const_cast<void*>(ctx->GetOutputTensor(0)->GetAddr());
    args_table[0] = static_cast<void*>(input_0);
    args_table[1] = static_cast<void*>(input_1);
    args_table[2] = static_cast<void*>(input_2);
    args_table[3] = static_cast<void*>(output_0);

    rtsLaunchKernelWithHostArgs(nullptr, 0, nullptr, nullptr, &args_table[0], 32, nullptr, 0);
    return SUCCESS;
  }
};

/**
 * 用例描述：fm外部设置，fm地址段不支持刷新，单次执行模型，args table正确
 *
 * 预置条件：
 * 1.构造计算图1，自定义算子直联Data和输出
 *  data0  data1  data2
 *     \    |      /
 *     \    |     /
 *       customop
 *          |
 *          |
 *       netoutput
 *
 * 测试步骤
 * 1.构造单个计算图1，设置fm地址段
 * 2.编译后执行计算图1
 * 3.判断argstable的一致性和正确性及args更新策略
 * 预期结果
 * 1.argstable的一致性和正确性均为成功，custom_op的args table为Execute流程分配，不走model args table的统一更新
 * 2.从dump图看产生了MemcpyAsyncTaskInfo
 */
TEST_F(CustomOpRefreshTest, model_execute_ok_with_customop_link_to_data) {
  MockForGenerateTask("DNN_VM_CUSTOM_OP_STORE", GenerateTaskForCustomOp);
  MockForGenerateTask("RTSLib", GenerateTaskForMemCopyAync);
  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;
  args_table = new void*[4];

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  auto compute_graph = ShareGraph::BuildOnlyCustomOpKnowShapeGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  CustomOpFactory::RegisterCustomOpCreator("CustomOp", []()->std::unique_ptr<BaseCustomOp> {
    return std::make_unique<TestBaseCustomOp>();
  });

  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    const auto input_shape = context->GetInputShape(0U);
    auto output = context->GetOutputShape(0);
    for (size_t dim = 0UL; dim < input_shape->GetDimNum(); dim++) {
      output->AppendDim(input_shape->GetDim(dim));
    }
    output->SetDimNum(input_shape->GetDimNum());
    return GRAPH_SUCCESS;
  };
  const auto infer_data_type_func = [](gert::InferDataTypeContext *context) -> graphStatus {
    const auto date_type = context->GetInputDataType(0U);
    EXPECT_EQ(context->SetOutputDataType(0, date_type), SUCCESS);
    return GRAPH_SUCCESS;
  };
  const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus {
    auto input_shape_range = context->GetInputShapeRange(0U);
    auto output_shape_range = context->GetOutputShapeRange(0U);
    output_shape_range->SetMin(const_cast<gert::Shape *>(input_shape_range->GetMin()));
    output_shape_range->SetMax(const_cast<gert::Shape *>(input_shape_range->GetMax()));
    return GRAPH_SUCCESS;
  };

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2(true);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("CustomOp");

  op_impl_func->infer_shape = infer_shape_func;
  op_impl_func->infer_datatype = infer_data_type_func;
  op_impl_func->infer_shape_range = infer_shape_range_func;
  op_impl_func->output_shape_depend_compute = 1UL;

  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));

  std::vector<std::pair<uint32_t, uint32_t>> io_indexes;
  EXPECT_EQ(summary->GetIOIndexesWithSameAddr(io_indexes), SUCCESS);
  EXPECT_EQ(io_indexes.size(), 0U);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructCustomInputOutputTensor(3, 1, inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  CHECK_GRAPH(PreRunAfterBuild) {
    args_checker = std::make_unique<ArgsChecker>(graph, graph_id, session.GetSessionId(), runtime_stub);
  };

  EXPECT_EQ(SUCCESS, args_checker->SetFmAddr((uint64_t)feature_mem.data(), feature_size));
  EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0, 1, 2}, inputs));
  EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs));
  EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());
  EXPECT_EQ(SUCCESS, args_checker->CheckNodesArgsNotUpdated({"custom_op"}));

  delete [] args_table;
  runtime_stub.Clear();
  mmSetEnv(kEnvValue, "", 1);
  ReInitGe();
}

/**
 * 用例描述：fm外部设置，fm地址段不支持刷新，单次执行模型，args table正确
 *
 * 预置条件：
 * 1.构造计算图1，自定义算子不在模型边界
 *
 *  data0  data1    data2  data3    data4  data5
 *     \    |         \     /          /    /
 *     \    |         \   /          /     /
 *         add0       add1          add2
 *              \       |         /
 *                \     |       /
 *                  customop           data6
 *                    |             /
 *                     |          /
 *                       add3
 *                        |
 *                      netoutput
 *
 *
 * 测试步骤
 * 1.构造单个计算图1，设置fm地址段
 * 2.编译后执行计算图1
 * 3.判断argstable的一致性和正确性及args更新策略
 * 预期结果
 * 1.argstable的一致性和正确性均为成功，custom_op的args table为Execute流程分配，不走model args table的统一更新
 * 2.从dump图看产生了未插入MemcpyAsyncTaskInfo
 */
TEST_F(CustomOpRefreshTest, model_execute_ok_with_customop_link_to_add) {
  MockForGenerateTask("DNN_VM_CUSTOM_OP_STORE", GenerateTaskForCustomOp);
  MockForGenerateTask("RTSLib", GenerateTaskForMemCopyAync);
  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;
  args_table = new void*[4];

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  auto compute_graph = ShareGraph::BuildCustomOpWithAddKnowShapeGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  CustomOpFactory::RegisterCustomOpCreator("CustomOp", []()->std::unique_ptr<BaseCustomOp> {
    return std::make_unique<TestBaseCustomOp>();
  });

 const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    const auto input_shape = context->GetInputShape(0U);
    auto output = context->GetOutputShape(0);
    for (size_t dim = 0UL; dim < input_shape->GetDimNum(); dim++) {
      output->AppendDim(input_shape->GetDim(dim));
    }
    output->SetDimNum(input_shape->GetDimNum());
    return GRAPH_SUCCESS;
  };
  const auto infer_data_type_func = [](gert::InferDataTypeContext *context) -> graphStatus {
    const auto date_type = context->GetInputDataType(0U);
    EXPECT_EQ(context->SetOutputDataType(0, date_type), SUCCESS);
    return GRAPH_SUCCESS;
  };
  const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus {
    auto input_shape_range = context->GetInputShapeRange(0U);
    auto output_shape_range = context->GetOutputShapeRange(0U);
    output_shape_range->SetMin(const_cast<gert::Shape *>(input_shape_range->GetMin()));
    output_shape_range->SetMax(const_cast<gert::Shape *>(input_shape_range->GetMax()));
    return GRAPH_SUCCESS;
  };

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2(true);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("CustomOp");

  op_impl_func->infer_shape = infer_shape_func;
  op_impl_func->infer_datatype = infer_data_type_func;
  op_impl_func->infer_shape_range = infer_shape_range_func;
  op_impl_func->output_shape_depend_compute = 1UL;

  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));

  std::vector<std::pair<uint32_t, uint32_t>> io_indexes;
  EXPECT_EQ(summary->GetIOIndexesWithSameAddr(io_indexes), SUCCESS);
  EXPECT_EQ(io_indexes.size(), 0U);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructCustomInputOutputTensor(7, 1, inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  CHECK_GRAPH(PreRunAfterBuild) {
    args_checker = std::make_unique<ArgsChecker>(graph, graph_id, session.GetSessionId(), runtime_stub);
  };

  EXPECT_EQ(SUCCESS, args_checker->SetFmAddr((uint64_t)feature_mem.data(), feature_size));
  EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0, 1, 2, 3, 4, 5, 6}, inputs));
  EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs));
  EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());
  EXPECT_EQ(SUCCESS, args_checker->CheckNodesArgsNotUpdated({"custom_op"}));

  delete [] args_table;
  runtime_stub.Clear();
  mmSetEnv(kEnvValue, "", 1);
  ReInitGe();
}

/**
 * 用例描述：fm外部设置，fm地址段支持刷新，单次执行模型，args table正确
 *
 * 预置条件：
 * 1.构造计算图1，自定义算子不在模型边界
 *
 *  data0  data1    data2  data3    data4  data5
 *     \    |         \     /          /    /
 *     \    |         \   /          /     /
 *         add0       add1          add2
 *              \       |         /
 *                \     |       /
 *                  customop           data6
 *                    |             /
 *                     |          /
 *                       add3
 *                        |
 *                      netoutput
 *
 *
 * 测试步骤
 * 1.构造单个计算图1，设置fm地址段
 * 2.编译后执行计算图1
 * 3.判断argstable的一致性和正确性及args更新策略
 * 预期结果
 * 1.argstable的一致性和正确性均为成功，custom_op的args table为Execute流程分配，不走model args table的统一更新
 * 2.从dump图看插入MemcpyAsyncTaskInfo
 */
TEST_F(CustomOpRefreshTest, model_execute_ok_with_customop_link_to_add_and_fm_refresh) {
  MockForGenerateTask("DNN_VM_CUSTOM_OP_STORE", GenerateTaskForCustomOp);
  MockForGenerateTask("RTSLib", GenerateTaskForMemCopyAync);
  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;
  args_table = new void*[4];

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  auto compute_graph = ShareGraph::BuildCustomOpWithAddKnowShapeGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  CustomOpFactory::RegisterCustomOpCreator("CustomOp", []()->std::unique_ptr<BaseCustomOp> {
    return std::make_unique<TestBaseCustomOp>();
  });

 const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    const auto input_shape = context->GetInputShape(0U);
    auto output = context->GetOutputShape(0);
    for (size_t dim = 0UL; dim < input_shape->GetDimNum(); dim++) {
      output->AppendDim(input_shape->GetDim(dim));
    }
    output->SetDimNum(input_shape->GetDimNum());
    return GRAPH_SUCCESS;
  };
  const auto infer_data_type_func = [](gert::InferDataTypeContext *context) -> graphStatus {
    const auto date_type = context->GetInputDataType(0U);
    EXPECT_EQ(context->SetOutputDataType(0, date_type), SUCCESS);
    return GRAPH_SUCCESS;
  };
  const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus {
    auto input_shape_range = context->GetInputShapeRange(0U);
    auto output_shape_range = context->GetOutputShapeRange(0U);
    output_shape_range->SetMin(const_cast<gert::Shape *>(input_shape_range->GetMin()));
    output_shape_range->SetMax(const_cast<gert::Shape *>(input_shape_range->GetMax()));
    return GRAPH_SUCCESS;
  };

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2(true);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("CustomOp");

  op_impl_func->infer_shape = infer_shape_func;
  op_impl_func->infer_datatype = infer_data_type_func;
  op_impl_func->infer_shape_range = infer_shape_range_func;
  op_impl_func->output_shape_depend_compute = 1UL;

  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));

  std::vector<std::pair<uint32_t, uint32_t>> io_indexes;
  EXPECT_EQ(summary->GetIOIndexesWithSameAddr(io_indexes), SUCCESS);
  EXPECT_EQ(io_indexes.size(), 0U);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructCustomInputOutputTensor(7, 1, inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  CHECK_GRAPH(PreRunAfterBuild) {
    args_checker = std::make_unique<ArgsChecker>(graph, graph_id, session.GetSessionId(), runtime_stub);
  };

  EXPECT_EQ(SUCCESS, args_checker->SetFmAddr((uint64_t)feature_mem.data(), feature_size));
  EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0, 1, 2, 3, 4, 5, 6}, inputs));
  EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs));
  EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());
  EXPECT_EQ(SUCCESS, args_checker->CheckNodesArgsNotUpdated({"custom_op"}));

  delete [] args_table;
  runtime_stub.Clear();
  mmSetEnv(kEnvValue, "", 1);
  ReInitGe();
}

}
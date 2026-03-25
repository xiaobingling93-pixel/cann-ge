/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ge_running_env/fake_op.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "ge_common/ge_api_types.h"
#include "runtime/rt.h"
#include "framework/executor/ge_executor.h"
#include "framework/generator/ge_generator.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "single_op/single_op.h"
#include "single_op/single_op_manager.h"
#include "utils/model_data_builder.h"
#include "single_op/task/build_task_utils.h"
#include "single_op/task/tbe_task_builder.h"
#include "utils/tensor_descs.h"
#include "utils/data_buffers.h"
#include "register/op_tiling_registry.h"
#include "graph/debug/ge_attr_define.h"
#include "hybrid/node_executor/aicore/aicore_node_executor.h"
#include "hybrid/node_executor/ge_local/ge_local_node_executor.h"
#include "graph/manager/mem_manager.h"
#include "utils/bench_env.h"
#include "utils/graph_factory.h"
#include "hybrid/model/hybrid_model_builder.h"
#include "ge_running_env/fake_ops_kernel_builder.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/tensor_utils.h"
#include "aicpu_task_struct.h"
#include "hybrid/node_executor/aicpu/aicpu_ext_info_handler.h"
#include "ge/ge_ir_build.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/attr_utils.h"
#include "graph/tuning_utils.h"
#include "graph/build/model_data_info.h"
#include "base/common/helper/model_parser_base.h"
#include "framework/common/helper/model_helper.h"
#include "common/share_graph.h"
#include "graph/ge_global_options.h"
#include "tests/depends/mmpa/src/mmpa_stub.h"
#include "faker/space_registry_faker.h"
#include "init_ge.h"
#include "framework/common/host_resource_center/tiling_resource_manager.h"
#include "api/gelib/gelib.h"
#include "utils/mock_ops_kernel_builder.h"
#include "framework/common/taskdown_common.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace ge {
namespace {
constexpr const char *kFormatTime = "[2023-08-08-20:08:00.001.001]";
constexpr const char *kPlatformIrrelevant =
    "This model is irrelevant to the host platform, parameters about host os and host cpu are ignored.";
graphStatus StubInferFunction(Operator &op) { return GRAPH_SUCCESS; }

void MockGenerateTask() {
  auto aicore_func = [](const ge::Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
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

    for (int32_t i = 0; i < 3; i++) {
      tasks.emplace_back(task_def);
    }
    return SUCCESS;
  };

  MockForGenerateTask("AiCoreLib", aicore_func);
}
}  // namespace
class GeIrBuildTest : public testing::Test {
 protected:
  void SetUp() {
    MockGenerateTask();
  }
  void TearDown() {
    OpsKernelBuilderRegistry::GetInstance().Unregister("AiCoreLib");
    GeRunningEnvFaker env;
    env.InstallDefault();
  }
};

TEST_F(GeIrBuildTest, TestWeightRefreshableGraphs) {
  auto graph = GraphFactory::BuildRefreshWeight();
  WeightRefreshableGraphs weight_refreshable_graphs;
  std::vector<AscendString> lora_names;
  lora_names.emplace_back(AscendString("const1"));
  lora_names.emplace_back(AscendString("const2"));
  auto ret = aclgrphConvertToWeightRefreshableGraphs(graph, lora_names, weight_refreshable_graphs);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  auto infer_compute_graph = ge::GraphUtilsEx::GetComputeGraph(weight_refreshable_graphs.infer_graph);
  EXPECT_NE(infer_compute_graph, nullptr);
  auto var1_node = infer_compute_graph->FindNode("const1_var");
  EXPECT_NE(var1_node, nullptr);
  auto dt = var1_node->GetOpDesc()->MutableOutputDesc(0)->GetDataType();
  EXPECT_EQ(dt, DT_FLOAT);
  auto format = var1_node->GetOpDesc()->MutableOutputDesc(0)->GetFormat();
  EXPECT_EQ(format, FORMAT_ND);

  std::vector<GraphWithOptions> graph_and_options;
  std::map<AscendString, AscendString> options;
  options.insert({AscendString("input_format"), AscendString("ND")});
  ModelBufferData model;
  ret = aclgrphBundleBuildModel(graph_and_options, model);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
  graph_and_options.emplace_back(GraphWithOptions{weight_refreshable_graphs.infer_graph, options});
  graph_and_options.emplace_back(GraphWithOptions{weight_refreshable_graphs.var_init_graph, options});
  graph_and_options.emplace_back(GraphWithOptions{weight_refreshable_graphs.var_update_graph, options});
  ret = aclgrphBundleBuildModel(graph_and_options, model);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  infer_compute_graph = ge::GraphUtilsEx::GetComputeGraph(weight_refreshable_graphs.infer_graph);
  EXPECT_NE(infer_compute_graph, nullptr);
  var1_node = infer_compute_graph->FindNode("const1_var");
  EXPECT_NE(var1_node, nullptr);
  dt = var1_node->GetOpDesc()->MutableOutputDesc(0)->GetDataType();
  EXPECT_EQ(dt, DT_FLOAT16);
  format = var1_node->GetOpDesc()->MutableOutputDesc(0)->GetFormat();
  EXPECT_EQ(format, FORMAT_FRACTAL_NZ);

  auto var_init_compute_graph = ge::GraphUtilsEx::GetComputeGraph(weight_refreshable_graphs.var_init_graph);
  EXPECT_NE(infer_compute_graph, nullptr);
  var1_node = infer_compute_graph->FindNode("const1_var");
  EXPECT_NE(var1_node, nullptr);
  dt = var1_node->GetOpDesc()->MutableOutputDesc(0)->GetDataType();
  EXPECT_EQ(dt, DT_FLOAT16);
  format = var1_node->GetOpDesc()->MutableOutputDesc(0)->GetFormat();
  EXPECT_EQ(format, FORMAT_FRACTAL_NZ);

  auto var_update_compute_graph = ge::GraphUtilsEx::GetComputeGraph(weight_refreshable_graphs.var_update_graph);
  EXPECT_NE(infer_compute_graph, nullptr);
  var1_node = infer_compute_graph->FindNode("const1_var");
  EXPECT_NE(var1_node, nullptr);
  dt = var1_node->GetOpDesc()->MutableOutputDesc(0)->GetDataType();
  EXPECT_EQ(dt, DT_FLOAT16);
  format = var1_node->GetOpDesc()->MutableOutputDesc(0)->GetFormat();
  EXPECT_EQ(format, FORMAT_FRACTAL_NZ);

  ret = aclgrphBundleSaveModel("./test", model);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(GeIrBuildTest, TestBuildModel) {
  BenchEnv::Init();
  std::shared_ptr<GELib> ge_lib = GELib::GetInstance();
  if (ge_lib != nullptr) {
    ge_lib->init_flag_=false;
  }
  std::map<AscendString, AscendString> init_options;
  init_options.emplace(ge::OPTION_EXEC_HCCL_FLAG, "0");
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  std::string res;
  GetThreadLocalContext().GetOption(JIT_COMPILE, res);
  EXPECT_EQ(res, "1");

  std::map<AscendString, AscendString> init_options1;
  init_options1.emplace("ge.autoTuneMode", "RA");
  EXPECT_NE(aclgrphBuildInitialize(init_options1), SUCCESS);

  setenv("ASCEND_OPP_PATH", "./", 0);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<AscendString, AscendString> build_options;
  build_options.emplace(ge::ir_option::INPUT_FORMAT, "NCHW");
  ModelBufferData model_buffer_data{};
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  std::string output_file = "saved_model.om";
  EXPECT_EQ(aclgrphSaveModel(output_file, model_buffer_data), SUCCESS);
  EXPECT_EQ(aclgrphSaveModel(output_file.c_str(), model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestExternalWeightCombined) {
  BenchEnv::Init();
  std::shared_ptr<GELib> ge_lib = GELib::GetInstance();
  if (ge_lib != nullptr) {
    ge_lib->init_flag_=false;
  }
  std::map<std::string, std::string> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  setenv("ASCEND_OPP_PATH", "./", 0);

  auto graph = GraphFactory::BuildRefreshWeight();
  init_options.emplace(ge::EXTERNAL_WEIGHT, "2");
  ModelBufferData model_buffer_data{};
  EXPECT_EQ(aclgrphBuildModel(graph, init_options, model_buffer_data), SUCCESS);
  std::string output_file = "./saved_model";
  EXPECT_EQ(aclgrphSaveModel(output_file, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelFailWtihL1OptimizeInVirtual) {
  BenchEnv::Init();
  std::map<AscendString, AscendString> init_options;
  init_options.emplace(ge::OPTION_EXEC_HCCL_FLAG, "0");
  init_options.emplace(ge::configure_option::VIRTUAL_TYPE, "1");
  init_options.emplace(ge::configure_option::BUFFER_OPTIMIZE, "l1_optimize");
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  setenv("ASCEND_OPP_PATH", "./", 0);

  auto graph = GraphFactory::SingeOpGraph2();
  (void)AttrUtils::SetBool(GraphUtilsEx::GetComputeGraph(graph), ATTR_NAME_OFF_SUPERKERNEL_ATTR, false);
  std::map<AscendString, AscendString> build_options;
  build_options.emplace(ge::ir_option::INPUT_FORMAT, "NCHW");

  ModelBufferData model_buffer_data{};
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

//TEST_F(GeIrBuildTest, TestInferShapePrepare) {
//  auto graph = GraphFactory::SingeOpGraph2();
//  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
//  EXPECT_TRUE(compute_graph != nullptr);
//}

TEST_F(GeIrBuildTest, test_build_and_bundlesave_flow_model_pp) {
  std::map<AscendString, AscendString> init_options;
  init_options[ge::OPTION_HOST_ENV_OS] = "linux";
  init_options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::VariableAddGraph();
  std::map<string, string> build_options = {{"input_shape", "1,1,-1,-1"}};
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(op_desc, "_static_to_dynamic_softsync_op", true);
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);

  ModelBufferData bundle_buffer;
  EXPECT_EQ(ModelHelper::SaveBundleModelBufferToMem({model_buffer_data, model_buffer_data}, 2048, bundle_buffer), SUCCESS);
  EXPECT_NE(aclgrphBundleSaveModel(nullptr, bundle_buffer), SUCCESS);
  EXPECT_EQ(aclgrphBundleSaveModel("bundle_flow_model", bundle_buffer), SUCCESS);
  EXPECT_NE(aclgrphSaveModel("./test1", bundle_buffer), SUCCESS); // type not support

  aclgrphBuildFinalize();
  system("rm -f ./bundle_flow_model.om");
}

TEST_F(GeIrBuildTest, test_build_and_bundle_save_variable_modle) {
  std::map<AscendString, AscendString> init_options;
  init_options[ge::OPTION_HOST_ENV_OS] = "linux";
  init_options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::VariableAddGraph();
  std::map<string, string> build_options = {{"input_shape", "1,1,-1,-1"}};
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(op_desc, "_static_to_dynamic_softsync_op", true);
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);

  ModelBufferData bundle_buffer;
  EXPECT_EQ(ModelHelper::SaveBundleModelBufferToMem({model_buffer_data, model_buffer_data}, 4096, bundle_buffer), SUCCESS);
  EXPECT_NE(aclgrphBundleSaveModel(nullptr, bundle_buffer), SUCCESS);
  EXPECT_EQ(aclgrphBundleSaveModel("bundle_var_model", bundle_buffer), SUCCESS);

  // load bundle model
  std::ifstream fs("./bundle_var_model_linux_x86_64.om", std::ifstream::binary);
  GE_MAKE_GUARD(free, [&fs]() {
    fs.close();
    system("rm -f ./bundle_var_model_linux_x86_64.om");
  });
  ASSERT_TRUE(fs.is_open());
  fs.seekg(0, std::ifstream::end);
  const uint64_t len = static_cast<uint64_t>(fs.tellg());
  fs.seekg(0, std::ifstream::beg);
  auto data_bin = std::unique_ptr<char_t[]>(new (std::nothrow) char_t[len]);
  fs.read(data_bin.get(), static_cast<std::streamsize>(len));
  ModelData data;
  data.model_data = data_bin.get();
  data.model_len = len;

  const ModelFileHeader *bundle_header{nullptr};
  ModelHelper::GetModelFileHead(data, bundle_header);
  ASSERT_NE(bundle_header, nullptr);
  EXPECT_EQ(bundle_header->modeltype, MODEL_TYPE_BUNDLE_MODEL);
  EXPECT_EQ(bundle_header->model_length, len);
  EXPECT_EQ(bundle_header->model_num, 2U);
  ModelPartitionTable *table = reinterpret_cast<ModelPartitionTable*>(&data_bin[sizeof(ModelFileHeader)]);
  ASSERT_NE(table, nullptr);
  EXPECT_TRUE(table->num == 3);
  size_t offset = sizeof(ModelPartitionTable) + sizeof(ModelPartitionMemInfo) * table->num;
  EXPECT_TRUE(table->partition[0].type == BUNDLE_MODEL_VAR_INFO);
  EXPECT_TRUE(table->partition[0].mem_size == sizeof(int64_t));
  EXPECT_TRUE(table->partition[0].mem_offset == offset);
  EXPECT_TRUE(table->partition[1].type == BUNDLE_MODEL_INFO);
  offset += table->partition[0].mem_size;
  EXPECT_TRUE(table->partition[1].mem_offset == offset);
  EXPECT_TRUE(table->partition[2].type == BUNDLE_MODEL_INFO);
  offset += table->partition[1].mem_size;
  EXPECT_TRUE(table->partition[2].mem_offset == offset);
  offset += table->partition[2].mem_size;
  EXPECT_EQ(offset + sizeof(ModelFileHeader), len);
  size_t data_offset = sizeof(ModelFileHeader) + sizeof(ModelPartitionTable) + sizeof(ModelPartitionMemInfo) * 3;
  EXPECT_EQ(*reinterpret_cast<int64_t *>(data_bin.get() + data_offset), 4096);
  for (uint32_t i = 1U; i < 3;++i) {
    ModelData sub_model;
    sub_model.model_data = data_bin.get() + sizeof(ModelFileHeader) + table->partition[i].mem_offset;
    sub_model.model_len = table->partition[i].mem_size;
    const ModelFileHeader *sub_header{nullptr};
    ModelHelper::GetModelFileHead(sub_model, sub_header);
    ASSERT_NE(sub_header, nullptr);
    EXPECT_EQ(sub_header->modeltype, MODEL_TYPE_IR_MODEL);
  }

  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, test_get_model_file_head_failed) {
  auto len = 10U;
  auto data_bin = std::unique_ptr<char_t[]>(new (std::nothrow) char_t[len]);
  ModelData data;
  data.model_data = nullptr;

  const ModelFileHeader *bundle_header{nullptr};
  EXPECT_EQ(ModelHelper::GetModelFileHead(data, bundle_header), ACL_ERROR_GE_PARAM_INVALID);

  data.model_data = data_bin.get();
  data.model_len = 0;
  EXPECT_EQ(ModelHelper::GetModelFileHead(data, bundle_header), ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);
}

TEST_F(GeIrBuildTest, TestGenerateForOp) {
  BenchEnv::Init();
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  Graph graph;
  Shape shape({1, 16});
  TensorDesc input_tensor_desc(shape);
  input_tensor_desc.SetConstData(MakeUnique<uint8_t[]>(64), 64);
  TensorDesc output_tensor_desc(shape);
  EXPECT_EQ(aclgrphGenerateForOp(NEG, {input_tensor_desc}, {output_tensor_desc},  graph), SUCCESS);
}

TEST_F(GeIrBuildTest, TestGenerateFromJsonForOp) {
  BenchEnv::Init();
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  ge::AscendString json("st_run_data/json/single_op/add_op.json");
  std::vector<Graph> graphs;
  auto ret = aclgrphGenerateForOp(json, graphs);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(graphs.size(), 8);
}

TEST_F(GeIrBuildTest, TestBuildOptions) {
  std::map<std::string, std::string> init_options;

  init_options[ge::ir_option::SPARSITY] = "1";
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  init_options[ge::ir_option::ALLOW_HF32] = "true";
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);
  init_options[ge::ir_option::ALLOW_HF32] = "1";
  EXPECT_EQ(aclgrphBuildInitialize(init_options), GRAPH_PARAM_INVALID);

  init_options[ge::ir_option::ENABLE_COMPRESS_WEIGHT] = "true";
  init_options[ge::ir_option::COMPRESS_WEIGHT_CONF] = "./";
  EXPECT_EQ(aclgrphBuildInitialize(init_options), GRAPH_PARAM_INVALID);

  init_options[ge::ir_option::ENABLE_COMPRESS_WEIGHT] = "yes";
  init_options[ge::ir_option::COMPRESS_WEIGHT_CONF] = "./";
  EXPECT_EQ(aclgrphBuildInitialize(init_options), GRAPH_PARAM_INVALID);

  init_options.clear();
  init_options[ge::OPTION_HOST_ENV_OS] = "linux";
  init_options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  EXPECT_EQ(aclgrphBuildInitialize(init_options), GRAPH_SUCCESS);

  init_options.clear();
  init_options[INPUT_HINT_SHAPE] = "0:[3];1:[3]";
  EXPECT_NE(aclgrphBuildInitialize(init_options), SUCCESS);

  init_options["ge.optionInvalid"] = "invalid";
  aclgrphBuildInitialize(init_options);

}

TEST_F(GeIrBuildTest, TestDumpGraph) {
  auto graph = GraphFactory::SingeOpGraph2();
  std::string file_path = "dump.bin";
  EXPECT_EQ(aclgrphDumpGraph(graph, file_path.c_str(), file_path.length()), ge::GRAPH_SUCCESS);
}

TEST_F(GeIrBuildTest, TestSetOpAttr) {
  auto graph = GraphFactory::SingeOpGraph2();

  // error attr type
  EXPECT_EQ(aclgrphSetOpAttr(graph, aclgrphAttrType(-1), "./"), GRAPH_FAILED);
  // empty config
  EXPECT_EQ(aclgrphSetOpAttr(graph, ATTR_TYPE_KEEP_DTYPE, nullptr), GRAPH_SUCCESS);

  // TODO config file
}

TEST_F(GeIrBuildTest, TestBuildModelWithShapeRange) {
  std::map<AscendString, AscendString> init_options;
  init_options[ge::OPTION_HOST_ENV_OS] = "linux";
  init_options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape", "1,1,-1,-1"}
  };
  ModelBufferData model_buffer_data{};
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), GRAPH_FAILED);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, -1);
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), GRAPH_FAILED);

  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(op_desc, "_static_to_dynamic_softsync_op", true);
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  std::string output_file = "saved_model";
  EXPECT_EQ(aclgrphSaveModel(output_file, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}


TEST_F(GeIrBuildTest, TestBuildModelWithShapeRangeInvalidInput) {
  std::map<AscendString, AscendString> init_options;
  init_options[ge::OPTION_HOST_ENV_OS] = "linux";
  init_options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape", "data,1,1,-1,-1"}
  };
  ModelBufferData model_buffer_data{};

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, -1);
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(op_desc, "_static_to_dynamic_softsync_op", true);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithNamedShapeRangeDynamic) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  std::string range_str = data_node->GetName() + ":1,1,222~226,-1";
  std::map<string, string> build_options = {
      {ge::ir_option::INPUT_SHAPE, range_str}
  };

  ModelBufferData model_buffer_data{};
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithNamedShapeRangeDynamic_InvalidInputShape) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  std::string range_str = data_node->GetName() + ":1,1,222~226,-1";
  std::map<string, string> build_options = {
      {ge::ir_option::INPUT_SHAPE, range_str},
      {"ge.dynamicBatchSize", "1,1,225,8"}
  };

  ModelBufferData model_buffer_data{};
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithShapeRange_invalid_param1) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape_range", "[1~-1, 1, -1, -1]"}
  };
  ModelBufferData model_buffer_data{};

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), GRAPH_FAILED);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithShapeRange_invalid_param2) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape_range", "[1~2~3, 1, -1, -1]"}
  };
  ModelBufferData model_buffer_data{};

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), GRAPH_FAILED);

  std::map<string, string> build_options1 = {{"ge.autoTuneMode", "RA"}};
  EXPECT_EQ(aclgrphBuildModel(graph, build_options1, model_buffer_data), GRAPH_FAILED);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithShapeRange_invalid_param3) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape_range", "[1]"}
  };
  ModelBufferData model_buffer_data{};

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), GRAPH_FAILED);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithNamedShapeRange) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  std::string range_str = data_node->GetName() + ":[1, 1, -1, -1]";
  string input_shape_str = data_node->GetName() + ":1,1,244,244";
  std::map<string, string> build_options = {
      {ge::ir_option::INPUT_SHAPE_RANGE, range_str},
      {ge::ir_option::INPUT_SHAPE, input_shape_str}
  };

  ModelBufferData model_buffer_data{};
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestGetIRVersion) {
  int32_t major_version = 0;
  int32_t minor_version = 0;
  int32_t patch_version = 0;
  EXPECT_EQ(aclgrphGetIRVersion(&major_version, &minor_version, &patch_version), SUCCESS);
}

TEST_F(GeIrBuildTest, TestBuildModel_buffer_optimize_invalid) {
  std::map<AscendString, AscendString> init_options = {
      {ge::ir_option::BUFFER_OPTIMIZE, "invalid"}
  };
  EXPECT_NE(aclgrphBuildInitialize(init_options), SUCCESS);
}

// TEST_F(GeIrBuildTest, TestBuildModel_insert_op_invalid) {
//   std::map<AscendString, AscendString> init_options = {
//       {ge::ir_option::INSERT_OP_FILE, "invalid"}
//   };
//   EXPECT_NE(aclgrphBuildInitialize(init_options), SUCCESS);
// }

TEST_F(GeIrBuildTest, TestBuildModel_reuse_memory_invalid) {
  std::map<AscendString, AscendString> init_options = {
      {ge::ir_option::EXEC_DISABLE_REUSED_MEMORY, "invalid"}
  };
  EXPECT_NE(aclgrphBuildInitialize(init_options), SUCCESS);
}

TEST_F(GeIrBuildTest, TestBuildModel_single_stream_invalid) {
  std::map<AscendString, AscendString> init_options = {
      {ge::ir_option::ENABLE_SINGLE_STREAM, "invalid"}
  };
  EXPECT_NE(aclgrphBuildInitialize(init_options), SUCCESS);
}

TEST_F(GeIrBuildTest, TestBuildModel_external_weight_invalid) {
  std::map<AscendString, AscendString> init_options = {
      {ge::ir_option::EXTERNAL_WEIGHT, "invalid"}
  };
  EXPECT_NE(aclgrphBuildInitialize(init_options), SUCCESS);
}

TEST_F(GeIrBuildTest, TestBuildModel_ac_parallel_enable_invalid) {
  std::map<AscendString, AscendString> init_options = {
      {ge::ir_option::AC_PARALLEL_ENABLE, "invalid"}
  };
  EXPECT_NE(aclgrphBuildInitialize(init_options), SUCCESS);
}

TEST_F(GeIrBuildTest, TestBuildModel_tiling_schedule_optimize_invalid) {
  std::map<AscendString, AscendString> init_options = {
      {ge::ir_option::TILING_SCHEDULE_OPTIMIZE, "invalid"}
  };
  EXPECT_NE(aclgrphBuildInitialize(init_options), SUCCESS);
}

TEST_F(GeIrBuildTest, TestBuildModel_quant_dumpable_invalid) {
  std::map<AscendString, AscendString> init_options = {
      {ge::ir_option::QUANT_DUMPABLE, "invalid"}
  };
  EXPECT_NE(aclgrphBuildInitialize(init_options), SUCCESS);
}

TEST_F(GeIrBuildTest, TestBuildModelWithDynamicBatch) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape", "data1:-1,3,16,16"},
      {"ge.dynamicBatchSize", "1,2,4,8,"}
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithDynamicBatch_invalid1) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"ge.dynamicBatchSize", "1,2,4,8"}
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithDynamicBatch_invalid2) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape", "data1:1,3,16,16"},
      {"ge.dynamicBatchSize", "1,2,4,8"}
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithDynamicBatch_invalid3) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape", "data1:1,3,16,16"},
      {"ge.dynamicBatchSize", "a,2,4,8"}
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithInvalidDynamicBatch_Fail) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape", "data1:-1,3,16,16"},
      {"ge.dynamicBatchSize", "a,2,4,8"}
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

/*
 * -------------------------------------------------
 *             data
 *              |
 *            netouput
 * ------------------------------------------------
 * 测试步骤
 * 1.构造单个计算图1，data和netputput直连
 * 2.ir接口初始化及编译图
 * 预期结果
 * 1.图1·编译成功，PreRunAfterOptimize2后图内有三个节点（data/memcpy/netoutput）
 */
TEST_F(GeIrBuildTest, TestBuildModelDataToNetoutput) {
  // 设置环境变量
  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::GraphDataToNetoutput();
  std::map<string, string> build_options = {};
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  DUMP_GRAPH_WHEN("PreRunAfterOptimize2")
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
  mmSetEnv(kEnvValue, "", 1);

  CHECK_GRAPH(PreRunAfterOptimize2) {
    EXPECT_EQ(graph->GetDirectNodesSize(), 3);
  };
}

TEST_F(GeIrBuildTest, TestBuildModelWithDynamicImage) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape", "data1:1,3,-1,-1"},
      {ge::ir_option::INPUT_FORMAT, "NCHW"},
      {"ge.dynamicImageSize", "16,16;32,32"}
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithDynamicImage_invalid1) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape", "data1:1,3,-1,-1"},
      {"ge.dynamicImageSize", "16,16;32,32"}
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithDynamicImage_invalid2) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape", "data1:1,3,-1,-1, 1"},
      {ge::ir_option::INPUT_FORMAT, "NCHW"},
      {"ge.dynamicImageSize", "16,16;32,32"}
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithDynamicImage_invalid3) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape", "data1:1,3,16,16"},
      {ge::ir_option::INPUT_FORMAT, "NCHW"},
      {"ge.dynamicImageSize", "16,16;32,32"}
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithDynamicImage_invalid4) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape", "data1:1,3,-1,-1"},
      {ge::ir_option::INPUT_FORMAT, "NCHW"},
      {"ge.dynamicImageSize", "a,16;32,32"}
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithDynamicImage_invalid5) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape", "data1:1,3,-1,-1"},
      {ge::ir_option::INPUT_FORMAT, "NCHW"},
      {"ge.dynamicImageSize", "16,16,16;32,32"}
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithDynamicDims) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape", "data1:1,-1,-1,-1"},
      {ge::ir_option::INPUT_FORMAT, "ND"},
      {"ge.dynamicDims", "3,16,16;3,32,32"}
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithDynamicDims_invalid1) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape", "data1:1,-1,-1,-1"},
      {ge::ir_option::INPUT_FORMAT, "NCHW"},
      {"ge.dynamicDims", "3,16,16;3,32,32"}
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithDynamicDims_invalid2) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape", "data1:1,-1,-1,-1,-1"},
      {ge::ir_option::INPUT_FORMAT, "ND"},
      {"ge.dynamicDims", "3,16,16;3,32,32"}
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithDynamicDims_invalid3) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape", "data1:1,3,16,16"},
      {ge::ir_option::INPUT_FORMAT, "ND"},
      {"ge.dynamicDims", "3,16,16;3,32,32"}
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithDynamicDims_invalid4) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape", "data1:1,-1,-1,-1"},
      {ge::ir_option::INPUT_FORMAT, "ND"},
      {"ge.dynamicDims", ""}
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithDynamicDims_invalid5) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape", "data1:1,-1,-1,-1"},
      {ge::ir_option::INPUT_FORMAT, "ND"},
      {"ge.dynamicDims", "3,16,16,16;3,32,32"}
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithDynamicDims_invalid6) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape", "data1:1,-1,-1,-1"},
      {ge::ir_option::INPUT_FORMAT, "ND"},
      {"ge.dynamicDims", "a,16,16;3,32,32"}
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithDynamicDims_invalid7) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {ge::ir_option::INPUT_FORMAT, "ND"},
      {"ge.dynamicDims", "a,16,16;3,32,32"}
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithDynamic_multi_invalid) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape", "data1:1,3,16,16"},
      {ge::ir_option::INPUT_FORMAT, "ND"},
      {"ge.dynamicImageSize", "16,16;32,32"},
      {"ge.dynamicDims", "a,16,16;3,32,32"}
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModel_input_format_invalid) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {ge::ir_option::INPUT_FORMAT, "NDND"},
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModel_build_mode_invalid) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {BUILD_MODE, "invalid"},
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModel_build_step_invalid) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {BUILD_STEP, "invalid"},
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModel_build_mode_lead_step) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {BUILD_MODE, BUILD_MODE_TUNING},
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModel_not_support_option) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"invalid", "invalid"},
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModel_shape_generalized_build_mode_invalid) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {ge::ir_option::SHAPE_GENERALIZED_BUILD_MODE, "invalid"},
  };
  ModelBufferData model_buffer_data{};
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  EXPECT_NE(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, test_build_and_save_big_model) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto sub_data_1 = OP_CFG(DATA).Attr("index", 0)
                                .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1024, 1024});
  GeTensorDesc data_tensor_desc(GeShape({1, 512, 1024, 1024}), FORMAT_NCHW, DT_FLOAT);
  std::vector<float32_t> data_value_vec1(1 * 512 * 1024 * 1024, 1);
  GeTensorPtr data_tensor1 = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec1.data(),
                                                   1 * 512 * 1024 * 1024 * sizeof(DT_FLOAT));
  auto const1 = OP_CFG(CONSTANT).Weight(data_tensor1);
  auto sub_add = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1024, 1024});
  auto sub_net_output = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1024, 1024});
  DEF_GRAPH(g1) {
    CHAIN(NODE("sub_data_1", sub_data_1)->EDGE(0, 0)->NODE("sub_add", sub_add));
    CHAIN(NODE("const1", const1)->EDGE(0, 1)->NODE("sub_add", sub_add));
    CHAIN(NODE("sub_add", sub_add)->EDGE(0, 0)->NODE("sub_net_output", sub_net_output));
  };

  const auto graph = ToGeGraph(g1);
  std::map<string, string> build_options{};
  ModelBufferData model_buffer_data{};
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  std::string output_file = "saved_model.om";
  EXPECT_EQ(aclgrphSaveModel(output_file, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, CheckPrecisionMode_FAILED_WhenValueIsInvalid) {
  std::map<std::string, std::string> init_options;
  init_options.insert(std::make_pair(ge::PRECISION_MODE, "invalid"));
  EXPECT_NE(aclgrphBuildInitialize(init_options), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, CheckPrecisionModeV2_FAILED_WhenValueIsInvalid) {
  std::map<std::string, std::string> init_options;
  init_options.insert(std::make_pair(ge::PRECISION_MODE_V2, "invalid"));
  EXPECT_NE(aclgrphBuildInitialize(init_options), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, evaluate_graph_mode) {
  std::map<string, string> init_options;
  init_options.emplace(VARIABLE_MEMORY_MAX_SIZE, std::to_string(1024UL * 1024UL * 1024UL));
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto sub_data_1 = OP_CFG(DATA).Attr("index", 0).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1024, 1024});
  auto var = OP_CFG(VARIABLE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1024, 1024 * 1024});
  auto sub_add = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1024, 1024 * 1024});
  auto sub_add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1024, 1024 * 1024});
  auto sub_net_output = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1024, 1024});
  DEF_GRAPH(g1) {
      CHAIN(NODE("sub_data_1", sub_data_1)->EDGE(0, 0)->NODE("sub_add", sub_add));
      CHAIN(NODE("var", var)->EDGE(0, 1)->NODE("sub_add", sub_add));
      CHAIN(NODE("sub_add", sub_add)->EDGE(0, 0)->NODE("sub_add1", sub_add1));
      CHAIN(NODE("var", var)->EDGE(0, 1)->NODE("sub_add1", sub_add1));
      CHAIN(NODE("sub_add1", sub_add1)->EDGE(0, 0)->NODE("sub_net_output", sub_net_output));
  };

  const auto graph = ToGeGraph(g1);
  std::map<string, string> build_options{};
  build_options.emplace(EVALUATE_GRAPH_RESOURCE_MODE, "1");
  ModelBufferData model_buffer_data{};
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, evaluate_graph_mode_simple) {
  std::map<string, string> init_options;
  init_options.emplace(VARIABLE_MEMORY_MAX_SIZE, std::to_string(1024UL * 1024UL * 1024UL));
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto sub_data_1 = OP_CFG(DATA).Attr("index", 0).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1024, 1024});
  auto var = OP_CFG(VARIABLE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1024, 1024 * 1024});
  auto sub_add = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1024, 1024 * 1024});
  auto sub_add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1024, 1024 * 1024});
  auto sub_net_output = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 512, 1024, 1024});
  DEF_GRAPH(g1) {
      CHAIN(NODE("sub_data_1", sub_data_1)->EDGE(0, 0)->NODE("sub_add", sub_add));
      CHAIN(NODE("var", var)->EDGE(0, 1)->NODE("sub_add", sub_add));
      CHAIN(NODE("sub_add", sub_add)->EDGE(0, 0)->NODE("sub_add1", sub_add1));
      CHAIN(NODE("var", var)->EDGE(0, 1)->NODE("sub_add1", sub_add1));
      CHAIN(NODE("sub_add1", sub_add1)->EDGE(0, 0)->NODE("sub_net_output", sub_net_output));
  };

  const auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  std::map<string, string> options{};
  ModelDataInfo model;
  auto ret = EvaluateGraphResource(options, compute_graph, model);
  EXPECT_EQ(ret, SUCCESS);
  // FakeOpsKernelBuilder::CalcOpRunningParam size加上了padding，导致offset变化了
  EXPECT_EQ(model.GetGraphMemorySize(), 4400193996288UL);
  EXPECT_EQ(model.GetVarMemorySize(), 2199023257088UL);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithShapeRangeWithCpuAndOsEmpty) {
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape_range", "[1, 1, -1, -1]"}
  };
  ModelBufferData model_buffer_data{};
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), GRAPH_FAILED);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, -1);
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), GRAPH_FAILED);

  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(op_desc, "_static_to_dynamic_softsync_op", true);
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  std::string output_file = "saved_model";
  EXPECT_EQ(aclgrphSaveModel(output_file, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithShapeRangeWihtOutputFileNameTooLong) {
  std::map<AscendString, AscendString> init_options;
  init_options[ge::OPTION_HOST_ENV_OS] = "linux";
  init_options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<string, string> build_options = {
      {"input_shape_range", "[1, 1, -1, -1]"}
  };
  ModelBufferData model_buffer_data{};
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), GRAPH_FAILED);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindFirstNodeMatchType(DATA);
  EXPECT_TRUE(data_node != nullptr);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, -1);
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), GRAPH_FAILED);

  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(op_desc, "_static_to_dynamic_softsync_op", true);
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);

  std::string output_file = "test12345678910_12345678910_12345678910_12345678910_12345678910_";
  output_file.append("12345678910_12345678910_12345678910_12345678910_12345678910_");
  output_file.append("12345678910_12345678910_12345678910_12345678910_12345678910_");
  output_file.append("12345678910_12345678910_12345678910_12345678910_12345678910");

  EXPECT_EQ(aclgrphSaveModel(output_file, model_buffer_data), PARAM_INVALID);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestBuildModelWithShapeRangeWithCpuNotExist) {
  std::map<AscendString, AscendString> init_options;
  init_options[ge::OPTION_HOST_ENV_OS] = "linux";
  init_options[ge::OPTION_HOST_ENV_CPU] = "eule";
  EXPECT_EQ(aclgrphBuildInitialize(init_options), ge::GRAPH_PARAM_INVALID);

  init_options[ge::OPTION_HOST_ENV_OS] = "windows";
  init_options[ge::OPTION_HOST_ENV_CPU] = "x86";
  EXPECT_EQ(aclgrphBuildInitialize(init_options), ge::GRAPH_PARAM_INVALID);

  aclgrphBuildFinalize();
}

static Graph ConstructUbFusionGraph() {
  DEF_GRAPH(fused_subgraph) {
    auto data_0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1})
        .Build("data_0");

    auto conv2d = OP_CFG(CONV2D)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16})
        .Build("conv2d");

    auto ret_val = OP_CFG("_RetVal")
        .InCnt(1)
        .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16})
        .Build("ret_val");
    CHAIN(NODE(data_0)->NODE(conv2d)->NODE(ret_val));
  };

  DEF_GRAPH(dynamic_graph) {
    auto data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1})
        .Build("data");

    auto fused_op = OP_CFG(MATMUL)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16})
        .Build("fused_op");

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1})
        .Build("net_output");

    CHAIN(NODE(data)->NODE(fused_op)->NODE(net_output));
  };

  auto org_graph = ToComputeGraph(fused_subgraph);
  auto root_graph = ToComputeGraph(dynamic_graph);
  auto fused_node = root_graph->FindNode("fused_op");
  EXPECT_TRUE(fused_node != nullptr);
  AttrUtils::SetGraph(fused_node->GetOpDesc(), "_original_fusion_graph", org_graph);
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  return graph;
}
/**
 * 用例描述：so进om后，解除执行时原型依赖，图模式离线流程 + 加载符合预期（包含ub融合）
 *
 * 预置条件：
 * 编译时注册部分算子原型
 *
 * 测试步骤：
 * 1. 构造计算图，包含ub融合子图
 * 2. 编译、保存为om文件
 * 3. 加载om
 * 4. 校验结果
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 加载后包含预期属性
 */
TEST_F(GeIrBuildTest, RecoverIrDefinition_graph_ub) {
  setenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM", "1", 0);
  BenchEnv::Init();
  auto graph = ConstructUbFusionGraph();

  std::map<AscendString, AscendString> init_options;
  init_options[ge::OPTION_HOST_ENV_OS] = "linux";
  init_options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);
  std::map<AscendString, AscendString> build_options;
  ModelBufferData model_buffer_data{};
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  std::string output_file = "./saved_model";
  EXPECT_EQ(aclgrphSaveModel(output_file, model_buffer_data), SUCCESS);

  ModelData model_data;
  GE_MAKE_GUARD(model_guard, [&model_data]() {
    if (model_data.model_data != nullptr) {
      delete[] static_cast<char_t *>(model_data.model_data);
      model_data.model_data = nullptr;
    }
  });

  EXPECT_EQ(ModelParserBase::LoadFromFile("./saved_model_linux_x86_64.om", 1, model_data), SUCCESS);
  // no memory allocated below
  EXPECT_EQ(ModelParserBase::LoadFromFile("", 1, model_data), ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID);
  system("touch test_xxx");
  EXPECT_EQ(ModelParserBase::LoadFromFile("./test_xxx", 1, model_data), ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID);
  system("rm -f ./test_xxx");

  ModelHelper model_helper;
  model_data.om_path = "./saved_model_linux_x86_64.om";
  EXPECT_EQ(model_helper.LoadRootModel(model_data), GRAPH_SUCCESS);
  auto root_model = model_helper.GetGeRootModel();
  ASSERT_NE(root_model, nullptr);
  auto root_graph = root_model->GetRootGraph();
  ASSERT_NE(root_graph, nullptr);
  NodePtr fused_node = nullptr;
  for (const auto &node : root_graph->GetAllNodes()){
    if (node->GetName() == "fused_op") {
      fused_node = node;
      break;
    }
  }
  ASSERT_NE(fused_node, nullptr);
  ComputeGraphPtr ub_graph = nullptr;
  AttrUtils::GetGraph(fused_node->GetOpDesc(),  "_original_fusion_graph", ub_graph);
  auto conv2d = ub_graph->FindNode("conv2d");
  ASSERT_NE(conv2d, nullptr);
  auto ir_inputs = conv2d->GetOpDesc()->GetIrInputs();
  auto ir_attr_names = conv2d->GetOpDesc()->GetIrAttrNames();
  const std::vector<std::string> target_ir_input = {"x", "filter", "bias", "offset_w"};
  const std::vector<std::string> target_ir_attr_name = {"strides", "pads", "dilations", "groups", "data_format", "offset_x"};
  EXPECT_EQ(ir_inputs.size(), target_ir_input.size());
  for (size_t i = 0U; i < ir_inputs.size(); ++i) {
    EXPECT_EQ(ir_inputs[i].first, target_ir_input[i]);
  }
  EXPECT_EQ(ir_attr_names.size(), target_ir_attr_name.size());
  for (size_t i = 0U; i < ir_attr_names.size(); ++i) {
    EXPECT_EQ(ir_attr_names[i], target_ir_attr_name[i]);
  }
  aclgrphBuildFinalize();
  unsetenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM");
}

static Graph ConstructBinReuseGraph() {
  DEF_GRAPH(g1) {
    OpDescPtr data[5];
    for (size_t i = 0U; i < 5U; ++i) {
      data[i] =
          OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {4, 4}).Build("data" + std::to_string(i));
    }

    OpDescPtr add[2];
    for (size_t i = 0U; i < 2U; ++i) {
      add[i] = OP_CFG(ADD).InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {4, 1}).Build("add" + std::to_string(i));
      add[i]->SetOpKernelLibName("AiCoreLib");
      add[i]->SetOpEngineName(kEngineNameAiCore);
    }

    OpDescPtr reduce[5];
    for (size_t i = 0U; i < 5U; ++i) {
      reduce[i] = OP_CFG(REDUCESUM)
                      .InCnt(1)
                      .OutCnt(1)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {4, 1})
                      .Build("reduce" + std::to_string(i));
      reduce[i]->SetOpKernelLibName("AiCoreLib");
      reduce[i]->SetOpEngineName(kEngineNameAiCore);
    }

    CHAIN(NODE(data[0])->EDGE(0, 0)->NODE(reduce[0]));
    CHAIN(NODE(data[1])->EDGE(0, 0)->NODE(reduce[1]));
    CHAIN(NODE(data[2])->EDGE(0, 0)->NODE(reduce[2]));
    CHAIN(NODE(data[3])->EDGE(0, 0)->NODE(reduce[3]));
    CHAIN(NODE(data[4])->EDGE(0, 0)->NODE(reduce[4])->EDGE(0, 2)->NODE("net_output", "NetOutput"));
    CHAIN(NODE(reduce[0])->EDGE(0, 0)->NODE(add[0])->EDGE(0, 0)->NODE("net_output", "NetOutput"));
    CHAIN(NODE(reduce[1])->EDGE(0, 1)->NODE(add[0]));
    CHAIN(NODE(reduce[2])->EDGE(0, 0)->NODE(add[1])->EDGE(0, 1)->NODE("net_output", "NetOutput"));
    CHAIN(NODE(reduce[3])->EDGE(0, 1)->NODE(add[1]));
  };
  auto graph = ToComputeGraph(g1);
  std::string tiling_data = "reduce_td";
  std::shared_ptr<optiling::utils::OpRunInfo> reduce_sum_run_info = std::make_shared<optiling::utils::OpRunInfo>();
  reduce_sum_run_info->AddTilingData(tiling_data.data(), tiling_data.size());
  reduce_sum_run_info->SetWorkspaces({1000});

  std::string atomic_tiling_data = "reduce_atomic_td";
  std::shared_ptr<optiling::utils::OpRunInfo> reduce_sum_atomic_run_info =
      std::make_shared<optiling::utils::OpRunInfo>();
  reduce_sum_atomic_run_info->AddTilingData(atomic_tiling_data.data(), atomic_tiling_data.size());
  reduce_sum_atomic_run_info->SetTilingKey(10U);
  reduce_sum_atomic_run_info->SetBlockDim(10U);
  reduce_sum_atomic_run_info->SetWorkspaces({1000, 1000});

  std::string add_tiling_data = "add_td";
  std::shared_ptr<optiling::utils::OpRunInfo> add_run_info = std::make_shared<optiling::utils::OpRunInfo>();
  add_run_info->AddTilingData(add_tiling_data.data(), add_tiling_data.size());
  add_run_info->SetTilingKey(20U);
  add_run_info->SetBlockDim(20U);
  add_run_info->SetWorkspaces({2000, 2000});

  std::shared_ptr<optiling::utils::OpRunInfo> empty_run_info = std::make_shared<optiling::utils::OpRunInfo>();
  empty_run_info->SetTilingKey(30U);
  empty_run_info->SetBlockDim(30U);
  empty_run_info->SetWorkspaces({3000, 3000});

  for (auto node : graph->GetAllNodesPtr()) {
    if (node->GetType() == "ReduceSum") {
      node->GetOpDescBarePtr()->SetExtAttr(ATTR_NAME_OP_RUN_INFO, reduce_sum_run_info);
      node->GetOpDescBarePtr()->SetExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO, reduce_sum_atomic_run_info);
    }
    if (node->GetType() == "Add") {
      node->GetOpDescBarePtr()->SetExtAttr(ATTR_NAME_OP_RUN_INFO, add_run_info);
      node->GetOpDescBarePtr()->SetExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO, empty_run_info);
    }
  }
  return GraphUtilsEx::CreateGraphFromComputeGraph(graph);
}

TEST_F(GeIrBuildTest, recover_op_runinfo_static_graph) {
  GeRunningEnvFaker ge_env;
  ge_env.InstallDefault().Install(FakeOp(REDUCESUM).InfoStoreAndBuilder("AicoreLib").InferShape(StubInferFunction));
  auto graph = ConstructBinReuseGraph();
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);
  std::map<AscendString, AscendString> build_options;
  ModelBufferData model_buffer_data{};
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  std::string output_file = "./saved_model_bin_reuse";
  EXPECT_EQ(aclgrphSaveModel(output_file, model_buffer_data), SUCCESS);
  ModelData model_data;
  GE_MAKE_GUARD(model_guard, [&model_data]() {
    if (model_data.model_data != nullptr) {
      delete[] static_cast<char_t *>(model_data.model_data);
      model_data.model_data = nullptr;
    }
  });
  EXPECT_EQ(ModelParserBase::LoadFromFile("./saved_model_bin_reuse.om", 1, model_data), SUCCESS);
  ModelHelper model_helper;
  model_data.om_path = "./saved_model_bin_reuse.om";
  EXPECT_EQ(model_helper.LoadRootModel(model_data), GRAPH_SUCCESS);
  auto root_model = model_helper.GetGeRootModel();
  ASSERT_NE(root_model, nullptr);
  auto root_graph = root_model->GetRootGraph();
  ASSERT_NE(root_graph, nullptr);
  ASSERT_NE(root_model->GetHostResourceCenterPtr(), nullptr);
  auto manager = root_model->GetHostResourceCenterPtr()->GetHostResourceMgr(HostResourceType::kTilingData);
  TilingResourceManager *tiling_resource_manager =
      const_cast<TilingResourceManager *>(dynamic_cast<const TilingResourceManager *>(manager));
  ASSERT_NE(tiling_resource_manager, nullptr);
  // unique size.
  EXPECT_EQ(tiling_resource_manager->GetResourceToKeys().size(), 4U);

  // check recovered attrs
  for (auto node : root_graph->GetAllNodesPtr()) {
    if (node->GetType() == "ReduceSum") {
      std::shared_ptr<optiling::utils::OpRunInfo> reduce_sum_run_info;
      std::shared_ptr<optiling::utils::OpRunInfo> reduce_sum_atomic_run_info;
      reduce_sum_run_info = node->GetOpDescBarePtr()->TryGetExtAttr(ATTR_NAME_OP_RUN_INFO, reduce_sum_run_info);
      reduce_sum_atomic_run_info =
          node->GetOpDescBarePtr()->TryGetExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO, reduce_sum_atomic_run_info);
      ASSERT_NE(reduce_sum_run_info.get(), nullptr);
      EXPECT_EQ(reduce_sum_run_info->GetAllTilingData().str(), "reduce_td");
      ASSERT_NE(reduce_sum_atomic_run_info.get(), nullptr);
      EXPECT_EQ(reduce_sum_atomic_run_info->GetAllTilingData().str(), "reduce_atomic_td");

      EXPECT_EQ(reduce_sum_atomic_run_info->GetTilingKey(), 10U);
      EXPECT_EQ(reduce_sum_atomic_run_info->GetBlockDim(), 10U);
      std::vector<int64_t> reduce_workspace{1000, 1000};
      EXPECT_EQ(reduce_sum_atomic_run_info->GetAllWorkspaces(), reduce_workspace);

    }
    if (node->GetType() == "Add") {
      // old func
      tiling_resource_manager->UseOpIdAsTilingId();
      tiling_resource_manager->GetResource(node->GetOpDesc(), static_cast<int64_t>(TilingResourceType::kAtomic));
      std::shared_ptr<optiling::utils::OpRunInfo> add_run_info;
      add_run_info = node->GetOpDescBarePtr()->TryGetExtAttr(ATTR_NAME_OP_RUN_INFO, add_run_info);
      ASSERT_NE(add_run_info.get(), nullptr);
      EXPECT_EQ(add_run_info->GetAllTilingData().str(), "add_td");
      EXPECT_EQ(add_run_info->GetTilingKey(), 20U);
      EXPECT_EQ(add_run_info->GetBlockDim(), 20U);
      std::vector<int64_t> add_workspace{2000, 2000};
      EXPECT_EQ(add_run_info->GetAllWorkspaces(), add_workspace);

      std::shared_ptr<optiling::utils::OpRunInfo> empty_run_info;
      empty_run_info = node->GetOpDescBarePtr()->TryGetExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO, empty_run_info);
      ASSERT_NE(empty_run_info.get(), nullptr);
      EXPECT_EQ(empty_run_info->GetAllTilingData().str(), "");
      EXPECT_EQ(empty_run_info->GetTilingKey(), 30U);
      EXPECT_EQ(empty_run_info->GetBlockDim(), 30U);
      std::vector<int64_t> golden_workspace{3000, 3000};
      EXPECT_EQ(empty_run_info->GetAllWorkspaces(), golden_workspace);
    }
  }

  aclgrphBuildFinalize();
  ReInitGe();
  system(("rm -f " + model_data.om_path).c_str());
}

static Graph ConstructDynamicBinReuseGraph() {
  DEF_GRAPH(sub) {
    OpDescPtr data = OP_CFG(DATA)
                         .InCnt(1)
                         .OutCnt(1)
                         .TensorDesc(FORMAT_ND, DT_FLOAT, {4, 4})
                         .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                         .Attr(ATTR_NAME_INDEX, 0)
                         .Build("sub_data");
    OpDescPtr reduce = OP_CFG(REDUCESUM).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {4, 1}).Build("reduce");
    auto sub_output = OP_CFG(ge::NETOUTPUT)
                          .TensorDesc(FORMAT_NCHW, DT_FLOAT, {4, 1})
                          .InCnt(1)
                          .OutCnt(1)
                          .Build("sub_output");

    AttrUtils::SetInt(sub_output->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);

    reduce->SetOpKernelLibName("AiCoreLib");
    reduce->SetOpEngineName(kEngineNameAiCore);
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(reduce)->EDGE(0, 0)->NODE(sub_output));
  };

  DEF_GRAPH(root) {
    OpDescPtr data = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1}).Build("data0");
    OpDescPtr reduce = OP_CFG(REDUCESUM).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {4, 4}).Build("reduce1");
    reduce->SetOpKernelLibName("AiCoreLib");
    reduce->SetOpEngineName(kEngineNameAiCore);

    auto root_netoutput =
        OP_CFG(ge::NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, -1}).InCnt(2).OutCnt(1).Build("NetOutput");
    root_netoutput->SetSrcName({"known_op", "reduce1"});
    root_netoutput->SetSrcIndex({0, 1});
    auto known_op =
        OP_CFG(ge::PARTITIONEDCALL).TensorDesc(FORMAT_NCHW, DT_FLOAT, {4, 4}).InCnt(1).OutCnt(1).Build("known_op");

    CHAIN(NODE(data)->NODE(known_op)->EDGE(0, 0)->NODE(root_netoutput));
    CHAIN(NODE(data)->NODE(reduce)->EDGE(0, 1)->NODE(root_netoutput));
  };

  auto graph = ToGeGraph(root);
  auto root_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  root_graph->SetGraphUnknownFlag(true);

  auto sub_graph1 = ToGeGraph(sub);
  auto compute_graph1 = ge::GraphUtilsEx::GetComputeGraph(sub_graph1);
  compute_graph1->SetGraphUnknownFlag(false);
  auto net_output = compute_graph1->FindNode("sub_output");
  net_output->GetOpDesc()->SetSrcName({"reduce"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  auto known_node = root_graph->FindNode("known_op");
  known_node->GetOpDesc()->RegisterSubgraphIrName("f", SubgraphType::kStatic);

  size_t index = known_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  known_node->GetOpDesc()->AddSubgraphName(compute_graph1->GetName());
  known_node->GetOpDesc()->SetSubgraphInstanceName(index, compute_graph1->GetName());
  compute_graph1->SetParentNode(known_node);
  compute_graph1->SetParentGraph(root_graph);
  root_graph->AddSubgraph(compute_graph1->GetName(), compute_graph1);

  std::string tiling_data = "reduce_td";
  std::shared_ptr<optiling::utils::OpRunInfo> reduce_sum_run_info = std::make_shared<optiling::utils::OpRunInfo>();
  reduce_sum_run_info->AddTilingData(tiling_data.data(), tiling_data.size());
  reduce_sum_run_info->SetWorkspaces({1000});

  std::string atomic_tiling_data = "reduce_atomic_td";
  std::shared_ptr<optiling::utils::OpRunInfo> reduce_sum_atomic_run_info =
      std::make_shared<optiling::utils::OpRunInfo>();
  reduce_sum_atomic_run_info->AddTilingData(atomic_tiling_data.data(), atomic_tiling_data.size());

  for (auto node : root_graph->GetAllNodesPtr()) {
    if (node->GetType() == "ReduceSum") {
      node->GetOpDescBarePtr()->SetExtAttr(ATTR_NAME_OP_RUN_INFO, reduce_sum_run_info);
      node->GetOpDescBarePtr()->SetExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO, reduce_sum_atomic_run_info);
    }
  }
  return graph;
}

TEST_F(GeIrBuildTest, recover_op_runinfo_dyn_graph) {
  GeRunningEnvFaker ge_env;
  ge_env.InstallDefault();
  ge_env.InstallDefault().Install(FakeOp(REDUCESUM).InfoStoreAndBuilder("AicoreLib").InferShape(StubInferFunction));
  auto graph = ConstructDynamicBinReuseGraph();
  std::map<AscendString, AscendString> init_options;

  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);
  std::map<AscendString, AscendString> build_options;
  ModelBufferData model_buffer_data{};
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);

  std::string output_file = "./saved_dyn_model_bin_reuse";
  EXPECT_EQ(aclgrphSaveModel(output_file, model_buffer_data), SUCCESS);
  ModelData model_data;
  GE_MAKE_GUARD(model_guard, [&model_data]() {
    if (model_data.model_data != nullptr) {
      delete[] static_cast<char_t *>(model_data.model_data);
      model_data.model_data = nullptr;
    }
  });
  EXPECT_EQ(ModelParserBase::LoadFromFile("./saved_dyn_model_bin_reuse_linux_x86_64.om", 1, model_data), SUCCESS);
  ModelHelper model_helper;
  model_data.om_path = "./saved_dyn_model_bin_reuse_linux_x86_64.om";
  EXPECT_EQ(model_helper.LoadRootModel(model_data), GRAPH_SUCCESS);
  auto root_model = model_helper.GetGeRootModel();
  ASSERT_NE(root_model, nullptr);
  auto root_graph = root_model->GetRootGraph();
  ASSERT_NE(root_graph, nullptr);
  EXPECT_NE(root_model->GetHostResourceCenterPtr(), nullptr);
  auto manager = root_model->GetHostResourceCenterPtr()->GetHostResourceMgr(HostResourceType::kTilingData);
  const TilingResourceManager *tiling_resource_manager = dynamic_cast<const TilingResourceManager *>(manager);
  ASSERT_NE(tiling_resource_manager, nullptr);
  // unique size.
  EXPECT_EQ(tiling_resource_manager->GetResourceToKeys().size(), 2U);

  for (auto &iter : root_model->GetSubgraphInstanceNameToModel()) {
    for (auto &node : iter.second->GetGraph()->GetDirectNode()) {
      if (node->GetName() == "reduce1") {
        std::shared_ptr<optiling::utils::OpRunInfo> default_run_info;
        std::shared_ptr<optiling::utils::OpRunInfo> add_run_info;
        add_run_info = node->GetOpDescBarePtr()->TryGetExtAttr(ATTR_NAME_OP_RUN_INFO, default_run_info);
        EXPECT_NE(add_run_info, nullptr);
      }
    }
  }

  for (auto &node : root_graph->GetAllNodes()) {
    if (node->GetName() == "reduce1") {
      std::shared_ptr<optiling::utils::OpRunInfo> default_run_info;
      std::shared_ptr<optiling::utils::OpRunInfo> add_run_info;
      add_run_info = node->GetOpDescBarePtr()->TryGetExtAttr(ATTR_NAME_OP_RUN_INFO, default_run_info);
      EXPECT_NE(add_run_info, nullptr);
    }
  }


  aclgrphBuildFinalize();
  ReInitGe();
  system(("rm -f " + model_data.om_path).c_str());
}

TEST_F(GeIrBuildTest, aclgrphGenerateForOp_singleop_test) {
  std::string pwd = __FILE__;
  std::size_t idx = pwd.find_last_of("/");
  pwd = pwd.substr(0, idx);
  std::string json_file = pwd + "/../st_run_data/json/single_op/add_op.json";
  ge::AscendString json(json_file.c_str());

  std::vector<Graph> graphs;
  auto ret = ge::aclgrphGenerateForOp(json, graphs);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(graphs.size(), 8);
}

TEST_F(GeIrBuildTest, TestScreenPrintMode_disable) {
  BenchEnv::Init();
  std::map<AscendString, AscendString> init_options;
  init_options[ge::OPTION_SCREEN_PRINT_MODE] = "disable";
  init_options[ge::OPTION_HOST_ENV_OS] = "linux";
  init_options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<AscendString, AscendString> build_options;
  build_options.emplace(ge::ir_option::INPUT_FORMAT, "NCHW");
  ModelBufferData model_buffer_data{};
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  std::string output_file = "./saved_model.om";

  std::stringstream ss;
  std::streambuf *coutbuf = std::cout.rdbuf();
  std::cout.rdbuf(ss.rdbuf());

  EXPECT_EQ(aclgrphSaveModel(output_file, model_buffer_data), SUCCESS);

  std::cout.rdbuf(coutbuf);
  std::string out_log = ss.str();
  EXPECT_EQ((out_log.find(kPlatformIrrelevant) != std::string::npos), false);

  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestScreenPrintMode_enable) {
  BenchEnv::Init();
  std::map<AscendString, AscendString> init_options;
  init_options[ge::OPTION_SCREEN_PRINT_MODE] = "enable";
  init_options[ge::OPTION_HOST_ENV_OS] = "linux";
  init_options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<AscendString, AscendString> build_options;
  build_options.emplace(ge::ir_option::INPUT_FORMAT, "NCHW");
  ModelBufferData model_buffer_data{};
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  std::string output_file = "./saved_model.om";

  std::stringstream ss;
  std::streambuf *coutbuf = std::cout.rdbuf();
  std::cout.rdbuf(ss.rdbuf());

  EXPECT_EQ(aclgrphSaveModel(output_file, model_buffer_data), SUCCESS);

  std::cout.rdbuf(coutbuf);
  std::string out_log = ss.str();
  EXPECT_EQ((out_log.find(kPlatformIrrelevant) != std::string::npos), true);

  aclgrphBuildFinalize();
  system(("rm -f " + output_file).c_str());
}

TEST_F(GeIrBuildTest, TestScreenPrintMode_err) {
  std::map<AscendString, AscendString> init_options;
  init_options[ge::OPTION_SCREEN_PRINT_MODE] = "disable_xxx";
  init_options[ge::OPTION_HOST_ENV_OS] = "linux";
  init_options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  EXPECT_EQ(aclgrphBuildInitialize(init_options), ge::GRAPH_PARAM_INVALID);
  aclgrphBuildFinalize();
}

TEST_F(GeIrBuildTest, TestSocVersionCheck_ok) {
  dlog_setlevel(GE_MODULE_NAME, 0, 0);
  BenchEnv::Init();
  ge::GEFinalize();
  GeRunningEnvFaker ge_env;
  ge_env.InstallDefault();
  std::map<AscendString, AscendString> init_options;
  init_options[ge::OPTION_HOST_ENV_OS] = "linux";
  init_options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  init_options[ge::configure_option::SOC_VERSION] = "";
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<AscendString, AscendString> build_options;
  build_options.emplace(ge::ir_option::INPUT_FORMAT, "NCHW");
  ModelBufferData model_buffer_data{};

  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);

  ModelData model_data;
  model_data.model_data = model_buffer_data.data.get();
  model_data.model_len = model_buffer_data.length;
  ModelHelper model_helper;
  model_helper.LoadModel(model_data);
  const auto &ge_model = model_helper.GetGeModel();
  EXPECT_NE(ge_model, nullptr);
  std::string soc_version;
  std::string arch_type;
  AttrUtils::GetStr(*ge_model, "soc_version", soc_version);
  AttrUtils::GetStr(*ge_model, "arch_type", arch_type);
  EXPECT_EQ(soc_version, "");
  EXPECT_EQ(arch_type, "0");
  ReInitGe();
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
}

TEST_F(GeIrBuildTest, TestSocVersionCheck_ok_Nano) {
  BenchEnv::Init();
  ge::GEFinalize();
  GeRunningEnvFaker ge_env;
  ge_env.InstallDefault();
  std::map<AscendString, AscendString> init_options;
  init_options[ge::OPTION_HOST_ENV_OS] = "linux";
  init_options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  init_options[ge::configure_option::SOC_VERSION] = "Ascend035";
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[ge::SOC_VERSION] = "Ascend035";
  (void)GetThreadLocalContext().SetGraphOption(graph_options);
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  auto graph = GraphFactory::SingeOpGraph2();
  std::map<AscendString, AscendString> build_options;
  //build_options.emplace(ge::ir_option::INPUT_FORMAT, "NCHW");
  ModelBufferData model_buffer_data{};

  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);

  ModelData model_data;
  model_data.model_data = model_buffer_data.data.get();
  model_data.model_len = model_buffer_data.length;
  ModelHelper model_helper;
  model_helper.LoadModel(model_data);
  const auto &ge_model = model_helper.GetGeModel();
  EXPECT_NE(ge_model, nullptr);
  std::string soc_version;
  std::string arch_type;
  AttrUtils::GetStr(*ge_model, "soc_version", soc_version);
  AttrUtils::GetStr(*ge_model, "arch_type", arch_type);
  //EXPECT_EQ(soc_version, "Ascend035");
  EXPECT_EQ(arch_type, "0");
  ReInitGe();
}

TEST_F(GeIrBuildTest, ir_build_so_in_om_multi_customize_priroity) {
  BenchEnv::Init();
  setenv("ASCEND_OPP_PATH", "./", 0);
  auto graph = ConstructUbFusionGraph();

  std::map<AscendString, AscendString> init_options;
  init_options[ge::OPTION_HOST_ENV_OS] = "linux";
  init_options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);
  std::map<AscendString, AscendString> build_options;
  ModelBufferData model_buffer_data{};
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);

  std::string pwd = __FILE__;
  size_t pos = pwd.find_last_of("/");
  pwd = pwd.substr(0, pos);
  auto tmp_path = pwd + "/temp_opp_path";
  system(("mkdir -p " + tmp_path).c_str());

  unsetenv("ASCEND_OPP_PATH");
  setenv("ASCEND_CUSTOM_OPP_PATH", tmp_path.c_str(), 0);
  setenv("ASCEND_OPP_PATH", tmp_path.c_str(), 0);
  std::string output_file = "./saved_model";
  EXPECT_EQ(aclgrphSaveModel(output_file, model_buffer_data), PARAM_INVALID);
  system(("rm -rf " + tmp_path).c_str());
  unsetenv("ASCEND_OPP_PATH");
}

void BuildAndCheckSimpleConstCastGraph(bool use_const, DataType dtype) {
  const std::vector<int64_t> shape = { 2, 3, 4, 5 };
  auto tensor = GenerateTensor(dtype, shape);
  auto const_0 = OP_CFG(use_const ? CONSTANT : CONSTANTOP).OutCnt(1)
                           .TensorDesc(FORMAT_NCHW, dtype, shape)
                           .Weight(tensor)
                           .Build("const_1");
  auto cast_0 = OP_CFG(CAST).InCnt(1).OutCnt(1)
                            .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                            .Build("cast_1");
  DEF_GRAPH(g1) {
    CHAIN(NODE(const_0)->NODE(cast_0));
  };
  auto graph = ToGeGraph(g1);

  std::vector<Operator> output_nodes;
  EXPECT_EQ(graph.FindOpByType(NETOUTPUT, output_nodes), SUCCESS);
  EXPECT_EQ(output_nodes.size(), 0U);

  std::map<AscendString, AscendString> options = {
    { OO_CONSTANT_FOLDING, "false" },
  };
  Session session(options);
  std::vector<Tensor> inputs;

  EXPECT_EQ(session.AddGraph(0, graph), SUCCESS);
  EXPECT_EQ(session.BuildGraph(0, inputs), SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto node_0 = graph->FindFirstNodeMatchType(CONSTANT);
    auto node_1 = graph->FindFirstNodeMatchType(CAST);

    EXPECT_NE(node_0, nullptr);
    EXPECT_NE(node_1, nullptr);
    EXPECT_EQ(node_0->GetOpDesc()->GetOutputDesc(0).GetDataType(), dtype);
    EXPECT_EQ(node_1->GetOpDesc()->GetInputDesc(0).GetDataType(), dtype);

    auto netoutput = graph->FindFirstNodeMatchType(NETOUTPUT);
    EXPECT_NE(netoutput, nullptr);
  };
}

TEST_F(GeIrBuildTest, Fp8ConstGraph_hif8_ok) {
  BuildAndCheckSimpleConstCastGraph(true, DT_HIFLOAT8);
  BuildAndCheckSimpleConstCastGraph(false, DT_HIFLOAT8);
}

TEST_F(GeIrBuildTest, Fp4ConstGraph_hif4_ok) {
  BuildAndCheckSimpleConstCastGraph(true, DT_HIFLOAT4);
  BuildAndCheckSimpleConstCastGraph(false, DT_HIFLOAT4);
}

TEST_F(GeIrBuildTest, Fp8ConstGraph_fp8e5m2_ok) {
  BuildAndCheckSimpleConstCastGraph(true, DT_FLOAT8_E5M2);
  BuildAndCheckSimpleConstCastGraph(false, DT_FLOAT8_E5M2);
}

TEST_F(GeIrBuildTest, Fp8ConstGraph_fp8e4m3fn_ok) {
  BuildAndCheckSimpleConstCastGraph(true, DT_FLOAT8_E4M3FN);
  BuildAndCheckSimpleConstCastGraph(false, DT_FLOAT8_E4M3FN);
}
}  // namespace ge

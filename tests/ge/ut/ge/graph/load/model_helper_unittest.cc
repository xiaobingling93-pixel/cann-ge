/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <gtest/gtest.h>
#include "mmpa/mmpa_api.h"
#include "macro_utils/dt_public_scope.h"
#include "framework/common/helper/model_helper.h"
#include "hybrid/node_executor/aicore/aicore_op_task.h"
#include "framework/omg/ge_init.h"
#include "common/model/ge_model.h"
#include "common/helper/model_parser_base.h"
#include "graph/buffer/buffer_impl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_local_context.h"
#include "depends/runtime/src/runtime_stub.h"
#include "macro_utils/dt_public_unscope.h"
#include "faker/space_registry_faker.h"
#include "common/path_utils.h"
#include "ge/framework/common/taskdown_common.h"
#include "stub/gert_runtime_stub.h"
#include "common/share_graph.h"
#include "proto/task.pb.h"
#include "faker/space_registry_faker.h"
#include "common/opskernel/ops_kernel_info_types.h"

using namespace std;
extern std::string g_runtime_stub_mock;

namespace ge {
using namespace hybrid;
namespace {
const char *const kEnvName = "ASCEND_OPP_PATH";
const char *const kEnvNameCustom = "ASCEND_CUSTOM_OPP_PATH";
const string kOpsProto = "libopsproto_rt2.0.so";
const string kOpMaster = "libopmaster_rt2.0.so";
const string kInner = "built-in";
const string kOpsProtoPath = "/op_proto/lib/linux/x86_64/";
const string kOpMasterPath = "/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/";

static void FillModelTaskDef(GeModelPtr ge_model, ModelTaskType task_type = ModelTaskType::MODEL_TASK_ALL_KERNEL,
                             ccKernelType kernel_type = ccKernelType::TE,
                             std::string so_name = "") {
  domi::ModelTaskDef model_task_def;
  std::shared_ptr<domi::ModelTaskDef> model_task_def_ptr = make_shared<domi::ModelTaskDef>(model_task_def);
  domi::TaskDef *task_def = model_task_def_ptr->add_task();
  ge_model->SetModelTaskDef(model_task_def_ptr);

  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
  task_def->set_type(static_cast<uint32_t>(task_type));
  domi::KernelDefWithHandle *kernel_with_handle = task_def->mutable_kernel_with_handle();
  kernel_with_handle->set_original_kernel_key("");
  kernel_with_handle->set_node_info("");
  kernel_with_handle->set_block_dim(32);
  kernel_with_handle->set_args_size(64);
  domi::KernelDef *kernel_def = task_def->mutable_kernel();
  kernel_def->set_block_dim(32);
  kernel_def->set_args_size(64);
  kernel_def->set_so_name(so_name);
  string args(64, '1');
  kernel_with_handle->set_args(args.data(), 64);
  domi::KernelContext *context = kernel_with_handle->mutable_context();
  context->set_op_index(1);
  context->set_kernel_type(static_cast<uint32_t>(kernel_type));    // ccKernelType::TE
  uint16_t args_offset[9] = {0};
  context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
}

static GeRootModelPtr ConstructGeRootModel(bool is_dynamic_shape = true,
                                           ModelTaskType task_type = ModelTaskType::MODEL_TASK_KERNEL,
                                           ccKernelType kernel_type = ccKernelType::TE,
                                           std::string so_name = "") {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  AttrUtils::SetBool(graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape);
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_root_model->subgraph_instance_name_to_model_["graph"] = ge_model;
  ge_model->SetGraph(graph);
  (void)ge::AttrUtils::SetStr(ge_model, ATTR_MODEL_HOST_ENV_OS, "linux");
  (void)ge::AttrUtils::SetStr(ge_model, ATTR_MODEL_HOST_ENV_CPU, "x86_64");

  GeModelPtr ge_model1 = std::make_shared<GeModel>();
  ge::ComputeGraphPtr graph1 = std::make_shared<ge::ComputeGraph>("graph1");
  ge_model1->SetGraph(graph1);
  ge_root_model->subgraph_instance_name_to_model_["graph1"] = ge_model1;
  FillModelTaskDef(ge_model, task_type, kernel_type, so_name);
  FillModelTaskDef(ge_model1, task_type, kernel_type, so_name);
  return ge_root_model;
}
}

class UtestModelHelper : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UtestModelHelper, save_size_to_modeldef)
{
  GeModelPtr ge_model = ge::MakeShared<ge::GeModel>();
  std::shared_ptr<domi::ModelTaskDef> task = ge::MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(task);
  ModelHelper model_helper;
  EXPECT_EQ(SUCCESS, model_helper.SaveSizeToModelDef(ge_model, 1));
  EXPECT_EQ(SUCCESS, model_helper.SaveSizeToModelDef(ge_model, 0));
}

TEST_F(UtestModelHelper, SaveModelPartitionInvalid)
{
  std::shared_ptr<OmFileSaveHelper> om_file_save_helper = std::make_shared<OmFileSaveHelper>();
  ModelPartitionType type = MODEL_DEF;
  uint8_t data[128] = {0};
  size_t size = 10000000000;
  size_t model_index = 0;
  ModelHelper model_helper;
  EXPECT_EQ(model_helper.SaveModelPartition(om_file_save_helper, type, (uint8_t*)data, size, model_index), PARAM_INVALID);
  type = WEIGHTS_DATA;
  EXPECT_EQ(model_helper.SaveModelPartition(om_file_save_helper, type, (uint8_t*)data, size, model_index), SUCCESS);
  type = TASK_INFO;
  EXPECT_EQ(model_helper.SaveModelPartition(om_file_save_helper, type, (uint8_t*)data, size, model_index), PARAM_INVALID);
  type = TBE_KERNELS;
  EXPECT_EQ(model_helper.SaveModelPartition(om_file_save_helper, type, (uint8_t*)data, size, model_index), SUCCESS);
  type = CUST_AICPU_KERNELS;
  EXPECT_EQ(model_helper.SaveModelPartition(om_file_save_helper, type, (uint8_t*)data, size, model_index), SUCCESS);
  type = MODEL_INOUT_INFO;
  EXPECT_EQ(model_helper.SaveModelPartition(om_file_save_helper, type, (uint8_t*)data, size, model_index), PARAM_INVALID);
  type = (ModelPartitionType)100;
  EXPECT_EQ(model_helper.SaveModelPartition(om_file_save_helper, type, (uint8_t*)data, size, model_index), PARAM_INVALID);
  type = SO_BINS;
  EXPECT_EQ(model_helper.SaveModelPartition(om_file_save_helper, type, (uint8_t*)data, size, model_index), PARAM_INVALID);
  size = 1024;
  EXPECT_EQ(model_helper.SaveModelPartition(om_file_save_helper, type, nullptr, size, model_index), PARAM_INVALID);
  EXPECT_EQ(model_helper.SaveModelPartition(om_file_save_helper, type, (uint8_t*)data, size, model_index), SUCCESS);
  ASSERT_FALSE(om_file_save_helper->model_contexts_.empty());
  om_file_save_helper->model_contexts_[0U].model_data_len_ = 4000000000U;
  model_index = 2000000000U;
  EXPECT_EQ(model_helper.SaveModelPartition(om_file_save_helper, type, (uint8_t*)data, size, model_index), PARAM_INVALID);
}

TEST_F(UtestModelHelper, SaveOriginalGraphToOmModel)
{
  Graph graph("graph");
  std::string output_file = "";
  ModelHelper model_helper;
  EXPECT_EQ(model_helper.SaveOriginalGraphToOmModel(graph, output_file), FAILED);
  output_file = "output.graph";
  EXPECT_EQ(model_helper.SaveOriginalGraphToOmModel(graph, output_file), FAILED);
  ge::OpDescPtr add_op(new ge::OpDesc("add1", "Add"));
  add_op->AddDynamicInputDesc("input", 2);
  add_op->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto add_node = compute_graph->AddNode(add_op);
  auto graph2 = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  EXPECT_EQ(model_helper.SaveOriginalGraphToOmModel(graph2, output_file), SUCCESS);
}

TEST_F(UtestModelHelper, GetGeModel)
{
  ModelHelper model_helper;
  model_helper.model_ = nullptr;
  EXPECT_NE(model_helper.GetGeModel(), nullptr);
}

TEST_F(UtestModelHelper, LoadTask)
{
  ModelHelper model_helper;
  OmFileLoadHelper om_load_helper;
  GeModelPtr cur_model = std::make_shared<GeModel>();
  size_t mode_index = 10;
  EXPECT_EQ(model_helper.LoadTask(om_load_helper, cur_model, mode_index), FAILED);
}

TEST_F(UtestModelHelper, LoadTaskByHelper)
{
  ModelHelper model_helper;
  OmFileLoadHelper om_load_helper;
  om_load_helper.is_inited_ = false;
  EXPECT_EQ(model_helper.LoadTask(om_load_helper, model_helper.model_, 0U), FAILED);
  om_load_helper.is_inited_ = true;
  ModelPartition mp;
  mp.type = TASK_INFO;
  om_load_helper.model_contexts_.emplace_back(OmFileContext{});
  om_load_helper.model_contexts_[0U].partition_datas_.push_back(mp);
  model_helper.model_ = std::make_shared<GeModel>();
  EXPECT_EQ(model_helper.LoadTask(om_load_helper, model_helper.model_, 0U), SUCCESS);
}

TEST_F(UtestModelHelper, SaveToOmRootModel)
{
  ModelHelper model_helper;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge::ComputeGraphPtr subgraph = std::make_shared<ge::ComputeGraph>("subgraph");
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(subgraph);
  ge_root_model->subgraph_instance_name_to_model_["graph"] = ge_model;
  std::string output_file = "output.file";
  ModelBufferData model;
  bool is_unknown_shape = true;
  EXPECT_NE(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, is_unknown_shape), SUCCESS);
  ge::ComputeGraphPtr subgraph2 = std::make_shared<ge::ComputeGraph>("subgraph2");
  GeModelPtr ge_model2 = std::make_shared<GeModel>();
  ge_root_model->subgraph_instance_name_to_model_["graph2"] = ge_model2;
  EXPECT_NE(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, is_unknown_shape), SUCCESS);
}

TEST_F(UtestModelHelper, SaveModelDef)
{
  ModelHelper model_helper;
  std::shared_ptr<OmFileSaveHelper> om_file_save_helper = std::make_shared<OmFileSaveHelper>();
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge::Buffer model_buffer;
  model_buffer.impl_->buffer_ = nullptr;
  size_t model_index = 0;
  EXPECT_NE(model_helper.SaveModelDef(om_file_save_helper, ge_model, model_buffer, model_index), SUCCESS);
}

TEST_F(UtestModelHelper, SaveAllModelPartiton)
{
  ModelHelper model_helper;
  std::shared_ptr<OmFileSaveHelper> om_file_save_helper = std::make_shared<OmFileSaveHelper>();
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge::Buffer model_buffer;
  model_buffer.impl_->buffer_ = nullptr;
  ge::Buffer task_buffer;
  size_t model_index = 0;
  EXPECT_EQ(model_helper.SaveAllModelPartiton(om_file_save_helper, ge_model, model_buffer, task_buffer, model_index), FAILED);
}

TEST_F(UtestModelHelper, SaveToOmModel)
{
  ModelHelper model_helper;
  GeModelPtr ge_model = std::make_shared<GeModel>();
  std::string output_file = "";
  ModelBufferData model;
  EXPECT_EQ(model_helper.SaveToOmModel(ge_model, output_file, model), FAILED);
}

TEST_F(UtestModelHelper, LoadFromFile_failed) {
  ModelParserBase base;
  ge::ModelData model_data;
  EXPECT_EQ(base.LoadFromFile("/tmp/123test", -1, model_data), ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID);
}

TEST_F(UtestModelHelper, SaveModelIntroduction) {
  ModelHelper model_helper;
  std::shared_ptr<OmFileSaveHelper> om_file_save_helper = std::make_shared<OmFileSaveHelper>();
  GeModelPtr ge_model = std::make_shared<GeModel>();
  size_t model_index = 0U;
  EXPECT_NE(model_helper.SaveModelIntroduction(om_file_save_helper, ge_model, model_index), SUCCESS);
}

TEST_F(UtestModelHelper, LoadPartInfoFromModel) {
  ModelHelper model_helper;
  ModelPartition partition;
  ModelBufferData model_buffer;
  ModelData model_data{model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", "/tmp/a"};
  EXPECT_EQ(model_helper.LoadPartInfoFromModel(model_data, partition), ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);
}

TEST_F(UtestModelHelper, GetWeightDataTest) {
  GeModel ge_model;
  uint8_t i = 0;
  DataBuffer weight_buf(&i, sizeof(i));
  ge_model.SetWeightDataBuf(weight_buf);
  const ge::Buffer weight = ge::Buffer::CopyFrom(&i, sizeof(i));
  ge_model.SetWeight(weight);
  EXPECT_TRUE((ge_model.GetWeightData() == &i));
  ge_model.ClearWeightDataBuf();
  EXPECT_FALSE((ge_model.GetWeightData() == &i));
}

TEST_F(UtestModelHelper, CheckOsCpuInfoAndOppVersion)
{
  auto paths = gert::CreateSceneInfo();
  auto path = paths[0];
  system(("realpath " + path).c_str());

  ModelHelper model_helper;
  std::vector<char> data;
  data.resize(256);
  ModelFileHeader *file_header = reinterpret_cast<ModelFileHeader *>(data.data());
  file_header->need_check_os_cpu_info = static_cast<uint8_t>(OsCpuInfoCheckTyep::NEED_CHECK);
  model_helper.file_header_ = file_header;

  std::string host_env_os = "linux";
  std::string host_env_cpu = "x86_64";
  model_helper.model_ = std::make_shared<GeModel>();
  ge::AttrUtils::SetStr(*(model_helper.model_.get()), "host_env_os", host_env_os);
  ge::AttrUtils::SetStr(*(model_helper.model_.get()), "host_env_cpu", host_env_cpu);
  model_helper.root_model_ = std::make_shared<GeRootModel>();
  ASSERT_EQ(model_helper.CheckOsCpuInfoAndOppVersion(), SUCCESS);

  system(("rm -f " + path).c_str());
}

TEST_F(UtestModelHelper, SoBinSaveToOmRootModel)
{
  ModelHelper model_helper; // 默认为离线模型
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  AttrUtils::SetBool(graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge::ComputeGraphPtr subgraph = std::make_shared<ge::ComputeGraph>("subgraph");
  ge_model->SetGraph(subgraph);
  ge_root_model->subgraph_instance_name_to_model_["graph"] = ge_model;
  ge_model->SetGraph(graph);
  FillModelTaskDef(ge_model);

  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1);
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize' > " + path_config).c_str());

  std::string inner_proto_path = opp_path + kInner + kOpsProtoPath;
  system(("mkdir -p " + inner_proto_path).c_str());
  inner_proto_path += kOpsProto;
  system(("touch " + inner_proto_path).c_str());
  system(("echo 'ops proto:123 ' > " + inner_proto_path).c_str());

  std::string inner_tiling_path = opp_path + kInner + kOpMasterPath;
  system(("mkdir -p " + inner_tiling_path).c_str());
  inner_tiling_path += kOpMaster;
  system(("touch " + inner_tiling_path).c_str());
  system(("echo 'op tiling:456 ' > " + inner_tiling_path).c_str());

  string cpu_info = "x86_64";
  string os_info = "linux";
  EXPECT_EQ(ge_root_model->CheckAndSetNeedSoInOM(), SUCCESS);
  EXPECT_EQ(ge_root_model->GetSoInOmFlag(), 0x8000);
  EXPECT_EQ(model_helper.GetSoBinData(cpu_info, os_info), SUCCESS);

  std::string output_file = "outputfile.om";
  ModelBufferData model;
  bool is_unknown_shape = true;
  std::map<std::string, std::string> options_map;
  options_map["ge.host_env_os"] = os_info;
  options_map["ge.host_env_cpu"] = cpu_info;
  GetThreadLocalContext().SetGlobalOption(options_map);

  EXPECT_EQ(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, is_unknown_shape), SUCCESS);

  output_file += "_linux_x86_64.om";
  system(("rm -rf " + path_vendors).c_str());
  system(("rm -rf " + output_file).c_str());
  system(("rm -rf " + opp_path + kInner).c_str());
}

TEST_F(UtestModelHelper, RepackSoToOm)
{
  ModelHelper model_helper; // 默认为离线模型
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  AttrUtils::SetBool(graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge::ComputeGraphPtr subgraph = std::make_shared<ge::ComputeGraph>("subgraph");
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(subgraph);
  ge_root_model->subgraph_instance_name_to_model_["graph"] = ge_model;
  ge_model->SetGraph(graph);

  GeModelPtr ge_model1 = std::make_shared<GeModel>();
  ge::ComputeGraphPtr graph1 = std::make_shared<ge::ComputeGraph>("graph1");
  ge_model1->SetGraph(graph1);
  ge_root_model->subgraph_instance_name_to_model_["graph1"] = ge_model1;


  {
    domi::ModelTaskDef model_task_def;
    std::shared_ptr<domi::ModelTaskDef> model_task_def_ptr = make_shared<domi::ModelTaskDef>(model_task_def);
    domi::TaskDef *task_def = model_task_def_ptr->add_task();
    ge_model->SetModelTaskDef(model_task_def_ptr);
    ge_model1->SetModelTaskDef(model_task_def_ptr);

    auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
    domi::KernelDefWithHandle *kernel_with_handle = task_def->mutable_kernel_with_handle();
    kernel_with_handle->set_block_dim(32);
    kernel_with_handle->set_args_size(64);
    kernel_with_handle->set_original_kernel_key("");
    kernel_with_handle->set_node_info("");

    string args(64, '1');
    uint16_t args_offset[9] = {0};
    kernel_with_handle->set_args(args.data(), 64);
    domi::KernelContext *context = kernel_with_handle->mutable_context();
    context->set_kernel_type(2);
    context->set_op_index(1);
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1);
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize' > " + path_config).c_str());

  std::string inner_proto_path = opp_path + kInner + kOpsProtoPath;
  system(("mkdir -p " + inner_proto_path).c_str());
  inner_proto_path += kOpsProto;
  system(("touch " + inner_proto_path).c_str());
  system(("echo 'ops proto:123 ' > " + inner_proto_path).c_str());

  std::string inner_tiling_path = opp_path + kInner + kOpMasterPath;
  system(("mkdir -p " + inner_tiling_path).c_str());
  inner_tiling_path += kOpMaster;
  system(("touch " + inner_tiling_path).c_str());
  system(("echo 'op tiling:456 ' > " + inner_tiling_path).c_str());

  string cpu_info = "x86_64";
  string os_info = "linux";
  std::map<std::string, std::string> options_map;
  options_map["ge.host_env_os"] = os_info;
  options_map["ge.host_env_cpu"] = cpu_info;
  GetThreadLocalContext().SetGlobalOption(options_map);

  std::string output_file = "outputfile.om";
  {
    ModelBufferData model;
    ModelHelper model_helper;
    model_helper.SetSaveMode(false);
    GeRootModelPtr ge_root_model = ConstructGeRootModel();
    EXPECT_EQ(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, true), SUCCESS);

    ModelData model_data;
    model_data.model_data = model.data.get();
    model_data.model_len = model.length;
    model_helper.SetRepackSoFlag(true);
    ModelBufferData buffer;
    ASSERT_EQ(model_helper.LoadRootModel(model_data), SUCCESS);
    EXPECT_EQ(model_helper.PackSoToModelData(model_data, output_file, buffer), SUCCESS);
    system("rm -rf outputfile_linux_x86_64.om");
  }

  {
    ModelBufferData model;
    ModelHelper model_helper;
    model_helper.SetSaveMode(false);
    GeRootModelPtr ge_root_model = ConstructGeRootModel(false);
    EXPECT_EQ(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, false), SUCCESS);

    ModelData model_data;
    model_data.model_data = model.data.get();
    model_data.model_len = model.length;
    model_helper.SetRepackSoFlag(true);
    ModelBufferData buffer;
    EXPECT_EQ(model_helper.LoadRootModel(model_data), SUCCESS);
    EXPECT_EQ(model_helper.PackSoToModelData(model_data, output_file, buffer), SUCCESS);
    system(("rm -rf " + output_file).c_str());
  }
  system(("rm -rf " + path_vendors).c_str());
  system(("rm -rf " + opp_path + kInner).c_str());
}

TEST_F(UtestModelHelper, RepackSoToOmWithZeroCopyAddr) {
  ModelHelper helper;
  uint8_t data[64];
  ModelData model_data;
  model_data.model_data = data;
  model_data.model_len = 64;
  GeRootModelPtr ge_root_model = ConstructGeRootModel(false);
  ASSERT_NE(ge_root_model, nullptr);
  helper.root_model_ = ge_root_model;
  ModelBufferData buffer_data;
  EXPECT_EQ(helper.PackSoToModelData(model_data, "./tmp", buffer_data, false), SUCCESS);

  EXPECT_EQ(buffer_data.data.get(), data);
  EXPECT_EQ(buffer_data.length, 64);
}

TEST_F(UtestModelHelper, LoadRootModel_GetFileConstantWeightDirOK) {
  ModelBufferData model;
  ModelHelper model_helper;
  model_helper.SetRepackSoFlag(true);
  model_helper.SetSaveMode(true);
  GeRootModelPtr ge_root_model = ConstructGeRootModel(false);
  std::string output_file = "./outputfile.om";
  EXPECT_EQ(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, false), SUCCESS);

  model_helper.SetSaveMode(false);
  EXPECT_EQ(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, true), SUCCESS);
  ModelData model_data;
  model_data.model_data = model.data.get();
  model_data.model_len = model.length;
  model_data.om_path = output_file;
  EXPECT_EQ(model_helper.LoadRootModel(model_data), SUCCESS);
  const auto root_model = model_helper.GetGeRootModel();
  ASSERT_NE(root_model, nullptr);
  std::string real_om_path = ge::RealPath(output_file.c_str());
  ASSERT_TRUE(!real_om_path.empty());
  std::string weight_path_expected = real_om_path.substr(0, real_om_path.rfind("/") + 1) + "weight/";
  EXPECT_EQ(root_model->GetFileConstantWeightDir(), weight_path_expected);
  system(("rm -rf " + output_file).c_str());
}

TEST_F(UtestModelHelper, LoadRootModel_GetFileConstantWeightDirWithWrongWeightPath) {
  ModelBufferData model;
  ModelHelper model_helper;
  model_helper.SetRepackSoFlag(true);
  model_helper.SetSaveMode(true);
  GeRootModelPtr ge_root_model = ConstructGeRootModel(false);
  std::string output_file = "./outputfile.om";
  EXPECT_EQ(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, false), SUCCESS);

  model_helper.SetSaveMode(false);
  EXPECT_EQ(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, true), SUCCESS);
  ModelData model_data;
  model_data.model_data = model.data.get();
  model_data.model_len = model.length;
  model_data.weight_path = "/home";
  EXPECT_NE(model_helper.LoadRootModel(model_data), SUCCESS);
  system(("rm -rf " + output_file).c_str());
}

TEST_F(UtestModelHelper, SaveOutNodesFromRootGraph) {
  ModelHelper model_helper;  // 默认为离线模型
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  AttrUtils::SetBool(graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge::ComputeGraphPtr subgraph1 = std::make_shared<ge::ComputeGraph>("subgraph1");
  GeModelPtr ge_model1 = std::make_shared<GeModel>();
  ge_model1->SetGraph(subgraph1);
  ge_root_model->subgraph_instance_name_to_model_["subgraph1"] = ge_model1;

  ge::ComputeGraphPtr subgraph2 = std::make_shared<ge::ComputeGraph>("subgraph2");
  GeModelPtr ge_model2 = std::make_shared<GeModel>();
  ge_model2->SetGraph(subgraph2);
  ge_root_model->subgraph_instance_name_to_model_["subgraph2"] = ge_model2;

  GeModelPtr first_ge_model = std::make_shared<GeModel>();
  model_helper.SaveOutNodesFromRootGraph(ge_root_model, first_ge_model);
  std::vector<std::string> out_node_name_get;
  (void)ge::AttrUtils::GetListStr(first_ge_model, ge::ATTR_MODEL_OUT_NODES_NAME, out_node_name_get);
  EXPECT_TRUE(out_node_name_get.empty());

  std::vector<std::string> out_node_name_set;
  out_node_name_set.emplace_back("test");
  (void)ge::AttrUtils::SetListStr(ge_model1, ge::ATTR_MODEL_OUT_NODES_NAME, out_node_name_set);

  model_helper.SaveOutNodesFromRootGraph(ge_root_model, first_ge_model);
  (void)ge::AttrUtils::GetListStr(first_ge_model, ge::ATTR_MODEL_OUT_NODES_NAME, out_node_name_get);
  EXPECT_TRUE(!out_node_name_get.empty());
}


TEST_F(UtestModelHelper, SoBinSaveToOmModel)
{
  ModelHelper model_helper; // 默认为离线模型
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  AttrUtils::SetBool(graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge::ComputeGraphPtr subgraph = std::make_shared<ge::ComputeGraph>("subgraph");
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(subgraph);
  ge_root_model->subgraph_instance_name_to_model_["graph"] = ge_model;
  ge_model->SetGraph(graph);

  {
    domi::ModelTaskDef model_task_def;
    std::shared_ptr<domi::ModelTaskDef> model_task_def_ptr = make_shared<domi::ModelTaskDef>(model_task_def);
    domi::TaskDef *task_def = model_task_def_ptr->add_task();
    ge_model->SetModelTaskDef(model_task_def_ptr);

    auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
    domi::KernelDefWithHandle *kernel_with_handle = task_def->mutable_kernel_with_handle();
    kernel_with_handle->set_original_kernel_key("");
    kernel_with_handle->set_node_info("");
    kernel_with_handle->set_block_dim(32);
    kernel_with_handle->set_args_size(64);
    string args(64, '1');
    kernel_with_handle->set_args(args.data(), 64);
    domi::KernelContext *context = kernel_with_handle->mutable_context();
    context->set_op_index(1);
    context->set_kernel_type(2);    // ccKernelType::TE
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1);
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize' > " + path_config).c_str());

  std::string inner_proto_path = opp_path + kInner + kOpsProtoPath;
  system(("mkdir -p " + inner_proto_path).c_str());
  inner_proto_path += kOpsProto;
  system(("touch " + inner_proto_path).c_str());
  system(("echo 'ops proto:123 ' > " + inner_proto_path).c_str());

  std::string inner_tiling_path = opp_path + kInner + kOpMasterPath;
  system(("mkdir -p " + inner_tiling_path).c_str());
  inner_tiling_path += kOpMaster;
  system(("touch " + inner_tiling_path).c_str());
  system(("echo 'op tiling:456 ' > " + inner_tiling_path).c_str());

  string cpu_info = "x86_64";
  string os_info = "linux";
  EXPECT_EQ(ge_root_model->CheckAndSetNeedSoInOM(), SUCCESS);
  EXPECT_EQ(ge_root_model->GetSoInOmFlag(), 0x8000);
  EXPECT_EQ(model_helper.GetSoBinData(cpu_info, os_info), SUCCESS);

  std::string output_file = "static_outputfile.om";
  ModelBufferData model;
  bool is_unknown_shape = false;

  std::map<std::string, std::string> options_map;
  options_map["ge.host_env_os"] = os_info;
  options_map["ge.host_env_cpu"] = cpu_info;
  GetThreadLocalContext().SetGlobalOption(options_map);

  EXPECT_EQ(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, is_unknown_shape), SUCCESS);

  output_file += "_linux_x86_64.om";
  system(("rm -rf " + path_vendors).c_str());
  system(("rm -rf " + inner_proto_path).c_str());
  system(("rm -rf " + output_file).c_str());
}

TEST_F(UtestModelHelper, LoadOpSoBinSuccess)
{
  OmFileLoadHelper load_helper;
  ModelHelper model_helper; // 默认为离线
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  AttrUtils::SetBool(graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);

  load_helper.is_inited_ = true;
  OmFileContext cur_ctx;
  ModelPartition so_patition;
  so_patition.type = ModelPartitionType::SO_BINS;
  auto so_data = std::unique_ptr<char[]>(new(std::nothrow) char[20]);
  so_patition.data = reinterpret_cast<uint8_t*>(so_data.get());
  so_patition.size = 20;
  cur_ctx.partition_datas_.push_back(so_patition);
  load_helper.model_contexts_.push_back(cur_ctx);
  EXPECT_EQ(model_helper.LoadOpSoBin(load_helper, ge_root_model), SUCCESS);
}

TEST_F(UtestModelHelper, LoadTilingDataSuccess) {
  OmFileLoadHelper load_helper;
  ModelHelper model_helper;  // 默认为离线
  load_helper.is_inited_ = true;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  AttrUtils::SetBool(graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);

  auto op_desc = std::make_shared<OpDesc>("Test", "Test");
  auto node = graph->AddNode(op_desc);
  ASSERT_TRUE(node != nullptr);

  std::string tiling_data = "hahahaha";
  std::shared_ptr<optiling::utils::OpRunInfo> add_run_info = std::make_shared<optiling::utils::OpRunInfo>();
  add_run_info->AddTilingData(tiling_data.data(), tiling_data.size());
  node->GetOpDescBarePtr()->SetExtAttr(ATTR_NAME_OP_RUN_INFO, add_run_info);

  HostResourceCenter center = HostResourceCenter();
  center.TakeOverHostResources(graph);

  uint8_t *data{data};
  std::size_t tiling_size;
  HostResourceSerializer serializer;
  ASSERT_EQ(serializer.SerializeTilingData(center, data, tiling_size), SUCCESS);

  OmFileContext cur_ctx;
  ModelPartition tiling_partition;
  tiling_partition.type = ModelPartitionType::TILING_DATA;
  tiling_partition.data = reinterpret_cast<uint8_t *>(data);
  tiling_partition.size = tiling_size;
  cur_ctx.partition_datas_.push_back(tiling_partition);
  load_helper.model_contexts_.push_back(cur_ctx);
  EXPECT_EQ(model_helper.LoadTilingData(load_helper, ge_root_model), SUCCESS);
}

TEST_F(UtestModelHelper, NotContainSoBin)
{
  OmFileLoadHelper load_helper;
  ModelHelper model_helper;
  load_helper.is_inited_ = true;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  AttrUtils::SetBool(graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  EXPECT_EQ(model_helper.LoadOpSoBin(load_helper, ge_root_model), SUCCESS);
}

TEST_F(UtestModelHelper, LoadOpSoBinDataFail)
{
  OmFileLoadHelper load_helper;
  ModelHelper model_helper;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  AttrUtils::SetBool(graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);

  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_root_model->subgraph_instance_name_to_model_["graph"] = ge_model;

  load_helper.is_inited_ = true;
  OmFileContext cur_ctx;
  ModelPartition so_patition;
  so_patition.type = ModelPartitionType::SO_BINS;
  so_patition.data = nullptr;
  so_patition.size = 0;
  cur_ctx.partition_datas_.push_back(so_patition);
  load_helper.model_contexts_.push_back(cur_ctx);
  model_helper.model_ = ge_model;
  EXPECT_EQ(model_helper.LoadOpSoBin(load_helper, ge_root_model), SUCCESS);
}

TEST_F(UtestModelHelper, GetBinDataSuccess) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1);
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize' > " + path_config).c_str());

  std::string inner_proto_path = opp_path + kInner + kOpsProtoPath;
  system(("mkdir -p " + inner_proto_path).c_str());
  inner_proto_path += kOpsProto;
  system(("touch " + inner_proto_path).c_str());
  system(("echo 'ops proto:123 ' > " + inner_proto_path).c_str());

  std::string inner_tiling_path = opp_path + kInner + kOpMasterPath;
  system(("mkdir -p " + inner_tiling_path).c_str());
  inner_tiling_path += kOpMaster;
  system(("touch " + inner_tiling_path).c_str());
  system(("echo 'op tiling:456 ' > " + inner_tiling_path).c_str());

  ModelHelper model_helper;
  string cpu_info = "x86_64";
  string os_info = "linux";
  auto ret = model_helper.GetSoBinData(cpu_info, os_info);
  EXPECT_EQ(ret, SUCCESS);

  system(("rm -rf " + path_vendors).c_str());
}

TEST_F(UtestModelHelper, SoBinSaveToOmModel_CPU_OS_EMPTY)
{
  ModelHelper model_helper; // 默认为离线模型
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  AttrUtils::SetBool(graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_root_model->subgraph_instance_name_to_model_["graph"] = ge_model;
  ge_model->SetGraph(graph);

  {
    domi::ModelTaskDef model_task_def;
    std::shared_ptr<domi::ModelTaskDef> model_task_def_ptr = make_shared<domi::ModelTaskDef>(model_task_def);
    domi::TaskDef *task_def = model_task_def_ptr->add_task();
    ge_model->SetModelTaskDef(model_task_def_ptr);

    auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
    domi::KernelDefWithHandle *kernel_with_handle = task_def->mutable_kernel_with_handle();
    kernel_with_handle->set_original_kernel_key("");
    kernel_with_handle->set_node_info("");
    kernel_with_handle->set_block_dim(32);
    kernel_with_handle->set_args_size(64);
    string args(64, '1');
    kernel_with_handle->set_args(args.data(), 64);
    domi::KernelContext *context = kernel_with_handle->mutable_context();
    context->set_op_index(1);
    context->set_kernel_type(2);    // ccKernelType::TE
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1);
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string inner_proto_path = opp_path + kInner + kOpsProtoPath;
  system(("mkdir -p " + inner_proto_path).c_str());
  inner_proto_path += kOpsProto;
  system(("touch " + inner_proto_path).c_str());
  system(("echo 'ops proto:123 ' > " + inner_proto_path).c_str());

  std::string inner_tiling_path = opp_path + kInner + kOpMasterPath;
  system(("mkdir -p " + inner_tiling_path).c_str());
  inner_tiling_path += kOpMaster;
  system(("touch " + inner_tiling_path).c_str());
  system(("echo 'op tiling:456 ' > " + inner_tiling_path).c_str());

  string cpu_info = "x86_64";
  string os_info = "linux";
  EXPECT_EQ(ge_root_model->CheckAndSetNeedSoInOM(), SUCCESS);
  EXPECT_EQ(ge_root_model->GetSoInOmFlag(), 0x8000);
  EXPECT_EQ(model_helper.GetSoBinData(cpu_info, os_info), SUCCESS);

  std::map<std::string, std::string> options_map;
  options_map["ge.host_env_os"] = "";
  options_map["ge.host_env_cpu"] = "";
  GetThreadLocalContext().SetGlobalOption(options_map);

  std::string output_file = "static_outputfile.om";
  ModelBufferData model;
  bool is_unknown_shape = false;

  EXPECT_EQ(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, is_unknown_shape), SUCCESS);
}

TEST_F(UtestModelHelper, SoBinSaveToOmRootModelErrByFileNameTooLong)
{
  ModelHelper model_helper; // 默认为离线模型
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  AttrUtils::SetBool(graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_root_model->subgraph_instance_name_to_model_["graph"] = ge_model;
  ge_model->SetGraph(graph);

  {
    domi::ModelTaskDef model_task_def;
    std::shared_ptr<domi::ModelTaskDef> model_task_def_ptr = make_shared<domi::ModelTaskDef>(model_task_def);
    domi::TaskDef *task_def = model_task_def_ptr->add_task();
    ge_model->SetModelTaskDef(model_task_def_ptr);

    auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
    domi::KernelDefWithHandle *kernel_with_handle = task_def->mutable_kernel_with_handle();
    kernel_with_handle->set_original_kernel_key("");
    kernel_with_handle->set_node_info("");
    kernel_with_handle->set_block_dim(32);
    kernel_with_handle->set_args_size(64);
    string args(64, '1');
    kernel_with_handle->set_args(args.data(), 64);
    domi::KernelContext *context = kernel_with_handle->mutable_context();
    context->set_op_index(1);
    context->set_kernel_type(2);    // ccKernelType::TE
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1);
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string inner_proto_path = opp_path + kInner + kOpsProtoPath;
  system(("mkdir -p " + inner_proto_path).c_str());
  inner_proto_path += kOpsProto;
  system(("touch " + inner_proto_path).c_str());
  system(("echo 'ops proto:123 ' > " + inner_proto_path).c_str());

  std::string inner_tiling_path = opp_path + kInner + kOpMasterPath;
  system(("mkdir -p " + inner_tiling_path).c_str());
  inner_tiling_path += kOpMaster;
  system(("touch " + inner_tiling_path).c_str());
  system(("echo 'op tiling:456 ' > " + inner_tiling_path).c_str());

  string cpu_info = "x86_64";
  string os_info = "linux";
  EXPECT_EQ(ge_root_model->CheckAndSetNeedSoInOM(), SUCCESS);
  EXPECT_EQ(ge_root_model->GetSoInOmFlag(), 0x8000);
  EXPECT_EQ(model_helper.GetSoBinData(cpu_info, os_info), SUCCESS);

  std::string output_file = "test12345678910_12345678910_12345678910_12345678910_12345678910_";
  output_file.append("12345678910_12345678910_12345678910_12345678910_12345678910_");
  output_file.append("12345678910_12345678910_12345678910_12345678910_12345678910_");
  output_file.append("12345678910_12345678910__12345678910_12345678910_12345678910.om");

  ModelBufferData model;
  bool is_unknown_shape = true;
  std::map<std::string, std::string> options_map;
  options_map["ge.host_env_os"] = os_info;
  options_map["ge.host_env_cpu"] = cpu_info;
  GetThreadLocalContext().SetGlobalOption(options_map);

  EXPECT_NE(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, is_unknown_shape), SUCCESS);
  system(("rm -rf " + opp_path + kInner).c_str());
}

TEST_F(UtestModelHelper, GetSoBinData_fail)
{
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1);
  opp_path += "/temp/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=mdc' > " + path_config).c_str());

  std::string inner_proto_path = opp_path + kInner + kOpsProtoPath;
  system(("mkdir -p " + inner_proto_path).c_str());
  inner_proto_path += kOpsProto;
  system(("touch " + inner_proto_path).c_str());
  system(("echo 'ops proto:123 ' > " + inner_proto_path).c_str());

  std::string vender_tiling_path = opp_path + "vendors/mdc/" + kOpMasterPath;
  system(("mkdir -p " + vender_tiling_path).c_str());
  vender_tiling_path += kOpMaster;
  system(("touch " + vender_tiling_path).c_str());
  system(("echo 'op tiling:456 ' > " + vender_tiling_path).c_str());
  system(("echo 'compiler_version=6.3.T5.0.B121' > " + opp_path + "vendors/mdc/" + "/version.info").c_str());

  std::string customize_tiling_path = opp_path + "customize" + kOpMasterPath;
  system(("mkdir -p " + customize_tiling_path).c_str());
  customize_tiling_path += kOpMaster;
  system(("touch " + customize_tiling_path).c_str());
  system(("echo 'op tiling:456 ' > " + customize_tiling_path).c_str());
  system(("echo 'compiler_version=6.4.T5.0.B121' > " + opp_path + "customize" + "/version.info").c_str());
  mmSetEnv(kEnvNameCustom, (opp_path + "customize").c_str(), 1);

  string cpu_info = "x86_64";
  string os_info = "linux";
  ModelHelper model_helper;
  EXPECT_EQ(model_helper.GetSoBinData(cpu_info, os_info), PARAM_INVALID);

  system(("rm -rf " + inner_proto_path).c_str());
  EXPECT_EQ(model_helper.GetSoBinData(cpu_info, os_info), PARAM_INVALID);
  GeModelPtr ge_model = std::make_shared<GeModel>();
  model_helper.SetModelCompilerVersion(ge_model);

  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestModelHelper, GetHardwareInfo_no_device) {
  dlog_setlevel(0, 0, 0);
  g_runtime_stub_mock = "rtGetDevice";

  std::map<std::string, std::string> options;
  options[SOC_VERSION] = "Ascend910";
  ModelHelper model_helper;
  EXPECT_EQ(model_helper.GetHardwareInfo(options), SUCCESS);
  EXPECT_NE(options.size(), 0);
  g_runtime_stub_mock = "";
  dlog_setlevel(0, 3, 0);
}

TEST_F(UtestModelHelper, GetHardwareInfo_no_device_get_count_from_rts_failed) {
  dlog_setlevel(0, 0, 0);
  class MockAclRuntimeStub: public AclRuntimeStub {
    aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t *value) override{
      return -1;
    }
  };
  MockAclRuntimeStub mock_acl_runtime_stub;
  AclRuntimeStub::Install(&mock_acl_runtime_stub);
  std::map<std::string, std::string> options;
  options[SOC_VERSION] = "Ascend910";
  ModelHelper model_helper;
  EXPECT_NE(model_helper.GetHardwareInfo(options), SUCCESS);
  dlog_setlevel(0, 3, 0);
  AclRuntimeStub::UnInstall(&mock_acl_runtime_stub);
}

TEST_F(UtestModelHelper, GetHardwareInfo_device0) {
  g_runtime_stub_mock = "rtGetDevice2";
  const char *const kVectorcoreNum = "ge.vectorcoreNum";

  std::map<std::string, std::string> options;
  options[AICORE_NUM] = "2";
  options[kVectorcoreNum] = "2";
  options[SOC_VERSION] = "Ascend910";
  ModelHelper model_helper;
  EXPECT_EQ(model_helper.GetHardwareInfo(options), SUCCESS);

  EXPECT_NE(options.size(), 0);
  g_runtime_stub_mock = "";
}

TEST_F(UtestModelHelper, GetSoBinData_upgraded_opp_success) {
  std::vector<std::string> paths;
  gert::CreateBuiltInSplitAndUpgradedSo(paths);
  std::string os_info{"linux"};
  std::string cpu_info{"x86_64"};
  ModelHelper model_helper;
  EXPECT_EQ(model_helper.GetSoBinData(cpu_info, os_info), SUCCESS);
  EXPECT_EQ(model_helper.op_so_store_.kernels_.size(), 4U);
  for (const auto &path : paths) {
    ge::PathUtils::RemoveDirectories(path);
  }
}

TEST_F(UtestModelHelper, SaveOpMasterDevice_BuiltIn_Success) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "/test_tmp/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  std::string inner_op_master = opp_path + "built_in/op_master_device/lib/";
  system(("mkdir -p " + inner_op_master).c_str());
  inner_op_master += "Ascend-V7.6-libopmaster.so";
  system(("touch " + inner_op_master).c_str());
  system(("echo 'Ascend-V7.6-libopmaster' > " + inner_op_master).c_str());

  ModelBufferData model;
  ModelHelper model_helper;
  model_helper.SetSaveMode(true);
  GeRootModelPtr ge_root_model =
      ConstructGeRootModel(false, ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL, ccKernelType::AI_CPU, inner_op_master);
  EXPECT_EQ(ge_root_model->CheckAndSetNeedSoInOM(), SUCCESS);
  EXPECT_EQ(ge_root_model->GetSoInOmFlag(), 0x4000);

  std::string output_file = opp_path + "/output.om";
  EXPECT_EQ(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, false), SUCCESS);

  ge::ModelParserBase base;
  ge::ModelData model_data;
  EXPECT_EQ(base.LoadFromFile(output_file.c_str(), -1, model_data), SUCCESS);
  EXPECT_EQ(model_helper.LoadRootModel(model_data), SUCCESS);
  if (model_data.model_data != nullptr) {
    delete[] reinterpret_cast<char_t *>(model_data.model_data);
  }

  const auto &root_model = model_helper.GetGeRootModel();
  const auto &so_list = root_model->GetAllSoBin();
  EXPECT_EQ(so_list.size(), 1UL);
  EXPECT_EQ(so_list[0UL]->GetSoBinType(), SoBinType::kOpMasterDevice);
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestModelHelper, SaveOpMasterDevice_WithSpaceRegistry_Success) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "/test_tmp/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  std::string inner_op_master = opp_path + "built_in/op_master_device/lib/";
  system(("mkdir -p " + inner_op_master).c_str());
  inner_op_master += "Ascend-V7.6-libopmaster.so";
  system(("touch " + inner_op_master).c_str());
  system(("echo 'Ascend-V7.6-libopmaster' > " + inner_op_master).c_str());

  std::string inner_proto_path = opp_path + kInner + kOpsProtoPath;
  system(("mkdir -p " + inner_proto_path).c_str());
  inner_proto_path += kOpsProto;
  system(("touch " + inner_proto_path).c_str());
  system(("echo 'ops proto' > " + inner_proto_path).c_str());

  std::string inner_tiling_path = opp_path + kInner + kOpMasterPath;
  system(("mkdir -p " + inner_tiling_path).c_str());
  inner_tiling_path += kOpMaster;
  system(("touch " + inner_tiling_path).c_str());
  system(("echo 'op tiling ' > " + inner_tiling_path).c_str());

  ModelBufferData model;
  ModelHelper model_helper;
  model_helper.SetSaveMode(true);
  GeRootModelPtr ge_root_model =
      ConstructGeRootModel(true, ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL, ccKernelType::AI_CPU, inner_op_master);
  EXPECT_EQ(ge_root_model->CheckAndSetNeedSoInOM(), SUCCESS);
  EXPECT_EQ(ge_root_model->GetSoInOmFlag(), 0xc000);

  std::map<std::string, std::string> options_map;
  options_map["ge.host_env_os"] = "linux";
  options_map["ge.host_env_cpu"] = "x86_64";
  GetThreadLocalContext().SetGlobalOption(options_map);
  std::string output_file = opp_path + "/output.om";
  EXPECT_EQ(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, false), SUCCESS);

  ge::ModelParserBase base;
  ge::ModelData model_data;
  EXPECT_EQ(base.LoadFromFile((opp_path + "output_linux_x86_64.om").c_str(), -1, model_data), SUCCESS);
  EXPECT_EQ(model_helper.LoadRootModel(model_data), SUCCESS);
  if (model_data.model_data != nullptr) {
    delete[] reinterpret_cast<char_t *>(model_data.model_data);
  }

  const auto &root_model = model_helper.GetGeRootModel();
  const auto &so_list = root_model->GetAllSoBin();
  EXPECT_EQ(so_list.size(), 3UL);
  unordered_map<SoBinType, uint32_t> res;
  res.emplace(SoBinType::kSpaceRegistry, 0U);
  res.emplace(SoBinType::kOpMasterDevice, 0U);
  std::for_each(so_list.begin(), so_list.end(), [&res](const OpSoBinPtr &so_bin) { res[so_bin->GetSoBinType()]++; });
  EXPECT_EQ(res[SoBinType::kSpaceRegistry], 2U);
  EXPECT_EQ(res[SoBinType::kOpMasterDevice], 1U);
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestModelHelper, SaveOpMasterDevice_So_Name_Invalid) {
  GeRootModelPtr ge_root_model = ConstructGeRootModel(false, ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL, ccKernelType::AI_CPU);
  EXPECT_NE(ge_root_model->CheckAndSetNeedSoInOM(), SUCCESS);
}

TEST_F(UtestModelHelper, CheckOsCpuInfoAndOppVersion_Success) {
  ModelHelper model_helper;
  std::vector<char> data;
  data.resize(256);
  ModelFileHeader *file_header = reinterpret_cast<ModelFileHeader *>(data.data());
  file_header->need_check_os_cpu_info = static_cast<uint8_t>(OsCpuInfoCheckTyep::NO_CHECK);
  model_helper.file_header_ = file_header;
  model_helper.is_unknown_shape_model_ = true;
  gert::GertRuntimeStub stub;
  stub.GetSlogStub().Clear();
  stub.GetSlogStub().SetLevelDebug();
  EXPECT_EQ(model_helper.CheckOsCpuInfoAndOppVersion(), SUCCESS);
  ASSERT_TRUE(stub.GetSlogStub().FindLog(DLOG_DEBUG, "Check opp version[] success") >= 0);
}

TEST_F(UtestModelHelper, UpdateSessionGraphId) {
  ModelHelper model_helper;
  bool refreshed = false;
  auto graph = gert::ShareGraph::BuildWithKnownSubgraphWithTwoConst();
  auto ret = model_helper.UpdateSessionGraphId(graph, "1", refreshed);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestModelHelper, SaveAndLoadOfflineAutofuseSo) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "/test_tmp/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  std::string inner_op_master = opp_path + "built_in/op_master_device/lib/";
  system(("mkdir -p " + inner_op_master).c_str());
  inner_op_master += "Ascend-V7.6-libopmaster.so";
  system(("touch " + inner_op_master).c_str());
  system(("echo 'Ascend-V7.6-libopmaster' > " + inner_op_master).c_str());

  ModelBufferData model;
  ModelHelper model_helper;
  model_helper.SetSaveMode(true);
  GeRootModelPtr ge_root_model =
      ConstructGeRootModel(false, ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL, ccKernelType::AI_CPU, inner_op_master);

  auto ge_root_graph = ge_root_model->GetRootGraph();

  OpDescBuilder op_desc_builder("test", "AscBackend");
  const auto &op_desc = op_desc_builder.Build();
  auto node = ge_root_graph->AddNode(op_desc);
  node->SetOwnerComputeGraph(ge_root_graph);
  auto autofuse_stub_so = __FILE__;
  std::cout << "bin path: " << autofuse_stub_so << std::endl;
  auto nodes = ge_root_graph->GetAllNodesPtr();
  for (auto n : nodes) {
    cout << n->GetName() << endl;
    (void)ge::AttrUtils::SetStr(n->GetOpDesc(), "bin_file_path", autofuse_stub_so);
  }

  EXPECT_EQ(ge_root_model->CheckAndSetNeedSoInOM(), SUCCESS);
  EXPECT_EQ(ge_root_model->GetSoInOmFlag(), 0x6000);

  std::string output_file = opp_path + "/output.om";
  EXPECT_EQ(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, false), SUCCESS);

  ge::ModelParserBase base;
  ge::ModelData model_data;
  EXPECT_EQ(base.LoadFromFile(output_file.c_str(), -1, model_data), SUCCESS);
  EXPECT_EQ(model_helper.LoadRootModel(model_data), SUCCESS);
  if (model_data.model_data != nullptr) {
    delete[] reinterpret_cast<char_t *>(model_data.model_data);
  }

  OmFileLoadHelper load_helper;
  load_helper.is_inited_ = true;
  OmFileContext cur_ctx;
  ModelPartition so_patition;
  so_patition.type = ModelPartitionType::SO_BINS;
  auto so_data = std::unique_ptr<char[]>(new(std::nothrow) char[20]);
  so_patition.data = reinterpret_cast<uint8_t*>(so_data.get());
  so_patition.size = 20;
  cur_ctx.partition_datas_.push_back(so_patition);
  load_helper.model_contexts_.push_back(cur_ctx);
  EXPECT_EQ(model_helper.LoadOpSoBin(load_helper, ge_root_model), SUCCESS);

  const auto &root_model = model_helper.GetGeRootModel();
  const auto &so_list = root_model->GetAllSoBin();
  EXPECT_EQ(so_list.size(), 2UL);
  EXPECT_EQ(so_list[0UL]->GetSoBinType(), SoBinType::kOpMasterDevice);
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestModelHelper, SaveToOm_for_SubPkg_Opp) {
  auto options = GetThreadLocalContext().GetAllGlobalOptions();
  std::vector<std::string> paths;
  gert::CreateBuiltInSubPkgSo(paths);
  for (const auto &path : paths) {
    GELOGI("Subpkg so:%s", path.c_str());
  }

  GeModelPtr ge_model = ge::MakeShared<GeModel>();
  ComputeGraphPtr graph = ge::MakeShared<ComputeGraph>("g1");
  ge_model->SetGraph(graph);
  std::string output_file{"subpkg_opp.om"};
  ModelBufferData buffer_data;
  GeRootModelPtr ge_root_model = ge::MakeShared<GeRootModel>();
  ComputeGraphPtr root_graph = ge::MakeShared<ComputeGraph>("subgraph");
  (void)AttrUtils::SetBool(root_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
  ge_root_model->SetRootGraph(root_graph);
  EXPECT_EQ(ge_root_model->CheckAndSetNeedSoInOM(), SUCCESS);
  EXPECT_EQ(ge_root_model->GetSoInOmFlag(), 0x8000);
  FillModelTaskDef(ge_model);
  std::map<string, string> env_options;
  env_options["ge.host_env_os"] = "linux";
  env_options["ge.host_env_cpu"] = "x86_64";
  (void)GetThreadLocalContext().SetGlobalOption(env_options);

  ModelHelper model_helper;
  (void)model_helper.SaveToOmModel(ge_model, output_file, buffer_data, ge_root_model);

  const auto so_bins = model_helper.op_so_store_.GetSoBin();
  EXPECT_EQ(so_bins.size(), 4);
  EXPECT_NE(so_bins[0]->GetSoName().find("libopgraph_math.so"), std::string::npos);
  EXPECT_NE(so_bins[1]->GetSoName().find("libophost_math.so"), std::string::npos);
  EXPECT_NE(so_bins[2]->GetSoName().find("libopgraph_math.so"), std::string::npos);
  EXPECT_NE(so_bins[3]->GetSoName().find("libophost_math.so"), std::string::npos);
  (void)GetThreadLocalContext().SetGlobalOption(options);
  for (const auto &path : paths) {
    ge::PathUtils::RemoveDirectories(path);
  }
}

// Testable ModelHelper subclass to expose private methods for testing
class TestableModelHelper : public ModelHelper {
 public:
  // Expose ShouldCompress method for testing
  bool TestShouldCompress() const {
    return ShouldCompress();
  }

  // Expose GetCompressionModeString method for testing
  const char* TestGetCompressionModeString() const {
    return attr_compression_enabled_ ? "enable" : "disable";
  }
};

// Test 1: enabled=true + offline=true + need_compress=true → should compress
TEST_F(UtestModelHelper, AttrCompression_EnabledWithOffline_ShouldCompress) {
  TestableModelHelper helper;
  helper.SetSaveMode(true);  // is_offline_ = true
  helper.SetAttrCompressionEnabled(true);  // attr_compression_enabled_ = true

  EXPECT_TRUE(helper.TestShouldCompress());
  EXPECT_STREQ(helper.TestGetCompressionModeString(), "enable");
}

// Test 2: enabled=true + offline=false → should not compress
TEST_F(UtestModelHelper, AttrCompression_EnabledWithOnline_ShouldNotCompress) {
  TestableModelHelper helper;
  helper.SetSaveMode(false);  // is_offline_ = false
  helper.SetAttrCompressionEnabled(true);  // attr_compression_enabled_ = true

  EXPECT_FALSE(helper.TestShouldCompress());
  EXPECT_STREQ(helper.TestGetCompressionModeString(), "enable");
}

// Test 3: enabled=false + offline=true → should not compress (disabled overrides offline)
TEST_F(UtestModelHelper, AttrCompression_DisabledWithOffline_ShouldNotCompress) {
  TestableModelHelper helper;
  helper.SetSaveMode(true);   // is_offline_ = true
  helper.SetAttrCompressionEnabled(false);  // attr_compression_enabled_ = false

  EXPECT_FALSE(helper.TestShouldCompress());
  EXPECT_STREQ(helper.TestGetCompressionModeString(), "disable");
}

// Test 4: enabled=false + offline=false → should not compress
TEST_F(UtestModelHelper, AttrCompression_DisabledWithOnline_ShouldNotCompress) {
  TestableModelHelper helper;
  helper.SetSaveMode(false);  // is_offline_ = false
  helper.SetAttrCompressionEnabled(false);  // attr_compression_enabled_ = false

  EXPECT_FALSE(helper.TestShouldCompress());
  EXPECT_STREQ(helper.TestGetCompressionModeString(), "disable");
}

// Test 5: ConfigureFromOptions with valid values (only "true" and "false" are accepted)
TEST_F(UtestModelHelper, AttrCompression_ConfigureFromOptions_ValidValues) {
  TestableModelHelper helper;

  // Test "true"
  EXPECT_EQ(helper.ConfigureAttrCompressionMode("true"), SUCCESS);
  EXPECT_STREQ(helper.TestGetCompressionModeString(), "enable");

  // Test "false"
  EXPECT_EQ(helper.ConfigureAttrCompressionMode("false"), SUCCESS);
  EXPECT_STREQ(helper.TestGetCompressionModeString(), "disable");
}

// Test 8: ConfigureFromOptions with other invalid values (auto, enable, disable, 1, 0)
TEST_F(UtestModelHelper, AttrCompression_ConfigureFromOptions_OtherInvalidValues) {
  TestableModelHelper helper;

  // Test "enable" - no longer supported
  EXPECT_EQ(helper.ConfigureAttrCompressionMode("enable"), PARAM_INVALID);

  // Test "disable" - no longer supported
  EXPECT_EQ(helper.ConfigureAttrCompressionMode("disable"), PARAM_INVALID);

  // Test "1" - no longer supported
  EXPECT_EQ(helper.ConfigureAttrCompressionMode("1"), PARAM_INVALID);

  // Test "0" - no longer supported
  EXPECT_EQ(helper.ConfigureAttrCompressionMode("0"), PARAM_INVALID);
}

// Test 6: ConfigureFromOptions with empty options (use default)
TEST_F(UtestModelHelper, AttrCompression_ConfigureFromOptions_EmptyOptions) {
  TestableModelHelper helper;

  std::map<std::string, std::string> options;
  EXPECT_EQ(helper.ConfigureAttrCompressionMode(""), PARAM_INVALID);
  EXPECT_STREQ(helper.TestGetCompressionModeString(), "enable");  // Default value is true
}
}  // namespace ge

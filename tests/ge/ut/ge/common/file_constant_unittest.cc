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
#include <fstream>

#include "macro_utils/dt_public_scope.h"
#include "common/file_constant_utils/file_constant_utils.h"
#include "macro_utils/dt_public_unscope.h"

#include "common/helper/file_saver.h"
#include "graph/passes/graph_builder_utils.h"
#include "graph/ge_local_context.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/manager/graph_external_weight_manager.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/constant_utils.h"
#include "graph/ge_context.h"
#include "depends/mmpa/src/mmpa_stub.h"

namespace ge {
namespace {

class MockMmpa : public MmpaStubApiGe {
 public:
  INT32 Open2(const CHAR *path_name, INT32 flags, MODE mode) override {
    return -1;
  }
};
class MockMmpaForFlockFailed : public MmpaStubApiGe {
 public:
  INT32 Open2(const CHAR *path_name, INT32 flags, MODE mode) override {
    return INT32_MAX;
  }
};
}
static ge::OpDescPtr CreateOpDesc(string name = "", string type = "") {
  auto op_desc = std::make_shared<ge::OpDesc>(name, type);
  op_desc->SetStreamId(0);
  op_desc->SetId(0);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetInputOffset({100, 200});
  op_desc->SetOutputOffset({100, 200});
  return op_desc;
}

namespace fileconstant {
class UtestFileConstantUtilTransfer : public testing::Test {
 protected:
  void SetUp() {
    ExternalWeightManagerPool::Instance().Destroy();
  }
  void TearDown() {
    ExternalWeightManagerPool::Instance().Destroy();
  }
};

TEST_F(UtestFileConstantUtilTransfer, GetFilePathFromOptionOK) {
  std::map<std::string, std::string> options;
  options["ge.exec.value_bins"] =
      "{\"value_bins\":[{\"value_bin_id\":\"vector_search_buchet_value_bin\", \"value_bin_file\":\"hello.bin\"}]}";
  ge::GetThreadLocalContext().SetGraphOption(options);
  std::map<std::string, std::string> file_id_and_path_map;
  EXPECT_EQ(FileConstantUtils::GetFileIdToPathMapFromOption(file_id_and_path_map), SUCCESS);
  EXPECT_EQ(file_id_and_path_map.size(), 1);
}

TEST_F(UtestFileConstantUtilTransfer, GetFilePathFromOptionFailed) {
  std::map<std::string, std::string> options;
  options["ge.exec.value_bins"] = "{\"value_bins\":";
  ge::GetThreadLocalContext().SetGraphOption(options);
  std::map<std::string, std::string> file_id_and_path_map;
  EXPECT_NE(FileConstantUtils::GetFileIdToPathMapFromOption(file_id_and_path_map), SUCCESS);
}

TEST_F(UtestFileConstantUtilTransfer, CopyOneWeightFromFileOK) {
  GetContext().SetSessionId(0U);
  std::unique_ptr<char[]> buf(new char[2048]);
  string file_name = "tmp_weight_pid/no_find_file";
  size_t file_const_size = 100;
  size_t left_size = 0;
  Status ret = FileConstantUtils::CopyOneWeightFromFile((void *)buf.get(), file_name, 0U, file_const_size, left_size);
  EXPECT_EQ(ret, PARAM_INVALID);
  left_size = file_const_size;
  ret = FileConstantUtils::CopyOneWeightFromFile((void *)buf.get(), "", 0U, file_const_size, left_size);
  EXPECT_EQ(ret, FAILED);
  ret = FileConstantUtils::CopyOneWeightFromFile((void *)buf.get(), file_name, 0U, file_const_size, left_size);
  EXPECT_EQ(ret, GRAPH_FAILED);

  std::unique_ptr<float[]> float_buf(new float[file_const_size / sizeof(float)]);
  file_name = "tmp_weight_pid/test_copy_one_weight.bin";
  std::ofstream out1(file_name, std::ios::binary);
  if (!out1.is_open()) {
    return;
  }
  out1.write((char *)float_buf.get(), file_const_size);
  out1.close();

  ret = FileConstantUtils::CopyOneWeightFromFile((void *)buf.get(), file_name, 0U, file_const_size, left_size);
  EXPECT_EQ(ret, SUCCESS);

  (void)remove("tmp_weight_pid/test_copy_one_weight.bin");
}

TEST_F(UtestFileConstantUtilTransfer, GetFilePathOK) {
  std::map<std::string, std::string> options;
  options["ge.exec.value_bins"] =
      "{\"value_bins\":[{\"value_bin_id\":\"vector_search_buchet_value_bin\", \"value_bin_file\":\"hello.bin\"}]}";
  ge::GetThreadLocalContext().SetGraphOption(options);
  std::map<std::string, std::string> file_id_and_path_map;
  FileConstantUtils::GetFileIdToPathMapFromOption(file_id_and_path_map);
  OpDescPtr op_desc = CreateOpDesc("FileConstant", FILECONSTANT);
  std::vector<int64_t> shape = {2,2,2,2};
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_FILE_CONSTANT_ID, "vector_search_buchet_value_bin"));
  std::string file_path;
  size_t offset = 0U;
  size_t length = 0U;
  Status ret = FileConstantUtils::GetFilePath(op_desc, file_id_and_path_map, file_path, offset, length);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(file_path, "hello.bin");
}

TEST_F(UtestFileConstantUtilTransfer, GetRealFilePathOK) {
  std::map<std::string, std::string> file_id_and_path_map;
  OpDescPtr op_desc = CreateOpDesc("FileConstant", FILECONSTANT);
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_FILE_PATH, "hello.bin"));
  std::string file_path;
  size_t offset = 0U;
  size_t length = 0U;
  Status ret = FileConstantUtils::GetFilePath(op_desc, file_id_and_path_map, file_path, offset, length);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(file_path, "hello.bin");
}

TEST_F(UtestFileConstantUtilTransfer, GetFileConstantPath) {
  OpDescPtr op_desc = CreateOpDesc("FileConstant", FILECONSTANT);
  EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_OFFSET, 64));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_LENGTH, 1024));
  std::string file_path;
  size_t offset = 0U;
  size_t length = 0U;
  std::map<std::string, std::string> file_id_and_path_map;
  Status ret = FileConstantUtils::GetFilePath(op_desc, file_id_and_path_map, file_path, offset, length);
  EXPECT_NE(ret, SUCCESS);
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_LOCATION, "/tmp/hello.bin"));
  ret = FileConstantUtils::GetFilePath(op_desc, file_id_and_path_map, file_path, offset, length);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(file_path, "/tmp/hello.bin");
  EXPECT_EQ(offset, 64);
  EXPECT_EQ(length, 1024);
}

TEST_F(UtestFileConstantUtilTransfer, test_set_external_path_with_dir_success) {
  OpDescPtr op_desc = CreateOpDesc("file_const_0", FILECONSTANT);
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_LOCATION, "tmp/hello.bin"));
  std::string cache_dir = "cache_dir/";
  auto ret = FileConstantUtils::SetExternalPath(op_desc, cache_dir.append("weight/"));
  EXPECT_EQ(ret, SUCCESS);
  std::map<std::string, std::string> file_id_and_path_map;
  std::string file_path;
  size_t offset = 0U;
  size_t length = 0U;
  ret = FileConstantUtils::GetFilePath(op_desc, file_id_and_path_map, file_path, offset, length);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(file_path, "cache_dir/weight/hello.bin");
}

TEST_F(UtestFileConstantUtilTransfer, test_set_external_path_success) {
  OpDescPtr op_desc = CreateOpDesc("file_const_0", FILECONSTANT);
  std::string om_dir = "om_path/";
  std::string om_path = om_dir + "hello.om";
  const std::string kWeightDir = om_dir + "weight/";
  std::string weight_dir;
  EXPECT_NE(FileConstantUtils::GetExternalWeightDirFromOmPath(om_path, weight_dir), ge::SUCCESS);

  std::string file_name = "om_path/weight/hello.bin";
  size_t file_const_size = 100;
  std::unique_ptr<float[]> float_buf(new float[file_const_size / sizeof(float)]);
  Status ret = FileSaver::SaveToFile(file_name, float_buf.get(), file_const_size);
  EXPECT_EQ(ret, SUCCESS);
  ret = FileSaver::SaveToFile(om_path, float_buf.get(), file_const_size);
  EXPECT_EQ(ret, SUCCESS);

  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_LOCATION, "hello.bin"));
  std::string file_path;
  size_t offset = 0U;
  size_t length = 0U;
  std::map<std::string, std::string> file_id_and_path_map;
  ret = FileConstantUtils::GetFilePath(op_desc, file_id_and_path_map, file_path, offset, length);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(file_path, "hello.bin");

  weight_dir = "";
  ge::ModelData model_data;
  model_data.weight_path = om_dir + "weight/";
  ASSERT_EQ(FileConstantUtils::GetExternalWeightDir(model_data, weight_dir), ge::SUCCESS);
  EXPECT_NE(weight_dir.find(kWeightDir), weight_dir.npos);
  ret = FileConstantUtils::SetExternalPath(op_desc, weight_dir);
  EXPECT_EQ(ret, SUCCESS);
  ret = FileConstantUtils::GetFilePath(op_desc, file_id_and_path_map, file_path, offset, length);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_NE(file_path.find("om_path/weight/hello.bin"), file_path.npos);

  weight_dir = "";
  ASSERT_EQ(FileConstantUtils::GetExternalWeightDirFromOmPath(om_path, weight_dir), ge::SUCCESS);
  EXPECT_NE(weight_dir.find(kWeightDir), weight_dir.npos);
  ret = FileConstantUtils::SetExternalPath(op_desc, weight_dir);
  EXPECT_EQ(ret, SUCCESS);
  ret = FileConstantUtils::GetFilePath(op_desc, file_id_and_path_map, file_path, offset, length);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_NE(file_path.find("om_path/weight/hello.bin"), file_path.npos);
  ge::ut::GraphBuilder builder("graph");
  auto graph = builder.GetGraph();
  om_path = "om_path/hello.om";
  weight_dir = "";
  ASSERT_EQ(FileConstantUtils::GetExternalWeightDirFromOmPath(om_path, weight_dir), ge::SUCCESS);
  EXPECT_NE(weight_dir.find(kWeightDir), weight_dir.npos);
  ret = FileConstantUtils::SetExternalPath(graph, weight_dir);
  EXPECT_EQ(ret, SUCCESS);
  weight_dir = "";
  ASSERT_EQ(FileConstantUtils::GetExternalWeightDirFromOmPath("", weight_dir), ge::SUCCESS);
  EXPECT_EQ(weight_dir, "");
  (void)mmRmdir("om_path");
}

TEST_F(UtestFileConstantUtilTransfer, test_convert_file_const_to_const_success) {
  std::string file_name = "tmp_weight_pid/graph1/hello.bin";
  size_t file_const_size = 12;
  size_t value_num = file_const_size / sizeof(float);
  std::unique_ptr<float[]> float_buf(new float[value_num]);
  for (size_t i = 0U; i < value_num; i++) {
    float_buf[i] = static_cast<float>(i);
  }
  Status ret =  FileSaver::SaveToFile(file_name, float_buf.get(), file_const_size);
  EXPECT_EQ(ret, SUCCESS);

  auto builder = ut::GraphBuilder("graph1");
  auto file_const = builder.AddNode("file_const", FILECONSTANT, 0, 1, FORMAT_ND, DT_FLOAT, {3});
  auto netoutput = builder.AddNode("Node_OutPut", "NetOutPut", 1, 0);
  OpDescPtr op_desc = file_const->GetOpDesc();
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_LOCATION, file_name));
  (void)AttrUtils::SetDataType(op_desc, "dtype", DT_FLOAT);

  builder.AddDataEdge(file_const, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  ret = FileConstantUtils::ConvertFileConstToConst(graph);
  EXPECT_EQ(ret, SUCCESS);
  auto const_node = graph->FindNode("file_const");
  EXPECT_EQ(const_node->GetType(), CONSTANT);
  const auto &weight = OpDescUtils::MutableWeights(const_node);
  auto const_value_size = weight[0]->GetData().GetSize();
  EXPECT_EQ(const_value_size, 12);
  auto const_value_data = reinterpret_cast<const float*>(weight[0]->GetData().GetData())[0];
  EXPECT_EQ(const_value_data, 0.0f);
  const_value_data = reinterpret_cast<const float*>(weight[0]->GetData().GetData())[1];
  EXPECT_EQ(const_value_data, 1.0f);
  const_value_data = reinterpret_cast<const float*>(weight[0]->GetData().GetData())[2];
  EXPECT_EQ(const_value_data, 2.0f);
  (void)mmRmdir("tmp_weight_pid");
}

TEST_F(UtestFileConstantUtilTransfer, ConvertConstToFileConst_Ok_MultipleModelSameConst) {
  const auto build_graph = []() {
    ge::ut::GraphBuilder builder("graph");
    auto const1 = builder.AddNode("const1", "Const", 0, 1);
    auto netoutput = builder.AddNode("Node_OutPut", "NetOutPut", 1, 0);
    ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
    std::vector<uint8_t> value(4 * 8 * 8);
    std::vector<int64_t> shape{1, 4, 8, 8};
    tensor->MutableTensorDesc().SetShape(GeShape(shape));
    tensor->SetData(value);
    tensor->MutableTensorDesc().SetDataType(DT_UINT8);
    ConstantUtils::SetWeight(const1->GetOpDesc(), 0, tensor);
    (void)AttrUtils::SetStr(const1->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
    builder.AddDataEdge(const1, 0, netoutput, 0);
    return builder.GetGraph();
  };
  const auto &external_weight_manager = ExternalWeightManagerPool::Instance().GetManager(GetContext().SessionId());
  ASSERT_NE(external_weight_manager, nullptr);
  external_weight_manager->SetWeightPath("./om_temp/weight");
  auto& meta = external_weight_manager->MutableMetaFile();
  meta.hash_to_weight_file.clear();
  // 第一次保存，没有权重
  auto graph_1 = build_graph();
  auto ret = FileConstantUtils::ConvertConstToFileConst(graph_1);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(meta.hash_to_weight_file.size(), 1);
  (void)mmRmdir("./om_temp");

  // 第二次保存相同模型，meta中有权重信息，但是没有对应权重文件，重新落盘
  auto graph_2 = build_graph();
  ret = FileConstantUtils::ConvertConstToFileConst(graph_2);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(mmAccess("./om_temp/weight/weight_1234567"), EOK);
  meta.hash_to_weight_file.clear();
  (void)mmRmdir("./om_temp");
}

TEST_F(UtestFileConstantUtilTransfer, test_convert_const_to_file_const_AscendWorkPath_success) {
  ge::ut::GraphBuilder builder("graph");
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto netoutput = builder.AddNode("Node_OutPut", "NetOutPut", 1, 0);
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value(4 * 8 * 8);
  std::vector<int64_t> shape{1, 4, 8, 8};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  ConstantUtils::SetWeight(const1->GetOpDesc(), 0, tensor);
  (void)AttrUtils::SetStr(const1->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");

  builder.AddDataEdge(const1, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  ge::char_t current_path[MMPA_MAX_PATH] = {'\0'};
  getcwd(current_path, MMPA_MAX_PATH);
  mmSetEnv("ASCEND_WORK_PATH", current_path, 1);
  auto ret = FileConstantUtils::ConvertConstToFileConst(graph);
  EXPECT_EQ(ret, SUCCESS);

  std::string file_const_path = current_path;
  file_const_path += "/tmp_weight_" + std::to_string(getpid()) + "_0";
  EXPECT_EQ(mmAccess(file_const_path.c_str()), EN_OK);
  unsetenv("ASCEND_WORK_PATH");
}

TEST_F(UtestFileConstantUtilTransfer, test_convert_const_to_file_const_no_const) {
  ge::ut::GraphBuilder builder("graph");
  auto netoutput = builder.AddNode("Node_OutPut", "NetOutPut", 1, 0);

  auto graph = builder.GetGraph();
  auto ret = FileConstantUtils::ConvertConstToFileConst(graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestFileConstantUtilTransfer, test_read_json_failed) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  ge::ut::GraphBuilder builder("graph");
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto netoutput = builder.AddNode("Node_OutPut", "NetOutPut", 1, 0);
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value(4 * 8 * 8);
  std::vector<int64_t> shape{1, 4, 8, 8};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  ConstantUtils::SetWeight(const1->GetOpDesc(), 0, tensor);
  (void)AttrUtils::SetStr(const1->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");

  builder.AddDataEdge(const1, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  auto ret = FileConstantUtils::ConvertConstToFileConst(graph);
  EXPECT_EQ(ret, FAILED);
  MmpaStub::GetInstance().Reset();
}

TEST_F(UtestFileConstantUtilTransfer, test_flock_failed) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForFlockFailed>());
  ge::ut::GraphBuilder builder("graph");
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto netoutput = builder.AddNode("Node_OutPut", "NetOutPut", 1, 0);
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value(4 * 8 * 8);
  std::vector<int64_t> shape{1, 4, 8, 8};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  ConstantUtils::SetWeight(const1->GetOpDesc(), 0, tensor);
  (void)AttrUtils::SetStr(const1->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");

  builder.AddDataEdge(const1, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  auto ret = FileConstantUtils::ConvertConstToFileConst(graph);
  EXPECT_EQ(ret, FAILED);
  MmpaStub::GetInstance().Reset();
}

TEST_F(UtestFileConstantUtilTransfer, test_convert_const_to_file_const_specify_file_path_reuse) {
  ge::ut::GraphBuilder builder("graph");
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto const2 = builder.AddNode("const2", "Const", 0, 1);
  auto netoutput = builder.AddNode("Node_OutPut", "NetOutPut", 1, 0);
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value(4 * 8 * 8);
  std::vector<int64_t> shape{1, 4, 8, 8};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  ConstantUtils::SetWeight(const1->GetOpDesc(), 0, tensor);
  ConstantUtils::SetWeight(const2->GetOpDesc(), 0, tensor);
  (void)AttrUtils::SetStr(const1->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  (void)AttrUtils::SetStr(const2->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");

  builder.AddDataEdge(const1, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  const auto &external_weight_manager = ExternalWeightManagerPool::Instance().GetManager(GetContext().SessionId());
  EXPECT_EQ(external_weight_manager != nullptr, true);
  external_weight_manager->SetWeightPath("./om_out/weight");
  auto ret = FileConstantUtils::ConvertConstToFileConst(graph);
  EXPECT_EQ(ret, SUCCESS);
  auto fileconstant1 = graph->FindFirstNodeMatchType(FILECONSTANT);
  ASSERT_NE(fileconstant1, nullptr);

  // reuse with meta.json
  ge::ut::GraphBuilder builder1("graph");
  auto const1_reuse = builder1.AddNode("const3", "Const", 0, 1);
  auto netoutput1 = builder1.AddNode("Node_OutPut", "NetOutPut", 1, 0);
  ConstantUtils::SetWeight(const1_reuse->GetOpDesc(), 0, tensor);
  (void)AttrUtils::SetStr(const1_reuse->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  builder1.AddDataEdge(const1_reuse, 0, netoutput1, 0);
  auto graph1 = builder1.GetGraph();
  ret = FileConstantUtils::ConvertConstToFileConst(graph1);
  EXPECT_EQ(ret, SUCCESS);
  (void)mmRmdir("om_out");
}

TEST_F(UtestFileConstantUtilTransfer, test_convert_const_to_file_const_specify_file_path_reuse_external_weight_dir) {
  ge::ut::GraphBuilder builder("graph");
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto const2 = builder.AddNode("const2", "Const", 0, 1);
  auto netoutput = builder.AddNode("Node_OutPut", "NetOutPut", 1, 0);
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value(4 * 8 * 8);
  std::vector<int64_t> shape{1, 4, 8, 8};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  ConstantUtils::SetWeight(const1->GetOpDesc(), 0, tensor);
  ConstantUtils::SetWeight(const2->GetOpDesc(), 0, tensor);
  (void)AttrUtils::SetStr(const1->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  (void)AttrUtils::SetStr(const2->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");

  builder.AddDataEdge(const1, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  const auto &external_weight_manager = ExternalWeightManagerPool::Instance().GetManager(GetContext().SessionId());
  EXPECT_EQ(external_weight_manager != nullptr, true);
  std::string option_str = ExternalWeightManager::GetWeightPathFromOption();
  std::cout << "test_convert_const_to_file_const_specify_file_path_reuse_external_weight_dir GetWeightPathFromOption option_str: " << option_str << "end" << std::endl;
  std::cout << "test_convert_const_to_file_const_specify_file_path_reuse_external_weight_dir weightPath begin." << std::endl;
  std::string weightPath = ExternalWeightManager::GetWeightPathFromOption();
  std::cout << "test_convert_const_to_file_const_specify_file_path_reuse_external_weight_dir GetWeightPathFromOption weightPath: " << weightPath << "end" << std::endl;
  external_weight_manager->SetWeightPath("ge.externalWeightDir");
  auto ret = FileConstantUtils::ConvertConstToFileConst(graph);
  EXPECT_EQ(ret, SUCCESS);
  auto fileconstant1 = graph->FindFirstNodeMatchType(FILECONSTANT);
  ASSERT_NE(fileconstant1, nullptr);

  // reuse with meta.json
  ge::ut::GraphBuilder builder1("graph");
  auto const1_reuse = builder1.AddNode("const3", "Const", 0, 1);
  auto netoutput1 = builder1.AddNode("Node_OutPut", "NetOutPut", 1, 0);
  ConstantUtils::SetWeight(const1_reuse->GetOpDesc(), 0, tensor);
  (void)AttrUtils::SetStr(const1_reuse->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  builder1.AddDataEdge(const1_reuse, 0, netoutput1, 0);
  auto graph1 = builder1.GetGraph();
  ret = FileConstantUtils::ConvertConstToFileConst(graph1);
  EXPECT_EQ(ret, SUCCESS);
  (void)mmRmdir("om_out");
}

TEST_F(UtestFileConstantUtilTransfer, test_convert_const_to_file_const_small_weight_size) {
  ge::ut::GraphBuilder builder("graph");
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto netoutput = builder.AddNode("Node_OutPut", "NetOutPut", 1, 0);
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value{1, 2, 3};
  std::vector<int64_t> shape{3};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  ConstantUtils::SetWeight(const1->GetOpDesc(), 0, tensor);

  builder.AddDataEdge(const1, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  auto ret = FileConstantUtils::ConvertConstToFileConst(graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestFileConstantUtilTransfer, test_convert_const_to_file_const_empty_tensor) {
  ge::ut::GraphBuilder builder("graph");
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto netoutput = builder.AddNode("Node_OutPut", "NetOutPut", 1, 0);

  builder.AddDataEdge(const1, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  auto ret = FileConstantUtils::ConvertConstToFileConst(graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = FileConstantUtils::ConvertConstToFileConst(const1);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestFileConstantUtilTransfer, test_change_file_path_success) {
  std::string file_name = "tmp_weight_pid/weight.bin";
  size_t file_const_size = 12;
  size_t value_num = file_const_size / sizeof(float);
  std::unique_ptr<float[]> float_buf(new float[value_num]);
  for (size_t i = 0U; i < value_num; i++) {
    float_buf[i] = static_cast<float>(i);
  }
  Status ret =  FileSaver::SaveToFile(file_name, float_buf.get(), file_const_size);
  EXPECT_EQ(ret, SUCCESS);

  auto builder = ut::GraphBuilder("graph1");
  auto file_const = builder.AddNode("file_const", FILECONSTANT, 0, 1, FORMAT_ND, DT_FLOAT, {3});
  auto netoutput = builder.AddNode("Node_OutPut", "NetOutPut", 1, 0);
  OpDescPtr op_desc = file_const->GetOpDesc();
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_LOCATION, file_name));
  (void)AttrUtils::SetDataType(op_desc, "dtype", DT_FLOAT);

  builder.AddDataEdge(file_const, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  std::string om_path = "om_path/hello.om";
  ret = FileConstantUtils::ChangeFilePath(graph, om_path);
  EXPECT_EQ(ret, SUCCESS);
  std::string weight_file;
  EXPECT_TRUE(AttrUtils::GetStr(op_desc, ATTR_NAME_LOCATION, weight_file));
  EXPECT_EQ(weight_file, "weight.bin");
  (void)mmRmdir("om_path");
}

TEST_F(UtestFileConstantUtilTransfer, test_reuse_external_weight_success) {
  GetContext().SetSessionId(0U);
  ge::ut::GraphBuilder builder("graph");
  auto const1 = builder.AddNode("const_1", "Const", 0, 1);
  auto const2 = builder.AddNode("const_2", "Const", 0, 1);
  auto const3 = builder.AddNode("const_1", "Const", 0, 1);
  auto netoutput = builder.AddNode("Node_OutPut", "NetOutPut", 3, 0);

  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value(4 * 8 * 8);
  std::vector<int64_t> shape{1, 4, 8, 8};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  ConstantUtils::SetWeight(const1->GetOpDesc(), 0, tensor);
  ConstantUtils::SetWeight(const2->GetOpDesc(), 0, tensor);
  ConstantUtils::SetWeight(const3->GetOpDesc(), 0, tensor);
  (void)AttrUtils::SetStr(const1->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  (void)AttrUtils::SetStr(const2->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  (void)AttrUtils::SetStr(const3->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");

  builder.AddDataEdge(const1, 0, netoutput, 0);
  builder.AddDataEdge(const2, 0, netoutput, 1);
  builder.AddDataEdge(const3, 0, netoutput, 2);
  auto graph = builder.GetGraph();
  auto options_back = GetThreadLocalContext().GetAllGraphOptions();
  auto options = options_back;
  options[OPTION_GRAPH_COMPILER_CACHE_DIR] = "./cache_dir";
  GetThreadLocalContext().SetGraphOption(options);
  auto ret = FileConstantUtils::ConvertConstToFileConst(graph);
  EXPECT_EQ(ret, SUCCESS);
  auto fileconstant1 = graph->FindFirstNodeMatchType(FILECONSTANT);
  ASSERT_NE(fileconstant1, nullptr);
  const auto &fileconstant_info = FileConstantUtils::GetFileConstantInfo(fileconstant1->GetOpDesc());
  ASSERT_NE(fileconstant_info.weight_path, "");
  ExternalWeightManagerPool::Instance().Destroy();
  GetThreadLocalContext().SetGraphOption(options_back);
}

TEST_F(UtestFileConstantUtilTransfer, test_reuse_external_weight_combined_use_om_name_success) {
  GetContext().SetSessionId(0U);
  ge::ut::GraphBuilder builder("graph");
  auto const1 = builder.AddNode("const_1", "Const", 0, 1);
  auto const2 = builder.AddNode("const_2", "Const", 0, 1);
  auto const3 = builder.AddNode("const_1", "Const", 0, 1);
  auto netoutput = builder.AddNode("Node_OutPut", "NetOutPut", 3, 0);

  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value(4 * 8 * 8);
  std::vector<int64_t> shape{1, 4, 8, 8};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  ConstantUtils::SetWeight(const1->GetOpDesc(), 0, tensor);
  ConstantUtils::SetWeight(const2->GetOpDesc(), 0, tensor);
  ConstantUtils::SetWeight(const3->GetOpDesc(), 0, tensor);
  (void)AttrUtils::SetStr(const1->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  (void)AttrUtils::SetStr(const2->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  (void)AttrUtils::SetStr(const3->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");

  builder.AddDataEdge(const1, 0, netoutput, 0);
  builder.AddDataEdge(const2, 0, netoutput, 1);
  builder.AddDataEdge(const3, 0, netoutput, 2);
  auto graph = builder.GetGraph();
  auto options_back = GetThreadLocalContext().GetAllGraphOptions();
  auto options = options_back;
  options[OPTION_GRAPH_COMPILER_CACHE_DIR] = "./cache_dir";
  GetThreadLocalContext().SetGraphOption(options);
  (void)AttrUtils::SetStr(graph, ATTR_MODEL_FILE_NAME_PREFIX, "./test_om_file.om");
  auto ret = FileConstantUtils::ConvertConstToFileConst(graph, true);
  EXPECT_EQ(ret, SUCCESS);
  auto fileconstant1 = graph->FindFirstNodeMatchType(FILECONSTANT);
  ASSERT_NE(fileconstant1, nullptr);
  const auto &fileconstant_info = FileConstantUtils::GetFileConstantInfo(fileconstant1->GetOpDesc());
  std::string weight_name = StringUtils::GetFileName(fileconstant_info.weight_path);
  ASSERT_EQ(weight_name, "test_om_file_weight_combined");
  ExternalWeightManagerPool::Instance().Destroy();
  GetThreadLocalContext().SetGraphOption(options_back);
}

TEST_F(UtestFileConstantUtilTransfer, test_reuse_external_weight_combined_use_graph_name_success) {
  GetContext().SetSessionId(0U);
  ge::ut::GraphBuilder builder("test_graph");
  auto const1 = builder.AddNode("const_1", "Const", 0, 1);
  auto const2 = builder.AddNode("const_2", "Const", 0, 1);
  auto const3 = builder.AddNode("const_1", "Const", 0, 1);
  auto netoutput = builder.AddNode("Node_OutPut", "NetOutPut", 3, 0);

  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value(4 * 8 * 8);
  std::vector<int64_t> shape{1, 4, 8, 8};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  ConstantUtils::SetWeight(const1->GetOpDesc(), 0, tensor);
  ConstantUtils::SetWeight(const2->GetOpDesc(), 0, tensor);
  ConstantUtils::SetWeight(const3->GetOpDesc(), 0, tensor);
  (void)AttrUtils::SetStr(const1->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  (void)AttrUtils::SetStr(const2->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  (void)AttrUtils::SetStr(const3->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");

  builder.AddDataEdge(const1, 0, netoutput, 0);
  builder.AddDataEdge(const2, 0, netoutput, 1);
  builder.AddDataEdge(const3, 0, netoutput, 2);
  auto graph = builder.GetGraph();
  auto options_back = GetThreadLocalContext().GetAllGraphOptions();
  auto options = options_back;
  options[OPTION_GRAPH_COMPILER_CACHE_DIR] = "./cache_dir";
  GetThreadLocalContext().SetGraphOption(options);
  auto ret = FileConstantUtils::ConvertConstToFileConst(graph, true);
  EXPECT_EQ(ret, SUCCESS);
  auto fileconstant1 = graph->FindFirstNodeMatchType(FILECONSTANT);
  ASSERT_NE(fileconstant1, nullptr);
  const auto &fileconstant_info = FileConstantUtils::GetFileConstantInfo(fileconstant1->GetOpDesc());
  std::string weight_name = StringUtils::GetFileName(fileconstant_info.weight_path);
  ASSERT_EQ(weight_name, "test_graph_weight_combined");
  ExternalWeightManagerPool::Instance().Destroy();
  GetThreadLocalContext().SetGraphOption(options_back);
}

TEST_F(UtestFileConstantUtilTransfer, test_convert_const_to_file_const_with_DT_STRING) {
  ge::ut::GraphBuilder builder("graph");
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto netoutput = builder.AddNode("Node_OutPut", "NetOutPut", 1, 0);
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value{1, 2, 3};
  std::vector<int64_t> shape{3};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  ConstantUtils::SetWeight(const1->GetOpDesc(), 0, tensor);
  const auto &output_desc = const1->GetOpDesc()->MutableOutputDesc(0U);
  output_desc->SetDataType(DT_STRING);


  builder.AddDataEdge(const1, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  auto ret = FileConstantUtils::ConvertConstToFileConst(graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestFileConstantUtilTransfer, ConvertConstToFileConst_ConstNodeNotChangedForEmptyTensor) {
  ge::ut::GraphBuilder builder("graph");
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto netoutput = builder.AddNode("Node_OutPut", "NetOutPut", 1, 0);
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value{};
  std::vector<int64_t> shape{0};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_INT64);
  ConstantUtils::SetWeight(const1->GetOpDesc(), 0, tensor);
  const auto &output_desc = const1->GetOpDesc()->MutableOutputDesc(0U);
  output_desc->SetDataType(DT_INT64);
  builder.AddDataEdge(const1, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  EXPECT_EQ(graph->FindFirstNodeMatchType(CONSTANT), const1);
  auto ret = FileConstantUtils::ConvertConstToFileConst(graph);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(graph->FindFirstNodeMatchType(CONSTANT), const1); // 空tensor的Const节点不会被转为FileConstant节点
}

TEST_F(UtestFileConstantUtilTransfer, test_refresh_relative_file_path_ok) {
  std::string file_name = "tmp_weight/hello.bin";
  auto builder = ut::GraphBuilder("graph");
  auto fileconst = builder.AddNode("file_const", FILECONSTANT, 0, 1);
  auto netoutput = builder.AddNode("Node_OutPut", "NetOutPut", 1, 0);
  OpDescPtr op_desc = fileconst->GetOpDesc();
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_LOCATION, file_name));
  builder.AddDataEdge(fileconst, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  EXPECT_EQ(FileConstantUtils::RefreshRelativePath(graph), SUCCESS);
  EXPECT_TRUE(AttrUtils::GetStr(op_desc, ATTR_NAME_LOCATION, file_name));
  EXPECT_EQ(file_name, "hello.bin");
}
}
}

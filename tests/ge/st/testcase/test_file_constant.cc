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
#include "common/file_constant_utils/file_constant_utils.h"
#include "graph/ge_local_context.h"
#include "graph/ge_context.h"

#include "macro_utils/dt_public_scope.h"
#include "graph/load/model_manager/davinci_model.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/constant_utils.h"
#include "graph/manager/graph_external_weight_manager.h"
#include "graph/manager/graph_var_manager.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "macro_utils/dt_public_unscope.h"

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
class DModelListener : public ModelListener {
 public:
  DModelListener(){};
  uint32_t OnComputeDone(uint32_t model_id, uint32_t data_index, uint32_t result, std::vector<gert::Tensor> &outputs) {
    return 0;
  }
};

shared_ptr<ModelListener> g_local_call_back(new DModelListener());
class StestFileConstantUtilTransfer : public testing::Test {
 protected:
  void SetUp() {
    ExternalWeightManagerPool::Instance().Destroy();
  }
  void TearDown() {
    ExternalWeightManagerPool::Instance().Destroy();
  }
};

TEST_F(StestFileConstantUtilTransfer, GetRealFilePathOK) {
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

TEST_F(StestFileConstantUtilTransfer, GetFileConstantPath) {
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

TEST_F(StestFileConstantUtilTransfer, Preprocess_Fileconstant_Op_OK) {
  DavinciModel model(0, g_local_call_back);
  model.ge_model_ = MakeShared<GeModel>();
  model.session_id_ = 0U;
  model.runtime_param_.mem_size = 51200;
  model.weights_mem_base_ = reinterpret_cast<uintptr_t>(malloc(model.runtime_param_.mem_size));
  model.file_constant_weight_dir_ = "output/weight/";
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  OpDescPtr op_desc = CreateOpDesc("FileConstant", FILECONSTANT);
  std::vector<int64_t> shape = {2,2,2,2};

  EXPECT_TRUE(AttrUtils::SetDataType(op_desc, "dtype", DT_FLOAT));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_FILE_CONSTANT_ID, "file"));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "shape", shape));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 128);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({0});
  graph->AddNode(op_desc);

  OpDescPtr op_desc2 = CreateOpDesc("FileConstant2", FILECONSTANT);
  EXPECT_TRUE(AttrUtils::SetDataType(op_desc2, "dtype", DT_FLOAT));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc2, ATTR_NAME_FILE_CONSTANT_ID, "file"));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc2, "shape", shape));
  op_desc2->AddOutputDesc(tensor_desc);
  op_desc2->SetOutputOffset({128});
  graph->AddNode(op_desc2); // test ExternalWeightManager::CheckAndSetWeightLoaded

  std::unique_ptr<float[]> float_buf(new float[16]);
  std::string file_name = "tmp_weight_file.bin";
  std::ofstream out1(file_name, std::ios::binary);
  if (!out1.is_open()) {
    return;
  }
  out1.write((char *)float_buf.get(), 16 * sizeof(float));
  out1.close();
  model.file_id_and_path_map_.insert(std::pair<std::string, std::string>("file", "tmp_weight_file.bin"));
  ModelParam default_parm;
  auto status = model.PreProcessFileConstants(graph, default_parm);
  EXPECT_EQ(status, SUCCESS);
  ASSERT_NE(model.runtime_param_.fileconstant_addr_mapping.find(static_cast<int64_t>(0)),
            model.runtime_param_.fileconstant_addr_mapping.end());
  ASSERT_NE(model.runtime_param_.fileconstant_addr_mapping.find(static_cast<int64_t>(128)),
            model.runtime_param_.fileconstant_addr_mapping.end());
  free(reinterpret_cast<void *>(model.runtime_param_.mem_base));
  free(reinterpret_cast<void *>(model.weights_mem_base_));

  VarManager::Instance(0U)->var_resource_ = MakeShared<VarResource>(0U);
  EXPECT_EQ(model.InitNodes(graph), SUCCESS);
  model.runtime_param_.mem_base = 0;
  (void)remove("tmp_weight_file.bin");
}

TEST_F(StestFileConstantUtilTransfer, Preprocess_Fileconstant_WeightCombined_OK) {
  DavinciModel model(0, g_local_call_back);
  model.ge_model_ = MakeShared<GeModel>();
  model.session_id_ = 2;
  model.runtime_param_.mem_size = 51200;
  model.weights_mem_base_ = reinterpret_cast<uintptr_t>(malloc(model.runtime_param_.mem_size));
  model.file_constant_weight_dir_ = "./";

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  std::vector<int64_t> shape = {2,2,2,2};

  OpDescPtr op_desc = CreateOpDesc("FileConstant0", FILECONSTANT);
  EXPECT_TRUE(AttrUtils::SetDataType(op_desc, "dtype", DT_INT32));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_OFFSET, 0));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_LOCATION, "weight_combined_2132345.bin"));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_LENGTH, 768));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "shape", shape));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 768);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({0});
  graph->AddNode(op_desc);

  OpDescPtr op_desc1 = CreateOpDesc("FileConstant1", FILECONSTANT);
  EXPECT_TRUE(AttrUtils::SetDataType(op_desc1, "dtype", DT_INT32));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc1, ATTR_NAME_OFFSET, 1024));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc1, ATTR_NAME_LOCATION, "weight_combined_2132345.bin"));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc1, ATTR_NAME_LENGTH, 1024));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc1, "shape", shape));
  GeTensorDesc tensor_desc1(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc1, 1024);
  op_desc1->AddOutputDesc(tensor_desc1);
  op_desc1->SetOutputOffset({1});
  graph->AddNode(op_desc1);

  std::unique_ptr<float[]> float_buf(new float[2048 / sizeof(float)]);
  std::string file_name = "weight_combined_2132345.bin";
  std::ofstream out1(file_name, std::ios::binary);
  EXPECT_TRUE(out1.is_open());
  out1.write((char *)float_buf.get(), 2048);
  out1.close();
  ModelParam default_parm;
  auto status = model.PreProcessFileConstants(graph, default_parm);
  EXPECT_EQ(status, SUCCESS);
  ASSERT_NE(model.runtime_param_.fileconstant_addr_mapping.find(static_cast<int64_t>(0)),
            model.runtime_param_.fileconstant_addr_mapping.end());
  ASSERT_NE(model.runtime_param_.fileconstant_addr_mapping.find(static_cast<int64_t>(1)),
            model.runtime_param_.fileconstant_addr_mapping.end());
  free(reinterpret_cast<void*>(model.weights_mem_base_));

  VarManager::Instance(0U)->var_resource_ = MakeShared<VarResource>(0U);
  EXPECT_EQ(model.InitNodes(graph), SUCCESS);
  model.runtime_param_.mem_base = 0;
  (void)remove("weight_combined_2132345.bin");
}

TEST_F(StestFileConstantUtilTransfer, Preprocess_Fileconstant_IndividualWeights_OK) {
  DavinciModel model(0, g_local_call_back);
  model.ge_model_ = MakeShared<GeModel>();
  model.session_id_ = 2;
  model.runtime_param_.mem_size = 51200;
  model.weights_mem_base_ = reinterpret_cast<uintptr_t>(malloc(model.runtime_param_.mem_size));
  model.file_constant_weight_dir_ = "./";

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  std::vector<int64_t> shape = {2,2,2,2};

  OpDescPtr op_desc = CreateOpDesc("FileConstant0", FILECONSTANT);
  EXPECT_TRUE(AttrUtils::SetDataType(op_desc, "dtype", DT_INT32));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_OFFSET, 0));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_LOCATION, "weight_combined_1.bin"));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_LENGTH, 768));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "shape", shape));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 768);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({0});
  graph->AddNode(op_desc);

  OpDescPtr op_desc1 = CreateOpDesc("FileConstant1", FILECONSTANT);
  EXPECT_TRUE(AttrUtils::SetDataType(op_desc1, "dtype", DT_INT32));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc1, ATTR_NAME_OFFSET, 0));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc1, ATTR_NAME_LOCATION, "weight_combined_2.bin"));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc1, ATTR_NAME_LENGTH, 1024));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc1, "shape", shape));
  GeTensorDesc tensor_desc1(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc1, 1024);
  op_desc1->AddOutputDesc(tensor_desc1);
  op_desc1->SetOutputOffset({1});
  graph->AddNode(op_desc1);

  std::unique_ptr<float[]> float_buf(new float[1024 / sizeof(float)]);
  std::string file_name = "weight_combined_1.bin";
  std::ofstream out1(file_name, std::ios::binary);
  EXPECT_TRUE(out1.is_open());
  out1.write((char *)float_buf.get(), 1024);
  out1.close();
  file_name = "weight_combined_2.bin";
  std::ofstream out2(file_name, std::ios::binary);
  EXPECT_TRUE(out2.is_open());
  out2.write((char *)float_buf.get(), 1024);
  out2.close();
  ModelParam default_parm;
  auto status = model.PreProcessFileConstants(graph, default_parm);
  EXPECT_EQ(status, SUCCESS);
  ASSERT_NE(model.runtime_param_.fileconstant_addr_mapping.find(static_cast<int64_t>(0)),
            model.runtime_param_.fileconstant_addr_mapping.end());
  ASSERT_NE(model.runtime_param_.fileconstant_addr_mapping.find(static_cast<int64_t>(1)),
            model.runtime_param_.fileconstant_addr_mapping.end());
  free(reinterpret_cast<void*>(model.weights_mem_base_));

  VarManager::Instance(0U)->var_resource_ = MakeShared<VarResource>(0U);
  EXPECT_EQ(model.InitNodes(graph), SUCCESS);
  model.runtime_param_.mem_base = 0;
  (void)remove("weight_combined_1.bin");
  (void)remove("weight_combined_2.bin");
}

TEST_F(StestFileConstantUtilTransfer, Preprocess_Fileconstant_IndividualWeights2_OK) {
  DavinciModel model(0, g_local_call_back);
  model.ge_model_ = MakeShared<GeModel>();
  model.session_id_ = 2;
  model.runtime_param_.mem_size = 51200;
  model.weights_mem_base_ = reinterpret_cast<uintptr_t>(malloc(model.runtime_param_.mem_size));
  model.file_constant_weight_dir_ = "./";

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  std::vector<int64_t> shape = {2,2,2,2};

  OpDescPtr op_desc = CreateOpDesc("FileConstant0", FILECONSTANT);
  EXPECT_TRUE(AttrUtils::SetDataType(op_desc, "dtype", DT_INT32));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_OFFSET, 0));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_LOCATION, "weight_combined_1.bin"));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_LENGTH, 768));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "shape", shape));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 768);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({0});
  graph->AddNode(op_desc);

  OpDescPtr op_desc1 = CreateOpDesc("FileConstant1", FILECONSTANT);
  EXPECT_TRUE(AttrUtils::SetDataType(op_desc1, "dtype", DT_INT32));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc1, ATTR_NAME_OFFSET, 0));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc1, ATTR_NAME_LOCATION, "weight_combined_2.bin"));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc1, ATTR_NAME_LENGTH, 1024));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc1, "shape", shape));
  GeTensorDesc tensor_desc1(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc1, 1024);
  op_desc1->AddOutputDesc(tensor_desc1);
  op_desc1->SetOutputOffset({128});
  graph->AddNode(op_desc1);

  OpDescPtr op_desc2 = CreateOpDesc("FileConstant1", FILECONSTANT);
  EXPECT_TRUE(AttrUtils::SetDataType(op_desc2, "dtype", DT_INT32));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc2, ATTR_NAME_OFFSET, 1024));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc2, ATTR_NAME_LOCATION, "weight_combined_1.bin"));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc2, ATTR_NAME_LENGTH, 1024));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc2, "shape", shape));
  GeTensorDesc tensor_desc2(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc2, 1024);
  op_desc2->AddOutputDesc(tensor_desc2);
  op_desc2->SetOutputOffset({256});
  graph->AddNode(op_desc2);

  std::unique_ptr<float[]> float_buf(new float[2048 / sizeof(float)]);
  std::string file_name = "weight_combined_1.bin";
  std::ofstream out1(file_name, std::ios::binary);
  EXPECT_TRUE(out1.is_open());
  out1.write((char *)float_buf.get(), 1024);
  out1.close();
  file_name = "weight_combined_2.bin";
  std::ofstream out2(file_name, std::ios::binary);
  EXPECT_TRUE(out2.is_open());
  out2.write((char *)float_buf.get(), 1024);
  out2.close();
  ModelParam default_parm;
  auto status = model.PreProcessFileConstants(graph, default_parm);
  EXPECT_EQ(status, SUCCESS);
  ASSERT_NE(model.runtime_param_.fileconstant_addr_mapping.find(static_cast<int64_t>(0)),
            model.runtime_param_.fileconstant_addr_mapping.end());
  ASSERT_NE(model.runtime_param_.fileconstant_addr_mapping.find(static_cast<int64_t>(128)),
            model.runtime_param_.fileconstant_addr_mapping.end());
  ASSERT_NE(model.runtime_param_.fileconstant_addr_mapping.find(static_cast<int64_t>(256)),
            model.runtime_param_.fileconstant_addr_mapping.end());
  free(reinterpret_cast<void*>(model.weights_mem_base_));

  VarManager::Instance(0U)->var_resource_ = MakeShared<VarResource>(0U);
  EXPECT_EQ(model.InitNodes(graph), SUCCESS);
  model.runtime_param_.mem_base = 0;
  (void)remove("weight_combined_1.bin");
  (void)remove("weight_combined_2.bin");
}

TEST_F(StestFileConstantUtilTransfer, Reuse_External_Weight_File_OK) {
  GetContext().SetSessionId(0U);
  DEF_GRAPH(g1) {
    auto const1 = OP_CFG(CONSTANT).InCnt(0).OutCnt(1);
    auto const2 = OP_CFG(CONSTANT).InCnt(0).OutCnt(1);
    auto const3 = OP_CFG(CONSTANT).InCnt(0).OutCnt(1);
    auto netoutput = OP_CFG(NETOUTPUT).InCnt(3).OutCnt(1);
    CHAIN(NODE("const1", const1)->EDGE(0, 0)->NODE("netoutput", netoutput));
    CHAIN(NODE("const2", const2)->EDGE(0, 1)->NODE("netoutput", netoutput));
    CHAIN(NODE("const3", const3)->EDGE(0, 2)->NODE("netoutput", netoutput));
  };

  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto const_node_1 = compute_graph->FindNode("const1");
  auto const_node_2 = compute_graph->FindNode("const2");
  auto const_node_3 = compute_graph->FindNode("const3");
  const_node_3->GetOpDesc()->SetName("const1");

  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value(4 * 8 * 8);
  std::vector<int64_t> shape{1, 4, 8, 8};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  ConstantUtils::SetWeight(const_node_1->GetOpDesc(), 0, tensor);
  ConstantUtils::SetWeight(const_node_2->GetOpDesc(), 0, tensor);
  ConstantUtils::SetWeight(const_node_3->GetOpDesc(), 0, tensor);
  (void)AttrUtils::SetStr(const_node_1->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  (void)AttrUtils::SetStr(const_node_2->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  (void)AttrUtils::SetStr(const_node_3->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  auto options_back = GetThreadLocalContext().GetAllGraphOptions();
  auto options = options_back;
  options[OPTION_GRAPH_COMPILER_CACHE_DIR] = "./cache_dir";
  GetThreadLocalContext().SetGraphOption(options);
  auto ret = FileConstantUtils::ConvertConstToFileConst(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  auto fileconstant_node_1 = compute_graph->FindFirstNodeMatchType(FILECONSTANT);
  ASSERT_NE(fileconstant_node_1, nullptr);
  const auto &fileconstant_info = FileConstantUtils::GetFileConstantInfo(fileconstant_node_1->GetOpDesc());
  ASSERT_NE(fileconstant_info.weight_path, "");
  ExternalWeightManagerPool::Instance().Destroy();
  GetThreadLocalContext().SetGraphOption(options_back);
}

TEST_F(StestFileConstantUtilTransfer, Reuse_External_Weight_Combined_File_Use_GraphName_OK) {
  GetContext().SetSessionId(0U);
  DEF_GRAPH(g1) {
    auto const1 = OP_CFG(CONSTANT).InCnt(0).OutCnt(1);
    auto const2 = OP_CFG(CONSTANT).InCnt(0).OutCnt(1);
    auto const3 = OP_CFG(CONSTANT).InCnt(0).OutCnt(1);
    auto netoutput = OP_CFG(NETOUTPUT).InCnt(3).OutCnt(1);
    CHAIN(NODE("const1", const1)->EDGE(0, 0)->NODE("netoutput", netoutput));
    CHAIN(NODE("const2", const2)->EDGE(0, 1)->NODE("netoutput", netoutput));
    CHAIN(NODE("const3", const3)->EDGE(0, 2)->NODE("netoutput", netoutput));
  };

  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto const_node_1 = compute_graph->FindNode("const1");
  auto const_node_2 = compute_graph->FindNode("const2");
  auto const_node_3 = compute_graph->FindNode("const3");
  const_node_3->GetOpDesc()->SetName("const1");

  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value(4 * 8 * 8);
  std::vector<int64_t> shape{1, 4, 8, 8};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  ConstantUtils::SetWeight(const_node_1->GetOpDesc(), 0, tensor);
  ConstantUtils::SetWeight(const_node_2->GetOpDesc(), 0, tensor);
  ConstantUtils::SetWeight(const_node_3->GetOpDesc(), 0, tensor);
  (void)AttrUtils::SetStr(const_node_1->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  (void)AttrUtils::SetStr(const_node_2->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  (void)AttrUtils::SetStr(const_node_3->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  auto options_back = GetThreadLocalContext().GetAllGraphOptions();
  auto options = options_back;
  options[OPTION_GRAPH_COMPILER_CACHE_DIR] = "./cache_dir";
  GetThreadLocalContext().SetGraphOption(options);
  auto ret = FileConstantUtils::ConvertConstToFileConst(compute_graph, true);
  EXPECT_EQ(ret, SUCCESS);
  auto fileconstant_node_1 = compute_graph->FindFirstNodeMatchType(FILECONSTANT);
  ASSERT_NE(fileconstant_node_1, nullptr);
  const auto &fileconstant_info = FileConstantUtils::GetFileConstantInfo(fileconstant_node_1->GetOpDesc());
  auto graph_name = compute_graph->GetName();
  std::string weight_name = StringUtils::GetFileName(fileconstant_info.weight_path);
  ASSERT_EQ(weight_name, graph_name + "_weight_combined");
  ExternalWeightManagerPool::Instance().Destroy();
  GetThreadLocalContext().SetGraphOption(options_back);
}

TEST_F(StestFileConstantUtilTransfer, Reuse_External_Weight_Combined_File_Use_OmName_OK) {
  GetContext().SetSessionId(0U);
  DEF_GRAPH(g1) {
    auto const1 = OP_CFG(CONSTANT).InCnt(0).OutCnt(1);
    auto const2 = OP_CFG(CONSTANT).InCnt(0).OutCnt(1);
    auto const3 = OP_CFG(CONSTANT).InCnt(0).OutCnt(1);
    auto netoutput = OP_CFG(NETOUTPUT).InCnt(3).OutCnt(1);
    CHAIN(NODE("const1", const1)->EDGE(0, 0)->NODE("netoutput", netoutput));
    CHAIN(NODE("const2", const2)->EDGE(0, 1)->NODE("netoutput", netoutput));
    CHAIN(NODE("const3", const3)->EDGE(0, 2)->NODE("netoutput", netoutput));
  };

  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto const_node_1 = compute_graph->FindNode("const1");
  auto const_node_2 = compute_graph->FindNode("const2");
  auto const_node_3 = compute_graph->FindNode("const3");
  const_node_3->GetOpDesc()->SetName("const1");

  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value(4 * 8 * 8);
  std::vector<int64_t> shape{1, 4, 8, 8};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  ConstantUtils::SetWeight(const_node_1->GetOpDesc(), 0, tensor);
  ConstantUtils::SetWeight(const_node_2->GetOpDesc(), 0, tensor);
  ConstantUtils::SetWeight(const_node_3->GetOpDesc(), 0, tensor);
  (void)AttrUtils::SetStr(const_node_1->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  (void)AttrUtils::SetStr(const_node_2->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  (void)AttrUtils::SetStr(const_node_3->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  auto options_back = GetThreadLocalContext().GetAllGraphOptions();
  auto options = options_back;
  options[OPTION_GRAPH_COMPILER_CACHE_DIR] = "./cache_dir";
  GetThreadLocalContext().SetGraphOption(options);
  (void)AttrUtils::SetStr(compute_graph, ATTR_MODEL_FILE_NAME_PREFIX, "./test_om_file.om");
  auto ret = FileConstantUtils::ConvertConstToFileConst(compute_graph, true);
  EXPECT_EQ(ret, SUCCESS);
  auto fileconstant_node_1 = compute_graph->FindFirstNodeMatchType(FILECONSTANT);
  ASSERT_NE(fileconstant_node_1, nullptr);
  const auto &fileconstant_info = FileConstantUtils::GetFileConstantInfo(fileconstant_node_1->GetOpDesc());
  std::string weight_name = StringUtils::GetFileName(fileconstant_info.weight_path);
  ASSERT_EQ(weight_name, "test_om_file_weight_combined");
  ExternalWeightManagerPool::Instance().Destroy();
  GetThreadLocalContext().SetGraphOption(options_back);
}

TEST_F(StestFileConstantUtilTransfer, Reuse_NO_CONST) {
  GetContext().SetSessionId(0U);
  DEF_GRAPH(g1) {
    auto netoutput = OP_CFG(NETOUTPUT).InCnt(3).OutCnt(1);
  };

  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();

  auto options_back = GetThreadLocalContext().GetAllGraphOptions();
  auto options = options_back;
  options[OPTION_GRAPH_COMPILER_CACHE_DIR] = "./cache_dir";
  GetThreadLocalContext().SetGraphOption(options);
  auto ret = FileConstantUtils::ConvertConstToFileConst(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(StestFileConstantUtilTransfer, Reuse_Flock_Failed) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForFlockFailed>());
  GetContext().SetSessionId(0U);
  DEF_GRAPH(g1) {
    auto const1 = OP_CFG(CONSTANT).InCnt(0).OutCnt(1);
    auto const2 = OP_CFG(CONSTANT).InCnt(0).OutCnt(1);
    auto const3 = OP_CFG(CONSTANT).InCnt(0).OutCnt(1);
    auto netoutput = OP_CFG(NETOUTPUT).InCnt(3).OutCnt(1);
    CHAIN(NODE("const1", const1)->EDGE(0, 0)->NODE("netoutput", netoutput));
    CHAIN(NODE("const2", const2)->EDGE(0, 1)->NODE("netoutput", netoutput));
    CHAIN(NODE("const3", const3)->EDGE(0, 2)->NODE("netoutput", netoutput));
  };

  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto const_node_1 = compute_graph->FindNode("const1");
  auto const_node_2 = compute_graph->FindNode("const2");
  auto const_node_3 = compute_graph->FindNode("const3");
  const_node_3->GetOpDesc()->SetName("const1");

  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value(4 * 8 * 8);
  std::vector<int64_t> shape{1, 4, 8, 8};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  ConstantUtils::SetWeight(const_node_1->GetOpDesc(), 0, tensor);
  ConstantUtils::SetWeight(const_node_2->GetOpDesc(), 0, tensor);
  ConstantUtils::SetWeight(const_node_3->GetOpDesc(), 0, tensor);
  (void)AttrUtils::SetStr(const_node_1->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  (void)AttrUtils::SetStr(const_node_2->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  (void)AttrUtils::SetStr(const_node_3->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  auto options_back = GetThreadLocalContext().GetAllGraphOptions();
  auto options = options_back;
  options[OPTION_GRAPH_COMPILER_CACHE_DIR] = "./cache_dir";
  GetThreadLocalContext().SetGraphOption(options);
  auto ret = FileConstantUtils::ConvertConstToFileConst(compute_graph);
  EXPECT_EQ(ret, FAILED);
  MmpaStub::GetInstance().Reset();
}

TEST_F(StestFileConstantUtilTransfer, Reuse_Read_Json_Failed) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  GetContext().SetSessionId(0U);
  DEF_GRAPH(g1) {
    auto const1 = OP_CFG(CONSTANT).InCnt(0).OutCnt(1);
    auto const2 = OP_CFG(CONSTANT).InCnt(0).OutCnt(1);
    auto const3 = OP_CFG(CONSTANT).InCnt(0).OutCnt(1);
    auto netoutput = OP_CFG(NETOUTPUT).InCnt(3).OutCnt(1);
    CHAIN(NODE("const1", const1)->EDGE(0, 0)->NODE("netoutput", netoutput));
    CHAIN(NODE("const2", const2)->EDGE(0, 1)->NODE("netoutput", netoutput));
    CHAIN(NODE("const3", const3)->EDGE(0, 2)->NODE("netoutput", netoutput));
  };

  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto const_node_1 = compute_graph->FindNode("const1");
  auto const_node_2 = compute_graph->FindNode("const2");
  auto const_node_3 = compute_graph->FindNode("const3");
  const_node_3->GetOpDesc()->SetName("const1");

  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value(4 * 8 * 8);
  std::vector<int64_t> shape{1, 4, 8, 8};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  ConstantUtils::SetWeight(const_node_1->GetOpDesc(), 0, tensor);
  ConstantUtils::SetWeight(const_node_2->GetOpDesc(), 0, tensor);
  ConstantUtils::SetWeight(const_node_3->GetOpDesc(), 0, tensor);
  (void)AttrUtils::SetStr(const_node_1->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  (void)AttrUtils::SetStr(const_node_2->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  (void)AttrUtils::SetStr(const_node_3->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, "1234567");
  auto options_back = GetThreadLocalContext().GetAllGraphOptions();
  auto options = options_back;
  options[OPTION_GRAPH_COMPILER_CACHE_DIR] = "./cache_dir";
  GetThreadLocalContext().SetGraphOption(options);
  auto ret = FileConstantUtils::ConvertConstToFileConst(compute_graph);
  EXPECT_EQ(ret, FAILED);
  MmpaStub::GetInstance().Reset();
}

TEST_F(StestFileConstantUtilTransfer, Refresh_Relative_File_Path_Success) {
  DEF_GRAPH(g1) {
    const auto fileconstant = OP_CFG(FILECONSTANT).Attr(ATTR_NAME_LOCATION, "tmp_weight/weight.bin");
    CHAIN(NODE("file_const", fileconstant)->EDGE(0, 0)->NODE(NODE_NAME_NET_OUTPUT, NETOUTPUT));
  };
  const auto &graph = ToComputeGraph(g1);
  const auto &file_const_op = graph->FindNode("file_const")->GetOpDesc();
  EXPECT_EQ(FileConstantUtils::RefreshRelativePath(graph), SUCCESS);
  std::string file_name;
  EXPECT_TRUE(AttrUtils::GetStr(file_const_op, ATTR_NAME_LOCATION, file_name));
  EXPECT_EQ(file_name, "weight.bin");
}
}
}

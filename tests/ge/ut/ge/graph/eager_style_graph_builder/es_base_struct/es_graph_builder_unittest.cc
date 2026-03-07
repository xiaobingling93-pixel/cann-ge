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
#include <fstream>
#include "common/summary_checker.h"
#include "common/topo_checker.h"
#include "utils/graph_utils_ex.h"
#include "es_graph_builder.h"
#include "es_c_graph_builder.h"
#include "esb_funcs.h"
#include "c_types.h"
#include "common/fp16_t/fp16_t.h"
#include "graph/operator_factory.h"
#include "compliant_node_builder.h"

using namespace ge::es;

std::string GetTempDirectory() {
  return "/tmp";
}

namespace {
// 检查默认的Tensor格式
auto check = [](EsTensorHolder &t) {
  ge::TensorDesc td1;
  t.GetProducer()->GetOutputDesc(0, td1);
  EXPECT_EQ(td1.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(td1.GetFormat(), ge::FORMAT_ND);
  EXPECT_EQ(td1.GetOriginFormat(), ge::FORMAT_ND);
  EXPECT_EQ(td1.GetShape().GetDims(), std::vector<int64_t>());
  EXPECT_EQ(td1.GetOriginShape().GetDims(), std::vector<int64_t>());
};
} // namespace
class EsGraphBuilderLLT : public ::testing::Test {
  protected:
    void SetUp() override {
    }
    void TearDown() override {
    }

    void CreateTmpFileDir() {
      temp_dir = GetTempDirectory() + "/binary_file_test";
      file_path = temp_dir + "/test_binary.bin";
      std::string command = "mkdir -p " + temp_dir;
      (void) std::system(command.c_str());
    }

    void CleanTmpFileDir() {
      std::string command = "rm -rf " + temp_dir;
      (void) std::system(command.c_str());
    }

    void CreateBinaryFile(const std::vector<char> &data) {
      std::ofstream file(file_path, std::ios::binary);
      EXPECT_TRUE(file.is_open()) << "Failed to create file: " << file_path;
      file.write(data.data(), data.size());
      file.close();
    }

    std::string temp_dir;
    std::string file_path;
};

TEST_F(EsGraphBuilderLLT, CreateTensorFromFileTest) {
  std::vector<char> test_data = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'};
  CleanTmpFileDir();
  CreateTmpFileDir();
  CreateBinaryFile(test_data);

  std::vector<int64_t> dims = {1};
  auto tensor = CreateTensorFromFile<int64_t>(file_path.c_str(), dims, ge::DT_INT64, ge::FORMAT_ALL);
  EXPECT_EQ(tensor->GetSize(), 8);
  auto tensor_data = reinterpret_cast<const char *>(tensor->GetData());
  EXPECT_NE(tensor_data, nullptr);
  CleanTmpFileDir();
}

TEST_F(EsGraphBuilderLLT, CreateInputs) {
  EsGraphBuilder builder("test_graph");
  auto t1 = builder.CreateInput(0, "input0", "Data");
  auto t2 = builder.CreateInput(1);
  auto t3 = builder.CreateInput(2, "input2");
  auto t4 = builder.CreateInput(3, "input3", ge::DT_INT32, ge::FORMAT_NCHW, {2, 2});
  EXPECT_NE(t1.GetCTensorHolder(), nullptr);
  EXPECT_NE(t2.GetCTensorHolder(), nullptr);
  EXPECT_NE(t3.GetCTensorHolder(), nullptr);
  EXPECT_NE(t4.GetCTensorHolder(), nullptr);
  check(t1);
  check(t2);
  check(t3);

  ge::TensorDesc td4;
  t4.GetProducer()->GetOutputDesc(0, td4);
  EXPECT_EQ(td4.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(td4.GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td4.GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td4.GetShape().GetDims(), std::vector<int64_t>({2, 2}));
  EXPECT_EQ(td4.GetOriginShape().GetDims(), std::vector<int64_t>({2, 2}));
}

TEST_F(EsGraphBuilderLLT, CreateInputsV2) {
  EsGraphBuilder builder("test_graph");
  auto t1 = builder.CreateInput(0, "input0", "Data");
  auto [t2, t3, t4] = builder.CreateInputs<3>(1);
  EXPECT_NE(t1.GetCTensorHolder(), nullptr);
  EXPECT_NE(t2.GetCTensorHolder(), nullptr);
  EXPECT_NE(t3.GetCTensorHolder(), nullptr);
  EXPECT_NE(t4.GetCTensorHolder(), nullptr);
  check(t1);
  check(t2);
  check(t3);
  check(t4);
  auto [t5, t6] = builder.CreateInputs<2>(1);  // 指定错误的索引
  EXPECT_EQ(t5.GetCTensorHolder(), nullptr);
  EXPECT_EQ(t6.GetCTensorHolder(), nullptr);
}

TEST_F(EsGraphBuilderLLT, CreateInputsV3) {
  EsGraphBuilder builder("test_graph");
  auto inputs = builder.CreateInputs(4);
  EXPECT_EQ(inputs.size(), 4U);
  auto t1 = inputs[0];
  auto t2 = inputs[1];
  auto t3 = inputs[2];
  auto t4 = inputs[3];
  EXPECT_NE(t1.GetCTensorHolder(), nullptr);
  EXPECT_NE(t2.GetCTensorHolder(), nullptr);
  EXPECT_NE(t3.GetCTensorHolder(), nullptr);
  EXPECT_NE(t4.GetCTensorHolder(), nullptr);
  check(t1);
  check(t2);
  check(t3);
  check(t4);
  inputs = builder.CreateInputs(2, 1);  // 指定错误的索引
  EXPECT_EQ(inputs.size(), 2U);
  EXPECT_EQ(inputs[0].GetCTensorHolder(), nullptr);
  EXPECT_EQ(inputs[1].GetCTensorHolder(), nullptr);
}

TEST_F(EsGraphBuilderLLT, CreateConsts) {
  EsGraphBuilder builder("test_graph");
  std::vector<int64_t> dims = {3};

  std::vector<int64_t> vec64 = {1, 2, 3};
  auto c1 = builder.CreateConst(vec64, dims);
  EXPECT_NE(c1.GetCTensorHolder(), nullptr);
  std::vector<int32_t> vec32 = {1, 2, 3};
  auto c2 = builder.CreateConst(vec32, dims);
  EXPECT_NE(c2.GetCTensorHolder(), nullptr);
  std::vector<uint64_t> vecu64 = {1, 2, 3};
  auto c3 = builder.CreateConst(vecu64, dims);
  EXPECT_NE(c3.GetCTensorHolder(), nullptr);
  std::vector<uint32_t> vecu32 = {1, 2, 3};
  auto c4 = builder.CreateConst(vecu32, dims);
  EXPECT_NE(c4.GetCTensorHolder(), nullptr);
  std::vector<float> vecf = {1.1, 2.2, 3.3};
  auto c5 = builder.CreateConst(vecf, dims);
  EXPECT_NE(c5.GetCTensorHolder(), nullptr);
}

TEST_F(EsGraphBuilderLLT, CreateConstValue) {
  EsGraphBuilder builder("test_graph");
  std::vector<float> vecf = {1.1, 2.0, 3.2, 4.4};
  std::vector<int64_t> dims = {4};
  auto c1 = builder.CreateConst(vecf, dims);
  EXPECT_NE(c1.GetCTensorHolder(), nullptr);
  auto graph = builder.BuildAndReset();
  EXPECT_NE(graph, nullptr);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(*graph);
  EXPECT_NE(compute_graph, nullptr);
  auto node = compute_graph->FindNode("Const_0");
  EXPECT_NE(node, nullptr) << "Node 'Constd_0' not found in graph";

  ge::AnyValue av;
  auto op_desc = node->GetOpDesc();
  auto status = op_desc->GetAttr("value", av);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS) << "Failed to get 'value' attr from node: " << node->GetName();
  EXPECT_FALSE(av.IsEmpty());
  auto tensor_attr = av.Get<ge::GeTensor>();
  EXPECT_TRUE(tensor_attr != nullptr);
  EXPECT_EQ(tensor_attr->GetTensorDesc().GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(tensor_attr->GetData().GetSize(), 16U);
  EXPECT_FLOAT_EQ((reinterpret_cast<const float *>(tensor_attr->GetData().data()))[3], 4.4);
}

TEST_F(EsGraphBuilderLLT, CreateScalarsAndVectors) {
  EsGraphBuilder builder("test_graph");

  std::vector<int64_t> vec64 = {1, 2, 3};
  auto v1 = builder.CreateVector(vec64);
  EXPECT_NE(v1.GetCTensorHolder(), nullptr);
  std::vector<int32_t> vec32 = {1, 2, 3};
  auto v2 = builder.CreateVector(vec32);
  EXPECT_NE(v2.GetCTensorHolder(), nullptr);
  std::vector<uint64_t> vecu64 = {1, 2, 3};
  auto v3 = builder.CreateVector(vecu64);
  EXPECT_NE(v3.GetCTensorHolder(), nullptr);
  std::vector<uint32_t> vecu32 = {1, 2, 3};
  auto v4 = builder.CreateVector(vecu32);
  EXPECT_NE(v4.GetCTensorHolder(), nullptr);
  std::vector<float> vecf = {1.1, 2.2, 3.3};
  auto v5 = builder.CreateVector(vecf);
  EXPECT_NE(v5.GetCTensorHolder(), nullptr);

  auto s1 = builder.CreateScalar(int64_t(42));
  auto s2 = builder.CreateScalar(int32_t(7));
  auto s3 = builder.CreateScalar(uint64_t(8));
  auto s4 = builder.CreateScalar(uint32_t(8));
  auto s5 = builder.CreateScalar(float(3.14f));
  EXPECT_NE(s1.GetCTensorHolder(), nullptr);
  EXPECT_NE(s2.GetCTensorHolder(), nullptr);
  EXPECT_NE(s3.GetCTensorHolder(), nullptr);
  EXPECT_NE(s4.GetCTensorHolder(), nullptr);
  EXPECT_NE(s5.GetCTensorHolder(), nullptr);
}

TEST_F(EsGraphBuilderLLT, CreateVectorValue) {
  EsGraphBuilder builder("test_graph");
  std::vector<int32_t> vec32 = {1, 2, 3};
  auto v1 = builder.CreateVector(vec32);
  EXPECT_NE(v1.GetCTensorHolder(), nullptr);
  auto graph = builder.BuildAndReset();
  EXPECT_NE(graph, nullptr);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(*graph);
  EXPECT_NE(compute_graph, nullptr);
  auto node = compute_graph->FindNode("Const_0");
  EXPECT_NE(node, nullptr) << "Node 'Constd_0' not found in graph";

  ge::AnyValue av;
  auto op_desc = node->GetOpDesc();
  auto status = op_desc->GetAttr("value", av);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS) << "Failed to get 'value' attr from node: " << node->GetName();
  EXPECT_FALSE(av.IsEmpty());
  auto tensor_attr = av.Get<ge::GeTensor>();
  EXPECT_TRUE(tensor_attr != nullptr);
  EXPECT_EQ(tensor_attr->GetTensorDesc().GetDataType(), ge::DT_INT32);
  EXPECT_EQ(tensor_attr->GetData().GetSize(), 12U);
  EXPECT_EQ((reinterpret_cast<const int32_t *>(tensor_attr->GetData().data()))[0], 1);
  EXPECT_EQ((reinterpret_cast<const int32_t *>(tensor_attr->GetData().data()))[1], 2);
}

TEST_F(EsGraphBuilderLLT, CreateScalarValue) {
  EsGraphBuilder builder("test_graph");
  auto s1 = builder.CreateScalar(int64_t(42));
  EXPECT_NE(s1.GetCTensorHolder(), nullptr);
  auto graph = builder.BuildAndReset();
  EXPECT_NE(graph, nullptr);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(*graph);
  EXPECT_NE(compute_graph, nullptr);
  auto node = compute_graph->FindNode("Const_0");
  EXPECT_NE(node, nullptr) << "Node 'Constd_0' not found in graph";

  ge::AnyValue av;
  auto op_desc = node->GetOpDesc();
  auto status = op_desc->GetAttr("value", av);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS) << "Failed to get 'value' attr from node: " << node->GetName();
  EXPECT_FALSE(av.IsEmpty());
  auto tensor_attr = av.Get<ge::GeTensor>();
  EXPECT_TRUE(tensor_attr != nullptr);
  EXPECT_EQ(tensor_attr->GetTensorDesc().GetDataType(), ge::DT_INT64);
  EXPECT_EQ(tensor_attr->GetData().GetSize(), 8U);
  EXPECT_EQ(*(reinterpret_cast<const int64_t *>(tensor_attr->GetData().data())), 42);
}

TEST_F(EsGraphBuilderLLT, CreateVariable) {
  EsGraphBuilder builder("test_graph");
  auto v1 = builder.CreateVariable(0, "var0");
  EXPECT_TRUE(v1.GetProducer()->HasAttr("index"));
  EXPECT_FALSE(v1.GetProducer()->HasAttr("value"));
  EXPECT_FALSE(v1.GetProducer()->HasAttr("container"));
  EXPECT_FALSE(v1.GetProducer()->HasAttr("shared_name"));
  EXPECT_NE(v1.GetCTensorHolder(), nullptr);
}

TEST_F(EsGraphBuilderLLT, SetOutputAndBuild) {
  EsGraphBuilder builder("test_graph");
  auto t1 = builder.CreateInput(0, "input0", "Data");
  auto t2 = builder.CreateScalar(int64_t(10));
  builder.SetOutput(t1, 0);
  builder.SetOutput(t2, 1);
  auto graph = builder.BuildAndReset();
  EXPECT_NE(graph, nullptr);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(*graph);
  EXPECT_NE(compute_graph, nullptr);
  gert::SummaryChecker checker(compute_graph);
  const std::map<std::string, size_t> &node_types_to_count = {{"Data", 1}, {"Const", 1}, {"NetOutput", 1}};
  checker.StrictDirectNodeTypes(node_types_to_count);
  STRICT_DIRECT_NODE_TYPES(compute_graph, node_types_to_count);
  for (const auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "NetOutput") {
      EXPECT_EQ(node->GetName(), "NetOutput_test_graph");
      EXPECT_EQ(gert::NodeTopoChecker(node).StrictConnectFrom({{"Data"}, {"Const"}}), "success");
    }
  }
}

TEST_F(EsGraphBuilderLLT, CheckDataNodeInputTensorDesc) {
  EsGraphBuilder builder("test_graph");
  auto t1 = builder.CreateInput(0, "input0", ge::DT_INT32, ge::FORMAT_NCHW, {2, 2});
  auto t2 = builder.CreateInput(1, "input1", ge::DT_FLOAT, ge::FORMAT_ND, {1});
  builder.SetOutput(t1, 0);
  builder.SetOutput(t2, 1);
  auto graph = builder.BuildAndReset();
  EXPECT_NE(graph, nullptr);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(*graph);
  EXPECT_NE(compute_graph, nullptr);
  gert::SummaryChecker checker(compute_graph);
  const std::map<std::string, size_t> &node_types_to_count = {{"Data", 2}, {"NetOutput", 1}};
  STRICT_DIRECT_NODE_TYPES(compute_graph, node_types_to_count);

  auto input_nodes = compute_graph->GetInputNodes();
  EXPECT_EQ(input_nodes.size(), 2);
  auto input0 = input_nodes.at(0);
  EXPECT_EQ(input0->GetName(), "input0");
  ge::GeTensorDesc input_td = input0->GetOpDesc()->GetInputDesc("x");
  EXPECT_EQ(input_td.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(input_td.GetFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input_td.GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input_td.GetShape().GetDims(), std::vector<int64_t>({2, 2}));
  EXPECT_EQ(input_td.GetOriginShape().GetDims(), std::vector<int64_t>({2, 2}));

  auto input1 = input_nodes.at(1);
  EXPECT_EQ(input1->GetName(), "input1");
  input_td = input1->GetOpDesc()->GetInputDesc("x");
  EXPECT_EQ(input_td.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(input_td.GetFormat(), ge::FORMAT_ND);
  EXPECT_EQ(input_td.GetOriginFormat(), ge::FORMAT_ND);
  EXPECT_EQ(input_td.GetShape().GetDims(), std::vector<int64_t>({1}));
  EXPECT_EQ(input_td.GetOriginShape().GetDims(), std::vector<int64_t>({1}));
}

TEST_F(EsGraphBuilderLLT, BuildWithOutputsVector) {
  EsGraphBuilder builder("test_graph");
  auto t1 = builder.CreateInput(0, "input0", "Data");
  auto t2 = builder.CreateScalar(int64_t(10));
  std::vector<EsTensorHolder> outputs = {t1, t2};
  auto graph = builder.BuildAndReset(outputs);
  EXPECT_NE(graph, nullptr);
}

TEST_F(EsGraphBuilderLLT, GetEsbGraph) {
  EsGraphBuilder builder("test_graph");
  EXPECT_NE(builder.GetCGraphBuilder(), nullptr);
}

TEST_F(EsGraphBuilderLLT, TensorsToEsCTensorHolders) {
  EsGraphBuilder builder("test_graph");
  auto t1 = builder.CreateScalar(int32_t(5));
  auto t2 = builder.CreateScalar(int64_t(6));
  std::vector<EsTensorHolder> tensors = {t1, t2};
  auto esb_tensors = ge::es::TensorsToEsCTensorHolders(tensors);
  EXPECT_EQ(esb_tensors.size(), 2);
  EXPECT_NE(esb_tensors[0], nullptr);
  EXPECT_NE(esb_tensors[1], nullptr);
}

// 异常用例1：测试Graph Input Indexes不连续的情况
TEST_F(EsGraphBuilderLLT, IsGraphValid_DiscontinuousInputIndexes) {
  EsGraphBuilder builder("test_graph");

  // 创建正常的输入0
  auto input0 = builder.CreateInput(0, "input0", "Data");
  EXPECT_NE(input0.GetCTensorHolder(), nullptr);

  auto input2 = builder.CreateInput(2, "input2", "Data");
  EXPECT_NE(input2.GetCTensorHolder(), nullptr);

  // 现在graph_input_indexes_包含{0, 2}，这是不连续的
  // 当调用BuildComputeGraph()时，内部会调用IsGraphValid()进行验证
  // 由于输入索引不连续，IsGraphValid应该返回false，导致构建失败

  auto compute_graph = builder.BuildAndReset();
}

// 异常用例2：测试Output Indexes不连续的情况
TEST_F(EsGraphBuilderLLT, IsGraphValid_DiscontinuousOutputIndexes) {
  EsGraphBuilder builder("test_graph");

  // 创建两个输入
  auto input0 = builder.CreateInput(0, "input0", "Data");
  auto input1 = builder.CreateInput(1, "input1", "Data");
  EXPECT_NE(input0.GetCTensorHolder(), nullptr);
  EXPECT_NE(input1.GetCTensorHolder(), nullptr);

  // 设置输出索引0和2，跳过索引1，创建不连续的输出索引
  builder.SetOutput(input0, 0);
  builder.SetOutput(input1, 2);  // 直接设置索引2，跳过索引1

  // 当调用BuildComputeGraph()时，内部会调用IsGraphValid()进行验证
  // 由于输出索引不连续{0, 2}，IsGraphValid应该返回false，导致构建失败
  auto graph = builder.BuildAndReset();
  EXPECT_TRUE(graph == nullptr);
}

// 测试EsCreateGraph和EsDestroyGraph函数
TEST_F(EsGraphBuilderLLT, EsCreateAndDestroyGraph) {
  // 测试使用默认名称创建图
  auto *graph1 = EsCreateGraphBuilder(nullptr);
  EXPECT_NE(graph1, nullptr);
  EXPECT_STREQ(graph1->GetGraph()->GetName().c_str(), "graph");
  EsDestroyGraphBuilder(graph1);

  // 测试使用自定义名称创建图
  auto *graph2 = EsCreateGraphBuilder("custom_graph");
  EXPECT_NE(graph2, nullptr);
  EXPECT_STREQ(graph2->GetGraph()->GetName().c_str(), "custom_graph");
  EsDestroyGraphBuilder(graph2);
}

// 测试EsCreateGraphInputWithDetails和EsCreateGraphInput函数
TEST_F(EsGraphBuilderLLT, EsCreateGraphInputFunctions) {
  auto *graph = EsCreateGraphBuilder("test_graph");
  ASSERT_NE(graph, nullptr);

  // 测试EsCreateGraphInputWithDetails
  auto *input1 = EsCreateGraphInputWithDetails(graph, 0, "input0", "Data", C_DT_BF16, C_FORMAT_C1HWC0, nullptr, 0);
  EXPECT_NE(input1, nullptr);

  // 测试EsCreateGraphInput（使用默认参数）
  auto *input2 = EsCreateGraphInput(graph, 1);
  EXPECT_NE(input2, nullptr);

  EsDestroyGraphBuilder(graph);
}

// EsGetProducer
TEST_F(EsGraphBuilderLLT, EsGetProducer) {
  auto *graph = EsCreateGraphBuilder("test_graph");
  ASSERT_NE(graph, nullptr);

  // 创建tensor
  auto *tensor = EsCreateGraphInput(graph, 0);
  EXPECT_NE(tensor, nullptr);

  // 获取tensor所属的图构建器
  auto *builder = EsGetOwnerBuilder(tensor);
  EXPECT_NE(builder, nullptr);

  // 获取tensor所属的节点
  auto *producer = static_cast<void *>(EsGetProducer(tensor));
  EXPECT_NE(producer, nullptr);
  ge::AscendString type;
  static_cast<ge::GNode *>(producer)->GetType(type);
  EXPECT_EQ(type, "Data");
  EsDestroyGraphBuilder(graph);
}

TEST_F(EsGraphBuilderLLT, EsSetInt64AttrForGraph) {
  EsGraphBuilder builder("test_graph");
  EXPECT_EQ(builder.SetAttr("int64_attr", static_cast<int64_t>(10)), ge::SUCCESS);
  auto graph = builder.BuildAndReset();
  EXPECT_NE(graph, nullptr);
  ge::AnyValue av;
  EXPECT_EQ(ge::GraphUtilsEx::GetComputeGraph(*graph)->GetAttr("int64_attr", av), ge::GRAPH_SUCCESS);
  int64_t t{-1};
  av.GetValue(t);
  EXPECT_EQ(t, 10);
  EXPECT_NE(builder.SetAttr("int64_attr2", static_cast<int64_t>(10)), ge::SUCCESS);
}

TEST_F(EsGraphBuilderLLT, EsSetStringAndBoolAttrForGraph) {
  EsGraphBuilder builder("test_graph");

  EXPECT_EQ(builder.SetAttr("str_attr", "hello"), ge::SUCCESS);
  EXPECT_EQ(builder.SetAttr("bool_attr", true), ge::SUCCESS);

  auto graph = builder.BuildAndReset();
  EXPECT_NE(graph, nullptr);

  ge::AnyValue av_str;
  EXPECT_EQ(ge::GraphUtilsEx::GetComputeGraph(*graph)->GetAttr("str_attr", av_str), ge::GRAPH_SUCCESS);
  std::string str_val;
  av_str.GetValue(str_val);
  EXPECT_EQ(str_val, "hello");

  ge::AnyValue av_bool;
  EXPECT_EQ(ge::GraphUtilsEx::GetComputeGraph(*graph)->GetAttr("bool_attr", av_bool), ge::GRAPH_SUCCESS);
  bool bool_val = false;
  av_bool.GetValue(bool_val);
  EXPECT_EQ(bool_val, true);

  EXPECT_NE(builder.SetAttr("str_attr", "world"), ge::SUCCESS);
  EXPECT_NE(builder.SetAttr("bool_attr", false), ge::SUCCESS);
}

TEST_F(EsGraphBuilderLLT, ListListTypeToPtrAndCounts) {
  const std::vector<std::vector<int64_t>> list_list_type = {{1, 2, 3}, {2, 3}, {3}};
  auto list_int_ptr = ListListTypeToPtrAndCounts(list_list_type);
  EXPECT_EQ(list_int_ptr.first.size(), 3);
  EXPECT_EQ(list_int_ptr.first.at(1)[1], 3);
  EXPECT_EQ(list_int_ptr.second.size(), 3);
  EXPECT_EQ(list_int_ptr.second.at(0), 3);
  EXPECT_EQ(list_int_ptr.second.at(1), 2);
  EXPECT_EQ(list_int_ptr.second.at(2), 1);
}

TEST_F(EsGraphBuilderLLT, GeGraphsToEsCGraphs) {
  EsGraphBuilder builder1("test_graph1");
  auto graph1 = builder1.BuildAndReset();
  EsGraphBuilder builder2("test_graph2");
  auto graph2 = builder1.BuildAndReset();
  EsGraphBuilder builder3("test_graph3");
  auto graph3 = builder1.BuildAndReset();
  std::vector<std::unique_ptr<ge::Graph>> graph_vec;
  graph_vec.emplace_back(std::move(graph1));
  graph_vec.emplace_back(std::move(graph2));
  graph_vec.emplace_back(std::move(graph3));

  auto es_c_graphs = GeGraphsToEsCGraphs(std::move(graph_vec));
  EXPECT_EQ(es_c_graphs.size(), 3);

  auto es_c_graph1 = static_cast<ge::Graph *>(static_cast<void *>(es_c_graphs[0]));
  auto es_c_graph2 = static_cast<ge::Graph *>(static_cast<void *>(es_c_graphs[1]));
  auto es_c_graph3 = static_cast<ge::Graph *>(static_cast<void *>(es_c_graphs[2]));
  ge::AscendString graph_name;
  EXPECT_EQ(es_c_graph1->GetName(graph_name), ge::GRAPH_SUCCESS);
  EXPECT_EQ(graph_name, ge::AscendString("test_graph1"));
  delete es_c_graph1;
  delete es_c_graph2;
  delete es_c_graph3;
}

TEST_F(EsGraphBuilderLLT, CreateTensor_float) {
  EsGraphBuilder builder("test_graph");
  std::vector<float> data = {-5.0};
  std::vector<int64_t> dims = {};
  auto tensor = builder.CreateTensor<float>(data, dims, ge::DT_FLOAT);
  EXPECT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->GetSize(), 4U);
  EXPECT_FLOAT_EQ((reinterpret_cast<const float *>(tensor->GetData()))[0], -5.0);
}
TEST_F(EsGraphBuilderLLT, CreateTensor_float16) {
  EsGraphBuilder builder("test_graph");
  std::vector<ge::fp16_t> data = {ge::fp16_t(5.0)};
  std::vector<int64_t> dims = {};
  auto tensor = builder.CreateTensor<ge::fp16_t>(data, dims, ge::DT_FLOAT16);
  EXPECT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->GetSize(), 2U);
  auto ret_data = reinterpret_cast<const ge::fp16_t *>(tensor->GetData())[0];
  EXPECT_TRUE(ret_data >= ge::fp16_t(static_cast<float>(4.999)) && ret_data <= ge::fp16_t(static_cast<float>(5.001)));
}
TEST_F(EsGraphBuilderLLT, CreateTensor_int8) {
  EsGraphBuilder builder("test_graph");
  std::vector<int8_t> data = {-5};
  std::vector<int64_t> dims = {};
  auto tensor = builder.CreateTensor<int8_t>(data, dims, ge::DT_INT8);
  EXPECT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->GetSize(), 1U);
  EXPECT_EQ((reinterpret_cast<const int8_t *>(tensor->GetData()))[0], -5);
}
TEST_F(EsGraphBuilderLLT, CreateTensor_int16) {
  EsGraphBuilder builder("test_graph");
  std::vector<int16_t> data = {-5};
  std::vector<int64_t> dims = {};
  auto tensor = builder.CreateTensor<int16_t>(data, dims, ge::DT_INT16);
  EXPECT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->GetSize(), 2U);
  EXPECT_EQ((reinterpret_cast<const int16_t *>(tensor->GetData()))[0], -5);
}
TEST_F(EsGraphBuilderLLT, CreateTensor_int32) {
  EsGraphBuilder builder("test_graph");
  std::vector<int32_t> data = {-5};
  std::vector<int64_t> dims = {};
  auto tensor = builder.CreateTensor<int32_t>(data, dims, ge::DT_INT32);
  EXPECT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->GetSize(), 4U);
  EXPECT_EQ((reinterpret_cast<const int32_t *>(tensor->GetData()))[0], -5);
}
TEST_F(EsGraphBuilderLLT, CreateTensor_int64) {
  EsGraphBuilder builder("test_graph");
  std::vector<int64_t> data = {-5};
  std::vector<int64_t> dims = {};
  auto tensor = builder.CreateTensor<int64_t>(data, dims, ge::DT_INT64);
  EXPECT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->GetSize(), 8U);
  EXPECT_EQ((reinterpret_cast<const int64_t *>(tensor->GetData()))[0], -5);
}
TEST_F(EsGraphBuilderLLT, CreateTensor_uint8) {
  EsGraphBuilder builder("test_graph");
  std::vector<uint8_t> data = {5};
  std::vector<int64_t> dims = {};
  auto tensor = builder.CreateTensor<uint8_t>(data, dims, ge::DT_UINT8);
  EXPECT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->GetSize(), 1U);
  EXPECT_EQ((reinterpret_cast<const uint8_t *>(tensor->GetData()))[0], 5);
}
TEST_F(EsGraphBuilderLLT, CreateTensor_uint16) {
  EsGraphBuilder builder("test_graph");
  std::vector<uint16_t> data = {5};
  std::vector<int64_t> dims = {};
  auto tensor = builder.CreateTensor<uint16_t>(data, dims, ge::DT_UINT16);
  EXPECT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->GetSize(), 2U);
  EXPECT_EQ((reinterpret_cast<const uint16_t *>(tensor->GetData()))[0], 5);
}
TEST_F(EsGraphBuilderLLT, CreateTensor_uint32) {
  EsGraphBuilder builder("test_graph");
  std::vector<uint32_t> data = {5};
  std::vector<int64_t> dims = {};
  auto tensor = builder.CreateTensor<uint32_t>(data, dims, ge::DT_UINT32);
  EXPECT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->GetSize(), 4U);
  EXPECT_EQ((reinterpret_cast<const uint32_t *>(tensor->GetData()))[0], 5);
}
TEST_F(EsGraphBuilderLLT, CreateTensor_uint64) {
  EsGraphBuilder builder("test_graph");
  std::vector<uint64_t> data = {5, 6};
  std::vector<int64_t> dims = {2};
  auto tensor = builder.CreateTensor<uint64_t>(data, dims, ge::DT_UINT64);
  EXPECT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->GetSize(), 16U);
  EXPECT_EQ((reinterpret_cast<const uint64_t *>(tensor->GetData()))[0], 5);
  EXPECT_EQ((reinterpret_cast<const uint64_t *>(tensor->GetData()))[1], 6);
}
TEST_F(EsGraphBuilderLLT, CreateTensor_bool) {
  EsGraphBuilder builder("test_graph");
  std::vector data = {true, false, true, false};
  std::vector<int64_t> dims = {4};
  auto tensor = builder.CreateBoolTensor(data, dims);
  EXPECT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->GetSize(), 4U);
  EXPECT_EQ((reinterpret_cast<const bool *>(tensor->GetData()))[0], true);
  EXPECT_EQ((reinterpret_cast<const bool *>(tensor->GetData()))[1], false);
  EXPECT_EQ((reinterpret_cast<const uint8_t *>(tensor->GetData()))[2], 1);
  EXPECT_EQ((reinterpret_cast<const uint8_t *>(tensor->GetData()))[3], 0);
}
TEST_F(EsGraphBuilderLLT, CreateTensor_unsupport) {
  std::vector<int64_t> data = {5};
  EsGraphBuilder builder("test_graph");
  std::vector<int64_t> dims = {};
  auto tensor = builder.CreateTensor<int64_t>(data, dims, ge::DT_STRING);
  EXPECT_EQ(tensor, nullptr);
}

TEST_F(EsGraphBuilderLLT, BuildStateCheck) {
  EsGraphBuilder builder("test_graph");
  auto graph = builder.BuildAndReset();
  EXPECT_NE(graph, nullptr);
  // cannot build again
  auto graph2 = builder.BuildAndReset();
  EXPECT_EQ(graph2, nullptr);
}

TEST_F(EsGraphBuilderLLT, AddEdgeAndUpdatePeerFormat) {
  ge::Graph g("test");
  g.SetValid();
  const char *op_type = "phony_1i1opi_1o";
  auto node0 = g.AddNodeByOp(ge::OperatorFactory::CreateOperator("op_test0", op_type));
  ge::TensorDesc src_td;
  EXPECT_EQ(node0.GetOutputDesc(0, src_td), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(src_td.GetShape().GetDims().empty());
  EXPECT_EQ(src_td.GetFormat(), ge::FORMAT_ND);
  EXPECT_EQ(src_td.GetDataType(), ge::DT_FLOAT);
  src_td.SetShape(ge::Shape({1, 1, 1, 1}));
  src_td.SetFormat(ge::FORMAT_NCHW);
  src_td.SetDataType(ge::DT_INT8);
  EXPECT_EQ(node0.UpdateOutputDesc(0, src_td), ge::GRAPH_SUCCESS);
  auto node1 = g.AddNodeByOp(ge::OperatorFactory::CreateOperator("op_test1", op_type));
  ge::TensorDesc dst_td;
  EXPECT_EQ(node1.GetInputDesc(0, dst_td), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(dst_td.GetShape().GetDims().empty());
  dst_td.SetFormat(ge::FORMAT_RESERVED);
  dst_td.SetDataType(ge::DT_UNDEFINED);
  node1.UpdateInputDesc(0, dst_td);
  EXPECT_EQ(ge::es::AddEdgeAndUpdatePeerDesc(g, node0, 0, node1, 0), ge::GRAPH_SUCCESS);
  EXPECT_EQ(node1.GetInputDesc(0, dst_td), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(dst_td.GetShape().GetDims().empty());
  EXPECT_EQ(dst_td.GetFormat(), ge::FORMAT_ND);
  EXPECT_EQ(dst_td.GetDataType(), ge::DT_FLOAT);
}
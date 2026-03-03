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

#include "macro_utils/dt_public_scope.h"
#include "graph/graph.h"
#include "graph/operator.h"
#include "graph/compute_graph.h"
#include "graph/normal_graph/compute_graph_impl.h"
#include "graph/op_desc.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/graph.h"
#include "graph/normal_graph/compute_graph_impl.h"
#include "common/util.h"
#include "common/proto_util.h"
#include "graph/utils/math_util.h"
#include "graph/passes/graph_builder_utils.h"
#include "graph/types.h"
#include "parser/tensorflow/tensorflow_parser.h"
#include "mmpa/mmpa_api.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
namespace formats {
class UtestUtilTransfer : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

int32_t mmAccess2(const char *pathName, int32_t mode) {
  return -1;
}

static ComputeGraphPtr BuildSubComputeGraph() {
  ut::GraphBuilder builder = ut::GraphBuilder("subgraph");
  auto data = builder.AddNode("sub_Data", "sub_Data", 0, 1);
  auto netoutput = builder.AddNode("sub_Netoutput", "sub_NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  return graph;
}

// construct graph which contains subgraph
static ComputeGraphPtr BuildComputeGraph() {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 0, 1);
  auto transdata = builder.AddNode("Transdata", "Transdata", 1, 1);
  transdata->GetOpDesc()->AddSubgraphName("subgraph");
  transdata->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph");
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, transdata, 0);
  builder.AddDataEdge(transdata, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  // add subgraph
  transdata->SetOwnerComputeGraph(graph);
  ComputeGraphPtr subgraph = BuildSubComputeGraph();
  subgraph->SetParentGraph(graph);
  subgraph->SetParentNode(transdata);
  graph->AddSubgraph("subgraph", subgraph);
  return graph;
}

TEST_F(UtestUtilTransfer, CheckOutputPathValid) {
  EXPECT_EQ(CheckOutputPathValid("", ""), false);
  EXPECT_EQ(CheckOutputPathValid("", "model"), false);

  char max_file_path[14097] = {0};
  memset(max_file_path, 1, 14097);
  EXPECT_EQ(CheckOutputPathValid(max_file_path, "model"), false);

  EXPECT_EQ(CheckOutputPathValid("$#%", ""), false);
  EXPECT_EQ(CheckOutputPathValid("./", ""), true);
  // system("touch test_util");
  // system("chmod 555 test_util");
  // EXPECT_EQ(CheckOutputPathValid("./test_util", ""), false);
  // system("rm -rf test_util");
}

TEST_F(UtestUtilTransfer, CheckInputPathValid) {
  EXPECT_EQ(CheckInputPathValid("", ""), false);
  EXPECT_EQ(CheckInputPathValid("", "model"), false);

  EXPECT_EQ(CheckInputPathValid("$#%", ""), false);

  EXPECT_EQ(CheckInputPathValid("./test_util", ""), false);
}

TEST_F(UtestUtilTransfer, GetFileLength_success) {
  system("rm -rf ./ut_graph1.txt");
  EXPECT_EQ(GetFileLength("./ut_graph1.txt"), -1);
  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);

  graph.SaveToFile("./ut_graph1.txt");
  EXPECT_NE(GetFileLength("./ut_graph1.txt"), -1);
}

TEST_F(UtestUtilTransfer, ReadBytesFromBinaryFile_success) {
  system("rm -rf ./ut_graph1.txt");
  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);

  graph.SaveToFile("./ut_graph1.txt");
  char* const_buffer = nullptr;
  int32_t len;
  EXPECT_EQ(ReadBytesFromBinaryFile("./ut_graph1.txt", &const_buffer, len), true);
  delete[] const_buffer;
}

TEST_F(UtestUtilTransfer, GetCurrentSecondTimestap) {
  EXPECT_NE(GetCurrentSecondTimestap(), FAILED);
  //ComputeGraphPtr cgp = BuildComputeGraph();
 // Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);
  //EXPECT_NE(GetCurrentSecondTimestap(), FAILED);
}

TEST_F(UtestUtilTransfer, ReadProtoFromText) {
  system("rm -rf ./ut_graph1.txt");
  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);
  graph.SaveToFile("./ut_graph1.txt");
  domi::tensorflow::GraphDefLibrary message;
  EXPECT_EQ(ReadProtoFromText("./ut_graph1.txt", &message), false);

  EXPECT_EQ(ReadProtoFromArray(nullptr, 0, nullptr), false);
  string str;
  EXPECT_EQ(GetFileLength(str), -1);
  str = "gegege";
  EXPECT_EQ(GetFileLength(str), -1);
  int32_t len = 0;
  EXPECT_EQ(ReadBytesFromBinaryFile(nullptr, nullptr, len), false);
  EXPECT_EQ(ReadBytesFromBinaryFile("gegege", nullptr, len), false);
  char a = 6;
  char *p1 = &a;
  char **p2 = &p1;
  EXPECT_EQ(ReadBytesFromBinaryFile("gegege", p2, len), false);
  EXPECT_EQ(ReadProtoFromText(nullptr, nullptr), false);
}

TEST_F(UtestUtilTransfer, GetAscendWorkPath_Env_Null_Failed) {
  mmSetEnv("ASCEND_WORK_PATH", "", 1);
  std::string ascend_work_path;
  auto ret = GetAscendWorkPath(ascend_work_path);
  EXPECT_EQ(ret, ge::FAILED);
  EXPECT_EQ(ascend_work_path, "");
  unsetenv("ASCEND_WORK_PATH");
}

TEST_F(UtestUtilTransfer, GetAscendWorkPath_Env_Invalid_Failed) {
  mmSetEnv("ASCEND_WORK_PATH", "", 1);
  std::string ascend_work_path;
  auto ret = GetAscendWorkPath(ascend_work_path);
  EXPECT_EQ(ret, ge::FAILED);
  EXPECT_EQ(ascend_work_path, "");
  unsetenv("ASCEND_WORK_PATH");
}

TEST_F(UtestUtilTransfer, GetAscendWorkPath_Success) {
  ge::char_t current_path[MMPA_MAX_PATH] = {'\0'};
  getcwd(current_path, MMPA_MAX_PATH);
  mmSetEnv("ASCEND_WORK_PATH", current_path, 1);
  std::string ascend_work_path;
  auto ret = GetAscendWorkPath(ascend_work_path);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(ascend_work_path, current_path);
  unsetenv("ASCEND_WORK_PATH");
}

// 1. 测试正常场景：单个配对
TEST_F(UtestUtilTransfer, Parse_Success_SinglePair) {
  std::string input = "0,0";
  std::vector<std::pair<size_t, size_t>> result;

  ParseOutputReuseInputMemIndexes(input, result);

  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].first, 0);
  EXPECT_EQ(result[0].second, 0);
}

// 2. 测试正常场景：多个配对
TEST_F(UtestUtilTransfer, Parse_Success_MultiPairs) {
  std::string input = "0,1|2,3|10,20";
  std::vector<std::pair<size_t, size_t>> result;

  ParseOutputReuseInputMemIndexes(input, result);

  EXPECT_EQ(result.size(), 3);
  // Check first pair
  EXPECT_EQ(result[0].first, 0);
  EXPECT_EQ(result[0].second, 1);
  // Check last pair
  EXPECT_EQ(result[2].first, 10);
  EXPECT_EQ(result[2].second, 20);
}

// 3. 测试空字符串
TEST_F(UtestUtilTransfer, Parse_EmptyString_NoOutput) {
  std::string input = "";
  std::vector<std::pair<size_t, size_t>> result;

  ParseOutputReuseInputMemIndexes(input, result);

  EXPECT_TRUE(result.empty());
}

// 4. 测试容错：包含空段 (||) 或 尾部竖线 (|)
TEST_F(UtestUtilTransfer, Parse_SkipEmptySegments) {
  // "||" 导致空子串，结尾 "|" 导致空子串
  std::string input = "1,1||2,2|";
  std::vector<std::pair<size_t, size_t>> result;

  ParseOutputReuseInputMemIndexes(input, result);

  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0].first, 1);
  EXPECT_EQ(result[1].first, 2);
}

// 5. 测试异常：包含负数 (应打印日志并跳过该项)
TEST_F(UtestUtilTransfer, Parse_InvalidNegative_Skip) {
  std::string input = "1,1|-1,0|2,2|0,-5";
  std::vector<std::pair<size_t, size_t>> result;

  ParseOutputReuseInputMemIndexes(input, result);

  // -1,0 和 0,-5 应该被跳过
  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0].first, 1);
  EXPECT_EQ(result[1].first, 2);
}

// 6. 测试异常：格式错误 (缺少逗号)
TEST_F(UtestUtilTransfer, Parse_InvalidFormatNoComma_Skip) {
  std::string input = "1,1|invalid|3,3";
  std::vector<std::pair<size_t, size_t>> result;

  ParseOutputReuseInputMemIndexes(input, result);

  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0].first, 1);
  EXPECT_EQ(result[1].first, 3);
}

// 7. 测试异常：非数字字符 (stoul 抛异常被 catch)
TEST_F(UtestUtilTransfer, Parse_InvalidChars_Skip) {
  std::string input = "a,b|1,1";
  std::vector<std::pair<size_t, size_t>> result;

  ParseOutputReuseInputMemIndexes(input, result);

  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].first, 1);
  EXPECT_EQ(result[0].second, 1);
}

// 8. 测试异常：超出整数范围
TEST_F(UtestUtilTransfer, Parse_InvalidRange_Skip) {
  std::string input = "99999999999999999999999999,1";
  std::vector<std::pair<size_t, size_t>> result;

  ParseOutputReuseInputMemIndexes(input, result);

  EXPECT_EQ(result.size(), 0);
}

class UtestIntegerChecker : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestIntegerChecker, All) {
#define DEFINE_VALUE(T)                                          \
  static uint64_t T##_max = std::numeric_limits<T>::max();       \
  static uint64_t u##T##_max = std::numeric_limits<u##T>::max(); \
  static int64_t T##_min = std::numeric_limits<T>::min();        \
  static int64_t u##T##_min = std::numeric_limits<u##T>::min();  \
  static uint64_t T##_up_overflow = T##_max + 1;                 \
  static uint64_t u##T##_up_overflow = u##T##_max + 1;           \
  static int64_t T##_lo_overflow = T##_min - 1;                  \
  static int64_t u##T##_lo_overflow = -1;

#define TEST_TYPE(T)                                              \
  EXPECT_TRUE(IntegerChecker<T>::Compat(T##_max));                \
  EXPECT_TRUE(IntegerChecker<T>::Compat(T##_min));                \
  EXPECT_TRUE(IntegerChecker<u##T>::Compat(u##T##_max));          \
  EXPECT_TRUE(IntegerChecker<u##T>::Compat(u##T##_min));          \
  EXPECT_FALSE(IntegerChecker<T>::Compat(T##_up_overflow));       \
  EXPECT_FALSE(IntegerChecker<T>::Compat(T##_lo_overflow));       \
  EXPECT_FALSE(IntegerChecker<u##T>::Compat(u##T##_up_overflow)); \
  EXPECT_FALSE(IntegerChecker<u##T>::Compat(u##T##_lo_overflow));

#define DEFINE_AND_TEST(T) \
  DEFINE_VALUE(T);         \
  TEST_TYPE(T);

  DEFINE_AND_TEST(int8_t);
  DEFINE_AND_TEST(int16_t);
  DEFINE_AND_TEST(int32_t);
  EXPECT_TRUE(IntegerChecker<int64_t>::Compat(std::numeric_limits<int64_t>::max()));
  EXPECT_TRUE(IntegerChecker<int64_t>::Compat(std::numeric_limits<int64_t>::min()));
  EXPECT_FALSE(IntegerChecker<int64_t>::Compat(std::numeric_limits<uint64_t>::max()));
  EXPECT_TRUE(IntegerChecker<uint64_t>::Compat(std::numeric_limits<uint64_t>::min()));
  EXPECT_TRUE(IntegerChecker<uint64_t>::Compat(std::numeric_limits<uint64_t>::max()));
  EXPECT_FALSE(IntegerChecker<uint64_t>::Compat(-1));
}
}  // namespace formats
}  // namespace ge

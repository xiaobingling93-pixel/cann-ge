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
#include "common/summary_checker.h"
#include "common/topo_checker.h"
#include "utils/graph_utils_ex.h"
#include "es_graph_builder.h"
#include "es_c_graph_builder.h"
#include "es_c_tensor_holder.h"
#include "esb_funcs.h"
#include "c_types.h"
#include "node_adapter.h"
#include "mmpa/mmpa_api.h"
#include <fstream>
#include <symengine/logic.h>
using namespace ge::es;

std::string GetTempDirectory() {
  return "/tmp";
}

class EsbFuncsLLT : public ::testing::Test {
 protected:
  void SetUp() override {
    _builder = new EsGraphBuilder("test");
  }
  void TearDown() override {
    delete _builder;
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

  template<typename T>
  void CreateBinaryFile(const std::vector<T>& data) {
    std::ofstream file(file_path, std::ios::binary);
    EXPECT_TRUE(file.is_open()) << "Failed to create file: " << file_path;
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
    file.close();
  }

  EsGraphBuilder *_builder = nullptr;
  std::string temp_dir;
  std::string file_path;
};

// 测试EsCreateVector和EsCreateScalar系列函数
TEST_F(EsbFuncsLLT, EsCreateVectorAndScalarFunctions) {
  auto *graph = EsCreateGraphBuilder("test_graph");
  ASSERT_NE(graph, nullptr);

  // 测试EsCreateVectorInt64
  int64_t vec_64_data[] = {1, 2, 3, 4};
  auto *vector64 = EsCreateVectorInt64(graph, vec_64_data, 4);
  EXPECT_NE(vector64, nullptr);

  // 测试EsCreateVectorInt32
  int32_t vec_32_data[] = {1, 2, 3, 4};
  auto *vector32 = EsCreateVectorInt32(graph, vec_32_data, 4);
  EXPECT_NE(vector32, nullptr);

  // 测试EsCreateVectorUInt64
  uint64_t vec_u64_data[] = {1, 2, 3, 4};
  auto *vectoru64 = EsCreateVectorUInt64(graph, vec_u64_data, 4);
  EXPECT_NE(vectoru64, nullptr);

  // 测试EsCreateVectorUInt32
  uint32_t vec_u32_data[] = {1, 2, 3, 4};
  auto *vectoru32 = EsCreateVectorUInt32(graph, vec_u32_data, 4);
  EXPECT_NE(vectoru32, nullptr);

  // 测试EsCreateVectorFloat
  float vec_float_data[] = {1.1, 2.2, 3.3, 4.0};
  auto *vectorf = EsCreateVectorFloat(graph, vec_float_data, 4);
  EXPECT_NE(vectorf, nullptr);

  // 测试EsCreateScalarInt64
  auto *scalar64 = EsCreateScalarInt64(graph, 42);
  EXPECT_NE(scalar64, nullptr);

  // 测试EsCreateScalarInt32
  auto *scalar32 = EsCreateScalarInt32(graph, 100);
  EXPECT_NE(scalar32, nullptr);

  // EsCreateScalarUInt64
  auto *scalaru64 = EsCreateScalarUInt64(graph, 100);
  EXPECT_NE(scalaru64, nullptr);

  // EsCreateScalarUInt32
  auto *scalaru32 = EsCreateScalarUInt32(graph, 100);
  EXPECT_NE(scalaru32, nullptr);

  // 测试EsCreateScalarFloat
  auto *scalarf = EsCreateScalarFloat(graph, 3.14f);
  EXPECT_NE(scalarf, nullptr);

  EsDestroyGraphBuilder(graph);
}

TEST_F(EsbFuncsLLT, EsCreateEsCTensorFromFile_success) {
  std::vector<int64_t> test_data = {1, 2, 3};
  CleanTmpFileDir();
  CreateTmpFileDir();
  CreateBinaryFile(test_data);
  EXPECT_TRUE(mmAccess(file_path.c_str()) == EN_OK);
  int64_t dims[] = {3};
  auto es_tensor =
      EsCreateEsCTensorFromFile(file_path.c_str(), dims, 1, C_DT_INT64, C_FORMAT_ALL);
    CleanTmpFileDir();
  EXPECT_NE(es_tensor, nullptr);
  auto inner_tensor = static_cast<const ge::Tensor *>(static_cast<void *>(es_tensor));
  EXPECT_NE(inner_tensor, nullptr);
  EXPECT_EQ((reinterpret_cast<const int64_t *>(inner_tensor->GetData()))[0], 1);
  EXPECT_EQ((reinterpret_cast<const int64_t *>(inner_tensor->GetData()))[1], 2);
  EXPECT_EQ((reinterpret_cast<const int64_t *>(inner_tensor->GetData()))[2], 3);
  delete inner_tensor;
  CleanTmpFileDir();
}

TEST_F(EsbFuncsLLT, EsCreateEsCTensorFromFile_success_scalar) {
  std::vector<int64_t> test_data = {1};
  CleanTmpFileDir();
  CreateTmpFileDir();
  CreateBinaryFile(test_data);
  EXPECT_TRUE(mmAccess(file_path.c_str()) == EN_OK);
  auto es_tensor = EsCreateEsCTensorFromFile(file_path.c_str(), nullptr, 0, C_DT_INT64, C_FORMAT_ALL);
  ASSERT_NE(es_tensor, nullptr);
  auto inner_tensor = static_cast<const ge::Tensor *>(static_cast<void *>(es_tensor));
  EXPECT_NE(inner_tensor, nullptr);
  EXPECT_EQ((reinterpret_cast<const int64_t *>(inner_tensor->GetData()))[0], 1);
  delete inner_tensor;
  CleanTmpFileDir();
}

TEST_F(EsbFuncsLLT, EsCreateEsCTensorFromFile_fail_of_file_size_wrong) {
  std::vector<int64_t> test_data = {1, 2, 3};
  CleanTmpFileDir();
  CreateTmpFileDir();
  CreateBinaryFile(test_data);
  EXPECT_TRUE(mmAccess(file_path.c_str()) == EN_OK);
  int64_t dims[] = {5};
  auto es_tensor = EsCreateEsCTensorFromFile(file_path.c_str(), dims, 1, C_DT_INT64, C_FORMAT_ALL);
  EXPECT_EQ(es_tensor, nullptr);
  CleanTmpFileDir();
}

TEST_F(EsbFuncsLLT, EsCreateEsCTensorFromFile_fail_of_file_size_wrong_scalar) {
  std::vector<int32_t> test_data = {1};
  CleanTmpFileDir();
  CreateTmpFileDir();
  CreateBinaryFile(test_data);
  EXPECT_TRUE(mmAccess(file_path.c_str()) == EN_OK);
  auto es_tensor = EsCreateEsCTensorFromFile(file_path.c_str(), nullptr, 0, C_DT_INT64, C_FORMAT_ALL);
  EXPECT_EQ(es_tensor, nullptr);
  CleanTmpFileDir();
}

TEST_F(EsbFuncsLLT, EsCreateEsCTensor_float) {
  std::vector<float> data = {5.0, 6.1};
  int64_t dims[] = {2};
  auto tmp = data.data();
  auto es_tensor = EsCreateEsCTensor(tmp, dims, 1, C_DT_FLOAT, C_FORMAT_ND);
  EXPECT_NE(es_tensor, nullptr);
  auto inner_tensor = static_cast<const ge::Tensor *>(static_cast<void *>(es_tensor));
  EXPECT_NE(inner_tensor, nullptr);
  EXPECT_FLOAT_EQ((reinterpret_cast<const float *>(inner_tensor->GetData()))[0], 5.0);
  EXPECT_FLOAT_EQ((reinterpret_cast<const float *>(inner_tensor->GetData()))[1], 6.1);
  delete inner_tensor;
}
TEST_F(EsbFuncsLLT, EsCreateEsCTensor_float16) {
  std::vector<float> data = {5.0};
  auto es_tensor = EsCreateEsCTensor(data.data(), nullptr, 1, C_DT_FLOAT16, C_FORMAT_ND);
  EXPECT_NE(es_tensor, nullptr);
  auto inner_tensor = static_cast<const ge::Tensor *>(static_cast<void *>(es_tensor));
  EXPECT_NE(inner_tensor, nullptr);
  delete inner_tensor;
}
TEST_F(EsbFuncsLLT, EsCreateEsCTensor_int8) {
  std::vector<int8_t> data = {5};
  auto es_tensor = EsCreateEsCTensor(data.data(), nullptr, 1, C_DT_INT8, C_FORMAT_ND);
  EXPECT_NE(es_tensor, nullptr);
  auto inner_tensor = static_cast<const ge::Tensor *>(static_cast<void *>(es_tensor));
  EXPECT_NE(inner_tensor, nullptr);
  delete inner_tensor;
}
TEST_F(EsbFuncsLLT, EsCreateEsCTensor_int16) {
  std::vector<int16_t> data = {5};
  auto es_tensor = EsCreateEsCTensor(data.data(), nullptr, 1, C_DT_INT16, C_FORMAT_ND);
  EXPECT_NE(es_tensor, nullptr);
  auto inner_tensor = static_cast<const ge::Tensor *>(static_cast<void *>(es_tensor));
  EXPECT_NE(inner_tensor, nullptr);
  delete inner_tensor;
}
TEST_F(EsbFuncsLLT, EsCreateEsCTensor_int32) {
  std::vector<int32_t> data = {5};
  auto es_tensor = EsCreateEsCTensor(data.data(), nullptr, 1, C_DT_INT32, C_FORMAT_ND);
  EXPECT_NE(es_tensor, nullptr);
  auto inner_tensor = static_cast<const ge::Tensor *>(static_cast<void *>(es_tensor));
  EXPECT_NE(inner_tensor, nullptr);
  delete inner_tensor;
}
TEST_F(EsbFuncsLLT, EsCreateEsCTensor_int64) {
  std::vector<int64_t> data = {5};
  auto es_tensor = EsCreateEsCTensor(data.data(), nullptr, 1, C_DT_INT64, C_FORMAT_ND);
  EXPECT_NE(es_tensor, nullptr);
  auto inner_tensor = static_cast<const ge::Tensor *>(static_cast<void *>(es_tensor));
  EXPECT_NE(inner_tensor, nullptr);
  delete inner_tensor;
}
TEST_F(EsbFuncsLLT, EsCreateEsCTensor_uint8) {
  std::vector<uint8_t> data = {5};
  auto es_tensor = EsCreateEsCTensor(data.data(), nullptr, 1, C_DT_UINT8, C_FORMAT_ND);
  EXPECT_NE(es_tensor, nullptr);
  auto inner_tensor = static_cast<const ge::Tensor *>(static_cast<void *>(es_tensor));
  EXPECT_NE(inner_tensor, nullptr);
  delete inner_tensor;
}
TEST_F(EsbFuncsLLT, EsCreateEsCTensor_uint16) {
  std::vector<uint16_t> data = {5};
  auto es_tensor = EsCreateEsCTensor(data.data(), nullptr, 1, C_DT_UINT16, C_FORMAT_ND);
  EXPECT_NE(es_tensor, nullptr);
  auto inner_tensor = static_cast<const ge::Tensor *>(static_cast<void *>(es_tensor));
  EXPECT_NE(inner_tensor, nullptr);
  delete inner_tensor;
}
TEST_F(EsbFuncsLLT, EsCreateEsCTensor_uint32) {
  std::vector<uint32_t> data = {5};
  auto es_tensor = EsCreateEsCTensor(data.data(), nullptr, 1, C_DT_UINT32, C_FORMAT_ND);
  EXPECT_NE(es_tensor, nullptr);
  auto inner_tensor = static_cast<const ge::Tensor *>(static_cast<void *>(es_tensor));
  EXPECT_NE(inner_tensor, nullptr);
  delete inner_tensor;
}
TEST_F(EsbFuncsLLT, EsCreateEsCTensor_uint64) {
  std::vector<uint64_t> data = {5, 6};
  int64_t dims[] = {2};
  auto es_tensor = EsCreateEsCTensor(data.data(), dims, 1, C_DT_UINT64, C_FORMAT_ND);
  EXPECT_NE(es_tensor, nullptr);
  auto inner_tensor = static_cast<const ge::Tensor *>(static_cast<void *>(es_tensor));
  EXPECT_NE(inner_tensor, nullptr);
  EXPECT_EQ((reinterpret_cast<const uint64_t *>(inner_tensor->GetData()))[0], 5);
  EXPECT_EQ((reinterpret_cast<const uint64_t *>(inner_tensor->GetData()))[1], 6);
  delete inner_tensor;
}
TEST_F(EsbFuncsLLT, EsCreateEsCTensor_bool) {
  std::vector<uint8_t> data = {true, false, true, false};
  int64_t dims[] = {4};
  auto es_tensor = EsCreateEsCTensor(data.data(), dims, 1, C_DT_BOOL, C_FORMAT_ND);
  EXPECT_NE(es_tensor, nullptr);
  auto inner_tensor = static_cast<const ge::Tensor *>(static_cast<void *>(es_tensor));
  EXPECT_NE(inner_tensor, nullptr);
  EXPECT_EQ((reinterpret_cast<const bool *>(inner_tensor->GetData()))[0], true);
  EXPECT_EQ((reinterpret_cast<const bool *>(inner_tensor->GetData()))[1], false);
  EXPECT_EQ((reinterpret_cast<const bool *>(inner_tensor->GetData()))[2], true);
  EXPECT_EQ((reinterpret_cast<const bool *>(inner_tensor->GetData()))[3], false);
  delete inner_tensor;
}
TEST_F(EsbFuncsLLT, EsCreateEsCTensor_unsupport) {
  std::vector<uint64_t> data = {5};
  auto es_tensor = EsCreateEsCTensor(data.data(), nullptr, 1, C_DT_STRING, C_FORMAT_ND);
  EXPECT_EQ(es_tensor, nullptr);
}

TEST_F(EsbFuncsLLT, EsAddControlEdge_success) {
  auto tensor0 = _builder->CreateInput(0).GetCTensorHolder();
  auto tensor1 = _builder->CreateInput(1).GetCTensorHolder();
  auto tensor2 = _builder->CreateInput(2).GetCTensorHolder();
  std::vector<EsCTensorHolder *> ctrl_ins = {tensor1, tensor2};
  EXPECT_EQ(ge::SUCCESS, EsAddControlEdge(tensor0, ctrl_ins.data(), 2));
  auto srd_node = tensor0->GetProducer();
  auto src_node_ctrl_ins = srd_node.GetInControlNodes();
  EXPECT_EQ(2, src_node_ctrl_ins.size());
  ge::AscendString ctrl_in_name;
  EXPECT_EQ(ge::GRAPH_SUCCESS, src_node_ctrl_ins.at(0)->GetName(ctrl_in_name));
  EXPECT_EQ(ge::AscendString("input_1"), ctrl_in_name.GetString());
  EXPECT_EQ(ge::GRAPH_SUCCESS, src_node_ctrl_ins.at(1)->GetName(ctrl_in_name));
  EXPECT_EQ(ge::AscendString("input_2"), ctrl_in_name.GetString());
}

TEST_F(EsbFuncsLLT, EsSetShap_failed) {
  auto graph_builder = ge::ComGraphMakeUnique<EsGraphBuilder>("tensor_holder_node_test4");
  auto t1 = graph_builder->CreateScalar(int64_t(111));
  EXPECT_NE(EsSetShape(t1.GetCTensorHolder(), nullptr, 1), ge::SUCCESS);
}

TEST_F(EsbFuncsLLT, EsSetOriginSymbolShape_failed) {
  auto graph_builder = ge::ComGraphMakeUnique<EsGraphBuilder>("tensor_holder_node_test4");
  auto t1 = graph_builder->CreateScalar(int64_t(111));
  EXPECT_NE(EsSetOriginSymbolShape(t1.GetCTensorHolder(), nullptr, 1), ge::SUCCESS);
}
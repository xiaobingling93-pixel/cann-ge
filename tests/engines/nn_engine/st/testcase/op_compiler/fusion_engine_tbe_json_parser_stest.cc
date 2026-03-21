/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <filesystem>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "fe_llt_utils.h"
#define protected public
#define private public
#include <graph/tensor.h>
#include "graph_optimizer/json_parser/tbe_json_parse.h"
#include "common/config_parser/op_debug_config_parser.h"
#include "common/fe_utils.h"
#include "common/configuration.h"
#include "common/fe_op_info_common.h"
#include "ops_store/op_kernel_info.h"
#include "ops_store/sub_op_info_store.h"
#include "ops_store/ops_kernel_manager.h"
#undef protected
#undef private

using namespace std;
using namespace fe;
using namespace ge;
using namespace nlohmann;
namespace fs = std::filesystem;


static Status ParseParams(const google::protobuf::Message* op_src, ge::Operator& op_dest)
{
    return fe::SUCCESS;
}

static Status InferShapeAndType(vector<ge::TensorDesc>& v_output_desc)
{
    return fe::SUCCESS;
}

static Status UpdateOpDesc(ge::Operator&)
{
    return fe::SUCCESS;
}

static Status GetWorkspaceSize(const ge::Operator&, std::vector<int64_t>&)
{
    return fe::SUCCESS;
}


static Status BuildTeBin(string& json_file_path, string& bin_file_path)
{
    return fe::SUCCESS;
}

static Status BuildTeBin1(string& json_file_path, string& bin_file_path)
{
    bin_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
    json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.json";
    return fe::SUCCESS;
}

static Status BuildTeBin2(string& json_file_path, string& bin_file_path)
{
    bin_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
    json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/cce_reductionLayer_1_10_float16__1_SUMSQ_1_with_so.json";
    return fe::SUCCESS;
}

static Status BuildTeBin5(string& json_file_path, string& bin_file_path)
{
    bin_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
    json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/cce_reduction_layer_1_10_float16__1_SUMSQ_1_0_null.json";
    return fe::SUCCESS;
}

static Status BuildTeBin6(string& json_file_path, string& bin_file_path)
{
    bin_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
    json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/cce_reduction_layer_1_10_float16__1_SUMSQ_1_0_error2.json";
    return fe::SUCCESS;
}

static Status BuildTeBin7(string& json_file_path, string& bin_file_path)
{
    bin_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reduction_layer_1_10_float16__1_SUMSQ_1_0_no_exist.o";
    json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.json";
    return fe::SUCCESS;
}

static Status BuildTeBin8(string& json_file_path, string& bin_file_path)
{
    bin_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
    json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/cce_reduction_layer_1_10_float16__1_SUMSQ_1_0_error3.json";
    return fe::SUCCESS;
}

static Status BuildTeBin9(string& json_file_path, string& bin_file_path)
{
    bin_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
    json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/cce_reduction_layer_1_10_float16__1_SUMSQ_1_0_noworkspace.json";
    return fe::SUCCESS;
}

static Status BuildTeBin10(string& json_file_path, string& bin_file_path)
{
    bin_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
    json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/cce_reduction_layer_1_10_float16__1_SUMSQ_1_0_error4.json";
    return fe::SUCCESS;
}

static Status BuildTeBin11(string& json_file_path, string& bin_file_path)
{
    bin_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
    json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/cce_reduction_layer_1_10_float16__1_SUMSQ_1_0_error5.json";
    return fe::SUCCESS;
}

static Status BuildTeBin12(string& json_file_path, string& bin_file_path)
{
    json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/cce_reduction_layer_1_10_float16__1_SUMSQ_1_0_not_exist.json";
    return fe::SUCCESS;
}

static Status BuildTeBin13(string& json_file_path, string& bin_file_path)
{
    bin_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
    json_file_path = GetCodeDir() + "/tests/engines/nn_engine/config/fe_config/atomic_test_parameters.json";
    return fe::SUCCESS;
}

static Status BuildTeBin14(string& json_file_path, string& bin_file_path)
{
    bin_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
    json_file_path = GetCodeDir() + "/tests/engines/nn_engine/config/fe_config/atomic_test_parameters_wrong_dtype.json";
    return fe::SUCCESS;
}

static Status BuildTeBin15(string& json_file_path, string& bin_file_path)
{
    bin_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
    json_file_path = GetCodeDir() + "/tests/engines/nn_engine/config/fe_config/atomic_test_parameters_workspace_not_equal.json";
    return fe::SUCCESS;
}

static Status BuildTeBin16(string& json_file_path, string& bin_file_path)
{
    bin_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
    json_file_path = GetCodeDir() + "/tests/engines/nn_engine/config/fe_config/atomic_test_parameters_size_not_equal.json";
    return fe::SUCCESS;
}

static Status BuildTeBin17(string& json_file_path, string& bin_file_path)
{
    bin_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
    json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/cce_reduction_layer_1_10_float16__1_SUMSQ_1_0_batch.json";
    return fe::SUCCESS;
}

static Status BuildTeBin18(string& json_file_path, string& bin_file_path)
{
    json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/te_conv2d_compress.json";
    return fe::SUCCESS;
}

static Status BuildTeBin19(string& json_file_path, string& bin_file_path)
{
  json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/mix_aic_task_ratio_0.json";
  return fe::SUCCESS;
}

static Status BuildTeBin20(string& json_file_path, string& bin_file_path)
{
  json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/mix_aiv_task_ratio_0.json";
  return fe::SUCCESS;
}

static Status BuildTeBin21(string& json_file_path, string& bin_file_path)
{
  json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/mix_aic_task_ratio_2.json";
  return fe::SUCCESS;
}

static Status BuildTeBin22(string& json_file_path, string& bin_file_path)
{
    json_file_path = GetCodeDir() + "/tests/engines/nn_engine/config/fe_config/atomic_test_new_parameters.json";
    return fe::SUCCESS;
}
static Status BuildTeBin23(string& json_file_path, string& bin_file_path)
{
  json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/mix_aic_optional.json";
  return fe::SUCCESS;
}

bool DumpFatbin() {
  uint64_t tilingKeyNum = 1;
  FatbinHeaderInfo fatbinHeadInfo(tilingKeyNum);
  fatbinHeadInfo.tilingKeyList.emplace_back(0);
  fatbinHeadInfo.binOffsets.emplace_back(1);
  size_t headSize = sizeof(fatbinHeadInfo.tilingKeyNum);
  size_t tilingKeyListSize = fatbinHeadInfo.tilingKeyList.size() * sizeof(uint64_t);
  size_t binOffsetListSize = fatbinHeadInfo.binOffsets.size() * sizeof(size_t);
  fatbinHeadInfo.binOffsets[0] = headSize + tilingKeyListSize + binOffsetListSize;
  std::vector<uint8_t> fatbinData(headSize + tilingKeyListSize + binOffsetListSize, 0);
  memcpy_s(fatbinData.data(), headSize, &fatbinHeadInfo.tilingKeyNum, headSize);
  memcpy_s(fatbinData.data() + headSize, tilingKeyListSize, fatbinHeadInfo.tilingKeyList.data(), tilingKeyListSize);
  memcpy_s(fatbinData.data() + headSize + tilingKeyListSize, binOffsetListSize,
           fatbinHeadInfo.binOffsets.data(), binOffsetListSize);
  std::string path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/ast_op_add.o";
  std::ofstream fatbinFile(path, std::ios::binary);
  if (!fatbinFile.is_open()) {
    return false;
  }
  fatbinFile.write(reinterpret_cast<const char*>(fatbinData.data()), fatbinData.size());
  KernelHeader kernelHeader;
  size_t offset = sizeof(kernelHeader);
  std::vector<uint8_t> opBinData = { 1 };
  std::vector<uint8_t> kernelBinData = { 1 };
  kernelHeader.dataOffset[static_cast<size_t>(KernelContextType::OpBinary)] = sizeof(kernelHeader);
  kernelHeader.dataSize[static_cast<size_t>(KernelContextType::OpBinary)] = opBinData.size();
  kernelHeader.dataOffset[static_cast<size_t>(KernelContextType::Kernel)] = sizeof(kernelHeader) + opBinData.size();
  kernelHeader.dataSize[static_cast<size_t>(KernelContextType::Kernel)] = kernelBinData.size();
  std::vector<uint8_t> binData(sizeof(kernelHeader), 0);
  memcpy_s(binData.data(), sizeof(kernelHeader), &kernelHeader, sizeof(kernelHeader));
  fatbinFile.write(reinterpret_cast<const char*>(binData.data()), binData.size());
  fatbinFile.write(reinterpret_cast<const char*>(opBinData.data()), opBinData.size());
  fatbinFile.write(reinterpret_cast<const char*>(kernelBinData.data()), kernelBinData.size());
  if (!fatbinFile.good()) {
    return false;
  }
  fatbinFile.close();
  return true;
}

class STEST_FE_TBE_JSON_PARSER: public testing::Test
{
protected:
    void SetUp()
    {

    }

    void TearDown()
    {

    }
public:

};


/*
*  测试函数：BuildCustomOp PackageTvmBinFile ParseTvmJsonFile
*  场景：正常json 算子bin
*  结果： fe::SUCCESS
 */
TEST_F(STEST_FE_TBE_JSON_PARSER, case_json_bin_all_success)
{
    OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>();
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr node = graph->AddNode(op_desc_ptr);
    TbeJsonFileParse json_file_parse(*node);
    std::string json_file_path;
    std::string bin_file_path;
    BuildTeBin1(json_file_path, bin_file_path);
    Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
    EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, case_json_bin_all_success_autofuse)
{
    OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("testdesc", "AscBackend");
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr node = graph->AddNode(op_desc_ptr);
    TbeJsonFileParse json_file_parse(*node);
    std::string json_file_path;
    std::string bin_file_path;
    BuildTeBin1(json_file_path, bin_file_path);
    Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
    EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, case_json_bin_mix_one_aic_taskratio_0_success)
{
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  std::string json_file_path;
  std::string bin_file_path;
  BuildTeBin19(json_file_path, bin_file_path);
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, case_vector_core_with_dy_ratio_suc)
{
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("Sigmoid", "sigmoid");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ffts/json/te_sigmoid_vector_core_mix_ratio.json";
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  EXPECT_NE(ret, fe::FAILED);
  bool dyn_ratio = false;
  (void)ge::AttrUtils::GetBool(op_desc_ptr, kMixDynamicRatio, dyn_ratio);
  EXPECT_EQ(dyn_ratio, true);
  vector<std::string> kernel_prefix_list;
  (void)ge::AttrUtils::GetListStr(op_desc_ptr, kKernelNamesPrefix, kernel_prefix_list);
  EXPECT_EQ(kernel_prefix_list.size(), 0);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, case_json_bin_mix_optionnal_success)
{
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("TEST_MIX_NODE", "TEST_MIX");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  std::string json_file_path;
  std::string bin_file_path;
  BuildTeBin23(json_file_path, bin_file_path);
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  EXPECT_NE(ret, fe::SUCCESS);
  ge::AttrUtils::SetInt(op_desc_ptr, FE_IMPLY_TYPE, 6);
  ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  EXPECT_NE(ret, fe::SUCCESS);

  FEOpsStoreInfo feOpsStoreInfo;
  OpKernelInfoPtr opKernelInfoPtr1 = std::make_shared<OpKernelInfo>("TEST_MIX");
  opKernelInfoPtr1->input_infos_.resize(2);
  SubOpInfoStorePtr subOpInfoStorePtr = std::make_shared<SubOpInfoStore>(feOpsStoreInfo);
  subOpInfoStorePtr->op_kernel_info_map_.emplace(std::make_pair("TEST_MIX", opKernelInfoPtr1));
  OpsKernelManager::Instance(AI_CORE_NAME).sub_ops_store_map_[EN_IMPL_HW_TBE] = subOpInfoStorePtr;
  op_desc_ptr->SetOpEngineName(AI_CORE_NAME);
  ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  EXPECT_EQ(ret, fe::SUCCESS);
  size_t all_size = 0;
  (void)ge::AttrUtils::GetInt(op_desc_ptr, kOpKernelAllInputSize, all_size);
  EXPECT_EQ(all_size, 2);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, case_json_bin_mix_one_aic_noratione_success)
{
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  std::string json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/mix_aic_task_no_ratio.json";
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, case_json_bin_mix_one_aiv_taskratio_0_success)
{
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  std::string json_file_path;
  std::string bin_file_path;
  BuildTeBin20(json_file_path, bin_file_path);
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, case_json_bin_mix_one_aic_taskratio_1_success)
{
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  std::string json_file_path;
  std::string bin_file_path;
  BuildTeBin21(json_file_path, bin_file_path);
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, case_json_bin_mix_one_aiv_taskratio_success)
{
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  std::string json_file_path;
  std::string bin_file_path;
  BuildTeBin20(json_file_path, bin_file_path);
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, case_not_exist_bin_failed)
{
    OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>();
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr node = graph->AddNode(op_desc_ptr);
    TbeJsonFileParse json_file_parse(*node);
    std::string json_file_path;
    std::string bin_file_path;
    BuildTeBin12(json_file_path, bin_file_path);
    Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
    EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, case_json_format_error_failed)
{
    OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>();
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr node = graph->AddNode(op_desc_ptr);
    TbeJsonFileParse json_file_parse(*node);
    string json_file_path;
    string bin_file_path;
    BuildTeBin10(json_file_path, bin_file_path);
    Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
    EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, case_json_bin_error2)
{
    string file_name = GetCodeDir() + "/tests/engines/nn_engine/stub/emptyfile";
    vector<char> buffer;
    TbeJsonFileParseImpl tbe_json_file_parse_impl;
    Status ret = tbe_json_file_parse_impl.ReadBytesFromBinaryFile(file_name.c_str(), buffer);
    EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, case_json_format_error_op_kb_hit)
{
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path;
  string bin_file_path;
  BuildTeBin11(json_file_path, bin_file_path);
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, case_json_manual_thread_mix_aic_aiv)
{
    OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("Sigmoid", "sigmoid");
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr node = graph->AddNode(op_desc_ptr);
    TbeJsonFileParse json_file_parse(*node);
    string json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/te_sigmoid_9a43f1.json";
    Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
    EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, set_omPath_node)
{
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/cce_reduction_layer_1_10_float16__1_SUMSQ_1_0_error3.json";
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  EXPECT_EQ(ret, fe::FAILED);
  json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/tbe_model_binary_test.json";
  CompileResultInfo result_info(json_file_path);
  result_info.bin_file_path = GetRealPath(GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/tbe_model_binary_test.om");
  ret = json_file_parse.PackageTvmJsonInfo(result_info);
  EXPECT_EQ(ret, fe::SUCCESS);
  std::string om_file_real_path;
  (void)ge::AttrUtils::GetStr(node->GetOpDesc(), "_om_binary_path", om_file_real_path);
  char resoved_path[260] =  {0x00};
  std::string om_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/tbe_model_binary_test.om";
  std::string om_real_path = realpath(om_path.c_str(), resoved_path);
  EXPECT_EQ(om_file_real_path, om_real_path);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_mem_check)
{
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("conv2d", "Conv2D");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/te_conv2d_compress.json";
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);

  // normal case
  Configuration &config = Configuration::Instance(AI_CORE_NAME);
  OpDebugConfigParserPtr op_debug_config_parser = std::dynamic_pointer_cast<OpDebugConfigParser>
                                                  (config.op_debug_config_parse_);
  op_debug_config_parser->enable_op_memory_check_ = true;
  ret = json_file_parse.ParseOpParaSize();
  EXPECT_EQ(ret, fe::SUCCESS);
  bool mem_check = false;
  (void)ge::AttrUtils::GetBool(json_file_parse.op_desc_, kMemoryCheck, mem_check);
  EXPECT_EQ(mem_check, false);

  json_file_parse.op_desc_->DelAttr(kMemoryCheck);
  (void)ge::AttrUtils::SetBool(json_file_parse.op_desc_, kOpDebugCompile, true);
  ret = json_file_parse.ParseOpParaSize();
  (void)ge::AttrUtils::GetBool(json_file_parse.op_desc_, kMemoryCheck, mem_check);
  EXPECT_EQ(mem_check, true);

  // check ori_op_para_size
  uint64_t ori_op_para_size = 0;
  (void)ge::AttrUtils::GetInt(json_file_parse.op_desc_, ORI_OP_PARA_SIZE, ori_op_para_size);
  EXPECT_EQ(ori_op_para_size, 10);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_op_para_size_and_ori_op_para_size_and_mem_check)
{
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("conv2d", "Conv2D");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/te_conv2d_compress.json";
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);

  // normal case
  Configuration &config = Configuration::Instance(AI_CORE_NAME);
  OpDebugConfigParserPtr op_debug_config_parser = std::dynamic_pointer_cast<OpDebugConfigParser>
                                                  (config.op_debug_config_parse_);
  op_debug_config_parser->enable_op_memory_check_ = true;
  ret = json_file_parse.ParseOpParaSize();
  EXPECT_EQ(ret, fe::SUCCESS);
  bool mem_check = false;
  (void)ge::AttrUtils::GetBool(json_file_parse.op_desc_, kMemoryCheck, mem_check);
  EXPECT_EQ(mem_check, false);

  json_file_parse.op_desc_->DelAttr(kMemoryCheck);
  (void)ge::AttrUtils::SetBool(json_file_parse.op_desc_, kOpDebugCompile, true);
  ret = json_file_parse.ParseOpParaSize();
  (void)ge::AttrUtils::GetBool(json_file_parse.op_desc_, kMemoryCheck, mem_check);
  EXPECT_EQ(mem_check, true);

  // check ori_op_para_size
  uint64_t ori_op_para_size = 0;
  (void)ge::AttrUtils::GetInt(json_file_parse.op_desc_, ORI_OP_PARA_SIZE, ori_op_para_size);
  EXPECT_EQ(ori_op_para_size, 10);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_opp_kernel_oom_check) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("conv2d", "Conv2D");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/te_conv2d_opp_kernel.json";
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  json_file_parse.ParseSupportInfo();
  bool mem_check = false;
  (void)ge::AttrUtils::GetBool(json_file_parse.op_desc_, kMemoryCheck, mem_check);
  EXPECT_EQ(mem_check, true);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_opp_kernel_oom_check2) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("conv2d", "Conv2D");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/te_conv2d_opp_kernel_2.json";
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  json_file_parse.ParseSupportInfo();
  bool mem_check = false;
  (void)ge::AttrUtils::GetBool(json_file_parse.op_desc_, kMemoryCheck, mem_check);
  EXPECT_EQ(mem_check, true);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, case_parse_tvm_parameters_success)
{
    ge::OpDescPtr op_desc_ = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    vector<int64_t> dims = {1, 2, 3, 4};
    GeShape shape(dims);
    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    op_desc_->AddInputDesc("x", in_desc1);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    op_desc_->AddOutputDesc("y1", out_desc1);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_HWCN);
    out_desc2.SetDataType(DT_FLOAT16);
    op_desc_->AddOutputDesc("y2", out_desc2);

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr node = graph->AddNode(op_desc_);
    TbeJsonFileParse json_file_parse(*node);
    string json_file_path;
    string bin_file_path;
    BuildTeBin22(json_file_path, bin_file_path);
    Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
    ret = json_file_parse.ParseTvmParameters();
    EXPECT_EQ(ret, fe::SUCCESS);
    
    std::vector<int32_t> dtype_list;
    std::vector<int32_t> dtype_list_check = {10, 0, 9, 8};
    ge::AttrUtils::GetListInt(op_desc_, TBE_OP_ATOMIC_DTYPES, dtype_list);
    EXPECT_EQ(dtype_list, dtype_list_check);

    std::vector<int64_t> init_value_int64_list;
    std::vector<int64_t> init_value_int64_list_check = {-1, -9223372036854775807, 4294967295};
    ge::AttrUtils::GetListInt(op_desc_, TBE_OP_ATOMIC_INT64_VALUES, init_value_int64_list);
    EXPECT_EQ(init_value_int64_list, init_value_int64_list_check);

    std::vector<float> init_value_float_list;
    std::vector<float> init_value_float_list_check = {3.40282e+38};
    ge::AttrUtils::GetListFloat(op_desc_, TBE_OP_ATOMIC_FLOAT_VALUES, init_value_float_list);
    EXPECT_EQ(init_value_float_list, init_value_float_list_check);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, case_parse_tvm_parameters_wrong_dtype)
{
    OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>();
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr node = graph->AddNode(op_desc_ptr);
    TbeJsonFileParse json_file_parse(*node);
    string json_file_path;
    string bin_file_path;
    BuildTeBin14(json_file_path, bin_file_path);
    Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
    ret = json_file_parse.ParseTvmParameters();
    EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, case_parse_tvm_wsp_mode_suc)
{
    OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>();
    (void)ge::AttrUtils::SetStr(op_desc_ptr, "_sgt_cube_vector_core_type", "VectorCore");
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr node = graph->AddNode(op_desc_ptr);
    TbeJsonFileParse json_file_parse(*node);
    string json_file_path;
    string bin_file_path;
    BuildTeBin22(json_file_path, bin_file_path);
    Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
    ret = json_file_parse.ParseTvmWspMode();
    EXPECT_EQ(ret, fe::SUCCESS);
    string wsp_mode = "";
    string wsp_mode_check = "folded";
    ge::AttrUtils::GetStr(op_desc_ptr, TBE_OP_ATOMIC_WSP_MODE, wsp_mode);
    EXPECT_EQ(wsp_mode, wsp_mode_check);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, set_op_debug_config_env_coverage) {
  OpDebugConfigParser op_debug_config_parser;
  const std::string env = "aaaaaaaaa";
  op_debug_config_parser.SetOpDebugConfigEnv(env);
  EXPECT_EQ(op_debug_config_parser.op_debug_config_, "dump_bin,dump_cce,dump_loc,ccec_g");
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_dfx_options_case1) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("sigmod", "sigmod");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/te_schedule_mode_1.json";
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  EXPECT_EQ(ret, fe::SUCCESS);
  std::vector<std::string> deb_opts;
  ge::AttrUtils::GetListStr(op_desc_ptr, kOpDfxOptions, deb_opts);
  EXPECT_EQ(deb_opts.size(), 2);
  int64_t buf_size = 0;
  ge::AttrUtils::GetInt(op_desc_ptr, kOpDfxBufferSize, buf_size);
  EXPECT_EQ(buf_size, 234567);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, case_parse_mix_core_type_suc)
{
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ffts/json/te_sigmoid_mix_ratio.json";
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  ret = json_file_parse.ParseTvmCoreType();
  EXPECT_EQ(ret, fe::SUCCESS);
  std::string core_type;
  ge::AttrUtils::GetStr(op_desc_ptr, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
  EXPECT_STREQ(core_type.c_str(), "MIX");
}

TEST_F(STEST_FE_TBE_JSON_PARSER, case_parse_custom_gentask_op_core_type_suc)
{
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("MatmulAllReduce", "MatmulAllReduce");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ffts/json/te_sigmoid_mix_ratio.json";
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  ret = json_file_parse.ParseTvmCoreType();
  EXPECT_EQ(ret, fe::SUCCESS);
  std::string core_type;
  ge::AttrUtils::GetStr(op_desc_ptr, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
  EXPECT_STREQ(core_type.c_str(), "MIX");
}

TEST_F(STEST_FE_TBE_JSON_PARSER, case_parse_option_output) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("conv2d", "Conv2D");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/mix_aic_optional.json";
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);

  json_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/mix_aic_optional_output.json";
  ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  ret = json_file_parse.ParseOptionalOutputMode();
  EXPECT_EQ(ret, fe::FAILED);
  AttrUtils::SetInt(op_desc_ptr, "_fe_imply_type", 6);
  ret = json_file_parse.ParseOptionalOutputMode();
  EXPECT_EQ(ret, fe::FAILED);

  string opt_output_mode;
  size_t all_output_size = 0;
  (void)ge::AttrUtils::GetStr(json_file_parse.op_desc_, "optionalOutputMode", opt_output_mode);
  EXPECT_EQ(opt_output_mode, kGenPlaceholder);
  (void)ge::AttrUtils::GetInt(json_file_parse.op_desc_, "_op_kernel_all_output_size", all_output_size);
  EXPECT_EQ(all_output_size, 0);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_simt_ubsize_case1) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("sigmod", "sigmod");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path = GetCodeDir() +
                          "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/te_simt_ub_size.json";
  string bin_file_path;
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  EXPECT_EQ(ret, fe::SUCCESS);
  uint32_t real_size = 0;
  ge::AttrUtils::GetInt(op_desc_ptr, kLocalMemorySize, real_size);
  EXPECT_EQ(real_size, 2334);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_ParseAndSetTilingInfo_suc) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("sigmod", "sigmod");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/te_tiling_info.json";
  string bin_file_path;
  json_file_parse.PackageTvmJsonInfo(json_file_path);
  Status ret = json_file_parse.ParseAndSetTilingInfo();
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_ParseAndSetTilingInfo_suc1) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("sigmod", "sigmod");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/te_tiling_info1.json";
  string bin_file_path;
  json_file_parse.PackageTvmJsonInfo(json_file_path);
  Status ret = json_file_parse.ParseAndSetTilingInfo();
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_ParseAndSetTilingInfo_fail) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("sigmod", "sigmod");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/te_tiling_info2.json";
  string bin_file_path;
  json_file_parse.PackageTvmJsonInfo(json_file_path);
  Status ret = json_file_parse.ParseAndSetTilingInfo();
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_ParseAndSetTilingInfo_fail2) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("sigmod", "sigmod");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/te_tiling_info3.json";
  string bin_file_path;
  json_file_parse.PackageTvmJsonInfo(json_file_path);
  Status ret = json_file_parse.ParseAndSetTilingInfo();
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_ParseAndSetTilingInfo_fail3) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("sigmod", "sigmod");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/te_tiling_info4.json";
  string bin_file_path;
  json_file_parse.PackageTvmJsonInfo(json_file_path);
  Status ret = json_file_parse.ParseAndSetTilingInfo();
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_ParseAndSetTilingInfo_fail4) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("sigmod", "sigmod");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/te_tiling_info5.json";
  string bin_file_path;
  json_file_parse.PackageTvmJsonInfo(json_file_path);
  Status ret = json_file_parse.ParseAndSetTilingInfo();
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_ParseAndSetTilingInfo_fail5) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("sigmod", "sigmod");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/te_tiling_info6.json";
  string bin_file_path;
  json_file_parse.PackageTvmJsonInfo(json_file_path);
  Status ret = json_file_parse.ParseAndSetTilingInfo();
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_ParseAndSetTilingInfo_fail6) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("sigmod", "sigmod");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/te_tiling_info7.json";
  string bin_file_path;
  json_file_parse.PackageTvmJsonInfo(json_file_path);
  Status ret = json_file_parse.ParseAndSetTilingInfo();
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_ParseAndSetTilingInfo_fail7) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("sigmod", "sigmod");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/te_tiling_info8.json";
  string bin_file_path;
  json_file_parse.PackageTvmJsonInfo(json_file_path);
  Status ret = json_file_parse.ParseAndSetTilingInfo();
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_ParseAndSetTilingInfo_fail8) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("sigmod", "sigmod");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/te_tiling_info9.json";
  string bin_file_path;
  json_file_parse.PackageTvmJsonInfo(json_file_path);
  Status ret = json_file_parse.ParseAndSetTilingInfo();
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_ParseAndSetTilingInfo_fail9) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("sigmod", "sigmod");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/te_tiling_info10.json";
  string bin_file_path;
  json_file_parse.PackageTvmJsonInfo(json_file_path);
  Status ret = json_file_parse.ParseAndSetTilingInfo();
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_ParseAndSetTilingInfo_fail10) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("sigmod", "sigmod");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/te_tiling_info11.json";
  string bin_file_path;
  json_file_parse.PackageTvmJsonInfo(json_file_path);
  Status ret = json_file_parse.ParseAndSetTilingInfo();
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_mc2_aicore_task) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("MoeDistributeDispatch", "MoeDistributeDispatch");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::AttrUtils::SetStr(op_desc_ptr, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ffts/json/te_sigmoid_mix_ratio.json";
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  string soc = PlatformUtils::Instance().short_soc_version_;
  PlatformUtils::Instance().short_soc_version_ = "Ascend910_93";
  (void)ge::AttrUtils::SetBool(op_desc_ptr, kTypeFFTSPlus, false);
  json_file_parse.ProcMixCoreType();
  PlatformUtils::Instance().short_soc_version_ = soc;
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_mc2_aicore_task_no_ffts) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("MoeDistributeDispatch", "MoeDistributeDispatch");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::AttrUtils::SetStr(op_desc_ptr, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ffts/json/te_sigmoid_mix_ratio.json";
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  string soc = PlatformUtils::Instance().short_soc_version_;
  PlatformUtils::Instance().short_soc_version_ = "Ascend910_93";
  (void)ge::AttrUtils::SetBool(op_desc_ptr, kTypeFFTSPlus, false);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::FftsMode)] = static_cast<int64_t>(FFTS_MODE_NO_FFTS);
  json_file_parse.ProcMixCoreType();
  PlatformUtils::Instance().short_soc_version_ = soc;
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_mix_task) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("MoeDistributeDispatch", "MoeDistributeDispatch");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::AttrUtils::SetStr(op_desc_ptr, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ffts/json/te_sigmoid_mix_ratio.json";
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  string soc = PlatformUtils::Instance().short_soc_version_;
  PlatformUtils::Instance().short_soc_version_ = "Ascend910_95";
  json_file_parse.ProcMixCoreType();
  bool task_type = false;
  ge::AttrUtils::GetBool(op_desc_ptr, kFftsplusTask, task_type);
  PlatformUtils::Instance().short_soc_version_ = soc;
  EXPECT_EQ(task_type, true);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_tile_fwk_parse_fatbin_info_fail1) {
  EXPECT_EQ(DumpFatbin(), true);
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("sigmod", "sigmod");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  (void)ge::AttrUtils::SetBool(op_desc_ptr, kAttrTileFwkOpStr, true);
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/ast_op_add_fail1.json";
  string bin_file_path;
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_tile_fwk_parse_fatbin_info_fail2) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("sigmod", "sigmod");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  (void)ge::AttrUtils::SetBool(op_desc_ptr, kAttrTileFwkOpStr, true);
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/ast_op_add_fail2.json";
  string bin_file_path;
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_tile_fwk_parse_fatbin_info_fail3) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("sigmod", "sigmod");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  (void)ge::AttrUtils::SetBool(op_desc_ptr, kAttrTileFwkOpStr, true);
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/ast_op_add_fail3.json";
  string bin_file_path;
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_tile_fwk_parse_fatbin_info_fail4) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("sigmod", "sigmod");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  (void)ge::AttrUtils::SetBool(op_desc_ptr, kAttrTileFwkOpStr, true);
  NodePtr node = graph->AddNode(op_desc_ptr);

  RunInfoPtr run_info = nullptr;
  run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  op_desc_ptr->SetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, run_info);
  Status ret = UpdateTileFwkKernelInfo(op_desc_ptr);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_FE_TBE_JSON_PARSER, test_tile_fwk_parse_fatbin_info_suc1) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("sigmod", "sigmod");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  (void)ge::AttrUtils::SetBool(op_desc_ptr, kAttrTileFwkOpStr, true);
  NodePtr node = graph->AddNode(op_desc_ptr);
  TbeJsonFileParse json_file_parse(*node);
  string json_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/ast_op_add.json";
  ASSERT_TRUE(fs::exists(json_file_path)) << "JSON file not found: " << json_file_path;
  Status ret = json_file_parse.PackageTvmJsonInfo(json_file_path);
  EXPECT_EQ(ret, fe::SUCCESS);

  RunInfoPtr run_info = nullptr;
  run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  op_desc_ptr->SetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, run_info);
  (void)ge::AttrUtils::SetInt(op_desc_ptr, ge::TVM_ATTR_NAME_BLOCKDIM, 24);
  ret = UpdateTileFwkKernelInfo(op_desc_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
}
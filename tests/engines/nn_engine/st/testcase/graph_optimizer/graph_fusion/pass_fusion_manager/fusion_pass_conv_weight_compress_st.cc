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
#include <nlohmann/json.hpp>
#include <iostream>
#include "fe_llt_utils.h"
#define protected public
#define private public
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "compiler/graph/common/compress/inc/compress.h"
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "platform/platform_info.h"
#include "fusion_manager/fusion_manager.h"
#include "ops_store/ops_kernel_manager.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/conv_weight_compress_fusion_pass.h"
#include "pass_manager.h"
#include "graph_optimizer/fusion_common/fusion_pass_manager.h"
#include "graph_optimizer/weight_compress_flag/weight_compress_judge.h"
#undef protected
#undef private

using namespace std;
using namespace ge;
using namespace fe;

class WeightCompressOptimizeUtilityST: public ge::OptimizeUtility {
 public:
  WeightCompressOptimizeUtilityST() {}
  virtual ~WeightCompressOptimizeUtilityST() override {}

  ge::Status InferShape(ComputeGraph &compute_graph) override{
    return ge::SUCCESS;
  }

  ge::Status InferShape(const ComputeGraphPtr &compute_graph) override {
    return ge::SUCCESS;
  }
#ifndef ONLY_COMPILE_BLUE
  ge::Status ConstantFolding(NodePtr &node) override {
    return ge::SUCCESS;
  }
#endif
};

class fusion_pass_conv_weight_compress_st : public testing::Test
{
public:
  FEGraphOptimizerPtr graph_optimizer_ptr;
  FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr;
  std::shared_ptr<OpCompilerNormal> op_compiler_ptr;
protected:
    void SetUp()
    {
      ops_kernel_info_store_ptr = make_shared<FEOpsKernelInfoStore>(AI_CORE_NAME);
      FEOpsStoreInfo tbe_custom {
              2,
              "tbe-custom",
              EN_IMPL_CUSTOM_TBE,
              GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
              GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
              false,
              false,
              false};

      vector<FEOpsStoreInfo> store_info = {tbe_custom};
      Configuration::Instance(AI_CORE_NAME).ops_store_info_vector_ = (store_info);
      OpsKernelManager::Instance(AI_CORE_NAME).Finalize();
      map<string, string> options;
      ops_kernel_info_store_ptr->Initialize(options);
      WeightCompressOptimizeUtilityST *optimize_utility_stub = new WeightCompressOptimizeUtilityST();
      graph_optimizer_ptr = make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr, AI_CORE_NAME);
      op_compiler_ptr = make_shared<OpCompilerNormal>("normal compiler", AI_CORE_NAME, nullptr);
    }
    void TearDown()
    {

    }

protected:
  static ComputeGraphPtr CreateGraphWithOneConv(DataType data_type, int32_t type=0, bool is_dynamic=false,
                                                std::vector<int64_t> dim={4, 4, 4, 4})
  {
      ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
      OpDescPtr op_desc_data = std::make_shared<OpDesc>("data", "Data");
      OpDescPtr op_desc_const1 = std::make_shared<OpDesc>("const1", "Const");
      OpDescPtr op_desc_const2 = std::make_shared<OpDesc>("const2", "Const");
      OpDescPtr op_desc_conv = std::make_shared<OpDesc>("conv", "Conv2D");
      OpDescPtr op_desc_relu = std::make_shared<OpDesc>("relu", "Relu");

      //add descriptor
      if (is_dynamic) {
        dim[1] = -1;
      }
      GeShape shape(dim);
      GeTensorDesc out_desc(shape);
      out_desc.SetFormat(FORMAT_NCHW);
      out_desc.SetOriginFormat(FORMAT_NCHW);
      out_desc.SetDataType(data_type);
      out_desc.SetOriginDataType(data_type);

      op_desc_data->AddOutputDesc(out_desc);
      op_desc_const1->AddOutputDesc(out_desc);
      op_desc_const2->AddOutputDesc(out_desc);
      op_desc_conv->AddInputDesc(out_desc);
      op_desc_conv->AddInputDesc(out_desc);
      op_desc_conv->AddInputDesc(out_desc);
      op_desc_conv->AddOutputDesc(out_desc);
      op_desc_relu->AddInputDesc(out_desc);
      op_desc_relu->AddOutputDesc(out_desc);

      NodePtr node_data = graph->AddNode(op_desc_data);
      NodePtr node_const1 = graph->AddNode(op_desc_const1);
      NodePtr node_const2 = graph->AddNode(op_desc_const2);
      NodePtr node_conv = graph->AddNode(op_desc_conv);
      NodePtr node_relu = graph->AddNode(op_desc_relu);

      if (!is_dynamic) {
        ge::GeTensorPtr tensor = std::make_shared<ge::GeTensor>();
        tensor->SetTensorDesc(out_desc);
        auto data_size = out_desc.GetShape().GetShapeSize() * ge::GetSizeByDataType(out_desc.GetDataType());
        unique_ptr<uint8_t[]> value(new(std::nothrow) uint8_t[data_size]);
        (void)memset_s(value.get(), data_size, 1, data_size);
        (void)tensor->SetData(reinterpret_cast<uint8_t *>(value.get()), data_size);
        std::vector<ge::GeTensorPtr> weight_tensors = {tensor};
        (void)OpDescUtils::SetWeights(node_const1, weight_tensors);
      }

      if (type == 1) {
        ge::AttrUtils::SetBool(node_conv->GetOpDesc(), ATTR_NAME_COMPRESS_WEIGHT, true);
        GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_conv->GetInDataAnchor(0));
        GraphUtils::AddEdge(node_const1->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
        GraphUtils::AddEdge(node_const2->GetOutDataAnchor(0), node_conv->GetInDataAnchor(2));
        GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_conv->GetInDataAnchor(1));
      } if (type == 2) {
        ge::AttrUtils::SetInt(op_desc_conv, ATTR_NAME_GROUPS, 2);
        GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_conv->GetInDataAnchor(0));
        GraphUtils::AddEdge(node_const1->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
        GraphUtils::AddEdge(node_const2->GetOutDataAnchor(0), node_conv->GetInDataAnchor(2));
        GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_conv->GetInDataAnchor(1));
      } if (type == 3) {
        AttrUtils::SetBool(node_conv->GetOpDesc(), ATTR_NAME_COMPRESS_WEIGHT, true);
        GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_conv->GetInDataAnchor(0));
        GraphUtils::AddEdge(node_const1->GetOutDataAnchor(0), node_conv->GetInDataAnchor(1));
        GraphUtils::AddEdge(node_const2->GetOutDataAnchor(0), node_conv->GetInDataAnchor(2));
        GraphUtils::AddEdge(node_conv->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
        GraphUtils::AddEdge(node_data->GetOutControlAnchor(), node_conv->GetInControlAnchor());
        GraphUtils::AddEdge(node_conv->GetOutControlAnchor(), node_relu->GetInControlAnchor());
      } else if (type == 4) {
        GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_conv->GetInDataAnchor(0));
        GraphUtils::AddEdge(node_const1->GetOutDataAnchor(0), node_conv->GetInDataAnchor(1));
        GraphUtils::AddEdge(node_const2->GetOutDataAnchor(0), node_conv->GetInDataAnchor(2));
        GraphUtils::AddEdge(node_conv->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
      } else {
        AttrUtils::SetBool(node_conv->GetOpDesc(), ATTR_NAME_COMPRESS_WEIGHT, true);
        GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_conv->GetInDataAnchor(0));
        GraphUtils::AddEdge(node_const1->GetOutDataAnchor(0), node_conv->GetInDataAnchor(1));
        GraphUtils::AddEdge(node_const2->GetOutDataAnchor(0), node_conv->GetInDataAnchor(2));
        GraphUtils::AddEdge(node_conv->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
      }
      return graph;
  }

  static ComputeGraphPtr CreateGraphWithOneConvCompress(int64_t weight_compress_type = 0)
  {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    OpDescPtr op_desc_data = std::make_shared<OpDesc>("data", "Data");
    OpDescPtr op_desc_quant = std::make_shared<OpDesc>("quant", "AscendQuant");
    OpDescPtr op_desc_conv = std::make_shared<OpDesc>("conv_compress", "Conv2DCompress");
    OpDescPtr op_desc_dequant = std::make_shared<OpDesc>("dequant", "AscendDequant");
    OpDescPtr op_desc_relu = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr op_desc_const = std::make_shared<OpDesc>("const", "Const");

    //add descriptor
    vector<int64_t> dim(4, 4);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);
    out_desc.SetFormat(FORMAT_NCHW);
    out_desc.SetOriginFormat(FORMAT_NCHW);
    out_desc.SetDataType(DT_INT8);
    out_desc.SetOriginDataType(DT_INT8);

    op_desc_data->AddOutputDesc(out_desc);
    op_desc_quant->AddInputDesc(out_desc);
    op_desc_quant->AddOutputDesc(out_desc);
    op_desc_conv->AddInputDesc(out_desc);
    op_desc_conv->AddInputDesc(out_desc);
    op_desc_conv->AddInputDesc(out_desc);
    op_desc_conv->AddInputDesc(out_desc);
    op_desc_conv->AddOutputDesc(out_desc);
    op_desc_dequant->AddInputDesc(out_desc);
    op_desc_dequant->AddOutputDesc(out_desc);
    op_desc_relu->AddInputDesc(out_desc);
    op_desc_const->AddOutputDesc(out_desc);

    vector<string> input_name_vec;
    input_name_vec.push_back("x");
    input_name_vec.push_back("filter");
    input_name_vec.push_back("bias");
    op_desc_conv->SetInputName(input_name_vec);

    NodePtr node_data = graph->AddNode(op_desc_data);
    NodePtr node_quant = graph->AddNode(op_desc_quant);
    NodePtr node_conv = graph->AddNode(op_desc_conv);
    NodePtr node_dequant = graph->AddNode(op_desc_dequant);
    NodePtr node_relu = graph->AddNode(op_desc_relu);
    NodePtr node_const = graph->AddNode(op_desc_const);

    AttrUtils::SetBool(node_conv->GetOpDesc(), ATTR_NAME_COMPRESS_WEIGHT, true);
    std::vector<int64_t> compress_param_vec = {2, 128};
    AttrUtils::SetListInt(node_conv->GetOpDesc(), ATTR_NAME_COMPRESS_PARAMETERS, compress_param_vec);
    if (weight_compress_type != -1) {
      AttrUtils::SetInt(node_conv->GetOpDesc(), ATTR_NAME_WEIGHT_COMPRESS_TYPE, weight_compress_type);
    }

    GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_quant->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_quant->GetOutDataAnchor(0), node_conv->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_const->GetOutDataAnchor(0), node_conv->GetInDataAnchor(1));
    GraphUtils::AddEdge(node_conv->GetOutDataAnchor(0), node_dequant->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_dequant->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));

    return graph;
  }
};

TEST_F(fusion_pass_conv_weight_compress_st, fusion_success_case1)
{
  ComputeGraphPtr graph = CreateGraphWithOneConv(ge::DT_FLOAT);
  ConvWeightCompressFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  EXPECT_EQ(fe::NOT_CHANGED, status);
}

TEST_F(fusion_pass_conv_weight_compress_st, fusion_success_case2)
{
  ComputeGraphPtr graph = CreateGraphWithOneConv(ge::DT_INT8);
  ConvWeightCompressFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  EXPECT_EQ(fe::NOT_CHANGED, status);
}

TEST_F(fusion_pass_conv_weight_compress_st, fusion_success_case3)
{
  ComputeGraphPtr graph = CreateGraphWithOneConv(ge::DT_INT8, 0, true);
  ConvWeightCompressFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  EXPECT_EQ(fe::NOT_CHANGED, status);
}

TEST_F(fusion_pass_conv_weight_compress_st, fusion_success_case4)
{
  PlatformInfo platform_info;
  platform_info.soc_info.ai_core_cnt = 32;
  std::string soc_version = PlatformUtils::Instance().GetSocVersion();
  PlatformInfoManager::Instance().platform_info_map_[soc_version] = platform_info;
  PlatformInfoManager::Instance().opti_compilation_info_.soc_version = soc_version;
  ComputeGraphPtr graph = CreateGraphWithOneConv(ge::DT_INT8);
  ConvWeightCompressFusionPass pass;
  vector<GraphPass*> passes = {&pass};

  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(fusion_pass_conv_weight_compress_st, fusion_success_case5)
{
  ComputeGraphPtr graph = CreateGraphWithOneConv(ge::DT_INT8, 1);
  ConvWeightCompressFusionPass pass;
  vector<GraphPass*> passes = {&pass};

  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(fe::NOT_CHANGED, status);
}

TEST_F(fusion_pass_conv_weight_compress_st, fusion_success_case6)
{
  ComputeGraphPtr graph = CreateGraphWithOneConv(ge::DT_INT8, 2);
  ConvWeightCompressFusionPass pass;
  vector<GraphPass*> passes = {&pass};

  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(fe::NOT_CHANGED, status);
}

TEST_F(fusion_pass_conv_weight_compress_st, fusion_success_case7)
{
  ComputeGraphPtr graph = CreateGraphWithOneConv(ge::DT_INT8, 3);
  ConvWeightCompressFusionPass pass;
  vector<GraphPass*> passes = {&pass};

  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(fusion_pass_conv_weight_compress_st, insert_compress_case1)
{
  ComputeGraphPtr graph = CreateGraphWithOneConvCompress();
  size_t size_before = graph->GetDirectNode().size();
  FE_LOGD("The number of nodes before is %zu.", size_before);

  // insert compress node
  Status status = graph_optimizer_ptr->HandleCompressOp(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    status = op_compiler_ptr->InsertCompressOp(node);
    EXPECT_EQ(fe::SUCCESS, status);
  }

  bool find_compress = false;
  vector<int64_t> param_vec = {2,128};
  vector<int64_t> index_shape = {128};
  for (NodePtr node : graph->GetDirectNode()) {
    OpDescPtr op_desc = node->GetOpDesc();
    if (node->GetType() == "Compress") {
      find_compress = true;
      EXPECT_EQ(op_desc->GetInputNameByIndex(0), "weight");
      EXPECT_EQ(op_desc->GetOutputNameByIndex(0), "weight_compress");
      EXPECT_EQ(op_desc->GetOutputNameByIndex(1), "compress_index");
      EXPECT_EQ(op_desc->GetOutputDescPtr(1)->GetShape().GetDims(), index_shape);
      vector<int64_t> compress_param_vec;
      ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_COMPRESS_PARAMETERS, compress_param_vec);
      EXPECT_EQ(compress_param_vec, param_vec);
      int64_t weight_compress_type = -1;
      (void)ge::AttrUtils::GetInt(op_desc, ATTR_NAME_WEIGHT_COMPRESS_TYPE, weight_compress_type);
      EXPECT_EQ(weight_compress_type, 0);
    }
    if (node->GetType() == "Conv2DCompress") {
      EXPECT_EQ(op_desc->GetInputDescPtr(2)->GetShape().GetDims(), index_shape);
    }
  }
  EXPECT_EQ(find_compress, true);

  status = graph_optimizer_ptr->HandleCompressOp(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    status = op_compiler_ptr->InsertCompressOp(node);
    EXPECT_EQ(fe::SUCCESS, status);
  }
}

namespace fe {
  extern void ProcCompressOpMemType(const ge::NodePtr &node);
}
TEST_F(fusion_pass_conv_weight_compress_st, ProcCompressOpMemType_case)
{
  ComputeGraphPtr graph = CreateGraphWithOneConvCompress();
  auto conv_node = graph->FindNode("conv_compress");
  auto relu_node = graph->FindNode("relu");
  (void)ge::AttrUtils::SetBool(conv_node->GetOpDesc(), kTypeFFTSPlus, true);
  ProcCompressOpMemType(conv_node);
  std::vector<int64_t> memory_type;
  (void)ge::AttrUtils::GetListInt(conv_node->GetOpDesc(), ge::ATTR_NAME_INPUT_MEM_TYPE_LIST, memory_type);
  EXPECT_EQ(memory_type.size(), conv_node->GetOpDesc()->GetInputsSize());
  ProcCompressOpMemType(relu_node);
}
TEST_F(fusion_pass_conv_weight_compress_st, fusion_sparsity_case1)
{
  PlatformInfo platform_info;
  platform_info.ai_core_spec.sparsity = 1;
  std::string soc_version = PlatformUtils::Instance().GetSocVersion();
  PlatformInfoManager::Instance().platform_info_map_[soc_version] = platform_info;
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::SparseMatrixWeight)] = 1;
  ComputeGraphPtr graph = CreateGraphWithOneConv(ge::DT_INT8);
  ConvWeightCompressFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(fe::SUCCESS, status);
}


TEST_F(fusion_pass_conv_weight_compress_st, fusion_sparsity_case2)
{
  PlatformInfo platform_info;
  platform_info.ai_core_spec.sparsity = 0;
  std::string soc_version = PlatformUtils::Instance().GetSocVersion();
  PlatformInfoManager::Instance().platform_info_map_[soc_version] = platform_info;
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::SparseMatrixWeight)] = 1;
  ComputeGraphPtr graph = CreateGraphWithOneConv(ge::DT_INT8);
  ConvWeightCompressFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(fe::SUCCESS, status);
}


TEST_F(fusion_pass_conv_weight_compress_st, fusion_sparsity_case3)
{
  PlatformInfo platform_info;
  platform_info.ai_core_spec.sparsity = 1;
  std::string soc_version = PlatformUtils::Instance().GetSocVersion();
  PlatformInfoManager::Instance().platform_info_map_[soc_version] = platform_info;
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::SparseMatrixWeight)] = 0;
  ComputeGraphPtr graph = CreateGraphWithOneConv(ge::DT_INT8);
  ConvWeightCompressFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(fusion_pass_conv_weight_compress_st, insert_compress_case2)
{
  PlatformInfo platform_info;
  platform_info.ai_core_spec.sparsity = 1;
  std::string soc_version = PlatformUtils::Instance().GetSocVersion();
  PlatformInfoManager::Instance().platform_info_map_[soc_version] = platform_info;
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::SparseMatrixWeight)] = 1;
  ComputeGraphPtr graph = CreateGraphWithOneConvCompress();
  size_t size_before = graph->GetDirectNode().size();
  FE_LOGD("The number of nodes before is %zu.", size_before);

  // insert compress node
  Status status = graph_optimizer_ptr->HandleCompressOp(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    status = op_compiler_ptr->InsertCompressOp(node);
    EXPECT_EQ(fe::SUCCESS, status);
  }

  std::cout <<"xxxxxxx num = " << graph->GetDirectNode().size()<< std::endl;
  EXPECT_EQ(fe::SUCCESS, status);
  bool find_compress = false;
  for (NodePtr node : graph->GetDirectNode()) {
    if (node->GetType() == "Compress") {
      find_compress = true;
      OpDescPtr op_desc = node->GetOpDesc();
      EXPECT_EQ(op_desc->GetInputNameByIndex(0), "weight");
      EXPECT_EQ(op_desc->GetOutputNameByIndex(0), "weight_compress");
      EXPECT_EQ(op_desc->GetOutputNameByIndex(1), "compress_index");
      int64_t weight_compress_type = -1;
      (void)ge::AttrUtils::GetInt(op_desc, ATTR_NAME_WEIGHT_COMPRESS_TYPE, weight_compress_type);
      EXPECT_EQ(weight_compress_type, -1);
    }
  }
  EXPECT_EQ(find_compress, true);
}

TEST_F(fusion_pass_conv_weight_compress_st, insert_compress_case3)
{
  PlatformInfo platform_info;
  platform_info.ai_core_spec.sparsity = 1;
  std::string soc_version = PlatformUtils::Instance().GetSocVersion();
  PlatformInfoManager::Instance().platform_info_map_[soc_version] = platform_info;
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::SparseMatrixWeight)] = 1;

  ComputeGraphPtr graph = CreateGraphWithOneConvCompress();
  size_t size_before = graph->GetDirectNode().size();
  FE_LOGD("The number of nodes before is %zu.", size_before);

  // insert compress node
  Status status = graph_optimizer_ptr->HandleCompressOp(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    status = op_compiler_ptr->InsertCompressOp(node);
    EXPECT_EQ(fe::SUCCESS, status);
  }
  std::cout <<"xxxxxxx num = " << graph->GetDirectNode().size()<< std::endl;
  EXPECT_EQ(fe::SUCCESS, status);
  bool find_compress = false;
  for (NodePtr node : graph->GetDirectNode()) {
    if (node->GetType() == "Compress") {
      find_compress = true;
      OpDescPtr op_desc = node->GetOpDesc();
      EXPECT_EQ(op_desc->GetInputNameByIndex(0), "weight");
      EXPECT_EQ(op_desc->GetOutputNameByIndex(0), "weight_compress");
      EXPECT_EQ(op_desc->GetOutputNameByIndex(1), "compress_index");
      int64_t weight_compress_type = -1;
      (void)ge::AttrUtils::GetInt(op_desc, ATTR_NAME_WEIGHT_COMPRESS_TYPE, weight_compress_type);
      EXPECT_EQ(weight_compress_type, -1);
    }
  }
  EXPECT_EQ(find_compress, true);
}

// get attr weight compress type valid
TEST_F(fusion_pass_conv_weight_compress_st, insert_compress_case4)
{
  PlatformInfo platform_info;
  platform_info.ai_core_spec.sparsity = 0;
  std::string soc_version = PlatformUtils::Instance().GetSocVersion();
  PlatformInfoManager::Instance().platform_info_map_[soc_version] = platform_info;
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::SparseMatrixWeight)] = 0;
  ComputeGraphPtr graph = CreateGraphWithOneConvCompress(1);
  size_t size_before = graph->GetDirectNode().size();
  FE_LOGD("The number of nodes before is %zu.", size_before);

  // insert compress node
  Status status = graph_optimizer_ptr->HandleCompressOp(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    status = op_compiler_ptr->InsertCompressOp(node);
    EXPECT_EQ(fe::SUCCESS, status);
  }

  bool find_compress = false;
  vector<int64_t> param_vec = {2,128};
  vector<int64_t> index_shape = {128};
  for (NodePtr node : graph->GetDirectNode()) {
    OpDescPtr op_desc = node->GetOpDesc();
    if (node->GetType() == "Compress") {
      find_compress = true;
      EXPECT_EQ(op_desc->GetInputNameByIndex(0), "weight");
      EXPECT_EQ(op_desc->GetOutputNameByIndex(0), "weight_compress");
      EXPECT_EQ(op_desc->GetOutputNameByIndex(1), "compress_index");
      EXPECT_EQ(op_desc->GetOutputDescPtr(1)->GetShape().GetDims(), index_shape);
      vector<int64_t> compress_param_vec;
      ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_COMPRESS_PARAMETERS, compress_param_vec);
      int64_t weight_compress_type = -1;
      (void)ge::AttrUtils::GetInt(op_desc, ATTR_NAME_WEIGHT_COMPRESS_TYPE, weight_compress_type);
      EXPECT_EQ(weight_compress_type, 1);
    }
    if (node->GetType() == "Conv2DCompress") {
      EXPECT_EQ(op_desc->GetInputDescPtr(2)->GetShape().GetDims(), index_shape);
    }
  }
  EXPECT_EQ(find_compress, true);

  status = graph_optimizer_ptr->HandleCompressOp(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    status = op_compiler_ptr->InsertCompressOp(node);
    EXPECT_EQ(fe::SUCCESS, status);
  }
}

// can not get attr
TEST_F(fusion_pass_conv_weight_compress_st, insert_compress_case5)
{
  ComputeGraphPtr graph = CreateGraphWithOneConvCompress(-1);
  size_t size_before = graph->GetDirectNode().size();
  FE_LOGD("The number of nodes before is %zu.", size_before);

  // insert compress node
  Status status = graph_optimizer_ptr->HandleCompressOp(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    status = op_compiler_ptr->InsertCompressOp(node);
    if (node->GetType() == "Conv2DCompress") {
      EXPECT_EQ(fe::FAILED, status);
    } else {
      EXPECT_EQ(fe::SUCCESS, status);
    }
  }
}

// get attr weight compress type invalid
TEST_F(fusion_pass_conv_weight_compress_st, insert_compress_case6)
{
  ComputeGraphPtr graph = CreateGraphWithOneConvCompress(3);
  size_t size_before = graph->GetDirectNode().size();
  FE_LOGD("The number of nodes before is %zu.", size_before);

  // insert compress node
  Status status = graph_optimizer_ptr->HandleCompressOp(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    status = op_compiler_ptr->InsertCompressOp(node);
    if (node->GetType() == "Conv2DCompress") {
      EXPECT_EQ(fe::FAILED, status);
    } else {
      EXPECT_EQ(fe::SUCCESS, status);
    }
  }
}

// weight data is not not multiple of 512 -> disable compress
TEST_F(fusion_pass_conv_weight_compress_st, weight_compress_judge_case1)
{
  PlatformInfo platform_info;
  platform_info.soc_info.ai_core_cnt = 32;
  std::string soc_version = PlatformUtils::Instance().GetSocVersion();
  PlatformInfoManager::Instance().platform_info_map_[soc_version] = platform_info;
  PlatformInfoManager::Instance().opti_compilation_info_.soc_version = soc_version;
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::CompressWeight)] = 1;
  ComputeGraphPtr graph = CreateGraphWithOneConv(ge::DT_INT8, 0, false, {16, 16, 1});
  ConvWeightCompressFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(fe::SUCCESS, status);

  WeightCompressOptimizeUtilityST *optimize_utility_stub = new WeightCompressOptimizeUtilityST();
  status = WeightCompressJudge::CompressTypeJudge(optimize_utility_stub, *graph);
  EXPECT_EQ(fe::SUCCESS, status);

  int64_t weight_compress_type = -1;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "WeightCompressHost" || node->GetType() == "Conv2DCompress") {
      (void)ge::AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_WEIGHT_COMPRESS_TYPE, weight_compress_type);
      EXPECT_EQ(weight_compress_type, -1);
    }
  }
}

TEST_F(fusion_pass_conv_weight_compress_st, weight_compress_judge_case2)
{
  PlatformInfo platform_info;
  platform_info.soc_info.ai_core_cnt = 32;
  std::string soc_version = PlatformUtils::Instance().GetSocVersion();
  PlatformInfoManager::Instance().platform_info_map_[soc_version] = platform_info;
  PlatformInfoManager::Instance().opti_compilation_info_.soc_version = soc_version;
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::CompressWeight)] = 1;
  ComputeGraphPtr graph = CreateGraphWithOneConv(ge::DT_INT8, 0, false, {16, 16, 2});
  ConvWeightCompressFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(fe::SUCCESS, status);

  WeightCompressOptimizeUtilityST *optimize_utility_stub = new WeightCompressOptimizeUtilityST();
  status = WeightCompressJudge::CompressTypeJudge(optimize_utility_stub, *graph);
  EXPECT_EQ(fe::SUCCESS, status);

  int64_t weight_compress_type = -1;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "WeightCompressHost" || node->GetType() == "Conv2DCompress") {
      (void)ge::AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_WEIGHT_COMPRESS_TYPE, weight_compress_type);
      EXPECT_EQ(weight_compress_type, 0);
    }
  }

  ge::NodePtr weight_node = graph->FindNode("const1");
  weight_compress_type = static_cast<int64_t>(WeightCompressJudge::CompareCompressType(weight_node, true));
  EXPECT_EQ(weight_compress_type, 1);

  weight_node->GetOpDesc()->DelAttr(ge::ATTR_NAME_WEIGHTS);
  weight_compress_type = static_cast<int64_t>(WeightCompressJudge::CompareCompressType(weight_node, true));
  EXPECT_EQ(weight_compress_type, 2);
}

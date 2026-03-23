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
#include "fe_llt_utils.h"
#include "common/fe_op_info_common.h"
#include "common/scope_allocator.h"
#include "compute_graph.h"
#include "graph/utils/attr_utils.h"
#include "itf_handler/itf_handler.h"

#define protected public
#define private public
#include "fusion_manager/fusion_manager.h"
#undef private
#undef protected

using namespace ge;

namespace fe {
class OptimizeOriginalGraphProcess910BTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "OptimizeOriginalGraphProcess910BTest TearDown" << endl;
    InitWithSocVersion("Ascend910B1", "must_keep_origin_dtype");
    FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
    map<string, string> options;
    EXPECT_EQ(graph_optimizer_ptr->Initialize(options, nullptr), SUCCESS);
  }

  static void TearDownTestCase() {
    cout << "OptimizeOriginalGraphProcess910BTest TearDown" << endl;
    Finalize();
  }

  ge::ComputeGraphPtr CreateGraphWithComplexDtype() {
    std::vector<int64_t> dim_nhwc = {32, 14, 14, 64};
    ge::GeShape shape_nhwc(dim_nhwc);
    ge::GeTensorDesc tensor_desc_fp32(shape_nhwc, ge::FORMAT_NHWC, ge::DT_FLOAT);
    tensor_desc_fp32.SetOriginShape(shape_nhwc);
    tensor_desc_fp32.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc_fp32.SetOriginFormat(ge::FORMAT_NHWC);

    ge::GeTensorDesc tensor_desc_cpx64(shape_nhwc, ge::FORMAT_NHWC, ge::DT_COMPLEX64);
    tensor_desc_cpx64.SetOriginShape(shape_nhwc);
    tensor_desc_cpx64.SetOriginDataType(ge::DT_COMPLEX64);
    tensor_desc_cpx64.SetOriginFormat(ge::FORMAT_NHWC);

    OpDescPtr data1_op = std::make_shared<OpDesc>("data1", "Data");
    OpDescPtr data2_op = std::make_shared<OpDesc>("data2", "Data");
    OpDescPtr net_output_op = std::make_shared<OpDesc>("net_output", "NetOutput");

    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr complex_op = std::make_shared<OpDesc>("complex", "Complex");
    OpDescPtr complex_abs_op = std::make_shared<OpDesc>("complex_abs", "ComplexAbs");
    OpDescPtr square_op = std::make_shared<OpDesc>("square", "Square");

    data1_op->AddOutputDesc(tensor_desc_fp32);
    data2_op->AddOutputDesc(tensor_desc_fp32);
    relu_op->AddInputDesc(tensor_desc_fp32);
    relu_op->AddOutputDesc(tensor_desc_fp32);
    complex_op->AddInputDesc(tensor_desc_fp32);
    complex_op->AddInputDesc(tensor_desc_fp32);
    complex_op->AddOutputDesc(tensor_desc_cpx64);
    complex_abs_op->AddInputDesc(tensor_desc_cpx64);
    complex_abs_op->AddOutputDesc(tensor_desc_fp32);
    square_op->AddInputDesc(tensor_desc_fp32);
    square_op->AddOutputDesc(tensor_desc_fp32);
    net_output_op->AddInputDesc(tensor_desc_fp32);

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data1_node = graph->AddNode(data1_op);
    NodePtr data2_node = graph->AddNode(data2_op);
    NodePtr relu_node = graph->AddNode(relu_op);
    NodePtr complex_node = graph->AddNode(complex_op);
    NodePtr complex_abs_node = graph->AddNode(complex_abs_op);
    NodePtr square_node = graph->AddNode(square_op);
    NodePtr net_output_node = graph->AddNode(net_output_op);

    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), complex_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), complex_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(complex_node->GetOutDataAnchor(0), complex_abs_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(complex_abs_node->GetOutDataAnchor(0), square_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(square_node->GetOutDataAnchor(0), net_output_node->GetInDataAnchor(0));

    graph->TopologicalSorting();
    AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "11_22");
    return graph;
  }

  ge::ComputeGraphPtr CreateGraphWithNLLLossOp(const ge::DataType data_type) {
    std::vector<int64_t> dim_nhwc = {32, 14, 14, 64};
    ge::GeShape shape_nhwc(dim_nhwc);
    ge::GeTensorDesc tensor_desc_fp(shape_nhwc, ge::FORMAT_NHWC, data_type);
    tensor_desc_fp.SetOriginShape(shape_nhwc);
    tensor_desc_fp.SetOriginDataType(data_type);
    tensor_desc_fp.SetOriginFormat(ge::FORMAT_NHWC);

    ge::GeTensorDesc tensor_desc_int(shape_nhwc, ge::FORMAT_NHWC, ge::DT_INT32);
    tensor_desc_int.SetOriginShape(shape_nhwc);
    tensor_desc_int.SetOriginDataType(ge::DT_INT32);
    tensor_desc_int.SetOriginFormat(ge::FORMAT_NHWC);

    OpDescPtr data1_op = std::make_shared<OpDesc>("data1", "Data");
    OpDescPtr data2_op = std::make_shared<OpDesc>("data2", "Data");
    OpDescPtr data3_op = std::make_shared<OpDesc>("data3", "Data");
    OpDescPtr nll_loss_op = std::make_shared<OpDesc>("nll_loss_op", "NLLLoss");
    OpDescPtr netoutput_op = std::make_shared<OpDesc>("netoutput", "NetOutput");
    data1_op->AddOutputDesc(tensor_desc_fp);
    data2_op->AddOutputDesc(tensor_desc_int);
    data3_op->AddOutputDesc(tensor_desc_fp);
    nll_loss_op->AddInputDesc("x", tensor_desc_fp);
    nll_loss_op->AddInputDesc("target", tensor_desc_int);
    nll_loss_op->AddInputDesc("weight", tensor_desc_fp);
    nll_loss_op->AddOutputDesc("y", tensor_desc_fp);
    nll_loss_op->AddOutputDesc("total_weight", tensor_desc_fp);
    netoutput_op->AddInputDesc(tensor_desc_fp);
    netoutput_op->AddInputDesc(tensor_desc_fp);

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data1_node = graph->AddNode(data1_op);
    NodePtr data2_node = graph->AddNode(data2_op);
    NodePtr data3_node = graph->AddNode(data3_op);
    NodePtr nll_loss_node = graph->AddNode(nll_loss_op);
    NodePtr net_output_node = graph->AddNode(netoutput_op);

    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), nll_loss_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), nll_loss_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(data3_node->GetOutDataAnchor(0), nll_loss_node->GetInDataAnchor(2));
    GraphUtils::AddEdge(nll_loss_node->GetOutDataAnchor(0), net_output_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(nll_loss_node->GetOutDataAnchor(1), net_output_node->GetInDataAnchor(1));

    AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "11_22");
    return graph;
  }

  ge::ComputeGraphPtr CreateGraphWithDeformableRoiPoolOp(const ge::DataType data_type) {
    std::vector<int64_t> dim_nhwc = {32, 14, 14, 64};
    ge::GeShape shape_nhwc(dim_nhwc);
    ge::GeTensorDesc tensor_desc_fp(shape_nhwc, ge::FORMAT_NHWC, data_type);
    tensor_desc_fp.SetOriginShape(shape_nhwc);
    tensor_desc_fp.SetOriginDataType(data_type);
    tensor_desc_fp.SetOriginFormat(ge::FORMAT_NHWC);

    OpDescPtr data1_op = std::make_shared<OpDesc>("data1", "Data");
    OpDescPtr data2_op = std::make_shared<OpDesc>("data2", "Data");
    OpDescPtr data3_op = std::make_shared<OpDesc>("data3", "Data");
    OpDescPtr de_pool_op = std::make_shared<OpDesc>("deformable_roi_pool", "DeformableRoiPool");
    OpDescPtr netoutput_op = std::make_shared<OpDesc>("netoutput", "NetOutput");
    data1_op->AddOutputDesc(tensor_desc_fp);
    data2_op->AddOutputDesc(tensor_desc_fp);
    data3_op->AddOutputDesc(tensor_desc_fp);
    de_pool_op->AddInputDesc("x", tensor_desc_fp);
    de_pool_op->AddInputDesc("rois", tensor_desc_fp);
    de_pool_op->AddInputDesc("offset", tensor_desc_fp);
    de_pool_op->AddOutputDesc("y", tensor_desc_fp);
    netoutput_op->AddInputDesc(tensor_desc_fp);

    ge::AttrUtils::SetListInt(de_pool_op, "output_size", {1,1,1,1});

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data1_node = graph->AddNode(data1_op);
    NodePtr data2_node = graph->AddNode(data2_op);
    NodePtr data3_node = graph->AddNode(data3_op);
    NodePtr de_pool_node = graph->AddNode(de_pool_op);
    NodePtr net_output_node = graph->AddNode(netoutput_op);

    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), de_pool_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), de_pool_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(data3_node->GetOutDataAnchor(0), de_pool_node->GetInDataAnchor(2));
    GraphUtils::AddEdge(de_pool_node->GetOutDataAnchor(0), net_output_node->GetInDataAnchor(0));

    AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "11_22");
    return graph;
  }

  ge::ComputeGraphPtr CreateGraphWithConv2DBackpropFilterD() {
    OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
    OpDescPtr weight_op = std::make_shared<OpDesc>("weight1", "Const");
    OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2DBackpropFilterD");
    OpDescPtr cast1_op = std::make_shared<OpDesc>("cast", "Cast");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr cast2_op = std::make_shared<OpDesc>("cast", "Cast");
    OpDescPtr netoutput_op = std::make_shared<OpDesc>("netoutput", "NetOutput");

    ge::GeShape shape({8,224,224,3});
    ge::GeTensorDesc tensor_nchw_fp32(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_nchw_fp32.SetOriginShape(shape);
    tensor_nchw_fp32.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_nchw_fp32.SetOriginDataType(ge::DT_FLOAT);
    ge::GeTensorDesc tensor_nchw_fp16(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    tensor_nchw_fp32.SetOriginShape(shape);
    tensor_nchw_fp32.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_nchw_fp32.SetOriginDataType(ge::DT_FLOAT16);

    data_op->AddOutputDesc(tensor_nchw_fp32);
    weight_op->AddOutputDesc(tensor_nchw_fp32);
    conv_op->AddInputDesc("x", tensor_nchw_fp32);
    conv_op->AddInputDesc("out_backprop", tensor_nchw_fp32);
    conv_op->AddOutputDesc("y", tensor_nchw_fp32);
    cast1_op->AddInputDesc("x", tensor_nchw_fp32);
    cast1_op->AddOutputDesc("y", tensor_nchw_fp16);
    relu_op->AddInputDesc("x", tensor_nchw_fp16);
    relu_op->AddOutputDesc("y", tensor_nchw_fp16);
    cast2_op->AddInputDesc("x", tensor_nchw_fp16);
    cast2_op->AddOutputDesc("y", tensor_nchw_fp32);
    netoutput_op->AddInputDesc("x", tensor_nchw_fp32);

    AttrUtils::SetListInt(conv_op, "filter_size", {1,1,1,1});
    AttrUtils::SetListInt(conv_op, "strides", {1,1,1,1});
    AttrUtils::SetListInt(conv_op, "pads", {1,1,1,1});

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data_node = graph->AddNode(data_op);
    NodePtr weight_node = graph->AddNode(weight_op);
    NodePtr conv_node = graph->AddNode(conv_op);
    NodePtr cast1_node = graph->AddNode(cast1_op);
    NodePtr relu_node = graph->AddNode(relu_op);
    NodePtr cast2_node = graph->AddNode(cast2_op);
    NodePtr net_output_node = graph->AddNode(netoutput_op);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(weight_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), cast1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(cast1_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), cast2_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(cast2_node->GetOutDataAnchor(0), net_output_node->GetInDataAnchor(0));
    AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "11_22");
    return graph;
  }

  ge::ComputeGraphPtr CreateGraphWithQuantMatmulOp() {
    std::vector<int64_t> dim_nhwc = {32, 14, 14, 64};
    ge::GeShape shape_nhwc(dim_nhwc);
    ge::GeTensorDesc tensor_desc_hif8(shape_nhwc, ge::FORMAT_NHWC, ge::DT_HIFLOAT8);
    tensor_desc_hif8.SetOriginShape(shape_nhwc);
    tensor_desc_hif8.SetOriginDataType(ge::DT_HIFLOAT8);
    tensor_desc_hif8.SetOriginFormat(ge::FORMAT_NHWC);

    ge::GeTensorDesc tensor_desc_fp32(shape_nhwc, ge::FORMAT_NHWC, ge::DT_FLOAT);
    tensor_desc_fp32.SetOriginShape(shape_nhwc);
    tensor_desc_fp32.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc_fp32.SetOriginFormat(ge::FORMAT_NHWC);

    ge::GeTensorDesc tensor_desc_uint64(shape_nhwc, ge::FORMAT_NHWC, ge::DT_UINT64);
    tensor_desc_uint64.SetOriginShape(shape_nhwc);
    tensor_desc_uint64.SetOriginDataType(ge::DT_UINT64);
    tensor_desc_uint64.SetOriginFormat(ge::FORMAT_NHWC);

    OpDescPtr data1_op = std::make_shared<OpDesc>("data1", "Data");
    OpDescPtr data2_op = std::make_shared<OpDesc>("data2", "Data");
    OpDescPtr data3_op = std::make_shared<OpDesc>("data3", "Data");
    OpDescPtr quant_matmulv3_op = std::make_shared<OpDesc>("quant_matmulv3", "QuantBatchMatnulV3");
    OpDescPtr netoutput_op = std::make_shared<OpDesc>("netoutput", "NetOutput");
    data1_op->AddOutputDesc(tensor_desc_fp32);
    data2_op->AddOutputDesc(tensor_desc_fp32);
    data3_op->AddOutputDesc(tensor_desc_uint64);
    quant_matmulv3_op->AddInputDesc("x1", tensor_desc_hif8);
    quant_matmulv3_op->AddInputDesc("x2", tensor_desc_hif8);
    quant_matmulv3_op->AddInputDesc("scale", tensor_desc_uint64);
    quant_matmulv3_op->AddOutputDesc("y", tensor_desc_hif8);
    netoutput_op->AddInputDesc(tensor_desc_fp32);

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data1_node = graph->AddNode(data1_op);
    NodePtr data2_node = graph->AddNode(data2_op);
    NodePtr data3_node = graph->AddNode(data3_op);
    NodePtr quant_matmulv3_node = graph->AddNode(quant_matmulv3_op);
    NodePtr net_output_node = graph->AddNode(netoutput_op);

    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), quant_matmulv3_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), quant_matmulv3_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(data3_node->GetOutDataAnchor(0), quant_matmulv3_node->GetInDataAnchor(2));
    GraphUtils::AddEdge(quant_matmulv3_node->GetOutDataAnchor(0), net_output_node->GetInDataAnchor(0));

    AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "11_22");
    return graph;
  }
};

TEST_F(OptimizeOriginalGraphProcess910BTest, optimize_origin_graph_complex_case1) {
  FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  ComputeGraphPtr graph = CreateGraphWithComplexDtype();
  SetPrecisionMode("must_keep_origin_dtype");
  Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 7);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 7);
  size_t trans_count = 0;
  size_t cast_count = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    std::cout << "==== " << op_desc->GetName() << " - " << op_desc->GetType() << std::endl;
    if (op_desc->GetType() == "TransData") {
      trans_count++;
    }
    if (op_desc->GetType() == "Cast") {
      cast_count++;
    }
  }
  EXPECT_EQ(trans_count, 0);
  EXPECT_EQ(cast_count, 0);
}

TEST_F(OptimizeOriginalGraphProcess910BTest, optimize_origin_graph_complex_case2) {
  FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  ComputeGraphPtr graph = CreateGraphWithComplexDtype();
  SetPrecisionMode("force_fp16");
  Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 7);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 11);
  size_t trans_count = 0;
  size_t cast_count = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    std::cout << "==== " << op_desc->GetName() << " - " << op_desc->GetType() << std::endl;
    if (op_desc->GetType() == "TransData") {
      trans_count++;
    }
    if (op_desc->GetType() == "Cast") {
      cast_count++;
    }
  }
  EXPECT_EQ(trans_count, 0);
  EXPECT_EQ(cast_count, 4);
}

TEST_F(OptimizeOriginalGraphProcess910BTest, optimize_origin_graph_complex_case3) {
  FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  ComputeGraphPtr graph = CreateGraphWithComplexDtype();
  SetPrecisionMode("allow_fp32_to_fp16");
  Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 7);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 7);
  size_t trans_count = 0;
  size_t cast_count = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    std::cout << "==== " << op_desc->GetName() << " - " << op_desc->GetType() << std::endl;
    if (op_desc->GetType() == "TransData") {
      trans_count++;
    }
    if (op_desc->GetType() == "Cast") {
      cast_count++;
    }
  }
  EXPECT_EQ(trans_count, 0);
  EXPECT_EQ(cast_count, 0);
}

TEST_F(OptimizeOriginalGraphProcess910BTest, optimize_origin_graph_nll_loss_judge_fp16) {
  FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  ComputeGraphPtr graph = CreateGraphWithNLLLossOp(ge::DT_FLOAT16);
  SetPrecisionMode("must_keep_origin_dtype");
  Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 9);

  size_t cast_count = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    std::cout << "==== " << op_desc->GetName() << " - " << op_desc->GetType() << std::endl;
    if (op_desc->GetType() == "NLLLoss") {
      EXPECT_EQ(op_desc->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(op_desc->GetInputDescPtr(1)->GetDataType(), ge::DT_INT32);
      EXPECT_EQ(op_desc->GetInputDescPtr(2)->GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(op_desc->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(op_desc->GetOutputDescPtr(1)->GetDataType(), ge::DT_FLOAT);
    }
    if (op_desc->GetType() == "Cast") {
      cast_count++;
    }
  }
  EXPECT_EQ(cast_count, 4);
}

TEST_F(OptimizeOriginalGraphProcess910BTest, optimize_origin_graph_nll_loss_judge_bf16) {
  FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  ComputeGraphPtr graph = CreateGraphWithNLLLossOp(ge::DT_BF16);
  SetPrecisionMode("must_keep_origin_dtype");
  Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);

  size_t cast_count = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    std::cout << "==== " << op_desc->GetName() << " - " << op_desc->GetType() << std::endl;
    if (op_desc->GetType() == "NLLLoss") {
      EXPECT_EQ(op_desc->GetInputDescPtr(0)->GetDataType(), ge::DT_BF16);
      EXPECT_EQ(op_desc->GetInputDescPtr(1)->GetDataType(), ge::DT_INT32);
      EXPECT_EQ(op_desc->GetInputDescPtr(2)->GetDataType(), ge::DT_BF16);
      EXPECT_EQ(op_desc->GetOutputDescPtr(0)->GetDataType(), ge::DT_BF16);
      EXPECT_EQ(op_desc->GetOutputDescPtr(1)->GetDataType(), ge::DT_BF16);
    }
    if (op_desc->GetType() == "Cast") {
      cast_count++;
    }
  }
  EXPECT_EQ(cast_count, 0);
}

TEST_F(OptimizeOriginalGraphProcess910BTest, optimize_origin_graph_nll_loss_judge_fp32) {
  FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  ComputeGraphPtr graph = CreateGraphWithNLLLossOp(ge::DT_FLOAT);
  SetPrecisionMode("must_keep_origin_dtype");
  Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);

  size_t cast_count = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    std::cout << "==== " << op_desc->GetName() << " - " << op_desc->GetType() << std::endl;
    if (op_desc->GetType() == "NLLLoss") {
      EXPECT_EQ(op_desc->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(op_desc->GetInputDescPtr(1)->GetDataType(), ge::DT_INT32);
      EXPECT_EQ(op_desc->GetInputDescPtr(2)->GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(op_desc->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(op_desc->GetOutputDescPtr(1)->GetDataType(), ge::DT_FLOAT);
    }
    if (op_desc->GetType() == "Cast") {
      cast_count++;
    }
  }
  EXPECT_EQ(cast_count, 0);
}

TEST_F(OptimizeOriginalGraphProcess910BTest, optimize_origin_graph_deformable_roi_pool_judge_fp16) {
  FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  ComputeGraphPtr graph = CreateGraphWithDeformableRoiPoolOp(ge::DT_FLOAT16);
  SetPrecisionMode("must_keep_origin_dtype");
  Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 8);

  size_t cast_count = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    std::cout << "==== " << op_desc->GetName() << " - " << op_desc->GetType() << std::endl;
    if (op_desc->GetType() == "DeformableRoiPool") {
      EXPECT_EQ(op_desc->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
      EXPECT_EQ(op_desc->GetInputDescPtr(1)->GetDataType(), ge::DT_FLOAT16);
      EXPECT_EQ(op_desc->GetInputDescPtr(2)->GetDataType(), ge::DT_FLOAT16);
      EXPECT_EQ(op_desc->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
    }
    if (op_desc->GetType() == "Cast") {
      cast_count++;
    }
  }
  EXPECT_EQ(cast_count, 0);
}

TEST_F(OptimizeOriginalGraphProcess910BTest, optimize_origin_graph_deformable_roi_pool_judge_bf16) {
  FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  ComputeGraphPtr graph = CreateGraphWithDeformableRoiPoolOp(ge::DT_BF16);
  SetPrecisionMode("must_keep_origin_dtype");
  Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 12);

  size_t cast_count = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    std::cout << "==== " << op_desc->GetName() << " - " << op_desc->GetType() << std::endl;
    if (op_desc->GetType() == "DeformableRoiPool") {
      EXPECT_EQ(op_desc->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(op_desc->GetInputDescPtr(1)->GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(op_desc->GetInputDescPtr(2)->GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(op_desc->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
    }
    if (op_desc->GetType() == "Cast") {
      cast_count++;
    }
  }
  EXPECT_EQ(cast_count, 4);
}

TEST_F(OptimizeOriginalGraphProcess910BTest, optimize_origin_graph_deformable_roi_pool_judge_fp32) {
  FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  ComputeGraphPtr graph = CreateGraphWithDeformableRoiPoolOp(ge::DT_FLOAT);
  SetPrecisionMode("must_keep_origin_dtype");
  Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 8);

  size_t cast_count = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    std::cout << "==== " << op_desc->GetName() << " - " << op_desc->GetType() << std::endl;
    if (op_desc->GetType() == "DeformableRoiPool") {
      EXPECT_EQ(op_desc->GetInputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(op_desc->GetInputDescPtr(1)->GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(op_desc->GetInputDescPtr(2)->GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(op_desc->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT);
    }
    if (op_desc->GetType() == "Cast") {
      cast_count++;
    }
  }
  EXPECT_EQ(cast_count, 0);
}

// TEST_F(OptimizeOriginalGraphProcess910BTest, optimize_origin_graph_atomic_write_fixpipe_case1) {
//   FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
//   ComputeGraphPtr graph = CreateGraphWithConv2DBackpropFilterD();
//   FillWeightValue(graph);
//   SetPrecisionMode("must_keep_origin_dtype");
//   Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
//   EXPECT_EQ(ret, SUCCESS);
//   EXPECT_EQ(graph->GetDirectNodesSize(), 7);
//   ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
//   EXPECT_EQ(ret, SUCCESS);
//   EXPECT_EQ(graph->GetDirectNodesSize(), 7);
//   ret = graph_optimizer_ptr->OptimizeOriginalGraph(*graph);
//   EXPECT_EQ(ret, SUCCESS);
//   EXPECT_EQ(graph->GetDirectNodesSize(), 7);
//   ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
//   EXPECT_EQ(ret, SUCCESS);
//   ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
//   EXPECT_EQ(ret, SUCCESS);
//   EXPECT_EQ(graph->GetDirectNodesSize(), 7);
//   size_t cast_count = 0;
//   for (const ge::NodePtr &node : graph->GetDirectNode()) {
//     ge::OpDescPtr op_desc = node->GetOpDesc();
//     std::cout << "==== " << op_desc->GetName() << " - " << op_desc->GetType() << std::endl;
//     if (op_desc->GetType() == "Cast") {
//       cast_count++;
//     }
//   }
//   EXPECT_EQ(cast_count, 2);
// }

TEST_F(OptimizeOriginalGraphProcess910BTest, optimize_origin_graph_case_cube_hif8) {
  FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  auto backup_list = Configuration::Instance(fe::AI_CORE_NAME).fp16_op_type_list_;
  Configuration::Instance(fe::AI_CORE_NAME).fp16_op_type_list_ = {"QuantBatchMatnulV3"};
  ComputeGraphPtr graph = CreateGraphWithQuantMatmulOp();
  SetPrecisionMode("cube_hif8");
  Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 8);

  size_t cast_count = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    std::cout << "==== " << op_desc->GetName() << " - " << op_desc->GetType() << std::endl;
    if (op_desc->GetType() == "QuantBatchMatnulV3") {
      EXPECT_EQ(op_desc->GetInputDescPtr(0)->GetDataType(), ge::DT_HIFLOAT8);
      EXPECT_EQ(op_desc->GetInputDescPtr(1)->GetDataType(), ge::DT_HIFLOAT8);
      EXPECT_EQ(op_desc->GetOutputDescPtr(0)->GetDataType(), ge::DT_HIFLOAT8);
    }
    if (op_desc->GetType() == "Cast") {
      cast_count++;
    }
  }
  EXPECT_EQ(cast_count, 3);
  Configuration::Instance(fe::AI_CORE_NAME).fp16_op_type_list_ = backup_list;
}

TEST_F(OptimizeOriginalGraphProcess910BTest, optimize_origin_graph_case_mixed_hif8) {
  FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  ComputeGraphPtr graph = CreateGraphWithQuantMatmulOp();
  SetPrecisionMode("mixed_hif8");
  Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 8);

  size_t cast_count = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    std::cout << "==== " << op_desc->GetName() << " - " << op_desc->GetType() << std::endl;
    if (op_desc->GetType() == "QuantBatchMatnulV3") {
      EXPECT_EQ(op_desc->GetInputDescPtr(0)->GetDataType(), ge::DT_HIFLOAT8);
      EXPECT_EQ(op_desc->GetInputDescPtr(1)->GetDataType(), ge::DT_HIFLOAT8);
      EXPECT_EQ(op_desc->GetOutputDescPtr(0)->GetDataType(), ge::DT_HIFLOAT8);
    }
    if (op_desc->GetType() == "Cast") {
      cast_count++;
    }
  }
  EXPECT_EQ(cast_count, 3);
}
}

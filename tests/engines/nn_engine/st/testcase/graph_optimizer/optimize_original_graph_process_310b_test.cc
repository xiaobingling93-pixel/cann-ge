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
class OptimizeOriginalGraphProcess310BTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "OptimizeOriginalGraphProcess310BTest TearDown" << endl;
    InitWithSocVersion("Ascend910B1", "must_keep_origin_dtype");
    FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
    map<string, string> options;
    EXPECT_EQ(graph_optimizer_ptr->Initialize(options, nullptr), SUCCESS);
  }

  static void TearDownTestCase() {
    cout << "OptimizeOriginalGraphProcess310BTest TearDown" << endl;
    Finalize();
  }

  ge::ComputeGraphPtr CreateQuantGraphWithConv2DAndAvgPool() {
    ge::GeShape shape_nchw_1({1, 272, 14, 14});
    ge::GeShape shape_nchw_2({272, 1, 3, 3});
    ge::GeShape shape_nchw_3({272});
    ge::GeShape shape_nchw_4({1, 272, 7, 7});
    ge::GeShape shape_nchw_5({1});

    ge::GeTensorDesc tensor_desc_fp32_1(shape_nchw_1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_desc_fp32_1.SetOriginShape(shape_nchw_1);
    tensor_desc_fp32_1.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc_fp32_1.SetOriginFormat(ge::FORMAT_NCHW);

    ge::GeTensorDesc tensor_desc_fp32_2(shape_nchw_4, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_desc_fp32_2.SetOriginShape(shape_nchw_4);
    tensor_desc_fp32_2.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc_fp32_2.SetOriginFormat(ge::FORMAT_NCHW);

    ge::GeTensorDesc tensor_desc_int8_1(shape_nchw_1, ge::FORMAT_NCHW, ge::DT_INT8);
    tensor_desc_int8_1.SetOriginShape(shape_nchw_1);
    tensor_desc_int8_1.SetOriginDataType(ge::DT_INT8);
    tensor_desc_int8_1.SetOriginFormat(ge::FORMAT_NCHW);

    ge::GeTensorDesc tensor_desc_int8_2(shape_nchw_2, ge::FORMAT_NCHW, ge::DT_INT8);
    tensor_desc_int8_2.SetOriginShape(shape_nchw_2);
    tensor_desc_int8_2.SetOriginDataType(ge::DT_INT8);
    tensor_desc_int8_2.SetOriginFormat(ge::FORMAT_NCHW);

    ge::GeTensorDesc tensor_desc_int32_1(shape_nchw_3, ge::FORMAT_NCHW, ge::DT_INT32);
    tensor_desc_int32_1.SetOriginShape(shape_nchw_3);
    tensor_desc_int32_1.SetOriginDataType(ge::DT_INT32);
    tensor_desc_int32_1.SetOriginFormat(ge::FORMAT_NCHW);

    ge::GeTensorDesc tensor_desc_int32_2(shape_nchw_4, ge::FORMAT_NCHW, ge::DT_INT32);
    tensor_desc_int32_2.SetOriginShape(shape_nchw_4);
    tensor_desc_int32_2.SetOriginDataType(ge::DT_INT32);
    tensor_desc_int32_2.SetOriginFormat(ge::FORMAT_NCHW);

    ge::GeTensorDesc tensor_desc_uint64(shape_nchw_5, ge::FORMAT_NCHW, ge::DT_UINT64);
    tensor_desc_uint64.SetOriginShape(shape_nchw_5);
    tensor_desc_uint64.SetOriginDataType(ge::DT_UINT64);
    tensor_desc_uint64.SetOriginFormat(ge::FORMAT_NCHW);

    OpDescPtr data1_op = std::make_shared<OpDesc>("data1", "Data");
    OpDescPtr const1_op = std::make_shared<OpDesc>("const1", "Const");
    OpDescPtr const2_op = std::make_shared<OpDesc>("const2", "Const");
    OpDescPtr const3_op = std::make_shared<OpDesc>("const3", "Const");
    OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
    OpDescPtr quant_op = std::make_shared<OpDesc>("quant", "AscendQuant");
    OpDescPtr dequant_op = std::make_shared<OpDesc>("dequant", "AscendDequant");
    OpDescPtr pool_op = std::make_shared<OpDesc>("pool_update", "AvgPoolUpdate");
    OpDescPtr net_output_op = std::make_shared<OpDesc>("net_output", "NetOutput");

    data1_op->AddOutputDesc(tensor_desc_fp32_1);
    const1_op->AddOutputDesc(tensor_desc_fp32_1);
    const2_op->AddOutputDesc(tensor_desc_fp32_1);
    const3_op->AddOutputDesc(tensor_desc_uint64);
    quant_op->AddInputDesc(tensor_desc_fp32_1);
    quant_op->AddOutputDesc(tensor_desc_int8_1);
    conv_op->AddInputDesc(tensor_desc_int8_1);
    conv_op->AddInputDesc(tensor_desc_int8_2);
    conv_op->AddInputDesc(tensor_desc_int32_1);
    conv_op->AddOutputDesc(tensor_desc_int32_2);
    dequant_op->AddInputDesc(tensor_desc_int32_2);
    dequant_op->AddInputDesc(tensor_desc_uint64);
    dequant_op->AddOutputDesc(tensor_desc_fp32_2);
    pool_op->AddInputDesc(tensor_desc_fp32_2);
    pool_op->AddInputDesc(tensor_desc_int8_1);
    pool_op->AddOutputDesc(tensor_desc_fp32_2);
    net_output_op->AddInputDesc(tensor_desc_fp32_2);

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data1_node = graph->AddNode(data1_op);
    NodePtr const1_node = graph->AddNode(const1_op);
    NodePtr const2_node = graph->AddNode(const2_op);
    NodePtr const3_node = graph->AddNode(const3_op);
    NodePtr quant_node = graph->AddNode(quant_op);
    NodePtr conv_node = graph->AddNode(conv_op);
    NodePtr dequant_node = graph->AddNode(dequant_op);
    NodePtr pool_node = graph->AddNode(pool_op);
    NodePtr net_output_node = graph->AddNode(net_output_op);

    AttrUtils::SetFloat(quant_op, "scale", 9.66);
    AttrUtils::SetFloat(quant_op, "offset", -128);
    AttrUtils::SetListInt(conv_op, "strides", {1, 1, 1, 1});
    AttrUtils::SetListInt(conv_op, "pads", {1, 1, 1, 1});
    AttrUtils::SetListInt(conv_op, "dilations", {1, 1, 1, 1});

    AttrUtils::SetListInt(pool_op, "strides", {1, 1, 1, 1});
    AttrUtils::SetListInt(pool_op, "ksize", {1, 1, 1, 1});

    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), quant_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(quant_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(quant_node->GetOutDataAnchor(0), pool_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(2));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), dequant_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0), dequant_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(dequant_node->GetOutDataAnchor(0), pool_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(pool_node->GetOutDataAnchor(0), net_output_node->GetInDataAnchor(0));

    graph->TopologicalSorting();
    AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "11_22");
    return graph;
  }
  ge::ComputeGraphPtr CreateThreeLayerConvQuantGraph() {
    OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
    OpDescPtr aipp_op = std::make_shared<OpDesc>("aipp", "Aipp");
    OpDescPtr weight1_op = std::make_shared<OpDesc>("weight1", "Const");
    OpDescPtr bias1_op = std::make_shared<OpDesc>("bias1", "Const");
    OpDescPtr conv1_op = std::make_shared<OpDesc>("conv1", "Conv2D");
    OpDescPtr relu1_op = std::make_shared<OpDesc>("relu1", "LeakyRelu");
    OpDescPtr pool_op = std::make_shared<OpDesc>("pool", "Pooling");

    OpDescPtr quant2_op = std::make_shared<OpDesc>("quant2", "AscendQuant");
    OpDescPtr weight2_op = std::make_shared<OpDesc>("weight2", "Const");
    OpDescPtr bias2_op = std::make_shared<OpDesc>("bias2", "Const");
    OpDescPtr conv2_op = std::make_shared<OpDesc>("conv2", "Conv2D");
    OpDescPtr dep_sacle2_op = std::make_shared<OpDesc>("dep_sacle2", "Const");
    OpDescPtr dequant2_op = std::make_shared<OpDesc>("dequant2", "AscendDequant");
    OpDescPtr relu2_op = std::make_shared<OpDesc>("relu2", "Relu");

    OpDescPtr quant3_op = std::make_shared<OpDesc>("quant3", "AscendQuant");
    OpDescPtr weight3_op = std::make_shared<OpDesc>("weight3", "Const");
    OpDescPtr bias3_op = std::make_shared<OpDesc>("bias3", "Const");
    OpDescPtr conv3_op = std::make_shared<OpDesc>("conv3", "Conv2D");
    OpDescPtr dep_sacle3_op = std::make_shared<OpDesc>("dep_sacle3", "Const");
    OpDescPtr dequant3_op = std::make_shared<OpDesc>("dequant3", "AscendDequant");
    OpDescPtr relu3_op = std::make_shared<OpDesc>("relu3", "LeakyRelu");

    ge::GeShape shape({8,224,224,3});
    ge::GeShape shape_64({64});
    ge::GeTensorDesc tensor_nchw_uint8(shape, ge::FORMAT_NCHW, ge::DT_UINT8);
    tensor_nchw_uint8.SetOriginShape(shape);
    tensor_nchw_uint8.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_nchw_uint8.SetOriginDataType(ge::DT_UINT8);

    ge::GeTensorDesc tensor_nchw_fp32(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_nchw_fp32.SetOriginShape(shape);
    tensor_nchw_fp32.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_nchw_fp32.SetOriginDataType(ge::DT_FLOAT);

    ge::GeTensorDesc tensor_nchw_int8(shape, ge::FORMAT_NCHW, ge::DT_INT8);
    tensor_nchw_int8.SetOriginShape(shape);
    tensor_nchw_int8.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_nchw_int8.SetOriginDataType(ge::DT_INT8);

    ge::GeTensorDesc tensor_nchw_int32(shape, ge::FORMAT_NCHW, ge::DT_INT32);
    tensor_nchw_int32.SetOriginShape(shape);
    tensor_nchw_int32.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_nchw_int32.SetOriginDataType(ge::DT_INT32);

    ge::GeTensorDesc tensor_nchw_uint64(shape_64, ge::FORMAT_NCHW, ge::DT_UINT64);
    tensor_nchw_uint64.SetOriginShape(shape_64);
    tensor_nchw_uint64.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_nchw_uint64.SetOriginDataType(ge::DT_UINT64);

    data_op->AddOutputDesc(tensor_nchw_uint8);
    aipp_op->AddInputDesc("image", tensor_nchw_uint8);
    aipp_op->AddOutputDesc("features", tensor_nchw_fp32);
    weight1_op->AddOutputDesc(tensor_nchw_fp32);
    bias1_op->AddOutputDesc(tensor_nchw_fp32);
    conv1_op->AddInputDesc("x", tensor_nchw_fp32);
    conv1_op->AddInputDesc("filter", tensor_nchw_fp32);
    conv1_op->AddInputDesc("bias", tensor_nchw_fp32);
    conv1_op->AddOutputDesc("y", tensor_nchw_fp32);
    relu1_op->AddInputDesc("x", tensor_nchw_fp32);
    relu1_op->AddOutputDesc("y", tensor_nchw_fp32);
    pool_op->AddInputDesc("x", tensor_nchw_fp32);
    pool_op->AddOutputDesc("y", tensor_nchw_fp32);

    quant2_op->AddInputDesc("x", tensor_nchw_fp32);
    quant2_op->AddOutputDesc("y", tensor_nchw_int8);
    weight2_op->AddOutputDesc(tensor_nchw_int8);
    bias2_op->AddOutputDesc(tensor_nchw_int32);
    conv2_op->AddInputDesc("x", tensor_nchw_int8);
    conv2_op->AddInputDesc("filter", tensor_nchw_int8);
    conv2_op->AddInputDesc("bias", tensor_nchw_int32);
    conv2_op->AddOutputDesc("y", tensor_nchw_int32);
    dep_sacle2_op->AddOutputDesc(tensor_nchw_uint64);
    dequant2_op->AddInputDesc("x", tensor_nchw_int32);
    dequant2_op->AddInputDesc("deq_scale", tensor_nchw_uint64);
    dequant2_op->AddOutputDesc("y", tensor_nchw_fp32);
    relu2_op->AddInputDesc("x", tensor_nchw_fp32);
    relu2_op->AddOutputDesc("y", tensor_nchw_fp32);

    quant3_op->AddInputDesc("x", tensor_nchw_fp32);
    quant3_op->AddOutputDesc("y", tensor_nchw_int8);
    weight3_op->AddOutputDesc(tensor_nchw_int8);
    bias3_op->AddOutputDesc(tensor_nchw_int32);
    conv3_op->AddInputDesc("x", tensor_nchw_int8);
    conv3_op->AddInputDesc("filter", tensor_nchw_int8);
    conv3_op->AddInputDesc("bias", tensor_nchw_int32);
    conv3_op->AddOutputDesc("y", tensor_nchw_int32);
    dep_sacle3_op->AddOutputDesc(tensor_nchw_uint64);
    dequant3_op->AddInputDesc("x", tensor_nchw_int32);
    dequant3_op->AddInputDesc("deq_scale", tensor_nchw_uint64);
    dequant3_op->AddOutputDesc("y", tensor_nchw_fp32);
    relu3_op->AddInputDesc("x", tensor_nchw_fp32);
    relu3_op->AddOutputDesc("y", tensor_nchw_fp32);

    AttrUtils::SetStr(aipp_op, kAippConfigPath, ".");
    AttrUtils::SetListInt(pool_op, "window", {1, 1, 2, 2});
    AttrUtils::SetListInt(pool_op, "stride", {1, 1, 2, 2});
    AttrUtils::SetListInt(pool_op, "pad", {3, 3, 3, 3});
    AttrUtils::SetListInt(pool_op, "dilation", {1, 1, 1, 1});
    AttrUtils::SetInt(pool_op, "mode", 1);
    AttrUtils::SetInt(pool_op, "ceil_mode", 1);
    AttrUtils::SetBool(pool_op, "global_pooling", true);
    AttrUtils::SetFloat(quant2_op, "scale", 15.66);
    AttrUtils::SetFloat(quant2_op, "offset", -128);
    AttrUtils::SetFloat(quant3_op, "scale", 32.66);
    AttrUtils::SetFloat(quant3_op, "offset", -128);

    AttrUtils::SetListInt(conv1_op, "strides", {1, 1, 2, 2});
    AttrUtils::SetListInt(conv1_op, "pads", {3, 3, 3, 3});
    AttrUtils::SetListInt(conv1_op, "dilations", {1, 1, 1, 1});
    AttrUtils::SetListInt(conv2_op, "strides", {1, 1, 2, 2});
    AttrUtils::SetListInt(conv2_op, "pads", {3, 3, 3, 3});
    AttrUtils::SetListInt(conv2_op, "dilations", {1, 1, 1, 1});
    AttrUtils::SetListInt(conv3_op, "strides", {1, 1, 2, 2});
    AttrUtils::SetListInt(conv3_op, "pads", {3, 3, 3, 3});
    AttrUtils::SetListInt(conv3_op, "dilations", {1, 1, 1, 1});

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data_node = graph->AddNode(data_op);
    NodePtr aipp_node = graph->AddNode(aipp_op);
    NodePtr weight1_node = graph->AddNode(weight1_op);
    NodePtr bias1_node = graph->AddNode(bias1_op);
    NodePtr conv1_node = graph->AddNode(conv1_op);
    NodePtr relu1_node = graph->AddNode(relu1_op);
    NodePtr pool_node = graph->AddNode(pool_op);

    NodePtr quant2_node = graph->AddNode(quant2_op);
    NodePtr weight2_node = graph->AddNode(weight2_op);
    NodePtr bias2_node = graph->AddNode(bias2_op);
    NodePtr conv2_node = graph->AddNode(conv2_op);
    NodePtr dep_sacle2_node = graph->AddNode(dep_sacle2_op);
    NodePtr dequant2_node = graph->AddNode(dequant2_op);
    NodePtr relu2_node = graph->AddNode(relu2_op);

    NodePtr quant3_node = graph->AddNode(quant3_op);
    NodePtr weight3_node = graph->AddNode(weight3_op);
    NodePtr bias3_node = graph->AddNode(bias3_op);
    NodePtr conv3_node = graph->AddNode(conv3_op);
    NodePtr dep_sacle3_node = graph->AddNode(dep_sacle3_op);
    NodePtr dequant3_node = graph->AddNode(dequant3_op);
    NodePtr relu3_node = graph->AddNode(relu3_op);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), aipp_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(aipp_node->GetOutDataAnchor(0), conv1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(weight1_node->GetOutDataAnchor(0), conv1_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(bias1_node->GetOutDataAnchor(0), conv1_node->GetInDataAnchor(2));
    GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0), relu1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu1_node->GetOutDataAnchor(0), pool_node->GetInDataAnchor(0));

    GraphUtils::AddEdge(pool_node->GetOutDataAnchor(0), quant2_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(quant2_node->GetOutDataAnchor(0), conv2_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(weight2_node->GetOutDataAnchor(0), conv2_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(bias2_node->GetOutDataAnchor(0), conv2_node->GetInDataAnchor(2));
    GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0), dequant2_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(dep_sacle2_node->GetOutDataAnchor(0), dequant2_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(dequant2_node->GetOutDataAnchor(0), relu2_node->GetInDataAnchor(0));

    GraphUtils::AddEdge(relu2_node->GetOutDataAnchor(0), quant3_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(quant3_node->GetOutDataAnchor(0), conv3_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(weight3_node->GetOutDataAnchor(0), conv3_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(bias3_node->GetOutDataAnchor(0), conv3_node->GetInDataAnchor(2));
    GraphUtils::AddEdge(conv3_node->GetOutDataAnchor(0), dequant3_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(dep_sacle3_node->GetOutDataAnchor(0), dequant3_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(dequant3_node->GetOutDataAnchor(0), relu3_node->GetInDataAnchor(0));

    AttrUtils::SetStr(graph, ATTR_NAME_SESSION_GRAPH_ID, "0_1");
    return graph;
  }
};

TEST_F(OptimizeOriginalGraphProcess310BTest, optimize_origin_graph_quant_case1) {
  FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  ComputeGraphPtr graph = CreateQuantGraphWithConv2DAndAvgPool();
  SetPrecisionMode("force_fp32");
  Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 9);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
  EXPECT_EQ(ret, FAILED);
  EXPECT_EQ(graph->GetDirectNodesSize(), 20);
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
    if (op_desc->GetType() == "AvgPoolUpdate") {
      EXPECT_EQ(op_desc->GetInputDescPtr(1)->GetDataType(), ge::DT_FLOAT);
      ge::NodePtr peer_node = node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();
      EXPECT_EQ(peer_node->GetType(), "Cast");
    }
  }
  EXPECT_EQ(trans_count, 5);
  EXPECT_EQ(cast_count, 4);
}
TEST_F(OptimizeOriginalGraphProcess310BTest, optimize_origin_graph_quant_case2) {
  FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  ComputeGraphPtr graph = CreateQuantGraphWithConv2DAndAvgPool();
  SetPrecisionMode("force_fp16");
  Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 9);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
  EXPECT_EQ(ret, FAILED);
  EXPECT_EQ(graph->GetDirectNodesSize(), 19);
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
    if (op_desc->GetType() == "AvgPoolUpdate") {
      EXPECT_EQ(op_desc->GetInputDescPtr(1)->GetDataType(), ge::DT_INT8);
      ge::NodePtr peer_node = node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();
      EXPECT_EQ(peer_node->GetType(), "AscendQuant");
    }
  }
  EXPECT_EQ(trans_count, 4);
  EXPECT_EQ(cast_count, 4);
}

// TEST_F(OptimizeOriginalGraphProcess310BTest, optimize_origin_graph_quant_dump_able_case1) {
//   FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
//   ComputeGraphPtr graph = CreateThreeLayerConvQuantGraph();
//   FillWeightValue(graph);
//   SetPrecisionMode("force_fp16");
//   SetContextOption(ge::QUANT_DUMPABLE, "1");
//   Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
//   EXPECT_EQ(ret, SUCCESS);
//   EXPECT_EQ(graph->GetDirectNodesSize(), 21);
//   ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
//   EXPECT_EQ(ret, SUCCESS);
//   EXPECT_EQ(graph->GetDirectNodesSize(), 21);
//   ret = graph_optimizer_ptr->OptimizeOriginalGraph(*graph);
//   EXPECT_EQ(ret, SUCCESS);
//   EXPECT_EQ(graph->GetDirectNodesSize(), 23);
//   ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
//   EXPECT_EQ(ret, SUCCESS);
//   ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
//   EXPECT_EQ(ret, SUCCESS);
//   // EXPECT_EQ(graph->GetDirectNodesSize(), 37);
//   size_t quant_count = 0;
//   for (const ge::NodePtr &node : graph->GetDirectNode()) {
//     ge::OpDescPtr op_desc = node->GetOpDesc();
//     std::cout << "==== " << op_desc->GetName() << " - " << op_desc->GetType() << std::endl;
//     if (op_desc->GetType() == "AscendQuant") {
//       if (op_desc->GetInputDescPtr(0)->GetFormat() != op_desc->GetInputDescPtr(0)->GetOriginFormat()) {
//         ASSERT_EQ(node->GetInDataNodesSize(), 1);
//         ge::NodePtr pre_node = node->GetInDataNodes().at(0);
//         EXPECT_EQ(pre_node->GetType(), "TransData");
//         bool is_dump_able = false;
//         AttrUtils::GetBool(pre_node->GetOpDesc(), kAttrDumpAble, is_dump_able);
//         EXPECT_EQ(is_dump_able, true);
//         EXPECT_EQ(pre_node->GetOpDesc()->GetInputDescPtr(0)->GetFormat(), op_desc->GetInputDescPtr(0)->GetOriginFormat());
//       }
//       if (op_desc->GetOutputDescPtr(0)->GetFormat() != op_desc->GetOutputDescPtr(0)->GetOriginFormat()) {
//         ASSERT_EQ(node->GetOutDataNodesSize(), 1);
//         ge::NodePtr post_node = node->GetOutDataNodes().at(0);
//         EXPECT_EQ(post_node->GetType(), "TransData");
//         bool is_dump_able = false;
//         AttrUtils::GetBool(post_node->GetOpDesc(), kAttrDumpAble, is_dump_able);
//         EXPECT_EQ(is_dump_able, true);
//         EXPECT_EQ(post_node->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(), op_desc->GetOutputDescPtr(0)->GetOriginFormat());
//       }
//       quant_count++;
//     }
//   }
//   EXPECT_EQ(quant_count, 2);
//   Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::QuantDumpable)] = 0;
// }
}

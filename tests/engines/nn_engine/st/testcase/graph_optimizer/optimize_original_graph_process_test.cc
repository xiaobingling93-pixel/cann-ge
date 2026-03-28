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
class OptimizeOriginalGraphProcessTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "OptimizeOriginalGraphProcessTest TearDown" << endl;
    string stub_cann_path = fe::GetCodeDir() + "/tests/engines/nn_engine/depends/CANN_910b_stub/cann";
    fe::EnvVarGuard cann_guard(MM_ENV_ASCEND_HOME_PATH, stub_cann_path.c_str());
    string stub_opp_path = fe::GetCodeDir() + "/tests/engines/nn_engine/depends/CANN_910b_stub/cann/opp";
    fe::EnvVarGuard opp_guard(MM_ENV_ASCEND_OPP_PATH, stub_opp_path.c_str());
    InitWithSocVersion("Ascend910B1", "allow_fp32_to_fp16");
    FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
    map<string, string> options;
    EXPECT_EQ(graph_optimizer_ptr->Initialize(options, nullptr), SUCCESS);
    cann_guard.Restore();
    opp_guard.Restore();
  }

  static void TearDownTestCase() {
    cout << "OptimizeOriginalGraphProcessTest TearDown" << endl;
    Finalize();
  }

  ge::ComputeGraphPtr CreateGraphForUserSemanticInfer() {
    std::vector<int64_t> dim_nd = {8, 1, 4096};
    std::vector<int64_t> dim_nz = {8, 32, 448, 16, 16};
    std::vector<int64_t> dim_nz_ed = {8, 64, 448, 16, 64};
    std::vector<int64_t> dim_nd_out = {384, 4096};
    ge::GeShape shape_nd(dim_nd);
    ge::GeShape shape_nz(dim_nz);
    ge::GeShape shape_nz_ed(dim_nz_ed);
    ge::GeShape shape_nd_out(dim_nd_out);

    ge::GeTensorDesc tensor_desc_nd(shape_nd, ge::FORMAT_ND, ge::DT_INT32);
    tensor_desc_nd.SetOriginShape(shape_nd);
    tensor_desc_nd.SetOriginDataType(ge::DT_INT32);
    tensor_desc_nd.SetOriginFormat(ge::FORMAT_ND);

    ge::GeTensorDesc tensor_desc_nz(shape_nz, ge::FORMAT_FRACTAL_NZ, ge::DT_INT32);
    tensor_desc_nd.SetOriginShape(shape_nz);
    tensor_desc_nd.SetOriginDataType(ge::DT_INT32);
    tensor_desc_nd.SetOriginFormat(ge::FORMAT_FRACTAL_NZ);

    ge::GeTensorDesc tensor_desc_nz_ed(shape_nz_ed, ge::FORMAT_FRACTAL_NZ, ge::DT_INT4);
    tensor_desc_nz_ed.SetOriginShape(shape_nz_ed);
    tensor_desc_nz_ed.SetOriginDataType(ge::DT_INT4);
    tensor_desc_nz_ed.SetOriginFormat(ge::FORMAT_FRACTAL_NZ);

    ge::GeTensorDesc tensor_desc_nd_out(shape_nd_out, ge::FORMAT_ND, ge::DT_BF16);
    tensor_desc_nd_out.SetOriginShape(shape_nd_out);
    tensor_desc_nd_out.SetOriginDataType(ge::DT_BF16);
    tensor_desc_nd_out.SetOriginFormat(ge::FORMAT_ND);

    OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
    OpDescPtr bitcast_op = std::make_shared<OpDesc>("bitcast", "Bitcast");
    OpDescPtr GMM_op = std::make_shared<OpDesc>("groupedmatmul", "GroupedMatmul");
    OpDescPtr net_output_op = std::make_shared<OpDesc>("net_output", "NetOutput");

    data_op->AddOutputDesc(tensor_desc_nz);
    bitcast_op->AddInputDesc(tensor_desc_nz);
    bitcast_op->AddOutputDesc(tensor_desc_nz_ed);
    GMM_op->AddInputDesc("weight", tensor_desc_nd);
    GMM_op->AddOutputDesc(tensor_desc_nd_out);
    net_output_op->AddInputDesc(tensor_desc_nd_out);

    AttrUtils::SetBool(data_op, "_enable_storage_format_spread", true);

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data_node = graph->AddNode(data_op);
    NodePtr bitcast_node = graph->AddNode(bitcast_op);
    NodePtr GMM_node = graph->AddNode(GMM_op);
    NodePtr net_output_node = graph->AddNode(net_output_op);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), bitcast_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(bitcast_node->GetOutDataAnchor(0), GMM_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(GMM_node->GetOutDataAnchor(0), net_output_node->GetInDataAnchor(0));
    return graph;
  }

  ge::ComputeGraphPtr CreateGraphWithType(const int32_t type = 0) {
    std::vector<int64_t> dim_nchw = {4, 20, 10, 10};
    std::vector<int64_t> dim_5hd = {4, 2, 10, 10, 16};
    if (type == 1) {
      dim_nchw = {10, 20};
    }
    ge::GeShape shape_nchw(dim_nchw);
    ge::GeShape shape_5hd(dim_5hd);
    ge::GeTensorDesc tensor_desc(shape_nchw, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_desc.SetOriginShape(shape_nchw);
    tensor_desc.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);

    ge::GeTensorDesc tensor_desc_5hd(shape_5hd, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
    tensor_desc_5hd.SetOriginShape(shape_nchw);
    tensor_desc_5hd.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc_5hd.SetOriginFormat(ge::FORMAT_NCHW);

    OpDescPtr data_op = std::make_shared<OpDesc>("data1", "Data");
    OpDescPtr ref_data_op = std::make_shared<OpDesc>("data2", "RefData");
    OpDescPtr net_output_op = std::make_shared<OpDesc>("net_output", "NetOutput");

    OpDescPtr const1_op = std::make_shared<OpDesc>("const1", "Const");
    OpDescPtr const2_op = std::make_shared<OpDesc>("const2", "Const");
    OpDescPtr const3_op = std::make_shared<OpDesc>("const3", "Const");

    OpDescPtr in_infer_v2_op = std::make_shared<OpDesc>("in_infer_v2", "INInferV2");
    OpDescPtr sqrt_op = std::make_shared<OpDesc>("sqrt", "Sqrt");
    OpDescPtr sign_op = std::make_shared<OpDesc>("sign", "Sign");
    OpDescPtr abs_op = std::make_shared<OpDesc>("abs", "Abs");

    OpDescPtr square_op = std::make_shared<OpDesc>("square", "Square");
    OpDescPtr sigmoid_op = std::make_shared<OpDesc>("sigmoid", "Sigmoid");
    OpDescPtr swish_op = std::make_shared<OpDesc>("swish", "Swish");

    OpDescPtr neg_op = std::make_shared<OpDesc>("neg", "Neg");
    OpDescPtr gelu_op = std::make_shared<OpDesc>("gelu", "Gelu");
    OpDescPtr tanh_op = std::make_shared<OpDesc>("tanh", "Tanh");

    data_op->AddOutputDesc(tensor_desc_5hd);
    ref_data_op->AddOutputDesc(tensor_desc_5hd);
    net_output_op->AddInputDesc(tensor_desc_5hd);

    const1_op->AddOutputDesc(tensor_desc);
    const2_op->AddOutputDesc(tensor_desc);
    const3_op->AddOutputDesc(tensor_desc);

    in_infer_v2_op->AddInputDesc("x", tensor_desc);
    in_infer_v2_op->AddInputDesc("gamma", tensor_desc);
    in_infer_v2_op->AddInputDesc("beta", tensor_desc);
    in_infer_v2_op->AddInputDesc("mean", tensor_desc);
    in_infer_v2_op->AddInputDesc("variance", tensor_desc);
    in_infer_v2_op->AddOutputDesc("y", tensor_desc);
    sqrt_op->AddInputDesc(tensor_desc);
    sqrt_op->AddOutputDesc(tensor_desc);
    sign_op->AddInputDesc(tensor_desc);
    sign_op->AddOutputDesc(tensor_desc);
    abs_op->AddInputDesc(tensor_desc);
    abs_op->AddOutputDesc(tensor_desc);
    square_op->AddInputDesc(tensor_desc);
    square_op->AddOutputDesc(tensor_desc);
    sigmoid_op->AddInputDesc(tensor_desc);
    sigmoid_op->AddOutputDesc(tensor_desc);
    swish_op->AddInputDesc(tensor_desc);
    swish_op->AddOutputDesc(tensor_desc);
    neg_op->AddInputDesc(tensor_desc);
    neg_op->AddOutputDesc(tensor_desc);
    gelu_op->AddInputDesc(tensor_desc);
    gelu_op->AddOutputDesc(tensor_desc);
    tanh_op->AddInputDesc(tensor_desc);
    tanh_op->AddOutputDesc(tensor_desc);

    AttrUtils::SetFloat(in_infer_v2_op, "epsilon", 1.1);
    AttrUtils::SetBool(data_op, "_is_heavy_op", true);
    AttrUtils::SetBool(ref_data_op, "_is_heavy_op", true);
    AttrUtils::SetBool(net_output_op, "_is_heavy_op", true);

    if (type == 1) {
      ge::GeShape shape_5hd_ch({1,1,20,1,16});
      ge::GeShape shape_5hd_nc({10,2,1,1,16});
      AttrUtils::SetStr(data_op->MutableOutputDesc(0), ge::ATTR_NAME_RESHAPE_INFER_TYPE, "CH");
      data_op->MutableOutputDesc(0)->SetShape(shape_5hd_ch);
      AttrUtils::SetStr(net_output_op->MutableInputDesc(0), ge::ATTR_NAME_RESHAPE_INFER_TYPE, "CH");
      net_output_op->MutableInputDesc(0)->SetShape(shape_5hd_ch);
      AttrUtils::SetStr(ref_data_op->MutableOutputDesc(0), ge::ATTR_NAME_RESHAPE_INFER_TYPE, "NC");
      ref_data_op->MutableOutputDesc(0)->SetShape(shape_5hd_nc);
    }

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data_node = graph->AddNode(data_op);
    NodePtr ref_data_node = graph->AddNode(ref_data_op);
    NodePtr in_infer_v2_node = graph->AddNode(in_infer_v2_op);

    NodePtr const1_node = graph->AddNode(const1_op);
    NodePtr const2_node = graph->AddNode(const2_op);
    NodePtr const3_node = graph->AddNode(const3_op);

    NodePtr sqrt_node = graph->AddNode(sqrt_op);
    NodePtr sign_node = graph->AddNode(sign_op);
    NodePtr abs_node = graph->AddNode(abs_op);

    NodePtr gelu_node = graph->AddNode(gelu_op);
    NodePtr neg_node = graph->AddNode(neg_op);
    NodePtr tanh_node = graph->AddNode(tanh_op);

    NodePtr square_node = graph->AddNode(square_op);
    NodePtr sigmoid_node = graph->AddNode(sigmoid_op);
    NodePtr swish_node = graph->AddNode(swish_op);
    NodePtr net_output_node = graph->AddNode(net_output_op);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), sqrt_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(sqrt_node->GetOutDataAnchor(0), sign_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(sign_node->GetOutDataAnchor(0), abs_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(abs_node->GetOutDataAnchor(0), in_infer_v2_node->GetInDataAnchor(0));

    GraphUtils::AddEdge(ref_data_node->GetOutDataAnchor(0), gelu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(gelu_node->GetOutDataAnchor(0), neg_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(neg_node->GetOutDataAnchor(0), tanh_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(tanh_node->GetOutDataAnchor(0), in_infer_v2_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(in_infer_v2_node->GetOutDataAnchor(0), square_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(square_node->GetOutDataAnchor(0), sigmoid_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(sigmoid_node->GetOutDataAnchor(0), swish_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(swish_node->GetOutDataAnchor(0), net_output_node->GetInDataAnchor(0));

    GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0), in_infer_v2_node->GetInDataAnchor(2));
    GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0), in_infer_v2_node->GetInDataAnchor(3));
    GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0), in_infer_v2_node->GetInDataAnchor(4));

    graph->TopologicalSorting();
    AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "11_22");
    return graph;
  }

  ge::ComputeGraphPtr CreateGraphWithConv(const bool has_flatten = false) {
    std::vector<int64_t> dim_nchw = {4, 20, 10, 10};
    std::vector<int64_t> dim_5hd = {4, 2, 10, 10, 16};
    std::vector<int64_t> dim_fz = {200, 1, 16, 16};
    ge::GeShape shape_nchw(dim_nchw);
    ge::GeShape shape_5hd(dim_5hd);
    ge::GeShape shape_fz(dim_fz);
    ge::GeTensorDesc tensor_desc(shape_nchw, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_desc.SetOriginShape(shape_nchw);
    tensor_desc.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);

    ge::GeTensorDesc tensor_desc_5hd(shape_5hd, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
    tensor_desc_5hd.SetOriginShape(shape_nchw);
    tensor_desc_5hd.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc_5hd.SetOriginFormat(ge::FORMAT_NCHW);

    ge::GeTensorDesc tensor_desc_fz(shape_fz, ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT);
    tensor_desc_fz.SetOriginShape(shape_nchw);
    tensor_desc_fz.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc_fz.SetOriginFormat(ge::FORMAT_NCHW);

    OpDescPtr data_op = std::make_shared<OpDesc>("data1", "Data");
    OpDescPtr ref_data_op = std::make_shared<OpDesc>("data2", "RefData");
    OpDescPtr net_output_op = std::make_shared<OpDesc>("net_output", "NetOutput");

    OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
    OpDescPtr flatten_op = std::make_shared<OpDesc>("flatten", "FlattenV2");
    OpDescPtr sqrt_op = std::make_shared<OpDesc>("sqrt", "Sqrt");
    OpDescPtr sign_op = std::make_shared<OpDesc>("sign", "Sign");
    OpDescPtr abs_op = std::make_shared<OpDesc>("abs", "Abs");

    OpDescPtr square_op = std::make_shared<OpDesc>("square", "Square");
    OpDescPtr sigmoid_op = std::make_shared<OpDesc>("sigmoid", "Sigmoid");
    OpDescPtr swish_op = std::make_shared<OpDesc>("swish", "Swish");

    OpDescPtr neg_op = std::make_shared<OpDesc>("neg", "Neg");
    OpDescPtr gelu_op = std::make_shared<OpDesc>("gelu", "Gelu");
    OpDescPtr tanh_op = std::make_shared<OpDesc>("tanh", "Tanh");

    data_op->AddOutputDesc(tensor_desc_5hd);
    ref_data_op->AddOutputDesc(tensor_desc_fz);
    net_output_op->AddInputDesc(tensor_desc_5hd);

    conv_op->AddInputDesc("x", tensor_desc);
    conv_op->AddInputDesc("filter", tensor_desc);
    conv_op->AddOutputDesc("y", tensor_desc);
    flatten_op->AddInputDesc(tensor_desc);
    flatten_op->AddOutputDesc(tensor_desc);

    sqrt_op->AddInputDesc(tensor_desc);
    sqrt_op->AddOutputDesc(tensor_desc);
    sign_op->AddInputDesc(tensor_desc);
    sign_op->AddOutputDesc(tensor_desc);
    abs_op->AddInputDesc(tensor_desc);
    abs_op->AddOutputDesc(tensor_desc);
    square_op->AddInputDesc(tensor_desc);
    square_op->AddOutputDesc(tensor_desc);
    sigmoid_op->AddInputDesc(tensor_desc);
    sigmoid_op->AddOutputDesc(tensor_desc);
    swish_op->AddInputDesc(tensor_desc);
    swish_op->AddOutputDesc(tensor_desc);
    neg_op->AddInputDesc(tensor_desc);
    neg_op->AddOutputDesc(tensor_desc);
    gelu_op->AddInputDesc(tensor_desc);
    gelu_op->AddOutputDesc(tensor_desc);
    tanh_op->AddInputDesc(tensor_desc);
    tanh_op->AddOutputDesc(tensor_desc);

    AttrUtils::SetListInt(conv_op, "strides", {1, 1, 1, 1});
    AttrUtils::SetListInt(conv_op, "pads", {1, 1, 1, 1});
    AttrUtils::SetListInt(conv_op, "dilations", {1, 1, 1, 1});
    AttrUtils::SetBool(data_op, "_is_heavy_op", true);
    AttrUtils::SetBool(ref_data_op, "_is_heavy_op", true);
    AttrUtils::SetBool(net_output_op, "_is_heavy_op", true);

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data_node = graph->AddNode(data_op);
    NodePtr ref_data_node = graph->AddNode(ref_data_op);
    NodePtr conv_node = graph->AddNode(conv_op);

    NodePtr sqrt_node = graph->AddNode(sqrt_op);
    NodePtr sign_node = graph->AddNode(sign_op);
    NodePtr abs_node = graph->AddNode(abs_op);

    NodePtr gelu_node = graph->AddNode(gelu_op);
    NodePtr neg_node = graph->AddNode(neg_op);
    NodePtr tanh_node = graph->AddNode(tanh_op);

    NodePtr square_node = graph->AddNode(square_op);
    NodePtr sigmoid_node = graph->AddNode(sigmoid_op);
    NodePtr swish_node = graph->AddNode(swish_op);
    NodePtr net_output_node = graph->AddNode(net_output_op);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), sqrt_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(sqrt_node->GetOutDataAnchor(0), sign_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(sign_node->GetOutDataAnchor(0), abs_node->GetInDataAnchor(0));
    if (has_flatten) {
      NodePtr flatten_node = graph->AddNode(flatten_op);
      GraphUtils::AddEdge(abs_node->GetOutDataAnchor(0), flatten_node->GetInDataAnchor(0));
      GraphUtils::AddEdge(flatten_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    } else {
      GraphUtils::AddEdge(abs_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    }

    GraphUtils::AddEdge(ref_data_node->GetOutDataAnchor(0), gelu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(gelu_node->GetOutDataAnchor(0), neg_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(neg_node->GetOutDataAnchor(0), tanh_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(tanh_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), square_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(square_node->GetOutDataAnchor(0), sigmoid_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(sigmoid_node->GetOutDataAnchor(0), swish_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(swish_node->GetOutDataAnchor(0), net_output_node->GetInDataAnchor(0));

    graph->TopologicalSorting();
    AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "11_22");
    return graph;
  }

  ge::ComputeGraphPtr CreateAippGraph() {
    std::vector<int64_t> dim_nhwc = {1, 224, 224, 3};
    ge::GeShape shape_nhwc(dim_nhwc);
    ge::GeTensorDesc tensor_desc_uint8(shape_nhwc, ge::FORMAT_NHWC, ge::DT_UINT8);
    tensor_desc_uint8.SetOriginShape(shape_nhwc);
    tensor_desc_uint8.SetOriginDataType(ge::DT_UINT8);
    tensor_desc_uint8.SetOriginFormat(ge::FORMAT_NHWC);

    ge::GeTensorDesc tensor_desc_fp32(shape_nhwc, ge::FORMAT_NHWC, ge::DT_FLOAT);
    tensor_desc_fp32.SetOriginShape(shape_nhwc);
    tensor_desc_fp32.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc_fp32.SetOriginFormat(ge::FORMAT_NHWC);

    OpDescPtr data1_op = std::make_shared<OpDesc>("data1", "Data");
    OpDescPtr data2_op = std::make_shared<OpDesc>("data2", "Data");
    OpDescPtr aipp_op = std::make_shared<OpDesc>("aipp", "Aipp");
    OpDescPtr cast_op = std::make_shared<OpDesc>("cast", "Cast");
    OpDescPtr sub_op = std::make_shared<OpDesc>("sub", "Sub");
    OpDescPtr net_output_op = std::make_shared<OpDesc>("net_output", "NetOutput");
    data1_op->AddOutputDesc(tensor_desc_uint8);
    data2_op->AddOutputDesc(tensor_desc_fp32);
    aipp_op->AddInputDesc(tensor_desc_uint8);
    aipp_op->AddOutputDesc(tensor_desc_uint8);
    cast_op->AddInputDesc(tensor_desc_uint8);
    cast_op->AddOutputDesc(tensor_desc_fp32);
    sub_op->AddInputDesc(tensor_desc_fp32);
    sub_op->AddInputDesc(tensor_desc_fp32);
    sub_op->AddOutputDesc(tensor_desc_fp32);
    net_output_op->AddInputDesc(tensor_desc_fp32);

    AttrUtils::SetStr(aipp_op, "aipp_config_path", "123");

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data1_node = graph->AddNode(data1_op);
    NodePtr data2_node = graph->AddNode(data2_op);
    NodePtr aipp_node = graph->AddNode(aipp_op);
    NodePtr cast_node = graph->AddNode(cast_op);
    NodePtr sub_node = graph->AddNode(sub_op);
    NodePtr net_output_node = graph->AddNode(net_output_op);

    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), aipp_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(aipp_node->GetOutDataAnchor(0), cast_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), sub_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), sub_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(sub_node->GetOutDataAnchor(0), net_output_node->GetInDataAnchor(0));

    AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "11_22");
    return graph;
  }

  ge::ComputeGraphPtr CreateThreeLayerConvQuantGraph(const bool is_sub_graph = false) {
    string input_op_type = is_sub_graph ? OP_TYPE_PLACE_HOLDER : DATA;
    OpDescPtr data_op = std::make_shared<OpDesc>("data", input_op_type);
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

    if (is_sub_graph) {
      AttrUtils::SetInt(aipp_op, "_fe_imply_type", 6);
      AttrUtils::SetInt(aipp_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
      AttrUtils::SetInt(conv1_op, "_fe_imply_type", 6);
      AttrUtils::SetInt(conv1_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
      AttrUtils::SetInt(relu1_op, "_fe_imply_type", 6);
      AttrUtils::SetInt(relu1_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
      AttrUtils::SetInt(pool_op, "_fe_imply_type", 6);
      AttrUtils::SetInt(pool_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
      AttrUtils::SetInt(quant2_op, "_fe_imply_type", 6);
      AttrUtils::SetInt(quant2_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
      AttrUtils::SetInt(conv2_op, "_fe_imply_type", 6);
      AttrUtils::SetInt(conv2_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
      AttrUtils::SetInt(dequant2_op, "_fe_imply_type", 6);
      AttrUtils::SetInt(dequant2_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
      AttrUtils::SetInt(relu2_op, "_fe_imply_type", 6);
      AttrUtils::SetInt(relu2_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
      AttrUtils::SetInt(quant3_op, "_fe_imply_type", 6);
      AttrUtils::SetInt(quant3_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
      AttrUtils::SetInt(conv3_op, "_fe_imply_type", 6);
      AttrUtils::SetInt(conv3_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
      AttrUtils::SetInt(dequant3_op, "_fe_imply_type", 6);
      AttrUtils::SetInt(dequant3_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
      AttrUtils::SetInt(relu3_op, "_fe_imply_type", 6);
      AttrUtils::SetInt(relu3_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
    }

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
    return graph;
  }
};

TEST_F(OptimizeOriginalGraphProcessTest, optimize_origin_graph_user_semantic) {
  FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  ComputeGraphPtr graph = CreateGraphForUserSemanticInfer();
  Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(OptimizeOriginalGraphProcessTest, optimize_origin_graph_case1) {
  FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  ComputeGraphPtr graph = CreateGraphWithType(0);

  Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 16);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  // EXPECT_EQ(graph->GetDirectNodesSize(), 28);
  size_t trans_count = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    std::cout << "==== " << op_desc->GetName() << " - " << op_desc->GetType() << std::endl;
    if (op_desc->GetType() == "TransData") {
      trans_count++;
    }
  }
  EXPECT_EQ(trans_count, 3);
}

TEST_F(OptimizeOriginalGraphProcessTest, optimize_origin_graph_case2) {
  FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  ComputeGraphPtr graph = CreateGraphWithType(1);

  Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 16);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 30);
  size_t trans_count = 0;
  size_t squze_count = 0;
  size_t unsquze_count = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    std::cout << "==== " << op_desc->GetName() << " - " << op_desc->GetType() << std::endl;
    if (op_desc->GetType() == "TransData") {
      trans_count++;
    } else if (op_desc->GetType() == "SqueezeV2") {
      squze_count++;
    } else if (op_desc->GetType() == "UnsqueezeV2") {
      unsquze_count++;
    }
  }
  EXPECT_EQ(trans_count, 7);
  EXPECT_EQ(squze_count, 2);
  EXPECT_EQ(unsquze_count, 5);
}

TEST_F(OptimizeOriginalGraphProcessTest, optimize_origin_graph_case3) {
  FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  ComputeGraphPtr graph = CreateGraphWithConv();

  Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 13);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
  EXPECT_EQ(ret, FAILED);
  EXPECT_EQ(graph->GetDirectNodesSize(), 15);
  size_t trans_cout = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    std::cout << "==== " << op_desc->GetName() << " - " << op_desc->GetType() << std::endl;
    if (op_desc->GetType() == "TransData") {
      trans_cout++;
    }
  }
  EXPECT_EQ(trans_cout, 0);
}

TEST_F(OptimizeOriginalGraphProcessTest, optimize_origin_graph_case4) {
  FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  ComputeGraphPtr graph = CreateGraphWithConv(true);

  Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 14);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
  EXPECT_EQ(ret, FAILED);
  EXPECT_EQ(graph->GetDirectNodesSize(), 18);
  size_t trans_cout = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    std::cout << "==== " << op_desc->GetName() << " - " << op_desc->GetType() << std::endl;
    if (op_desc->GetType() == "TransData") {
      trans_cout++;
    } else if (op_desc->GetType() != "Cast") {
      for (size_t i = 0; i < op_desc->GetAllInputsSize(); ++i) {
        ge::GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(i);
        if (tensor_desc != nullptr) {
          int32_t primary_format = ge::GetPrimaryFormat(static_cast<int32_t>(tensor_desc->GetFormat()));
          EXPECT_EQ(primary_format == 0 || primary_format == 3 || primary_format == 4, true);
        }
      }
      for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
        ge::GeTensorDescPtr tensor_desc = op_desc->MutableOutputDesc(i);
        if (tensor_desc != nullptr) {
          int32_t primary_format = ge::GetPrimaryFormat(static_cast<int32_t>(tensor_desc->GetFormat()));
          EXPECT_EQ(primary_format == 0 || primary_format == 3 || primary_format == 4, true);
        }
      }
    }
  }
  EXPECT_EQ(trans_cout, 2);
}

TEST_F(OptimizeOriginalGraphProcessTest, optimize_origin_graph_aipp_case1) {
  FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  ComputeGraphPtr graph = CreateAippGraph();

  Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 6);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 7);
  size_t trans_cout = 0;
  size_t cast_cout = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    std::cout << "==== " << op_desc->GetName() << " - " << op_desc->GetType() << std::endl;
    if (op_desc->GetType() == "TransData") {
      trans_cout++;
      ge::Format input_format = static_cast<ge::Format>(ge::GetPrimaryFormat(op_desc->MutableInputDesc(0)->GetFormat()));
      EXPECT_EQ(input_format, ge::FORMAT_NC1HWC0);
      EXPECT_EQ(op_desc->MutableOutputDesc(0)->GetFormat(), ge::FORMAT_NHWC);

    }
    if (op_desc->GetType() == "Cast") {
      cast_cout++;
//      EXPECT_EQ(op_desc->MutableInputDesc(0)->GetFormat(), ge::FORMAT_NHWC);
//      EXPECT_EQ(op_desc->MutableOutputDesc(0)->GetFormat(), ge::FORMAT_NHWC);
    }
  }
  EXPECT_EQ(trans_cout, 1);
  EXPECT_EQ(cast_cout, 1);
}

// TEST_F(OptimizeOriginalGraphProcessTest, optimize_origin_graph_quant_dump_able_case1) {
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
//   // EXPECT_EQ(graph->GetDirectNodesSize(), 24);
//   ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
//   EXPECT_EQ(ret, SUCCESS);
//   ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
//   // EXPECT_EQ(ret, SUCCESS);
//   // EXPECT_EQ(graph->GetDirectNodesSize(), 43);
//   size_t quant_count = 0;
//   for (const ge::NodePtr &node : graph->GetDirectNode()) {
//     ge::OpDescPtr op_desc = node->GetOpDesc();
//     std::cout << "==== " << op_desc->GetName() << " - " << op_desc->GetType() << std::endl;
//     if (op_desc->GetType() == "AscendQuant") {
//       EXPECT_EQ(op_desc->GetInputDescPtr(0)->GetFormat(), op_desc->GetInputDescPtr(0)->GetOriginFormat());
//       EXPECT_EQ(op_desc->GetOutputDescPtr(0)->GetFormat(), op_desc->GetOutputDescPtr(0)->GetOriginFormat());
//       quant_count++;
//     }
//   }
//   EXPECT_EQ(quant_count, 2);
//   Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::QuantDumpable)] = 0;
// }

TEST_F(OptimizeOriginalGraphProcessTest, optimize_sub_graph_quant_dump_able_case1) {
  FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  ComputeGraphPtr graph = CreateThreeLayerConvQuantGraph(true);
  FillWeightValue(graph);
  SetPrecisionMode("force_fp16");
  SetContextOption(ge::QUANT_DUMPABLE, "1");
  Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
  OpSetter::SetQuantDumpableAttr(*graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 21);
  ret = graph_optimizer_ptr->OptimizeFusedGraph(*graph);
  // EXPECT_EQ(ret, SUCCESS);
  // EXPECT_EQ(graph->GetDirectNodesSize(), 22);
  size_t quant_count = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    std::cout << "==== " << op_desc->GetName() << " - " << op_desc->GetType() << std::endl;
    if (op_desc->GetType() == "AscendQuant") {
      EXPECT_EQ(op_desc->GetInputDescPtr(0)->GetFormat(), op_desc->GetInputDescPtr(0)->GetOriginFormat());
      EXPECT_EQ(op_desc->GetOutputDescPtr(0)->GetFormat(), op_desc->GetOutputDescPtr(0)->GetOriginFormat());
      quant_count++;
    }
  }
  EXPECT_EQ(quant_count, 2);
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::QuantDumpable)] = 0;
}
}

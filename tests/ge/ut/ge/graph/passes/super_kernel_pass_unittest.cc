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
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/passes/feature/super_kernel_pass.h"
#include "graph_builder_utils.h"
#include "utils/op_desc_utils.h"

namespace ge {
bool has_sk = false;
namespace {
class SuperKernelPassTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}

  ComputeGraphPtr BuildGraph() {
    auto builder = ut::GraphBuilder("test");
    auto data = builder.AddNode("data", DATA, 0, 1);
    auto transdata1 = builder.AddNode("transdata1", TRANSDATA, 1, 1);
    auto transdata2 = builder.AddNode("transdata2", TRANSDATA, 1, 1);
    auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
    // set transdata1 format&shape
    auto transdata1_input_desc = transdata1->GetOpDesc()->MutableInputDesc(0);
    transdata1_input_desc->SetFormat(FORMAT_FRACTAL_Z);  //src format
    transdata1_input_desc->SetShape(GeShape({1, 1, 16, 16})); //src shape
    transdata1_input_desc->SetOriginFormat(FORMAT_NCHW); // src origin format
    transdata1_input_desc->SetOriginShape(GeShape({16, 1})); // src orgin shape
    auto transdata1_output_desc = transdata1->GetOpDesc()->MutableOutputDesc(0);
    transdata1_output_desc->SetFormat(FORMAT_NCHW);
    transdata1_output_desc->SetShape(GeShape({1, 16, 1, 1}));
    transdata1_output_desc->SetOriginFormat(FORMAT_NCHW);
    transdata1_output_desc->SetOriginShape(GeShape({16, 1}));

    auto transdata2_input_desc = transdata2->GetOpDesc()->MutableInputDesc(0);
    transdata2_input_desc->SetFormat(FORMAT_NCHW);
    transdata2_input_desc->SetShape(GeShape({16, 1, 1, 1}));
    transdata2_input_desc->SetOriginFormat(FORMAT_NCHW);
    transdata2_input_desc->SetOriginShape(GeShape({16, 1, 1, 1}));
    auto transdata2_output_desc = transdata2->GetOpDesc()->MutableOutputDesc(0);
    transdata2_output_desc->SetFormat(FORMAT_FRACTAL_Z); // dst format
    transdata2_output_desc->SetShape(GeShape({1, 1, 16, 16})); // dst shape
    transdata2_output_desc->SetOriginFormat(FORMAT_NCHW); // dst origin format
    transdata2_output_desc->SetOriginShape(GeShape({16, 1, 1, 1})); //dst origin shape, only orgin shape not symmetry

    builder.AddDataEdge(data, 0, transdata1, 0);
    builder.AddDataEdge(transdata1, 0, transdata2, 0);
    builder.AddDataEdge(transdata2, 0, netoutput, 0);

    std::vector<int64_t> data_output_offset{100};
    std::vector<int64_t> transdata1_input_offset{100};
    std::vector<int64_t> transdata1_output_offset{200};
    std::vector<int64_t> transdata2_input_offset{200};
    std::vector<int64_t> transdata2_output_offset{300};
    data->GetOpDesc()->SetOutputOffset(data_output_offset);
    transdata1->GetOpDesc()->SetInputOffset(transdata1_input_offset);
    transdata1->GetOpDesc()->SetOutputOffset(transdata1_output_offset);
    transdata2->GetOpDesc()->SetInputOffset(transdata2_input_offset);
    transdata2->GetOpDesc()->SetOutputOffset(transdata2_output_offset);
    netoutput->GetOpDesc()->SetInputOffset(transdata2_output_offset);

    std::vector<int64_t> v_memory_type{1};
    AttrUtils::SetListInt(data->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
    AttrUtils::SetListInt(transdata1->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type);
    AttrUtils::SetListInt(transdata1->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
    AttrUtils::SetListInt(transdata2->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type);
    AttrUtils::SetListInt(transdata2->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
    AttrUtils::SetListInt(netoutput->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type);

    AttrUtils::SetStr(transdata1->GetOpDesc(), "_super_kernel_scope", "scope_1");
    AttrUtils::SetInt(transdata1->GetOpDesc(), "supportSuperKernel", 1);
    AttrUtils::SetStr(transdata2->GetOpDesc(), "_super_kernel_scope", "scope_1");
    AttrUtils::SetInt(transdata2->GetOpDesc(), "supportSuperKernel", 1);
    return builder.GetGraph();
  }

};
}  // namespace

TEST_F(SuperKernelPassTest, super_kernel_pass_run_success) {
  auto graph = BuildGraph();
  SuperKernelPass super_kernel_pass;
  Status ret = super_kernel_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_NE(sk_node, nullptr);
  std::vector<int64_t> data_output_offset{100};
  std::vector<int64_t> transdata2_output_offset{300};
  std::vector<int64_t> target_memory_type{1};
  EXPECT_EQ(sk_node->GetOpDesc()->GetInputOffset(), data_output_offset);
  EXPECT_EQ(sk_node->GetOpDesc()->GetOutputOffset(), transdata2_output_offset);
  std::vector<int64_t> cur_memory_type;
  EXPECT_TRUE(AttrUtils::GetListInt(sk_node->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, cur_memory_type));
  EXPECT_EQ(cur_memory_type, target_memory_type);
  EXPECT_TRUE(AttrUtils::GetListInt(sk_node->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, cur_memory_type));
  EXPECT_EQ(cur_memory_type, target_memory_type);

  ComputeGraphPtr sub_graph;
  sub_graph = sk_node->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sub_graph);
  EXPECT_NE(sub_graph, nullptr);
  EXPECT_TRUE(sub_graph->GetDirectNodesSize() == 4);

}

TEST_F(SuperKernelPassTest, super_kernel_pass_run_stream_id_not_equal) {
  auto graph = BuildGraph();
  auto trans1_node = graph->FindNode("transdata1");
  EXPECT_NE(trans1_node, nullptr);
  trans1_node->GetOpDesc()->SetStreamId(0);
  trans1_node->GetOpDesc()->DelAttr("supportSuperKernel");
  auto trans2_node = graph->FindNode("transdata2");
  EXPECT_NE(trans2_node, nullptr);
  trans2_node->GetOpDesc()->SetStreamId(1);
  trans2_node->GetOpDesc()->DelAttr("supportSuperKernel");
  SuperKernelPass super_kernel_pass;
  Status ret = super_kernel_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_EQ(sk_node, nullptr);
  ComputeGraphPtr sub_graph;
}

TEST_F(SuperKernelPassTest, super_kernel_verify_abort) {
  DEF_GRAPH(g1) {
    const auto BatchMatMul1 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto Dequantize1 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto cast1= OP_CFG(CAST).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto send1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 1);
    const auto rcv2 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 1);
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("BatchMatMul1", BatchMatMul1)->EDGE(0, 0)->
        EDGE(0, 0)->NODE("Dequantize1", Dequantize1)->EDGE(0, 0)->
        NODE("cast1", cast1)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->EDGE(0, 0)->NODE("Cmo2", CMO)->EDGE(0, 1)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("BatchMatMul1")->CTRL_EDGE()->NODE("Cmo2"));
    CHAIN(NODE("BatchMatMul1")->CTRL_EDGE()->NODE("send1", send1));
    CHAIN(NODE("rcv2", rcv2)->CTRL_EDGE()->NODE("Cmo2"));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto BatchMatMul1 = compute_graph->FindNode("BatchMatMul1");
  auto Dequantize1 = compute_graph->FindNode("Dequantize1");
  auto cast1 = compute_graph->FindNode("cast1");
  auto Cmo2 = compute_graph->FindNode("Cmo2");
  auto send1 = compute_graph->FindNode("send1");
  auto rcv2 = compute_graph->FindNode("rcv2");
  BatchMatMul1->GetOpDesc()->SetStreamId(1);
  Dequantize1->GetOpDesc()->SetStreamId(1);
  cast1->GetOpDesc()->SetStreamId(1);
  send1->GetOpDesc()->SetStreamId(1);

  Cmo2->GetOpDesc()->SetStreamId(2);
  rcv2->GetOpDesc()->SetStreamId(2);
  SuperKernelPass super_kernel_pass;

  AttrUtils::SetStr(BatchMatMul1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=xxx");
  AttrUtils::SetStr(cast1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=xxx");
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_NE(ret, SUCCESS);
  AttrUtils::SetStr(BatchMatMul1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=");
  AttrUtils::SetStr(cast1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=");
  ret = super_kernel_pass.Run(compute_graph);
  EXPECT_NE(ret, SUCCESS);

  AttrUtils::SetStr(BatchMatMul1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=abort");
  AttrUtils::SetStr(Dequantize1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=abort");
  AttrUtils::SetStr(cast1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=abort");
  BatchMatMul1->GetOpDesc()->DelAttr("supportSuperKernel");
  ret = super_kernel_pass.Run(compute_graph);
  EXPECT_NE(ret, SUCCESS);
  AttrUtils::SetInt(BatchMatMul1->GetOpDesc(), "supportSuperKernel", 1);

  Dequantize1->GetOpDesc()->DelAttr("supportSuperKernel");
  ret = super_kernel_pass.Run(compute_graph);
  EXPECT_NE(ret, SUCCESS);
  AttrUtils::SetInt(Dequantize1->GetOpDesc(), "supportSuperKernel", 1);

  cast1->GetOpDesc()->DelAttr("supportSuperKernel");
  ret = super_kernel_pass.Run(compute_graph);
  EXPECT_NE(ret, SUCCESS);
  AttrUtils::SetInt(cast1->GetOpDesc(), "supportSuperKernel", 1);

  AttrUtils::SetStr(Dequantize1->GetOpDesc(), "_super_kernel_scope", "scope_another");
  ret = super_kernel_pass.Run(compute_graph);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(SuperKernelPassTest, super_kernel_not_fusion_data) {
  DEF_GRAPH(g1) {
    const auto BatchMatMul1 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto Dequantize1 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto cast1= OP_CFG(CAST).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("BatchMatMul1", BatchMatMul1)->EDGE(0, 0)->
        EDGE(0, 0)->NODE("Dequantize1", Dequantize1)->EDGE(0, 0)->
        NODE("cast1", cast1)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto data1 = compute_graph->FindNode("data1");
  data1->GetOpDesc()->SetStreamId(-1);
  AttrUtils::SetStr(data1->GetOpDesc(), "_super_kernel_scope", "scope1");
  auto BatchMatMul1 = compute_graph->FindNode("BatchMatMul1");
  auto Dequantize1 = compute_graph->FindNode("Dequantize1");
  auto cast1 = compute_graph->FindNode("cast1");
  BatchMatMul1->GetOpDesc()->SetStreamId(1);
  Dequantize1->GetOpDesc()->SetStreamId(1);
  cast1->GetOpDesc()->SetStreamId(1);

  SuperKernelPass super_kernel_pass;
  AttrUtils::SetStr(data1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=abort");
  AttrUtils::SetStr(BatchMatMul1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=abort");
  AttrUtils::SetStr(Dequantize1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=abort");
  AttrUtils::SetStr(cast1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=abort");
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node = nullptr;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_NE(sk_node, nullptr);
  EXPECT_EQ(sk_node->GetOpDesc()->GetStreamId(), 1);
  EXPECT_FALSE(data1->GetOpDesc()->HasAttr("_super_kernel_scope"));
}

TEST_F(SuperKernelPassTest, super_kernel_verify_bypass) {
  DEF_GRAPH(g1) {
    const auto BatchMatMul1 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto Dequantize1 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto cast1= OP_CFG(CAST).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto cast2= OP_CFG(CAST).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto send1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 1);
    const auto rcv2 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 1);
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("BatchMatMul1", BatchMatMul1)->EDGE(0, 0)->
        EDGE(0, 0)->NODE("Dequantize1", Dequantize1)->EDGE(0, 0)->
        NODE("cast1", cast1)->EDGE(0, 0)->NODE("cast2", cast2)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->EDGE(0, 0)->NODE("Cmo2", CMO)->EDGE(0, 1)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("BatchMatMul1")->CTRL_EDGE()->NODE("Cmo2"));
    CHAIN(NODE("BatchMatMul1")->CTRL_EDGE()->NODE("send1", send1));
    CHAIN(NODE("rcv2", rcv2)->CTRL_EDGE()->NODE("Cmo2"));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto BatchMatMul1 = compute_graph->FindNode("BatchMatMul1");
  auto Dequantize1 = compute_graph->FindNode("Dequantize1");
  auto cast1 = compute_graph->FindNode("cast1");
  auto cast2 = compute_graph->FindNode("cast2");
  auto Cmo2 = compute_graph->FindNode("Cmo2");
  auto send1 = compute_graph->FindNode("send1");
  auto rcv2 = compute_graph->FindNode("rcv2");
  BatchMatMul1->GetOpDesc()->SetStreamId(1);
  Dequantize1->GetOpDesc()->SetStreamId(1);
  cast1->GetOpDesc()->SetStreamId(1);
  cast2->GetOpDesc()->SetStreamId(1);
  send1->GetOpDesc()->SetStreamId(1);

  Cmo2->GetOpDesc()->SetStreamId(2);
  rcv2->GetOpDesc()->SetStreamId(2);
  SuperKernelPass super_kernel_pass;

  AttrUtils::SetStr(BatchMatMul1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=bypass");
  AttrUtils::SetStr(Dequantize1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=bypass");
  AttrUtils::SetStr(cast1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=bypass");
  AttrUtils::SetStr(cast2->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=bypass");
  NodePtr sk_node;
  BatchMatMul1->GetOpDesc()->DelAttr("supportSuperKernel");
  Dequantize1->GetOpDesc()->DelAttr("supportSuperKernel");
  EXPECT_EQ(super_kernel_pass.Run(compute_graph), SUCCESS);
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_EQ(sk_node, nullptr);
  AttrUtils::SetInt(BatchMatMul1->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetStr(BatchMatMul1->GetOpDesc(), "_super_kernel_scope", "scope1");
  AttrUtils::SetStr(BatchMatMul1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=bypass");

  Dequantize1->GetOpDesc()->DelAttr("supportSuperKernel");
  EXPECT_EQ(super_kernel_pass.Run(compute_graph), SUCCESS);
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_EQ(sk_node, nullptr);
  AttrUtils::SetInt(Dequantize1->GetOpDesc(), "supportSuperKernel", 1);

  cast1->GetOpDesc()->DelAttr("supportSuperKernel");
  EXPECT_EQ(super_kernel_pass.Run(compute_graph), SUCCESS);
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_EQ(sk_node, nullptr);
  AttrUtils::SetInt(cast1->GetOpDesc(), "supportSuperKernel", 1);

  AttrUtils::SetStr(Dequantize1->GetOpDesc(), "_super_kernel_scope", "scope_another");
  AttrUtils::SetStr(cast2->GetOpDesc(), "_super_kernel_scope", "scope_another");
  EXPECT_EQ(super_kernel_pass.Run(compute_graph), SUCCESS);
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_EQ(sk_node, nullptr);
}

TEST_F(SuperKernelPassTest, super_kernel_verify_single_stream_not_match) {
  DEF_GRAPH(g1) {
    const auto BatchMatMul1 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto Dequantize1 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto cast1= OP_CFG(CAST).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto send1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 1);
    const auto rcv2 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 1);
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("BatchMatMul1", BatchMatMul1)->EDGE(0, 0)->
        EDGE(0, 0)->NODE("Dequantize1", Dequantize1)->EDGE(0, 0)->
        NODE("cast1", cast1)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->EDGE(0, 0)->NODE("Cmo2", CMO)->EDGE(0, 1)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("BatchMatMul1")->CTRL_EDGE()->NODE("Cmo2"));
    CHAIN(NODE("BatchMatMul1")->CTRL_EDGE()->NODE("send1", send1));
    CHAIN(NODE("rcv2", rcv2)->CTRL_EDGE()->NODE("Cmo2"));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto BatchMatMul1 = compute_graph->FindNode("BatchMatMul1");
  auto Dequantize1 = compute_graph->FindNode("Dequantize1");
  auto cast1 = compute_graph->FindNode("cast1");
  auto Cmo2 = compute_graph->FindNode("Cmo2");
  auto send1 = compute_graph->FindNode("send1");
  auto rcv2 = compute_graph->FindNode("rcv2");
  BatchMatMul1->GetOpDesc()->SetStreamId(1);
  Dequantize1->GetOpDesc()->SetStreamId(1);
  Dequantize1->GetOpDesc()->DelAttr("_super_kernel_scope");
  cast1->GetOpDesc()->SetStreamId(1);
  send1->GetOpDesc()->SetStreamId(1);

  Cmo2->GetOpDesc()->SetStreamId(2);
  rcv2->GetOpDesc()->SetStreamId(2);
  SuperKernelPass super_kernel_pass;
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_NE(sk_node, nullptr);
}

TEST_F(SuperKernelPassTest, super_kernel_select_non_hccl_stream) {
  DEF_GRAPH(g1) {
    const auto hcom_all_gather1 = OP_CFG(HCOMALLGATHER).Attr("_super_kernel_scope", "scope1");
    const auto Dequantize1 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto cast1= OP_CFG(CAST).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto send1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 100);
    const auto rcv2 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 100);
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("hcom_all_gather1", hcom_all_gather1)->EDGE(0, 0)->
        EDGE(0, 0)->NODE("Dequantize1", Dequantize1)->EDGE(0, 0)->
        NODE("cast1", cast1)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("hcom_all_gather1")->CTRL_EDGE()->NODE("Dequantize1"));
    CHAIN(NODE("hcom_all_gather1")->CTRL_EDGE()->NODE("send1", send1));
    CHAIN(NODE("rcv2", rcv2)->CTRL_EDGE()->NODE("Dequantize1"));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto hcom_all_gather = compute_graph->FindNode("hcom_all_gather1");
  auto Dequantize1 = compute_graph->FindNode("Dequantize1");
  auto cast1 = compute_graph->FindNode("cast1");
  auto send1 = compute_graph->FindNode("send1");
  auto rcv2 = compute_graph->FindNode("rcv2");
  hcom_all_gather->GetOpDesc()->SetStreamId(1);
  AttrUtils::SetBool(hcom_all_gather->GetOpDesc(), "_hccl", true);
  send1->GetOpDesc()->SetStreamId(1);

  Dequantize1->GetOpDesc()->SetStreamId(2);
  cast1->GetOpDesc()->SetStreamId(2);
  rcv2->GetOpDesc()->SetStreamId(2);
  SuperKernelPass super_kernel_pass;
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_NE(sk_node, nullptr);
  EXPECT_EQ(sk_node->GetOpDesc()->GetStreamId(), 2);
}

TEST_F(SuperKernelPassTest, super_kernel_cmo_scene) {
  DEF_GRAPH(g1) {
    const auto BatchMatMul1 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto Dequantize1 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto cast1= OP_CFG(CAST).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto send1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 100);
    const auto rcv2 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 100);
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("BatchMatMul1", BatchMatMul1)->EDGE(0, 0)->
        EDGE(0, 0)->NODE("Dequantize1", Dequantize1)->EDGE(0, 0)->
        NODE("cast1", cast1)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->EDGE(0, 0)->NODE("Cmo2", CMO)->EDGE(0, 1)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("BatchMatMul1")->CTRL_EDGE()->NODE("Cmo2"));
    CHAIN(NODE("BatchMatMul1")->CTRL_EDGE()->NODE("send1", send1));
    CHAIN(NODE("rcv2", rcv2)->CTRL_EDGE()->NODE("Cmo2"));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto BatchMatMul1 = compute_graph->FindNode("BatchMatMul1");
  auto Dequantize1 = compute_graph->FindNode("Dequantize1");
  auto cast1 = compute_graph->FindNode("cast1");
  auto Cmo2 = compute_graph->FindNode("Cmo2");
  auto send1 = compute_graph->FindNode("send1");
  auto rcv2 = compute_graph->FindNode("rcv2");
  BatchMatMul1->GetOpDesc()->SetStreamId(1);
  Dequantize1->GetOpDesc()->SetStreamId(1);
  cast1->GetOpDesc()->SetStreamId(1);
  send1->GetOpDesc()->SetStreamId(1);

  Cmo2->GetOpDesc()->SetStreamId(2);
  rcv2->GetOpDesc()->SetStreamId(2);
  SuperKernelPass super_kernel_pass;
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node;
  NodePtr rcv_node;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
    if (node->GetType() == "RecvMem") {
      rcv_node = node;
    }
  }
  EXPECT_NE(sk_node, nullptr);
  EXPECT_NE(rcv_node, nullptr);
  EXPECT_EQ(rcv_node->GetOpDesc()->GetStreamId(), 2);
  uint32_t rcv_event_id = 999;
  EXPECT_TRUE(AttrUtils::GetInt(rcv_node->GetOpDesc(), RECV_ATTR_EVENT_ID, rcv_event_id));
  ComputeGraphPtr sub_graph;
  sub_graph = sk_node->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sub_graph);
  EXPECT_NE(sub_graph, nullptr);
  NodePtr send_node;
  for (auto &node : sub_graph->GetDirectNode()) {
    if (node->GetType() == "SendMem") {
      send_node = node;
    }
  }
  EXPECT_NE(send_node, nullptr);
  EXPECT_EQ(send_node->GetOpDesc()->GetStreamId(), 1);
  uint32_t send_event_id = 99;
  EXPECT_TRUE(AttrUtils::GetInt(rcv_node->GetOpDesc(), SEND_ATTR_EVENT_ID, send_event_id));
  EXPECT_EQ(rcv_event_id, send_event_id);
  EXPECT_EQ(rcv_event_id, INT32_MAX / 2);
}

TEST_F(SuperKernelPassTest, super_kernel_multi_stream_no_fusion) {
  DEF_GRAPH(g1) {
    const auto ffn1_1 = OP_CFG("Ffn");
    const auto hcom_all_gather1 = OP_CFG(HCOMALLGATHER).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto attn1_1 = OP_CFG("Attn");
    const auto hcom_reduce_scatter1 = OP_CFG(HCOMREDUCESCATTER).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto ffn1_2 = OP_CFG("Ffn").Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1").Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine");
    const auto attn1_2 = OP_CFG("Attn");

    const auto attn2_1 = OP_CFG("Attn");
    const auto ffn2_1 = OP_CFG("Ffn").Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1").Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine");
    const auto hcom_all_gather2 = OP_CFG(HCOMALLGATHER).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto attn2_2 = OP_CFG("Attn").Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1").Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine");
    const auto hcom_reduce_scatter2 = OP_CFG(HCOMREDUCESCATTER).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto ffn2_2 = OP_CFG("Ffn");

    const auto send1_1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 100);
    const auto rcv2_1 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 100);

    const auto send1_2 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 101);
    const auto rcv2_2 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 101);

    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("ffn1_1", ffn1_1)->EDGE(0, 0)->
        NODE("hcom_all_gather1", hcom_all_gather1)->EDGE(0, 0)->
        NODE("attn1_1", attn1_1)->EDGE(0, 0)->NODE("hcom_reduce_scatter1", hcom_reduce_scatter1)->EDGE(0, 0)->
        NODE("ffn1_2", ffn1_2)->EDGE(0, 0)->NODE("attn1_2", attn1_2)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("data2", DATA)->EDGE(0, 0)->NODE("attn2_1", attn2_1)->EDGE(0, 0)->NODE("ffn2_1", ffn2_1)->EDGE(0, 0)->
        NODE("hcom_all_gather2", hcom_all_gather2)->EDGE(0, 0)->
        NODE("attn2_2", attn2_2)->EDGE(0, 0)->NODE("hcom_reduce_scatter2", hcom_reduce_scatter2)->EDGE(0, 0)->
        NODE("ffn2_2", ffn2_2)->EDGE(0, 1)->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("attn1_1")->CTRL_EDGE()->NODE("send1_1", send1_1));
    CHAIN(NODE("rcv2_1", rcv2_1)->CTRL_EDGE()->NODE("attn2_2"));
    CHAIN(NODE("ffn1_2")->CTRL_EDGE()->NODE("send1_2", send1_2));
    CHAIN(NODE("rcv2_2", rcv2_2)->CTRL_EDGE()->NODE("ffn2_2"));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();

  auto ffn1_1 = compute_graph->FindNode("ffn1_1");
  auto hcom_all_gather1 = compute_graph->FindNode("hcom_all_gather1");
  auto attn1_1 = compute_graph->FindNode("attn1_1");
  auto hcom_reduce_scatter1 = compute_graph->FindNode("hcom_reduce_scatter1");
  auto ffn1_2 = compute_graph->FindNode("ffn1_2");
  auto attn1_2 = compute_graph->FindNode("attn1_2");
  auto send1_1 = compute_graph->FindNode("send1_1");
  auto send1_2 = compute_graph->FindNode("send1_2");
  ffn1_1->GetOpDesc()->SetStreamId(1);
  hcom_all_gather1->GetOpDesc()->SetStreamId(1);
  AttrUtils::SetBool(hcom_all_gather1->GetOpDesc(), "_hccl", true);
  attn1_1->GetOpDesc()->SetStreamId(1);
  hcom_reduce_scatter1->GetOpDesc()->SetStreamId(1);
  AttrUtils::SetBool(hcom_all_gather1->GetOpDesc(), "_hccl", true);
  ffn1_2->GetOpDesc()->SetStreamId(1);
  attn1_2->GetOpDesc()->SetStreamId(1);
  send1_1->GetOpDesc()->SetStreamId(1);
  send1_2->GetOpDesc()->SetStreamId(1);

  auto attn2_1 = compute_graph->FindNode("attn2_1");
  auto ffn2_1 = compute_graph->FindNode("ffn2_1");
  auto hcom_all_gather2 = compute_graph->FindNode("hcom_all_gather2");
  AttrUtils::SetBool(hcom_all_gather2->GetOpDesc(), "_hccl", true);
  auto attn2_2 = compute_graph->FindNode("attn2_2");
  auto hcom_reduce_scatter2 = compute_graph->FindNode("hcom_reduce_scatter2");
  AttrUtils::SetBool(hcom_reduce_scatter2->GetOpDesc(), "_hccl", true);
  auto ffn2_2 = compute_graph->FindNode("ffn2_2");
  auto rcv2_1 = compute_graph->FindNode("rcv2_1");
  auto rcv2_2 = compute_graph->FindNode("rcv2_2");
  attn2_1->GetOpDesc()->SetStreamId(2);
  ffn2_1->GetOpDesc()->SetStreamId(2);
  hcom_all_gather2->GetOpDesc()->SetStreamId(2);
  attn2_2->GetOpDesc()->SetStreamId(2);
  hcom_reduce_scatter2->GetOpDesc()->SetStreamId(2);
  ffn2_2->GetOpDesc()->SetStreamId(2);
  rcv2_1->GetOpDesc()->SetStreamId(2);
  rcv2_2->GetOpDesc()->SetStreamId(2);

  SuperKernelPass super_kernel_pass;
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  NodePtr sk_node;
  size_t sk_cnt = 0;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
      sk_cnt++;
    }
  }
  EXPECT_NE(sk_node, nullptr);
  EXPECT_EQ(sk_cnt, 2);
}

TEST_F(SuperKernelPassTest, super_kernel_pass_multi_scope) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("trans1", TRANSDATA)->EDGE(0, 0)->NODE("reshape", RESHAPE)
              ->EDGE(0, 0)->NODE("trans2", TRANSDATA)->EDGE(0, 0)->NODE("trans3", TRANSDATA)->EDGE(0, 0)->
        EDGE(0, 0)->NODE("trans4", TRANSDATA)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("const1", CONSTANT)->EDGE(0, 1)->NODE("reshape", RESHAPE));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto trans1_node = compute_graph->FindNode("trans1");
  auto reshape_node = compute_graph->FindNode("reshape");
  auto trans2_node = compute_graph->FindNode("trans2");
  auto trans3_node = compute_graph->FindNode("trans3");
  auto trans4_node = compute_graph->FindNode("trans4");

  AttrUtils::SetStr(trans1_node->GetOpDesc(), "_super_kernel_scope", "scope_1");
  AttrUtils::SetInt(trans1_node->GetOpDesc(), "supportSuperKernel", 1);

  SuperKernelPass super_kernel_pass;
  AttrUtils::SetStr(trans2_node->GetOpDesc(), "_super_kernel_scope", "scope_1");
  AttrUtils::SetInt(trans2_node->GetOpDesc(), "supportSuperKernel", 1);
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_NE(sk_node, nullptr);

  AttrUtils::SetStr(reshape_node->GetOpDesc(), "_super_kernel_scope", "scope_1");
  AttrUtils::SetInt(reshape_node->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetStr(trans3_node->GetOpDesc(), "_super_kernel_scope", "scope_2");
  AttrUtils::SetInt(trans3_node->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetStr(trans4_node->GetOpDesc(), "_super_kernel_scope", "scope_2");
  AttrUtils::SetInt(trans4_node->GetOpDesc(), "supportSuperKernel", 1);
  ret = super_kernel_pass.Run(compute_graph);
  size_t super_cnt = 0;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
      ++super_cnt;
    }
  }
  EXPECT_EQ(super_cnt, 3);
  EXPECT_NE(sk_node, nullptr);
  ComputeGraphPtr sub_graph;
  sub_graph = sk_node->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sub_graph);
  EXPECT_NE(sub_graph, nullptr);
}

TEST_F(SuperKernelPassTest, super_kernel_ringing) {
  DEF_GRAPH(g1) {
    const auto matmul_1 = OP_CFG(MATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto dequant_1 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto batch_matmul_1 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto hcom_reduce_scatter_2 = OP_CFG(HCOMREDUCESCATTER);
    const auto send1_2 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 100);
    const auto rcv1_2 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 100);
    const auto send2_1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 101);
    const auto rcv2_1 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 101);

    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("matmul_1", matmul_1)->EDGE(0, 0)->
        NODE("dequant_1", dequant_1)->EDGE(0, 0)->NODE("hcom_reduce_scatter_2", hcom_reduce_scatter_2)->EDGE(0, 0)->
        NODE("batch_matmul_1", batch_matmul_1)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("dequant_1")->CTRL_EDGE()->NODE("send1_2", send1_2));
    CHAIN(NODE("rcv1_2", rcv1_2)->CTRL_EDGE()->NODE("hcom_reduce_scatter_2"));

    CHAIN(NODE("hcom_reduce_scatter_2")->CTRL_EDGE()->NODE("send2_1", send2_1));
    CHAIN(NODE("rcv2_1", rcv2_1)->CTRL_EDGE()->NODE("batch_matmul_1"));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();

  auto matmul_1 = compute_graph->FindNode("matmul_1");
  auto dequant_1 = compute_graph->FindNode("dequant_1");
  auto batch_matmul_1 = compute_graph->FindNode("batch_matmul_1");
  auto hcom_reduce_scatter_2 = compute_graph->FindNode("hcom_reduce_scatter_2");
  auto send1_2 = compute_graph->FindNode("send1_2");
  auto rcv1_2 = compute_graph->FindNode("rcv1_2");
  auto send2_1 = compute_graph->FindNode("send2_1");
  auto rcv2_1 = compute_graph->FindNode("rcv2_1");

  matmul_1->GetOpDesc()->SetStreamId(1);
  dequant_1->GetOpDesc()->SetStreamId(1);
  batch_matmul_1->GetOpDesc()->SetStreamId(1);
  hcom_reduce_scatter_2->GetOpDesc()->SetStreamId(2);
  AttrUtils::SetBool(hcom_reduce_scatter_2->GetOpDesc(), "_hccl", true);
  send1_2->GetOpDesc()->SetStreamId(1);
  rcv2_1->GetOpDesc()->SetStreamId(1);
  rcv1_2->GetOpDesc()->SetStreamId(2);
  send2_1->GetOpDesc()->SetStreamId(2);

  SuperKernelPass super_kernel_pass;
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node;
  size_t sk_cnt = 0;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
      ++sk_cnt;
    }
  }
  EXPECT_NE(sk_node, nullptr);
  EXPECT_EQ(sk_cnt, 1);
  ComputeGraphPtr sub_graph;
  sub_graph = sk_node->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sub_graph);
  EXPECT_NE(sub_graph, nullptr);
  auto sub_dequant_1 = sub_graph->FindNode("dequant_1");
  EXPECT_NE(sub_dequant_1, nullptr);
  auto sub_batch_matmul_1 = sub_graph->FindNode("batch_matmul_1");
  EXPECT_NE(sub_batch_matmul_1, nullptr);

  auto out_ctl_nodes = sub_dequant_1->GetOutControlNodes();
  EXPECT_EQ(out_ctl_nodes.size(), 1);
  EXPECT_EQ(out_ctl_nodes.at(0)->GetType(), "SendMem");
  uint32_t event_1_send_event_id = 999;
  EXPECT_TRUE(AttrUtils::GetInt(out_ctl_nodes.at(0)->GetOpDesc(), SEND_ATTR_EVENT_ID, event_1_send_event_id));
  auto in_ctl_nodes = hcom_reduce_scatter_2->GetInControlNodes();
  EXPECT_EQ(in_ctl_nodes.size(), 1);
  EXPECT_EQ(in_ctl_nodes.at(0)->GetType(), "RecvMem");
  uint32_t event_1_rcv_event_id = 999;
  EXPECT_TRUE(AttrUtils::GetInt(in_ctl_nodes.at(0)->GetOpDesc(), RECV_ATTR_EVENT_ID, event_1_rcv_event_id));
  EXPECT_EQ(event_1_send_event_id, event_1_rcv_event_id);

  out_ctl_nodes = hcom_reduce_scatter_2->GetOutControlNodes();
  EXPECT_EQ(out_ctl_nodes.size(), 1);
  EXPECT_EQ(out_ctl_nodes.at(0)->GetType(), "SendMem");
  uint32_t event_2_send_event_id = 999;
  EXPECT_TRUE(AttrUtils::GetInt(out_ctl_nodes.at(0)->GetOpDesc(), SEND_ATTR_EVENT_ID, event_2_send_event_id));
  in_ctl_nodes = sub_batch_matmul_1->GetInControlNodes();
  EXPECT_EQ(in_ctl_nodes.size(), 1);
  EXPECT_EQ(in_ctl_nodes.at(0)->GetType(), "RecvMem");
  uint32_t event_2_rcv_event_id = 999;
  EXPECT_TRUE(AttrUtils::GetInt(in_ctl_nodes.at(0)->GetOpDesc(), RECV_ATTR_EVENT_ID, event_2_rcv_event_id));
  EXPECT_EQ(event_2_send_event_id, event_2_rcv_event_id);
}

TEST_F(SuperKernelPassTest, super_kernel_two_graph) {
  auto graph = BuildGraph();
  SuperKernelPass super_kernel_pass;
  Status ret = super_kernel_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_NE(sk_node, nullptr);
  ComputeGraphPtr sub_graph;
  sub_graph = sk_node->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sub_graph);
  EXPECT_NE(sub_graph, nullptr);
  EXPECT_TRUE(sub_graph->GetDirectNodesSize() == 4);

  DEF_GRAPH(g1) {
    const auto BatchMatMul1 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto Dequantize1 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto cast1= OP_CFG(CAST).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto send1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 100);
    const auto rcv2 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 100);
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("BatchMatMul1", BatchMatMul1)->EDGE(0, 0)->
        EDGE(0, 0)->NODE("Dequantize1", Dequantize1)->EDGE(0, 0)->
        NODE("cast1", cast1)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->EDGE(0, 0)->NODE("Cmo2", CMO)->EDGE(0, 1)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("BatchMatMul1")->CTRL_EDGE()->NODE("Cmo2"));
    CHAIN(NODE("BatchMatMul1")->CTRL_EDGE()->NODE("send1", send1));
    CHAIN(NODE("rcv2", rcv2)->CTRL_EDGE()->NODE("Cmo2"));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto BatchMatMul1 = compute_graph->FindNode("BatchMatMul1");
  auto Dequantize1 = compute_graph->FindNode("Dequantize1");
  auto cast1 = compute_graph->FindNode("cast1");
  auto Cmo2 = compute_graph->FindNode("Cmo2");
  auto send1 = compute_graph->FindNode("send1");
  auto rcv2 = compute_graph->FindNode("rcv2");
  BatchMatMul1->GetOpDesc()->SetStreamId(1);
  Dequantize1->GetOpDesc()->SetStreamId(1);
  cast1->GetOpDesc()->SetStreamId(1);
  send1->GetOpDesc()->SetStreamId(1);

  Cmo2->GetOpDesc()->SetStreamId(2);
  rcv2->GetOpDesc()->SetStreamId(2);
  SuperKernelPass super_kernel_pass_other;
  ret = super_kernel_pass_other.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node_other;
  NodePtr rcv_node;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node_other = node;
    }
    if (node->GetType() == "RecvMem") {
      rcv_node = node;
    }
  }
  EXPECT_NE(sk_node_other, nullptr);
  EXPECT_NE(rcv_node, nullptr);
  EXPECT_EQ(rcv_node->GetOpDesc()->GetStreamId(), 2);
  uint32_t rcv_event_id = 999;
  EXPECT_TRUE(AttrUtils::GetInt(rcv_node->GetOpDesc(), RECV_ATTR_EVENT_ID, rcv_event_id));
  ComputeGraphPtr sub_graph_other;
  sub_graph_other = sk_node_other->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sub_graph_other);
  EXPECT_NE(sub_graph_other, nullptr);
  NodePtr send_node;
  for (auto &node : sub_graph_other->GetDirectNode()) {
    if (node->GetType() == "SendMem") {
      send_node = node;
    }
  }
  EXPECT_NE(send_node, nullptr);
  EXPECT_EQ(send_node->GetOpDesc()->GetStreamId(), 1);
  uint32_t send_event_id = 99;
  EXPECT_TRUE(AttrUtils::GetInt(rcv_node->GetOpDesc(), SEND_ATTR_EVENT_ID, send_event_id));
  EXPECT_EQ(rcv_event_id, send_event_id);
  EXPECT_EQ(rcv_event_id, INT32_MAX / 2);
}

TEST_F(SuperKernelPassTest, super_kernel_two_sk_sync_end) {
  DEF_GRAPH(g1) {
    const auto matmul_1 = OP_CFG(MATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto dequant_1 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto batch_matmul_1 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");

    const auto matmul_2 = OP_CFG(MATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope2");
    const auto dequant_2 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope2");
    const auto batch_matmul_2 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope2");

    const auto matmul_3 = OP_CFG(MATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope3");
    const auto dequant_3 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope3");
    const auto batch_matmul_3 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope3");

    const auto send_1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 100);
    const auto rcv_1 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 100);

    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("matmul_1", matmul_1)->EDGE(0, 0)->
        NODE("dequant_1", dequant_1)->EDGE(0, 0)->
        NODE("batch_matmul_1", batch_matmul_1)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("data2", DATA)->EDGE(0, 0)->NODE("matmul_2", matmul_2)->EDGE(0, 0)->
        NODE("dequant_2", dequant_2)->EDGE(0, 0)->
        NODE("batch_matmul_2", batch_matmul_2)->EDGE(0, 1)->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("data3", DATA)->EDGE(0, 0)->NODE("matmul_3", matmul_3)->EDGE(0, 0)->
        NODE("dequant_3", dequant_3)->EDGE(0, 0)->
        NODE("batch_matmul_3", batch_matmul_3)->EDGE(0, 2)->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("batch_matmul_1")->CTRL_EDGE()->NODE("send_1", send_1));
    CHAIN(NODE("rcv_1", rcv_1)->CTRL_EDGE()->NODE("matmul_2"));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();

  auto matmul_1 = compute_graph->FindNode("matmul_1");
  auto dequant_1 = compute_graph->FindNode("dequant_1");
  auto batch_matmul_1 = compute_graph->FindNode("batch_matmul_1");
  auto send_1 = compute_graph->FindNode("send_1");

  matmul_1->GetOpDesc()->SetStreamId(1);
  dequant_1->GetOpDesc()->SetStreamId(1);
  batch_matmul_1->GetOpDesc()->SetStreamId(1);
  send_1->GetOpDesc()->SetStreamId(1);

  auto matmul_2 = compute_graph->FindNode("matmul_2");
  auto dequant_2 = compute_graph->FindNode("dequant_2");
  auto batch_matmul_2 = compute_graph->FindNode("batch_matmul_2");
  auto rcv_1 = compute_graph->FindNode("rcv_1");

  matmul_2->GetOpDesc()->SetStreamId(2);
  dequant_2->GetOpDesc()->SetStreamId(2);
  batch_matmul_2->GetOpDesc()->SetStreamId(2);
  rcv_1->GetOpDesc()->SetStreamId(2);

  SuperKernelPass super_kernel_pass;
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  size_t sk_node_cnt = 0;
  size_t send_rcv_num = 0;
  NodePtr send_mem, rcv_mem, matmul_1_after, matmul_2_after;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      ++sk_node_cnt;
      ComputeGraphPtr sk_sub_graph = nullptr;
      sk_sub_graph = node->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sk_sub_graph);
      for (auto &sub_node : sk_sub_graph->GetDirectNode()) {
        if (sub_node->GetName() == "matmul_1") {
          matmul_1_after = sub_node;
        }
        if (sub_node->GetName() == "matmul_2") {
          matmul_2_after = sub_node;
        }
      }
    }
    if (node->GetType() == SEND) {
      ++send_rcv_num;
      send_mem = node;
    }
    if (node->GetType() == RECV) {
      ++send_rcv_num;
      rcv_mem = node;
    }
  }
  EXPECT_EQ(sk_node_cnt, 3);
  EXPECT_EQ(send_rcv_num, 2);
  uint32_t send_inner_1_event_id = 99;
  EXPECT_TRUE(AttrUtils::GetInt(send_mem->GetOpDesc(), SEND_ATTR_EVENT_ID, send_inner_1_event_id));

  uint32_t rcv_inner_1_event_id = 999;
  EXPECT_TRUE(AttrUtils::GetInt(rcv_mem->GetOpDesc(), RECV_ATTR_EVENT_ID, rcv_inner_1_event_id));
  EXPECT_EQ(rcv_inner_1_event_id, 100);
  EXPECT_EQ(rcv_inner_1_event_id, send_inner_1_event_id);

  EXPECT_NE(matmul_1_after, nullptr);
  EXPECT_NE(matmul_2_after, nullptr);

  std::vector<uint32_t> sk_rcv_event_ids;
  (void)AttrUtils::GetListInt(matmul_2_after->GetOpDesc(), "_sk_rcv_event_ids", sk_rcv_event_ids);
  EXPECT_EQ(sk_rcv_event_ids.size(), 0);
}

TEST_F(SuperKernelPassTest, super_kernel_two_sk_sync_split_logic_stream) {
  DEF_GRAPH(g1) {
    const auto matmul_1 = OP_CFG(MATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto dequant_1 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto batch_matmul_1 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");

    const auto matmul_2 = OP_CFG(MATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto dequant_2 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto batch_matmul_2 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");

    const auto matmul_3 = OP_CFG(MATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope3");
    const auto dequant_3 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope3");
    const auto batch_matmul_3 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope3");

    const auto send_1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 100);
    const auto rcv_1 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 100);

    const auto send_2 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 101);
    const auto rcv_2 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 101);

    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("matmul_1", matmul_1)->EDGE(0, 0)->
        NODE("dequant_1", dequant_1)->EDGE(0, 0)->
        NODE("batch_matmul_1", batch_matmul_1)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("data2", DATA)->EDGE(0, 0)->NODE("matmul_2", matmul_2)->EDGE(0, 0)->
        NODE("dequant_2", dequant_2)->EDGE(0, 0)->
        NODE("batch_matmul_2", batch_matmul_2)->EDGE(0, 1)->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("data3", DATA)->EDGE(0, 0)->NODE("matmul_3", matmul_3)->EDGE(0, 0)->
        NODE("dequant_3", dequant_3)->EDGE(0, 0)->
        NODE("batch_matmul_3", batch_matmul_3)->EDGE(0, 2)->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("batch_matmul_1")->CTRL_EDGE()->NODE("send_1", send_1));
    CHAIN(NODE("rcv_1", rcv_1)->CTRL_EDGE()->NODE("matmul_2"));

    CHAIN(NODE("batch_matmul_2")->CTRL_EDGE()->NODE("send_2", send_2));
    CHAIN(NODE("rcv_2", rcv_2)->CTRL_EDGE()->NODE("matmul_3"));
  };
  auto compute_graph = ToComputeGraph(g1);
  EXPECT_EQ(compute_graph->TopologicalSorting(), GRAPH_SUCCESS);

  auto matmul_1 = compute_graph->FindNode("matmul_1");
  auto dequant_1 = compute_graph->FindNode("dequant_1");
  auto batch_matmul_1 = compute_graph->FindNode("batch_matmul_1");
  auto send_1 = compute_graph->FindNode("send_1");

  matmul_1->GetOpDesc()->SetStreamId(1);
  dequant_1->GetOpDesc()->SetStreamId(1);
  batch_matmul_1->GetOpDesc()->SetStreamId(1);
  send_1->GetOpDesc()->SetStreamId(1);

  auto matmul_2 = compute_graph->FindNode("matmul_2");
  auto dequant_2 = compute_graph->FindNode("dequant_2");
  auto batch_matmul_2 = compute_graph->FindNode("batch_matmul_2");
  auto rcv_1 = compute_graph->FindNode("rcv_1");
  auto send_2 = compute_graph->FindNode("send_2");

  matmul_2->GetOpDesc()->SetStreamId(2);
  dequant_2->GetOpDesc()->SetStreamId(2);
  batch_matmul_2->GetOpDesc()->SetStreamId(2);
  rcv_1->GetOpDesc()->SetStreamId(2);
  send_2->GetOpDesc()->SetStreamId(2);

  auto matmul_3 = compute_graph->FindNode("matmul_3");
  auto dequant_3 = compute_graph->FindNode("dequant_3");
  auto batch_matmul_3 = compute_graph->FindNode("batch_matmul_3");
  auto rcv_2 = compute_graph->FindNode("rcv_2");

  matmul_3->GetOpDesc()->SetStreamId(3);
  dequant_3->GetOpDesc()->SetStreamId(3);
  batch_matmul_3->GetOpDesc()->SetStreamId(3);
  rcv_2->GetOpDesc()->SetStreamId(3);

  SuperKernelPass super_kernel_pass;
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  size_t sk_node_cnt = 0;
  size_t send_rcv_num = 0;
  NodePtr send_node, rcv_node, matmul_1_after, matmul_2_after;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      ++sk_node_cnt;
      ComputeGraphPtr sk_sub_graph = nullptr;
      sk_sub_graph = node->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sk_sub_graph);
      for (auto &sub_node : sk_sub_graph->GetDirectNode()) {
        if (sub_node->GetName() == "matmul_1") {
          matmul_1_after = sub_node;
        }
        if (sub_node->GetName() == "matmul_2") {
          matmul_2_after = sub_node;
        }
      }
    }
    if (node->GetType() == SEND) {
      ++send_rcv_num;
      send_node = node;
    }
    if (node->GetType() == RECV) {
      ++send_rcv_num;
      rcv_node = node;
    }
  }
  EXPECT_EQ(sk_node_cnt, 2);
  EXPECT_EQ(send_rcv_num, 2);
  uint32_t send_inner_1_event_id = 99;
  EXPECT_TRUE(AttrUtils::GetInt(send_node->GetOpDesc(), SEND_ATTR_EVENT_ID, send_inner_1_event_id));

  uint32_t rcv_inner_1_event_id = 999;
  EXPECT_TRUE(AttrUtils::GetInt(rcv_node->GetOpDesc(), RECV_ATTR_EVENT_ID, rcv_inner_1_event_id));
  EXPECT_EQ(rcv_inner_1_event_id, 100);
  EXPECT_EQ(rcv_inner_1_event_id, send_inner_1_event_id);

  EXPECT_EQ(send_node->GetOpDesc()->GetStreamId(), 1);
  EXPECT_EQ(rcv_node->GetOpDesc()->GetStreamId(), 3);

  EXPECT_NE(matmul_1_after, nullptr);
  EXPECT_NE(matmul_2_after, nullptr);

  std::vector<uint32_t> sk_rcv_event_ids;
  (void)AttrUtils::GetListInt(matmul_2_after->GetOpDesc(), "_sk_rcv_event_ids", sk_rcv_event_ids);
  EXPECT_EQ(sk_rcv_event_ids.size(), 1);
}


TEST_F(SuperKernelPassTest, super_kernel_sk_split_test) {
  DEF_GRAPH(g1) {
    const auto matmul_1 = OP_CFG(MATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto dequant_1 = OP_CFG(DEQUANTIZE).Attr("_super_kernel_scope", "scope1");
    const auto batch_matmul_1 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");

    const auto send_1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 100);
    const auto rcv_1 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 100);


    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("matmul_1", matmul_1)->EDGE(0, 0)->
        NODE("dequant_1", dequant_1)->EDGE(0, 0)->
        NODE("batch_matmul_1", batch_matmul_1)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("matmul_1")->CTRL_EDGE()->NODE("send_1", send_1));
    CHAIN(NODE("rcv_1", rcv_1)->CTRL_EDGE()->NODE("dequant_1"));

  };
  auto compute_graph = ToComputeGraph(g1);
  EXPECT_EQ(compute_graph->TopologicalSorting(), GRAPH_SUCCESS);

  auto matmul_1 = compute_graph->FindNode("matmul_1");
  auto dequant_1 = compute_graph->FindNode("dequant_1");
  auto batch_matmul_1 = compute_graph->FindNode("batch_matmul_1");
  auto send_1 = compute_graph->FindNode("send_1");
  auto rcv_1 = compute_graph->FindNode("rcv_1");

  matmul_1->GetOpDesc()->SetStreamId(1);
  send_1->GetOpDesc()->SetStreamId(1);
  rcv_1->GetOpDesc()->SetStreamId(2);
  dequant_1->GetOpDesc()->SetStreamId(2);
  batch_matmul_1->GetOpDesc()->SetStreamId(2);

  dlog_setlevel(1,1,1);
  SuperKernelPass super_kernel_pass;
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  size_t sk_node_cnt = 0;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      ++sk_node_cnt;
    }
  }
  EXPECT_EQ(sk_node_cnt, 2);
}
TEST_F(SuperKernelPassTest, super_kernel_pass_simt) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("trans1", TRANSDATA)->EDGE(0, 0)->NODE("reshape", RESHAPE)
              ->EDGE(0, 0)->NODE("trans2", TRANSDATA)->EDGE(0, 0)->NODE("trans3", TRANSDATA)->EDGE(0, 0)->
        EDGE(0, 0)->NODE("trans4", TRANSDATA)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("const1", CONSTANT)->EDGE(0, 1)->NODE("reshape", RESHAPE));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto trans1_node = compute_graph->FindNode("trans1");
  auto reshape_node = compute_graph->FindNode("reshape");
  auto trans2_node = compute_graph->FindNode("trans2");
  auto trans3_node = compute_graph->FindNode("trans3");
  auto trans4_node = compute_graph->FindNode("trans4");

  AttrUtils::SetStr(trans1_node->GetOpDesc(), "_super_kernel_scope", "scope_1");
  AttrUtils::SetInt(trans1_node->GetOpDesc(), "local_memory_size", 1);
  AttrUtils::SetInt(trans1_node->GetOpDesc(), "supportSuperKernel", 1);

  SuperKernelPass super_kernel_pass;
  AttrUtils::SetInt(trans2_node->GetOpDesc(), "local_memory_size", 1);
  AttrUtils::SetStr(trans2_node->GetOpDesc(), "_super_kernel_scope", "scope_1");
  AttrUtils::SetInt(trans2_node->GetOpDesc(), "supportSuperKernel", 1);
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_EQ(sk_node, nullptr);

  AttrUtils::SetInt(reshape_node->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetInt(reshape_node->GetOpDesc(), "local_memory_size", 1);
  AttrUtils::SetStr(reshape_node->GetOpDesc(), "_super_kernel_scope", "scope_1");
  AttrUtils::SetInt(trans3_node->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetInt(trans3_node->GetOpDesc(), "local_memory_size", 1);
  AttrUtils::SetStr(trans3_node->GetOpDesc(), "_super_kernel_scope", "scope_1");
  ret = super_kernel_pass.Run(compute_graph);
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_EQ(sk_node, nullptr);

  AttrUtils::SetInt(trans3_node->GetOpDesc(), "local_memory_size", 0);
  AttrUtils::SetInt(trans4_node->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetInt(trans4_node->GetOpDesc(), "local_memory_size", 0);
  AttrUtils::SetStr(trans4_node->GetOpDesc(), "_super_kernel_scope", "scope_1");
  ret = super_kernel_pass.Run(compute_graph);
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_NE(sk_node, nullptr);
  ComputeGraphPtr sub_graph;
  sub_graph = sk_node->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sub_graph);
  EXPECT_NE(sub_graph, nullptr);
}
} // namespace ge

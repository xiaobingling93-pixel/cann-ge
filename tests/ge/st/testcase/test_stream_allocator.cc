/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <common/multi_stream_share_graph.h>
#include <gtest/gtest.h>
#include <register/register_custom_pass.h>

#include "ge/ge_api.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/operator_reg.h"
#include "framework/common/types.h"
#include "graph/utils/op_desc_utils.h"
#include "api/gelib/gelib.h"

#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "graph/utils/graph_utils_ex.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "common/share_graph.h"
#include "framework/engine/dnnengine.h"
#include "utils/mock_ops_kernel_builder.h"
#include "framework/common/taskdown_common.h"
#include "base/err_msg.h"
#include "base/err_mgr.h"
#include "common/opskernel/ops_kernel_info_types.h"

extern std::string g_runtime_stub_mock;

using namespace std;
using namespace ge;
namespace {
constexpr const char_t *kDisableIneffectiveMultiStreamOptimize = "DISABLE_INEFFECTIVE_MULTI_STREAM_OPTIMIZE";
/**
 *    Const   Const     Data      Const   Const
 *       \     /       /    \        \     /
 *       GenMask     Relu   Relu     GenMask
 *            \      /         \      /
 *             DoMask           DoMask
 *                   \         /
 *                    NetOutput
 */
Graph BuildGenmaskGraph() {
  int32_t data_value_vec1[1] = {1};
  GeTensorDesc data_tensor_desc(GeShape({1}), FORMAT_ND, DT_INT32);
  GeTensorPtr data_tensor1 = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec1, sizeof(int32_t));
  auto const1 = OP_CFG(CONSTANT).Weight(data_tensor1);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->EDGE(0, 0)->NODE("domask1", DROPOUTDOMASK)->NODE("output", NETOUTPUT));
    CHAIN(NODE("const1", const1)->NODE("DropOutGenMask1", DROPOUTGENMASK)->EDGE(0, 1)->NODE("domask1"));
    CHAIN(NODE("const2", const1)->NODE("DropOutGenMask1"));
    CHAIN(NODE("data1")->NODE("relu2", RELU)->EDGE(0, 0)->NODE("domask2", DROPOUTDOMASK)->NODE("output"));
    CHAIN(NODE("const3", const1)->NODE("DropOutGenMask2", DROPOUTGENMASK)->EDGE(0, 1)->NODE("domask2"));
    CHAIN(NODE("const4", const1)->NODE("DropOutGenMask2"));
  };
  return ToGeGraph(g1);
}

/**
 *    Data   Const
 *       \     /  \
 *       MatMul   Cmo
 *         \      /
 *          \    /
 *         NetOutput
 */
Graph BuildCmoGraph() {
  int32_t data_value_vec1[1] = {1};
  GeTensorDesc data_tensor_desc(GeShape({1}), FORMAT_ND, DT_INT32);
  GeTensorPtr data_tensor1 = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec1, sizeof(int32_t));
  auto const1 = OP_CFG(CONSTANT).Weight(data_tensor1);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("matmul", MATMUL)->NODE("output", NETOUTPUT));
    CHAIN(NODE("const1", const1)->EDGE(0, 1)->NODE("matmul"));
    CHAIN(NODE("const1", const1)->NODE("cmo1", "Cmo")->Ctrl()->NODE("output", NETOUTPUT));
  };

  auto graph = ToGeGraph(g1);
  auto root_graph = GraphUtilsEx::GetComputeGraph(graph);
  GE_ASSERT_TRUE(root_graph != nullptr);
  auto cmo = root_graph->FindNode("cmo1");
  GE_ASSERT_TRUE(cmo != nullptr);
  AttrUtils::SetInt(cmo->GetOpDesc(), "max_size", 10);

  cmo->GetOpDesc()->SetOpKernelLibName("RTSLib");
  cmo->GetOpDesc()->SetOpEngineName("DNN_VM_RTS");
  return graph;
}

Status BuildHCCLGraphStreamPass(const ConstGraphPtr &graph, StreamPassContext &context) {
  AscendString graph_name;
  graph->GetName(graph_name);
  if (graph_name != "BuildHCCLGraphStreamPass") {
    return SUCCESS;
  }
  std::cout << "before current max stream id is " << context.GetCurrMaxStreamId() << std::endl;
  for (auto n: graph->GetDirectNode()) {
    AscendString name;
    n.GetName(name);
    if (name != "hcom1") {
      continue;
    }
    context.SetStreamId(n, 1);
  }
  std::cout << "after current max stream id is " << context.GetCurrMaxStreamId() << std::endl;
  return SUCCESS;
}

Status ReluCustomStreamPass(const ConstGraphPtr &graph, StreamPassContext &context) {
  AscendString graph_name;
  graph->GetName(graph_name);
  if (graph_name != "ReluCustomStreamPass") {
    return SUCCESS;
  }
  std::cout << "before current max stream id is " << context.GetCurrMaxStreamId() << std::endl;
  for (auto n: graph->GetDirectNode()) {
    AscendString name;
    n.GetName(name);
    if (name != "relu1") {
      continue;
    }
    context.SetStreamId(n, context.AllocateNextStreamId());
  }
  std::cout << "after current max stream id is " << context.GetCurrMaxStreamId() << std::endl;
  return SUCCESS;
}

/**
 *     Data1    Data2
 *        \      /
 *     HcomAllReduce1
 *        /      \
 *     Relu1    Relu2
 *        \      /
 *       NetOutput
 */
Graph BuildHCCLGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("hcom1", HCOMALLREDUCE)->NODE("relu1", RELU)->NODE("output", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->NODE("hcom1")->NODE("relu2", RELU)->NODE("output"));
  };
  return ToGeGraph(g1);
}

Graph BuildHCCLGraphWith5HcclNode() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->NODE("hcom1", HCOMALLREDUCE)->NODE("hcom2", HCOMALLREDUCE)
                            ->NODE("hcom3", HCOMALLREDUCE)->NODE("hcom4", HCOMALLREDUCE)
                            ->NODE("hcom5", HCOMALLREDUCE)->NODE("relu1", RELU)->NODE("output", NETOUTPUT));
                  CHAIN(NODE("data2", DATA)->NODE("hcom1")->NODE("hcom2")
                            ->NODE("hcom3")->NODE("hcom4")->NODE("hcom5")->NODE("relu2", RELU)->NODE("output"));
                };
  return ToGeGraph(g1);
}

Graph MakeGraphWithPartitionedCall() {
  DEF_GRAPH(sub) {
                   auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("sub_output1");
                   CHAIN(NODE("sub_relu", RELU)->NODE(netoutput));
                 };

  DEF_GRAPH(root_graph) {
                          auto relu = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("relu");
                          CHAIN(NODE("partitionedcall", PARTITIONEDCALL, sub)->NODE(relu)->NODE("output1", NETOUTPUT));
                        };

  auto graph = ToGeGraph(root_graph);
  return graph;
}

/**
 *    Data1    Data2
 *       \      /
 *        Switch
 *       /      \
 *     700      700
 *    Relus    Relus
 *       \      /
 *        Merge
 *       /     \
 *     Relu   Relu
 *       \     /
 *      NetOutput
 */
Graph BuildSwitchMergeBigGraph() {
  auto pred_node = OP_CFG(DATA).TensorDesc(FORMAT_ND, DT_BOOL, {}).InCnt(1).OutCnt(1);

  DEF_GRAPH(g1) {
    for (size_t i = 1; i < 700; ++i) {
      std::string name_true_src = "true_relu" + std::to_string(i);
      std::string name_false_src = "false_relu" + std::to_string(i);
      std::string name_true_dst = "true_relu" + std::to_string(i + 1);
      std::string name_false_dst = "false_relu" + std::to_string(i + 1);
      auto relu_true = OP_CFG(RELU).Attr(ATTR_NAME_PARALLEL_GROUP, "true");
      auto relu_false = OP_CFG(RELU).Attr(ATTR_NAME_PARALLEL_GROUP, "false");
      CHAIN(NODE(name_true_src, relu_true)->NODE(name_true_dst, relu_true));
      CHAIN(NODE(name_false_src, relu_false)->NODE(name_false_dst, relu_false));
    }
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("switch1", SWITCH)->EDGE(0, 0)->NODE("true_relu1"));
    CHAIN(NODE("data2", pred_node)->EDGE(0, 1)->NODE("switch1")->EDGE(1, 0)->NODE("false_relu1"));
    CHAIN(NODE("true_relu700", RELU)->EDGE(0, 0)->NODE("merge1", MERGE)->EDGE(0, 0)->NODE("relu1", RELU)
          ->NODE("output", NETOUTPUT));
    CHAIN(NODE("false_relu700", RELU)->EDGE(0, 1)->NODE("merge1")->EDGE(1, 0)->NODE("relu2", RELU)->NODE("output"));
  };
  return ToGeGraph(g1);
}

/**
 *      var0          var1
 *       |            |
 *     relu0         relu1
 *       |            |
 *   allgather0   allgather1
 *          \      /
 *            add0     data2
 *              \     /
 *               add1
 *                |
 *             NetOutput
 */
Graph BuildParallelGroupTagGraph() {
  DEF_GRAPH(g1) {
    auto allgather0 = OP_CFG(HCOMALLGATHER).InCnt(1).OutCnt(1).Attr(ATTR_NAME_PARALLEL_GROUP, "-1").Build("allgather0");
    auto allgather1 = OP_CFG(HCOMALLGATHER).InCnt(1).OutCnt(1).Attr(ATTR_NAME_PARALLEL_GROUP, "").Build("allgather1");
    auto add0 = OP_CFG(ADD).InCnt(2).OutCnt(1).Build("add0");
    auto add1 = OP_CFG(ADD).InCnt(2).OutCnt(1).Build("add1");
    auto w0 = OP_CFG(VARIABLE).InCnt(1).OutCnt(1).Build("w0");
    auto w1 = OP_CFG(VARIABLE).InCnt(1).OutCnt(1).Build("w1");
    auto data0 = OP_CFG(DATA).InCnt(1).OutCnt(1).Build("data0");
    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("net_output");

    CHAIN(NODE(w0)->NODE("relu0", RELU)->EDGE(0, 0)->NODE(allgather0));
    CHAIN(NODE(w1)->NODE("relu1", RELU)->EDGE(0, 0)->NODE(allgather1));
    CHAIN(NODE(allgather0)->EDGE(0, 0)->NODE(add0));
    CHAIN(NODE(allgather1)->EDGE(0, 1)->NODE(add0));

    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(add1));
    CHAIN(NODE(add0)->EDGE(0, 1)->NODE(add1));
    CHAIN(NODE(add1)->NODE(net_output));
  };
  auto comput_graph = ToComputeGraph(g1);
  auto geGraph = ToGeGraph(g1);
  return geGraph;
}

/**
 *    sub_data1            Data1
 *        |                  |
 *     700Relu   ==>>  PartitionedCall
 *        |                  |
 *   HcomAllReduce        NetOutput
 *        |
 *   sub_output1
 */
Graph BuildPartitionedCallGraph() {
  DEF_GRAPH(sub) {
    for (size_t i = 1; i < 700; ++i) {
      std::string sub_relu_src = "sub_relu" + std::to_string(i);
      std::string sub_relu_dst = "sub_relu" + std::to_string(i + 1);
      CHAIN(NODE(sub_relu_src, RELU)->NODE(sub_relu_dst, RELU));
    }
    CHAIN(NODE("sub_relu700")->NODE("hcom1", HCOMALLREDUCE)->NODE("sub_output1", NETOUTPUT));
  };

  DEF_GRAPH(root_graph) {
    CHAIN(NODE("data1", DATA)->NODE("partitionedcall", PARTITIONEDCALL, sub)->NODE("output1", NETOUTPUT));
  };
  sub.Layout();

  return ToGeGraph(root_graph);
}

/**
 *        Data0            Data1
 *        |                  |
 *     PartitionedCall0---->PartitionedCall1
 *        \                  /
 *         \                /
 *            NetOutput
 */
Graph BuildStaticFftsGraph() {
  DEF_GRAPH(root) {
    const auto partitioned_0 = OP_CFG(PARTITIONEDCALL).Attr(ATTR_NAME_STREAM_LABEL, "1");
    const auto partitioned_1 = OP_CFG(PARTITIONEDCALL).Attr(ATTR_NAME_STREAM_LABEL, "2");

    CHAIN(NODE("data0", DATA)->NODE("partitionedCall0", partitioned_0)->NODE(NODE_NAME_NET_OUTPUT, NETOUTPUT));
    CHAIN(NODE("data1", DATA)->NODE("partitionedCall1", partitioned_1)->NODE(NODE_NAME_NET_OUTPUT, NETOUTPUT));
    CHAIN(NODE("partitionedCall0", PARTITIONEDCALL)->Ctrl()->NODE("partitionedCall1", PARTITIONEDCALL));
  };
  auto root_graph = ToComputeGraph(root);

  const auto ffts_call_node0 = root_graph->FindNode("partitionedCall0");
  EXPECT_NE(ffts_call_node0, nullptr);
  AttrUtils::SetBool(ffts_call_node0->GetOpDesc(), ATTR_NAME_FFTS_PLUS_SUB_GRAPH, true);
  const auto ffts_call_node1 = root_graph->FindNode("partitionedCall1");
  EXPECT_NE(ffts_call_node1, nullptr);
  AttrUtils::SetBool(ffts_call_node1->GetOpDesc(), ATTR_NAME_FFTS_PLUS_SUB_GRAPH, true);

  DEF_GRAPH(g0) {
    const auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    CHAIN(NODE("sgt0/_arg_0", data_0)
              ->EDGE(0, 0)
              ->NODE("sgt0/trans_TransData_0", TRANSDATA)
              ->EDGE(0, 0)
              ->NODE("sgt0/Node_Output", NETOUTPUT));
  };
  g0.Layout();
  auto sgt0 = ToComputeGraph(g0);
  sgt0->SetGraphUnknownFlag(false);

  int64_t max_size = 1;
  GeTensorDesc tensor_desc(GeShape(), FORMAT_ND, DT_INT64);
  GeTensorPtr const_tensor = MakeShared<GeTensor>(tensor_desc, reinterpret_cast<uint8_t *>(&max_size), sizeof(int64_t));
  const auto const_op = OP_CFG(CONSTANTOP).OutCnt(1).Weight(const_tensor);

  DEF_GRAPH(g1) {
    const auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);

    const auto conv_0 = OP_CFG(CONV2D)
                            .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
                            .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                            .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    const auto relu_0 = OP_CFG(RELU)
                            .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_AIV")
                            .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                            .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");

    CHAIN(NODE("sgt1/_arg_0", data_0)
              ->EDGE(0, 0)
              ->NODE("sgt1/Conv2D", conv_0)
              ->EDGE(0, 0)
              ->NODE("sgt1/Relu", relu_0)
              ->EDGE(0, 0)
              ->NODE("sgt1/Node_Output", NETOUTPUT));
    CHAIN(NODE("sgt1/weight", const_op)->EDGE(0, 1)->NODE("sgt1/Conv2D", conv_0));
  };
  g1.Layout();
  auto sgt1 = ToComputeGraph(g1);
  sgt1->SetGraphUnknownFlag(false);

  return ToGeGraph(root);
}

/**
 *      Data
 *       |
 *     relu1
 *       |    \
 *     relu2     HcomAllReduce
 *       |         |
 *     relu3       |
 *       |       /
 *     netoutput
 */
Graph BuildGraphWithAicoreHcclParallel() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("Data", DATA)->NODE("relu1", RELU)->NODE("relu2", RELU)->NODE("relu3", RELU)->NODE("output", NETOUTPUT));
    CHAIN(NODE("relu1")->NODE("HcomAllReduce", HCOMALLREDUCE)->EDGE(0, 1)->NODE("relu3"));
  };
  return ToGeGraph(g1);
}

/**
 *      Data
 *       |
 *     relu1
 *       |
 *    HcomAllReduce
 *       |
 *     relu2
 *       |
 *     relu3
 *       |
 *   netoutput
 */
Graph BuildGraphWithAicoreHcclSerial() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("Data", DATA)
              ->NODE("relu1", RELU)
              ->NODE("HcomAllReduce", HCOMALLREDUCE)
              ->NODE("relu2", RELU)
              ->NODE("relu3", RELU)
              ->NODE("output", NETOUTPUT));
  };
  return ToGeGraph(g1);
}

/**
 *      Data
 *       |
 *     relu1
 *       |
 *    HcomAllReduce1
 *       |       \
 *     relu2    HcomAllReduce2
 *       |         /
 *       relu3    /
 *       |       /
 *       netoutput
 */
Graph BuildGraphWithAicoreHcclSerialAndMultiHcclSerial() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("Data", DATA)
        ->NODE("relu1", RELU)
        ->NODE("HcomAllReduce1", HCOMALLREDUCE)
        ->CTRL_EDGE()
        ->NODE("relu2", RELU)
        ->NODE("relu3", RELU)
        ->NODE("output", NETOUTPUT));
    CHAIN(NODE("HcomAllReduce1")
        ->NODE("HcomAllReduce2", HCOMALLREDUCE)
        ->NODE("output"));
  };
  return ToGeGraph(g1);
}

/**
 *      data1
 *        |
 *  trans1 (USER_STREAM_LABEL)
 *        |
 *      relu
 *        |
 *  trans2 (USER_STREAM_LABEL)
 *        |
 *   netoutput
 */
Graph BuildGraphWithStreamLabelAndIneffectiveMultiStream() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("trans1", TRANSDATA)->NODE("relu", RELU)->NODE("trans2", TRANSDATA)->NODE("output",
      NETOUTPUT));
  };
  auto graph = ToGeGraph(g1);
  return graph;
}

Graph BuildGraphWithBigSqeNum() {
  DEF_GRAPH(g1) {
    auto relu = OP_CFG(RELU).Attr(ATTR_NAME_NODE_SQE_NUM, 1023);
    CHAIN(NODE("Data", DATA)->NODE("relu1", RELU)->NODE("relu2", RELU)->NODE("relu3", relu));
  };

  return ToGeGraph(g1);
}

ComputeGraphPtr BuildDynamicFftsGraph() {
  DEF_GRAPH(g0) {
    const auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    CHAIN(NODE("sgt0/_arg_0", data_0)->EDGE(0, 0)->NODE("sgt0/trans_TransData_0", TRANSDATA)->EDGE(0, 0)
              ->NODE("sgt0/Node_Output", NETOUTPUT));
  };
  auto sgt0 = ToComputeGraph(g0);
  sgt0->SetGraphUnknownFlag(false);

  DEF_GRAPH(root) {
    auto data_0 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1});
    CHAIN(NODE("data0", data_0)->NODE("partitionedCall0", PARTITIONEDCALL, g0)->NODE(NODE_NAME_NET_OUTPUT, NETOUTPUT));
  };
  auto root_graph = ToComputeGraph(root);

  return root_graph;
}

Graph BuildGraphWithAttachedResources() {
  NamedAttrs named_attr_stream0;
  EXPECT_TRUE(AttrUtils::SetStr(named_attr_stream0, ATTR_NAME_ATTACHED_STREAM_POLICY, "group"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attr_stream0, ATTR_NAME_ATTACHED_STREAM_KEY, "as0"));

  NamedAttrs named_attr_stream1;
  EXPECT_TRUE(AttrUtils::SetStr(named_attr_stream1, ATTR_NAME_ATTACHED_STREAM_POLICY, "group"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attr_stream1, ATTR_NAME_ATTACHED_STREAM_KEY, "as1"));

  uint32_t notify_type = 1; // on_device
  NamedAttrs named_attr_notify0;
  EXPECT_TRUE(AttrUtils::SetStr(named_attr_notify0, ATTR_NAME_ATTACHED_NOTIFY_POLICY, "group"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attr_notify0, ATTR_NAME_ATTACHED_NOTIFY_KEY, "an0"));
  EXPECT_TRUE(AttrUtils::SetInt(named_attr_notify0, ATTR_NAME_ATTACHED_NOTIFY_TYPE, notify_type));

  NamedAttrs named_attr_notify1;
  EXPECT_TRUE(AttrUtils::SetStr(named_attr_notify1, ATTR_NAME_ATTACHED_NOTIFY_POLICY, "group"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attr_notify1, ATTR_NAME_ATTACHED_NOTIFY_KEY, "an1"));
  EXPECT_TRUE(AttrUtils::SetInt(named_attr_notify1, ATTR_NAME_ATTACHED_NOTIFY_TYPE, notify_type));


  DEF_GRAPH(g0) {
    const auto data_0 = OP_CFG(DATA);
    // g0通信域4个节点, 需要两个attach stream， 两个attach notify
    const auto mc2_0 = OP_CFG(MATMUL)
        .Attr("group", "g0")
        .Attr(ATTR_NAME_ATTACHED_STREAM_INFO, named_attr_stream0)
        .Attr(ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attr_notify0);
    const auto mc2_1 = OP_CFG(MATMUL)
        .Attr("group", "g0")
        .Attr(ATTR_NAME_ATTACHED_STREAM_INFO, named_attr_stream0)
        .Attr(ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attr_notify0);
    const auto mc2_2 = OP_CFG(MATMUL)
        .Attr("group", "g0")
        .Attr(ATTR_NAME_ATTACHED_STREAM_INFO, named_attr_stream1)
        .Attr(ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attr_notify1);
    const auto mc2_3 = OP_CFG(MATMUL)
        .Attr("group", "g0")
        .Attr(ATTR_NAME_ATTACHED_STREAM_INFO, named_attr_stream1)
        .Attr(ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attr_notify1);

    // g1通信域2个节点
    const auto mc2_4 = OP_CFG(MATMUL)
        .Attr("group", "g1")
        .Attr(ATTR_NAME_ATTACHED_STREAM_INFO, named_attr_stream0)
        .Attr(ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attr_notify0);
    const auto mc2_5 = OP_CFG(MATMUL)
        .Attr("group", "g1")
        .Attr(ATTR_NAME_ATTACHED_STREAM_INFO, named_attr_stream0)
        .Attr(ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attr_notify0);
    CHAIN(NODE("data0", data_0)
              ->EDGE(0, 0)
              ->NODE("mc2_0", mc2_0)
              ->EDGE(0, 0)
              ->NODE("mc2_1", mc2_1)
              ->EDGE(0, 0)
              ->NODE("mc2_2", mc2_2)
              ->EDGE(0, 0)
              ->NODE("mc2_3", mc2_3)
              ->EDGE(0, 0)
              ->NODE("mc2_4", mc2_4)
              ->EDGE(0, 0)
              ->NODE("mc2_5", mc2_5)
              ->EDGE(0, 0)
              ->NODE("output0", NETOUTPUT));
  };
  return ToGeGraph(g0);
}

/*
 * 适配V2场景，构图：
 * 1. 算子与FE之间通过context交互（contextt对外）
 * 2. GE与FE之间继续通过属性交互
 * */

ge::NamedAttrs CreateResourceInfo(const int64_t &group_id = 0, const int64_t &resource_id = -1,
                                               const int64_t resource_type = -1, const bool required_flag = true, bool force_reuse = false) {
  NamedAttrs named_attr;
  EXPECT_TRUE(AttrUtils::SetStr(named_attr, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group_" + std::to_string(group_id)));
  EXPECT_TRUE(AttrUtils::SetStr(named_attr, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "attached_" + std::to_string(resource_id)));
  EXPECT_TRUE(AttrUtils::SetBool(named_attr, ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true));
  EXPECT_TRUE(AttrUtils::SetBool(named_attr, ATTR_NAME_ATTACHED_RESOURCE_FORCE_REUSE, force_reuse));
  EXPECT_TRUE(AttrUtils::SetInt(named_attr, ATTR_NAME_ATTACHED_RESOURCE_TYPE, resource_type));
  return named_attr;
}

Graph BuildGraphWithMultiAttachedStreamAndHugeNodesAndHugeNodes(size_t node_num,
                                                                const std::vector<int64_t> &attached_streams) {
  std::vector<NamedAttrs> named_attrs;
  for (auto attached_stream_id : attached_streams) {
    named_attrs.emplace_back(CreateResourceInfo(0, attached_stream_id));
  }

  DEF_GRAPH(g1) {
    const auto data_0 = OP_CFG(DATA);
    auto mc2 = OP_CFG(MATMUL).Attr(ATTR_NAME_ATTACHED_STREAM_INFO_LIST, named_attrs);
    auto new_mc2 = OP_CFG(MATMUL).Attr(ATTR_NAME_ATTACHED_STREAM_INFO_LIST, named_attrs);
    CHAIN(NODE("data0", data_0)->EDGE(0, 0)->NODE("mc2_0", mc2));
    for (size_t i = 0U; i < node_num; i++) {
      new_mc2 = OP_CFG(MATMUL).Attr(ATTR_NAME_ATTACHED_STREAM_INFO_LIST, named_attrs).Attr("used_stream_num", 2);
      CHAIN(NODE("mc2_" + std::to_string(i), mc2)->EDGE(0, 0)->NODE("mc2_" + std::to_string(i + 1), new_mc2));
      mc2 = new_mc2;
    }
    CHAIN(NODE("mc2_" + std::to_string(node_num), new_mc2)->EDGE(0, 0)->NODE("output0", NETOUTPUT));
  };
  return ToGeGraph(g1);
}

Graph BuildGraphWithAttachedResourcesWithOpenSource() {
  std::vector<NamedAttrs> named_attrs_gp_0_stream0 = {CreateResourceInfo(0, 0)};  // gp0 stream0
  std::vector<NamedAttrs> named_attrs_gp_0_stream1 = {CreateResourceInfo(0, 1)};  // gp0 stream1
  std::vector<NamedAttrs> named_attrs_gp_1_stream0 = {CreateResourceInfo(1, 0)};  // gp1 stream0
  std::vector<NamedAttrs> named_attrs_gp_1_stream1 = {CreateResourceInfo(1, 1)};  // gp1 stream1
  std::vector<NamedAttrs> named_attrs_gp_2_stream1 = {CreateResourceInfo(2, 1)};  // gp2 stream1

  std::vector<NamedAttrs> named_attrs_gp_0_event0 = {CreateResourceInfo(0, 0, 0)};  // gp0 event0
  std::vector<NamedAttrs> named_attrs_gp_0_event1 = {CreateResourceInfo(0, 1, 0)};  // gp0 event1
  std::vector<NamedAttrs> named_attrs_gp_1_event0 = {CreateResourceInfo(1, 0, 0)};  // gp1 event0
  std::vector<NamedAttrs> named_attrs_gp_1_event1 = {CreateResourceInfo(1, 1, 0)};  // gp1 event1
  std::vector<NamedAttrs> named_attrs_gp_2_event1 = {CreateResourceInfo(2, 1, 0)};  // gp2 event1

  std::vector<NamedAttrs> named_attrs_gp_0_notify0 = {CreateResourceInfo(0, 0, 1)};  // gp0 notify0
  std::vector<NamedAttrs> named_attrs_gp_0_notify1 = {CreateResourceInfo(0, 1, 1)};  // gp0 notify1
  std::vector<NamedAttrs> named_attrs_gp_1_notify0 = {CreateResourceInfo(1, 0, 1)};  // gp1 notify0
  std::vector<NamedAttrs> named_attrs_gp_1_notify1 = {CreateResourceInfo(1, 1, 1)};  // gp1 notify1

  DEF_GRAPH(g1) {
    const auto data_0 = OP_CFG(DATA);
    // g0通信域4个节点, 需要两个attach stream， 两个attach notify
    const auto mc2_0 = OP_CFG(MATMUL)
                           .Attr(ATTR_NAME_ATTACHED_STREAM_INFO_LIST, named_attrs_gp_0_stream0)
                           .Attr(ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, named_attrs_gp_0_notify0);
    const auto mc2_1 = OP_CFG(MATMUL)
                           .Attr(ATTR_NAME_ATTACHED_STREAM_INFO_LIST, named_attrs_gp_0_stream0)
                           .Attr(ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, named_attrs_gp_0_notify0);
    const auto mc2_2 = OP_CFG(MATMUL)
                           .Attr(ATTR_NAME_ATTACHED_STREAM_INFO_LIST, named_attrs_gp_0_stream1)
                           .Attr(ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, named_attrs_gp_0_notify1);
    const auto mc2_3 = OP_CFG(MATMUL)
                           .Attr(ATTR_NAME_ATTACHED_STREAM_INFO_LIST, named_attrs_gp_0_stream1)
                           .Attr(ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, named_attrs_gp_0_notify1);

    // g1通信域2个节点
    const auto mc2_4 = OP_CFG(MATMUL)
                           .Attr(ATTR_NAME_ATTACHED_STREAM_INFO_LIST, named_attrs_gp_1_stream0)
                           .Attr(ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, named_attrs_gp_1_notify0);
    const auto mc2_5 = OP_CFG(MATMUL)
                           .Attr("group", "g1")
                           .Attr(ATTR_NAME_ATTACHED_STREAM_INFO_LIST, named_attrs_gp_1_stream0)
                           .Attr(ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, named_attrs_gp_1_notify0);

    // g2通信域2个节点
    const auto mc2_6 = OP_CFG(MATMUL)
                           .Attr(ATTR_NAME_ATTACHED_STREAM_INFO_LIST, named_attrs_gp_2_stream1)
                           .Attr(ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, named_attrs_gp_2_event1);
    const auto mc2_7 = OP_CFG(MATMUL)
                           .Attr(ATTR_NAME_ATTACHED_STREAM_INFO_LIST, named_attrs_gp_2_stream1)
                           .Attr(ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, named_attrs_gp_2_event1);
    CHAIN(NODE("data0", data_0)
              ->EDGE(0, 0)
              ->NODE("mc2_0", mc2_0)
              ->EDGE(0, 0)
              ->NODE("mc2_1", mc2_1)
              ->EDGE(0, 0)
              ->NODE("mc2_2", mc2_2)
              ->EDGE(0, 0)
              ->NODE("mc2_3", mc2_3)
              ->EDGE(0, 0)
              ->NODE("mc2_4", mc2_4)
              ->EDGE(0, 0)
              ->NODE("mc2_5", mc2_5)
              ->EDGE(0, 0)
              ->NODE("mc2_6", mc2_6)
              ->EDGE(0, 0)
              ->NODE("mc2_7", mc2_7)
              ->EDGE(0, 0)
              ->NODE("output0", NETOUTPUT));
  };
  return ToGeGraph(g1);
}

Graph BuildGraphWithDifferentMainStreamWithSameAttachedStream() {
  std::vector<NamedAttrs> named_attrs_gp_0_stream0 = {CreateResourceInfo(0, 0, -1, true, true)};  // gp0 stream0
  std::vector<NamedAttrs> named_attrs_gp_0_notify0 = {CreateResourceInfo(0, 0, 1)};  // gp0 notify0

  DEF_GRAPH(g1) {
    const auto data_0 = OP_CFG(DATA);
    const auto mc2_0 = OP_CFG(MATMUL)
                           .Attr(ATTR_NAME_ATTACHED_STREAM_INFO_LIST, named_attrs_gp_0_stream0)
                           .Attr(ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, named_attrs_gp_0_notify0)
                           .Attr(public_attr::USER_STREAM_LABEL, "111");
    const auto mc2_1 = OP_CFG(MATMUL)
                           .Attr(ATTR_NAME_ATTACHED_STREAM_INFO_LIST, named_attrs_gp_0_stream0)
                           .Attr(ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, named_attrs_gp_0_notify0)
                           .Attr(public_attr::USER_STREAM_LABEL, "222");
    CHAIN(NODE("data0", data_0)
              ->EDGE(0, 0)
              ->NODE("mc2_0", mc2_0)
              ->EDGE(0, 0)
              ->NODE("mc2_1", mc2_1)
              ->EDGE(0, 0)
              ->NODE("output0", NETOUTPUT));
  };
  return ToGeGraph(g1);
}

Graph BuildGraphHcclSplitAttachedStream() {
  std::vector<NamedAttrs> named_attrs_gp_0_stream0 = {CreateResourceInfo(0, 0, -1, true, true)};  // gp0 stream0
  std::vector<NamedAttrs> named_attrs_gp_0_notify0 = {CreateResourceInfo(0, 0, 1)};  // gp0 notify0

  DEF_GRAPH(g1) {
    int64_t hccl_task_num = 888L;
    const auto data_0 = OP_CFG(DATA);
    const auto mc2_0 = OP_CFG(HCOMALLREDUCE)
                           .Attr(ATTR_NAME_ATTACHED_STREAM_INFO_LIST, named_attrs_gp_0_stream0)
                           .Attr(ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, named_attrs_gp_0_notify0)
                           .Attr(ATTR_NAME_HCCL_TASK_NUM, hccl_task_num)
                           .Attr(ATTR_NAME_HCCL_ATTACHED_TASK_NUM, hccl_task_num);
    const auto mc2_1 = OP_CFG(HCOMALLREDUCE)
                           .Attr(ATTR_NAME_ATTACHED_STREAM_INFO_LIST, named_attrs_gp_0_stream0)
                           .Attr(ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, named_attrs_gp_0_notify0)
                           .Attr(ATTR_NAME_HCCL_TASK_NUM, hccl_task_num)
                           .Attr(ATTR_NAME_HCCL_ATTACHED_TASK_NUM, hccl_task_num);
    CHAIN(NODE("data0", data_0)
              ->EDGE(0, 0)
              ->NODE("mc2_0", mc2_0)
              ->EDGE(0, 0)
              ->NODE("mc2_1", mc2_1)
              ->EDGE(0, 0)
              ->NODE("output0", NETOUTPUT));
  };
  return ToGeGraph(g1);
}

REG_OP(Data)
    .INPUT(data, TensorType::ALL())
    .OUTPUT(out, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)

REG_OP(Variable)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Variable)

REG_OP(MatMulV2)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(MatMulV2)

Graph BuildGraph2WithAttachedResources() {
  auto data0 = op::Data("data0");
  auto variable = op::Variable("variable");
  auto matmu0 = op::MatMulV2("matmu0").set_input_x1(data0).set_input_x2(data0);
  auto matmu1 = op::MatMulV2("matmu1").set_input_x1(matmu0).set_input_x2(variable);

  Graph graph("test_graph");
  std::vector<Operator> inputs {data0};
  std::vector<Operator> outputs {matmu1};
  graph.SetInputs(inputs).SetOutputs(outputs);

  std::vector<NamedAttrs> attrs_stream0;
  NamedAttrs named_attr_stream0;
  EXPECT_TRUE(AttrUtils::SetStr(named_attr_stream0, "_attached_stream_key", "feature0"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attr_stream0, "_attached_stream_depend_value_list", "0"));
  attrs_stream0.emplace_back(named_attr_stream0);

  std::vector<NamedAttrs> attrs_stream1;
  NamedAttrs named_attr_stream1;
  EXPECT_TRUE(AttrUtils::SetStr(named_attr_stream1, "_attached_stream_key", "feature1"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attr_stream1, "_attached_stream_depend_value_list", "0,1"));
  attrs_stream1.emplace_back(named_attr_stream1);

  std::vector<NamedAttrs> attrs_event0;
  NamedAttrs named_attr_event0;
  EXPECT_TRUE(AttrUtils::SetStr(named_attr_event0, "_attached_sync_res_type", "event"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attr_event0, "_attached_sync_res_key", "res_key0"));
  attrs_event0.emplace_back(named_attr_event0);

  std::vector<NamedAttrs> attrs_event1;
  NamedAttrs named_attr_event1;
  EXPECT_TRUE(AttrUtils::SetStr(named_attr_event1, "_attached_sync_res_type", "event"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attr_event1, "_attached_sync_res_key", "res_key1"));
  attrs_event1.emplace_back(named_attr_event1);

  auto ge_graph = GraphUtilsEx::GetComputeGraph(graph);
  EXPECT_NE(ge_graph, nullptr);
  auto m0 = ge_graph->FindNode("matmu0");
  EXPECT_NE(m0, nullptr);
  auto m1 = ge_graph->FindNode("matmu1");
  EXPECT_NE(m1, nullptr);

  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(m0->GetOpDesc(), "_attached_sync_res_info", attrs_event0));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(m0->GetOpDesc(), "_attached_stream_info", attrs_stream0));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(m1->GetOpDesc(), "_attached_sync_res_info", attrs_event1));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(m1->GetOpDesc(), "_attached_stream_info", attrs_stream1));
  return graph;
}

//                                    index   data
//                                       \     /
//                     ...................Case ...................
//                   .`                     |                      `.
//                 .`                   NetOutput                    `.
//   .--------------------------.                       .-----------------------------.
//   |        Sub_Data1         |                       |         Sub_Data2           |
//   |            |             |                       |              |              |
//   |          Relu1           |                       |            Relu2            |
//   |            |             |                       |              |              |
//   |       Sub_Output1        |                       |         Sub_Output2         |
//   |__________________________|                       |_____________________________|
ComputeGraphPtr BuildGraph3WithAttachedResources() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      auto index = OP_CFG(DATA)
         .InCnt(1)
         .OutCnt(1)
         .Build("index");
      auto data = OP_CFG(DATA)
         .InCnt(1)
         .OutCnt(1)
         .Build("data");
      auto case_node = OP_CFG(CASE)
         .InCnt(1)
         .OutCnt(1)
         .Build("case");
      auto net_output = OP_CFG(NETOUTPUT)
         .InCnt(1)
         .OutCnt(1);
      CHAIN(NODE(index)->NODE(case_node)->NODE("NetOutput", net_output));
      CHAIN(NODE(data)->EDGE(0, 1)->NODE(case_node));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("index")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(main_graph->FindNode("data")->GetOpDesc(), "index", 1);

  auto case1_graph = []() {
    DEF_GRAPH(g) {
      auto data1 = OP_CFG(DATA)
         .InCnt(1)
         .OutCnt(1)
         .Build("data1");
      auto relu1 = OP_CFG(RELU)
         .InCnt(1)
         .OutCnt(1)
         .Build("relu1");
      auto net_output = OP_CFG(NETOUTPUT)
         .InCnt(1)
         .OutCnt(1);
      CHAIN(NODE(data1)->NODE(relu1)->NODE("NetOutput", net_output));
    };
    return ToComputeGraph(g);
  }();
  case1_graph->SetName("case1_graph");
  ge::AttrUtils::SetInt(case1_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  auto net_out_op_desc1 = case1_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc();
  ge::AttrUtils::SetInt(net_out_op_desc1->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  std::vector<NamedAttrs> attrs_stream1;
  NamedAttrs named_attr_stream1;
  EXPECT_TRUE(AttrUtils::SetStr(named_attr_stream1, "_attached_stream_key", "feature1"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attr_stream1, "_attached_stream_depend_value_list", "0"));
  attrs_stream1.emplace_back(named_attr_stream1);
  std::vector<NamedAttrs> attrs_event1;
  NamedAttrs named_attr_event1;
  EXPECT_TRUE(AttrUtils::SetStr(named_attr_event1, "_attached_sync_res_type", "event"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attr_event1, "_attached_sync_res_key", "res_key1"));
  attrs_event1.emplace_back(named_attr_event1);

  auto relu = case1_graph->FindNode("relu1");
  EXPECT_NE(relu, nullptr);
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu->GetOpDesc(), "_attached_sync_res_info", attrs_event1));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu->GetOpDesc(), "_attached_stream_info", attrs_stream1));
  relu->GetOpDesc()->AppendIrInput("x", kIrInputRequired);

  auto case2_graph = []() {
    DEF_GRAPH(g) {
      auto data2 = OP_CFG(DATA)
         .InCnt(1)
         .OutCnt(1)
         .Build("data2");
      auto relu2 = OP_CFG(RELU)
         .InCnt(1)
         .OutCnt(1)
         .Build("relu2");
      auto net_output = OP_CFG(NETOUTPUT)
         .InCnt(1)
         .OutCnt(1);
      CHAIN(NODE(data2)->NODE(relu2)->NODE("NetOutput", net_output));
    };
    return ToComputeGraph(g);
  }();
  case2_graph->SetName("case2_graph");
  ge::AttrUtils::SetInt(case2_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  auto net_out_op_desc2 = case2_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc();
  ge::AttrUtils::SetInt(net_out_op_desc2->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  auto case_node = main_graph->FindNode("case");
  case_node->GetOpDesc()->AppendIrInput("branch_index", kIrInputRequired);
  case_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);

  main_graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"case"});
  main_graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0});

  auto &name_index = case_node->GetOpDesc()->MutableAllInputName();
  name_index.clear();
  name_index["branch_index"] = 0;
  name_index["input0"] = 1;
  case1_graph->SetParentGraph(main_graph);
  case1_graph->SetParentNode(case_node);
  case2_graph->SetParentGraph(main_graph);
  case2_graph->SetParentNode(case_node);

  main_graph->AddSubgraph(case1_graph);
  main_graph->AddSubgraph(case2_graph);
  case_node->GetOpDesc()->AddSubgraphName("case1_graph");
  case_node->GetOpDesc()->AddSubgraphName("case2_graph");
  case_node->GetOpDesc()->SetSubgraphInstanceName(0, "case1_graph");
  case_node->GetOpDesc()->SetSubgraphInstanceName(1, "case2_graph");

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"case"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  main_graph->TopologicalSorting();
  return main_graph;
}

/**
 *   Data1 -- 350Relu -- NetOutput
 */
Graph BuildContinueStreamGraph() {
  DEF_GRAPH(g1) {
    for (size_t i = 1; i < 350; ++i) {
      std::string relu_src = "relu" + std::to_string(i);
      std::string relu_dst = "relu" + std::to_string(i + 1);
      if (i + 20 > 350) {
        auto relu_continue = OP_CFG(RELU).Attr(ATTR_NAME_CONTINUOUS_STREAM_LABEL, "continue");
        CHAIN(NODE(relu_src, relu_continue)->NODE(relu_dst, relu_continue));
      } else {
        CHAIN(NODE(relu_src, RELU)->NODE(relu_dst, RELU));
      }
    }
    CHAIN(NODE("data1", DATA)->NODE("relu1"));
    CHAIN(NODE("relu350")->NODE("output1", NETOUTPUT));
  };
  return ToGeGraph(g1);
}
void MockGenerateTask() {
  auto aicore_func = [](const ge::Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    if (node.GetType() == CONSTANT) {
      return SUCCESS;
    }

    auto op_desc = node.GetOpDesc();
    op_desc->SetOpKernelLibName("AiCoreLib");
    ge::AttrUtils::SetStr(op_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_KERNEL_BIN_ID, op_desc->GetName() + "_fake_id");
    const char kernel_bin[] = "kernel_bin";
    vector<char> buffer(kernel_bin, kernel_bin + strlen(kernel_bin));
    ge::OpKernelBinPtr kernel_bin_ptr = std::make_shared<ge::OpKernelBin>("test", std::move(buffer));
    op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, kernel_bin_ptr);
    size_t arg_size = 100;
    std::vector<uint8_t> args(arg_size, 0);
    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    auto kernel_info = task_def.mutable_kernel();
    kernel_info->set_args(args.data(), args.size());
    kernel_info->set_args_size(arg_size);
    kernel_info->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    kernel_info->set_kernel_name(node.GetName());
    kernel_info->set_block_dim(1);
    uint16_t args_offset[2] = {0};
    kernel_info->mutable_context()->set_args_offset(args_offset, 2 * sizeof(uint16_t));
    kernel_info->mutable_context()->set_op_index(node.GetOpDesc()->GetId());

    task_def.set_stream_id(node.GetOpDesc()->GetStreamId());
    for (int32_t i = 0; i < 3; i++) {
      tasks.emplace_back(task_def);
    }
    if (node.GetOpDesc()->HasValidAttachedStreamId()) {
      for (auto attached_stream_id : node.GetOpDesc()->GetAttachedStreamIds()) {
        for (int32_t i = 0; i < 3; i++) {
          task_def.set_stream_id(attached_stream_id);
          tasks.emplace_back(task_def);
        }
      }
    }
    return SUCCESS;
  };

  MockForGenerateTask("AiCoreLib", aicore_func);

  auto vector_core_func = [](const ge::Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    if (node.GetType() == CONSTANT) {
      return SUCCESS;
    }

    auto op_desc = node.GetOpDesc();
    op_desc->SetOpKernelLibName("VectorCoreLib");
    size_t arg_size = 100;
    std::vector<uint8_t> args(arg_size, 0);
    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    auto kernel_info = task_def.mutable_kernel();
    kernel_info->set_args(args.data(), args.size());
    kernel_info->set_args_size(arg_size);
    kernel_info->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    kernel_info->set_kernel_name(node.GetName());
    kernel_info->set_block_dim(1);
    uint16_t args_offset[2] = {0};
    kernel_info->mutable_context()->set_args_offset(args_offset, 2 * sizeof(uint16_t));
    kernel_info->mutable_context()->set_op_index(node.GetOpDesc()->GetId());

    task_def.set_stream_id(node.GetOpDesc()->GetStreamId());
    for (int32_t i = 0; i < 3; i++) {
      tasks.emplace_back(task_def);
    }
    if (node.GetOpDesc()->HasValidAttachedStreamId()) {
      for (auto attached_stream_id : node.GetOpDesc()->GetAttachedStreamIds()) {
        for (int32_t i = 0; i < 3; i++) {
          task_def.set_stream_id(attached_stream_id);
          tasks.emplace_back(task_def);
        }
      }
    }
    return SUCCESS;
  };
  MockForGenerateTask("VectorEngine", vector_core_func);
}
}

namespace ge {
class STEST_stream_allocator : public testing::Test {
 protected:
  void SetUp() {
    MockGenerateTask();
  }

  void TearDown() {
    OpsKernelBuilderRegistry::GetInstance().Unregister("AiCoreLib");
    OpsKernelBuilderRegistry::GetInstance().Unregister("VectorEngine");
  }
};

TEST_F(STEST_stream_allocator, link_genmask_nodes) {
  auto graph = BuildGenmaskGraph();
  auto ge_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto relu2 = ge_graph->FindNode("relu2");
  ASSERT_NE(relu2, nullptr);
  auto relu2_op_desc = relu2->GetOpDesc();
  ASSERT_NE(relu2_op_desc, nullptr);
  (void)AttrUtils::SetBool(relu2_op_desc, public_attr::OP_EXEC_NEVER_TIMEOUT, true);
  // new session & add graph
  map<string, string> options;
  options["ge.streamMaxParallelNum"] = "DNN_HCCL:1";
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  Session session(options);
  ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    // ge_default_running_env.cc 中修改了DROPOUTGENMASK引擎，图中插入了一些send/recv，算子数量变化了。
    EXPECT_EQ(graph->GetDirectNodesSize(), 9);
    auto genmask1 = graph->FindNode("DropOutGenMask1");
    EXPECT_NE(genmask1, nullptr);
    auto stream_id = genmask1->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 0);
    auto genmask2 = graph->FindNode("DropOutGenMask2");
    EXPECT_NE(genmask2, nullptr);
    stream_id = genmask2->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 0);
  };
}

TEST_F(STEST_stream_allocator, cmo_with_label_nodes) {
  auto graph = BuildCmoGraph();

  // new session & add graph
  map<string, string> options;
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  Session session(options);
  ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto cmo = graph->FindNode("cmo1");
    ASSERT_NE(cmo, nullptr);
    auto stream_id = cmo->GetOpDesc()->GetStreamId();
    EXPECT_TRUE(cmo->GetOpDesc()->HasAttr(ATTR_NAME_STREAM_LABEL));
    auto matmul = graph->FindNode("matmul");
    ASSERT_NE(matmul, nullptr);
    auto stream_id1 = matmul->GetOpDesc()->GetStreamId();
    EXPECT_NE(stream_id1, stream_id);
  };
}

TEST_F(STEST_stream_allocator, single_stream) {
  auto graph = BuildGenmaskGraph();
  map<string, string> options;
  options["ge.streamMaxParallelNum"] = "DNN_HCCL:10";
  options["ge.enableSingleStream"] = "true";
  GetThreadLocalContext().SetGlobalOption(options);

  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  Session session(options);
  ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    // ge_default_running_env.cc 中修改了DROPOUTGENMASK引擎，图中插入了一些send/recv，算子数量变化了。
    EXPECT_EQ(graph->GetDirectNodesSize(), 9);
    auto genmask1 = graph->FindNode("DropOutGenMask1");
    EXPECT_NE(genmask1, nullptr);
    auto stream_id = genmask1->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 0);
    auto genmask2 = graph->FindNode("DropOutGenMask2");
    EXPECT_NE(genmask2, nullptr);
    stream_id = genmask2->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 0);
  };

  options["ge.enableSingleStream"] = "false";
  GetThreadLocalContext().SetGlobalOption(options);
  ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(STEST_stream_allocator, single_stream_with_huge_stream) {
  auto graph = BuildHCCLGraphWith5HcclNode();
  map<string, string> options;
  options["ge.streamMaxParallelNum"] = "DNN_HCCL:10";
  options["ge.enableSingleStream"] = "true";
  GetThreadLocalContext().SetGlobalOption(options);

  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  Session session(options);
  ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto hcom1 = graph->FindNode("hcom1");
    EXPECT_NE(hcom1, nullptr);
    auto hcom1_stream_id = hcom1->GetOpDesc()->GetStreamId();
    EXPECT_EQ(hcom1_stream_id, 0);
    auto hcom5 = graph->FindNode("hcom5");
    EXPECT_NE(hcom5, nullptr);
    auto hcom5_stream_id = hcom5->GetOpDesc()->GetStreamId();
    EXPECT_EQ(hcom5_stream_id, hcom1_stream_id);
  };

  options["ge.enableSingleStream"] = "false";
  GetThreadLocalContext().SetGlobalOption(options);
  ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(STEST_stream_allocator, hcom_nodes_independent_stream) {
  auto graph = BuildHCCLGraph();
  map<string, string> options;
  options["ge.hcomParallel"] = "1";

  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    // ge_default_running_env.cc 中修改了DROPOUTGENMASK引擎，图中插入了一些send/recv，算子数量变化了。
    EXPECT_EQ(graph->GetDirectNodesSize(), 12);
    auto hcom1 = graph->FindNode("hcom1");
    EXPECT_NE(hcom1, nullptr);
    auto stream_id = hcom1->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 0);
    EXPECT_EQ(hcom1->GetInControlNodesSize(), 1);
    EXPECT_EQ(hcom1->GetInControlNodes().at(0)->GetType(), "Recv"); // hcom的跨流等待节点
    auto relu1 = graph->FindNode("relu1");
    EXPECT_NE(relu1, nullptr);
    stream_id = relu1->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 2);
  };
}

TEST_F(STEST_stream_allocator, cv_parallel_assign_diff_stream) {
  auto graph = gert::ShareGraph::BuildCVParallelGraph();
  const auto back_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  auto options = back_options;
  options["ge.autoMultistreamParallelMode"] = "cv";
  ge::GetThreadLocalContext().SetSessionOption(options);

  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto aiv1 = graph->FindNode("aiv1");
    EXPECT_NE(aiv1, nullptr);
    auto stream_id = aiv1->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 0);
    auto aiv2 = graph->FindNode("aiv2");
    EXPECT_NE(aiv2, nullptr);
    stream_id = aiv2->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 1);
    auto aiv3 = graph->FindNode("aiv3");
    EXPECT_NE(aiv3, nullptr);
    stream_id = aiv3->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 1);
  };
}

TEST_F(STEST_stream_allocator, cv_serial_assign_same_stream) {
  auto graph = gert::ShareGraph::BuildCVSerialGraph();
  const auto back_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  auto options = back_options;
  options["ge.autoMultistreamParallelMode"] = "cv";
  ge::GetThreadLocalContext().SetSessionOption(options);

  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto aiv1 = graph->FindNode("aiv1");
    EXPECT_NE(aiv1, nullptr);
    auto stream_id = aiv1->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 0);
    auto aiv2 = graph->FindNode("aiv2");
    EXPECT_NE(aiv2, nullptr);
    stream_id = aiv2->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 0);
  };
}

 /** 纯静态模型，支持自定义流pass接入修改hcom1的stream id，使其随前序节点的流*/
TEST_F(STEST_stream_allocator, hcom_nodes_independent_stream_custom_pass) {
  REGISTER_CUSTOM_PASS("BuildHCCLGraphStreamPass")
   .CustomAllocateStreamPassFn(BuildHCCLGraphStreamPass)
   .Stage(CustomPassStage::kAfterAssignLogicStream);
  auto graph = BuildHCCLGraph();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->SetName("BuildHCCLGraphStreamPass");
  map<string, string> options;
  options["ge.hcomParallel"] = "1";

  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto hcom1 = graph->FindNode("hcom1");
    EXPECT_NE(hcom1, nullptr);
    auto stream_id = hcom1->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 1);
    EXPECT_EQ(hcom1->GetInControlNodesSize(), 0); // hcom无跨流等待节点
  };
}

TEST_F(STEST_stream_allocator, custom_stream_pass_assign_stream) {
  REGISTER_CUSTOM_PASS("ReluCustomStreamPass")
      .CustomAllocateStreamPassFn(ReluCustomStreamPass)
      .Stage(CustomPassStage::kAfterAssignLogicStream);
  auto graph = BuildHCCLGraph();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->SetName("ReluCustomStreamPass");
  map<string, string> options;

  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build_graph through session
  ret = session.CompileGraph(0);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto hcom1 = graph->FindNode("relu1");
    EXPECT_NE(hcom1, nullptr);
    auto stream_id = hcom1->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 3);
    auto compiled_summary = session.GetCompiledGraphSummary(0);
    ASSERT_NE(compiled_summary, nullptr);
    std::string expect_stream_info_3 =
        "logic_stream_id: 3, user_stream_label: , is_assigned_by_user_stream_pass: true, attached_stream_ids: "
        ", physical_model_stream_num: 1, hccl_followed_stream_num: 0.\n";
    std::shared_ptr<StreamAllocationSummary> stream_summary;
    EXPECT_EQ(compiled_summary->GetStreamAllocationSummary(stream_summary), SUCCESS);
    auto graph_to_stream_infos = stream_summary->GetAllLogicalStreamInfos();
    ASSERT_EQ(graph_to_stream_infos.size(), 1U);
    auto iter = graph_to_stream_infos.find(AscendString("ReluCustomStreamPass"));
    ASSERT_TRUE(iter != graph_to_stream_infos.end());
    ASSERT_EQ(iter->second.size(), 4U);
    const auto &logical_stream_3_info = iter->second[3];
    EXPECT_EQ(logical_stream_3_info.ToStringInfo().GetString(), expect_stream_info_3);
    EXPECT_EQ(logical_stream_3_info.IsAssignedByStreamPass(), true);
  };
}

  /** 纯静态模型，支持自定义stream label修改hcom1的stream，使其随前序节点的流*/
TEST_F(STEST_stream_allocator, hcom_nodes_independent_stream_user_stream_label) {
  auto graph = BuildHCCLGraph();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto hcom1 = compute_graph->FindNode("hcom1");
  auto relu1 = compute_graph->FindNode("relu1");
  auto data1 = compute_graph->FindNode("data1");
  AttrUtils::SetStr(data1->GetOpDesc(), public_attr::USER_STREAM_LABEL, "stream1");
  AttrUtils::SetStr(hcom1->GetOpDesc(), public_attr::USER_STREAM_LABEL, "stream1");
  AttrUtils::SetStr(relu1->GetOpDesc(), public_attr::USER_STREAM_LABEL, "stream1");

  map<string, string> options;
  options["ge.hcomParallel"] = "1";

  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build_graph through session
  ret = session.CompileGraph(0);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto hcom1 = graph->FindNode("hcom1");
    EXPECT_NE(hcom1, nullptr);
    auto relu1 = graph->FindNode("relu1");
    EXPECT_NE(hcom1, nullptr);
    auto hcom1_stream_id = hcom1->GetOpDesc()->GetStreamId();
    auto relu1_stream_id = relu1->GetOpDesc()->GetStreamId();
    EXPECT_EQ(hcom1_stream_id, relu1_stream_id);

    auto data1 = compute_graph->FindNode("data1");
    std::string user_stream_label;
    AttrUtils::GetStr(data1->GetOpDesc(), public_attr::USER_STREAM_LABEL, user_stream_label);
    EXPECT_TRUE(user_stream_label.empty());

    auto compiled_summary = session.GetCompiledGraphSummary(0);
    ASSERT_NE(compiled_summary, nullptr);
    std::string expect_stream_info_0 =
        "logic_stream_id: 0, user_stream_label: stream1, is_assigned_by_user_stream_pass: false, attached_stream_ids: "
        ", physical_model_stream_num: 1, hccl_followed_stream_num: 0.\n";
    std::string expect_stream_info_1 =
        "logic_stream_id: 1, user_stream_label: , is_assigned_by_user_stream_pass: false, attached_stream_ids: , "
        "physical_model_stream_num: 1, hccl_followed_stream_num: 0.\n";
    std::shared_ptr<StreamAllocationSummary> stream_summary;
    EXPECT_EQ(compiled_summary->GetStreamAllocationSummary(stream_summary), SUCCESS);
    auto graph_to_stream_infos = stream_summary->GetAllLogicalStreamInfos();
    ASSERT_EQ(graph_to_stream_infos.size(), 1U);
    auto iter = graph_to_stream_infos.find(AscendString("g1"));
    ASSERT_TRUE(iter != graph_to_stream_infos.end());
    ASSERT_EQ(iter->second.size(), 2U);
    const auto &logical_stream_0_info = iter->second[0];
    const auto &logical_stream_1_info = iter->second[1];
    EXPECT_EQ(logical_stream_0_info.ToStringInfo().GetString(), expect_stream_info_0);
    EXPECT_EQ(logical_stream_1_info.ToStringInfo().GetString(), expect_stream_info_1);

    std::string expect_user_stream_label = "stream1";
    EXPECT_EQ(logical_stream_0_info.GetUsrStreamLabel().GetString(), expect_user_stream_label);
  };
}

TEST_F(STEST_stream_allocator, hcom_nodes_independent_stream_fail) {
  auto graph = BuildHCCLGraph();
  map<string, string> options;
  options["ge.hcomParallel"] = "1";

  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  g_runtime_stub_mock = "rtGetMaxStreamAndTask";
  ret = session.BuildGraph(0, inputs);
  EXPECT_NE(ret, SUCCESS);
  g_runtime_stub_mock = "";
}

TEST_F(STEST_stream_allocator, switch_merge_big_graph_split_stream) {
  auto graph =  BuildSwitchMergeBigGraph();
  // new session & add graph
  map<string, string> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto switch1 = graph->FindFirstNodeMatchType(STREAMSWITCH);
    EXPECT_NE(switch1, nullptr);
    auto stream_id = switch1->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 0);
    auto merge1 = graph->FindFirstNodeMatchType(STREAMMERGE);
    EXPECT_NE(merge1, nullptr);
    stream_id = merge1->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 3);
    auto active1 = graph->FindFirstNodeMatchType(STREAMACTIVE);
    EXPECT_NE(active1, nullptr);
    stream_id = active1->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 4);
    auto true_relu700 = graph->FindNode("true_relu700");
    EXPECT_NE(true_relu700, nullptr);
    stream_id = true_relu700->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 10);
    auto false_relu700 = graph->FindNode("false_relu700");
    EXPECT_NE(false_relu700, nullptr);
    stream_id = false_relu700->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 12);
  };
}

TEST_F(STEST_stream_allocator, partitionedcall_with_subgraph) {
  auto graph =  BuildPartitionedCallGraph();
  // new session & add graph
  map<string, string> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 1);
    auto subgraph = graph->GetAllSubgraphs().at(0);
    EXPECT_NE(subgraph, nullptr);
    auto active1 = subgraph->FindFirstNodeMatchType(STREAMACTIVE);
    EXPECT_NE(active1, nullptr);
    auto stream_id = active1->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 0);
    std::vector<int64_t> active_stream_list;
    AttrUtils::GetListInt(active1->GetOpDesc(), "active_stream_list", active_stream_list);
    EXPECT_EQ(active_stream_list.size(), 2);
    EXPECT_EQ(active_stream_list[0], 1);
    EXPECT_EQ(active_stream_list[1], 2);
  };
}

TEST_F(STEST_stream_allocator, continue_stream_graph) {
  auto graph = BuildContinueStreamGraph();
  // new session & add graph
  map<string, string> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto relu335 = graph->FindNode("relu335");
    EXPECT_NE(relu335, nullptr);
    auto stream_id = relu335->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 1);
  };
}

TEST_F(STEST_stream_allocator, ffts_static_graph_with_diff_stream) {
  auto graph = BuildStaticFftsGraph();
  // new session & add graph
  map<string, string> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    auto partitionedCall0 = graph->FindNode("partitionedCall0");
    EXPECT_NE(partitionedCall0, nullptr);
    auto stream_id_0 = partitionedCall0->GetOpDesc()->GetStreamId();

    auto partitionedCall1 = graph->FindNode("partitionedCall1");
    EXPECT_NE(partitionedCall1, nullptr);
    auto stream_id_1 = partitionedCall1->GetOpDesc()->GetStreamId();
    EXPECT_NE(stream_id_0, stream_id_1);

    bool has_send{false};
    for(const auto &out_node :partitionedCall0->GetOutControlNodes()) {
      if (out_node == nullptr) {
        continue;
      }
      if (out_node->GetType() == SEND) {
        has_send = true;
        break;
      }
    }
    EXPECT_TRUE(has_send);
    
    bool has_recv{false};
    for(const auto &in_node :partitionedCall1->GetInControlNodes()) {
      if (in_node == nullptr) {
        continue;
      }
      if (in_node->GetType() == RECV) {
        has_recv = true;
        break;
      }
    }
    EXPECT_TRUE(has_recv);
  };
}

TEST_F(STEST_stream_allocator, ffts_dynamic_graph_with_diff_stream) {
  setenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM", "1", 0);
  auto root_graph = BuildDynamicFftsGraph();
  for (const auto &node : root_graph->GetAllNodes()) {
    if (node->GetName() == "sgt0/Node_Output") {
      AttrUtils::SetInt(node->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
    }
    if (node->GetName() == "partitionedCall0") {
      AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_FFTS_PLUS_SUB_GRAPH, true);
      node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
      AttrUtils::SetStr(node->GetOpDesc(), ATTR_NAME_COMPOSITE_ENGINE_KERNEL_LIB_NAME, "DNN_VM_GE_LOCAL_OP_STORE");
    }
  }
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  // new session & add graph
  map<string, string> options = {{"ge.runFlag", "0"}};
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);
  unsetenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM");
}

TEST_F(STEST_stream_allocator, switch_merge_big_graph_split_stream1) {
  auto graph =  BuildSwitchMergeBigGraph();
  // new session & add graph
  map<string, string> options;
  Session session(options);
  options[EVENT] = "notify";
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto switch1 = graph->FindFirstNodeMatchType(STREAMSWITCH);
    EXPECT_NE(switch1, nullptr);
    auto stream_id = switch1->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 0);
    auto merge1 = graph->FindFirstNodeMatchType(STREAMMERGE);
    EXPECT_NE(merge1, nullptr);
    stream_id = merge1->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 3);
    auto active1 = graph->FindFirstNodeMatchType(STREAMACTIVE);
    EXPECT_NE(active1, nullptr);
    stream_id = active1->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 4);
    auto true_relu700 = graph->FindNode("true_relu700");
    EXPECT_NE(true_relu700, nullptr);
    stream_id = true_relu700->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 10);
    auto false_relu700 = graph->FindNode("false_relu700");
    EXPECT_NE(false_relu700, nullptr);
    stream_id = false_relu700->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 12);
  };
}

TEST_F(STEST_stream_allocator, PartitionedcallWithSubgraph_Success_EnableNotify) {
  auto graph =  BuildPartitionedCallGraph();
  // new session & add graph
  map<string, string> options;
  options[EVENT] = "notify";
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 1);
    auto subgraph = graph->GetAllSubgraphs().at(0);
    EXPECT_NE(subgraph, nullptr);
    auto active1 = subgraph->FindFirstNodeMatchType(STREAMACTIVE);
    EXPECT_NE(active1, nullptr);
    auto stream_id = active1->GetOpDesc()->GetStreamId();
    EXPECT_EQ(stream_id, 0);
    std::vector<int64_t> active_stream_list;
    AttrUtils::GetListInt(active1->GetOpDesc(), "active_stream_list", active_stream_list);
    EXPECT_EQ(active_stream_list.size(), 2);
    EXPECT_EQ(active_stream_list[0], 1);
    EXPECT_EQ(active_stream_list[1], 2);
  };
}

TEST_F(STEST_stream_allocator, AicoreHcclParallel) {
  auto graph =  BuildGraphWithAicoreHcclParallel();
  std::map<string, string> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  std::vector<InputTensorInfo> inputs;
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto all_reduce = graph->FindNode("HcomAllReduce");
    ASSERT_NE(all_reduce, nullptr);
    EXPECT_EQ(all_reduce->GetOpDesc()->GetStreamId(), 0);
    auto relu2 = graph->FindNode("relu2");
    ASSERT_NE(relu2, nullptr);
    EXPECT_EQ(relu2->GetOpDesc()->GetStreamId(), 1);
  };
}

TEST_F(STEST_stream_allocator, AicoreHcclSerial) {
  auto graph =  BuildGraphWithAicoreHcclSerial();
  std::map<string, string> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  std::vector<InputTensorInfo> inputs;
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto all_reduce = graph->FindNode("HcomAllReduce");
    ASSERT_NE(all_reduce, nullptr);
    EXPECT_EQ(all_reduce->GetOpDesc()->GetStreamId(), 0);
    auto relu2 = graph->FindNode("relu2");
    ASSERT_NE(relu2, nullptr);
    EXPECT_EQ(relu2->GetOpDesc()->GetStreamId(), 0);
  };
}

TEST_F(STEST_stream_allocator, AicoreHcclSerialAndMultiHcclSerial) {
  auto graph =  BuildGraphWithAicoreHcclSerialAndMultiHcclSerial();
  std::map<string, string> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  std::vector<InputTensorInfo> inputs;
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto all_reduce1 = graph->FindNode("HcomAllReduce1");
    ASSERT_NE(all_reduce1, nullptr);
    EXPECT_EQ(all_reduce1->GetOpDesc()->GetStreamId(), 0);
    auto all_reduce2 = graph->FindNode("HcomAllReduce2");
    ASSERT_NE(all_reduce2, nullptr);
    EXPECT_EQ(all_reduce2->GetOpDesc()->GetStreamId(), 0);
    auto relu2 = graph->FindNode("relu2");
    ASSERT_NE(relu2, nullptr);
    EXPECT_EQ(relu2->GetOpDesc()->GetStreamId(), 1);
  };
}

TEST_F(STEST_stream_allocator, DisableOptimizeIneffectiveMultiStream) {
  mmSetEnv(kDisableIneffectiveMultiStreamOptimize, "1", 1);
  auto graph =  BuildGraphWithAicoreHcclSerial();
  std::map<string, string> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  std::vector<InputTensorInfo> inputs;
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto all_reduce = graph->FindNode("HcomAllReduce");
    ASSERT_NE(all_reduce, nullptr);
    EXPECT_EQ(all_reduce->GetOpDesc()->GetStreamId(), 0);
    auto relu2 = graph->FindNode("relu2");
    ASSERT_NE(relu2, nullptr);
    EXPECT_EQ(relu2->GetOpDesc()->GetStreamId(), 1);
  };
  mmSetEnv(kDisableIneffectiveMultiStreamOptimize, "0", 1);
}

TEST_F(STEST_stream_allocator, optimize_ineffective_multi_stream_not_move_to_stream_label) {
  GeRunningEnvFaker().InstallDefault();
  auto graph = BuildGraphWithStreamLabelAndIneffectiveMultiStream();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto trans1 = compute_graph->FindNode("trans1");
  auto trans2 = compute_graph->FindNode("trans2");
  AttrUtils::SetStr(trans1->GetOpDesc(), public_attr::USER_STREAM_LABEL, "stream_label_test");
  AttrUtils::SetStr(trans2->GetOpDesc(), public_attr::USER_STREAM_LABEL, "stream_label_test");


  map<string, string> options;
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  Session session(options);
  ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  std::vector<InputTensorInfo> inputs;
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto trans1_node = graph->FindNode("trans1");
    ASSERT_NE(trans1_node, nullptr);
    auto relu_node = graph->FindNode("relu");
    ASSERT_NE(relu_node, nullptr);
    auto trans2_node = graph->FindNode("trans2");
    ASSERT_NE(trans2_node, nullptr);

    auto trans1_stream_id = trans1_node->GetOpDesc()->GetStreamId();
    auto trans2_stream_id = trans2_node->GetOpDesc()->GetStreamId();
    auto relu_stream_id = relu_node->GetOpDesc()->GetStreamId();

    EXPECT_EQ(trans1_stream_id, trans2_stream_id);
    EXPECT_NE(trans1_stream_id, relu_stream_id);
  };
}

TEST_F(STEST_stream_allocator, ParallelGroupPass_Success_EnableHcomReuseStreamId) {
  auto graph =  BuildParallelGroupTagGraph();
  // new session & add graph
  map<string, string> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    int32_t send_cnt = 0;
    int32_t recv_cnt = 0;
    for (const auto &node : graph->GetDirectNode()) {
      if (node->GetType() == SEND) {
        send_cnt++;
      }
      if (node->GetType() == RECV) {
        recv_cnt++;
      }
    }
    EXPECT_EQ(send_cnt, 2);
    EXPECT_EQ(recv_cnt, 2);
    auto send_node = graph->FindFirstNodeMatchType(SEND);
    EXPECT_NE(send_node, nullptr);
    auto recv_node = graph->FindFirstNodeMatchType(RECV);
    EXPECT_NE(recv_node, nullptr);

    auto relu0 = graph->FindNode("relu0");
    auto allgather0 = graph->FindNode("allgather0");
    EXPECT_NE(relu0, nullptr);
    EXPECT_NE(allgather0, nullptr);
    EXPECT_EQ(relu0->GetOpDesc()->GetStreamId(), allgather0->GetOpDesc()->GetStreamId());

    auto relu1 = graph->FindNode("relu1");
    auto allgather1 = graph->FindNode("allgather1");
    auto add0 = graph->FindNode("add0");
    EXPECT_NE(relu1, nullptr);
    EXPECT_NE(allgather1, nullptr);
    EXPECT_NE(add0, nullptr);
    EXPECT_NE(relu1->GetOpDesc()->GetStreamId(), allgather1->GetOpDesc()->GetStreamId());
    EXPECT_NE(add0->GetOpDesc()->GetStreamId(), allgather1->GetOpDesc()->GetStreamId());
  };
}
TEST_F(STEST_stream_allocator, AttachedResourceAssign) {
  auto graph = BuildGraphWithAttachedResources();
  // new session & add graph
  map<string, string> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    // check 不同的通信域的从流不复用
    auto mc2_0 = graph->FindNode("mc2_0");
    EXPECT_NE(mc2_0, nullptr);
    auto mc2_1 = graph->FindNode("mc2_1");
    EXPECT_NE(mc2_1, nullptr);
    auto mc2_2 = graph->FindNode("mc2_2");
    EXPECT_NE(mc2_2, nullptr);
    auto mc2_3 = graph->FindNode("mc2_3");
    EXPECT_NE(mc2_3, nullptr);
    auto mc2_4 = graph->FindNode("mc2_4");
    EXPECT_NE(mc2_4, nullptr);
    auto mc2_5 = graph->FindNode("mc2_5");
    EXPECT_NE(mc2_5, nullptr);

    // 通信域g0内节点按照attached key分配了不同资源
    EXPECT_NE(mc2_0->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
    EXPECT_NE(mc2_1->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
    EXPECT_EQ(mc2_0->GetOpDescBarePtr()->GetAttachedStreamId(), mc2_1->GetOpDescBarePtr()->GetAttachedStreamId());
    EXPECT_NE(mc2_2->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
    EXPECT_NE(mc2_2->GetOpDescBarePtr()->GetAttachedStreamId(), mc2_1->GetOpDescBarePtr()->GetAttachedStreamId());
    EXPECT_EQ(mc2_1->GetOutControlNodes().size(), 1U);
    EXPECT_EQ(mc2_3->GetOutControlNodes().size(), 1U);
    EXPECT_EQ(mc2_5->GetOutControlNodes().size(), 1U);
    // 每个附属流的last节点都跟模型的输出节点建立了同步关系
    const auto mc2_1_out = mc2_1->GetOutControlNodes().at(0U);
    const auto mc2_3_out = mc2_3->GetOutControlNodes().at(0U);
    const auto mc2_5_out = mc2_5->GetOutControlNodes().at(0U);
    EXPECT_EQ(mc2_1_out->GetType(), SEND);
    EXPECT_EQ(mc2_3_out->GetType(), SEND);
    EXPECT_EQ(mc2_5_out->GetType(), SEND);
    // send节点跟mc2的附属流同流
    EXPECT_EQ(mc2_1->GetOpDesc()->GetAttachedStreamId(), mc2_1_out->GetOpDesc()->GetStreamId());
    EXPECT_EQ(mc2_3->GetOpDesc()->GetAttachedStreamId(), mc2_3_out->GetOpDesc()->GetStreamId());
    EXPECT_EQ(mc2_5->GetOpDesc()->GetAttachedStreamId(), mc2_5_out->GetOpDesc()->GetStreamId());
    // recv节点跟netoutput同流
    auto output_node = graph->FindNode("output0");
    EXPECT_NE(output_node, nullptr);
    EXPECT_EQ(output_node->GetInControlNodesSize(), 3U);
    EXPECT_EQ(output_node->GetInControlNodes().at(0U)->GetType(), RECV);
    EXPECT_EQ(output_node->GetInControlNodes().at(0U)->GetOpDesc()->GetStreamId(),
              output_node->GetOpDesc()->GetStreamId());
    EXPECT_EQ(output_node->GetInControlNodes().at(1U)->GetType(), RECV);
    EXPECT_EQ(output_node->GetInControlNodes().at(1U)->GetOpDesc()->GetStreamId(),
              output_node->GetOpDesc()->GetStreamId());
    EXPECT_EQ(output_node->GetInControlNodes().at(2U)->GetType(), RECV);
    EXPECT_EQ(output_node->GetInControlNodes().at(2U)->GetOpDesc()->GetStreamId(),
              output_node->GetOpDesc()->GetStreamId());

    // Build图中相关属性因为生命周期结束，已经被清理
    EXPECT_FALSE(mc2_0->GetOpDescBarePtr()->HasAttr(ATTR_NAME_ATTACHED_STREAM_POLICY));
    EXPECT_FALSE(mc2_0->GetOpDescBarePtr()->HasAttr(ATTR_NAME_ATTACHED_NOTIFY_POLICY));
  };
}

TEST_F(STEST_stream_allocator, AttachedResourceAssign2) {
  auto graph = BuildGraph2WithAttachedResources();
  // new session & add graph
  map<string, string> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    // check 不同的通信域的从流不复用
    auto mm_0 = graph->FindNode("matmu0");
    EXPECT_NE(mm_0, nullptr);
    auto mm_1 = graph->FindNode("matmu1");
    EXPECT_NE(mm_1, nullptr);

    // 节点按照attached key分配了不同资源
    EXPECT_NE(mm_0->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
    EXPECT_NE(mm_1->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
    EXPECT_NE(mm_0->GetOpDescBarePtr()->GetAttachedStreamId(), mm_1->GetOpDescBarePtr()->GetAttachedStreamId());

    // Build图中相关属性因为生命周期结束，已经被清理
    EXPECT_FALSE(mm_0->GetOpDescBarePtr()->HasAttr(ATTR_NAME_ATTACHED_STREAM_POLICY));
    EXPECT_FALSE(mm_1->GetOpDescBarePtr()->HasAttr("_attached_sync_res_info"));
  };
}

TEST_F(STEST_stream_allocator, AttachedResourceAssign3) {
  auto compute_graph = BuildGraph3WithAttachedResources();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto ge_env = GeRunningEnvFaker();
  ge_env.Reset()
         .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine").GraphOptimizer("AIcoreEngine"))
         .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER"))
         .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("RTSLib").GraphOptimizer("DNN_VM_RTS_GRAPH_OPTIMIZER_STORE").Priority(PriorityEnum::COST_1))
         .Install(FakeOp(RELU).Inputs({"x"}).InfoStoreAndBuilder("AIcoreEngine"))
         .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(CASE).Inputs({"branch_index", "input"}).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp("LabelSwitchByIndex").InfoStoreAndBuilder("RTSLib"))
         .Install(FakeOp("LabelSet").InfoStoreAndBuilder("RTSLib"))
         .Install(FakeOp("LabelGotoEx").InfoStoreAndBuilder("RTSLib"))
         .Install(FakeOp("Send").InfoStoreAndBuilder("RTSLib"))
         .Install(FakeOp("Recv").InfoStoreAndBuilder("RTSLib"))
         .Install(FakeOp(IDENTITY).InfoStoreAndBuilder("RTSLib"))
         .Install(FakeOp(STREAMACTIVE).InfoStoreAndBuilder("RTSLib"))
         .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"));
  // new session & add graph
  map<string, string> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build_graph through session
  ret = session.CompileGraph(0);
  EXPECT_EQ(ret, SUCCESS);
  auto compiled_summary = session.GetCompiledGraphSummary(0);
  ASSERT_NE(compiled_summary, nullptr);
  std::shared_ptr<StreamAllocationSummary> stream_summary;
  EXPECT_EQ(compiled_summary->GetStreamAllocationSummary(stream_summary), SUCCESS);
  EXPECT_TRUE(stream_summary->ToStreamGraph().size()>0);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto case_node = graph->FindNode("case");
    EXPECT_NE(case_node, nullptr);
    auto case_sub_graph1 = NodeUtils::GetSubgraph(*case_node, 0);
    EXPECT_NE(case_sub_graph1, nullptr);
    auto relu1 = case_sub_graph1->FindFirstNodeMatchType(RELU);
    EXPECT_NE(relu1, nullptr);
    auto stream_active = case_sub_graph1->FindFirstNodeMatchType(STREAMACTIVE);
    EXPECT_NE(stream_active, nullptr);
    EXPECT_EQ(stream_active->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
    EXPECT_NE(relu1->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
    auto send_node = case_sub_graph1->FindFirstNodeMatchType(SEND);
    EXPECT_NE(send_node, nullptr);
    EXPECT_TRUE(stream_active->GetOutControlAnchor()->IsLinkedWith(send_node->GetInControlAnchor()));

    // Build图中相关属性因为生命周期结束，已经被清理
    EXPECT_FALSE(relu1->GetOpDescBarePtr()->HasAttr(ATTR_NAME_ATTACHED_STREAM_POLICY));
    EXPECT_FALSE(relu1->GetOpDescBarePtr()->HasAttr("_attached_sync_res_info"));
  };
  ge_env.InstallDefault();
}

TEST_F(STEST_stream_allocator, MultiAttachedStreamWithHugeNodeSuccess) {
  std::vector<int64_t> attached_streams = {1, 2};
  auto graph = BuildGraphWithMultiAttachedStreamAndHugeNodesAndHugeNodes(345, attached_streams);
  // new session & add graph
  map<string, string> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build_graph through session
  ret = session.CompileGraph(0);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    auto mc2 = graph->FindNode("mc2_345");
    EXPECT_NE(mc2, nullptr);

    ASSERT_EQ(mc2->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 2);

    // 每个附属流的last节点都跟模型的输出节点建立了同步关系
    ASSERT_EQ(mc2->GetOutControlNodes().size(), 2U);
    const auto mc2_out_0 = mc2->GetOutControlNodes().at(0U);
    const auto mc2_out_1 = mc2->GetOutControlNodes().at(1U);
    EXPECT_EQ(mc2_out_0->GetType(), SEND);
    EXPECT_EQ(mc2_out_1->GetType(), SEND);
    // send节点跟mc2的附属流同流
    EXPECT_EQ(mc2->GetOpDesc()->GetAttachedStreamIds()[0], mc2_out_1->GetOpDesc()->GetStreamId());
    EXPECT_EQ(mc2->GetOpDesc()->GetAttachedStreamIds()[1], mc2_out_0->GetOpDesc()->GetStreamId());
    // recv节点跟netoutput同流
    auto output_node = graph->FindNode("output0");
    EXPECT_NE(output_node, nullptr);
    EXPECT_EQ(output_node->GetInControlNodesSize(), 2U);
    EXPECT_EQ(output_node->GetInControlNodes().at(0U)->GetType(), RECV);
    EXPECT_EQ(output_node->GetInControlNodes().at(0U)->GetOpDesc()->GetStreamId(),
              output_node->GetOpDesc()->GetStreamId());
    EXPECT_EQ(output_node->GetInControlNodes().at(1U)->GetType(), RECV);
    EXPECT_EQ(output_node->GetInControlNodes().at(1U)->GetOpDesc()->GetStreamId(),
              output_node->GetOpDesc()->GetStreamId());
    auto compiled_summary = session.GetCompiledGraphSummary(0);
    ASSERT_NE(compiled_summary, nullptr);
    std::string expect_stream_info_0 =
        "logic_stream_id: 0, user_stream_label: , is_assigned_by_user_stream_pass: false, attached_stream_ids: 1 2 , "
        "physical_model_stream_num: 2, hccl_followed_stream_num: 4.\n";
    std::string expect_stream_info_1 =
        "logic_stream_id: 1, user_stream_label: , is_assigned_by_user_stream_pass: false, attached_stream_ids: , "
        "physical_model_stream_num: 2, hccl_followed_stream_num: 0.\n";
    std::string expect_stream_info_2 =
        "logic_stream_id: 2, user_stream_label: , is_assigned_by_user_stream_pass: false, attached_stream_ids: , "
        "physical_model_stream_num: 2, hccl_followed_stream_num: 0.\n";
    std::shared_ptr<StreamAllocationSummary> stream_summary;
    EXPECT_EQ(compiled_summary->GetStreamAllocationSummary(stream_summary), SUCCESS);
    auto graph_to_stream_infos = stream_summary->GetAllLogicalStreamInfos();
    ASSERT_EQ(graph_to_stream_infos.size(), 1U);
    auto iter = graph_to_stream_infos.find(AscendString("g1"));
    ASSERT_TRUE(iter != graph_to_stream_infos.end());
    ASSERT_EQ(iter->second.size(), 3U);
    EXPECT_EQ(iter->second[0].ToStringInfo().GetString(), expect_stream_info_0);
    EXPECT_EQ(iter->second[1].ToStringInfo().GetString(), expect_stream_info_1);
    EXPECT_EQ(iter->second[2].ToStringInfo().GetString(), expect_stream_info_2);

    const auto &logical_stream_0_stream_info = iter->second[0];
    EXPECT_EQ(logical_stream_0_stream_info.GetLogicalStreamId(), 0);
    std::vector<int64_t> expect_attached_stream_ids = {1, 2};
    EXPECT_EQ(logical_stream_0_stream_info.GetAttachedStreamIds(), expect_attached_stream_ids);
    EXPECT_EQ(logical_stream_0_stream_info.GetPhysicalStreamNum(), 2U);
    EXPECT_EQ(logical_stream_0_stream_info.GetHcclFollowedStreamNum(), 4U);
    EXPECT_EQ(logical_stream_0_stream_info.IsAssignedByStreamPass(), false);

    auto stream_graphs = stream_summary->ToStreamGraph();
    ASSERT_EQ(stream_graphs.size(), 1U);
    auto stream_graphs_iter = stream_graphs.find(AscendString("g1"));
    ASSERT_TRUE(stream_graphs_iter != stream_graphs.end());

    std::map<AscendString, std::vector<AscendString>> graph_to_string_infos;
    ASSERT_EQ(GEStreamAllocationSummaryGetStringInfos(*compiled_summary, graph_to_string_infos), SUCCESS);
    ASSERT_EQ(graph_to_string_infos.begin()->second.size(), 3U);
    EXPECT_EQ(graph_to_string_infos.begin()->second[0].GetString(), expect_stream_info_0);

    std::map<AscendString, std::vector<int64_t>> graph_to_logical_stream_ids;
    ASSERT_EQ(GEStreamAllocationSummaryGetLogicalStreamIds(*compiled_summary, graph_to_logical_stream_ids), SUCCESS);
    EXPECT_EQ(graph_to_logical_stream_ids.begin()->second.size(), 3U);
    EXPECT_EQ(graph_to_logical_stream_ids.begin()->second[0], 0U);

    std::map<AscendString, std::vector<AscendString>> graph_to_user_stream_labels;
    ASSERT_EQ(GEStreamAllocationSummaryGetUsrStreamLabels(*compiled_summary, graph_to_user_stream_labels), SUCCESS);
    EXPECT_EQ(graph_to_user_stream_labels.begin()->second.size(), 3U);
    EXPECT_EQ(graph_to_user_stream_labels.begin()->second[0].GetLength(), 0U);

    std::map<AscendString, std::vector<bool>> graph_to_is_assigned_by_stream_pass;
    ASSERT_EQ(GEStreamAllocationSummaryIsAssignedByStreamPass(*compiled_summary, graph_to_is_assigned_by_stream_pass),
              SUCCESS);
    EXPECT_EQ(graph_to_is_assigned_by_stream_pass.begin()->second.size(), 3U);
    EXPECT_EQ(graph_to_is_assigned_by_stream_pass.begin()->second[0], false);

    std::map<AscendString, std::vector<std::vector<int64_t>>> graph_to_attached_stream_ids;
    ASSERT_EQ(GEStreamAllocationSummaryGetAttachedStreamIds(*compiled_summary, graph_to_attached_stream_ids), SUCCESS);
    EXPECT_EQ(graph_to_attached_stream_ids.begin()->second.size(), 3U);
    EXPECT_EQ(graph_to_attached_stream_ids.begin()->second[0], expect_attached_stream_ids);

    std::map<AscendString, std::vector<int64_t>> graph_to_physical_stream_nums;
    ASSERT_EQ(GEStreamAllocationSummaryGetPhysicalStreamNums(*compiled_summary, graph_to_physical_stream_nums),
              SUCCESS);
    EXPECT_EQ(graph_to_physical_stream_nums.begin()->second.size(), 3U);
    EXPECT_EQ(graph_to_physical_stream_nums.begin()->second[0], 2U);

    std::map<AscendString, std::vector<int64_t>> graph_to_hccl_followed_stream_nums;
    ASSERT_EQ(GEStreamAllocationSummaryGetHcclFollowedStreamNums(*compiled_summary, graph_to_hccl_followed_stream_nums),
              SUCCESS);
    EXPECT_EQ(graph_to_hccl_followed_stream_nums.begin()->second.size(), 3U);
    EXPECT_EQ(graph_to_hccl_followed_stream_nums.begin()->second[0], 4U);

    std::map<AscendString, std::vector<std::vector<GNode>>> graph_to_all_nodes;
    ASSERT_EQ(GEStreamAllocationSummaryGetAllNodes(*compiled_summary, graph_to_all_nodes), SUCCESS);
    EXPECT_EQ(graph_to_all_nodes.begin()->second.size(), 3U);
    EXPECT_NE(graph_to_all_nodes.begin()->second[0].size(), 0U);

    std::map<AscendString, AscendString> graph_to_stream_graphs;
    ASSERT_EQ(GEStreamAllocationSummaryGetStreamGraphs(*compiled_summary, graph_to_stream_graphs), SUCCESS);
    EXPECT_NE(graph_to_stream_graphs.begin()->second.GetLength(), 0U);

    size_t stream_num = 0U;
    EXPECT_EQ(compiled_summary->GetStreamNum(stream_num), SUCCESS);
    EXPECT_EQ(stream_num, 6);
  };
}

TEST_F(STEST_stream_allocator, AttachedResourceAssignWithOpenSource) {
  auto graph = BuildGraphWithAttachedResourcesWithOpenSource();
  // new session & add graph
  map<string, string> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    // check 不同的通信域的从流不复用
    auto mc2_0 = graph->FindNode("mc2_0");
    EXPECT_NE(mc2_0, nullptr);
    auto mc2_1 = graph->FindNode("mc2_1");
    EXPECT_NE(mc2_1, nullptr);
    auto mc2_2 = graph->FindNode("mc2_2");
    EXPECT_NE(mc2_2, nullptr);
    auto mc2_3 = graph->FindNode("mc2_3");
    EXPECT_NE(mc2_3, nullptr);
    auto mc2_4 = graph->FindNode("mc2_4");
    EXPECT_NE(mc2_4, nullptr);
    auto mc2_5 = graph->FindNode("mc2_5");
    EXPECT_NE(mc2_5, nullptr);
    auto mc2_6 = graph->FindNode("mc2_6");
    EXPECT_NE(mc2_6, nullptr);
    auto mc2_7 = graph->FindNode("mc2_7");
    EXPECT_NE(mc2_7, nullptr);

    // 通信域g0内节点按照attached key分配了不同资源
    EXPECT_EQ(mc2_0->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
    EXPECT_EQ(mc2_1->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
    EXPECT_EQ(mc2_0->GetOpDescBarePtr()->GetAttachedStreamIds()[0], mc2_1->GetOpDescBarePtr()->GetAttachedStreamIds()[0]);

    EXPECT_EQ(mc2_2->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
    EXPECT_NE(mc2_2->GetOpDescBarePtr()->GetAttachedStreamIds()[0], -1);
    EXPECT_NE(mc2_2->GetOpDescBarePtr()->GetAttachedStreamIds()[0], mc2_1->GetOpDescBarePtr()->GetAttachedStreamIds()[0]);
    EXPECT_EQ(mc2_1->GetOutControlNodes().size(), 1U);
    EXPECT_EQ(mc2_3->GetOutControlNodes().size(), 1U);
    EXPECT_EQ(mc2_5->GetOutControlNodes().size(), 1U);

    // 每个附属流的last节点都跟模型的输出节点建立了同步关系
    const auto mc2_1_out = mc2_1->GetOutControlNodes().at(0U);
    const auto mc2_3_out = mc2_3->GetOutControlNodes().at(0U);
    const auto mc2_5_out = mc2_5->GetOutControlNodes().at(0U);
    EXPECT_EQ(mc2_1_out->GetType(), SEND);
    EXPECT_EQ(mc2_3_out->GetType(), SEND);
    EXPECT_EQ(mc2_5_out->GetType(), SEND);
    // send节点跟mc2的附属流同流
    EXPECT_EQ(mc2_1->GetOpDesc()->GetAttachedStreamIds()[0], mc2_1_out->GetOpDesc()->GetStreamId());
    EXPECT_EQ(mc2_3->GetOpDesc()->GetAttachedStreamIds()[0], mc2_3_out->GetOpDesc()->GetStreamId());
    EXPECT_EQ(mc2_5->GetOpDesc()->GetAttachedStreamIds()[0], mc2_5_out->GetOpDesc()->GetStreamId());
    // recv节点跟netoutput同流
    auto output_node = graph->FindNode("output0");
    EXPECT_NE(output_node, nullptr);
    EXPECT_EQ(output_node->GetInControlNodesSize(), 4U);
    EXPECT_EQ(output_node->GetInControlNodes().at(0U)->GetType(), RECV);
    EXPECT_EQ(output_node->GetInControlNodes().at(0U)->GetOpDesc()->GetStreamId(),
              output_node->GetOpDesc()->GetStreamId());
    EXPECT_EQ(output_node->GetInControlNodes().at(1U)->GetType(), RECV);
    EXPECT_EQ(output_node->GetInControlNodes().at(1U)->GetOpDesc()->GetStreamId(),
              output_node->GetOpDesc()->GetStreamId());
    EXPECT_EQ(output_node->GetInControlNodes().at(2U)->GetType(), RECV);
    EXPECT_EQ(output_node->GetInControlNodes().at(2U)->GetOpDesc()->GetStreamId(),
              output_node->GetOpDesc()->GetStreamId());
  };
}

TEST_F(STEST_stream_allocator, DifferentMainStreamAssignSameAttachedStream) {
  auto graph = BuildGraphWithDifferentMainStreamWithSameAttachedStream();
  // new session & add graph
  map<string, string> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    auto mc2_0 = graph->FindNode("mc2_0");
    EXPECT_NE(mc2_0, nullptr);
    auto mc2_1 = graph->FindNode("mc2_1");
    EXPECT_NE(mc2_1, nullptr);
    EXPECT_NE(mc2_0->GetOpDesc()->GetStreamId(), mc2_1->GetOpDesc()->GetStreamId());
    EXPECT_EQ(mc2_0->GetOpDesc()->GetAttachedStreamIds(), mc2_1->GetOpDesc()->GetAttachedStreamIds());
  };
}

TEST_F(STEST_stream_allocator, HcclSplitAttachedStream) {
  auto graph = BuildGraphHcclSplitAttachedStream();
  // new session & add graph
  map<string, string> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    auto mc2_0 = graph->FindNode("mc2_0");
    EXPECT_NE(mc2_0, nullptr);
    auto mc2_1 = graph->FindNode("mc2_1");
    EXPECT_NE(mc2_1, nullptr);
    EXPECT_NE(mc2_0->GetOpDesc()->GetStreamId(), mc2_1->GetOpDesc()->GetStreamId());
    EXPECT_NE(mc2_0->GetOpDesc()->GetAttachedStreamIds(), mc2_1->GetOpDesc()->GetAttachedStreamIds());
  };
}

TEST_F(STEST_stream_allocator, ParallelGroupPass_Success_EnableMdeTopo) {
  auto graph = BuildPartitionedCallGraph();
  auto ge_graph = GraphUtilsEx::GetComputeGraph(graph);
  int64_t new_stream_id = 0;
  std::map<ComputeGraphPtr, int64_t> graph_to_stream_id;
  for (const auto &node : ge_graph->GetAllNodes()) {
    const auto &op = node->GetOpDesc();
    if (graph_to_stream_id.find(node->GetOwnerComputeGraph()) == graph_to_stream_id.cend()) {
      graph_to_stream_id[node->GetOwnerComputeGraph()] = new_stream_id++;
    }
    EXPECT_EQ(AttrUtils::SetInt(op, "NewStreamId", new_stream_id), true);
    if (op->HasAttr(ATTR_NAME_PARALLEL_GROUP)) {
      EXPECT_EQ(op->DelAttr(ATTR_NAME_PARALLEL_GROUP), GRAPH_SUCCESS);
    }
  }
  // new session & add graph
  map<string, string> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    ASSERT_EQ(graph->GetAllSubgraphs().size(), 1);
    ComputeGraphPtr subgraph_res = graph->GetAllSubgraphs().at(0);

    auto relu0 = subgraph_res->FindFirstNodeMatchType(RELU);
    auto hcom1 = subgraph_res->FindFirstNodeMatchType(HCOMALLREDUCE);
    ASSERT_NE(relu0, nullptr);
    ASSERT_NE(hcom1, nullptr);
    EXPECT_EQ(relu0->GetOpDesc()->GetStreamId(), 0);
    EXPECT_EQ(hcom1->GetOpDesc()->GetStreamId(), 1);
  };
}

TEST_F(STEST_stream_allocator, task_exceed_limit_split_stream_succ) {
  auto graph = BuildGraphWithBigSqeNum();
  map<string, string> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto relu1 = graph->FindNode("relu1");
    ASSERT_NE(relu1, nullptr);
    EXPECT_EQ(relu1->GetOpDesc()->GetStreamId(), 0);
    auto relu2 = graph->FindNode("relu2");
    ASSERT_NE(relu2, nullptr);
    EXPECT_EQ(relu2->GetOpDesc()->GetStreamId(), 0);
    auto relu3 = graph->FindNode("relu3");
    ASSERT_NE(relu3, nullptr);
    EXPECT_EQ(relu3->GetOpDesc()->GetStreamId(), 1);
  };
}

TEST_F(STEST_stream_allocator, rts_not_split_stream_success) {
  mmSetEnv("SET_CAPA_VALUE", "stream_unlimited_depth", 1);
  auto graph = BuildGraphWithBigSqeNum();
  map<string, string> options;
  Session session(options);
  auto ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto relu1 = graph->FindNode("relu1");
    ASSERT_NE(relu1, nullptr);
    EXPECT_EQ(relu1->GetOpDesc()->GetStreamId(), 0);
    auto relu2 = graph->FindNode("relu2");
    ASSERT_NE(relu2, nullptr);
    EXPECT_EQ(relu2->GetOpDesc()->GetStreamId(), 0);
    auto relu3 = graph->FindNode("relu3");
    ASSERT_NE(relu3, nullptr);
    EXPECT_EQ(relu3->GetOpDesc()->GetStreamId(), 0);
  };
  mmSetEnv("SET_CAPA_VALUE", "", 1);
}

TEST_F(STEST_stream_allocator, single_stream_with_partitionedcall) {
  auto ge_env = GeRunningEnvFaker();
  ge_env.Reset()
      .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AiCoreLib"))
      .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("DNN_VM_RTS_OP_STORE"))
      .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp("RefData").InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(RESHAPE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(STREAMACTIVE).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(RELU).InfoStoreAndBuilder("AiCoreLib"))
      .Install(FakeOp(CAST).InfoStoreAndBuilder("AiCoreLib"));
  auto graph = MakeGraphWithPartitionedCall();

  // new session & add graph
  map<string, string> options;
  options["ge.enableSingleStream"] = "true";
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  Session session(options);
  ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    bool has_stream_active = false;
    for (const auto &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
      auto op_desc = node->GetOpDesc();
      ASSERT_EQ(op_desc->GetStreamId(), 0);
      if (node->GetType() == ge::STREAMACTIVE) {
        has_stream_active = true;
        std::vector<int64_t> stream_ids;
        (void) AttrUtils::GetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, stream_ids);
        for (auto stream_id : stream_ids) {
          ASSERT_NE(stream_id, op_desc->GetStreamId());
        }
      }
    }
    EXPECT_TRUE(has_stream_active);
  };
  ge_env.Reset();
  ge_env.InstallDefault();
}
/**
 *          data
 *         /    \
 *    trans1    cast
 *  (test_label)  |
 *        \      /
 *        netoutput
 * 测试场景：模拟构图指定stream label
 * 校验点：
 *    （1）最大的stream id是1
 *     (2)两个transdata stream id不同
 */
TEST_F(STEST_stream_allocator, user_defined_stream_label_success) {
  GeRunningEnvFaker().InstallDefault();
  auto graph = MultiStreamShareGraph::TwoNodeGraphWithUserStreamLabel();
  // new session & add graph
  map<string, string> options;
  options["ge.enableSingleStream"] = "false";
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  Session session(options);
  ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    int64_t stream_id = 0;
    int64_t trans1_stream_id = 0;
    int64_t trans2_stream_id = 0;
    for (const auto node : graph->GetDirectNodePtr()) {
      auto op_desc = node->GetOpDesc();
      if (op_desc->GetName() == "trans1") {
        trans1_stream_id = op_desc->GetStreamId();
      }
      if (op_desc->GetName() == "cast") {
        trans2_stream_id = op_desc->GetStreamId();
      }
      stream_id = std::max(op_desc->GetStreamId(), stream_id);
    }
    EXPECT_EQ(stream_id, 1);
    EXPECT_NE(trans1_stream_id, trans2_stream_id);
  };
}

/**
  *  *          data
   *         /    \
   *    trans1    cast
   *  (test_label)  |
   *        \      /
   *        netoutput
   * 测试场景：构图指定stream label,同时开启单流选项
   * 校验点：
   *    （1）编译接口失败
   *     (2) 打屏错误码提示两个配置冲突
 */
TEST_F(STEST_stream_allocator, UserDefinedStreamLabel_with_SingleStreamOption_failed) {
  error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE);
  auto graph = MultiStreamShareGraph::TwoNodeGraphWithUserStreamLabel();
  // new session & add graph
  map<string, string> options;
  options["ge.enableSingleStream"] = "true";
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  Session session(options);
  ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, FAILED); // Stream labels are not supported in SingleStream mode
  // Check ErrorMsg Print in Screen
  auto error_msg = std::string(error_message::GetErrMgrErrorMessage().get());
  EXPECT_TRUE(error_msg.find("Stream labels are not supported in SingleStream mode") != std::string::npos);
}
}  // namespace ge

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "es_ge_test_ops.h"
#include "framework/common/types.h"
#include "jit_share_graph.h"

#include "graph/op_kernel_bin.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils_ex.h"
namespace ge {
namespace {
void AddCompileResultByGNode(ComputeGraphPtr &cg, ge::GNode *gnode, bool atomic, const char *compile_info_json) {
  if (gnode == nullptr || cg == nullptr) {
    return;
  }
  AscendString node_name;
  gnode->GetName(node_name);
  auto node = cg->FindNode(node_name.GetString());
  if (node != nullptr) {
    JitShareGraph::AddCompileResult(node, atomic, compile_info_json);
  }
}
} // namespace

void JitShareGraph::AddCompileResult(const ge::NodePtr &node, bool atomic, const char *compile_info_json) {
  AttrUtils::SetStr(node->GetOpDesc(), "compile_info_json", compile_info_json);
  AttrUtils::SetInt(node->GetOpDesc(), "op_para_size", 2048);
  auto bin = std::make_shared<OpKernelBin>("name", std::vector<char>({'F', 'a', 'k', 'e', 'b', 'i', 'n'}));
  node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, bin);
  AttrUtils::SetStr(node->GetOpDesc(), TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  AttrUtils::SetStr(node->GetOpDesc(), TVM_ATTR_NAME_METADATA, "FakeMeta");
  AttrUtils::SetStr(node->GetOpDesc(), node->GetName() + "_kernelname", "FakeKernelName");
  AttrUtils::SetStr(node->GetOpDesc(), ATTR_NAME_KERNEL_BIN_ID, "te_fake_node_123");
  node->GetOpDesc()->SetWorkspaceBytes({20});
  AttrUtils::SetStr(node->GetOpDesc(), "_kernel_bin_id", "te_" + node->GetType() + "_12345");
  if (atomic) {
    AttrUtils::SetStr(node->GetOpDesc(), "_atomic_compile_info_json", "{}");
    AttrUtils::SetInt(node->GetOpDesc(), "atomic_op_para_size", 2048);

    auto atomic_bin = std::make_shared<OpKernelBin>(
        "name", std::vector<char>({'F', 'a', 'k', 'e', 'A', 't', 'o', 'm', 'i', 'c', 'B', 'i', 'n'}));
    node->GetOpDesc()->SetExtAttr(EXT_ATTR_ATOMIC_TBE_KERNEL, atomic_bin);
    AttrUtils::SetStr(node->GetOpDesc(), ATOMIC_ATTR_TVM_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
    AttrUtils::SetStr(node->GetOpDesc(), ATOMIC_ATTR_TVM_METADATA, "FakeAtomicMeta");
    AttrUtils::SetStr(node->GetOpDesc(), "_memset_kernel_bin_id", "te_fake_atomic_1234");
    AttrUtils::SetStr(node->GetOpDesc(), node->GetName() + "_atomic_kernelname", "FakeAtomicKernelName");
  }
}
/**
 *      data
 *        |
 *       relu
 *        |
 *       relu1
 *        |
 *    netoutput
 **/
UniqueGraphPtr JitShareGraph::AllNormalNodes(const std::vector<int64_t> &input_dims) {
  es::EsGraphBuilder es_graph("test_graph");
  auto data = es_graph.CreateInput(0, "data", DATA);
  if (input_dims.size() == 0) {
    data.SetShape({-1, -1, -1, -1});
  } else {
    data.SetShape(input_dims);
  }
  auto relu = es::Relu(data);
  auto relu1 = es::Relu(relu);
  es::EsGraphBuilder::SetOutput(relu1, 0);
  auto graph = es_graph.BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  AddCompileResultByGNode(cg, relu.GetProducer(), true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  AddCompileResultByGNode(cg, relu1.GetProducer(), true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  return graph;
}

/**
 *      data
 *        |
 *       relu
 *        |
 *       relu1
 *        |
 *    netoutput
 **/
UniqueGraphPtr JitShareGraph::AllNormalNodesStaticShape() {
  es::EsGraphBuilder es_graph("test_graph");
  auto data = es_graph.CreateInput(0, "data", DATA);
  data.SetShape({2, 3, 3, 2});
  auto relu = es::Relu(data);
  auto relu1 = es::Relu(relu);
  es::EsGraphBuilder::SetOutput(relu1, 0);
  auto graph = es_graph.BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  AddCompileResultByGNode(cg, relu.GetProducer(), true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  AddCompileResultByGNode(cg, relu1.GetProducer(), true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  return graph;
}
/**
 *      data
 *        |
 *       relu
 *        |
 *       unique
 *        |
 *       relu1
 *        |
 *    netoutput
 **/
UniqueGraphPtr JitShareGraph::OneUniqueNode() {
  // Note: es::Unique is not generated in es_ge_test, using Relu as placeholder
  es::EsGraphBuilder es_graph("test_graph");
  auto data = es_graph.CreateInput(0, "data", DATA);
  data.SetShape({-1, -1, -1, -1});
  auto relu = es::Relu(data);
  auto relu1 = es::Relu(relu);  // placeholder for Unique
  auto relu2 = es::Relu(relu1);
  es::EsGraphBuilder::SetOutput(relu2, 0);
  return es_graph.BuildAndReset();
}

UniqueGraphPtr JitShareGraph::OneReshapeNodeWithHostInput(const std::vector<int64_t> &input1_dims,
                                                          const std::vector<int64_t> &input2_dims,
                                                          const std::vector<int64_t> &input3_dims) {
  es::EsGraphBuilder es_graph("test_graph");
  auto data = es_graph.CreateInput(0, "data0", nullptr);
  auto data1 = es_graph.CreateInput(1, "data1", nullptr);
  auto data2 = es_graph.CreateInput(2, "data2", nullptr);
  if (input1_dims.size() == 0) {
    data.SetShape({-1, -1, -1, -1});
  } else {
    data.SetShape(input1_dims);
  }

  if (input2_dims.size() == 0) {
    data1.SetShape({-1, -1, -1, -1});
  } else {
    data1.SetShape(input1_dims);
  }

  if (input3_dims.size() == 0) {
    data2.SetShape({-1});
  } else {
    data2.SetShape(input2_dims);
  }
  auto add = es::Add(data, data1);
  auto reshape = es::Reshape(add, data2, 4, 4);
  auto relu1 = es::Relu(reshape);
  es::EsGraphBuilder::SetOutput(relu1, 0);
  auto graph = es_graph.BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  AddCompileResultByGNode(cg, relu1.GetProducer(), true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  return graph;
}
/**
 *      data0
 *        |
 *       relu data1
 *        |     |
 *        reshape
 *        |
 *       relu1
 *        |
 *    netoutput
 **/
UniqueGraphPtr JitShareGraph::OneReshapeNode(const std::vector<int64_t> &input1_dims, const std::vector<int64_t> &input2_dims) {
  es::EsGraphBuilder es_graph("test_graph");
  auto data = es_graph.CreateInput(0, "data0", nullptr);
  auto data1 = es_graph.CreateInput(1, "data1", nullptr);
  if (input1_dims.size() == 0) {
    data.SetShape({-1, -1, -1, -1});
  } else {
    data.SetShape(input1_dims);
  }
  if (input2_dims.size() == 0) {
    data1.SetShape({-1});
  } else {
    data1.SetShape(input2_dims);
  }
  auto relu = es::Relu(data);
  auto reshape = es::Reshape(relu, data1, 4, 4);
  auto relu1 = es::Relu(reshape);
  es::EsGraphBuilder::SetOutput(relu1, 0);
  auto graph = es_graph.BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  AddCompileResultByGNode(cg, relu.GetProducer(), true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  AddCompileResultByGNode(cg, relu1.GetProducer(), true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  return graph;
}
/**
 *      data0
 *        |
 *       relu data1  data2
 *        |     |      |
 *        reshape    relu1
 *            \       /
 *           netoutput
 **/
UniqueGraphPtr JitShareGraph::OneReshapeNodeTwoRelu() {
  es::EsGraphBuilder es_graph("test_graph");
  auto data = es_graph.CreateInput(0, "data0", nullptr);
  auto data1 = es_graph.CreateInput(1, "data1", nullptr);
  auto data2 = es_graph.CreateInput(2, "data2", nullptr);
  data.SetShape({-1, -1, -1, -1});
  data1.SetShape({-1});
  data2.SetShape({-1, -1, -1});
  auto relu = es::Relu(data);
  auto reshape = es::Reshape(relu, data1, 4, 4);
  auto relu1 = es::Relu(data2);
  es::EsGraphBuilder::SetOutput(reshape, 0);
  es::EsGraphBuilder::SetOutput(relu1, 1);
  auto graph = es_graph.BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  AddCompileResultByGNode(cg, relu.GetProducer(), true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  AddCompileResultByGNode(cg, relu1.GetProducer(), true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  return graph;
}
/**
 *      data0
 *        |
 *       relu     data1
 *        |     /    |
 *        reshape0   |
 *              \    |
 *              reshape1
 *                 |
 *               relu1
 *                /
 *        netoutput
 **/
UniqueGraphPtr JitShareGraph::TwoReshapeNodeTwoRelu() {
  es::EsGraphBuilder es_graph("test_graph");
  auto data0 = es_graph.CreateInput(0, "data0", nullptr);
  auto data1 = es_graph.CreateInput(1, "data1", nullptr);
  data0.SetShape({-1, -1, -1, -1});
  data1.SetShape({-1});
  auto relu = es::Relu(data0);
  auto reshape = es::Reshape(relu, data1, 4, 4);
  auto reshape1 = es::Reshape(reshape, data1, 4, 4);
  auto relu1 = es::Relu(reshape1);
  es::EsGraphBuilder::SetOutput(relu1, 0);
  auto graph = es_graph.BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  AddCompileResultByGNode(cg, relu.GetProducer(), true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  AddCompileResultByGNode(cg, relu1.GetProducer(), true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  return graph;
}

/*
 * 有三类算子unique的图，在unique算子切分
 *      data  data
 *        \   /
 *         add
 *          |
 *        unique
 *          |
 *      netoutput
 */
UniqueGraphPtr JitShareGraph::AddUniqueNode() {
  // Note: es::Unique is not generated in es_ge_test, using Relu as placeholder
  es::EsGraphBuilder es_graph("test_graph");
  auto data0 = es_graph.CreateInput(0, "data0", DATA);
  data0.SetShape({-1, -1, -1, -1});
  auto data1 = es_graph.CreateInput(1, "data1", DATA);
  data1.SetShape({-1, -1, -1, -1});
  auto add = es::Add(data0, data1);
  auto relu = es::Relu(add);  // placeholder for Unique
  auto add2 = es::Add(relu, relu);
  es::EsGraphBuilder::SetOutput(add2, 0);
  return es_graph.BuildAndReset();
}

/**
 *      data0   data1
 *         \     /
 *           add
 *            |
 *        netoutput
 **/
UniqueGraphPtr JitShareGraph::OneAddNode() {
  es::EsGraphBuilder es_graph("test_graph");
  auto data0 = es_graph.CreateInput(0, "data0", nullptr);
  auto data1 = es_graph.CreateInput(1, "data1", nullptr);
  data0.SetShape({-1, -1, -1, -1});
  data1.SetShape({-1, -1, -1, -1});
  auto add = es::Add(data0, data1);
  es::EsGraphBuilder::SetOutput(add, 0);
  auto graph = es_graph.BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  AddCompileResultByGNode(cg, add.GetProducer(), true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  AddCompileResultByGNode(cg, add.GetProducer(), true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  return graph;
}

/**
 *      data0
 *        |
 *       relu data1
 *        |     |
 *        reshape0  const
 *              \    |
 *              reshape1
 *                 |
 *               relu1
 *                /
 *        netoutput
 **/
UniqueGraphPtr JitShareGraph::OneConstTwoReshapeNodeTwoRelu() {
  es::EsGraphBuilder es_graph("test_graph");
  auto data0 = es_graph.CreateInput(0, "data0", nullptr);
  auto data1 = es_graph.CreateInput(1, "data1", nullptr);
  data0.SetShape({-1, -1, -1, -1});
  data1.SetShape({-1});
  auto relu = es::Relu(data0);
  auto reshape = es::Reshape(relu, data1, 4, 4);
  std::vector<int64_t> const_data0 = {2, 3, 3, 2};
  std::vector<int64_t> const_dim = {4};
  es::EsTensorHolder const_node(EsCreateConstInt64(es_graph.GetCGraphBuilder(), const_data0.data(), const_dim.data(), const_dim.size()));
  auto reshape1 = es::Reshape(reshape, const_node, 4, 4);
  auto relu1 = es::Relu(reshape1);
  es::EsGraphBuilder::SetOutput(relu1, 0);
  auto graph = es_graph.BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  AddCompileResultByGNode(cg, relu.GetProducer(), true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  AddCompileResultByGNode(cg, relu1.GetProducer(), true,
                   "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
                   "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
  return graph;
}
} // ge
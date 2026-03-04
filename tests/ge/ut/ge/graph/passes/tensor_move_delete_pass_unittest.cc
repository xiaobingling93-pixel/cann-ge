/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include <gtest/gtest.h>

#include "macro_utils/dt_public_scope.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/passes/base_pass.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_builder_utils.h"
#include "ge_local_context.h"
#include "graph/passes/standard_optimize/tensor_move_delete_pass.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg_box.h"
#include "easy_graph/builder/graph_dsl.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/operator_reg.h"
#include "external/ge_common/ge_api_types.h"
#include "stub/gert_runtime_stub.h"

using namespace std;
using namespace testing;
using namespace ge;

namespace {
REG_OP(Cast)
.INPUT(x, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8, DT_INT64,
                      DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128,
                      DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32})) /* input tensor */
.OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8, DT_INT64,
                       DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128,
                       DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32})) /* output tensor */
.ATTR(dst_type, Int, 0)
.ATTR(truncate, Bool, false)
.OP_END_FACTORY_REG(Cast)

bool SetTransDataTensorDesc(const ComputeGraphPtr &root_graph, const std::vector<std::string> &node_names, Format format = FORMAT_NCL) {
  GeTensorDesc tensor_desc{GeShape{{2022, 2023}}, format, DT_FLOAT16};
  std::map<std::string, NodePtr> all_transdata_map;
  for (auto &node : root_graph->GetAllNodes()) {
    if (node->GetType() == TRANSDATA) {
      all_transdata_map[node->GetName()] = node;
    }
  }
  for (const auto &node_name : node_names) {
    const auto iter = all_transdata_map.find(node_name);
    if (iter != all_transdata_map.end()) {
      iter->second->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
    } else {
      std::cout << "========================================" << std::endl;
      std::cout << "can not find " << node_name << std::endl;
      std::cout << "========================================" << std::endl;
      return false;
    }
  }
  return true;
}

using NetoutputParentIndexes = std::vector<std::pair<std::string, std::vector<uint32_t>>>;
bool AddParentIndexForNetoutput(ComputeGraphPtr &root_graph, NetoutputParentIndexes &indexes) {
  std::map<std::string, NodePtr> netoutput_map;
  for (auto &node : root_graph->GetAllNodes()) {
    netoutput_map[node->GetName()] = node;
  }
  for (auto &name_indexes_pair : indexes) {
    const auto iter = netoutput_map.find(name_indexes_pair.first);
    if (iter == netoutput_map.end()) {
      std::cout << "========================================" << std::endl;
      std::cout << "can not find " << name_indexes_pair.first << std::endl;
      std::cout << "========================================" << std::endl;
      return false;
    }
    auto op_desc = iter->second->GetOpDesc();
    size_t input_index = 0U;
    if (name_indexes_pair.second.size() != op_desc->GetInputsSize()) {
      std::cout << "========================================" << std::endl;
      std::cout << name_indexes_pair.first << " real inputs size: " << op_desc->GetInputsSize()
                << ", but name_indexes_pair.second.size(): " << name_indexes_pair.second.size() << std::endl;
      std::cout << "========================================" << std::endl;
      return false;
    }
    for (auto parent_index : name_indexes_pair.second) {
      auto tensor_desc = op_desc->MutableInputDesc(input_index++);
      AttrUtils::SetInt(tensor_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index);
    }
  }
  return true;
}

void SetRefOutput(const NodePtr &node, const uint32_t output_idx = 0U, const int32_t input_idx = 0) {
  auto out_desc = node->GetOpDescBarePtr()->MutableOutputDesc(output_idx);
  ge::TensorUtils::SetReuseInput(*out_desc, true);
  ge::TensorUtils::SetReuseInputIndex(*out_desc, input_idx);
}
}

class UtestTensorMoveDeletePass : public Test {
  protected:
  void SetUp() {
    dlog_setlevel(0, 0, 0);
  }

  void TearDown() {
    dlog_setlevel(0, 3, 0);
    unsetenv("DUMP_GRAPH_LEVEL");
    unsetenv("DUMP_GE_GRAPH");
    GetThreadLocalContext().SetGraphOption({});
  }
};

/**
 *       Relu
 *        |
 *     TensorMove
 *        |
 *     NetOutput
 *
 * 说明：
 * - TensorMove 前后均为普通节点
 * - Relu无分支
 *
 * 预期：
 * - 删除 TensorMove，Relu 直连 NetOutput
 */
TEST_F(UtestTensorMoveDeletePass, TensorMoveFromComputeNodeToNetOutput_Deleted) {
  setenv("DUMP_GRAPH_LEVEL", "2", 1);
  setenv("DUMP_GE_GRAPH", "2", 1);
  dlog_setlevel(0, 0, 0);
  auto builder = ut::GraphBuilder("g1");
  auto relu_node = builder.AddNode("Relu", RELU, 1, 1);
  auto node1 = builder.AddNode("TensorMove", TENSORMOVE, 1, 1);
  auto node2 = builder.AddNode("NetOutput", NETOUTPUT, 1, 1);

  GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  EXPECT_EQ(node1->GetOutDataNodes().size(), 1);

  ge::GEPass pass(builder.GetGraph());
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(builder.GetGraph()->FindNode("TensorMove"), nullptr);
}

/**
 *       Data
 *        |
 *     TensorMove
 *        |
 *     ReShape1 (refop)
 *        |
 *     ReShape2 (refop)
 *        |
 *     NetOutput
 *
 * 说明：
 * - TensorMove 位于 Data 与 reshape之间
 * - 开启输出复用输入内存
 *
 * 预期：
 * - 删除 TensorMove，Data 直连 ReShape1
 */
TEST_F(UtestTensorMoveDeletePass, TensorMoveFromDataToNetOutput_ThroughRefOps_Deleted) {
  setenv("DUMP_GRAPH_LEVEL", "2", 1);
  setenv("DUMP_GE_GRAPH", "2", 1);
  dlog_setlevel(0, 0, 0);
  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "1,1|0,0";
  ge::GetThreadLocalContext().SetGraphOption(options);

  auto builder = ut::GraphBuilder("g1");
  auto relu_node = builder.AddNode("Data", DATA, 1, 1);
  auto node1 = builder.AddNode("TensorMove", TENSORMOVE, 1, 1);
  auto node2 = builder.AddNode("ReShape1", RESHAPE, 1, 1);
  auto node22 = builder.AddNode("ReShape2", RESHAPE, 1, 1);
  auto node3 = builder.AddNode("NetOutput", NETOUTPUT, 1, 1);

  AttrUtils::SetInt(relu_node->GetOpDesc(), ATTR_NAME_INDEX, 0);

  GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node22->GetInDataAnchor(0));
  GraphUtils::AddEdge(node22->GetOutDataAnchor(0), node3->GetInDataAnchor(0));

  EXPECT_EQ(node1->GetOutDataNodes().size(), 1);

  ge::GEPass pass(builder.GetGraph());
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(builder.GetGraph()->FindNode("TensorMove"), nullptr);

  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 *       Data
 *        |
 *     ReShape (refop)
 *        |
 *     TensorMove
 *        |
 *     NetOutput(复用内存)
 *
 * 说明：
 * - TensorMove 位于 ReShape 与 NetOutput 之间，实际输入为根图Data
 * - 单输入单输出，无分支，复用内存
 *
 * 预期：
 * - 删除 TensorMove，ReShape 直连 NetOutput
 */

TEST_F(UtestTensorMoveDeletePass, TensorMoveFromDataToNetOutput_ThroughSingleRefOp_Deleted) {
  setenv("DUMP_GRAPH_LEVEL", "2", 1);
  setenv("DUMP_GE_GRAPH", "2", 1);
  dlog_setlevel(0, 0, 0);
  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "1,1|0,0";
  ge::GetThreadLocalContext().SetGraphOption(options);

  auto builder = ut::GraphBuilder("g1");
  auto data_node = builder.AddNode("Data", DATA, 1, 1);
  auto tensor_move_node = builder.AddNode("TensorMove", TENSORMOVE, 1, 1);
  auto reshape_node = builder.AddNode("ReShape", RESHAPE, 1, 1);
  auto netoutput_node = builder.AddNode("NetOutput", NETOUTPUT, 1, 1);

  AttrUtils::SetInt(data_node->GetOpDesc(), ATTR_NAME_INDEX, 0);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), reshape_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(reshape_node->GetOutDataAnchor(0), tensor_move_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  EXPECT_EQ(tensor_move_node->GetOutDataNodes().size(), 1);

  ge::GEPass pass(builder.GetGraph());
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(builder.GetGraph()->FindNode("TensorMove"), nullptr);

  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 *       Data
 *        |
 *      Reshape -- Add
 *        |
 *     TensorMove
 *        |
 *     NetOutput
 *
 * 说明：
 * - TensorMove的实际源输入为根图Data，Reshape 的输出同时供给 Add 节点和 TensorMove
 *
 * 预期：
 * - 不删除 TensorMove，图结构不变
 *
 */
TEST_F(UtestTensorMoveDeletePass, TensorMoveFromDataViaRefOp_WithBranch_Kept) {
  setenv("DUMP_GRAPH_LEVEL", "2", 1);
  setenv("DUMP_GE_GRAPH", "2", 1);
  dlog_setlevel(0, 0, 0);
  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "1,1|0,0";
  ge::GetThreadLocalContext().SetGraphOption(options);

  auto builder = ut::GraphBuilder("g1");
  // 创建算子：1个输入，2个输出
  auto ref_node2 = builder.AddNode("ref_node2", ADD, 1, 2);
  auto op_desc = ref_node2->GetOpDescBarePtr();
  auto out_desc_0 = op_desc->MutableOutputDesc(0);
  auto out_desc_1 = op_desc->MutableOutputDesc(1);
  ge::TensorUtils::SetReuseInput(*out_desc_0, true);
  ge::TensorUtils::SetReuseInputIndex(*out_desc_0, 0); // 复用第0个输入
  ge::TensorUtils::SetReuseInput(*out_desc_1, true);
  ge::TensorUtils::SetReuseInputIndex(*out_desc_1, 0); // 同样复用第0个输入
  auto data_node = builder.AddNode("Data", DATA, 1, 1);
  auto tensor_move_node = builder.AddNode("TensorMove", TENSORMOVE, 1, 1);
  auto add_node = builder.AddNode("Add", ADD, 1, 1);
  auto netoutput_node = builder.AddNode("NetOutput", NETOUTPUT, 1, 1);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), ref_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(ref_node2->GetOutDataAnchor(0), tensor_move_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(ref_node2->GetOutDataAnchor(1), add_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  EXPECT_EQ(tensor_move_node->GetOutDataNodes().size(), 1);

  ge::GEPass pass(builder.GetGraph());
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_NE(builder.GetGraph()->FindNode("TensorMove"), nullptr);

  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 *        Data
 *       /   \
 * TensorMove1 Add
 *      |
 *    RefOp
 *      |
 * TensorMove2
 *      |
 *   NetOutput
 *
 * 说明：
 * - Data 被 TensorMove1 和 Add 同时使用，TensorMove1 不能删除
 * - TensorMove2 向上溯源经过 RefOp 后遇到 TensorMove1，停止穿透
 *
 * 预期：
 * - TensorMove1 保留
 * - TensorMove2 删除
 */
TEST_F(UtestTensorMoveDeletePass, TensorMoveTraceStopsAtUpstreamTensorMove_DataBranched_DownstreamDeleted) {
  setenv("DUMP_GRAPH_LEVEL", "2", 1);
  setenv("DUMP_GE_GRAPH", "2", 1);
  dlog_setlevel(0, 0, 0);
  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "1,1|0,0";
  ge::GetThreadLocalContext().SetGraphOption(options);

  auto builder = ut::GraphBuilder("g1");
  auto data_node = builder.AddNode("Data", DATA, 1, 1);
  auto tensor_move1_node = builder.AddNode("TensorMove1", TENSORMOVE, 1, 1);
  auto ref_node = builder.AddNode("RefOp", ADD, 1, 1);
  auto tensor_move2_node = builder.AddNode("TensorMove2", TENSORMOVE, 1, 1);
  auto add_node = builder.AddNode("Add", ADD, 1, 1);
  auto netoutput_node = builder.AddNode("NetOutput", NETOUTPUT, 2, 1);

  SetRefOutput(ref_node, 0, 0);
  AttrUtils::SetInt(data_node->GetOpDesc(), ATTR_NAME_INDEX, 0);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), tensor_move1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move1_node->GetOutDataAnchor(0), ref_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(ref_node->GetOutDataAnchor(0), tensor_move2_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(1));

  ge::GEPass pass(builder.GetGraph());
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  EXPECT_NE(builder.GetGraph()->FindNode("TensorMove1"), nullptr);
  EXPECT_EQ(builder.GetGraph()->FindNode("TensorMove2"), nullptr);

  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 *         Data
 *          |
 *   TensorMove1(reserved)
 *      /         \
 *   RefOp        Add
 *    |            |
 * TensorMove2  NetOutput
 *    |
 * NetOutput
 *
 * 说明：
 * - TensorMove1 通过保留属性禁止删除
 * - TensorMove1 输出有分支（RefOp 与 Add）
 * - TensorMove2 溯源遇到 TensorMove1 停止，随后按单路径规则校验失败
 *
 * 预期：
 * - TensorMove1 保留
 * - TensorMove2 保留
 */
TEST_F(UtestTensorMoveDeletePass, TensorMoveTraceStopsAtReservedUpstreamTensorMove_WithBranch_DownstreamKept) {
  setenv("DUMP_GRAPH_LEVEL", "2", 1);
  setenv("DUMP_GE_GRAPH", "2", 1);
  dlog_setlevel(0, 0, 0);
  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "1,1|0,0";
  ge::GetThreadLocalContext().SetGraphOption(options);

  auto builder = ut::GraphBuilder("g1");
  auto data_node = builder.AddNode("Data", DATA, 1, 1);
  auto tensor_move1_node = builder.AddNode("TensorMove1", TENSORMOVE, 1, 1);
  auto ref_node = builder.AddNode("RefOp", ADD, 1, 1);
  auto tensor_move2_node = builder.AddNode("TensorMove2", TENSORMOVE, 1, 1);
  auto add_node = builder.AddNode("Add", ADD, 1, 1);
  auto netoutput_node = builder.AddNode("NetOutput", NETOUTPUT, 2, 1);

  SetRefOutput(ref_node, 0, 0);
  AttrUtils::SetInt(data_node->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetBool(tensor_move1_node->GetOpDesc(), ATTR_NAME_CANNOT_BE_DELETED, true);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), tensor_move1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move1_node->GetOutDataAnchor(0), ref_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(ref_node->GetOutDataAnchor(0), tensor_move2_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move1_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(1));

  ge::GEPass pass(builder.GetGraph());
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  EXPECT_NE(builder.GetGraph()->FindNode("TensorMove1"), nullptr);
  EXPECT_NE(builder.GetGraph()->FindNode("TensorMove2"), nullptr);

  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 *       Data
 *        |
 *     TensorMove
 *        |
 *     NetOutput
 *
 * 说明：
 * - TensorMove 输入为根图Data，直连NetOutput
 * - 未设置地址复用
 *
 * 预期行为：
 * - 根图Data为外部输入，未声明输出复用输入内存，不删除TensorMove
 */
TEST_F(UtestTensorMoveDeletePass, TensorMoveFromData_NoReuseConfig_Kept) {  // 还能补充一个：配置了option，但是不包含0，0
  setenv("DUMP_GRAPH_LEVEL", "2", 1);
  setenv("DUMP_GE_GRAPH", "2", 1);
  dlog_setlevel(0, 0, 0);
  auto builder = ut::GraphBuilder("g1");
  auto data_node = builder.AddNode("Data", DATA, 1, 1);
  auto node1 = builder.AddNode("TensorMove", TENSORMOVE, 1, 1);
  auto node2 = builder.AddNode("NetOutput", NETOUTPUT, 1, 1);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  EXPECT_EQ(node1->GetOutDataNodes().size(), 1);

  ge::GEPass pass(builder.GetGraph());
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_NE(builder.GetGraph()->FindNode("TensorMove"), nullptr);
}

/**
 *       Data
 *        |
 *     TensorMove
 *        |
 *     NetOutput  (复用输入地址)
 *
 * 说明：
 * - TensorMove 输入为根图Data，直连NetOutput
 * - 设置了输出复用输入内存
 * - 单输入、单输出
 * - 无分支、无子图
 *
 * 预期行为：
 * - 删除TensorMove
 */
TEST_F(UtestTensorMoveDeletePass, TensorMoveFromData_MemReuse_Deleted) {
  setenv("DUMP_GRAPH_LEVEL", "2", 1);
  setenv("DUMP_GE_GRAPH", "2", 1);
  dlog_setlevel(0, 0, 0);
  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "1,1|0,0";
  ge::GetThreadLocalContext().SetGraphOption(options);
  auto builder = ut::GraphBuilder("g1");
  auto data_node = builder.AddNode("Data", DATA, 1, 1);
  auto node1 = builder.AddNode("TensorMove", TENSORMOVE, 1, 1);
  auto node2 = builder.AddNode("NetOutput", NETOUTPUT, 1, 1);

  AttrUtils::SetInt(data_node->GetOpDesc(), ATTR_NAME_INDEX, 0);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  EXPECT_EQ(node1->GetOutDataNodes().size(), 1);

  ge::GEPass pass(builder.GetGraph());
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(builder.GetGraph()->FindNode("TensorMove"), nullptr);

  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 * 原始模型：
 *
 *         Relu
 *        /    \
 *   TensorMove  TransData
 *        |
 *     NetOutput
 *
 * 说明：
 * - TensorMove的输入为Relu，直连NetOutput
 * - Relu还有另一输出连接到 TransData
 *
 * 预期行为：
 * - Relu输出给到多个节点，不删除TensorMove
 */
TEST_F(UtestTensorMoveDeletePass, TensorMoveFromRelu_SourceHasMultiOutputs_Kept) {
  setenv("DUMP_GRAPH_LEVEL", "2", 1);
  setenv("DUMP_GE_GRAPH", "2", 1);
  dlog_setlevel(0, 0, 0);
  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "1,1|0,0";
  ge::GetThreadLocalContext().SetGraphOption(options);
  setenv("DUMP_GRAPH_LEVEL", "2", 1);
  setenv("DUMP_GE_GRAPH", "2", 1);
  dlog_setlevel(0, 0, 0);

  auto builder = ut::GraphBuilder("g1");
    // 创建算子：1个输入，2个输出
  auto ref_node2 = builder.AddNode("ref_node2", ADD, 1, 2);
  auto op_desc = ref_node2->GetOpDescBarePtr();
  auto out_desc_0 = op_desc->MutableOutputDesc(0);
  auto out_desc_1 = op_desc->MutableOutputDesc(1);
  ge::TensorUtils::SetReuseInput(*out_desc_0, true);
  ge::TensorUtils::SetReuseInputIndex(*out_desc_0, 0); // 复用第0个输入
  ge::TensorUtils::SetReuseInput(*out_desc_1, true);
  ge::TensorUtils::SetReuseInputIndex(*out_desc_1, 0); // 同样复用第0个输入
  auto node1 = builder.AddNode("TensorMove", TENSORMOVE, 1, 1);
  auto node2 = builder.AddNode("NetOutput", NETOUTPUT, 1, 1);
  auto node3 = builder.AddNode("Transdata", TRANSDATA, 1, 1);

  GraphUtils::AddEdge(ref_node2->GetOutDataAnchor(0), node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(ref_node2->GetOutDataAnchor(1), node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  EXPECT_EQ(node1->GetOutDataNodes().size(), 1);

  ge::GEPass pass(builder.GetGraph());
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_NE(builder.GetGraph()->FindNode("TensorMove"), nullptr);

  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 * 父图:                子图 sub_1:
 * Data                 sub_Data (ParentIndex: 0)
 * |                      |
 * PartitionedCall ------> TensorMove
 * |                      |
 * NetOutput            sub_NetOutput
 * (复用输入地址)
 *
 * 场景说明：
 * - 子图内部 TensorMove 的前驱是 sub_Data，其在父图的实际源头是 Data。
 * - 设置根图 NetOutput 复用输入内存，触发 TensorMove 优化逻辑。
 *
 * 预期行为：
 * - Trace 能够跨越子图边界识别到 Data 是源头。
 * - TensorMove 被成功识别并删除。
 */
TEST_F(UtestTensorMoveDeletePass, TensorMoveInSubgraph_FromParentData_Deleted) {
  dlog_setlevel(0, 0, 0);

  // 1. 设置内存复用选项：设置根图的第 0 个输出复用第 0 个输入
  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "0,0";
  ge::GetThreadLocalContext().SetGraphOption(options);

  // 2. 构造子图 sub_1
  // sub_Data 的 ParentNodeIndex(0) 代表它对应父图中 PartitionedCall 的第 0 个 Input
  const auto sub_data_cfg = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("sub_data", sub_data_cfg)
          ->NODE("sub_tensormove", TENSORMOVE)
          ->NODE("sub_netoutput", NETOUTPUT));
  };

  // 3. 构造父图 g1
  const auto data_cfg = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", data_cfg)
          ->EDGE(0, 0)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
          ->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
  };

  // 4. 将子图挂载到父图
  auto compute_graph = ToComputeGraph(g1);
  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);

  auto p_call_node = compute_graph->FindNode("partitioned_call");
  ASSERT_NE(p_call_node, nullptr);

  // 设置父子图关联属性
  sub_graph_1->SetParentGraph(compute_graph);
  sub_graph_1->SetParentNode(p_call_node);

  // 5. 设置 NetOutput 的映射关系 (用于 Trace 穿透)
  // 根图 NetOutput 的输入来自 PartitionedCall:0
  // 子图 sub_NetOutput 的输入来自 sub_tensormove
  NetoutputParentIndexes indexes{{"netoutput", {0}},
                                 {"sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  // 6. 执行 Pass
  ge::GEPass pass(compute_graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  // 7. 验证结果：子图内部的 tensormove 应该被删除
  // 注意：FindNode 在子图中查找
  EXPECT_EQ(sub_graph_1->FindNode("sub_tensormove"), nullptr);

  // 清理环境
  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 * 父图:
 * Data -> PartitionedCall -> NetOutput
 *
 * 子图 sub_1:
 * sub_Data(ParentIndex:0) -> sub_tensormove -> sub_NetOutput
 *
 * 场景说明：
 * - sub_tensormove 从子图 Data 跳出到父图时，source path 中会插入 PartitionedCall 节点。
 *
 * 预期行为：
 * - trace 路径日志中包含 partitioned_call 节点；
 * - partitioned_call 仅打印节点名，不打印 out anchor；
 * - sub_tensormove 可被删除（保证路径后续规则可正常处理空 anchor）。
 */
TEST_F(UtestTensorMoveDeletePass, TensorMoveInSubgraph_PartitionedCallSourcePath_LogWithoutOutAnchor) {
  dlog_setlevel(0, 0, 0);

  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "0,0";
  ge::GetThreadLocalContext().SetGraphOption(options);

  const auto sub_data_cfg = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("sub_data", sub_data_cfg)->NODE("sub_tensormove", TENSORMOVE)->NODE("sub_netoutput", NETOUTPUT));
  };

  const auto data_cfg = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", data_cfg)
          ->EDGE(0, 0)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
          ->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
  };

  auto compute_graph = ToComputeGraph(g1);
  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);

  auto p_call_node = compute_graph->FindNode("partitioned_call");
  ASSERT_NE(p_call_node, nullptr);
  sub_graph_1->SetParentGraph(compute_graph);
  sub_graph_1->SetParentNode(p_call_node);

  NetoutputParentIndexes indexes{{"netoutput", {0}},
                                 {"sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevel(DLOG_INFO);
  runtime_stub.GetSlogStub().Clear();

  ge::GEPass pass(compute_graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  EXPECT_EQ(sub_graph_1->FindNode("sub_tensormove"), nullptr);
  EXPECT_NE(runtime_stub.GetSlogStub().FindLog(
      DLOG_INFO, "Trace reach real source: data(out:0)-->partitioned_call-->sub_data(out:0)-->(in:0)sub_tensormove"), -1);
  EXPECT_EQ(runtime_stub.GetSlogStub().FindLogRegex(
      DLOG_INFO, "Trace reach real source: .*partitioned_call\\(out:"), -1);

  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 * 主图：
 *         Data
 *        /    \
 *     Cast    PartitionedCall
 *      |        |
 *   TransData  NetOutput
 *
 * 子图 sub_1：
 *      sub_Data
 *          |
 *    sub_partitioned_call
 *          |
 *     TensorMove
 *          |
 *     sub_NetOutput
 *
 * 子子图 sub_sub_1：
 *       sub_sub_data
 *        /       \
 *     Cast        \
 *      |           \
 *   TransData      Add
 *      \            /
 *      Add        /
 *        \      /
 *      sub_sub_NetOutput
 *
 *
 * 预期行为：
 * - 删除 TensorMove,sub_sub_NetOutput两个输出，一个空悬，一个给到TensorMove，但是任意一个的输入都是计算节点(TransData或Add)
 */
TEST_F(UtestTensorMoveDeletePass, TensorMove_NestedPCall_FromAdd_Deleted) {
  dlog_setlevel(0, 0, 0);

  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "1,1|0,0";
  ge::GetThreadLocalContext().SetGraphOption(options);
  const auto sub_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_sub_1) {
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->NODE("sub_sub_cast", CAST)
                                   ->NODE("sub_sub_transdata", TRANSDATA)
                                   ->NODE("sub_sub_add0", ADD)->NODE("sub_sub_netoutput", NETOUTPUT));
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->EDGE(0, 0)->NODE("sub_sub_add1", ADD)
                                   ->NODE("sub_sub_netoutput", NETOUTPUT));
                   };
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub_data", sub_data)->NODE("sub_partitioned_call", PARTITIONEDCALL, sub_sub_1)
                               ->NODE("sub_tensor_move", TENSORMOVE)
                               ->NODE("sub_netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("cast", CAST)->NODE("transdata", TRANSDATA)
                            ->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data", DATA)
                    ->EDGE(0, 0)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
                    ->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
                };
  auto sub_sub_1_graph = ToComputeGraph(sub_sub_1);
  sub_1.Layout();
  auto compute_graph = ToComputeGraph(g1);
  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);

  auto sub_partitioned_call_node = sub_graph_1->FindNode("sub_partitioned_call");
  ASSERT_NE(sub_partitioned_call_node, nullptr);
  sub_sub_1_graph->SetParentGraph(compute_graph);
  sub_sub_1_graph->SetParentNode(sub_partitioned_call_node);
  compute_graph->AddSubGraph(sub_sub_1_graph);  // 嵌套子图

  const auto sub_sub_graph_1 = compute_graph->GetSubgraph("sub_sub_1");
  ASSERT_NE(sub_sub_graph_1, nullptr);

  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"transdata", "sub_sub_transdata"}));

  NetoutputParentIndexes indexes{{"sub_netoutput", {0}},
                                 {"sub_sub_netoutput", {0, 1}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  ge::GEPass pass(compute_graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_NE(sub_graph_1->FindNode("sub_tensor_move"), nullptr);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(sub_graph_1->FindNode("sub_tensor_move"), nullptr);

  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 *        data
 *          |
 *  PartitionedCall
 *          |
 *      netoutput
 *
 * 子图 sub_1：
 *
 *      sub_data
 *          |
 *  sub_partitioned_call
 *          |
 *       tensormove
 *          |
 *     sub_netoutput
 *
 * 子子图 sub_sub_1：
 *
 *       sub_sub_data
 *        /       \
 *   sub_sub_cast   sub_sub_add1
 *        |           |
 * sub_sub_transdata  |
 *        |           |
 *   sub_sub_add0 -----
 *        |
 *  sub_sub_netoutput
 *
 * 预期结果：
 * tensormove的输入是sub_sub_add0，sub_sub_add0只有一条路径，删除
 */
TEST_F(UtestTensorMoveDeletePass, TensorMoveInSub_FromSubSubAdd_Deleted) {
  dlog_setlevel(0, 0, 0);

  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "1,1|0,0";
  ge::GetThreadLocalContext().SetGraphOption(options);
  const auto sub_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_sub_1) {
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->NODE("sub_sub_cast", CAST)
                                   ->NODE("sub_sub_transdata", TRANSDATA)
                                   ->NODE("sub_sub_add0", ADD)->NODE("sub_sub_netoutput", NETOUTPUT));
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->EDGE(0, 0)->NODE("sub_sub_add1", ADD)
                                   ->NODE("sub_sub_netoutput", NETOUTPUT));
                   };
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub_data", sub_data)->NODE("sub_partitioned_call", PARTITIONEDCALL, sub_sub_1)
                               ->NODE("sub_tensor_move", TENSORMOVE)
                               ->NODE("sub_netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)
                    ->EDGE(0, 0)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
                    ->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
                };
  auto sub_sub_1_graph = ToComputeGraph(sub_sub_1);
  sub_1.Layout();
  auto compute_graph = ToComputeGraph(g1);
  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);

  auto sub_partitioned_call_node = sub_graph_1->FindNode("sub_partitioned_call");
  ASSERT_NE(sub_partitioned_call_node, nullptr);
  sub_sub_1_graph->SetParentGraph(compute_graph);
  sub_sub_1_graph->SetParentNode(sub_partitioned_call_node);
  compute_graph->AddSubGraph(sub_sub_1_graph);  // 嵌套子图


  const auto sub_sub_graph_1 = compute_graph->GetSubgraph("sub_sub_1");
  ASSERT_NE(sub_sub_graph_1, nullptr);

  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"sub_sub_transdata"}));

  NetoutputParentIndexes indexes{{"sub_netoutput", {0}},
                                 {"sub_sub_netoutput", {0, 1}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  ge::GEPass pass(compute_graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_NE(sub_graph_1->FindNode("sub_tensor_move"), nullptr);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(sub_graph_1->FindNode("sub_tensor_move"), nullptr);

  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 * 主图 g1：
 *
 *        data
 *          |
 *   PartitionedCall
 *          |
 *       tensormove
 *          |
 *       netoutput
 *
 * 子图 sub_1：
 *
 *      sub_data
 *          |
 *  sub_partitioned_call
 *          |
 *     sub_tensormove
 *          |
 *     sub_netoutput
 *
 * 子子图 sub_sub_1：
 *
 *       sub_sub_data
 *        /       \
 *   sub_sub_cast   sub_sub_add1
 *        |           |
 * sub_sub_transdata  |
 *        |           |
 *   sub_sub_add0     |
 *        |
 *     sub_sub_netoutput

 *
 * 预期结果：
 * - 主图 tensormove 的真实输入应追溯至子子图的 sub_sub_add0，tensormove 被成功删除；
 * - sub_tensormove 的真实输入应追溯至子子图中的 sub_sub_add0，sub_tensormove 也被成功删除；
 */

TEST_F(UtestTensorMoveDeletePass, TensorMoveInRootAndSub_FromSubSubAdd_Deleted) {
  dlog_setlevel(0, 0, 0);

  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "1,1|0,0";
  ge::GetThreadLocalContext().SetGraphOption(options);
  const auto sub_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_sub_1) {
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->NODE("sub_sub_cast", CAST)
                                   ->NODE("sub_sub_transdata", TRANSDATA)
                                   ->NODE("sub_sub_add0", ADD)->NODE("sub_sub_netoutput", NETOUTPUT));
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->EDGE(0, 0)->NODE("sub_sub_add1", ADD)
                                   ->NODE("sub_sub_netoutput", NETOUTPUT));
                   };
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub_data", sub_data)->NODE("sub_partitioned_call", PARTITIONEDCALL, sub_sub_1)
                               ->NODE("sub_tensormove", TENSORMOVE)
                               ->NODE("sub_netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)
                    ->EDGE(0, 0)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
                    ->EDGE(0, 0)->NODE("tensormove", TENSORMOVE)
                    ->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
                };
  auto sub_sub_1_graph = ToComputeGraph(sub_sub_1);
  sub_1.Layout();
  auto compute_graph = ToComputeGraph(g1);
  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);

  auto sub_partitioned_call_node = sub_graph_1->FindNode("sub_partitioned_call");
  ASSERT_NE(sub_partitioned_call_node, nullptr);
  sub_sub_1_graph->SetParentGraph(compute_graph);
  sub_sub_1_graph->SetParentNode(sub_partitioned_call_node);
  sub_sub_1_graph->SetOutputSize(2);
  compute_graph->AddSubGraph(sub_sub_1_graph);  // 嵌套子图

  const auto sub_sub_graph_1 = compute_graph->GetSubgraph("sub_sub_1");
  ASSERT_NE(sub_sub_graph_1, nullptr);

  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"sub_sub_transdata"}));

  NetoutputParentIndexes indexes{{"sub_netoutput", {0}},
                                 {"sub_sub_netoutput", {0, 1}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  ge::GEPass pass(compute_graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  EXPECT_EQ(compute_graph->FindNode("tensormove"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("sub_tensormove"), nullptr);

  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 * 主图 g1：
 *
 *   PartitionedCall_1
 *          |
 *   PartitionedCall_2
 *          |
 *         relu
 *          |
 *       netoutput
 *
 * 子图 sub_1：
 *     sub_1_constant
 *          |
 *     sub_1_netoutput
 *
 * 子图 sub_2：
 *      sub_2_data
 *          |
 *     sub_2_tensormove
 *          |
 *     sub_2_netoutput
 *
 *
 * 预期结果：
 * - sub_2_tensormove的真实输入应追溯至sub_1_constant，sub_2_tensormove 不能删除；
 */
TEST_F(UtestTensorMoveDeletePass, TensorMoveInSub2_TraceToSub1Const_NotDeleted) {
  dlog_setlevel(0, 0, 0);

  // 1. 定义 sub_1： Const -> NetOutput
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("sub_1_constant", CONSTANT)->NODE("sub_1_netoutput", NETOUTPUT));
  };

  // 2. 定义 sub_2： Data -> TensorMove -> NetOutput
  // ParentNodeIndex(0) 表示 sub_2_data 对应主图 wrapper 节点的第0个输入
  const auto sub_2_data_cfg = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_2) {
    CHAIN(NODE("sub_2_data", sub_2_data_cfg)
            ->NODE("sub_2_tensormove", TENSORMOVE)
            ->NODE("sub_2_netoutput", NETOUTPUT));
  };

  // 3. 定义主图 g1 结构: PartitionedCall_1 -> PartitionedCall_2 -> Relu -> NetOutput
  DEF_GRAPH(g1) {
    CHAIN(NODE("PartitionedCall_1", PARTITIONEDCALL, sub_1)
            ->EDGE(0, 0)->NODE("PartitionedCall_2", PARTITIONEDCALL, sub_2)
            ->NODE("relu", RELU)
            ->NODE("netoutput", NETOUTPUT));
  };

  auto sub_1_graph = ToComputeGraph(sub_1);
  auto sub_2_graph = ToComputeGraph(sub_2);
  auto compute_graph = ToComputeGraph(g1);

  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);
  const auto sub_graph_2 = compute_graph->GetSubgraph("sub_2");
  ASSERT_NE(sub_graph_2, nullptr);

  NetoutputParentIndexes indexes{
      {"sub_1_netoutput", {0}}, // sub_1_netoutput 的 input:0 对应 PartitionedCall_1 的 output:0
      {"sub_2_netoutput", {0}}  // sub_2_netoutput 的 input:0 对应 PartitionedCall_2 的 output:0
  };
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  GE_DUMP(compute_graph, "GraphBefore_SiblingTrace");

  ge::GEPass pass(compute_graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);

  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  // 预期：sub_2_tensormove 应该依然存在 (未被删除)
  EXPECT_NE(sub_2_graph->FindNode("sub_2_tensormove"), nullptr);

  GE_DUMP(compute_graph, "GraphAfter_SiblingTrace");
}

/**
 * 主图 g1：
 *
 *        data
 *          |
 *        relu
 *          |
 *          IF
 *          |
 *      transdata
 *          |
 *      tensormove
 *          |
 *      netoutput
 *
 * if 分支子图 if_sub：
 *
 *        if_sub_data
 *           |\
 *           | if_transdata
 *           |     |
 *           |  if_tensormove
 *           |     |
 *           |   if_relu
 *           |     |
 *           ----if_sub_netoutput
 *
 * then 分支子图 then_sub：
 *
 *      then_sub_data
 *           |
 *       then_relu
 *           |
 *     then_sub_netoutput
 *
 * 预期行为：
 * - tensormove的输入是transdata，只有一条路径，被删除
 * - if_tensormove的输入是if_transdata，只有一条路径，被删除
 */
TEST_F(UtestTensorMoveDeletePass, TensorMoveInRootAndIfSub_ViaTransData_Deleted) {
  dlog_setlevel(0, 0, 0);
  const auto if_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(if_sub) {
    CHAIN(NODE("if_sub_data", if_sub_data)->EDGE(0, 0)
              ->NODE("if_sub_netoutput", NETOUTPUT));
    CHAIN(NODE("if_sub_data", if_sub_data)
              ->EDGE(0, 0)->NODE("if_transdata", TRANSDATA)
              ->NODE("if_tensormove", TENSORMOVE)
              ->NODE("if_relu", RELU)
              ->Ctrl()->NODE("if_sub_netoutput", NETOUTPUT));
  };
  const auto then_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(then_sub) {
    CHAIN(NODE("then_sub_data", then_sub_data)->NODE("then_relu", RELU)->NODE("then_sub_netoutput", NETOUTPUT));
  };
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", DATA)->NODE("relu", RELU)->NODE("if", IF, if_sub, then_sub)->NODE("transdata", TRANSDATA)
              ->NODE("tensormove", TENSORMOVE)->NODE("netoutput", NETOUTPUT));
  };

  auto compute_graph = ToComputeGraph(g1);
  const auto then_sub_graph = compute_graph->GetSubgraph("then_sub");
  ASSERT_NE(then_sub_graph, nullptr);
  const auto if_sub_graph = compute_graph->GetSubgraph("if_sub");
  ASSERT_NE(if_sub_graph, nullptr);

  compute_graph->TopologicalSorting();

  NetoutputParentIndexes indexes{{"if_sub_netoutput", {0}}, {"then_sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  ge::GEPass pass(compute_graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_NE(compute_graph->FindNode("tensormove"), nullptr);
  EXPECT_NE(if_sub_graph->FindNode("if_tensormove"), nullptr);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(compute_graph->FindNode("tensormove"), nullptr);
  EXPECT_EQ(if_sub_graph->FindNode("if_tensormove"), nullptr);
}

/**
 * 主图 g1：
 *
 *        data
 *          |
 *        relu
 *          |
 *          IF
 *          |
 *     transdata
 *          |
 *     tensormove
 *          |
 *     netoutput
 *
 *
 * if 分支子图 if_sub：
 *                if_sub_data
 *                 /       \
 *                /         \
 *   if_sub_netoutput     if_tensormove
 *                             |
 *                           if_relu
 *                             |
 *                     if_sub_netoutput
 *
 * then 分支子图 then_sub：
 *
 *        then_sub_data
 *              |
 *          then_relu
 *              |
 *      then_sub_netoutput
 *
 * 预期行为：
 * - if_sub_graph 中的 if_tensormove 保留，上游源节点为 if_sub_data，但 if_sub_data 存在多条输出路径
 * - 主图中的 tensormove 被成功删除，上游源节点为 transdata，transdata → tensormove → netoutput
 */
TEST_F(UtestTensorMoveDeletePass, TensorMove_RootDeleted_SubKept_DueToSourceBranching) {
  dlog_setlevel(0, 0, 0);
  const auto if_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(if_sub) {
    CHAIN(NODE("if_sub_data", if_sub_data)->EDGE(0, 0)
              ->NODE("if_sub_netoutput", NETOUTPUT));
    CHAIN(NODE("if_sub_data", if_sub_data)
              ->EDGE(0, 0)->NODE("if_tensormove", TENSORMOVE)
              ->EDGE(0, 0)->NODE("if_relu", RELU)
              ->EDGE(0, 0)->NODE("if_sub_netoutput", NETOUTPUT));
  };
  const auto then_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(then_sub) {
    CHAIN(NODE("then_sub_data", then_sub_data)->NODE("then_relu", RELU)->NODE("then_sub_netoutput", NETOUTPUT));
  };
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", DATA)->NODE("relu", RELU)->NODE("if", IF, if_sub, then_sub)->NODE("transdata", TRANSDATA)
              ->NODE("tensormove", TENSORMOVE)->NODE("netoutput", NETOUTPUT));
  };

  auto compute_graph = ToComputeGraph(g1);
  const auto then_sub_graph = compute_graph->GetSubgraph("then_sub");
  ASSERT_NE(then_sub_graph, nullptr);
  const auto if_sub_graph = compute_graph->GetSubgraph("if_sub");
  ASSERT_NE(if_sub_graph, nullptr);

  compute_graph->TopologicalSorting();

  NetoutputParentIndexes indexes{{"if_sub_netoutput", {0}}, {"then_sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  EXPECT_NE(if_sub_graph->FindNode("if_tensormove"), nullptr);
  ge::GEPass pass(compute_graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_NE(if_sub_graph->FindNode("if_tensormove"), nullptr);
  EXPECT_NE(compute_graph->FindNode("tensormove"), nullptr);

  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  EXPECT_NE(if_sub_graph->FindNode("if_tensormove"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("tensormove"), nullptr);
}

/**
 * 主图 g1：
 *
 *        data
 *          |
 *        relu
 *          |
 *          IF
 *          |
 *     transdata
 *          |
 *     tensormove
 *          |
 *     netoutput
 *
 *
 * if 分支子图 if_sub：
 *        if_sub_data
 *             |
 *        if_tensormove
 *             |
 *     if_sub_netoutput
 *
 * then 分支子图 then_sub：
 *        then_sub_data
 *              |
 *          then_relu
 *              |
 *      then_sub_netoutput
 *
 * 预期行为：
 * - if_sub_graph 中的 if_tensormove 保留，其源输入为主图中的 relu，但 relu 的下游节点是 IF 控制流算子
 * - 主图中的 tensormove 被成功删除，其源输入为 transdata，输出是netoutput
 */
TEST_F(UtestTensorMoveDeletePass, TensorMove_RootDeleted_SubInIfKept_DueToIfOp) {
  dlog_setlevel(0, 0, 0);
  const auto if_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(if_sub) {
    CHAIN(NODE("if_sub_data", if_sub_data)->EDGE(0, 0)->NODE("if_tensormove", TENSORMOVE)
              ->EDGE(0, 0)->NODE("if_sub_netoutput", NETOUTPUT));
  };
  const auto then_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(then_sub) {
    CHAIN(NODE("then_sub_data", then_sub_data)->NODE("then_relu", RELU)->NODE("then_sub_netoutput", NETOUTPUT));
  };
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", DATA)->NODE("relu", RELU)->NODE("if", IF, if_sub, then_sub)->NODE("transdata", TRANSDATA)
              ->NODE("tensormove", TENSORMOVE)->NODE("netoutput", NETOUTPUT));
  };

  auto compute_graph = ToComputeGraph(g1);
  const auto then_sub_graph = compute_graph->GetSubgraph("then_sub");
  ASSERT_NE(then_sub_graph, nullptr);
  const auto if_sub_graph = compute_graph->GetSubgraph("if_sub");
  ASSERT_NE(if_sub_graph, nullptr);

  compute_graph->TopologicalSorting();

  NetoutputParentIndexes indexes{{"if_sub_netoutput", {0}}, {"then_sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  ge::GEPass pass(compute_graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  EXPECT_NE(if_sub_graph->FindNode("if_tensormove"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("tensormove"), nullptr);
}

/**
 * 主图 g1：
 *          data
 *            |
 *            IF
 *            |
 *        tensormove
 *            |
 *        netoutput
 *
 * if_sub：
 *        if_sub_data
 *          |      \
 *          |       \
 *          |        if_tensormove
 *          |           |
 *          |          |
 *          |         |
 *     if_sub_netoutput
 *
 * then 分支子图 then_sub：
 *
 *        then_sub_data
 *              |
 *          then_relu
 *              |
 *      then_sub_netoutput
 *
 * 预期行为：
 * - if 分支子图中的 if_tensormove 不删除，其输入为根图Data,路径上有IF算子
 * - 主图中的 tensormove 不删除，其输入为根图Data,路径上有IF算子
 */
TEST_F(UtestTensorMoveDeletePass, TensorMove_InRootAndSub_ConnectedToIf_Kept) {
  dlog_setlevel(0, 0, 0);
  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "1,1|0,0";
  ge::GetThreadLocalContext().SetGraphOption(options);
  const auto if_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(if_sub) {
    CHAIN(NODE("if_sub_data", if_sub_data)
              ->EDGE(0, 1)->NODE("if_sub_netoutput", NETOUTPUT));
    CHAIN(NODE("if_sub_data", if_sub_data)
              ->EDGE(0, 0)->NODE("if_tensormove", TENSORMOVE)
              ->EDGE(0, 0)->NODE("if_sub_netoutput", NETOUTPUT));
  };
  const auto then_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(then_sub) {
    CHAIN(NODE("then_sub_data", then_sub_data)->NODE("then_relu", RELU)->NODE("then_sub_netoutput", NETOUTPUT));
  };
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", DATA)->NODE("if", IF, if_sub, then_sub)
              ->NODE("tensormove", TENSORMOVE)->NODE("netoutput", NETOUTPUT));
  };

  auto compute_graph = ToComputeGraph(g1);
  const auto then_sub_graph = compute_graph->GetSubgraph("then_sub");
  ASSERT_NE(then_sub_graph, nullptr);
  const auto if_sub_graph = compute_graph->GetSubgraph("if_sub");
  ASSERT_NE(if_sub_graph, nullptr);

  compute_graph->TopologicalSorting();

  NetoutputParentIndexes indexes{{"if_sub_netoutput", {0, 1}}, {"then_sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  ge::GEPass pass(compute_graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_NE(if_sub_graph->FindNode("if_tensormove"), nullptr);
  EXPECT_NE(compute_graph->FindNode("tensormove"), nullptr);

  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  EXPECT_NE(if_sub_graph->FindNode("if_tensormove"), nullptr);
  EXPECT_NE(compute_graph->FindNode("tensormove"), nullptr);

  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 *       Data
 *        |
 *     TensorMove
 *        |
 *     NetOutput  (复用输入地址)
 *
 * 说明：
 * - TensorMove 输入为根图Data，直连NetOutput
 * - 设置了输出复用输入内存
 * - 单输入、单输出
 * - 无分支、无子图
 *
 * 预期行为：
 * - 删除TensorMove
 */
TEST_F(UtestTensorMoveDeletePass, TensorMoveFromData_MemReuse_Deleted2) {
  setenv("DUMP_GRAPH_LEVEL", "2", 1);
  setenv("DUMP_GE_GRAPH", "2", 1);
  dlog_setlevel(0, 0, 0);
  std::map<std::string, std::string> options;
  options[OPTION_INPUT_REUSE_MEM_INDEXES] = "0";
  ge::GetThreadLocalContext().SetGraphOption(options);
  auto builder = ut::GraphBuilder("g1");
  auto data_node = builder.AddNode("Data", DATA, 1, 1);
  auto node1 = builder.AddNode("TensorMove", TENSORMOVE, 1, 1);
  auto node2 = builder.AddNode("NetOutput", NETOUTPUT, 1, 1);

  AttrUtils::SetInt(data_node->GetOpDesc(), ATTR_NAME_INDEX, 0);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  EXPECT_EQ(node1->GetOutDataNodes().size(), 1);

  ge::GEPass pass(builder.GetGraph());
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(builder.GetGraph()->FindNode("TensorMove"), nullptr);

  ge::GetThreadLocalContext().SetGraphOption({});
}

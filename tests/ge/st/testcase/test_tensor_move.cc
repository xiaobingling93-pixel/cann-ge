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
#include "ge/ut/ge/graph/passes/graph_builder_utils.h"
#include "ge_local_context.h"
#include "graph_utils_ex.h"
#include "graph/passes/standard_optimize/tensor_move_delete_pass.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg_box.h"
#include "easy_graph/builder/graph_dsl.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/operator_reg.h"
#include "graph_metadef/external/ge_common/ge_api_types.h"
#include "api/gelib/gelib.h"
#include "ge/ge_api.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "exe_graph/runtime/infer_shape_range_context.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "graph/utils/constant_utils.h"
#include "host_kernels/kernel.h"
#include "host_kernels/kernel_factory.h"

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

REG_OP(TensorMove)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT32, DT_INT16, DT_UINT16, DT_INT8, DT_UINT8,
                          DT_UINT64, DT_INT64, DT_BOOL, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_COMPLEX32, DT_COMPLEX64}))
    .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT32, DT_INT16, DT_UINT16, DT_INT8, DT_UINT8,
                           DT_UINT64, DT_INT64, DT_BOOL, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_COMPLEX32, DT_COMPLEX64}))
    .OP_END_FACTORY_REG(TensorMove)

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

void SetWeightForConstNode(NodePtr &const_node) {
  // new a tensor
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int64_t> shape{9};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  ConstantUtils::SetWeight(const_node->GetOpDesc(), 0, tensor);
}

const char *AddNYes = "AddNYes";
const char *ShapeNo = "ShapeNo";
class TestAddNKernel : public Kernel {
public:
  Status Compute(const ge::OpDescPtr op_desc_ptr, const std::vector<ge::ConstGeTensorPtr> &input,
                 std::vector<ge::GeTensorPtr> &v_output) override {
    auto output = std::make_shared<GeTensor>();
    std::vector<uint8_t> data{1, 2, 3};
    std::vector<int64_t> shape{3};
    output->MutableTensorDesc().SetShape(GeShape(shape));
    output->SetData(data);
    output->MutableTensorDesc().SetDataType(DT_UINT8);
    v_output.push_back(output);
    return SUCCESS;
  }
};

REGISTER_COMPUTE_NODE_KERNEL(AddNYes, TestAddNKernel);
}

class TensorMoveTest : public Test {
  protected:
  void SetUp() {
    dlog_setlevel(0, 0, 0);
    std::map<std::string, std::string> options = {{"ge.oo.level", "O3"}};
    GetThreadLocalContext().SetGraphOption(options);
    GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());

    GeRunningEnvFaker().Reset().InstallDefault()
        .Install(FakeOp(AddNYes).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(ShapeNo).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeEngine("DNN_VM_AICPU_ASCEND").KernelInfoStore("aicpu_ascend_kernel"));
  }

  void TearDown() {
    dlog_setlevel(0, 3, 0);
    unsetenv("DUMP_GRAPH_LEVEL");
    unsetenv("DUMP_GE_GRAPH");
    GetThreadLocalContext().SetGraphOption({});
  }
};

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
TEST_F(TensorMoveTest, TensorMoveInSubgraph_FromParentData_Deleted) {
  dlog_setlevel(0, 0, 0);

  // 1. 设置内存复用选项：设置根图的第 0 个输出复用第 0 个输入
  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "0,0";
  options["ge.oo.level"] = "O3";
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
TEST_F(TensorMoveTest, TensorMove_NestedPCall_FromAdd_Deleted) {
  dlog_setlevel(0, 0, 0);

  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "1,1|0,0";
  options["ge.oo.level"] = "O3";
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
TEST_F(TensorMoveTest, TensorMoveInSub_FromSubSubAdd_Deleted) {
  dlog_setlevel(0, 0, 0);

  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "1,1|0,0";
  options["ge.oo.level"] = "O3";
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

TEST_F(TensorMoveTest, TensorMoveInRootAndSub_FromSubSubAdd_Deleted) {
  dlog_setlevel(0, 0, 0);

  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "1,1|0,0";
  options["ge.oo.level"] = "O3";
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
TEST_F(TensorMoveTest, TensorMoveInRootAndIfSub_ViaTransData_Deleted) {
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
TEST_F(TensorMoveTest, TensorMove_RootDeleted_SubKept_DueToSourceBranching) {
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
TEST_F(TensorMoveTest, TensorMove_RootDeleted_SubInIfKept_DueToIfOp) {
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
TEST_F(TensorMoveTest, TensorMove_InRootAndSub_ConnectedToIf_Kept) {
  dlog_setlevel(0, 0, 0);
  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "1,1|0,0";
  options["ge.oo.level"] = "O3";
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
TEST_F(TensorMoveTest, TensorMoveInSubgraph_FromParentData_Deleted2) {
  // 1. 设置内存复用选项：设置根图的第 0 个输出复用第 0 个输入
  std::map<std::string, std::string> options;
  options[OPTION_INPUT_REUSE_MEM_INDEXES] = "0";
  options["ge.oo.level"] = "O3";
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

TEST_F(TensorMoveTest, TensorMoveInSubgraph_FromParentData_NotDeleted) {
  // 1. 设置内存复用选项：设置根图的第 0 个输出复用第 0 个输入
  std::map<std::string, std::string> options;
  options["ge.oo.level"] = "O3";
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
  EXPECT_NE(sub_graph_1->FindNode("sub_tensormove"), nullptr);

  // 清理环境
  ge::GetThreadLocalContext().SetGraphOption({});
}

// 公共子表达式消除场景，添加内置Identity
TEST_F(TensorMoveTest, Add_InnerIdentity1) {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Build("assign");
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("add1", ADD)->EDGE(0, 0)->NODE(assign)->CTRL_EDGE()->NODE("add3", ADD));
    CHAIN(NODE("data1")->EDGE(0, 1)->NODE("add1"));
    CHAIN(NODE("add1")->EDGE(0, 0)->NODE("add3"));
    CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE(assign));
    CHAIN(NODE("data1")->EDGE(0,0)->NODE("add2", ADD)->EDGE(0, 1)->NODE("add3"));
    CHAIN(NODE("data1")->EDGE(0,1)->NODE("add2"));
  };

  auto graph = ToGeGraph(g1);
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
    size_t add_count = 0U;
    for (const auto &node : graph->GetAllNodes()) {
      if (node->GetType() == ADD) {
        add_count++;
      }
    }
    // 公共子表达式消除，add1和add2合并
    EXPECT_EQ(add_count, 2U);

    auto identity = graph->FindFirstNodeMatchType(IDENTITY);
    ASSERT_NE(identity, nullptr);
    auto assign = graph->FindFirstNodeMatchType(ASSIGN);
    ASSERT_NE(assign, nullptr);
    EXPECT_EQ(assign->GetInDataNodes().at(0), identity);
  };
}

// 常量折叠场景，添加内置Identity
TEST_F(TensorMoveTest, Add_InnerIdentity2) {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Build("assign");
    CHAIN(NODE("const1", CONSTANT)->NODE("addn", AddNYes)->NODE(assign)->CTRL_EDGE()->NODE("shape1", ShapeNo));
    CHAIN(NODE("const2", CONSTANT)->EDGE(0, 1)->NODE("addn"));
    CHAIN(NODE("data", DATA)->EDGE(0, 1)->NODE(assign));
    CHAIN(NODE("addn")->EDGE(0, 0)->NODE("shape1")->NODE("net_output",NETOUTPUT));
  };
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto const1 = compute_graph->FindNode("const1");
  auto const2 = compute_graph->FindNode("const2");
  SetWeightForConstNode(const1);
  SetWeightForConstNode(const2);
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
    auto identity = graph->FindFirstNodeMatchType(IDENTITY);
    ASSERT_NE(identity, nullptr);
    auto assign = graph->FindFirstNodeMatchType(ASSIGN);
    ASSERT_NE(assign, nullptr);
    EXPECT_EQ(assign->GetInDataNodes().at(0), identity);
  };
}

// relu多引用，连给两个ref op，且ref之间没有连边关系，需要插入内置inner Identity
TEST_F(TensorMoveTest, Add_InnerIdentity3) {
  DEF_GRAPH(g1) {
    auto assign1 = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Build("assign1");
    auto assign2 = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Build("assign2");
    CHAIN(NODE("data",DATA)->NODE("relu", RELU)->NODE(assign1));
    CHAIN(NODE("data")->EDGE(0, 1)->NODE(assign1));
    CHAIN(NODE("relu")->EDGE(0, 0)->NODE(assign2));
    CHAIN(NODE("data")->EDGE(0, 1)->NODE(assign2));
  };
  auto graph = ToGeGraph(g1);
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
    size_t identity_count = 0U;
    for (const auto &node : graph->GetAllNodes()) {
      if (node->GetType() == IDENTITY) {
        identity_count++;
      }
    }
    EXPECT_EQ(identity_count, 2U);
  };
}

// relu多引用，且relu的另一个输出节点依赖ref算子，不需要插入内置inner Identity
TEST_F(TensorMoveTest, InnerIdentity_Delete1) {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Build("assign");
    CHAIN(NODE("data1", DATA)->NODE("relu", RELU)->NODE(assign));
    CHAIN(NODE("relu")->EDGE(0, 0)->NODE("add", ADD)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("data2", DATA))->EDGE(0, 1)->NODE(assign)->EDGE(0, 1)->NODE("add");
  };

  auto graph = ToGeGraph(g1);
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
    auto identity = graph->FindFirstNodeMatchType(IDENTITY);
    ASSERT_EQ(identity, nullptr);
  };
}
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
#include "ge/ge_api.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_adapter.h"
#include "framework/common/types.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator_reg.h"

#include "ge_graph_dsl/assert/graph_assert.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/tensor_utils.h"
#include "common/mem_conflict_share_graph.h"
#include "ge_running_env/fake_op.h"
#include "ge_context.h"
#include "ge_local_context.h"
#include "stub/gert_runtime_stub.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "compiler/graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_optimizer.h"

class MemLayoutConflictTest : public testing::Test {
 protected:
  void SetUp() {
    global_options_ = ge::GetThreadLocalContext().GetAllGlobalOptions();
    graph_options_ = ge::GetThreadLocalContext().GetAllGraphOptions();
    session_options_ = ge::GetThreadLocalContext().GetAllSessionOptions();
    ge::GetThreadLocalContext().SetGlobalOption({});
    ge::GetThreadLocalContext().SetGraphOption({});
    ge::GetThreadLocalContext().SetSessionOption({});

    ge::GeRunningEnvFaker ge_env;
    ge_env.InstallDefault();

    mmSetEnv(kEnvValue, "", 1);
  }

  void TearDown() {
    ge::GetThreadLocalContext().SetGlobalOption(global_options_);
    ge::GetThreadLocalContext().SetGraphOption(graph_options_);
    ge::GetThreadLocalContext().SetSessionOption(session_options_);
  }
  std::map<std::string, std::string> global_options_;
  std::map<std::string, std::string> graph_options_;
  std::map<std::string, std::string> session_options_;
  const char_t * const kEnvValue = "SET_CAPA_VALUE";
};

namespace ge {
REG_OP(TensorMove)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT32, DT_INT16, DT_UINT16, DT_INT8, DT_UINT8,
                          DT_UINT64, DT_INT64, DT_BOOL, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_COMPLEX32, DT_COMPLEX64}))
    .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT32, DT_INT16, DT_UINT16, DT_INT8, DT_UINT8,
                           DT_UINT64, DT_INT64, DT_BOOL, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_COMPLEX32, DT_COMPLEX64}))
    .OP_END_FACTORY_REG(TensorMove)

/*
 *             data1    data2                                            data11   partitioned_call1 ┌──────────┐
 *                │       │                                                │         │              │constant1 │
 *                │       │                                                │         │              │    │     │
 *                │       │                                                │         │              │    │     │
 *          ┌─────┴──┐ ┌──┘                                         ┌──────┴───┐ ┌───┘              │netoutput5│
 *          │        │ │                                            │          │ │                  └──────────┘
 *          │        │ │                                            │          │ │
 * partitioned_call2 If1                                           swt         If2
 * ┌───────────┐      │  then_subgraph1     else_subgraph1          │           │   then_subgraph2     else_subgraph2
 * │ data12    │      │ ┌─────────────┐   ┌─────────────┐           │           │  ┌─────────────┐   ┌─────────────┐
 * │   ┼       │      │ │ data3  data4│   │ data5  data6│           │           │  │ data7  data8│   │ data9 data10│
 * │ netoutput6│      │ │   │         │   │          │  │           │           │  │   │         │   │          │  │
 * └───┬───────┘      │ │   └──┬      │   │       ┬──┘  │           │           │  │   └──┬      │   │       ┬──┘  │
 *     │              │ │      │      │   │       │     │           │           │  │      │      │   │       │     │
 *     │              │ │   netoutput1│   │  netoutput2 │           │           │  │   netoutput3│   │  netoutput4 │
 *     │              │ └─────────────┘   └─────────────┘           │          op4 └─────────────┘   └─────────────┘
 *     │             op3                                            │           |
 *     │              │                                             │           │
 *     │              │                                             │           │
 *     │              └─────────────────────────────┐ ┌─────────────┘           │
 *     │                                            │ │                         │
 *     └──────────────────────────────────────────┐ │ │                         │
 *                                                │ │ │  +──────────────────────┘
 *                                                │ │ │  │
 *                                                netoutput8
 *
 *
 *                                        data2                              data11             partitioned_call1 ┌──────────┐
 *               data1                      │                                  │                   │              │constant1 │
 *                  │                       │                                  │                   │              │    │     │
 *                  │                       │                                  │                   │              │    │     │
 *                  │                    ┌──┘                           ───────┴───┐           ┌───┘              │netoutput5│
 *            ┌─────┴────┐               │                           identity      │           │                  └──────────┘
 *            │          │ctrl           │                              ┼          │ctrl       │
 *            │          │         then_sub1(partitioned_call)         swt      const      then_sub2(partitioned_call)
 *   partitioned_call2   const          │  then_sub1rsubgraph           │                     │   then_subgraph2
 *   ┌───────────┐                      │ ┌─────────────┐               │                     │  ┌─────────────┐
 *   │ data12    │                      │ │ data3  data4│               │                     │  │ data7  data8│
 *   │   ┼       │                      │ │   │     │   │               │                     │  │   │      │  │
 *   │ netoutput6│                      │ │   └─┐ ┌─┘   │               │                     │  │   └─┐ ┌──┘  │
 *   └────┬──────┘                      │ │     │ │ctrl │               │                     │  │     │ │ctrl │
 *        │                             │ │   netoutput1│              identity               │  │   netoutput3│
 *        │                             │ └─────────────┘               │                    op4 └─────────────┘
 *        │                            op3                              │                     │
 *        │                             │                               │                     │
 *        │                             └─────────────────────────────┐ │                     │
 *        │                                                           │ │                     │
 *        └─────────────────────────────────────────────────────────┐ │ │                     │
 *                                                                  │ │ │  ├──────────────────┘
 *                                                                  │ │ │  │
 *                                                                  netoutput8
 *
 * 用例场景：本用例意图测试内存类型冲突功能，
 * 构造用户输入与用户输入/用户输入与不可遍地址输入/用户输入与不可支持地址刷新算子/用户输入与用户输出等多种冲突的图
 * 步骤：
 * step 1. 构造一张用户输入与多种不同类型内存的冲突场景的图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功，校验identity插入位置
 */
TEST_F(MemLayoutConflictTest, UserInputAndUserInputAndConstantAndUserOutAndNotRefreshableInput_InsertIdentity_Success) {
  gert::GertRuntimeStub runtime_stub;
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto root_graph = MemConflictShareGraph::BuildUserInAndOtherConflictTypeGraph();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(root_graph), options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(root_graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["netoutput8"], nullptr);
    EXPECT_EQ(name_to_node["netoutput8"]->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    ASSERT_NE(name_to_node["netoutput8"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);

    ASSERT_NE(name_to_node["swt"], nullptr);
    EXPECT_EQ(name_to_node["swt"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}

/*
 *             data1    data2
 *          ┌─────┴──┐ ┌──┘
 *          │        │ │ +---op0
 *          │        │ │ |0
 * partitioned_call2  If1
 * ┌───────────┐      │  then_subgraph1     else_subgraph1
 * │ data7     │      │ ┌─────────────┐   ┌─────────────┐
 * │   ┼       │      │ │ data3  data4│   │ data5  data6│
 * │ netoutput3│      │ │   │         │   │  |       │  │
 * └───┬───────┘      │ │   └──┬      │   │ op2   ┬──┘  │
 *     │              │ │      │      │   │       │     │
 *     │              │ │   netoutput1│   │  netoutput2 │
 *     │              │ └─────────────┘   └─────────────┘
 *     │             op1
 *     │              |
 *     +-----+  +-----+
 *           |  |
 *         netoutput4
 *
 *         ||
 *         \/
 *             data1    data2
 *          ┌─────┴──┐ ┌──┘
 *          │        │ │ +---op0
 *          │        │ │ |0
 * partitioned_call2  If1
 * ┌───────────┐      │  then_subgraph1     else_subgraph1
 * │ data7     │      │ ┌─────────────┐   ┌─────────────┐
 * │   ┼       │      │ │ data3  data4│   │ data5  data6│
 * │ netoutput3│      │ │   │         │   │  |       │  │
 * └───┬───────┘      │ │ identity    │   │op2  identity│
 *     │              │ │      │      │   │       │     │
 *     │              │ │   netoutput1│   │  netoutput2 │
 *     │              │ └─────────────┘   └─────────────┘
 *  identity         op1
 *     │              |
 *     +-----+  +-----+
 *           |  |
 *         netoutput4
 *
 * 用例场景：本用例意图测试用户输入与用户输入/输出相连时会插入identity
 * 步骤：
 * step 1. 构造一张用户输入与用户输入/输出相连的图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功，校验identity插入位置
 */
TEST_F(MemLayoutConflictTest, UserInputAndUserIO_InsertIdentity_Success) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto root_graph = MemConflictShareGraph::BuildUserInAndUserIO();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(root_graph), options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(root_graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["netoutput4"], nullptr);
    ASSERT_NE(name_to_node["netoutput4"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);

    ASSERT_NE(name_to_node["netoutput1"], nullptr);
    EXPECT_EQ(name_to_node["netoutput1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    ASSERT_NE(name_to_node["netoutput2"], nullptr);
    EXPECT_EQ(name_to_node["netoutput2"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}

/*
 *          data1
 *            │
 *            │
 *            │    op1
 *     ┌──────┴───┐ |
 *     │          │ |
 *     │          │ |0
 *    swt         If1
 *     │           │   then_subgraph1   else_subgraph1
 *     │           │  ┌─────────────┐   ┌──────────---───┐
 *     │           │  │ data2       │   │data3 constant1 │
 *     │           │  │   │         │   │        │       │
 *     │           │  │   └──┬      │   │        │       │
 *     │           │  │      │      │   │        │       │
 *     │           │  │   netoutput1│   │    netoutput2  │
 *     │          op2 └─────────────┘   └──────────---───┘
 *     │           |
 *     +--+  +-----+
 *        |  |
 *      netoutput
 *
 *      ||
 *      \/
 *          data1
 *            │
 *            │
 *            │    op1
 *     ┌──────┴───┐ |
 * identity       │ |
 *     │          │ |0
 *    swt         If1
 *     │           │   then_subgraph1   else_subgraph1
 *     │           │  ┌─────────────┐   ┌──────────---───┐
 *     │           │  │ data2       │   │data3 constant1 │
 *     │           │  │   │         │   │        │       │
 * identity        │  │ identity    │   │       identity │
 *     │           │  │      │      │   │        │       │
 *     │           │  │   netoutput1│   │    netoutput2  │
 *     │          op2 └─────────────┘   └──────────---───┘
 *     │           |
 *     +--+  +-----+
 *        |  |
 *      netoutput
 *
 * 用例场景：本用例意图测试用户输入与constant/不支持刷新的算子输入相连时会插入identity
 * 步骤：
 * step 1. 构造一张用户输入与constant/不支持刷新的算子输入相连的图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功，校验identity插入位置
 */
TEST_F(MemLayoutConflictTest, UserInputAndUnRefreshableInputAndConstInput_InsertIdentity_Success) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto root_graph = MemConflictShareGraph::BuildUserInAndUnRefreshableInputAndConstInput();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(root_graph), options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(root_graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["netoutput"], nullptr);
    EXPECT_EQ(name_to_node["netoutput"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    ASSERT_NE(name_to_node["swt"], nullptr);
    EXPECT_EQ(name_to_node["swt"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);

    ASSERT_NE(name_to_node["netoutput1"], nullptr);
    EXPECT_EQ(name_to_node["netoutput1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    ASSERT_NE(name_to_node["netoutput2"], nullptr);
    EXPECT_EQ(name_to_node["netoutput2"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}

/*
 *
 *   data--netoutput   ----> data---netoutput
 *
 * 用例场景：本用例意图测试用户输入与用户输出直接相连不需要插入identity
 * 步骤：
 * step 1. 构造一张用户输入直连用户输出的图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功，校验不插入identity
 */
TEST_F(MemLayoutConflictTest, UserInputConnectUserOutput_NotInsertIdentity_ExecuteSuccess) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto root_graph = MemConflictShareGraph::BuildUserInConnectNetoutputGraph();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(root_graph), options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["netoutput"], nullptr);
    EXPECT_NE(name_to_node["netoutput"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}

/*
 *  constant---netoutput   ---->  constant--identity--netoutput
 *
 * 用例场景：本用例意图测试constant与用户输出直接相连需要插入identity
 * 步骤：
 * step 1. 构造一张constant直连用户输出的图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功，校验插入identity
 */
TEST_F(MemLayoutConflictTest, ConstantConnectUserOutput_InsertIdentity_ExecuteSuccess) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto root_graph = MemConflictShareGraph::BuildConstantConnectNetoutputGraph();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(root_graph), options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["netoutput"], nullptr);
    EXPECT_EQ(name_to_node["netoutput"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}


/*
 *      op1                      partitioned_call2    op3         op4
 *       │                          ┌───────────┐      │           │
 *     ┌─┼───────┐                  │ constant1 │      ├────┐      ├────┐
 *     │ │       │                  │     │     │      │    │      │    │
 *     │ │  partitioned_call1       │     │     │      │   swt     │   hcom
 *     │ │  ┌────────────┐          │ netoutput2│      │    │      │    │
 *     │ │  │   data1    │          └─────┬─────┘      │    │      │    │
 *     │ │  │    ┼       │                │            │    │      │    │
 *     │ │  │  netoutput1│                │            │    │      │    │
 *     │ │  └────┬───────┘                │            │    │      │    │
 *     │ │       │                        │            │    │      │    │
 *     │ │       └───────────────┐ ┌──────┘            │    │      │    │
 *     │ │                       │ │                   │    │      │    │
 *     │ └─────────────────────┐ │ │ ┌─────────────────┘    │      │    │
 *     │                       │ │ │ │                      │      │    │
 *     └─────────────────────┐ │ │ │ │  ┌───────────────────┘      │    │
 *                           │ │ │ │ │  │                          │    │
 *                           │ │ │ │ │  │  ┌───────────────────────┘    │
 *                           │ │ │ │ │  │  │                            │
 *                           │ │ │ │ │  │  │  ┌─────────────────────────┘
 *                          0│1│2│3│4│ 5│ 6│ 7│
 *                              netoutput
 *
 * 用例场景：本用例意图测试用户输出与其他特殊类型的图结构是否正确的插入identity
 * 步骤：
 * step 1. 构造一张包含用户输出与其他所有类型内存的图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功，校验netoutput 插入identity
 */
TEST_F(MemLayoutConflictTest,
       UserOutputAndUserOutputAndConstantAndNotRefreshableInputOutputAndRtsSpecialInputOutput_InsertIdentity_Success) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto root_graph = MemConflictShareGraph::BuildUserOutAndOtherConflictTypeGraph();

  map<AscendString, AscendString> options;
  options[OPTION_FEATURE_BASE_REFRESHABLE] = "0";
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(root_graph), options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["netoutput"], nullptr);
    EXPECT_NE(name_to_node["netoutput"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_NE(name_to_node["netoutput"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_NE(name_to_node["netoutput"]->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_NE(name_to_node["netoutput"]->GetInDataAnchor(3)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["netoutput"]->GetInDataAnchor(4)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["netoutput"]->GetInDataAnchor(5)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_NE(name_to_node["netoutput"]->GetInDataAnchor(6)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["netoutput"]->GetInDataAnchor(7)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);

  };
}

/*
 *   op1 partitioned_call1
 *    |   | | |
 *    |   | | |  op2
 *   0\  1|2/3/4/
 *       if
 *        ┼
 *        |
 *       op5
 *        ┼
 *       netoutput
 *
 *
 *   partitioned_call1          then_sub                             else_sub
 *  ┌────────────────────┐    ┌───────────────────────────────┐   ┌───────────────────────────────────────┐
 *  │ var const1 refdata1│    │ data1 data2 data3 data4 data5 │   │  data6 data7 data8 data9 data10       │
 *  │  │     │      │    │    │         ┼     │     │     │   │   │                                       │
 *  │  └───┐ │  ┌───┘    │    │       refnode │     │     │   │   │   refdata2       op3                  │
 *  │      │ │  │        │    │         |     │     │     │   │   │      ┼            ┼                   │
 *  │    netoutput1      │    │         | +---+     │     │   │   │   transdata1    transdata2   swt  op4 │
 *  └────────────────────┘    │         + | +-------+     │   │   │ (ref refdata1)  (ref var)     │    │  │
 *                            │         | | | ┌───────────┘   │   │          │           │ ┌──────┘    │  │
 *                            │         ┼ ┼ ┼ │               │   │          └────────┐  │ │ +---------+  │
 *                            │        netoutput2             │   │                   │  │ │ |            │
 *                            └───────────────────────────────┘   │                 netoutput3            │
 *                                                                └───────────────────────────────────────┘
 *                          ||
 *                          \/
 *   partitioned_call1          then_sub                             else_sub
 *  ┌────────────────────┐    ┌───────────────────────────────┐   ┌────────────────────────────────────--───┐
 *  │ var const1 refdata1│    │ data1 data2 data3 data4 data5 │   │  data6 data7 data8 data9 data10         │
 *  │  │     │      │    │    │         ┼     │     │     │   │   │                                         │
 *  │  └───┐ │  ┌───┘    │    │       refnode │     │     │   │   │   refdata2       op3                    │
 *  │      │ │  │        │    │         | identity identity   │   │      ┼            ┼                     │
 *  │    netoutput1      │    │         | +---+     │ identity│   │   transdata1    transdata2   swt  op4   │
 *  └────────────────────┘    │         + | +-------+     │   │   │ (ref refdata1)  (ref var)     │ identity│
 *                            │   identity| | ┌───────────┘   │   │          │     identity┌──identity │    │
 *                            │         ┼ ┼ ┼ │               │   │      identity───┐    | │ +---------+    │
 *                            │        netoutput2             │   │                 │    │ │ |              │
 *                            └───────────────────────────────┘   │                 netoutput3              │
 *                                                                └──────────────────────────────────────--─┘
 * 用例场景：本用例意图测试不可变地址输出与其他类型输出冲突是否正确的插入identity
 * 步骤：
 * step 1. 构造一张包含不可变地址输出与其他所有类型内存的图，
 * 不可变地址包括const/constant/variable及带有REF_VAR_SRC_VAR_NAME的节点
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功，校验插入identity
 */
TEST_F(MemLayoutConflictTest, ImmutableOutAndOtherType_InsertIdentity_Success) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto root_graph = MemConflictShareGraph::BuildImmutableOutAndOtherTypeGraph();

  map<AscendString, AscendString> options;
  options[OPTION_FEATURE_BASE_REFRESHABLE] = "1";
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(root_graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["netoutput2"], nullptr);
    EXPECT_EQ(name_to_node["netoutput2"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["netoutput2"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["netoutput2"]->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["netoutput2"]->GetInDataAnchor(3)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);

    ASSERT_NE(name_to_node["netoutput3"], nullptr);
    EXPECT_EQ(name_to_node["netoutput3"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["netoutput3"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["netoutput3"]->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_NE(name_to_node["netoutput3"]->GetInDataAnchor(3)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);

    auto tensor_move = graph->FindFirstNodeMatchType(TENSORMOVE);
    EXPECT_EQ(tensor_move, nullptr);
  };
}

/*
 *   op1  hcom1 op2
 *     \   /    / \
 *      | |   /    \
 *       swt1      hcom2
 *       dsa地址不可刷新，hcom1 hcom2 特殊内存类型P2P
 *
 *       ||
 *       \/
 *
 *   op1       hcom1      op2
 *     \        /        /   \
 *      |      |       /       \
 *  identity identity identity hcom2
 *    \       |         /
 *        swt1
 *       dsa地址不可刷新，hcom1 hcom2 特殊内存类型P2P
 *
 * 用例场景：本用例意图测试不可刷新地址输入与其他特殊类型的图结构是否正确的插入identity
 * 步骤：
 * step 1. 构造一张包含不可刷新地址输入与其他所有类型内存的图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功，校验dsa输入前插入identity
 */
TEST_F(MemLayoutConflictTest,
       NotSupportRefreshInputAndRtsSpecialInputOutputAndNormalOut_InsertIdentity_Success) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto root_graph = MemConflictShareGraph::BuildNotSupportRefreshInputAndOtherConflictTypeGraph();

  map<AscendString, AscendString> options;
  options[OPTION_FEATURE_BASE_REFRESHABLE] = "1";
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(root_graph), options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["swt"], nullptr);
    EXPECT_EQ(name_to_node["swt"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Identity");
    EXPECT_EQ(name_to_node["swt"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Identity");
    EXPECT_EQ(name_to_node["swt"]->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Identity");
  };
}

TEST_F(MemLayoutConflictTest,
       RtsSpecialInputAndRtsSpecialInputOutputAndNormalOut_InsertIdentity_Success) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto root_graph = MemConflictShareGraph::BuildRtsSpecialInAndOtherTypeGraph();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(root_graph), options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["hcom"], nullptr);
    EXPECT_EQ(name_to_node["hcom"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}

/*
 *    op1
 *     |
 *    if           then_sub1         else_sub1
 *    /\           +--------------+  +--------------+
 *  op3 op4        | data1        |  | data1        |
 *   |   |         |  |           |  |              |
 *   netoutput     | memcpy1      |  | op2          |
 *                 |  |           |  |  |           |
 *                 | swt1    swt2 |  | memcpy2  op3 |
 *                 |  \       /   |  |  \       /   |
 *                 |   netoutput1 |  |   netoutput1 |
 *                 +--------------+  +--------------+
 */
TEST_F(MemLayoutConflictTest,
       NotSupportRefreshOutAndRtsSpecailIOAndNormalOut_InsertIdentity_Success) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto root_graph = MemConflictShareGraph::BuildNotSupportRefreshOutAndRtsSpecailIOAndNormalOutGraph();

  map<AscendString, AscendString> options;
  options[OPTION_FEATURE_BASE_REFRESHABLE] = "1";
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(root_graph), options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["swt1"], nullptr);
    EXPECT_EQ(name_to_node["swt1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "MemcpyAsync");

    ASSERT_NE(name_to_node["netoutput1"], nullptr);
    EXPECT_EQ(name_to_node["netoutput1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Identity");
    EXPECT_EQ(name_to_node["netoutput1"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Identity");
  };
}

/*
 *    pred
 *     |
 *    cast   input
 *      \    /
 *        if
 *        |
 *    netoutput
 *
 * then_subgraph          else_subgraph
 * +------------------+   +------------------+
 * |data1--netoutput1 |   |data2--netoutput2 |
 * +------------------+   +------------------+
 *
 */
TEST_F(MemLayoutConflictTest, UserInputAndUserInput) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildIfGraph();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["netOutput1"], nullptr);
    EXPECT_EQ(name_to_node["netOutput1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    ASSERT_NE(name_to_node["netOutput2"], nullptr);
    EXPECT_EQ(name_to_node["netOutput2"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}

/*
 *        input
 *          |
 *        cast0
 *          |
 *        while1
 *          |
 *        cast1
 *          |
 *       netoutput
 *
 * subgraph cond             subgraph body
 * +-------------------+     +-------------------+
 * | data0--netoutput0 |     | data1--netoutput1 |
 * +-------------------+     +-------------------+
 *
 */
TEST_F(MemLayoutConflictTest, WhileBodyDataToNetoutput) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildWhileDataToNetoutputGraph();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["netoutput1"], nullptr);
    EXPECT_EQ(name_to_node["netoutput1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), DATA);
  };
}

/*
 *        input
 *          |
 *        cast0
 *          |
 *        while1
 *          |
 *        cast1
 *          |
 *       netoutput
 *
 * subgraph cond             subgraph body
 * +-------------------+     +--------------------------------+
 * | data0--netoutput0 |     | data1--while_cast1--netoutput1 |
 * +-------------------+     +--------------------------------+
 *
 */
TEST_F(MemLayoutConflictTest, WhileBodyDataMulNetoutput) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildWhileDataMulNetoutputGraph();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["while_mul1"], nullptr);
    EXPECT_EQ(name_to_node["while_mul1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    ASSERT_NE(name_to_node["netoutput1"], nullptr);
    EXPECT_EQ(name_to_node["netoutput1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}

/*
 *        input
 *          |
 *        cast0
 *          |
 *        while1
 *          |
 *        cast1
 *          |
 *       netoutput
 *
 * subgraph cond             subgraph body
 * +-------------------+     +---------------------------------------------+
 * | data0--netoutput0 |     | data1--while_cast1--while_cast2--netoutput1 |
 * +-------------------+     +---------------------------------------------+
 *
 */
TEST_F(MemLayoutConflictTest, WhileBodyDataMulMulNetoutput) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildWhileDataMulMulNetoutputGraph();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["while_mul1"], nullptr);
    EXPECT_EQ(name_to_node["while_mul1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Identity");
    ASSERT_NE(name_to_node["netoutput1"], nullptr);
    EXPECT_EQ(name_to_node["netoutput1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Identity");
  };
}

/*
 *    input1 input2
 *       |      |
 *     cast1  cast2
 *        \    /
 *        while1
 *         \ /
 *         add
 *          |
 *       netoutput
 *
 * subgraph cond             subgraph body
 * +-------------------+     +-------------------------+
 * | data0--netoutput0 |     | data1--\ /---netoutput1 |
 * +-------------------+     | data2--/ \--+           |
 *                           +-------------------------+
 */
TEST_F(MemLayoutConflictTest, WhileBodyDataToNetoutputExchange) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildWhileTwoDataToNetoutputExchangeGraph();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["netoutput1"], nullptr);
    EXPECT_EQ(name_to_node["netoutput1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["netoutput1"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    // TODO: subgraph pass 删除后：
    // ASSERT_NE(name_to_node["netoutput1"], nullptr);
    // EXPECT_EQ(name_to_node["netoutput1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    // EXPECT_EQ(name_to_node["netoutput1"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    // auto out_identityn_name = name_to_node["netoutput1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName();
    // ASSERT_NE(name_to_node[out_identityn_name], nullptr);
    // EXPECT_EQ(name_to_node[out_identityn_name]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    // EXPECT_EQ(name_to_node[out_identityn_name]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}

TEST_F(MemLayoutConflictTest, RtsSpecailOutAndRtsSpecailOut_InsertOneIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildRtsSpecialOutAndRtsSpecialOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);

  std::vector<NodePtr> identity_nodes;
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == IDENTITY) {
      identity_nodes.emplace_back(node);
    }
  }
  EXPECT_EQ(identity_nodes.size(), 1U);
}

TEST_F(MemLayoutConflictTest, NotSupportRefreshInAndRtsSpecailIn_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshInAndRtsSpecialInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);

  std::vector<NodePtr> identity_nodes;
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == IDENTITY) {
      identity_nodes.emplace_back(node);
    }
  }
  EXPECT_EQ(identity_nodes.size(), 0U);
}

TEST_F(MemLayoutConflictTest,
       NotSupportRefreshInAndRtsSpecailIn_InsertIdentityWhenFeatureMapRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshInAndRtsSpecialInGraph();
  MemLayoutConflictOptimizer mem_check_pass;

  auto old_graph_options_ = GetThreadLocalContext().GetAllGraphOptions();
  std::map<std::string, std::string> new_graph_options = old_graph_options_;
  new_graph_options[OPTION_FEATURE_BASE_REFRESHABLE] = "1";
  GetThreadLocalContext().SetGraphOption(new_graph_options);

  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
  std::vector<NodePtr> identity_nodes;
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == IDENTITY) {
      identity_nodes.emplace_back(node);
    }
  }

  GetThreadLocalContext().SetGraphOption(old_graph_options_);
  EXPECT_EQ(identity_nodes.size(), 1U);
}

TEST_F(MemLayoutConflictTest,
       NotSupportRefreshOutAndRtsSpecailIn_InsertIdentityWhenFeatureMapRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshOutAndRtsSpecialInGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  auto old_graph_options_ = GetThreadLocalContext().GetAllGraphOptions();
  std::map<std::string, std::string> new_graph_options = old_graph_options_;
  new_graph_options[OPTION_FEATURE_BASE_REFRESHABLE] = "1";
  GetThreadLocalContext().SetGraphOption(new_graph_options);

  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);

  std::vector<NodePtr> identity_nodes;
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == IDENTITY) {
      identity_nodes.emplace_back(node);
    }
  }

  GetThreadLocalContext().SetGraphOption(old_graph_options_);
  EXPECT_EQ(identity_nodes.size(), 1U);
}

TEST_F(MemLayoutConflictTest,
       NotSupportRefreshOutAndNormalOut_InsertIdentityWhenFeatureMapRefreshable_Success) {
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshOutAndNormalOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  auto old_graph_options_ = GetThreadLocalContext().GetAllGraphOptions();
  std::map<std::string, std::string> new_graph_options = old_graph_options_;
  new_graph_options[OPTION_FEATURE_BASE_REFRESHABLE] = "1";
  GetThreadLocalContext().SetGraphOption(new_graph_options);

  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);

  std::vector<NodePtr> identity_nodes;
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == IDENTITY) {
      identity_nodes.emplace_back(node);
    }
  }

  GetThreadLocalContext().SetGraphOption(old_graph_options_);
  EXPECT_EQ(identity_nodes.size(), 1U);
}

TEST_F(MemLayoutConflictTest, WhileBodyDataToNetoutputExchange_OnlyMemCheckPass) {
  auto graph = MemConflictShareGraph::BuildWhileTwoDataToNetoutputExchangeGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);

  std::vector<NodePtr> identity_nodes;
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == IDENTITY) {
      identity_nodes.emplace_back(node);
    }
  }
  EXPECT_EQ(identity_nodes.size(), 2U);
}

/*
 *    pred
 *     |
 *    cast   input
 *      \    /
 *        if
 *        \/
 *    netoutput
 *
 * then_subgraph        else_subgraph
 * +----------------+   +---------------+
 * |     data1      |   |    data2      |
 * |      |         |   |      |        |
 * |     mul1       |   |     / \       |
 * |      |         |   |   mu2  mu3    |
 * |     / \        |   |    |    |     |
 * |   netoutput1   |   |   netoutput2  |
 * +----------------+   +---------------+
 */
TEST_F(MemLayoutConflictTest, IfSingleOutMultiRefToNetoutput_CheckReturnSuccess_OnlyMemCheckPass) {
  auto graph = MemConflictShareGraph::BuildIfSingleOutMultiRefToNetoutputSubGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);

  std::vector<NodePtr> identity_nodes;
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == IDENTITY) {
      identity_nodes.emplace_back(node);
    }
  }
  EXPECT_EQ(identity_nodes.size(), 1U);
}

/*
 *     data
 *      |
 *   phonysplit (输出引用输入
 *     /  \      ，且NoPadding连续输出内存)
 *    a    b
 * 用例场景：用户输入直连PhonySplit， (非 load model withq, 非 offline, 非单算子) 场景， 用户输入可以直接零拷贝
 * 步骤：
 * step 1. 构造用户输入直连需要Nopadding连续输出且输出引用输入的算子
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验逻辑地址
 * 期望： 1. phony_split第一个输出地址等于 data 输出地址
 *       2. phony_split第二个输出地址等于第一个输出地址+第一个输出size
 */
TEST_F(MemLayoutConflictTest, UserInputConnectToNoPaddingContinuousOutput_NotInsertIdentity_CheckAddressCorrect) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildUserInConnectNoPaddingContinuousOutputGraph();

  map<AscendString, AscendString> options;
  options[OPTION_BUILD_GRAPH_MODE] = "";
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["phony_split"], nullptr);
  };

  const auto data_node = graph->FindNode("data");
  ASSERT_NE(data_node, nullptr);
  const auto phony_split_node = graph->FindNode("phony_split");
  ASSERT_NE(phony_split_node, nullptr);

  const auto data_out_offsets = data_node->GetOpDescBarePtr()->GetOutputOffset();
  const auto split_out_offsets = phony_split_node->GetOpDescBarePtr()->GetOutputOffset();
  int64_t split_out_size = 0;
  auto split_out_0_tensor = phony_split_node->GetOpDescBarePtr()->GetOutputDesc(0);
  TensorUtils::GetTensorSizeInBytes(split_out_0_tensor, split_out_size);

  EXPECT_EQ(split_out_offsets[0], data_out_offsets[0]);
  EXPECT_EQ(split_out_offsets[1], split_out_offsets[0] + split_out_size);
}

/*                                            data
 *                                             |
 *     data                                 identity
 *      |                    ====>             |
 *   phonysplit (输出引用输入                 phonysplit
 *     /  \      ，且NoPadding连续输出内存)      /  \
 *    a    b                                 a    b
 * 用例场景：用户输入直连PhonySplit，插入identity，内存分配正确. (load model withq, offline, 单算子) 场景
 * 步骤：
 * step 1. 构造用户输入直连需要Nopadding连续输出且输出引用输入的算子
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验逻辑地址
 * 期望： phony_split第二个输出地址等于第一个输出地址+第一个输出size
 *       phony_split前插入identity
 */
TEST_F(MemLayoutConflictTest, UserInputConnectToNoPaddingContinuousOutput_InsertIdentity_CheckAddressCorrect) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildUserInConnectNoPaddingContinuousOutputGraph();
  ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);  // set graph to signle op

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["phony_split"], nullptr);
    EXPECT_EQ(name_to_node["phony_split"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };

  const auto data_node = graph->FindNode("data");
  ASSERT_NE(data_node, nullptr);
  const auto phony_split_node = graph->FindNode("phony_split");
  ASSERT_NE(phony_split_node, nullptr);

  const auto data_out_offsets = data_node->GetOpDescBarePtr()->GetOutputOffset();
  const auto split_out_offsets = phony_split_node->GetOpDescBarePtr()->GetOutputOffset();
  int64_t split_out_size = 0;
  auto split_out_0_tensor = phony_split_node->GetOpDescBarePtr()->GetOutputDesc(0);
  TensorUtils::GetTensorSizeInBytes(split_out_0_tensor, split_out_size);

  EXPECT_EQ(split_out_offsets[1], split_out_offsets[0] + split_out_size);
}

/*
 *                                        const
 *                                          |
 *     const                              identity
 *      |                    ====>          |
 *   phony_split (输出引用输入，            phony_split
 *     /  \      且NoPadding连续输出内存)     /  \
 *    a    b                               a    b
 * 用例场景：不可变地址输出直连PhonySplit，不插入identity，内存分配正确
 * 步骤：
 * step 1. 构造用户输入直连需要Nopadding连续输出且输出引用输入的算子
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：
 * phony_split第二个输出地址等于第一个输出地址+第一个输出size
 * phony_split第0个输出地址等于第0个输入地址
 * phony_split前插入identity
 */
TEST_F(MemLayoutConflictTest, ConstConnectToNoPaddingContinuousOutput_InsertIdentity_CheckAddressCorrect) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildImmutableOutAndNoPaddingContinuousOutput();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["phony_split"], nullptr);
    EXPECT_EQ(name_to_node["phony_split"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };

  const auto const_node = graph->FindNode("const");
  ASSERT_NE(const_node, nullptr);
  const auto phony_split_node = graph->FindNode("phony_split");
  ASSERT_NE(phony_split_node, nullptr);

  const auto split_in_offsets = phony_split_node->GetOpDescBarePtr()->GetInputOffset();
  const auto split_out_offsets = phony_split_node->GetOpDescBarePtr()->GetOutputOffset();
  int64_t split_out_size = 0;
  auto split_out_0_tensor = phony_split_node->GetOpDescBarePtr()->GetOutputDesc(0);
  ge::TensorUtils::GetTensorSizeInBytes(split_out_0_tensor, split_out_size);
  EXPECT_EQ(split_out_offsets[0], split_in_offsets[0]);
  EXPECT_EQ(split_out_offsets[1], split_out_offsets[0] + split_out_size);
}

/*
 *     refdata
 *      |
 *   phony_split (输出引用输入，
 *     /  \      且NoPadding连续输出内存)
 *    a    b
 * 用例场景：refdata(用户输入)直连PhonySplit，不插入identity，内存分配正确
 * 步骤：
 * step 1. 构造用户输入直连需要Nopadding连续输出且输出引用输入的算子
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：
 * phony_split第二个输出地址等于第一个输出地址+第一个输出size
 * phony_split第0个输出地址等于第0个输入地址
 * phony_split前不插入identity
 */
TEST_F(MemLayoutConflictTest, RefDataConnectToNoPaddingContinuousOutput_InsertIdentity_CheckAddressCorrect) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildRefDataAndNoPaddingContinuousOutput();

  map<AscendString, AscendString> options;
  options[OPTION_BUILD_GRAPH_MODE] = "";
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  ASSERT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["split"], nullptr);
    EXPECT_NE(name_to_node["split"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };

  const auto refdata_node = graph->FindNode("refdata");
  ASSERT_NE(refdata_node, nullptr);
  const auto phony_split_node = graph->FindNode("split");
  ASSERT_NE(phony_split_node, nullptr);

  const auto split_in_offsets = phony_split_node->GetOpDescBarePtr()->GetInputOffset();
  const auto split_out_offsets = phony_split_node->GetOpDescBarePtr()->GetOutputOffset();
  int64_t split_out_size = 0;
  auto split_out_0_tensor = phony_split_node->GetOpDescBarePtr()->GetOutputDesc(0);
  ge::TensorUtils::GetTensorSizeInBytes(split_out_0_tensor, split_out_size);
  EXPECT_EQ(split_out_offsets[0], split_in_offsets[0]);
  EXPECT_EQ(split_out_offsets[1], split_out_offsets[0] + split_out_size);
}

/*
 *      split (连续输出)            split (连续输出)
 *      /    \        ====>       /    \
 * netoutput  b              identity   b
 *                             |
 *                           netoutput
 * 用例场景：连续输出的算子连接用户输出地址，插入identity
 * 步骤：
 * step 1. 构造连续输出的算子连接用户输出的图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：
 * split前插入identity
 */
TEST_F(MemLayoutConflictTest, UserOutConnectToContinuousOutput_InsertIdentity) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildUserOutAndContinuousOutputGraph();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["netoutput"], nullptr);
    EXPECT_EQ(name_to_node["netoutput"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}
/*
 * data1  data2                                   data1  data2
 *   |      |                                      |      |
 *   a      b                                      a      b
 *   \      /                                      \      /
 *   phony_concat (NoPaddingContinuousInput, ==>  phony_concat
 *      |           and otuput ref input)              |
 *    netoutput                                     netoutput
 *
 * 用例场景：NoPadding连续输入，且输出引用输入，且输出直连用户输出，不插入identity
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：
 * 不插入identity
 */
TEST_F(MemLayoutConflictTest, UserOutConnectToNoPaddingContinuousOutputByRef_NotInsertIdentity) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildUserOutAndNoPaddingContinuousInputByReferenceGraph();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["netoutput"], nullptr);
    EXPECT_NE(name_to_node["netoutput"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}

/*
 *   data1                                     data1
 *     |                                         |
 *    op1                                       op1
 *     |                               ==>       |
 * partitioned_call  +----------------+      partitioned_call  +----------------+
 *     |             |data2  constant |          |             |data2  constant |
 *    op2            |         |      |         op2            |         |      |
 *     |             |      netoutput1|          |             |      netoutput1|
 *  netoutput2       +----------------+       netoutput2       +----------------+
 *
 * 用例场景：静态图的子图中const/constant/fileconstant连netputout，不需要插入identity
 * 步骤：
 * step 1. 构造纯静态图包含子图场景，子图中constant连netoutput
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：
 * const后不插入identity
 */
TEST_F(MemLayoutConflictTest, ConstInSubgraphConnectToNetoutput_NotInsertIdentity) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildConstantConnectNetoutputSubGraph();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["netoutput1"], nullptr);
    EXPECT_EQ(name_to_node["netoutput1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    // 将已有pass归一后
    //EXPECT_NE(name_to_node["netoutput1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}
/*
*        input             input
 *          |                 |
 *        cast0             cast0
 *          |\                |\
 *          |  \        identity \
 *          |    \            |    \
 *        while1 relu ==>   while1 relu
 *          |     |           |     |
 *        cast1   |         cast1   |
 *          |    /            |    /
 *        netoutput         netoutput
 * 用例场景：while算子的输入节点同时给到多个算子作为输入
 * 步骤：
 * step 1. 构造纯静态图， while算子的输入节点同时给到多个算子作为输入
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：
 * while前插入identity
 */
TEST_F(MemLayoutConflictTest, WhileInNodeConnectToMultiNodes_NotInsertIdentity) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildWhileInNodeConnectToMultiNodesGraph();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["while1"], nullptr);
    EXPECT_EQ(name_to_node["while1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}
/*
 * refdata--hcom--netoutput  ===> refdata--identity--hcom--netoutput
 *
 * 用例场景：refdata连hcom算子，纯静态图中hcom算子不支持地址刷新，而refdata一定要刷新地址才行，
 * 所以期望之间插入identity，这样在图分配时会认为refdata时可以零拷贝的。
 * 步骤：
 * step 1. 构造纯静态图，refdata直连hcom算子
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：
 * hcom前插入identity
 */
TEST_F(MemLayoutConflictTest, RefDataConnectHcom_NotInsertIdentity) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshInputAndRefDataGraph();

  map<AscendString, AscendString> options;
  options[OPTION_FEATURE_BASE_REFRESHABLE] = "0";
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["hcom"], nullptr);
    EXPECT_EQ(name_to_node["hcom"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}
/*
 *                                                                                 data2   data4
 *                                                                                   |       |
 *  data1 const1 var1 const2  data2                      data1 const1 var1 const2  identity identity
 *    \      |    \    /        |                          \      |    \    /        |     /
 *     \     |    assign       hcom2                  identity identity assign       hcom2
 *      \    |     /           /   \                         \    |       |         /   \
 *       \   |   /            /   partitioned_call ==>        \   |    identity    /   partitioned_call
 *         hcom1             /    /  +---------+                hcom1----+        /    /  +---------+
 *           \              /    /   | data3   |                  \              /    /   | data3   |
 *            \            /    /    |   |     |                   \            /    /    |   |     |
 *              \        /    /      |   a     |                     \    identity /      |   a     |
 *               \      /   /        |   |     |                      \      /   /        |   |     |
 *                netoutput          |netoutput1|                      netoutput          |netoutput1|
 *
 * 用例场景：用户输入接连续输入，不可变地址接连续输入，用户输出通过引用接连续输出，连续输出接用户输出场景
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：
 * 用户输入/const,variable后面插入identity
 * 用户输出前面插入identity
 *
 */
TEST_F(MemLayoutConflictTest, UserInOutAndContinuousInOut_InsertIdentity) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildUserInOutConnectContinuousInAndOutGraph();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["hcom1"], nullptr);
    EXPECT_EQ(name_to_node["hcom1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["hcom1"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["hcom1"]->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["netoutput"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["hcom2"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_NE(name_to_node["assign"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);

    auto tensor_move = graph->FindFirstNodeMatchType(TENSORMOVE);
    EXPECT_EQ(tensor_move, nullptr);
  };
}

/*
 *                                                                                 data2
 *                                                                                   |
 *  data1 const1 var1 const2  data2                     data1 const1 var1 const2  identity
 *    \      |    \    /        |                          \      |    \    /        |
 *     \     |    assign     phonysplit               identity identity assign     phonysplit
 *      \    |     /           /   \                         \    |       |         /   \
 *       \   |   /            /   partitioned_call ==>        \   |    identity    /   partitioned_call
 *     phonyconcat1          /    /  +---------+         phonyconcat1----+        /    /  +---------+
 *           \              /    /   | data3   |                  \              /    /   | data3   |
 *            \            /    /    |   |     |                   \            /    /    |   |     |
 *              \        /    /      |   a     |                     \    identity /      |   a     |
 *               \      /   /        |   |     |                      \      /   /        |   |     |
 *                netoutput          |netoutput1|                      netoutput          |netoutput1|
 *
 * 用例场景：用户输入接NoPadding连续输入，不可变地址接NoPadding连续输入，用户输出通过引用接NoPadding连续输出，NoPadding连续输出接用户输出场景
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：
 * phonyconcat1 输入前面插入identity
 * phonysplit output_0 后面插入identity
 * phonysplit 输入插入identity (offline 编译， signle op, load model withq 场景)
 */
TEST_F(MemLayoutConflictTest, UserInOutAndNoPaddingContinuousInOut_InsertIdentity) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildUserInOutConnectNoPaddingContinuousInAndOutGraph();
  std::map<AscendString, AscendString> options = {
    { OPTION_BUILD_GRAPH_MODE, "offline" },
  };

  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["phonyconcat1"], nullptr);
    EXPECT_EQ(name_to_node["phonyconcat1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["phonyconcat1"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["phonyconcat1"]->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["netoutput"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["phonysplit"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_NE(name_to_node["assign"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);

    auto tensor_move = graph->FindFirstNodeMatchType(TENSORMOVE);
    EXPECT_EQ(tensor_move, nullptr);
  };
}
/*
 *   data          data
 *    |             |
 *    a             a
 *    |             |
 *   swt1   ==>   identity
 *    |             |
 *   swt2          swt1
 *    |             |
 *    b           identity
 *    |             |
 *  netoutput     swt2
 *                  |
 *                identity
 *                  |
 *                 netoutput
 *
 * 用例场景：不可刷新地址输出接与不可刷新地址输入间只插入一个identity，不能插入2个identity
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：
 * 只插入一个identity，不能插入2个identity
 */
TEST_F(MemLayoutConflictTest, NotSupportRefreshOutAndNotSupportRefreshIn_InsertIdentity) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshOutAndNotSupportRefreshInGraph();

  map<AscendString, AscendString> options;
  options[OPTION_FEATURE_BASE_REFRESHABLE] = "1";
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["swt1"], nullptr);
    ASSERT_NE(name_to_node["swt2"], nullptr);
    ASSERT_NE(name_to_node["b"], nullptr);
    EXPECT_EQ(name_to_node["swt1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Identity");
    EXPECT_EQ(name_to_node["swt2"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Identity");
    EXPECT_EQ(name_to_node["b"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Identity");
    EXPECT_EQ(name_to_node["swt1"]->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode(),
              name_to_node["swt2"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode());
  };
}
/*
 * data1     data2 data3          data4      data1     data2 data3          data4
 *   |         |     |             |           |         |     |             |
 *   a         b     c             d           a         b     c             d
 *   |         |     |             |           |         |     |             |
 *   +---+     |     +-------+     |   ==>     +---+     |     +-------+     |
 *   |    \    /     |        \    /           |    \    /     |        \    /
 *   |     hcom      |      phonyconcat     identity hcom   identity    phonyconcat
 *   |      |        |          |              |      |        |          |
 *   |      +--+  +--+          |              |      +--+  +--+          |
 *   +-------+ |  | +-----------+              +-------+ |  | +-----------+
 *          0|1| 2|3|                                 0|1| 2|3|
 *           netoutput                                 netoutput
 *
 * 用例场景：用户输出和NoPadding连续输入，用户输出和连续输入
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：
 * netoutput第0 2输入前插入个identity； netoutput第1 3不输入插入identity.
 */
TEST_F(MemLayoutConflictTest, UserOutAndNoPaddingAndContinuousInput_InsertIdentity) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildUserOutAndNoPaddingAndContinuousInputGraph();

  map<AscendString, AscendString> options;
  options[OPTION_FEATURE_BASE_REFRESHABLE] = "1";
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["netoutput"], nullptr);
    EXPECT_EQ(name_to_node["netoutput"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["netoutput"]->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_NE(name_to_node["netoutput"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_NE(name_to_node["netoutput"]->GetInDataAnchor(3)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}
/*
 *                             const     var1        var2
 *                               \         |          /
 * const var1   var2           identity  identity  identity
 *  \   /       |                   \    /          |
 *   hcom     phonysplit ==>         hcom       phonysplit
 *   / \        /  \                  / \        /  \
 *  a   b      c    d                a   b      c    d
 *  |     \  /      |                |     \  /      |
 *  +---netoutput---+                +---netoutput---+
 * 用例场景：不可变地址算子通过输出引用输入和连续输出在同一个符号内
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：
 * 1 校验不可变地址算子后面插入identity
 */
TEST_F(MemLayoutConflictTest, ImmutableOutAndNoPaddingAndContinuousByRef_InsertIdentity) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildImmutableOutAndNoPaddingAndContinuousByRefOutput();

  map<AscendString, AscendString> options;
  options[OPTION_FEATURE_BASE_REFRESHABLE] = "1";
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  size_t identity_cnt = 0U;
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
      if (node->GetType() == IDENTITY) {
        identity_cnt++;
      }
    }
    EXPECT_EQ(identity_cnt, 3U);
    ASSERT_NE(name_to_node["netoutput"], nullptr);
    EXPECT_EQ(name_to_node["hcom"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["hcom"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["phonysplit"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}
/*
 *   swt1  swt3                   swt1     swt3
 *    /\     /\                    |        |
 * swt2 concat swt4 ==>         identity identity
 *    \   |   /                   /\       /\
 *    netoutput            identity  concat  identity
 *                             /       |          \
 *                           swt2      |         swt4
 *                              \      |        /
 *                            identity |  identity
 *                                \    |    /
 *                                 netoutput
       *
 * 用例场景：地址不可刷新算子输入与连续输入场景
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：
 * 1 由于davinci model会校验不可刷新的输入/输出对端只能是identity或memcpy，且identity/memcpy的输出anchor只能有一个输出
 * 给到不可刷新算子输入上，否则davinci model不知道将fix地址刷到哪个算子上。所以dsa1 dsa2间会有两个identity.
 * 2  不能插入多余的identity.
 */
TEST_F(MemLayoutConflictTest, NotSupportRefreshInputAndContinuousInput_InsertIdentity) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildNotSupportRefreshInputAndContinuousInputGraph();

  map<AscendString, AscendString> options;
  options[OPTION_FEATURE_BASE_REFRESHABLE] = "1";
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  size_t identity_cnt = 0U;
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
      if (node->GetType() == IDENTITY) {
        identity_cnt++;
      }
    }
    EXPECT_EQ(identity_cnt, 6U);
    ASSERT_NE(name_to_node["netoutput"], nullptr);
    EXPECT_EQ(name_to_node["concat"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["concat"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["swt2"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["swt4"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["netoutput"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["netoutput"]->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_NE(name_to_node["netoutput"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}
/*
 *     a     b             a                b
 *     /\    /\            /\               /\
 *    |  hcom  |          | identity identity |
 *     \       /           \      \ /        /
 *   c  concat2  d ==>      \     hcom     /
 *    \   |     /             \    |     /
 *      concat3                  concat2
 *        |                        |
 *      netoutput              c identity d
 *                              \   |    /
 *                               concat3
 *                                  |
 *                               netoutput
* 用例场景：连续输入与连续输入在同一个符号内的两个场景构成一个图
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：
 * 1 插入3个identity, 2 位置见校验点
 */
TEST_F(MemLayoutConflictTest, ContinuousInAndContinuousIn_InsertIdentity) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildContinuousInAndContinuousInMixGraph();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  size_t identity_cnt = 0U;
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
      if (node->GetType() == IDENTITY) {
        identity_cnt++;
      }
    }
    EXPECT_EQ(identity_cnt, 3U);
    ASSERT_NE(name_to_node["hcom"], nullptr);
    ASSERT_NE(name_to_node["concat2"], nullptr);
    ASSERT_NE(name_to_node["hcom"]->GetInDataAnchor(1), nullptr);
    ASSERT_NE(name_to_node["concat2"]->GetInDataAnchor(1), nullptr);
    EXPECT_EQ(name_to_node["hcom"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["hcom"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);

    EXPECT_EQ(name_to_node["concat2"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "a");
    EXPECT_EQ(name_to_node["concat2"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "b");
  };
}
/*
 *      a     b             a                b
 *      /\    /\            /\               /\
 *     |  hcom  |          | identity identity |
 *      \       /           \      \ /        /
 *   c phonyconcat d ==>      \     hcom     /
 *    \    |      /             \          /
 *       concat3                phonyconcat
 *         |                        |
 *      netoutput              c identity d
 *                              \   |    /
 *                               concat3
 *                                  |
 *                               netoutput
* 用例场景：连续输入与NoPadding连续输入在同一个符号内的两个场景构成一个图
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：
 * 1 插入3个identity, 2 位置见校验点
 */
TEST_F(MemLayoutConflictTest, ContinuousInAndNoPaddingContinuousIn_InsertIdentity) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildContinuousInAndNoPaddingContinuousInMixGraph();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  size_t identity_cnt = 0U;
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
      if (node->GetType() == IDENTITY) {
        identity_cnt++;
      }
    }
    EXPECT_EQ(identity_cnt, 3U);
    ASSERT_NE(name_to_node["hcom"], nullptr);
    ASSERT_NE(name_to_node["phonyconcat"], nullptr);
    ASSERT_NE(name_to_node["hcom"]->GetInDataAnchor(1), nullptr);
    ASSERT_NE(name_to_node["phonyconcat"]->GetInDataAnchor(1), nullptr);
    EXPECT_EQ(name_to_node["hcom"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["hcom"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["concat3"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);

    EXPECT_EQ(name_to_node["concat3"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "c");
    EXPECT_EQ(name_to_node["concat3"]->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "d");
  };
}
/*
 *  a  split
 *  \   / \
 *  concat  b
 */
TEST_F(MemLayoutConflictTest, ContinuousInAndContinuousOut_InsertIdentity) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildContinuousInAndContinuousOutGraph();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  size_t identity_cnt = 0U;
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
      if (node->GetType() == IDENTITY) {
        identity_cnt++;
      }
    }
    EXPECT_EQ(identity_cnt, 0U);
    ASSERT_NE(name_to_node["concat"], nullptr);
    ASSERT_NE(name_to_node["concat"]->GetInDataAnchor(1), nullptr);
    EXPECT_EQ(name_to_node["concat"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), "Split");

    EXPECT_EQ(name_to_node["concat"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "a");
  };
}
/*
 *   var   hcom1         var           hcom1
 *     \   /  \           |            /   \
 *     hcom2   c  ==>  identity identity    c
 *     /   \                 \   /
 *    a     b                hcom2
 *                           /   \
 *                          a     b
 * 用例场景：连续输出与连续输出，及不可变地址输出与连续输出的场景
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * VariablePrepareOpPass::DealWritableNode 会给hcom2_out_0打上 ref_var_src_var_name属性，
 * 在Checker::IsSkip中会跳过，所以增加了一个特殊处理。
 * step 3. 校验
 * 期望：
 * 1 插入2个identity, 2 位置见校验点
 */
TEST_F(MemLayoutConflictTest, ContinuousOutAndContinuousOutHcomByRef_InsertIdentity) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildContinuousOutAndContinuousOutHcomByRefGraph();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  size_t identity_cnt = 0U;
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
      if (node->GetType() == IDENTITY) {
        identity_cnt++;
      }
    }
    EXPECT_EQ(identity_cnt, 2U);
    ASSERT_NE(name_to_node["hcom2"], nullptr);
    ASSERT_NE(name_to_node["hcom2"]->GetInDataAnchor(1), nullptr);
    EXPECT_EQ(name_to_node["hcom2"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["hcom2"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);

    auto tensor_move = graph->FindFirstNodeMatchType(TENSORMOVE);
    EXPECT_EQ(tensor_move, nullptr);
  };
}
/*
 *        split              split
 *         /  \               /  \
 * phony_plit  c ==>    identity  c
 *     /   \               |
 *    a     b           phony_plit
 *                        /   \
 *                       a     b
 * 用例场景：连续输出与NoPadding连续输出的场景
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：
 * 1 插入1个identity, 2 位置见校验点
 */
TEST_F(MemLayoutConflictTest, ContinuousOutAndNoPaddingContinuousOutByRef_InsertIdentity) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildContinuousOutAndNoPaddingContinuousOutByRefGraph();

  map<AscendString, AscendString> options;
  options[OPTION_FEATURE_BASE_REFRESHABLE] = "1";
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  size_t identity_cnt = 0U;
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
      if (node->GetType() == IDENTITY) {
        identity_cnt++;
      }
    }
    EXPECT_EQ(identity_cnt, 1U);
    ASSERT_NE(name_to_node["phony_plit"], nullptr);
    ASSERT_NE(name_to_node["phony_plit"]->GetInDataAnchor(0), nullptr);
    EXPECT_EQ(name_to_node["phony_plit"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}
/*
 *      a           b             a           b
 *      /\          /\            /\          /\
 *     | phonyconcat1 |          | phonyconcat1 |
 *      \            /            \            /
 *    c  phonyconcat2  d  ==>   c  phonyconcat2  d
 *     \      |       /          \      |       /
 *       phonyconcat3              phonyconcat3
 *            |                         |
 *         netoutput                 netoutput
 *
 * 用例场景：NoPadding连续输入与NoPadding连续输入，两个节点输入完全相同不冲突，级联场景不认为时冲突。
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：不插入identity
 */
TEST_F(MemLayoutConflictTest, NoPaddingContinuousInAndNoPaddingContinuousInMix_NotInsertIdentity) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousInMixGraph();

  map<AscendString, AscendString> options;
  options[OPTION_FEATURE_BASE_REFRESHABLE] = "1";
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  size_t identity_cnt = 0U;
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
      if (node->GetType() == IDENTITY) {
        identity_cnt++;
      }
    }
    EXPECT_TRUE(identity_cnt == 0U);
  };
}
/*
 *                                                  b
 *                                                  /\
 *   a          b                    a       identity \
 *    \         /\                    \         /      \
 *   phonyconcat phonysplit  ==>     phonyconcat  phonysplit
 *       |         / \                   |          / \
 *       c        d   e                  c         d   e
 *        \      /   /                    \       /   /
 *         netoutput                       netoutput
 * 用例场景：NoPadding连续输入与NoPadding连续输出
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：
 * NoPadding连续输入前插入1个identity
 */
TEST_F(MemLayoutConflictTest, NoPaddingContinuousInAndNoPaddingContinuousOut_InsertIdentity) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousOutByRefGraph();

  map<AscendString, AscendString> options;
  options[OPTION_FEATURE_BASE_REFRESHABLE] = "1";
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  size_t identity_cnt = 0U;
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
      if (node->GetType() == IDENTITY) {
        identity_cnt++;
      }
    }
    EXPECT_EQ(identity_cnt, 1U);
    ASSERT_NE(name_to_node["phonyconcat"], nullptr);
    ASSERT_NE(name_to_node["phonyconcat"]->GetInDataAnchor(1), nullptr);
    EXPECT_EQ(name_to_node["phonyconcat"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}
/*
 *   phony_split1         phony_split1
 *         /  \                 /  \
 * phony_plit2  c ==>     identity  c
 *     /   \                /
 *    a     b       phony_plit2
 *                      /   \
 *                     a     b
 * 用例场景：NoPadding连续输出与NoPadding连续输出
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：
 * phony_plit2前插入1个identity
 */
TEST_F(MemLayoutConflictTest, NoPaddingContinuousOutAndNoPaddingContinuousOut_InsertIdentity) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousOutAndNoPaddingContinuousOutGraph();

  map<AscendString, AscendString> options;
  options[OPTION_FEATURE_BASE_REFRESHABLE] = "1";
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  size_t identity_cnt = 0U;
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
      if (node->GetType() == IDENTITY) {
        identity_cnt++;
      }
    }
    EXPECT_EQ(identity_cnt, 1U);
    ASSERT_NE(name_to_node["phony_plit2"], nullptr);
    ASSERT_NE(name_to_node["phony_plit2"]->GetInDataAnchor(0), nullptr);
    EXPECT_EQ(name_to_node["phony_plit2"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}
/*
 *                         b
 *                         |
 *      b                 hcom1            c
 *      |                  |              /\
 *   a hcom1 c          a identity identity f
 *    \ |   /\           \   |    /         /
 *     hcom2  f ==>         hcom2          e
 *      |      |             |           /
 *      d      e             d         /
 *       \    /               \      /
 *       netoutput            netoutput
 *
 * hcom1: 需要p2p内存，且输出引用输入
 * hcom2:连续输入
 * c: 一个输出给2个人
 * switch: 需要ts内存
 *
 * 用例场景：连续输入需要普通内存，但是其两个输入上需要特殊类型内存
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：
 * 插入identity将特殊内存与连续输入内存隔绝开
 */
TEST_F(MemLayoutConflictTest, ContinuousInAndRtsSpecailInOut_CheckMemType_SUCCESS) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildContinuousInAndRtsSpecailInOutGraph();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  size_t identity_cnt = 0U;
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
      if (node->GetType() == IDENTITY) {
        identity_cnt++;
      }
    }
    EXPECT_EQ(identity_cnt, 2U);
    ASSERT_NE(name_to_node["hcom2"], nullptr);
    ASSERT_NE(name_to_node["hcom2"]->GetInDataAnchor(0), nullptr);
    EXPECT_EQ(name_to_node["hcom2"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["hcom2"]->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);

    auto tensor_move = graph->FindFirstNodeMatchType(TENSORMOVE);
    EXPECT_EQ(tensor_move, nullptr);
  };
}
/*
 *                      b
 *                      |
 *      b             identity
 *      |               |
 *  a  hcom1 ==>    a  hcom1
 *   \ /             \ /
 *   hcom2           hcom2
 *
 *  b:ts mem type
 *  hcom1: out ref input, p2p input and output
 *  hcom2: continuous input, p2p input
 *
 * 用例场景：连续输入需要p2p内存，其输入上存在p2p和ts两种特殊类型内存
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：
 * 插入identity将两种不同的特殊内存隔绝开
 * 连续输入上不插入identity
 */
TEST_F(MemLayoutConflictTest, ContinuousInAndRtsSpecailInOutSameMemType_CheckMemType_SUCCESS) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildContinuousInAndRtsSpecailInOutSameMemTypeGraph();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  size_t identity_cnt = 0U;
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
      if (node->GetType() == IDENTITY) {
        identity_cnt++;
      }
    }
    EXPECT_EQ(identity_cnt, 1U);
    ASSERT_NE(name_to_node["hcom1"], nullptr);
    ASSERT_NE(name_to_node["hcom1"]->GetInDataAnchor(0), nullptr);
    EXPECT_EQ(name_to_node["hcom1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);

    auto tensor_move = graph->FindFirstNodeMatchType(TENSORMOVE);
    EXPECT_EQ(tensor_move, nullptr);
  };
}

/*
 *   var  const         var  const
 *    \    /             \    /
 *    assign             assign
 *      |       ==>        |
 *  hcombroadcast        identity
 *      |                  |
 *      a              hcombroadcast
 *      |                  |
 *    netoutput            a
 *                         |
 *                       netoutput
 *
 * 用例场景：variable-assign-rts输出内存，校验只在assign后面插入identity，不在variable后面插入。
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：校验只在assign后面插入tensormove，不在variable后面插入。
 */
TEST_F(MemLayoutConflictTest, ImmutableOutAndRtsSpecailInByAssign_Insert_SUCCESS) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildImmutableOutAndRtsSpecailInByAssignGraph();

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  size_t identity_cnt = 0U;
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
      if (node->GetType() == IDENTITY) {
        identity_cnt++;
      }
    }
    EXPECT_EQ(identity_cnt, 1U);
    ASSERT_NE(name_to_node["hcombroadcast"], nullptr);
    ASSERT_NE(name_to_node["hcombroadcast"]->GetInDataAnchor(0), nullptr);
    EXPECT_EQ(name_to_node["hcombroadcast"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}
/*
 *    a    b
 *     \   /
 *  PhonyConcat
 *       |
 *   PhonySplit
 *      /  \
 *     c    d
 * 用例场景：需要连续输入的节点，直连需要来连续输出的节点。不冲突，不插入identity
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：不插入identity
 */
TEST_F(MemLayoutConflictTest, NoPaddingContinuousInAndNoPaddingContinuousOutConnect_NotInsert_SUCCESS) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousOutConnectGraph();
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  size_t identity_cnt = 0U;
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
      if (node->GetType() == IDENTITY) {
        identity_cnt++;
      }
    }
    EXPECT_TRUE(identity_cnt == 0U);
  };
}

/*
 *   a            b          a                 b
 *   /\1         0/\         /\                /\
 *  | phony_concat1 |  ==>  |   \            /   |
 *  \0            1/        \     \1       0/     |
 *    phony_concat2    identity  phony_concat1  identity
 *                            \0              1/
 *                               phony_concat2
 *
 * 用例场景：NoPadding连续输入与NoPadding连续输入，两个节点输入顺序不相同，认为是冲突。
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：任意一个phonyconcat插入2 identity
 */
TEST_F(MemLayoutConflictTest, NoPaddingContinuousInAndNoPaddingContinuousOutConnectCross_Insert_SUCCESS) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousInCrossGraph();
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  size_t identity_cnt = 0U;
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
      if (node->GetType() == IDENTITY) {
        identity_cnt++;
      }
    }
    EXPECT_EQ(identity_cnt, 2U);
    bool check_success = false;
    if ((name_to_node["phony_concat1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType() == IDENTITY)
        || (name_to_node["phony_concat1"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType() == IDENTITY)
        ||(name_to_node["phony_concat2"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType() == IDENTITY)
        || (name_to_node["phony_concat2"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType() == IDENTITY)) {
      check_success = true;
    }
    EXPECT_TRUE(check_success);
  };
}
/*
 *    var  const            var  const
 *     \    /                \    /
 *     assign   c            assign
 *       |     /               |
 *   hcombroadcast  ==>     identity   c
 *       / \                   |      /
 *      a   b               hcombroadcast (need p2p in/out, continuous in/out, out ref in)
 *      |   |                   / \
 *     netoutput               a   b
 *                             |   |
 *                            netoutput
 * 用例场景：variable-assign-连连续内存输出/rts输出内存输出/连续内存输入，校验只在assign后面插入identity，不在variable后面插入。
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：校验只在assign后面插入tensormove，不在variable后面插入。
 */
TEST_F(MemLayoutConflictTest, ImmutableOutAnRtsSpecailOutContinuousInOutByAssign_Insert_SUCCESS) {
  auto infer_fun = [](Operator &op) -> graphStatus {
    for (size_t i = 0U; i < op.GetOutputsSize(); i++) {
      auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
      op_desc->MutableOutputDesc(i)->SetShape(GeShape(vector<int64_t>{1, 1, 224, 224}));
      op_desc->MutableOutputDesc(i)->SetOriginShape(GeShape(vector<int64_t>{1, 1, 224, 224}));
    }
    for (size_t i = 0U; i < op.GetInputsSize(); i++) {
      auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
      op_desc->MutableInputDesc(i)->SetShape(GeShape(vector<int64_t>{1, 1, 224, 224}));
      op_desc->MutableInputDesc(i)->SetOriginShape(GeShape(vector<int64_t>{1, 1, 224, 224}));
    }
    return GRAPH_SUCCESS;
  };
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildImmutableOutAnRtsSpecailOutContinuousInOutByAssignGraph();
  auto hcombroadcast = graph->FindNode("hcombroadcast");
  hcombroadcast->GetOpDesc()->AddInferFunc(infer_fun);
  auto c_node = graph->FindNode("hcombroadcast");
  hcombroadcast->GetOpDesc()->AddInferFunc(infer_fun);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  size_t identity_cnt = 0U;
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
      if (node->GetType() == IDENTITY) {
        identity_cnt++;
      }
    }
    EXPECT_EQ(identity_cnt, 1U);
    EXPECT_EQ(name_to_node["hcombroadcast"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
}
/*
 *    op1
 *     +--hcombroadcast (p2p input)
 *     |    |
 *   netoutput
 * 用例场景：用户输出和rts特殊输入/输出输出内存，需要插入identity
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：在netoutput节点前面插入identity
 */
TEST_F(MemLayoutConflictTest, UserOutAndRtsSpecialIn_Insert_SUCCESS) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildUserOutAndRtsSpecialInGraph();
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  size_t identity_cnt = 0U;
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
      if (node->GetType() == IDENTITY) {
        identity_cnt++;
      }
    }
    EXPECT_EQ(identity_cnt, 2U);
    EXPECT_EQ(name_to_node["netoutput"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
    EXPECT_EQ(name_to_node["netoutput"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);

    auto tensor_move = graph->FindFirstNodeMatchType(TENSORMOVE);
    EXPECT_EQ(tensor_move, nullptr);
  };
}
/*
 *     a b c d      a d
 *     | | | |      | |
 *   phonyconcat1 phonyconcat2
 *
 * 用例场景：两个需要NoPadding连续输入的算子拥有部分相同输入，中间有间隔，属于冲突场景
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：在其中一个需要连续输入的节点前面插入identity
 */
TEST_F(MemLayoutConflictTest, NoPaddingContinuousInWithParialSameInputs_Insert_SUCCESS) {
  DUMP_GRAPH_WHEN("PreRunAfterMemConflictProc");
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousInAndNoPaddingContinuousInPartialSameInputsConflictGraph();
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);
  size_t identity_cnt = 0U;
  CHECK_GRAPH(PreRunAfterMemConflictProc) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
      if (node->GetType() == IDENTITY) {
        identity_cnt++;
      }
    }
    EXPECT_EQ(identity_cnt, 2U);
    bool check_pass = false;
    if (name_to_node["phony_concat2"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType() == IDENTITY
        || name_to_node["phony_concat2"]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType() == IDENTITY
        || name_to_node["phony_concat1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType() == IDENTITY
        || name_to_node["phony_concat1"]->GetInDataAnchor(3)->GetPeerOutAnchor()->GetOwnerNode()->GetType() == IDENTITY) {
      check_pass = true;
    }
    EXPECT_TRUE(check_pass);
  };
}

REG_OP(Data)
    .INPUT(data, TensorType::ALL())
    .OUTPUT(out, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)

REG_OP(Cast)
  .INPUT(x, TensorType::BasicType())
  .OUTPUT(y, TensorType::BasicType())
  .REQUIRED_ATTR(dst_type, Int)
  .OP_END_FACTORY_REG(Cast)

REG_OP(Split)
    .INPUT(split_dim, TensorType({DT_INT32}))
    .INPUT(x, TensorType::BasicType())
    .DYNAMIC_OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(Split)

REG_OP(Relu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE,
                          DT_INT8, DT_INT32, DT_INT16, DT_INT64,
                          DT_UINT8, DT_UINT16, DT_QINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE,
                            DT_INT8, DT_INT32, DT_INT16, DT_INT64,
                            DT_UINT8, DT_UINT16, DT_QINT8}))
    .OP_END_FACTORY_REG(Relu)

/*
                    ┌────────┐  (0,2)
                    │   b    │ ──────────────────────────┐
                    └────────┘                           │
                      ∧                                  │
                      │ (1,0)                            │
                      │                                  ∨
┌────────┐  (0,0)   ┌────────┐  (0,0)   ┌───┐  (0,1)   ┌───────────┐
│ split1 │ ───────> │ split2 │ ───────> │ a │ ───────> │ netoutput │
└────────┘          └────────┘          └───┘          └───────────┘
  │                                                      ∧
  │ (1,0)                                                │
  ∨                                                      │
┌────────┐  (0,0)                                        │
│   c    │ ──────────────────────────────────────────────┘
└────────┘
* a是inplace算子，split2是输出连续，这样会导致用户输出复用split2的输出，导致内存冲突
*/
TEST_F(MemLayoutConflictTest, Inplace_Continous_input_conflict) {
  auto graph = ge::MemConflictShareGraph::BuildContinuousOutGraph();
  auto node = graph->FindNode("a");
  auto op_desc = node->GetOpDesc();
  AttrUtils::SetListListInt(op_desc, ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}});

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(0, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);
  bool reuse_flag;
  (void)TensorUtils::GetReuseInput(op_desc->GetOutputDesc(0), reuse_flag);
  ASSERT_FALSE(reuse_flag);
}

/*
                    ┌────────┐  (0,2)
                    │   b    │ ──────────────────────────┐
                    └────────┘                           │
                      ∧                                  │
                      │ (1,0)                            │
                      │                                  ∨
┌────────┐  (0,0)   ┌────────┐  (0,0)   ┌───┐  (0,1)   ┌───────────┐
│ split1 │ ───────> │ split2 │ ───────> │ a │ ───────> │ netoutput │
└────────┘          └────────┘          └───┘          └───────────┘
  │                                                      ∧
  │ (1,0)                                                │
  ∨                                                      │
┌────────┐  (0,0)                                        │
│   c    │ ──────────────────────────────────────────────┘
└────────┘
* a是inplace算子，split2是不是输出连续，可以inplace
*/
TEST_F(MemLayoutConflictTest, Inplace_Success_01) {
  auto graph = ge::MemConflictShareGraph::BuildNotContinuousOutGraph();
  auto node = graph->FindNode("a");
  auto op_desc = node->GetOpDesc();
  AttrUtils::SetListListInt(op_desc, ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}});

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(0, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);
  bool reuse_flag;
  (void)TensorUtils::GetReuseInput(op_desc->GetOutputDesc(0), reuse_flag);
  ASSERT_TRUE(reuse_flag);
}

TEST_F(MemLayoutConflictTest, Inplace_tensor_size_not_equal) {
  auto graph = ge::MemConflictShareGraph::BuildNotContinuousOutGraph();
  auto node = graph->FindNode("a");
  auto op_desc = node->GetOpDesc();
  AttrUtils::SetListListInt(op_desc, ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}});
  op_desc->UpdateOutputDesc(0, GeTensorDesc(GeShape({1, 2, 3, 4}), FORMAT_NCHW, DT_FLOAT));

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(0, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);
  bool reuse_flag;
  (void)TensorUtils::GetReuseInput(op_desc->GetOutputDesc(0), reuse_flag);
  ASSERT_FALSE(reuse_flag);
}

/*
┌────────┐  (1,0)   ┌───┐  (0,0)   ┌────────┐  (0,0)   ┌───────────┐
│ split1 │ ───────> │ c │ ───────> │ while1 │ ───────> │ netoutput │ <┐
└────────┘          └───┘          └────────┘          └───────────┘  │
  │                                                      ∧            │
  │ (0,0)                                                │            │
  ∨                                                      │            │
┌────────┐  (0,0)   ┌───┐  (0,1)                         │            │
│ split2 │ ───────> │ a │ ───────────────────────────────┘            │ (0,2)
└────────┘          └───┘                                             │
  │                                                                   │
  │ (1,0)                                                             │
  ∨                                                                   │
┌────────┐                                                            │
│   b    │ ───────────────────────────────────────────────────────────┘
└────────┘

               g

┌───────┐  (0,0)   ┌────────────┐
│ data0 │ ───────> │ netoutput0 │
└───────┘          └────────────┘

                           g

┌───────┐  (0,0)   ┌────────────┐  (0,0)   ┌────────────┐
│ data1 │ ───────> │ while_mul1 │ ───────> │ netoutput1 │
└───────┘          └────────────┘          └────────────┘
* while循环可以inpalce
*/
TEST_F(MemLayoutConflictTest, Inplace_While_Success) {
  auto graph = ge::MemConflictShareGraph::BuildWhileSplitGraph();
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(0, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);
  bool reuse_flag;
  (void)TensorUtils::GetReuseInput(graph->FindNode("acast")->GetOpDesc()->GetOutputDesc(0), reuse_flag);
  ASSERT_TRUE(reuse_flag);
}

/*
 *   refdata const            refdata const
 *       \   /                    \   /
 *      assign       ==>        assign
 *       |                         |
 *      hcom need p2p output       identity
 *                                  |
 *                                 hcom need p2p output
 */
TEST_F(MemLayoutConflictTest, RefdataAndRtsSpecialOut_InsertIdentity_Success) {
  auto graph = MemConflictShareGraph::BuildUserInRefDataAndRtsSpecialOutGraph();
  MemLayoutConflictOptimizer mem_check_pass;
  ASSERT_EQ(mem_check_pass.Run(graph), GRAPH_SUCCESS);
}
} // namespace ge

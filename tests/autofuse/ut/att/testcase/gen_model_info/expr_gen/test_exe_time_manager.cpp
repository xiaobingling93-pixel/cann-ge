/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include "gtest/gtest.h"
#include "test_brc_buf_graph.h"
#include "parser/ascend_graph_parser.h"
#include "expr_gen/arg_list_manager.h"
#include "expr_gen/exe_time_pass.h"
#include "graph_construct_utils.h"

namespace att {
class TestExeTimePass : public ::testing::Test {
 public:
  static void TearDownTestCase()
  {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase()
  {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override
  {
  }
  void TearDown() override
  {
  }
};

void GetExeTime(const TuningSpacePtr &tuning_space, TernaryOp &exe_cast0, TernaryOp &exe_store) {
  ExeTimePassManager exe_mgr(tuning_space);
  Expr exe_time;
  for (const auto &node : tuning_space->node_infos) {
    exe_time = CreateExpr(1U);
    for (auto &loop_axis : node.loop_axes) {
        exe_time = ge::sym::Mul(exe_time, loop_axis->repeat);
    }
    if (node.name == "cast0") {
        exe_cast0 = exe_mgr.UpdateNodeExeTime(node, exe_time);
    } else if (node.name == "store") {
        exe_store = exe_mgr.UpdateNodeExeTime(node, exe_time);
    }
  }
}

TEST_F(TestExeTimePass, case1)
{
  ge::AscGraph graph("graph");
  att::BrcBufBeforeAutoFuse1(graph);
  att::BrcBufAfterScheduler1(graph);
  att::BrcBufAfterQueBufAlloc1(graph);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  auto ret = ascend_graph_parser.GraphParser(graph);
  TernaryOp exe_time_cast0;
  TernaryOp exe_time_store;
  GetExeTime(tuning_space, exe_time_cast0, exe_time_store);
  EXPECT_EQ(exe_time_cast0.GetTernaryOpStr(), "z0z2Tb_size");
  EXPECT_EQ(exe_time_store.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
}

TEST_F(TestExeTimePass, case2)
{
  ge::AscGraph graph("graph");
  att::BrcBufBeforeAutoFuse2(graph);
  att::BrcBufAfterScheduler1(graph);
  att::BrcBufAfterQueBufAlloc1(graph);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  auto ret = ascend_graph_parser.GraphParser(graph);
  TernaryOp exe_time_cast0;
  TernaryOp exe_time_store;
  GetExeTime(tuning_space, exe_time_cast0, exe_time_store);
  EXPECT_EQ(exe_time_cast0.GetTernaryOpStr(), "TernaryOp(IsEqual(z0z2Tb_size, 1.0), 1, (Ceiling((Z1 / (z1t_size))) * z0z2Tb_size))");
  EXPECT_EQ(exe_time_store.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
}

TEST_F(TestExeTimePass, case3)
{
  ge::AscGraph graph("graph");
  att::BrcBufBeforeAutoFuse3(graph);
  att::BrcBufAfterScheduler3(graph);
  att::BrcBufAfterQueBufAlloc3(graph);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  auto ret = ascend_graph_parser.GraphParser(graph);
  TernaryOp exe_time_cast0;
  TernaryOp exe_time_store;
  GetExeTime(tuning_space, exe_time_cast0, exe_time_store);
  EXPECT_EQ(exe_time_cast0.GetTernaryOpStr(), "z0z2Tb_size");
  EXPECT_EQ(exe_time_store.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
}

TEST_F(TestExeTimePass, case4)
{
  ge::AscGraph graph("graph");
  att::BrcBufBeforeAutoFuse4(graph);
  att::BrcBufAfterScheduler4(graph);
  att::BrcBufAfterQueBufAlloc4(graph);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  auto ret = ascend_graph_parser.GraphParser(graph);
  TernaryOp exe_time_cast0;
  TernaryOp exe_time_store;
  GetExeTime(tuning_space, exe_time_cast0, exe_time_store);
  EXPECT_EQ(exe_time_cast0.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
  EXPECT_EQ(exe_time_store.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
}

TEST_F(TestExeTimePass, exec_condition) {
  ge::AscGraph graph("graph");
  att::BrcBufBeforeAutoFuse3(graph);
  att::BrcBufAfterScheduler3(graph);
  att::BrcBufAfterQueBufAlloc3(graph);
  auto load = graph.FindNode("cast0");
  load->attr.sched.exec_condition = ge::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  auto ret = ascend_graph_parser.GraphParser(graph);
  TernaryOp exe_time_cast0;
  TernaryOp exe_time_store;
  GetExeTime(tuning_space, exe_time_cast0, exe_time_store);
  EXPECT_EQ(exe_time_cast0.GetTernaryOpStr(), "Max(1, (Ceiling((Z1 / (z1t_size))) * z0z2Tb_size / (Ceiling((Z2 / (z2t_size))))))");
  EXPECT_EQ(exe_time_store.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
}
}

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "ascendc_ir.h"
#include "ascir_ops_utils.h"
#include "ascir_utils.h"
#include "asc_graph_utils.h"
#include "task_generator/recompute_case_generator.h"
#include "asc_graph_builder.h"
#include "ascgraph_info_complete.h"

namespace schedule {
using namespace optimize;
using namespace ge;
using namespace ge::ops;
using ge::testing::AscGraphBuilder;
using ge::testing::Sym;

class RecomputeCaseGeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
};

TEST_F(RecomputeCaseGeneratorTest, TestDynamicGraphSplit) {
  auto s0 = Sym("s0"), s1 = Sym("s1"), s2 = Sym("s2"), s3 = Sym("s3"), s4 = Sym("s4");
  auto graph = AscGraphBuilder("brc_abs")
    .Loops({s0, s1, s2, s3, s4})
    .Data("RE_data0", 0, ge::DT_FLOAT16)
    .Load("RE_load0", "RE_data0", {One, One, One, s3, s4})
    .Abs("RE_abs", "RE_load0")
    .Broadcast("brc1", "RE_abs", {s0, s1, s2, s3, s4})
    .Store("store", "brc1")
    .Output("out", "store", 0)
    .Abs("ST_abs2", "RE_abs")
    .Store("ST_store1", "ST_abs2")
    .Output("ST_out1", "ST_store1", 1)
    .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), ge::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 2UL);
}

TEST_F(RecomputeCaseGeneratorTest, TestDynamicGraphSplitTwoLine) {
  auto s0 = Sym("s0"), s1 = Sym("s1"), s2 = Sym("s2"), s3 = Sym("s3"), s4 = Sym("s4");
  auto graph = AscGraphBuilder("brc_abs")
    .Loops({s0, s1, s2, s3, s4})
    .Data("RE_data0", 0, ge::DT_FLOAT16)
    .Load("RE_load0", "RE_data0", {One, One, One, s3, s4})
    .Abs("RE_abs", "RE_load0")
    .Broadcast("brc1", "RE_abs", {s0, s1, s2, s3, s4})
    .Store("store", "brc1")
    .Output("out", "store", 0)
    .Abs("ST_abs2", "RE_abs")
    .Store("ST_store1", "ST_abs2")
    .Output("ST_out1", "ST_store1", 1)
    .Abs("ST_abs3", "ST_abs2")
    .Abs("ST_abs4", "ST_abs3")
    .Abs("ST_abs5", "ST_abs4")
    .Abs("ST_abs6", "ST_abs5")
    .Store("ST_store2", "ST_abs6")
    .Output("ST_out2", "ST_store2", 2)
    .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), ge::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 2UL);
}

TEST_F(RecomputeCaseGeneratorTest, TestStaticGraphSplitTwoLine) {
  auto s0 = Sym(1024), s1 = Sym(256);
  auto graph = AscGraphBuilder("brc_abs")
    .Loops({s0, s1})
    .Data("RE_data0", 0, ge::DT_FLOAT16)
    .Load("RE_load0", "RE_data0", {One, s1})
    .Abs("RE_abs", "RE_load0")
    .Broadcast("brc1", "RE_abs", {s0, s1})
    .Store("store", "brc1")
    .Output("out", "store", 0)
    .Abs("ST_abs2", "RE_abs")
    .Store("ST_store1", "ST_abs2")
    .Output("ST_out1", "ST_store1", 1)
    .Abs("ST_abs3", "ST_abs2")
    .Abs("ST_abs4", "ST_abs3")
    .Abs("ST_abs5", "ST_abs4")
    .Abs("ST_abs6", "ST_abs5")
    .Store("ST_store2", "ST_abs6")
    .Output("ST_out2", "ST_store2", 2)
    .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), ge::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 2UL);
}

TEST_F(RecomputeCaseGeneratorTest, TestStaticGraphSplitWithBrc) {
  auto s0 = Sym(1024), s1 = Sym(32), s2 = Sym(32);
  auto graph = AscGraphBuilder("brc_abs")
    .Loops({s0, s1, s2})
    .Data("RE_data0", 0, ge::DT_FLOAT16)
    .Load("RE_load0", "RE_data0", {One, One, s2})
    .Abs("RE_abs", "RE_load0")
    .Broadcast("brc1", "RE_abs", {One, s1, s2})
    .Broadcast("brc2", "brc1", {s0, s1, s2})
    .Relu("relu0", "brc2")
    .Relu("relu1", "relu0")
    .Relu("relu2", "relu1")
    .Store("store", "relu2")
    .Output("out", "store", 0)
    .Abs("ST_abs2", "brc1")
    .Exp("ST_exp", "brc1")
    .Add("ST_add", "ST_abs2", "ST_exp")
    .Store("ST_store1", "ST_add")
    .Output("ST_out1", "ST_store1", 1)
    .Abs("ST_abs3", "ST_abs2")
    .Abs("ST_abs4", "ST_abs3")
    .Abs("ST_abs5", "ST_abs4")
    .Abs("ST_abs6", "ST_abs5")
    .Store("ST_store2", "ST_abs6")
    .Output("ST_out2", "ST_store2", 2)
    .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), ge::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 2UL);
}
}  // namespace schedule

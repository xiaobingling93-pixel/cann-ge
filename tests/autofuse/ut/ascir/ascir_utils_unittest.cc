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
#include "graph/symbolizer/symbolic.h"

#include "ascendc_ir.h"
#include "ascir_utils.h"
#define private public
#include "asc_graph_dumper_context.h"
#undef private

class AscendGraphDumpUT : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {
    unsetenv("AUTOFUSE_DFX_FLAGS");
    ::ascir::utils::ResetDumpConfig();
    system("rm -rf ./ge_onnx* ./autofuse_compile_debug");
  }
};

namespace ge {
TEST_F(AscendGraphDumpUT, test_dump_when_env_not_set) {
  AscGraph graph("test");
  ::ascir::utils::DumpGraph(graph, "empty_stage0");

  AscGraph graph1("test1");
  ::ascir::utils::DumpGraph(graph1, "empty1_stage0");
  ::ascir::utils::DumpGraph(graph, "empty_stage1");
  ::ascir::utils::DumpGraph(graph1, "empty1_stage1");
  ::ascir::utils::DumpGraph(graph, "empty_stage2");

  auto &watched_graphs = ::ascir::AscGraphDumperContext::GetThreadLocalCtx().orderd_graphs_;
  ASSERT_EQ(watched_graphs.size(), 2U);

  EXPECT_EQ(watched_graphs.front().first, "empty1_stage1");
  EXPECT_EQ(watched_graphs.front().second.GetName(), graph1.GetName());
  EXPECT_EQ(watched_graphs.back().first, "empty_stage2");
  EXPECT_EQ(watched_graphs.back().second.GetName(), graph.GetName());
  ::ascir::AscGraphDumperContext::GetThreadLocalCtx().DumpWatchedGraphs();
  EXPECT_EQ(watched_graphs.size(), 2U);
  ::ascir::AscGraphDumperContext::GetThreadLocalCtx().ClearAllWatchGraphs();
  EXPECT_TRUE(watched_graphs.empty());
}

TEST_F(AscendGraphDumpUT, test_not_watch_when_env_set) {
  ::ascir::utils::ResetDumpConfig();
  setenv("AUTOFUSE_DFX_FLAGS", "codegen_compile_debug=true", 1);
  AscGraph graph("test");
  ::ascir::utils::DumpGraph(graph, "empty");

  AscGraph graph1("test1");
  ::ascir::utils::DumpGraph(graph, "empty1");

  auto &watched_graphs = ::ascir::AscGraphDumperContext::GetThreadLocalCtx().orderd_graphs_;
  EXPECT_TRUE(watched_graphs.empty());

  AscGraph sub("sub");
  graph.AddSubGraph(sub);
  ::ascir::utils::AlwaysDumpGraph(graph, "AutoFuseBeforeOptimize");
  EXPECT_TRUE(watched_graphs.empty());

  ::ascir::utils::DumpGraph(graph, "AutoFuseBeforeOptimize");
  unsetenv("AUTOFUSE_DFX_FLAGS");
}
}  // namespace ge

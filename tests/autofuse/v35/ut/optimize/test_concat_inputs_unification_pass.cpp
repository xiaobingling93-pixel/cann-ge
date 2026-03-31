/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asc_graph_builder.h"
#include "gtest/gtest.h"
#include "ascendc_ir.h"
#include "ascir_ops_utils.h"
#include "asc_graph_utils.h"
#include "task_generator/concat_inputs_unification_pass.h"
#include "platform/platform_factory.h"

namespace schedule {
using namespace optimize;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;

class ConcatInputUnificationPassTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}

  static AscGraph BuildGraph(const ge::Expression &s0, const ge::Expression &s1,
                             const std::vector<bool> &inputs_is_load, DataType dtype = ge::DT_FLOAT16) {
    auto s2 = s1 * Symbol(inputs_is_load.size());
    ge::testing::AscGraphBuilder graph_builder("test_graph");
    graph_builder.Loops({s0, s2});
    std::vector<std::string> concat_inputs;
    for (size_t i = 0; i < inputs_is_load.size(); ++i) {
      const auto is_load = inputs_is_load[i];
      const auto data_node_name = "data" + std::to_string(i);
      const auto load_node_name = "load" + std::to_string(i);
      graph_builder.Data(data_node_name, static_cast<int64_t>(i), dtype)
          .Load(load_node_name, data_node_name, {s0, s1}, {s1, ge::sym::kSymbolOne});
      if (is_load) {
        concat_inputs.push_back(load_node_name);
      } else {
        const auto relu_node_name = "relu" + std::to_string(i);
        graph_builder.Relu(relu_node_name, load_node_name);
        concat_inputs.push_back(relu_node_name);
      }
    }
    graph_builder.Concat("concat", concat_inputs).Store("store", "concat").Output("out", "store");
    return graph_builder.Build();
  }

  static AscGraph Build2InputGraph(const ge::Expression &s0, const ge::Expression &s1, DataType dtype = ge::DT_FLOAT16) {
    const std::vector<bool> inputs_is_load{true, false};
    return BuildGraph(s0, s1, inputs_is_load, dtype);
  }
};

TEST_F(ConcatInputUnificationPassTest, RunSuccess) {
  const auto graph = Build2InputGraph(ge::Symbol("s0"), ge::Symbol(15));
  std::vector<::ascir::ImplGraph> graphs{graph};
  EXPECT_EQ(ConcatInputUnificationPass::Run(graphs), ge::SUCCESS);
  EXPECT_TRUE(graphs[0].FindNode("ub_cpy_load0") != nullptr);
}

TEST_F(ConcatInputUnificationPassTest, DstColSizeOverLimit) {
  const auto graph = Build2InputGraph(ge::Symbol("s0"), ge::Symbol(34));
  std::vector<::ascir::ImplGraph> graphs{graph};
  EXPECT_EQ(ConcatInputUnificationPass::Run(graphs), ge::SUCCESS);
  EXPECT_TRUE(graphs[0].FindNode("ub_cpy_load0") == nullptr);
}

TEST_F(ConcatInputUnificationPassTest, SrcColSizeIsAligned) {
  const auto graph = Build2InputGraph(ge::Symbol("s0"), ge::Symbol(2));
  std::vector<::ascir::ImplGraph> graphs{graph};
  EXPECT_EQ(ConcatInputUnificationPass::Run(graphs), ge::SUCCESS);
  EXPECT_TRUE(graphs[0].FindNode("ub_cpy_load0") == nullptr);
}

TEST_F(ConcatInputUnificationPassTest, AllInputsIsLoad) {
  const std::vector<bool> inputs_is_load{true, true};
  const auto graph = BuildGraph(ge::Symbol("s0"), ge::Symbol(15), inputs_is_load);
  std::vector<::ascir::ImplGraph> graphs{graph};
  EXPECT_EQ(ConcatInputUnificationPass::Run(graphs), ge::SUCCESS);
  EXPECT_TRUE(graphs[0].FindNode("ub_cpy_load0") == nullptr);
}
}  // namespace schedule
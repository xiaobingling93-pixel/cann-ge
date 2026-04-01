/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "platform_context.h"
#include "task_generator/concat_schedule_case_generator.h"
#include "task_generator/concat_group_partitioner.h"
#include "platform/platform_factory.h"
#include "runtime_stub.h"
#include "task_generator/concat_score_function_generator.h"

namespace schedule {
using namespace optimize;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;

class ConcatScheduleCaseGeneratorV2Test : public ::testing::Test {
 protected:
  void SetUp() override {
    // dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<RuntimeStubV2>();
    RuntimeStub::SetInstance(stub_v2);
  }
  void TearDown() override {
    // dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }

  static void CreateConcatAscGraph(ge::AscGraph &graph, const std::string &head_dim,
                                   const std::vector<std::string> &concat_dims,
                                   const std::vector<std::string> &tail_dims) {
    std::vector<ge::Expression> concat_axis_sizes;
    ge::Expression concat_axis_size = ge::Symbol(0);
    for (const auto &dim : concat_dims) {
      if (dim[0] == 's') {
        concat_axis_sizes.emplace_back(graph.CreateSizeVar(dim));
      } else {
        concat_axis_sizes.emplace_back(ge::Symbol(std::strtol(dim.c_str(), nullptr, 10)));
      }
      concat_axis_size = concat_axis_size + concat_axis_sizes.back();
    }
    std::vector<ge::Expression> tail_dim_sizes;
    for (const auto &dim : tail_dims) {
      if (dim[0] == 's') {
        tail_dim_sizes.emplace_back(graph.CreateSizeVar(dim));
      } else {
        tail_dim_sizes.emplace_back(ge::Symbol(std::strtol(dim.c_str(), nullptr, 10)));
      }
    }

    auto head_dim_size = graph.CreateSizeVar(head_dim);
    auto z0 = graph.CreateAxis("z0", head_dim_size);
    auto z1 = graph.CreateAxis("z1", concat_axis_size);
    std::vector<Axis> tail_axes;
    tail_axes.reserve(tail_dims.size());
    for (size_t i = 0U; i < tail_dims.size(); ++i) {
      tail_axes.emplace_back(graph.CreateAxis("z" + std::to_string(2 + i), tail_dim_sizes[i]));
    }

    std::vector<std::shared_ptr<Data>> data_ops;
    std::vector<AscOpOutput> concat_inputs;
    ge::Expression concat_dim_size = ge::Symbol(0);
    for (size_t i = 0; i < concat_axis_sizes.size(); ++i) {
      std::string name = "x" + std::to_string(i);
      auto x_op = std::make_shared<Data>(name.c_str(), graph);
      x_op->y.dtype = ge::DT_FLOAT16;
      *x_op->y.repeats = {head_dim_size, concat_axis_sizes[i]};
      for (const auto &dim_size : tail_dim_sizes) {
        x_op->y.repeats->push_back(dim_size);
      }
      x_op->ir_attr.SetIndex(static_cast<int64_t>(i));
      data_ops.emplace_back(x_op);
      concat_inputs.push_back(x_op->y);
      concat_dim_size = concat_dim_size + concat_axis_sizes[i];
    }

    ascir_op::Concat concat_op("concat");
    concat_op.x = {concat_inputs[0], concat_inputs[1], concat_inputs[2], concat_inputs[3], concat_inputs[4]};
    concat_op.y.repeats->emplace_back(head_dim_size);
    concat_op.y.repeats->emplace_back(concat_dim_size);
    for (const auto &dim_size : tail_dim_sizes) {
      concat_op.y.repeats->push_back(dim_size);
    }
  }

  static std::string ExpressToStr(const std::vector<ge::Expression> &exprs) {
    std::stringstream ss;
    for (auto &size_expr : exprs) {
      ss << std::string(size_expr.Str().get()) << ", ";
    }
    return ss.str();
  }

  static std::string RepeatsToStr(const ge::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    return ExpressToStr(node->outputs[0].attr.repeats);
  }

  static std::string StridesToStr(const ge::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    return ExpressToStr(node->outputs[0].attr.strides);
  }
};

TEST_F(ConcatScheduleCaseGeneratorV2Test, ConcatTailDim_SplitConcat) {
  ge::AscGraph graph("concat_last_dim_graph");
  std::vector<int> concat_dim_sizes{412,
                                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                                    16, 16, 16,};
  auto s0 = graph.CreateSizeVar("s0");
  auto concat_size = ge::Expression(ge::Symbol(0));
  std::vector<std::shared_ptr<Data>> data_ops;
  std::vector<AscOpOutput> outputs;
  for (size_t i = 0; i < concat_dim_sizes.size(); ++i) {
    ge::Expression s_i;
    s_i = graph.CreateSizeVar(concat_dim_sizes[i]);
    concat_size = (concat_size + s_i);
    auto data_op = std::make_shared<Data>(("Data" + std::to_string(i + 1)).c_str(), graph);
    data_op->y.dtype = ge::DT_FLOAT;
    *data_op->y.repeats = {s0, s_i};
    *data_op->y.strides = {s_i, ge::ops::One};
    data_ops.emplace_back(data_op);
    outputs.emplace_back(data_ops.back()->y);
  }

  ascir_op::Concat concat_op("concat");
  concat_op.x = outputs;
  concat_op.y.dtype = ge::DT_FLOAT;
  *concat_op.y.repeats = {s0, concat_size};
  *concat_op.y.strides = {concat_size, ge::ops::One};

  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);

  optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), ge::SUCCESS);
  size_t index = 0;
  size_t last_end = 0;
  for (const auto &group : groups) {
    std::cout << "index: " << index << ", start: " << group.start << ", end: " << group.end << ", type: " << group.group_type << std::endl;
    std::vector<int> dims(concat_dim_sizes.begin() + static_cast<int64_t>(group.start),
                          concat_dim_sizes.begin() + static_cast<int64_t>(group.end));
    std::cout << "  " << ge::ToString(dims) << "count = " << group.end - group.start << ", size = " << group.size << std::endl;
    EXPECT_EQ(group.start, last_end);
    last_end = group.end;
    ++index;
  }
  EXPECT_EQ(groups.size(), 6);
  EXPECT_EQ(groups[0].end - groups[0].start, 17);
}

TEST_F(ConcatScheduleCaseGeneratorV2Test, ConcatTailDim_SplitConcat_LargeRowNum) {
  ge::AscGraph graph("concat_last_dim_graph");
  std::vector<int> concat_dim_sizes{64, 6, 28, 42};
  auto s0 = graph.CreateSizeVar(64 * 64);
  auto concat_size = ge::Expression(ge::Symbol(0));
  std::vector<std::shared_ptr<Data>> data_ops;
  std::vector<AscOpOutput> outputs;
  for (size_t i = 0; i < concat_dim_sizes.size(); ++i) {
    ge::Expression s_i;
    s_i = graph.CreateSizeVar(concat_dim_sizes[i]);
    concat_size = (concat_size + s_i);
    auto data_op = std::make_shared<Data>(("Data" + std::to_string(i + 1)).c_str(), graph);
    data_op->y.dtype = ge::DT_FLOAT;
    *data_op->y.repeats = {s0, s_i};
    *data_op->y.strides = {s_i, ge::ops::One};
    data_ops.emplace_back(data_op);
    outputs.emplace_back(data_ops.back()->y);
  }

  ascir_op::Concat concat_op("concat");
  concat_op.x = outputs;
  concat_op.y.dtype = ge::DT_FLOAT;
  *concat_op.y.repeats = {s0, concat_size};
  *concat_op.y.strides = {concat_size, ge::ops::One};

  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);

  optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), ge::SUCCESS);
  size_t index = 0;
  size_t last_end = 0;
  for (const auto &group : groups) {
    std::cout << "index: " << index << ", start: " << group.start << ", end: " << group.end << ", type: " << group.group_type << std::endl;
    std::vector<int> dims(concat_dim_sizes.begin() + static_cast<int64_t>(group.start),
                          concat_dim_sizes.begin() + static_cast<int64_t>(group.end));
    std::cout << "  " << ge::ToString(dims) << "count = " << group.end - group.start << ", size = " << group.size << std::endl;
    EXPECT_EQ(group.start, last_end);
    last_end = group.end;
    ++index;
  }
  EXPECT_EQ(groups.size(), 1);
}

TEST_F(ConcatScheduleCaseGeneratorV2Test, ConcatFirstDim_InsertAxis) {
  ge::AscGraph graph("concat_last_dim_graph");
  std::vector<int> concat_dim_sizes{1, 2, 1, 1, 1, 1};
  auto s0 = graph.CreateSizeVar(1);
  auto concat_size = ge::Expression(ge::Symbol(0));
  std::vector<std::shared_ptr<Data>> data_ops;
  std::vector<AscOpOutput> outputs;
  for (size_t i = 0; i < concat_dim_sizes.size(); ++i) {
    ge::Expression s_i;
    s_i = graph.CreateSizeVar(concat_dim_sizes[i]);
    concat_size = (concat_size + s_i);
    auto data_op = std::make_shared<Data>(("Data" + std::to_string(i + 1)).c_str(), graph);
    data_op->y.dtype = ge::DT_FLOAT;
    *data_op->y.repeats = {s0, s_i};
    *data_op->y.strides = {s_i, ge::ops::One};
    data_ops.emplace_back(data_op);
    outputs.emplace_back(data_ops.back()->y);
  }

  ascir_op::Concat concat_op("concat");
  concat_op.x = outputs;
  concat_op.y.dtype = ge::DT_FLOAT;
  *concat_op.y.repeats = {s0, concat_size};
  *concat_op.y.strides = {concat_size, ge::ops::One};

  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);

  optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), ge::SUCCESS);
  size_t index = 0;
  size_t last_end = 0;
  for (const auto &group : groups) {
    std::cout << "index: " << index << ", start: " << group.start << ", end: " << group.end << ", type: " << group.group_type << std::endl;
    std::vector<int> dims(concat_dim_sizes.begin() + static_cast<int64_t>(group.start),
                          concat_dim_sizes.begin() + static_cast<int64_t>(group.end));
    std::cout << "  " << ge::ToString(dims) << "count = " << group.end - group.start << ", size = " << group.size << std::endl;
    EXPECT_EQ(group.start, last_end);
    last_end = group.end;
    ++index;
  }
}

TEST_F(ConcatScheduleCaseGeneratorV2Test, OptimizeSameShapeConcat) {
  auto dtype = ge::DT_INT16;
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol(7);
  auto s2 = s1 + s1;
  auto graph = ge::testing::AscGraphBuilder("test_graph")
                   .Loops({s0, s2})
                   .Data("data0", 0, dtype)
                   .Load("load0", "data0", {s0, s1}, {s1, ge::sym::kSymbolOne})
                   .Data("data1", 1, dtype)
                   .Load("load1", "data1", {s0, s1}, {s1, ge::sym::kSymbolOne})
                   .Relu("relu0", "load1")
                   .Concat("concat", {"load0", "relu0"})
                   .Store("store", "concat")
                   .Output("out", "store")
                   .Build();
  auto concat_node = graph.FindNode("concat");
  std::vector<std::string> score_functions;
  std::vector<::ascir::ImplGraph> graphs;
  optimize::ConcatFusionCaseGenerator generator;
  EXPECT_EQ(generator.Generate(graph, graphs, score_functions), SUCCESS);
  EXPECT_EQ(graphs.size(), 2);
  EXPECT_TRUE(graphs[0].FindNode("ub_cpy_load0") != nullptr);
}
}  // namespace schedule
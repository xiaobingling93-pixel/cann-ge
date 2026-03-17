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
#include "ascir_utils.h"
#include "asc_graph_utils.h"
#include "graph_utils.h"
#include "task_generator/concat_schedule_case_generator.h"
#include "task_generator/concat_group_partitioner.h"
#include "task_generator/concat_score_function_generator.h"

namespace schedule {
using namespace optimize;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;

class ConcatScheduleCaseGeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
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

    ascir_op::Store store_op("store");
    store_op.x = concat_op.y;
    store_op.attr.api.compute_type = ge::ComputeType::kComputeStore;
    store_op.y.repeats = concat_op.y.repeats;
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

TEST_F(ConcatScheduleCaseGeneratorTest, ConcatTailDim_SplitConcat_continuous_small_tail) {
  ge::AscGraph graph("concat_last_dim_graph");
  std::vector<int> concat_dim_sizes{
      412, 4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
      4,   4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
      4,   4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  16, 16, 16,
      16,  16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
  };
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
    std::cout << "index: " << index << ", start: " << group.start << ", end: " << group.end
              << ", type: " << group.group_type << std::endl;
    std::vector<int> dims(concat_dim_sizes.begin() + static_cast<int64_t>(group.start),
                          concat_dim_sizes.begin() + static_cast<int64_t>(group.end));
    std::cout << "  " << ge::ToString(dims) << "count = " << group.end - group.start << ", size = " << group.size
              << std::endl;
    EXPECT_EQ(group.start, last_end);
    last_end = group.end;
    ++index;
  }
  EXPECT_EQ(groups.size(), 5);
}

TEST_F(ConcatScheduleCaseGeneratorTest, ConcatTailDim_SplitConcat_1648_inputs) {
  ge::AscGraph graph("concat_last_dim_graph");

  std::vector<int> concat_dim_sizes{
      5,   9,   5,   5,   5,   5,   5,   9,   5,   5,   5,   9,   5,   527, 25,  5,   5,   5,   5,   5,   5,   5,   5,
      5,   5,   5,   5,   129, 129, 5,   5,   5,   13,  21,  5,   25,  5,   5,   5,   5,   5,   5,   5,   5,   5,   5,
      5,   5,   5,   9,   1,   5,   5,   5,   5,   5,   5,   9,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,
      5,   5,   5,   5,   5,   5,   1,   1,   1,   45,  1,   129, 45,  1,   1,   45,  1,   145, 25,  1,   17,  5,   21,
      25,  9,   1,   9,   9,   537, 529, 473, 457, 9,   69,  1,   69,  1,   69,  1,   1,   1,   1,   1,   1,   1,   1,
      1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   17,  1,   1,   1,   1,   21,  1,   1,   1,   17,  1,   1,   17,
      1,   17,  1,   17,  1,   1,   1,   1,   1,   1,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,
      5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,
      5,   5,   5,   5,   5,   5,   5,   5,   17,  5,   5,   1,   5,   1,   1,   1,   17,  1,   1,   1,   17,  1,   1,
      17,  1,   17,  1,   17,  1,   1,   1,   5,   5,   5,   441, 5,   1,   1,   5,   17,  9,   321, 1,   17,  1,   17,
      1,   9,   1,   9,   17,  9,   1,   17,  305, 537, 529, 473, 457, 17,  449, 17,  377, 5,   17,  1,   1,   1,   1,
      17,  1,   1,   1,   17,  1,   1,   17,  1,   17,  1,   17,  1,   1,   1,   5,   1,   5,   5,   1,   1,   1,   1,
      1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
      1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
      1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   9,   9,   9,
      9,   9,   9,   9,   33,  33,  33,  321, 385, 17,  333, 381, 385, 385, 385, 377, 305, 1,   9,   21,  305, 377, 305,
      441, 369, 369, 369, 377, 305, 23,  1,   1,   1,   29,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
      1,   1,   1,   1,   1,   21,  25,  1,   33,  1,   5,   5,   39,  17,  5,   5,   5,   5,   5,   5,   5,   5,   5,
      5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   1,   17,  33,  17,  33,  33,  17,  5,   5,   5,   5,   5,
      13,  5,   13,  29,  5,   29,  13,  13,  13,  13,  5,   13,  13,  29,  29,  13,  13,  13,  13,  17,  17,  17,  17,
      17,  17,  95,  17,  17,  21,  21,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
      1,   1,   1,   1,   1,   1,   1,   77,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
      1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   21,  5,   13,  13,  13,  5,
      13,  5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   13,  13,  5,   5,   13,  5,
      5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   21,  21,  21,  21,  21,  21,  21,  21,  21,
      21,  1,   5,   21,  21,  45,  1,   17,  25,  1,   17,  145, 25,  1,   1,   1,   17,  1,   17,  1,   17,  1,   1,
      1,   1,   1,   17,  1,   1,   9,   1,   25,  1,   9,   1,   25,  1,   17,  1,   1,   45,  1,   45,  1,   45,  17,
      9,   5,   5,   5,   21,  21,  5,   13,  13,  29,  29,  29,  5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,
      5,   5,   5,   5,   5,   13,  13,  13,  13,  5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,
      5,   5,   21,  21,  21,  21,  21,  21,  21,  21,  21,  21,  5,   21,  17,  33,  33,  33,  1,   1,   9,   493, 493,
      493, 629, 1,   25,  25,  17,  17,  17,  17,  17,  9,   9,   17,  17,  17,  17,  9,   9,   17,  17,  25,  73,  5,
      29,  5,   29,  13,  29,  5,   45,  29,  25,  25,  89,  25,  25,  73,  25,  33,  25,  29,  33,  33,  73,  65,  65,
      65,  65,  65,  65,  73,  5,   5,   29,  13,  13,  13,  5,   13,  13,  29,  5,   29,  33,  33,  33,  5,   13,  5,
      29,  73,  65,  25,  33,  9,   441, 9,   441, 9,   441, 9,   441, 9,   441, 9,   441, 29,  5,   37,  37,  5,   5,
      37,  5,   37,  5,   37,  5,   5,   37,  5,   5,   37,  37,  5,   5,   37,  5,   37,  5,   37,  5,   5,   37,  5,
      5,   1,   37,  1,   1,   37,  1,   1,   5,   1,   1,   5,   1,   1,   37,  1,   1,   5,   37,  5,   1,   1,   37,
      1,   1,   5,   5,   37,  1,   1,   5,   5,   1,   37,  1,   1,   37,  1,   1,   5,   1,   1,   5,   1,   1,   37,
      1,   1,   5,   37,  5,   1,   1,   37,  1,   1,   5,   5,   37,  1,   1,   5,   5,   1,   1,   5,   5,   1,   17,
      17,  1,   1,   1,   1,   1,   5,   1,   5,   5,   1,   1,   1,   1,   5,   17,  1,   1,   1,   1,   1,   1,   25,
      129, 1,   69,  29,  29,  29,  29,  1,   25,  1,   1,   1,   1,   41,  41,  9,   1,   177, 41,  1,   1,   77,  1,
      137, 161, 1,   1,   65,  1,   29,  9,   17,  9,   9,   5,   65,  33,  1,   1,   1,   69,  1,   33,  17,  1,   1,
      1,   37,  1,   5,   17,  5,   9,   33,  9,   1,   5,   1,   1,   73,  1,   1,   1,   1,   1,   1,   1,   1,   1,
      269, 1,   1,   1,   1,   33,  17,  1,   1,   1,   1,   73,  13,  25,  1,   1,   49,  9,   1,   177, 17,  1,   1,
      17,  1,   1,   1,   17,  17,  1,   1,   1,   1,   1,   1,   17,  1,   5,   5,   5,   5,   5,   5,   5,   15,  25,
      1,   1,   1,   9,   451, 1,   17,  17,  1,   1,   1,   25,  475, 1,   1,   1,   1,   21,  17,  17,  17,  1,   1,
      17,  17,  1,   1,   17,  1,   17,  1,   1,   465, 1,   9,   1,   1,   17,  17,  17,  1,   1,   1,   1,   1,   1,
      177, 1,   1,   1,   33,  17,  17,  33,  33,  17,  17,  17,  1,   9,   33,  33,  49,  33,  33,  33,  1,   17,  17,
      1,   1,   73,  17,  1,   189, 17,  21,  1,   33,  1,   33,  1,   25,  33,  1,   1,   1,   9,   45,  155, 25,  17,
      9,   9,   9,   5,   5,   21,  45,  9,   17,  17,  25,  1,   17,  1,   1,   1,   17,  25,  1,   9,   9,   9,   5,
      1,   1,   1,   45,  1,   1,   45,  1,   1,   45,  1,   9,   17,  5,   21,  9,   9,   1,   9,   1,   5,   21,  21,
      45,  1,   9,   5,   17,  1,   1,   5,   25,  165, 53,  33,  9,   25,  201, 33,  1,   33,  29,  17,  29,  33,  49,
      33,  145, 33,  9,   41,  9,   41,  9,   41,  25,  9,   9,   1,   41,  41,  1,   41,  25,  5,   1,   5,   5,   17,
      1,   17,  49,  17,  1,   21,  1,   33,  25,  1,   21,  1,   5,   5,   5,   9,   9,   1,   1,   1,   1,   13,  1,
      1,   1,   1,   1,   1,   1,   63,  17,  1,   1,   1,   1,   1,   1,   1,   9,   13,  9,   9,   9,   9,   9,   5,
      17,  5,   5,   13,  17,  17,  465, 29,  197, 1,   27,  33,  1,   33,  33,  33,  21,  1,   1,   1,   1,   33,  33,
      1,   17,  21,  41,  21,  393, 9,   447, 429, 447, 5,   5,   33,  45,  33,  33,  13,  9,   21,  1,   21,  1,   13,
      13,  13,  1,   13,  13,  13,  31,  1,   1,   1,   1,   1,   1,   1,   1,   1,   393, 17,  5,   1,   13,  9,   1,
      1,   1,   25,  1,   1,   233, 1,   17,  17,  1,   17,  17,  17,  1,   1,   21,  17,  647, 1,   1,   1,   1,   1,
      17,  1,   17,  21,  9,   13,  13,  9,   13,  9,   9,   5,   1,   21,  33,  21,  17,  1,   133, 9,   1,   1,   1,
      33,  1,   59,  9,   1,   17,  9,   1,   1,   1,   1,   1,   1,   1,   1,   9,   1,   9,   9,   1,   9,   1,   1,
      1,   1,   17,  17,  17,  17,  17,  17,  17,  1,   5,   17,  1,   1,   1,   17,  1,   17,  17,  1,   17,  17,  17,
      1,   1,   1,   17,  1,   17,  1,   1,   1,   17,  17,  1,   17,  1,   17,  1,   1,   1,   1,   9,   1,   1,   1,
      1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   17,  1,   1,   1,   25,  13,  9,
      13,  45,  45,  45,  45,  45,  45,  45,  45,  45,  45,  9,   1,   17,  17,  5,   5,   5,   5,   5,   33,  5,   17,
      17,  17,  5,   5,   5,   5,   5,   5,   5,   5,   5,   17,  17,  17,  17,  17,  17,  17,  17,  17,  17,  9,   9,
      9,   9,   9,   9,   9,   9,   9,   9,   5,   5,   5,   5,   5,   5,   17,  17,  1,   5,   5,   5,   5,   5,   5,
      5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   9,   9,   17,  1,   1,   1,   5,   5,   1,   17,
      17,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
      1,   1,   1,   1,   1,   1,   1,   1,   1,   17,  1,   1,   1,   1,   17,  1,   1,   1,   17,  1,   1,   17,  1,
      17,  1,   17,  1,   1,   1,   17,  1,   1,   1,   1,   17,  1,   1,   1,   17,  1,   1,   17,  1,   17,  1,   17,
      1,   1,   1,   69,  9,   9,   537, 529, 473, 457, 9,   69,  537, 529, 473, 457, 9,   69,  17,  17,  9,   9,   17,
      17,  17,  17,  5,   5,   5,   5,   17,  17,  9,   9,   17,  17,  17,  17};
  auto s0 = graph.CreateSizeVar("s0");
  auto concat_size = ge::Expression(ge::Symbol(0));
  std::vector<std::shared_ptr<Data>> data_ops;
  std::vector<AscOpOutput> outputs;
  for (size_t i = 0; i < concat_dim_sizes.size(); ++i) {
    ge::Expression s_i;
    s_i = graph.CreateSizeVar(concat_dim_sizes[i]);
    concat_size = (concat_size + s_i);
    auto data_op = std::make_shared<Data>(("Data" + std::to_string(i + 1)).c_str(), graph);
    data_op->y.dtype = ge::DT_FLOAT16;
    *data_op->y.repeats = {s0, s_i};
    *data_op->y.strides = {s_i, ge::ops::One};
    data_ops.emplace_back(data_op);
    outputs.emplace_back(data_ops.back()->y);
  }

  ascir_op::Concat concat_op("concat");
  concat_op.x = outputs;
  concat_op.y.dtype = ge::DT_FLOAT16;
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
    std::cout << "index: " << index << ", start: " << group.start << ", end: " << group.end
              << ", type: " << group.group_type << std::endl;
    std::vector<int> dims(concat_dim_sizes.begin() + static_cast<int64_t>(group.start),
                          concat_dim_sizes.begin() + static_cast<int64_t>(group.end));
    std::cout << "  " << ge::ToString(dims) << "count = " << group.end - group.start << ", size = " << group.size
              << std::endl;
    EXPECT_EQ(group.start, last_end);
    last_end = group.end;
    ++index;
  }
  //  EXPECT_EQ(groups.size(), 36);
}

TEST_F(ConcatScheduleCaseGeneratorTest, ConcatTailDim_SplitConcatStatiThenDynamic) {
  ge::AscGraph graph("concat_last_dim_graph");

  std::vector<std::string> concat_dim_sizes{"4", "s1", "s2", "s3"};
  auto s0 = graph.CreateSizeVar("s0");
  auto concat_size = ge::Expression(ge::Symbol(0));
  std::vector<std::shared_ptr<Data>> data_ops;
  std::vector<AscOpOutput> outputs;
  for (size_t i = 0; i < concat_dim_sizes.size(); ++i) {
    ge::Expression s_i;
    if (concat_dim_sizes[i][0] == 's') {
      s_i = graph.CreateSizeVar(concat_dim_sizes[i]);
    } else {
      s_i = graph.CreateSizeVar(std::strtol(concat_dim_sizes[i].c_str(), nullptr, 10));
    }
    concat_size = (concat_size + s_i);
    auto data_op = std::make_shared<Data>(("Data" + std::to_string(i + 1)).c_str(), graph);
    data_op->y.dtype = ge::DT_FLOAT16;
    *data_op->y.repeats = {s0, s_i};
    *data_op->y.strides = {s_i, ge::ops::One};
    data_ops.emplace_back(data_op);
    outputs.emplace_back(data_ops.back()->y);
  }

  ascir_op::Concat concat_op("concat");
  concat_op.x = outputs;
  concat_op.y.dtype = ge::DT_FLOAT16;
  *concat_op.y.repeats = {s0, concat_size};
  *concat_op.y.strides = {concat_size, ge::ops::One};

  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);

  optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), ge::SUCCESS);
  std::vector<std::vector<std::string>> results;
  for (const auto &group : groups) {
    std::cout << "start: " << group.start << ", end: " << group.end << ", type: " << group.group_type << std::endl;
    std::vector<std::string> dims(concat_dim_sizes.begin() + static_cast<int64_t>(group.start),
                                  concat_dim_sizes.begin() + static_cast<int64_t>(group.end));
    std::cout << "  " << ge::ToString(dims) << ", size = " << group.size << std::endl;
    results.emplace_back(dims);
  }
  EXPECT_EQ(results.size(), 4);
  EXPECT_EQ(results[0], (std::vector<std::string>{"4"}));
  EXPECT_EQ(results[1], (std::vector<std::string>{"s1"}));
  EXPECT_EQ(results[2], (std::vector<std::string>{"s2"}));
  EXPECT_EQ(results[3], (std::vector<std::string>{"s3"}));
}

TEST_F(ConcatScheduleCaseGeneratorTest, ConcatTailDim_SplitConcat) {
  ge::AscGraph graph("concat_last_dim_graph");

  std::vector<std::string> concat_dim_sizes{"412", "1",  "6",  "6",  "6", "6", "16", "16", "33", "16", "32",
                                            "32",  "s1", "s2", "32", "1", "2", "3",  "16", "1",  "222"};
  auto s0 = graph.CreateSizeVar("s0");
  auto concat_size = ge::Expression(ge::Symbol(0));
  std::vector<std::shared_ptr<Data>> data_ops;
  std::vector<AscOpOutput> outputs;
  for (size_t i = 0; i < concat_dim_sizes.size(); ++i) {
    ge::Expression s_i;
    if (concat_dim_sizes[i][0] == 's') {
      s_i = graph.CreateSizeVar(concat_dim_sizes[i]);
    } else {
      s_i = graph.CreateSizeVar(std::strtol(concat_dim_sizes[i].c_str(), nullptr, 10));
    }
    concat_size = (concat_size + s_i);
    auto data_op = std::make_shared<Data>(("Data" + std::to_string(i + 1)).c_str(), graph);
    data_op->y.dtype = ge::DT_FLOAT16;
    *data_op->y.repeats = {s0, s_i};
    *data_op->y.strides = {s_i, ge::ops::One};
    data_ops.emplace_back(data_op);
    outputs.emplace_back(data_ops.back()->y);
  }

  ascir_op::Concat concat_op("concat");
  concat_op.x = outputs;
  concat_op.y.dtype = ge::DT_FLOAT16;
  *concat_op.y.repeats = {s0, concat_size};
  *concat_op.y.strides = {concat_size, ge::ops::One};

  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);

  optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), ge::SUCCESS);
  std::vector<std::vector<std::string>> results;
  for (const auto &group : groups) {
    std::cout << "start: " << group.start << ", end: " << group.end << ", type: " << group.group_type << std::endl;
    std::vector<std::string> dims(concat_dim_sizes.begin() + static_cast<int64_t>(group.start),
                                  concat_dim_sizes.begin() + static_cast<int64_t>(group.end));
    std::cout << "  " << ge::ToString(dims) << ", size = " << group.size << std::endl;
    results.emplace_back(dims);
  }
  EXPECT_EQ(results.size(), 7);
  EXPECT_EQ(results[0], (std::vector<std::string>{"412"}));
  EXPECT_EQ(results[1], (std::vector<std::string>{"1", "6", "6", "6", "6", "16", "16", "33"}));
  EXPECT_EQ(results[2], (std::vector<std::string>{"16", "32", "32"}));
  EXPECT_EQ(results[3], (std::vector<std::string>{"s1"}));
  EXPECT_EQ(results[4], (std::vector<std::string>{"s2"}));
  EXPECT_EQ(results[5], (std::vector<std::string>{"32", "1", "2", "3", "16", "1"}));
  EXPECT_EQ(results[6], (std::vector<std::string>{"222"}));
}

TEST_F(ConcatScheduleCaseGeneratorTest, ConcatTailDim_SplitConcat_412_1) {
  ge::AscGraph graph("concat_last_dim_graph");

  std::vector<std::string> concat_dim_sizes(412, "1");
  concat_dim_sizes.emplace_back("16");
  concat_dim_sizes.emplace_back("16");
  concat_dim_sizes.emplace_back("1");
  concat_dim_sizes.emplace_back("2");
  auto s0 = graph.CreateSizeVar("s0");
  auto concat_size = ge::Expression(ge::Symbol(0));
  std::vector<std::shared_ptr<Data>> data_ops;
  std::vector<AscOpOutput> outputs;
  for (size_t i = 0; i < concat_dim_sizes.size(); ++i) {
    ge::Expression s_i;
    if (concat_dim_sizes[i][0] == 's') {
      s_i = graph.CreateSizeVar(concat_dim_sizes[i]);
    } else {
      s_i = graph.CreateSizeVar(std::strtol(concat_dim_sizes[i].c_str(), nullptr, 10));
    }
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
  std::vector<std::vector<std::string>> results;
  for (const auto &group : groups) {
    std::cout << "start: " << group.start << ", end: " << group.end << ", type: " << group.group_type << std::endl;
    std::vector<std::string> dims(concat_dim_sizes.begin() + static_cast<int64_t>(group.start),
                                  concat_dim_sizes.begin() + static_cast<int64_t>(group.end));
    std::cout << "  " << ge::ToString(dims) << ", size = " << group.size << std::endl;
    results.emplace_back(dims);
  }
  EXPECT_EQ(results.size(), 13);
  std::vector<std::string> expect = {28, "1"};
  expect.push_back("16");
  expect.push_back("16");
  expect.push_back("1");
  expect.push_back("2");
  EXPECT_EQ(results[12], expect);
}

TEST_F(ConcatScheduleCaseGeneratorTest, ConcatTailDim_SplitConcat_AlignAndSmallTail) {
  dlog_setlevel(0, 0, 1);
  ge::AscGraph graph("concat_last_dim_graph");

  std::vector<std::string> concat_dim_sizes{"32", "32", "32", "32", "32", "32", "16", "16", "16", "16", "16", "17"};
  auto s0 = graph.CreateSizeVar("s0");
  auto concat_size = ge::Expression(ge::Symbol(0));
  std::vector<std::shared_ptr<Data>> data_ops;
  std::vector<AscOpOutput> outputs;
  for (size_t i = 0; i < concat_dim_sizes.size(); ++i) {
    ge::Expression s_i;
    if (concat_dim_sizes[i][0] == 's') {
      s_i = graph.CreateSizeVar(concat_dim_sizes[i]);
    } else {
      s_i = graph.CreateSizeVar(std::strtol(concat_dim_sizes[i].c_str(), nullptr, 10));
    }
    concat_size = (concat_size + s_i);
    auto data_op = std::make_shared<Data>(("Data" + std::to_string(i + 1)).c_str(), graph);
    data_op->y.dtype = ge::DT_FLOAT16;
    *data_op->y.repeats = {s0, s_i};
    *data_op->y.strides = {s_i, ge::ops::One};
    data_ops.emplace_back(data_op);
    outputs.emplace_back(data_ops.back()->y);
  }

  ascir_op::Concat concat_op("concat");
  concat_op.x = outputs;
  concat_op.y.dtype = ge::DT_FLOAT16;
  *concat_op.y.repeats = {s0, concat_size};
  *concat_op.y.strides = {concat_size, ge::ops::One};

  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);
  optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), ge::SUCCESS);
  std::vector<std::vector<std::string>> results;
  for (const auto &group : groups) {
    std::cout << "start: " << group.start << ", end: " << group.end << ", type: " << group.group_type << std::endl;
    std::vector<std::string> dims(concat_dim_sizes.begin() + static_cast<int64_t>(group.start),
                                  concat_dim_sizes.begin() + static_cast<int64_t>(group.end));
    std::cout << "  " << ge::ToString(dims) << ", size = " << group.size << std::endl;
    results.emplace_back(dims);
  }
  EXPECT_EQ(results.size(), 2);
}

TEST_F(ConcatScheduleCaseGeneratorTest, ConcatTailDim_SplitConcat_ConvertSmallGroup) {
  ge::AscGraph graph("concat_last_dim_graph");
  std::vector<int> concat_dim_sizes{64, 6, 28, 42};
  auto s0 = graph.CreateSizeVar(32 * 64);
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
  dlog_setlevel(0, 0, 1);
  optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), ge::SUCCESS);
  size_t index = 0;
  size_t last_end = 0;
  for (const auto &group : groups) {
    std::cout << "index: " << index << ", start: " << group.start << ", end: " << group.end
              << ", type: " << group.group_type << std::endl;
    std::vector<int> dims(concat_dim_sizes.begin() + static_cast<int64_t>(group.start),
                          concat_dim_sizes.begin() + static_cast<int64_t>(group.end));
    std::cout << "  " << ge::ToString(dims) << "count = " << group.end - group.start << ", size = " << group.size
              << std::endl;
    EXPECT_EQ(group.start, last_end);
    last_end = group.end;
    ++index;
  }
  EXPECT_EQ(groups.size(), 2);
}

TEST_F(ConcatScheduleCaseGeneratorTest, ConcatTailDim_SplitConcat_ConvertSmallGroup2) {
  ge::AscGraph graph("concat_last_dim_graph");
  std::vector<int> concat_dim_sizes{4, 8, 8, 8, 8, 16, 16};
  auto s0 = graph.CreateSizeVar(32 * 64);
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
  dlog_setlevel(0, 0, 1);
  optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), ge::SUCCESS);
  size_t index = 0;
  size_t last_end = 0;
  for (const auto &group : groups) {
    std::cout << "index: " << index << ", start: " << group.start << ", end: " << group.end
              << ", type: " << group.group_type << std::endl;
    std::vector<int> dims(concat_dim_sizes.begin() + static_cast<int64_t>(group.start),
                          concat_dim_sizes.begin() + static_cast<int64_t>(group.end));
    std::cout << "  " << ge::ToString(dims) << "count = " << group.end - group.start << ", size = " << group.size
              << std::endl;
    EXPECT_EQ(group.start, last_end);
    last_end = group.end;
    ++index;
  }
}

TEST_F(ConcatScheduleCaseGeneratorTest, concat_tail_dim1_scene) {
  ge::AscGraph graph("concat_last_dim_graph");
  auto s0 = ge::Symbol("s0");
  auto s2 = ge::Symbol("s2");
  auto sym2 = ge::Symbol(2);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", ge::sym::kSymbolOne);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", sym2);

  Data data0("data0", graph);
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);
  Load load0("load0");
  load0.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.repeats = {s0, ge::sym::kSymbolOne, s2, ge::sym::kSymbolOne};
  *load0.y.strides = {s2, ge::sym::kSymbolZero, ge::sym::kSymbolOne, ge::sym::kSymbolZero};

  Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  Load load1("load1");
  load1.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.repeats = {s0, ge::sym::kSymbolOne, s2, ge::sym::kSymbolOne};
  *load1.y.strides = {s2, ge::sym::kSymbolZero, ge::sym::kSymbolOne, ge::sym::kSymbolZero};

  Concat concat("concat");
  concat.attr.api.compute_type = ge::ComputeType::kComputeConcat;
  concat.y.dtype = ge::DT_FLOAT16;
  concat.x = {load0.y, load1.y};
  concat.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.axis = {z0.id, z1.id, z2.id, z3.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.repeats = {s0, ge::sym::kSymbolOne, s2, sym2};
  *concat.y.strides = {s2 * sym2, ge::sym::kSymbolZero, sym2, ge::sym::kSymbolOne};

  Store store("store");
  store.y.dtype = ge::DT_FLOAT16;
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.repeats = {s0, ge::sym::kSymbolOne, s2, sym2};
  *store.y.strides = {s2 * sym2, ge::sym::kSymbolZero, sym2, ge::sym::kSymbolOne};

  Output out("out");
  out.attr.api.type = ge::ApiType::kAPITypeBuffer;
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  optimize::ConcatFusionCaseGenerator generator;
  std::vector<AscGraph> generated_graphs;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.Generate(graph, generated_graphs, score_functions), ge::SUCCESS);
  EXPECT_EQ(generated_graphs.size(), 1UL);
  std::string load0_repeats = RepeatsToStr(generated_graphs[0], "load0");
  EXPECT_EQ(load0_repeats, "s0, 1, s2, 1, ");
  std::string load0_strides = StridesToStr(generated_graphs[0], "load0");
  EXPECT_EQ(load0_strides, "s2, 0, 1, 0, ");
}

TEST_F(ConcatScheduleCaseGeneratorTest, concat_data_to_different_axis) {
  ge::AscGraph graph("concat_last_dim_graph");
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s1_1 = ge::Symbol("s1_1");
  auto s1_2 = ge::Symbol("s1_2");
  auto s1_3 = ge::Symbol("s1_3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z1_1 = graph.CreateAxis("z1_1", s1_1);
  auto z1_2 = graph.CreateAxis("z1_2", s1_2);
  auto z1_3 = graph.CreateAxis("z1_3", s1_3);

  Data data0("data0", graph);
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);

  Load load1("load1");
  load1.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  load1.x = data0.y;
  load1.attr.sched.axis = {z0.id, z1_1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1_1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.repeats = {s0, s1_1};
  *load1.y.strides = {s1_1, ge::sym::kSymbolOne};

  Load load2("load2");
  load2.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load2.y.dtype = ge::DT_FLOAT16;
  load2.x = data0.y;
  load2.attr.sched.axis = {z0.id, z1_2.id};
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1_2.id};
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.repeats = {s0, s1_2};
  *load2.y.strides = {s1_2, ge::sym::kSymbolOne};

  Load load3("load3");
  load3.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load3.y.dtype = ge::DT_FLOAT16;
  load3.x = data0.y;
  load3.attr.sched.axis = {z0.id, z1_3.id};
  load3.y.dtype = ge::DT_FLOAT16;
  *load3.y.axis = {z0.id, z1_3.id};
  load3.y.dtype = ge::DT_FLOAT16;
  *load3.y.repeats = {s0, s1_3};
  *load3.y.strides = {s1_3, ge::sym::kSymbolOne};

  Concat concat("concat");
  concat.attr.api.compute_type = ge::ComputeType::kComputeConcat;
  concat.y.dtype = ge::DT_FLOAT16;
  concat.x = {load1.y, load2.y, load3.y};
  concat.attr.sched.axis = {z0.id, z1.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.axis = {z0.id, z1.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.repeats = {s0, s1};
  *concat.y.strides = {s1, ge::sym::kSymbolOne};

  Store store("store");
  store.y.dtype = ge::DT_FLOAT16;
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, ge::sym::kSymbolOne};

  Output out("out");
  out.attr.api.type = ge::ApiType::kAPITypeBuffer;
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  optimize::ConcatFusionCaseGenerator generator;
  std::vector<AscGraph> generated_graphs;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.Generate(graph, generated_graphs, score_functions), ge::SUCCESS);
  ASSERT_EQ(generated_graphs.size(), 3UL);

  auto cg0 = ge::AscGraphUtils::GetComputeGraph(generated_graphs[0]);
  auto cg1 = ge::AscGraphUtils::GetComputeGraph(generated_graphs[1]);
  EXPECT_EQ(cg0->GetAllNodesSize(), 9UL);
  EXPECT_EQ(cg1->GetAllNodesSize(), 12UL);
}

TEST_F(ConcatScheduleCaseGeneratorTest, ConcatBackwardFusion) {
  ge::AscGraph graph("concat_last_dim_graph");
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s1_1 = ge::Symbol("s1_1");
  auto s1_2 = ge::Symbol("s1_2");
  auto s1_3 = ge::Symbol("s1_3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z1_1 = graph.CreateAxis("z1_1", s1_1);
  auto z1_2 = graph.CreateAxis("z1_2", s1_2);
  auto z1_3 = graph.CreateAxis("z1_3", s1_3);

  Data data0("data0", graph);
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);

  Data data1("data1", graph);
  data1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  load1.x = data0.y;
  load1.attr.sched.axis = {z0.id, z1_1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1_1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.repeats = {s0, s1_1};
  *load1.y.strides = {s1_1, ge::sym::kSymbolOne};

  Load load2("load2");
  load2.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load2.y.dtype = ge::DT_FLOAT16;
  load2.x = data0.y;
  load2.attr.sched.axis = {z0.id, z1_2.id};
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1_2.id};
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.repeats = {s0, s1_2};
  *load2.y.strides = {s1_2, ge::sym::kSymbolOne};

  Load load3("load3");
  load3.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load3.y.dtype = ge::DT_FLOAT16;
  load3.x = data0.y;
  load3.attr.sched.axis = {z0.id, z1_3.id};
  load3.y.dtype = ge::DT_FLOAT16;
  *load3.y.axis = {z0.id, z1_3.id};
  load3.y.dtype = ge::DT_FLOAT16;
  *load3.y.repeats = {s0, s1_3};
  *load3.y.strides = {s1_3, ge::sym::kSymbolOne};

  Load load4("load4");
  load4.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load4.y.dtype = ge::DT_FLOAT16;
  load4.x = data1.y;
  load4.attr.sched.axis = {z0.id, z1.id};
  load4.y.dtype = ge::DT_FLOAT16;
  *load4.y.axis = {z0.id, z1.id};
  load4.y.dtype = ge::DT_FLOAT16;
  *load4.y.repeats = {s0, s1};
  *load4.y.strides = {s1, ge::sym::kSymbolOne};

  Concat concat("concat");
  concat.attr.api.compute_type = ge::ComputeType::kComputeConcat;
  concat.y.dtype = ge::DT_FLOAT16;
  concat.x = {load1.y, load2.y, load3.y};
  concat.attr.sched.axis = {z0.id, z1.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.axis = {z0.id, z1.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.repeats = {s0, s1};
  *concat.y.strides = {s1, ge::sym::kSymbolOne};

  Relu relu("relu");
  relu.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  relu.y.dtype = ge::DT_FLOAT16;
  relu.x = concat.y;
  relu.attr.sched.axis = {z0.id, z1.id};
  relu.y.dtype = ge::DT_FLOAT16;
  *relu.y.axis = {z0.id, z1.id};
  relu.y.dtype = ge::DT_FLOAT16;
  *relu.y.repeats = {s0, s1};
  *relu.y.strides = {s1, ge::sym::kSymbolOne};

  Add add("add");
  add.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  add.y.dtype = ge::DT_FLOAT16;
  add.x1 = concat.y;
  add.x2 = load4.y;
  add.attr.sched.axis = {z0.id, z1.id};
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.axis = {z0.id, z1.id};
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.repeats = {s0, s1};
  *add.y.strides = {s1, ge::sym::kSymbolOne};

  Store store("store");
  store.y.dtype = ge::DT_FLOAT16;
  store.x = relu.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, ge::sym::kSymbolOne};

  Store store_1("store_1");
  store_1.y.dtype = ge::DT_FLOAT16;
  store_1.x = add.y;
  store_1.attr.sched.axis = {z0.id, z1.id};
  store_1.y.dtype = ge::DT_FLOAT16;
  *store_1.y.axis = {z0.id, z1.id};
  store_1.y.dtype = ge::DT_FLOAT16;
  *store_1.y.repeats = {s0, s1};
  *store_1.y.strides = {s1, ge::sym::kSymbolOne};

  Output out("out");
  out.attr.api.type = ge::ApiType::kAPITypeBuffer;
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  Output out_1("out_1");
  out_1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  out_1.x = store_1.y;
  out_1.ir_attr.SetIndex(1);

  optimize::ConcatFusionCaseGenerator generator;
  std::vector<AscGraph> generated_graphs;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.Generate(graph, generated_graphs, score_functions), ge::SUCCESS);
  ASSERT_EQ(generated_graphs.size(), 3UL);

  auto cg0 = ge::AscGraphUtils::GetComputeGraph(generated_graphs[0]);
  auto cg1 = ge::AscGraphUtils::GetComputeGraph(generated_graphs[1]);
  EXPECT_EQ(cg0->GetAllNodesSize(), 15UL);
  EXPECT_EQ(cg1->GetAllNodesSize(), 30UL);
}

TEST_F(ConcatScheduleCaseGeneratorTest, ConcatBackwardFusion_WithSameLoad) {
  ge::AscGraph graph("concat_last_dim_graph");
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s1_1 = ge::Symbol("s1_1");
  auto s1_2 = ge::Symbol("s1_2");
  auto s1_3 = ge::Symbol(1);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z1_1 = graph.CreateAxis("z1_1", s1_1);
  auto z1_2 = graph.CreateAxis("z1_2", s1_2);
  auto z1_3 = graph.CreateAxis("z1_3", s1_3);

  Data data0("data0", graph);
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);

  Load load1("load1");
  load1.x = data0.y;
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.repeats = {s0, s1_1};
  *load1.y.strides = {s1_1, ge::sym::kSymbolOne};
  load1.ir_attr.SetOffset(sym::kSymbolZero);

  Load load2("load2");
  load2.x = data0.y;
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.repeats = {s0, s1_2};
  *load2.y.strides = {s1_2, ge::sym::kSymbolOne};
  load2.ir_attr.SetOffset(sym::kSymbolZero);

  Load load3("load3");
  load3.y.dtype = ge::DT_FLOAT16;
  load3.x = data0.y;
  *load3.y.repeats = {s0, s1_3};
  *load3.y.strides = {s1_3, ge::sym::kSymbolOne};
  load3.ir_attr.SetOffset(sym::kSymbolZero);

  Broadcast brc1("brc1");
  brc1.x = load3.y;
  *brc1.y.repeats = {s0, s1_2};
  *brc1.y.strides = {s1_2, ge::sym::kSymbolOne};

  Add add1("add1");
  add1.x1 = load2.y;
  add1.x2 = brc1.y;
  *add1.y.repeats = {s0, s1_2};
  *add1.y.strides = {s1_2, ge::sym::kSymbolOne};

  Concat concat("concat");
  concat.y.dtype = ge::DT_FLOAT16;
  concat.x = {load1.y, add1.y, load3.y};
  concat.attr.sched.axis = {z0.id, z1.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.repeats = {s0, s1};
  *concat.y.strides = {s1, ge::sym::kSymbolOne};

  Broadcast brc("brc");
  brc.x = load3.y;
  *brc.y.repeats = {s0, s1};
  *brc.y.strides = {s1, ge::sym::kSymbolOne};

  Add add("add");
  add.x1 = concat.y;
  add.x2 = brc.y;
  *add.y.repeats = {s0, s1};
  *add.y.strides = {s1, ge::sym::kSymbolOne};

  Store store("store_1");
  store.x = add.y;
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, ge::sym::kSymbolOne};

  Output out("out");
  out.attr.api.type = ge::ApiType::kAPITypeBuffer;
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  optimize::ConcatFusionCaseGenerator generator;
  std::vector<AscGraph> generated_graphs;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.Generate(graph, generated_graphs, score_functions), ge::SUCCESS);
  ASSERT_EQ(generated_graphs.size(), 3UL);
  for (const auto & generated_graph : generated_graphs) {
    auto cg = ge::AscGraphUtils::GetComputeGraph(generated_graph);
    auto add_node = cg->FindFirstNodeMatchType("Add");
    ASSERT_TRUE(add_node != nullptr);
    EXPECT_EQ(add_node->GetInDataNodesSize(), 2);
    for (const auto &node : cg->GetAllNodes()) {
      if (node->GetType() == "Load") {
        auto asc_node = std::dynamic_pointer_cast<ge::AscNode>(node);
        const auto ir_attr = asc_node->attr.ir_attr->DownCastTo<ge::ascir_op::Load::AscLoadIrAttrDef>();
        ge::Expression offset;
        EXPECT_EQ(ir_attr->GetOffset(offset), GRAPH_SUCCESS);
        EXPECT_EQ(offset, sym::kSymbolZero);
      }
    }
  }
}

TEST_F(ConcatScheduleCaseGeneratorTest, ConcatScoreFunc) {
  ge::AscGraph graph("concat_last_dim_graph");

  CreateConcatAscGraph(graph, {"s0"}, {"s1", "s2", "s3", "s4", "s5"}, {"2", "1"});
  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);
  optimize::ConcatScoreFunctionGenerator generator(graph, concat_node, 1);
  std::string score_func;
  EXPECT_EQ(generator.Generate(score_func), ge::SUCCESS);
  EXPECT_TRUE(!score_func.empty());
}

TEST_F(ConcatScheduleCaseGeneratorTest, ConcatScoreFunc_stride_aligned) {
  ge::AscGraph graph("concat_last_dim_graph");

  CreateConcatAscGraph(graph, {"s0"}, {"s1", "s2", "s3", "s4", "s5"}, {"2", "8"});
  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);
  optimize::ConcatScoreFunctionGenerator generator(graph, concat_node, 1);
  std::string score_func;
  EXPECT_EQ(generator.Generate(score_func), ge::SUCCESS);
  EXPECT_TRUE(score_func.find("return 1;") != std::string::npos);
}

TEST_F(ConcatScheduleCaseGeneratorTest, ConcatScoreFunc_concat_dim_aligned) {
  ge::AscGraph graph("concat_last_dim_graph");

  CreateConcatAscGraph(graph, {"s0"}, {"4", "8", "12", "16", "20"}, {"2", "2"});
  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);
  optimize::ConcatScoreFunctionGenerator generator(graph, concat_node, 1);
  std::string score_func;
  EXPECT_EQ(generator.Generate(score_func), ge::SUCCESS);
  EXPECT_TRUE(score_func.find("return 1;") != std::string::npos);
}

TEST_F(ConcatScheduleCaseGeneratorTest, ConcatScoreFunc_concat_dim_unaligned) {
  ge::AscGraph graph("concat_last_dim_graph");

  CreateConcatAscGraph(graph, {"s0"}, {"3", "8", "12", "16", "20"}, {"2", "2"});
  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);
  optimize::ConcatScoreFunctionGenerator generator(graph, concat_node, 1);
  std::string score_func;
  EXPECT_EQ(generator.Generate(score_func), ge::SUCCESS);
  EXPECT_TRUE(score_func.find("return -1;") != std::string::npos);
}

TEST_F(ConcatScheduleCaseGeneratorTest, ConcatScoreFunc_concat_dim_dynamic) {
  ge::AscGraph graph("concat_last_dim_graph");

  CreateConcatAscGraph(graph, {"s0"}, {"3", "8", "12", "16", "20"}, {"s1", "s2"});
  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);
  optimize::ConcatScoreFunctionGenerator generator(graph, concat_node, 1);
  std::string score_func;
  EXPECT_EQ(generator.Generate(score_func), ge::SUCCESS);
  // 验证生成实际的函数
  EXPECT_TRUE(score_func.find("graph0_result0_g0_tiling_data") != std::string::npos);

  score_func = "";
  optimize::ConcatScoreFunctionGenerator generator_1(graph, concat_node, 1);
  EXPECT_EQ(generator_1.GenerateForCheckSmallTail(score_func), ge::SUCCESS);
  std::cout << score_func << std::endl;
  EXPECT_TRUE(score_func.find("graph0_result1_g0_tiling_data") != std::string::npos);
}

TEST_F(ConcatScheduleCaseGeneratorTest, UseSmallTailConcatApi_TailDim_NotAligned) {
  ge::AscGraph graph("concat_last_dim_graph");
  CreateConcatAscGraph(graph, {"s0"}, {"1", "1", "1", "1", "1"}, {});
  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);
  auto store_node = graph.FindNode("store");
  ASSERT_TRUE(store_node != nullptr);
  store_node->outputs[0].attr.repeats = {ge::Symbol("s0"), ge::Symbol(5)};
  store_node->outputs[0].attr.strides = {ge::Symbol(5), ge::ops::One};
  bool output_need_align = false;
  EXPECT_TRUE(::ascir::utils::UseSmallTailConcatApi(*concat_node, &output_need_align));
  EXPECT_FALSE(output_need_align);

  store_node->outputs[0].attr.strides = {ge::Symbol(10), ge::ops::One};
  EXPECT_TRUE(::ascir::utils::UseSmallTailConcatApi(*concat_node, &output_need_align));
  EXPECT_TRUE(output_need_align);
}

TEST_F(ConcatScheduleCaseGeneratorTest, UseSmallTailConcatApi_TailDim_Aligned) {
  ge::AscGraph graph("concat_last_dim_graph");
  CreateConcatAscGraph(graph, {"s0"}, {"1", "1", "1", "1", "12"}, {});
  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);
  auto store_node = graph.FindNode("store");
  ASSERT_TRUE(store_node != nullptr);
  store_node->outputs[0].attr.repeats = {ge::Symbol("s0"), ge::Symbol(16)};
  store_node->outputs[0].attr.strides = {ge::Symbol(16), ge::ops::One};
  bool output_need_align = false;
  EXPECT_TRUE(::ascir::utils::UseSmallTailConcatApi(*concat_node, &output_need_align));
  EXPECT_FALSE(output_need_align);

  store_node->outputs[0].attr.strides = {ge::Symbol(32), ge::ops::One};
  EXPECT_TRUE(::ascir::utils::UseSmallTailConcatApi(*concat_node, &output_need_align));
  EXPECT_FALSE(output_need_align);
}

TEST_F(ConcatScheduleCaseGeneratorTest, UseSmallTailConcatApi_NonTailDim_NotAligned) {
  ge::AscGraph graph("concat_last_dim_graph");
  CreateConcatAscGraph(graph, {"s0"}, {"1", "1", "1", "1", "1"}, {"2"});
  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);
  auto store_node = graph.FindNode("store");
  ASSERT_TRUE(store_node != nullptr);
  store_node->outputs[0].attr.repeats = {ge::Symbol("s0"), ge::Symbol(5), ge::Symbol(2)};
  store_node->outputs[0].attr.strides = {ge::Symbol(10), ge::Symbol(2), ge::ops::One};
  bool output_need_align = false;
  EXPECT_TRUE(::ascir::utils::UseSmallTailConcatApi(*concat_node, &output_need_align));
  EXPECT_FALSE(output_need_align);

  // 1.2 not aligned, store with stride
  store_node->outputs[0].attr.strides = {ge::Symbol(20), ge::Symbol(2), ge::ops::One};
  EXPECT_FALSE(::ascir::utils::UseSmallTailConcatApi(*concat_node, &output_need_align));
}

TEST_F(ConcatScheduleCaseGeneratorTest, UseSmallTailConcatApi_NonTailDim_Aligned) {
  ge::AscGraph graph("concat_last_dim_graph");
  CreateConcatAscGraph(graph, {"s0"}, {"1", "1", "1", "1", "4"}, {"2"});
  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);
  auto store_node = graph.FindNode("store");
  ASSERT_TRUE(store_node != nullptr);
  store_node->outputs[0].attr.repeats = {ge::Symbol("s0"), ge::Symbol(5), ge::Symbol(2)};
  store_node->outputs[0].attr.strides = {ge::Symbol(10), ge::Symbol(2), ge::ops::One};
  bool output_need_align = false;
  EXPECT_TRUE(::ascir::utils::UseSmallTailConcatApi(*concat_node, &output_need_align));
  EXPECT_FALSE(output_need_align);

  store_node->outputs[0].attr.strides = {ge::Symbol(20), ge::Symbol(2), ge::ops::One};
  EXPECT_TRUE(::ascir::utils::UseSmallTailConcatApi(*concat_node, &output_need_align));
  EXPECT_FALSE(output_need_align);
}

TEST_F(ConcatScheduleCaseGeneratorTest, RecomputeNode) {
  ge::AscGraph graph("concat_last_dim_graph");
  auto s0 = ge::Symbol("s0");
  auto s1_1 = ge::Symbol("s1_1");
  auto s1_2 = ge::Symbol("s1_2");
  auto s1_3 = ge::Symbol("s1_3");

  auto s1 = s1_2 + s1_1 + s1_1 + s1_1 + s1_2;

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z1_1 = graph.CreateAxis("z1_1", s1_1);
  auto z1_2 = graph.CreateAxis("z1_2", s1_2);
  auto z1_3 = graph.CreateAxis("z1_3", s1_3);

  Data data0("data0", graph);
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);

  Data data1("data1", graph);
  data1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  load1.x = data0.y;
  load1.attr.sched.axis = {z0.id, z1_1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1_1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.repeats = {s0, s1_1};
  *load1.y.strides = {s1_1, ge::sym::kSymbolOne};

  Load load2("load2");
  load2.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load2.y.dtype = ge::DT_FLOAT16;
  load2.x = data0.y;
  load2.attr.sched.axis = {z0.id, z1_1.id};
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1_1.id};
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.repeats = {s0, s1_1};
  *load2.y.strides = {s1_1, ge::sym::kSymbolOne};

  Load load3("load3");
  load3.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load3.y.dtype = ge::DT_FLOAT16;
  load3.x = data0.y;
  load3.attr.sched.axis = {z0.id, z1_1.id};
  load3.y.dtype = ge::DT_FLOAT16;
  *load3.y.axis = {z0.id, z1_1.id};
  load3.y.dtype = ge::DT_FLOAT16;
  *load3.y.repeats = {s0, s1_1};
  *load3.y.strides = {s1_1, ge::sym::kSymbolOne};

  Load load4("load4");
  load4.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load4.y.dtype = ge::DT_FLOAT16;
  load4.x = data1.y;
  load4.attr.sched.axis = {z0.id, z1_2.id};
  load4.y.dtype = ge::DT_FLOAT16;
  *load4.y.axis = {z0.id, z1_2.id};
  load4.y.dtype = ge::DT_FLOAT16;
  *load4.y.repeats = {s0, s1_2};
  *load4.y.strides = {s1_2, ge::sym::kSymbolOne};

  Mul mul1("mul1");
  mul1.x1 = load1.y;
  mul1.x2 = load2.y;
  *mul1.y.repeats = {s0, s1_1};

  Mul mul2("mul2");
  mul2.x1 = load1.y;
  mul2.x2 = load3.y;
  *mul2.y.repeats = {s0, s1_1};

  Concat concat("concat");
  concat.attr.api.compute_type = ge::ComputeType::kComputeConcat;
  concat.y.dtype = ge::DT_FLOAT16;
  concat.x = {load4.y, load1.y, mul1.y, mul2.y, load4.y};
  concat.attr.sched.axis = {z0.id, z1.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.axis = {z0.id, z1.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.repeats = {s0, s1};
  *concat.y.strides = {s1, ge::sym::kSymbolOne};

  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);

  ::ascir::utils::DumpGraph(graph, "BeforeGroup");
  optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), ge::SUCCESS);
}

TEST_F(ConcatScheduleCaseGeneratorTest, RecomputeNode_HorizontalFuse) {
  ge::AscGraph graph("concat_last_dim_graph");
  auto s0 = ge::Symbol("s0");
  auto s1_1 = ge::Symbol("s1_1");

  auto s1 = s1_1 + s1_1;
  Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);

  Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.y.dtype = ge::DT_FLOAT16;
  load1.x = data0.y;

  Load load2("load2");
  load2.x = data1.y;
  *load2.y.repeats = {s0, s1_1};

  Mul mul1("mul1");
  mul1.x1 = load1.y;
  mul1.x2 = load1.y;
  *mul1.y.repeats = {s0, s1_1};

  Mul mul2("mul2");
  mul2.x1 = mul1.y;
  mul2.x2 = load2.y;

  Concat concat("concat");
  concat.x = {mul1.y, load2.y};
  *concat.y.repeats = {s0, s1};
  *concat.y.strides = {s1, ge::sym::kSymbolOne};

  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);

  ::ascir::utils::DumpGraph(graph, "BeforeGroup");
  optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), ge::SUCCESS);
  ::ascir::utils::DumpGraph(graph, "AfterGroup");
  ASSERT_FALSE(partitioner.HasRecompute());
}

TEST_F(ConcatScheduleCaseGeneratorTest, RecomputeSingeItemGroups) {
  ge::AscGraph graph("concat_last_dim_graph");
  auto s0 = ge::Symbol("s0");
  auto s1_1 = ge::Symbol("s1_1");

  auto s1 = One + s1_1;

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z1_1 = graph.CreateAxis("z1_1", s1_1);

  Data data0("data0", graph);
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);

  Data data1("data1", graph);
  data1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  load1.x = data0.y;
  load1.attr.sched.axis = {z0.id, z1_1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1_1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.repeats = {s0, One};
  *load1.y.strides = {One, ge::sym::kSymbolOne};

  Load load3("load3");
  load3.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load3.y.dtype = ge::DT_FLOAT16;
  load3.x = data1.y;
  load3.attr.sched.axis = {z0.id, z1_1.id};
  load3.y.dtype = ge::DT_FLOAT16;
  *load3.y.axis = {z0.id, z1_1.id};
  load3.y.dtype = ge::DT_FLOAT16;
  *load3.y.repeats = {s0, s1_1};
  *load3.y.strides = {s1_1, ge::sym::kSymbolOne};

  Mul mul1("mul1");
  mul1.x1 = load1.y;
  mul1.x2 = load3.y;
  *mul1.y.repeats = {s0, s1_1};

  Concat concat("concat");
  concat.attr.api.compute_type = ge::ComputeType::kComputeConcat;
  concat.y.dtype = ge::DT_FLOAT16;
  concat.x = {load1.y, mul1.y};
  concat.attr.sched.axis = {z0.id, z1.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.axis = {z0.id, z1.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.repeats = {s0, s1};
  *concat.y.strides = {s1, ge::sym::kSymbolOne};

  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);

  optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  EXPECT_EQ(partitioner.RecomputeDiffAxes(), SUCCESS);
  EXPECT_TRUE(partitioner.HasRecompute());
}

TEST_F(ConcatScheduleCaseGeneratorTest, BackwardFusionAndRecompute) {
  auto s0 = ge::Symbol(2);
  auto s1 = ge::Symbol(4);
  auto s2 = ge::Symbol(32);
  auto s3_0 = ge::Symbol(128);
  auto s3_1 = ge::Symbol(1);
  auto s3 = s3_0 + s3_1;
  std::vector<Expression> strides_0 = {s1 * s2 * s3_0, s2 * s3_0, s3_0, ge::sym::kSymbolOne};
  std::vector<Expression> strides_1 = {s1 * s2 * s3_0, s2 * s3_0, s3_1, ge::sym::kSymbolOne};
  auto graph = ge::testing::AscGraphBuilder("test_graph")
                   .Loops({s0, s1, s2, s3})
                   .Data("data0", 0, DT_FLOAT)
                   .Load("load0", "data0", {s0, s1, s2, s3_0}, strides_0)
                   .Data("data1", 1, DT_FLOAT)
                   .Load("load1", "data1", {s0, s1, s2, s3_1}, strides_1)
                   .Data("data2", 2, DT_FLOAT)
                   .Load("load2", "data2", {s0, s1, s2, s3_1}, strides_1)
                   .Broadcast("brc0", "load2", {1, 1, 1, 128})
                   .Broadcast("brc1", "load2", {1, 1, 1, 129})
                   .Add("Ne", "load1", "brc0")
                   .Add("add0", "load0", "Ne")
                   .Add("Max0", "add0", "brc0")
                   .Concat("concat0", {"Max0", "load2"})
                   .Add("Ge", "concat0", "brc1")
                   .Add("Where", "Ge", "concat0")
                   .Store("store", "Where")
                   .Output("out", "store")
                   .Build();
  auto compute_graph = ge::AscGraphUtils::GetComputeGraph(graph);

  optimize::ConcatFusionCaseGenerator generator;
  std::vector<AscGraph> generated_graphs;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.Generate(graph, generated_graphs, score_functions), ge::SUCCESS);
  EXPECT_EQ(generated_graphs.size(), 2);
  const auto &converted_graph = generated_graphs[1];
  for (const auto &node : converted_graph.GetAllNodes()) {
    if (node->GetType() == "Load") {
      EXPECT_NE(node->GetOutDataNodesSize(), 0);
    }
  }
}
}  // namespace schedule

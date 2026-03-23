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
#include "gen_model_info.h"
#include "test_fa_ascir_graph.h"
#define private public
#include "expr_gen/generate_tiling_expr.h"
#include "parser/ascend_graph_parser.h"
#undef private
#include "tests/autofuse/ut/att/utils/graph_construct_utils.h"

namespace ge {
namespace ascir {
namespace cg {
Status BuildGatherAscendGraphND(AscGraph &graph) {
  // create default axis
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");
  auto BL = ge::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto ND = ge::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2 = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 1);
      auto gather = Gather("gather1", load1, load2, 0, false);
      auto store1 = Store("store1", gather);
      GE_ASSERT_SUCCESS(att::GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                         {load1, load2, gather, store1}, 2));
      auto output1 = Output("output1", store1);
    }
  }
  auto load2_node = graph.FindNode("load2");
  EXPECT_NE(load2_node, nullptr);
  load2_node->attr.sched.exec_condition = ge::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  att::GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  return ge::SUCCESS;
}

Status BuildReduceAscendGraphND(AscGraph &graph) {
  // create default axis
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");
  auto BL = ge::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto ND = ge::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto reduce_sum = Sum("reduce1", load1);
      auto store1 = Store("store1", reduce_sum);
      GE_ASSERT_SUCCESS(
          att::GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt}, {load1, reduce_sum, store1}, 2));
      auto output1 = Output("output1", store1);
    }
  }
  auto reduce_sum_node = graph.FindNode("reduce1");
  EXPECT_NE(reduce_sum_node, nullptr);
  reduce_sum_node->attr.api.compute_type = ge::ComputeType::kComputeReduce;
  reduce_sum_node->attr.sched.exec_condition = ge::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  att::GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  return ge::SUCCESS;
}
}  // namespace cg
}  // namespace ascir
}  // namespace ge
namespace att {
class TestAscendGraphParser : public ::testing::Test {
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
    graph = std::make_shared<ge::AscGraph>("graph");
    EXPECT_NE(graph, nullptr);
    att::FaBeforeAutoFuse(*graph);
    att::FaAfterScheduler(*graph);
    att::FaAfterQueBufAlloc(*graph);
  }
  void TearDown() override
  {
  }
  std::shared_ptr<ge::AscGraph> graph;
};

TEST_F(TestAscendGraphParser, case1)
{
  ge::AscGraph graph1("graph");
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  EXPECT_EQ(ascend_graph_parser.GraphParser(graph1), ge::SUCCESS);
}

TEST_F(TestAscendGraphParser, test_gather_graph_parse)
{
  ge::AscGraph graph1("gather_graph");
  ASSERT_EQ(ge::ascir::cg::BuildGatherAscendGraphND(graph1), ge::SUCCESS);
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  EXPECT_EQ(ascend_graph_parser.GraphParser(graph1), ge::SUCCESS);
  EXPECT_TRUE(tuning_space->reserve_ub.find("simt_dcache") != tuning_space->reserve_ub.cend());
  EXPECT_EQ(tuning_space->reserve_ub.size(), 2);
  int32_t cache_count = 0;
  for (const auto &node_info : ascend_graph_parser.tuning_space_->node_infos) {
    if (node_info.exec_condition == ge::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis) {
      cache_count++;
    }
  }
  EXPECT_EQ(cache_count, 1);
}

TEST_F(TestAscendGraphParser, test_reduce_graph_parse)
{
  ge::AscGraph graph1("reduce_graph");
  ASSERT_EQ(ge::ascir::cg::BuildReduceAscendGraphND(graph1), ge::SUCCESS);
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  EXPECT_EQ(ascend_graph_parser.GraphParser(graph1), ge::SUCCESS);
  int32_t cache_count = 0;
  for (const auto &node_info : ascend_graph_parser.tuning_space_->node_infos) {
    if (node_info.exec_condition == ge::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis) {
      cache_count++;
    }
  }
  EXPECT_EQ(cache_count, 0);
}

extern AxisPosition ConvertAxisType(const ge::Axis::Type &type);
TEST_F(TestAscendGraphParser, ValidAxisTypes)
{
  EXPECT_EQ(ConvertAxisType(ge::Axis::kAxisTypeOriginal), AxisPosition::ORIGIN);
  EXPECT_EQ(ConvertAxisType(ge::Axis::kAxisTypeBlockOuter), AxisPosition::OUTER);
  EXPECT_EQ(ConvertAxisType(ge::Axis::kAxisTypeBlockInner), AxisPosition::INNER);
  EXPECT_EQ(ConvertAxisType(ge::Axis::kAxisTypeTileOuter), AxisPosition::OUTER);
  EXPECT_EQ(ConvertAxisType(ge::Axis::kAxisTypeTileInner), AxisPosition::INNER);
  EXPECT_EQ(ConvertAxisType(ge::Axis::kAxisTypeMerged), AxisPosition::MERGED);
}

TEST_F(TestAscendGraphParser, InvalidAxisType)
{
  EXPECT_EQ(ConvertAxisType(static_cast<ge::Axis::Type>(-1)), AxisPosition::POSERR);
}

TEST_F(TestAscendGraphParser, BasicOriginAxisParsing)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser parser(tuning_space);

  parser.ParserSchedInfo(*graph);
  parser.ParserOriginAxis(*graph);
  EXPECT_EQ(parser.orig_to_first_vec_id_.size(), 7);
  EXPECT_EQ(parser.orig_axes_info_.size(), 18);
}

TEST_F(TestAscendGraphParser, ComplexSchedInfoParsing)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser parser(tuning_space);

  parser.ParserSchedInfo(*graph);
  EXPECT_EQ(parser.axes_info_.size(), 18);
  EXPECT_EQ(parser.topo_order_node_.size(), 34);
  EXPECT_EQ(parser.graph_sched_info_.size(), 34);
}

TEST_F(TestAscendGraphParser, BasicSubAxisInfoCreation)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser parser(tuning_space);

  ge::Status status = parser.CreateSubAxisInfo(*graph);
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_EQ(parser.sub_axes_info_.size(), 18);
  EXPECT_EQ(parser.parent_axes_info_.size(), 11);
}

TEST_F(TestAscendGraphParser, BasicSubAxisParsing)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser parser(tuning_space);
  auto sub_axis_ptr = std::make_unique<SubAxis>();
  EXPECT_NE(sub_axis_ptr, nullptr);

  EXPECT_GE(graph->GetAllAxis().size(), 1);
  parser.ParserSubAxis(graph->GetAllAxis()[0], sub_axis_ptr);
  EXPECT_EQ(sub_axis_ptr->name, "b");
  EXPECT_FALSE(sub_axis_ptr->is_bind_multi_core);
  EXPECT_FALSE(sub_axis_ptr->enable_pad);
  EXPECT_TRUE(sub_axis_ptr->enable_tail);
  EXPECT_EQ(sub_axis_ptr->axis_type, AxisPosition::ORIGIN);
  EXPECT_FALSE(sub_axis_ptr->is_split);
  EXPECT_EQ(sub_axis_ptr->align, 1);
}

TEST_F(TestAscendGraphParser, TmpBuffer)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser parser(tuning_space);
  std::map<int64_t, Expr> max_tmp_buffers_map;
  std::vector<ge::TmpBuffer> tmp_buffers;
  tmp_buffers = {
      {{CreateExpr(100), 1}},
      {{CreateExpr(200), 2}},
      {{CreateExpr(50), 1}},
      {{CreateExpr(300), 2}}
  };
  parser.SaveTmpBufferInfos("", max_tmp_buffers_map, tmp_buffers);
  tmp_buffers = {
      {{CreateExpr(220), 1}},
      {{CreateExpr(200), 2}},
      {{CreateExpr(50), 1}},
      {{CreateExpr(300), 2}}
  };
  parser.SaveTmpBufferInfos("", max_tmp_buffers_map, tmp_buffers);
  Expr sum = CreateExpr(0.0f);
  for (const auto& pair : max_tmp_buffers_map) {
    sum = sum + pair.second;
  }
  EXPECT_EQ(Str(sum), "770");
}

TEST_F(TestAscendGraphParser, BasicPrioritySetting)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser parser(tuning_space);

  ge::AscGraph graph1("graph1");
  NodeInfo node1;
  node1.node_type = "Load";
  node1.inputs.push_back(std::make_shared<Tensor>());
  node1.outputs.push_back(std::make_shared<Tensor>());
  auto sub_axis1 = std::make_unique<SubAxis>();
  auto sub_axis2 = std::make_unique<SubAxis>();
  node1.inputs[0]->dim_info.emplace_back(sub_axis1.get());
  node1.outputs[0]->dim_info.emplace_back(sub_axis2.get());
  parser.tuning_space_->node_infos.push_back(node1);

  NodeInfo node2;
  node2.node_type = "Store";
  node2.inputs.push_back(std::make_shared<Tensor>());
  node2.outputs.push_back(std::make_shared<Tensor>());
  node2.inputs[0]->dim_info.emplace_back(sub_axis1.get());
  node2.outputs[0]->dim_info.emplace_back(sub_axis2.get());
  parser.tuning_space_->node_infos.push_back(node2);
  parser.orig_axes_info_[1] = {2};
  parser.orig_axes_info_[2] = {3};

  auto sub_axis_ptr1 = std::make_unique<SubAxis>();
  sub_axis_ptr1->name = "sub_axis1";
  parser.sub_axes_info_[1] = std::move(sub_axis_ptr1);
  auto sub_axis_ptr2 = std::make_unique<SubAxis>();
  sub_axis_ptr2->name = "sub_axis2";
  parser.sub_axes_info_[2] = std::move(sub_axis_ptr2);
  auto sub_axis_ptr3 = std::make_unique<SubAxis>();
  sub_axis_ptr3->name = "sub_axis3";
  parser.sub_axes_info_[3] = std::move(sub_axis_ptr3);

  parser.SetAxisPriority(graph1);
  EXPECT_FALSE(parser.sub_axes_info_[1]->is_last);
  EXPECT_FALSE(parser.sub_axes_info_[2]->is_last);
  EXPECT_FALSE(parser.sub_axes_info_[3]->is_last);
}

TEST_F(TestAscendGraphParser, BasicAssemble)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser parser(tuning_space);

  att::ContainerPtr queue_container1 = std::make_shared<att::Queue>("queue_container1");
  parser.queue_containers_[1] = queue_container1;
  att::ContainerPtr queue_container2 = std::make_shared<att::Queue>("queue_container2");
  parser.queue_containers_[2] = queue_container2;
  att::ContainerPtr buf_container1 = std::make_shared<att::Buf>("buf_container1");
  parser.buf_containers_[3] = buf_container1;
  att::ContainerPtr buf_container2 = std::make_shared<att::Buf>("buf_container2");
  parser.buf_containers_[4] = buf_container2;

  parser.AssembleTensorInfos();
  ASSERT_EQ(parser.tuning_space_->containers.size(), 4);
  EXPECT_EQ(parser.tuning_space_->containers[0]->name, "queue_container1");
  EXPECT_EQ(parser.tuning_space_->containers[1]->name, "queue_container2");
  EXPECT_EQ(parser.tuning_space_->containers[2]->name, "buf_container1");
  EXPECT_EQ(parser.tuning_space_->containers[3]->name, "buf_container2");
}

TEST_F(TestAscendGraphParser, NoQueueContainers)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser parser(tuning_space);

  att::ContainerPtr buf_container1 = std::make_shared<att::Buf>("buf_container1");
  parser.buf_containers_[1] = buf_container1;
  att::ContainerPtr buf_container2 = std::make_shared<att::Buf>("buf_container2");
  parser.buf_containers_[2] = buf_container2;

  parser.AssembleTensorInfos();
  ASSERT_EQ(parser.tuning_space_->containers.size(), 2);
  EXPECT_EQ(parser.tuning_space_->containers[0]->name, "buf_container1");
  EXPECT_EQ(parser.tuning_space_->containers[1]->name, "buf_container2");
}

TEST_F(TestAscendGraphParser, GenGlobalContainers)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser parser(tuning_space);

  att::ContainerPtr global_container_gm = std::make_shared<att::GlobalCache>("GlobalContainer-GM");
  parser.global_containers_[HardwareDef::GM] = global_container_gm;

  parser.AssembleTensorInfos();
  ASSERT_EQ(parser.tuning_space_->global_containers.size(), 1);
  EXPECT_EQ(parser.tuning_space_->global_containers[0]->name, "GlobalContainer-GM");
}

TEST_F(TestAscendGraphParser, NoBufContainers)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser parser(tuning_space);

  att::ContainerPtr queue_container1 = std::make_shared<att::Queue>("queue_container1");
  parser.queue_containers_[1] = queue_container1;
  att::ContainerPtr queue_container2 = std::make_shared<att::Queue>("queue_container2");
  parser.queue_containers_[2] = queue_container2;

  parser.AssembleTensorInfos();
  ASSERT_EQ(parser.tuning_space_->containers.size(), 2);
  EXPECT_EQ(parser.tuning_space_->containers[0]->name, "queue_container1");
  EXPECT_EQ(parser.tuning_space_->containers[1]->name, "queue_container2");
}

TEST_F(TestAscendGraphParser, NoContainers)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser parser(tuning_space);

  parser.AssembleTensorInfos();
  ASSERT_EQ(parser.tuning_space_->containers.size(), 0);
}

TEST_F(TestAscendGraphParser, NonSparseScenario)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser parser(tuning_space);

  parser.ParserBlockDimInfo();
  ASSERT_EQ(parser.tuning_space_->block_dims.size(), 0);
}

/*
TEST_F(TestAscendGraphParser, BasicScenario)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser parser(tuning_space);

graph_input_infos.optional_atts.size()  parser.ParserOptionalInfos(*graph);
  ASSERT_EQ(parser.tuning_space_->, 1);
  EXPECT_EQ(parser.tuning_space_->graph_input_infos.optional_atts[1].optional_name, "head_num");
  EXPECT_EQ(parser.tuning_space_->graph_input_infos.optional_atts[1].data_type, "int32_t");
  EXPECT_EQ(parser.tuning_space_->graph_input_infos.optional_atts[1].min_value, "1");
  EXPECT_EQ(parser.tuning_space_->graph_input_infos.optional_atts[1].max_value, "10");
}
*/

TEST_F(TestAscendGraphParser, ConvertToTuningSpace)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser parser(tuning_space);

  ge::Status status = parser.ConvertToTuningSpace(*graph);
  EXPECT_EQ(status, ge::SUCCESS);
  ASSERT_EQ(parser.tuning_space_->sub_axes.size(), 0);
  //ASSERT_EQ(parser.tuning_space_->graph_input_infos.optional_atts.size(), 1);
  ASSERT_EQ(parser.tuning_space_->block_dims.size(), 0);
}

TEST_F(TestAscendGraphParser, case_global_container)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  ge::AscTensorAttr ascir_tensor_info;
  ascir_tensor_info.mem.hardware == ge::MemHardware::kMemHardwareGM;
  ascir_tensor_info.mem.reuse_id = 0;
  EXPECT_EQ(ascend_graph_parser.ConstructGlobalContainer(ascir_tensor_info), ge::SUCCESS);
}

TEST_F(TestAscendGraphParser, TestPrint)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser parser(tuning_space);

  Tensor tensor;
  EXPECT_NE(parser.TuningSpacePrint(tensor), "");

  SubAxis sub_axis;
  EXPECT_NE(parser.TuningSpacePrint(sub_axis), "");

  NodeInfo node;
  EXPECT_NE(parser.TuningSpacePrint(node), "");

  Queue container = Queue("queue");
  EXPECT_NE(parser.TuningSpacePrint(container), "");

  EXPECT_NE(parser.TuningSpacePrint(), "");
}
TEST_F(TestAscendGraphParser, get_need_ub_mc_tradeoff_1_dim)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  TensorPtr tensor = std::make_shared<att::Tensor>();
  tensor->loc = HardwareDef::GM;
  Expr dim1 = CreateExpr("s0");
  tensor->repeat = {dim1};
  tensor->gm_stride = {ge::sym::kSymbolOne};
  NodeInfo node;
  node.outputs.emplace_back(tensor); 
  tuning_space->node_infos.emplace_back(node);
  att::GenerateTilingExpr tiling_expr(tuning_space);
  ModelInfo model_info;
  tiling_expr.UpdateNeedUBMCTradeoff(model_info);
  EXPECT_EQ(model_info.tiling_schedule_config.trade_off_config.is_enable, false);
}

TEST_F(TestAscendGraphParser, get_need_ub_mc_tradeoff)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  TensorPtr tensor = std::make_shared<att::Tensor>();
  tensor->loc = HardwareDef::GM;
  Expr dim1 = CreateExpr("s0");
  Expr dim2 = CreateExpr("s1");
  Expr dim3 = CreateExpr("s2");
  tensor->repeat = {dim1, dim2, dim3};
  tensor->gm_stride = {dim2 * dim3, dim3 * dim2, ge::sym::kSymbolOne};
  NodeInfo node;
  node.outputs.emplace_back(tensor); 
  tuning_space->node_infos.emplace_back(node);
  att::GenerateTilingExpr tiling_expr(tuning_space);
  ModelInfo model_info;
  tiling_expr.UpdateNeedUBMCTradeoff(model_info);
  EXPECT_EQ(model_info.tiling_schedule_config.trade_off_config.is_enable, true);
}

TEST_F(TestAscendGraphParser, get_need_ub_mc_tradeoff_inputs)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  TensorPtr tensor = std::make_shared<att::Tensor>();
  tensor->loc = HardwareDef::GM;
  Expr dim1 = CreateExpr("s0");
  Expr dim2 = CreateExpr("s1");
  Expr dim3 = CreateExpr("s2");
  tensor->repeat = {dim1, dim2, dim3};
  tensor->gm_stride = {dim2 * dim3, dim3 * dim2, ge::sym::kSymbolOne};
  NodeInfo node;
  node.inputs.emplace_back(tensor); 
  tuning_space->node_infos.emplace_back(node);
  att::GenerateTilingExpr tiling_expr(tuning_space);
  ModelInfo model_info;
  tiling_expr.UpdateNeedUBMCTradeoff(model_info);
  EXPECT_EQ(model_info.tiling_schedule_config.trade_off_config.is_enable, true);
}

TEST_F(TestAscendGraphParser, get_need_ub_mc_tradeoff_non_gm)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  TensorPtr tensor = std::make_shared<att::Tensor>();
  tensor->loc = HardwareDef::UB;
  Expr dim1 = CreateExpr("s0");
  Expr dim2 = CreateExpr("s1");
  Expr dim3 = CreateExpr("s2");
  tensor->repeat = {dim1, dim2, dim3};
  tensor->gm_stride = {dim2 * dim3, dim3 * dim2, ge::sym::kSymbolOne};
  NodeInfo node;
  node.outputs.emplace_back(tensor); 
  tuning_space->node_infos.emplace_back(node);
  att::GenerateTilingExpr tiling_expr(tuning_space);
  ModelInfo model_info;
  tiling_expr.UpdateNeedUBMCTradeoff(model_info);
  EXPECT_EQ(model_info.tiling_schedule_config.trade_off_config.is_enable, false);
}

TEST_F(TestAscendGraphParser, get_need_ub_mc_tradeoff_symbol_zero)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  TensorPtr tensor = std::make_shared<att::Tensor>();
  tensor->loc = HardwareDef::UB;
  Expr dim1 = CreateExpr("s0");
  Expr dim2 = CreateExpr("s1");
  Expr dim3 = CreateExpr("s2");
  tensor->repeat = {dim1, dim2, dim3};
  tensor->gm_stride = {dim2 * dim3, ge::sym::kSymbolZero, ge::sym::kSymbolOne};
  NodeInfo node;
  node.outputs.emplace_back(tensor); 
  tuning_space->node_infos.emplace_back(node);
  att::GenerateTilingExpr tiling_expr(tuning_space);
  ModelInfo model_info;
  tiling_expr.UpdateNeedUBMCTradeoff(model_info);
  EXPECT_EQ(model_info.tiling_schedule_config.trade_off_config.is_enable, false);
}
}
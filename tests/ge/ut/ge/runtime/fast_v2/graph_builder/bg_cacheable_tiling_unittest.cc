/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_builder/bg_tiling.h"
#include "graph_builder/bg_platform.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "ge_graph_dsl/graph_dsl.h"
#include "engine/gelocal/inputs_converter.h"
#include "register/node_converter_registry.h"
#include "exe_graph_comparer.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "common/share_graph.h"
#include "engine/aicore/fe_rt2_common.h"
#include "common/topo_checker.h"
#include "common/summary_checker.h"
#include "register/op_tiling_registry.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "common/const_data_helper.h"
#include "kernel/common_kernel_impl/tiling.h"
#include "graph/utils/graph_dump_utils.h"
#include "graph/ge_local_context.h"
#include "depends/runtime/src/runtime_stub.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph_builder/bg_core_type.h"

namespace gert {
namespace bg {
namespace {
using namespace ge;
constexpr char const *kStubJson =
    "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", \"ub_size\": 126464, "
    "\"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}";
constexpr char const *kStubJsonErrorFormat =
    "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", \"ub_size\": 126464, "
    "\"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}, }";
constexpr char const *kExpectAtomicJson =
    "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", \"ub_size\": 126464, "
    "\"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}, \"_workspace_index_list\":[0,1]}";
void AddCompiledJson(const ge::NodePtr &node, bool atomic, const char *json = nullptr) {
  if (json == nullptr) {
    json = kStubJson;
  }

  if (atomic) {
    AttrUtils::SetStr(node->GetOpDesc(), "_atomic_compile_info_json", json);
    AttrUtils::SetInt(node->GetOpDesc(), "atomic_op_para_size", 2048);
  } else {
    AttrUtils::SetStr(node->GetOpDesc(), "compile_info_json", json);
    AttrUtils::SetInt(node->GetOpDesc(), "op_para_size", 2048);
  }
}
/*
 *
 *     add1
 *    /  \
 * data1 data2
 */
ComputeGraphPtr BuildTwoInputsGraph(const std::string &node_type) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE(node_type, node_type)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE(node_type));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  data1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8, 3, 224, 224}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8, 3, 224, 224}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);

  auto data2 = graph->FindNode("data2");
  data2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8, 3, 224, 224}));
  data2->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8, 3, 224, 224}));
  data2->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data2->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);
  data2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  data2->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);

  auto add1 = graph->FindNode(node_type);
  add1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8, 3, 224, 224}));
  add1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8, 3, 224, 224}));
  add1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  add1->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);
  add1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  add1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);

  add1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({8, 3, 224, 224}));
  add1->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({8, 3, 224, 224}));
  add1->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  add1->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(DT_FLOAT);
  add1->GetOpDesc()->MutableInputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  add1->GetOpDesc()->MutableInputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);

  add1->GetOpDesc()->MutableInputDesc(1)->SetShape(GeShape({8, 3, 224, 224}));
  add1->GetOpDesc()->MutableInputDesc(1)->SetOriginShape(GeShape({8, 3, 224, 224}));
  add1->GetOpDesc()->MutableInputDesc(1)->SetDataType(DT_FLOAT);
  add1->GetOpDesc()->MutableInputDesc(1)->SetOriginDataType(DT_FLOAT);
  add1->GetOpDesc()->MutableInputDesc(1)->SetFormat(ge::FORMAT_NCHW);
  add1->GetOpDesc()->MutableInputDesc(1)->SetOriginFormat(ge::FORMAT_NCHW);
  return graph;
}

/*      add2
 *      /    \
 *     add1   \
 *    /     \  \
 * data1    data2
 */
ComputeGraphPtr BuildDifferentOppImplVersionGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("add1", "Add")->NODE("add2", "Add")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1"));
    CHAIN(NODE("data2")->EDGE(0, 1)->NODE("add2"));
  };
  auto graph = ToComputeGraph(g1);

  GeTensorDesc tensor_desc;
  tensor_desc.SetShape(GeShape({8, 3, 224, 224}));
  tensor_desc.SetOriginShape(GeShape({8, 3, 224, 224}));
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetFormat(ge::FORMAT_NCHW);
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);

  const auto &data1 = graph->FindNode("data1");
  *data1->GetOpDescBarePtr()->MutableOutputDesc(0) = tensor_desc;
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);

  auto data2 = graph->FindNode("data2");
  *data2->GetOpDescBarePtr()->MutableOutputDesc(0) = tensor_desc;
  AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 1);
  // add1 没有指定 _opp_path
  auto add1 = graph->FindNode("add1");
  *add1->GetOpDesc()->MutableOutputDesc(0) = tensor_desc;
  *add1->GetOpDesc()->MutableInputDesc(0) = tensor_desc;
  *add1->GetOpDesc()->MutableInputDesc(1) = tensor_desc;
  // add2 指定 _opp_path 为 1
  auto add2 = graph->FindNode("add2");
  *add2->GetOpDesc()->MutableOutputDesc(0) = tensor_desc;
  *add2->GetOpDesc()->MutableInputDesc(0) = tensor_desc;
  *add2->GetOpDesc()->MutableInputDesc(1) = tensor_desc;
  AttrUtils::SetInt(add2->GetOpDesc(), ATTR_NAME_BINARY_SOURCE, 1);

  return graph;
}

class StubCompileInfoJson : public optiling::CompileInfoBase {
 public:
  StubCompileInfoJson(const std::string &json) : json_str_(json) {}
  ~StubCompileInfoJson() {}
  std::string GetJsonStr() {
    return json_str_;
  };

 private:
  std::string json_str_;
};
optiling::CompileInfoPtr StubOpParseFuncV4(const ge::Operator &op, const ge::AscendString &compileinfo) {
  optiling::CompileInfoPtr info = std::make_shared<StubCompileInfoJson>("testStubOpParseFuncV4");
  return info;
}
bool StubOpTilingFuncV4(const ge::Operator &op, const optiling::CompileInfoPtr compile_info,
                        optiling::OpRunInfoV2 &op_run_info) {
  op_run_info.SetTilingKey(11);
  return true;
}

NodePtr BuildAtomicNode(ge::ComputeGraphPtr &graph, const char *json = kStubJson) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", "Data")->EDGE(0, 0)->NODE("DynamicAtomicAddrClean", "DynamicAtomicAddrClean"));
    CHAIN(NODE("data1", "Data")->EDGE(0, 1)->NODE("DynamicAtomicAddrClean", "DynamicAtomicAddrClean"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 2)->NODE("DynamicAtomicAddrClean", "DynamicAtomicAddrClean"));
  };
  graph = ToComputeGraph(g1);
  auto data0 = graph->FindNode("data0");
  AttrUtils::SetInt(data0->GetOpDesc(), "index", 0);
  auto data1 = graph->FindNode("data1");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 1);
  auto data2 = graph->FindNode("data2");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 2);

  auto aac = graph->FindNode("DynamicAtomicAddrClean");
  aac->GetOpDesc()->SetExtAttr<std::vector<NodePtr>>("holder", {data0, data1, data2});
  AttrUtils::SetListInt(aac->GetOpDesc(), "WorkspaceIndexes", {0, 1});
  if (json != nullptr) {
    AddCompiledJson(aac, true, json);
  }
  return aac;
}

struct FakeArgsInfo {
  ::domi::ArgsInfo::ArgsType arg_type;
  ::domi::ArgsInfo::ArgsFormat arg_fmt;
  int32_t statrt_index;
  uint32_t arg_num;
};
LoweringGlobalData GlobalDataWithArgsInfo(const NodePtr &aicore_node, const LoweringGlobalData &origin_global_data,
                                          std::vector<std::vector<FakeArgsInfo>> &args_infos) {
  LoweringGlobalData global_data;
  LoweringGlobalData::NodeCompileResult aicore_node_compile_result;
  auto task_defs = origin_global_data.FindCompiledResult(aicore_node)->GetTaskDefs();
  if (task_defs.size() != args_infos.size()) {
    GELOGE(ge::PARAM_INVALID,
           "task_defs.size()[%zu] must be equal to args_infos.size()[%zu], return origin global data instead.",
           task_defs.size(), args_infos.size());
    return origin_global_data;
  }
  for (size_t i = 0U; i < task_defs.size(); ++i) {
    auto task_def = task_defs[i];
    auto kernel_def = task_def.mutable_kernel();
    for (size_t idx = 0U; idx < args_infos[i].size(); ++idx) {
      auto args_info_ = kernel_def->add_args_info();
      args_info_->set_arg_type(args_infos[i][idx].arg_type);
      args_info_->set_arg_format(args_infos[i][idx].arg_fmt);
      args_info_->set_start_index(args_infos[i][idx].statrt_index);
      args_info_->set_size(args_infos[i][idx].arg_num);
    }
    aicore_node_compile_result.task_defs.push_back(task_def);
  }
  global_data.AddCompiledResult(aicore_node, aicore_node_compile_result);
  return global_data;
}
}  // namespace

class BgCacheableTilingUT : public BgTestAutoCreateFrame {
 public:
  void TilingTopoCorrect(ge::ExecuteGraph *exe_graph, const std::vector<ValueHolderPtr> &tiling_rets,
                         const std::vector<ValueHolderPtr> &io_shapes, const ValueHolderPtr &platform) {
    for (const auto &tiling_ret : tiling_rets) {
      ASSERT_NE(tiling_ret, nullptr);
    }
    std::vector<FastSrcNode> expect_from;
    for (const auto &io_shape : io_shapes) {
      expect_from.emplace_back(io_shape);
    }
    expect_from.emplace_back("TilingParse");
    expect_from.emplace_back(platform);
    // UT中未执行CEM，因此PrepareTilingFwkData还在main图上，需要将校验对象InnerData替换为PrepareTilingFwkData
    expect_from.emplace_back("PrepareCacheableTilingFwkData");
    expect_from.emplace_back("InnerData");
    expect_from.emplace_back("InnerData");
    ASSERT_EQ(FastNodeTopoChecker(tiling_rets[0]).StrictConnectFrom(expect_from), "success");
    auto tiling_parse_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "TilingParse");
    ASSERT_NE(tiling_parse_node, nullptr);
    ASSERT_EQ(FastNodeTopoChecker(tiling_parse_node).StrictConnectFrom({{"Const"}, {"Data"}, {"Const"}, {"Const"}}),
              "success");
    auto find_tiling_func_node =
        ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "FindTilingFunc");
    ASSERT_NE(find_tiling_func_node, nullptr);
    ASSERT_EQ(FastNodeTopoChecker(find_tiling_func_node).StrictConnectFrom({{"Const"}, {"GetSpaceRegistry"}}),
              "success");
    auto tiling_fwk_data_node =
        ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "PrepareCacheableTilingFwkData");
    // Check "FindTilingFunc" in init graph link to "PrepareCacheableTilingFwkData" in main graph (ut only)
    ConnectFromInitToMain(find_tiling_func_node, 0, tiling_fwk_data_node, 0);
  }

  void CompatibleTopoCorrect(ge::ExecuteGraph *exe_graph, const std::vector<ValueHolderPtr> &tiling_ret,
                             const std::vector<ValueHolderPtr> &io_shapes) {
    for (const auto &tr : tiling_ret) {
      ASSERT_NE(tr, nullptr);
    }

    auto find_tiling_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "FindCompatibleTilingFunc");
    ASSERT_NE(find_tiling_node, nullptr);
    EXPECT_EQ(FastNodeTopoChecker(find_tiling_node).StrictConnectFrom({{"Const"}}), "success");

    auto tiling_parse_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "CompatibleTilingParse");
    ASSERT_NE(tiling_parse_node, nullptr);
    EXPECT_EQ(
        FastNodeTopoChecker(tiling_parse_node)
            .StrictConnectFrom({{"CreateOpFromBuffer"},
                                {"Const"},  // compile_info_json_holder
                                {"Const"},  // compile_info_key_holder
                                {"FindCompatibleTilingFunc",
                                 static_cast<size_t>(kernel::FindCompatibleTilingFuncOutputIndex::kTilingVersion)},
                                {"FindCompatibleTilingFunc",
                                 static_cast<size_t>(kernel::FindCompatibleTilingFuncOutputIndex::kTilingParseFunc)}}),
        "success");
    ASSERT_NE(tiling_ret[0], nullptr);
    auto tiling_node = tiling_ret[0]->GetFastNode();
    ASSERT_NE(tiling_node, nullptr);

    std::vector<FastSrcNode> expect_tiling_node_from = {
        {"CreateOpFromBuffer"},
        {"CompatibleTilingParse", 0},
        {"FindCompatibleTilingFunc", static_cast<size_t>(kernel::FindCompatibleTilingFuncOutputIndex::kTilingVersion)},
        {"PrepareTilingFwkData", 0}};
    for (const auto &io_shape : io_shapes) {
      expect_tiling_node_from.emplace_back(io_shape);
    }
    EXPECT_EQ(FastNodeTopoChecker(tiling_node).StrictConnectFrom(expect_tiling_node_from), "success");
    auto tiling_fwk_data_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "PrepareTilingFwkData");
    ASSERT_NE(tiling_fwk_data_node, nullptr);
    std::vector<FastSrcNode> fwk_data_expect_nodes = {
        {"FindCompatibleTilingFunc", static_cast<size_t>(kernel::FindCompatibleTilingFuncOutputIndex::kTilingFunc)},
        {"AllocLaunchArg", static_cast<size_t>(AllocLaunchArgOutputs::kRtArg)}};
    EXPECT_EQ(FastNodeTopoChecker(tiling_fwk_data_node).StrictConnectFrom(fwk_data_expect_nodes, true), "success");
  }

 protected:
  void SetUp() override {
    BgTestAutoCreateFrame::SetUp();
    auto init = ValueHolder::CreateVoid<bg::ValueHolder>("Init", {});
    auto main = ValueHolder::CreateVoid<bg::ValueHolder>("Main", {});
    auto de_init = ValueHolder::CreateVoid<bg::ValueHolder>("DeInit", {});

    ValueHolder::PushGraphFrame(init, "Init");
    init_frame_ = ValueHolder::PopGraphFrame({}, {});

    ValueHolder::PushGraphFrame(de_init, "DeInit");
    de_init_frame_ = ValueHolder::PopGraphFrame();

    ValueHolder::PushGraphFrame(main, "Main");
    auto launch_arg_output =
        bg::ValueHolder::CreateDataOutput("AllocLaunchArg", {}, static_cast<size_t>(AllocLaunchArgOutputs::kNum));
    fake_launch_arg_ = launch_arg_output[static_cast<size_t>(AllocLaunchArgOutputs::kRtArg)];
  }

  void TearDown() override {
    BgTest::TearDown();
    init_frame_.reset();
    de_init_frame_.reset();
    fake_launch_arg_ = nullptr;
  }

  std::unique_ptr<GraphFrame> init_frame_;
  std::unique_ptr<GraphFrame> de_init_frame_;
  ValueHolderPtr fake_launch_arg_;
};

TEST_F(BgCacheableTilingUT, BgTiling_Ok_TopoCorrectSameNodeWithDifferentOppImplVersion) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildDifferentOppImplVersionGraph();
  ASSERT_NE(graph, nullptr);
  auto add_node1 = graph->FindNode("add1");
  ASSERT_NE(add_node1, nullptr);
  AddCompiledJson(add_node1, false);
  auto add_node2 = graph->FindNode("add2");
  ASSERT_NE(add_node2, nullptr);
  AddCompiledJson(add_node2, false);

  // 构造 global_data 并设置 opp, opp_kernel 的 space_registry
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  auto space_registry_array = SpaceRegistryFaker().BuildRegistryArray();
  global_data.SetSpaceRegistriesV2(*space_registry_array);

  auto launch_arg_output =
      bg::ValueHolder::CreateDataOutput("AllocLaunchArg", {}, static_cast<size_t>(AllocLaunchArgOutputs::kNum));
  auto fake_launch_arg2 = launch_arg_output[static_cast<size_t>(AllocLaunchArgOutputs::kRtArg)];
  bg::LowerConstDataNode(global_data);
  auto tiling_rets1 = Tiling(add_node1, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto tiling_rets2 = Tiling(add_node2, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg2});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph, "CachebaleTilingUT");

  // check main 图中节点数量, 由于没走CEM, main图上有俩InnerData连给PrepareCacheableTilingFwkData
  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 6},
                                                                  {"CacheableTiling", 2},
                                                                  {"TilingAppendDfxInfo", 2},
                                                                  {"TilingParse", 2},
                                                                  {"InnerData", 4},
                                                                  {"SplitRtStreams", 1},
                                                                  {"Const", 23},
                                                                  {"CalcTensorSizeFromStorage", 2},
                                                                  {"PrepareCacheableTilingFwkData", 2},
                                                                  {"AllocLaunchArg", 2}}),
            "success");
  ge::DumpGraph(init_frame_->GetExecuteGraph().get(), "CachebableTilingUTInit");

  // check init 图中节点数量
  ASSERT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictAllNodeTypes({
                    {"InnerNetOutput", 1},
                    {"FindTilingFunc", 2},
                    {"ConstData", 3},
                    {"Data", 1},
                    {"Const", 8},
                    {"SplitRtStreams", 1},
                    {"GetSpaceRegistry", 2},
                }),
            "success");
  // check topo连接关系，包括子图内部以及子图之间
  TilingTopoCorrect(exe_graph, tiling_rets1, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets1.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
  TilingTopoCorrect(exe_graph, tiling_rets2, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets2.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
}

TEST_F(BgCacheableTilingUT, BgTiling_TopoCorrect_KnownWorkspace) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node, false);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  auto op_desc = add_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetWorkspaceBytes({4, 8});
  AttrUtils::SetStr(op_desc, ATTR_NAME_ALIAS_ENGINE_NAME, "mix_l2");

  auto tiling_rets = Tiling(add_node, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph, "CachebaleTilingUT");

  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 6},
                                                                  {"CacheableTiling", 1},
                                                                  {"TilingParse", 1},
                                                                  {"InnerData", 3},
                                                                  {"SplitRtStreams", 1},
                                                                  {"Const", 13},
                                                                  {"TilingAppendWorkspace", 1},
                                                                  {"TilingAppendDfxInfo", 1},
                                                                  {"CalcTensorSizeFromStorage", 1},
                                                                  {"PrepareCacheableTilingFwkData", 1},
                                                                  {"AllocLaunchArg", 1}}),
            "success");
  TilingTopoCorrect(exe_graph, tiling_rets, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
}

TEST_F(BgCacheableTilingUT, BgTiling_TopoCorrect_MemCheck) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node, false);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  auto op_desc = add_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetWorkspaceBytes({4, 8});
  AttrUtils::SetStr(op_desc, ATTR_NAME_ALIAS_ENGINE_NAME, "mix_l2");
  ge::AttrUtils::SetBool(op_desc, "_memcheck", true);
  auto tiling_rets = Tiling(add_node, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  DumpGraph(exe_graph, "CachebaleTilingUT");

  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 6},
                                                                  {"CacheableTiling", 1},
                                                                  {"TilingParse", 1},
                                                                  {"InnerData", 3},
                                                                  {"SplitRtStreams", 1},
                                                                  {"Const", 13},
                                                                  {"TilingAppendWorkspace", 1},
                                                                  {"TilingAppendDfxInfo", 1},
                                                                  {"CalcTensorSizeFromStorage", 1},
                                                                  {"PrepareCacheableTilingFwkData", 1},
                                                                  {"AllocLaunchArg", 1}}),
            "success");
  TilingTopoCorrect(exe_graph, tiling_rets, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
}

TEST_F(BgCacheableTilingUT, BgTiling_TopoCorrect_MemCheck2) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node, false);
  auto op_desc = add_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  op_desc->AppendIrInput("input0", kIrInputDynamic);
  op_desc->AppendIrInput("input1", kIrInputDynamic);
  op_desc->AppendIrOutput("output", kIrOutputDynamic);
  (void)ge::AttrUtils::SetStr(op_desc, "dynamicParamMode", "folded_with_desc");
  (void)ge::AttrUtils::SetListListInt(op_desc, "_dynamic_inputs_indexes", {{1}});
  (void)ge::AttrUtils::SetListListInt(op_desc, "_dynamic_outputs_indexes", {{0}});
  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  domi::FftsPlusCtxDef *ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
  ctx_def->set_op_index(op_desc->GetId());
  ctx_def->set_context_id(0);
  ctx_def->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIV));
  domi::FftsPlusMixAicAivCtxDef *mixctx_def = ctx_def->mutable_mix_aic_aiv_ctx();
  mixctx_def->set_args_format("{i0}{i_instance0}{i_desc1}{o_desc0}{o_instance0}{ws0}{overflow_addr}");

  LoweringGlobalData global_data;
  global_data.AddCompiledResult(add_node, {{task_def}});
  auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  bg::LowerConstDataNode(global_data);
  auto task_defs = global_data.FindCompiledResult(add_node)->GetTaskDefs();
  auto task_def_tmp = task_defs[0];
  auto kernel_def_tmp = task_def_tmp.mutable_kernel();
  GELOGI("arg format: %s", kernel_def_tmp->context().args_format().c_str());

  op_desc->SetWorkspaceBytes({4, 8});
  ge::AttrUtils::SetStr(op_desc, ATTR_NAME_ALIAS_ENGINE_NAME, "mix_l2");
  ge::AttrUtils::SetBool(op_desc, "_memcheck", true);
  auto tiling_rets = Tiling(add_node, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  DumpGraph(exe_graph, "CachebaleTilingUT");

  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 4},
                                                                  {"CacheableTiling", 1},
                                                                  {"TilingParse", 1},
                                                                  {"InnerData", 3},
                                                                  {"Const", 12},
                                                                  {"TilingAppendWorkspace", 1},
                                                                  {"TilingAppendDfxInfo", 1},
                                                                  {"CalcTensorSizeFromStorage", 1},
                                                                  {"PrepareCacheableTilingFwkData", 1},
                                                                  {"AllocLaunchArg", 1}}),
            "success");
  TilingTopoCorrect(exe_graph, tiling_rets, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
}

TEST_F(BgCacheableTilingUT, BgTiling_TopoCorrect_MemCheck3) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node, false);
  auto op_desc = add_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  op_desc->AppendIrInput("input0", kIrInputDynamic);
  op_desc->AppendIrInput("input1", kIrInputDynamic);
  op_desc->AppendIrOutput("output", kIrOutputRequired);
  (void)ge::AttrUtils::SetStr(op_desc, "dynamicParamMode", "folded_with_desc");
  (void)ge::AttrUtils::SetListListInt(op_desc, "_dynamic_inputs_indexes", {{1}});
  (void)ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 3);
  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ge::ModelTaskType::MODEL_TASK_KERNEL));
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->set_stub_func("stub_func");
  kernel_def->set_args_size(64);
  string args(64, '1');
  kernel_def->set_args(args.data(), 64);
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_op_index(op_desc->GetId());
  context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  context->set_args_format("{i0}{i_desc1}{i_instance0}{o0}{o_instance0}{ws0}{overflow_addr}");

  LoweringGlobalData global_data;
  global_data.AddCompiledResult(add_node, {{task_def}});
  auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  bg::LowerConstDataNode(global_data);
  auto task_defs = global_data.FindCompiledResult(add_node)->GetTaskDefs();
  auto task_def_tmp = task_defs[0];
  auto kernel_def_tmp = task_def_tmp.mutable_kernel();
  GELOGI("arg format: %s", kernel_def_tmp->context().args_format().c_str());

  op_desc->SetWorkspaceBytes({4, 8});
  ge::AttrUtils::SetBool(op_desc, "_memcheck", true);
  auto tiling_rets = Tiling(add_node, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  DumpGraph(exe_graph, "CachebaleTilingUT");

  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 4},
                                                                  {"CacheableTiling", 1},
                                                                  {"TilingParse", 1},
                                                                  {"InnerData", 3},
                                                                  {"Const", 12},
                                                                  {"TilingAppendWorkspace", 1},
                                                                  {"TilingAppendDfxInfo", 1},
                                                                  {"CalcTensorSizeFromStorage", 1},
                                                                  {"PrepareCacheableTilingFwkData", 1},
                                                                  {"AllocLaunchArg", 1}}),
            "success");
  TilingTopoCorrect(exe_graph, tiling_rets, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
}

TEST_F(BgCacheableTilingUT, BgTiling_TopoCorrect_MemCheck4) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node, false);
  auto op_desc = add_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  op_desc->AppendIrInput("input0", kIrInputDynamic);
  op_desc->AppendIrInput("input1", kIrInputDynamic);
  op_desc->AppendIrOutput("output", kIrOutputRequired);
  (void)ge::AttrUtils::SetStr(op_desc, "dynamicParamMode", "folded_with_desc");
  (void)ge::AttrUtils::SetListListInt(op_desc, "_dynamic_inputs_indexes", {{1}});
  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ge::ModelTaskType::MODEL_TASK_ALL_KERNEL));
  domi::KernelDefWithHandle *kernel_def = task_def.mutable_kernel_with_handle();
  kernel_def->set_args_size(64);
  string args(64, '1');
  kernel_def->set_args(args.data(), 64);
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_op_index(op_desc->GetId());
  context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  context->set_args_format("{i0}{i_instance0}{i_desc1}{o0}{o_instance0}{ws0}{overflow_addr}");

  LoweringGlobalData global_data;
  global_data.AddCompiledResult(add_node, {{task_def}});
  auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  bg::LowerConstDataNode(global_data);
  auto task_defs = global_data.FindCompiledResult(add_node)->GetTaskDefs();
  auto task_def_tmp = task_defs[0];
  auto kernel_def_tmp = task_def_tmp.mutable_kernel();
  GELOGI("arg format: %s", kernel_def_tmp->context().args_format().c_str());

  op_desc->SetWorkspaceBytes({4, 8});
  ge::AttrUtils::SetBool(op_desc, "_memcheck", true);
  auto tiling_rets = Tiling(add_node, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  DumpGraph(exe_graph, "CachebaleTilingUT");

  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 4},
                                                                  {"CacheableTiling", 1},
                                                                  {"TilingParse", 1},
                                                                  {"InnerData", 3},
                                                                  {"Const", 12},
                                                                  {"TilingAppendWorkspace", 1},
                                                                  {"TilingAppendDfxInfo", 1},
                                                                  {"CalcTensorSizeFromStorage", 1},
                                                                  {"PrepareCacheableTilingFwkData", 1},
                                                                  {"AllocLaunchArg", 1}}),
            "success");
  TilingTopoCorrect(exe_graph, tiling_rets, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
}

TEST_F(BgCacheableTilingUT, BgTiling_TopoCorrect_MemCheck5) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node, false);
  auto op_desc = add_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  op_desc->AppendIrInput("input0", kIrInputRequired);
  op_desc->AppendIrInput("input1", kIrInputOptional);
  op_desc->AppendIrOutput("output0", kIrOutputRequired);
  op_desc->AppendIrOutput("output1", kIrOutputRequired);
  (void)ge::AttrUtils::SetListListInt(op_desc, "_dynamic_inputs_indexes", {{1}});
  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ge::ModelTaskType::MODEL_TASK_ALL_KERNEL));
  domi::KernelDefWithHandle *kernel_def = task_def.mutable_kernel_with_handle();
  kernel_def->set_args_size(64);
  string args(64, '1');
  kernel_def->set_args(args.data(), 64);
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_op_index(op_desc->GetId());
  context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  context->set_args_format("{i0}{i_instance0}{}{o0}{}{ws0}{overflow_addr}");

  LoweringGlobalData global_data;
  global_data.AddCompiledResult(add_node, {{task_def}});
  auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  bg::LowerConstDataNode(global_data);
  auto task_defs = global_data.FindCompiledResult(add_node)->GetTaskDefs();
  auto task_def_tmp = task_defs[0];
  auto kernel_def_tmp = task_def_tmp.mutable_kernel();
  GELOGI("arg format: %s", kernel_def_tmp->context().args_format().c_str());

  op_desc->SetWorkspaceBytes({4, 8});
  ge::AttrUtils::SetBool(op_desc, "_memcheck", true);
  auto tiling_rets = Tiling(add_node, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  DumpGraph(exe_graph, "CachebaleTilingUT");

  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 4},
                                                                  {"CacheableTiling", 1},
                                                                  {"TilingParse", 1},
                                                                  {"InnerData", 3},
                                                                  {"Const", 12},
                                                                  {"TilingAppendWorkspace", 1},
                                                                  {"TilingAppendDfxInfo", 1},
                                                                  {"CalcTensorSizeFromStorage", 1},
                                                                  {"PrepareCacheableTilingFwkData", 1},
                                                                  {"AllocLaunchArg", 1}}),
            "success");
  TilingTopoCorrect(exe_graph, tiling_rets, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
}

// Node without new impl and without compatible impl, go new tiling use auto tiling
TEST_F(BgCacheableTilingUT, ConstructAutoTilingOk) {
  std::string node_type = "bg_node_with_auto_tiling";
  auto graph = BuildTwoInputsGraph(node_type);
  auto test_node = graph->FindNode(node_type);
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_JSON, "{ key : value}");
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_KEY, "{}");

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);

  ASSERT_EQ(data1_ret.out_shapes.size(), 1);
  ASSERT_EQ(data2_ret.out_shapes.size(), 1);

  auto add_out_shape = test_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  auto assemble_platform_infos = AppendCoreTypeToPlatform(test_node, data_input.global_data);
  // use data2_out_shape as output_shapes of add_node
  auto tiling_ret = Tiling(test_node, {data1_ret.out_shapes[0], data2_ret.out_shapes[0]}, {data2_ret.out_shapes[0]},
                           {assemble_platform_infos[static_cast<size_t>(bg::AssemblePlatformInfoIndex::kPlatformInfo)],
                            global_data, fake_launch_arg_});
  ASSERT_EQ(tiling_ret.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
  ASSERT_NE(tiling_ret[0], nullptr);
  EXPECT_EQ(tiling_ret[0]->GetFastNode()->GetType(), "CacheableTiling");
}

TEST_F(BgCacheableTilingUT, GetFrameworkOpTypeUsingAutoTilingOk) {
  const string real_node_type = "Add";
  auto graph = ShareGraph::FrameworkOPGraph(real_node_type);
  auto test_node = graph->FindNode("add1");
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_JSON, "{ key : value}");
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_KEY, "{}");

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);

  ASSERT_EQ(data1_ret.out_shapes.size(), 1);
  ASSERT_EQ(data2_ret.out_shapes.size(), 1);

  auto assemble_platform_infos = AppendCoreTypeToPlatform(test_node, data_input.global_data);
  // use data2_out_shape as output_shapes of add_node
  auto tiling_ret = Tiling(test_node, {data1_ret.out_shapes[0], data2_ret.out_shapes[0]}, {data2_ret.out_shapes[0]},
                           {assemble_platform_infos[static_cast<size_t>(bg::AssemblePlatformInfoIndex::kPlatformInfo)],
                            *(data_input.global_data), fake_launch_arg_});
  ASSERT_EQ(tiling_ret.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
  ASSERT_NE(tiling_ret[0], nullptr);
  EXPECT_EQ(tiling_ret[0]->GetFastNode()->GetType(), "CacheableTiling");

  auto find_node =
      ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "FindTilingFunc");
  ASSERT_EQ(find_node->GetInDataNodes().size(), 2);
  auto node_type = *find_node->GetInDataNodes().begin();
  ASSERT_NE(node_type, nullptr);
  ge::Buffer buffer;
  ASSERT_TRUE(ExeGraphComparer::GetAttr(node_type, buffer));
  ASSERT_NE(buffer.GetData(), nullptr);
  EXPECT_STREQ(reinterpret_cast<char *>(buffer.GetData()), "Add");
}

TEST_F(BgCacheableTilingUT, FallibleTiling_Ok_TopoCorrectWithMemCheck) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node, false);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  auto op_desc = add_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetWorkspaceBytes({4, 8});
  AttrUtils::SetStr(op_desc, ATTR_NAME_ALIAS_ENGINE_NAME, "mix_l2");
  ge::AttrUtils::SetBool(op_desc, "_memcheck", true);
  auto tiling_rets =
      FallibleTiling(add_node, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  DumpGraph(exe_graph, "TilingUT");

  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 6},
                                                                  {"TilingParse", 1},
                                                                  {"InnerData", 3},
                                                                  {"SplitRtStreams", 1},
                                                                  {"Const", 12},
                                                                  {"CacheableFallibleTiling", 1},
                                                                  {"TilingAppendDfxInfo", 1},
                                                                  {"CalcTensorSizeFromStorage", 1},
                                                                  {"PrepareCacheableTilingFwkData", 1},
                                                                  {"AllocLaunchArg", 1}}),
            "success");
  TilingTopoCorrect(exe_graph, tiling_rets, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::FallibleTilingExOutputIndex::kFallibleOutputNum));
}

TEST_F(BgCacheableTilingUT, BgTiling_JsonConstCorrect) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node, false);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  auto tiling_rets = Tiling(add_node, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);

  auto tiling_parse_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "TilingParse");
  ASSERT_NE(tiling_parse_node, nullptr);
  auto edge = tiling_parse_node->GetInDataEdgeByIndex(0);
  ASSERT_NE(edge, nullptr);
  auto json_const = edge->src;
  ASSERT_NE(json_const, nullptr);
  ASSERT_EQ(json_const->GetType(), "Const");
  ge::Buffer json_buf;
  ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(json_const->GetOpDescBarePtr(), "value", json_buf));
  ASSERT_EQ(json_buf.GetSize(), strlen(kStubJson) + 1);
  EXPECT_EQ(memcmp(json_buf.GetData(), kStubJson, strlen(kStubJson) + 1), 0);

  edge = tiling_parse_node->GetInDataEdgeByIndex(2);
  ASSERT_NE(edge, nullptr);
  auto type_const = edge->src;
  ASSERT_NE(type_const, nullptr);
  ASSERT_EQ(type_const->GetType(), "Const");
  ge::Buffer type_buf;
  ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(type_const->GetOpDescBarePtr(), "value", type_buf));
  ASSERT_STREQ(reinterpret_cast<const char *>(type_buf.GetData()), "Add");

  auto find_tiling_func_node =
      ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "FindTilingFunc");
  ASSERT_NE(find_tiling_func_node, nullptr);
  edge = find_tiling_func_node->GetInDataEdgeByIndex(0);
  ASSERT_NE(edge, nullptr);
  auto type_const1 = edge->src;
  ASSERT_NE(type_const1, nullptr);
  ASSERT_EQ(type_const1->GetType(), "Const");
  ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(type_const1->GetOpDescBarePtr(), "value", type_buf));
  ASSERT_STREQ(reinterpret_cast<const char *>(type_buf.GetData()), "Add");
}
TEST_F(BgCacheableTilingUT, BgTiling_Failed_NoCompileJson) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  auto tiling_rets = Tiling(add_node, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
  for (const auto &tiling_ret : tiling_rets) {
    ASSERT_EQ(tiling_ret, nullptr);
  }
}
TEST_F(BgCacheableTilingUT, BgTiling_Failed_NodeIsNullptr) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  LoweringGlobalData global_data;
  auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  bg::LowerConstDataNode(global_data);
  auto tiling_rets = Tiling(nullptr, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
  for (const auto &tiling_ret : tiling_rets) {
    ASSERT_EQ(tiling_ret, nullptr);
  }
}

TEST_F(BgCacheableTilingUT, BgAtomicTiling_TopoCorrect) {
  ge::ComputeGraphPtr graph;
  auto aac_node = BuildAtomicNode(graph);
  ASSERT_NE(aac_node, nullptr);

  auto ws_size = ValueHolder::CreateFeed(0);
  auto output_0_size = ValueHolder::CreateFeed(1);
  auto output_1_size = ValueHolder::CreateFeed(2);
  auto launch_arg = ValueHolder::CreateFeed(3);
  LoweringGlobalData global_data;
  auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  bg::LowerConstDataNode(global_data);
  auto tiling_rets = TilingForAtomic(aac_node, ws_size, {output_0_size, output_1_size}, launch_arg, global_data);
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);

  // todo exe graph summary check
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
  for (const auto &tiling_ret : tiling_rets) {
    ASSERT_NE(tiling_ret, nullptr);
  }
  auto tiling_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "Tiling");
  ASSERT_NE(tiling_node, nullptr);
      ASSERT_EQ(FastNodeTopoChecker(tiling_node)
                    .StrictConnectFrom(
                        {ws_size, output_0_size, output_1_size, {"TilingParse"}, {"PrepareTilingFwkData"},
                         {"InnerData"}, {"InnerData"}}),
                "success");
  auto tiling_parse_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "TilingParse");
  ASSERT_NE(tiling_parse_node, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(tiling_parse_node).StrictConnectFrom({{"Const"}, {"InnerData"}, {"Const"}, {"Const"}}),
            "success");

  auto find_tiling_func_node =
      ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "FindTilingFunc");
  ASSERT_NE(find_tiling_func_node, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(find_tiling_func_node).StrictConnectFrom({{"Const"}, {"GetSpaceRegistry"}}), "success");
  auto tiling_fwk_data_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "PrepareTilingFwkData");
  // atomic tiling 没有platform
  ConnectFromInitToMain(find_tiling_func_node, 0, tiling_fwk_data_node, 0);
}
TEST_F(BgCacheableTilingUT, BgAtomicTiling_JsonConstCorrect) {
  ge::ComputeGraphPtr graph;
  auto aac_node = BuildAtomicNode(graph);
  ASSERT_NE(aac_node, nullptr);

  auto ws_size = ValueHolder::CreateFeed(0);
  auto output_0_size = ValueHolder::CreateFeed(1);
  auto output_1_size = ValueHolder::CreateFeed(2);
  auto launch_arg = ValueHolder::CreateFeed(3);
  LoweringGlobalData global_data;
  auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  bg::LowerConstDataNode(global_data);
  auto tiling_rets = TilingForAtomic(aac_node, ws_size, {output_0_size, output_1_size}, launch_arg, global_data);
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);

  auto tiling_parse_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "TilingParse");
  ASSERT_NE(tiling_parse_node, nullptr);
  auto edge = tiling_parse_node->GetInDataEdgeByIndex(0);
  ASSERT_NE(edge, nullptr);
  auto json_const = edge->src;
  ASSERT_NE(json_const, nullptr);
  ASSERT_EQ(json_const->GetType(), "Const");
  ge::Buffer json_buf;
  ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(json_const->GetOpDescBarePtr(), "value", json_buf));
  nlohmann::basic_json json_actual;
  nlohmann::basic_json json_expect;
  EXPECT_NO_THROW(json_actual = nlohmann::json::parse(json_buf.GetData()));
  EXPECT_NO_THROW(json_expect = nlohmann::json::parse(kExpectAtomicJson));
  EXPECT_EQ(json_actual, json_expect);

  edge = tiling_parse_node->GetInDataEdgeByIndex(2);
  ASSERT_NE(edge, nullptr);
  auto type_const = edge->src;
  ASSERT_NE(type_const, nullptr);
  ASSERT_EQ(type_const->GetType(), "Const");
  ge::Buffer type_buf;
  ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(type_const->GetOpDescBarePtr(), "value", type_buf));
  ASSERT_STREQ(reinterpret_cast<const char *>(type_buf.GetData()), "DynamicAtomicAddrClean");

  auto find_tiling_func_node =
      ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "FindTilingFunc");
  ASSERT_NE(find_tiling_func_node, nullptr);
  edge = find_tiling_func_node->GetInDataEdgeByIndex(0);
  ASSERT_NE(edge, nullptr);
  auto type_const1 = edge->src;
  ASSERT_NE(type_const1, nullptr);
  ASSERT_EQ(type_const1->GetType(), "Const");
  ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(type_const1->GetOpDescBarePtr(), "value", type_buf));
  ASSERT_STREQ(reinterpret_cast<const char *>(type_buf.GetData()), "DynamicAtomicAddrClean");
}
TEST_F(BgCacheableTilingUT, BgAtomicTiling_Failed_NoCompileJson) {
  ge::ComputeGraphPtr graph;
  auto aac_node = BuildAtomicNode(graph, nullptr);
  ASSERT_NE(aac_node, nullptr);

  auto ws_size = ValueHolder::CreateFeed(0);
  auto output_0_size = ValueHolder::CreateFeed(1);
  auto output_1_size = ValueHolder::CreateFeed(2);
  auto launch_arg = ValueHolder::CreateFeed(3);
  LoweringGlobalData global_data;
  auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  bg::LowerConstDataNode(global_data);
  auto tiling_rets = TilingForAtomic(aac_node, ws_size, {output_0_size, output_1_size}, launch_arg, global_data);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
  for (const auto &tiling_ret : tiling_rets) {
    ASSERT_EQ(tiling_ret, nullptr);
  }
}
TEST_F(BgCacheableTilingUT, BgAtomicTiling_Failed_InvalidJonsFormat) {
  ge::ComputeGraphPtr graph;
  auto aac_node = BuildAtomicNode(graph, kStubJsonErrorFormat);
  ASSERT_NE(aac_node, nullptr);

  auto ws_size = ValueHolder::CreateFeed(0);
  auto output_0_size = ValueHolder::CreateFeed(1);
  auto output_1_size = ValueHolder::CreateFeed(2);
  auto launch_arg = ValueHolder::CreateFeed(3);
  LoweringGlobalData global_data;
  auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  bg::LowerConstDataNode(global_data);
  auto tiling_rets = TilingForAtomic(aac_node, ws_size, {output_0_size, output_1_size}, launch_arg, global_data);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
  for (const auto &tiling_ret : tiling_rets) {
    ASSERT_EQ(tiling_ret, nullptr);
  }
}

TEST_F(BgCacheableTilingUT, GetCoreTypeMix) {
  std::string node_type = "bg_node_with_auto_tiling";
  auto graph = BuildTwoInputsGraph(node_type);
  auto test_node = graph->FindNode(node_type);
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_JSON, "{ key : value}");
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_KEY, "{}");

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);

  ASSERT_EQ(data1_ret.out_shapes.size(), 1);
  ASSERT_EQ(data2_ret.out_shapes.size(), 1);

  auto add_out_shape = test_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  ge::AttrUtils::SetStr(test_node->GetOpDesc(), ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX");
  ge::AttrUtils::SetBool(test_node->GetOpDesc(), "_mix_is_aiv", true);
  GetCoreType(test_node, data_input.global_data);
}

TEST_F(BgCacheableTilingUT, GetInputNumUsingAllocRtArgOk) {
  auto graph = ShareGraph::AicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput lower_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                            {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                            &global_data};
  auto compile_result = lower_input.global_data->FindCompiledResult(graph->FindNode("add1"));
  auto &task_def = compile_result->task_defs.back();
  constexpr char const *kMaxTilingDataSize = "op_para_size";
  auto launch_arg = bg::AllocRtArg(graph->FindNode("add1"), task_def.kernel_with_handle(), kMaxTilingDataSize);

  ASSERT_EQ(launch_arg.size(), static_cast<size_t>(AllocLaunchArgOutputs::kNum));
  EXPECT_EQ(launch_arg[0]->GetFastNode()->GetType(), "AllocLaunchArg");
  auto const_node = *(launch_arg[0]->GetFastNode()->GetInDataNodes().begin());
  ASSERT_NE(const_node, nullptr);

  ge::Buffer buffer;
  ASSERT_TRUE(ExeGraphComparer::GetAttr(const_node, buffer));
  ASSERT_NE(buffer.GetData(), nullptr);
  auto node_desc = reinterpret_cast<RtKernelLaunchArgsEx::ComputeNodeDesc *>(buffer.GetData());
  ASSERT_NE(node_desc, nullptr);
  EXPECT_EQ(node_desc->input_num, 2);
}

TEST_F(BgCacheableTilingUT, AllocRtArgWithArgsInfo1_SUCCESS) {
  auto graph = ShareGraph::AddWith4InputsAicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto ori_global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("Add", false).Build();
  auto add_node = graph->FindNode("add1");
  std::vector<FakeArgsInfo> args_info{
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, -1, 0},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_SECONDARY_ADDR, 1, 2},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_SECONDARY_ADDR, 3, 1},
      {::domi::ArgsInfo_ArgsType_OUTPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1}};
  std::vector<std::vector<FakeArgsInfo>> args_infos{args_info};
  auto global_data = GlobalDataWithArgsInfo(add_node, ori_global_data, args_infos);
  LowerInput lower_input = {{}, {}, &global_data};

  auto compile_result = lower_input.global_data->FindCompiledResult(add_node);
  auto &task_def = compile_result->task_defs.back();
  constexpr char const *kMaxTilingDataSize = "op_para_size";
  auto launch_arg = bg::AllocRtArg(add_node, task_def.kernel(), kMaxTilingDataSize);

  ASSERT_EQ(launch_arg.size(), static_cast<size_t>(AllocLaunchArgOutputs::kNum));
  EXPECT_EQ(launch_arg[0]->GetFastNode()->GetType(), "AllocLaunchArg");
  auto compute_node_desc_const_node = *(launch_arg[0]->GetFastNode()->GetInDataNodes().begin());
  ASSERT_NE(compute_node_desc_const_node, nullptr);

  ge::Buffer buffer;
  ASSERT_TRUE(ExeGraphComparer::GetAttr(compute_node_desc_const_node, buffer));
  ASSERT_NE(buffer.GetData(), nullptr);
  auto node_desc = reinterpret_cast<RtKernelLaunchArgsEx::ComputeNodeDesc *>(buffer.GetData());
  ASSERT_NE(node_desc, nullptr);
  EXPECT_EQ(node_desc->input_num, 4);
  EXPECT_EQ(node_desc->output_num, 1);

  auto args_info_desc_const_node = *(launch_arg[0]->GetFastNode()->GetInDataNodes().begin() + 1);
  ASSERT_NE(args_info_desc_const_node, nullptr);
  ASSERT_TRUE(ExeGraphComparer::GetAttr(args_info_desc_const_node, buffer));
  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(buffer.GetData());

  EXPECT_EQ(sizeof(ArgsInfosDesc) + args_info_desc->GetArgsInfoSize(), buffer.GetSize());

  EXPECT_EQ(args_info_desc->GetInputArgsInfoNum(), 4);
  EXPECT_EQ(args_info_desc->GetOutputArgsInfoNum(), 1);

  EXPECT_EQ(args_info_desc->GetArgsInfoNum(), 5);
  auto args_infos_buffer = reinterpret_cast<ArgsInfosDesc::ArgsInfo *>(args_info_desc->MutableArgsInfoBase());
  EXPECT_EQ(args_infos_buffer[0].arg_type, ArgsInfosDesc::ArgsInfo::ArgsInfoType::INPUT);
  EXPECT_EQ(args_infos_buffer[0].arg_format, ArgsInfosDesc::ArgsInfo::ArgsInfoFormat::DIRECT_ADDR);
  EXPECT_EQ(args_infos_buffer[0].arg_size, 1);
  EXPECT_EQ(args_infos_buffer[0].start_index, 0);
  EXPECT_EQ(args_infos_buffer[1].arg_type, ArgsInfosDesc::ArgsInfo::ArgsInfoType::INPUT);
  EXPECT_EQ(args_infos_buffer[1].arg_format, ArgsInfosDesc::ArgsInfo::ArgsInfoFormat::DIRECT_ADDR);
  EXPECT_EQ(args_infos_buffer[1].arg_size, 0);
  EXPECT_EQ(args_infos_buffer[1].start_index, -1);
  EXPECT_EQ(args_infos_buffer[2].arg_type, ArgsInfosDesc::ArgsInfo::ArgsInfoType::INPUT);
  EXPECT_EQ(args_infos_buffer[2].arg_format, ArgsInfosDesc::ArgsInfo::ArgsInfoFormat::FOLDED_DESC_ADDR);
  EXPECT_EQ(args_infos_buffer[2].arg_size, 2);
  EXPECT_EQ(args_infos_buffer[2].start_index, 1);
  EXPECT_EQ(args_infos_buffer[3].arg_type, ArgsInfosDesc::ArgsInfo::ArgsInfoType::INPUT);
  EXPECT_EQ(args_infos_buffer[3].arg_format, ArgsInfosDesc::ArgsInfo::ArgsInfoFormat::FOLDED_DESC_ADDR);
  EXPECT_EQ(args_infos_buffer[3].arg_size, 1);
  EXPECT_EQ(args_infos_buffer[3].start_index, 3);
  EXPECT_EQ(args_infos_buffer[4].arg_type, ArgsInfosDesc::ArgsInfo::ArgsInfoType::OUTPUT);
  EXPECT_EQ(args_infos_buffer[4].arg_format, ArgsInfosDesc::ArgsInfo::ArgsInfoFormat::DIRECT_ADDR);
  EXPECT_EQ(args_infos_buffer[4].arg_size, 1);
  EXPECT_EQ(args_infos_buffer[4].start_index, 0);
}

TEST_F(BgCacheableTilingUT, AllocRtArgWithArgsInfo2_SUCCESS) {
  auto graph = ShareGraph::AddWith4InputsAicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto ori_global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("Add", false).Build();
  auto add_node = graph->FindNode("add1");
  std::vector<FakeArgsInfo> args_info{
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, -1, 0},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_SECONDARY_ADDR, 1, 2},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_SECONDARY_ADDR, 3, 1},
      {::domi::ArgsInfo_ArgsType_OUTPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, -1, 0},
      {::domi::ArgsInfo_ArgsType_OUTPUT, ::domi::ArgsInfo_ArgsFormat_SECONDARY_ADDR, 0, 1},
  };
  std::vector<std::vector<FakeArgsInfo>> args_infos{args_info};
  auto global_data = GlobalDataWithArgsInfo(add_node, ori_global_data, args_infos);
  LowerInput lower_input = {{}, {}, &global_data};

  auto compile_result = lower_input.global_data->FindCompiledResult(add_node);
  auto &task_def = compile_result->task_defs.back();
  constexpr char const *kMaxTilingDataSize = "op_para_size";
  auto launch_arg = bg::AllocRtArg(add_node, task_def.kernel(), kMaxTilingDataSize);

  ASSERT_EQ(launch_arg.size(), static_cast<size_t>(AllocLaunchArgOutputs::kNum));
  EXPECT_EQ(launch_arg[0]->GetFastNode()->GetType(), "AllocLaunchArg");
  auto compute_node_desc_const_node = *(launch_arg[0]->GetFastNode()->GetInDataNodes().begin());
  ASSERT_NE(compute_node_desc_const_node, nullptr);

  ge::Buffer buffer;
  ASSERT_TRUE(ExeGraphComparer::GetAttr(compute_node_desc_const_node, buffer));
  ASSERT_NE(buffer.GetData(), nullptr);
  auto node_desc = reinterpret_cast<RtKernelLaunchArgsEx::ComputeNodeDesc *>(buffer.GetData());
  ASSERT_NE(node_desc, nullptr);
  EXPECT_EQ(node_desc->input_num, 4);
  EXPECT_EQ(node_desc->output_num, 1);

  auto args_info_desc_const_node = *(launch_arg[0]->GetFastNode()->GetInDataNodes().begin() + 1);
  ASSERT_NE(args_info_desc_const_node, nullptr);
  ASSERT_TRUE(ExeGraphComparer::GetAttr(args_info_desc_const_node, buffer));
  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(buffer.GetData());

  EXPECT_EQ(sizeof(ArgsInfosDesc) + args_info_desc->GetArgsInfoSize(), buffer.GetSize());

  EXPECT_EQ(args_info_desc->GetInputArgsInfoNum(), 4);
  EXPECT_EQ(args_info_desc->GetOutputArgsInfoNum(), 2);

  EXPECT_EQ(args_info_desc->GetArgsInfoNum(), 6);
  auto args_infos_buffer = reinterpret_cast<ArgsInfosDesc::ArgsInfo *>(args_info_desc->MutableArgsInfoBase());
  EXPECT_EQ(args_infos_buffer[0].arg_type, ArgsInfosDesc::ArgsInfo::ArgsInfoType::INPUT);
  EXPECT_EQ(args_infos_buffer[0].arg_size, 1);
  EXPECT_EQ(args_infos_buffer[0].start_index, 0);
  EXPECT_EQ(args_infos_buffer[1].arg_type, ArgsInfosDesc::ArgsInfo::ArgsInfoType::INPUT);
  EXPECT_EQ(args_infos_buffer[1].arg_size, 0);
  EXPECT_EQ(args_infos_buffer[1].start_index, -1);
  EXPECT_EQ(args_infos_buffer[2].arg_type, ArgsInfosDesc::ArgsInfo::ArgsInfoType::INPUT);
  EXPECT_EQ(args_infos_buffer[2].arg_size, 2);
  EXPECT_EQ(args_infos_buffer[2].start_index, 1);
  EXPECT_EQ(args_infos_buffer[3].arg_type, ArgsInfosDesc::ArgsInfo::ArgsInfoType::INPUT);
  EXPECT_EQ(args_infos_buffer[3].arg_size, 1);
  EXPECT_EQ(args_infos_buffer[3].start_index, 3);
  EXPECT_EQ(args_infos_buffer[4].arg_type, ArgsInfosDesc::ArgsInfo::ArgsInfoType::OUTPUT);
  EXPECT_EQ(args_infos_buffer[4].arg_size, 0);
  EXPECT_EQ(args_infos_buffer[4].start_index, -1);
  EXPECT_EQ(args_infos_buffer[5].arg_type, ArgsInfosDesc::ArgsInfo::ArgsInfoType::OUTPUT);
  EXPECT_EQ(args_infos_buffer[5].arg_size, 1);
  EXPECT_EQ(args_infos_buffer[5].start_index, 0);
}

TEST_F(BgCacheableTilingUT, AllocRtArg_ArgsInfoNumNotMatchNodeIoNum_SUCCESS) {
  auto graph = ShareGraph::AddWith4InputsAicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto ori_global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("Add", false).Build();
  auto add_node = graph->FindNode("add1");
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), kAttrDynamicParamMode, kFoldedWithDesc);
  std::vector<std::vector<int64_t>> dyn_in_vv = {{1}};
  (void)ge::AttrUtils::SetListListInt(add_node->GetOpDesc(), kDynamicInputsIndexes, dyn_in_vv);
  std::vector<FakeArgsInfo> args_info{
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 1, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 2, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 3, 1},
      {::domi::ArgsInfo_ArgsType_OUTPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1}};
  std::vector<std::vector<FakeArgsInfo>> args_infos{args_info};
  auto global_data = GlobalDataWithArgsInfo(add_node, ori_global_data, args_infos);
  LowerInput lower_input = {{}, {}, &global_data};

  auto compile_result = lower_input.global_data->FindCompiledResult(add_node);
  auto &task_def = compile_result->task_defs.back();
  constexpr char const *kMaxTilingDataSize = "op_para_size";
  auto launch_arg = bg::AllocRtArg(add_node, task_def.kernel(), kMaxTilingDataSize);
  ASSERT_EQ(launch_arg.size(), static_cast<size_t>(AllocLaunchArgOutputs::kNum));
  EXPECT_EQ(launch_arg[0]->GetFastNode()->GetType(), "AllocLaunchArg");
  auto compute_node_desc_const_node = *(launch_arg[0]->GetFastNode()->GetInDataNodes().begin());
  ASSERT_NE(compute_node_desc_const_node, nullptr);

  ge::Buffer buffer;
  ASSERT_TRUE(ExeGraphComparer::GetAttr(compute_node_desc_const_node, buffer));
  ASSERT_NE(buffer.GetData(), nullptr);
  auto node_desc = reinterpret_cast<RtKernelLaunchArgsEx::ComputeNodeDesc *>(buffer.GetData());
  ASSERT_NE(node_desc, nullptr);
  EXPECT_EQ(node_desc->input_num, 4);
  EXPECT_EQ(node_desc->output_num, 1);

  auto args_info_desc_const_node = *(launch_arg[0]->GetFastNode()->GetInDataNodes().begin() + 1);
  ASSERT_NE(args_info_desc_const_node, nullptr);
  ASSERT_TRUE(ExeGraphComparer::GetAttr(args_info_desc_const_node, buffer));
  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(buffer.GetData());

  EXPECT_EQ(sizeof(ArgsInfosDesc) + args_info_desc->GetArgsInfoSize(), buffer.GetSize());

  EXPECT_EQ(args_info_desc->GetInputArgsInfoNum(), 4);
  EXPECT_EQ(args_info_desc->GetOutputArgsInfoNum(), 1);

  EXPECT_EQ(args_info_desc->GetArgsInfoNum(), 5);
  auto args_infos_buffer = reinterpret_cast<ArgsInfosDesc::ArgsInfo *>(args_info_desc->MutableArgsInfoBase());
  EXPECT_EQ(args_infos_buffer[0].arg_type, ArgsInfosDesc::ArgsInfo::ArgsInfoType::INPUT);
  EXPECT_EQ(args_infos_buffer[0].arg_size, 1);
  EXPECT_EQ(args_infos_buffer[0].start_index, 0);
  EXPECT_EQ(args_infos_buffer[1].arg_type, ArgsInfosDesc::ArgsInfo::ArgsInfoType::INPUT);
  EXPECT_EQ(args_infos_buffer[1].arg_size, 1);
  EXPECT_EQ(args_infos_buffer[1].start_index, 1);
  EXPECT_EQ(args_infos_buffer[2].arg_type, ArgsInfosDesc::ArgsInfo::ArgsInfoType::INPUT);
  EXPECT_EQ(args_infos_buffer[2].arg_size, 1);
  EXPECT_EQ(args_infos_buffer[2].start_index, 2);
  EXPECT_EQ(args_infos_buffer[3].arg_type, ArgsInfosDesc::ArgsInfo::ArgsInfoType::INPUT);
  EXPECT_EQ(args_infos_buffer[3].arg_size, 1);
  EXPECT_EQ(args_infos_buffer[3].start_index, 3);
  EXPECT_EQ(args_infos_buffer[4].arg_type, ArgsInfosDesc::ArgsInfo::ArgsInfoType::OUTPUT);
  EXPECT_EQ(args_infos_buffer[4].arg_size, 1);
  EXPECT_EQ(args_infos_buffer[4].start_index, 0);
}

TEST_F(BgCacheableTilingUT, AllocRtArg_CheckArgsInfo_Failed1) {
  auto graph = ShareGraph::AddWith4InputsAicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto ori_global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("Add", false).Build();
  auto add_node = graph->FindNode("add1");
  std::vector<FakeArgsInfo> args_info{
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, -2, 0},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_SECONDARY_ADDR, 1, 2},
      {::domi::ArgsInfo_ArgsType_OUTPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1}};
  std::vector<std::vector<FakeArgsInfo>> args_infos{args_info};
  auto global_data = GlobalDataWithArgsInfo(add_node, ori_global_data, args_infos);
  LowerInput lower_input = {{}, {}, &global_data};

  auto compile_result = lower_input.global_data->FindCompiledResult(add_node);
  auto &task_def = compile_result->task_defs.back();
  constexpr char const *kMaxTilingDataSize = "op_para_size";
  auto launch_arg = bg::AllocRtArg(add_node, task_def.kernel(), kMaxTilingDataSize);
  ASSERT_EQ(launch_arg.size(), 0);
}

TEST_F(BgCacheableTilingUT, AllocRtArg_CheckArgsInfo_Failed2) {
  auto graph = ShareGraph::AddWith4InputsAicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto ori_global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("Add", false).Build();
  auto add_node = graph->FindNode("add1");
  std::vector<FakeArgsInfo> args_info{
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_SECONDARY_ADDR, 1, 5},
      {::domi::ArgsInfo_ArgsType_OUTPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1}};
  std::vector<std::vector<FakeArgsInfo>> args_infos{args_info};
  auto global_data = GlobalDataWithArgsInfo(add_node, ori_global_data, args_infos);
  LowerInput lower_input = {{}, {}, &global_data};

  auto compile_result = lower_input.global_data->FindCompiledResult(add_node);
  auto &task_def = compile_result->task_defs.back();
  constexpr char const *kMaxTilingDataSize = "op_para_size";
  auto launch_arg = bg::AllocRtArg(add_node, task_def.kernel(), kMaxTilingDataSize);
  ASSERT_EQ(launch_arg.size(), 0);
}

TEST_F(BgCacheableTilingUT, AllocRtArg_CheckArgsInfo_Failed3) {
  auto graph = ShareGraph::AddWith4InputsAicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto ori_global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("Add", false).Build();
  auto add_node = graph->FindNode("add1");
  std::vector<FakeArgsInfo> args_info{
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, -1, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_SECONDARY_ADDR, 0, 4},
      {::domi::ArgsInfo_ArgsType_OUTPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1}};
  std::vector<std::vector<FakeArgsInfo>> args_infos{args_info};
  auto global_data = GlobalDataWithArgsInfo(add_node, ori_global_data, args_infos);
  LowerInput lower_input = {{}, {}, &global_data};

  auto compile_result = lower_input.global_data->FindCompiledResult(add_node);
  auto &task_def = compile_result->task_defs.back();
  constexpr char const *kMaxTilingDataSize = "op_para_size";
  auto launch_arg = bg::AllocRtArg(add_node, task_def.kernel(), kMaxTilingDataSize);
  ASSERT_EQ(launch_arg.size(), 0);
}

TEST_F(BgCacheableTilingUT, AllocRtArg_CheckArgsInfo_Failed4) {
  auto graph = ShareGraph::AddWith4InputsAicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto ori_global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("Add", false).Build();
  auto add_node = graph->FindNode("add1");
  std::vector<FakeArgsInfo> args_info{
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 2},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_SECONDARY_ADDR, 2, 2},
      {::domi::ArgsInfo_ArgsType_OUTPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1}};
  std::vector<std::vector<FakeArgsInfo>> args_infos{args_info};
  auto global_data = GlobalDataWithArgsInfo(add_node, ori_global_data, args_infos);
  LowerInput lower_input = {{}, {}, &global_data};

  auto compile_result = lower_input.global_data->FindCompiledResult(add_node);
  auto &task_def = compile_result->task_defs.back();
  constexpr char const *kMaxTilingDataSize = "op_para_size";
  auto launch_arg = bg::AllocRtArg(add_node, task_def.kernel(), kMaxTilingDataSize);
  ASSERT_EQ(launch_arg.size(), 0);
}

TEST_F(BgCacheableTilingUT, AllocRtArg_CheckArgsInfo_Failed_Mix) {
  auto graph = ShareGraph::AddWith4InputsAicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto ori_global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("Add", false).Build();
  auto add_node = graph->FindNode("add1");
  std::vector<FakeArgsInfo> args_info{
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0xFFFF, 2},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_SECONDARY_ADDR, 2, 2},
      {::domi::ArgsInfo_ArgsType_OUTPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1}};
  std::vector<std::vector<FakeArgsInfo>> args_infos{args_info};
  auto global_data = GlobalDataWithArgsInfo(add_node, ori_global_data, args_infos);
  LowerInput lower_input = {{}, {}, &global_data};

  auto compile_result = lower_input.global_data->FindCompiledResult(add_node);
  auto &task_def = compile_result->task_defs.back();
  constexpr char const *kMaxTilingDataSize = "op_para_size";
  auto launch_arg = bg::AllocRtArg(add_node, task_def.kernel(), kMaxTilingDataSize);
  ASSERT_EQ(launch_arg.size(), 4);
}

TEST_F(BgCacheableTilingUT, BgFallibleTiling_Failed_WhenInputsIsNullptr) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node, false);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  auto tiling_rets =
      FallibleTiling(nullptr, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  ASSERT_EQ(tiling_rets[0], nullptr);
}

TEST_F(BgCacheableTilingUT, BgTilingLegacy_Ok_CompatibleTilingV4) {
  std::string node_type = "bg_node_with_tiling_v4";
  auto graph = BuildTwoInputsGraph(node_type);
  auto test_node = graph->FindNode(node_type);
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_JSON, "{ key : value}");
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_KEY, "{}");
  // mock tiling func v4
  optiling::OpTilingFuncRegistry(test_node->GetType(), StubOpTilingFuncV4, StubOpParseFuncV4);

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);

  ASSERT_EQ(data1_ret.out_shapes.size(), 1);
  ASSERT_EQ(data2_ret.out_shapes.size(), 1);

  auto add_out_shape = test_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  auto assemble_platform_infos = AppendCoreTypeToPlatform(test_node, data_input.global_data);
  // 1. tiling input node is null
  auto tiling_ret = TilingLegacy(
      nullptr, {data1_ret.out_shapes[0], data2_ret.out_shapes[0]}, {data2_ret.out_shapes[0]},
      assemble_platform_infos[static_cast<size_t>(bg::AssemblePlatformInfoIndex::kPlatformInfo)], global_data);
  ASSERT_EQ(tiling_ret.size(), TilingContext::kOutputNum);
  EXPECT_EQ(tiling_ret[0], nullptr);
  // 2. build tiling successfully
  tiling_ret = TilingLegacy(test_node, {data1_ret.out_shapes[0], data2_ret.out_shapes[0]}, {data2_ret.out_shapes[0]},
                            assemble_platform_infos[static_cast<size_t>(bg::AssemblePlatformInfoIndex::kPlatformInfo)],
                            global_data);
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(tiling_ret.size(), TilingContext::kOutputNum);
  EXPECT_EQ(tiling_ret[0]->GetFastNode()->GetType(), "CompatibleTilingLegacy");

  // check data
  auto find_tiling_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "FindCompatibleTilingFunc");
  ASSERT_NE(find_tiling_node, nullptr);
  auto exe_node_type = FastNodeTopoChecker(find_tiling_node).InChecker().DataFromByType("Const").GetFastNode();
  ge::Buffer buffer;
  ASSERT_TRUE(ExeGraphComparer::GetAttr(exe_node_type, buffer));
  ASSERT_NE(buffer.GetData(), nullptr);
  EXPECT_STREQ(reinterpret_cast<char *>(buffer.GetData()), node_type.c_str());
}

TEST_F(BgCacheableTilingUT, BgTiling_TopoCorrect_CompatibleTilingV4) {
  std::string node_type = "bg_node_with_tiling_v4";
  auto graph = BuildTwoInputsGraph(node_type);
  auto test_node = graph->FindNode(node_type);
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_JSON, "{ key : value}");
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_KEY, "{}");
  // mock tiling func v4
  optiling::OpTilingFuncRegistry(test_node->GetType(), StubOpTilingFuncV4, StubOpParseFuncV4);

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);

  ASSERT_EQ(data1_ret.out_shapes.size(), 1);
  ASSERT_EQ(data2_ret.out_shapes.size(), 1);

  auto add_out_shape = test_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  auto assemble_platform_infos = AppendCoreTypeToPlatform(test_node, data_input.global_data);
  // use data2_out_shape as output_shapes of add_node
  auto tiling_ret = Tiling(test_node, {data1_ret.out_shapes[0], data2_ret.out_shapes[0]}, {data2_ret.out_shapes[0]},
                           {assemble_platform_infos[static_cast<size_t>(bg::AssemblePlatformInfoIndex::kPlatformInfo)],
                            global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(tiling_ret.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
  EXPECT_EQ(tiling_ret[0]->GetFastNode()->GetType(), "CompatibleTiling");

  // check topo
  CompatibleTopoCorrect(exe_graph, tiling_ret,
                        {data1_ret.out_shapes[0], data2_ret.out_shapes[0], data2_ret.out_shapes[0]});

  // check data
  auto find_tiling_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "FindCompatibleTilingFunc");
  ASSERT_NE(find_tiling_node, nullptr);
  auto exe_node_type = FastNodeTopoChecker(find_tiling_node).InChecker().DataFromByType("Const").GetFastNode();
  ge::Buffer buffer;
  ASSERT_TRUE(ExeGraphComparer::GetAttr(exe_node_type, buffer));
  ASSERT_NE(buffer.GetData(), nullptr);
  EXPECT_STREQ(reinterpret_cast<char *>(buffer.GetData()), node_type.c_str());
}

TEST_F(BgCacheableTilingUT, BgFallibleTiling_OutputsCountCorrect_CompatibleV4) {
  std::string node_type = "bg_node_with_tiling_v4";
  auto graph = BuildTwoInputsGraph(node_type);
  auto test_node = graph->FindNode(node_type);
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_JSON, "{ key : value}");
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_KEY, "{}");
  // mock tiling func v4
  optiling::OpTilingFuncRegistry(test_node->GetType(), StubOpTilingFuncV4, StubOpParseFuncV4);

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  LowerInput data_input = {{}, {}, &global_data};
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);

  ASSERT_EQ(data1_ret.out_shapes.size(), 1);
  ASSERT_EQ(data2_ret.out_shapes.size(), 1);

  auto add_out_shape = test_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  auto assemble_platform_infos = AppendCoreTypeToPlatform(test_node, data_input.global_data);
  // use data2_out_shape as output_shapes of add_node
  auto tiling_ret =
      FallibleTiling(test_node, {data1_ret.out_shapes[0], data2_ret.out_shapes[0]}, {data2_ret.out_shapes[0]},
                     {assemble_platform_infos[static_cast<size_t>(bg::AssemblePlatformInfoIndex::kPlatformInfo)],
                      *(data_input.global_data), fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);

  ASSERT_EQ(tiling_ret.size(), static_cast<size_t>(kernel::FallibleTilingExOutputIndex::kFallibleOutputNum));
  EXPECT_EQ(tiling_ret[0]->GetFastNode()->GetType(), "FallibleCompatibleTiling");

  // check topo
  CompatibleTopoCorrect(exe_graph, tiling_ret,
                        {data1_ret.out_shapes[0], data2_ret.out_shapes[0], data2_ret.out_shapes[0]});
}

TEST_F(BgCacheableTilingUT, BgTilingLegacy_Ok_CoreNum) {
  std::string node_type = "bg_node_with_tiling_v4";
  auto graph = BuildTwoInputsGraph(node_type);
  auto test_node = graph->FindNode(node_type);
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_JSON, "{ key : value}");
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_KEY, "{}");
  // mock tiling func v4
  optiling::OpTilingFuncRegistry(test_node->GetType(), StubOpTilingFuncV4, StubOpParseFuncV4);

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);

  ASSERT_EQ(data1_ret.out_shapes.size(), 1);
  ASSERT_EQ(data2_ret.out_shapes.size(), 1);
  dlog_setlevel(GE_MODULE_NAME, 0, 0);

  auto add_out_shape = test_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  AttrUtils::SetStr(test_node->GetOpDesc(), "_op_aicore_num", "5");
  AttrUtils::SetStr(test_node->GetOpDesc(), "_op_vectorcore_num", "10");
  auto assemble_platform_infos = AppendCoreTypeToPlatform(test_node, data_input.global_data);
  // 1. tiling input node is null
  auto tiling_ret = TilingLegacy(
      nullptr, {data1_ret.out_shapes[0], data2_ret.out_shapes[0]}, {data2_ret.out_shapes[0]},
      assemble_platform_infos[static_cast<size_t>(bg::AssemblePlatformInfoIndex::kPlatformInfo)], global_data);
  ASSERT_EQ(tiling_ret.size(), TilingContext::kOutputNum);
  EXPECT_EQ(tiling_ret[0], nullptr);
  // 2. build tiling successfully
  tiling_ret = TilingLegacy(test_node, {data1_ret.out_shapes[0], data2_ret.out_shapes[0]}, {data2_ret.out_shapes[0]},
                            assemble_platform_infos[static_cast<size_t>(bg::AssemblePlatformInfoIndex::kPlatformInfo)],
                            global_data);
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(tiling_ret.size(), TilingContext::kOutputNum);
  EXPECT_EQ(tiling_ret[0]->GetFastNode()->GetType(), "CompatibleTilingLegacy");

  // check data
  auto find_tiling_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "FindCompatibleTilingFunc");
  ASSERT_NE(find_tiling_node, nullptr);
  auto exe_node_type = FastNodeTopoChecker(find_tiling_node).InChecker().DataFromByType("Const").GetFastNode();
  ge::Buffer buffer;
  ASSERT_TRUE(ExeGraphComparer::GetAttr(exe_node_type, buffer));
  ASSERT_NE(buffer.GetData(), nullptr);
  EXPECT_STREQ(reinterpret_cast<char *>(buffer.GetData()), node_type.c_str());
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
}

TEST_F(BgCacheableTilingUT, BgTilingLegacy_Ok_CoreNumInvalid) {
  std::string node_type = "bg_node_with_tiling_v4";
  auto graph = BuildTwoInputsGraph(node_type);
  auto test_node = graph->FindNode(node_type);
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_JSON, "{ key : value}");
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_KEY, "{}");
  // mock tiling func v4
  optiling::OpTilingFuncRegistry(test_node->GetType(), StubOpTilingFuncV4, StubOpParseFuncV4);

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);

  ASSERT_EQ(data1_ret.out_shapes.size(), 1);
  ASSERT_EQ(data2_ret.out_shapes.size(), 1);
  dlog_setlevel(GE_MODULE_NAME, 0, 0);

  auto add_out_shape = test_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  AttrUtils::SetStr(test_node->GetOpDesc(), "_op_aicore_num", "5");
  AttrUtils::SetStr(test_node->GetOpDesc(), "_op_vectorcore_num", "aa");
  auto assemble_platform_infos = AppendCoreTypeToPlatform(test_node, data_input.global_data);
  ASSERT_EQ(assemble_platform_infos.size(), 0);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
}

TEST_F(BgCacheableTilingUT, AppendCoreTypeToPlatform_NullGlobalData) {
  auto graph = BuildTwoInputsGraph("Add");
  auto test_node = graph->FindNode("Add");
  ASSERT_NE(test_node, nullptr);
  auto assemble_platform_infos = AppendCoreTypeToPlatform(test_node, nullptr);
  ASSERT_EQ(assemble_platform_infos.size(), 0);
}

TEST_F(BgCacheableTilingUT, AppendCoreTypeToPlatform_InvalidAiCoreNum) {
  std::string node_type = "bg_node_invalid_aicore";
  auto graph = BuildTwoInputsGraph(node_type);
  auto test_node = graph->FindNode(node_type);
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_JSON, "{ key : value}");
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_KEY, "{}");
  optiling::OpTilingFuncRegistry(test_node->GetType(), StubOpTilingFuncV4, StubOpParseFuncV4);

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  AttrUtils::SetStr(test_node->GetOpDesc(), "_op_aicore_num", "invalid");
  auto assemble_platform_infos = AppendCoreTypeToPlatform(test_node, data_input.global_data);
  ASSERT_EQ(assemble_platform_infos.size(), 0);
}

TEST_F(BgCacheableTilingUT, BgTiling_TopoCorrect) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node, false);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  auto tiling_rets = Tiling(add_node, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph, "TilingUT");

  // check main 图中节点数量
  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 6},
                                                                  {"CacheableTiling", 1},
                                                                  {"TilingAppendDfxInfo", 1},
                                                                  {"TilingParse", 1},
                                                                  {"InnerData", 3},
                                                                  {"SplitRtStreams", 1},
                                                                  {"Const", 12},
                                                                  {"CalcTensorSizeFromStorage", 1},
                                                                  {"PrepareCacheableTilingFwkData", 1},
                                                                  {"AllocLaunchArg", 1}}),
            "success");
  ge::DumpGraph(init_frame_->GetExecuteGraph().get(), "TilingUTInit");

  // check init 图中节点数量
  ASSERT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictAllNodeTypes({
                    {"InnerNetOutput", 1},
                    {"FindTilingFunc", 1},
                    {"ConstData", 3},
                    {"Data", 1},
                    {"Const", 6},
                    {"SplitRtStreams", 1},
                    {"GetSpaceRegistry", 1},
                }),
            "success");
  // check topo连接关系，包括子图内部以及子图之间
  TilingTopoCorrect(exe_graph, tiling_rets, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
}

TEST_F(BgCacheableTilingUT, BgTiling_TopoCorrect_FallibleTiling_MemCheck) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node, false);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  auto op_desc = add_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetWorkspaceBytes({4, 8});
  AttrUtils::SetStr(op_desc, ATTR_NAME_ALIAS_ENGINE_NAME, "mix_l2");
  ge::AttrUtils::SetBool(op_desc, "_memcheck", true);
  auto tiling_rets =
      FallibleTiling(add_node, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  DumpGraph(exe_graph, "TilingUT");

  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 6},
                                                                  {"TilingParse", 1},
                                                                  {"InnerData", 3},
                                                                  {"SplitRtStreams", 1},
                                                                  {"Const", 12},
                                                                  {"CacheableFallibleTiling", 1},
                                                                  {"TilingAppendDfxInfo", 1},
                                                                  {"CalcTensorSizeFromStorage", 1},
                                                                  {"PrepareCacheableTilingFwkData", 1},
                                                                  {"AllocLaunchArg", 1}}),
            "success");
  TilingTopoCorrect(exe_graph, tiling_rets, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::FallibleTilingExOutputIndex::kFallibleOutputNum));
}

TEST_F(BgCacheableTilingUT, BgTiling_TopoCorrect_FallibleTiling_MemCheck2) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node, false);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  auto op_desc = add_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetWorkspaceBytes({4, 8});
  AttrUtils::SetStr(op_desc, ATTR_NAME_ALIAS_ENGINE_NAME, "mix_l2");
  ge::AttrUtils::SetBool(op_desc, "_memcheck", true);
  std::map<std::string, std::string> options;
  options.insert(std::pair<std::string, std::string>(ge::TILING_SCHEDULE_OPTIMIZE, "1"));
  ge::GetThreadLocalContext().SetGlobalOption(options);
  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
  RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_DEV_CAP_SUPPORT);

  auto tiling_rets =
      FallibleTiling(add_node, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  DumpGraph(exe_graph, "TilingUT");

  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 6},
                                                                  {"TilingParse", 1},
                                                                  {"InnerData", 3},
                                                                  {"SplitRtStreams", 1},
                                                                  {"Const", 12},
                                                                  {"CacheableFallibleTiling", 1},
                                                                  {"TilingAppendDfxInfo", 1},
                                                                  {"CalcTensorSizeFromStorage", 1},
                                                                  {"PrepareCacheableTilingFwkData", 1},
                                                                  {"AllocLaunchArg", 1}}),
            "success");
  TilingTopoCorrect(exe_graph, tiling_rets, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::FallibleTilingExOutputIndex::kFallibleOutputNum));
  options.clear();
  ge::GetThreadLocalContext().SetGlobalOption(options);
}

TEST_F(BgCacheableTilingUT, BgTiling_TopoCorrect_FallibleTiling_MemCheck3) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node, false);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  auto op_desc = add_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetWorkspaceBytes({4, 8});
  AttrUtils::SetStr(op_desc, ATTR_NAME_ALIAS_ENGINE_NAME, "mix_l2");
  ge::AttrUtils::SetBool(op_desc, "_memcheck", true);
  std::map<std::string, std::string> options;
  options.insert(std::pair<std::string, std::string>(ge::TILING_SCHEDULE_OPTIMIZE, "1"));
  ge::GetThreadLocalContext().SetGlobalOption(options);
  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
  RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_NOT_SUPPORT);

  auto tiling_rets =
      FallibleTiling(add_node, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  DumpGraph(exe_graph, "TilingUT");

  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 6},
                                                                  {"TilingParse", 1},
                                                                  {"InnerData", 3},
                                                                  {"SplitRtStreams", 1},
                                                                  {"Const", 12},
                                                                  {"CacheableFallibleTiling", 1},
                                                                  {"TilingAppendDfxInfo", 1},
                                                                  {"CalcTensorSizeFromStorage", 1},
                                                                  {"PrepareCacheableTilingFwkData", 1},
                                                                  {"AllocLaunchArg", 1}}),
            "success");
  TilingTopoCorrect(exe_graph, tiling_rets, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::FallibleTilingExOutputIndex::kFallibleOutputNum));
  options.clear();
  ge::GetThreadLocalContext().SetGlobalOption(options);
}

TEST_F(BgCacheableTilingUT, BgFallibleTiling_OutputsCountCorrect) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node, false);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  auto tiling_rets =
      FallibleTiling(add_node, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph, "TilingUT");

  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 6},
                                                                  {"CacheableFallibleTiling", 1},
                                                                  {"TilingParse", 1},
                                                                  {"InnerData", 3},
                                                                  {"SplitRtStreams", 1},
                                                                  {"CalcTensorSizeFromStorage", 1},
                                                                  {"TilingAppendDfxInfo", 1},
                                                                  {"Const", 12},
                                                                  {"PrepareCacheableTilingFwkData", 1},
                                                                  {"AllocLaunchArg", 1}}),
            "success");
  TilingTopoCorrect(exe_graph, tiling_rets, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::FallibleTilingExOutputIndex::kFallibleOutputNum));
}
}  // namespace bg
}  // namespace gert

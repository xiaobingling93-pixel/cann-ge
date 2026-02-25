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
#define private public
#define protected public
#include "slice_result_mocker.h"
#include "es_ge_test_ops.h"
#include "common_setup.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_symbolizer.h"
#include "dflow/compiler/model/flow_model_cache.h"
#include "api/session/jit_execution/utils/guarded_execution_point_util.h"
#include "graph/optimize/symbolic/shape_env_guarder.h"
#include "framework/common/helper/model_save_helper.h"
#include "dflow/inc/data_flow/model/flow_model_helper.h"
#include "graph/utils/graph_utils_ex.h"
#undef private
#undef protected
using namespace ge;

namespace ge {
constexpr uint32_t NUM_GUARD_PATTERNS = 3;
constexpr const char *kGuardCheckSoDataResult = "_guard_check_so_data";

void MakeAttrPattern0(ShapeEnvAttr &attr) {
  auto symbol0 = attr.CreateSymbol(3, MakeShared<InputShapeSource>(0, 0));
  auto symbol1 = attr.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto symbol2 = attr.CreateSymbol(5, MakeShared<InputShapeSource>(0, 2));

  auto symbol4 = attr.CreateSymbol(3, MakeShared<InputShapeSource>(1, 0));
  auto symbol5 = attr.CreateSymbol(4, MakeShared<InputShapeSource>(1, 1));
  auto symbol6 = attr.CreateSymbol(5, MakeShared<InputShapeSource>(1, 2));
  EXPECT_SYMBOL_EQ(symbol0, symbol4);
  EXPECT_SYMBOL_EQ(symbol1, symbol5);
  EXPECT_SYMBOL_EQ(symbol2, symbol6);
}

void MakeAttrPattern1(ShapeEnvAttr &attr) {
  auto symbol0 = attr.CreateSymbol(3, MakeShared<InputShapeSource>(0, 0));
  auto symbol1 = attr.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto symbol2 = attr.CreateSymbol(5, MakeShared<InputShapeSource>(0, 2));
  auto symbol3 = attr.CreateSymbol(7, MakeShared<InputShapeSource>(1, 0));
  auto symbol4 = attr.CreateSymbol(3, MakeShared<InputShapeSource>(1, 1));
  auto symbol5 = attr.CreateSymbol(4, MakeShared<InputShapeSource>(1, 2));
  auto symbol6 = attr.CreateSymbol(5, MakeShared<InputShapeSource>(1, 3));
  EXPECT_SYMBOL_EQ(symbol0, symbol4);
  EXPECT_SYMBOL_NE(symbol1, symbol2);
  EXPECT_SYMBOL_LE(sym::Max(symbol0, symbol5), sym::Min(symbol6, Symbol(8)));
  EXPECT_SYMBOL_LT(sym::Pow(symbol1, sym::Rational(1, 2)), symbol3);
  EXPECT_SYMBOL_GE(sym::Pow(symbol4 + symbol5, Symbol(2)), sym::Ceiling(symbol3));
  EXPECT_SYMBOL_GT(sym::Abs(symbol0 - symbol6), sym::Log(Symbol(1)));
}

void MakeAttrPattern2(ShapeEnvAttr &attr) {
  auto symbol0 = attr.CreateSymbol(12, MakeShared<InputValueSumSource>(0, DT_INT32));
  auto symbol1 = attr.CreateSymbol(15, MakeShared<InputValueSumSource>(1, DT_INT32));
  auto symbol2 = attr.CreateSymbol(18, MakeShared<InputValueSumSource>(2, DT_INT32));

  auto symbol3 = attr.CreateSymbol(7, MakeShared<InputShapeSource>(3, 0));
  auto symbol4 = attr.CreateSymbol(12, MakeShared<InputShapeSource>(3, 1));
  auto symbol5 = attr.CreateSymbol(4, MakeShared<InputShapeSource>(3, 2));
  auto symbol6 = attr.CreateSymbol(20, MakeShared<InputShapeSource>(3, 3));
  EXPECT_SYMBOL_EQ(symbol0, symbol4);
  EXPECT_SYMBOL_NE(symbol1, symbol2);
  EXPECT_SYMBOL_LE(sym::Min(symbol6, Symbol(8)), sym::Max(symbol0, symbol1));
  EXPECT_SYMBOL_LT(sym::Pow(symbol1, sym::Rational(1, 2)), symbol3);
  EXPECT_SYMBOL_GE(sym::Pow(symbol4 + symbol5, Symbol(2)), sym::Ceiling(symbol3));
  EXPECT_SYMBOL_GT(sym::Abs(symbol0 - symbol6), sym::Log(Symbol(1)));
}

GeRootModelPtr BuildGeRootModel(const string &name, const ComputeGraphPtr &graph) {
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  auto ge_model = MakeShared<ge::GeModel>();
  auto model_task_def = MakeShared<domi::ModelTaskDef>();
  model_task_def->set_version("test_v100_r001");
  ge_model->SetModelTaskDef(model_task_def);
  ge_model->SetName(name);
  ge_model->SetGraph(graph);
  ge_root_model->SetModelName(name);
  ge_root_model->SetSubgraphInstanceNameToModel(name, ge_model);
  return ge_root_model;
}

void CheckGuardFunc(const ComputeGraphPtr &gt_graph, const ComputeGraphPtr &test_graph) {
  std::string gt_buffer, test_buffer;
  /* get the GuardFunc binary byte string of ground-truth and test compute_graph */
  EXPECT_EQ(ge::AttrUtils::GetStr(gt_graph, kGuardCheckSoDataResult, gt_buffer), true);
  EXPECT_EQ(ge::AttrUtils::GetStr(test_graph, kGuardCheckSoDataResult, test_buffer), true);
  EXPECT_EQ(test_buffer.compare(gt_buffer), 0);
}
}

std::vector<ComputeGraphPtr> SliceResultMocker::gt_graphs_with_pattern_;

std::unordered_map<std::string, uint32_t> SliceResultMocker::gep_graph_key_to_pattern_map_;

int64_t SliceResultMocker::instance_id_ = 0;

SliceResultMocker::SliceResultMocker(const std::string &user_graph_key, uint32_t num_eps,
  const std::vector<uint32_t> &num_geps)
    : user_graph_key_(user_graph_key),
      num_eps_(num_eps), num_geps_(num_geps) {
  EXPECT_EQ(num_eps, num_geps.size());
}

void SliceResultMocker::InitGtGraph() {
  /* generate the ground-truth compute graphs with guards */
  gt_graphs_with_pattern_.clear();
  gt_graphs_with_pattern_.emplace_back(GenGraphWithGuard("gt_graph_0", MakeAttrPattern0));
  gt_graphs_with_pattern_.emplace_back(GenGraphWithGuard("gt_graph_1", MakeAttrPattern1));
  gt_graphs_with_pattern_.emplace_back(GenGraphWithGuard("gt_graph_2", MakeAttrPattern2));
}

ComputeGraphPtr SliceResultMocker::GenGraphWithGuard(const string &graph_name,
                                                     std::function<void(ShapeEnvAttr &attr)> func) {
  GuardCodegen codegen;
  auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(
    EsCreateGraphBuilder(graph_name.c_str()), EsDestroyGraphBuilder);
  auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*ge_graph);
  auto attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  EXPECT_NE(attr, nullptr);
  ShapeEnvGuarder guard(attr);
  func(*attr);

  // generate guard func
  EXPECT_EQ(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);
  return compute_graph;
}

ComputeGraphPtr SliceResultMocker::GenGraph(const string &graph_name) {
  auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(
    EsCreateGraphBuilder(graph_name.c_str()), EsDestroyGraphBuilder);
  auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
  return GraphUtilsEx::GetComputeGraph(*ge_graph);
}

GuardedExecutionPoint *SliceResultMocker::GenGEP(ExecutionPoint &ep, CompiledModelCache &cmc,
	                                         const std::string &cache_dir, std::string user_graph_key) {
  auto *gep = new GuardedExecutionPoint(&ep);
  uint32_t pattern = rand() % NUM_GUARD_PATTERNS; // 0-2

  const ComputeGraphPtr compiled_graph = gt_graphs_with_pattern_[pattern];
  gep->compiled_graph_ = compiled_graph;
  gep->SetCompiled(instance_id_++, compiled_graph);
  std::map<std::string, std::string> options;
  cmc.CreateKeyOptionForGuardedExecutionPoint(gep, options);

  /* save the corresponding .om file */
  std::string gep_graph_key;
  EXPECT_EQ(cmc.GetGuardedExecutionPointGraphKey(gep, gep_graph_key), SUCCESS);
  gep_graph_key_to_pattern_map_.emplace(gep_graph_key, pattern);;
  GenOmFile(cache_dir, gep_graph_key, compiled_graph);
  return gep;
}

void SliceResultMocker::GenOmFile(const std::string &cache_dir,
  const std::string &graph_key, const ComputeGraphPtr &graph) {
  auto old_session_options = GetThreadLocalContext().GetAllSessionOptions();
  auto old_graph_options = GetThreadLocalContext().GetAllGraphOptions();

  GetThreadLocalContext().SetSessionOption({{"ge.graph_compiler_cache_dir", cache_dir + "/jit/"}});
  GetThreadLocalContext().SetGraphOption({{"ge.graph_key", graph_key}});
  GeRootModelPtr ge_root_model = BuildGeRootModel(graph->GetName(), graph);
  ModelData model_data{};
  ModelBufferData model_buffer_data;
  bool is_unknown_shape = false;
  EXPECT_EQ(ge_root_model->CheckIsUnknownShape(is_unknown_shape), SUCCESS);
  const auto model_save_helper =
      ModelSaveHelperFactory::Instance().Create(OfflineModelFormat::OM_FORMAT_DEFAULT);
  EXPECT_NE(model_save_helper, nullptr);
  model_save_helper->SetSaveMode(false);
  EXPECT_EQ(model_save_helper->SaveToOmRootModel(ge_root_model, "NoUse", model_buffer_data, is_unknown_shape), SUCCESS);
  model_data.model_data = model_buffer_data.data.get();
	model_data.model_len = model_buffer_data.length;
  FlowModelPtr flow_model = MakeShared<ge::FlowModel>(graph);
  EXPECT_EQ(flow_model->AddSubModel(FlowModelHelper::ToPneModel(model_data, graph), PNE_ID_NPU), SUCCESS);

  {
    FlowModelCache flow_model_cache;
    EXPECT_EQ(flow_model_cache.Init(graph), SUCCESS);
    EXPECT_EQ(flow_model_cache.TryCacheFlowModel(flow_model), SUCCESS);
  }

  /* restore the options */
  GetThreadLocalContext().SetSessionOption(old_session_options);
  GetThreadLocalContext().SetGraphOption(old_graph_options);
}

std::unique_ptr<ExecutionPoint> SliceResultMocker::GenExecutionPoint(int64_t ep_idx, const uint32_t num_geps,
  CompiledModelCache &cmc, const std::string &cache_dir, std::string user_graph_key, const bool hasRemGraph) {
  ComputeGraphPtr slice_graph = GenGraph("slice_graph");
  ComputeGraphPtr rem_graph = nullptr;
  if (hasRemGraph) { // last ep's remaining graph should be nullptr
    rem_graph = GenGraph("rem_graph");
  }
  auto ep =  MakeUnique<ExecutionPoint>(ep_idx, slice_graph, rem_graph);
  for (uint32_t gep_idx = 0; gep_idx < num_geps; ++gep_idx) {
    auto gep = GenGEP(*ep, cmc, cache_dir, user_graph_key);
    ep->models_.cache_models_.emplace_back(gep);
  }
  return ep;
}

void SliceResultMocker::GenExecutionOrder(ExecutionOrder &order, CompiledModelCache &cmc,
                                          const std::string &cache_dir, std::string user_graph_key) const {
   for (int64_t ep_idx = 0; ep_idx < num_eps_; ++ep_idx) {
    const bool hasRemGraph = (ep_idx != num_eps_ - 1);
    auto ep = GenExecutionPoint(ep_idx, num_geps_[ep_idx], cmc, cache_dir, user_graph_key, hasRemGraph);
    order.slice_graphs_.emplace_back(std::move(ep));
  }
}

void SliceResultMocker::GenSlicingResultFiles(const std::string &cache_dir, std::string user_graph_key, CompiledModelCache &cmc) {
  map<std::string, std::string> global_options = {
    {"ge.graph_compiler_cache_dir", cache_dir},
    {"ge.graph_key", user_graph_key_}
  };
  ExecutionOrder order({12345u, GenGraph("user_graph")});
  GenExecutionOrder(order, cmc, cache_dir, user_graph_key); // user cmc saver to generate files
  EXPECT_EQ(cmc.SaveCache(order), SUCCESS);
}

void SliceResultMocker::CheckFileGenResult(const ExecutionOrder &order, const std::string &user_graph_key,
  const std::string &cache_dir) {
  const std::string user_graph_base_dir = cache_dir + "/jit/slicing_hierarchy/" + user_graph_key + "/";
  const std::string slice_res_path = user_graph_base_dir + "slicing_result.json";
  EXPECT_EQ(FlowModelCache::CheckFileExist(slice_res_path), true);      // check slicing_result.json

  const auto num_eps = order.slice_graphs_.size();
  for (uint32_t ep_idx = 0; ep_idx < num_eps; ++ep_idx) {
    const std::string slice_graph_bas_dir = user_graph_base_dir + std::to_string(ep_idx) + "/";
    const std::string gep_list_path = slice_graph_bas_dir + "gep_list.json";
    const std::string slice_graph_pb_path = slice_graph_bas_dir + "slice_graph.pb";
    const std::string rem_graph_pb_path = slice_graph_bas_dir + "rem_graph.pb";
    EXPECT_EQ(FlowModelCache::CheckFileExist(gep_list_path), true);         // check gep_list.json
    EXPECT_EQ(FlowModelCache::CheckFileExist(slice_graph_pb_path), true);   // check slice_graph.pb
    if (ep_idx != num_eps - 1) {
      EXPECT_EQ(FlowModelCache::CheckFileExist(rem_graph_pb_path), true);   // check rem_graph.pb
    }
  }
}

void SliceResultMocker::CheckMemObjResult(const ExecutionOrder &order, CompiledModelCache &cmc) {
  for (auto &ep: order.slice_graphs_) {
    for (auto &gep: ep->models_.GetCache()) {
      std::string gep_graph_key;
      EXPECT_EQ(cmc.GetGuardedExecutionPointGraphKey(gep.get(), gep_graph_key), SUCCESS);
      auto iter = gep_graph_key_to_pattern_map_.find(gep_graph_key);
      EXPECT_NE(iter, gep_graph_key_to_pattern_map_.end());

      /* check matcher */
      EXPECT_NE(gep->matcher_.func_, nullptr);

      /* check guard func correctness */
      const auto pattern = iter->second;
      CheckGuardFunc(gt_graphs_with_pattern_[pattern], gep->GetGraph());
    }
  }
}
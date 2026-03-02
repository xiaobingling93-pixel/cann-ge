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
#include "jit_execution/cache/compiled_model_cache.h"
#include "jit_execution/utils/guarded_execution_point_util.h"
#include "es_ge_test_ops.h"
#include "ge_running_env/dir_env.h"
#include "stub/gert_runtime_stub.h"
#include "common_setup.h"
#include "mmpa/mmpa_api.h"
#include "graph_utils_ex.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_symbolizer.h"
#include "dflow/compiler/model/flow_model_cache.h"
#include "slice_result_mocker.h"
#include "api/gelib/gelib.h"
#include "common/mem_conflict_share_graph.h"
#include "graph/execute/model_executor.h"
#undef private
#undef protected

namespace ge {
Status GenDirectory(const std::string &dir_path) {
  if (mmAccess(dir_path.c_str()) != EN_OK) { // construct the root dir for the user_graph
    system(("mkdir -p " + dir_path).c_str());
  }
  return SUCCESS;
}

void DeleteDirectory(const std::string &dir_path) {
  if (mmAccess(dir_path.c_str()) == EN_OK) { // construct the root dir for the user_graph
    system(("rm -rf " + dir_path).c_str());
  }
}

class CompiledModelCacheUT : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    DirEnv::GetInstance().InitDir();
  }

  void SetUp() override {
    env = getenv("LD_PRELOAD");
    unsetenv("LD_PRELOAD");
    CommonSetupUtil::CommonSetup();
    GetThreadLocalContext().SetSessionOption({});
    GetThreadLocalContext().SetGraphOption({});

    SliceResultMocker::InitGtGraph();
    DeleteDirectory(cache_dir_);  // clean old results
    GenDirectory(cache_dir_);
  }

  void TearDown() override {
    CommonSetupUtil::CommonTearDown();
    DNNEngineManager::GetInstance().plugin_mgr_.ClearHandles_();
    if (env != nullptr) {
      setenv("LD_PRELOAD", env, 1);
    }
  }

  std::string cache_dir_ = "./build_cache_dir";

  std::string user_graph_key_ = "user_graph_key";

  uint32_t user_graph_id_ = 1234u;

  ComputeGraphPtr user_graph_ = SliceResultMocker::GenGraph("user_graph");

  map<std::string, std::string> global_options_ = {{"ge.graph_compiler_cache_dir", cache_dir_},
                                               {"ge.graph_key", user_graph_key_}};
  map<std::string, std::string> graph_options_ = {{"ge.graph_key", user_graph_key_}};

  const char *env;
};

TEST_F(CompiledModelCacheUT, check_add_gep_graph_key) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  CompileContext context(graph_manager);

  CompiledModelCache cmc(user_graph_id_, context, graph_manager);
  const auto ep = new ExecutionPoint(1, nullptr, nullptr);
  const auto gep0 = new GuardedExecutionPoint(ep);
  const auto *gep1 = new GuardedExecutionPoint(ep);
  GuardedExecutionPointUtil gep_util;
  EXPECT_EQ(gep_util.AddGuardedExecutionPointGraphKey(cache_dir_, user_graph_key_, gep0), ge::SUCCESS);
  EXPECT_EQ(gep_util.AddGuardedExecutionPointGraphKey(cache_dir_, user_graph_key_, gep1), ge::SUCCESS);

  EXPECT_EQ(graph_manager.Finalize(), ge::SUCCESS);
  delete ep;
  delete gep0;
  delete gep1;
}

TEST_F(CompiledModelCacheUT, check_get_gep_graph_key) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  CompileContext context(graph_manager);

  int64_t slice_graph_id = 1;
  std::string gep_graph_key_prefix_gt = user_graph_key_ + "_" + std::to_string(slice_graph_id) + "_";
  auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("Hello"), EsDestroyGraphBuilder);
  const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));

  EXPECT_EQ(graph_manager.AddGraph(user_graph_id_,
    *ge_graph, {{"ge.graph_key", user_graph_key_},
    {"ge.graph_compiler_cache_dir", cache_dir_}}, OmgContext()), ge::SUCCESS);

  CompiledModelCache cmc(user_graph_id_, context, graph_manager);

  auto ep = new ExecutionPoint(slice_graph_id, nullptr, nullptr);
  auto gep0 = new GuardedExecutionPoint(ep);
  GuardedExecutionPointUtil gep_util;
  EXPECT_EQ(gep_util.AddGuardedExecutionPointGraphKey(cache_dir_, user_graph_key_, gep0), ge::SUCCESS);

  std::string gep_graph_key;
  EXPECT_EQ(gep_util.GetGuardedExecutionPointGraphKey(gep0, gep_graph_key), ge::SUCCESS);
  EXPECT_EQ(gep_graph_key.rfind(gep_graph_key_prefix_gt, 0), 0);
  EXPECT_EQ(graph_manager.Finalize(), ge::SUCCESS);

  delete ep;
  delete gep0;
}

TEST_F(CompiledModelCacheUT, check_emplace_gep_option) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  CompileContext context(graph_manager);

  constexpr int64_t slice_graph_id = 1;
  const std::string gep_graph_key_prefix_gt = user_graph_key_ + "_" + std::to_string(slice_graph_id) + "_";
  const auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("Hello"), EsDestroyGraphBuilder);
  const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));

  EXPECT_EQ(graph_manager.AddGraph(user_graph_id_, *ge_graph,
    {{"ge.graph_key", user_graph_key_},
    {"ge.graph_compiler_cache_dir", cache_dir_}}, OmgContext()), ge::SUCCESS);

  CompiledModelCache cmc(user_graph_id_, context, graph_manager);

  ExecutionPoint *ep = new ExecutionPoint(slice_graph_id, nullptr, nullptr);
  GuardedExecutionPoint *gep0 = new GuardedExecutionPoint(ep);
  GuardedExecutionPointUtil gep_util;
  EXPECT_EQ(gep_util.AddGuardedExecutionPointGraphKey(cache_dir_, user_graph_key_, gep0), ge::SUCCESS);

  std::map<std::string, std::string> options;
  EXPECT_EQ(gep_util.EmplaceGuardedExecutionPointOption(cache_dir_, user_graph_key_, gep0, options), ge::SUCCESS);
  EXPECT_NE(options.find("ge.graph_key"), options.end());
  EXPECT_NE(options.find("ge.graph_compiler_cache_dir"), options.end());

  std::string gep_graph_key = options["ge.graph_key"];
  EXPECT_EQ(gep_graph_key.rfind(gep_graph_key_prefix_gt, 0), 0);

  delete ep;
  delete gep0;
  EXPECT_EQ(graph_manager.Finalize(), ge::SUCCESS);
}

/*
 * Construct a UT with 2 EPs, each with 2 GEPs and their corresponding files, to see whether
 * the EO/EP/GEPs can be correctly loaded
 */
TEST_F(CompiledModelCacheUT, check_restore_cache) {
  /* initialize the ids, graph_keys and dirs */
  /* generate the mock user_graph */
  auto user_graph_ptr = JitShareGraph::AllNormalNodes();
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(*user_graph_ptr);

  if (compute_graph == nullptr) {
    printf("compute_graph == nullptr 1 \n");
  }
  OptionSetter option_setter(global_options_);
  /* init the CMC and start the test */
  ModelExecutor model_executor;
  model_executor.Initialize(global_options_, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize(global_options_, &model_executor), SUCCESS);
  CompileContext context(graph_manager);
  CompiledModelCache cmc(user_graph_id_, context, graph_manager);

  /* construct the cache data structure for testing */
  SliceResultMocker mocker(user_graph_key_, 2, {3, 4});
  mocker.GenSlicingResultFiles(cache_dir_, user_graph_key_, cmc);

  ExecutionOrder order(UserGraph{user_graph_id_, compute_graph});
  mocker.GenExecutionOrder(order, cmc, cache_dir_, user_graph_key_);
  EXPECT_EQ(cmc.SaveCache(order), ge::SUCCESS);

  ExecutionOrder order_restore(UserGraph{user_graph_id_, compute_graph});
  CompiledModelCache cmc_restore(user_graph_id_, context, graph_manager);
  EXPECT_EQ(cmc_restore.RestoreCache(order_restore), ge::SUCCESS);

  /* check whether the GuardFunc is correctly loaded */
  mocker.CheckMemObjResult(order_restore, cmc_restore);

  EXPECT_EQ(graph_manager.Finalize(), ge::SUCCESS);
}

TEST_F(CompiledModelCacheUT, check_restore_execution_order_dir_not_exist) {
  uint32_t user_graph_id = user_graph_id_ + 1;
  ExecutionOrder order(UserGraph{user_graph_id, nullptr});
  ExecutionOrderUtil eo_util;
  const string user_graph_key = "";
  EXPECT_EQ(eo_util.RestoreExecutionOrder(cache_dir_ + "/not_exist_dir", user_graph_key, order), ge::SUCCESS);
}

TEST_F(CompiledModelCacheUT, check_save_execution_point_no_ep) {
  uint32_t user_graph_id = user_graph_id_ + 2;
  const string user_graph_key = "";
  ExecutionPointUtil ep_util;
  std::unique_ptr<ExecutionPoint> exec_point_ptr = std::make_unique<ExecutionPoint>(user_graph_id, nullptr, nullptr);
  EXPECT_EQ(ep_util.SaveExecutionPoint(cache_dir_, user_graph_key, exec_point_ptr), ge::SUCCESS);
}

TEST_F(CompiledModelCacheUT, check_save_execution_point_no_update_ep) {
  uint32_t user_graph_id = user_graph_id_ + 2;
  const string user_graph_key = "";
  ExecutionPointUtil ep_util;
  ComputeGraphPtr sliced_graph = std::make_shared<ComputeGraph>("sliced_graph");
  std::unique_ptr<ExecutionPoint> exec_point_ptr = std::make_unique<ExecutionPoint>(user_graph_id, sliced_graph, nullptr);
  EXPECT_EQ(ep_util.SaveExecutionPoint(cache_dir_, user_graph_key, exec_point_ptr), ge::SUCCESS);
}
}
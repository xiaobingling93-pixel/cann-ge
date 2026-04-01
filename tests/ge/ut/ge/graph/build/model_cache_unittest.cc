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
#include <fstream>
#include "mmpa/mmpa_api.h"
#include "graph/build/model_cache.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/ge_local_context.h"
#include "graph/build/memory/var_mem_assign_util.h"
#include "common/model/ge_root_model.h"
#include "common/helper/model_parser_base.h"
#include "framework/common/types.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "framework/common/helper/om_file_helper.h"
#include "common/mem_conflict_share_graph.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "framework/common/helper/model_save_helper.h"
#include "helper/model_helper.h"

namespace ge {
namespace {
ComputeGraphPtr FakeComputeGraph(const string &graph_name) {
  DEF_GRAPH(graph1) {
    auto data_0 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("FakeOpNpu").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("_arg_0", data_0)->NODE("fused_op1", fake_type2_op1)->NODE("Node_Output", net_output));
  };

  auto root_graph = ToComputeGraph(graph1);
  root_graph->SetName(graph_name);
  root_graph->SetSessionID(0);
  AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0_1");

  auto op_desc = root_graph->FindNode("Node_Output")->GetOpDesc();
  std::vector<std::string> src_name{"out"};
  op_desc->SetSrcName(src_name);
  std::vector<int64_t> src_index{0};
  op_desc->SetSrcIndex(src_index);
  return root_graph;
}

ComputeGraphPtr FakeGraphWithSubGraph(const string &graph_name, ComputeGraphPtr sub_graph) {
  DEF_GRAPH(graph1) {
    auto data = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto partitioned_call_op =
        OP_CFG(PARTITIONEDCALL).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {-1}).Build("partitioned_call_op");

    partitioned_call_op->RegisterSubgraphIrName("f", SubgraphType::kStatic);
    partitioned_call_op->AddSubgraphName(sub_graph->GetName());
    partitioned_call_op->SetSubgraphInstanceName(0, sub_graph->GetName());

    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    CHAIN(NODE("data", data)->NODE(partitioned_call_op)->NODE("Node_Output", net_output));
  };

  auto root_graph = ToComputeGraph(graph1);
  root_graph->SetName(graph_name);
  root_graph->SetSessionID(0);
  root_graph->AddSubGraph(sub_graph);
  sub_graph->SetParentGraph(root_graph);
  AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0_1");
  return root_graph;
}

ComputeGraphPtr FakeComputeGraphWithConstant(const string &graph_name) {
  DEF_GRAPH(graph1) {
    auto constant_0 = OP_CFG(CONSTANTOP).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("FakeOpNpu").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE(graph_name + "_constant_0", constant_0)->NODE("fused_op1", fake_type2_op1)->NODE("Node_Output", net_output));
  };

  auto root_graph = ToComputeGraph(graph1);
  root_graph->SetName(graph_name);
  root_graph->SetSessionID(0);
  AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0_1");

  auto const_op_desc = root_graph->FindFirstNodeMatchType(CONSTANTOP)->GetOpDesc();
  GeTensor weight;
  std::vector<int32_t> data(16, 1);
  weight.SetData((uint8_t *)data.data(), data.size() * sizeof(int32_t));
  GeTensorDesc weight_desc;
  weight_desc.SetShape(GeShape({16}));
  weight_desc.SetOriginShape(GeShape({16}));
  weight.SetTensorDesc(weight_desc);
  AttrUtils::SetTensor(const_op_desc, "value", weight);

  auto op_desc = root_graph->FindNode("Node_Output")->GetOpDesc();
  std::vector<std::string> src_name{"out"};
  op_desc->SetSrcName(src_name);
  std::vector<int64_t> src_index{0};
  op_desc->SetSrcIndex(src_index);
  return root_graph;
}

ComputeGraphPtr FakeComputeGraphWithVar(const string &graph_name) {
  DEF_GRAPH(graph1) {
    auto var_0 = OP_CFG(VARIABLE).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("FakeOpNpu").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("1_ascend_mbatch_batch_0", var_0)->NODE("fused_op1", fake_type2_op1)->NODE("Node_Output", net_output));
  };

  auto root_graph = ToComputeGraph(graph1);
  root_graph->SetName(graph_name);
  root_graph->SetSessionID(0);
  AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0_1");

  auto op_desc = root_graph->FindNode("Node_Output")->GetOpDesc();
  std::vector<std::string> src_name{"out"};
  op_desc->SetSrcName(src_name);
  std::vector<int64_t> src_index{0};
  op_desc->SetSrcIndex(src_index);
  return root_graph;
}

GeRootModelPtr BuildGeRootModel(const string &name, ComputeGraphPtr graph) {
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

GeRootModelPtr BuildGeModel(const string &name, ComputeGraphPtr graph) {
  auto graph1 = FakeComputeGraph("graph1");
  return BuildGeRootModel("graph1", graph1);
}

GeRootModelPtr BuildGeModel2(const string &name, ComputeGraphPtr graph) {
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);
  auto graph1 = FakeComputeGraphWithConstant("graph1");
  EXPECT_EQ(VarMemAssignUtil::AssignConstantOpMemory(graph1), SUCCESS);
  const auto &constant_node = graph1->FindNode("graph1_constant_0");
  const auto &fake_node = graph1->FindNode("fused_op1");
  fake_node->GetOpDesc()->SetInputOffset(constant_node->GetOpDesc()->GetOutputOffset());
  fake_node->GetOpDesc()->SetOutputOffset(constant_node->GetOpDesc()->GetOutputOffset());
  auto ge_root_model1 = BuildGeRootModel("graph1", graph1);
  return ge_root_model1;
}

bool ReadIndexFile(const std::string &index_file, std::vector<ge::CacheFileIdx> &cache_file_list) {
  nlohmann::json json_obj;
  std::ifstream file_stream(index_file);
  if (!file_stream.is_open()) {
    std::cout << "Failed to open cache index file:" << index_file << std::endl;
    return false;
  }

  try {
    file_stream >> json_obj;
  } catch (const nlohmann::json::exception &e) {
    std::cout << "Failed to read cache index file:" << index_file << ", err msg:" << e.what() << std::endl;
    file_stream.close();
    return false;
  }

  try {
    cache_file_list = json_obj["cache_file_list"].get<std::vector<ge::CacheFileIdx>>();
  } catch (const nlohmann::json::exception &e) {
    std::cout << "Failed to read cache index file:" << index_file << ", err msg:" << e.what() << std::endl;
    file_stream.close();
    return false;
  }
  file_stream.close();
  return true;
}

bool CheckCacheResult(const std::string &cache_dir, const std::string &graph_key, size_t expect_cache_size, bool compare_key = true) {
  const auto cache_idx_file = cache_dir + "/" + graph_key + ".idx";
  auto check_ret = mmAccess(cache_idx_file.c_str());
  if (check_ret != 0) {
    std::cout << "Cache index file:" << cache_idx_file << " is not exist" << std::endl;
    return false;
  }

  std::vector<ge::CacheFileIdx> cache_file_list;
  if (!ReadIndexFile(cache_idx_file, cache_file_list)) {
    std::cout << "Faile to read cache index file:" << cache_idx_file << std::endl;
    return false;
  }
  for (auto &idx : cache_file_list) {
    idx.cache_file_name = cache_dir + "/" + idx.cache_file_name;
    if (!idx.var_desc_file_name.empty()) {
      idx.var_desc_file_name = cache_dir + "/" + idx.var_desc_file_name;
    }
  }
  if (cache_file_list.size() != expect_cache_size) {
    std::cout << "Cache file size[" << cache_file_list.size() << "] error, expect = " << expect_cache_size << std::endl;
    return false;
  }

  for (const auto &cache_index : cache_file_list) {
    if (compare_key && cache_index.graph_key != graph_key) {
      std::cout << "Cache graph_key[" << cache_index.graph_key << "] error, expect = " << graph_key << std::endl;
      return false;
    }

    if (cache_index.cache_file_name.empty()) {
      std::cout << "Cache om file:" << cache_index.cache_file_name << " is empty" << std::endl;
      return false;
    }

    check_ret = mmAccess(cache_index.cache_file_name.c_str());
    if (check_ret != 0) {
      std::cout << "Cache om file:" << cache_index.cache_file_name << " is not exist" << std::endl;
      return false;
    }
    if (!cache_index.var_desc_file_name.empty()) {
      check_ret = mmAccess(cache_index.var_desc_file_name.c_str());
      if (check_ret != 0) {
        std::cout << "Cache desc file:" << cache_index.var_desc_file_name << " is not exist" << std::endl;
        return false;
      }
    }
  }
  return true;
}

std::string GetCacheFileName(const std::string &cache_dir, const std::string &graph_key) {
  const auto cache_idx_file = cache_dir + "/" + graph_key + ".idx";
  auto check_ret = mmAccess(cache_idx_file.c_str());
  if (check_ret != 0) {
    std::cout << "Cache index file:" << cache_idx_file << " is not exist" << std::endl;
    return "";
  }

  std::vector<ge::CacheFileIdx> cache_file_list;
  if (!ReadIndexFile(cache_idx_file, cache_file_list)) {
    std::cout << "Faile to read cache index file:" << cache_idx_file << std::endl;
    return "";
  }
  for (auto &idx : cache_file_list) {
    idx.cache_file_name = cache_dir + "/" + idx.cache_file_name;
  }

  for (const auto &cache_index : cache_file_list) {
    if (cache_index.graph_key != graph_key) {
      std::cout << "Cache graph_key[" << cache_index.graph_key << "] error, expect = " << graph_key << std::endl;
      continue;
    }

    if (cache_index.cache_file_name.empty()) {
      std::cout << "Cache om file:" << cache_index.cache_file_name << " is empty" << std::endl;
      return "";
    }
    return cache_index.cache_file_name;
  }
  return "";
}

class MockMmpaForOpenFailed : public MmpaStubApiGe {
 public:
  INT32 Open2(const CHAR *path_name, INT32 flags, MODE mode) override {
    return -1;
  }
};

class MockMmpaForFlockFailed : public MmpaStubApiGe {
 public:
  INT32 Open2(const CHAR *path_name, INT32 flags, MODE mode) override {
    return INT32_MAX;
  }
};

}  // namespace

class ModelCacheTest : public testing::Test {
 public:
  static void PrepareForCacheConfig(bool cache_manual_check, bool cache_debug_mode) {
    std::string cache_config_file = "./ut_cache_dir/cache.conf";
    {
      nlohmann::json cfg_json = {
                                  {"cache_manual_check", cache_manual_check},
                                  {"cache_debug_mode", cache_debug_mode}};
      std::ofstream json_file(cache_config_file);
      json_file << cfg_json << std::endl;
    }
  }

  static void RemoveCacheConfig() {
    remove("./ut_cache_dir/cache.conf");
  }

 protected:
  static void SetUpTestSuite() {
    origin_session_options_ = GetThreadLocalContext().GetAllSessionOptions();
    origin_graph_options_ = GetThreadLocalContext().GetAllGraphOptions();
    std::string cmd = R"(
mkdir -p ./workspace/release/
cd ./workspace/release/
touch test1_release.om
touch test1_release.so
echo "Hello" > test1_release.om
echo "test1_release" > test1_release.so
tar -cvf test1_release.tar.gz test1_release.om test1_release.so
cd -
)";
  (void)system(cmd.c_str());
  }
  static void TearDownTestSuite() {
    GetThreadLocalContext().SetSessionOption(origin_session_options_);
    GetThreadLocalContext().SetGraphOption(origin_graph_options_);
    (void)system("rm -rf ./workspace");
  }
  void SetUp() {
    VarManager::Instance(0)->Init(0, 0, 0, 0);
    GetThreadLocalContext().SetSessionOption({});
    GetThreadLocalContext().SetGraphOption({});
    (void)system("mkdir ./ut_cache_dir");
  }

  void TearDown() {
    (void)system("rm -fr ./ut_cache_dir");
    VarManager::Instance(0)->Destory();
  }

  static void SetCacheDirOption(const std::string &cache_dir) {
    GetThreadLocalContext().SetSessionOption({{"ge.graph_compiler_cache_dir", cache_dir}});
  }
  static void ClearCacheDirOption() {
    GetThreadLocalContext().SetSessionOption({});
  }
  static void SetGraphKeyOption(const std::string &graph_key) {
    GetThreadLocalContext().SetGraphOption({{"ge.graph_key", graph_key}});
  }
  static void ClearGraphKeyOption() {
    GetThreadLocalContext().SetGraphOption({});
  }

 private:
  static std::map<std::string, std::string> origin_session_options_;
  static std::map<std::string, std::string> origin_graph_options_;
};
std::map<std::string, std::string> ModelCacheTest::origin_session_options_;
std::map<std::string, std::string> ModelCacheTest::origin_graph_options_;

TEST_F(ModelCacheTest, save_cache_no_need_cache) {
  ModelCache model_cache;
  auto graph1 = FakeComputeGraph("graph1");
  auto ret = model_cache.Init(graph1, nullptr);
  EXPECT_EQ(ret, SUCCESS);

  GeRootModelPtr ge_root_model;
  ret = model_cache.TryCacheModel(ge_root_model);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ModelCacheTest, save_cache_init_graph_no_need_cache) {
  ModelCache model_cache;
  auto graph1 = FakeComputeGraph("graph1_init_by_20000");
  (void)AttrUtils::SetStr(graph1, "_suspend_graph_original_name", "graph1");
  auto ret = model_cache.Init(graph1, nullptr);
  EXPECT_EQ(ret, SUCCESS);

  GeRootModelPtr ge_root_model;
  ret = model_cache.TryCacheModel(ge_root_model);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ModelCacheTest, save_cache_only_cache_dir_option) {
  SetCacheDirOption("./ut_cache_dir/");
  auto graph1 = FakeComputeGraph("graph1");
  ModelCache model_cache;
  auto ret = model_cache.Init(graph1, nullptr);
  EXPECT_EQ(ret, SUCCESS);
  GeRootModelPtr ge_root_model;
  ret = model_cache.TryCacheModel(ge_root_model);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ModelCacheTest, save_cache_only_graph_key_option) {
  SetGraphKeyOption("graph_key_1");
  ModelCache model_cache;
  auto graph1 = FakeComputeGraph("graph1");
  auto ret = model_cache.Init(graph1, nullptr);
  EXPECT_EQ(ret, SUCCESS);
  GeRootModelPtr ge_root_model;
  ret = model_cache.TryCacheModel(ge_root_model);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ModelCacheTest, save_cache_cache_dir_not_exit) {
  SetCacheDirOption("./ut_cache_dir_not_exit/");
  SetGraphKeyOption("graph_key_1");
  ModelCache model_cache;
  auto graph1 = FakeComputeGraph("graph1");
  auto ret = model_cache.Init(graph1, nullptr);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(ModelCacheTest, save_and_load_ge_root_model) {
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_1");
  auto graph = FakeComputeGraph("test_graph_cache");
  GeRootModelPtr ge_root_model = BuildGeRootModel("test_graph_cache", graph);
  GraphRebuildStateCtrl ctrl;
  {
    ModelCache model_cache;
    auto ret = model_cache.Init(graph, &ctrl);
    EXPECT_EQ(ret, SUCCESS);
    ret = model_cache.TryCacheModel(ge_root_model);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = CheckCacheResult("./ut_cache_dir", "graph_key_1", 1);
    EXPECT_EQ(check_ret, true);
  }

  {
    ModelCache model_cache_for_load;
    auto ret = model_cache_for_load.Init(graph, &ctrl);
    EXPECT_EQ(ret, SUCCESS);
    GeRootModelPtr load_model;
    ret = model_cache_for_load.TryLoadModelFromCache(graph, load_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_model, nullptr);
  }
}

TEST_F(ModelCacheTest, save_and_load_flow_model) {
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_flow_model1");
  auto graph = FakeComputeGraph("root_graph");
  auto ge_root_model = BuildGeModel("flow_model1", graph);
  GraphRebuildStateCtrl ctrl;
  {
    ModelCache model_cache;
    auto ret = model_cache.Init(graph, &ctrl);
    EXPECT_EQ(ret, SUCCESS);
    ret = model_cache.TryCacheModel(ge_root_model);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = CheckCacheResult("./ut_cache_dir", "graph_key_flow_model1", 1);
    EXPECT_EQ(check_ret, true);
  }
  {
    auto fake_graph = FakeComputeGraph("test_load_graph");
    AttrUtils::SetStr(*fake_graph, ATTR_NAME_SESSION_GRAPH_ID, "100_99");
    ModelCache flow_model_cache_for_load;
    auto ret = flow_model_cache_for_load.Init(fake_graph, &ctrl);
    EXPECT_EQ(ret, SUCCESS);
    GeRootModelPtr load_model;
    ret = flow_model_cache_for_load.TryLoadModelFromCache(fake_graph, load_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_model, nullptr);
  }
}

TEST_F(ModelCacheTest, save_and_load_with_subgraph) {
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_flow_model_with_sub_graph");
  auto sub_graph = FakeComputeGraph("sub_graph");
  auto root_graph = FakeGraphWithSubGraph("root_graph", sub_graph);
  auto ge_root_model = BuildGeModel("flow_model1", root_graph);
  GraphRebuildStateCtrl ctrl;
  {
    ModelCache model_cache;
    auto ret = model_cache.Init(root_graph, &ctrl);
    EXPECT_EQ(ret, SUCCESS);
    ret = model_cache.TryCacheModel(ge_root_model);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = CheckCacheResult("./ut_cache_dir", "graph_key_flow_model_with_sub_graph", 1);
    EXPECT_EQ(check_ret, true);
    // save will not change orgin graph
    EXPECT_EQ(root_graph->GetAllSubgraphs().size(), 1);
  }
  {
    auto fake_graph = FakeComputeGraph("test_load_graph");
    AttrUtils::SetStr(*fake_graph, ATTR_NAME_SESSION_GRAPH_ID, "100_99");
    ModelCache model_cache_for_load;
    auto ret = model_cache_for_load.Init(fake_graph, &ctrl);
    EXPECT_EQ(ret, SUCCESS);
    GeRootModelPtr load_model;
    ret = model_cache_for_load.TryLoadModelFromCache(fake_graph, load_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_model, nullptr);
    
    const auto &loaded_graph = load_model->GetRootGraph();
    // flow model root graph not save subgraph
    EXPECT_EQ(loaded_graph->GetAllSubgraphs().size(), 0);
  }
}

TEST_F(ModelCacheTest, TransModelDataToComputeGraph) {
  auto data1 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16})
        .Attr(ATTR_NAME_INDEX, 0);
  auto neg = OP_CFG(NEG).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});
  auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT32, {-1});
  DEF_GRAPH(graph) {
      CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("neg", neg)->NODE("Node_Output", netoutput));
  };
  auto root_graph = ToComputeGraph(graph);
  root_graph->SetName("graph");
  root_graph->SetSessionID(0);
  AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0_1");
  auto output_node = root_graph->FindNode("Node_Output");
  output_node->GetOpDesc()->SetSrcIndex({0});
  output_node->GetOpDesc()->SetSrcName({"neg"});
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(root_graph), ge::SUCCESS);
  auto ge_model = MakeShared<ge::GeModel>();
  auto model_task_def = MakeShared<domi::ModelTaskDef>();
  model_task_def->set_version("test_v100_r001");
  ge_model->SetModelTaskDef(model_task_def);
  ge_model->SetName("graph");
  ge_model->SetGraph(root_graph);
  ge_root_model->SetModelName("graph");	
  ge_root_model->SetSubgraphInstanceNameToModel("graph", ge_model);	
  bool is_unknown_shape = false;
  EXPECT_EQ(ge_root_model->CheckIsUnknownShape(is_unknown_shape), ge::SUCCESS);
  ModelBufferData model_buffer_data{};
  const auto model_save_helper =
    ModelSaveHelperFactory::Instance().Create(OfflineModelFormat::OM_FORMAT_DEFAULT);
  model_save_helper->SetSaveMode(false);
  EXPECT_EQ(model_save_helper->SaveToOmRootModel(ge_root_model, "graph", model_buffer_data, is_unknown_shape), ge::SUCCESS);
  ModelData model_data{};
  model_data.model_data = model_buffer_data.data.get();
  model_data.model_len = model_buffer_data.length;
  ModelHelper model_helper;
  EXPECT_EQ(model_helper.LoadRootModel(model_data), ge::SUCCESS);
  ComputeGraphPtr test_compute_graph = model_helper.GetGeRootModel()->GetRootGraph();
  EXPECT_NE(test_compute_graph, nullptr);
  EXPECT_EQ(test_compute_graph->GetName(), "graph");
}

TEST_F(ModelCacheTest, save_and_load_flow_model_with_constant) {
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_flow_model1");
  auto graph = FakeComputeGraphWithConstant("root_graph");
  auto ge_root_model = BuildGeModel2("flow_model1", graph);
  GraphRebuildStateCtrl ctrl;
  {
    ModelCache model_cache;
    auto ret = model_cache.Init(graph, &ctrl);
    EXPECT_EQ(ret, SUCCESS);
    ret = model_cache.TryCacheModel(ge_root_model);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = CheckCacheResult("./ut_cache_dir", "graph_key_flow_model1", 1);
    EXPECT_EQ(check_ret, true);
  }
  {
    auto fake_graph = FakeComputeGraphWithConstant("test_load_graph");
    AttrUtils::SetStr(*fake_graph, ATTR_NAME_SESSION_GRAPH_ID, "100_99");
    ModelCache model_cache_for_load;
    auto ret = model_cache_for_load.Init(fake_graph, &ctrl);
    EXPECT_EQ(ret, SUCCESS);
    GeRootModelPtr load_model;
    ret = model_cache_for_load.TryLoadModelFromCache(fake_graph, load_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_model, nullptr);
  }
}

TEST_F(ModelCacheTest, save_and_load_flow_model_with_var) {
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_flow_model1");
  auto graph = FakeComputeGraphWithVar("root_graph");
  auto ge_root_model = BuildGeModel2("flow_model1", graph);

  GraphRebuildStateCtrl ctrl;
  {
    ModelCache model_cache;
    auto ret = model_cache.Init(graph, &ctrl);
    EXPECT_EQ(ret, SUCCESS);
    ret = model_cache.TryCacheModel(ge_root_model);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = CheckCacheResult("./ut_cache_dir", "graph_key_flow_model1", 1);
    EXPECT_EQ(check_ret, true);
  }
  {
    auto fake_graph = FakeComputeGraphWithVar("test_load_graph");
    AttrUtils::SetStr(*fake_graph, ATTR_NAME_SESSION_GRAPH_ID, "100_99");
    ModelCache model_cache_for_load;
    auto ret = model_cache_for_load.Init(fake_graph, &ctrl);
    EXPECT_EQ(ret, SUCCESS);
    GeRootModelPtr load_model;
    ret = model_cache_for_load.TryLoadModelFromCache(fake_graph, load_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_model, nullptr);
  }
}

TEST_F(ModelCacheTest, save_and_load_flow_model_with_invalid_graph_key) {
  // cannot as file name.
  std::string graph_key = "can/not/be/file/name";
  std::string expect_hash_idx_name("hash_");
  auto hash_code = std::hash<std::string>{}(graph_key);
  expect_hash_idx_name.append(std::to_string(hash_code));

  GeTensorDesc tensor_desc(GeShape({16, 16}), FORMAT_ND, DT_INT16);
  EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("some_var", nullptr, tensor_desc, RT_MEMORY_HBM), SUCCESS);

  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption(graph_key);
  auto graph = FakeComputeGraph("root_graph");
  GraphRebuildStateCtrl ctrl;
  ModelCache model_cache;
  auto ret = model_cache.Init(graph, &ctrl);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(ModelCacheTest, save_and_load_flow_model_check_rang_id_and_priority) {
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_flow_model_1");
  auto graph = FakeComputeGraph("root_graph");
  auto ge_root_model = BuildGeModel("flow_model1", graph);
  GraphRebuildStateCtrl ctrl;
  {
    ModelCache model_cache;
    auto ret = model_cache.Init(graph, &ctrl);
    EXPECT_EQ(ret, SUCCESS);
    ret = model_cache.TryCacheModel(ge_root_model);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = CheckCacheResult("./ut_cache_dir", "graph_key_flow_model_1", 1);
    EXPECT_EQ(check_ret, true);
  }
  {
    auto fake_graph = FakeComputeGraph("test_load_graph");
    AttrUtils::SetStr(*fake_graph, ATTR_NAME_SESSION_GRAPH_ID, "100_99");
    ModelCache model_cache_for_load;
    auto ret = model_cache_for_load.Init(fake_graph, &ctrl);
    EXPECT_EQ(ret, SUCCESS);
    GeRootModelPtr load_model;
    ret = model_cache_for_load.TryLoadModelFromCache(fake_graph, load_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_model, nullptr);
  }
}

TEST_F(ModelCacheTest, save_and_load_flow_model_compile_and_deploy) {
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_flow_model1");
  auto graph = FakeComputeGraph("root_graph");
  auto ge_root_model = BuildGeModel("flow_model1", graph);
  GraphRebuildStateCtrl ctrl;
  {
    ModelCache model_cache;
    auto ret = model_cache.Init(graph, &ctrl);
    EXPECT_EQ(ret, SUCCESS);
    ret = model_cache.TryCacheModel(ge_root_model);
    GELOGD("model_cache after TryCacheModel");
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = CheckCacheResult("./ut_cache_dir", "graph_key_flow_model1", 1);
    EXPECT_EQ(check_ret, true);
  }
  {
    auto fake_graph = FakeComputeGraph("test_load_graph");
    AttrUtils::SetStr(*fake_graph, ATTR_NAME_SESSION_GRAPH_ID, "100_99");
    ModelCache model_cache_for_load;
    auto ret = model_cache_for_load.Init(fake_graph, &ctrl);
    EXPECT_EQ(ret, SUCCESS);
    GeRootModelPtr load_model;
    ret = model_cache_for_load.TryLoadModelFromCache(fake_graph, load_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_model, nullptr);
    

    // load with no index file
    (void)system("rm -fr ./ut_cache_dir/graph_key_flow_model1.idx");
    GeRootModelPtr load_model2;
    ret = model_cache_for_load.TryLoadModelFromCache(fake_graph, load_model2);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_model2, nullptr);
  }
}

TEST_F(ModelCacheTest, load_and_save_flow_model) {
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_flow_model1");

  GeTensorDesc tensor_desc(GeShape({16, 16}), FORMAT_ND, DT_INT16);
  EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("some_var", nullptr, tensor_desc, RT_MEMORY_HBM), SUCCESS);

  auto fake_graph = FakeComputeGraph("test_load_graph");
  AttrUtils::SetStr(*fake_graph, ATTR_NAME_SESSION_GRAPH_ID, "100_99");
  GraphRebuildStateCtrl ctrl;
  GeRootModelPtr load_model;
  {
    ModelCache model_cache;
    auto ret = model_cache.Init(fake_graph, &ctrl);
    EXPECT_EQ(ret, SUCCESS);
    // not match
    ret = model_cache.TryLoadModelFromCache(fake_graph, load_model);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(load_model, nullptr);

    // mock var fusion
    GeTensorDesc tensor_desc1(GeShape({16, 16}), FORMAT_NCHW, DT_INT16);
    EXPECT_EQ(VarManager::Instance(0)->RecordStagedVarDesc(0, "some_var", tensor_desc1), SUCCESS);
    EXPECT_EQ(VarManager::Instance(0)->SetChangedGraphId("some_var", 0), SUCCESS);
    TransNodeInfo trans_node_info;
    VarTransRoad fusion_road;
    fusion_road.emplace_back(trans_node_info);
    EXPECT_EQ(VarManager::Instance(0)->SetTransRoad("some_var", fusion_road), SUCCESS);

    GeTensorDesc tensor_desc2(GeShape({16, 16}), FORMAT_NHWC, DT_INT16);
    EXPECT_EQ(VarManager::Instance(0)->RenewCurVarDesc("some_var", tensor_desc2), SUCCESS);

    auto graph = FakeComputeGraph("root_graph");
    auto ge_root_model = BuildGeModel("flow_model1", graph);
    ret = model_cache.TryCacheModel(ge_root_model);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = CheckCacheResult("./ut_cache_dir", "graph_key_flow_model1", 1);
    EXPECT_EQ(check_ret, true);
  }
  {
    ModelCache model_cache_for_load;
    auto ret = model_cache_for_load.Init(fake_graph, &ctrl);
    EXPECT_EQ(ret, SUCCESS);

    // resume var
    EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("some_var", nullptr, tensor_desc, RT_MEMORY_HBM), SUCCESS);
    // match cache
    ret = model_cache_for_load.TryLoadModelFromCache(fake_graph, load_model);
    EXPECT_EQ(ret, SUCCESS);

    ASSERT_NE(load_model, nullptr);
  }
}

TEST_F(ModelCacheTest, save_flow_model_open_lock_file_failed) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForOpenFailed>());
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_flow_model1");
  auto graph = FakeComputeGraph("root_graph");
  GraphRebuildStateCtrl ctrl;
  ModelCache model_cache;
  auto ret = model_cache.Init(graph, &ctrl);
  EXPECT_EQ(ret, FAILED);
  MmpaStub::GetInstance().Reset();
}

TEST_F(ModelCacheTest, save_flow_model_open_lock_failed) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForFlockFailed>());
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_flow_model1");
  auto graph = FakeComputeGraph("root_graph");
  GraphRebuildStateCtrl ctrl;
  ModelCache model_cache;
  auto ret = model_cache.Init(graph, &ctrl);
  EXPECT_EQ(ret, FAILED);
  MmpaStub::GetInstance().Reset();
}

TEST_F(ModelCacheTest, update_model_task_addr) {
  DEF_GRAPH(test_graph) {
    auto file_constant =
        OP_CFG(FILECONSTANT).InCnt(0).OutCnt(1)
            .Attr("shape", GeShape{})
            .Attr("dtype", DT_FLOAT)
            .Attr("file_id", "fake_id");
    auto ffts_plus_neg = OP_CFG(NEG).InCnt(1).OutCnt(1);
    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {});
    CHAIN(NODE("file_constant", file_constant)->NODE("ffts_plus_neg", ffts_plus_neg)->NODE("Node_Output", net_output));
  };

  auto compute_graph = ToComputeGraph(test_graph);
  compute_graph->SetName(test_graph.GetName());
  compute_graph->SetSessionID(0);
  AttrUtils::SetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, "0_1");

  auto op_desc = compute_graph->FindNode("Node_Output")->GetOpDesc();
  std::vector<std::string> src_name{"out"};
  op_desc->SetSrcName(src_name);
  std::vector<int64_t> src_index{0};
  op_desc->SetSrcIndex(src_index);

  auto fc_node = compute_graph->FindFirstNodeMatchType(FILECONSTANT);
  fc_node->GetOpDesc()->SetOutputOffset({0x10000000});
  auto neg_node = compute_graph->FindFirstNodeMatchType(NEG);
  neg_node->GetOpDesc()->SetInputOffset({0x10000000});
  auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  auto ge_model = MakeShared<ge::GeModel>();
  ge_model->SetName("test");
  ge_model->SetGraph(compute_graph);
  auto model_task_def = MakeShared<domi::ModelTaskDef>();
  auto task_def = model_task_def->add_task();
  task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto ffts_plus_task = task_def->mutable_ffts_plus_task();
  auto ctx_def = ffts_plus_task->add_ffts_plus_ctx();
  auto mutable_mix_aic_aiv_ctx = ctx_def->mutable_mix_aic_aiv_ctx();
  mutable_mix_aic_aiv_ctx->add_task_addr(0x10000000);
  mutable_mix_aic_aiv_ctx->add_task_addr(0x20000000);
  model_task_def->set_version("test_v100_r001");
  ge_model->SetModelTaskDef(model_task_def);
  ge_root_model->SetModelName("test_model");
  ge_root_model->SetSubgraphInstanceNameToModel("model_name", ge_model);
  PrepareForCacheConfig(false, false);
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_1");
  std::string cmd = R"(
  touch ./ut_cache_dir/graph_key_1.om
  )";
  (void)system(cmd.c_str());
  GraphRebuildStateCtrl ctrl;
  {
    ModelCache model_cache;
    auto ret = model_cache.Init(compute_graph, &ctrl);
    EXPECT_EQ(ret, SUCCESS);
    ret = model_cache.TryCacheModel(ge_root_model);
    EXPECT_EQ(ret, SUCCESS);
    auto check_ret = CheckCacheResult("./ut_cache_dir", "graph_key_1", 1);
    EXPECT_EQ(check_ret, true);
  }

  {
    ModelCache model_cache_for_load;
    auto ret = model_cache_for_load.Init(compute_graph, &ctrl);
    EXPECT_EQ(ret, SUCCESS);
    GeRootModelPtr load_model;
    ret = model_cache_for_load.TryLoadModelFromCache(compute_graph, load_model);
    EXPECT_EQ(ret, SUCCESS);
    ASSERT_NE(load_model, nullptr);
  }
  RemoveCacheConfig();
}

TEST_F(ModelCacheTest, ReadCacheConfig_Failed) {
  DEF_GRAPH(test_graph) {
    auto file_constant =
        OP_CFG(FILECONSTANT).InCnt(0).OutCnt(1)
            .Attr("shape", GeShape{})
            .Attr("dtype", DT_FLOAT)
            .Attr("file_id", "fake_id");
    auto ffts_plus_neg = OP_CFG(NEG).InCnt(1).OutCnt(1);
    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {});
    CHAIN(NODE("file_constant", file_constant)->NODE("ffts_plus_neg", ffts_plus_neg)->NODE("Node_Output", net_output));
  };

  auto compute_graph = ToComputeGraph(test_graph);
  compute_graph->SetName(test_graph.GetName());
  compute_graph->SetSessionID(0);
  AttrUtils::SetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, "0_1");

  std::string cache_config_file = "./ut_cache_dir/cache.conf";
  {
    nlohmann::json cfg_json = {
        {"cache_manual_check", "failed"},
        {"cache_debug_mode", "failed"}};
    std::ofstream json_file(cache_config_file);
    json_file << cfg_json << std::endl;
  }
  SetCacheDirOption("./ut_cache_dir");
  SetGraphKeyOption("graph_key_1");
  GraphRebuildStateCtrl ctrl;
  {
    ModelCache model_cache;
    auto ret = model_cache.Init(compute_graph, &ctrl);
    EXPECT_NE(ret, SUCCESS);
  }
  RemoveCacheConfig();
}
}  // namespace ge

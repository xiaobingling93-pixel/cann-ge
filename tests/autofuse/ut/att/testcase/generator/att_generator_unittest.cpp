/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <iostream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#define private public
#define protected public
#include "tiling_code_generator.h"
#include "high_perf_tiling_code_gen_impl.h"
#include "tiling_code_gen_impl.h"
#undef private
#undef protected
#include "args_manager.h"
#include "generator_utils/tilingdata_gen_utils.h"

#include <symengine/symengine_rcp.h>
#include <symengine/basic.h>
#include <symengine/symbol.h>
#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/integer.h>
#include "solver_pass_manager/stub_model_info.h"
#include "reuse_group_utils/reuse_group_utils.h"
#include "tiling_data_gen/tiling_data_generator.h"
#include "base/base_types.h"

const std::string op_name = "OpTest";

namespace att {
class MockHighPerfTilingCodeGenImpl : public HighPerfTilingCodeGenImpl {
 public:
  MockHighPerfTilingCodeGenImpl(const std::string &mock_op_name, const TilingCodeGenConfig &config,
                                const TilingModelInfo &model_infos, const ScoreFuncs &score_funcs,
                                const bool is_uniq_group)
      : HighPerfTilingCodeGenImpl(mock_op_name, config, model_infos, score_funcs, is_uniq_group) {}
};

class MockTilingCodeGenerator : public TilingCodeGenerator {
 protected:
  TilingCodeGenImplPtr CreateTilingCodeGenImpl(const std::string &mock_op_name, const TilingCodeGenConfig &config,
                                               const TilingModelInfo &model_infos, const ScoreFuncs &score_funcs,
                                               const bool is_uniq_group) override {
    std::shared_ptr<MockHighPerfTilingCodeGenImpl> impl =
        std::make_shared<MockHighPerfTilingCodeGenImpl>(mock_op_name, config, model_infos, score_funcs, is_uniq_group);
    return impl;
  }
};

class GeneratorUT : public testing::Test {};

TEST(GeneratorUT, Normal) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo();
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = true;
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), ge::SUCCESS);
}

TEST(GeneratorUT, NormalStaticUint32Shape) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo(1, ge::ExprType::kExprConstantInteger);
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = false;
  config.gen_tiling_data = false;
  TilingCodeGenerator generator;
  std::map<size_t, std::map<size_t, std::vector<ModelInfo>>> model_infos_new;
  model_infos_new[0][0] = model_infos;
  std::map<std::string, std::string> tiling_res;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config, tiling_res), ge::SUCCESS);
  ASSERT_EQ(tiling_res.size(), 4);
}

TEST(GeneratorUT, NormalStaticRationShape) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo(1, ge::ExprType::kExprConstantRation);
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = false;
  config.gen_tiling_data = false;
  TilingCodeGenerator generator;
  std::map<size_t, std::map<size_t, std::vector<ModelInfo>>> model_infos_new;
  model_infos_new[0][0] = model_infos;
  std::map<std::string, std::string> tiling_res;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config, tiling_res), ge::SUCCESS);
  ASSERT_EQ(tiling_res.size(), 4);
}

TEST(GeneratorUT, GenTilingSolverSuccess) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo();
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  MockTilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), ge::SUCCESS);
}

TEST(GeneratorUT, InvalidConfig) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo;
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::MAX;
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_NE(generator.GenTilingCode(op_name, model_infos, config), ge::SUCCESS);
}

TEST(GeneratorUT, TestSymengine) {
  using namespace SymEngine;
  using SymEngine::RCP;
  using SymEngine::make_rcp;
  using SymEngine::Basic;
  using SymEngine::Symbol;
  const RCP<const Basic> x = make_rcp<SymEngine::Symbol>("x");
  EXPECT_EQ(x->__str__(), "x");
}

TEST(GeneratorUT, TestSymengine2) {
  using namespace SymEngine;
  RCP<const Basic> x1 = symbol("x1");
  RCP<const Basic> x2 = symbol("x2");
  RCP<const Basic> int1 = integer(1);
  RCP<const Basic> int2 = integer(2);
  RCP<const Basic> y = mul(x2, add(x1, int1));
  RCP<const Basic> z = mul(add(int1, x1), x2);
  EXPECT_EQ(x1->__str__(), "x1");
  EXPECT_EQ(x2->__str__(), "x2");
  EXPECT_EQ(y->__str__(), "x2*(1 + x1)");
  EXPECT_EQ(z->__str__(), "x2*(1 + x1)");
  EXPECT_EQ(is_a<Symbol>(*x1), true);
  EXPECT_EQ(is_a<Symbol>(*x2), true);
  EXPECT_EQ(is_a<Symbol>(*y), false);
  EXPECT_EQ(is_a<Symbol>(*z), false);
  EXPECT_EQ(is_a<Integer>(*int1), true);
  RCP<const Basic> multi_add = add(add(int1, x1), int2);
  EXPECT_EQ(multi_add->__str__(), "3 + x1");
  RCP<const Basic> m = add(mul(add(int1, x1), x2), int2);
  EXPECT_EQ(m->get_args()[0]->__str__(), "2");
  EXPECT_EQ(m->get_args()[1]->__str__(), "x2*(1 + x1)");
}

TEST(GeneratorUT, AddElementInTilingData) {
  ge::CodePrinter dumper;
  TilingDataGenUtils::AddStructElementDefinition(dumper, "TCubeTiling", "mm_tiling");
  EXPECT_TRUE(dumper.GetOutputStr().find("TCubeTiling, mm_tiling") != std::string::npos);
}

TEST(GeneratorUT, TestSchedGroup) {
  ModelInfo modelInfo = CreateModelInfo();
  FusedParsedScheduleResult fused_schedule_result;
  auto &all_model_infos = fused_schedule_result[0];
  std::map<size_t, std::vector<ModelInfo>> model_infos1;

  model_infos1[0] = {modelInfo, modelInfo};
  model_infos1[0][0].schedule_group_ident.impl_graph_id = 0;
  model_infos1[0][0].schedule_group_ident.group_id = 0;
  model_infos1[0][0].tiling_case_id = 0;
  model_infos1[0][1].schedule_group_ident.impl_graph_id = 0;
  model_infos1[0][1].schedule_group_ident.group_id = 0;
  model_infos1[0][1].tiling_case_id = 1;

  model_infos1[1] = {modelInfo};
  model_infos1[1][0].schedule_group_ident.impl_graph_id = 0;
  model_infos1[1][0].schedule_group_ident.group_id = 1;
  model_infos1[1][0].tiling_case_id = 2;
  for (auto &model_info : model_infos1) {
    EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_info.second), ge::SUCCESS);
  }
  all_model_infos[0].groups_tiling_model_info = model_infos1;
  all_model_infos[0].impl_graph_id = 0;
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = true;
  config.gen_extra_infos = true;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  EXPECT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), ge::SUCCESS);
}

TEST(GeneratorUT, TestSchedGroupEnableGroupParallel) {
  ModelInfo modelInfo = CreateModelInfo();
  FusedParsedScheduleResult fused_schedule_result;
  auto &all_model_infos = fused_schedule_result[0];
  std::map<size_t, std::vector<ModelInfo>> model_infos1;

  model_infos1[0] = {modelInfo, modelInfo};
  model_infos1[0][0].schedule_group_ident.impl_graph_id = 0;
  model_infos1[0][0].schedule_group_ident.group_id = 0;
  model_infos1[0][0].tiling_case_id = 0;
  model_infos1[0][0].enable_group_parallel = true;
  model_infos1[0][1].schedule_group_ident.impl_graph_id = 0;
  model_infos1[0][1].schedule_group_ident.group_id = 0;
  model_infos1[0][1].tiling_case_id = 1;
  model_infos1[0][1].enable_group_parallel = true;

  model_infos1[1] = {modelInfo};
  model_infos1[1][0].schedule_group_ident.impl_graph_id = 0;
  model_infos1[1][0].schedule_group_ident.group_id = 1;
  model_infos1[1][0].tiling_case_id = 2;
  model_infos1[1][0].enable_group_parallel = true;
  for (auto &model_info : model_infos1) {
    EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_info.second), ge::SUCCESS);
  }
  all_model_infos[0].groups_tiling_model_info = model_infos1;
  all_model_infos[0].impl_graph_id = 0;
  all_model_infos[0].enable_group_parallel = true;

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = true;
  config.gen_extra_infos = true;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  EXPECT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), ge::SUCCESS);
  bool flag_arrange = false;
  bool flag_parallel = false;
  for (const auto &[key, value] : tiling_res) {
    if (value.find("  ArrangeBlockOffsetsAscGraph0Result0(") != std::string::npos) {
      flag_arrange = true;
    }
    if (value.find("UpdateCurPerfAndBlockByGroup(") != std::string::npos) {
      flag_parallel = true;
    }
  }
  EXPECT_EQ(flag_arrange && flag_parallel, true);
}

TEST(GeneratorUT, CreateAxesReorderTilingCodeGenImplSuccess) {
  TilingModelInfo model_infos;
  model_infos.emplace_back(CreateModelInfo());
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::AXES_REORDER;
  config.gen_extra_infos = true;
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), ge::SUCCESS);
}

TEST(GeneratorUT, TilingCodeGenImplConstruct) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;
  MockHighPerfTilingCodeGenImpl impl("test", config, tiling_model_info, score_funcs, true);
  EXPECT_EQ(config.force_template_op_name, "test");
  impl.GenGetAllSchedulesResults({});
  EXPECT_EQ(impl.tiling_func_.GetOutputStr().empty(), true);
}

TEST(GeneratorUT, TilingCodeGenImplPGO) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;
  ModelInfo info;
  tiling_model_info.push_back(info);
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, false);

  genImpl.config_.enable_autofuse_pgo = true;
  EXPECT_EQ(genImpl.GenTilingImplPublicFunc(), ge::SUCCESS);

  std::string expectCode = R"rawliteral(  bool GetTiling(TilingData &tiling_data) {
    OP_LOGD(OP_NAME, "Execute DoTiling.");
    if (!DoTiling(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to do tiling.");
      return false;
    }
    if (is_empty_tensor_) {
      OP_LOGW(OP_NAME, "Empty tensor, skip DoApiTiling and GeneralTiling.");
      return true;
    }
    DoApiTiling(tiling_data);
    GeneralTiling(tiling_data);
    TilingSummary(tiling_data);
    return true;
  }
  virtual double GetPerf(TilingData &tiling_data) { return 0.0; }
  virtual std::string GetScheduleName() { return ""; }
  virtual void TilingSummary(TilingData &tiling_data) = 0;
  virtual bool ExecutePGOSolver(TilingData &tiling_data, std::vector<AutofuseTilingDataPerf>& tiling_data_list, AutofuseTilingData* autofuse_tiling_data, void* stream, std::unordered_map<int64_t, uint64_t> &workspace_map, std::vector<uint32_t*> block_dim_vec={}) {
    return false;
  }
  virtual int32_t CalcScore(const TilingData &tiling_data) { return 0;}
  virtual void GetTilingData(TilingDataCopy &from_tiling, TilingData &to_tiling) {};
  virtual void SetTilingData(TilingData &from_tiling, TilingDataCopy &to_tiling) {};
  virtual void SetWorkspaceSize(TilingData &tiling_data, std::unordered_map<int64_t, uint64_t> &workspace_map) {};
)rawliteral";
  EXPECT_EQ(genImpl.tiling_func_.output_.str(), expectCode);
}

TEST(GeneratorUT, GenTilingHeadPGO) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  EnableGroupParallels enable_group_parallels;
  std::map<std::string, std::string> tiling_res;
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;
  ModelInfo info;
  ReuseScheduleGroup reuse_schedule_group;
  info.reuse_schedule_group = std::make_shared<ReuseScheduleGroup>();
  tiling_model_info.push_back(info);
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, false);

  genImpl.config_.enable_autofuse_pgo = true;
  genImpl.GenTilingHead(tiling_res, enable_group_parallels);
  std::string expectCode = R"rawliteral(#include "autofuse_tiling_func_common.h"
namespace optiling {

} // namespace optiling
)rawliteral";
  EXPECT_EQ(genImpl.tiling_func_.output_.str(), expectCode);
}

TEST(GeneratorUT, GenScheduleGroupTilingTailPGOSuccess) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  EnableGroupParallels enable_group_parallels;
  std::map<std::string, std::string> tiling_res;
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;

  ModelInfo info;
  ReuseScheduleGroup reuse_schedule_group;
  info.reuse_schedule_group = std::make_shared<ReuseScheduleGroup>();
  tiling_model_info.push_back(info);
  enable_group_parallels[0][0] = true;

  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, false);
  genImpl.config_.enable_autofuse_pgo = true;
  genImpl.config_.gen_tiling_data = false;
  genImpl.enable_group_parallels_ = enable_group_parallels;
  EXPECT_EQ(genImpl.GenScheduleGroupTilingTail(), ge::SUCCESS);

  EXPECT_EQ(genImpl.tiling_func_.GetOutputStr().empty(), false);
}

TEST(GeneratorUT, GenTilingPGOSuccess) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  EnableGroupParallels enable_group_parallels;
  std::map<std::string, std::string> tiling_res;
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;

  ModelInfo info;
  ReuseScheduleGroup reuse_schedule_group;
  info.reuse_schedule_group = std::make_shared<ReuseScheduleGroup>();
  tiling_model_info.push_back(info);
  enable_group_parallels[0][0] = true;

  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, true);
  genImpl.config_.enable_autofuse_pgo = true;
  genImpl.config_.gen_tiling_data = false;
  EXPECT_EQ(genImpl.GenTiling(tiling_res, {}, 0, enable_group_parallels), ge::SUCCESS);

  EXPECT_EQ(genImpl.tiling_func_.GetOutputStr().empty(), false);
}

TEST(GeneratorUT, GenGetScheduleResultPGOSuccess) {
  TilingCodeGenConfig config;
  config.tiling_data_type_name = "AutofuseTilingData";
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;

  TilingModelInfo tiling_model_info;
  ModelInfo group0_info;
  group0_info.schedule_group_ident.asc_graph_id = 0;
  group0_info.schedule_group_ident.impl_graph_id = 0;
  group0_info.schedule_group_ident.group_id = 0;
  tiling_model_info.push_back(group0_info);
  ModelInfo group1_info;
  group1_info.schedule_group_ident.asc_graph_id = 0;
  group1_info.schedule_group_ident.impl_graph_id = 0;
  group1_info.schedule_group_ident.group_id = 1;
  tiling_model_info.push_back(group1_info);

  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, true);

  std::map<size_t, std::pair<std::string, std::string>> graph_info;
  graph_info[0] = std::make_pair("ScheduleResult0", "group0");
  graph_info[1] = std::make_pair("ScheduleResult0", "group1");

  std::map<std::string, std::set<std::string>> hardware_map;
  hardware_map["group0"].insert("block_dim");
  hardware_map["group1"].insert("block_dim");

  genImpl.tiling_func_.Reset();
  EXPECT_EQ(genImpl.GenPGOGetScheduleResult(0, 0, graph_info, hardware_map), ge::SUCCESS);

  std::string expectCode = R"rawliteral(inline bool GetScheduleResult0PGO(std::vector<AutofuseTilingDataPerf>& tiling_data_list, const uint32_t ori_block_dim, const int32_t tiling_case_id,AutofuseTilingData &tiling_data, double &cur_perf, double &best_perf, uint32_t &cur_block_dim,void* stream, uint32_t workspaceSize, std::vector<uint32_t*> multi_group_block_dim_list = {}) {
  std::vector<AutofuseTilingDataPerf> tiling_data_list_tmp{};
  workspaceSize = 0;
  std::unordered_map<int64_t, uint64_t> workspace_map_filter_use{};
  tiling_data.set_graph0_tiling_key(0);
  auto &group0_tiling_data = tiling_data.group0_tiling_data;
  group0_tiling_data.set_block_dim(ori_block_dim);
  auto result0 = ScheduleResult0::PGOSearchTilingKey(tiling_data_list_tmp, group0_tiling_data, tiling_case_id, &tiling_data, stream, workspaceSize, best_perf, workspace_map_filter_use, multi_group_block_dim_list);
  if (result0) {
    bool has_solution = true;
    for (auto &tiling_data_perf : tiling_data_list_tmp) {
      auto &tiling_data = tiling_data_perf.tiling_data;
      std::unordered_map<int64_t, uint64_t> workspace_map;
      workspace_map.reserve(workspace_map_filter_use.size());
      workspace_map.insert(workspace_map_filter_use.begin(), workspace_map_filter_use.end());
      tiling_data.group1_tiling_data.set_block_dim(ori_block_dim);
      has_solution = ScheduleResult0::GetTiling(tiling_data.group1_tiling_data, workspace_map, -1);
      if (!has_solution) {
        OP_LOGI(OP_NAME, "No solution for group0 at group1");
        continue;
      }
      auto workspaceSizeTmp = GetWorkspaceSize(tiling_data);
      if (workspaceSizeTmp > workspaceSize) {
        workspaceSize = workspaceSizeTmp;
      }
    }
    workspaceSize += 16 * 1024 * 1024;
    PgoConfig::Instance().batch_callback(stream, workspaceSize, &tiling_data_list_tmp);
    for (auto &tiling_data_perf : tiling_data_list_tmp) {
      tiling_data_list.push_back(tiling_data_perf);
      if (tiling_data_perf.best_perf < best_perf) {
        tiling_data = tiling_data_perf.tiling_data;
        best_perf = tiling_data_perf.best_perf;
      }
    }
  }
  auto &group1_tiling_data = tiling_data.group1_tiling_data;
  group1_tiling_data.set_block_dim(ori_block_dim);
  auto result1 = ScheduleResult0::PGOSearchTilingKey(tiling_data_list_tmp, group1_tiling_data, tiling_case_id, &tiling_data, stream, workspaceSize, best_perf, workspace_map_filter_use, multi_group_block_dim_list);
  if (result1) {
    bool has_solution = true;
    for (auto &tiling_data_perf : tiling_data_list_tmp) {
      auto &tiling_data = tiling_data_perf.tiling_data;
      std::unordered_map<int64_t, uint64_t> workspace_map;
      workspace_map.reserve(workspace_map_filter_use.size());
      workspace_map.insert(workspace_map_filter_use.begin(), workspace_map_filter_use.end());
      auto workspaceSizeTmp = GetWorkspaceSize(tiling_data);
      if (workspaceSizeTmp > workspaceSize) {
        workspaceSize = workspaceSizeTmp;
      }
    }
    workspaceSize += 16 * 1024 * 1024;
    PgoConfig::Instance().batch_callback(stream, workspaceSize, &tiling_data_list_tmp);
    for (auto &tiling_data_perf : tiling_data_list_tmp) {
      tiling_data_list.push_back(tiling_data_perf);
      if (tiling_data_perf.best_perf < best_perf) {
        tiling_data = tiling_data_perf.tiling_data;
        best_perf = tiling_data_perf.best_perf;
      }
    }
  }
  return true;
}
)rawliteral";
  EXPECT_EQ(genImpl.tiling_func_.output_.str(), expectCode);

  EnableGroupParallels enable_group_parallels;
  enable_group_parallels[0][0] = true;
  genImpl.tiling_func_.Reset();
  genImpl.enable_group_parallels_ = enable_group_parallels;
  EXPECT_EQ(genImpl.GenPGOGetScheduleResult(0, 0, graph_info, hardware_map), ge::SUCCESS);
  EXPECT_EQ(genImpl.tiling_func_.GetOutputStr().empty(), false);
}

// UT测试：验证tiling_data.set参数溢出修复
// 测试用例1: 验证 MemoryTilingDataGen::GenFuncImpl 使用 static_cast<uint32_t>()
TEST(GeneratorUT, MemoryTilingDataGen_GenFuncImpl_UseStaticCast) {
  ModelInfo model_info;
  // 创建大数值表达式: 70000 * 70000 * 4 > UINT32_MAX(4294967295)
  // 70000 * 70000 = 4900000000 > UINT32_MAX
  Expr large_expr = ge::Symbol(70000, "tmp") * ge::Symbol(70000, "tmp") * ge::Symbol(4, "tmp");
  model_info.container_exprs["LargeContainer"] = large_expr;

  // 创建 MemoryTilingDataGen 对象
  auto memory_gen = att::MemoryTilingDataGen(model_info);
  EXPECT_EQ(memory_gen.Init(), ge::SUCCESS);

  // 获取生成的函数实现代码
  const std::vector<std::string> func_impls = memory_gen.GetTilingFuncImpl("TestTilingData");

  // 验证生成的代码包含 "static_cast<uint32_t>("
  bool found_static_cast = false;
  for (const auto &code : func_impls) {
    if (code.find("static_cast<uint32_t>(") != std::string::npos) {
      found_static_cast = true;
      break;
    }
  }
  EXPECT_TRUE(found_static_cast) << "Generated code should contain 'static_cast<uint32_t>(' to prevent overflow";
}

// 测试用例2: 验证硬件约束代码生成使用 double 类型
TEST(GeneratorUT, GenHardwareCheckCode_UseDoubleType) {
  ModelInfo model_info;
  // 创建大数值硬件约束: 102400 * 102400 = 10485760000，可能导致 uint32_t 溢出
  Expr large_hardware_expr = ge::Symbol(102400, "tmp") * ge::Symbol(102400, "tmp");
  model_info.hardware_cons[HardwareDef::UB] = large_hardware_expr;

  TilingModelInfo model_infos;
  model_infos.emplace_back(model_info);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = false;
  config.gen_tiling_data = false;
  MockTilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), ge::SUCCESS);

  // 获取生成的代码
  std::map<std::string, std::string> tiling_res;
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config, tiling_res), ge::SUCCESS);

  // 验证生成的代码包含 "double " 类型声明
  bool found_double_type = false;
  for (const auto &[key, code] : tiling_res) {
    if (code.find("double ") != std::string::npos) {
      found_double_type = true;
      break;
    }
  }
  EXPECT_TRUE(found_double_type) << "Generated hardware check code should contain 'double ' type to prevent overflow";
}

}

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "base/base_types.h"
#define private public
#include "generator/solver_pass_gen/axes_reorder_solver/axes_reorder_solver_gen.h"
#include "gen_model_info/api_perf_register/v1/perf_param_v1.h"
using namespace att;

class TestAxesReorderSolverGen : public ::testing::Test {
 public:
  static void TearDownTestCase()
  {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase()
  {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
     // Code here will be called immediately after the constructor (right
     // before each test).
  }

  void TearDown() override {
     // Code here will be called immediately after each test (right
     // before the destructor).
  }
};

TEST_F(TestAxesReorderSolverGen, TEST_ARRANGE)
{
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("x1");
  std::vector<Expr> cut_cons;
  std::vector<Expr> input_args;
  std::vector<Expr> search_args{x0,x1};
  std::map<HardwareDef, Expr> hardware_cons;
  std::map<Expr, Expr, ExprCmp> arg_align_map;
  std::map<Expr, uint32_t, ExprCmp> const_args;
  std::map<Expr, uint32_t, ExprCmp> axis_priority;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  arg_align_map[x0] = ge::Symbol(16);
  cut_cons.emplace_back(x0 - x1);
  hardware_cons[HardwareDef::L2] = x0 + x1;
  from_axes_map[x0] = {x1};
  axis_priority[x0] = 1;
  axis_priority[x1] = 2;

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetTotalCutCons(cut_cons);
  solver_gen.SetBufferUseAlg(hardware_cons);
  solver_gen.SetConstArgs(const_args);
  solver_gen.SetInputArgs(input_args);
  solver_gen.SetSearchArgs(search_args);
  solver_gen.SetFromAxesMap(from_axes_map);
  solver_gen.SetVarPriority(axis_priority);
  solver_gen.Arrange();
  
  auto vars = solver_gen.local_buffer_tiling_vars_;
  EXPECT_NE(vars.size(), 0);
  EXPECT_EQ(Str(vars[0]), "x0");
}

TEST_F(TestAxesReorderSolverGen, TEST_ARRANGE_SIZE_VAR_AS_INPUT)
{
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("x1");
  Expr x2 = CreateExpr("x2");
  std::vector<Expr> cut_cons;
  std::vector<Expr> input_args;
  std::map<HardwareDef, Expr> hardware_cons;
  std::vector<Expr> search_args;
  std::map<Expr, Expr, ExprCmp> arg_align_map;
  std::map<Expr, uint32_t, ExprCmp> const_args;
  std::map<Expr, uint32_t, ExprCmp> axis_priority;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  arg_align_map[x0] = ge::Symbol(16);
  cut_cons.emplace_back(x0 - x1);
  hardware_cons[HardwareDef::UB] = x0 + x1 + x2;
  from_axes_map[x0] = {x1};
  axis_priority[x0] = 1;
  axis_priority[x1] = 2;
  search_args = {x0, x1};

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetTotalCutCons(cut_cons);
  solver_gen.SetBufferUseAlg(hardware_cons);
  solver_gen.SetConstArgs(const_args);
  solver_gen.SetInputArgs(input_args);
  solver_gen.SetFromAxesMap(from_axes_map);
  solver_gen.SetVarPriority(axis_priority);
  solver_gen.SetSearchArgs(search_args);
  solver_gen.Arrange();
  
  auto vars = solver_gen.input_args_;
  EXPECT_NE(vars.size(), 0);
  EXPECT_EQ(Str(vars[0]), "x2");
}


TEST_F(TestAxesReorderSolverGen, TEST_GEN_SOLVER)
{
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("x1");
  std::vector<Expr> cut_cons;
  std::vector<Expr> input_args;
  std::vector<Expr> search_args{x0,x1};
  std::map<HardwareDef, Expr> hardware_cons;
  std::map<Expr, Expr, ExprCmp> arg_align_map;
  std::map<Expr, uint32_t, ExprCmp> const_args;
  std::map<Expr, uint32_t, ExprCmp> axis_priority;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  arg_align_map[x0] = ge::Symbol(16);
  cut_cons.emplace_back(x0 - x1);
  hardware_cons[HardwareDef::L2] = x0 + x1;
  from_axes_map[x0] = {x1};
  axis_priority[x0] = 2;
  axis_priority[x1] = 1;

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetTotalCutCons(cut_cons);
  solver_gen.SetBufferUseAlg(hardware_cons);
  solver_gen.SetConstArgs(const_args);
  solver_gen.SetInputArgs(input_args);
  solver_gen.SetSearchArgs(search_args);
  solver_gen.SetFromAxesMap(from_axes_map);
  solver_gen.SetVarPriority(axis_priority);
  solver_gen.Arrange();

  auto vars = solver_gen.local_buffer_tiling_vars_;
  EXPECT_EQ(Str(vars[0]), "x1");

  std::string impl_code = solver_gen.GenSolverFuncImpl();
  std::string invoke_code = solver_gen.GenSolverFuncInvoke();
  EXPECT_NE(impl_code, "");
  EXPECT_NE(invoke_code, "");
}

TEST_F(TestAxesReorderSolverGen, TEST_GEN_SOLVER_case2)
{
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("block_dim");
  std::vector<Expr> cut_cons;
  std::vector<Expr> input_args;
  std::vector<Expr> search_args;
  input_args.emplace_back(x1);
  std::map<HardwareDef, Expr> hardware_cons;
  std::map<Expr, Expr, ExprCmp> arg_align_map;
  std::map<Expr, uint32_t, ExprCmp> const_args;
  std::map<Expr, uint32_t, ExprCmp> axis_priority;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  arg_align_map[x0] = ge::Symbol(16);
  cut_cons.emplace_back(x0 - x1);
  hardware_cons[HardwareDef::L2] = x0 + x1;
  from_axes_map[x0] = {x1};
  axis_priority[x0] = 2;
  axis_priority[x1] = 1;

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetTotalCutCons(cut_cons);
  solver_gen.SetBufferUseAlg(hardware_cons);
  solver_gen.SetConstArgs(const_args);
  solver_gen.SetInputArgs(input_args);
  solver_gen.SetSearchArgs(search_args);
  solver_gen.SetFromAxesMap(from_axes_map);
  solver_gen.SetVarPriority(axis_priority);
  solver_gen.Arrange();

  auto vars = solver_gen.local_buffer_tiling_vars_;

  std::string impl_code = solver_gen.GenSolverClassImpl();
  std::string invoke_code = solver_gen.GenSolverFuncInvoke();
  EXPECT_NE(impl_code, "");
  EXPECT_NE(invoke_code, "");
}

TEST_F(TestAxesReorderSolverGen, TEST_GEN_SOLVER_ub_cache_code)
{
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("block_dim");
  Expr x2 = CreateExpr("x2");
  std::vector<Expr> cut_cons;
  std::vector<Expr> input_args;
  std::vector<Expr> search_args;
  input_args.emplace_back(x1);
  search_args.emplace_back(x2);
  std::map<HardwareDef, Expr> hardware_cons;
  std::map<Expr, Expr, ExprCmp> arg_align_map;
  std::map<Expr, uint32_t, ExprCmp> const_args;
  std::map<Expr, uint32_t, ExprCmp> axis_priority;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  arg_align_map[x0] = ge::Symbol(16);
  cut_cons.emplace_back(x0 - x1);
  hardware_cons[HardwareDef::L2] = x2;
  hardware_cons[HardwareDef::CORENUM] = x0 + x1;
  from_axes_map[x0] = {x1};
  from_axes_map[x2] = {x0};
  axis_priority[x0] = 2;
  axis_priority[x1] = 1;
  std::vector<CacheLineConfig> config;
  config.push_back({"test_node", CreateExpr(120), 128});

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetTotalCutCons(cut_cons);
  solver_gen.SetBufferUseAlg(hardware_cons);
  solver_gen.SetConstArgs(const_args);
  solver_gen.SetInputArgs(input_args);
  solver_gen.SetSearchArgs(search_args);
  solver_gen.SetFromAxesMap(from_axes_map);
  solver_gen.SetVarPriority(axis_priority);
  solver_gen.Arrange();

  std::string impl_code = solver_gen.GenSolverClassImpl();
  std::string invoke_code = solver_gen.GenSolverFuncInvoke();
  EXPECT_NE(impl_code, "");
  EXPECT_NE(invoke_code, "");  
  EXPECT_FALSE(impl_code.find("// check node ") != std::string::npos);
  auto vars = solver_gen.local_buffer_tiling_vars_;

  solver_gen.SetCacheLineConfig(&config);
  solver_gen.Arrange();
  impl_code = solver_gen.GenSolverClassImpl();
  invoke_code = solver_gen.GenSolverFuncInvoke();
  EXPECT_NE(impl_code, "");
  EXPECT_NE(invoke_code, "");  
  EXPECT_TRUE(impl_code.find("// check node ") != std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenUpperBoundFunc_ConstExpr) {
    Expr var = CreateExpr("var");
    AxesReorderSolverGen solver_gen("case_test", "TilingData");
    solver_gen.from_axes_map_[var] = {CreateExpr(2), CreateExpr(3)};
    
    std::string result = solver_gen.GenUpperBoundFunc(var);
    
    std::string expected = R"(    GetUpperBoundFuncPtr var_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      upper_bound *= 2;
      upper_bound *= 3;
      return upper_bound;
    };
    var.upper_bound = var_upper_bound;
)";
    EXPECT_EQ(result, expected);
}

TEST_F(TestAxesReorderSolverGen, GenUpperBoundInfo_ConstExpr) {
    Expr var = CreateExpr("var");
    AxesReorderSolverGen solver("case_test", "TilingData");
    solver.from_axes_map_[var] = {CreateExpr(5)}; // Constant expression
    EXPECT_EQ(solver.GenUpperBoundInfo(var), "");
}

TEST_F(TestAxesReorderSolverGen, InitiateArgs_MixedValidInvalidArgs) {
    // Setup test data with mixed valid and invalid args
    AxesReorderSolverGen solver_("case_test", "TilingData");
    Expr valid = CreateExpr("valid");
    Expr invalid = CreateExpr("invalid");
    solver_.input_args_ = {valid, invalid};
    solver_.input_align_ = {{valid, ge::Symbol(4)}, {invalid, ge::Symbol(1)}}; // One needs alignment
    solver_.mc_args_ = {};
    solver_.local_buffer_tiling_vars_ = {};

    // Expected output
    std::string expected = 
        "    Variable valid;\n"
        "    valid.value = (tiling_data.get_valid() + std::max(1, 4) - 1) / std::max(1, 4) * std::max(1, 4);\n"
        "    Variable invalid;\n"
        "    invalid.value = tiling_data.get_invalid();\n";

    std::string result = solver_.InitiateArgs();
    EXPECT_EQ(result, expected);
}

TEST_F(TestAxesReorderSolverGen, NeitherInPriorityMap) {
    // Neither expression in priority map
    // Should use string comparison
    Expr expr1 = CreateExpr("var1");
    Expr expr2 = CreateExpr("var2");
    Expr expr3 = CreateExpr("var3");
    Expr expr4 = CreateExpr("var4");
    AxesReorderSolverGen solver("case_test", "TilingData");
    solver.VarCmp(expr1, expr2);
}

TEST_F(TestAxesReorderSolverGen, GenUpperBoundFunc_ConstantAxes) {
    // Test with constant axes
    Expr var = CreateExpr("test_var");
    AxesReorderSolverGen solver("case_test", "TilingData");
    solver.from_axes_map_[var] = {CreateExpr(2), CreateExpr(3), CreateExpr(5)};
    
    std::string expected = R"(    GetUpperBoundFuncPtr test_var_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      upper_bound *= 2;
      upper_bound *= 3;
      upper_bound *= 5;
      return upper_bound;
    };
    test_var.upper_bound = test_var_upper_bound;
)";
  EXPECT_EQ(solver.GenUpperBoundFunc(var), expected);
}

TEST_F(TestAxesReorderSolverGen, HandlesBothVars) {
    std::vector<Expr> tiling_vars = {CreateExpr("t_var1"), CreateExpr("t_var2")};
    std::vector<Expr> cons_vars = {CreateExpr("c_var1"), CreateExpr("c_var2")};
    AxesReorderSolverGen solver("case_test", "TilingData");
    Expr cons = CreateExpr("c_var1 * c_var2");
    std::string result = solver.GenConsFunc(0, ConsType::BUFFER, cons, tiling_vars, cons_vars);
    EXPECT_NE(result, "");
}

TEST_F(TestAxesReorderSolverGen, GenInputInfoTest) {
    std::vector<Expr> all_cons;
    std::vector<Expr> local_buffer_cons;
    std::vector<Expr> mc_mixed_cons;

    AxesReorderSolverGen solver("case_test", "TilingData");
    std::map<HardwareDef, Expr> hardware_use_map_;
    std::vector<Expr> total_cut_cons_;
    std::vector<Expr> mc_args_;
    std::vector<Expr> local_buffer_tiling_vars_;
    ExprExprMap arg_align_map_;
    ExprUintMap arg_prompt_align_map_;
    ExprUintMap is_concat_outer_map_;
    std::vector<Expr> concat_inner_dims_;

    // Initialize test data
    hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(1);
    hardware_use_map_[HardwareDef::GM] = CreateExpr(1024);
    hardware_use_map_[HardwareDef::UB] = CreateExpr(512);
    
    total_cut_cons_.push_back(CreateExpr(256));
    total_cut_cons_.push_back(CreateExpr(128));

    
    mc_args_.push_back(CreateExpr("mc_var1"));
    mc_args_.push_back(CreateExpr("mc_var2"));

    Expr var1 = CreateExpr("local_var1");
    Expr var2 = CreateExpr("local_var2");
    Expr var3 = ge::Symbol(255, "local_var3");

    
    local_buffer_tiling_vars_.push_back(var1);
    local_buffer_tiling_vars_.push_back(var2);
    
    arg_align_map_[var1] = ge::Symbol(64);
    arg_align_map_[var2] = ge::Symbol(32);
    
    arg_prompt_align_map_[var1] = 128;
    arg_prompt_align_map_[var3] = 64;
    
    is_concat_outer_map_[var1] = 1;
    is_concat_outer_map_[var2] = 0;
    
    concat_inner_dims_.push_back(var3);
    
    // Set the test data to the solver
    solver.hardware_use_map_ = hardware_use_map_;
    solver.total_cut_cons_ = total_cut_cons_;
    solver.mc_args_ = mc_args_;
    solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
    solver.arg_align_map_ = arg_align_map_;
    solver.arg_prompt_align_map_ = arg_prompt_align_map_;
    solver.is_concat_outer_map_ = is_concat_outer_map_;
    solver.concat_inner_dims_ = concat_inner_dims_;
    
    std::string result = solver.GenInputInfo(all_cons, local_buffer_cons, mc_mixed_cons);

    // 4. Check local buffer variables processing
    // Verify alignment strings are generated
    EXPECT_TRUE(result.find("local_var1.align = std::max(1, 64)") != std::string::npos);
    EXPECT_TRUE(result.find("local_var2.align = std::max(1, 32)") != std::string::npos);
    
    std::cout<<result<<std::endl;
    // Verify prompt alignment for concat outer case
    EXPECT_TRUE(result.find("local_var1.prompt_align = 128") != std::string::npos);
    
    // Verify no prompt alignment for non-concat outer case
    EXPECT_TRUE(result.find("local_var2.prompt_align") == std::string::npos);
}


TEST_F(TestAxesReorderSolverGen, GenInputInfoCase2Test) {
    std::vector<Expr> all_cons;
    std::vector<Expr> local_buffer_cons;
    std::vector<Expr> mc_mixed_cons;

    AxesReorderSolverGen solver("case_test", "TilingData");
    std::map<HardwareDef, Expr> hardware_use_map_;
    std::vector<Expr> total_cut_cons_;
    std::vector<Expr> mc_args_;
    std::vector<Expr> local_buffer_tiling_vars_;
    ExprExprMap arg_align_map_;
    ExprUintMap arg_prompt_align_map_;
    ExprUintMap is_concat_outer_map_;
    std::vector<Expr> concat_inner_dims_;

    // Initialize test data
    hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(1);
    hardware_use_map_[HardwareDef::GM] = CreateExpr(1024);
    hardware_use_map_[HardwareDef::UB] = CreateExpr(512);
    
    total_cut_cons_.push_back(CreateExpr(256));
    total_cut_cons_.push_back(CreateExpr(128));

    
    mc_args_.push_back(CreateExpr("mc_var1"));
    mc_args_.push_back(CreateExpr("mc_var2"));

    Expr var1 = CreateExpr("local_var1");
    Expr var2 = CreateExpr("local_var2");
    Expr var3 = ge::Symbol("local_var3");

    
    local_buffer_tiling_vars_.push_back(var1);
    local_buffer_tiling_vars_.push_back(var2);
    
    arg_align_map_[var1] = ge::Symbol(64);
    arg_align_map_[var2] = ge::Symbol(32);
    
    arg_prompt_align_map_[var1] = 128;
    arg_prompt_align_map_[var3] = 64;
    
    is_concat_outer_map_[var1] = 1;
    is_concat_outer_map_[var2] = 0;
    
    concat_inner_dims_.push_back(var3);
    
    // Set the test data to the solver
    solver.hardware_use_map_ = hardware_use_map_;
    solver.total_cut_cons_ = total_cut_cons_;
    solver.mc_args_ = mc_args_;
    solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
    solver.arg_align_map_ = arg_align_map_;
    solver.arg_prompt_align_map_ = arg_prompt_align_map_;
    solver.is_concat_outer_map_ = is_concat_outer_map_;
    solver.concat_inner_dims_ = concat_inner_dims_;
    
    std::string result = solver.GenInputInfo(all_cons, local_buffer_cons, mc_mixed_cons);

    // 4. Check local buffer variables processing
    // Verify alignment strings are generated
    EXPECT_TRUE(result.find("local_var1.align = std::max(1, 64)") != std::string::npos);
    EXPECT_TRUE(result.find("local_var2.align = std::max(1, 32)") != std::string::npos);
    
    std::cout<<result<<std::endl;
    // Verify prompt alignment for concat outer case
    EXPECT_TRUE(result.find("local_var1.prompt_align = 128") != std::string::npos);
    
    // Verify no prompt alignment for non-concat outer case
    EXPECT_TRUE(result.find("local_var2.prompt_align") == std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenUBThresholdFunc_WithUBAndVariables) {
  // Case 3: Normal case with UB and variables
  AxesReorderSolverGen solver("case_test", "TilingData");
  std::vector<Expr> mc_args_;
  std::vector<Expr> local_buffer_tiling_vars_;
  std::map<HardwareDef, Expr> hardware_use_map_;
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr var3 = CreateExpr("var3");
  Expr var4 = CreateExpr("var4");

  mc_args_.push_back(var1);
  mc_args_.push_back(var2);

  local_buffer_tiling_vars_.push_back(var3);
  local_buffer_tiling_vars_.push_back(var4);

  hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(100) / var1 * var3;
  hardware_use_map_[HardwareDef::UB] = var3 * var4;

  solver.mc_args_ = mc_args_;
  solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
  solver.hardware_use_map_ = hardware_use_map_;
  solver.enable_multicore_ub_tradeoff_ = true;
  std::string actual = solver.GenUBThresholdFunc();
  EXPECT_TRUE(actual.find("return (ub_size - 0) > static_cast<uint32_t>(input_.ub_threshold * input_.ub_size);") != std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenUBThresholdFunc_WithUBAndVariablesNotFound) {
  // Case 3: Normal case with UB and variables
  AxesReorderSolverGen solver("case_test", "TilingData");
  std::vector<Expr> mc_args_;
  std::vector<Expr> local_buffer_tiling_vars_;
  std::map<HardwareDef, Expr> hardware_use_map_;
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr var3 = CreateExpr("var3");
  Expr var4 = CreateExpr("var4");

  mc_args_.push_back(var1);
  mc_args_.push_back(var2);

  local_buffer_tiling_vars_.push_back(var3);
  local_buffer_tiling_vars_.push_back(var4);

  hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(100) / var1 * var3;

  solver.input_args_ = {var1, var2, var3, var4};
  solver.mc_args_ = mc_args_;
  solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
  solver.hardware_use_map_ = hardware_use_map_;
  solver.enable_multicore_ub_tradeoff_ = true;
  std::string actual = solver.GenUBThresholdFunc();
  EXPECT_TRUE(actual.find("return ub_size > static_cast<uint32_t>(input_.ub_threshold * input_.ub_size);") == std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenSolverFuncImpl_WithUBAndVariables) {
  // Case 3: Normal case with UB and variables
  AxesReorderSolverGen solver("case_test", "TilingData");
  std::vector<Expr> mc_args_;
  std::vector<Expr> local_buffer_tiling_vars_;
  std::map<HardwareDef, Expr> hardware_use_map_;
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr var3 = CreateExpr("var3");
  Expr var4 = CreateExpr("var4");

  mc_args_.push_back(var1);
  mc_args_.push_back(var2);

  local_buffer_tiling_vars_.push_back(var3);
  local_buffer_tiling_vars_.push_back(var4);

  hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(100) / var1 * var3;
  hardware_use_map_[HardwareDef::UB] = var3 * var4;

  solver.mc_args_ = mc_args_;
  solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
  solver.hardware_use_map_ = hardware_use_map_;
  solver.enable_multicore_ub_tradeoff_ = true;
  std::string actual = solver.GenSolverFuncImpl();
  EXPECT_TRUE(actual.find("solver.Run(true, ") != std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, test_not_contain_heavy_op) {
  // Case 3: Normal case with UB and variables
  AxesReorderSolverGen solver("case_test", "TilingData");
  std::vector<Expr> mc_args_;
  std::vector<Expr> local_buffer_tiling_vars_;
  std::map<HardwareDef, Expr> hardware_use_map_;
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr var3 = CreateExpr("var3");
  Expr var4 = CreateExpr("var4");

  mc_args_.push_back(var1);
  mc_args_.push_back(var2);

  local_buffer_tiling_vars_.push_back(var3);
  local_buffer_tiling_vars_.push_back(var4);

  hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(100) / var1 * var3;
  hardware_use_map_[HardwareDef::UB] = var3 * var4;

  TilingScheduleConfigTableV1 tiling_schedule_config_table;
  solver.mc_args_ = mc_args_;
  solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
  solver.hardware_use_map_ = hardware_use_map_;
  solver.tiling_schedule_config_table_ = &tiling_schedule_config_table;
  std::string actual = solver.GenSolverFuncImpl();
  EXPECT_TRUE(actual.find("solver.Run(false, ") != std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenPgo_SolverwithClassImpl) {
  std::string className = "AxesReorderSolverGen";
  AxesReorderSolverGen solver("GenPgo_test", "TilingData");
  solver.enable_autofuse_pgo_ = true;
  solver.GenSolverClassImpl();
  solver.GenSolverFuncImpl();
}

/**
 * @brief 测试：当 enable_multicore_ub_tradeoff 为 true 时，GenInput 使用成员变量而非硬编码常量
 *
 * 原问题背景： GenInput() 函数中
 * 使用硬编码常量 kDefaultSolverUbThreshold(0.2) 和 kDefaultSolverCoreNumThreshold(0.4)
 * 而非成员变量 ub_threshold_ 和 corenum_threshold_。
 *
 * 本测试用例用于看护该修复，防止未来代码回退。
 */
TEST_F(TestAxesReorderSolverGen, GenInput_UseMemberVariableWhenTradeOffEnabled) {
  AxesReorderSolverGen solver("case_test", "TilingData");

  // 设置非默认阈值 (0.0 和 1.0 是边界值，常用于环境变量配置)
  solver.SetUBThreshold(0.0);
  solver.SetCoreNumThreshold(1.0);
  solver.SetEnableMulticoreUBTradeoff(true);
  solver.SetTilingCaseIdent({{0, 0, 0}, 1, ""});

  std::vector<Expr> all_cons;
  TradeOffConfig trade_off_config;
  trade_off_config.is_enable = false;

  std::string gen_code = solver.GenInput(trade_off_config, all_cons);

  // 验证生成的代码使用成员变量值
  // 注意: std::to_string() 默认生成6位小数格式
  EXPECT_TRUE(gen_code.find("input.ub_threshold = 0.000000;") != std::string::npos)
      << "GenInput should use ub_threshold_ member variable (0.0), not hardcoded default.\n"
      << "Generated code:\n" << gen_code;

  EXPECT_TRUE(gen_code.find("input.corenum_threshold = 1.000000;") != std::string::npos)
      << "GenInput should use corenum_threshold_ member variable (1.0), not hardcoded default.\n"
      << "Generated code:\n" << gen_code;

  // 防止回退检查：确保代码中不包含硬编码的默认值 0.200000 和 0.400000
  EXPECT_EQ(gen_code.find("0.200000"), std::string::npos)
      << "Generated code contains hardcoded default ub_threshold 0.2!\n"
      << "This indicates the fix from commit 3d7b57d5 may have regressed.\n"
      << "Generated code:\n" << gen_code;

  EXPECT_EQ(gen_code.find("0.400000"), std::string::npos)
      << "Generated code contains hardcoded default corenum_threshold 0.4!\n"
      << "This indicates the fix from commit 3d7b57d5 may have regressed.\n"
      << "Generated code:\n" << gen_code;
}

/**
 * @brief 测试：验证 enable_multicore_ub_tradeoff 为 false 时使用默认值
 */
TEST_F(TestAxesReorderSolverGen, GenInput_UseDefaultWhenTradeOffDisabled) {
  AxesReorderSolverGen solver("case_test", "TilingData");

  // 即使设置了非默认值，关闭开关后应使用默认值
  solver.SetUBThreshold(0.0);
  solver.SetCoreNumThreshold(1.0);
  solver.SetEnableMulticoreUBTradeoff(false);
  solver.SetTilingCaseIdent({{0, 0, 0}, 1, ""});

  std::vector<Expr> all_cons;
  TradeOffConfig trade_off_config;
  trade_off_config.is_enable = false;

  std::string gen_code = solver.GenInput(trade_off_config, all_cons);

  // 应该使用默认值 0.2 和 0.4 (std::to_string 生成6位小数)
  EXPECT_TRUE(gen_code.find("input.ub_threshold = 0.200000;") != std::string::npos);
  EXPECT_TRUE(gen_code.find("input.corenum_threshold = 0.400000;") != std::string::npos);
}

/**
 * @brief 测试：验证 trade_off_config 优先级低于 enable_multicore_ub_tradeoff
 */
TEST_F(TestAxesReorderSolverGen, GenInput_TradeOffConfigPriority) {
  AxesReorderSolverGen solver("case_test", "TilingData");

  solver.SetUBThreshold(0.0);
  solver.SetCoreNumThreshold(1.0);
  solver.SetEnableMulticoreUBTradeoff(true);
  solver.SetTilingCaseIdent({{0, 0, 0}, 1, ""});

  std::vector<Expr> all_cons;
  TradeOffConfig trade_off_config;
  trade_off_config.is_enable = true;
  trade_off_config.ub_ratio = CreateExpr(0.3);
  trade_off_config.core_num_ratio = CreateExpr(0.5);

  std::string gen_code = solver.GenInput(trade_off_config, all_cons);

  // enable_multicore_ub_tradeoff 优先级更高，应该使用成员变量值
  EXPECT_TRUE(gen_code.find("input.ub_threshold = 0.000000;") != std::string::npos);
  EXPECT_TRUE(gen_code.find("input.corenum_threshold = 1.000000;") != std::string::npos);
}

/**
 * @brief 测试：验证 trade_off_config 在 enable_multicore_ub_tradeoff 为 false 时生效
 */
TEST_F(TestAxesReorderSolverGen, GenInput_UseTradeOffConfigWhenMemberDisabled) {
  AxesReorderSolverGen solver("case_test", "TilingData");

  solver.SetUBThreshold(0.0);
  solver.SetCoreNumThreshold(1.0);
  solver.SetEnableMulticoreUBTradeoff(false);
  solver.SetTilingCaseIdent({{0, 0, 0}, 1, ""});

  std::vector<Expr> all_cons;
  TradeOffConfig trade_off_config;
  trade_off_config.is_enable = true;
  trade_off_config.ub_ratio = CreateExpr(0.3);
  trade_off_config.core_num_ratio = CreateExpr(0.5);

  std::string gen_code = solver.GenInput(trade_off_config, all_cons);

  // 应该使用 trade_off_config 中的值
  EXPECT_TRUE(gen_code.find("input.ub_threshold = 0.3;") != std::string::npos);
  EXPECT_TRUE(gen_code.find("input.corenum_threshold = 0.5;") != std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenSolverFuncImpl_DisableGroupParallel_EnableTradeOff) {
  // 验证当 enable_group_parallel_ 为 false 时，
  // 如果 trade_off_config.is_enable 为 true，multicore_ub_tradeoff 应该是 "true"
  AxesReorderSolverGen solver("case_test", "TilingData");
  std::vector<Expr> mc_args_;
  std::vector<Expr> local_buffer_tiling_vars_;
  std::map<HardwareDef, Expr> hardware_use_map_;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr var3 = CreateExpr("var3");
  Expr var4 = CreateExpr("var4");

  mc_args_.push_back(var1);
  mc_args_.push_back(var2);
  local_buffer_tiling_vars_.push_back(var3);
  local_buffer_tiling_vars_.push_back(var4);
  from_axes_map[var3] = {var1};

  hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(100) / var1 * var3;
  hardware_use_map_[HardwareDef::UB] = var3 * var4;

  TilingScheduleConfig tiling_schedule_config;
  tiling_schedule_config.trade_off_config.is_enable = true;
  tiling_schedule_config.trade_off_config.ub_ratio = ge::Symbol(0.1);
  tiling_schedule_config.trade_off_config.core_num_ratio = ge::Symbol(0.8);

  solver.mc_args_ = mc_args_;
  solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
  solver.hardware_use_map_ = hardware_use_map_;
  solver.from_axes_map_ = from_axes_map;
  solver.SetTilingScheduleConfig(tiling_schedule_config);
  solver.SetEnableParallel(false);  // enable_group_parallel_ = false

  std::string actual = solver.GenSolverFuncImpl();
  // 验证生成的代码中 multicore_ub_tradeoff 为 "true"
  EXPECT_TRUE(actual.find("solver.Run(true, ") != std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenGetBlockDimStatic_ShouldClampToOneWhenCoreNumExprIsZero) {
  AxesReorderSolverGen solver("case_test", "TilingData");

  Expr corenum_cons = CreateExpr(0);
  std::string actual = solver.GenGetBlockDimStatic(corenum_cons);

  EXPECT_TRUE(actual.find("return 0;") == std::string::npos) << actual;
  EXPECT_TRUE(actual.find("return std::max(1, static_cast<int32_t>(0));") != std::string::npos) << actual;
}

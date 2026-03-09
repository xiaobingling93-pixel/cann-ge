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
#include "base/att_const_values.h"
#include "gen_model_info.h"
#include "ascir_ops.h"
#include "tiling_code_generator.h"
#include "api_tiling_gen/gen_api_tiling.h"
#include "autofuse_config/auto_fuse_config.h"
#include "gen_tiling_impl.h"
#include "graph_construct_utils.h"
#include "result_checker_utils.h"
#include "common/test_common_utils.h"
#include "test_common_utils.h"

using namespace ge::ascir_op;
namespace ascir {
constexpr int64_t ID_NONE = -1; //取多少？
using namespace ge;
using HintGraph=AscGraph;
}
namespace {
bool IsFileContainsString(const std::string& filename, const std::string& searchString) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "无法打开文件: " << filename << std::endl;
    return false;
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.find(searchString) != std::string::npos) {
      file.close();
      return true;
    }
  }
  file.close();
  return false;
}
}
using namespace att;

class TestGenAddLayerNormalModelInfo : public ::testing::Test {
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
//    dlog_setlevel(GE, 0, 1);
    att::AutoFuseConfig::MutableAttStrategyConfig().Reset();
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
  }

  void TearDown() override {
    // Code here will be called immediately after each test (right
    // before the destructor).
    // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
  }
};

void Add_Layer_Norm_Normal_BeforeAutofuseConstInput(ascir::HintGraph &graph) {
  auto ONE = ge::sym::kSymbolOne;
  auto ZERO = ge::sym::kSymbolZero;
  // 定义轴的大小
  auto A = ge::Symbol(128, "A");
  auto R = ge::Symbol("R");
  auto BL = ge::Symbol(8, "BL");

  // 定义轴
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  // 定义节点
  int exec_order = 0;
  Data x1("x1", graph);
  x1.attr.sched.exec_order = exec_order++;
  x1.attr.sched.axis = {a.id, r.id, bl.id};
  x1.y.dtype = ge::DT_FLOAT16;
  *x1.y.axis = {a.id, r.id, bl.id};
  *x1.y.repeats = {A, R, ONE};
  *x1.y.strides = {R, ONE, ZERO};

  Load x1Local("x1Local");
  x1Local.x = x1.y;
  x1Local.attr.sched.exec_order = exec_order++;
  x1Local.attr.sched.axis = {a.id, r.id, bl.id};
  x1Local.y.dtype = ge::DT_FLOAT16;
  *x1Local.y.axis = {a.id, r.id, bl.id};
  *x1Local.y.repeats = {A, R, ONE};
  *x1Local.y.strides = {R, ONE, ZERO};

  Data x2("x2", graph);
  x2.attr.sched.exec_order = exec_order++;
  x2.attr.sched.axis = {a.id, r.id, bl.id};
  x2.y.dtype = ge::DT_FLOAT16;
  *x2.y.axis = {a.id, r.id, bl.id};
  *x2.y.repeats = {A, R, ONE};
  *x2.y.strides = {R, ONE, ZERO};

  Load x2Local("x2Local");
  x2Local.x = x2.y;
  x2Local.attr.sched.exec_order = exec_order++;
  x2Local.attr.sched.axis = {a.id, r.id, bl.id};
  x2Local.y.dtype = ge::DT_FLOAT16;
  *x2Local.y.axis = {a.id, r.id, bl.id};
  *x2Local.y.repeats = {A, R, ONE};
  *x2Local.y.strides = {R, ONE, ZERO};

  Data bias("bias", graph);
  bias.attr.sched.exec_order = exec_order++;
  bias.attr.sched.axis = {a.id, r.id, bl.id};
  bias.y.dtype = ge::DT_FLOAT16;
  *bias.y.axis = {a.id, r.id, bl.id};
  *bias.y.repeats = {A, R, ONE};
  *bias.y.strides = {R, ONE, ZERO};

  Load biasLocal("biasLocal");
  biasLocal.x = bias.y;
  biasLocal.attr.sched.exec_order = exec_order++;
  biasLocal.attr.sched.axis = {a.id, r.id, bl.id};
  biasLocal.y.dtype = ge::DT_FLOAT16;
  *biasLocal.y.axis = {a.id, r.id, bl.id};
  *biasLocal.y.repeats = {A, R, ONE};
  *biasLocal.y.strides = {R, ONE, ZERO};

  Concat mean("mean");
  mean.x = {x1Local.y, x2Local.y, biasLocal.y};
  mean.attr.sched.exec_order = exec_order++;
  mean.attr.sched.axis = {a.id, r.id, bl.id};
  mean.y.dtype = ge::DT_FLOAT;        // x fp32
  *mean.y.axis = {a.id, r.id, bl.id};
  *mean.y.repeats = {A, R, ONE};
  *mean.y.strides = {R, ONE, ZERO};

  Store x_out("x_out");
  x_out.attr.sched.exec_order = exec_order++;
  x_out.attr.sched.axis = {a.id, r.id, bl.id};
  x_out.x = mean.y;
  x_out.y.dtype = ge::DT_FLOAT16;
  *x_out.y.axis = {a.id, r.id, bl.id};
  *x_out.y.repeats = {A, R, ONE};
  *x_out.y.strides = {R, ONE, ZERO};

  Store mean_out("mean_out");
  mean_out.attr.sched.exec_order = exec_order++;
  mean_out.attr.sched.axis = {a.id, r.id, bl.id};
  mean_out.x = mean.y;
  mean_out.y.dtype = ge::DT_FLOAT;
  *mean_out.y.axis = {a.id, r.id, bl.id};
  *mean_out.y.repeats = {A, ONE, ONE};
  *mean_out.y.strides = {ONE, ZERO, ZERO};

  Data one("one", graph);
  one.attr.sched.exec_order = exec_order++;
  one.attr.sched.axis = {a.id, r.id, bl.id};
  one.y.dtype = ge::DT_FLOAT;
  *one.y.axis = {a.id, r.id, bl.id};
  *one.y.repeats = {ONE, ONE, BL};
  *one.y.strides = {ZERO, ZERO, ONE};

  Concat rstd("rstd");
  rstd.attr.sched.exec_order = exec_order++;
  rstd.attr.sched.axis = {a.id, r.id, bl.id};
  rstd.x = {mean.y, mean.y, one.y};
  rstd.y.dtype = ge::DT_FLOAT;      // x-mean
  *rstd.y.axis = {a.id, r.id, bl.id};
  *rstd.y.repeats = {A, R, ONE};
  *rstd.y.strides = {R, ONE, ZERO};

  Store rstd_out("rstd_out");
  rstd_out.attr.sched.exec_order = exec_order++;
  rstd_out.attr.sched.axis = {a.id, r.id, bl.id};
  rstd_out.x = rstd.y;
  rstd_out.y.dtype = ge::DT_FLOAT;
  *rstd_out.y.axis = {a.id, r.id, bl.id};
  *rstd_out.y.repeats = {A, ONE, ONE};
  *rstd_out.y.strides = {ONE, ZERO, ZERO};

  Data beta("beta", graph);
  beta.attr.sched.exec_order = exec_order++;
  beta.attr.sched.axis = {a.id, r.id, bl.id};
  beta.y.dtype = ge::DT_FLOAT16;
  *beta.y.axis = {a.id, r.id, bl.id};
  *beta.y.repeats = {ONE, R, ONE};
  *beta.y.strides = {ZERO, ONE, ZERO};

  Load betaLocal("betaLocal");
  betaLocal.x = beta.y;
  betaLocal.attr.sched.exec_order = exec_order++;
  betaLocal.attr.sched.axis = {a.id, r.id, bl.id};
  betaLocal.y.dtype = ge::DT_FLOAT16;
  *betaLocal.y.axis = {a.id, r.id, bl.id};
  *betaLocal.y.repeats = {ONE, R, ONE};
  *betaLocal.y.strides = {ZERO, ONE, ZERO};

  Data gamma("gamma", graph);
  gamma.attr.sched.exec_order = exec_order++;
  gamma.attr.sched.axis = {a.id, r.id, bl.id};
  gamma.y.dtype = ge::DT_FLOAT16;
  *gamma.y.axis = {a.id, r.id, bl.id};
  *gamma.y.repeats = {ONE, R, ONE};
  *gamma.y.strides = {ZERO, ONE, ZERO};

  Load gammaLocal("gammaLocal");
  gammaLocal.x = gamma.y;
  gammaLocal.attr.sched.exec_order = exec_order++;
  gammaLocal.attr.sched.axis = {a.id, r.id, bl.id};
  gammaLocal.y.dtype = ge::DT_FLOAT16;
  *gammaLocal.y.axis = {a.id, r.id, bl.id};
  *gammaLocal.y.repeats = {ONE, R, ONE};
  *gammaLocal.y.strides = {ZERO, ONE, ZERO};

  Concat y("y");
  y.attr.api.unit = ge::ComputeUnit::kUnitVector;
  y.attr.sched.exec_order = exec_order++;
  y.attr.sched.axis = {a.id, r.id, bl.id};
  y.x = {rstd.y, betaLocal.y, gammaLocal.y, rstd.y};                 // x-mean
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {a.id, r.id, bl.id};
  *y.y.repeats = {A, R, ONE};
  *y.y.strides = {R, ONE, ZERO};

  Store y_out("y_out");
  y_out.attr.sched.exec_order = exec_order++;
  y_out.attr.sched.axis = {a.id, r.id, bl.id};
  y_out.x = y.y;
  y_out.y.dtype = ge::DT_FLOAT16;
  *y_out.y.axis = {a.id, r.id, bl.id};
  *y_out.y.repeats = {A, R, ONE};
  *y_out.y.strides = {R, ONE, ZERO};

  Output buf1("buf1");
  buf1.x = x_out.y;
  buf1.attr.sched.exec_order = exec_order++;
  buf1.y.dtype = ge::DT_FLOAT16;
  *buf1.y.axis = {a.id, r.id, bl.id};
  *buf1.y.repeats = {A, R, ONE};
  *buf1.y.strides = {R, ONE, ZERO};

  Output buf2("buf2");
  buf2.x = mean_out.y;
  buf2.attr.sched.exec_order = exec_order++;
  buf2.y.dtype = ge::DT_FLOAT;
  *buf2.y.axis = {a.id, r.id, bl.id};
  *buf2.y.repeats = {A, ONE, ONE};
  *buf2.y.strides = {ONE, ZERO, ZERO};

  Output buf3("buf3");
  buf3.x = rstd_out.y;
  buf3.attr.sched.exec_order = exec_order++;
  buf3.y.dtype = ge::DT_FLOAT;
  *buf3.y.axis = {a.id, r.id, bl.id};
  *buf3.y.repeats = {A, ONE, ONE};
  *buf3.y.strides = {ONE, ZERO, ZERO};

  Output buf("buf");
  buf.x = y_out.y;
  buf.attr.sched.exec_order = exec_order++;
  buf.y.dtype = ge::DT_FLOAT16;
  *buf.y.axis = {a.id, r.id, bl.id};
  *buf.y.repeats = {A, R, ONE};
  *buf.y.strides = {R, ONE, ZERO};
}


void Add_Layer_Norm_Normal_BeforeAutofuse(ascir::HintGraph &graph, const std::string &ident = "") {
  auto ONE = ge::sym::kSymbolOne;
  auto ZERO = ge::sym::kSymbolZero;
  // 定义轴的大小
  std::string axis_name1("A");
  axis_name1.append(ident);
  std::string axis_name2("R");
  axis_name2.append(ident);
  std::string axis_name3("BL");
  axis_name3.append(ident);
  auto A = ge::Symbol(axis_name1.c_str());
  auto R = ge::Symbol(axis_name2.c_str());
  auto BL = ge::Symbol(8, axis_name3.c_str());

  // 定义轴
  auto a = graph.CreateAxis(axis_name1, A);
  auto r = graph.CreateAxis(axis_name2, R);
  auto bl = graph.CreateAxis(axis_name3, BL);

  // 定义节点
  int exec_order = 0;
  Data x1("x1", graph);
  x1.attr.sched.exec_order = exec_order++;
  x1.attr.sched.axis = {a.id, r.id, bl.id};
  x1.y.dtype = ge::DT_FLOAT16;
  *x1.y.axis = {a.id, r.id, bl.id};
  *x1.y.repeats = {A, R, ONE};
  *x1.y.strides = {R, ONE, ZERO};

  Load x1Local("x1Local");
  x1Local.x = x1.y;
  x1Local.attr.sched.exec_order = exec_order++;
  x1Local.attr.sched.axis = {a.id, r.id, bl.id};
  x1Local.y.dtype = ge::DT_FLOAT16;
  *x1Local.y.axis = {a.id, r.id, bl.id};
  *x1Local.y.repeats = {A, R, ONE};
  *x1Local.y.strides = {R, ONE, ZERO};

  Data x2("x2", graph);
  x2.attr.sched.exec_order = exec_order++;
  x2.attr.sched.axis = {a.id, r.id, bl.id};
  x2.y.dtype = ge::DT_FLOAT16;
  *x2.y.axis = {a.id, r.id, bl.id};
  *x2.y.repeats = {A, R, ONE};
  *x2.y.strides = {R, ONE, ZERO};

  Load x2Local("x2Local");
  x2Local.x = x2.y;
  x2Local.attr.sched.exec_order = exec_order++;
  x2Local.attr.sched.axis = {a.id, r.id, bl.id};
  x2Local.y.dtype = ge::DT_FLOAT16;
  *x2Local.y.axis = {a.id, r.id, bl.id};
  *x2Local.y.repeats = {A, R, ONE};
  *x2Local.y.strides = {R, ONE, ZERO};

  Data bias("bias", graph);
  bias.attr.sched.exec_order = exec_order++;
  bias.attr.sched.axis = {a.id, r.id, bl.id};
  bias.y.dtype = ge::DT_FLOAT16;
  *bias.y.axis = {a.id, r.id, bl.id};
  *bias.y.repeats = {A, R, ONE};
  *bias.y.strides = {R, ONE, ZERO};

  Load biasLocal("biasLocal");
  biasLocal.x = bias.y;
  biasLocal.attr.sched.exec_order = exec_order++;
  biasLocal.attr.sched.axis = {a.id, r.id, bl.id};
  biasLocal.y.dtype = ge::DT_FLOAT16;
  *biasLocal.y.axis = {a.id, r.id, bl.id};
  *biasLocal.y.repeats = {A, R, ONE};
  *biasLocal.y.strides = {R, ONE, ZERO};

  Concat mean("mean");
  mean.x = {x1Local.y, x2Local.y, biasLocal.y};
  mean.attr.sched.exec_order = exec_order++;
  mean.attr.sched.axis = {a.id, r.id, bl.id};
  mean.y.dtype = ge::DT_FLOAT;        // mean
  *mean.y.axis = {a.id, r.id, bl.id};
  *mean.y.repeats = {A, ONE, ONE};
  *mean.y.strides = {ONE, ZERO, ZERO};

  Store x_out("x_out");
  x_out.attr.sched.exec_order = exec_order++;
  x_out.attr.sched.axis = {a.id, r.id, bl.id};
  x_out.x = mean.y;
  x_out.y.dtype = ge::DT_FLOAT16;
  *x_out.y.axis = {a.id, r.id, bl.id};
  *x_out.y.repeats = {A, R, ONE};
  *x_out.y.strides = {R, ONE, ZERO};

  Store mean_out("mean_out");
  mean_out.attr.sched.exec_order = exec_order++;
  mean_out.attr.sched.axis = {a.id, r.id, bl.id};
  mean_out.x = mean.y;
  mean_out.y.dtype = ge::DT_FLOAT;
  *mean_out.y.axis = {a.id, r.id, bl.id};
  *mean_out.y.repeats = {A, ONE, ONE};
  *mean_out.y.strides = {ONE, ZERO, ZERO};

  Data one("one", graph);
  one.attr.sched.exec_order = exec_order++;
  one.attr.sched.axis = {a.id, r.id, bl.id};
  one.y.dtype = ge::DT_FLOAT;
  *one.y.axis = {a.id, r.id, bl.id};
  *one.y.repeats = {ONE, ONE, BL};
  *one.y.strides = {ZERO, ZERO, ONE};

  Concat rstd("rstd");
  rstd.attr.sched.exec_order = exec_order++;
  rstd.attr.sched.axis = {a.id, r.id, bl.id};
  rstd.x = {mean.y, mean.y, one.y};
  rstd.y.dtype = ge::DT_FLOAT;      // x-mean
  *rstd.y.axis = {a.id, r.id, bl.id};
  *rstd.y.repeats = {A, R, ONE};
  *rstd.y.strides = {R, ONE, ZERO};

  Store rstd_out("rstd_out");
  rstd_out.attr.sched.exec_order = exec_order++;
  rstd_out.attr.sched.axis = {a.id, r.id, bl.id};
  rstd_out.x = rstd.y;
  rstd_out.y.dtype = ge::DT_FLOAT;
  *rstd_out.y.axis = {a.id, r.id, bl.id};
  *rstd_out.y.repeats = {A, ONE, ONE};
  *rstd_out.y.strides = {ONE, ZERO, ZERO};

  Data beta("beta", graph);
  beta.attr.sched.exec_order = exec_order++;
  beta.attr.sched.axis = {a.id, r.id, bl.id};
  beta.y.dtype = ge::DT_FLOAT16;
  *beta.y.axis = {a.id, r.id, bl.id};
  *beta.y.repeats = {ONE, R, ONE};
  *beta.y.strides = {ZERO, ONE, ZERO};

  Load betaLocal("betaLocal");
  betaLocal.x = beta.y;
  betaLocal.attr.sched.exec_order = exec_order++;
  betaLocal.attr.sched.axis = {a.id, r.id, bl.id};
  betaLocal.y.dtype = ge::DT_FLOAT16;
  *betaLocal.y.axis = {a.id, r.id, bl.id};
  *betaLocal.y.repeats = {ONE, R, ONE};
  *betaLocal.y.strides = {ZERO, ONE, ZERO};

  Data gamma("gamma", graph);
  gamma.attr.sched.exec_order = exec_order++;
  gamma.attr.sched.axis = {a.id, r.id, bl.id};
  gamma.y.dtype = ge::DT_FLOAT16;
  *gamma.y.axis = {a.id, r.id, bl.id};
  *gamma.y.repeats = {ONE, R, ONE};
  *gamma.y.strides = {ZERO, ONE, ZERO};

  Load gammaLocal("gammaLocal");
  gammaLocal.x = gamma.y;
  gammaLocal.attr.sched.exec_order = exec_order++;
  gammaLocal.attr.sched.axis = {a.id, r.id, bl.id};
  gammaLocal.y.dtype = ge::DT_FLOAT16;
  *gammaLocal.y.axis = {a.id, r.id, bl.id};
  *gammaLocal.y.repeats = {ONE, R, ONE};
  *gammaLocal.y.strides = {ZERO, ONE, ZERO};

  Concat y("y");
  y.attr.api.unit = ge::ComputeUnit::kUnitVector;
  y.attr.sched.exec_order = exec_order++;
  y.attr.sched.axis = {a.id, r.id, bl.id};
  y.x = {rstd.y, betaLocal.y, gammaLocal.y, rstd.y};                 // x-mean, beta, gamma, rstd
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {a.id, r.id, bl.id};
  *y.y.repeats = {A, R, ONE};
  *y.y.strides = {R, ONE, ZERO};

  Store y_out("y_out");
  y_out.attr.sched.exec_order = exec_order++;
  y_out.attr.sched.axis = {a.id, r.id, bl.id};
  y_out.x = y.y;
  y_out.y.dtype = ge::DT_FLOAT16;
  *y_out.y.axis = {a.id, r.id, bl.id};
  *y_out.y.repeats = {A, R, ONE};
  *y_out.y.strides = {R, ONE, ZERO};

  Output buf1("buf1");
  buf1.x = x_out.y;
  buf1.attr.sched.exec_order = exec_order++;
  buf1.y.dtype = ge::DT_FLOAT16;
  *buf1.y.axis = {a.id, r.id, bl.id};
  *buf1.y.repeats = {A, R, ONE};
  *buf1.y.strides = {R, ONE, ZERO};

  Output buf2("buf2");
  buf2.x = mean_out.y;
  buf2.attr.sched.exec_order = exec_order++;
  buf2.y.dtype = ge::DT_FLOAT;
  *buf2.y.axis = {a.id, r.id, bl.id};
  *buf2.y.repeats = {A, ONE, ONE};
  *buf2.y.strides = {ONE, ZERO, ZERO};

  Output buf3("buf3");
  buf3.x = rstd_out.y;
  buf3.attr.sched.exec_order = exec_order++;
  buf3.y.dtype = ge::DT_FLOAT;
  *buf3.y.axis = {a.id, r.id, bl.id};
  *buf3.y.repeats = {A, ONE, ONE};
  *buf3.y.strides = {ONE, ZERO, ZERO};

  Output buf("buf");
  buf.x = y_out.y;
  buf.attr.sched.exec_order = exec_order++;
  buf.y.dtype = ge::DT_FLOAT16;
  *buf.y.axis = {a.id, r.id, bl.id};
  *buf.y.repeats = {A, R, ONE};
  *buf.y.strides = {R, ONE, ZERO};
}

/*
for aBO
  for aBIO
    for aBII
      for r
        load x1
        load x2
        load bias
        CalcMean
        CalcRstd
        Store X
        Store mean
        Load beta
        Load gamma
        CalcRstd
        Store rstd
        CalcY
        Store y
*/

void Add_Layer_Norm_Normal_AfterScheduler(ascir::HintGraph &graph, const std::string &ident = "") {
  auto a = graph.FindAxis(0)->id;
  auto r = graph.FindAxis(1)->id;
  auto bl = graph.FindAxis(2)->id;

  auto [aBO, aBI] = graph.BlockSplit(a, "nbi" + ident, "nbo" + ident);   // AB Ab
  auto [aBIO, aBII] = graph.TileSplit(aBI->id, "nii" + ident, "nio" + ident);  // AbT Abt
  // graph.UpdateAxisAlign(aBI.id, 1u);
  // graph.UpdateAxisAlign(aBII.id, 8u);
  auto x1 = graph.FindNode("x1");
  graph.ApplySplit(x1, aBO->id, aBI->id);
  graph.ApplySplit(x1, aBIO->id, aBII->id);
  x1->attr.sched.loop_axis = aBIO->id;
  x1->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto x2 = graph.FindNode("x2");
  graph.ApplySplit(x2, aBO->id, aBI->id);
  graph.ApplySplit(x2, aBIO->id, aBII->id);
  x2->attr.sched.loop_axis = aBIO->id;
  x2->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto bias = graph.FindNode("bias");
  graph.ApplySplit(bias, aBO->id, aBI->id);
  graph.ApplySplit(bias, aBIO->id, aBII->id);
  bias->attr.sched.loop_axis = aBIO->id;
  bias->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto x1Local = graph.FindNode("x1Local");
  graph.ApplySplit(x1Local, aBO->id, aBI->id);
  graph.ApplySplit(x1Local, aBIO->id, aBII->id);
  x1Local->attr.sched.loop_axis = aBIO->id;
  x1Local->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto x2Local = graph.FindNode("x2Local");
  graph.ApplySplit(x2Local, aBO->id, aBI->id);
  graph.ApplySplit(x2Local, aBIO->id, aBII->id);
  x2Local->attr.sched.loop_axis = aBIO->id;
  x2Local->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto biasLocal = graph.FindNode("biasLocal");
  graph.ApplySplit(biasLocal,aBO->id, aBI->id);
  graph.ApplySplit(biasLocal, aBIO->id, aBII->id);
  biasLocal->attr.sched.loop_axis = aBIO->id;
  biasLocal->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto mean = graph.FindNode("mean");
  mean->attr.api.unit = ge::ComputeUnit::kUnitVector;
  graph.ApplySplit(mean,aBO->id, aBI->id);
  graph.ApplySplit(mean,aBIO->id, aBII->id);
  mean->attr.sched.loop_axis = aBIO->id;
  mean->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto x_out = graph.FindNode("x_out");
  graph.ApplySplit(x_out, aBO->id, aBI->id);
  graph.ApplySplit(x_out, aBIO->id, aBII->id);
  x_out->attr.sched.loop_axis = aBIO->id;
  x_out->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto mean_out = graph.FindNode("mean_out");
  graph.ApplySplit(mean_out, aBO->id, aBI->id);
  graph.ApplySplit(mean_out, aBIO->id, aBII->id);
  mean_out->attr.sched.loop_axis = aBIO->id;
  mean_out->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto rstd = graph.FindNode("rstd");
  rstd->attr.api.unit = ge::ComputeUnit::kUnitVector;
  graph.ApplySplit(rstd,aBO->id, aBI->id);
  graph.ApplySplit(rstd,aBIO->id, aBII->id);
  rstd->attr.sched.loop_axis = aBIO->id;
  rstd->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto rstd_out = graph.FindNode("rstd_out");
  rstd_out->attr.api.unit = ge::ComputeUnit::kUnitVector;
  graph.ApplySplit(rstd_out,aBO->id, aBI->id);
  graph.ApplySplit(rstd_out,aBIO->id, aBII->id);
  rstd_out->attr.sched.loop_axis = aBIO->id;
  rstd_out->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto betaLocal = graph.FindNode("betaLocal");
  graph.ApplySplit(betaLocal,aBO->id, aBI->id);
  graph.ApplySplit(betaLocal,aBIO->id, aBII->id);
  betaLocal->attr.sched.loop_axis = aBIO->id;
  betaLocal->outputs[0].attr.vectorized_axis = {r};

  auto gammaLocal = graph.FindNode("gammaLocal");
  graph.ApplySplit(gammaLocal,aBO->id, aBI->id);
  graph.ApplySplit(gammaLocal,aBIO->id, aBII->id);
  gammaLocal->attr.sched.loop_axis = aBIO->id;
  gammaLocal->outputs[0].attr.vectorized_axis = {r};

  auto y = graph.FindNode("y");
  graph.ApplySplit(y,aBO->id, aBI->id);
  graph.ApplySplit(y,aBIO->id, aBII->id);
  y->attr.sched.loop_axis = aBIO->id;
  y->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto y_out = graph.FindNode("y_out");
  graph.ApplySplit(y_out,aBO->id, aBI->id);
  graph.ApplySplit(y_out,aBIO->id, aBII->id);
  y_out->attr.sched.loop_axis = aBIO->id;
  y_out->outputs[0].attr.vectorized_axis = {aBII->id, r};
}

void Add_Layer_Norm_Normal_AfterQueBufAlloc(ascir::HintGraph &graph) {
  int tensorID = 0;
  int queID = 0;
  int bufID = 0;
  int x1Que = queID++;
  int x2Que = queID++;
  int biasQue = queID++;
  int gammaQue = queID++;
  int betaQue = queID++;
  int meanQue = queID++;
  int rstdQue = queID++;
  int yQue = queID++;
  int xQue = queID++;
  int x32Queue = queID++;
  int oneTBuf = bufID++;

  auto x1 = graph.FindNode("x1");
  x1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto x2 = graph.FindNode("x2");
  x2->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto bias = graph.FindNode("bias");
  bias->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  bias->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto x1Local = graph.FindNode("x1Local");
  x1Local->outputs[0].attr.mem.tensor_id = tensorID++;
  x1Local->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  x1Local->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  x1Local->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  x1Local->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  x1Local->outputs[0].attr.buf.id = ascir::ID_NONE;
  x1Local->outputs[0].attr.que.id = x1Que;
  x1Local->outputs[0].attr.que.depth = 1;
  x1Local->outputs[0].attr.que.buf_num = 1;
  x1Local->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto x2Local = graph.FindNode("x2Local");
  x2Local->outputs[0].attr.mem.tensor_id = tensorID++;
  x2Local->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  x2Local->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  x2Local->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  x2Local->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  x2Local->outputs[0].attr.buf.id = ascir::ID_NONE;
  x2Local->outputs[0].attr.que.id = x2Que;
  x2Local->outputs[0].attr.que.depth = 1;
  x2Local->outputs[0].attr.que.buf_num = 1;
  x2Local->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto biasLocal = graph.FindNode("biasLocal");
  biasLocal->outputs[0].attr.mem.tensor_id = tensorID++;
  biasLocal->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  biasLocal->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  biasLocal->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  biasLocal->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  biasLocal->outputs[0].attr.buf.id = ascir::ID_NONE;
  biasLocal->outputs[0].attr.que.id = biasQue;
  biasLocal->outputs[0].attr.que.depth = 1;
  biasLocal->outputs[0].attr.que.buf_num = 1;
  biasLocal->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto mean = graph.FindNode("mean");
  mean->outputs[0].attr.mem.tensor_id = tensorID++;
  mean->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  mean->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  mean->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  mean->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  mean->outputs[0].attr.buf.id = ascir::ID_NONE;
  mean->outputs[0].attr.que.id = meanQue;
  mean->outputs[0].attr.que.depth = 1;
  mean->outputs[0].attr.que.buf_num = 1;
  mean->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto x_out = graph.FindNode("x_out");
  x_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  x_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto mean_out = graph.FindNode("mean_out");
  mean_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  mean_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto one = graph.FindNode("one");
  one->outputs[0].attr.mem.tensor_id = tensorID++;
  one->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  one->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  one->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  one->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  one->outputs[0].attr.buf.id = oneTBuf;
  one->outputs[0].attr.que.id = ascir::ID_NONE;
  one->outputs[0].attr.que.depth = ascir::ID_NONE;
  one->outputs[0].attr.que.buf_num = ascir::ID_NONE;
  one->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto rstd = graph.FindNode("rstd");
  rstd->outputs[0].attr.mem.tensor_id = tensorID++;
  rstd->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  rstd->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  rstd->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  rstd->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  rstd->outputs[0].attr.buf.id =ascir::ID_NONE;
  rstd->outputs[0].attr.que.id = yQue;
  rstd->outputs[0].attr.que.depth = 1;
  rstd->outputs[0].attr.que.buf_num = 1;
  rstd->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto rstd_out = graph.FindNode("rstd_out");
  rstd_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  rstd_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto beta = graph.FindNode("beta");
  beta->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  beta->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto betaLocal = graph.FindNode("betaLocal");
  betaLocal->outputs[0].attr.mem.tensor_id = tensorID++;
  betaLocal->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  betaLocal->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  betaLocal->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  betaLocal->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  betaLocal->outputs[0].attr.buf.id = ascir::ID_NONE;
  betaLocal->outputs[0].attr.que.id = betaQue;
  betaLocal->outputs[0].attr.que.depth = 1;
  betaLocal->outputs[0].attr.que.buf_num = 1;
  betaLocal->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto gamma = graph.FindNode("gamma");
  gamma->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  gamma->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto gammaLocal = graph.FindNode("gammaLocal");
  gammaLocal->outputs[0].attr.mem.tensor_id = tensorID++;
  gammaLocal->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  gammaLocal->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  gammaLocal->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  gammaLocal->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  gammaLocal->outputs[0].attr.buf.id = ascir::ID_NONE;
  gammaLocal->outputs[0].attr.que.id = gammaQue;
  gammaLocal->outputs[0].attr.que.depth = 1;
  gammaLocal->outputs[0].attr.que.buf_num = 1;
  gammaLocal->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto y = graph.FindNode("y");
  y->outputs[0].attr.mem.tensor_id = tensorID++;
  y->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  y->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  y->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  y->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  y->outputs[0].attr.buf.id = ascir::ID_NONE;
  y->outputs[0].attr.que.id = yQue;
  y->outputs[0].attr.que.depth = 1;
  y->outputs[0].attr.que.buf_num = 1;
  y->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto y_out = graph.FindNode("y_out");
  y_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  y_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;
}

void Add_Layer_Norm_Slice_BeforeAutofuse(ascir::HintGraph &graph) {
  auto ONE = ge::sym::kSymbolOne;
  auto ZERO = ge::sym::kSymbolZero;
  // 定义轴的大小
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");

  // 定义轴
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);

  // 定义节点
  int exec_order = 0;
  Data x1("x1" ,graph);
  x1.attr.sched.exec_order = exec_order++;
  x1.attr.sched.axis = {a.id, r.id};
  x1.y.dtype = ge::DT_FLOAT16;
  *x1.y.axis = {a.id, r.id};
  *x1.y.repeats = {A, R};
  *x1.y.strides = {R, ONE};

  Load x1Local("x1Local");
  x1Local.x = x1.y;
  x1Local.attr.sched.exec_order = exec_order++;
  x1Local.attr.sched.axis = {a.id, r.id};
  x1Local.y.dtype = ge::DT_FLOAT16;
  *x1Local.y.axis = {a.id, r.id};
  *x1Local.y.repeats = {A, R};
  *x1Local.y.strides = {R, ONE};

  Data x2("x2" ,graph);
  x2.attr.sched.exec_order = exec_order++;
  x2.attr.sched.axis = {a.id, r.id};
  x2.y.dtype = ge::DT_FLOAT16;
  *x2.y.axis = {a.id, r.id};
  *x2.y.repeats = {A, R};
  *x2.y.strides = {R, ONE};

  Load x2Local("x2Local");
  x2Local.x = x2.y;
  x2Local.attr.sched.exec_order = exec_order++;
  x2Local.attr.sched.axis = {a.id, r.id};
  x2Local.y.dtype = ge::DT_FLOAT16;
  *x2Local.y.axis = {a.id, r.id};
  *x2Local.y.repeats = {A, R};
  *x2Local.y.strides = {R, ONE};

  Data bias("bias" ,graph);
  bias.attr.sched.exec_order = exec_order++;
  bias.attr.sched.axis = {a.id, r.id};
  bias.y.dtype = ge::DT_FLOAT16;
  *bias.y.axis = {a.id, r.id};
  *bias.y.repeats = {A, R};
  *bias.y.strides = {R, ONE};

  Load biasLocal("biasLocal");
  biasLocal.x = bias.y;
  biasLocal.attr.sched.exec_order = exec_order++;
  biasLocal.attr.sched.axis = {a.id, r.id};
  biasLocal.y.dtype = ge::DT_FLOAT16;
  *biasLocal.y.axis = {a.id, r.id};
  *biasLocal.y.repeats = {A, R};
  *biasLocal.y.strides = {R, ONE};

  Concat mean("mean");
  mean.attr.api.unit = ge::ComputeUnit::kUnitVector;
  mean.x = {x1Local.y, x2Local.y, biasLocal.y};
  mean.attr.sched.exec_order = exec_order++;
  mean.attr.sched.axis = {a.id, r.id};
  mean.y.dtype = ge::DT_FLOAT;        // mean
  *mean.y.axis = {a.id, r.id};
  *mean.y.repeats = {A, ONE};
  *mean.y.strides = {ONE, ONE};

  Store x_out("x_out");
  x_out.attr.sched.exec_order = exec_order++;
  x_out.attr.sched.axis = {a.id, r.id};
  x_out.x = mean.y;
  x_out.y.dtype = ge::DT_FLOAT16;
  *x_out.y.axis = {a.id, r.id};
  *x_out.y.repeats = {A, R};
  *x_out.y.strides = {R, ONE};

  Concat rstd("rstd");
  rstd.attr.api.unit = ge::ComputeUnit::kUnitVector;
  rstd.attr.sched.exec_order = exec_order++;
  rstd.attr.sched.axis = {a.id, r.id};
  rstd.x = {mean.y, mean.y};
  rstd.y.dtype = ge::DT_FLOAT;      // x-mean
  *rstd.y.axis = {a.id, r.id};
  *rstd.y.repeats = {A, R};
  *rstd.y.strides = {R, ONE};

  Store mean_out("mean_out");
  mean_out.attr.sched.exec_order = exec_order++;
  mean_out.attr.sched.axis = {a.id, r.id};
  mean_out.x = mean.y;
  mean_out.y.dtype = ge::DT_FLOAT;
  *mean_out.y.axis = {a.id, r.id};
  *mean_out.y.repeats = {A, ONE};
  *mean_out.y.strides = {ONE, ONE};

  Store rstd_out("rstd_out");
  rstd_out.attr.sched.exec_order = exec_order++;
  rstd_out.attr.sched.axis = {a.id, r.id};
  rstd_out.x = rstd.y;
  rstd_out.y.dtype = ge::DT_FLOAT;
  *rstd_out.y.axis = {a.id, r.id};
  *rstd_out.y.repeats = {A, ONE};
  *rstd_out.y.strides = {ONE, ONE};

  Data beta("beta" ,graph);
  beta.attr.sched.exec_order = exec_order++;
  beta.attr.sched.axis = {a.id, r.id};
  beta.y.dtype = ge::DT_FLOAT16;
  *beta.y.axis = {a.id, r.id};
  *beta.y.repeats = {ONE, R};
  *beta.y.strides = {ZERO, ONE};

  Load betaLocal("betaLocal");
  betaLocal.x = beta.y;
  betaLocal.attr.sched.exec_order = exec_order++;
  betaLocal.attr.sched.axis = {a.id, r.id};
  betaLocal.y.dtype = ge::DT_FLOAT16;
  *betaLocal.y.axis = {a.id, r.id};
  *betaLocal.y.repeats = {ONE, R};
  *betaLocal.y.strides = {ZERO, ONE};

  Data gamma("gamma" ,graph);
  gamma.attr.sched.exec_order = exec_order++;
  gamma.attr.sched.axis = {a.id, r.id};
  gamma.y.dtype = ge::DT_FLOAT16;
  *gamma.y.axis = {a.id, r.id};
  *gamma.y.repeats = {ONE, R};
  *gamma.y.strides = {ZERO, ONE};

  Load gammaLocal("gammaLocal");
  gammaLocal.x = gamma.y;
  gammaLocal.attr.sched.exec_order = exec_order++;
  gammaLocal.attr.sched.axis = {a.id, r.id};
  gammaLocal.y.dtype = ge::DT_FLOAT16;
  *gammaLocal.y.axis = {a.id, r.id};
  *gammaLocal.y.repeats = {ONE, R};
  *gammaLocal.y.strides = {ZERO, ONE};

  Concat y("y");
  y.attr.api.unit = ge::ComputeUnit::kUnitVector;
  y.attr.sched.exec_order = exec_order++;
  y.attr.sched.axis = {a.id, r.id};
  y.x = {rstd.y, betaLocal.y, gammaLocal.y, rstd.y};                 // x-mean
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {a.id, r.id};
  *y.y.repeats = {A, R};
  *y.y.strides = {R, ONE};

  Store y_out("y_out");
  y_out.attr.sched.exec_order = exec_order++;
  y_out.attr.sched.axis = {a.id, r.id};
  y_out.x = y.y;
  y_out.y.dtype = ge::DT_FLOAT16;
  *y_out.y.axis = {a.id, r.id};
  *y_out.y.repeats = {A, R};
  *y_out.y.strides = {R, ONE};

  Output buf1("buf1");
  buf1.x = x_out.y;
  buf1.attr.sched.exec_order = exec_order++;
  buf1.y.dtype = ge::DT_FLOAT16;
  *buf1.y.axis = {a.id, r.id};
  *buf1.y.repeats = {A, R};
  *buf1.y.strides = {R, ONE};

  Output buf2("buf2");
  buf2.x = mean_out.y;
  buf2.attr.sched.exec_order = exec_order++;
  buf2.y.dtype = ge::DT_FLOAT;
  *buf2.y.axis = {a.id, r.id};
  *buf2.y.repeats = {A, ONE};
  *buf2.y.strides = {ONE, ONE};

  Output buf3("buf3");
  buf3.x = rstd_out.y;
  buf3.attr.sched.exec_order = exec_order++;
  buf3.y.dtype = ge::DT_FLOAT;
  *buf3.y.axis = {a.id, r.id};
  *buf3.y.repeats = {A, ONE};
  *buf3.y.strides = {ONE, ONE};

  Output buf("buf");
  buf.x = y_out.y;
  buf.attr.sched.exec_order = exec_order++;
  buf.y.dtype = ge::DT_FLOAT16;
  *buf.y.axis = {a.id, r.id};
  *buf.y.repeats = {A, R};
  *buf.y.strides = {R, ONE};

}

/*
for aBO
  for aBI
    for rO
      for rI
        load x1
        load x2
        load bias
        CalcMean
        Store X
        Load beta
        Load gamma
        CalcRstd
        Store mean
        Store rstd
        CalcY
        Store y
*/

void Add_Layer_Norm_Slice_AfterScheduler(ascir::HintGraph &graph) {
  auto a = graph.FindAxis(0)->id;
  auto r = graph.FindAxis(1)->id;

  auto [aBO, aBI] = graph.BlockSplit(a, "sbi", "sbo");
  auto [rO, rI] = graph.TileSplit(r, "sii", "sio");
//  graph.UpdateAxisAlign(rI->id, 16); // 这个接口现在没了直接赋值就好了
  rI->align = ge::Symbol(16);
  auto x1 = graph.FindNode("x1");
  graph.ApplySplit(x1,aBO->id, aBI->id);
  graph.ApplySplit(x1, rO->id, rI->id);
  x1->attr.sched.loop_axis = rO->id;
  x1->outputs[0].attr.vectorized_axis = {rI->id};

  auto x2 = graph.FindNode("x2");
  graph.ApplySplit(x2,aBO->id, aBI->id);
  graph.ApplySplit(x2, rO->id, rI->id);
  x2->attr.sched.loop_axis = rO->id;
  x2->outputs[0].attr.vectorized_axis = {rI->id};

  auto bias = graph.FindNode("bias");
  graph.ApplySplit(bias,aBO->id, aBI->id);
  graph.ApplySplit(bias, rO->id, rI->id);
  bias->attr.sched.loop_axis = rO->id;
  bias->outputs[0].attr.vectorized_axis = {rI->id};

  auto x1Local = graph.FindNode("x1Local");
  graph.ApplySplit(x1Local,aBO->id, aBI->id);
  graph.ApplySplit(x1Local, rO->id, rI->id);
  x1Local->attr.sched.loop_axis = rO->id;
  x1Local->outputs[0].attr.vectorized_axis = {rI->id};

  auto x2Local = graph.FindNode("x2Local");
  graph.ApplySplit(x2Local,aBO->id, aBI->id);
  graph.ApplySplit(x2Local, rO->id, rI->id);
  x2Local->attr.sched.loop_axis = rO->id;
  x2Local->outputs[0].attr.vectorized_axis = {rI->id};

  auto biasLocal = graph.FindNode("biasLocal");
  graph.ApplySplit(biasLocal,aBO->id, aBI->id);
  graph.ApplySplit(biasLocal, rO->id, rI->id);
  biasLocal->attr.sched.loop_axis = rO->id;
  biasLocal->outputs[0].attr.vectorized_axis = {rI->id};

  auto mean = graph.FindNode("mean");
  graph.ApplySplit(mean,aBO->id, aBI->id);
  graph.ApplySplit(mean, rO->id, rI->id);
  mean->attr.sched.loop_axis = rO->id;
  mean->outputs[0].attr.vectorized_axis = {rI->id};

  auto x_out = graph.FindNode("x_out");
  graph.ApplySplit(x_out,aBO->id, aBI->id);
  graph.ApplySplit(x_out, rO->id, rI->id);
  x_out->attr.sched.loop_axis = rO->id;
  x_out->outputs[0].attr.vectorized_axis = {rI->id};

  auto mean_out = graph.FindNode("mean_out");
  graph.ApplySplit(mean_out,aBO->id, aBI->id);
  graph.ApplySplit(mean_out, rO->id, rI->id);
  mean_out->attr.sched.loop_axis = rO->id;
  mean_out->outputs[0].attr.vectorized_axis = {rI->id};

  auto rstd = graph.FindNode("rstd");
  graph.ApplySplit(rstd,aBO->id, aBI->id);
  graph.ApplySplit(rstd, rO->id, rI->id);
  rstd->attr.sched.loop_axis = rO->id;
  rstd->outputs[0].attr.vectorized_axis = {rO->id, rI->id};

  auto rstd_out = graph.FindNode("rstd_out");
  graph.ApplySplit(rstd_out,aBO->id, aBI->id);
  graph.ApplySplit(rstd_out, rO->id, rI->id);
  rstd_out->attr.sched.loop_axis = rO->id;
  rstd_out->outputs[0].attr.vectorized_axis = {rI->id};

  auto betaLocal = graph.FindNode("betaLocal");
  graph.ApplySplit(betaLocal,aBO->id, aBI->id);
  graph.ApplySplit(betaLocal, rO->id, rI->id);
  betaLocal->attr.sched.loop_axis = rO->id;
  betaLocal->outputs[0].attr.vectorized_axis = {rO->id, rI->id};

  auto gammaLocal = graph.FindNode("gammaLocal");
  graph.ApplySplit(gammaLocal,aBO->id, aBI->id);
  graph.ApplySplit(gammaLocal, rO->id, rI->id);
  gammaLocal->attr.sched.loop_axis = rO->id;
  gammaLocal->outputs[0].attr.vectorized_axis = {rO->id, rI->id};

  auto y = graph.FindNode("y");
  graph.ApplySplit(y,aBO->id, aBI->id);
  graph.ApplySplit(y, rO->id, rI->id);
  y->attr.sched.loop_axis = rO->id;
  y->outputs[0].attr.vectorized_axis = {rO->id, rI->id};

  auto y_out = graph.FindNode("y_out");
  graph.ApplySplit(y_out,aBO->id, aBI->id);
  graph.ApplySplit(y_out, rO->id, rI->id);
  y_out->attr.sched.loop_axis = rO->id;
  y_out->outputs[0].attr.vectorized_axis = {rI->id};
}

void Add_Layer_Norm_Slice_AfterQueBufAlloc(ascir::HintGraph &graph) {
  int tensorID = 0;
  int queID = 0;
  int bufID = 0;
  int x1Que = queID++;
  int x2Que = queID++;
  int biasQue = queID++;
  int xQue = queID++;
  int yQue = queID++;
  int betaQue = queID++;
  int gammaQue = queID++;
  int meanQue = queID++;
  int rstdQue = queID++;

  auto x1 = graph.FindNode("x1");
  x1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto x2 = graph.FindNode("x2");
  x2->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto bias = graph.FindNode("bias");
  bias->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  bias->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto x1Local = graph.FindNode("x1Local");
  x1Local->outputs[0].attr.mem.tensor_id = tensorID++;
  x1Local->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  x1Local->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  x1Local->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  x1Local->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  x1Local->outputs[0].attr.buf.id = ascir::ID_NONE;
  x1Local->outputs[0].attr.que.id = x1Que;
  x1Local->outputs[0].attr.que.depth = 1;
  x1Local->outputs[0].attr.que.buf_num = 1;
  x1Local->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto x2Local = graph.FindNode("x2Local");
  x2Local->outputs[0].attr.mem.tensor_id = tensorID++;
  x2Local->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  x2Local->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  x2Local->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  x2Local->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  x2Local->outputs[0].attr.buf.id = ascir::ID_NONE;
  x2Local->outputs[0].attr.que.id = x2Que;
  x2Local->outputs[0].attr.que.depth = 1;
  x2Local->outputs[0].attr.que.buf_num = 1;
  x2Local->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto biasLocal = graph.FindNode("biasLocal");
  biasLocal->outputs[0].attr.mem.tensor_id = tensorID++;
  biasLocal->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  biasLocal->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  biasLocal->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  biasLocal->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  biasLocal->outputs[0].attr.buf.id = ascir::ID_NONE;
  biasLocal->outputs[0].attr.que.id = biasQue;
  biasLocal->outputs[0].attr.que.depth = 1;
  biasLocal->outputs[0].attr.que.buf_num = 1;
  biasLocal->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto mean = graph.FindNode("mean");
  mean->outputs[0].attr.mem.tensor_id = tensorID++;
  mean->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  mean->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  mean->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  mean->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  mean->outputs[0].attr.buf.id = ascir::ID_NONE;
  mean->outputs[0].attr.que.id = meanQue;
  mean->outputs[0].attr.que.depth = 1;
  mean->outputs[0].attr.que.buf_num = 1;
  mean->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto x_out = graph.FindNode("x_out");
  x_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  x_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto mean_out = graph.FindNode("mean_out");
  mean_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  mean_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto rstd = graph.FindNode("rstd");
  rstd->outputs[0].attr.mem.tensor_id = tensorID++;
  rstd->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  rstd->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  rstd->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  rstd->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  rstd->outputs[0].attr.buf.id =ascir::ID_NONE;
  rstd->outputs[0].attr.que.id = yQue;
  rstd->outputs[0].attr.que.depth = 1;
  rstd->outputs[0].attr.que.buf_num = 1;
  rstd->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto rstd_out = graph.FindNode("rstd_out");
  rstd_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  rstd_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto beta = graph.FindNode("beta");
  beta->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  beta->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto betaLocal = graph.FindNode("betaLocal");
  betaLocal->outputs[0].attr.mem.tensor_id = tensorID++;
  betaLocal->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  betaLocal->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  betaLocal->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  betaLocal->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  betaLocal->outputs[0].attr.buf.id = ascir::ID_NONE;
  betaLocal->outputs[0].attr.que.id = betaQue;
  betaLocal->outputs[0].attr.que.depth = 1;
  betaLocal->outputs[0].attr.que.buf_num = 1;
  betaLocal->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto gamma = graph.FindNode("gamma");
  gamma->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  gamma->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto gammaLocal = graph.FindNode("gammaLocal");
  gammaLocal->outputs[0].attr.mem.tensor_id = tensorID++;
  gammaLocal->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  gammaLocal->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  gammaLocal->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  gammaLocal->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  gammaLocal->outputs[0].attr.buf.id = ascir::ID_NONE;
  gammaLocal->outputs[0].attr.que.id = gammaQue;
  gammaLocal->outputs[0].attr.que.depth = 1;
  gammaLocal->outputs[0].attr.que.buf_num = 1;
  gammaLocal->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto y = graph.FindNode("y");
  y->outputs[0].attr.mem.tensor_id = tensorID++;
  y->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  y->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  y->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  y->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  y->outputs[0].attr.buf.id = ascir::ID_NONE;
  y->outputs[0].attr.que.id = yQue;
  y->outputs[0].attr.que.depth = 1;
  y->outputs[0].attr.que.buf_num = 1;
  y->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto y_out = graph.FindNode("y_out");
  y_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  y_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;
}

void Add_Layer_Norm_Welford_BeforeAutofuse(ascir::HintGraph &graph) {
  auto ONE = ge::sym::kSymbolOne;
  auto ZERO = ge::sym::kSymbolZero;
  // 定义轴的大小
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");

  // 定义轴
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);

  // 定义节点
  int exec_order = 0;
  Data x1("x1" ,graph);
  x1.attr.sched.exec_order = exec_order++;
  x1.attr.sched.axis = {a.id, r.id};
  x1.y.dtype = ge::DT_FLOAT16;
  *x1.y.axis = {a.id, r.id};
  *x1.y.repeats = {A, R};
  *x1.y.strides = {R, ONE};

  Load x1Local("x1Local");
  x1Local.x = x1.y;
  x1Local.attr.sched.exec_order = exec_order++;
  x1Local.attr.sched.axis = {a.id, r.id};
  x1Local.y.dtype = ge::DT_FLOAT16;
  *x1Local.y.axis = {a.id, r.id};
  *x1Local.y.repeats = {A, R};
  *x1Local.y.strides = {R, ONE};

  Data x2("x2" ,graph);
  x2.attr.sched.exec_order = exec_order++;
  x2.attr.sched.axis = {a.id, r.id};
  x2.y.dtype = ge::DT_FLOAT16;
  *x2.y.axis = {a.id, r.id};
  *x2.y.repeats = {A, R};
  *x2.y.strides = {R, ONE};

  Load x2Local("x2Local");
  x2Local.x = x2.y;
  x2Local.attr.sched.exec_order = exec_order++;
  x2Local.attr.sched.axis = {a.id, r.id};
  x2Local.y.dtype = ge::DT_FLOAT16;
  *x2Local.y.axis = {a.id, r.id};
  *x2Local.y.repeats = {A, R};
  *x2Local.y.strides = {R, ONE};

  Data bias("bias" ,graph);
  bias.attr.sched.exec_order = exec_order++;
  bias.attr.sched.axis = {a.id, r.id};
  bias.y.dtype = ge::DT_FLOAT16;
  *bias.y.axis = {a.id, r.id};
  *bias.y.repeats = {A, R};
  *bias.y.strides = {R, ONE};

  Load biasLocal("biasLocal");
  biasLocal.x = bias.y;
  biasLocal.attr.sched.exec_order = exec_order++;
  biasLocal.attr.sched.axis = {a.id, r.id};
  biasLocal.y.dtype = ge::DT_FLOAT16;
  *biasLocal.y.axis = {a.id, r.id};
  *biasLocal.y.repeats = {A, R};
  *biasLocal.y.strides = {R, ONE};

  Concat part1("part1");
  part1.attr.api.unit = ge::ComputeUnit::kUnitVector;
  part1.x = {x1Local.y, x2Local.y, biasLocal.y};
  part1.attr.sched.exec_order = exec_order++;
  part1.attr.sched.axis = {a.id, r.id};
  part1.y.dtype = ge::DT_FLOAT16;        // x out
  *part1.y.axis = {a.id, r.id};
  *part1.y.repeats = {A, R};
  *part1.y.strides = {R, ONE};

  Store x_out("x_out");
  x_out.attr.sched.exec_order = exec_order++;
  x_out.attr.sched.axis = {a.id, r.id};
  x_out.x = part1.y;
  x_out.y.dtype = ge::DT_FLOAT16;
  *x_out.y.axis = {a.id, r.id};
  *x_out.y.repeats = {A, R};
  *x_out.y.strides = {R, ONE};

  Store x_fp32_out("x_fp32_out");
  x_fp32_out.attr.sched.exec_order = exec_order++;
  x_fp32_out.attr.sched.axis = {a.id, r.id};
  x_fp32_out.x = part1.y;
  x_fp32_out.y.dtype = ge::DT_FLOAT;
  *x_fp32_out.y.axis = {a.id, r.id};
  *x_fp32_out.y.repeats = {A, R};
  *x_fp32_out.y.strides = {R, ONE};

  Concat part1Final("part1Final");
  part1Final.attr.api.unit = ge::ComputeUnit::kUnitVector;
  part1Final.attr.sched.exec_order = exec_order++;
  part1Final.attr.sched.axis = {a.id, r.id};
  part1Final.x = {part1.y, part1.y};
  part1Final.y.dtype = ge::DT_FLOAT;      // mean
  *part1Final.y.axis = {a.id, r.id};
  *part1Final.y.repeats = {A, ONE};
  *part1Final.y.strides = {ONE, ONE};

  Store mean_out("mean_out");
  mean_out.attr.sched.exec_order = exec_order++;
  mean_out.attr.sched.axis = {a.id, r.id};
  mean_out.x = part1Final.y;
  mean_out.y.dtype = ge::DT_FLOAT;
  *mean_out.y.axis = {a.id, r.id};
  *mean_out.y.repeats = {A, ONE};
  *mean_out.y.strides = {ONE, ONE};

  Store rstd_out("rstd_out");
  rstd_out.attr.sched.exec_order = exec_order++;
  rstd_out.attr.sched.axis = {a.id, r.id};
  rstd_out.x = part1Final.y;
  rstd_out.y.dtype = ge::DT_FLOAT;
  *rstd_out.y.axis = {a.id, r.id};
  *rstd_out.y.repeats = {A, ONE};
  *rstd_out.y.strides = {ONE, ONE};

  Load x32("x32");
  x32.x = x_fp32_out.y;
  x32.attr.sched.exec_order = exec_order++;
  x32.attr.sched.axis = {a.id, r.id};
  x32.y.dtype = ge::DT_FLOAT;
  *x32.y.axis = {a.id, r.id};
  *x32.y.repeats = {A, R};
  *x32.y.strides = {R, ONE};

  Data beta("beta" ,graph);
  beta.attr.sched.exec_order = exec_order++;
  beta.attr.sched.axis = {a.id, r.id};
  beta.y.dtype = ge::DT_FLOAT16;
  *beta.y.axis = {a.id, r.id};
  *beta.y.repeats = {ONE, R};
  *beta.y.strides = {ZERO, ONE};

  Load betaLocal("betaLocal");
  betaLocal.x = beta.y;
  betaLocal.attr.sched.exec_order = exec_order++;
  betaLocal.attr.sched.axis = {a.id, r.id};
  betaLocal.y.dtype = ge::DT_FLOAT16;
  *betaLocal.y.axis = {a.id, r.id};
  *betaLocal.y.repeats = {ONE, R};
  *betaLocal.y.strides = {ZERO, ONE};

  Data gamma("gamma" ,graph);
  gamma.attr.sched.exec_order = exec_order++;
  gamma.attr.sched.axis = {a.id, r.id};
  gamma.y.dtype = ge::DT_FLOAT16;
  *gamma.y.axis = {a.id, r.id};
  *gamma.y.repeats = {ONE, R};
  *gamma.y.strides = {ZERO, ONE};

  Load gammaLocal("gammaLocal");
  gammaLocal.x = gamma.y;
  gammaLocal.attr.sched.exec_order = exec_order++;
  gammaLocal.attr.sched.axis = {a.id, r.id};
  gammaLocal.y.dtype = ge::DT_FLOAT16;
  *gammaLocal.y.axis = {a.id, r.id};
  *gammaLocal.y.repeats = {ONE, R};
  *gammaLocal.y.strides = {ZERO, ONE};

  Concat y("y");
  y.attr.api.unit = ge::ComputeUnit::kUnitVector;
  y.attr.sched.exec_order = exec_order++;
  y.attr.sched.axis = {a.id, r.id};
  y.x = {x32.y, betaLocal.y, gammaLocal.y, x32.y};
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {a.id, r.id};
  *y.y.repeats = {A, R};
  *y.y.strides = {R, ONE};

  Store y_out("y_out");
  y_out.attr.sched.exec_order = exec_order++;
  y_out.attr.sched.axis = {a.id, r.id};
  y_out.x = y.y;
  y_out.y.dtype = ge::DT_FLOAT16;
  *y_out.y.axis = {a.id, r.id};
  *y_out.y.repeats = {A, R};
  *y_out.y.strides = {R, ONE};

  Output buf1("buf1");
  buf1.x = x_out.y;
  buf1.attr.sched.exec_order = exec_order++;
  buf1.y.dtype = ge::DT_FLOAT16;
  *buf1.y.axis = {a.id, r.id};
  *buf1.y.repeats = {A, R};
  *buf1.y.strides = {R, ONE};

  Output buf2("buf2");
  buf2.x = mean_out.y;
  buf2.attr.sched.exec_order = exec_order++;
  buf2.y.dtype = ge::DT_FLOAT;
  *buf2.y.axis = {a.id, r.id};
  *buf2.y.repeats = {A, ONE};
  *buf2.y.strides = {ONE, ONE};

  Output buf3("buf3");
  buf3.x = rstd_out.y;
  buf3.attr.sched.exec_order = exec_order++;
  buf3.y.dtype = ge::DT_FLOAT;
  *buf3.y.axis = {a.id, r.id};
  *buf3.y.repeats = {A, ONE};
  *buf3.y.strides = {ONE, ONE};

  Output buf("buf");
  buf.x = y_out.y;
  buf.attr.sched.exec_order = exec_order++;
  buf.y.dtype = ge::DT_FLOAT16;
  *buf.y.axis = {a.id, r.id};
  *buf.y.repeats = {A, R};
  *buf.y.strides = {R, ONE};
}

void Add_Layer_Norm_Welford_AfterScheduler(ascir::HintGraph &graph) {
  auto a = graph.FindAxis(0)->id;
  auto r = graph.FindAxis(1)->id;

  auto [aBO, aBI] = graph.BlockSplit(a, "wbi", "wbo");
  auto [rO, rI] = graph.TileSplit(r, "wii", "wio");

  auto x1 = graph.FindNode("x1");
  graph.ApplySplit(x1,aBO->id, aBI->id);
  graph.ApplySplit(x1, rO->id, rI->id);
  x1->attr.sched.loop_axis = rO->id;
  x1->outputs[0].attr.vectorized_axis = {rI->id};

  auto x2 = graph.FindNode("x2");
  graph.ApplySplit(x2,aBO->id, aBI->id);
  graph.ApplySplit(x2, rO->id, rI->id);
  x2->attr.sched.loop_axis = rO->id;
  x2->outputs[0].attr.vectorized_axis = {rI->id};

  auto bias = graph.FindNode("bias");
  graph.ApplySplit(bias,aBO->id, aBI->id);
  graph.ApplySplit(bias, rO->id, rI->id);
  bias->attr.sched.loop_axis = rO->id;
  bias->outputs[0].attr.vectorized_axis = {rI->id};

  auto x1Local = graph.FindNode("x1Local");
  graph.ApplySplit(x1Local,aBO->id, aBI->id);
  graph.ApplySplit(x1Local, rO->id, rI->id);
  x1Local->attr.sched.loop_axis = rO->id;
  x1Local->outputs[0].attr.vectorized_axis = {rI->id};

  auto x2Local = graph.FindNode("x2Local");
  graph.ApplySplit(x2Local,aBO->id, aBI->id);
  graph.ApplySplit(x2Local, rO->id, rI->id);
  x2Local->attr.sched.loop_axis = rO->id;
  x2Local->outputs[0].attr.vectorized_axis = {rI->id};

  auto biasLocal = graph.FindNode("biasLocal");
  graph.ApplySplit(biasLocal,aBO->id, aBI->id);
  graph.ApplySplit(biasLocal, rO->id, rI->id);
  biasLocal->attr.sched.loop_axis = rO->id;
  biasLocal->outputs[0].attr.vectorized_axis = {rI->id};

  auto part1 = graph.FindNode("part1");
  graph.ApplySplit(part1,aBO->id, aBI->id);
  graph.ApplySplit(part1, rO->id, rI->id);
  part1->attr.sched.loop_axis = rO->id;
  part1->outputs[0].attr.vectorized_axis = {rI->id};

  auto x_out = graph.FindNode("x_out");
  graph.ApplySplit(x_out,aBO->id, aBI->id);
  graph.ApplySplit(x_out, rO->id, rI->id);
  x_out->attr.sched.loop_axis = rO->id;
  x_out->outputs[0].attr.vectorized_axis = {rI->id};

  auto x_fp32_out = graph.FindNode("x_fp32_out");
  graph.ApplySplit(x_fp32_out,aBO->id, aBI->id);
  graph.ApplySplit(x_fp32_out, rO->id, rI->id);
  x_fp32_out->attr.sched.loop_axis = rO->id;
  x_fp32_out->outputs[0].attr.vectorized_axis = {rI->id};

  auto mean_out = graph.FindNode("mean_out");
  graph.ApplySplit(mean_out,aBO->id, aBI->id);
  graph.ApplySplit(mean_out, rO->id, rI->id);
  mean_out->attr.sched.loop_axis = rO->id;
  mean_out->outputs[0].attr.vectorized_axis = {rI->id};

  auto part1Final = graph.FindNode("part1Final");
  graph.ApplySplit(part1Final,aBO->id, aBI->id);
  graph.ApplySplit(part1Final, rO->id, rI->id);
  part1Final->attr.sched.loop_axis = rO->id;
  part1Final->outputs[0].attr.vectorized_axis = {rI->id};

  auto rstd_out = graph.FindNode("rstd_out");
  graph.ApplySplit(rstd_out,aBO->id, aBI->id);
  graph.ApplySplit(rstd_out, rO->id, rI->id);
  rstd_out->attr.sched.loop_axis = rO->id;
  rstd_out->outputs[0].attr.vectorized_axis = {rI->id};

  auto x32 = graph.FindNode("x32");
  graph.ApplySplit(x32,aBO->id, aBI->id);
  graph.ApplySplit(x32, rO->id, rI->id);
  x32->attr.sched.loop_axis = rO->id;
  x32->outputs[0].attr.vectorized_axis = {rI->id};

  auto betaLocal = graph.FindNode("betaLocal");
  graph.ApplySplit(betaLocal,aBO->id, aBI->id);
  graph.ApplySplit(betaLocal, rO->id, rI->id);
  betaLocal->attr.sched.loop_axis = rO->id;
  betaLocal->outputs[0].attr.vectorized_axis = {rI->id};

  auto gammaLocal = graph.FindNode("gammaLocal");
  graph.ApplySplit(gammaLocal,aBO->id, aBI->id);
  graph.ApplySplit(gammaLocal, rO->id, rI->id);
  gammaLocal->attr.sched.loop_axis = rO->id;
  gammaLocal->outputs[0].attr.vectorized_axis = {rI->id};

  auto y = graph.FindNode("y");
  graph.ApplySplit(y,aBO->id, aBI->id);
  graph.ApplySplit(y, rO->id, rI->id);
  y->attr.sched.loop_axis = rO->id;
  y->outputs[0].attr.vectorized_axis = {rI->id};

  auto y_out = graph.FindNode("y_out");
  graph.ApplySplit(y_out,aBO->id, aBI->id);
  graph.ApplySplit(y_out, rO->id, rI->id);
  y_out->attr.sched.loop_axis = rO->id;
  y_out->outputs[0].attr.vectorized_axis = {rI->id};
}

void Add_Layer_Norm_Welford_AfterQueBufAlloc(ascir::HintGraph &graph) {
  int tensorID = 0;
  int queID = 0;
  int bufID = 0;
  int x1Que = queID++;
  int x2Que = queID++;
  int biasQue = queID++;
  int xQue = queID++;
  int yQue = queID++;
  int x32Que = queID++;
  int vQue = queID++;
  int meanQue = queID++;
  int rstdQue = queID++;

  auto x1 = graph.FindNode("x1");
  x1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto x2 = graph.FindNode("x2");
  x2->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto bias = graph.FindNode("bias");
  bias->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  bias->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto x1Local = graph.FindNode("x1Local");
  x1Local->outputs[0].attr.mem.tensor_id = tensorID++;
  x1Local->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  x1Local->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  x1Local->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  x1Local->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  x1Local->outputs[0].attr.buf.id = ascir::ID_NONE;
  x1Local->outputs[0].attr.que.id = x1Que;
  x1Local->outputs[0].attr.que.depth = 1;
  x1Local->outputs[0].attr.que.buf_num = 1;
  x1Local->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto x2Local = graph.FindNode("x2Local");
  x2Local->outputs[0].attr.mem.tensor_id = tensorID++;
  x2Local->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  x2Local->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  x2Local->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  x2Local->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  x2Local->outputs[0].attr.buf.id = ascir::ID_NONE;
  x2Local->outputs[0].attr.que.id = x2Que;
  x2Local->outputs[0].attr.que.depth = 1;
  x2Local->outputs[0].attr.que.buf_num = 1;
  x2Local->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto biasLocal = graph.FindNode("biasLocal");
  biasLocal->outputs[0].attr.mem.tensor_id = tensorID++;
  biasLocal->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  biasLocal->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  biasLocal->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  biasLocal->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  biasLocal->outputs[0].attr.buf.id = ascir::ID_NONE;
  biasLocal->outputs[0].attr.que.id = biasQue;
  biasLocal->outputs[0].attr.que.depth = 1;
  biasLocal->outputs[0].attr.que.buf_num = 1;
  biasLocal->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto part1 = graph.FindNode("part1");
  part1->outputs[0].attr.mem.tensor_id = tensorID++;
  part1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  part1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  part1->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  part1->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  part1->outputs[0].attr.buf.id = ascir::ID_NONE;
  part1->outputs[0].attr.que.id = xQue;
  part1->outputs[0].attr.que.depth = 1;
  part1->outputs[0].attr.que.buf_num = 1;
  part1->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto x_out = graph.FindNode("x_out");
  x_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  x_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto x_fp32_out = graph.FindNode("x_fp32_out");
  x_fp32_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  x_fp32_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto part1Final = graph.FindNode("part1Final");
  part1Final->outputs[0].attr.mem.tensor_id = tensorID++;
  part1Final->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  part1Final->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  part1Final->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  part1Final->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  part1Final->outputs[0].attr.buf.id =ascir::ID_NONE;
  part1Final->outputs[0].attr.que.id = meanQue;
  part1Final->outputs[0].attr.que.depth = 1;
  part1Final->outputs[0].attr.que.buf_num = 1;
  part1Final->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto mean_out = graph.FindNode("mean_out");
  mean_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  mean_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto rstd_out = graph.FindNode("rstd_out");
  rstd_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  rstd_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto x32 = graph.FindNode("x32");
  x32->outputs[0].attr.mem.tensor_id = tensorID++;
  x32->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  x32->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  x32->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  x32->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  x32->outputs[0].attr.buf.id = ascir::ID_NONE;
  x32->outputs[0].attr.que.id = vQue;
  x32->outputs[0].attr.que.depth = 1;
  x32->outputs[0].attr.que.buf_num = 1;
  x32->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto beta = graph.FindNode("beta");
  beta->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  beta->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto betaLocal = graph.FindNode("betaLocal");
  betaLocal->outputs[0].attr.mem.tensor_id = tensorID++;
  betaLocal->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  betaLocal->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  betaLocal->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  betaLocal->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  betaLocal->outputs[0].attr.buf.id = ascir::ID_NONE;
  betaLocal->outputs[0].attr.que.id = x2Que;
  betaLocal->outputs[0].attr.que.depth = 1;
  betaLocal->outputs[0].attr.que.buf_num = 1;
  betaLocal->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto gamma = graph.FindNode("gamma");
  gamma->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  gamma->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto gammaLocal = graph.FindNode("gammaLocal");
  gammaLocal->outputs[0].attr.mem.tensor_id = tensorID++;
  gammaLocal->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  gammaLocal->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  gammaLocal->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  gammaLocal->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  gammaLocal->outputs[0].attr.buf.id = ascir::ID_NONE;
  gammaLocal->outputs[0].attr.que.id = biasQue;
  gammaLocal->outputs[0].attr.que.depth = 1;
  gammaLocal->outputs[0].attr.que.buf_num = 1;
  gammaLocal->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto y = graph.FindNode("y");
  y->outputs[0].attr.mem.tensor_id = tensorID++;
  y->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  y->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  y->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  y->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  y->outputs[0].attr.buf.id = ascir::ID_NONE;
  y->outputs[0].attr.que.id = yQue;
  y->outputs[0].attr.que.depth = 1;
  y->outputs[0].attr.que.buf_num = 1;
  y->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto y_out = graph.FindNode("y_out");
  y_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  y_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;
}

TEST_F(TestGenAddLayerNormalModelInfo, case0)
{
  std::vector<ascir::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  // dtype 固定fp16, 1000
  // 固定需要输出x-out  100
  // 分三个模板 normal(0)  slice(1) welford(5)
  // 固定带bias，并且不需要broadcast 1

  // 1101
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  Add_Layer_Norm_Normal_BeforeAutofuse(graph_normal);
  Add_Layer_Norm_Normal_AfterScheduler(graph_normal);
  Add_Layer_Norm_Normal_AfterQueBufAlloc(graph_normal);
  graphs.emplace_back(graph_normal);

  // 1111
  ascir::AscGraph graph_slice("graph_slice");
  graph_slice.SetTilingKey(1111u);
  Add_Layer_Norm_Slice_BeforeAutofuse(graph_slice);
  Add_Layer_Norm_Slice_AfterScheduler(graph_slice);
  Add_Layer_Norm_Slice_AfterQueBufAlloc(graph_slice);
  graphs.emplace_back(graph_slice);

  // 1151
  ascir::AscGraph graph_welford("graph_welford");
  graph_welford.SetTilingKey(1151u);
  Add_Layer_Norm_Welford_BeforeAutofuse(graph_welford);
  Add_Layer_Norm_Welford_AfterScheduler(graph_welford);
  Add_Layer_Norm_Welford_AfterQueBufAlloc(graph_welford);
  graphs.emplace_back(graph_welford);
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["gen_extra_info"] = "1";
  options["solver_type"] = "HighPerf";
  EXPECT_EQ(GenTilingImpl("AddLayerNorm", graphs, options), true);

  std::system("pwd");
  auto ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/tiling_func_main_add_layer_norm.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system("g++ tiling_func_main_add_layer_norm.cpp AddLayerNorm_*_tiling_func.cpp -o tiling_func_main_add_layer_norm -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main_add_layer_norm");
}


TEST_F(TestGenAddLayerNormalModelInfo, case_axes_reorder)
{
  std::vector<ascir::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  // dtype 固定fp16, 1000
  // 固定需要输出x-out  100
  // 分三个模板 normal(0)  slice(1) welford(5)
  // 固定带bias，并且不需要broadcast 1

  // 1101
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  Add_Layer_Norm_Normal_BeforeAutofuseConstInput(graph_normal);
  Add_Layer_Norm_Normal_AfterScheduler(graph_normal);
  Add_Layer_Norm_Normal_AfterQueBufAlloc(graph_normal);
  graphs.emplace_back(graph_normal);

  // 1111
  ascir::AscGraph graph_slice("graph_slice");
  graph_slice.SetTilingKey(1111u);
  Add_Layer_Norm_Slice_BeforeAutofuse(graph_slice);
  Add_Layer_Norm_Slice_AfterScheduler(graph_slice);
  Add_Layer_Norm_Slice_AfterQueBufAlloc(graph_slice);
  graphs.emplace_back(graph_slice);

  // 1151
  ascir::AscGraph graph_welford("graph_welford");
  graph_welford.SetTilingKey(1151u);
  Add_Layer_Norm_Welford_BeforeAutofuse(graph_welford);
  Add_Layer_Norm_Welford_AfterScheduler(graph_welford);
  Add_Layer_Norm_Welford_AfterQueBufAlloc(graph_welford);
  graphs.emplace_back(graph_welford);
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["gen_extra_info"] = "1";
  options["solver_type"] = "AxesReorder";
  EXPECT_EQ(GenTilingImpl("AddLayerNorm", graphs, options), true);

  std::system("pwd");
  auto ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/tiling_func_main_add_layer_norm.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system("g++ tiling_func_main_add_layer_norm.cpp AddLayerNorm_*_tiling_func.cpp -o tiling_func_main_add_layer_norm -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main_add_layer_norm");
}

TEST_F(TestGenAddLayerNormalModelInfo, case_axes_reorder_replace)
{
  std::vector<ascir::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  // dtype 固定fp16, 1000
  // 固定需要输出x-out  100
  // 分三个模板 normal(0)  slice(1) welford(5)
  // 固定带bias，并且不需要broadcast 1

  // 1101
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  Add_Layer_Norm_Normal_BeforeAutofuseConstInput(graph_normal);
  Add_Layer_Norm_Normal_AfterScheduler(graph_normal);
  Add_Layer_Norm_Normal_AfterQueBufAlloc(graph_normal);
  graphs.emplace_back(graph_normal);

  // 1111
  ascir::AscGraph graph_slice("graph_slice");
  graph_slice.SetTilingKey(1111u);
  Add_Layer_Norm_Slice_BeforeAutofuse(graph_slice);
  Add_Layer_Norm_Slice_AfterScheduler(graph_slice);
  Add_Layer_Norm_Slice_AfterQueBufAlloc(graph_slice);
  graphs.emplace_back(graph_slice);

  // 1151
  ascir::AscGraph graph_welford("graph_welford");
  graph_welford.SetTilingKey(1151u);
  Add_Layer_Norm_Welford_BeforeAutofuse(graph_welford);
  Add_Layer_Norm_Welford_AfterScheduler(graph_welford);
  Add_Layer_Norm_Welford_AfterQueBufAlloc(graph_welford);
  graphs.emplace_back(graph_welford);
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["gen_extra_info"] = "1";
  options["duration_level"] = "1";
  options["do_variable_replace"] = "1";
  options["solver_type"] = "AxesReorder";
  EXPECT_EQ(GenTilingImpl("AddLayerNorm", graphs, options), true);
}

TEST_F(TestGenAddLayerNormalModelInfo, test_autofuse_v2_axes_reorder)
{
  setenv("OPEN_COMPILE_STATS", "open", 1);
  setenv("OPEN_TILINGFUNC_STATS", "open", 1);
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduleGroup schedule_group3;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  ascir::FusedScheduledResult schedule_results;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  // dtype 固定fp16, 1000
  // 固定需要输出x-out  100
  // 分三个模板 normal(0)  slice(1) welford(5)
  // 固定带bias，并且不需要broadcast 1

  // 1101
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  Add_Layer_Norm_Normal_BeforeAutofuse(graph_normal);
  Add_Layer_Norm_Normal_AfterScheduler(graph_normal);
  Add_Layer_Norm_Normal_AfterQueBufAlloc(graph_normal);
  schedule_group1.impl_graphs.emplace_back(graph_normal);

  // 1111
  ascir::AscGraph graph_slice("graph_slice");
  graph_slice.SetTilingKey(1111u);
  Add_Layer_Norm_Slice_BeforeAutofuse(graph_slice);
  Add_Layer_Norm_Slice_AfterScheduler(graph_slice);
  Add_Layer_Norm_Slice_AfterQueBufAlloc(graph_slice);
  schedule_group2.impl_graphs.emplace_back(graph_slice);

  // 1151
  ascir::AscGraph graph_welford("graph_welford");
  graph_welford.SetTilingKey(1151u);
  Add_Layer_Norm_Welford_BeforeAutofuse(graph_welford);
  Add_Layer_Norm_Welford_AfterScheduler(graph_welford);
  Add_Layer_Norm_Welford_AfterQueBufAlloc(graph_welford);
  schedule_group3.impl_graphs.emplace_back(graph_welford);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group3.impl_graphs);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.schedule_groups.emplace_back(schedule_group3);
  std::vector<ascir::ScheduledResult> schedule_result1s{schedule_result1, schedule_result2};
  schedule_results.node_idx_to_scheduled_results.emplace_back(schedule_result1s);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "AddLayerNorm";
  options.emplace(kGenConfigType, "AxesReorder");

  auto res = GenTilingImplAutoFuseV3(op_name, schedule_results, options, tiling_funcs, true);
  for (auto & [key, value] : tiling_funcs) {
    if (key == "TilingHead") {
      std::ofstream oss_head;
      oss_head.open("autofuse_tiling_func_common.h", std::ios::out);
      oss_head << "#include \"AddLayerNorm_tiling_data.h\"\n";
      oss_head << value;
      oss_head.close();
    } else {
      std::ofstream oss;
      oss.open("add_layer_norm_autofuse_tiling_func_" + key + "_3.cpp", std::ios::out);
      oss << value;
      oss.close();
    }
  }
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(schedule_results, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  std::ofstream oss1;
  oss1.open("AddLayerNorm_tiling_data.h", std::ios::out);
  oss1 << tiling_res["graph_normalTilingData"];
  oss1.close();

  auto ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/tiling_func_main_add_layer_norm_sche.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system("g++ tiling_func_main_add_layer_norm_sche.cpp add_layer_norm_autofuse_tiling_func_*_3.cpp -o tiling_func_main_add_layer_norm_autofuse -I ./");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main_add_layer_norm_autofuse");
  EXPECT_EQ(ret, 0);
}
std::string RemoveAutoFuseTilingHeadGuards(const std::string &input) {
  std::istringstream iss(input);
  std::ostringstream oss;
  std::string line;
  const std::string guard_token = "__AUTOFUSE_TILING_FUNC_COMMON_H__";

  while (std::getline(iss, line)) {
    // 如果当前行不包含 guard_token，则保留
    if (line.find(guard_token) == std::string::npos) {
      oss << line << "\n";
    }
  }

  return oss.str();
}

void CombineTilings(const std::map<std::string, std::string> &tilings, std::string &result) {
  const std::string tiling_head = "TilingHead";  // TilingHead作为开头拼接其他文件
  const std::string tiling_data = "TilingData";  // 要排除的 TilingData 子串
  result += RemoveAutoFuseTilingHeadGuards(tilings.at(tiling_head));  // 删除头文件的宏保护，cpp文件不需要
  const std::string include_str = "#include \"autofuse_tiling_func_common.h\"";

  // 遍历所有非 TilingHead 和 TilingData 的条目，去掉第一行后拼接
  for (const auto &[key, value] : tilings) {
    if (key == tiling_head || key.find(tiling_data) != std::string::npos) {
      continue;
    }

    // 查找并跳过第一行头文件行
    size_t include_pos = value.find(include_str);
    if (include_pos != std::string::npos) {
      // 找到 include 行，跳过它，并去掉后面的换行符
      size_t content_start = include_pos + include_str.length();
      while (content_start < value.size() && (value[content_start] == '\n' || value[content_start] == '\r')) {
        content_start++;
      }
      result += value.substr(content_start);
    } else {
      // 如果没有 include 行，直接拼接整个内容
      result += value;
    }

    if (!result.empty() && result.back() != '\n') {
      result += '\n';
    }
  }
}

// 测试autofuse v2 轴重排开启group parallel
// 测试场景：
// 1. 轴重排开启group parallel，预期tiling key为1
// 2. 轴重排关闭group parallel，预期tiling key为0
// 预期：
// 1. 轴重排开启group parallel，预期tiling key为1
TEST_F(TestGenAddLayerNormalModelInfo, test_autofuse_v2_axes_reorder_enable_group_parallel)
{
  setenv("OPEN_COMPILE_STATS", "open", 1);
  setenv("OPEN_TILINGFUNC_STATS", "open", 1);
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduleGroup schedule_group3;
  ascir::ScheduleGroup schedule_group4;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  ascir::FusedScheduledResult schedule_results;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  // dtype 固定fp16, 1000
  // 固定需要输出x-out  100
  // 分三个模板 normal(0)  slice(1) welford(5)
  // 固定带bias，并且不需要broadcast 1

  // 1101
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  Add_Layer_Norm_Normal_BeforeAutofuse(graph_normal);
  Add_Layer_Norm_Normal_AfterScheduler(graph_normal);
  Add_Layer_Norm_Normal_AfterQueBufAlloc(graph_normal);
  schedule_group1.impl_graphs.emplace_back(graph_normal);

  // 1111
  ascir::AscGraph graph_normal1("graph_normal1");
  Add_Layer_Norm_Normal_BeforeAutofuse(graph_normal1);
  Add_Layer_Norm_Normal_AfterScheduler(graph_normal1);
  Add_Layer_Norm_Normal_AfterQueBufAlloc(graph_normal1);
  graph_normal1.SetTilingKey(1111u);
  schedule_group2.impl_graphs.emplace_back(graph_normal1);

  // 1151
  ascir::AscGraph graph_normal2("graph_normal2");
  Add_Layer_Norm_Normal_BeforeAutofuse(graph_normal2);
  Add_Layer_Norm_Normal_AfterScheduler(graph_normal2);
  Add_Layer_Norm_Normal_AfterQueBufAlloc(graph_normal2);
  graph_normal2.SetTilingKey(1151u);
  schedule_group3.impl_graphs.emplace_back(graph_normal2);
  // 1155
  ascir::AscGraph graph_slice("graph_slice");
  Add_Layer_Norm_Slice_BeforeAutofuse(graph_slice);
  Add_Layer_Norm_Slice_AfterScheduler(graph_slice);
  Add_Layer_Norm_Slice_AfterQueBufAlloc(graph_slice);
  graph_slice.SetTilingKey(1155u);
  schedule_group4.impl_graphs.emplace_back(graph_slice);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group3.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group4.impl_graphs);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.schedule_groups.emplace_back(schedule_group3);
  schedule_result2.schedule_groups.emplace_back(schedule_group4);
  schedule_result2.enable_group_parallel = true;
  std::vector<ascir::ScheduledResult> schedule_result1s{schedule_result1, schedule_result2};
  schedule_results.node_idx_to_scheduled_results.emplace_back(schedule_result1s);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "AddLayerNorm";
  options.emplace(kGenConfigType, "AxesReorder");

  auto res = GenTilingImplAutoFuseV3(op_name, schedule_results, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("add_layer_norm_autofuse_*_tiling_func.cpp", std::ios::out);
  oss << "#include \"AddLayerNorm_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);
  EXPECT_TRUE(tiling_func.find("  ArrangeBlockOffsetsAscGraph0Result0(") == std::string::npos);
  EXPECT_TRUE(tiling_func.find("  ArrangeBlockOffsetsAscGraph0Result1(") != std::string::npos);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(schedule_results, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("AddLayerNorm_tiling_data.h", std::ios::out);
  oss << tiling_res["graph_normalTilingData"];
  oss.close();
  std::string kAddLayerNormTilingFunc = R"(
 #include <iostream>
 #include "AddLayerNorm_tiling_data.h"
 using namespace optiling;

 void PrintResult(graph_normalTilingData& tilingData) {
   std::cout << "====================================================" << std::endl;
   auto tiling_key = tilingData.get_graph0_tiling_key();
   std::cout << "get_tiling_key"<< " = " << tiling_key << std::endl;
   MY_ASSERT_EQ(tiling_key, 1);
   std::cout << "====================================================" << std::endl;
 }

 int main() {
   graph_normalTilingData tilingData;
   tilingData.set_block_dim(64);
   tilingData.set_ub_size(245760);
   auto &schedule0_g0_tiling_data = tilingData.graph0_result0_g0_tiling_data;
   auto &schedule0_g1_tiling_data = tilingData.graph0_result0_g1_tiling_data;
   auto &schedule1_g0_tiling_data = tilingData.graph0_result1_g0_tiling_data;
   auto &schedule1_g1_tiling_data = tilingData.graph0_result1_g1_tiling_data;
   schedule0_g0_tiling_data.set_A(1536);
   schedule0_g0_tiling_data.set_R(128);
   schedule0_g0_tiling_data.set_BL(8);
   schedule0_g1_tiling_data.set_A(1536);
   schedule0_g1_tiling_data.set_R(128);
   schedule0_g1_tiling_data.set_BL(8);
   schedule1_g0_tiling_data.set_A(1536);
   schedule1_g0_tiling_data.set_R(128);
   schedule1_g0_tiling_data.set_BL(8);
   schedule1_g1_tiling_data.set_A(1536);
   schedule1_g1_tiling_data.set_R(128);
   if (GetTiling(tilingData)) {
     PrintResult(tilingData);
   } else {
     std::cout << "addlayernorm tiling func execute failed." << std::endl;
     return -1;
   }
   return 0;
 }
)";
  oss.open("tiling_func_main_add_layer_norm_sche.cpp", std::ios::out);
  oss << ResultCheckerUtils::DefineCheckerFunction() << kAddLayerNormTilingFunc;
  oss.close();
  auto ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system("g++ tiling_func_main_add_layer_norm_sche.cpp add_layer_norm_autofuse_*_tiling_func.cpp -o tiling_func_main_add_layer_norm_autofuse -I ./");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_add_layer_norm_autofuse");
  EXPECT_EQ(ret, 0);
}

TEST_F(TestGenAddLayerNormalModelInfo, test_autofuse_v2_axes_reorder_uniq_group)
{
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduledResult schedule_result1;
  ascir::FusedScheduledResult scheduled_results;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  // dtype 固定fp16, 1000
  // 固定需要输出x-out  100
  // 分三个模板 normal(0)  slice(1) welford(5)
  // 固定带bias，并且不需要broadcast 1

  // 1101
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  Add_Layer_Norm_Normal_BeforeAutofuse(graph_normal);
  Add_Layer_Norm_Normal_AfterScheduler(graph_normal);
  Add_Layer_Norm_Normal_AfterQueBufAlloc(graph_normal);
  schedule_group1.impl_graphs.emplace_back(graph_normal);

  // 1111
  ascir::AscGraph graph_slice("graph_slice");
  graph_slice.SetTilingKey(1111u);
  Add_Layer_Norm_Slice_BeforeAutofuse(graph_slice);
  Add_Layer_Norm_Slice_AfterScheduler(graph_slice);
  Add_Layer_Norm_Slice_AfterQueBufAlloc(graph_slice);
  schedule_group1.impl_graphs.emplace_back(graph_slice);

  // 1151
  ascir::AscGraph graph_welford("graph_welford");
  graph_welford.SetTilingKey(1151u);
  Add_Layer_Norm_Welford_BeforeAutofuse(graph_welford);
  Add_Layer_Norm_Welford_AfterScheduler(graph_welford);
  Add_Layer_Norm_Welford_AfterQueBufAlloc(graph_welford);
  schedule_group1.impl_graphs.emplace_back(graph_welford);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  std::vector<ascir::ScheduledResult> schedule_result1s{schedule_result1};
  scheduled_results.node_idx_to_scheduled_results.emplace_back(schedule_result1s);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "AddLayerNorm";
  options.emplace(kGenConfigType, "AxesReorder");

  auto res = GenTilingImplAutoFuseV3(op_name, scheduled_results, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("add_layer_norm_autofuse_*_tiling_func.cpp", std::ios::out);
  oss << "#include \"AddLayerNorm_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(scheduled_results, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("AddLayerNorm_tiling_data.h", std::ios::out);
  oss << tiling_res["graph_normalTilingData"];
  oss.close();

  auto ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/tiling_func_main_add_layer_norm.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system("g++ tiling_func_main_add_layer_norm.cpp add_layer_norm_autofuse_*_tiling_func.cpp -o tiling_func_main_add_layer_norm_autofuse -I ./");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main_add_layer_norm_autofuse > ./info.log");
  EXPECT_EQ(ret, 0);
}

TEST_F(TestGenAddLayerNormalModelInfo, test_autofuse_v2_axes_reorder_reuse_solver)
{
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  // dtype 固定fp16, 1000
  // 固定需要输出x-out  100
  // 分三个模板 normal(0)  slice(1) welford(5)
  // 固定带bias，并且不需要broadcast 1

  // 1101
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  Add_Layer_Norm_Normal_BeforeAutofuse(graph_normal);
  Add_Layer_Norm_Normal_AfterScheduler(graph_normal);
  Add_Layer_Norm_Normal_AfterQueBufAlloc(graph_normal);
  schedule_group1.impl_graphs.emplace_back(graph_normal);

  // 1111
  ascir::AscGraph graph_slice("graph_slice");
  graph_slice.SetTilingKey(1111u);
  Add_Layer_Norm_Normal_BeforeAutofuse(graph_slice, "_slice");
  Add_Layer_Norm_Normal_AfterScheduler(graph_slice, "_slice");
  Add_Layer_Norm_Normal_AfterQueBufAlloc(graph_slice);
  schedule_group2.impl_graphs.emplace_back(graph_slice);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.score_func = "int32_t CalcScore(graph_normalTilingData &tiling_data) { return 1;}";
  schedule_result2.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.score_func = "int32_t CalcScore(graph_normalTilingData &tiling_data) { return 2;}";
  ascir::FusedScheduledResult scheduled_results;
  std::vector<ascir::ScheduledResult> schedule_result1s{schedule_result1, schedule_result2};
  scheduled_results.node_idx_to_scheduled_results.emplace_back(schedule_result1s);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "AddLayerNorm";
  options.emplace(kGenConfigType, "AxesReorder");
  options.emplace("enable_score_func", "1");

  auto res = GenTilingImplAutoFuseV3(op_name, scheduled_results, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("add_layer_norm_autofuse_*_tiling_func.cpp", std::ios::out);
  oss << "#include \"AddLayerNorm_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(scheduled_results, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("AddLayerNorm_tiling_data.h", std::ios::out);
  oss << tiling_res["graph_normalTilingData"];
  oss.close();

  auto ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/tiling_func_main_add_layer_norm_reuse_solver.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system("g++ -g -O0 tiling_func_main_add_layer_norm_reuse_solver.cpp add_layer_norm_autofuse_*_tiling_func.cpp -o tiling_func_main_add_layer_norm_autofuse -I ./");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main_add_layer_norm_autofuse > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
}

TEST_F(TestGenAddLayerNormalModelInfo, test_autofuse_v2_axes_reorder_with_score_funcs)
{
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduleGroup schedule_group3;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  // dtype 固定fp16, 1000
  // 固定需要输出x-out  100
  // 分三个模板 normal(0)  slice(1) welford(5)
  // 固定带bias，并且不需要broadcast 1

  // 1101
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  Add_Layer_Norm_Normal_BeforeAutofuse(graph_normal);
  Add_Layer_Norm_Normal_AfterScheduler(graph_normal);
  Add_Layer_Norm_Normal_AfterQueBufAlloc(graph_normal);
  schedule_group1.impl_graphs.emplace_back(graph_normal);

  // 1111
  ascir::AscGraph graph_slice("graph_slice");
  graph_slice.SetTilingKey(1111u);
  Add_Layer_Norm_Slice_BeforeAutofuse(graph_slice);
  Add_Layer_Norm_Slice_AfterScheduler(graph_slice);
  Add_Layer_Norm_Slice_AfterQueBufAlloc(graph_slice);
  schedule_group2.impl_graphs.emplace_back(graph_slice);

  // 1151
  ascir::AscGraph graph_welford("graph_welford");
  graph_welford.SetTilingKey(1151u);
  Add_Layer_Norm_Welford_BeforeAutofuse(graph_welford);
  Add_Layer_Norm_Welford_AfterScheduler(graph_welford);
  Add_Layer_Norm_Welford_AfterQueBufAlloc(graph_welford);
  schedule_group3.impl_graphs.emplace_back(graph_welford);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group3.impl_graphs);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.schedule_groups.emplace_back(schedule_group2);
  schedule_result1.score_func = "int32_t CalcScore(graph_normalTilingData &tiling_data) { return 1;}";
  schedule_result2.schedule_groups.emplace_back(schedule_group3);
  schedule_result2.score_func = "int32_t CalcScore(graph_normalTilingData &tiling_data) { return 2;}";
  ascir::FusedScheduledResult scheduled_results;
  std::vector<ascir::ScheduledResult> schedule_result1s{schedule_result1, schedule_result2};
  scheduled_results.node_idx_to_scheduled_results.emplace_back(schedule_result1s);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "AddLayerNorm";
  options.emplace(kGenConfigType, "AxesReorder");
  auto res = GenTilingImplAutoFuseV3(op_name, scheduled_results, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  EXPECT_NE(tiling_func.find("ScheduleResult0::CalcScore"), std::string::npos);
  std::ofstream oss;
  oss.open("add_layer_norm_autofuse_*_tiling_func.cpp", std::ios::out);
  oss << "#include \"AddLayerNorm_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(scheduled_results, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("AddLayerNorm_tiling_data.h", std::ios::out);
  oss << tiling_res["graph_normalTilingData"];
  oss.close();

  auto ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/tiling_func_main_add_layer_norm_sche.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system("g++ tiling_func_main_add_layer_norm_sche.cpp add_layer_norm_autofuse_*_tiling_func.cpp -o tiling_func_main_add_layer_norm_autofuse -I ./");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main_add_layer_norm_autofuse > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
}

TEST_F(TestGenAddLayerNormalModelInfo, test_autofuse_v2_axes_reorder_with_score_funcs_one_group)
{
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduleGroup schedule_group3;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  // dtype 固定fp16, 1000
  // 固定需要输出x-out  100
  // 分三个模板 normal(0)  slice(1) welford(5)
  // 固定带bias，并且不需要broadcast 1

  // 1101
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  Add_Layer_Norm_Normal_BeforeAutofuse(graph_normal);
  Add_Layer_Norm_Normal_AfterScheduler(graph_normal);
  Add_Layer_Norm_Normal_AfterQueBufAlloc(graph_normal);
  schedule_group1.impl_graphs.emplace_back(graph_normal);

  // 1151
  ascir::AscGraph graph_welford("graph_welford");
  graph_welford.SetTilingKey(1151u);
  Add_Layer_Norm_Welford_BeforeAutofuse(graph_welford);
  Add_Layer_Norm_Welford_AfterScheduler(graph_welford);
  Add_Layer_Norm_Welford_AfterQueBufAlloc(graph_welford);
  schedule_group3.impl_graphs.emplace_back(graph_welford);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group3.impl_graphs);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.score_func = "int32_t CalcScore(graph_normalTilingData &tiling_data) { return 1;}";
  schedule_result2.schedule_groups.emplace_back(schedule_group3);
  schedule_result2.score_func = "int32_t CalcScore(graph_normalTilingData &tiling_data) { return 2;}";
  ascir::FusedScheduledResult scheduled_results;
  std::vector<ascir::ScheduledResult> schedule_result1s{schedule_result1, schedule_result2};
  scheduled_results.node_idx_to_scheduled_results.emplace_back(schedule_result1s);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "AddLayerNorm";
  options.emplace(kGenConfigType, "AxesReorder");
  options.emplace("enable_score_func", "1");

  auto res = GenTilingImplAutoFuseV3(op_name, scheduled_results, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("add_layer_norm_autofuse_*_tiling_func.cpp", std::ios::out);
  oss << "#include \"AddLayerNorm_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(scheduled_results, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("AddLayerNorm_tiling_data.h", std::ios::out);
  oss << tiling_res["graph_normalTilingData"];
  oss.close();

  auto ret = std::system(
      std::string("cp ").append(ST_DIR).append("/testcase/tiling_func_main_add_layer_norm_sche.cpp ./ -f").c_str());
  ret = system("sed -i '/schedule_result0_g1_tiling_data/d' ./tiling_func_main_add_layer_norm_sche.cpp");
  ret = system("sed -i '/schedule0_g1_tiling_data/d' ./tiling_func_main_add_layer_norm_sche.cpp");

  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);
  ret = std::system(
      "g++ tiling_func_main_add_layer_norm_sche.cpp add_layer_norm_autofuse_*_tiling_func.cpp -o "
      "tiling_func_main_add_layer_norm_autofuse -I ./");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main_add_layer_norm_autofuse > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
}

TEST_F(TestGenAddLayerNormalModelInfo, test_autofuse_v2_high_perf_choose_first_according_to_perf)
{
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduleGroup schedule_group3;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  // dtype 固定fp16, 1000
  // 固定需要输出x-out  100
  // 分三个模板 normal(0)  slice(1) welford(5)
  // 固定带bias，并且不需要broadcast 1

  // 1101
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  Add_Layer_Norm_Normal_BeforeAutofuse(graph_normal);
  Add_Layer_Norm_Normal_AfterScheduler(graph_normal);
  Add_Layer_Norm_Normal_AfterQueBufAlloc(graph_normal);
  schedule_group1.impl_graphs.emplace_back(graph_normal);

  // 1111
  ascir::AscGraph graph_slice("graph_slice");
  graph_slice.SetTilingKey(1111u);
  Add_Layer_Norm_Slice_BeforeAutofuse(graph_slice);
  Add_Layer_Norm_Slice_AfterScheduler(graph_slice);
  Add_Layer_Norm_Slice_AfterQueBufAlloc(graph_slice);
  schedule_group2.impl_graphs.emplace_back(graph_slice);

  // 1151
  ascir::AscGraph graph_welford("graph_welford");
  graph_welford.SetTilingKey(1151u);
  Add_Layer_Norm_Welford_BeforeAutofuse(graph_welford);
  Add_Layer_Norm_Welford_AfterScheduler(graph_welford);
  Add_Layer_Norm_Welford_AfterQueBufAlloc(graph_welford);
  schedule_group3.impl_graphs.emplace_back(graph_welford);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group3.impl_graphs);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.schedule_groups.emplace_back(schedule_group3);
  ascir::FusedScheduledResult scheduled_results;
  std::vector<ascir::ScheduledResult> schedule_result1s{schedule_result1, schedule_result2};
  scheduled_results.node_idx_to_scheduled_results.emplace_back(schedule_result1s);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "AddLayerNorm";
  options.emplace(kGenConfigType, "HighPerf");

  auto res = GenTilingImplAutoFuseV3(op_name, scheduled_results, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("add_layer_norm_autofuse_*_tiling_func.cpp", std::ios::out);
  oss << "#include \"AddLayerNorm_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(scheduled_results, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("AddLayerNorm_tiling_data.h", std::ios::out);
  oss << tiling_res["graph_normalTilingData"];
  oss.close();

  auto ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/tiling_func_main_add_layer_norm_sche.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system("g++ tiling_func_main_add_layer_norm_sche.cpp add_layer_norm_autofuse_*_tiling_func.cpp -o tiling_func_main_add_layer_norm_autofuse -I ./");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main_add_layer_norm_autofuse > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
}

TEST_F(TestGenAddLayerNormalModelInfo, test_autofuse_v2_high_perf_choose_second_according_to_perf)
{
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduleGroup schedule_group3;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  // dtype 固定fp16, 1000
  // 固定需要输出x-out  100
  // 分三个模板 normal(0)  slice(1) welford(5)
  // 固定带bias，并且不需要broadcast 1

  // 1101
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  Add_Layer_Norm_Normal_BeforeAutofuse(graph_normal);
  Add_Layer_Norm_Normal_AfterScheduler(graph_normal);
  Add_Layer_Norm_Normal_AfterQueBufAlloc(graph_normal);
  schedule_group1.impl_graphs.emplace_back(graph_normal);

  // 1111
  ascir::AscGraph graph_slice("graph_slice");
  graph_slice.SetTilingKey(1111u);
  Add_Layer_Norm_Slice_BeforeAutofuse(graph_slice);
  Add_Layer_Norm_Slice_AfterScheduler(graph_slice);
  Add_Layer_Norm_Slice_AfterQueBufAlloc(graph_slice);
  schedule_group2.impl_graphs.emplace_back(graph_slice);

  // 1151
  ascir::AscGraph graph_welford("graph_welford");
  graph_welford.SetTilingKey(1151u);
  Add_Layer_Norm_Welford_BeforeAutofuse(graph_welford);
  Add_Layer_Norm_Welford_AfterScheduler(graph_welford);
  Add_Layer_Norm_Welford_AfterQueBufAlloc(graph_welford);
  schedule_group3.impl_graphs.emplace_back(graph_welford);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group3.impl_graphs);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.schedule_groups.emplace_back(schedule_group3);
  schedule_result2.schedule_groups.emplace_back(schedule_group2);
  ascir::FusedScheduledResult scheduled_results;
  std::vector<ascir::ScheduledResult> schedule_result1s{schedule_result1, schedule_result2};
  scheduled_results.node_idx_to_scheduled_results.emplace_back(schedule_result1s);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string>tiling_funcs;
  std::string op_name = "AddLayerNorm";
  options.emplace(kGenConfigType, "HighPerf");

  auto res = GenTilingImplAutoFuseV3(op_name, scheduled_results, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("add_layer_norm_autofuse_*_tiling_func.cpp", std::ios::out);
  oss << "#include \"AddLayerNorm_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(scheduled_results, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("AddLayerNorm_tiling_data.h", std::ios::out);
  oss << tiling_res["graph_normalTilingData"];
  oss.close();

  auto ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/tiling_func_main_add_layer_norm_sche.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system("g++ tiling_func_main_add_layer_norm_sche.cpp add_layer_norm_autofuse_*_tiling_func.cpp -o tiling_func_main_add_layer_norm_autofuse -I ./");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main_add_layer_norm_autofuse > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
}

TEST_F(TestGenAddLayerNormalModelInfo, test_autofuse_v2_set_log_debug)
{
  setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", 1);
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduleGroup schedule_group3;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;
  // 1101
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  Add_Layer_Norm_Normal_BeforeAutofuse(graph_normal);
  Add_Layer_Norm_Normal_AfterScheduler(graph_normal);
  Add_Layer_Norm_Normal_AfterQueBufAlloc(graph_normal);
  schedule_group1.impl_graphs.emplace_back(graph_normal);
  // 1111
  ascir::AscGraph graph_slice("graph_slice");
  graph_slice.SetTilingKey(1111u);
  Add_Layer_Norm_Slice_BeforeAutofuse(graph_slice);
  Add_Layer_Norm_Slice_AfterScheduler(graph_slice);
  Add_Layer_Norm_Slice_AfterQueBufAlloc(graph_slice);
  schedule_group2.impl_graphs.emplace_back(graph_slice);
  // 1151
  ascir::AscGraph graph_welford("graph_welford");
  graph_welford.SetTilingKey(1151u);
  Add_Layer_Norm_Welford_BeforeAutofuse(graph_welford);
  Add_Layer_Norm_Welford_AfterScheduler(graph_welford);
  Add_Layer_Norm_Welford_AfterQueBufAlloc(graph_welford);
  schedule_group3.impl_graphs.emplace_back(graph_welford);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group3.impl_graphs);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.schedule_groups.emplace_back(schedule_group3);
  ascir::FusedScheduledResult scheduled_results;
  std::vector<ascir::ScheduledResult> schedule_result1s{schedule_result1, schedule_result2};
  scheduled_results.node_idx_to_scheduled_results.emplace_back(schedule_result1s);
  std::map<std::string, std::string> options;
  std::map<std::string, std::string>tiling_funcs;
  std::string op_name = "AddLayerNorm";
  options.emplace(kGenConfigType, "HighPerf");
  auto res = GenTilingImplAutoFuseV3(op_name, scheduled_results, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  EXPECT_NE(tiling_func.find(R"( GELOGD("[%s]" fmt, name, ##__VA_ARGS__))"), std::string::npos);
  unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
}

TEST_F(TestGenAddLayerNormalModelInfo, test_autofuse_v2_axes_reorder_by_env)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--autofuse_att_algorithm=AxesReorder", 1);
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduleGroup schedule_group3;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;
  // 1101
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  Add_Layer_Norm_Normal_BeforeAutofuse(graph_normal);
  Add_Layer_Norm_Normal_AfterScheduler(graph_normal);
  Add_Layer_Norm_Normal_AfterQueBufAlloc(graph_normal);
  schedule_group1.impl_graphs.emplace_back(graph_normal);
  // 1111
  ascir::AscGraph graph_slice("graph_slice");
  graph_slice.SetTilingKey(1111u);
  Add_Layer_Norm_Slice_BeforeAutofuse(graph_slice);
  Add_Layer_Norm_Slice_AfterScheduler(graph_slice);
  Add_Layer_Norm_Slice_AfterQueBufAlloc(graph_slice);
  schedule_group2.impl_graphs.emplace_back(graph_slice);
  // 1151
  ascir::AscGraph graph_welford("graph_welford");
  graph_welford.SetTilingKey(1151u);
  Add_Layer_Norm_Welford_BeforeAutofuse(graph_welford);
  Add_Layer_Norm_Welford_AfterScheduler(graph_welford);
  Add_Layer_Norm_Welford_AfterQueBufAlloc(graph_welford);
  schedule_group3.impl_graphs.emplace_back(graph_welford);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group3.impl_graphs);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.schedule_groups.emplace_back(schedule_group3);
  ascir::FusedScheduledResult scheduled_results;
  std::vector<ascir::ScheduledResult> schedule_result1s{schedule_result1, schedule_result2};
  scheduled_results.node_idx_to_scheduled_results.emplace_back(schedule_result1s);
  std::map<std::string, std::string> options;
  std::map<std::string, std::string>tiling_funcs;
  std::string op_name = "AddLayerNorm";
  auto res = GenTilingImplAutoFuseV3(op_name, scheduled_results, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  EXPECT_NE(tiling_func.find("axes reorder solver"), std::string::npos);
  std::ofstream oss;
  oss.open("add_layer_norm_autofuse_*_tiling_func.cpp", std::ios::out);
  oss << "#include \"AddLayerNorm_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(TestGenAddLayerNormalModelInfo, case_axes_reorder_by_env)
{
  setenv("AUTOFUSE_DFX_FLAGS",
         "--att_enable_multicore_ub_tradeoff=true;--autofuse_att_algorithm=AxesReorder;--att_enable_small_shape_"
         "strategy=true;--att_accuracy_level=1",
         1);
  std::vector<ascir::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  // dtype 固定fp16, 1000
  // 固定需要输出x-out  100
  // 分三个模板 normal(0)  slice(1) welford(5)
  // 固定带bias，并且不需要broadcast 1

  // 1101
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  Add_Layer_Norm_Normal_BeforeAutofuseConstInput(graph_normal);
  Add_Layer_Norm_Normal_AfterScheduler(graph_normal);
  Add_Layer_Norm_Normal_AfterQueBufAlloc(graph_normal);
  graphs.emplace_back(graph_normal);

  // 1111
  ascir::AscGraph graph_slice("graph_slice");
  graph_slice.SetTilingKey(1111u);
  Add_Layer_Norm_Slice_BeforeAutofuse(graph_slice);
  Add_Layer_Norm_Slice_AfterScheduler(graph_slice);
  Add_Layer_Norm_Slice_AfterQueBufAlloc(graph_slice);
  graphs.emplace_back(graph_slice);

  // 1151
  ascir::AscGraph graph_welford("graph_welford");
  graph_welford.SetTilingKey(1151u);
  Add_Layer_Norm_Welford_BeforeAutofuse(graph_welford);
  Add_Layer_Norm_Welford_AfterScheduler(graph_welford);
  Add_Layer_Norm_Welford_AfterQueBufAlloc(graph_welford);
  graphs.emplace_back(graph_welford);
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["gen_extra_info"] = "1";
  options["solver_type"] = "AxesReorder";
  EXPECT_EQ(GenTilingImpl("AddLayerNorm", graphs, options), true);

  std::system("pwd");
  auto ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/tiling_func_main_add_layer_norm.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system("g++ tiling_func_main_add_layer_norm.cpp AddLayerNorm_*_tiling_func.cpp -o tiling_func_main_add_layer_norm -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main_add_layer_norm");
  unsetenv("AUTOFUSE_DFX_FLAGS");
  att::AutoFuseConfig::MutableAttStrategyConfig().enable_multicore_ub_tradeoff = false;
  att::AutoFuseConfig::MutableAttStrategyConfig().enable_small_shape_strategy = false;
  att::AutoFuseConfig::MutableAttStrategyConfig().solution_accuracy_level = 0;
}
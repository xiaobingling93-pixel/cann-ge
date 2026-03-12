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
#include "ascir_ops.h"
#include "tiling_code_generator.h"
#include "gen_tiling_impl.h"
#include "base/att_const_values.h"
#include "graph_construct_utils.h"
#include "test_common_utils.h"

using namespace ge::ascir_op;
namespace ascir {
constexpr int64_t ID_NONE = -1; //取多少？
using namespace ge;
using HintGraph=AscGraph;
}
using namespace att;

class TestGenConcat : public ::testing::Test {
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
    att::AutoFuseConfig::MutableAttStrategyConfig().Reset();
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
  }

  void TearDown() override {
    // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
  }
};
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
void AddHeaderGuardToFile(const std::string& file_name, const std::string& macro_name) {
    std::string content;
    std::ifstream in_file(file_name);
    if (in_file.is_open()) {
        std::string line;
        while (std::getline(in_file, line)) {
            content += line + "\n";
        }
        in_file.close();
    }

    std::ofstream out_file;
    out_file.open(file_name, std::ios::out);
    out_file << "#ifndef " << macro_name << "\n";
    out_file << "#define " << macro_name << "\n";
    out_file << "\n";
    out_file << content;
    out_file << "\n";
    out_file << "#endif // " << macro_name << "\n";
    out_file.close();
}
void Concat_Normal_BeforeAutofuse(ascir::HintGraph &graph) {
  auto ONE = ge::sym::kSymbolOne;
  auto ZERO = ge::sym::kSymbolZero;
  // 定义轴的大小
  auto A = ge::Symbol("A");
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
  y.x = {rstd.y, betaLocal.y, gammaLocal.y, rstd.y};
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {a.id, r.id, bl.id};
  *y.y.repeats = {A, R, ONE};
  *y.y.strides = {R, ONE, ZERO};

  Concat concat("concat");
  concat.x = {x1Local.y, x2Local.y};
  concat.attr.sched.axis = {a.id, r.id, bl.id};
  *concat.y.axis = {a.id, r.id, bl.id};
  *concat.y.repeats = {A, R, ONE};
  *concat.y.strides = {R, ONE, ZERO};

  Store concat_out("cat_out");
  concat_out.attr.sched.exec_order = exec_order++;
  concat_out.attr.sched.axis = {a.id, r.id, bl.id};
  concat_out.x = y.y;
  concat_out.y.dtype = ge::DT_FLOAT16;
  *concat_out.y.axis = {a.id, r.id, bl.id};
  *concat_out.y.repeats = {A, R, ONE};
  *concat_out.y.strides = {R, ONE, ZERO};

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

  Output buf4("buf4");
  buf4.x = concat_out.y;
  buf4.attr.sched.exec_order = exec_order++;
  buf4.y.dtype = ge::DT_FLOAT16;
  *buf4.y.axis = {a.id, r.id, bl.id};
  *buf4.y.repeats = {A, R, ONE};
  *buf4.y.strides = {R, ONE, ZERO};
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

void Concat_Normal_AfterScheduler(ascir::HintGraph &graph) {
  auto a = graph.FindAxis(0)->id;
  auto r = graph.FindAxis(1)->id;
  auto bl = graph.FindAxis(2)->id;

  auto [aBO, aBI] = graph.BlockSplit(a, "nbi", "nbo");   // AB Ab
  auto [aBIO, aBII] = graph.TileSplit(aBI->id, "nii", "nio");  // AbT Abt
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


  auto concat = graph.FindNode("concat");
  graph.ApplySplit(concat,aBO->id, aBI->id);
  graph.ApplySplit(concat,aBIO->id, aBII->id);
  concat->attr.api.unit = ge::ComputeUnit::kUnitVector;
  concat->attr.sched.loop_axis = aBIO->id;
  concat->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto y_out = graph.FindNode("y_out");
  graph.ApplySplit(y_out,aBO->id, aBI->id);
  graph.ApplySplit(y_out,aBIO->id, aBII->id);
  y_out->attr.sched.loop_axis = aBIO->id;
  y_out->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto cat_out = graph.FindNode("cat_out");
  graph.ApplySplit(cat_out,aBO->id, aBI->id);
  graph.ApplySplit(cat_out,aBIO->id, aBII->id);
  cat_out->attr.sched.loop_axis = aBIO->id;
  cat_out->outputs[0].attr.vectorized_axis = {aBII->id, r};
}

void Concat_Normal_AfterQueBufAlloc(ascir::HintGraph &graph) {
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

  auto concat = graph.FindNode("concat");
  concat->outputs[0].attr.mem.tensor_id = tensorID++;
  concat->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  concat->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  concat->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  concat->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  concat->outputs[0].attr.buf.id = ascir::ID_NONE;
  concat->outputs[0].attr.que.id = yQue;
  concat->outputs[0].attr.que.depth = 1;
  concat->outputs[0].attr.que.buf_num = 1;
  concat->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto cat_out = graph.FindNode("cat_out");
  cat_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  cat_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;
}

TEST_F(TestGenConcat, case_axes_reorder)
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
  Concat_Normal_BeforeAutofuse(graph_normal);
  Concat_Normal_AfterScheduler(graph_normal);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph_normal);
  Concat_Normal_AfterQueBufAlloc(graph_normal);
  graphs.emplace_back(graph_normal);
  graphs.emplace_back(graph_normal);


  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["dump_debug_info"] = "./";
  options["gen_extra_info"] = "1";
  options["duration_level"] = "1";
  options["solver_type"] = "AxesReorder";
  EXPECT_EQ(GenTilingImpl("Concat", graphs, options), true);
}

TEST_F(TestGenConcat, case_axes_reorder_by_env)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--autofuse_att_algorithm=AxesReorder;--att_enable_small_shape_strategy=true;--att_enable_multicore_ub_tradeoff=true;--att_corenum_threshold=70;--att_ub_threshold=30", 1);
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
  Concat_Normal_BeforeAutofuse(graph_normal);
  Concat_Normal_AfterScheduler(graph_normal);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph_normal);
  Concat_Normal_AfterQueBufAlloc(graph_normal);
  graphs.emplace_back(graph_normal);


  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["dump_debug_info"] = "./";
  options["gen_extra_info"] = "1";
  options["duration_level"] = "1";
  options["solver_type"] = "AxesReorder";
  EXPECT_EQ(GenTilingImpl("Concat", graphs, options), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(TestGenConcat, case_axes_reorder_replace)
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
  Concat_Normal_BeforeAutofuse(graph_normal);
  Concat_Normal_AfterScheduler(graph_normal);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph_normal);
  Concat_Normal_AfterQueBufAlloc(graph_normal);
  graphs.emplace_back(graph_normal);
  graphs.emplace_back(graph_normal);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["dump_debug_info"] = "./";
  options["gen_extra_info"] = "1";
  options["duration_level"] = "1";
  options["do_variable_replace"] = "1";
  options["solver_type"] = "AxesReorder";
  EXPECT_EQ(GenTilingImpl("Concat", graphs, options), true);
}
namespace ge {
namespace ascir {
namespace cg {
Status BuildTqueTbufAscendGraph_single_case(ge::AscGraph &graph, bool reuse_temp_buffer) {
  auto A = ge::Symbol(10, "A");
  auto R = ge::Symbol(20, "R");
  auto BL = ge::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto ND = ge::Symbol(10, "ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2 = Load("load2", data2).TBuf(Position::kPositionVecOut);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2);
      GE_ASSERT_SUCCESS(
          GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt}, {load1, load2, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  auto data = graph.FindNode("load1");
  data->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(16 * 1024), -1}, {}, reuse_temp_buffer ? 1 : 0});
  return ge::SUCCESS;
}

Status BuildTqueTbufAscendGraph_multi_case_g0(ge::AscGraph &graph) {
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
  auto data3 = graph.CreateContiguousData("input3", DT_FLOAT, {nd});
  auto data4 = graph.CreateContiguousData("input4", DT_FLOAT, {nd});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load_tque0 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tbuf0 = Load("load2", data2).TBuf(Position::kPositionVecIn);
      auto load_tbuf1 = Load("load3", data3).TBuf(Position::kPositionVecIn);
      auto load_tbuf2 = Load("load4", data4).TBuf(Position::kPositionVecIn);
      auto store1 = Store("store1", load_tque0);
      auto store2 = Store("store2", load_tbuf0);
      auto store3 = Store("store2", load_tbuf1);
      auto store4 = Store("store2", load_tbuf2);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes(
          {*ndB, *ndbT, *ndb, *ndbt}, {load_tque0, load_tbuf0, load_tbuf1, load_tbuf2, store1, store2, store3, store4},
          2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
      auto output3 = Output("output2", store3);
      auto output4 = Output("output2", store4);
    }
  }
  auto data_node = graph.FindNode("load1");
  data_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(16 * 1024), -1}, {},0});
  auto data1_node = graph.FindNode("load2");
  data1_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(1024), 0}, {},0});
  data1_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(2*1024), 0}, {},0});
  return ge::SUCCESS;
}

Status BuildTqueTbufAscendGraph_multi_case_g1(ge::AscGraph &graph) {
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");
  auto BL = ge::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto S0 = ge::Symbol("S0");
  auto z0 = graph.CreateAxis("z0", S0);
  auto [z0B, z0b] = graph.BlockSplit(z0.id);
  auto [z0bT, z0bt] = graph.TileSplit(z0b->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z0});
  auto data3 = graph.CreateContiguousData("input3", DT_FLOAT, {z0});
  auto data4 = graph.CreateContiguousData("input4", DT_FLOAT, {z0});
  LOOP(*z0B) {
    LOOP(*z0bT) {
      auto load_tque0 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tque1 = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tque2 = Load("load3", data3).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tbuf0 = Load("load4", data4).TBuf(Position::kPositionVecIn);
      auto store1 = Store("store1", load_tque0);
      auto store2 = Store("store2", load_tque1);
      auto store3 = Store("store2", load_tque2);
      auto store4 = Store("store2", load_tbuf0);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes(
          {*z0B, *z0bT, *z0b, *z0bt}, {load_tque0, load_tque1, load_tque2, load_tbuf0, store1, store2, store3, store4},
          2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
      auto output3 = Output("output2", store3);
      auto output4 = Output("output2", store4);
    }
  }
  auto data_node = graph.FindNode("load1");
  data_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(16 * 1024), 0}, {}, 0});
  return ge::SUCCESS;
}
}
}
}
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

void TestGenConcat_tque_tbuf_test(bool reuse_temp_buffer) {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::vector<ascir::ScheduledResult> schedule_results;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  const std::string kFirstGraphName = "tque_tbuf_case0";
  ascir::AscGraph tque_tbuf_case0(kFirstGraphName.c_str());
  ASSERT_EQ(ge::ascir::cg::BuildTqueTbufAscendGraph_single_case(tque_tbuf_case0, reuse_temp_buffer), ge::SUCCESS);
  tque_tbuf_case0.SetTilingKey(0U);
  schedule_group1.impl_graphs.emplace_back(tque_tbuf_case0);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_results.emplace_back(schedule_result1);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[ge::sym::kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = false;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  ret = autofuse::test::CopyStubFiles(UT_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

void PrintResult(tque_tbuf_case0TilingData& tilingData) {
  std::cout << "====================================================" << std::endl;)";
  if (!reuse_temp_buffer) {
    kRunTilingFuncMainLocal += R"(std::cout << "b0_size"<< " = " << tilingData.get_b0_size() << std::endl;)";
  }
  kRunTilingFuncMainLocal += R"(
  std::cout << "q0_size"<< " = " << tilingData.get_q0_size() << std::endl;
  std::cout << "b1_size"<< " = " << tilingData.get_b1_size() << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  tque_tbuf_case0TilingData tilingData;)";
  if (!reuse_temp_buffer) {
    kRunTilingFuncMainLocal += R"(tilingData.set_b0_size(64);)";
  }
  kRunTilingFuncMainLocal += R"(
  tilingData.set_q0_size(128);
  tilingData.set_b1_size(512);
  PrintResult(tilingData);
  return 0;
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();
  ret = std::system("g++ -ggdb3 -O0 tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(ret, 0);
}

TEST_F(TestGenConcat, tque_tbuf_case0)
{
  TestGenConcat_tque_tbuf_test(false);
}

TEST_F(TestGenConcat, tque_tbuf_reuse_temp_buffer)
{
  TestGenConcat_tque_tbuf_test(true);
}

TEST_F(TestGenConcat, tque_tbuf_case1)
{
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::vector<ascir::ScheduledResult> schedule_results;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  const std::string kFirstGraphName = "case0";
  const std::string kSecondGraphName = "case1";
  ascir::AscGraph graph_0(kFirstGraphName.c_str());
  ascir::AscGraph graph_1(kSecondGraphName.c_str());
  ASSERT_EQ(ge::ascir::cg::BuildTqueTbufAscendGraph_multi_case_g0(graph_0), ge::SUCCESS);
  graph_0.SetTilingKey(0U);
  ASSERT_EQ(ge::ascir::cg::BuildTqueTbufAscendGraph_multi_case_g1(graph_1), ge::SUCCESS);
  graph_1.SetTilingKey(1U);
  schedule_group1.impl_graphs.emplace_back(graph_0);
  schedule_group2.impl_graphs.emplace_back(graph_1);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 1;}").c_str();
  schedule_result2.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
  schedule_results.emplace_back(schedule_result1);
  schedule_results.emplace_back(schedule_result2);

  std::map<std::string, std::string> options;
    std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  options.emplace("enable_score_func", "1");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[ge::sym::kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = false;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  ret = autofuse::test::CopyStubFiles(UT_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

void PrintResult(AscGraph0ScheduleResult0G0TilingData& tilingData0,AscGraph0ScheduleResult1G0TilingData& tilingData1) {
  std::cout << "========================AscGraph0ScheduleResult0G0TilingData============================" << std::endl;
  std::cout << "b0_size"<< " = " << tilingData0.get_b0_size() << std::endl;
  std::cout << "q0_size"<< " = " << tilingData0.get_q0_size() << std::endl;
  std::cout << "b1_size"<< " = " << tilingData0.get_b1_size() << std::endl;
  std::cout << "b2_size"<< " = " << tilingData0.get_b2_size() << std::endl;
  std::cout << "b3_size"<< " = " << tilingData0.get_b3_size() << std::endl;
  std::cout << "========================AscGraph0ScheduleResult1G0TilingData============================" << std::endl;
  std::cout << "b0_size"<< " = " << tilingData1.get_b0_size() << std::endl;
  std::cout << "q0_size"<< " = " << tilingData1.get_q0_size() << std::endl;
  std::cout << "q1_size"<< " = " << tilingData1.get_q1_size() << std::endl;
  std::cout << "q2_size"<< " = " << tilingData1.get_q2_size() << std::endl;
  std::cout << "b3_size"<< " = " << tilingData1.get_b3_size() << std::endl;
}

int main() {
  AscGraph0ScheduleResult0G0TilingData tilingData0;
  AscGraph0ScheduleResult1G0TilingData tilingData1;
  tilingData0.set_b0_size(64);
  tilingData0.set_q0_size(128);
  tilingData0.set_b1_size(512);
  tilingData0.set_b2_size(512);
  tilingData0.set_b3_size(512);

  tilingData1.set_b0_size(64);
  tilingData1.set_q0_size(128);
  tilingData1.set_q1_size(512);
  tilingData1.set_q2_size(512);
  tilingData1.set_b3_size(512);
  PrintResult(tilingData0,tilingData1);
  return 0;
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();
  ret = std::system("g++ -ggdb3 -O0 tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(ret, 0);
}
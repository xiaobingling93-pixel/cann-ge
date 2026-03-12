/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_regbase_perf.h"
namespace att {
namespace ascendcperf_v2 {
RepeatParams CalculateRepeatParams(const std::string &input_dtype, const Expr& cal_count) {
  Expr repeat_elm = kRptSizeFloat;
  auto it = kRptEleMap.find(input_dtype);
  if (it != kRptEleMap.end()) {
    repeat_elm = it->second;
  }
  GE_ASSERT_TRUE(repeat_elm != ge::sym::kSymbolZero, "repeat_elm is [%s].", ge::SymbolicUtils::ToString(repeat_elm).c_str());
  Expr repeat_time = ge::sym::Ceiling(cal_count / repeat_elm);
  return {repeat_elm, repeat_time};
}

/*
===========================================================================
【功能描述】Compare Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  分支判断（依据 sizeof (T) 的值）：
  2.1 若 sizeof (T) == 8：
  2.1.1 计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T) * 2
  2.1.2 计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  2.1.3 定义 T = 32
  2.1.4 循环（共 repeatTime 次）：
  根据 mode 执行对应操作：
    case Eq:
    调用 vf_ins_vcmp_eq * 2
    执行 MaskAnd
    case Ne:
    调用 vf_ins_vcmp_ne * 2
    执行 MaskOr
    case Gt:
    调用 vf_ins_vcmp_eq
    调用 vf_ins_vcmp_gt * 2
    调用 vf_ins_vsel
    case Ge:
    调用 vf_ins_vcmp_eq
    调用 vf_ins_vcmp_ge * 2
    调用 vf_ins_vsel
    case Lt:
    调用 vf_ins_vcmp_eq
    调用 vf_ins_vcmp_lt * 2
    调用 vf_ins_vsel
    case Le:
    调用 vf_ins_vcmp_eq
    调用 vf_ins_vcmp_le * 2
    调用 vf_ins_vsel
  2.2 若 sizeof (T) != 8：
  2.2.1 计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  2.2.2 计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  2.2.3 循环（共 repeatTime 次）：
    调用 vf_ins_vcmp (mode)
===========================================================================
*/
ge::Status CompareSpecificPerf(const std::string compare_mode, const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Compare mode[%s]: node info is %s.", compare_mode.c_str(), node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  Expr repeat_elm = kRptSizeFloat;
  auto it = kRptEleMap.find(node_info.input_dtype[0]);
  if (it != kRptEleMap.end()) {
    repeat_elm = it->second;
  }
  GE_ASSERT_TRUE(repeat_elm != ge::sym::kSymbolZero, "repeat_elm is [%s].", ge::SymbolicUtils::ToString(repeat_elm).c_str());
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  if (node_info.input_dtype[0] == kUInt64 || node_info.input_dtype[0] == kInt64) {
    repeat_elm = repeat_elm * kSymTwo;
    Expr repeat_time = ge::sym::Ceiling(cal_count / repeat_elm);
    GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
           ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
    if (compare_mode == kEq) {
      GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kEq, kFloat32, max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
      GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskAnd, kUInt8, max_latency, all_vf_instruct_cost, repeat_time));
    } else if (compare_mode == kNe) {
      GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kNe, kFloat32, max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
      GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskOr, kUInt8, max_latency, all_vf_instruct_cost, repeat_time));
    } else if (compare_mode == kGt) {
      GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kEq, kFloat32, max_latency, all_vf_instruct_cost, repeat_time));
      GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kGt, kFloat32, max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
      GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskSel, kUInt8, max_latency, all_vf_instruct_cost, repeat_time));
    } else if (compare_mode == kGe) {
      GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kEq, kFloat32, max_latency, all_vf_instruct_cost, repeat_time));
      GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kGe, kFloat32, max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
      GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskSel, kUInt8, max_latency, all_vf_instruct_cost, repeat_time));
    } else if (compare_mode == kLt) {
      GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kEq, kFloat32, max_latency, all_vf_instruct_cost, repeat_time));
      GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLt, kFloat32, max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
      GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskSel, kUInt8, max_latency, all_vf_instruct_cost, repeat_time));
    } else {
      GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kEq, kFloat32, max_latency, all_vf_instruct_cost, repeat_time));
      GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLe, kFloat32, max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
      GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskSel, kUInt8, max_latency, all_vf_instruct_cost, repeat_time));
    }
  } else {
    Expr repeat_time = ge::sym::Ceiling(cal_count / repeat_elm);
    GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
           ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(compare_mode, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  }
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

ge::Status CompareGEPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  return CompareSpecificPerf(kGe, node_info, perf);
}

ge::Status CompareEQPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  return CompareSpecificPerf(kEq, node_info, perf);
}

ge::Status CompareNEPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  return CompareSpecificPerf(kNe, node_info, perf);
}

ge::Status CompareGTPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  return CompareSpecificPerf(kGt, node_info, perf);
}

ge::Status CompareLEPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  return CompareSpecificPerf(kLe, node_info, perf);
}

ge::Status CompareLTPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  return CompareSpecificPerf(kLt, node_info, perf);
}

/*
===========================================================================
【功能描述】Abs Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vabs
===========================================================================
*/
ge::Status AbsPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Abs node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAbs, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Exp Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vexp
===========================================================================
*/
ge::Status ExpPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Exp node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kExp, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Ln Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vln
===========================================================================
*/
ge::Status LnPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Ln node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLn, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Sqrt Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vsqrt
===========================================================================
*/
ge::Status SqrtPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Sqrt node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSqrt, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Rsqrt Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vsqrt
    调用 vf_ins_vdiv
    调用 vf_ins_vsel
===========================================================================
*/
ge::Status RsqrtPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Rsqrt node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSqrt, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDiv, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Div Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vdiv
===========================================================================
*/
ge::Status DivPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Div node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDiv, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Reciprocal Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vdup
    调用 vf_ins_vdiv
===========================================================================
*/
ge::Status ReciprocalPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Reciprocal node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDiv, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Relu Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vrelu
===========================================================================
*/
ge::Status ReluPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Relu node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kRelu, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Max Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vmax
===========================================================================
*/
ge::Status MaxPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Max node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMax, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Min Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vmin
===========================================================================
*/
ge::Status MinPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Min node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMin, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Neg Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vmuls
===========================================================================
*/
ge::Status NegPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Neg node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMuls, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Mean Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vmuls
===========================================================================
*/
ge::Status MeanPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Mean node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr all_vf_instruct_cost = CreateExpr(0);
  Expr max_latency = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMuls, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Add Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vadd
===========================================================================
*/
ge::Status AddPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Add node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAdd, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Sub Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vsub
===========================================================================
*/
ge::Status SubPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Sub node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSub, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Mul Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vmul
===========================================================================
*/
ge::Status MulPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Mul node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMul, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】LeakyRelu Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vlrelu
===========================================================================
*/
ge::Status LeakyReluPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("LeakyRelu node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLeakyRelu, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Cast Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：oneRepSize = VECTOR_REG_WIDTH / (sizeof(src_T) < sizeof(dst_T) ? sizeof(dst_T) : sizeof(src_T))
  计算总迭代次数：repeatTime = Ceil (count / oneRepSize)
  循环（共 repeatTime 次）：
    若 src_type == int32 且 dst_type == half：
      调用 vf_ins_vcvt(float, src_type)
      调用 vf_ins_vcvt(dst_type, float)
    否则：
      调用 vf_ins_vcvt(dst_type, src_type)
===========================================================================
*/
ge::Status CastPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Cast node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  Expr oneRepSize = kRptSizeFloat;
  auto it = kRptEleMap.find(node_info.input_dtype[0]);
  if (it != kRptEleMap.end()) {
    oneRepSize = it->second;
  }
  it = kRptEleMap.find(node_info.output_dtype[0]);
  if (it != kRptEleMap.end()) {
    oneRepSize = ge::sym::Min(oneRepSize, it->second);
  }
  GE_ASSERT_TRUE(oneRepSize != ge::sym::kSymbolZero, "oneRepSize is [%s].", ge::SymbolicUtils::ToString(oneRepSize).c_str());
  Expr repeat_time = ge::sym::Ceiling(cal_count / oneRepSize);
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], oneRepSize is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(oneRepSize).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
  if (node_info.input_dtype[0] == kInt32 && node_info.output_dtype[0] == kFloat16) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCast, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCast, kFloat32, max_latency, all_vf_instruct_cost, repeat_time));
  } else {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCast, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  }
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Sum Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：oneRepSize = VECTOR_REG_WIDTH / sizeof (T)
  分支判断（依据 count 与 oneRepSize 的大小关系）：
  1. 若 count <= oneRepSize：
    调用 ReduceSumCount(1)
  2. 若 count > oneRepSize 且 count <= oneRepSize * oneRepSize：
    计算中间次数：count2 = CeilDiv (count, oneRepSize)
    调用 ReduceSumCount(count2)
    调用 ReduceSumCount(1)
  3. 其他情况（count > oneRepSize * oneRepSize）：
    计算中间次数1：count2 = CeilDiv (count, oneRepSize)
    计算中间次数2：count3 = CeilDiv (count2, oneRepSize)
    调用 ReduceSumCount(count2)
    调用 ReduceSumCount(count3)
    调用 ReduceSumCount(1)

  辅助函数 ReduceSumCount(repeat)：
    循环（共 repeat 次）：
      调用 vf_ins_vcadd
===========================================================================
*/
ge::Status SumPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Sum node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  Expr oneRepSize = kRptSizeFloat;
  auto it = kRptEleMap.find(node_info.input_dtype[0]);
  if (it != kRptEleMap.end()) {
    oneRepSize = it->second;
  }
  GE_ASSERT_TRUE(oneRepSize != ge::sym::kSymbolZero, "oneRepSize is [%s].", ge::SymbolicUtils::ToString(oneRepSize).c_str());
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  Expr repeat = CreateExpr("reduce_sum_node");
  Expr small_repeat = CreateExpr(1U);
  GELOGD("small_repeat is [%s].", ge::SymbolicUtils::ToString(small_repeat).c_str());
  Expr mid_repeat = ge::sym::Ceiling(cal_count / oneRepSize) + CreateExpr(1U);
  GELOGD("mid_repeat is [%s].", ge::SymbolicUtils::ToString(mid_repeat).c_str());
  Expr large_repeat = ge::sym::Ceiling(ge::sym::Ceiling(cal_count / oneRepSize) / oneRepSize)
                      + ge::sym::Ceiling(cal_count / oneRepSize) + CreateExpr(1U);
  GELOGD("large_repeat is [%s].", ge::SymbolicUtils::ToString(large_repeat).c_str());
  std::shared_ptr<IfCase> branch_small = std::make_shared<IfCase>(small_repeat);
  GE_ASSERT_NOTNULL(branch_small);
  std::shared_ptr<IfCase> branch_mid = std::make_shared<IfCase>(mid_repeat);
  std::shared_ptr<IfCase> branch_large = std::make_shared<IfCase>(large_repeat);
  GE_ASSERT_NOTNULL(branch_mid);
  GE_ASSERT_NOTNULL(branch_large);
  std::shared_ptr<IfCase> branch_not_small = std::make_shared<IfCase>
      (CondType::K_LE, cal_count, oneRepSize * oneRepSize, std::move(branch_mid), std::move(branch_large));
  GE_ASSERT_NOTNULL(branch_not_small);
  TernaryOp ternary_op = TernaryOp(CondType::K_LE, cal_count, oneRepSize, std::move(branch_small), std::move(branch_not_small));
  ternary_op.SetVariable(repeat);
  perf.ternary_ops[repeat] = ternary_op;
  GELOGD("cal_count is [%s], oneRepSize is [%s], repeat is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(oneRepSize).c_str(), ge::SymbolicUtils::ToString(repeat).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kVcadd, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】RemovePad Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vsqz
===========================================================================
*/
ge::Status RemovePadPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("RemovePad node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kVsqz, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Where Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  按输入数据类型数量执行向量复制：vf_ins_vdup * node_info.input_dtype.size()
  循环（共 repeatTime 次）：
    调用 vf_ins_vcmps_ne（向量不等于比较）
    调用 vf_ins_vsel（向量选择，按比较结果赋值）
===========================================================================
*/
ge::Status WherePerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Where node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   CreateExpr(node_info.input_dtype.size())));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarNE, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

inline ge::Status ProcessFloatPow(const NodeDetail &node_info, Expr &cal_count, Expr &max_latency, Expr &all_vf_instruct_cost) {
  Expr eleCountPerVL = kRptSizeFloat;
  Expr repeatTimes = ge::sym::Ceiling(cal_count / eleCountPerVL);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(eleCountPerVL).c_str(), ge::SymbolicUtils::ToString(repeatTimes).c_str());
  if (node_info.input_dtype[0] != kFloat32) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCast, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, CreateExpr(kNumTwo)));
  }
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarEQ, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeatTimes * CreateExpr(kNumSix)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarLT, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeatTimes * CreateExpr(kNumTwo)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeatTimes * CreateExpr(kNumSix)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kXor, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeatTimes));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeatTimes * CreateExpr(kNumFive)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAnd, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeatTimes));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kNeg, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeatTimes));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kNe, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeatTimes));
  return ge::SUCCESS;
}

inline ge::Status ProcessIntegerPow(const NodeDetail &node_info, Expr &cal_count, Expr &max_latency, Expr &all_vf_instruct_cost) {
  Expr dataSize = ge::sym::kSymbolZero;
  auto it = kDataTypeSizeMap.find(node_info.input_dtype[0]);
  if (it != kDataTypeSizeMap.end()) {
    dataSize = it->second;
  }
  GE_ASSERT_TRUE(dataSize != ge::sym::kSymbolZero, "dataSize is [%s].", ge::SymbolicUtils::ToString(dataSize).c_str());
  Expr maxLoop = dataSize * CreateExpr(8);
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr eleCountPerVL = params.repeat_elm;
  Expr repeat_times = params.repeat_time;
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(eleCountPerVL).c_str(), ge::SymbolicUtils::ToString(repeat_times).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, CreateExpr(1)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMul, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_times * maxLoop * CreateExpr(kNumTwo)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_times * maxLoop));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAnd, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_times * maxLoop));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarEQ, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_times * maxLoop));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_times * maxLoop));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kVshrs, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_times * maxLoop));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarEQ, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_times * CreateExpr(kNumTwo)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_times));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_times));
  if (node_info.input_dtype[0] == kInt8 || node_info.input_dtype[0] == kInt16 || node_info.input_dtype[0] == kInt32) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarNE, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_times));
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarLT, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_times));
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_times));
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_times));
  }
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Pow Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  分支判断（依据数据类型分类处理）：
  1. 若为浮点类型（float, half, bfloat16）：
      计算单次处理元素数：eleCountPerVL = 256 / sizeof(float)
      计算总迭代次数：repeatTimes = Ceil (count / eleCountPerVL)
      循环（共 repeatTimes 次）：
        若输入数据类型非 float32：
          调用 vf_ins_vcvt * 2
        调用 vf_ins_vcmps_eq * 6
        调用 vf_ins_vcmps_lt * 2
        调用 vf_ins_vdup * 6
        调用 vf_ins_vxor
        调用 vf_ins_vsel * 5
        调用 vf_ins_vand
        调用 vf_ins_vneg
        调用 vf_ins_vcmp_ne

  2. 若为整数类型（uint8, int8, uint16, int16, uint32, int32）：
      计算最大循环次数：maxLoop = sizeof(T) * 8
      定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
      计算单次处理元素数：eleCountPerVL = VECTOR_REG_WIDTH / sizeof(T)
      计算总迭代次数：repeatTime = Ceil (count / eleCountPerVL)
      调用 vf_ins_vdup
      循环（共 repeatTime 次）：
        内层循环（共 maxLoop 次）：
          调用 vf_ins_vmul * 2
          调用 vf_ins_vdup
          调用 vf_ins_vand
          调用 vf_ins_vcmps_eq
          调用 vf_ins_vsel
          调用 vf_ins_vshrs
        调用 vf_ins_vcmps_eq * 2
        调用 vf_ins_vdup
        调用 vf_ins_vsel
        若为有符号整数类型（int8, int16, int32）：
          调用 vf_ins_vcmps_ne
          调用 vf_ins_vcmps_lt
          调用 vf_ins_vdup
          调用 vf_ins_vsel
===========================================================================
*/
ge::Status PowPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Pow node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  if (node_info.input_dtype[0] == kFloat16 || node_info.input_dtype[0] == kFloat32 || node_info.input_dtype[0] == kBfloat16) {
    GE_ASSERT_SUCCESS(ProcessFloatPow(node_info, cal_count, max_latency, all_vf_instruct_cost));
  } else {
    GE_ASSERT_SUCCESS(ProcessIntegerPow(node_info, cal_count, max_latency, all_vf_instruct_cost));
  }
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Erf Regbase版本逻辑说明
【输入】dst(输出张量), src(输入张量), cal_count(数据量)
【伪代码】
1. 计算单次处理元素数：repeat_elm = 256 / sizeof B32
2. 计算总迭代次数：repeat_time = 向上取整(cal_count / repeat_elm)
3. 主循环（共repeat_time次）：
   vf_ins_plt UpdateMask
   vf_ins_datacopy_ub2reg
   分支：(T = half)
     vf_ins_vcvt(float, half)
   vf_ins_vmins
   vf_ins_vmaxs
   vf_ins_vmul * 11
   vf_ins_vmuls
   vf_ins_vadds * 10
   vf_ins_vdiv
   分支：(T = half)
     vf_ins_vcvt(half, float)
   vf_ins_datacopy_reg2ub
===========================================================================
*/
ge::Status ErfPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr cal_count = node_info.input_dims[kNumZero];
  Expr repeat_elm = kRptSizeFloat;
  Expr repeat_time = ge::sym::Ceiling(cal_count / repeat_elm);
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("Erf node[%s], cal_count is [%s], repeat_elm is [%s], repeat_time is [%s], max_latency is [%s].",
         node_info.ToString().c_str(), ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str(),
         ge::SymbolicUtils::ToString(max_latency).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  if (node_info.input_dtype[0] == kFloat16) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCast, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  }
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMins, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaxs, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMul, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time * kSymEleven));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMuls, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAdds, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time * kSymTen));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDiv, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  if (node_info.input_dtype[0] == kFloat16) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCast, kFloat32, max_latency, all_vf_instruct_cost, repeat_time));
  }
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Tanh Regbase版本逻辑说明
【输入】dst(输出张量), src(输入张量), cal_count(数据量)
【伪代码】
1. 计算单次处理元素数：repeat_elm = 256 / sizeof B32
2. 计算总迭代次数：repeat_time = 向上取整(cal_count / repeat_elm)
3. 主循环（共repeat_time次）：
   vf_ins_plt UpdateMask
   vf_ins_datacopy_ub2reg
   分支：(T = half)
     vf_ins_vcvt(float, half)
   vf_ins_vmins
   vf_ins_vmaxs
   vf_ins_vmuls
   vf_ins_vexp
   vf_ins_vadds * 2
   vf_ins_vdiv
   分支：(T = half)
     vf_ins_vcvt(half, float)
   vf_ins_datacopy_reg2ub
===========================================================================
*/
ge::Status TanhPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Tanh node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  Expr repeat_elm = kRptSizeFloat;
  Expr repeat_time = ge::sym::Ceiling(cal_count / repeat_elm);
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("Tanh node[%s], cal_count is [%s], repeat_elm is [%s], repeat_time is [%s], max_latency is [%s].",
         node_info.ToString().c_str(), ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str(),
         ge::SymbolicUtils::ToString(max_latency).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  if (node_info.input_dtype[0] == kFloat16) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCast, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  }
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMins, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaxs, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMuls, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kExp, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAdds, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDiv, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  if (node_info.input_dtype[0] == kFloat16) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCast, kFloat32, max_latency, all_vf_instruct_cost, repeat_time));
  }
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Sigmoid Regbase版本逻辑说明
【输入】dst(输出张量), src(输入张量), cal_count(数据量)
【伪代码】
1. 计算单次处理元素数：repeat_elm = 256 / sizeof B32
2. 计算总迭代次数：repeat_time = 向上取整(cal_count / repeat_elm)
3. 主循环（共repeat_time次）：
   vf_ins_plt UpdateMask
   vf_ins_datacopy_ub2reg
   分支：(T = half)
     vf_ins_vcvt(float, half)
   vf_ins_vmuls
   vf_ins_vexp
   vf_ins_vadds
   vf_ins_vdup
   vf_ins_vdiv
   分支：(T = half)
     vf_ins_vcvt(half, float)
   vf_ins_datacopy_reg2ub
===========================================================================
*/
ge::Status SigmoidPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Sigmoid node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  Expr repeat_elm = kRptSizeFloat;
  Expr repeat_time = ge::sym::Ceiling(cal_count / repeat_elm);
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("Sigmoid node[%s], cal_count is [%s], repeat_elm is [%s], repeat_time is [%s], max_latency is [%s].",
         node_info.ToString().c_str(), ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str(),
         ge::SymbolicUtils::ToString(max_latency).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  if (node_info.input_dtype[0] == kFloat16) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCast, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  }
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMuls, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kExp, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAdds, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDiv, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  if (node_info.input_dtype[0] == kFloat16) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCast, kFloat32, max_latency, all_vf_instruct_cost, repeat_time));
  }
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Gelu Regbase版本逻辑说明
【输入】dst(输出张量), src(输入张量), cal_count(数据量)
【伪代码】
1. 计算单次处理元素数：repeat_elm = 256 / sizeof T
2. 计算总迭代次数：repeat_time = 向上取整(cal_count / repeat_elm)
3. 主循环（共repeat_time次）：
   vf_ins_plt UpdateMask
   vf_ins_datacopy_ub2reg
   vf_ins_vmul * 3
   vf_ins_vmuls * 3
   vf_ins_vadd
   vf_ins_vmins
   vf_ins_vexp * 2
   vf_ins_vabs
   vf_ins_vadds
   vf_ins_vdiv
   vf_ins_datacopy_reg2ub
===========================================================================
*/
ge::Status GeluPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_elm = params.repeat_elm;
  Expr repeat_time = params.repeat_time;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("Gelu node[%s], cal_count is [%s], repeat_elm is [%s], repeat_time is [%s], max_latency is [%s].",
         node_info.ToString().c_str(), ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str(),
         ge::SymbolicUtils::ToString(max_latency).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMul, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time * kSymThree));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMuls, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time * kSymThree));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAdd, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMins, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kExp, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAbs, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAdds, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDiv, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】Sign Regbase版本逻辑说明
【输入】dst(输出张量), src(输入张量), cal_count(数据量)
【伪代码】
1. 计算单次处理元素数：repeat_elm = 256 / sizeof T
2. 计算总迭代次数：repeat_time = 向上取整(cal_count / repeat_elm)
3. 初始化：vf_ins_vdup * 3
4. 主循环（共repeat_time次）：
   vf_ins_plt UpdateMask
   vf_ins_datacopy_ub2reg
   vf_ins_vcmps_lt
   vf_ins_vcmps_gt
   vf_ins_vsel * 2
   vf_ins_datacopy_reg2ub
===========================================================================
*/
ge::Status SignPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_elm = params.repeat_elm;
  Expr repeat_time = params.repeat_time;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("Sign node[%s], cal_count is [%s], repeat_elm is [%s], repeat_time is [%s], max_latency is [%s].",
         node_info.ToString().c_str(), ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str(),
         ge::SymbolicUtils::ToString(max_latency).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, kSymThree));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarLT, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarGT, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】LogicalNot Regbase版本逻辑说明
【输入】dst(输出张量), src(输入张量), cal_count(数据量)
【伪代码】
1. 计算单次处理元素数：repeat_elm = 256 / sizeof T
2. 计算总迭代次数：repeat_time = 向上取整(cal_count / repeat_elm)
3. 初始化：vf_ins_vdup * 2
4. 主循环（共repeat_time次）：
   vf_ins_plt UpdateMask
   vf_ins_datacopy_ub2reg
   vf_ins_vcmps_eq
   分支：(sizeof(T) = 2)
     vf_ins_ppack MaskPack * 2
   分支：(其他)
     vf_ins_ppack MaskPack * 4
   vf_ins_vsel
   vf_ins_datacopy_reg2ub
===========================================================================
*/
ge::Status LogicalNotPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("LogicalNot node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_elm = params.repeat_elm;
  Expr repeat_time = params.repeat_time;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("LogicalNot node[%s], cal_count is [%s], repeat_elm is [%s], repeat_time is [%s], max_latency is [%s].",
         node_info.ToString().c_str(), ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str(),
         ge::SymbolicUtils::ToString(max_latency).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarEQ, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  if (node_info.input_dtype[0] == kFloat16 || node_info.input_dtype[0] == kBfloat16 || node_info.input_dtype[0] == kUInt16
  || node_info.input_dtype[0] == kInt16) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskPack, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
  } else {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskPack, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time * kSymFour));
  }
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】LogicalAnd/Or Regbase版本逻辑说明
【输入】dst(输出张量), src(输入张量), cal_count(数据量)
【伪代码】
1. 计算单次处理元素数：repeat_elm = 256 / sizeof T
2. 计算总迭代次数：repeat_time = 向上取整(cal_count / repeat_elm)
3. 初始化：vf_ins_vdup * 2
4. 主循环（共repeat_time次）：
   vf_ins_plt UpdateMask
   vf_ins_datacopy_ub2reg * 2
   vf_ins_vcmps_ne * 2
   MaskAnd/MaskOr
   分支：(sizeof(T) = 2)
     vf_ins_ppack MaskPack * 2
   分支：(其他)
     vf_ins_ppack MaskPack * 4
   vf_ins_vsel
   vf_ins_datacopy_reg2ub
===========================================================================
*/
inline ge::Status LogicalAndOrImpl(const std::string &type, const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_elm = params.repeat_elm;
  Expr repeat_time = params.repeat_time;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("LogicalAndOrImpl node[%s], cal_count is [%s], repeat_elm is [%s], repeat_time is [%s], max_latency is [%s].",
         node_info.ToString().c_str(), ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str(),
         ge::SymbolicUtils::ToString(max_latency).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarNE, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
  if (type == kMaskAnd) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskAnd, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  } else {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskOr, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  }
  if (node_info.input_dtype[0] == kFloat16 || node_info.input_dtype[0] == kBfloat16 ||
      node_info.input_dtype[0] == kUInt16 || node_info.input_dtype[0] == kInt16) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskPack, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
  } else {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskPack, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time * kSymFour));
  }
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

ge::Status LogicalOrPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  return LogicalAndOrImpl(kMaskOr, node_info, perf);
}

ge::Status LogicalAndPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  return LogicalAndOrImpl(kMaskAnd, node_info, perf);
}

/*
===========================================================================
【功能描述】ClipByValue Regbase版本逻辑说明
【输入】dst(输出张量), src(输入张量), cal_count(数据量)
【伪代码】
1. 计算单次处理元素数：repeat_elm = 256 / sizeof T
2. 计算总迭代次数：repeat_time = 向上取整(cal_count / repeat_elm)
3. 主循环（共repeat_time次）：
   vf_ins_plt UpdateMask
   vf_ins_datacopy_ub2reg * 3
   vf_ins_vmax
   vf_ins_vmin
   vf_ins_datacopy_reg2ub
===========================================================================
*/
ge::Status ClipByValuePerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_elm = params.repeat_elm;
  Expr repeat_time = params.repeat_time;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("ClipByValue node[%s], cal_count is [%s], repeat_elm is [%s], repeat_time is [%s], max_latency is [%s].",
         node_info.ToString().c_str(), ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str(),
         ge::SymbolicUtils::ToString(max_latency).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time * kSymThree));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMax, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMin, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】BitwiseAnd Regbase版本逻辑说明
【输入】dst(输出张量), src(输入张量), cal_count(数据量)
【伪代码】
1. 计算单次处理元素数：repeat_elm = 256 / sizeof T
2. 计算总迭代次数：repeat_time = 向上取整(cal_count / repeat_elm) / 2
3. 主循环（共repeat_time次）：
   vf_ins_plt UpdateMask * 2
   vf_ins_datacopy_ub2reg * 4
   vf_ins_vand * 2
   vf_ins_datacopy_reg2ub * 2
4. 后续
   vf_ins_plt UpdateMask
   vf_ins_datacopy_ub2reg * 2
   vf_ins_vand
   vf_ins_datacopy_reg2ub
===========================================================================
*/
ge::Status BitwiseAndPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("BitwiseAnd node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_elm = params.repeat_elm;
  Expr repeat_time = params.repeat_time / kSymTwo;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(cal_count).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time * kSymFour));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAnd, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, kSymOne));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAnd, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, kSymOne));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, kSymOne));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}

/*
===========================================================================
【功能描述】FloorDiv Regbase版本逻辑说明
【输入】dst(输出张量), src(输入张量), size(数据量)
【伪代码】
1. 计算单次处理元素数：repeat_elm = 256 / sizeof T
2. 计算总迭代次数：repeat_time = 向上取整(size / repeat_elm)
3. 初始化：vf_ins_vdup * 2
4. 主循环（共repeat_time次）：
   vf_ins_plt UpdateMask
   vf_ins_datacopy_ub2reg * 2
   vf_ins_vdiv
   vf_ins_vtrc
   vf_ins_vcmp_eq
   vf_ins_vmul
   vf_ins_vmula
   vf_ins_vcmp_gt
   vf_ins_vadd
   vf_ins_vsel * 2
   vf_ins_datacopy_reg2ub
===========================================================================
*/
ge::Status FloorDivPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("FloorDiv node info is %s.", node_info.ToString().c_str());
  Expr size = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], size);
  Expr repeat_elm = params.repeat_elm;
  Expr repeat_time = params.repeat_time;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("size is [%s], repeat_elm is [%s], repeat_time is [%s].", ge::SymbolicUtils::ToString(size).c_str(),
         ge::SymbolicUtils::ToString(repeat_elm).c_str(), ge::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDiv, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kTruncate, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kEq, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMul, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMulAddDst, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kGt, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAdd, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return ge::SUCCESS;
}
}
}  // namespace att

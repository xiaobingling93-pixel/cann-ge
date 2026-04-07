/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_api_call_base.h"
#include "reduce_api_call.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"
#include "api_call/utils/api_call_utils.h"

namespace codegen {
using namespace std;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace ascgen_utils;
using namespace reduce_base;

int64_t ReduceApiCall::GetTmpBufIdByLifeTime(int64_t life_time, const std::string &api_name) const {
  auto it = this->tmp_buf_id.find(life_time);
  GE_ASSERT_TRUE(it != this->tmp_buf_id.end(),
                 "ReduceApiCall(%s) cannot find tmp buffer id for life_time=%ld.", api_name.c_str(), life_time);
  return it->second;
}

Status ReduceApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                               const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                               const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                               std::string &result) const {
  auto iter = reduce_type_map.find(this->api_name_);
  GE_CHK_BOOL_RET_STATUS(iter != reduce_type_map.end(), ge::FAILED, "Codegen unsupported reduce api::%s", this->api_name_.c_str());
  auto &[type_value, instr_type] = iter->second;

  auto x = inputs[0].get();
  auto y = outputs[0].get();

  // 获取tmp_buf复用TBuf的id
  int64_t id = GetTmpBufIdByLifeTime(-1L, this->api_name_);

  std::string reduce_pattern;
  GetIsArAndPattern(y, x.isAr, reduce_pattern);

  // 获取dtype_name：ArgMax系列算子使用value类型（x.dtype），其他算子使用输出类型（y.dtype）
  std::string dtype_name;
  GE_CHK_STATUS_RET(GetDtypeNameForReduce(this->api_name_, x, y, dtype_name),
                    "Codegen get dtype name failed for api:%s", this->api_name_.c_str());

  stringstream ss;

  ReduceMergedSizeCodeGen(tpipe, ss, x, y);

  ReduceDimACodeGen(x, this->api_name_, ss);

  ReduceInitCodeGen(x, y, type_value, ss, tpipe, dtype_name);

  // 生成accumulated_offset变量声明（ArgMax和ArgMaxMultiRPhase1需要）
  GenAccumulatedOffsetDeclForArgMax(this->api_name_, x, y, tpipe, ss);

  ss << "uint32_t tmp_reduce_shape[] = {first_actual, last};" << std::endl;

  std::string new_api_name = this->api_name_ == "Mean" ? "Sum" : this->api_name_;
  if (!IsNeedMultiReduce(tpipe.tiler, x, y, current_axis.back())) {
    if (new_api_name == "ArgMax") {
      ss << "ArgMaxExtend<int64_t, " << dtype_name << ", " << reduce_pattern << ">("
         << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], "
         << x << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], "
         << tpipe.tmp_buf << "_" << std::to_string(id) << ", tmp_reduce_shape, false);" << std::endl;
    } else if (new_api_name == "Sum" && dtype_name == "int32_t") {
      ss << "ReduceSumInt32<" << dtype_name << ", " << reduce_pattern << ", false>("
         << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], "
         << x << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], "
         << tpipe.tmp_buf << "_" << std::to_string(id) <<", tmp_reduce_shape, true);" << std::endl;
    } else {
      ss << "Reduce" << new_api_name << "<" << dtype_name << ", " << reduce_pattern << ", false>("
         << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], "
         << x << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], "
         << tpipe.tmp_buf << "_" << std::to_string(id) <<", tmp_reduce_shape, true);" << std::endl;
    }
    if (this->api_name_== "Mean") {
      ReduceMeanCodeGen(dtype_name, tpipe, y, ss);
    }
  } else {
    int64_t tmp_lifetime_0_id = GetTmpBufIdByLifeTime(0L, this->api_name_);

    if (new_api_name == "ArgMax") {
      // ArgMax 特殊处理：需要维护全局最大值和索引，以及累加的 offset
      // ArgMax 有三个额外的tmp_buf：
      //   - desc2(life_time=0)：索引的临时存储
      //   - desc3(life_time=1)：当前迭代的value临时存储
      //   - desc4(life_time=2)：value的历史最大结果

      // 索引：生命周期0的 tmp_argmax_index (desc2)
      ss << "LocalTensor<int64_t> tmp_argmax_index;" << std::endl;
      ss << "tmp_argmax_index = " << tpipe.tmp_buf << "_" << std::to_string(tmp_lifetime_0_id)
         << ".template ReinterpretCast<int64_t>();" << std::endl;

      // 当前计算的value：生命周期1的 tmp_argmax_value (desc3)
      int64_t tmp_lifetime_1_id = GetTmpBufIdByLifeTime(1L, "ArgMax");
      ss << "LocalTensor<" << dtype_name << "> tmp_argmax_value;" << std::endl;
      ss << "tmp_argmax_value = " << tpipe.tmp_buf << "_" << std::to_string(tmp_lifetime_1_id)
         << ".template ReinterpretCast<" << dtype_name << ">();" << std::endl;

      // 历史最大value：生命周期2的 tmp_argmax_value_saved (desc4)
      int64_t tmp_lifetime_2_id = GetTmpBufIdByLifeTime(2L, "ArgMax");
      ss << "LocalTensor<" << dtype_name << "> tmp_argmax_value_saved;" << std::endl;
      ss << "tmp_argmax_value_saved = " << tpipe.tmp_buf << "_" << std::to_string(tmp_lifetime_2_id)
         << ".template ReinterpretCast<" << dtype_name << ">();" << std::endl;

      // 调用 ArgMaxExtend 获取本次迭代的局部索引
      ss << "ArgMaxExtend<int64_t, " << dtype_name << ", " << reduce_pattern << ">("
         << "tmp_argmax_index[0], "
         << x << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], "
         << tpipe.tmp_buf << "_" << std::to_string(id) << ", tmp_reduce_shape, false);" << std::endl;

      // 调用 ReduceMax 获取本次迭代的局部最大值
      ss << "ReduceMax<" << dtype_name << ", " << reduce_pattern << ", false>("
         << "tmp_argmax_value[0], "
         << x << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], "
         << tpipe.tmp_buf << "_" << std::to_string(id) << ", tmp_reduce_shape, true);" << std::endl;

      ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;

      // 如果是第一次迭代，直接赋值；否则使用 UpdateMaxIndexAndValue 更新全局最大值和索引
      ss << "uint32_t temp_size_index = " << KernelUtils::SizeAlign() << "(" << y.actual_size << ", 4);" << std::endl;
      ss << "uint32_t temp_size_value = " << KernelUtils::SizeAlign() << "(" << y.actual_size << ", 32/sizeof(" << dtype_name << "));" << std::endl;
      ss << "if (" << tpipe.tiler.GetAxis(current_axis.back()) << " == 0) {" << std::endl;
      ss << "DataCopyExtend(" << y << "[0], " << "tmp_argmax_index[0], " << "temp_size_index);" << std::endl;
      ss << "DataCopyExtend(" << "tmp_argmax_value_saved[0], " << "tmp_argmax_value[0], temp_size_value);" << std::endl;
      ss << "} else {" << std::endl;
      // 使用当前的 accumulated_offset（标量）来更新全局最大值和索引
      // tmp_argmax_value_current是当前计算的value，tmp_argmax_value_saved是历史最大
      ss << "UpdateMaxIndexAndValue<" << dtype_name << ">(tmp_argmax_index[0], tmp_argmax_value[0], "
         << y << "[0], " << "tmp_argmax_value_saved[0], "
         << "accumulated_offset, " << tpipe.tmp_buf << "_" << std::to_string(id) << ", temp_size_value);" << std::endl;
      ss << "}" << std::endl;

      // 累加 offset：accumulated_offset += 本次处理的 R 轴 actual_size
      // 根据 AR/RA 模式决定累加哪个值
      if (x.isAr) {
        // AR 模式：累加 vectorized_axis 的 actual_size
        ss << "accumulated_offset += " << tpipe.tiler.GetAxis(x.vectorized_axis.back()).actual_size << ";" << std::endl;
      } else {
        // RA 模式：累加 first_actual
        ss << "accumulated_offset += first_actual;" << std::endl;
      }
    } else if (new_api_name == "ArgMaxMultiRPhase1") {
      // ArgMaxMultiRPhase1特殊处理：在IsNeedMultiReduce分支中也需要处理
      // ArgMaxMultiRPhase1 有两个额外的tmp_buf：
      //   - desc2(life_time=0)：索引的临时存储
      //   - desc3(life_time=1)：当前迭代的value临时存储
      // 注意：ArgMaxMultiRPhase1本身自带两个输出，所以不需要历史最大值的tmp_buf

      // 索引：生命周期0的 tmp_argmax1_index (desc2)
      ss << "LocalTensor<int64_t> tmp_argmax1_index;" << std::endl;
      ss << "tmp_argmax1_index = " << tpipe.tmp_buf << "_" << std::to_string(tmp_lifetime_0_id)
         << ".template ReinterpretCast<int64_t>();" << std::endl;

      // 当前value：生命周期1的 tmp_argmax1_value (desc3)
      int64_t tmp_lifetime_1_id = GetTmpBufIdByLifeTime(1L, "ArgMaxMultiRPhase1");
      ss << "LocalTensor<" << dtype_name << "> tmp_argmax1_value;" << std::endl;
      ss << "tmp_argmax1_value = " << tpipe.tmp_buf << "_" << std::to_string(tmp_lifetime_1_id)
         << ".template ReinterpretCast<" << dtype_name << ">();" << std::endl;

      // 调用 ArgMaxExtend 获取本次迭代的局部索引
      ss << "ArgMaxExtend<int64_t, " << dtype_name << ", " << reduce_pattern << ">("
         << "tmp_argmax1_index[0], "
         << x << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], "
         << tpipe.tmp_buf << "_" << std::to_string(id) << ", tmp_reduce_shape, false);" << std::endl;

      // 调用 ReduceMax 获取本次迭代的局部最大值
      ss << "ReduceMax<" << dtype_name << ", " << reduce_pattern << ", false>("
         << "tmp_argmax1_value[0], "
         << x << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], "
         << tpipe.tmp_buf << "_" << std::to_string(id) << ", tmp_reduce_shape, true);" << std::endl;

      ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;

      // ArgMaxMultiRPhase1有两个输出：
      //   - outputs[0]: value
      //   - outputs[1]: index
      GE_ASSERT_TRUE(outputs.size() >= 2, "ArgMaxMultiRPhase1 requires at least 2 outputs.");
      auto y_value = outputs[0].get();
      auto y_index = outputs[1].get();

      // 如果是第一次迭代，直接赋值；否则使用UpdateMaxIndexAndValue更新全局最大值和索引
      ss << "uint32_t temp_size_index = " << KernelUtils::SizeAlign() << "(" << y_index.actual_size << ", 4);" << std::endl;
      ss << "uint32_t temp_size_value = " << KernelUtils::SizeAlign() << "(" << y_value.actual_size << ", 32/sizeof(" << dtype_name << "));" << std::endl;
      ss << "if (" << tpipe.tiler.GetAxis(current_axis.back()) << " == 0) {" << std::endl;
      // 第一次迭代：直接复制到输出
      ss << "DataCopyExtend(" << y_value << "[0], tmp_argmax1_value[0], temp_size_value);" << std::endl;
      ss << "DataCopyExtend(" << y_index << "[0], tmp_argmax1_index[0], temp_size_index);" << std::endl;
      ss << "} else {" << std::endl;
      // 后续迭代：使用UpdateMaxIndexAndValue更新全局最大值和索引
      // 注意：这里需要offset，offset = 当前核id * R轴每块大小 + 累加的offset
      // 暂时传入accumulated_offset，需要在循环外初始化
      ss << "UpdateMaxIndexAndValue<" << dtype_name << ">(tmp_argmax1_index[0], tmp_argmax1_value[0], "
         << y_index << "[0], " << y_value << "[0], "
         << "accumulated_offset + block_dim * r_axis_block_size, " << tpipe.tmp_buf << "_" << std::to_string(id) << ", temp_size_value);" << std::endl;
      ss << "}" << std::endl;

      // 累加 offset：accumulated_offset += 本次处理的 R 轴 actual_size
      // 根据 AR/RA 模式决定累加哪个值
      if (x.isAr) {
        // AR 模式：累加 vectorized_axis 的 actual_size
        ss << "accumulated_offset += " << tpipe.tiler.GetAxis(x.vectorized_axis.back()).actual_size << ";" << std::endl;
      } else {
        // RA 模式：累加 first_actual
        ss << "accumulated_offset += first_actual;" << std::endl;
      }
    } else if (new_api_name == "ArgMaxMultiRPhase2") {
      // ArgMaxMultiRPhase2特殊处理：有两个输入和一个输出，需要处理多次迭代
      //   - inputs[0]: value (来自Phase1的value输出，是该块的最大值)
      //   - inputs[1]: index (来自Phase1的index输出，是该块最大值的位置)
      //   - outputs[0]: 最终的index输出
      // 注意：Phase2也是R轴分核，需要调用ArgmaxExtend和ReduceMax

      GE_ASSERT_TRUE(inputs.size() >= 2, "ArgMaxMultiRPhase2 requires at least 2 inputs.");
      GE_ASSERT_TRUE(outputs.size() >= 1, "ArgMaxMultiRPhase2 requires at least 1 output.");
      auto x_value = inputs[0].get();
      auto x_index = inputs[1].get();

      // 索引：生命周期0的第一个 tmp_argmax2_index
      ss << "LocalTensor<int64_t> tmp_argmax2_index;" << std::endl;
      ss << "tmp_argmax2_index = " << tpipe.tmp_buf << "_" << std::to_string(tmp_lifetime_0_id)
         << ".template ReinterpretCast<int64_t>();" << std::endl;

      // 当前计算的value：生命周期1的 tmp_argmax2_value (desc3)
      int64_t tmp_lifetime_1_id = GetTmpBufIdByLifeTime(1L, "ArgMaxMultiRPhase2");
      ss << "LocalTensor<" << dtype_name << "> tmp_argmax2_value;" << std::endl;
      ss << "tmp_argmax2_value = " << tpipe.tmp_buf << "_" << std::to_string(tmp_lifetime_1_id)
         << ".template ReinterpretCast<" << dtype_name << ">();" << std::endl;

      // 历史最大value：生命周期2的 tmp_argmax2_value_saved (desc4)
      int64_t tmp_lifetime_2_id = GetTmpBufIdByLifeTime(2L, "ArgMaxMultiRPhase2");
      ss << "LocalTensor<" << dtype_name << "> tmp_argmax2_value_saved;" << std::endl;
      ss << "tmp_argmax2_value_saved = " << tpipe.tmp_buf << "_" << std::to_string(tmp_lifetime_2_id)
         << ".template ReinterpretCast<" << dtype_name << ">();" << std::endl;

      // 调用 ArgMaxExtend 获取本次迭代的局部索引
      ss << "ArgMaxExtend<int64_t, " << dtype_name << ", " << reduce_pattern << ">("
         << "tmp_argmax2_index[0], "
         << x_value << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x_value) << "], "
         << tpipe.tmp_buf << "_" << std::to_string(id) << ", tmp_reduce_shape, false);" << std::endl;

      // 调用 ReduceMax 获取本次迭代的局部最大值
      ss << "ReduceMax<" << dtype_name << ", " << reduce_pattern << ", false>("
         << "tmp_argmax2_value[0], "
         << x_value << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x_value) << "], "
         << tpipe.tmp_buf << "_" << std::to_string(id) << ", tmp_reduce_shape, true);" << std::endl;

      ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;

      // 如果是第一次迭代，直接赋值；否则使用UpdateMaxIndexAndValue更新全局最大值和索引
      ss << "uint32_t temp_size_index = " << KernelUtils::SizeAlign() << "(" << y.actual_size << ", 4);" << std::endl;
      ss << "uint32_t temp_size_value = " << KernelUtils::SizeAlign() << "(" << y.actual_size << ", 32/sizeof(" << dtype_name << "));" << std::endl;
      ss << "if (" << tpipe.tiler.GetAxis(current_axis.back()) << " == 0) {" << std::endl;
      ss << "DataCopyExtend(" << y << "[0], tmp_argmax2_index[0], temp_size_index);" << std::endl;
      ss << "DataCopyExtend(" << "tmp_argmax2_value_saved[0], " << "tmp_argmax2_value[0], temp_size_value);" << std::endl;
      ss << "} else {" << std::endl;
      // 使用UpdateMaxIndexAndValue更新，注意这里offset传入0（因为Phase1已经处理了offset）
      ss << "UpdateMaxIndexAndValue<" << dtype_name << ">(tmp_argmax2_index[0], tmp_argmax2_value[0], "
         << y << "[0], " << "tmp_argmax2_value_saved[0], "
         << "0, " << tpipe.tmp_buf << "_" << std::to_string(id) << ", temp_size_value);" << std::endl;
      ss << "}" << std::endl;
    } else {
      ss << "LocalTensor<" << dtype_name << "> tmp_reduce;" << std::endl;
      ss << "tmp_reduce = " << tpipe.tmp_buf << "_" << std::to_string(tmp_lifetime_0_id) << ".template ReinterpretCast<" << dtype_name << ">();" << std::endl;
      if (new_api_name == "Sum" && dtype_name == "int32_t") {
        ss << "ReduceSumInt32<" << dtype_name << ", " << reduce_pattern << ", false>"
           << "(tmp_reduce[0], " << x << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], "
           << tpipe.tmp_buf << "_" << std::to_string(id) << ", tmp_reduce_shape, true);" << std::endl;
      } else {
        ss << "Reduce" << new_api_name << "<" << dtype_name << "," << reduce_pattern << ", false>"
           << "(tmp_reduce[0], " << x << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], "
           << tpipe.tmp_buf << "_" << std::to_string(id) << ", tmp_reduce_shape, true);" << std::endl;
      }
      ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;
      ss << "uint32_t temp_size = " << KernelUtils::SizeAlign() << "(" << y.actual_size << ", 32/sizeof(" << dtype_name << "));" << std::endl;
      ss << "if (" << tpipe.tiler.GetAxis(current_axis.back()) << " == 0) {" << std::endl;
      ss << "DataCopyExtend(" << y << "[0], " << "tmp_reduce[0], " << "temp_size);" << std::endl;
      ss << "} else {" << std::endl;
      ss << "AscendC::" << instr_type << "(" << y << "[0], " << "tmp_reduce[0], " << y << "[0], temp_size);\n"
         << "}" << std::endl;
    }
  }

  ss << "}" << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

static ApiCallRegister<ReduceApiCall> register_reduce_api_call("ReduceApiCall");

}  // namespace codegen
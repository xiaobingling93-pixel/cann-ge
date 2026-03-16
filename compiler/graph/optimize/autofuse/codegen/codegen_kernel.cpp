/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_kernel.h"

#include <sstream>
#include <string>
#include <functional>
#include <stack>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"
#include "ascir_utils.h"
#include "backend/backend_spec.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "ascendc_api_registry.h"
#include "optimize/platform/platform_factory.h"
#include "common/platform_context.h"

using namespace std;
using namespace ge::ops;
using namespace codegen;
using namespace ge::ascir_op;
using namespace ascgen_utils;

namespace {
constexpr size_t kDoubleAxisSize = 2U;
constexpr size_t kDoubleTileAxisSize = 2;
constexpr uint64_t kStrideOne = 1U;
const std::string kEnCacheOriginBroadcastAxis = "enable_cache_origin_brc_axis";
const std::string kEnCacheFusedBroadcastAxis = "enable_cache_fused_brc_axis";
const std::string kEnCacheA = "dis_enable_cache_a";
const std::string kEnCacheR = "dis_enable_cache_r";
constexpr uint32_t kFuncIdBegin = 20000000U;
constexpr const char kInputTensorDescName[] = "input_tensor_desc";
constexpr const char kOutputTensorDescName[] = "output_tensor_desc";
const std::string kKernelTaskTypeAIVOnly = "KERNEL_TYPE_AIV_ONLY";
const std::string kKernelTaskTypeMixAIVOneZero = "KERNEL_TYPE_MIX_AIV_1_0";
}  // namespace

std::ostream &operator<<(std::ostream &os, const Code &obj) {
  return os << obj.Str();
}

Type::Type(const string &type_name) : name(type_name) {}

std::string Type::Str() const {
  return name;
}

Variable::Variable(const Type &var_type, const string &var_name) : type(var_type), name(var_name) {}

std::string Variable::Str() const {
  return name;
}

std::string Variable::AsArg() const {
  stringstream ss;
  ss << this->type << " " << this->name;
  return ss.str();
}

std::string Variable::Define(std::string &&init, bool define_const) const {
  std::stringstream ss;
  if (define_const) {
    ss << "const ";
  }

  if (init.empty()) {
    ss << type << " " << name << ";";
  } else {
    ss << type << " " << name << " = " << std::move(init) << ";";
  }
  return ss.str();
}

std::string Variable::Assign(std::string &value) const {
  std::stringstream ss;
  ss << name << " = " << value << ";";
  return ss.str();
}

Axis::Axis(const ascir::Axis &axis)
    : ascir::Axis(axis),
      Variable(kIntT, axis.name),
      loop_size(Variable(kInt64T, axis.name + "_loop_size")),
      elem_size(axis.name + "_elem_size"),
      actual_size(axis.name + "_actual_size"),
      axis_size(axis.name + "_axis_size"),
      tail_size(axis.name + "_tail_size"),
      size_expr(ge::Symbol(axis.size.Str().get())) {}

Status Tensor::DtypeName(ge::DataType dtype, std::string &dtype_name) {
  static const std::string kTypeNames[] = {
      [ge::DT_FLOAT] = "float",     [ge::DT_FLOAT16] = "half",    [ge::DT_INT8] = "int8_t",
      [ge::DT_INT32] = "int32_t",   [ge::DT_UINT8] = "uint8_t",   "",
      [ge::DT_INT16] = "int16_t",   [ge::DT_UINT16] = "uint16_t", [ge::DT_UINT32] = "uint32_t",
      [ge::DT_INT64] = "int64_t",   [ge::DT_UINT64] = "uint64_t", [ge::DT_DOUBLE] = "",
      [ge::DT_BOOL] = "uint8_t",    [ge::DT_STRING] = "",         [ge::DT_DUAL_SUB_INT8] = "",
      [ge::DT_DUAL_SUB_UINT8] = "", [ge::DT_COMPLEX64] = "",      [ge::DT_COMPLEX128] = "",
      [ge::DT_QINT8] = "",          [ge::DT_QINT16] = "",         [ge::DT_QINT32] = "",
      [ge::DT_QUINT8] = "",         [ge::DT_QUINT16] = "",        [ge::DT_RESOURCE] = "",
      [ge::DT_STRING_REF] = "",     [ge::DT_DUAL] = "",           [ge::DT_VARIANT] = "",
      [ge::DT_BF16] = "bfloat16_t",   [ge::DT_UNDEFINED] = "",      [ge::DT_INT4] = "int4_t",
      [ge::DT_UINT1] = "",          [ge::DT_INT2] = "",           [ge::DT_UINT2] = "",
      [ge::DT_COMPLEX32] = "",
  };
  GE_CHK_BOOL_RET_STATUS((dtype < (sizeof(kTypeNames) / sizeof(kTypeNames[0])) && kTypeNames[dtype] != ""), ge::FAILED,
                         "Codegen unsupported data type:%d", static_cast<int32_t>(dtype));
  dtype_name = kTypeNames[dtype];
  return ge::SUCCESS;
}

const Type Tensor::GlobalTensorTypes(std::string &dtype_name) {
  return Type("GlobalTensor<" + dtype_name + ">");
}

const Type Tensor::LocalTensorTypes(std::string &dtype_name) {
  return Type("LocalTensor<" + dtype_name + ">");
}

Tensor::Tensor(const ascir::TensorAttr &tensor, std::string &dtype_name, const std::string &tensor_name)
    : Variable((ge::ascir::AscTensorUtils::IsConstTensor(tensor))                ? Type(dtype_name)
               : (tensor.attr.mem.alloc_type == ge::AllocType::kAllocTypeGlobal) ? GlobalTensorTypes(dtype_name)
                                                                                 : LocalTensorTypes(dtype_name),
               (ge::ascir::AscTensorUtils::IsConstTensor(tensor)) ? ("scalar_" + to_string(tensor.attr.mem.tensor_id))
               : (tensor.attr.mem.alloc_type == ge::AllocType::kAllocTypeGlobal)
                   ? ("global_" + to_string(tensor.attr.mem.tensor_id))
                   : ("local_" + to_string(tensor.attr.mem.tensor_id))),
      id(tensor.attr.mem.tensor_id),
      reuse_id(tensor.attr.mem.reuse_id),
      dtype(tensor.attr.dtype),
      alloc_type(tensor.attr.mem.alloc_type),
      position(tensor.attr.mem.position),
      axis(tensor.attr.axis),
      axis_size(tensor.attr.repeats),
      axis_strides(tensor.attr.strides),
      vectorized_axis(tensor.attr.vectorized_axis),
      vectorized_strides(tensor.attr.vectorized_strides),
      que_id(tensor.attr.que.id),
      buf_id(tensor.attr.buf.id),
      size(this->name + "_size"),
      actual_size(this->name + "_actual_size"),
      que_depth(this->name + "_que_depth"),
      que_buf_num(this->name + "_que_buf_num"),
      que_share_offset("q" + std::to_string(tensor.attr.que.id) + "_reuse" + std::to_string(tensor.attr.mem.reuse_id) +
                       "_offset"),
      const_value(""),
      const_value_expr(ge::Symbol(0)),
      que_depth_value(tensor.attr.que.depth),
      que_buf_num_value(tensor.attr.que.buf_num),
      merge_scope(tensor.attr.opt.merge_scope),
      is_constant(ge::ascir::AscTensorUtils::IsConstTensor(tensor)),
      ub_scalar_name(this->name + "_ub_scalar") {
  (void)tensor_name;
}

Tensor::Tensor(const ascir::TensorAttr &tensor, std::string &dtype_name, const ascir::SizeExpr &value,
               const std::string &tensor_name)
    : Tensor(tensor, dtype_name, tensor_name) {
  this->const_value_expr = value;
  this->is_constant = true;
}

Tensor::Tensor(const std::string &value, const ascir::TensorAttr &tensor, std::string &dtype_name,
               const std::string &tensor_name)
    : Tensor(tensor, dtype_name, tensor_name) {
  this->const_value = value;
  this->is_constant = true;
}

Status Tensor::Init() {
  for (auto vec_axis : this->vectorized_axis) {
    auto pos = std::find(this->axis.begin(), this->axis.end(), vec_axis);
    GE_ASSERT_TRUE((pos != this->axis.end()), "Codegen vectorized axis[%ld] not found", vec_axis);
    this->vectorized_axis_pos.push_back(pos - this->axis.begin());
  }

  // repeates 大于0的时候, 先初始化is_ub_scalar为true, 如有有其一repeate不是One, 则赋值为false, 认为不是ub_scalar场景
  is_ub_scalar = (this->axis_size.size() > 0U);
  for (auto &repeate : this->axis_size) {
    if (repeate != One) {
      is_ub_scalar = false;
      break;
    }
  }

  GELOGD("t_name:%s, axis_id:%s, size:%s, strides:%s, v_axis_id:%s, v_axis_pos:%s, v_strides:%s, is_ub_scalar:%d",
         name.c_str(), VectorToStr(this->axis).c_str(), VectorToStr(this->axis_size).c_str(),
         VectorToStr(this->axis_strides).c_str(), VectorToStr(this->vectorized_axis).c_str(),
         VectorToStr(this->vectorized_axis_pos).c_str(), VectorToStr(this->vectorized_strides).c_str(),
         static_cast<int32_t>(is_ub_scalar));

  return ge::SUCCESS;
}

Status Tensor::InitUbScalar(std::string &result) const {
  std::stringstream ss;
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(this->dtype, dtype_name), "data type:%d failed",
                    static_cast<int32_t>(this->dtype));
  ss << ub_scalar_name << " = ";
  ss << name << ".GetValue(0);" << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

Status Tensor::GenDuplicateValueOfUbScalar(std::string &result) const {
  std::stringstream ss;
  std::string dtype_name;
  Tensor::DtypeName(this->dtype, dtype_name);

  std::string event_id = this->name + "_event_id";
  ss << "AscendC::PipeBarrier<PIPE_ALL>();" << std::endl;
  ss << "Duplicate(" <<  this->name << "[0], "
     << ub_scalar_name << ", " << "32/sizeof(" << dtype_name <<"));" << std::endl;
  ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

Status Tensor::DefineUbScalar(std::string &result) const {
  std::stringstream ss;
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(this->dtype, dtype_name), "data type:%d failed",
                    static_cast<int32_t>(this->dtype));
  ss << dtype_name << " " << ub_scalar_name << ";" << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

Status Tensor::SetGlobalBuffer(GM_ADDR global, const std::string &offset, std::string &result) const {
  std::stringstream ss;
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(this->dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(this->dtype));

  if (!offset.empty() && offset != "0") {
    ss << name << ".SetGlobalBuffer("
       << "(__gm__ " << dtype_name << "*)((__gm__ uint8_t*)(" << global << ") + (" << offset << ")));";
  } else {
    ss << name << ".SetGlobalBuffer("
       << "(__gm__ " << dtype_name << "*)" << global << ");";
  }
  result = ss.str();
  return ge::SUCCESS;
}

Status codegen::PositionValue(ascir::Position position, std::string &result) {
  static std::unordered_map<size_t, std::string> position_values = {
    {static_cast<size_t>(ge::Position::kPositionGM), "TPosition::GM"},
    {static_cast<size_t>(ge::Position::kPositionVecIn), "TPosition::VECIN"},
    {static_cast<size_t>(ge::Position::kPositionVecCalc), "TPosition::VECCALC"},
    {static_cast<size_t>(ge::Position::kPositionVecOut), "TPosition::VECOUT"}
  };

  auto it = position_values.find(static_cast<size_t>(position));
  if (it == position_values.end()) {
    GELOGE(ge::FAILED, "Codegen position value[%d] invalid", static_cast<int32_t>(position));
    return ge::FAILED;
  }

  result = it->second;
  return ge::SUCCESS;
}

MergeScope::MergeScope(ascir::MergeScopeId merge_scope_id, ascir::Position pos)
    : id(merge_scope_id),
      position(pos),
      size("m" + to_string(merge_scope_id) + "_size"),
      depth("m" + to_string(merge_scope_id) + "_que_depth"),
      buf_num("m" + to_string(merge_scope_id) + "_que_buf_num") {}

TQue::TQue(ascir::QueId que_id, ascir::Position pos, std::string &position_name)
    : Variable(Type("TQue<" + position_name + ", " + "1>"), "q" + to_string(que_id)),
      id(que_id),
      position(pos),
      size(this->name + "_size"),
      depth(this->name + "_depth"),
      buf_num(this->name + "_buf_num"),
      buf(Type("LocalTensor<uint8_t>"), name + "_buf") {}

TQue::TQue(ascir::QueId que_id, ascir::Position src_position, const std::string &src_position_name,
           const std::string &dst_position_name)
    : Variable(Type("TQueBind<" + src_position_name + ", " + dst_position_name + ", 1>"), "q" + to_string(que_id)),
      id(que_id),
      position(src_position),
      size(this->name + "_size"),
      depth(this->name + "_depth"),
      buf_num(this->name + "_buf_num"),
      buf(Type("LocalTensor<uint8_t>"), name + "_buf") {}

std::string TQue::AllocBuf(const bool with_define) const {
  stringstream ss;
  if (with_define && !is_cv_ub_fusion) {
    ss << this->buf.AsArg();
  } else {
    ss << this->buf.Str();
  }
  ss << " = " << this->name << ".AllocTensor<uint8_t>();" << std::endl;
  return ss.str();
}

std::string TQue::FreeBuf() const {
  stringstream ss;
  ss << this->name << ".FreeTensor(" << this->buf << ");" << std::endl;
  return ss.str();
}

std::string TQue::EnqueBuf() const {
  stringstream ss;
  ss << this->name << ".EnQue(" << this->buf << ");" << std::endl;
  return ss.str();
}

std::string TQue::DequeBuf(const bool is_unit_first) const {
  stringstream ss;
  if (is_unit_first) {
    ss << this->buf.AsArg() << " = " << this->name << ".DeQue<uint8_t>();" << std::endl;
  } else {
    ss << this->buf.name << " = " << this->name << ".DeQue<uint8_t>();" << std::endl;
  }
  return ss.str();
}

TBuf::TBuf(ascir::BufId buf_id, const ascir::Position pos, std::string &position_name)
    : Variable(Type("TBuf<" + position_name + ">"), "b" + to_string(buf_id)),
      id(buf_id),
      position(pos),
      size(this->name + "_size"),
      buf(Type("LocalTensor<uint8_t>"), name + "_buf") {}

std::string TBuf::AllocBuf(const bool with_define) const {
  stringstream ss;
  if (with_define) {
    ss << this->buf.AsArg();
  } else {
    ss << this->buf.Str();
  }
  ss << " = " << this->name << ".Get<uint8_t>();";
  return ss.str();
}

std::string TBuf::AllocBuf(std::string buf_name, std::string dtype_name, const bool with_define) const {
  stringstream ss;
  if (with_define) {
    ss << "LocalTensor<" << dtype_name << "> " << buf_name << " = " << this->name << ".Get<" << dtype_name << ">();";
  } else {
    ss << buf_name << " = " << this->name << ".Get<" << dtype_name << ">();";
  }
  return ss.str();
}

Tiler::Tiler(const std::string &tiling_data_type, const std::string &tiling_data_name)
    : tiling_data(Type{tiling_data_type}, tiling_data_name), gm_tiling(kGmAddrT, "gm_tiling"), block_dim("block_dim") {}

std::string Tiler::Offset(const std::vector<ascir::AxisId> &current_axis, const std::vector<ascir::AxisId> &axis,
                          const std::vector<ascir::SizeExpr> &strides) const {
  std::stringstream ss;
  bool is_first = true;

  for (auto iter = axis.begin(); iter != axis.end(); ++iter) {
    bool is_from = false;
    for (auto ca : current_axis) {
      if (this->IsFrom(ca, *iter)) {
        is_from = true;
        break;
      }
    }

    if (!is_from) {
      continue;
    }

    if (is_first) {
      is_first = false;
    } else {
      ss << " + ";
    }

    auto stride = strides[iter - axis.begin()];
    if (stride == 0) {
      ss << "0";
    } else if (stride == 1) {
      ss << "(int64_t)" << this->GetAxis(*iter);
    } else {
      ss << "(int64_t)" << this->GetAxis(*iter) << " * " << "(int64_t)" << this->Size(stride);
    }
  }

  if (is_first) {
    // Not axis in current_axis
    ss << "0";
  }
  return ss.str();
}

std::string Tiler::TensorVectorizedOffset(const std::vector<ascir::AxisId> &current_axis, const Tensor &tensor) const {
  std::vector<ascir::AxisId> current_vectorized_axis;
  for (auto a : current_axis) {
    if (find(tensor.vectorized_axis.begin(), tensor.vectorized_axis.end(), a) != tensor.vectorized_axis.end()) {
      current_vectorized_axis.emplace_back(a);
    }
  }
  return this->Offset(current_vectorized_axis, tensor.vectorized_axis, tensor.vectorized_strides);
}

std::string Tiler::Str() const {
  return tiling_data.Str();
}

void codegen::Tiler::AddSizeVar(ascir::SizeVar size) {
  std::string var_define;
  if (!(size.expr.IsConstExpr())) {
    var_define = std::string(size.expr.Str().get());
    std::string tiling_var = this->tiling_data.Str() + "->" + var_define;
    ge::Expression tiling_sizevar = ge::Symbol(tiling_var.c_str());
    this->sizes.emplace_back(std::make_pair(size.expr, tiling_sizevar));
  }
}

uint32_t codegen::Tiler::GetTilingCaseId() const {
  return this->tiling_case_id;
}

void codegen::Tiler::SetTilingCaseId(uint32_t tilingCaseId) {
  this->tiling_case_id = tilingCaseId;
}

Status codegen::Tiler::AddAxis(const ascir::Axis &axis) {
  auto [new_axis, insert_success] = this->axis_map.emplace(axis.id, codegen::Axis(axis));
  (void)new_axis;
  if (!insert_success) {
    GELOGE(ge::FAILED, "Codegen insert axis[%ld] fail", axis.id);
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

static bool GetSplitBAttr(const Tiler *tiler, const Axis &axis) {
  if (axis.type == ascir::Axis::Type::kAxisTypeTileInner || axis.type == ascir::Axis::Type::kAxisTypeTileOuter) {
    return true;
  }
  if (axis.type == ascir::Axis::Type::kAxisTypeInvalid) {
    return false;
  }
  for (const auto from : axis.from) {
    auto from_axis = tiler->GetAxis(from);
    if (GetSplitBAttr(tiler, from_axis)) {
      return true;
    }
  }
  return false;
}

void codegen::Tiler::AddAxisSplitBAttr() {
  for (auto &[id, cur_axis] : axis_map) {
    (void)id;
    cur_axis.is_split_b = GetSplitBAttr(this, cur_axis);
  }
}

static bool IsOuter(const ascir::Axis &axis) {
  return (axis.type == ascir::Axis::Type::kAxisTypeBlockOuter || axis.type == ascir::Axis::Type::kAxisTypeTileOuter);
}

static bool IsInner(const ascir::Axis &axis) {
  return (axis.type == ascir::Axis::Type::kAxisTypeBlockInner || axis.type == ascir::Axis::Type::kAxisTypeTileInner);
}

static bool IsTileInner(const ascir::Axis &axis) {
  return (axis.type == ascir::Axis::Type::kAxisTypeTileInner);
}

static bool IsMergeFromInner(const Tiler &tiler, const ascir::Axis &axis) {
  if (axis.from.size() == 0) {
    return IsInner(axis);
  }

  bool contain_inner = false;
  std::function<void(int32_t)> func = [&tiler, &contain_inner, &func](int32_t current_axis_id) {
    const auto &current_axis = tiler.GetAxis(current_axis_id);
    for (const auto &from : current_axis.from) {
      const auto &from_axis = tiler.GetAxis(from);
      if (IsInner(from_axis)) {
        contain_inner = true;
        break;
      } else if (from_axis.type == ascir::Axis::Type::kAxisTypeMerged) {
        func(from);
      }
    }
  };
  func(axis.id);
  return contain_inner;
}

bool Tiler::IsFrom(ascir::AxisId src, ascir::AxisId dst) const {
  if (src == dst) {
    return true;
  }

  const auto &axis = this->GetAxis(src);
  for (const auto &from : axis.from) {
    if (from == dst || IsFrom(from, dst)) {
      return true;
    }
  }

  return false;
}

bool Tiler::HasSameOriginAxis(ascir::AxisId src, ascir::AxisId dst) const {
  std::set<ascir::AxisId> src_origins;
  std::set<ascir::AxisId> dst_origins;
  std::function<void(int32_t, std::set<ascir::AxisId> &)> func = [this, &func](int32_t current_axis_id,
                                                                 std::set<ascir::AxisId> &origin_ids) {
    const auto &axis = this->GetAxis(current_axis_id);
    for (const auto &from : axis.from) {
      if (this->GetAxis(from).type == Axis::Type::kAxisTypeOriginal) {
        origin_ids.insert(from);
      } else {
        func(from, origin_ids);
      }
    }
  };

  func(src, src_origins);
  func(dst, dst_origins);

  for (auto id : src_origins) {
    if (dst_origins.count(id) != 0) {
      return true;
    }
  }
  return false;
}

std::string codegen::Tiler::Size(const ascir::SizeExpr &size, bool using_int_tiling_data) const {
  std::string const_expr_str = std::string(size.Str().get());
  if (size.IsConstExpr()) {
    return (const_expr_str.find("Rational") != std::string::npos) ?
            ge::SymbolicUtils::AsNumerDenomToString(size) : const_expr_str;
  }
  std::string str_ret = std::string((size.Replace(this->sizes)).Str().get());
  return (using_int_tiling_data || str_ret.find("Rational") != std::string::npos) ?
          ge::SymbolicUtils::AsNumerDenomToString(size.Replace(this->sizes)) : str_ret;
}

std::string codegen::Tiler::ActualSize(const ascir::SizeExpr &size, bool using_int_tiling_data) const {
  auto replace_actual = size.Replace(this->actual_sizes);
  std::string str_ret = std::string((replace_actual.Replace(this->sizes)).Str().get());
  return (using_int_tiling_data || str_ret.find("Rational") != std::string::npos) ?
          ge::SymbolicUtils::AsNumerDenomToString(replace_actual.Replace(this->sizes)) : str_ret;
}

std::string Tiler::TensorActualSize(const Tensor &tensor) const {
  if (tensor.vectorized_axis.size() == 0) {
    return "1";
  }

  stringstream ss;
  int64_t count = 0;
  for (size_t i = 0; i < tensor.vectorized_axis.size(); i++) {
    auto &stride = tensor.vectorized_strides[i];
    if (stride == 0) {
      continue;
    }

    if (count >= 1) {
      ss << " + ";
    }

    auto axis = GetAxis(tensor.vectorized_axis[i]);
    auto axis_pos = tensor.vectorized_axis_pos[i];
    auto axis_size = tensor.axis_size[axis_pos];
    bool size_equal = ge::SymbolicUtils::StaticCheckEq(axis_size, axis.size_expr) == ge::TriBool::kTrue;
    if (axis.type == Axis::Type::kAxisTypeTileInner || size_equal) {
      ss << "(" + axis.actual_size.Str() + " - 1)";
    } else {
      ss << "(" + this->Size(axis_size) + " - 1)";
    }

    if (stride != 1) {
      ss << " * " << this->Size(stride);
    }
    count++;
  }

  ss << ((ss.str().size() == 0u) ? "1" : " + 1");
  return ss.str();
}

std::string Tiler::TensorVectorizedSize(const Tensor &tensor) const {
  if (tensor.vectorized_axis.size() == 0) {
    return "1";
  }
  stringstream ss;
  std::string blk_align;
  (void)KernelUtils::BlkAlign(tensor.dtype, blk_align);
  ss << blk_align << "(";
  int64_t count = 0;
  for (size_t i = 0; i < tensor.vectorized_axis.size(); i++) {
    auto axis_pos = tensor.vectorized_axis_pos[i];
    auto axis_size = tensor.axis_size[axis_pos];
    auto &stride = tensor.vectorized_strides[i];
    if (stride == 0) {
      continue;
    }
    if (count >= 1) {
      ss << " + ";
    }
    ss << "(" + this->Size(axis_size) + " - 1)";
    if (stride != 1) {
      ss << " * " << this->Size(stride);
    }
    count++;
  }

  ss << ((ss.str().size() == 0u) ? "1" : " + 1)");  // stride全为0的ub_scalar场景, size返回1
  return ss.str();
}

const Axis &Tiler::GetAxis(const ascir::AxisId id) const {
  auto iter = this->axis_map.find(id);
  if (iter == this->axis_map.end()) {
    GELOGE(ge::FAILED, "Codegen axis[%ld] not found", id);
    throw std::runtime_error("Axis not found " + to_string(id));
  }

  return iter->second;
}

std::string codegen::Tiler::AxisSize(const ascir::AxisId id) const {
  return this->Size(this->GetAxis(id).size);
}

std::string codegen::Tiler::AxisSize(const Axis &axis) const {
  return this->Size(axis.size);
}

std::string codegen::Tiler::GenAxisSizeNew(const ascir::AxisId id) const {
  stringstream ss;

  const auto &axis = this->GetAxis(id);
  bool is_reduce_block = axis.type == ascir::Axis::Type::kAxisTypeBlockOuter && axis.from.size() > 1;
  if (axis.type == ascir::Axis::Type::kAxisTypeOriginal) {
    ss << "const " << axis.axis_size.AsArg() << " = " << this->AxisSize(axis) << ";" << endl;
    ss << "const " << axis.loop_size.AsArg() << " = " << axis.axis_size.Str() << ";" << endl;
    ss << "const " << axis.actual_size.AsArg() << " = " << axis.axis_size.Str() << ";" << endl;
  } else if (IsOuter(axis) && !is_reduce_block) {
    const auto &from = this->GetAxis(axis.from[0]);
    const auto &inner = this->GetAxis(axis.split_pair_other_id);
    ss << "const " << axis.axis_size.AsArg() << " = " << from.loop_size.Str() << " / " << inner.axis_size.Str() << ";"
       << endl;
    ss << "const " << axis.loop_size.AsArg() << " = " << axis.axis_size.Str() << " + (" << inner.tail_size << " > 0);"
       << endl;
  } else if (IsInner(axis)) {
    const auto &from = this->GetAxis(axis.from[0]);
    ss << "const " << axis.axis_size.AsArg() << " = " << this->AxisSize(axis) << ";" << endl;
    ss << "const " << axis.tail_size.AsArg() << " = " << from.loop_size << " % " << axis.axis_size << ";" << endl;
  } else if (axis.type == ascir::Axis::Type::kAxisTypeMerged || is_reduce_block) {
    ss << "const " << axis.axis_size.AsArg() << " = ";
    for (const auto &f : axis.from) {
      ss << this->GetAxis(f).loop_size.Str() << " * ";
    }
    ss << "1;" << endl;

    ss << "const " << axis.loop_size.AsArg() << " = " << axis.axis_size.Str() << ";" << endl;
    ss << "const " << axis.actual_size.AsArg() << " = " << axis.axis_size.Str() << ";" << endl;
  }

  return ss.str();
}

std::string codegen::Tiler::GenInnerLoopSizeAndActualSize(const ascir::AxisId id, const ascir::AxisId loop_axis,
                                                          bool is_need_divide_sum, bool is_define) const {
  stringstream ss;

  auto axis = this->GetAxis(id);
  if (!IsInner(axis)) {
    return "";
  }

  if (IsFrom(loop_axis, axis.split_pair_other_id)) {
    auto outter = this->GetAxis(axis.split_pair_other_id);
    ss << (is_define ? axis.actual_size.AsArg() : axis.actual_size.Str()) << " = " << outter.Str() << " < "
       << outter.axis_size << " ? " << axis.axis_size.Str() << " : " << axis.tail_size.Str() << ";" << endl;
    ss << (is_define ? axis.loop_size.AsArg() : axis.loop_size.Str()) << " = " << axis.actual_size.Str() << ";" << endl;
    if (is_need_divide_sum && IsTileInner(axis)) {
      ss << "is_last_inner_block_tail" << " = (" << axis.actual_size.Str() << " == " << axis.tail_size.Str() << ");"
         << endl;
    }

    ge::Expression actual_size = ge::Symbol(axis.actual_size.name.c_str());
    this->actual_sizes.emplace_back(std::make_pair(axis.size_expr, actual_size));
  }

  return ss.str();
}

std::string codegen::Tiler::CalcFromAxis(const ascir::AxisId id, bool is_define) const {
  stringstream ss;
  auto axis = this->GetAxis(id);
  if (IsInner(axis)) {
    auto from = this->GetAxis(axis.from[0]);
    ss << (is_define ? from.AsArg() : from.Str()) << " = " << "block_dim_offset" << " + "
       << axis.Str() << ";" << std::endl;
    ss << this->CalcFromAxis(from.id, is_define);
  } else if (axis.type == Axis::Type::kAxisTypeMerged) {
    for (size_t i = 0; i < axis.from.size(); i++) {
      auto &from = this->GetAxis(axis.from[i]);
      ss << (is_define ? from.AsArg() : from.Str()) << " = " << axis.Str();
      ss << " / ";
      ss << "(";
      for (size_t j = i + 1; j < axis.from.size(); j++) {
        ss << this->GetAxis(axis.from[j]).loop_size << " * ";
      }
      ss << "1)";
      ss << " % " << this->GetAxis(axis.from[i]).loop_size << ";" << std::endl;
    }
    for (auto from : axis.from) {
      ss << this->CalcFromAxis(from, is_define);
    }
  }

  return ss.str();
}

void codegen::Tiler::BlockOutterAxisDefine(const ascir::AxisId id, std::stringstream &ss) {
  auto axis = this->GetAxis(id);
  if (IsInner(axis)) {
    return;
  }
  for (size_t i = 0; i < axis.from.size(); i++) {
    auto &from = this->GetAxis(axis.from[i]);
    ss << from.AsArg() << " = " << axis.Str();
    ss << " / (";
    for (size_t j = i + 1; j < axis.from.size(); j++) {
      ss << this->GetAxis(axis.from[j]).loop_size << " * ";
    }
    ss << "1)";
    ss << " % " << this->GetAxis(axis.from[i]).loop_size << ";" << std::endl;
    if (from.type == Axis::Type::kAxisTypeMerged) {
      BlockOutterAxisDefine(axis.from[i], ss);
    }
  }
}

std::string codegen::Tiler::BlockOutterAxisDefine() {
  stringstream code;
  code << this->block_dim.Define("GetBlockIdx()") << std::endl;
  if (enable_group_parallel_) {
    code << "const uint32_t block_offset = " << tiling_data.name << "->ub_size;  // resue as block_offset"
         << std::endl;
    code << this->block_dim.name << " = " << this->block_dim.name << " >= block_offset ? "
         << this->block_dim.name << " - block_offset : "
         << this->block_dim.name << " + GetBlockNum() - block_offset;" << std::endl;
    // block_dim范围在调用前校验了，此处不需要重复校验
  } else {
    code << "if (" << this->block_dim.name << " >= " << tiling_data.name << "->block_dim) { " << std::endl
         << "  return;" << std::endl << "}" << std::endl;
  }
  for (auto &[id, axis] : this->axis_map) {
    (void)id;
    if (axis.type != ascir::Axis::Type::kAxisTypeBlockOuter) {
      continue;
    }

    stringstream axis_value;
    axis_value << this->block_dim.name << " % " << axis.loop_size;
    code << axis.Define(axis_value.str(), true);
    code << " " << std::endl;
    if (axis.from.size() > 1) {
      BlockOutterAxisDefine(id, code);
    }
  }

  return code.str();
}

void Tiler::EnableGroupParallel(bool enable_group_parallel) {
  enable_group_parallel_ = enable_group_parallel;
}

std::string KernelUtils::Max() {
  return "KernelUtils::Max";
}

std::string KernelUtils::Sum() {
  return "KernelUtils::Sum";
}

Status KernelUtils::BlkNum(ge::DataType dtype, std::string &result) {
  std::stringstream ss;
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(dtype));
  ss << "KernelUtils::BlkNum<" << dtype_name << ">";
  result = ss.str();
  return ge::SUCCESS;
}

Status KernelUtils::BlkAlign(ge::DataType dtype, std::string &result) {
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(dtype));
  result = "KernelUtils::BlkAlign<" + dtype_name + ">";
  return ge::SUCCESS;
}

std::string KernelUtils::SizeAlign() {
  return "KernelUtils::SizeAlign";
}

std::string KernelUtils::FindNearestPower2() {
  return "KernelUtils::FindNearestPower2";
}

TPipe::TPipe(const std::string &tpipe_name, const Tiler &tpipe_tiler)
    : Variable(Type{"TPipe"}, tpipe_name), tiler(tpipe_tiler), tmp_buf(Type{"LocalTensor<uint8_t>"}, "tmp_buf") {}

Status TPipe::AddTensor(const Tensor &tensor) {
  auto [ret, is_insert1] = this->tensors.emplace(tensor.id, tensor);
  GE_CHK_BOOL_RET_STATUS(is_insert1, ge::FAILED, "Codegen tensor[%ld,%s] is already added", tensor.id,
                         tensor.name.c_str());

  auto &t = ret->second;
  if (t.merge_scope != ge::kIdNone &&
      (t.alloc_type == ge::AllocType::kAllocTypeQueue || t.alloc_type == ge::AllocType::kAllocTypeBuffer)) {
    auto merge_scope = this->merge_scopes.find(t.merge_scope);
    if (merge_scope == this->merge_scopes.end()) {
      auto [new_scope, is_insert2] = this->merge_scopes.emplace(t.merge_scope, MergeScope{t.merge_scope, t.position});
      GE_CHK_BOOL_RET_STATUS(is_insert2, ge::FAILED, "Codegen emplace merge_scope [%ld] failed", t.merge_scope);
      new_scope->second.tensors.push_back(t.id);
    } else {
      GE_CHK_BOOL_RET_STATUS(merge_scope->second.position == t.position, ge::FAILED,
                             "Merge scope for tensor[%s] position mismatch between %d and %d", t.name.c_str(),
                             static_cast<int32_t>(t.position), static_cast<int32_t>(merge_scope->second.position));
      merge_scope->second.tensors.push_back(t.id);
    }
  }

  if (t.alloc_type == ge::AllocType::kAllocTypeQueue) {
    GE_CHK_BOOL_RET_STATUS(t.que_id != ge::kIdNone, ge::FAILED, "Codegen tensor[%ld,%s] queue is none", t.id,
                           t.name.c_str());
    TQue *que = nullptr;
    auto iter = this->ques.find(t.que_id);
    GE_ASSERT_TRUE(iter != this->ques.end(), "Cannot find que with id [%ld], it may not be initialized correctly",
                   t.que_id);
    que = &iter->second;
    if (t.merge_scope != ge::kIdNone) {
      que->merge_scopes.insert(t.merge_scope);
    } else {
      que->not_merge_tensors.insert(t.id);
    }
    que->share_group[t.reuse_id].push_back(t.id);
  } else if (t.alloc_type == ge::AllocType::kAllocTypeBuffer) {
    GE_CHK_BOOL_RET_STATUS(t.buf_id != ge::kIdNone, ge::FAILED, "Codegen tensor[%ld,%s] buffer is none", t.id,
                           t.name.c_str());

    TBuf *buf = nullptr;
    auto iter = this->bufs.find(t.buf_id);
    if (iter == this->bufs.end()) {
      std::string position;
      GE_CHK_STATUS_RET(PositionValue(t.position, position), "Codegen get position value failed");
      auto [new_buf, is_insert5] = this->bufs.emplace(t.buf_id, TBuf{t.buf_id, t.position, position});
      GE_CHK_BOOL_RET_STATUS(is_insert5, ge::FAILED, "Codegen emplace tbuf [%ld] failed", t.buf_id);
      buf = &new_buf->second;
    } else {
      buf = &iter->second;
    }

    GE_CHK_BOOL_RET_STATUS(buf->position == t.position, ge::FAILED,
                           "Codegen buf position mismatch for tensor[%s] between %d and %d", t.name.c_str(),
                           static_cast<int32_t>(t.position), static_cast<int32_t>(buf->position));

    if (t.merge_scope != ge::kIdNone) {
      buf->merge_scopes.insert(t.merge_scope);
    } else {
      buf->not_merge_tensors.insert(t.id);
    }
  }

  return ge::SUCCESS;
}

Status TPipe::AddTensor(const ascir::TensorAttr &tensor_attr, const std::string &tensor_name) {
  auto tensor_val_name = GenValidName(tensor_name);
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(tensor_attr.attr.dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(tensor_attr.attr.dtype));
  Tensor tensor(tensor_attr, dtype_name, tensor_val_name);
  GE_CHK_STATUS_RET(tensor.Init(), "Codegen tensor init failed");
  GE_CHK_STATUS_RET(this->AddTensor(tensor), "Codegen add tensor failed");
  return ge::SUCCESS;
}

Status TPipe::AddTensor(const std::string &const_value, const ascir::TensorAttr &tensor_attr,
                        const std::string &tensor_name) {
  auto tensor_val_name = GenValidName(tensor_name);
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(tensor_attr.attr.dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(tensor_attr.attr.dtype));

  // const_value预处理
  std::string pre_process_value;
  GE_CHK_STATUS_RET(ascgen_utils::ScalarValuePreProcess(const_value, dtype_name, pre_process_value),
                    "Scalar value pre process failed, ori_value:%s, dtype:%s", const_value.c_str(), dtype_name.c_str());
  GELOGD("ori_value:%s, dtype:%s, pre_process_value:%s", const_value.c_str(), dtype_name.c_str(),
         pre_process_value.c_str());
  Tensor tensor(pre_process_value, tensor_attr, dtype_name, tensor_val_name);
  GE_CHK_STATUS_RET(tensor.Init(), "Codegen tensor init failed");
  GE_CHK_STATUS_RET(this->AddTensor(tensor), "Codegen add tensor failed");

  return ge::SUCCESS;
}

Status TPipe::AddTensor(const ascir::TensorAttr &tensor_attr, const ascir::SizeExpr &const_value,
                        const std::string &tensor_name) {
  auto tensor_val_name = GenValidName(tensor_name);
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(tensor_attr.attr.dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(tensor_attr.attr.dtype));
  Tensor tensor(tensor_attr, dtype_name, const_value, tensor_val_name);
  GE_CHK_STATUS_RET(tensor.Init(), "Codegen tensor init failed");
  GE_CHK_STATUS_RET(this->AddTensor(tensor), "Codegen add tensor failed");

  return ge::SUCCESS;
}

std::string TPipe::AllocTmpBuf(const TBuf &buf, const bool with_define) const {
  stringstream ss;
  if (with_define) {
    ss << this->tmp_buf.AsArg();
  } else {
    ss << this->tmp_buf.Str();
  }
  ss << "_" << to_string(buf.id)  << " = " << buf.name << ".Get<uint8_t>();" << std::endl;
  return ss.str();
} 

static bool IsNextNodeSupportScalar(const ascir::NodeView &node) {
  std::set<std::string> support_ub_scalar_nodes = {Load::Type, Store::Type, Div::Type, TrueDiv::Type, Mul::Type, 
    Add::Type, Sub::Type, Minimum::Type, Maximum::Type, LogicalOr::Type, LogicalAnd::Type, Broadcast::Type,
    ClipByValue::Type, Eq::Type, Ne::Type, Gt::Type, Lt::Type, Ge::Type, Le::Type, Pow::Type, Where::Type};
  return support_ub_scalar_nodes.count(node->GetType()) > 0U;
}

Status Kernel::OutputTensorIsUbScalar(const ascir::NodeView &node, bool &is_ub_scalar) const {
  is_ub_scalar = true;
  auto desc = node->GetOpDesc();
  for (auto output : node->outputs()) {
    auto output_index = ge::ascir::AscTensorUtils::Index(*output);
    auto tensor_name = node->GetName() + "_" + desc->GetOutputNameByIndex(output_index);

    auto tensor_val_name = GenValidName(tensor_name);
    std::string dtype_name;
    GE_CHK_STATUS_RET(Tensor::DtypeName(output->attr.dtype, dtype_name), "Codegen get data type:%d failed",
                      static_cast<int32_t>(output->attr.dtype));
    Tensor tensor(*output, dtype_name, tensor_val_name);
    GE_CHK_STATUS_RET(tensor.Init(), "Codegen tensor init failed");

    if (!tensor.is_ub_scalar) {
      is_ub_scalar = false;
      break;
    }
  }
  GELOGI("node:%s, output tensor is_ub_scalar:%d", node->GetNamePtr(), static_cast<int32_t>(is_ub_scalar));
  return ge::SUCCESS;
}

static bool IsOutputOnlyLink2VFNode(const ascir::TensorView &tensor) {
  for (const auto &peer_input : tensor.anchor.GetPeerInDataAnchors()) {
    auto output_node = std::dynamic_pointer_cast<ge::AscNode>(peer_input->GetOwnerNode());
    if (output_node == nullptr || output_node->GetType() != VectorFunc::Type) {
      return false;
    }
  }
  return true;
}

Status Kernel::ParseUbScalarOptimizationInfo(const ascir::NodeView& node, Tensor& t, ascir::TensorId id,
                                             bool is_all_link_vf) {
  if (t.is_ub_scalar && !ge::ops::IsOps<ge::ascir_op::Scalar>(node) && !is_all_link_vf) {
    // 下游节点有其中之一输出tensor不是ub_scalar, 且下游节点支持scalar输入, 则本tensor需要生成get value
    bool a_tenor_of_next_node_is_not_ub_scalar = false;
    bool is_next_node_support_ub_scalar = false;
    for (auto &out : node->outputs()) {
      if (out == nullptr) {
        continue;
      }
      for (auto &peer_input : out->anchor.GetPeerInDataAnchors()) {
        auto next_node = std::dynamic_pointer_cast<ge::AscNode>(peer_input->GetOwnerNode());
        t.need_duplicate_value_of_ub_scalar = IsSupportBlkTensorInput(next_node) ?
          true : t.need_duplicate_value_of_ub_scalar;
        bool is_ub_scalar;
        GE_CHK_STATUS_RET(OutputTensorIsUbScalar(next_node, is_ub_scalar));
        if (!is_ub_scalar) {
          a_tenor_of_next_node_is_not_ub_scalar = true;
          is_next_node_support_ub_scalar = IsNextNodeSupportScalar(next_node);
          break;
        }
      }
      if (a_tenor_of_next_node_is_not_ub_scalar) {
        this->ub_scalar_tensors.emplace_back(id);
        break;
      }
    }
    t.need_gen_get_value_of_ub_scalar = a_tenor_of_next_node_is_not_ub_scalar && is_next_node_support_ub_scalar;
    GELOGD("node:%s, tensor_id:%d, is_ub_scalar:%d, need_gen_get_value_of_ub_scalar:%d", node->GetNamePtr(),
           static_cast<int32_t>(id), static_cast<int32_t>(t.is_ub_scalar),
           static_cast<int32_t>(t.need_gen_get_value_of_ub_scalar));
  }
  return ge::SUCCESS;
}

Status Kernel::JudgeIsLoadLinkStoreAndVec(const ascir::NodeView& node, Tensor& t, ascir::TensorId id) {
  // todo: 解决load多引用场景，被store, vec 同时引用的缺少mte3到mte2的同步的问题,
  // 临时方案，从这里解析下是否该场景
  if ((node->attr.api.compute_type == ascir::ComputeType::kComputeLoad) && (!IsOps<Gather>(node))) {
    bool link_to_store = false;
    bool link_to_vec = false;
    for (auto &out : node->outputs()) {
      GE_CHK_BOOL_RET_STATUS_NOLOG(out != nullptr, ge::SUCCESS);  // 节点悬空认为合法
      for (auto &peer_input : out->anchor.GetPeerInDataAnchors()) {
        auto next_node = std::dynamic_pointer_cast<ge::AscNode>(peer_input->GetOwnerNode());
        link_to_store = IsOps<Store>(next_node) ? true : link_to_store;
        link_to_vec = IsOps<Store>(next_node) ? link_to_vec : true;
      }
    }
    t.is_load_link_store_and_vec = link_to_store && link_to_vec;
    GELOGD("node:%s, tensor_id:%d, is_load_link_store_and_vec:%d", node->GetNamePtr(), static_cast<int32_t>(id),
           static_cast<int32_t>(t.is_load_link_store_and_vec));
  }
  return ge::SUCCESS;
}

Status Kernel::ParseOptimizeInfo(const ascir::NodeView &node, const ascir::TensorView &tensor) {
  // 如果是reduce节点，强制设置是非ub_scalar场景
  std::set<std::string> force_non_ub_scalar = {Max::Type,  Sum::Type, Min::Type, Mean::Type,
                                               Prod::Type, Any::Type, All::Type};
  ascir::TensorId id = tensor.attr.mem.tensor_id;
  auto tensor_ptr = this->tpipe.GetTensor(id);
  GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, ge::FAILED, "Check[Param] tensor_ptr is nullptr");
  auto &t = *tensor_ptr;
  t.is_ub_scalar = (force_non_ub_scalar.count(node->GetType()) > 0U) ? false : t.is_ub_scalar;
  // 遍历下游节点不是白名单, 则返回, 下游节点是白名单
  GELOGD("node:%s, tensor_id:%d, is_ub_scalar:%d", node->GetNamePtr(), static_cast<int32_t>(id),
         static_cast<int32_t>(t.is_ub_scalar));
  bool is_all_link_vf = IsOutputOnlyLink2VFNode(tensor);
  GE_CHK_STATUS_RET(ParseUbScalarOptimizationInfo(node, t, id, is_all_link_vf));
  GE_CHK_STATUS_RET(JudgeIsLoadLinkStoreAndVec(node, t, id));
  ParseScalarNeedGenBlkTensors(node, id);

  return ge::SUCCESS;
}

Status Kernel::ParseScalarNeedGenBlkTensors(const ascir::NodeView &node, ascir::TensorId id) {
  // 是scalar的节点，判断下是否支持 blk tensor 输入的 Ascir
  if (!IsOps<Scalar>(node)) {
    return ge::SUCCESS;
  }
  for (auto &out : node->outputs()) {
    GE_CHK_BOOL_EXEC(out != nullptr, continue);
    for (auto &peer_input : out->anchor.GetPeerInDataAnchors()) {
      auto next_node = std::dynamic_pointer_cast<ge::AscNode>(peer_input->GetOwnerNode());
      if (IsSupportBlkTensorInput(next_node)) {
        this->tpipe.need_gen_blk_tensors.emplace_back(id);
        break;
      }
    }
  }
  return ge::SUCCESS;
}

const TQue* TPipe::GetQue(const ascir::QueId id) const {
  auto iter = this->ques.find(id);
  GE_CHK_BOOL_EXEC(iter != this->ques.end(), return nullptr, "Codegen que[%d] not found", id);
  return &iter->second;
}

const TBuf &TPipe::GetBuf(const ascir::BufId id) const {
  auto iter = this->bufs.find(id);
  if (iter == this->bufs.end()) {
    GELOGE(ge::FAILED, "Codegen buf[%d] not found", id);
    throw std::runtime_error("Buf not found " + to_string(id));
  }

  return iter->second;
}

const Tensor *TPipe::GetTensor(ascir::TensorId id) const {
  auto iter = tensors.find(id);
  GE_CHK_BOOL_EXEC(iter != tensors.end(), return nullptr, "Codegen tensor[%ld] not found", id);
  return &iter->second;
}

Tensor *TPipe::GetTensor(ascir::TensorId id) {
  auto iter = tensors.find(id);
  GE_CHK_BOOL_EXEC(iter != tensors.end(), return nullptr, "Codegen tensor[%ld] not found", id);
  return &iter->second;
}

Status TPipe::TensorAlloc(const Tensor &tensor, std::string &result) const {
  if (tensor.is_constant) {
    result = "";
    return ge::SUCCESS;
  }

  std::stringstream ss;
  if (this->cv_fusion_type != ascir::CubeTemplateType::kUBFuse) {
    ss << tensor.Define() << std::endl;
  }

  const Variable *buf;
  if (tensor.alloc_type == ge::AllocType::kAllocTypeBuffer) {
    buf = &GetBuf(tensor.buf_id).buf;
  } else if (tensor.alloc_type == ge::AllocType::kAllocTypeQueue) {
    auto t_que = GetQue(tensor.que_id);
    GE_CHK_BOOL_RET_STATUS(t_que != nullptr, ge::FAILED, "Codegen que[%ld] not found", tensor.que_id);
    buf = &t_que->buf;
  } else if (tensor.alloc_type == ge::AllocType::kAllocTypeGlobal) {
    buf = &tensor;
  } else {
    GELOGE(ge::FAILED, "Codegen tensor[%ld, %s] alloc type[%d] not supported", tensor.id, tensor.name.c_str(),
           static_cast<int32_t>(tensor.alloc_type));
    return ge::FAILED;
  }

  GE_CHK_BOOL_RET_STATUS(tensor.merge_scope == ge::kIdNone, ge::FAILED,
                         "Codegen tensor[%ld, %s] merge scope[%ld] not supported", tensor.id, tensor.name.c_str(),
                         tensor.merge_scope);

  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(tensor.dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(tensor.dtype));
  if (tensor.alloc_type == ge::AllocType::kAllocTypeQueue) {
    ss << tensor << " = " << *buf << "[" << tensor.que_share_offset << "]"
        << ".template ReinterpretCast<" << dtype_name << ">();" << std::endl;
  } else {
    ss << tensor << " = " << *buf << ".template ReinterpretCast<" << dtype_name << ">();" << std::endl;
  }
  result = ss.str();
  return ge::SUCCESS;
}

Status TPipe::InitTQueBuffers(const TQue &que, std::string &result) const {
  stringstream ss;
  std::string blk_align;
  GE_CHK_STATUS_RET(KernelUtils::BlkAlign(ge::DT_UINT8, blk_align), "Codegen blk align failed");
  if (this->cv_fusion_type == ascir::CubeTemplateType::kUBFuse || !using_att_calc_qbt_size_) {
    ss << this->name << "."
       << "InitBuffer(" << que << ", " << que.buf_num << ", " << blk_align << "(" << que.size << "));";
  } else {
    ss << "// ";
    ss << this->name << "."
       << "InitBuffer(" << que << ", " << que.buf_num << ", " << blk_align << "(" << que.size << "));" << std::endl;
    ss << this->name << "."
       << "InitBuffer(" << que << ", " << que.buf_num << ", t->q" << std::to_string(que.id) << "_size);";
  }
  result = ss.str();
  return ge::SUCCESS;
}

Status TPipe::InitTBufBuffer(const TBuf &buf, std::string &result) const {
  stringstream ss;
  std::string blk_align;
  GE_CHK_STATUS_RET(KernelUtils::BlkAlign(ge::DT_UINT8, blk_align), "Codegen blk align failed");
  if (this->cv_fusion_type == ascir::CubeTemplateType::kUBFuse || !using_att_calc_qbt_size_) {
    ss << this->name << "."
       << "InitBuffer(" << buf << ", " << blk_align << "(" << buf.size << "));";
  } else {
    ss << "// ";
    ss << this->name << "."
       << "InitBuffer(" << buf << ", " << blk_align << "(" << buf.size << "));" << std::endl;
    ss << this->name << "."
       << "InitBuffer(" << buf << ", t->b" << std::to_string(buf.id) << "_size);";
  }

  result = ss.str();
  return ge::SUCCESS;
}

std::string TPipe::TensorSizeCalc() const {
  stringstream ss;

  for (const auto &pair : this->tensors) {
    const auto &t = pair.second;
    if (t.alloc_type == ge::AllocType::kAllocTypeQueue) {
      ss << t.size.DefineConst(this->tiler.TensorVectorizedSize(t)) << std::endl;
      ss << t.que_buf_num.DefineConst(to_string(t.que_buf_num_value)) << std::endl;
    } else if (t.alloc_type == ge::AllocType::kAllocTypeBuffer) {
      ss << t.size.DefineConst(this->tiler.TensorVectorizedSize(t)) << std::endl;
    }
  }
  return ss.str();
}

std::string TPipe::TensorActualSizeCalc(const ascir::TensorId id) const {
  stringstream ss;
  auto t_ptr = GetTensor(id);
  GE_CHK_BOOL_EXEC(t_ptr != nullptr, return "", "t_ptr nullptr");
  auto &t = *t_ptr;
  if (this->cv_fusion_type != ascir::CubeTemplateType::kUBFuse) {
    ss << t.actual_size.DefineConst(this->tiler.TensorActualSize(t));
  } else {
    ss << t.actual_size.Str() << " = " << this->tiler.TensorActualSize(t) << ";";
  }
  ss << std::endl;

  return ss.str();
}

Status TPipe::MergeScopeSizeCalc(std::string &result) const {
  stringstream ss;

  for (const auto &pair : this->merge_scopes) {
    const auto &merge_scope = pair.second;
    stringstream tensor_size_sum;
    stringstream tensor_bufnum_max;

    tensor_size_sum << KernelUtils::Sum() << "(";
    tensor_bufnum_max << KernelUtils::Max() << "(";

    bool first = true;
    for (auto tid : merge_scope.tensors) {
      auto tensor = this->tensors.find(tid);
      if (tensor == this->tensors.end()) {
        GELOGE(ge::FAILED, "Codegen tensor[%ld] not found", tid);
        return ge::FAILED;
      }

      if (tensor->second.alloc_type != ge::AllocType::kAllocTypeQueue &&
          tensor->second.alloc_type != ge::AllocType::kAllocTypeBuffer) {
        GELOGE(ge::FAILED, "Codegen tensor[%ld] is not alloc from que/buf", tid);
        return ge::FAILED;
      }

      if (first) {
        first = false;
      } else {
        tensor_size_sum << ", ";
        if (tensor->second.alloc_type == ge::AllocType::kAllocTypeQueue) {
          tensor_bufnum_max << ", ";
        }
      }

      std::string dtype_name;
      GE_CHK_STATUS_RET(Tensor::DtypeName(tensor->second.dtype, dtype_name), "Codegen get data type:%d failed",
                        static_cast<int32_t>(tensor->second.dtype));
      tensor_size_sum << tensor->second.size << " * " << "sizeof(" << dtype_name << ")";
      if (tensor->second.alloc_type == ge::AllocType::kAllocTypeQueue) {
        tensor_bufnum_max << tensor->second.que_buf_num;
      }
    }

    tensor_size_sum << ")";
    tensor_bufnum_max << ")";

    ss << merge_scope.size.DefineConst(tensor_size_sum.str()) << std::endl;
    ss << merge_scope.buf_num.DefineConst(tensor_bufnum_max.str()) << std::endl;
  }

  result = ss.str();
  return ge::SUCCESS;
}

Status TPipe::LocalTQueAlloc(std::string &result) const {
  stringstream ss;

  for (auto &[id, que] : this->ques) {
    if (id == this->cube_output_que_id) {
      continue;
    }
    stringstream tensor_size_max;
    stringstream tensor_bufnum_max;

    tensor_size_max << KernelUtils::Max() << "(";
    tensor_bufnum_max << KernelUtils::Max() << "(";

    bool is_first = true;

    for (auto mid : que.merge_scopes) {
      auto merge_scope = this->merge_scopes.find(mid);
      if (merge_scope == this->merge_scopes.end()) {
        GELOGE(ge::FAILED, "Codegen merge scope not found:%ld", mid);
        return ge::FAILED;
      }

      if (is_first) {
        is_first = false;
      } else {
        tensor_size_max << ", ";
        tensor_bufnum_max << ", ";
      }

      tensor_size_max << merge_scope->second.size;
      tensor_bufnum_max << merge_scope->second.buf_num;
    }

    uint32_t tensor_buf_num_max_val = 0;
    for (auto tid : que.not_merge_tensors) {
      auto tensor = this->tensors.find(tid);
      if (tensor == this->tensors.end()) {
        GELOGE(ge::FAILED, "Codegen tensor not found:%ld", tid);
        return ge::FAILED;
      }

      if (is_first) {
        is_first = false;
      } else {
        tensor_size_max << ", ";
      }

      std::string dtype_name;
      GE_CHK_STATUS_RET(Tensor::DtypeName(tensor->second.dtype, dtype_name), "Codegen get data type:%d failed",
                        static_cast<int32_t>(tensor->second.dtype));
      tensor_size_max << tensor->second.size << " * sizeof(" << dtype_name << ")";
      tensor_buf_num_max_val = std::max(tensor_buf_num_max_val, tensor->second.que_buf_num_value);
    }
    tensor_bufnum_max << (que.merge_scopes.empty() ? "" : ", ") << tensor_buf_num_max_val;
    for (auto share_tensors : que.share_group) {
      if (share_tensors.second.size() <= 1) {  // 仅有1个tensor 不需要合并计算
        continue;
      }
      // 此处肯定不是第一个size
      tensor_size_max << ", ";
      // 多个tensor在一个que中且reuse id相同，需要计算reuse id相同这一组所有tensor size的综合
      bool is_first_share = true;
      for (auto tid : share_tensors.second) {
        auto tensor = this->tensors.find(tid);
        GE_ASSERT_TRUE(tensor != this->tensors.end(), "Codegen share tensor not found:%ld", tid);
        if (is_first_share) {
          is_first_share = false;
        } else {
          tensor_size_max << " + ";
        }
        std::string dtype_name;
        GE_CHK_STATUS_RET(Tensor::DtypeName(tensor->second.dtype, dtype_name), "Codegen get data type:%d failed",
                          static_cast<int32_t>(tensor->second.dtype));
        tensor_size_max << tensor->second.size << " * sizeof(" << dtype_name << ")";
      }
    }

    tensor_size_max << ")";
    tensor_bufnum_max << ")";

    if (this->cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
      ss << que.size.DefineConst(tensor_size_max.str()) << std::endl;
      ss << que.buf_num.DefineConst(tensor_bufnum_max.str()) << std::endl;
    } else if (!using_att_calc_qbt_size_) {
      ss << que.size.DefineConst(tensor_size_max.str()) << std::endl;
      ss << que.buf_num.DefineConst(tensor_bufnum_max.str()) << std::endl;
      ss << que.Define() << std::endl;
    } else {
      ss << "// " << que.size.DefineConst(tensor_size_max.str()) << std::endl;
      ss << que.buf_num.DefineConst(tensor_bufnum_max.str()) << std::endl;
      ss << que.Define() << std::endl;
    }
    std::string init;
    GE_CHK_STATUS_RET(this->InitTQueBuffers(que, init), "Codegen init tque buffers failed");
    ss << init << std::endl;
  }

  result = ss.str();
  return ge::SUCCESS;
}

Status TPipe::BlkTensorAllocAndInit(std::string &result) const {
  // 遍历所有需要生成blk tensor 的 tensor id
  stringstream ss;
  for (auto &id : this->need_gen_blk_tensors) {
    auto tensor_ptr = this->GetTensor(id);
    GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, ge::FAILED, "BlkTensorAllocAndInit need_gen_blk_tensors failed");
    std::string scalar_t_buf_name = tensor_ptr->name + "_tbuf";
    std::string scalar_local_blk_tensor_name = "local_blk_tensor_of_" + tensor_ptr->name;
    ss << "TBuf<TPosition::VECCALC> " << scalar_t_buf_name << ";" << std::endl;
    ss << "tpipe.InitBuffer(" << scalar_t_buf_name << ", 32);" << std::endl;
    ss << "LocalTensor<" << tensor_ptr->type << "> " << scalar_local_blk_tensor_name << " = " << scalar_t_buf_name << ".Get<" << tensor_ptr->type << ">();" << std::endl;

    ss << "Duplicate(" << scalar_local_blk_tensor_name << "[0], static_cast<" << tensor_ptr->type
       << ">(" << tensor_ptr->const_value << "), static_cast<uint64_t>(32/"
       << "sizeof(" << tensor_ptr->type << ")));" << std::endl;
    ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;
  }
  result = ss.str();
  return ge::SUCCESS;
}

std::string TPipe::GenDuplicateBufAlloc(const std::set<std::pair<std::string, std::string>>& pre_api_extract_dup) const {
  std::stringstream ss;
  int32_t i = 1;
  for (auto [const_val, const_dtype] : pre_api_extract_dup) {
    const std::string index_str = std::to_string(i);
    ss << "TBuf<TPosition::VECCALC> builtin_tmp_buffer_" << index_str << ";" << std::endl;
    ss << "tpipe.InitBuffer(builtin_tmp_buffer_" << index_str << ", ONE_BLK_SIZE);" << std::endl;
    ss << "LocalTensor<uint8_t> builtin_tmp_buf_" << index_str << " = builtin_tmp_buffer_" << index_str <<
        ".Get<uint8_t>();" << std::endl;
    std::string local_tensor_name = "local_blk_tensor_of_" + const_dtype + "_" + const_val;
    ss << "LocalTensor<" << const_dtype << "> " << local_tensor_name <<
        " = builtin_tmp_buf_" << index_str << ".template ReinterpretCast<" << const_dtype << ">();" << std::endl;
    if (const_dtype == "half" || const_dtype == "float" || const_dtype == "double") {
      const_val += ".0";
    }
    ss << "Duplicate(" << local_tensor_name << "[0], (" << const_dtype << ")" << const_val <<
        ", ONE_BLK_SIZE / sizeof(" << const_dtype << "));"<< std::endl;
    i++;
  }
  return ss.str();
}

Status TPipe::LocalTBufAlloc(const TBuf &buf, std::string &result, const bool with_define) const {
  stringstream ss;
  std::string reuse_dtype_name = "";
  std::vector<const Tensor *> reuse_buf_tensors;
  bool is_buf_reuse = true;
  stringstream tensor_size_max;

  GE_CHK_STATUS_RET(ParseTBufReuse(buf, reuse_dtype_name, is_buf_reuse, reuse_buf_tensors,
                    tensor_size_max), "Codegen parse tbuf reuse failed");

  if (this->cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
    ss << buf.size.DefineConst(tensor_size_max.str()) << std::endl;
  } else if (!using_att_calc_qbt_size_) {
    ss << buf.size.DefineConst(tensor_size_max.str()) << std::endl;
    ss << buf.Define() << std::endl;
  } else {
    ss << "// " << buf.size.DefineConst(tensor_size_max.str()) << std::endl;
    ss << buf.Define() << std::endl;
  }
  std::string init;
  GE_CHK_STATUS_RET(this->InitTBufBuffer(buf, init), "Codegen init tbuf buffer failed");
  ss << init << std::endl;
  
  if (!is_buf_reuse) {
    ss << buf.AllocBuf(with_define) << std::endl;
  } else {
    ss << buf.AllocBuf(reuse_buf_tensors[0]->name, reuse_dtype_name, with_define) << std::endl;
    reuse_buf_tensors[0]->no_need_realloc = true;
    for (size_t i = 1UL; i < reuse_buf_tensors.size(); i++) {
      reuse_buf_tensors[i]->no_need_realloc = true;
      ss << "LocalTensor<" << reuse_dtype_name << "> " << reuse_buf_tensors[i]->name << " = "
          << reuse_buf_tensors[0]->name << ";" << std::endl;
    }
  }
  result = ss.str();
  return ge::SUCCESS;
}

Status TPipe::LocalTBufAllocLoopTwice(std::string &result, const bool with_define) const {
  stringstream ss;
  std::string tmp;
  for (auto &pair : this->bufs) {
    auto &buf = pair.second;
    if (!buf.tmp_buf_reuse) {
      GE_CHK_STATUS_RET(this->LocalTBufAlloc(buf, tmp, with_define), "Codegen TBuf alloc failed(no tmp buf).");
      ss << tmp;
    }
  }
  for (auto &pair : this->bufs) {
    auto &buf = pair.second;
    if (buf.tmp_buf_reuse) {
      GE_CHK_STATUS_RET(this->LocalTBufAlloc(buf, tmp, with_define), "Codegen TBuf alloc failed(tmp buf).");
      ss << tmp;
      ss << this->AllocTmpBuf(buf, with_define);
    }
  }
  ss << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

Status TPipe::LocalTensorQueBufAlloc(std::string &result) const {
  stringstream ss;
  std::string tmp;
  ss << this->TensorSizeCalc();
  GE_CHK_STATUS_RET(this->MergeScopeSizeCalc(tmp), "Codegen merge scope size failed");
  ss << tmp;
  ss << std::endl;
  GE_CHK_STATUS_RET(this->LocalTQueAlloc(tmp), "Codegen alloc local tque failed");
  ss << tmp;
  GE_CHK_STATUS_RET(this->LocalTBufAllocLoopTwice(tmp), "Codegen alloc local tbuf failed");
  ss << tmp << std::endl;

  result = ss.str();
  return ge::SUCCESS;
}

std::string TPipe::SyncMte3ToMte2(const Tensor in_tensor) const {
  stringstream ss;
  std::string event_name = in_tensor.Str() + "_e";
  std::string sync_name = in_tensor.Str() + "_s";

  ss << "auto " << event_name << " = tpipe.AllocEventID<HardEvent::MTE3_MTE2>();" << std::endl
      << "TQueSync<PIPE_MTE3, PIPE_MTE2> " << sync_name << ";" << std::endl
      << sync_name << ".SetFlag(" << event_name << ");" << std::endl
      << sync_name << ".WaitFlag(" << event_name << ");" << std::endl
      << "tpipe.ReleaseEventID<HardEvent::MTE3_MTE2>(" << event_name << ");" << std::endl;
  return ss.str();
}

std::string TPipe::SyncMte2ToMte3(const Tensor in_tensor) const {
  stringstream ss;
  std::string event_name = in_tensor.Str() + "_e";
  std::string sync_name = in_tensor.Str() + "_s";

  ss << "auto " << event_name << " = tpipe.AllocEventID<HardEvent::MTE2_MTE3>();" << std::endl
     << "TQueSync<PIPE_MTE2, PIPE_MTE3> " << sync_name << ";" << std::endl
     << sync_name << ".SetFlag(" << event_name << ");" << std::endl
     << sync_name << ".WaitFlag(" << event_name << ");" << std::endl
     << "tpipe.ReleaseEventID<HardEvent::MTE2_MTE3>(" << event_name << ");" << std::endl;
  return ss.str();
}

Status TPipe::CollectQues(const ascir::ImplGraph &graph) {
  std::unordered_map<ascir::QueId, ge::Position> que_id_to_src_position;
  std::set<ascir::QueId> need_bind_que_id;
  for (auto node : graph.GetAllNodes()) {
    if (node->attr.api.type == ge::ApiType::kAPITypeBuffer) {
      continue;
    }
    for (const auto &out_tensor : node->outputs()) {
      if (out_tensor->attr.mem.alloc_type != ge::AllocType::kAllocTypeQueue) {
        continue;
      }
      const int64_t tensor_que_id = out_tensor->attr.que.id;
      if (out_tensor->attr.mem.position == ge::Position::kPositionVecIn) {
        que_id_to_src_position.emplace(tensor_que_id, out_tensor->attr.mem.position);

        std::set<std::string> peer_node_types;
        for (const auto &peer_in_anchor : out_tensor->anchor.GetPeerInDataAnchorsPtr()) {
          if (peer_in_anchor != nullptr && peer_in_anchor->GetOwnerNodeBarePtr() != nullptr) {
            peer_node_types.emplace(peer_in_anchor->GetOwnerNodeBarePtr()->GetType());
          }
        }
        if ((peer_node_types.size() == 1U) && *peer_node_types.begin() == Store::Type) {
          need_bind_que_id.emplace(tensor_que_id);
        }
      } else if (out_tensor->attr.mem.position == ge::Position::kPositionVecOut) {
        que_id_to_src_position.emplace(tensor_que_id, out_tensor->attr.mem.position);
      }
    }
  }

  // Allocate
  for (const auto &iter : que_id_to_src_position) {
    if (this->ques.count(iter.first) > 0UL) {
      continue;
    }
    std::string position;
    GE_CHK_STATUS_RET(PositionValue(iter.second, position), "Codegen get position value failed");
    // 如果qid被非load->store的load复用，则不能使用TQueBind
    bool need_que_bind = need_bind_que_id.count(iter.first) > 0UL;
    if (need_que_bind) {
      std::string dst_position;
      GE_CHK_STATUS_RET(PositionValue(ge::Position::kPositionVecOut, dst_position),
                        "Codegen get position value failed");
      auto new_que = this->ques.emplace(iter.first, TQue{iter.first, iter.second, position, dst_position});
      GE_CHK_BOOL_RET_STATUS(new_que.second, ge::FAILED, "Codegen emplace que [%ld] failed", iter.first);
    } else {
      auto new_que = this->ques.emplace(iter.first, TQue{iter.first, iter.second, position});
      GE_CHK_BOOL_RET_STATUS(new_que.second, ge::FAILED, "Codegen emplace que [%ld] failed", iter.first);
    }
  }
  for (auto &[id, que] : this->ques) {
    if (id != this->cube_output_que_id) {
      que.is_cv_ub_fusion = (this->cv_fusion_type == ascir::CubeTemplateType::kUBFuse);
    }
  }
  return ge::SUCCESS;
}

void TPipe::SetUsingAttCalcQBTSizeConfig(bool using_att_calc_qbt_size) {
  using_att_calc_qbt_size_ = using_att_calc_qbt_size;
}

Status LoopAxisDistance(const std::vector<ascir::AxisId> &current_loop,
                        const std::vector<ascir::AxisId> &node_sched_axis, const ascir::AxisId node_loop_axis,
                        int32_t &distance) {
  if (node_sched_axis.size() == 0 || node_loop_axis == ge::kIdNone) {
    distance = -1 * current_loop.size();
    return ge::SUCCESS;
  }

  int32_t same_axis_num = 0;
  for (size_t i = 0; i < node_sched_axis.size() && i < current_loop.size(); ++i) {
    if (node_sched_axis[i] == current_loop[i]) {
      same_axis_num++;
    } else if (node_sched_axis[i] == node_loop_axis) {
      break;
    }
  }

  int32_t loop_axis_pos = -1;
  for (size_t i = 0; i < node_sched_axis.size(); ++i) {
    if (node_loop_axis == node_sched_axis[i]) {
      loop_axis_pos = i;
      break;
    }
  }

  if (loop_axis_pos < 0) {
    GELOGE(ge::FAILED, "Codegen node loop axis not found in node_sched_axis");
    return ge::FAILED;
  }

  if (static_cast<size_t>(same_axis_num) < current_loop.size()) {
    if (loop_axis_pos < same_axis_num) {
      distance = -(current_loop.size() - loop_axis_pos);
      return ge::SUCCESS;
    } else {
      distance = -(current_loop.size() - same_axis_num);
      return ge::SUCCESS;
    }
  } else {
    distance = (loop_axis_pos + 1) - current_loop.size();
    return ge::SUCCESS;
  }

  return ge::SUCCESS;
}

ApiTensor::ApiTensor()
    : id(ge::kIdNone),
      reuse_id(ge::kIdNone),
      reuse_from(nullptr),
      reuse_next(nullptr),
      share_prev(nullptr),
      share_next(nullptr),
      share_order(-1),
      write(nullptr) {}

Status ApiCall::Init(const ascir::NodeView &node) {
  this->unit = node->attr.api.unit;
  this->type = node->GetType();
  this->compute_type = node->attr.api.compute_type;
  for (auto tmp_buffer : node->attr.tmp_buffers) {
    if (tmp_buffer.id == -1L) {
      continue;
    }
    this->tmp_buf_id[tmp_buffer.buf_desc.life_time_axis_id] = tmp_buffer.id;
  }
  if (!IsOps<Output>(node)) {
    for (auto o : node->outputs()) {
      auto &t = this->outputs.emplace_back();
      t.id = o->attr.mem.tensor_id;
      t.reuse_id = o->attr.mem.reuse_id;
      t.write = this;
    }
  }
  GE_CHK_STATUS_RET(ParseAttr(node));
  return ge::SUCCESS;
}

Status ApiCall::ParseAttr(const ascir::NodeView &node) {
  (void) node;
  return ge::SUCCESS;
}

Status ZerosLikeApicall(std::string binaryname, const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                        const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                        const std::vector<std::reference_wrapper<const Tensor>> &outputs, const ApiAttr &api_attr,
                        std::string &result) {
  (void)binaryname;
  (void)api_attr;
  auto x = inputs[0].get();
  auto y = outputs[0].get();
  stringstream ss;
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(y.dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(y.dtype));
  std::string int64_tmp_buf = "";
  if (y.dtype == ge::DT_INT64) {
    int64_tmp_buf = ", " + tpipe.tmp_buf.name;
  }
  ss << "Duplicate(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], (" << dtype_name
     << ")0, " << x.actual_size << int64_tmp_buf << ");" << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

Status ApiCall::PreProcess(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                           const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                           std::string &result) const {
  stringstream ss;
  bool is_all_outputs_ub_scalar = std::all_of(outputs.begin(), outputs.end(),
      [](const std::reference_wrapper<const Tensor> &t) { return t.get().is_ub_scalar; });
  bool is_any_output_need_two_loop = std::any_of(outputs.begin(), outputs.end(),
      [](const std::reference_wrapper<const Tensor> &t) { return t.get().alloc_type == ge::AllocType::kAllocTypeQueue &&
          t.get().que_buf_num_value == 2 && t.get().need_gen_get_value_of_ub_scalar; });
  if (is_all_outputs_ub_scalar && !current_axis.empty()) {
    const auto loop_axis = tpipe.tiler.GetAxis(current_axis.back());
    // 如果当前节点输出tensor是ub_scalar，且ub的queue buffer num是2，且需要生成ub_scalar的get value代码
    // 则代表存在一个输出节点的输出tensor不是ub_scalar，此时下一个节点的计算不在if (loop_axis < 1)的逻辑包含中
    // 而下一个节点依赖当前节点的输出tensor，因此需要改成if (loop_axis < 2)，保证DOUBLE_BUFFER流程中两个buffer都被计算
    if (is_any_output_need_two_loop) {
      ss << "if (" << loop_axis << " < 2) {" << std::endl;
    } else {
      ss << "if (" << loop_axis << " < 1) {" << std::endl;
    }
  }

  result = ss.str();
  return ge::SUCCESS;
}

Status ApiCall::PostProcess(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                            const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                            std::string &result) const {
  (void)tpipe;
  stringstream ss;
  bool is_all_outputs_ub_scalar = std::all_of(outputs.begin(), outputs.end(),
      [](const std::reference_wrapper<const Tensor> &t) { return t.get().is_ub_scalar; });
  bool first_gen_get_value = true;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto &ub = outputs[i].get();
    if (ub.is_ub_scalar && !current_axis.empty()) {
      GELOGD("t_name:%s, need_gen_get_value_of_ub_scalar:%d", ub.Str().c_str(),
            static_cast<int32_t>(ub.need_gen_get_value_of_ub_scalar));
      // 生成ub_scalar的变量初始化定义
      if (ub.need_gen_get_value_of_ub_scalar) {
        if (first_gen_get_value) {
          std::string sync_type = (this->type == Load::Type) ? "MTE2_S" : "V_S";
          ss << "event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::";
          ss << sync_type;
          ss << "));" << std::endl;

          ss << "SetFlag<HardEvent::";
          ss << sync_type;
          ss << ">(eventID);" << std::endl;

          ss << "WaitFlag<HardEvent::";
          ss << sync_type;
          ss << ">(eventID);" << std::endl;

          first_gen_get_value = false;
        }
        std::string tmp;
        GE_CHK_STATUS_RET(ub.InitUbScalar(tmp));
        ss << tmp;
        if (ub.need_duplicate_value_of_ub_scalar) {
          GE_CHK_STATUS_RET(ub.GenDuplicateValueOfUbScalar(tmp));
          ss << tmp;
        }
      }
      if (is_all_outputs_ub_scalar) {
        ss << "}" << std::endl;
      }
    }
  }

  result = ss.str();
  return ge::SUCCESS;
}

Status ApiCall::GenerateFuncDefinition(const TPipe &tpipe, const Tiler &tiler, stringstream &ss) const {
  (void)tpipe;
  (void)tiler;
  (void)ss;
  return ge::SUCCESS;
}

Status ApiCall::GenerateMacro(std::string &result) const {
  (void)result;
  return ge::SUCCESS;
}

Status ApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                         std::string &result) const {
  std::vector<reference_wrapper<const Tensor>> input_tensors;
  for (const auto &in : this->inputs) {
    auto tensor_ptr = tpipe.GetTensor(in->id);
    GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, ge::FAILED, "Check[Param] tensor_ptr is nullptr");
    input_tensors.emplace_back(*tensor_ptr);
  }

  std::vector<reference_wrapper<const Tensor>> output_tensors;
  for (const auto &out : this->outputs) {
    auto tensor_ptr = tpipe.GetTensor(out.id);
    GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, ge::FAILED, "Check[Param] tensor_ptr is nullptr");
    output_tensors.emplace_back(*tensor_ptr);
  }

  stringstream ss;
  // apicall pre process
  std::string pre_result;
  GE_CHK_STATUS_RET(PreProcess(tpipe, current_axis, output_tensors, pre_result),
                    "Codegen generate API call pre_p failed");
  ss << pre_result;
  std::string local_result;
  GE_CHK_STATUS_RET(Generate(tpipe, current_axis, input_tensors, output_tensors, local_result),
                    "Codegen generate API call failed, api_name: %s", api_name_.c_str());
  ss << local_result;

  // apicall post precess
  std::string post_result;
  GE_CHK_STATUS_RET(PostProcess(tpipe, current_axis, output_tensors, post_result),
                    "Codegen generate API call post_p failed");
  ss << post_result;

  result = ss.str();
  return ge::SUCCESS;
}

static bool IsFirstRead(const ApiCall &call, const ApiTensor &tensor) {
  return !tensor.reads.empty() && tensor.reads[0] == &call;
}

bool IsUnitFirstRead(const ApiCall &call, const ApiTensor &tensor) {
  for (auto r : tensor.reads) {
    if (r->unit != call.unit) {
      continue;
    }

    if (r == &call) {
      return true;
    } else {
      return false;
    }
  }

  return false;
}

static bool IsLastRead(const ApiCall &call, const ApiTensor &tensor) {
  return !tensor.reads.empty() && tensor.reads.back() == &call;
}

static bool IsInplaceOutput(const ApiCall &call, const ApiTensor &tensor) {
  for (auto in : call.inputs) {
    if (in->reuse_next == &tensor) {
      return true;
    }
  }
  return false;
}

static bool IsInShareQue(const ApiTensor &tensor) {
  return (tensor.share_prev != nullptr) || (tensor.share_next != nullptr);
}

static bool IsFirstShare(const ApiTensor &tensor) {
  return (tensor.share_prev == nullptr);
}

static bool IsLastShare(const ApiTensor &tensor) {
  return (tensor.share_next == nullptr);
}

static bool IsReuseWithDefine(const ApiTensor &tensor) {
  if (tensor.reuse_from == nullptr) {
    return true;
  }
  if (tensor.reuse_from->write == nullptr) {
    return false;
  }
  ascir::AxisId cur_axis = tensor.write->axis;
  ascir::AxisId reuse_axis = tensor.reuse_from->write->axis;
  bool with_define = cur_axis == reuse_axis ? false : true;
  return with_define;
}

/*
 * que共用只存在于 多个load的输出连接同一vec运算算子的输入场景
 *  load0 load1 load2
 *     \   |     /
 *        vec
 *
 * load0 load1 load2 的输出共用一个que
 */
BoolType ApiCall::WaitShareInputs(const TPipe &tpipe, const ApiTensor *in, const Tensor t,
                                  std::stringstream &ss) const {
  if (IsInShareQue(*in)) {
    if (in->share_prev == nullptr && IsFirstRead(*this, *in)) {
      auto t_que = tpipe.GetQue(t.que_id);
      GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, BoolType::FAILED, "Codegen que[%ld] not found", t.que_id);
      ss << t_que->DequeBuf(false);
    }
    return BoolType::TRUE;
  }
  return BoolType::FALSE;
}

ge::Status DefineShareOffsets(const TPipe &tpipe, const ApiTensor &out, const Tensor &t, std::stringstream &ss) {
  std::map<int32_t, const ApiTensor *> order_to_tensor;
  for (auto it = out.share_prev; it != nullptr; it = it->share_prev) {
    order_to_tensor.emplace(it->share_order, it);
  }
  for (auto it = &out; it != nullptr; it = it->share_next) {
    order_to_tensor.emplace(it->share_order, it);
  }
  const auto cur_order = out.share_order;
  int32_t prev_max = -1;
  for (auto it = out.share_prev; it != nullptr; it = it->share_prev) {
    if (it->share_order > prev_max) {
      prev_max = it->share_order;
    }
  }
  for (int32_t i = prev_max + 1; i <= cur_order; ++i) {
    if (i == 0) {
      continue;
    }
    auto prev_var_name = t.que_share_offset.name;
    if (i - 1 > 0) {
      prev_var_name += ("_part_" + std::to_string(i - 1));
    }
    auto prev_tensor = tpipe.GetTensor(order_to_tensor[i - 1]->id);
    GE_ASSERT_NOTNULL(prev_tensor, "Check[Param] tensor_ptr is nullptr");
    auto size = ge::GetSizeByDataType(prev_tensor->dtype);
    auto var_size = prev_var_name + " + " + prev_tensor->size.name + " * " + std::to_string(size);
    const auto &cur_var_name = t.que_share_offset.name + "_part_" + std::to_string(i);
    decltype(t.que_share_offset) offset_var(cur_var_name);
    ss << offset_var.DefineConst(std::move(var_size)) << std::endl;
  }
  return ge::SUCCESS;
}

BoolType ApiCall::AllocShareOutputs(const TPipe &tpipe, const ApiTensor &out, const Tensor t,
                                    std::stringstream &ss) const {
  std::string relative_offset;
  if (IsFirstShare(out)) {
    // 第一个share tensor 或 仅复用的tenosr 初始化为 uint32_t q<id>_reuse<id>_offset = 0;
    ss << t.que_share_offset.Define("0") << std::endl;
  }
  if (IsInShareQue(out)) {
    if (this->unit == ge::ComputeUnit::kUnitMTE2 && t.position == ge::Position::kPositionVecIn) {
      if (out.reuse_from != nullptr && out.share_prev == nullptr) {
        bool with_define = IsReuseWithDefine(out);
        auto t_que = tpipe.GetQue(t.que_id);
        GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, BoolType::FAILED, "Codegen que[%ld] not found", t.que_id);
        ss << t_que->AllocBuf(with_define);
      } else if (out.reuse_from == nullptr && out.share_prev == nullptr) {
        auto t_que = tpipe.GetQue(t.que_id);
        GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, BoolType::FAILED, "Codegen que[%ld] not found", t.que_id);
        ss << t_que->AllocBuf();
      }
    }
    if (out.share_order == -1) {
      if (!IsFirstShare(out)) {
        auto tensor_ptr = tpipe.GetTensor(out.share_prev->id);
        GE_CHK_BOOL_RET_SPECIAL_STATUS(tensor_ptr == nullptr, BoolType::FAILED, "Check[Param] tensor_ptr is nullptr");
        auto prev_tensor = *tensor_ptr;
        auto size = ge::GetSizeByDataType(prev_tensor.dtype);
        relative_offset = t.que_share_offset.name + " + " + prev_tensor.size.name + " * " + std::to_string(size);
        ss << t.que_share_offset.Assign(relative_offset);
        ss << std::endl;
      }
    } else {
      // 需要按给定顺序计算offset
      GE_CHK_BOOL_RET_SPECIAL_STATUS(DefineShareOffsets(tpipe, out, t, ss), BoolType::FAILED);
      auto cur_var_name = t.que_share_offset.name;
      std::string offset = "0";
      if (out.share_order > 0) {
        offset = t.que_share_offset.name + "_part_" + std::to_string(out.share_order);
      }
      ss << t.que_share_offset.Assign(offset);
      ss << std::endl;
    }
    return BoolType::TRUE;
  }
  return BoolType::FALSE;
}

bool ApiCall::WaitInputVector(const TPipe &tpipe, const ApiTensor *in, const Tensor &t, std::stringstream &ss) const {
  if (t.position == ge::Position::kPositionVecIn && IsFirstRead(*this, *in)) {
    auto t_que = tpipe.GetQue(t.que_id);
    GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, false, "Codegen que[%ld] not found", t.que_id);
    if (t.que_id != tpipe.cube_output_que_id) {
      ss << t_que->DequeBuf(false);
    }
  } else if (t.position == ge::Position::kPositionVecCalc && IsFirstRead(*this, *in)) {
    ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;
  } else if (t.position == ge::Position::kPositionVecOut && IsUnitFirstRead(*this, *in)) {
    ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;
  }
  return true;
}

bool ApiCall::WaitInputMte(const TPipe &tpipe, const ApiTensor *in, const Tensor &t, std::stringstream &ss) const {
  // 1. load->store 2. load->store store 3. load->vec store store
  if (this->type == Store::Type &&
      ((in->write->compute_type == ascir::ComputeType::kComputeLoad) && (in->write->type != Gather::Type)) &&
      IsUnitFirstRead(*this, *in)) {
    ss << tpipe.SyncMte2ToMte3(t) << std::endl;
  }
  if ((t.position == ge::Position::kPositionVecOut) && IsUnitFirstRead(*this, *in)) {
    // 1. vec->store 2. vec->vec store store 3. vec->store vec store
    auto t_que = tpipe.GetQue(t.que_id);
    GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, false, "Codegen que[%ld] not found", t.que_id);
    ss << t_que->DequeBuf(false);
  } else if ((t.position == ge::Position::kPositionVecIn) && IsFirstRead(*this, *in)) {
    // 1. load->store
    auto t_que = tpipe.GetQue(t.que_id);
    GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, false, "Codegen que[%ld] not found", t.que_id);
    ss << t_que->DequeBuf(false);
  } else if (t.position == ge::Position::kPositionGM && IsUnitFirstRead(*this, *in) && in->write->type == Store::Type) {
    // store->workspace->load
    ss << tpipe.SyncMte3ToMte2(t) << std::endl;
  }
  return true;
}

bool ApiCall::WaitInputs(const TPipe &tpipe, std::stringstream &ss) const {
  std::vector<int64_t> handled_tensors;
  for (auto in : this->inputs) {
    auto it = std::find(handled_tensors.begin(), handled_tensors.end(), in->id);
    if (it != handled_tensors.end()) {
      GELOGI("WaitInputs tensor id[%ld] from same tensor", in->id);
      continue;
    }
    handled_tensors.push_back(in->id);
    auto tensor_ptr = tpipe.GetTensor(in->id);
    GE_CHK_BOOL_EXEC(tensor_ptr != nullptr, return false, "tensor_ptr nullptr");
    auto t = *tensor_ptr;
    if (this->unit == ge::ComputeUnit::kUnitVector) {
      BoolType ret = this->WaitShareInputs(tpipe, in, t, ss);
      GE_CHK_BOOL_RET_SPECIAL_STATUS(ret == BoolType::FAILED, false, "Func WaitShareInputs return FAILED");
      if (ret == BoolType::TRUE) {
        continue;
      }
      GE_CHK_BOOL_RET_SPECIAL_STATUS(!this->WaitInputVector(tpipe, in, t, ss), false,
                                     "Func WaitInputVector return false");
    } else if (this->unit == ge::ComputeUnit::kUnitMTE2 && (t.que_id != tpipe.cube_output_que_id)) {
      GE_CHK_BOOL_RET_SPECIAL_STATUS(!this->WaitInputMte(tpipe, in, t, ss), false, "Func WaitInputMte return false");
    }
  }
  return true;
}

Status ApiCall::HandleVecOutAlloc(const TPipe &tpipe, const ApiTensor &out, const Tensor &t, std::stringstream &ss,
                                  bool with_define) const {
  if (IsInplaceOutput(*this, out)) {
    return ge::SUCCESS;
  }
  if (out.reuse_from == nullptr) {
    auto t_que = tpipe.GetQue(t.que_id);
    GE_CHK_BOOL_RET_STATUS(t_que != nullptr, ge::FAILED, "Codegen que[%ld] not found", t.que_id);
    ss << t_que->AllocBuf();
    return ge::SUCCESS;
  }
  if (out.reuse_from->write->unit == ge::ComputeUnit::kUnitVector) {
    auto tensor_ptr = tpipe.GetTensor(out.reuse_from->id);
    GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, ge::FAILED, "Check[Param] tensor_ptr is nullptr");
    if (tensor_ptr->position == ge::Position::kPositionVecOut) {
      auto t_que = tpipe.GetQue(t.que_id);
      GE_CHK_BOOL_RET_STATUS(t_que != nullptr, ge::FAILED, "Codegen que[%ld] not found", t.que_id);
      ss << t_que->AllocBuf(with_define);
    } else {
      ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;
    }
    return ge::SUCCESS;
  }
  if (out.reuse_from->write->unit == ge::ComputeUnit::kUnitMTE2) {
    auto t_que = tpipe.GetQue(t.que_id);
    GE_CHK_BOOL_RET_STATUS(t_que != nullptr, ge::FAILED, "Codegen que[%ld] not found", t.que_id);
    ss << t_que->AllocBuf(with_define);
  }
  return ge::SUCCESS;
}

Status ApiCall::AllocOutputs(const TPipe &tpipe, std::stringstream &ss, bool create_sync) const {
  if (!create_sync) {
    return ge::SUCCESS;
  }
  for (auto &out : this->outputs) {
    auto tensor_ptr = tpipe.GetTensor(out.id);
    GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, ge::FAILED, "Check[Param] tensor_ptr is nullptr");
    auto t = *tensor_ptr;

    if (t.alloc_type == ge::AllocType::kAllocTypeBuffer) {
      if (out.reuse_from != nullptr) {
        ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;
      }
      ss << tpipe.TensorActualSizeCalc(t.id);
      std::string tmp;
      if (!t.no_need_realloc) {
        GE_CHK_STATUS_RET(tpipe.TensorAlloc(t, tmp), "Codegen alloc tensor failed");
      }
      ss << tmp;
      continue;
    }
    if (t.alloc_type != ge::AllocType::kAllocTypeQueue) {
      continue;
    }
    bool with_define = IsReuseWithDefine(out);
    BoolType ret = this->AllocShareOutputs(tpipe, out, t, ss);
    GE_CHK_BOOL_RET_STATUS(ret != BoolType::FAILED, ge::FAILED, "AllocShareOutputs return BoolType::FAILED");
    if (ret == BoolType::TRUE) {
      GELOGI("tensor id %ld alloc with share", out.id);
    } else if (this->unit == ge::ComputeUnit::kUnitMTE2 && t.position == ge::Position::kPositionVecIn) {
      if (out.reuse_from != nullptr) {
        auto t_que = tpipe.GetQue(t.que_id);
        GE_CHK_BOOL_RET_STATUS(t_que != nullptr, ge::FAILED, "Codegen que[%ld] not found", t.que_id);
        ss << t_que->AllocBuf(with_define);
      } else {
        auto t_que = tpipe.GetQue(t.que_id);
        GE_CHK_BOOL_RET_STATUS(t_que != nullptr, ge::FAILED, "Codegen que[%ld] not found", t.que_id);
        ss << t_que->AllocBuf();
      }
    } else if (this->unit == ge::ComputeUnit::kUnitVector && t.position == ge::Position::kPositionVecOut) {
      if (HandleVecOutAlloc(tpipe, out, t, ss, with_define) != ge::SUCCESS) {
        return ge::FAILED;
      }
    } else if (this->unit == ge::ComputeUnit::kUnitVector && t.position == ge::Position::kPositionVecCalc) {
      if (out.reuse_from == nullptr) {
        // vec ..> store
        // vec ..> vec ..> store
        auto t_que = tpipe.GetQue(t.que_id);
        GE_CHK_BOOL_RET_STATUS(t_que != nullptr, ge::FAILED, "Codegen que[%ld] not found", t.que_id);
        ss << t_que->AllocBuf();
      } else {
        auto tensor_ptr = tpipe.GetTensor(out.reuse_from->id);
        GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, ge::FAILED, "Check[Param] tensor_ptr is nullptr");
        if (tensor_ptr->position == ge::Position::kPositionVecOut) {
          // vec -> store -> free ..> alloc -> vec
          auto t_que = tpipe.GetQue(t.que_id);
          GE_CHK_BOOL_RET_STATUS(t_que != nullptr, ge::FAILED, "Codegen que[%ld] not found", t.que_id);
          ss << t_que->AllocBuf(with_define);
        } else {
          // load ..> vec
          // vec ..> vec
          ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;
        }
      }
    } else {
      std::runtime_error("Unsupported case.");
    }
    ss << tpipe.TensorActualSizeCalc(t.id);
    std::string tmp;
    if (!(t.alloc_type == ge::AllocType::kAllocTypeBuffer && t.no_need_realloc)) {
      GE_CHK_STATUS_RET(tpipe.TensorAlloc(t, tmp), "Codegen alloc tensor failed");
    }
    ss << tmp;
  }  // end for
  return ge::SUCCESS;
}

bool ApiCall::SyncOutputs(const TPipe &tpipe, std::stringstream &ss) const {
  for (auto out : this->outputs) {
    auto tensor_ptr = tpipe.GetTensor(out.id);
    GE_CHK_BOOL_EXEC(tensor_ptr != nullptr, return false, "tensor_ptr nullptr");
    auto t = *tensor_ptr;
    if (t.alloc_type == ge::AllocType::kAllocTypeQueue) {
      if (IsLastShare(out)) {  // 非共用 或者 最后一个共用
        if (t.position == ge::Position::kPositionVecIn || t.position == ge::Position::kPositionVecOut) {
          auto t_que = tpipe.GetQue(t.que_id);
          GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, false, "Codegen que[%ld] not found", t.que_id);
          ss << t_que->EnqueBuf();
        }
      }
    }
  }
  return true;
}

bool ApiCall::IsReadOutersideWrite(ascir::AxisId &target_id) const {
  uint64_t count = 0;
  uint64_t total_count = 0;
  int64_t prev_depth = INT64_MAX;
  for (auto output : outputs) {
    for (auto read : output.reads) {
      total_count++;
      if (read->depth < depth) {
        count++;
        target_id = read->depth < prev_depth ? read->axis : target_id;
        prev_depth = read->depth < prev_depth ? read->depth : prev_depth;
      }
    }
  }
  return count == total_count;
}

bool ApiCall::FreeInputs(const TPipe &tpipe, std::stringstream &ss) const {
  std::vector<ascir::QueId> freed_que;
  for (auto in : this->inputs) {
    if (!IsLastRead(*this, *in)) {
      continue;
    }
    auto tensor_ptr = tpipe.GetTensor(in->id);
    GE_CHK_BOOL_EXEC(tensor_ptr != nullptr, return false, "tensor_ptr nullptr");
    auto t = *tensor_ptr;
    auto reuse_next = in->reuse_next == nullptr ? nullptr : tpipe.GetTensor(in->reuse_next->id);
    auto it = find(freed_que.begin(), freed_que.end(), t.que_id);
    if (it != freed_que.end()) {
      continue;
    }
    if (!IsLastShare(*in)) {
      continue;
    }
    if (t.alloc_type == ge::AllocType::kAllocTypeQueue) {
      if (t.que_id == tpipe.cube_output_que_id) {
        continue;
      }
      auto t_que = tpipe.GetQue(t.que_id);
      if (reuse_next == nullptr) {
        // 1 alloc -> load ..> vec ..> vec ..> vec -> free
        // 2 alloc -> vec ..> vec ..> store -> free
        // 3 alloc -> vec ..> store -> free -> alloc ..> vec -> free
        GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, false, "Codegen que[%ld] not found", t.que_id);
        ss << t_que->FreeBuf();
        freed_que.push_back(t.que_id);
      } else if (t.position == ge::Position::kPositionVecOut) {
        // vec ..> store -> free -> alloc -> vec
        GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, false, "Codegen que[%ld] not found", t.que_id);
        ss << t_que->FreeBuf();
        freed_que.push_back(t.que_id);
      } else if (reuse_next != nullptr && reuse_next->position == ge::Position::kPositionVecIn) {
        // alloc -> load -> vec -> free ..> alloc -> load
        GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, false, "Codegen que[%ld] not found", t.que_id);
        ss << t_que->FreeBuf();
        freed_que.push_back(t.que_id);
      }
    }
  }
  return true;
}

bool ApiCall::FreeUnusedOutputs(const TPipe &tpipe, std::stringstream &ss) const {
  std::vector<ascir::QueId> freed_que;
  for (auto out : this->outputs) {
    if (out.reads.size() != 0) {
      continue;
    }
    auto tensor_ptr = tpipe.GetTensor(out.id);
    GE_CHK_BOOL_EXEC(tensor_ptr != nullptr, return false, "tensor_ptr nullptr");
    auto t = *tensor_ptr;
    auto it = find(freed_que.begin(), freed_que.end(), t.que_id);
    if (it != freed_que.end()) {
      continue;
    }
    if (t.alloc_type == ge::AllocType::kAllocTypeQueue) {
      if (out.reuse_next == nullptr && IsLastShare(out)) {
        auto t_que = tpipe.GetQue(t.que_id);
        GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, false, "Codegen que[%ld] not found", t.que_id);
        ss << t_que->FreeBuf();
        freed_que.push_back(t.que_id);
      } else if (t.position == ge::Position::kPositionVecOut) {
        auto t_que = tpipe.GetQue(t.que_id);
        GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, false, "Codegen que[%ld] not found", t.que_id);
        ss << t_que->FreeBuf();
        freed_que.push_back(t.que_id);
      }
    }
  }
  return true;
}
Status ApiCall::Generate(const TPipe &tpipe, const vector<ascir::AxisId> &current_axis,
                         const vector<std::reference_wrapper<const Tensor>> &input,
                         const vector<std::reference_wrapper<const Tensor>> &output, string &result) const {
  (void) tpipe;
  (void) current_axis;
  (void) input;
  (void) output;
  (void) result;
  return ge::SUCCESS;
}

bool ApiCall::IsUnitLastRead(const ApiTensor &tensor) const {
  for (int32_t i = tensor.reads.size() - 1; i >= 0; i--) {
    if (tensor.reads[i]->unit == this->unit) {
      return (tensor.reads[i] == this);
    }
  }
  return false;
}

Loop::Loop(const ascir::AxisId axis) : axis_id(axis), parent(nullptr) {}

void Loop::AddLoop(Loop *loop) {
  LoopBody tmp;
  tmp.type = LoopType::LOOP;
  tmp.loop = loop;
  tmp.loop->is_graph_has_reduce_node = this->is_graph_has_reduce_node;
  tmp.loop->is_ar = this->is_ar;
  this->bodys.emplace_back(tmp);
  loop->parent = this;
}

void Loop::AddCall(ApiCall *call) {
  LoopBody tmp;
  tmp.type = LoopType::CALL;
  tmp.call = call;
  this->bodys.emplace_back(tmp);
}

static bool IsReduceOp(const ascir::NodeView &node) {
  return IsOps<Max>(node) || IsOps<Min>(node) || IsOps<Sum>(node) || IsOps<Mean>(node) || IsOps<Prod>(node) ||
         IsOps<All>(node) || IsOps<Any>(node);
}

static bool IsRemovePadLinkBroadcast(const ascir::NodeView &node) {
  return IsOps<RemovePad>(node) && node->GetInDataNodesSize() == 1UL && IsOps<Broadcast>(node->GetInDataNodes().at(0));
}

static bool IsLoadNodeSplitB(const ascir::NodeView &node, const Tiler &tiler, std::string &enable_cache_with_condition,
                             bool is_ar, bool is_link_to_brc) {
  auto out = node->outputs[0];
  bool split_b = false;
  std::ostringstream ss;
  int32_t matching_counts = 0;
  int32_t matching_success_counts = 0;
  int32_t matching_success_current = 0;
  for (auto axis : out.attr.vectorized_axis) {
    if (axis == ge::kIdNone || tiler.GetAxis(axis).type != ascir::Axis::Type::kAxisTypeTileInner) {
      continue;
    }
    auto axis_iter = std::find(out.attr.axis.begin(), out.attr.axis.end(), axis);
    if (axis_iter != out.attr.axis.end()) {
      matching_counts++;
      bool res = (out.attr.strides.at(axis_iter - out.attr.axis.begin()) == 0);
      if (res) {
        matching_success_current = matching_counts;
        matching_success_counts++;
      }
      split_b = split_b || res;
    }
  }

  if (matching_success_counts > 1) {
    ss << kEnCacheR;
  } else if (matching_success_counts == 1) {
    const bool enableCacheA = (matching_success_current == 1 && is_ar) || (matching_success_current != 1 && !is_ar);

    ss << (enableCacheA ? kEnCacheA : kEnCacheR);
    if (enableCacheA) {
      ss << " || control_dis_enable_cache_a";
    }
  }

  enable_cache_with_condition = ss.str();

  if (is_link_to_brc) {
    return split_b;
  } else {
    const auto platform = optimize::PlatformFactory::GetInstance().GetPlatform();
    GE_ASSERT_NOTNULL(platform);
    return split_b && IsLinkToBrdcst(node, platform->BroadcastTypes());
  }
}

static bool IsNodeSplitB(const ascir::NodeView &node, const Tiler &tiler, std::string &enable_cache_with_condition,
                         bool is_ar, bool is_link_to_brc = false) {
  if (IsOps<Data>(node) || node->GetInDataNodesSize() == 0U) {
    return false;
  }

  if (node->attr.api.compute_type == ge::ComputeType::kComputeLoad) {
    return IsLoadNodeSplitB(node, tiler, enable_cache_with_condition, is_ar, is_link_to_brc);
  }

  const auto platform = optimize::PlatformFactory::GetInstance().GetPlatform();
  GE_ASSERT_NOTNULL(platform);
  bool node_link_to_brc = IsLinkToBrdcst(std::dynamic_pointer_cast<ge::AscNode>(node), platform->BroadcastTypes());
  bool remove_pad_link_brc = IsRemovePadLinkBroadcast(std::dynamic_pointer_cast<ge::AscNode>(node));
  if (!node_link_to_brc && !remove_pad_link_brc) {
    return false;
  }
  for (const auto &in_node : node->GetInDataNodes()) {
    GE_ASSERT_NOTNULL(in_node, "Input of node %s[%s] is null", node->GetTypePtr(), node->GetNamePtr());
    GE_ASSERT_NOTNULL(std::dynamic_pointer_cast<ge::AscNode>(in_node));
    const auto &prev_node = std::dynamic_pointer_cast<ge::AscNode>(in_node);
    if (!IsNodeSplitB(prev_node, tiler, enable_cache_with_condition, is_ar, true)) {
      return false;
    }
  }
  return true;
}

static bool IsValidCacheCondition(const ge::ExecuteCondition &exec_condition) {
  return exec_condition == ge::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis ||
         exec_condition == ge::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis;
}

static void TraverseGraphForReduceNodes(ascir::NodeViewVisitorConst nodes, bool &is_graph_has_reduce_node,
                                        bool &is_ar) {
  for (auto node : nodes) {
    if (IsReduceOp(node)) {
      is_graph_has_reduce_node = true;
      ge::AscNodeOutputs node_outputs = node->outputs;
      if (!node_outputs().empty() && !node_outputs[0].attr.vectorized_strides.empty()) {
        is_ar = node_outputs[0].attr.vectorized_strides.back() == 0;
        return;
      }
    }
  }
  is_graph_has_reduce_node = false;
  return;
}

static int64_t GetLifecycleEdge(ascir::NodeViewVisitorConst nodes, const TPipe &tpipe) {
  int64_t lifecycle_edge = 0;
  if (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
    for (const auto &node : nodes) {
      if (IsOps<Load>(node) && (node->outputs[0].attr.mem.tensor_id == tpipe.cube_output_tensor_id)) {
        for (const auto &out_node : node->GetOutDataNodesPtr()) {
          lifecycle_edge = std::max(lifecycle_edge, out_node->GetOpDescBarePtr()->GetId());
        }
      }
    }
  }
  return lifecycle_edge;
}
 
static void InitApiCallContext(const ascir::NodeView &node, const TPipe &tpipe, ApiCall *call, int64_t lifecycle_edge) {
  if (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
    int64_t node_topo_id = node->GetOpDescBarePtr()->GetId();
    if (IsOps<Load>(node) && (node->outputs[0].attr.mem.tensor_id == tpipe.cube_output_tensor_id)) {
      call->api_call_context.scene = ApiScene::kCVFuseUBLoad;
    }
    if (node_topo_id <= lifecycle_edge) {
      call->api_call_context.stage = ComputeStage::kCVFuseStage1;
    } else {
      call->api_call_context.stage = ComputeStage::kCVFuseStage2;
    }
  }
}
 
Status Loop::ConstructFromNodes(ascir::NodeViewVisitorConst nodes, const Tiler &tiler, TPipe &tpipe) {
  auto current_loop = this;
  std::vector<ascir::AxisId> current_axis;

  std::map<ascir::TensorId, ApiCall *> tensor_calls;
  map<ascir::BufId, ApiTensor *> buf_last_use;
  map<ascir::QueId, ApiTensor *> que_last_use;
  map<ascir::QueId, map<ascir::ReuseId, ApiTensor *>> que_last_share;
  TraverseGraphForReduceNodes(nodes, current_loop->is_graph_has_reduce_node, current_loop->is_ar);
  auto lifecycle_edge = GetLifecycleEdge(nodes, tpipe);
  for (auto node : nodes) {
    // Loop enter or create
    GELOGI("node:%s, ComputeUnit:%u\r\n", node->GetNamePtr(), static_cast<uint32_t>(node->attr.api.unit));
    if (node->attr.api.unit != ge::ComputeUnit::kUnitNone) {
      auto node_axis = node->attr.sched.axis;
      auto node_loop_axis = node->attr.sched.loop_axis;
      int32_t loop_distance;
      GE_CHK_STATUS_RET(LoopAxisDistance(current_axis, node_axis, node_loop_axis, loop_distance),
                        "Codegen get loop axis distance failed");
      while (loop_distance != 0) {
        if (loop_distance > 0) {
          auto axis = node_axis[current_axis.size()];
          current_axis.push_back(axis);
          current_loop->AddLoop(new Loop(axis));
          current_loop = current_loop->bodys.back().loop;
        } else {
          current_axis.pop_back();
          current_loop = current_loop->parent;
        }

        GE_CHK_STATUS_RET(LoopAxisDistance(current_axis, node_axis, node_loop_axis, loop_distance),
                          "Codegen get loop axis distance failed");
      }
    }

    // Add call
    auto call = CreateApiCallObject(node);
    GE_ASSERT_NOTNULL(call, "Create api call object failed, ascir type:%s", node->GetTypePtr());
    call->api_attr.offset = Zero;
    GE_CHK_STATUS_RET(call->Init(node), "ApiCall Init failed, ascir type:%s", node->GetTypePtr());
    current_loop->AddCall(call);
    call->exec_condition = node->attr.sched.exec_condition;
    call->enable_cache = this->is_graph_has_reduce_node
                              ? IsNodeSplitB(node, tiler, call->enable_cache_with_condition, current_loop->is_ar)
                              : IsValidCacheCondition(call->exec_condition);
    call->axis = current_loop->axis_id;
    call->depth = current_axis.size();
    InitApiCallContext(node, tpipe, call, lifecycle_edge);
    const auto is_cont_buf_required = call->IsContiguousBufRequired();
    int32_t input_index = 0;
    for (auto in : node->inputs()) {
      if (in == nullptr) {
        call->inputs.emplace_back(nullptr);
        continue;
      }

      auto in_call = tensor_calls.find(in->attr.mem.tensor_id);
      GE_CHK_BOOL_RET_STATUS(in_call != tensor_calls.end(), ge::FAILED,
                             "Codegen node[%s] no API call found for input tensor id[%ld]", node->GetNamePtr(),
                             in->attr.mem.tensor_id);

      auto in_index = ge::ascir::AscTensorUtils::Index(*in);
      auto in_tensor = &in_call->second->outputs[in_index];
      if (is_cont_buf_required) {
        in_tensor->share_order = input_index;
      }
      in_tensor->reads.push_back(call);
      call->inputs.emplace_back(in_tensor);
      GELOGI("node[%s] input tensor id[%ld] from call type[%s] outputs[%d], read by call type[%s]", node->GetNamePtr(),
             in->attr.mem.tensor_id, in_call->second->type.c_str(), in_index, call->type.c_str());
      ++input_index;
    }

    if (IsOps<Output>(node)) {
      continue;
    }
    for (auto out : node->outputs()) {
      tensor_calls.insert({out->attr.mem.tensor_id, call});

      auto out_index = ge::ascir::AscTensorUtils::Index(*out);
      if (out->attr.mem.alloc_type == ge::AllocType::kAllocTypeQueue) {
        GELOGI("Que[%ld] update last use call type[%s] output[%d]", out->attr.que.id, call->type.c_str(), out_index);
        GE_CHK_BOOL_RET_STATUS(out->attr.que.id != ge::kIdNone && out->attr.mem.reuse_id != ge::kIdNone, ge::FAILED,
                               "ConstructFromNodes tensor[%ld] que id[%ld] or reuse id[%ld] invalid",
                               call->outputs[out_index].id, out->attr.que.id, out->attr.mem.reuse_id);
        map<ascir::ReuseId, ApiTensor *> &last_share = que_last_share[out->attr.que.id];
        auto share_tensor = last_share.find(out->attr.mem.reuse_id);
        if (share_tensor != last_share.end()) {
          auto t_ptr = tpipe.GetTensor(out->attr.mem.tensor_id);
          auto t_share_prev_ptr = tpipe.GetTensor(share_tensor->second->id);
          GE_CHK_BOOL_RET_STATUS(t_ptr != nullptr, ge::FAILED, "Check[Param] t_ptr is nullptr");
          GE_CHK_BOOL_RET_STATUS(t_share_prev_ptr != nullptr, ge::FAILED, "Check[Param] t_share_prev_ptr is nullptr");
          auto &t = *t_ptr;
          auto &t_share_prev = *t_share_prev_ptr;
          t.share_pre_size = t_share_prev.size.name;
          share_tensor->second->share_next = &call->outputs[out_index];
          call->outputs[out_index].share_prev = share_tensor->second;
          GELOGI("Que[%ld] reuse id[%ld] tensor id[%ld] share with id[%ld]", out->attr.que.id, out->attr.mem.reuse_id,
                 call->outputs[out_index].id, share_tensor->second->id);
        }
        last_share[out->attr.mem.reuse_id] = &call->outputs[out_index];

        auto reused_tensor = que_last_use.find(out->attr.que.id);
        if (reused_tensor != que_last_use.end()) {
          if (reused_tensor->second->reuse_id != call->outputs[out_index].reuse_id) {
            reused_tensor->second->reuse_next = &call->outputs[out_index];
            call->outputs[out_index].reuse_from = reused_tensor->second;
            GELOGI("Que[%ld] reuse id[%ld] tensor id[%ld] reuse from tensor id[%ld] reuse id[%ld]", out->attr.que.id,
                   out->attr.mem.reuse_id, call->outputs[out_index].id, reused_tensor->second->id,
                   reused_tensor->second->reuse_id);
          } else {
            GELOGI("Que[%ld] reuse id[%ld] tensor id[%ld] share with last same que tensor", out->attr.que.id,
                   out->attr.mem.reuse_id, call->outputs[out_index].id, reused_tensor->second->id);
          }
        }
        que_last_use[out->attr.que.id] = &call->outputs[out_index];
      } else if (out->attr.mem.alloc_type == ge::AllocType::kAllocTypeBuffer) {
        GELOGI("Buf[%ld] update last use call type[%s] output[%d]", out->attr.buf.id, call->type.c_str(), out_index);
        auto reused_tensor = buf_last_use.find(out->attr.buf.id);
        if (reused_tensor != buf_last_use.end()) {
          reused_tensor->second->reuse_next = &call->outputs[out_index];
          call->outputs[out_index].reuse_from = reused_tensor->second;
          GELOGI("Buf[%ld] tensor id[%ld] reuse from tensor id[%ld]", out->attr.buf.id, call->outputs[out_index].id,
                 reused_tensor->second->id);
        }
        buf_last_use[out->attr.buf.id] = &call->outputs[out_index];
      }
    }
  }
  return ge::SUCCESS;
}

void Loop::Destruct() {
  for (auto body : this->bodys) {
    if (body.type == LoopType::LOOP) {
      body.loop->Destruct();
      delete body.loop;
    } else if (body.type == LoopType::CALL) {
      delete body.call;
    }
  }
}

void Loop::CollectTensorCrossLoop(std::map<ascir::AxisId, std::vector<ApiCall *>> &api_calls) {
  if (this->bodys.size() <= 1) {
    return;
  }
  for (auto body : this->bodys) {
    if (body.type == LoopType::LOOP) {
      for (auto inner_body : body.loop->bodys) {
        if (inner_body.type != LoopType::CALL) {
          inner_body.loop->CollectTensorCrossLoop(api_calls);
          continue;
        }
        ascir::AxisId target_axis;
        bool flag = inner_body.call->IsReadOutersideWrite(target_axis);
        if (flag) {
          api_calls[target_axis].emplace_back(inner_body.call);
        }
      }
    }
  }
  return;
}

bool Loop::IsFindInUsedCalls(const ApiCall *call) const {
  if (parent == nullptr) {
    return false;
  }
  if (parent->used_calls.count(call) > 0) {
    return true;
  }
  return parent->IsFindInUsedCalls(call);
}

bool Loop::IsBodyContainLoop() const {
  size_t loop_count = 0;
  for (auto &body : bodys) {
    if (body.type == LoopType::LOOP) {
      loop_count++;
    }
  }
  return loop_count != 0;
}

static bool IsReduceDoubleTile(const Tiler &tiler, const TPipe &tpipe, bool has_reduce_node) {
  (void)tiler;
  for (const auto &tensor : tpipe.tensors) {
    size_t tile_inner_size = 0;
    for (auto axis_id : tensor.second.vectorized_axis) {
      auto &axis = tpipe.tiler.GetAxis(axis_id);
      if (axis.type == ascir::Axis::Type::kAxisTypeTileInner) {
        tile_inner_size += 1;
      }
    }
    if (tile_inner_size < kDoubleAxisSize) {
      continue;
    }
    return has_reduce_node;
  }
  return false;
}

Status Loop::GenerateBody(const Tiler &tiler, const TPipe &tpipe, std::vector<ascir::AxisId> &current_axis,
                          std::stringstream &ss) {
  bool need_collect = this->bodys.size() > 1;
  std::map<ascir::AxisId, std::vector<ApiCall *>> api_calls_cross_loop;
  if (need_collect) {
    CollectTensorCrossLoop(api_calls_cross_loop);
  }
  auto target_calls = api_calls_cross_loop[this->axis_id];

  for (const auto &body : this->bodys) {
    if ((body.type == LoopType::CALL) && (body.call->api_call_context.scene == ApiScene::kCVFuseUBLoad ||
         body.call->api_call_context.stage != this->compute_stage)) {
      continue;
    }
    if (body.type == LoopType::LOOP) {
      for (auto call : target_calls) {
        GE_CHK_STATUS_RET(call->AllocOutputs(tpipe, ss), "Codegen alloc outputs failed");
        used_calls.insert(call);
      }
      body.loop->compute_stage = this->compute_stage;
      GE_CHK_STATUS_RET(body.loop->GenerateLoop(tiler, tpipe, current_axis, ss), "Generate loop for body failed");
      for (auto call : target_calls) {
        GE_CHK_BOOL_RET_STATUS(call->SyncOutputs(tpipe, ss), ge::FAILED, "Func SyncOutputs return false");
      }
      used_calls.clear();
    } else {
      if (body.call->unit == ge::ComputeUnit::kUnitNone) {
        continue;
      }
      GE_CHK_BOOL_RET_STATUS(body.call->WaitInputs(tpipe, ss), ge::FAILED, "Func WaitInputs return false");
      if (!IsFindInUsedCalls(body.call)) {
        GE_CHK_STATUS_RET(body.call->AllocOutputs(tpipe, ss), "Codegen alloc outputs failed");
      }
      std::string call;

      if (this->axis_id != ge::kIdNone) {
        auto axis = tiler.GetAxis(this->axis_id);
        bool is_enable_cache = axis.is_split_b && body.call->enable_cache;
        bool is_double_tile = IsReduceDoubleTile(tiler, tpipe, this->is_graph_has_reduce_node) &&
                              current_axis.size() > kDoubleTileAxisSize;
        if (is_enable_cache && is_double_tile) {
          ss << "if (" << body.call->enable_cache_with_condition << ") {" << std::endl;
        } else if (is_enable_cache && !this->is_graph_has_reduce_node) {
          if (body.call->exec_condition == ge::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis) {
            ss << "if (" << kEnCacheOriginBroadcastAxis << ") {" << std::endl;
          } else if (body.call->exec_condition == ge::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis) {
            ss << "if (" << kEnCacheFusedBroadcastAxis << ") {" << std::endl;
          }
        }
      }
      GE_CHK_STATUS_RET(body.call->Generate(tpipe, current_axis, call), "Codegen generate call failed");
      ss << call;

      if (this->axis_id != ge::kIdNone) {
        auto axis = tiler.GetAxis(this->axis_id);
        bool is_enable_cache = axis.is_split_b && body.call->enable_cache;
        bool is_double_tile = IsReduceDoubleTile(tiler, tpipe, this->is_graph_has_reduce_node) &&
                              current_axis.size() > kDoubleTileAxisSize;
        if (is_enable_cache && (is_double_tile || !this->is_graph_has_reduce_node)) {
          ss << "}" << std::endl;
        }
      }

      if (!IsFindInUsedCalls(body.call)) {
        GE_CHK_BOOL_RET_STATUS(body.call->SyncOutputs(tpipe, ss), ge::FAILED, "Func SyncOutputs return false");
      }
      GE_CHK_BOOL_RET_STATUS(body.call->FreeInputs(tpipe, ss), ge::FAILED, "Func FreeInputs return false");
      GE_CHK_BOOL_RET_STATUS(body.call->FreeUnusedOutputs(tpipe, ss), ge::FAILED,
                             "Func FreeUnusedOutputs return false");
      ss << std::endl;
    }
  }

  return ge::SUCCESS;
}

std::string Loop::GetReduceType() const {
  std::vector<std::string> reduce_map = {Max::Type, Sum::Type, Min::Type, Mean::Type, Prod::Type};
  for (auto body : this->bodys) {
    if (body.type == LoopType::CALL) {
      for (size_t i = 0; i < reduce_map.size(); ++i) {
        if (reduce_map[i] == body.call->type) {
          return reduce_map[i];
        }
      }
    }
  }
  GELOGI("No Reduce type found");
  return "";
}

bool Loop::IsReduceAxisNeedDivideSum(const TPipe &tpipe) const {
  (void)tpipe;
  auto reduce_type = GetReduceType();
  return false;
}

/* 获取reduce api的输入/输出tensor */
const Tensor &Loop::GetReduceApiTensor(const TPipe &tpipe, bool is_input) const {
  for (auto it = this->bodys.rbegin(); it != this->bodys.rend(); ++it) {
    if (it->type != LoopType::CALL) {
      continue;
    }
    if (is_input) {
      auto in_tensor_ptr = tpipe.GetTensor(it->call->inputs[0]->id);
      return *in_tensor_ptr;
    }
    auto out_tensor_ptr = tpipe.GetTensor(it->call->outputs[0].id);
    return *out_tensor_ptr;
  }
  throw std::runtime_error("No valid tensor found.");
}

static void CreateInnerLoopSizeAndActualSize(const TPipe &tpipe, const Tiler &tiler, const Axis &axis, std::stringstream &ss) {
  if (axis.from.size() == 1) {
    ss << tiler.GenInnerLoopSizeAndActualSize(axis.split_pair_other_id, axis.id, false);
    return;
  }

  ascir::AxisId inner_id = -1;
  ascir::AxisId outer_id = -1;
  for (size_t i = 0; i < axis.from.size(); i++) {
    auto current_axis = tiler.GetAxis(axis.from[i]);
    if (IsOuter(current_axis) &&
        tiler.GetAxis(current_axis.split_pair_other_id).type == Axis::Type::kAxisTypeBlockInner) {
      outer_id = current_axis.id;
      inner_id = current_axis.split_pair_other_id;
      break;
    }
  }
  ss << tiler.GenInnerLoopSizeAndActualSize(inner_id, outer_id, false);
  std::set<ascir::AxisId> vectorized_axis;
  for (const auto &tensor : tpipe.tensors) {
    for (auto axis_id : tensor.second.vectorized_axis) {
      int32_t count = 0;
      for (auto &from : axis.from) {
        if (tiler.HasSameOriginAxis(axis_id, from)) {
          count++;
        }
      }
      if (vectorized_axis.find(axis_id) == vectorized_axis.end() &&
          (count != 0 && !tiler.HasSameOriginAxis(axis_id, inner_id))) {
        ss << tpipe.tiler.GenInnerLoopSizeAndActualSize(axis_id, axis.id, false);
        vectorized_axis.insert(axis_id);
      }
    }
  }
}

Status Loop::GenerateLoop(const Tiler &tiler, const TPipe &tpipe, std::vector<ascir::AxisId> &current_axis,
                          std::stringstream &ss) {
  if (this->axis_id == ge::kIdNone) {
    GE_CHK_STATUS_RET(this->GenerateBody(tiler, tpipe, current_axis, ss),
                      "Codegen generate body failed when axis id is none");
    return ge::SUCCESS;
  }

  const auto &axis = tiler.GetAxis(this->axis_id);
  current_axis.push_back(this->axis_id);
  if (axis.type == Axis::Type::kAxisTypeBlockOuter) {
    CreateInnerLoopSizeAndActualSize(tpipe, tiler, axis, ss);
    GE_CHK_STATUS_RET(this->GenerateBody(tiler, tpipe, current_axis, ss),
                      "Codegen generate body failed when axis type is block outer");
  } else {
    std::string reduce_dim_a = "reduce_dim_a";
    if (GetReduceType() == "Mean") {
      ss << "uint32_t " << reduce_dim_a << ";" << std::endl;
    }
    if (axis.type != Axis::Type::kAxisTypeBlockInner && this->is_graph_has_reduce_node) {
      ss << "bool control_dis_enable_cache_a = true;" << std::endl;
      ss << "if ( " << axis.loop_size.Str() << " == 1) {" << std::endl;
      ss << "control_dis_enable_cache_a = false;" << std::endl;
      ss << "}" << std::endl;
    }
    if (axis.type == Axis::Type::kAxisTypeBlockInner) {
      auto peer = tiler.GetAxis(axis.split_pair_other_id);
      ss << "int32_t block_dim_offset = " << peer.Str() << " * " << tiler.Size(axis.size) << ";" << std::endl;
    }
    if (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse && axis.type == Axis::Type::kAxisTypeTileOuter) {
      ss << axis.loop_size.AsArg() << " = 1;" << std::endl;
    }
    ss << "for (" << axis.AsArg() << " = 0; " << axis << " < " << axis.loop_size.Str() << "; " << axis << "++) "
       << "{" << std::endl;
    if (tpipe.cv_fusion_type != ascir::CubeTemplateType::kUBFuse) {
      ss << tiler.CalcFromAxis(axis.id);
    }
    GenerateEnCacheCondition(tiler, tpipe, axis, ss);
    if (tpipe.cv_fusion_type != ascir::CubeTemplateType::kUBFuse) {
      std::set<ascir::AxisId> vectorized_axis;
      for (const auto &tensor : tpipe.tensors) {
        for (auto axis_id : tensor.second.vectorized_axis) {
          if (vectorized_axis.find(axis_id) == vectorized_axis.end()) {
            ss << tpipe.tiler.GenInnerLoopSizeAndActualSize(axis_id, this->axis_id, false);
            vectorized_axis.insert(axis_id);
          }
        }
      }
    } else if (axis.type == Axis::Type::kAxisTypeTileOuter) {
      const auto &tile_inner = tiler.GetAxis(axis.split_pair_other_id);
      ge::Expression actual_size = ge::Symbol(tile_inner.actual_size.name.c_str());
      tpipe.tiler.actual_sizes.emplace_back(std::make_pair(tile_inner.size_expr, actual_size));
      ss << tile_inner.actual_size.AsArg() << " = stageSize;" << std::endl; // 多轮循环不能使用curAivM * curAivN，否则奇数尾块计算有精度问题
      auto ub_tensor = tpipe.GetTensor(tpipe.cube_output_tensor_id);
      GE_CHK_BOOL_RET_STATUS(ub_tensor != nullptr, ge::FAILED, "Codegen CV Fusion MatmulOutput UB tensor id[%ld] "
                             "not found", tpipe.cube_output_tensor_id);
      ss << ub_tensor->Str() << "_actual_size = " << tile_inner.actual_size.Str() << ";" << std::endl;
    }
    GE_CHK_STATUS_RET(this->GenerateBody(tiler, tpipe, current_axis, ss),
                      "Codegen generate body failed for normal loop");
    ss << "}" << std::endl;
    if (IsReduceDoubleTile(tiler, tpipe, this->is_graph_has_reduce_node) && GetReduceType() == "Mean") {
      auto reduce_dst_tensor = GetReduceApiTensor(tpipe, false);
      std::string dtype_name;
      Tensor::DtypeName(reduce_dst_tensor.dtype, dtype_name);
      std::set<ascir::AxisId> r_from_axis;
      for (size_t i = 0; i < reduce_dst_tensor.axis_strides.size(); i++) {
        if (reduce_dst_tensor.axis_strides[i] == 0) {  // 如果目标张量的轴步长为0
          auto axis_id = reduce_dst_tensor.axis[i];    // 获取当前轴ID
          // 定义递归函数用于收集原始轴
          std::function<void(int32_t)> collect_original_axes = [&tiler, &r_from_axis, &collect_original_axes](int32_t current_axis_id) {
            auto axis = tiler.GetAxis(current_axis_id);  // 获取当前轴对象
            if (axis.type == ascir::Axis::Type::kAxisTypeOriginal) {
              r_from_axis.insert(current_axis_id);  // 如果是原始轴则加入集合
            } else {
              // 否则递归处理所有来源轴
              for (auto from_axis_id : axis.from) {
                collect_original_axes(from_axis_id);
              }
            }
          };
          collect_original_axes(axis_id);  // 从当前轴开始递归收集
        }
      }
      ss << "const float dimr_recip = 1.0f / (";
      uint32_t count = 0;
      for (auto axis_id : r_from_axis) {
        if (count == 0) {
          ss << tiler.AxisSize(axis_id);
          count++;
        } else {
          ss << " * " << tiler.AxisSize(axis_id);
        }
      }
      ss << ");" << std::endl;
      ss << "Muls(" << reduce_dst_tensor << ", " << reduce_dst_tensor << ", " << "dimr_recip, "
         << KernelUtils::SizeAlign() << "(" << reduce_dim_a << ", 32 / sizeof(" << dtype_name << ")));" << std::endl;
    }
  }
  current_axis.pop_back();
  return ge::SUCCESS;
}

static const Axis &GetTileOutAxis(const Tiler &tiler, const Axis &axis) {
  for (auto &[id, cur_axis] : tiler.axis_map) {
    (void)id;
    if (cur_axis.type == ascir::Axis::Type::kAxisTypeTileOuter) {
      return cur_axis;
    }
  }
  return axis;
}

static const Axis &GetTileOutAxisAnother(const Tiler &tiler, const Axis &axis) {
  int32_t count = 0;
  for (auto &[id, cur_axis] : tiler.axis_map) {
    (void)id;
    if (cur_axis.type == ascir::Axis::Type::kAxisTypeTileOuter) {
      count++;
    }

    if (count > 1) {
      return cur_axis;
    }
  }
  return axis;
}

void Loop::GenerateEnCacheCondition(const Tiler &tiler, const TPipe &tpipe, const Axis &axis,
                                    std::stringstream &ss) const {
  (void)tpipe;
  if (!axis.is_split_b) {
    return;
  }
  ge::Expression block_in_size = Zero;
  for (auto &[id, cur_axis] : tiler.axis_map) {
    (void)id;
    if (cur_axis.type == ascir::Axis::Type::kAxisTypeBlockInner) {
      block_in_size = ge::Symbol(cur_axis.axis_size.name.c_str());
      break;
    }
  }
  Axis tile_out_axis = GetTileOutAxis(tiler, axis);
  bool is_double_tile = IsReduceDoubleTile(tiler, tpipe, this->is_graph_has_reduce_node);
  bool is_cache_valid = this->IsBodyContainLoop() || axis.type == ascir::Axis::Type::kAxisTypeTileOuter ||
                        axis.type == ascir::Axis::Type::kAxisTypeMerged;
  if (is_double_tile && is_cache_valid) {
    Axis thile_out_axis_another = GetTileOutAxisAnother(tiler, axis);
    for (auto body : this->bodys) {
      if (body.type == LoopType::LOOP) {
        ss << "bool " << kEnCacheA << " = (" << axis << " < 1) || ((" << tiler.block_dim << " * "
           << block_in_size.Str().get() << " + " << axis << ") % " << tile_out_axis.loop_size << " < 1);" << std::endl;
        break;
      }

      if (body.call->unit == ge::ComputeUnit::kUnitNone || !body.call->enable_cache) {
        continue;
      }

      ss << "bool " << kEnCacheR << " = (" << axis << " < 1) || ((" << axis << ") % "
         << thile_out_axis_another.loop_size << " < 1);" << std::endl;
      break;
    }
  } else {
    bool need_create_fused_cond = true;
    bool need_create_origin_cond = true;
    for (auto body : this->bodys) {
      if (body.type != LoopType::CALL || body.call->unit == ge::ComputeUnit::kUnitNone || !body.call->enable_cache) {
        continue;
      }
      if (body.call->exec_condition == ge::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis &&
          need_create_fused_cond) {
        ss << "bool " << kEnCacheFusedBroadcastAxis << " = (" << axis << " < 1) || ((" << tiler.block_dim << " * "
           << block_in_size.Str().get() << " + " << axis << ") % " << tile_out_axis.loop_size << " < 1);" << std::endl;
        need_create_fused_cond = false;
        continue;
      }
      if (body.call->exec_condition == ge::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis &&
          need_create_origin_cond) {
        ss << "bool " << kEnCacheOriginBroadcastAxis << " = (" << axis << " < 1);" << std::endl;
        need_create_origin_cond = false;
        continue;
      }
    }
  }
}

Status Loop::Generate(const Tiler &tiler, const TPipe &tpipe, std::string &result, ComputeStage stage) {
  std::vector<ascir::AxisId> current_axis;
  this->compute_stage = stage;
  stringstream ss;
  GE_CHK_STATUS_RET(this->GenerateLoop(tiler, tpipe, current_axis, ss), "Generate loop failed");
  result = ss.str();
  return ge::SUCCESS;
}

Kernel::Kernel(const std::string &kernel_name)
    : workspace_arg("workspace"),
      name(kernel_name),
      tiler(kernel_name + "TilingData", "t"),
      tpipe("tpipe", this->tiler),
      root_loop(ge::kIdNone) {}

Kernel::~Kernel() {
  root_loop.Destruct();
}

std::string Kernel::TilingKeyFuncDeclare(const std::string &impl_graph_name, const std::string &tiling_data) const {
  const char *flags[] = {"inline", "__aicore__"};
  const char *return_type = "void";

  std::stringstream ss;
  for (auto flag : flags) {
    ss << flag << " ";
  }
  ss << return_type << " ";
  ss << CamelToLowerSneak(impl_graph_name) << "(";
  if (use_list_tensor_) {
    ss << "ListTensorDesc &input_tensor_desc, ListTensorDesc &output_tensor_desc, ";
  } else {
    for (auto &input : this->inputs) {
      ss << input.AsArg() << ", ";
    }
    for (auto &output : this->outputs) {
      ss << output.AsArg() << ", ";
    }
  }
  ss << this->workspace_arg.AsArg() << ", ";
  for (auto &workspace : this->workspaces) {
    ss << workspace.AsArg() << ", ";
  }
  ss << "const "<< tiling_data << " *t";
  ss << ")";
  return ss.str();
}

Status Kernel::GlobalTensorInit(std::string &result) const {
  std::stringstream ss;
  for (std::size_t i = 0; i < this->inputs.size(); i++) {
    const auto &tensor = this->tpipe.tensors.find(this->input_tensors[i]);
    if (tensor == this->tpipe.tensors.end()) {
      GELOGE(ge::FAILED, "Codegen input tensor id[%ld] not found", this->input_tensors[i]);
      return ge::FAILED;
    }

    ss << tensor->second.Define() << std::endl;
    std::string local_result;
    if (use_list_tensor_) {
      auto input_index = input_name_to_index_.at(this->inputs[i].Str());
      auto input = GM_ADDR("input_tensor_desc.GetDataPtr<__gm__ uint8_t>(" + std::to_string(input_index) + ")");
      GE_CHK_STATUS_RET(tensor->second.SetGlobalBuffer(input, "", local_result),
                        "Codegen set global buffer failed");
    } else {
      GE_CHK_STATUS_RET(tensor->second.SetGlobalBuffer(this->inputs[i], "", local_result),
                        "Codegen set global buffer failed");
    }
    ss << local_result << std::endl;
  }

  for (std::size_t i = 0; i < this->outputs.size(); i++) {
    const auto &tensor = this->tpipe.tensors.find(this->output_tensors[i]);
    if (tensor == this->tpipe.tensors.end()) {
      GELOGE(ge::FAILED, "Codegen output tensor id[%ld] not found", this->output_tensors[i]);
      return ge::FAILED;
    }

    ss << tensor->second.Define() << std::endl;
    std::string local_result;
    if (use_list_tensor_) {
      auto output_index = output_name_to_index_.at(this->outputs[i].Str());
      auto output = GM_ADDR("output_tensor_desc.GetDataPtr<__gm__ uint8_t>(" + std::to_string(output_index) + ")");
      GE_CHK_STATUS_RET(tensor->second.SetGlobalBuffer(output, "", local_result),
                        "Codegen set global buffer failed");
    } else {
      GE_CHK_STATUS_RET(tensor->second.SetGlobalBuffer(this->outputs[i], "", local_result),
                        "Codegen set global buffer failed");
    }
    ss << local_result << std::endl;
  }

  for (std::size_t i = 0; i < this->constant_tensors.size(); i++) {
    auto tensor = this->tpipe.tensors.find(this->constant_tensors[i]);
    if (tensor == this->tpipe.tensors.end()) {
      GELOGE(ge::FAILED, "Codegen concat tensor id[%ld] not found", this->constant_tensors[i]);
      return ge::FAILED;
    }
    GELOGI("const_value_expr: %s", tensor->second.const_value_expr.Str().get());

    string const_value = tensor->second.const_value_expr == 0 ? tensor->second.const_value
                                                              : tiler.Size(tensor->second.const_value_expr, true);
    ss << tensor->second.DefineConst(const_value.c_str()) << std::endl;
    GELOGI("Define ss value: %s", ss.str().c_str());
  }

  for (std::size_t i = 0; i < this->ub_scalar_tensors.size(); i++) {
    auto tensor = this->tpipe.tensors.find(this->ub_scalar_tensors[i]);
    GE_ASSERT_TRUE((tensor != this->tpipe.tensors.end()), "Codegen ub_scalar tensor id[%ld] not found",
                   this->ub_scalar_tensors[i]);

    std::string def_ub_scalar;
    GE_CHK_STATUS_RET(tensor->second.DefineUbScalar(def_ub_scalar));
    ss << def_ub_scalar;
    GELOGI("Define ub_scalar var:", def_ub_scalar.c_str());
  }

  std::stringstream offset_ss;
  offset_ss << "0";
  auto it_ws_tensors = this->workspace_tensors.begin();
  for (size_t i = 0UL; i < this->workspaces.size(); i++) {
    GELOGI("Define workspace tensor id: %ld", it_ws_tensors->first);
    auto tensor = this->tpipe.tensors.find(it_ws_tensors->first);
    if (tensor == this->tpipe.tensors.end()) {
      GELOGE(ge::FAILED, "Codegen workspace tensor id[%ld] not found", it_ws_tensors->first);
      return ge::FAILED;
    }

    ss << tensor->second.Define() << std::endl;
    std::string local_result;
    GE_CHK_STATUS_RET(tensor->second.SetGlobalBuffer(this->workspace_arg, offset_ss.str(), local_result),
                      "Codegen set global buffer failed");
    ss << local_result << std::endl;
    offset_ss << " + " << "(" << this->workspaces[i] << ")";
    it_ws_tensors++;
  }
  result = ss.str();
  return ge::SUCCESS;
}

Status Kernel::LocalTensorQueBufAlloc(std::string &result, const ascir::ImplGraph &graph) const {
  (void)graph;
  stringstream ss;
  std::string tmp;

  ss << this->tpipe.Define() << std::endl;

  if (!this->pre_api_extract_dup.empty()) {
    ss << this->tpipe.GenDuplicateBufAlloc(this->pre_api_extract_dup) << std::endl;
  }

  GE_CHK_STATUS_RET(this->tpipe.BlkTensorAllocAndInit(tmp), "Codegen BlkTensorAllocAndInit failed");
  ss << tmp << std::endl;

  GE_CHK_STATUS_RET(this->tpipe.LocalTensorQueBufAlloc(tmp), "Codegen alloc local tensor que buf failed");
  ss << tmp << std::endl;

  result = ss.str();
  return ge::SUCCESS;
}

Status Kernel::ParseWorkspaceTensor(const ascir::TensorAttr *tensor,
                                    const ascir::FusedScheduledResult &fused_schedule_result,
                                    std::set<int64_t> &output_indices,
                                    const std::unordered_map<ascir::TensorId, size_t> &output_tensorid_to_index,
                                    const std::map<size_t, std::string> output_index_to_name) {
  if (this->tpipe.tensors.find(tensor->attr.mem.tensor_id) == this->tpipe.tensors.cend()) {
    GE_CHK_STATUS_RET(this->tpipe.AddTensor(*tensor, "workspace"), "Codegen add tensor failed");
  }
  // workspace 复用 output节点场景需要将该节点添加到该kernel中 用于声明kernel函数入参及变量
  if (output_tensorid_to_index.find(tensor->attr.mem.tensor_id) != output_tensorid_to_index.cend()) {
    int64_t index;
    const auto &out_node = fused_schedule_result.output_nodes[output_tensorid_to_index.at(tensor->attr.mem.tensor_id)];
    GE_CHK_GRAPH_STATUS_RET(out_node->attr.ir_attr->GetAttrValue("index", index),
                            "Failed to get Workspace reuse Output index, node = %s", out_node->GetNamePtr());
    GE_ASSERT_TRUE(index >= 0, "invalid Workspace reuse Output index, node = %s, index = %ld", out_node->GetNamePtr(),
                   index);
    GE_ASSERT_TRUE(output_index_to_name.find(index) != output_index_to_name.cend(),
                   "Get workspace reuse output name failed.");
    const auto &output_name = output_index_to_name.at(index);
    GE_ASSERT_TRUE(!output_name.empty(), "Failed to get workspace reuse arg name, output_node = %s, index = %ld",
                   out_node->GetNamePtr(), index);
    if (output_indices.emplace(index).second) {
      this->outputs.emplace_back(GM_ADDR(GenValidName(output_name)));
      this->output_tensors.emplace_back(out_node->inputs[0].attr.mem.tensor_id);
    }
  }
  return ge::SUCCESS;
}

bool Kernel::ProcessRequiredInput(const ge::AscNodePtr &node, size_t index, size_t count,
                                  std::vector<ge::DataType> &input_dtypes) const {
  GE_ASSERT_EQ(count, 1U);
  GE_ASSERT_TRUE(static_cast<uint32_t>(index) < node->inputs.Size());
  const auto &tensor = node->inputs[index];
  input_dtypes.push_back(tensor.attr.dtype);
  return true;
}

bool Kernel::ProcessDynamicInput(const ge::AscNodePtr &node, size_t index, size_t count,
                                 std::vector<ge::DataType> &input_dtypes) const {
  std::set<ge::DataType> unique_dtypes;
  for (size_t i = index; i < index + count; ++i) {
    GE_ASSERT_TRUE(static_cast<uint32_t>(i) < node->inputs.Size());
    unique_dtypes.insert(node->inputs[i].attr.dtype);
  }
  GE_ASSERT_TRUE(unique_dtypes.size() == 1U, "%s dynamic_input should have uniform dtypes", node->GetOpDesc()->GetNamePtr());
  input_dtypes.push_back(*unique_dtypes.begin());
  return true;
}

bool Kernel::CollectInputDtypesForOutput(const ascir::NodeView &node, std::vector<ge::DataType> &input_dtypes) const {
  std::set<ge::DataType> unique_dtypes;
  for (const auto input : node->inputs()) {
    unique_dtypes.insert(input->attr.dtype);
  }
  GE_ASSERT_TRUE(unique_dtypes.size() == 1U, "%s %s should have uniform dtypes", node->GetNamePtr(), node->GetTypePtr());
  input_dtypes.push_back(*unique_dtypes.begin());
  return true;
}

bool Kernel::CollectInputDtypesForWorkspace(const ascir::NodeView &node, std::vector<ge::DataType> &input_dtypes) const {
  std::set<ge::DataType> unique_dtypes;
  if (node->inputs().size() != 0) {
    for (const auto input : node->inputs()) {
      unique_dtypes.insert(input->attr.dtype);
    }
  } else {
    unique_dtypes.insert(node->outputs()[0]->attr.dtype);
  }

  GE_ASSERT_TRUE(unique_dtypes.size() == 1U, "%s %s should have uniform dtypes", node->GetNamePtr(), node->GetTypePtr());
  input_dtypes.push_back(*unique_dtypes.begin());
  return true;
}

bool Kernel::CollectInputDtypes(const ascir::NodeView &node, std::vector<ge::DataType> &input_dtypes) const {
  if (node->GetType() == ge::ascir_op::Output::Type) {
    // Output因为前面做了一个可变ir的操作，即ir是必选输入，但是实际行为支持是动态输入或者必选两种，因此特殊处理一下
    return CollectInputDtypesForOutput(node, input_dtypes);
  }
  if (node->GetType() == ge::ascir_op::Workspace::Type) {
    // Workspace连接两张子图时，后一张子图的输入是没有显示指定的，因此输入数据的类型按照输出数据类型特殊处理一下
    return CollectInputDtypesForWorkspace(node, input_dtypes);
  }
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc, "op_desc is nullptr!");

  const auto &ir_inputs = op_desc->GetIrInputs();
  std::map<size_t, std::pair<size_t, size_t>> ir_input_2_range;
  GE_ASSERT_GRAPH_SUCCESS(ge::OpDescUtils::GetIrInputRawDescRange(op_desc, ir_input_2_range),
                          "op %s %s has invalid ir desc", op_desc->GetNamePtr(), op_desc->GetTypePtr());

  size_t index = 0;
  for (size_t ir_input_index = 0; ir_input_index < ir_inputs.size(); ++ir_input_index) {
    const auto &range_iter = ir_input_2_range.find(ir_input_index);
    GE_ASSERT_TRUE(range_iter != ir_input_2_range.end(), "Invalid ir_input_index: %zu", ir_input_index);

    const auto &start_and_count = range_iter->second;
    const auto count = start_and_count.second;
    const auto &ir_input_type = ir_inputs[ir_input_index].second;

    switch (ir_input_type) {
      case ge::IrInputType::kIrInputRequired:
        GE_ASSERT_TRUE(ProcessRequiredInput(node, index, count, input_dtypes), "ProcessRequiredInput failed, node = %s",
                       node->GetNamePtr());
        break;
      case ge::IrInputType::kIrInputDynamic:
        GE_ASSERT_TRUE(ProcessDynamicInput(node, index, count, input_dtypes), "ProcessDynamicInput failed, node = %s",
                       node->GetNamePtr());
        break;
      default:
        GELOGE(ge::FAILED, "%s %s unsupported input type %ld at ir index %zu", op_desc->GetNamePtr(),
               op_desc->GetTypePtr(), static_cast<int64_t>(ir_input_type), ir_input_index);
        return false;
    }
    index += count;
  }
  return true;
}

bool Kernel::CollectOutputDtypes(const ascir::NodeView &node, std::vector<ge::DataType> &output_dtypes) const {
  // 由于目前schedule在某些场景下会丢失Output节点输出tensor的数据类型，这里暂时按照输入tensor的数据类型收集，schedule解决后删除.
  if (node->GetType() == ge::ascir_op::Output::Type) {
    output_dtypes.emplace_back(node->inputs()[0]->attr.dtype);
    return true;
  }
  std::set<ge::DataType> unique_dtypes;
  for (auto output : node->outputs()) {
    if (output->attr.dtype == ge::DT_UNDEFINED) {
      return true;
    }
    unique_dtypes.insert(output->attr.dtype);
  }
  GE_ASSERT_TRUE(unique_dtypes.size() == 1U, "%s dynamic_input should have uniform dtypes", node->GetOpDesc()->GetNamePtr());
  output_dtypes.push_back(*unique_dtypes.begin());
  return true;
}

Status Kernel::IsDataTypeSupported(const ascir::ImplGraph &graph) const {
  std::set<string> ignore_node_type = {"Ge", "Eq", "Ne", "Gt", "Le", "Broadcast", "Nop", "Sign", "LogicalNot",
                                       "LogicalOr", "LogicalAnd", "Concat", "Select", "Where", "Ub2ub", "BitwiseAnd", "Split"};
  for (const auto &node : graph.GetAllNodes()) {
    // 对于动态输入和动态输出的节点，不进行类型检测
    const auto &ir_inputs = node->GetOpDescBarePtr()->GetIrInputs();
    const auto &ir_outputs = node->GetOpDescBarePtr()->GetIrOutputs();
    if (ir_inputs.size() != 0 && ir_inputs[0].second == ge::IrInputType::kIrInputDynamic && ir_outputs.size() != 0 &&
        ir_outputs[0].second == ge::IrOutputType::kIrOutputDynamic) {
      continue;
    }
    std::vector<ge::DataType> input_dtypes;
    std::vector<ge::DataType> output_dtypes;
    GE_ASSERT_TRUE(CollectInputDtypes(node, input_dtypes), "Collect input dtypes failed, node = %s",
                   node->GetNamePtr());
    GE_ASSERT_TRUE(CollectOutputDtypes(node, output_dtypes), "Collect output dtypes failed, node = %s",
                   node->GetNamePtr());
    // 一些api暂不支持int64输入，但是有一些存量st，因此临时屏蔽这些api的int64类型检测，ascir支持后放开.
    if ((ignore_node_type.find(node->GetType()) != ignore_node_type.end() &&
         std::find(input_dtypes.begin(), input_dtypes.end(), ge::DT_INT64) != input_dtypes.end())) {
      continue;
    }
    std::string npu_arch;
    GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetCurrentPlatformString(npu_arch));
    if (ge::ascir::CommonInferDtype(node->GetType(), input_dtypes, output_dtypes, npu_arch) != ge::SUCCESS) {
      GELOGE(ge::FAILED, "ASCIR(%s) not support dtypes(input dtype:%s, output dtype:%s), node:%s", node->GetTypePtr(),
             VectorToStr(input_dtypes).c_str(), VectorToStr(output_dtypes).c_str(), node->GetNamePtr());
      return ge::FAILED;
    }
  }
  return ge::SUCCESS;
}

Status Kernel::ParseGraph(const ascir::ImplGraph &graph, const ascir::FusedScheduledResult &fused_schedule_result,
                          Kernel &kernel) {
  // Parse kernel input output
  std::map<size_t, std::string> input_index_to_name;
  std::map<size_t, std::string> output_index_to_name;
  std::unordered_map<ascir::TensorId, size_t> output_tensorid_to_index;
  if (kernel.IsDataTypeSupported(graph)) {
    return ge::FAILED;
  }
  for (size_t i = 0U; i < fused_schedule_result.input_nodes.size(); ++i) {
    const auto &input = fused_schedule_result.input_nodes[i];
    GE_ASSERT_TRUE(IsOps<Data>(input), "Codegen unsupported input[%s] type[%s]", input->GetName().c_str(),
                   input->GetType().c_str());
    const auto &normalized_name = GenValidName(input->GetName());
    kernel.input_name_to_index_[normalized_name] = i;
    input_index_to_name[i] = normalized_name;
    GELOGD("input_index = %zu, input_name = %s", i, normalized_name.c_str());
  }
  for (size_t i = 0U; i < fused_schedule_result.output_nodes.size(); ++i) {
    const auto &output = fused_schedule_result.output_nodes[i];
    const auto &normalized_name = GenValidName(output->GetName());
    kernel.output_name_to_index_[normalized_name] = i;
    output_index_to_name[i] = output->GetName();
    output_tensorid_to_index[output->inputs[0].attr.mem.tensor_id] = i;
    GELOGD("output_index = %zu, output_name = %s", i, normalized_name.c_str());
  }
  std::set<int64_t> input_indices;
  std::set<int64_t> output_indices;
  std::map<int64_t, std::pair<std::string, ascir::TensorId>> kernel_inputs;
  std::map<int64_t, std::pair<std::string, ascir::TensorId>> kernel_outputs;
  bool has_gather = false;
  for (const auto &node : graph.GetAllNodes()) {
    if (IsOps<Data>(node)) {
      int64_t index;
      GE_CHK_GRAPH_STATUS_RET(node->attr.ir_attr->GetAttrValue("index", index), "Failed to get Data index, node = %s",
                              node->GetNamePtr());
      GE_ASSERT_TRUE(index >= 0, "invalid Data index, node = %s, index = %ld", node->GetNamePtr(), index);
      const auto &input_name = input_index_to_name[index];
      GE_ASSERT_TRUE(!input_name.empty(), "Failed to get arg name, input_node = %s, index = %ld", node->GetNamePtr(),
                     index);
      if (input_indices.emplace(index).second) {
        kernel_inputs[index] = std::make_pair(input_name, node->outputs[0].attr.mem.tensor_id);
      }
      continue;
    }
    if (IsOps<Output>(node)) {
      int64_t index;
      GE_CHK_GRAPH_STATUS_RET(node->attr.ir_attr->GetAttrValue("index", index), "Failed to get Data index, node = %s",
                              node->GetNamePtr());
      GE_ASSERT_TRUE(index >= 0, "invalid Data index, node = %s, index = %ld", node->GetNamePtr(), index);
      const auto &output_name = output_index_to_name[index];
      GE_ASSERT_TRUE(!output_name.empty(), "Failed to get arg name, output_node = %s, index = %ld", node->GetNamePtr(),
                     index);
      if (output_indices.emplace(index).second) {
        kernel_outputs[index] = std::make_pair(output_name, node->inputs[0].attr.mem.tensor_id);
      }
      continue;
    }
    has_gather = (has_gather || IsOps<Gather>(node));
  }
  for (const auto &pair : kernel_inputs) {
    kernel.inputs.emplace_back(GM_ADDR(GenValidName(pair.second.first)));
    kernel.input_tensors.emplace_back(pair.second.second);
  }
  for (const auto &pair : kernel_outputs) {
    kernel.outputs.emplace_back(GM_ADDR(GenValidName(pair.second.first)));
    kernel.output_tensors.emplace_back(pair.second.second);
  }

  std::vector<ascir::TensorId> workspace_tensor_id = GetWorkspaceTensorIdListInOneScheduleResult(fused_schedule_result);
  for (auto tId : workspace_tensor_id) {
    std::string workspaceStr = "workspace";
    workspaceStr = workspaceStr + std::to_string(tId);
    kernel.workspaces.emplace_back(Uint32(workspaceStr.c_str()));
    kernel.workspace_tensors[tId] = "0";
  }

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Scalar>(node) || IsOps<IndexExpr>(node)) {
      kernel.constant_tensors.emplace_back(node->outputs[0].attr.mem.tensor_id);
    }
  }

  // Parse for tiler
  for (auto axis : graph.GetAllAxis()) {
    GE_CHK_STATUS_RET(kernel.tiler.AddAxis(*axis), "Codegen add axis failed");
  }
  kernel.tiler.AddAxisSplitBAttr();

  for (auto size : graph.GetAllSizeVar()) {
    kernel.tiler.AddSizeVar(*size);
  }

  GE_ASSERT_SUCCESS(kernel.tpipe.CollectQues(graph));
  // Parse for tpipe
  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Output>(node) || IsOps<Data>(node)) {
      continue;
    }

    auto desc = node->GetOpDesc();
    for (auto output : node->outputs()) {
      auto output_index = ge::ascir::AscTensorUtils::Index(*output);
      auto tensor_name = node->GetName() + "_" + desc->GetOutputNameByIndex(output_index);
      if (IsOps<Scalar>(node)) {
        std::string const_value;
        auto ir_attr = node->attr.ir_attr.get();
        if (ir_attr->GetAttrValue("value", const_value) != ge::GRAPH_SUCCESS) {
          GELOGE(ge::FAILED, "GetAttrValue const value faild");
          return ge::FAILED;
        }
        GELOGI("const value %s", const_value.c_str());
        // 不要将const_value放在参数第二个位置，会导致overload出现歧义
        GE_CHK_STATUS_RET(kernel.tpipe.AddTensor(const_value, *output, tensor_name), "Codegen add tensor failed");
        GE_CHK_STATUS_RET(kernel.ParseOptimizeInfo(node, *output));
      } else if (IsOps<IndexExpr>(node)) {
        int64_t size_id = 0;
        auto ir_attr = node->attr.ir_attr.get();
        if (ir_attr->GetAttrValue("expr", size_id) != ge::GRAPH_SUCCESS) {
          GELOGE(ge::FAILED, "GetAttrValue index expr faild, size_id = %lld", size_id);
          return ge::FAILED;
        }
        GELOGI("size_id = %lld", size_id);
        // todo index在Expression上是指什么？  暂时按照AddSizeVar顺序
        auto all_sizevar = graph.GetAllSizeVar();
        GE_CHK_STATUS_RET(kernel.tpipe.AddTensor(*output, all_sizevar.at(size_id)->expr, tensor_name),
                          "Codegen add tensor failed");
      } else if (IsOps<Workspace>(node)) {
        GE_CHK_STATUS_RET(kernel.ParseWorkspaceTensor(output, fused_schedule_result, output_indices,
                                                      output_tensorid_to_index, output_index_to_name),
                          "Codegen parse workspace tensor failed");
        kernel.has_workspace_node = true;
      } else if (IsOps<Store>(node)) {
        // 1. 多个Store节点写同一个Output的不同offset场景
        // 2. 多个schedule group之间通过workspace承接输出
        if (kernel.tpipe.tensors.find(output->attr.mem.tensor_id) == kernel.tpipe.tensors.cend()) {
          GE_CHK_STATUS_RET(kernel.tpipe.AddTensor(*output, tensor_name), "Codegen add tensor failed");
        }
      } else {
        GE_CHK_STATUS_RET(kernel.tpipe.AddTensor(*output, tensor_name), "Codegen add tensor failed");
        GE_CHK_STATUS_RET(kernel.ParseOptimizeInfo(node, *output));
      }
    }
  }

  for (auto node : fused_schedule_result.input_nodes) {
    auto desc = node->GetOpDesc();
    for (auto output : node->outputs()) {
      auto tensor_name = node->GetName() + "_" + desc->GetOutputNameByIndex(ge::ascir::AscTensorUtils::Index(*output));
      kernel.tpipe.AddTensor(*output, tensor_name);
    }
  }

  for (auto node : fused_schedule_result.workspace_nodes) {
    GE_ASSERT_TRUE(IsOps<Workspace>(node), "fused_schedule_result node[%s] is not workspace", node->GetName().c_str());
    auto &output = node->outputs[0U];
    // workspace作为schedule group之间的输入时需要在此处AddTensor
    // 作为schedule group之间的输出时映射为store的输出，在Store节点被AddTensor
    if (kernel.tpipe.tensors.find(output.attr.mem.tensor_id) == kernel.tpipe.tensors.cend()) {
      GELOGD("Workspace node[%s] is input in a schedule group, tensor id[%ld]", node->GetName().c_str(),
             output.attr.mem.tensor_id);
      GE_CHK_STATUS_RET(kernel.tpipe.AddTensor(output, "workspace"), "Codegen add tensor failed");
    }
  }

  for (auto node : graph.GetAllNodes()) {
    for (auto tmp_buffer : node->attr.tmp_buffers) {
      if (tmp_buffer.id == -1) {
        continue;
      }
      auto it = kernel.tpipe.bufs.find(tmp_buffer.id);
      GELOGD("reuse tmp buffer id is %ld.", tmp_buffer.id);
      if (it == kernel.tpipe.bufs.end()) {
        std::string position = "TPosition::VECCALC";
        ascir::Position tensor_position = ge::Position::kPositionVecCalc;
        auto [new_buf, is_insert5] = kernel.tpipe.bufs.emplace(tmp_buffer.id, TBuf{tmp_buffer.id, tensor_position, position});
        GE_CHK_BOOL_RET_STATUS(is_insert5, ge::FAILED, "Codegen emplace tbuf [%ld] failed", tmp_buffer.id);
        new_buf->second.tmp_buf_size_list.emplace_back(tmp_buffer.buf_desc.size);
        new_buf->second.tmp_buf_reuse = true;
      } else {
        it->second.tmp_buf_size_list.emplace_back(tmp_buffer.buf_desc.size);
        it->second.tmp_buf_reuse = true;
      }
    }
  }
  uint32_t total_blk_num = 0U;
  GetApiExtractDupSet(graph, kernel.pre_api_extract_dup, total_blk_num);
  kernel.SetEnableParallelCompile((!has_gather));
  if (IsCVFusionUBGraph(graph, kernel.tpipe.cv_fusion_type)) {
    GE_CHK_STATUS_RET(kernel.tpipe.GetCVFusionCubeOutputUBTensorIdAndQueId(graph),
                      "get cube output tensor id failed");
  }
  // Parse for loop
  return kernel.root_loop.ConstructFromNodes(graph.GetAllNodes(), kernel.tiler, kernel.tpipe);
}

Status Kernel::GenerateSubGraphFuncDef(const Loop *loop, std::stringstream &ss) const {
  // 用栈存储待处理的Loop节点，模拟递归调用栈
  GE_ASSERT_NOTNULL(loop);
  std::stack<const Loop *> loop_stack;
  loop_stack.push(loop);

  // 循环处理栈中所有节点，直到栈为空
  while (!loop_stack.empty()) {
    // 弹出栈顶节点（当前要处理的节点）
    const Loop *current_loop = loop_stack.top();
    loop_stack.pop();

    // 遍历当前节点的所有body
    for (auto &body : current_loop->bodys) {
      if (body.type == LoopType::LOOP) {
        GE_ASSERT_NOTNULL(body.loop);
        loop_stack.push(body.loop);
      } else if (body.type == LoopType::CALL) {
        GE_ASSERT_NOTNULL(body.call);
        GE_ASSERT_SUCCESS(body.call->GenerateFuncDefinition(tpipe, tiler, ss), "gen func definition failed, api_name:%s",
                          body.call->api_name_.c_str());
      }
    }
  }

  return ge::SUCCESS;
}

Status Kernel::Generate(const std::string &impl_graph_name, const std::string &tiling_data, std::string &result,
                        const ascir::ImplGraph &graph) {
  if (ascgen_utils::IsCubeType(graph)) {
    return ge::SUCCESS;
  }
  stringstream ss;

  GE_ASSERT_SUCCESS(GenerateSubGraphFuncDef(&(this->root_loop), ss));

  ss << this->TilingKeyFuncDeclare(impl_graph_name, tiling_data) << " {" << std::endl;

  std::string tmp;
  for (auto &[id, axis] : tiler.axis_map) {
    if (IsInner(axis) || IsMergeFromInner(tiler, axis)) {
      continue;
    }

    if (IsOuter(axis) && !(axis.type == ascir::Axis::Type::kAxisTypeBlockOuter && axis.from.size() > 1)) {
      ss << tiler.GenAxisSizeNew(axis.split_pair_other_id);
    }
    ss << tiler.GenAxisSizeNew(id);
  }

  ss << this->tiler.BlockOutterAxisDefine();
  ss << std::endl;
  GE_CHK_STATUS_RET(this->GlobalTensorInit(tmp), "Codegen global tensor init failed");
  ss << tmp;
  ss << std::endl;
  GE_CHK_STATUS_RET(this->LocalTensorQueBufAlloc(tmp, graph), "Codegen alloc local tensor que buf failed");
  ss << tmp;

  GE_CHK_STATUS_RET(this->root_loop.Generate(this->tiler, this->tpipe, tmp), "Codegen root loop Generate failed");
  ss << tmp;

  ss << "}" << std::endl;

  result = ss.str();
  return ge::SUCCESS;
}

std::string Kernel::GetIncludeApiHeaderFiles(const ascir::FusedScheduledResult &fused_schedule_result) {
  std::set<std::string> api_header_list = {
    "basic_api/kernel_tpipe.h",
    "basic_api/kernel_tensor.h",
    "basic_api/kernel_type.h",
    "basic_api/kernel_operator_block_sync_intf.h",
    "basic_api/kernel_operator_data_copy_intf.h",
    "basic_api/kernel_common.h",
    "basic_api/kernel_operator_common_intf.h",
    "basic_api/kernel_operator_sys_var_intf.h",
    "basic_api/kernel_struct_binary.h",
  };
  std::stringstream ss;
  for (const auto &header : api_header_list) {
    ss << "#include \"" << header << "\"" << std::endl;
  }
  for (size_t graph_id = 0; graph_id < fused_schedule_result.node_idx_to_scheduled_results.size(); graph_id++) {
    auto scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
    for (size_t i = 0; i < scheduled_results.size(); i++) {
      auto schedule_groups = scheduled_results[i].schedule_groups;
      for (size_t j = 0; j < schedule_groups.size(); j++) {
        auto schedule_graphs = schedule_groups[j].impl_graphs;
        for (size_t k = 0; k < schedule_graphs.size(); k++) {
          for (const auto &node : schedule_graphs[k].GetAllNodes()) {
            auto impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
            GE_ASSERT_NOTNULL(impl, "GetAscIrCodegenImpl of node %s[%s] is null", node->GetTypePtr(),
                              node->GetNamePtr());
            for (const auto &header_str : impl->IncludeApiHeaderFiles()) {
              if (api_header_list.count(header_str) == 0) {
                api_header_list.insert(header_str);
                ss << "#include \"" << header_str << "\"" << std::endl;
              }
            }
          }
        }
      }
    }
  }
  return ss.str();
}

std::string Kernel::IncludeAndDefines(const ascir::FusedScheduledResult &fused_schedule_result,
                                      const std::string &kernel_task_type, bool use_tensor_desc, bool is_inductor) {
  std::stringstream ss;

  ss << Kernel::GetIncludeApiHeaderFiles(fused_schedule_result);
  if (use_tensor_desc) {
    ss << "#include \"kernel_operator_list_tensor_intf.h\"" << std::endl;
  }
  ss << "#include \"autofuse_tiling_data.h\"" << std::endl;
  ss << std::endl;
  ss << "using namespace AscendC;" << std::endl;
  ss << std::endl;
  if (!is_inductor) {
    ss << "KERNEL_TASK_TYPE_DEFAULT(" << kernel_task_type << ");" << std::endl;
  }
  ss << std::endl;

  const static string kAscendcUtilsExtend = {
#include "utils_str.h"
  };

  const static string kAscendcBrcInline = {
    #include "brc_inline_api_str.h"
  };

  ss << kAscendcUtilsExtend << kAscendcBrcInline << std::endl;

  return ss.str();
}

std::string Kernel::KernelFuncDeclare(const std::string &graph_name,
                                      const ascir::FusedScheduledResult &fused_schedule_result, bool use_list_tensor,
                                      bool is_inductor) {
  std::stringstream ss;
  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
    ss << "template <int8_t API_LEVEL, int8_t A_TRANS, int8_t B_TRANS, int8_t BATCH_MODEL, int8_t MODEL, int8_t "
          "FULL_LOAD, int8_t L0C2OUT_MODEL>" << std::endl;
    const char *flags[] = {"__global__", "__aicore__"};
    for (auto flag : flags) {
      ss << flag << " ";
    }
  } else {
    const char *flags[] = {"extern \"C\"", "__global__", "__aicore__"};
    for (auto flag : flags) {
      ss << flag << " ";
    }
  }
  const char *return_type = "void";
  ss << return_type << " ";
  ss << CamelToLowerSneak(graph_name) << "(";

  if (use_list_tensor) {
    ss << GM_ADDR("inputs").AsArg() << ", ";
    ss << GM_ADDR("outputs").AsArg() << ", ";
  } else {
    for (auto &input : fused_schedule_result.input_nodes) {
      ss << GM_ADDR(GenValidName(input->GetName())).AsArg() << ", ";
    }
    for (auto &output : fused_schedule_result.output_nodes) {
      ss << GM_ADDR(GenValidName(output->GetName())).AsArg() << ", ";
    }
  }
  ss << GM_ADDR("workspace").AsArg() << ", ";
  if (is_inductor) {
    ss << "AutofuseTilingData t";
  } else {
    ss << GM_ADDR("gm_tiling_data").AsArg();
  }
  ss << ")";
  return ss.str();
}

std::string Kernel::GenTilingFuncCall(const std::string &impl_graph_name, const std::string &tiling_data,
                                      uint32_t index, bool enable_group_parallel, bool need_sync_all) const {
  std::stringstream ss;
  ss << (index == 0 ? "    if (" : " else if (");
  if (enable_group_parallel) {
    ss << "MatchTilingKeyAndBlockDim(" << CamelToLowerSneak(tiling_data) << ", " << index << ")";
  } else {
    ss << CamelToLowerSneak(tiling_data) << ".tiling_key == " << std::to_string(index);
  }
  ss << ") {" << std::endl;
  ss << "      " << CamelToLowerSneak(impl_graph_name) << "(";
  if (use_list_tensor_) {
    ss << kInputTensorDescName << ", " << kOutputTensorDescName << ", ";
  } else {
    for (auto &input : this->inputs) {
      ss << input.Str() << ", ";
    }
    for (auto &output : this->outputs) {
      ss << output.Str() << ", ";
    }
  }
  ss << this->workspace_arg.Str() << ", ";
  for (auto &workspace : this->workspaces) {
    ss << "t." << workspace.Str() << ", ";
  }
  ss << "&" << tiling_data;
  ss << ");" << std::endl;
  if (need_sync_all) {
    ss << "      SyncAll();" << std::endl;
  }
  ss << "    }";
  return ss.str();
}

std::string Kernel::GenTilingFuncCall(const std::string &impl_graph_name, const std::string &tiling_data) const {
  std::stringstream string_stream;
  string_stream << CamelToLowerSneak(impl_graph_name) << "(";
  if (use_list_tensor_) {
    string_stream << kInputTensorDescName << ", " << kOutputTensorDescName << ", ";
  } else {
    for (auto &input : this->inputs) {
      string_stream << input.Str() << ", ";
    }
    for (auto &output : this->outputs) {
      string_stream << output.Str() << ", ";
    }
  }
  string_stream << this->workspace_arg.Str() << ", ";
  for (auto &workspace : this->workspaces) {
    string_stream << "t."<< workspace.Str() << ", ";
  }
  string_stream << "&" << tiling_data;
  string_stream << ");";
  return string_stream.str();
}

Status Kernel::GenSingleGroupKernelWithRegTilingKey(const ascir::FusedScheduledResult &fused_schedule_result,
                                                    const CodegenConfig& config, std::stringstream &ss,
                                                    std::stringstream &ss1, bool use_list_tensor) {
  std::string tiling_data_type = "AutofuseTilingData";
  std::unordered_set<const std::string *> kernel_file_ptr;
  auto schedule_graphs = fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  std::vector<TilingFuncCall> func_calls;
  uint32_t tiling_key = 0U;
  for (size_t i = 0; i < schedule_graphs.size(); i++) {
    Kernel kernel(schedule_graphs[i].GetName());
    kernel.tiler.SetTilingCaseId(i);
    kernel.SetUsingAttCalcQBTSizeConfig(config.using_att_calc_qbt_size);
    kernel.SetUseListTensor(use_list_tensor);
    GE_CHK_STATUS_RET(Kernel::ParseGraph(schedule_graphs[i], fused_schedule_result, kernel),
                      "Codegen parse graph failed");
    GE_CHK_STATUS_RET(kernel.GenerateKernelByNode(schedule_graphs[i], ss, kernel_file_ptr));
    std::string tmp;
    GE_CHK_STATUS_RET(kernel.Generate(schedule_graphs[i].GetName(), tiling_data_type, tmp, schedule_graphs[i]),
                      "Codegen generate kernel for %s failed", schedule_graphs[i].GetName().c_str());
    ss << tmp;
    std::string func_call;
    if (ascgen_utils::IsCubeType(schedule_graphs[i])) {
      func_call = kernel.GenCubeTilingFuncCall(schedule_graphs[i]);
    } else {
      func_call = kernel.GenTilingFuncCall(schedule_graphs[i].GetName(), "t");
    }
    func_calls.emplace_back(func_call, kernel.has_workspace_node, false);
  }
  std::vector<TilingFuncCall> current;
  std::vector<std::vector<TilingFuncCall>> per_group_func_calls;
  per_group_func_calls.emplace_back(std::move(func_calls));
  AppendFuncCall(ss1, per_group_func_calls, current, 0, tiling_key, ascgen_utils::IsCubeFusedScheduled(fused_schedule_result));
  return ge::SUCCESS;
}

Status Kernel::GenMulGroupKernelWithRegTilingKey(const ascir::FusedScheduledResult &fused_schedule_result,
                                                 const CodegenConfig& config, std::stringstream &ss,
                                                 std::stringstream &ss1, bool use_list_tensor) {
  std::string tiling_data_type = "AutofuseTilingData";
  std::unordered_set<const std::string *> kernel_file_ptr;
  uint32_t tiling_key = 0U;
  for (size_t graph_id = 0; graph_id < fused_schedule_result.node_idx_to_scheduled_results.size(); graph_id++) {
    auto scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
    for (size_t i = 0; i < scheduled_results.size(); i++) {
      auto schedule_groups = scheduled_results[i].schedule_groups;
      auto enable_group_parallel = scheduled_results[i].enable_group_parallel;
      std::vector<std::vector<TilingFuncCall>> per_group_func_calls;
      ascir::CubeTemplateType cv_fusion_type = scheduled_results[i].cube_type;
      if (cv_fusion_type == ascir::CubeTemplateType::kCommon) { // 暂时不处理兜底模板
        continue;
      }
      for (size_t j = 0; j < schedule_groups.size(); j++) {
        auto schedule_graphs = schedule_groups[j].impl_graphs;
        std::vector<TilingFuncCall> func_calls;
        bool vector_no_db_flag = true;
        for (size_t k = 0; k < schedule_graphs.size(); k++) {
          std::string tiling_data = "AscGraph" + std::to_string(graph_id) + "ScheduleResult" + std::to_string(i) +
                                    "G" + std::to_string(j) + "TilingData";
          Kernel kernel(schedule_graphs[k].GetName());
          kernel.SetUsingAttCalcQBTSizeConfig(config.using_att_calc_qbt_size);
          kernel.SetUseListTensor(use_list_tensor);
          kernel.tiler.SetTilingCaseId(k);
          kernel.tiler.EnableGroupParallel(enable_group_parallel);
          kernel.tpipe.cv_fusion_type = cv_fusion_type;
          GE_CHK_STATUS_RET(Kernel::ParseGraph(schedule_graphs[k], fused_schedule_result, kernel),
                            "Codegen parse graph failed");
          GE_CHK_STATUS_RET(kernel.GenerateKernelByNode(schedule_graphs[k], ss, kernel_file_ptr));
          if (IsCVFusionUBGraph(schedule_graphs[k], cv_fusion_type)) {
            GE_CHK_STATUS_RET(kernel.GenerateVecFuncOfCVFusion(ss, vector_no_db_flag), "Gen CV fusion Func failed");
          } else {
            std::string tmp;
            GE_CHK_STATUS_RET(kernel.Generate(schedule_graphs[k].GetName(), tiling_data, tmp, schedule_graphs[k]),
                              "Codegen generate kernel for %s failed", schedule_graphs[k].GetName().c_str());
            ss << tmp;
          }
          std::string filed_name = "t.graph" + std::to_string(graph_id) + "_result" + std::to_string(i) + "_g" +
                                   std::to_string(j) + "_tiling_data";
          bool need_sync_all = kernel.has_workspace_node && j != schedule_groups.size() - 1;
          std::string func_call;
          if (ascgen_utils::IsCubeType(schedule_graphs[k])) {
            func_calls.emplace_back(kernel.GenCubeTilingFuncCall(schedule_graphs[k]), kernel.has_workspace_node, true);
          } else if (cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
            GE_CHK_STATUS_RET(kernel.InitCVFusionAddr(ss1, vector_no_db_flag), "Init CV Fusion Addr failed");
            vector_no_db_flag = false; // 和schedule约定，不开db的vector在前面
            continue;
          } else {
            func_calls.emplace_back(kernel.GenTilingFuncCall(schedule_graphs[k].GetName(), filed_name),
                                    kernel.has_workspace_node, need_sync_all);
          }
        }
        if (!func_calls.empty()) {
          per_group_func_calls.emplace_back(std::move(func_calls));
        }
      }
      std::vector<TilingFuncCall> current;
      AppendFuncCall(ss1, per_group_func_calls, current, 0, tiling_key, ascgen_utils::IsCubeFusedScheduled(fused_schedule_result));
    }
  }
  return ge::SUCCESS;
}

Status Kernel::GenKernelFuncWithRegTilingKey(const ascir::FusedScheduledResult &fused_schedule_result,
                                             const CodegenConfig& config, std::stringstream &ss, std::stringstream &ss1,
                                             bool use_list_tensor) {
  if (ascgen_utils::IsSingleGroup(fused_schedule_result)) {
    GE_ASSERT_SUCCESS(GenSingleGroupKernelWithRegTilingKey(fused_schedule_result, config, ss, ss1, use_list_tensor));
  } else {
    GE_ASSERT_SUCCESS(GenMulGroupKernelWithRegTilingKey(fused_schedule_result, config, ss, ss1, use_list_tensor));
  }
  return ge::SUCCESS;
}

Status Kernel::GenSingleGroupKernelWithParseTilingData(const ascir::FusedScheduledResult &fused_schedule_result,
                                                       const std::vector<ge::AscGraph> &schedule_graphs,
                                                       const CodegenConfig& config, std::stringstream &ss,
                                                       std::stringstream &ss1, bool use_list_tensor,
                                                       std::unordered_set<const std::string *> &kernel_file_ptr) {
  for (size_t i = 0; i < schedule_graphs.size(); i++) {
    Kernel kernel(schedule_graphs[i].GetName());
    kernel.SetUsingAttCalcQBTSizeConfig(config.using_att_calc_qbt_size);
    kernel.SetUseListTensor(use_list_tensor);
    kernel.tiler.SetTilingCaseId(i);
    GE_CHK_STATUS_RET(Kernel::ParseGraph(schedule_graphs[i], fused_schedule_result, kernel),
                      "Codegen parse graph failed");
    GE_CHK_STATUS_RET(kernel.GenerateKernelByNode(schedule_graphs[i], ss, kernel_file_ptr));
    std::string tmp;
    GE_CHK_STATUS_RET(kernel.Generate(schedule_graphs[i].GetName(), "AutofuseTilingData", tmp, schedule_graphs[i]),
                      "Codegen generate kernel for %s failed", schedule_graphs[i].GetName().c_str());
    ss << tmp;
    if (ascgen_utils::IsCubeType(schedule_graphs[i])) {
      ss1 << kernel.GenCubeTilingFuncCall(schedule_graphs[i]);
    } else {
      ss1 << kernel.GenTilingFuncCall(schedule_graphs[i].GetName(), "t", i);
    }
  }
  return ge::SUCCESS;
}

Status Kernel::GenCubeCommonFuncForScheduleGroup(const ascir::FusedScheduledResult &fused_schedule_result,
                                                 const size_t graph_id, const size_t common_index,
                                                 const size_t group_index, const CodegenConfig &config,
                                                 std::stringstream &ss, std::stringstream &res_ss, const bool use_list_tensor,
                                                 std::unordered_set<const std::string *> &kernel_file_ptr) {
  const auto &scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
  const auto &schedule_groups = scheduled_results[common_index].schedule_groups;
  auto enable_group_parallel = scheduled_results[common_index].enable_group_parallel;
  ascir::CubeTemplateType cv_fusion_type = scheduled_results[common_index].cube_type;
  const auto &schedule_graphs = schedule_groups[group_index].impl_graphs;
  GE_ASSERT_TRUE(!schedule_graphs.empty(), "schedule_graphs is empty");
  for (size_t k = 0U; k < schedule_graphs.size(); k++) {
    Kernel kernel(schedule_graphs[k].GetName());
    kernel.SetUsingAttCalcQBTSizeConfig(config.using_att_calc_qbt_size);
    kernel.SetUseListTensor(use_list_tensor);
    kernel.tiler.SetTilingCaseId(k);
    kernel.tiler.EnableGroupParallel(enable_group_parallel);
    kernel.tpipe.cv_fusion_type = cv_fusion_type;
    GE_CHK_STATUS_RET(Kernel::ParseGraph(schedule_graphs[k], fused_schedule_result, kernel),
                      "Codegen parse graph failed");
    GE_CHK_STATUS_RET(kernel.GenerateKernelByNode(schedule_graphs[k], ss, kernel_file_ptr), "Gen api headers failed");

    if (ascgen_utils::IsCubeType(schedule_graphs[k])) {
      res_ss << kernel.GenCubeCommonTilingSingleFuncCall(schedule_graphs[k]);
      return ge::SUCCESS;
    } else {
      std::string tmp;
      GE_CHK_STATUS_RET(kernel.Generate(schedule_graphs[k].GetName(), "AutofuseTilingData", tmp, schedule_graphs[k]),
                        "Codegen generate cv kernel for %s failed", schedule_graphs[k].GetName().c_str());
      ss << tmp;
      res_ss << kernel.GenTilingFuncCall(schedule_graphs[k].GetName(), "t", k, enable_group_parallel, false)
             << std::endl;
    }
  }
  return ge::SUCCESS;
}

Status Kernel::GenCubeCommonFuncForAIV(const ascir::FusedScheduledResult &fused_schedule_result, const size_t graph_id,
                                       const size_t common_index, const size_t group_index, const CodegenConfig &config,
                                       std::stringstream &ss, std::stringstream &vec_ss, const bool use_list_tensor,
                                       std::unordered_set<const std::string *> &kernel_file_ptr) {
  vec_ss << "if ASCEND_IS_AIV {" << std::endl;
  vec_ss << "    SyncAll<false>();" << std::endl;
  vec_ss << "    #ifdef CV_AIV_NUM" << std::endl;
  vec_ss << "        if (GetBlockIdx() >= CV_AIV_NUM) {" << std::endl;
  vec_ss << "            return;" << std::endl;
  vec_ss << "        }" << std::endl;
  vec_ss << "    #endif" << std::endl;
  if (!IsEmptyTensorSence(fused_schedule_result)) {
    vec_ss << "    GET_TILING_DATA(t, gm_tiling_data);" << std::endl;
    GE_ASSERT_SUCCESS(GenCubeCommonFuncForScheduleGroup(fused_schedule_result, graph_id, common_index, group_index,
                                                        config, ss, vec_ss, use_list_tensor, kernel_file_ptr));
  }
  vec_ss << "}" << std::endl;
  return ge::SUCCESS;
}

Status Kernel::GenCubeCommonFuncForAICMix(const ascir::FusedScheduledResult &fused_schedule_result,
                                          const size_t graph_id, const size_t common_index, const size_t group_index,
                                          const CodegenConfig &config, std::stringstream &ss,
                                          std::stringstream &cube_ss, const bool use_list_tensor,
                                          std::unordered_set<const std::string *> &kernel_file_ptr) {
  cube_ss << "    #ifdef CV_AIC_NUM" << std::endl;
  cube_ss << "      if ASCEND_IS_AIC {" << std::endl;
  cube_ss << "        if (GetBlockIdx() >= CV_AIC_NUM) {" << std::endl;
  cube_ss << "            SyncAll<false>();" << std::endl;
  cube_ss << "            return;" << std::endl;
  cube_ss << "        }" << std::endl;
  cube_ss << "      }" << std::endl;
  cube_ss << "    #endif" << std::endl;
  cube_ss << "    uint32_t vec_wss =  0U;" << std::endl;
  cube_ss << "    #ifdef CV_VEC_WSS" << std::endl;
  cube_ss << "        vec_wss =  CV_VEC_WSS;" << std::endl;
  cube_ss << "    #endif" << std::endl;
  GE_ASSERT_SUCCESS(GenCubeCommonFuncForScheduleGroup(fused_schedule_result, graph_id, common_index, group_index,
                                                      config, ss, cube_ss, use_list_tensor, kernel_file_ptr));
  cube_ss << "    if ASCEND_IS_AIC {" << std::endl;
  cube_ss << "      SyncAll<false>();" << std::endl;
  cube_ss << "    }" << std::endl;
  return ge::SUCCESS;
}

Status Kernel::GenCubeCommonFuncForAIC(const ascir::FusedScheduledResult &fused_schedule_result, const size_t graph_id,
                                       const size_t common_index, const size_t group_index, const CodegenConfig &config,
                                       std::stringstream &ss, std::stringstream &cube_ss, const bool use_list_tensor,
                                       std::unordered_set<const std::string *> &kernel_file_ptr) {
  cube_ss << "if ASCEND_IS_AIC {" << std::endl;
  cube_ss << "    #ifdef CV_AIC_NUM" << std::endl;
  cube_ss << "        if (GetBlockIdx() >= CV_AIC_NUM) {" << std::endl;
  cube_ss << "            SyncAll<false>();" << std::endl;
  cube_ss << "            return;" << std::endl;
  cube_ss << "        }" << std::endl;
  cube_ss << "    #endif" << std::endl;
  cube_ss << "    uint32_t vec_wss =  0U;" << std::endl;
  cube_ss << "    #ifdef CV_VEC_WSS" << std::endl;
  cube_ss << "        vec_wss =  CV_VEC_WSS;" << std::endl;
  cube_ss << "    #endif" << std::endl;
  GE_ASSERT_SUCCESS(GenCubeCommonFuncForScheduleGroup(fused_schedule_result, graph_id, common_index, group_index,
                                                      config, ss, cube_ss, use_list_tensor, kernel_file_ptr));
  cube_ss << "    SyncAll<false>();" << std::endl;
  cube_ss << "}" << std::endl;
  return ge::SUCCESS;
}

Status Kernel::GenCubeCommonFuncOfCVFusion(const ascir::FusedScheduledResult &fused_schedule_result,
                                           const size_t graph_id, const size_t common_index,
                                           const CodegenConfig &config, std::stringstream &ss, std::stringstream &ss1,
                                           const bool use_list_tensor,
                                           std::unordered_set<const std::string *> &kernel_file_ptr) {
  const auto &scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
  const auto &schedule_groups = scheduled_results[common_index].schedule_groups;
  std::vector<std::vector<std::string>> per_group_func_calls;
  for (size_t j = 0U; j < schedule_groups.size(); j++) {
    std::vector<std::string> func_calls;
    auto schedule_graphs = schedule_groups[j].impl_graphs;
    GE_ASSERT_TRUE(!schedule_graphs.empty(), "schedule_graphs is empty");
    bool is_cube_group = ascgen_utils::IsCubeType(schedule_graphs[0]);
    if (is_cube_group) {
      // 区别cube group和vector group，cube group只有一张schedule_graphs
      std::stringstream cube_ss;
      cube_ss << "#ifdef CV_SAFETY_FUSION_MIX_MODE" << std::endl;
      GE_ASSERT_SUCCESS(GenCubeCommonFuncForAICMix(fused_schedule_result, graph_id, common_index, j, config, ss,
                                                   cube_ss, use_list_tensor, kernel_file_ptr));
      cube_ss << "#else" << std::endl;
      GE_ASSERT_SUCCESS(GenCubeCommonFuncForAIC(fused_schedule_result, graph_id, common_index, j, config, ss, cube_ss,
                                                use_list_tensor, kernel_file_ptr));
      cube_ss << "#endif" << std::endl;
      func_calls.emplace_back(cube_ss.str());
    } else {
      std::stringstream vec_ss;
      GE_ASSERT_SUCCESS(GenCubeCommonFuncForAIV(fused_schedule_result, graph_id, common_index, j, config, ss, vec_ss,
                                                use_list_tensor, kernel_file_ptr));
      func_calls.emplace_back(vec_ss.str());
    }
    if (!func_calls.empty()) {
      if (is_cube_group) {
        // 保证cube处理代码在vector之前
        per_group_func_calls.insert(per_group_func_calls.cbegin(), std::move(func_calls));
      } else {
        per_group_func_calls.emplace_back(std::move(func_calls));
      }
    }
  }
  AppendFuncCall(ss1, per_group_func_calls.cbegin(), per_group_func_calls.cend(), false);
  return ge::SUCCESS;
}

Status Kernel::GenMulGroupKernelWithParseTilingData(const ascir::FusedScheduledResult &fused_schedule_result,
                                                    const size_t graph_id, const CodegenConfig &config,
                                                    std::stringstream &ss, std::stringstream &ss1, bool use_list_tensor,
                                                    std::unordered_set<const std::string *> &kernel_file_ptr) {
  auto scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
  uint32_t function_id = kFuncIdBegin;
  for (size_t i = 0; i < scheduled_results.size(); i++) {
    auto schedule_groups = scheduled_results[i].schedule_groups;
    auto enable_group_parallel = scheduled_results[i].enable_group_parallel;
    ascir::CubeTemplateType cv_fusion_type = scheduled_results[i].cube_type;
    if (cv_fusion_type == ascir::CubeTemplateType::kCommon) {
      GE_ASSERT_SUCCESS(GenCubeCommonFuncOfCVFusion(fused_schedule_result, graph_id, i, config, ss, ss1,
                                                    use_list_tensor, kernel_file_ptr));
      continue;
    } else if (cv_fusion_type == ascir::CubeTemplateType::kDefault) {
      ss1 << (i == 0 ? "  if" : " else if ") << "(t." << "graph" << std::to_string(graph_id)
          << "_tiling_key == " << std::to_string(i) << ") {" << std::endl;
    }
    std::vector<std::vector<std::string>> per_group_func_calls;
    bool enable_parallel_compile = true;
    for (size_t j = 0; j < schedule_groups.size(); j++) {
      std::vector<std::string> func_calls;
      auto schedule_graphs = schedule_groups[j].impl_graphs;
      bool vector_no_db_flag = true;
      for (size_t k = 0; k < schedule_graphs.size(); k++) {
        std::string tiling_data = "AscGraph" + std::to_string(graph_id) + "ScheduleResult" + std::to_string(i) +
                                  "G" + std::to_string(j) + "TilingData";
        Kernel kernel(schedule_graphs[k].GetName());
        kernel.SetUsingAttCalcQBTSizeConfig(config.using_att_calc_qbt_size);
        kernel.SetUseListTensor(use_list_tensor);
        kernel.tiler.SetTilingCaseId(k);
        kernel.tiler.EnableGroupParallel(enable_group_parallel);
        kernel.tpipe.cv_fusion_type = cv_fusion_type;
        GE_CHK_STATUS_RET(Kernel::ParseGraph(schedule_graphs[k], fused_schedule_result, kernel),
                          "Codegen parse graph failed");
        GE_CHK_STATUS_RET(kernel.GenerateKernelByNode(schedule_graphs[k], ss, kernel_file_ptr),
                          "Gen api headers failed");

        std::string filed_name = "t.graph" + std::to_string(graph_id) + "_result" + std::to_string(i) + "_g" +
                                 std::to_string(j) + "_tiling_data";
        bool need_sync_all = kernel.has_workspace_node && j != schedule_groups.size() - 1;
        if (ascgen_utils::IsCubeType(schedule_graphs[k])) {
          func_calls.emplace_back(kernel.GenCubeTilingFuncCall(schedule_graphs[k]));
        } else if (cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
          GE_CHK_STATUS_RET(kernel.GenerateVecFuncOfCVFusion(ss, vector_no_db_flag), "Gen CV fusion Func failed");
          GE_CHK_STATUS_RET(kernel.InitCVFusionAddr(ss1, vector_no_db_flag), "Init CV Fusion Addr failed");
          vector_no_db_flag = false;  // 和schedule约定，不开db的vector在前面
          continue;
        } else {
          std::string tmp;
          GE_CHK_STATUS_RET(kernel.Generate(schedule_graphs[k].GetName(), tiling_data, tmp, schedule_graphs[k]),
                            "Codegen generate kernel for %s failed", schedule_graphs[k].GetName().c_str());
          ss << tmp;
          func_calls.emplace_back(kernel.GenTilingFuncCall(schedule_graphs[k].GetName(), filed_name, k,
                                                           enable_group_parallel, need_sync_all));
          enable_parallel_compile = (enable_parallel_compile && kernel.GetEnableParallelCompile());
        }
      }
      if (!func_calls.empty()) {
        per_group_func_calls.emplace_back(std::move(func_calls));
      }
    }
    auto max_group_per_compile_unit = GetMaxGroupPerCompileUnit(enable_parallel_compile);
    if (per_group_func_calls.size() <= static_cast<size_t>(max_group_per_compile_unit)) {
      AppendFuncCall(ss1, per_group_func_calls.cbegin(), per_group_func_calls.cend());
    } else {
      const auto kernel_args = PackingFuncArgs("AutofuseTilingData", fused_schedule_result, use_list_tensor);
      const auto packing_func_names =
          Kernel::GenPackingFunctions(ss, kernel_args, per_group_func_calls, max_group_per_compile_unit, function_id);
      GenPackingFunctionCalls(ss1, kernel_args, packing_func_names);
    }
    if (cv_fusion_type == ascir::CubeTemplateType::kDefault) {
      ss1 << "  }";
    }
  }
  FakeTilingIds(ss, function_id);
  return ge::SUCCESS;
}

Status Kernel::GenKernelFuncWithParseTilingData(const ascir::FusedScheduledResult &fused_schedule_result,
                                                const CodegenConfig& config, std::stringstream &ss,
                                                std::stringstream &ss1, bool use_list_tensor) {
  std::unordered_set<const std::string *> kernel_file_ptr;
  for (size_t graph_id = 0; graph_id < fused_schedule_result.node_idx_to_scheduled_results.size(); graph_id++) {
    auto scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
    if ((fused_schedule_result.node_idx_to_scheduled_results.size() == 1) && (scheduled_results.size() == 1) &&
        (scheduled_results[0].schedule_groups.size() == 1)) {
      auto schedule_graphs = scheduled_results[0].schedule_groups[0].impl_graphs;
      GE_ASSERT_SUCCESS(GenSingleGroupKernelWithParseTilingData(fused_schedule_result, schedule_graphs, config, ss,
                                                                ss1, use_list_tensor, kernel_file_ptr));
    } else {
      GE_ASSERT_SUCCESS(GenMulGroupKernelWithParseTilingData(fused_schedule_result, graph_id, config, ss,
                                                             ss1, use_list_tensor, kernel_file_ptr));
    }
  }
  return ge::SUCCESS;
}

int64_t Kernel::GetMaxGroupPerCompileUnit(bool enable_parallel_compile) {
  uint32_t max_group_per_compile_unit = std::numeric_limits<uint32_t>::max();
  if (enable_parallel_compile) {
    auto backend_spec = optimize::BackendSpec::GetInstance();
    if (backend_spec != nullptr) {
      max_group_per_compile_unit = backend_spec->max_group_num_per_compile_unit;
    }
  }
  return max_group_per_compile_unit;
}

ge::Status Kernel::GenCubeCommonTiling(std::stringstream &ss, const bool is_batch) const {
  if (is_batch) {
    ss << "  batch_mat_mul_v3<";
  } else {
    ss << "  mat_mul_v3<";
  }

  ss << "API_LEVEL, A_TRANS, B_TRANS, BATCH_MODEL, MODEL, FULL_LOAD, L0C2OUT_MODEL>(";
  return ge::SUCCESS;
}

std::string Kernel::GenCubeTilingSingleFuncCall(const bool is_batch, const bool is_cv_fuse, bool is_bias,
                                                bool is_offset_w) const {
  std::stringstream ss;
  GE_CHK_STATUS(GenCubeCommonTiling(ss, is_batch), "GenCubeCommonTilingHead failed");

  if (use_list_tensor_) {
    ss << kInputTensorDescName << ", " << kOutputTensorDescName << ", ";
  } else {
    if (this->inputs.size() < (2U + (is_bias ? 1U : 0U) + (is_offset_w ? 1U : 0U))) {
      ss << this->inputs[0].Str() << ", "; // a矩阵、b矩阵同输入存在ascgraph的matmul有两个输入，Ascackend只有一个输入，需多加一个
    }
    for (auto &input : this->inputs) {
      ss << input.Str() << ", ";
    }
    if (!is_bias) { // 无bias场景
      ss << "nullptr, ";
    }
    if (!is_offset_w) { // 无offset_w场景
      ss << "nullptr, ";
    }
    for (auto &output : this->outputs) {
      ss << output.Str() << ", ";
    }
    if (this->outputs.empty()) {
      ss << (is_cv_fuse ? "nullptr, " : "output_0, ");
    }
  }
  ss << this->workspace_arg.Str() << ", ";
  ss << "gm_tiling_data";
  ss << (is_cv_fuse ? ", &CV_FUSION_ADDR" : "");
  ss << ");" << std::endl;
  return ss.str();
}

std::string Kernel::GenCubeCommonTilingSingleFuncCall(const ascir::ImplGraph &impl_graph) const {
  auto is_batch = ascgen_utils::IsCubeTypeWithBatch(impl_graph);
  auto has_bias = ascgen_utils::IsCubeTypeWithBias(impl_graph);
  auto has_offset_w = ascgen_utils::IsCubeTypeWithOffsetW(impl_graph);
  std::stringstream ss;
  GE_CHK_STATUS(GenCubeCommonTiling(ss, is_batch), "GenCubeCommonTilingHead failed");
  if (use_list_tensor_) {
    ss << kInputTensorDescName << ", " << kOutputTensorDescName << ", ";
  } else {
    auto min_inputs_num = 1U + (has_bias ? 1U : 0U) + (has_offset_w ? 1U : 0U);
    GE_ASSERT_TRUE(this->inputs.size() >= min_inputs_num, "cube inputs num [%u] < min_inputs_num [%u]",
                   this->inputs.size(), min_inputs_num);
    // a矩阵、b矩阵同输入存在ascgraph的matmul有两个输入，Ascackend只有一个输入，需多加一个输入再生成kernel函数
    (this->inputs.size() == min_inputs_num) ? (ss << this->inputs[0].Str() << ", ") : (ss <<  "");
    for (auto &input : this->inputs) {
      ss << input.Str() << ", ";
    }
    if (!has_bias) { // 无bias场景
      ss << "nullptr, ";
    }
    if (!has_offset_w) { // 无offset_w场景
      ss << "nullptr, ";
    }
    for (auto &output : this->outputs) {
      ss << output.Str() << ", ";
    }
    if (this->outputs.empty()) {
      ss << this->workspace_arg.Str() << ", ";  // cube输出到workspace地址
    }
  }
  ss << this->workspace_arg.Str();
  // cube workspace的位置需要计算 workspace + vector的偏移
  ss << " + vec_wss, gm_tiling_data);" << std::endl;
  return ss.str();
}

std::string Kernel::GenCubeTilingFuncCall(const ascir::ImplGraph &impl_graph) const {
  auto is_batch = ascgen_utils::IsCubeTypeWithBatch(impl_graph);
  auto is_bias = ascgen_utils::IsCubeTypeWithBias(impl_graph);
  auto is_offset_w = ascgen_utils::IsCubeTypeWithOffsetW(impl_graph);
  std::stringstream ss;
  ss << "#ifdef CV_UB_FUSION" << std::endl;
  ss << GenCubeTilingSingleFuncCall(is_batch, true, is_bias, is_offset_w);
  ss << "#else" << std::endl;
  ss << GenCubeTilingSingleFuncCall(is_batch, false, is_bias, is_offset_w);
  ss << "#endif" << std::endl;
  return ss.str();
}

Status Kernel::GenKernelFuncByTilingKey(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss,
                                        bool use_list_tensor, const CodegenConfig& config,
                                        const std::string &kernel_task_type) {
  std::stringstream ss1;
  std::string graph_name = GenValidName(fused_schedule_result.fused_graph_name.GetString());
  if (config.is_inductor) {
    ss1 << Kernel::KernelFuncDeclare(graph_name, fused_schedule_result, use_list_tensor, config.is_inductor) << " {"
        << std::endl;
    ss1 << "    KERNEL_TASK_TYPE_DEFAULT(" << kernel_task_type << ");" << std::endl;
  } else {
    ss1 << Kernel::KernelFuncDeclare(graph_name, fused_schedule_result, use_list_tensor, config.is_inductor) << " {"
        << std::endl;
    if (!ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
      ss1 << "  REGISTER_TILING_DEFAULT(" << "AutofuseTilingData);" << std::endl;
      if (IsEmptyTensorSence(fused_schedule_result)) {
        ss1 << std::endl << "}" << std::endl;
        ss << ss1.str();
        return ge::SUCCESS;
      } else {
        ss1 << "  GET_TILING_DATA(t, gm_tiling_data);" << std::endl;
      }
    } else if (ascgen_utils::IsCubeCommonFusedScheduled(fused_schedule_result)){
      // cv 兜底模板需要在具体cube实现前先引入含有模板宏定义的头文件防止核类型启动错误
      ss << "#include \"autofuse_cube_tiling_data.h\"" << std::endl;
    }
  }
  if (use_list_tensor) {
    ss1 << "  ListTensorDesc " << kInputTensorDescName << "((__gm__ void *)inputs);" << std::endl;
    ss1 << "  ListTensorDesc " << kOutputTensorDescName << "((__gm__ void *)outputs);" << std::endl;
  }

  // cv 兜底模板当前没有区分c、v的tiling_key，默认都走tiling_data解析
  if (ascgen_utils::CanUseTilingKey(fused_schedule_result) && !config.is_inductor &&
      !ascgen_utils::IsCubeCommonFusedScheduled(fused_schedule_result)) {
    GE_ASSERT_SUCCESS(GenKernelFuncWithRegTilingKey(fused_schedule_result, config, ss, ss1, use_list_tensor));
  } else {
    GE_ASSERT_SUCCESS(GenKernelFuncWithParseTilingData(fused_schedule_result, config, ss, ss1, use_list_tensor));
  }
  ss1 << std::endl << "}" << std::endl;
  ss << ss1.str();
  return ge::SUCCESS;
}

void Kernel::SetUseListTensor(bool use_list_tensor) {
  use_list_tensor_ = use_list_tensor;
}

void Kernel::SetUsingAttCalcQBTSizeConfig(bool using_att_calc_qbt_size) {
  this->tpipe.SetUsingAttCalcQBTSizeConfig(using_att_calc_qbt_size);
}

void Kernel::SetEnableParallelCompile(bool enable_parallel_compile) {
  enable_parallel_compile_ = enable_parallel_compile;
}

bool Kernel::GetEnableParallelCompile() const {
  return enable_parallel_compile_;
}

void Kernel::AppendFuncCall(std::stringstream &ss, std::vector<std::vector<std::string>>::const_iterator begin,
                            std::vector<std::vector<std::string>>::const_iterator end, bool need_sync_all) {
  for (auto it = begin; it != end; ++it) {
    if (it != begin && need_sync_all) {
      ss << "    AscendC::PipeBarrier<PIPE_ALL>();" << std::endl;
    }
    for (const auto &call_statement : *it) {
      ss << call_statement;
    }
    ss << std::endl;
  }
}

void Kernel::AppendFuncCall(std::stringstream &ss,
                            std::vector<std::vector<TilingFuncCall>> &per_group_func_calls,
                            std::vector<TilingFuncCall> &current, size_t depth, uint32_t &tiling_key, bool is_cube) {
  if (depth == per_group_func_calls.size())  {
    if (!is_cube) {
      ss << (tiling_key == 0U ? "  if " : " else if ") << "(TILING_KEY_IS(" << std::to_string(tiling_key) << ")) {" << std::endl;
    }
    for (const auto &tiling_func_call : current) {
      ss << "    " << tiling_func_call.func_call_ << std::endl;
      if (tiling_func_call.need_sync_all_) {
        ss << "    SyncAll();" << std::endl;
      }
    }
    if (!is_cube) {
      ss << "  }";
    }
    tiling_key++;
    return;
  }
  for (const auto &func_call : per_group_func_calls[depth]) {
    current.push_back(func_call);
    AppendFuncCall(ss, per_group_func_calls, current, depth + 1, tiling_key, is_cube);
    current.pop_back();
  }
}

std::vector<Variable> Kernel::PackingFuncArgs(const std::string &tiling_data_type,
                                              const ascir::FusedScheduledResult &fused_schedule_result,
                                              bool use_list_tensor) {
  std::vector<Variable> args;
  if (use_list_tensor) {
    args.emplace_back(Type("ListTensorDesc&"), kInputTensorDescName);
    args.emplace_back(Type("ListTensorDesc&"), kOutputTensorDescName);
  } else {
    for (auto &input : fused_schedule_result.input_nodes) {
      args.emplace_back(GM_ADDR(GenValidName(input->GetName())));
    }
    for (auto &output : fused_schedule_result.output_nodes) {
      args.emplace_back(GM_ADDR(GenValidName(output->GetName())));
    }
  }
  args.emplace_back(GM_ADDR("workspace"));
  args.emplace_back(Type(tiling_data_type + "&"), "t");
  return args;
}

std::vector<std::string> Kernel::GenPackingFunctions(std::stringstream &ss_define,
                                                     const std::vector<Variable> &kernel_args,
                                                     const std::vector<std::vector<std::string>> &per_group_func_calls,
                                                     int64_t max_group_per_compile_unit,
                                                     uint32_t &function_id) {
  std::vector<std::string> func_names;
  auto remaining_groups = static_cast<int64_t>(per_group_func_calls.size());
  auto begin = per_group_func_calls.cbegin();
  while (remaining_groups > 0) {
    const auto num = std::min(remaining_groups, max_group_per_compile_unit);
    const auto end = begin + num;
    // 函数名需要以数字结尾, 但不能与tiling_key相同, 否则rts加载kernel会失败
    const auto &func_name = "packed_functions_8" + std::to_string(function_id);
    // codegen packing func definition
    ss_define << PackingFuncDeclare(func_name, kernel_args) << ";" << std::endl;
    ss_define << "#if TILING_KEY_VAR == " << function_id << std::endl;
    ss_define << PackingFuncDeclare(func_name, kernel_args) << "{" << std::endl;
    AppendFuncCall(ss_define, begin, end);
    ss_define << "}" << std::endl;
    ss_define << "#endif" << std::endl;
    remaining_groups -= num;
    begin += num;
    function_id += 1;
    func_names.emplace_back(func_name);
  }
  return func_names;
}

std::string Kernel::PackingFuncDeclare(const std::string &func_name, const std::vector<Variable> &kernel_args) {
  std::stringstream ss;
  ss << "extern \"C\" __aicore__ void ";
  ss << func_name << "(";
  std::vector<std::string> args;
  args.reserve(kernel_args.size());
  for (const auto &arg : kernel_args) {
    args.emplace_back(arg.AsArg());
  }
  ss << ge::StringUtils::Join(args.cbegin(), args.cend(), ", ");
  ss << ")";
  return ss.str();
}

void Kernel::GenPackingFunctionCalls(stringstream &ss, const vector<Variable> &kernel_args,
                                     const vector<std::string> &func_names) {
  std::vector<std::string> args;
  args.reserve(kernel_args.size());
  for (const auto &arg : kernel_args) {
    args.emplace_back(arg.Str());
  }
  bool need_sync = false;
  const auto &func_args = ge::StringUtils::Join(args.cbegin(), args.cend(), ", ");
  for (const auto &func_name : func_names) {
    if (need_sync) {
      ss << "    AscendC::PipeBarrier<PIPE_ALL>();" << std::endl;
    }
    std::string function_id = func_name.substr(func_name.rfind("_") + 1);
    ss << "    " << func_name << "(" << func_args << ");" << std::endl;
    need_sync = true;
  }
}

void Kernel::FakeTilingIds(stringstream &ss, uint32_t function_id_end) {
  if (function_id_end != kFuncIdBegin) {
    ss << "inline void fake_tiling_ids() {" << std::endl;
    ss << "  int32_t g_tilingKey = -1;" << std::endl;
    ss << "  if (TILING_KEY_IS(0)) {}" << std::endl;
    for (uint32_t function_id = kFuncIdBegin; function_id < function_id_end; ++function_id) {
      ss << "  if (TILING_KEY_IS(" << function_id << ")) {}" << std::endl;
    }
    ss << "}" << std::endl;
  }
}

Status Kernel::GenerateMacro(stringstream &ss) {
  std::stack<const Loop *> loop_stack;
  loop_stack.push(&(this->root_loop));
  while (!loop_stack.empty()) {
    const Loop *current_loop = loop_stack.top();
    loop_stack.pop();

    for (auto &body : current_loop->bodys) {
      if (body.type == LoopType::LOOP) {
        GE_ASSERT_NOTNULL(body.loop);
        loop_stack.push(body.loop);
      } else if (body.call->unit == ge::ComputeUnit::kUnitCube) {
        string macro_result;
        body.call->GenerateMacro(macro_result);
        ss << macro_result << std::endl;
        break;
      }
    }
  }
  return ge::SUCCESS;
}

Status Kernel::GenerateKernelByNode(const ascir::ImplGraph &graph, stringstream &ss,
                                    std::unordered_set<const std::string *> &kernel_file_ptr) {
  GE_CHK_STATUS_RET(GenerateMacro(ss), "Generate Macro failed");
  std::string npu_arch;
  GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetCurrentPlatformString(npu_arch));
  const bool need_marco = (npu_arch == "3510");
  if (need_marco) {
    ss << "#if defined(__DAV_C310__) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3510))"
       << std::endl;
  }
  for (const auto &node : graph.GetAllNodes()) {
    auto impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
    GE_ASSERT_NOTNULL(impl, "GetAscIrCodegenImpl of node %s[%s] is null", node->GetTypePtr(), node->GetNamePtr());
    for (const auto &header_str : impl->LoadApiHeaderFiles()) {
      const auto &file = AscendCApiRegistry::GetInstance().GetFileContent(header_str);
      if (!file.empty()) {
        if (kernel_file_ptr.find(&(file)) == kernel_file_ptr.end()) {
          kernel_file_ptr.insert(&(file));
          ss << file;
        }
      }
    }
  }
  if (need_marco) {
    ss << "#endif" << std::endl;
  }
  return ge::SUCCESS;
}

Status Kernel::GlobalTensorDefine(std::string &result) const {
  std::stringstream ss;
  for (std::size_t i = 0; i < this->inputs.size(); i++) {
    const auto &tensor = this->tpipe.tensors.find(this->input_tensors[i]);
    GE_ASSERT_TRUE((tensor != this->tpipe.tensors.end()), "Codegen input tensor id[%ld] not found",
                   this->input_tensors[i]);
    ss << "    " << tensor->second.Define() << std::endl;
  }

  for (std::size_t i = 0; i < this->outputs.size(); i++) {
    const auto &tensor = this->tpipe.tensors.find(this->output_tensors[i]);
    GE_ASSERT_TRUE((tensor != this->tpipe.tensors.end()), "Codegen output tensor id[%ld] not found",
                   this->output_tensors[i]);
    ss << "    " << tensor->second.Define() << std::endl;
  }

  for (std::size_t i = 0; i < this->constant_tensors.size(); i++) {
    auto tensor = this->tpipe.tensors.find(this->constant_tensors[i]);
    GE_ASSERT_TRUE((tensor != this->tpipe.tensors.end()), "Codegen concat tensor id[%ld] not found",
                   this->constant_tensors[i]);
    GELOGI("const_value_expr: %s", tensor->second.const_value_expr.Str().get());

    string const_value = tensor->second.const_value_expr == 0 ? tensor->second.const_value
                                                              : tiler.Size(tensor->second.const_value_expr, true);
    ss << "    " << tensor->second.DefineConst(const_value.c_str()) << std::endl;
    GELOGI("Define ss value: %s", ss.str().c_str());
  }

  for (std::size_t i = 0; i < this->ub_scalar_tensors.size(); i++) {
    auto tensor = this->tpipe.tensors.find(this->ub_scalar_tensors[i]);
    GE_ASSERT_TRUE((tensor != this->tpipe.tensors.end()), "Codegen ub_scalar tensor id[%ld] not found",
                   this->ub_scalar_tensors[i]);

    std::string def_ub_scalar;
    GE_CHK_STATUS_RET(tensor->second.DefineUbScalar(def_ub_scalar));
    ss << "    " << def_ub_scalar;
    GELOGI("Define ub_scalar var:", def_ub_scalar.c_str());
  }
  ss << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

Status Kernel::GlobalTensorAssign(std::string &result) const {
  std::stringstream ss;
  for (std::size_t i = 0; i < this->inputs.size(); i++) {
    const auto &tensor = this->tpipe.tensors.find(this->input_tensors[i]);
    GE_ASSERT_TRUE((tensor != this->tpipe.tensors.end()), "Codegen input tensor id[%ld] not found",
                   this->input_tensors[i]);
    std::string local_result;
    GE_CHK_STATUS_RET(tensor->second.SetGlobalBuffer(this->inputs[i], "", local_result),
                      "Codegen set global buffer failed");
    ss << local_result << std::endl;
  }

  for (std::size_t i = 0; i < this->outputs.size(); i++) {
	  const auto &tensor = this->tpipe.tensors.find(this->output_tensors[i]);
    GE_ASSERT_TRUE((tensor != this->tpipe.tensors.end()), "Codegen output tensor id[%ld] not found",
                   this->output_tensors[i]);

    std::string local_result;
    GE_CHK_STATUS_RET(tensor->second.SetGlobalBuffer(this->outputs[i], "", local_result),
                      "Codegen set global buffer failed");
    ss << local_result << std::endl;
  }
  ss << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

Status TPipe::GetCVFusionCubeOutputUBTensorIdAndQueId(const ascir::ImplGraph &graph) {
  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Workspace>(node)) {
      for (auto &peer_input : node->outputs[0].anchor.GetPeerInDataAnchors()) {
        auto next_node = std::dynamic_pointer_cast<ge::AscNode>(peer_input->GetOwnerNode());
        GE_ASSERT_NOTNULL(next_node, "Codegen CV Fusion get next node after workspace node failed");
        if (IsOps<Load>(next_node)) {
          this->cube_output_tensor_id = next_node->outputs[0].attr.mem.tensor_id;
          this->cube_output_que_id = next_node->outputs[0].attr.que.id;
          return ge::SUCCESS;
        }
      }
      GELOGE(ge::FAILED, "Codegen CV Fusion Load node next to Workspace not found");
      return ge::FAILED;
    }
  }
  GELOGE(ge::FAILED, "Codegen CV Fusion get Workspace node failed");
  return ge::FAILED;
}

static void AddCommaIfNeeded(bool &is_first, std::stringstream &tensor_size_max) {
  if (is_first) {
    is_first = false;
  } else {
    tensor_size_max << ", ";
  }
}

Status TPipe::ParseTBufReuse(TBuf buf, std::string& reuse_dtype_name, bool& is_buf_reuse,
                             std::vector<const Tensor *>& reuse_buf_tensors, std::stringstream &tensor_size_max) const {
  tensor_size_max << KernelUtils::Max() << "(";

  bool is_first = true;
  for (auto mid : buf.merge_scopes) {
    auto merge_scope = this->merge_scopes.find(mid);
    if (merge_scope == this->merge_scopes.end()) {
      GELOGE(ge::FAILED, "Codegen merge scope not found:%ld", mid);
      return ge::FAILED;
    }
    AddCommaIfNeeded(is_first, tensor_size_max);
    tensor_size_max << merge_scope->second.size;
  }

  for (auto tid : buf.not_merge_tensors) {
    auto tensor = this->tensors.find(tid);
    if (tensor == this->tensors.end()) {
      GELOGE(ge::FAILED, "Codegen tensor not found:%ld", tid);
      return ge::FAILED;
    }
    AddCommaIfNeeded(is_first, tensor_size_max);
    std::string dtype_name;
    GE_CHK_STATUS_RET(Tensor::DtypeName(tensor->second.dtype, dtype_name), "Codegen get data type:%d failed",
                      static_cast<int32_t>(tensor->second.dtype));
    if (is_buf_reuse) {
      if (reuse_dtype_name == "") {
        reuse_dtype_name = dtype_name;
      } else {
        if (reuse_dtype_name != dtype_name) {
          is_buf_reuse = false;
        }
      }
    }
    tensor_size_max << tensor->second.size << " * sizeof(" << dtype_name << ")";
    reuse_buf_tensors.push_back(&tensor->second);
  }

  for (auto tmp_buf_size : buf.tmp_buf_size_list) {
    AddCommaIfNeeded(is_first, tensor_size_max);
    if (this->cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
      tensor_size_max << this->tiler.ActualSize(tmp_buf_size);
    } else {
      tensor_size_max << this->tiler.Size(tmp_buf_size);
    }
  }

  if (reuse_buf_tensors.size() == 0) {
    is_buf_reuse = false;
  }
  tensor_size_max << ")";
  return ge::SUCCESS;
}

Status TPipe::LocalTensorDefine(std::string &result) const {
  stringstream ss;
  for (auto &pair : this->tensors) {
    auto &t = pair.second;
    if (t.alloc_type != ge::AllocType::kAllocTypeGlobal) {
      ss << "    " << t.AsArg() << ";" << std::endl;
    }
  }
  ss << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

std::string TPipe::TensorSizeDefine() const {
  stringstream ss;

  for (auto &pair : this->tensors) {
    auto &t = pair.second;
    if ((t.alloc_type == ge::AllocType::kAllocTypeQueue) || (t.alloc_type == ge::AllocType::kAllocTypeBuffer)) {
      ss << "    " << t.size.Define() << std::endl;
      ss << "    " << t.actual_size.Define() << std::endl;
    }
  }
  return ss.str();
}

Status TPipe::TensorSizeAssign(std::string dtype_name, std::string &result) const {
  stringstream ss;

  for (auto &pair : this->tensors) {
    auto &t = pair.second;
    if ((t.alloc_type == ge::AllocType::kAllocTypeQueue) || (t.alloc_type == ge::AllocType::kAllocTypeBuffer)) {
      if (t.is_ub_scalar) {
        std::string tensor_dtype_name;
        GE_CHK_STATUS_RET(Tensor::DtypeName(t.dtype, tensor_dtype_name), "Codegen get data type:%d failed",
                          static_cast<int32_t>(t.dtype));
 	      ss << t.size.Str() << " = KernelUtils::BlkAlign<" << tensor_dtype_name << ">(1);" << std::endl;
 	    } else {
 	      ss << t.size.Str() << " = stage_size / sizeof(" << dtype_name << ");" << std::endl;
 	    }
    }
  }
  ss << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

std::string TPipe::GenDuplicateBufDefine(const std::set<std::pair<std::string, std::string>>& pre_api_extract_dup) const {
  std::stringstream ss;
  int32_t i = 1;
  for (auto [const_val, const_dtype] : pre_api_extract_dup) {
    const std::string index_str = std::to_string(i);
    ss << "    TBuf<TPosition::VECCALC> builtin_tmp_buffer_" << index_str << ";" << std::endl;
    std::string local_tensor_name = "local_blk_tensor_of_" + const_dtype + "_" + const_val;
    ss << "    LocalTensor<" << const_dtype << "> " << local_tensor_name << ";" << std::endl;
    i++;
  }
  return ss.str();
}

std::string TPipe::GenDuplicateBufAssign(const std::set<std::pair<std::string, std::string>>& pre_api_extract_dup) const {
  std::stringstream ss;
  int32_t i = 1;
  for (auto [const_val, const_dtype] : pre_api_extract_dup) {
    const std::string index_str = std::to_string(i);
    ss << "tpipe.InitBuffer(builtin_tmp_buffer_" << index_str << ", ONE_BLK_SIZE);" << std::endl;
    ss << "LocalTensor<uint8_t> builtin_tmp_buf_" << index_str << " = builtin_tmp_buffer_" << index_str
       << ".Get<uint8_t>();" << std::endl;
    std::string local_tensor_name = "local_blk_tensor_of_" + const_dtype + "_" + const_val;
    ss << local_tensor_name << " = builtin_tmp_buf_" << index_str << ".template ReinterpretCast<" << const_dtype
       << ">();" << std::endl;
    if (const_dtype == "half" || const_dtype == "float" || const_dtype == "double") {
      const_val += ".0";
    }
    ss << "Duplicate(" << local_tensor_name << "[0], (" << const_dtype << ")" << const_val <<
        ", ONE_BLK_SIZE / sizeof(" << const_dtype << "));"<< std::endl;
    i++;
  }
  return ss.str();
}

Status TPipe::BlkTensorDefine(std::string &result) const {
  // 遍历所有需要生成blk tensor 的 tensor id
  stringstream ss;
  for (auto &id : this->need_gen_blk_tensors) {
    auto tensor_ptr = this->GetTensor(id);
    GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, ge::FAILED, "BlkTensorAllocAndInit need_gen_blk_tensors failed");
    std::string scalar_t_buf_name = tensor_ptr->name + "_tbuf";
    std::string scalar_local_blk_tensor_name = "local_blk_tensor_of_" + tensor_ptr->name;
    ss << "    TBuf<TPosition::VECCALC> " << scalar_t_buf_name << ";" << std::endl;
    ss << "    LocalTensor<" << tensor_ptr->type << "> " << scalar_local_blk_tensor_name << ";" << std::endl;
  }
  ss << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

Status TPipe::BlkTensorAssign(std::string &result) const {
  // 遍历所有需要生成blk tensor 的 tensor id
  stringstream ss;
  for (auto &id : this->need_gen_blk_tensors) {
    auto tensor_ptr = this->GetTensor(id);
    GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, ge::FAILED, "BlkTensorAllocAndInit need_gen_blk_tensors failed");
    std::string scalar_t_buf_name = tensor_ptr->name + "_tbuf";
    std::string scalar_local_blk_tensor_name = "local_blk_tensor_of_" + tensor_ptr->name;
    ss << "tpipe.InitBuffer(" << scalar_t_buf_name << ", 32);" << std::endl;
    ss << scalar_local_blk_tensor_name << " = " << scalar_t_buf_name << ".Get<" << tensor_ptr->type << ">();" << std::endl;

    ss << "Duplicate(" << scalar_local_blk_tensor_name << "[0], static_cast<" << tensor_ptr->type
       << ">(" << tensor_ptr->const_value << "), static_cast<uint64_t>(32/"
       << "sizeof(" << tensor_ptr->type << ")));" << std::endl;
    ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;
  }
  ss << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

Status Loop::ActualSizeDefine(const Tiler &tiler, const TPipe &tpipe, std::string &result) {
  std::stringstream ss;
  if (this->axis_id == ge::kIdNone) {
    for (const auto &body : this->bodys) {
      if (body.type == LoopType::LOOP) {
        GE_CHK_STATUS_RET(body.loop->ActualSizeDefine(tiler, tpipe, result), "Get axis id failed.");
      }
    }
    return ge::SUCCESS;
  }
  const auto &axis = tiler.GetAxis(this->axis_id);
  if (axis.type == Axis::Type::kAxisTypeTileOuter) {
    const auto &tile_inner = tiler.GetAxis(axis.split_pair_other_id);
    ge::Expression actual_size = ge::Symbol(tile_inner.actual_size.name.c_str());
    tpipe.tiler.actual_sizes.emplace_back(std::make_pair(tile_inner.size_expr, actual_size));
    ss << tile_inner.actual_size.AsArg() << " = stage_size;" << std::endl;
  }
  result = ss.str();
  return ge::SUCCESS;
}

Status Kernel::GenerateVecFuncOfCVFusion(std::stringstream &result, bool vector_no_db_flag) {
  std::string tiling_data_type = "AutofuseTilingData";
  if (vector_no_db_flag) {
    result << R"(
#include "cmct/block/block_scheduler_policy.h"
#include "cmct/block/block_scheduler_utils.h"
#include "cmct/utils/status_utils.h"
#include "autofuse_cube_tiling_data.h"
)" << std::endl;
    result << "#ifdef CV_UB_NO_DB" << std::endl;
  } else {
    result << "#ifdef CV_UB_DB" << std::endl;
  }
  result << R"(
class AutoFusionVector {
  public:
    __aicore__ inline AutoFusionVector() {};
)" << std::endl;
  result << "    struct Arguments {" << std::endl;
  for (auto &input : this->inputs) {
    result << "     " << input.AsArg() << "{nullptr};" << std::endl;
  }
  for (auto &output : this->outputs) {
    result << "     " << output.AsArg() << "{nullptr};" << std::endl;
  }
  result << "    };" << std::endl << std::endl;

  result << "    struct Params {" << std::endl;
  for (auto &input : this->inputs) {
    result << "     " << input.AsArg() << "{nullptr};" << std::endl;
  }
  for (auto &output : this->outputs) {
    result << "     " << output.AsArg() << "{nullptr};" << std::endl;
  }
  result << "    };" << std::endl;
  for (auto &input : this->inputs) {
    result << "    " << input.AsArg() << "{nullptr};" << std::endl;
  }
  for (auto &output : this->outputs) {
    result << "    " << output.AsArg() << "{nullptr};" << std::endl;
  }
  result << std::endl;
  auto ub_tensor = this->tpipe.GetTensor(this->tpipe.cube_output_tensor_id);
  GE_CHK_BOOL_RET_STATUS(ub_tensor != nullptr, ge::FAILED, "Codegen CV Fusion MatmulOutput UB tensor id[%ld] "
                         "not found", this->tpipe.cube_output_tensor_id);
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(ub_tensor->dtype, dtype_name), "data type:%d failed",
                    static_cast<int32_t>(ub_tensor->dtype));
  std::string tmp;
  GE_CHK_STATUS_RET(this->tpipe.LocalTensorDefine(tmp), "Local tbuf define failed");
  result << tmp;
  result << "    TPipe tpipe;" << std::endl << std::endl;
  for (auto &[id, que] : this->tpipe.ques) {
    if (id == this->tpipe.cube_output_que_id) {
      continue;
    }
    result << "    " << que.Define() << std::endl;
    result << "    " << que.buf.AsArg() << ";" << std::endl;
  }
  result << std::endl;
  for (auto &pair : this->tpipe.bufs) {
    auto &buf = pair.second;
    result << "    " << buf.Define() << std::endl;
    result << "    " << buf.buf.AsArg() << ";" << std::endl;
    if (buf.tmp_buf_reuse) {
      result << "    " << this->tpipe.tmp_buf.AsArg() << "_" << to_string(buf.id) << ";" << std::endl;  
    }
  }
  result << "    TBuf<TPosition::VECCALC> buf_cube;" << std::endl;

  if (!this->pre_api_extract_dup.empty()) {
    result << this->tpipe.GenDuplicateBufDefine(this->pre_api_extract_dup) << std::endl;
  }

  GE_CHK_STATUS_RET(this->tpipe.BlkTensorDefine(tmp), "Block tensor define failed");
  result << tmp;

  GE_CHK_STATUS_RET(this->GlobalTensorDefine(tmp), "Global tensor define in cv-ub-fuse case failed");
  result << tmp;

  result << this->tpipe.TensorSizeDefine() << std::endl;
  result << "    LocalTensor<" << dtype_name << "> cLocal_;" << std::endl;

  result << "__aicore__ inline void Init(Params const& params, AscendC::LocalTensor<" << dtype_name
         << ">& cLocal, int64_t l1M, int64_t l1NAlign, int64_t ubOffset, int64_t &stage_size_type) {";
  result << std::endl;
  result << "GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tmpTilingData, tmpTilingGM);" << std::endl;
  result << "const int32_t ub_align_value = 32 / sizeof(" << dtype_name << ");" << std::endl;
  result << "const int32_t basen_align = (tmpTilingData.baseN + ub_align_value - 1) / ub_align_value * ub_align_value;"
         << std::endl;
  result << "const int32_t basen_basem_align = (tmpTilingData.baseM * basen_align * sizeof(" << dtype_name
         << ")) / 2 + basen_align * sizeof(" << dtype_name << ");" << std::endl;
  result << "AutofuseTilingData autofuse_tiling_size;" << std::endl;
  // 下面的144为16*16/2+16，按照cube tiling最小块16*16计算得到
  result << "int32_t stage_size = autofuse_tiling_size.STAGE_SIZE_NAME > 144 ? " << std::endl;
  result << "autofuse_tiling_size.STAGE_SIZE_NAME * sizeof(" << dtype_name
         << ") : basen_basem_align;" << std::endl;
  result << "stage_size_type = static_cast<int64_t>(autofuse_tiling_size.STAGE_SIZE_NAME > 144 ? " << std::endl;
  result << "autofuse_tiling_size.STAGE_SIZE_NAME : basen_basem_align / sizeof(" << dtype_name << "));" << std::endl;

  GE_CHK_STATUS_RET(this->root_loop.ActualSizeDefine(this->tiler, this->tpipe, tmp), "actual size define failed");
  result << tmp;

  result << ub_tensor->Str() << "_actual_size =  stage_size_type;" << std::endl << std::endl;
 
  GE_CHK_STATUS_RET(this->tpipe.TensorSizeAssign(dtype_name, tmp), "Tensor size assign failed");
  result << tmp;

  for (auto &input : this->inputs) {
    result << input << " = params." << input << ";" << std::endl;
  }
  for (auto &output : this->outputs) {
    result << output << " = params." << output << ";" << std::endl;
  }
  GE_CHK_STATUS_RET(this->GlobalTensorAssign(tmp), "Global tensor assign in cv-ub-fuse case failed");
  result << tmp;
  result << "tpipe.InitBuffer(buf_cube, basen_basem_align);" << std::endl;
  result << ub_tensor->name << " = buf_cube.Get<" << dtype_name << ">();" << std::endl;
  result << "cLocal = " << ub_tensor->name << ";" << std::endl << std::endl;
  result << "cLocal_ = " << ub_tensor->name << ";" << std::endl << std::endl;

  GE_CHK_STATUS_RET(this->tpipe.LocalTBufAllocLoopTwice(tmp, false), "Local tbuf define failed");
  result << tmp;

  if (!this->pre_api_extract_dup.empty()) {
    result << this->tpipe.GenDuplicateBufAssign(this->pre_api_extract_dup) << std::endl;
  }

  GE_CHK_STATUS_RET(this->tpipe.BlkTensorAssign(tmp), "Block tensor assign failed");
  result << tmp;

  GE_CHK_STATUS_RET(this->tpipe.LocalTQueAlloc(tmp), "Codegen alloc local tque failed");
  result << tmp;
  result << "}" << std::endl << std::endl;

  stringstream ss;
  GE_ASSERT_SUCCESS(this->GenerateSubGraphFuncDef(&(this->root_loop), ss));
  result << ss.str() << std::endl;

  result << "inline __aicore__ void auto_fusion_vector_stage1(int64_t offset, int64_t curAivM, int64_t curAivN, "
            "int64_t shapeN, int64_t curAlignN, int64_t stageSize) {";
  result << std::endl;
  GE_CHK_STATUS_RET(this->root_loop.Generate(this->tiler, this->tpipe, tmp, ComputeStage::kCVFuseStage1),
                    "Codegen root loop Generate failed");
  result << tmp;
  result << "}" << std::endl;

  result << "inline __aicore__ void auto_fusion_vector_stage2(int64_t offset, int64_t curAivM, int64_t curAivN, "
            "int64_t shapeN, int64_t curAlignN, int64_t stageSize) {";
  result << std::endl;
  GE_CHK_STATUS_RET(this->root_loop.Generate(this->tiler, this->tpipe, tmp, ComputeStage::kCVFuseStage2),
                    "Codegen root loop Generate failed");
  result << tmp;
  result << "}" << std::endl;

  result << "inline __aicore__ void operator()(int64_t offset, int64_t curAivM, int64_t curAivN, int64_t shapeN, "
            "int64_t curAlignN, int64_t stageSize, int64_t stageOffset, uint8_t stage = 0) {"
         << std::endl
         << ub_tensor->name << " = cLocal_[stageOffset].template ReinterpretCast<" << dtype_name << ">();" << std::endl
         << "if (stage == 1) {" << std::endl
         << "  auto_fusion_vector_stage1(offset, curAivM, curAivN, shapeN, curAlignN, stageSize);" << std::endl
         << "} else if (stage == 2) {" << std::endl
         << "  auto_fusion_vector_stage2(offset, curAivM, curAivN, shapeN, curAlignN, stageSize);" << std::endl
         << "} else {" << std::endl
         << "  auto_fusion_vector_stage1(offset, curAivM, curAivN, shapeN, curAlignN, stageSize);" << std::endl
         << "  auto_fusion_vector_stage2(offset, curAivM, curAivN, shapeN, curAlignN, stageSize);" << std::endl
         << "}" << std::endl
         << "}" << std::endl;

  result << "};" << std::endl;
  result << "#endif" << std::endl; // CV_UB_NO_DB/CV_UB_DB
  return ge::SUCCESS;
}

Status Kernel::InitCVFusionAddr(std::stringstream &result, bool vector_no_db_flag) {
  if (vector_no_db_flag) {
    result << "  AutoFusionVector::Params CV_FUSION_ADDR;\n";
    for (auto input : this->inputs) {
      result << "  CV_FUSION_ADDR." << input.Str() << " = " << input.Str() << ";" << std::endl;
    }
    size_t output_idx = 0;
    for (auto output : this->outputs) {
      result << "  CV_FUSION_ADDR." << output.Str() << " = " << output.Str() << ";" << std::endl;
      result << "  GM_ADDR output_" << output_idx++ << " = " << output.Str() << ";" << std::endl;
    }
  }
  return ge::SUCCESS;
}

static std::string GetScheduledResultInputOutput(const ascir::FusedScheduledResult &fused_schedule_result,
                                          bool is_kernel_func_call) {
  std::stringstream ss;
  for (size_t i = 0U; i < fused_schedule_result.input_nodes.size(); i++) {
    ss << (is_kernel_func_call ? "(uint8_t*)" : "void* ") << "input" << i << ", ";
  }
  int32_t index = 0;
  for (const auto &node : fused_schedule_result.output_nodes) {
    if (IsOps<Output>(node)) {
      ss << (is_kernel_func_call ? "(uint8_t*)" : "void* ") << "output" << index++ << ", ";
    }
  }
  return ss.str();
}

std::string Kernel::GenKernelFuncCallForInductor(const ascir::FusedScheduledResult &fused_schedule_result) {
  std::string tiling_data_name = "AutofuseTilingData";
  std::string graph_name = CamelToLowerSneak(GenValidName(fused_schedule_result.fused_graph_name.GetString()));
  std::string extern_c = "extern \"C\"";
  std::stringstream ss;

  // 适配AscendC新的单算子编译工程
  ss << "void init_" << graph_name << "(void) {}" << std::endl;
  ss << extern_c << " int64_t AutofuseLaunch(uint32_t blockDim, void* stream, ";
  ss << GetScheduledResultInputOutput(fused_schedule_result, false);
  ss << "void* workspace, " << tiling_data_name << "* tiling_data)" << std::endl;
  ss << "{" << std::endl;
  ss << "  " << graph_name << "<<<blockDim, nullptr, stream>>>(";
  ss << GetScheduledResultInputOutput(fused_schedule_result, true);
  ss << "(uint8_t*)workspace, *tiling_data);" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;

  // 二级指针方式 launch
  ss << extern_c << " uint32_t AutofuseLaunchV2"
     << "(uint32_t blockDim, void* stream, void** input_data, int32_t input_num, void** output_data, int32_t output_num"
     << ", void* workspace, void* tiling_data)" << std::endl;
  ss << "{" << std::endl;
  ss << "  " << graph_name << "<<<blockDim, nullptr, stream>>>(";
  int32_t index = 0;
  for (auto input : fused_schedule_result.input_nodes) {
    ss << "(uint8_t*)input_data[" << index++ << "], ";
  }
  index = 0;
  for (auto node : fused_schedule_result.output_nodes) {
    if (IsOps<Output>(node)) {
      ss << "(uint8_t*)output_data[" << index++ << "], ";
    }
  }
  ss << "(uint8_t*)workspace, ";
  ss << "*(" << tiling_data_name << "*)tiling_data);" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

static ApiCallRegister<ApiCall> register_api_call("ApiCall");

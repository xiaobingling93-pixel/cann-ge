/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include <thread>
#include <unistd.h>
#include <ctime>
#include <chrono>
#include "common/fe_context_utils.h"
#include "common/configuration.h"
#include "common/platform_utils.h"
#include "common/fe_inner_attr_define.h"
#include "common/range_format_transfer/transfer_range_according_to_format.h"
#include "common/unknown_shape_util.h"
#include "common/scope_allocator.h"
#include "common/fe_error_code.h"
#include "common/fe_type_utils.h"
#include "common/fe_report_error.h"

#include "common/graph/fe_graph_utils.h"
#include "ge/ge_api_types.h"
#include "graph/tuning_utils.h"
#include "graph/ge_context.h"
#include "common/ffts_plus_type.h"
#include "common/tile_fwk_op_info.h"
#include "ops_store/ops_kernel_manager.h"
#include "framework/common/ge_types.h"
#include "trace_handle_manager/trace_handle_manager.h"
#include "trace_handle_manager/trace_msg/compile_process_trace_msg.h"
#include "trace_handle_manager/trace_msg/long_time_trace_msg.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "graph/utils/op_type_utils.h"
#include "exe_graph/lowering/exe_res_generation_ctx_builder.h"
#include "graph/ascend_string.h"

namespace fe {
namespace {
constexpr char const *kTbeSoName = "libte_fusion.so";
constexpr char const *kOpCompileSoName = "libop_compile_adapter.so";
constexpr char const *kHiddenSize = "hidden_size";
constexpr char const *kInputSize = "input_size";
constexpr char const *kStateSize = "state_size";
constexpr char const *kShapeNotSupport = "The shape is not support now";
constexpr char const *kOnlyFusionCheck = "_only_fusion_check";
constexpr char const *kJsonFilePath = "json_file_path";
constexpr char const *kBinFilePath = "bin_file_path";
constexpr char const *kJsonValuePtr = "json_value_ptr";
constexpr char const *kBinValuePtr = "bin_value_ptr";
constexpr const char* kOpDebugConfigTe = "op_debug_config_te";
constexpr const char* kEnableSuperkernelPlus = "enable_superkernel_plus";
constexpr const char* kTileFwkOpFlag = "tileFwkOp";
constexpr int64_t kThreadSleepCost = 25;
constexpr uint32_t kMaxPathSize = 1024U;
const std::string kStageGeneralizeGraph = "[GraphOptimizePrepare][ShapeAndValueGeneralize][GeneralizeGraph]";

const std::map<CompileStrategy, std::string> kCompileStrategyStrMap {
        {CompileStrategy::COMPILE_STRATEGY_KEEP_OPTUNE, "set by fe: keep compiling in op tune"},
        {CompileStrategy::COMPILE_STRATEGY_NO_TUNE, "NoTune"},
};

const std::map<RangeLimitType, bool> kRangeLimitBoolMap {
    {RangeLimitType::LIMITED, true},
    {RangeLimitType::UNLIMITED, false}
};

std::string GetCoreType(const std::string &engine_name) {
  if (engine_name == VECTOR_CORE_NAME || FEContextUtils::GetCoreType() == VECTOR_CORE_TYPE) {
    return VECTOR_CORE_TYPE;
  }
  return AI_CORE_TYPE;
}

void RestoreDataType(const ge::OpDescPtr &op_desc,
    std::pair<std::vector<size_t>, std::vector<size_t>> &in_out_changed_idx_vec) {
  if (!in_out_changed_idx_vec.first.empty()) {
    for (auto &input_idx : in_out_changed_idx_vec.first) {
      auto input_desc = op_desc->MutableInputDesc(input_idx);
      if(input_desc != nullptr) {
        input_desc->SetDataType(ge::DT_FLOAT);
      }
    }
  }
  if (!in_out_changed_idx_vec.second.empty()) {
    for (auto &output_idx : in_out_changed_idx_vec.second) {
      auto output_desc = op_desc->MutableOutputDesc(output_idx);
      FE_CHECK(output_desc == nullptr, REPORT_FE_ERROR("output_desc is nullptr"), return);
      output_desc->SetDataType(ge::DT_FLOAT);
    }
  }
}

void CalcSliceShapeByRange(const std::vector<ffts::DimRange> &dim_range, ge::GeShape &slice_shape) {
  vector<int64_t> dims;
  for (auto &range : dim_range) {
    dims.emplace_back(range.higher - range.lower);
  }
  slice_shape = ge::GeShape(dims);
}

void SetSgtSliceShaeForEachTensor(size_t tensor_idx, int32_t thread_idx, const ge::Node *node,
                                  const ge::OpDesc::Vistor<ge::GeTensorDescPtr> &tensors,
                                  const vector<vector<vector<ffts::DimRange>>> &slice_info,
                                  const vector<vector<vector<int64_t>>> &ori_slice_shape, const string &attr_name) {

  vector<vector<int64_t>> slice_dims_head_tail;
  /* The shape is an array in which the first one is head slice shape and
   * the second one is the tail slice shape */
  (void)ge::AttrUtils::GetListListInt(tensors.at(tensor_idx), attr_name, slice_dims_head_tail);

  ge::GeShape slice_shape;

  if (attr_name == ATTR_NAME_SGT_SLICE_SHAPE) {
    if (slice_info.empty()) {
      FE_LOGD("Slice info is empty.");
      return;
    }
    CalcSliceShapeByRange(slice_info[thread_idx][tensor_idx], slice_shape);
  } else {
    if (ori_slice_shape.empty()) {
      FE_LOGD("Slice info is empty.");
      return;
    }
    slice_shape = ge::GeShape(ori_slice_shape[thread_idx][tensor_idx]);
  }

  slice_dims_head_tail.emplace_back(slice_shape.GetDims());
  auto tensor = tensors.at(tensor_idx);
  (void)ge::AttrUtils::SetListListInt(tensor, attr_name, slice_dims_head_tail);
  FE_LOGD("Optype:%s, opname:%s, set thread %d's slice shape %s for tensor %s, tensor index %zu.",
          node->GetType().c_str(), node->GetName().c_str(), thread_idx,
          StringUtils::IntegerVecToString(slice_shape.GetDims()).c_str(),
          tensor->GetName().c_str(), tensor_idx);
  FE_LOGD("Original shape is %s, shape is %s.",
          StringUtils::IntegerVecToString(tensor->GetOriginShape().GetDims()).c_str(),
          StringUtils::IntegerVecToString(tensor->MutableShape().GetDims()).c_str());
}

void ClearSgtAttr(std::vector<ge::Node *> &nodes) {
  for (auto node : nodes) {
    auto op_desc = node->GetOpDesc();
    size_t input_size = op_desc->GetAllInputsSize();
    for (size_t i = 0; i < input_size; i++) {
      auto input_desc =  op_desc->MutableInputDesc(i);
      if (input_desc == nullptr) {
        continue;
      }
      vector<vector<int64_t>> sgt_slice;
      vector<vector<int64_t>> empty_sgt_slice;
      if (ge::AttrUtils::GetListListInt(input_desc, ATTR_NAME_SGT_SLICE_SHAPE, sgt_slice)) {
        (void)ge::AttrUtils::SetListListInt(input_desc, ATTR_NAME_SGT_SLICE_SHAPE, empty_sgt_slice);
      }
    }

    size_t output_size = op_desc->GetAllOutputsDescSize();
    for (size_t i = 0; i < output_size; i++) {
      auto output_desc =  op_desc->MutableOutputDesc(i);
      if (output_desc == nullptr) {
        continue;
      }
      vector<vector<int64_t>> empty_sgt_slice;
      if (ge::AttrUtils::HasAttr(output_desc, ATTR_NAME_SGT_SLICE_SHAPE)) {
        (void)ge::AttrUtils::SetListListInt(output_desc, ATTR_NAME_SGT_SLICE_SHAPE, empty_sgt_slice);
      }
    }
  }
}

void SetThreadNodeName(const std::vector<ge::Node *> &nodes, vector<string> &old_names, const int32_t &i) {
  for (size_t j = 0; j < nodes.size(); j++) {
    string old_name = nodes[j]->GetOpDesc()->GetName();
    old_names.push_back(old_name);
    nodes[j]->GetOpDesc()->SetName(old_name + "_thread_" + to_string(i));
  }
}

void SetNameForNodes(const vector<ge::Node *> &nodes,
                     const vector<string> &names) {
  for (size_t i = 0; i < nodes.size(); i++) {
    nodes[i]->GetOpDesc()->SetName(names[i]);
  }
}

bool CheckIsInnerOpStore(const FEOpsStoreInfo &ops_store) {
  if (ops_store.op_impl_type == EN_IMPL_HW_TBE || ops_store.op_impl_type == EN_IMPL_VECTOR_CORE_HW_TBE) {
    return true;
  }
  return false;
}
} // namespace

TbeOpStoreAdapter::TbeOpStoreAdapter(const std::string &engine_name) : engine_name_(engine_name) {}

Status TbeOpStoreAdapter::SerialPreCompileOp(vector<PreCompileNodePara> &compile_para_vec) {
  for (auto &comp_para : compile_para_vec) {
    FE_CHECK(comp_para.node == nullptr,
             REPORT_FE_ERROR("[SubGraphOpt][Compile][SerialPreComOp] compPara.node is nullptr."),
             return FAILED);
    FE_LOGD("TbeOpStoreAdapter::PreCompile Op begin, node name: %s, node type %s.",
            comp_para.node->GetOpDesc()->GetName().c_str(), comp_para.node->GetOpDesc()->GetType().c_str());

    TbeOpInfoPtr tbe_op_info_ptr = PreCompSetTbeOpInfo(comp_para);
    if (tbe_op_info_ptr == nullptr) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][SerialPreComOp] Setting TbeOpInfo Failed.");
      return FAILED;
    }

    string op_pattern_before_buff_fus;
    bool need_precompile_node = false;
    (void)ge::AttrUtils::GetBool(comp_para.node->GetOpDesc(), NEED_RE_PRECOMPILE, need_precompile_node);
    if (need_precompile_node) {
      if (!ge::AttrUtils::GetStr(comp_para.node->GetOpDesc(), kOpPattern, op_pattern_before_buff_fus)) {
        FE_LOGW("Can't get BuffFus op %s pattern before precompile.", comp_para.node->GetName().c_str());
      }
    }
    if (!Configuration::Instance(AI_CORE_NAME).IsEnableUbFusion()) {
      tbe_op_info_ptr->SetPattern(kInvalidPattern);
    }
    FE_CHECK(PreBuildTbeOp == nullptr,
             REPORT_FE_ERROR("[SubGraphOpt][Compile][SerialPreComOp] PreBuildTbeOp is nullptr."), return FAILED);

    FE_TIMECOST_START(PreBuild);
    // call pre-compile func, and return pattern of op, such as reduction,
    bool result = PreBuildTbeOp(*tbe_op_info_ptr, 0, 0);
    if (!result) {
      ErrorMessageDetail error_msg(EM_COMPLIE_FAILED,
          {comp_para.node->GetOpDesc()->GetName(), comp_para.node->GetOpDesc()->GetType()});
      ReportErrorMessage(error_msg);
      REPORT_FE_ERROR("[SubGraphOpt][Compile][SerialPreComOp] Failed to pre-build Tbe op.");
      return FAILED;
    }
    FE_TIMECOST_END(PreBuild, "PreBuildTbe during FEGraphOptimizer::OptimizeFusedGraph");

    if (SetPreCompilePattern(comp_para.node->GetOpDesc(), *tbe_op_info_ptr, comp_para.op_kernel_info_ptr) == FAILED) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][SerialPreComOp] %s setting op pattern failed.",
                      comp_para.node->GetName().c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

Status TbeOpStoreAdapter::SetPreCompilePattern(ge::OpDescPtr op_desc, te::TbeOpInfo &op_info,
                                               const OpKernelInfoPtr &op_kernel_info_ptr) const {
  string op_pattern = op_info.GetPattern();
  if (op_pattern.empty()) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][SetPtn] %s's pattern is empty", op_desc->GetName().c_str());
    return FAILED;
  }

  // set op pattern to op's desc
  op_pattern = (op_desc->GetType() == CAST) ? kInvalidPattern : op_pattern;
  FE_LOGD("Node[%s]: the pattern after precompile is %s.", op_desc->GetName().c_str(), op_pattern.c_str());
  if (!ge::AttrUtils::SetStr(op_desc, op_desc->GetName() + kOpPattern, op_pattern) ||
      !ge::AttrUtils::SetStr(op_desc, kOpPattern, op_pattern)) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][SetPtn] %s set pattern attr failed.", op_desc->GetName().c_str());
      return FAILED;
  }

  FE_LOGD("After pre compile, core type and engine type of op[%s, %s] are [%s] and [%s].",
          op_desc->GetNamePtr(), op_desc->GetTypePtr(),
          op_info.GetOpCoreType().c_str(), op_info.GetEngineType().c_str());
  if (op_info.GetOpCoreType() == "Default") {
    string core_type_str;
    if (op_info.GetEngineType() == VECTOR_CORE_TYPE) {
      core_type_str = VECTOR_CORE_TYPE;
    } else {
      FE_CHECK_NOTNULL(op_kernel_info_ptr);
      if (op_kernel_info_ptr->GetCoreType().empty()) {
        core_type_str = (PlatformUtils::Instance().GetCubeVecState() == CubeVecStateNew::CUBE_VEC_SPLIT)
                        ? VECTOR_CORE_TYPE : AI_CORE_TYPE;
      } else {
        core_type_str = op_kernel_info_ptr->GetCoreType();
      }
    }
    op_info.SetOpCoreType(core_type_str);
    FE_LOGD("Fix core type of op[%s, %s] to [%s] for origin core type is Default.",
            op_desc->GetNamePtr(), op_desc->GetTypePtr(), core_type_str.c_str());
  }

  (void)ge::AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, op_info.GetOpCoreType());
  FE_LOGD("Node[%s, %s]: do pre-compile successfully. Pattern is %s and core type is %s.",
          op_desc->GetNamePtr(), op_desc->GetTypePtr(), op_pattern.c_str(),
          op_info.GetOpCoreType().c_str());
  return SUCCESS;
}

Status TbeOpStoreAdapter::ProcessFailPreCompTask(CompileTaskPara &task_para) const {
  for (auto &fin_task_pair : task_para.failed_tasks) {
    auto task_id = fin_task_pair.first;
    auto task_iter = task_para.task_node_map.find(task_id);
    if (task_iter == task_para.task_node_map.end()) {
      REPORT_FE_ERROR("[SubGraphOpt][Pre-Comp] thread[%lu], not find task[%lu].", GetCurThreadId(), task_id);
      return FAILED;
    }

    ge::Node *node = task_para.task_node_map[task_id];
    REPORT_FE_ERROR("[SubGraphOpt][Pre-Comp][Node %s] Failed to pre-compile. Tid is [%lu], TaskId is [%lu].",
                    node->GetName().c_str(), GetCurThreadId(), task_id);
  }

  if (!task_para.failed_tasks.empty()) {
    FE_LOGD("Failed to process task_num[%zu], tid[%lu].", task_para.failed_tasks.size(), GetCurThreadId());
    return FAILED;
  }

  return SUCCESS;
}

Status TbeOpStoreAdapter::ProcessSuccPreCompTask(CompileTaskPara &task_para) const {
  for (auto &fin_task_pair : task_para.succ_tasks) {
    auto task_id = fin_task_pair.first;
    auto task_iter = task_para.task_node_map.find(task_id);
    if (task_iter == task_para.task_node_map.end()) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][ProSucTask] Thread[%lu], not find task[%lu]", GetCurThreadId(),
                      task_id);
      return FAILED;
    }

    auto task_kernel_iter = task_para.task_kernel_info_map.find(task_id);
    if (task_kernel_iter == task_para.task_kernel_info_map.end()) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][ProSucTask] Thread[%lu], not find kernel info ptr for task[%lu]",
                      GetCurThreadId(), task_id);
      return FAILED;
    }

    ge::Node *node = task_para.task_node_map[task_id];
    TbeOpInfoPtr tbe_op_info_ptr = task_para.task_tbe_info_map[task_id];
    FE_LOGD("Thread[%lu], get task[%lu], node[%s].", GetCurThreadId(), task_id, node->GetName().c_str());

    string op_pattern_before_buff_fus;
    bool need_precompile_node = false;
    (void)ge::AttrUtils::GetBool(node->GetOpDesc(), NEED_RE_PRECOMPILE, need_precompile_node);
    if (need_precompile_node) {
      if (!ge::AttrUtils::GetStr(node->GetOpDesc(), kOpPattern, op_pattern_before_buff_fus)) {
        FE_LOGW("Can't get buff_fus op[%s] pattern before precompile.", node->GetName().c_str());
      }
    }

    if (SetPreCompilePattern(node->GetOpDesc(), *tbe_op_info_ptr, task_kernel_iter->second) == FAILED) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][ProSucTask] %s set op pattern failed", node->GetName().c_str());
      return FAILED;
    }
  }

  FE_LOGD("Process succeeded with task_num[%zu]. tid[%lu].", task_para.succ_tasks.size(), GetCurThreadId());
  return SUCCESS;
}

TbeOpInfoPtr TbeOpStoreAdapter::PreCompSetTbeOpInfo(PreCompileNodePara &comp_para) {
  if (comp_para.op_dsl_file_path.empty()) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][PreCompSetInfo] Op dsl path is invalid.");
    return nullptr;
  }

  auto op_desc = comp_para.node->GetOpDesc();
  string op_name = op_desc->GetName();
  if (!comp_para.session_graph_id.empty()) {
    op_name = comp_para.session_graph_id + "_" + op_desc->GetName();
  }

  string opFuncName = op_desc->GetType();
  TbeOpInfoPtr tbe_op_info_ptr;
  string engine_name = op_desc->GetOpEngineName();
  FE_MAKE_SHARED(tbe_op_info_ptr = std::make_shared<te::TbeOpInfo>(op_name, comp_para.op_dsl_file_path, opFuncName,
                                                                   GetCoreType(engine_name)), return nullptr);
  tbe_op_info_ptr->SetRealName(op_desc->GetName());
  GetAndSetOpsPathNamePrefix(comp_para.op_kernel_info_ptr, *tbe_op_info_ptr);

  bool is_dynamic_impl = IsOpDynamicImpl(op_desc);
  tbe_op_info_ptr->SetDynamicImpl(is_dynamic_impl);

  if (op_desc->HasAttr(ge::ATTR_NAME_UNREGST_OPPATH)) {
    if (tbe_single_op_info_assembler_ptr_->AssembleSingleTbeInfo(comp_para.node, *tbe_op_info_ptr, engine_name) !=
        SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][PreCompSetInfo] AssembleTbeInfo failed.");
      return nullptr;
    }
  } else if (ge::OpTypeUtils::IsAutofuseNode(op_desc)) {
    if (tbe_info_assembler_ptr_->AssembleAutoFuseTbeInfo(comp_para.node, *tbe_op_info_ptr) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][PreCompSetInfo] AssembleAutoFuseTbeInfo failed.");
      return nullptr;
    }
  } else {
    if (tbe_info_assembler_ptr_->AssembleTbeInfo(comp_para.node, comp_para.op_kernel_info_ptr, *tbe_op_info_ptr,
                                                 engine_name) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][PreCompSetInfo] AssembleTbeInfo failed.");
      return nullptr;
    }
    SetOpSpecificInfoToTbeOpInfo(comp_para.op_kernel_info_ptr, *tbe_op_info_ptr);
  }

  if (UnknownShapeUtils::IsUnknownShapeOp(*comp_para.node->GetOpDesc()) || IsStcToDynSoftSyncOp(op_desc)) {
    tbe_op_info_ptr->SetIsUnknownShape(true);
  }
  string op_slice_info_str;
  te::LX_QUERY_STATUS status = GetOpInfo(*tbe_op_info_ptr, op_slice_info_str);
  if (status == te::LX_QUERY_SUCC) {
    (void)ge::AttrUtils::SetStr(comp_para.node->GetOpDesc(), OP_SLICE_INFO, op_slice_info_str);
    FE_LOGD("Obtain slice info %s from tbe api for node[%s].", op_slice_info_str.c_str(),
            comp_para.node->GetName().c_str());
  } else {
    FE_LOGD("Not obtain slice info from tbe api for node[%s].", comp_para.node->GetName().c_str());
  }

  // set custom flag to node
  SetOpDescCustomOp(comp_para.node->GetOpDesc());
  tbe_op_info_ptr->SetNode(comp_para.node->shared_from_this());

  return tbe_op_info_ptr;
}
/*
 *  @ingroup fe
 *  @brief   pre-compile and return pattern of op
 *  @param   [in]  node        node pointer
 *  @param   [in]  info_store   op info store pointer
 *  @param   [in] imply_type_str  op imply type
 *  @param   [in] op_dsl_file_path  python DSL file for op
 *  @return  SUCCESS or FAILED
 */
Status TbeOpStoreAdapter::PreCompileOp(vector<PreCompileNodePara> &compile_para_vec) {
  if (!support_parallel_compile) {
    return SerialPreCompileOp(compile_para_vec);
  } else {
    return ParallelPreCompileOp(compile_para_vec);
  }
}

Status TbeOpStoreAdapter::SetTensorDescShape(const ge::OpDescPtr &op_desc,
                                             const ge::GeTensorDescPtr &cur_tensor_desc) const {
  const ge::GeShape &ori_shape = cur_tensor_desc->GetOriginShape();
  ge::GeShape new_shape;
  ShapeAndFormat shape_and_format_info = {ori_shape,
                                          new_shape,
                                          cur_tensor_desc->GetOriginFormat(),
                                          static_cast<ge::Format>(ge::GetPrimaryFormat(cur_tensor_desc->GetFormat())),
                                          cur_tensor_desc->GetDataType(),
                                          static_cast<int64_t>(OpImplType::EN_IMPL_HW_TBE)};
  int64_t hidden_size = 1;
  int64_t input_size = 1;
  int64_t state_size = -1;
  (void)ge::AttrUtils::GetInt(op_desc, kHiddenSize, hidden_size);
  (void)ge::AttrUtils::GetInt(op_desc, kInputSize, input_size);
  (void)ge::AttrUtils::GetInt(op_desc, kStateSize, state_size);
  shape_and_format_info.extra_attr.input_size = input_size;
  shape_and_format_info.extra_attr.hidden_size = hidden_size;
  shape_and_format_info.extra_attr.state_size = state_size;
  Status ret = GetShapeAccordingToFormat(shape_and_format_info);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[PreComp][GetShape] Failed to get shape. old format is %s, new format is %s",
                    ge::TypeUtils::FormatToSerialString(shape_and_format_info.old_format).c_str(),
                    ge::TypeUtils::FormatToSerialString(shape_and_format_info.new_format).c_str());
    return FAILED;
  }
  auto ori_shape_vec = ori_shape.GetDims();
  auto shape_vec = new_shape.GetDims();
  (void)ge::AttrUtils::SetListInt(cur_tensor_desc, kSoftSyncDynOriShape, ori_shape_vec);
  (void)ge::AttrUtils::SetListInt(cur_tensor_desc, kSoftSyncDynShape, shape_vec);
  FE_LOGD("Set op[name:%s, type:%s] shape, ori_shape: %s, new shape: %s, old format: %s, new format: %s.",
          op_desc->GetName().c_str(), op_desc->GetType().c_str(),
          GetShapeDims(ori_shape).c_str(), GetShapeDims(new_shape).c_str(),
          ge::TypeUtils::FormatToSerialString(shape_and_format_info.old_format).c_str(),
          ge::TypeUtils::FormatToSerialString(shape_and_format_info.new_format).c_str());
  return SUCCESS;
}

Status TbeOpStoreAdapter::SetTensorDescRange(const ge::OpDescPtr &op_desc,
                                             const ge::GeTensorDescPtr &cur_tensor_desc) const {
  vector<std::pair<int64_t, int64_t>> new_range_shape;
  vector<std::pair<int64_t, int64_t>> ori_shape_range = GetOriginShapeRange(*cur_tensor_desc.get());
  vector<std::pair<int64_t, int64_t>> old_shape_range = GetAlignShapeRange(ori_shape_range,
                                                                           cur_tensor_desc->GetOriginShape());
  RangeAndFormat range_and_format_info = {cur_tensor_desc->GetOriginShape(),
                                          old_shape_range,
                                          new_range_shape,
                                          cur_tensor_desc->GetOriginFormat(),
                                          static_cast<ge::Format>(ge::GetPrimaryFormat(cur_tensor_desc->GetFormat())),
                                          cur_tensor_desc->GetDataType(),
                                          static_cast<int64_t>(OpImplType::EN_IMPL_HW_TBE)};
  int64_t hidden_size = 1;
  int64_t input_size = 1;
  int64_t state_size = -1;
  (void)ge::AttrUtils::GetInt(op_desc, kHiddenSize, hidden_size);
  (void)ge::AttrUtils::GetInt(op_desc, kInputSize, input_size);
  (void)ge::AttrUtils::GetInt(op_desc, kStateSize, state_size);
  range_and_format_info.extra_attr.input_size = input_size;
  range_and_format_info.extra_attr.hidden_size = hidden_size;
  range_and_format_info.extra_attr.state_size = state_size;
  Status ret = RangeTransferAccordingToFormat::GetRangeAccordingToFormat(range_and_format_info);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[PreComp][GetShpRange] Failed to get shape range. old format is %s, new format is %s",
                    ge::TypeUtils::FormatToSerialString(range_and_format_info.old_format).c_str(),
                    ge::TypeUtils::FormatToSerialString(range_and_format_info.new_format).c_str());
    return FAILED;
  }
  std::vector<std::pair<int64_t, int64_t>> &new_shape_range = range_and_format_info.new_range;
  if (cur_tensor_desc->SetShapeRange(new_shape_range) != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("[PreComp][SetShpRange] Set shape range of op[name:%s,type:%s] failed.",
                    op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }
  FE_LOGD("Set shape range to op[name:%s,type:%s]. old format: %u, new format: %u. old range: %s, new range: %s.",
          op_desc->GetName().c_str(), op_desc->GetType().c_str(),
          range_and_format_info.old_format, range_and_format_info.new_format,
          ShapeRangeToStr(range_and_format_info.old_range).c_str(), ShapeRangeToStr(new_shape_range).c_str());
  return SUCCESS;
}

Status TbeOpStoreAdapter::GenerateSingleOpRange(const PreCompileNodePara &comp_para) const {
  ge::NodePtr node_ptr = comp_para.node->shared_from_this();
  bool stc_to_dyn_soft_sync = false;
  (void)ge::AttrUtils::GetBool(node_ptr->GetOpDesc(), kStaticToDynamicSoftSyncOp, stc_to_dyn_soft_sync);
  bool stc_tiling_depend = false;
  (void)ge::AttrUtils::GetBool(node_ptr->GetOpDesc(), kDynamicTilingDependOp, stc_tiling_depend);
  if (stc_to_dyn_soft_sync || stc_tiling_depend) {
    te::TE_GENERALIZE_TYPE generalize_type = te::TE_GENERALIZE_TYPE::DEFAULT_TBE_OP_INFO;
    if (comp_para.op_kernel_info_ptr->GetRangeLimitType() != RangeLimitType::DEFAULT) {
      generalize_type = te::TE_GENERALIZE_TYPE::REGISTER_FUNC;
    }
    ge::OpDescPtr op_desc = node_ptr->GetOpDesc();
    ge::OpDescPtr op_desc_bak = ge::AttrUtils::CloneOpDesc(op_desc);
    TbeOpInfoPtr tbe_op_info_ptr;
    string engine_name = op_desc->GetOpEngineName();
    string op_name = op_desc->GetName();
    string op_func_name = op_desc->GetType();
    FE_MAKE_SHARED(tbe_op_info_ptr =
        std::make_shared<te::TbeOpInfo>(op_name, comp_para.op_dsl_file_path, op_func_name, GetCoreType(engine_name)),
        return FAILED);
    tbe_op_info_ptr->SetRealName(op_name);

    bool is_dynamic_impl = IsOpDynamicImpl(op_desc);
    tbe_op_info_ptr->SetDynamicImpl(is_dynamic_impl);
    if (tbe_info_assembler_ptr_->AssembleTbeInfo(comp_para.node, comp_para.op_kernel_info_ptr, *tbe_op_info_ptr,
                                                 engine_name) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][PreCompile][GenerateSingleOpRange] AssembleTbeInfo failed, node[%s].",
                      op_desc->GetName().c_str());
      return FAILED;
    }
    SetOpSpecificInfoToTbeOpInfo(comp_para.op_kernel_info_ptr, *tbe_op_info_ptr);
    if (TeGeneralize(*tbe_op_info_ptr.get(), generalize_type, node_ptr)) {
      auto all_input_desc = op_desc->GetAllInputsDescPtr();
      auto all_input_desc_bak = op_desc_bak->GetAllInputsDescPtr();
      for (size_t i = 0; i < all_input_desc.size(); i++) {
        auto cur_tensor_desc = all_input_desc.at(i);
        if (SetTensorDescShape(op_desc, cur_tensor_desc) != SUCCESS) {
          REPORT_FE_ERROR("[PreComp][SetShape] Failed to set input shape for op[name:%s, type:%s], idx: %zu.",
                          op_desc->GetName().c_str(), op_desc->GetType().c_str(), i);
          return FAILED;
        }
        if (SetTensorDescRange(op_desc, cur_tensor_desc) != SUCCESS) {
          REPORT_FE_ERROR("[PreComp][SetShpRange] Failed to set input shape range for op[name:%s,type:%s], index: [%zu].",
                          op_desc->GetName().c_str(), op_desc->GetType().c_str(), i);
          return FAILED;
        }
        cur_tensor_desc->SetOriginShape(all_input_desc_bak.at(i)->GetOriginShape());
      }
      auto all_output_desc = op_desc->GetAllOutputsDescPtr();
      auto all_output_desc_bak = op_desc_bak->GetAllOutputsDescPtr();
      for (size_t i = 0; i < all_output_desc.size(); i++) {
        auto cur_tensor_desc = all_output_desc.at(i);
        if (SetTensorDescShape(op_desc, cur_tensor_desc) != SUCCESS) {
          REPORT_FE_ERROR("[PreComp][SetShape] Failed to set output shape for op[name:%s, type:%s], idx: %zu.",
                          op_desc->GetName().c_str(), op_desc->GetType().c_str(), i);
          return FAILED;
        }
        if (SetTensorDescRange(op_desc, cur_tensor_desc) != SUCCESS) {
          REPORT_FE_ERROR("[PreComp][SetShpRange] Failed to set output shape range of op[name:%s,type:%s], idx: [%zu].",
                          op_desc->GetName().c_str(), op_desc->GetType().c_str(), i);
          return FAILED;
        }
        cur_tensor_desc->SetOriginShape(all_output_desc_bak.at(i)->GetOriginShape());
      }
    } else {
      node_ptr->GetOpDesc()->DelAttr(kStaticToDynamicSoftSyncOp);
      node_ptr->GetOpDesc()->DelAttr(kSoftsyncDynamicImpl);
      node_ptr->GetOpDesc()->DelAttr(ATTR_NAME_SUPPORT_DYNAMIC_SHAPE);
      node_ptr->GetOpDesc()->DelAttr(ATTR_NAME_IS_OP_DYNAMIC_IMPL);
    }
  }
  return SUCCESS;
}

Status TbeOpStoreAdapter::ParallelPreCompileOp(vector<PreCompileNodePara> &compile_para_vec) {
  uint64_t thread_id = GetCurThreadId();
  CompileTaskPara task_para;
  task_para.task_num = 0;
  for (auto &comp_para : compile_para_vec) {
    FE_CHECK(comp_para.node == nullptr, REPORT_FE_ERROR("compPara.node is nullptr"), return FAILED);
    if (GenerateSingleOpRange(comp_para) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt] [Pre-Comp] Gen shape range failed.");
      return FAILED;
    }
    comp_para.tbe_op_info_ptr = PreCompSetTbeOpInfo(comp_para);
    if (comp_para.tbe_op_info_ptr == nullptr) {
      REPORT_FE_ERROR("[SubGraphOpt] [Pre-Comp] Set TbeOpInfo Failed");
      return FAILED;
    }
    if (!Configuration::Instance(AI_CORE_NAME).IsEnableUbFusion()) {
      comp_para.tbe_op_info_ptr->SetPattern(kInvalidPattern);
    }
  }

  for (auto &comp_para : compile_para_vec) {
    te::BUILD_TYPE build_type;
    if (IsFuzzBuild()) {
      build_type = te::FUZZILY_BUILD;
    } else {
      build_type = te::ACCURATELY_BUILD;
    }
    task_para.task_num++;
    comp_para.tbe_op_info_ptr->SetBuildType(build_type);

    uint64_t taskId = GetAtomicId();
    task_para.task_node_map.insert(make_pair(taskId, comp_para.node));
    task_para.task_tbe_info_map.insert(make_pair(taskId, comp_para.tbe_op_info_ptr));
    task_para.task_kernel_info_map.insert(make_pair(taskId, comp_para.op_kernel_info_ptr));

    bool result = PreBuildTbeOp(*comp_para.tbe_op_info_ptr, taskId, thread_id);
    if (!result) {
      // op_name,op_type,graph_id,thread_id,task_id
      ErrorMessageDetail error_msg(EM_COMPLIE_FAILED, {comp_para.node->GetOpDesc()->GetName(),
                                   comp_para.node->GetOpDesc()->GetType()});
      ReportErrorMessage(error_msg);
    }

    if (!result) {
      REPORT_FE_ERROR("[SubGraphOpt][Pre-Comp]Failed to pre-compile node %s. thread id is [%lu], task is [%lu].",
                      comp_para.node->GetName().c_str(), thread_id, taskId);
      return FAILED;
    }
    FE_LOGD("Set precompile task[%s] successfully, tid[%lu], taskId[%lu].", comp_para.node->GetName().c_str(),
            thread_id, taskId);
  }

  FE_LOGD("Thread[%lu], set %lu tasks to pre-compile.", thread_id, task_para.task_num);
  FE_TIMECOST_START(WaitTaskFinish);
  if (WaitTaskFinish(task_para) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][Pre-Comp]Failed to wait thread[%lu]'s task finish.", thread_id);
    return FAILED;
  }
  FE_TIMECOST_END(WaitTaskFinish, "ParallelPreCompileOp.WaitTaskFinish");

  if (ProcessFailPreCompTask(task_para) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Pre-Comp]Failed to process failed task. Thread_id is [%lu].", thread_id);
    return FAILED;
  }

  if (ProcessSuccPreCompTask(task_para) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][Pre-Comp]Failed to process successful task. Thread_id is [%lu].", thread_id);
    return FAILED;
  }

  return SUCCESS;
}

Status TbeOpStoreAdapter::SetTaskFusionTask(const uint64_t thread_id, const int64_t scope_id,
                                            const std::vector<ge::Node *> &nodes, CompileTaskPara &task_para) const {
  uint64_t taskId = GetAtomicId();
  task_para.task_scope_id_map.emplace(taskId, scope_id);
  ge::Node *first_node = nodes[0];
  te::OpBuildResCode ret = TaskFusionFunc(nodes, first_node->GetOpDesc(), taskId, thread_id);
  if (ret != te::OP_BUILD_SUCC) {
    ErrorMessageDetail error_msg(EM_COMPLIE_FAILED, {first_node->GetName(), first_node->GetType()});
    ReportErrorMessage(error_msg);
    REPORT_FE_ERROR("[SubGraphOpt][TaskFusion] Failed to do task fusion for nodes[%s, %s], task id[%lu], thread id[%lu].",
                    first_node->GetName().c_str(), first_node->GetType().c_str(), taskId, thread_id);
    return FAILED;
  }
  FE_LOGD("Set task fusion task for nodes[%s], task id[%lu], thread id[%lu].",
          GetFusionNodesDescStr(nodes).c_str(), taskId, thread_id);
  task_para.task_num++;
  return SUCCESS;
}

Status TbeOpStoreAdapter::TaskFusion(const ScopeNodeIdMap &fusion_nodes_map, CompileResultMap &compile_ret_map) {
  FE_TIMECOST_START(TaskFusion)
  FE_CHECK_NOTNULL(TaskFusionFunc);
  if (fusion_nodes_map.empty()) {
    return SUCCESS;
  }
  uint64_t thread_id = GetCurThreadId();
  CompileTaskPara task_para(0);
  for (const std::pair<const int64_t, std::vector<ge::Node *>> &nodes_pair : fusion_nodes_map) {
    if (nodes_pair.second.empty()) {
      continue;
    }
    if (SetTaskFusionTask(thread_id, nodes_pair.first, nodes_pair.second, task_para) != SUCCESS) {
      return FAILED;
    }
  }

  if (WaitTaskFinish(task_para) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][TaskFusion] Thread[%lu] failed to wait task finish.", thread_id);
    return FAILED;
  }

  if (!task_para.failed_tasks.empty()) {
    for (const auto &task_pair : task_para.failed_tasks) {
      auto task_iter = task_para.task_scope_id_map.find(task_pair.first);
      if (task_iter != task_para.task_scope_id_map.end()) {
        auto nodes_iter = fusion_nodes_map.find(task_iter->second);
        if (nodes_iter != fusion_nodes_map.end()) {
          FE_LOGE("Failed to do task fusion for nodes[%s], scope id[%ld].",
                  GetFusionNodesDescStr(nodes_iter->second).c_str(), task_iter->second);
        }
      }
    }
    REPORT_FE_ERROR("[SubGraphOpt][TaskFusion] Thread[%lu] failed to wait task finish.", thread_id);
    return FAILED;
  }

  for (const auto &task_pair : task_para.succ_tasks) {
    auto task_iter = task_para.task_scope_id_map.find(task_pair.first);
    if (task_iter != task_para.task_scope_id_map.end()) {
      if (SetOpCompileResult(task_iter->second, task_pair.second.teNodeOpDesc, true, compile_ret_map) != SUCCESS) {
        return FAILED;
      }
    }
  }

  FE_TIMECOST_END(TaskFusion, "Task Fusion")
  FE_LOGD("Finish task fusion");
  return SUCCESS;
}

Status TbeOpStoreAdapter::SetSuperKernelTask(const uint64_t thread_id, const int64_t scope_id,
                                             const std::vector<ge::Node *> &nodes, CompileTaskPara &task_para) const {
  uint64_t taskId = GetAtomicId();
  task_para.task_scope_id_map.emplace(taskId, scope_id);
  ge::Node *first_node = nodes[0];
  te::OpBuildResCode ret = BuildSuperKernel(nodes, first_node->GetOpDesc(), taskId, thread_id);
  if (ret != te::OP_BUILD_SUCC) {
    ErrorMessageDetail error_msg(EM_COMPLIE_FAILED, {first_node->GetName(), first_node->GetType()});
    ReportErrorMessage(error_msg);
    REPORT_FE_ERROR("[SPK] Failed to do super kernel compile for nodes[%s, %s], task id[%lu], thread id[%lu].",
    first_node->GetName().c_str(), first_node->GetType().c_str(), taskId, thread_id);
    return FAILED;
  }
  FE_LOGD("Set SuperKernelTask for nodes[%s], task id[%lu], thread id[%lu].",
  GetFusionNodesDescStr(nodes).c_str(), taskId, thread_id);
  task_para.task_num++;
  return SUCCESS;
}

Status TbeOpStoreAdapter::SuperKernelCompile(const ScopeNodeIdMap &fusion_nodes_map,
                                             CompileResultMap &compile_ret_map) {
  FE_TIMECOST_START(SuperKernelCompile)
  FE_CHECK_NOTNULL(BuildSuperKernel);
  if (fusion_nodes_map.empty()) {
    return SUCCESS;
  }
  uint64_t thread_id = GetCurThreadId();
  CompileTaskPara task_para(0);
  for (const std::pair<const int64_t, std::vector<ge::Node *>> &nodes_pair : fusion_nodes_map) {
    if (nodes_pair.second.empty()) {
      continue;
    }
    if (SetSuperKernelTask(thread_id, nodes_pair.first, nodes_pair.second, task_para) != SUCCESS) {
      return FAILED;
    }
  }
  
  if (WaitTaskFinish(task_para) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][SuperKernelCompile] Thread[%lu] failed to wait task finish.", thread_id);
    return FAILED;
  }
  
  if (!task_para.failed_tasks.empty()) {
    REPORT_FE_ERROR("[SubGraphOpt][SuperKernelCompile] Thread[%lu] failed to wait task finish.", thread_id);
    return FAILED;
  }
  
  for (const auto &task_pair : task_para.succ_tasks) {
    auto task_iter = task_para.task_scope_id_map.find(task_pair.first);
    if (task_iter != task_para.task_scope_id_map.end()) {
      if (SetOpCompileResult(task_iter->second, task_pair.second.teNodeOpDesc, true, compile_ret_map) != SUCCESS) {
        return FAILED;
      }
      (void)task_pair.second.teNodeOpDesc->DelAttr(kJsonFilePath);
      (void)task_pair.second.teNodeOpDesc->DelAttr(kBinFilePath);
    }
  }
  
  FE_TIMECOST_END(SuperKernelCompile, "SuperKernelCompile")
  FE_LOGD("Finish SuperKernelCompile");
  return SUCCESS;
}

void TbeOpStoreAdapter::SetCustomFlag(const ScopeNodeIdMap &fusion_nodes_map) const {
  for (const auto &iter : fusion_nodes_map) {
    for (const ge::Node *node : iter.second) {
      if (node == nullptr) {
        continue;
      }

      auto op_desc = node->GetOpDesc();
      bool is_custom_op = IsCustomOp(*(op_desc.get()));
      if (!ge::AttrUtils::SetBool(op_desc, IS_CUSTOM_OP, is_custom_op)) {
        FE_LOGW("Node[%s]: Failed to set is_custom_op[%d].", op_desc->GetName().c_str(), is_custom_op);
      }
    }
  }
}

bool TbeOpStoreAdapter::CheckOpsPathPrefix(const ScopeNodeIdMap &fusion_nodes_map) const {
  for (const auto &iter : fusion_nodes_map) {
    for (const ge::Node *node : iter.second) {
      if (node == nullptr) {
        continue;
      }
      if (ge::OpTypeUtils::IsAutofuseNode(node->GetOpDesc())) {
        continue;
      }

      std::string ops_path_name_prefix = "";
      if (!ge::AttrUtils::GetStr(node->GetOpDesc(), OPS_PATH_NAME_PREFIX, ops_path_name_prefix)) {
        FE_LOGW("Node[%s]: Failed to get ops_path_name_prefix", node->GetName().c_str());
        std::string un_supported_reason;
        if (fe_ops_kernel_info_store_ptr_ == nullptr || !fe_ops_kernel_info_store_ptr_->CheckSupported(
                                                                node->GetOpDesc(), un_supported_reason)) {
          FE_LOGE("[ChkSpt][OpChk][Node %s, type %s] This op is not supported inside"
                  " its implementation. Reason is %s",
                  node->GetName().c_str(), node->GetType().c_str(), un_supported_reason.c_str());
          return false;
        }
      }
    }
  }
  return true;
}

void TbeOpStoreAdapter::GetAutoMode(const ScopeNodeIdMap &fusion_nodes_map, bool &auto_mode) const {
  for (const auto &iter : fusion_nodes_map) {
    for (const ge::Node *node : iter.second) {
      if (node == nullptr) {
        continue;
      }
      uint32_t thread_mode = 0;
      (void)ge::AttrUtils::GetInt(node->GetOpDesc(), kThreadMode, thread_mode);
      if (thread_mode != 0) {
        auto_mode = true;
        break;
      }
    }
  }
}

/*
 *  @ingroup fe
 *  @brief   compile fused op and single op, and generate .o and json files
 *  @param   [in]  fusion_nodes_map  op id and fused sub-graph
 *  @ptaram  [out] json_path_map    keep path of .o and json of each op
 *  @return  SUCCESS or FAILED
 */
Status TbeOpStoreAdapter::CompileOp(CompileInfoParam &compile_info) {
  FE_LOGD("TbeOpStoreAdapter::Compile Op begin.");
  // If the map is empty, then there is no fusion op.
  if (compile_info.fusion_nodes_map.empty()) {
    FE_LOGD("Call Fusion Engine successfully, but there is no fusion op");
    return SUCCESS;
  }

  SetCustomFlag(compile_info.fusion_nodes_map);
  if (!CheckOpsPathPrefix(compile_info.fusion_nodes_map)) {
    return FAILED;
  }
  bool auto_mode = false;
  GetAutoMode(compile_info.fusion_nodes_map, auto_mode);
  FE_LOGD("TbeOpStoreAdapter::Compile Op auto_mode:%d.", auto_mode);
  if (PlatformUtils::Instance().GetCubeVecState() == CubeVecStateNew::CUBE_VEC_SPLIT &&
      PlatformUtils::Instance().GetFftsMode() == FFTS_MODE_FFTS_PLUS && auto_mode) {
    return CompileMultiKernelSliceOp(compile_info.fusion_nodes_map, compile_info.compile_ret_map,
                                     compile_info.buff_fus_compile_failed_nodes, compile_info.buff_fus_to_del_nodes);
  } else {
    return ParallelCompileOp(compile_info);
  }
}

bool TbeOpStoreAdapter::IsL1FusionOptimizedNodes(const std::vector<ge::Node *> &nodes) const {
  if (nodes.empty()) {
    return false;
  }
  ge::Node *first_node = nodes[0];
  return ScopeAllocator::HasL1ScopeAttr(first_node->GetOpDesc());
}

bool TbeOpStoreAdapter::IsBuffFusOptimizedNodes(const std::vector<ge::Node *> &scope_op) const {
  bool need_precompile_node = false;
  bool ret_lx;
  for (auto &op : scope_op) {
    if (op == nullptr) {
      continue;
    }
    ret_lx = ge::AttrUtils::GetBool(op->GetOpDesc(), NEED_RE_PRECOMPILE, need_precompile_node);
    if (!ret_lx) {
      return false;
    }
    if (!need_precompile_node) {
      return false;
    }
  }
  return true;
}

void TbeOpStoreAdapter::SetFusionFailedId(const vector<ge::Node *> &fusion_nodes,
                                          const int64_t &fusion_failed_id) const {
  for (ge::Node *node : fusion_nodes) {
    if (node == nullptr) {
      continue;
    }
    string name = node->GetName();
    if (ge::AttrUtils::SetInt(node->GetOpDesc(), FUSION_FAILED_ID_ATTR, fusion_failed_id)) {
      FE_LOGD("Node[%s]: set failed_id[%ld] successfully.", name.c_str(), fusion_failed_id);
    }
  }
}

bool TbeOpStoreAdapter::StopCompileOpInTuningAndAfterUBMatchMode() const {
  std::string build_mode_value = FEContextUtils::GetBuildMode();
  std::string step_mode_value = FEContextUtils::GetBuildStep();
  if (build_mode_value == ge::BUILD_MODE_TUNING && step_mode_value == ge::BUILD_STEP_AFTER_UB_MATCH) {
    FE_LOGI("No need to try recovery if build_mode is [%s] and step is [%s].", build_mode_value.c_str(),
            step_mode_value.c_str());
    return true;
  }
  return false;
}

inline bool TbeOpStoreAdapter::IsFuzzCompileStrategy(const CompileStrategy &compile_strategy) const {
  return compile_strategy == CompileStrategy::COMPILE_STRATEGY_ONLINE_FUZZ;
}

Status TbeOpStoreAdapter::GetRangeLimit(const NodeGeneralInfoPtr &node_info_ptr,
                                        const ge::NodePtr &node_ptr) const {
  auto op_kernel_ptr = node_info_ptr->op_kernel;
  if (op_kernel_ptr == nullptr) {
    return SUCCESS;
  }
  RangeLimitType range_limit_type = op_kernel_ptr->GetRangeLimitType();
  if (range_limit_type != RangeLimitType::DEFAULT) {
    auto iter = kRangeLimitBoolMap.find(range_limit_type);
    if (iter != kRangeLimitBoolMap.end()) {
      node_info_ptr->is_limited_range = iter->second;
    } else if (range_limit_type == RangeLimitType::DYNAMIC) {
      (void)GetRangeLimitType(node_ptr, *(node_info_ptr->op_info.get()),
                              node_info_ptr->is_limited_range);
    } else {
      FE_LOGW("Invalid limited value for node[%s].", node_ptr->GetName().c_str());
      return FAILED;
    }
  } else {
    FE_LOGD("Get rangeLimit value from opkernel is null, node[%s] range is unlimited.", node_ptr->GetName().c_str());
    node_info_ptr->is_limited_range = false;
  }
  return SUCCESS;
}

bool TbeOpStoreAdapter::StopWaitTaskFinishInTuningAndAfterBuilderMode(const bool is_fusion_check,
                                                                      const CompileStrategy &compile_strategy) const {
  if (is_fusion_check) {
    FE_LOGD("Current compile task is do fusion check, must wait task finish");
    return false;
  }
  std::string build_mode_value = FEContextUtils::GetBuildMode();
  std::string step_mode_value = FEContextUtils::GetBuildStep();
  bool no_need_to_wait_task_finish =
      ((build_mode_value == ge::BUILD_MODE_TUNING && step_mode_value == ge::BUILD_STEP_AFTER_BUILDER) ||
       (build_mode_value == ge::BUILD_MODE_TUNING && step_mode_value == ge::BUILD_STEP_AFTER_BUILDER_SUB));
  if (compile_strategy == CompileStrategy::COMPILE_STRATEGY_OP_SPEC && no_need_to_wait_task_finish) {
    FE_LOGI("No need to wait task finish if build_mode is [%s] and step is [%s] and flag is %d.",
            build_mode_value.c_str(), step_mode_value.c_str(), static_cast<int32_t>(compile_strategy));
    return true;
  }
  return false;
}

Status TbeOpStoreAdapter::SetTaskToTeFusion(CompileTaskPara &task_para,
                                            const std::vector<ge::NodePtr> &buff_fus_to_del_nodes,
                                            const CompileStrategy &compile_strategy, const bool &is_fusion_check) {
  for (auto &iter : *task_para.fusion_nodes_map) {
    task_para.task_num++;
    uint64_t taskId = GetAtomicId();
    task_para.task_scope_id_map.insert(std::make_pair(taskId, iter.first));
    FE_LOGD("%lu, taskId %lu, scope_id %ld, set compile %s task.", GetCurThreadId(), taskId, iter.first,
            iter.second.empty() ? "none" : iter.second[0]->GetName().c_str());
    // set compile task
    if (SetTeTask(iter.second, taskId, buff_fus_to_del_nodes, compile_strategy, is_fusion_check) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][FillTaskPara] Failed to set compile task for op [%s]",
                      iter.second.empty() ? "none" : iter.second[0]->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status TbeOpStoreAdapter::ProcessLxFusionFailCompileTasks(CompileTaskPara &task_para,
                                                          std::vector<ge::NodePtr> &l1_fusion_failed_nodes,
                                                          std::vector<ge::NodePtr> &buff_fus_failed_nodes) const {
  if (task_para.failed_tasks.empty()) {
    return SUCCESS;
  }
  for (auto iter = task_para.failed_tasks.begin(); iter != task_para.failed_tasks.end();) {
    uint64_t task_id = iter->first;
    std::map<uint64_t, int64_t>::const_iterator scope_id_iter = task_para.task_scope_id_map.find(task_id);
    if (scope_id_iter == task_para.task_scope_id_map.end()) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][ProcessLxFusionFailTask] can not find scope id by task id[%lu]",
                      task_id);
      return FAILED;
    }

    int64_t scope_id = scope_id_iter->second;
    ScopeNodeIdMap::const_iterator nodes_iter = task_para.fusion_nodes_map->find(scope_id);
    if (nodes_iter == task_para.fusion_nodes_map->end()) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][ProcessLxFusionFailTask] can not find fusion nodes by scope id[%ld].",
                      scope_id);
      return FAILED;
    }
    // l1 fuson failed nodes
    if (IsL1FusionOptimizedNodes(nodes_iter->second)) {
      FE_LOGI("Compile L1 fusion optimized nodes not successfully. Scope id[%ld], task id[%lu].", scope_id, task_id);
      iter = task_para.failed_tasks.erase(iter);
      for (auto &node : nodes_iter->second) {
        RemoveL1FusionScopeAttr(node->GetOpDesc());
        l1_fusion_failed_nodes.push_back(node->shared_from_this());
        FE_LOGD("L1 fusion compile unsuccess node[%s, %s].", node->GetName().c_str(), node->GetType().c_str());
      }
      continue;
    }
    // other nodes after lxfusion
    if (IsBuffFusOptimizedNodes(nodes_iter->second)) {
      FE_LOGI("Compile nodes who are optimized by lxfusion not successfully. "
              "Scope id[%ld], task id[%lu].", scope_id, task_id);
      iter = task_para.failed_tasks.erase(iter);
      for (auto &node : nodes_iter->second) {
        buff_fus_failed_nodes.push_back(node->shared_from_this());
        FE_LOGD("Other lxfusion compile unsuccess node[%s, %s].", node->GetName().c_str(), node->GetType().c_str());
      }
      continue;
    }
    iter++;
  }
  return SUCCESS;
}

void TbeOpStoreAdapter::SaveMsTuneErrorMsg(CompileTaskPara &task_para) const {
  std::map<uint64_t, int64_t> &pre_scope_id_map = task_para.task_scope_id_map;
  for (auto &fin_task_pair : task_para.failed_tasks) {
    uint64_t task_id = fin_task_pair.first;
    std::map<uint64_t, int64_t>::const_iterator task_iter = pre_scope_id_map.find(task_id);
    if (task_iter == pre_scope_id_map.end()) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][SaveMsTuneErrorMsg] Thread[%lu], not find taskId[%lu]", GetCurThreadId(),
                      task_id);
      return;
    }
    int64_t scope_id = task_iter->second;
    ScopeNodeIdMap::const_iterator nodes_iter = task_para.fusion_nodes_map->find(scope_id);
    if (nodes_iter == task_para.fusion_nodes_map->end()) {
      FE_LOGD("Can not find fusion nodes by scope id[%ld]", scope_id);
      return;
    }
    FE_LOGD("Save compile msg, taskId[%lu], tid[%lu].", task_id, GetCurThreadId());

    string node_name;
    for (auto &node : nodes_iter->second) {
      node_name += node->GetName();
      node_name += ", ";
    }
    FE_LOGI("Nodes {%s} compile not successfully.", node_name.c_str());

    ge::ComputeGraphPtr owner_graph = nodes_iter->second.at(0)->GetOwnerComputeGraph();
    FE_LOGD("Graph name is %s.", owner_graph->GetName().c_str());
    SaveErrorMessage("S40000", "op_name", nodes_iter->second.at(0)->GetName());
  }
}

Status TbeOpStoreAdapter::RetryCompileFailOp(CompileTaskPara &task_para) {
  if (ProcessFailCompileTask(task_para, CompileStrategy::COMPILE_STRATEGY_OP_SPEC) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][RetryCompFailOp] Thread[%lu] failed when processing the task that had failed to compile.", GetCurThreadId());
    return FAILED;
  }
  // wait for finish
  FE_TIMECOST_START(WaitTaskFinish);
  if (WaitTaskFinish(task_para) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][RetryCompFailOp] Thread[%lu] waiting task finish failed", GetCurThreadId());
    return FAILED;
  }
  FE_TIMECOST_END(WaitTaskFinish, "RetryCompileFailOp.WaitTaskFinish");

  if (!task_para.failed_tasks.empty()) {
    for (auto &fin_task_pair : task_para.failed_tasks) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][RetryCompFailOp] Thread[%lu] failed to recompile single op[%s]",
                      GetCurThreadId(), fin_task_pair.second.teNodeOpDesc->GetName().c_str());
    }
    return FAILED;
  }

  // process successful sgt sliced task
  if (ProcessSuccSgtSliceTask(task_para) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][RetrySgtSliceOp] Thread[%lu] failed to process successful sgt task.",
                    GetCurThreadId());
  }
  // process successful task
  if (ProcessSuccCompileTask(task_para) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][RetryCompFailOp] Thread[%lu] failed to process successful task.",
                    GetCurThreadId());
    return FAILED;
  }

  return SUCCESS;
}

Status TbeOpStoreAdapter::ProcessAllFailedCompileTasks(CompileTaskPara &task_para,
                                                       std::vector<ge::NodePtr> &buff_fus_compile_failed_nodes,
                                                       std::vector<ge::NodePtr> &l1_fusion_failed_nodes,
                                                       const CompileStrategy &compile_strategy) {
  if (!task_para.failed_tasks.empty()) {
    SaveMsTuneErrorMsg(task_para);

    if (ProcessLxFusionFailCompileTasks(task_para, l1_fusion_failed_nodes, buff_fus_compile_failed_nodes) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][ProcLxFusFailedCompTask] Thread[%lu] failed to process lx failed tasks",
                      GetCurThreadId());
      return FAILED;
    }

    if (StopCompileOpInTuningAndAfterUBMatchMode()) {
      FE_LOGI("No need to try recovery fused op");
      // Because there will be rollback function for l1 fusion,
      // l1 fusion failure will be considered as success
      // !l1_fusion_failed_nodes.empty() - l1 fusion failed
      // task_para.failed_tasks.empty() - no other fusion failed except l1 and l2
      // buff_fus_compile_failed_nodes.empty() - no l2 fusion failed
      bool is_only_l1_failed = !l1_fusion_failed_nodes.empty() && task_para.failed_tasks.empty() &&
                               buff_fus_compile_failed_nodes.empty();
      if (is_only_l1_failed) {
        FE_LOGD("All unsuccess nodes are l1 fusion nodes and they will be backed off to ub fusion later");
        return SUCCESS;
      }
      return FAILED;
    }

    if (IsFuzzCompileStrategy(compile_strategy)) {
      FE_LOGI("online fuzzy compile, no need retry.");
      return FAILED;
    }

    if (ProcessFailCompileTask(task_para, compile_strategy) == FAILED) {
      REPORT_FE_ERROR("Thread[%lu] processing fail task failed", GetCurThreadId());
      return FAILED;
    }
    // wait for finish
    FE_TIMECOST_START(WaitTaskFinish);
    if (WaitTaskFinish(task_para) == FAILED) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][ProcFailedCompTask] Thread[%lu] waiting task finish failed",
                      GetCurThreadId());
      return FAILED;
    }
    FE_TIMECOST_END(WaitTaskFinish, "ProcessAllFailedCompileTasks.WaitTaskFinish");

    if (!task_para.failed_tasks.empty()) {
      for (auto &fin_task_pair : task_para.failed_tasks) {
        REPORT_FE_ERROR("[SubGraphOpt][Compile][ProcFailedCompTask] Thread[%lu] failed to recompile single op[%s]",
                        GetCurThreadId(), fin_task_pair.second.teNodeOpDesc->GetName().c_str());

      }
      return FAILED;
    }
  }
  return SUCCESS;
}

Status TbeOpStoreAdapter::ParallelCompileOp(CompileInfoParam &compile_info) {
  FE_CHECK(TeFusion == nullptr, REPORT_FE_ERROR("[SubGraphOpt][Compile][ParalCompOp] TeFusion is nullptr."),
           return FAILED);
  FE_TIMECOST_START(TeFusion);
  CompileTaskPara task_para(0, compile_info.is_fusion_check, &compile_info.compile_ret_map,
                            &compile_info.fusion_nodes_map);
  if (SetTaskToTeFusion(task_para, compile_info.buff_fus_compile_failed_nodes, compile_info.compile_strategy,
                        compile_info.is_fusion_check) != SUCCESS) {
    return FAILED;
  }

  FE_LOGD("Thread[%lu], setting %lu tasks to compile.", GetCurThreadId(), task_para.task_num);
  if (StopWaitTaskFinishInTuningAndAfterBuilderMode(compile_info.is_fusion_check, compile_info.compile_strategy)) {
    FE_LOGI("No need to wait task(%s) finish.", GetStrTaskIdByMap(task_para.task_scope_id_map).c_str());
    return SUCCESS;
  }
  // wait for finish
  FE_TIMECOST_START(WaitTaskFinish);
  if (WaitTaskFinish(task_para) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][ParalCompOp] Thread[%lu] waiting task finish failed", GetCurThreadId());
    return FAILED;
  }
  FE_TIMECOST_END(WaitTaskFinish, "ParallelCompileOp.WaitTaskFinish");
  // process success task
  if (ProcessSuccCompileTask(task_para) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][ParalCompOp] Thread[%lu] failed when processing the task that had been successfully compiled.",
                    GetCurThreadId());
    return FAILED;
  }
  if (compile_info.is_fusion_check) {
    if (DelScopeIdOfFailedNodes(task_para) != SUCCESS) {
      return FAILED;
    }
    FE_TIMECOST_END(TeFusion, "Fusion check during FEGraphOptimizer::OptimizeFusedGraph");
  } else {
    // process failed task
    if (ProcessAllFailedCompileTasks(task_para, compile_info.buff_fus_compile_failed_nodes,
                                     compile_info.l1_fusion_failed_nodes, compile_info.compile_strategy) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][ParalCompOp] Thread[%lu] failed when processing the task that had failed to compile.", GetCurThreadId());
      return FAILED;
    }
    // process success task
    if (ProcessSuccCompileTask(task_para) == FAILED) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][ParalCompOp] Thread[%lu] failed when processing the task that had been successfully compiled.",
                      GetCurThreadId());
      return FAILED;
    }
    FE_TIMECOST_END(TeFusion, "TeFusion during FEGraphOptimizer::OptimizeFusedGraph");
  }

  FE_LOGD("TbeOpStoreAdapter::Compile Op successfully, tid:%lu.", GetCurThreadId());

  return SUCCESS;
}

Status TbeOpStoreAdapter::SetOpCompileResult(const int64_t scope_id, const ge::OpDescPtr &compile_op_desc,
                                             const bool is_replace, CompileResultMap &compile_ret_map) const {
  FE_CHECK_NOTNULL(compile_op_desc);
  CompileResultInfo com_ret_info;
  (void)ge::AttrUtils::GetStr(compile_op_desc, kJsonFilePath, com_ret_info.json_file_path);
  FE_LOGD("Json path for node %s is %s, scope id is [%ld].",
          compile_op_desc->GetNamePtr(), com_ret_info.json_file_path.c_str(), scope_id);

  (void)ge::AttrUtils::GetStr(compile_op_desc, kBinFilePath, com_ret_info.bin_file_path);
  FE_LOGD("Bin path for node %s is %s, scope id is [%ld].",
          compile_op_desc->GetNamePtr(), com_ret_info.bin_file_path.c_str(), scope_id);

  com_ret_info.json_ptr = compile_op_desc->TryGetExtAttr<TbeJsonPtr>(kJsonValuePtr, nullptr);
  com_ret_info.bin_ptr = compile_op_desc->TryGetExtAttr<ge::OpKernelBinPtr>(kBinValuePtr, nullptr);
  if (com_ret_info.json_ptr == nullptr || com_ret_info.bin_ptr == nullptr) {
    FE_LOGD("Json or bin ext attr from node whose scope id is [%ld] is null.", scope_id);
  }

  if (com_ret_info.json_file_path.empty()) {
    REPORT_FE_ERROR("[SubGraphOpt][SetOpCompileResult] Json path of node %s is empty",
                    compile_op_desc->GetName().c_str());
    return FAILED;
  }

  if (is_replace && !compile_ret_map[scope_id].empty()) {
    compile_ret_map[scope_id].clear();
  }

  // keep json path
  compile_ret_map[scope_id].push_back(com_ret_info);
  return SUCCESS;
}

void TbeOpStoreAdapter::SetOpDescCustomOp(ge::OpDescPtr op_desc) const {
  int64_t tmp_imply_type = 0;
  if (!ge::AttrUtils::GetInt(op_desc, FE_IMPLY_TYPE, tmp_imply_type)) {
    FE_LOGD("Node[%s]: get fe_imply_type unsuccessful.", op_desc->GetName().c_str());
  }
  int impl_type = GetMainImplType<int>(tmp_imply_type);
  bool is_custom_op = true;
  if (BUILT_IN_IMPLY_TYPE.count(impl_type) != 0) {
    is_custom_op = false;
  }
  if (!ge::AttrUtils::SetBool(op_desc, IS_CUSTOM_OP, is_custom_op)) {
    FE_LOGD("Node[%s]: set is_custom_op[%d] unsuccessful.", op_desc->GetName().c_str(), is_custom_op);
  }
}

Status TbeOpStoreAdapter::DoFuzzBuildTbeOp(std::vector<ge::Node *> &node_vec, uint64_t taskId, uint64_t thread_id) {
  if (node_vec.size() != 1) {
    return NOT_CHANGED;
  }

  ge::Node *node = node_vec[0];
  auto op_desc = node->GetOpDesc();
  if (!IsFuzzBuildOp(*op_desc)) {
    FE_LOGD("[SubGraphOpt][DoFuzzBuild]No Need to do fuzzy build tbe op.");
    return NOT_CHANGED;
  }

  FE_LOGD("Start to do fuzz build tbe op[%s].", node->GetName().c_str());
  te::OpBuildResCode result = FuzzBuildTbeOp(taskId, thread_id, *node);
  if (result == te::OP_BUILD_FAIL) {
    ErrorMessageDetail error_msg(EM_COMPLIE_FAILED, {op_desc->GetName(), op_desc->GetType()});
    ReportErrorMessage(error_msg);
    REPORT_FE_ERROR("[SubGraphOpt][Compile][DoFuzzBuild]Failed to fuzz compile te fusion op %s, tid:%lu, taskId:%lu.",
        op_desc->GetName().c_str(), thread_id, taskId);
    return FAILED;
  }
  FE_LOGD("Set op[%s] successfully, thread[%lu], taskId[%lu].", op_desc->GetName().c_str(), thread_id, taskId);
  return SUCCESS;
}

Status TbeOpStoreAdapter::SetTeTask(std::vector<ge::Node *> &node_vec, uint64_t taskId,
                                    const std::vector<ge::NodePtr> &buff_fus_to_del_nodes,
                                    const CompileStrategy &compile_strategy, const bool &is_fusion_check) {
  if (node_vec.empty()) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][SetTeTask] NodeVec is empty.");
    return FAILED;
  }

  std::shared_ptr<ge::OpDesc> op_desc_ptr = nullptr;
  FE_MAKE_SHARED(op_desc_ptr = std::make_shared<ge::OpDesc>(node_vec[0]->GetName(), ""), return FAILED);

  uint64_t thread_id = GetCurThreadId();

  std::string op_compile_strategy;
  for (ge::Node *node : node_vec) {
    if (ge::AttrUtils::GetStr(node->GetOpDesc(), ge::ATTR_NAME_OP_COMPILE_STRATEGY, op_compile_strategy) &&
        !op_compile_strategy.empty()) {
      FE_LOGD("Compile strategy of node[%s, %s] is [%s].", node->GetName().c_str(), node->GetType().c_str(),
              op_compile_strategy.c_str());
      break;
    }
  }
  for (ge::Node *node : node_vec) {
    (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kOnlyFusionCheck, is_fusion_check);
  }
  FE_LOGD("Get _op_compile_strategy attr from graph is %s.", op_compile_strategy.c_str());
  FE_LOGD("Flag of compile strategy is %d.", static_cast<int32_t>(compile_strategy));
  if (compile_strategy != CompileStrategy::COMPILE_STRATEGY_OP_SPEC) {
    auto compile_strategy_iter = kCompileStrategyStrMap.find(compile_strategy);
    if (compile_strategy_iter != kCompileStrategyStrMap.end()) {
      op_compile_strategy = compile_strategy_iter->second;
      FE_LOGD("Op compile strategy has been modified to %s due to compile strategy.", op_compile_strategy.c_str());
    }
  }

  // judge fuzz compile
  Status res = DoFuzzBuildTbeOp(node_vec, taskId, thread_id);
  if (res == SUCCESS) {
    FE_LOGD("Node: %s, do fuzz build tbe op successfully.", node_vec[0]->GetOpDesc()->GetName().c_str());
    return SUCCESS;
  } else if (res == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][SetTeTask] Node: %s, do fuzz build tbe op failed.",
                    node_vec[0]->GetOpDesc()->GetName().c_str());
    return FAILED;
  }

  te::OpBuildResCode result =
      TeFusion(node_vec, op_desc_ptr, buff_fus_to_del_nodes, taskId, thread_id, op_compile_strategy);
  if (result == te::OP_BUILD_FAIL) {
    ErrorMessageDetail error_msg(EM_COMPLIE_FAILED,
                                 {node_vec[0]->GetOpDesc()->GetName(), node_vec[0]->GetOpDesc()->GetType()});
    ReportErrorMessage(error_msg);
    REPORT_FE_ERROR("[SubGraphOpt][Compile][SetTeTask] Compile te fusion op %s failed, tid:%lu, taskId:%lu",
                    op_desc_ptr->GetName().c_str(), thread_id, taskId);
    return FAILED;
  }

  for (ge::Node *node : node_vec) {
    (void)node->GetOpDesc()->DelAttr(kOnlyFusionCheck);
  }
  FE_LOGD("Set op[%s] successfully, thread[%lu], taskId[%lu].", op_desc_ptr->GetName().c_str(), thread_id, taskId);
  return SUCCESS;
}

void TbeOpStoreAdapter::SgtGetCompileStrategy(std::vector<ge::Node *> &node_vec, std::string &op_compile_strategy,
                                              const CompileStrategy &compile_strategy) const {
  (void)ge::AttrUtils::GetStr(node_vec.at(0)->GetOpDesc(), ge::ATTR_NAME_OP_COMPILE_STRATEGY, op_compile_strategy);
  FE_LOGD("Get _op_compile_strategy attr from graph is %s.", op_compile_strategy.c_str());
  FE_LOGD("Flag of compile strategy is %d.", static_cast<int32_t>(compile_strategy));
  if (compile_strategy != CompileStrategy::COMPILE_STRATEGY_OP_SPEC) {
    auto compile_strategy_iter = kCompileStrategyStrMap.find(compile_strategy);
    if (compile_strategy_iter != kCompileStrategyStrMap.end()) {
      op_compile_strategy = compile_strategy_iter->second;
      FE_LOGD("Op compile strategy has been modified to %s due to compile strategy.", op_compile_strategy.c_str());
    }
  }
}

Status TbeOpStoreAdapter::SgtSetTeTask(std::vector<ge::Node *> &node_vec, uint64_t taskId,
                                       const std::vector<ge::NodePtr> &buff_fus_to_del_nodes,
                                       const CompileStrategy &compile_strategy, uint64_t slice_shape_index) {
  if (node_vec.empty()) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][SgtSetTeTask] NodeVec is empty.");
    return FAILED;
  }

  std::shared_ptr<ge::OpDesc> op_desc_ptr = nullptr;
  FE_MAKE_SHARED(op_desc_ptr = std::make_shared<ge::OpDesc>(node_vec[0]->GetName(), ""), return FAILED);

  uint64_t thread_id = GetCurThreadId();
  std::string op_compile_strategy;
  SgtGetCompileStrategy(node_vec, op_compile_strategy, compile_strategy);

  // in ffts plus auto mode, static shape can not reuse om model binary
  if (!ge::AttrUtils::SetBool(node_vec[0]->GetOpDesc(), kCanNotReuseOm, true)) {
    FE_LOGW("Node[%s] set not reuse om attr failed.", node_vec[0]->GetOpDesc()->GetName().c_str());
  }

  // judge fuzz compile
  Status res = DoFuzzBuildTbeOp(node_vec, taskId, thread_id);
  if (res == SUCCESS) {
    FE_LOGD("Node: %s, do fuzz build tbe op success.", node_vec[0]->GetOpDesc()->GetName().c_str());
    return SUCCESS;
  } else if (res == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][SgtSetTeTask] Node: %s, do fuzz build tbe op failed.",
                    node_vec[0]->GetOpDesc()->GetName().c_str());
    return FAILED;
  }

  te::OpBuildResCode result = TeFusionV(node_vec, op_desc_ptr, buff_fus_to_del_nodes, taskId, thread_id,
                                        slice_shape_index, op_compile_strategy);
  if (result == te::OP_BUILD_FAIL) {
    ErrorMessageDetail error_msg(EM_COMPLIE_FAILED,
                                 {node_vec[0]->GetOpDesc()->GetName(), node_vec[0]->GetOpDesc()->GetType()});
    ReportErrorMessage(error_msg);
    REPORT_FE_ERROR("[SubGraphOpt][Compile][SgtSetTeTask] Compile te fusion op %s failed, tid:%lu, taskId:%lu",
                    op_desc_ptr->GetName().c_str(), thread_id, taskId);
    return FAILED;
  }

  FE_LOGD("Set op[%s] successfully, thread[%lu], taskId[%lu].", op_desc_ptr->GetName().c_str(), thread_id, taskId);
  return SUCCESS;
}

Status TbeOpStoreAdapter::WaitTaskFinish(CompileTaskPara &task_para) const {
  uint64_t time_interval_threshold_second = Configuration::Instance(engine_name_).GetCompileTaskTraceTimeInterval();
  uint64_t time_const_threshold_second = Configuration::Instance(engine_name_).GetCompileTaskTraceTimeConstThreshold();
  vector<te::FinComTask> fin_com_task;
  task_para.succ_tasks.clear();
  task_para.failed_tasks.clear();

  uint64_t thread_id = GetCurThreadId();
  uint64_t task_num = task_para.task_num;
  bool is_op_compile = !task_para.task_scope_id_map.empty() && !task_para.is_fusion_check &&
                       task_para.fusion_nodes_map != nullptr && !task_para.fusion_nodes_map->empty();
  const uint64_t total_task_num = task_para.task_num;
  auto compile_trace_time = std::chrono::high_resolution_clock::now();
  auto long_trace_time = compile_trace_time;
  while (task_num > 0) {
    fin_com_task.clear();
    bool ret = WaitAllFinished(thread_id, fin_com_task);
    if (!ret) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][Wait] waiting for compile task finish failed. thread[%lu]", thread_id);
      return FAILED;
    }
    // not get task
    if (fin_com_task.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kThreadSleepCost));
      continue;
    }

    for (auto it = fin_com_task.begin(); it != fin_com_task.end();) {
      if (task_para.succ_tasks.count(it->taskId) > 0 || task_para.failed_tasks.count(it->taskId) > 0) {
        FE_LOGW("[Compile][Wait] Get duplicated task: graphId[%lu], taskId[%lu]", it->graphId, it->taskId);
        it = fin_com_task.erase(it);
        continue;
      }
      FE_LOGI("[Compile][Wait] Report finished task: [%lu:%lu]", it->graphId, it->taskId);

      if (it->status == SUCCESS) {
        task_para.succ_tasks.emplace(it->taskId, *it);
      } else {
        task_para.failed_tasks.emplace(it->taskId, *it);
      }
      FE_LOGD("Tid[%lu], taskId[%lu], task_num[%lu], fin_task_num[%lu], status[%d].", thread_id, it->taskId, task_num,
              fin_com_task.size(), it->status);
      it++;
    }

    if (task_num < fin_com_task.size()) {
      REPORT_FE_ERROR(
          "[SubGraphOpt][Compile][Wait] fin_com_task is [%s], taskNum %lu is less than fin size %zu. Maybe it's caused "
          "by buffer_manager not reset",
          GetStrByFinComTaskVec(fin_com_task).c_str(), task_num, fin_com_task.size());
      return FAILED;
    }
    task_num -= fin_com_task.size();
    if (is_op_compile) {
      std::chrono::time_point<std::chrono::high_resolution_clock> now_time = std::chrono::high_resolution_clock::now();
      // submit compile process trace
      SubmitCompileProcessTrace(total_task_num, task_num, time_interval_threshold_second, now_time, compile_trace_time);
      // submit long time cost compile op
      SubmitLongTimeConstTrace(fin_com_task, task_para, time_const_threshold_second, now_time, long_trace_time);
    }
  }
  if (is_op_compile) {
    SubmitCompileFinishTrace(total_task_num);
  }

  FE_LOGD("Tid:%lu, total_num[%lu], succ_task_num[%zu], fail_task_num[%lu].", thread_id, task_para.task_num,
          task_para.succ_tasks.size(), task_para.failed_tasks.size());
  return SUCCESS;
}

void TbeOpStoreAdapter::SubmitCompileFinishTrace(const uint64_t total_task_num) const {
  TraceMsgBasePtr compile_process_trace = nullptr;
  FE_MAKE_SHARED(compile_process_trace = std::make_shared<CompileProcessTraceMsg>(total_task_num),
                 return);
  TraceHandleManager::Instance().SubmitGlobalTrace(compile_process_trace);
}

void TbeOpStoreAdapter::SubmitCompileProcessTrace(const uint64_t total_task_num, const uint64_t wait_task_num,
  const uint64_t time_interval_threshold, const std::chrono::time_point<std::chrono::high_resolution_clock> &now_time,
  std::chrono::time_point<std::chrono::high_resolution_clock> &last_trace_time) const {
  uint64_t time_interval =
          static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(now_time - last_trace_time).count());
  if (time_interval > time_interval_threshold) {
    TraceMsgBasePtr compile_process_trace = nullptr;
    FE_MAKE_SHARED(compile_process_trace = std::make_shared<CompileProcessTraceMsg>(total_task_num, wait_task_num),
                   return);
    TraceHandleManager::Instance().SubmitGlobalTrace(compile_process_trace);
    last_trace_time = now_time;
  }
}

void TbeOpStoreAdapter::SubmitLongTimeConstTrace(const vector<te::FinComTask> &fin_com_task,
  const CompileTaskPara &task_para, const uint64_t time_interval_threshold,
  const std::chrono::time_point<std::chrono::high_resolution_clock> &now_time,
  std::chrono::time_point<std::chrono::high_resolution_clock> &last_trace_time) const {
  if (fin_com_task.empty() || task_para.fusion_nodes_map == nullptr) {
    return;
  }
  uint64_t op_time_interval =
          static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(now_time - last_trace_time).count());
  last_trace_time = now_time;
  if (op_time_interval < time_interval_threshold) {
    return;
  }
  for (const te::FinComTask &fin_task : fin_com_task) {
    auto scope_iter = task_para.task_scope_id_map.find(fin_task.taskId);
    if (scope_iter == task_para.task_scope_id_map.end()) {
      continue;
    }
    auto nodes_iter = task_para.fusion_nodes_map->find(scope_iter->second);
    if (nodes_iter == task_para.fusion_nodes_map->end()) {
      continue;
    }
    if (nodes_iter->second.empty()) {
      continue;
    }
    bool is_fusion_op = nodes_iter->second.size() > 1;
    ge::OpDescPtr first_op_desc = nodes_iter->second.at(0)->GetOpDesc();
    TraceMsgBasePtr long_time_trace = nullptr;
    FE_MAKE_SHARED(long_time_trace = std::make_shared<LongTimeTraceMsg>(is_fusion_op, first_op_desc->GetId(),
                                                                        first_op_desc->GetType(), op_time_interval),
                   return);
    TraceHandleManager::Instance().SubmitGlobalTrace(long_time_trace);
  }
}

Status TbeOpStoreAdapter::ProcessSuccCompileTask(CompileTaskPara &task_para) const {
  if (!task_para.is_fusion_check) {
    for (auto &fin_task_pair : task_para.succ_tasks) {
      FE_LOGD("Process task with first node %s.", fin_task_pair.second.teNodeOpDesc->GetName().c_str());
      auto task_id = fin_task_pair.first;
      auto task_iter = task_para.task_scope_id_map.find(task_id);
      if (task_iter == task_para.task_scope_id_map.end()) {
        REPORT_FE_ERROR("[SubGraphOpt][Compile][ProSucCmplTask] %lu, not find taskId[%lu]", GetCurThreadId(), task_id);
        return FAILED;
      }

      int64_t scope_id = task_para.task_scope_id_map[task_id];
      FE_LOGD("Tid[%lu], get taskId[%lu], scope_id[%ld].", GetCurThreadId(), task_id, scope_id);
      if (SetOpCompileResult(scope_id, fin_task_pair.second.teNodeOpDesc, true, *task_para.compile_ret_map) == FAILED) {
        REPORT_FE_ERROR("[SubGraphOpt][Compile][ProSucCmplTask] %s set op json path failed",
                        (*task_para.fusion_nodes_map)[scope_id][0]->GetName().c_str());
        return FAILED;
      }
      SetSPKAttr((*task_para.fusion_nodes_map)[scope_id], fin_task_pair.second.teNodeOpDesc);
      SetOpCompileInfo(fin_task_pair.second.teNodeOpDesc, (*task_para.fusion_nodes_map)[scope_id]);
      SetOpTilingKey((*task_para.fusion_nodes_map)[scope_id], fin_task_pair.second.teNodeOpDesc);
    }
  }
  task_para.succ_tasks.clear();
  FE_LOGD("Process success task_num[%zu], tid[%lu].", task_para.succ_tasks.size(), GetCurThreadId());
  return SUCCESS;
}

Status TbeOpStoreAdapter::DelScopeIdOfFailedNodes(CompileTaskPara &task_para) {
  const std::map<uint64_t, int64_t> &pre_scope_id_map = task_para.task_scope_id_map;

  for (auto &fin_task_pair : task_para.failed_tasks) {
    auto task_id = fin_task_pair.first;
    auto task_iter = pre_scope_id_map.find(task_id);
    if (task_iter == pre_scope_id_map.end()) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][AutoFusionCheck] tid[%lu], not find taskId[%lu]", GetCurThreadId(),
                      task_id);
      return FAILED;
    }

    int64_t pre_scope_id = task_iter->second;
    std::vector<ge::Node *> &failed_nodes = (*task_para.fusion_nodes_map)[pre_scope_id];
    for (auto &node : failed_nodes) {
      if (node->GetOpDesc()->HasAttr(SCOPE_ID_ATTR)) {
        if (node->GetOpDesc()->DelAttr(SCOPE_ID_ATTR) != ge::GRAPH_SUCCESS) {
          REPORT_FE_ERROR("[SubGraphOpt][Compile][AutoFusionCheck] Failed to delete scope_id");
          return FAILED;
        }
      }
    }
  }
  return SUCCESS;
}

void TbeOpStoreAdapter::RollBackAttributes(std::vector<ge::Node *> &failed_nodes) const {
  for (auto node : failed_nodes) {
    std::vector<string> roll_back_attrs;
    auto op_desc = node->GetOpDesc();
    (void)ge::AttrUtils::GetListStr(op_desc, ROLLBACK_IF_FAILED, roll_back_attrs);
    FE_LOGD("Remove attr: node name %s size %zu.", node->GetName().c_str(), roll_back_attrs.size());
    for (auto &attr : roll_back_attrs) {
      if (ge::AttrUtils::HasAttr(op_desc, attr)) {
        op_desc->DelAttr(attr);
      }
      if (attr == "reuse_input") {
        for (size_t i = 0; i < op_desc->GetAllOutputsDescSize(); i++) {
          auto out_desc = op_desc->MutableOutputDesc(i);
          if (out_desc == nullptr) {
            continue;
          }
          ge::TensorUtils::SetReuseInput(*out_desc.get(), false);
        }
        FE_LOGD("remove reuse_input for node %s.", node->GetName().c_str());
      }
    }
  }
}

Status TbeOpStoreAdapter::SetFailedOpCompileTask(ge::Node* node, CompileTaskPara &task_para,
                                                 const CompileStrategy &compile_strategy) {
  int64_t tmp_imply_type = 0;
  if (!ge::AttrUtils::GetInt(node->GetOpDesc(), FE_IMPLY_TYPE, tmp_imply_type)) {
    REPORT_FE_ERROR(
        "[SubGraphOpt][Compile][ProcFailedCompTask] get imply type failed, op[%s, type %s], op_imply_type[%ld]",
        node->GetOpDesc()->GetName().c_str(), node->GetType().c_str(), tmp_imply_type);
    return FAILED;
  }
  vector<ge::Node *> node_vec = {node};
  int64_t scope_id = ScopeAllocator::Instance().AllocateNegScopeId();
  task_para.fusion_nodes_map->insert(make_pair(scope_id, node_vec));
  if (!(ScopeAllocator::SetScopeAttr(node->GetOpDesc(), scope_id))) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][ProcFailedCompTask] set op[%s] scope id failed.",
                    node->GetName().c_str());
    return FAILED;
  }
  FE_LOGD("Node[%s, %s]: do fusion unsuccessful. Now compile it as single op, which scopeid is %ld.",
          node->GetOpDesc()->GetName().c_str(), node->GetType().c_str(), scope_id);

  // set compile task
  std::vector<ge::NodePtr> buff_fus_to_del_nodes;

  Status result = SetTaskForOneScope(node_vec, scope_id, buff_fus_to_del_nodes, task_para,
                                     compile_strategy);
  if (result != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][ProcFailedCompTask] failed to re-compile single op %s.",
                    node->GetName().c_str());
    return result;
  }
  return SUCCESS;
}

void TbeOpStoreAdapter::ClearTaskPara(CompileTaskPara &task_para) const {
  task_para.task_num = 0;
  task_para.task_scope_id_map.clear();
  task_para.scope_task_ids_map.clear();
  task_para.failed_task_able_to_delete.clear();
}
Status TbeOpStoreAdapter::ProcessFailCompileTask(CompileTaskPara &task_para,
                                                 const CompileStrategy &compile_strategy) {
  if (task_para.failed_tasks.empty()) {
    ClearTaskPara(task_para);
    return SUCCESS;
  }

  std::map<uint64_t, int64_t> pre_scope_id_map = task_para.task_scope_id_map;
  ClearTaskPara(task_para);
  for (auto &fin_task_pair : task_para.failed_tasks) {
    auto task_id = fin_task_pair.first;
    auto task_iter = pre_scope_id_map.find(task_id);
    if (task_iter == pre_scope_id_map.end()) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][ProcFailedCompTask] tid[%lu], not find taskId[%lu]", GetCurThreadId(),
                      task_id);
      return FAILED;
    }

    int64_t pre_scope_id = pre_scope_id_map[task_id];
    FE_LOGD("Retry compile %s, taskId[%lu], tid[%lu].",
            (*task_para.fusion_nodes_map)[pre_scope_id][0]->GetName().c_str(),
            task_id, GetCurThreadId());

    std::vector<ge::Node *> &failed_nodes = (*task_para.fusion_nodes_map)[pre_scope_id];
    RollBackAttributes(failed_nodes);

    for (auto &node : failed_nodes) {
      if (SetFailedOpCompileTask(node, task_para, compile_strategy) != SUCCESS) {
        return FAILED;
      }
    }
    // When every op compile successful, setting failed_id for ops of fusion failed.
    std::vector<ge::Node *> fusion_failed_nodes = failed_nodes;
    SetFusionFailedId(fusion_failed_nodes, pre_scope_id);
    task_para.fusion_nodes_map->erase(pre_scope_id);
  }
  FE_LOGD("Tid[%lu], retry task_num[%zu].", GetCurThreadId(), task_para.failed_tasks.size());
  return SUCCESS;
}

Status TbeOpStoreAdapter::InitTbeFunctions(const PluginManagerPtr &plugin_manager_ptr_param) {
  const string TBE_SELECT_FORMAT_FUNC_NAME = "SelectTbeOpFormat";
  Status ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<bool, const te::TbeOpInfo &, std::string &>(
      TBE_SELECT_FORMAT_FUNC_NAME, SelectTbeOpFormat);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", TBE_SELECT_FORMAT_FUNC_NAME.c_str()), return FAILED);

  const string TBE_CHECK_SUPPORTED_FUNC_NAME = "CheckOpSupported";
  ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<bool, te::TbeOpInfo &, te::CheckSupportedInfo &>(
      TBE_CHECK_SUPPORTED_FUNC_NAME, CheckTbeSupported);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", TBE_CHECK_SUPPORTED_FUNC_NAME.c_str()), return FAILED);

  const string TBE_PRE_COMPILER_FUNC_NAME = "PreBuildTbeOp";
  ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<bool, te::TbeOpInfo &, uint64_t, uint64_t>(
      TBE_PRE_COMPILER_FUNC_NAME, PreBuildTbeOp);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", TBE_PRE_COMPILER_FUNC_NAME.c_str()), return FAILED);

  const string TBE_GET_OP_INFO_FUNC_NAME = "GetOpInfo";
  ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<te::LX_QUERY_STATUS, const te::TbeOpInfo &, std::string &>(
      TBE_GET_OP_INFO_FUNC_NAME, GetOpInfo);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", TBE_GET_OP_INFO_FUNC_NAME.c_str()), return FAILED);

  const string TBE_COMPILER_FUNC_NAME = "TeFusion";
  ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<te::OpBuildResCode, std::vector<ge::Node *>, ge::OpDescPtr,
      const std::vector<ge::NodePtr> &, uint64_t, uint64_t, const std::string &>(
      TBE_COMPILER_FUNC_NAME, TeFusion);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", TBE_COMPILER_FUNC_NAME.c_str()), return FAILED);

  const string TBE_COMPILER_FUNC_NAME_V = "TeFusionV";
  ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<te::OpBuildResCode, std::vector<ge::Node *>, ge::OpDescPtr,
      const std::vector<ge::NodePtr> &, uint64_t, uint64_t, uint64_t, const std::string &>(
      TBE_COMPILER_FUNC_NAME_V, TeFusionV);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", TBE_COMPILER_FUNC_NAME_V.c_str()), return FAILED);

  const string TBE_FUZZ_COMPILER_FUNC_NAME = "FuzzBuildTbeOp";
  ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<te::OpBuildResCode, uint64_t, uint64_t, ge::Node &>(
      TBE_FUZZ_COMPILER_FUNC_NAME, FuzzBuildTbeOp);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", TBE_FUZZ_COMPILER_FUNC_NAME.c_str()), return FAILED);

  const string TBE_TASK_FUSION_FUNC_NAME = "TaskFusion";
  ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<te::OpBuildResCode, const std::vector<ge::Node *> &, ge::OpDescPtr,
      uint64_t, uint64_t>(TBE_TASK_FUSION_FUNC_NAME, TaskFusionFunc);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", TBE_TASK_FUSION_FUNC_NAME.c_str()), return FAILED);

  const string TBE_WAIT_FINISH_FUNC_NAME = "WaitAllFinished";
  ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<bool, uint64_t, vector<te::FinComTask> &>(
      TBE_WAIT_FINISH_FUNC_NAME, WaitAllFinished);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", TBE_WAIT_FINISH_FUNC_NAME.c_str()), return FAILED);

  const string TBE_INIT_FUNC_NAME = "TbeInitialize";
  ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<bool, const std::map<std::string, std::string> &, bool *>(
      TBE_INIT_FUNC_NAME, TbeInitialize);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", TBE_INIT_FUNC_NAME.c_str()), return FAILED);

  const string TBE_FINALIZE_FUNC_NAME = "TbeFinalize";
  ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<bool>(TBE_FINALIZE_FUNC_NAME, TbeFinalize);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", TBE_FINALIZE_FUNC_NAME.c_str()), return FAILED);

  const string TBE_FINALIZE_SESSION_INFO_FUNC_NAME = "TbeFinalizeSessionInfo";
  ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<bool, const std::string &>(TBE_FINALIZE_SESSION_INFO_FUNC_NAME, TbeFinalizeSessionInfo);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", TBE_FINALIZE_SESSION_INFO_FUNC_NAME.c_str()), return FAILED);

  const string CHECK_IS_TBE_GENERALIZEFUNC_REGISTERED = "CheckIsTbeGeneralizeFuncRegistered";
  ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<bool, const te::TbeOpInfo &, bool &>(
      CHECK_IS_TBE_GENERALIZEFUNC_REGISTERED, CheckIsTbeGeneralizeFuncRegistered);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", CHECK_IS_TBE_GENERALIZEFUNC_REGISTERED.c_str()),
           return FAILED);

  const string TE_GENERALIZE = "TeGeneralize";
  ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<bool, const te::TbeOpInfo &, \
      const te::TE_GENERALIZE_TYPE &, const ge::NodePtr &>(TE_GENERALIZE, TeGeneralize);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", TE_GENERALIZE.c_str()), return FAILED);

  const string GET_SPECIFIC_INFO = "GetOpSpecificInfo";
  ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<bool, const te::TbeOpInfo &, \
      std::string &>(GET_SPECIFIC_INFO, GetOpSpecificInfo);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", GET_SPECIFIC_INFO.c_str()), return FAILED);

  const string GET_OP_UNIQUE_KEY_FUNC_NAME = "GetOpUniqueKeys";
  ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<bool, const te::TbeOpInfo &, std::vector<std::string> &>(
      GET_OP_UNIQUE_KEY_FUNC_NAME, GetOpUniqueKeyFunc);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", GET_OP_UNIQUE_KEY_FUNC_NAME.c_str()), return FAILED);

  const string DYNAMIC_SHAPE_RANGE_CHECK = "DynamicShapeRangeCheck";
  ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<bool, const te::TbeOpInfo &, \
      bool &, std::vector<size_t> &, std::vector<size_t> &>(DYNAMIC_SHAPE_RANGE_CHECK, DynamicShapeRangeCheck);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", DYNAMIC_SHAPE_RANGE_CHECK.c_str()), return FAILED);

  const string QUERYOPPATTERN = "QueryOpPattern";
  ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<bool,
      std::vector<std::pair<std::string, std::string>> &>(QUERYOPPATTERN, QueryOpPattern);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", QUERYOPPATTERN.c_str()), return FAILED);

  const string GET_ALL_COMPILE_STATISTICS = "GetAllCompileStatistics";
  ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<void,
          std::vector<std::string> &>(GET_ALL_COMPILE_STATISTICS, GetAllCompileStatistics);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", GET_ALL_COMPILE_STATISTICS.c_str()), return FAILED);

  const string ISOPPKERNELINSTALL = "IsOppKernelInstalled";
  ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<bool, bool, int64_t>(ISOPPKERNELINSTALL, IsOppKernelInstalled);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", ISOPPKERNELINSTALL.c_str()), return FAILED);

  const string BUILD_SUPER_KERNEL_FUNC_NAME = "BuildSuperKernel";
  ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<te::OpBuildResCode, const std::vector<ge::Node *> &, ge::OpDescPtr,
        uint64_t, uint64_t>(BUILD_SUPER_KERNEL_FUNC_NAME, BuildSuperKernel);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", BUILD_SUPER_KERNEL_FUNC_NAME.c_str()), return FAILED);

  const string GET_KERNEL_META_DIR = "GetKernelMetaDir";
  ret = plugin_manager_ptr_param->GetFunctionFromTbePlugin<std::string>(GET_KERNEL_META_DIR, GetKernelMetaDir);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to get function[%s].", GET_KERNEL_META_DIR.c_str()), return FAILED);
  return SUCCESS;
}

std::string TbeOpStoreAdapter::GetCurKernelMetaDir() const {
  FE_CHECK(GetKernelMetaDir == nullptr,
           REPORT_FE_ERROR("The function GetKernelMetaDir of TeFusion is nullptr."),
           return "");
  return GetKernelMetaDir();
}

void TbeOpStoreAdapter::SetSPKAttr(std::vector<ge::Node *> &nodes, const ge::OpDescPtr &op_desc_ptr) const {
  if (nodes.size() != 1) {
    FE_LOGD("Only set attr for SPK sub op.");
    return;
  }
  auto node = nodes[0];
  FE_CHECK(node == nullptr, FE_LOGE("Invalid nullptr."), return);
  int spk_scope = -1;
  (void)ge::AttrUtils::GetInt(node->GetOpDesc(), kAscendcSuperKernelScope, spk_scope);
  if (spk_scope == -1) {
    FE_LOGD("Only set attr for SPK sub op.");
    return;
  }
  std::string json_file_path;
  std::string bin_file_path;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, "json_file_path", json_file_path);
  (void)ge::AttrUtils::GetStr(op_desc_ptr, "bin_file_path", bin_file_path);
  FE_LOGD("[SetSPKAttr] Op[%s, %s] get json file path[%s], bin file path[%s].",
          node->GetNamePtr(), node->GetTypePtr(), json_file_path.c_str(), bin_file_path.c_str());
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), "json_file_path", json_file_path);
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), "bin_file_path", bin_file_path);
}

Status TbeOpStoreAdapter::InitializeInnerHelp() {
  FE_MAKE_SHARED(tbe_info_assembler_ptr_ = std::make_shared<TbeInfoAssembler>(), return FAILED);
  FE_CHECK(tbe_info_assembler_ptr_ == nullptr,
           REPORT_FE_ERROR("[GraphOpt][InitializeInner][InitTbeFunc] tbeInfoAssemblerPtr_ is null."),
           return FAILED);
  if (tbe_info_assembler_ptr_->Initialize() != SUCCESS) {
    return FAILED;
  }
  FE_MAKE_SHARED(tbe_single_op_info_assembler_ptr_ = std::make_shared<TbeSingleOpInfoAssembler>(), return FAILED);
  FE_CHECK(tbe_single_op_info_assembler_ptr_ == nullptr,
           REPORT_FE_ERROR("[GraphOpt][InitializeInner][InitTbeFunc] tbeSingleOpInfoAssemblerPtr_ is null."),
           return FAILED);
  if (tbe_single_op_info_assembler_ptr_->Initialize() != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

void TbeOpStoreAdapter::ParseHardwareInfos(std::map<std::string, std::string> &new_options) {
  std::map<std::string, std::string> hardware_info = Configuration::Instance(AI_CORE_NAME).GetHardwareInfo();
  new_options.insert(hardware_info.begin(), hardware_info.end());
  if (new_options[ge::AICORE_NUM].empty() && new_options.find("ai_core_cnt") != new_options.end()) {
    new_options[ge::AICORE_NUM] = new_options["ai_core_cnt"];
  }
}

void TbeOpStoreAdapter::DealOpDebugDir(std::map<std::string, std::string> &new_options) {
  FE_LOGD("Start to deal debug dir.");
  auto option_iter = new_options.find(ge::DEBUG_DIR);
  if ((option_iter == new_options.end()) || (option_iter->second == "")) {
    char *path = new(std::nothrow) char[kMaxPathSize];
    if (path == nullptr) {
      return;
    }
    if (getcwd(path, kMaxPathSize) != nullptr) {
      string cur_real_path = GetRealPath(string(path));
      if (cur_real_path.empty()) {
        delete[] path;
        return;
      }
      new_options.emplace(std::make_pair(ge::DEBUG_DIR, cur_real_path));
      FE_LOGD("Debug dir has not been set. Now set as %s.", path);
    }
    delete[] path;
  }
}

Status TbeOpStoreAdapter::InitializeInner(const std::map<std::string, std::string> &options) {
  // return SUCCESS if graph optimizer has been initialized.
  if (init_flag) {
    FE_LOGW("TbeOpStoreAdapter has been initialized.");
    return SUCCESS;
  }

  if (InitializeInnerHelp() != SUCCESS) {
    return FAILED;
  }

  string root_path = Configuration::Instance(engine_name_).GetRootPath();
  FE_LOGD("Start to initialize tbe compiler adapter.");
  string real_path = root_path + kTbeSoName;
  string plugin_mgr_name = kTbeSoName;

  if (!GetRealPath(root_path + kOpCompileSoName).empty()) {
    real_path = root_path + kOpCompileSoName;
    plugin_mgr_name = kOpCompileSoName;
    FE_LOGI("FE using libop_compile_adapter.so.");
  }

  FE_MAKE_SHARED(plugin_manager_ptr = std::make_shared<PluginManager>(plugin_mgr_name), return FAILED);
  FE_CHECK(plugin_manager_ptr == nullptr,
           REPORT_FE_ERROR("[GraphOpt][InitializeInner][InitTbeFunc]pluginManagerPtr is nullptr."),
           return FAILED);

  if (plugin_manager_ptr->OpenPlugin(real_path) != SUCCESS) {
    REPORT_FE_ERROR("[FEInit][OpPluginSo] Failed to open plugin so.");
    return FAILED;
  }

  Status ret = InitTbeFunctions(plugin_manager_ptr);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][InitializeInner][InitTbeFunc]: Failed to initialize TbeFunctions.");
    return FAILED;
  }

  if (InitializeTeFusion(options) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][InitializeInner][InitTeFusion]: Failed to initialize TeFusion.");
    return FAILED;
  }

  init_flag = true;
  FE_LOGI("Initialize tbe op store adapter successfully.");
  return SUCCESS;
}

Status TbeOpStoreAdapter::InitializeTeFusion(const std::map<std::string, std::string> &options) {
  std::map<std::string, std::string> new_options = options;
  DealOpDebugDir(new_options);
  ParseHardwareInfos(new_options);
  ChangeBufferOptimize(options, new_options);

  new_options.emplace("short_soc_version", PlatformUtils::Instance().GetShortSocVersion());

  Configuration &config = Configuration::Instance(engine_name_);
  string l2_mode_str = config.IsEnableL2Buffer() ? "1" : "0";
  new_options.emplace(TBE_L2_MODE, l2_mode_str);
  FE_LOGD("Set option [l2_mode, %s].", l2_mode_str.c_str());
  if (config.IsEnableSuperkernelPlus()) {
    new_options.emplace(kEnableSuperkernelPlus, kStrTrue);
    FE_LOGD("Set option [%s, %s].", kEnableSuperkernelPlus, kStrTrue.c_str());
  }
  for (const std::pair<const std::string, std::string> &op_binary_path_pair : config.GetBinaryPathMap()) {
    new_options.emplace(op_binary_path_pair.first, op_binary_path_pair.second);
    FE_LOGD("Insert options [%s, %s].", op_binary_path_pair.first.c_str(), op_binary_path_pair.second.c_str());
  }
  new_options.emplace(kOpDebugConfigTe, config.GetOpDebugConfig());
  FE_LOGD("Set option [%s, %s].", kOpDebugConfigTe, config.GetOpDebugConfig().c_str());

  if (!TbeInitialize(new_options, &support_parallel_compile)) {
    REPORT_FE_ERROR("[GraphOpt][InitializeInner][InitTbeFunc] Failed to init tbe.");
    if (plugin_manager_ptr->CloseHandle() != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][InitializeInner][InitTbeFunc] Failed to close tbe plugin handle.");
    }
    return FAILED;
  }
  TraceHandleManager::Instance().SubmitGlobalTrace("TeFusion has been initialized successfully.");
  return SUCCESS;
}

/*
 *  @ingroup fe
 *  @brief   initial resources needed by TbeOpStoreAdapter, such as dlopen so
 * files
 *           and load function symbols etc.
 *  @return  SUCCESS or FAILED
 */
Status TbeOpStoreAdapter::Initialize(const std::map<std::string, std::string> &options) {
  Status result = InitializeInner(options);
  if (result != SUCCESS) {
    if (plugin_manager_ptr != nullptr) {
      (void)plugin_manager_ptr->CloseHandle();
    }
    return result;
  }
  return SUCCESS;
}

void TbeOpStoreAdapter::ChangeBufferOptimize(const std::map<std::string, std::string> &options,
                                             std::map<std::string, std::string> &new_options) {
  auto iter = options.find(ge::BUFFER_OPTIMIZE);
  if (iter != options.end()) {
    if (iter->second == L2_OPTIMIZE && !Configuration::Instance(AI_CORE_NAME).EnableL2Fusion()) {
      new_options[ge::BUFFER_OPTIMIZE] = OFF_OPTIMIZE;
    }
  } else {
    new_options.insert(std::pair<string, string>(ge::BUFFER_OPTIMIZE, OFF_OPTIMIZE));
  }
}

/*
 *  @ingroup fe
 *  @brief   finalize resources initialized in Initialize function,
 *           such as dclose so files etc.
 *  @return  SUCCESS or FAILED
 */
Status TbeOpStoreAdapter::Finalize() {
  // return SUCCESS if graph optimizer has been initialized.
  if (!init_flag) {
    REPORT_FE_ERROR("[GraphOpt][Finalize] TbeOpStoreAdapter not allowed to finalize before initialized.");
    return FAILED;
  }

  FE_LOGD("Start to finalize tbe compiler adapter.");

  // release TBE resources
  if (!TbeFinalize()) {
    REPORT_FE_ERROR("[GraphOpt][Finalize] Release tbe resources failed.");
    return FAILED;
  }
  TraceHandleManager::Instance().SubmitGlobalTrace("TeFusion has been finalized successfully.");

  // close dlopen handler
  if (plugin_manager_ptr != nullptr) {
    if (plugin_manager_ptr->CloseHandle() != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][Finalize] Failed to close tbe plugin handle.");
      return FAILED;
    }
  }

  init_flag = false;
  FE_LOGI("Finalize tbe op store adapter successfully.");
  return SUCCESS;
}

Status TbeOpStoreAdapter::FinalizeSessionInfo(const std::string& session_graph_id) {
  FE_LOGD("Start to finalize tbe session info.");

  // release TBE session resources
  if (TbeFinalizeSessionInfo == nullptr) {
    REPORT_FE_ERROR("[GraphOpt][Finalize] FinalizeSessionInfo failed.");
    return FAILED;
  }
  if (!TbeFinalizeSessionInfo(session_graph_id)) {
    REPORT_FE_ERROR("[GraphOpt][Finalize] FinalizeSessionInfo session[%s] failed.", session_graph_id.c_str());
    return FAILED;
  }
  return SUCCESS;
}

// we reset intput or output dtype when precision mode is allow_fp32_tofp16 and intput or
// output dtype is all supporrted fp16 in op store.
// input0.dtype=float16,int8,float16   ======>  update input dtype from fp32 to fp16
// input0.dtype=float,int8,float16   ======>  do not update input dtype from fp32 to fp16
bool TbeOpStoreAdapter::UpdateInputOrOutputDtype(const ge::OpDescPtr &op_desc, const ge::GeTensorDescPtr &tensor_desc,
                                                 const size_t input_or_output_index) const {
  if (tensor_desc == nullptr) {
    return false;
  }
  bool need_update_dtype_when_op_checksupport_flag = false;
  (void) ge::AttrUtils::GetBool(tensor_desc, NEED_UPDATE_DTYPE_WHEN_OP_CHECKSUPPORT,
                                need_update_dtype_when_op_checksupport_flag);
  if (need_update_dtype_when_op_checksupport_flag) {
    FE_LOGD("Node[%s, %s]: current precision mode is allow_fp32_tofp16.",
            op_desc->GetName().c_str(), op_desc->GetType().c_str());
    FE_LOGD("Node[%s, %s]: input_or_output[%zu] dtype only supports fp16. Modify dtype from fp32 to fp16.",
            op_desc->GetName().c_str(), op_desc->GetType().c_str(), input_or_output_index);
    tensor_desc->SetDataType(ge::DT_FLOAT16);
    return true;
  }
  return false;
}

void TbeOpStoreAdapter::UpdateDtypeByAllowFp32ToFp16(const ge::OpDescPtr &op_desc,
    size_t input_or_output_index, std::pair<std::vector<size_t>, std::vector<size_t>> &in_out_changed_idx_vec,
    const bool &isinput) const {
  FE_LOGD("Current precision mode is allow_fp32_tofp16, update dtype.");
  std::vector<size_t> input_idx_vec;
  std::vector<size_t> output_idx_vec;
  for (size_t index = 0; index < input_or_output_index; ++index) {
    if (isinput) {
      ge::GeTensorDescPtr input_tensor_desc = op_desc->MutableInputDesc(index);
      if (UpdateInputOrOutputDtype(op_desc, input_tensor_desc, index)) {
        input_idx_vec.emplace_back(index);
      }
    } else {
      ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc(index);
      if (UpdateInputOrOutputDtype(op_desc, output_tensor_desc, index)) {
        output_idx_vec.emplace_back(index);
      }
    }
  }
  if (isinput) {
    in_out_changed_idx_vec.first = input_idx_vec;
  } else {
    in_out_changed_idx_vec.second = output_idx_vec;
  }
}
Status TbeOpStoreAdapter::UpdateTensorByMixPrecisionMode(const ge::NodePtr &node,
    const OpKernelInfoPtr &op_kernel_info_ptr,
    std::pair<std::vector<size_t>, std::vector<size_t>> &in_out_changed_idx_vec) const {
  /* If the Auto Mix precision switch is on, we need to do the
   * checksupport in op by fp16, when the current datatype is fp32 and
   * the op is in white list or current precision mode is force fp16 */
  FE_CHECK_NOTNULL(op_kernel_info_ptr);
  ge::OpDescPtr op_desc = node->GetOpDesc();
  if (IsDtypeSensitiveOp(op_desc->GetType())) {
    return SUCCESS;
  }
  PrecisionPolicy op_kernel_policy = op_kernel_info_ptr->GetPrecisionPolicy();
  PrecisionPolicy op_precision_policy = Configuration::Instance(engine_name_).GetPrecisionPolicy(
      op_desc->GetType(), op_kernel_policy);
  bool white_list_op = op_precision_policy == WHITE;

  int64_t keep_dtype = 0;
  (void)ge::AttrUtils::GetInt(op_desc, KEEP_DTYPE, keep_dtype);
  fe::PrecisionMode precision_mode = fe::PrecisionMode::ENUM_UNDEFINED;
  FeGraphUtils::GetPrecisionModeFromGraph(*(node->GetOwnerComputeGraph()), precision_mode);
  bool fp16_flag = (precision_mode == fe::PrecisionMode::ENUM_FORCE_FP16 ||
                   ((precision_mode == fe::PrecisionMode::ENUM_ALLOW_MIX_PRECISION_FP16) && white_list_op)) &&
                   (keep_dtype != 1);
  bool bf16_flag = (precision_mode == fe::PrecisionMode::ENUM_ALLOW_MIX_PRECISION_BF16 && white_list_op);

  ge::DataType final_dtype;
  std::string reason = "in white list with mix precision switch on or precision_mode is force_fp16";
  if (fp16_flag) {
    FE_LOGI("Change node %s datatype to DT_FLOAT16. Reason is %s", op_desc->GetName().c_str(), reason.c_str());
    final_dtype = ge::DT_FLOAT16;
  } else if (bf16_flag) {
    FE_LOGI("Node %s is in white list and the mix precision_bf16 switch is on. Change datatype to DT_BF16",
             op_desc->GetName().c_str());
    final_dtype = ge::DT_BF16;
  } else if (precision_mode == fe::PrecisionMode::ENUM_ALLOW_FP32_TO_FP16) {
    UpdateDtypeByAllowFp32ToFp16(op_desc, op_desc->GetAllInputsSize(), in_out_changed_idx_vec, true);
    UpdateDtypeByAllowFp32ToFp16(op_desc, op_desc->GetAllOutputsDescSize(), in_out_changed_idx_vec, false);
    return SUCCESS;
  } else {
    return SUCCESS;
  }

  std::vector<size_t> input_idx_vec;
  std::vector<size_t> output_idx_vec;
  for (size_t i = 0; i < op_desc->GetAllInputsSize(); i++) {
    auto input_desc = op_desc->MutableInputDesc(i);
    if (input_desc != nullptr && input_desc->GetDataType() == ge::DT_FLOAT) {
      bool need_update = true;
      (void)ge::AttrUtils::GetBool(input_desc, NEED_UPDATE_DTYPE_WHEN_OP_CHECKSUPPORT, need_update);
      if (need_update) {
        input_desc->SetDataType(final_dtype);
        input_idx_vec.emplace_back(i);
      }
    }
  }
  in_out_changed_idx_vec.first = input_idx_vec;
  for (size_t i = 0; i < op_desc->GetAllOutputsDescSize(); i++) {
    auto output_desc = op_desc->MutableOutputDesc(i);
    if (output_desc != nullptr && output_desc->GetDataType() == ge::DT_FLOAT) {
      bool need_update_dtype = true;
      (void)ge::AttrUtils::GetBool(output_desc, NEED_UPDATE_DTYPE_WHEN_OP_CHECKSUPPORT, need_update_dtype);
      if (need_update_dtype) {
        output_desc->SetDataType(final_dtype);
        output_idx_vec.emplace_back(i);
      }
    }
  }
  in_out_changed_idx_vec.second = output_idx_vec;
  return SUCCESS;
}

bool TbeOpStoreAdapter::AssembleTbeByMixPrecisionMode(const ge::NodePtr &node,
                                                      const OpKernelInfoPtr &op_kernel_info_ptr,
                                                      const bool &is_dynamic_impl,
                                                      te::TbeOpInfo &op_info) const {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  string op_name = op_desc_ptr->GetName();
  string op_type = op_desc_ptr->GetType();
  std::pair<std::vector<size_t>, std::vector<size_t>> in_out_changed_idx_vec;
  (void)UpdateTensorByMixPrecisionMode(node, op_kernel_info_ptr, in_out_changed_idx_vec);
  op_info.SetDynamicImpl(is_dynamic_impl);
  op_info.SetNode(node);
  if (tbe_info_assembler_ptr_->AssembleTbeInfo(node, op_kernel_info_ptr, engine_name_, op_info) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][CheckSupport][AssembleTbeInfo] Failed to Assemble tbe info for node %s type %s.",
                    op_name.c_str(), op_type.c_str());
    return false;
  }
  SetOpSpecificInfoToTbeOpInfo(op_kernel_info_ptr, op_info);
  RestoreDataType(op_desc_ptr, in_out_changed_idx_vec);
  return true;
}
/*
 *  @ingroup fe
 *  @brief   check support something
 */
bool TbeOpStoreAdapter::CheckSupport(const ge::NodePtr &node, CheckSupportParam &check_param,
                                     const bool &is_dynamic_impl, std::string &reason) {
  ge::OpDescPtr op_desc = node->GetOpDesc();
  const string &op_name = op_desc->GetName();
  const string &op_type = op_desc->GetType();

  if (!check_param.op_kernel_ptr->IsNeedCheckSupport()) {
    return true;
  }

  FE_LOGD("[ChkSpt][OpChk] Node[%s, %s]: start to check op implementation file.", op_name.c_str(), op_type.c_str());
  /* If this op is supported in ops store, we still need to check whether
   * it is supported by specific op plugin.
   * If the mix precision switch is on, we try the dtype float16 of this op if
   * the original dtype is fp32. */
  std::string op_dsl_file_path;
  if (!GetOpDslFilePath(op_desc, check_param.op_kernel_ptr, op_dsl_file_path)) {
    REPORT_FE_ERROR("[GraphOpt][Setcheck][CheckSupport][%s, %s] Failed to get op dsl file path.",
                    op_name.c_str(), op_type.c_str());
    return false;
  }

  te::TbeOpInfo op_info(op_name, op_dsl_file_path, op_type, engine_name_);
  GetAndSetOpsPathNamePrefix(check_param.op_kernel_ptr, op_info);
  if (!AssembleTbeByMixPrecisionMode(node, check_param.op_kernel_ptr,
      is_dynamic_impl, op_info)) {
    return false;
  }

  FE_CHECK(CheckTbeSupported == nullptr,
           REPORT_FE_ERROR("[GraphOpt][CheckSupport] Function CheckTbeSupported of TeFusion is nullptr."),
           return false);

  te::CheckSupportedInfo check_res;
  if (CheckTbeSupported(op_info, check_res)) {
    FE_LOGD("Node[%s]: the result of check tbe supported is %s.", op_name.c_str(),
            GetCheckSupportedString(check_res.isSupported).c_str());
    FE_LOGD("Node[%s]: the all_impl_checked flag is %d and the is_dynamic_impl flag is %d.", op_name.c_str(),
            check_res.allImplChecked, check_res.dynamicCompileStatic);
    check_param.all_impl_checked = check_res.allImplChecked;
    check_param.dynamic_compile_static = check_res.dynamicCompileStatic;
    bool result = ConvertCheckSupportResult(node, check_res.reason, check_res.isSupported);
    if (result) {
      FE_LOGD("[ChkSpt][OpChk] Node[%s, %s]: this op is supported by implementation.",
              op_name.c_str(), op_type.c_str());
    } else {
      reason = check_res.reason;
      FE_LOGI("[ChkSpt][OpChk] Node[%s, %s]: this op is not supported by implementation. Reason is [%s].",
              op_name.c_str(), op_type.c_str(), check_res.reason.c_str());
    }
    return result;
  }
  FE_LOGI("Invoke CheckTbeSupported of TeFusion not successfully.");
  return false;
}

bool TbeOpStoreAdapter::CheckUnsupportReason(const ge::NodePtr &node, const std::string &reason,
                                             te::CheckSupportedResult &is_supported) const {
  if (reason.find(kShapeNotSupport) == 0) {
    is_supported = te::FULLY_SUPPORTED;
    (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kOpShapeOrRangeUnsupport, true);
    FE_LOGD("Node[%s]: set attr shape_or_range_unsupport.", node->GetName().c_str());
    return true;
  }
  return false;
}

bool TbeOpStoreAdapter::ConvertCheckSupportResult(const ge::NodePtr &node,
                                                  const std::string &reason,
                                                  te::CheckSupportedResult &is_supported) const {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  (void)op_desc_ptr->DelAttr(STR_PARTIALLY_SUPPORTED);
  if (is_supported == te::FULLY_SUPPORTED) {
    return true;
  } else if (is_supported == te::PARTIALLY_SUPPORTED) {
    (void)ge::AttrUtils::SetBool(op_desc_ptr, STR_PARTIALLY_SUPPORTED, true);
    FE_LOGD("Node[%s]: set attr partially supported.", op_desc_ptr->GetName().c_str());
    return true;
  } else if (is_supported == te::NOT_SUPPORTED) {
    return CheckUnsupportReason(node, reason, is_supported);
  } else {
    return false;
  }
}

template <typename T>
Status TbeOpStoreAdapter::ParseJsonByKey(const std::string &json_str, const std::string &key, T &value) const {
  try {
    nlohmann::json specific_info = nlohmann::json::parse(json_str);
    value = specific_info.at(key).get<T>();
  } catch (std::exception &e) {
    FE_LOGW("Parse json_str failed, string is %s and the reason is %s.", json_str.c_str(), e.what());
    return FAILED;
  }
  return SUCCESS;
}

template <typename T>
Status TbeOpStoreAdapter::GetOpSpecificInfoByKey(const te::TbeOpInfo &op_info, const std::string &key,
                                                 T &value) const {
  FE_CHECK(GetOpSpecificInfo == nullptr,
           REPORT_FE_ERROR("[AssembleTbeInfo][GetOpSpecificInfoByKey] function GetOpSpecificInfo is nullptr."),
           return FAILED);
  std::string specific_info_str;
  if (!GetOpSpecificInfo(op_info, specific_info_str)) {
    FE_LOGW("[AssembleTbeInfo][GetOpSpecificInfoByKey] Failed to call GetOpSpecificInfo function.");
    return FAILED;
  }
  return ParseJsonByKey(specific_info_str, key, value);
}

bool TbeOpStoreAdapter::GetSelectOpFormat(const ge::NodePtr &node, std::string &op_select_format_str) const {
  gert::ExeResGenerationCtxBuilder exe_ctx_builder;
  auto res_ptr_holder = exe_ctx_builder.CreateOpCheckContext(*node.get());
  FE_CHECK(res_ptr_holder == nullptr,
           FE_LOGW("Node[%s, %s] res_ptr_holder is null.", node->GetNamePtr(), node->GetTypePtr()),
           return false);
  auto op_check_content = reinterpret_cast<gert::OpCheckContext *>(res_ptr_holder->context_);
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
      static_cast<gert::OppImplVersionTag>(node->GetOpDesc()->GetOppImplVersion()));
  FE_CHECK(space_registry == nullptr,
           FE_LOGW("Node[%s, %s] space_registry is null.", node->GetNamePtr(), node->GetTypePtr()),
           return false);
  auto funcs = space_registry->GetOpImpl(node->GetType().c_str());
  FE_CHECK(funcs == nullptr,
           FE_LOGW("Node[%s, %s] funcs is null.", node->GetNamePtr(), node->GetTypePtr()),
           return false);
  FE_CHECK(funcs->op_select_format == nullptr,
           FE_LOGW("Node[%s, %s] op_select_format func is null.", node->GetNamePtr(), node->GetTypePtr()),
           return false);
  ge::AscendString result;
  if (funcs->op_select_format(op_check_content, result) != ge::GRAPH_SUCCESS) {
    FE_LOGW("Node[%s, %s]: can not call op_select_format func.", node->GetNamePtr(), node->GetTypePtr());
    return false;
  }
  op_select_format_str = result.GetString();
  return true;
}

void TbeOpStoreAdapter::SetPrebuildPattern(const OpKernelInfoPtr &op_kernel_info_ptr, te::TbeOpInfo &op_info) const {
  GetAndSetOpsPathNamePrefix(op_kernel_info_ptr, op_info);
  std::string prebuild_pattern = op_kernel_info_ptr->GetPrebuildPattern();
  if (prebuild_pattern == kStrDynamic || prebuild_pattern == kFunc) {
    if (GetOpSpecificInfoByKey(op_info, kStrPrebuildPattern, prebuild_pattern) != SUCCESS) {
      FE_LOGW("[TbeOpStoreAdapter][SetPrebuildPattern] get prebuild pattern failed from tbe function.");
    }
  }
  FE_LOGD("[TbeOpStoreAdapter] Prebuild pattern from op store is %s.",
          prebuild_pattern.empty() ? "undefined" : prebuild_pattern.c_str());
  op_info.SetPrebuildPattern(prebuild_pattern);
}

void TbeOpStoreAdapter::SetOpSpecificInfoToTbeOpInfo(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                     te::TbeOpInfo &op_info) const {
  SetPrebuildPattern(op_kernel_info_ptr, op_info);
}

Status TbeOpStoreAdapter::GetLXOpCoreType(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                                          const bool &is_dynamic_impl, std::string &lx_op_core_type_str) {
  string op_name = node->GetOpDesc()->GetName();
  string op_type = node->GetOpDesc()->GetType();
  FE_LOGD("Node[%s, %s]: start to GetLXOpCoreType.", op_name.c_str(), op_type.c_str());
  std::string op_dsl_file_path;
  if (!GetOpDslFilePath(node->GetOpDesc(), op_kernel_info_ptr, op_dsl_file_path)) {
    REPORT_FE_ERROR("[GraphOpt][Setcheck][GetLXOpCoreType][%s, %s] Failed to get op dsl file path.",
                    op_name.c_str(), op_type.c_str());
    return FAILED;
  }

  te::TbeOpInfo tbe_op_info(op_name, op_dsl_file_path, op_type, engine_name_);
  tbe_op_info.SetDynamicImpl(is_dynamic_impl);
  // 2. assemble the information
  if (tbe_info_assembler_ptr_->AssembleTbeInfo(node, op_kernel_info_ptr, engine_name_, tbe_op_info) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Setcheck][GetLXOpCoreType][Op %s, type %s] failed to assemble_tbe_info.", op_name.c_str(),
                    op_type.c_str());
    return FAILED;
  }
  SetOpSpecificInfoToTbeOpInfo(op_kernel_info_ptr, tbe_op_info);

  // 3. call the function of TeFusion
  if (GetOpSpecificInfoByKey(tbe_op_info, CORE_TYPE_VALUE, lx_op_core_type_str) != SUCCESS) {
    FE_LOGW("[GraphOpt][Setcheck][GetLXOpCoreType][Op %s, type %s] Failed to call op core type function.",
            op_name.c_str(), op_type.c_str());
    return FAILED;
  }

  FE_LOGD("Node[%s, %s]: end to GetLXOpCoreType. The lx_op_core_type_str is %s.", op_name.c_str(),
          op_type.c_str(), lx_op_core_type_str.c_str());
  return SUCCESS;
}

Status TbeOpStoreAdapter::GetDynamicPromoteType(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                                                std::string &promote_str) const {
  FE_CHECK_NOTNULL(node);
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  FE_LOGD("Node[%s, %s]: start to GetDynamicPromoteType.", op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());

  std::string op_dsl_file_path;
  if (!GetOpDslFilePath(op_desc_ptr, op_kernel_info_ptr, op_dsl_file_path)) {
    REPORT_FE_ERROR("[GraphOpt][Setcheck][SetTbeOpSliceInfo][%s, %s] Failed to get op dsl file path.",
                    op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());
    return FAILED;
  }

  te::TbeOpInfo op_info(op_desc_ptr->GetName(), op_dsl_file_path, op_desc_ptr->GetType(), engine_name_);
  GetAndSetOpsPathNamePrefix(op_kernel_info_ptr, op_info);
  if (tbe_info_assembler_ptr_->AssembleTbeInfo(node, op_kernel_info_ptr, engine_name_, op_info) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][CheckSupport][AssembleTbeInfo] Failed to Assemble tbe info for node %s type %s.",
                    op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());
    return FAILED;
  }

  if (GetOpSpecificInfoByKey(op_info, kStrPromoteType, promote_str) != SUCCESS) {
    FE_LOGW("[GetDynamicPromoteType] Node[%s] get promoteType from op specific info failed.",
            op_desc_ptr->GetNamePtr());
    return FAILED;
  }
  return SUCCESS;
}

Status TbeOpStoreAdapter::SelectOpFormat(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                                         const bool &is_dynamic_impl, const HeavyFormatInfo &heavy_format_info,
                                         std::string &op_format_dtype_str) {
  auto op_desc = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  string op_name = op_desc->GetName();
  string op_type = op_desc->GetType();
  FE_LOGD("Node[%s, %s]: start to SelectOpFormat.", op_name.c_str(), op_type.c_str());

  std::string op_dsl_file_path;
  if (!GetOpDslFilePath(op_desc, op_kernel_info_ptr, op_dsl_file_path)) {
    REPORT_FE_ERROR("[GraphOpt][Setcheck][SelectOpFormat][%s, %s] Failed to get op dsl file path.",
                    op_name.c_str(), op_type.c_str());
    return FAILED;
  }

  te::TbeOpInfo tbe_op_info(op_name, op_dsl_file_path, op_type, engine_name_);
  tbe_op_info.SetDynamicImpl(is_dynamic_impl);
  // 2. assemble the information
  if (tbe_info_assembler_ptr_->AssembleTbeInfo(node, op_kernel_info_ptr, heavy_format_info, engine_name_,
                                               tbe_op_info) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Setcheck][SeleFormat][Op %s, type %s] failed to assemble_tbe_info.", op_name.c_str(),
                    op_type.c_str());
    return FAILED;
  }
  tbe_op_info.SetNode(node);
  SetOpSpecificInfoToTbeOpInfo(op_kernel_info_ptr, tbe_op_info);

  // 3. call the function of TeFusion
  FE_CHECK(SelectTbeOpFormat == nullptr,
           REPORT_FE_ERROR("[GraphOpt][Setcheck][SeleFormat] Node[%s, %s]: the function SelectTbeOpFormat is nullptr.",
                           op_name.c_str(), op_type.c_str()),
           return FAILED);
  if (!SelectTbeOpFormat(tbe_op_info, op_format_dtype_str)) {
    FE_LOGW("[GraphOpt][Setcheck][SeleFormat][Op %s, type %s] Failed to call the op select format function.",
            op_name.c_str(), op_type.c_str());
    return FAILED;
  }

  FE_LOGD("Node[%s, %s]: end to SelectOpFormat. The op_format_dtype_str is %s.", op_desc->GetName().c_str(),
          op_desc->GetType().c_str(), op_format_dtype_str.c_str());
  return SUCCESS;
}

void TbeOpStoreAdapter::SetOpCompileInfo(const ge::OpDescPtr &op_desc_ptr, const std::vector<ge::Node *> &nodes) {
  std::string op_compile_info_json;
  ge::AttrUtils::GetStr(op_desc_ptr, COMPILE_INFO_JSON, op_compile_info_json);
  FE_LOGD("Compile info json after compiling is [%s].", op_compile_info_json.c_str());
  std::string op_compile_info_key;
  ge::AttrUtils::GetStr(op_desc_ptr, COMPILE_INFO_KEY, op_compile_info_key);
  FE_LOGD("Compile info key after compiling is [%s].", op_compile_info_key.c_str());

  for (const ge::Node *node : nodes) {
    if (node == nullptr) {
      continue;
    }
    if (!op_compile_info_json.empty()) {
      (void)ge::AttrUtils::SetStr(node->GetOpDesc(), COMPILE_INFO_JSON, op_compile_info_json);
      FE_LOGD("Set [%s] attr [%s] for node[%s, %s].", COMPILE_INFO_JSON.c_str(), op_compile_info_json.c_str(),
              node->GetOpDesc()->GetName().c_str(), node->GetOpDesc()->GetType().c_str());
    }
    if (!op_compile_info_key.empty()) {
      (void)ge::AttrUtils::SetStr(node->GetOpDesc(), COMPILE_INFO_KEY, op_compile_info_key);
      FE_LOGD("Set [%s] attr [%s] for node[%s, %s].", COMPILE_INFO_KEY.c_str(), op_compile_info_key.c_str(),
              node->GetOpDesc()->GetName().c_str(), node->GetOpDesc()->GetType().c_str());
    }
  }
}

void TbeOpStoreAdapter::SetOpTilingKey(std::vector<ge::Node *> &nodes, const ge::OpDescPtr &op_desc_ptr) const {
  for (auto &node : nodes) {
    ge::OpDescPtr cur_op_desc_ptr = node->GetOpDesc();
    if (cur_op_desc_ptr == nullptr) {
      FE_LOGW("Op desc is null");
      continue;
    }
    string cur_op_type = cur_op_desc_ptr->GetType();
    string cur_op_name = cur_op_desc_ptr->GetName();
    std::string op_tiling_key;
    if (ge::AttrUtils::GetStr(op_desc_ptr, kTilingRemoveDuplicates, op_tiling_key)) {
      FE_LOGD("Node[%s,%s] op tiling key is:%s", cur_op_name.c_str(), cur_op_type.c_str(), op_tiling_key.c_str());
      (void)ge::AttrUtils::SetStr(cur_op_desc_ptr, kTilingRemoveDuplicates, op_tiling_key);
    } else {
      FE_LOGD("Node[%s, %s]: Can not find op tiling key after compiling.", cur_op_name.c_str(),
              cur_op_type.c_str());
    }
  }
}

/* 1. If one thread of the node is failed and the node is optimized by lx-fusion,
 * we add it into vector need_rollback_nodes. (Although sgt slicing is conflict with
 * lx-fusion, we still use a set to store all to-be-rolled-back nodes to remove
 * duplicates.)
 * 2. If this node is not optimized by lx-fusion, we remove all other duplicated
 * failed tasks.
 * When we re-compile single op, for one sgt-sliced node, we will separate it
 * into several(two) tasks. */
Status TbeOpStoreAdapter::GetSgtSliceTaskRollbackNode(CompileTaskPara &task_para,
                                                      std::vector<ge::NodePtr> &need_rollback_nodes) const {
  if (task_para.failed_tasks.empty()) {
    return SUCCESS;
  }

  auto &pre_scope_id_map = task_para.task_scope_id_map;
  auto &failed_tasks = task_para.failed_tasks;
  unordered_set<ge::NodePtr> del_nodes;
  unordered_set<uint64_t> all_tasks_related_to_failed_task;

  for (auto task_itr = failed_tasks.begin(); task_itr != failed_tasks.end();) {
    auto task_id = task_itr->first;
    if (pre_scope_id_map.find(task_id) == pre_scope_id_map.end()) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][GetSgtSliceTaskRlb] Tid[%lu], not find taskId[%lu].", GetCurThreadId(),
                      task_id);
      return FAILED;
    }

    int64_t scope_id = pre_scope_id_map[task_id];
    auto &scope_task_ids_map = task_para.scope_task_ids_map;
    // if not find in scope_task_ids_map, it is a no slice node(normal node)
    // only slice nodes can be set to scope_task_ids_map in function named SetSgtSliceTaskToTeFusion
    // slice scopeid has 2 tasks at least, normal scopeid does not need to be rollbacked, only for protect
    auto task_ids = scope_task_ids_map.find(scope_id);
    if (task_ids == scope_task_ids_map.end() || task_ids->second.size() == 1) {
      FE_LOGD("Tid[%lu], not find scope_id[%ld].", GetCurThreadId(), scope_id);
      task_itr++;
      continue;
    }

    // record all failed tasks for deleting from task_para.succ_tasks
    all_tasks_related_to_failed_task.insert(task_ids->second.cbegin(), task_ids->second.cend());

    auto &all_nodes_in_one_task = (*task_para.fusion_nodes_map)[scope_id];
    bool is_optimized_by_lxfusion = IsBuffFusOptimizedNodes(all_nodes_in_one_task);

    if (is_optimized_by_lxfusion) {
      // all nodes(in this scope) optimized by lxfusion need to be rolled back
      for (auto &op : all_nodes_in_one_task) {
        if (del_nodes.find(op->shared_from_this()) == del_nodes.end()) {
          del_nodes.insert(op->shared_from_this());
          FE_LOGD("Delete op name: %s, type: %s.", op->GetOpDesc()->GetName().c_str(), op->GetType().c_str());
        }
      }
      task_itr = failed_tasks.erase(task_itr);
    } else {
      //
      auto scope_able_to_del = task_para.failed_task_able_to_delete.find(scope_id);
      if (scope_able_to_del != task_para.failed_task_able_to_delete.end() &&
          scope_able_to_del->second) {
        FE_LOGD("Delete task %lu for scope %ld", task_itr->first, scope_id);
        task_itr = failed_tasks.erase(task_itr);
      } else {
        FE_LOGD("Do not delete task %lu for scope %ld. Set task to true.", task_itr->first, scope_id);
        task_para.failed_task_able_to_delete[scope_id] = true;
        task_itr++;
      }
    }
  }

  // succ_task needs to delete relative failed tasks
  // can be optimized
  for (auto &task_id : all_tasks_related_to_failed_task) {
    if (task_para.succ_tasks.find(task_id) != task_para.succ_tasks.end()) {
      FE_LOGD("Delete task_id %lu because one of its peer tasks unsuccessful.", task_id);
      task_para.succ_tasks.erase(task_id);
    }
  }

  need_rollback_nodes.insert(need_rollback_nodes.cend(), del_nodes.cbegin(), del_nodes.cend());
  return SUCCESS;
}

Status TbeOpStoreAdapter::SetSgtTensorSliceInfoToNodes(std::vector<ge::Node*> &compile_nodes,
                                                       int32_t thread_idx) const {
  for (const auto &node : compile_nodes) {
    ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
    slice_info_ptr = node->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
    if (slice_info_ptr == nullptr) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][SetSgtTensSliceInfo] This node has no slice info. op_type:%s, op_name:%s.",
                      node->GetType().c_str(), node->GetName().c_str());
      return FAILED;
    }
    // set slice shape to tensor
    auto input_tensors = node->GetOpDesc()->GetAllInputsDescPtr();
    FE_LOGD("Node[%s]: set input slice attribute.", node->GetName().c_str());
    for (size_t i = 0; i < input_tensors.size(); i++) {
      SetSgtSliceShaeForEachTensor(i, thread_idx, node, input_tensors,
          slice_info_ptr->input_tensor_slice,
          slice_info_ptr->ori_input_tensor_shape, ATTR_NAME_SGT_SLICE_SHAPE);
      SetSgtSliceShaeForEachTensor(i, thread_idx, node, input_tensors,
          slice_info_ptr->ori_input_tensor_slice,
          slice_info_ptr->ori_input_tensor_shape, ATTR_NAME_SGT_ORI_SLICE_SHAPE);
    }

    auto output_tensors = node->GetOpDesc()->GetAllOutputsDescPtr();
    FE_LOGD("Node[%s]: set output slice attribute.", node->GetName().c_str());
    for (size_t i = 0; i < output_tensors.size(); i++) {
      SetSgtSliceShaeForEachTensor(i, thread_idx, node, output_tensors,
          slice_info_ptr->output_tensor_slice,
          slice_info_ptr->ori_output_tensor_shape, ATTR_NAME_SGT_SLICE_SHAPE);
      SetSgtSliceShaeForEachTensor(i, thread_idx, node, output_tensors,
          slice_info_ptr->ori_output_tensor_slice,
          slice_info_ptr->ori_output_tensor_shape, ATTR_NAME_SGT_ORI_SLICE_SHAPE);
    }
  }
  return SUCCESS;
}

Status TbeOpStoreAdapter::SetTaskForOneScope(std::vector<ge::Node *> &nodes,
                                             const int64_t scope_id,
                                             const std::vector<ge::NodePtr> &to_del_nodes,
                                             CompileTaskPara &task_para,
                                             const CompileStrategy &compile_strategy) {
  ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
  slice_info_ptr = nodes[0]->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  string first_node_name = nodes[0]->GetName();
  // normal nodes
  if (!OpIsAutoThread(slice_info_ptr)) { // normal op or manual mode
    task_para.task_num++;
    uint64_t taskId = GetAtomicId();
    task_para.task_scope_id_map.insert(std::make_pair(taskId, scope_id));
    FE_LOGD("%lu, taskId %lu , scope_id %ld, set compile %s task.", GetCurThreadId(), taskId, scope_id,
            first_node_name.c_str());
    // set compile task
    const bool is_fusion_check = false;
    if (SetTeTask(nodes, taskId, to_del_nodes, compile_strategy, is_fusion_check) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][SetSgtSliceTask] The op[%s] set compile task failed",
                      first_node_name.c_str());
      return FAILED;
    }
  } else { // slice nodes
    // slice nodes need to clear attribute ATTR_NAME_SGT_SLICE_SHAPE
    // and ATTR_NAME_SGT_ORI_SLICE_SHAPE
    ClearSgtAttr(nodes);

    int32_t slice_size = static_cast<int32_t>(slice_info_ptr->input_tensor_slice.size());
    uint64_t slice_shape_index = 0;
    FE_LOGD("Set slice(size: %d) shape for scope %ld, first node %s.", slice_size, scope_id, first_node_name.c_str());

    for (auto &scope_task_ids : task_para.scope_task_ids_map) {
      FE_LOGD("%s.", StringUtils::IntegerVecToString(scope_task_ids.second).c_str());
    }
    for (int32_t i = 0; i < slice_size; i++) {
      if (i != 0 && i != (slice_size - 1)) {
        /* Here we only care the head slice and the tail slice because all
         * middle slices are same. */
        continue;
      }
      // every thread slice needs to be compiled once
      (void)SetSgtTensorSliceInfoToNodes(nodes, i);
      task_para.task_num++;
      uint64_t taskId = GetAtomicId();

      task_para.task_scope_id_map.insert(std::make_pair(taskId, scope_id));
      auto scope_task_iter = task_para.scope_task_ids_map.find(scope_id);
      if (scope_task_iter != task_para.scope_task_ids_map.end()) {
        scope_task_iter->second.emplace_back(taskId);
        FE_LOGI("Slice size is %d.", slice_size);
      } else {
        vector<uint64_t> task_id_vec = {taskId};
        task_para.scope_task_ids_map.emplace(scope_id, task_id_vec);
      }

      FE_LOGD("%lu, taskId %lu, scope_id %ld, set slice %d compile %s task.", GetCurThreadId(), taskId,
              scope_id, i, first_node_name.c_str());

      // Before compilation, we need to give all nodes a thread node
      // name to find all the precomp information.
      vector<string> old_names;
      SetThreadNodeName(nodes, old_names, i);
      // set compile task
      if (SgtSetTeTask(nodes, taskId, to_del_nodes,
                       compile_strategy, slice_shape_index) != SUCCESS) {
        REPORT_FE_ERROR("[SubGraphOpt][Compile][SetSgtSliceTask] The op[%s] set compile task failed.",
                        first_node_name.c_str());
        SetNameForNodes(nodes, old_names);
        return FAILED;
      }
      SetNameForNodes(nodes, old_names);
      slice_shape_index++;
    }
  }
  return SUCCESS;
}

Status TbeOpStoreAdapter::SetSgtSliceTaskToTeFusion(CompileTaskPara &task_para,
                                                    const std::vector<ge::NodePtr> &to_del_nodes) {
  for (auto &scope_task_ids : task_para.scope_task_ids_map) {
    FE_LOGD("Start read scope_task_ids_map, first: %ld, second: %s.", scope_task_ids.first,
            StringUtils::IntegerVecToString(scope_task_ids.second).c_str());
  }
  // iter: {scope id, node vector}
  for (auto &iter : *task_para.fusion_nodes_map) {
    if (SetTaskForOneScope(iter.second, iter.first, to_del_nodes, task_para,
                           CompileStrategy::COMPILE_STRATEGY_OP_SPEC) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][SetTsk]Failed to set task for scope %ld with first node %s.",
                      iter.first, iter.second[0]->GetName().c_str());
      return FAILED;
    }
  }

  for (auto &scope_task_ids : task_para.scope_task_ids_map) {
    FE_LOGD("End read scope_task_ids_map, first: %ld, second: %s.", scope_task_ids.first,
            StringUtils::IntegerVecToString(scope_task_ids.second).c_str());
  }
  return SUCCESS;
}

Status TbeOpStoreAdapter::ProcessSuccSgtSliceTask(CompileTaskPara &task_para) const {
  auto &scope_task_ids_map = task_para.scope_task_ids_map;
  FE_LOGD("The size of scope_task_ids_map is %zu.", scope_task_ids_map.size());
  for (auto &scope_tasks_pair : scope_task_ids_map) {
    int64_t scope_id = scope_tasks_pair.first;
    vector<uint64_t> &task_ids = scope_tasks_pair.second;
    FE_LOGD("All task id for scope id %ld is %s.", scope_id, StringUtils::IntegerVecToString(task_ids).c_str());
    // filter failed tasks
    if (task_para.succ_tasks.find(task_ids.at(0)) == task_para.succ_tasks.end()) {
      continue;
    }

    // set every json path and compileinfo
    for (auto &task_id : task_ids) {
      auto fin_task_itr = task_para.succ_tasks.find(task_id);
      if (fin_task_itr == task_para.succ_tasks.end()) {
        REPORT_FE_ERROR("[SubGraphOpt][Compile][ProcSucSgtSlcTsk] Thread[%lu], Task[%lu]: not find in successful tasks",
                        GetCurThreadId(), task_id);
        return FAILED;
      }
      FE_LOGD("Process sgt task with first node %s.", fin_task_itr->second.teNodeOpDesc->GetName().c_str());

      FE_LOGD("tid[%lu], get taskId[%lu], scope_id[%ld]", GetCurThreadId(), task_id, scope_id);
      if (SetOpCompileResult(scope_id, fin_task_itr->second.teNodeOpDesc, false,
                             *task_para.compile_ret_map) == FAILED) {
        REPORT_FE_ERROR("[SubGraphOpt][Compile][ProcSucSgtSlcTsk] %s set op json path failed.",
                        (*task_para.fusion_nodes_map)[scope_id][0]->GetName().c_str());
        return FAILED;
      }
      task_para.succ_tasks.erase(task_id);
    }
  }
  FE_LOGD("Process sgt success task_num[%zu], tid[%lu].", task_para.succ_tasks.size(), GetCurThreadId());
  return SUCCESS;
}

Status TbeOpStoreAdapter::CompileMultiKernelSliceOp(ScopeNodeIdMap &fusion_nodes_map, CompileResultMap &compile_ret_map,
                                                    std::vector<ge::NodePtr> &compile_failed_nodes,
                                                    const std::vector<ge::NodePtr> &to_del_nodes) {
  FE_CHECK(TeFusionV == nullptr, REPORT_FE_ERROR("[SubGraphOpt][Compile][CompSgtSliceOp] TeFusionV is nullptr."),
           return FAILED);
  FE_TIMECOST_START(TeFusionV);
  CompileTaskPara task_para(0, false, &compile_ret_map, &fusion_nodes_map);

  if (SetSgtSliceTaskToTeFusion(task_para, to_del_nodes) != SUCCESS) {
    return FAILED;
  }

  FE_LOGD("Thread[%lu], setting %lu tasks to compile.", GetCurThreadId(), task_para.task_num);

  // wait for finish
  FE_TIMECOST_START(WaitTaskFinish);
  if (WaitTaskFinish(task_para) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][CompSgtSliceOp] Thread[%lu] wait task finish failed", GetCurThreadId());
    return FAILED;
  }
  FE_TIMECOST_END(WaitTaskFinish, "CompileMultiKernelSliceOp.WaitTaskFinish");
  // get need recovered nodes
  if (GetSgtSliceTaskRollbackNode(task_para, compile_failed_nodes) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][CompSgtSliceOp] Thread[%lu] get recover task failed", GetCurThreadId());
    return FAILED;
  }

  // process success slice task
  if (ProcessSuccSgtSliceTask(task_para) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][CompSgtSliceOp] Thread[%lu] failed to process successful sgt task.",
                    GetCurThreadId());
    return FAILED;
  }

  // process success normal task
  if (ProcessSuccCompileTask(task_para) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][CompSgtSliceOp] Thread[%lu] failed when processing the task that had been successfully compiled.",
                    GetCurThreadId());
    return FAILED;
  }
  // failed tasks with no slicemap need to be recompiled
  if (!task_para.failed_tasks.empty()) {
    SaveMsTuneErrorMsg(task_para);
    if (RetryCompileFailOp(task_para) == FAILED) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][CompSgtSliceOp] Thread[%lu] failed to retry compile fail op.",
                      GetCurThreadId());
      return FAILED;
    }
  }

  FE_TIMECOST_END(TeFusionV, "TeFusion during FEGraphOptimizer::OptimizeFusedGraph");
  FE_LOGI("TbeOpStoreAdapter::Compile Op successfully, tid:%lu.", GetCurThreadId());

  return SUCCESS;
}

Status TbeOpStoreAdapter::SetTbeOpSliceInfo(const ge::NodePtr &node_ptr, OpKernelInfoPtr &op_kernel_info_ptr) {
  FE_CHECK_NOTNULL(node_ptr);
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);

  string op_name = op_desc_ptr->GetName();
  string op_type = op_desc_ptr->GetType();

  FE_LOGD("[TbeAdapter][SetTbeOpSliceInfo][Node %s type %s] Start to set tbe op slice info.", op_name.c_str(),
          op_type.c_str());
  std::string op_dsl_file_path;
  if (!GetOpDslFilePath(op_desc_ptr, op_kernel_info_ptr, op_dsl_file_path)) {
    REPORT_FE_ERROR("[GraphOpt][Setcheck][SetTbeOpSliceInfo][%s, %s] Failed to get op dsl file path.",
                    op_name.c_str(), op_type.c_str());
    return FAILED;
  }

  te::TbeOpInfo op_info(op_name, op_dsl_file_path, op_type, engine_name_);
  GetAndSetOpsPathNamePrefix(op_kernel_info_ptr, op_info);
  if (tbe_info_assembler_ptr_->AssembleTbeInfo(node_ptr, op_kernel_info_ptr, engine_name_, op_info) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][CheckSupport][AssembleTbeInfo] Failed to Assemble tbe info for node %s type %s.",
                    op_name.c_str(), op_type.c_str());
    return SUCCESS;
  }
  SetOpSpecificInfoToTbeOpInfo(op_kernel_info_ptr, op_info);

  if (UnknownShapeUtils::IsUnknownShapeOp(*op_desc_ptr)) {
    op_info.SetIsUnknownShape(true);
  }
  bool is_dynamic_impl = IsOpDynamicImpl(op_desc_ptr);
  op_info.SetDynamicImpl(is_dynamic_impl);
  FE_CHECK(GetOpInfo == nullptr,
           REPORT_FE_ERROR("[GraphOpt][CheckSupport] Function CheckTbeSupported of TeFusion is nullptr."),
           return SUCCESS);
  op_info.SetNode(node_ptr);
  string op_slice_info_str;
  Status status = GetOpInfo(op_info, op_slice_info_str);
  if (status == te::LX_QUERY_SUCC) {
    (void)ge::AttrUtils::SetStr(op_desc_ptr, OP_SLICE_INFO, op_slice_info_str);
    FE_LOGD("Obtain slice info %s from tbe api for node[%s].", op_slice_info_str.c_str(),
            op_name.c_str());
  } else {
    FE_LOGD("Not obtain slice info from tbe api for node[%s].", op_name.c_str());
  }
  return SUCCESS;
}

Status TbeOpStoreAdapter::GeneralizeNode(const ge::NodePtr &node, const te::TbeOpInfo &op_info,
    te::TE_GENERALIZE_TYPE generalize_type) {
  FE_LOGI("Begin to generalize node[%s, %s].", node->GetName().c_str(), node->GetType().c_str());
  auto op_desc = node->GetOpDesc();
  FE_CHECK(op_desc == nullptr,
           FE_LOGW("[GraphOptimizePrepare][ShapeAndValueGeneralize][GeneralizeGraph] Thread[%lu]: failed to get op_desc.",
                   GetCurThreadId()),
           return FAILED);

  node->GetOpDesc()->DelAttr(ATTR_NAME_UNKNOWN_SHAPE);
  FE_LOGD("Begin to run function[TeGeneralize], node[%s, %s].", node->GetName().c_str(), node->GetType().c_str());
  if (!TeGeneralize(op_info, generalize_type, node)) {
    FE_LOGW("[GraphOptimizePrepare][ShapeAndValueGeneralize][GeneralizeGraph] Node[%s]: failed to generalize node.",
            op_desc->GetType().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status TbeOpStoreAdapter::GetRangeLimitType(const ge::NodePtr &node_ptr, const te::TbeOpInfo &tbe_op_info,
                                            bool &is_limited) const {
  std::string node_name = node_ptr->GetName();
  std::string limit_type;
  if (GetOpSpecificInfoByKey(tbe_op_info, kStrRangeLimit, limit_type) != SUCCESS) {
    FE_LOGW("[AssembleTbeInfo][SetPrebuildPattern] Node[%s] get limit_type from op specific info failed.",
            node_name.c_str());
    return FAILED;
  }

  if (limit_type == "limited") {
    is_limited = true;
  } else if (limit_type == "unlimited") {
    is_limited = false;
  } else {
    FE_LOGW("[GraphOptimizePrepare][Generalize] node[%s] rangeLimit[%s] from tbe is invalid.",
            node_name.c_str(), limit_type.c_str());
    is_limited = false;
  }
  return SUCCESS;
}

Status TbeOpStoreAdapter::LimitedNodesCheck(bool &is_support, const te::TbeOpInfo &tbe_op_info,
    std::vector<size_t> &upper_limited_input_indexs, std::vector<size_t> &lower_limited_input_indexs) {
  if (!DynamicShapeRangeCheck(tbe_op_info, is_support, upper_limited_input_indexs, lower_limited_input_indexs)) {
    return FAILED;
  }
  return SUCCESS;
}

Status TbeOpStoreAdapter::IsGeneralizeFuncRegistered(bool &is_registered, const te::TbeOpInfo &op_info) {
  FE_CHECK(!CheckIsTbeGeneralizeFuncRegistered(op_info, is_registered),
           FE_LOGW("%s Thread[%lu] failed to check whether it is registered.", kStageGeneralizeGraph.c_str(),
                   GetCurThreadId()),
           return FAILED);
  return SUCCESS;
}

bool TbeOpStoreAdapter::GetOpDslFilePath(const ge::OpDescPtr &op_desc, const OpKernelInfoPtr &op_kernel_info_ptr,
                                         std::string &op_dsl_file_path) const {
  if (op_desc == nullptr || op_kernel_info_ptr == nullptr) {
    return false;
  }

  OpImplType impl_type = op_kernel_info_ptr->GetOpStoreImplType();
  FEOpsStoreInfo op_store_info;
  if (Configuration::Instance(engine_name_).GetOpStoreInfoByImplType(impl_type, op_store_info) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Setcheck][GetOpInfo][Op %s, type %s] failed to get op store info by impl_type [%ld].",
                    op_desc->GetName().c_str(), op_desc->GetType().c_str(), impl_type);
    return false;
  }

  bool is_custom_op = false;
  (void)ge::AttrUtils::GetBool(op_desc, NON_PERSISTENT_CUSTOM_OP_FLAG, is_custom_op);
  bool ret_status = (is_custom_op && op_kernel_info_ptr != nullptr && !op_kernel_info_ptr->GetOpImpPath().empty());
  if (ret_status) {
    op_dsl_file_path = op_kernel_info_ptr->GetOpImpPath();
  } else {
    op_dsl_file_path = op_store_info.op_impl_file_path;
  }
  FE_LOGD("Op dsl file path of op[%s, %s] is [%s].",
          op_desc->GetName().c_str(), op_desc->GetType().c_str(), op_dsl_file_path.c_str());
  return true;
}

void TbeOpStoreAdapter::GetAndSetOpsPathNamePrefix(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                   te::TbeOpInfo &tbe_op_info) const {
  if (op_kernel_info_ptr == nullptr) {
    return;
  }
  std::string ops_path_name_prefix = op_kernel_info_ptr->GetOpsPathNamePrefix();
  tbe_op_info.SetOpsPathNamePrefix(ops_path_name_prefix);
}

Status TbeOpStoreAdapter::GetOpUniqueKeys(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                                          std::vector<std::string> &op_unique_keys) {
  FE_CHECK_NOTNULL(node);
  const ge::OpDescPtr op_desc = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  string op_name = op_desc->GetName();
  string op_type = op_desc->GetType();
  FE_LOGD("Op[name=%s,type=%s]: start to GetOpUniqueKey.", op_name.c_str(), op_type.c_str());
  std::string op_dsl_file_path;
  if (!GetOpDslFilePath(op_desc, op_kernel_info_ptr, op_dsl_file_path)) {
    REPORT_FE_ERROR("[GraphOpt][Setcheck][GetOpUniqueKey][%s, %s] Failed to get op dsl file path.",
                    op_name.c_str(), op_type.c_str());
    return FAILED;
  }

  te::TbeOpInfo tbe_op_info(op_name, op_dsl_file_path, op_type, engine_name_);
  tbe_op_info.SetNode(node);
  tbe_op_info.SetDynamicImpl(IsOpDynamicImpl(op_desc));
  // 2. assemble the information
  if (tbe_info_assembler_ptr_->AssembleTbeInfo(node, op_kernel_info_ptr, engine_name_, tbe_op_info) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Setcheck][GetOpUniqueKey][%s, %s] Failed to assemble_tbe_info.",
                    op_name.c_str(), op_type.c_str());
    return FAILED;
  }
  SetOpSpecificInfoToTbeOpInfo(op_kernel_info_ptr, tbe_op_info);

  // 3. call the function of TeFusion
  FE_CHECK(GetOpUniqueKeyFunc == nullptr,
           REPORT_FE_ERROR("[GraphOpt][Setcheck][GetOpUniqueKey] Function GetOpUniqueKey is null."),
           return FAILED);
  if (!GetOpUniqueKeyFunc(tbe_op_info, op_unique_keys)) {
    FE_LOGW("[GraphOpt][Setcheck][GetOpUniqueKey][Op %s, type %s] Failed to call GetOpUniqueKey function.",
            op_name.c_str(), op_type.c_str());
    return FAILED;
  }

  FE_LOGD("End of GetOpUniqueKey, op unique key size of op[%s, %s] is [%zu].",
          op_desc->GetName().c_str(), op_desc->GetType().c_str(), op_unique_keys.size());
  return SUCCESS;
}

Status TbeOpStoreAdapter::FeedNodeGeneralInfo(const ge::NodePtr &node_ptr,
                                              NodeGeneralInfoPtr &node_info_ptr) const {
  Status ret = FeedNodeGeneralInfoFromOpStore(node_ptr, node_info_ptr);
  if (ret != SUCCESS) {
    FE_LOGD("Node[name:%s, type;%s] get general info unsuccessful",
            node_ptr->GetName().c_str(), node_ptr->GetType().c_str());
    return ret;
  }
  return GetRangeLimit(node_info_ptr, node_ptr);
}

Status TbeOpStoreAdapter::FeedNodeGeneralInfoFromOpStore(const ge::NodePtr &node_ptr,
                                                         NodeGeneralInfoPtr &node_info_ptr) const {
  FE_LOGD("Node[%s, %s] begin to check support dynamic shape.", node_ptr->GetNamePtr(), node_ptr->GetTypePtr());
  const std::vector<FEOpsStoreInfo> &fe_ops_store_info_vec = Configuration::Instance(engine_name_).GetOpsStoreInfo();
  std::ostringstream reason_oss;
  TbeOpInfoPtr op_info_ptr;
  OpKernelInfoPtr op_kernel_ptr = nullptr;
  te::TbeOpInfo tbe_op_info_bk("", "", "", "");
  for (auto &ops_store : fe_ops_store_info_vec) {
    if (!CheckIsInnerOpStore(ops_store)) {
      continue;
    }
    UnSupportedReason sub_store_reason;
    OpStoreAdapterPtr op_store_adapter = nullptr;
    op_kernel_ptr = OpsKernelManager::Instance(engine_name_)
        .GetOpKernelInfoByOpType(ops_store.fe_ops_store_name, node_ptr->GetType());
    if (op_kernel_ptr == nullptr) {
      continue;
    }
    FE_LOGD("GetOpKernel successfully, node[%s, %s].", node_ptr->GetName().c_str(), node_ptr->GetType().c_str());

    std::string op_dsl_file_path;
    bool ret_status = (op_kernel_ptr != nullptr && !op_kernel_ptr->GetOpImpPath().empty());
    if (ret_status) {
      op_dsl_file_path = op_kernel_ptr->GetOpImpPath();
    } else {
      op_dsl_file_path = ops_store.op_impl_file_path;
    }

    FE_MAKE_SHARED(op_info_ptr = std::make_shared<te::TbeOpInfo>(node_ptr->GetName(), op_dsl_file_path,
                                                                 node_ptr->GetType(), GetCoreType(engine_name_)),
                   return OP_STORE_MAKE_SHARED_FAILED);
    if (tbe_info_assembler_ptr_->AssembleTbeInfo(node_ptr, op_kernel_ptr, engine_name_, *op_info_ptr) != SUCCESS) {
      FE_LOGW("[GraphOpt][ShapeAndValueGeneralize][CheckIsGeneralizableGraph] Node[%s, %s]: failed to assemble tbe info.",
              node_ptr->GetName().c_str(), node_ptr->GetType().c_str());
      return INTERNAL_ERROR;
    }
    SetOpSpecificInfoToTbeOpInfo(op_kernel_ptr, *op_info_ptr);
    FE_LOGD("AssembleTbeInfo finished, node[%s, %s].", node_ptr->GetName().c_str(), node_ptr->GetType().c_str());

    node_info_ptr->is_found_in_opstore = true;
    node_info_ptr->op_kernel = op_kernel_ptr;
    node_info_ptr->op_info = op_info_ptr;
    node_info_ptr->is_support_dynamic_shape = op_kernel_ptr->IsSupportDynamicShape();
    return SUCCESS;
  }
  FE_MAKE_SHARED(op_info_ptr = std::make_shared<te::TbeOpInfo>(tbe_op_info_bk),
                 return OP_STORE_MAKE_SHARED_FAILED);
  node_info_ptr->is_found_in_opstore = false;
  node_info_ptr->op_info = op_info_ptr;
  FE_LOGD("Could not found the op in opstores, tbeopinfo is default, node[%s, %s].",
          node_ptr->GetName().c_str(), node_ptr->GetType().c_str());
  return SUCCESS;
}

Status TbeOpStoreAdapter::UpdatePrebuildPattern() {
  FE_CHECK(QueryOpPattern == nullptr,
           REPORT_FE_ERROR("[Init][UpdatePrebuildPattern]The function QueryOpPattern of TeFusion is nullptr."),
           return FAILED);
  std::vector<std::pair<std::string, std::string>> op_prebuild_patterns;
  if (!QueryOpPattern(op_prebuild_patterns)) {
    FE_LOGW("[Init][UpdatePrebuildPattern] Failed to get op prebuild patterns from tefusion.");
    return INTERNAL_ERROR;
  }
  OpsKernelManager::Instance(engine_name_).UpdatePatternForAllKernel(op_prebuild_patterns);
  return SUCCESS;
}

void TbeOpStoreAdapter::GetAllCompileStatisticsMsg(std::vector<std::string> &statistics_msg) const {
  if (GetAllCompileStatistics == nullptr) {
    FE_LOGW("Function GetAllCompileStatistics is nullptr.");
    return;
  }
  GetAllCompileStatistics(statistics_msg);
  FE_LOGD("Get All compile statistics, size is [%zu].", statistics_msg.size());
}

bool TbeOpStoreAdapter::JudgeBuiltInOppKernelInstalled() const {
  FE_CHECK(IsOppKernelInstalled == nullptr,
           REPORT_FE_ERROR("[Init][UpdatePrebuildPattern]The function IsOppKernelInstalled of TeFusion is nullptr."),
           return FAILED);
  return IsOppKernelInstalled(false, static_cast<int64_t>(EN_IMPL_HW_TBE));
}

bool TbeOpStoreAdapter::IsNeedSkipOpJudge(const ge::NodePtr &node,
                                          [[maybe_unused]] const OpKernelInfoPtr &op_kernel_info_ptr) const {
  FE_CHECK(node == nullptr, FE_LOGE("Node is nullptr!"), return false);
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  FE_CHECK(op_desc_ptr == nullptr, FE_LOGE("Op desc is nullptr!"), return false);
  std::string op_type = node->GetType();
  bool tile_fwk_op_flag = false;
  if (TileFwkOpInfo::Instance().GetTileFwkOpFlag(op_type, tile_fwk_op_flag)) {
    if (tile_fwk_op_flag) {
      (void)ge::AttrUtils::SetBool(op_desc_ptr, kAttrTileFwkOpStr, true);
    }
    return tile_fwk_op_flag;
  }
  std::string op_select_format_str;
  if (!GetSelectOpFormat(node, op_select_format_str) || op_select_format_str.empty()) {
    FE_LOGW("Node[%s, %s]: can not get op_select_format_str.", op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());
    TileFwkOpInfo::Instance().SetTileFwkOpFlag(op_type, false);
    return false;
  }
  std::string tile_fwk_op_flag_str;
  if (ParseJsonByKey(op_select_format_str,  kTileFwkOpFlag, tile_fwk_op_flag_str) != SUCCESS) {
    FE_LOGW("Node[%s, %s]: can not get tileFwkOp from op_select_format_str[%s].",
            op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr(), op_select_format_str.c_str());
    return false;
  }
  FE_LOGD("Node[%s, %s]: tile_fwk_op_flag_str is %s.", op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr(),
          tile_fwk_op_flag_str.c_str());
  tile_fwk_op_flag = tile_fwk_op_flag_str == "true";
  if (tile_fwk_op_flag) {
    (void)ge::AttrUtils::SetBool(op_desc_ptr, kAttrTileFwkOpStr, true);
  }
  TileFwkOpInfo::Instance().SetTileFwkOpFlag(op_type, tile_fwk_op_flag);
  return tile_fwk_op_flag;
}

void TbeOpStoreAdapter::SetOpsKernelInfoStore(const std::shared_ptr<ge::OpsKernelInfoStore> ops_kernel_info_store_ptr) {
  fe_ops_kernel_info_store_ptr_ = ops_kernel_info_store_ptr;
}
}  // namespace fe

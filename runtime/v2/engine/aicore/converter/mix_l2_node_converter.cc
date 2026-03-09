/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicore_ffts_node_converter.h"
#include "aicore_node_converter.h"
#include "engine/node_converter_utils.h"
#include "engine/ffts_plus/converter/ffts_plus_common.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"
#include "register/ffts_node_calculater_registry.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "engine/aicore/fe_rt2_common.h"
#include "graph_builder/bg_memory.h"
#include "graph_builder/bg_tiling.h"
#include "graph_builder/bg_infer_shape.h"
#include "engine/aicore/converter/bg_kernel_launch.h"
#include "graph_builder/bg_platform.h"
#include "graph_builder/value_holder_generator.h"
#include "framework/common/ge_types.h"
#include "aicore_compile_results.h"
#include "exe_graph/runtime/tiling_context.h"
#include "engine/aicore/kernel/rt_ffts_plus_launch_args.h"
#include "engine/aicore/kernel/mixl2_update_kernel.h"
#include "engine/aicore/graph_builder/bg_aicore_memory.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/def_types.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph_builder/bg_condition.h"
#include "register/op_tiling_info.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"

namespace gert {
namespace {
  const char* kMixL2TaskInfoKey = "mix_l2";
  const std::string kMixDynamicRatio = "_mix_dynamic_ratio";
  const std::string kDynRatioAttr = "mix_tiling_with_ratio_attr";
  const std::string kDynRatioTiling = "mix_tiling_key";
  const std::string kDynRatioCRatio = "mix_tiling_c_ratio";
  const std::string kDynRatioVRatio = "mix_tiling_v_ratio";
  const std::string kTilingDataStr = "_tiling_data_str";

  struct MixProcArgs {
    std::vector<bg::ValueHolderPtr> output_shapes;
    std::vector<bg::ValueHolderPtr> tiling_ret;
    std::vector<bg::ValueHolderPtr> launch_arg;
    std::vector<bg::ValueHolderPtr> ordered_holders;
    bg::ValueHolderPtr args_para;
    std::vector<bg::ValueHolderPtr> task_info_para;
    bg::ValueHolderPtr atomic_launch{nullptr};
    bg::ValueHolderPtr workspaces{nullptr};
    bg::ValueHolderPtr block_dim{nullptr};
    bg::ValueHolderPtr schedule_mode{nullptr};
    bg::ValueHolderPtr tiling_key{nullptr};
    FFTSAllMemPara all_mem_para;
  };

  struct DynRatioInfos {
    bg::ValueHolderPtr tiling_key_holder;
    bg::ValueHolderPtr c_ratio_holder;
    bg::ValueHolderPtr v_ratio_holder;
  };

struct InsertOptArgs {
  bg::ValueHolderPtr empty_shape;
  bg::DevMemValueHolderPtr empty_val;
  InsertOptArgs(bg::ValueHolderPtr shape, bg::DevMemValueHolderPtr val): empty_shape(shape), empty_val(val){}
};

ge::Status UpdateContextByDynRatio(const ge::NodePtr &node, const MixProcArgs &proc_args,
                                   std::vector<bg::ValueHolderPtr> &inputs) {
  DynRatioInfos ratio_info;
  bool is_mix_ratio = false;
  if (!node->GetOpDesc()->HasAttr(kMixDynamicRatio)) {
    GELOGD("Not dynamic ratio type, back to mixl2");
    inputs.emplace_back(bg::ValueHolder::CreateConst(&is_mix_ratio, sizeof(is_mix_ratio)));
    return ge::SUCCESS;
  }
  GELOGD("Update dynamic ratio by tilingKey.");
  ge::GeAttrValue::NAMED_ATTRS tiling_with_ratio;
  if (!ge::AttrUtils::GetNamedAttrs(node->GetOpDesc(), kDynRatioAttr, tiling_with_ratio)) {
    return ge::FAILED;
  }

  std::vector<std::string> tiling_key_vec;
  tiling_with_ratio.GetItem(kDynRatioTiling).GetValue<std::vector<std::string>>(tiling_key_vec);
  size_t kernel_num = tiling_key_vec.size();
  std::vector<uint64_t> tiling_key_num_vec;
  std::vector<int64_t> c_ratio_vec;
  std::vector<int64_t> v_ratio_vec;
  tiling_with_ratio.GetItem(kDynRatioCRatio).GetValue<std::vector<int64_t>>(c_ratio_vec);
  tiling_with_ratio.GetItem(kDynRatioVRatio).GetValue<std::vector<int64_t>>(v_ratio_vec);
  if (c_ratio_vec.size() != v_ratio_vec.size() || v_ratio_vec.size() != kernel_num || kernel_num == 0) {
    return ge::FAILED;
  }

  for (size_t i = 0; i < kernel_num; ++i) {
    tiling_key_num_vec.emplace_back(std::stoull(tiling_key_vec[i]));
  }
  ratio_info.tiling_key_holder = bg::CreateContVecHolder(tiling_key_num_vec);
  ratio_info.c_ratio_holder = bg::CreateContVecHolder(c_ratio_vec);
  ratio_info.v_ratio_holder = bg::CreateContVecHolder(v_ratio_vec);
  GELOGD("Get dynamic ratio attr success.");
  is_mix_ratio = true;
  inputs.emplace_back(bg::ValueHolder::CreateConst(&is_mix_ratio, sizeof(is_mix_ratio)));
  inputs.emplace_back(proc_args.tiling_key);
  inputs.emplace_back(ratio_info.tiling_key_holder);
  inputs.emplace_back(ratio_info.c_ratio_holder);
  inputs.emplace_back(ratio_info.v_ratio_holder);
  return ge::SUCCESS;
}

bg::ValueHolderPtr UpdateMixL2Context(const ge::NodePtr &node, const MixProcArgs &proc_args,
                                      bg::ValueHolderPtr flush_data) {
  std::vector<bg::ValueHolderPtr> inputs;
  inputs.emplace_back(flush_data);
  inputs.emplace_back(proc_args.block_dim);
  inputs.emplace_back(proc_args.schedule_mode);
  inputs.emplace_back(proc_args.task_info_para[static_cast<size_t>(TaskPreOutKey::TASK_INFO)]);
  uint32_t contextId = 0;
  (void)ge::AttrUtils::GetInt(node->GetOpDesc(), kContextId, contextId);
  inputs.emplace_back(bg::ValueHolder::CreateConst(&contextId, sizeof(contextId)));
  if (UpdateContextByDynRatio(node, proc_args, inputs) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "Get dynamic ratio info failed.");
    return nullptr;
  }
  return bg::ValueHolder::CreateVoid<bg::ValueHolder>("MixL2UpdateContext", inputs);
}

std::vector<bg::ValueHolderPtr> MixL2UpdateArgs(std::vector<bg::ValueHolderPtr> &inputs,
    const std::vector<bg::DevMemValueHolderPtr> &input_addrs, const std::vector<bg::DevMemValueHolderPtr> &output_addrs,
    const std::vector<bg::ValueHolderPtr> &input_shapes, const std::vector<bg::ValueHolderPtr> &output_shapes) {
  size_t io_addr_num = input_addrs.size() + output_addrs.size();
  auto io_num_holder = bg::ValueHolder::CreateConst(&io_addr_num, sizeof(io_addr_num));
  if (io_num_holder == nullptr) {
    GELOGE(ge::FAILED, "Create io add num holder failed.");
    return {};
  }
  inputs.emplace_back(io_num_holder);
  GELOGD("Mixl2 input num[%zu], output num[%zu].", input_addrs.size(), output_addrs.size());
  inputs.insert(inputs.cend(), input_shapes.cbegin(), input_shapes.cend());
  inputs.insert(inputs.cend(), output_shapes.cbegin(), output_shapes.cend());
  inputs.insert(inputs.cend(), input_addrs.cbegin(), input_addrs.cend());
  inputs.insert(inputs.cend(), output_addrs.cbegin(), output_addrs.cend());
  return bg::ValueHolder::CreateDataOutput("FFTSUpdateMixL2Args", inputs,
                                           static_cast<size_t>(kernel::MixL2ArgsOutKey::kNUM));
}

bg::ValueHolderPtr UpdateMixL2DataDumpInfo(const ge::NodePtr &node, const LowerInput &lower_input,
                                           const std::vector<bg::DevMemValueHolderPtr> &output_addrs) {
  if (!IsDataDumpOpen()) {
    return nullptr;
  }

  std::vector<bg::ValueHolderPtr> inputs;
  // CONTEXT_ID
  uint32_t context_id = 0;
  (void)ge::AttrUtils::GetInt(node->GetOpDesc(), kContextId, context_id);
  inputs.emplace_back(bg::ValueHolder::CreateConst(&context_id, sizeof(context_id)));
  // IN_NUM
  size_t in_num = lower_input.input_addrs.size();
  inputs.emplace_back(bg::ValueHolder::CreateConst(&in_num, sizeof(in_num)));
  // OUT_NUM
  size_t out_num = output_addrs.size();
  inputs.emplace_back(bg::ValueHolder::CreateConst(&out_num, sizeof(out_num)));
  // IO_ADDRS
  inputs.insert(inputs.cend(), lower_input.input_addrs.cbegin(), lower_input.input_addrs.cend());
  inputs.insert(inputs.cend(), output_addrs.cbegin(), output_addrs.cend());

  return bg::ValueHolder::CreateVoid<bg::ValueHolder>("MixL2UpdateDataDumpInfo", inputs);
}

bg::ValueHolderPtr UpdateMixL2ExceptionDumpInfo(const std::vector<bg::ValueHolderPtr> &args_ret,
                                                const bg::ValueHolderPtr &workspace_addrs) {
  if (!IsExceptionDumpOpen()) {
    return nullptr;
  }

  std::vector<bg::ValueHolderPtr> inputs;
  // WORKSPACE
  inputs.emplace_back(workspace_addrs);
  // ARGS_PARA
  inputs.emplace_back(args_ret[static_cast<size_t>(kernel::MixL2ArgsOutKey::ARGS_PARA)]);
  return bg::ValueHolder::CreateVoid<bg::ValueHolder>("MixL2UpdateExceptionDumpInfo", inputs);
}

size_t CalcOpAllInputSize(const ge::NodePtr &node) {
  std::vector<std::vector<int64_t>> dyn_in_vv;
  (void)ge::AttrUtils::GetListListInt(node->GetOpDesc(), kDynamicInputsIndexes, dyn_in_vv);
  size_t dy_add_num = 0;
  for (const auto &dy_in_v : dyn_in_vv) {
    if (!dy_in_v.empty()) {
      dy_add_num += dy_in_v.size() - 1;
    }
  }
  size_t all_input_size = 0;
  (void)ge::AttrUtils::GetInt(node->GetOpDesc(), kOpKernelAllInputSize, all_input_size);
  GELOGD("Op kernel input size [%zu], total anchor size [%zu], dynamic addition size [%zu].",
         all_input_size, node->GetAllInDataAnchorsSize(), dy_add_num);
  all_input_size += dy_add_num;
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kOpKernelAllInputSize, all_input_size);
  return all_input_size;
}

ge::graphStatus CalcMixL2InArgsNum(const ge::NodePtr &node, RtFFTSKernelLaunchArgs::ComputeNodeDesc &node_desc) {
  auto opt_mode = ge::AttrUtils::GetStr(node->GetOpDesc(), kOptionalInputMode);
  if (opt_mode != nullptr && *opt_mode == kGenPlaceHolder) {
    size_t all_input_size = CalcOpAllInputSize(node);
    if (all_input_size == 0) {
      GELOGE(ge::PARAM_INVALID, "Node[%s(%s)] Invalid all_input_size: %zu.",
             node->GetNamePtr(), node->GetTypePtr(), all_input_size);
      return ge::GRAPH_FAILED;
    }
    node_desc.input_num = all_input_size;
  } else {
    node_desc.input_num = node->GetInDataNodesAndAnchors().size();
  }
  auto dy_mode = ge::AttrUtils::GetStr(node->GetOpDesc(), kAttrDynamicParamMode);
  if (dy_mode != nullptr && *dy_mode == kFoldedWithDesc) {
    GELOGD("MixL2 node[%s] is dynamic with folded.", node->GetNamePtr());
    node_desc.dynamic_folded = true;
  }
  return ge::GRAPH_SUCCESS;
}

size_t CalcOpAllOutputSize(const ge::NodePtr &node) {
  std::vector<std::vector<int64_t>> dyn_out_vv;
  (void)ge::AttrUtils::GetListListInt(node->GetOpDesc(), kDynamicOutputsIndexes, dyn_out_vv);
  size_t dy_add_num = 0;
  for (const auto &dy_out_v : dyn_out_vv) {
    if (!dy_out_v.empty()) {
      dy_add_num += dy_out_v.size() - 1;
    }
  }
  size_t all_output_size = 0;
  (void)ge::AttrUtils::GetInt(node->GetOpDesc(), kOpKernelAllOutputSize, all_output_size);
  GELOGD("Op kernel output size %zu, all anchor size[%zu], dynamic add size[%zu].",
         all_output_size, node->GetAllOutDataAnchorsSize(), dy_add_num);
  all_output_size += dy_add_num;
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kOpKernelAllOutputSize, all_output_size);
  return all_output_size;
}

ge::graphStatus CalcMixL2OutArgsNum(const ge::NodePtr &node, RtFFTSKernelLaunchArgs::ComputeNodeDesc &node_desc) {
  auto opt_mode = ge::AttrUtils::GetStr(node->GetOpDesc(), kOptionalOutputMode);
  if (opt_mode != nullptr && *opt_mode == kGenPlaceHolder) {
    size_t all_output_size = CalcOpAllOutputSize(node);
    if (all_output_size == 0) {
      GELOGE(ge::PARAM_INVALID, "Node[%s(%s)] Invalid all_output_size: %zu.",
             node->GetNamePtr(), node->GetTypePtr(), all_output_size);
      return ge::GRAPH_FAILED;
    }
    node_desc.output_num = all_output_size;
  } else {
    node_desc.output_num = node->GetAllOutDataAnchorsSize();
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalcMixL2ArgsMem(const ge::NodePtr &node, const LoweringGlobalData *global_data, size_t &total_size,
    size_t &pre_data_size, std::unique_ptr<uint8_t[]> &pre_data_ptr) {
  (void)global_data;
  RtFFTSKernelLaunchArgs::ComputeNodeDesc node_desc{};
  node_desc.addr_num = 1; // current is 1
  (void)CalcMixL2InArgsNum(node, node_desc);
  (void)CalcMixL2OutArgsNum(node, node_desc);
  node_desc.thread_num_max = 1;
  bool is_unknown = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), kUnknownShapeFromFe, is_unknown);
  bool sta_tiling_depend = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_DYNAMIC_TILING_DEPEND_OP, sta_tiling_depend);
  if (node->GetOpDesc()->HasAttr(optiling::COMPILE_INFO_JSON) || sta_tiling_depend) {
    int64_t tiling_max_size = -1;
    if (!ge::AttrUtils::GetInt(node->GetOpDesc(), bg::kMaxTilingSize, tiling_max_size) || tiling_max_size < 0) {
      GELOGE(ge::PARAM_INVALID, "Node[%s(%s)] Invalid max tiling size: %zu.",
             node->GetName().c_str(), node->GetType().c_str(), tiling_max_size);
      return ge::GRAPH_FAILED;
    }
    auto aligned_max_size = ge::RoundUp(static_cast<uint64_t>(tiling_max_size), sizeof(uintptr_t));
    node_desc.max_tiling_data = aligned_max_size;
    if (IsThirdClassOp(node->GetOpDesc())) {
      node_desc.output_num++;
    }
  }
  node_desc.workspace_cap = node->GetOpDesc()->GetWorkspaceBytes().size();
  if (node->GetOpDesc()->HasAttr(kAtomicCtxIdList)) {
    node_desc.need_atomic = true;
    if (node->GetOpDesc()->HasAttr(optiling::ATOMIC_COMPILE_INFO_JSON)) {
      int64_t tiling_size = -1;
      if (!ge::AttrUtils::GetInt(node->GetOpDesc(), bg::kMaxAtomicCleanTilingSize, tiling_size) || tiling_size < 0) {
        GELOGE(ge::PARAM_INVALID, "Node[%s]Invalid atomic tiling size:%ld.", node->GetName().c_str(), tiling_size);
        return ge::GRAPH_FAILED;
      }
      auto aligned_atomic_max_size = ge::RoundUp(static_cast<uint64_t>(tiling_size), sizeof(uintptr_t));
      node_desc.max_atom_tiling_data = aligned_atomic_max_size;
      node_desc.max_atom_tail_tiling_data = 0U;
      GELOGD("Node [%s] needs an atomic clean with a maximum tiling size of %zu.", node->GetName().c_str(), aligned_atomic_max_size);
    }
  }
  GELOGD("node_desc detail %lu|%lu|%lu|%lu|%lu|%lu", node_desc.addr_num, node_desc.max_tiling_data, node_desc.input_num,
         node_desc.output_num, node_desc.workspace_cap, node_desc.thread_num_max);
  auto rt_arg = RtFFTSKernelLaunchArgs::Create(node, node_desc, total_size);
  FE_ASSERT_NOTNULL(rt_arg);
  GELOGD("Calculate MixL2 node[%s]'s args size:%zu", node->GetNamePtr(), total_size);
  pre_data_size = total_size;
  pre_data_ptr = std::move(rt_arg);
  return ge::GRAPH_SUCCESS;
}
FFTS_REGISTER_NODE_CALCULATER(ge::kFFTSMixL2CalcFunc, CalcMixL2ArgsMem);

ge::Status MixL2ProtoTransfer(const ge::NodePtr &node, const LoweringGlobalData &global_data, size_t &total_size,
    std::unique_ptr<uint8_t[]> &task_info_ptr) {
  const auto find_node_handle = [node](const uint32_t op_index) -> ge::OpDescPtr {
    (void)op_index;
    return node->GetOpDesc();
  };
  std::vector<uintptr_t> io_addrs;
  std::vector<size_t> mode_addr_idx;
  const ge::RuntimeParam runtime_param;
  std::vector<void *> ext_args;
  FftsPlusProtoTransfer ffts_proto_transfer(0U, io_addrs, runtime_param, ext_args, mode_addr_idx);
  ffts_proto_transfer.SetFindNodeHandle(find_node_handle);

  auto compile_result = global_data.FindCompiledResult(node);
  FE_ASSERT_NOTNULL(compile_result);
  if (compile_result->task_defs.empty()) {
    GELOGE(ge::FAILED, "MixL2 node[%s] has no taskdef.", node->GetName().c_str());
    return ge::FAILED;
  }
  const domi::TaskDef &task_def = compile_result->task_defs.at(0U);
  const domi::FftsPlusTaskDef &ffts_plus_task_def = task_def.ffts_plus_task();
  auto trans_task_info = ffts_proto_transfer.Transfer(node->GetOpDesc(), ffts_plus_task_def, total_size);
  FE_ASSERT_NOTNULL(trans_task_info);
  task_info_ptr = std::move(trans_task_info);
  return ge::SUCCESS;
}

std::vector<bg::ValueHolderPtr> MixL2TaskInfoPreProc(const ge::NodePtr &node, FFTSAllMemPara &all_mem_para) {
  auto task_para = CreateNodeMemParam(node, all_mem_para, kMixL2TaskInfoKey);
  auto task_info = bg::ValueHolder::CreateSingleDataOutput("FftsTaskInfoPreProc",
      {all_mem_para.task_data, task_para});
  if (task_info == nullptr) {
    GELOGE(ge::FAILED, "MixL2 node[%s] pre proc task return null.", node->GetName().c_str());
    return {};
  }
  return {task_para, task_info};
}

void InsertMissOptAddr(size_t &arg_idx, std::vector<uint32_t> &insert_pos_vec, InsertOptArgs &insertArgs,
    std::vector<bg::ValueHolderPtr> &placed_input_shapes, std::vector<bg::DevMemValueHolderPtr> &placed_input_addrs) {
  for (auto insert_pos : insert_pos_vec) {
    if (arg_idx == insert_pos) {
      GELOGD("Insert optional input at position %u.", insert_pos);
      placed_input_addrs.emplace_back(insertArgs.empty_val);
      placed_input_shapes.emplace_back(insertArgs.empty_shape);
      arg_idx++;
    }
  }
  return;
}

ge::Status InitInputsForMixL2(const ge::NodePtr &node, const LowerInput &lower_input,
    std::vector<bg::DevMemValueHolderPtr> &placed_input_addrs, std::vector<bg::ValueHolderPtr> &placed_input_shapes) {
  auto op_desc = node->GetOpDesc();
  auto optional_input_mode = ge::AttrUtils::GetStr(op_desc, kOptionalInputMode);
  if (optional_input_mode == nullptr || *optional_input_mode != kGenPlaceHolder) {
    placed_input_addrs.insert(placed_input_addrs.cbegin(), lower_input.input_addrs.cbegin(),
                              lower_input.input_addrs.cend());
    placed_input_shapes.insert(placed_input_shapes.cbegin(), lower_input.input_shapes.cbegin(),
                               lower_input.input_shapes.cend());
    return ge::SUCCESS;
  }
  GELOGD("Node [%s:%s] needs to generate input placeholder", op_desc->GetNamePtr(), op_desc->GetTypePtr());
  size_t all_input_size = 0;
  (void)ge::AttrUtils::GetInt(op_desc, kOpKernelAllInputSize, all_input_size);
  FE_ASSERT_TRUE(all_input_size >= node->GetAllInDataAnchorsSize());
  std::vector<uint32_t> insert_pos_vec;
  (void)ge::AttrUtils::GetListInt(op_desc, kInputInsertOptPosList, insert_pos_vec);
  int32_t tmp_val = 1;
  auto empty_shape = bg::ValueHolder::CreateConst(&tmp_val, sizeof(tmp_val));
  GertTensorData empty_data(0, kOnDeviceHbm, -1, nullptr);
  auto empty_val = bg::DevMemValueHolder::CreateConst(&empty_data, sizeof(empty_data), op_desc->GetStreamId());
  size_t index = 0;
  size_t anchor_index = 0;
  InsertOptArgs insertArgs(empty_shape, empty_val);
  for (const auto &anchor : node->GetAllInDataAnchorsPtr()) {
    InsertMissOptAddr(anchor_index, insert_pos_vec, insertArgs, placed_input_shapes, placed_input_addrs);
    if ((anchor == nullptr) || (anchor->GetPeerOutAnchor() == nullptr)) {
      placed_input_addrs.emplace_back(empty_val);
      placed_input_shapes.emplace_back(empty_shape);
      GELOGD("Input anchor [%zu] is suspend.", anchor_index);
      anchor_index++;
      continue;
    }
    GELOGD("Input anchor at index [%zu] is not optional.", anchor_index);
    placed_input_addrs.emplace_back(lower_input.input_addrs[index]);
    placed_input_shapes.emplace_back(lower_input.input_shapes[index]);
    index++;
    anchor_index++;
  }
  InsertMissOptAddr(anchor_index, insert_pos_vec, insertArgs, placed_input_shapes, placed_input_addrs);
  return ge::SUCCESS;
}

ge::Status InitOutputsForMixL2(const ge::NodePtr &node, const std::vector<bg::DevMemValueHolderPtr> &output_addrs,
    const std::vector<bg::ValueHolderPtr> &output_shapes, std::vector<bg::DevMemValueHolderPtr> &placed_output_addrs,
    std::vector<bg::ValueHolderPtr> &placed_output_shapes) {
  auto op_desc = node->GetOpDesc();
  auto optional_output_mode = ge::AttrUtils::GetStr(op_desc, kOptionalOutputMode);
  if (optional_output_mode == nullptr || *optional_output_mode != kGenPlaceHolder) {
    placed_output_addrs.insert(placed_output_addrs.cbegin(), output_addrs.cbegin(), output_addrs.cend());
    placed_output_shapes.insert(placed_output_shapes.cbegin(), output_shapes.cbegin(), output_shapes.cend());
    return ge::SUCCESS;
  }
  GELOGD("Node [%s:%s] needs to generate output placeholder", op_desc->GetNamePtr(), op_desc->GetTypePtr());
  size_t all_output_size = 0;
  size_t node_anchors_size = node->GetAllOutDataAnchorsSize();
  (void)ge::AttrUtils::GetInt(op_desc, kOpKernelAllOutputSize, all_output_size);
  FE_ASSERT_TRUE(all_output_size >= node_anchors_size);
  int32_t tmp_val = 1;
  auto empty_shape = bg::ValueHolder::CreateConst(&tmp_val, sizeof(tmp_val));
  GertTensorData empty_data(0, kOnDeviceHbm, -1, nullptr);
  auto empty_val = bg::DevMemValueHolder::CreateConst(&empty_data, sizeof(empty_data), op_desc->GetStreamId());
  size_t index = 0;
  for (size_t anchor_index = 0;anchor_index < node_anchors_size;anchor_index++) {
    int32_t calc_type = 0;
    auto output_desc_ptr = op_desc->MutableOutputDesc(anchor_index);
    (void)ge::AttrUtils::GetInt(output_desc_ptr, ge::ATTR_NAME_MEMORY_SIZE_CALC_TYPE, calc_type);
    if (calc_type == static_cast<int32_t>(ge::MemorySizeCalcType::ALWAYS_EMPTY)) {
      placed_output_addrs.emplace_back(empty_val);
      placed_output_shapes.emplace_back(empty_shape);
      GELOGD("Output anchor [%zu] is suspended.", anchor_index);
      continue;
    }
    GELOGD("Output anchor at index [%zu] is not optional.", anchor_index);
    placed_output_addrs.emplace_back(output_addrs[index]);
    placed_output_shapes.emplace_back(output_shapes[index]);
    index++;
  }
  size_t append_size = all_output_size - placed_output_addrs.size();
  for (size_t i = 0; i < append_size; ++i) {
    placed_output_addrs.emplace_back(empty_val);
    placed_output_shapes.emplace_back(empty_shape);
  }
  return ge::SUCCESS;
}

void MakeUpdateArgsInputs(const ge::NodePtr &node, std::vector<bg::ValueHolderPtr> &inputs,
                          const bg::ValueHolderPtr sink_ret, bg::ValueHolderPtr shape_buffer) {
  uint32_t need_mode_addr = 0U;
  (void)ge::AttrUtils::GetInt(node->GetOpDesc(), kNeedModeAddr, need_mode_addr);
  bg::ValueHolderPtr need_addr = bg::ValueHolder::CreateConst(&need_mode_addr, sizeof(need_mode_addr));
  inputs.emplace_back(need_addr);
  inputs.emplace_back(sink_ret);
  inputs.emplace_back(shape_buffer);
  uint32_t need_assert = GetDfxOptFlagByType(node, OpDfxOpt::ASSERT);
  inputs.emplace_back(bg::ValueHolder::CreateConst(&need_assert, sizeof(need_assert)));
}

ge::Status AllocMixL2AllMem(const ge::NodePtr &node, const LowerInput &lower_input,
    FFTSAllMemPara &all_mem_para) {
  std::unique_ptr<uint8_t[]> task_data = nullptr;
  size_t task_size = 0;
  if (MixL2ProtoTransfer(node, *lower_input.global_data, task_size, task_data) != ge::SUCCESS) {
    return ge::FAILED;
  }
  all_mem_para.task_data = bg::ValueHolder::CreateConst(task_data.get(), task_size);
  FFTSAddNodeMemPara(all_mem_para, task_size, task_size, nullptr, kMixL2TaskInfoKey);

  size_t size = 0;
  size_t pre_size = 0;
  std::unique_ptr<uint8_t[]> pre_args_data = nullptr;
  auto cal_func = GetNodeCalculater(node);
  FE_ASSERT_NOTNULL(cal_func);
  if (cal_func(node, lower_input.global_data, size, pre_size, pre_args_data) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::FAILED, "Node[%s] calculate size failed.", node->GetName().c_str());
    return ge::FAILED;
  }
  FFTSAddNodeMemPara(all_mem_para, size, pre_size, std::move(pre_args_data), node->GetName());
  if (CreateMemoryGuard(all_mem_para) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "Node[%s] create memory guard failed.", node->GetName().c_str());
    return ge::FAILED;
  }

  GELOGD("Allocate mix l2 total size: %zu.", all_mem_para.total_size);
  all_mem_para.host_addr_base = bg::AllocMemOnInit(kOnHost, all_mem_para.total_size, *lower_input.global_data);
  FE_ASSERT_NOTNULL(all_mem_para.host_addr_base);
  all_mem_para.dev_addr_base = bg::AllocMemOnInit(kOnDeviceHbm, all_mem_para.total_size, *lower_input.global_data);
  FE_ASSERT_NOTNULL(all_mem_para.dev_addr_base);
  return ge::SUCCESS;
}

bg::ValueHolderPtr CopyKernelTilingdata(const ge::NodePtr &node, MixProcArgs &proc_args) {
  GELOGD("Begin to copy tiling data.");
  std::vector<bg::ValueHolderPtr> inputs;
  std::string tiling_data;
  (void)ge::AttrUtils::GetStr(node->GetOpDesc(), kTilingDataStr, tiling_data);
  size_t size = tiling_data.size();
  inputs.emplace_back(proc_args.args_para);
  inputs.emplace_back(bg::ValueHolder::CreateConst(tiling_data.c_str(), size + 1, true));
  inputs.emplace_back(bg::ValueHolder::CreateConst(&size, sizeof(size)));
  return bg::ValueHolder::CreateVoid<bg::ValueHolder>("CopyTilingdata", inputs);
}

ge::Status LoweringMixL2PreProc(const ge::NodePtr &node, const LowerInput &lower_input,
                                MixProcArgs &proc_args, bool is_unknown, bg::ValueHolderPtr &copy_ret) {
  bool sta_tiling_depend = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_DYNAMIC_TILING_DEPEND_OP, sta_tiling_depend);
  std::shared_ptr<optiling::utils::OpRunInfo> tiling_info = nullptr;
  tiling_info = node->GetOpDesc()->TryGetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, tiling_info);
  if (is_unknown || sta_tiling_depend) {
    auto platform_info_vec = bg::AppendCoreTypeToPlatform(node, lower_input.global_data);
    if (platform_info_vec.empty()) {
      GELOGE(ge::INTERNAL_ERROR, "platform_info_vec is empty! Node: %s.", node->GetName().c_str());
      return ge::FAILED;
    }
    auto platform_info = platform_info_vec[static_cast<size_t>(bg::AssemblePlatformInfoIndex::kPlatformInfo)];
    if (platform_info == nullptr) {
      GELOGE(ge::INTERNAL_ERROR, "platform_info is nullptr! Node: %s.", node->GetName().c_str());
      return ge::FAILED;
    }
    proc_args.tiling_ret = bg::TilingLegacy(node, lower_input.input_shapes, proc_args.output_shapes, platform_info,
                                            *lower_input.global_data);
    CHECK_HOLDERS_ALL_OK_RET(proc_args.tiling_ret, static_cast<size_t>(TilingContext::kOutputNum), return ge::FAILED);
    bg::ValueHolder::AddDependency(proc_args.launch_arg[static_cast<size_t>(AllocFFTSArgOutputs::kTilingDataBase)],
                                   proc_args.tiling_ret[TilingContext::kOutputTilingData]);
    proc_args.tiling_ret[TilingContext::kOutputTilingData]->RefFrom(
        proc_args.launch_arg[static_cast<size_t>(AllocFFTSArgOutputs::kTilingDataBase)]);
    proc_args.workspaces = proc_args.tiling_ret[TilingContext::kOutputWorkspace];
    proc_args.block_dim = proc_args.tiling_ret[TilingContext::kOutputBlockDim];
    proc_args.schedule_mode = proc_args.tiling_ret[TilingContext::kOutputScheduleMode];
    proc_args.tiling_key = proc_args.tiling_ret[TilingContext::kOutputTilingKey];
  } else if (!is_unknown && tiling_info != nullptr) {
    GELOGD("Static and reuse binary situation.");
    const std::vector<int64_t> workspace_bytes = node->GetOpDesc()->GetWorkspaceBytes();
    proc_args.workspaces = bg::CreateContVecHolder(workspace_bytes);
    FE_ASSERT_NOTNULL(proc_args.workspaces);
    int32_t block_dim = tiling_info->GetBlockDim();
    proc_args.block_dim = bg::ValueHolder::CreateConst(&block_dim, sizeof(block_dim));
    uint32_t schedule_mode = tiling_info->GetScheduleMode();
    proc_args.schedule_mode = bg::ValueHolder::CreateConst(&schedule_mode, sizeof(schedule_mode));
    auto tiling_key = tiling_info->GetTilingKey();
    proc_args.tiling_key = bg::ValueHolder::CreateConst(&tiling_key, sizeof(tiling_key));
    copy_ret = CopyKernelTilingdata(node, proc_args);
  } else {
    proc_args.workspaces = ConvertWorkspaceSize(node);
    FE_ASSERT_NOTNULL(proc_args.workspaces);
    int32_t block_dim = 0;
    (void)ge::AttrUtils::GetInt(node->GetOpDesc(), ge::TVM_ATTR_NAME_BLOCKDIM, block_dim);
    proc_args.block_dim = bg::ValueHolder::CreateConst(&block_dim, sizeof(block_dim));
    uint32_t schedule_mode = 0;
    (void)ge::AttrUtils::GetInt(node->GetOpDesc(), "_soft_sync_schedule_mode", schedule_mode);
    proc_args.schedule_mode = bg::ValueHolder::CreateConst(&schedule_mode, sizeof(schedule_mode));
  }
  uint32_t ctx_id = 0U;
  (void)ge::AttrUtils::GetInt(node->GetOpDesc(), kContextId, ctx_id);
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kContextIdList, {ctx_id});
  return ge::SUCCESS;
}

bg::ValueHolderPtr LaunchMixl2Memset(const ge::NodePtr &node, const LowerInput &lower_input,
    AtomicFFTSLowerArg &lower_args, bg::ValueHolderPtr task_info, bg::ValueHolderPtr args_para) {
  FFTSLowerInput ffts_input;
  ffts_input.global_data = lower_input.global_data;
  uint32_t tmp_val = 1;
  auto one_val = bg::ValueHolder::CreateConst(&tmp_val, sizeof(tmp_val));
  FE_ASSERT_NOTNULL(one_val);
  ffts_input.thread_dim = one_val;
  ffts_input.window_size = one_val;
  ffts_input.task_info = task_info;
  ffts_input.args_para = args_para;
  size_t out_num = node->GetAllOutDataAnchorsSize();
  std::vector<uint32_t> mem_pool_types(out_num, 0);
  lower_args.out_mem_type = bg::CreateContVecHolder(mem_pool_types);
  FE_ASSERT_NOTNULL(lower_args.out_mem_type);
  kernel::AICoreThreadParam node_mem; // no matter
  lower_args.thread_para = bg::ValueHolder::CreateConst(&node_mem, sizeof(node_mem));
  FE_ASSERT_NOTNULL(lower_args.thread_para);
  return LaunchFFTSAtomicClean(node, ffts_input, lower_args);
}

std::vector<bg::DevMemValueHolderPtr> LoweringProc(const ge::NodePtr &node, const LowerInput &lower_input,
                                                   MixProcArgs &proc_args, bool is_unknown) {
  bg::ValueHolderPtr copy_ret = nullptr;
  auto ret_preproc = LoweringMixL2PreProc(node, lower_input, proc_args, is_unknown, copy_ret);
  FE_ASSERT_TRUE(ret_preproc == ge::SUCCESS);
  auto workspaces_addr = bg::AllocAiCoreWorkspaceMem(node, kOnDeviceHbm, proc_args.workspaces,
                                                     *(lower_input.global_data));
  FE_ASSERT_NOTNULL(workspaces_addr);
  auto shapebuffer_addr = bg::AllocShapeBufferMem(kOnDeviceHbm, node, *(lower_input.global_data));
  auto output_sizes = bg::CalcTensorSize(node, proc_args.output_shapes);
  auto output_addrs = bg::AllocOutputMemory(kOnDeviceHbm, node, output_sizes, *(lower_input.global_data));
  auto sink_ret = SinkBinForMixAiCore(node, proc_args.tiling_ret);
  FE_ASSERT_NOTNULL(sink_ret);
  AtomicFFTSLowerArg lower_args = {proc_args.tiling_ret, workspaces_addr, output_sizes, {}, output_addrs,
                                   nullptr, proc_args.launch_arg, nullptr};
  auto atomic_launch = LaunchMixl2Memset(node, lower_input, lower_args,
      proc_args.task_info_para[static_cast<size_t>(TaskPreOutKey::TASK_INFO)], proc_args.args_para);

  std::vector<bg::DevMemValueHolderPtr> placed_input_addrs;
  std::vector<bg::ValueHolderPtr> placed_input_shapes;
  auto input_init_ret = InitInputsForMixL2(node, lower_input, placed_input_addrs, placed_input_shapes);
  FE_ASSERT_TRUE(input_init_ret == ge::SUCCESS);

  std::vector<bg::DevMemValueHolderPtr> placed_output_addrs;
  std::vector<bg::ValueHolderPtr> placed_output_shapes;
  auto output_init_ret = InitOutputsForMixL2(node, output_addrs, proc_args.output_shapes, placed_output_addrs,
      placed_output_shapes);
  FE_ASSERT_TRUE(output_init_ret == ge::SUCCESS);

  std::vector<bg::ValueHolderPtr> inputs = {workspaces_addr};
  MakeUpdateArgsInputs(node, inputs, sink_ret, shapebuffer_addr);
  auto args_ret = MixL2UpdateArgs(inputs, placed_input_addrs, placed_output_addrs,
                                  placed_input_shapes, placed_output_shapes);
  CHECK_HOLDERS_ALL_OK_RET(args_ret, static_cast<size_t>(kernel::MixL2ArgsOutKey::kNUM), return {});
  if (copy_ret != nullptr) {
    bg::ValueHolder::AddDependency(copy_ret, args_ret[static_cast<size_t>(kernel::MixL2ArgsOutKey::FLUSH_DATA)]);
    bg::ValueHolder::AddDependency(proc_args.launch_arg[static_cast<size_t>(AllocFFTSArgOutputs::kTilingDataBase)],
                                   copy_ret);
  }
  (void)bg::ValueHolder::AddDependency(proc_args.args_para,
                                       args_ret[static_cast<size_t>(kernel::MixL2ArgsOutKey::ARGS_PARA)]);
  args_ret[static_cast<size_t>(kernel::MixL2ArgsOutKey::ARGS_PARA)]->RefFrom(proc_args.args_para);

  auto update_ret = UpdateMixL2Context(node, proc_args,
                                       args_ret[static_cast<size_t>(kernel::MixL2ArgsOutKey::FLUSH_DATA)]);
  FE_ASSERT_NOTNULL(update_ret);
  // data dump info
  auto data_dump_ret = UpdateMixL2DataDumpInfo(node, lower_input, output_addrs);
  if (data_dump_ret != nullptr) {
    bg::ValueHolder::AddDependency(data_dump_ret, update_ret);
  }

  // exception dump info
  auto exception_dump_ret = UpdateMixL2ExceptionDumpInfo(args_ret, workspaces_addr);
  if (exception_dump_ret != nullptr) {
    bg::ValueHolder::AddDependency(exception_dump_ret, update_ret);
  }
  auto task_ret = FFTSTaskAndArgsLaunch({node, lower_input.global_data, false, nullptr, workspaces_addr},
                                        proc_args.all_mem_para, proc_args.task_info_para);
  CHECK_HOLDERS_ALL_OK_RET(task_ret, static_cast<size_t>(TaskProcKey::kNUM), return {});
  if (atomic_launch != nullptr) {
    bg::ValueHolder::AddDependency(atomic_launch, task_ret[static_cast<size_t>(TaskProcKey::H2D_COPY)]);
  }
  auto ref_out_shapes = SetOutputShape(node, shapebuffer_addr, task_ret[static_cast<size_t>(TaskProcKey::TASK_LAUNCH)],
      lower_input.global_data->GetStream(), proc_args.output_shapes);
  if (!ref_out_shapes.empty()) {
    proc_args.output_shapes = ref_out_shapes;
  }
  auto free_holder = bg::FreeWorkspaceMem(kOnDeviceHbm, workspaces_addr);
  LowerResult lower_ret = {HyperStatus::Success(), {update_ret}, proc_args.output_shapes, output_addrs};
  std::vector<bg::ValueHolderPtr> alloc_vec(output_addrs.cbegin(), output_addrs.cend());
  alloc_vec.insert(alloc_vec.cbegin(), lower_input.input_addrs.cbegin(), lower_input.input_addrs.cend());
  auto ret = LoweringGraphPostProc(&lower_ret, task_ret, {free_holder}, alloc_vec);
  FE_ASSERT_TRUE(ret == ge::SUCCESS);
  proc_args.ordered_holders.insert(proc_args.ordered_holders.end(), lower_ret.order_holders.begin(),
                                   lower_ret.order_holders.end());
  proc_args.ordered_holders.insert(proc_args.ordered_holders.end(), ref_out_shapes.begin(), ref_out_shapes.end());
  return output_addrs;
}

LowerResult LoweringFFTSPlusMixL2(const ge::NodePtr &node, const LowerInput &lower_input) {
  MixProcArgs proc_args;
  auto ret = AllocMixL2AllMem(node, lower_input, proc_args.all_mem_para);
  FE_RET_ERR_RET_IF((ret != ge::SUCCESS), "MixL2 pre alloc memory failed.");
  proc_args.task_info_para = MixL2TaskInfoPreProc(node, proc_args.all_mem_para);
  FE_RET_ERR_RET_IF(proc_args.task_info_para.empty(), "Task info proc failed.");

  proc_args.args_para = CreateNodeMemParam(node, proc_args.all_mem_para);
  proc_args.launch_arg = RedirectLaunchArgs(proc_args.args_para);
  CONVERTER_CHECK_HOLDERS_ALL_OK(proc_args.launch_arg, static_cast<size_t>(AllocFFTSArgOutputs::kNum));
  bool is_unknown = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), kUnknownShapeFromFe, is_unknown);
  GELOGD("MixL2 %s node[%s] begin to lowering.", (is_unknown ? "dynamic" : "static"), node->GetNamePtr());
  proc_args.output_shapes = InferAiCoreStorageShape(node, lower_input.input_shapes, *(lower_input.global_data));
  FE_RET_ERR_RET_IF(proc_args.output_shapes.empty(), "Node infer output shapes failed.");

  if (!NeedCheckEmptyOutput(node) || !is_unknown) {
    std::vector<bg::DevMemValueHolderPtr> output_addrs = LoweringProc(node, lower_input, proc_args, is_unknown);
    FE_RET_ERR_RET_IF(output_addrs.empty(), "Lowering with flag failed");
    if (proc_args.atomic_launch != nullptr) {
      GELOGD("Node [%s] adds atomic launch to order holders.", node->GetName().c_str());
      proc_args.ordered_holders.emplace_back(proc_args.atomic_launch);
    }
    return {HyperStatus::Success(), proc_args.ordered_holders, proc_args.output_shapes, output_addrs};
  }
  bg::ValueHolderPtr cond = bg::ValueHolder::CreateSingleDataOutput("CheckOutputShapesEmpty", proc_args.output_shapes);
  auto if_outputs = bg::If<bg::DevMemValueHolder>(cond,
      [&node, &proc_args, &lower_input]()->std::vector<bg::ValueHolderPtr> {
        auto output_sizes = bg::CalcTensorSize(node, proc_args.output_shapes);
        auto memory = bg::AllocOutputMemory(kOnDeviceHbm, node, output_sizes,
                                            lower_input.input_addrs, *(lower_input.global_data));
        std::vector<bg::ValueHolderPtr> ret_lambda(memory.begin(), memory.end());
        return ret_lambda;
      },
      [&node, &lower_input, &proc_args, is_unknown]()->std::vector<bg::ValueHolderPtr> {
        auto result = LoweringProc(node, lower_input, proc_args, is_unknown);
        std::vector<bg::ValueHolderPtr> ret_lambda(result.begin(), result.end());
        return ret_lambda;
      },
      node->GetOpDesc()->GetStreamId());
  if (if_outputs.size() != proc_args.output_shapes.size() || if_outputs.empty()) {
    return {HyperStatus::ErrorStatus(static_cast<const char*>("Lowering with flag failed")), {}, {}, {}};
  }

  return {HyperStatus::Success(), {if_outputs[0]}, proc_args.output_shapes, if_outputs};
}
REGISTER_NODE_CONVERTER_PLACEMENT(ge::kFFTSMixL2LowerFunc.c_str(), kOnDeviceHbm, LoweringFFTSPlusMixL2);
}  // namespace
}  // namespace gert

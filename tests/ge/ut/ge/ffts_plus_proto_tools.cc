/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge/ut/ge/ffts_plus_proto_tools.h"

#include <memory>

#include "runtime/rt.h"
#include "common/sgt_slice_type.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/types.h"
#include "common/plugin/ge_make_unique_util.h"
#include "aicpu_task_struct.h"
#include "hybrid/node_executor/aicpu/aicpu_ext_info_handler.h"

namespace ge {
void SetKnownOpKernel(const ComputeGraphPtr &graph, uint32_t &mem_offset) {
  const static std::set<std::string> kGeLocalTypes{
      DATA, CONSTANT, CONSTANTOP, VARIABLE, NETOUTPUT, AIPPDATA, FILECONSTANT
  };
  const static std::set<std::string> kRtsLibTypes{
      IDENTITY, IDENTITYN, READVARIABLEOP, PROFILINGTRAININGTRACE, MEMCPYASYNC,
      STREAMACTIVE, STREAMSWITCH, STREAMMERGE, ENTER, REFENTER, LOOPCOND, NEXTITERATION, REFNEXTITERATION,
      EXIT, REFEXIT, LABELSET, LABELGOTO, LABELGOTOEX, LABELSWITCH, LABELSWITCHBYINDEX
  };
  static uint32_t node_index = 0U;

  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT); // sizeof(float) * 1 * 4 * 4 * 8 = 512
  TensorUtils::SetSize(tensor, 512);

  const bool owner_is_unknown = graph->GetGraphUnknownFlag();
  if (owner_is_unknown) {
    AttrUtils::SetBool(graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, owner_is_unknown);
  }

  const auto all_nodes = graph->GetDirectNode();
  for (const auto &node : all_nodes) {
    const auto &op_desc = node->GetOpDesc();
    op_desc->SetId(node_index++);
    AttrUtils::SetBool(op_desc, "OwnerGraphIsUnknown", owner_is_unknown);
    if (kGeLocalTypes.count(op_desc->GetType()) > 0U) {
      op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
    } else if (kRtsLibTypes.count(op_desc->GetType()) > 0U) {
      op_desc->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");
    } else {
      op_desc->SetOpKernelLibName("AIcoreEngine");
    }

    std::vector<int64_t> output_offset;
    for (size_t i = 0U; i < op_desc->GetOutputsSize(); ++i) {
      op_desc->UpdateOutputDesc(i, tensor);
      output_offset.emplace_back(mem_offset);
      mem_offset += 512U;
    }
    op_desc->SetOutputOffset(output_offset);
    op_desc->SetWorkspace({});
    op_desc->SetWorkspaceBytes({});
  }

  for (const auto &node : all_nodes) {
    const auto &op_desc = node->GetOpDesc();
    if ((op_desc->GetType() == DATA) || (op_desc->GetType() == VARIABLE)) {
      continue;
    }
    std::vector<int64_t> input_offset;
    for (size_t i = 0U; i < op_desc->GetInputsSize(); ++i) {
      op_desc->UpdateInputDesc(i, tensor);
      if ((node->GetType() == NETOUTPUT) && (node->GetName() != NODE_NAME_NET_OUTPUT)) {
        AttrUtils::SetInt(op_desc->MutableInputDesc(i), ATTR_NAME_PARENT_NODE_INDEX, i);
      }

      const auto in_anchor = node->GetInDataAnchor(i);
      const auto out_anchor = in_anchor->GetPeerOutAnchor();
      const auto peer_node = out_anchor->GetOwnerNode();
      const std::vector<int64_t> output_offset = peer_node->GetOpDesc()->GetOutputOffset();
      input_offset.emplace_back(output_offset.at(out_anchor->GetIdx()));
    }
    op_desc->SetInputOffset(input_offset);
  }
}

static int32_t g_node_index = 0;
void ResetNodeIndex() {
  g_node_index = 0;
}

OpDescPtr CreateOpDesc(std::string name, std::string type, int in_num, int out_num, bool is_dynamic) {
  const auto op_desc = std::make_shared<OpDesc>(name, type);
  op_desc->SetStreamId(0);
  op_desc->SetId(g_node_index++);

  GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_INT64);
  TensorUtils::SetSize(tensor, 64);
  std::vector<int64_t> input_offset;
  for (int i = 0; i < in_num; ++i) {
    op_desc->AddInputDesc(tensor);
    input_offset.emplace_back(g_node_index * 64 + i * 64);
  }
  op_desc->SetInputOffset(input_offset);

  std::vector<int64_t> output_offset;
  for (int i = 0; i < out_num; ++i) {
    op_desc->AddOutputDesc(tensor);
    output_offset.emplace_back(g_node_index * 64 + in_num * 64 + i * 64);
  }
  op_desc->SetOutputOffset(output_offset);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});

  AttrUtils::SetFloat(op_desc, ATTR_NAME_ALPHA, 0);
  AttrUtils::SetFloat(op_desc, ATTR_NAME_BETA, 0);

  AttrUtils::SetListStr(op_desc, ATTR_NAME_WEIGHT_NAME, {});
  AttrUtils::SetInt(op_desc, POOLING_ATTR_MODE, 0);
  AttrUtils::SetInt(op_desc, POOLING_ATTR_PAD_MODE, 0);
  AttrUtils::SetInt(op_desc, POOLING_ATTR_DATA_MODE, 0);
  AttrUtils::SetInt(op_desc, POOLING_ATTR_CEIL_MODE, 0);
  AttrUtils::SetInt(op_desc, POOLING_ATTR_NAN_OPT, 0);
  AttrUtils::SetListInt(op_desc, POOLING_ATTR_WINDOW, {});
  AttrUtils::SetListInt(op_desc, POOLING_ATTR_PAD, {});
  AttrUtils::SetListInt(op_desc, POOLING_ATTR_STRIDE, {});

  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  if (is_dynamic) {
    AttrUtils::SetBool(op_desc, "support_dynamicshape", is_dynamic);
  }
  return op_desc;
}

NodePtr CreateNode(ComputeGraph &graph, const std::string &name, const std::string &type, int in_num, int out_num) {
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  op_desc->SetStreamId(0);
  static int32_t index = 0;
  op_desc->SetId(index++);

  GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_INT64);
  TensorUtils::SetSize(tensor, 64);
  vector<int64_t> input_offset;
  for (int i = 0; i < in_num; i++) {
    op_desc->AddInputDesc(tensor);
    input_offset.emplace_back(index * 64 + i * 64);
  }
  op_desc->SetInputOffset(input_offset);

  vector<int64_t> output_offset;
  for (int i = 0; i < out_num; i++) {
    op_desc->AddOutputDesc(tensor);
    output_offset.emplace_back(index * 64 + in_num * 64 + i * 64);
  }
  op_desc->SetOutputOffset(output_offset);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});

  const static std::set<std::string> kGeLocalTypes{ DATA, CONSTANT, VARIABLE, NETOUTPUT };
  op_desc->SetOpKernelLibName((kGeLocalTypes.count(type) > 0U) ? "DNN_VM_GE_LOCAL_OP_STORE" : "DNN_VM_RTS_OP_STORE");

  return graph.AddNode(op_desc);
}

void SetNodeAnchorStatus(const NodePtr &node) {
  NodeUtils::SetAllAnchorStatus(node);
  for (auto &anchor : node->GetAllInDataAnchors()) {
    const auto peer_anchor = anchor->GetPeerOutAnchor();
    if (peer_anchor == nullptr) {
      AnchorUtils::SetStatus(anchor, ANCHOR_SUSPEND);
      continue;
    }

    std::string const_type;
    if (NodeUtils::GetConstOpType(peer_anchor->GetOwnerNode(), const_type) && (const_type == CONSTANT)) {
      AnchorUtils::SetStatus(anchor, ANCHOR_CONST);
    } else {
      AnchorUtils::SetStatus(anchor, ANCHOR_DATA);
    }
  }
}

void InitFftsThreadSliceMap(const OpDescPtr &op_desc) {
  std::vector<std::vector<int64_t>> in_dim_range;
  for (size_t i = 0; i < op_desc->GetInputsSize(); ++i) {
    std::vector<int64_t> dim_range;
    dim_range.emplace_back(4);
    dim_range.emplace_back(16);
    in_dim_range.emplace_back(dim_range);
  }

  std::vector<std::vector<int64_t>> out_dim_range;
  for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
    std::vector<int64_t> dim_range;
    dim_range.emplace_back(4);
    dim_range.emplace_back(16);
    out_dim_range.emplace_back(dim_range);
  }

  const auto slice_info = MakeShared<ffts::ThreadSliceMapDy>();
  slice_info->slice_instance_num = 2;
  for (uint32_t i = 0U; i < slice_info->slice_instance_num; ++i) {
    slice_info->input_tensor_slice.emplace_back(in_dim_range);
    slice_info->output_tensor_slice.emplace_back(out_dim_range);
  }

  op_desc->SetExtAttr(ffts::kAttrSgtStructInfoDy, slice_info);
  AttrUtils::SetStr(op_desc, "compile_info_key", "op_compile_info_key");
  AttrUtils::SetStr(op_desc, "compile_info_json", "op_compile_info_json");
}

void InitTaskSQEInfo(domi::FftsPlusTaskDef *task_def) {
  domi::FftsPlusSqeDef *sqedef = task_def->mutable_ffts_plus_sqe();
  //header
  domi::StarsSqeHeaderDef *headerdef = sqedef->mutable_sqe_header();
  headerdef->set_l1_lock(1);
  headerdef->set_l1_unlock(1);
  headerdef->set_block_dim(1);
  //sqe
  sqedef->set_wrr_ratio(1);
  sqedef->set_sqe_index(1);

  sqedef->set_total_context_num(2);
  sqedef->set_ready_context_num(1);
  sqedef->set_preload_context_num(1);

  sqedef->set_prefetch_ost_num(1);
  sqedef->set_cmaint_ost_num(1);

  sqedef->set_aic_prefetch_lower(1);
  sqedef->set_aic_prefetch_upper(1);
  sqedef->set_aiv_prefetch_lower(1);
  sqedef->set_aiv_prefetch_upper(1);
}

void InitTaskAdditionalDataInfo(domi::FftsPlusTaskDef *task_def) {
  domi::AdditionalDataDef *additionaldata = task_def->add_additional_data();
  additionaldata->set_data_type(0);
  additionaldata->add_context_id(0);
  additionaldata->add_context_id(1);
  additionaldata->add_context_id(2);
  domi::AdditionalDataDef *additionaldata1 = task_def->add_additional_data();
  additionaldata1->set_data_type(2);
  additionaldata1->add_context_id(0);
  additionaldata1->add_context_id(3);
  additionaldata1->add_context_id(5);
}

void InitCachePersistentCtx(domi::FftsPlusCachePersistCtxDef *ctx_def) {
  ctx_def->set_successor_num(26);
  ctx_def->set_aten(1);
  ctx_def->set_pred_cnt_init(1);
  ctx_def->set_pred_cnt(1);
  for (int i = 1; i < RT_CTX_SUCCESSOR_NUM; ++i) {
    ctx_def->add_successor_list(1); // 16 bits, len = 26
  }
  ctx_def->set_persistent_en(1);
  ctx_def->set_persistent_id(1);
  ctx_def->set_persistent_size(1);
}

void InitAicAivCtx(domi::FftsPlusAicAivCtxDef *ctx_def, bool is_known) {
  ctx_def->set_successor_num(26);
  ctx_def->set_aten(1);
  ctx_def->set_prefetch_config(1);
  ctx_def->set_pred_cnt_init(1);
  ctx_def->set_pred_cnt(1);
  for (int i = 1; i < RT_CTX_SUCCESSOR_NUM; ++i) {
    ctx_def->add_successor_list(1); // 16 bits, len = 26
  }
  ctx_def->set_schem(1);
  ctx_def->set_atm(1);
  ctx_def->set_prefetch_enable_bitmap(1);
  ctx_def->set_prefetch_once_bitmap(1);

  ctx_def->set_pmg(1);
  ctx_def->set_ns(1);
  ctx_def->set_part_id(1);
  ctx_def->set_qos(1);

  ctx_def->set_thread_id(2);
  if (is_known) { // for unknown thread dim is 0.
    ctx_def->set_thread_dim(1);
  }

  ctx_def->set_non_tail_block_dim(6);
  ctx_def->set_tail_block_dim(5);

  //ctx_def->set_task_param_ptr_base(0x235689);
  ctx_def->set_task_param_ptr_offset(32);
  // task_addr = {0,200,700,1000,2000, 3500}
  // task_addr_offset = {20,40,2,100,200}
  ctx_def->add_task_addr(0);
  ctx_def->add_task_addr(200);
  ctx_def->add_task_addr(700);
  ctx_def->add_task_addr(1000);
  ctx_def->add_task_addr(2000);
  ctx_def->add_task_addr(3500);

  ctx_def->add_task_addr_offset(20);
  ctx_def->add_task_addr_offset(40);
  ctx_def->add_task_addr_offset(2);
  ctx_def->add_task_addr_offset(100);
  ctx_def->add_task_addr_offset(200);

  ctx_def->set_input_output_count(3);
  ctx_def->set_save_task_addr(1);
  if (is_known) {
    ctx_def->add_kernel_name("aictest");
  }
  for (int j = 1; j < 4; ++j) {
    ctx_def->add_src_slot(1);  // len = 4, context ID for source data which is out of subgraph
  }
}

void InitMixAicAivCtx(domi::FftsPlusMixAicAivCtxDef *ctx_def, bool is_auto, bool is_known) {
  ctx_def->set_successor_num(RT_CTX_SUCCESSOR_NUM);
  ctx_def->set_aten(1);
  ctx_def->set_prefetch_config(1);
  ctx_def->set_pred_cnt_init(1);
  ctx_def->set_pred_cnt(1);
  for (int i = 0; i < RT_CTX_SUCCESSOR_NUM; ++i) {
    ctx_def->add_successor_list(1); // len = 26
  }
  ctx_def->set_schem(1);

  if (is_auto) {
    ctx_def->set_atm(1);
  }

  ctx_def->set_prefetch_enable_bitmap(1);
  ctx_def->set_prefetch_once_bitmap(1);

  ctx_def->set_pmg(1);
  ctx_def->set_ns(1);
  ctx_def->set_part_id(1);
  ctx_def->set_qos(1);

  ctx_def->set_non_tail_block_ratio_n(1);
  ctx_def->set_tail_block_ratio_n(1);

  ctx_def->set_thread_id(1);
  if (is_known) { // for unknown thread dim is 0.
    ctx_def->set_thread_dim(1);
  }

  ctx_def->set_non_tail_block_dim(1);
  ctx_def->set_tail_block_dim(1);

  ctx_def->set_aiv_task_param_ptr_offset(1);
  ctx_def->set_aic_task_param_ptr_offset(1);

  ctx_def->add_kernel_name("mixaic_a");
  ctx_def->add_kernel_name("mixaiv_b");
  if (is_auto) {
    ctx_def->add_kernel_name("mixaic_b");
    ctx_def->add_kernel_name("mixaiv_a");
  }

  // task_addr = {0,200,700,1000,2000, 3500}
  // task_addr_offset = {20,40,2,100,200}
  ctx_def->add_task_addr(0x12);
  ctx_def->add_task_addr(0x16);
  ctx_def->add_task_addr(0x1a);
  ctx_def->add_task_addr(0x1e);
  ctx_def->add_task_addr(0x22);
  ctx_def->add_task_addr(0x26);

  ctx_def->add_task_addr_offset(32);
  ctx_def->add_task_addr_offset(32);
  ctx_def->add_task_addr_offset(32);
  ctx_def->add_task_addr_offset(32);
  ctx_def->add_task_addr_offset(32);

  ctx_def->set_input_output_count(1);
  ctx_def->set_save_task_addr(1);
  for (int j = 0; j < 4; ++j) {
    ctx_def->add_src_slot(1);  // len = 4, context ID for source data which is out of subgraph
  }
}

void InitMixAicAivCtxForSingleKernel(domi::FftsPlusMixAicAivCtxDef *ctx_def, bool is_auto, bool is_known) {
  ctx_def->set_successor_num(RT_CTX_SUCCESSOR_NUM);
  ctx_def->set_aten(1);
  ctx_def->set_prefetch_config(1);
  ctx_def->set_pred_cnt_init(1);
  ctx_def->set_pred_cnt(1);
  for (int i = 0; i < RT_CTX_SUCCESSOR_NUM; ++i) {
    ctx_def->add_successor_list(1); // len = 26
  }
  ctx_def->set_schem(1);

  if (is_auto) {
    ctx_def->set_atm(1);
  }

  ctx_def->set_prefetch_enable_bitmap(1);
  ctx_def->set_prefetch_once_bitmap(1);

  ctx_def->set_pmg(1);
  ctx_def->set_ns(1);
  ctx_def->set_part_id(1);
  ctx_def->set_qos(1);

  ctx_def->set_non_tail_block_ratio_n(1);
  ctx_def->set_tail_block_ratio_n(1);

  ctx_def->set_thread_id(1);
  if (is_known) { // for unknown thread dim is 0.
    ctx_def->set_thread_dim(1);
  }

  ctx_def->set_non_tail_block_dim(1);
  ctx_def->set_tail_block_dim(1);

  ctx_def->set_aiv_task_param_ptr_offset(1);
  ctx_def->set_aic_task_param_ptr_offset(1);

  ctx_def->add_kernel_name("mix_a");
  if (is_auto) {
    ctx_def->add_kernel_name("mix_b");
  }

  // task_addr = {0,200,700,1000,2000, 3500}
  // task_addr_offset = {20,40,2,100,200}
  ctx_def->add_task_addr(0x12);
  ctx_def->add_task_addr(0x16);
  ctx_def->add_task_addr(0x1a);
  ctx_def->add_task_addr(0x1e);
  ctx_def->add_task_addr(0x22);
  ctx_def->add_task_addr(0x26);

  ctx_def->add_task_addr_offset(32);
  ctx_def->add_task_addr_offset(32);
  ctx_def->add_task_addr_offset(32);
  ctx_def->add_task_addr_offset(32);
  ctx_def->add_task_addr_offset(32);

  ctx_def->set_input_output_count(1);
  ctx_def->set_save_task_addr(1);
  for (int j = 0; j < 4; ++j) {
    ctx_def->add_src_slot(1);  // len = 4, context ID for source data which is out of subgraph
  }
}

void InitSdmaCtx(domi::FftsPlusSdmaCtxDef *ctx_def) {
  ctx_def->set_successor_num(26);
  ctx_def->set_aten(1);
  ctx_def->set_pred_cnt_init(1);
  ctx_def->set_pred_cnt(1);
  for (int i = 0; i < RT_CTX_SUCCESSOR_NUM; ++i) {
    ctx_def->add_successor_list(1); // len = 26
  }

  ctx_def->set_atm(1);
  ctx_def->set_pmg(1);
  ctx_def->set_ns(1);
  ctx_def->set_part_id(1);
  ctx_def->set_qos(1);

  ctx_def->set_thread_id(1);
  ctx_def->set_thread_dim(1);

  ctx_def->set_sdma_sqe_header(1);

  ctx_def->set_src_stream_id(1);
  ctx_def->set_src_sub_stream_id(1);
  ctx_def->set_dst_stream_id(1);
  ctx_def->set_dst_sub_stream_id(1);

  ctx_def->set_src_addr_base(0x457);
  ctx_def->set_src_addr_offset(32);
  ctx_def->set_dst_addr_base(0x126);
  ctx_def->set_dst_addr_offset(32);

  ctx_def->set_non_tail_data_len(1);
  ctx_def->set_tail_data_len(1);
}

void InitNotifyCtx(domi::FftsPlusNotifyCtxDef *ctx_def) {
  ctx_def->set_successor_num(26);
  ctx_def->set_aten(1);
  ctx_def->set_pred_cnt_init(1);
  ctx_def->set_pred_cnt(1);
  for (int i = 0; i < RT_CTX_SUCCESSOR_NUM; ++i) {
    ctx_def->add_successor_list(1); // len = 26
  }
  ctx_def->set_atm(1);
  ctx_def->set_satm(1);

  ctx_def->set_thread_id(1);
  ctx_def->set_thread_dim(1);

  ctx_def->set_notify_id_base(1);
  ctx_def->set_auto_window(1);
  for (int i = 0; i < 16; ++i) {
    ctx_def->add_notify_id(1);
  }
}

void InitWriteValueCtx(domi::FftsPlusWriteValueCtxDef *ctx_def) {
  ctx_def->set_successor_num(26);
  ctx_def->set_aten(1);
  ctx_def->set_pred_cnt_init(1);
  ctx_def->set_pred_cnt(1);
  for (int i = 0; i < RT_CTX_SUCCESSOR_NUM; ++i) {
    ctx_def->add_successor_list(1); // len = 26
  }
  ctx_def->set_atm(1);
  ctx_def->set_thread_id(1);
  ctx_def->set_thread_dim(1);

  ctx_def->set_aw_size(1);
  ctx_def->set_aw_snoop(1);
  ctx_def->set_aw_cache(1);
  ctx_def->set_aw_prot(1);
  ctx_def->set_aw_va(1);

  ctx_def->set_ar_size(1);
  ctx_def->set_ar_snoop(1);
  ctx_def->set_ar_cache(1);
  ctx_def->set_ar_prot(1);
  ctx_def->set_ar_va(1);

  ctx_def->set_write_addr_base(0x147);
  ctx_def->set_write_addr_offset(32);
  for (int j = 0; j < 4; ++j) {
    ctx_def->add_write_value(1);
  }
}

void InitAicpuCtxCtx(const OpDescPtr &op_desc, domi::FftsPlusAicpuCtxDef *ctx_def, bool is_known) {
  ctx_def->set_successor_num(26);
  ctx_def->set_aten(1);
  ctx_def->set_pred_cnt_init(1);
  ctx_def->set_pred_cnt(1);
  for (int j = 0; j < RT_CTX_SUCCESSOR_NUM; ++j) {
    ctx_def->add_successor_list(1);   // len = 26
  }
  ctx_def->set_atm(1);
  ctx_def->set_sqe_index(1);
  ctx_def->set_kernel_type(2);
  ctx_def->set_bm(1);
  ctx_def->set_topic_type(1);
  ctx_def->set_qos(1);

  ctx_def->set_thread_id(1);
  if (is_known) {
    ctx_def->set_thread_dim(1);
  }

  ctx_def->set_non_tail_block_dim(1);
  ctx_def->set_tail_block_dim(1);

  ctx_def->set_sub_topic_id(1);
  ctx_def->set_topic_id(1);
  ctx_def->set_group_id(1);

  const uint32_t addr_size = op_desc->GetInputsSize() + op_desc->GetOutputsSize();
  const size_t task_args_len = sizeof(aicpu::AicpuParamHead) + (sizeof(uintptr_t) * addr_size);
  ctx_def->set_task_param_offset(task_args_len);

  domi::aicpuKernelDef *kerneldef = ctx_def->mutable_kernel();
  std::vector<char> args_val(task_args_len, '0');
  kerneldef->set_args_size(args_val.size());
  kerneldef->set_args(args_val.data(), args_val.size());
  kerneldef->set_so_name("libaicpu");
  kerneldef->set_kernel_name("aicpu");

  std::vector<char> exts_val;
  size_t ext_info_size = 0U;
  {
    const size_t info_len = sizeof(int32_t);
    exts_val.resize(ext_info_size + sizeof(aicpu::FWKAdapter::ExtInfo) + info_len);
    aicpu::FWKAdapter::ExtInfo &ext_info = *(aicpu::FWKAdapter::ExtInfo *)(&exts_val[ext_info_size]);
    ext_info.infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE;
    ext_info.infoLen = info_len;
    *((int32_t *)ext_info.infoMsg) = 0;
    ext_info_size += sizeof(aicpu::FWKAdapter::ExtInfo) + info_len;
  }

  if (op_desc->GetInputsSize() > 0) {
    const size_t info_len = sizeof(aicpu::FWKAdapter::ShapeAndType) * op_desc->GetInputsSize();
    exts_val.resize(ext_info_size + sizeof(aicpu::FWKAdapter::ExtInfo) + info_len);
    aicpu::FWKAdapter::ExtInfo &ext_info = *(aicpu::FWKAdapter::ExtInfo *)(&exts_val[ext_info_size]);
    ext_info.infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE;
    ext_info.infoLen = info_len;
    aicpu::FWKAdapter::ShapeAndType *shape_type = (aicpu::FWKAdapter::ShapeAndType *)(ext_info.infoMsg);
    for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
      shape_type[i].type = DT_FLOAT;
      shape_type[i].dims[0] = std::numeric_limits<int64_t>::min();
    }
    ext_info_size += sizeof(aicpu::FWKAdapter::ExtInfo) + info_len;
  }

  if (op_desc->GetOutputsSize() > 0) {
    const size_t info_len = sizeof(aicpu::FWKAdapter::ShapeAndType) * op_desc->GetOutputsSize();
    exts_val.resize(ext_info_size + sizeof(aicpu::FWKAdapter::ExtInfo) + info_len);
    aicpu::FWKAdapter::ExtInfo &ext_info = *(aicpu::FWKAdapter::ExtInfo *)(&exts_val[ext_info_size]);
    ext_info.infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE;
    ext_info.infoLen = info_len;
    aicpu::FWKAdapter::ShapeAndType *shape_type = (aicpu::FWKAdapter::ShapeAndType *)(ext_info.infoMsg);
    for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
      shape_type[i].type = DT_FLOAT;
      shape_type[i].dims[0] = std::numeric_limits<int64_t>::min();
    }
    ext_info_size += sizeof(aicpu::FWKAdapter::ExtInfo) + info_len;
  }

  {
    const size_t info_len = sizeof(uint64_t);
    exts_val.resize(ext_info_size + sizeof(aicpu::FWKAdapter::ExtInfo) + info_len);
    aicpu::FWKAdapter::ExtInfo &ext_info = *(aicpu::FWKAdapter::ExtInfo *)(&exts_val[ext_info_size]);
    ext_info.infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_BITMAP;
    ext_info.infoLen = info_len;
    *((uint64_t *)ext_info.infoMsg) = 0U;
    ext_info_size += sizeof(aicpu::FWKAdapter::ExtInfo) + info_len;
  }

  {
    const size_t info_len = sizeof(aicpu::FWKAdapter::AsyncWait);
    exts_val.resize(ext_info_size + sizeof(aicpu::FWKAdapter::ExtInfo) + info_len);
    aicpu::FWKAdapter::ExtInfo &ext_info = *(aicpu::FWKAdapter::ExtInfo *)(&exts_val[ext_info_size]);
    ext_info.infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT;
    ext_info.infoLen = info_len;
    aicpu::FWKAdapter::AsyncWait &async_wait_info = *(aicpu::FWKAdapter::AsyncWait *)(ext_info.infoMsg);
    async_wait_info.waitType = 0;
    async_wait_info.waitId = 0;
    async_wait_info.timeOut = 0;
    async_wait_info.reserved = 0;
    ext_info_size += sizeof(aicpu::FWKAdapter::ExtInfo) + info_len;
  }

  kerneldef->set_kernel_ext_info(exts_val.data(), ext_info_size);
  kerneldef->set_kernel_ext_info_size(ext_info_size);
}

void InitAicpuFwkCtxCtx(domi::FftsPlusAicpuCtxDef *ctx_def) {
  ctx_def->set_successor_num(26);
  ctx_def->set_aten(1);
  ctx_def->set_pred_cnt_init(1);
  ctx_def->set_pred_cnt(1);
  for (int j = 0; j < RT_CTX_SUCCESSOR_NUM; ++j) {
    ctx_def->add_successor_list(1);   // len = 26
  }
  ctx_def->set_atm(1);
  ctx_def->set_sqe_index(1);
  ctx_def->set_kernel_type(1);
  ctx_def->set_bm(1);
  ctx_def->set_topic_type(1);
  ctx_def->set_qos(1);

  ctx_def->set_thread_id(1);
  ctx_def->set_thread_dim(1);

  ctx_def->set_non_tail_block_dim(1);
  ctx_def->set_tail_block_dim(1);

  ctx_def->set_sub_topic_id(1);
  ctx_def->set_topic_id(1);
  ctx_def->set_group_id(1);

  ctx_def->set_task_param_offset(32);

  domi::aicpuKernelDef *kerneldef = ctx_def->mutable_kernel();
  std::vector<char> args_val(sizeof(STR_FWK_OP_KERNEL), '0');
  kerneldef->set_args_size(args_val.size());
  kerneldef->set_args(args_val.data(), args_val.size());
  kerneldef->set_so_name("libaicpu");
  kerneldef->set_kernel_name("aicpu");

  std::vector<char> exts_val(sizeof(aicpu::FWKAdapter::ExtInfo) + sizeof(uint64_t), '0');
  aicpu::FWKAdapter::ExtInfo &ext_info = *(aicpu::FWKAdapter::ExtInfo *)(exts_val.data());
  ext_info.infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_BITMAP;
  ext_info.infoLen = sizeof(uint64_t);
  *((uint64_t *)ext_info.infoMsg) = 0U;
  kerneldef->set_kernel_ext_info(exts_val.data(), exts_val.size());
  kerneldef->set_kernel_ext_info_size(exts_val.size());
}

void InitDataCtx(domi::FftsPlusDataCtxDef *ctx_def) {
  ctx_def->set_successor_num(26);
  ctx_def->set_aten(1);
  ctx_def->set_cnt_init(1);
  ctx_def->set_cnt(1);
  for (int i = 0; i < RT_CTX_SUCCESSOR_NUM; ++i) {
    ctx_def->add_successor_list(1); // len = 26
  }
  ctx_def->set_atm(1);
  ctx_def->set_pmg(1);
  ctx_def->set_ns(1);
  ctx_def->set_part_id(1);
  ctx_def->set_qos(1);

  ctx_def->set_orig_consumer_counter(1);
  ctx_def->set_run_consumer_counter(1);

  ctx_def->set_thread_id(1);
  ctx_def->set_thread_dim(1);

  ctx_def->set_addr_base(0x125);
  ctx_def->set_addr_offset(32);

  ctx_def->set_non_tail_num_outter(1);
  ctx_def->set_non_tail_num_inner(1);
  ctx_def->set_non_tail_len_inner(1);
  ctx_def->set_non_tail_stride_outter(1);
  ctx_def->set_non_tail_stride_inner(1);

  ctx_def->set_tail_num_outter(1);
  ctx_def->set_tail_num_inner(1);
  ctx_def->set_tail_len_inner(1);
  ctx_def->set_tail_stride_outter(1);
  ctx_def->set_tail_stride_inner(1);
}

void InitAicpuFwkCtxAndExtInfo(domi::FftsPlusAicpuCtxDef *ctx_def) {
  ctx_def->set_successor_num(26);
  ctx_def->set_aten(1);
  ctx_def->set_pred_cnt_init(1);
  ctx_def->set_pred_cnt(1);
  for (int j = 0; j < RT_CTX_SUCCESSOR_NUM; ++j) {
    ctx_def->add_successor_list(1);   // len = 26
  }
  ctx_def->set_atm(1);
  ctx_def->set_sqe_index(1);
  ctx_def->set_kernel_type(1);
  ctx_def->set_bm(1);
  ctx_def->set_topic_type(1);
  ctx_def->set_qos(1);

  ctx_def->set_thread_id(1);
  ctx_def->set_thread_dim(1);

  ctx_def->set_non_tail_block_dim(1);
  ctx_def->set_tail_block_dim(1);

  ctx_def->set_sub_topic_id(1);
  ctx_def->set_topic_id(1);
  ctx_def->set_group_id(1);

  ctx_def->set_task_param_offset(32);

  const size_t len = sizeof(hybrid::AicpuExtInfo) + sizeof(hybrid::AsyncWaitInfo);
  std::vector<char> aicpu_ext_info(len, 0);
  int offset = 0;
  hybrid::AicpuExtInfo *ext_info = reinterpret_cast<hybrid::AicpuExtInfo *>(aicpu_ext_info.data() + offset);
  ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT;
  ext_info->infoLen = sizeof(hybrid::AsyncWaitInfo);
  offset += sizeof(hybrid::AicpuExtInfo);
  hybrid::AsyncWaitInfo *async_wait_info = reinterpret_cast<hybrid::AsyncWaitInfo *>(aicpu_ext_info.data() + offset);
  async_wait_info->waitType = 0;
  async_wait_info->waitId = 0;
  async_wait_info->timeOut = 0;
  async_wait_info->reserved = 0;

  std::vector<char> aicpu_arg_info(sizeof(STR_FWK_OP_KERNEL), '0');
  domi::aicpuKernelDef *kerneldef = ctx_def->mutable_kernel();
  kerneldef->set_args_size(aicpu_arg_info.size());
  kerneldef->set_args(aicpu_arg_info.data(), aicpu_arg_info.size());
  kerneldef->set_so_name("libaicpu");
  kerneldef->set_kernel_name("aicpu");

  kerneldef->set_kernel_ext_info(aicpu_ext_info.data(), aicpu_ext_info.size());
  kerneldef->set_kernel_ext_info_size(aicpu_ext_info.size());
}

void InitAicpuCtxAndExtInfo(domi::FftsPlusAicpuCtxDef *ctx_def) {
  ctx_def->set_successor_num(26);
  ctx_def->set_aten(1);
  ctx_def->set_pred_cnt_init(1);
  ctx_def->set_pred_cnt(1);
  for (int j = 0; j < RT_CTX_SUCCESSOR_NUM; ++j) {
    ctx_def->add_successor_list(1);   // len = 26
  }
  ctx_def->set_atm(1);
  ctx_def->set_sqe_index(1);
  ctx_def->set_kernel_type(2);
  ctx_def->set_bm(1);
  ctx_def->set_topic_type(1);
  ctx_def->set_qos(1);

  ctx_def->set_thread_id(1);
  ctx_def->set_thread_dim(1);
  ctx_def->set_non_tail_block_dim(1);
  ctx_def->set_tail_block_dim(1);
  ctx_def->set_sub_topic_id(1);
  ctx_def->set_topic_id(1);
  ctx_def->set_group_id(1);

  ctx_def->set_task_param_offset(32);

  int len = sizeof(hybrid::AicpuExtInfo) + sizeof(hybrid::AsyncWaitInfo);
  vector<char> aicpu_ext_info(len, 0);
  char *buf = aicpu_ext_info.data();
  int offset = 0;
  hybrid::AicpuExtInfo *ext_info = reinterpret_cast<hybrid::AicpuExtInfo*>(buf + offset);
  ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT;
  ext_info->infoLen = sizeof(hybrid::AsyncWaitInfo);
  offset += sizeof(hybrid::AicpuExtInfo);
  hybrid::AsyncWaitInfo *async_wait_info = reinterpret_cast<hybrid::AsyncWaitInfo*>(buf + offset);
  async_wait_info->waitType = 0;
  async_wait_info->waitId = 0;
  async_wait_info->timeOut = 0;
  async_wait_info->reserved = 0;

  std::vector<char> aicpu_arg_info(sizeof(STR_FWK_OP_KERNEL), '0');
  domi::aicpuKernelDef *kerneldef = ctx_def->mutable_kernel();
  kerneldef->set_args_size(aicpu_arg_info.size());
  kerneldef->set_args(aicpu_arg_info.data(), aicpu_arg_info.size());
  kerneldef->set_so_name("libaicpu");
  kerneldef->set_kernel_name("aicpu");

  kerneldef->set_kernel_ext_info(buf, len);
  kerneldef->set_kernel_ext_info_size(len);
}

void InitCustomAicpuCtxAndExtInfo(domi::FftsPlusAicpuCtxDef *ctx_def) {
  ctx_def->set_successor_num(26);
  ctx_def->set_aten(1);
  ctx_def->set_pred_cnt_init(1);
  ctx_def->set_pred_cnt(1);
  for (int j = 0; j < RT_CTX_SUCCESSOR_NUM; ++j) {
    ctx_def->add_successor_list(1);   // len = 26
  }
  ctx_def->set_atm(1);
  ctx_def->set_sqe_index(1);
  ctx_def->set_kernel_type(4);
  ctx_def->set_bm(1);
  ctx_def->set_topic_type(1);
  ctx_def->set_qos(1);

  ctx_def->set_thread_id(1);
  ctx_def->set_thread_dim(1);
  ctx_def->set_non_tail_block_dim(1);
  ctx_def->set_tail_block_dim(1);
  ctx_def->set_sub_topic_id(1);
  ctx_def->set_topic_id(1);
  ctx_def->set_group_id(1);

  ctx_def->set_task_param_offset(32);

  int len = sizeof(hybrid::AicpuExtInfo) + sizeof(hybrid::AsyncWaitInfo);
  vector<char> aicpu_ext_info(len, 0);
  char *buf = aicpu_ext_info.data();
  int offset = 0;
  hybrid::AicpuExtInfo *ext_info = reinterpret_cast<hybrid::AicpuExtInfo*>(buf + offset);
  ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT;
  ext_info->infoLen = sizeof(hybrid::AsyncWaitInfo);
  offset += sizeof(hybrid::AicpuExtInfo);
  hybrid::AsyncWaitInfo *async_wait_info = reinterpret_cast<hybrid::AsyncWaitInfo*>(buf + offset);
  async_wait_info->waitType = 0;
  async_wait_info->waitId = 0;
  async_wait_info->timeOut = 0;
  async_wait_info->reserved = 0;

  std::vector<char> aicpu_arg_info(sizeof(STR_FWK_OP_KERNEL), '0');
  domi::aicpuKernelDef *kerneldef = ctx_def->mutable_kernel();
  kerneldef->set_args_size(aicpu_arg_info.size());
  kerneldef->set_args(aicpu_arg_info.data(), aicpu_arg_info.size());
  kerneldef->set_so_name("libcustomaicpu");
  kerneldef->set_kernel_name("customaicpu");

  kerneldef->set_kernel_ext_info(buf, len);
  kerneldef->set_kernel_ext_info_size(len);
}

void InitAtStartCtx(domi::FftsPlusAtStartCtxDef *ctx_def) {
  ctx_def->set_successor_num(26);
  ctx_def->set_aten(1);
  ctx_def->set_pred_cnt_init(1);
  ctx_def->set_pred_cnt(1);
  for (int i = 0; i < RT_CTX_SUCCESSOR_NUM; ++i) {
    ctx_def->add_successor_list(i); // len = 26
  }
  ctx_def->set_thread_id(1);
  ctx_def->set_thread_dim(1);

  ctx_def->set_thread_id_init(1);
  ctx_def->set_thread_window_size(1);
}

void InitAtEndCtx(domi::FftsPlusAtEndCtxDef *ctx_def) {
  ctx_def->set_at_start_slot_num(12);
  ctx_def->set_out_label_slot_num(12);
  ctx_def->set_aten(1);

  ctx_def->set_pred_cnt_init(1);
  ctx_def->set_pred_cnt(1);
  for (int i = 0; i < RT_CTX_SUCC_AT_START_SLOT_NUM; ++i) {
    ctx_def->add_succ_at_start_slot(i);     // len = 12
    ctx_def->add_succ_out_label_slot(1);    // len = 12
  }

  ctx_def->set_thread_id(1);
}

void InitLabelCtx(domi::FftsPlusLabelCtxDef *ctx_def) {
  ctx_def->set_successor_num(26);
  ctx_def->set_pred_cnt_init(1);
  ctx_def->set_pred_cnt(1);
  for (int i = 0; i < RT_CTX_SUCCESSOR_NUM; ++i) {
    ctx_def->add_successor_list(1); // len = 26
  }
}

void InitCaseSwitchCtx(domi::FftsPlusCaseSwitchCtxDef *ctx_def) {
  ctx_def->set_successor_num(26);
  ctx_def->set_aten(32);
  ctx_def->set_start_label_id(32);
  ctx_def->set_label_list_len(32);
  ctx_def->set_pred_cnt_init(32);
  ctx_def->set_pred_cnt(32);
  for (int i = 0; i < RT_CTX_SUCCESSOR_NUM; ++i) {
    ctx_def->add_successor_list(1); // len = 26
  }
  ctx_def->set_atm(32);

  ctx_def->set_thread_id(32);
  ctx_def->set_thread_dim(32);

  ctx_def->set_ar_size(32);
  ctx_def->set_snoop(32);
  ctx_def->set_ar_cache(32);
  ctx_def->set_ar_prot(32);
  ctx_def->set_va(32);

  ctx_def->set_load_addr0_base(0x123);
  ctx_def->set_ld0_en(32);
  ctx_def->set_load_addr0_offset(32);

  ctx_def->set_load_addr1_base(0x124);
  ctx_def->set_ld1_en(32);
  ctx_def->set_load_addr1_offset(32);
}

void InitCaseDefaultCtx(domi::FftsPlusCaseDefaultCtxDef *ctx_def) {
  ctx_def->set_successor_num(26);
  ctx_def->set_aten(32);
  ctx_def->set_start_label_id(1);
  ctx_def->set_label_list_len(32);
  ctx_def->set_pred_cnt_init(1);
  ctx_def->set_pred_cnt(32);
  for (int i = 0; i < RT_CTX_SUCCESSOR_NUM; ++i) {
    ctx_def->add_successor_list(2); // len = 26
  }
}

void InitCondSwitchCtx(domi::FftsPlusCondSwitchCtxDef *ctx_def) {
  ctx_def->set_true_successor_num(12);
  ctx_def->set_false_successor_num(14);
  ctx_def->set_aten(32);

  ctx_def->set_condition(4);
  ctx_def->set_pred_cnt_init(32);
  ctx_def->set_pred_cnt(32);

  for (int i = 0; i < RT_CTX_FALSE_SUCCESSOR_NUM; ++i) {
    if (i < RT_CTX_TRUE_SUCCESSOR_NUM) {
      ctx_def->add_true_successor_list(1);    // len = 12
    }
    ctx_def->add_false_successor_list(1);   // len = 14
  }
  ctx_def->set_atm(32);

  ctx_def->set_thread_id(1);
  ctx_def->set_thread_dim(32);

  ctx_def->set_ar_size(32);
  ctx_def->set_snoop(32);
  ctx_def->set_ar_cache(32);
  ctx_def->set_ar_prot(32);
  ctx_def->set_va(32);

  ctx_def->set_load_addr0_base(0x142);
  ctx_def->set_ld0_en(32);
  ctx_def->set_load_addr0_offset(32);

  ctx_def->set_load_addr1_base(0x365);
  ctx_def->set_ld1_en(64);
  ctx_def->set_load_addr1_offset(32);

  ctx_def->set_cmp_value_1(1);
  ctx_def->set_cmp_value_2(1);
}

void InitDsaCtx(domi::FftsPlusDsaCtxDef *ctx_def, const bool is_set_value) {
  ctx_def->set_successor_num(14);
  ctx_def->set_aten(32);

  ctx_def->set_pred_cnt_init(32);
  ctx_def->set_pred_cnt(32);

  for (int i = 0; i < RT_CTX_SUCCESSOR_NUM; ++i) {
    ctx_def->add_successor_list(1);
  }
  ctx_def->set_atm(32);

  ctx_def->set_thread_id(1);
  ctx_def->set_thread_dim(1);

  ctx_def->set_start(32);
  ctx_def->set_distribution_type(32);
  ctx_def->set_data_type(32);
  ctx_def->set_alg_type(32);
  ctx_def->set_input_vld(32);
  ctx_def->set_input_value_addr_flag(0);

  if (is_set_value) {
    ctx_def->set_input1_value_or_ptr(1);
    ctx_def->set_input2_value_or_ptr(1);
    ctx_def->set_seed_value_or_ptr(1);
    ctx_def->set_random_count_value_or_ptr(1);
  } else {
    ctx_def->set_input1_value_or_ptr(0);
    ctx_def->set_input2_value_or_ptr(0);
    ctx_def->set_seed_value_or_ptr(0);
    ctx_def->set_random_count_value_or_ptr(0);
  }

  domi::DSATaskArgsDef *dsa_task_args_def = ctx_def->mutable_args();
  dsa_task_args_def->set_output_addr(32);
  dsa_task_args_def->set_workspace_philox_count_addr(32);
  dsa_task_args_def->set_workspace_input_addr(32);
  dsa_task_args_def->set_seed_value_or_addr("0");
  dsa_task_args_def->set_random_count_value_or_addr("0");
  dsa_task_args_def->set_input1_value_or_addr("1");
  dsa_task_args_def->set_input2_value_or_addr("1");
}
}

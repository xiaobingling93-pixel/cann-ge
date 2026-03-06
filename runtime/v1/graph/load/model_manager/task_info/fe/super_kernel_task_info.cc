/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/fe/super_kernel_task_info.h"

#include <securec.h>
#include "common/checker.h"
#include "aicpu_task_struct.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "framework/common/types.h"
#include "graph/load/model_manager/memory_app_type_classifier.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "common/dump/dump_utils.h"
#include "graph/manager/util/hcom_ome_util.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils_ex.h"
#include "graph/utils/node_utils.h"
#include "runtime/kernel.h"
#include "common/kernel_handles_manager/kernel_handle_utils.h"
#include "graph/load/model_manager/kernel/kernel_register_info_builder.h"

namespace {
const std::string kLocalMemorySize = "local_memory_size";
constexpr uint32_t k2BitsMask = 0x00000003U;   // 2  bits, 0000,0011
constexpr int64_t kDefaultDimInfo = 0x100000001;
constexpr uint64_t kDefaultShapeNum = 0x100000000U;
constexpr uint64_t kBitFlag8 = 0x00FFFFFFFFFFFFFFUL;
constexpr uint64_t kLevel2BitFlagWithShape = 0x0200000000000000UL;
constexpr uint64_t kLevel2BitFlagTilingData = 0x0300000000000000UL;
constexpr char_t const *kMaxTilingSize = "op_para_size";
constexpr uint64_t kMaxTilingDataSize = 16UL * 1024UL;

void AppendShapeDesc(const ge::GeTensorDesc &tensor_desc, std::vector<int64_t> &shape_infos) {
  const auto &shape = tensor_desc.GetShape();
  if (shape.IsScalar()) {
    shape_infos.push_back(kDefaultDimInfo);
    shape_infos.push_back(0x1);  // shape value [1]
  } else {
    uint64_t dim_info{kDefaultShapeNum};
    dim_info |= (static_cast<uint64_t>(shape.GetDimNum()));
    shape_infos.push_back(static_cast<int64_t>(dim_info));
    for (const int64_t dim : shape.GetDims()) {
      shape_infos.push_back(dim);
    }
  }
}
}

namespace ge {
Status SuperKernelV2TaskInfo::ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                              TaskRunParam &task_run_param) {
  const domi::KernelDef &kernel_def = task_def.kernel();
  GE_ASSERT_TRUE(kernel_def.sm_desc().empty());
  args_size_ = static_cast<uint32_t>(kernel_def.args().size());

  domi::KernelContext context = kernel_def.context();
  GE_ASSERT_TRUE(!context.args_format().empty());
  kernel_type_ = static_cast<ccKernelType>(context.kernel_type());

  GE_CHECK_NOTNULL(davinci_model);
  op_desc_ = davinci_model->GetOpByIndex(context.op_index());
  GE_CHECK_NOTNULL(op_desc_);

  task_type_ = static_cast<ModelTaskType>(task_def.type());

  GE_ASSERT_SUCCESS(ArgsFormatDesc::Parse(op_desc_, context.args_format(), args_format_holder_.arg_descs),
                    "Formatted args [%s] parsed failed.", context.args_format().c_str());
  GE_ASSERT_SUCCESS(ParseArgsFormat(args_format_holder_.arg_descs),
    "ParseArgsFormat failed, op:[%s].", op_desc_->GetNamePtr());

  const size_t format_args_size = GetArgsSizeByFormat();
  args_size_ = std::max(args_size_, static_cast<uint32_t>(format_args_size));
  GELOGI("OP [%s] has formatted args_format:[%s], args size by format is [%" PRIu64 "], final size is [%u]",
          op_desc_->GetNamePtr(), context.args_format().c_str(), format_args_size, args_size_);

  size_t extra_size = 0U;
  for (const auto &args_format_holder : sub_node_args_format_holder_list_) {
    extra_size += args_format_holder.level1_addr_cnt * sizeof(int64_t);
  }
  GELOGI("Op[%s] args size from_task[%u], extra_size[%zu]", op_desc_->GetNamePtr(), args_size_, extra_size);
  GE_ASSERT_TRUE(!AddOverflow(args_size_, static_cast<uint32_t>(extra_size), args_size_));

  const RuntimeParam &rts_param = davinci_model->GetRuntimeParam();
  size_t num = sub_node_op_desc_list_.size();
  sub_node_input_addrs_list_.resize(num);
  sub_node_output_addrs_list_.resize(num);
  sub_node_workspace_addrs_list_.resize(num);
  sub_node_input_mem_types_list_.resize(num);
  sub_node_output_mem_types_list_.resize(num);
  sub_node_workspace_mem_types_list_.resize(num);
  int index = 0;
  for (const auto &op_desc : sub_node_op_desc_list_) {
    sub_node_input_addrs_list_[index] =
      ModelUtils::GetInputAddrsValue(rts_param, op_desc, sub_node_input_mem_types_list_[index]);
    sub_node_output_addrs_list_[index] =
      ModelUtils::GetOutputAddrsValue(rts_param, op_desc, sub_node_output_mem_types_list_[index]);
    sub_node_workspace_addrs_list_[index] =
      ModelUtils::GetWorkspaceDataAddrsValue(rts_param, op_desc, sub_node_workspace_mem_types_list_[index]);

    for (size_t i = 0UL; i < sub_node_input_addrs_list_[index].size(); i++) {
      task_run_param.parsed_input_addrs.push_back(
        {sub_node_input_addrs_list_[index][i], sub_node_input_mem_types_list_[index][i], true, {0}});
    }
    for (size_t i = 0UL; i < sub_node_output_addrs_list_[index].size(); i++) {
      task_run_param.parsed_output_addrs.push_back(
        {sub_node_output_addrs_list_[index][i], sub_node_output_mem_types_list_[index][i], true, {0}});
    }
    for (size_t i = 0UL; i < sub_node_workspace_mem_types_list_[index].size(); i++) {
      task_run_param.parsed_workspace_addrs.push_back(
        {sub_node_workspace_addrs_list_[index][i], sub_node_workspace_mem_types_list_[index][i], true, {0}});
    }
    index++;
  }

  task_run_param.args_descs.push_back({static_cast<int64_t>(MemSizeAlign(static_cast<size_t>(args_size_),
      static_cast<uint32_t>(sizeof(uintptr_t)))), args_placement_});
  GELOGD(
      "Get args size[%u] of op[%s], is known node[%d], task_type: %d, placement: %d, sub node num: %d.",
      args_size_, op_desc_->GetName().c_str(),
      static_cast<int32_t>(davinci_model->IsFeatureBaseRefreshable()), static_cast<int32_t>(task_type_),
      args_placement_, num);
  return SUCCESS;
}

rtFuncHandle SuperKernelV2TaskInfo::GetFuncHandle() {
  auto kernel_handles_manager = davinci_model_->GetKernelHandlesManager(KernelHandleType::kAicore);
  GE_ASSERT_NOTNULL(kernel_handles_manager);
  KernelRegisterInfo register_info;
  GE_ASSERT_SUCCESS(KernelRegisterInfoBuilder::ConstructAicoreRegisterInfo(op_desc_, false, davinci_model_->GetModelId(), register_info));
  const auto bin_name = kernel_handles_manager->GenerateKey(register_info);
  auto bin_handle = kernel_handles_manager->GetOrRegisterKernel(register_info, bin_name);
  GE_ASSERT_NOTNULL(bin_handle);
  GE_ASSERT_NOTNULL(op_desc_);
  std::string attr_kernel_name = op_desc_->GetName() + "_kernelname";
  std::string kernel_name;
  (void)AttrUtils::GetStr(op_desc_, attr_kernel_name, "_kernelname", kernel_name);
  GELOGD("[%s][%s] get kernel name: %s from attr: %s.", op_desc_->GetNamePtr(), op_desc_->GetTypePtr(),
      kernel_name.c_str(), attr_kernel_name.c_str());
  return KernelHandleUtils::GetFuncHandle(bin_handle, kernel_name);
}

Status SuperKernelV2TaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                 const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                                 const IowAddrs &iow_addrs) {
  GE_CHECK_NOTNULL(davinci_model);
  GE_CHECK_NOTNULL(op_desc_);
  GELOGI("SuperKernelV2TaskInfo Init Start, op: %s", op_desc_->GetNamePtr());

  (void)persistent_workspace;
  davinci_model_ = davinci_model;
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model_->GetStreamList()));

  GE_ASSERT_SUCCESS(UpdateIoAndWorkspaceAddrs(iow_addrs));
  GE_CHK_STATUS_RET_NOLOG(InitKernel(task_def, args));
  func_handle_ = GetFuncHandle();
  GE_ASSERT_NOTNULL(func_handle_);
  io_addr_mem_types_.resize(io_addrs_.size(), static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix));
  GE_ASSERT_SUCCESS(args_io_addrs_updater_.Init(davinci_model_->GetLogicalMemAllocation(), io_addrs_,
      io_addr_mem_types_, {op_desc_->GetName(), op_desc_->GetType()}));

  GELOGI("SuperKernelV2TaskInfo Init Success, node: %s, logic stream id: %u, stream: %p.",
    op_desc_->GetName().c_str(), task_def.stream_id(), stream_);
  return SUCCESS;
}

Status SuperKernelV2TaskInfo::Distribute() {
  GE_ASSERT_NOTNULL(op_desc_);
  GELOGI("SuperKernelV2TaskInfo Distribute Start, op: %s", op_desc_->GetName().c_str());
  const TaskProfGuarder prof_guarder(this);
  GE_CHECK_NOTNULL(op_desc_);

  GE_ASSERT_SUCCESS(ReportL0ExceptionDumpInfo(op_desc_, l0_dump_list_),
    "[%s] report l0 exception dump addr failed", op_desc_->GetNamePtr());

  // call rtKernelLaunch for current task
  const string op_name = op_desc_->GetName();
  GELOGI("Start to launch super kernel of %s, dump flag %d", op_name.c_str(), dump_flag_);
  SetTaskTag(op_name.c_str());
  args_ex_.args = args_;
  args_ex_.argsSize = args_size_;
  args_ex_.isNoNeedH2DCopy = 1U;
  cfg_.dumpflag = dump_flag_;
  cfg_.localMemorySize = local_memory_size_;
  
  LaunchKernelParam launch_kernel_param;
  launch_kernel_param.args = args_;
  launch_kernel_param.args_size = args_size_;
  launch_kernel_param.block_dim = block_dim_;
  launch_kernel_param.stream = stream_;
  launch_kernel_param.launch_config.schedule_mode = cfg_.schemMode;
  launch_kernel_param.launch_config.local_memory_size = local_memory_size_;
  launch_kernel_param.launch_config.block_dim_offset = cfg_.blockDimOffset;
  launch_kernel_param.launch_config.is_block_task_prefetch = is_block_task_prefetch_;
  launch_kernel_param.launch_config.is_data_dump = is_data_dump_;
  GE_ASSERT_SUCCESS(KernelHandleUtils::LaunchKernel(func_handle_, launch_kernel_param));
  call_save_dump_ = true;

  // set for task_id_
  UpdateTaskId();
  GELOGI(
      "SuperKernelV2TaskInfo Distribute Success, node: %s, task_type: %u, args: %p, argsize: %u, "
      "is no need h2d copy : %u, block dim: %u, stream_id: %u, stream: %p, task_id: %u, local memory size: %u, "
      "stubfunc: %p.",
      op_desc_->GetName().c_str(), static_cast<uint32_t>(task_type_), args_ex_.args, args_ex_.argsSize,
      args_ex_.isNoNeedH2DCopy, block_dim_, stream_id_, stream_, task_id_, local_memory_size_, stub_func_);

  if (!domi::GetContext().is_online_model) {
    op_desc_.reset(); // Release OpDesc after Distribute.
    sub_node_op_desc_list_.clear();
    operator_.reset();
  }

  return SUCCESS;
}

Status SuperKernelV2TaskInfo::GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) {
  GELOGI("KernelTaskInfo::GetTaskArgsRefreshInfos in.");
  GE_CHECK_NOTNULL(davinci_model_);
  args_io_addrs_updater_.GenArgsRefreshInfos(infos, io_addr_offset_, args_placement_);
  return SUCCESS;
}

Status SuperKernelV2TaskInfo::Release() {
  rtContext_t ctx = nullptr;
  GE_CHK_RT(rtCtxGetCurrent(&ctx));
  args_ = nullptr;

  return SUCCESS;
}

int64_t SuperKernelV2TaskInfo::ParseOpIndex(const domi::TaskDef &task_def) const {
  const domi::KernelDef &kernel_def = task_def.kernel();
  domi::KernelContext context = kernel_def.context();
  return static_cast<int64_t>(context.op_index());
}

Status SuperKernelV2TaskInfo::FindSkSubNode(const OpDescPtr &sk_op, const int32_t id,  NodePtr &sub_node) const {
  GE_ASSERT_NOTNULL(sk_op);
  if (sk_op->GetId() == static_cast<int64_t>(id)) {
    sub_node = NodeUtils::CreatNodeWithoutGraph(sk_op);
    GE_ASSERT_NOTNULL(sub_node);
    GELOGI("current id %d is sk node %s", id, sk_op->GetNamePtr());
    return SUCCESS;
  }
  ComputeGraphPtr sub_graph = nullptr;
  sub_graph = sk_op->TryGetExtAttr("_sk_sub_graph", sub_graph);
  GE_ASSERT_NOTNULL(sub_graph);
  for (const auto &node : sub_graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    if (node->GetOpDesc()->GetId() == static_cast<int64_t>(id)) {
      sub_node = node;
      GELOGI("find %d sub node %s from sk node %s", id, node->GetNamePtr(), sk_op->GetNamePtr());
      return SUCCESS;
    }
  }
  GELOGE(FAILED, "can not find %d sub node from sk node %s", id, sk_op->GetNamePtr());
  return FAILED;
}

Status SuperKernelV2TaskInfo::GenSubNodeIoToSuperKernelIoMap(size_t node_idx, const NodePtr &sub_node) {
  size_t sub_input_num = sub_node->GetInDataNodesSize();
  for (size_t in_idx = 0U; in_idx < sub_input_num; in_idx++) {
    auto in_node = sub_node->GetInDataNodes().at(in_idx);
    if (in_node->GetType() == "Data") {
        size_t parent_id = 0U;
        GE_ASSERT_TRUE(AttrUtils::GetInt(in_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_id));
        SubNodeIoIndex sub_node_io_idx = {node_idx, in_idx, true};
        sub_node_io_idx_to_super_kernel_io_idx_[sub_node_io_idx] = parent_id;
        GELOGI("subnode[%zu][%s], input idx[%" PRIu64 "] match to super kernel[%s] input idx[%" PRIu64 "]",
          sub_node_io_idx.node_idx, sub_node->GetOpDesc()->GetNamePtr(), sub_node_io_idx.io_idx,
          op_desc_->GetNamePtr(), sub_node_io_idx_to_super_kernel_io_idx_[sub_node_io_idx]);
    }
  }

  size_t sub_output_num = sub_node->GetAllOutDataAnchorsSize();
  for (size_t out_idx = 0U; out_idx < sub_output_num; out_idx++) {
    auto out_data_anchor = sub_node->GetOutDataAnchor(out_idx);
    GE_ASSERT_NOTNULL(out_data_anchor);
    for (const auto &peer_in_anchor : out_data_anchor->GetPeerInDataAnchorsPtr()) {
      if (peer_in_anchor == nullptr) {
        continue;
      } else {
        auto dst_node = peer_in_anchor->GetOwnerNodeBarePtr();
        if (dst_node->GetType() == "NetOutput") {
          size_t parent_id = 0U;
          auto in_id = peer_in_anchor->GetIdx();
          GE_ASSERT_TRUE(
            AttrUtils::GetInt(dst_node->GetOpDesc()->GetInputDesc(in_id), ATTR_NAME_PARENT_NODE_INDEX, parent_id));
          SubNodeIoIndex sub_node_io_idx = {node_idx, out_idx, false};
          sub_node_io_idx_to_super_kernel_io_idx_[sub_node_io_idx] = parent_id;
          GELOGI("subnode[%zu][%s], output idx[%" PRIu64 "] match to super kernel[%s] output idx[%" PRIu64 "]",
            sub_node_io_idx.node_idx, sub_node->GetOpDesc()->GetNamePtr(), sub_node_io_idx.io_idx,
            op_desc_->GetNamePtr(), sub_node_io_idx_to_super_kernel_io_idx_[sub_node_io_idx]);
          break; // 只对应一个netoutput
        }
      }
    }
  }

  return SUCCESS;
}

void SuperKernelV2TaskInfo::InsertL0DumpList(size_t node_idx, size_t io_idx, bool is_input) {
  size_t super_kernel_input_num = op_desc_->GetAllInputsDescPtr().size();
  SubNodeIoIndex sub_node_io_idx = {node_idx, io_idx, is_input};
  if (sub_node_io_idx_to_super_kernel_io_idx_.find(sub_node_io_idx) != sub_node_io_idx_to_super_kernel_io_idx_.end()) {
    size_t super_kernel_io_idx = sub_node_io_idx_to_super_kernel_io_idx_[sub_node_io_idx];
    if (!is_input) {
      super_kernel_io_idx += super_kernel_input_num;
    }
    GELOGI(
      "subnode[%zu][%s] %s idx[%" PRIu64 "] match to super kernel[%s] %s idx[%" PRIu64
      "] io idx[%" PRIu64 "] l0 dump list index[%" PRIu64 "]",
      node_idx, sub_node_op_desc_list_[node_idx]->GetNamePtr(), is_input ? "input" : "output", io_idx,
      op_desc_->GetNamePtr(), is_input ? "input" : "output",
      sub_node_io_idx_to_super_kernel_io_idx_[sub_node_io_idx], super_kernel_io_idx,
      l0_dump_list_.size());
    l0_dump_list_.push_back(super_kernel_io_idx);
  } else {
    l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max()); // 占位
  }
}

void SuperKernelV2TaskInfo::InsertCustToRelevantOffset(size_t node_idx, size_t io_idx, bool is_input) {
  size_t super_kernel_input_num = op_desc_->GetAllInputsDescPtr().size();
  SubNodeIoIndex sub_node_io_idx = {node_idx, io_idx, is_input};
  if (sub_node_io_idx_to_super_kernel_io_idx_.find(sub_node_io_idx) != sub_node_io_idx_to_super_kernel_io_idx_.end()) {
    size_t super_kernel_io_idx = sub_node_io_idx_to_super_kernel_io_idx_[sub_node_io_idx];
    if (!is_input) {
      super_kernel_io_idx += super_kernel_input_num;
    }
    cust_to_relevant_offset_[super_kernel_io_idx] = io_addrs_.size();
    GELOGI("subnode[%zu][%s] node idx %s idx[%" PRIu64 "] match to super kernel[%s] %s idx[%" PRIu64
      "] io idx[%" PRIu64 "] args offset[%" PRIu64 "]",
      node_idx, sub_node_op_desc_list_[node_idx]->GetNamePtr(), is_input ? "input" : "output", io_idx,
      op_desc_->GetNamePtr(), is_input ? "input" : "output",
      sub_node_io_idx_to_super_kernel_io_idx_[sub_node_io_idx], super_kernel_io_idx,
      cust_to_relevant_offset_[super_kernel_io_idx]);
  } else {
    GELOGI("subnode[%zu][%s] %s idx[%" PRIu64 "] match to super kernel[%s] %s "
      "idx[null] io idx[null] args offset[%" PRIu64 "]",
      node_idx, sub_node_op_desc_list_[node_idx]->GetNamePtr(), is_input ? "input" : "output", io_idx,
      op_desc_->GetNamePtr(), is_input ? "input" : "output", io_addrs_.size());
  }
}

Status SuperKernelV2TaskInfo::ParseArgsFormat(const std::vector<ArgDesc> &args_descs) {
  std::unordered_map<uint64_t, std::vector<ArgDesc>> op_index_to_args_desc;
  for (const auto &args_desc : args_descs) {
    int64_t sub_op_id = (args_desc.addr_type == AddrType::SUPER_KERNEL_SUB_NODE) ? args_desc.ir_idx : op_desc_->GetId();
    op_index_to_args_desc[sub_op_id].emplace_back(args_desc);
  }

  std::unordered_set<uint64_t> seen;
  for (const auto &args_desc : args_descs) {
    int64_t sub_op_id = (args_desc.addr_type == AddrType::SUPER_KERNEL_SUB_NODE) ? args_desc.ir_idx : op_desc_->GetId();
    if (seen.find(sub_op_id) == seen.end()) {
      seen.insert(sub_op_id);
      sub_node_op_index_list_.emplace_back(sub_op_id);
      ArgsFormatInfo sub_node_args_format_holder = {};
      for (const auto &sub_node_args_desc : op_index_to_args_desc[sub_op_id]) {
        if (sub_node_args_desc.addr_type != AddrType::SUPER_KERNEL_SUB_NODE) {
          GELOGI("super kernel[%s] has addr type[%d], ir idx[%d]",
                 op_desc_->GetNamePtr(), static_cast<int32_t>(sub_node_args_desc.addr_type), sub_node_args_desc.ir_idx);
          sub_node_args_format_holder.arg_descs.emplace_back(sub_node_args_desc);
          continue;
        }
        ArgDesc args_desc_convert = {};
        int32_t sub_id = 0;
        GE_ASSERT_SUCCESS(ArgsFormatDesc::ConvertArgDescSkToNormal(sub_node_args_desc, args_desc_convert, sub_id));
        sub_node_args_format_holder.arg_descs.emplace_back(args_desc_convert);
        GELOGI("super kernel[%s] has sub node op index[%d], after convert addr type[%d], ir idx[%d]",
          op_desc_->GetNamePtr(), sub_op_id,
          static_cast<int32_t>(args_desc_convert.addr_type), args_desc_convert.ir_idx);
      }

      NodePtr node_ptr;
      GE_ASSERT_SUCCESS(FindSkSubNode(op_desc_, sub_op_id, node_ptr));
      OpDescPtr op_desc = node_ptr->GetOpDesc();
      GE_ASSERT_NOTNULL(op_desc);
      GE_ASSERT_SUCCESS(GenSubNodeIoToSuperKernelIoMap(sub_node_op_desc_list_.size(), node_ptr));
      sub_node_op_desc_list_.emplace_back(op_desc);

      (void)OpDescUtils::GetIrInputInstanceDescRange(op_desc, sub_node_args_format_holder.ir_input_2_range);
      (void)OpDescUtils::GetIrOutputDescRange(op_desc, sub_node_args_format_holder.ir_output_2_range);

      auto input_descs = op_desc->GetAllInputsDescPtr();
      for (const auto &arg_format : sub_node_args_format_holder.arg_descs) {
        if (arg_format.addr_type == AddrType::INPUT_DESC) {
          GE_ASSERT(arg_format.ir_idx >= 0 &&
                    static_cast<size_t>(arg_format.ir_idx) < sub_node_args_format_holder.ir_input_2_range.size());
          const auto &ir_range = sub_node_args_format_holder.ir_input_2_range[static_cast<size_t>(arg_format.ir_idx)];
          std::vector<int64_t> shape_info{0};  // placeholder for offset

          for (size_t idx = 0UL; idx < ir_range.second; ++idx) {
            const size_t instance_idx = static_cast<size_t>(ir_range.first + idx);
            GE_ASSERT_TRUE(instance_idx < input_descs.size(), "Instance index [%zu] is out of range, max_size:[%zu].",
                          instance_idx, input_descs.size());
            AppendShapeDesc(*input_descs.at(instance_idx), shape_info);
          }
          shape_info[0UL] = static_cast<int64_t>(shape_info.size() * sizeof(uintptr_t));
          sub_node_args_format_holder.level1_addr_cnt += ir_range.second + shape_info.size();
          sub_node_args_format_holder.shape_infos.push_back(shape_info);
        } else if (arg_format.addr_type == AddrType::OUTPUT_DESC) {
          GE_ASSERT(arg_format.ir_idx >= 0 &&
                    static_cast<size_t>(arg_format.ir_idx) < sub_node_args_format_holder.ir_output_2_range.size());
          const auto &ir_range = sub_node_args_format_holder.ir_output_2_range[static_cast<size_t>(arg_format.ir_idx)];
          std::vector<int64_t> shape_info{0};  // placeholder for offset
          for (size_t idx = 0UL; idx < ir_range.second; ++idx) {
            auto output_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(ir_range.first + idx));
            GE_ASSERT_NOTNULL(output_desc);
            AppendShapeDesc(*output_desc, shape_info);
          }
          shape_info[0UL] = static_cast<int64_t>(shape_info.size() * sizeof(uintptr_t));
          sub_node_args_format_holder.level1_addr_cnt += ir_range.second + shape_info.size();
          sub_node_args_format_holder.shape_infos.push_back(shape_info);
        } else if (arg_format.addr_type == AddrType::TILING_CONTEXT &&
                   arg_format.ir_idx == static_cast<int32_t>(TilingContextSubType::TILING_CONTEXT)) {
          const auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
              .GetSpaceRegistry(static_cast<gert::OppImplVersionTag>(op_desc->GetOppImplVersion()));
          const auto op_impl = space_registry->GetOpImpl(op_desc->GetTypePtr());
          GE_ASSERT_NOTNULL(op_impl, "Failed to get op registry func for node %s.", op_desc->GetNamePtr());
          for (size_t i = 0UL; i < op_desc->GetInputsSize(); ++i) {
            size_t ir_index = 0UL;
            GE_ASSERT_SUCCESS(ge::OpDescUtils::GetInputIrIndexByInstanceIndex(op_desc, i, ir_index));
            if (op_impl->IsTilingInputDataDependency(ir_index)) {
              GELOGI("Node [%s]'s [%zu]th input has tiling dependency.", op_desc->GetNamePtr(), i);
              sub_node_args_format_holder.tiling_depends_input_idx.push_back(i);
            }
          }
        }
      }

      sub_node_args_format_holder_list_.emplace_back(std::move(sub_node_args_format_holder));
    }
  }
  return SUCCESS;
}

size_t SuperKernelV2TaskInfo::GetArgsSizeByFormat() const {
  const auto &arg_descs = args_format_holder_.arg_descs;
  size_t tmp_size = 0U;
  for (const auto &arg_desc : arg_descs) {
    (void)ArgsFormatDesc::GetArgSize(op_desc_, arg_desc, tmp_size);
  }
  return tmp_size;
}

Status SuperKernelV2TaskInfo::UpdateIoAndWorkspaceAddrs(const IowAddrs &iow_addrs) {
  // 增加size的校验
  size_t index = 0U;
  for (size_t i = 0UL; i < sub_node_input_addrs_list_.size(); i++) {
    for (size_t j = 0UL; j < sub_node_input_addrs_list_[i].size(); j++) {
      sub_node_input_addrs_list_[i][j] = iow_addrs.input_logic_addrs[index].logic_addr;
      sub_node_input_mem_types_list_[i][j] = iow_addrs.input_logic_addrs[index].memory_type;
      index++;
    }
  }

  index = 0U;
  for (size_t i = 0UL; i < sub_node_output_addrs_list_.size(); i++) {
    for (size_t j = 0UL; j < sub_node_output_addrs_list_[i].size(); j++) {
      sub_node_output_addrs_list_[i][j] = iow_addrs.output_logic_addrs[index].logic_addr;
      sub_node_output_mem_types_list_[i][j] = iow_addrs.output_logic_addrs[index].memory_type;
      index++;
    }
  }

  index = 0U;
  for (size_t i = 0UL; i < sub_node_workspace_addrs_list_.size(); i++) {
    for (size_t j = 0UL; j < sub_node_workspace_addrs_list_[i].size(); j++) {
      sub_node_workspace_addrs_list_[i][j] = iow_addrs.workspace_logic_addrs[index].logic_addr;
      sub_node_workspace_mem_types_list_[i][j] = iow_addrs.workspace_logic_addrs[index].memory_type;
      index++;
    }
  }

  return SUCCESS;
}

void SuperKernelV2TaskInfo::AppendIoAddr(const uint64_t addr, const uint64_t addr_type) {
  io_addrs_.push_back(addr);
  io_addr_mem_types_.push_back(addr_type);
}

Status SuperKernelV2TaskInfo::AppendInputOutputAddrByInstanceIndex(size_t node_idx, size_t ins_idx, bool is_input) {
  if (is_input) {
    GE_ASSERT_TRUE(ins_idx < sub_node_input_addrs_list_[node_idx].size(),
                   "Node idx [%zu] instance idx [%zu] is invalid, input_size:[%zu]",
                   node_idx, ins_idx, sub_node_input_addrs_list_[node_idx].size());
    InsertL0DumpList(node_idx, ins_idx, is_input);
    InsertCustToRelevantOffset(node_idx, ins_idx, is_input);
    AppendIoAddr(sub_node_input_addrs_list_[node_idx][ins_idx], sub_node_input_mem_types_list_[node_idx][ins_idx]);
  } else {
    GE_ASSERT_TRUE(ins_idx < sub_node_output_addrs_list_[node_idx].size(),
                   "Node idx [%zu] instance idx [%zu] is invalid, output_size:[%zu]",
                   node_idx, sub_node_output_addrs_list_[node_idx].size());
    InsertL0DumpList(node_idx, ins_idx, is_input);
    InsertCustToRelevantOffset(node_idx, ins_idx, is_input);
    AppendIoAddr(sub_node_output_addrs_list_[node_idx][ins_idx], sub_node_output_mem_types_list_[node_idx][ins_idx]);
  }
  return SUCCESS;
}

Status SuperKernelV2TaskInfo::AppendInputOutputAddr(size_t node_idx, size_t ir_idx, bool is_input) {
  const std::map<size_t, std::pair<size_t, size_t>> &ir_2_range =
      is_input ? sub_node_args_format_holder_list_[node_idx].ir_input_2_range :
      sub_node_args_format_holder_list_[node_idx].ir_output_2_range;
  const auto iter = ir_2_range.find(ir_idx);
  GE_ASSERT(iter != ir_2_range.end(),
    "sub node[%zu] ir idx[%zu] is not found, input flag %u.", node_idx, ir_idx, is_input);
  const auto &range_pair = iter->second;
  if (is_input && range_pair.second == 0UL) {
    // optional input placeholder
    l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max());
    AppendIoAddr(0UL, kAbsoluteMemType);
    return SUCCESS;
  }
  size_t begin_idx = range_pair.first;
  std::vector<uint64_t> &addrs =
    is_input ? sub_node_input_addrs_list_[node_idx] : sub_node_output_addrs_list_[node_idx];
  std::vector<uint64_t> &types =
    is_input ? sub_node_input_mem_types_list_[node_idx] : sub_node_output_mem_types_list_[node_idx];
  for (size_t i = 0UL; i < range_pair.second; ++i, ++begin_idx) {
    GE_ASSERT(begin_idx < addrs.size(), "sub node[%zu] ir idx[%zu], begin_index[%zu] is out of range, max_size[%zu].",
              node_idx, ir_idx, begin_idx, addrs.size());
    InsertL0DumpList(node_idx, begin_idx, is_input);
    InsertCustToRelevantOffset(node_idx, begin_idx, is_input);
    AppendIoAddr(addrs[begin_idx], types[begin_idx]);
  }
  return SUCCESS;
}

Status SuperKernelV2TaskInfo::AppendWorkspaceAddr(size_t node_idx, int32_t ir_idx) {
  if (ir_idx < 0) {
    l0_dump_list_.insert(l0_dump_list_.end(),
      sub_node_workspace_addrs_list_[node_idx].size(), std::numeric_limits<uint64_t>::max()); // 占位
    (void)io_addrs_.insert(io_addrs_.cend(), sub_node_workspace_addrs_list_[node_idx].cbegin(),
                           sub_node_workspace_addrs_list_[node_idx].cend());
    (void)io_addr_mem_types_.insert(io_addr_mem_types_.cend(), sub_node_workspace_mem_types_list_[node_idx].cbegin(),
                                    sub_node_workspace_mem_types_list_[node_idx].cend());
  } else {
    const size_t idx = static_cast<size_t>(ir_idx);
    GE_ASSERT(idx < sub_node_workspace_addrs_list_[node_idx].size(), "workspace idx:[%zu] is output of range, max_size:[%zu]",
              idx, sub_node_workspace_addrs_list_[node_idx].size());
    uint64_t index = std::numeric_limits<uint64_t>::max();
    if (sub_node_op_desc_list_[node_idx] == op_desc_) {
      index = op_desc_->GetInputsSize() + op_desc_->GetOutputsSize() + static_cast<uint64_t>(ir_idx);
    }
    l0_dump_list_.push_back(index);
    AppendIoAddr(sub_node_workspace_addrs_list_[node_idx][idx], sub_node_workspace_mem_types_list_[node_idx][idx]);
  }
  return SUCCESS;
}

Status SuperKernelV2TaskInfo::AssembleShapeInfoAddrs(
  const std::vector<std::vector<ArgDesc>> &sub_node_dynamic_args_desc,
  const std::vector<std::vector<size_t>> &sub_node_level2_addr_idx) {
  size_t node_idx = 0UL;
  for (const auto &dynamic_args_desc : sub_node_dynamic_args_desc) {
    if (dynamic_args_desc.size() == 0UL) {
      node_idx++;
      continue;
    }

    GE_ASSERT(dynamic_args_desc.size() == sub_node_args_format_holder_list_[node_idx].shape_infos.size());
    for (size_t i = 0UL; i < dynamic_args_desc.size(); i++) {
      std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range =
        sub_node_args_format_holder_list_[node_idx].ir_input_2_range;
      std::map<size_t, std::pair<size_t, size_t>> &ir_output_2_range =
        sub_node_args_format_holder_list_[node_idx].ir_output_2_range;

      auto &shape_info = sub_node_args_format_holder_list_[node_idx].shape_infos[i];
      const size_t ptr_offset_idx = io_addrs_.size();
      size_t addr_idx = sub_node_level2_addr_idx[node_idx][i];
      GE_ASSERT(addr_idx < io_addrs_.size());

      // addr to ptr offset
      io_addrs_[addr_idx] =
        PtrToValue(args_) + static_cast<uint64_t>(ptr_offset_idx * sizeof(uint64_t));
      GELOGI("subnode[%zu] set ptr_offset idx[%zu], addr idx[%" PRIx64 "] value[%" PRIx64 "]",
        node_idx, ptr_offset_idx, addr_idx, io_addrs_[addr_idx]);
      // copy shape_infos
      (void)io_addrs_.insert(io_addrs_.cend(), shape_info.cbegin(), shape_info.cend());
      (void)io_addr_mem_types_.insert(io_addr_mem_types_.cend(), shape_info.size(), kAbsoluteMemType);

      if (dynamic_args_desc[i].addr_type == AddrType::INPUT_DESC) {
        const size_t ir_idx = static_cast<size_t>(dynamic_args_desc[i].ir_idx);
        const auto &range_pair = ir_input_2_range[ir_idx];
        size_t begin_idx = range_pair.first;
        for (size_t idx = 0UL; idx < range_pair.second; ++idx) {
          GE_ASSERT(begin_idx < sub_node_input_addrs_list_[node_idx].size(),
                    "subnode[%zu] ir_idx [%zu], begin_index [%zu] is out of range, max_size:[%zu].",
                    node_idx, ir_idx, begin_idx, sub_node_input_addrs_list_[node_idx].size());
          InsertCustToRelevantOffset(node_idx, begin_idx, true);
          AppendIoAddr(sub_node_input_addrs_list_[node_idx][begin_idx],
            sub_node_input_mem_types_list_[node_idx][begin_idx]);
          ++begin_idx;
        }
      } else if (dynamic_args_desc[i].addr_type == AddrType::OUTPUT_DESC) {
        const size_t ir_idx = static_cast<size_t>(dynamic_args_desc[i].ir_idx);
        const auto &range_pair = ir_output_2_range[ir_idx];
        size_t begin_idx = range_pair.first;
        for (size_t idx = 0UL; idx < range_pair.second; ++idx) {
          GE_ASSERT(begin_idx < sub_node_output_addrs_list_[node_idx].size(),
                    "subnode[%zu] ir_idx:[%zu], begin_index [%zu] is out of range, max_size:[%zu].",
                    node_idx, ir_idx, begin_idx, sub_node_output_addrs_list_[node_idx].size());
          InsertCustToRelevantOffset(node_idx, begin_idx, false);
          AppendIoAddr(sub_node_output_addrs_list_[node_idx][begin_idx], sub_node_output_mem_types_list_[node_idx][begin_idx]);
          ++begin_idx;
        }
      }
    }
    node_idx++;
  }

  return SUCCESS;
}

// 为了做到tensor.GetTensorData().GetAddr()返回地址的可刷新，这里需要做两次8字节对齐，即TensorData首地址对其以及Tensor大小的对齐
void SuperKernelV2TaskInfo::GetAddrAlignedGertTensorSize(size_t &io_aligned_offset,
                                                         size_t &double_aliged_tensor_size) const {
  gert::Tensor tensor;
  tensor.MutableTensorData();
  const size_t raw_addr_offset =
      static_cast<size_t>(ge::PtrToValue(&tensor.MutableTensorData()) - ge::PtrToValue(&tensor));
  io_aligned_offset = ge::MemSizeAlign(raw_addr_offset, static_cast<uint32_t>(sizeof(uint64_t)));
  double_aliged_tensor_size = sizeof(gert::Tensor) + io_aligned_offset;
  double_aliged_tensor_size = ge::MemSizeAlign(double_aliged_tensor_size, static_cast<uint32_t>(sizeof(uint64_t)));
}

Status SuperKernelV2TaskInfo::AssembleTilingSinkTensors(
    std::map<int32_t ,std::map<size_t, gert::AddrRefreshedTensor>> &index_to_tensor) {
  for (size_t i = 0; i < sub_node_op_index_list_.size(); ++i) {
    if (sub_node_args_format_holder_list_[i].tiling_depends_input_idx.empty()) {
      return SUCCESS;
    }
    size_t rt_tensor_offset{0UL};
    size_t rt_tensor_size{0UL};
    GetAddrAlignedGertTensorSize(rt_tensor_offset, rt_tensor_size);
    GELOGI("IoAddr Offset:[%zu] double aligned tensor size:[%zu].", rt_tensor_offset, rt_tensor_size);
    sub_node_args_format_holder_list_[i].sink_tensor_size = rt_tensor_size *
        sub_node_args_format_holder_list_[i].tiling_depends_input_idx.size();
    const size_t addr_num = sub_node_args_format_holder_list_[i].sink_tensor_size / sizeof(uint64_t);
    io_addrs_.resize(addr_num);
    io_addr_mem_types_.resize(addr_num, kAbsoluteMemType);
    size_t tensor_cnt = 0UL;
    const auto args = (task_type_ != ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL) ?
                      PtrToValue(args_) : (PtrToValue(args_) + sizeof(aicpu::AicpuParamHead));
    GELOGI("tiling sink task[%u] args[0x%" PRIx64 "] for [%s]",
      static_cast<uint32_t>(task_type_), args, op_desc_->GetNamePtr());
    for (auto tiling_idx : sub_node_args_format_holder_list_[i].tiling_depends_input_idx) {
      index_to_tensor[i][tiling_idx].device_addr = args + rt_tensor_size * tensor_cnt + rt_tensor_offset;
      gert::Tensor *host_tensor =
          reinterpret_cast<gert::Tensor *>(PtrToValue(io_addrs_.data()) + rt_tensor_size * tensor_cnt + rt_tensor_offset);
      GE_ASSERT_NOTNULL(host_tensor);
      GE_ASSERT(tiling_idx < sub_node_input_addrs_list_[i].size(), "Input index [%zu] is invalid, inputs size:[%zu]", tiling_idx,
                sub_node_input_addrs_list_[i].size());
      host_tensor->MutableTensorData().SetAddr(ValueToPtr(sub_node_input_addrs_list_[i][tiling_idx]), nullptr);
      const size_t addr_offset =
          static_cast<size_t>(PtrToValue(&host_tensor->MutableTensorData()) - PtrToValue(io_addrs_.data()));
      const size_t addr_idx = addr_offset / sizeof(uintptr_t);
      GE_ASSERT(addr_idx < io_addr_mem_types_.size(), "Tensor addr index [%zu] is invalid, io mem type size:[%zu].",
                addr_idx, io_addr_mem_types_.size());
      io_addr_mem_types_[addr_idx] = sub_node_input_mem_types_list_[i][tiling_idx];
      GELOGI("Set tensor addr index [%zu] memory type with [%" PRIu64 "] by input idx:[%zu]", addr_idx,
             sub_node_input_mem_types_list_[i][tiling_idx], tiling_idx);

      index_to_tensor[i][tiling_idx].host_addr = host_tensor;
      ++tensor_cnt;
    }
  }
  return SUCCESS;
}

Status SuperKernelV2TaskInfo::AssembleTilingContextArgs(int32_t node_idx, const ArgDesc &arg_desc,
                                                 std::map<size_t, gert::AddrRefreshedTensor> &index_to_tensor) {
  std::shared_ptr<TilingContextAddr> default_ctx_ptr = nullptr;
  std::shared_ptr<TilingContextAddr> tiling_context_addr =
      sub_node_op_desc_list_[node_idx]->TryGetExtAttr(kTilingContextAddrs, default_ctx_ptr);
  const TilingContextSubType sub_type = static_cast<TilingContextSubType>(arg_desc.ir_idx);
  switch (sub_type) {
    case TilingContextSubType::TILING_CONTEXT:
      if (tiling_context_addr == nullptr) {
        NodePtr sub_node = nullptr;
        GE_ASSERT_SUCCESS(FindSkSubNode(op_desc_, sub_node_op_index_list_[node_idx], sub_node));
        // init platform info on device
        void *platform_infos_addr{nullptr};
        GE_ASSERT_SUCCESS(davinci_model_->LaunchPlatformInfos(platform_infos_addr, sub_node));
        GE_ASSERT_NOTNULL(platform_infos_addr, "Please check platform_infos_addr.");
        GELOGD("platform_infos_addr = %" PRIu64, PtrToValue(platform_infos_addr));
        GE_ASSERT_SUCCESS(ArgsFormatUtils::SinkTilingContext(sub_node, *davinci_model_, index_to_tensor,
                                                             platform_infos_addr, false, 0));
        tiling_context_addr = sub_node_op_desc_list_[node_idx]->TryGetExtAttr(kTilingContextAddrs, default_ctx_ptr);
        GE_ASSERT_NOTNULL(tiling_context_addr, "Failed to sink tiling context.");
      }
      AppendIoAddr(tiling_context_addr->tiling_context_addr, kAbsoluteMemType);
      break;
    case TilingContextSubType::TILING_DATA:
      GE_ASSERT_NOTNULL(tiling_context_addr, "Please check whether tiling task has been launched.");
      AppendIoAddr(tiling_context_addr->tiling_data_addr, kAbsoluteMemType);
      break;
    case TilingContextSubType::TILING_KEY:
      GE_ASSERT_NOTNULL(tiling_context_addr, "Please check whether tiling task has been launched.");
      AppendIoAddr(tiling_context_addr->tiling_key_addr, kAbsoluteMemType);
      break;
    case TilingContextSubType::BLOCK_DIM:
      GE_ASSERT_NOTNULL(tiling_context_addr, "Please check whether tiling task has been launched.");
      AppendIoAddr(tiling_context_addr->block_dim_addr, kAbsoluteMemType);
      break;
    default:
      GELOGE(FAILED, "ir index [%d] is invalid.", arg_desc.ir_idx);
      break;
  }
  return SUCCESS;
}

Status SuperKernelV2TaskInfo::AssembleIoByArgsFormat() {
  std::vector<std::vector<size_t>> sub_node_level_addr_idx;
  std::vector<std::vector<ArgDesc>> sub_node_dynamic_args_desc;

  size_t node_num = sub_node_args_format_holder_list_.size();
  sub_node_dynamic_args_desc.resize(node_num);
  sub_node_level_addr_idx.resize(node_num);

  std::map<int32_t ,std::map<size_t, gert::AddrRefreshedTensor>> idx_to_sink_tensor_map;
  GE_ASSERT_SUCCESS(AssembleTilingSinkTensors(idx_to_sink_tensor_map));

  size_t node_idx = 0U;
  GE_ASSERT_TRUE(node_num == sub_node_input_addrs_list_.size());
  GE_ASSERT_TRUE(node_num == sub_node_output_addrs_list_.size());
  GE_ASSERT_TRUE(node_num == sub_node_workspace_addrs_list_.size());
  for (const auto &args_format_holder : sub_node_args_format_holder_list_) {
    const std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range = args_format_holder.ir_input_2_range;
    const std::map<size_t, std::pair<size_t, size_t>> &ir_output_2_range = args_format_holder.ir_output_2_range;
    for (const auto &arg_format : args_format_holder.arg_descs) {
      switch (arg_format.addr_type) {
        case AddrType::INPUT_INSTANCE: {
          GE_ASSERT_SUCCESS(AppendInputOutputAddrByInstanceIndex(node_idx, static_cast<size_t>(arg_format.ir_idx), true));
          break;
        }
        case AddrType::OUTPUT_INSTANCE: {
          GE_ASSERT_SUCCESS(AppendInputOutputAddrByInstanceIndex(node_idx, static_cast<size_t>(arg_format.ir_idx), false));
          break;
        }
        case AddrType::INPUT_DESC: {
          // l0 exception dump处理
          const size_t ir_idx = static_cast<size_t>(arg_format.ir_idx);
          const auto iter = ir_input_2_range.find(ir_idx);
          GE_ASSERT(iter != ir_input_2_range.end(), "sub node[%zu] input ir idx[%zu] is not found", node_idx, ir_idx);
          // level2_addr
          uint64_t level_num = iter->second.second & kBitFlag8;
          level_num |= kLevel2BitFlagWithShape;
          l0_dump_list_.push_back(level_num);
          // level1
          for (size_t i = 0UL; i < iter->second.second; ++i) {
            InsertL0DumpList(node_idx, iter->second.first + i, true);
          }

          sub_node_level_addr_idx[node_idx].emplace_back(io_addrs_.size());
          sub_node_dynamic_args_desc[node_idx].emplace_back(arg_format);
          AppendIoAddr(0UL, kAbsoluteMemType);
          break;
        }
        case AddrType::OUTPUT_DESC: {
          // l0 exception dump处理
          const size_t ir_idx = static_cast<size_t>(arg_format.ir_idx);
          const auto iter = ir_output_2_range.find(ir_idx);
          GE_ASSERT(iter != ir_output_2_range.end(), "sub node[%zu] output ir idx[%zu] is not found", node_idx, ir_idx);
          // level2_addr
          uint64_t level_num = iter->second.second & kBitFlag8;
          level_num |= kLevel2BitFlagWithShape;
          l0_dump_list_.push_back(level_num);
          // level1
          for (size_t i = 0UL; i < iter->second.second; ++i) {
            InsertL0DumpList(node_idx, iter->second.first + i, false);
          }

          sub_node_level_addr_idx[node_idx].emplace_back(io_addrs_.size());
          sub_node_dynamic_args_desc[node_idx].emplace_back(arg_format);
          AppendIoAddr(0UL, kAbsoluteMemType);
          break;
        }
        case AddrType::INPUT: {
          GE_ASSERT_SUCCESS(AppendInputOutputAddr(node_idx, static_cast<size_t>(arg_format.ir_idx), true));
          break;
        }
        case AddrType::OUTPUT: {
          GE_ASSERT_SUCCESS(AppendInputOutputAddr(node_idx, static_cast<size_t>(arg_format.ir_idx), false));
          break;
        }
        case AddrType::WORKSPACE: {
          GE_ASSERT_SUCCESS(AppendWorkspaceAddr(node_idx, arg_format.ir_idx));
          break;
        }
        case AddrType::FFTS_ADDR: {
          uint64_t mode_addr = 0U;
          uint32_t len = 0U;
          GE_CHK_RT_RET(rtGetC2cCtrlAddr(&mode_addr, &len));
          l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max()); // 占位
          AppendIoAddr(mode_addr, kAbsoluteMemType);
          break;
        }
        case AddrType::PLACEHOLDER: {
          l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max()); // 占位
          AppendIoAddr(0UL, kAbsoluteMemType);
          break;
        }
        case AddrType::HIDDEN_INPUT: {
          const HiddenInputsType hi_type = *reinterpret_cast<const HiddenInputsType *>(arg_format.reserved);
          if (HcomOmeUtil::IsHCOMOp(sub_node_op_desc_list_[node_idx]->GetType())) {
            GE_ASSERT_SUCCESS(SetHcomAttr(node_idx));
          }
          std::vector<void *> context_addrs;
          GE_ASSERT_SUCCESS(ArgsFormatUtils::GetHcomHiddenInputs(sub_node_op_desc_list_[node_idx],
                                                                 *davinci_model_, context_addrs, hi_type));
          const size_t ir_idx = static_cast<size_t>(arg_format.ir_idx);
          GE_ASSERT_TRUE(ir_idx < context_addrs.size());
          l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max());  // 占位
          AppendIoAddr(PtrToValue(context_addrs[ir_idx]), kAbsoluteMemType);
          break;
        }
        case AddrType::EVENT_ADDR: {
          const uint32_t mem_event_id = static_cast<uint32_t>(arg_format.ir_idx);
          l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max());  // 占位
          AppendIoAddr(PtrToValue(davinci_model_->GetMemEventIdAddr(mem_event_id)), kAbsoluteMemType);
          break;
        }
        case AddrType::OP_TYPE: {
          std::shared_ptr<TilingContextAddr> default_ctx_ptr = nullptr;
          std::shared_ptr<TilingContextAddr> tiling_context_addr =
              sub_node_op_desc_list_[node_idx]->TryGetExtAttr(kTilingContextAddrs, default_ctx_ptr);
          GE_ASSERT_NOTNULL(tiling_context_addr);
          l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max());  // 占位
          AppendIoAddr(tiling_context_addr->op_type_addr, kAbsoluteMemType);
          break;
        }
        case AddrType::TILING_CONTEXT: {
          GE_ASSERT_SUCCESS(AssembleTilingContextArgs(node_idx, arg_format, idx_to_sink_tensor_map[node_idx]));
          if (arg_format.ir_idx == static_cast<int32_t>(TilingContextSubType::TILING_DATA)) {
            uint64_t tiling_data_size = kMaxTilingDataSize;
            int64_t max_size = -1;
            if (ge::AttrUtils::GetInt(sub_node_op_desc_list_[node_idx], kMaxTilingSize, max_size) && max_size > 0) {
              tiling_data_size = static_cast<uint64_t>(max_size);
            }
            tiling_data_size &= kBitFlag8;
            tiling_data_size |= kLevel2BitFlagTilingData;
            l0_dump_list_.push_back(tiling_data_size);
          } else {
            l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max());
          }
          break;
        }
        case AddrType::OVERFLOW_ADDR: {
          bool has_overflow = (davinci_model_->GetOverflowAddr() != nullptr) &&
                 AttrUtils::HasAttr(sub_node_op_desc_list_[node_idx], GLOBALWORKSPACE_TYPE);
          if (has_overflow) {
            l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max());  // 占位
            AppendIoAddr(PtrToValue(davinci_model_->GetOverflowAddr()),
                         static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix));
          }
          break;
        }
        case AddrType::TILING:
        case AddrType::CUSTOM_VALUE: {
          GELOGE(FAILED, "super kernel no support args format add type %d", arg_format.addr_type);
          return FAILED;
        }
        default:
          l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max());  // 占位
          break;
      }
    }
    node_idx++;
  }

  GE_ASSERT_SUCCESS(AssembleShapeInfoAddrs(sub_node_dynamic_args_desc, sub_node_level_addr_idx));
  return SUCCESS;
}

Status SuperKernelV2TaskInfo::SetHcomAttr(const size_t node_idx) {
  void *input_data_addr = sub_node_input_addrs_list_[node_idx].empty() ?
                         nullptr : ValueToPtr(sub_node_input_addrs_list_[node_idx][0]);
  void *output_data_addr = sub_node_output_addrs_list_[node_idx].empty() ?
                          nullptr : ValueToPtr(sub_node_output_addrs_list_[node_idx][0]);
  void *ws_addr = sub_node_workspace_addrs_list_[node_idx].empty() ?
                          nullptr : ValueToPtr(sub_node_workspace_addrs_list_[node_idx][0]);
  auto &hcom_opdesc = sub_node_op_desc_list_[node_idx];
  GE_ASSERT_TRUE(AttrUtils::SetInt(hcom_opdesc, "_skn_hcom_input_addr", PtrToValue(input_data_addr)));
  GE_ASSERT_TRUE(AttrUtils::SetInt(hcom_opdesc, "_skn_hcom_output_addr", PtrToValue(output_data_addr)));
  GE_ASSERT_TRUE(AttrUtils::SetInt(hcom_opdesc, "_skn_hcom_ws_addr", PtrToValue(ws_addr)));

  bool is_refresh_addr_op = false;
  auto &input_mem_types = sub_node_input_mem_types_list_[node_idx];
  auto &output_mem_types = sub_node_output_mem_types_list_[node_idx];
  auto &workspace_mem_types = sub_node_workspace_mem_types_list_[node_idx];
  for (auto &input_mem_type : input_mem_types) {
    is_refresh_addr_op |= ModelUtils::IsSuppoprtAddrRefreshable(input_mem_type);
  }
  for (auto &output_mem_type : output_mem_types) {
    is_refresh_addr_op |= ModelUtils::IsSuppoprtAddrRefreshable(output_mem_type);
  }
  for (auto &workspace_mem_type : workspace_mem_types) {
    is_refresh_addr_op |= ModelUtils::IsSuppoprtAddrRefreshable(workspace_mem_type);
  }
  is_refresh_addr_op = !davinci_model_->IsStaticAddrFixed() && davinci_model_->IsFeatureBaseRefreshable()
                       && is_refresh_addr_op;
  GE_ASSERT_TRUE(AttrUtils::SetBool(hcom_opdesc, "_skn_hcom_need_refresh", is_refresh_addr_op));
  GELOGI("set hcom %s input addr %p, output addr %p, ws addr %p, is_refresh_addr_op %d, "
         "IsStaticAddrFixed %d, IsFeatureBaseRefreshable %d", hcom_opdesc->GetNamePtr(), input_data_addr,
          output_data_addr, ws_addr, is_refresh_addr_op, davinci_model_->IsStaticAddrFixed(),
          davinci_model_->IsFeatureBaseRefreshable());
  return SUCCESS;
}

Status SuperKernelV2TaskInfo::InitContext(const domi::KernelContext &context) {
  if ((context.args_offset().size() / sizeof(uint16_t)) < 1U) {
    REPORT_INNER_ERR_MSG("E19999", "args_offset().size():%zu / sizeof(uint16_t) less than 1, op:%s(%s), check invalid",
                       context.args_offset().size(), op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    GELOGE(FAILED, "[Check][Param]invalid, args_offset().size():%zu / sizeof(uint16_t) less than 1, op:%s(%s)",
           context.args_offset().size(), op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    return FAILED;
  }

  uint16_t args_offset = 0U;
  GE_ASSERT_EOK(memcpy_s(&args_offset, sizeof(uint16_t), context.args_offset().data(), sizeof(uint16_t)));
  GE_CHECK_LE(args_offset, args_size_);
  io_addr_offset_ = static_cast<size_t>(args_offset);
  GELOGD("Get args_offset[%u] of op[%s]", static_cast<uint32_t>(args_offset), op_desc_->GetName().c_str());
  return SUCCESS;
}


Status SuperKernelV2TaskInfo::InitTask(const domi::KernelDef &kernel_def) {
  GELOGD("Do InitTask of %s.", op_desc_->GetName().c_str());
  GE_CHK_STATUS_RET_NOLOG(InitContext(kernel_def.context()));

  cfg_.schemMode = static_cast<uint8_t>(kernel_def.schedule_mode() & k2BitsMask);
  GELOGD("OpName: %s set schedule mode from kernel def: %u",
      op_desc_->GetName().c_str(), static_cast<uint32_t>(cfg_.schemMode));

  // load with queue 零拷贝场景未作适配
  return SUCCESS;
}

Status SuperKernelV2TaskInfo::InitKernel(const domi::TaskDef &task_def, const PisToArgs &args) {
  GELOGD("SuperKernelV2TaskInfo init start, kernel_type: %d.", static_cast<int32_t>(kernel_type_));
  GE_CHECK_NOTNULL(op_desc_);
  (void)AttrUtils::GetInt(op_desc_, kLocalMemorySize, local_memory_size_);
  const domi::KernelDef &kernel_def = task_def.kernel();
  is_block_task_prefetch_ = kernel_def.is_block_task_prefetch();
  const domi::KernelContext &context = kernel_def.context();
  if (context.origin_op_index_size() > CC_FUSION_OP_MAX) {
    REPORT_INNER_ERR_MSG("E19999", "origin_op_index_size:%d is more than CC_FUSION_OP_MAX(%d), op:%s(%s), check invalid",
        context.origin_op_index_size(), CC_FUSION_OP_MAX, op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param]invalid, origin_op_index_size:%d is more than CC_FUSION_OP_MAX(%d), op:%s(%s)",
           context.origin_op_index_size(), CC_FUSION_OP_MAX, op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    return PARAM_INVALID;
  }

  // Old model will not take this value, its default value is 0,need to convert to the real default value 1.
  block_dim_ = (kernel_def.block_dim() == 0U) ? 1U : kernel_def.block_dim();
  cfg_.blockDimOffset = kernel_def.block_dim_offset();
  operator_ = davinci_model_->GetOperatorByIndex(context.op_index());
  GE_CHECK_NOTNULL(operator_);

  GE_ASSERT_TRUE((args[static_cast<size_t>(args_placement_)].dev_addr != 0U),
                 "[Check][Param] Op:%s, dev addr is nullptr.", op_desc_->GetName().c_str());
  args_ = ValueToPtr(args[static_cast<size_t>(args_placement_)].dev_addr);

  GE_ASSERT_SUCCESS(AssembleIoByArgsFormat(), "[Assemble][Addresses] failed, op = %s.", op_desc_->GetNamePtr());

  Status ret = InitTask(kernel_def);
  GELOGD("SuperKernelV2TaskInfo %s init finish, result=%u.", op_desc_->GetNamePtr(), ret);

  if ((davinci_model_->OpNeedDump(op_desc_) || davinci_model_->OpNeedPrint(op_desc_))) {
    GELOGI("Op %s need dump or print in task info", op_desc_->GetName().c_str());
    dump_args_ = PtrAdd(PtrToPtr<void, uint8_t>(args_), static_cast<size_t>(args_size_), io_addr_offset_);
    dump_flag_ = RT_KERNEL_DUMPFLAG;
    is_data_dump_ = true;
  }
  return ret;
}

void SuperKernelV2TaskInfo::UpdateTaskId() {
  if (davinci_model_ != nullptr) {
    GE_CHK_RT_EXEC(rtsGetThreadLastTaskId(&task_id_), return);
    GE_CHK_RT_EXEC(rtsStreamGetId(stream_, reinterpret_cast<int32_t*>(&stream_id_)), return);
    GELOGD("UpdateTaskId:UpdateTaskId [%u], stream id [%u]:", task_id_, stream_id_);
  }
}

void SuperKernelV2TaskInfo::PostProcess(const domi::TaskDef &task_def) {
  const auto &context_def = task_def.kernel().context();
  davinci_model_->SaveDfxInfo(context_def.op_index(), task_def, *this);
  ResetArgsEx();
}

REGISTER_TASK_INFO(MODEL_TASK_SUPER_KERNEL, SuperKernelV2TaskInfo);
}  // namespace ge
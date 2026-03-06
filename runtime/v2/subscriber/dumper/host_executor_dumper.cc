/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host_executor_dumper.h"

#include <utility>
#include <aicore/launch_kernel/ai_core_launch_kernel.h>
#include <anchor_utils.h>
#include <tuning_utils.h>
#include <dlog_pub.h>
#include <anchor.h>
#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include "common/checker.h"
#include "common/ge_inner_error_codes.h"
#include "mmpa/mmpa_api.h"
#include "common/debug/memory_dumper.h"
#include "graph/utils/tensor_utils.h"
#include "framework/common/string_util.h"
#include "runtime/subscriber/global_dumper.h"
#include "runtime/subscriber/built_in_subscriber_definitions.h"
#include "framework/common/util.h"

namespace {
constexpr int32_t kDecimal = 10;
}

namespace gert {
std::mutex HostExecutorDumper::mutex_;

HostExecutorDumper::HostExecutorDumper(const std::shared_ptr<const SubscriberExtendInfo> &extend_info)
    : ExecutorDumper(extend_info), extend_info_(std::move(extend_info)) {
  if ((extend_info_ == nullptr) || (extend_info_->exe_graph == nullptr) || (extend_info_->executor == nullptr)) {
    GELOGE(ge::FAILED, "Exe graph is nullptr, dumper will do nothing.");
    return;
  }
  if (IsEnable(DumpType::kHostDump)) {
    if (Init() != ge::SUCCESS) {
      GELOGE(ge::FAILED, "Init dumper failed, dumper does nothing.");
      return;
    }
  }
}

ge::Status HostExecutorDumper::GetDumpAddrFromChainAddrOnHost(const NodeDumpUnit &dump_unit, bool is_input,
                                                              std::vector<uintptr_t> &dump_addrs) const {
  const auto &chain_addrs = is_input ? dump_unit.input_addrs : dump_unit.output_addrs;
  for (size_t i = 0UL; i < chain_addrs.size(); ++i) {
    if (chain_addrs[i] == nullptr) {
      dump_addrs.emplace_back(reinterpret_cast<uintptr_t>(chain_addrs[i]));
      continue;
    }

    const auto tensor_data = chain_addrs[i]->GetPointer<TensorData>();
    GE_ASSERT_NOTNULL(tensor_data);
    auto host_addr = tensor_data->GetAddr();
    dump_addrs.emplace_back(reinterpret_cast<uintptr_t>(host_addr));
  }
  return ge::SUCCESS;
}

void HostExecutorDumper::SetOpDescInfo(NodeDumpUnit &dump_unit, ge::OpDescPtr &op_desc, ge::OpDescInfo &op_desc_info,
                                       const std::vector<uintptr_t> &input_addrs,
                                       const std::vector<uintptr_t> &output_addrs) const {
  op_desc_info.op_name = dump_unit.node->GetName();
  op_desc_info.op_type = dump_unit.node->GetType();
  int64_t desc_size = 0;
  for (size_t in_index = 0UL; in_index < dump_unit.input_addrs.size(); ++in_index) {
    const auto &in_tensor_desc = op_desc->MutableInputDesc(in_index);
    if ((in_tensor_desc == nullptr) || (input_addrs[in_index] == 0U)) {
      continue;
    }
    op_desc_info.input_data_type.emplace_back(in_tensor_desc->GetDataType());
    op_desc_info.input_format.emplace_back(in_tensor_desc->GetFormat());
    GE_CHK_STATUS(ge::TensorUtils::GetTensorSizeInBytes(*in_tensor_desc, desc_size), "op %s input %zu",
                  dump_unit.node->GetName().c_str(), in_index);
    op_desc_info.input_size.emplace_back(desc_size);
    op_desc_info.input_shape.emplace_back(in_tensor_desc->GetShape().GetDims());
    op_desc_info.input_addrs.emplace_back(reinterpret_cast<void *>(input_addrs[in_index]));
  }

  for (size_t out_index = 0UL; out_index < dump_unit.output_addrs.size(); ++out_index) {
    const auto &out_tensor_desc = op_desc->MutableOutputDesc(out_index);
    if ((out_tensor_desc == nullptr) || (output_addrs[out_index] == 0U)) {
      continue;
    }
    op_desc_info.output_data_type.emplace_back(out_tensor_desc->GetDataType());
    op_desc_info.output_format.emplace_back(out_tensor_desc->GetFormat());
    GE_CHK_STATUS(ge::TensorUtils::GetTensorSizeInBytes(*out_tensor_desc, desc_size), "op %s output %zu",
                  dump_unit.node->GetName().c_str(), out_index);
    op_desc_info.output_size.emplace_back(desc_size);
    op_desc_info.output_shape.emplace_back(out_tensor_desc->GetShape().GetDims());
    op_desc_info.output_addrs.emplace_back(reinterpret_cast<void *>(output_addrs[out_index]));
  }

  const auto workspace_size = op_desc->GetWorkspaceBytes();
  for (size_t i = 0U; (i < workspace_size.size()) && (i < dump_unit.workspace_info.size()); ++i) {
    op_desc_info.space_addrs.emplace_back(reinterpret_cast<void *>(dump_unit.workspace_info[i].first));
    op_desc_info.workspace_bytes.emplace_back(workspace_size[i]);
  }
}

bool HostExecutorDumper::IsInDumpStep(const int64_t step_id, const std::string &dump_step) {
  if (!dump_step.empty()) {
    const auto step = step_set_.find(step_id);
    if (step != step_set_.end()) {
      return true;
    }
    for (size_t i = 0UL; i < step_range_.size(); i++) {
      if ((step_id >= step_range_[i].first) && (step_id <= step_range_[i].second)) {
        return true;
      }
    }
    return false;
  }
  return true;
}

ge::Status HostExecutorDumper::DoHostDataDump(NodeDumpUnit &dump_unit, const ge::DumpProperties &dump_properties) {
  // todo  default session id
  const auto name = dump_unit.node->GetName();
  const auto type = dump_unit.node->GetType();
  GELOGI("[Dumper] Start to dump, op name: %s, type: %s", name.c_str(), type.c_str());
  // todo change with DoDataDump
  if (type == "Identity" || type == "MemcpyAsync") {
    return ge::SUCCESS;
  }

  void *step_id = nullptr;
  GE_ASSERT_RT_OK(aclrtMalloc(&step_id, sizeof(uint64_t), ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  const auto callback = [&dump_unit, &step_id]() {
    GE_CHK_RT(aclrtFree(step_id));
    dump_unit.Clear();
  };
  GE_MAKE_GUARD(dump_release, callback); 

  // todo single_op and graph have different switch, which will be normalized
  const std::string model_name = extend_info_->model_name;
  if (!dump_properties.IsOpDebugOpen() &&
      !dump_properties.IsLayerNeedDump(model_name, extend_info_->model_data.om_name, name) &&
      !dump_properties.IsSingleOpNeedDump()) {
    GELOGI("[Dumper] [%s] is not in dump list, no need to dump", name.c_str());
    return ge::SUCCESS;
  }

  // todo model_id does not matter
  GELOGD("[Data][Dumper]model name[%s], model id[%u].", model_name.c_str(), extend_info_->model_id);
  const auto op_desc = dump_unit.node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  ge::OpDescPtr op_desc_dump = nullptr;
  GE_MAKE_SHARED(op_desc_dump = std::make_shared<ge::OpDesc>(*op_desc), return ge::FAILED);
  dump_unit.UpdateInputShapes(op_desc_dump);
  dump_unit.UpdateOutputShapes(op_desc_dump);
  std::vector<uintptr_t> input_addrs;
  if (GetDumpAddrFromChainAddrOnHost(dump_unit, true, input_addrs) != ge::SUCCESS) {
    // skip and continue dump other nodes
    return ge::SUCCESS;
  }
  std::vector<uintptr_t> output_addrs;
  if (GetDumpAddrFromChainAddrOnHost(dump_unit, false, output_addrs) != ge::SUCCESS) {
    // skip and continue dump other nodes
    return ge::SUCCESS;
  }

  if ((output_addrs.size() != op_desc->GetAllOutputsDescSize()) ||
      (input_addrs.size() != op_desc->GetAllInputsSize())) {
    GELOGW(
        "[Dumper] Node %s input addr or output addr size is invalid, input addr size is %zu, output addr size is %zu, "
        "op desc input size is %zu, output size is %zu.",
        name.c_str(), input_addrs.size(), output_addrs.size(), op_desc->GetInputsSize(), op_desc->GetOutputsSize());
    // skip and continue dump
    return ge::SUCCESS;
  }

  ge::OpDescInfo op_desc_info;
  ge::ExceptionDumper exception_dumper;
  SetOpDescInfo(dump_unit, op_desc_dump, op_desc_info, input_addrs, output_addrs);
  auto iteration_num = GetIterationNum();
  GE_ASSERT_RT_OK(aclrtMemcpy(step_id, sizeof(uint64_t), &iteration_num,
      sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
  GELOGD("[Dumper] Is single op %d", static_cast<int32_t>(dump_properties.IsSingleOpNeedDump()));
  const auto &dump_step = dump_properties.GetDumpStep();
  if (!IsInDumpStep(static_cast<int64_t>(iteration_num), dump_step)) {
    GELOGW("[Dumper] The current step is %lld, not in the dump step", iteration_num);
    return ge::SUCCESS;
  }
  const auto dump_path = dump_properties.GetDumpPath();
  const auto worker_id = dump_properties.GetDumpWorkerId();
  GELOGD("[Dumper] Get worker id succeed, value is %s.", worker_id.c_str());
  std::string file_name = model_name;
  replace(file_name.begin(), file_name.end(), '/', '_');
  replace(file_name.begin(), file_name.end(), '\\', '_');
  replace(file_name.begin(), file_name.end(), '.', '_');
  std::string file_path = "/";
  if (!dump_path.empty()) {
    const std::lock_guard<std::mutex> lock(mutex_);
    // multiprocess shared directory creates first to avoid fuilure of creating full path
    const std::string shared_path = dump_path + "/0/";
    (void)ge::CreateDirectory(shared_path);
    file_path = shared_path + "pid" + std::to_string(mmGetPid()) + "/" + file_name + "/" +
                std::to_string(extend_info_->model_id) + "/" +
                worker_id + "/" + std::to_string(iteration_num) + "/";
    const int32_t directory_ret = ge::CreateDirectory(file_path);
    if (directory_ret != 0) {
        GELOGW("Can not create directory[%s].", file_path.c_str());
        return ge::SUCCESS;
    }
  }
  GE_CHK_STATUS_RET(exception_dumper.DumpNodeInfo(op_desc_info, file_path, false, false, dump_properties),
                    "Dump op %s failed", name.c_str());
  GELOGI("[Dumper] Launch dump op:%s Successfully", name.c_str());
  return ge::SUCCESS;
}

ge::Status HostExecutorDumper::OnUpdateDumpUnitForHostDump(const Node &node) {
  for (auto &dump_unit : kernel_idxes_to_dump_units_[node.node_id]) {
    ++dump_unit->cur_update_count;
    GE_ASSERT_SUCCESS(SaveWorkSpaceAddrForAiCpuLaunchCCNode(node));

    if (dump_unit->cur_update_count == dump_unit->total_update_count) {
      const auto &dump_properties = ge::DumpManager::GetInstance().GetDumpProperties(GetSessionId());
      GE_ASSERT_SUCCESS(DoHostDataDump(*dump_unit, dump_properties));
    }
  }
  return ge::SUCCESS;
}

void HostExecutorDumper::ParseDumpStep() {
  if (is_parsed_) {
    return;
  }
  const auto &dump_properties = ge::DumpManager::GetInstance().GetDumpProperties(GetSessionId());
  const auto &dump_step = dump_properties.GetDumpStep();
  if (!dump_step.empty()) {
    auto match_vecs = ge::StringUtils::Split(dump_step, '|');
    const std::regex pattern(R"(\d{1,})");
    std::smatch result;
    for (const auto &match_vec : match_vecs) {
      if (regex_match(match_vec, result, pattern)) {
        step_set_.insert(static_cast<int64_t>(std::strtol(match_vec.c_str(), nullptr, kDecimal)));
        GELOGI("[HostDumper] Insert one step:%d", std::strtol(match_vec.c_str(), nullptr, kDecimal));
      } else {
        auto vec_split = ge::StringUtils::Split(match_vec, '-');
        const auto lower_range = static_cast<int64_t>(std::strtol(vec_split[0UL].c_str(), nullptr, kDecimal));
        const auto higher_range = static_cast<int64_t>(std::strtol(vec_split[1UL].c_str(), nullptr, kDecimal));
        step_range_.emplace_back(lower_range, higher_range);
        GELOGI("[HostDumper] Insert step range from %d to %d", lower_range, higher_range);
      }
    }
  }
  is_parsed_ = true;
  return;
}

ge::Status HostExecutorDumper::HostDataDump(const Node *node, ExecutorEvent event) {
  if (event == ExecutorEvent::kModelStart) {
    GE_ASSERT_SUCCESS(Init());
    SaveSessionId();
    ParseDumpStep();
  }
  if (event == ExecutorEvent::kExecuteEnd) {
    GE_ASSERT_SUCCESS(OnUpdateDumpUnitForHostDump(*node));
  }
  if (event == ExecutorEvent::kModelEnd) {
    CountIterNum();
  }
  return ge::SUCCESS;
}

void HostExecutorDumper::OnExecuteEvent(int32_t sub_exe_graph_type, HostExecutorDumper *dumper, ExecutorEvent event,
                                        const void *node, KernelStatus result) {
  (void)sub_exe_graph_type;
  (void)result;
  if (dumper == nullptr) {
    return;
  }
  if (dumper->IsEnable(DumpType::kHostDump)) {
    (void)dumper->HostDataDump(static_cast<const Node *>(node), event);
    return;
  }
}
}  // namespace gert

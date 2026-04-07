/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DATA_DUMPER_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DATA_DUMPER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "common/dump/dump_properties.h"
#include "graph/node.h"
#include "graph/compute_graph.h"
#include "proto/ge_ir.pb.h"
#include "proto/op_mapping.pb.h"
#include "runtime/mem.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "framework/common/ge_types.h"
#include "runtime/base.h"
#include "dump/adump_pub.h"
#include "dump/adump_api.h"

namespace ge {
struct FirstLevelAddressInfo {
  bool address_type;
  std::vector<uintptr_t> address;
};

struct LayerOpOnWatcherModeInfo {
  uint32_t task_id;
  uint32_t stream_id;
  std::shared_ptr<OpDesc> op_desc;
};

class DataDumper {
 public:
  struct InnerRealAddressAndSize {
    uint64_t address;
    uint64_t size;
  };

  struct InnerContext {
    uint32_t context_id;
    uint32_t thread_id;
    std::vector<InnerRealAddressAndSize> input;
    std::vector<InnerRealAddressAndSize> output;
  };

  explicit DataDumper(RuntimeParam *const rsh) : runtime_param_(rsh) {
    InitAdumpCapability();
  }

  ~DataDumper() noexcept;

  void SetModelName(const std::string &model_name) { model_name_ = model_name; }

  void SetModelId(const uint32_t model_id) { model_id_ = model_id; }

  void SetDeviceId(const uint32_t device_id) { device_id_ = device_id; }

  void SetComputeGraph(const ComputeGraphPtr &compute_graph) { compute_graph_ = compute_graph; };

  void SetRefInfo(const std::map<OpDescPtr, void *> &ref_info) { ref_info_ = ref_info; };

  void SetL1FusionAddr(const uintptr_t addr) { l1_fusion_addr_ = addr; };

  void SetWorkSpaceAddr(const std::shared_ptr<OpDesc> &op_desc, const std::vector<uint64_t> &space_addr);
  void SetWorkSpaceAddrForPrint(const std::shared_ptr<OpDesc> &op_desc, const std::vector<uint64_t> &space_addr);
  void SetLoopAddr(const uintptr_t global_step, const uintptr_t loop_per_iter, const uintptr_t loop_cond);

  void SaveDumpInput(const std::shared_ptr<Node> &node);

  // args is device memory stored first output addr
  void SaveDumpTask(const OpDescInfoId &id, const std::shared_ptr<OpDesc> &op_desc, const uintptr_t args,
                    const FirstLevelAddressInfo &first_level_address_info = {false, {}},
                    const std::map<uint64_t, uint64_t> &cust_to_relevant_offset = {},
                    const ModelTaskType task_type = ModelTaskType::MODEL_TASK_KERNEL, bool is_op_debug = false, rtStream_t stream = nullptr);

  void SavePrintDumpTask(const OpDescInfoId &id, const std::shared_ptr<OpDesc> &op_desc, const uintptr_t args,
                         const FirstLevelAddressInfo &first_level_address_info = {false, {}},
                         const ModelTaskType task_type = ModelTaskType::MODEL_TASK_KERNEL, rtStream_t stream = nullptr);

  void SaveEndGraphId(const uint32_t task_id, const uint32_t stream_id);

  void SetOmName(const std::string &om_name) { om_name_ = om_name; }
  void SetSingleOpDebug() { is_single_op_debug_ = true; }
  Status SaveOpDebugId(const uint32_t task_id, const uint32_t stream_id, const void *const op_debug_addr, const bool is_op_debug);

  Status LoadDumpInfo();

  void UnloadDumpInfo();

  Status ReLoadDumpInfo();

  Status UnloadDumpInfoByModel(uint32_t model_id);

  void DumpShrink();

  void SetDumpProperties(const DumpProperties &dump_properties) { dump_properties_ = dump_properties; }
  const DumpProperties &GetDumpProperties() const { return dump_properties_; }

  void SaveLayerOpInfoOnWatcherMode(LayerOpOnWatcherModeInfo &op_info) {
    layer_op_on_watcher_mode_list_.emplace_back(std::move(op_info));
  }
  void InitAdumpCapability();
  bool IsDumpOpWithAdump() const;
 private:
  void PrintCheckLog(std::string &dump_list_key);

  std::string model_name_;

  // for inference data dump
  std::string om_name_;

  uint32_t model_id_{0U};
  RuntimeParam *runtime_param_;
  void *dev_mem_load_{nullptr};
  void *dev_mem_unload_{nullptr};
  void *dev_mem_unload_for_model_{nullptr};

  struct InnerDumpInfo;
  struct InnerInputMapping;

  std::vector<OpDescInfo> op_desc_info_;
  std::vector<InnerDumpInfo> op_list_;  // release after DavinciModel::Init
  std::vector<InnerDumpInfo> op_print_list_;  // release after DavinciModel::Init
  std::vector<LayerOpOnWatcherModeInfo> layer_op_on_watcher_mode_list_;
  uint32_t end_graph_task_id_{0U};
  uint32_t end_graph_stream_id_{0U};
  bool is_end_graph_ = false;
  std::multimap<std::string, InnerInputMapping> input_map_;  // release after DavinciModel::Init
  bool load_flag_{false};
  uint32_t device_id_{0U};
  uintptr_t global_step_{0U};
  uintptr_t loop_per_iter_{0U};
  uintptr_t loop_cond_{0U};
  ComputeGraphPtr compute_graph_;  // release after DavinciModel::Init
  std::map<OpDescPtr, void *> ref_info_;     // release after DavinciModel::Init
  uintptr_t l1_fusion_addr_{0U};
  uint32_t op_debug_task_id_{0U};
  uint32_t op_debug_stream_id_{0U};
  const void *op_debug_addr_{nullptr};
  bool is_op_debug_ = false;
  bool is_single_op_debug_ = false;
  bool need_generate_op_buffer_ = false;
  DumpProperties dump_properties_;
  InnerContext context_;
  toolkit::aicpu::dump::OpMappingInfo op_mapping_info_;
  bool overflow_enabled_{false};
  bool persistent_unlimited_enabled_{false};
  bool adump_interface_available_{false};
  // Build task info of op mapping info
  Status BuildTaskInfoForDumpOutput(toolkit::aicpu::dump::OpMappingInfo &op_mapping_info,
                                    const InnerDumpInfo &dump_info, toolkit::aicpu::dump::Task &task);
  Status BuildTaskInfoForDumpInput(toolkit::aicpu::dump::OpMappingInfo &op_mapping_info,
                                   const InnerDumpInfo &dump_info, toolkit::aicpu::dump::Task &task);
  Status BuildTaskInfoForDumpAllorOpDebug(toolkit::aicpu::dump::OpMappingInfo &op_mapping_info,
                                          const InnerDumpInfo &dump_info, toolkit::aicpu::dump::Task &task);
  void InitAicpuDumpTask(const InnerDumpInfo &dump_info, toolkit::aicpu::dump::Task &task) const;
  Status BuildTaskInfo(toolkit::aicpu::dump::OpMappingInfo &op_mapping_info);
  Status BuildTaskInfoForPrint(toolkit::aicpu::dump::OpMappingInfo &op_mapping_info);
  Status GetDumpPath(std::string &dump_path);
  void DumpWorkspace(const InnerDumpInfo &inner_dump_info, toolkit::aicpu::dump::Task &task,
                     const std::shared_ptr<OpDesc> &op_desc) const;
  Status DumpOutput(const InnerDumpInfo &inner_dump_info, toolkit::aicpu::dump::Task &task);
  Status DumpRefOutput(const DataDumper::InnerDumpInfo &inner_dump_info, toolkit::aicpu::dump::Output &output,
                       const size_t i, const std::string &node_name_index);
  Status DumpOutputWithTask(const InnerDumpInfo &inner_dump_info, toolkit::aicpu::dump::Task &task);
  Status DumpOutputWithRawAddress(const InnerDumpInfo &inner_dump_info, toolkit::aicpu::dump::Task &task);
  Status DumpInput(const InnerDumpInfo &inner_dump_info, toolkit::aicpu::dump::Task &task);
  Status DumpInputWithRawAddress(const InnerDumpInfo &inner_dump_info, toolkit::aicpu::dump::Task &task);
  Status DumpRefInput(const DataDumper::InnerDumpInfo &inner_dump_info, toolkit::aicpu::dump::Input &input,
                      const size_t i, const std::string &node_name_index);
  void DumpContext(const InnerDumpInfo &inner_dump_info, toolkit::aicpu::dump::Task &task);
  Status ExecuteLoadDumpInfo(const toolkit::aicpu::dump::OpMappingInfo &op_mapping_info);
  void SetEndGraphIdToAicpu(toolkit::aicpu::dump::OpMappingInfo &op_mapping_info);
  void SetOpDebugIdToAicpu(const uint32_t task_id, const uint32_t stream_id, const void *const op_debug_addr,
                           toolkit::aicpu::dump::OpMappingInfo &op_mapping_info) const;
  Status ExecuteUnLoadDumpInfo(const toolkit::aicpu::dump::OpMappingInfo &op_mapping_info);
  Status GenerateInput(toolkit::aicpu::dump::Input &input, const OpDesc::Vistor<GeTensorDescPtr> &tensor_descs,
                       const uintptr_t addr, const size_t index, const uint64_t offset = 0U);
  Status GenerateOutput(toolkit::aicpu::dump::Output &output, const OpDesc::Vistor<GeTensorDescPtr> &tensor_descs,
                        const uintptr_t addr, const size_t index, const uint64_t offset = 0U);
  void GenerateOpBuffer(const uint64_t size, toolkit::aicpu::dump::Task &task);
  uint64_t GetOffset(const InnerDumpInfo &inner_dump_info, const size_t i, const size_t input_size) const;
  Status GetOffsetFromJson(const InnerDumpInfo &inner_dump_info, size_t tensor_idx, size_t input_size,
                           uint64_t &offset) const;
  Status UpdateOpMappingInfo();
  Status DumpOpWithAdump(const InnerDumpInfo &dump_info);
  void FillWorkspaceTensorInfos(const InnerDumpInfo &dump_info, std::vector<Adx::TensorInfo>& tensors) const;
  void FillInputTensorInfos(const OpDescPtr &op_desc, uintptr_t args_base,
                            const std::map<uint64_t, uint64_t>& cust_offset,
                            std::vector<Adx::TensorInfo>& tensors);
  void FillOutputTensorInfos(const OpDescPtr &op_desc, uintptr_t args_base, size_t input_count,
                            const std::map<uint64_t, uint64_t>& cust_offset,
                            std::vector<Adx::TensorInfo>& tensors);
  Status FillRawTensorInfos(const InnerDumpInfo &dump_info, std::vector<Adx::TensorInfo> &tensors,
                          bool dump_input = true, bool dump_output = true) const;
  std::vector<Adx::DumpAttr> BuildDumpAttrs() const;
  bool IsInInputOpBlackIist(const std::shared_ptr<OpDesc>& op_desc, size_t index) const;
  bool IsInOutputOpBlackIist(const std::shared_ptr<OpDesc>& op_desc, size_t index) const;
};
struct DataDumper::InnerDumpInfo {
  uint32_t task_id;
  uint32_t stream_id;
  uint32_t context_id;
  uint32_t thread_id;
  std::shared_ptr<OpDesc> op;
  uintptr_t args;
  bool is_task;
  int32_t input_anchor_index;
  int32_t output_anchor_index;
  std::vector<int64_t> dims;
  int64_t data_size;
  bool is_raw_address;
  std::vector<uintptr_t> address;
  std::vector<uint64_t> space_addr;
  std::map<uint64_t, uint64_t> cust_to_relevant_offset_;
  ModelTaskType task_type;
  bool is_op_debug;  // for op debug
  rtStream_t stream;
};

struct DataDumper::InnerInputMapping {
  std::shared_ptr<OpDesc> data_op;
  int32_t input_anchor_index;
  int32_t output_anchor_index;
};
}  // namespace ge

#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DATA_DUMPER_H_

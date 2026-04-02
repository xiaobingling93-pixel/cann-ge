/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_CPU_QUEUE_SCHEDULE_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_CPU_QUEUE_SCHEDULE_H_

#include <cstdint>
#include "acl/acl_rt.h"
#include "framework/common/ge_types.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/load/model_manager/zero_copy_offset.h"

namespace ge {
constexpr uint32_t kMaxDumpStepStrLen = 1024U;
constexpr uint32_t kReservedParamsNum = 30U;

// For AICPU task "modelDequeue" / "modelEnqueue"
struct MbufQueueInfo {
  uint32_t queue_id;        // Op queue id
  uintptr_t in_mbuf;        // addr for input mbuf
};

// For AICPU task "modelReportStatus"
struct ReportStatusInfo {
  uint32_t model_uuid;                // model uuid
  QueueAttrs status_output_queue;    // enqueue queue
  uint32_t input_num;                 // num of input
};

// For AICPU task "markStep"
struct MarkStepInfo {
  uint32_t group_total_count{1U};
  uint32_t group_index{0U};
  uint32_t group_policy{0U};        // load balance policy
  uint64_t step_id_addr{0UL};        // current step id addr
  uint64_t rsv{0UL};                 // for aicpu use
  uint8_t  is_head{0};
  uint64_t reserved[kReservedParamsNum]{0U};
  char_t dump_step[kMaxDumpStepStrLen]{'\0'};
};

// For AICPU task "modelProcessOutput"
enum class ProcessStage : uint32_t {
  kPrepare,
  kPostDynamic,
  kPostStatic
};

struct ProcessOutputInfo {
  uint32_t data_size;       // output Tensor size
  uintptr_t data_addr;      // output Tensor addr
  uintptr_t in_mbuf;        // input mbuf, for fill output mbuf header
  uintptr_t out_mbuf;       // output mbuf addr
};

// For AICPU task "modelZeroCopy"
enum class ZeroCpyType : uint32_t{
  kAllStatic,
  kAllDynamic,
  kMixedCpy
};

struct ZeroCpyArgs {
  ZeroCpyType cpy_type;
  bool has_tensor_desc;
  bool need_distribute;
  std::vector<int32_t> fusion_offsets;
};

struct AddrMapInfo {
  uint32_t addr_num{0U};
  uint64_t src_addr_list{0UL};
  uint64_t dst_addr_list{0UL};
};

struct AddrMapInfoV2 {
  uint32_t addr_num{0U};
  uint64_t src_addr_list{0UL};
  uint64_t dst_addr_list{0UL};
  uint64_t is_no_tiling_list{0UL};
  uint32_t len{0U};
  char_t extend_info[0];
};

struct BatchEnqueueKernelArgs {
  uint32_t num_inputs;
  uint32_t align_interval;
  uint64_t align_offsets_addr;
  uint64_t queue_ids_addr;
  uint64_t mbuf_addrs_addr;
};

struct GatherDequeueKernelArgs {
  uint32_t input_nums;
  int32_t inputs_align_timeout;
  uint32_t inputs_align_max_cache_num;
  uint32_t inputs_align_drop_out;
  uint64_t queue_ids_addr;  // uint32_t
  uint64_t mbuf_addrs_addr; // uintptr
  uint64_t queue_device_ids_addr;  // uint32
  uint64_t queue_device_type_addr; // uint32 0 NPU 1 CPU
};

struct InputCopyAddrMapInfo {
  uint32_t addr_num{0U};
  uint64_t src_addr_list{0UL};
  uint64_t dst_addr_list{0UL};
  uint64_t data_len_list{0UL};
  uint64_t input_fusion_offset_list{0UL};
};

struct GroupInfo {
  uint32_t group_total_count;
  uint32_t group_index;
  uint32_t group_policy;
};

///
/// @ingroup ge
/// @brief CpuTask base, inherit from TaskInfo used for manage.
///
class CpuTaskInfo : public TaskInfo {
 public:
  explicit CpuTaskInfo(aclrtStream const stream);
  ~CpuTaskInfo() override;

  Status Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
              const PisToArgs &args = {}, const PisToPersistentWorkspace &persistent_workspace = {},
              const IowAddrs &iow_addrs = {{}, {}, {}}) override {
    (void)task_def;
    (void)davinci_model;
    (void)args;
    (void)persistent_workspace;
    (void)iow_addrs;
    return SUCCESS;
  }

 protected:
  Status LaunchCpuKernel(const std::string &kernel_name) const;
  void *args_{nullptr};
  uint32_t args_size_{0U};
 private:
  CpuTaskInfo &operator=(const CpuTaskInfo &) & = delete;
  CpuTaskInfo(const CpuTaskInfo &) = delete;
};

///
/// @ingroup ge
/// @brief definiteness queue schedule, bind input queue to task.
///
class CpuTaskModelDequeue : public CpuTaskInfo {
 public:
  explicit CpuTaskModelDequeue(aclrtStream const stream) : CpuTaskInfo(stream) {}
  ~CpuTaskModelDequeue() override = default;

  Status Init(const uint32_t queue_id, uintptr_t &in_mbuf);

  Status Distribute() override;

 private:
  using CpuTaskInfo::Init;
};

///
/// @ingroup ge
/// @brief definiteness queue schedule, bind input queue to task.
///
class CpuTaskModelBatchDequeue : public CpuTaskInfo {
 public:
  explicit CpuTaskModelBatchDequeue(aclrtStream const stream) : CpuTaskInfo(stream) {}
  ~CpuTaskModelBatchDequeue() override = default;

  Status Init(const uint32_t align_interval,
              const std::vector<uint32_t> &queue_ids,
              const std::vector<uint32_t> &align_offsets,
              std::vector<uintptr_t> &in_mbufs);

  Status Distribute() override;

 private:
  using CpuTaskInfo::Init;
};

class CpuTaskModelGatherDequeue : public CpuTaskInfo {
 public:
  explicit CpuTaskModelGatherDequeue(aclrtStream const stream) : CpuTaskInfo(stream) {}
  ~CpuTaskModelGatherDequeue() override = default;

  Status Init(const std::vector<QueueAttrs> &queues, const InputAlignAttrs &align_attrs,
              std::vector<uintptr_t> &in_mbufs);

  Status Distribute() override;

 private:
  using CpuTaskInfo::Init;
};

///
/// @ingroup ge
/// @brief definiteness queue schedule, zero copy.
///
class CpuTaskZeroCopy : public CpuTaskInfo {
 public:
  explicit CpuTaskZeroCopy(aclrtStream const stream) : CpuTaskInfo(stream) {}
  ~CpuTaskZeroCopy() override;

  Status Init(std::vector<uintptr_t> &mbuf_list,
              const std::map<uint32_t, ZeroCopyOffset> &outside_addrs,
              const std::vector<bool> &is_no_tiling_list,
              ZeroCpyArgs &cpy_args);

  Status Distribute() override;

 private:
  using CpuTaskInfo::Init;
  Status InitAddrs(std::vector<uintptr_t> &mbuf_list,
                   const std::map<uint32_t, ZeroCopyOffset> &outside_addrs,
                   const vector_bit_t &is_no_tiling_list,
                   const ZeroCpyArgs &cpy_args);
  void SetAddrs(uintptr_t src_addr, const ZeroCpyArgs &cpy_args, const bool is_no_tiling, const uintptr_t virtual_addr,
                const bool dest_is_tiling, const int32_t fusion_offset);

 private:
  void *src_addr_{nullptr};
  void *dst_addr_{nullptr};
  void *no_tiling_addr_{nullptr};
  void *dest_is_tiling_addr_{nullptr};
  void *fusion_offsets_addr_{nullptr};
  bool has_tensor_desc_ = false;
  uint32_t addr_num_ = 0U;
  std::vector<uint64_t> src_addrs_;
  std::vector<uint64_t> dst_addrs_;
  std::vector<int32_t> no_tilings_;
  std::vector<int32_t> dest_is_tilings_;
  std::vector<int32_t> fusion_offsets_;
};

///
/// @ingroup ge
/// @brief definiteness queue schedule, active original model stream.
///
class CpuTaskProcessOutput : public CpuTaskInfo {
 public:
  CpuTaskProcessOutput(aclrtStream const stream, const ProcessStage stage,
                       const bool out_has_tensor_desc = false)
      : CpuTaskInfo(stream), stage_(stage), out_has_tensor_desc_(out_has_tensor_desc) {}
  ~CpuTaskProcessOutput() override = default;

  Status Init(const uintptr_t addr, const uint32_t size, const uintptr_t in_mbuf, uintptr_t &out_mbuf,
              const InputOutputDescInfo *const output_desc = nullptr);

  Status Distribute() override;

 private:
  using CpuTaskInfo::Init;
  ProcessStage stage_;
  bool out_has_tensor_desc_;
};

///
/// @ingroup ge
/// @brief definiteness queue schedule, mark dump step info.
///
class CpuTaskMarkStep : public CpuTaskInfo {
 public:
  explicit CpuTaskMarkStep(aclrtStream const stream) : CpuTaskInfo(stream) {}
  ~CpuTaskMarkStep() override = default;

  Status Init(const GroupInfo &group_info, const std::string &dump_step,
              const uintptr_t step_id_addr, bool is_head);

  Status Distribute() override;

 private:
  using CpuTaskInfo::Init;
};

class CpuTaskModelEnqueue : public CpuTaskInfo {
 public:
  explicit CpuTaskModelEnqueue(aclrtStream const stream) : CpuTaskInfo(stream) {}
  ~CpuTaskModelEnqueue() override = default;

  Status Init(const uint32_t queue_id, const uintptr_t out_mbuf);

  Status Distribute() override;

 private:
  using CpuTaskInfo::Init;
};

///
/// @ingroup ge
/// @brief definiteness queue schedule, active entry stream.
///
class CpuTaskActiveEntry : public CpuTaskInfo {
 public:
  explicit CpuTaskActiveEntry(aclrtStream const stream) : CpuTaskInfo(stream) {}
  ~CpuTaskActiveEntry() override = default;

  Status Init(aclrtStream const stream);

  Status Distribute() override;

 private:
  using CpuTaskInfo::Init;
  aclrtStream active_stream_{nullptr};
};

///
/// @ingroup ge
/// @brief definiteness queue schedule, wait for end graph.
///
class CpuTaskWaitEndGraph : public CpuTaskInfo {
 public:
  explicit CpuTaskWaitEndGraph(aclrtStream const stream) : CpuTaskInfo(stream) {}
  ~CpuTaskWaitEndGraph() override = default;

  Status Init(const uint32_t model_id);

  Status Distribute() override;

 private:
  using CpuTaskInfo::Init;
};

///
/// @ingroup ge
/// @brief definiteness queue schedule, model report status.
///
class CpuTaskModelReportStatus : public CpuTaskInfo {
 public:
  explicit CpuTaskModelReportStatus(aclrtStream const stream) : CpuTaskInfo(stream) {}
  ~CpuTaskModelReportStatus() override = default;

  Status Init(const uint32_t model_uuid,
              const QueueAttrs &status_output_queue,
              const std::vector<QueueAttrs> &input_queues);

  Status Distribute() override;

 private:
  using CpuTaskInfo::Init;
};

///
/// @ingroup ge
/// @brief definiteness queue schedule, repeat run model.
///
class CpuTaskModelRepeat : public CpuTaskInfo {
 public:
  explicit CpuTaskModelRepeat(aclrtStream const stream) : CpuTaskInfo(stream) {}
  ~CpuTaskModelRepeat() override = default;

  Status Init(const uint32_t model_id);

  Status Distribute() override;

 private:
  using CpuTaskInfo::Init;
};

///
/// @ingroup ge
/// @brief this task is provided for non-zeroCopy inputs such as data->ENTER.
///
class CpuTaskProcessInputsMemCopy : public CpuTaskInfo {
 public:
  explicit CpuTaskProcessInputsMemCopy(aclrtStream const stream) : CpuTaskInfo(stream) {}
  ~CpuTaskProcessInputsMemCopy() override;

  Status Init(const std::vector<uintptr_t> &mbuf_list,
              const std::vector<uintptr_t> &data_addr_list,
              const std::vector<uint64_t> &length_list,
              const std::vector<int32_t> &input_fusion_offset_list);

  Status Distribute() override;

 private:
  using CpuTaskInfo::Init;
  void *src_addr_ = nullptr;
  void *dst_addr_ = nullptr;
  void *len_list_ = nullptr;
  void *input_fusion_offset_list_ = nullptr;
};

class CpuTaskProcessInputsShapeCheck : public CpuTaskInfo {
 public:
  explicit CpuTaskProcessInputsShapeCheck(aclrtStream const stream) : CpuTaskInfo(stream) {}
  ~CpuTaskProcessInputsShapeCheck() override;

  Status Init(const std::vector<uintptr_t> &mbuf_list,
              const std::vector<int32_t> &input_fusion_offset_list);

  Status Distribute() override;

  struct ShapeValidation {
    uint64_t mbuf_addr;
    uint64_t offset;
    char_t rsv[16];
  };

  struct ShapeValidationInfo {
    uint64_t validation_num;
    uint64_t validation_info_device_addr;
  };

 private:
  using CpuTaskInfo::Init;
  void *shape_validation_addr_ = nullptr;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_CPU_QUEUE_SCHEDULE_H_

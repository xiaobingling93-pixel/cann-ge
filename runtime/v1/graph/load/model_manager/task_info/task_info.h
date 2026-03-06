/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_TASK_INFO_H_

#include <vector>
#include <sstream>
#include <array>
#include <set>
#include "common/math/math_util.h"
#include "framework/common/taskdown_common.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/load/model_manager/ts_mem_mall.h"
#include "graph/load/model_manager/task_info/task_info_factory.h"
#include "proto/task.pb.h"
#include "runtime/rt_dfx.h"
#include "aprof_pub.h"
#include "runtime/rts/rts_stream.h"
#include "runtime/rts/rts_kernel.h"

namespace ge {
class DavinciModel;
class TaskInfo;
struct MemInfo {
  int64_t logic_memory_base ;
  int64_t memory_size;
  uint8_t *memory_base;
  uint64_t memory_type;
  std::string memory_key;
  bool is_fixed_addr_prior;
  MemInfo() : MemInfo(0, 0, nullptr, false) {}

  MemInfo(int64_t logic_memory_base_tmp, int64_t memory_size_tmp, uint8_t *const memory_base_tmp,
          bool is_fixed_addr_prior_tmp = false)
      : logic_memory_base(logic_memory_base_tmp),
        memory_size(memory_size_tmp),
        memory_base(memory_base_tmp),
        memory_type(RT_MEMORY_HBM),
        is_fixed_addr_prior(is_fixed_addr_prior_tmp) {}

  friend bool operator<(const MemInfo &left, const MemInfo &right) noexcept {
    // 加上大小是为了地址段匹配处理，不要擅自修改
    // 如logic_memory_base=0, memory_size=100, logic_memory_base=100, memory_size=300
    // logic_addr为[0, 100)匹配到的基地址就是logic_memory_base=0
    // logic_addr为[100, 400)匹配到的基地址就是logic_memory_base=100
    return (left.logic_memory_base + left.memory_size) < (right.logic_memory_base + right.memory_size);
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << "memory_size:" << memory_size << ", logic_memory_base:" << logic_memory_base << ", memory_base:0x"
       << &std::hex << PtrToValue(PtrToPtr<uint8_t, void>(memory_base)) << ", memory_type:" << memory_type
       << ", memory_key:" << memory_key << ", is_fixed_addr_prior:" << is_fixed_addr_prior;
    return ss.str();
  }

  void *GetMemory(const int64_t offset, const int64_t bytes) const {
    if (bytes <= 0) {
      return nullptr;
    }
    GE_CHK_STATUS_EXEC(CheckInt64SubOverflow(offset, logic_memory_base), return nullptr,
        "[Get][Memory] failed,Out of range, total size:%" PRId64 ", offset:%" PRId64 ", logic_memory_base:%" PRId64 ".",
        memory_size, offset, logic_memory_base);
    const int64_t real_offset = offset - logic_memory_base;

    GE_CHK_STATUS_EXEC(CheckInt64AddOverflow(real_offset, bytes), return nullptr,
                       "[Get][Memory] failed,Out of range, total size:%" PRId64 ", offset:%" PRId64 ", bytes:%" PRId64 ".",
                       memory_size, real_offset, bytes);

    if ((real_offset + bytes) <= memory_size) {
      return ValueToPtr(PtrToValue(memory_base) + static_cast<uint64_t>(real_offset));
    }

    REPORT_INNER_ERR_MSG("E19999", "Out of range, total size:%" PRId64 ", offset:%" PRId64 ", bytes:"
		       "%" PRId64 ".", memory_size, real_offset, bytes);
    GELOGE(OUT_OF_MEMORY, "Out of range, total size:%" PRId64 ", offset:%" PRId64 ", bytes:%" PRId64 ".",
      memory_size, real_offset, bytes);
    return nullptr;
  }
};

struct RuntimeParam {
  RuntimeParam() {
    ts_mem_mall = MakeShared<TsMemMall>();
    aicpu_mem_mall = MakeShared<TsMemMall>(RT_MEMORY_HBM);
  }
  ~RuntimeParam() = default;

  std::string ToString() const {
    std::stringstream ss;
    ss << "session_id:" << session_id << ", device_id:" << device_id << ", stream_num:" << stream_num
       << ", notify_num:" << notify_num << ", event_num:" << event_num << ", label_num:" << label_num
       << ", logic_mem_base:" << &std::hex << logic_mem_base << ", host_logic_mem_base:" << host_logic_mem_base
       << ", host_svm_logic_mem_base:" << host_svm_logic_mem_base << ", logic_weight_base:" << logic_weight_base
       << ", logic_var_base:" << logic_var_base << &std::dec << ", memory_size:" << mem_size
       << ", host_mem_size:" << host_mem_size << ", host_svm_size:" << host_svm_size << ", weight_size:"
       << weight_size << ", var_size:" << var_size << ", zero_copy_size:" << zero_copy_size
       << ", fixed_feature_memory_base:" << &std::hex << fixed_mem_base
       << ", fixed_mem_size: " << &std::dec << fixed_mem_size
       << ", p2p_fixed_mem_base: " << &std::hex << p2p_fixed_mem_base
       << ", p2p_fixed_mem_size: " << &std::dec << p2p_fixed_mem_size
       << ", ex_memory_info:";
    for (const auto &it : memory_infos) {
      ss << "[memory_type:" << it.first << ", memory_size:" << it.second.memory_size << "]";
    }
    ss << ", hbm_memory_info:";
    int64_t total_hbm_size = 0;
    for (const auto &it : fm_memory_infos) {
      ss << "[memory_type:" << it.memory_type << ", memory_size:" << it.memory_size << "]";
      total_hbm_size += it.memory_size;
    }
    // total_hbm_size的大小理论上应该等于（mem_size-zero_copy_size）
    ss << ", total_hbm_size: " << total_hbm_size;
    return ss.str();
  }

  void *GetMemAddr(int64_t logic_offset) const {
    MemInfo fm_info{};
    fm_info.logic_memory_base = logic_offset;
    auto it = sorted_memory_infos.upper_bound(fm_info);
    void *memory_addr = nullptr;
    if ((it != sorted_memory_infos.end()) && (logic_offset >= it->logic_memory_base)
        && (logic_offset < (it->logic_memory_base + it->memory_size))) {
      memory_addr = static_cast<void *>(it->memory_base + (logic_offset - it->logic_memory_base));
      GELOGI("logic_offset:%" PRId64 ", logic_memory_base:%" PRId64 ", memory_base:%p, memory_addr:%p",
             logic_offset, it->logic_memory_base, it->memory_base, memory_addr);
    } else {
      memory_addr = ValueToPtr(mem_base + static_cast<uint64_t>(logic_offset));
      GELOGI("logic_offset:%" PRId64 ", memory_base0x:%" PRIx64 ", memory_addr:%p", logic_offset, mem_base, memory_addr);
    }
    return memory_addr;
  }

  uint64_t mem_size = 0U;
  uint64_t logic_mem_base = 0U;
  uintptr_t mem_base = 0U;
  uint64_t host_mem_size = 0U;
  uint64_t host_logic_mem_base = 0U;
  uintptr_t host_mem_base = 0U;
  uint64_t host_svm_size = 0U;
  uint64_t host_svm_logic_mem_base = 0U;
  uintptr_t host_svm_mem_base = 0U;
  uint64_t weight_size = 0U;
  uint64_t logic_weight_base = 0U;
  uintptr_t weight_base = 0U;
  uint64_t var_size = 0U;
  uint64_t logic_var_base = 0U;
  uintptr_t var_base = 0U;
  int64_t zero_copy_size = 0;
  std::map<uint64_t, MemInfo> memory_infos;
  // fm_memory_infos如果有多个的话，当前代表的是动态shape的静态子图的feature map分段场景;
  // 当前只有hbm类型会分段
  // fm_memory_infos中不包含零拷贝的内存信息
  std::vector<MemInfo> fm_memory_infos;
  std::set<MemInfo> sorted_memory_infos;
  uint64_t fixed_mem_base = 0U; // memory type hbm
  uint64_t fixed_mem_size = 0U; // memory type hbm
  uint64_t p2p_fixed_mem_base = 0U;
  uint64_t p2p_fixed_mem_size = 0U;
  std::vector<MemInfo> fixed_fm_memory_infos;
  uint32_t batch_num = 0U;
  uint32_t stream_num = 0U;
  uint32_t notify_num = 0U;
  std::vector<uint32_t> notify_types;
  uint32_t event_num = 0U;
  uint32_t label_num = 0U;
  uint64_t session_id = 0U;
  uint32_t graph_id = 0U;
  bool is_single_op = false;
  uint32_t root_graph_id = 0U;
  uint32_t device_id = 0U;
  std::string graph_name;

  std::shared_ptr<TsMemMall> ts_mem_mall;
  std::shared_ptr<TsMemMall> aicpu_mem_mall;
  std::map<int64_t, uintptr_t> fileconstant_addr_mapping;
};

struct FusionOpInfo {
  std::vector<std::string> original_op_names;
  std::string op_name;
  uint32_t op_index = 0U;
  uint32_t task_id = 0U;
  uint32_t stream_id = 0U;
};

struct MemAllocation {
  enum Type : int32_t {
    INPUT = 0,
    OUTPUT = 1,
    FEATURE_MAP = 2,
    FIXED_FEATURE_MAP = 3,
    ABSOLUTE = 4, // absolute addr, no need to refresh
  };

  static const char_t *GetTypeStr(Type it) {
    if (it == INPUT) {
      return "INPUT";
    } else if (it == OUTPUT) {
      return "OUTPUT";
    } else if (it == FEATURE_MAP) {
      return "FEATURE_MAP";
    } else if (it == FIXED_FEATURE_MAP) {
      return "FIXED_FEATURE_MAP";
    } else if (it == ABSOLUTE) {
      return "ABSOLUTE";
    } else {
      return "unknown";
    }
  }

  uint32_t id;
  uint64_t logical_addr;
  uint64_t data_size;
  MemAllocation::Type type;
  uint32_t index_in_type;
  uint64_t mem_type;
  uint64_t hit_count;
  uint64_t tensor_size;  // for model input only, the real size of user input

  friend bool operator<(const MemAllocation &left, const MemAllocation &right) noexcept {
    // 加上大小是为了地址段匹配处理，不要擅自修改
    return (left.logical_addr + left.data_size) < (right.logical_addr + right.data_size);
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << "id:" << id << ", logical_addr:0x" << &(std::hex) << logical_addr << ", data_size:0x" << &(std::hex)
       <<  data_size << ", type:" << GetTypeStr(type) << ", index_in_type:" << index_in_type << ", mem_type:"
       << mem_type << ", hit_count:" << hit_count;
    return ss.str();
  }
};

struct MemAllocationAndOffset {
  size_t id;
  uint64_t offset;
  MemAllocation::Type type;
};

struct MemAllocationSlice {
  uint32_t id;
  uint64_t offset;
  uint64_t data_size;

  std::string ToString() const {
    std::stringstream ss;
    ss << "id:" << id << ", offset:0x" << &(std::hex) << offset << ", data_size:0x" << &(std::hex) <<  data_size;
    return ss.str();
  }
};

struct AddrDesc {
  uint64_t logic_addr;
  uint64_t memory_type; // fm, const, ....
  bool support_refresh; // true/false rts算子，以及dsa task info 发现是老的2包，也要返回false；其他大部分应该返回为true
  uint8_t reserved[3]; // 8字节对齐
};

enum class ArgsPlacement : int32_t {
  kArgsPlacementHbm = 0,      // hbm
  kArgsPlacementTs = 1,       // ts
  kArgsPlacementSqe = 2,      // sqe
  kArgsPlacementHostSvm = 3,  // sqe
  kEnd = 4                    // end
};
const char *GetArgsPlacementStr(ArgsPlacement placement);

struct TaskArgsDesc {
  int64_t args_len;
  ArgsPlacement placement; // hbm, ts, sqe, host_svm, end
};
// about persistent workspace memory desc
using PersistentWorkspaceDesc = TaskArgsDesc;
struct TaskRunParam {
  // 现在由TaskInfo 解析出来，再经过DavinciModel后，通过Init函数传回给TaskInfo是有些奇怪的，但是由于历史原因，暂时维持这一层关系
  std::vector<AddrDesc> parsed_input_addrs;
  std::vector<AddrDesc> parsed_output_addrs;
  std::vector<AddrDesc> parsed_workspace_addrs;
  std::vector<PersistentWorkspaceDesc> persistent_workspace_descs;
  std::vector<TaskArgsDesc> args_descs;
};

struct IowAddrs {
  std::vector<AddrDesc> input_logic_addrs;
  std::vector<AddrDesc> output_logic_addrs;
  std::vector<AddrDesc> workspace_logic_addrs;
};

struct TaskProfInfo {
  uint64_t begin_time;
  uint64_t end_time;
  uint32_t stream_id;
};

struct HostArg {
  void *addr;
  int64_t len;
  ArgsPlacement placement;
};

struct ArgAddrAndLen {
  uint64_t dev_addr;
  void *host_addr;
  int64_t len;
};

enum class ArgsFormatPolicy : int32_t {
  kAddrAll  = 0,
  kAddrLow32Bit = 1,
  kAddrHigh32Bit = 2,
  kAddrEnd = 3
};

struct TaskArgsRefreshInfo {
  uint32_t id;              // allocatin id
  uint64_t offset;          // offset of active mem base addr of the allocation id
  uint64_t io_index;        // io index, ffts level1 ctx defaults to 0
  uint64_t args_offset;     // offset of the task args base addr
  ArgsPlacement placement;  // hbm, ts, sqe, host_svm, end
  ArgsFormatPolicy args_format_policy; // Args use active addr whole value or low 32-bit value or high 32-bit value

  std::string ToString() const {
  std::stringstream ss;
  ss << "id:" << id << ", offset:0x" << &(std::hex) << offset << ", io_index:" << io_index <<
    ", args_offset:0x" << &(std::hex) << args_offset << ", placement:" << GetArgsPlacementStr(placement) <<
    ", args_format_policy:" << static_cast<int32_t>(args_format_policy) ;
  return ss.str();
  }
};

enum class PaRemapPolicy : int32_t {
  KSupport = 0,
  KConditionSupport = 1,
  KNoSupport = 2,
  KEnd = 3,
};

struct IowPaRemapInfo {
  TaskInfo *task_info;
  uint32_t allocation_id;
  uint64_t allocation_offset;
  uint64_t tensor_size;
  PaRemapPolicy policy;
  std::string op_name;
  friend bool operator<(const IowPaRemapInfo &left, const IowPaRemapInfo &right) noexcept {
    return (left.allocation_offset + left.tensor_size) < (right.allocation_offset + right.tensor_size);
  }

  std::string ToString() const {
  std::stringstream ss;
  ss << "op_name:" << op_name.c_str() << ", allocation id:" << allocation_id << ", offset:0x" << &(std::hex) <<
    allocation_offset << ", tensor_size:0x" << tensor_size << ", PaRemapPolicy:0x" <<
    static_cast<int32_t>(policy) ;
  return ss.str();
  }
};

// placement indexes to args
using PisToArgs = std::array<ArgAddrAndLen, static_cast<size_t>(ArgsPlacement::kEnd)>;
using PisToPersistentWorkspace = std::array<ArgAddrAndLen, static_cast<size_t>(ArgsPlacement::kEnd)>;

class TaskInfo {
 public:
  TaskInfo() {}

  virtual ~TaskInfo() { stream_ = nullptr; }

  virtual Status Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                      const PisToArgs &args = {}, const PisToPersistentWorkspace &persistent_workspace = {},
                      const IowAddrs &iow_addrs = {{}, {}, {}}) = 0;

  virtual Status Distribute() = 0;

  virtual void PostProcess(const domi::TaskDef &task_def) {
    (void)task_def;
  }

  virtual Status UpdateArgs() { return SUCCESS; }
  virtual Status UpdateHostArgs(const std::vector<uint64_t> &active_mem_base_addr, void *const host_args,
                                const size_t host_args_max_len) {
    (void)active_mem_base_addr;
    (void)host_args;
    (void)host_args_max_len;
    // todo print log not implenmented
    return SUCCESS;
  }

  virtual Status UpdateHostArgs(const std::vector<uint64_t> &active_mem_base_addr,
                                const std::vector<HostArg> &host_args) {
    if (host_args.size() != 1UL) {
      GELOGE(FAILED, "[Update][HostArgs]args does not support, host args size is %zu.", host_args.size());
      return FAILED;
    }
    auto &arg = host_args.at(0);
    return UpdateHostArgs(active_mem_base_addr, arg.addr, static_cast<size_t>(arg.len));
  }

  virtual Status UpdateDumpInfos(void *const host_args, const size_t host_args_max_len) {
    (void)host_args;
    (void)host_args_max_len;
    return SUCCESS;
  }

  virtual Status UpdateDumpInfos(const std::vector<HostArg> &host_args) {
    if (host_args.size() == 0UL) {
      GELOGE(FAILED, "[Update][DumpInfo] does not support, host args size is 0.");
      return FAILED;
    }
    auto &arg = host_args.at(0);
    return UpdateDumpInfos(arg.addr, static_cast<size_t>(arg.len));
  }

  virtual Status Release() { return SUCCESS; }

  virtual const ccOpContext *GetCtx() const { return nullptr; }

  virtual uint32_t GetTaskID() const { return 0xFFFFFFFFU; }

  virtual bool CallSaveDumpInfo() const { return false; }

  virtual uint32_t GetStreamId() const { return 0xFFFFFFFFU; }

  virtual uintptr_t GetDumpArgs() const { return 0U; }

  virtual uint32_t GetSktTaskID() const { return 0xFFFFFFFFU; }

  virtual const std::vector<FusionOpInfo> &GetAllFusionOpInfo() const {
    static const std::vector<FusionOpInfo> all_fusion_op_info;
    return all_fusion_op_info;
  }

  virtual uintptr_t GetArgs() const { return 0U; }

  virtual void GetTilingKeyAndData(uint32_t &tiling_key, std::string &tiling_data) const {
    tiling_key = 0U;
    tiling_data = "";
  }
  virtual size_t GetArgSize() const {
    return 0UL;
  }
  const TaskProfInfo &GetProfApi() const {
    return prof_api_;
  }

  TaskProfInfo &MutableProfApi() {
    return prof_api_;
  }

  virtual uint32_t GetMemType() {
    return RT_MEMORY_HBM;
  }

  virtual int64_t ParseOpIndex(const domi::TaskDef &task_def) const {
    (void)task_def;
    return -1;
  }
  // todo fix ParseTaskRunParam const
  virtual Status ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                   TaskRunParam &task_run_param) {
    (void)task_def;
    (void)davinci_model;
    (void)task_run_param;
    return SUCCESS;
  }

  virtual Status GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) {
    (void)infos;
    return SUCCESS;
  }

  virtual std::map<uint64_t, uint64_t> GetCustToRelevantOffset() const {
    return {};
  }

  virtual Status GetTaskIowPaRemapInfos(std::vector<IowPaRemapInfo> &infos) {
    (void)infos;
    return SUCCESS;
  }

  virtual uint64_t GetTaskStream() {
    return ge::PtrToValue(stream_);
  }

  virtual bool IsSupportReDistribute() {
    return is_support_redistribute_;
  }

 protected:
  TaskInfo(const TaskInfo &) = default;
  TaskInfo &operator=(const TaskInfo &) & = default;
  Status SetStream(const uint32_t stream_id, const std::vector<rtStream_t> &stream_list);
  static void SetTaskTag(const char_t *const op_name);

  void *stream_{nullptr};
  TaskProfInfo prof_api_{};
  bool is_support_redistribute_{false};
};

class TaskProfGuarder {
 public:
  explicit TaskProfGuarder(TaskInfo *task_info) {
    task_info_ = task_info;
    task_info_->MutableProfApi().begin_time = MsprofSysCycleTime();
  }
  ~TaskProfGuarder() noexcept {
    auto &prof_api = task_info_->MutableProfApi();
    prof_api.end_time = MsprofSysCycleTime();
    (void)rtsStreamGetId(reinterpret_cast<rtStream_t>(task_info_->GetTaskStream()), reinterpret_cast<int32_t*>(&prof_api.stream_id));
  }

 private:
  TaskInfo *task_info_{nullptr};
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_TASK_INFO_H_

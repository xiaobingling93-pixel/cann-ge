/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_MODEL_ARGS_MANAGER_H_
#define AIR_CXX_EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_MODEL_ARGS_MANAGER_H_

#include "graph/def_types.h"
#include "graph/load/model_manager/task_info/task_info_factory.h"
#include "graph/utils/math_util.h"
#include "proto/task.pb.h"

#include <map>
#include <memory>
#include <cmath>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "common/context/local_context.h"
#include "common/context/ome_context.h"
#include "common/dump/data_dumper.h"
#include "common/dump/opdebug_register.h"
#include "common/opskernel/ge_task_info.h"
#include "device_memory_ptr.h"
#include "ge/ge_api_types.h"
#include "framework/common/ge_types.h"
#include "framework/common/helper/model_helper.h"
#include "framework/common/helper/om_file_helper.h"
#include "framework/common/util.h"
#include "framework/runtime/subscriber/global_profiler.h"
#include "graph/load/model_manager/aicpu_resources.h"
#include "graph/load/model_manager/aipp_utils.h"
#include "graph/load/model_manager/cpu_queue_schedule.h"
#include "graph/load/model_manager/data_inputer.h"
#include "graph/load/model_manager/model_args_manager.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/load/model_manager/task_info/task_info_factory.h"
#include "graph/load/model_manager/tbe_kernel_handle.h"
#include "graph/load/model_manager/zero_copy_offset.h"
#include "graph/model.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/utils/tensor_utils.h"
#include "memory_app_type_classifier.h"
#include "mmpa/mmpa_api.h"
#include "model_args_layout_planner.h"
#include "proto/task.pb.h"
#include "task_args_refresh_type_classifier.h"
#include "aprof_pub.h"

namespace ge {
constexpr uint32_t kAddrRefreshOpParamOffset = 48U;
constexpr uint32_t kAlign32B = 32;
struct UpdateModelParamTilingData {
    uint32_t totalActiveBaseTblCnt;
    uint32_t blockCnt;
    uint32_t tileCnt;
    uint32_t tailCnt;
    uint32_t lastTailCnt;
    uint16_t tileNum;
    uint16_t lastTileNum;
    uint32_t lastTailCntOri;
    uint32_t reserve_2;
  };
class ModelArgsManager {
 public:
  struct ModelArgPartition {
    UpdateTriggerType partition_type;
    int64_t offset;
    int64_t len;
  };

  struct ModelArgs {
    ArgsPlacement placement;
    std::unique_ptr<uint8_t[]> model_args_host_addr;
    uint64_t model_args_device_addr;
    std::vector<ModelArgPartition> model_args_partitions;
  };

  enum UpdatePolicy : int32_t {
    // add new policy definitions here
    kNoNeedUpdate = 0,
    KUpdateHostInput = 1,     // 仅更新host input
    kUpdateModelIo = 2,       // 仅更新model io
    kUpdateFmAndModelIo = 3,  // 更新fm与model io
    kInitOneTime = 4,         // 全量更新
    kUpdatePolicyEnd = 5,
  };
  struct UpdateHostArgsArg {  // for UpdateHostArgs
    size_t task_index;
    TaskInfo *task_info;
    std::vector<HostArg> host_args;
  };
  struct H2DCopyArg {
    uint64_t len;
    uint64_t device_addr;
    void *host_addr;
  };
  struct SqeUpdateArg {
    uint32_t stream_id;
    uint32_t task_id;
    uint64_t dev_addr;
    uint64_t len;
  };
  struct ArgsUpdateData {
    std::vector<UpdateHostArgsArg> update_datas;
    SmallVector<H2DCopyArg, static_cast<uint32_t>(ArgsPlacement::kEnd)> h2d_copy_datas;
    // SqeUpdateArg 可以被理解为“后处理”，未来如果在完成hd拷贝后，有其他后处理，那么可以考虑将其做成基类
    std::vector<SqeUpdateArg> seq_update_datas;
  };

  struct FixedAddrPiece {
    TaskArgsRefreshTypeClassifier::TaskFixedAddr desc;
    uint64_t device_addr;
  };

  struct FixedAddrBulk {
    void *device_addr;
    std::vector<FixedAddrPiece> pieces;
  };

  struct ModelArgsRefreshInfo {
    uint32_t id;           // allocatin id
    uint64_t offset;       // offset of active mem base addr of the allocation id
    void *host_args_addr;
    uint64_t device_args_addr;

    std::string ToString() const {
    std::stringstream ss;
    ss << "id:" << id << ", offset:0x" << &std::hex << offset << ", host_args_addr:0x" <<
      &std::hex << PtrToValue(host_args_addr) << ", device_args_addr:0x" << &std::hex << device_args_addr;
    return ss.str();
  }
  };

  enum AllocForType : int32_t {
    kAllocForArgs = 0,
    kAllocForPersistentWorkspace = 1,
    kAllocForInvalid = 2,
  };

  enum ModelArgsManagerStage : size_t {
    kStagePrepareBegin = 0,
    kStageCalcUpdatePolicyBegin = 1,
    kStageUpdateHostArgsBegin = 2,
    kStageActiveMembaseMemcpyBegin = 3,
    kStageKernelLaunchBegin = 4,
    kStageHostArgsH2dBegin = 5,
    kStageUpdateDsaSqeBegin = 6,
    kStageStatisticsEnd = 7,
    kStageMax = 8,
  };

  struct ModelArgsDfxInfo {
    bool enable_flag = false;
    bool get_model_args_device_table_flag = false;
    std::string graph_name;
    uint32_t graph_id = 0U;
    uint32_t model_id = 0U;
    uint64_t update_addr_num = 0UL;
    bool fm_refreshable = false;
    bool known = false;
    uint64_t active_mem_base_addr_len;
    uint64_t stage_time_info[kStageMax]{};
  };

  struct KernelLaunchOpArgs {
    uint64_t model_offset_args_device_addr;
    uint64_t model_index_args_device_addr;
    uint64_t active_mem_base_device_addr;
    uint64_t model_args_table_addr;
    uint64_t workspace_addr;
    uint64_t tiling_addr;
    UpdateModelParamTilingData tiling_data;
  };

 public:
  ModelArgsManager() : ModelArgsManager(nullptr) {}
  ~ModelArgsManager() noexcept;
  explicit ModelArgsManager(DavinciModel *davinci_model) : davinci_model_(davinci_model){};

  void Init(DavinciModel *const davinci_model) {
    davinci_model_ = davinci_model;
  }
  Status Init(domi::ModelTaskDef &model_task_def, std::vector<TaskInfoPtr> *task_list_ptr);
  const std::vector<ModelArgs> &GetModelArgs() const;
  const FixedAddrBulk &GetFixedAddrBulk() const;
  Status UpdateForExecute(uint32_t &up, const aclrtStream stm = nullptr, const uint32_t model_execute_stage = 1);  // todo 删掉默认值
  void GenModelArgsAaddrAfterDistributed();
  Status ReportKernelLaunchOpProfilingData(const uint64_t begin_time) const;
  Status OnTaskDistributed(const size_t task_index, const TaskInfo *task_info);
  void SetAllocationHitCount(const uint64_t fm_hit_count, const uint64_t model_io_hit_count) {
    fm_hit_count_ = fm_hit_count;
    model_io_hit_count_ = model_io_hit_count;
  }

  std::vector<uint32_t>& GetId2Policy() {
    return id_to_plicy_;
  }

  // KernelLaunchOpArgs | activemembase | host input size
  Status AllocKernelLaunchArgsHostMem(uint64_t active_mem_base_addr_len, uint64_t append_size = 0U) {
    uint32_t active_mem_base_addr_len_align32b =
      (active_mem_base_addr_len + sizeof(uint32_t) - 1U) / sizeof(uint32_t) * sizeof(uint32_t);

    launched_args_unique_ptr_ = ge::MakeUnique<uint8_t[]>(sizeof(KernelLaunchOpArgs) +
      active_mem_base_addr_len_align32b * sizeof(uint64_t) + append_size);
    GE_ASSERT_NOTNULL(launched_args_unique_ptr_);

    kernel_launch_args_ptr_ =
      reinterpret_cast<KernelLaunchOpArgs*>(static_cast<void*>(launched_args_unique_ptr_.get()));
    *(reinterpret_cast<uint64_t*>(static_cast<void*>(launched_args_unique_ptr_.get()))
      + sizeof(KernelLaunchOpArgs)/sizeof(uint64_t) + active_mem_base_addr_len - 1U) = 0xFFFFFFFFFFFFFFFFUL;

    host_input_size_ = append_size;
    host_input_host_ptr_ = static_cast<uint8_t *>(launched_args_unique_ptr_.get()) + sizeof(KernelLaunchOpArgs) +
      active_mem_base_addr_len_align32b * sizeof(uint64_t);

    return SUCCESS;
  }

  uint64_t *GetActivateMemBaseAddrs() {
    if (launched_args_unique_ptr_ == nullptr) {
      return nullptr;
    }

    return  reinterpret_cast<uint64_t*>(static_cast<void*>(launched_args_unique_ptr_.get()))
      + sizeof(KernelLaunchOpArgs)/sizeof(uint64_t);
  }

  void SetFuncHandle(const rtFuncHandle &func_handle) {
    func_handle_ = func_handle;
  }

  Status CalculateUpdateModelParamTiling(uint32_t active_base_len, uint32_t index_len,
    uint32_t &block_dim, UpdateModelParamTilingData &tiling) const;
  Status GetHostInputMem(uint64_t &host_addr, uint64_t &device_addr, uint64_t &len);

  void InitDfx(bool enable_flag, std::string graph_name, uint32_t graph_id, uint32_t model_id, bool refreshable,
               bool known, bool get_model_args_device_table_flag) {
    dfx_info_.enable_flag = enable_flag;
    dfx_info_.graph_name = graph_name;
    dfx_info_.graph_id = graph_id;
    dfx_info_.model_id = model_id;
    dfx_info_.fm_refreshable = refreshable;
    dfx_info_.known = known;
    dfx_info_.get_model_args_device_table_flag = get_model_args_device_table_flag;
  }
  void InitDfxStage1Begin();
  void InitDfxStatsticsEnd();
  void CalculateDfxTime(std::stringstream &ss, const uint32_t model_execute_stage = 1);
  void PrintDfxStatistics(const uint32_t model_execute_stage = 1);
  Status PaRemapped(const uint64_t va, const uint64_t new_pa, const uint64_t len,
                    std::vector<std::pair<uint64_t, uint64_t>> &overlap_range);
 private:
  struct OneTaskUpdateData {
    UpdateHostArgsArg update_data;
    bool has_sqe_placement;
    SqeUpdateArg sqe_update_arg;
    std::vector<PisToArgs> *task_indexes_to_args;
  };

 private:
  Status InitTaskInfoV2(domi::ModelTaskDef &model_task_def);
  void ChangeMemcpyTaskTypeToAddrIfNeed(domi::TaskDef *const task_def) const;
  Status AllocModelArgs(const ModelArgsLayoutPlannedResult &layout,
                        std::vector<ModelArgs> &model_args, std::vector<uint64_t> &model_args_len,
                        ArgsPlacement &pls);
  Status ConstructUpdateData(const TaskNodeMap &task_node_map, const ModelArgsLayoutPlannedResult &layout,
                             const std::vector<TaskRunParam> &task_indexes_to_param,
                             std::vector<PisToArgs> &task_indexes_to_args);

  Status ConstructOneTaskUpdateData(const size_t task_index, const OneTaskArgsLayoutResult &task_arg_results,
                                    const std::vector<TaskRunParam> &task_indexes_to_param,
                                    const std::array<const ModelArgsManager::ModelArgs *,
                                                     static_cast<size_t>(ArgsPlacement::kEnd)> &pis_to_model_args,
                                    OneTaskUpdateData &task_update_data, const AddrUseFor addr_use_for) const;
  Status AddToTaskUpdateDataToPolicies(
      const size_t task_index,
      const SmallVector<ModelArgsManager::UpdatePolicy, ModelArgsManager::kUpdatePolicyEnd> &upis,
      const OneTaskUpdateData &one_task_update_data);
  Status AllocFixedAddrs(const TaskNodeMap &task_node_map,
                         const TaskArgsRefreshTypeClassifier::FixedAddrs &fixed_addrs);
  Status GenAddrRefreshIndexAndOffset(const uint64_t &offset_num);
  Status PrintKernelLaunchArgsDfxInfo(aclrtStream const stm);

  Status GenKernelLaunchArgs(uint64_t &offset_num);
  Status ParseModelTaskDef(domi::ModelTaskDef &model_task_def, std::vector<TaskRunParam> &task_indexes_to_run_param,
                           TaskNodeMap &task_node_map);
  Status ConstructTaskInitParams(
      const std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> &task_indexes_to_refresh_type,
      const std::map<std::pair<uint64_t, uint64_t>, MemoryAppType> &logical_addrs_to_mem_app_type,
      std::vector<TaskRunParam> &&task_indexes_to_param, std::vector<IowAddrs> &task_indexes_to_init_param) const;
  static Status ConstructH2DCopyParams(const ModelArgs &model_arg, const UpdatePolicy up, H2DCopyArg &cp_arg);

  using TriggerPolicies = SmallVector<ModelArgsManager::UpdatePolicy, ModelArgsManager::kUpdatePolicyEnd>;
  using TriggerTypesToPolicies = std::array<TriggerPolicies, static_cast<uint32_t>(UpdateTriggerType::kEnd)>;
  TriggerTypesToPolicies GenerateTriggerTypesToCorrespondingUpdatePolicies() const;

  UpdatePolicy CalcUpdatePolicy(const std::vector<uint64_t> &active_mem_base_addr);
  void DebugLogTaskUpdatePolicies(const TaskNodeMap &task_node_map,
                                  const TriggerPolicies &upis, size_t task_index) const;
  Status ValidateTaskRunParam(const std::vector<TaskArgsDesc> &args_descs) const;
  Status TaskArgsVa2PaAssociatedWithModelIO(aclrtStream const stm) const;
  void GetStageTimeInfo(ModelArgsManagerStage stage);
  void UpdateHostArgs(uint64_t* active_mem_base_addr);

  Status GenModelArgsRefreshInfosForTask(std::vector<TaskArgsRefreshInfo> &infos,
                                         PisToArgs &pls_to_args, const NodePtr &node);

  Status GenAllocationToIowPaRemapInfos(TaskInfoPtr task_info, const NodePtr &node, std::vector<IowPaRemapInfo> pa_remap_infos);

  void InitForUpdate();
 private:
  uint32_t update_version_{2};
  std::vector<TaskInfoPtr> *task_list_ptr_{nullptr};
  DavinciModel *davinci_model_;
  uint64_t fm_hit_count_{0U};
  uint64_t model_io_hit_count_{0U};
  ModelArgsDfxInfo dfx_info_{};
  UpdatePolicy up_;
  std::vector<ModelArgs> model_args_;
  std::vector<ModelArgs> model_persistent_workspace_;
  // quick-access data structure for model args table update
  std::array<std::unique_ptr<ArgsUpdateData>, kUpdatePolicyEnd> update_policies_to_model_data_;
  // 在task完成distribute之后，对update_policies_to_model_data_做增量修改
  // 例如sqe的更新任务，需要在distributed之后才能获取到task id等参数
  std::unordered_map<size_t, SmallVector<std::function<void(const TaskInfo *)>, kUpdatePolicyEnd>>
      task_indexes_to_update_data_appenders_on_distributed_;
  FixedAddrBulk fixed_addr_bulk_{};
  bool has_args_{false};
  std::vector<uint64_t> last_bases_;
  std::vector<uint32_t> id_to_plicy_;
  std::vector<uint64_t> id_to_len_;
  std::vector<uint64_t> model_args_len_;
  std::unique_ptr<rtHostInputInfo_t> args_input_info_ptr_;
  bool need_dev_va_2_pa_{false};
  rtArgsEx_t addr_update_op_args_{};
  bool active_mem_base_table_h2d_copy_flag_{false};
  std::unique_ptr<uint8_t[]> launched_args_unique_ptr_;
  KernelLaunchOpArgs* kernel_launch_args_ptr_{nullptr};
  uint64_t host_input_size_{0U};
  uint8_t *host_input_host_ptr_{nullptr};
  uint64_t host_input_device_ptr_{0U};
  uint64_t host_input_partition_len_{0U};
  int8_t logLevel_ {DLOG_DEBUG};
  uint32_t block_dim_{0};
  rtFuncHandle func_handle_{nullptr};
  std::vector<uint64_t> index_dingwei;
  std::vector<uint64_t> offset_dingwei;
  ArgsPlacement op_refresh_placement_{ArgsPlacement::kEnd};

  std::vector<std::vector<ModelArgsRefreshInfo>> allocation_ids_to_model_args_refresh_infos_addr_all;
  std::vector<std::vector<ModelArgsRefreshInfo>> allocation_ids_to_model_args_refresh_infos_addr_low_32bit;
  std::vector<std::vector<ModelArgsRefreshInfo>> allocation_ids_to_model_args_refresh_infos_addr_high_32bit;

  std::vector<std::multiset<IowPaRemapInfo>> allocation_ids_to_iow_pa_remap_infos_;
  uint64_t pa_remap_match_support_num_{0UL};
  uint64_t pa_remap_match_nosupport_num_{0UL};
  //  要刷新地址的device addr的host表
  void *model_args_device_offset_;
  void *model_args_device_index_;
  //  要刷新地址的device addr的device表
  void *activate_mem_base_device_addrs_dev_;
  void *workspace_addr_device_;
  bool is_pcie_bar_copy_{false};
};

rtMemType_t GetRtsMemoryType(const ArgsPlacement placement, const int64_t size);

}  // namespace ge
#endif  // AIR_CXX_EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_MODEL_ARGS_MANAGER_H_

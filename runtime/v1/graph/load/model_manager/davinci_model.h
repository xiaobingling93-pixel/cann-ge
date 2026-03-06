/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DAVINCI_MODEL_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DAVINCI_MODEL_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "framework/common/helper/model_helper.h"
#include "framework/common/helper/om_file_helper.h"
#include "common/opskernel/ge_task_info.h"
#include "common/dump/exception_dumper.h"
#include "common/dump/opdebug_register.h"
#include "graph/load/model_manager/aipp_utils.h"
#include "common/dump/data_dumper.h"
#include "graph/load/model_manager/data_inputer.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/load/model_manager/model_args_manager.h"
#include "graph/load/model_manager/tbe_kernel_handle.h"
#include "graph/load/model_manager/zero_copy_offset.h"
#include "graph/model.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/args_format_desc_utils.h"
#include "proto/task.pb.h"
#include "graph/load/model_manager/task_info/task_info_factory.h"
#include "common/context/local_context.h"
#include "common/context/ome_context.h"
#include "graph/load/model_manager/aicpu_resources.h"
#include "graph/load/model_manager/cpu_queue_schedule.h"
#include "ge/ge_api_types.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "aprof_pub.h"
#include "framework/runtime/subscriber/global_profiler.h"
#include "framework/runtime/subscriber/global_dumper.h"
#include "register/op_tiling_info.h"
#include "reusable_stream_allocator.h"
#include "memory_block_manager.h"
#include "common/memory/tensor_trans_utils.h"
#include "graph/load/model_manager/kernel/model_kernel_handles_manager.h"
#include "common/kernel_handles_manager/kernel_handle_utils.h"
#include "runtime/rts/rts_dqs.h"

namespace ge {
enum class ModelProcStage : uint32_t {
  MODEL_LOAD_START = 1,
  MODEL_LOAD_END,
  MODEL_PRE_PROC_START,
  MODEL_PRE_PROC_END,
  MODEL_INFER_START,
  MODEL_INFER_END,
  MODEL_AFTER_PROC_START,
  MODEL_AFTER_PROC_END,
  MODEL_PROC_INVALID,
};

struct ProfileInfo {
  FusionOpInfo fusion_info;
  std::vector<MsprofAdditionalInfo> prof_fusion_data_lst;
};

struct NodeBasicInfoWrapper {
  MsprofCompactInfo node_basic_info;
  std::string op_name;
  std::string op_type;
};

struct ApiInfoWrapper {
  MsprofApi api;
  std::string op_name;
};

struct DataCopyInfo {
  void *dst;
  void *src;
  uint64_t dst_length;  // 模型需要的数据大小
  uint32_t src_placement;
  uint64_t src_length;  // 实际传入的数据大小
};


enum DavinciModelExecuteStage : size_t {
    kStageBeforeH2D = 0,
    kStageBeforeRtExecute = 1,
    kStageAfterRtExecute = 2,
    kStageEnd = 3,
};

enum class ExecuteMode : uint32_t {
  INITIALIZATION,
  SYNCHRONIZATION,
  ASYNCHRONIZATION,
};

enum ModelProfStage : int32_t {
  kInitMdlMem,
  kInitIoNodes,
  kTransAllData,
  kInitAllNodes,
  kDoTaskSink,
  kCopyMdlData,
  kMdlExecute,
  kCopyOutputData,
  kMdlProfStageNameEnd
};

struct DavinciModelProf {
  bool enable_flag = false;
  bool get_model_args_device_table_flag = false;
  uint64_t init_begin;
  uint64_t init_end;
  uint64_t execute_begin;
  uint64_t execute_end;
  // 以下定义数据记录受环境变量控制
  std::map<ModelProfStage, uint64_t> stage_to_timestamp;
  std::map<uint32_t, uint64_t> task_type_to_distribute_time;
  std::map<uint32_t, uint64_t> task_type_to_distribute_num;
};

struct CopyHostInputInfo {
  int32_t input_index;
  void *host_addr;
  uint64_t device_addr;
  uint64_t tensor_size;
  CopyHostInputInfo() :input_index(0), host_addr(nullptr), device_addr(0u), tensor_size(0U) {}
};

// comments
class DavinciModel {
 public:
  /// @ingroup ge
  /// @brief DavinciModel constructor
  /// @author
  DavinciModel(const int32_t priority, const std::shared_ptr<ModelListener> &listener);

  /// @ingroup ge
  /// @brief DavinciModel desctructor, free Parse and Init resources
  ///        if has not finalized, call three Finalize methods in order
  /// @author
  ~DavinciModel() noexcept;

  /// @ingroup ge
  /// @brief  Finalize includes "UnbindTaskSinkStream -> DestroyStream -> DestroyResources",
  ///         which must be called in order.
  ///         If rtStreams are reused by dynamic model's several static graphs, they must unbind all models before
  ///         being destroyed
  void UnbindTaskSinkStream();

  void DestroyStream();

  void DestroyResources();

  /// @ingroup ge
  /// @brief apply model to model_def_
  void Assign(const GeModelPtr &ge_model);

  /// @ingroup ge
  /// @brief check model whther it has input or output queue.
  /// @return true : no input or output queue, false : has input or output queue.
  bool CheckModelNoInputAndOutput() const;

  /// @ingroup ge
  /// @brief DavinciModel initialization, including Stream, ccHandle, Event, etc
  /// @return execute result
  /// @author
  Status Init(const ModelParam &param = {}, void *outer_fm_mem = nullptr);

  /// @ingroup ge
  /// @brief ACL case, Load task list with queue.
  /// @param [in] input_queue_attrs: input queue ids from user, nums equal Data Op.
  /// @param [in] output_queue_attrs: input queue ids from user, nums equal NetOutput Op.
  /// @return: 0 for success / others for fail
  Status SetQueIds(const std::vector<QueueAttrs> &input_queue_attrs,
                   const std::vector<QueueAttrs> &output_queue_attrs);

  Status SetQueueType();

  void SetStatusQueue(const QueueAttrs &status_output_queue);

  void SetModelUuid(const uint32_t model_uuid);
  void SetNeedModelConfig(const bool flag);

  void SetNeedReportStatus(bool need_report_status);

  bool IsArgsUpdateByDeviceAicpu() const {
    return ((!input_queue_attrs_.empty()) || (!output_queue_attrs_.empty()));
  }

  bool GetForbiddenStreamFlag() const {
    return is_forbidden_stream_;
  }

  void SetInputFusionOffsets(const std::vector<int32_t> &fusion_offsets);

  /// @ingroup ge
  /// @brief get model id
  /// @return model ID
  uint32_t Id() const { return model_id_; }

  /// @ingroup ge
  /// @brief set model id
  /// @return model ID
  void SetId(const uint32_t model_id);

  /// @ingroup ge
  /// @brief Get SubModelId
  /// @return sub model ID
  uint32_t SubModelId() const { return sub_model_id_; }

  /// @ingroup ge
  /// @brief Get SubModelId
  /// @return sub model ID
  void SetSubModelId(const uint32_t sub_model_id) { sub_model_id_ = sub_model_id; }

  void SetRunContext(const OmeContext &context) { run_context_ = context; }

  void Run();

  void *GetOverflowAddr() const {
    return globalworkspace_overflow_addr_;
  }

  void SetOverflowAddr(void *const overflow_addr) {
    globalworkspace_overflow_addr_ = overflow_addr;
  }

  /// @ingroup ge
  /// @brief NnExecute
  /// @param [in] stream   execute stream
  /// @param [in] async_mode  is asynchronize mode.
  /// @param [in] input_data  model input data
  /// @param [out] output_data  model output data
  Status NnExecute(rtStream_t const stream, const bool async_mode,
                   const InputData &input_data, OutputData &output_data,
                   const std::vector<GeTensor> &input_tensor,
                   const std::vector<GeTensor> &output_tensor);

  /// @ingroup ge
  /// @brief NnExecute
  /// @param [in] stream   execute stream
  /// @param [in] async_mode  is asynchronize mode.
  /// @param [in] input_data  model input data
  /// @param [out] output_data  model output data
  Status NnExecute(rtStream_t const stream, const bool async_mode,
                   const std::vector<gert::Tensor> &input_tensor,
                   std::vector<gert::Tensor> &output_tensor);

  Status CheckRtStreamSynchronize(rtError_t rt_ret);

  Status InitAddrRefreshKernelBin();

  /// @ingroup ge
  /// @brief Copy Input/Output/Feature mem to model for direct use.
  /// @param [in] const InputData &input_data: user input data info.
  /// @param [in/out] OutputData &output_data: user output data info.
  /// @return SUCCESS handle successfully / others handle failed
  Status CopyModelData(const std::vector<gert::Tensor> &input_tensor,
                       const std::vector<gert::Tensor> &output_tensor);

  void GetGeTensorBlobs(InputData &input_data,
                        const std::vector<gert::Tensor> &input_tensor) const;

  Status UpdateAllNodeArgs(const InputData &input_data, const OutputData &output_data,
                           const std::vector<gert::Tensor> &input_tensor,
                           const std::vector<gert::Tensor> &output_tensor);

  Status ConstructZeroCopyIoActiveBaseAddrs(const std::vector<std::pair<uint32_t, uint32_t>> &refreshable_index_to_allocation_ids,
                                            const std::vector<DataBuffer> &blobs,
                                            const std::vector<gert::Tensor> &tensors,
                                            bool is_input, uint32_t &ret_up,
                                            std::vector<uint32_t>& id_to_plicy);

  Status CopyInputForNoZeroCopy(const std::vector<DataBuffer> &blobs,
                                const std::map<uint32_t, MemAllocationSlice> &copy_infos,
                                const std::vector<gert::Tensor> &tensors);

  Status CopyOutputData(const std::vector<gert::Tensor> &output_tensor);

  Status CopyOutputForNoZeroCopy(const std::vector<gert::Tensor> &output_tensor,
                                 const std::map<uint32_t, MemAllocationSlice> &copy_infos);

  std::vector<int64_t> GetTensorDims(const gert::Shape &shape) const;

  void UpdateOutputTensorShape(std::vector<gert::Tensor> &output_tensor);

  void GeShapeAsRtShape(const ge::GeShape &ge_shape, gert::Shape &gert_shape) const;

  std::vector<int64_t> GetGertTensorDims(gert::Shape &gert_shape) const;

  uint32_t GetNoFrozenInputAllocationBaseId() const {
    return no_frozen_input_allocation_base_id_;
  }

  void SetNoFrozenInputIndexes(std::unordered_set<uint32_t> frozen_input_indexes) {
    frozen_input_indexes_ = frozen_input_indexes;
  }

  void PrintNoFrozenInputIndexes();

  /// @ingroup ge
  /// @brief get Push Data to Queue
  /// @return 0 for success / others for fail
  Status Push(const std::shared_ptr<RunArgs> &args) {
    return data_inputer_.Push(args);
  }

  uint32_t GetDataInputerSize() {
    return data_inputer_.Size();
  }

  // get model priority
  int32_t Priority() const { return priority_; }

  // get total mem size
  size_t TotalMemSize() const { return runtime_param_.mem_size; }

  /// @ingroup ge
  /// @brief Get total useful size, in known subgraph, no need to allocate zero copy memory during initialization.
  /// @param [in] total_useful_size: total mem size - zero copy size.
  /// @return Status
  Status GetTotalMemSizeExcludeZeroCopy(int64_t &total_useful_size);

  size_t TotalVarMemSize() const { return runtime_param_.var_size; }

  // get base memory address
  uintptr_t FeatureMapBase() const { return mem_base_; }

  // get Notify list
  const std::vector<rtEvent_t> &GetNotifyList() const { return notify_list_; }

  // get Event list
  const std::vector<rtEvent_t> &GetEventList() const { return event_list_; }

  const std::vector<rtStream_t> &GetStreamList() const { return stream_list_; }

  void SetReusableStreamAllocator(ReusableStreamAllocator *reusable_stream_allocator) {
    reusable_stream_allocator_ = reusable_stream_allocator;
    is_outer_allocator_ = true;
  }
  ReusableStreamAllocator *GetReusableStreamAllocator() const { return reusable_stream_allocator_; }

  Status InitStreamInfoOfTask(const ComputeGraphPtr &compute_graph);
  Status GetTaskNumOfTaskdef(const domi::TaskDef &task_def, int32_t &task_num,
                             std::map<std::pair<int64_t, int32_t>, std::set<uint32_t>> &taskdef_task_num) const;
  int32_t GetTaskNumOfStream(const uint32_t stream_id) const;

  bool IsLogicStreamActiveByOthers(const uint32_t stream_id) const {
    return active_stream_indication_.find(stream_id) != active_stream_indication_.end();
  }

  const std::vector<rtLabel_t> &GetLabelList() const { return label_list_; }

  uint64_t GetAllStreamNum() const { return stream_list_.size() + all_hccl_stream_list_.size(); }

  Status GetLabelGotoAddr(const uint32_t label_index, const rtMemType_t mem_type, void *&arg_addr, uint32_t &arg_size);

  uint32_t GetRuntimeModelId() const { return runtime_model_id_; }

  Status DestroyThread();

  // get Op
  OpDescPtr GetOpByIndex(const uint32_t op_index) const {
    const auto it = op_list_.find(static_cast<int64_t>(op_index));
    if (it == op_list_.end()) {
      return nullptr;
    }
    return it->second;
  }

  void *GetMemEventIdAddr(const uint32_t mem_event_id);

  // get Operator
  std::shared_ptr<Operator> GetOperatorByIndex(const uint32_t op_index) const {
    const auto it = operator_list_.find(static_cast<int64_t>(op_index));
    if (it == operator_list_.end()) {
      return nullptr;
    }
    return it->second;
  }

  void SetModelQueueParam(const ModelQueueParam &model_queue_param);

  void SetGlobalStep(const uintptr_t step_addr, const uint64_t step_size);
  uintptr_t GetGlobalStep() const { return global_step_addr_; }
  uintptr_t GetLoopPerIter() const { return loop_per_iter_addr_; }
  uintptr_t GetLoopCond() const { return loop_cond_addr_; }
  // get updated task info list
  const std::vector<TaskInfoPtr> &GetTaskList() const { return task_list_; }

    // MDC特定形态下多单流模型加载保证串行，需要加锁保证不同流之间不串
  Status SetStreamLockOrUnlocK(rtStream_t stm, const bool is_lock) const;

  rtModel_t GetRtModelHandle() const { return rt_model_handle_; }

  uint64_t GetRtBaseAddr() const { return runtime_param_.logic_mem_base; }

  uint32_t GetFlowctrlIndex(const uint32_t op_index);

  void PushHcclStream(rtStream_t const hccl_stream);

  void SetHcclTaskStream(rtStream_t const hccl_stream);

  CustAICPUKernelPtr GetCustAICPUKernel(const OpDescPtr &op_desc) const;

  const std::string GetBinHandleKey(const OpDesc &op_desc, const std::string &prefix = "",
                                    const bool is_atomic_kernel = false) const;

  /// @ingroup ge
  /// @brief get model input and output desc info
  /// @param [out] input_shape  model input size
  /// @param [out] output_shape model output size
  /// @return execute result
  Status GetInputOutputDescInfo(std::vector<InputOutputDescInfo> &input_desc,
                                std::vector<InputOutputDescInfo> &output_desc) const;

  Status GetInputOutputDescInfo(std::vector<InputOutputDescInfo> &input_desc,
                                std::vector<InputOutputDescInfo> &output_desc,
                                std::vector<uint32_t> &input_formats, std::vector<uint32_t> &output_formats,
                                const bool by_dims) const;

  /// @ingroup ge
  /// @brief Get dynamic batch_info
  /// @param [out] batch_info
  /// @param [out] dynamic_type
  /// @return execute result
  Status GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type) const;

  /// @ingroup ge
  /// @brief Get combined dynamic dims info
  /// @param [out] batch_info
  /// @return None
  void GetCombinedDynamicDims(std::vector<std::vector<int64_t>> &batch_info) const;

  void GetUserDesignateShapeOrder(std::vector<std::string> &user_input_shape_order) const;

  void GetCurrentShape(std::vector<int64_t> &batch_info, int32_t &dynamic_type) const;

  Status GetNodeAttr(const std::string &op_name, const std::string &attr_name, std::string &attr_info) const;

  void GetOutputShapeInfo(std::vector<std::string> &out_shape_info) const;

  /// @ingroup ge
  /// @brief Get AIPP input info
  /// @param [in] index
  /// @param [out] aipp_info
  /// @return execute result
  Status GetAippInfo(const uint32_t index, AippConfigInfo &aipp_info) const;

  Status GetAippType(const uint32_t index, InputAippType &aipp_type, size_t &aipp_index) const;

  /// @ingroup ge
  /// @brief Get model_id.
  /// @return model_id
  uint32_t GetModelId() const { return model_id_; }

  /// @ingroup ge
  /// @brief get unique identification for op when load two or more models
  /// @param [in] op_desc : current op.
  /// @param [in] std::string identification: unique identification for current op.
  /// @return None
  void GetUniqueId(const OpDescPtr &op_desc, std::string &unique_identification) const;

  void AssembleListenerOutput(const std::shared_ptr<RunArgs> &args, const uint32_t data_id,
                              std::vector<gert::Tensor> &outputs);

  void OnComputeDoneWithResult(const uint32_t data_id, uint32_t result, std::vector<gert::Tensor> &outputs);
  void OnComputeDoneWithResultCallback(const std::shared_ptr<RunArgs> &args, const uint32_t data_id, uint32_t result,
                                       std::vector<gert::Tensor> &outputs);
  Status UpdateHbmFmMemBasesWithInnerMemory(uint8_t *mem_base, const size_t mem_size,
                                            const size_t data_size);
  void ReturnSequenceResult(const std::shared_ptr<RunArgs> &args, const uint32_t data_id, bool seq_end_flag);

  Status ModelRunStart();

  /// @ingroup ge
  /// @brief stop run model
  /// @return Status
  Status ModelRunStop();

  /// @ingroup ge
  /// @brief Get Session Id
  /// @return sessionID
  uint64_t GetSessionId() const { return session_id_; }

  const error_message::ErrorManagerContext &GetErrorContext() const { return error_context_; }

  /// @ingroup ge
  /// @brief SetDeviceId
  /// @return void
  void SetDeviceId(const uint32_t device_id) { device_id_ = device_id; }

  /// @ingroup ge
  /// @brief Get device Id
  /// @return  device id
  uint32_t GetDeviceId() const {
    return device_id_;
  }

  Status UpdateSessionId(const uint64_t session_id);

  const RuntimeParam &GetRuntimeParam() const { return runtime_param_; }

  fe::PlatFormInfos &MutablePlatformInfo() { return platform_infos_; }

  Status DisableZeroCopy(const void *const addr, const bool fusion_flag = false);

  Status DisableZeroCopyInReuseMemoryMode(const NodePtr &node, const size_t idx, const void *const addr);

  void DelDependentHcclStreams(const ComputeGraphPtr &compute_graph);

  bool GetOpDugReg() const { return is_op_debug_reg_; }

  Status ReportFusionOpInfo();

  Status ReportModelExtInfo(const uint32_t tid, const uint32_t model_id);

  /// @ingroup ge
  /// @brief Save outside address of Data or NetOutput used info for ZeroCopy.
  /// @param [in] const OpDescPtr &op_desc: current op desc
  /// @param [in] const std::vector<void *> &outside_addrs: address of task
  /// @param [in] const void *args_offset: arguments address save the address.
  /// @return None.
  void SetZeroCopyAddr(const OpDescPtr &op_desc, const std::vector<uint64_t> &outside_addrs,
                       const void *const args_info, const uintptr_t args_base, const size_t args_size,
                       const size_t offset, const std::vector<bool> &io_tiling_list);

  void SetLogicalOutsideAddrs(const std::map<uintptr_t, std::set<size_t>> &args_offset,
                              const std::vector<bool> &tiling_list,
                              const uintptr_t args_device_addr);

  std::set<size_t> GetZeroCopyArgsIndex(const std::vector<uint64_t> &arg_logical_addrs) const;

  Status Mapping2BundleZeroCopy(const OpDescPtr &op_desc, const std::map<uintptr_t, std::set<size_t>> &args_offset,
                                const std::vector<bool> &tiling_list, const size_t args_size,
                                const void *const args_host_copy, void *&args_device_addr, const bool &own_memory,
                                const bool is_all_kernel = false);

  void SetDynamicSize(const std::vector<uint64_t> &batch_num, const int32_t dynamic_type);

  void SetProfileTime(const ModelProcStage stage, const uint64_t end_time = 0U);

  void SaveSpecifyAttrValues(const OpDescPtr &op_desc);

  Status ReportProfilingData();
  Status ReportTaskTimeL1Info();
  void ClearProfilingDataCache();

  void SaveDfxInfo(const uint32_t op_idx, const domi::TaskDef &task_def, const TaskInfo &task_info);
  void SaveDfxInfo(const int64_t op_idx, const TaskInfo &task_info);

  void SaveProfilingTaskDescInfo(const OpDescPtr &op_desc, const TaskInfo &task_info, const domi::TaskDef &task_def);

  void SaveFftsPlusProfilingTask(const domi::TaskDef &task_def, const TaskInfo &task_info);

  void SaveFftsExceptionDumpInfo(const domi::FftsPlusCtxDef &ctx_def, const OpDescPtr &op_desc,
                                 const TaskInfo &task_info, const std::pair<uintptr_t, size_t> &args,
                                 const std::map<uint64_t, uint64_t> &cust_to_relevant_offset);
  void SaveFftsDfxInfo(const domi::FftsPlusCtxDef &ctx_def, const OpDescPtr &op_desc, const TaskInfo &task_info);

  void SaveExceptionDumpInfo(const OpDescPtr &op_desc, const TaskInfo &task_info);

  void SaveDumpTask(const OpDescInfoId &id, const shared_ptr<OpDesc> &op_desc, const uintptr_t args,
                    const FirstLevelAddressInfo &first_level_address_info = {false, {}},
                    const std::map<uint64_t, uint64_t> &cust_to_relevant_offset = {},
                    const ModelTaskType task_type = ModelTaskType::MODEL_TASK_KERNEL) {
    data_dumper_.SaveDumpTask(id, op_desc, args, first_level_address_info, cust_to_relevant_offset, task_type,
                              is_op_debug_reg_);
  }

  void SavePrintDumpTask(const OpDescInfoId &id, const shared_ptr<OpDesc> &op_desc, const uintptr_t args,
                         const FirstLevelAddressInfo &first_level_address_info = {false, {}},
                         const ModelTaskType task_type = ModelTaskType::MODEL_TASK_KERNEL) {
    data_dumper_.SavePrintDumpTask(id, op_desc, args, first_level_address_info, task_type);
  }

  void SaveLayerOpInfoOnWatcherMode(LayerOpOnWatcherModeInfo &op_info) {
    data_dumper_.SaveLayerOpInfoOnWatcherMode(op_info);
  }

  void SaveWorkInfo(const shared_ptr<OpDesc> &op_desc) {
    const auto workspace_addrs = ModelUtils::GetWorkspaceDataAddrsValue(runtime_param_, op_desc);
    data_dumper_.SetWorkSpaceAddr(op_desc, workspace_addrs);
  }

  void SavePrintWorkInfo(const shared_ptr<OpDesc> &op_desc) {
    const auto workspace_addrs = ModelUtils::GetWorkspaceDataAddrsValue(runtime_param_, op_desc);
    data_dumper_.SetWorkSpaceAddrForPrint(op_desc, workspace_addrs);
  }

  void DumperShrink() {
    data_dumper_.DumpShrink();
  }

  void ReLoadDumpInfo() {
    OpDebugRegister();
    data_dumper_.ReLoadDumpInfo();
  }

  void UnloadDumpInfo() {
    data_dumper_.UnloadDumpInfo();
    OpDebugUnRegister();
  }

  void ResetDumpFsmState() {
    dump_fsm_state_.clear();
    size_t num = GetDumpProperties().GetDumpOpRangeSize(dump_model_name_, om_name_);
    if (num > 0U) {
      GELOGI("Model[%s] om name[%s] opname range size[%zu].", dump_model_name_.c_str(), om_name_.c_str(), num);
      dump_fsm_state_.resize(num, DumpProcState::kInit);
    }

    dump_op_in_range_.clear();
  }

  Status SetDumpFsmState(const std::string &op_name, const ModelTaskType task_type) {
    if (dump_fsm_state_.empty()) {
      return SUCCESS;
    }

    bool is_update_dump_op_range = true;
    if (task_type == ModelTaskType::MODEL_TASK_FFTS_PLUS) {
      is_update_dump_op_range = false;
      GELOGW("op[%s] task type[%d] no support dump with opname range",
        op_name.c_str(), static_cast<int32_t>(task_type));
    }
    return GetDumpProperties().SetDumpFsmState(
      dump_model_name_, om_name_, op_name, dump_fsm_state_, dump_op_in_range_, is_update_dump_op_range);
  }

  bool IsInDumpOpRange(const std::string &op_name) {
    return (dump_op_in_range_.count(op_name) == 1U);
  }

  bool OpNeedDump(const std::string &op_name) {
    if (GetDumpProperties().IsDumpWatcherModelEnable()) {
      return false;
    }

    return GetDumpProperties().IsLayerNeedDump(dump_model_name_, om_name_, op_name) || IsInDumpOpRange(op_name);
  }

  bool OpNeedSetDumpFlagOnWatcherModel(const std::string &op_name) const {
    return (GetDumpProperties().IsLayerOpOnWatcherMode(op_name) || GetDumpProperties().IsWatcherNode(op_name));
  }

  bool OpNeedDumpOnWatcherModel(const std::string &op_name) const {
    return GetDumpProperties().IsWatcherNode(op_name);
  }

  bool OpNoNeedDumpOnWatcherModel(const std::string &op_name) const {
    return (GetDumpProperties().IsLayerOpOnWatcherMode(op_name) &&
      (!GetDumpProperties().IsWatcherNode(op_name)));
  }

  bool IsDumpWatcherModelEnable() const {
    return GetDumpProperties().IsDumpWatcherModelEnable();
  }

  bool OpNeedDump(const OpDescPtr &op_desc);

  bool OpNeedPrint(const OpDescPtr &op_desc) const;

  bool IsDumpLayerOpModelEnable() const {
    return GetDumpProperties().IsDumpLayerOpModelEnable();
  }
  KernelHandlesManagerPtr GetKernelHandlesManager(KernelHandleType kernel_type) const {
    return model_kernel_handles_manager_.GetKernelHandle(kernel_type);
  }
  bool ModelNeedDump() const;

  void SetEndGraphId(const uint32_t task_id, const uint32_t stream_id);
  DavinciModel &operator=(const DavinciModel &model) & = delete;

  DavinciModel(const DavinciModel &model) = delete;

  const std::map<int64_t, std::vector<rtStream_t>> &GetHcclFolowStream() const {
    return main_follow_stream_mapping_;
  }
  void SaveHcclFollowStream(const int64_t main_stream_id, rtStream_t stream);
  Status InitRuntimeParams();
  Status InitVariableMem();
  Status UpdateRuntimeParamBase();

  // 使用整段内存更新多段fm
  Status UpdateHbmFmMemBases(const uintptr_t mem_base, const size_t size,
                             size_t &used_size, const bool is_init = false);

  // 动态shape静态子图刷新场景或者静态图可刷新场景1
  Status UpdateHbmFmMemBases(const std::vector<uint8_t *> &hbm_fm_mem_bases,
                             const size_t size = SIZE_MAX, const bool is_init = false);

  Status UpdateExMemBase(const uint64_t memory_type, uint8_t *const mem_base) {
    const auto iter = runtime_param_.memory_infos.find(memory_type);
    if (iter == runtime_param_.memory_infos.end()) {
      GELOGE(FAILED, "memory_type[0x%" PRIx64 "] does not exist", memory_type);
      return FAILED;
    }
    iter->second.memory_base = mem_base;
    return SUCCESS;
  }

  void SetKnownNode(const bool known_node) {
    known_node_ = known_node;
    feature_base_refreshable_ = known_node;
  }

  void SetFeatureBaseRefreshable(const bool feature_base_refreshable) {
    feature_base_refreshable_ = feature_base_refreshable;
  }
  bool IsFeatureBaseRefreshable() const;
  bool IsKnownNode() const;
  bool NeedUpdateCoreCountWithOpDesc(const NodePtr &node, fe::PlatFormInfos &platform_infos, std::string &addr_key_out) const;
  bool UpdateCoreCountWithOpDesc(const NodePtr &node, fe::PlatFormInfos &platform_infos) const;
  Status UpdatePlatformInfos(const NodePtr &node, fe::PlatFormInfos &platform_infos) const;
  void* AllocPlatformInfosMem(size_t total_size, bool need_update_op_desc, bool is_custom);
  Status SerializeAndCopyToDevice(fe::PlatFormInfos &platform_infos, void *dev_addr, size_t copy_size,
                                  size_t total_size) const;
  // todo 临时方案
  // 此处因解决地址可刷新问题临时去掉const修饰
  // 待重构后去掉临时方案，因此该方法虽然没有const修饰入参，实现里也最好不要修改入参。
  Status UpdateKnownNodeArgs(const std::vector<uint64_t> &inputs, const std::vector<uint64_t> &outputs);

  Status GetOrigInputInfo(const uint32_t index, OriginInputInfo &orig_input_info) const;
  Status GetAllAippInputOutputDims(const uint32_t index, std::vector<InputOutputDims> &input_dims,
                                   std::vector<InputOutputDims> &output_dims) const;

  // om file name
  void SetOmName(const std::string &om_name) { om_name_ = om_name; }
  void SetDumpModelName(const std::string &dump_model_name) { dump_model_name_ = dump_model_name; }
  void SetFileConstantWeightDir(const std::string &file_constant_weight_dir) {
    file_constant_weight_dir_ = file_constant_weight_dir;
  }
  void SetFileConstantUserDeviceMem(const std::vector<FileConstantMem> &file_constant_mems) {
    for (const auto &file_constant_mem : file_constant_mems) {
      file_constant_user_device_mems_[file_constant_mem.file_name] = file_constant_mem;
    }
  }
  void SetFileConstantDeviceMem(const std::string &file_name, const void *device_mem, size_t mem_size) {
      file_constant_user_device_mems_[file_name] = {file_name, device_mem, mem_size};
  }
  const FileConstantMem *GetFileConstantUserDeviceMem(const std::string &file_name) const {
    const auto iter = file_constant_user_device_mems_.find(file_name);
    if (iter != file_constant_user_device_mems_.end()) {
      return &iter->second;
    }
    return nullptr;
  }
  bool IsUserDeviceMemForFileConstant(uintptr_t address) const {
    for (const auto &file_constant_mem : file_constant_user_device_mems_) {
      if (reinterpret_cast<uintptr_t>(file_constant_mem.second.device_mem) == address) {
        return true;
      }
    }
    return false;
  }
  string GetOmName() { return om_name_; }
  string GetDumpModelName() { return dump_model_name_; }
  uint32_t GetDumpModelId() const;
  void SetDumpProperties(const DumpProperties &dump_properties) { data_dumper_.SetDumpProperties(dump_properties); }
  const DumpProperties &GetDumpProperties() const { return data_dumper_.GetDumpProperties(); }

  bool GetOpDescInfo(const uint32_t stream_id, const uint32_t task_id, OpDescInfo &op_desc_info) {
    return exception_dumper_.GetOpDescInfo(OpDescInfoId(task_id, stream_id, static_cast<int32_t>(GetDeviceId())),
                                           op_desc_info);
  }
  void UpdateOpIOAddrs(const uint32_t task_id, const uint32_t stream_id, const std::vector<uint64_t> &io_addrs);

  bool GetRunningFlag() const { return running_flg_; }
  void SetRunningFlag(const bool flag) { running_flg_ = flag; }

  bool GetAiCpuCustFlag() const { return aicpu_flg_; }
  void SetAiCpuCustFlag(const bool flag) { aicpu_flg_ = flag; }

  // for blocking aicpu op
  Status GetEventByStream(rtStream_t const stream, rtEvent_t &rt_event);
  Status GetEventIdForBlockingAicpuOp(const OpDescPtr &op_desc, rtStream_t const stream, uint32_t &event_id);

  uint32_t GetResultCode();
  Status ResetResult();

  Status GetAddrAndPrefCnt(const OpDescPtr &op_desc, const std::string &kernel_name, const std::string &prefix,
                           std::vector<std::pair<void *, uint32_t>> &addr_pref_cnt) const;

  void SetRootGraphId(const uint32_t root_graph_id) { runtime_param_.root_graph_id = root_graph_id; }

  Status ReportProfilingData(const uint32_t graph_id);

  bool HasZeroCopyAddr(const OpDescPtr &op_desc) const;

  bool GetAsyncMode() const {
    return is_async_mode_;
  }

  void SetAsyncMode(bool is_async_mode) {
    is_async_mode_ = is_async_mode;
  }

  void SetClearDfxCacheFlagAfterInit(const bool clear_cache);

  bool NeedClearDfxCacheFlagAfterInit() const;

  rtStream_t GetModelExecuteStream() const {
    return rt_model_stream_;
  }

  Status MallocExMem();
  void *MallocDynamicMemory(const size_t size, const rtMemType_t mem_type = RT_MEMORY_HBM);
  void FreeDynamicWorkspaceMemory();

  Status InitSpaceRegistry(const GeRootModelPtr &root_model);
  std::shared_ptr<gert::OpImplSpaceRegistryV2Array> GetSpaceRegistries() const {
    return space_registries_;
  }
  std::shared_ptr<gert::OpImplSpaceRegistryV2> GetSpaceRegistry(gert::OppImplVersionTag version_tag) const {
    GE_ASSERT_TRUE(version_tag < gert::OppImplVersionTag::kVersionEnd);
    return space_registries_->at(static_cast<size_t>(version_tag));
  }
  void SetSpaceRegistries(std::shared_ptr<gert::OpImplSpaceRegistryV2Array> space_registries) {
    space_registries_ = space_registries;
  }

  ExceptionDumper *MutableExceptionDumper() {
    return &exception_dumper_;
  }

  void UpdateOutputTensorShape(std::vector<GeTensor> &output_tensor);
  /// @ingroup ge
  /// @brief Init model stream for NN model.
  /// @return Status
  Status InitModelStream(rtStream_t const stream);

  std::vector<MemAllocation> &GetLogicalMemAllocation() {
    return logical_mem_allocations_;
  }

  const std::vector<MemInfo> &GetFmMemoryInfos() const {
    return runtime_param_.fm_memory_infos;
  }

  Status CopyOutputData(const OutputData &output_data, const std::vector<GeTensor> &output_tensor);

  bool HasHcclTask() const {
    return has_hccl_task_;
  }
  ComputeGraphPtr GetCompiledComputeGraph() const {
    return ge_model_->GetGraph();
  }

  uint64_t GetReportedProfCount() const {
    return prof_count_.load();
  }

  void SetReportedProfCount(const uint64_t count) {
    prof_count_.store(count);
  }

  size_t GetFmMemAllocationsSize() const {
    return logical_fm_mem_allocations_size_;
  }

  size_t GetFmMemAllocationsStartId() const {
    return fm_mem_allocations_start_id_;
  }

  uint32_t GetStreamFlagById(uint32_t stream_id) const;

  bool IsStaticAddrFixed() const {
    return is_static_model_addr_fixed_;
  }
  Status LoadPlatformInfos(const fe::PlatFormInfos *const plat_form_info_ptr, size_t &copy_size, void *&dev_addr,
                           bool is_custom = false, const NodePtr &node = nullptr);
  Status LoadCustPlatformInfos(void *&cust_platform_infos_addr, const NodePtr &node);
  Status LaunchFromPlatformSo(const std::string &platform_so_path);
  Status LaunchCustPlatformInfos();
  Status LaunchFromOpMasterSo();
  Status LaunchPlatformInfos(void *&platform_infos_addr, const NodePtr &node);

  Status PaRemapped(const uint64_t va, const uint64_t new_pa, const uint64_t len,
                    std::vector<std::pair<uint64_t, uint64_t>> &overlap_range) {
    return args_manager_.PaRemapped(va, new_pa, len, overlap_range);
  }

  Status RestoreDeviceVarMem(const std::vector<NodePtr> &variable_nodes, const ModelParam &param);

  bool GetPhysicalMemoryRefreshable() const;

  Status LaunchEventForHcclGroupOrderedStream(rtStream_t const stream);

  Status RecoverModel();

  void SetTilingSinkTaskArgDescs(uint32_t op_index, std::vector<ArgDesc> &arg_descs) {
    tiling_sink_task_arg_descs_list_[op_index] = arg_descs;
    for (size_t i = 0 ; i < arg_descs.size(); i++) {
      GELOGI("set op index: %u, arg desc index: %zu, arg desc type: %d",
        op_index, i, static_cast<int32_t>(arg_descs[i].addr_type));
    }
  }

  Status GetAndEraseTilingSinkTaskArgDescs(uint32_t op_index, std::vector<ArgDesc> &arg_descs) {
    const auto it = tiling_sink_task_arg_descs_list_.find(op_index);
    if (it == tiling_sink_task_arg_descs_list_.end()) {
      return FAILED;
    }

    arg_descs = std::move(it->second);
    for (size_t i = 0 ; i < arg_descs.size(); i++) {
      GELOGI("get op index: %u, arg desc index: %zu, arg desc type: %d",
        op_index, i, static_cast<int32_t>(arg_descs[i].addr_type));
    }

    (void) tiling_sink_task_arg_descs_list_.erase(it); // 每个op_index只调用一次, 使用后释放
    return SUCCESS;
  }

 private:
  friend class DavinciModelFaker;  // for DT
  // memory address of weights
  uintptr_t weights_mem_base_{0U};
  uintptr_t var_mem_base_{0U};
  // memory address of model
  uintptr_t fixed_mem_base_ {0U};  // Initial of mem_base_, keep forever.
  // refresh fm地址，分段场景为首个refresh fm段的地址
  uintptr_t mem_base_{0U};
  // fm段的总长度(fix fm和refresh fm), 用来判断分配fm内存时是否包含io段内存
  size_t mem_base_size_{0U};
  // 所有fix fm段的大小, 用户设置的fix内存需要大于等于该长度
  size_t fixed_mem_size_{0U};
  // 纯静态图，分段场景，零拷贝内存为外部分配时，以fm地址的最大值作为io段的起始地址，防止和fm段产生内存交叉
  uintptr_t io_mem_base_{0U};
  // 纯静态图GE实际分配的内存大小
  std::vector<std::pair<uint8_t *, size_t>> active_memorys_;

  bool is_hw_q_{false};
  bool is_inner_mem_base_{false};
  bool is_inner_weight_base_{false};
  // input data manager
  DataInputer data_inputer_;
  uint64_t load_begin_time_{0U};
  uint64_t load_end_time_{0U};
  uint64_t execute_start_time_{0UL};
  uint64_t execute_end_time_{0UL};
  MsprofApi input_data_pre_{};
  MsprofApi output_data_pre_{};
  MsprofEvent model_load_event_{};
  int32_t dataInputTid{0};
  bool enable_input_batch_cpy_{false};

  Status InitRuntimeResource();
  Status InitSupplyResource();

  /// @ingroup ge
  /// @brief Copy Check input size and model op size.
  /// @param [in] const int64_t &input_size: input size.
  /// @param [in] const int64_t &op_size: model op size.
  /// @param [in] is_dynamic: dynamic batch input flag.
  /// @return true if success
  bool CheckUserAndModelSize(const int64_t size, const int64_t op_size, const char *model_io_type) const;

  /// @ingroup ge
  /// @brief Set copy only for No task feed NetOutput address.
  /// @return None.
  Status SetCopyOnlyOutput();

  /// @ingroup ge
  /// @brief Record profiling time.
  /// @return None.
  void RecordProfileTime();

  /// @ingroup ge
  /// @brief Copy Input/Output/Feature mem to model for direct use.
  /// @param [in] const InputData &input_data: user input data info.
  /// @param [in/out] OutputData &output_data: user output data info.
  /// @return SUCCESS handle successfully / others handle failed
  Status CopyModelData(InputData &input_data, OutputData &output_data,
      const std::vector<GeTensor> &input_tensor, const std::vector<GeTensor> &output_tensor);

  Status ConstructZeroCopyIoActiveBaseAddrs(std::vector<std::pair<uint32_t, uint32_t>> &refreshable_index_to_allocation_ids,
                                            const std::vector<DataBuffer> &blobs,
                                            const std::vector<GeTensor> &tensors,
                                            bool is_input, uint32_t &ret_up,
                                            std::vector<uint32_t>& id_to_plicy);

  Status UpdateAllNodeArgs(const InputData &input_data, const OutputData &output_data,
                           const std::vector<GeTensor> &input_tensor, const std::vector<GeTensor> &output_tensor);
  Status CheckIoReuseAddrs(const std::vector<DataBuffer> &input_blobs,
                           const std::vector<DataBuffer> &output_blobs,
                           const std::vector<gert::Tensor> &input_tensors,
                           const std::vector<gert::Tensor> &output_tensors) const;
  Status CheckIoReuseAddrs(const std::vector<DataBuffer> &input_blobs,
                           const std::vector<DataBuffer> &output_blobs,
                           const std::vector<GeTensor> &input_tensors,
                           const std::vector<GeTensor> &output_tensors) const;
  Status CopyInputForNoZeroCopy(const std::vector<DataBuffer> &blobs,
                                const std::map<uint32_t, MemAllocationSlice> &copy_infos,
                                const std::vector<GeTensor> &tensors);
  Status CopyOutputForNoZeroCopy(const std::vector<GeTensor> &output_tensor,
                                 const std::vector<DataBuffer> &blobs,
                                 const std::map<uint32_t, MemAllocationSlice> &copy_infos);
  Status ConstructActiveMemBaseAddrsForKnownNode(uint32_t &ret_up,
    const std::vector<uint64_t> &inputs, const std::vector<uint64_t> &outputs);

  void ConstructActiveMemBaseAddrs();

  void ConstructFmActiveMemBaseAddrs(uint32_t &ret_up, std::vector<uint32_t> &active_mem_base_id_to_plicy);

  void FreeInnerFeatureMapMem();

  void GetGeTensorBlobs(InputData &input_data, const std::vector<GeTensor> &input_tensor) const;

  Status HandleInputData(InputData &input_data);

  Status CopyInputDataWithMergeH2D(const InputData &input_data);

  Status CopyInputData(const InputData &input_data);

  void ResetMemcpyBatchParams();

  Status CopyOutputDataLegacy(const OutputData &output_data);

  Status UpdateStepInfoWithStream();

  Status InitWeightMem(const uintptr_t mem_ptr, const uintptr_t weight_ptr, const size_t weight_size);
  Status InitFixedFeatureMap(const uintptr_t fixed_mem_ptr, const size_t fixed_mem_size);
  Status InitFeatureMapAndP2PMem(const uintptr_t mem_ptr, const size_t mem_size, void *outer_fm_mem = nullptr);

  void CreateInputDimsInfo(const OpDescPtr &op_desc, const Format format, ShapeDescription &shape_info,
                           ShapeDescription &dims_info) const;

  void SetInputDimsInfo(const std::vector<int64_t> &input_dims, const Format format,
                        ShapeDescription &shape_info) const;

  Status GetInputDescInfo(std::vector<InputOutputDescInfo> &input_desc,
                          std::vector<uint32_t> &input_format, const bool by_dims) const;
  Status GetOutputDescInfo(std::vector<InputOutputDescInfo> &output_desc, std::vector<uint32_t> &output_format) const;

  // todo 临时方案
  // 此处因解决地址可刷新问题临时去掉const修饰
  // 待重构后去掉临时方案，因此该方法虽然没有const修饰入参，实现里也最好不要修改入参。
  Status InitTaskInfo(domi::ModelTaskDef &model_task_def);

  void UnbindHcomStream();

  Status DistributeTask(const domi::ModelTaskDef &model_task_def);

  uint8_t *MallocFeatureMapMem(const size_t data_size);

  uint8_t *MallocWeightsMem(const size_t weights_size) const;

  uint8_t *MallocFileConstantMem(const size_t weights_size) const;

  void FreeFeatureMapMem();

  void FreeWeightsMem();

  void FreeFileConstantMem();

  void FreeExMem();

  void ReleaseTask();

  void ClearTaskAddrs();

  bool IsAicpuKernelConnectSpecifiedLayer(const ComputeGraphPtr &compute_graph) const;

  /// @ingroup ge
  /// @brief Reduce memory usage after task sink.
  /// @return: void
  void Shrink();

  Status InitIoNodes(const ComputeGraphPtr &compute_graph, std::vector<NodePtr> &variable_nodes);
  void CollectHcomRelatedStreams(const OpDescPtr &op_desc);
  /// @ingroup ge
  /// @brief Travel all nodes and preprocess fileconstant nodes only in offline training process.
  /// @param [in] compute_graph: ComputeGraph to load.
  /// @return Status
  Status PreProcessFileConstants(const ComputeGraphPtr &compute_graph, const ModelParam &param);

 private:
  Status SetExternalPath(const ComputeGraphPtr &compute_graph);
  Status InitVarResourceIfNeeded(const ModelParam &param, bool &var_resource_inited);
  Status ProcessFileConstantNode(const NodePtr &node, const ModelParam &param, bool &var_resource_inited,
                                 bool &is_weight_combined, std::string &combined_weight_file,
                                 std::map<NodePtr, std::pair<size_t, int64_t>> &node_to_offset_and_size);
  Status AllocateCombinedWeightMemory(const std::string &combined_weight_file,
                                      const void *&real_dev_addr,
                                      int64_t &file_size,
                                      bool &is_user_mem);
  Status MapNodeAddressesToCombinedWeight(const std::map<NodePtr, std::pair<size_t, int64_t>> &node_to_offset_and_size,
                                         const void *base_addr,
                                         int64_t file_size,
                                         const std::string &file_path);

  Status HandleCombinedWeights(const std::map<NodePtr, std::pair<size_t, int64_t>> &node_to_offset_and_size,
                               const std::string &combined_weight_file);
  Status HandleIndividualWeights(const std::map<NodePtr, std::pair<size_t, int64_t>> &node_to_offset_and_size);

  /// @ingroup ge
  /// @brief First, retrieve the external weight file name from the FileConstant node.
  /// Then, use this file name to check if the user has set a corresponding device address.
  /// @param [in] op_desc: current op.
  /// @param [in] weights_size: file constant value size.
  /// @param [out] user_mem: user set device address, nullptr if not found.
  /// @return Status
  Status GetUserDeviceMemForFileConstant(const OpDescPtr &op_desc, size_t weights_size, void *&user_mem) const;

  /// @ingroup ge
  /// @brief Travel all nodes and do some init.
  /// @param [in] compute_graph: ComputeGraph to load.
  /// @return Status
  Status InitNodes(const ComputeGraphPtr &compute_graph);
  Status InitNoTaskAndDumpNeededNode(const OpDescPtr &op_desc);

  void PrintHcclOps(const std::vector<std::pair<std::string, std::string>> &hccl_ops) const;

  /// @ingroup ge
  /// @brief Data Op Initialize.
  /// @param [in] ComputeGraphPtr: root graph of the model.
  /// @param [in] NodePtr: Data Op.
  /// @param [in/out] data_op_index: index of courrent count.
  /// @param [in/out] index_to_data: Data ordered by index.
  /// @return Status
  Status InitDataOp(const ComputeGraphPtr &graph, const NodePtr &node, uint32_t &data_op_index,
                    std::map<uint32_t, OpDescPtr> &index_to_data, std::set<uint64_t> &input_outside_addrs);

  /// @ingroup ge
  /// @brief Sort Data op list by index.
  /// @param [in] index_to_data: map of Data Op.
  /// @param [in] output_op_list: list of NetOutput op.
  /// @return Status
  Status GenInputOutputInfo(const std::map<uint32_t, OpDescPtr> &index_to_data,
                            const std::vector<OpDescPtr> &output_op_list);

  /// @ingroup ge
  /// @brief NetOutput Op Initialize.
  /// @param [in] ComputeGraphPtr: root graph of the model.
  /// @param [in] NodePtr: NetOutput Op.
  /// @param [in/out] std::vector<OpDescPtr>: All NetOutput node in model.
  /// @return Status
  Status InitNetOutput(const ComputeGraphPtr &graph, const NodePtr &node, std::vector<OpDescPtr> &output_op_list,
                       std::set<uint64_t> &output_outside_addrs);

  /// @ingroup ge
  /// @brief Constant Op Init.
  /// @return Status
  Status InitConstant(const OpDescPtr &op_desc);

  Status InitVariable(const OpDescPtr &op_desc, std::map<std::string, OpDescPtr> &variable_by_name);

  /// @ingroup ge
  /// @brief Get Op rtStream.
  /// @param [in] op_desc: Op descriptor.
  /// @param [in] stream_id: Logical stream id.
  /// @param [out] stream: rt stream.
  /// @return Status
  Status GetOpStream(const OpDescPtr &op_desc, const size_t stream_id, rtStream_t &stream);

  /// @ingroup ge
  /// @brief LabelSet Op Initialize.
  /// @param [in] op_desc: LabelSet Op descriptor.
  /// @return Status
  Status InitLabelSet(const OpDescPtr &op_desc);

  Status InitStreamSwitch(const OpDescPtr &op_desc);

  Status InitStreamActive(const OpDescPtr &op_desc);

  /// @ingroup ge
  /// @brief Case Op Init.
  /// @return Status
  Status InitCase(const OpDescPtr &op_desc);

  Status SetDynamicBatchInfo(const OpDescPtr &op_desc, const uint32_t batch_num);

  /// @ingroup ge
  /// @brief TVM Op Init.
  /// @return Status
  Status InitTbeHandle(const OpDescPtr &op_desc);

  /// @ingroup ge
  /// @brief Make active stream list and bind to model.
  /// @return: 0 for success / others for fail
  Status BindModelStream();

  /// @ingroup ge
  /// @brief ACL, Load task list with queue entrance.
  /// @return: 0 for success / others for fail
  Status LoadWithQueue();

  Status LaunchDqsTask(rtDqsTaskType task_type, void* cfg = nullptr);

  Status LoadWithHardwareQueue();

  Status GetZcpyReplaceAddrsMap(const std::map<uint32_t, ZeroCopyOffset> &outside_addrs,
                                std::map<size_t, std::vector<uint64_t>> &replace_addrs_map);

  Status LaunchInputZeroCpyCfg();

  Status LaunchOutputZeroCpyCfg();

  /// @ingroup ge
  /// @brief ACL, Bind Data Op addr to input queue.
  /// @return: 0 for success / others for fail
  Status BindInputQueue();

  Status BindControlInputQueue();

  Status CpuTaskModelZeroCopy(std::vector<uintptr_t> &mbuf_list,
                              const std::map<uint32_t, ZeroCopyOffset> &outside_addrs,
                              const std::vector<bool> &is_no_tiling_list,
                              ZeroCpyArgs &cpy_args);

  /// @ingroup ge
  /// @brief ACL, Bind NetOutput Op addr to output queue.
  /// @return: 0 for success / others for fail
  Status BindOutputQueue();
  Status BindControlOutputQueue();
  Status CpuModelPrepareOutput(const size_t output_idx, const uintptr_t addr, const uint32_t data_size);

  /// @ingroup ge
  /// @brief definiteness queue schedule, bind input queue to task.
  /// @param [in] queue_id: input queue id from user.
  /// @param [in] addr: Data Op output tensor address.
  /// @param [in] size: Data Op output tensor size.
  /// @return: 0 for success / others for fail
  Status CpuModelDequeue(const uint32_t queue_id);
  Status CpuModelDequeue();
  Status CpuModelBatchDequeue();
  Status CpuModelGatherDequeue();

  /// @ingroup ge
  /// @brief definiteness queue schedule, active original model stream.
  /// @return: 0 for success / others for fail
  Status CpuActiveStream();

  /// @ingroup ge
  /// @brief definiteness queue schedule, mark dump step info.
  /// @return: 0 for success / others for fail
  Status CpuMarkStep(const uint32_t group_total_count,
                     const uint32_t group_index,
                     const uint32_t group_policy,
                     const std::string &dump_step);

  /// @ingroup ge
  /// @brief definiteness queue schedule, wait for end graph.
  /// @return: 0 for success / others for fail
  Status CpuWaitEndGraph();

  /// @ingroup ge
  /// @brief definiteness queue schedule, post process after end graph.
  /// @return: 0 for success / others for fail
  Status CpuPostProcess();
  Status CpuModelPostProcess(const size_t output_idx, const uintptr_t addr, const uint32_t data_size,
                             const ProcessStage stage);

  Status BindEnqueue();
  Status CpuModelEnqueue(const uint32_t queue_id, const uintptr_t out_mbuf);

  /// @ingroup ge
  /// @brief definiteness queue schedule, model report status.
  /// @return: 0 for success / others for fail
  Status CpuModelReportStatus();

  /// @ingroup ge
  /// @brief definiteness queue schedule, repeat run model.
  /// @return: 0 for success / others for fail
  Status CpuModelRepeat();

  /// @ingroup ge
  /// @brief construct task for input not sopput zero copy.
  /// @return: 0 for success / others for fail
  Status CpuInputCopyProcess();

  Status CpuStaticInputShapeValidate();

  Status AddHeadStream();

  /// @ingroup ge
  /// @brief set ts device.
  /// @return: 0 for success / others for fail
  Status SetTSDevice();

  Status OpDebugRegister();

  void OpDebugUnRegister();

  Status SetModelConfig() const;
  void SetStaticModelShapeConfig();

  Status UpdateStaticModelArgsByFm();

  Status CreateHcclGroupOrderedEvent();

  Status DoTaskSink();

  void CreateOutput(const size_t index, const OpDescPtr &op_desc, InputOutputDescInfo &output,
                    uint32_t &format_result) const;

  Status TransAllVarData(const ComputeGraphPtr &graph, const std::vector<NodePtr> &variable_nodes) const;

  Status SetDataDumperArgs(const ComputeGraphPtr &graph, const std::map<std::string, OpDescPtr> &variable_by_name);

  Status InitL1DataDumperArgs();

  Status InitFusionProfiling(const FusionOpInfo &fusion_op_info);

  void SinkTimeProfile(const uint32_t data_index, const uint64_t request_id);

  Status DisableZeroCopyNode(const OpDescPtr &op_desc);

  Status InitOutputTensorInfo(const OpDescPtr &op_desc);
  Status GenOutputTensorInfo(OutputData &output_data, std::vector<gert::Tensor> &outputs);
  Status BuildOutputShapeInfo(const size_t output_idx, std::vector<int64_t> &output_shape, int64_t &output_size);

  Status InitInputDescInfo(const OpDescPtr &op_desc);
  Status InitOutputDescInfo(const OpDescPtr &op_desc, const std::vector<std::string> &out_node_name);

  Status InitAippInfo(const uint32_t index, const OpDescPtr &op_desc);
  Status InitAippType(const uint32_t index, const OpDescPtr &op_desc, const std::map<uint32_t, OpDescPtr> &data_list);

  /// @ingroup domi_ome
  /// @brief Get cur_dynamic_dims for all input.
  /// @param [in] std::vector<vector<int64_t>> &tensor_input_dims: dims info of all user_inputs.
  /// @param [out] std::vector<int32_t> &cur_dynamic_dims: real dims gather, where the index of -1.
  /// @return 0: SUCCESS / others: INTERNAL_ERROR
  Status GetCurDynamicDims(const std::vector<std::vector<int64_t>> &tensor_input_dims,
                           std::vector<int32_t> &cur_dynamic_dims) const;

  void ParseInputsDimsForData(const std::vector<std::vector<int64_t>> &tensor_input_dims,
                              std::vector<std::vector<int64_t>> &real_input_dims) const;
  Status ParseInputsDimsForGetNextNoSinkAndData(const std::vector<NodePtr> &dynamic_nodes,
                                                const std::vector<std::vector<int64_t>> &tensor_input_dims,
                                                std::vector<std::vector<int64_t>> &real_input_dims) const;
  Status ParseInputsDims(const std::vector<std::vector<int64_t>> &tensor_input_dims,
                         std::vector<std::vector<int64_t>> &real_input_dims) const;

  void ParseDynamicOutShape(const std::vector<std::string> &str_info,
                            std::vector<std::vector<int64_t>> &vec_info) const;
  bool IsGetNextSinkDynamic(const OpDescPtr &op_desc) const;

  Status InitRealSizeAndShapeInfo(const ComputeGraphPtr &compute_graph, const NodePtr &node);
  Status GetDynamicDimsNodeInfo(const NodePtr &node);
  Status GetGearAndRealOutSizeInfo(const ComputeGraphPtr &graph, const NodePtr &node);
  Status GetRealOutputSizeOfCase(const ComputeGraphPtr &graph, const size_t input_index, const NodePtr &case_node);
  Status GetGearAndRealOutShapeInfo(const NodePtr &node);

  Status AllocateResource(const Node &node);
  Status AllocateQueueResource(const Node &node, const OpDescPtr &op_desc, const NamedAttrs &resource);
  Status AllocateDvppChlResource(const OpDescPtr &op_desc);
  Status AllocateVdecChlResource(const OpDescPtr &op_desc);

  Status UpdateOpInputValue(const OpDescPtr &op_desc, const int32_t input_index, const uint32_t queue_id) const;

  Status InitFileConstant(const NodePtr &node);

  Status InitQueueDataNodes(const std::vector<NodePtr> &queue_data_nodes,
                            const uint32_t data_index,
                            std::set<uint64_t> &input_outside_addrs);

  Status InitInputZeroCopy(const OpDescPtr &op_desc, const uint32_t data_index,
                           std::set<uint64_t> &input_outside_addrs);

  void SaveFusionOpInfo(const OpDescPtr &op_desc, ProfileInfo &profile) const;

  /// @brief check the op_type whether is send/recv operator or not
  /// @return true
  /// @return false
  static bool IsSendRecvOp(const std::string &op_type);

  Status GenFmMemAllocations();

  Status GenFixedFmMemAllocations();

  void ConstructFixedFmActiveMemBase();

  void CreateMultiBatchDataBuffer(std::vector<DataBuffer> &blobs);

  Status GenInputMemAllocations(const std::map<uint32_t, OpDescPtr> &index_to_data);

  Status GenOutputMemAllocations(const std::vector<OpDescPtr> &output_op_list);
  Status GenSliceOutputMemAllocations(const std::vector<OpDescPtr> &output_op_list);

  Status GenMemAllocations(const std::map<uint32_t, OpDescPtr> &index_to_data,
                           const std::vector<OpDescPtr> &output_op_list);

  void InitModelInputsMergeCopyHostMem();

  void InitBatchMemcpyH2d();

  Status GetMemAllocationByLogicAddr(const uint64_t addr, MemAllocationAndOffset &allocation_info) const;

  void CalculateMemAllocationsHitInfo();
  std::map<uint32_t, ZeroCopyOffset> FilterZeroCopyAddrs() const;

  bool CopyOnlyAddrCheck() const;

  void LogModelDevMemInfo() const;

  void InitExceptionDumpInfo(const OpDescPtr &op_desc, uintptr_t args, size_t arg_size,
                             const std::map<uint64_t, uint64_t> &cust_to_relevant_offset,
                             ExtraOpInfo &extra_dump_info) const;
  uint32_t GetGraphId() const;
  std::string GetWeightsMemId() const;
  Status ParseHostInputIndexOption(const size_t input_num);

  std::string FindAddrRefreshKernelFile(const std::string& npu_arch) const;
  Status LoadAndRegisterAddrRefreshKernel(const std::string& file_path);
  std::string FindKernelInPath(const std::string& path, const std::string& npu_arch) const;

  bool is_weight_mem_has_inited_{false};
  bool is_feature_map_mem_has_inited_{false};

  uint32_t model_id_{0U};
  uint32_t runtime_model_id_{0U};
  uint32_t sub_model_id_{0U};
  std::string name_;

  // used for inference data dump
  std::string om_name_;
  std::string dump_model_name_;
  // used for file constant path
  std::string file_constant_weight_dir_;

  // user set FileConstant device memory, key is file name
  std::map<std::string, FileConstantMem> file_constant_user_device_mems_;

  // 使用智能指针管理外置权重的归一内存，通过自定义deleter实现自动释放
  // 若用户提供了内存，则deleter为空操作；否则deleter调用MemManager::FreeMemory
  std::unique_ptr<void, std::function<void(void*)>> external_weight_combined_mem_addr_;

  uint32_t version_{0U};
  GeModelPtr ge_model_;  // release after DavinciModel::Init

  std::map<int64_t, OpDescPtr> op_list_;  // release after DavinciModel::Init
  std::map<int64_t, std::shared_ptr<Operator>> operator_list_;
  bool need_clear_dfx_cache_{false}; // clear profiling and dump cache after DavinciModel::Init

  uintptr_t global_step_addr_{0U};
  uintptr_t loop_per_iter_addr_{0U};
  uintptr_t loop_cond_addr_{0U};
  uint64_t global_step_size_{0U};
  bool need_free_global_step_addr_{false};

  std::map<uint32_t, ZeroCopyOffset> input_data_info_;
  std::map<uint32_t, ZeroCopyOffset> output_data_info_;
  std::map<uint32_t, bool> output_data_to_slice_flag_;

  std::set<uint64_t> real_virtual_addrs_;

  // output op: save cce op actual needed memory size
  std::vector<int64_t> output_memory_size_list_;

  std::thread thread_id_;

  std::shared_ptr<ModelListener> listener_;

  bool run_flg_{false};
  // check whether model is running with data
  bool running_flg_{false};
  bool aicpu_flg_{false};

  std::mutex mux_run_flg_;

  bool has_finalized_{false};
  // label if reusable_stream_allocator_ is generated by Init or outer input(such as rt2)
  bool is_outer_allocator_{false};
  ReusableStreamAllocator *reusable_stream_allocator_{nullptr};
  std::unordered_map<uint32_t, int32_t> stream_task_num_;
  std::map<uint32_t, uint32_t> stream_to_first_task_id_;
  int32_t priority_;

  std::vector<rtStream_t> stream_list_;
  std::vector<uint32_t> stream_flag_list_;

  std::mutex all_hccl_stream_list_mutex_;
  std::vector<rtStream_t> all_hccl_stream_list_; // hccl 从流

  // for reuse hccl_follow_stream
  std::mutex capacity_of_stream_mutex_;
  std::map<int64_t, std::vector<rtStream_t>> main_follow_stream_mapping_;

  std::vector<rtNotify_t> notify_list_;
  std::vector<rtEvent_t> event_list_;

  std::unordered_set<std::string > hccl_group_id_set_;
  std::vector<rtEvent_t> hccl_group_ordered_event_list_;
  std::vector<rtStream_t> hccl_group_ordered_stream_list_; // 流资源为hccl管理

  std::mutex hccl_task_stream_set_mutex_;
  std::unordered_set<uint64_t> hccl_task_stream_set_; // hccl task所在的流
  std::map<uint64_t, std::vector<size_t>> stream_to_task_index_list_;
  std::map<int64_t, int64_t> split_logic_stream_2_origin_logic_stream_;

  std::vector<rtLabel_t> label_list_;
  std::set<uint32_t> label_id_indication_;

  std::mutex label_args_mutex_;
  std::map<uint32_t, std::pair<void *, uint32_t>> label_goto_args_;

  std::mutex outside_addrs_mutex_;

  struct CopyOnlyAddrs {
    std::set<uint64_t> copy_only_addrs;
    std::unordered_set<uint64_t> refdata_virtual_addrs;
    Status Insert(uint64_t addr) {
      if (refdata_virtual_addrs.count(addr) > 0) {
        return FAILED;
      }
      (void)copy_only_addrs.insert(addr);
      return SUCCESS;
    }
    int32_t Count(uint64_t addr) const {
      return copy_only_addrs.count(addr);
    }
    bool IsRefDataAddr(uint64_t addr) const {
      return (refdata_virtual_addrs.count(addr) > 0);
    }
  };

  CopyOnlyAddrs copy_only_addrs_;     // Address need copy to original place.

  std::vector<TaskInfoPtr> task_list_;
  // rt_model_handle
  rtModel_t rt_model_handle_{nullptr};

  rtStream_t rt_model_stream_{nullptr};

  rtStream_t rt_stream_to_destroy_{nullptr};

  // label if rt_model_stream_ is (1)true: inner created, (2)false: outer stream
  bool is_inner_model_stream_{false};

  bool is_forbidden_stream_{false};

  bool is_async_mode_{false};  // For NN execute, Async mode use rtMemcpyAsync on rt_model_stream_.
  ExecuteMode last_execute_mode_{ExecuteMode::INITIALIZATION};

  bool is_stream_list_bind_{false};
  bool is_pure_head_stream_{false};
  rtStream_t rt_head_stream_{nullptr};
  rtStream_t rt_entry_stream_{nullptr};

  // ACL queue schedule, save queue ids for Init.
  std::unordered_map<uint32_t, bool> is_queue_data_;  // key:data_index
  std::vector<TaskInfoPtr> cpu_task_list_;
  std::vector<QueueAttrs> input_queue_attrs_;    // input queue created by caller.
  std::vector<QueueAttrs> output_queue_attrs_;   // output queue created by caller.
  std::vector<uintptr_t> input_mbuf_list_;   // input mbuf created by dequeue task.
  std::vector<uintptr_t> output_mbuf_list_;  // output mbuf created by dequeue task.
  std::vector<int32_t> input_fusion_offsets_; // input fusion offsets set up by caller.

  uint64_t session_id_{0U};
  error_message::ErrorManagerContext error_context_{};

  uint32_t device_id_{0U};

  std::mutex flowctrl_op_index_internal_map_mutex_;
  std::map<uint32_t, uint32_t> flowctrl_op_index_internal_map_;

  std::vector<rtStream_t> active_stream_list_;
  std::set<uint32_t> active_stream_indication_;

  std::set<uint32_t> hcom_streams_;
  std::set<uint32_t> hcom_attach_streams_;
  RuntimeParam runtime_param_;
  fe::PlatFormInfos platform_infos_{};

  ModelKernelHandlesManager model_kernel_handles_manager_;
  TBEKernelHandle bin_kernel_handle_;

  // for profiling task and graph info
  std::vector<TaskDescInfo> task_desc_info_;
  std::vector<ProfileInfo> profile_list_;
  std::vector<NodeBasicInfoWrapper> node_basic_infos_{};
  std::vector<ApiInfoWrapper> prof_launch_apis_{};
  std::vector<gert::ContextIdInfoWrapper> context_id_infos_{};
  std::unordered_map<uint32_t, std::set<uint32_t>> logic_stream_ids_to_physic_stream_ids_{};
  // for data dump
  DataDumper data_dumper_;
  ModelQueueParam model_queue_param_;
  ExceptionDumper exception_dumper_;
  OpdebugRegister opdebug_register_;
  uint64_t iterator_count_{0U};
  std::map<OpDescPtr, void *> saved_task_addrs_;  // release after DavinciModel::Init
  void *l1_fusion_addr_ = nullptr;

  bool known_node_ = false;
  bool feature_base_refreshable_ = false;

  OmeContext run_context_;
  std::vector<std::vector<int64_t>> batch_info_;
  std::vector<std::vector<int64_t>> combined_batch_info_;
  std::vector<std::string> user_designate_shape_order_;
  int32_t dynamic_type_ = 0;
  bool is_dynamic_ = false;

  std::vector<uint64_t> batch_size_;

  // if model is first execute
  bool is_first_execute_{true};
  // for op debug
  bool is_op_debug_reg_ = false;
  bool is_online_infer_dynamic_ = false;
  bool is_getnext_sink_dynamic_ = false;
  std::vector<int32_t> cur_dynamic_dims_;
  void *netoutput_last_input_addr_ = nullptr;
  int64_t netoutput_last_input_size_ = 0;
  size_t shape_of_cur_dynamic_dims_ = 0U;
  // key: input_index: input is merge node; value: each gear info and each output size
  std::map<size_t, std::map<std::vector<int32_t>, int64_t>> merge_nodes_gear_and_real_out_size_info_;
  // key: input_index: input is merge node; value: each gear info and each output shape
  std::map<size_t, std::map<std::vector<int32_t>, std::vector<int64_t>>> merge_nodes_gear_and_real_out_shape_info_;
  std::vector<std::vector<int32_t>> all_gears_info_;

  bool has_output_node_ = false;
  bool is_dynamic_aipp_ = false;
  std::vector<std::string> dynamic_output_shape_info_;

  std::vector<std::vector<uint64_t>> input_addrs_list_;
  std::vector<std::vector<uint64_t>> output_addrs_list_;

  std::vector<int64_t> output_buffer_size_;
  std::vector<GeShape> output_shape_info_;

  std::vector<bool> input_no_tiling_flag_;
  bool has_no_tiling_input_ = false;
  std::vector<bool> output_no_tiling_flag_;
  bool has_no_tiling_output_ = false;
  std::map<uint32_t, uint64_t> output_no_tiling_data_addr_;

  std::map<uint32_t, OriginInputInfo> orig_input_info_;
  std::map<uint32_t, AippConfigInfo> aipp_info_list_;
  std::map<uint32_t, std::pair<InputAippType, size_t>> aipp_type_list_;
  std::map<uint32_t, std::pair<std::vector<InputOutputDims>, std::vector<InputOutputDims>>> aipp_dims_info_;

  std::map<uint32_t, NamedAttrs> align_attrs_;
  std::vector<InputOutputDescInfo> origin_input_descs_;
  std::vector<InputOutputDescInfo> input_descs_;
  std::vector<InputOutputDescInfo> input_descs_dims_;
  std::vector<uint32_t> input_formats_;
  std::vector<InputOutputDescInfo> output_descs_;
  std::vector<uint32_t> output_formats_;

  // op name to attrs mapping
  std::map<std::string, std::map<std::string, std::vector<std::string>>> op_name_to_attrs_;

  std::map<rtStream_t, rtEvent_t> stream_2_event_;

  AiCpuResources aicpu_resources_;
  std::map<std::string, std::string> file_id_and_path_map_;

  // for support overflow detection
  void *globalworkspace_overflow_addr_ = nullptr;
  bool use_control_input_queue_ = false;
  bool use_control_output_queue_ = false;
  bool isGraphLevelSat_ = false;
  // stream sync time
  int32_t stream_sync_timeout_ = 0;
  bool is_stream_sync_timeout_ = false;

  std::shared_ptr<gert::OpImplSpaceRegistryV2Array> space_registries_;
  MsprofGeTaskType GetTaskType(const domi::FftsPlusCtxDef &ctx_def) const;
  uint32_t GetBlockDim(const domi::FftsPlusCtxDef &ctx_def) const;
  uint32_t GetThreadId(const domi::FftsPlusCtxDef &ctx_def) const;
  uint32_t GetBlockDim(const ModelTaskType type, const domi::TaskDef &task_def, const OpDescPtr &op_desc) const;
  ModelArgsManager args_manager_;
  size_t logical_fm_mem_allocations_size_{0U};
  std::vector<MemAllocation> logical_mem_allocations_;
  uint64_t fm_hit_count_{0U};
  uint64_t model_io_hit_count_{0U};
  uint64_t *allocation_ids_to_active_base_addr_{nullptr};
  std::vector<std::pair<uint32_t, uint32_t>> refreshable_input_index_and_allocation_ids_;
  std::vector<std::pair<uint32_t, uint32_t>> refreshable_output_index_and_allocation_ids_;
  std::vector<std::pair<uint32_t, uint32_t>> refreshable_input_index_no_frozen_and_allocation_ids_;
  std::vector<std::pair<uint32_t, uint32_t>> refreshable_fm_index_and_allocation_ids_;
  std::vector<std::pair<uint32_t, uint32_t>> fixed_fm_index_and_allocation_ids_;
  std::map<uint32_t, MemAllocationSlice> input_indexes_to_copy_info_;
  std::map<uint32_t, MemAllocationSlice> output_indexes_to_copy_info_;
  std::vector<uint32_t> input_index_to_allocation_ids_;  // 保存零拷贝的input index和allocation id的关系
  std::vector<uint32_t> output_index_to_allocation_ids_; // 保存零拷贝的output index和allocation id的关系
  std::vector<uint64_t> input_index_to_active_mem_base_addrs_;  // 保存零拷贝的input index和对应的active mem base的关系
  std::vector<uint64_t> output_index_to_active_mem_base_addrs_; // 保存零拷贝的output index和对应的active mem base的关系
  std::vector<uint32_t> zero_copy_input_indexes_;               // 保存零拷贝的input indexes
  std::vector<uint32_t> zero_copy_output_indexes_;              // 保存零拷贝的output indexes
  std::vector<uint32_t> zero_copy_input_indexes_no_frozen_;    // 保存执行时需要零拷贝的output indexes
  bool is_first_time_model_execute_ = false;

  // for input fusion h2d copy
  std::shared_ptr<uint8_t> input_merge_copy_mem_base_;  // host buffer
  uint64_t input_merge_copy_mem_size_{0UL};
  std::map<uint32_t, uint64_t> input_index_to_merge_copy_offset_;
  uint32_t fisrt_input_index_of_merge_copy_{0U};  // index of input locate at the begin of merge copy buffer
  MemcpyBatchParam memcpy_batch_params_;

  void InitModelProf();
  void InitModelExecuteProf();
  void GetCurTimestamp(uint64_t &cur_time);
  void GetStageTimestampStart(ModelProfStage stage);
  void GetStageTimestampEnd(ModelProfStage stage);
  void PrintfModelProfOfModelLoad();
  void PrintfModelProfOfModelExecute();
  void UpdateTaskTypeStat(const uint32_t task_type, const uint64_t start_t, const uint64_t end_t);
  DavinciModelProf mdl_prof_{};

  // 编译态识别能零拷贝的输入/输出，但是由于在执行态，用户给的输入/输出是host placement的，需要临时组装需要拷贝的信息
  // 这些临时组装的需要强制拷贝的copy_info信息，用以下数据结构表达
  std::map<uint32_t, uint64_t> output_indexes_to_tensor_size_;
  std::map<uint32_t, MemAllocationSlice> host_pls_input_indexes_to_copy_info_;
  std::map<uint32_t, MemAllocationSlice> host_pls_output_indexes_to_copy_info_;
  Status ReportTaskTimeL0Info(const uint32_t prof_model_id);
  void SaveNodeBasicProfInfo(const OpDescPtr &op_desc, NodeBasicInfoWrapper &node_basic_info,
                             const TaskProfInfo &prof_api, uint32_t block_dim, uint32_t task_type);
  void SaveNodeApiProfInfo(const std::string &op_name, const TaskProfInfo &prof_api);
  void SaveTaskProfInfo(const std::string &op_name, const OpDescPtr &op_desc, const TaskProfInfo &prof_api);
  Status SaveProfilingInfoByContext(const domi::FftsPlusCtxDef &ctx_def, const OpDescPtr &sgt_node,
                                    const TaskProfInfo &prof_api, bool ffts_flag);
  void SaveProfilingInfoByPartitionCall(const domi::FftsPlusTaskDef &ffts_plus_task_def,
                                        const OpDescPtr &sgt_node, const TaskProfInfo &prof_api);
  Status MallocPhysicalMemory();
  Status InitCopyHostInputInfos();
  size_t fm_mem_allocations_start_id_{0U};
  bool has_hccl_task_ = false;

  std::atomic<uint64_t> prof_count_{0UL};

  // dynamic sched info
  bool need_model_config_ {false};
  uint32_t model_uuid_ = 0U;
  uint8_t logLevel_ = DLOG_DEBUG;
  QueueAttrs status_output_queue_{};
  bool need_report_status_ = false;

  bool is_static_model_addr_fixed_ = false;
  size_t fixed_fm_mem_allocations_start_id_{0U};
  size_t logical_fixed_fm_mem_allocations_size_{0U};

  std::map<std::string, void *> platform_infos_addr_;
  std::map<std::string, void *> cust_platform_infos_addr_; // 已经launch的缓存
  std::map<std::string, std::pair<void *, size_t>> cust_platform_infos_addr_to_launch_; // 已申请内存，待launch的缓存
  bool is_dump_to_std_enable_ = false;
  std::unordered_set<uint32_t> frozen_input_indexes_;

  uint64_t no_frozen_input_allocation_base_id_{0U};

  struct ModelDevMemStatistic {
    uint64_t alloc_size;
    uint64_t shared_size;
  };

  std::map<uint32_t, void *> mem_event_id_mem_map_;
  ModelDevMemStatistic dev_mem_statistic_{};
  bool support_extend_memory_full_{false};
  std::chrono::time_point<std::chrono::system_clock> davinci_model_stage_time_[kStageEnd]{};
  std::unordered_set<uint32_t> copy_host_input_indexes_;
  std::vector<CopyHostInputInfo> copy_host_input_infos_;
  uint64_t host_input_size_{0UL};
  std::map<uint32_t, std::shared_ptr<MemoryBlockManager>> mem_type_to_allocator_;

  std::map<uint32_t, std::vector<ArgDesc>> tiling_sink_task_arg_descs_list_;

  std::vector<DumpProcState> dump_fsm_state_;
  std::unordered_set<std::string> dump_op_in_range_;

  // For output reuse input memory address validation
  std::vector<std::pair<size_t, size_t>> io_same_addr_pairs_;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DAVINCI_MODEL_H_

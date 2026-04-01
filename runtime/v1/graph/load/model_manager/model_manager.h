/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_MODEL_MANAGER_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_MODEL_MANAGER_H_

#include <algorithm>
#include <map>
#include <unordered_map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <thread>

#include "fwk_adpt_struct.h"
#include "common/dump/dump_properties.h"
#include "common/model/ge_root_model.h"
#include "common/model/executor.h"
#include "common/context/ome_context.h"
#include "ge/ge_api_types.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/helper/model_helper.h"
#include "framework/common/helper/om_file_helper.h"
#include "framework/common/ge_types.h"
#include "graph/model.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/load/model_manager/davinci_model.h"
#include "hybrid/hybrid_davinci_model.h"
#include "mmpa/mmpa_api.h"
#include "runtime/context.h"
#include "runtime/rt.h"
#include "framework/runtime/rt_session.h"
#include "graph/manager/session_id_manager.h"

namespace ge {
struct AICPUKernelHolder {
  KernelBinPtr kernel_ptr;
  size_t launch_count;
};

class ModelManager {
 public:
  static ModelManager &GetInstance();

  /// @ingroup domi_ome
  /// @brief load and init model without input or output queue.
  /// @param [out] model_id: model id.
  /// @param [in] root_model: RootModel load from offline model file.
  /// @param [in] priority: model priority.
  /// @return Status run result
  /// @author @
  Status LoadModelWithoutQ(uint32_t &model_id, const GeRootModelPtr &root_model, const int32_t priority = 0);

  /// @ingroup domi_ome
  /// @brief load and init model
  /// @param [in] model_id model id
  /// @param [in] model including model ptr and size
  /// @param [in] rt_session runtime context
  /// @param [in/out] info model task generate info
  /// @return Status run result
  /// @author
  Status LoadModelOffline(const ModelData &model, const ModelParam &model_param, uint32_t &model_id,
                          const gert::RtSession *rt_session = nullptr);

  /// @ingroup domi_ome
  /// @brief load and init model and this function may be used in multi-threaded scenarios, so pay attention
  /// @param [out] model_id model id
  /// @param [in] model modeldef datatype
  /// @param [in] listener used to return result
  /// @param [in] isTrainMode model type
  /// @return Status run result
  /// @author @
  Status LoadModelOnline(uint32_t &model_id, const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node,
                         const uint32_t device_id, const aclrtStream stream = nullptr);

  Status DoLoadHybridModelOnline(const uint32_t model_id,
                                 const ModelData &model,
                                 const uint32_t device_id,
                                 const GeRootModelPtr &ge_root_model,
                                 const std::shared_ptr<ModelListener> &listener,
                                 const aclrtStream stream = nullptr);

  /// @ingroup ge
  /// @brief ACL case, Load task list with queue.
  /// @param [out] model_id: model id for manager.
  /// @param [in] model_data: Model data load from offline model file.
  /// @param [in] arg: input/output queue ids from user, num equals Data Op, and file constant mems
  /// @return: 0 for success / others for fail
  Status LoadModelWithQ(uint32_t &model_id, const ModelData &model_data, const ModelQueueArg &arg);

  /// @ingroup ge
  /// @brief ACL case, Load task list with queue.
  /// @param [out] model_id: model id for manager.
  /// @param [in] root_model: RootModel load from offline model file.
  /// @param [in] model_queue_param: params and queue ids and create from user.
  /// @param [in] priority: model priority.
  /// @return: 0 for success / others for fail
  Status LoadModelWithQueueParam(uint32_t &model_id,
                                 const GeRootModelPtr &root_model,
                                 const ModelQueueParam &model_queue_param,
                                 const int32_t priority = 0,
                                 const bool need_update_session_id = true);

  Status LoadModelWithQueueParam(uint32_t &model_id,
                                 const ModelData &model_data,
                                 const ModelQueueParam &model_queue_param);

  /// @ingroup domi_ome
  /// @brief unload model and free resources
  /// @param [in] model_id model id
  /// @return Status run result
  /// @author
  Status Unload(const uint32_t model_id);

  /// @ingroup domi_ome
  /// @brief process input data and run asynchronously
  /// cannot be invoked by multiple thread
  /// if one fails, other continue
  /// @param [in] input_data   input data
  /// @return SUCCESS          success
  /// @return PARAM_INVALID    parameter invalid
  /// @return MODEL_NOT_READY  model not ready
  /// @return PUSH_DATA_FAILED push data into model queue failed
  /// @author
  Status SyncExecuteModel(const uint32_t model_id, const std::vector<gert::Tensor> &inputs,
                          std::vector<gert::Tensor> &outputs);

  Status DataInputTensor(const uint32_t model_id, const std::shared_ptr<RunArgs> &args);
  /// @ingroup domi_ome
  /// @brief model start to run
  Status Start(const uint32_t model_id);

  /// @ingroup domi_ome
  /// @brief  ACL case, do not start new thread, return result
  /// @param [in] model_id  model id
  /// @param [in] stream   model stream
  /// @param [in] async_mode  is asynchronize mode.
  /// @param [in] input_data  model input data
  /// @param [in] input_desc  description of model input data
  /// @param [out] output_data  model output data
  /// @param [out] output_desc  description of model output data
  Status ExecuteModel(const uint32_t model_id, const aclrtStream stream, const bool async_mode,
                      const InputData &input_data, const std::vector<GeTensorDesc> &input_desc,
                      OutputData &output_data, std::vector<GeTensorDesc> &output_desc,
                      const std::vector<GeTensor> &input_tensor, const std::vector<GeTensor> &output_tensor);

  /// @ingroup domi_ome
  /// @brief  ACL case, do not start new thread, return result
  /// @param [in] model_id  model id
  /// @param [in] stream   model stream
  /// @param [in] async_mode  is asynchronize mode.
  /// @param [in] inputs  model inputs
  /// @param [in] outputs  model outputs
  Status ExecuteModel(const uint32_t model_id, const aclrtStream stream, const bool async_mode,
                      const std::vector<GeTensor> &input_tensor, std::vector<GeTensor> &output_tensor);

  Status ExecuteModelAsync(const uint32_t model_id, const aclrtStream stream, const bool async_mode,
                           const std::vector<GeTensor> &input_tensor,
                           std::vector<GeTensor> &output_tensor);

  Status ExecuteModelWithStreamAsync(const uint32_t model_id, const GraphNodePtr &graph_node,
                                     const std::vector<gert::Tensor> &input_tensor,
                                     std::vector<gert::Tensor> &output_tensor, const aclrtStream stream);
  Status ExecuteModelWithStream(const uint32_t model_id, const aclrtStream stream, const bool async_mode,
                                  const std::vector<gert::Tensor> &input_tensor,
                                  std::vector<gert::Tensor> &output_tensor);

  Status ExecuteModelWithStreamAsync(const uint32_t model_id, const GraphNodePtr &graph_node,
                                     const std::vector<GeTensor> &input_tensor, std::vector<GeTensor> &output_tensor,
                                     const aclrtStream stream = nullptr);

  Status SyncExecuteHybridModel(const uint32_t model_id, const std::vector<gert::Tensor> &inputs,
                                std::vector<gert::Tensor> &outputs);

  /// @ingroup domi_ome
  /// @brief model stop
  Status Stop(const uint32_t model_id);

  /// @ingroup domi_ome
  /// @brief comment handle function
  static Status HandleCommand(const Command &cmd_info);

  /// @ingroup domi_ome
  /// @brief get model memory usage
  /// @param [in] model_id  model id
  /// @return SUCCESS          success
  /// @return PARAM_INVALID    parameter invalid
  Status GetMaxUsedMemory(const uint32_t model_id, uint64_t &max_size);

  /// @ingroup domi_ome
  /// @brief get model input and output size
  /// @param [in] model_id  model id
  /// @param [out] input_shape   input tensor
  /// @param [out] output_shape  output tensor
  /// @return SUCCESS          success
  /// @return PARAM_INVALID    parameter invalid
  Status GetInputOutputDescInfo(const uint32_t model_id, std::vector<InputOutputDescInfo> &input_desc,
                                std::vector<InputOutputDescInfo> &output_desc);

  Status GetInputOutputDescInfo(const uint32_t model_id, std::vector<InputOutputDescInfo> &input_desc,
                                std::vector<InputOutputDescInfo> &output_desc, std::vector<uint32_t> &inputFormats,
                                std::vector<uint32_t> &outputFormats, const bool new_model_desc = false);

  /// @ingroup ge
  /// @brief Get dynamic batch_info
  /// @param [in] model_id
  /// @param [out] batch_info
  /// @param [out] dynamic_type
  /// @return execute result
  Status GetDynamicBatchInfo(const uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info,
                             int32_t &dynamic_type);

  /// @ingroup ge
  /// @brief Get combined dynamic dims info
  /// @param [in] model_id
  /// @param [out] batch_info
  /// @return execute result
  Status GetCombinedDynamicDims(const uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info);

  /// @ingroup ge
  /// @brief Get user designate shape order
  /// @param [in] model_id
  /// @param [out] user_input_shape_order
  /// @return execute result
  Status GetUserDesignateShapeOrder(const uint32_t model_id, std::vector<std::string> &user_input_shape_order);

  /// @ingroup ge
  /// @brief Get AIPP info
  /// @param [in] model_id
  /// @param [in] index
  /// @param [out] aipp_info
  /// @return execute result
  Status GetAippInfo(const uint32_t model_id, const uint32_t index, AippConfigInfo &aipp_info);

  Status GetAippType(const uint32_t model_id, const uint32_t index, InputAippType &type, size_t &aipp_index);

  Status GetCurrentShape(const uint32_t model_id, std::vector<int64_t> &batch_info, int32_t &dynamic_type);

  Status GetNodeAttr(const uint32_t model_id, const std::string &op_name, const std::string &attr_name,
                     std::string &attr_info);

  Status GetOutputShapeInfo(const uint32_t model_id, std::vector<std::string> &dynamic_output_shape_info);

  Status SetDynamicSize(const uint32_t model_id, const std::vector<uint64_t> &batch_num, const int32_t dynamic_type);

  /// @ingroup domi_ome
  /// @brief Get model according to given id
  std::shared_ptr<DavinciModel> GetModel(const uint32_t id);

  bool IsModelSharedSession(const uint32_t model_id);

  Status RecoverAllModel(const int32_t device_id);

  std::shared_ptr<hybrid::HybridDavinciModel> GetHybridModel(const uint32_t id);

  Status KernelLaunchEx(const aicpu::FWKAdapter::FWKOperateType op_type, const uint64_t session_id,
                        const uint32_t model_id, const uint32_t sub_model_id);

  Status CreateAicpuSession(const uint64_t session_id);

  static Status GetModelMemAndWeightSize(const ModelData &model, size_t &mem_size, size_t &weight_size);

  Status DestroyAicpuSessionForDevice(const uint64_t session_id,
                                      const uint32_t device_id,
                                      const bool need_set_device = false);

  void DestroyAicpuSession(const uint64_t session_id, const bool single_device = false, const uint32_t device_id = 0U);

  Status DestroyAicpuKernel(const uint64_t session_id, const uint32_t model_id, const uint32_t sub_model_id);

  void CreateAicpuKernel(const uint64_t session_id, const uint32_t model_id, const uint32_t sub_model_id,
                         const uint64_t kernel_id);

  Status DestroyAicpuSessionForInfer(const uint32_t model_id);

  Status LoadCustAicpuSo(const OpDescPtr &op_desc, const std::string &so_name, bool &loaded);
  Status LoadCustAicpuSo(const CustAICPUKernelPtr &aicpu_kernel, const std::string &so_name, bool &loaded);
  Status GetCustAicpuSo(const std::string &so_name, CustAICPUKernelPtr &aicpu_kernel);

  Status LaunchCustAicpuSo();

  Status ClearAicpuSo();

  Status GetPlatformInfosSoName(std::string &so_name);

  Status LaunchKernelCustAicpuSo(const std::string &kernel_name);

  Status LaunchKernelCheckAicpuOp(const std::vector<std::string> &aicpu_optype_list,
                                  const std::vector<std::string> &aicpu_tf_optype_list);

  Status CheckAicpuOpList(const GeModelPtr &ge_model);

  Status GetOrigInputInfo(const uint32_t model_id, const uint32_t index, OriginInputInfo &orig_input_info);

  Status GetAllAippInputOutputDims(const uint32_t model_id, const uint32_t index,
                                   std::vector<InputOutputDims> &input_dims,
                                   std::vector<InputOutputDims> &output_dims);

  bool IsDynamicShape(const uint32_t model_id);
  bool IsNeedHybridLoad(const GeRootModel &ge_root_model) const;
  Status GetOpDescInfo(const uint32_t device_id, const uint32_t stream_id, const uint32_t task_id,
                       OpDescInfo &desc_info) const;

  bool IsSocketClose() const { return is_socket_close_; }

  void SetSocketCloseStatus(const bool status) { is_socket_close_ = status; }

  uint32_t GetRunningFlag(const uint32_t model_id);
  uint32_t GetDataInputerSize(const uint32_t model_id);

  Status SetCallbackHybridLoad(const uint32_t model_id, const GeRootModelPtr &ge_root_model,
                               const RunAsyncCallbackV2 &callback);
  Status ModelSubscribe(const uint32_t graph_id);

  Status UpdateFeatureMemoryBase(const uint32_t model_id, const uintptr_t mem_base, const size_t size);

  Status PaRemapped(const uint32_t model_id, const uint64_t va, const uint64_t new_pa, const uint64_t len,
                    std::vector<std::pair<uint64_t, uint64_t>> &cross_ranges);

  Status GetRuntimeModelId(const uint32_t model_id, uint32_t &model_runtime_id);

  Status SetCallBackFuncForDumpManager();
  Status LoadTaskForDavinciModel(const DumpProperties &dump_properties);
  Status UnloadTaskForDavinciModel(const DumpProperties &dump_properties);
  std::string GetCustTilingDeviceUniqueSoName(const uint32_t model_id, const std::string &so_name);
  KernelBinPtr GetCustTilingDeviceSoBin(const std::string &unique_so_name);
  KernelBinPtr GetBuiltinTilingDeviceSoBin(const std::string &so_name);
  std::string GetBuiltinTilingDeviceSoName(const std::string &so_name) const;
  Status LoadCustAicpuSoAndUpdateSoName(const uint32_t model_id, std::string &so_name);
  Status LoadBuiltinAicpuSoAndUpdateSoName(const uint32_t device_id, std::string &so_name);
  Status LaunchBuiltinAicpuSo(const uint32_t device_id);
  Status ClearBuiltinAicpuSo(const uint32_t device_id);
  Status InitOpMasterDeviceSo(const uint32_t &model_id, const GeRootModelPtr &ge_root_model);
  void GenModelId(uint32_t &id);
  static uint8_t *MallocWeightsMem(const std::string &weights_mem_id, const uint32_t device_id,
                                   const size_t weights_size);
  static Status FreeWeightsMem(const std::string &weights_mem_id, const uint32_t device_id, uint8_t *weights_mem_base);
  aclrtBinHandle GetPlatformBinHandle() const {
    return platform_bin_handle_;
  }
  void SetPlatformBinHandle(const aclrtBinHandle &bin_handle) {
    if (platform_bin_handle_ == nullptr) {
      platform_bin_handle_ = bin_handle;
    }
  }
 private:
  /// @ingroup domi_ome
  /// @brief constructor
  ModelManager();

  /// @ingroup domi_ome
  /// @brief destructor
  ~ModelManager();

  /// @ingroup domi_ome
  /// @brief insert new model into model manager set
  void InsertModel(const uint32_t model_id, const std::shared_ptr<DavinciModel> &davinci_model);
  void InsertModel(const uint32_t model_id, const std::shared_ptr<hybrid::HybridDavinciModel> &hybrid_model);

  /// @ingroup domi_ome
  /// @brief delete model from model manager set
  Status DeleteModel(const uint32_t id);

  static Status HandleDumpCommand(const Command &cmd_info);
  static Status HandleProfModelSubscribeCommand(const Command &cmd_info);
  static Status HandleProfModelUnsubscribeCommand(const Command &cmd_info);
  static Status HandleProfInitCommand(const Command &cmd_info);
  static Status HandleProfFinalizeCommand(const Command &cmd_info);
  static Status HandleProfStartCommand(const Command &cmd_info);
  static Status HandleProfStopCommand(const Command &cmd_info);

  static Status GetModelIdByCmd(const Command &cmd_info, uint32_t &model_id);

  Status ProfModelSubscribe(const uint64_t module, const uint32_t model_id, const uint32_t graph_id);

  void CreateMonitorThread();

  void RecordTsSnapshot();

  void ClearMonitorThread();

  void SetDumpProperties(const std::shared_ptr<DavinciModel> &davinci_model);

  static void getDevMsgCallback(const char_t *const msg, const uint32_t len);

  void GenDataInputOutputData(const uint32_t model_id, const std::vector<Tensor> &inputs, InputData &input_data,
                              OutputData &output_data);

  Status ExternalAllocatorMalloc(const GraphId graph_id, const uint32_t model_id, const GraphNodePtr &graph_node,
                                 const aclrtStream stream);
  const std::map<std::string, AICPUKernelHolder> CollectWorkingBuiltinAicpuSo(const std::string &kernel_name,
                                                                              const uint32_t device_id);
  Status LoadBuiltinAicpuSo(const KernelBinPtr &aicpu_kernel, const uint32_t device_id, const std::string &so_name);
  Status LaunchKernelBuiltinAicpuSo(const std::string &kernel_name, const uint32_t device_id);
  void AddSharedSessionModel(const uint32_t model_id);
  void DeleteSharedSessionModel(const uint32_t model_id);

  std::mutex model_shared_session_mutex_;
  std::set<uint32_t> model_shared_session_; // 存在多个模型共享同一份rtSession资源场景，在此记录这些modelID
  std::map<uint32_t, std::shared_ptr<DavinciModel>> model_map_;
  std::map<uint32_t, std::shared_ptr<hybrid::HybridDavinciModel>> hybrid_model_map_;
  std::map<std::string, std::vector<uint64_t>> model_aicpu_kernel_;
  std::atomic<uint32_t> max_model_id_ = 1U;
  std::recursive_mutex map_mutex_;
  std::map<uint64_t, std::set<uint32_t>> sess_id_to_device_ids_;
  std::mutex cust_aicpu_mutex_;
  std::map<uintptr_t, std::map<std::string, AICPUKernelHolder>> cust_aicpu_so_;
  std::mutex builtin_aicpu_mutex_;
  std::map<uint32_t, std::map<std::string, AICPUKernelHolder>> builtin_aicpu_so_;

  std::mutex dump_properties_mutex_;
  DumpProperties dump_properties_;
  bool is_socket_close_ = false;

  std::mutex monitor_mtx_;
  std::condition_variable monitor_cv_;
  std::thread monitor_thread_;
  bool stop_monitor_ = true;
  static std::string record_file_name_;
  std::string trigger_file_name_;
  bool is_dump_registered_ = false;
  std::mutex dump_regis_mutex_;
  aclrtBinHandle platform_bin_handle_{nullptr};
  std::mutex op_master_device_mutex_;
  std::unordered_map<std::string, OpSoBinPtr> built_in_op_master_so_names_to_bin_;
  std::unordered_map<std::string, OpSoBinPtr> cust_op_master_so_names_to_bin_;
  std::unordered_map<std::string, std::string> cust_op_master_so_names_to_unique_name_;
  std::unordered_map<std::string, std::string> cust_op_master_so_datas_to_name_;

  /**
   * 临时方案： 单session多实例场景通过sess_id_id+graph_id+graph_name标识相同权重可共享
   * 正式方案： 后续通过属复用ge.constLifecycle方式实现，和动态shape模型权重共享方案一起实现。
   *          预计25年H1落地
   */
  struct SharedWeightAddrInfo {
    uint64_t shared_num{0};
    uint8_t *weight_addr_{nullptr};
  };
  static std::mutex weights_mem_mtx_;
  static std::unordered_map<std::string, SharedWeightAddrInfo> weights_mem_ids_to_addr_info_;
};
Status MallocConstMemory(const GraphId &graph_id, const GraphNodePtr &graph_node,
                         const AllocatorPtr external_allocator);

Status MallocFeatureMemory(const GraphId &graph_id, const uint32_t model_id, const GraphNodePtr &graph_node,
                           const AllocatorPtr external_allocator);

Status CalcTensorSizeByShape(const GeShape &shape, DataType data_type, size_t &ret_tensor_size);

Status SetNetOutputTensorInfo(const GraphId &graph_id, const GraphNodePtr &graph_node);

Status MallocOutputsMemory(const GraphId &graph_id, const GraphNodePtr &graph_node,
                           const AllocatorPtr external_allocator, std::vector<GeTensor> &outputs);

Status MallocOutputsMemory(const GraphId &graph_id, const GraphNodePtr &graph_node,
                           const AllocatorPtr external_allocator, std::vector<gert::Tensor> &outputs);
Status CreateGertTensor(const GeTensorDescPtr ge_tensor_desc, gert::Tensor &gert_tensor);
void FreeFeatureMemory(const ge::GraphNodePtr &graph_node);
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_MODEL_MANAGER_H_

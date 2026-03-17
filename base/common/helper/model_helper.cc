/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <base/err_msg.h>
#include "framework/common/helper/model_helper.h"
#include "common/checker.h"
#include "common/helper/model_parser_base.h"
#include "common/model/ge_model.h"
#include "common/model/ge_root_model.h"
#include "common/op_so_store/op_so_store_utils.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "framework/omg/version.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/unfold/graph_unfolder.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "framework/omg/omg_inner_types.h"
#include "mmpa/mmpa_api.h"
#include "graph/types.h"
#include "common/proto_util/proto_util.h"
#include "graph/ge_local_context.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/build/memory/var_mem_assign_util.h"
#include "graph/manager/graph_var_manager.h"
#include "common/math/math_util.h"
#include "common/file_constant_utils/file_constant_utils.h"
#include "common/helper/file_saver.h"
#include "common/model/model_introduction.h"
#include "common/model/model_compress_manager.h"
#include "common/host_resource_center/host_resource_serializer.h"
#include "graph/utils/math_util.h"
#include "ge_context.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "register/core_num_utils.h"
#include "acl/acl_rt.h"

namespace {
constexpr uint32_t kOriginalOmPartitionNum = 1U;
constexpr int32_t kModuleTypeVectorCore = 7;
const string kOpsProto = "libopsproto_rt2.0.so";
const string kOpMaster = "libopmaster_rt2.0.so";
const string kInner = "built-in";
const string kOpsProtoPath = "/op_proto/lib/";
const string kOpsGraphPath = "/op_graph/lib/";
const string kOpMasterPath = "/op_impl/ai_core/tbe/op_tiling/lib/";
const string kOpHostPath = "/op_impl/ai_core/tbe/op_host/lib/";
const std::string kHcomGroups = "hcom_group_names";
const std::string kSocInfoKey = "SoCInfo";
const std::string kAicCntKey = "ai_core_cnt";
const std::string kVecCoreCntKey = "vector_core_cnt";
const std::string kEnumNameCubeNum = "cube_num";
const std::string kEnumNameVectorNum = "vector_num";
const std::string kSoSuffix = ".so";
const std::string kRt2SoSuffix = "rt2.0.so";
const std::string kRtSoSuffix = "rt.so";
const std::string kLegacySoSuffix = "_legacy.so";
const std::string kCompilerVersion = "compiler_version=";
const std::string kOppVersion = "Version=";
const std::string kVersionInfo = "/version.info";
const std::string kHardwareInfo = "ge.hardwareInfo";
}

namespace ge {
namespace {
// 模型中大小支持超4G的Partition类型，当前只识别了表格中部分，其他部分分析支持超4G后也可以加到这个表格中
std::unordered_set<ModelPartitionType> kSupportBeyond4GTypes = {WEIGHTS_DATA, TBE_KERNELS, CUST_AICPU_KERNELS};
const std::string kMultiBatchNodePostfix = "_ascend_mbatch_batch_";
void UpdateFftsPlusTaskAddr(domi::TaskDef &task_def, const std::map<int64_t, int64_t> &logical_addr_mapping) {
  auto *const mutable_ffts_plus_task = task_def.mutable_ffts_plus_task();
  for (int32_t i = 0; i < mutable_ffts_plus_task->ffts_plus_ctx_size(); ++i) {
    auto *const mutable_ctx_def = mutable_ffts_plus_task->mutable_ffts_plus_ctx(i);
    auto *const mutable_mix_aic_aiv_ctx = mutable_ctx_def->mutable_mix_aic_aiv_ctx();
    for (int32_t k = 0; k < mutable_mix_aic_aiv_ctx->task_addr_size(); ++k) {
      const auto orig_task_addr = mutable_mix_aic_aiv_ctx->task_addr(k);
      const auto &it = logical_addr_mapping.find(static_cast<int64_t>(orig_task_addr));
      if (it != logical_addr_mapping.cend()) {
        mutable_mix_aic_aiv_ctx->set_task_addr(k, static_cast<uint64_t>(it->second));
        GELOGD("update task_addr[%d], [%ld] to [%ld]", k, orig_task_addr, it->second);
      } else {
        GELOGW("update task_addr[%d] [%ld] NOT match", k, orig_task_addr);
      }
    }
  }
}

NodePtr TryGetVarNodeByOffset(const std::map<int64_t, NodePtr> &var_offsets, const int64_t offset) {
  const auto find_ret = var_offsets.find(offset);
  return (find_ret == var_offsets.cend() ? nullptr : find_ret->second);
}

std::string GetOppPkgPath(const std::string &opp_path, const string &whole_pkg_path,
                          const string &sub_pkg_path, const string &os_cpu_type, bool &is_sub_pkg) {
  is_sub_pkg = false;
  const auto idx = opp_path.find(kInner);
  if (idx != std::string::npos) {
    return PluginManager::GetOppPkgPath(opp_path.substr(0, idx) + kInner, whole_pkg_path, sub_pkg_path, os_cpu_type,
                                        is_sub_pkg);
  }
  return opp_path;
}
}  // namespace

std::string ModelHelper::output_file_name_;

Status ModelHelper::SaveModelPartition(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                       const ModelPartitionType type, const uint8_t* const data,
                                       const size_t size, const size_t model_index) const {
  if ((size < 1U) || ((size > UINT32_MAX) && (kSupportBeyond4GTypes.count(type) == 0UL))) {
    GELOGE(PARAM_INVALID, "[Add][ModelPartition]Failed, partition size %zu invalid", size);
    if (size > UINT32_MAX) {
      std::string item = "item";
      static std::map<ModelPartitionType, string> item_type_map = {
        {MODEL_DEF, "model info"},
        {TASK_INFO, "task info"},
        {SO_BINS, "so bins"},
        {TILING_DATA, "tiling data"},
        {MODEL_INOUT_INFO, "model introductions"},
        {STATIC_TASK_DESC, "static task desc"},
        {DYNAMIC_TASK_DESC, "dynamic task desc"},
        {TASK_PARAM, "task param"},
        {PRE_MODEL_DESC, "pre model desc"},
        {PRE_MODEL_SQE, "pre model task"},
        {PRE_KERNEL_ARGS, "pre kernel args"},
        {PRE_MODEL_DESC_EXTEND, "pre model desc extend"},
      };
      if (item_type_map.find(type) != item_type_map.end()) {
        item = item_type_map[type];
      }
      (void)REPORT_PREDEFINED_ERR_MSG(
          "E13023", std::vector<const char *>({"size", "item", "maxsize"}),
          std::vector<const char *>({std::to_string(size).c_str(), item.c_str(), std::to_string(UINT32_MAX).c_str()}));
    }
    REPORT_INNER_ERR_MSG("E19999", "Add model partition failed, partition size %zu "
                       "invalid", size);
    return PARAM_INVALID;
  }
  if (data == nullptr) {
    GELOGE(PARAM_INVALID, "[Add][ModelPartition]Failed, data is null");
    REPORT_INNER_ERR_MSG("E19999", "Add model partition failed, data is null");
    return PARAM_INVALID;
  }
  ModelPartition partition_model;
  partition_model.data = data;
  partition_model.size = static_cast<uint64_t>(size);
  partition_model.type = type;
  if (om_file_save_helper->AddPartition(partition_model, model_index) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Add][ModelPartition]Failed, partition size %zu", size);
    REPORT_INNER_ERR_MSG("E19999", "Add model partition failed, partition size %zu", size);
    return PARAM_INVALID;
  }
  GELOGI("[Add][ModelPartition]Success, partition type[%d] size[%zu]", static_cast<int32_t>(type), size);
  return SUCCESS;
}

Status ModelHelper::SaveSizeToModelDef(const GeModelPtr &ge_model, const size_t model_index) const {
  std::vector<int64_t> om_info;
  GELOGD("SaveSizeToModelDef weight_data_size is %zu, ge_model_weight data is %p", ge_model->GetWeightSize(),
         ge_model->GetWeightData());
  om_info.push_back(static_cast<int64_t>(ge_model->GetWeightSize()));

  const auto &tbe_kernel_store = ge_model->GetTBEKernelStore();
  GELOGD("SaveSizeToModelDef tbe_kernels_size is %zu", tbe_kernel_store.DataSize());
  om_info.push_back(static_cast<int64_t>(tbe_kernel_store.DataSize()));

  const auto &cust_aicpu_kernel_store = ge_model->GetCustAICPUKernelStore();
  GELOGD("SaveSizeToModelDef cust aicpu kernels size is %zu", cust_aicpu_kernel_store.DataSize());
  om_info.push_back(static_cast<int64_t>(cust_aicpu_kernel_store.DataSize()));

  const std::shared_ptr<domi::ModelTaskDef> model_task_def = ge_model->GetModelTaskDefPtr();
  if (model_task_def == nullptr) {
    GELOGD("SaveSizeToModelDef task_info_size is 0.");
    om_info.push_back(0);
  } else {
    const size_t partition_task_size = model_task_def->ByteSizeLong();
    GELOGD("SaveSizeToModelDef task_info_size is %zu", partition_task_size);
    om_info.push_back(static_cast<int64_t>(partition_task_size));
  }

  if (model_index == 0U) {
    GELOGD("SaveSizeToModelDef so store size is %zu", GetOpStoreDataSize());
    om_info.push_back(static_cast<int64_t>(GetOpStoreDataSize()));
  } else {
    om_info.push_back(0U); // only first model save so.
  }

  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListInt(*(ge_model.get()), "om_info_list", om_info),
                   GELOGE(FAILED, "SetListInt of om_info_list failed.");
                   return FAILED);

  return SUCCESS;
}

Status ModelHelper::ConfigureAttrCompressionMode(const string &mode) {
  if (mode != "true" && mode != "false") {
    GELOGE(PARAM_INVALID, "[Validate][AttrCompressionMode] Invalid value '%s'. "
           "Only 'true' or 'false' are allowed.", mode.c_str());
    return PARAM_INVALID;
  }

  attr_compression_enabled_ = (mode == "true");
  GELOGI("[AttrCompression] Configured from options: enabled=%s",
         attr_compression_enabled_ ? "true" : "false");
  return SUCCESS;
}

Status ModelHelper::SaveModelDef(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                                 ge::Buffer &model_buffer, const size_t model_index) const {
  // Calculate final compression decision based on enabled flag
  const bool should_compress = ShouldCompress();

  // Detailed logging for debugging
  GELOGD("[AttrCompression] enabled=%s, is_offline=%d, is_need_compress=%d, decision=%s",
         attr_compression_enabled_ ? "true" : "false",
         static_cast<int32_t>(is_offline_),
         static_cast<int32_t>(is_need_compress_),
         should_compress ? "COMPRESS" : "SKIP");

  if (should_compress) {
    (void)ModelCompressManager::Compress(ge_model);
  }
  const ModelPtr model_tmp = ge::MakeShared<ge::Model>(ge_model->GetName(), ge_model->GetPlatformVersion());
  if (model_tmp == nullptr) {
    GELOGE(FAILED, "[Creat][Model]Failed, Model %s Ptr", ge_model->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "Create Model %s Ptr failed", ge_model->GetName().c_str());
    return FAILED;
  }
  model_tmp->SetGraph(ge_model->GetGraph());
  model_tmp->SetVersion(ge_model->GetVersion());
  const Status ret = SaveSizeToModelDef(ge_model, model_index);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Save][SizeToModelDef]Failed, model %s, error_code %u",
           ge_model->GetName().c_str(), ret);
    REPORT_INNER_ERR_MSG("E19999", "Save SizeToModelDef failed, model %s, error_code %u",
                      ge_model->GetName().c_str(), ret);
    return ret;
  }

  model_tmp->SetAttr(ge_model->MutableAttrMap());
  GE_ASSERT_SUCCESS(model_tmp->SaveWithoutSeparate(model_buffer));
  GELOGD("MODEL_DEF size is %zu", model_buffer.GetSize());
  if (model_buffer.GetSize() > 0U) {
    if (SaveModelPartition(om_file_save_helper, ModelPartitionType::MODEL_DEF, model_buffer.GetData(),
                           model_buffer.GetSize(), model_index) != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Add][ModelPartition]Failed, model %s, model_def size %zu, model_index %zu",
             ge_model->GetName().c_str(), model_buffer.GetSize(), model_index);
      REPORT_INNER_ERR_MSG("E19999", "Add model graph partititon failed, model %s, model_def %zu, "
                        "model_index %zu", ge_model->GetName().c_str(), model_buffer.GetSize(), model_index);
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

Status ModelHelper::SaveModelWeights(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                                     const size_t model_index) const {
  GELOGD("WEIGHTS_DATA size is %zu, %p", ge_model->GetWeightSize(), ge_model->GetWeightData());
  // weight is not necessary
  if (ge_model->GetWeightSize() > 0U) {
    GE_CHK_STATUS_RET(SaveModelPartition(om_file_save_helper,
                                         ModelPartitionType::WEIGHTS_DATA,
                                         ge_model->GetWeightData(),
                                         ge_model->GetWeightSize(),
                                         model_index),
                      "Add weight partition failed");
  }
  return SUCCESS;
}

Status ModelHelper::SaveModelTbeKernel(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                       const GeModelPtr &ge_model, const size_t model_index) const {
  const auto &tbe_kernel_store = ge_model->GetTBEKernelStore();
  GELOGD("TBE_KERNELS size is %zu", tbe_kernel_store.DataSize());
  if (tbe_kernel_store.DataSize() > 0U) {
    GE_CHK_STATUS_RET(
        SaveModelPartition(om_file_save_helper, ModelPartitionType::TBE_KERNELS,
                           tbe_kernel_store.Data(), tbe_kernel_store.DataSize(),
                           model_index),
        "Add tbe kernel partition failed");
  }
  return SUCCESS;
}

Status ModelHelper::SaveModelCustAICPU(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                       const GeModelPtr &ge_model, const size_t model_index) const {
  const auto &cust_aicpu_kernel_store = ge_model->GetCustAICPUKernelStore();
  GELOGD("cust aicpu kernels size is %zu", cust_aicpu_kernel_store.DataSize());
  if (cust_aicpu_kernel_store.DataSize() > 0U) {
    GE_CHK_STATUS_RET(SaveModelPartition(om_file_save_helper,
                                         ModelPartitionType::CUST_AICPU_KERNELS,
                                         cust_aicpu_kernel_store.Data(),
                                         cust_aicpu_kernel_store.DataSize(), model_index),
                      "Add cust aicpu kernel partition failed");
  }
  return SUCCESS;
}

Status ModelHelper::SaveModelIntroduction(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                          const GeModelPtr &ge_model, const bool is_dynamic) const {
  std::unique_ptr<ModelIntroduction> modelIntroduction = ge::MakeUnique<ModelIntroduction>();
  GE_IF_BOOL_EXEC(modelIntroduction == nullptr,
                  REPORT_INNER_ERR_MSG("E19999", "ModelIntroduction failed, it is nullptr, model %s",
                  ge_model->GetName().c_str());
                  GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Create][ModelIntroduction]Failed, it is nullptr, "
                  "model %s", ge_model->GetName().c_str()); return ACL_ERROR_GE_MEMORY_ALLOCATION);
  GE_ASSERT_SUCCESS(modelIntroduction->Init(ge_model, is_dynamic),
                    "ModelIntroduction Init Failed, model %s", ge_model->GetName().c_str());
  std::shared_ptr<uint8_t> buff = modelIntroduction->Data();
  ge_model->SetModelInOutInfo(buff);
  GELOGD("MODEL_INOUT_INFO size is %d", modelIntroduction->DataSize());
  if (modelIntroduction->DataSize() > 0U) {
    GE_CHK_STATUS_RET(SaveModelPartition(om_file_save_helper,
                                         ModelPartitionType::MODEL_INOUT_INFO,
                                         buff.get(),
                                         static_cast<size_t>(modelIntroduction->DataSize()), 0U),
                      "Add model introduction partition failed");
  }
  return SUCCESS;
}

Status ModelHelper::SaveModelTaskDef(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                                     ge::Buffer &task_buffer, const size_t model_index) const {
  const std::shared_ptr<domi::ModelTaskDef> model_task_def = ge_model->GetModelTaskDefPtr();
  if (model_task_def == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Creat][ModelTaskDef]Failed, it is nullptr, "
           "model %s", ge_model->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "Creat model task def failed, it is nullptr, model %s",
                      ge_model->GetName().c_str());
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  const size_t partition_task_size = model_task_def->ByteSizeLong();
  GE_IF_BOOL_EXEC((partition_task_size == 0U) || (static_cast<int32_t>(partition_task_size) > INT_MAX),
                  GELOGE(FAILED, "[Check][ModelDefSize]Invalid, size %zu, model %s",
                         partition_task_size, ge_model->GetName().c_str());
                  REPORT_INNER_ERR_MSG("E19999", "Model def size %zu check invalid, model %s",
                                    partition_task_size, ge_model->GetName().c_str());
                      return FAILED);

  task_buffer = ge::Buffer(partition_task_size);
  if (task_buffer.GetSize() == 0U) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Allocate][ModelTaskDefBuffer]Failed, "
           "model def size %zu, model %s", partition_task_size, ge_model->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "Allocate model task def buffer failed, model def size %zu "
                      "model %s", partition_task_size, ge_model->GetName().c_str());
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  (void)model_task_def->SerializePartialToArray(task_buffer.GetData(), static_cast<int32_t>(partition_task_size));

  GELOGD("TASK_INFO op_size:%d, stream_num:%u", model_task_def->op().size(), model_task_def->stream_num());
  GELOGD("TASK_INFO size is %zu", partition_task_size);

  if (SaveModelPartition(om_file_save_helper, ModelPartitionType::TASK_INFO, task_buffer.GetData(),
                         partition_task_size, model_index) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Add][ModelTaskDefPartition]Failed, model def size %zu, "
           "model_index %zu, model %s",
           partition_task_size, model_index, ge_model->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "Add model task def partition failed, model def size %zu "
                      "model_index %zu, model %s",
                      partition_task_size, model_index, ge_model->GetName().c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status ModelHelper::SaveModelHeader(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                                    const size_t model_num, const bool need_check_os_cpu,
                                    const bool is_unknow_shape) const {
  // Save target/version to model_header
  ModelFileHeader &model_header = om_file_save_helper->GetModelFileHeader();
  model_header.platform_type = ge_model->GetPlatformType();
  model_header.om_ir_version = ge_model->GetVersion();
  model_header.model_num = static_cast<uint32_t>(model_num);
  model_header.is_unknow_model = is_unknow_shape ? 1U : 0U;
  const std::string platform_version = ge_model->GetPlatformVersion();

  errno_t err = memcpy_s(model_header.platform_version, static_cast<size_t>(PLATFORM_VERSION_LEN),
                         platform_version.c_str(), platform_version.size() + 1U);
  if (err != EOK) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION,
           "[Save][Model]Failed while allocating memory for platform_version %s, model %s, "
           "errno %d",
           platform_version.c_str(), ge_model->GetName().c_str(), err);
    REPORT_INNER_ERR_MSG("E19999",
                      "ModelHelper save model %s failed while "
                      "allocating memory for platform_version %s, errno %d",
                      ge_model->GetName().c_str(), platform_version.c_str(), err);
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  const std::string version = reinterpret_cast<char_t *>(model_header.platform_version);
  GELOGD("Platform version save: %s", version.c_str());
  if (need_check_os_cpu) {
    model_header.need_check_os_cpu_info = static_cast<uint8_t>(OsCpuInfoCheckTyep::NEED_CHECK);
    GELOGD("need_check_os_cpu_info save:%u", model_header.need_check_os_cpu_info);
  }

  size_t name_size = ge_model->GetName().size();
  name_size = (name_size > (MODEL_NAME_LENGTH - 1U)) ? (MODEL_NAME_LENGTH - 1U) : name_size;
  err = memcpy_s(model_header.name, MODEL_NAME_LENGTH, ge_model->GetName().c_str(), name_size);
  if (err != EOK) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION,
           "[Save][Model]Failed while allocating memory for model %s, errno %d",
           ge_model->GetName().c_str(), err);
    REPORT_INNER_ERR_MSG("E19999", "ModelHelper save model failed while allocating memory "
                      "for model %s,errno %d", ge_model->GetName().c_str(), err);
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  const std::string model_name = reinterpret_cast<char_t *>(model_header.name);
  GELOGD("Model name save:%s", model_name.c_str());
  return SUCCESS;
}

Status ModelHelper::LoadAndStoreOppSo(const std::unordered_set<std::string> &op_so_set, const SoBinType so_bin_type) {
  for (const auto &op_so : op_so_set) {
    uint32_t bin_len = 0U;
    auto op_so_bin = GetBinDataFromFile(op_so, bin_len);
    GE_ASSERT_NOTNULL(op_so_bin, "open so fail, path=%s", op_so.c_str());
    const auto &pos = op_so.find_last_of("/");
    GE_ASSERT_TRUE(pos != std::string::npos);
    const auto &so_name = op_so.substr(pos + 1UL);
    const auto &vendor_name = op_so.substr(0, pos);
    const auto proto_bin = ge::MakeShared<OpSoBin>(so_name, vendor_name, std::move(op_so_bin), bin_len, so_bin_type);
    GE_ASSERT_NOTNULL(proto_bin);
    op_so_store_.AddKernel(proto_bin);
    GELOGD("Add op so:%s[%u] success.", op_so.c_str(), so_bin_type);
  }
  return SUCCESS;
}

Status ModelHelper::LoadAndStoreOppSo(const string &path, bool is_split, bool is_sub_pkg) {
  const bool is_built_in_path = (path.find(kInner) != std::string::npos);
  std::vector<std::string> op_so_list;
  if ((!is_sub_pkg) && is_built_in_path) {
    const std::string so_buff = is_split ? kRtSoSuffix : kRt2SoSuffix;
    ge::PluginManager::GetFileListWithSuffix(path, so_buff, op_so_list);
  } else {
    ge::PluginManager::GetFileListWithSuffix(path, kSoSuffix, op_so_list);
  }

  if (is_built_in_path && op_so_list.empty()) {
    GELOGE(FAILED, "Can not find any op so in path:%s", path.c_str());
    return FAILED;
  }

  std::unordered_set<std::string> op_so_set(op_so_list.begin(), op_so_list.end());
  GE_ASSERT_SUCCESS(LoadAndStoreOppSo(op_so_set, SoBinType::kSpaceRegistry));
  return SUCCESS;
}

Status ModelHelper::GetSoBinData(const string &cpu_info, const string &os_info) {
  const auto is_split = PluginManager::IsSplitOpp();

  std::vector<std::string> vendors;
  PluginManager::GetPackageSoPath(vendors);
  const std::string os_cpu_type = os_info + "/" + cpu_info;
  bool is_sub_pkg = false;
  for (size_t i = 0U; i < vendors.size(); i++) {
    if (vendors[i].empty()) {
      continue;
    }
    GELOGD("Begin to scan op proto and master so from path:%s", vendors[i].c_str());
    const auto last_kernel_num = op_so_store_.GetKernelNum();
    const auto op_proto_path = GetOppPkgPath(vendors[i] + kOpsProtoPath + os_cpu_type, kOpsProtoPath,
                                             kOpsGraphPath, os_cpu_type, is_sub_pkg);
    GE_ASSERT_SUCCESS(LoadAndStoreOppSo(op_proto_path, is_split, is_sub_pkg),
                      "Load and store op proto so failed, path:%s",
                      op_proto_path.c_str());

    const auto op_master_path = GetOppPkgPath(vendors[i] + kOpMasterPath + os_cpu_type, kOpMasterPath,
                                              kOpHostPath, os_cpu_type, is_sub_pkg);
    GE_ASSERT_SUCCESS(LoadAndStoreOppSo(op_master_path, is_split, is_sub_pkg),
                      "Load and store op master so failed, path:%s",
                      op_master_path.c_str());

    // 保存打包的自定义算子包版本号
    if ((vendors[i].find(kInner) == std::string::npos) && (op_so_store_.GetKernelNum() > last_kernel_num)) {
      std::string compiler_version;
      if (PluginManager::GetVersionFromPathWithName(vendors[i] + kVersionInfo, compiler_version, kCompilerVersion)) {
        (void)custom_compiler_versions_.emplace(compiler_version);
      }
    }
  }

  // 加载升级包
  if (is_split) {
    std::vector<std::string> path_vec;
    std::string ops_proto_path;
    if (PluginManager::GetUpgradedOpsProtoPath(ops_proto_path) == ge::SUCCESS) {
      PluginManager::SplitPath(ops_proto_path, path_vec);
      for (const auto &path : path_vec) {
        const auto root_path = GetOppPkgPath(path + "/lib/" + os_cpu_type, kOpsProtoPath, kOpsGraphPath, os_cpu_type,
                                             is_sub_pkg);
        (void)LoadAndStoreOppSo(root_path, true, is_sub_pkg);
      }
    }
    path_vec.clear();
    std::string op_tiling_path;
    if (PluginManager::GetUpgradedOpMasterPath(op_tiling_path) == ge::SUCCESS) {
      PluginManager::SplitPath(op_tiling_path, path_vec);
      for (const auto &path : path_vec) {
        const auto root_path = GetOppPkgPath(path + "/op_tiling/lib/" + os_cpu_type, kOpMasterPath, kOpHostPath,
                                             os_cpu_type, is_sub_pkg);
        (void)LoadAndStoreOppSo(root_path, true, is_sub_pkg);
      }
    }
  }
  return SUCCESS;
}

const uint8_t *ModelHelper::GetOpSoStoreData() const { return op_so_store_.Data(); }

size_t ModelHelper::GetOpStoreDataSize() const { return op_so_store_.DataSize(); }

Status ModelHelper::SetModelCompilerVersion(const GeModelPtr &first_ge_model) {
  if (!custom_compiler_versions_.empty()) {
    std::string compiler_version;
    for (const auto &it : custom_compiler_versions_) {
      compiler_version.empty() ? compiler_version.append(it) : compiler_version.append("," + it);
    }
    GE_ASSERT_TRUE(ge::AttrUtils::SetStr(*(first_ge_model.get()), ATTR_MODEL_COMPILER_VERSION, compiler_version),
                   "Ge model set compiler version failed");
    GELOGD("Ge model set compiler version:%s success.", compiler_version.c_str());
  }
  return SUCCESS;
}

Status ModelHelper::SaveSpaceRegistrySoBin(const GeRootModelPtr &ge_root_model, const GeModelPtr &first_ge_model,
                                           string &output_file_name) {
  if (!OpSoStoreUtils::IsSoBinType(ge_root_model->GetSoInOmFlag(), SoBinType::kSpaceRegistry)) {
    return SUCCESS;
  }
  std::string host_env_os;
  std::string host_env_cpu;
  (void)GetThreadLocalContext().GetOption("ge.host_env_os", host_env_os);
  (void)GetThreadLocalContext().GetOption("ge.host_env_cpu", host_env_cpu);

  GELOGI("get host env cpu:%s, os:%s", host_env_cpu.c_str(), host_env_os.c_str());
  if (host_env_os.empty() || host_env_cpu.empty()) {
    GELOGW("[SaveSoStoreModelPartitionInfo] get cpu or os info empty!!");
    return SUCCESS;
  }
  GE_ASSERT_SUCCESS(GetSoBinData(host_env_cpu, host_env_os));
  const auto position = output_file_name.find(".om");
  const string cpu_os_str = "_" + host_env_os + "_" + host_env_cpu;
  if (position < output_file_name.length()) {
    (void)output_file_name.insert(position, cpu_os_str);
  } else {
    (void)output_file_name.append(cpu_os_str);
  }

  GE_ASSERT_SUCCESS(SetModelCompilerVersion(first_ge_model));
  is_so_store_ = true;
  GELOGI("[SpaceRegistry]Save to OpSoStore success.");
  return SUCCESS;
}

Status ModelHelper::SaveOpMasterDeviceSoBin(const GeRootModelPtr &ge_root_model) {
  if (!OpSoStoreUtils::IsSoBinType(ge_root_model->GetSoInOmFlag(), SoBinType::kOpMasterDevice)) {
    return SUCCESS;
  }
  GE_ASSERT_SUCCESS(LoadAndStoreOppSo(ge_root_model->GetOpMasterDeviceSoSet(), SoBinType::kOpMasterDevice));
  GELOGI("[OpMasterDevice]Save to OpSoStore success.");
  return SUCCESS;
}

Status ModelHelper::SaveAutofuseSoBin(const GeRootModelPtr &ge_root_model) {
  GE_ASSERT_NOTNULL(ge_root_model);
  auto root_graph = ge_root_model->GetRootGraph();
  GE_ASSERT_NOTNULL(root_graph);
  auto bin_file_buffer = root_graph->GetExtAttr<std::map<std::string, ge::OpSoBinPtr>>("bin_file_buffer");
  if (bin_file_buffer != nullptr) {
    GELOGD("No need to save autofuse so to om, since bin_file_buffer already exists.");
    return SUCCESS;
  }
  if (!OpSoStoreUtils::IsSoBinType(ge_root_model->GetSoInOmFlag(), SoBinType::kAutofuse)) {
    return SUCCESS;
  }
  GE_ASSERT_SUCCESS(LoadAndStoreOppSo(ge_root_model->GetAutofuseSoSet(), SoBinType::kAutofuse));
  GELOGI("[AutofuseSo]Save to AutofuseSo success.");
  return SUCCESS;
}

Status ModelHelper::SaveSoStoreModelPartitionInfo(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                                  const GeRootModelPtr &ge_root_model, string &output_file_name,
                                                  const GeModelPtr &first_ge_model) {
  GELOGI("so in om flag:0x%x", ge_root_model->GetSoInOmFlag());
  if (ge_root_model->GetSoInOmFlag() == 0U) {
    output_file_name_ = output_file_name;
    return SUCCESS;
  }

  GE_ASSERT_SUCCESS(SaveSpaceRegistrySoBin(ge_root_model, first_ge_model, output_file_name));
  GE_ASSERT_SUCCESS(SaveOpMasterDeviceSoBin(ge_root_model));
  GE_ASSERT_SUCCESS(SaveAutofuseSoBin(ge_root_model));

  GE_ASSERT_TRUE(op_so_store_.Build(), "Build op so store failed.");
  const auto op_so_data = GetOpSoStoreData();
  const auto data_size = GetOpStoreDataSize();
  GE_ASSERT_SUCCESS(SaveModelPartition(om_file_save_helper, ModelPartitionType::SO_BINS, op_so_data, data_size, 0UL),
                    "Add so store failed");
  output_file_name_ = output_file_name;
  return SUCCESS;
}

Status ModelHelper::SaveTilingData(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                   const GeRootModelPtr &ge_root_model) {
  const HostResourceCenterPtr &host_resource_center = ge_root_model->GetHostResourceCenterPtr();
  GE_ASSERT_NOTNULL(host_resource_center);
  uint8_t *data{nullptr};
  size_t data_len{0UL};
  GE_ASSERT_SUCCESS(host_serializer_.SerializeTilingData(*host_resource_center, data, data_len));
  if (data_len > 0UL) {
    GELOGI("Tiling data partition size: %zu.", data_len);
    GE_ASSERT_SUCCESS(SaveModelPartition(om_file_save_helper, ModelPartitionType::TILING_DATA, data, data_len, 0U),
                      "Save tiling data to model failed.");
  }
  return SUCCESS;
}

Status ModelHelper::SaveAllModelPartiton(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                         const GeModelPtr &ge_model, ge::Buffer &model_buffer, ge::Buffer &task_buffer,
                                         const size_t model_index) const {
  GE_CHK_STATUS_RET(EnsureKernelBuilt(ge_model), "ensure kernel built failed, model=%s.", ge_model->GetName().c_str());
  if (SaveModelDef(om_file_save_helper, ge_model, model_buffer, model_index) != SUCCESS) {
    GELOGE(FAILED, "[Save][ModelDef]Failed, model %s, model index %zu",
           ge_model->GetName().c_str(), model_index);
    return FAILED;
  }

  if (SaveModelWeights(om_file_save_helper, ge_model, model_index) != SUCCESS) {
    GELOGE(FAILED, "[Save][ModelWeights]Failed, model %s, model index %zu",
           ge_model->GetName().c_str(), model_index);
    REPORT_INNER_ERR_MSG("E19999", "ModelHelper save mode weights failed, model %s, model index %zu",
                      ge_model->GetName().c_str(), model_index);
    return FAILED;
  }

  if (SaveModelTbeKernel(om_file_save_helper, ge_model, model_index) != SUCCESS) {
     GELOGE(FAILED, "[Save][ModelTbeKernel]Failed, model %s, model index %zu",
            ge_model->GetName().c_str(), model_index);
     REPORT_INNER_ERR_MSG("E19999", "ModelHelper save model tbe kernel failed, model %s, "
                       "model index %zu", ge_model->GetName().c_str(), model_index);
    return FAILED;
  }

  if (SaveModelCustAICPU(om_file_save_helper, ge_model, model_index) != SUCCESS) {
    GELOGE(FAILED, "[Save][ModelCustAICPU]Failed, model %s, model index %zu",
           ge_model->GetName().c_str(), model_index);
    REPORT_INNER_ERR_MSG("E19999", "ModelHelper save model cust aicpu failed, model %s "
                      "model index %zu", ge_model->GetName().c_str(), model_index);
    return FAILED;
  }

  if (SaveModelTaskDef(om_file_save_helper, ge_model, task_buffer, model_index) != SUCCESS) {
    GELOGE(FAILED, "[Save][TaskDef]Failed, model %s, model index %zu",
           ge_model->GetName().c_str(), model_index);
    return FAILED;
  }
  return SUCCESS;
}

Status ModelHelper::SaveToOmModel(const GeModelPtr &ge_model, const std::string &output_file,
                                  ModelBufferData &model, const GeRootModelPtr &ge_root_model) {
  GE_ASSERT_NOTNULL(ge_model, "Ge_model is nullptr");
  if (output_file.empty()) {
    GELOGE(FAILED, "[Save][Model]GraphBuilder SaveModel received invalid file name prefix, "
           "model %s", ge_model->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "GraphBuilder SaveModel received invalid file name prefix, "
                      "model %s", ge_model->GetName().c_str());
    return FAILED;
  }

  std::shared_ptr<OmFileSaveHelper> om_file_save_helper = ge::MakeShared<OmFileSaveHelper>();
  GE_CHECK_NOTNULL(om_file_save_helper);
  ge::Buffer model_buffer;
  ge::Buffer task_buffer;
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetStr(*(ge_model.get()), ATTR_MODEL_ATC_CMDLINE,
                   domi::GetContext().atc_cmdline),
                   GELOGE(FAILED, "SetStr for atc_cmdline failed.");
                   return FAILED);
  std::string cur_version;
  auto ret = GetOppVersion(cur_version);
  if ((ret != SUCCESS) || (!ge::AttrUtils::SetStr(*(ge_model.get()), ATTR_MODEL_OPP_VERSION, cur_version))) {
    GELOGW("Ge model set opp version unsuccessful!");
  }

  string output_file_name = output_file;
  if (ge_root_model != nullptr) {
    if (is_offline_) {
      GE_ASSERT_SUCCESS(SaveSoStoreModelPartitionInfo(om_file_save_helper, ge_root_model, output_file_name, ge_model),
                        "[SaveSoStoreModelPartition]Failed");
    }
    GE_ASSERT_SUCCESS(SaveTilingData(om_file_save_helper, ge_root_model), "Save tiling data to model failed.");
  }

  GE_IF_BOOL_EXEC(SaveModelIntroduction(om_file_save_helper, ge_model) != SUCCESS,
                  GELOGE(FAILED, "[Save][ModelIntroduction]Failed");
                  REPORT_INNER_ERR_MSG("E19999", "Save model introduction failed.");
                  return FAILED);

  ret = SaveAllModelPartiton(om_file_save_helper, ge_model, model_buffer, task_buffer);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Save][AllModelPartition]Failed, model %s, error_code %u",
           ge_model->GetName().c_str(), ret);
    REPORT_INNER_ERR_MSG("E19999", "OmFileSaveHelper save all model partition failed, model %s "
                       "error_code %u", ge_model->GetName().c_str(), ret);
    return ret;
  }

  // static case 3th param is model_num :1U
  ret = SaveModelHeader(om_file_save_helper, ge_model, 1U, is_so_store_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Save][ModelHeader]Failed, model %s, error_code %u",
           ge_model->GetName().c_str(), ret);
    REPORT_INNER_ERR_MSG("E19999", "OmFileSaveHelper save model header failed, model %s "
                       "error_code %u", ge_model->GetName().c_str(), ret);
    return ret;
  }

  ret = om_file_save_helper->SaveModel(output_file_name.c_str(), model, is_offline_ && save_to_file_);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Save][Model]Failed, model %s, output file %s",
           ge_model->GetName().c_str(), output_file.c_str());
    REPORT_INNER_ERR_MSG("E19999", "OmFileSaveHelper save model failed, model %s, "
                       "output file %s", ge_model->GetName().c_str(), output_file.c_str());
    return ret;
  }
  return SUCCESS;
}

void ModelHelper::SaveOutNodesFromRootGraph(const GeRootModelPtr &ge_root_model, GeModelPtr &first_ge_model) const {
  for (const auto &ge_model : ge_root_model->GetSubgraphInstanceNameToModel()) {
    std::vector<std::string> out_node_name;
    (void)ge::AttrUtils::GetListStr(ge_model.second, ge::ATTR_MODEL_OUT_NODES_NAME, out_node_name);
    if (!out_node_name.empty()) {
      GELOGD("Get model out node names from %s success, size = %zu", ge_model.first.c_str(), out_node_name.size());
      (void)ge::AttrUtils::SetListStr(first_ge_model, ge::ATTR_MODEL_OUT_NODES_NAME, out_node_name);
      break;
    }
  }
}

Status ModelHelper::SaveToOmRootModel(const GeRootModelPtr &ge_root_model, const std::string &output_file,
                                      ModelBufferData &model, const bool is_unknown_shape) {
  GE_ASSERT_NOTNULL(ge_root_model, "[Check][GERootModel]Ge_root_model is nullptr");
  GE_ASSERT_TRUE(!output_file.empty(), "[Save][Model]GraphBuilder SaveModel received invalid file name prefix.");
  const auto &name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
  GE_ASSERT_TRUE(!name_to_ge_model.empty(), "[Get][SubModel]Ge_root_model has no sub model.");
  if (!is_unknown_shape) {
    auto &model_root = name_to_ge_model.begin()->second;
    return SaveToOmModel(model_root, output_file, model, ge_root_model);
  }

  GeModelPtr first_ge_model;
  const auto &first_model_it = name_to_ge_model.find(ge_root_model->GetRootGraph()->GetName());
  if (first_model_it == name_to_ge_model.end()) {
    first_ge_model = MakeShared<GeModel>();
    GE_CHECK_NOTNULL(first_ge_model);
    first_ge_model->SetGraph(ge_root_model->GetRootGraph());
  } else {
    first_ge_model = first_model_it->second;
  }
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetStr(*(first_ge_model.get()), ATTR_MODEL_ATC_CMDLINE,
                   domi::GetContext().atc_cmdline),
                   GELOGE(FAILED, "SetStr for atc_cmdline failed.");
                   return FAILED);
  std::string cur_version;
  const auto get_ret = GetOppVersion(cur_version);
  if ((get_ret != SUCCESS) || (!ge::AttrUtils::SetStr(*(first_ge_model.get()), ATTR_MODEL_OPP_VERSION, cur_version))) {
    GELOGW("Ge model set opp version unsuccessful!");
  }
  // ge root model must be the first to be loaded
  std::vector<std::string> model_names{ge_root_model->GetRootGraph()->GetName()};
  for (auto &item : name_to_ge_model) {
    if (item.first != model_names.front()) {
      (void)model_names.emplace_back(item.first);
    }
  }

  std::vector<ge::Buffer> model_buffers(model_names.size());
  std::vector<ge::Buffer> task_buffers(model_names.size());

  is_need_compress_ = true;
  const auto &root_graph = ge_root_model->GetRootGraph();
  GE_ASSERT_NOTNULL(root_graph);
  const auto asc_node_ascbackend = root_graph->FindFirstNodeMatchType("AscBackend");
  const auto asc_node_fusedascbackend = root_graph->FindFirstNodeMatchType("FusedAscBackend");
  const bool has_asc_node = (asc_node_ascbackend != nullptr) || (asc_node_fusedascbackend != nullptr);
  if (has_asc_node) {
    is_need_compress_ = false;
    GE_ASSERT_SUCCESS(ge_root_model->CheckAndSetNeedSoInOM(), "Check so in om failed, model id:%u.", ge_root_model->GetModelId());
  }

  string output_file_name = output_file;
  // only save in model index 0
  std::shared_ptr<OmFileSaveHelper> om_file_save_helper = ge::MakeShared<OmFileSaveHelper>();
  GE_CHECK_NOTNULL(om_file_save_helper);
  if (is_offline_ || has_asc_node) {
    GE_ASSERT_SUCCESS(SaveSoStoreModelPartitionInfo(om_file_save_helper, ge_root_model,
        output_file_name, first_ge_model), "[SaveSoStoreModelPartitionInfo]Failed");
  }
  GE_ASSERT_SUCCESS(SaveTilingData(om_file_save_helper, ge_root_model));

  size_t cur_index = 0U;
  bool is_partitioned = false;
  // shape partitioned scene no need to load root_graph ge_model.
  (void) AttrUtils::GetBool(root_graph,  ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_partitioned);
  if (is_partitioned) {
    // out_nodes should find in all subgraphs in case of first_ge_model has no out_nodes info
    SaveOutNodesFromRootGraph(ge_root_model, first_ge_model);
    GE_ASSERT_SUCCESS(SaveModelIntroduction(om_file_save_helper, first_ge_model, is_unknown_shape));
    if (gert::GraphUnfolder::IsGraphNeedUnfold(root_graph)) {
      GELOGD("only save first model MODEL_DEF");
      GE_ASSERT_SUCCESS(SaveModelDef(om_file_save_helper, first_ge_model, model_buffers[cur_index], cur_index),
                        "Save model def failed, cur_index %zu", cur_index);
    } else {
      GE_ASSERT_SUCCESS(SaveAllModelPartiton(om_file_save_helper, first_ge_model, model_buffers[cur_index],
                                             task_buffers[cur_index], cur_index));
    }
    is_need_compress_ = false;
    ++cur_index;
  }

  for (; cur_index < model_names.size(); ++cur_index) {
    const auto model_name = model_names[cur_index];
    GELOGD("cur model %s index is %zu", model_name.c_str(), cur_index);
    const GeModelPtr &ge_model = name_to_ge_model.at(model_name);
    GE_ASSERT_SUCCESS(SaveAllModelPartiton(om_file_save_helper, ge_model, model_buffers[cur_index],
                                           task_buffers[cur_index], cur_index),
                      "[Save][AllModelPartition]Failed, model name %s, cur_index %zu", model_name.c_str(), cur_index);
  }

  GE_ASSERT_SUCCESS(
      SaveModelHeader(om_file_save_helper, first_ge_model, model_names.size(), is_so_store_, is_unknown_shape),
      "[Save][ModelHeader]Failed, model name %s", first_ge_model->GetName().c_str());

  GE_ASSERT_SUCCESS(om_file_save_helper->SaveModel(output_file_name.c_str(), model, is_offline_ && save_to_file_),
                    "[Save][Model]OmFileSaveHelper save model eturn fail, output_file %s", output_file_name.c_str());
  return SUCCESS;
}

Status ModelHelper::SaveOriginalGraphToOmModel(const ge::Graph &graph, const std::string &output_file) const {
  if (output_file.empty()) {
    GELOGE(FAILED, "[Save][Model]Received invalid file name prefix, output_file %s", output_file.c_str());
    (void)REPORT_PREDEFINED_ERR_MSG(
          "E10059", std::vector<const char *>({"stage", "reason"}),
          std::vector<const char *>({"SaveOriginalGraphToOmModel", "The model output file name cannot be empty"}));
    return FAILED;
  }
  // Get computegraph from graph
  const auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  if (compute_graph == nullptr) {
    GELOGE(FAILED, "[Save][Model]Failed for compute_graph null");
    REPORT_INNER_ERR_MSG("E19999", "Save model failed for compute_graph null");
    return FAILED;
  }
  GE_DUMP(compute_graph, "OriginalGraph");
  // Model
  const ModelPtr model_ptr = ge::MakeUnique<ge::Model>();
  GE_CHECK_NOTNULL_EXEC(model_ptr, return MEMALLOC_FAILED);
  const std::string original_model_name = compute_graph->GetName() + "_original";
  model_ptr->SetName(original_model_name);
  model_ptr->SetGraph(compute_graph);
  model_ptr->SetVersion(static_cast<uint32_t>(OM_PROTO_VERSION));
  std::string framework_version;
  const Status frame_rt = PlatformVersionManager::GetPlatformVersion(framework_version);
  if (frame_rt == SUCCESS) {
    const std::string model_framework_version = framework_version + "." + std::to_string(0);
    model_ptr->SetPlatformVersion(model_framework_version);
  }
  // Model def
  ge::Buffer model_buffer;
  const ge::graphStatus status = model_ptr->SaveWithoutSeparate(model_buffer);
  if (status != ge::GRAPH_SUCCESS) {
    GELOGE(FAILED, "[Save][Model]Failed for save buffer fail, model %s",
           model_ptr->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "Save model %s failed for save buffer fail",
                      model_ptr->GetName().c_str());
    return FAILED;
  }
  const std::unique_ptr<OmFileSaveHelper> om_file_save_helper = ge::MakeUnique<OmFileSaveHelper>();
  GE_CHECK_NOTNULL_EXEC(om_file_save_helper, return MEMALLOC_FAILED);
  ModelPartition partition_model;
  partition_model.data = model_buffer.GetData();
  partition_model.size = static_cast<uint32_t>(model_buffer.GetSize());
  partition_model.type = ModelPartitionType::MODEL_DEF;
  GELOGI("Original Model type[%u],size[%" PRIu64 "]", partition_model.type, partition_model.size);
  if ((partition_model.data != nullptr) && (partition_model.size > 0U)) {
    (void)om_file_save_helper->AddPartition(partition_model);
    // Condition of AddPartition is established, no need to check value
  }
  // Save target/version to model_header
  ModelFileHeader &model_header = om_file_save_helper->GetModelFileHeader();
  model_header.om_ir_version = model_ptr->GetVersion();
  model_header.headsize = MODEL_FILE_HEAD_LEN;
  const std::string platform_version = model_ptr->GetPlatformVersion();
  errno_t
      err = memcpy_s(model_header.platform_version, static_cast<size_t>(PLATFORM_VERSION_LEN), platform_version.c_str(),
                     platform_version.size() + 1U);
  if (err != EOK) {
    GELOGE(FAILED, "[Save][Model]Failed for platform_version %s, model %s, errno %d",
           platform_version.c_str(), model_ptr->GetName().c_str(), err);
    REPORT_INNER_ERR_MSG("E19999", "Save model %s failed for platform_version %s, errno %d",
                      model_ptr->GetName().c_str(), platform_version.c_str(), err);
    return FAILED;
  }
  size_t name_size = model_ptr->GetName().size();
  name_size = (name_size > (MODEL_NAME_LENGTH - 1U)) ? (MODEL_NAME_LENGTH - 1U) : name_size;
  err = memcpy_s(model_header.name, MODEL_NAME_LENGTH, model_ptr->GetName().c_str(), name_size);
  if (err != EOK) {
    GELOGE(FAILED, "[Save][Model]Failed for memory copy %s failed, errno %d",
           model_ptr->GetName().c_str(), err);
    REPORT_INNER_ERR_MSG("E19999", "Save model failed for memory copy %s failed, errno %d",
                      model_ptr->GetName().c_str(), err);
    return FAILED;
  }
  ModelBufferData model;
  const Status ret = om_file_save_helper->SaveModel(output_file.c_str(), model, is_offline_ && save_to_file_);
  return ((ret == SUCCESS) ? SUCCESS : FAILED);
}

Status ModelHelper::SaveBundleModelBufferToMem(const std::vector<ModelBufferData> &model_buffers, uint64_t var_size,
                                               ModelBufferData &output_buffer) {
  const size_t model_num = model_buffers.size();
  constexpr size_t var_partition_num = 1U;
  constexpr size_t var_size_len = sizeof(uint64_t);
  const size_t mem_info_partition_num = model_num + var_partition_num; // add var partition
  const size_t header_size =
      sizeof(ModelFileHeader) + sizeof(ModelPartitionTable) + sizeof(ModelPartitionMemInfo) * mem_info_partition_num;
  size_t sub_model_size{0UL};
  for (size_t i = 0UL; i < model_num; ++i) {
    sub_model_size += model_buffers[i].length;
  }
  const size_t total_size = header_size + var_size_len + sub_model_size;
  auto buff_data = MakeUnique<uint8_t[]>(total_size);
  ModelFileHeader *header = PtrToPtr<uint8_t, ModelFileHeader>(buff_data.get());
  GE_ASSERT_NOTNULL(header);
  header->modeltype = MODEL_TYPE_BUNDLE_MODEL;
  header->model_num = static_cast<uint32_t>(model_num);
  header->model_length = total_size;
  ModelPartitionTable *table = PtrToPtr<uint8_t, ModelPartitionTable>(&buff_data[sizeof(ModelFileHeader)]);
  GE_ASSERT_NOTNULL(table);
  table->num = static_cast<uint32_t>(mem_info_partition_num);
  size_t offset{sizeof(ModelPartitionTable) + sizeof(ModelPartitionMemInfo) * mem_info_partition_num};
  // add BUNDLE_MODEL_VAR_INFO part
  table->partition[0].type = BUNDLE_MODEL_VAR_INFO;
  table->partition[0].mem_size = sizeof(int64_t);
  table->partition[0].mem_offset = offset;
  *reinterpret_cast<uint64_t*>(&buff_data[sizeof(ModelFileHeader) + offset]) = var_size;
  offset += var_size_len;
  // add sub model partition
  for (size_t i = 0U; i < model_num; ++i) {
    const size_t sub_model_id = i + var_partition_num;
    table->partition[sub_model_id].type = BUNDLE_MODEL_INFO;
    table->partition[sub_model_id].mem_size = model_buffers[i].length;
    table->partition[sub_model_id].mem_offset = offset;
    GE_ASSERT_SUCCESS(
        ge::GeMemcpy(&buff_data[sizeof(ModelFileHeader) + offset], sub_model_size,
        model_buffers[i].data.get(), model_buffers[i].length),
        "Copy model buffer failed.");
    sub_model_size -= model_buffers[i].length;
    offset += model_buffers[i].length;
  }

  // save bundle models.
  output_buffer.data.reset(buff_data.release(), std::default_delete<uint8_t[]>());
  output_buffer.length = total_size;
  GELOGI("Gathering bundle model successfully, total size: %[%zu], var_size is %lu", total_size, var_size);
  return SUCCESS;
}

Status ModelHelper::LoadModel(const ge::ModelData &model_data) {
  if ((model_data.model_data == nullptr) || (model_data.model_len == 0U)) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID,
           "[Load][Model]Model_data is nullptr or model_data_size is 0");
    REPORT_INNER_ERR_MSG("E19999", "Load model failed, "
                       "Model_data is nullptr or model_data_size is 0");
    return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID;
  }

  if (is_assign_model_) {
    GELOGE(ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED, "[Load][Model]Model helper has already loaded!");
    return ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED;
  }

  uint8_t *model_data_addr = nullptr;
  uint64_t model_data_size = 0UL;
  Status status = ModelParserBase::ParseModelContent(model_data, model_data_addr, model_data_size);
  if (status != SUCCESS) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Parse][ModelContent]Failed!");
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  file_header_ = reinterpret_cast<ModelFileHeader *>(model_data.model_data);
  OmFileLoadHelper om_load_helper;
  status = om_load_helper.Init(model_data_addr, model_data_size, file_header_);
  if (status != SUCCESS) {
    GELOGE(status, "[Init][OmLoadHelper]Failed");
    model_data_addr = nullptr;
    return status;
  }
  const auto partition_table = reinterpret_cast<ModelPartitionTable *>(model_data_addr);
  if (partition_table->num == kOriginalOmPartitionNum) {
    model_data_addr = nullptr;
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][OmModel]Error, please use executable om model");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  // Encrypt model need to del temp model/no encrypt model don't need to del model
  model_data_addr = nullptr;
  GeModelPtr first_ge_model = nullptr;
  status = GenerateGeModel(om_load_helper, model_, first_ge_model, 0U, false);
  if (status != SUCCESS) {
    GELOGE(status, "[Generate][GEModel]Failed");
    return status;
  }
  GELOGD("in ModelHelper::LoadModel, is_assign_model_ is setted to true!");
  is_assign_model_ = true;
  return SUCCESS;
}

Status ModelHelper::LoadPartInfoFromModel(const ge::ModelData &model_data, ModelPartition &partition) {
  if ((model_data.model_data == nullptr) || (model_data.model_len == 0U)) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID, "[Load][RootModel] "
           "Model_data is nullptr or model data is empty.");
    REPORT_INNER_ERR_MSG("E19999", "Load root model failed, model_data is nullptr or its size is 0");
    return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID;
  }

  GE_IF_BOOL_EXEC(is_assign_model_,
                  GELOGE(ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED, "[Load][RootModel]Model helper ha already loaded!");
                  return ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED);

  uint8_t *model_data_addr = nullptr;
  uint64_t model_data_size = 0U;
  Status status = ModelParserBase::ParseModelContent(model_data, model_data_addr, model_data_size);
  GE_IF_BOOL_EXEC(status != SUCCESS,
                  GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Parse][RootModelContent]Failed!");
                  return ACL_ERROR_GE_PARAM_INVALID);
  file_header_ = reinterpret_cast<ModelFileHeader *>(model_data.model_data);
  
  OmFileLoadHelper om_load_helper;
  status = om_load_helper.Init(model_data_addr, model_data_size, file_header_);
  GE_IF_BOOL_EXEC(status != SUCCESS,
                  GELOGE(status, "[Init][OmLoeadHelper]Failed");
                  model_data_addr = nullptr;
                  return status);
  // Encrypt model need to del temp model/no encrypt model don`t need to del model
  model_data_addr = nullptr;
  GE_IF_BOOL_EXEC(om_load_helper.GetModelPartition(partition.type, partition) != SUCCESS,
                  GELOGE(FAILED, "[Get][ModelPartition]Failed, partition type is %d", partition.type);
                  REPORT_INNER_ERR_MSG("E19999", "[Get][ModelPartition]Failed, partition type is %d", partition.type);
                  return FAILED);

  is_assign_model_ = true;
  return SUCCESS;
}


Status ModelHelper::GetModelFileHead(const ge::ModelData &model_data, const ModelFileHeader *&file_header) {
  if (model_data.model_data == nullptr) {
    (void)REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char*>({"parameter", "value", "reason"}),
                       std::vector<const char*>({"om", model_data.om_name.c_str(), "Model data cannot be nullptr."}));
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Param] Invalid model. Model data can not be nullptr.");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  if (model_data.model_len < sizeof(ModelFileHeader)) {
    std::string reason = "Invalid om file. The model data size " + std::to_string(model_data.model_len) +
                         " is smaller than " + std::to_string(sizeof(ModelFileHeader)) + ".";
    (void)REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                              std::vector<const char *>({"om", model_data.om_name.c_str(), reason.c_str()}));
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID,
           "[Check][Param] Invalid model. Model data size %" PRIu64 " must be greater than or equal to %zu.",
           model_data.model_len, sizeof(ModelFileHeader));
    return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID;
  }
  file_header = reinterpret_cast<const ModelFileHeader *>(model_data.model_data);
  return SUCCESS;
}

Status ModelHelper::GetOppVersion(std::string &version) {
  std::string opp_path;
  (void)PluginManager::GetOppPath(opp_path);
  const std::string version_path = opp_path + "/version.info";
  if (!PluginManager::GetVersionFromPathWithName(version_path, version, kOppVersion)) {
    GELOGW("Get opp version information from %s unsuccessful!", version_path.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status ModelHelper::CheckOsCpuInfoAndOppVersion() {
  if (file_header_->need_check_os_cpu_info == static_cast<uint8_t>(OsCpuInfoCheckTyep::NEED_CHECK)) {
    std::string host_env_os;
    std::string host_env_cpu;
    (void) ge::AttrUtils::GetStr(*(model_.get()), ATTR_MODEL_HOST_ENV_OS, host_env_os);
    (void) ge::AttrUtils::GetStr(*(model_.get()), ATTR_MODEL_HOST_ENV_CPU, host_env_cpu);
    std::string cur_host_env_os;
    std::string cur_host_env_cpu;
    ge::PluginManager::GetCurEnvPackageOsAndCpuType(cur_host_env_os, cur_host_env_cpu);
    if ((host_env_os.compare(cur_host_env_os) != 0) || (host_env_cpu.compare(cur_host_env_cpu) != 0)) {
      REPORT_INNER_ERR_MSG("E19999", "The os/cpu type of the model does not match the current system,"
                         "Model is[%s][%s], system is[%s][%s], please use the matching platform",
                         host_env_os.c_str(), host_env_cpu.c_str(),
                         cur_host_env_os.c_str(), cur_host_env_cpu.c_str());
      GELOGE(FAILED, "The os/cpu type of the model does not match the current system,"
             "Model is[%s][%s], system is[%s][%s]",
             host_env_os.c_str(), host_env_cpu.c_str(),
             cur_host_env_os.c_str(), cur_host_env_cpu.c_str());
      return FAILED;
    }
    GELOGD("Check os[%s], cpu[%s] success.", host_env_os.c_str(), host_env_cpu.c_str());

    const auto &so_in_om_info = root_model_->GetSoInOmInfo();
    GE_ASSERT_TRUE(PluginManager::IsVendorVersionValid(so_in_om_info.opp_version, so_in_om_info.compiler_version),
                   "Invalid opp version [%s] or compiler_version [%s],"
                   "Please check if it is within the required range",
                   so_in_om_info.opp_version.c_str(), so_in_om_info.compiler_version.c_str());
    GELOGD("Check opp_version[%s], compiler_version[%s] success.", so_in_om_info.opp_version.c_str(),
           so_in_om_info.compiler_version.c_str());
  }

  if ((file_header_->need_check_os_cpu_info == static_cast<uint8_t>(OsCpuInfoCheckTyep::NO_CHECK))
      && is_unknown_shape_model_) {
    std::string version;
    (void) ge::AttrUtils::GetStr(*(model_.get()), ATTR_MODEL_OPP_VERSION, version);
    std::string cur_version;
    // opp_kernel独立升级之后，单算子动态shape离线模型记录的opp_version和算子实际选择的不同，不做强校验
    if ((GetOppVersion(cur_version) == SUCCESS) && (version.compare(cur_version) != 0)) {
      REPORT_INNER_ERR_MSG("E19999", "The opp version of the model does not match the current opp run package,"
                         "Model is[%s], opp run package is[%s], try to convert the om again!",
                         version.c_str(), cur_version.c_str());
      GELOGE(FAILED, "The opp version of the model does not match the current opp run package,"
             "Model is[%s], opp run package is[%s]", version.c_str(), cur_version.c_str());
      return FAILED;
    }
    GELOGD("Check opp version[%s] success.", version.c_str());
  }
  return SUCCESS;
}

Status ModelHelper::CheckIfWeightPathValid(const ge::ComputeGraphPtr &graph,
                                           const ge::ModelData &model_data) const {
  size_t index = 0UL;
  GE_ASSERT_NOTNULL(graph);
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == FILECONSTANT) {
      index++;
    }
  }
  if ((index == 0UL) && (!model_data.weight_path.empty())) {
    GELOGE(FAILED, "Weight path[%s] should be empty if model has no external weight",
        model_data.weight_path.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status ModelHelper::LoadRootModel(const ge::ModelData &model_data) {
  if (is_assign_model_) {
    GELOGE(ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED, "[Load][RootModel]Model helper has already loaded!");
    return ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED;
  }
  GE_CHK_STATUS_RET(GetModelFileHead(model_data, file_header_), "[Get][ModelFileHead] failed.");
  // check file header
  GE_ASSERT_TRUE(file_header_->modeltype != static_cast<uint8_t>(MODEL_TYPE_BUNDLE_MODEL),
                 "The bundle model does not support loading through non-bundle interfaces.");
  // model verison 1.0 file header does not have model_num member
  is_unknown_shape_model_ = ModelParserBase::IsDynamicModel(*file_header_);
  GELOGD("Cur om model is ge root model or no %d, model version %u, model num: %u.",
         static_cast<int32_t>(is_unknown_shape_model_), file_header_->version, file_header_->model_num);
  uint8_t *model_data_addr = nullptr;
  uint64_t model_data_size = 0UL;
  Status status = ModelParserBase::ParseModelContent(model_data, model_data_addr, model_data_size);
  if (status != SUCCESS) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Parse][RootModelContent]Failed!");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  OmFileLoadHelper om_load_helper;
  if (is_unknown_shape_model_) {
    status = om_load_helper.Init(model_data_addr, model_data_size, file_header_->model_num, file_header_);
  } else {
    status = om_load_helper.Init(model_data_addr, model_data_size, file_header_);
  }
  if (status != SUCCESS) {
    GELOGE(status, "[Init][OmLoadHelper]Failed");
    model_data_addr = nullptr;
    return status;
  }
  // Encrypt model need to del temp model/no encrypt model don't need to del model
  model_data_addr = nullptr;

  GE_ASSERT_SUCCESS(GenerateGeRootModel(om_load_helper, model_data));
  std::string file_constant_weight_dir;
  GE_ASSERT_SUCCESS(CheckIfWeightPathValid(root_model_->GetRootGraph(), model_data));
  GE_ASSERT_SUCCESS(FileConstantUtils::GetExternalWeightDir(model_data, file_constant_weight_dir));
  root_model_->SetFileConstantWeightDir(file_constant_weight_dir);

  GELOGD("In ModelHelper::LoadRootModel, is_assign_model_ is setted to true!");
  is_assign_model_ = true;

  if (is_repack_so_) {
    GELOGD("ModelHelper::repack so!");
    return SUCCESS;
  }

  if (CheckOsCpuInfoAndOppVersion() != SUCCESS) {
    GELOGE(FAILED, "[Check][OsCpuOppVersion]Failed");
    return FAILED;
  }
  return SUCCESS;
}

Status ModelHelper::GenerateGeModel(const OmFileLoadHelper &om_load_helper, GeModelPtr &cur_model,
                                    GeModelPtr &first_ge_model, const size_t mode_index,
                                    const bool is_dyn_root) const {
  cur_model = MakeShared<GeModel>();
  GE_CHECK_NOTNULL(cur_model);
  Status ret = LoadModelData(om_load_helper, cur_model, first_ge_model, mode_index);
  if (ret != SUCCESS) {
    return ACL_ERROR_GE_EXEC_LOAD_MODEL_PARTITION_FAILED;
  }

  // shape partitioned scene no need to load root_graph ge_model.
  if (is_dyn_root && IsPartitionedGraph(cur_model)) {
    first_ge_model = MakeShared<GeModel>();
    GE_CHECK_NOTNULL(first_ge_model);
    first_ge_model->SetGraph(cur_model->GetGraph());
    first_ge_model->SetName(cur_model->GetName());
    first_ge_model->SetVersion(cur_model->GetVersion());
    first_ge_model->SetPlatformVersion(cur_model->GetPlatformVersion());
    first_ge_model->SetAttrMap(cur_model->MutableAttrMap());
    if (gert::GraphUnfolder::IsGraphNeedUnfold(cur_model->GetGraph())) {
      return SUCCESS;
    }
  }

  ret = LoadWeights(om_load_helper, cur_model, mode_index);
  if (ret != SUCCESS) {
    return ACL_ERROR_GE_EXEC_LOAD_WEIGHT_PARTITION_FAILED;
  }
  ret = LoadTask(om_load_helper, cur_model, mode_index);
  if (ret != SUCCESS) {
    return ACL_ERROR_GE_EXEC_LOAD_TASK_PARTITION_FAILED;
  }
  ret = LoadTBEKernelStore(om_load_helper, cur_model, mode_index);
  if (ret != SUCCESS) {
    return ACL_ERROR_GE_EXEC_LOAD_KERNEL_PARTITION_FAILED;
  }
  ret = LoadCustAICPUKernelStore(om_load_helper, cur_model, mode_index);
  if (ret != SUCCESS) {
    return ACL_ERROR_GE_EXEC_LOAD_KERNEL_PARTITION_FAILED;
  }
  return SUCCESS;
}

Status ModelHelper::GenerateGeRootModel(const OmFileLoadHelper &om_load_helper, const ModelData &model_data) {
  GELOGD("Begin to generate ge root model");
  root_model_ = MakeShared<GeRootModel>();
  GE_CHECK_NOTNULL(root_model_);
  GeModelPtr first_ge_model = nullptr;
  if (!is_unknown_shape_model_) {
    GE_CHK_STATUS_RET(GenerateGeModel(om_load_helper, model_, first_ge_model, 0U, false),
                      "[Generate][GERootModel]Failed");
    GE_CHECK_NOTNULL(model_);
    GE_ASSERT_SUCCESS(root_model_->Initialize(model_->GetGraph()));
    root_model_->SetModelName(model_->GetName());
    GE_CHK_STATUS_RET(LoadOpSoBin(om_load_helper, root_model_), "[Generate][LoadOpSoBin]Failed");
    GE_CHK_STATUS_RET(LoadTilingData(om_load_helper, root_model_), "[Generate][LoadTilingData]Failed");
    root_model_->SetSubgraphInstanceNameToModel(model_->GetGraph()->GetName(), model_);
    return SUCCESS;
  }
  for (size_t mode_index = 0U;  mode_index < file_header_->model_num; ++mode_index) {
    GeModelPtr cur_model;
    GE_CHK_STATUS_RET_NOLOG(GenerateGeModel(om_load_helper, cur_model, first_ge_model, mode_index, mode_index == 0U));
    GE_ASSERT_NOTNULL(cur_model->GetGraph());
    if (mode_index == 0U) {
      GE_ASSERT_SUCCESS(root_model_->Initialize(cur_model->GetGraph()));
      root_model_->SetModelName(cur_model->GetName());
      model_ = cur_model;
      GE_CHK_STATUS_RET(LoadOpSoBin(om_load_helper, root_model_), "[Generate][LoadOpSoBin]Failed");
      GE_CHK_STATUS_RET(LoadTilingData(om_load_helper, root_model_), "[Generate][LoadTilingData]Failed");
      if (IsPartitionedGraph(cur_model)) {
        if (!gert::GraphUnfolder::IsGraphNeedUnfold(cur_model->GetGraph())) {
          root_model_->SetSubgraphInstanceNameToModel(cur_model->GetGraph()->GetName(), cur_model);
        }
        continue;
      }
    }
    cur_model->SetOmName(model_data.om_name);
    root_model_->SetSubgraphInstanceNameToModel(cur_model->GetGraph()->GetName(), cur_model);
  }
  GE_ASSERT_SUCCESS(root_model_->ModifyOwnerGraphForSubModels());
  return SUCCESS;
}

bool ModelHelper::IsPartitionedGraph(const GeModelPtr &cur_model) const {
  // shape partitioned scene no need to load root_graph ge_model.
  bool is_partitioned = false;
  const auto &root_graph = cur_model->GetGraph();
  (void)AttrUtils::GetBool(root_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_partitioned);
  return is_partitioned;
}

Status ModelHelper::SetModelToGeModel(const GeModelPtr &ge_model, const GeModelPtr &first_ge_model, Model &model) {
  ge_model->SetGraph(model.GetGraph());
  ge_model->SetName(model.GetName());
  ge_model->SetVersion(model.GetVersion());
  ge_model->SetPlatformVersion(model.GetPlatformVersion());
  ge_model->SetAttrMap(model.MutableAttrMap());
  if (first_ge_model != nullptr) {
    // 对于动态shape，有partiton场景，字典统一存储在根图的属性中
    // 子图解压时需要先获取根图的字典
    GE_ASSERT_SUCCESS(ModelCompressManager::CpyModelAttrs2Dst(first_ge_model, ge_model));
  }
  GE_ASSERT_SUCCESS(ModelCompressManager::Decompress(ge_model));
  if (first_ge_model != nullptr) {
    // 子图解压后删除子图字典
    ModelCompressManager::DeleteModelAttrs(ge_model);
  }
  return SUCCESS;
}

Status ModelHelper::LoadModelData(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                                  const GeModelPtr &first_ge_model, const size_t mode_index) const {
  ModelPartition partition_model_def;
  // no need to check value, DATA->NetOutput
  (void) om_load_helper.GetModelPartition(ModelPartitionType::MODEL_DEF, partition_model_def, mode_index);
  GELOGD("Model_def partition addr:%p,size:%" PRIu64 "", partition_model_def.data, partition_model_def.size);

  ge::Model model;
  if (ge::Model::LoadWithMultiThread(partition_model_def.data,
      static_cast<size_t>(partition_model_def.size), model) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Load][Model]Failed, model_def partition addr:%p, size:%" PRIu64 "",
           partition_model_def.data, partition_model_def.size);
    REPORT_INNER_ERR_MSG("E19999", "Load model failed, model_def partition addr:%p, size:%" PRIu64 "",
                      partition_model_def.data, partition_model_def.size);
    return INTERNAL_ERROR;
  }

  // only root model has soc_version/arch_type infos
  if ((mode_index == 0U) && !is_repack_so_) {
    GE_ASSERT_SUCCESS(om_load_helper.CheckModelCompatibility(model), "Check model compatibility failed.");
  }
  GE_ASSERT_SUCCESS(SetModelToGeModel(cur_model, first_ge_model, model));
  return SUCCESS;
}

Status ModelHelper::LoadWeights(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                                const size_t mode_index) const {
  ModelPartition partition;
  if (om_load_helper.GetModelPartition(ModelPartitionType::WEIGHTS_DATA, partition, mode_index) != SUCCESS) {
    GELOGE(FAILED, "[Get][ModelPartition]Failed, GetWeight size:%" PRIu64 "", partition.size);
    REPORT_INNER_ERR_MSG("E19999", "[Get][ModelPartition]Failed, GetWeight size:%" PRIu64 "",
                      partition.size);
    return FAILED;
  }
  if (is_shared_weight_) {
    GELOGD("current weight is shared, size:%" PRIu64 "", partition.size);
    DataBuffer weight_buf(const_cast<uint8_t *>(partition.data), static_cast<size_t>(partition.size));
    cur_model->SetWeightDataBuf(weight_buf);
  } else {
    const ge::Buffer weight = ge::Buffer::CopyFrom(partition.data, static_cast<size_t>(partition.size));
    cur_model->SetWeight(weight);
  }
  GELOGD("GetWeight size:%" PRIu64 "", partition.size);
  return SUCCESS;
}

Status ModelHelper::LoadTask(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                             const size_t mode_index) const {
  ModelPartition task_partition;
  if (om_load_helper.GetModelPartition(ModelPartitionType::TASK_INFO, task_partition, mode_index) != SUCCESS) {
    GELOGE(FAILED, "Get task model partition failed.");
    GELOGE(FAILED, "[Get][ModelTaskPartition]Failed, task_partition size %" PRIu64 ", mode_index %zu",
           task_partition.size, mode_index);
    REPORT_INNER_ERR_MSG("E19999", "Get model task partition failed, "
                       "task_partition size %" PRIu64 ", mode_index %zu", task_partition.size, mode_index);
    return FAILED;
  }
  const std::shared_ptr<domi::ModelTaskDef> task = ge::MakeShared<domi::ModelTaskDef>();
  GE_CHECK_NOTNULL(task);
  if (task_partition.size != 0U) {
    if (!ReadProtoFromArray(task_partition.data, static_cast<int32_t>(task_partition.size), task.get())) {
      GELOGE(INTERNAL_ERROR, "[Read][ProtoFromArray]Failed, task_partition size %" PRIu64 "",
             task_partition.size);
      REPORT_INNER_ERR_MSG("E19999", "Read proto from array failed, task_partition size %" PRIu64 "",
                        task_partition.size);
      return INTERNAL_ERROR;
    }
    GELOGD("TASK_INFO op_size:%d, stream_num:%u", task->op().size(), task->stream_num());
  }
  cur_model->SetModelTaskDef(task);
  return SUCCESS;
}

Status ModelHelper::LoadTBEKernelStore(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                                       const size_t mode_index) const {
  // Load tbe kernels
  ModelPartition partition_kernel_def;
  if (om_load_helper.GetModelPartition(ModelPartitionType::TBE_KERNELS, partition_kernel_def, mode_index) == SUCCESS) {
    GELOGD("Kernels partition size:%" PRIu64 "", partition_kernel_def.size);
    if (cur_model->LoadTBEKernelStore(partition_kernel_def.data, static_cast<size_t>(partition_kernel_def.size))) {
      GELOGD("Load tbe kernels success");
    } else {
      GELOGW("Load tbe kernels unsuccessful");
    }
  }
  return SUCCESS;
}

Status ModelHelper::LoadCustAICPUKernelStore(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                                             const size_t mode_index) const {
  // Load cust aicpu kernels
  ModelPartition partition_kernel_def;
  if (om_load_helper.GetModelPartition(ModelPartitionType::CUST_AICPU_KERNELS, partition_kernel_def, mode_index)
      == SUCCESS) {
    GELOGD("Kernels partition size:%" PRIu64 "", partition_kernel_def.size);
    if (cur_model->LoadAICPUKernelStore(partition_kernel_def.data, partition_kernel_def.size)) {
      GELOGD("Load cust aicpu kernels success");
    } else {
      GELOGW("Load cust aicpu kernels unsuccessful");
    }
  }
  return SUCCESS;
}

Status ModelHelper::LoadOpSoBin(const OmFileLoadHelper &om_load_helper,
                                const GeRootModelPtr &ge_root_model) const {
  ModelPartition partition_kernel_def;
  if (om_load_helper.GetModelPartition(ModelPartitionType::SO_BINS, partition_kernel_def, 0U)
      == SUCCESS) {
    GELOGD("Kernels partition size:%" PRIu64 "", partition_kernel_def.size);
    if (ge_root_model->LoadSoBinData(partition_kernel_def.data, partition_kernel_def.size)) {
      // 取出AutofuseSo并存放到扩展属性
      auto root_graph = ge_root_model->GetRootGraph();
      GE_ASSERT_NOTNULL(root_graph);
      std::map<std::string, ge::OpSoBinPtr> bin_file_buffer;
      auto all_so_bin = ge_root_model->GetAllSoBin();
      auto new_end = std::remove_if(all_so_bin.begin(), all_so_bin.end(),
        [&bin_file_buffer, &root_graph](const OpSoBinPtr& op_so_bin_ptr) {
          if (op_so_bin_ptr != nullptr && op_so_bin_ptr->GetSoBinType() == SoBinType::kAutofuse) {
            std::string so_path = op_so_bin_ptr->GetVendorName() + "/" + op_so_bin_ptr->GetSoName();
            bin_file_buffer[so_path] = op_so_bin_ptr;
            GELOGD("Added autofuse so_path:%s", so_path.c_str());
            root_graph->SetExtAttr<std::map<std::string, ge::OpSoBinPtr>>("bin_file_buffer", bin_file_buffer);
            return true;
          }
          return false;
      });
      (void)all_so_bin.erase(new_end, all_so_bin.end()); // 防止后续缓存落盘时重复保存AutofuseSo
      SaveOpSoInfo(ge_root_model);
      GELOGD("Load so bin store success");
    } else {
      GELOGW("Load so bin store unsuccessful");
    }
  }
  return SUCCESS;
}

Status ModelHelper::LoadTilingData(const OmFileLoadHelper &om_load_helper, const GeRootModelPtr &ge_root_model) const {
  ModelPartition partition_kernel_def;
  (void)om_load_helper.GetModelPartition(ModelPartitionType::TILING_DATA, partition_kernel_def, 0U);
  GELOGI("Tiling data partition size:%" PRIu64 "", partition_kernel_def.size);
  if (partition_kernel_def.size > 0UL) {
    GE_ASSERT_NOTNULL(ge_root_model->GetRootGraph());
    const HostResourceCenterPtr &host_resource_center = ge_root_model->GetHostResourceCenterPtr();
    GE_ASSERT_NOTNULL(host_resource_center);
    GE_ASSERT_SUCCESS(HostResourceSerializer::DeSerializeTilingData(*host_resource_center, partition_kernel_def.data,
                                                                    partition_kernel_def.size));
    GE_ASSERT_SUCCESS(
        HostResourceSerializer::RecoverOpRunInfoToExtAttrs(*host_resource_center, ge_root_model->GetRootGraph()));
  }
  return SUCCESS;
}

void ModelHelper::SaveOpSoInfo(const GeRootModelPtr &ge_root_model) const {
  SoInOmInfo so_info;
  (void) ge::AttrUtils::GetStr(*(model_.get()), "host_env_os", so_info.os_info);
  (void) ge::AttrUtils::GetStr(*(model_.get()), "host_env_cpu", so_info.cpu_info);
  (void) ge::AttrUtils::GetStr(*(model_.get()), ATTR_MODEL_OPP_VERSION, so_info.opp_version);
  (void) ge::AttrUtils::GetStr(*(model_.get()), ATTR_MODEL_COMPILER_VERSION, so_info.compiler_version);
  GELOGD("Save so info with host_env_os:%s, host_env_cpu:%s, opp_version:%s, compiler_version:%s",
         so_info.os_info.c_str(), so_info.cpu_info.c_str(), so_info.opp_version.c_str(),
         so_info.compiler_version.c_str());
  ge_root_model->SetSoInOmInfo(so_info);
}

GeModelPtr ModelHelper::GetGeModel() {
  if (model_ != nullptr) {
    return model_;
  }

  GELOGD("Model has not been loaded!");
  const std::shared_ptr<ge::GeModel> out_model = ge::MakeShared<ge::GeModel>();
  if (out_model == nullptr) {
    return nullptr;
  }
  return out_model;
}

GeRootModelPtr ModelHelper::GetGeRootModel() {
  if (root_model_ != nullptr) {
    return root_model_;
  }

  GELOGD("Model has not been loaded!");
  const std::shared_ptr<ge::GeRootModel> out_model = ge::MakeShared<ge::GeRootModel>();
  if (out_model == nullptr) {
    return nullptr;
  }

  if (model_ != nullptr) {
    const auto root_graph = model_->GetGraph();
    if (root_graph != nullptr) {
      GE_ASSERT_SUCCESS(out_model->Initialize(root_graph));
      out_model->SetModelName(model_->GetName());
      out_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), model_);
    }
  }
  return out_model;
}

Status ModelHelper::GetBaseNameFromFileName(const std::string &file_name, std::string &base_name) const {
  GELOGD("Get base_name from file, file_name:%s", file_name.c_str());
  if (file_name.empty()) {
    GELOGW("File path may not valid, check params --output");
    return FAILED;
  }
  size_t start_position = 0U;
  // using output as base_name (ignore ".om" or ".exeom")
  size_t filename_suffixes = 0U;
  if (file_name.find_last_of('.') != std::string::npos) {
    filename_suffixes = file_name.length() - file_name.find_last_of('.');
  }
  if (file_name.find_last_of('/') != std::string::npos) {
    start_position = file_name.find_last_of('/') + 1U;
  }
  const size_t end_position = file_name.length() - filename_suffixes;
  base_name = file_name.substr(start_position, end_position - start_position);
  if (base_name.empty()) {
    GELOGW("Get base_name unsuccessful, check params --output");
    return FAILED;
  }
  GELOGD("Get base_name from file success, base_name:%s", base_name.c_str());
  return SUCCESS;
}

Status ModelHelper::GetHardwareInfo(std::map<std::string, std::string> &options) const {
  int32_t device_id = -1;
  (void)aclrtGetDevice(&device_id);

  const auto iter = options.find(SOC_VERSION);
  GE_ASSERT_TRUE(iter != options.end());
  const auto soc_version = iter->second;

  fe::PlatformInfo platform_info;
  int32_t virtual_type = 0;
  GE_CHK_STATUS_RET(GetPlatformInfo(device_id, soc_version, platform_info, virtual_type, options),
                    "Get platform info failed, device id: %d, soc_version: %s.", device_id, soc_version.c_str());

  fe::PlatFormInfos platform_infos;
  GE_CHK_STATUS_RET(SetPlatformInfos(soc_version, platform_info, platform_infos), "Set platform infos failed.");

  std::stringstream platform_option;
  platform_option << "ai_core_cnt:" << std::to_string(platform_info.soc_info.ai_core_cnt)
                  << ";cube_core_cnt:" << std::to_string(platform_info.soc_info.ai_core_cnt)
                  << ";vector_core_cnt:" << std::to_string(platform_info.soc_info.vector_core_cnt)
                  << ";l2_size:" << std::to_string(platform_info.soc_info.l2_size)
                  << ";memory_size:" << std::to_string(platform_info.soc_info.memory_size);
  (void)options.emplace(std::make_pair(kHardwareInfo, platform_option.str()));
  GELOGI("Soc_version: %s, set attr %s, value: %s.", soc_version.c_str(), kHardwareInfo.c_str(), platform_option.str().c_str());

  if (virtual_type > 0) {
    (void)options.emplace(std::make_pair(VIRTUAL_TYPE, std::to_string(virtual_type)));
    GELOGI("Set attr %s, value: %d.", VIRTUAL_TYPE.c_str(), virtual_type);
  }

  return SUCCESS;
}

Status ModelHelper::InitRuntimePlatform() {
  int32_t device_id = -1;
  GE_CHK_RT_RET(aclrtGetDevice(&device_id));
  // init platform info
  const char *soc_version = aclrtGetSocName();
  GE_ASSERT_NOTNULL(soc_version);
  GE_ASSERT_TRUE(fe::PlatformInfoManager::GeInstance().InitRuntimePlatformInfos(std::string(soc_version)) == 0U,
      "[Init][PlatformInfo]init runtime platform info failed, SocVersion = %s", soc_version);

  uint32_t aicore_num = 0U;
  GE_ASSERT_RT_OK(rtGetAiCoreCount(&aicore_num));
  int64_t vec_core_num = 0U;
  // some chips has no vector core
  GE_ASSERT_RT_OK(aclrtGetDeviceInfo(static_cast<uint32_t>(device_id),
 	                                   ACL_DEV_ATTR_VECTOR_CORE_NUM, &vec_core_num));

  fe::PlatFormInfos platform_infos;
  GE_ASSERT_TRUE(
      fe::PlatformInfoManager::GeInstance().GetRuntimePlatformInfosByDevice(device_id, platform_infos) == 0,
      "Get runtime platformInfos by device failed, deviceId = %d", device_id);

  std::map<std::string, std::string> res;
  GE_ASSERT_TRUE(platform_infos.GetPlatformResWithLock(kSocInfoKey, res));

  res[kAicCntKey] = std::to_string(aicore_num);
  res[kVecCoreCntKey] = std::to_string(vec_core_num);
  platform_infos.SetPlatformResWithLock(kSocInfoKey, res);
  GE_ASSERT_TRUE(
      fe::PlatformInfoManager::GeInstance().UpdateRuntimePlatformInfosByDevice(device_id, platform_infos) == 0U,
      "Update runtime platformInfos by device failed, deviceId = %d", device_id);
  return SUCCESS;
}

Status ModelHelper::InitRuntimeAndGetDevicePlatformInfos(int32_t device_id, const std::string &soc_version,fe::PlatFormInfos &platform_infos_device) const {
  // 初始化 runtime platform info
  GE_ASSERT_TRUE(fe::PlatformInfoManager::GeInstance().InitRuntimePlatformInfos(soc_version) == 0U,
      "[Init][PlatformInfo]init runtime platform info failed, SocVersion = %s", soc_version.c_str());

  // 获取指定 device 的 platform info
  GE_ASSERT_TRUE(fe::PlatformInfoManager::GeInstance().GetRuntimePlatformInfosByDevice(static_cast<uint32_t>(device_id), platform_infos_device, true) == 0,
     "Get runtime platformInfos by device failed, deviceId = %d", device_id);

  return SUCCESS;
}

Status ModelHelper::HandleDeviceInfo(fe::PlatFormInfos &platform_infos) const {
  fe::PlatformInfo platform_info;
  return HandleDeviceInfo(platform_infos, platform_info);
}

Status ModelHelper::HandleDeviceInfo(fe::PlatFormInfos &platform_infos, fe::PlatformInfo &origin_platform_info) const {
  GELOGD("Begin to handle device info.");
  int32_t device_id = -1;
  GE_CHK_RT_RET(aclrtGetDevice(&device_id));

  const char *soc_version = aclrtGetSocName();
  GE_ASSERT_NOTNULL(soc_version);

  GE_ASSERT_SUCCESS(CoreNumUtils::GetGeDefaultPlatformInfo(soc_version, origin_platform_info));

  fe::PlatformInfo platform_info;
  int32_t virtual_type = 0;
  GE_CHK_STATUS_RET(GetPlatformInfo(device_id, soc_version, platform_info, virtual_type), "Get platform info failed.");

  GE_CHK_STATUS_RET(SetPlatformInfos(soc_version, platform_info, platform_infos), "Set platform infos failed.");

  GELOGD("Succeed to handle device info, device id: %d, soc_version: %s, virtual_type: %d.", device_id, soc_version,
         virtual_type);
  return SUCCESS;
}

Status ModelHelper::GetPlatformInfo(int32_t device_id, const std::string &soc_version,
                                    fe::PlatformInfo &platform_info, int32_t &virtual_type) const {
  std::map<std::string, std::string> options;
  return GetPlatformInfo(device_id, soc_version, platform_info, virtual_type, options);
}

Status ModelHelper::GetPlatformInfo(int32_t device_id, const std::string &soc_version,
                                    fe::PlatformInfo &platform_info, int32_t &virtual_type,
                                    std::map<std::string, std::string> &options) const {
#ifdef __GNUC__
  GE_ASSERT_SUCCESS(CoreNumUtils::GetGeDefaultPlatformInfo(soc_version, platform_info));
#endif

  const uint32_t aicore_cnt_ini = platform_info.soc_info.ai_core_cnt;
  const uint32_t vec_core_cnt_ini = platform_info.soc_info.vector_core_cnt;

  fe::PlatFormInfos platformInfos_device;
  GE_CHK_STATUS_RET(InitRuntimeAndGetDevicePlatformInfos(device_id, soc_version, platformInfos_device));

  GE_CHK_STATUS_RET(UpdatePlatfromInfoWithRuntime(device_id, aicore_cnt_ini, vec_core_cnt_ini, platform_info,
    virtual_type));

  GE_CHK_STATUS_RET(UpdatePlatfromInfoWithDevice(platformInfos_device, aicore_cnt_ini, vec_core_cnt_ini, platform_info));

  GE_CHK_STATUS_RET(UpdatePlatfromInfoWithOption(options, aicore_cnt_ini, vec_core_cnt_ini, platform_info));

  GELOGI("Get platform info of device id: %d, aicore num: %u, vector core num: %u, l2size: %u bytes, memory_size: "
         "%zu bytes, virtual_type: %d.", device_id, platform_info.soc_info.ai_core_cnt, platform_info.soc_info.vector_core_cnt,
         platform_info.soc_info.l2_size, platform_info.soc_info.memory_size, virtual_type);

  return SUCCESS;
}

Status ModelHelper::UpdatePlatfromInfoWithOption(std::map<std::string, std::string> &options, const uint32_t ai_core_cnt_ini,
  const uint32_t vector_core_cnt_ini, fe::PlatformInfo &platform_info) const {
  // 用从option/context获取到的核数刷新platform info
  GE_CHK_STATUS_RET(UpdateCoreCountWithOption(AICORE_NUM, AICORE_NUM, ai_core_cnt_ini,
    platform_info.soc_info.ai_core_cnt, options));
  GE_CHK_STATUS_RET(UpdateCoreCountWithOption(kVectorCoreNum, kVectorCoreNum, vector_core_cnt_ini,
    platform_info.soc_info.vector_core_cnt, options));
  return SUCCESS;
}

Status ModelHelper::UpdatePlatfromInfoWithDevice(fe::PlatFormInfos &platformInfos_device, const uint32_t ai_core_cnt_ini,
  const uint32_t vector_core_cnt_ini, fe::PlatformInfo &platform_info) const {
  std::map<std::string, std::string> res;

  (void)platformInfos_device.GetPlatformResWithLock(kSocInfoKey, res);
  std::string aicore_num_device = res[kAicCntKey];
  std::string vec_core_num_device = res[kVecCoreCntKey];
  GELOGI("Get platform info from device, aicore_num: %s, vector_core_num: %s.",
         aicore_num_device.c_str(), vec_core_num_device.c_str());

  GE_CHK_STATUS_RET(UpdateCoreCountWithDevice(kEnumNameCubeNum, ai_core_cnt_ini, aicore_num_device,
    platform_info.soc_info.ai_core_cnt));
  GE_CHK_STATUS_RET(UpdateCoreCountWithDevice(kEnumNameVectorNum, vector_core_cnt_ini, vec_core_num_device,
    platform_info.soc_info.vector_core_cnt));
  return SUCCESS;
}

Status ModelHelper::UpdatePlatfromInfoWithRuntime(const int32_t device_id, const uint32_t ai_core_cnt_ini, const uint32_t vector_core_cnt_ini,
  fe::PlatformInfo &platform_info, int32_t &virtual_type) const {
  if (device_id < 0) {
    GELOGI("Offline scene, skip update platform info from runtime.");
    return SUCCESS;
  }
  int64_t aic_core_cnt = 0;
  if (aclrtGetDeviceInfo(static_cast<uint32_t>(device_id),
      ACL_DEV_ATTR_AICORE_CORE_NUM, &aic_core_cnt) != ACL_SUCCESS) {
    GELOGE(FAILED, "Failed to get AICore count from device.");
    return FAILED;
  }

  if (static_cast<uint32_t>(aic_core_cnt) < ai_core_cnt_ini) {
    virtual_type = 1;
  }

  int64_t vector_core_cnt = kModuleTypeVectorCore;
  // some chips have no vector core
  (void)aclrtGetDeviceInfo(static_cast<uint32_t>(device_id), ACL_DEV_ATTR_VECTOR_CORE_NUM, &vector_core_cnt);

  // 用从rts获取到的核数刷新platform info
  UpdateCoreCountWithRuntime(AICORE_NUM, ai_core_cnt_ini, aic_core_cnt,
    platform_info.soc_info.ai_core_cnt);
  UpdateCoreCountWithRuntime(kVectorCoreNum, vector_core_cnt_ini, vector_core_cnt,
    platform_info.soc_info.vector_core_cnt);

   size_t free_mem = 0U;
   size_t total_mem_size = 0U;
   if (aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem_size) == ACL_SUCCESS) {
     GELOGI("Change memory_size from platform %lu to rts %zu bytes.",
            platform_info.soc_info.memory_size, total_mem_size);
     platform_info.soc_info.memory_size = total_mem_size;
   }

  return SUCCESS;
}

Status ModelHelper::SetPlatformInfos(const std::string &soc_version, const fe::PlatformInfo &platform_info,
                                     fe::PlatFormInfos &platform_infos) const {
#ifdef __GNUC__
  fe::OptionalInfos optional_infos;
  if (fe::PlatformInfoManager::GeInstance().GetPlatformInfos(soc_version, platform_infos, optional_infos) != 0) {
    GELOGE(FAILED, "Unable to get platform info of soc_version: %s.", soc_version.c_str());
    return FAILED;
  }

  std::map<std::string, std::string> res;
  if (!platform_infos.GetPlatformResWithLock("SoCInfo", res)) {
    GELOGE(FAILED, "Unable to get platform info.");
    return FAILED;
  }

  res["ai_core_cnt"] = std::to_string(platform_info.soc_info.ai_core_cnt);
  res["cube_core_cnt"] = std::to_string(platform_info.soc_info.ai_core_cnt);
  res["vector_core_cnt"] = std::to_string(platform_info.soc_info.vector_core_cnt);
  res["l2_size"] = std::to_string(platform_info.soc_info.l2_size);
  res["memory_size"] = std::to_string(platform_info.soc_info.memory_size);

  GELOGD("Begin to update platform infos, aicore_cnt: %d, vector_core_cnt: %d, cube_core_cnt: %s, l2_size: %d, memory_size: %d",
         platform_info.soc_info.ai_core_cnt, platform_info.soc_info.vector_core_cnt, res["cube_core_cnt"].c_str(),
         platform_info.soc_info.l2_size, platform_info.soc_info.memory_size);

  platform_infos.SetPlatformResWithLock("SoCInfo", res);

  GE_ASSERT_TRUE(fe::PlatformInfoManager::GeInstance().UpdatePlatformInfos(soc_version, platform_infos) == 0U, "Update platform infos of GeInstance failed.");

  GE_ASSERT_TRUE(fe::PlatformInfoManager::Instance().InitializePlatformInfo() == 0U, "Initialize platform info of Instance failed.");
  GE_ASSERT_TRUE(fe::PlatformInfoManager::Instance().UpdatePlatformInfos(soc_version, platform_infos) == 0U, "Update platform infos of Instance failed.");
#endif
  return SUCCESS;
}

void ModelHelper::SetRepackSoFlag(const bool val) {
  is_repack_so_ = val;
}

Status ModelHelper::PackSoToModelData(const ModelData &model_data, const std::string &output_file,
                                      ModelBufferData &model_buffer, const bool save_to_file) {
  GE_ASSERT_NOTNULL(root_model_);
  const auto &compute_graph = root_model_->GetRootGraph();
  GE_CHECK_NOTNULL(compute_graph);
  GE_CHK_STATUS_RET(FileConstantUtils::ChangeFilePath(compute_graph, output_file),
                    "Failed to change file path, graph name:%s", compute_graph->GetName().c_str());
  save_to_file_ = save_to_file;
  is_offline_ = true;
  GE_ASSERT_SUCCESS(root_model_->CheckAndSetNeedSoInOM(), "Check so in om failed, model id:%u.",
                    root_model_->GetModelId());
  if (root_model_->GetSoInOmFlag() == 0U) {
    if (save_to_file) {
      const Status ret =
          FileSaver::SaveToFile(output_file, static_cast<void *>(model_data.model_data), model_data.model_len);
      if (ret == SUCCESS) {
        FileSaver::PrintModelSaveLog();
      }
      return ret;
    } else {
      // zero copy.
      model_buffer.length = model_data.model_len;
      model_buffer.data = std::shared_ptr<uint8_t>(PtrToPtr<void, uint8_t>(model_data.model_data),
                                                   [](const uint8_t *const pointer) { (void)pointer; });
      return SUCCESS;
    }
  }

  const auto &root_graph = root_model_->GetRootGraph();
  GE_ASSERT_NOTNULL(root_graph);
  if (gert::GraphUnfolder::IsGraphNeedUnfold(root_graph)) {
    const auto root_ge_model = GetGeModel();
    root_model_->SetSubgraphInstanceNameToModel(root_graph->GetName(), root_ge_model);
  }

  GE_ASSERT_SUCCESS(SaveToOmRootModel(root_model_, output_file, model_buffer, is_unknown_shape_model_),
                    "SaveToOmRootModel fail");
  GELOGD("SaveToOmRootModel success.");
  return SUCCESS;
}

Status ModelHelper::EnsureKernelBuilt(const GeModelPtr &model) {
  if (model->GetTBEKernelStore().DataSize() == 0U) {
    TBEKernelStore tbe_kernel_store = model->GetTBEKernelStore();
    GE_CHK_BOOL_RET_STATUS(tbe_kernel_store.Build(), FAILED, "tbe_kernel_store build failed, model=%s.",
                           model->GetName().c_str());
    model->SetTBEKernelStore(tbe_kernel_store);
  }

  if (model->GetCustAICPUKernelStore().DataSize() == 0U) {
    CustAICPUKernelStore cust_aicpu_kernel_store = model->GetCustAICPUKernelStore();
    GE_CHK_BOOL_RET_STATUS(cust_aicpu_kernel_store.Build(), FAILED, "cust_aicpu_kernel_store build failed, model=%s.",
                           model->GetName().c_str());
    model->SetCustAICPUKernelStore(cust_aicpu_kernel_store);
  }
  return SUCCESS;
}
REGISTER_MODEL_SAVE_HELPER(OM_FORMAT_DEFAULT, ModelHelper);

Status ModelHelper::UpdateCoreCountWithOption(const std::string &key, const std::string &context_key, uint32_t core_num_ini,
                                            uint32_t &platform_info_count, std::map<std::string, std::string> &options) const {
  std::string core_num_str;
  auto iter = options.find(key);
  if (iter != options.end()) {
    core_num_str = iter->second;
    GELOGI("%s in options, value: [%s].", key.c_str(), core_num_str.c_str());
  } else {
    (void)GetThreadLocalContext().GetOption(context_key, core_num_str);
    GELOGI("%s in ThreadLocalContext, value: [%s].", context_key.c_str(), core_num_str.c_str());
  }

  int32_t core_num = -1;
  if (!core_num_str.empty()) {
    GE_CHK_STATUS_RET(CoreNumUtils::ParseAndValidateCoreNum(ge::GetContext().GetReadableName(AICORE_NUM), core_num_str, 0, core_num_ini, core_num));
  }

  if (core_num > 0) {
    GELOGI("Change %s from platform %u to option %d.", key.c_str(), core_num_ini, core_num);
    platform_info_count = static_cast<uint32_t>(core_num);
  }
  return SUCCESS;
}

Status ModelHelper::UpdateCoreCountWithDevice(const std::string &key, uint32_t core_num_ini,
                                            const std::string &core_num_str, uint32_t &platform_info_count) const {
  int32_t core_num = -1;
  if (!core_num_str.empty()) {
    GE_CHK_STATUS_RET(CoreNumUtils::ParseAndValidateCoreNum(ge::GetContext().GetReadableName(AICORE_NUM), core_num_str, 0, core_num_ini, core_num));
  }

  if (core_num > 0) {
    GELOGI("Change %s from platform %u to device %d.", key.c_str(), core_num_ini, core_num);
    platform_info_count = static_cast<uint32_t>(core_num);
  }

  return SUCCESS;
}

void ModelHelper::UpdateCoreCountWithRuntime(const std::string &key, uint32_t platform_count, const int64_t core_num_rts,
                                          uint32_t &platform_info_count) const {
  GELOGI("Change %s from platform %u to rts %ld.", key.c_str(), platform_count, core_num_rts);
  platform_info_count = static_cast<uint32_t>(core_num_rts);
}

Status ModelHelper::UpdateGeRootModelTaskAddr(const GeRootModelPtr &ge_root_model,
                                              const ComputeGraphPtr &root_graph,
                                              std::set<ComputeGraph *> &refreshed_graphs,
                                              const bool is_cache) {
  GE_ASSERT_NOTNULL(root_graph);
  GE_ASSERT_NOTNULL(ge_root_model);
  std::map<int64_t, int64_t> logical_addr_mapping;
  const auto session_id = root_graph->GetSessionID();
  GE_CHK_STATUS_RET(
      RefreshAndGetAddrMapping(root_graph, session_id, is_cache, logical_addr_mapping, refreshed_graphs),
      "Refresh and get addr mapping in submodel graph failed. graph name %s", root_graph->GetName().c_str());
  GE_CHK_STATUS_RET(UpdateModelTaskAddr(ge_root_model, session_id, logical_addr_mapping, refreshed_graphs));
  return SUCCESS;
}

Status ModelHelper::RefreshAndGetAddrMapping(const ComputeGraphPtr &graph, const uint64_t session_id,
                                             const bool is_cache,
                                             std::map<int64_t, int64_t> &logical_addr_mapping,
                                             std::set<ComputeGraph *> &refreshed_graphs) {
  if (refreshed_graphs.find(graph.get()) != refreshed_graphs.cend()) {
    GELOGI("graph[%s] has been refreshed, no need refresh again", graph->GetName().c_str());
    return SUCCESS;
  }
  std::map<int64_t, NodePtr> unrefreshed_offsets;
  for (const auto &node : graph->GetAllNodes()) {
    const auto &node_type = node->GetType();
    // not support variable in om in current version
    GE_CHK_BOOL_RET_STATUS((is_cache || !OpTypeUtils::IsVariableNode(node_type)), FAILED,
                           "Variable in om is not supported in current version. Node %s is variable.",
                           node->GetName().c_str());
    if ((node_type == VARIABLE) || (node_type == CONSTANTOP) || (node_type == FILECONSTANT)) {
      const auto &op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      const auto &output_offsets = op_desc->GetOutputOffset();
      GE_CHK_BOOL_RET_STATUS(!output_offsets.empty(), FAILED, "Node:%s output offsets is empty",
                             node->GetName().c_str());
      (void) unrefreshed_offsets.emplace(output_offsets[0U], node);
      if (node_type == VARIABLE) {
        const auto &name = node->GetName();
        const auto pos = name.find(kMultiBatchNodePostfix);
        if (pos != std::string::npos) {
          VarManager::Instance(session_id)->SetBatchVariablesKeyName(name, name.substr(0, pos));
        }
      }
    }
  }
  NodeRefreshInfo inputs_need_refresh;
  NodeRefreshInfo outputs_need_refresh;
  GE_CHK_STATUS_RET(RecordOffsetsRefreshInfo(graph, unrefreshed_offsets, inputs_need_refresh, outputs_need_refresh),
                    "Failed to record nodes need refresh offsets in graph:%s", graph->GetName().c_str());
  GE_CHK_STATUS_RET(VarMemAssignUtil::AssignConstantOpMemory(graph),
                    "assign constant op memory failed, graph_name=%s.", graph->GetName().c_str());
  GE_CHK_STATUS_RET(RefreshNodeOffset(inputs_need_refresh, outputs_need_refresh, logical_addr_mapping),
                    "Failed to refresh node offset in graph:%s", graph->GetName().c_str());
  (void)refreshed_graphs.emplace(graph.get());
  return SUCCESS;
}

Status ModelHelper::RecordOffsetsRefreshInfo(const ComputeGraphPtr &graph,
                                             const std::map<int64_t, NodePtr> &unrefreshed_offsets,
                                             NodeRefreshInfo &inputs_need_refresh,
                                             NodeRefreshInfo &outputs_need_refresh) {
  for (const auto &node : graph->GetAllNodes()) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const auto &input_offsets = op_desc->GetInputOffset();
    const auto &input_descs = op_desc->GetAllInputsDescPtr();  // not null, optional input excluded
    GE_CHK_BOOL_RET_STATUS(input_offsets.size() <= input_descs.size(), PARAM_INVALID,
                           "number of input offsets(%zu) mismatches that of input tensor descs'(%zu)",
                           input_offsets.size(), input_descs.size());
    for (size_t i = 0U; i < input_offsets.size(); ++i) {
      auto offset = input_offsets[i];
      int64_t inner_offset = 0;
      (void) ge::AttrUtils::GetInt(input_descs.at(i), ATTR_NAME_INNER_OFFSET, inner_offset);
      GELOGD("Node[%s] input[%zu], offset = %ld, inner_offset = %ld, raw_offset = %ld",
             op_desc->GetName().c_str(), i, inner_offset, offset, offset - inner_offset);
      GE_ASSERT_SUCCESS(CheckInt64SubOverflow(offset, inner_offset));
      offset -= inner_offset;
      const auto &var_node = TryGetVarNodeByOffset(unrefreshed_offsets, offset);
      if ((var_node == nullptr) || (var_node == node)) {
        continue;
      }
      (void)inputs_need_refresh[var_node][node].emplace_back(i, inner_offset);
      GELOGI("Find node:%s input offset:%zu use var:%s offset:%ld", node->GetName().c_str(), i,
             var_node->GetName().c_str(), offset);
    }
    const auto &output_offsets = op_desc->GetOutputOffset();
    GE_CHECK_LE(output_offsets.size(), std::numeric_limits<uint32_t>::max());
    for (size_t i = 0U; i < output_offsets.size(); ++i) {
      auto offset = output_offsets[i];
      int64_t inner_offset = 0;
      (void) ge::AttrUtils::GetInt(op_desc->GetOutputDesc(static_cast<uint32_t>(i)),
                                  ATTR_NAME_INNER_OFFSET, inner_offset);
      GELOGD("Node[%s] output[%zu], offset = %ld, inner_offset = %ld, raw_offset = %ld",
             op_desc->GetName().c_str(), i, offset, inner_offset, offset - inner_offset);
      GE_ASSERT_SUCCESS(CheckInt64SubOverflow(offset, inner_offset));
      offset -= inner_offset;
      const auto &var_node = TryGetVarNodeByOffset(unrefreshed_offsets, offset);
      if ((var_node == nullptr) || (var_node == node)) {
        continue;
      }
      (void)outputs_need_refresh[var_node][node].emplace_back(i, inner_offset);
      GELOGI("Find node:%s output offset:%zu use var:%s offset:%ld", node->GetName().c_str(), i,
             var_node->GetName().c_str(), offset);
    }
  }
  GELOGI("Success to record offsets refresh info of graph:%s", graph->GetName().c_str());
  return SUCCESS;
}

Status ModelHelper::RefreshNodeOffset(const NodeRefreshInfo &inputs_need_refresh,
                                      const NodeRefreshInfo &outputs_need_refresh,
                                      std::map<int64_t, int64_t> &logical_addr_mapping) {
  for (const auto &item : inputs_need_refresh) {
    const auto &var_node = item.first;
    const auto &var_output_offsets = var_node->GetOpDesc()->GetOutputOffset();
    GE_CHK_BOOL_RET_STATUS(!var_output_offsets.empty(), FAILED, "Failed to get output offset of node:%s",
                           var_node->GetName().c_str());
    const int64_t var_offset = var_node->GetOpDesc()->GetOutputOffset()[0U];
    for (const auto &node_to_refresh_inputs : item.second) {
      const auto &refresh_node = node_to_refresh_inputs.first;
      std::vector<int64_t> input_offsets = refresh_node->GetOpDesc()->GetInputOffset();
      for (const auto &index_and_inner_offset : node_to_refresh_inputs.second) {
        const size_t index = index_and_inner_offset.first;
        const int64_t inner_offset = index_and_inner_offset.second;
        GE_CHK_BOOL_RET_STATUS((index < input_offsets.size()), FAILED,
                               "Node:%s input index:%zu is out of range, output size:%zu",
                               refresh_node->GetName().c_str(), index, input_offsets.size());
        const auto orig_input_offset = input_offsets[index];
        GE_ASSERT_SUCCESS(CheckInt64AddOverflow(var_offset, inner_offset));
        const int64_t new_input_offset = var_offset + inner_offset;
        if (new_input_offset != orig_input_offset) {
          input_offsets[index] = new_input_offset;
          GELOGI("Node:%s input:[%zu] offset use var offset:%ld",
                 refresh_node->GetName().c_str(),
                 index,
                 new_input_offset);
          logical_addr_mapping[orig_input_offset] = new_input_offset;
          GELOGD("mapping [%ld] to [%ld]", orig_input_offset, new_input_offset);
        }
      }
      refresh_node->GetOpDesc()->SetInputOffset(input_offsets);
    }
  }
  for (const auto &item : outputs_need_refresh) {
    const auto &var_node = item.first;
    const auto &var_output_offsets = var_node->GetOpDesc()->GetOutputOffset();
    GE_CHK_BOOL_RET_STATUS(!var_output_offsets.empty(), FAILED, "Failed to get output offset of node:%s",
                           var_node->GetName().c_str());
    const int64_t var_offset = var_node->GetOpDesc()->GetOutputOffset()[0U];
    for (const auto &node_to_refresh_outputs : item.second) {
      const auto &refresh_node = node_to_refresh_outputs.first;
      std::vector<int64_t> output_offsets = refresh_node->GetOpDesc()->GetOutputOffset();
      for (const auto &index_and_inner_offset : node_to_refresh_outputs.second) {
        const size_t index = index_and_inner_offset.first;
        const int64_t inner_offset = index_and_inner_offset.second;
        GE_CHK_BOOL_RET_STATUS((index < output_offsets.size()), FAILED,
                               "Node:%s output index:%zu is out of range, output size:%zu",
                               refresh_node->GetName().c_str(), index, output_offsets.size());
        const auto orig_output_offset = output_offsets[index];
        GE_ASSERT_SUCCESS(CheckInt64AddOverflow(var_offset, inner_offset));
        const int64_t new_output_offset = var_offset + inner_offset;
        if (new_output_offset != orig_output_offset) {
          output_offsets[index] = new_output_offset;
          GELOGI("Node:%s output:[%zu] offset use var offset:%ld",
                 refresh_node->GetName().c_str(),
                 index,
                 new_output_offset);
          logical_addr_mapping[orig_output_offset] = new_output_offset;
          GELOGD("mapping [%ld] to [%ld]", output_offsets[index], new_output_offset);
        }
      }
      refresh_node->GetOpDesc()->SetOutputOffset(output_offsets);
    }
  }
  return SUCCESS;
}

Status ModelHelper::UpdateModelTaskAddr(const GeRootModelPtr &ge_root_model, const uint64_t session_id,
                                        const std::map<int64_t, int64_t> &logical_addr_mapping,
                                        std::set<ComputeGraph *> &refreshed_graphs) {
  GE_ASSERT_NOTNULL(ge_root_model);
  for (auto &name_and_model : ge_root_model->GetSubgraphInstanceNameToModel()) {
    const auto &ge_model = name_and_model.second;
    GE_ASSERT_NOTNULL(ge_model);
    auto model_task_def = ge_model->GetModelTaskDefPtr();
    if (model_task_def != nullptr) {
      for (int32_t task_idx = 0; task_idx < model_task_def->task_size(); ++task_idx) {
        auto *task_def = model_task_def->mutable_task(task_idx);
        if (static_cast<ModelTaskType>(task_def->type()) == ModelTaskType::MODEL_TASK_FFTS_PLUS) {
          UpdateFftsPlusTaskAddr(*task_def, logical_addr_mapping);
        }
      }
    }
    // for dynamic shape, graphs in ge model are used to generate lowering kernel in rt2 procedure
    const auto model_graph = ge_model->GetGraph();
    GE_ASSERT_NOTNULL(model_graph);
    std::map<int64_t, int64_t> useless_mapping;
    GE_CHK_STATUS_RET(RefreshAndGetAddrMapping(model_graph, session_id, true, useless_mapping, refreshed_graphs),
                      "Refresh and get addr mapping in ge model graph failed. graph name %s", model_graph->GetName().c_str());
  }
  return SUCCESS;
}

Status ModelHelper::UpdateSessionGraphId(const ComputeGraphPtr &graph,
                                         const std::string &session_graph_id,
                                         bool &refreshed) {
  GE_CHECK_NOTNULL(graph);
  std::string value;
  const std::string* value_ptr = AttrUtils::GetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID);
  if (value_ptr != nullptr) {
    value = *value_ptr;
    if (value == session_graph_id) {
      GELOGD("graph[%s] session graph id is same, no need update.", graph->GetName().c_str());
      return SUCCESS;
    }
  }
  refreshed = true;
  GELOGI("need update graph[%s] session graph id from %s to %s.", graph->GetName().c_str(), value.c_str(),
         session_graph_id.c_str());
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id), FAILED,
                         "Set ATTR_NAME_SESSION_GRAPH_ID[%s] failed for graph:%s", session_graph_id.c_str(),
                         graph->GetName().c_str());
  for (const auto &node : graph->GetDirectNode()) {
    const auto &opdesc = node->GetOpDesc();
    GE_CHECK_NOTNULL(opdesc);
    const std::string* op_session_graph_id = AttrUtils::GetStr(opdesc, ATTR_NAME_SESSION_GRAPH_ID);
    // some op save session graph id in opdesc
    if (op_session_graph_id != nullptr && (*op_session_graph_id != session_graph_id)) {
      GE_CHK_BOOL_RET_STATUS(AttrUtils::SetStr(opdesc, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id), FAILED,
                             "Set ATTR_NAME_SESSION_GRAPH_ID[%s] failed for op:%s", session_graph_id.c_str(),
                             opdesc->GetName().c_str());
    }
  }
  for (const auto &subgraph : graph->GetAllSubgraphs()) {
    GE_CHK_BOOL_RET_STATUS(AttrUtils::SetStr(*subgraph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id), FAILED,
                           "Set ATTR_NAME_SESSION_GRAPH_ID[%s] failed for graph:%s", session_graph_id.c_str(),
                           subgraph->GetName().c_str());
  }
  return SUCCESS;
}

ModelSaveHelperFactory &ModelSaveHelperFactory::Instance() {
  static ModelSaveHelperFactory instance;
  return instance;
}
}  // namespace ge

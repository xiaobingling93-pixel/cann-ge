/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/manager/graph_var_manager.h"

#include "framework/common/types.h"
#include "common/file_constant_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/tuning_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/ge_context.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/math/math_util.h"
#include "common/const_place_holder_utils.h"
#include "runtime/dev.h"
#include "common/checker.h"
#include "formats/utils/formats_trans_utils.h"
#include "base/err_msg.h"

namespace ge {
namespace {
ge::Status InitVarIfHasInitValue(const VarDevAddrMgr *const var_mgr, void *var_dev_addr) {
  ge::ConstGeTensorPtr init_value = nullptr;
  (void)AttrUtils::GetTensor(&var_mgr->tensor_desc, ATTR_NAME_INIT_VALUE, init_value);
  if (init_value != nullptr) {
    int64_t var_size = 0;
    const auto &init_desc = init_value->GetTensorDesc();
    const auto &var_desc = var_mgr->tensor_desc;
    GE_ASSERT_TRUE(var_desc.GetShape().GetDims() == init_desc.GetShape().GetDims(), "_init_value shape not match,"
                   " var_shape: %s, init_shape: %s", var_desc.GetShape().ToString().c_str(),
                   init_desc.GetShape().ToString().c_str());
    GE_ASSERT_TRUE(var_desc.GetDataType() == init_desc.GetDataType(), "_init_value data type not match, "
                   "var data type: %s, init value data type: %s",
                   TypeUtils::DataTypeToSerialString(var_desc.GetDataType()).c_str(),
                   TypeUtils::DataTypeToSerialString(init_desc.GetDataType()).c_str());
    GE_ASSERT_TRUE(var_desc.GetFormat() == init_desc.GetFormat(), "_init_value format not match, "
                   "var format: %s, init format: %s",
                   TypeUtils::FormatToSerialString(var_desc.GetFormat()).c_str(),
                   TypeUtils::FormatToSerialString(init_desc.GetFormat()).c_str());

    (void)TensorUtils::GetSize(var_mgr->tensor_desc, var_size);
    const auto init_value_size = init_value->GetData().GetSize();
    GE_ASSERT_TRUE(init_value_size <= static_cast<size_t>(var_size), "_init_value size too big."
                   " var_size: %" PRId64 ", init_value_size: %zu", var_size, init_value_size);
    GE_CHK_RT_RET(rtMemcpy(var_dev_addr, static_cast<uint64_t>(var_size), init_value->GetData().GetData(), init_value_size, RT_MEMCPY_HOST_TO_DEVICE)); 
    GELOGI("variable offset[%p] has _init_value attr, init value success, var_dev_addr: %p, tensor size: %" PRId64 ","
           " value size: %zu, ", var_mgr->logic_addr, var_dev_addr, var_size, init_value_size);
  } 
  return SUCCESS;
}
}
VarResource::VarResource(const uint64_t session_id) : session_id_(session_id) {}

VarResource::~VarResource() {
  var_offset_map_.clear();
  var_addr_mgr_map_.clear();
  file_constant_var_map_.clear();
  cur_var_tensor_desc_map_.clear();
  var_broad_cast_info_.clear();
  var_dev_addr_mgr_map_.clear();
  device_id_to_var_dev_addr_mgr_map_.clear();
  graph_id_to_changed_var_names_.clear();
  graph_id_to_staged_var_desc_.clear();
  var_is_instance_.clear();
}

ge::Status VarResource::GetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc,
                                   uint8_t **const dev_ptr, rtMemType_t &memory_type) const {
  if (dev_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param dev_ptr is nullptr, var_name:%s, session_id:%" PRIu64 ", "
                       "check invalid", var_name.c_str(), session_id_);
    GELOGE(FAILED, "[Check][Param] Param dev_ptr is nullptr, var_name:%s, session_id:%" PRIu64 "",
           var_name.c_str(), session_id_);
    return FAILED;
  }
  const std::string var_key = VarKey(var_name, tensor_desc);
  GELOGD("VarResource::GetVarAddr, var_key = %s.", var_key.c_str());

  const auto iter = var_addr_mgr_map_.find(var_key);
  if (iter == var_addr_mgr_map_.end()) {
    REPORT_INNER_ERR_MSG("E19999", "var_key:%s can't find in var_addr_mgr_map_, var_name:%s, session_id:%" PRIu64 ", "
                       "check invalid", var_key.c_str(), var_name.c_str(), session_id_);
    GELOGE(FAILED, "[Check][Param] var_key:%s can't find in var_addr_mgr_map_, var_name:%s, session_id:%" PRIu64 "",
           var_key.c_str(), var_name.c_str(), session_id_);
    return FAILED;
  }

  *dev_ptr = const_cast<uint8_t *>(iter->second.address);
  memory_type = iter->second.memory_type;

  return SUCCESS;
}

int32_t VarResource::GetSizeByTensoDataType(const OpDescPtr &op_desc) const {
  const auto &output_tensor = op_desc->GetOutputDescPtr(0);
  if (output_tensor == nullptr) {
    GELOGW("The const %s does not have output 0, skip to fusion", op_desc->GetName().c_str());
    return -1;
  }
  return GetSizeByDataType(output_tensor->GetDataType());
}

Status VarResource::GetFileConstantReuseAddr(const OpDescPtr &op_desc,
                                             uint8_t **const dev_ptr,
                                             rtMemType_t &memory_type) const {
  const auto &fileconstant_info = FileConstantUtils::GetFileConstantInfo(op_desc);
  const auto &file_name = fileconstant_info.weight_path;
  if (file_name.empty()) {
    GELOGW("Failed to get file constant file of %s", op_desc->GetName().c_str());
    return FAILED;
  }

  const auto reuse_key = file_constant_var_map_.find(file_name + ":" + std::to_string(fileconstant_info.weight_offset));
  if (reuse_key == file_constant_var_map_.cend()) {
    // no reusable find.
    return FAILED;
  }

  const auto reuse_var = var_addr_mgr_map_.find(reuse_key->second);
  if (reuse_var == var_addr_mgr_map_.cend()) {
    GELOGW("Failed to find reuse addr by key[%s], file_name=%s, op_name=%s", reuse_key->second.c_str(),
           file_name.c_str(), op_desc->GetName().c_str());
    return FAILED;
  }
  const auto &var_addr_mgr = reuse_var->second;
  *dev_ptr = const_cast<uint8_t *>(var_addr_mgr.address);
  memory_type = var_addr_mgr.memory_type;
  GELOGI("[IMAS]AssignVarMem Set session_%" PRIu64 " name[%s] output[%d] offset to [%" PRIu64
         "] , reuse address of node: %s.",
         session_id_, op_desc->GetName().c_str(), 0, var_addr_mgr.offset, var_addr_mgr.op_desc->GetName().c_str());
  return SUCCESS;
}

Status VarResource::GetReuseAddr(const OpDescPtr &op_desc, uint8_t **const dev_ptr, rtMemType_t &memory_type) const {
  GE_CHECK_NOTNULL(op_desc);
  if (op_desc->GetType() == FILECONSTANT) {
    return GetFileConstantReuseAddr(op_desc, dev_ptr, memory_type);
  }
  const auto type_size = GetSizeByTensoDataType(op_desc);
  if (type_size <= 0) {
    return FAILED;
  }
  GeTensorPtr weight;
  if (!AttrUtils::MutableTensor(op_desc, ATTR_NAME_WEIGHTS, weight)) {
    GELOGW("The const node %s does not have weight attr, skip it", op_desc->GetName().c_str());
    return FAILED;
  }

  const auto &values = weight->MutableData().GetAlignedPtr();
  if (values == nullptr) {
    GELOGD("aligned_ptr is null.");
    return FAILED;
  }
  const auto weight_size = weight->MutableData().size();
  for (const auto &var_maps : var_addr_mgr_map_) {
    const auto &var_map = var_maps.second;
    const bool skip_var = (var_map.op_desc == nullptr) || (var_map.op_desc->GetType() != CONSTANTOP) ||
                          (GetSizeByTensoDataType(var_map.op_desc) != type_size);
    if (skip_var) {
      continue;
    }

    GeTensorPtr tmp_weight;
    if (!AttrUtils::MutableTensor(var_map.op_desc, ATTR_NAME_WEIGHTS, tmp_weight)) {
      GELOGW("The const node %s does not have weight attr, skip it", var_map.op_desc->GetName().c_str());
      continue;
    }

    if ((tmp_weight->MutableData().size() != weight_size) || (tmp_weight->MutableData().GetAlignedPtr() == nullptr)) {
      continue;
    }

    if (memcmp(values->Get(), tmp_weight->MutableData().GetAlignedPtr()->Get(), weight_size) == 0) {
      const uint64_t real_size =
          (weight_size + kSessionMemAlignSize - 1U) / kSessionMemAlignSize * kSessionMemAlignSize;
      GELOGD("[IMAS]AssignVarMem Set session_%" PRIu64 " name[%s] output[%d] offset to [%" PRIu64 "] size[%" PRIu64 "]"
             "realsize[%" PRIu64 "], reuse address of node: %s.", session_id_, op_desc->GetName().c_str(), 0,
             var_map.offset, real_size + (kSessionMemAlignSize * kSessionMemAlignUnit), real_size,
             var_map.op_desc->GetName().c_str());
      *dev_ptr = const_cast<uint8_t *>(var_map.address);
      memory_type = var_map.memory_type;
      return SUCCESS;
    }
  }
  return FAILED;
}

void VarResource::CheckAndCacheFileConstantVar(const OpDescPtr &op_desc, const std::string &var_key) {
  if ((op_desc != nullptr) && (op_desc->GetType() == FILECONSTANT)) {
    const auto file_constant_info = FileConstantUtils::GetFileConstantInfo(op_desc);
    const auto key = file_constant_info.weight_path + ":" + std::to_string(file_constant_info.weight_offset);
    if (file_constant_var_map_.find(key) == file_constant_var_map_.end()) {
      file_constant_var_map_[key] = var_key;
    }
  }
}

void VarResource::SetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc,
                             const uint8_t *const dev_ptr, const rtMemType_t memory_type, const OpDescPtr &op_desc) {
  const std::string var_key = VarKey(var_name, tensor_desc);
  GELOGI("VarResource::SetVarAddr, var_key = %s, mem_type:%u.", var_key.c_str(), memory_type);
  if (var_addr_mgr_map_.count(var_key) == 0U) {
    GELOGI("SetVarAddr node_name %s, tensor_desc data_type %s, format %s", var_name.c_str(),
           TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
           TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());
    uint64_t offset = 0U;
    if (memory_type == RT_MEMORY_HBM) {
      offset = PtrToValue(dev_ptr) - VarManager::Instance(session_id_)->GetVarMemLogicBase();
    }
    var_addr_mgr_map_[var_key] = {tensor_desc, dev_ptr, offset, memory_type, op_desc};
    CheckAndCacheFileConstantVar(op_desc, var_key);
  }

  cur_var_tensor_desc_map_[GetBatchVarKeyName(var_name)] = tensor_desc;
}

ge::Status VarResource::SaveVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc,
                                    const uint8_t *const address, const rtMemType_t memory_type,
                                    const OpDescPtr &op_desc) {
  const std::string var_key = VarKey(var_name, tensor_desc);
  GE_CHECK_NOTNULL(VarManager::Instance(session_id_));
  GELOGD("VarResource::SaveVarAddr, var_key = %s.", var_key.c_str());
  if (var_addr_mgr_map_.count(var_key) == 0U) {
    uint64_t logic_address = PtrToValue(address);
    if (memory_type == RT_MEMORY_HBM) {
      logic_address += VarManager::Instance(session_id_)->GetVarMemLogicBase();
    }
    var_addr_mgr_map_[var_key] = {tensor_desc, PtrToPtr<void, uint8_t>(ValueToPtr(logic_address)), PtrToValue(address),
                                  memory_type, op_desc};
    var_offset_map_[logic_address] = memory_type;
    CheckAndCacheFileConstantVar(op_desc, var_key);
    uint8_t *device_addr = nullptr;
    if ((op_desc != nullptr) && (op_desc->GetType() == CONSTPLACEHOLDER)) {
      GE_ASSERT_GRAPH_SUCCESS(GetConstPlaceHolderAddr(op_desc, device_addr));
    }
    const bool is_extern_mem = (device_addr != nullptr);

    var_dev_addr_mgr_map_[logic_address] = {tensor_desc, reinterpret_cast<uint8_t *>(logic_address),
                                            device_addr, is_extern_mem};
    GELOGI("SaveVarAddr node_name %s, tensor_desc format %s, data_type %s, is_extern_mem %d, logic_address %p.",
           var_name.c_str(), TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str(),
           TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(), is_extern_mem,
           reinterpret_cast<uint8_t *>(logic_address));
    return SUCCESS;
  }

  REPORT_INNER_ERR_MSG("E19999", "var_key:%s conflict in var_addr_mgr_map_, var_name:%s, session_id:%" PRIu64 ", "
                     "check invalid", var_key.c_str(), var_name.c_str(),
                     session_id_);
  GELOGE(FAILED, "[Check][Param] var_key:%s conflict in var_addr_mgr_map_, var_name:%s, session_id:%" PRIu64 "",
         var_key.c_str(), var_name.c_str(), session_id_);
  return FAILED;
}

bool VarResource::IsVarExist(const std::string &var_name, const ge::GeTensorDesc &tensor_desc) const {
  const std::string var_key = VarKey(var_name, tensor_desc);
  return var_addr_mgr_map_.count(var_key) != 0U;
}

bool VarResource::IsVarExist(const std::string &var_name) const {
  return cur_var_tensor_desc_map_.count(GetBatchVarKeyName(var_name)) != 0U;
}

void VarResource::SetVarIsReady(const std::string &var_name,
                                const ge::GeTensorDesc &tensor_desc,
                                const uint32_t device_id) {
  std::string var_key = VarKey(var_name, tensor_desc);
  (void) var_is_instance_[device_id].emplace(var_key);
}

bool VarResource::IsVarReady(const std::string &var_name,
                             const ge::GeTensorDesc &tensor_desc,
                             const uint32_t device_id) const {
  const auto &iter = var_is_instance_.find(device_id);
  if (iter == var_is_instance_.cend()) {
    return false;
  }
  return iter->second.count(VarKey(var_name, tensor_desc)) != 0U;
}

std::string VarResource::VarKey(const std::string &var_name, const ge::GeTensorDesc &tensor_desc) const {
  std::string var_key(GetBatchVarKeyName(var_name));
  (void)var_key.append(std::to_string(static_cast<int32_t>(tensor_desc.GetFormat())))
    .append("_")
    .append(std::to_string(static_cast<int32_t>(tensor_desc.GetDataType())));
  return var_key;
}

std::string VarResource::GetBatchVarKeyName(const std::string &var_name) const {
  const auto iter = batch_var_name_map_.find(var_name);
  return (iter == batch_var_name_map_.end()) ? (var_name) : (iter->second);
}

ge::Status VarResource::GetCurVarDesc(const std::string &var_name, ge::GeTensorDesc &tensor_desc) {
  const auto var_key_name = GetBatchVarKeyName(var_name);
  if (cur_var_tensor_desc_map_.count(var_key_name) == 0U) {
    return FAILED;
  }
  tensor_desc = cur_var_tensor_desc_map_[var_key_name];
  return SUCCESS;
}

ge::Status VarResource::RecordStagedVarDesc(const uint32_t graph_id,
                                            const std::string &var_name,
                                            const GeTensorDesc &tensor_desc) {
  graph_id_to_staged_var_desc_[graph_id][var_name] = tensor_desc;
  return SUCCESS;
}

const std::map<std::string, GeTensorDesc> &VarResource::GetStagedVarDescs(const uint32_t graph_id) const {
  const auto &iter = graph_id_to_staged_var_desc_.find(graph_id);
  if (iter != graph_id_to_staged_var_desc_.cend()) {
    return iter->second;
  }
  static std::map<std::string, GeTensorDesc> empty;
  return empty;
}

ge::Status VarResource::RenewCurVarDesc(const std::string &var_name, const GeTensorDesc &tensor_desc) {
  const auto var_key_name = GetBatchVarKeyName(var_name);
  if (cur_var_tensor_desc_map_.count(var_key_name) == 0U) {
    GELOGI("There is no this node[%s] key[%s] in var tensor_desc map. so no need renew!",
           var_name.c_str(), var_key_name.c_str());
    return SUCCESS;
  }

  ge::GeTensorDesc curr_desc;
  (void) GetCurVarDesc(var_name, curr_desc);  // already checked
  GELOGI("Renew variable data for %s from format %s to %s success",
         var_name.c_str(), TypeUtils::FormatToSerialString(curr_desc.GetFormat()).c_str(),
         TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());
  std::string key = VarKey(var_name, curr_desc);
  curr_desc.SetOriginFormat(tensor_desc.GetOriginFormat());
  curr_desc.SetFormat(tensor_desc.GetFormat());
  cur_var_tensor_desc_map_[var_key_name] = curr_desc;
  const auto iter = var_addr_mgr_map_.find(key);
  GE_CHK_BOOL_RET_STATUS(iter != var_addr_mgr_map_.end(), FAILED,
                         "[Check][Param] var_key:%s can't find in var_addr_mgr_map_,"
                         "var_name:%s, session_id:%" PRIu64 "",
                         key.c_str(), var_name.c_str(), session_id_);
  auto val = iter->second;
  val.tensor_desc.SetOriginFormat(tensor_desc.GetOriginFormat());
  val.tensor_desc.SetFormat(tensor_desc.GetFormat());
  (void)var_addr_mgr_map_.erase(iter);
  key = VarKey(var_name, curr_desc);
  var_addr_mgr_map_[key] = val;
  return SUCCESS;
}

ge::Status VarResource::RenewCurVarDesc(const std::string &var_name, const ge::OpDescPtr &op_desc) {
  const auto var_key_name = GetBatchVarKeyName(var_name);
  if (cur_var_tensor_desc_map_.count(var_key_name) == 0U) {
    GELOGI("There is no this node[%s] key[%s] in var tensor_desc map. so no need renew!",
           var_name.c_str(), var_key_name.c_str());
    return SUCCESS;
  }

  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param op_desc is nullptr, var_name:%s, session_id:%" PRIu64 ", check invalid",
                       var_name.c_str(), session_id_);
    GELOGE(FAILED, "[Check][Param] input opdesc is nullptr, var_name:%s, session_id:%" PRIu64 "",
           var_name.c_str(), session_id_);
    return FAILED;
  }

  ge::GeTensorDesc curr_desc;
  const ge::Status ret = GetCurVarDesc(var_name, curr_desc);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Get][CurVarDesc] fail, var_name:%s, session_id:%" PRIu64 "", var_name.c_str(), session_id_);
    return FAILED;
  }
  GELOGI("Trans variable data for %s from format %s to %s success",
         var_name.c_str(), TypeUtils::FormatToSerialString(curr_desc.GetFormat()).c_str(),
         TypeUtils::FormatToSerialString((op_desc->GetOutputDesc(0U)).GetFormat()).c_str());
  std::string key = VarKey(var_name, curr_desc);
  curr_desc.SetOriginFormat((op_desc->GetOutputDesc(0U)).GetOriginFormat());
  curr_desc.SetFormat((op_desc->GetOutputDesc(0U)).GetFormat());
  cur_var_tensor_desc_map_[var_key_name] = curr_desc;
  const auto iter = var_addr_mgr_map_.find(key);
  if (iter == var_addr_mgr_map_.end()) {
    REPORT_INNER_ERR_MSG("E19999", "var_key:%s can't find in var_addr_mgr_map_, var_name:%s, "
                       "session_id:%" PRIu64 ", op:%s(%s), check invalid", key.c_str(), var_name.c_str(),
                       session_id_, op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Check][Param] var_key:%s can't find in var_addr_mgr_map_, var_name:%s, "
           "session_id:%" PRIu64 ", op:%s(%s)", key.c_str(), var_name.c_str(), session_id_,
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }
  auto val = iter->second;
  val.tensor_desc.SetOriginFormat((op_desc->GetOutputDesc(0U)).GetOriginFormat());
  val.tensor_desc.SetFormat((op_desc->GetOutputDesc(0U)).GetFormat());
  (void)var_addr_mgr_map_.erase(iter);
  key = VarKey(var_name, curr_desc);
  var_addr_mgr_map_[key] = val;

  return SUCCESS;
}

void VarResource::SaveBroadCastInfo(const uint32_t graph_id, const VarBroadCastInfo &broad_cast_info) {
  var_broad_cast_info_[graph_id][broad_cast_info.var_name] = broad_cast_info;
}

bool VarResource::IsVarAddr(const int64_t offset) const {
  return var_offset_map_.count(static_cast<uint64_t>(offset)) > 0U;
}

rtMemType_t VarResource::GetVarMemType(const int64_t offset) {
  if (var_offset_map_.count(static_cast<uint64_t>(offset)) > 0U) {
    return var_offset_map_[static_cast<uint64_t>(offset)];
  }
  return RT_MEMORY_RESERVED;
}

void VarResource::UpdateDevVarMgrInfo(const uint32_t device_id) {
  GELOGI("Update var manager info on device[%u]", device_id);
  const auto &device_id_and_var_dev_addr_mgr = device_id_to_var_dev_addr_mgr_map_.find(device_id);
  if (device_id_and_var_dev_addr_mgr == device_id_to_var_dev_addr_mgr_map_.end()) {
    device_id_to_var_dev_addr_mgr_map_[device_id] = var_dev_addr_mgr_map_;
    return;
  }
  auto &var_dev_addr_mgr_map = device_id_and_var_dev_addr_mgr->second;
  for (const auto &iter : var_dev_addr_mgr_map_) {
    if (var_dev_addr_mgr_map.find(iter.first) == var_dev_addr_mgr_map.end()) {
      var_dev_addr_mgr_map[iter.first] = iter.second;
    }
  }
}

VarDevAddrMgr *VarResource::GetVarMgrInfo(const uint32_t device_id, const int64_t offset) {
  const auto &iter = device_id_to_var_dev_addr_mgr_map_.find(device_id);
  if (iter == device_id_to_var_dev_addr_mgr_map_.end()) {
    GELOGW("Var manager info on device[%u] has not been initialized", device_id);
    return nullptr;
  }
  const auto &var_dev_addr_mgr = iter->second.find(static_cast<uint64_t>(offset));
  if (var_dev_addr_mgr != iter->second.end()) {
    return &var_dev_addr_mgr->second;
  }
  return nullptr;
}

void VarResource::SetVarLoaded(const uint32_t device_id, const std::string &var_name, const int64_t offset) {
  (void)dev_loaded_var_offset_[device_id].emplace(std::make_pair(var_name, offset));
}

bool VarResource::IsVarLoaded(const uint32_t device_id, const int64_t offset, std::string &loaded_var_name) const {
  const auto &iter = dev_loaded_var_offset_.find(device_id);
  if (iter == dev_loaded_var_offset_.cend()) {
    return false;
  }
  for (const auto &var_offsets : iter->second) {
    if (var_offsets.second == offset) {
      loaded_var_name = var_offsets.first;
      return true;
    }
  }
  return false;
}

ge::Status VarResource::SetVarMgrDevAddr(const uint32_t device_id, const int64_t offset, uint8_t *const dev_addr) {
  const auto &iter = device_id_to_var_dev_addr_mgr_map_.find(device_id);
  if (iter == device_id_to_var_dev_addr_mgr_map_.cend()) {
    GELOGW("Var manager info on device[%u] has not been initialized", device_id);
    return ge::INTERNAL_ERROR;
  }
  const auto &var_dev_addr_mgr = iter->second.find(static_cast<uint64_t>(offset));
  if (var_dev_addr_mgr != iter->second.end()) {
    var_dev_addr_mgr->second.dev_addr = dev_addr;
    return SUCCESS;
  }
  return ge::INTERNAL_ERROR;
}

ge::Status VarResource::CheckLogicAddrVaild(const uint32_t device_id,
                                            const uint8_t *const logic_addr,
                                            uint64_t &inner_offset_tmp,
                                            uint64_t &logic_addr_tmp) {
  GELOGD("[VarResource] Begin to check logic addr is vaild.");
  inner_offset_tmp = 0U;
  logic_addr_tmp = PtrToValue(logic_addr);
  const auto &var_dev_addr_mgr_map = device_id_to_var_dev_addr_mgr_map_[device_id];
  for (const auto &info : var_dev_addr_mgr_map) {
    int64_t tensor_size = 0;
    (void)TensorUtils::GetSize(info.second.tensor_desc, tensor_size);
    if ((PtrToValue(logic_addr) > info.first) &&
        (PtrToValue(logic_addr) < (info.first + static_cast<uint64_t>(tensor_size)))) {
      inner_offset_tmp = PtrToValue(logic_addr) - info.first;
      logic_addr_tmp = info.first;
      return SUCCESS;
    }
  }
  return ge::INTERNAL_ERROR;
}

VarTransRoad *VarResource::GetTransRoad(const std::string &var_name) {
  const auto iter = var_to_trans_road_.find(GetBatchVarKeyName(var_name));
  if (iter == var_to_trans_road_.end()) {
    return nullptr;
  } else {
    return &(iter->second);
  }
}

Status VarResource::GetChangedGraphId(const std::string &var_name, uint32_t &graph_id) const {
  const auto iter = var_names_to_changed_graph_id_.find(GetBatchVarKeyName(var_name));
  if (iter == var_names_to_changed_graph_id_.end()) {
    return FAILED;
  } else {
    graph_id = iter->second;
    return SUCCESS;
  }
}

std::set<std::string> VarResource::GetChangedVarNames(const uint32_t graph_id) const {
  const auto &iter = graph_id_to_changed_var_names_.find(graph_id);
  if (iter != graph_id_to_changed_var_names_.cend()) {
    return iter->second;
  }
  return std::set<std::string>();
}

Status VarResource::GetAllocatedGraphId(const std::string &var_name, uint32_t &graph_id) const {
  const auto iter = var_names_to_allocated_graph_id_.find(GetBatchVarKeyName(var_name));
  if (iter == var_names_to_allocated_graph_id_.end()) {
    return FAILED;
  } else {
    graph_id = iter->second;
    return SUCCESS;
  }
}

Status VarResource::SetAllocatedGraphId(const std::string &var_name, uint32_t graph_id) {
  if (GetAllocatedGraphId(var_name, graph_id) == SUCCESS) {
    GELOGW("VarManager var[%s] has been allocated in graph[%d]", var_name.c_str(), graph_id);
    return SUCCESS;
  }
  var_names_to_allocated_graph_id_[GetBatchVarKeyName(var_name)] = graph_id;
  return SUCCESS;
}

Status VarResource::VarDescInfoToSerial(deployer::VarDescInfo &desc_info) const {
  for (const auto &info : cur_var_tensor_desc_map_) {
    proto::TensorDescriptor tensor_desc_proto;
    GeTensorSerializeUtils::GeTensorDescAsProto(info.second, &tensor_desc_proto);
    (void)desc_info.mutable_cur_var_tensor_desc_map()->insert({info.first, tensor_desc_proto});
  }

  for (const auto &info : var_to_trans_road_) {
    deployer::TransNodeMultiInfo trans_node_info;
    for (auto &x : info.second) {
      deployer::SingleTransNodeInfo *const single_info = trans_node_info.add_node_info();
      single_info->set_node_type(x.node_type);
      GeTensorSerializeUtils::GeTensorDescAsProto(x.input, single_info->mutable_input());
      GeTensorSerializeUtils::GeTensorDescAsProto(x.output, single_info->mutable_output());
    }
    (void)desc_info.mutable_var_to_trans_road()->insert({info.first, trans_node_info});
  }
  return SUCCESS;
}

Status VarResource::VarResourceToSerial(deployer::VarResourceInfo *const var_resource_info) const {
  GELOGD("[VarResource] Begin to serial var_resource object.");
  GE_CHECK_NOTNULL(var_resource_info);
  for (auto &info : var_offset_map_) {
    (void)var_resource_info->mutable_var_offset_map()->insert({info.first, info.second});
  }

  for (auto &info : var_addr_mgr_map_) {
    deployer::VarAddrMgrInfo  var_addr_mgr_info;
    GeTensorSerializeUtils::GeTensorDescAsProto(info.second.tensor_desc, var_addr_mgr_info.mutable_desc());
    var_addr_mgr_info.set_address(PtrToValue(info.second.address));
    var_addr_mgr_info.set_offset(info.second.offset);
    var_addr_mgr_info.set_memory_type(static_cast<uint64_t>(info.second.memory_type));
    (void)var_resource_info->mutable_var_addr_mgr_map()->insert({info.first, var_addr_mgr_info});
  }

  for (auto &info : cur_var_tensor_desc_map_) {
    proto::TensorDescriptor tensor_desc_proto;
    GeTensorSerializeUtils::GeTensorDescAsProto(info.second, &tensor_desc_proto);
    (void)var_resource_info->mutable_cur_var_tensor_desc_map()->insert({info.first, tensor_desc_proto});
  }

  for (auto &info : var_to_trans_road_) {
    deployer::TransNodeMultiInfo trans_node_info;
    for (auto &x : info.second) {
      deployer::SingleTransNodeInfo *const single_info = trans_node_info.add_node_info();
      single_info->set_node_type(x.node_type);
      GeTensorSerializeUtils::GeTensorDescAsProto(x.input, single_info->mutable_input());
      GeTensorSerializeUtils::GeTensorDescAsProto(x.output, single_info->mutable_output());
    }
    (void)var_resource_info->mutable_var_to_trans_road()->insert({info.first, trans_node_info});
  }

  for (const auto &info : var_dev_addr_mgr_map_) {
    deployer::VarDevAddrMgr  var_dev_addr_mgr;
    GeTensorSerializeUtils::GeTensorDescAsProto(info.second.tensor_desc, var_dev_addr_mgr.mutable_desc());
    var_dev_addr_mgr.set_address(PtrToValue(info.second.logic_addr));
    var_dev_addr_mgr.set_dev_addr(PtrToValue(info.second.dev_addr));
    (void)var_resource_info->mutable_var_dev_addr_mgr_map()->insert({info.first, var_dev_addr_mgr});
  }

  for (auto &info : var_names_to_changed_graph_id_) {
    (void)var_resource_info->mutable_var_names_to_changed_graph_id()->insert({info.first, info.second});
  }

  for (auto &info : var_names_to_allocated_graph_id_) {
    (void)var_resource_info->mutable_var_names_to_allocated_graph_id()->insert({info.first, info.second});
  }

  for (auto &info : var_broad_cast_info_) {
    deployer::BroadcastMultiInfo broadcast_multi_info;
    for (auto &x : info.second) {
      deployer::BroadcastInfo broadcast_info;
      broadcast_info.set_var_name(x.second.var_name);
      broadcast_info.set_broadcast_name(x.second.broadcast_name);
      broadcast_info.set_idx(x.second.idx);
      broadcast_info.set_input_offset(x.second.input_offset);
      broadcast_info.set_input_size(x.second.input_size);
      broadcast_info.set_output_offset(x.second.output_offset);
      broadcast_info.set_output_size(x.second.output_size);
      (void)broadcast_multi_info.mutable_broadcast_info()->insert({x.first, broadcast_info});
    }
    (void)var_resource_info->mutable_var_broad_cast_info()->insert({info.first, broadcast_multi_info});
  }
  GELOGD("[VarResource] Success to serial var_resource object.");
  return SUCCESS;
}

Status VarResource::VarResourceToDeserial(const deployer::VarResourceInfo *const var_resource_info) {
  GELOGD("[VarResource] Begin to deserial var_resource object.");
  GE_CHECK_NOTNULL(var_resource_info);
  auto name_changed_graph_id_map = var_resource_info->var_names_to_changed_graph_id();
  auto name_alloc_graph_id_map = var_resource_info->var_names_to_allocated_graph_id();
  for (const auto &x : var_resource_info->var_offset_map()) {
    (void)var_offset_map_.insert(std::pair<uint64_t, rtMemType_t>(x.first, x.second));
  }

  for (const auto &x : var_resource_info->var_addr_mgr_map()) {
    GeTensorDesc tensor_desc;
    GeTensorSerializeUtils::AssembleGeTensorDescFromProto(&x.second.desc(), tensor_desc);
    const struct VarAddrMgr addr_mgr = {tensor_desc, PtrToPtr<void, uint8_t>(ValueToPtr(x.second.address())),
        static_cast<uint64_t>(x.second.offset()), static_cast<rtMemType_t>(x.second.memory_type()), nullptr};
    (void)var_addr_mgr_map_.insert(std::pair<std::string, VarAddrMgr>(x.first, addr_mgr));
  }

  for (const auto &x : var_resource_info->cur_var_tensor_desc_map()) {
    GeTensorDesc tensor_desc;
    GeTensorSerializeUtils::AssembleGeTensorDescFromProto(&x.second, tensor_desc);
    (void)cur_var_tensor_desc_map_.insert(std::pair<std::string, GeTensorDesc>(x.first, tensor_desc));
  }

  for (const auto &x : var_resource_info->var_to_trans_road()) {
    std::vector<TransNodeInfo> trans_node_info_vec;
    for (int32_t i = 0; i < x.second.node_info_size(); i++) {
      TransNodeInfo trans_node_info;
      trans_node_info.node_type = x.second.node_info(i).node_type();
      const proto::TensorDescriptor &input_tensor_desc = x.second.node_info(i).input();
      const proto::TensorDescriptor &output_tensor_desc = x.second.node_info(i).output();
      GeTensorSerializeUtils::AssembleGeTensorDescFromProto(&input_tensor_desc, trans_node_info.input);
      GeTensorSerializeUtils::AssembleGeTensorDescFromProto(&output_tensor_desc, trans_node_info.output);
      (void)trans_node_info_vec.emplace_back(trans_node_info);
    }
    (void)var_to_trans_road_.insert(std::pair<std::string, std::vector<TransNodeInfo>>(x.first, trans_node_info_vec));
  }

  for (const auto &x : var_resource_info->var_dev_addr_mgr_map()) {
    GeTensorDesc tensor_desc;
    GeTensorSerializeUtils::AssembleGeTensorDescFromProto(&x.second.desc(), tensor_desc);
    const struct VarDevAddrMgr dev_addr_mgr = {tensor_desc, reinterpret_cast<uint8_t *>(x.second.address()),
                                               reinterpret_cast<uint8_t *>(x.second.dev_addr()), false};
    (void)var_dev_addr_mgr_map_.insert(std::pair<uint64_t, VarDevAddrMgr>(x.first, dev_addr_mgr));
  }

  var_names_to_changed_graph_id_.insert(name_changed_graph_id_map.begin(), name_changed_graph_id_map.end());
  var_names_to_allocated_graph_id_.insert(name_alloc_graph_id_map.begin(), name_alloc_graph_id_map.end());
  for (const auto &x : var_resource_info->var_broad_cast_info()) {
    std::unordered_map<std::string, VarBroadCastInfo> var_broadcast_info;
    const deployer::BroadcastMultiInfo &boardcast_multi_info = x.second;
    for (const auto &broadcast_info : boardcast_multi_info.broadcast_info()) {
      const auto &bc = broadcast_info.second;
      const struct VarBroadCastInfo info = {bc.var_name(),   bc.broadcast_name(), bc.idx(),        bc.input_offset(),
                                            bc.input_size(), bc.output_offset(),  bc.output_size()};
      (void)var_broadcast_info.insert(std::pair<std::string, VarBroadCastInfo>(broadcast_info.first, info));
    }
    (void)var_broad_cast_info_.insert(
        std::pair<uint32_t, std::unordered_map<std::string, VarBroadCastInfo>>(x.first, var_broadcast_info));
  }
  GELOGD("[VarResource] Success to deserial var_resource object.");
  return SUCCESS;
}

void VarResource::SetBatchVariablesKeyName(const std::string &batch_var_name, const std::string &key_name) {
  batch_var_name_map_[batch_var_name] = key_name;
}

bool VarResource::HasSharedVarMemBetweenBatch() const {
  return !batch_var_name_map_.empty();
}

MemResource::MemResource() {}

std::shared_ptr<MemResource> MemResource::BuildMemResourceFromType(const rtMemType_t mem_type) {
  std::shared_ptr<MemResource> resource = nullptr;
  switch (mem_type) {
    case RT_MEMORY_HBM:
      resource = MakeShared<HbmMemResource>();
      break;
    case RT_MEMORY_RDMA_HBM:
      resource = MakeShared<RdmaMemResource>();
      break;
    case RT_MEMORY_HOST:
      resource = MakeShared<HostMemResource>();
      break;
    default:
      break;
  }
  return resource;
}

Status HbmMemResource::AssignVarMem(const std::string &var_name, const uint64_t size, const uint64_t session_id,
                                    size_t &mem_offset, const OpDescPtr &op_desc) {
  FMK_UINT64_ADDCHECK(size, kSessionMemAlignSize);
  uint64_t align_size = (size + kSessionMemAlignSize - 1U) / kSessionMemAlignSize * kSessionMemAlignSize;
  const uint64_t real_size = align_size;
  GE_CHECK_NOTNULL(VarManager::Instance(session_id));
  mem_offset = var_mem_size_;

  // offset for next, align 512 BYTE
  FMK_UINT64_ADDCHECK(align_size, kSessionMemAlignSize);
  align_size = align_size + kSessionMemAlignSize;
  FMK_UINT64_ADDCHECK(var_mem_size_, align_size);
  var_mem_size_ = var_mem_size_ + align_size;

  // align 512 BYTE
  FMK_UINT64_ADDCHECK(var_mem_size_, kSessionMemAlignSize);
  var_mem_size_ = var_mem_size_ + kSessionMemAlignSize;
  if ((op_desc != nullptr) && (op_desc->GetType() == CONSTPLACEHOLDER)) {
      const_place_holder_mem_size_ += var_mem_size_ - mem_offset;
  }
  GELOGI("[IMAS]AssignVarMem Set session_%" PRIu64 " name[%s] output[%d] offset to [%zu] size["
         "%" PRIu64 "] realsize[%" PRIu64 "].", session_id, var_name.c_str(), 0,
	        mem_offset, (var_mem_size_ - mem_offset), real_size);
  return SUCCESS;
}

Status RdmaMemResource::AssignVarMem(const std::string &var_name, const uint64_t size, const uint64_t session_id,
                                     size_t &mem_offset, const OpDescPtr &op_desc) {
  (void)op_desc;
  GE_CHECK_NOTNULL(VarManager::Instance(session_id));
  uint8_t *const buffer = VarManager::Instance(session_id)->GetRdmaPoolMemory(RT_MEMORY_HBM, size);
  if (buffer == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "malloc rdma memory fail, var_size:%" PRIu64 ", var_name:%s",
                      size, var_name.c_str());
    GELOGE(MEMALLOC_FAILED, "[Malloc][RdmaMemory] for node %s failed, size = %" PRIu64 "", var_name.c_str(), size);
    return MEMALLOC_FAILED;
  }
  mem_offset = static_cast<size_t>(PtrToValue(buffer));
  FMK_UINT64_ADDCHECK(var_mem_size_, size);
  var_mem_size_ += size;
  GELOGI("[IMAS]AssignVarMem Set session_%" PRIu64 " name[%s] output[%d] addr to [%p] size[%" PRIu64 "].",
         session_id, var_name.c_str(), 0, buffer, size);
  return SUCCESS;
}

Status HostMemResource::AssignVarMem(const std::string &var_name, const uint64_t size, const uint64_t session_id,
                                     size_t &mem_offset, const OpDescPtr &op_desc) {
  (void)op_desc;
  GELOGD("Start to malloc host memory, size=%zu.", size);
  GE_CHECK_NOTNULL(VarManager::Instance(session_id));
  uint8_t *const buffer = VarManager::Instance(session_id)->GetHostPoolMemory(RT_MEMORY_HBM, size);
  if (buffer == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "malloc host memory fail, var_size:%" PRIu64 ", var_name:%s",
                      size, var_name.c_str());
    GELOGE(MEMALLOC_FAILED, "[Malloc][HostMemory] for node %s failed, size = %" PRIu64 "", var_name.c_str(), size);
    return MEMALLOC_FAILED;
  }
  const auto ret = memset_s(buffer, size, 0, size);
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "[MemSet][Buffer] failed, ret = %d.", ret);
  mem_offset = static_cast<size_t>(PtrToValue(buffer));
  FMK_UINT64_ADDCHECK(var_mem_size_, size);
  var_mem_size_ += size;
  GELOGI("[IMAS]AssignVarMem Set session_%" PRIu64 " name[%s] output[%zu] size[%lu]",
         session_id, var_name.c_str(), mem_offset, size);
  return SUCCESS;
}

uint64_t MemResource::GetVarMemSize() const {
  return var_mem_size_;
}

void MemResource::UpdateVarMemSize(const int64_t mem_size) {
  var_mem_size_ = static_cast<uint64_t>(mem_size);
};

uint64_t MemResource::GetVarConstPlaceHolderMemSize() const {
  return const_place_holder_mem_size_;
}

VarManager::VarManager(const uint64_t session_id) : session_id_(session_id) {}

std::shared_ptr<VarManager> VarManager::Instance(const uint64_t session_id) {
  GELOGD("VarManager::Instance, session id = %" PRIu64 ".", session_id);
  return VarManagerPool::Instance().GetVarManager(session_id);
}

void VarManager::Destory() {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGI("VarManager::Destory, session id = %" PRIu64 ".", session_id_);
  (void)FreeVarMemory();
  version_ = SessionVersion::OTHER_VERSION;
  device_id_ = kDefaultDeviceId;
  session_id_ = 0U;
  mem_resource_map_.clear();
  var_memory_allocator_ = nullptr;
  LogGraphVarMemInfo();
}

Status VarManager::Init(const uint32_t version, const uint64_t session_id, const uint32_t device_id,
                        const uint64_t job_id) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGI("VarManager::Init, session id = %" PRIu64 " device id = %u.", session_id, device_id);
  if (var_resource_ == nullptr) {
    GE_ASSERT_TRUE(version <= static_cast<uint32_t>(SessionVersion::OTHER_VERSION), "Version [%u] is invalid", version);
    version_ = static_cast<SessionVersion>(version);
    device_id_ = device_id;
    session_id_ = session_id;
    job_id_ = job_id;
    var_resource_ = MakeShared<VarResource>(session_id_);
    if (var_resource_ == nullptr) {
      GELOGW("VarManager init failed session id = %" PRIu64 ".", session_id);
      return ge::INTERNAL_ERROR;
    }
  } else {
    GELOGW("VarManager::has been inited, session id = %" PRIu64 " device id = %u.", session_id, device_id);
  }
  return SUCCESS;
}

const uint64_t &VarManager::SessionId() const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  return session_id_;
}

ge::Status VarManager::SetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc,
                                  const uint8_t *const dev_ptr, const rtMemType_t memory_type,
                                  const OpDescPtr &op_desc) {
  GELOGI("VarManager::SetVarAddr var_name = %s, data_type = %s, data_format = %s.", var_name.c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());

  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  var_resource_->SetVarAddr(var_name, tensor_desc, dev_ptr, memory_type, op_desc);
  return ge::SUCCESS;
}

ge::Status VarManager::GetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc,
                                  uint8_t *&dev_ptr, rtMemType_t &memory_type) const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGD("VarManager::GetVarAddr var_name = %s, data_type = %s, data_format = %s", var_name.c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());

  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  const auto ret = var_resource_->GetVarAddr(var_name, tensor_desc, &dev_ptr, memory_type);
  if (ret != SUCCESS) {
    GELOGW("GetVarAddr fail.");
    return ge::INTERNAL_ERROR;
  }
  return SUCCESS;
}

ge::Status VarManager::GetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc,
                                  uint8_t *&dev_ptr) const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  rtMemType_t memory_type = RT_MEMORY_HBM;
  return GetVarAddr(var_name, tensor_desc, dev_ptr, memory_type);
}

void VarManager::SetExternalVar(void *external_var_addr, uint64_t external_var_size) {
  external_var_addr_ = external_var_addr;
  external_var_size_ = external_var_size;
}

int64_t VarManager::GetVarMemSize(const rtMemType_t memory_type) const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  std::shared_ptr<MemResource> mem_resource = nullptr;
  const auto iter = mem_resource_map_.find(memory_type);
  if (iter == mem_resource_map_.end()) {
    return 0;
  } else {
    mem_resource = iter->second;
  }

  if (mem_resource == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Find no mem_resource in map, memory_type:%u, session_id:%" PRIu64 "",
                       memory_type, session_id_);
    GELOGE(ge::INTERNAL_ERROR, "[Check][Param] MemResource is invalid, memory_type:%u, session_id:%" PRIu64 "",
           memory_type, session_id_);
    return 0;
  }

  if (memory_type == RT_MEMORY_HBM) {
    return static_cast<int64_t>(mem_resource->GetVarMemSize()) - \
           static_cast<int64_t>(mem_resource->GetVarConstPlaceHolderMemSize());
  }
  return static_cast<int64_t>(mem_resource->GetVarMemSize());
}

int64_t VarManager::GetVarConstPlaceHolderMemSize(const rtMemType_t memory_type) const {
  int64_t ret = 0L;
  if (memory_type != RT_MEMORY_HBM) {
    return ret;
  }

  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  const auto iter = mem_resource_map_.find(memory_type);
  if (iter != mem_resource_map_.end() && iter->second != nullptr) {
    ret = static_cast<int64_t>(iter->second->GetVarConstPlaceHolderMemSize());
  }
  return ret;
}

ge::Status VarManager::AssignVarMem(const std::string &var_name, const OpDescPtr &op_desc,
                                    const ge::GeTensorDesc &tensor_desc, rtMemType_t memory_type) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGI("VarManager::AssignVarMem var_name = %s, data_type = %s, data_format = %s.", var_name.c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());

  int64_t tensor_desc_size = 0;
  ge::Status result = TensorUtils::GetSize(tensor_desc, tensor_desc_size);
  if (result != ge::SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Get size from tensor fail, var_name:%s, memory_type:%u, session_id:%" PRIu64 "",
                      var_name.c_str(), memory_type, session_id_);
    GELOGE(result, "[Get][Size] from tensor fail, var_name:%s, memory_type:%u, session_id:%" PRIu64 "",
           var_name.c_str(), memory_type, session_id_);
    return result;
  }

  std::shared_ptr<MemResource> mem_resource = GetOrCreateMemoryResourceByType(memory_type);
  GE_CHECK_NOTNULL(mem_resource);
  GE_CHECK_NOTNULL(var_resource_);

  ge::GeTensorDesc cur_tensor_desc;
  int64_t cur_tensor_desc_size = 0;
  uint8_t *mem_offset = nullptr;
  result = var_resource_->GetCurVarDesc(var_name, cur_tensor_desc);
  // reuse old format variable memory
  if (result == SUCCESS) {
    result = var_resource_->GetVarAddr(var_name, cur_tensor_desc, &mem_offset, memory_type);
    if (result == SUCCESS) {
      result = TensorUtils::GetSize(cur_tensor_desc, cur_tensor_desc_size);
      GELOGD("tensor_desc_size is %ld, cur_tensor_desc_size is %ld, memoffset is %" PRIu64 "", tensor_desc_size,
             cur_tensor_desc_size, PtrToValue(mem_offset));
    }
  } else {
    result = var_resource_->GetReuseAddr(op_desc, &mem_offset, memory_type);
    if (result == SUCCESS) {
      cur_tensor_desc_size = tensor_desc_size;
    }
  }

  const bool can_not_reuse_old_memory = (result != SUCCESS) || (tensor_desc_size > cur_tensor_desc_size);
  if (can_not_reuse_old_memory) {
    size_t tmp_mem_offset = 0UL;
    result = mem_resource->AssignVarMem(var_name, static_cast<uint64_t>(tensor_desc_size),
                                        session_id_, tmp_mem_offset, op_desc);
    if (result != SUCCESS) {
      GELOGE(ge::INTERNAL_ERROR, "[Assign][VarMem] by offset failed, session_id:%" PRIu64 ".", session_id_);
      return ge::INTERNAL_ERROR;
    }

    mem_offset = PtrToPtr<void, uint8_t>(ValueToPtr(tmp_mem_offset));
    result = var_resource_->SaveVarAddr(var_name, tensor_desc, mem_offset, memory_type, op_desc);
    if (result != SUCCESS) {
      GELOGE(ge::INTERNAL_ERROR, "[Save][VarAddr] by offset failed, memory type:%u, session_id:%" PRIu64 ".",
             memory_type, session_id_);
      return ge::INTERNAL_ERROR;
    }
  }
  // old does not exist only save new tensor
  result = var_resource_->GetCurVarDesc(var_name, cur_tensor_desc);
  if (result != SUCCESS) {
    var_resource_->SetVarAddr(var_name, tensor_desc, mem_offset, memory_type, op_desc);
    return SUCCESS;
  }
  const bool format_changed = (cur_tensor_desc.GetFormat() != tensor_desc.GetFormat()) ||
                              (cur_tensor_desc.GetDataType() != tensor_desc.GetDataType()) ||
                              (cur_tensor_desc.GetShape().GetDims() != tensor_desc.GetShape().GetDims());
  if (format_changed) {
    GELOGI("var %s assigned new memory (format, data type, shape) (%s, %s, %zu) from (%s, %s, %zu)", var_name.c_str(),
           ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
           ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str(),
           tensor_desc.GetShape().GetDims().size(),
           ge::TypeUtils::DataTypeToSerialString(cur_tensor_desc.GetDataType()).c_str(),
           ge::TypeUtils::FormatToSerialString(cur_tensor_desc.GetFormat()).c_str(),
           cur_tensor_desc.GetShape().GetDims().size());
    var_resource_->SetVarAddr(var_name, tensor_desc, mem_offset, memory_type, op_desc);
  }

  return SUCCESS;
}

Status VarManager::RestoreVarMem(const std::string &var_name, const OpDescPtr &op_desc, const GeTensorDesc &tensor_desc,
                                 rtMemType_t memory_type) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGI("VarManager::RestoreVarMem var_name = [%s], data_type = [%s], format = [%s].", var_name.c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());
  GE_CHECK_NOTNULL(var_resource_);
  int64_t tensor_desc_size{-1};
  GE_ASSERT_SUCCESS(TensorUtils::GetSize(tensor_desc, tensor_desc_size),
                    "Get size from tensor fail, var_name:%s, memory_type:%d, session_id:%" PRIu64 "", var_name.c_str(),
                    memory_type, session_id_);
  GE_ASSERT_TRUE(tensor_desc_size >= 0, "Var [%s]'s tensor_size [%ld] is invalid.", var_name.c_str(), tensor_desc_size);

  uint8_t *mem_offset{nullptr};
  ge::GeTensorDesc cur_tensor_desc;
  Status result = var_resource_->GetCurVarDesc(var_name, cur_tensor_desc);
  if (result == SUCCESS) {
    result = var_resource_->GetVarAddr(var_name, cur_tensor_desc, &mem_offset, memory_type);
  }
  if (result == SUCCESS) {
    const bool format_changed = (cur_tensor_desc.GetFormat() != tensor_desc.GetFormat()) ||
                                (cur_tensor_desc.GetDataType() != tensor_desc.GetDataType()) ||
                                (cur_tensor_desc.GetShape().GetDims() != tensor_desc.GetShape().GetDims());
    if (format_changed) {
      int64_t cur_tensor_desc_size{0};
      GE_ASSERT_SUCCESS(TensorUtils::GetSize(cur_tensor_desc, cur_tensor_desc_size));
      GE_ASSERT_TRUE(tensor_desc_size <= cur_tensor_desc_size,
                     "The format of [%s] is changed, and the new tensor size [%ld] exceeds the size [%d] of the tensor "
                     "in original "
                     "format, the compute graph needs to be recompiled.",
                     var_name.c_str(), tensor_desc_size, cur_tensor_desc_size);
      GELOGI("var [%s] reuse addr (format, data type) (%s, %s) from (%s, %s)", var_name.c_str(),
             ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
             ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str(),
             ge::TypeUtils::DataTypeToSerialString(cur_tensor_desc.GetDataType()).c_str(),
             ge::TypeUtils::FormatToSerialString(cur_tensor_desc.GetFormat()).c_str());

      var_resource_->SetVarAddr(var_name, tensor_desc, mem_offset, memory_type, op_desc);
      return SUCCESS;
    }
  } else {
    std::vector<int64_t> output_offset = op_desc->GetOutputOffset();
    GE_ASSERT(!output_offset.empty());
    const int64_t logic_offset = output_offset[0UL] - static_cast<int64_t>(var_mem_logic_base_);
    GE_ASSERT_TRUE(logic_offset >= 0, "The output offset [%ld] of [%s] is invalid.", output_offset[0UL],
                   var_name.c_str());
    const uint64_t total_size = static_cast<uint64_t>(logic_offset) + static_cast<uint64_t>(tensor_desc_size);
    GE_ASSERT_TRUE(total_size <= var_mem_max_size_,
                   "Variable offset overflow max_size:[%lu], offset:[%ld], tensor_size:[%ld].", var_mem_max_size_,
                   logic_offset, tensor_desc_size);
    uint8_t *mem_addr = nullptr;
    mem_addr = PtrToPtr<void, uint8_t>(ValueToPtr(static_cast<uint64_t>(logic_offset)));
    GE_ASSERT_SUCCESS(var_resource_->SaveVarAddr(var_name, tensor_desc, mem_addr, memory_type, op_desc),
                      "[Save][VarAddr] by offset failed, memory type:%u, session_id:%" PRIu64 ".", memory_type,
                      session_id_);
    var_resource_->SetVarAddr(var_name, tensor_desc, mem_addr, memory_type, op_desc);
  }
  return SUCCESS;
}

void VarManager::SetVarIsReady(const std::string &var_name,
                               const ge::GeTensorDesc &tensor_desc,
                               const uint32_t device_id) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGD("VarManager::SetVarIsReady var_name = %s, data_type = %s, data_format = %s", var_name.c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());

  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return;
  }
  var_resource_->SetVarIsReady(var_name, tensor_desc, device_id);
}

bool VarManager::IsVarReady(const std::string &var_name,
                            const ge::GeTensorDesc &tensor_desc,
                            const uint32_t device_id) const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGD("VarManager::IsVarReady var_name = %s, data_type = %s, data_format = %s", var_name.c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());

  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return false;
  }
  return var_resource_->IsVarReady(var_name, tensor_desc, device_id);
}

bool VarManager::IsVarExist(const std::string &var_name, const ge::GeTensorDesc &tensor_desc) const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGD("VarManager::IsVarExist var_name = %s, data_type = %s, data_format = %s", var_name.c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());

  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return false;
  }
  return var_resource_->IsVarExist(var_name, tensor_desc);
}

bool VarManager::IsVarExist(const std::string &var_name) const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return false;
  }
  return var_resource_->IsVarExist(var_name);
}

ge::Status VarManager::VarDescInfoToSerial(const uint64_t session_id, deployer::VarDescInfo &info) const {
  GELOGD("[VarManager] Begin to serial var desc info, the session id is %" PRIu64 ".", session_id);
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGE(INTERNAL_ERROR, "Var manager has not been inited.");
    return INTERNAL_ERROR;
  }

  return var_resource_->VarDescInfoToSerial(info);
}

ge::Status VarManager::VarManagerToSerial(const uint64_t session_id, deployer::VarManagerInfo &info) const {
  GELOGD("[VarManager] Begin to serial var manager objection, the session id is %" PRIu64 ".", session_id);
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGE(INTERNAL_ERROR, "Var manager has not been inited.");
    return INTERNAL_ERROR;
  }

  info.set_version(static_cast<uint32_t>(version_));
  info.set_session_id(session_id_);
  info.set_device_id(device_id_);
  info.set_job_id(job_id_);
  info.set_graph_mem_max_size(graph_mem_max_size_);
  info.set_var_mem_max_size(var_mem_max_size_);
  info.set_var_mem_logic_base(var_mem_logic_base_);
  info.set_use_max_mem_size(use_max_mem_size_);
  deployer::VarResourceInfo *const var_resource_info = info.mutable_var_resource();
  (void)var_resource_->VarResourceToSerial(var_resource_info);

  auto const resource_map = info.mutable_mem_resource_map();
  for (auto &mem_resource : mem_resource_map_) {
    deployer::MemResourceInfo source_info;
    if (mem_resource.second == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Find no mem_resource in map, memory_type:%u, session_id:"
		         "%" PRIu64 ".", mem_resource.first, session_id_);
      GELOGE(ge::INTERNAL_ERROR, "[Check][Param] MemResource is invalid, memory_type:%u, "
		                 "session_id:%" PRIu64 ".", mem_resource.first, session_id_);
      return INTERNAL_ERROR;
    }
    source_info.set_var_mem_size(mem_resource.second->GetVarMemSize());
    (void)resource_map->insert({mem_resource.first, source_info});
  }
  GELOGD("[VarManager] Success to serial var manager objection, the session id is %" PRIu64 ".", session_id);
  return SUCCESS;
}

ge::Status VarManager::VarManagerToDeserial(const uint64_t session_id, const deployer::VarManagerInfo &info) {
  GELOGD("[VarManager] Begin to deserial var manager objection, the session id is %" PRIu64 ".", session_id);
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    version_ = static_cast<SessionVersion>(info.version());
    int32_t device_id = -1;
    GE_CHK_RT_RET(rtGetDevice(&device_id));
    device_id_ = static_cast<uint32_t>(device_id);
    GELOGD("[VarManager] Success to get device id = %u.", device_id_);
    session_id_ = info.session_id();
    job_id_ = info.job_id();
    UpdateMemoryConfig(info.graph_mem_max_size(), info.var_mem_max_size(), info.var_mem_logic_base(),
                       info.use_max_mem_size());
    var_resource_ = MakeShared<VarResource>(session_id_);
    if (var_resource_ == nullptr) {
      GELOGE(ge::INTERNAL_ERROR, "VarManager init failed session id = %" PRIu64 ".", session_id);
      return ge::INTERNAL_ERROR;
    }
  }

  (void)var_resource_->VarResourceToDeserial(&info.var_resource());

  for (const auto &x : info.mem_resource_map()) {
    const rtMemType_t memory_type = x.first;
    std::shared_ptr<MemResource> mem_resource = GetOrCreateMemoryResourceByType(memory_type);
    GE_ASSERT_NOTNULL(mem_resource);
    mem_resource->UpdateVarMemSize(static_cast<int64_t>(x.second.var_mem_size()));
  }
  GELOGD("[VarManager] Success to deserial var manager objection, the session id is "
         "%" PRIu64 ".", session_id);
  return SUCCESS;
}

ge::Status VarManager::GetCurVarDesc(const std::string &var_name, ge::GeTensorDesc &tensor_desc) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGI("VarManager::GetCurVarDesc var_name = %s.", var_name.c_str());

  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  return var_resource_->GetCurVarDesc(var_name, tensor_desc);
}

ge::Status VarManager::SaveBroadCastInfo(const uint32_t graph_id, const VarBroadCastInfo &broad_cast_info) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGI("VarManager::SaveBroadCastInfo var_name = %s, broadcast name = %s, "
         "idx = %d, input_offset = %ld, input_size = %" PRIu64 ", output_offset = %ld, output_size = %" PRIu64 "",
         broad_cast_info.var_name.c_str(), broad_cast_info.broadcast_name.c_str(), broad_cast_info.idx,
         broad_cast_info.input_offset, broad_cast_info.input_size, broad_cast_info.output_offset,
         broad_cast_info.output_size);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  var_resource_->SaveBroadCastInfo(graph_id, broad_cast_info);
  return SUCCESS;
}

ge::Status VarManager::RenewCurVarDesc(const std::string &var_name, const GeTensorDesc &tensor_desc) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGD("Begin to renew current var desc, var_name = %s.", var_name.c_str());

  if (var_resource_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "VarManager has not been init, session_id:%" PRIu64 ", check invalid", session_id_);
    GELOGE(ge::INTERNAL_ERROR, "[Check][Param] VarManager has not been init, session_id:%" PRIu64 "", session_id_);
    return ge::INTERNAL_ERROR;
  }
  return var_resource_->RenewCurVarDesc(var_name, tensor_desc);
}

ge::Status VarManager::RecordStagedVarDesc(const uint32_t graph_id,
                                           const std::string &var_name,
                                           const GeTensorDesc &tensor_desc) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGD("Begin to record staged var desc, var_name = %s.", var_name.c_str());

  if (var_resource_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "VarManager has not been init, session_id:%" PRIu64 ", check invalid", session_id_);
    GELOGE(ge::INTERNAL_ERROR, "[Check][Param] VarManager has not been init, session_id:%" PRIu64 "", session_id_);
    return ge::INTERNAL_ERROR;
  }
  return var_resource_->RecordStagedVarDesc(graph_id, var_name, tensor_desc);
}

const std::map<std::string, GeTensorDesc> &VarManager::GetStagedVarDescs(const uint32_t graph_id) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    static std::map<std::string, GeTensorDesc> empty;
    REPORT_INNER_ERR_MSG("E19999", "VarManager has not been init, session_id:%" PRIu64 ", check invalid", session_id_);
    GELOGE(ge::INTERNAL_ERROR, "[Check][Param] VarManager has not been init, session_id:%" PRIu64 "", session_id_);
    return empty;
  }
  return var_resource_->GetStagedVarDescs(graph_id);
}

ge::Status VarManager::RenewCurVarDesc(const std::string &var_name, ge::OpDescPtr op_desc) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGD("VarManager::RenewCurVarDesc var_name = %s.", var_name.c_str());

  if (var_resource_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "VarManager has not been init, op:%s(%s), session_id:%" PRIu64 ", check invalid",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                       session_id_);
    GELOGE(ge::INTERNAL_ERROR, "[Check][Param] VarManager has not been init, op:%s(%s), session_id:%" PRIu64 "",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), session_id_);
    return ge::INTERNAL_ERROR;
  }
  return var_resource_->RenewCurVarDesc(var_name, std::move(op_desc));
}

bool VarManager::IsVarAddr(const int64_t offset) const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGD("VarManager has not been init.");
    return false;
  }
  return var_resource_->IsVarAddr(offset);
}

bool VarManager::CheckAndSetVarLoaded(const OpDescPtr &op_desc, const uint32_t device_id) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGD("VarManager has not been init.");
    return false;
  }
  int64_t inner_offset = 0;
  (void)AttrUtils::GetInt(op_desc->GetOutputDesc(0U), ATTR_NAME_INNER_OFFSET, inner_offset);
  const auto &v_output_offset = op_desc->GetOutputOffset();
  GE_ASSERT_TRUE(!v_output_offset.empty(), "Output param is empty, node:%s", op_desc->GetName().c_str());
  const auto offset = v_output_offset[0] - inner_offset;
  std::string loaded_var_name;
  if (IsVarAddr(offset) && var_resource_->IsVarLoaded(device_id, offset, loaded_var_name)) {
    GE_ASSERT_TRUE(!loaded_var_name.empty());
    GELOGI("Constant op[%s] can reuse the memory address of constant op[%s], skip load again",
           op_desc->GetName().c_str(), loaded_var_name.c_str());
    return true;
  }
  var_resource_->SetVarLoaded(device_id, op_desc->GetName(), offset);
  return false;
}

rtMemType_t VarManager::GetVarMemType(const int64_t offset) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return RT_MEMORY_RESERVED;
  }
  return var_resource_->GetVarMemType(offset);
}

void VarManager::SetMemManager(MemoryManager *const mem_manager) {
  // Better use shared_ptr instead, reconsitution later.
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  mem_manager_ = mem_manager;
}

Status VarManager::InitExpandableMemoryAllocator(ExpandableMemoryAllocatorPtr var_memory_allocator) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if ((var_memory_allocator_ == nullptr) && (var_memory_allocator != nullptr)
      && (var_memory_allocator->IsSupportExpandableMemory())) {
    var_memory_allocator->SetReuse(false);
    // 单独物理内存池，session析构时物理内存池也析构
    var_memory_allocator->SetSharePhyAllocator(false);
    var_memory_allocator_ = std::move(var_memory_allocator);
  }
  return SUCCESS;
}

uint8_t *VarManager::GetVarMemoryAddr(const uint8_t *const logic_addr, const rtMemType_t memory_type,
                                      const uint32_t device_id) {
  std::string graph_name;
  return GetVarMemoryAddr(graph_name, logic_addr, memory_type, device_id);
}

uint8_t *VarManager::GetVarMemoryAddr(const std::string &graph_name,
                                      const uint8_t *const logic_addr,
                                      const rtMemType_t memory_type,
                                      const uint32_t device_id) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (mem_manager_ == nullptr) {
    GELOGE(FAILED, "[Check][Param] MemManager has not been init.");
    REPORT_INNER_ERR_MSG("E19999", "MemManager has not been init, session_id: %" PRIu64 "", session_id_);
    return nullptr;
  }

  if (memory_type == RT_MEMORY_RDMA_HBM) {
    return const_cast<uint8_t *>(logic_addr);
  }

  if ((external_var_addr_ != nullptr) && (memory_type == RT_MEMORY_HBM)) {
    const uint64_t inner_var_size = static_cast<uint64_t>(GetVarMemSize(RT_MEMORY_HBM));
    GE_ASSERT_TRUE(external_var_size_ >= inner_var_size, "external var size %ld can not be smaller than %ld",
                   external_var_size_, inner_var_size);
    GE_ASSERT_TRUE(PtrToValue(logic_addr) >= var_mem_logic_base_, "logic offset %lu can not be smaller than logic base %lu",
                   PtrToValue(logic_addr), var_mem_logic_base_);
    const uint64_t real_offset = PtrToValue(logic_addr) - var_mem_logic_base_;
    GE_ASSERT_TRUE(real_offset < external_var_size_, "real offset %lu should be smaller than external var size %lu",
                   real_offset, external_var_size_);
    uint8_t *ret_p = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(external_var_addr_) + real_offset));
    GELOGI("external var scene, real_offset is %lu, logic offset %lu, base %lu, ret_p %p",
           real_offset, PtrToValue(logic_addr), var_mem_logic_base_, ret_p);
    return ret_p;
  }

  return GetAutoMallocVarAddr(graph_name, logic_addr, memory_type, device_id);
}

ge::Status VarManager::GetVarMallocSize(const ge::GeTensorDesc &var_desc, int64_t &malloc_size) {
  (void)TensorUtils::GetSize(var_desc, malloc_size);
  malloc_size = (malloc_size + static_cast<int64_t>(kSessionMemAlignSize) - 1) /
      static_cast<int64_t>(kSessionMemAlignSize) * static_cast<int64_t>(kSessionMemAlignSize);
  // align 1024 BYTE
  GE_CHK_STATUS_RET(CheckUint64AddOverflow(static_cast<uint64_t>(malloc_size),
                                           kSessionMemAlignSize * kSessionMemAlignUnit),
                    "Add var size:%lu uint64 overflow.", static_cast<uint64_t>(malloc_size));
  malloc_size = malloc_size +
      static_cast<int64_t>(kSessionMemAlignSize) * static_cast<int64_t>(kSessionMemAlignUnit);
  return SUCCESS;
}

uint8_t *VarManager::GetAutoMallocVarAddr(const std::string &graph_name,
                                          const uint8_t *const logic_addr,
                                          const rtMemType_t memory_type,
                                          const uint32_t device_id) {
  GE_ASSERT_NOTNULL(var_resource_, "VarResource has not been init, session_id: %lu", session_id_);
  GELOGD("[Variable][Addr] Begin malloc memory for var.");
  var_resource_->UpdateDevVarMgrInfo(device_id);
  auto var_mgr = var_resource_->GetVarMgrInfo(device_id, static_cast<int64_t>(PtrToValue(logic_addr)));
  uint64_t inner_offset_tmp = 0U;
  uint64_t logic_addr_tmp = PtrToValue(logic_addr);
  if (var_mgr == nullptr) {
    const auto ret = var_resource_->CheckLogicAddrVaild(device_id, logic_addr, inner_offset_tmp, logic_addr_tmp);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "[Check][Param] Logic addr info is not assign.");
      return nullptr;
    }
    var_mgr = var_resource_->GetVarMgrInfo(device_id, static_cast<int64_t>(logic_addr_tmp));
  }
  // same var no need to re-malloc
  if (var_mgr->dev_addr != nullptr) {
    if (!graph_name.empty()) {
      auto &mem_statistic = graph_var_mem_statistic_[graph_name];
      const auto &it = mem_statistic.dev_addrs.find(var_mgr->dev_addr);
      int64_t variable_size = 0;
      const bool is_shared_mem =
          (GetVarMallocSize(var_mgr->tensor_desc, variable_size) == SUCCESS) && (it == mem_statistic.dev_addrs.cend());
      if (is_shared_mem) {
        const uint64_t add_size = static_cast<uint64_t>(variable_size);
        GE_ASSERT_TRUE(mem_statistic.shared_size <= (UINT64_MAX - add_size), "Var shared_size overflow.");
        mem_statistic.shared_size += add_size;
      }
    }

    if (var_mgr->is_extern_mem) {
        return reinterpret_cast<uint8_t *>(var_mgr->dev_addr);
    }
    return reinterpret_cast<uint8_t *>(PtrToValue(var_mgr->dev_addr) + inner_offset_tmp + kSessionMemAlignSize);
  }

  // malloc for the variable
  uint8_t *var_dev_addr = nullptr;
  int64_t variable_size = 0;
  if (GetVarMallocSize(var_mgr->tensor_desc, variable_size) != SUCCESS) {
    GELOGE(FAILED, "Failed to get var malloc size");
    return nullptr;
  }

  const std::string purpose("Auto malloc var and constant op memory");
  if (var_memory_allocator_ != nullptr) {
    var_dev_addr = var_memory_allocator_->MallocMemory(purpose, variable_size, true);
  }
  if ((var_dev_addr == nullptr) && (!IsVariableUse1gHugePageOnly())) {
    // var_memory_allocator 不支持扩展模式
    var_memory_allocator_ = nullptr;
    var_dev_addr = mem_manager_->MallocMemory(memory_type, purpose, static_cast<size_t>(variable_size), device_id);
  }
  if (var_dev_addr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "[Malloc] Malloc var caching mem failed. size:%" PRId64 "", variable_size);
    GELOGE(FAILED, "[Malloc] Malloc var caching mem failed. size:%ld", variable_size);
    return nullptr;
  }

  var_malloc_mem_size_ += variable_size;
  (void)var_resource_->SetVarMgrDevAddr(device_id, static_cast<int64_t>(logic_addr_tmp), var_dev_addr);
  if (!graph_name.empty()) {
    graph_var_mem_statistic_[graph_name].alloc_size += static_cast<uint64_t>(variable_size);
    (void)graph_var_mem_statistic_[graph_name].dev_addrs.emplace(var_dev_addr);
  }

  const uint64_t addr_val = PtrToValue(var_dev_addr);
  GE_ASSERT_TRUE(addr_val <= (UINT64_MAX - inner_offset_tmp), "var_dev_addr_tmp overflow.");
  const uint64_t temp_addr = addr_val + inner_offset_tmp;
  GE_ASSERT_TRUE(temp_addr <= (UINT64_MAX - kSessionMemAlignSize), "var_dev_addr_tmp overflow.");
  uint8_t *var_dev_addr_tmp = PtrToPtr<void, uint8_t>(ValueToPtr(temp_addr + kSessionMemAlignSize));

  GE_ASSERT_SUCCESS(InitVarIfHasInitValue(var_mgr, reinterpret_cast<void*>(var_dev_addr_tmp)));
  GELOGI("[Variable][Addr] Malloc var addr success.");
  return var_dev_addr_tmp;
}

void VarManager::LogGraphVarMemInfo() const {
  for (const auto &it : graph_var_mem_statistic_) {
    const auto &graph_name = it.first;
    const auto &mem_info = it.second;
    GEEVENT("model_metrics:name=%s, alloc_var_mem=%lu B, shared_var_mem=%lu B",
            graph_name.c_str(), mem_info.alloc_size, mem_info.shared_size);
  }
}

std::shared_ptr<MemResource> VarManager::GetOrCreateMemoryResourceByType(rtMemType_t memory_type) {
  const auto it = mem_resource_map_.find(memory_type);
  if (it != mem_resource_map_.end()) {
    return it->second;
  }
  auto mem_resource = MemResource::BuildMemResourceFromType(memory_type);
  GE_ASSERT_NOTNULL(mem_resource);
  mem_resource_map_[memory_type] = mem_resource;
  return mem_resource;
}

ge::Status VarManager::FreeVarMemory() {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (mem_manager_ == nullptr) {
    GELOGW("MemManager is nullptr please check if it has been initialized.");
    return FAILED;
  }

  if (var_memory_allocator_ != nullptr) {
    (void)var_memory_allocator_->FreeMemory();
  }
  if (var_resource_ == nullptr) {
    GELOGD("VarResource is nullptr no need to free var manager.");
    return SUCCESS;
  }
  for (const auto &dev_and_infos : var_resource_->GetAllDevVarMgrInfo()) {
    const uint32_t device_id = dev_and_infos.first;
    const auto &infos = dev_and_infos.second;
    for (const auto &info : infos) {
      if (info.second.dev_addr == nullptr || info.second.is_extern_mem) {
        continue;
      }
      if (var_memory_allocator_ == nullptr) {
        (void) mem_manager_->FreeMemory(RT_MEMORY_HBM, PtrToPtr<uint8_t, void>(info.second.dev_addr), device_id);
      }
      (void)var_resource_->SetVarMgrDevAddr(device_id, static_cast<int64_t>(info.first), nullptr);
    }
  }
  var_malloc_mem_size_ = 0L;
  var_memory_allocator_ = nullptr;
  return SUCCESS;
}

uint8_t *VarManager::GetRdmaPoolMemory(const rtMemType_t memory_type, const size_t mem_size) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (mem_manager_ == nullptr) {
    GELOGE(FAILED, "[Check][Param] MemManager has not been init.");
    REPORT_INNER_ERR_MSG("E19999", "MemManager has not been init, session_id: %" PRIu64 "", session_id_);
    return nullptr;
  }

  return mem_manager_->GetRdmaPoolMemory(memory_type, mem_size, device_id_);
}

uint8_t *VarManager::GetHostPoolMemory(const rtMemType_t memory_type, const size_t mem_size) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (mem_manager_ == nullptr) {
    GELOGE(FAILED, "[Check][Param] MemManager has not been init.");
    REPORT_INNER_ERR_MSG("E19999", "MemManager has not been init, session_id: %" PRIu64 "", session_id_);
    return nullptr;
  }

  return mem_manager_->GetHostPoolMemory(memory_type, mem_size);
}

ge::Status VarManager::SetTransRoad(const std::string &var_name, const VarTransRoad &trans_road) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  return var_resource_->SetTransRoad(var_name, trans_road);
}

VarTransRoad *VarManager::GetTransRoad(const std::string &var_name) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return nullptr;
  }
  return var_resource_->GetTransRoad(var_name);
}

Status VarManager::SetChangedGraphId(const std::string &var_name, const uint32_t graph_id) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return INTERNAL_ERROR;
  }
  return var_resource_->SetChangedGraphId(var_name, graph_id);
}

Status VarManager::GetChangedGraphId(const std::string &var_name, uint32_t &graph_id) const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return INTERNAL_ERROR;
  }
  return var_resource_->GetChangedGraphId(var_name, graph_id);
}

std::set<std::string> VarManager::GetChangedVarNames(const uint32_t graph_id) const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return std::set<std::string>();
  }
  return var_resource_->GetChangedVarNames(graph_id);
}

void VarManager::UpdateMemoryConfig(const size_t graph_mem_max_size, const size_t var_mem_max_size,
                                    const size_t var_mem_logic_base, const size_t use_max_mem_size) {
  graph_mem_max_size_ = graph_mem_max_size;
  var_mem_max_size_ = var_mem_max_size;
  var_mem_logic_base_ = var_mem_logic_base;
  use_max_mem_size_ = use_max_mem_size;
}

Status VarManager::SetAllMemoryMaxValue(const std::map<std::string, std::string> &options) {
  (void)options;
  GEEVENT("The graph_mem_max_size is %zu and the var_mem_max_size is %zu", graph_mem_max_size_, var_mem_max_size_);

  FMK_SIZET_ADDCHECK(graph_mem_max_size_, kGraphMemoryBuffer);
  // set var memory logic base as a fix value 128G
  // offline scenario with fileconstant may compile and execute in different soc version
  var_mem_logic_base_ = kVarMemoryLogicBase;

  FMK_SIZET_ADDCHECK(graph_mem_max_size_, var_mem_max_size_);
  use_max_mem_size_ = graph_mem_max_size_ + var_mem_max_size_;
  if (use_max_mem_size_ > kMaxMemorySize) {
    REPORT_INNER_ERR_MSG("E19999", "all mem_use size:%" PRIu64 " can not exeed limit:%" PRIu64 ", "
		       "session_id:%" PRIu64 ", check invalid", use_max_mem_size_, kMaxMemorySize, session_id_);
    GELOGE(ge::GE_GRAPH_OPTIONS_INVALID, "[Check][Param] kUseMaxMemorySize:%zu can not exceed "
           "max memory size:%zu, session_id:%" PRIu64 ".", use_max_mem_size_, kMaxMemorySize, session_id_);
    return ge::GE_GRAPH_OPTIONS_INVALID;
  }
  GELOGI("Set memory malloc size successfully");
  return SUCCESS;
}

Status VarManager::SetMemoryMallocSize(const std::map<std::string, std::string> &options, const size_t total_mem_size) {
  GEEVENT("Total memory size is %zu", total_mem_size);
  use_max_mem_size_ = static_cast<size_t>(
      floor(static_cast<float64_t>(total_mem_size) * (kGraphMemoryManagerMallocRatio + kVarMemoryManagerMallocRatio)));
  graph_mem_max_size_ = static_cast<size_t>(
      floor(static_cast<float64_t>(total_mem_size) * kGraphMemoryManagerMallocRatio));
  var_mem_max_size_ = static_cast<size_t>(floor(static_cast<float64_t>(total_mem_size) * kVarMemoryManagerMallocRatio));

  return SetAllMemoryMaxValue(options);
}

void VarManager::RemoveChangedGraphId(const std::string &var_name) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return;
  }
  var_resource_->RemoveChangedGraphId(var_name);
}

Status VarManager::SetAllocatedGraphId(const std::string &var_name, const uint32_t graph_id) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return INTERNAL_ERROR;
  }
  return var_resource_->SetAllocatedGraphId(var_name, graph_id);
}

Status VarManager::GetAllocatedGraphId(const std::string &var_name, uint32_t &graph_id) const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return INTERNAL_ERROR;
  }
  return var_resource_->GetAllocatedGraphId(var_name, graph_id);
}

Status VarManager::GetAllVariables(std::map<std::string, GeTensorDesc> &all_variables) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been inited.");
    return INTERNAL_ERROR;
  }
  auto new_variable_desc = var_resource_->GetAllVarDesc();
  if (new_variable_desc.size() == 0U) {
    GELOGW("VarManager don't have variables.");
    return INTERNAL_ERROR;
  }

  for (auto iter = new_variable_desc.begin(); iter != new_variable_desc.end(); ++iter) {
    const auto trans_road = var_resource_->GetTransRoad(iter->first);
    if ((trans_road == nullptr) || trans_road->empty()) {
      GELOGI("The variable %s does not have any trans road", iter->first.c_str());
      all_variables[iter->first] = iter->second;
    } else {
      // get origin trans info : the first trans node info
      all_variables[iter->first] = trans_road->at(0U).input;
    }
  }
  return SUCCESS;
}

void VarManager::SetBatchVariablesKeyName(const std::string &batch_var_name, const std::string &key_name) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been inited.");
    return;
  }
  var_resource_->SetBatchVariablesKeyName(batch_var_name, key_name);
}

bool VarManager::HasSharedVarMemBetweenBatch() const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been inited.");
    return false;
  }
  return var_resource_->HasSharedVarMemBetweenBatch();
}

bool VarManager::HasMemoryManager() const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  return mem_manager_ != nullptr;
}

VarManagerPool::~VarManagerPool() { Destory(); }

VarManagerPool &VarManagerPool::Instance() {
  static VarManagerPool var_manager_pool;
  return var_manager_pool;
}

void VarManagerPool::Destory() noexcept {
  const std::lock_guard<std::mutex> lock(var_manager_mutex_);
  for (auto &it : var_manager_map_) {
    if (it.second != nullptr) {
      it.second->Destory();
    }
  }
  var_manager_map_.clear();
}

std::shared_ptr<VarManager> VarManagerPool::GetVarManager(const uint64_t session_id) {
  const std::lock_guard<std::mutex> lock(var_manager_mutex_);
  const auto it = var_manager_map_.find(session_id);
  if (it != var_manager_map_.end()) {
    return it->second;
  }

  const std::shared_ptr<VarManager> var_manager = MakeShared<VarManager>(session_id);
  if (var_manager == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New VarManager fail, session_id:%" PRIu64 "", session_id);
    GELOGE(INTERNAL_ERROR, "[New][VarManager] fail, session_id:%" PRIu64 "", session_id);
    return nullptr;
  }
  var_manager_map_[session_id] = var_manager;
  return var_manager;
}

void VarManagerPool::RemoveVarManager(const uint64_t session_id) {
  std::shared_ptr<VarManager> var_manager = nullptr;
  {
    const std::lock_guard<std::mutex> lock(var_manager_mutex_);
    const auto it = var_manager_map_.find(session_id);
    if (it != var_manager_map_.end()) {
      var_manager = it->second;
      (void)var_manager_map_.erase(it);
    }
  }

  if (var_manager != nullptr) {
    var_manager->Destory();
  }
}
}  // namespace ge

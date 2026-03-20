/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <regex>
#include "om2_codegen_utils.h"
#include "common/ge_common/debug/ge_log.h"
#include "common/om2/codegen/om2_codegen_types.h"
#include "graph/utils/op_type_utils.h"
#include "graph/op_desc.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/taskdown_common.h"
#include "graph/debug/ge_op_types.h"


namespace ge {
namespace {
const std::regex kOpNameInvalidRegex("[./]");
}
std::string Om2CodegenUtils::GetKernelNameWithExtension(const std::string &kernel_name) {
  const auto pos = kernel_name.find("__kernel");
  if (pos != std::string::npos) {
    return kernel_name.substr(0, pos) + ".o";
  }
  return kernel_name + ".o";
}

std::string Om2CodegenUtils::GetOpName(const OpDescPtr &op_desc) {
  std::string origin_op_name = op_desc->GetName();
  return std::regex_replace(origin_op_name, kOpNameInvalidRegex, "_");
}

ge::Status Om2CodegenUtils::GetMagic(const OpDescPtr &op_desc, std::string &magic) {
  std::string json_string;
  const std::string *json_string_ptr = AttrUtils::GetStr(op_desc, TVM_ATTR_NAME_MAGIC);
  if (json_string_ptr != nullptr) {
    GELOGI("[OM2] Get json_string of tvm_magic from op_desc.");
    json_string = *json_string_ptr;
  }
  static const std::unordered_map<std::string, std::string> rt_to_acl_magic = {
      {"RT_DEV_BINARY_MAGIC_ELF", "ACL_RT_BINARY_MAGIC_ELF_AICORE"},
      {"RT_DEV_BINARY_MAGIC_ELF_AIVEC", "ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE"},
      {"RT_DEV_BINARY_MAGIC_ELF_AICUBE", "ACL_RT_BINARY_MAGIC_ELF_CUBE_CORE"},
  };
  if (json_string == "RT_DEV_BINARY_MAGIC_ELF_AICPU" || json_string == "RT_DEV_BINARY_MAGIC_ELF" ||
      json_string == "RT_DEV_BINARY_MAGIC_ELF_AIVEC" || json_string == "RT_DEV_BINARY_MAGIC_ELF_AICUBE") {
    magic =
        (rt_to_acl_magic.find(json_string) == rt_to_acl_magic.end()) ? json_string : rt_to_acl_magic.at(json_string);
  } else {
    GELOGE(PARAM_INVALID, "[OM2][Check][JsonStr]Attr:%s in op:%s(%s), value:%s check invalid", TVM_ATTR_NAME_MAGIC.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), json_string.c_str());
    REPORT_INNER_ERR_MSG("E19999", "Attr:%s in op:%s(%s), value:%s check invalid", TVM_ATTR_NAME_MAGIC.c_str(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str(), json_string.c_str());
    return PARAM_INVALID;
  }

  return SUCCESS;
}

bool Om2CodegenUtils::IsSupportedTask(ModelTaskType model_task_type) {
  return model_task_type == ModelTaskType::MODEL_TASK_KERNEL ||
         model_task_type == ModelTaskType::MODEL_TASK_ALL_KERNEL ||
         model_task_type == ModelTaskType::MODEL_TASK_VECTOR_KERNEL ||
         model_task_type == ModelTaskType::MODEL_TASK_VECTOR_ALL_KERNEL ||
         model_task_type == ModelTaskType::MODEL_TASK_END_GRAPH ||
         model_task_type == ModelTaskType::MODEL_TASK_FUSION_START ||
         model_task_type == ModelTaskType::MODEL_TASK_FUSION_END;
}

bool Om2CodegenUtils::RequireBinaryKernel(const ModelTaskType model_task_type) {
  return model_task_type == ModelTaskType::MODEL_TASK_KERNEL ||
         model_task_type == ModelTaskType::MODEL_TASK_ALL_KERNEL ||
         model_task_type == ModelTaskType::MODEL_TASK_VECTOR_KERNEL ||
         model_task_type == ModelTaskType::MODEL_TASK_VECTOR_ALL_KERNEL ||
         model_task_type == ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL;
}

bool Om2CodegenUtils::RequireArgsTable(ModelTaskType model_task_type) {
  return model_task_type == ModelTaskType::MODEL_TASK_KERNEL ||
         model_task_type == ModelTaskType::MODEL_TASK_ALL_KERNEL ||
         model_task_type == ModelTaskType::MODEL_TASK_VECTOR_KERNEL ||
         model_task_type == ModelTaskType::MODEL_TASK_VECTOR_ALL_KERNEL ||
         model_task_type == ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL;
}

bool Om2CodegenUtils::IsAllKernel(const ModelTaskType task_type) {
  return (task_type == ModelTaskType::MODEL_TASK_ALL_KERNEL) ||
         (task_type == ModelTaskType::MODEL_TASK_VECTOR_ALL_KERNEL);
}

bool Om2CodegenUtils::IsAICoreKernel(const ge::ccKernelType kernel_type) {
  static std::set<ge::ccKernelType> aicore_kernel_type{ge::ccKernelType::TE, ge::ccKernelType::MIX_AICORE,
                                                       ge::ccKernelType::MIX_VECTOR_CORE};
  return aicore_kernel_type.count(kernel_type) > 0UL;
}

bool Om2CodegenUtils::IsSuppoprtAddrRefreshable(const uint64_t mem_type) {
  return (mem_type == static_cast<uint64_t>(Om2MemoryAppType::kMemoryTypeFeatureMap)) ||
         (mem_type == static_cast<uint64_t>(Om2MemoryAppType::kMemoryTypeModelIo));
}

bool Om2CodegenUtils::IsUnsupportedNodeType(const std::string &type) {
  return ((type == VARIABLE) || (type == CONSTANTOP) || (type == CONSTPLACEHOLDER) || (type == QUEUE_DATA)
          || (type == FILECONSTANT) || (type == REFDATA) || (type == QUEUE_DATA) || (type == "SuperKernel"));
}

bool Om2CodegenUtils::IsNeedAtomicCleanTask(const OpDescPtr &op_desc) {
  bool need_gentask_atomic = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_NEED_GENTASK_ATOMIC, need_gentask_atomic);
  bool is_soft_sync = false;
  (void)AttrUtils::GetBool(op_desc, ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, is_soft_sync);
  return need_gentask_atomic || is_soft_sync;
}

bool Om2CodegenUtils::IsSeparatelyCleanTask(const OpDescPtr &op_desc, const std::string &kernel_name) {
  const std::string attr_key_kernel_name = op_desc->GetName() + "_atomic_kernelname";
  std::string attr_val_kernel_name;
  bool has_set_atomic_kernel_name =
      ge::AttrUtils::GetStr(op_desc, attr_key_kernel_name, "_atomic_kernelname", attr_val_kernel_name) &&
      (kernel_name.compare(attr_val_kernel_name) == 0);
  // fe set atomic clean/memset kernel name into corresponding nodes
  // as this, ge can use this attr to find separate atomic clean/memset op
  has_set_atomic_kernel_name =
      has_set_atomic_kernel_name && (op_desc->GetType() != ge::ATOMICADDRCLEAN) && (op_desc->GetType() != ge::MEMSET);
  if (!has_set_atomic_kernel_name) {
    return false;
  }
  const bool is_need_atomic_clean = IsNeedAtomicCleanTask(op_desc);
  GELOGD("Node: %s has_set_atomic_kernel_name: %d, is_need_atomic_clean: %d.", op_desc->GetNamePtr(),
         static_cast<int32_t>(has_set_atomic_kernel_name), static_cast<int32_t>(is_need_atomic_clean));
  return is_need_atomic_clean;
}

bool Om2CodegenUtils::OpNeedPrint(const OpDescPtr &op_desc) {
  const std::string kOpDfxOptions = "_op_dfx_options";
  const std::string kOpDfxPrintf = "printf";
  std::vector<std::string> dfx_opts;
  if (!ge::AttrUtils::GetListStr(op_desc, kOpDfxOptions, dfx_opts) ||
      (std::find(dfx_opts.begin(), dfx_opts.end(), kOpDfxPrintf) == dfx_opts.end())) {
    GELOGD("op[%s] does not have print dfx option", op_desc->GetName().c_str());
    return false;
  }
  return true;
}

bool Om2CodegenUtils::IsSoftSyncOp(const OpDescPtr &op_desc) {
  bool is_soft_sync_op = false;
  (void)AttrUtils::GetBool(op_desc, ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, is_soft_sync_op);
  return is_soft_sync_op;
}

bool Om2CodegenUtils::IsBlockingAicpuOp(const OpDescPtr &op_desc) {
  bool is_blocking_aicpu_op = false;
  (void)AttrUtils::GetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, is_blocking_aicpu_op);
  return is_blocking_aicpu_op;
}

int32_t Om2CodegenUtils::TopicTypeToRtsFlag(const int32_t topic_type) {
  // Use the 5th and 6th bits of `type` indicate the value of topic_type.
  // xxxxxxxx xxxxxxxx xxxxxxxx xx00xxxx: DEVICE_ONLY
  // xxxxxxxx xxxxxxxx xxxxxxxx xx01xxxx: DEVICE_FIRST
  // xxxxxxxx xxxxxxxx xxxxxxxx xx10xxxx: HOST_ONLY
  // xxxxxxxx xxxxxxxx xxxxxxxx xx11xxxx: HOST_FIRST
  // Use the 9th-11th bits of `type` indicate the value of qos. 12th indicate qos on/off
  // xxxxxxxx xxxxxxxx xxxx0000 xxxxxxxx: qos off
  // xxxxxxxx xxxxxxxx xxxx1000 xxxxxxxx: qos on, level=0(min level)
  // xxxxxxxx xxxxxxxx xxxx1111 xxxxxxxx: qos on, level=7(max level)
  const auto it = kTopicTypeToRtsFlagMap.find(static_cast<int32_t>(((static_cast<uint32_t>(topic_type)) & 0x30U) >> 4));
  if (it != kTopicTypeToRtsFlagMap.end()) {
    return it->second;
  }

  return -1;
}
}

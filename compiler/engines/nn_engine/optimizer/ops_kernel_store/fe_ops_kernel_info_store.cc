/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include <utils/tensor_utils.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "common/fe_error_code.h"
#include "common/fe_inner_attr_define.h"
#include "common/fe_inner_error_codes.h"
#include "common/fe_utils.h"
#include "common/op_slice_util.h"
#include "common/range_format_transfer/transfer_range_according_to_format.h"
#include "common/unknown_shape_util.h"
#include "common/fe_op_info_common.h"
#include "common/fe_report_error.h"
#include "common/util/op_info_util.h"
#include "common/fe_graph_common.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_tensor.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/node_utils_ex.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "graph_optimizer/json_parser/tbe_json_parse.h"
#include "ops_store/ops_kernel_manager.h"
#include "param_calculate/tensorsize_calculator.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "graph_optimizer/dynamic_shape_optimizer/fuzzy_compiler/fuzzy_generalize.h"
#include "graph_optimizer/dynamic_shape_optimizer/fuzzy_compiler/input_node_generalize.h"
#include "graph_optimizer/ub_fusion/fusion_graph_merge/fusion_graph_merge.h"
#include "common/op_tiling/op_tiling_rt2.h"

using std::map;
using std::string;
using std::vector;
namespace fe {
namespace {
uint64_t GenerateHashKey(const ge::OpDescPtr &op_desc_ptr) {
  int64_t src_format = 0;
  int64_t dst_format = 0;
  (void)ge::AttrUtils::GetInt(op_desc_ptr, ATTR_NAME_DST_FORMAT, dst_format);
  (void)ge::AttrUtils::GetInt(op_desc_ptr, ATTR_NAME_SRC_FORMAT, src_format);

  /* hash key is 64-bits unsigned integer which consists of 8 bits of input0's
   * format (bit 0 to 7), 8 bits of input0's data type (bit 8 to 15), 8
   * bits of output0's format (bit 16 to 23), 8 bits of output0's data type
   * (bit 24 to 31), 8 bits of attribute src_format (bit 32 to 39) and 8 bits
   * attribute dst_format (bbit 40 to 47). The last 16 bits is reserved. */

  if(op_desc_ptr->GetInputDescPtr(0) == nullptr || op_desc_ptr->GetOutputDescPtr(0) == nullptr){
    FE_LOGE("op_desc_ptr's input desc ptr or output desc ptr is nullptr.");
    return -1;
  }
  uint64_t input0_format = static_cast<uint64_t>(ge::GetPrimaryFormat(op_desc_ptr->GetInputDescPtr(0)->GetFormat()));
  uint64_t input0_data_type = static_cast<uint64_t>(op_desc_ptr->GetInputDescPtr(0)->GetDataType());
  uint64_t output0_format = static_cast<uint64_t>(ge::GetPrimaryFormat(op_desc_ptr->GetOutputDescPtr(0)->GetFormat()));
  uint64_t output0_data_type = static_cast<uint64_t>(op_desc_ptr->GetOutputDescPtr(0)->GetDataType());

  uint64_t hash_key = ((input0_format) | (input0_data_type << static_cast<uint32_t>(BitShift::BIT_SHIFT_8)) |
                       (output0_format << static_cast<uint32_t>(BitShift::BIT_SHIFT_16)) |
                       (output0_data_type << static_cast<uint32_t>(BitShift::BIT_SHIFT_24)) |
                       (static_cast<uint64_t>(src_format) << static_cast<uint32_t>(BitShift::BIT_SHIFT_32)) |
                       (static_cast<uint64_t>(dst_format) << static_cast<uint32_t>(BitShift::BIT_SHIFT_40)));
  return hash_key;
}

bool IsGeOp(const ge::NodePtr &node) {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  bool is_ge_op = false;
  if (ge::AttrUtils::GetBool(op_desc_ptr, IS_GE_OP, is_ge_op) && is_ge_op) {
    FE_LOGI("Op %s is ge op!", op_desc_ptr->GetName().c_str());
    return true;
  }
  return false;
}
} // namespace

using fe::StringUtils;
using ScopeJsonMap_t = std::map<int64_t, std::string>;
const std::string kAttrNameOriginalFusionGraph = "_original_fusion_graph";
const std::string kDnnVmDvpp = "DNN_VM_DVPP";
const std::string kDvppOpsKernel = "dvpp_ops_kernel";
const std::map<std::string, std::string> ATOMIC_ATTR_MAP {
  {OP_PARA_SIZE, "atomic_op_para_size"},
  {ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "_atomic_cube_vector_core_type"},
  {COMPILE_INFO_JSON, "_atomic_compile_info_json"},
  {COMPILE_INFO_KEY, "_atomic_compile_info_key"},
  {kGlobalworkspaceSize, "_atomic_globalworkspace_size"},
  {kGlobalworkspaceType, "_atomic_globalworkspace_type"},
  {kKernelName, "_atomic_kernelname"},
  {TBE_OP_ATOMIC_WSP_MODE, "wspMode"}};
/*
 *  @ingroup fe
 *  @brief   check specified GeAttrValue::ValueType of op_desc_attr.Value() is in
 * info_op_attr.vector<Values>
 *  @param   [in] value      : used to specified GeAttrValue::ValueType for
 * template
 *  @param   [in] op_desc_attr : GeAttrValue from OpDesc
 *  @param   [in] info_op_attr : vector<GeAttrValue> from OpKernelInfo
 *  @return  true or false
 */
template <typename T>
bool FindValueInVector(T &value, const ge::GeAttrValue &op_desc_attr, const vector<ge::GeAttrValue> &info_op_attr);

FEOpsKernelInfoStore::FEOpsKernelInfoStore(std::string engine_name)
    : enable_shared_from_this(),
      init_flag_(false),
      map_all_sub_store_info_(),
      op_kernel_store_type_(),
      engine_name_(engine_name),
      check_support_static_flag_(false),
      optimize_utility_(nullptr),
      fusion_priority_mgr_ptr_(nullptr) {}

FEOpsKernelInfoStore::~FEOpsKernelInfoStore() {}

Status FEOpsKernelInfoStore::Initialize(const map<string, string> &options) {
  (void)options;
  if (init_flag_) {
    FE_LOGD("FEOpsKernelStore has already been initialized.");
    return SUCCESS;
  }
  /* Before FEOpsKernelInfoStore is initialized, Configuration class has
     already loaded ops info lib info vector */
  init_flag_ = true;
  Status status = OpsKernelManager::Instance(engine_name_).Initialize();
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][SetCheck][Init] Failed to initialize ops kernel manager.");
    return FAILED;
  }

  const std::vector<FEOpsStoreInfo> &ops_store_info_vec = Configuration::Instance(engine_name_).GetOpsStoreInfo();
  for (const FEOpsStoreInfo &ops_sub_store_info : ops_store_info_vec) {
    FE_LOGD("Init sub ops store %s.", ops_sub_store_info.fe_ops_store_name.c_str());
    SubOpsStorePtr sub_ops_kernel_info_store_ptr = nullptr;
    FE_MAKE_SHARED(sub_ops_kernel_info_store_ptr = std::make_shared<SubOpsStore>(engine_name_),
                   return OP_STORE_MAKE_SHARED_FAILED);
    FE_CHECK(sub_ops_kernel_info_store_ptr == nullptr,
             REPORT_FE_ERROR("[GraphOpt][SetCheck][Init] subOpsKernelInfoStorePtr is nullptr."),
             return PARAM_INVALID);
    sub_ops_kernel_info_store_ptr->SetSubStoreInfo(ops_sub_store_info);
    Status result = sub_ops_kernel_info_store_ptr->InitializeSubStore();
    if (result == SUCCESS) {
      map_all_sub_store_info_.emplace(
          std::make_pair(ops_sub_store_info.fe_ops_store_name, sub_ops_kernel_info_store_ptr));
    }
  }

  return SUCCESS;
}

Status FEOpsKernelInfoStore::Finalize() {
  FE_LOGD("Finalizing the FEOpsKernelStore.");
  if (!init_flag_) {
    FE_LOGD("FEOpsKernelInfoStore has not been initialized, Finalize is not allowed.");
    return SUCCESS;
  }
  Status status = OpsKernelManager::Instance(engine_name_).Finalize();
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][SetCheck][Finalize] Failed to finalize ops kernel manager.");
    return FAILED;
  }
  init_flag_ = false;
  return SUCCESS;
}

Status FEOpsKernelInfoStore::CreateSession(const std::map<std::string, std::string> &session_options) {
  (void)session_options;
  return SUCCESS;
}

Status FEOpsKernelInfoStore::DestroySession(const std::map<std::string, std::string> &session_options) {
  (void)session_options;
  return SUCCESS;
}

const std::string& FEOpsKernelInfoStore::GetFEOpsKernelInfoStoreName() const { return engine_name_; }

void FEOpsKernelInfoStore::GetAllOpsKernelInfo(map<string, ge::OpInfo> &infos) const {
  OpsKernelManager::Instance(engine_name_).GetAllOpsKernelInfo(infos);
}

Status FEOpsKernelInfoStore::GetSubOpsStore(const std::string &op_store_name,
                                            SubOpsStorePtr &sub_ops_store, std::ostringstream &reason_oss) const {
  auto sub_ops_store_iter = map_all_sub_store_info_.find(op_store_name);
  if (sub_ops_store_iter == map_all_sub_store_info_.end() || sub_ops_store_iter->second == nullptr) {
    reason_oss << "op store[" << op_store_name << "] is not found.";
    return FAILED;
  }
  sub_ops_store = sub_ops_store_iter->second;
  return SUCCESS;
}

SubOpsStorePtr FEOpsKernelInfoStore::GetSubOpsStore(const OpImplType &impe_type) const {
  FEOpsStoreInfo ops_store;
  if (Configuration::Instance(engine_name_).GetOpStoreInfoByImplType(impe_type, ops_store) != SUCCESS) {
    return nullptr;
  }
  auto sub_ops_store_iter = map_all_sub_store_info_.find(ops_store.fe_ops_store_name);
  if (sub_ops_store_iter == map_all_sub_store_info_.end() || sub_ops_store_iter->second == nullptr) {
    FE_LOGD("Op store[%s] is not found.", ops_store.fe_ops_store_name.c_str());
    return nullptr;
  }

  return sub_ops_store_iter->second;
}

Status FEOpsKernelInfoStore::GetOpKernel(const std::string &op_type, const FEOpsStoreInfo &ops_store,
                                         OpKernelInfoPtr &op_kernel_ptr, std::ostringstream &reason_oss,
                                         uint64_t &not_support_reason_id) const {
  op_kernel_ptr = OpsKernelManager::Instance(engine_name_)
                      .GetOpKernelInfoByOpType(ops_store.fe_ops_store_name, op_type);
  if (op_kernel_ptr == nullptr) {
    reason_oss << "[" << ops_store.fe_ops_store_name << "]:"
               << "op type " << op_type << " is not found in this op store.";
    uint64_t offset_num = ops_store.op_impl_type * NOT_SUPPORTED_REASON_OFFSET_BIT_NUM;
    not_support_reason_id += (static_cast<uint64_t>(OpNotSupportedReasonID::EN_TYPE_NOT_FOUND) << offset_num);
    return FAILED;
  }
  return SUCCESS;
}

void FEOpsKernelInfoStore::GetOpKernelInfoPtr(const ge::NodePtr &node_ptr, OpKernelInfoPtr &op_kernel_ptr) const {
  FE_LOGD("Node[%s, %s] begin to get kernel info by op store.",
          node_ptr->GetNamePtr(), node_ptr->GetTypePtr());
  const std::vector<FEOpsStoreInfo> &fe_ops_store_info_vec = Configuration::Instance(engine_name_).GetOpsStoreInfo();
  std::ostringstream reason_oss;
  for (auto &ops_store : fe_ops_store_info_vec) {
    UnSupportedReason sub_store_reason;
    OpStoreAdapterPtr op_store_adapter = nullptr;
    SubOpsStorePtr sub_ops_store = nullptr;
    if (GetOpKernel(node_ptr->GetType(), ops_store, op_kernel_ptr,
        reason_oss, sub_store_reason.reason_id) != SUCCESS) {
      continue;
    }
    FE_LOGD("GetOpKernel successfully, OpType[%s].", node_ptr->GetTypePtr());
    break;
  }
}

bool FEOpsKernelInfoStore::CheckIsDynamicShape(const SubOpsStorePtr &sub_ops_store_ptr, const ge::NodePtr &node,
    const CheckSupportMode &check_mode, CheckSupportParam &check_param) const {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  bool is_dynamic_shape = UnknownShapeUtils::IsUnknownShapeOp(*op_desc_ptr);
  bool is_support_dynamic = is_dynamic_shape && check_param.op_kernel_ptr->IsSupportDynamicShape();
  bool is_static_to_dynamic_softsync_op = CheckVirtualSoftsyncOp(check_param.op_kernel_ptr, op_desc_ptr);
  bool is_support_dim = check_param.op_kernel_ptr->IsSupportDynamicRank() ||
                        !UnknownShapeUtils::IsContainUnknownDimNum(*op_desc_ptr);
  bool is_support_dim_and_dynamic = is_support_dynamic && is_support_dim;
  bool is_support_dynamic_compile_static = !is_dynamic_shape && check_param.op_kernel_ptr->IsDynamicCompileStatic();
  bool is_check_dynamic_shape = is_support_dim_and_dynamic || is_support_dynamic_compile_static;
  if (is_static_to_dynamic_softsync_op) {
    (void)ge::AttrUtils::SetBool(op_desc_ptr, kSoftsyncDynamicImpl, true);
    if (sub_ops_store_ptr->CheckSubStoreSupported(node, check_mode, true, check_param)) {
      (void)ge::AttrUtils::SetBool(op_desc_ptr, kStaticToDynamicSoftSyncOp, true);
      (void)ge::AttrUtils::SetBool(op_desc_ptr, ATTR_NAME_SUPPORT_DYNAMIC_SHAPE, true);
      (void)ge::AttrUtils::SetBool(op_desc_ptr, ATTR_NAME_IS_OP_DYNAMIC_IMPL, true);
      return true;
    } else {
      check_param.check_static = false;
    }
  }
  if (is_check_dynamic_shape) {
    if (sub_ops_store_ptr->CheckSubStoreSupported(node, check_mode, true, check_param)) {
      if (is_support_dynamic) {
        (void)ge::AttrUtils::SetBool(op_desc_ptr, ATTR_NAME_SUPPORT_DYNAMIC_SHAPE, true);
      }
      bool is_dynamic_impl = check_param.all_impl_checked ? check_param.dynamic_compile_static : true;
      if (is_dynamic_impl) {
        (void)ge::AttrUtils::SetBool(op_desc_ptr, ATTR_NAME_IS_OP_DYNAMIC_IMPL, is_dynamic_impl);
      }
      FE_LOGD("[CheckSupp][CheckDynamic] Node[%s, %s]: check support success.", node->GetNamePtr(), node->GetTypePtr());
      FE_LOGD("[CheckSupp][CheckDynamic] Node[%s, %s]: is_dynamic=%d, is_all_impl_checked=%d, is_dynamic_impl=%d.",
          node->GetNamePtr(), node->GetTypePtr(), is_support_dynamic, check_param.all_impl_checked, is_dynamic_impl);
      return true;
    }
  }

  if (check_param.all_impl_checked) {
    check_param.check_static = false;
    return false;
  }

  bool is_only_support_dynamic = is_support_dynamic && !is_support_dim;
  if (IsFuzzBuild() && is_only_support_dynamic) {
    if (!sub_ops_store_ptr->CheckSubStoreSupported(node, check_mode, true, check_param)) {
      return false;
    }
    ge::AttrUtils::SetBool(op_desc_ptr, kOpShapeOrRangeUnsupport, true);
    return true;
  }
  return false;
}

bool FEOpsKernelInfoStore::CheckIsStaticShape(const SubOpsStorePtr &sub_ops_store_ptr,
                                              const ge::NodePtr &node, const CheckSupportMode &check_mode,
                                              CheckSupportParam &check_param) const {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  bool is_dynamic_shape = UnknownShapeUtils::IsUnknownShapeOp(*op_desc_ptr);
  if (is_dynamic_shape) {
    return false;
  }
  if (sub_ops_store_ptr->CheckSubStoreSupported(node, check_mode, false, check_param)) {
    (void)ge::AttrUtils::SetBool(op_desc_ptr, ATTR_NAME_IS_OP_DYNAMIC_IMPL, false);
    return true;
  }
  return false;
}

void FEOpsKernelInfoStore::JoinNotSupportDynamicReason(const OpKernelInfoPtr &op_kernel_ptr,
                                                       const ge::NodePtr &node,
                                                       UnSupportedReason &reason) const {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  bool is_dynamic_shape = UnknownShapeUtils::IsUnknownShapeOp(*op_desc_ptr);
  bool is_not_support_dynamic = is_dynamic_shape &&
      (!op_kernel_ptr->IsSupportDynamicShape() ||
       (!op_kernel_ptr->IsSupportDynamicRank() && UnknownShapeUtils::IsContainUnknownDimNum(*op_desc_ptr)));
  if (is_not_support_dynamic) {
    reason.reason += "The op is dynamic shape, but is not configured to support dynamic shape in op store";
    reason.reason_id = static_cast<uint64_t>(OpNotSupportedReasonID::EN_NOT_SUPPORT_DYNAMIC_SHAPE);
  }
}

void FEOpsKernelInfoStore::GetAndSetCustomOpFilePath(const ge::NodePtr &node, const FEOpsStoreInfo &ops_store) const {
  if (ops_store.is_custom_store) {
    if (node->GetOpDesc()->HasAttr(CUSTOM_OP_FILE_PATH)) {
      return;
    }
    size_t pos = ops_store.cfg_file_path.rfind("/op_impl/");
    if (pos == std::string::npos) {
      return;
    }
    std::string custom_op_file_path = ops_store.cfg_file_path.substr(0, pos);
    (void)ge::AttrUtils::SetStr(node->GetOpDesc(), CUSTOM_OP_FILE_PATH, custom_op_file_path);
    FE_LOGD("Node[%s] set custom_op_file_path attribute to [%s].", node->GetName().c_str(), custom_op_file_path.c_str());
  }
}

void FEOpsKernelInfoStore::SetOpsPathNamePrefix(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_ptr) const {
  std::string ops_path_name_prefix = op_kernel_ptr->GetOpsPathNamePrefix();
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), OPS_PATH_NAME_PREFIX, ops_path_name_prefix);
}

bool FEOpsKernelInfoStore::CheckSupportedByOpsStore(const ge::NodePtr &node, const CheckSupportMode &check_mode,
                                                    CheckSupportParam &check_param, OpImplType &impl_type) const {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();

  std::string op_attr_engine_name;
  std::string op_attr_kernel_name;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, ge::ATTR_NAME_OP_SPECIFIED_ENGINE_NAME, op_attr_engine_name);
  (void)ge::AttrUtils::GetStr(op_desc_ptr, ge::ATTR_NAME_OP_SPECIFIED_KERNEL_LIB_NAME, op_attr_kernel_name);
  if (op_attr_engine_name == kDnnVmDvpp || op_attr_kernel_name == kDvppOpsKernel) {
    return false;
  }

  const std::vector<FEOpsStoreInfo> &fe_ops_store_info_vec = Configuration::Instance(engine_name_).GetOpsStoreInfo();

  std::ostringstream reason_oss;
  size_t index = 0;
  for (auto &ops_store : fe_ops_store_info_vec) {
    index++;
    /*
     * non-persistent custom op,
     * only use non-persistent-tbe-custom op info lib and tbe-custom op info lib
     * other op can not use non-persistent-tbe-custom op info lib
     */
    bool is_custom_op = CheckCustomOp(node, ops_store);
    if (is_custom_op) {
      FE_LOGD("This node [%s, %s] is custom op while op store is not custom store.",
              node->GetName().c_str(), node->GetType().c_str());
      continue;
    }

    SubOpsStorePtr sub_ops_store_ptr = nullptr;
    bool skip_check =
        GetSubOpsStore(ops_store.fe_ops_store_name, sub_ops_store_ptr, reason_oss) != SUCCESS ||
        GetOpKernel(node->GetType(), ops_store, check_param.op_kernel_ptr, reason_oss,
                    check_param.unsupport_reason.reason_id) != SUCCESS;
    if (skip_check) {
      check_param.unsupport_reason.reason = reason_oss.str();
      continue;
    }
    bool support_dynamic = check_param.op_kernel_ptr->IsSupportDynamicShape();
    (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kNodeSupportDynamicShape, support_dynamic);
    if (!CheckVirtualSoftsyncOp(check_param.op_kernel_ptr, op_desc_ptr) &&
        !Configuration::Instance(engine_name_).IsDynamicImplFirst()) {
      if (CheckIsStaticShape(sub_ops_store_ptr, node, check_mode, check_param)) {
        GetAndSetCustomOpFilePath(node, ops_store);
        SetOpsPathNamePrefix(node, check_param.op_kernel_ptr);
        impl_type = ops_store.op_impl_type;
        return true;
      }
    }

    if (CheckIsDynamicShape(sub_ops_store_ptr, node, check_mode, check_param)) {
      GetAndSetCustomOpFilePath(node, ops_store);
      SetOpsPathNamePrefix(node, check_param.op_kernel_ptr);
      impl_type = ops_store.op_impl_type;
      return true;
    }

    if (check_param.check_static && Configuration::Instance(engine_name_).IsDynamicImplFirst()) {
      if (CheckIsStaticShape(sub_ops_store_ptr, node, check_mode, check_param)) {
        GetAndSetCustomOpFilePath(node, ops_store);
        SetOpsPathNamePrefix(node, check_param.op_kernel_ptr);
        impl_type = ops_store.op_impl_type;
        return true;
      }
    }

    JoinNotSupportDynamicReason(check_param.op_kernel_ptr, node, check_param.unsupport_reason);
    uint64_t offset_num = ops_store.op_impl_type * NOT_SUPPORTED_REASON_OFFSET_BIT_NUM;
    FE_UINT64_ADDCHECK(check_param.unsupport_reason.reason_id,
        (static_cast<uint64_t>(check_param.unsupport_reason.reason_id) << offset_num));
    check_param.unsupport_reason.reason_id +=
        (static_cast<uint64_t>(check_param.unsupport_reason.reason_id) << offset_num);
    check_param.unsupport_reason.reason = reason_oss.str() + check_param.unsupport_reason.reason;
    reason_oss.clear();
    reason_oss << check_param.unsupport_reason.reason;
  }

  FE_LOGI("This op[%s, %s] is not supported. Reason:%s", node->GetNamePtr(), node->GetTypePtr(),
          check_param.unsupport_reason.reason.c_str());
  return false;
}

Status FEOpsKernelInfoStore::GetNotSupportedReasonByAttr(const uint64_t &reason, std::ostringstream &reason_oss) const {
  const std::vector<FEOpsStoreInfo> &fe_ops_store_info_vec = Configuration::Instance(engine_name_).GetOpsStoreInfo();
  for (auto &ops_store : fe_ops_store_info_vec) {
    uint64_t offset_num = static_cast<uint64_t>(ops_store.op_impl_type << 2);
    int64_t reason_id = static_cast<int64_t>((reason >> offset_num) & NOT_SUPPORTED_REASON_OFFSET_UNIT);
    OpNotSupportedReasonID sub_reason_id = static_cast<OpNotSupportedReasonID>(reason_id);
    auto reason_inter = ID_REASON_MAP.find(sub_reason_id);
    if (reason_inter != ID_REASON_MAP.end()) {
      reason_oss << reason_inter->second << " Op information library name is " << ops_store.fe_ops_store_name << ".";
    } else {
      FE_LOGD("Get unsupported reason in %s unsuccessful.", ops_store.fe_ops_store_name.c_str());
    }
  }

  return SUCCESS;
}

void FEOpsKernelInfoStore::SetCheckSupportedStaticFlag(bool stat_flag) { check_support_static_flag_ = stat_flag; }

bool FEOpsKernelInfoStore::CheckSupported(const ge::OpDescPtr &op_desc_ptr, std::string &un_supported_reason) const {
  ge::ComputeGraphPtr graph = nullptr;
  FE_MAKE_SHARED(graph = std::make_shared<ge::ComputeGraph>("Node"), return false);
  FE_CHECK(graph == nullptr, REPORT_FE_ERROR("[GraphOpt][SetCheckSupportedStaticFlag] Graph is null."), return false);
  ge::NodePtr node = graph->AddNode(op_desc_ptr);
  FE_CHECK(node == nullptr, REPORT_FE_ERROR("[GraphOpt][SetCheckSupportedStaticFlag] AddNode failed."), return false);
  return CheckSupported(node, un_supported_reason);
}

bool FEOpsKernelInfoStore::CheckAccuracySupported(const ge::OpDescPtr &op_desc_ptr, std::string &un_supported_reason,
                                                  bool real_query) const {
  ge::ComputeGraphPtr graph = nullptr;
  FE_MAKE_SHARED(graph = std::make_shared<ge::ComputeGraph>("Node"), return false);
  FE_CHECK(graph == nullptr, REPORT_FE_ERROR("[GraphOpt][CheckAccuracySupported] Graph is null."), return false);
  ge::NodePtr node = graph->AddNode(op_desc_ptr);
  FE_CHECK(node == nullptr, REPORT_FE_ERROR("[GraphOpt][CheckAccuracySupported] Failed to add node."), return false);
  return CheckAccuracySupported(node, un_supported_reason, real_query);
}

bool FEOpsKernelInfoStore::CheckSupported(const ge::NodePtr &node, std::string &un_supported_reason,
                                          [[maybe_unused]] ge::CheckSupportFlag &flag) const {
  bool bres = false;
  try {
    bres = CheckSupported(node, un_supported_reason);
  } catch (const std::exception &exp) {
    FE_LOGE("Node[%s, %s] check support failed, exception message is [%s].",
            node->GetNamePtr(), node->GetTypePtr(), exp.what());
    bres = false;
  }
  return bres;
}

bool FEOpsKernelInfoStore::CheckSupported(const ge::NodePtr &node, std::string &un_supported_reason) const {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  FE_LOGD("Node[%s, %s] begin CheckSupported.", op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());
  /* Op Cast need to be checked accurately, because the input and output should
   * be matched correspondingly */
  bool ret = false;
  if (IsDtypeSensitiveOp(op_desc_ptr->GetType()) || IsFormatSensitiveOp(op_desc_ptr->GetType())) {
    ret = CheckSupportedBase(node, un_supported_reason, CheckSupportMode::ACCURACY_MODE, true);
  } else {
    ret = CheckSupportedBase(node, un_supported_reason, CheckSupportMode::DTYPE_FORMAT_MODE, true);
  }
  return ret;
}

bool FEOpsKernelInfoStore::CheckAccuracySupported(const ge::NodePtr &node, std::string &un_supported_reason,
                                                  bool real_query) const {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  FE_CHECK(op_desc_ptr == nullptr,
           REPORT_FE_ERROR("[GraphOpt][SetDynmCustomOpStoreInfo] op_desc_ptr is null."), return false);
  FE_LOGD("Node[%s, %s] begin CheckAccuracySupported.", op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());
  bool ret = CheckSupportedBase(node, un_supported_reason, CheckSupportMode::ACCURACY_MODE, real_query);
  if (!VerifyCastC0Format(op_desc_ptr)) {
    un_supported_reason = "Cast op not support current c0 format.";
    return false;
  }
  return ret;
}

bool FEOpsKernelInfoStore::CheckAccuracySupportByCache(const ge::OpDescPtr &op_desc_ptr) {
  uint64_t hash_key = GenerateHashKey(op_desc_ptr);
  auto iter = checksupport_cache_.find(hash_key);
  if (iter != checksupport_cache_.end()) {
    FE_LOGD("Op %s's hash key %lx is found in cache.", op_desc_ptr->GetName().c_str(), hash_key);
    if (iter->second.result) {
      (void)ge::AttrUtils::SetInt(op_desc_ptr, FE_IMPLY_TYPE, iter->second.fe_impl_type);
      (void)ge::AttrUtils::SetInt(op_desc_ptr, ge::ATTR_NAME_IMPLY_TYPE, iter->second.ge_impl_type);
      return true;
    }
  }
  return false;
}

Status FEOpsKernelInfoStore::StoreCheckSuportResultForTransNodes(const ge::OpDescPtr &op_desc_ptr, bool result) {
  uint64_t hash_key = GenerateHashKey(op_desc_ptr);
  int64_t fe_impl_type = 0;
  int64_t ge_impl_type = 0;
  (void)ge::AttrUtils::GetInt(op_desc_ptr, FE_IMPLY_TYPE, fe_impl_type);
  (void)ge::AttrUtils::GetInt(op_desc_ptr, ge::ATTR_NAME_IMPLY_TYPE, ge_impl_type);

  CheckSuportResult check_suport_result = {result, fe_impl_type, ge_impl_type};
  checksupport_cache_.emplace(std::make_pair(hash_key, check_suport_result));
  FE_LOGD("Store op %s's {hash key %lx, result %u, impl_type %ld, %ld} into cache.", op_desc_ptr->GetNamePtr(),
          hash_key, result, fe_impl_type, ge_impl_type);
  return SUCCESS;
}

bool FEOpsKernelInfoStore::CheckSupportedBase(const ge::NodePtr &node, std::string &un_supported_reason,
                                              CheckSupportMode mode, bool real_query) const {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  int64_t is_check_supported = 0;
  std::ostringstream ge_reason_oss;
  /* real_query = true means we need skip the cache value of IS_CHECK_SUPPORTED
   * and do the checksupport again */
  if (!real_query && ge::AttrUtils::GetInt(op_desc_ptr, IS_CHECK_SUPPORTED, is_check_supported)) {
    /* The highest bit of "isCheckSupported" indicates whether or not it is
     * supported, 0: supported 1: not supported.
     **/
    uint64_t reason = static_cast<uint64_t>(is_check_supported);
    if ((reason & NOT_SUPPORTED_FLAG_BIT) == 0) {
      FE_LOGD("Node %s has been check_supported, result is support.", op_desc_ptr->GetName().c_str());
      return true;
    } else {
      FE_LOGI_IF(GetNotSupportedReasonByAttr(reason, ge_reason_oss) != SUCCESS,
                 "Get Node %s not supported reason not success.", op_desc_ptr->GetName().c_str());
      un_supported_reason = ge_reason_oss.str();
      FE_LOGI("Node %s has been check_supported, result is not supported.", op_desc_ptr->GetName().c_str());
      return false;
    }
  }

  OpImplType imply_type = EN_RESERVED;
  CheckSupportParam check_param;
  if (CheckSupportedByOpsStore(node, mode, check_param, imply_type)) {
    FE_LOGD("[ChkSpt][FEChk][Node %s, type %s] This op is supported by FE with impl type %u in mode %u.",
            op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr(), static_cast<uint32_t>(imply_type),
            static_cast<uint32_t>(mode));
    OpImplType main_type = GetMainImplType<OpImplType>(imply_type);
    // 1. check the main_type
    auto iter = IMPL_TYPE_MAP.find(main_type);
    if (iter == IMPL_TYPE_MAP.end()) {
      REPORT_FE_ERROR("[GraphOpt][Setcheck] Node[%s, %s]: the FE imply type %ld is not found in GE imply type map.",
                      op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), main_type);
      return false;
    }

    // 2. set the fe and ge imply type of the op
    (void)ge::AttrUtils::SetInt(op_desc_ptr, FE_IMPLY_TYPE, static_cast<int64_t>(imply_type));
    FE_LOGD("Op[name=%s,type=%s]: set the FE_IMPLY_TYPE attribute [%s], set the IMPLY_TYPE attribute [%s].",
            op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr(), GetImplTypeString(imply_type).c_str(),
            GetGeImplTypeString(iter->second).c_str());
    /* If the op is ge op, we only use the ops kernel info to select its
     * format and data type. Actually there is no op implementation for this op.
     * So we return false and then GE will not distribute this op to FE to
     * precompile and compile. */
    if (IsGeOp(node)) {
      FE_LOGI("FE does not support Ge Op [%s,%s], although it's in ops info library.", op_desc_ptr->GetName().c_str(),
              op_desc_ptr->GetType().c_str());
      return false;
    }

    (void)ge::AttrUtils::SetInt(op_desc_ptr, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(iter->second));
    return true;
  }

  FE_LOGI_IF(GetNotSupportedReasonByAttr(check_param.unsupport_reason.reason_id, ge_reason_oss) != SUCCESS,
             "Get Node %s not supported reason not success.", op_desc_ptr->GetNamePtr());
  un_supported_reason = check_param.unsupport_reason.reason;
  FE_LOGW("This Op [%s, %s] is not supported in all op information libraries.",
          op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());
  return false;
}

Status FEOpsKernelInfoStore::GetAllSubOpsStore(std::map<std::string, SubOpsStorePtr> &all_sub_store_ptr) const {
  all_sub_store_ptr = map_all_sub_store_info_;
  return SUCCESS;
}

Status FEOpsKernelInfoStore::QueryHighPrioOpImplType(const ge::NodePtr &node, OpImplType &impl_type,
                                                     OpKernelInfoPtr &op_kernel_ptr) const {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  FE_LOGD("QueryHighPrioOpImplType, querying the highest implement type of Op %s.", op_desc_ptr->GetTypePtr());
  CheckSupportParam check_param;
  /* Op Cast need to be checked accurately, because the input and output should
   * be matched correspondingly */
  if (IsDtypeSensitiveOp(op_desc_ptr->GetType()) || IsFormatSensitiveOp(op_desc_ptr->GetType())) {
    if (!CheckSupportedByOpsStore(node, CheckSupportMode::ACCURACY_MODE, check_param, impl_type)) {
      FE_LOGW("Op[%s, %s] is not supported in all op information librarys by accurate mode.",
              op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());
      op_kernel_ptr = check_param.op_kernel_ptr;
      return OP_NOT_FOUND_IN_QUERY_HIGH_PRIO_IMPL;
    }
  } else {
    if (!CheckSupportedByOpsStore(node, CheckSupportMode::DTYPE_FORMAT_MODE, check_param, impl_type)) {
      FE_LOGW("Op[%s, %s] is not supported in all op information librarys.",
              op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());
      op_kernel_ptr = check_param.op_kernel_ptr;
      return OP_NOT_FOUND_IN_QUERY_HIGH_PRIO_IMPL;
    }
  }
  op_kernel_ptr = check_param.op_kernel_ptr;
  return SUCCESS;
}

Status FEOpsKernelInfoStore::CompileOpGetImplType(const ge::NodePtr &node, OpImplType &impl_type) const {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  impl_type = EN_RESERVED;
  int64_t tmp_imply_type = -1;
  if (!ge::AttrUtils::GetInt(op_desc_ptr, FE_IMPLY_TYPE, tmp_imply_type)) {
    // chose high prior op info lib for node
    OpKernelInfoPtr op_kernel_ptr;
    if (QueryHighPrioOpImplType(node, impl_type, op_kernel_ptr) == SUCCESS) {
      (void)ge::AttrUtils::SetInt(op_desc_ptr, FE_IMPLY_TYPE, impl_type);
      FE_LOGD("Set FE op imply type [%ld] of OP [%s].", impl_type, op_desc_ptr->GetName().c_str());
    } else {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][CompOpGetImplType] OP [%s] is not supported by FE.",
                      op_desc_ptr->GetName().c_str());
      return FAILED;
    }

    // set ge imply type
    (void)ge::AttrUtils::SetInt(op_desc_ptr, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(impl_type));
    FE_LOGD("Set node name %s GE imply type, %s = %ld.", op_desc_ptr->GetName().c_str(),
            ge::ATTR_NAME_IMPLY_TYPE.c_str(), impl_type);
  } else {
    impl_type = static_cast<OpImplType>(tmp_imply_type);
  }

  return SUCCESS;
}

Status FEOpsKernelInfoStore::CompileOpTiling(const ge::NodePtr &node) const {
  if (!IsAtomicStaticReuseBinaryOp(node)) {
    FE_LOGD("Memset Node[%s, %s]: it is not reuse op.", node->GetNamePtr(), node->GetTypePtr());
    return SUCCESS;
  }
  if (!IsStaticOrAutoFuseReuseBinaryOp(node)) {
    FE_LOGD("Node[%s, %s]: it is not reuse op.", node->GetNamePtr(), node->GetTypePtr());
    return SUCCESS;
  }
  if (TilingForOneNode(node) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status FEOpsKernelInfoStore::CompileOpGetTvmJsonInfo(ScopeNodeIdMap &fusion_nodes_map,
                                                     CompileResultMap &compile_ret_map) const {
  // PreCompile tbe-builtin op
  for (auto &fusion_item : fusion_nodes_map) {
    int64_t scope_id = fusion_item.first;
    ge::Node &node = *((fusion_item.second)[0]);
    ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
    // find the json file path according to scope id
    CompileResultMap::const_iterator json_iter = compile_ret_map.find(scope_id);
    if (json_iter == compile_ret_map.cend() || json_iter->second.empty()) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][CompOpGetTvmJsInfo] Node[%s, %s]: json file of scopeId %ld is not found.",
                      op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), scope_id);
      return OP_COMPILER_CHECK_FALSE_FAILED;
    }

    // package tvm json info
    TbeJsonFileParsePtr parse_ptr = nullptr;
    FE_MAKE_SHARED(parse_ptr = std::make_shared<TbeJsonFileParse>(node),
                   return fe::OP_COMPILER_MAKE_SHARED_FAILED);
    if (parse_ptr->PackageTvmJsonInfo(json_iter->second[0]) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][CompOpGetTvmJsInfo] Node[%s, %s]: Failed to package tvm json.",
                      op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
      return FAILED;
    }
    if (CompileOpTiling(node.shared_from_this()) == FAILED) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][CompileOpTiling] Tiling failed, op name = %s, op type = %s.",
                      op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

std::vector<uint32_t> FEOpsKernelInfoStore::CompileGetAtomicOutput(const ge::OpDescPtr &op_desc_ptr) const {
  std::vector<uint32_t> output_index;
  std::vector<uint32_t> tmp_output_index;
  if (ge::AttrUtils::GetListInt(op_desc_ptr, TBE_OP_ATOMIC_OUTPUT_INDEX, tmp_output_index)) {
    uint32_t output_size = tmp_output_index.size();
    for (uint32_t i = 0; i < output_size; i++) {
      // get atomic output index
      if (tmp_output_index[i] == 1) {
        output_index.push_back(i);
      }
    }
  }

  return output_index;
}

Status FEOpsKernelInfoStore::CompileSetAtomicMemSetWorkSpace(ge::NodePtr &node_ptr,
                                                             MemsetWorkspaceInfo &memset_workspace_info) const {
  std::vector<uint32_t> output_index;
  output_index = CompileGetAtomicOutput(node_ptr->GetOpDesc());
  for (auto &idx : output_index) {
    memset_workspace_info.work_space.push_back(0);
    if (TensorSizeCalculator::CalculateOpTensorSize(node_ptr) != SUCCESS) {
      return FAILED;
    }
    if (UnknownShapeUtils::IsUnknownShapeOp(*(node_ptr->GetOpDesc()))) {
      memset_workspace_info.work_space_bytes.push_back(-1);
      continue;
    }
    int64_t tensor_size = 0;
    (void)ge::TensorUtils::GetSize(node_ptr->GetOpDesc()->GetOutputDesc(idx), tensor_size);
    memset_workspace_info.work_space_bytes.push_back(tensor_size);
    FE_LOGD("Get op:%s, idx:%u, tensor_size:%ld.", node_ptr->GetNamePtr(), idx, tensor_size);
  }

  std::map<string, std::map<int64_t, int64_t>> sub_node_workspace_info;
  sub_node_workspace_info = node_ptr->GetOpDesc()->TryGetExtAttr(ge::EXT_ATTR_ATOMIC_WORKSPACE_INFO,
      sub_node_workspace_info);
  if (!sub_node_workspace_info.empty()) {
    std::map<string, std::map<int64_t, int64_t>>::const_iterator sub_node_workspace_value =
      sub_node_workspace_info.find(node_ptr->GetOpDesc()->GetName());
    if (sub_node_workspace_value != sub_node_workspace_info.cend()) {
      std::map<int64_t, int64_t> workspace_bytes_map = sub_node_workspace_value->second;
      for (auto workspace_size : workspace_bytes_map) {
        memset_workspace_info.work_space.push_back(0);
        if (UnknownShapeUtils::IsUnknownShapeOp(*(node_ptr->GetOpDesc()))) {
          memset_workspace_info.work_space_bytes.push_back(-1);
          continue;
        }
        memset_workspace_info.work_space_bytes.push_back(workspace_size.second);
      }
      FE_LOGD("Get op:%s, workspace clean size count:%zu.", node_ptr->GetOpDesc()->GetNamePtr(),
              memset_workspace_info.work_space_bytes.size());
    }
  }

  ge::AttrUtils::GetListInt(node_ptr->GetOpDesc(), TBE_OP_ATOMIC_DTYPES, memset_workspace_info.dtype_list);
  ge::AttrUtils::GetListInt(node_ptr->GetOpDesc(), TBE_OP_ATOMIC_INT64_VALUES,
                            memset_workspace_info.init_value_int64_list);
  ge::AttrUtils::GetListFloat(node_ptr->GetOpDesc(), TBE_OP_ATOMIC_FLOAT_VALUES,
                              memset_workspace_info.init_value_float_list);
  return SUCCESS;
}

Status FEOpsKernelInfoStore::GetAllAtomicCleanNode(ge::NodePtr &node_ptr, vector<ge::NodePtr> &atomic_node_vec) const {
  FE_CHECK_NOTNULL(node_ptr);
  /* if the node is atomic node and connected to netoutput,
     then add it into atomic_node_vec and continue to next node */
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  bool is_atomic_node = false;
  bool is_net_output = false;
  if (ge::AttrUtils::GetBool(op_desc_ptr, ge::ATOMIC_ATTR_IS_ATOMIC_NODE, is_atomic_node) &&
      ge::AttrUtils::GetBool(op_desc_ptr, ATTR_NAME_IS_CONNECTED_TO_NETOUTPUT, is_net_output)) {
    if (is_atomic_node && is_net_output) {
      atomic_node_vec.push_back(node_ptr);
      FE_LOGD(
          "op:%s is atomic node and connected to the netoutput, do not need"
          "to be compiled, will just create and compile an atomic clean node for it.",
          node_ptr->GetName().c_str());
    }
  }

  // add atomic clean node, and compile.
  bool atomic_node_flag = false;
  if (SetAtomicOpAttr(op_desc_ptr, atomic_node_flag) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][GetAllAtoCleanNd] op:%s, failed to set atomic info.",
                    node_ptr->GetName().c_str());
    return FAILED;
  }
  if (atomic_node_flag) {
    atomic_node_vec.push_back(node_ptr);
  }
  return SUCCESS;
}

Status FEOpsKernelInfoStore::SetWorkSpaceForAtomicMemSet(const MemsetWorkspaceInfo &memset_workspace_info,
                                                         const ge::NodePtr &node,
                                                         const ge::ComputeGraphPtr &tmp_graph,
                                                         vector<ge::NodePtr> &memset_nodes) const {
  FE_CHECK_NOTNULL(tmp_graph);
  if (!memset_workspace_info.work_space.empty()) {
    ge::OpDescPtr memset_op_desc_ptr = nullptr;
    FE_MAKE_SHARED(memset_op_desc_ptr = make_shared<ge::OpDesc>("memset_node", MEMSET_OP_TYPE),
                   return FAILED);
    FE_CHECK_NOTNULL(memset_op_desc_ptr);
    auto op_desc_ptr = node->GetOpDesc();
    bool is_unknown_shape_op = UnknownShapeUtils::IsUnknownShapeOp(*op_desc_ptr);
    FE_LOGD("Node[%s, %s]: unknown shape flag is %d.", op_desc_ptr->GetNamePtr(),
            op_desc_ptr->GetTypePtr(), is_unknown_shape_op);
    if (is_unknown_shape_op) {
      (void)ge::AttrUtils::SetBool(memset_op_desc_ptr, ATTR_NAME_SUPPORT_DYNAMIC_SHAPE, true);
      (void)ge::AttrUtils::SetBool(memset_op_desc_ptr, ATTR_NAME_IS_OP_DYNAMIC_IMPL, true);
      (void)ge::AttrUtils::SetBool(memset_op_desc_ptr, ATTR_NAME_UNKNOWN_SHAPE, true);
    }

    string name = memset_op_desc_ptr->GetName() + to_string(GetAtomicId());
    memset_op_desc_ptr->SetName(name);
    memset_op_desc_ptr->SetWorkspace(memset_workspace_info.work_space);
    memset_op_desc_ptr->SetWorkspaceBytes(memset_workspace_info.work_space_bytes);
    memset_op_desc_ptr->SetExtAttr(ATTR_NAME_ORIGINAL_NODE, node);
    (void)ge::AttrUtils::SetListInt(memset_op_desc_ptr, ge::ATTR_NAME_AUTOMIC_ADD_START,
                                    memset_workspace_info.work_space);
    (void)ge::AttrUtils::SetListInt(memset_op_desc_ptr, ge::ATTR_NAME_ATOMIC_MEMSET_SIZES,
                                    memset_workspace_info.work_space_bytes);
    (void)ge::AttrUtils::SetListInt(memset_op_desc_ptr, ge::ATTR_NAME_ATOMIC_MEMSET_DTYPES,
                                    memset_workspace_info.dtype_list);
    (void)ge::AttrUtils::SetListInt(memset_op_desc_ptr, ge::ATTR_NAME_ATOMIC_MEMSET_VALUES_INT,
                                    memset_workspace_info.init_value_int64_list);
    (void)ge::AttrUtils::SetListFloat(memset_op_desc_ptr, ge::ATTR_NAME_ATOMIC_MEMSET_VALUES_FLOAT,
                                      memset_workspace_info.init_value_float_list);
    ge::NodePtr memset_node = tmp_graph->AddNode(memset_op_desc_ptr, op_desc_ptr->GetId());
    FE_CHECK_NOTNULL(memset_node);
    memset_nodes.push_back(memset_node);
    op_desc_ptr->SetExtAttr(ATTR_NAME_MEMSET_NODE, memset_node);
    FE_LOGD("Create memset op:%s for op:%s, work_space_vec_num:%zu.",
            name.c_str(), node->GetNamePtr(), memset_workspace_info.work_space.size());
    ge::ComputeGraphPtr owner_graph = node->GetOwnerComputeGraph();
    if (owner_graph == nullptr) {
      FE_LOGW("Node[%s] can not get owner graph, memset op %s, set single op scene flag unsuccessful.",
              node->GetName().c_str(), memset_op_desc_ptr->GetName().c_str());
      return SUCCESS;
    }
    bool is_single_flag = false;
    (void)ge::AttrUtils::GetBool(*owner_graph, kFESingleOpScene, is_single_flag);
    FE_LOGW("Memset op %s, setting single op scene flag [%d] to temporary graph %s.", memset_op_desc_ptr->GetNamePtr(),
            is_single_flag, tmp_graph->GetName().c_str());
    (void)ge::AttrUtils::SetBool(*tmp_graph, kFESingleOpScene, is_single_flag);
  }
  return SUCCESS;
}

Status FEOpsKernelInfoStore::CompileAndSetKernelNameForMemSet(const vector<ge::NodePtr> &node_vec,
                                                              vector<ge::NodePtr> &memset_nodes) {
  if (!memset_nodes.empty()) {
    if (FEOpsKernelInfoStore::CompileOp(memset_nodes) != SUCCESS) {
      return FAILED;
    }
    /* link the atomic memset node bin file to the corresponding atomic node with
       the extra attribute tbe_atomic_kernel */
    for (auto &node_ptr : node_vec) {
      ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
      ge::NodePtr memset_node = nullptr;
      memset_node = op_desc_ptr->TryGetExtAttr(ATTR_NAME_MEMSET_NODE, memset_node);
      FE_CHECK(memset_node == nullptr,
               REPORT_FE_ERROR("[SubGraphOpt][Compile][CompSetMemsetNm] Node[%s]: failed to get attr memset_node_ptr.",
                               op_desc_ptr->GetName().c_str()),
               return FAILED);
      ge::OpKernelBinPtr tbe_kernel_ptr = nullptr;
      tbe_kernel_ptr = memset_node->GetOpDesc()->TryGetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
      FE_CHECK(tbe_kernel_ptr == nullptr,
               REPORT_FE_ERROR("[SubGraphOpt][Compile][CompSetMemsetNm] Node[%s]: failed to get attr tbe_kernel.",
                               memset_node->GetName().c_str()),
               return FAILED);
      op_desc_ptr->SetExtAttr(TBE_ATOMIC_KERNEL, tbe_kernel_ptr);
      op_desc_ptr->SetExtAttr(ge::EXT_ATTR_ATOMIC_TBE_KERNEL, tbe_kernel_ptr);
      FE_LOGD("Set node:%s's extra attribute tbe_atomic_kernel success.", op_desc_ptr->GetNamePtr());
      int64_t atomic_op_para_size = 0;
      (void)ge::AttrUtils::GetInt(memset_node->GetOpDesc(), OP_PARA_SIZE, atomic_op_para_size);
      (void)ge::AttrUtils::SetInt(op_desc_ptr, ATOMIC_ATTR_MAP.at(OP_PARA_SIZE),
                                  static_cast<int64_t>(atomic_op_para_size));

      // set core type
      std::string core_type;
      if (ge::AttrUtils::GetStr(memset_node->GetOpDesc(), ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type)) {
        (void)ge::AttrUtils::SetStr(op_desc_ptr, ATOMIC_ATTR_MAP.at(ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE), core_type);
      }

      // set compile info
      std::string compile_info_json;
      if (ge::AttrUtils::GetStr(memset_node->GetOpDesc(), COMPILE_INFO_JSON, compile_info_json)) {
        (void)ge::AttrUtils::SetStr(op_desc_ptr, ATOMIC_ATTR_MAP.at(COMPILE_INFO_JSON), compile_info_json);
      }
      std::string compile_info_key;
      if (ge::AttrUtils::GetStr(memset_node->GetOpDesc(), COMPILE_INFO_KEY, compile_info_key)) {
        (void)ge::AttrUtils::SetStr(op_desc_ptr, ATOMIC_ATTR_MAP.at(COMPILE_INFO_KEY), compile_info_key);
      }
      int64_t global_work_space_size;
      if (ge::AttrUtils::GetInt(memset_node->GetOpDesc(), kGlobalworkspaceSize, global_work_space_size)) {
        (void)ge::AttrUtils::SetInt(op_desc_ptr, ATOMIC_ATTR_MAP.at(kGlobalworkspaceSize), global_work_space_size);
      }
      int64_t global_work_space_type;
      if (ge::AttrUtils::GetInt(memset_node->GetOpDesc(), kGlobalworkspaceType, global_work_space_type)) {
        (void)ge::AttrUtils::SetInt(op_desc_ptr, ATOMIC_ATTR_MAP.at(kGlobalworkspaceType), global_work_space_type);
      }
      std::shared_ptr<optiling::utils::OpRunInfo> tiling_info = nullptr;
      tiling_info = memset_node->GetOpDesc()->TryGetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, tiling_info);
      if (tiling_info != nullptr) {
        FE_LOGD("Set node:%s extra attribute op runinfo success.", op_desc_ptr->GetName().c_str());
        op_desc_ptr->SetExtAttr(kAttrAtomicOpRunInfo, tiling_info);
      }
      // Set attr atomic_kernelname
      string AttrKeyAtomicKernelName = "_atomic_kernelname";
      string AttrValKernelName;
      if (ge::AttrUtils::GetStr(memset_node->GetOpDesc(), kKernelName, AttrValKernelName)) {
        (void)ge::AttrUtils::SetStr(op_desc_ptr, AttrKeyAtomicKernelName, AttrValKernelName);
        (void)ge::AttrUtils::SetStr(op_desc_ptr, ATOMIC_ATTR_MAP.at(kKernelName), AttrValKernelName);
        FE_LOGD("Set node[name:%s,type:%s]'s attribute atomic_kernelname success.", op_desc_ptr->GetNamePtr(),
                op_desc_ptr->GetTypePtr());
      } else {
        FE_LOGD("Get node[name:%s,type:%s]'s attribute kernelname unsuccessful.", memset_node->GetName().c_str(),
                memset_node->GetType().c_str());
      }

      // Set attr wsp_mode
      string wsp_mode;
      if (ge::AttrUtils::GetStr(memset_node->GetOpDesc(), TBE_OP_ATOMIC_WSP_MODE, wsp_mode)) {
        (void)ge::AttrUtils::SetStr(op_desc_ptr, ATOMIC_ATTR_MAP.at(TBE_OP_ATOMIC_WSP_MODE), wsp_mode);
      }

      std::string kernel_bin_id;
      if (ge::AttrUtils::GetStr(memset_node->GetOpDesc(), kAttrKernelBinId, kernel_bin_id)) {
        (void)ge::AttrUtils::SetStr(op_desc_ptr, kAttrMemsetKernelBinId, kernel_bin_id);
      }
      FE_LOGD("Memset node[%s] bin id[%s].", op_desc_ptr->GetNamePtr(), kernel_bin_id.c_str());
    }
  }
  return SUCCESS;
}

Status FEOpsKernelInfoStore::CompileMemSet(vector<ge::NodePtr> &node_vec) {
  vector<ge::NodePtr> memset_nodes;
  ge::ComputeGraphPtr tmp_graph;
  FE_MAKE_SHARED((tmp_graph = std::make_shared<ge::ComputeGraph>("OpCompileGraph")), return FAILED);

  for (auto &node_ptr : node_vec) {
    ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
    if (IsStcToDynSoftSyncOp(op_desc_ptr)) {
      FE_LOGD("[SubGraphOpt][Compile][CompileMemSet] op[%s] is static_to_dynamic_softsync_op",
              op_desc_ptr->GetNamePtr());
    }
    MemsetWorkspaceInfo memset_workspace_info;

    if (CompileSetAtomicMemSetWorkSpace(node_ptr, memset_workspace_info) == FAILED) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][CompileMemSet] op:%s Failed to set work_space.",
                      op_desc_ptr->GetName().c_str());
    }

    // set atomic memset op work_space_info
    if (SetWorkSpaceForAtomicMemSet(memset_workspace_info, node_ptr, tmp_graph, memset_nodes) !=
        SUCCESS) {
      return FAILED;
    }
  }

  // atomic memset to compile
  if (CompileAndSetKernelNameForMemSet(node_vec, memset_nodes) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
 }

Status FEOpsKernelInfoStore::PrePareCompileParameter(
    const ge::NodePtr &node, const string &op_type, OpImplType &impl_type,
    std::unordered_map<OpStoreAdapterPtr, vector<PreCompileNodePara>> &node_map) {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();

  // caculate input output size
  OpStoreAdapterPtr op_store_adapter;
  if (OpStoreAdapterManager::Instance(engine_name_).GetOpStoreAdapter(impl_type, op_store_adapter) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][PrepCompParm] Failed to get op store adapter by impl type %s.",
                    GetImplTypeString(impl_type).c_str());
    return FAILED;
  }

  FEOpsStoreInfo op_store_info;
  if (Configuration::Instance(engine_name_).GetOpStoreInfoByImplType(impl_type, op_store_info) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][PrepCompParm] Failed to get op information library info by impl type %s.",
                    GetImplTypeString(impl_type).c_str());
    return OP_COMPILER_CHECK_FALSE_FAILED;
  }

  // get registered precompile function
  OpKernelInfoPtr op_kernel_info_ptr =
      OpsKernelManager::Instance(engine_name_).GetOpKernelInfoByOpType(op_store_info.fe_ops_store_name, op_type);
  if (!op_desc_ptr->HasAttr(ge::ATTR_NAME_UNREGST_OPPATH)) {
    FE_CHECK(op_kernel_info_ptr == nullptr,
             REPORT_FE_ERROR("[SubGraphOpt][Compile][PrepCompParm] opKernelInfoPtr is nullptr."), return FAILED);
  }

  bool is_custom_op = false;
  (void)ge::AttrUtils::GetBool(op_desc_ptr, NON_PERSISTENT_CUSTOM_OP_FLAG, is_custom_op);
  std::string op_dsl_file_path;
  bool ret_status = (is_custom_op && op_kernel_info_ptr != nullptr && !op_kernel_info_ptr->GetOpImpPath().empty());
  if (op_desc_ptr->HasAttr(ge::ATTR_NAME_UNREGST_OPPATH)) {
    if (!ge::AttrUtils::GetStr(op_desc_ptr, ge::ATTR_NAME_UNREGST_OPPATH, op_dsl_file_path)) {
      FE_LOGI("Get attr:_unregst_oppath of node[%s] not successfully", op_desc_ptr->GetName().c_str());
      return FAILED;
    }
  } else if (ret_status) {
    op_dsl_file_path = op_kernel_info_ptr->GetOpImpPath();
  } else {
    op_dsl_file_path = op_store_info.op_impl_file_path;
  }

  if (IsTbe(impl_type)) {
    string session_graph_id;
    if (!ge::AttrUtils::GetStr(node->GetOpDesc(), ge::ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
      FE_LOGW("%s get session_graph_id failed.", node->GetNamePtr());
    }

    PreCompileNodePara pre_comp_node_para = {node.get(), op_kernel_info_ptr, op_store_info.fe_ops_store_name,
                                             op_dsl_file_path, session_graph_id, nullptr};
    if (node_map.find(op_store_adapter) == node_map.end()) {
      vector<PreCompileNodePara> pre_comp_node_para_vec;
      pre_comp_node_para_vec.push_back(pre_comp_node_para);
      node_map.emplace(make_pair(op_store_adapter, pre_comp_node_para_vec));
    } else {
      node_map[op_store_adapter].push_back(pre_comp_node_para);
    }
  }
  return SUCCESS;
}

Status FEOpsKernelInfoStore::FuzzCompileAndGetResult(const ge::NodePtr &node,
                                                     const OpStoreAdapterPtr &op_store_adapter,
                                                     ScopeNodeIdMap &fusion_node_map) const {
  std::vector<ge::NodePtr> buff_fus_compile_failed_nodes;
  CompileInfoParam compile_info_param{buff_fus_compile_failed_nodes};
  compile_info_param.fusion_nodes_map = fusion_node_map;
  compile_info_param.compile_strategy = CompileStrategy::COMPILE_STRATEGY_ONLINE_FUZZ;
  if (op_store_adapter->CompileOp(compile_info_param) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][PrepCompAndComp] CompileOp failed, node name = %s.",
                    (node != nullptr) ? node->GetName().c_str() : "");
    return FAILED;
  }
  if (CompileOpGetTvmJsonInfo(compile_info_param.fusion_nodes_map, compile_info_param.compile_ret_map) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][PrepCompAndComp] CompileOp failed, node name = %s.",
                    (node != nullptr) ? node->GetName().c_str() : "");
    return FAILED;
  }

  return SUCCESS;
}

Status FEOpsKernelInfoStore::PreCompileAndCompile(
    std::unordered_map<OpStoreAdapterPtr, vector<PreCompileNodePara>> &node_map, const ge::NodePtr &node,
    ScopeNodeIdMap &fusion_node_map, const bool is_fuzz_build) {
  for (auto &comp_para : node_map) {
    OpStoreAdapterPtr op_store_adapter = comp_para.first;
    if (op_store_adapter->PreCompileOp(comp_para.second) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][PrepCompAndComp] PreCompileOp failed");
      return FAILED;
    }
  }
  // get tbe adapter
  OpStoreAdapterPtr op_store_adapter = nullptr;
  if (OpStoreAdapterManager::Instance(engine_name_).GetOpStoreAdapter(EN_IMPL_HW_TBE, op_store_adapter) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][PrepCompAndComp] Failed to get op store adapter by imply type %ld.",
                    EN_IMPL_HW_TBE);
    return FAILED;
  }

  if (is_fuzz_build) {
    return FuzzCompileAndGetResult(node, op_store_adapter, fusion_node_map);
  }
  // Compile tbe op
  std::vector<ge::NodePtr> buff_fus_compile_failed_nodes;
  CompileInfoParam compile_info(buff_fus_compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_node_map;
  if (op_store_adapter->CompileOp(compile_info) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][PrepCompAndComp] CompileOp failed, node name = %s.",
                    !(node == nullptr) ? node->GetName().c_str() : "");
    return FAILED;
  }

  if (CompileOpGetTvmJsonInfo(compile_info.fusion_nodes_map, compile_info.compile_ret_map) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][PrepCompAndComp] CompileOp failed, node name = %s.",
                    !(node == nullptr) ? node->GetName().c_str() : "");
    return FAILED;
  }

  return SUCCESS;
}

Status FEOpsKernelInfoStore::CompileSingleOp(const ge::NodePtr &node_ptr, const bool is_fuzz_build) {
  ScopeNodeIdMap fusion_node_map;
  int64_t scope_id = -1;
  std::unordered_map<OpStoreAdapterPtr, vector<PreCompileNodePara>> node_map;
  vector<ge::NodePtr> atomic_node_vec;
  auto &node = *node_ptr;
  string op_type = node.GetType();
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);

  string const_op_type;
  bool const_flag = ge::NodeUtils::GetConstOpType(node_ptr, const_op_type);
  if ((op_type == OP_TYPE_PLACE_HOLDER || op_type == OP_TYPE_END || op_type == "Data" || const_flag)) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][CompSingleOp] Compile single op failed, op %s not supported.",
                    op_type.c_str());
    return FAILED;
  }

  // if the node is atomic node and connected to netoutput, then add it into atomic_node_vec and continue to next node
  bool is_atomic_node = false;
  bool is_net_output = false;
  if (ge::AttrUtils::GetBool(op_desc_ptr, ge::ATOMIC_ATTR_IS_ATOMIC_NODE, is_atomic_node) &&
      ge::AttrUtils::GetBool(op_desc_ptr, "is_connected_to_netoutput", is_net_output)) {
    if (is_atomic_node && is_net_output) {
      if (op_type == "MemSet") {
        REPORT_FE_ERROR("[SubGraphOpt][Compile][CompSingleOp] Node[%s]: it is atomic node and connected to netoutput.",
                        node.GetName().c_str());
        return FAILED;
      }
      if (CompileMemSet(atomic_node_vec) != SUCCESS) {
        return FAILED;
      }
      FE_LOGI("op:%s is atomic and connected to the netoutput, only compile atomic clean.", node.GetName().c_str());
    }
  }

  if (!IsNeededCompile(op_desc_ptr)) {
    return SUCCESS;
  }

  OpImplType op_impl_type = EN_IMPL_HW_TBE;
  if (PrePareCompileParameter(node_ptr, op_type, op_impl_type, node_map) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][CompSingleOp] PreCompile single op failed, op %s not supported.",
                    op_type.c_str());
    return FAILED;
  }

  std::vector<ge::Node *> node_list_new;
  node_list_new.push_back(&node);
  fusion_node_map.emplace(std::pair<int64_t, std::vector<ge::Node *>>(scope_id, node_list_new));
  scope_id--;

  return PreCompileAndCompile(node_map, node_ptr, fusion_node_map, is_fuzz_build);
}

static void FillFusionNodeMapForTbeOp(ge::Node &node, OpImplType impl_type, ScopeNodeIdMap &fusion_node_map,
                               int64_t &scope_id) {
  auto op_desc = node.GetOpDesc();
  bool impl_type_check = IsTbe(impl_type);
  int64_t tmp_scope_id = -1;
  GetFusionScopeAttr(op_desc, tmp_scope_id);
  if (impl_type_check) {
    if (tmp_scope_id > 0) {
      const auto iter = fusion_node_map.find(tmp_scope_id);
      if (iter == fusion_node_map.cend()) {
        std::vector<ge::Node *> node_list_new;
        node_list_new.emplace_back(&node);
        fusion_node_map.emplace(std::pair<int64_t, std::vector<ge::Node *>>(tmp_scope_id, node_list_new));
      } else {
        iter->second.emplace_back(&node);
      }
    } else {
      std::vector<ge::Node *> node_list_new;
      node_list_new.push_back(&node);
      fusion_node_map.emplace(std::pair<int64_t, std::vector<ge::Node *>>(scope_id, node_list_new));
      scope_id--;
    }
  }
}

Status FEOpsKernelInfoStore::CompileMultipleOp(vector<ge::NodePtr> &node_vec, const bool is_fuzz_build) {
  ScopeNodeIdMap fusion_node_map;
  int64_t scope_id = -1;
  std::unordered_map<OpStoreAdapterPtr, vector<PreCompileNodePara>> node_map;
  vector<ge::NodePtr> atomic_node_vec;
  if (node_vec.empty()) {
    FE_LOGW("Node vector param is empty.");
    return SUCCESS;
  }

  for (auto &node_ptr : node_vec) {
    auto &node = *node_ptr;
    string op_type = node.GetType();
    ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
    FE_CHECK_NOTNULL(op_desc_ptr);

    if (!IsNeededCompile(op_desc_ptr)) {
      continue;
    }

    string const_op_type;
    bool const_flag = ge::NodeUtils::GetConstOpType(node_ptr, const_op_type);
    if ((op_type == OP_TYPE_PLACE_HOLDER || op_type == OP_TYPE_END || op_type == "Data" || const_flag)) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][CompMultiOp] Compile single op failed, op %s not supported.",
                      op_type.c_str());
      return FAILED;
    }

    OpImplType impl_type = EN_RESERVED;
    if (CompileOpGetImplType(node_ptr, impl_type) == FAILED) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][CompMultiOp] Compile single op %s failed, get impl_type failed.",
                      op_type.c_str());
      return FAILED;
    }
    if (PrePareCompileParameter(node_ptr, op_type, impl_type, node_map) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][CompMultiOp] PreCompile single op failed, op %s not supported.",
                      op_type.c_str());
      return FAILED;
    }
    FillFusionNodeMapForTbeOp(node, impl_type, fusion_node_map, scope_id);
  }

  if (PreCompileAndCompile(node_map, node_vec[0], fusion_node_map, is_fuzz_build) != SUCCESS) {
    return FAILED;
  }

  return MultipleOpMergeFusionGraph(node_vec, atomic_node_vec, is_fuzz_build);
}

Status FEOpsKernelInfoStore::MultipleOpMergeFusionGraph(vector<ge::NodePtr> &node_vec,
                                                        vector<ge::NodePtr> &atomic_node_vec,
                                                        const bool is_fuzz_build) {
  for (auto &node_ptr : node_vec) {
    if (GetAllAtomicCleanNode(node_ptr, atomic_node_vec) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][CompMultiOp] Get atomic node for op[name:%s,type:%s] failed.",
                      node_ptr->GetOpDesc()->GetName().c_str(), node_ptr->GetOpDesc()->GetType().c_str());
      return FAILED;
    }
  }
  // create and compile atomic clean node for node in atomic_node_vec
  if (!atomic_node_vec.empty() && CompileMemSet(atomic_node_vec) != SUCCESS) {
    return FAILED;
  }

  if (is_fuzz_build) {
    // online fuzz build, nodes comes from origin_sub_graph, no need merge again
    FE_LOGD("online fuzz build, no need merge fusion.");
    return SUCCESS;
  }

  GraphCommPtr graph_comm_ptr = nullptr;
  FusionGraphMergePtr fusion_graph_merge_ptr = nullptr;
  FE_MAKE_SHARED(graph_comm_ptr = std::make_shared<GraphComm>(engine_name_), return FAILED);
  if (graph_comm_ptr->Initialize() != SUCCESS) {
    return FAILED;
  }
  FE_MAKE_SHARED(fusion_graph_merge_ptr = std::make_shared<FusionGraphMerge>(SCOPE_ID_ATTR, graph_comm_ptr),
                 return FAILED);
  if (fusion_graph_merge_ptr->MergeFusionGraph(*(node_vec[0]->GetOwnerComputeGraph())) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][CompMultiOp] Failed to merge fusion graph.");
    return FAILED;
  }
  return SUCCESS;
}

Status FEOpsKernelInfoStore::CompileOp(vector<ge::NodePtr> &node_vec) {
  FE_TIMECOST_START(FEOpsKernelCompileOp);
  if (node_vec.empty()) {
    FE_LOGD("No nodes need to do compile.");
    return SUCCESS;
  }
  if (node_vec[0]->GetOpDesc()->HasAttr(ge::ATTR_NAME_UNREGST_OPPATH)) {
    if (CompileSingleOp(node_vec[0]) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][CompileOp] Failed to compile node[%s].", node_vec[0]->GetName().c_str());
      return FAILED;
    }
  } else {
    if (CompileMultipleOp(node_vec) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][CompileOp] Failed to compile op.");
      return FAILED;
    }
  }
  FE_TIMECOST_END_LOGI(FEOpsKernelCompileOp, "FEOpsKernelInfoStore::CompileOp");
  return SUCCESS;
}

Status FEOpsKernelInfoStore::SetMemSetOpWorkspaceInfo(ge::OpDescPtr &op_desc, bool &atomic_node_flag) const {
  std::vector<int64_t> tmp_workspace_index;
  if (ge::AttrUtils::GetListInt(op_desc, TBE_OP_ATOMIC_WORKSPACE_INDEX, tmp_workspace_index)) {
    std::vector<int64_t> workspace_index;
    size_t workspace_size = tmp_workspace_index.size();
    for (size_t i = 0; i < workspace_size; i++) {
      if (tmp_workspace_index[i] == 1) {
        workspace_index.push_back(i);
      }
    }
    std::map<int64_t, int64_t> workspace_info;
    std::vector<int64_t> workspace_bytes_vec = op_desc->GetWorkspaceBytes();
    if (!workspace_index.empty()) {
      for (int64_t index : workspace_index) {
        if (index >= static_cast<int64_t>(workspace_bytes_vec.size())) {
          continue;
        }
        workspace_info.insert(std::make_pair(index, workspace_bytes_vec[index]));
      }
      std::map<string, std::map<int64_t, int64_t>> sub_node_workspace_info;
      sub_node_workspace_info.insert(std::make_pair(op_desc->GetName(), workspace_info));
      if (!op_desc->SetExtAttr(ge::EXT_ATTR_ATOMIC_WORKSPACE_INFO, sub_node_workspace_info)) {
        REPORT_FE_ERROR("[SubGraphOpt][SetAttr][SetExtAttr] Set op [%s] workspace atomic info failed!",
                        op_desc->GetName().c_str());
        return FAILED;
      }
      FE_LOGD("Finish set op [%s] workspace atomic info.", op_desc->GetNamePtr());
    }
    atomic_node_flag = true;
  }
  return SUCCESS;
}

Status FEOpsKernelInfoStore::SetAtomicOpAttr(ge::OpDescPtr &op_desc, bool &atomic_node_flag) const {
  std::vector<uint32_t> output_index;
  // only process when get output_index success
  output_index = CompileGetAtomicOutput(op_desc);
  if (!output_index.empty()) {
    if (!ge::AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, output_index)) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][SetAtomOpAttr] Set op [%s] output atomic info to op_desc failed!",
                      op_desc->GetName().c_str());
      return FAILED;
    }
    atomic_node_flag = true;
  }
  if (SetMemSetOpWorkspaceInfo(op_desc, atomic_node_flag) != SUCCESS) {
    FE_LOGE("Op[name:%s, type:%s] set workspace info failed.", op_desc->GetName().c_str(),
            op_desc->GetType().c_str());
    return FAILED;
  }
  FE_LOGD("Set op[name:%s, type:%s] output index and workspace info, outputsize:%zu.", op_desc->GetNamePtr(),
          op_desc->GetTypePtr(), output_index.size());
  return SUCCESS;
}

Status FEOpsKernelInfoStore::CompileOpRun(vector<ge::NodePtr> &node_vec) {
  if (CompileOp(node_vec) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status FEOpsKernelInfoStore::GetOpImplyRealPath(std::string op_imply_relative_path, const OpImplType &op_impl_type,
                                                std::string &op_store_real_path, std::string &op_imply_real_path,
                                                const ge::NodePtr &node_ptr) const {
  // get op_store Path prefix, for op imply path
  std::string op_imply_path_prefix;
  if (op_impl_type == EN_IMPL_NON_PERSISTENT_CUSTOM_TBE) {
    int32_t pos = op_store_real_path.rfind('/');
    if (pos < 0) {
      REPORT_FE_ERROR("[GraphOpt][SetDynmCustomOpStoreInfo] The path of node %s does not contain /.",
                      node_ptr->GetName().c_str());
      return FAILED;
    }
    op_imply_path_prefix = op_store_real_path.substr(0, pos);
  } else {
    op_imply_path_prefix = op_store_real_path;
  }

  // Check to see if it has refreshed op_imply_absolute_path
  if (op_imply_relative_path.find(op_imply_path_prefix) != string::npos) {
    FE_LOGD("Imply path of Op type[%s] has been refreshed, path is %s.", node_ptr->GetType().c_str(),
            op_imply_relative_path.c_str());
    op_imply_real_path = op_imply_relative_path;
    return SUCCESS;
  }

  // verify Relative Path, first char should not be '/'
  if (op_imply_relative_path.substr(0, 1) == "/") {
    REPORT_FE_ERROR("[GraphOpt][SetDynmCustomOpStoreInfo] Imply path of Op type[%s] is invalid, path is %s.",
                    node_ptr->GetType().c_str(), op_imply_relative_path.c_str());
    return FAILED;
  }

  std::string op_imply_absolute_path;
  if (op_imply_relative_path == "./") {
    op_imply_absolute_path = op_imply_path_prefix;
  } else {
    op_imply_absolute_path = op_imply_path_prefix + "/" + op_imply_relative_path;
  }

  op_imply_real_path = RealPath(op_imply_absolute_path);
  if (op_imply_real_path.empty()) {
    REPORT_FE_ERROR("[GraphOpt][SetDynmCustomOpStoreInfo] The op_impl_file_path of node %s not exist in %s",
                    node_ptr->GetName().c_str(), op_imply_real_path.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status FEOpsKernelInfoStore::UpdateOpImplyPath(const ge::NodePtr &node_ptr, std::string &op_store_real_path,
                                               const OpImplType &op_impl_type,
                                               SubOpInfoStorePtr &sub_custom_ops_kernel_ptr) const {
  FE_CHECK_NOTNULL(sub_custom_ops_kernel_ptr);
  std::string op_name = node_ptr->GetName();
  std::string op_type = node_ptr->GetType();

  // get op imply relative path from op_content
  OpContent op_content;
  Status status = sub_custom_ops_kernel_ptr->GetOpContentByOpType(op_type, op_content);
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][SetDynmCustomOpStoreInfo] Op type[%s] not exist in op information library[%ld].",
                    op_type.c_str(), op_impl_type);
    return FAILED;
  }

  OpKernelInfoConstructor op_kernel_info_constructor;
  std::string op_imply_relative_path;
  Status ret =
      op_kernel_info_constructor.GetStrFromOpContent(op_content, STR_IMP_PATH, STR_PATH, op_imply_relative_path);
  if (ret != SUCCESS || op_imply_relative_path.empty()) {
    if (op_impl_type == EN_IMPL_NON_PERSISTENT_CUSTOM_TBE) {
      REPORT_FE_ERROR("[GraphOpt][SetDynmCustomOpStoreInfo] Get op[%s]'s imply relative path failed.", op_name.c_str());
      return FAILED;
    } else {
      FE_LOGD("impl path of tbe-custom allow not exist.");
      return SUCCESS;
    }
  }

  std::string op_imply_real_path;
  if (GetOpImplyRealPath(op_imply_relative_path, op_impl_type, op_store_real_path, op_imply_real_path, node_ptr) !=
      SUCCESS) {
    FE_LOGD("Node %s get OpImplyRealPath unsuccessful.", op_name.c_str());
    return FAILED;
  }

  // set op imply absolute path to op_content
  OpKernelInfoPtr op_kernel_ptr = sub_custom_ops_kernel_ptr->GetOpKernelByOpType(op_type);
  if (op_impl_type != EN_IMPL_NON_PERSISTENT_CUSTOM_TBE && op_kernel_ptr != nullptr) {
    op_kernel_ptr->SetOpImpPath(op_imply_real_path);
  }

  map<string, string> map_temp;
  map_temp.emplace(std::make_pair(STR_PATH, op_imply_real_path));
  op_content.map_kernel_info_[STR_IMP_PATH] = map_temp;
  (void)sub_custom_ops_kernel_ptr->SetOpContent(op_content);
  return SUCCESS;
}

bool FEOpsKernelInfoStore::IsExistInTBECustom(const ge::NodePtr &node_ptr) {
  const std::vector<FEOpsStoreInfo> &ops_store_info_vec = Configuration::Instance(engine_name_).GetOpsStoreInfo();
  for (auto &op_store_info : ops_store_info_vec) {
    OpImplType impl_type = op_store_info.op_impl_type;
    if (impl_type != EN_IMPL_CUSTOM_TBE &&  impl_type < EN_RESERVED) {
      FE_LOGI("Not custom impl:%ld continue.", impl_type);
      continue;
    }
    OpKernelInfoPtr op_kernel_info_ptr =
        OpsKernelManager::Instance(engine_name_).GetOpKernelInfoByOpType(impl_type, node_ptr->GetType());
    if (op_kernel_info_ptr == nullptr) {
      FE_LOGI("Can not get impl:%ld kernel of op:%s.", impl_type, node_ptr->GetNamePtr());
      continue;
    }
    // updata tbe-custom op imply path if exist
    std::string op_store_real_path = RealPath(op_store_info.cfg_file_path);
    SubOpInfoStorePtr custom_ops_ptr =
        OpsKernelManager::Instance(engine_name_).GetSubOpsKernelByStoreName(op_store_info.fe_ops_store_name);
    if (custom_ops_ptr == nullptr) {
      FE_LOGI("There is no default custom op info library.");
      continue;
    }
    if (UpdateOpImplyPath(node_ptr, op_store_real_path, impl_type, custom_ops_ptr) != SUCCESS) {
      FE_LOGW("Update op[%s] imply path failed.", node_ptr->GetName().c_str());
      continue;
    } else {
      return true;
    }
  }
  return false;
}

Status FEOpsKernelInfoStore::GetDynamicCustomOpStoreInfoByNode(const ge::NodePtr &node_ptr,
                                                               vector<std::string> &json_files,
                                                               SubOpInfoStorePtr &sub_dyna_custom_ops_store_ptr) {
  FE_LOGD("GetDynamicCustomOpStoreInfo for node[%s, %s].", node_ptr->GetNamePtr(), node_ptr->GetTypePtr());
  std::string op_name = node_ptr->GetName();

  // if node's op_store_path attr not exist or value is empty, return
  std::string op_store_path;
  if ((!ge::AttrUtils::GetStr(node_ptr->GetOpDesc(), CUSTOM_OP_IMPL_CONFIG_PATH, op_store_path)) ||
      op_store_path.empty()) {
    if (!IsExistInTBECustom(node_ptr)) {
      ErrorMessageDetail err_msg(EM_OPEN_FILE_FAILED, {op_store_path});
      ReportErrorMessage(err_msg);
      REPORT_FE_ERROR("[GraphOpt][SetDynmCustomOpStoreInfo] The op information library path of node %s not exist in %s",
                      op_name.c_str(), op_store_path.c_str());
      return FAILED;
    }
    return SUCCESS;
  }

  std::string op_store_real_path = RealPath(op_store_path);
  if (op_store_real_path.empty()) {
    ErrorMessageDetail err_msg(EM_OPEN_FILE_FAILED, {op_store_path});
    ReportErrorMessage(err_msg);
    REPORT_FE_ERROR("[GraphOpt][SetDynmCustomOpStoreInfo] The op_store_file_path of node %s not exist in %s",
                    op_name.c_str(), op_store_path.c_str());
    return FAILED;
  }

  // don't need to load the same json file again
  bool has_been_load = false;
  for (auto &iter : json_files) {
    if (iter == op_store_real_path) {
      has_been_load = true;
    }
  }
  if (!has_been_load) {
    if (sub_dyna_custom_ops_store_ptr->LoadOpJsonFile(op_store_real_path) != SUCCESS) {
      ErrorMessageDetail err_msg(EM_OPEN_FILE_FAILED, {op_store_real_path});
      ReportErrorMessage(err_msg);
      REPORT_FE_ERROR("[GraphOpt][SetDynmCustomOpStoreInfo] Failed to load json file[%s].", op_store_real_path.c_str());
      return FAILED;
    }
    json_files.push_back(op_store_real_path);
  }

  // updata op imply path
  if (UpdateOpImplyPath(node_ptr, op_store_real_path, EN_IMPL_NON_PERSISTENT_CUSTOM_TBE,
                        sub_dyna_custom_ops_store_ptr) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][SetDynmCustomOpStoreInfo] Update op[%s] imply path failed.", op_name.c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status FEOpsKernelInfoStore::GetDefFeOpsStoreInfo(FEOpsStoreInfo &fe_ops_store) {
  const std::vector<FEOpsStoreInfo> &fe_ops_store_info_vec = Configuration::Instance(engine_name_).GetOpsStoreInfo();

  for (auto ops_store_info : fe_ops_store_info_vec) {
    if (ops_store_info.op_impl_type == EN_IMPL_CUSTOM_TBE) {
      fe_ops_store = ops_store_info;
      return SUCCESS;
    }
  }

  return FAILED;
}

Status FEOpsKernelInfoStore::SetDynaCustomOpStoreToAllStore(FEOpsStoreInfo &fe_ops_store,
                                                            SubOpInfoStorePtr &sub_dyna_custom_ops_kernel_ptr) {
  Status status = sub_dyna_custom_ops_kernel_ptr->ConstructOpKernelInfo(engine_name_);
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][SetDynmCustomOpStoreInfo] Failed to initialize non persistent custom sub ops kernel.");
    return FAILED;
  }
  status = OpsKernelManager::Instance(engine_name_).AddSubOpsKernel(sub_dyna_custom_ops_kernel_ptr);
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][SetDynmCustomOpStoreInfo] Failed to add non persistent custom sub ops kernel.");
    return FAILED;
  }

  Configuration::Instance(engine_name_).SetOpsStoreInfo(fe_ops_store);
  SubOpsStorePtr sub_dyna_custom_ops_store_ptr = nullptr;
  FE_MAKE_SHARED(sub_dyna_custom_ops_store_ptr = std::make_shared<SubOpsStore>(engine_name_),
                 return FAILED);
  FE_CHECK(sub_dyna_custom_ops_store_ptr == nullptr,
           REPORT_FE_ERROR("[GraphOpt][SetDynmCustomOpStoreInfo] subDynaCustomOpsStorePtr is nullptr."), return FAILED);

  FE_MAKE_SHARED(sub_dyna_custom_ops_store_ptr->format_dtype_querier_ptr_ =
      std::make_shared<FormatDtypeQuerier>(engine_name_), return FAILED);
  FE_CHECK_NOTNULL(sub_dyna_custom_ops_store_ptr->format_dtype_querier_ptr_);

  sub_dyna_custom_ops_store_ptr->SetSubStoreInfo(fe_ops_store);
  map_all_sub_store_info_.emplace(std::make_pair(fe_ops_store.fe_ops_store_name, sub_dyna_custom_ops_store_ptr));

  return SUCCESS;
}

Status FEOpsKernelInfoStore::SetDynamicCustomOpStoreInfo(ge::ComputeGraph &graph) {
  FE_TIMECOST_START(SetDynamicCustomOpStoreInfo);
  FEOpsStoreInfo fe_ops_store = {NON_PERSISTENT_CUSTOM_PRIORITY,
                                 STR_NON_PERSISTENT_CUSTOM_TBE,
                                 EN_IMPL_NON_PERSISTENT_CUSTOM_TBE,
                                 "",
                                 "",
                                 true,
                                 true,
                                 true};
  FEOpsStoreInfo def_fe_ops_store;
  if (GetDefFeOpsStoreInfo(def_fe_ops_store) == SUCCESS) {
    fe_ops_store.cfg_file_path = def_fe_ops_store.cfg_file_path;
    fe_ops_store.op_impl_file_path = def_fe_ops_store.op_impl_file_path;
  }

  SubOpInfoStorePtr sub_dyna_custom_ops_kernel_ptr = nullptr;
  FE_MAKE_SHARED(sub_dyna_custom_ops_kernel_ptr = std::make_shared<SubOpInfoStore>(fe_ops_store),
                 return OP_STORE_MAKE_SHARED_FAILED);
  FE_CHECK(sub_dyna_custom_ops_kernel_ptr == nullptr,
           REPORT_FE_ERROR("[GraphOpt][SetDynmCustomOpStoreInfo] subOpsKernelInfoStorePtr is nullptr."),
           return PARAM_INVALID);

  bool is_custom_op = false;
  bool has_dynamic_custom_op = false;
  vector<std::string> json_files;
  for (auto &node : graph.GetAllNodes()) {
    if ((ge::AttrUtils::GetBool(node->GetOpDesc(), NON_PERSISTENT_CUSTOM_OP_FLAG, is_custom_op)) && is_custom_op) {
      has_dynamic_custom_op = true;
      if (GetDynamicCustomOpStoreInfoByNode(node, json_files, sub_dyna_custom_ops_kernel_ptr) != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][SetDynmCustomOpStoreInfo] Node %s get dynamic custom op_store info failed.",
                        node->GetName().c_str());
        return FAILED;
      }
    }
  }

  if (has_dynamic_custom_op) {
    if (SetDynaCustomOpStoreToAllStore(fe_ops_store, sub_dyna_custom_ops_kernel_ptr) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][SetDynmCustomOpStoreInfo] Set dynamic custom op_store info failed.");
      return FAILED;
    }
  }
  FE_TIMECOST_END(SetDynamicCustomOpStoreInfo,
                  "SetDynamicCustomOpStoreInfo during FEGraphOptimizer::OptimizeOriginalGraph");
  return SUCCESS;
}

bool FEOpsKernelInfoStore::CheckCustomOp(const ge::NodePtr &node, const FEOpsStoreInfo &ops_store) const {
  bool is_custom_op = false;
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  if ((ge::AttrUtils::GetBool(op_desc_ptr, NON_PERSISTENT_CUSTOM_OP_FLAG, is_custom_op)) && (is_custom_op)) {
    return ((ops_store.op_impl_type != EN_IMPL_CUSTOM_TBE) &&
            (ops_store.op_impl_type != EN_IMPL_NON_PERSISTENT_CUSTOM_TBE));
  } else {
    return (ops_store.op_impl_type == EN_IMPL_NON_PERSISTENT_CUSTOM_TBE);
  }
}

bool FEOpsKernelInfoStore::IsNeededCompile(ge::OpDescPtr &op_desc_ptr) const {
  std::string magic;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, ge::TVM_ATTR_NAME_MAGIC, magic);
  if (magic.empty()) {
    FE_LOGD("Op[name:%s, type:%s] magic is empty, need compile.", op_desc_ptr->GetNamePtr(),
            op_desc_ptr->GetTypePtr());
    return true;
  }
  if (op_desc_ptr->HasAttr(TBE_OP_ATOMIC_OUTPUT_INDEX) || op_desc_ptr->HasAttr(TBE_OP_ATOMIC_WORKSPACE_INDEX)) {
    FE_LOGD("Atomic op[name:%s,type:%s] has been compiled, do not need compile.", op_desc_ptr->GetNamePtr(),
            op_desc_ptr->GetTypePtr());
    return false;
  }
  return true;
}

Status FEOpsKernelInfoStore::SetCutSupportedInfo(const ge::NodePtr &node) {
  string slice_info;
  FE_LOGI("Set cut info for node %s in mds scenario.", node->GetNamePtr());
  (void)ge::AttrUtils::GetStr(node->GetOpDesc(), OP_SLICE_INFO, slice_info);
  if (slice_info.empty()) {
    return SUCCESS;
  }
  fe::OpCalcInfo op_calc_info;
  (void)GetOpSliceInfoFromJson(op_calc_info, slice_info);
  if (OpSliceUtil::SetOpCutInfoOnTensor(node->GetOpDesc(), op_calc_info) != SUCCESS) {
    FE_LOGE("Failed to set op info for node %s.", node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

void FEOpsKernelInfoStore::SetGeneralizeRelatedParam(ge::OptimizeUtility *const utility,
                                                     const FusionPriorityMgrPtr &fusion_priority_mgr_ptr) {
  optimize_utility_ = utility;
  fusion_priority_mgr_ptr_ = fusion_priority_mgr_ptr;
}

Status FEOpsKernelInfoStore::FuzzCompileOp(vector<ge::NodePtr> &node_vec) {
  Status ret = SUCCESS;
  for (const auto &node_ptr : node_vec) {
    ge::OpDescPtr op_desc = node_ptr->GetOpDesc();
    if (!ge::AttrUtils::HasAttr(op_desc, kAttrNameOriginalFusionGraph)) {
      FE_LOGD("Node[name:%s] is single op.", op_desc->GetNamePtr());
      ret = FuzzGeneralAndCompileSingleOp(node_ptr);
    } else {
      FE_LOGD("Node[name:%s] is fusion op.", op_desc->GetNamePtr());
      ret = FuzzGeneralAndCompileFusionOp(node_ptr);
    }
    if (ret != SUCCESS) {
      return ret;
    }
    FE_LOGD("Node[name:%s] fuzz compile success.", op_desc->GetName().c_str());
  }
  return SUCCESS;
}

void FEOpsKernelInfoStore::UpdateNodeShapeAndRange(const ge::NodePtr &node_ptr) const {
  // tefusion only update ori shape, update shape and range before compile
  ge::OpDescPtr op_desc = node_ptr->GetOpDesc();
  for (const auto &in_tensor_desc : op_desc->GetAllInputsDescPtr()) {
    UpdateTensorShapeAndRange(op_desc, in_tensor_desc);
  }

  for (const auto &out_tensor_desc : op_desc->GetAllOutputsDescPtr()) {
    UpdateTensorShapeAndRange(op_desc, out_tensor_desc);
  }
}

void FEOpsKernelInfoStore::UpdateTensorShapeAndRange(const ge::OpDescPtr &op_desc,
                                                     const ge::GeTensorDescPtr &tensor_desc) const {
  std::vector<std::pair<int64_t, int64_t>> ori_shape_range;
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  tensor_desc->GetOriginShapeRange(ori_shape_range);
  tensor_desc->GetShapeRange(shape_range);
  if (tensor_desc->GetOriginFormat() == static_cast<ge::Format>(ge::GetPrimaryFormat(tensor_desc->GetFormat()))) {
    tensor_desc->SetShape(tensor_desc->GetOriginShape());
    tensor_desc->SetShapeRange(ori_shape_range);
  } else {
    int64_t impl_type = EN_IMPL_HW_TBE;
    int64_t hidden_size = 1;
    int64_t input_size = 1;
    int64_t state_size = -1;
    (void)ge::AttrUtils::GetInt(op_desc, FE_IMPLY_TYPE, impl_type);
    (void)ge::AttrUtils::GetInt(op_desc, "hidden_size", hidden_size);
    (void)ge::AttrUtils::GetInt(op_desc, "input_size", input_size);
    (void)ge::AttrUtils::GetInt(op_desc, "state_size", state_size);
    CalcShapeExtraAttr extra_attr = {hidden_size, input_size, state_size};
    int64_t main_type = GetMainImplType<int64_t>(impl_type);
    ge::GeShape new_shape;
    std::vector<std::pair<int64_t, int64_t>> new_shape_range;
    FE_LOGD("Node name[%s] before shape:%s, shape range:%s.", op_desc->GetName().c_str(),
            ShapeToString(tensor_desc->GetShape().GetDims()).c_str(), RangeToString(shape_range).c_str());
    ShapeAndFormat input_shape_and_format_info = {tensor_desc->GetOriginShape(),
                                                  new_shape,
                                                  tensor_desc->GetOriginFormat(),
                                                  tensor_desc->GetFormat(),
                                                  tensor_desc->GetDataType(),
                                                  GROUPS_DEFAULT_VALUE,
                                                  extra_attr};
    (void)GetShapeAccordingToFormat(input_shape_and_format_info);
    RangeAndFormat input_range_and_format_info = {tensor_desc->GetOriginShape(),
                                                  ori_shape_range,
                                                  new_shape_range,
                                                  tensor_desc->GetOriginFormat(),
                                                  tensor_desc->GetFormat(),
                                                  tensor_desc->GetDataType(),
                                                  main_type,
                                                  extra_attr};
    (void)RangeTransferAccordingToFormat::GetRangeAccordingToFormat(input_range_and_format_info);
    tensor_desc->SetShape(input_shape_and_format_info.new_shape);
    tensor_desc->SetShapeRange(input_range_and_format_info.new_range);
    FE_LOGD("Node name[%s] after shape:%s, shape range:%s.", op_desc->GetName().c_str(),
            ShapeToString(tensor_desc->GetShape().GetDims()).c_str(),
            RangeToString(input_range_and_format_info.new_range).c_str());
  }
}

void FEOpsKernelInfoStore::ClearOpAttr(const ge::NodePtr &node_ptr) const {
  auto op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    return;
  }
  (void)op_desc->DelAttr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE);
  (void)op_desc->DelAttr(ge::TVM_ATTR_NAME_MAGIC);
  (void)op_desc->DelAttr(ge::TVM_ATTR_NAME_BLOCKDIM);
  (void)op_desc->DelAttr(kTaskRadio);
  (void)op_desc->DelAttr(kModeInArgsFirstField);
  (void)op_desc->DelAttr(kAttrIntercoreSync);
  (void)op_desc->DelAttr(ge::ATTR_N_BATCH_SPILT);
  (void)op_desc->DelAttr(ge::TVM_ATTR_NAME_WORKSPACE_TYPE);
  (void)op_desc->DelAttr(ge::TVM_ATTR_NAME_METADATA);
  (void)op_desc->DelAttr(op_desc->GetName() + kKernelName);
  (void)op_desc->DelAttr(kKernelName);
  (void)op_desc->DelAttr(ATTR_NAME_COMPRESS_PARAMETERS);
  (void)op_desc->DelAttr(ATTR_NAME_WEIGHT_REPEAT);
  (void)op_desc->DelAttr(OP_PARA_SIZE);
  (void)op_desc->DelAttr(ge::ATTR_NAME_TBE_KERNEL_NAME);
  (void)op_desc->DelAttr(ge::ATTR_NAME_TBE_KERNEL_BUFFER);
  (void)op_desc->DelAttr(ATTR_NAME_TBE_KERNEL_SIZE);
  (void)op_desc->DelAttr(ATTR_NAME_KERNEL_LIST_FIRST_NAME);
}

Status FEOpsKernelInfoStore::FuzzGeneralAndCompileSingleOp(const ge::NodePtr &node_ptr) {
  Status ret = GeneralizeSingleOp(node_ptr);
  if (ret != SUCCESS) {
    FE_LOGW("Node name[%s] generalize single op failed.", node_ptr->GetName().c_str());
    return ret;
  }

  ClearOpAttr(node_ptr);
  ret = CompileSingleOp(node_ptr, true);
  if (ret != SUCCESS) {
    FE_LOGW("Node name[%s] fuzz compile single op failed.", node_ptr->GetName().c_str());
  }
  return ret;
}

Status FEOpsKernelInfoStore::GeneralizeSingleOp(const ge::NodePtr &node_ptr) const {
  OpStoreAdapterPtr op_store_adapter = nullptr;
  if (OpStoreAdapterManager::Instance(engine_name_).GetOpStoreAdapter(EN_IMPL_HW_TBE, op_store_adapter) != SUCCESS) {
    FE_LOGW("get op store adapter failed");
    return FAILED;
  }

  NodeGeneralInfoPtr node_info_ptr;
  FE_MAKE_SHARED(node_info_ptr = std::make_shared<NodeGeneralInfo>(), return FAILED);
  Status ret = op_store_adapter->FeedNodeGeneralInfo(node_ptr, node_info_ptr);
  if (ret != SUCCESS) {
    FE_LOGW("Node[name:%s] get general info failed", node_ptr->GetName().c_str());
    return FAILED;
  }

  InputNodeGeneralize single_op_generalize(op_store_adapter);
  ret = single_op_generalize.GeneralizeOneNode(node_ptr, node_info_ptr);
  if (ret != SUCCESS) {
    FE_LOGW("Node[name:%s] generalize one node failed", node_ptr->GetName().c_str());
    return FAILED;
  }
  UpdateNodeShapeAndRange(node_ptr);
  return SUCCESS;
}

void FEOpsKernelInfoStore::BackupGraphParentNodeAndIndex(const ge::ComputeGraphPtr &graph,
                                                         FusionParenNodeAndIndex &node_map) const {
  // generalize fusion graph need remove parent node and dataop's parent_node_index attr
  // without parent node and parent node index, subgraph will be considered as origingraph
  node_map.parent_node_ptr = graph->GetParentNode();
  graph->SetParentNode(nullptr);
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == DATA) {
      int64_t node_index = -1;
      if (ge::AttrUtils::GetInt(node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, node_index)) {
        node_map.data_attr_parent_index_map.emplace(std::make_pair(node, node_index));
        node->GetOpDesc()->DelAttr(ge::ATTR_NAME_PARENT_NODE_INDEX);
      }
    }
  }
}

void FEOpsKernelInfoStore::RollbackGraphParentNodeAndIndex(const ge::ComputeGraphPtr &graph,
                                                           const FusionParenNodeAndIndex &node_map) const {
  graph->SetParentNode(node_map.parent_node_ptr);
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == DATA) {
      const auto iter = node_map.data_attr_parent_index_map.find(node);
      if (iter != node_map.data_attr_parent_index_map.end()) {
        ge::AttrUtils::SetInt(node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, iter->second);
      }
    }
  }
}

void FEOpsKernelInfoStore::SetConstValueAttr(ge::ConstNodePtr &const_node_ptr, const ge::OpDescPtr &op_desc_ptr) const {
  for (ge::InDataAnchorPtr &in_data_anchor : const_node_ptr->GetAllInDataAnchors()) {
    if (in_data_anchor == nullptr ||
        in_data_anchor->GetPeerOutAnchor() == nullptr ||
        in_data_anchor->GetPeerOutAnchor()->GetOwnerNode() == nullptr) {
      continue;
    }
    ge::NodePtr peer_in_node = in_data_anchor->GetPeerOutAnchor()->GetOwnerNode();
    if (peer_in_node->GetType() != CONSTANT && peer_in_node->GetType() != CONSTANTOP) {
      continue;
    }
    std::vector<ge::GeTensorPtr> ge_tensor_vec = ge::OpDescUtils::MutableWeights(peer_in_node);
    if (ge_tensor_vec.empty()) {
      continue;
    }
    ge::GeTensorDescPtr tensor_desc = op_desc_ptr->MutableInputDesc(static_cast<uint32_t>(in_data_anchor->GetIdx()));
    if (tensor_desc == nullptr) {
      continue;
    }
    (void)ge::AttrUtils::SetTensor(tensor_desc, ge::ATTR_NAME_VALUE, ge_tensor_vec[0]);
  }
}

bool FEOpsKernelInfoStore::GetNodeSupportInfo(const ge::OperatorPtr &op, std::string &support_info) {
  FE_CHECK(op == nullptr, FE_LOGE("OperatorPtr is nullptr."), return false);
  ge::ConstNodePtr const_node_ptr = ge::NodeUtilsEx::GetNodeFromOperator(*op);
  const ge::OpDescPtr op_desc_ptr = ge::OpDescUtils::GetOpDescFromOperator(*op);
  ge::ComputeGraphPtr graph_ptr = nullptr;
  FE_MAKE_SHARED(graph_ptr = std::make_shared<ge::ComputeGraph>(op->GetName()), return false);
  ge::NodePtr node_ptr = graph_ptr->AddNode(op_desc_ptr);
  FE_CHECK(node_ptr == nullptr, FE_LOGE("Node is nullptr."), return false);
  // 1. set const value for op_desc
  SetConstValueAttr(const_node_ptr, op_desc_ptr);

  // 2. check support
  std::string unsupport_reason;
  if (!CheckSupported(node_ptr, unsupport_reason)) {
    FE_LOGD("The op[%s, %s] is not supported, reason is [%s].",
            node_ptr->GetName().c_str(), node_ptr->GetType().c_str(), unsupport_reason.c_str());
    return false;
  }
  // 3. generate support info json
  int64_t op_impl_type = static_cast<int64_t>(EN_RESERVED);
  (void)ge::AttrUtils::GetInt(op_desc_ptr, FE_IMPLY_TYPE, op_impl_type);
  OpImplType impl_type = static_cast<OpImplType>(op_impl_type);
  OpKernelInfoPtr op_kernel_info_ptr =
          OpsKernelManager::Instance(engine_name_).GetOpKernelInfoByOpType(impl_type, op_desc_ptr->GetType());
  FE_CHECK(op_kernel_info_ptr == nullptr,
           FE_LOGD("Op kernel info of op[%s, %s] is not found.",
                   op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str()),
           return false);
  SubOpsStorePtr sub_ops_store_ptr = GetSubOpsStore(impl_type);
  FE_CHECK(sub_ops_store_ptr == nullptr,
           FE_LOGD("Sub op store[%s] is not found.", GetImplTypeString(impl_type).c_str()),
           return false);
  FormatDtypeInfo format_dtype_info;
  bool is_dynamic_impl = false;
  (void)ge::AttrUtils::GetBool(op_desc_ptr, ATTR_NAME_IS_OP_DYNAMIC_IMPL, is_dynamic_impl);
  Status ret = sub_ops_store_ptr->GetSupportFormatAndDtype(node_ptr, op_kernel_info_ptr, is_dynamic_impl,
                                                           format_dtype_info);
  if (ret != SUCCESS) {
    FE_LOGD("Format and data type info of op[%s, %s] is not found.",
            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
    return false;
  }
  GenerateOpSupportInfo(op_kernel_info_ptr, is_dynamic_impl, format_dtype_info.format_map,
                        format_dtype_info.data_type_map, support_info);
  FE_LOGD("The support info of op[%s, %s] is [%s].",
          op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), support_info.c_str());
  return true;
}

Status FEOpsKernelInfoStore::CompileSubGraph(const ge::ComputeGraphPtr &graph) {
  optimize_utility_->InferShape(graph);

  vector<ge::NodePtr> node_vec{};
  for (auto &fuzzy_node : graph->GetDirectNode()) {
    if (fuzzy_node->GetInAllNodes().empty() || fuzzy_node->GetOutAllNodes().empty()) {
      continue;
    }
    if (fuzzy_node->GetOpDesc()->HasAttr(ge::TVM_ATTR_NAME_MAGIC)) {
      // fusion node will check all nodes should have same tvm_magic
      fuzzy_node->GetOpDesc()->DelAttr(ge::TVM_ATTR_NAME_MAGIC);
    }
    node_vec.emplace_back(fuzzy_node);
  }

  return CompileMultipleOp(node_vec, true);
}

void FEOpsKernelInfoStore::CopyTensorDesc(const ge::GeTensorDescPtr &src, const ge::GeTensorDescPtr &dst) const {
  if (src == nullptr || dst == nullptr) {
    return;
  }
  const ge::GeShape &ori_src_shape = src->GetOriginShape();
  const ge::GeShape &src_shape = src->GetShape();
  dst->SetOriginShape(ori_src_shape);
  dst->SetShape(src_shape);

  std::vector<std::pair<int64_t, int64_t>> ori_src_shape_range;
  std::vector<std::pair<int64_t, int64_t>> src_shape_range;
  src->GetOriginShapeRange(ori_src_shape_range);
  src->GetShapeRange(src_shape_range);
  dst->SetOriginShapeRange(ori_src_shape_range);
  dst->SetShapeRange(src_shape_range);
  FE_LOGD("Ori Shape is:%s, shape is:%s, Ori Range is %s, range is %s.",
          ShapeToString(ori_src_shape.GetDims()).c_str(), ShapeToString(src_shape.GetDims()).c_str(),
          RangeToString(ori_src_shape_range).c_str(), RangeToString(src_shape_range).c_str());
}

Status FEOpsKernelInfoStore::CompareTensorDescAndSubgraphData(const ge::ComputeGraphPtr &graph,
                                                              const ge::NodePtr &node_ptr,
                                                              const bool is_to_subgraph) const {
  auto node_opdesc = node_ptr->GetOpDesc();
  for (auto &data_node : graph->GetDirectNode()) {
    if (data_node->GetType() != DATA) {
      continue;
    }
    const auto &data_opdesc = data_node->GetOpDesc();
    int32_t ref_i;
    if (!ge::AttrUtils::GetInt(data_opdesc, ge::ATTR_NAME_PARENT_NODE_INDEX, ref_i)) {
      FE_LOGW("node[%s] get parent_node_index failed", data_opdesc->GetName().c_str());
      return FAILED;
    }
    auto input_desc = node_opdesc->MutableInputDesc(ref_i);
    if (input_desc == nullptr) {
      FE_LOGW("node[%s] has no input[%d]", node_opdesc->GetName().c_str(), ref_i);
      return FAILED;
    }
    auto data_input_td = data_opdesc->MutableInputDesc(0);
    auto data_output_td = data_opdesc->MutableOutputDesc(0);
    if (is_to_subgraph) {
      // clear shape range, re-GeneralizeGraph
      input_desc->SetOriginShapeRange({});
      input_desc->SetShapeRange({});
      CopyTensorDesc(input_desc, data_input_td);
      CopyTensorDesc(input_desc, data_output_td);
    } else {
      CopyTensorDesc(data_input_td, input_desc);
    }
  }
  return SUCCESS;
}

void FEOpsKernelInfoStore::UpdateSubGraphShapeAndRange(const ge::ComputeGraphPtr &graph) const {
  for (auto &data_node : graph->GetDirectNode()) {
    if (data_node->GetType() != DATA) {
      continue;
    }
    UpdateNodeShapeAndRange(data_node);
  }
}

Status FEOpsKernelInfoStore::FuzzGeneralAndCompileFusionOp(const ge::NodePtr &node_ptr) {
  ge::OpDescPtr op_desc = node_ptr->GetOpDesc();
  ge::ComputeGraphPtr graph_ptr = nullptr;
  if (!ge::AttrUtils::GetGraph(op_desc, kAttrNameOriginalFusionGraph, graph_ptr)) {
    FE_LOGW("Op[name:%s] does not have the attribute _original_fusion_graph", op_desc->GetName().c_str());
    return FAILED;
  }

  Status ret;
  ret = CompareTensorDescAndSubgraphData(graph_ptr, node_ptr, true);
  if (ret != SUCCESS) {
    return FAILED;
  }
  FusionParenNodeAndIndex node_map{};
  BackupGraphParentNodeAndIndex(graph_ptr, node_map);

  FuzzyGeneralize fusionop_fuzzy_generalize(optimize_utility_, shared_from_this(),
                                            fusion_priority_mgr_ptr_);
  ret = fusionop_fuzzy_generalize.GeneralizeGraph(*graph_ptr);
  if (ret != SUCCESS) {
    FE_LOGW("Op[name:%s] generalize graph failed", op_desc->GetName().c_str());
    goto out;
  }

  UpdateSubGraphShapeAndRange(graph_ptr);
  ret = CompileSubGraph(graph_ptr);
  if (ret != SUCCESS) {
    FE_LOGW("Op[name:%s] compile subgraph failed", op_desc->GetName().c_str());
    goto out;
  }
out:
  RollbackGraphParentNodeAndIndex(graph_ptr, node_map);
  if (ret == SUCCESS) {
    (void)CompareTensorDescAndSubgraphData(graph_ptr, node_ptr, false);
  }
  return ret;
}

Status FEOpsKernelInfoStore::GetOpStoreAdapter(const OpImplType &impl_type, OpStoreAdapterPtr &op_store_adapter) {
  if (OpStoreAdapterManager::Instance(engine_name_).GetOpStoreAdapter(impl_type, op_store_adapter) != SUCCESS) {
    REPORT_FE_ERROR("Failed to get op_store_adapter by op impl type [%ld].", impl_type);
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace fe

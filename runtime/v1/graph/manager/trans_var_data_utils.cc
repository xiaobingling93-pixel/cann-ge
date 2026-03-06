/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/manager/trans_var_data_utils.h"

#include "common/math/math_util.h"
#include "formats/formats.h"
#include "base/err_mgr.h"
#include "formats/utils/formats_trans_utils.h"
#include "common/datatype_transfer.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "common/thread_pool.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
namespace {
constexpr uint32_t kDefaultVarTransThreadNum = 16U;

class RtContextSwitchGuard {
 public:
  RtContextSwitchGuard(const rtCtxMode_t mode, const uint32_t device_id) {
    auto ret = rtCtxGetCurrent(&last_);
    if (ret != RT_ERROR_NONE) {
      REPORT_INNER_ERR_MSG("E19999", "Call rtCtxGetCurrent failed, device_id:%u, ret:%d,",
                        device_id, ret);
      GELOGE(RT_FAILED, "[Call][RtCtxGetCurrent] Failed to get current context, device_id:%u, ret:%d",
             device_id, ret);
      return;
    }

    if (rtCtxCreate(&current_, mode, static_cast<int32_t>(device_id)) != RT_ERROR_NONE) {
      REPORT_INNER_ERR_MSG("E19999", "Call CtxSetCurrent failed, device_id:%u", device_id);
      GELOGE(RT_FAILED, "[Call][RtCtxSetCurrent] failed, device_id:%u", device_id);
      return;
    }

    ret = rtCtxSetCurrent(current_);
    if (ret != RT_ERROR_NONE) {
      REPORT_INNER_ERR_MSG("E19999", "Call rtCtxSetCurrent failed, device_id:%u, ret:%d", device_id, ret);
      GELOGE(RT_FAILED, "[Call][RtCtxSetCurrent] failed, device_id:%u, ret:%d", device_id, ret);
      return;
    }
    GELOGD("Create and switch rt context %p type %d for device %u, backup last %p.", current_, mode, device_id, last_);
  }

  ~RtContextSwitchGuard() {
    try {
      if (current_ != nullptr) {
        const auto ret = rtCtxDestroy(current_);
        GELOGD("Destory current context %p result %d", current_, ret);
      }
      if (last_ != nullptr) {
        const auto ret = rtCtxSetCurrent(last_);
        GELOGD("Recovery last context %p result %d.", last_, ret);
      }
    } catch (...) {
      // no processing
    }
  }

 private:
  rtContext_t last_ = nullptr;
  rtContext_t current_ = nullptr;
};

int64_t CalcVarSizeInBytes(const GeTensorDesc &desc) {
  int64_t var_size = GetSizeByDataType(desc.GetDataType());
  if (var_size <= 0) {
    REPORT_INNER_ERR_MSG("E19999", "Data type:%s in desc, it's size:%" PRId64 " < 0, check invalid",
                       TypeUtils::DataTypeToSerialString(desc.GetDataType()).c_str(), var_size);
    GELOGE(PARAM_INVALID, "[Calc][VarDataSize] by data type %s failed.",
           TypeUtils::DataTypeToSerialString(desc.GetDataType()).c_str());
    return -1;
  }
  const auto shape = desc.GetShape();
  const auto dim_num = shape.GetDimNum();
  for (size_t dim_index = 0U; dim_index < dim_num; ++dim_index) {
    if (CheckInt64MulOverflow(var_size, shape.GetDim(dim_index)) != SUCCESS) {
      return -1;
    }
    var_size *= shape.GetDim(dim_index);
  }
  return var_size;
}

Status CopyVarToDevice(const NodePtr &var, const formats::TransResult &trans_result, void *const var_addr) {
  GE_CHECK_NOTNULL(var);
  GELOGD("Copy var %s from host to device, size %zu", var->GetName().c_str(), trans_result.length);
  const auto ret = rtMemcpy(var_addr, trans_result.length, PtrToPtr<uint8_t, void>(trans_result.data.get()),
                            trans_result.length, RT_MEMCPY_HOST_TO_DEVICE);
  if (ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtMemcpy failed, op:%s(%s), size:%" PRIu64 ", ret:%d,", var->GetName().c_str(),
                      var->GetType().c_str(), static_cast<uint64_t>(trans_result.length), ret);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, op:%s(%s), size:%" PRIu64 ", ret:%d,", var->GetName().c_str(),
           var->GetType().c_str(), trans_result.length, ret);
    return RT_FAILED;
  }
  return SUCCESS;
}

Status CopyVarFromDevice(const uint64_t session_id, const NodePtr &var, std::unique_ptr<uint8_t[]> &var_data,
                         const GeTensorDesc &input_desc, const uint32_t device_id) {
  uint8_t *var_logic = nullptr;
  GE_CHECK_NOTNULL(var);
  GE_CHECK_NOTNULL(VarManager::Instance(session_id));
  const Status ret = VarManager::Instance(session_id)->GetVarAddr(var->GetName(), input_desc, var_logic);
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Get][VarAddr] failed, node:%s, session_id:%" PRIu64 ", ret:%d.", var->GetName().c_str(),
           session_id, ret);
    return INTERNAL_ERROR;
  }

  uint8_t *const var_addr = VarManager::Instance(session_id)->GetVarMemoryAddr(var_logic, RT_MEMORY_HBM, device_id);
  if (var_addr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Get variable memory addr failed, mem_type:%u, op:%s(%s), session_id:%" PRIu64 "",
                      RT_MEMORY_HBM, var->GetName().c_str(), var->GetType().c_str(), session_id);
    GELOGE(INTERNAL_ERROR, "[Get][VarMemoryAddr] failed, mem_type:%u, op:%s(%s), session_id:%" PRIu64,
           RT_MEMORY_HBM, var->GetName().c_str(), var->GetType().c_str(), session_id);
    return INTERNAL_ERROR;
  }

  const int64_t var_size_bytes = CalcVarSizeInBytes(input_desc);
  if (var_size_bytes <= 0) {
    return INTERNAL_ERROR;
  }

  std::unique_ptr<uint8_t[]> var_host = MakeUnique<uint8_t[]>(static_cast<size_t>(var_size_bytes));
  if (var_host == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New host memory failed, size:%" PRId64 ", op:%s(%s), session_id:%" PRIu64 "",
                      var_size_bytes, var->GetName().c_str(), var->GetType().c_str(), session_id);
    GELOGE(OUT_OF_MEMORY, "[New][Memory] for rt-host failed, size:%" PRId64 ", op:%s(%s), session_id:%" PRIu64,
           var_size_bytes, var->GetName().c_str(), var->GetType().c_str(), session_id);
    return OUT_OF_MEMORY;
  }

  const auto rt_rslt = rtMemcpy(PtrToPtr<uint8_t, void>(var_host.get()), static_cast<uint64_t>(var_size_bytes),
                                PtrToPtr<uint8_t, void>(var_addr), static_cast<uint64_t>(var_size_bytes),
                                RT_MEMCPY_DEVICE_TO_HOST);
  if (rt_rslt != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtMemcpy failed, size:%" PRId64 ", op:%s(%s), session_id:%" PRIu64 ", ret:%d",
                      var_size_bytes, var->GetName().c_str(), var->GetType().c_str(), session_id, rt_rslt);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%" PRId64 ", op:%s(%s), session_id:%" PRIu64 ", ret:%d",
           var_size_bytes, var->GetName().c_str(), var->GetType().c_str(), session_id, rt_rslt);
    return RT_FAILED;
  }

  GELOGD("Copy var %s from device to host, size %" PRId64, var->GetName().c_str(), var_size_bytes);
  var_data.swap(var_host);

  GELOGI("var_logic:%p, var_addr:%p", var_logic, var_addr);

  return SUCCESS;
}
bool IsNoNeedTrans(const std::string &node_type) {
  return (node_type == RESHAPE) || (node_type == REFORMAT) ||
         (node_type == SQUEEZEV2) || (node_type == UNSQUEEZEV2);
}
Status TransVarOnHost(uint8_t *const var_data, const VarTransRoad &trans_road, formats::TransResult &result) {
  formats::TransResult result_last_time{};
  bool use_init_data = true;
  for (const auto &trans_info : trans_road) {
    if (IsNoNeedTrans(trans_info.node_type)) {
      GELOGD("Skip to trans variable data on the reshape/reformat node");
      continue;
    }
    uint8_t *src_data = nullptr;
    if (use_init_data) {
      src_data = var_data;
      use_init_data = false;
    } else {
      src_data = result_last_time.data.get();
    }

    formats::TransResult tmp_result{};
    if ((trans_info.node_type == TRANSDATA) || (trans_info.node_type == TRANSPOSED)) {
      const auto src_format = trans_info.input.GetFormat();
      const auto src_shape = trans_info.input.GetShape().GetDims();
      const auto dst_format = trans_info.output.GetFormat();
      const auto dst_shape = trans_info.output.GetShape().GetDims();
      const auto data_type = trans_info.input.GetDataType();
      const Format src_primary_format = static_cast<Format>(GetPrimaryFormat(src_format));
      const Format dst_primary_format = static_cast<Format>(GetPrimaryFormat(dst_format));
      const Format src_sub_format = static_cast<Format>(GetSubFormat(src_format));
      const Format dst_sub_format = static_cast<Format>(GetSubFormat(dst_format));
      const int64_t src_c0_format = GetC0Value(static_cast<int32_t>(src_format));
      const int64_t dst_c0_format = GetC0Value(static_cast<int32_t>(dst_format));
      GELOGD("Trans format from %s to %s, primary formats from %s to %s, src c0 is %" PRId64 ", dts c0 is %" PRId64 ", "
             "shape %s to %s, data-type %s",
             TypeUtils::FormatToSerialString(src_format).c_str(), TypeUtils::FormatToSerialString(dst_format).c_str(),
             TypeUtils::FormatToSerialString(src_primary_format).c_str(),
             TypeUtils::FormatToSerialString(dst_primary_format).c_str(), src_c0_format, dst_c0_format,
             ToString(src_shape).c_str(), ToString(dst_shape).c_str(),
             TypeUtils::DataTypeToSerialString(data_type).c_str());
      const Status ret = formats::TransDataFormat(
          {src_data, src_format, dst_format, src_primary_format, dst_primary_format, src_sub_format, dst_sub_format,
           src_c0_format, dst_c0_format, src_shape, dst_shape, data_type},
          tmp_result);
      if (ret != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Trans format from %s to %s, primary formats from %s to %s, src c0 is "
                                    "[%" PRId64 "], dts c0 is [%" PRId64 "], shape %s to %s failed, "
                                    "data type:%s, ret:%u,",
                          TypeUtils::FormatToSerialString(src_format).c_str(),
                          TypeUtils::FormatToSerialString(dst_format).c_str(),
                          TypeUtils::FormatToSerialString(src_primary_format).c_str(),
                          TypeUtils::FormatToSerialString(dst_primary_format).c_str(), src_c0_format, dst_c0_format,
                          ToString(src_shape).c_str(), ToString(dst_shape).c_str(),
                          TypeUtils::DataTypeToSerialString(data_type).c_str(), ret);
        GELOGE(INTERNAL_ERROR, "[Trans][Format] from %s to %s, primary formats from %s to %s, shape %s to %s failed, "
                               "src c0 is %" PRId64 ", dts c0 is %" PRId64 ", data type %s error code %u",
               TypeUtils::FormatToSerialString(src_format).c_str(), TypeUtils::FormatToSerialString(dst_format).c_str(),
               TypeUtils::FormatToSerialString(src_primary_format).c_str(),
               TypeUtils::FormatToSerialString(dst_primary_format).c_str(),
               ToString(src_shape).c_str(), ToString(dst_shape).c_str(), src_c0_format, dst_c0_format,
               TypeUtils::DataTypeToSerialString(data_type).c_str(), ret);
        return ret;
      }
    } else if (trans_info.node_type == CAST) {
      const auto input_shape = trans_info.input.GetShape();
      const int64_t src_data_size = (input_shape.GetShapeSize() == 0) ? 1 : input_shape.GetShapeSize();
      const auto src_data_type = trans_info.input.GetDataType();
      const auto dst_data_type = trans_info.output.GetDataType();
      GELOGD("Trans data type from %s to %s, input shape %s, data size %" PRId64,
             TypeUtils::DataTypeToSerialString(src_data_type).c_str(),
             TypeUtils::DataTypeToSerialString(dst_data_type).c_str(), ToString(input_shape.GetDims()).c_str(),
             src_data_size);
      const Status ret = formats::TransTensorDataType(
          {src_data, static_cast<size_t>(src_data_size), src_data_type, dst_data_type}, tmp_result);
      if (ret != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Trans data type from %s to %s failed, input shape %s, "
			  "data size %" PRId64 ", ret:%u",
                          TypeUtils::DataTypeToSerialString(src_data_type).c_str(),
                          TypeUtils::DataTypeToSerialString(dst_data_type).c_str(),
                          ToString(input_shape.GetDims()).c_str(), src_data_size, ret);
        GELOGE(INTERNAL_ERROR, "[Trans][DataType] from %s to %s failed, input shape %s, "
               "data size %" PRId64 ", error code %u",
               TypeUtils::DataTypeToSerialString(src_data_type).c_str(),
               TypeUtils::DataTypeToSerialString(dst_data_type).c_str(), ToString(input_shape.GetDims()).c_str(),
               src_data_size, ret);
        return ret;
      }
    } else {
      REPORT_INNER_ERR_MSG("E19999", "Trans var data failed, the trans type %s does not supported, check invalid",
                         trans_info.node_type.c_str());
      GELOGE(UNSUPPORTED, "[Trans][VarData] failed, the trans type %s does not supported",
             trans_info.node_type.c_str());
      return UNSUPPORTED;
    }
    result_last_time = tmp_result;
  }

  result = result_last_time;
  return SUCCESS;
}

/// re-alloc var memory on device using var-manager
/// free origin var memory(var manager does not support now)
/// @param session_id
/// @param var
/// @param var_size_bytes
/// @param var_device
/// @return
Status ReAssignVarAddr(const uint64_t session_id,
                       const std::string &var_name,
                       const GeTensorDesc &tensor_desc,
                       void *&var_device,
                       const uint32_t device_id) {
  GE_CHECK_NOTNULL(VarManager::Instance(session_id));
  uint8_t *var_logic = nullptr;
  const Status ret = VarManager::Instance(session_id)->GetVarAddr(var_name, tensor_desc, var_logic);
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Get][VarAddr] failed, var name:%s, session_id:%" PRIu64 ", ret:%u",
           var_name.c_str(), session_id, ret);
    return INTERNAL_ERROR;
  }

  uint8_t *const var_addr = VarManager::Instance(session_id)->GetVarMemoryAddr(var_logic, RT_MEMORY_HBM, device_id);
  if (var_addr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Get variable memory addr failed, mem_type:%u, var_name:%s, session_id:%" PRIu64 ",",
                      RT_MEMORY_HBM, var_name.c_str(), session_id);
    GELOGE(INTERNAL_ERROR, "[Get][VarMemoryAddr] failed, mem_type:%u, var_name:%s, session_id:%" PRIu64,
           RT_MEMORY_HBM, var_name.c_str(), session_id);
    return INTERNAL_ERROR;
  }
  var_device = var_addr;

  GELOGI("var_logic:%p, var_addr:%p", var_logic, var_addr);

  return SUCCESS;
}

Status TransVarData(const NodePtr &var, const VarTransRoad &trans_road, const uint64_t session_id,
                    const uint32_t device_id) {
  // do not need to do anything if only all reshape/reformat node on the trans_road
  GE_CHECK_NOTNULL(var);
  bool need_trans = false;
  for (auto &road : trans_road) {
    if (!IsNoNeedTrans(road.node_type)) {
      need_trans = true;
      break;
    }
  }
  if (!need_trans) {
    return SUCCESS;
  }

  // Sync var data from device
  std::unique_ptr<uint8_t[]> var_data;
  if (trans_road.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "Param trans_road is empty, session_id:%" PRIu64 ", check invalid", session_id);
    GELOGE(INTERNAL_ERROR, "[Check][Param] trans_road is empty, session_id:%" PRIu64, session_id);
    return INTERNAL_ERROR;
  }
  const GeTensorDesc &input_desc = trans_road.begin()->input;
  auto ret = CopyVarFromDevice(session_id, var, var_data, input_desc, device_id);
  if (ret != SUCCESS) {
    return ret;
  }

  formats::TransResult trans_result{};
  ret = TransVarOnHost(var_data.get(), trans_road, trans_result);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][TransVarOnHost] failed, session_id:%" PRIu64 ", ret:%u", session_id, ret);
    return ret;
  }

  void *var_device = nullptr;

  /// It is a temporary solution to use the last GeTensorDesc to assign variable memory because the variable manager
  /// depends on TensorDesc and it is difficult to be modified. The correct solution is to assign memory based on the
  /// size of the converted variable. To complete the final solution, the dependency of the variable manager on
  /// TensorDesc needs to be removed. This change is large and needs to be performed step by step.
  ret = ReAssignVarAddr(session_id, var->GetName(), trans_road.rbegin()->output, var_device, device_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][ReAssignVarAddr] failed, session id:%" PRIu64 ", op:%s, ret:%u",
           session_id, var->GetName().c_str(), ret);
    return ret;
  }

  // sync new data to device
  ret = CopyVarToDevice(var, trans_result, var_device);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][CopyVarToDevice] failed, var:%s, ret:%u", var->GetName().c_str(), ret);
    return ret;
  }

  return SUCCESS;
}

Status TransTensor(const uint8_t *const var_data, const NodePtr &var_src, const NodePtr &var_dst,
                   formats::TransResult &result) {
  GE_CHECK_NOTNULL(var_src);
  GE_CHECK_NOTNULL(var_src->GetOpDesc());
  GE_CHECK_NOTNULL(var_dst);
  GE_CHECK_NOTNULL(var_dst->GetOpDesc());
  const int64_t src_data_shape_size = var_src->GetOpDesc()->GetOutputDesc(0U).GetShape().GetShapeSize();
  const auto src_data_datatype = var_src->GetOpDesc()->GetOutputDesc(0U).GetDataType();
  const auto dst_data_datatype = var_dst->GetOpDesc()->GetOutputDesc(0U).GetDataType();
  GE_IF_BOOL_EXEC(
      src_data_datatype != dst_data_datatype,
      const Status ret = formats::TransTensorDataType(
          {var_data, static_cast<size_t>(src_data_shape_size), src_data_datatype, dst_data_datatype}, result);
      if (ret != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Trans data type from %s to %s failed, data size %" PRId64 ", ret:%u",
                          TypeUtils::DataTypeToSerialString(src_data_datatype).c_str(),
                          TypeUtils::DataTypeToSerialString(dst_data_datatype).c_str(), src_data_shape_size, ret);
        GELOGE(INTERNAL_ERROR, "[Trans][DataType] from %s to %s failed, data size %" PRId64 ", ret:%u",
               TypeUtils::DataTypeToSerialString(src_data_datatype).c_str(),
               TypeUtils::DataTypeToSerialString(dst_data_datatype).c_str(), src_data_shape_size, ret);
        return ret;
      });
  return SUCCESS;
}

Status CopyTensorFromSrcVarNode(const NodePtr &var_src,
                                const NodePtr &var_dst,
                                const uint64_t session_id,
                                const uint32_t device_id) {
  /// after FE fusion pass, input num of applymomentum op was changed, 0th input is var_fp32, 6th input is
  /// var_fp16(new).
  /// unlink edges between var_fp32 and "dst_node" (need fp16) of var_fp32, add edge between var_fp16 and dst_node.
  /// need copy value from var_fp32 to var_fp16.
  /// [opdesc of var_src and var_dst are checked before passed in, no need to check if they are nullptr]
  GE_IF_BOOL_EXEC((var_src == nullptr) || (var_dst == nullptr) ||
                  (var_src->GetOpDesc() == nullptr) || (var_dst->GetOpDesc() == nullptr),
                  REPORT_INNER_ERR_MSG("E19999", "Param var_src or var_dst is nullptr, session_id:"
			             "%" PRIu64 ", device_id:%u, check invalid", session_id, device_id);
                  GELOGE(FAILED, "[Check][Param] Param var_src or var_dst is nullptr, "
			         "session_id:%" PRIu64 ", device_id:%u", session_id, device_id);
                  return FAILED);
  // src_node output_desc (fp32)
  const GeTensorDesc output_desc = var_src->GetOpDesc()->GetOutputDesc(0U);
  const DataType src_data_type = output_desc.GetDataType();
  const GeShape src_shape = output_desc.GetShape();
  const Format src_format = output_desc.GetFormat();
  GELOGI("src_node %s, src_format %s, src_shape %s, src_type %s.", var_src->GetName().c_str(),
         TypeUtils::FormatToSerialString(src_format).c_str(), ToString(src_shape.GetDims()).c_str(),
         TypeUtils::DataTypeToSerialString(src_data_type).c_str());
  // dst_node output_desc (fp16)
  const GeTensorDesc dst_tensor_desc = var_dst->GetOpDesc()->GetOutputDesc(0U);
  const DataType data_type = dst_tensor_desc.GetDataType();
  const GeShape data_shape = dst_tensor_desc.GetShape();
  const Format data_format = dst_tensor_desc.GetFormat();
  GELOGI("dst_node %s, src_format %s, src_shape %s, src_type %s.", var_dst->GetName().c_str(),
         TypeUtils::FormatToSerialString(data_format).c_str(), ToString(data_shape.GetDims()).c_str(),
         TypeUtils::DataTypeToSerialString(data_type).c_str());
  // Sync var data from device
  std::unique_ptr<uint8_t[]> var_src_data;
  const RtContextSwitchGuard switch_context(RT_CTX_NORMAL_MODE, device_id);
  // copy from src_node
  auto ret = CopyVarFromDevice(session_id, var_src, var_src_data, output_desc, device_id);
  GE_IF_BOOL_EXEC(ret != SUCCESS,
                  GELOGE(FAILED, "[Call][CopyVarFromDevice] failed, session id:%" PRIu64 ", var_src:%s",
                         session_id, var_src->GetName().c_str());
                  return ret);
  // trans dtype
  formats::TransResult trans_result{};
  ret = TransTensor(var_src_data.get(), var_src, var_dst, trans_result);
  GE_IF_BOOL_EXEC(ret != SUCCESS,
                  GELOGE(INTERNAL_ERROR, "[Trans][Tensor] failed, var_src:%s, var_dst:%s",
                         var_src->GetName().c_str(), var_dst->GetName().c_str());
                  return ret);
  // reset src value.
  void *var_device = nullptr;
  ret = ReAssignVarAddr(session_id, var_dst->GetName(), dst_tensor_desc, var_device, device_id);
  GE_IF_BOOL_EXEC(ret != SUCCESS,
                  GELOGE(INTERNAL_ERROR, "[Call][ReAssignVarAddr] failed, session id:%" PRIu64 ", var_dst:%s",
                         session_id, var_dst->GetName().c_str());
                  return ret);
  // copy to device
  ret = CopyVarToDevice(var_dst, trans_result, var_device);
  GE_IF_BOOL_EXEC(ret != SUCCESS,
                  GELOGE(ret, "[Call][CopyVarToDevice] failed, var_dst:%s, ret:%u",
                         var_dst->GetName().c_str(), ret);
                  return ret);
  return SUCCESS;
}
} // namespace
Status TransVarDataUtils::TransAllVarData(const std::vector<NodePtr> &variable_nodes,
                                          const uint64_t session_id,
                                          const uint32_t graph_id,
                                          const uint32_t device_id) {
  if (variable_nodes.empty()) {
    GELOGI("No vars need trans, just return.");
    return SUCCESS;
  }

  rtContext_t context = nullptr;
  GE_CHK_RT_RET(rtCtxGetCurrent(&context));

  ThreadPool executor("ge_vartrans", kDefaultVarTransThreadNum, true);
  std::vector<std::future<Status>> vector_future;
  for (auto &node : variable_nodes) {
    if (node == nullptr) {
      continue;
    }

    if (node->GetType() != VARIABLE) {
      continue;
    }

    auto const trans_func = [](const NodePtr &inner_node, const uint64_t inner_session_id, rtContext_t const ctx,
                               const uint32_t inner_graph_id, const uint32_t inner_device_id,
                               const error_message::ErrorManagerContext &error_context) -> Status {
      error_message::SetErrMgrContext(error_context);
      const rtError_t rt_ret = rtCtxSetCurrent(ctx);
      if (rt_ret != RT_ERROR_NONE) {
        REPORT_INNER_ERR_MSG("E19999", "Call rtCtxSetCurrent failed, session_id:%" PRIu64 ", graph_id:%u, ret:%d.",
                          inner_session_id, inner_graph_id, rt_ret);
        GELOGE(RT_FAILED, "[Call][RtCtxSetCurrent] failed, session_id:%" PRIu64 ", graph_id:%u, ret:%d.",
          inner_session_id, inner_graph_id, rt_ret);
        return RT_ERROR_TO_GE_STATUS(rt_ret);
      }
      uint32_t allocated_graph_id = 0U;
      GE_CHECK_NOTNULL(VarManager::Instance(inner_session_id));
      Status ret = VarManager::Instance(inner_session_id)->GetAllocatedGraphId(inner_node->GetName(),
                                                                               allocated_graph_id);
      if (ret != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Get allocated GraphId failed, session_id:%" PRIu64 ", graph_id:%u, ret:0x%X.",
                          inner_session_id, inner_graph_id, ret);
        GELOGE(INTERNAL_ERROR, "[Get][AllocatedGraphId] failed, node:%s, graph_id:%u.", inner_node->GetName().c_str(),
               inner_graph_id);
        return INTERNAL_ERROR;
      }
      uint32_t changed_graph_id = 0U;
      ret = VarManager::Instance(inner_session_id)->GetChangedGraphId(inner_node->GetName(), changed_graph_id);
      const bool call_trans_var = ((ret == SUCCESS) && (changed_graph_id == inner_graph_id) &&
                                   (changed_graph_id != allocated_graph_id));
      if (call_trans_var) {
        GELOGI("VarManager::GetChangedGraphId() success, node:%s, graph_id:%u.", inner_node->GetName().c_str(),
               inner_graph_id);
        VarTransRoad *const trans_road = VarManager::Instance(inner_session_id)->GetTransRoad(inner_node->GetName());
        if (trans_road == nullptr) {
          GELOGI("The variable %s does not have any trans road", inner_node->GetName().c_str());
          return SUCCESS;
        }
        ret = TransVarData(inner_node, *trans_road, inner_session_id, inner_device_id);
        if (ret != SUCCESS) {
          GELOGE(INTERNAL_ERROR, "[Trans][VarData] failed, node:%s, graph_id:%u, session_id:%" PRIu64 ".",
                 inner_node->GetName().c_str(), inner_graph_id, inner_session_id);
          return INTERNAL_ERROR;
        }
        VarManager::Instance(inner_session_id)->RemoveChangedGraphId(inner_node->GetName());
      }
      return SUCCESS;
    };

    std::future<Status> f = executor.commit(trans_func, node, session_id, context, graph_id, device_id,
                                            error_message::GetErrMgrContext());
    if (!f.valid()) {
      GELOGE(FAILED, "[Check][Param] Future is invalid, session id:%" PRIu64 ", graph id:%u", session_id, graph_id);
      return FAILED;
    }
    vector_future.push_back(std::move(f));
  }

  Status ret_status;
  for (size_t i = 0U; i < vector_future.size(); ++i) {
    ret_status = vector_future[i].get();
    if (ret_status != SUCCESS) {
      GELOGE(ret_status, "[Check][Param] trans %zu vardata failed.", i);
      return ret_status;
    }
  }

  return SUCCESS;
}

Status TransVarDataUtils::CopyVarData(const ComputeGraphPtr &compute_graph, const std::vector<NodePtr> &variable_nodes,
                                      const uint64_t session_id, const uint32_t device_id) {
  GELOGD("CopyVarData start: session_id:%" PRIu64 ".", session_id);
  if (compute_graph == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param compute_graph is nullptr, session_id:%" PRIu64 ", device_id:%u, check invalid.",
                       session_id, device_id);
    GELOGE(FAILED, "[Check][Param] compute_graph is nullptr, session_id:%" PRIu64 ", device_id:%u.",
      session_id, device_id);
    return FAILED;
  }

  bool copy_value = false;
  for (const auto &node : variable_nodes) {
    GE_IF_BOOL_EXEC(((node->GetOpDesc() == nullptr) || (node->GetOpDesc()->GetType() != VARIABLE)), continue);
    auto cp_from_node = ge::AttrUtils::GetStr(node->GetOpDesc(), "_copy_from_var_node");
    if ((cp_from_node != nullptr) && (cp_from_node->length() != 0U)) {
      GELOGI("Get original type of cp_from_node");
      (void)ge::AttrUtils::GetBool(node->GetOpDesc(), "_copy_value", copy_value);  // no need to check value
      if (!copy_value) {
        const auto src_node = compute_graph->FindNode(*cp_from_node);
        GE_CHECK_NOTNULL(src_node);
        GELOGI("current_var_node__: [%s] copy_from_var_node__: [%s].", node->GetName().c_str(),
               src_node->GetName().c_str());
        const auto ret = CopyTensorFromSrcVarNode(src_node, node, session_id, device_id);
        GE_IF_BOOL_EXEC(ret != SUCCESS,
                        GELOGE(FAILED, "[Copy][Tensor] failed, src_node:%s, node:%s, session_id:%" PRIu64 ", "
                          "device_id:%u", src_node->GetName().c_str(), node->GetName().c_str(), session_id, device_id);
                        return FAILED);
        // only copy once
        (void)ge::AttrUtils::SetBool(node->GetOpDesc(), "_copy_value", true);  // no need to check value
      }
    }
  }
  return SUCCESS;
}
}  // namespace ge

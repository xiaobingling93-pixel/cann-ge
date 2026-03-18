/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cpu_kernel_builder.h"
#include <runtime/rt.h>
#include "fwk_adpt_struct.h"
#include "aicpu_task_struct.h"
#include "util/constant.h"
#include "error_code/error_code.h"
#include "util/log.h"
#include "util/util.h"
#include "config/ops_json_file.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/compute_graph.h"
#include "graph/detail/attributes_holder.h"
#include "graph/op_desc.h"
#include "proto/aicpu/cpu_attr.pb.h"
#include "proto/aicpu/cpu_node_def.pb.h"
#include "proto/aicpu/cpu_tensor.pb.h"
#include "proto/aicpu/cpu_tensor_shape.pb.h"
#include "cpu_engine_util.h"
#include "common/sgt_slice_type.h"
#include "framework/common/types.h"

namespace {
const std::string kDefaultKernelSo = "libcpu_kernels.so";
const std::string kDefaultFunctionName = "RunCpuKernel";
const std::string kEngineNameAicpuFfts = "ffts_plus_aicpu_ascend";

constexpr uint32_t kCceAiCpuKernelType = 1; /* cce aicpu, same with CCE_AI_CPU */
constexpr uint32_t kAiCpuKernelType = 6;    /* custom aicpu, same with AI_CPU */
constexpr uint32_t kCustAiCpuKernelType = 7; /* custom aicpu, same CUST_AI_CPU */
constexpr uint32_t kHostAiCpuKernelType = 8; /* host aicpu same with HOST_CPU */
constexpr uint32_t kFftsAiCpuKernelType = 2; /* aicpu same with device侧的KERNEL_TYPE_AICPU */
constexpr uint32_t kFftsCustAiCpuKernelType = 4; /* host aicpu same with HOST_CPU */
constexpr uint32_t kBasicAicpuOpSqeNumber = 1;    /* sqe number of basic operation */
constexpr uint32_t kAsyncAicpuOpSqeNumber = 3;    /* sqe number of async operation */

static uint64_t g_aicpu_kernel_id = 0;
static uint64_t g_aicpu_session_id = 0;
const uint64_t kDataAndShapeNum = 2;
static int64_t g_op_index;

uint64_t GenerateUniqueKernelId()
{
    if (g_aicpu_kernel_id == ULLONG_MAX) {
        g_aicpu_kernel_id = 0;
    }
    return g_aicpu_kernel_id++;
}

uint64_t GenerateUniqueSessionId()
{
    if (g_aicpu_session_id == ULLONG_MAX) {
        g_aicpu_session_id = 0;
    }
    return g_aicpu_session_id++;
}
}

namespace aicpu {
ge::Status CpuKernelBuilder::CalcOpRunningParam(const ge::Node &node) const
{
    std::string node_name = node.GetName();
    std::string node_type = node.GetType();
    AICPUE_LOGI("CPUKernel's op %s[%s] run CalcOpRunningParam", node_name.c_str(), node_type.c_str());

    std::shared_ptr<ge::OpDesc> op_desc_ptr = node.GetOpDesc();
    AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, ErrorCode::INPUT_PARAM_NULL)

    bool cust_aicpu_flag = false;
    (void) ge::AttrUtils::GetBool(op_desc_ptr, kCustAicpuFlag, cust_aicpu_flag);
    int64_t workspace_size = 0;
    if (cust_aicpu_flag) {
        (void)ge::AttrUtils::GetInt(op_desc_ptr, kWorkspaceSize, workspace_size);
        std::vector<uint32_t> mem_type;
        mem_type.push_back(static_cast<uint32_t>(ge::AicpuWorkSpaceType::CUST_LOG));
        // 这里是第一次使用workspace，后续如果要增加，需要往mem_type 追加
        (void)ge::AttrUtils::SetListInt(op_desc_ptr, ge::ATTR_NAME_AICPU_WORKSPACE_TYPE, mem_type);
    }
    // 一个算子可以存在多个workspace
    op_desc_ptr->SetWorkspaceBytes({workspace_size});
    AICPU_CHECK_RES_WITH_LOG(SetOutPutsSize(op_desc_ptr),
        "Call SetOutPutsSize function failed, op[%s].", node_name.c_str());

    // 可以针对不同的workspace设置不同的复用类型
    AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::SetListBool(op_desc_ptr, kWorkspaceReuseFlag, {true}),
        AICPU_REPORT_INNER_ERR_MSG("Set workspace memory reuse flag failed, op[%s].",
            node_name.c_str());
        return ErrorCode::ADD_ATTR_FAILED)

    AICPU_CHECK_RES(SetAttrResource(node_name, op_desc_ptr));

    if (op_desc_ptr->HasAttr(kAttrNameThreadScopeId)) {
      AICPU_CHECK_RES_WITH_LOG(CalFftsMaxThread(op_desc_ptr), "cal ffts plus task max thread size failed, op[%s].",
                               node.GetName().c_str())
    }
    AICPUE_LOGI("CPUKernel's op %s[%s] run CalcOpRunningParam success. workspace total size is %ld", node_name.c_str(),
                node_type.c_str(), workspace_size);
    return ge::SUCCESS;
}

ge::Status CpuKernelBuilder::BuildMemCopyInfo(
    const ge::OpDescPtr &op_desc_ptr, [[maybe_unused]] const ge::RunContext &run_context,
    domi::KernelDef *&kernel_def) const {
  kernel_def->set_kernel_ext_info_size(0);

  uint32_t param_len = static_cast<uint32_t>(sizeof(AicpuParamHead));
  // get input and output total number, no need to check overflow
  uint32_t io_addrs_num = static_cast<uint32_t>(op_desc_ptr->GetInputsSize() +
                                                op_desc_ptr->GetOutputsSize());
  // get input and output addrs size, no need to check overflow
  uint32_t io_addrs_size =
      io_addrs_num * static_cast<uint32_t>(sizeof(uint64_t));
  // refresh param_len, no need to check overflow
  param_len += io_addrs_size;

  ge::Buffer bytes;
  bool has_customized_attr =
      ge::AttrUtils::GetZeroCopyBytes(op_desc_ptr, kCustomizedOpDef, bytes);
  // When it's aicpu customized ops, get customized attr
  if (has_customized_attr) {
    CHECK_UINT32_ADD_OVERFLOW(
        param_len, sizeof(uint32_t), ErrorCode::DATA_OVERFLOW,
        "Overflow when param total bytes[%u] add 4bytes, op[%s]", param_len,
        op_desc_ptr->GetName().c_str())
    param_len += sizeof(uint32_t);
    // Customized attr length must be less than UINT32_MAX, no need to check
    // overflow
    uint32_t customized_attr_len = static_cast<uint32_t>(bytes.GetSize());
    CHECK_UINT32_ADD_OVERFLOW(
        param_len, customized_attr_len, ErrorCode::DATA_OVERFLOW,
        "Overflow when calculate total bytes of task param[%u] and custom "
        "attr[%u], op[%s]",
        param_len, customized_attr_len, op_desc_ptr->GetName().c_str())
    param_len += customized_attr_len;
  }

  uint32_t ext_info_length = 0;
  uint64_t ext_info_addrs = 0;
  std::string task_args;
  AicpuParamHead param_head = {.length = param_len,
                               .ioAddrNum = io_addrs_num,
                               .extInfoLength = ext_info_length,
                               .extInfoAddr = ext_info_addrs};
  task_args.append(reinterpret_cast<const char *>(&param_head),
                   sizeof(AicpuParamHead));
  // TaskArgs append ioAddrs
  task_args.append(io_addrs_size, ' ');

  // When it's aicpu customized ops, task_args should append customized attr
  if (has_customized_attr) {
    const uint8_t *node_def_data = bytes.GetData();
    AICPU_CHECK_NOTNULL(node_def_data)
    size_t customized_attr_len = bytes.GetSize();
    uint32_t node_def_len = static_cast<uint32_t>(customized_attr_len);
    task_args.append(reinterpret_cast<const char *>(&node_def_len),
                     sizeof(uint32_t));
    task_args.append(reinterpret_cast<const char *>(node_def_data),
                     customized_attr_len);
  }

  kernel_def->set_args(task_args.data(), param_len);
  kernel_def->set_args_size(param_len);
  kernel_def->set_so_name(kDefaultKernelSo);
  kernel_def->set_kernel_name(kDefaultFunctionName);

  domi::KernelContext *context = kernel_def->mutable_context();
  AICPU_CHECK_NOTNULL(context)
  context->set_op_index(g_op_index);
  context->set_kernel_type(kAiCpuKernelType);

  AICPUE_LOGI(
      "GenerateMemCopyTask Task_args length is [%zu] param_len is [%u], "
      "kernel_so_name is [%s], func_name is [%s]",
      task_args.length(), param_len, kDefaultKernelSo.c_str(),
      kDefaultFunctionName.c_str());
  return ge::SUCCESS;
}

ge::Status CpuKernelBuilder::GenerateMemCopyTask(
    uint64_t data_info_size, const ge::RunContext &run_context,
    std::vector<domi::TaskDef> &tasks) const {
  static int copy_count = 0;
  std::string node_type("MemCopy");

  // MemCopy has four inputs and zero outputs
  int in_num = 4;
  int out_num = 0;
  ge::Format format = ge::FORMAT_NCHW;
  ge::DataType data_type = ge::DT_UINT64;
  std::vector<int64_t> shape = {};
  shape.push_back(data_info_size);
  std::string node_name(node_type + "_" + std::to_string(copy_count));
  ge::NodePtr node = aicpu::GenGeNode(node_name, node_type, in_num, out_num,
                                      format, data_type, shape);
  AICPU_CHECK_NOTNULL(node);
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  AICPU_CHECK_NOTNULL(op_desc_ptr)
  AICPU_CHECK_FALSE_EXEC(
      ge::AttrUtils::SetInt(op_desc_ptr, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE,
                            ge::DEPEND_IN_SHAPE),
      AICPU_REPORT_INNER_ERR_MSG(
          "Call AttrUtils::SetInt failed to set attr[%s], op[%s].",
          ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE.c_str(),
          op_desc_ptr->GetName().c_str());
      return ErrorCode::ADD_ATTR_FAILED)
  AICPU_CHECK_FALSE_EXEC(
      ge::AttrUtils::SetInt(op_desc_ptr, "num", data_info_size),
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::SetInt failed to set attr[num], op[%s].",
          node_name.c_str());
      return ErrorCode::ADD_ATTR_FAILED)
  AICPUE_LOGI("Op[%s], op type[%s] start GenerateMemCopyTask",
              node->GetName().c_str(), node->GetType().c_str());

  aicpuops::NodeDef node_def;
  AICPU_CHECK_RES_WITH_LOG(
      BuildAicpuNodeDef(op_desc_ptr, node_def),
      "Call BuildMemCopyInfo function BuildAicpuNodeDef failed op[%s].",
      node->GetName().c_str())

  auto state = InsertAicpuNodeDefAttrToOp(op_desc_ptr, node_def, kCustomizedOpDef);
  if (state != ge::SUCCESS) {
    return state;
  }
  domi::TaskDef task_def;
  task_def.set_type(RT_MODEL_TASK_KERNEL);
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  AICPU_CHECK_NOTNULL(kernel_def);

  AICPU_CHECK_RES_WITH_LOG(
      BuildMemCopyInfo(op_desc_ptr, run_context, kernel_def),
      "Call BuildMemCopyInfo function failed op[%s].", node->GetName().c_str())
  AICPUE_LOGI("CPUKernel's op [%s][%s] run GenerateMemCopyTask success.",
              node->GetName().c_str(), node->GetType().c_str());
  tasks.emplace_back(task_def);
  return ge::SUCCESS;
}

ge::Status GetAicpuFftsPlusKerneLType(const ge::OpDescPtr op_desc_ptr,
                                      uint32_t &kernel_type) {
  const std::string *kernel_lib_name = ge::AttrUtils::GetStr(op_desc_ptr, "opKernelLib");
  AICPU_CHECK_NOTNULL(kernel_lib_name);
  if (*kernel_lib_name == kCustAicpuKernelInfoChoice) {
    kernel_type = kFftsCustAiCpuKernelType;
  } else if (*kernel_lib_name == kAicpuKernelInfoChoice) {
    kernel_type = kFftsAiCpuKernelType;
  } else {
    AICPU_REPORT_INNER_ERR_MSG(
        "Get kernel type failed, node[%s], kernel type[%s].",
        op_desc_ptr->GetName().c_str(), kernel_lib_name->c_str());
    return ErrorCode::KERNEL_TYPE_INVALID;
  }
  // aicpu node which in libcpu_kernels.v0.1.so need run with
  // custaicpu_sd，depends on cust_aicpu_flag
  bool cust_aicpu_flag = false;
  (void)ge::AttrUtils::GetBool(op_desc_ptr, kCustAicpuFlag, cust_aicpu_flag);
  if (cust_aicpu_flag) {
    kernel_type = kFftsCustAiCpuKernelType;
  }
  return ge::SUCCESS;
}

ge::Status GetAicpuKernelType(const ge::OpDescPtr op_desc_ptr,
                              uint32_t &kernel_type) {
  const std::string *kernel_lib_name = ge::AttrUtils::GetStr(op_desc_ptr, "opKernelLib");
  AICPU_CHECK_NOTNULL(kernel_lib_name);
  if (*kernel_lib_name == kCustAicpuKernelInfoChoice) {
    kernel_type = kCustAiCpuKernelType;
  } else if (*kernel_lib_name == kHostCpuKernelInfoChoice) {
    kernel_type = kHostAiCpuKernelType;
  } else if (*kernel_lib_name == kAicpuKernelInfoChoice) {
    kernel_type = kAiCpuKernelType;
  } else {
    AICPU_REPORT_INNER_ERR_MSG(
        "Get kernel type failed, node[%s], kernel type[%s].",
        op_desc_ptr->GetName().c_str(), kernel_lib_name->c_str());
    return ErrorCode::KERNEL_TYPE_INVALID;
  }
  // aicpu node which in libcpu_kernels.v0.1.so need run with
  // custaicpu_sd，depends on cust_aicpu_flag
  bool cust_aicpu_flag = false;
  (void)ge::AttrUtils::GetBool(op_desc_ptr, kCustAicpuFlag, cust_aicpu_flag);
  if (cust_aicpu_flag) {
    kernel_type = kCustAiCpuKernelType;
  }
  return ge::SUCCESS;
}

ge::Status CpuKernelBuilder::GenerateTask(const ge::Node &node,
                                          const ge::RunContext &run_context,
                                          std::vector<domi::TaskDef> &tasks) {
  AICPUE_LOGI("CPUKernel's op %s[%s] run GenerateTask.", node.GetName().c_str(),
              node.GetType().c_str());
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, INPUT_PARAM_NULL)
  if (op_desc_ptr->HasAttr(kAttrNameThreadScopeId)) {
    AICPUE_LOGI("op %s[%s] run GenerateFftsPlusTask.", node.GetName().c_str(),
                node.GetType().c_str());
    AICPU_CHECK_RES_WITH_LOG(
        GenerateFftsPlusTask(node),
        "Call GenerateFftsPlusTask function failed, op[%s].",
        node.GetName().c_str())
    return ge::SUCCESS;
  }
  g_op_index = op_desc_ptr->GetId();
  domi::TaskDef task_def;
  task_def.set_type(RT_MODEL_TASK_KERNEL);
  task_def.set_sqe_num(kBasicAicpuOpSqeNumber);
  bool is_blocking_aicpu_op = false;
  (void)ge::AttrUtils::GetBool(op_desc_ptr, ge::ATTR_NAME_IS_BLOCKING_OP, is_blocking_aicpu_op);
  if (is_blocking_aicpu_op) {
    task_def.set_sqe_num(kAsyncAicpuOpSqeNumber);
  }

  // no need to set streamID for task_def, ge will reallocate stream
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  AICPU_CHECK_NOTNULL(kernel_def)
  AICPU_CHECK_RES_WITH_LOG(
      BuildAndLaunchKernel(node, kernel_def),
      "Call BuildAndLaunchKernel function failed, op[%s].",
      node.GetName().c_str())

  tasks.emplace_back(task_def);

  if (ge::AttrUtils::HasAttr(op_desc_ptr, kAttrNameUnknownShape)) {
    int32_t shape_type = 0;
    // unknow shape
    AICPU_CHECK_FALSE_EXEC(
        ge::AttrUtils::GetInt(op_desc_ptr, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE,
                              shape_type),
        AICPU_REPORT_INNER_ERR_MSG(
            "Call AttrUtils::GetInt failed to get attr[%s], op[%s].",
            ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE.c_str(),
            op_desc_ptr->GetName().c_str());
        return ErrorCode::ADD_ATTR_FAILED)
    if (shape_type == ge::DEPEND_COMPUTE) {
      uint64_t data_info_size =
          static_cast<uint64_t>(op_desc_ptr->GetOutputsSize()) *
          kDataAndShapeNum;
      AICPU_CHECK_RES_WITH_LOG(
          GenerateMemCopyTask(data_info_size, run_context, tasks),
          "GenerateMemCopyTask op [%s][%s] failed.", node.GetName().c_str(),
          node.GetType().c_str());
      AICPUE_LOGI("GenerateMemCopyTask op [%s][%s] task.size [%zu].",
                  node.GetName().c_str(), node.GetType().c_str(), tasks.size());
    }
  }
  return ge::SUCCESS;
}

ge::Status CpuKernelBuilder::UpdateTask(const ge::Node &node,
  std::vector<domi::TaskDef> &tasks) {
  std::shared_ptr<ge::OpDesc> op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, ErrorCode::INPUT_PARAM_NULL)

  for (size_t index = 0; index < tasks.size(); index++) {
    tasks[index].set_stream_id(static_cast<uint32_t>(op_desc_ptr->GetStreamId()));
  }

  AICPUE_LOGI("CpuKernel's op[%s], op type[%s] run UpdateTask, stream_id=%ld, task def size:%zu.",
    node.GetName().c_str(), node.GetType().c_str(), op_desc_ptr->GetStreamId(), tasks.size());
  return ge::SUCCESS;
}

// Make and init task extend info
ge::Status CpuKernelBuilder::MakeAicpuKernelExtInfo(
    const ge::OpDescPtr &op_desc_ptr, vector<char> &task_ext_info,
    const FftsPlusInfo &ffts_info) const {
  const std::string *kernel_lib_name = ge::AttrUtils::GetStr(op_desc_ptr, "opKernelLib");
  AICPU_CHECK_RES_WITH_LOG(
      MakeBaseExtInfo(op_desc_ptr, task_ext_info, ffts_info),
      "Call MakeTaskExtInfo funtion failed, op[%s].",
      op_desc_ptr->GetName().c_str())
  AICPU_CHECK_RES_WITH_LOG(MakeNoTilingExtInfo(op_desc_ptr, task_ext_info),
                           "Call MakeNoTilingExtInfo funtion failed, op[%s].",
                           op_desc_ptr->GetName().c_str())
  uint64_t extend_info_len = task_ext_info.size();
  extend_info_len += aicpu::FWKAdapter::kExtInfoHeadSize;
  extend_info_len += sizeof(SessionInfo);

  uint64_t ext_info_offset = task_ext_info.size();
  task_ext_info.resize(extend_info_len, 0);
  char *ext_info_buf = task_ext_info.data();
  auto ext_info = reinterpret_cast<aicpu::FWKAdapter::ExtInfo *>(ext_info_buf + ext_info_offset);
  ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_SESSION_INFO;
  ext_info->infoLen = (sizeof(SessionInfo));
  ext_info_offset += aicpu::FWKAdapter::kExtInfoHeadSize;
  SessionInfo *session_info = reinterpret_cast<SessionInfo *>(ext_info_buf + ext_info_offset);
  session_info->sessionId = GenerateUniqueSessionId();
  session_info->kernelId = GenerateUniqueKernelId();
  session_info->sessFlag = false;

  // 针对cust op设定workspace size
  if ((kernel_lib_name != nullptr) && (*kernel_lib_name == kCustAicpuKernelInfoChoice)) {
    extend_info_len = task_ext_info.size();
    extend_info_len += aicpu::FWKAdapter::kExtInfoHeadSize;
    extend_info_len += sizeof(aicpu::FWKAdapter::WorkSpaceInfo);

    ext_info_offset = task_ext_info.size();
    task_ext_info.resize(extend_info_len, 0);
    ext_info_buf = task_ext_info.data();
    ext_info = reinterpret_cast<aicpu::FWKAdapter::ExtInfo *>(ext_info_buf + ext_info_offset);
    ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_WORKSPACE_INFO;
    ext_info->infoLen = (sizeof(aicpu::FWKAdapter::WorkSpaceInfo));
    ext_info_offset += aicpu::FWKAdapter::kExtInfoHeadSize;
    aicpu::FWKAdapter::WorkSpaceInfo *workspace_info =
        reinterpret_cast<aicpu::FWKAdapter::WorkSpaceInfo *>(ext_info_buf + ext_info_offset);
    workspace_info->size = 0UL;
    workspace_info->addr = 0UL;
  }

  return ge::SUCCESS;
}

ge::Status BuildArgs(const ge::Node &node, std::string &task_args,
                     uint32_t &args_len, const uint32_t index,
                     FftsPlusInfo &ffts_info) {
  const ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  // param_length: AicpuParamHead.len + io_addrs_size + customizedAttr.len
  uint32_t param_length = static_cast<uint32_t>(sizeof(AicpuParamHead));
  size_t input_size = op_desc_ptr->GetInputsSize();
  size_t output_size = op_desc_ptr->GetOutputsSize();
  // get input and output total number, no need to check overflow
  uint32_t io_addrs_num = static_cast<uint32_t>(input_size +
                                                output_size);
  // get input and output addrs size, no need to check overflow
  uint32_t io_addrs_size =
      io_addrs_num * static_cast<uint32_t>(sizeof(uint64_t));
  // refresh param_length, no need to check overflow
  param_length += io_addrs_size;
  ge::Buffer bytes;
  std::string k_name = kCustomizedOpDef;

  if ((index != 0) && (index + 1 == ffts_info.slice_instance_num)) {
    k_name = kCustomizedTailOpDef;
  }
  bool has_customized_attr =
      ge::AttrUtils::GetZeroCopyBytes(op_desc_ptr, k_name, bytes);
  // When it's aicpu customized ops, get customized attr
  if (has_customized_attr) {
    CHECK_UINT32_ADD_OVERFLOW(
        param_length, sizeof(uint32_t), ErrorCode::DATA_OVERFLOW,
        "Overflow when param total bytes[%u] add 4bytes, op[%s]", param_length,
        op_desc_ptr->GetName().c_str())
    param_length += sizeof(uint32_t);
    // Customized attr length must be less than UINT32_MAX, no need to check
    // overflow
    uint32_t customized_attr_len = static_cast<uint32_t>(bytes.GetSize());
    CHECK_UINT32_ADD_OVERFLOW(
        param_length, customized_attr_len, ErrorCode::DATA_OVERFLOW,
        "Overflow when calculate total bytes of task param[%u] and custom "
        "attr[%u], op[%s]",
        param_length, customized_attr_len, op_desc_ptr->GetName().c_str())
    param_length += customized_attr_len;
  }

  uint64_t ext_info_addrs = 0ul;
  // Create task_args: AicpuParamHead + ioAddrs + customizedAttr
  AicpuParamHead param_head = {param_length, io_addrs_num, ffts_info.ext_len,
                               ext_info_addrs};
  task_args.append(reinterpret_cast<const char *>(&param_head),
                   sizeof(AicpuParamHead));
  bool auto_thread_flag = ffts_info.auto_static_split;
  for (uint32_t i = 0; i < input_size; i++) {
    const uint64_t offset =
        auto_thread_flag ? ffts_info.input_addr_offset[i] * index : 0UL;
    task_args.append(reinterpret_cast<const char *>(&offset), sizeof(uint64_t));
  }
  for (uint32_t i = 0; i < output_size; i++) {
    const uint64_t offset =
        auto_thread_flag ? ffts_info.output_addr_offset[i] * index : 0UL;
    task_args.append(reinterpret_cast<const char *>(&offset), sizeof(uint64_t));
  }

  // device not use flag_async,remove it
  // When it's aicpu customized ops, task_args should append customized attr
  if (has_customized_attr) {
    const uint8_t *node_def = bytes.GetData();
    AICPU_CHECK_NOTNULL(node_def)
    size_t customized_attr_len = bytes.GetSize();
    uint32_t node_def_len = static_cast<uint32_t>(customized_attr_len);
    task_args.append(reinterpret_cast<const char *>(&node_def_len),
                     sizeof(uint32_t));
    task_args.append(reinterpret_cast<const char *>(node_def),
                     customized_attr_len);
  }
  args_len = param_length;
  return ge::SUCCESS;
}

ge::Status CpuKernelBuilder::BuildAndLaunchKernel(
    const ge::Node &node, domi::KernelDef *&kernel_def) const {
  std::vector<char> task_ext_info;
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  FftsPlusInfo ffts_info;
  ffts_info.valid = false;
  AICPU_CHECK_RES_WITH_LOG(
      MakeAicpuKernelExtInfo(op_desc_ptr, task_ext_info, ffts_info),
      "Call MakeAicpuKernelExtInfo function failed, op[%s]",
      op_desc_ptr->GetName().c_str())

  kernel_def->set_kernel_ext_info_size(task_ext_info.size());
  if (task_ext_info.size() != 0) {
    kernel_def->set_kernel_ext_info(
        reinterpret_cast<void *>(task_ext_info.data()), task_ext_info.size());
  }

  // set kernel so name's len and offset addr
  std::string kernel_so_name;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, "kernelSo", kernel_so_name);
  if (kernel_so_name.empty()) {
    kernel_so_name = kDefaultKernelSo;
  }
  // set kernel function name's len and offset addr
  std::string func_name;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, "funcName", func_name);
  if (func_name.empty()) {
    func_name = kDefaultFunctionName;
  }
  std::string task_args;
  uint32_t args_len = 0u;
  AICPU_CHECK_RES_WITH_LOG(
      BuildArgs(node, task_args, args_len, 0, ffts_info),
      "Call BuildArgs function failed, op[%s]",
      op_desc_ptr->GetName().c_str())
  // set kernel blockdim
  uint32_t block_dim = GetOpBlockDim(op_desc_ptr);
  kernel_def->set_block_dim(block_dim);
  kernel_def->set_args(task_args.data(), args_len);
  kernel_def->set_args_size(args_len);
  kernel_def->set_so_name(kernel_so_name);
  kernel_def->set_kernel_name(func_name);

  // get kernel type
  uint32_t kernel_type;
  if (GetAicpuKernelType(op_desc_ptr, kernel_type) != ge::SUCCESS) {
    return ErrorCode::KERNEL_TYPE_INVALID;
  }
  domi::KernelContext *context = kernel_def->mutable_context();
  AICPU_CHECK_NOTNULL(context)
  context->set_op_index(op_desc_ptr->GetId());
  context->set_kernel_type(kernel_type);

  AICPUE_LOGI(
      "Task_args length is %zu, args_len is %u, kernel_so_name is %s, "
      "func_name is %s, kernel_type is %d",
      task_args.length(), args_len, kernel_so_name.c_str(), func_name.c_str(),
      kernel_type);
  return ge::SUCCESS;
}

ge::Status CpuKernelBuilder::BuildFftsInfo(const ge::OpDescPtr &op_desc_ptr,
                                           FftsPlusInfo &ffts_info) const {
  ffts::ThreadSliceMapPtr slice_info = nullptr;
  slice_info = op_desc_ptr->TryGetExtAttr(kAttrNameSgtStruct, slice_info);
  if (slice_info == nullptr) {
    AICPU_REPORT_INNER_ERR_MSG("The Node[%s] has no _sgt_struct_info",
                            op_desc_ptr->GetName().c_str());
    return INVOKE_GRAPH_ITF_FAILED;
  }
  ffts_info.valid = true;
  ffts_info.thread_mode = slice_info->thread_mode;
  bool is_unknown_shape = false;
  if (ge::AttrUtils::HasAttr(op_desc_ptr, kAttrNameUnknownShape)) {
    is_unknown_shape = true;
  }

  ffts_info.is_unknown_shape = is_unknown_shape;
  ffts_info.auto_static_split = (!is_unknown_shape) && (ffts_info.thread_mode);
  ffts_info.thread_id = (ffts_info.thread_mode) ? 0u : slice_info->thread_id;
  ffts_info.slice_instance_num =
      (ffts_info.auto_static_split) ? slice_info->slice_instance_num : kAicpuManualSliceNum;
  if (ffts_info.auto_static_split) {
    auto thread_input_range = slice_info->input_tensor_slice;
    GetThreadTensorShape(thread_input_range, ffts_info.thread_input_shape);
    auto thread_output_range = slice_info->output_tensor_slice;
    GetThreadTensorShape(thread_output_range, ffts_info.thread_output_shape);
  } else {
    std::vector<std::vector<int64_t>> inputs_shape;
    std::vector<std::vector<int64_t>> outputs_shape;
    GetInOutPutsShape(op_desc_ptr, inputs_shape, outputs_shape);
    ffts_info.thread_input_shape.push_back(inputs_shape);
    ffts_info.thread_output_shape.push_back(outputs_shape);
  }
  AICPUE_LOGD("ffts_info [%s][%s]slicenum[%u], thread_mode[%u]",
              op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(),
              ffts_info.slice_instance_num, ffts_info.thread_mode);
  return ge::SUCCESS;
}

ge::Status CpuKernelBuilder::BuildAiCpuCtx(const ge::OpDescPtr &op_desc_ptr,
                                           const FftsPlusInfo &ffts_info,
                                           domi::FftsPlusAicpuCtxDef *ctx) const {
  ctx->set_atm(static_cast<uint32_t>(ffts_info.thread_mode));
  ctx->set_topic_type(static_cast<uint32_t>(GetOpNodeTopicType(op_desc_ptr)));
  ctx->set_bm(kDefaultNum);
  ctx->set_group_id(kDefaultNum);
  ctx->set_topic_id(kEventHwTsKernelMsg);
  ctx->set_sub_topic_id(kEventFftsPlusMsg);
  if (ffts_info.is_unknown_shape) {
    ctx->set_thread_dim(0);
  } else {
    ctx->set_thread_dim(ffts_info.slice_instance_num);
  }
  ctx->set_thread_id(ffts_info.thread_id);
  uint32_t block_dim = GetOpBlockDimForFftsPlus(op_desc_ptr, ffts_info, 0);
  ctx->set_non_tail_block_dim(block_dim);
  uint32_t block_tail_dim = block_dim; // manual mode hardware used block_tail_dim
  ctx->set_tail_block_dim(block_tail_dim);
  if (ffts_info.auto_static_split && ffts_info.slice_instance_num > 1) {
    uint32_t tail_index = ffts_info.slice_instance_num - 1;
    block_tail_dim = GetOpBlockDimForFftsPlus(op_desc_ptr, ffts_info, tail_index);
    ctx->set_tail_block_dim(block_tail_dim);
  }
  AICPUE_LOGD("mode[%d] [%s][%s] block_dim is[%u], block_tail_dim is[%u].",
              ffts_info.auto_static_split,
              op_desc_ptr->GetName().c_str(),
              op_desc_ptr->GetType().c_str(),
              block_dim,
              block_tail_dim);

  uint32_t kernel_type;
  if (GetAicpuFftsPlusKerneLType(op_desc_ptr, kernel_type) != ge::SUCCESS) {
    return ErrorCode::KERNEL_TYPE_INVALID;
  }
  ctx->set_kernel_type(kernel_type);
  return ge::SUCCESS;
}

ge::Status CpuKernelBuilder::BuildAiCpuFftsKernelDef(const ge::Node &node, FftsPlusInfo &ffts_info,
    domi::FftsPlusAicpuCtxDef *aicpu_ctx) const {
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, INPUT_PARAM_NULL)
  // set kernel so name's len and offset addr
  std::string kernel_so_name;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, "kernelSo", kernel_so_name);
  if (kernel_so_name.empty()) {
    kernel_so_name = kDefaultKernelSo;
  }
  domi::aicpuKernelDef *kernel_def = aicpu_ctx->mutable_kernel();
  kernel_def->set_so_name(kernel_so_name);
  std::string func_name;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, "funcName", func_name);
  if (func_name.empty()) {
    func_name = kDefaultFunctionName;
  }
  kernel_def->set_kernel_name(func_name);

  std::string args;
  std::string info;
  uint32_t args_size = 0u;
  uint32_t info_size = 0u;
  uint32_t args_len = 0u;
  bool first = true;
  ffts_info.slice_instance_index = 0;
  GetFftsPlusInOutAddrOffset(op_desc_ptr, ffts_info);
  for (uint32_t i = 0u; i < ffts_info.slice_instance_num; i++) {
    // modify task_args中的ioaddrs，custom，ext_info_addrs
    std::vector<char> ext_info;
    std::string tmp_args;
    ffts_info.slice_instance_index = i;
    AICPU_CHECK_RES_WITH_LOG(MakeAicpuKernelExtInfo(op_desc_ptr, ext_info, ffts_info),
        "Call MakeAicpuKernelExtInfo function failed, op[%s]",
        op_desc_ptr->GetName().c_str());
    ffts_info.ext_len = ext_info.size();
    (void)BuildArgs(node, tmp_args, args_len, i, ffts_info);
    args.append(tmp_args.data(), args_len);
    if (first) {
      aicpu_ctx->set_task_param_offset(args_len);
      first = false;
    }
    info.append(ext_info.data(), ext_info.size());
    info_size += ext_info.size();
    args_size += args_len;
  }
  kernel_def->set_args(args.data(), args_size);
  kernel_def->set_args_size(args_size);
  kernel_def->set_kernel_ext_info(info.data(), info_size);
  kernel_def->set_kernel_ext_info_size(info_size);
  AICPUE_LOGI("op [%s][%s]arg size[%u], extinfo size[%u],kernel[%s],fun[%s]",
              op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(),
              args_size, info_size, kernel_so_name.c_str(), func_name.c_str());
  return ge::SUCCESS;
}

ge::Status CpuKernelBuilder::GenerateFftsPlusTask(const ge::Node &node) const {
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, INPUT_PARAM_NULL)

  if (ge::AttrUtils::HasAttr(op_desc_ptr, kAttrNameUnknownShape)) {
    (void)ge::AttrUtils::SetStr(op_desc_ptr, "_ge_attr_lowering_func", kEngineNameAicpuFfts);
  }

  FftsPlusInfo ffts_info;
  ge::Status state = BuildFftsInfo(op_desc_ptr, ffts_info);
  if (state != ge::SUCCESS) {
    return state;
  }
  // build ctxdef
  FftsPlusCtxDefPtr ffts_plus_ctx_def = nullptr;
  try {
      ffts_plus_ctx_def = std::make_shared<domi::FftsPlusCtxDef>();
  } catch (...) {
    AICPU_REPORT_INNER_ERR_MSG("op[%s] Create FftsPlusCtxDef fail",
                             node.GetName().c_str());
    return MEMORY_ALLOC_FAILED;
  }

  ffts_plus_ctx_def->set_context_type(kCtxTypeAicpu);
  domi::FftsPlusAicpuCtxDef *aicpu_ctx = ffts_plus_ctx_def->mutable_aicpu_ctx();
  AICPU_CHECK_NOTNULL(aicpu_ctx);
  state = BuildAiCpuCtx(op_desc_ptr, ffts_info, aicpu_ctx);
  if (state != ge::SUCCESS) {
    AICPU_REPORT_INNER_ERR_MSG("op[%s] call BuildAiCpuCtx fail",
                             node.GetName().c_str());
    return state;
  }
  state = BuildAiCpuFftsKernelDef(node, ffts_info, aicpu_ctx);
  (void)op_desc_ptr->SetExtAttr(kAttrNameFftsPlusCtxDef, ffts_plus_ctx_def);
  AICPUE_LOGI("op[%s][%s] run GenerateFftsPlusTask state[%u]",
              node.GetType().c_str(), node.GetName().c_str(), state);

  return state;
}

int64_t CpuKernelBuilder::CeilDivisor(const int64_t x, const int64_t base) const
{
  if (base == 0) {
    return kDefaultAicpuBlockDim;
  }
  int64_t result = x / base;
  if (x % base != 0) {
    result++;
  }
  return result;
}

uint32_t CpuKernelBuilder::GetOpBlockDim(const ge::OpDescPtr &op_desc_ptr) const
{
  if (!IsSupportBlockDim(op_desc_ptr)) {
    AICPUE_LOGD("op[%s] is not support block dim.", op_desc_ptr->GetName().c_str());
    return kDefaultAicpuBlockDim;
  }

  int32_t block_dim_index = -1;
  ge::AttrUtils::GetInt(op_desc_ptr, kBlockDimByIndex, block_dim_index);

  // only user first input to cala blockdim
  auto input0_desc = op_desc_ptr->GetInputDesc(0);
  auto input_shape = input0_desc.GetShape();
  int64_t total;
  if (block_dim_index == -1) {
    total = input_shape.GetShapeSize();
  } else {
    total = input_shape.GetDim(block_dim_index);
  }

  return CalcBlockDimByShapeSize(total);
}

uint32_t CpuKernelBuilder::GetOpBlockDimForFftsPlus(const ge::OpDescPtr &op_desc_ptr,
                                                    const FftsPlusInfo &ffts_info,
                                                    const uint32_t thread_index) const
{
  if (!IsSupportBlockDim(op_desc_ptr)) {
    AICPUE_LOGD("op[%s] is not support block dim.", op_desc_ptr->GetName().c_str());
    return kDefaultAicpuBlockDim;
  }
  AICPUE_LOGD("op[%s], thread_index[%u].", op_desc_ptr->GetName().c_str(), thread_index);

  if (thread_index >= ffts_info.thread_input_shape.size()) {
    AICPUE_LOGW("op[%s] get dim error. thread_id[%d]",
                op_desc_ptr->GetName().c_str(),
                thread_index);
    return kDefaultAicpuBlockDim;
  }
  vector<int64_t> input_0 = ffts_info.thread_input_shape[thread_index][0];
  int32_t block_dim_index = -1;
  ge::AttrUtils::GetInt(op_desc_ptr, kBlockDimByIndex, block_dim_index);
  int64_t total = 0;
  if (block_dim_index == -1) {
    total = std::accumulate(input_0.begin(), input_0.end(), 1,
                            [](int64_t i, int64_t j) -> int64_t {return i * j;});
  } else {
    int32_t vec_size = static_cast<int32_t>(input_0.size());
    if (block_dim_index < vec_size) {
      total = input_0[block_dim_index];
    } else {
      AICPUE_LOGW("op[%s] get total error.", op_desc_ptr->GetName().c_str());
      return kDefaultAicpuBlockDim;
    }
  }
  AICPUE_LOGD("op[%s], total[%ld]", op_desc_ptr->GetName().c_str(), total);
  return CalcBlockDimByShapeSize(total);
}

bool CpuKernelBuilder::IsSupportBlockDim(const ge::OpDescPtr &op_desc_ptr) const
{
  if (IsUnknowShape(op_desc_ptr)) {
    AICPUE_LOGD("op[%s] is unknow shape.", op_desc_ptr->GetName().c_str());
    return false;
  }
  bool support_block_dim = false;
  if (ge::AttrUtils::HasAttr(op_desc_ptr, kSupportBlockDim)) {
    if (!ge::AttrUtils::GetBool(op_desc_ptr, kSupportBlockDim, support_block_dim)) {
      AICPUE_LOGW("Call ge::AttrUtils::GetBool failed to get attr[%s], op[%s].", kSupportBlockDim.c_str(),
                  op_desc_ptr->GetName().c_str());
      return false;
    }
  }
  if (!support_block_dim) {
    return false;
  }

  if (ge::AttrUtils::HasAttr(op_desc_ptr, kBlockDimByIndex)) {
    int32_t block_dim_index = -1;
    if (!ge::AttrUtils::GetInt(op_desc_ptr, kBlockDimByIndex, block_dim_index)) {
      AICPUE_LOGW("Call ge::AttrUtils::GetInt failed to get attr[%s], op[%s].", kBlockDimByIndex.c_str(),
                  op_desc_ptr->GetName().c_str());
      return false;
    }
  }
  return true;
}

uint32_t CpuKernelBuilder::CalcBlockDimByShapeSize(const int64_t total) const
{
  // get aicpucount and calc blockDim
  uint32_t ai_cpu_cnt;
  auto ret = rtGetAiCpuCount(&ai_cpu_cnt);
  AICPUE_LOGD("Call rtGetAiCpuCount get ai_cpu_cnt[%u].", ai_cpu_cnt);
  if (ret != 0) {
    AICPUE_LOGW("Call rtGetAiCpuCount get ai_cpu_cnt failed.");
    return kDefaultAicpuBlockDim;
  }

  const int64_t max_shard_num = static_cast<int64_t>(ai_cpu_cnt) * 2;
  int64_t per_unit_size = total / std::min(std::max(1L, static_cast<int64_t>(ai_cpu_cnt)), total);
  int64_t block_size = std::max(int64_t{1}, std::min(total, per_unit_size));
  int64_t shard_num = CeilDivisor(total, block_size);
  shard_num = std::min(max_shard_num, shard_num);
  block_size = CeilDivisor(total, shard_num);
  uint32_t block_dim = CeilDivisor(total, block_size);
  AICPUE_LOGD("GetOpBlockDim total[%ld], ai_cpu_cnt[%u], blockSize[%ld], blockDim[%u]",
              total, ai_cpu_cnt, block_size, block_dim);
  return block_dim;
}
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_task_code_generator.h"
#include <cinttypes>
#include "common/om2/codegen/code_generator_factory.h"
#include "common/om2/codegen/om2_model_utils.h"
#include "common/checker.h"
#include "common/ge_common/debug/ge_log.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "common/math/math_util.h"
#include "graph/args_format_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"


namespace ge {
namespace {
constexpr uint32_t k2BitsMask = 0x00000003U;
const std::string kAllShapeInAicpu = "_AllShape";
constexpr int64_t kDefaultDimInfo = 0x100000001;
constexpr uint64_t kDefaultShapeNum = 0x100000000U;
const std::string kWspUnfoldedMode = "unfolded";
const std::string kWspFoldedMode = "folded";
const std::string kAttrNameAtomicWspMode = "wspMode";
constexpr char_t const *kMaxTilingSize = "op_para_size";
constexpr char_t const *kMaxAtomicCleanTilingSize = "atomic_op_para_size";
constexpr uint32_t kUBAlignedLen = 32U;

std::string GetConfigString(const Om2LaunchKernelConfig &cfg) {
  std::ostringstream oss;
  oss << "{" << static_cast<uint32_t>(cfg.schedule_mode) << "U, " << cfg.engine_type << ", " << cfg.block_dim_offset
      << "U, " << (cfg.is_block_task_prefetch ? "true" : "false") << ", " << (cfg.is_data_dump ? "true" : "false")
      << ", " << cfg.time_out << "U"
      << "}";
  return oss.str();
}

void AppendShapeDesc(const ge::GeTensorDesc &tensor_desc, std::vector<int64_t> &shape_infos) {
  const auto &shape = tensor_desc.GetShape();
  if (shape.IsScalar()) {
    shape_infos.push_back(kDefaultDimInfo);
    shape_infos.push_back(0x1);  // shape value [1]
  } else {
    uint64_t dim_info{kDefaultShapeNum};
    dim_info |= (static_cast<uint64_t>(shape.GetDimNum()));
    shape_infos.push_back(static_cast<int64_t>(dim_info));
    for (const int64_t dim : shape.GetDims()) {
      shape_infos.push_back(dim);
    }
  }
}

bool IsWspAddrFolded(const OpDescPtr &op_desc) {
  std::string wsp_mode = kWspUnfoldedMode;
  return ge::AttrUtils::GetStr(op_desc, kAttrNameAtomicWspMode, wsp_mode) && (wsp_mode == kWspFoldedMode);
}

std::string GenShapeData(const std::vector<int64_t>& shape) {
  std::ostringstream oss;
  oss << "{";
  for (size_t i = 0; i < shape.size(); ++i) {
    oss << shape[i];
    if (i + 1 != shape.size()) {
      oss << ", ";
    }
  }
  oss << "}";
  return oss.str();
}
}  // namespace

Status KernelTaskCodeGenerator::CheckTaskSupport(TaskDistributionContext &context) {
  if (Om2CodegenUtils::OpNeedPrint(context.op_desc)) {
    REPORT_INNER_ERR_MSG("E19999", "Unsupport scenario for dfx.");
    GELOGE(FAILED, "Unsupport scenario for dfx.");
    return FAILED;
  }

  auto task_type = static_cast<ModelTaskType>(context.task_def.type());
  uint32_t kernel_type = 0U;
  if (Om2CodegenUtils::IsAllKernel(task_type)) {
    kernel_type = context.task_def.kernel_with_handle().context().kernel_type();
    if (Om2CodegenUtils::IsSoftSyncOp(context.op_desc)) {
      REPORT_INNER_ERR_MSG("E19999", "Unsupport scenario for dfx.");
      GELOGE(FAILED, "Unsupport scenario for static_to_dynamic_softsync_op.");
      return FAILED;
    }
  } else {
    const domi::KernelDef &kernel_def = context.task_def.kernel();
    kernel_type = kernel_def.context().kernel_type();
    if (Om2CodegenUtils::IsSeparatelyCleanTask(context.op_desc, kernel_def.kernel_name())) {
      REPORT_INNER_ERR_MSG("E19999", "Unsupport scenario for dfx.");
      GELOGE(FAILED, "Unsupport scenario for atomic clean task.");
      return FAILED;
    }
  }

  if (static_cast<ccKernelType>(kernel_type) == ge::ccKernelType::AI_CPU) {
    if (Om2CodegenUtils::IsBlockingAicpuOp(context.op_desc)) {
      REPORT_INNER_ERR_MSG("E19999", "Unsupport scenario for dfx.");
      GELOGE(FAILED, "Unsupport scenario for blocking_op.");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status KernelTaskCodeGenerator::GenTaskDistributionCode(TaskDistributionContext &context) {
  GELOGD("[OM2] start to generate task distribute code.");
  std::stringstream code_stream;
  const auto op_index = context.op_index;
  const auto &kernel_name = context.task_def.kernel().kernel_name();
  auto task_type = static_cast<ModelTaskType>(context.task_def.type());
  auto kernel_type = Om2CodegenUtils::IsAllKernel(task_type) ? context.task_def.kernel_with_handle().context().kernel_type()
                                                      : context.task_def.kernel().context().kernel_type();
  GE_ASSERT_SUCCESS(CheckTaskSupport(context));
  context.nodes.push_back(RAW_CODE_STMT(context.ast_ctx,
    "  // ============================= " + Om2CodegenUtils::GetOpName(context.op_desc)
    + " ==============================="));
  GE_ASSERT_SUCCESS(GenArgsCode(context));
  std::stringstream args_var_names;
  for (const auto &args_addr_node : args_addr_nodes_) {
    (void) context.nodes.insert(context.nodes.cend(), args_addr_node.nodes.cbegin(), args_addr_node.nodes.cend());
    if (!args_var_names.str().empty()) {
      args_var_names << ", ";
    }
    args_var_names << args_addr_node.var_name;
  }
  // aicore
  if (Om2CodegenUtils::IsAllKernel(task_type) || Om2CodegenUtils::IsAICoreKernel(static_cast<ccKernelType>(kernel_type))) {
    GE_ASSERT_TRUE(context.func_handle_indices.find(kernel_name) != context.func_handle_indices.end());
    const std::string cfg_holder_var_name = "op" + std::to_string(op_index) + "_cfg_holder";
    Om2LaunchKernelParam param;
    AssembleLaunchKernelConfig(context.op_desc, context.task_def, param);
    code_stream << EmitLaunchConfigSetupCode(op_index, param.launch_config);
    // gen kernel launch code
    code_stream << "  OM2_CHK_STATUS((KernelTaskDistribute(FlattenHostArgs(" + args_var_names.str() +
        "), args_table_.GetArgsInfo(" << context.args_table_index
        << "), func_handles_[" << context.func_handle_indices[kernel_name] << "], " << param.block_dim
        << ", stream_list_[" << param.stream_id << "], &" << cfg_holder_var_name << ".cfg)));\n";
  } else if (static_cast<ccKernelType>(kernel_type) == ge::ccKernelType::AI_CPU) {
    // aicpu
    std::string op_type = context.op_desc->GetType();
    std::string aicpu_kernel_sign = op_type + kernel_name;
    GE_ASSERT_TRUE(context.func_handle_indices.find(aicpu_kernel_sign) != context.func_handle_indices.end());

    const std::string ioaddr_var_name = "op" + std::to_string(op_index) + "_iow_addr";
    code_stream << "  std::vector<uint64_t> " << ioaddr_var_name << " = FlattenHostArgs(" + args_var_names.str() + ");\n";
    const std::string args_var_name = "op" + std::to_string(context.op_index) + "_args";
    std::string aicpu_args_code;
    GE_ASSERT_SUCCESS(AssembleAicpuArgsCode(context, ioaddr_var_name, args_var_name, aicpu_args_code));
    code_stream << aicpu_args_code;
    const std::string cfg_holder_var_name = "op" + std::to_string(op_index) + "_cfg_holder";
    Om2LaunchKernelParam param;
    AssembleLaunchKernelConfig(context.op_desc, context.task_def, param);
    code_stream << EmitLaunchConfigSetupCode(op_index, param.launch_config);
    code_stream << "  OM2_CHK_STATUS((AicpuKernelTaskDistribute("+ args_var_name +", args_table_.GetArgsInfo(" << context.args_table_index
                << "), func_handles_[" << context.func_handle_indices[kernel_name] << "], " << param.block_dim
                << ", stream_list_[" << param.stream_id << "], &" << cfg_holder_var_name << ".cfg)));\n";
  } else {
    REPORT_INNER_ERR_MSG("E19999", "Unsupported task type %d", static_cast<int32_t>(task_type));
    GELOGE(FAILED, "[OM2] Unsupported task type %d, task def %s", static_cast<int32_t>(task_type),
             context.task_def.ShortDebugString().c_str());
    return FAILED;
  }

  context.nodes.push_back(RAW_CODE_STMT(context.ast_ctx, code_stream.str()));
  ++context.args_table_index;
  return SUCCESS;
}

Status KernelTaskCodeGenerator::GenDistributionImplCode(TaskDistributionImplContext &context) {
  std::stringstream code_stream;
  code_stream << R"(struct LaunchKernelCfgHolder {
  aclrtLaunchKernelCfg cfg{};
  aclrtLaunchKernelAttr attrs[max_launch_cfg_num];
};

struct LaunchKernelConfig {
  uint8_t schedule_mode{0U};
  aclrtEngineType engine_type{ACL_RT_ENGINE_TYPE_AIC};
  uint32_t block_dim_offset{0U};
  bool is_block_task_prefetch{false};
  bool is_data_dump{false};
  uint16_t time_out{0U};
  uint32_t local_memory_size{0U};
};

void AssembleLaunchConfig(LaunchKernelCfgHolder &holder, const LaunchKernelConfig &launch_config) {
  size_t actual_cfg_num = 0UL;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
  holder.attrs[actual_cfg_num].value.schemMode = launch_config.schedule_mode;
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
  holder.attrs[actual_cfg_num].value.engineType = launch_config.engine_type;
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_BLOCKDIM_OFFSET;
  holder.attrs[actual_cfg_num].value.blockDimOffset = launch_config.block_dim_offset;
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_BLOCK_TASK_PREFETCH;
  holder.attrs[actual_cfg_num].value.isBlockTaskPrefetch =
     static_cast<uint8_t>(launch_config.is_block_task_prefetch);
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_DATA_DUMP;
  holder.attrs[actual_cfg_num].value.isDataDump = static_cast<uint8_t>(launch_config.is_data_dump);
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
  holder.attrs[actual_cfg_num].value.timeout = launch_config.time_out;
  actual_cfg_num++;
  holder.cfg.attrs = &holder.attrs[0];
  holder.cfg.numAttrs = actual_cfg_num;
}

aclError KernelTaskDistribute(const std::vector<uint64_t>& io_addrs, ArgsInfo *args_info, aclrtFuncHandle func_handle,
                              uint32_t block_dim, aclrtStream stream, aclrtLaunchKernelCfg *config) {
  OM2_CHK_NOTNULL(args_info);
  OM2_CHK_STATUS(memcpy_s(args_info->host_addr, args_info->size, io_addrs.data(), io_addrs.size() * sizeof(uint64_t)));
  OM2_CHK_STATUS(
    aclrtLaunchKernelV2(
      func_handle,
      block_dim,
      args_info->dev_addr,
      args_info->size,
      config,
      stream
    )
  );
  return ACL_SUCCESS;
}
)";
  context.nodes.push_back(RAW_CODE_STMT(context.ast_ctx, code_stream.str()));
  return SUCCESS;
}

Status KernelTaskCodeGenerator::GenArgsCode(TaskDistributionContext &context) {
  GELOGD("[OM2] start to generate args code.");
  domi::KernelContext kernel_context;
  uint32_t args_size = 0U;
  auto task_type = static_cast<ModelTaskType>(context.task_def.type());
  uint32_t kernel_type = 0U;
  GE_ASSERT_SUCCESS(GetKernelTaskMeta(context.task_def, kernel_context, args_size, kernel_type));

  // 需要确认以下是否这样处理合适，IsAllKernel是否代表是Aicore
  if (Om2CodegenUtils::IsAllKernel(task_type) || Om2CodegenUtils::IsAICoreKernel(static_cast<ccKernelType>(kernel_type))) {
    GE_ASSERT_TRUE(!kernel_context.args_format().empty());
    ArgsFormatInfo args_format_holder;
    GE_ASSERT_SUCCESS(
        ArgsFormatDesc::Parse(context.op_desc, kernel_context.args_format(), args_format_holder.arg_descs),
        "[OM2] Formatted args [%s] parsed failed.", kernel_context.args_format().c_str());
    const size_t format_args_size = GetArgsSizeByFormat(context.op_desc, args_format_holder);
    GE_ASSERT_SUCCESS(ParseArgsFormat(context, args_format_holder), "[OM2] ParseArgsFormat failed, op:[%s].",
                      context.op_desc->GetNamePtr());
    args_size = std::max(args_size, static_cast<uint32_t>(format_args_size));
    const size_t extra_args_size =
        GetExtraArgsSize(context.op_desc, static_cast<ccKernelType>(kernel_type), args_format_holder);
    GELOGD("[OM2] Op:[%s] args size from_task:[%u], extra_size:[%zu]", context.op_desc->GetNamePtr(), args_size,
           extra_args_size);
    GE_ASSERT_TRUE(!AddOverflow(args_size, static_cast<uint32_t>(extra_args_size), args_size));
    context.args_info.args_sizes.push_back(args_size);
    context.args_info.args_offset.push_back(context.args_info.host_args_len);

    // 生成args代码
    GE_ASSERT_SUCCESS(Om2ModelUtils::GenWorkspaceAddrsCode(context, workspace_addr_nodes_));
    GE_ASSERT_SUCCESS(Om2ModelUtils::GenInputAddrCode(context, input_addr_nodes_));
    GE_ASSERT_SUCCESS(Om2ModelUtils::GenOutputAddrCode(context, output_addr_nodes_, true));
    GE_ASSERT_SUCCESS(GenArgsByArgsFormat(context, args_format_holder));

    context.args_info.host_args_len += args_size;
  } else if (static_cast<ccKernelType>(kernel_type) == ge::ccKernelType::AI_CPU) {
    GE_ASSERT_SUCCESS(Om2ModelUtils::GenInputAddrCode(context, input_addr_nodes_));
    GE_ASSERT_SUCCESS(Om2ModelUtils::GenOutputAddrCode(context, output_addr_nodes_, true));
    args_addr_nodes_.resize(input_addr_nodes_.size() + output_addr_nodes_.size());
    (void)args_addr_nodes_.insert(args_addr_nodes_.cend(), input_addr_nodes_.cbegin(), input_addr_nodes_.cend());
    (void)args_addr_nodes_.insert(args_addr_nodes_.cend(), output_addr_nodes_.cbegin(), output_addr_nodes_.cend());
  } else {
    REPORT_INNER_ERR_MSG("E19999", "Unsupported task type %d", static_cast<int32_t>(task_type));
    GELOGE(FAILED, "[OM2] Unsupported task type %d, task def %s", static_cast<int32_t>(task_type),
             context.task_def.ShortDebugString().c_str());
    return FAILED;
  }

  GELOGD("[OM2] generate args code end.");
  return SUCCESS;
}

Status KernelTaskCodeGenerator::AssembleAicpuArgsCode(TaskDistributionContext &context, const std::string iow_addr_var_name, const std::string args_var_name, std::string &aicpu_args_code) {
  GELOGD("[OM2] start to assemble aicpu args code.");
  std::stringstream aicpu_args_code_stream;
  domi::KernelContext kernel_context;
  uint32_t args_size = 0U;
  auto task_type = static_cast<ModelTaskType>(context.task_def.type());
  uint32_t kernel_type = 0U;
  GE_ASSERT_SUCCESS(GetKernelTaskMeta(context.task_def, kernel_context, args_size, kernel_type));
  if (static_cast<ccKernelType>(kernel_type) == ge::ccKernelType::AI_CPU) {
    auto args = context.task_def.kernel().args();
    auto ext_info = context.task_def.kernel().kernel_ext_info();
    context.args_info.args_sizes.push_back(args_size);
    context.args_info.args_offset.push_back(context.args_info.host_args_len);
    // aicpu场景考虑一下什么时候在io_addr_offset记录model io地址
    context.args_info.host_args_len += args_size;
    auto ext_info_size = static_cast<size_t>(ext_info.size());
    GELOGD("[OM2] args size %u, ext info size %u.", args_size, ext_info_size);

    int32_t session_info_offset = -1;

    std::vector<uint8_t> args_buffer(args.begin(), args.end());
    std::vector<uint8_t> ext_info_buffer(ext_info.begin(), ext_info.end());

    auto ret = InitAicpuTaskExtInfo(ext_info_buffer.data(), ext_info_size, context.op_desc,
                                    context.task_def, session_info_offset);
    if (ret != SUCCESS) {
      GELOGW("[OM2] InitAicpuTaskExtInfo failed.");
      return ret;
    }

    const std::string args_str = SerializeBytesToOctalString(args_buffer);
    const std::string ext_info_str = SerializeBytesToOctalString(ext_info_buffer);
    const std::string &args_str_var_name = "op" + std::to_string(context.op_index) + "_args_str";
    const std::string &ext_info_str_var_name = "op" + std::to_string(context.op_index) + "_ext_info_str";
    EMIT_CODE(aicpu_args_code_stream, "  const char* " + args_str_var_name + " = \"" + args_str + "\";");
    EMIT_CODE(aicpu_args_code_stream, "  const char* " + ext_info_str_var_name + " = \"" + ext_info_str + "\";");
    EMIT_CODE(aicpu_args_code_stream, "  std::vector<uint8_t> " + args_var_name + "(" + std::to_string(args_size) + ");");
    EMIT_CODE(aicpu_args_code_stream, "  AssembleAicpuExtInfo(reinterpret_cast<uint8_t*>(const_cast<char*>(" + ext_info_str_var_name + ")), " + std::to_string(ext_info_size) + ", " +
                               std::to_string(session_info_offset) +
                               ", session_id_, &kernel_id_, dev_ext_info_mem_ptrs_, " +
                               std::to_string(context.aicpu_task_index) + ");");
    EMIT_CODE(aicpu_args_code_stream, "  AssembleAicpuArgs(reinterpret_cast<uint8_t*>(const_cast<char*>(" + args_str_var_name + ")), " + std::to_string(args_size) +
                               ", dev_ext_info_mem_ptrs_[" + std::to_string(context.aicpu_task_index) + "], " +
                               std::to_string(ext_info_size) + ", " + iow_addr_var_name + ", " + args_var_name + ".data());");
  } else {
    REPORT_INNER_ERR_MSG("E19999", "Unsupported task type %d", static_cast<int32_t>(task_type));
    GELOGE(FAILED, "[OM2] Unsupported task type %d, task def %s", static_cast<int32_t>(task_type),
             context.task_def.ShortDebugString().c_str());
    return FAILED;
  }
  aicpu_args_code = aicpu_args_code_stream.str();
  GELOGD("[OM2] AssembleAicpuArgsCode end.");
  return SUCCESS;
}

Status KernelTaskCodeGenerator::GetKernelTaskMeta(const domi::TaskDef &task_def, domi::KernelContext &kernel_context,
                                                  uint32_t &args_size, uint32_t &kernel_type) const {
  if (Om2CodegenUtils::IsAllKernel(static_cast<ModelTaskType>(task_def.type()))) {
    const domi::KernelDefWithHandle &kernel_def = task_def.kernel_with_handle();
    args_size = static_cast<uint32_t>(kernel_def.args().size());
    kernel_context = kernel_def.context();
  } else {
    const domi::KernelDef &kernel_def = task_def.kernel();
    args_size = static_cast<uint32_t>(kernel_def.args().size());
    kernel_context = kernel_def.context();
  }
  kernel_type = kernel_context.kernel_type();
  return SUCCESS;
}

std::string KernelTaskCodeGenerator::SerializeBytesToOctalString(const std::vector<uint8_t> &buffer) {
  std::ostringstream code_stream;
  for (size_t i = 0; i < buffer.size(); ++i) {
    code_stream << "\\";
    code_stream << std::oct << std::setw(3) << std::setfill('0') << static_cast<int>(buffer[i]);
  }
  return code_stream.str();
}

std::string KernelTaskCodeGenerator::EmitLaunchConfigSetupCode(size_t op_index,
                                                               const Om2LaunchKernelConfig &launch_config) {
  std::stringstream code_stream;
  const std::string cfg_holder_var_name = "op" + std::to_string(op_index) + "_cfg_holder";
  code_stream << "  LaunchKernelCfgHolder " << cfg_holder_var_name << ";\n";
  code_stream << "  AssembleLaunchConfig(" << cfg_holder_var_name << ", " << GetConfigString(launch_config)
              << ");\n";
  return code_stream.str();
}

int64_t KernelTaskCodeGenerator::ParseOpIndex(const domi::TaskDef &task_def) {
  const auto task_type = static_cast<ModelTaskType>(task_def.type());
  domi::KernelContext context;
  if (!Om2CodegenUtils::IsAllKernel(task_type)) {
    const domi::KernelDef &kernel_def = task_def.kernel();
    context = kernel_def.context();
  } else {
    const domi::KernelDefWithHandle &kernel_def = task_def.kernel_with_handle();
    context = kernel_def.context();
  }
  return static_cast<int64_t>(context.op_index());
}

void KernelTaskCodeGenerator::AssembleLaunchKernelConfig(const OpDescPtr &op_desc, const domi::TaskDef &task_def,
                                                         Om2LaunchKernelParam &launch_param) {
  const auto task_type = static_cast<ModelTaskType>(task_def.type());
  launch_param.stream_id = task_def.stream_id();
  auto &cfg = launch_param.launch_config;
  GELOGD("[OM2] op %s has task type %u", op_desc->GetName().c_str(), task_def.type());
  if ((task_type == ModelTaskType::MODEL_TASK_VECTOR_ALL_KERNEL) ||
      (task_type == ModelTaskType::MODEL_TASK_VECTOR_KERNEL)) {
    cfg.engine_type = "ACL_RT_ENGINE_TYPE_AIV";
  }
  bool op_exec_never_timeout = false;
  if (AttrUtils::GetBool(op_desc, public_attr::OP_EXEC_NEVER_TIMEOUT, op_exec_never_timeout) && op_exec_never_timeout) {
    cfg.time_out = op_exec_never_timeout;
    GELOGI("[OM2] op %s type %s set never timeout", op_desc->GetName().c_str(), op_desc->GetTypePtr());
  }
  if (Om2CodegenUtils::IsAllKernel(task_type)) {
    const auto &kernel_def = task_def.kernel_with_handle();
    cfg.block_dim_offset = kernel_def.block_dim_offset();
    cfg.is_block_task_prefetch = kernel_def.is_block_task_prefetch();
    launch_param.block_dim = kernel_def.block_dim() == 0U ? 1U : kernel_def.block_dim();
    cfg.schedule_mode = static_cast<uint8_t>(kernel_def.schedule_mode() & k2BitsMask);
    // soft sync op场景tiling
  } else {
    const auto &kernel_def = task_def.kernel();
    cfg.block_dim_offset = kernel_def.block_dim_offset();
    cfg.is_block_task_prefetch = kernel_def.is_block_task_prefetch();
    launch_param.block_dim = kernel_def.block_dim() == 0U ? 1U : kernel_def.block_dim();
    auto kernel_type = static_cast<ccKernelType>(kernel_def.context().kernel_type());
    if (Om2CodegenUtils::IsAICoreKernel(kernel_type)) {  // 其他场景下，可能会从RunInfo中获取，待后续补齐
      cfg.schedule_mode = static_cast<uint8_t>(kernel_def.schedule_mode() & k2BitsMask);
    }
  }
}

Status KernelTaskCodeGenerator::UpdateShapeAndType(const std::vector<int64_t> &dims, const DataType data_type,
                                                   AicpuShapeAndType &shape_and_type) const {
  const auto dim_num = dims.size();
  if (dim_num > aicpu::FWKAdapter::kMaxShapeDims) {
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  size_t index = 0U;
  for (; index < dim_num; ++index) {
    shape_and_type.dims[index] = dims[index];
  }
  if (index < aicpu::FWKAdapter::kMaxShapeDims) {
    shape_and_type.dims[index] = kDimEndFlag;
  }

  shape_and_type.type = static_cast<int32_t>(data_type);
  return SUCCESS;
}

Status KernelTaskCodeGenerator::UpdateShapeAndType(const GeShape &shape, const DataType data_type,
                                               AicpuShapeAndType *const shape_and_type) {
  static_cast<void>(data_type);
  const auto dim_num = shape.GetDimNum();
  if (dim_num > aicpu::FWKAdapter::kMaxShapeDims) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[OM2][Check][DimNum]Update shape and type failed, as dim_num %zu is over max shape dims %u.",
           dim_num, aicpu::FWKAdapter::kMaxShapeDims);
    REPORT_INNER_ERR_MSG("E19999", "Update shape and type failed, as dim_num %zu is over max shape dims %u.",
                       dim_num, aicpu::FWKAdapter::kMaxShapeDims);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  size_t index = 0U;
  for (; index < dim_num; ++index) {
    shape_and_type->dims[index] = shape.GetDim(index);
  }
  if (index < aicpu::FWKAdapter::kMaxShapeDims) {
    shape_and_type->dims[index] = kDimEndFlag;
  }

  // now only support update shape, type is not support
  return SUCCESS;
}


Status KernelTaskCodeGenerator::InitAicpuTaskExtInfo(uint8_t *ext_info, size_t ext_info_len, const OpDescPtr op_desc, const domi::TaskDef &task_def, int32_t &session_info_offset) {
  GELOGD("[OM2] start to init aicpu task ext info.");
  (void)task_def;
  std::string node_name = op_desc->GetName();
  const uint32_t num_inputs = static_cast<uint32_t>(op_desc->GetInputsSize());
  const uint32_t num_outputs = static_cast<uint32_t>(op_desc->GetOutputsSize());

  std::vector<AicpuShapeAndType *> input_shape_and_type;
  std::vector<AicpuShapeAndType *> output_shape_and_type;
  input_shape_and_type.clear();
  output_shape_and_type.clear();

  bool all_shape = false;
  (void)AttrUtils::GetBool(op_desc, kAllShapeInAicpu, all_shape);
  size_t offset = 0UL;
  while ((offset + sizeof(AicpuExtInfo)) <= ext_info_len) {
    auto tmp_ext_info_data = PtrAdd(ext_info, ext_info_len, offset);
    GE_CHECK_NOTNULL(tmp_ext_info_data);
    auto &aicpu_ext_info = *(reinterpret_cast<AicpuExtInfo *>(tmp_ext_info_data));
    GELOGD("[OM2] Ext infoType=%d, infoLen=%u.", aicpu_ext_info.infoType, aicpu_ext_info.infoLen);
    switch (aicpu_ext_info.infoType) {
      case aicpu::FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE:
        GELOGI("[OM2] Reserve infoType[%d] for Node[%s].",
               aicpu_ext_info.infoType, node_name.c_str());
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE: {
        GE_IF_BOOL_EXEC(aicpu_ext_info.infoLen != (num_inputs * sizeof(AicpuShapeAndType)),
                  REPORT_INNER_ERR_MSG("E19999", "Node[%s] parse ext input shape failed as infoLen must be "
                                     "input_num[%u]*sizeof(ShapeAndType)[%zu] but %u.",
                                     node_name.c_str(), num_inputs, sizeof(AicpuShapeAndType),
                                     aicpu_ext_info.infoLen);
                  GELOGE(ACL_ERROR_GE_PARAM_INVALID,
                         "[OM2][Check][DataLen]Node[%s] parse ext input shape failed as infoLen must be "
                         "input_num[%u]*sizeof(ShapeAndType)[%zu] but %u.",
                         node_name.c_str(), num_inputs, sizeof(AicpuShapeAndType), aicpu_ext_info.infoLen);
                  return ACL_ERROR_GE_PARAM_INVALID;);

        const auto input = reinterpret_cast<AicpuShapeAndType *>(aicpu_ext_info.infoMsg);
        if (all_shape) {
          for (uint32_t i = 0U; i < num_inputs; ++i) {
            input_shape_and_type.emplace_back(PtrAdd<AicpuShapeAndType>(input, static_cast<size_t>(num_inputs),
                                                                        static_cast<size_t>(i)));
            const auto input_desc = op_desc->MutableInputDesc(i);
            GE_CHECK_NOTNULL(input_desc);
            const auto &shape = input_desc->GetShape();
            GE_CHK_STATUS_RET(UpdateShapeAndType(shape, input_desc->GetDataType(),
                                       input_shape_and_type[static_cast<size_t>(i)]),
                    "[OM2][Update][ShapeAndType] failed, Node[%s] input[%u] .",
                    node_name.c_str(), i);
          }
        }
        GELOGI("[OM2]Node[%s] parse ext input shape success infoLen=%u.", node_name.c_str(), aicpu_ext_info.infoLen);
        break;
      }
      case aicpu::FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE:{
        GE_IF_BOOL_EXEC(aicpu_ext_info.infoLen != (num_outputs * sizeof(AicpuShapeAndType)),
                        REPORT_INNER_ERR_MSG("E19999", "Node[%s] parse ext output shape failed as infoLen must be "
                                          "output_num[%u]*sizeof(ShapeAndType)[%zu] but %u.",
                                          node_name.c_str(), num_outputs, sizeof(AicpuShapeAndType),
                                          aicpu_ext_info.infoLen);
                        GELOGE(ACL_ERROR_GE_PARAM_INVALID,
                              "[OM2][Check][DataLen]Node[%s] parse ext output shape failed as infoLen must be "
                              "output_num[%u]*sizeof(ShapeAndType)[%zu] but %u.",
                              node_name.c_str(), num_outputs, sizeof(AicpuShapeAndType), aicpu_ext_info.infoLen);
                        return ACL_ERROR_GE_PARAM_INVALID;);

        const auto output = reinterpret_cast<AicpuShapeAndType *>(aicpu_ext_info.infoMsg);
        if (all_shape) {
          for (uint32_t i = 0U; i < num_outputs; ++i) {
            output_shape_and_type.emplace_back(PtrAdd<AicpuShapeAndType>(output, static_cast<size_t>(num_outputs),
                                                                        static_cast<size_t>(i)));
            const auto output_desc = op_desc->MutableOutputDesc(i);
            GE_CHECK_NOTNULL(output_desc);
            const auto &shape = output_desc->GetShape();
            GE_CHK_STATUS_RET(UpdateShapeAndType(shape, output_desc->GetDataType(),
                                       output_shape_and_type[static_cast<size_t>(i)]),
                    "[OM2][Update][ShapeAndType] failed, Node[%s] input[%u] .", node_name.c_str(), i);
          }
        }
        GELOGI("[OM2] Node[%s] parse ext output shape success infoLen=%u.", node_name.c_str(), aicpu_ext_info.infoLen);
        break;
      }
      case aicpu::FWKAdapter::FWK_ADPT_EXT_SESSION_INFO:
        session_info_offset = offset + 8;
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_BITMAP: {
        GE_IF_BOOL_EXEC(aicpu_ext_info.infoLen != sizeof(uint64_t),
                        REPORT_INNER_ERR_MSG("E19999",
                                          "Node[%s] parse bit_map info failed as infoLen must be %zu but %u.",
                                          node_name.c_str(), sizeof(uint64_t), aicpu_ext_info.infoLen);
                        GELOGE(PARAM_INVALID,
                              "[OM2][Check][DataLen]Node[%s] parse bit_map info failed as infoLen must be %zu but %u.",
                              node_name.c_str(), sizeof(uint64_t), aicpu_ext_info.infoLen);
                        return PARAM_INVALID;);

        uint64_t *bit_map = reinterpret_cast<uint64_t *>(aicpu_ext_info.infoMsg);
        *(bit_map) |= 1UL;
        GELOGI("[OM2] Node[%s] bit_map info success infoLen=%u, value = %" PRIu64 ".",
               node_name.c_str(), aicpu_ext_info.infoLen, *(bit_map));
        break;
      }
      case aicpu::FWKAdapter::FWK_ADPT_EXT_TOPIC_TYPE: {
        const int32_t type = *(reinterpret_cast<int32_t *>(aicpu_ext_info.infoMsg));
        int32_t deploy_type_flag = Om2CodegenUtils::TopicTypeToRtsFlag(type);
        if (deploy_type_flag == -1) {
          GELOGE(ACL_ERROR_GE_PARAM_INVALID,
            "[Check][Type]Node[%s] parse ext shape type failed as need %d %d %d %d but %d.",
            node_name.c_str(),
            aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_ONLY,
            aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_FIRST,
            aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_ONLY,
            aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_FIRST,
            type);
          return ACL_ERROR_GE_PARAM_INVALID;
        } else if (deploy_type_flag == static_cast<int32_t>(RT_KERNEL_HOST_ONLY)) {
          REPORT_INNER_ERR_MSG("E19999", "Unsupport scenario. Node[%s], infoType=%d, infoLen=%u.", node_name.c_str(),
                            aicpu_ext_info.infoType, aicpu_ext_info.infoLen);
          GELOGE(FAILED, "[OM2] Unsupport scenario. Node[%s], infoType=%d, infoLen=%u.",
                       node_name.c_str(), aicpu_ext_info.infoType, aicpu_ext_info.infoLen);
          return FAILED;
        }
        break;
      }
      case aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT: {
        REPORT_INNER_ERR_MSG("E19999", "Unsupport scenario. Node[%s], infoType=%d, infoLen=%u.", node_name.c_str(),
                          aicpu_ext_info.infoType, aicpu_ext_info.infoLen);
        GELOGE(FAILED, "[OM2] Unsupport scenario. Node[%s], infoType=%d, infoLen=%u.",
                       node_name.c_str(), aicpu_ext_info.infoType, aicpu_ext_info.infoLen);
        return FAILED;
      }
      default:
        GELOGD("[OM2] Node[%s] ignore infoType=%d, infoLen=%u.",
               node_name.c_str(), aicpu_ext_info.infoType, aicpu_ext_info.infoLen);
        break;
    }
    offset += sizeof(AicpuExtInfo);
    offset += aicpu_ext_info.infoLen;
  }

  GE_IF_BOOL_EXEC(offset != ext_info_len,
                  REPORT_INNER_ERR_MSG("E19999", "Node[%s] ext_info format error, parse not reach end,"
                                     "offset=%zu, ext_info_len=%zu.", node_name.c_str(), offset, ext_info_len);
                  GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[OM2][Check][Size]Node[%s] ext_info format error,"
                         "parse not reach end, offset=%zu, ext_info_len=%zu.",
                         node_name.c_str(), offset, ext_info_len);
                  return ACL_ERROR_GE_PARAM_INVALID;);
  GELOGI("[OM2] Node[%s] parse ext info end.", node_name.c_str());
  return SUCCESS;
}

Status KernelTaskCodeGenerator::ParseArgsFormat(TaskDistributionContext &context, ArgsFormatInfo &args_format_holder) {
  (void)OpDescUtils::GetIrInputInstanceDescRange(context.op_desc, args_format_holder.ir_input_2_range);
  (void)OpDescUtils::GetIrOutputDescRange(context.op_desc, args_format_holder.ir_output_2_range);
  auto &arg_descs = args_format_holder.arg_descs;
  auto input_descs = context.op_desc->GetAllInputsDescPtr();
  for (const auto &arg_format : arg_descs) {
    if (arg_format.addr_type == AddrType::INPUT_DESC) {
      GE_ASSERT(arg_format.ir_idx >= 0 &&
                static_cast<size_t>(arg_format.ir_idx) < args_format_holder.ir_input_2_range.size());
      const auto &ir_range = args_format_holder.ir_input_2_range[static_cast<size_t>(arg_format.ir_idx)];
      std::vector<int64_t> shape_info{0};  // placeholder for offset
      for (size_t idx = 0UL; idx < ir_range.second; ++idx) {
        const size_t instance_idx = static_cast<size_t>(ir_range.first + idx);
        GE_ASSERT_TRUE(instance_idx < input_descs.size(), "Instance index [%zu] is out of range, max_size:[%zu].",
                       instance_idx, input_descs.size());
        AppendShapeDesc(*input_descs.at(instance_idx), shape_info);
      }
      shape_info[0UL] = static_cast<int64_t>(shape_info.size() * sizeof(uintptr_t));
      args_format_holder.level1_addr_cnt += ir_range.second + shape_info.size();
      args_format_holder.shape_infos.push_back(shape_info);
    } else if (arg_format.addr_type == AddrType::OUTPUT_DESC) {
      GE_ASSERT(arg_format.ir_idx >= 0 &&
                static_cast<size_t>(arg_format.ir_idx) < args_format_holder.ir_output_2_range.size());
      const auto &ir_range = args_format_holder.ir_output_2_range[static_cast<size_t>(arg_format.ir_idx)];
      std::vector<int64_t> shape_info{0};  // placeholder for offset
      args_format_holder.level1_addr_cnt += ir_range.second;
      for (size_t idx = 0UL; idx < ir_range.second; ++idx) {
        auto output_desc = context.op_desc->MutableOutputDesc(static_cast<uint32_t>(ir_range.first + idx));
        GE_ASSERT_NOTNULL(output_desc);
        AppendShapeDesc(*output_desc, shape_info);
      }
      shape_info[0UL] = static_cast<int64_t>(shape_info.size() * sizeof(uintptr_t));
      args_format_holder.level1_addr_cnt += ir_range.second + shape_info.size();
      args_format_holder.shape_infos.push_back(shape_info);
    } else if (arg_format.addr_type == AddrType::TILING_CONTEXT &&
               arg_format.ir_idx == static_cast<int32_t>(TilingContextSubType::TILING_CONTEXT)) {
      REPORT_INNER_ERR_MSG("E19999", "Unsupport scenario. addr_type[%d], ir_idx[%d].",
                     static_cast<int32_t>(AddrType::TILING_CONTEXT),
                     static_cast<int32_t>(TilingContextSubType::TILING_CONTEXT));
      GELOGE(FAILED, "[OM2] Unsupport scenario. addr_type[%d], ir_idx[%d].", static_cast<int32_t>(AddrType::TILING_CONTEXT),
                     static_cast<int32_t>(TilingContextSubType::TILING_CONTEXT));
      return FAILED;
    } else if ((arg_format.addr_type == AddrType::TILING_CONTEXT) &&
               (arg_format.ir_idx == static_cast<int32_t>(TilingContextSubType::TILING_DATA))) {
      REPORT_INNER_ERR_MSG("E19999", "Unsupport scenario. addr_type[%d], ir_idx[%d].",
                     static_cast<int32_t>(AddrType::TILING_CONTEXT),
                     static_cast<int32_t>(TilingContextSubType::TILING_DATA));
      GELOGE(FAILED, "[OM2] Unsupport scenario. addr_type[%d], ir_idx[%d].", static_cast<int32_t>(AddrType::TILING_CONTEXT),
                     static_cast<int32_t>(TilingContextSubType::TILING_DATA));
      return FAILED;
    } else {
      // misra
    }
  }
  return SUCCESS;
}

size_t KernelTaskCodeGenerator::GetArgsSizeByFormat(const OpDescPtr op_desc, ArgsFormatInfo &args_format_holder) const {
  const auto &arg_descs = args_format_holder.arg_descs;
  size_t tmp_size = 0U;
  for (const auto &arg_desc : arg_descs) {
    (void)ArgsFormatDesc::GetArgSize(op_desc, arg_desc, tmp_size);
  }
  return tmp_size;
}

size_t KernelTaskCodeGenerator::GetExtraArgsSize(const OpDescPtr &op_desc, const ccKernelType kernel_type,
                                                 ArgsFormatInfo &args_format_holder) {
  size_t extra_size = 0UL;
  int32_t max_tiling_len{-1};
  (void)AttrUtils::GetInt(op_desc, kMaxTilingSize, max_tiling_len);
  int32_t max_atomic_tiling_len{-1};
  (void)AttrUtils::GetInt(op_desc, kMaxAtomicCleanTilingSize, max_atomic_tiling_len);
  if ((max_tiling_len > 0) || (max_atomic_tiling_len > 0)) {
    extra_size += kAddressLen;
  }

  if (kernel_type == ccKernelType::TE) {
    const auto is_wsp_addr_folded = IsWspAddrFolded(op_desc);
    if (is_wsp_addr_folded) {
      // kAddressLen: if folded mode, need add a memory for point to wsl addr list
      // kUBAlignedLen:
      // reserved 32B for aligned start with wsl addr list
      // -----------------------------------------------------------
      // | point to wsl addr list | over flow addr | wsl addr list |
      // -----------------------------------------------------------
      extra_size += kAddressLen + kUBAlignedLen;
    }
  }

  // level2 addr
  const size_t shape_info_size = args_format_holder.level1_addr_cnt * sizeof(int64_t);
  extra_size += shape_info_size;

  // reserved tiling sink tensor size
  return extra_size;
}

Status KernelTaskCodeGenerator::GenInputOutputAddrByInstanceIndex(TaskDistributionContext &context, size_t inst_idx, bool is_input) {
  std::vector<AddrGenInfo> &addrs = is_input ? input_addr_nodes_ : output_addr_nodes_;
  GE_ASSERT_TRUE(inst_idx < addrs.size(), "[OM2] Instance idx [%zu] is invalid, size:[%zu]", inst_idx, addrs.size());
  AppendIoAddrNodes(context, addrs[inst_idx]);
  return SUCCESS;
}

Status KernelTaskCodeGenerator::GenInputOutputAddr(TaskDistributionContext &context, ArgsFormatInfo &args_format_holder,
                                                   size_t ir_idx, bool is_input) {
  const std::map<size_t, std::pair<size_t, size_t>> &ir_2_range =
      is_input ? args_format_holder.ir_input_2_range : args_format_holder.ir_output_2_range;
  const auto iter = ir_2_range.find(ir_idx);
  GE_ASSERT(iter != ir_2_range.end(), "Ir idx [%zu] is not found, input flag %u.", ir_idx, is_input);
  const auto &range_pair = iter->second;
  if (is_input && range_pair.second == 0UL) {
    // optional input placeholder
    AppendPlaceholder(context);
    return SUCCESS;
  }
  size_t begin_idx = range_pair.first;

  std::vector<AddrGenInfo> &addrs = is_input ? input_addr_nodes_ : output_addr_nodes_;
  for (size_t i = 0UL; i < range_pair.second; ++i, ++begin_idx) {
    GE_ASSERT(begin_idx < addrs.size(), "[OM2] ir_idx:[%zu], begin_index [%zu] is out of range, max_size:[%zu].",
              ir_idx, begin_idx, addrs.size());
    AppendIoAddrNodes(context, addrs[begin_idx]);
  }
  return SUCCESS;
}

Status KernelTaskCodeGenerator::GenWorkspaceAddr(TaskDistributionContext &context, int32_t ir_idx) {
  auto host_args_offset = context.args_info.host_args_len + host_args_offset_;
  auto host_args_offset_str = host_args_offset == 0 ? "" : " + " + std::to_string(host_args_offset);
  if (ir_idx < 0) {
    for (const auto &inner : workspace_addr_nodes_) {
      AppendIoAddrNodes(context, inner);
    }
  } else {
    const size_t idx = static_cast<size_t>(ir_idx);
    GE_ASSERT(idx < workspace_addr_nodes_.size(),
              "[OM2] workspace index[%zu] is output of workspace addr nodes range[%zu]", idx,
              workspace_addr_nodes_.size());
    AppendIoAddrNodes(context, workspace_addr_nodes_[idx]);
  }
  return SUCCESS;
}

void KernelTaskCodeGenerator::AppendPlaceholder(TaskDistributionContext &context) {
  const auto op_index = context.op_index;
  AddrGenInfo addr_gen_info;
  addr_gen_info.var_name = "op" + std::to_string(op_index) + "_place_holder" + std::to_string(place_holder_var_index_);
  addr_gen_info.nodes.push_back(RAW_CODE_STMT(context.ast_ctx, "  uint64_t " + addr_gen_info.var_name + " = 0UL);"));
  AppendIoAddrNodes(context, addr_gen_info);
  place_holder_var_index_++;
}

void KernelTaskCodeGenerator::GenCustomValue(TaskDistributionContext &context, const uint64_t custom_value) {
  const auto op_index = context.op_index;
  AddrGenInfo addr_gen_info;
  addr_gen_info.var_name = "op" + std::to_string(op_index) + "_custom_value" + std::to_string(place_holder_var_index_);
  addr_gen_info.nodes.push_back(
    RAW_CODE_STMT(context.ast_ctx, "  auto " + addr_gen_info.var_name + " = " + std::to_string(custom_value) + ";"));
  AppendIoAddrNodes(context, addr_gen_info);
  cust_value_var_index_++;
}

Status KernelTaskCodeGenerator::AssembleShapeInfoAddrs(TaskDistributionContext &context,
                                                       const std::vector<ArgDesc> &dynamic_args_desc,
                                                       const std::vector<size_t> &level2_addr_idx,
                                                       ArgsFormatInfo &args_format_holder) {
  std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range = args_format_holder.ir_input_2_range;
  std::map<size_t, std::pair<size_t, size_t>> &ir_output_2_range = args_format_holder.ir_output_2_range;
  // append additional level1 addr
  GE_ASSERT(dynamic_args_desc.size() == args_format_holder.shape_infos.size());
  const auto op_index = context.op_index;
  for (size_t i = 0UL; i < dynamic_args_desc.size(); ++i) {
    auto &shape_info = args_format_holder.shape_infos[i];
    GE_ASSERT(level2_addr_idx[i] < args_addr_nodes_.size());

    // addr to ptr offset
    std::string io_desc_var_name = "op" + std::to_string(op_index) + "_io_desc" + std::to_string(i);
    auto host_args_offset = context.args_info.host_args_len + host_args_offset_;
    auto host_args_offset_str = std::to_string(host_args_offset);
    args_addr_nodes_[level2_addr_idx[i]].nodes.push_back(RAW_CODE_STMT(context.ast_ctx,
      "  auto " + io_desc_var_name + " = args_table_.GetDevArgAddr(" + host_args_offset_str + ");"));
    args_addr_nodes_[level2_addr_idx[i]].nodes.push_back(
        RAW_CODE_STMT(context.ast_ctx, "  OM2_CHK_NOTNULL(" + io_desc_var_name + ");"));
    GELOGD("[OM2] Set ptr_offset idx:[%zu], io index:[%zu]", args_addr_nodes_.size(), level2_addr_idx[i]);
    args_addr_nodes_[level2_addr_idx[i]].var_name = io_desc_var_name;

    // copy shape_infos
    AddrGenInfo addr_gen_info;
    addr_gen_info.var_name = "op" + std::to_string(op_index) + "_shape_info" + std::to_string(i);
    addr_gen_info.nodes.push_back(
        RAW_CODE_STMT(context.ast_ctx, "  std::vector<int64_t> " + addr_gen_info.var_name + " = " + GenShapeData(shape_info) + ";"));
    AppendIoAddrNodes(context, addr_gen_info, shape_info.size() * kAddressLen);

    auto addr_type = dynamic_args_desc[i].addr_type;
    if (addr_type != AddrType::INPUT_DESC && addr_type != AddrType::OUTPUT_DESC) {
      continue;
    }
    const size_t ir_idx = static_cast<size_t>(dynamic_args_desc[i].ir_idx);
    const auto &range_pair = addr_type == AddrType::INPUT_DESC ? ir_input_2_range[ir_idx] : ir_output_2_range[ir_idx];
    size_t begin_idx = range_pair.first;
    std::vector<AddrGenInfo> &addrs = addr_type == AddrType::INPUT_DESC ? input_addr_nodes_ : output_addr_nodes_;
    for (size_t idx = 0UL; idx < range_pair.second; ++idx) {
      AppendIoAddrNodes(context, addrs[begin_idx]);
      ++begin_idx;
    }
  }
  return SUCCESS;
}

Status KernelTaskCodeGenerator::GenArgsByArgsFormat(TaskDistributionContext &context, ArgsFormatInfo &args_format_holder) {
  const auto &arg_descs = args_format_holder.arg_descs;
  std::vector<ArgDesc> dynamic_args_desc;
  std::vector<size_t> level_addr_idx;
  std::vector<void *> context_addrs;
  for (const auto &arg_format : arg_descs) {
    switch (arg_format.addr_type) {
      case AddrType::INPUT_INSTANCE: {
        GE_ASSERT_SUCCESS(GenInputOutputAddrByInstanceIndex(context, static_cast<size_t>(arg_format.ir_idx), true));
        break;
      }
      case AddrType::OUTPUT_INSTANCE: {
        GE_ASSERT_SUCCESS(GenInputOutputAddrByInstanceIndex(context, static_cast<size_t>(arg_format.ir_idx), false));
        break;
      }
      case AddrType::INPUT_DESC:
      case AddrType::OUTPUT_DESC:
        level_addr_idx.push_back(args_addr_nodes_.size());
        dynamic_args_desc.push_back(arg_format);
        args_addr_nodes_.emplace_back();
        break;
      case AddrType::INPUT: {
        GE_ASSERT_SUCCESS(
            GenInputOutputAddr(context, args_format_holder, static_cast<size_t>(arg_format.ir_idx), true));
        break;
      }
      case AddrType::OUTPUT: {
        GE_ASSERT_SUCCESS(
            GenInputOutputAddr(context, args_format_holder, static_cast<size_t>(arg_format.ir_idx), false));
        break;
      }
      case AddrType::WORKSPACE: {
        GE_ASSERT_SUCCESS(GenWorkspaceAddr(context, arg_format.ir_idx));
        break;
      }
      case AddrType::PLACEHOLDER: {
        AppendPlaceholder(context);
        break;
      }
      case AddrType::CUSTOM_VALUE: {
        GenCustomValue(context, *reinterpret_cast<const uint64_t *>(arg_format.reserved));
        break;
      }
      case AddrType::HIDDEN_INPUT:
      case AddrType::TILING:
      case AddrType::OVERFLOW_ADDR:
      case AddrType::OP_TYPE:
      case AddrType::TILING_CONTEXT:
      case AddrType::FFTS_ADDR:
      case AddrType::EVENT_ADDR:
      default:
        GELOGE(FAILED, "Args Format type %d is currently not supported.", static_cast<int32_t>(arg_format.addr_type));
        GELOGE(FAILED, "[OM2] Args Format type %d is currently not supported.",
               static_cast<int32_t>(arg_format.addr_type));
        return FAILED;
    }
  }
  GE_ASSERT_SUCCESS(AssembleShapeInfoAddrs(context, dynamic_args_desc, level_addr_idx, args_format_holder));
  GELOGD("[OM2] All Args are successfully generated according to the args format.");

  return SUCCESS;
}

void KernelTaskCodeGenerator::AppendIoAddrNodes(TaskDistributionContext &context, const AddrGenInfo &src, uint64_t args_len) {
  args_addr_nodes_.emplace_back();
  (void) args_addr_nodes_.back().nodes.insert(args_addr_nodes_.back().nodes.cend(), src.nodes.cbegin(),
                                              src.nodes.cend());
  args_addr_nodes_.back().var_name = src.var_name;
  if (src.mem_type == Om2MemoryAppType::kMemoryTypeModelIo) {
    context.args_info.io_addr_offset.push_back(context.args_info.host_args_len + host_args_offset_);
  }
  host_args_offset_ += args_len;
}

REGISTER_TASK_CODE_GENERATOR(MODEL_TASK_KERNEL, KernelTaskCodeGenerator);
REGISTER_TASK_CODE_GENERATOR(MODEL_TASK_ALL_KERNEL, KernelTaskCodeGenerator);
REGISTER_TASK_CODE_GENERATOR(MODEL_TASK_VECTOR_KERNEL, KernelTaskCodeGenerator);
REGISTER_TASK_CODE_GENERATOR(MODEL_TASK_VECTOR_ALL_KERNEL, KernelTaskCodeGenerator);
REGISTER_TASK_CODE_GENERATOR(MODEL_TASK_PREPROCESS_KERNEL, KernelTaskCodeGenerator);
}  // namespace ge

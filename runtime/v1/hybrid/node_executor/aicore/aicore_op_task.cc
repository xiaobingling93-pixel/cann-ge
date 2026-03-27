/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/debug/log.h"
#include "common/profiling_definitions.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "common/op_tiling/op_tiling_rt2.h"
#include "common/checker.h"
#include "graph/types.h"
#include "framework/common/taskdown_common.h"
#include "framework/generator/ge_generator.h"
#include "graph/bin_cache/op_binary_cache.h"
#include "graph/ge_context.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/node_executor/aicore/aicore_task_builder.h"
#include "single_op/task/build_task_utils.h"
#include "single_op/task/tbe_task_builder.h"
#include "common/utils/executor_utils.h"
#include "hybrid/node_executor/aicore/aicore_op_task.h"

namespace ge {
namespace hybrid {
namespace {
constexpr char_t const *kAttrOpParamSize = "op_para_size";
constexpr char_t const *kAttrAtomicOpParamSize = "atomic_op_para_size";
constexpr char_t const *kDefaultCoreType = "AiCore";
std::atomic<std::uint64_t> log_id(0U);
// size,dim1,...,dim8: 9*4=36
const uint32_t kMaxDimNum = 8U;
const size_t kShapeBufferNum = 1U + kMaxDimNum;
const size_t kShapeBufferSize = sizeof(uint32_t) * kShapeBufferNum;
const size_t kOverflowAddrIndex = 2U;  // distance of tiling_addr to the end
}  // namespace

Status AiCoreOpTask::Init(const NodePtr &node, const domi::TaskDef &task_def) {
  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  GE_CHK_STATUS_RET_NOLOG(DoInit(node, task_def));
  int32_t unknown_shape_op_type_val = static_cast<int32_t>(DEPEND_IN_SHAPE);
  (void)AttrUtils::GetInt(op_desc, ::ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_op_type_val);
  unknown_shape_op_type_ = static_cast<UnknowShapeOpType>(unknown_shape_op_type_val);
  GELOGD("Op [%s] unknown shape type is %d", op_desc->GetName().c_str(), unknown_shape_op_type_);
  if (unknown_shape_op_type_ == DEPEND_SHAPE_RANGE) {
    const size_t size = kShapeBufferSize * (op_desc->GetOutputsSize());
    if (size == 0U) {
      GELOGE(PARAM_INVALID, "Op [%s] unknown shape type is %d, but outputs size is 0.", op_desc->GetName().c_str(),
             unknown_shape_op_type_);
      return PARAM_INVALID;
    }
    const auto npu_mem_allocator = NpuMemoryAllocator::GetAllocator();
    GE_CHECK_NOTNULL(npu_mem_allocator);
    shape_buffer_ = TensorBuffer::Create(npu_mem_allocator, size);
    GE_CHECK_NOTNULL(shape_buffer_);
    GELOGD("Op [%s] allocate memory for outputs shape success, size=%zu", op_desc->GetName().c_str(), size);
    GE_CHK_RT_RET(rtMemset(shape_buffer_->GetData(), shape_buffer_->GetSize(), 0U, shape_buffer_->GetSize()));
    host_shape_buffer_ = MakeUnique<uint8_t[]>(shape_buffer_->GetSize());
    GE_CHECK_NOTNULL(host_shape_buffer_);
  }
  return SUCCESS;
}

Status AiCoreOpTask::DoInit(const NodePtr &node, const domi::TaskDef &task_def) {
  const auto op_desc_ptr = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc_ptr);
  op_type_ = op_desc_ptr->GetType();
  op_name_ = op_desc_ptr->GetName();
  log_name_ = op_desc_ptr->GetName() + "_tvmbin";
  log_id_ = log_id++;

  const auto task_info = BuildTaskUtils::GetTaskInfo(op_desc_ptr);
  GELOGI("[TASK_INFO] %lu/%s %s.", log_id_, log_name_.c_str(), task_info.c_str());
  const auto op_desc = *op_desc_ptr;
  need_overflow_addr_ = AttrUtils::HasAttr(op_desc, GLOBALWORKSPACE_TYPE) && (overflow_addr_ != nullptr);
  GE_CHK_STATUS_RET_NOLOG(InitTilingInfo(op_desc));
  GE_CHK_STATUS_RET_NOLOG(InitWithTaskDef(node, task_def));

  GE_CHECK_LE(op_desc.GetOutputsSize(), static_cast<size_t>(INT_MAX));
  const int32_t outputs_size = static_cast<int32_t>(op_desc.GetOutputsSize());

  for (int32_t i = 0; i < outputs_size; ++i) {
    const GeTensorDescPtr tensor_desc = op_desc.MutableOutputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      GELOGW("Op: %s, Index: %d, Tensor Desc is null", op_desc.GetName().c_str(), i);
      continue;
    }

    if (TensorUtils::IsMemorySizeCalcTypeAlwaysEmpty(*tensor_desc)) {
      output_indices_to_skip_.push_back(i);
    }
  }
  return SUCCESS;
}

Status AiCoreOpTask::UpdateOutputsShape(const TaskContext &context) const {
  GELOGD("Node[%s] start update outputs shape.", context.GetNodeName());
  GE_CHECK_NOTNULL(shape_buffer_);
  GE_CHECK_NOTNULL(host_shape_buffer_);
  GE_CHK_RT_RET(rtMemcpy(host_shape_buffer_.get(), shape_buffer_->GetSize(), shape_buffer_->GetData(),
                         shape_buffer_->GetSize(), RT_MEMCPY_DEVICE_TO_HOST));
  const auto outputs_shape = reinterpret_cast<uint32_t(*)[kShapeBufferNum]>(host_shape_buffer_.get());
  const int32_t num_outputs = context.NumOutputs();
  for (int32_t i = 0; i < num_outputs; ++i) {
    if (outputs_shape[i][0] != 0U) {
      const uint32_t dim_num = outputs_shape[i][0];
      if (dim_num > kMaxDimNum) {
        GELOGE(PARAM_INVALID, "Node[%s] output[%d] dim_num=[%u] is greater than MaxDimNum[%u]", context.GetNodeName(),
               i, dim_num, kMaxDimNum);
        return PARAM_INVALID;
      }
      std::vector<int64_t> dims;
      for (uint32_t j = 1U; j <= dim_num; ++j) {
        dims.emplace_back(static_cast<int64_t>(outputs_shape[i][j]));
      }
      const GeShape shape_new(dims);
      GELOGD("Node[%s] output[%d] shape:%s.", context.GetNodeName(), i, ToString(dims).c_str());
      GE_CHK_STATUS_RET_NOLOG(UpdateShapeToOutputDesc(context, shape_new, i));
    }
  }
  return SUCCESS;
}

Status AiCoreOpTask::UpdateShapeToOutputDesc(const TaskContext &context, const GeShape &shape,
                                             const int32_t output_index) const {
  const auto output_desc = context.MutableOutputDesc(output_index);
  GE_CHECK_NOTNULL(output_desc);
  const auto origin_format = output_desc->GetOriginFormat();
  const auto format = output_desc->GetFormat();
  if (origin_format != format) {
    GELOGE(PARAM_INVALID, "Node[%s] output[%d] origin_format[%s] != foramt[%s], is not supported.", context.GetNodeName(),
           output_index, GetFormatName(origin_format), GetFormatName(format));
    return PARAM_INVALID;
  }
  const auto &shape_old = output_desc->GetShape();
  const auto &origin_shape_old = output_desc->GetOriginShape();
  const auto node_state = context.GetNodeState();
  GE_CHECK_NOTNULL(node_state);
  GELOGD("Node[%s] try to update output[%d] shape from [%s] to [%s], origin_shape "
         "from [%s] to [%s].",
         context.GetNodeName(), output_index, shape_old.ToString().c_str(), shape.ToString().c_str(),
         origin_shape_old.ToString().c_str(), shape.ToString().c_str());
  GE_CHK_STATUS_RET(node_state->UpdateOutputShapes(output_index, shape, shape),
                    "Node[%s] try to update output[%d] shape from [%s] to [%s], origin_shape "
                    "from [%s] to [%s] failed.",
                    context.GetNodeName(), output_index, shape_old.ToString().c_str(), shape.ToString().c_str(),
                    origin_shape_old.ToString().c_str(), shape.ToString().c_str());
  return SUCCESS;
}

Status AiCoreOpTask::InitWithKernelDef(const OpDesc &op_desc, const domi::TaskDef &task_def) {
  const domi::KernelDef &kernel_def = task_def.kernel();
  const domi::KernelContext &context = kernel_def.context();
  stub_name_ = (is_single_op_ ? std::to_string(log_id_) : std::to_string(model_id_)) + kernel_def.stub_func();
  GE_CHK_STATUS_RET(BinRegisterUtils::RegisterBin(op_desc, stub_name_, GetAttrNamesForBin(), stub_func_));
  args_size_without_tiling_ = kernel_def.args_size();
  if (kernel_def.args().size() < args_size_without_tiling_) {
    GELOGE(INTERNAL_ERROR, "[Check][Size]args size:%zu of kernel_def is smaller than args_size_:%u, op:%s op_type:%s",
           kernel_def.args().size(), args_size_without_tiling_, op_desc.GetName().c_str(), op_desc.GetType().c_str());
    REPORT_INNER_ERR_MSG("E19999", "args size:%zu of kernel_def is smaller than args_size_:%u op:%s op_type:%s.",
                       kernel_def.args().size(), args_size_without_tiling_, op_desc.GetName().c_str(),
                       op_desc.GetType().c_str());
    return INTERNAL_ERROR;
  }

  GE_CHK_STATUS_RET(CheckUint32AddOverflow(args_size_without_tiling_, max_tiling_size_),
                    "arg size is beyond the UINT32_MAX.");
  args_size_ = args_size_without_tiling_ + max_tiling_size_;
  if (ExecutorUtils::HasHostMemInput(MakeShared<OpDesc>(op_desc))) {
    host_mem_input_data_offset_ = static_cast<size_t>(args_size_);
    GE_CHK_STATUS_RET(CheckUint32AddOverflow(args_size_, static_cast<uint32_t>(kMaxHostMemInputLen)),
                      "arg size is beyond the UINT32_MAX.");
    args_size_ += static_cast<uint32_t>(kMaxHostMemInputLen);
    GELOGD("[%s] has host memory input, args size is extened %zu, args_size_ = %u, host_mem_input_data_offset = %zu",
           GetName().c_str(), kMaxHostMemInputLen, args_size_, host_mem_input_data_offset_);
  }
  block_dim_ = kernel_def.block_dim();
  // malloc args memory
  args_ = MakeUnique<uint8_t[]>(static_cast<size_t>(args_size_));
  GE_CHECK_NOTNULL(args_);
  const errno_t err = memcpy_s(args_.get(), static_cast<size_t>(args_size_), kernel_def.args().data(),
                               static_cast<size_t>(args_size_without_tiling_));
  if (err != EOK) {
    GELOGE(INTERNAL_ERROR, "[Update][Date]AiCoreTask memcpy args failed, op:%s op_type:%s.",
           op_desc.GetName().c_str(), op_desc.GetType().c_str());
    REPORT_INNER_ERR_MSG("E19999", "AiCoreTask memcpy args failed, op:%s op_type:%s.",
                       op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return INTERNAL_ERROR;
  }

  if (context.args_offset().size() < sizeof(uint16_t)) {
    GELOGE(INTERNAL_ERROR, "[Check][Size]Invalid args_offset,"
           "size:%zu is smaller than size of uint16_t, op:%s op_type:%s",
           context.args_offset().size(), op_desc.GetName().c_str(), op_desc.GetType().c_str());
    REPORT_INNER_ERR_MSG("E19999", "Invalid args_offset, size:%zu is smaller than size of uint16_t, op:%s op_type:%s",
                       context.args_offset().size(), op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return INTERNAL_ERROR;
  }

  const auto *const args_offset_buffer = PtrToPtr<const char_t, const uint16_t>(context.args_offset().data());
  offset_ = *args_offset_buffer;
  if (offset_ > args_size_without_tiling_) {
    GELOGE(INTERNAL_ERROR, "[Check][Offset][%s] Arg offset out of range. offset = %u,"
           "arg size = %u , op:%s op_type:%s", GetName().c_str(), offset_, args_size_without_tiling_,
           op_desc.GetName().c_str(), op_desc.GetType().c_str());
    REPORT_INNER_ERR_MSG("E19999", "[%s] Arg offset out of range. offset = %u, arg size = %u"
                       "op:%s op_type:%s", GetName().c_str(), offset_, args_size_without_tiling_,
                       op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return INTERNAL_ERROR;
  }

  arg_base_ = PtrToPtr<uint8_t, uintptr_t>(&args_[static_cast<size_t>(offset_)]);
  max_arg_count_ = static_cast<uint32_t>(static_cast<size_t>(args_size_without_tiling_ - offset_) / sizeof(void *));
  GELOGD("[%s] Done setting kernel args successfully. stub_func = %s, block_dim = %d,"
         "arg base = %p, arg size = %u",
         op_desc.GetName().c_str(),  stub_name_.c_str(),
         block_dim_, arg_base_, args_size_without_tiling_);
  return SUCCESS;
}

Status AiCoreOpTask::InitWithKernelDefWithHandle(const OpDesc &op_desc, const domi::TaskDef &task_def) {
  const domi::KernelDefWithHandle &kernel_with_handle = task_def.kernel_with_handle();
  const domi::KernelContext &context = kernel_with_handle.context();
  GE_CHK_STATUS_RET(BinRegisterUtils::RegisterBinWithHandle(op_desc, GetAttrNamesForBin(), handle_));
  node_info_ = kernel_with_handle.node_info() + "/";
  args_size_without_tiling_ = kernel_with_handle.args_size();
  if (kernel_with_handle.args().size() < args_size_without_tiling_) {
    GELOGE(INTERNAL_ERROR, "[Check][Size]args size:%zu of kernel_def is smaller than args_size_:%u. op:%s op_type:%s",
           kernel_with_handle.args().size(), args_size_without_tiling_, op_desc.GetName().c_str(),
           op_desc.GetType().c_str());
    REPORT_INNER_ERR_MSG("E19999", "args size:%zu of kernel_def is smaller than args_size_:%u. op:%s op_type:%s",
                       kernel_with_handle.args().size(), args_size_without_tiling_,
                       op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return INTERNAL_ERROR;
  }

  block_dim_ = kernel_with_handle.block_dim();
  GE_ASSERT_TRUE(!ge::AddOverflow(args_size_without_tiling_, max_tiling_size_, args_size_));
  if (ExecutorUtils::HasHostMemInput(MakeShared<OpDesc>(op_desc))) {
    host_mem_input_data_offset_ = static_cast<size_t>(args_size_);
    GE_ASSERT_TRUE(!ge::AddOverflow(args_size_, static_cast<uint32_t>(kMaxHostMemInputLen), args_size_));
    GELOGD("[%s] has host memory input, args size is extened %zu, args_size_ = %u, host_mem_input_data_offset = %zu",
           GetName().c_str(), kMaxHostMemInputLen, args_size_, host_mem_input_data_offset_);
  }
  // malloc args memory
  args_ = MakeUnique<uint8_t[]>(static_cast<size_t>(args_size_));
  GE_CHECK_NOTNULL(args_);
  const errno_t err = memcpy_s(args_.get(), static_cast<size_t>(args_size_), kernel_with_handle.args().data(),
                               static_cast<size_t>(args_size_without_tiling_));
  if (err != EOK) {
    GELOGE(INTERNAL_ERROR, "[Update][Date]AiCoreTask memcpy args failed. op:%s op_type:%s",
           op_desc.GetName().c_str(), op_desc.GetType().c_str());
    REPORT_INNER_ERR_MSG("E19999", "AiCoreTask memcpy args failed. op:%s op_type:%s",
                      op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return INTERNAL_ERROR;
  }

  if (context.args_offset().size() < sizeof(uint16_t)) {
    GELOGE(INTERNAL_ERROR, "[Check][Size]Invalid args_offset, size:%zu is smaller"
           "than size of uint16_t. op:%s op_type:%s", context.args_offset().size(),
           op_desc.GetName().c_str(), op_desc.GetType().c_str());
    REPORT_INNER_ERR_MSG("E19999", "Invalid args_offset, size:%zu is smaller"
                       "than size of uint16_t. op:%s op_type:%s", context.args_offset().size(),
                       op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return INTERNAL_ERROR;
  }

  const auto *const args_offset_buffer = PtrToPtr<const char_t, const uint16_t>(context.args_offset().data());
  offset_ = *args_offset_buffer;
  if (offset_ > args_size_without_tiling_) {
    GELOGE(INTERNAL_ERROR, "[Check][Offset][%s] Arg offset out of range. offset = %u, arg size = %u"
           "op:%s op_type:%s", GetName().c_str(), offset_, args_size_without_tiling_,
           op_desc.GetName().c_str(), op_desc.GetType().c_str());
    REPORT_INNER_ERR_MSG("E19999", "[%s] Arg offset out of range. offset = %u, arg size = %u"
                       "op:%s op_type:%s", GetName().c_str(), offset_, args_size_without_tiling_,
                       op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return INTERNAL_ERROR;
  }

  arg_base_ = PtrToPtr<uint8_t, uintptr_t>(&args_[static_cast<size_t>(offset_)]);
  max_arg_count_ = static_cast<uint32_t>(static_cast<size_t>((args_size_without_tiling_ - offset_)) / sizeof(void *));
  return SUCCESS;
}

Status AiCoreOpTask::InitWithTaskDef(const NodePtr &node, const domi::TaskDef &task_def) {
  const auto op_desc = *(node->GetOpDesc()); // checked not null outside
  const auto rt_ret = ValidateTaskDef(task_def);
  if (rt_ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "op:%s(op_type:%s) failed to validate task def:%s",
                      op_desc.GetName().c_str(), op_desc.GetType().c_str(), task_def.DebugString().c_str());
    GELOGE(rt_ret, "[Invoke][ValidateTaskDef]failed for op:%s(op_type:%s) to validate task def:%s",
           op_desc.GetName().c_str(), op_desc.GetType().c_str(), task_def.DebugString().c_str());
    return rt_ret;
  }

  KernelLaunchBinType bin_type = KernelLaunchBinType::kBinTypeEnd;
  if (static_cast<ModelTaskType>(task_def.type()) != ModelTaskType::MODEL_TASK_ALL_KERNEL) {
    bin_type = KernelLaunchBinType::kStubFunc;
    GE_CHK_STATUS_RET(InitWithKernelDef(op_desc, task_def));
  } else {
    bin_type = KernelLaunchBinType::kWithHandle;
    GE_CHK_STATUS_RET(InitWithKernelDefWithHandle(op_desc, task_def));
  }
  (void)AddCompileCacheItem(bin_type, node);

  args_ex_.hasTiling = need_tiling_;
  args_ex_.args = args_.get();
  args_ex_.argsSize = args_size_;
  if (need_tiling_) {
    const size_t offset = need_overflow_addr_ ? kOverflowAddrIndex * sizeof(void *) : sizeof(void *);
    GE_CHECK_GE(args_size_without_tiling_, offset);
    args_ex_.tilingAddrOffset = static_cast<uint32_t>(args_size_without_tiling_ - offset);
    args_ex_.tilingDataOffset = static_cast<uint32_t>(args_size_without_tiling_);
    tiling_data_idx_ = args_size_without_tiling_ / sizeof(void *);
  }
  tiling_info_ = MakeUnique<optiling::utils::OpRunInfo>(0U, true, 0UL);
  GE_CHECK_NOTNULL(tiling_info_);
  return SUCCESS;
}

Status AiCoreOpTask::AddCompileCacheItem(const KernelLaunchBinType bin_type, const NodePtr &node) const {
  const auto bin_handle = (bin_type == KernelLaunchBinType::kStubFunc) ? stub_func_ : handle_;
  NodeCompileCacheItem item;
  GE_CHK_STATUS_RET_NOLOG(NodeCompileCacheItem::Build(bin_type, node, bin_handle, item));
  const auto ret = OpBinaryCache::GetInstance().GetNodeCcm()->AddCompileCache(node, item);
  if (ret == nullptr) {
    GELOGW("Fail to add compile cache %s.", node->GetName().c_str());
    return SUCCESS;
  }
  GELOGD("Add compile cache of %s successfully.", node->GetName().c_str());
  return SUCCESS;
}

Status AiCoreOpTask::ValidateTaskDef(const domi::TaskDef &task_def) {
  const auto task_type = static_cast<ModelTaskType>(task_def.type());
  if ((task_type != ModelTaskType::MODEL_TASK_KERNEL) && (task_type != ModelTaskType::MODEL_TASK_ALL_KERNEL)) {
    GELOGE(INTERNAL_ERROR,
           "[Check][TaskType]Invalid task type (%d) in AiCore CreateTask.", static_cast<int32_t>(task_type));
    return INTERNAL_ERROR;
  }
  const auto &context = (task_type == ModelTaskType::MODEL_TASK_KERNEL) ? task_def.kernel().context() :
      task_def.kernel_with_handle().context();
  const auto kernel_type = static_cast<ccKernelType>(context.kernel_type());
  if (kernel_type != ccKernelType::TE) {
    GELOGE(INTERNAL_ERROR,
           "[Check][TaskType]Invalid kernel type(%d) in AiCore TaskDef.", static_cast<int32_t>(kernel_type));
    REPORT_INNER_ERR_MSG("E19999", "Invalid kernel type(%d) in AiCore TaskDef.",
                       static_cast<int32_t>(kernel_type));
    return INTERNAL_ERROR;
  }

  return SUCCESS;
}

Status AiCoreOpTask::PrepareWithShape(const TaskContext &context) {
  if (is_dynamic_) {
    return UpdateTilingInfo(context);
  }
  return SUCCESS;
}

Status AiCoreOpTask::UpdateTilingInfo(const TaskContext &context) {
  const auto node = context.GetNodeItem().node;
  GE_CHECK_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  GELOGD("[%s] Start to calc tiling data for task: [%s]", node->GetName().c_str(), stub_name_.c_str());
  tiling_info_->ResetWorkspace();
  const auto ptr_size = sizeof(void *);
  GE_CHECK_LE(offset_ + max_tiling_size_ + (ptr_size * tiling_data_idx_), args_size_);
  void *const tiling_data_addr = ValueToPtr(PtrToValue(arg_base_) + (ptr_size * tiling_data_idx_));
  tiling_info_->ResetAddrBase(tiling_data_addr, static_cast<size_t>(max_tiling_size_));
  const auto execution_context = context.GetExecutionContext();
  GE_CHECK_NOTNULL(context.GetNodeState());
  const auto op = context.GetNodeState()->GetOperator(execution_context->stage_id);
  GE_CHECK_NOTNULL(op);
  RECORD_EXECUTION_EVENT(execution_context, context.GetNodeName(), "[CalcTilingInfo] Start");
  GE_CHK_STATUS_RET_NOLOG(CalcTilingInfo(node, *op));
  RECORD_EXECUTION_EVENT(execution_context, context.GetNodeName(), "[CalcTilingInfo] End");
  if (tiling_info_->GetAllTilingData().tellp() > 0) {
    GELOGD("[%s] Need to copy tiling data for task: [%s].", node->GetName().c_str(), stub_name_.c_str());
    const std::string tiling_data = tiling_info_->GetAllTilingData().str();
    if (memcpy_s(tiling_data_addr, static_cast<size_t>(max_tiling_size_), tiling_data.data(),
                 tiling_data.size()) != EOK) {
      GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Update][Args]failed, dst length is %u, src length is %zu.",
             max_tiling_size_, tiling_data.size());
      REPORT_INNER_ERR_MSG("E19999", "update kernel args failed of %s.", context.GetNodeName());
      return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
    }
    tiling_info_->GetAllTilingData().str("");
  }
  // update op args by tiling info
  block_dim_ = tiling_info_->GetBlockDim();
  clear_atomic_ = tiling_info_->GetClearAtomic();
  tiling_key_ = tiling_info_->GetTilingKey();
  local_memory_size_ = tiling_info_->GetLocalMemorySize();
  GELOGD(
      "[%s] Successfully getting [block dim] : %u, [clear atomic] : %d, [tiling key] : %lu, [local memory size] :%u, "
      "for task [%s].",
      node->GetName().c_str(), block_dim_, static_cast<int32_t>(clear_atomic_), tiling_key_, local_memory_size_,
      stub_name_.c_str());

  GELOGD("[%s] Done updating tiling info for task: [%s]", node->GetName().c_str(), stub_name_.c_str());
  return SUCCESS;
}

Status AiCoreOpTask::CalcTilingInfo(const NodePtr &node, const Operator &op) const {
  GELOGD("[%s] Start to invoke OpParaCalculate.", node->GetName().c_str());
  graphStatus ret;
  if (optiling::EnableRt2Tiling(node->GetOpDesc())) {
    ret = optiling::AicoreRtParseAndTiling(op, platform_infos_, *tiling_info_);
  } else {
    ret = optiling::OpParaCalculateV2(op, *tiling_info_);
  }
  if (ret != SUCCESS) {
    return ret;
  }
  // Only non atomic task need update workspace
  const auto op_desc = node->GetOpDesc();
  op_desc->SetWorkspaceBytes(tiling_info_->GetAllWorkspaces());
  GELOGD("[%s] Done invoking OpParaCalculate successfully.", node->GetName().c_str());
  return SUCCESS;
}

Status AiCoreOpTask::SetPlatformInfo(const NodePtr &node, const fe::PlatFormInfos &platform_info) {
  platform_infos_ = platform_info;
  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  const std::string *core_type = AttrUtils::GetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE);
#ifdef __GNUC__
  if (core_type == nullptr) {
    platform_infos_.SetCoreNumByCoreType(kDefaultCoreType);
  } else {
    platform_infos_.SetCoreNumByCoreType(*core_type);
  }
#endif
  return SUCCESS;
}

Status AiCoreOpTask::UpdateBinHandle(const TaskContext &task_context, const NodeCompileCacheItem &cci) {
  if (cci.GetBinType() == KernelLaunchBinType::kBinTypeEnd) {
    return SUCCESS;
  }
  GELOGI("Node %s update bin handle. Got compile_cache_item id:%lu bin_type is %d.", op_name_.c_str(),
         cci.GetCacheItemId(), static_cast<int32_t>(cci.GetBinType()));
  if (cci.GetBinType() == KernelLaunchBinType::kStubFunc) {
    stub_func_ = cci.GetBinHandle();
    handle_ = nullptr;
  } else if (cci.GetBinType() == KernelLaunchBinType::kWithHandle) {
    handle_ = cci.GetBinHandle();
    stub_func_ = nullptr;
  } else {
    GELOGW("Unsupported bin_type %d.", static_cast<int32_t>(cci.GetBinType()));
    return FAILED;
  }

  const auto &node = task_context.GetNodeItem().node;
  GE_CHECK_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  // update compile info on node
  (void)AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, cci.GetCompileInfo()->key);
  (void)AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, cci.GetCompileInfo()->str);
  (void)AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, cci.GetAtomicCompileInfo()->key);
  (void)AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, cci.GetAtomicCompileInfo()->str);
  (void)AttrUtils::SetInt(op_desc, kAttrOpParamSize, cci.GetMaxTilingSize());
  (void)AttrUtils::SetInt(op_desc, kAttrAtomicOpParamSize, cci.GetAtomicMaxTilingSize());
  (void)AttrUtils::SetBool(op_desc, kAttrSupportDynamicShape, cci.IsSupportDynamic());
  GE_CHK_STATUS_RET_NOLOG(InitTilingInfo(*op_desc, true));
  return SUCCESS;
}

Status AiCoreOpTask::UpdateHostMemInputArgs(const TaskContext &task_context) {
  args_ex_.hostInputInfoPtr = nullptr;
  args_ex_.hostInputInfoNum = 0U;
  if (!IsNeedHostMemOpt()) {
    return SUCCESS;
  }
  std::vector<rtHostInputInfo_t> host_inputs;
  GE_CHK_RT_RET(ExecutorUtils::UpdateHostMemInputArgs(task_context, PtrToPtr<uintptr_t, uint64_t>(arg_base_),
                                                      args_size_, host_mem_input_data_offset_, host_inputs));

  host_inputs_info_ = MakeUnique<rtHostInputInfo_t []>(host_inputs.size());
  GE_CHECK_NOTNULL(host_inputs_info_);
  size_t idx = 0UL;
  for (const auto &host_input : host_inputs) {
    host_inputs_info_[idx] = host_input;
    idx++;
  }
  args_ex_.hostInputInfoPtr = host_inputs_info_.get();
  args_ex_.hostInputInfoNum = host_inputs.size();
  return SUCCESS;
}

Status AiCoreOpTask::UpdateArgBase(const TaskContext &task_context, uint32_t &index, const size_t overflow_arg_index) {
  for (int32_t i = 0; i < task_context.NumInputs(); ++i) {
    const auto input = task_context.GetInput(i);
    GE_CHECK_NOTNULL(input);
    arg_base_[index] = PtrToValue(input->GetData());
    index++;
  }
  GE_CHK_STATUS_RET(UpdateHostMemInputArgs(task_context), "[Update][HostMemInputArgs] failed.") ;
  for (int32_t i = 0; i < task_context.NumOutputs(); ++i) {
    const auto output = task_context.GetOutput(i);
    GE_CHECK_NOTNULL(output);
    if (find(output_indices_to_skip_.begin(), output_indices_to_skip_.end(), i) !=
        output_indices_to_skip_.end()) {
      GELOGD("Node:%s output[%d] is an optional, the address don't need to be saved.",
             task_context.GetNodeName(), i);
      continue;
    }
    arg_base_[index] = PtrToValue(output->GetData());
    index++;
  }

  if (shape_buffer_ != nullptr) {
    arg_base_[index] = PtrToValue(shape_buffer_->GetData());
    index++;
    GELOGD("Node:%s add shape buffer addr to args.", task_context.GetNodeName());
  }

  const int32_t workspace_num = static_cast<int32_t>(task_context.NumWorkspaces());
  for (int32_t i = 0; i < workspace_num; ++i) {
    const auto workspace = task_context.MutableWorkspace(i);
    GE_CHECK_NOTNULL(workspace);
    arg_base_[index] = PtrToValue(workspace);
    index++;
  }

  // Refresh aicore overflow detetcion addr
  if (need_overflow_addr_) {
    arg_base_[overflow_arg_index] = PtrToValue(overflow_addr_);
    index++;
  }

  return SUCCESS;
}

Status AiCoreOpTask::ExpandArgsByLength(const size_t length, const std::string &name) {
  std::unique_ptr<uint8_t[]> new_args = MakeUnique<uint8_t[]>(length);
  GE_CHECK_NOTNULL(new_args);
  if (memcpy_s(new_args.get(), length, args_.get(), static_cast<size_t>(args_size_)) != EOK) {
    GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Update][Args]failed, dst length is %zu, src length is %u.",
           length, args_size_);
    REPORT_INNER_ERR_MSG("E19999", "update kernel args failed of %s.", name.c_str());
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }
  args_ = std::move(new_args);
  args_size_ = static_cast<uint32_t>(length);
  arg_base_  = PtrToPtr<uint8_t, uintptr_t>(&args_[static_cast<size_t>(offset_)]);
  args_ex_.args = args_.get();

  return SUCCESS;
}

Status AiCoreOpTask::UpdateArgs(TaskContext &task_context) {
  const int32_t input_output_ws_num = task_context.NumInputs() + task_context.NumOutputs() +
      task_context.NumWorkspaces();
  size_t expected_arg_count = static_cast<size_t>(input_output_ws_num) - output_indices_to_skip_.size();
  if (need_tiling_) {
    ++expected_arg_count;
  }

  if (shape_buffer_ != nullptr) {
    ++expected_arg_count;
  }

  if (need_overflow_addr_) {
    ++expected_arg_count;
  }

  if (expected_arg_count > max_arg_count_) {
    GELOGD("Need to reset size of args_ from %u to %zu.", max_arg_count_, expected_arg_count);
    const size_t host_mem_ext_size = need_tiling_ ? kMaxHostMemInputLen : 0U;
    const auto length = (expected_arg_count * sizeof(uintptr_t)) + offset_ + max_tiling_size_ + host_mem_ext_size;
    GE_CHK_STATUS_RET(ExpandArgsByLength(length, task_context.GetNodeItem().NodeName()),
                      "Expand args of %s failed.", task_context.GetNodeName());
    max_arg_count_ = static_cast<uint32_t>(expected_arg_count);
    args_ex_.argsSize = args_size_;
  }

  if (need_tiling_ && (expected_arg_count > tiling_data_idx_)) {
    const auto ptr_size = sizeof(uintptr_t);
    const uint32_t tiling_addr_index =
        need_overflow_addr_ ? (expected_arg_count - kOverflowAddrIndex) : static_cast<uint32_t>(expected_arg_count - 1);
    void *const arg_tiling_addr = ValueToPtr(PtrToValue(arg_base_) + (ptr_size * tiling_addr_index));
    void *const arg_tiling_data = ValueToPtr(PtrToValue(arg_base_) + (ptr_size * expected_arg_count));
    void *const arg_tiling_old = ValueToPtr(PtrToValue(arg_base_) + (ptr_size * tiling_data_idx_));

    if (memmove_s(arg_tiling_data, static_cast<size_t>(max_tiling_size_),
                  arg_tiling_old, static_cast<size_t>(max_tiling_size_)) != EOK) {
      GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Update][Args]failed of node %s.", task_context.GetNodeName());
      REPORT_INNER_ERR_MSG("E19999", "update kernel args failed of node %s.", task_context.GetNodeName());
      return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
    }
    // must after tiling data copy
    *(PtrToPtr<void, uintptr_t>(arg_tiling_addr)) = PtrToValue(arg_tiling_data);

    tiling_data_idx_ = expected_arg_count;
    args_ex_.tilingAddrOffset = static_cast<uint32_t>(tiling_addr_index * sizeof(void *));
    args_ex_.tilingDataOffset = static_cast<uint32_t>(expected_arg_count * sizeof(void *));
  }

  uint32_t index = 0U;
  GE_CHK_STATUS_RET_NOLOG(UpdateArgBase(task_context, index, expected_arg_count - 1U));

  if (task_context.IsTraceEnabled()) {
    for (uint32_t i = 0U; i < (index + 1U); ++i) {
      GELOGD("[%s] Arg[%u] = %lu", stub_name_.c_str(), i, arg_base_[i]);
    }
  }

  return SUCCESS;
}

Status AiCoreOpTask::LaunchKernel(const rtStream_t stream) {
  rtTaskCfgInfo_t cfg = {};
  cfg.localMemorySize = local_memory_size_;

  if (handle_ != nullptr) {
    GELOGD("AiCoreOpTask rtKernelLaunchWithHandleV2(tiling key = %lu, block dim = %u, local memory size = %u).",
           tiling_key_, block_dim_, local_memory_size_);
    SetTaskTag();
    PROFILING_START(-1, profiling::kRtKernelLaunch);
    GE_CHK_RT_RET(rtKernelLaunchWithHandleV2(handle_, tiling_key_, block_dim_, &args_ex_, nullptr, stream, &cfg));
    PROFILING_END(-1, profiling::kRtKernelLaunch);
  } else {
    GELOGD("AiCoreOpTask LaunchKernel(task = %s, block dim = %u, local memory size = %u).", stub_name_.c_str(),
           block_dim_, local_memory_size_);
    SetTaskTag();
    PROFILING_START(-1, profiling::kRtKernelLaunch);
    GE_CHK_RT_RET(rtKernelLaunchWithFlagV2(stub_func_, block_dim_, &args_ex_, nullptr, stream, 0U, &cfg));
    PROFILING_END(-1, profiling::kRtKernelLaunch);
  }
  GELOGI("[TASK_INFO] %lu/%s", log_id_, log_name_.c_str());
  return SUCCESS;
}

Status AiCoreOpTask::InitTilingInfo(const OpDesc &op_desc, bool is_updating) {
  (void)AttrUtils::GetBool(op_desc, kAttrSupportDynamicShape, is_dynamic_);
  if (!is_dynamic_) {
    GELOGD("[%s] Dynamic shape is not supported.", op_desc.GetName().c_str());
    return SUCCESS;
  }

  GELOGD("Start to get tiling data size of node %s.", op_desc.GetName().c_str());
  int64_t max_size = -1LL;
  (void)AttrUtils::GetInt(op_desc, GetKeyForOpParamSize(), max_size);
  GELOGD("Got op param size by key: %s, ret = %ld", GetKeyForOpParamSize().c_str(), max_size);
  if (max_size < 0) {
    GELOGE(PARAM_INVALID, "[Check][Size][%s(%s)] Invalid op_param_size: %ld.",
           op_desc.GetName().c_str(), op_desc.GetType().c_str(), max_size);
    REPORT_INNER_ERR_MSG("E19999", "[%s(%s)] Invalid op_param_size: %" PRId64 ".",
                       op_desc.GetName().c_str(), op_desc.GetType().c_str(), max_size);
    return PARAM_INVALID;
  }
  args_ex_.argsSize -= max_tiling_size_;
  uint32_t total_len = 0U;
  GE_ASSERT_TRUE(!ge::AddOverflow(static_cast<uint32_t>(max_size), sizeof(uintptr_t), total_len));
  const auto new_size = (total_len - 1U) / sizeof(uintptr_t) * sizeof(uintptr_t);
  const auto name = op_desc.GetName();
  if ((new_size > max_tiling_size_) && is_updating) {
    const auto length = args_size_ - max_tiling_size_ + new_size;
    GELOGD("Expand args size of %s from %u to %zu.", op_desc.GetName().c_str(), args_size_, length);
    GE_CHK_STATUS_RET(ExpandArgsByLength(length, name), "Expand args of %s failed.", name.c_str());
  }
  max_tiling_size_ = new_size;
  args_ex_.argsSize += max_tiling_size_;
  need_tiling_ = max_size > 0;
  return SUCCESS;
}

bool AiCoreOpTask::IsDynamicShapeSupported() const {
  return is_dynamic_;
}

const std::string &AiCoreOpTask::GetName() const {
  return stub_name_;
}

const std::string &AiCoreOpTask::GetOpType() const {
  return op_type_;
}

std::string AiCoreOpTask::GetKeyForOpParamSize() const {
  return kAttrOpParamSize;
}

std::string AiCoreOpTask::GetKeyForTbeKernel() const {
  return OP_EXTATTR_NAME_TBE_KERNEL;
}

std::string AiCoreOpTask::GetKeyForTvmMagic() const {
  return TVM_ATTR_NAME_MAGIC;
}

std::string AiCoreOpTask::GetKeyForTvmMetaData() const {
  return TVM_ATTR_NAME_METADATA;
}

std::string AiCoreOpTask::GetKeyForKernelName(const OpDesc &op_desc) const {
  return op_desc.GetName() + "_kernelname";
}

void AiCoreOpTask::SetTaskTag() const {
  const rtError_t rt_set_tag = rtSetTaskTag(op_name_.c_str());
  if (rt_set_tag != RT_ERROR_NONE) {
    GELOGW("[Call][rtSetTaskTag] failed, ret:0x%X", rt_set_tag);
  }
}

Status AtomicAddrCleanOpTask::Init(const NodePtr &node, const domi::TaskDef &task_def) {
  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  GE_CHK_STATUS_RET_NOLOG(AiCoreOpTask::DoInit(node, task_def));
  return InitAtomicAddrCleanIndices(*op_desc);
}

Status AtomicAddrCleanOpTask::InitAtomicAddrCleanIndices(const OpDesc &op_desc) {
  GELOGD("[%s] Start to setup AtomicAddrClean task.", op_desc.GetName().c_str());
  std::vector<int64_t> atomic_output_indices;
  (void)ge::AttrUtils::GetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);
  std::map<std::string, std::map<int64_t, int64_t>> workspace_info; // op_name, ws_index, ws_offset
  workspace_info = op_desc.TryGetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspace_info);
  if (atomic_output_indices.empty() && workspace_info.empty()) {
    GELOGE(INTERNAL_ERROR,
           "[Check][Size] [%s(%s)] Get %s and get %s failed. check invalid",
           op_desc.GetName().c_str(), op_desc.GetType().c_str(), ATOMIC_ATTR_OUTPUT_INDEX.c_str(),
           EXT_ATTR_ATOMIC_WORKSPACE_INFO.c_str());
    REPORT_INNER_ERR_MSG("E19999", "[%s(%s)] Get %s and get %s failed. check invalid",
                       op_desc.GetName().c_str(), op_desc.GetType().c_str(), ATOMIC_ATTR_OUTPUT_INDEX.c_str(),
                       EXT_ATTR_ATOMIC_WORKSPACE_INFO.c_str());
    return INTERNAL_ERROR;
  }

  for (const auto output_index : atomic_output_indices) {
    GELOGD("[%s] Adding output index [%ld]", op_desc.GetName().c_str(), output_index);
    GE_CHECK_GE(output_index, 0);
    GE_CHECK_LE(output_index, INT32_MAX);
    atomic_output_indices_.emplace_back(static_cast<int32_t>(output_index));
  }

  for (auto &iter : workspace_info) {
    for (auto &info_iter : iter.second) {
      const auto workspace_index = info_iter.first;
      GELOGD("[%s] Adding workspace index [%ld]", op_desc.GetName().c_str(), workspace_index);
      GE_CHECK_GE(workspace_index, 0);
      GE_CHECK_LE(workspace_index, INT32_MAX);
      atomic_workspace_indices_.emplace_back(static_cast<int32_t>(workspace_index));
    }
  }

  size_t arg_count = atomic_workspace_indices_.size() + atomic_output_indices_.size();
  if (need_tiling_) {
    arg_count += 1U;
  }
  if (need_overflow_addr_) {
    arg_count += 1U;
  }

  if (arg_count > max_arg_count_) {
    GELOGE(INTERNAL_ERROR, "[Check][arg_count][%s] Invalid arg memory, max arg count = %u,"
           "but expect = %zu", GetName().c_str(), max_arg_count_, arg_count);
    REPORT_INNER_ERR_MSG("E19999", "[%s] Invalid arg memory, max arg count = %u, but expect = %zu",
                       GetName().c_str(), max_arg_count_, arg_count);
    return INTERNAL_ERROR;
  }

  return SUCCESS;
}

std::string AtomicAddrCleanOpTask::GetKeyForOpParamSize() const {
  return kAttrAtomicOpParamSize;
}

std::string AtomicAddrCleanOpTask::GetKeyForTbeKernel() const {
  return EXT_ATTR_ATOMIC_TBE_KERNEL;
}

std::string AtomicAddrCleanOpTask::GetKeyForTvmMagic() const {
  return ATOMIC_ATTR_TVM_MAGIC;
}

std::string AtomicAddrCleanOpTask::GetKeyForTvmMetaData() const {
  return ATOMIC_ATTR_TVM_METADATA;
}

std::string AtomicAddrCleanOpTask::GetKeyForKernelName(const OpDesc &op_desc) const {
  return op_desc.GetName() + "_atomic_kernelname";
}

const std::string &AtomicAddrCleanOpTask::GetOpType() const {
  return kAtomicOpType;
}

Status AtomicAddrCleanOpTask::CalcTilingInfo(const NodePtr &node, const Operator &op) const {
  graphStatus ret;
  if (optiling::EnableAtomicRt2Tiling(node->GetOpDesc())) {
    GELOGD("[%s] Start to invoke OpAtomicCalculate on rt2.", node->GetName().c_str());
    ret = optiling::AtomicRtParseAndTiling(op, platform_infos_, *tiling_info_);
  } else {
    GELOGD("[%s] Start to invoke OpAtomicCalculate on rt1.", node->GetName().c_str());
    ret = optiling::OpAtomicCalculateV2(*node, *tiling_info_);
  }
  GE_CHK_STATUS_RET(ret,
                    "[Invoke][OpAtomicCalculate]Failed calc tiling data of node %s(%s).",
                    node->GetName().c_str(), node->GetType().c_str());
  GELOGD("[%s] Done invoking OpAtomicCalculate successfully.", node->GetName().c_str());
  return SUCCESS;
}

Status AtomicAddrCleanOpTask::UpdateArgs(TaskContext &task_context) {
  // refresh atomic output addr
  uint32_t index = 0U;
  for (const auto atomic_output_index : atomic_output_indices_) {
    const auto output_tensor = task_context.GetOutput(atomic_output_index);
    GE_CHECK_NOTNULL(output_tensor);
    arg_base_[index] = PtrToValue(output_tensor->GetData());
    index++;
  }

  // refresh atomic workspace addr
  for (const auto atomic_ws_index : atomic_workspace_indices_) {
    const auto workspace_tensor = task_context.MutableWorkspace(atomic_ws_index);
    GE_CHECK_NOTNULL(workspace_tensor);
    arg_base_[index] = PtrToValue(workspace_tensor);
    index++;
  }

  if (need_tiling_) {
    index++;
  }

  // Refresh atomic overflow detetcion addr
  if (need_overflow_addr_) {
    arg_base_[index] = PtrToValue(overflow_addr_);
  }

  if (task_context.IsTraceEnabled()) {
    for (uint32_t i = 0U; i < index; ++i) {
      GELOGD("[%s] Arg[%u] = %lu", GetName().c_str(), i, arg_base_[i]);
    }
  }

  return SUCCESS;
}

AttrNameOfBinOnOp AiCoreOpTask::GetAttrNamesForBin() const {
  return {OP_EXTATTR_NAME_TBE_KERNEL, TVM_ATTR_NAME_METADATA, "_kernelname", TVM_ATTR_NAME_MAGIC};
}

AttrNameOfBinOnOp AtomicAddrCleanOpTask::GetAttrNamesForBin() const {
  return {EXT_ATTR_ATOMIC_TBE_KERNEL, ATOMIC_ATTR_TVM_METADATA, "_atomic_kernelname", ATOMIC_ATTR_TVM_MAGIC};
}
}  // namespace hybrid
}  // namespace ge

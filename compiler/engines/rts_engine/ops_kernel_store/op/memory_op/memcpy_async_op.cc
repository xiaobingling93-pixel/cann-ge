/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "memcpy_async_op.h"

#include "op_factory.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/util/log.h"
#include "../../../inc/framework/common/runtime_model_ge.h"
#include "../acl_rt_memcpy_kind.h"
constexpr uint32_t MEMCPY_ASYNC_UNIT_SIZE = 64U * 1024U * 1024U;
constexpr uint64_t MAX_MEMCPY_SIZE_OF_D2D = 4ULL * 1024ULL * 1024ULL * 1024ULL;  // 4G

using namespace ge;
namespace cce {
namespace runtime {
MemcpyAsyncOp::MemcpyAsyncOp(const Node &node, RunContext &runContext) : Op(node, runContext) {}

Status MemcpyAsyncOp::Init() {
  RTS_LOGI("MemcpyAsyncOp Init start, node:%s.", name_.c_str());
  // Set input number、output number
  input_num_ = op_desc_->GetInputsSize();
  output_num_ = op_desc_->GetOutputsSize();
  RTS_LOGD("MemcpyAsyncOp input_num:%zu, output_num:%zu.", input_num_, output_num_);
  return Op::Init();
}

Status MemcpyAsyncOp::CheckPara() {
  if (v_output_data_addr_.size() != v_input_data_addr_.size() || v_output_data_addr_.size() != v_output_size_.size() ||
      v_input_data_addr_.size() != v_input_size_.size()) {
    RTS_REPORT_CALL_ERROR(
        "Run memcpy async op failed, input num(%zu) input data addr num(%zu) output num(%zu)"
        " output addr num(%zu) invalid, node:%s.",
        v_input_size_.size(), v_input_data_addr_.size(), v_output_size_.size(), v_output_data_addr_.size(),
        name_.c_str());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

graphStatus MemcpyAsyncOp::GetTensorSizeInBytesWithNoErrorOutput(const GeTensorDesc &desc, int64_t &size) {
  const Format format = desc.GetFormat();
  const DataType data_type = desc.GetDataType();
  int64_t output_mem_size = 0;

  bool is_no_tiling = false;
  (void)AttrUtils::GetBool(desc, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, is_no_tiling);
  graphStatus graph_status;
  if (is_no_tiling) {
    graph_status = TensorUtils::CalcTensorMemSizeForNoTiling(desc, format, data_type, output_mem_size);
  } else {
    graph_status = TensorUtils::CalcTensorMemSize(desc.GetShape(), format, data_type, output_mem_size);
  }
  if (graph_status != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  if (output_mem_size < 0) {
    return GRAPH_FAILED;
  }

  size = output_mem_size;
  return GRAPH_SUCCESS;
}

Status MemcpyAsyncOp::CheckInputSize(size_t index, int64_t &inputSize) {
  if (op_desc_ == nullptr) {
    RTS_LOGE("CheckInputSize failed, op_desc_ is null");
    return RT_FAILED;
  }
  auto inputDesc = op_desc_->GetInputDesc(index);
  if (GetTensorSizeInBytesWithNoErrorOutput(inputDesc, inputSize) != GRAPH_SUCCESS) {
    RTS_LOGW("Ignore tensor size for node %s", op_desc_->GetName().c_str());
    return RT_FAILED;
  }

  if (index >= v_output_size_.size()) {
    RTS_LOGE("CheckInputSize failed, index is Invalid, index=%zu, v_output_size_=%zu", index, v_output_size_.size());
    return RT_FAILED;
  }

  if (v_output_size_[index] < inputSize) {
    inputSize = v_output_size_[index];
  }
  if (inputSize == 0) {
    RTS_LOGW("memcpy size of input index:%zu is 0, skip generate task. node_name: %s", index, node_.GetName().c_str());
    return RT_FAILED;
  }
  return SUCCESS;
}

uint64_t MemcpyAsyncOp::CalculateMemcpyAsyncSingleMaxSize(const rtMemcpyKind_t kind) const {
  uint64_t sqSize = MEMCPY_ASYNC_UNIT_SIZE;
  if ((kind == RT_MEMCPY_DEVICE_TO_DEVICE) || (kind == RT_MEMCPY_ADDR_DEVICE_TO_DEVICE)) {
    sqSize = MAX_MEMCPY_SIZE_OF_D2D;
  }
  return sqSize;
}

Status MemcpyAsyncOp::Run(vector<TaskDef> &tasks) {
  if (dynamic_flag_) {
    return SUCCESS;
  }
  if (CheckPara() != SUCCESS) {
    return FAILED;
  }

  for (size_t index = 0; index < v_input_data_addr_.size(); index++) {
    int64_t inputSize = 0;
    if (CheckInputSize(index, inputSize) != SUCCESS) {
      continue;
    }
    RTS_LOGI("MemcpyAsyncOp run, index:%zu, outSize:%" PRId64 ", inSize:%" PRId64, index, v_output_size_[index],
             inputSize);

    rtMemcpyKind_t copyType = RT_MEMCPY_DEVICE_TO_DEVICE;
    int64_t kind;
    bool getAttrSucc = AttrUtils::GetInt(op_desc_, "_rt_memcpy_kind", kind);
    if (getAttrSucc) {
      copyType = static_cast<rtMemcpyKind_t>(kind);
      RTS_LOGI("Get memory copy type success, type = %d", copyType);
    }
    const uint64_t sqSize = CalculateMemcpyAsyncSingleMaxSize(copyType);
    copyType = (copyType == RT_MEMCPY_HOST_TO_DEVICE_EX) ? RT_MEMCPY_HOST_TO_DEVICE : copyType;
    copyType = (copyType == RT_MEMCPY_DEVICE_TO_HOST_EX) ? RT_MEMCPY_DEVICE_TO_HOST : copyType;
    uint64_t realSize = inputSize;
    uint64_t remainSize = inputSize;
    uint64_t doneSize = 0U;
    while (remainSize > 0U) {
      const uint64_t doingSize = (remainSize >= sqSize) ? sqSize : remainSize;
      realSize = doingSize;

      domi::TaskDef taskDef = {};
      taskDef.set_type(ACL_RT_MODEL_TASK_MEMCPY_ASYNC);
      taskDef.set_stream_id(op_desc_->GetStreamId());
      domi::MemcpyAsyncDef *memcpyDef = taskDef.mutable_memcpy_async();
      memcpyDef->set_op_index(static_cast<uint32_t>(op_desc_->GetId()));
      memcpyDef->set_dst(reinterpret_cast<uintptr_t>(v_output_data_addr_[index]) + doneSize);
      memcpyDef->set_dst_max(doingSize);
      memcpyDef->set_src(reinterpret_cast<uintptr_t>(v_input_data_addr_[index]) + doneSize);
      memcpyDef->set_count(doingSize);
      memcpyDef->set_kind(copyType);
      tasks.push_back(taskDef);
      doneSize += realSize;
      remainSize -= realSize;
    }
    auto retCode = UpdateOutDescFromInDesc(v_input_data_addr_[index], v_output_data_addr_[index], tasks);
    if (retCode != SUCCESS) {
      RTS_LOGE("Update output desc failed, ret=%d, index:%zu.", retCode, index);
      return retCode;
    }
  }

  return SUCCESS;
}

void MemcpyAsyncOp::ConstructSdmaSqeHeader(rt_fftsplus_memcpy_async_op_sqe_header_t *sdma_header) {
  sdma_header->opcode = 0U;  // RT_MEMCPY_DEVICE_TO_DEVICE
  sdma_header->ie = 0U;
  sdma_header->sssv = 1U;
  sdma_header->dssv = 1U;
  sdma_header->sns = 1U;
  sdma_header->dns = 1U;
  sdma_header->qos = 0U;
  sdma_header->sro = 0U;
  sdma_header->dro = 0U;
  sdma_header->partid = 0U;
  sdma_header->mpamns = 0U;
  sdma_header->pmg = 0U;
  sdma_header->format = 0U;
}

Status MemcpyAsyncOp::FillContextInfo(const Node &node, size_t index) {
  int64_t inputSize = 0;
  if (!dynamic_flag_ && CheckInputSize(index, inputSize) != SUCCESS) {
    return FAILED;
  }
  std::shared_ptr<domi::FftsPlusCtxDef> fftsContext = nullptr;
  try {
    fftsContext = std::make_shared<domi::FftsPlusCtxDef>();
  } catch (...) {
    RTS_REPORT_INNER_ERROR("Failed to create a context information.");
    return FAILED;
  }
  fftsContext->set_op_index(node.GetOpDesc()->GetId());
  fftsContext->set_context_type(RT_CTX_TYPE_SDMA);
  domi::FftsPlusSdmaCtxDef *sdma_ctx = fftsContext->mutable_sdma_ctx();

  uint32_t sdma_header_value = 0U;
  rt_fftsplus_memcpy_async_op_sqe_header_t *sdma_header =
      reinterpret_cast<rt_fftsplus_memcpy_async_op_sqe_header_t *>(&sdma_header_value);
  ConstructSdmaSqeHeader(sdma_header);

  sdma_ctx->set_atm(0U);
  sdma_ctx->set_ns(1U);
  sdma_ctx->set_thread_id(0U);
  sdma_ctx->set_thread_dim(1U);
  sdma_ctx->set_sdma_sqe_header(sdma_header_value);
  sdma_ctx->set_src_stream_id(0U);
  sdma_ctx->set_src_sub_stream_id(0U);
  sdma_ctx->set_dst_stream_id(0U);
  sdma_ctx->set_dst_sub_stream_id(0U);
  sdma_ctx->set_src_addr_offset(0U);
  sdma_ctx->set_dst_addr_offset(0U);
  if (dynamic_flag_) {
    sdma_ctx->set_src_addr_base(0UL);
    sdma_ctx->set_dst_addr_base(0UL);
    sdma_ctx->set_non_tail_data_len(0);
    sdma_ctx->set_tail_data_len(0);
  } else {
    inputSize = 0;
    if (CheckInputSize(index, inputSize) != SUCCESS) {
      return FAILED;
    }
    sdma_ctx->set_src_addr_base(reinterpret_cast<uintptr_t>(v_input_data_addr_[index]));
    sdma_ctx->set_dst_addr_base(reinterpret_cast<uintptr_t>(v_output_data_addr_[index]));
    sdma_ctx->set_non_tail_data_len(static_cast<uint64_t>(inputSize));
    sdma_ctx->set_tail_data_len(static_cast<uint64_t>(inputSize));
  }

  node.GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", fftsContext);

  return SUCCESS;
}

Status MemcpyAsyncOp::GenerateCtxDef(const Node &node) {
  if (node.GetOpDesc() == nullptr) {
    RTS_REPORT_CALL_ERROR("Create op failed, param can not be NULL.");
    return FAILED;
  }
  if (dynamic_flag_) {
    return FillContextInfo(node, 0);
  }
  Status ret = CheckPara();
  if (ret != SUCCESS) {
    return ret;
  }

  if (v_input_data_addr_.size() > 1) {
    RTS_LOGW("Only support 1 SDMA context for 1 node now.");
  }

  for (size_t index = 0; index < v_input_data_addr_.size(); index++) {
    ret = FillContextInfo(node, index);
    if (ret != SUCCESS) {
      return FAILED;
    }
  }
  RTS_LOGI("Generate FFTSPlus context for MemcpyAsyncOp success, node name %s", name_.c_str());
  return SUCCESS;
}

}  // namespace runtime
}  // namespace cce

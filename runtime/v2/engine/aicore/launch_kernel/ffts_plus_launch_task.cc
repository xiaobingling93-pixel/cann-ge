/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include <iomanip>
#include "adump_pub.h"
#include "adump_api.h"
#include "runtime/rt_ffts_plus.h"
#include "graph/ge_error_codes.h"
#include "common/runtime_api_wrapper.h"
#include "register/kernel_registry_impl.h"
#include "register/ffts_node_calculater_registry.h"
#include "kernel/memory/mem_block.h"
#include "kernel/memory/multi_stream_mem_block.h"
#include "common/checker.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "engine/aicore/fe_rt2_common.h"
#include "aprof_pub.h"

namespace gert {
namespace kernel {
const size_t kContextLen = 128UL;
const size_t kDescBufLen = 32UL;
const int32_t kHexDumpWidth = 8;
void DumpFftsPlusTask(void *const desc_buf, const size_t desc_buf_len) {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    return;
  }
  GELOGD("========Dump FftsPlusTask-begin-context, descBufLen =%" PRIu64 "========", desc_buf_len);
  std::stringstream ss;
  for (size_t i = 0UL; i < (desc_buf_len / kContextLen); ++i) {
    uint32_t *const buf = static_cast<uint32_t *>(desc_buf) + (i * kDescBufLen);
    for (size_t j = 0UL; j < kDescBufLen; ++j) {
      if (buf[j] > 0U) {
        ss << "idx:[" << std::dec << j << "]=[0x" << std::setfill('0') << std::setw(kHexDumpWidth) << std::hex
           << buf[j] << "]";
      }
    }
    GELOGD("Dump FftsPlusTask-The %zu context: [%s]", i, ss.str().c_str());
    ss.clear();
    ss.str("");
  }
  GELOGD("========Dump FftsPlusTask-end-context========");
}

ge::graphStatus DfxPrintProc(KernelContext *context, rtStream_t stream) {
  auto exe_arg = context->GetInputPointer<gert::DfxExeArg>(static_cast<size_t>(3U));
  FE_ASSERT_NOTNULL(exe_arg);
  if (exe_arg->need_print) {
    auto workspace = context->GetInputPointer<ContinuousVector>(static_cast<size_t>(4U));
    FE_ASSERT_NOTNULL(workspace);
    if (workspace->GetSize() == 0) {
      return ge::GRAPH_SUCCESS;
    }
    auto work_addr = reinterpret_cast<GertTensorData *const *>(workspace->GetData());
    const auto compute_node_info = static_cast<const ComputeNodeInfo *>(context->GetComputeNodeExtend());
    FE_ASSERT_NOTNULL(compute_node_info);
    Adx::AdumpPrintWorkSpace(work_addr[0]->GetAddr(), exe_arg->buffer_size, stream, compute_node_info->GetNodeType());
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFTSPlusTaskLaunch(KernelContext *context) {
  (void)context;
  GELOGW("[Launch_kernel][FFTSPlusTaskLaunch]Not used interface call.");
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(LaunchFFTSPlusTask).RunFunc(FFTSPlusTaskLaunch);

ge::graphStatus LaunchFFTSPlusTaskNoCopy(KernelContext *context) {
  auto need_launch = context->GetInputValue<uint32_t>(static_cast<size_t>(2));
  if (need_launch == 0U) {
    GELOGD("No need to launch this task.");
    return ge::GRAPH_SUCCESS;
  }
  auto stream = context->GetInputValue<rtStream_t>(static_cast<size_t>(0));
  auto task_info_para = context->GetInputValue<NodeMemPara*>(static_cast<size_t>(1));
  FE_ASSERT_NOTNULL(task_info_para);
  auto host_task_info = reinterpret_cast<TransTaskInfo*>(task_info_para->host_addr);
  FE_ASSERT_NOTNULL(host_task_info);
  auto dev_task_info = reinterpret_cast<TransTaskInfo*>(task_info_para->dev_addr);
  FE_ASSERT_NOTNULL(dev_task_info);
  size_t buf_offset = host_task_info->offsets[static_cast<size_t>(InfoStType::kDescBuf)];
  void *ori_desc_buf = const_cast<void *>(host_task_info->rt_task_info.descBuf);
  FE_ASSERT_NOTNULL(ori_desc_buf);
  DumpFftsPlusTask(ori_desc_buf, host_task_info->rt_task_info.descBufLen);
  host_task_info->rt_task_info.descBuf = &dev_task_info->args[buf_offset];
  GELOGD("Desc buff base: %lx offset: %zu, desc_buf_len: %zu.", host_task_info->rt_task_info.descBuf, buf_offset,
         host_task_info->rt_task_info.descBufLen);
  host_task_info->rt_task_info.descAddrType = RT_FFTS_PLUS_CTX_DESC_ADDR_TYPE_DEVICE;
  // need set descBufAddrType with 1 to tell rts not h2d desc buf
  const uint32_t dump_flag = (host_task_info->rt_task_info.fftsPlusDumpInfo.loadDumpInfo == nullptr ?
    RT_KERNEL_DEFAULT : RT_KERNEL_FFTSPLUS_DYNAMIC_SHAPE_DUMPFLAG);
  GE_ASSERT_EQ(ge::rtFftsPlusTaskLaunchWithFlag(&host_task_info->rt_task_info, stream, dump_flag), RT_ERROR_NONE);
  host_task_info->rt_task_info.descBuf = ori_desc_buf;
  GELOGD("Update descbuf to origin: %lx.", ori_desc_buf);
  return DfxPrintProc(context, stream);
}
REGISTER_KERNEL(LaunchFFTSPlusTaskNoCopy).RunFunc(LaunchFFTSPlusTaskNoCopy);
}  // namespace kernel
}  // namespace gert

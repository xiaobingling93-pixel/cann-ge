/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "file_constant_kernel.h"
#include <fstream>
#include "securec.h"
#include "graph/ge_error_codes.h"
#include "graph/def_types.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "framework/runtime/rt_session.h"
#include "register/kernel_registry.h"
#include "kernel/tensor_attr.h"
#include "common/checker.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "common/file_constant_utils/file_constant_utils.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "kernel/kernel_log.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "core/debug/kernel_tracing.h"
#include "core/utils/tensor_utils.h"
#include "hybrid/model/hybrid_model_builder.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"

namespace gert {
namespace kernel {
namespace {
constexpr size_t kOffsetAttrIndex = 4U;
constexpr size_t kLengthAttrIndex = 5U;
constexpr size_t kLocationAttrIndex = 6U;
constexpr int64_t kBlockSize = 10485760;

struct FileConstantInfo {
  std::string file_path;
  size_t file_offset;
  size_t file_length;

  std::string DebugString() const {
    std::stringstream ss;
    ss << "Get file constant info from private attr. File path: " << file_path << ", file offset=" << file_offset
       << ", file length=" << file_length << std::endl;
    return ss.str();
  }
};

ge::graphStatus CopyWeightFromFileAsync(const void *const curr_dev_ptr, const FileConstantInfo &file_constant_info,
                                        size_t &left_size, rtStream_t stream) {
  ge::graphStatus ret = ge::GRAPH_SUCCESS;
  GE_ASSERT_TRUE(left_size >= file_constant_info.file_length);
  const std::string real_path = ge::RealPath(file_constant_info.file_path.c_str());
  std::ifstream ifs(real_path, std::ifstream::binary);
  GE_ASSERT_TRUE(ifs.is_open(), "[Open][File] %s failed.", file_constant_info.file_path.c_str());
  ifs.clear();
  (void)ifs.seekg(static_cast<int64_t>(file_constant_info.file_offset), std::ifstream::beg);
  size_t used_memory = 0U;
  std::string compress_nodes;
  compress_nodes.reserve(static_cast<size_t>(kBlockSize));

  while ((!ifs.eof()) && (used_memory < file_constant_info.file_length)) {
    (void)ifs.read(&compress_nodes[0U], kBlockSize);
    auto copy_len_once = static_cast<size_t>(ifs.gcount());
    if ((file_constant_info.file_length - used_memory) < copy_len_once) {
      copy_len_once = file_constant_info.file_length - used_memory;
    }
    GELOGI("copy %zu bytes to memory.", copy_len_once);
    void *const cur_dev_ptr = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(curr_dev_ptr) + used_memory);
    const rtError_t rts_error = rtMemcpyAsync(cur_dev_ptr, left_size - used_memory, &compress_nodes[0U], copy_len_once,
                                              RT_MEMCPY_HOST_TO_DEVICE_EX, stream);
    if (rts_error != RT_ERROR_NONE) {
      GELOGE(ge::GRAPH_FAILED, "copy failed, result code = %d.", rts_error);
      ret = RT_ERROR_TO_GE_STATUS(rts_error);
      break;
    }
    used_memory += copy_len_once;
  }
  ifs.close();
  left_size -= used_memory;
  GELOGI("used memory is %zu.", used_memory);
  return ret;
}

ge::graphStatus AllocHbmMemForFileConstant(const size_t tensor_size, KernelContext *context) {
  auto gert_allocator = context->GetInputValue<gert::GertAllocator *>(
      static_cast<size_t>(FileConstantKernelInputIdx::kAllocatorIdx));
  KERNEL_CHECK_NOTNULL(gert_allocator);
  auto tensor_data =
      context->GetOutputPointer<GertTensorData>(static_cast<size_t>(FileConstantKernelOutputIdx::kOutAddrIdx));
  KERNEL_CHECK_NOTNULL(tensor_data);
  auto mem_block = reinterpret_cast<memory::MultiStreamMemBlock *>(gert_allocator->Malloc(tensor_size));
  KERNEL_CHECK_NOTNULL(mem_block);
  KERNEL_CHECK(mem_block->GetAddr() != nullptr, "malloc failed, tensor size=%zu", tensor_size);
  *tensor_data = TensorUtils::ToGertTensorData(
      mem_block, gert_allocator->GetPlacement(), gert_allocator->GetStreamId());
  KERNEL_TRACE(TRACE_STR_ALLOC_MEM", tensor size %zu", gert_allocator->GetStreamId(), mem_block, mem_block->GetAddr(),
               mem_block->GetSize(), tensor_size);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetFileConstantInfoFromPrivateAttr(KernelContext *context, FileConstantInfo &file_constant_info) {
  const auto file_constant_weight_dir_holder = context->GetInputPointer<ge::char_t *>(
      static_cast<size_t>(FileConstantKernelInputIdx::kFileConstantWeightDirIdx));
  GE_ASSERT_NOTNULL(file_constant_weight_dir_holder);
  std::string file_constant_weight_dir(*file_constant_weight_dir_holder);
  GELOGD("Get file constant weight dir[%s].", file_constant_weight_dir.c_str());

  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  GE_ASSERT_NOTNULL(compute_node_info);
  auto attrs = compute_node_info->GetAttrs();
  auto path = attrs->GetAttrPointer<char>(kLocationAttrIndex);
  GE_ASSERT_NOTNULL(path);
  std::string location(path);
  if (location.empty()) {
    GE_ASSERT_TRUE(file_constant_weight_dir.empty(),
                   "File constant weight dir[%s] exists, while file name from location is empty.",
                   file_constant_weight_dir.c_str());
    return ge::GRAPH_SUCCESS;
  }
  const int64_t *const attr_offset = attrs->GetAttrPointer<int64_t>(kOffsetAttrIndex);
  GE_ASSERT_NOTNULL(attr_offset, "%s offset does not exist, ", extend_context->GetKernelName());
  file_constant_info.file_offset = static_cast<size_t>(*attr_offset);
  const int64_t *const attr_length = attrs->GetAttrPointer<int64_t>(kLengthAttrIndex);
  GE_ASSERT_NOTNULL(attr_length, "%s length does not exist, ", extend_context->GetKernelName());
  file_constant_info.file_length = static_cast<size_t>(*attr_length);
  file_constant_info.file_path =
      (file_constant_weight_dir.empty()) ? location : file_constant_weight_dir.append(location);
  GELOGD("%s", file_constant_info.DebugString().c_str());
  return ge::GRAPH_SUCCESS;
}
}  // namespace

ge::graphStatus CreateFileConstantOutput(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto tensor_data_chain = context->GetOutput(static_cast<size_t>(FileConstantKernelOutputIdx::kOutAddrIdx));
  GE_ASSERT_NOTNULL(tensor_data_chain);
  auto td = new (std::nothrow) GertTensorData();
  GE_ASSERT_NOTNULL(td);
  tensor_data_chain->SetWithDefaultDeleter(td);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CreateFileConstantUserMemOutput(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto tensor_data_chain = context->GetOutput(0U);
  GE_ASSERT_NOTNULL(tensor_data_chain);
  auto td = new (std::nothrow) GertTensorData();
  GE_ASSERT_NOTNULL(td);
  tensor_data_chain->SetWithDefaultDeleter(td);
  return ge::GRAPH_SUCCESS;
}

// 外置权重文件路径优先级
// 1. 私有属性location(走这个分支的话，3个私有属性缺一不可)  场景:onnx传递下来的 || 开启权重外置
// 2. ir属性file_path  用户构图时设置
// 3. 使用file_id属性从GEContext的option map中索引  file_id离线场景才用（ATC场景，不开启权重外置）
ge::graphStatus FileConstantKernel(KernelContext *context) {
  const auto rt_session =
      context->GetInputValue<RtSession *>(static_cast<size_t>(FileConstantKernelInputIdx::kSessionIdx));
  GE_ASSERT_NOTNULL(rt_session);
  const auto var_mgr = rt_session->GetVarManager();
  if (var_mgr != nullptr) {
    GELOGD("Get fileconstant output addr from var manager.");
    const auto var_id =
        context->GetInputValue<ge::char_t *>(static_cast<size_t>(FileConstantKernelInputIdx::kVarIdIdx));
    GE_ASSERT_NOTNULL(var_id);
    TensorData tmp_td;
    StorageShape shape;
    GE_ASSERT_SUCCESS(var_mgr->GetVarShapeAndMemory(var_id, shape, tmp_td),
                      "Var manager %p get variable '%s' shape and memory failed", var_mgr, var_id);
    auto gtd =
        context->GetOutputPointer<GertTensorData>(static_cast<size_t>(FileConstantKernelOutputIdx::kOutAddrIdx));
    GE_ASSERT_NOTNULL(gtd);
    auto gert_allocator =
        context->GetInputValue<GertAllocator *>(static_cast<size_t>(FileConstantKernelInputIdx::kAllocatorIdx));
    GE_ASSERT_NOTNULL(gert_allocator);
    GE_ASSERT_SUCCESS(TensorUtils::ShareTdToGtd(tmp_td, *gert_allocator, *gtd));
  } else {
    // 离线场景路径拼接
    GELOGD("Get fileconstant output addr by allocator malloc.");
    FileConstantInfo file_constant_info{"", 0U, 0U};
    GE_ASSERT_GRAPH_SUCCESS(GetFileConstantInfoFromPrivateAttr(context, file_constant_info));
    if (file_constant_info.file_path.empty()) {
      const auto file_name_holer =
          context->GetInputPointer<ge::char_t *>(static_cast<size_t>(FileConstantKernelInputIdx::kFileNameIdx));
      GE_ASSERT_NOTNULL(file_name_holer);
      file_constant_info.file_path = std::string(*file_name_holer);
      GELOGD("Get file constant weight path %s.", file_constant_info.file_path.c_str());
    }
    auto left_size = context->GetInputValue<size_t>(static_cast<size_t>(FileConstantKernelInputIdx::kOutputSizeIdx));
    file_constant_info.file_length =
        (file_constant_info.file_length == 0U ? left_size : file_constant_info.file_length);

    // 离线场景分配权重内存并异步加载
    GE_ASSERT_GRAPH_SUCCESS(AllocHbmMemForFileConstant(left_size, context));
    auto rt_stream = context->GetInputValue<rtStream_t>(static_cast<size_t>(FileConstantKernelInputIdx::kStreamIdx));
    GE_ASSERT_NOTNULL(rt_stream);
    auto tensor_data =
        context->GetOutputPointer<GertTensorData>(static_cast<size_t>(FileConstantKernelOutputIdx::kOutAddrIdx));
    GE_ASSERT_NOTNULL(tensor_data);
    auto output_addr = tensor_data->GetAddr();
    GE_ASSERT_GRAPH_SUCCESS(CopyWeightFromFileAsync(output_addr, file_constant_info, left_size, rt_stream),
                            "Failed to copy data to file constant.");
  }
  return ge::GRAPH_SUCCESS;
}

// user set FileConstant device memory via aclmdlSetExternalWeightAddress
static ge::graphStatus FileConstantUserMemKernel(KernelContext *context) {
  const auto user_mem =
      context->GetInputValue<void *>(static_cast<size_t>(FileConstantUserMemKernelInput::kUserMem));
  GE_ASSERT_NOTNULL(user_mem);
  const auto mem_size =
      context->GetInputValue<size_t>(static_cast<size_t>(FileConstantUserMemKernelInput::kUserMemSize));

  auto tensor_data = context->GetOutputPointer<GertTensorData>(0U);
  GE_ASSERT_NOTNULL(tensor_data);
  tensor_data->SetPlacement(TensorPlacement::kOnDeviceHbm);
  tensor_data->MutableTensorData().SetAddr(user_mem, nullptr);
  tensor_data->MutableTensorData().SetSize(mem_size);
  KERNEL_TRACE("use user addr: %p, size: %zu", user_mem, mem_size);
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(FileConstantKernel)
    .RunFunc(FileConstantKernel)
    .OutputsCreator(CreateFileConstantOutput)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);

REGISTER_KERNEL(FileConstantUserMemKernel)
    .RunFunc(FileConstantUserMemKernel)
    .OutputsCreator(CreateFileConstantUserMemOutput)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);
}  // namespace kernel
}  // namespace gert

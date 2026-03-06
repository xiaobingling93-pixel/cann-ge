/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "args_format_utils.h"

#include "common/op_tiling/op_tiling_rt2.h"
#include "graph/ge_local_context.h"
#include "register/hidden_inputs_func_registry.h"
#include "graph/args_format_desc.h"
#include "exe_graph/lowering/bg_kernel_context_extend.h"

namespace ge {
constexpr char_t const *kHcomHiddenInputs = "_rt_resource_list";
constexpr char_t const *kResourceTypes = "_rt_resource_type";
constexpr size_t kMaxTilingDataSize = 16UL * 1024UL;
constexpr size_t kMaxWorkspaceCount = 16UL;
constexpr char_t const *kMaxTilingSize = "op_para_size";

enum StreamType : int64_t {
  kDefaultStream = 0,
  kNeedActiveByOther,
  kEnd,
};

Status ArgsFormatUtils::GetHcomHiddenInputs(const OpDescPtr &op_desc, const DavinciModel &davinci_model,
                                            std::vector<void *> &hidden_addrs, const HiddenInputsType hi_type) {
  GE_ASSERT_NOTNULL(op_desc);
  shared_ptr<std::vector<void *>> rt_resource_list = MakeShared<std::vector<void *>>();
  GE_CHECK_NOTNULL(rt_resource_list);
  const std::vector<rtStream_t> &stream_list = davinci_model.GetStreamList();
  const auto stream_ids = op_desc->GetAttachedStreamIds();
  if (!stream_ids.empty()) {
    GE_ASSERT_TRUE(stream_ids.size() == 1U, "Currently only support one attached stream id but got %zu, node %s",
                   stream_ids.size(), op_desc->GetName().c_str());
    const size_t stream_id = static_cast<size_t>(stream_ids[0U]);
    GE_ASSERT(stream_id < stream_list.size(), "attached stream_id [%zu]is invalid.", stream_id);
    rt_resource_list->push_back(stream_list[stream_id]);
    GE_ASSERT_TRUE(op_desc->SetExtAttr(kHcomHiddenInputs, rt_resource_list));
    int64_t stream_type = kDefaultStream;
    if (davinci_model.IsLogicStreamActiveByOthers(static_cast<uint32_t>(stream_id))) {
      stream_type = kNeedActiveByOther;
    }
    shared_ptr<std::vector<int64_t>> resource_type_list = MakeShared<std::vector<int64_t>>();
    GE_CHECK_NOTNULL(resource_type_list);
    resource_type_list->push_back(stream_type);
    GE_ASSERT_TRUE(op_desc->SetExtAttr(kResourceTypes, resource_type_list));
    GE_ASSERT_EQ(rt_resource_list->size(), resource_type_list->size());
    GELOGI("Op %s %s set attached stream %zu stream handle %p and stream type %" PRId64 " successfully.",
      op_desc->GetNamePtr(), op_desc->GetTypePtr(), stream_id, stream_list[stream_id], stream_type);
  }
  GELOGI("start to to call op %s %s hidden func", op_desc->GetNamePtr(), op_desc->GetTypePtr());
  const auto hiddens_func = HiddenInputsFuncRegistry::GetInstance().FindHiddenInputsFunc(hi_type);
  GE_ASSERT_NOTNULL(hiddens_func, "Hidden func for type HCOM is not found, op:[%s].", op_desc->GetNamePtr());
  GE_ASSERT_SUCCESS(hiddens_func(op_desc, hidden_addrs), "call %s hidden inputs func fail", op_desc->GetNamePtr());
  return SUCCESS;
}

Status ArgsFormatUtils::GetTileFwkHiddenInputs(const OpDescPtr &op_desc, const DavinciModel &davinci_model,
                                               std::vector<void *> &hidden_addrs, const HiddenInputsType hi_type) {
  (void)davinci_model;
  const auto hiddens_func = HiddenInputsFuncRegistry::GetInstance().FindHiddenInputsFunc(hi_type);
  GE_ASSERT_NOTNULL(hiddens_func, "Hidden func for type TILEFWK is not found, op:[%s].", op_desc->GetNamePtr());
  return hiddens_func(op_desc, hidden_addrs);
}

//  |--tiling_data--|--workspace--|--tiling_context--|
Status ArgsFormatUtils::SinkTilingContext(const NodePtr &node, DavinciModel &davinci_model,
                                          std::map<size_t, gert::AddrRefreshedTensor> &index_to_tensor,
                                          void *platform_infos_addr, const bool is_args_exception_enable,
                                          const uint64_t atomic_index) {
  size_t total_plain_size{0UL};

  const size_t device_tiling_size = gert::DeviceTilingContextBuilder::CalcTotalTiledSize(node->GetOpDesc());
  const auto aligned_tiling_context_size = ge::RoundUp(static_cast<uint64_t>(device_tiling_size), sizeof(uintptr_t));
  total_plain_size += aligned_tiling_context_size;

  // calc tiling size
  int64_t max_size = -1;
  if (!ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), kMaxTilingSize, max_size) || max_size < 0) {
    GELOGI("No max tiling size in opdesc.");
    max_size = static_cast<int64_t>(kMaxTilingDataSize);
  }
  const size_t aligned_max_tiling_size = ge::RoundUp(static_cast<uint64_t>(max_size), sizeof(uintptr_t));
  GELOGI("Aligned max tiling size: %zu, max tiling size: %zu.", aligned_max_tiling_size, max_size);

  const auto aligned_tiling_size = aligned_max_tiling_size + sizeof(gert::TilingData);
  total_plain_size += aligned_tiling_size;

  // calc workspace size
  const size_t workspace_addr_size = kMaxWorkspaceCount * sizeof(size_t) + sizeof(gert::ContinuousVector);
  total_plain_size += workspace_addr_size;

  // compute node info
  size_t compute_node_info_size{0UL};
  gert::bg::BufferPool buffer_pool;
  auto compute_node_extend_holder = gert::bg::CreateComputeNodeInfo(node, buffer_pool, compute_node_info_size);
  GE_ASSERT_NOTNULL(compute_node_extend_holder);
  total_plain_size += compute_node_info_size;

  auto host_pointer = ge::MakeUnique<uint8_t[]>(total_plain_size);
  GE_ASSERT_NOTNULL(host_pointer);
  void *device_addr = davinci_model.MallocDynamicMemory(total_plain_size, RT_MEMORY_TS);
  GE_ASSERT_NOTNULL(device_addr);
  // copy tiling data
  const auto tiling_holder = gert::TilingData::CreateCap(aligned_max_tiling_size);
  GE_ASSERT_NOTNULL(tiling_holder);
  gert::TilingData *host_data = reinterpret_cast<gert::TilingData *>(tiling_holder.get());
  GE_ASSERT_NOTNULL(host_data);
  host_data->Init(aligned_max_tiling_size, ValueToPtr(PtrToValue(device_addr) + sizeof(gert::TilingData)));

  // 添加atomic index
  if (is_args_exception_enable && (aligned_max_tiling_size >= sizeof(uint64_t))) {
    uint64_t *atomic_index_data = static_cast<uint64_t *>
      (ValueToPtr(PtrToValue(host_data) + aligned_tiling_size - sizeof(uint64_t)));
    *atomic_index_data = atomic_index;
    GELOGI("aligned tiling size with cap: %zu, atomic index offset in tiling data: %zu, atomic index: %" PRIu64 ".",
      aligned_tiling_size, aligned_max_tiling_size - sizeof(uint64_t), atomic_index);
  }

  GE_ASSERT_EOK(memcpy_s(host_pointer.get(), aligned_tiling_size, tiling_holder.get(), aligned_tiling_size));

  // copy workspace
  const auto workspace_holder = gert::ContinuousVector::Create<size_t>(kMaxWorkspaceCount);
  GE_ASSERT_NOTNULL(workspace_holder);
  GE_ASSERT_EOK(
      memcpy_s(&host_pointer[aligned_tiling_size], workspace_addr_size, workspace_holder.get(), workspace_addr_size));

  // set begin addr
  uint8_t *context_host_begin = &host_pointer[aligned_tiling_size + workspace_addr_size];
  uint64_t context_dev_begin = PtrToValue(device_addr) + aligned_tiling_size + workspace_addr_size;

  std::string deterministic_str;
  (void)ge::GetThreadLocalContext().GetOption(ge::DETERMINISTIC, deterministic_str);
  int32_t deterministic = deterministic_str == "1" ? 1 : 0;
  gert::TiledKernelContextHolder tiling_context_holder;
  tiling_context_holder.compute_node_info_size_ = compute_node_info_size;
  tiling_context_holder.host_compute_node_info_ = compute_node_extend_holder.get();
  int32_t deterministic_level = 0;
  GE_ASSERT_SUCCESS(optiling::GetDeterministicLevel(deterministic_level));

  auto context_builder = gert::DeviceTilingContextBuilder();
  Status ret =
      context_builder.PlatformInfo(platform_infos_addr)
          .TilingData(device_addr)
          .Deterministic(deterministic)
          .DeterministicLevel(deterministic_level)
          .Workspace(ValueToPtr(PtrToValue(device_addr) + aligned_tiling_size))
          .AddrRefreshedInputTensor(index_to_tensor)
          .TiledHolder(context_host_begin, context_dev_begin, aligned_tiling_context_size + compute_node_info_size)
          .Build(node, tiling_context_holder);
  if (ret != ge::SUCCESS) {
    GELOGE(FAILED, "Failed to build device tiling context.");
    return FAILED;
  }

  // H2D
  GE_CHK_RT_RET(aclrtMemcpy(device_addr, total_plain_size, host_pointer.get(),
      total_plain_size, ACL_MEMCPY_HOST_TO_DEVICE));

  std::shared_ptr<TilingContextAddr> tiling_context_addr = MakeShared<TilingContextAddr>();
  tiling_context_addr->tiling_context_addr = tiling_context_holder.dev_context_addr_;
  tiling_context_addr->op_type_addr = tiling_context_holder.dev_op_type_addr_;
  GE_ASSERT_TRUE(tiling_context_holder.output_addrs_.size() >=
                 static_cast<size_t>(gert::TilingContext::TilingOutputIndex::kOutputNum));
  tiling_context_addr->tiling_key_addr =
      tiling_context_holder.output_addrs_[gert::TilingContext::TilingOutputIndex::kOutputTilingKey];
  tiling_context_addr->block_dim_addr =
      tiling_context_holder.output_addrs_[gert::TilingContext::TilingOutputIndex::kOutputBlockDim];
  tiling_context_addr->tiling_data_addr = PtrToValue(device_addr) + sizeof(gert::TilingData);
  GE_ASSERT_TRUE(node->GetOpDescBarePtr()->SetExtAttr(kTilingContextAddrs, tiling_context_addr));
  return SUCCESS;
}
}  // namespace ge

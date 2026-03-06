/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_task_struct.h"
#include "formats/utils/formats_trans_utils.h"
#include "framework/common/tlv/tlv.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/node_utils.h"
#include "runtime/rt.h"
#include "aicpu_resources.h"

namespace ge {
namespace {
constexpr size_t kNameMaxLength = 128U;
const std::string kResourceTypeQueue = "RES_QUEUE";
const std::string kResourceTypeChannel = "RES_CHANNEL";
const std::string kResourceTypeVdecChannel = "RES_VDEC_CHANNEL";
const std::string kKernelNameCreateChannel = "CreateChannel";
const std::string kKernelNameDestroyChannel = "DestroyChannel";
const std::string kKernelNameCreateVdecChannel = "CreateVdecChannel";
const std::string kKernelNameDestroyVdecChannel = "DestroyVdecChannel";
const std::string kKernelNameCreateQueue = "CreateQueue";
const std::string kKernelNameDestroyQueue = "DestroyQueue";
const std::string kAttrFieldQueueName = "queue_name";
const std::string kAttrFieldQueueDepth = "queue_depth";
const std::string kAttrFieldQueueIdIdx = "queue_id_idx";
const std::string kSoNameBuiltin = "libbuiltin_kernels.so";
const std::string kKernelNameModelConfig = "AicpuModelConfig";
const std::string kKernelNameModelShapeConfig = "AicpuModelShapeConfig";
constexpr int32_t kDefaultPriority = 0;
constexpr uint32_t kKernelBlockDim = 1U;
constexpr uint32_t kAiCpuQueueDepth = 8U;

struct TlvBuf {
  uint32_t type;
  uint32_t len;
  const void *data;
};
}

AiCpuResources::~AiCpuResources() noexcept {
  if ((!aicpu_queues_.empty()) || (!aicpu_channels_.empty()) || (!aicpu_vdec_channels_.empty())) {
    ReleaseResources();
  }
}

Status AiCpuResources::CreateQueue(const std::string &name, const uint32_t depth, uint32_t &queue_id) {
  GELOGD("Start to create queue, name = %s, depth = %u", name.c_str(), depth);
  std::vector<uint8_t> task_args;
  void *queue_id_dev = nullptr;
  GE_CHK_RT_RET(aclrtMalloc(&queue_id_dev, sizeof(queue_id), ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  GE_MAKE_GUARD(queue_id_dev, [&queue_id_dev]() {
    GE_CHK_RT(aclrtFree(queue_id_dev));
  });
  GE_CHK_STATUS_RET_NOLOG(
      BuildCreateQueueTask(static_cast<uintptr_t>(PtrToValue(queue_id_dev)), name, depth, task_args));
  GE_CHK_STATUS_RET(ExecuteKernel(kKernelNameCreateQueue, task_args));
  GE_CHK_RT_RET(aclrtMemcpy(&queue_id, sizeof(queue_id), queue_id_dev,
      sizeof(queue_id), ACL_MEMCPY_DEVICE_TO_HOST));
  GELOGD("Queue created successfully, name = %s, queue id = %u", name.c_str(), queue_id);
  return SUCCESS;
}

Status AiCpuResources::BuildCreateQueueTask(const uintptr_t queue_id_dev,
                                            const std::string &name,
                                            const uint32_t depth,
                                            std::vector<uint8_t> &task_args) {
  constexpr size_t args_size = sizeof(aicpu::AicpuParamHead) + sizeof(uintptr_t) + kNameMaxLength + sizeof(uint32_t);
  GE_CHK_STATUS_RET(Resize(task_args, args_size), "Failed to resize task args, args_size=%zu", args_size);

  auto &param_head = *(static_cast<aicpu::AicpuParamHead *>(static_cast<void *>(task_args.data())));
  param_head.length = args_size;
  param_head.ioAddrNum = 1U;  // single input: queue_id
  size_t args_pos = sizeof(aicpu::AicpuParamHead);

  // assign queue id
  *(static_cast<uintptr_t *>(static_cast<void *>(&task_args[args_pos]))) = queue_id_dev;
  args_pos += sizeof(uintptr_t);

  // assign queue name
  if (strcpy_s(static_cast<char_t *>(static_cast<void *>(&task_args[args_pos])), kNameMaxLength, name.c_str()) != EOK) {
    GELOGE(INTERNAL_ERROR, "Failed to copy queue name");
    return INTERNAL_ERROR;
  }
  args_pos += kNameMaxLength;

  // assign queue depth
  *(static_cast<uint32_t *>(static_cast<void *>(&task_args[args_pos]))) = depth;

  GELOGD("%s task args constructed, size = %zu", kKernelNameCreateQueue.c_str(), args_size);
  return SUCCESS;
}

Status AiCpuResources::CreateChannel(const int32_t rt_stream_id) {
  GELOGD("Start to create channel, rt stream id = %d", rt_stream_id);
  std::vector<uint8_t> task_args;
  GE_CHK_STATUS_RET_NOLOG(BuildCreateChannelTask(rt_stream_id, task_args));
  GE_CHK_STATUS_RET(ExecuteKernel(kSoNameBuiltin.c_str(), kKernelNameCreateChannel, task_args));
  GELOGD("Channel created successfully, rt stream id = %d", rt_stream_id);
  return SUCCESS;
}

Status AiCpuResources::BuildCreateChannelTask(const int32_t rt_stream_id,
                                              std::vector<uint8_t> &task_args) {
  constexpr size_t args_size = sizeof(aicpu::AicpuParamHead) + sizeof(int32_t);
  GE_CHK_STATUS_RET(Resize(task_args, args_size), "Failed to resize task args, args_size=%zu", args_size);

  auto &param_head = *(static_cast<aicpu::AicpuParamHead *>(static_cast<void *>(task_args.data())));
  param_head.length = args_size;
  param_head.ioAddrNum = 0U;
  task_args[sizeof(aicpu::AicpuParamHead)] = static_cast<uint8_t>(rt_stream_id);

  // assign rt stream id
  GELOGD("%s task args constructed, size = %zu", kKernelNameCreateChannel.c_str(), args_size);
  return SUCCESS;
}

Status AiCpuResources::CreateVdecChannel(const int32_t rt_stream_id) {
  GELOGD("Start to create Vdec channel, rt stream id = %d", rt_stream_id);
  std::vector<uint8_t> task_args;
  GE_CHK_STATUS_RET_NOLOG(BuildCreateVdecChannelTask(rt_stream_id, task_args));
  GE_CHK_STATUS_RET(ExecuteKernel(kSoNameBuiltin.c_str(), kKernelNameCreateVdecChannel, task_args));
  GELOGD("Vdec channel created successfully, rt stream id = %d", rt_stream_id);
  return SUCCESS;
}

Status AiCpuResources::BuildCreateVdecChannelTask(const int32_t rt_stream_id,
                                                  std::vector<uint8_t> &task_args) {
  constexpr size_t args_size = sizeof(aicpu::AicpuParamHead) + sizeof(int32_t);
  GE_CHK_STATUS_RET(Resize(task_args, args_size), "Failed to resize task args, args_size=%zu", args_size);

  auto &param_head = *(static_cast<aicpu::AicpuParamHead *>(static_cast<void *>(task_args.data())));
  param_head.length = args_size;
  param_head.ioAddrNum = 0U;
  task_args[sizeof(aicpu::AicpuParamHead)] = static_cast<uint8_t>(rt_stream_id);

  // assign rt stream id
  GELOGD("%s task args constructed, size = %zu", kKernelNameCreateVdecChannel.c_str(), args_size);
  return SUCCESS;
}

Status AiCpuResources::ExecuteKernel(const char_t *const so_name,
                                     const std::string &kernel_name,
                                     const std::vector<uint8_t> &task_args) {
  rtStream_t stream = nullptr;
  GE_CHK_RT_RET(rtStreamCreate(&stream, kDefaultPriority));
  GE_MAKE_GUARD_RTSTREAM(stream);
  rtArgsEx_t args_info = {};
  args_info.args = const_cast<void *>(static_cast<const void *>(task_args.data()));
  args_info.argsSize = static_cast<uint32_t>(task_args.size());
  args_info.isNoNeedH2DCopy = 0U;
  GE_CHK_RT_RET(rtCpuKernelLaunchWithFlag(so_name,
      kernel_name.c_str(), kKernelBlockDim, &args_info, nullptr, stream, RT_KERNEL_DEFAULT));
  GELOGD("Launch kernel successfully, kernel name = %s", kernel_name.c_str());
  GE_CHK_RT_RET(rtStreamSynchronize(stream));
  GELOGD("Sync stream successfully, kernel name = %s", kernel_name.c_str());
  return SUCCESS;
}

Status AiCpuResources::ExecuteKernel(const std::string &kernel_name,
                                     const std::vector<uint8_t> &task_args) {
  return ExecuteKernel(nullptr, kernel_name, task_args);
}

Status AiCpuResources::DestroyQueue(const uint32_t queue_id) {
  GELOGD("Start to destroy queue, id = %u", queue_id);
  std::vector<uint8_t> task_args;
  GE_CHK_STATUS_RET(BuildDestroyQueueTask(queue_id, task_args), "Failed to init task args");
  GE_CHK_STATUS_RET(ExecuteKernel(kKernelNameDestroyQueue, task_args), "Failed to launch kernel");
  GELOGD("Queue destroyed successfully, queue id = %u", queue_id);
  return SUCCESS;
}

Status AiCpuResources::BuildDestroyQueueTask(const uint32_t queue_id, std::vector<uint8_t> &task_args) {
  const size_t args_size = sizeof(aicpu::AicpuParamHead) + sizeof(queue_id);
  GE_CHK_STATUS_RET(Resize(task_args, args_size), "Failed to resize task args, args_size=%zu", args_size);
  auto &param_head = *PtrToPtr<uint8_t, aicpu::AicpuParamHead>(task_args.data());
  param_head.length = args_size;
  param_head.ioAddrNum = 0U;  // no input

  // assign queue id
  *(static_cast<uint32_t *>(static_cast<void *>(&task_args[sizeof(aicpu::AicpuParamHead)]))) = queue_id;
  GELOGD("%s task args constructed, size = %zu", kKernelNameDestroyQueue.c_str(), args_size);
  return SUCCESS;
}

Status AiCpuResources::DestroyChannel(const int32_t rt_stream_id) {
  GELOGD("Start to destroy channel, rt stream id = %d", rt_stream_id);
  std::vector<uint8_t> task_args;
  GE_CHK_STATUS_RET(BuildDestroyChannelTask(rt_stream_id, task_args), "Failed to init task args");
  GE_CHK_STATUS_RET(ExecuteKernel(kSoNameBuiltin.c_str(), kKernelNameDestroyChannel, task_args),
                    "Failed to launch kernel");
  GELOGD("Channel destroyed successfully, rt stream id = %d", rt_stream_id);
  return SUCCESS;
}

Status AiCpuResources::BuildDestroyChannelTask(const int32_t rt_stream_id, std::vector<uint8_t> &task_args) {
  const size_t args_size = sizeof(aicpu::AicpuParamHead) + sizeof(rt_stream_id);
  GE_CHK_STATUS_RET(Resize(task_args, args_size), "Failed to resize task args, args_size=%zu", args_size);
  auto &param_head = *PtrToPtr<uint8_t, aicpu::AicpuParamHead>(task_args.data());
  param_head.length = args_size;
  param_head.ioAddrNum = 0U;  // no input

  // assign rt stream id
  *(static_cast<int32_t *>(static_cast<void *>(&task_args[sizeof(aicpu::AicpuParamHead)]))) = rt_stream_id;
  GELOGD("%s task args constructed, size = %zu", kKernelNameDestroyChannel.c_str(), args_size);
  return SUCCESS;
}

Status AiCpuResources::DestroyVdecChannel(const int32_t rt_stream_id) {
  GELOGD("Start to destroy Vdec channel, rt stream id = %d", rt_stream_id);
  std::vector<uint8_t> task_args;
  GE_CHK_STATUS_RET(BuildDestroyVdecChannelTask(rt_stream_id, task_args), "Failed to init task args");
  GE_CHK_STATUS_RET(ExecuteKernel(kSoNameBuiltin.c_str(), kKernelNameDestroyVdecChannel, task_args),
                    "Failed to launch kernel");
  GELOGD("Vdec channel destroyed successfully, rt stream id = %d", rt_stream_id);
  return SUCCESS;
}

Status AiCpuResources::BuildDestroyVdecChannelTask(const int32_t rt_stream_id, std::vector<uint8_t> &task_args) {
  const size_t args_size = sizeof(aicpu::AicpuParamHead) + sizeof(rt_stream_id);
  GE_CHK_STATUS_RET(Resize(task_args, args_size), "Failed to resize task args, args_size=%zu", args_size);
  auto &param_head = *PtrToPtr<uint8_t, aicpu::AicpuParamHead>(task_args.data());
  param_head.length = args_size;
  param_head.ioAddrNum = 0U;  // no input

  // assign rt stream id
  *(static_cast<int32_t *>(static_cast<void *>(&task_args[sizeof(aicpu::AicpuParamHead)]))) = rt_stream_id;
  GELOGD("%s task args constructed, size = %zu", kKernelNameDestroyVdecChannel.c_str(), args_size);
  return SUCCESS;
}

const std::string &AiCpuResources::ResourceTypeQueue() {
  return kResourceTypeQueue;
}

const std::string &AiCpuResources::ResourceTypeChannel() {
  return kResourceTypeChannel;
}

const std::string &AiCpuResources::ResourceTypeVdecChannel() {
  return kResourceTypeVdecChannel;
}

Status AiCpuResources::AllocateQueueResource(const OpDescPtr &op_desc,
                                             const NamedAttrs &resource_attr,
                                             int32_t &input_idx,
                                             uint32_t &queue_id) {
  std::string queue_name;
  int64_t input_index = -1;
  if (!AttrUtils::GetStr(resource_attr, kAttrFieldQueueName, queue_name)) {
    GELOGE(PARAM_INVALID, "[%s] Failed to get queue name", op_desc->GetName().c_str());
    return PARAM_INVALID;
  }
  if (!AttrUtils::GetInt(resource_attr, kAttrFieldQueueIdIdx, input_index)) {
    GELOGE(PARAM_INVALID, "[%s] Failed to get input index for queue %s",
           op_desc->GetName().c_str(), queue_name.c_str());
    return PARAM_INVALID;
  }
  uint32_t queue_depth = kAiCpuQueueDepth;
  if (AttrUtils::GetInt(resource_attr, kAttrFieldQueueDepth, queue_depth)) {
    GELOGD("Got queue depth from attribute = %u", queue_depth);
  }
  GE_CHECK_GE(input_index, 0);
  GE_CHECK_LE(input_index, INT32_MAX);
  input_idx = static_cast<int32_t>(input_index);
  GE_CHK_STATUS_RET_NOLOG(GetOrCreateQueue(queue_name, queue_depth, queue_id));
  return SUCCESS;
}

Status AiCpuResources::GetOrCreateQueue(const std::string &queue_name, const uint32_t queue_depth, uint32_t &queue_id) {
  const std::lock_guard<std::mutex> lk(mu_);
  const auto it = aicpu_queues_.find(queue_name);
  if (it != aicpu_queues_.end()) {
    queue_id = it->second;
    GELOGD("Queue [%s] already created, queue_id = %u", queue_name.c_str(), queue_id);
    return SUCCESS;
  }
  GE_CHK_STATUS_RET(CreateQueue(queue_name, queue_depth, queue_id),
                    "Failed to create queue, name = %s",
                    queue_name.c_str());
  (void)aicpu_queues_.emplace(queue_name, queue_id);
  return SUCCESS;
}

Status AiCpuResources::AllocateChannelResource(const OpDescPtr &op_desc,
                                               const int32_t rt_stream_id) {
  const std::lock_guard<std::mutex> lk(mu_);
  const auto it = aicpu_channels_.find(rt_stream_id);
  if (it != aicpu_channels_.end()) {
    GELOGD("[%s] Channel already created, rt_stream_id = %d", op_desc->GetName().c_str(), rt_stream_id);
    return SUCCESS;
  }
  GE_CHK_STATUS_RET(CreateChannel(rt_stream_id), "Failed to create channel, id = %d", rt_stream_id);
  (void)aicpu_channels_.emplace(rt_stream_id);
  return SUCCESS;
}

Status AiCpuResources::AllocateVdecChannelResource(const OpDescPtr &op_desc,
                                                   const int32_t rt_stream_id) {
  const std::lock_guard<std::mutex> lk(mu_);
  const auto it = aicpu_vdec_channels_.find(rt_stream_id);
  if (it != aicpu_vdec_channels_.end()) {
    GELOGD("[%s] Channel already created, rt_stream_id = %d", op_desc->GetName().c_str(), rt_stream_id);
    return SUCCESS;
  }
  GE_CHK_STATUS_RET(CreateVdecChannel(rt_stream_id), "Failed to create vdec channel, id = %d", rt_stream_id);
  (void)aicpu_vdec_channels_.emplace(rt_stream_id);
  return SUCCESS;
}

void AiCpuResources::ReleaseResources() {
  const std::lock_guard<std::mutex> lk(mu_);
  GELOGD("Release queue resource started, size = %zu", aicpu_queues_.size());
  for (const auto &it : aicpu_queues_) {
    GE_CHK_STATUS(DestroyQueue(it.second),
                  "Failed to destroy queue, name = %s, queue id = %u", it.first.c_str(), it.second);
  }
  aicpu_queues_.clear();

  GELOGD("Release channel resource started, size = %zu", aicpu_queues_.size());
  for (const int32_t it : aicpu_channels_) {
    GE_CHK_STATUS(DestroyChannel(it), "Failed to destroy channel, rt stream id = %d", it);
  }
  aicpu_channels_.clear();

  GELOGD("Release vdec channel resource started, size = %zu", aicpu_queues_.size());
  for (const int32_t it : aicpu_vdec_channels_) {
    GE_CHK_STATUS(DestroyVdecChannel(it), "Failed to destroy vdec channel, rt stream id = %d", it);
  }
  aicpu_vdec_channels_.clear();

  GELOGD("Release ended");
}

Status AiCpuResources::BuildModelConfigTask(const AiCpuModelConfig &config, std::vector<uint8_t> &task_args) {
  const size_t args_size = sizeof(aicpu::AicpuParamHead) + sizeof(config);
  GE_CHK_STATUS_RET(Resize(task_args, args_size), "Failed to resize task args, args_size=%zu", args_size);
  auto &param_head = *(PtrToPtr<uint8_t, aicpu::AicpuParamHead>(task_args.data()));
  param_head.length = args_size;
  param_head.ioAddrNum = 0U;  // no input

  // assign queue id
  *(PtrToPtr<uint8_t, AiCpuModelConfig>(&task_args[sizeof(aicpu::AicpuParamHead)])) = config;
  GELOGD("%s task args constructed, size = %zu", kKernelNameModelConfig.c_str(), args_size);
  return SUCCESS;
}

Status AiCpuResources::SetModelConfig(const AiCpuModelConfig &config) const {
  GELOGD("Start to model config");
  std::vector<uint8_t> task_args;
  GE_CHK_STATUS_RET(BuildModelConfigTask(config, task_args), "Failed to init task args");
  GE_CHK_STATUS_RET(ExecuteKernel(kKernelNameModelConfig, task_args),
                    "Failed to launch kernel");
  GELOGD("Model config successfully");
  return SUCCESS;
}

Status AiCpuResources::BuildModelShapeConfigTask(const AiCpuModelShapeConfig &config, std::vector<uint8_t> &task_args) {
  const size_t args_size = sizeof(aicpu::AicpuParamHead) + sizeof(config);
  GE_CHK_STATUS_RET(Resize(task_args, args_size), "Failed to resize task args, args_size=%zu", args_size);
  auto &param_head = *(PtrToPtr<uint8_t, aicpu::AicpuParamHead>(task_args.data()));
  param_head.length = args_size;
  param_head.ioAddrNum = 0U;  // no input

  // assign queue id
  *(PtrToPtr<uint8_t, AiCpuModelShapeConfig>(&task_args[sizeof(aicpu::AicpuParamHead)])) = config;
  GELOGD("%s task args constructed, size = %zu", kKernelNameModelConfig.c_str(), args_size);
  return SUCCESS;
}

bool AiCpuResources::GetStaticModelShapeConfigRet() const {
  return static_model_shape_config_result_;
}

Status AiCpuResources::SetStaticModelShapeConfig(const AiCpuModelShapeConfig &config,
                                                 const std::vector<InputOutputDescInfo> &input_desc_list) {
  const std::function<bool(std::vector<int64_t>)> is_static_shape = [](const std::vector<int64_t> &dims) -> bool {
    GELOGI("Input shape is %s", ToString(dims).c_str());
    return std::all_of(dims.begin(), dims.end(), [](int64_t dim) ->bool { return dim >= 0; });
  };
  static_model_shape_config_result_ = false;
  // value agreed with aicpu  1000  10001
  constexpr uint32_t kTagShape = 1000U;
  constexpr uint32_t kTagType = 1001U;
  std::vector<TlvBuf> tlv_data = {};
  size_t tlv_data_len = 0;
  for (const auto &input_desc : input_desc_list) {
    const std::vector<int64_t> &shape = input_desc.shape_info.dims;
    if (!is_static_shape(shape)) {
      GELOGI("Input [%s] is not static shape[%s].", input_desc.name.c_str(), formats::ShapeToString(shape).c_str());
      return SUCCESS;
    }
    GE_CHK_STATUS_RET(
        ge::CheckUint32MulOverflow(static_cast<uint32_t>(shape.size()), static_cast<uint32_t>(sizeof(int64_t))),
        "%zu and %zu multiplication can result in overflow!", shape.size(), sizeof(int64_t));
    const uint32_t shape_len = static_cast<uint32_t>(shape.size() * sizeof(int64_t));
    const TlvBuf shape_tlv = {.type = kTagShape, .len = shape_len, .data = shape.data()};
    const TlvBuf type_tlv = {
        .type = kTagType, .len = static_cast<uint32_t>(sizeof(uint32_t)), .data = &input_desc.data_type};
    tlv_data.emplace_back(shape_tlv);
    tlv_data.emplace_back(type_tlv);
    tlv_data_len += (sizeof(TlvHead) + sizeof(TlvHead) +
                     static_cast<size_t>(shape_tlv.len) + static_cast<size_t>(type_tlv.len));
  }
  if (tlv_data_len == 0U) {
    return SUCCESS;
  }

  std::vector<uint8_t> config_buff(tlv_data_len);
  uint8_t *config_tlv_begin = config_buff.data();
  const uint8_t *config_tlv_end = PtrAdd(config_tlv_begin, tlv_data_len + 1U, tlv_data_len);
  for (const auto &tlv_item : tlv_data) {
    int64_t left_size = config_tlv_end - config_tlv_begin;
    errno_t ret = memcpy_s(config_tlv_begin, static_cast<size_t>(left_size), &tlv_item, sizeof(TlvHead));
    GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "copy tlv head failed");
    config_tlv_begin += sizeof(TlvHead);
    if (tlv_item.len > 0U) {
      left_size = config_tlv_end - config_tlv_begin;
      ret = memcpy_s(config_tlv_begin, static_cast<size_t>(left_size),
                     tlv_item.data, static_cast<size_t>(tlv_item.len));
      GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "copy tlv data failed");
    }
    config_tlv_begin += tlv_item.len;
  }

  void *tlv_device_addr = nullptr;
  GE_CHK_RT_RET(aclrtMalloc(&tlv_device_addr, config_buff.size(), ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  GE_MAKE_GUARD(tlv_device_addr, [&tlv_device_addr]() { GE_CHK_RT(aclrtFree(tlv_device_addr)); });
  GE_CHK_RT_RET(aclrtMemcpy(tlv_device_addr, config_buff.size(), config_buff.data(),
      config_buff.size(), ACL_MEMCPY_HOST_TO_DEVICE));
  AiCpuModelShapeConfig config_with_input_desc = config;
  GE_CHK_BOOL_RET_STATUS(tlv_data_len <= UINT32_MAX, FAILED, "tlv_data_len %zu greater than uint32_max.", tlv_data_len);
  config_with_input_desc.data_len = static_cast<uint32_t>(tlv_data_len);
  config_with_input_desc.data_device_addr = PtrToValue(tlv_device_addr);

  std::vector<uint8_t> task_args;
  GE_CHK_STATUS_RET(BuildModelShapeConfigTask(config_with_input_desc, task_args));
  GE_CHK_STATUS_RET(ExecuteKernel(kKernelNameModelShapeConfig, task_args), "Failed to launch kernel");
  static_model_shape_config_result_ = true;
  GELOGD("Static model shape config successfully");
  return SUCCESS;
}
}  // namespace ge

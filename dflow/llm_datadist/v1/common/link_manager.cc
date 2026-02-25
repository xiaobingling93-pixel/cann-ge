/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/link_manager.h"
#include <algorithm>
#include "common/llm_ge_api.h"
#include "common/llm_checker.h"
#include "common/def_types.h"
#include "common/llm_scope_guard.h"

namespace llm {
namespace {
constexpr uint32_t kGraphId = 0U;
constexpr uint32_t kMaxLinkAndUnlinkNum = 16U;
constexpr uint64_t kMillsToMicros = 1000UL;
constexpr uint32_t kIpValidSize = 4U;
constexpr uint32_t kIpRangeSize = 256U;
constexpr int32_t kDefaultTimeout = 3000;
constexpr int32_t kHostExtraTimeout = 3000;
constexpr int64_t kFetchExtraTimeInMicros = 1000000;
std::string ToIp(uint32_t little_endian_decimal) {
  std::string ip;
  for (uint32_t i = 0U; i < kIpValidSize; ++i) {
    ip.append(std::to_string(little_endian_decimal % kIpRangeSize));
    little_endian_decimal /= kIpRangeSize;
    if (i < (kIpValidSize - 1)) {
      ip.append(".");
    }
  }
  return ip;
}

std::string ToDesc(const IpInfo &ip_info) {
  std::string desc;
  desc.append("[ip:")
      .append(ToIp(ip_info.ip))
      .append("_")
      .append(std::to_string(ip_info.ip))
      .append(", port:")
      .append(std::to_string(ip_info.port))
      .append("]");
  return desc;
}
}  // namespace

ge::Status LinkManager::CheckClusterInfo(const std::vector<ClusterInfo> &clusters, const size_t sliced_num) const {
  LLM_CHK_BOOL_RET_STATUS(!clusters.empty(), ge::LLM_PARAM_INVALID, "clusters info is empty");
  const auto local_ip_infos = clusters.front().local_ip_infos;
  for (const auto &cluster : clusters) {
    LLMLOGI("Link remote cluster:%lu, remote role type:%d", cluster.remote_cluster_id, cluster.remote_role_type);
    LLM_CHK_BOOL_RET_STATUS(((!cluster.local_ip_infos.empty()) && (!cluster.remote_ip_infos.empty())),
                           ge::LLM_PARAM_INVALID, "clusters info is empty");
    LLM_CHK_BOOL_RET_STATUS(cluster.local_ip_infos.size() == cluster.remote_ip_infos.size(), ge::LLM_PARAM_INVALID,
                           "local ip infos size:%zu, remote ip infos:%zu", cluster.local_ip_infos.size(),
                           cluster.remote_ip_infos.size());
    LLM_CHK_BOOL_RET_STATUS(cluster.local_ip_infos.size() == sliced_num, ge::LLM_PARAM_INVALID,
                           "cluster ip info size:%zu not match sliced_num:%zu", cluster.local_ip_infos.size(),
                           sliced_num);
    LLM_CHK_BOOL_RET_STATUS(cluster.local_ip_infos == local_ip_infos, ge::LLM_PARAM_INVALID,
                           "local_ip_infos should be same");
    for (const auto &remote_ip : cluster.remote_ip_infos) {
      LLMLOGI("Link remote ip:%u", ToDesc(remote_ip).c_str());
    }
    for (const auto &local_ip : cluster.local_ip_infos) {
      LLMLOGI("Link local ip:%u", ToDesc(local_ip).c_str());
    }
  }
  return ge::SUCCESS;
}

void LinkManager::Initialize(const std::vector<uint32_t> &link_input_indices,
                             const std::vector<uint32_t> &link_output_indices, const size_t device_num,
                             uint64_t cluster_id) {
  link_input_indices_ = link_input_indices;
  link_output_indices_ = link_output_indices;
  unlink_input_indices_ = link_input_indices;
  unlink_output_indices_ = link_output_indices;
  device_num_ = device_num;
  cluster_id_ = cluster_id;
}

void LinkManager::SetDifferentUnlinkIndices(const std::vector<uint32_t> &unlink_input_indices,
                                            const std::vector<uint32_t> &unlink_output_indices) {
  unlink_input_indices_ = unlink_input_indices;
  unlink_output_indices_ = unlink_output_indices;
  need_extra_host_timeout_ = false;
}

ge::Status LinkManager::PrepareLinkInfoTensors(const std::vector<ClusterInfo> &clusters,
                                               const LinkOperator operator_type, const size_t sliced_num,
                                               const int32_t timeout, bool force_flag) {
  LLM_CHK_BOOL_RET_STATUS(CheckClusterInfo(clusters, sliced_num) == ge::SUCCESS, ge::LLM_PARAM_INVALID,
                         "check cluster info failed");
  uint64_t link_info_timeout = timeout <= 0 ? 0UL : static_cast<uint64_t>(timeout) * kMillsToMicros;
  cluster_num_ = clusters.size();
  LLM_CHK_BOOL_RET_STATUS(cluster_num_ <= kMaxLinkAndUnlinkNum, ge::LLM_CLUSTER_NUM_EXCEED_LIMIT,
                         "link or unlink size:%zu exceeds the upper limit:%u", cluster_num_, kMaxLinkAndUnlinkNum);
  size_t total_inner_clusters_info_size = 0U;
  for (size_t i = 0UL; i < cluster_num_; ++i) {
    const auto &cluster = clusters[i];
    total_inner_clusters_info_size += sizeof(InnerClusterInfo) + cluster.local_ip_infos.size() * sizeof(InnerIpInfo);
  }
  const auto &link_info_size = sizeof(LinkInfo) + total_inner_clusters_info_size;
  LinkInfo *link_info = (LinkInfo *)malloc(link_info_size);
  LLM_CHK_BOOL_RET_STATUS(link_info != nullptr, ge::LLM_PARAM_INVALID, "allocate link info failed");
  LLM_MAKE_GUARD(link_info, [link_info] { free(link_info); });
  link_info->operate_type = static_cast<uint32_t>(operator_type);
  link_info->cluster_info_num = cluster_num_;
  link_info->timeout = link_info_timeout;
  link_info->force_flag = force_flag ? 1U : 0U;
  for (size_t i = 0U; i < cluster_num_; ++i) {
    const auto &cluster = clusters[i];
    const size_t inner_cluster_info_size =
        sizeof(InnerClusterInfo) + cluster.local_ip_infos.size() * sizeof(InnerIpInfo);
    auto inner_cluster_info_addr = llm::PtrToValue(link_info->cluster_infos) + i * (inner_cluster_info_size);
    LLM_ASSERT_EOK(memcpy_s(llm::ValueToPtr(inner_cluster_info_addr), sizeof(uint64_t), &cluster_id_,
                           sizeof(uint64_t)));
    // 给InnerClusterInfo remote_cluster_id赋值
    inner_cluster_info_addr += sizeof(uint64_t);
    LLM_ASSERT_EOK(memcpy_s(llm::ValueToPtr(inner_cluster_info_addr), sizeof(uint64_t), &cluster.remote_cluster_id,
                           sizeof(uint64_t)));
    // 给给InnerClusterInfo ip_infos_num赋值
    const uint32_t ip_infos_num = static_cast<uint32_t>(cluster.local_ip_infos.size());
    auto ip_infos_num_addr = inner_cluster_info_addr + sizeof(uint64_t);
    LLM_ASSERT_EOK(memcpy_s(llm::ValueToPtr(ip_infos_num_addr), sizeof(uint32_t), &ip_infos_num, sizeof(uint32_t)));
    for (size_t j = 0U; j < cluster.local_ip_infos.size(); ++j) {
      InnerIpInfo inner_ip_info;
      inner_ip_info.local_ip = cluster.local_ip_infos[j].ip;
      inner_ip_info.local_port = cluster.local_ip_infos[j].port;
      inner_ip_info.remote_ip = cluster.remote_ip_infos[j].ip;
      inner_ip_info.remote_port = cluster.remote_ip_infos[j].port;
      LLM_ASSERT_EOK(memcpy_s(llm::ValueToPtr(ip_infos_num_addr + sizeof(uint32_t) + j * sizeof(InnerIpInfo)),
                             sizeof(InnerIpInfo), &inner_ip_info, sizeof(InnerIpInfo)));
      InnerIpInfo *ptr =
          static_cast<InnerIpInfo *>(llm::ValueToPtr(ip_infos_num_addr + sizeof(uint32_t) + j * sizeof(InnerIpInfo)));
      LLMLOGI("Link local ip:%u", ptr->local_ip);
      LLMLOGI("Link remote ip:%u", ptr->remote_ip);
    }
  }
  link_inputs_.clear();
  std::vector<int64_t> dims{static_cast<int64_t>(link_info_size / sizeof(uint8_t))};
  ge::TensorDesc tensor_desc(ge::Shape(dims), ge::FORMAT_ND, ge::DT_UINT8);
  link_inputs_.emplace_back(ge::Tensor(tensor_desc, llm::PtrToPtr<LinkInfo, uint8_t>(link_info), link_info_size));
  return ge::SUCCESS;
}
ge::Status LinkManager::ParseOutputRetStatus(const ge::Tensor &output, std::vector<ge::Status> &result) const {
  const auto data = output.GetData();
  LLM_CHK_BOOL_RET_STATUS(output.GetSize() / sizeof(int32_t) >= cluster_num_, ge::LLM_PARAM_INVALID,
                         "The output contains invalid return values");
  for (size_t i = 0U; i < cluster_num_; ++i) {
    result.emplace_back(ConvertRetCode(*llm::PtrToPtr<uint8_t, int32_t>(data + i * sizeof(int32_t))));
  }
  return ge::SUCCESS;
}

ge::Status LinkManager::LinkOrUnlinkAsync(int32_t timeout, bool is_link) {
  ge::DataFlowInfo dataFlowInfo;
  auto input_indices = is_link ? link_input_indices_ : unlink_input_indices_;
  LLMLOGI("input_indices_ size:%zu, tensor size:%zu", input_indices.size(), link_inputs_.size());
  const auto ret = FeedInputs(kGraphId, input_indices, link_inputs_, timeout, is_link);
  LLM_CHK_BOOL_RET_STATUS(ret == ge::SUCCESS, ret, "link or unlink feed data failed.");
  return ge::SUCCESS;
}

ge::Status LinkManager::FeedInputs(const uint32_t graph_id,
                                   const std::vector<uint32_t> &indices,
                                   const std::vector<ge::Tensor> &inputs,
                                   const int32_t timeout,
                                   bool is_link) {
  (void)is_link;
  ge::DataFlowInfo data_flow_info;
  return GeApi::GetInstance().FeedDataFlowGraph(graph_id, indices, inputs, data_flow_info, timeout);
}

ge::Status LinkManager::FetchOutputs(const uint32_t graph_id,
                                     const std::vector<uint32_t> &indices,
                                     std::vector<ge::Tensor> &outputs,
                                     int64_t timeout,
                                     uint64_t transaction_id) {
  (void)transaction_id;
  ge::DataFlowInfo flow_info;
  auto ret = GeApi::GetInstance().FetchDataFlowGraph(graph_id, indices, outputs, flow_info,
                                                     static_cast<int32_t>(timeout / kMillsToMicros));
  LLM_CHK_BOOL_RET_STATUS(ret == ge::SUCCESS, ge::FAILED, "Fetch failed.");
  return ret;
}

void LinkManager::HandleFetchEmpty(ge::Status status, const std::vector<ge::Tensor> &outputs,
                                   std::vector<ge::Status> &rets) const {
  // if link timeout happened in host, take it as not recoverable.
  if ((status != ge::SUCCESS) && outputs.empty()) {
    for (size_t i = 0U; i < cluster_num_; ++i) {
      LLMLOGI("Set ret to %u when fetch result is empty.", status);
      rets[i] = status;
    }
  }
}

ge::Status LinkManager::GetLinkResult(const std::vector<ClusterInfo> &clusters, int64_t left_timeout,
                                      std::vector<ge::Status> &rets, std::vector<ClusterInfo> &need_rollback_clusters,
                                      uint64_t transaction_id) {
  ge::DataFlowInfo flow_info;
  std::vector<ge::Tensor> outputs;
  ge::Status ret = FetchOutputs(kGraphId, link_output_indices_, outputs, left_timeout, transaction_id);
  rets.resize(cluster_num_, ge::SUCCESS);
  HandleFetchEmpty(ret, outputs, rets);
  LLM_CHK_BOOL_RET_STATUS(ret == ge::SUCCESS, ret, "decoder graph fetch data failed.");
  // loop for device
  std::vector<size_t> rollback_indexes;
  for (const auto &output : outputs) {
    std::vector<ge::Status> device_link_rets;
    LLM_CHK_STATUS_RET(ParseOutputRetStatus(output, device_link_rets), "Parse output ret status failed.");
    for (size_t i = 0U; i < cluster_num_; ++i) {
      // cluster already has a error
      if (std::find(rollback_indexes.begin(), rollback_indexes.end(), i) != rollback_indexes.end()) {
        continue;
      }
      if (rets[i] == ge::SUCCESS) {
        rets[i] = device_link_rets[i];
      }
      if ((rets[i] == ge::LLM_LINK_FAILED) || (rets[i] == ge::LLM_WAIT_PROC_TIMEOUT) ||
          (rets[i] == ge::LLM_NOTIFY_PROMPT_UNLINK_FAILED)) {
        rollback_indexes.emplace_back(i);
        need_rollback_clusters.emplace_back(clusters[i]);
      }
    }
  }
  return ret;
}

ge::Status LinkManager::GetUnLinkResult(int64_t left_timeout, std::vector<ge::Status> &rets, uint64_t transaction_id) {
  ge::DataFlowInfo flow_info;
  std::vector<ge::Tensor> outputs;
  ge::Status ret = FetchOutputs(kGraphId, unlink_output_indices_, outputs, left_timeout, transaction_id);
  rets.resize(cluster_num_, ge::SUCCESS);
  HandleFetchEmpty(ret, outputs, rets);
  LLM_CHK_BOOL_RET_STATUS((ret == ge::SUCCESS) || (ret == ge::LLM_WAIT_PROC_TIMEOUT), ret,
                         "decoder graph fetch data failed.");
  for (const auto &output : outputs) {
    std::vector<ge::Status> result;
    const auto status_ret = ParseOutputRetStatus(output, result);
    LLM_CHK_BOOL_RET_STATUS(status_ret == ge::SUCCESS, status_ret, "parser output ret status failed");
    for (size_t i = 0U; i < cluster_num_; ++i) {
      ge::Status &cluster_ret = rets[i];
      if ((cluster_ret != ge::SUCCESS) && (cluster_ret != ge::LLM_NOT_YET_LINK)) {
        continue;
      }
      cluster_ret = result[i];
    }
  }
  return ret;
}

ge::Status LinkManager::LinkClusters(const std::vector<ClusterInfo> &clusters, std::vector<ge::Status> &rets,
                                     const int32_t timeout) {
  std::unique_lock<std::timed_mutex> lock(link_mtx_, std::defer_lock);
  if (!lock.try_lock_for(std::chrono::microseconds(timeout))) {
    LLMEVENT("Link is currently processing. Try again later.");
    return ge::LLM_PROCESSING_LINK;
  }
  const auto link_start = std::chrono::steady_clock::now();
  ge::Status ret = PrepareLinkInfoTensors(clusters, LinkOperator::Link, device_num_, timeout, false);
  LLM_CHK_BOOL_RET_STATUS(ret == ge::SUCCESS, ret, "Input clusters info is invalid");
  const auto prepare_input_end = std::chrono::steady_clock::now();
  auto cost_in_us = std::chrono::duration_cast<std::chrono::microseconds>(prepare_input_end - link_start);
  LLMEVENT("[LlmPerf] prepare link input tensor end, elapsed_us = %ld", cost_in_us.count());
  auto link_timeout = timeout < 0 ? kDefaultTimeout : timeout;
  link_timeout = need_extra_host_timeout_ ? (link_timeout + kHostExtraTimeout) : link_timeout;
  auto transaction_id = link_transaction_id_;
  ret = LinkOrUnlinkAsync(link_timeout, true);
  LLM_CHK_BOOL_RET_STATUS(ret == ge::SUCCESS, ret, "link feed input failed");
  const auto feed_end = std::chrono::steady_clock::now();
  cost_in_us = std::chrono::duration_cast<std::chrono::microseconds>(feed_end - prepare_input_end);
  LLMEVENT("[LlmPerf] feed link input tensor end, elapsed_us = %ld", cost_in_us.count());
  auto feed_time_cost = std::chrono::duration_cast<std::chrono::microseconds>(feed_end - link_start).count();
  auto timeout_in_micros = static_cast<int64_t>(link_timeout * kMillsToMicros);
  LLM_CHK_BOOL_RET_STATUS(feed_time_cost < timeout_in_micros, ge::LLM_WAIT_PROC_TIMEOUT,
                         "Feed timeout, cost:%ld us, left:%ld us.", feed_time_cost, timeout_in_micros);

  std::vector<ge::Tensor> outputs;
  std::vector<ClusterInfo> need_rollback_clusters;
  // make sure device timeout first.
  ret = GetLinkResult(clusters, timeout_in_micros + kFetchExtraTimeInMicros, rets, need_rollback_clusters, transaction_id);
  LLM_CHK_BOOL_RET_STATUS(ret == ge::SUCCESS, ret, "link fetch output failed");
  const auto link_end = std::chrono::steady_clock::now();
  cost_in_us = std::chrono::duration_cast<std::chrono::microseconds>(link_end - feed_end);
  LLMEVENT("[LlmPerf] fetch link result end, elapsed_us = %ld", cost_in_us.count());
  if (need_rollback_clusters.empty()) {
    return ret;
  }
  lock.unlock();
  std::vector<ge::Status> rollback_rets;
  ret = UnlinkClusters(need_rollback_clusters, LinkOperator::ClientUnlink, rollback_rets, timeout);
  LLM_CHK_BOOL_RET_STATUS(ret == ge::SUCCESS, ret, "rollback clusters failed");
  const auto rollback_end = std::chrono::steady_clock::now();
  cost_in_us = std::chrono::duration_cast<std::chrono::microseconds>(rollback_end - link_end);
  LLMEVENT("[LlmPerf] rollback link failed clusters end, elapsed_us = %ld", cost_in_us.count());
  return ge::LLM_LINK_FAILED;
}

ge::Status LinkManager::UnlinkClusters(const std::vector<ClusterInfo> &clusters, const LinkOperator operator_type,
                                       std::vector<ge::Status> &rets, const int32_t timeout, bool force_flag) {
  std::unique_lock<std::timed_mutex> lock(link_mtx_, std::defer_lock);
  if (!lock.try_lock_for(std::chrono::microseconds(timeout))) {
    LLMEVENT("Link is currently processing. Try again later.");
    return ge::LLM_PROCESSING_LINK;
  }
  const auto unlink_start = std::chrono::steady_clock::now();
  ge::Status ret = PrepareLinkInfoTensors(clusters, operator_type, device_num_, timeout, force_flag);
  LLM_CHK_BOOL_RET_STATUS(ret == ge::SUCCESS, ret, "Input clusters info is invalid");
  const auto prepare_input_end = std::chrono::steady_clock::now();
  auto cost_in_us = std::chrono::duration_cast<std::chrono::microseconds>(prepare_input_end - unlink_start);
  LLMEVENT("[LlmPerf] prepare unlink input tensor end, elapsed_us = %ld", cost_in_us.count());
  auto unlink_timeout = timeout < 0 ? kDefaultTimeout : timeout;
  unlink_timeout = need_extra_host_timeout_ ? (unlink_timeout + kHostExtraTimeout) : unlink_timeout;

  auto transaction_id = unlink_transaction_id_;
  ret = LinkOrUnlinkAsync(unlink_timeout, false);
  LLM_CHK_BOOL_RET_STATUS(ret == ge::SUCCESS, ret, "Unlink feed input failed");
  const auto feed_end = std::chrono::steady_clock::now();
  cost_in_us = std::chrono::duration_cast<std::chrono::microseconds>(feed_end - prepare_input_end);
  LLMEVENT("[LlmPerf] feed unlink input tensor end, elapsed_us = %ld", cost_in_us.count());
  auto feed_time_cost = std::chrono::duration_cast<std::chrono::microseconds>(feed_end - unlink_start).count();
  auto time_out_in_micros = static_cast<int64_t>(unlink_timeout * kMillsToMicros);
  LLM_CHK_BOOL_RET_STATUS(feed_time_cost < time_out_in_micros, ge::LLM_WAIT_PROC_TIMEOUT,
                         "Feed timeout, cost:%ld us, left:%ld us.", feed_time_cost, time_out_in_micros);

  std::vector<ge::Tensor> outputs;
  ret = GetUnLinkResult(time_out_in_micros + kFetchExtraTimeInMicros, rets, transaction_id);
  LLM_CHK_BOOL_RET_STATUS(ret == ge::SUCCESS, ret, "Unlink fetch output failed");
  const auto unlink_end = std::chrono::steady_clock::now();
  cost_in_us = std::chrono::duration_cast<std::chrono::microseconds>(unlink_end - feed_end);
  LLMEVENT("[LlmPerf] fetch unlink result end, elapsed_us = %ld", cost_in_us.count());
  if (std::find_if(rets.begin(), rets.end(), [](ge::Status &status) {
        return ((status != ge::SUCCESS) && (status != ge::LLM_NOT_YET_LINK));
      }) != rets.end()) {
    return ge::LLM_UNLINK_FAILED;
  }
  return ge::SUCCESS;
}
}  // namespace llm

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/op_tiling/op_tiling_rt2.h"
#include "common/utils/executor_utils.h"
#include "register/op_tiling_info.h"

namespace optiling {
bool EnableRt2Tiling(const ge::OpDescPtr &op_desc) {
  (void)op_desc;
  return true;
}
bool EnableAtomicRt2Tiling(const ge::OpDescPtr &op_desc) {
  (void)op_desc;
  return true;
}
ge::graphStatus RtParseAndTiling(const ge::Operator &op, const char *compile_info,
                                 const fe::PlatFormInfos &platform_infos, OutputsConvertorFun callback) {
  (void)op;
  (void)compile_info;
  (void)platform_infos;
  (void)callback;
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus AicoreRtParseAndTiling(const ge::Operator &op, const fe::PlatFormInfos &platform_infos,
                                       OpRunInfoV2 &run_info) {
  (void)op;
  if (platform_infos.GetCoreNum() == 4) {
    std::vector<int64_t> workspace1 = {1, 10, 20, 30};
    run_info.SetWorkspaces(workspace1);
  } else if (platform_infos.GetCoreNum() == 8) {
    std::vector<int64_t> workspace2 = {5, 8, 15, 50};
    run_info.SetWorkspaces(workspace2);
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus AtomicRtParseAndTiling(const ge::Operator &op, const fe::PlatFormInfos &platform_infos,
                                       OpRunInfoV2 &run_info) {
  (void)op;
  (void)platform_infos;
  (void)run_info;
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus SoftSyncOpRtParseAndTiling(const ge::Operator &op, fe::PlatFormInfos &platform_infos,
                                           rtArgsEx_t &args_ex, OpRunInfoV2 &run_info,
                                           const gert::OpImplSpaceRegistryV2Ptr space_registry) {
  (void)op;
  (void)platform_infos;
  (void)args_ex;
  (void)run_info;
  (void)space_registry;
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus FftsRtParseAndTiling(const ge::Operator &op, const fe::PlatFormInfos &platform_infos,
                                     std::vector<OpRunInfoV2> &op_run_infos) {
  (void)op;
  (void)platform_infos;
  (void)op_run_infos;
  return ge::GRAPH_SUCCESS;
}

namespace utils {
ByteBuffer tiling_data_;

OpRunInfo::OpRunInfo() {
}

OpRunInfo::OpRunInfo(const uint32_t &block_dim, const bool &clear_atomic, const uint64_t &tiling_key) {
  (void)block_dim;
  (void)clear_atomic;
  (void)tiling_key;
}

OpRunInfo::OpRunInfo(const OpRunInfo &runinfo) {
  (void)runinfo;
}

OpRunInfo::OpRunInfo(OpRunInfo &&runinfo) {
  (void)runinfo;
}

void OpRunInfo::SetBlockDim(const uint32_t &block_dim) {
  (void)block_dim;
}

uint32_t OpRunInfo::GetBlockDim() const {
  return 0U;
}

void OpRunInfo::AddWorkspace(const int64_t &workspace) {
  (void)workspace;
}

size_t OpRunInfo::GetWorkspaceNum() const {
  return 0U;
}

ge::graphStatus OpRunInfo::GetWorkspace(const size_t &idx, int64_t &workspace) const {
  (void)idx;
  (void)workspace;
  return ge::GRAPH_SUCCESS;
}

void OpRunInfo::GetAllWorkspaces(std::vector<int64_t> &workspaces) const {
  (void)workspaces;
}

std::vector<int64_t> value;
const std::vector<int64_t> &OpRunInfo::GetAllWorkspaces() const {
  return value;
}

void OpRunInfo::SetWorkspaces(const std::vector<int64_t> &workspaces) {
  (void)workspaces;
}

void OpRunInfo::InternelSetTiling(const ByteBuffer &value) {
  (void)value;
}

void OpRunInfo::AddTilingData(const ge::char_t *value, const size_t size) {
  (void)value;
  (void)size;
}

void OpRunInfo::AlignOffsetWith64() {
}

void* OpRunInfo::GetAddrBase(uint64_t& max_size) const {
  (void)max_size;
  return nullptr;
}

void OpRunInfo::SetAddrBaseOffset(const uint64_t size) {
  (void)size;
}

ByteBuffer &OpRunInfo::GetAllTilingData() {
  return tiling_data_;
}

const ByteBuffer &OpRunInfo::GetAllTilingData() const {
  return tiling_data_;
}
uint64_t OpRunInfo::GetTilingDataSize() const {
  return 0U;
}
void OpRunInfo::SetClearAtomic(const bool clear_atomic) {
  (void)clear_atomic;
}

bool OpRunInfo::GetClearAtomic() const {
  return true;
}

void OpRunInfo::SetTilingKey(const uint64_t &new_tiling_key) {
  (void)new_tiling_key;
}

uint64_t OpRunInfo::GetTilingKey() const {
  return 0U;
}

void OpRunInfo::ResetWorkspace() {
}

void OpRunInfo::ResetAddrBase(void *const addr_base, const uint64_t max_size) {
  (void)addr_base;
  (void)max_size;
}

void OpRunInfo::SetTilingCond(const int32_t tiling_cond) {
  (void)tiling_cond;
}

int32_t OpRunInfo::GetTilingCond() const {
  return 0;
}
}
}  // namespace optiling

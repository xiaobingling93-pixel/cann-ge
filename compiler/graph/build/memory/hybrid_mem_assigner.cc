/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/memory/hybrid_mem_assigner.h"
#include <utility>
#include <vector>
#include "framework/common/debug/ge_log.h"
#include "graph/build/memory/binary_block_mem_assigner.h"
#include "graph/build/memory/max_block_mem_assigner.h"
#include "graph/build/memory/mem_inplace.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/thread_pool/thread_pool.h"
#include "common/checker.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"

namespace ge {
static bool CompareMemorySize(const std::pair<std::string, std::pair<std::unique_ptr<BlockMemAssigner>, size_t>> &left,
    const std::pair<std::string, std::pair<std::unique_ptr<BlockMemAssigner>, size_t>> &right) {
  return left.second.second < right.second.second;
}

HybridMemAssigner::HybridMemAssigner(ge::ComputeGraphPtr compute_graph)
    : compute_graph_(std::move(compute_graph)), priority_assigner_(nullptr) {}

Status HybridMemAssigner::AssignMemory(BlockMemAssigner *block_assigner, size_t &mem_size,
                                       const GEThreadLocalContext &context) {
  GetThreadLocalContext() = context;
  std::vector<int64_t> ranges;
  GE_CHECK_NOTNULL(block_assigner);
  if (block_assigner->GetMemoryRanges(ranges) != SUCCESS) {
    GELOGE(FAILED, "[Get][MemoryRanges] Fail!");
    return FAILED;
  }

  GE_ASSERT_SUCCESS(block_assigner->AssignMemoryWithReuse(ranges));

  // total size
  for (auto it : block_assigner->GetMemOffsets()) {
    mem_size += it.second;
  }
  return SUCCESS;
}

Status HybridMemAssigner::ReuseCheckerInit() {
  reuse_checker_ = MakeUnique<ReuseChecker>(mem_assist_info_.compute_graph, mem_assist_info_.anchor_to_symbol,
                                            mem_assist_info_.symbol_to_anchors);
  GE_ASSERT_NOTNULL(reuse_checker_);
  const auto ret = reuse_checker_->Init();
  if (ret != SUCCESS) {
    GELOGW("reuse checker init failed, set reuse checker nullptr.");
    reuse_checker_ = nullptr;
  }
  return SUCCESS;
}

void HybridMemAssigner::ReuseCheckerDeInit() {
  reuse_checker_ = nullptr;
}

Status HybridMemAssigner::Assign() {
  if (GraphUtils::GetRefMapping(compute_graph_, mem_assist_info_.symbol_to_anchors,
                                mem_assist_info_.anchor_to_symbol) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Get ref-mapping for graph %s failed", compute_graph_->GetName().c_str());
    GELOGE(FAILED, "[Get][RefMapping] for graph %s failed.", compute_graph_->GetName().c_str());
    return FAILED;
  }
  mem_assist_info_.compute_graph = compute_graph_;
  GE_ASSERT_SUCCESS(ProcessInplace(mem_assist_info_));
  GE_ASSERT_SUCCESS(BlockMemAssigner::PreparationForAssign(mem_assist_info_));
  // name, memory assigner, memory size
  std::vector<std::pair<std::string, std::pair<std::unique_ptr<BlockMemAssigner>, size_t>>> memory_assigners;
  auto binary_assigner = MakeUnique<BinaryBlockMemAssigner>(mem_assist_info_);
  GE_CHECK_NOTNULL(binary_assigner);
  const bool memory_priority_mode = binary_assigner->IsMemoryPriorityMode();
  binary_assigner->SetReuseStrategy(ReuseStrategy(false, true, false, memory_priority_mode));

  std::future<void> reuse_checker_init_future = std::async(std::launch::async, [this] () { (void)ReuseCheckerInit(); });
  memory_assigners.emplace_back(std::make_pair("binary-block", std::make_pair(std::move(binary_assigner), 0U)));

  auto max_assigner = MakeUnique<MaxBlockMemAssigner>(mem_assist_info_);
  GE_CHECK_NOTNULL(max_assigner);
  max_assigner->SetReuseStrategy(ReuseStrategy(true));
  memory_assigners.emplace_back(std::make_pair("max-block", std::make_pair(std::move(max_assigner), 0U)));

  if (memory_priority_mode) {
    auto range_binary_assigner_descending_frfr = MakeUnique<BinaryBlockMemAssigner>(mem_assist_info_);
    GE_CHECK_NOTNULL(range_binary_assigner_descending_frfr);
    range_binary_assigner_descending_frfr->SetReuseStrategy(ReuseStrategy(true, false, true, memory_priority_mode));
    memory_assigners.emplace_back(std::make_pair("range-binary-block-descending-frfr",
                                                 std::make_pair(std::move(range_binary_assigner_descending_frfr), 0U)));

    auto range_binary_assigner_ascending_frfr = MakeUnique<BinaryBlockMemAssigner>(mem_assist_info_);
    GE_CHECK_NOTNULL(range_binary_assigner_ascending_frfr);
    range_binary_assigner_ascending_frfr->SetReuseStrategy(ReuseStrategy(true, true, true, memory_priority_mode));
    memory_assigners.emplace_back(std::make_pair("range-binary-block-ascending-frfr",
                                                 std::make_pair(std::move(range_binary_assigner_ascending_frfr), 0U)));

    auto range_binary_assigner_descending_frlr = MakeUnique<BinaryBlockMemAssigner>(mem_assist_info_);
    GE_CHECK_NOTNULL(range_binary_assigner_descending_frlr);
    range_binary_assigner_descending_frlr->SetReuseStrategy(ReuseStrategy(true, false, false, memory_priority_mode));
    memory_assigners.emplace_back(std::make_pair("range-binary-block-descending-frlr",
                                                 std::make_pair(std::move(range_binary_assigner_descending_frlr), 0U)));

    auto range_binary_assigner_ascending_frlr = MakeUnique<BinaryBlockMemAssigner>(mem_assist_info_);
    GE_CHECK_NOTNULL(range_binary_assigner_ascending_frlr);
    range_binary_assigner_ascending_frlr->SetReuseStrategy(ReuseStrategy(true, true, false, memory_priority_mode));
    memory_assigners.emplace_back(std::make_pair("range-binary-block-ascending-frlr",
                                                 std::make_pair(std::move(range_binary_assigner_ascending_frlr), 0U)));
  }

  ThreadPool executor("ge_asignmem", memory_assigners.size(), false);
  std::vector<std::future<Status>> vector_future;
  for (auto &memory_assigner : memory_assigners) {
    std::future<Status> f = executor.commit(HybridMemAssigner::AssignMemory, memory_assigner.second.first.get(),
                                            std::ref(memory_assigner.second.second),
                                            std::cref(GetThreadLocalContext()));
    if (f.valid()) {
      vector_future.emplace_back(std::move(f));
    }
  }

  for (size_t i = 0U; i < vector_future.size(); ++i) {
    GE_CHK_STATUS_RET(vector_future[i].get(), "[Assign][Memory] Fail!");
  }
  reuse_checker_init_future.get();
  // ascending sort by memory size, so assigner 0 is priority assigner
  std::sort(memory_assigners.begin(), memory_assigners.end(), CompareMemorySize);
  for (const auto &memory_assigner : memory_assigners) {
    GELOGI("%s memory assigner memory size:%zu", memory_assigner.first.c_str(), memory_assigner.second.second);
  }
  if ((!vector_future.empty()) && (!memory_assigners.empty())) {
    memory_assigners[0].second.first->SetOpMemOffset(false);
    mem_offsets_ = memory_assigners[0].second.first->GetMemOffsets();
    memory_stat_ = memory_assigners[0].second.first->GetMemoryStat();
    priority_assigner_ = std::move(memory_assigners[0].second.first);
  } else {
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace ge

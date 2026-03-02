/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/manager/active_memory_allocator.h"
#include <string>
#include "common/math/math_util.h"
#include "common/checker.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/manager/graph_var_manager.h"

namespace {
const std::string k1gHugePageFirstMallocFail = "The option ge.variableUse1gHugePage was set to 2, but the "
    "allocation of 1GB of huge page memory failed. Instead, an attempt was made to allocate 2M large pages. "
    "This error may affect the performance of the operator execution, resulting in the inability to fully "
    "maximize the performance benefits of this option. If you do not care about these performance benefits, "
    "the above error log can be ignored.";

const std::string k1gHugePageOnlyMallocFail = "The option ge.variableUse1gHugePage was set to 1, but the "
    "allocation of 1GB of super large page memory failed.";

std::string GetPgType(size_t pg_type) {
  if (pg_type == ge::kDrvMemPropPgType2M) {
    return "2M";
  } else if (pg_type == ge::kDrvMemPropPgType1G) {
    return "1G";
  }
  return "unknown";
}
// align_size是2的n次方
size_t AlignSize(const size_t size, const size_t align_size) {
  return ((size + align_size - 1U) & (~(align_size - 1U)));
}

uint8_t *Malloc(const std::string &purpose, const rtMemType_t memory_type, const uint32_t device_id,
                const size_t align_size, size_t &malloc_size) {
  void *memory_addr = nullptr;
  // 2M对齐减少内存浪费，如果不对齐，比如申请大小2.1M，最终驱动会申请4M内存，后面1.9M没使用
  // 对齐后这里就会申请4M，实际使用2.1M剩余的可以继续被复用
  if (malloc_size > align_size) {
    malloc_size = AlignSize(malloc_size, align_size);
  }

  if (rtMalloc(&memory_addr, malloc_size, memory_type, GE_MODULE_NAME_U16) != RT_ERROR_NONE) {
    memory_addr = nullptr;
  }

  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, ge::ToMallocMemInfo(purpose, memory_addr, device_id, GE_MODULE_NAME_U16).c_str(),
                          malloc_size);
  return reinterpret_cast<uint8_t *>(memory_addr);
}

static bool CompareSize(const ge::LogicalMemoryBlock &left, const ge::LogicalMemoryBlock &right) {
  return left.memory_size > right.memory_size;
}

bool HasIntersection(const ge::PageRecord &left, const ge::PageRecord &right) {
  return (left.head_offset < (right.head_offset + right.using_size)) &&
      (right.head_offset < (left.head_offset + left.using_size));
}

uint32_t TransMemType(const uint32_t mem_type) {
  const std::map<uint32_t, uint32_t> kRtMemTypeToDrvMemType = {{RT_MEMORY_HBM, 0U}, // 0: MEM_HBM_TYPE
                                                               {RT_MEMORY_DDR, 1U}, // 1: MEM_DDR_TYPE
                                                               {RT_MEMORY_P2P_HBM, 2U}, // 2: MEM_P2P_HBM_TYPE
                                                               {RT_MEMORY_P2P_DDR, 3U}, // 3: MEM_P2P_DDR_TYPE
                                                               {RT_MEMORY_TS, 4U}}; // 4: MEM_TS_DDR_TYPE
  const std::map<uint32_t, uint32_t>::const_iterator it = kRtMemTypeToDrvMemType.find(mem_type);
  if (it == kRtMemTypeToDrvMemType.cend()) {
    return 0U;
  } else {
    return it->second;
  }
}
}

namespace ge {
uint8_t *ActiveMemoryAllocator::MallocMemory(const std::string &purpose, LogicalMemorys &logical_memorys,
                                             std::vector<std::pair<uint8_t *, size_t>> &active_memorys,
                                             const uint32_t device_id) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  support_extend_memory_full_ = ge::VarManager::IsGeUseExtendSizeMemoryFull();
  used_count_++;
  int64_t memory_size = 0;
  GELOGI("Before malloc logical_memorys:");
  for (auto &logical_memory : logical_memorys) {
    GELOGI("%s", logical_memory.ToString().c_str());
    GE_ASSERT_TRUE(logical_memory.memory_size >= 0);
    if (logical_memory.alloc) {
      memory_size += logical_memory.memory_size;
    }
  }
  GE_ASSERT_TRUE(memory_size >= 0);
  memory_size_ = memory_size;
  for (auto &block : active_memorys_) {
    block.Reset();
  }

  // 用已有内存进行分配
  MallocByActiveMemorys(logical_memorys);
  // 合并连续在一起的内存
  MergeBlocks(logical_memorys);
  // 分配增量ActiveMemory
  MallocActiveMemorys(purpose, logical_memorys, active_memorys);

  GELOGI("After malloc logical_memorys:");
  bool all_success = logical_memorys.empty() ? false : true;
  for (auto &logical_memory : logical_memorys) {
    // 有一个没分配成功最后就返回nullptr
    if (logical_memory.active_addr == nullptr) {
      all_success = false;
    }
    GELOGI("%s", logical_memory.ToString().c_str());
  }

  GELOGI("ActiveMemoryAllocator::MallocMemory device_id = %u, size = %" PRIu64 " used_count = %zu.",
    device_id, memory_size, used_count_);
  return all_success ? logical_memorys[0].active_addr : nullptr;
}

Status ActiveMemoryAllocator::FreeMemory(const uint32_t device_id) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (used_count_ > 0U) {
    used_count_--;
  }

  if (used_count_ == 0U) {
    if (IsSupportExpandableMemoryFull()) {
      expandable_memory_allocator_.FreeMemory();
    } else {
      for (auto memory : active_memorys_) {
        (void) rtFree(reinterpret_cast<void *>(memory.active_addr));
      }
    }
    active_memorys_.clear();
    memory_size_ = 0;
    maximum_active_addr_ = nullptr;
    peak_size_ = 0U;
  }

  GELOGI("ActiveMemoryAllocator::FreeMemory used_count = %zu device_id = %u", used_count_, device_id);
  return ge::SUCCESS;
}

int64_t ActiveMemoryAllocator::MemorySize() {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  return peak_size_;
}

void ActiveMemoryAllocator::MallocByActiveMemorys(LogicalMemorys &logical_memorys) {
  // 按内存从大到小排序，减少内存浪费
  std::sort(logical_memorys.begin(), logical_memorys.end(), CompareSize);
  for (auto &block : active_memorys_) {
    for (auto &logical_memory : logical_memorys) {
      if (logical_memory.active_addr == reinterpret_cast<uint8_t *>(logical_memory.logical_addr)
          || (logical_memory.active_addr == nullptr)) {
        logical_memory.active_addr = block.Malloc(static_cast<size_t>(logical_memory.memory_size));
      }
    }
  }
  // 恢复原来按逻辑地址排序
  std::sort(logical_memorys.begin(), logical_memorys.end());
}

void ActiveMemoryAllocator::MergeBlocks(LogicalMemorys &logical_memorys) const {
  if (logical_memorys.size() <= 1U) {
    return;
  }
  LogicalMemorys merged_memorys;
  auto logical_memory = logical_memorys[0];
  for (size_t i = 1U; i < logical_memorys.size(); ++i) {
    // logic_memory_base和memory_base都满足连续的，进行合并，零拷贝段不参加合并
    const auto &current = logical_memorys[i];
    const auto &pre = logical_memorys[i - 1U];
    if (current.alloc && (current.logical_addr == (pre.logical_addr + pre.memory_size))
        && ((current.active_addr == (pre.active_addr + pre.memory_size))
            || ((current.active_addr == nullptr) && (pre.active_addr == nullptr))) && (!current.is_zero_copy)) {
      logical_memory.memory_size += current.memory_size;
    } else {
      merged_memorys.emplace_back(logical_memory);
      logical_memory = current;
    }
    // 最后一个直接添加到合并列表
    if (i == (logical_memorys.size() - 1U)) {
      merged_memorys.emplace_back(logical_memory);
    }
  }

  logical_memorys = merged_memorys;
}

void ActiveMemoryAllocator::MallocActiveMemorys(const std::string &purpose, LogicalMemorys &logical_memorys,
                                                std::vector<std::pair<uint8_t *, size_t>> &active_memorys) {
  for (auto &logical_memory : logical_memorys) {
    if (logical_memory.active_addr != nullptr) {
      continue;
    }
    // Malloc memory
    if (logical_memory.alloc) {
      size_t malloc_size = static_cast<size_t>(logical_memory.memory_size);
      if (support_extend_memory_full_) {
        logical_memory.active_addr = expandable_memory_allocator_.MallocMemory(purpose, malloc_size, true);
      }
      // 有可能驱动不支持内存扩展
      if (logical_memory.active_addr == nullptr) {
        logical_memory.active_addr = Malloc(purpose, memory_type_, device_id_, kDrvPageSize, malloc_size);
      }

      if (logical_memory.active_addr == nullptr) {
        return;
      }
      if ((logical_memory.active_addr + malloc_size) > maximum_active_addr_) {
        maximum_active_addr_ = logical_memory.active_addr + malloc_size;
      }

      ActiveMemoryBlock
          active_memory(logical_memory.active_addr, malloc_size, static_cast<size_t>(logical_memory.memory_size));
      active_memorys_.emplace_back(std::move(active_memory));
      std::sort(active_memorys_.begin(), active_memorys_.end());
    } else {
      // 防止地址交叉，取最大内存地址
      logical_memory.active_addr = maximum_active_addr_;
    }
  }

  size_t total_size = 0U;
  size_t used_size = 0U;
  for (auto &active_memory : active_memorys_) {
    GELOGI("%s", active_memory.ToString().c_str());
    total_size += active_memory.total_size;
    used_size += active_memory.used_size;
    if (active_memory.used_size == 0U) {
      continue;
    }
    // 纯静态图加载和动态图执行存在并发，这里需要把纯静态图用到的物理内存提前绑定（之前分配的纯静态图内存）
    // 并记录到active_memorys中，后面执行阶段同步申请和回收处理
    if (support_extend_memory_full_ && (!active_memory.new_add)) {
      size_t reuse_size = 0U;
      (void) expandable_memory_allocator_.GetPyhsicalMemoryAllocator().MallocPhysicalMemory(purpose,
          active_memory.active_addr, active_memory.used_size, reuse_size);
    }
    active_memorys.emplace_back(active_memory.active_addr, active_memory.used_size);
    (void) rtMemset(active_memory.active_addr, active_memory.used_size, 0U, active_memory.used_size);
  }
  if (used_size > peak_size_) {
    peak_size_ = used_size;
  }
  GELOGI("Total active memory alloc_size:%zu, used_size:%zu, peak_size:%zu", total_size, used_size, peak_size_);
}

bool ActiveMemoryAllocator::IsSupportExpandableMemoryFull() {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  return expandable_memory_allocator_.IsSupportExpandableMemory() && support_extend_memory_full_;
}

// 用实际图的内存大小进行申请和释放
Status ActiveMemoryAllocator::MallocPhysicalMemory(const std::string &purpose,
                                                   const std::vector<std::pair<uint8_t *, size_t>> &active_memorys) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  size_t malloced_size = 0U;
  size_t reuse_size = 0U;
  for (const auto &addr : active_memorys) {
    const Status ret = expandable_memory_allocator_.GetPyhsicalMemoryAllocator().MallocPhysicalMemory(purpose,
        addr.first, addr.second, reuse_size);
    GE_ASSERT_SUCCESS(ret, "ret = %u.", ret);
    malloced_size += addr.second;
  }
  GELOGI("ActiveMemoryAllocator::MallocPhysicalMemory memory_size:%zu device_id = %u memory_type = %u",
         malloced_size, device_id_, memory_type_);
  return SUCCESS;
}

void ActiveMemoryAllocator::Recycle(const std::vector<std::pair<uint8_t *, size_t>> &active_memorys) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  size_t freed_size = 0U;
  for (const auto &addr : active_memorys) {
    (void) expandable_memory_allocator_.GetPyhsicalMemoryAllocator().FreePhysicalMemory(addr.first, addr.second);
    freed_size += addr.second;
  }
  GELOGI("ActiveMemoryAllocator::Recycle memory_size:%zu device_id = %u memory_type = %u",
         freed_size, device_id_, memory_type_);
}

uint8_t *ExpandableActiveMemoryAllocator::MallocMemory(const std::string &purpose, int64_t memory_size,
                                                       bool incremental) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GE_WARN_ASSERT(support_reserve_mem_address_);
  GE_ASSERT_TRUE(memory_size >= 0);
  // incremental为true是再扩展memory_size大小，否则是扩展到memory_size的大小
  const size_t malloc_size =
      incremental ? (used_memory_size_ + static_cast<size_t>(memory_size)) : static_cast<size_t>(memory_size);
  if (MallocVirtualMemory(malloc_size) != SUCCESS) {
    support_reserve_mem_address_ = false;
    GELOGW("Maybe not support rtReserveMemAddress.");
    return nullptr;
  }
  GE_ASSERT_SUCCESS(MallocPhysicalMemoryAndMap(purpose, malloc_size));
  auto memory_addr = virtual_active_addr_;
  if (incremental) {
    memory_addr = virtual_active_addr_ + used_memory_size_;
    used_memory_size_ += static_cast<size_t>(memory_size);
  } else {
    used_count_++;
    used_memory_size_ = std::max(used_memory_size_, static_cast<size_t>(memory_size));
  }
  GELOGI("[%s] Total[virtual_memory_addr:%p virtual_memory_size:%zu, used_memory_size:%zu],"
         " current[memory_addr:%p memory_size:%lld incremental:%d]", purpose.c_str(), virtual_active_addr_,
         virtual_memory_size_, used_memory_size_, memory_addr, memory_size,
         static_cast<int32_t>(incremental));
  return memory_addr;
}

Status ExpandableActiveMemoryAllocator::FreeMemory() {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (used_count_ > 0U) {
    used_count_--;
  }

  if (used_count_ == 0U) {
    for (const auto &addr : mapped_memory_addrs_) {
      (void) active_memory_allocator_.FreePhysicalMemory(addr.first, addr.second);
    }
    mapped_memory_addrs_.clear();
    used_memory_size_ = 0U;
    if (virtual_active_addr_ != nullptr) {
      active_memory_allocator_.ReleaseVirtualMemory();
      virtual_active_addr_ = nullptr;
    }
  }
  GELOGI("ActiveMemoryAllocator::FreeMemory used_count = %zu device_id = %u memory_type = %u",
         used_count_, device_id_, memory_type_);
  return SUCCESS;
}

Status ExpandableActiveMemoryAllocator::MallocVirtualMemory(const size_t memory_size) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (virtual_active_addr_ == nullptr) {
    size_t free_mem = 0U;
    size_t total_mem_size = 0U;
    GE_ASSERT_RT_OK(rtMemGetInfoEx(RT_MEMORYINFO_HBM, &free_mem, &total_mem_size));
    if (total_mem_size == 0U) {
      rtMemInfoType_t mem_info_type = memory_type_ == RT_MEMORY_HBM ? RT_MEMORYINFO_DDR : RT_MEMORYINFO_DDR_P2P_HUGE;
      GE_ASSERT_RT_OK(rtMemGetInfoEx(mem_info_type, &free_mem, &total_mem_size));
    }
    (void) free_mem;
    GE_ASSERT_TRUE(memory_size <= total_mem_size, "memory_size: %zu, total_mem_size: %zu", memory_size, total_mem_size);
    virtual_memory_size_ = total_mem_size;
    virtual_active_addr_ = active_memory_allocator_.ReserveVirtualMemory(virtual_memory_size_, device_id_,
                                                                         memory_type_, share_phy_allocator_);
    GE_WARN_ASSERT(virtual_active_addr_ != nullptr);
    active_memory_allocator_.SetReuse(reuse_);
  }
  return SUCCESS;
}

Status ExpandableActiveMemoryAllocator::MallocPhysicalMemoryAndMap(const std::string &purpose,
                                                                   const size_t memory_size) {
  if (memory_size > used_memory_size_) {
    auto virtual_active_addr = virtual_active_addr_ + used_memory_size_;
    size_t reuse_size = 0U;
    GE_ASSERT_SUCCESS(active_memory_allocator_.MallocPhysicalMemory(purpose, virtual_active_addr,
                                                                    memory_size - used_memory_size_, reuse_size));
    mapped_memory_addrs_.emplace_back(virtual_active_addr, memory_size - used_memory_size_);
  }
  return SUCCESS;
}

MemBlock *FixedBaseExpandableAllocator::Malloc(size_t size) {
  const auto ptr = ex_active_allocator_.MallocMemory(kFixedBasePurpose, static_cast<int64_t>(size),
                                                                 false);
  GE_ASSERT_NOTNULL(ptr, "malloc failed, [%s], size: %zu", kFixedBasePurpose.c_str(), size);
  auto block = new (std::nothrow) MemBlock(*this, ptr, size);
  GE_ASSERT_NOTNULL(block);
  GELOGI("[%s] malloc, block: %p, addr: %p, size: %zu", kFixedBasePurpose.c_str(), block, ptr, size);
  return block;
}

// 优先使用mem_block->Free()会首先减计数，减为0后，再调用本函数
void FixedBaseExpandableAllocator::Free(MemBlock *mem_block) {
  if (mem_block != nullptr) {
    (void)ex_active_allocator_.FreeMemory();
    GELOGI("[%s] free, block: %p, addr: %p, size: %zu", kFixedBasePurpose.c_str(), mem_block,
           mem_block->GetAddr(), mem_block->GetSize());
    delete mem_block;
  }
}

ExpandableActiveMemoryAllocatorImp::~ExpandableActiveMemoryAllocatorImp() {
  try {
    ReleaseVirtualMemory();
  } catch (const std::exception &) {
    // do nothing
  }
}

uint8_t *ExpandableActiveMemoryAllocatorImp::ReserveVirtualMemory(size_t &virtual_memory_size,
                                                                  const uint32_t device_id,
                                                                  const rtMemType_t memory_type,
                                                                  const bool share_phy_allocator) {
  if (virtual_memory_addr_base_ != nullptr) {
    return virtual_memory_addr_base_;
  }
  // 动态shape扩展模式统一页表大小，variable和fm
  if ((page_size_!= kDrv1GPageSize) && ge::ModelUtils::IsGeUseExtendSizeMemory(true)) {
    page_size_bits_ = kLargePageSizeBits;
    page_size_ = (1U << page_size_bits_);
    page_size_mask_ = (page_size_ - 1U);
  }
  virtual_memory_size = AlignSize(virtual_memory_size, page_size_);
  virtual_memory_size_ = virtual_memory_size;

  GE_ASSERT_EQ((virtual_memory_size_ & page_size_mask_), 0U);
  const size_t memory_block_count = virtual_memory_size_ >> page_size_bits_;
  physical_memorys_.resize(memory_block_count);
  device_id_ = device_id;
  free_physical_memorys_.reserve(memory_block_count);
  pa_list_.reserve(memory_block_count);
  map_count_ = 0U;

  if (share_phy_allocator) {
    physical_memory_allocator_ =
      PhysicalMemoryAllocatorMgr::Instance().CreateAllocator(device_id, memory_type, page_size_);
  } else {
    physical_memory_allocator_ = ge::MakeShared<PhysicalMemoryAllocator>(device_id, memory_type);
  }
  GE_ASSERT_TRUE(physical_memory_allocator_ != nullptr);

  void *virtual_addr = nullptr;
  GE_WARN_ASSERT(physical_memory_allocator_->ReserveMemAddress(&virtual_addr, virtual_memory_size_) == SUCCESS);
  virtual_memory_addr_base_ = reinterpret_cast<uint8_t *>(virtual_addr);
  GE_WARN_ASSERT(virtual_memory_addr_base_ != nullptr);

  // 虚拟内存分配成功再Initialize，否则会导致physical_memory_allocator内存无法释放，有引用计数
  GE_ASSERT_SUCCESS(physical_memory_allocator_->Initialize(page_size_, memory_block_count));

  log_level_ = dlog_getlevel(GE_MODULE_NAME, nullptr);
  GEEVENT("virtual_active_addr_base:%p virtual_memory_size:%s device_id:%u, memory_type:%u, share_phy_allocator:%d.",
          virtual_memory_addr_base_, ActiveMemoryUtil::SizeToString(virtual_memory_size_).c_str(), device_id_,
          memory_type, share_phy_allocator);
  return virtual_memory_addr_base_;
}

void ExpandableActiveMemoryAllocatorImp::ReleaseVirtualMemory() noexcept {
  if (virtual_memory_addr_base_ != nullptr) {
    (void) FreePhysicalMemory(virtual_memory_addr_base_, virtual_memory_size_);
    (void) physical_memory_allocator_->Finalize(virtual_memory_addr_base_, virtual_memory_size_);
    GEEVENT("virtual_active_addr_base:%p virtual_memory_size:%zu device_id:%u.",
            virtual_memory_addr_base_, virtual_memory_size_, device_id_);
    virtual_memory_size_ = 0U;
    virtual_memory_addr_base_ = nullptr;
    for (auto physical_memorys : physical_memorys_) {
      if (physical_memorys.is_using || (physical_memorys.ref_count != 0U)
          || (physical_memorys.physical_map_count != 0U) || (physical_memorys.handle != nullptr)) {
        GELOGE(ge::FAILED, "index:%zu is_using:%d ref_count:%zu physical_map_count:%zu handle:%p",
               physical_memorys.index, physical_memorys.is_using, physical_memorys.ref_count,
               physical_memorys.physical_map_count, physical_memorys.handle);
      }
    }
    physical_memorys_.clear();
  }
}

Status ExpandableActiveMemoryAllocatorImp::GetIndex(const uint8_t *const virtual_memory_addr,
                                                    const size_t virtual_memory_size,
                                                    size_t &index_begin,
                                                    size_t &index_end,
                                                    size_t &end_remain_size) const {
  const size_t
      offset_begin = static_cast<size_t>(PtrToValue(virtual_memory_addr) - PtrToValue(virtual_memory_addr_base_));
  index_begin = offset_begin >> page_size_bits_;
  index_end = (offset_begin + virtual_memory_size) >> page_size_bits_;

  // |****|****|****|****|
  //           |****|
  //         begin end
  // end恰好处于边界点时，实际上用的是index - 1对应的物理内存
  if ((index_end >= 1U) && ((offset_begin + virtual_memory_size) & page_size_mask_) == 0U) {
    index_end -= 1U;
  }

  // |****|****|****|****|
  //              |******|
  const size_t physical_memorys_counts = physical_memorys_.size();
  if ((index_end >= physical_memorys_counts) && (physical_memorys_counts > 0U)) {
    if ((offset_begin + virtual_memory_size) > (physical_memorys_counts << page_size_bits_)) {
      GELOGE(FAILED, "virtual_memory_addr:%p virtual_memory_size:%zu is invalid.",
             virtual_memory_addr, virtual_memory_size);
      return FAILED;
    }
    index_end = physical_memorys_counts - 1U;
  }
  // 这里保证index合法性，后续使用时不需要再校验
  GE_ASSERT_TRUE(index_begin < physical_memorys_counts);
  GE_ASSERT_TRUE(index_end < physical_memorys_counts);

  // end_remain_size非0需要多申请一块
  if (index_begin < index_end) {
    // ref_count为0表示尾块空闲可以当一个完整块使用
    if (physical_memorys_[index_end].ref_count == 0U) {
      end_remain_size = 0U;
    } else {
      end_remain_size = (offset_begin + virtual_memory_size) & page_size_mask_;
    }
    GELOGI("index_begin:%zu index_end:%zu begin_remain_size:%zu end_remain_size:%zu ref_count:%zu", index_begin,
           index_end, offset_begin & page_size_mask_, end_remain_size, physical_memorys_[index_end].ref_count);
  }
  return SUCCESS;
}

void ExpandableActiveMemoryAllocatorImp::PutToFreeList(PhysicalMemoryInfo &physical_memory) {
  if ((!physical_memory.in_free_list) && (physical_memory.ref_count == 0U) && (!physical_memory.is_using)
      && (physical_memory.handle != nullptr)) {
    free_physical_memorys_.emplace_back(physical_memory.index);
    physical_memory.in_free_list = true;
  }
}

bool ExpandableActiveMemoryAllocatorImp::PopFromFreeList(size_t &index) {
  while(!free_physical_memorys_.empty()) {
    auto &physical_memory = physical_memorys_[free_physical_memorys_.back()];
    free_physical_memorys_.pop_back();
    physical_memory.in_free_list = false;
    if ((physical_memory.ref_count == 0U) && (!physical_memory.is_using) && (physical_memory.handle != nullptr)) {
      index = physical_memory.index;
      return true;
    }
  }
  return false;
}

Status ExpandableActiveMemoryAllocatorImp::MallocPhysicalMemory(uint8_t *const virtual_memory_addr,
    const std::vector<size_t> &pa_list, bool need_map) {
  const size_t pa_list_size = pa_list.size();
  GELOGI("pa_list_size:%zu index_begin:%zu index_end:%zu", pa_list_size, pa_list.front(), pa_list.back());
  GE_ASSERT_TRUE(!vapa_check_failed_, "ProcPageRecord failed during the last call, return failed immediately");
  auto unmap_func = [&pa_list, virtual_memory_addr, this](const size_t index) {
    for (size_t i = 0U; i < index; ++i) {
      auto &physical_memory = physical_memorys_[pa_list[i]];
      if (physical_memory.physical_map_count > 0U) {
        physical_memory.physical_map_count--;
      }
      physical_memory.is_using = false;
      void *map_addr = reinterpret_cast<void *>(virtual_memory_addr + (i << page_size_bits_));
      (void) physical_memory_allocator_->UnmapMem(map_addr, HanleToIndex(physical_memory.handle));
    }
  };

  for (size_t index = 0U; index < pa_list_size; ++index) {
    auto &phys_mem = physical_memorys_[pa_list[index]];
    if (!phys_mem.is_using) {
      phys_mem.is_using = true;
      if (!need_map) {
        continue;
      }
      void *map_addr = reinterpret_cast<void *>(virtual_memory_addr + (index << page_size_bits_));
      if (physical_memory_allocator_->MapMem(map_addr, HanleToIndex(phys_mem.handle)) != SUCCESS) {
        phys_mem.is_using = false;
        unmap_func(index);
        return FAILED;
      }
      HP_LOGD("MapMem addr:%p index:%zu size:%zu success, virtual_memory_addr:%p.",
              map_addr, phys_mem.index, page_size_, virtual_memory_addr);
      map_count_++;
      phys_mem.physical_map_count++;
      continue;
    }
    GELOGI("PHYSICAL_MEM_USING index:%zu ref_count:%zu is_using:%d", pa_list[index],
           phys_mem.ref_count, phys_mem.is_using);
    // index对应的物理内存被占用，回滚未被占用的物理内存状态
    for (size_t i = 0U; i < index; ++i) {
      auto &physical_memory = physical_memorys_[pa_list[i]];
      if (physical_memory.is_using) {
        physical_memory.is_using = false;
      }
      PutToFreeList(physical_memory);
    }
    return PHYSICAL_MEM_USING;
  }
  if (ProcPageRecordByPaList(virtual_memory_addr, pa_list, PageRecordAction::kAdd) != SUCCESS) {
    unmap_func(pa_list_size);
    (void)ProcPageRecordByPaList(virtual_memory_addr, pa_list, PageRecordAction::kDel);
    vapa_check_failed_ = true;
    GELOGE(FAILED, "malloc failed. virtual_memory_addr: %p, pa_list_size: %zu", virtual_memory_addr, pa_list_size);
    return FAILED;
  }
  return SUCCESS;
}

Status ExpandableActiveMemoryAllocatorImp::ProcessPhysicalMemoryUsing(size_t index_begin, size_t index_end,
    size_t index_using, size_t end_remain_size, size_t &reuse_size) {
  size_t roll_back_end = index_end;
  if ((roll_back_end > 0U) && (end_remain_size > 0U)) {
    roll_back_end--;
    // 回滚尾块内存状态
    auto &physical_memory = physical_memorys_[index_end];
    if (physical_memory.ref_count > 0U) {
      physical_memory.ref_count--;
    }
  }

  // 低地址没分内存，高地址内存被占用，跳过已经分配的内存块
  if (NeedAllocPa(index_begin)) {
    reuse_size = 0U;
    for (int64_t index = static_cast<int64_t>(roll_back_end);
         (index >= 0) && (physical_memorys_[index].handle != nullptr); --index) {
      reuse_size += page_size_;
    }
    reuse_size += end_remain_size;
  }

  // 回滚其它内存块状态
  size_t roll_back_size = end_remain_size;
  for (size_t index = roll_back_end; ((index > index_using) && (roll_back_size < reuse_size)); --index) {
    auto &free_physical_memory = physical_memorys_[index];
    if ((free_physical_memory.ref_count > 0U) && (free_physical_memory.is_using)) {
      free_physical_memory.ref_count--;
      free_physical_memory.is_using = false;
      if (!free_physical_memory.in_free_list) {
        PutToFreeList(free_physical_memory);
      }
    }
    roll_back_size += page_size_;
  }
  return PHYSICAL_MEM_USING;
}

Status ExpandableActiveMemoryAllocatorImp::ProcessNewVa(size_t index_end, size_t end_remain_size, size_t new_va_size,
    size_t &reuse_size) {
  for (const auto index : pa_list_) {
    auto &physical_memory = physical_memorys_[index];
    if (physical_memory.ref_count > 0U) {
      physical_memory.ref_count--;
    }
    if (physical_memory.is_using) {
      physical_memory.is_using = false;
    }
  }

  if (end_remain_size > 0U) {
    auto &physical_memory = physical_memorys_[index_end];
    if (physical_memory.ref_count > 0U) {
      physical_memory.ref_count--;
    }
  }

  reuse_size += end_remain_size;
  GE_ASSERT_EQ(new_va_size, pa_list_.size());
  GELOGI("pa_list_size:%zu", pa_list_.size());
  return NEW_VA;
}

size_t ExpandableActiveMemoryAllocatorImp::RecyclePhysicalMemory(size_t new_va_count, size_t end_remain_size,
    size_t index_end, size_t &index_begin, bool &recycle) {
  size_t index = 0U;
  size_t recycle_count = 0U;
  recycle = (((theory_min_size_ * kRatioBase) / (physical_memory_size_ + page_size_)) < kTheoryRatio);
  while(recycle && (pa_list_.size() < new_va_count) && PopFromFreeList(index)) {
    auto &free_physical_memory = physical_memorys_[index];
    free_physical_memory.is_using = true;
    pa_list_.emplace_back(free_physical_memory.index);
    recycle_count++;
  }

  if (recycle_count > 0U) {
    index_begin = index_end - (new_va_count - recycle_count);
    // 尾块完整可用
    if ((end_remain_size == 0U) && (index_begin < index_end)) {
      index_begin++;
    }
  }
  GELOGI("Current physical_memory_size:%zu theory_size:%zu page_size:%zu reach theory rate:%.2f%s recycle:%d "
         "new_va_count:%zu recycle_count:%zu", physical_memory_size_, theory_size_, page_size_,
         (kRatioBase * static_cast<float>(theory_min_size_)) / static_cast<float>(physical_memory_size_ + page_size_),
         "%", recycle, new_va_count, recycle_count);
  return recycle_count;
}

void ExpandableActiveMemoryAllocatorImp::ReleasePhysicalMemory(PhysicalMemoryInfo &physical_memory) {
  if (physical_memory.ref_count > 0U) {
    physical_memory.ref_count--;
    if (physical_memory.ref_count == 0U) {
      ReleasePhysicalMemoryByIndex(physical_memory.index);
    }
  }
  if (physical_memory.ref_count == 0U) {
    physical_memory.is_using = false;
  }
}

Status ExpandableActiveMemoryAllocatorImp::MallocPhysicalMemoryByIndex(const std::string &purpose, size_t index,
    size_t index_end, bool need_alloc_pa, size_t &reuse_size) {
  auto release_func = [index_end, this](const size_t index_begin) {
    for (int64_t i = static_cast<int64_t>(index_end); i > static_cast<int64_t>(index_begin); --i) {
      ReleasePhysicalMemory(physical_memorys_[i]);
    }

    // 释放回收的空洞对应的物理内存
    for (const auto idx : pa_list_) {
      ReleasePhysicalMemory(physical_memorys_[idx]);
    }
  };

  auto &malloc_physical_memory = physical_memorys_[index];
  void *map_addr = reinterpret_cast<void *>(virtual_memory_addr_base_ + (index << page_size_bits_));
  if (physical_memory_allocator_->MallocPhysical(purpose, malloc_physical_memory.last_pa_index, map_addr, reuse_)
      != SUCCESS) {
    release_func(index);
    return FAILED;
  }

  if (physical_memory_allocator_->MapMem(map_addr, malloc_physical_memory.last_pa_index) != SUCCESS) {
    (void) physical_memory_allocator_->FreePhysical(malloc_physical_memory.last_pa_index);
    malloc_physical_memory.last_pa_index = kInvalidIndex;
    release_func(index);
    return FAILED;
  }
  map_count_++;
  malloc_physical_memory.index = index;
  malloc_physical_memory.handle = IndexToHanle(malloc_physical_memory.last_pa_index);
  malloc_physical_memory.ref_count = 1UL;
  malloc_physical_memory.is_using = true;
  malloc_physical_memory.physical_map_count = 1U;
  malloc_physical_memory.is_physical_recycle = false;
  HP_LOGD("MapMem addr:%p index:%zu last_pa_index:%zu size:%zu success, total physical_memory_size:%zu.",
          map_addr, index, malloc_physical_memory.last_pa_index, page_size_, physical_memory_size_);

  physical_memory_size_ += page_size_;
  reuse_size += page_size_;
  if (need_alloc_pa) {
    pa_list_.emplace_back(index);
  }
  return SUCCESS;
}

Status ExpandableActiveMemoryAllocatorImp::MallocPhysicalMemory(const std::string &purpose,
    const uint8_t *const virtual_memory_addr, const size_t virtual_memory_size, size_t &reuse_size) {
  if (!IsValidVirtualAddr(virtual_memory_addr)) {
    return SUCCESS;
  }
  GE_ASSERT_TRUE(!vapa_check_failed_, "ProcPageRecord failed during the last call, return failed immediately");
  size_t index_begin = 0U;
  size_t index_end = 0U;
  size_t end_remain_size = 0U;
  GE_ASSERT_SUCCESS(GetIndex(virtual_memory_addr, virtual_memory_size, index_begin, index_end, end_remain_size));

  pa_list_.clear();
  size_t recycle_count = 0U;
  bool recycle = false;
  size_t new_va_count = AlignSize(virtual_memory_size, page_size_) >> page_size_bits_;
  for (int64_t index = static_cast<int64_t>(index_end); index >= static_cast<int64_t>(index_begin); --index) {
    auto &physical_memory = physical_memorys_[index];
    if (physical_memory.handle == nullptr) {
      // 物理内存被回收后会，需要直接给VA分配内存并绑定，不能创建NEW_VA
      if ((!physical_memory.is_physical_recycle) && (!recycle)) {
        recycle_count = RecyclePhysicalMemory(new_va_count, end_remain_size, index_end, index_begin, recycle);
      }
      // 连续可用内存块加回收的内存块已经满足申请内存大小直接退出，否则会多分一块内存
      if (((recycle_count > 0U) && (pa_list_.size() == new_va_count)) || (index < static_cast<int64_t>(index_begin))) {
        HP_LOGD("recycle_count:%zu pa_list_size:%zu new_va_count%:zu index:%" PRId64 " index_begin:%zu", recycle_count,
                pa_list_.size(), new_va_count, index, index_begin);
        break;
      }
      GE_ASSERT_SUCCESS(MallocPhysicalMemoryByIndex(purpose, index, index_end, NeedAllocPa(index_begin), reuse_size));
    } else {
      // end_remain_size 不为0 ref_count为0，整个尾块也可用，否则部分可用
      HP_LOGD("index:%zu ref_count:%zu, is_using:%d, total physical_memory_size:%zu.",
              index, physical_memory.ref_count, physical_memory.is_using, physical_memory_size_);
      if ((physical_memory.ref_count > 0U) && IsBoundary(index, index_begin, index_end)) {
        physical_memory.ref_count++;
        continue;
      }
      if (physical_memory.is_using) {
        HP_LOGD("PHYSICAL_MEM_USING index:%zu ref_count:%zu.", index, physical_memory.ref_count);
        // 默认跳过整个需要分配的内存大小
        reuse_size = virtual_memory_size;
        return ProcessPhysicalMemoryUsing(index_begin, index_end, index, end_remain_size, reuse_size);
      }

      physical_memory.is_using = true;
      physical_memory.ref_count++;
      reuse_size += page_size_;
      if (NeedAllocPa(index_begin)) {
        pa_list_.emplace_back(index);
      }
    }
  }
  GELOGI("virtual_memory_addr:%p virtual_memory_size:%zu success, total physical_memory_size:%zu.",
         virtual_memory_addr, virtual_memory_size, physical_memory_size_);
  if (recycle && (recycle_count > 0U)) {
    return ProcessNewVa(index_end, end_remain_size, new_va_count, reuse_size);
  }
  GE_ASSERT_SUCCESS(ProcPageRecord(virtual_memory_addr, virtual_memory_size, PageRecordAction::kAdd), "malloc failed,"
                    " virtual_memory_addr: %p, virtual_memory_size: %zu", virtual_memory_addr, virtual_memory_size);
  return SUCCESS;
}

Status ExpandableActiveMemoryAllocatorImp::FreePhysicalMemory(uint8_t *const virtual_memory_addr,
                                                              const std::vector<size_t> &pa_list,
                                                              const bool reduce_ref,
                                                              const bool release) {
  size_t pa_list_size = pa_list.size();
  GELOGI("pa_list_size:%zu index_begin:%zu index_end:%zu reduce_ref:%d release:%d",
         pa_list_size, pa_list.front(), pa_list.back(), reduce_ref, release);
  GE_ASSERT_SUCCESS(ProcPageRecordByPaList(virtual_memory_addr, pa_list, PageRecordAction::kDel));
  for (size_t index = 0U; index < pa_list_size; ++index) {
    const auto malloc_index = pa_list[index];
    auto &physical_memory = physical_memorys_[malloc_index];
    if (physical_memory.handle == nullptr) {
      GELOGW("index:%zu physical_memory.handle is nullptr", malloc_index);
      continue;
    }
    if (reduce_ref && physical_memory.is_using) {
      physical_memory.is_using = false;
    }
    if (release) {
      // new va需要用新的index
      void *map_addr = reinterpret_cast<void *>(virtual_memory_addr + (index << page_size_bits_));
      (void) physical_memory_allocator_->UnmapMem(map_addr, HanleToIndex(physical_memory.handle));
      map_count_--;
      if (physical_memory.physical_map_count > 0U) {
        physical_memory.physical_map_count--;
      }
    } else {
      PutToFreeList(physical_memory);
    }
  }
  if (release) {
    (void) physical_memory_allocator_->ReleaseMemAddress(virtual_memory_addr, pa_list_size * page_size_);
    GEEVENT("virtual_memory_addr:%p virtual_memory_size:%zu device_id:%u", virtual_memory_addr,
            pa_list.size() * page_size_, device_id_);
  }
  return SUCCESS;
}

Status ExpandableActiveMemoryAllocatorImp::ProcPageRecord(const uint8_t *const malloc_addr,
                                                          const size_t malloc_size,
                                                          const PageRecordAction &action) {
  // only check vapa when info/debug level
  if (log_level_ > DLOG_INFO) {
    return SUCCESS;
  }
  size_t index_begin;
  size_t index_end;
  size_t end_remain_size;
  GE_ASSERT_SUCCESS(GetIndex(malloc_addr, malloc_size, index_begin, index_end, end_remain_size));
  (void)end_remain_size;

  // 检查失败需要释放内存，删除记录
  ge::ScopeGuard guarder([this, malloc_addr, malloc_size, index_end, index_begin]() {
    for (auto i = static_cast<int64_t>(index_end); i > static_cast<int64_t>(index_begin); --i) {
      ReleasePhysicalMemory(physical_memorys_[i]);
    }
    // 释放回收的空洞对应的物理内存
    for (const auto idx : pa_list_) {
      ReleasePhysicalMemory(physical_memorys_[idx]);
    }
    (void)ProcPageRecord(malloc_addr, malloc_size, PageRecordAction::kDel);
    vapa_check_failed_ = true;
  });

  for (auto index = index_begin; index <= index_end; ++index) {
    const uint8_t *const map_addr = virtual_memory_addr_base_ + (index << page_size_bits_);
    const auto head_offset = (index == index_begin) ? (PtrToValue(malloc_addr) - PtrToValue(map_addr)) : 0U;
    const auto using_size = (index == index_end) ?
        (PtrToValue(malloc_addr) + malloc_size - PtrToValue(map_addr) - head_offset) : (page_size_ - head_offset);
    PageRecord page_record{map_addr, static_cast<size_t>(head_offset), static_cast<size_t>(using_size),
                           malloc_addr, malloc_size};
    if (action == PageRecordAction::kAdd) {
      GE_ASSERT_SUCCESS(physical_memory_allocator_->AddPageRecord(HanleToIndex(physical_memorys_[index].handle),
          page_record), "virtual and physical page mapping check failed, base: %p, page_size_: %zu, index_begin: %zu,"
          " index_end: %zu, index: %zu",  virtual_memory_addr_base_, page_size_, index_begin, index_end, index);
    } else {
      physical_memory_allocator_->DelPageRecord(HanleToIndex(physical_memorys_[index].handle), page_record);
    }
  }
  guarder.Dismiss();
  return SUCCESS;
}

Status ExpandableActiveMemoryAllocatorImp::ProcPageRecordByPaList(const uint8_t *const malloc_addr,
                                                                  const std::vector<size_t> &pa_list,
                                                                  const PageRecordAction &action) {
  // only check vapa when info/debug level
  if (log_level_ > DLOG_INFO) {
    return SUCCESS;
  }
  const size_t pa_list_size = pa_list.size();
  const auto malloc_size = pa_list_size << page_size_bits_;
  for (size_t index = 0U; index < pa_list_size; ++index) {
    const auto &phys_mem = physical_memorys_[pa_list[index]];
    const uint8_t *const map_addr = malloc_addr + (index << page_size_bits_);
    PageRecord page_record{map_addr, 0U, page_size_, malloc_addr, malloc_size};
    if (action == PageRecordAction::kAdd) {
      GE_ASSERT_SUCCESS(physical_memory_allocator_->AddPageRecord(HanleToIndex(phys_mem.handle), page_record),
          "virtual and physical page mapping check failed, page_size_: %zu, pa_list_size: %zu, index: %zu",
          page_size_, pa_list_size, index);
    } else {
      physical_memory_allocator_->DelPageRecord(HanleToIndex(phys_mem.handle), page_record);
    }
  }
  return SUCCESS;
}

Status ExpandableActiveMemoryAllocatorImp::FreePhysicalMemory(const uint8_t *const virtual_memory_addr,
                                                              const size_t virtual_memory_size,
                                                              const bool reduce_ref,
                                                              const bool release) {
  if (!IsValidVirtualAddr(virtual_memory_addr)) {
    return SUCCESS;
  }
  GE_ASSERT_SUCCESS(ProcPageRecord(virtual_memory_addr, virtual_memory_size, PageRecordAction::kDel));
  size_t index_begin = 0U;
  size_t index_end = 0U;
  size_t end_remain_size = 0U;
  GE_ASSERT_SUCCESS(GetIndex(virtual_memory_addr, virtual_memory_size, index_begin, index_end, end_remain_size));
  for (size_t index = index_begin; index <= index_end; ++index) {
    auto &physical_memory = physical_memorys_[index];
    if (physical_memory.handle == nullptr) {
      continue;
    }
    if (reduce_ref && (physical_memory.ref_count > 0U)) {
      physical_memory.ref_count--;
      HP_LOGD("Reduce reference count index:%zu ref_count:%zu, is_using:%d, total physical_memory_size:%zu.",
             index, physical_memory.ref_count,
             (physical_memory.ref_count > 0U) ? physical_memory.is_using : false, physical_memory_size_);
      if (physical_memory.ref_count > 0U) {
        continue;
      }
      if (physical_memory.is_using) {
        physical_memory.is_using = false;
      }
      if (!physical_memory.in_free_list) {
        PutToFreeList(physical_memory);
      }
    }

    // new va还在占用时ref_count为0，is_using为true，此时不能释放
    if (release && (physical_memory.ref_count == 0U) && (!physical_memory.is_using)) {
      ReleasePhysicalMemoryByIndex(index);
    }
  }
  if (release || reduce_ref) {
    GELOGI("virtual_memory_addr:%p virtual_memory_size:%zu success, total physical_memory_size:%zu.",
           virtual_memory_addr, virtual_memory_size, physical_memory_size_);
  }
  return SUCCESS;
}

void ExpandableActiveMemoryAllocatorImp::ReleasePhysicalMemoryByIndex(const size_t index) {
  auto &physical_memory = physical_memorys_[index];
  GELOGD("index:%zu malloc index:%zu.", index, physical_memory.index);
  if (physical_memory.physical_map_count > 0U) {
    physical_memory.physical_map_count--;
  }
  if (physical_memory.handle == nullptr) {
    return;
  }

  void *map_addr = nullptr;
  if (physical_memory.physical_map_count == 0U) {
    map_addr = reinterpret_cast<void *>(virtual_memory_addr_base_ + (index << page_size_bits_));
    (void) physical_memory_allocator_->UnmapMem(map_addr, HanleToIndex(physical_memory.handle));
    map_count_--;
    (void) physical_memory_allocator_->FreePhysical(HanleToIndex(physical_memory.handle));
    if (physical_memory_size_ >= page_size_) {
      physical_memory_size_ -= page_size_;
    }
    physical_memory.handle = nullptr;
    physical_memory.is_physical_recycle = true;
    GELOGI("Unmap and free addr:%p size:%zu success, total physical_memory_size:%zu index:%zu physical_map_count:%zu.",
           map_addr, page_size_, physical_memory_size_, index, physical_memory.physical_map_count);
  }
}

std::string ActiveMemoryUtil::SizeToString(size_t size) {
  const size_t kG = 1024UL * 1024UL * 1024UL;
  const size_t kM = 1024UL * 1024UL;
  const size_t kK = 1024UL;
  const size_t kPrecision = 2U;
  std::string return_value;
  std::stringstream ss;
  ss << std::fixed << std::setprecision(kPrecision);
  if (size > kG) {
    double sizeG = static_cast<double>(size) / static_cast<double>(kG);
    ss << sizeG;
    return_value = std::to_string(size);
    return_value.append("(");
    return_value.append(ss.str());
    return_value.append("G)");
  } else if (size > kM) {
    double sizeM = static_cast<double>(size) / static_cast<double>(kM);
    ss << sizeM;
    return_value = std::to_string(size);
    return_value.append("(");
    return_value.append(ss.str());
    return_value.append("M)");
  } else if (size > kK) {
    double sizeK = static_cast<double>(size) / static_cast<double>(kK);
    ss << sizeK;
    return_value = std::to_string(size);
    return_value.append("(");
    return_value.append(ss.str());
    return_value.append("K)");
  } else {
    return_value = std::to_string(size);
  }
  return return_value;
}

size_t ActiveMemoryUtil::IntegerBitsNum(size_t page_size) {
  size_t bits_num = 0U;
  while (page_size > 1U) {
    page_size >>= 1U;
    bits_num++;
  }
  return bits_num;
}

Status PhysicalMemoryAllocator::Initialize(size_t page_size, size_t max_page_count) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  log_level_ = dlog_getlevel(GE_MODULE_NAME, nullptr);
  GELOGI("Current ref_count:%zu.", ref_count_);
  ref_count_++;
  if (ref_count_ != 1U) {
    GE_ASSERT_EQ(page_size_, page_size);
  } else {
    page_size_ = page_size;
    prop_.side = 1U; // device memory
    prop_.devid = device_id_;
    prop_.module_id = GE_MODULE_NAME_U16;
    prop_.pg_type = kDrvMemPropPgType2M;
    if (page_size == kDrv1GPageSize) {
      prop_.pg_type = kDrvMemPropPgType1G;
    }
    prop_.mem_type = TransMemType(memory_type_);
    prop_.reserve = 0U;
    if (max_page_count > physical_memorys_.capacity()) {
      physical_memorys_.reserve(max_page_count);
      free_physical_memorys_.reserve(max_page_count);
    }
  }
  return SUCCESS;
}

Status PhysicalMemoryAllocator::Finalize(uint8_t *const va, size_t size) {
  (void) ReleaseMemAddress(va, size);
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (ref_count_ > 0U) {
    ref_count_--;
  }
  HP_LOGI("Current ref_count:%zu.", ref_count_);
  for (auto &physical_memory : physical_memorys_) {
    // va,new va会map多次，要都unmap掉后才能释放物理内存
    if (ref_count_ != 0U) {
      continue;
    }

    if ((physical_memory.ref_count != 0U) || (!physical_memory.map_addrs.empty())) {
      GELOGE(FAILED, "PhysicalMemoryAllocator pa_index:%zu ref_count:%zu.",
             physical_memory.index, physical_memory.ref_count);
      continue;
    }

    FreePhysicalPage(physical_memory);
  }
  if (physical_memory_size_ == 0U) {
    physical_memorys_.clear();
  }
  return SUCCESS;
}

void PhysicalMemoryAllocator::FreePhysicalPage(PhysicalMemoryInfo &physical_memory) {
  if (physical_memory.handle != nullptr) {
    (void) rtFreePhysical(physical_memory.handle);
    physical_memory.handle = nullptr;
    physical_memory_size_ -= page_size_;
    HP_LOGI("rtFreePhysical success pa_index:%zu, current physical memory size:%zu, pg_type:%s",
            physical_memory.index, physical_memory_size_, GetPgType(prop_.pg_type).c_str());
  }
}

Status PhysicalMemoryAllocator::MallocPhysicalPage(const std::string &purpose, size_t &pa_index, const void *const va,
                                                   bool reuse) {
  PhysicalMemoryInfo malloc_physical_memory;
  pa_index = physical_memorys_.size();
  malloc_physical_memory.index = pa_index;
  malloc_physical_memory.ref_count = 1UL;

  rtDrvMemHandle handle = nullptr;
  auto ret = rtMallocPhysical(&handle, page_size_, &prop_, 0U);
  if (ret != RT_ERROR_NONE) {
    if (prop_.pg_type == kDrvMemPropPgType1G) {
      const bool use_1g_only = VarManager::IsVariableUse1gHugePageOnly();
      const auto failed_log = use_1g_only ? k1gHugePageOnlyMallocFail : k1gHugePageFirstMallocFail;
      REPORT_INNER_ERR_MSG("E19999", "call rtMallocPhysical failed, size:%zu, pg_type: %s, %s", page_size_,
                         GetPgType(prop_.pg_type).c_str(), failed_log.c_str());
      GELOGE(FAILED, "call rtMallocPhysical failed, size:%zu, pg_type: %s, %s", page_size_,
             GetPgType(prop_.pg_type).c_str(), failed_log.c_str());
      if (use_1g_only) {
        return FAILED;
      }
      prop_.pg_type = kDrvMemPropPgType2M;
      ret = rtMallocPhysical(&handle, page_size_, &prop_, 0U);
    }
  }
  if (ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "call rtMallocPhysical failed, size:%zu,  pg_type: %s", page_size_,
                       GetPgType(prop_.pg_type).c_str());
    GELOGE(FAILED, "call rtMallocPhysical failed, size:%zu, pg_type: %s", page_size_, GetPgType(prop_.pg_type).c_str());
    return FAILED;
  }
  malloc_physical_memory.handle = handle;

  physical_memorys_.emplace_back(std::move(malloc_physical_memory));
  physical_memory_size_ += page_size_;
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, ToMallocMemInfo(purpose, va, device_id_, GE_MODULE_NAME_U16).c_str(), page_size_);
  HP_LOGI("rtMallocPhysical success pa_index:%zu, current physical memory size:%zu reuse:%d, memory_type:%u,"
      " pg_type:%s.", pa_index, physical_memory_size_, reuse, memory_type_, GetPgType(prop_.pg_type).c_str());
  return SUCCESS;
}

Status PhysicalMemoryAllocator::MallocPhysical(const std::string &purpose, size_t &pa_index, void *const va,
                                               bool reuse) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (pa_index < physical_memorys_.size()) {
    auto &physical_memory = physical_memorys_[pa_index];
    // 原有内存空闲，直接复用，如果被占用需要先ummap掉再重新分配
    if (physical_memory.handle != nullptr) {
      if (physical_memory.ref_count == 0U) {
        pa_index = physical_memory.index;
        physical_memory.ref_count++;
        HP_LOGI("reuse self pa_index:%zu.", pa_index);
        return SUCCESS;
      } else {
        auto it = physical_memory.map_addrs.find(va);
        GE_ASSERT_TRUE(it != physical_memory.map_addrs.end());
        physical_memory.map_addrs.erase(it);
        (void) rtUnmapMem(va);
        HP_LOGI("rtUnmapMem success pa_index:%zu va:%p.", pa_index, va);
      }
    }
  }
  // 找其他可用空闲内存，已经被复用过的，直接丢弃
  while(reuse && (!free_physical_memorys_.empty())) {
    auto &physical_memory = physical_memorys_[free_physical_memorys_.back()];
    free_physical_memorys_.pop_back();
    physical_memory.in_free_list = false;
    if ((physical_memory.ref_count == 0U) && (physical_memory.handle != nullptr)) {
      physical_memory.ref_count++;
      HP_LOGI("get from pool pa_index:%zu.", physical_memory.index);
      pa_index = physical_memory.index;
      return SUCCESS;
    }
  }

  // 没有空闲内存尝试新分配
  GE_ASSERT_SUCCESS(MallocPhysicalPage(purpose, pa_index, va, reuse));
  return SUCCESS;
}

Status PhysicalMemoryAllocator::FreePhysical(size_t pa_index) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GE_ASSERT_TRUE(pa_index < physical_memorys_.size());
  auto &physical_memory = physical_memorys_[pa_index];
  if (physical_memory.ref_count > 0U) {
    physical_memory.ref_count--;
  }
  if ((!physical_memory.in_free_list) && (physical_memory.ref_count == 0U)
      && (physical_memory.handle != nullptr)) {
    free_physical_memorys_.emplace_back(physical_memory.index);
    physical_memory.in_free_list = true;
    HP_LOGI("put to pool success pa_index:%zu.", pa_index);
  } else {
    HP_LOGI("pa_index:%zu in_free_list:%d ref_count:%zu.", pa_index,
            physical_memory.in_free_list, physical_memory.ref_count);
  }
  return SUCCESS;
}

Status PhysicalMemoryAllocator::AddPageRecord(const size_t pa_index, const PageRecord &page_record) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GE_ASSERT_TRUE(pa_index < physical_memorys_.size(), "pa_index: %zu, physical_memorys_ size: %zu",
                 pa_index, physical_memorys_.size());
  auto &va_record_table = physical_memorys_[pa_index].va_record_table;
  if (!va_record_table.empty()) {
    const auto &first_record = *va_record_table.begin();
    GE_ASSERT_TRUE(first_record.map_va == page_record.map_va, "pa_index[%zu] is using. using va info: %s, new va info: "
                   "%s", pa_index, first_record.ToString().c_str(), page_record.ToString().c_str());
    for (const auto &using_va : va_record_table) {
      if (using_va.head_offset != page_record.head_offset) {
        GE_ASSERT_TRUE(!HasIntersection(using_va, page_record), "va intersection, pa_index[%zu]. using va info: %s, "
            "new va info: %s", pa_index, using_va.ToString().c_str(), page_record.ToString().c_str());
      }
    }
  }
  va_record_table.emplace_back(page_record);
  return SUCCESS;
}

void PhysicalMemoryAllocator::DelPageRecord(const size_t pa_index, const PageRecord &page_record) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (pa_index < physical_memorys_.size()) {
    auto &va_record_table = physical_memorys_[pa_index].va_record_table;
    for (auto iter = va_record_table.begin(); iter < va_record_table.end(); ++iter) {
      if ((iter->head_offset == page_record.head_offset) &&
          (iter->using_size == page_record.using_size)) {
        va_record_table.erase(iter);
        return;
      }
    }
  }
}

Status PhysicalMemoryAllocator::MapMem(void *const va, size_t pa_index) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GE_ASSERT_TRUE(pa_index < physical_memorys_.size());
  auto &physical_memory = physical_memorys_[pa_index];
  auto it = physical_memory.map_addrs.find(va);
  // 已经绑定过而且是空闲状态直接返回
  if (it != physical_memory.map_addrs.end()) {
    HP_LOGI("pa_index:%zu addr:%p has been mapped.", pa_index, *it);
    return SUCCESS;
  }
  if (rtMapMem(va, page_size_, 0U, physical_memory.handle, 0U) != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "call rtMapMem failed, map_addr:%p.", va);
    GELOGE(FAILED, "call rtMapMem failed, map_addr:%p.", va);
    return FAILED;
  }
  physical_memory.map_addrs.insert(va);
  HP_LOGI("rtMapMem success pa_index:%zu addr:%p.", pa_index, va);
  return SUCCESS;
}

Status PhysicalMemoryAllocator::UnmapMem(void *const va, size_t pa_index) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GE_ASSERT_TRUE(pa_index < physical_memorys_.size());
  auto &physical_memory = physical_memorys_[pa_index];
  auto it = physical_memory.map_addrs.find(va);
  // 延迟去绑定提高性能
  GE_ASSERT_TRUE(it != physical_memory.map_addrs.end());
  HP_LOGI("pa_index:%zu addr:%p will be delay unmapped.", pa_index, *it);
  return SUCCESS;
}

Status PhysicalMemoryAllocator::ReserveMemAddress(void **va, size_t size) const {
  GE_WARN_ASSERT(rtReserveMemAddress(va, size, 0U, nullptr, 1U) == RT_ERROR_NONE);
  HP_LOGI("rtReserveMemAddress success va:%p size:%zu, device:%u, memory_type:%u.", *va, size, device_id_, memory_type_);
  return SUCCESS;
}

Status PhysicalMemoryAllocator::ReleaseMemAddress(void *const va, size_t size) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  // 释放虚拟地址前需要先unmap
  for (auto &physical_memory : physical_memorys_) {
    for (auto it = physical_memory.map_addrs.begin(); it != physical_memory.map_addrs.end();) {
      const auto map_addr = *it;
      // 在虚拟地址段内的unmap掉
      const auto map_addr_value = PtrToValue(map_addr);
      const auto base_addr_value = PtrToValue(va);
      if ((map_addr_value >= base_addr_value) && ((map_addr_value - base_addr_value) < size)) {
        (void) rtUnmapMem(map_addr);
        it = physical_memory.map_addrs.erase(it);
        HP_LOGI("rtUnmapMem pa_index:%zu addr:%p.", physical_memory.index, map_addr);
      } else {
        ++it;
      }
      HP_LOGD("pa_index:%zu map_addrs:%zu addr:%p.", physical_memory.index, physical_memory.map_addrs.size(), map_addr);
    }
  }
  GE_WARN_ASSERT(rtReleaseMemAddress(va) == RT_ERROR_NONE);
  HP_LOGI("rtReleaseMemAddress success va:%p size:%zu, device:%u, memory_type:%u.", va, size, device_id_, memory_type_);
  return SUCCESS;
}

PhysicalMemoryAllocatorMgr &PhysicalMemoryAllocatorMgr::Instance() {
  static PhysicalMemoryAllocatorMgr instance;
  return instance;
}

std::shared_ptr<PhysicalMemoryAllocator> PhysicalMemoryAllocatorMgr::CreateAllocator(uint32_t device_id,
                                                                                     rtMemType_t memory_type,
                                                                                     size_t page_size) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto dev_it = physical_memory_allocators_.find(device_id);
  if (dev_it != physical_memory_allocators_.end()) {
    auto type_it = dev_it->second.find(memory_type);
    if (type_it != dev_it->second.end()) {
      auto it = type_it->second.find(page_size);
      if (it != type_it->second.end()) {
        GELOGI("Reuse PhysicalMemoryAllocator success device id:%u, memory_type: %u, page_size: %zu.",
               device_id, memory_type, page_size);
        return it->second;
      }
    }
  }

  auto allocator = ge::MakeShared<PhysicalMemoryAllocator>(device_id, memory_type);
  GE_ASSERT_TRUE(allocator != nullptr);
  physical_memory_allocators_[device_id][memory_type][page_size] = allocator;
  GELOGI("Create PhysicalMemoryAllocator success device id:%u, memory_type: %u, page_size: %zu.",
         device_id, memory_type, page_size);
  return allocator;
}

template <>
SessionMemAllocator<FixedBaseExpandableAllocator> &SessionMemAllocator<FixedBaseExpandableAllocator>::Instance() {
  static SessionMemAllocator session_allocator;
  return session_allocator;
}

template <>
SessionMemAllocator<ActiveMemoryAllocator> &SessionMemAllocator<ActiveMemoryAllocator>::Instance() {
  static SessionMemAllocator session_allocator;
  return session_allocator;
}

template <>
SessionMemAllocator<ExpandableActiveMemoryAllocator> &SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance() {
  static SessionMemAllocator session_allocator;
  return session_allocator;
}
}  // namespace ge

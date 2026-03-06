/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_MANAGER_ACTIVE_MEMORY_ALLOCATOR_H_
#define GE_GRAPH_MANAGER_ACTIVE_MEMORY_ALLOCATOR_H_

#include <iostream>
#include <sstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <list>
#include <set>
#include <cxxabi.h>

#include "framework/common/debug/ge_log.h"
#include "graph/node.h"
#include "graph/def_types.h"
#include "runtime/mem.h"
#include "runtime/mem_allocator.h"
#include "graph/manager/mem_manager.h"
#include "common/util/mem_utils.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/checker.h"

// 提高日志级别判断性能
#define HP_LOGD(fmt, ...)                                                                                       \
  do {                                                                                                          \
    if (DLOG_DEBUG >= log_level_) {                                                                             \
      auto class_name = abi::__cxa_demangle(typeid(*this).name(), nullptr, nullptr, nullptr);                   \
      if (class_name != nullptr) {                                                                              \
        dlog_debug(GE_MODULE_NAME, "%" PRIu64 " %s::%s:" fmt, GeLog::GetTid(),                                          \
            class_name,  __FUNCTION__, ##__VA_ARGS__);                                                          \
        free(class_name);                                                                                       \
      }                                                                                                         \
    }                                                                                                           \
  } while (false)

#define HP_LOGI(fmt, ...)                                                                                       \
  do {                                                                                                          \
    if (DLOG_INFO >= log_level_) {                                                                              \
      auto class_name = abi::__cxa_demangle(typeid(*this).name(), nullptr, nullptr, nullptr);                   \
      if (class_name != nullptr) {                                                                              \
        dlog_info(GE_MODULE_NAME, "%" PRIu64 " %s::%s:" fmt, GeLog::GetTid(),                                           \
            class_name,  __FUNCTION__, ##__VA_ARGS__);                                                          \
        free(class_name);                                                                                       \
      }                                                                                                         \
    }                                                                                                           \
  } while (false)

namespace ge {
constexpr int32_t kSizeWidth = 10;
constexpr int32_t kAddrWidth = 15;
constexpr size_t kLargePageSizeBits = 23U;
constexpr size_t kLargePageSize = 1U << kLargePageSizeBits; // 8U * 1024U * 1024U
constexpr size_t kLargePageSizeMask = kLargePageSize - 1U;
constexpr size_t kDrvPageSizeBits = 21U;
constexpr size_t kDrv1GPageSizeBits = 30U;
constexpr size_t kDrvPageSize = 1U << kDrvPageSizeBits; // 2U * 1024U * 1024U
constexpr size_t kDrv1GPageSize = 1U << kDrv1GPageSizeBits; // 1U *1024 * 1024U * 1024U
constexpr int32_t PHYSICAL_MEM_USING = 2;
constexpr int32_t NEW_VA = 3;
constexpr size_t kRatioBase = 100U;
constexpr size_t kTheoryRatio = 99U;
constexpr size_t kInvalidIndex = std::numeric_limits<size_t>::max();
constexpr size_t kHandleBase = 1U;
const std::string kFixedBasePurpose = "fixed_base_expandable_memory";
constexpr size_t kDrvMemPropPgType2M = 1U;
constexpr size_t kDrvMemPropPgType1G = 2U;
class ActiveMemoryUtil {
 public:
  static std::string SizeToString(size_t size);
  static size_t IntegerBitsNum(size_t page_size);
};

struct LogicalMemoryBlock {
  int64_t logical_addr;
  int64_t memory_size;
  uint8_t *active_addr;
  bool alloc;
  rtMemType_t memory_type;
  bool is_zero_copy;

  LogicalMemoryBlock(int64_t logical_addr_tmp, int64_t memory_size_tmp, bool alloc_tmp = true,
                     bool is_zero_copy_tmp = false)
      : logical_addr(logical_addr_tmp),
        memory_size(memory_size_tmp),
        active_addr(nullptr),
        alloc(alloc_tmp),
        memory_type(RT_MEMORY_HBM),
        is_zero_copy(is_zero_copy_tmp) {}

  friend bool operator<(const LogicalMemoryBlock &left, const LogicalMemoryBlock &right) noexcept {
    return left.logical_addr < right.logical_addr;
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << "LogicalMemoryBlock logical_addr:" << std::left << std::setw(kSizeWidth) << logical_addr
       << " memory_size:" << std::setw(kSizeWidth) << memory_size
       << " active_addr:" << std::setw(kAddrWidth) << static_cast<void *>(active_addr)
       << " memory_type:" << memory_type << ", alloc:" << alloc;
    return ss.str();
  }
};

using LogicalMemorys = std::vector<LogicalMemoryBlock>;

struct ActiveMemoryBlock {
  uint8_t *active_addr;
  size_t total_size;
  size_t used_size;
  bool new_add;

  ActiveMemoryBlock(uint8_t *const active_addr_tmp, size_t total_size_tmp, size_t used_size_tmp)
      : active_addr(active_addr_tmp),
        total_size(total_size_tmp),
        used_size(used_size_tmp),
        new_add(true) {}

  uint8_t *Malloc(size_t memory_size) {
    uint8_t *addr = nullptr;
    if ((used_size <= total_size) && (memory_size <= (total_size - used_size))) {
      addr = active_addr + used_size;
      used_size += memory_size;
    }
    return addr;
  }

  void Reset() {
    used_size = 0U;
    new_add = false;
  }

  // 按内存块大小从大到小排序
  friend bool operator<(const ActiveMemoryBlock &left, const ActiveMemoryBlock &right) noexcept {
    return left.total_size > right.total_size;
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << "ActiveMemoryBlock active_addr_begin:"
       << std::left << std::setw(kAddrWidth) << static_cast<void *>(active_addr)
       << " active_addr_end:" << std::left << std::setw(kAddrWidth) << static_cast<void *>(active_addr + total_size)
       << " total_size:" << std::setw(kSizeWidth) << total_size
       << " used_size:" << std::setw(kSizeWidth) << used_size
       << " new_add:" << new_add;
    return ss.str();
  }
};

using ActiveMemorys = std::vector<ActiveMemoryBlock>;

// for vapa check
struct PageRecord {
  std::string ToString() const {
    std::stringstream ss;
    ss << "map_va: " << static_cast<const void *>(map_va)
       << ", head_offset: " << std::dec << head_offset << ", using_size: " << using_size
       << ", malloc_addr: " << static_cast<const void *>(malloc_addr)
       << ", malloc_size: " << std::dec << malloc_size;
    return ss.str();
  }
  const uint8_t *map_va;       // page mapped va
  size_t head_offset;          // for head page,  head_offset = malloc_addr - map_va; otherwise, head_offset = 0
  size_t using_size;           // for head/tail page, using_size <= page_size; otherwise, using_size = page_size
  const uint8_t *malloc_addr;  // for dfx
  size_t malloc_size;          // for dfx
};
using VaRecordTable = std::vector<PageRecord>;

enum class PageRecordAction {
  kAdd,
  kDel,
};

struct PhysicalMemoryInfo {
  rtDrvMemHandle handle = nullptr;
  size_t ref_count = 0U;
  size_t index = 0U;
  size_t physical_map_count = 0U;
  bool in_free_list = false;
  bool is_using = false;
  bool is_physical_recycle = false;
  size_t last_pa_index = kInvalidIndex; //上次分配的物理内存在物理内存池里的index
  std::set<void *> map_addrs;
  VaRecordTable va_record_table; // for va pa check
};

// 会被多线程调用
class PhysicalMemoryAllocator {
 public:
  explicit PhysicalMemoryAllocator(uint32_t device_id, rtMemType_t memory_type = RT_MEMORY_HBM)
      : device_id_(device_id), memory_type_(memory_type), ref_count_(0U), page_size_(kLargePageSize),
        physical_memory_size_(0U), log_level_(DLOG_ERROR), prop_() {}
  Status Initialize(size_t page_size, size_t max_page_count);
  Status Finalize(uint8_t *const va, size_t size);
  Status MallocPhysical(const std::string &purpose, size_t &pa_index, void *const va, bool reuse);
  Status FreePhysical(size_t pa_index);
  Status MapMem(void *const va, size_t pa_index);
  Status UnmapMem(void *const va, size_t pa_index);
  Status ReserveMemAddress(void **va, size_t size) const;
  Status ReleaseMemAddress(void *const va, size_t size);
  // for va pa check
  Status AddPageRecord(const size_t pa_index, const PageRecord &page_record);
  void DelPageRecord(const size_t pa_index, const PageRecord &page_record);

 private:
  void FreePhysicalPage(PhysicalMemoryInfo &physical_memory);
  Status MallocPhysicalPage(const std::string &purpose, size_t &pa_index, const void *const va, bool reuse);
 private:
  uint32_t device_id_;
  rtMemType_t memory_type_;
  size_t ref_count_;
  size_t page_size_;
  size_t physical_memory_size_;
  int32_t log_level_;
  std::recursive_mutex mutex_;
  DrvMemProp prop_;
  std::vector<PhysicalMemoryInfo> physical_memorys_;
  std::vector<size_t> free_physical_memorys_;
};

// 会被多线程调用,device个数有限，可以最后清理allocator对象
class PhysicalMemoryAllocatorMgr {
 public:
  explicit PhysicalMemoryAllocatorMgr() = default;
  ~PhysicalMemoryAllocatorMgr() = default;
  static PhysicalMemoryAllocatorMgr &Instance();
  std::shared_ptr<PhysicalMemoryAllocator> CreateAllocator(uint32_t device_id, rtMemType_t memory_type = RT_MEMORY_HBM,
                                                           size_t page_size = kDrvPageSize);

 private:
  std::recursive_mutex mutex_;
  // device_id, memory_type, page_size
  std::map<uint32_t, std::map<rtMemType_t, std::map<size_t, std::shared_ptr<PhysicalMemoryAllocator>>>>
      physical_memory_allocators_;
};

class ExpandableActiveMemoryAllocatorImp {
 public:
  explicit ExpandableActiveMemoryAllocatorImp(size_t page_size_bits)
      : device_id_(0U),
        virtual_memory_addr_base_(nullptr),
        virtual_memory_size_(0U),
        physical_memory_size_(0U),
        page_size_bits_(page_size_bits),
        page_size_(1U << page_size_bits_),
        page_size_mask_(page_size_ - 1U),
        reuse_(true),
        vapa_check_failed_(false) {}

  ~ExpandableActiveMemoryAllocatorImp();

  // forbidden copy
  ExpandableActiveMemoryAllocatorImp(const ExpandableActiveMemoryAllocatorImp &) = delete;
  ExpandableActiveMemoryAllocatorImp &operator=(const ExpandableActiveMemoryAllocatorImp &) & = delete;

  // forbidden move
  ExpandableActiveMemoryAllocatorImp(const ExpandableActiveMemoryAllocatorImp &&) = delete;
  ExpandableActiveMemoryAllocatorImp &operator=(const ExpandableActiveMemoryAllocatorImp &&) & = delete;

  Status MallocPhysicalMemory(const std::string &purpose,
                              const uint8_t *const virtual_memory_addr,
                              const size_t virtual_memory_size,
                              size_t &reuse_size);

  Status MallocPhysicalMemory(uint8_t *const virtual_memory_addr,
                              const std::vector<size_t> &pa_list,
                              bool need_map = false);

  Status FreePhysicalMemory(const uint8_t *const virtual_memory_addr,
                            const size_t virtual_memory_size,
                            const bool reduce_ref = true,
                            const bool release = true);

  Status FreePhysicalMemory(uint8_t *const virtual_memory_addr,
                            const std::vector<size_t> &pa_list,
                            const bool reduce_ref = true,
                            const bool release = true);

  uint8_t *ReserveVirtualMemory(size_t &virtual_memory_size, const uint32_t device_id,
                                const rtMemType_t memory_type = RT_MEMORY_HBM,
                                const bool share_phy_allocator = true);

  void ReleaseVirtualMemory() noexcept;

  size_t ActiveMemorySize() const { return physical_memory_size_; }

  bool IsValidVirtualAddr(const uint8_t *const virtual_active_addr) const {
    return (virtual_memory_addr_base_ != nullptr) && (virtual_active_addr >= virtual_memory_addr_base_)
        && (static_cast<size_t>(virtual_active_addr - virtual_memory_addr_base_) < virtual_memory_size_);
  }

  std::vector<size_t> &GetPaList() {
    return pa_list_;
  }

  PhysicalMemoryAllocator &GetPhysicalMemoryAllocator() const {
    return *physical_memory_allocator_;
  }

  void SetTheorySize(size_t theory_size) {
    theory_size_ = theory_size;
  }

  void SetTheoryMinSize(size_t theory_min_size) {
    theory_min_size_ = theory_min_size;
  }

  void SetReuse(bool reuse) {
    reuse_ = reuse;
  }

 private:
  Status GetIndex(const uint8_t *const virtual_memory_addr,
                  const size_t virtual_memory_size, size_t &index_begin, size_t &index_end,
                  size_t &end_remain_size) const;
  void ReleasePhysicalMemoryByIndex(const size_t index);
  void PutToFreeList(PhysicalMemoryInfo &physical_memory);
  bool PopFromFreeList(size_t &index);
  Status ProcessPhysicalMemoryUsing(size_t index_begin, size_t index_end, size_t index_using, size_t end_remain_size,
                                    size_t &reuse_size);
  Status ProcessNewVa(size_t index_end, size_t end_remain_size, size_t new_va_size, size_t &reuse_size);
  size_t RecyclePhysicalMemory(size_t new_va_count, size_t end_remain_size, size_t index_end, size_t &index_begin,
                               bool &recycle);
  Status MallocPhysicalMemoryByIndex(const std::string &purpose, size_t index, size_t index_end, bool need_alloc_pa,
                                     size_t &reuse_size);
  void ReleasePhysicalMemory(PhysicalMemoryInfo &physical_memory);

  // check pa va map
  Status ProcPageRecord(const uint8_t *const malloc_addr, const size_t malloc_size, const PageRecordAction &action);
  Status ProcPageRecordByPaList(const uint8_t *const malloc_addr, const std::vector<size_t> &pa_list,
                                const PageRecordAction &action);

  inline bool NeedAllocPa(size_t index_begin) const {
    return (physical_memorys_[index_begin].handle == nullptr);
  }

  inline bool IsBoundary(int64_t index, size_t index_begin, size_t index_end) const {
    return (static_cast<size_t>(index) == index_begin) || (static_cast<size_t>(index) == index_end);
  }

  // index为0时转成handle会被当作空指针，这里需要+kHandleBase
  rtDrvMemHandle IndexToHanle(size_t index) const {
    return reinterpret_cast<rtDrvMemHandle>(ValueToPtr(index + kHandleBase));
  }

  // 内部代码互转能保证合法性
  size_t HanleToIndex(const rtDrvMemHandle handle) const {
    return PtrToValue(handle) - kHandleBase;
  }

 private:
  uint32_t device_id_;
  uint8_t *virtual_memory_addr_base_;
  size_t virtual_memory_size_;
  size_t physical_memory_size_;
  size_t page_size_bits_;
  size_t page_size_;
  size_t page_size_mask_;
  std::vector<PhysicalMemoryInfo> physical_memorys_;
  std::vector<size_t> free_physical_memorys_;
  std::vector<size_t> pa_list_;
  size_t map_count_;
  int32_t log_level_;
  std::shared_ptr<PhysicalMemoryAllocator> physical_memory_allocator_;
  size_t theory_size_ = 0U;
  size_t theory_min_size_ = 0U;
  bool reuse_; // 表示申请内存时是否可以复用空闲内存
  bool vapa_check_failed_; // CachingMemAllocator::AllocateWithTryRecycle会重试，重试立即失败。
};

class ExpandableActiveMemoryAllocator : public ExpandableMemoryAllocator {
 public:
  explicit ExpandableActiveMemoryAllocator(const uint32_t device_id, const rtMemType_t memory_type = RT_MEMORY_HBM,
                                           const size_t page_size = kDrvPageSize)
      : ExpandableMemoryAllocator(), device_id_(device_id), memory_type_(memory_type),
        used_count_(0U),
        virtual_active_addr_(nullptr),
        virtual_memory_size_(0U),
        used_memory_size_(0U),
        support_reserve_mem_address_(true),
        page_size_(page_size),
        reuse_(true),
        share_phy_allocator_(true) {
  }

  ~ExpandableActiveMemoryAllocator() override = default;

  // forbidden copy
  ExpandableActiveMemoryAllocator(const ExpandableActiveMemoryAllocator &) = delete;
  ExpandableActiveMemoryAllocator &operator=(const ExpandableActiveMemoryAllocator &) & = delete;

  // forbidden move
  ExpandableActiveMemoryAllocator(const ExpandableActiveMemoryAllocator &&) = delete;
  ExpandableActiveMemoryAllocator &operator=(const ExpandableActiveMemoryAllocator &&) & = delete;

  uint8_t *MallocMemory(const std::string &purpose, int64_t memory_size, bool incremental = false) override;

  Status FreeMemory() override;

  bool IsSupportExpandableMemory() const override { return support_reserve_mem_address_; }

  Status MallocVirtualMemory(const size_t memory_size);

  ExpandableActiveMemoryAllocatorImp &GetPyhsicalMemoryAllocator() {
    return active_memory_allocator_;
  }

  void SetReuse(bool reuse) override {
    reuse_ = reuse;
  }
  void SetSharePhyAllocator(bool share_phy_allocator) override{
    share_phy_allocator_ = share_phy_allocator;
  }
  uint8_t *GetVirtualActiveAddress() const {
    return virtual_active_addr_;
  }
  size_t GetUsedSize() const {
    return used_memory_size_;
  }
  rtMemType_t GetMemType() const {
    return memory_type_;
  }

 private:
  Status MallocPhysicalMemoryAndMap(const std::string &purpose, const size_t memory_size);

 private:
  std::recursive_mutex mutex_;
  uint32_t device_id_;
  rtMemType_t memory_type_;
  size_t used_count_;
  uint8_t *virtual_active_addr_;
  size_t virtual_memory_size_;
  size_t used_memory_size_;
  bool support_reserve_mem_address_;
  size_t page_size_;
  ExpandableActiveMemoryAllocatorImp active_memory_allocator_{ActiveMemoryUtil::IntegerBitsNum(page_size_)};
  std::vector<std::pair<uint8_t *, size_t>> mapped_memory_addrs_;
  bool reuse_; // 表示申请内存时是否可以复用空闲内存
  bool share_phy_allocator_; // 表示是否共享物理内存池
};

class FixedBaseExpandableAllocator : public Allocator {
 public:
  explicit FixedBaseExpandableAllocator(const uint32_t device_id, const rtMemType_t memory_type = RT_MEMORY_HBM,
                                        const size_t page_size = kDrvPageSize)
      : ex_active_allocator_(device_id, memory_type, page_size) {}

  ~FixedBaseExpandableAllocator() override = default;

  // forbidden copy
  FixedBaseExpandableAllocator(const FixedBaseExpandableAllocator &) = delete;
  FixedBaseExpandableAllocator &operator=(const FixedBaseExpandableAllocator &) & = delete;

  // forbidden move
  FixedBaseExpandableAllocator(const FixedBaseExpandableAllocator &&) = delete;
  FixedBaseExpandableAllocator &operator=(const FixedBaseExpandableAllocator &&) & = delete;

  MemBlock *Malloc(size_t size) override;

  void Free(MemBlock *mem_block) override;

  const uint8_t *GetAddress() const {
    return ex_active_allocator_.GetVirtualActiveAddress();
  }

  size_t GetSize() const {
    return ex_active_allocator_.GetUsedSize();
  }
 private:
  ExpandableActiveMemoryAllocator ex_active_allocator_;
};

class ActiveMemoryAllocator {
 public:
  explicit ActiveMemoryAllocator(const uint32_t device_id = 0U, const rtMemType_t memory_type = RT_MEMORY_HBM,
                                 const size_t page_size = kDrvPageSize)
      : memory_type_(memory_type), used_count_(0U), peak_size_(0U), maximum_active_addr_(nullptr),
        device_id_(device_id), memory_size_(0), support_extend_memory_full_(false),
        expandable_memory_allocator_(device_id) {
    (void)page_size;
  }

  virtual ~ActiveMemoryAllocator() = default;

  /// @ingroup ge_graph
  /// @brief malloc memory
  /// @param [in] purpose memory usage
  /// @param [in] size memory size list
  /// @param [out] active_memorys
  /// @param [in] device_id device id
  /// @return  memory addr
  uint8_t *MallocMemory(const std::string &purpose, LogicalMemorys &logical_memorys,
                        std::vector<std::pair<uint8_t *, size_t>> &active_memorys, const uint32_t device_id = 0U);

  /// @ingroup ge_graph
  /// @brief free memory
  /// @param [in] device_id device id
  /// @return Status result of function
  Status FreeMemory(const uint32_t device_id = 0U);

  /// @ingroup ge_graph
  /// @brief get max memory malloced
  /// @return max memory malloced
  int64_t MemorySize();

  Status MallocPhysicalMemory(const std::string &purpose,
                              const std::vector<std::pair<uint8_t *, size_t>> &active_memorys);

  void Recycle(const std::vector<std::pair<uint8_t *, size_t>> &active_memorys);

  bool IsSupportExpandableMemoryFull();

 private:
  void MallocByActiveMemorys(LogicalMemorys &logical_memorys);

  void MergeBlocks(LogicalMemorys &logical_memorys) const;

  void MallocActiveMemorys(const std::string &purpose, LogicalMemorys &logical_memorys,
                           std::vector<std::pair<uint8_t *, size_t>> &active_memorys);

 private:
  rtMemType_t memory_type_;
  size_t used_count_;
  size_t peak_size_;
  uint8_t *maximum_active_addr_;
  uint32_t device_id_;
  int64_t memory_size_;
  bool support_extend_memory_full_;
  ActiveMemorys active_memorys_;
  std::recursive_mutex mutex_;
  ExpandableActiveMemoryAllocator expandable_memory_allocator_;
};

template <typename T>
class SessionMemAllocator {
 public:
  SessionMemAllocator() = default;
  ~SessionMemAllocator() {
    const std::lock_guard<std::mutex> lock(map_mutex_);
    session_allocator_map_.clear();
  }

  // forbidden copy
  SessionMemAllocator(const SessionMemAllocator &) = delete;
  SessionMemAllocator &operator=(const SessionMemAllocator &) & = delete;

  // forbidden move
  SessionMemAllocator(const SessionMemAllocator &&) = delete;
  SessionMemAllocator &operator=(const SessionMemAllocator &&) & = delete;

  static SessionMemAllocator &Instance();

  std::shared_ptr<T> GetMemAllocator(const uint64_t session_id, const uint32_t device_id,
                                     const rtMemType_t memory_type = RT_MEMORY_HBM,
                                     const size_t page_size = kDrvPageSize) {
    const std::lock_guard<std::mutex> lock(map_mutex_);
    auto &find_mem_allocator = session_allocator_map_[session_id][device_id][memory_type][page_size];
    if (find_mem_allocator == nullptr) {
      find_mem_allocator = std::make_shared<T>(device_id, memory_type, page_size);
      GELOGI("create session allocator, typeid: %s, session_id: %llu, device_id: %" PRIu64 ", memory_type: %u, "
        "page_size: %zu", typeid(T).name(), session_id, device_id, memory_type, page_size);
    } else {
      GELOGI("get session allocator, typeid: %s, session_id: %llu, device_id: %" PRIu64 ", memory_type: %u, "
        "page_size: %zu", typeid(T).name(), session_id, device_id, memory_type, page_size);
    }
    return find_mem_allocator;
  }

  void RemoveAllocator(const uint64_t session_id, const uint32_t device_id) {
    const std::lock_guard<std::mutex> lock(map_mutex_);
    const auto iter = session_allocator_map_.find(session_id);
    if (iter == session_allocator_map_.end()) {
      return;
    }
    const auto device_iter = iter->second.find(device_id);
    if (device_iter == iter->second.end()) {
      return;
    }
    iter->second.erase(device_iter);
    if (iter->second.empty()) {
      session_allocator_map_.erase(iter);
    }
    GELOGI("remove session allocator, typeid: %s, session_id: %llu, device_id: %" PRIu64,
           typeid(T).name(), session_id, device_id);
  }

  void RemoveAllocator(const uint64_t session_id) {
    const std::lock_guard<std::mutex> lock(map_mutex_);
    const auto iter = session_allocator_map_.find(session_id);
    if (iter == session_allocator_map_.end()) {
      return;
    }
    session_allocator_map_.erase(iter);
    GELOGI("remove session allocator, class: %s, session_id: %llu", typeid(T).name(), session_id);
  }

 private:
  std::mutex map_mutex_;
  // session_id, device_id, memory_type, page_size
  std::unordered_map<uint64_t, std::unordered_map<uint32_t, std::unordered_map<uint64_t,
      std::unordered_map<uint64_t, std::shared_ptr<T>>>>> session_allocator_map_;
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_GRAPH_MEM_ALLOCATOR_H_

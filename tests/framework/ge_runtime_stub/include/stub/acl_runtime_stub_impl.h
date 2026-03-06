/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_ACL_RUNTIME_STUB_IMPL_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_ACL_RUNTIME_STUB_IMPL_H_
#include <list>
#include <map>
#include <string>
#include <unordered_map>

#include "ge_fake_launch_args.h"
#include "ge_fake_rtmemcpy_args.h"
#include "ge_fake_rtswitch_args.h"
#include "depends/ascendcl/src/ascendcl_stub.h"
#include "runtime_stub_impl.h"

namespace gert {
using HandleArgsPtrList = std::list<ge::GeFakeLaunchArgs *>;
class AclRuntimeStubImpl : public ge::AclRuntimeStub {
 public:
  struct MemoryInfo {
    void *addr;
    size_t size;
    rtMemType_t rts_mem_type;
    uint16_t model_id;
  };
  const std::map<const void *, HandleArgsPtrList> &GetLaunchWithHandleArgs();
  void Clear();
  AclRuntimeStubImpl();
  ge::GeFakeLaunchArgs *PopLaunchArgsBy(const void *handle);
  ge::GeFakeLaunchArgs *PopLaunchArgsBy(const void *handle, uint64_t devFunc);
  ge::GeFakeLaunchArgs *PopLaunchArgsByStubFunc(const void *stubFunc);
  ge::GeFakeLaunchArgs *PopLaunchArgsByStubName(const std::string &stub_name);
  ge::GeFakeLaunchArgs *PopLaunchArgsByBinary(const rtDevBinary_t *bin);
  ge::GeFakeLaunchArgs *PopLaunchArgsByBinary(const rtDevBinary_t *bin, uint64_t devFunc);
  ge::GeFakeLaunchArgs *PopCpuLaunchArgsByKernelName(const std::string &kernel_name);
  uintptr_t FindSrcAddrCpyToDst(uintptr_t dst_addr);
  const std::vector<ge::GeFakeRtMemcpyArgs> &GetRtMemcpyRecords() const;
  const std::vector<ge::GeFakeRtMemcpyArgs> &GetRtMemcpySyncRecords() const;
  const std::list<ge::GeFakeLaunchArgs> &GetAllLaunchArgs() const;
  const std::list<ge::GetAllSwitchArgs> &GetAllSwitchArgs() const;

  const std::unordered_map<void *, MemoryInfo> &GetAllocatedRtsMemory() const {
    return addrs_to_mem_info_;
  }
  const std::list<ge::GeLaunchSqeUpdateTaskArgs> &GetLaunchSqeUpdateTaskArgs() const {
    return all_launch_sqe_update_records_;
  }

  const std::vector<uintptr_t> &GetLiteEceptionArgs() const {
    return lite_exception_args_;
  }
  const std::map<rtEvent_t, std::vector<rtStream_t>> &GetRtEventRecordRecords() const {
    return events_to_record_records_;
  }
  const std::vector<TaskTypeOnStream> GetAllTaskOnStream(rtStream_t stream) {
    return stream_stub_.GetAllTaskOfStream(stream);
  }
  const std::vector<rtStream_t> GetAllRtStreams() {
    return stream_stub_.GetAllRtStreams();
  }

  void LaunchTaskToStream(TaskTypeOnStream task_type, rtStream_t stream) {
    stream_stub_.LaunchTaskToStream(task_type, stream);
  }

 protected:
  aclError aclrtStreamGetId(aclrtStream stream, int32_t *streamId) override;

  aclError aclrtGetThreadLastTaskId(uint32_t *taskId) override;

  aclError aclrtPersistentTaskClean(aclrtStream stream) override;

  aclError aclrtCreateStream(aclrtStream *stream) override;

  aclError aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag) override;

  aclError aclrtDestroyStream(aclrtStream stream) override;

  aclError aclrtDestroyStreamForce(aclrtStream stream) override;

  aclError aclrtGetStreamAvailableNum(uint32_t *streamCount) override;

  aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) override;

  aclError aclrtMemcpyAsync(void *dst, size_t dest_max, const void *src, size_t src_count, aclrtMemcpyKind kind,
                            aclrtStream stream) override;

  aclError aclrtMalloc(void **dev_ptr, size_t size, aclrtMemMallocPolicy policy) override;

  aclError aclrtFree(void *dev_ptr) override;

  aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total) override;

  aclError aclrtSwitchStream(void *leftValue, aclrtCondition cond, void *rightValue, aclrtCompareDataType dataType,
                             aclrtStream trueStream, aclrtStream falseStream, aclrtStream stream) override;

  aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) override;

  aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) override;

  aclError aclmdlRIExecuteAsync(aclmdlRI modelRI, aclrtStream stream) override;

 private:
  using BinHandle = uint64_t;
  std::map<std::string, BinHandle> stub_names_to_handles_;

  struct BinData {
    explicit BinData(const rtDevBinary_t *binary)
        : magic(binary->magic),
          version(binary->version),
          data(reinterpret_cast<uint64_t>(binary->data)),
          length(binary->length) {}
    bool operator<(const BinData &rht) const {
      return this->data < rht.data;
    }
    uint32_t magic;    // magic number
    uint32_t version;  // version of binary
    uint64_t data;     // binary data
    uint64_t length;   // binary length
  };

  std::map<BinData, BinHandle> bin_data_to_handles_;
  std::list<ge::GeFakeLaunchArgs> all_launch_args_;
  std::map<const void *, HandleArgsPtrList> launch_with_handle_args_;
  std::map<std::string, HandleArgsPtrList> cpu_launch_args_;
  std::map<uintptr_t, uintptr_t> dst_addrs_to_src_addrs_;
  std::vector<ge::GeFakeRtMemcpyArgs> rt_memcpy_args_;
  std::vector<ge::GeFakeRtMemcpyArgs> rt_memcpy_sync_args_;
  std::list<ge::GetAllSwitchArgs> all_switch_args_;
  std::unordered_map<void *, MemoryInfo> addrs_to_mem_info_;
  std::list<ge::GeLaunchSqeUpdateTaskArgs> all_launch_sqe_update_records_;
  std::mutex mtx_;
  std::unique_ptr<std::string> last_tag_;
  // uint32_t task_id_{0}; // 同一个模型的task在一条流上分配，只用来区分不同task
  uint64_t last_stream_;
  std::map<uint64_t, uint32_t> stream_to_task_id_;
  std::vector<uintptr_t> lite_exception_args_;
  std::map<rtEvent_t, std::vector<rtStream_t>> events_to_record_records_;
  GertStreamStub stream_stub_;
  GertEventStub event_stub_;
};
}  // namespace gert

#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_ACL_RUNTIME_STUB_IMPL_H_

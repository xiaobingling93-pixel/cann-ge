/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNEL_BUILDER_H_
#define AICPU_KERNEL_BUILDER_H_

#include <nlohmann/json.hpp>

#include "aicpu_ops_kernel_info_store/op_struct.h"
#include "aicpu_ops_kernel_builder/aicpu_ops_kernel_builder.h"
#include "error_code/error_code.h"
#include "factory/factory.h"
#include "register/ops_kernel_builder_registry.h"
namespace aicpu {

struct FftsPlusInfo {
  bool valid;
  bool thread_mode;
  bool is_unknown_shape;
  bool auto_static_split;
  uint32_t slice_instance_num;
  uint32_t slice_instance_index;
  uint32_t thread_id;
  uint32_t ext_len;
  std::vector<int64_t> input_addr_offset;
  std::vector<int64_t> output_addr_offset;
  std::vector<std::vector<std::vector<int64_t>>> thread_input_shape;
  std::vector<std::vector<std::vector<int64_t>>> thread_output_shape;
  FftsPlusInfo() {
    valid = false;
    thread_mode = false;
    is_unknown_shape = false;
    auto_static_split = false;
    slice_instance_num = 0u;
    slice_instance_index = 0u;
    thread_id = 0u;
    ext_len = 0u;
  }
};

class KernelBuilder {
 public:
  /**
   * constructor
   * @param void
   */
  KernelBuilder() = default;
  /**
   * Destructor
   */
  virtual ~KernelBuilder() = default;

  /**
   * init kernel builder
   * @return status whether this operation success
   */
  virtual ge::Status Initialize() { return ge::SUCCESS; }
  /**
   * Release related resources of the aicpu kernel builder
   * @return status whether this operation success
   */
  virtual ge::Status Finalize() { return ge::SUCCESS; }

  /**
   * Calc the running size of Operator,then GE will alloc the memsize from
   * runtime The size is consist of the part as follow: 1.StrFWKKernel; 2.Input
   * and output size; 3.NodeDef in tf; 4.FuncDef in tf.
   * @param node Node information, return task_memsize in node's attr
   * @return status whether this operation success
   */
  virtual ge::Status CalcOpRunningParam(const ge::Node &node) const = 0;

  /**
   * Calc the running size of Operator,then GE will alloc the memsize from
   * runtime The size is consist of the part as follow: 1.StrFWKKernel; 2.Input
   * and output size; 3.NodeDef in tf; 4.FuncDef in tf.
   * @param node Node information, return task_memsize in node's attr
   * @return status whether this operation success
   */
  virtual ge::Status GenerateTask(const ge::Node &node,
                                  const ge::RunContext &context,
                                  std::vector<domi::TaskDef> &tasks) = 0;

  virtual ge::Status UpdateTask(const ge::Node &node,
                                std::vector<domi::TaskDef> &tasks) = 0;
  /**
   * Generate the task
   * @param node Node information
   * @param task[out]
   * @param task_info[out]
   * @return status whether this operation success
   */
  virtual ge::Status GenSingleOpRunTask(const ge::NodePtr &node,
                                        STR_FWK_OP_KERNEL &task,
                                        std::string &task_info) {
    (void)node;
    (void)task;
    (void)task_info;
    return ge::FAILED;
  };

  /**
   * Generate the task
   * @param count the memcopy times
   * @param task[out]
   * @param task_info[out]
   * @return status whether this operation success
   */
  virtual ge::Status GenMemCopyTask(uint64_t count, STR_FWK_OP_KERNEL &task,
                                    std::string &task_info) {
    (void)count;
    (void)task;
    (void)task_info;
    return ge::FAILED;
  };

 protected:
  /**
   * Get workspace info
   * @param op_desc_ptr Ge op description pointer
   * @param data_mem_base Data memory base addr
   * @param workspace_bytes_size Workspace bytes size
   * @return whether handle success
   */
  ge::Status GetWorkspaceInfo(const ge::OpDescPtr &op_desc_ptr,
                              const uint8_t *data_mem_base,
                              uint64_t &workspace_bytes_size) const;

  /**
   * Make extend info for op name: must be the first extend info (RUNTIME only
   * decipher the first extend info)
   * @param op_desc_ptr Ge op description pointer
   * @param tastExtInfo task extend info
   * @return whether handle success
   */
  ge::Status MakeExtInfoForOpName(const ge::OpDescPtr &op_desc_ptr,
                                  std::vector<char> &task_ext_info) const;

  /**
   * Make common task extend info
   * @param op_desc_ptr Ge op description pointer
   * @param tastExtInfo task extend info
   * @return whether handle success
   */
  ge::Status MakeBaseExtInfo(const ge::OpDescPtr &op_desc_ptr,
                             std::vector<char> &task_ext_info,
                             const FftsPlusInfo &ffts_info) const;

  /**
   * Make notiling task extend info
   * @param op_desc_ptr Ge op description pointer
   * @param tastExtInfo task extend info
   * @return whether handle success
   */
  ge::Status MakeNoTilingExtInfo(const ge::OpDescPtr &op_desc_ptr, std::vector<char> &task_ext_info) const;

  /**
   * Get input and output shape
   * @param op_desc_ptr Ge op description pointer
   * @param inputs_shape vector store input shape
   * @param outputs_shape vector store output shape
   */
  void GetInOutPutsShape(const ge::OpDescPtr &op_desc_ptr,
                         std::vector<std::vector<int64_t>> &inputs_shape,
                         std::vector<std::vector<int64_t>> &outputs_shape) const;
  
  /*
  * Calc comon task extend info len
  * @param op_desc_ptr Ge op description pointer
  * @param op_async_flag op async flag
  * @param extend_info_len task extend info len
  * @return void
  */
  void CalcBaseExtInfoLen(const ge::OpDescPtr &op_desc_ptr,
                          const bool op_async_flag,
                          uint64_t &extend_info_len) const;

  /**
   * Get input and output type
   * @param op_desc_ptr Ge op description pointer
   * @param inputs_type vector store input type
   * @param outputs_type vector store output type
   */
  virtual void GetInOutPutsDataType([[maybe_unused]] const ge::OpDescPtr &op_desc_ptr,
                                    [[maybe_unused]] std::vector<uint32_t> &inputs_type,
                                    [[maybe_unused]] std::vector<uint32_t> &outputs_type) const {}

  ge::Status GetFftsPlusInAddrOffset(const ge::OpDescPtr &op_desc_ptr,
                                     FftsPlusInfo &ffts_info) const;

  ge::Status GetFftsPlusOutAddrOffset(const ge::OpDescPtr &op_desc_ptr,
                                      FftsPlusInfo &ffts_info) const;

  ge::Status GetFftsPlusInOutAddrOffset(const ge::OpDescPtr &op_desc_ptr,
                                        FftsPlusInfo &ffts_info) const;

  ge::Status CalFftsMaxThread(const ge::OpDescPtr &op_desc_ptr) const;

  FWKAdapter::FWKExtTopicType GetOpNodeTopicType(const ge::OpDescPtr &op_desc_ptr) const;

  /**
   * Set attr queue resource
   * @param node_name Ge op mode name
   * @param op_desc_ptr op desc
   * @param resource_list attr result
   * @return whether handle success
   */
  ge::Status SetAttrQueueResource(
      const std::string &node_name, std::shared_ptr<ge::OpDesc> &op_desc_ptr,
      std::vector<ge::GeAttrValue::NAMED_ATTRS> &resource_list) const;

  /**
   * Set attr resource
   * @param node_name Ge op mode name
   * @param op_desc_ptr op desc
   * @return whether handle success
   */
  ge::Status SetAttrResource(const std::string &node_name,
                             std::shared_ptr<ge::OpDesc> &op_desc_ptr) const;
};

#define FACTORY_KERNEL_BUILDER Factory<KernelBuilder>

#define FACTORY_KERNEL_BUILDER_CLASS_KEY(CLASS, KEY) \
  FACTORY_KERNEL_BUILDER::Register<CLASS> __##CLASS(KEY);

using FftsPlusCtxDefPtr = std::shared_ptr<domi::FftsPlusCtxDef>;
}  // namespace aicpu

#endif  // AICPU_KERNEL_BUILDER_H_

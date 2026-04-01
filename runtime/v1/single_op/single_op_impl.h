/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_SINGLE_OP_SINGLE_OP_IMPL_H_
#define GE_SINGLE_OP_SINGLE_OP_IMPL_H_

#include "graph/utils/object_pool.h"
#include "single_op/task/op_task.h"
#include "hybrid/executor/hybrid_model_executor.h"
#include "hybrid/executor/hybrid_model_rt_v1_executor.h"
#include "common/profiling_definitions.h"

namespace ge {
class SingleOpImpl {
 public:
  SingleOpImpl(StreamResource *const stream_res, std::mutex *const stream_mutex, aclrtStream const stream);
  ~SingleOpImpl() = default;

  Status ExecuteAsync(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs);

  int64_t GetProfilingNodeIndex() const noexcept;

  Status MallocOnExecute();

  const uint8_t *GetMemoryBase() const;

  void FreeAllocatedMem();

 private:
  Status ValidateArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs);

  Status UpdateArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs);

  Status GetArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs);

  bool CheckHostMemInputOptimization(const std::vector<DataBuffer> &input_buffers) const;

  friend class SingleOpModel;
  StreamResource *stream_resource_;
  std::mutex *stream_mutex_;
  aclrtStream stream_;
  std::vector<const void *> input_addr_list_;
  std::vector<size_t> input_sizes_;
  std::vector<const void *> output_addr_list_;
  std::vector<size_t> output_sizes_;
  std::vector<uintptr_t> args_;

  std::vector<std::unique_ptr<OpTask>> tasks_;
  std::vector<std::vector<uintptr_t *>> arg_table_;
  std::unique_ptr<SingleOpModelParam> model_param_;
  std::vector<GeTensorDesc> inputs_desc_;
  int64_t profiling_node_type_index_{gert::profiling::kUnknownName};
  ComputeGraphPtr root_graph_ = nullptr;
  ge::MemBlock *allocated_mem_{nullptr};
};

class DynamicSingleOpImpl {
 public:
  DynamicSingleOpImpl(ObjectPool<GeTensor> *const tensor_pool, const uintptr_t resource_id,
                      std::mutex *const stream_mutex, aclrtStream const stream);
  ~DynamicSingleOpImpl() = default;

  Status ExecuteAsync(const std::vector <GeTensorDesc> &input_desc,
                      const std::vector <DataBuffer> &input_buffers,
                      std::vector <GeTensorDesc> &output_desc,
                      std::vector <DataBuffer> &output_buffers);

  int64_t GetProfilingNodeIndex() const noexcept;

 private:
  friend class SingleOpModel;

  Status ValidateParams(const std::vector <GeTensorDesc> &input_desc,
                        const std::vector <DataBuffer> &inputs,
                        const std::vector <GeTensorDesc> &output_desc,
                        const std::vector <DataBuffer> &outputs) const;

  Status SetHostTensorValue(const std::vector <std::pair<size_t, uint64_t>> &inputs_size,
                            const std::vector <GeTensorDesc> &input_desc,
                            const std::vector <DataBuffer> &input_buffers);

  Status SetHostTensorValue(const std::vector <GeTensorDesc> &input_desc,
                            const std::vector <DataBuffer> &input_buffers);

  bool CheckHostMemInputOptimization(const std::vector <DataBuffer> &input_buffers);

  void InjectRuntimeContext();

  std::unique_ptr <OpTask> op_task_;
  std::unique_ptr <hybrid::HybridModel> hybrid_model_;
  std::unique_ptr <hybrid::HybridModelRtV1Executor> hybrid_model_executor_;
  std::map <int32_t, int64_t> hostmem_node_id_map_;
  std::map <int32_t, std::pair<int32_t, int32_t>> input_node_anchor_map_;
  std::vector <NodePtr> node_with_hostmem_;

  ObjectPool <GeTensor> *tensor_pool_;
  int64_t profiling_node_type_index_ = -1;
  uintptr_t resource_id_;
  std::mutex *stream_mutex_;
  aclrtStream stream_;
  size_t num_inputs_ = 0U;
  size_t num_outputs_ = 0U;
  ComputeGraphPtr compute_graph_;
  std::queue <std::unique_ptr<GeTensor>> shared_tensors_;
  RuntimeInferenceContext runtime_context_;
};
} // namespace ge
#endif // GE_SINGLE_OP_SINGLE_OP_IMPL_H_

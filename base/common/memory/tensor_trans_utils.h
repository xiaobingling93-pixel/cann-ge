/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_TENSOR_TRANS_UTILS_H
#define CANN_GRAPH_ENGINE_TENSOR_TRANS_UTILS_H
#include "ge_common/ge_api_types.h"
#include "ge/ge_allocator.h"
#include "exe_graph/runtime/tensor.h"
#include "graph/tensor.h"
#include "graph/ge_tensor.h"
#include "mem_base.h"
namespace ge {

struct MemcpyBatchParam {
  std::vector<void *> dsts;
  // fallback时调用rtMemcpy接口时使用
  std::vector<size_t> dst_aligned_sizes;
  std::vector<void *> srcs;
  std::vector<size_t> src_sizes;
  std::vector<rtMemcpyBatchAttr> attrs;
  std::vector<size_t> attr_idxs;
  int32_t device_id{};
};

struct MemcpyParam {
  void *dst;
  uint64_t dst_aligned_size;
  void *src;
  uint64_t src_size;
  size_t idx;
};

class TensorTransUtils {
 public:
  static GeShape ContructGeShapeFromRtShape(const gert::Shape &rt_shape);
  static gert::Shape ContructRtShapeFromShape(const Shape &ge_shape);
  static gert::Shape ContructRtShapeFromGeShape(const GeShape &ge_shape);
  static gert::Shape ContructRtShapeFromVector(const std::vector<int64_t> &dims);
  static std::vector<int64_t> GetDimsFromGertShape(const gert::Shape &gert_shape);

  static Status TransHostGertTensorsToDevice(Allocator *allocator, const std::vector<gert::Tensor> &src_tensors, std::vector<gert::Tensor> &dst_tensors,
                                             std::vector<MemBlock *> &inputs_memblocks, bool enable_input_batch_cpy);
  static Status HostTensorToDeviceGertTensor(ge::Allocator *allocator, const void *src_tensor_addr, uint64_t src_tensor_length,
                                             gert::Tensor &dst_tensor, MemBlock *&mem_block_to_keep);
  static Status FillRtTensorDesc(const Tensor &src_tensor, gert::Tensor &dst_tensor);
  static Status AllocDeviceMemory(ge::Allocator *allocator, uint64_t src_tensor_length, gert::Tensor &dst_tensor,
                                MemBlock *&mem_block_to_keep, size_t &dst_aligned_size);
  static GeTensor TransRtTensorToGeTensor(const gert::Tensor &input);
  static Status TransTensorToGertTensor(const Tensor &tensor, gert::Tensor &rt_tensor);
  static Status TransRtTensorToGeTensor(const gert::Tensor &src, GeTensor &dst);

  // before call this, make sure device tensor is ready, which means stream has been syncronized
  static Status TransRtTensorToTensor(const std::vector<gert::Tensor> &srcs, std::vector<Tensor> &dsts,
                                      bool with_value);

  // gert::Tensor -> GeTensor
  // gert_tensor会增加引用计数，ge_tensor会设置deleter，各自析构时触发引用计数减一，减到零就释放内存
  static Status GertTensor2GeTensor(const gert::Tensor &gert_tensor, GeTensor &ge_tensor);
  static Status GertTensors2GeTensors(const std::vector<gert::Tensor> &gert_tensors, std::vector<GeTensor> &ge_tensors);

  // gert::Tensor -> Tensor
  // gert_tensor会增加引用计数，ge_tensor会设置deleter，各自析构时触发引用计数减一，减到零就释放内存
  static Status GertTensor2Tensor(const gert::Tensor &gert_tensor, Tensor &ge_tensor);
  static Status GertTensors2Tensors(const std::vector<gert::Tensor> &gert_tensors, std::vector<Tensor> &ge_tensors);

  // GeTensor -> gert::Tensor
  // GeTensor底层使用shared_ptr管理内存，首先通过TensorUtils::ShareTensor获取ge_tensor_copy，
  // 然后将ge_tensor_copy作为gert_tensor的data，ge_tensor和gert_tensor可单独安全释放
  static Status GeTensor2GertTensor(const GeTensor &ge_tensor, gert::Tensor &gert_tensor);
  static Status GeTensors2GertTensors(const std::vector<GeTensor> &ge_tensors,
    std::vector<gert::Tensor> &gert_tensors);

  // Tensor -> gert::Tensor
  static Status Tensor2GertTensor(const Tensor &ge_tensor, gert::Tensor &gert_tensor);
  static Status Tensors2GertTensors(const std::vector<Tensor> &ge_tensors,
    std::vector<gert::Tensor> &gert_tensors);

  // Tensor -> gert::Tensor
  // 比Tensor2GertTensor性能高，注意：tensor_view是ge_tensor的引用/视图，
  // 仅在ge_tensor的生命周期内有效。当ge_tensor被释放后，tensor_view将变为悬空引用，
  // 继续使用会导致未定义行为。
  static Status AsTensorView(const Tensor &ge_tensor, gert::Tensor &tensor_view);
  static Status AsTensorsView(const std::vector<Tensor> &ge_tensors,
    std::vector<gert::Tensor> &tensors_view);

  // device gert tensor to host, host内存使用AlignedPtr
  static Status TransGertTensorToHost(const gert::Tensor &device_tensor, gert::Tensor &host_tensor);
  static Status TransGertTensorsToHost(const std::vector<gert::Tensor> &device_tensors,
    std::vector<gert::Tensor> &host_tensors);
  static std::vector<gert::Tensor> ShareFromGertTenosrs(const std::vector<gert::Tensor> &gert_tensors);
  static void AddMemcpyBatchParam(const MemcpyParam &param, MemcpyBatchParam &memcpy_batch_params);
  static Status TryBatchMemcpy(MemcpyBatchParam &args);
};
} // ge
#endif  // CANN_GRAPH_ENGINE_TENSOR_TRANS_UTILS_H

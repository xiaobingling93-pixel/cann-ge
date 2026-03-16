/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_UT_GE_RUNTIME_V2_FAKER_KERNEL_RUN_CONTEXT_FACKER_H_
#define AIR_CXX_TESTS_UT_GE_RUNTIME_V2_FAKER_KERNEL_RUN_CONTEXT_FACKER_H_
#include <memory>
#include <vector>
#include <cstring>
#include "exe_graph/runtime/kernel_run_context.h"
#include "exe_graph/runtime/context_extend.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/lowering/buffer_pool.h"
#include "graph/any_value.h"
#include "graph/node.h"
#include "lowering/kernel_run_context_builder.h"
#include "exe_graph/runtime/gert_mem_allocator.h"

namespace gert {
struct FakeKernelContextHolder {
  template<typename T>
  T *GetContext() {
    return reinterpret_cast<T*>(holder.context_);
  }
  ComputeNodeInfo *MutableComputeNodeInfo() {
    return reinterpret_cast<ComputeNodeInfo *>(holder.compute_node_extend_holder_.get());
  }
  size_t kernel_input_num;
  size_t kernel_output_num;
  KernelContextHolder holder;
};
FakeKernelContextHolder BuildKernelRunContext(size_t input_num, size_t output_num);

class KernelRunContextFaker {
 public:
  KernelRunContextFaker() = default;
  KernelRunContextFaker &KernelIONum(size_t input_num, size_t output_num);
  KernelRunContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  KernelRunContextFaker &IrInputNum(size_t input_num);
  KernelRunContextFaker &IrOutputNum(size_t input_num);
  KernelRunContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num);
  KernelRunContextFaker &IrOutputInstanceNum(std::vector<uint32_t> instance_num);
  KernelRunContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                     ge::Format storage_format);
  KernelRunContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                      ge::Format storage_format);
  KernelRunContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value);
  KernelRunContextFaker &Inputs(std::vector<void *> inputs);
  KernelRunContextFaker &Outputs(std::vector<void *> outputs);

  FakeKernelContextHolder Build() const;

 private:
  ge::OpDescPtr FakeOp() const;

 private:
  size_t kernel_input_num_;
  size_t kernel_output_num_;
  size_t node_input_num_;
  size_t node_output_num_;
  std::vector<uint32_t> ir_instance_num_;
  std::vector<uint32_t> ir_output_instance_num_{};
  std::vector<CompileTimeTensorDesc> node_input_tds_;
  std::vector<CompileTimeTensorDesc> node_output_tds_;
  std::vector<void *> inputs_;
  std::vector<void *> outputs_;
  std::vector<std::pair<std::string, ge::AnyValue>> attrs_;
};

class InferShapeContextFaker {
 public:
  InferShapeContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  InferShapeContextFaker &IrInputNum(size_t input_num) {
    base_faker_.IrInputNum(input_num);
    return *this;
  }
  InferShapeContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num) {
    base_faker_.IrInstanceNum(std::move(instance_num));
    return *this;
  }
  InferShapeContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                      ge::Format storage_format) {
    base_faker_.NodeInputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  InferShapeContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                       ge::Format storage_format) {
    base_faker_.NodeOutputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  InferShapeContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
    base_faker_.NodeAttrs(std::move(keys_to_value));
    return *this;
  }

  InferShapeContextFaker &InputShapes(std::vector<void *> input_shapes);
  InferShapeContextFaker &OutputShapes(std::vector<void *> output_shapes);

  FakeKernelContextHolder Build() const;

 private:
  enum InputsAppend { kInputsInferShapeFunc, kInputsAppendEnd };

 private:
  KernelRunContextFaker base_faker_;
};

class InferSymbolShapeContextFaker {
 public:
  InferSymbolShapeContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  InferSymbolShapeContextFaker &IrInputNum(size_t input_num);
  InferSymbolShapeContextFaker &IrInputInstanceNum(std::vector<uint32_t> instance_num);
  InferSymbolShapeContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                            ge::Format storage_format);
  InferSymbolShapeContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                             ge::Format storage_format);
  InferSymbolShapeContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value);

  InferSymbolShapeContextFaker &Inputs(std::vector<void *> input_shapes);
  InferSymbolShapeContextFaker &Outputs(std::vector<void *> output_shapes);

  FakeKernelContextHolder Build() const;

 private:
  enum InputsAppend { kInputsInferShapeFunc, kInputsAppendEnd };
  KernelRunContextFaker base_faker_;
};

class InferShapeRangeContextFaker {
 public:
  InferShapeRangeContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  InferShapeRangeContextFaker &IrInputNum(size_t input_num) {
    base_faker_.IrInputNum(input_num);
    return *this;
  }
  InferShapeRangeContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num) {
    base_faker_.IrInstanceNum(std::move(instance_num));
    return *this;
  }
  InferShapeRangeContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                      ge::Format storage_format) {
    base_faker_.NodeInputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  InferShapeRangeContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                       ge::Format storage_format) {
    base_faker_.NodeOutputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  InferShapeRangeContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
    base_faker_.NodeAttrs(std::move(keys_to_value));
    return *this;
  }

  InferShapeRangeContextFaker &InputShapeRanges(std::vector<void *> input_shape_ranges);
  InferShapeRangeContextFaker &OutputShapeRanges(std::vector<void *> output_shape_ranges);

  FakeKernelContextHolder Build() const;

 private:
  enum InputsAppend { kInputsInferShapeRangeFunc, kInputsAppendEnd };

 private:
  KernelRunContextFaker base_faker_;
};

class InferDataTypeContextFaker {
 public:
  InferDataTypeContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  InferDataTypeContextFaker &IrInputNum(size_t input_num) {
    base_faker_.IrInputNum(input_num);
    return *this;
  }
  InferDataTypeContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num) {
    base_faker_.IrInstanceNum(std::move(instance_num));
    return *this;
  }
  InferDataTypeContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                      ge::Format storage_format) {
    base_faker_.NodeInputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  InferDataTypeContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                       ge::Format storage_format) {
    base_faker_.NodeOutputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  InferDataTypeContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
    base_faker_.NodeAttrs(std::move(keys_to_value));
    return *this;
  }

  InferDataTypeContextFaker &InputDataTypes(std::vector<void *> input_datatypes);
  InferDataTypeContextFaker &OutputDataTypes(std::vector<void *> output_datatypes);

  FakeKernelContextHolder Build() const;

 private:
  enum InputsAppend { kInputsInferDataTypeFunc, kInputsAppendEnd };

 private:
  std::vector<void *> inputs_;
  std::vector<void *> outputs_;
  KernelRunContextFaker base_faker_;
};

class TilingContextFaker {
 public:
  TilingContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  TilingContextFaker &IrInputNum(size_t input_num) {
    base_faker_.IrInputNum(input_num);
    return *this;
  }
  TilingContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num) {
    base_faker_.IrInstanceNum(std::move(instance_num));
    return *this;
  }
  TilingContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format, ge::Format storage_format) {
    base_faker_.NodeInputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  TilingContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                   ge::Format storage_format) {
    base_faker_.NodeOutputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  TilingContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
    base_faker_.NodeAttrs(std::move(keys_to_value));
    return *this;
  }
  TilingContextFaker &InputShapes(std::vector<gert::StorageShape *> input_shapes);
  TilingContextFaker &OutputShapes(std::vector<gert::StorageShape *> output_shapes);
  TilingContextFaker &CompileInfo(void *compile_info);
  TilingContextFaker &PlatformInfo(void *platform_info);
  TilingContextFaker &TilingData(void *tiling_data);
  TilingContextFaker &Workspace(ContinuousVector *workspace);

  FakeKernelContextHolder Build() const;

 private:
  void UpdateInputs();

 private:
  enum InputsAppend { kInputsCompileInfo, kInputsPlatformInfo, kInputsTilingFunc, kInputsAppendEnd };

  KernelRunContextFaker base_faker_;
  std::vector<gert::StorageShape *> input_shapes_;
  std::vector<gert::StorageShape *> output_shapes_;
  std::vector<void *> outputs_ {TilingContext::kOutputNum};

  void *compile_info_;
  void *platform_info_;
};

class OpExecuteContextFaker {
 public:
  OpExecuteContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  OpExecuteContextFaker &IrInputNum(size_t input_num) {
    base_faker_.IrInputNum(input_num);
    return *this;
  }
  OpExecuteContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num) {
    base_faker_.IrInstanceNum(std::move(instance_num));
    return *this;
  }
  OpExecuteContextFaker &IrOutputInstanceNum(std::vector<uint32_t> output_instance_num) {
    base_faker_.IrOutputInstanceNum(std::move(output_instance_num));
    return *this;
  }
  OpExecuteContextFaker &NodeInputTd(int32_t index, ge::DataType dt,
      ge::Format origin_format, ge::Format storage_format) {
    base_faker_.NodeInputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  OpExecuteContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                   ge::Format storage_format) {
    base_faker_.NodeOutputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  OpExecuteContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
    base_faker_.NodeAttrs(std::move(keys_to_value));
    return *this;
  }
  OpExecuteContextFaker &InputTensor(std::vector<gert::Tensor *> input_tensor);
  OpExecuteContextFaker &OutputTensor(std::vector<gert::Tensor *> output_tensor);
  OpExecuteContextFaker &OutputMem(std::shared_ptr<std::vector<gert::GertMemBlock *>> &output_block_memory);
  OpExecuteContextFaker &Allocate(void *allocator);
  OpExecuteContextFaker &Stream(void *stream);
  OpExecuteContextFaker &ExecuteOption(void *execute_option);
  OpExecuteContextFaker &ExecuteFunc(void *execute_func);
  OpExecuteContextFaker &OpAicoreNum(int64_t *op_aicore_num);
  OpExecuteContextFaker &OpVecCoreNum(int64_t *op_vec_core_num);
  OpExecuteContextFaker &GlobalAicoreNum(int64_t *global_aicore_num);
  OpExecuteContextFaker &GlobalVecCoreNum(int64_t *global_vec_core_num);
  FakeKernelContextHolder Build();

 private:
  void UpdateInputs();
  void UpdateOutputs();
 private:
  enum InputsAppend {kAllocate, kStream, kExecuteOption, kExecuteFunc, kOpAicoreNum, kOpVecCoreNum, kGlobalAicoreNum, kGlobalVecCoreNum, kEnd};

  KernelRunContextFaker base_faker_;
  std::vector<gert::Tensor *> input_tensor_;
  std::vector<gert::Tensor *> output_tensor_;
  std::shared_ptr<std::vector<gert::GertMemBlock *>> output_block_memory_;
  void *allocator_ = nullptr;
  void *stream_ = nullptr;
  void *execute_option_ = nullptr;
  void *execute_func_ = nullptr;
  int64_t *op_aicore_num_ = nullptr;
  int64_t *op_vec_core_num_ = nullptr;
  int64_t *global_aicore_num_ = nullptr;
  int64_t *global_vec_core_num_ = nullptr;
};

class OpExecutePrepareContextFaker {
 public:

  OpExecutePrepareContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  OpExecutePrepareContextFaker &IrInputNum(size_t input_num) {
    base_faker_.IrInputNum(input_num);
    return *this;
  }
  OpExecutePrepareContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num) {
    base_faker_.IrInstanceNum(std::move(instance_num));
    return *this;
  }
  OpExecutePrepareContextFaker &IrOutputInstanceNum(std::vector<uint32_t> output_instance_num) {
    base_faker_.IrOutputInstanceNum(std::move(output_instance_num));
    return *this;
  }
  OpExecutePrepareContextFaker &NodeInputTd(int32_t index, ge::DataType dt,
                                     ge::Format origin_format, ge::Format storage_format) {
    base_faker_.NodeInputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  OpExecutePrepareContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                      ge::Format storage_format) {
    base_faker_.NodeOutputTd(index, dt, origin_format, storage_format);
    return *this;
  }

  OpExecutePrepareContextFaker &InputTensor(std::vector<gert::Tensor *> input_tensor);
  OpExecutePrepareContextFaker &OutputTensor(std::vector<gert::Tensor *> output_tensor);
  OpExecutePrepareContextFaker &ExecuteOption(void *execute_option);
  OpExecutePrepareContextFaker &ExecuteFunc(void *execute_func);
  OpExecutePrepareContextFaker &OpApiParams(void *param);
  OpExecutePrepareContextFaker &WorkspaceSize(uint8_t* ws_size_vec);
  FakeKernelContextHolder Build();
 private:
  enum InputsAppend {kExecuteOption, kExecuteFunc, kEnd};
  void UpdateInputs();
  void UpdateOutputs();

  KernelRunContextFaker base_faker_;
  std::vector<gert::Tensor *> input_tensor_;
  std::vector<gert::Tensor *> output_tensor_;
  void *execute_option_ = nullptr;
  void *execute_func_ = nullptr;
  void *param_ = nullptr;
  uint8_t* ws_size_;
};

class OpExecuteLaunchContextFaker {
 public:

  OpExecuteLaunchContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  OpExecuteLaunchContextFaker &IrInputNum(size_t input_num) {
    base_faker_.IrInputNum(input_num);
    return *this;
  }
  OpExecuteLaunchContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num) {
    base_faker_.IrInstanceNum(std::move(instance_num));
    return *this;
  }
  OpExecuteLaunchContextFaker &IrOutputInstanceNum(std::vector<uint32_t> output_instance_num) {
    base_faker_.IrOutputInstanceNum(std::move(output_instance_num));
    return *this;
  }
  OpExecuteLaunchContextFaker &NodeInputTd(int32_t index, ge::DataType dt,
                                            ge::Format origin_format, ge::Format storage_format) {
    base_faker_.NodeInputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  OpExecuteLaunchContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                             ge::Format storage_format) {
    base_faker_.NodeOutputTd(index, dt, origin_format, storage_format);
    return *this;
  }

  OpExecuteLaunchContextFaker &InputTensor(std::vector<gert::Tensor *> input_tensor);
  OpExecuteLaunchContextFaker &OutputTensor(std::vector<gert::Tensor *> output_tensor);
  OpExecuteLaunchContextFaker &OpApiParams(void *param);
  OpExecuteLaunchContextFaker &WorkspaceSize(uint8_t* ws_size_vec);
  OpExecuteLaunchContextFaker &WorkspaceAddr(uint8_t* ws_addr_vec);
  OpExecuteLaunchContextFaker &Stream(void* stream);
  FakeKernelContextHolder Build();
 private:
  enum InputsAppend {kOpApiParams, kWorkspaceSize, kWorkspaceAddr, kStream, kEnd};
  void UpdateInputs();
  void UpdateOutputs();

  KernelRunContextFaker base_faker_;
  std::vector<gert::Tensor *> input_tensor_;
  std::vector<gert::Tensor *> output_tensor_;
  void *param_ = nullptr;
  uint8_t *ws_size_;
  uint8_t *ws_addr_;
  void *stream_;
};

struct DummyOpApiParams {
  uint8_t *dummy_data;
};


class EagerOpExecutionContextFaker {
 public:
  EagerOpExecutionContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  EagerOpExecutionContextFaker &IrInputNum(size_t input_num) {
    base_faker_.IrInputNum(input_num);
    return *this;
  }
  EagerOpExecutionContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num) {
    base_faker_.IrInstanceNum(std::move(instance_num));
    return *this;
  }
  EagerOpExecutionContextFaker &IrOutputInstanceNum(std::vector<uint32_t> output_instance_num) {
    base_faker_.IrOutputInstanceNum(std::move(output_instance_num));
    return *this;
  }
  EagerOpExecutionContextFaker &NodeInputTd(int32_t index, ge::DataType dt,
                                            ge::Format origin_format, ge::Format storage_format) {
    base_faker_.NodeInputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  EagerOpExecutionContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                             ge::Format storage_format) {
    base_faker_.NodeOutputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  EagerOpExecutionContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
    base_faker_.NodeAttrs(std::move(keys_to_value));
    return *this;
  }
  EagerOpExecutionContextFaker &InputTensor(std::vector<gert::Tensor *> input_tensor);
  EagerOpExecutionContextFaker &OutputTensor(std::vector<gert::Tensor *> output_tensor);
  EagerOpExecutionContextFaker &OutputMem(std::shared_ptr<std::vector<gert::GertMemBlock *>> &output_block_memory);
  EagerOpExecutionContextFaker &Allocator(void *allocator);
  EagerOpExecutionContextFaker &Stream(void *stream);
  EagerOpExecutionContextFaker &OpDesc(ge::OpDesc *op);
  EagerOpExecutionContextFaker &ExecuteFunc(void *execute_func);
  FakeKernelContextHolder Build();

 private:
  void UpdateInputs();
  void UpdateOutputs();
 private:
  enum InputsAppend {kAllocator, kStream, kOpDesc, kExecuteFunc, kEnd};

  KernelRunContextFaker base_faker_;
  std::vector<gert::Tensor *> input_tensor_;
  std::vector<gert::Tensor *> output_tensor_;
  std::shared_ptr<std::vector<gert::GertMemBlock *>> output_block_memory_;
  void *allocator_ = nullptr;
  ge::OpDesc *op_desc_ = nullptr;
  void *stream_ = nullptr;
  void *execute_func_ = nullptr;
};
}  // namespace gert
#endif  //AIR_CXX_TESTS_UT_GE_RUNTIME_V2_FAKER_KERNEL_RUN_CONTEXT_FACKER_H_

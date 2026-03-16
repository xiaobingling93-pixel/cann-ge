/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_run_context_faker.h"
#include "graph/compute_graph.h"
#include "exe_graph/lowering/bg_kernel_context_extend.h"
#include "exe_graph/runtime/tiling_context.h"

namespace gert {
FakeKernelContextHolder BuildKernelRunContext(size_t input_num, size_t output_num) {
  return KernelRunContextFaker().KernelIONum(input_num, output_num).Build();
}
KernelRunContextFaker &KernelRunContextFaker::KernelIONum(size_t input_num, size_t output_num) {
  kernel_input_num_ = input_num;
  kernel_output_num_ = output_num;
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::NodeIoNum(size_t input_num, size_t output_num) {
  node_input_num_ = input_num;
  node_output_num_ = output_num;
  node_input_tds_.resize(input_num);
  node_output_tds_.resize(output_num);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::IrInputNum(size_t input_num) {
  ir_instance_num_ = std::vector<uint32_t>(input_num, 1);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::IrInstanceNum(std::vector<uint32_t> instance_num) {
  ir_instance_num_ = std::move(instance_num);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::IrOutputNum(size_t output_num) {
  ir_output_instance_num_ = std::vector<uint32_t>(output_num, 1);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::IrOutputInstanceNum(std::vector<uint32_t> instance_num) {
  ir_output_instance_num_ = std::move(instance_num);
  return *this;
}
ge::OpDescPtr KernelRunContextFaker::FakeOp() const {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  size_t input_index = 0;
  for (size_t ir_index = 0; ir_index < ir_instance_num_.size(); ++ir_index) {
    auto ir_ins_num = ir_instance_num_[ir_index];
    auto prefix = "x_" + std::to_string(ir_index) + "_";
    op_desc->AppendIrInput(prefix, ge::kIrInputDynamic);
    for (size_t i = 0; i < ir_ins_num; ++i, ++input_index) {
      auto td = ge::GeTensorDesc();
      if (node_input_tds_.size() > input_index) {
        td.SetOriginFormat(node_input_tds_[input_index].GetOriginFormat());
        td.SetFormat(node_input_tds_[input_index].GetStorageFormat());
        td.SetDataType(node_input_tds_[input_index].GetDataType());
        td.SetOriginDataType(node_input_tds_[input_index].GetDataType());
      }
      op_desc->AddInputDesc(prefix + std::to_string(i), td);
    }
  }
  // fill it when not set
  std::vector<uint32_t> ir_output_instance_num;
  if (ir_output_instance_num_.empty()) {
    for (size_t i = 0; i < node_output_num_; ++i) {
      ir_output_instance_num.emplace_back(1U);
    }
  } else {
    ir_output_instance_num = ir_output_instance_num_;
  }
  size_t output_index = 0;
  for (size_t ir_index = 0; ir_index < ir_output_instance_num.size(); ++ir_index) {
    auto ir_ins_num = ir_output_instance_num[ir_index];
    auto prefix = "y_" + std::to_string(ir_index) + "_";
    op_desc->AppendIrOutput(prefix, ge::kIrOutputDynamic);
    for (size_t i = 0; i < ir_ins_num; ++i, ++output_index) {
      auto td = ge::GeTensorDesc();
      if (node_output_tds_.size() > output_index) {
        td.SetOriginFormat(node_output_tds_[output_index].GetOriginFormat());
        td.SetFormat(node_output_tds_[output_index].GetStorageFormat());
        td.SetDataType(node_output_tds_[output_index].GetDataType());
        td.SetOriginDataType(node_output_tds_[output_index].GetDataType());
      }
      op_desc->AddOutputDesc(prefix + std::to_string(i), td);
    }
  }

  for (const auto &attr : attrs_) {
    op_desc->AppendIrAttrName(attr.first);
    op_desc->SetAttr(attr.first, attr.second);
  }
  return op_desc;
}

FakeKernelContextHolder KernelRunContextFaker::Build() const {
  FakeKernelContextHolder fake_holder;
  fake_holder.kernel_input_num = kernel_input_num_;
  fake_holder.kernel_output_num = kernel_output_num_;
  KernelRunContextBuilder kernel_context_builder;
  auto op_desc = FakeOp();
  if (inputs_.size() != kernel_input_num_ || outputs_.size() != kernel_output_num_) {
    std::vector<void *> inputs(kernel_input_num_, nullptr);
    std::vector<void *> outputs(kernel_output_num_, nullptr);
    fake_holder.holder = kernel_context_builder.Inputs(inputs).Outputs(outputs).Build(op_desc);
    return fake_holder;
  }
  fake_holder.holder = kernel_context_builder.Inputs(inputs_).Outputs(outputs_).Build(op_desc);
  return fake_holder;
}
KernelRunContextFaker &KernelRunContextFaker::NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                                          ge::Format storage_format) {
  node_input_tds_[index].SetDataType(dt);
  node_input_tds_[index].SetOriginFormat(origin_format);
  node_input_tds_[index].SetStorageFormat(storage_format);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                                           ge::Format storage_format) {
  node_output_tds_[index].SetDataType(dt);
  node_output_tds_[index].SetOriginFormat(origin_format);
  node_output_tds_[index].SetStorageFormat(storage_format);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::Inputs(std::vector<void *> inputs) {
  inputs_ = std::move(inputs);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::Outputs(std::vector<void *> outputs) {
  outputs_ = std::move(outputs);
  return *this;
}
KernelRunContextFaker &
KernelRunContextFaker::NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
  attrs_ = std::move(keys_to_value);
  return *this;
}
InferShapeContextFaker &InferShapeContextFaker::NodeIoNum(size_t input_num, size_t output_num) {
  base_faker_.KernelIONum(input_num + kInputsAppendEnd, output_num);
  base_faker_.NodeIoNum(input_num, output_num);
  return *this;
}
InferShapeContextFaker &InferShapeContextFaker::InputShapes(std::vector<void *> input_shapes) {
  std::vector<void *> inputs(std::move(input_shapes));
  inputs.push_back(nullptr);  // infershape func
  base_faker_.Inputs(std::move(inputs));
  return *this;
}
InferShapeContextFaker &InferShapeContextFaker::OutputShapes(std::vector<void *> output_shapes) {
  base_faker_.Outputs(std::move(output_shapes));
  return *this;
}
FakeKernelContextHolder InferShapeContextFaker::Build() const {
  return base_faker_.Build();
}

InferSymbolShapeContextFaker &InferSymbolShapeContextFaker::NodeIoNum(size_t input_num, size_t output_num) {
  base_faker_.KernelIONum(input_num + kInputsAppendEnd, output_num);
  base_faker_.NodeIoNum(input_num, output_num);
  return *this;
}

InferSymbolShapeContextFaker &InferSymbolShapeContextFaker::IrInputNum(size_t input_num) {
  base_faker_.IrInputNum(input_num);
  return *this;
}

InferSymbolShapeContextFaker &InferSymbolShapeContextFaker::IrInputInstanceNum(std::vector<uint32_t> instance_num) {
  base_faker_.IrInstanceNum(std::move(instance_num));
  return *this;
}

InferSymbolShapeContextFaker &InferSymbolShapeContextFaker::NodeInputTd(int32_t index, ge::DataType dt,
                                                                        ge::Format origin_format,
                                                                        ge::Format storage_format) {
  base_faker_.NodeInputTd(index, dt, origin_format, storage_format);
  return *this;
}

InferSymbolShapeContextFaker &InferSymbolShapeContextFaker::NodeOutputTd(int32_t index, ge::DataType dt,
                                                                         ge::Format origin_format,
                                                                         ge::Format storage_format) {
  base_faker_.NodeOutputTd(index, dt, origin_format, storage_format);
  return *this;
}

InferSymbolShapeContextFaker &InferSymbolShapeContextFaker::NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
  base_faker_.NodeAttrs(std::move(keys_to_value));
  return *this;
}

InferSymbolShapeContextFaker &InferSymbolShapeContextFaker::Inputs(std::vector<void *> input_shapes) {
  std::vector<void *> inputs(std::move(input_shapes));
  inputs.push_back(nullptr);  // infershape func
  base_faker_.Inputs(std::move(inputs));
  return *this;
}

InferSymbolShapeContextFaker &InferSymbolShapeContextFaker::Outputs(std::vector<void *> output_shapes) {
  base_faker_.Outputs(std::move(output_shapes));
  return *this;
}

FakeKernelContextHolder InferSymbolShapeContextFaker::Build() const {
  return base_faker_.Build();
}

InferShapeRangeContextFaker &InferShapeRangeContextFaker::NodeIoNum(size_t input_num, size_t output_num) {
  base_faker_.KernelIONum(input_num + kInputsAppendEnd, output_num);
  base_faker_.NodeIoNum(input_num, output_num);
  return *this;
}
InferShapeRangeContextFaker &InferShapeRangeContextFaker::InputShapeRanges(std::vector<void *> input_shape_ranges) {
  std::vector<void *> inputs(std::move(input_shape_ranges));
  inputs.push_back(nullptr);  // infershaperange func
  base_faker_.Inputs(std::move(inputs));
  return *this;
}
InferShapeRangeContextFaker &InferShapeRangeContextFaker::OutputShapeRanges(std::vector<void *> output_shape_ranges) {
  base_faker_.Outputs(std::move(output_shape_ranges));
  return *this;
}
FakeKernelContextHolder InferShapeRangeContextFaker::Build() const {
    return base_faker_.Build();
}
InferDataTypeContextFaker &InferDataTypeContextFaker::NodeIoNum(size_t input_num, size_t output_num) {
  base_faker_.KernelIONum(input_num + kInputsAppendEnd, output_num);
  base_faker_.NodeIoNum(input_num, output_num);
  return *this;
}
InferDataTypeContextFaker &InferDataTypeContextFaker::InputDataTypes(std::vector<void *> input_datatypes) {
  std::vector<void *> inputs(std::move(input_datatypes));
  inputs_ = inputs;
  base_faker_.Inputs(std::move(inputs));
  return *this;
}
InferDataTypeContextFaker &InferDataTypeContextFaker::OutputDataTypes(std::vector<void *> output_datatypes) {
  outputs_ = output_datatypes;
  base_faker_.Outputs(std::move(output_datatypes));
  return *this;
}
FakeKernelContextHolder InferDataTypeContextFaker::Build() const {
  auto context_holder =  base_faker_.Build();
  auto origin_context = context_holder.GetContext<KernelContext>();
  for (size_t i = 0U; i < inputs_.size(); ++i) {
    memcpy_s(origin_context->MutableInputPointer<void *>(i), sizeof(void *), inputs_[i], sizeof(ge::DataType));
  }
  for (size_t i = 0U; i < outputs_.size(); ++i) {
    memcpy_s(origin_context->GetOutputPointer<void *>(i), sizeof(void *), outputs_[i], sizeof(ge::DataType));
  }
  return context_holder;
}

TilingContextFaker &TilingContextFaker::NodeIoNum(size_t input_num, size_t output_num) {
  base_faker_.KernelIONum(input_num + output_num + kInputsAppendEnd, gert::TilingContext::kOutputNum);
  base_faker_.NodeIoNum(input_num, output_num);
  return *this;
}
TilingContextFaker &TilingContextFaker::InputShapes(std::vector<gert::StorageShape *> input_shapes) {
  input_shapes_ = std::move(input_shapes);
  UpdateInputs();
  return *this;
}
TilingContextFaker &TilingContextFaker::OutputShapes(std::vector<gert::StorageShape *> output_shapes) {
  output_shapes_ = std::move(output_shapes);
  UpdateInputs();
  return *this;
}
TilingContextFaker &TilingContextFaker::CompileInfo(void *compile_info) {
  compile_info_ = compile_info;
  UpdateInputs();
  return *this;
}
TilingContextFaker &TilingContextFaker::PlatformInfo(void *platform_info) {
  platform_info_ = platform_info;
  UpdateInputs();
  return *this;
}
TilingContextFaker &TilingContextFaker::TilingData(void *tiling_data) {
  outputs_[TilingContext::kOutputTilingData] = tiling_data;
  base_faker_.Outputs(outputs_);
  return *this;
}
TilingContextFaker &TilingContextFaker::Workspace(ContinuousVector *workspace) {
  outputs_[TilingContext::kOutputWorkspace] = workspace;
  base_faker_.Outputs(outputs_);
  return *this;
}
FakeKernelContextHolder TilingContextFaker::Build() const {
  return base_faker_.Build();
}
void TilingContextFaker::UpdateInputs() {
  std::vector<void *> inputs;
  for (const auto input_shape : input_shapes_) {
    inputs.push_back(input_shape);
  }
  for (const auto output_shape : output_shapes_) {
    inputs.push_back(output_shape);
  }
  inputs.push_back(compile_info_);  // kInputsCompileInfo
  inputs.push_back(platform_info_);
  inputs.push_back(nullptr);        // kInputsTilingFunc
  base_faker_.Inputs(std::move(inputs));
}

OpExecuteContextFaker &OpExecuteContextFaker::NodeIoNum(size_t input_num, size_t output_num) {
  base_faker_.KernelIONum(input_num + output_num + kEnd, 1UL);
  base_faker_.NodeIoNum(input_num, output_num);
  return *this;
}

OpExecuteContextFaker &OpExecuteContextFaker::InputTensor(std::vector<gert::Tensor *> input_tensor) {
  input_tensor_ = std::move(input_tensor);
  return *this;
}

OpExecuteContextFaker &OpExecuteContextFaker::OutputTensor(std::vector<gert::Tensor *> output_tensor) {
  output_tensor_ = std::move(output_tensor);
  return *this;
}

OpExecuteContextFaker &OpExecuteContextFaker::OutputMem(std::shared_ptr<std::vector<gert::GertMemBlock *>> &output_block_memory) {
  output_block_memory_ = output_block_memory;
  return *this;
}

OpExecuteContextFaker &OpExecuteContextFaker::Allocate(void *allocator) {
  allocator_ = allocator;
  return *this;
}

OpExecuteContextFaker &OpExecuteContextFaker::Stream(void *stream) {
  stream_ = stream;
  return *this;
}
OpExecuteContextFaker &OpExecuteContextFaker::ExecuteOption(void *execute_option) {
  execute_option_ = execute_option;
  return *this;
}
OpExecuteContextFaker &OpExecuteContextFaker::ExecuteFunc(void *execute_func) {
  execute_func_ = execute_func;
  return *this;
}
OpExecuteContextFaker &OpExecuteContextFaker::OpAicoreNum(int64_t *op_aicore_num) {
  op_aicore_num_ = op_aicore_num;
  return *this;
}
OpExecuteContextFaker &OpExecuteContextFaker::OpVecCoreNum(int64_t *op_vec_core_num) {
  op_vec_core_num_ = op_vec_core_num;
  return *this;
}
OpExecuteContextFaker &OpExecuteContextFaker::GlobalAicoreNum(int64_t *global_aicore_num) {
  global_aicore_num_ = global_aicore_num;
  return *this;
}
OpExecuteContextFaker &OpExecuteContextFaker::GlobalVecCoreNum(int64_t *global_vec_core_num) {
  global_vec_core_num_ = global_vec_core_num;
  return *this;
}


void OpExecuteContextFaker::UpdateInputs() {
  std::vector<void *> inputs;
  for (const auto input_tensor : input_tensor_) {
    inputs.push_back(input_tensor);
  }
  for (const auto output_tensor : output_tensor_) {
    inputs.push_back(output_tensor);
  }
  inputs.push_back(allocator_);
  inputs.push_back(stream_);
  inputs.push_back(execute_option_);
  inputs.push_back(execute_func_);
  inputs.push_back(reinterpret_cast<void*>(*op_aicore_num_));
  inputs.push_back(reinterpret_cast<void*>(*op_vec_core_num_));
  inputs.push_back(reinterpret_cast<void*>(*global_aicore_num_));
  inputs.push_back(reinterpret_cast<void*>(*global_vec_core_num_));
  base_faker_.Inputs(std::move(inputs));
}

void OpExecuteContextFaker::UpdateOutputs() {
  std::vector<void *> outputs;
  outputs.push_back(output_block_memory_.get());
  base_faker_.Outputs(std::move(outputs));
}

FakeKernelContextHolder OpExecuteContextFaker::Build() {
  UpdateInputs();
  UpdateOutputs();
  return base_faker_.Build();
}

OpExecutePrepareContextFaker &OpExecutePrepareContextFaker::NodeIoNum(size_t input_num, size_t output_num) {
  base_faker_.KernelIONum(input_num + output_num + kEnd, 2UL);
  base_faker_.NodeIoNum(input_num, output_num);
  return *this;
}

OpExecutePrepareContextFaker &OpExecutePrepareContextFaker::InputTensor(std::vector<gert::Tensor *> input_tensor) {
  input_tensor_ = std::move(input_tensor);
  return *this;
}

OpExecutePrepareContextFaker &OpExecutePrepareContextFaker::OutputTensor(std::vector<gert::Tensor *> output_tensor) {
  output_tensor_ = std::move(output_tensor);
  return *this;
}

OpExecutePrepareContextFaker &OpExecutePrepareContextFaker::ExecuteOption(void *execute_option) {
  execute_option_ = execute_option;
  return *this;
}
OpExecutePrepareContextFaker &OpExecutePrepareContextFaker::ExecuteFunc(void *execute_func) {
  execute_func_ = execute_func;
  return *this;
}

OpExecutePrepareContextFaker &OpExecutePrepareContextFaker::OpApiParams(void *param) {
  param_ = param;
  return *this;
}
OpExecutePrepareContextFaker &OpExecutePrepareContextFaker::WorkspaceSize(uint8_t* ws_size_vec) {
  ws_size_ = ws_size_vec;
  return *this;
}

FakeKernelContextHolder OpExecutePrepareContextFaker::Build() {
  UpdateInputs();
  UpdateOutputs();
  return base_faker_.Build();
}

void OpExecutePrepareContextFaker::UpdateInputs() {
  std::vector<void *> inputs;
  for (const auto input_tensor : input_tensor_) {
    inputs.push_back(input_tensor);
  }
  for (const auto output_tensor : output_tensor_) {
    inputs.push_back(output_tensor);
  }
  inputs.push_back(execute_option_);
  inputs.push_back(execute_func_);
  base_faker_.Inputs(std::move(inputs));
}

void OpExecutePrepareContextFaker::UpdateOutputs() {
  std::vector<void *> outputs;
  outputs.push_back(param_);
  outputs.push_back(ws_size_);
  base_faker_.Outputs(std::move(outputs));
}

OpExecuteLaunchContextFaker &OpExecuteLaunchContextFaker::NodeIoNum(size_t input_num, size_t output_num) {
  base_faker_.KernelIONum(input_num + output_num + kEnd, 0UL);
  base_faker_.NodeIoNum(input_num, output_num);
  return *this;
}

OpExecuteLaunchContextFaker &OpExecuteLaunchContextFaker::InputTensor(std::vector<gert::Tensor *> input_tensor) {
  input_tensor_ = std::move(input_tensor);
  return *this;
}

OpExecuteLaunchContextFaker &OpExecuteLaunchContextFaker::OutputTensor(std::vector<gert::Tensor *> output_tensor) {
  output_tensor_ = std::move(output_tensor);
  return *this;
}

OpExecuteLaunchContextFaker &OpExecuteLaunchContextFaker::OpApiParams(void *param) {
  param_ = param;
  return *this;
}
OpExecuteLaunchContextFaker &OpExecuteLaunchContextFaker::WorkspaceSize(uint8_t* ws_size_vec) {
  ws_size_ = ws_size_vec;
  return *this;
}

OpExecuteLaunchContextFaker &OpExecuteLaunchContextFaker::WorkspaceAddr(uint8_t* ws_addr_vec) {
  ws_addr_ = ws_addr_vec;
  return *this;
}

OpExecuteLaunchContextFaker &OpExecuteLaunchContextFaker::Stream(void* stream) {
  stream_ = stream;
  return *this;
}

FakeKernelContextHolder OpExecuteLaunchContextFaker::Build() {
  UpdateInputs();
  UpdateOutputs();
  return base_faker_.Build();
}

void OpExecuteLaunchContextFaker::UpdateInputs() {
  std::vector<void *> inputs;
  for (const auto input_tensor : input_tensor_) {
    inputs.push_back(input_tensor);
  }
  for (const auto output_tensor : output_tensor_) {
    inputs.push_back(output_tensor);
  }
  inputs.push_back(param_);
  inputs.push_back(ws_addr_);
  inputs.push_back(ws_size_);
  inputs.push_back(stream_);
  base_faker_.Inputs(std::move(inputs));
}

void OpExecuteLaunchContextFaker::UpdateOutputs() {
  std::vector<void *> outputs;
  base_faker_.Outputs(std::move(outputs));
}

EagerOpExecutionContextFaker &EagerOpExecutionContextFaker::NodeIoNum(size_t input_num, size_t output_num) {
  base_faker_.KernelIONum(input_num + kEnd, output_num + 1UL);
  base_faker_.NodeIoNum(input_num, output_num);
  return *this;
}

EagerOpExecutionContextFaker &EagerOpExecutionContextFaker::InputTensor(std::vector<gert::Tensor *> input_tensor) {
  input_tensor_ = std::move(input_tensor);
  return *this;
}

EagerOpExecutionContextFaker &EagerOpExecutionContextFaker::OutputTensor(std::vector<gert::Tensor *> output_tensor) {
  output_tensor_ = std::move(output_tensor);
  return *this;
}

EagerOpExecutionContextFaker &EagerOpExecutionContextFaker::OutputMem(std::shared_ptr<std::vector<gert::GertMemBlock *>> &output_block_memory) {
  output_block_memory_ = output_block_memory;
  return *this;
}

EagerOpExecutionContextFaker &EagerOpExecutionContextFaker::Allocator(void *allocator) {
  allocator_ = allocator;
  return *this;
}

EagerOpExecutionContextFaker &EagerOpExecutionContextFaker::OpDesc(ge::OpDesc *op) {
  op_desc_ = op;
  return *this;
}

EagerOpExecutionContextFaker &EagerOpExecutionContextFaker::Stream(void *stream) {
  stream_ = stream;
  return *this;
}

EagerOpExecutionContextFaker &EagerOpExecutionContextFaker::ExecuteFunc(void *execute_func) {
  execute_func_ = execute_func;
  return *this;
}

void EagerOpExecutionContextFaker::UpdateInputs() {
  std::vector<void *> inputs;
  for (const auto input_tensor : input_tensor_) {
    inputs.push_back(input_tensor);
  }
  inputs.push_back(allocator_);
  inputs.push_back(stream_);
  inputs.push_back(op_desc_);
  inputs.push_back(execute_func_);
  base_faker_.Inputs(std::move(inputs));
}

void EagerOpExecutionContextFaker::UpdateOutputs() {
  std::vector<void *> outputs;
  for (const auto output_tensor : output_tensor_) {
    outputs.push_back(output_tensor);
  }
  outputs.push_back(output_block_memory_.get());
  base_faker_.Outputs(std::move(outputs));
}

FakeKernelContextHolder EagerOpExecutionContextFaker::Build() {
  UpdateInputs();
  UpdateOutputs();
  return base_faker_.Build();
}
}  // namespace gert

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid_model_executor.h"
#include "common/memory/tensor_trans_utils.h"
#include "graph/ge_context.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace hybrid {
namespace {
constexpr int32_t kDataOutputFirstIndex = 0;
constexpr uint32_t kPlaceDeviceData = 1U;
const size_t kValAlignment = 64U;
}

Status HybridModelExecutor::InitInputDesc() {
  int32_t input_index = 0;
  for (const auto &input_node : model_->GetRootGraphItem()->GetInputNodes()) {
    GELOGD("Init input[%u], node = %s, is_dynamic = %d", input_index,
           input_node->NodeName().c_str(), static_cast<int32_t>(input_node->is_dynamic));
    auto output_desc = input_node->MutableOutputDesc(kDataOutputFirstIndex);
    GE_CHECK_NOTNULL(output_desc);
    int64_t tensor_size = -1;
    if (!input_node->is_dynamic) {
      GE_CHK_GRAPH_STATUS_RET(TensorUtils::GetSize(*output_desc, tensor_size),
                              "[Get][Size] from %s failed",
                              input_node->NodeName().c_str());

      if (tensor_size == 0) {
        GELOGW("[%s] Tensor size == 0", input_node->NodeName().c_str());
        GE_CHK_GRAPH_STATUS_RET(TensorUtils::GetTensorMemorySizeInBytes(*output_desc, tensor_size),
                                "[Get][TensorMemorySize] Failed to calc tensor size");
        GELOGD("[%s] Tensor size updated to %ld", input_node->NodeName().c_str(), tensor_size);
      }
    }

    (void)index_to_tensor_size_.emplace(input_index, tensor_size);
    (void)index_to_tensor_desc_.emplace(input_index, output_desc);
    is_input_dynamic_.push_back(input_node->is_dynamic);
    input_index += 1;
  }
  return SUCCESS;
}

Status HybridModelExecutor::SyncVarData() const {
  GELOGI("Sync var data, model id:%u", model_id_);
  TensorValue *const global_step_var = model_->GetVariable(NODE_NAME_GLOBAL_STEP);
  if (global_step_var != nullptr) {
    std::vector<uint64_t> v_step;
    v_step.push_back(iterator_count_);
    GE_CHK_RT_RET(rtMemcpy(global_step_var->MutableData(),
                           global_step_var->GetSize(),
                           v_step.data(),
                           v_step.size() * sizeof(uint64_t),
                           RT_MEMCPY_HOST_TO_DEVICE));
  } else {
    GELOGD("No GLOBAL_STEP variable was found.");
  }
  return SUCCESS;
}

Status HybridModelExecutor::PrepareDynamicInput(HybridModelExecutor::ExecuteArgs &args, const size_t input_index,
                                                const GeShape &shape, const DataBuffer &data_buf,
                                                int64_t &tensor_size) {
  auto &tensor_desc = index_to_tensor_desc_[input_index];
  std::vector<std::pair<int64_t, int64_t>> range;
  const auto range_ret = tensor_desc->GetShapeRange(range);
  GE_CHK_BOOL_RET_STATUS(range_ret == GRAPH_SUCCESS, INTERNAL_ERROR,
                         "[Invoke][GetShapeRange] failed, ret=%u, model_id = %u.", range_ret, model_id_);
  // one-node-multiple-bin mode does not need to check shape range which will be modified in fuzz compile
  if (model_->GetNodeBinMode() == fuzz_compile::kOneNodeSingleBinMode) {
    for (size_t k = 0U; k < range.size(); ++k) {
      if (k >= shape.GetDimNum()) {
        break;
      }
      // range[k].second can be -1
      const bool is_out_of_range = (shape.GetDim(k) < range[k].first) ||
                                    ((range[k].second >= 0) && (shape.GetDim(k) > range[k].second));
      if (is_out_of_range) {
        GELOGE(PARAM_INVALID,
               "[Check][Range]Dim out of range, shape idx = %zu, dim idx = %zu,"
               "dim = %ld, range = [%ld, %ld], model_id = %u.",
               input_index, k, shape.GetDim(k), range[k].first, range[k].second, model_id_);
        REPORT_INNER_ERR_MSG("E19999",
                           "Dim out of range, shape idx = %zu, dim idx = %zu, dim = %" PRId64 ","
                           "range = [%" PRId64 ", %" PRId64 "], model_id = %u.",
                           input_index, k, shape.GetDim(k), range[k].first, range[k].second, model_id_);
        return PARAM_INVALID;
      }
    }
  }
  tensor_desc->SetShape(shape);
  tensor_desc->SetOriginShape(shape);
  GELOGD("Update shape[%s] of input[%zu] to [%s]",
         shape.ToString().c_str(), input_index, tensor_desc->MutableShape().ToString().c_str());
  if (tensor_desc->GetDataType() == DT_STRING) {
    tensor_size = static_cast<int64_t>(data_buf.length);
  } else {
    GE_CHK_GRAPH_STATUS_RET(TensorUtils::GetTensorMemorySizeInBytes(*tensor_desc, tensor_size),
                            "[Invoke][GetTensorMemorySizeInBytes]Failed to calc tensor size,"
                            "index = %zu, shape = [%s], model_id = %u.",
                            input_index, tensor_desc->GetShape().ToString().c_str(), model_id_);
  }
  GELOGD("Input tensor[%zu] size = %ld", input_index, tensor_size);
  TensorUtils::SetSize(*tensor_desc, tensor_size);
  args.input_desc[input_index] = tensor_desc;
  return SUCCESS;
}

Status HybridModelExecutor::CopyDataToExecutArgs(const int64_t tensor_size, HybridModelExecutor::ExecuteArgs &args,
                                                 const size_t input_index, const DataBuffer &data_buf) const {
  const auto mem_size = static_cast<uint64_t>(tensor_size);
  if (mem_size < data_buf.length) {
    REPORT_INNER_ERR_MSG("E19999",
                       "input data size(%" PRIu64 ") does not match model required size(%" PRIu64 "), "
		                   "ret failed, model_id = %u.", data_buf.length, mem_size, model_id_);
    GELOGE(PARAM_INVALID,
           "[Check][Size]input data size(%lu) does not match model required size(%lu), ret failed, model_id = %u.",
           data_buf.length, mem_size, model_id_);
    return PARAM_INVALID;
  }

  if (data_buf.placement == kPlaceDeviceData) {
    args.inputs.emplace_back(data_buf.data, data_buf.length);
    return SUCCESS;
  }
  AllocationAttr attr;
  if (ge::GetContext().GetHostExecFlag()) {
    attr.SetMemType(MemStorageType::HOST_DDR);
  }
  const auto allocator = NpuMemoryAllocator::GetAllocator();
  GE_CHECK_NOTNULL(allocator);
  auto tensor_buffer = TensorBuffer::Create(allocator, static_cast<size_t>(tensor_size), &attr);
  GE_CHECK_NOTNULL(tensor_buffer);
  args.inputs.emplace_back(std::shared_ptr<TensorBuffer>(tensor_buffer.release()));

  GELOGD("To copy input data for input[%zu]", input_index);
  if (data_buf.length > 0U) {
    GELOGI("[IMAS]CopyPlainData memcpy graph_%u type[F] output[%zu] memaddr[%p] mem_size[%zu] datasize[%lu]",
           model_->GetRootGraph() != nullptr ? model_->GetRootGraph()->GetGraphID() : 0,
           input_index,
           args.inputs[input_index].GetData(),
           mem_size,
           data_buf.length);
    GE_CHK_RT_RET(rtMemcpy(args.inputs[input_index].MutableData(),
                           mem_size,
                           data_buf.data,
                           data_buf.length,
                           RT_MEMCPY_HOST_TO_DEVICE));
  }
  return SUCCESS;
}

Status HybridModelExecutor::PrepareExecuteArgs(const InputData &current_data,
                                               HybridModelExecutor::ExecuteArgs &args) {
  if (current_data.blobs.size() < index_to_tensor_desc_.size()) {
    GELOGE(PARAM_INVALID,
           "[Check][Size]Blob size mismatches, expect at least %zu, but got %zu, model_id = %u",
           index_to_tensor_desc_.size(), current_data.blobs.size(), model_id_);
    REPORT_INNER_ERR_MSG("E19999", "Blob size mismatches, expect at least %zu, but got %zu, model_id = %u.",
                       index_to_tensor_desc_.size(), current_data.blobs.size(), model_id_);
    return PARAM_INVALID;
  }

  args.input_desc.resize(index_to_tensor_desc_.size());
  const std::vector<DataBuffer> &blobs = current_data.blobs;
  for (size_t input_index = 0U; input_index < index_to_tensor_desc_.size(); ++input_index) {
    auto tensor_size = index_to_tensor_size_[input_index];
    if (is_input_dynamic_[input_index]) {
      if (input_index >= current_data.shapes.size()) {
        GELOGE(PARAM_INVALID,
               "[Check][Range]Shape index out of range, index = %zu, shape size = "
	             "%zu model_id = %u.", input_index, current_data.shapes.size(), model_id_);
        REPORT_INNER_ERR_MSG("E19999", "Shape index out of range, index = %zu, shape size = %zu, model_id = %u.",
                           input_index, current_data.shapes.size(), model_id_);
        return PARAM_INVALID;
      }
      const GeShape shape(current_data.shapes[input_index]);
      const DataBuffer &data_buf = blobs[input_index];
      GE_CHK_STATUS_RET(PrepareDynamicInput(args, input_index, shape, data_buf, tensor_size),
                        "Prepare Dynamic input failed for index = %zu", input_index);
    }

    GE_CHECK_GE(tensor_size, 0);
    const DataBuffer &data_buf = blobs[input_index];
    GE_CHK_STATUS_RET(CopyDataToExecutArgs(tensor_size, args, input_index, data_buf),
                      "Copy input data failed for index = %zu", input_index);
  }
  return SUCCESS;
}

Status HybridModelExecutor::OnComputeDone(const uint32_t data_index, const uint32_t result_code,
                                          std::vector<ge::Tensor> &outputs,
                                          const std::shared_ptr<ModelListener> listener) const {
  GELOGD("OnComputeDone. model id = %u, data index = %u, execution ret = %u", model_id_, data_index, result_code);
  if (listener != nullptr) {
    std::vector<gert::Tensor> gert_outputs;
    GE_ASSERT_SUCCESS(TensorTransUtils::Tensors2GertTensors(outputs, gert_outputs));
    GE_CHK_STATUS(listener->OnComputeDone(model_id_, data_index, result_code, gert_outputs),
                  "[Invoke][OnComputeDone] failed, model_id = %u.", model_id_);
  }
  return result_code;
}

Status HybridModelExecutor::OnComputeDone(const uint32_t data_index, const uint32_t result_code,
                                          std::vector<gert::Tensor> &outputs,
                                          const std::shared_ptr<ModelListener> listener) const {
  GELOGD("OnComputeDone. model id = %u, data index = %u, execution ret = %u", model_id_, data_index, result_code);
  if (listener != nullptr) {
    GE_CHK_STATUS(listener->OnComputeDone(model_id_, data_index, result_code, outputs),
                  "[Invoke][OnComputeDone] failed, model_id = %u.", model_id_);
  }
  return result_code;
}

/*
 *  args是输入，output_data和outputs是输出
 *  1. 根据shape重新计算大小，申请host内存，并把数据从args中拷贝过去。这块内存生命周期由outputs管理，output_data只是引用。
 *     为什么还需要重新计算大小呢？难道是担心args中的size是加了padding了？
 *  2. shape使用args.output_desc上的，更新到outputs中。
 */
Status HybridModelExecutor::CopyOutputs(HybridModelExecutor::ExecuteArgs &args, OutputData *const output_data,
                                        std::vector<ge::Tensor> &outputs) const {
  // copy output data from op to designated position
  std::vector<ConstGeTensorDescPtr> &output_tensor_desc_list = args.output_desc;
  std::vector<TensorValue> &output_tensors = args.outputs;
  if (output_tensor_desc_list.size() != output_tensors.size()) {
    GELOGE(INTERNAL_ERROR,
           "[Check][Size]Output sizes mismatch. From op_desc = %zu, and from output tensors = %zu, model_id = %u.",
           output_tensor_desc_list.size(), output_tensors.size(), model_id_);
    REPORT_INNER_ERR_MSG("E19999",
                       "Output sizes mismatch. From op_desc = %zu, and from output tensors = %zu, model_id = %u.",
                       output_tensor_desc_list.size(), output_tensors.size(), model_id_);
    return INTERNAL_ERROR;
  }

  GELOGD("Number of outputs = %zu", output_tensor_desc_list.size());
  for (size_t i = 0U; i < output_tensors.size(); ++i) {
    GELOGD("Start to process output[%zu]", i);
    auto &output_tensor = output_tensors[i];
    auto &tensor_desc = output_tensor_desc_list.at(i);
    GE_CHECK_NOTNULL(tensor_desc);
    int64_t output_size = -1;
    if (tensor_desc->GetDataType() == DT_STRING) {
      output_size = static_cast<int64_t>(output_tensor.GetSize());
    } else {
      GE_CHK_GRAPH_STATUS_RET(TensorUtils::CalcTensorMemSize(tensor_desc->GetShape(),
                                                             tensor_desc->GetFormat(),
                                                             tensor_desc->GetDataType(),
                                                             output_size),
                              "[Calc][TensorMemSize]Failed for output[%zu]. shape = [%s], type = %s, format = %s",
                              i,
                              tensor_desc->GetShape().ToString().c_str(),
                              TypeUtils::DataTypeToSerialString(tensor_desc->GetDataType()).c_str(),
                              TypeUtils::FormatToSerialString(tensor_desc->GetFormat()).c_str());
    }
    GELOGD("Got tensor size for output[%zu] successfully. shape = [%s], type = %s, format = %s, size = %ld",
           i,
           tensor_desc->GetShape().ToString().c_str(),
           TypeUtils::DataTypeToSerialString(tensor_desc->GetDataType()).c_str(),
           TypeUtils::FormatToSerialString(tensor_desc->GetFormat()).c_str(),
           output_size);

    GE_CHECK_GE(output_size, 0);
    if (output_tensor.GetSize() < static_cast<size_t>(output_size)) {
      GELOGE(INTERNAL_ERROR,
             "[Check][Size]output[%zu] tensor size(%zu) is not enough for output shape [%s], model_id = %u.",
             i, output_tensor.GetSize(), tensor_desc->GetShape().ToString().c_str(), model_id_);
      REPORT_INNER_ERR_MSG("E19999", "output[%zu] tensor size(%zu) is not enough for output shape [%s] model_id = %u",
                         i, output_tensor.GetSize(), tensor_desc->GetShape().ToString().c_str(), model_id_);
      return INTERNAL_ERROR;
    }

    const GeShape ge_shape(tensor_desc->GetShape().GetDims());
    GeTensorDesc ge_tensor_desc;
    ge_tensor_desc.SetShape(ge_shape);
    if (output_size > 0) {
      if (NeedBuildDeviceTensorAsOutput()) {
        GE_CHK_STATUS_RET(BuildDeviceTensor(output_tensor, ge_tensor_desc, output_size, outputs),
                          "[Build][DeviceTensor] failed");
        output_data->blobs.emplace_back(output_tensor.Release(), static_cast<uint32_t>(output_size), false,
                                        static_cast<uint32_t>(kPlacementDevice));
      } else {
        const auto aligned_ptr = MakeShared<AlignedPtr>(output_size, kValAlignment);
        GE_CHECK_NOTNULL(aligned_ptr);
        auto data_buf = aligned_ptr->MutableGet();
        GE_CHECK_NOTNULL(data_buf);
        GE_CHK_RT_RET(rtMemcpy(data_buf, static_cast<uint64_t>(output_size), output_tensor.GetData(),
                               static_cast<uint64_t>(output_size), RT_MEMCPY_DEVICE_TO_HOST));
        GeTensor ge_tensor(ge_tensor_desc);
        ge_tensor.SetData(aligned_ptr, static_cast<size_t>(output_size));
        output_data->blobs.emplace_back(data_buf, static_cast<uint32_t>(output_size), false);
        auto tensor = TensorAdapter::AsTensor(ge_tensor);
        outputs.emplace_back(std::move(tensor));
      }
    } else {
      GELOGW("Output [%zu] is empty. shape = [%s]", i, tensor_desc->GetShape().ToString().c_str());
      GeTensor ge_tensor(ge_tensor_desc);
      (void)ge_tensor.SetData(nullptr, 0U);
      output_data->blobs.emplace_back(nullptr, 0U, false);
      auto tensor = TensorAdapter::AsTensor(ge_tensor);
      outputs.emplace_back(std::move(tensor));
    }
    GELOGD("Output[%zu] added, type = %s, shape = [%s], size = %ld", i,
           TypeUtils::DataTypeToSerialString(tensor_desc->GetDataType()).c_str(),
           tensor_desc->GetShape().ToString().c_str(), output_size);
  }

  return SUCCESS;
}

/*
 *  1. 根据shape重新计算大小，申请host内存，并把数据从executor_outputs中拷贝过去。这块内存生命周期由uer_outputs管理.
 *     为什么还需要重新计算大小呢？executor_outputs中内存大小可能加了padding
 *  2. shape使用executor_outputs上的，更新到outputs中。
 */
Status HybridModelExecutor::CopyOutputs(const std::vector<gert::Tensor> &executor_outputs,
  std::vector<gert::Tensor> &uer_outputs) const {
  uer_outputs.clear();
  uer_outputs.reserve(executor_outputs.size());
  for (size_t i = 0U; i < executor_outputs.size(); ++i) {
    GELOGD("Start to process output[%zu]", i);
    const auto &arg_output = executor_outputs.at(i);
    const auto ge_shape = TensorTransUtils::ContructGeShapeFromRtShape(arg_output.GetShape().GetStorageShape());
    int64_t output_size = -1;
    if (arg_output.GetDataType() == DT_STRING) {
      output_size = static_cast<int64_t>(arg_output.GetSize());
    } else {
      GE_CHK_GRAPH_STATUS_RET(TensorUtils::CalcTensorMemSize(ge_shape,
                                                             arg_output.GetStorageFormat(),
                                                             arg_output.GetDataType(),
                                                             output_size),
                              "[Calc][TensorMemSize]Failed for output[%zu]. shape = [%s], type = %s, format = %s",
                              i,
                              ge_shape.ToString().c_str(),
                              TypeUtils::DataTypeToSerialString(arg_output.GetDataType()).c_str(),
                              TypeUtils::FormatToSerialString(arg_output.GetStorageFormat()).c_str());
    }
    GELOGD("Got tensor size for output[%zu] successfully. shape = [%s], type = %s, format = %s, size = %ld", i,
           ge_shape.ToString().c_str(), TypeUtils::DataTypeToSerialString(arg_output.GetDataType()).c_str(),
           TypeUtils::FormatToSerialString(arg_output.GetStorageFormat()).c_str(), output_size);

    GE_CHECK_GE(output_size, 0);
    if (arg_output.GetSize() < static_cast<size_t>(output_size)) {
      GELOGE(INTERNAL_ERROR,
             "[Check][Size]output[%zu] tensor size(%zu) is not enough for output shape [%s], model_id = %u.",
             i, arg_output.GetSize(), ge_shape.ToString().c_str(), model_id_);
      REPORT_INNER_ERR_MSG("E19999", "output[%zu] tensor size(%zu) is not enough for output shape [%s] model_id = %u",
                         i, arg_output.GetSize(), ge_shape.ToString().c_str(), model_id_);
      return INTERNAL_ERROR;
    }

    if (output_size > 0) {
      if (NeedBuildDeviceTensorAsOutput()) {
        gert::Tensor copy_tensor(arg_output.GetShape(), arg_output.GetFormat(), arg_output.GetDataType());
        copy_tensor.MutableTensorData().ShareFrom(arg_output.GetTensorData());
        copy_tensor.MutableTensorData().SetSize(output_size);
        uer_outputs.emplace_back(std::move(copy_tensor));
      } else {
        // size使用没有padding的
        const auto aligned_ptr = MakeShared<AlignedPtr>(output_size, kValAlignment);
        GE_CHECK_NOTNULL(aligned_ptr);
        auto data_buf = aligned_ptr->MutableGet();
        GE_CHECK_NOTNULL(data_buf);
        GE_CHK_RT_RET(rtMemcpy(data_buf, static_cast<uint64_t>(output_size), arg_output.GetAddr(),
                               static_cast<uint64_t>(output_size), RT_MEMCPY_DEVICE_TO_HOST));
        GeTensor ge_tensor;
        ge_tensor.SetData(aligned_ptr, static_cast<size_t>(output_size));
        gert::Tensor host_tensor;
        GE_ASSERT_SUCCESS(TensorTransUtils::GeTensor2GertTensor(ge_tensor, host_tensor));

        // shape, format, data type
        host_tensor.MutableFormat() = arg_output.GetFormat();
        host_tensor.SetDataType(arg_output.GetDataType());
        host_tensor.MutableOriginShape() = arg_output.GetOriginShape();
        host_tensor.MutableStorageShape() = arg_output.GetStorageShape();
        host_tensor.MutableTensorData().SetPlacement(gert::TensorPlacement::kOnHost);
        uer_outputs.emplace_back(std::move(host_tensor));
      }
    } else {
      gert::Tensor copy_tensor(arg_output.GetShape(), arg_output.GetFormat(), gert::TensorPlacement::kOnHost,
        arg_output.GetDataType(), nullptr);
      GELOGW("Output [%zu] is empty. shape size = [%ld]", i, arg_output.GetStorageShape().GetShapeSize());
      uer_outputs.emplace_back(std::move(copy_tensor));
    }
    GELOGD("Output[%zu] added, type = %s, shape = [%s], size = %ld", i,
           TypeUtils::DataTypeToSerialString(arg_output.GetDataType()).c_str(),
           ge_shape.ToString().c_str(), output_size);
  }
  return SUCCESS;
}

void HybridModelExecutor::GenDataInputOutputData(const uint32_t model_id, const std::vector<gert::Tensor> &inputs,
    InputData &input_data, OutputData &output_data) const {
  input_data.model_id = model_id;
  input_data.timeout = 0U;
  input_data.timestamp = 0U;
  input_data.index = 0U;
  input_data.blobs.reserve(inputs.size());
  for (size_t i = 0U; i < inputs.size(); ++i) {
    input_data.shapes.emplace_back(TensorTransUtils::GetDimsFromGertShape(inputs[i].GetStorageShape()));
    DataBuffer data_blob;
    data_blob.data = ValueToPtr(PtrToValue(inputs[i].GetAddr()));
    data_blob.length = inputs[i].GetSize();
    data_blob.placement = static_cast<uint32_t>(gert::TensorPlacementUtils::IsOnDevice(inputs[i].GetPlacement()) ?
      Placement::kPlacementDevice : Placement::kPlacementHost);
    input_data.blobs.push_back(data_blob);
  }
  output_data.model_id = model_id;
  output_data.index = 0U;
}

// 当前几个执行器的处理逻辑一致，放到了基类实现，后面实现不同可以在子类中重新实现
Status HybridModelExecutor::HandleResult(const Status exec_ret,
                                         const uint32_t data_id,
                                         HybridModelExecutor::ExecuteArgs &args,
                                         OutputData *const output_data,
                                         std::shared_ptr<ModelListener> listener) const {
  GELOGD("Start to handle result. model id = %u, data index = %u, execution ret = %u", model_id_, data_id, exec_ret);
  std::vector<ge::Tensor> output_tensor_info_list;
  if (args.ctrl_args.is_eos) {
    GELOGI("End of sequence, model id = %u.", model_id_);
    GE_CHK_STATUS_RET_NOLOG(OnComputeDone(data_id, END_OF_SEQUENCE, output_tensor_info_list, listener));
    return SUCCESS;
  }

  if (exec_ret != SUCCESS) {
    GELOGE(exec_ret, "[Check][Param:Status] failed to execute graph. model_id = %u", model_id_);
    REPORT_INNER_ERR_MSG("E19999", "failed to execute graph. model_id = %u", model_id_);
    return OnComputeDone(data_id, INTERNAL_ERROR, output_tensor_info_list, listener);
  }

  GE_CHECK_NOTNULL(output_data);
  const auto ret = CopyOutputs(args, output_data, output_tensor_info_list);
  if (ret != SUCCESS) {
    (void)OnComputeDone(data_id, INTERNAL_ERROR, output_tensor_info_list, listener);
    return INTERNAL_ERROR;
  }

  GELOGD("Executed graph successfully, model id = %u, data_index = %u.", model_id_, data_id);
  return OnComputeDone(data_id, SUCCESS, output_tensor_info_list, listener);
}

// 当前几个执行器的处理逻辑一致，放到了基类实现，后面实现不同可以在子类中重新实现
Status HybridModelExecutor::HandleResult(const Status exec_ret, const uint32_t data_id,
  HybridModelExecutor::CtrlArgs &ctrl_args, std::vector<gert::Tensor> &outputs,
  std::shared_ptr<ModelListener> listener) const {
  GELOGD("Start to handle result. model id = %u, data index = %u, execution ret = %u", model_id_, data_id, exec_ret);
  std::vector<gert::Tensor> host_outputs;
  if (ctrl_args.is_eos) {
    GELOGI("End of sequence, model id = %u.", model_id_);
    GE_CHK_STATUS_RET_NOLOG(OnComputeDone(data_id, END_OF_SEQUENCE, host_outputs, listener));
    return SUCCESS;
  }

  if (exec_ret != SUCCESS) {
    GELOGE(exec_ret, "[Check][Param:Status] failed to execute graph. model_id = %u", model_id_);
    REPORT_INNER_ERR_MSG("E19999", "failed to execute graph. model_id = %u", model_id_);
    return OnComputeDone(data_id, INTERNAL_ERROR, host_outputs, listener);
  }

  const auto ret = CopyOutputs(outputs, host_outputs);
  if (ret != SUCCESS) {
    (void)OnComputeDone(data_id, INTERNAL_ERROR, host_outputs, listener);
    return INTERNAL_ERROR;
  }

  GELOGD("Executed graph successfully, model id = %u, data_index = %u.", model_id_, data_id);
  return OnComputeDone(data_id, SUCCESS, host_outputs, listener);
}

void HybridModelExecutor::ParserContextOption(const string &option_name, string &option_value) {
  auto result = ge::GetContext().GetOption(option_name, option_value);
  if (result != SUCCESS) {
    GELOGW("Can not get %s attr.", option_name.c_str());
  }
  GELOGD("The %s is %s.", option_name.c_str(), option_value.c_str());
}

Status HybridModelExecutor::ExecuteWithStreamAsync(const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs,
                                                   const aclrtStream stream) {
  (void)inputs;
  (void)outputs;
  (void)stream;
  GELOGE(ge::GRAPH_FAILED, "ExecuteWithStreamAsync only support dynamic model with rt2 executor currently!");
  return ge::FAILED;
}

Status HybridModelExecutor::ExecuteWithStreamAsync(const std::vector<gert::Tensor> &inputs,
                                                                  std::vector<gert::Tensor> &outputs,
                                                                  const aclrtStream stream) {
  (void)inputs;
  (void)outputs;
  (void)stream;
  GELOGE(ge::GRAPH_FAILED, "ExecuteWithStreamAsync only support dynamic model with rt2 executor currently!");
  return ge::FAILED;
}

Status HybridModelExecutor::BuildDeviceTensor(TensorValue &output_tensor, GeTensorDesc &ge_tensor_desc,
                                              const int64_t output_size, vector<ge::Tensor> &outputs) const {
  GELOGD("Start to build device tensor with details [%s].", output_tensor.DebugString().c_str());
  const MemStorageType mem_type = output_tensor.GetMemType();
  GELOGD("Mem type is %d", static_cast<uint32_t>(mem_type));
  const auto deleter = [this, mem_type](uint8_t *const device_data) {
    if (device_data != nullptr) {
      GELOGD("Free device addr is %p", device_data);
      const auto allocator = NpuMemoryAllocator::GetAllocator(device_id_, stream_);
      if (allocator != nullptr) {
        allocator->Deallocate(device_data, mem_type);
      }
    }
  };
  ge_tensor_desc.SetPlacement(kPlacementDevice);
  GeTensor ge_tensor(ge_tensor_desc);
  auto tensor = TensorAdapter::AsTensor(ge_tensor);
  GE_CHK_STATUS_RET(
      tensor.SetData(PtrToPtr<void, uint8_t>(output_tensor.Release()), static_cast<size_t>(output_size), deleter));
  outputs.emplace_back(std::move(tensor));
  return SUCCESS;
}
}  // namespace hybrid
}

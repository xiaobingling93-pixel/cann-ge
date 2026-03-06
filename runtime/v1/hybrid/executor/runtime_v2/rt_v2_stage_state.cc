/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rt_v2_stage_state.h"

#include <utility>

#include "common/checker.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_type_utils.h"
#include "common/compile_profiling/ge_call_wrapper.h"

namespace gert {
std::unique_ptr<StageState> StageState::Create(const ge::GeRootModelPtr &model, RtSession *session) {
  GE_ASSERT_NOTNULL(model);
  GE_ASSERT_NOTNULL(model->GetRootGraph());
  auto executor = gert::RtV2SimpleExecutor::Create(model, session);
  GE_ASSERT_NOTNULL(executor, "Failed create executor for model %s", model->GetRootGraph()->GetName().c_str());
  return std::unique_ptr<StageState>(new (std::nothrow)
                                         StageState(model->GetRootGraph()->GetName(), std::move(executor)));
}

StageState::StageState(std::string id, std::unique_ptr<gert::RtV2ExecutorInterface> &&executor)
    : id_(std::move(id)), executor_(std::move(executor)) {}

ge::Status StageState::Load(const gert::ModelExecuteArg &arg, const gert::ModelLoadArg &load_arg, bool daemon) {
  GELOGI("Start load stage %s, daemon %s", id_.c_str(), (daemon ? "true" : "false"));
  static_cast<void>(arg);
  GE_ASSERT_NOTNULL(executor_);
  auto model_input_desc = executor_->GetAllInputsDesc(num_inputs_);
  GE_ASSERT_NOTNULL(model_input_desc);
  executor_inputs_.reserve(num_inputs_);
  auto model_output_desc = executor_->GetAllOutputsDesc(num_outputs_);
  GE_ASSERT_NOTNULL(model_output_desc);
  executor_outputs_.reserve(num_outputs_);

  GELOGI("Start load executor of stage %s", id_.c_str());
  ModelDescToTensorSpec(model_input_desc, num_inputs_, executor_inputs_holder_);
  for (size_t i = 0U; i < num_inputs_; i++) {
    GELOGI("  Input %zu %s", i, ge::hybrid::DebugString(executor_inputs_holder_[i], false).c_str());
    executor_inputs_.emplace_back(&executor_inputs_holder_[i]);
  }
  ModelDescToTensorSpec(model_output_desc, num_outputs_, executor_outputs_holder_);
  for (size_t i = 0U; i < num_outputs_; i++) {
    GELOGI("  Output %zu %s", i, ge::hybrid::DebugString(executor_outputs_holder_[i], false).c_str());
    executor_outputs_.emplace_back(&executor_outputs_holder_[i]);
  }
  ModelDescToTensorSpec(model_output_desc, num_outputs_, stage_outputs_);

  if (daemon) {
    GE_ASSERT_RT_OK(rtStreamCreate(&execute_arg_.stream, 0));  // Daemon load on stage-owned stream
  } else {
    execute_arg_ = arg;  // Non daemon stage will use model stream
  }
  GE_ASSERT_SUCCESS(executor_->Load(execute_arg_, load_arg));
  return ge::SUCCESS;
}

ge::Status StageState::LoadAsDaemon(const gert::ModelExecuteArg &arg, const gert::ModelLoadArg &load_arg,
                                    CtxInitializer ctx_initializer, std::shared_ptr<StageNotification> &notification) {
  notification_ = ge::MakeShared<StageNotification>();
  GE_ASSERT_NOTNULL(notification_, "Failed create notification for stage %s", id_.c_str());
  notification = notification_;
  GE_ASSERT_SUCCESS(Load(arg, load_arg, true));  // Daemon load will create new stream
  worker_ = std::thread([this, ctx_initializer]() {
    SET_THREAD_NAME(pthread_self(), "ge_exe_ldstage");
    ctx_initializer();
    while (true) {
      StageTask task = notification_->GetTask();
      if (task.signal == StageTask::Signal::RUN) {
        GELOGI("Stage %s got an run task with expect steps %zu", id_.c_str(), task.num_steps);
        (void)Run(task);
        notification_->Done();
      } else {
        GELOGI("Stage %s worker thread exiting", id_.c_str());
        return;
      }
    }
  });
  return ge::SUCCESS;
}

ge::Status StageState::Run(const StageTask &args) {
  auto status = RunInternal(args);
  error_status_ = (status != ge::SUCCESS);
  return status;
}

ge::Status StageState::RunInternal(const StageTask &args) {
  GE_ASSERT_SUCCESS(AssembleModelInputs(args.inputs, args.num_input));
  do {
    GE_ASSERT_SUCCESS(ConsumeInputs());
    GE_ASSERT_SUCCESS(RunTask());
    running_step_++;
    GE_ASSERT_SUCCESS(ProduceOutputs());
    done_steps_++;
  } while (done_steps_ < args.num_steps);

  for (auto &item : fetch_2_stage_outputs_) {
    GE_ASSERT(item.first < args.num_output);
    *args.outputs[item.first] = std::move(*executor_outputs_[item.second]);
    GELOGI("Stage %s step %zu output %zu assemble to fetch output %zu %s", id_.c_str(), done_steps_.load(), item.second,
           item.first, ge::hybrid::DebugString(*args.outputs[item.first], false).c_str());
  }
  FreeInterimOutputs();
  return ge::SUCCESS;
}

void StageState::Reset() {
  running_step_ = 0U;
  done_steps_ = 0U;
  error_status_ = false;
}

ge::Status StageState::Stop() {
  if (worker_.joinable()) {
    GELOGI("Start stop worker thread for stage %s", id_.c_str());
    StageTask args;
    args.signal = StageTask::Signal::STOP;
    notification_->Notify(args);
    worker_.join();
    if (execute_arg_.stream != nullptr) {
      (void)rtStreamDestroy(execute_arg_.stream);
      execute_arg_.stream = nullptr;
    }
    GELOGI("Worker thread for stage %s exited", id_.c_str());
  }
  GELOGI("Unload executor for stage %s", id_.c_str());
  return executor_->Unload();
}

StageState::~StageState() {
  Stop();
}

bool StageState::IsErrorStatus() const {
  return error_status_;
}

const std::string &StageState::Id() const {
  return id_;
}

ge::Status StageState::MappingStageIO(const std::map<ge::NodePtr, StageState *> &named_stages) {
  for (auto &node_and_stage : named_stages) {
    auto &stage_node = node_and_stage.first;
    auto &stage = node_and_stage.second;
    std::map<ge::NodePtr, std::map<size_t, std::set<size_t>>> node_input_infos;
    size_t input_index = 0U;
    for (auto &node_and_anchor : stage_node->GetInDataNodesAndAnchors()) {
      auto &node = node_and_anchor.first;
      int32_t output_index = node_and_anchor.second->GetIdx();
      GE_ASSERT(output_index >= 0);
      if (ge::OpTypeUtils::IsDataNode(node->GetType())) {
        size_t index = 0U;
        GE_ASSERT(ge::AttrUtils::GetInt(node->GetOpDesc(), ge::ATTR_NAME_INDEX, index));
        GELOGI("Mapping stage %s input %zu to feed %zu", stage_node->GetName().c_str(), input_index, index);
        (void)stage->feed_2_stage_inputs_[index].insert(input_index++);
      } else {
        GELOGI("Mapping stage %s input %zu to stage %s output %u", stage_node->GetName().c_str(), input_index,
               node->GetName().c_str(), output_index);
        (void)node_input_infos[node][static_cast<size_t>(output_index)].insert(input_index++);
      }
    }
    for (auto &node : stage_node->GetInControlNodes()) {
      GE_ASSERT_NOTNULL(node);
      GELOGI("Stage %s touch control input node %s", stage_node->GetName().c_str(), node->GetName().c_str());
      (void)node_input_infos[node];  // Touch control stage
    }

    for (auto &node_input_info : node_input_infos) {
      auto iter = named_stages.find(node_input_info.first);
      GE_ASSERT(iter != named_stages.end(), "Stage for node %s not found", stage_node->GetName().c_str());
      stage->stage_input_infos_[iter->second] = std::move(node_input_info.second);
      (void)iter->second->output_stages_.insert(stage);
    }

    for (auto &node_and_anchor : stage_node->GetOutDataNodesAndAnchors()) {
      auto &node = node_and_anchor.first;
      if (node->GetType() == ge::NETOUTPUT) {
        auto &anchor = node_and_anchor.second;  // Stage node peer node input anchor
        GE_ASSERT_NOTNULL(anchor->GetPeerOutAnchor());
        auto output_index = anchor->GetPeerOutAnchor()->GetIdx();
        GE_ASSERT(output_index >= 0 && static_cast<size_t>(output_index) < stage_node->GetAllOutDataAnchorsSize());
        GE_ASSERT(anchor->GetIdx() >= 0);
        GELOGI("Mapping stage %s output %u to fetch %u", stage_node->GetName().c_str(), output_index, anchor->GetIdx());
        stage->fetch_2_stage_outputs_[static_cast<size_t>(anchor->GetIdx())] = output_index;
      }
    }
  }

  return ge::SUCCESS;
}

ge::Status StageState::GetConsumedModelInputDesc(std::map<size_t, ModelIoDesc> &descs) const {
  GE_ASSERT_NOTNULL(executor_);
  size_t num_input = 0U;
  auto stage_input_descs = executor_->GetAllInputsDesc(num_input);
  GE_ASSERT_NOTNULL(stage_input_descs);

  for (auto &item : feed_2_stage_inputs_) {
    auto &model_feed_index = item.first;
    auto &stage_input_index = item.second;
    GE_ASSERT(std::all_of(stage_input_index.begin(), stage_input_index.end(),
                          [&num_input](const size_t &i) { return i < num_input; }));
    descs[model_feed_index] = stage_input_descs[*stage_input_index.begin()];
  }

  return ge::SUCCESS;
}

ge::Status StageState::GetProducedModelOutputDesc(std::map<size_t, ModelIoDesc> &descs) const {
  GE_ASSERT_NOTNULL(executor_);
  size_t num_output = 0U;
  auto stage_output_descs = executor_->GetAllOutputsDesc(num_output);
  GE_ASSERT_NOTNULL(stage_output_descs);

  for (auto &item : fetch_2_stage_outputs_) {
    auto &model_fetch_index = item.first;
    auto &stage_output_index = item.second;
    GE_ASSERT(stage_output_index < num_output);
    descs[model_fetch_index] = stage_output_descs[stage_output_index];
  }

  return ge::SUCCESS;
}

namespace {
// Pipeline request stage output tensor owned memory
bool IsTensorOwnedMemory(gert::Tensor &tensor) {
  auto &tensor_data = tensor.GetTensorData();
  return !tensor.GetTensorData().IsSharedWith(
      TensorData(tensor_data.GetAddr(), nullptr, tensor_data.GetSize(), tensor_data.GetPlacement()));
}
}  // namespace

ge::Status StageState::ConsumeInputs() {
  for (auto &input_info : stage_input_infos_) {
    auto &input_stage = input_info.first;
    GE_ASSERT_NOTNULL(input_stage);
    GELOGI("Stage %s consume stage %s outputs, step %zu", id_.c_str(), input_stage->id_.c_str(), done_steps_.load());
    while ((input_stage->done_steps_ <= done_steps_) && (!input_stage->error_status_)) {
    }
    GE_ASSERT(!input_stage->error_status_, "Stage %s input stage %s status error", id_.c_str(),
              input_stage->id_.c_str());
    for (auto &item : input_info.second) {
      for (auto &index : item.second) {
        GE_ASSERT_SUCCESS(input_stage->GetOutput(item.first, *executor_inputs_[index]));
        GELOGI("Stage %s step %zu get input %zu from stage %s output %zu %s", id_.c_str(), done_steps_.load(), index,
               input_stage->id_.c_str(), item.first, ge::hybrid::DebugString(*executor_inputs_[index], true).c_str());
      }
    }
  }

  GELOGI("Stage %s inputs consumed, step %zu", id_.c_str(), done_steps_.load());
  return ge::SUCCESS;
}

void StageState::FreeInterimOutputs() {
  const static gert::TensorData kClearedTensorData;
  for (auto &output : executor_outputs_) {
    // MutableTensorData().Free() is not enough as it maybe not owned the output memory
    GELOGI("Clear model output holder with block %p", output->GetAddr());
    output->MutableTensorData().ShareFrom(kClearedTensorData);
  }
}

ge::Status StageState::RunTask() {
  GELOGI("Stage %s run task, step %zu", id_.c_str(), done_steps_.load());
  FreeInterimOutputs();
  GE_ASSERT_SUCCESS(executor_->Execute(execute_arg_, executor_inputs_.data(), executor_inputs_.size(),
                                       executor_outputs_.data(), executor_outputs_.size()));
  GE_ASSERT_RT_OK(rtStreamSynchronize(execute_arg_.stream), "Stage %s sync stream failed", id_.c_str());
  return ge::SUCCESS;
}

ge::Status StageState::ProduceOutputs() {
  for (auto &output_stage : output_stages_) {
    GE_ASSERT_NOTNULL(output_stage);
    GELOGI("Stage %s wait stage %s consume outputs, step %zu", id_.c_str(), output_stage->id_.c_str(),
           done_steps_.load());
    while (output_stage->running_step_ < done_steps_ && (!output_stage->error_status_)) {
    }
    GE_ASSERT(!output_stage->error_status_, "Stage %s output stage %s status error", id_.c_str(),
              output_stage->id_.c_str());
  }
  GELOGI("Stage %s produce outputs, step %zu", id_.c_str(), done_steps_.load());
  for (size_t i = 0U; i < num_outputs_; i++) {
    GE_ASSERT(IsTensorOwnedMemory(*executor_outputs_[i]), "Stage %s output %zu not owned memory", id_.c_str(), i);
    auto &model_output = *executor_outputs_[i];
    auto &stage_output = stage_outputs_[i];
    stage_output.SetPlacement(model_output.GetPlacement());
    stage_output.MutableTensorData().ShareFrom(model_output.MutableTensorData());
    stage_output.MutableOriginShape() = model_output.GetOriginShape();
    stage_output.MutableStorageShape() = model_output.GetStorageShape();
    GELOGI("Stage %s step %zu output %zu %s", id_.c_str(), done_steps_.load(), i,
           ge::hybrid::DebugString(stage_output, true).c_str());
  }
  return ge::SUCCESS;
}

ge::Status StageState::GetOutput(size_t index, gert::Tensor &tensor) {
  GE_ASSERT(index < stage_outputs_.size());
  auto &output_tensor = stage_outputs_[index];
  tensor.MutableStorageShape() = output_tensor.GetStorageShape();
  tensor.MutableOriginShape() = output_tensor.GetOriginShape();
  tensor.SetPlacement(output_tensor.GetPlacement());
  tensor.MutableTensorData().SetAddr(output_tensor.GetAddr(), nullptr);
  tensor.MutableTensorData().SetSize(output_tensor.GetSize());
  tensor.MutableTensorData().SetPlacement(output_tensor.GetPlacement());
  return ge::SUCCESS;
}

void StageState::ModelDescToTensorSpec(const gert::ModelIoDesc *desc, size_t num, std::vector<gert::Tensor> &tensors) {
  tensors.resize(num);
  for (size_t i = 0U; i < num; i++) {
    auto &holder = tensors[i];
    holder.SetData(gert::TensorData(nullptr));
    holder.SetDataType(static_cast<ge::DataType>(desc[i].GetDataType()));
    holder.SetOriginFormat(desc[i].GetOriginFormat());
    holder.SetStorageFormat(desc[i].GetStorageFormat());
  }
}

ge::Status StageState::AssembleModelInputs(gert::Tensor **inputs, size_t input_num) {
  for (auto &item : feed_2_stage_inputs_) {
    GE_ASSERT(item.first < input_num);
    for (auto &index : item.second) {
      executor_inputs_[index] = inputs[item.first];
      GELOGI("Stage %s step %zu get input %zu from feed input %zu %s", id_.c_str(), done_steps_.load(), index,
             item.first, ge::hybrid::DebugString(*executor_inputs_[index], true).c_str());
    }
  }
  return ge::SUCCESS;
}
}  // namespace gert

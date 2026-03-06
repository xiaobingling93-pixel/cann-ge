/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "jit_executor.h"

#include <checker.h>
#include <runtime/stream.h>
#include "framework/runtime/gert_api.h"
#include "common/memory/tensor_trans_utils.h"
#include "graph/manager/util/rt_context_util.h"
#include "graph/ge_context.h"
#include "common/model/external_allocator_manager.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/utils/op_type_utils.h"

#define JIT_ASSERT(exp, tsk, ...)              \
  do {                                          \
    bool tmp_ret = (exp);                       \
    if (!tmp_ret) {                             \
      std::vector<gert::Tensor> error_outputs;        \
      if ((tsk.callback) != nullptr) {            \
        tsk.callback(ge::FAILED, error_outputs); \
      }                                         \
      GE_ASSERT_TRUE(tmp_ret, __VA_ARGS__);     \
      return ::ErrorResult();                   \
    }                                           \
  } while (false)

#define JIT_ASSERT_NOTNULL(v, t, ...) JIT_ASSERT(((v) != nullptr), (t), __VA_ARGS__)
#define JIT_ASSERT_SUCCESS(v, t, ...) JIT_ASSERT(((v) == ge::SUCCESS), (t),  __VA_ARGS__)
#define JIT_ASSERT_RT_OK(v, t, ...) JIT_ASSERT(((v) == 0), (t),  __VA_ARGS__)

namespace ge {
namespace {
const uint64_t kStopOnFailure = 1U;

void PrepareOutputs(const ExecutionPoint &ep, std::vector<gert::Tensor> &outputs,
                    std::vector<GeTensor> &output_ge_tensors) {
  outputs.resize(ep.GetEpOutNum());
  output_ge_tensors.resize(ep.GetEpOutNum());
  for (auto &tensor : outputs) {
    tensor.SetOriginFormat(FORMAT_RESERVED);
    tensor.SetData(gert::TensorData());
  }
}

bool IsEnableBatchCpy(const std::vector<gert::Tensor> &inputs) {
  std::string input_batch_cpy_str;
  (void)GetThreadLocalContext().GetOption(configure_option::INPUT_BATCH_CPY, input_batch_cpy_str);
  GELOGI("Get input_batch_cpy_str=%s, size of inputs=%zu", input_batch_cpy_str.c_str(), inputs.size());
  return (!input_batch_cpy_str.empty() && input_batch_cpy_str == "1" && inputs.size() > 1);
}

// todo if host exec option, remember to handle
Status CopyHostInputsToDevice(UserGraphExecution &execution_task, Allocator *const allocator,
  std::vector<gert::Tensor> &device_gert_tensors) {
  const auto *external_rt_inputs = execution_task.external_rt_inputs;
  auto &inputs_memblocks = execution_task.inputs_memblocks;
  bool enable_input_batch_cpy = IsEnableBatchCpy(*external_rt_inputs);
  GE_ASSERT_SUCCESS(TensorTransUtils::TransHostGertTensorsToDevice(allocator, *external_rt_inputs,
    device_gert_tensors, inputs_memblocks, enable_input_batch_cpy));
  return SUCCESS;
}

Status FreeInputsAllocByJit(std::vector<MemBlock *> &input_blocks) {
  for (auto &mem_block: input_blocks) {
    GE_ASSERT_NOTNULL(mem_block);
    mem_block->Free();
  }
  return SUCCESS;
}

Status CopyHostInputToDeviceAfterSlice(std::vector<gert::Tensor> *inputs, std::vector<MemBlock *> &input_mem_block,
                                       std::shared_ptr<ge::Allocator> device_allocator) {
  MemBlock *mem_block_to_keep = nullptr;
  for (size_t i = 0U; i < inputs->size(); i++) {
    if ((*inputs)[i].GetPlacement() == gert::TensorPlacement::kOnHost) {
      GE_ASSERT_SUCCESS(TensorTransUtils::HostTensorToDeviceGertTensor(
          device_allocator.get(), (*inputs)[i].GetAddr(), (*inputs)[i].GetSize(), (*inputs)[i], mem_block_to_keep));
      GE_ASSERT_NOTNULL(mem_block_to_keep);
      input_mem_block.emplace_back(mem_block_to_keep);
    }
  }
  return SUCCESS;
}

Status GetAllCondInputData(const ComputeGraphPtr &graph, std::set<size_t> &data_idx) {
  GE_ASSERT_NOTNULL(graph);
  for (const auto &node : graph->GetAllNodes()) {
    auto cond_input = SymbolicInferUtil::GetCondInput(node);
    if (cond_input == nullptr) {
      continue;
    }
    if (!OpTypeUtils::IsDataNode(cond_input->GetType())) {
      continue;
    }
    int32_t data_index = -1;
    GE_ASSERT_TRUE(AttrUtils::GetInt(cond_input->GetOpDesc(), "index", data_index),
      "get data node %s index failed", cond_input->GetNamePtr());
    data_idx.insert(static_cast<size_t>(data_index));
  }
  return SUCCESS;
}

Status BuildCompileInputs(const std::vector<gert::Tensor> &ori_inputs, const ComputeGraphPtr &graph,
  std::vector<gert::Tensor> &compile_inputs) {
  std::set<size_t> need_host_data_idx;
  GE_ASSERT_SUCCESS(GetAllCondInputData(graph, need_host_data_idx));

  compile_inputs = TensorTransUtils::ShareFromGertTenosrs(ori_inputs);
  for (size_t data_idx : need_host_data_idx) {
    GELOGD("input[%u] need copy data to host.", data_idx);
    GE_ASSERT_TRUE(data_idx < compile_inputs.size());
    gert::Tensor host_tensor;
    GE_ASSERT_SUCCESS(TensorTransUtils::TransGertTensorToHost(compile_inputs[data_idx], host_tensor));
    compile_inputs[data_idx] = std::move(host_tensor);
  }
  return SUCCESS;
}
}  // namespace
JitExecutor::JitExecutor(GraphManager &graph_manager, UserGraphExecutionQueue &task_queue, ExecutionOrder &order,
                         CompileContext &compile_context, CompiledModelCache &cmc, std::mutex &mutex)
    : graph_manager_(graph_manager),
      task_queue_(task_queue),
      order_(order),
      compile_context_(compile_context),
      cmc_(cmc),
      mutex_(mutex) {}

std::unique_ptr<JitExecutor> JitExecutor::Create(GraphManager &graph_manager, UserGraphExecutionQueue &task_queue,
                                                 ExecutionOrder &order, CompileContext &compile_context,
                                                 CompiledModelCache &cmc, std::mutex &mutex) {
  auto jit = std::unique_ptr<JitExecutor>(new JitExecutor(graph_manager, task_queue, order, compile_context, cmc, mutex));
  GE_ASSERT_NOTNULL(jit);

  // add rt context before create jix executor
  jit.get()->device_id_ = static_cast<int32_t>(GetContext().DeviceId());
  GE_ASSERT_RT_OK(rtSetDevice(jit.get()->device_id_));
  GELOGI("Set device, device id:%u.", GetContext().DeviceId());
  GE_ASSERT_RT_OK(rtStreamCreate(&(jit.get()->stream_), 0));
  GE_ASSERT_RT_OK(rtStreamSetMode(jit.get()->stream_, kStopOnFailure));
  // prepare allocator
  auto device_allocator = gert::AllocatorFactory::Create("usergraph", gert::kOnDeviceHbm);
  GE_ASSERT_NOTNULL(device_allocator);
  jit.get()->device_allocator_ = std::move(device_allocator);
  GE_ASSERT_SUCCESS(graph_manager.RegisterExternalAllocator(jit.get()->stream_, jit.get()->device_allocator_));
  // 对于Execute接口，子图间的output是jit内部给的，静态图场景且没有外置allocator时需要手动申请内存，上面的device_allocator是针对tf场景不考虑用户会外置allocator
  auto jit_allocator = gert::AllocatorFactory::Create("usergraph", gert::kOnDeviceHbm);
  GE_ASSERT_NOTNULL(jit_allocator);
  jit.get()->external_allocator_ = std::move(jit_allocator);
  return jit;
}

Status JitExecutor::Finalize() {
  // remove每张子图的时候需要降序进行remove，因为前一张图中的内存会给下一张内存复用，例如输出tensor，如果前一张先释放，后一张释放的时候会出现heap-use-after-free
  // 通过给jit挂外置allocator的方法必须在load graph之前，因为LoadGraph接口中会默认create一个allocator，而在execute阶段会优先使用默认create的allocator
  // 在load graph之前外置allocator会导致load过程中去申请const、feature等内存，这就强制要求这个外置allocator的生命周期大于整个GE的生命周期，否则会在remove的时候释放const内存失败。所以不在JIT中外置allocator
  auto sorted_geps_to_inner_graph_id = SortMapByValue(geps_to_inner_ge_graph_id_, false);
  for (const auto &gep_2_id : sorted_geps_to_inner_graph_id) {
    GELOGI("[Jit]RemoveGraph %u", gep_2_id.second);
    GE_ASSERT_SUCCESS(graph_manager_.RemoveGraph(gep_2_id.second));
  }
  geps_to_inner_ge_graph_id_.clear();
  compiled_ge_graph_id_.clear();
  GE_ASSERT_RT_OK(rtSetDevice(device_id_));
  GE_ASSERT_SUCCESS(graph_manager_.UnregisterExternalAllocator(stream_));
  device_allocator_ = nullptr;
  external_allocator_ = nullptr;
  GE_ASSERT_RT_OK(rtStreamDestroy(stream_));
  GE_ASSERT_RT_OK(rtDeviceReset(device_id_));
  return SUCCESS;
}

Status JitExecutor::CompileGraph(UserGraphExecution &task, uint64_t session_id) {
  ExecutionPoint *ep;
  GE_ASSERT_RT_OK(rtSetDevice(device_id_));
  std::vector<GeTensor> ge_tensors;
  GE_ASSERT_SUCCESS(TensorTransUtils::GertTensors2GeTensors(*task.external_rt_inputs, ge_tensors));
  GE_ASSERT_SUCCESS(order_.FirstPoint(ge_tensors, ep));
  GELOGD("Get EP[%ld] of USER_GRAPH[%u] for CompileGraph", ep->GetId(), task.user_graph_id);

  auto gep = ep->FindOrCreateGuarded(*task.external_rt_inputs);
  GE_ASSERT_NOTNULL(gep);
  GELOGD("Get GEP[compiled_graph_id:%u] [compiled? %d] of EP[%ld] USER_GRAPH[%u].", gep->GetCompiledGraphId(),
         gep->Compiled(), ep->GetId(), task.user_graph_id);
  std::vector<Tensor> tensors;
  for (auto ge_tensor : ge_tensors) {
    tensors.emplace_back(TensorAdapter::AsTensor(ge_tensor));
  }
  GE_ASSERT_SUCCESS(Compile(tensors, gep, session_id));
  if (!ep->IsLast()) {
    GELOGD("Get EP[%ld] of USER_GRAPH[%u] is not last, need compile whole graph", ep->GetId(), task.user_graph_id);
    return ge::GE_GRAPH_NOT_BUILT;
  }
  return SUCCESS;
}

Status JitExecutor::LoadGraph(UserGraphExecution &task) {
  ExecutionPoint *ep;
  GE_ASSERT_RT_OK(rtSetDevice(device_id_));
  rtStream_t const stream = (task.stream == nullptr) ? stream_ : task.stream;
  std::vector<GeTensor> ge_tensors;
  GE_ASSERT_SUCCESS(TensorTransUtils::GertTensors2GeTensors(*task.external_rt_inputs, ge_tensors));
  GE_ASSERT_SUCCESS(order_.FirstPoint(ge_tensors, ep));
  GELOGD("Get EP[%ld] of USER_GRAPH[%u] for LoadGraph", ep->GetId(), task.user_graph_id);

  auto gep = ep->FindGuarded(*task.external_rt_inputs);
  if (gep == nullptr || !gep->Compiled()) {
    GELOGE(ge::FAILED, "Guarde is not exist or Compiled EP[%ld], USER_GRAPH[%u]", ep->GetId(), task.user_graph_id);
    return FAILED;
  }
  GELOGD("Get GEP[compiled_graph_id:%u] [compiled? %d] of EP[%ld] USER_GRAPH[%u].", gep->GetCompiledGraphId(),
         gep->Compiled(), ep->GetId(), task.user_graph_id);
  auto iter = geps_to_inner_ge_graph_id_.find(gep);
  GE_ASSERT_TRUE(iter != geps_to_inner_ge_graph_id_.end());
  
  GE_ASSERT_SUCCESS(compile_context_.Load(iter->second, task.load_options, stream));
  return SUCCESS;
}

Status JitExecutor::RunWithCallback(UserGraphExecution &&task) {
  ExecutionPoint *ep;
  GE_ASSERT_RT_OK(rtSetDevice(device_id_));
  JIT_ASSERT_NOTNULL(task.external_rt_inputs, task);

  std::vector<GeTensor> ge_tensors;
  ep = order_.GetFirstPoint();
  if (ep == nullptr) {
    JIT_ASSERT_SUCCESS(TensorTransUtils::GertTensors2GeTensors(*task.external_rt_inputs, ge_tensors), task);
    JIT_ASSERT_SUCCESS(order_.FirstPoint(ge_tensors, ep), task);  
  }
  GELOGD("Get EP[%ld] of USER_GRAPH[%u]", ep->GetId(), task.user_graph_id);

  std::vector<gert::Tensor> tensors0;
  GE_MAKE_GUARD(free_input_mem, [&task]() { (void)FreeInputsAllocByJit(task.inputs_memblocks); });
  JIT_ASSERT_SUCCESS(CopyHostInputsToDevice(task, device_allocator_.get(), tensors0), task);

  std::vector<gert::Tensor> tensors1;
  auto inputs = &tensors0;
  auto outputs = &tensors1;

  std::vector<MemBlock *> input_mem_block;
  auto free_mem_block_callback = [&input_mem_block] () {
    for (auto &mem_block : input_mem_block) {
      mem_block->Free();
    }
    input_mem_block.clear();
  };
  GE_MAKE_GUARD(free_mem, free_mem_block_callback);

  while (ep != nullptr) {
    PrepareOutputs(*ep, *outputs, ge_tensors);
    GE_ASSERT_SUCCESS(ProcessAndExecuteGraphAsync(task, stream_, *inputs, *outputs, ep));
    for (size_t i = 0U; i < ge_tensors.size(); ++i) {
      JIT_ASSERT_SUCCESS(TensorTransUtils::TransRtTensorToGeTensor((*outputs)[i], ge_tensors[i]), task);
    }
    JIT_ASSERT_SUCCESS(order_.NextPoint(*ep, ge_tensors, ep), task);
    free_mem_block_callback();
    if (ep != nullptr) {
      std::swap(inputs, outputs);
      GE_ASSERT_SUCCESS(CopyHostInputToDeviceAfterSlice(inputs, input_mem_block, device_allocator_));
    }
  }
  JIT_ASSERT_RT_OK(rtStreamSynchronize(stream_), task);
  GE_CHECK_NOTNULL(task.callback);
  std::vector<gert::Tensor> host_tensors;
  GE_ASSERT_SUCCESS(TensorTransUtils::TransGertTensorsToHost(*outputs, host_tensors));
  task.callback(SUCCESS, host_tensors);
  return SUCCESS;
}

Status JitExecutor::Execute(UserGraphExecution &&task) {
  const auto ret = TryExecuteWithoutProcess(task);
  if (ret != ge::UNSUPPORTED) {
    return ret;
  }
  ExecutionPoint *ep;
  GE_ASSERT_RT_OK(rtSetDevice(device_id_));
  rtStream_t const stream = (task.stream == nullptr) ? stream_ : task.stream;
  const bool has_allocator = (ExternalAllocatorManager::GetExternalAllocator(stream) != nullptr);

  std::vector<GeTensor> ge_tensors;
  std::vector<Tensor> tensors;
  GE_ASSERT_SUCCESS(TensorTransUtils::TransRtTensorToTensor(*task.external_rt_inputs, tensors, true));
  for (auto &input_tensor : tensors) {
    ge_tensors.emplace_back(TensorAdapter::AsGeTensor(input_tensor));
  }
  GE_ASSERT_SUCCESS(order_.FirstPoint(ge_tensors, ep));
  GELOGD("Get EP[%ld] of USER_GRAPH[%u]", ep->GetId(), task.user_graph_id);

  std::vector<gert::Tensor> tensors0;
  auto outputs = &tensors0;
  PrepareOutputs(*ep, *outputs, ge_tensors);
  outputs = (ep->IsLast()) ? task.rt_outputs : outputs;
  // user没有外置allcator且不是最后一张slice graph需要尝试进行output内存申请。因为子图间的output是jit内部给的，静态图场景且没有外置allocator时需要手动申请内存
  const bool need_malloc_outputs = (!has_allocator && !ep->IsLast());
  GE_ASSERT_SUCCESS(ExecuteFirstPoint(task, stream, *outputs, ge_tensors, ep, need_malloc_outputs));
  task.load_options.clear(); // 当前load option只对第一张图有效
  if (ep == nullptr) {
    return SUCCESS;
  }
  auto inputs = outputs;
  std::vector<gert::Tensor> tensors1;
  outputs = &tensors1;

  while (ep != nullptr) {
    PrepareOutputs(*ep, *outputs, ge_tensors);
    outputs = (ep->IsLast()) ? task.rt_outputs : outputs;
    const bool need_malloc_outputs_local = (!has_allocator && !ep->IsLast());
    GE_ASSERT_SUCCESS(ProcessAndExecuteGraphAsync(task, stream, *inputs, *outputs, ep, need_malloc_outputs_local));
    for (size_t i = 0U; i < ge_tensors.size(); ++i) {
      GE_ASSERT_SUCCESS(TensorTransUtils::TransRtTensorToGeTensor((*outputs)[i], ge_tensors[i]));
    }
    GE_ASSERT_SUCCESS(order_.NextPoint(*ep, ge_tensors, ep));
    if (ep != nullptr) {
      std::swap(inputs, outputs);
    }
  }
  return SUCCESS;
}

Status JitExecutor::ExecuteFirstPoint(UserGraphExecution &task, rtStream_t const stream,
                                      std::vector<gert::Tensor> &outputs, std::vector<GeTensor> &ge_tensors,
                                      ExecutionPoint *&ep, bool need_malloc_output) {
  auto external_inputs = task.external_rt_inputs;
  GE_ASSERT_SUCCESS(ProcessAndExecuteGraphAsync(task, stream, *external_inputs, outputs, ep, need_malloc_output));
  for (size_t i = 0U; i < ge_tensors.size(); ++i) {
    GE_ASSERT_SUCCESS(TensorTransUtils::TransRtTensorToGeTensor((outputs)[i], ge_tensors[i]));
  }
  GE_ASSERT_SUCCESS(order_.NextPoint(*ep, ge_tensors, ep));
  return SUCCESS;
}

Status JitExecutor::MallocOutputsForStatic(uint32_t guarded_ep_instance_id, const GuardedExecutionPoint *gep,
                                           std::vector<gert::Tensor> &outputs) {
  CompiledGraphSummaryPtr summary{nullptr};
  GE_ASSERT_SUCCESS(graph_manager_.GetCompiledGraphSummary(guarded_ep_instance_id, summary));
  GraphNodePtr graph_node = make_shared<GraphNode>(guarded_ep_instance_id);
  graph_node->SetComputeGraph(gep->GetGraph());
  // 只有静态的slice graph需要手动申请output内存，动态的ge内部会申请
  if (summary->IsStatic()) {
    GE_ASSERT_SUCCESS(MallocOutputsMemory(guarded_ep_instance_id, graph_node, external_allocator_, outputs));
  }
  return SUCCESS;
}

Status JitExecutor::ProcessAndExecuteGraphAsync(UserGraphExecution &task, const rtStream_t stream,
                                                const std::vector<gert::Tensor> &inputs,
                                                std::vector<gert::Tensor> &outputs, ExecutionPoint *ep,
                                                bool need_malloc_output) {
  GuardedExecutionPoint *gep = nullptr;
  std::vector<gert::Tensor> compile_inputs;
  GE_ASSERT_SUCCESS(BuildCompileInputs(inputs, ep->GetSlicedGraph(), compile_inputs));
  uint32_t guarded_ep_instance_id;
  {
    std::lock_guard<std::mutex> locker(mutex_);
    // 需要value
    gep = ep->FindOrCreateGuarded(compile_inputs);
    JIT_ASSERT_NOTNULL(gep, task);
    GELOGD("Get GEP[compiled_graph_id:%u] [compiled? %d] of EP[%ld] USER_GRAPH[%u], session_id:%llu.",
      gep->GetCompiledGraphId(), gep->Compiled(), ep->GetId(), task.user_graph_id, task.session_id);

    // 需要value
    JIT_ASSERT_SUCCESS(CompileAndLoad(compile_inputs, gep, guarded_ep_instance_id, stream, task.load_options, task.session_id),
      task);
  }
  JIT_ASSERT_NOTNULL(gep, task);
  GELOGD("ExecuteGraphWithStreamAsync GEP[ins_id:%u] of EP[%ld] USER_GRAPH[%u].", guarded_ep_instance_id,
         gep->GetOwnerEp()->GetId(), task.user_graph_id);
  GE_ASSERT_RT_OK(rtSetDevice(device_id_));

  // 非最后一张slice graph以外的图需要尝试进行output内存的申请，因为子图间的output是jit内部给的，静态图场景且没有外置allocator时需要手动申请内存
  if (need_malloc_output) {
    GE_ASSERT_SUCCESS(MallocOutputsForStatic(guarded_ep_instance_id, gep, outputs));
  }
  JIT_ASSERT_SUCCESS(graph_manager_.ExecuteGraphWithStreamAsync(guarded_ep_instance_id, stream, inputs, outputs), task);
  return SUCCESS;
}

Status JitExecutor::TryExecuteWithoutProcess(UserGraphExecution &task) {
  auto first_ep = order_.GetFirstPoint();
  if (first_ep != nullptr && first_ep->IsLast()) {
    GELOGD("Get EP[%ld] of USER_GRAPH[%u] for LoadGraph", first_ep->GetId(), task.user_graph_id);
    auto gep = first_ep->FindGuarded(*(task.external_rt_inputs));
    if (gep == nullptr || !gep->Compiled()) {
      return ge::UNSUPPORTED;
    }
    GELOGD("Get GEP[compiled_graph_id:%u] [compiled? %d] of EP[%ld] USER_GRAPH[%u].", gep->GetCompiledGraphId(),
         gep->Compiled(), first_ep->GetId(), task.user_graph_id);
    auto iter = geps_to_inner_ge_graph_id_.find(gep);
    if (iter != geps_to_inner_ge_graph_id_.end()) {
      GELOGD("Graph id:%u No need Execute with Jit process.", iter->second);
      rtStream_t const stream = (task.stream == nullptr) ? stream_ : task.stream;
      GE_ASSERT_SUCCESS(graph_manager_.ExecuteGraphWithStreamAsync(iter->second, stream, *(task.external_rt_inputs), *(task.rt_outputs)));
      return SUCCESS;
    }
  }
  return ge::UNSUPPORTED;
}

Status JitExecutor::Compile(const std::vector<ge::Tensor> &inputs, GuardedExecutionPoint *gep, uint64_t session_id) {
  std::lock_guard<std::mutex> locker(mutex_);
  if (!gep->Compiled()) {
    auto instance_id = compile_context_.GenNewGraphId();
    GELOGI("Start to compile GEP[%u] for EP[%ld].", instance_id, gep->GetOwnerEp()->GetId());
    GE_ASSERT_TRUE(geps_to_inner_ge_graph_id_.emplace(gep, instance_id).second);

    GE_ASSERT_SUCCESS(compile_context_.Compile(instance_id, gep->GetGraph(), inputs, session_id));
    GE_ASSERT_RT_OK(rtSetDevice(device_id_));
    compiled_ge_graph_id_.emplace_back(instance_id);
    GE_ASSERT_TRUE(gep->SetCompiled(instance_id, gep->GetGraph()));
  }
  return SUCCESS;
}

Status JitExecutor::CompileAndLoad(const std::vector<gert::Tensor> &inputs, GuardedExecutionPoint *gep,
    uint32_t &instance_id, const rtStream_t stream, const std::map<AscendString, AscendString> &load_options,
    uint64_t session_id) {
  /*
   * | epm status    | instance exists | instance does not exist |
   * |---------------|-----------------|--------------------|
   * | not compiled  | ERROR           | compile + load     |
   * | compiled      | load            | fork + load        |
   * | loaded        | do nothing      | ERROR              |
   *
   *  NOTE: The "Compiled & Instance Exists" scenario represents a cache aged case,
   *        which has not been implemented yet.
   */
  // todo handle mutex here
  // compile and load need to lock mutex_
  // find instance just read mutex
  if (!gep->Compiled()) {
    instance_id = compile_context_.GenNewGraphId();
    GELOGI("Start to compile GEP[%u] for EP[%ld], session_id: %llu.", instance_id, gep->GetOwnerEp()->GetId(),
      session_id);
    GE_ASSERT_TRUE(geps_to_inner_ge_graph_id_.emplace(gep, instance_id).second);

    std::map<std::string, std::string> options;
    GE_ASSERT_SUCCESS(cmc_.CreateKeyOptionForGuardedExecutionPoint(gep, options));
    GE_ASSERT_SUCCESS(compile_context_.Compile(instance_id, gep->GetGraph(), inputs, options, session_id),
      "GEP:%u, EP:%ld, session_id:%llu", instance_id, gep->GetOwnerEp()->GetId(), session_id);
    GE_ASSERT_RT_OK(rtSetDevice(device_id_));
    // todo 编译失败的时候，需要处理死锁问题
    compiled_ge_graph_id_.emplace_back(instance_id);
    GE_ASSERT_TRUE(gep->SetCompiled(instance_id, gep->GetGraph()));
    GE_ASSERT_SUCCESS(compile_context_.Load(instance_id, load_options, stream));
  } else {
    auto iter = geps_to_inner_ge_graph_id_.find(gep);
    if (iter == geps_to_inner_ge_graph_id_.end()) {
      instance_id = compile_context_.GenNewGraphId();
      GE_ASSERT_SUCCESS(compile_context_.Fork(gep->GetCompiledGraphId(), instance_id));
      GE_ASSERT_RT_OK(rtSetDevice(device_id_));
      GE_ASSERT_SUCCESS(compile_context_.Load(instance_id, load_options, stream));
      GE_ASSERT_TRUE(geps_to_inner_ge_graph_id_.emplace(gep, instance_id).second);
      gep->SetForked(instance_id);
    } else {
      instance_id = iter->second;
    }
  }
  return SUCCESS;
}
bool JitExecutor::IsUserGraphNeedRebuild() {
  return std::any_of(compiled_ge_graph_id_.cbegin(), compiled_ge_graph_id_.cend(), [this](uint32_t graph_id) {
    const auto is_graph_need_rebuild = compile_context_.IsGraphNeedRebuild(graph_id);
    GELOGI("Graph instance id %u need rebuild : %d", graph_id, is_graph_need_rebuild);
    return is_graph_need_rebuild;
  });
}
}  // namespace ge
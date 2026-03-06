/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rt_v2_pipeline_executor.h"

#include "rt_v2_simple_executor.h"
#include "rt_v2_utils.h"
#include "rt_v2_stage_state.h"
#include "framework/common/types.h"
#include "framework/runtime/model_desc.h"
#include "ge/ge_api.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/ge_local_context.h"
#include "runtime/model_v2_executor.h"
#include "lowering/model_converter.h"

#include <thread>
#include <chrono>

namespace gert {
namespace {
ge::NodePtr GraphAddInput(ge::ComputeGraphPtr &graph, const std::string &name, size_t index,
                          const ge::GeTensorDescPtr &desc) {
  auto data_desc = ge::MakeShared<ge::OpDesc>(name, ge::DATA);
  GE_ASSERT_NOTNULL(data_desc);
  GE_ASSERT_GRAPH_SUCCESS(data_desc->AddOutputDesc("y", *desc));
  ge::AttrUtils::SetInt(data_desc, "index", static_cast<int64_t>(index));
  return graph->AddNode(data_desc);
}

ge::NodePtr GraphAddOutput(ge::ComputeGraphPtr &graph, const std::string &name,
                           const ge::OpDesc::Vistor<ge::GeTensorDescPtr> &descs) {
  auto netoutput_desc = ge::MakeShared<ge::OpDesc>(name, ge::NETOUTPUT);
  GE_ASSERT_NOTNULL(netoutput_desc);
  for (auto &desc : descs) {
    netoutput_desc->AddInputDesc(*desc);
  }
  return graph->AddNode(netoutput_desc);
}

using NamedGraph = std::pair<std::string, ge::ComputeGraphPtr>;
ge::Status GetNodeSubgraphs(const ge::NodePtr &node, const ge::ComputeGraphPtr &library,
                            std::queue<NamedGraph> &subgraphs) {
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  GE_ASSERT_NOTNULL(library);
  for (auto &gn : node->GetOpDesc()->GetSubgraphInstanceNames()) {
    subgraphs.push({gn, library->GetSubgraph(gn)});
    GE_ASSERT_NOTNULL(subgraphs.back().second, "Subgraph %s of node %s not found", gn.c_str(), node->GetName().c_str());
    GELOGI("Collect subgraph %s of node %s", subgraphs.back().second->GetName().c_str(), node->GetName().c_str());
  }
  return ge::SUCCESS;
}

ge::Status GetNodeRelatedSubgraphs(const ge::NodePtr &node, const ge::ComputeGraphPtr &library,
                                   std::vector<NamedGraph> &subgraphs) {
  std::queue<NamedGraph> graph_queue;
  GE_ASSERT_SUCCESS(GetNodeSubgraphs(node, library, graph_queue));
  std::set<std::string> seen_subgraphs;
  while (!graph_queue.empty()) {
    auto named_graph = graph_queue.front();
    graph_queue.pop();
    if (seen_subgraphs.insert(named_graph.first).second) {
      subgraphs.push_back(named_graph);
      for (auto &n : named_graph.second->GetDirectNode()) {
        GE_ASSERT_SUCCESS(GetNodeSubgraphs(n, library, graph_queue));
      }
    }
  }
  return ge::SUCCESS;
}

class StageModelRecover {
 public:
  explicit StageModelRecover(const ge::GeRootModelPtr &model)
      : model_(model), root_(model->GetRootGraph()), named_models_(model->GetSubgraphInstanceNameToModel()) {}

  void GuardSubgraphParent(const ge::ComputeGraphPtr &graph, const ge::NodePtr &parent) {
    graph_to_parents_[graph] = parent;
  }
  ~StageModelRecover() {
    model_->SetRootGraph(root_);
    model_->SetFlattenGraph(nullptr);  // Do not recover as related filed maybe changed
    for (auto &graph_to_parent : graph_to_parents_) {
      graph_to_parent.first->SetParentNode(graph_to_parent.second);
      graph_to_parent.first->SetParentGraph(graph_to_parent.second->GetOwnerComputeGraph());
    }
    for (auto &named_model : named_models_) {
      model_->SetSubgraphInstanceNameToModel(named_model.first, named_model.second);
    }
  }

 private:
  ge::GeRootModelPtr model_;
  ge::ComputeGraphPtr root_;
  std::map<std::string, ge::GeModelPtr> named_models_;
  std::map<ge::ComputeGraphPtr, ge::NodePtr> graph_to_parents_;
};

ge::Status BuildStageNodeAndGraph(const ge::NodePtr &node, ge::NodePtr &stage_node, ge::ComputeGraphPtr &stage_graph) {
  std::string stage_name = "Stage_" + node->GetName();
  stage_graph = std::make_shared<ge::ComputeGraph>(stage_name);
  GE_ASSERT_NOTNULL(stage_graph);
  auto stage_desc = ge::MakeShared<ge::OpDesc>(stage_name, ge::PARTITIONEDCALL);
  GE_ASSERT_NOTNULL(stage_desc, "Failed create stage op desc for stage %s", stage_name.c_str());

  std::vector<ge::NodePtr> inputs;
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  for (auto &desc : node->GetOpDesc()->GetAllInputsDescPtr()) {
    GE_ASSERT_GRAPH_SUCCESS(stage_desc->AddInputDesc("args" + std::to_string(inputs.size()), *desc));
    inputs.emplace_back(GraphAddInput(stage_graph, "data" + std::to_string(inputs.size()), inputs.size(), desc));
    GE_ASSERT_NOTNULL(inputs.back());
  }
  auto netoutput = GraphAddOutput(stage_graph, "netoutput", node->GetOpDesc()->GetAllOutputsDescPtr());
  GE_ASSERT_NOTNULL(netoutput);
  size_t idx = 0U;
  for (auto &desc : node->GetOpDesc()->GetAllOutputsDescPtr()) {
    GE_ASSERT_GRAPH_SUCCESS(stage_desc->AddOutputDesc("output" + std::to_string(idx++), *desc));
  }

  stage_node = stage_graph->AddNode(stage_desc);
  GE_ASSERT_NOTNULL(stage_node);
  GE_ASSERT_NOTNULL(stage_node->GetOpDesc());

  for (size_t i = 0U; i < inputs.size(); i++) {
    GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::AddEdge(inputs[i]->GetOutDataAnchor(0), stage_node->GetInDataAnchor(i)));
  }
  std::vector<std::string> input_names;
  std::vector<int64_t> input_indexes;
  for (uint32_t i = 0U; i < stage_node->GetAllOutDataAnchorsSize(); i++) {
    GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::AddEdge(stage_node->GetOutDataAnchor(i), netoutput->GetInDataAnchor(i)));
    input_names.emplace_back(stage_node->GetName());
    input_indexes.emplace_back(static_cast<int64_t>(i));
  }
  GE_ASSERT_NOTNULL(netoutput->GetOpDesc());
  netoutput->GetOpDesc()->SetSrcName(input_names);
  netoutput->GetOpDesc()->SetSrcIndex(input_indexes);
  stage_graph->TopologicalSorting();

  stage_node->GetOpDesc()->AddSubgraphName("f");
  auto stage_subgraph_names = node->GetOpDesc()->GetSubgraphInstanceNames();
  GE_ASSERT_EQ(stage_subgraph_names.size(), 1U);
  stage_node->GetOpDesc()->SetSubgraphInstanceName(0, stage_subgraph_names.front());
  return ge::SUCCESS;
}

ge::Status CutoutUnusedGraphModel(const std::vector<NamedGraph> &used_graph, ge::GeRootModelPtr &stage_model) {
  auto name_2_models = stage_model->GetSubgraphInstanceNameToModel();
  std::unordered_set<std::string> used_graph_names;
  for (auto &item : used_graph) {
    used_graph_names.insert(item.first);
  }
  for (auto &item : name_2_models) {
    auto &graph_name = item.first;
    if (used_graph_names.count(graph_name) == 0U) {
      GELOGI("Cutout unused model of graph %s", graph_name.c_str());
      stage_model->RemoveInstanceSubgraphModel(graph_name);
    } else {
      GE_ASSERT(name_2_models.find(graph_name) != name_2_models.end(), "Model for stage subgraph %s is nullptr",
                graph_name.c_str());
    }
  }
  return ge::SUCCESS;
}

std::unique_ptr<StageModelRecover> CropStageModel(const ge::NodePtr &node, ge::GeRootModelPtr model,
                                                  ge::GeRootModelPtr &stage_model) {
  ge::NodePtr stage_node = nullptr;
  ge::ComputeGraphPtr stage_root = nullptr;
  GE_ASSERT_SUCCESS(BuildStageNodeAndGraph(node, stage_node, stage_root));

  auto library = model->GetRootGraph();
  std::vector<NamedGraph> subgraphs;
  GE_ASSERT_SUCCESS(GetNodeRelatedSubgraphs(stage_node, library, subgraphs));
  GE_ASSERT_NOTNULL(stage_node->GetOpDesc());
  auto stage_subgraph_names = stage_node->GetOpDesc()->GetSubgraphInstanceNames();
  GE_ASSERT_EQ(stage_subgraph_names.size(), 1U);

  auto model_recover = ge::MakeUnique<StageModelRecover>(model);
  GE_ASSERT_NOTNULL(model_recover);
  for (auto &named_subgraph : subgraphs) {
    auto &subgraph = named_subgraph.second;
    if (named_subgraph.first == stage_subgraph_names.front()) {
      GELOGI("Change subgraph %s parent node to stage node %s", named_subgraph.first.c_str(),
             stage_node->GetName().c_str());
      model_recover->GuardSubgraphParent(subgraph, node);
      subgraph->SetParentNode(stage_node);
      subgraph->SetParentGraph(stage_root);
    }
    GELOGI("Collect subgraph %s of stage %s to stage root graph %s", named_subgraph.first.c_str(),
           stage_node->GetName().c_str(), stage_root->GetName().c_str());
    stage_root->AddSubgraph(named_subgraph.first, subgraph);
  }
  stage_model = model;
  stage_model->SetRootGraph(stage_root);
  stage_model->SetFlattenGraph(nullptr);  // Keep flatten graph empty for re-lower new root graph
  GE_ASSERT_SUCCESS(CutoutUnusedGraphModel(subgraphs, stage_model));  // For saving weights memory
  GE_DUMP(stage_root, "CroppedStageGraph");
  return model_recover;
}

bool IsStageNode(const ge::NodePtr &node) {
  return (node->GetType() == ge::PARTITIONEDCALL) && (ge::AttrUtils::HasAttr(node->GetOpDesc(), ge::ATTR_STAGE_LEVEL));
}

bool IsStatefulNode(const ge::NodePtr &node) {
  // Maybe mark more stateful ops here, and now we do this only for the unexpected
  // stage with only 'global step' variable
  const static std::set<std::string> kStatelessOps = {ge::NETOUTPUT, ge::VARIABLE};
  bool stateful = (kStatelessOps.count(node->GetType()) == 0U);
  GELOGD("Node %s %s stateful: %s", node->GetName().c_str(), node->GetType().c_str(), (stateful ? "true" : "false"));
  return stateful;
}

ge::Status GetStageMeaning(const ge::NodePtr &node, const ge::ComputeGraphPtr &graph, bool &meaningful) {
  if (node->GetOutDataNodesSize() != 0U) {  // Stage with data outputs
    GELOGI("Stage %s is meaningful as has data outputs", node->GetName().c_str());
    meaningful = true;
    return ge::SUCCESS;
  }
  for (auto &control_node : node->GetOutControlNodes()) {  // Stage with non-output control nodes
    if (control_node->GetType() != ge::NETOUTPUT) {
      GELOGI("Stage %s is meaningful as control %s", node->GetName().c_str(), control_node->GetName().c_str());
      meaningful = true;
      return ge::SUCCESS;
    }
  }
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  for (auto fn : node->GetOpDesc()->GetSubgraphInstanceNames()) {
    auto subgraph = graph->GetSubgraph(fn);
    GE_ASSERT_NOTNULL(subgraph, "Subgraph %s of stage node %s not found", fn.c_str(), node->GetName().c_str());
    auto nodes = subgraph->GetAllNodes();
    // Stage has subgraph with global side effect
    if (std::any_of(nodes.begin(), nodes.end(), IsStatefulNode)) {
      GELOGI("Stage %s is meaningful as contain stateful nodes", node->GetName().c_str());
      meaningful = true;
      return ge::SUCCESS;
    }
  }
  meaningful = false;
  return ge::SUCCESS;
}
}  // namespace
std::unique_ptr<RtV2PipelineExecutor> RtV2PipelineExecutor::Create(const ge::GeRootModelPtr &model,
                                                                   const ge::DevResourceAllocator &allocator,
                                                                   RtSession *session) {
  (void)allocator;
  return Create(model, session);
}

std::unique_ptr<RtV2PipelineExecutor> RtV2PipelineExecutor::Create(const ge::GeRootModelPtr &model,
    RtSession *session) {
  auto root_graph = model->GetRootGraph();  // Never topo sorting here as compile results not been read
  GE_ASSERT_NOTNULL(root_graph);
  auto executor = std::unique_ptr<RtV2PipelineExecutor>(new (std::nothrow) RtV2PipelineExecutor());
  GE_ASSERT_NOTNULL(executor, "Failed create pipeline executor for model %s", root_graph->GetName().c_str());

  std::map<ge::NodePtr, StageState *> stage_2_states;
  for (auto &node : root_graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    if (!IsStageNode(node)) {
      const static std::set<std::string> kStageGraphOps = {ge::DATA, ge::REFDATA, ge::AIPPDATA, ge::NETOUTPUT};
      GE_ASSERT((kStageGraphOps.count(node->GetType()) > 0U), "Found non-stage node %s %s in stage graph %s",
                node->GetName().c_str(), node->GetType().c_str(), root_graph->GetName().c_str());
      continue;
    }
    bool is_stage_meaningful = true;
    GE_ASSERT_SUCCESS(GetStageMeaning(node, root_graph, is_stage_meaningful));
    if (!is_stage_meaningful) {
      GELOGI("Skip build state for meaning-less stage node %s", node->GetName().c_str());
      continue;
    }
    GELOGI("Start build executor for stage node %s", node->GetName().c_str());
    ge::GeRootModelPtr stage_model = nullptr;
    auto model_recover_guarder = CropStageModel(node, model, stage_model);  // Recover model once executor created
    GE_ASSERT_NOTNULL(model_recover_guarder);
    GE_ASSERT_NOTNULL(stage_model, "Failed crop model for stage %s", node->GetName().c_str());
    executor->stage_executors_.emplace_back(StageState::Create(stage_model, session));
    GE_ASSERT_NOTNULL(executor->stage_executors_.back(), "Failed create stage state for stage node %s",
                      node->GetName().c_str());
    stage_2_states[node] = executor->stage_executors_.back().get();
  }
  GE_ASSERT_SUCCESS(StageState::MappingStageIO(stage_2_states));
  GE_ASSERT_SUCCESS(executor->Initialize());
  return executor;
}

ge::Status RtV2PipelineExecutor::Initialize() {
  std::map<size_t, ModelIoDesc> model_input_descs;
  std::map<size_t, ModelIoDesc> model_output_descs;
  for (auto &stage_state : stage_executors_) {
    std::map<size_t, ModelIoDesc> input_descs;
    GE_ASSERT_SUCCESS(stage_state->GetConsumedModelInputDesc(input_descs));

    for (auto &item : input_descs) {
      inputs_desc_.resize(std::max(item.first + 1U, inputs_desc_.size()));
      inputs_desc_[item.first] = item.second;
      (void)model_input_descs.insert(item);
    }

    std::map<size_t, ModelIoDesc> output_descs;
    GE_ASSERT_SUCCESS(stage_state->GetProducedModelOutputDesc(output_descs));

    for (auto &item : output_descs) {
      GE_ASSERT(model_output_descs.insert(item).second);  // Multi-stage flow to same output
      outputs_desc_.resize(std::max(item.first + 1U, outputs_desc_.size()));
      outputs_desc_[item.first] = item.second;
    }
  }
  if (!model_input_descs.empty()) {
    GE_ASSERT_EQ(model_input_descs.rbegin()->first + 1U, model_input_descs.size());
  }
  if (!model_output_descs.empty()) {
    GE_ASSERT_EQ(model_output_descs.rbegin()->first + 1U, model_output_descs.size());
  }
  return ge::SUCCESS;
}

ge::Status RtV2PipelineExecutor::Load(const gert::ModelExecuteArg &arg, const gert::ModelLoadArg &load_arg) {
  GE_ASSERT(!stage_executors_.empty());
  GELOGI("Load stage state %s", stage_executors_.front()->Id().c_str());
  stage_executors_.front()->Load(arg, load_arg);
  rtContext_t ctx = nullptr;
  GE_ASSERT_RT_OK(rtCtxGetCurrent(&ctx));
  StageState::CtxInitializer ctx_initializer = [ctx]() {
    GELOGI("Initialize stage state worker thread runtime ctx");
    (void)rtCtxSetCurrent(ctx);
  };
  for (size_t i = 1U; i < stage_executors_.size(); i++) {
    notifications_.emplace_back(nullptr);
    GELOGI("Load daemon stage state %s", stage_executors_.front()->Id().c_str());
    GE_ASSERT_SUCCESS(stage_executors_[i]->LoadAsDaemon(arg, load_arg, ctx_initializer, notifications_.back()));
  }
  return ge::SUCCESS;
}

ge::Status RtV2PipelineExecutor::Execute(const gert::ModelExecuteArg &arg, gert::Tensor **inputs, size_t input_num,
                                         gert::Tensor **outputs, size_t output_num, const RunConfig &config) {
  static_cast<void>(arg);
  for (auto &stage_executor : stage_executors_) {
    stage_executor->Reset();
  }
  StageTask args;
  args.num_steps = config.iterations_per_loop;
  args.signal = StageTask::Signal::RUN;
  args.inputs = inputs;
  args.num_input = input_num;
  args.outputs = outputs;
  args.num_output = output_num;
  for (auto &notification : notifications_) {
    notification->Notify(args);
  }
  (void)stage_executors_.front()->Run(args);
  for (auto &notification : notifications_) {
    notification->Wait();
  }
  for (auto &stage_executor : stage_executors_) {
    GE_ASSERT(!stage_executor->IsErrorStatus(), "Pipeline failed as stage %s status error",
              stage_executor->Id().c_str());
  }
  return ge::SUCCESS;
}

ge::Status RtV2PipelineExecutor::Unload() {
  ge::Status result = ge::SUCCESS;
  for (auto &stage_executor : stage_executors_) {
    GELOGI("Start stop stage state %s", stage_executor->Id().c_str());
    auto status = stage_executor->Stop();
    GELOGI("Stage state %s stop %s", stage_executor->Id().c_str(), ((status == ge::SUCCESS) ? "succeed" : "failed"));
    result = (status == ge::SUCCESS) ? result : status;
  }
  return result;
}
}  // namespace gert

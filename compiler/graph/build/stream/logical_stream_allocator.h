/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_STREAM_LOGICAL_STREAM_ALLOCATOR_H_
#define GE_GRAPH_BUILD_STREAM_LOGICAL_STREAM_ALLOCATOR_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "engines/manager/engine_manager/dnnengine_manager.h"
#include "graph/manager/graph_manager_utils.h"
#include "stream_utils.h"

namespace ge {
// Define default fuctions for stream passes.
#define STREAM_PASS_DEFAULT_FUNC(CLASS)  \
  CLASS() : LogicalStreamPass(#CLASS) {} \
  ~CLASS() override = default;           \
  CLASS(const CLASS &) = delete;         \
  CLASS &operator=(const CLASS &) = delete

#define OPTIMIZE_BY_STRUCTURE_PASS_DEFAULT_FUNC(CLASS)  \
  CLASS() : OptimizeByTopoPass(#CLASS) {} \
  ~CLASS() override = default;           \
  CLASS(const CLASS &) = delete;         \
  CLASS &operator=(const CLASS &) = delete

// Base stream class.
class LogicalStreamPass {
 public:
  struct Context {
    int64_t default_stream = kInvalidStream;
    int64_t next_stream = 0;
    bool enable_single_stream = false;
    bool enable_hcom_parallel = false;
  };

  explicit LogicalStreamPass(const std::string &name);
  LogicalStreamPass(const LogicalStreamPass &) = delete;
  LogicalStreamPass &operator=(const LogicalStreamPass &) = delete;
  virtual ~LogicalStreamPass() = default;

  const std::string &GetName() const;
  virtual Status Run(ComputeGraphPtr graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) = 0;

 private:
  std::string name_;
  friend class LogicalStreamAllocator;
};

using LogicalStreamPassPtr = std::shared_ptr<LogicalStreamPass>;

// Allocate streams by label.
class AssignByLabelPass : public LogicalStreamPass {
 public:
  STREAM_PASS_DEFAULT_FUNC(AssignByLabelPass);
  Status Run(ComputeGraphPtr graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) override;
};

// Engines such as hccl require independent Stream.
class IndependentStreamPass : public LogicalStreamPass {
 public:
  STREAM_PASS_DEFAULT_FUNC(IndependentStreamPass);
  Status Run(ComputeGraphPtr graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) override;
};

// Reuse streams or assign new streams based on dependencies.
class AssignByDependencyPass : public LogicalStreamPass {
 public:
  STREAM_PASS_DEFAULT_FUNC(AssignByDependencyPass);
  Status Run(ComputeGraphPtr graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) override;

 private:
  SubgraphPtr GetReusableSubgraph(const SubgraphPtr &subgraph,
                                  const std::unordered_map<NodePtr, SubgraphPtr> &end_subgraph_map,
                                  const std::unordered_map<NodePtr, SubgraphPtr> &pld_subgraph_map) const;

  int64_t AssignNewStream(SubgraphPtr subgraph);

  void UpdateAssignedSubgraphs(Context &context);
  void UpdateReusedSubgraphs();

  bool IsForceAttach(const SubgraphPtr &subgraph) const;
  bool CouldReuse(const SubgraphPtr &subgraph, const SubgraphPtr &pred_subgraph,
                  const std::unordered_map<NodePtr, SubgraphPtr> &pld_subgraph_map) const;

  bool SubGraphCouldReuse(const SubgraphPtr &subgraph, const SubgraphPtr &pred_subgraph,
                          const std::unordered_map<NodePtr, SubgraphPtr> &pld_subgraph_map) const;

  bool IsMemoryPriority() const;

  // <engine name, next stream id>
  std::map<std::string, int64_t> engine_next_streams_;

  // <engine name, stream num>
  std::map<std::string, int64_t> engine_stream_num_;

  // Subgraphs of assign stream by engine
  std::vector<SubgraphPtr> assigned_subgraphs_;

  // <current subgraph, reused subgraph>
  std::vector<std::pair<SubgraphPtr, SubgraphPtr>> reused_subgraphs_;

  mutable std::unordered_set<SubgraphPtr> visited_subgraphs_;
  bool is_memory_priority_ {false};
};

// All nodes in the graph are assigned the same stream.
class SingleStreamPass : public LogicalStreamPass {
 public:
  STREAM_PASS_DEFAULT_FUNC(SingleStreamPass);
  Status Run(ComputeGraphPtr graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) override;
};

// Update the stream of subgraphs to nodes.
class NodeStreamUpdatePass : public LogicalStreamPass {
 public:
  STREAM_PASS_DEFAULT_FUNC(NodeStreamUpdatePass);
  Status Run(ComputeGraphPtr graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) override;
};

// assign stream by parallel group
class UpdateForParallelGroupPass : public LogicalStreamPass {
 public:
  STREAM_PASS_DEFAULT_FUNC(UpdateForParallelGroupPass);
  Status Run(ComputeGraphPtr graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) override;

 private:
  Status UpdateStreamIdFromPreNode(const NodePtr &cur_node,
                                   const std::unordered_map<ge::NodePtr, ge::NodePtr> &total_pld_to_end) const;
};

// Update stream by mde group
class UpdateForMdeGroupPass : public LogicalStreamPass {
 public:
  STREAM_PASS_DEFAULT_FUNC(UpdateForMdeGroupPass);
  Status Run(ComputeGraphPtr graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) override;
};

// Update the stream of subgraphs to nodes.
class UpdateForSkippedEnginePass : public LogicalStreamPass {
 public:
  STREAM_PASS_DEFAULT_FUNC(UpdateForSkippedEnginePass);
  /* Optimize for case like:
   NodeA(stream1) -> Const(stream2) -> NodeB(stream1)
  To case:
   NodeA(stream1) -> Const(stream1) -> NodeB(stream1)
  Which could reduce event number (Const could be other type which belong to skipped engine subgraph)
  */
  Status Run(ComputeGraphPtr graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) override;

 private:
  int64_t GetSingleInoutStream(const NodePtr &node) const;
  // Judge if all predecessors' streams of node are kInvalidStream
  bool AreAllPredStreamsInvalid(const NodePtr &node) const;
};

// AllReduce and backward operators execute in parallel.
class AllReduceParallelPass : public LogicalStreamPass {
 public:
  STREAM_PASS_DEFAULT_FUNC(AllReduceParallelPass);
  Status Run(ComputeGraphPtr graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) override;
 private:
  bool IsHcomNode(const std::string& node_type) const;
  int64_t GetFusion(const NodePtr &node) const;
};

class OptimizeByTopoPass {
 public:
  explicit OptimizeByTopoPass(const std::string &name);
  OptimizeByTopoPass(const OptimizeByTopoPass &) = delete;
  OptimizeByTopoPass &operator=(const OptimizeByTopoPass &) = delete;
  virtual ~OptimizeByTopoPass() = default;
  virtual Status Run(const ComputeGraphPtr &graph) = 0;
  const std::string &GetName() const;

 private:
  std::string name_;
};

using OptimizeByTopoPassPtr = std::shared_ptr<OptimizeByTopoPass>;

class OptimizeIneffectiveMultiStreamPass : public OptimizeByTopoPass {
 public:
  OPTIMIZE_BY_STRUCTURE_PASS_DEFAULT_FUNC(OptimizeIneffectiveMultiStreamPass);
  Status Run(const ComputeGraphPtr &graph) override;
};

// Assign logical streams which is not limited by the number of tasks.
class LogicalStreamAllocator {
  using Context = LogicalStreamPass::Context;

 public:
  explicit LogicalStreamAllocator(const std::map<std::string, int32_t> &max_parallel_num);
  LogicalStreamAllocator(const LogicalStreamAllocator &) = delete;
  LogicalStreamAllocator &operator=(const LogicalStreamAllocator &) = delete;
  ~LogicalStreamAllocator() = default;

  void EnableSingleStream(bool enable);
  void EnableHcomParallel(bool enable);

  Status Assign(const ComputeGraphPtr &root_graph, const Graph2SubGraphInfoList &subgraph_map,
                int64_t &total_stream_num, int64_t &main_stream_num);

 private:
  Status DoAssign(const ComputeGraphPtr &graph, const Graph2SubGraphInfoList &subgraph_map,
                  const std::map<std::string, EngineConfPtr> &engine_confs);
  Status RunPasses(const ComputeGraphPtr &graph, const std::vector<SubgraphPtr> &subgraphs);
  Status RunOptimizeByTopoPasses(const ComputeGraphPtr &graph);
  void RefreshContinuousStreams(const ComputeGraphPtr &graph);

  const std::map<std::string, int32_t> &max_parallel_num_;
  Context context_;
};
}  // namespace ge

#endif  // GE_GRAPH_BUILD_STREAM_LOGICAL_STREAM_ALLOCATOR_H_

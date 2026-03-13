 /**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_V2_VECTOR_FUNC_PARTITIONER_H
#define OPTIMIZE_PLATFORM_V2_VECTOR_FUNC_PARTITIONER_H

#include "graph/compute_graph.h"
#include "cluster.h"
#include "ascir_ops.h"
#include "cluster_dict.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "ascir_register.h"

namespace optimize {
class VectorFuncPartitioner {
 public:
  explicit VectorFuncPartitioner(ge::AscGraph &impl_graph) : impl_graph_(impl_graph) {};
  ge::Status Partition();

 private:
  using InsertOrderMap = std::vector<std::pair<ge::OutDataAnchorPtr, std::vector<ge::InDataAnchorPtr>>>;
  void DebugMergeLog() const;
  ge::Status InitClusters();
  ClusterPtr CreateAndInitCluster(const ge::AscNodePtr &node, size_t &rank);
  void EstablishClusterConnections(ClusterPtr &cluster, const ge::AscNodePtr &node);
  void FixAllCompareClusterConnections();
  void RefineEnableVFFlag(const ge::AscNodePtr &node, bool &enable_vf);
  bool HasReduceNodeInGraph(const ge::AscGraph &impl_graph);
  static ge::Status InitClusterAttr(const std::unique_ptr<ge::ascir::AscIrCodegen> &codegen_impl,
                                    const ge::AscNodePtr &node, ClusterPtr &cluster);
  ge::Status MergeClusters();
  static bool CanMergeClusters(const Cluster &from, const Cluster &to);
  ge::Status SortClustersForBuildSubgraph();
  ge::Status BuildSubgraphs();
  ge::Status ModifySubgraphAttrs(ge::AscGraph &vf_graph);
  static ge::Status SetSubGraphAttrs(ge::AscGraph &vf_graph);
  ge::Status MergeContinuousVectorAxis(ge::AscGraph &vf_graph);
  ge::Status BuildSubgraph(const ClusterPtr &cluster, ge::AscGraph &vf_graph, ge::ascir_op::VectorFunc &vf_op);
  static ge::Status InsertDataAndLoadNode(ge::AscGraph &asc_graph, const ge::OutDataAnchorPtr &out_anchor,
                                          const std::vector<ge::InDataAnchorPtr> &in_anchors, int64_t parent_in_index);
  static ge::Status InsertScalarNode(ge::AscGraph &asc_graph, const ge::OutDataAnchorPtr &out_anchor,
                                     const std::vector<ge::InDataAnchorPtr> &in_anchors, int64_t parent_in_index);
  static ge::Status InsertStoreAndOutputNode(ge::AscGraph &asc_graph, ge::AscNode &pre_node, size_t out_anchor_index,
                                             int64_t parent_out_index);

  static ge::Status TopologicalSortingForVfGraph(ge::AscGraph &graph);

  static ge::Status ReorderAxesForBrcInline(const ge::AscGraph &graph);

  static ge::Status AddRemovePadForBrcInline(ge::AscGraph &graph);

  static ge::Status AddInputDataAnchors(const ge::NodePtr &node, InsertOrderMap &out_data_to_peer_in_anchors);

  static ge::Status AddOutputDataAnchors(const ge::NodePtr &node, InsertOrderMap &out_data_to_peer_in_anchors);

  static bool HasDetectedCycle(const Cluster *const src, const Cluster *const dst);
  static bool IsCompareOp(const ge::AscNodePtr &node);
  bool TryMergeCompareOutputs(const ge::AscNodePtr &compare_node, ClusterPtr &cluster);
  void FixCompareClusterConnections(const ClusterPtr &cluster, const ge::AscNodePtr &compare_node);

  ge::AscGraph &impl_graph_;
  ge::ComputeGraphPtr root_graph_;
  std::vector<ge::AscGraph> sub_graphs_;
  // need merged axis id to new axis id
  std::map<std::vector<ge::AxisId>, ge::AxisPtr> from_id_to_merged_axis_;
  // dictionary for node->cluster
  ClusterDict cluster_dict_;
  size_t subgraph_id_ = 0UL;
  bool graph_has_reduce_node_ = false;  // 缓存图是否有reduce节点
};
}  // namespace optimize
#endif  // OPTIMIZE_PLATFORM_V2_VECTOR_FUNC_PARTITIONER_H

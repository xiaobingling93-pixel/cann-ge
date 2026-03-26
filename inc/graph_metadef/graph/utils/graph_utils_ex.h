/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __INC_METADEF_GRAPH_UTILS_EX_H
#define __INC_METADEF_GRAPH_UTILS_EX_H

#include "graph/node.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"

namespace ge {
class GraphUtilsEx {
 public:
  // Detach from ComputeGraph
  static graphStatus InferOriginFormat(const ComputeGraphPtr &graph);
  static graphStatus InferShapeInNeed(const ComputeGraphPtr &graph);

  // Detach from GraphUtils
  __attribute__((weak)) static ComputeGraphPtr GetComputeGraph(const Graph &graph);
  static ComputeGraphPtr CreateGraphFromOperator(const std::string &name, const std::vector<Operator> &inputs);

  /**
   * 使用ops中的算子为graph对象构造图，且构造出来的图中的算子需要按照ops中的顺序排序
   * @param graph 需要构造图的graph对象
   * @param ops 用于生成计算图的算子
   * @return 计算图指针，成功时，返回生成的ComputeGraph指针 失败返回nullptr
   */
  static graphStatus CreateGraphFromOperatorWithStableTopo(Graph &graph,
      const std::vector<Operator> &ops);
  /**
   * 使用ops中的算子构造计算图，且构造出来的图中的算子需要按照ops中的顺序排序
   * @param graph 需要构造图的graph对象
   * @param ops 用于生成计算图的算子
   * @return 计算图指针，成功时，返回生成的ComputeGraph指针 失败返回nullptr
   */
  static ComputeGraphPtr CreateComputeGraphFromOperatorWithStableTopo(const std::string &name,
      const std::vector<Operator> &ops);

  __attribute__((weak)) static Graph CreateGraphFromComputeGraph(const ComputeGraphPtr compute_graph);
  __attribute__((weak)) static Graph CreateGraph();
  static GraphPtr CreateGraphPtrFromComputeGraph(const ComputeGraphPtr compute_graph);
  static std::unique_ptr<Graph> CreateGraphUniquePtrFromComputeGraph(const ComputeGraphPtr &compute_graph);
  static void BreakConnect(const std::map<OperatorImplPtr, NodePtr> &all_nodes_infos);
  static graphStatus RecoverGraphOperators(const Graph &graph);
  static graphStatus CopyGraph(const Graph &src_graph, Graph &dst_graph);
  __attribute__((weak)) static Operator CreateOperator(const char_t *const operator_name,
                                                       const char_t *const operator_type);

  /**
   * 获取所有需要用户传入输入Tensor的Data节点，当前会排除掉分档场景新插入的Data节点
   * @param graph 图对象
   * @return 用户输入节点集合，失败时返回空集合
   */
  static std::vector<NodePtr> GetUserInputDataNodes(const ComputeGraphPtr &compute_graph);
 private:
  static graphStatus CopyGraphImpl(const Graph &src_graph, Graph &dst_graph,
                                   const std::map<ConstNodePtr, NodePtr> &node_old_2_new,
                                   const std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new);
};
} // namespace ge
#endif // __INC_METADEF_GRAPH_UTILS_EX_H

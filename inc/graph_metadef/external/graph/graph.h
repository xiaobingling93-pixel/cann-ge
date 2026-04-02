/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_GRAPH_GRAPH_H_
#define INC_EXTERNAL_GRAPH_GRAPH_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "graph/operator.h"
#include "graph/gnode.h"

namespace ge {
class Graph;
class GraphImpl;
class GraphBuffer;

using GraphImplPtr = std::shared_ptr<GraphImpl>;
using GraphPtr = std::shared_ptr<Graph>;

using ConstGraphPtr = std::shared_ptr<const Graph>;

/*lint -e148*/
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Graph {
  friend class GraphUtils;
  friend class GraphUtilsEx;

 public:
  ATTRIBUTED_DEPRECATED(Graph(const char_t *))
  explicit Graph(const std::string &name);

  explicit Graph(const char_t *name);

  Graph() = default;

  ~Graph() = default;
  /**
   * 触发内部图的构建, 用于基于Operator的IR构图场景
   * @param inputs 图的输入节点
   * @return
   */
  Graph &SetInputs(const std::vector<Operator> &inputs);

  Graph &SetOutputs(const std::vector<Operator> &outputs);

  /**
   * @brief 设置图的输出节点和节点的输出索引, 此处的`输出索引`代表的是对应的节点的第几个输出，并非是图的第几个输出
   * @param outputs 图的输出节点和节点的输出索引，容器为有序容器，顺序代表图的输出顺序
   * @return graphStatus 设置成功返回GRAPH_SUCCESS, 失败返回其他
   */
  graphStatus SetOutputs(const std::vector<std::pair<GNode, int32_t>> &outputs);

  Graph &SetOutputs(const std::vector<std::pair<Operator, std::vector<size_t>>> &output_indexs);

  ATTRIBUTED_DEPRECATED(Graph &SetOutputs(const std::vector<std::pair<ge::Operator, AscendString) &)
  Graph &SetOutputs(const std::vector<std::pair<ge::Operator, std::string>> &outputs);

  Graph &SetOutputs(const std::vector<std::pair<ge::Operator, AscendString>> &outputs);

  Graph &SetTargets(const std::vector<Operator> &targets);

  bool IsValid() const;
  graphStatus SetValid();

  graphStatus AddOp(const ge::Operator &op);

  ATTRIBUTED_DEPRECATED(graphStatus FindOpByName(const char_t *, ge::Operator &))
  graphStatus FindOpByName(const std::string &name, ge::Operator &op) const;

  graphStatus FindOpByName(const char_t *name, ge::Operator &op) const;

  ATTRIBUTED_DEPRECATED(graphStatus FindOpByType(const char_t *, std::vector<ge::Operator> &))
  graphStatus FindOpByType(const std::string &type, std::vector<ge::Operator> &ops) const;

  graphStatus FindOpByType(const char_t *type, std::vector<ge::Operator> &ops) const;

  ATTRIBUTED_DEPRECATED(graphStatus GetAllOpName(std::vector<AscendString> &) const)
  graphStatus GetAllOpName(std::vector<std::string> &op_name) const;

  graphStatus GetAllOpName(std::vector<AscendString> &names) const;

  ATTRIBUTED_DEPRECATED(graphStatus SaveToFile(const char_t *file_name) const)
  graphStatus SaveToFile(const std::string &file_name) const;

  graphStatus SaveToFile(const char_t *file_name) const;

  ATTRIBUTED_DEPRECATED(graphStatus LoadFromFile(const char_t *))
  graphStatus LoadFromFile(const std::string &file_name);

  graphStatus LoadFromFile(const char_t *file_name);

  graphStatus LoadFromSerializedModelArray(const void *serialized_model, size_t size);

  graphStatus SaveToMem(GraphBuffer &graph_buffer) const;

  graphStatus LoadFromMem(const GraphBuffer &graph_buffer);

  graphStatus LoadFromMem(const uint8_t *data, const size_t len);

  ATTRIBUTED_DEPRECATED(graphStatus GetName(AscendString &) const)
  const std::string &GetName() const;

  graphStatus GetName(AscendString &name) const;

  ///
  /// Set is need train iteration.
  /// If set true, it means this graph need to be run iteration some
  /// times(according variant "npu_runconfig/iterations_per_loop").
  /// @param need_iteration need_iteration:whether to set iteration or not
  ///
  void SetNeedIteration(bool need_iteration);

  std::vector<GNode> GetAllNodes() const;

  /**
   * @brief 获取所有的子图
   */
  std::vector<GraphPtr> GetAllSubgraphs() const;

  /**
   * @brief 根据name获取子图
   * @param name 子图名称
   * @return 返回子图指针
   */
  GraphPtr GetSubGraph(const char *name) const;

  /**
   * @brief 添加子图，以子图的name为key，不允许出现重复。若添加name相同的子图，添加子图失败
   * @param subgraph 子图实例
   * @return 添加成功返回GRAPH_SUCCESS, 失败返回其他
   */
  graphStatus AddSubGraph(const Graph &subgraph);

  /**
   * @brief 根据name移除子图
   * @param name 子图名称
   * @return 移除成功返回GRAPH_SUCCESS, 失败返回其他
   */
  graphStatus RemoveSubgraph(const char *name);

  std::vector<GNode> GetDirectNode () const;

  graphStatus RemoveNode(GNode &node);

  graphStatus RemoveNode(GNode &node, bool contain_subgraph);

  graphStatus RemoveEdge(GNode &src_node, const int32_t src_port_index, GNode &dst_node, const int32_t dst_port_index);

  GNode AddNodeByOp(const Operator &op);

  graphStatus AddDataEdge(GNode &src_node, const int32_t src_port_index,
                          GNode &dst_node, const int32_t dst_port_index);

  graphStatus AddControlEdge(GNode &src_node, GNode &dst_node);

  graphStatus CopyFrom(const Graph &src_graph);

  /**
   * @brief Find the GNode with the target node_name in the graph
   * @param node_name GNode name
   * @return GNodePtr GNode pointer in the graph, return nullptr if failed
   */
  GNodePtr FindNodeByName(const AscendString &node_name) const;

  /**
   * @brief Get the parent graph of current sub graph
   * @return ConstGraphPtr The parent graph shared pointer of current graph, return nullptr if failed
   */
  ConstGraphPtr GetParentGraph() const;

  /**
   * @brief Get the parent node of current sub graph
   * @return GNodePtr The parent node shared pointer of current graph, return nullptr if failed
   */
  GNodePtr GetParentNode() const;

  static GraphPtr ConstructFromInputs(const std::vector<Operator> &inputs, const AscendString &name);

  // 添加AttrValue类型的属性支持
  graphStatus SetAttr(const AscendString &name, const AttrValue &attr_value);
  graphStatus GetAttr(const AscendString &name, AttrValue &attr_value) const;

  enum class DumpFormat : uint32_t {
    kOnnx,
    kTxt,
    kReadable
  };
  /**
   * 将graph序列化到ostream中
   * 不包含权重等数据，只包含图结构及相关属性
   * @param format
   * @param o_stream
   * @return
   */
  graphStatus Dump(DumpFormat format, std::ostream &o_stream) const;

  /**
   * 将graph序列化到执行路径下的文件中
   * 不包含权重等数据，只包含图结构及相关属性
   * @param suffix
   * @return
   */
  graphStatus DumpToFile(DumpFormat format, const AscendString &suffix) const;

 private:

  GraphImplPtr impl_{nullptr};
};
}  // namespace ge

#endif  // INC_EXTERNAL_GRAPH_GRAPH_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_CAN_FUSE_BACKEND_BACKEND_UTILS_H_
#define AUTOFUSE_CAN_FUSE_BACKEND_BACKEND_UTILS_H_
#include "ge_common/ge_api_types.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/graph_utils.h"
#include "utils/autofuse_attrs.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "autoschedule/axis_group.h"

namespace ge {
const std::string kLoadType = "Load";
const std::string kStoreType = "Store";
const std::string kDataType = "Data";
const std::string kScalarType = "Scalar";
const std::string kCastType = "Cast";
const std::string kBroadcastType = "Broadcast";
const std::string kTransposeType = "Transpose";
const std::string kSliceType = "Slice";
const std::string kSplitType = "Split";
const std::string kOutputType = "Output";
const std::string kNetOutputType = "NetOutput";
const std::string kAscBackendType = "AscBackend";
const std::string kAscBackendNoKernelType = "AscBackendNoKernelOp";
const std::string kFusedAscBackendType = "FusedAscBackend";
const std::string kConcatType = "Concat";
const std::string kGatherType = "Gather";
const std::string kReluType = "Relu";
const std::string kExpandDimsType = "ExpandDims";
const std::string kReshapeType = "Reshape";
const std::string kSqueezeType = "Squeeze";
const std::string kUnsqueezeType = "Unsqueeze";
const ge::Expression kSymbolZero = ge::Symbol(0);
const ge::Expression kSymbolOne = ge::Symbol(1);
const std::set kPureSplitIncludedAscirNodeTypes({kSplitType, kDataType, kLoadType, kStoreType, kOutputType});

struct NodeFuseInfo;
struct ComparePairs;
using AxisPair = std::pair<int64_t, int64_t>;
using AxisPairSet = std::set<AxisPair, ComparePairs>;
using NodeAttrPair = std::pair<std::unique_ptr<AscNodeAttr>, std::unique_ptr<AscTensorAttr>>;
using AxisIdAndSizePair = std::vector<std::pair<int64_t, ge::Expression>>;

struct ComparePairs {
  bool operator()(const AxisPair &lhs, const AxisPair &rhs) const {
    if (lhs.second < rhs.second) {
      return true;
    } else if (lhs.second > rhs.second) {
      return false;
    } else {
      return lhs.first < rhs.first;
    }
  }
};

struct CompareSecond {
  bool operator()(const AxisPair &lhs, const AxisPair &rhs) const {
    return lhs.second < rhs.second;
  }
};

// 定义 反推输出数据结构
struct SliceViewOpAttrInfo {
  std::vector<ge::Expression> pre_load_offsets;       //还原到每个维度的offset信息
  std::vector<ge::Expression> pre_load_strides;       //存储每个维度的原始stride信息
  std::vector<ge::Expression> load_offsets;           //还原到每个维度的offset信息
  std::vector<ge::Expression> load_strides;           //存储每个维度的原始stride信息
  std::vector<ge::Expression> pre_data_strides_save;  //存储每个维度的原始stride信息
  std::vector<ge::Expression> pre_data_repeat_save;   //存储每个维度的原始stride信息
  std::vector<ge::Expression> pre_load_strides_save;  //存储每个维度的原始stride信息
  Expression offset_backend;
  std::vector<ge::Expression> backend_strides;
  bool two_slice_node_flag;
};

struct ViewOpAttrInfo {
  std::vector<int64_t> broadcast_info;                      // 用于 broadcast 的轴信息
  std::vector<std::pair<int64_t, int64_t>> transpose_info;  // 用于 transpose 的轴对信息
  SliceViewOpAttrInfo slice_info;                          // 用于 slice 的轴信息

  Status clear() {
    broadcast_info.clear();
    transpose_info.clear();
    slice_info.two_slice_node_flag = false;
    slice_info.offset_backend = Symbol("");
    slice_info.pre_load_offsets.clear();
    slice_info.pre_load_strides.clear();
    slice_info.load_offsets.clear();
    slice_info.load_strides.clear();
    slice_info.pre_data_strides_save.clear();
    slice_info.pre_data_repeat_save.clear();
    slice_info.pre_load_strides_save.clear();
    slice_info.backend_strides.clear();
    return SUCCESS;
  }
};
// 定义 反推输入数据结构
struct TensorAttrInfo {
  std::vector<int64_t> sched_axis;
  std::vector<int64_t> axis;
  std::vector<ge::Expression> repeats;
  std::vector<ge::Expression> strides;
  int64_t topo_id;
  AscTensorDataType dtype;
};

class BackendUtils {
 public:
  /**
   * 该函数用于判断两个节点是否是没有view的操作节点。
   * 最简单的load是它们的输入和输出的 repeats 和 strides 完全一致。
   *
   * @param node ascbackend node。
   * @param data_node 数据节点指针，表示一个操作节点。
   * @param load_node 加载节点指针，表示另一个操作节点。
   * @return 如果两个节点是最简单的操作节点，则返回 true；否则返回 false。
   */
  static bool IsSimplestLoad(const NodePtr &node, const NodePtr &load_node, const NodePtr &data_node,
                             std::vector<ViewOpAttrInfo> &attr_infos, const bool is_condition_with_node_type = true);

  /**
   * 该函数用于查找指定输出锚点的前置 Load 节点。
   * 它会递归地遍历节点的前置节点，并将所有 Load 节点添加到 nodes 列表中。
   *
   * @param o_anchor 输出数据锚点指针，表示查找的起始点。
   * @param nodes 用于存储找到的 Load 节点的向量。
   * @return 如果操作成功，返回 SUCCESS；否则返回错误状态。
   */
  static Status FindPrefaceLoadNodes(const OutDataAnchorPtr &o_anchor, std::vector<NodePtr> &nodes);

  /**
   * 该函数用于删除无效的 Load 节点，并根据条件将数据输出 Tensor 转移到其他节点上。
   * 主要场景包括：
   * 1. 如果 Load 节点包含 View 操作，且当前节点是 Reduction 输出，则将数据输出 Tensor 更新到 store_peer_node 节点上。
   * 2. 如果 Load 节点包含 View 操作，且当前节点不是 Reduction 输出，则根据前序 Load 节点的状态，选择直接更新数据输出
   * Tensor 或反向推导 View 操作并应用到当前数据节点上。
   * 3. 如果 Load 节点是最简形式的 Load（不包含 View 操作），则直接删除该 Load 节点，并更新节点的输入输出映射。
   *
   * @param store_peer_node 与 Load 节点关联的 store_peer_node 节点指针。
   * @param in_anchor_peer 与 Load 节点关联的 in_anchor_peer 锚点指针。
   * @param data_node 当前处理的 data 节点指针。
   * @param del_output_and_store_nodes 记录需要删除的node信息。
   * @param is_reduction 标识当前节点是否为 Reduction 输出。
   * @return 返回 Status 类型的操作结果，SUCCESS 表示操作成功，其他值表示操作失败。
   */
  static Status DeleteInvalidLoad(const NodePtr &node1, const NodePtr &node2, const NodePtr &store_peer_node,
                                  const OutDataAnchorPtr &in_anchor_peer, const NodePtr &data_node,
                                  std::vector<NodePtr> &del_output_and_store_nodes, bool is_reduction);

  /**
   * 该函数为子图（sub_graph）创建输入数据节点，并将这些数据节点与指定的节点（node）连接。
   * 它会根据输入数量（in_nums）创建相应数量的数据节点，并将这些数据节点的输出连接到指定节点的输入。
   *
   * @param sub_graph 要创建输入数据节点的子图指针
   * @param node 需要连接输入数据节点的目标节点指针
   * @param in_nums 需要创建的输入数据节点的数量
   * @param pre_nodes node前序节点信息
   * @return 返回操作状态，SUCCESS 表示成功，其他值表示失败
   */
  static Status CreateSubGraphInput(const ComputeGraphPtr &sub_graph, const NodePtr &node, uint32_t in_nums,
                                    const std::vector<std::pair<ge::NodePtr, int32_t>> &pre_nodes,
                                    bool need_inherit_pre_node_tensor = true);

  /**
   * 该函数为子图（sub_graph）创建网络输出节点。
   * 它会根据输出数量（out_nums）创建相应数量的输入和输出端口，并返回创建的网络输出节点。
   *
   * @param sub_graph 要创建网络输出节点的子图指针
   * @param out_nums 需要创建的输出端口的数量
   * @return 返回创建的网络输出节点指针，如果创建失败则返回 nullptr
   */
  static NodePtr CreateNetOutput(const ComputeGraphPtr &sub_graph, uint32_t out_nums);

  /**
   * 该函数为子图（sub_graph）创建子图输出节点，并将指定的节点（node）的输出连接到这些输出节点。
   * 它会根据输出数量（out_nums）和输出索引（node_output_index）创建相应的输出节点，并将它们与指定节点的输出连接。
   *
   * @param sub_graph 要创建输出节点的子图指针
   * @param node 需要连接输出节点的目标节点指针
   * @param out_nums 需要创建的输出节点的数量
   * @param node_output_index 指定节点输出端口的索引列表
   * @return 返回操作状态，SUCCESS 表示成功，其他值表示失败
   */
  static Status CreateSubGraphOutput(const ComputeGraphPtr &sub_graph, const NodePtr &node, uint32_t out_nums,
                                     const std::vector<uint32_t> &node_output_index);

  /**
   * 该函数判断指定的节点（node）是否asc graph的输出节点。
   * 如果节点的类型为 kOutputType，则返回 true。
   *
   * @param node 需要判断的节点指针
   * @return 返回布尔值，true 表示节点是asc graph的输出节点，false 表示不是asc graph的输出节点
   */
  static bool IsOutputNode(const NodePtr &node);

  /**
   * 该函数判断指定的节点（node）是否输入节点。
   * 如果节点的类型为 kDataType，则返回 true。
   *
   * @param node 需要判断的节点指针
   * @return 返回布尔值，true 表示节点是输入节点，false 表示不是输入节点
   */
  static bool IsInputNode(const NodePtr &node);

  /**
   * 该函数判断指定的节点（node）是否原图的输出节点。
   * 如果节点的类型为 kNetOutputType，则返回 true。
   *
   * @param node 需要判断的节点指针
   * @return 返回布尔值，true 表示节点是原图的输出节点，false 表示不是原图的输出节点
   */
  static bool IsNetOutputNode(const NodePtr &node);

  /**
   * 该函数判断指定的节点（node）是否是空Tensor节点。
   * 如果节点的类型为 kAscBackendNoKernelType，则返回 true。
   *
   * @param node 需要判断的节点指针
   * @return 返回布尔值，true 表示节点是kAscBackendNoKernelType节点，false 表示不是
   */
  static bool IsAscBackendNoKernelNode(const NodePtr &node);

  /**
   * 该函数判断指定的节点（node）是否在融合黑名单中。
   * 如果节点的类型为 kNetOutputType 或 kDataType，则返回 true，表示该节点不需要进行融合。
   *
   * @param node 需要判断的节点指针
   * @return 返回布尔值，true 表示节点在黑名单中，false 表示不在黑名单中
   */
  static bool IsCanFuseBlackList(const NodePtr &node);

  /**
   * 该函数获取指定节点（node）的自动融合属性（AutoFuseAttrs）。
   * 它会从节点的 OpDesc 中获取 AutoFuseAttrs 属性，并返回该属性的指针。
   *
   * @param node 需要获取属性的节点指针
   * @return 返回 AutoFuseAttrs 属性指针，如果属性不存在则返回 nullptr
   */
  static AutoFuseAttrs *GetNodeAutoFuseAttr(const NodePtr &node);

  /**
   * 该函数用于获取指定节点的融合子图（fused_graph）的computer graph。
   *
   * @param node 需要获取融合子图的节点指针
   * @param fused_graph 输出参数，用于存储获取到的融合子图指针
   * @return 返回 SUCCESS 表示成功，否则返回错误状态
   */
  static Status GetNodeFusedGraph(const NodePtr &node, ComputeGraphPtr &fused_graph);

  /**
   *该函数用于获取指定节点的融合子图（fused_asc_graph）
   *
   * @param node 需要获取属性的节点指针
   * @return 返回 Ascgraph属性指针，如果属性不存在则返回 nullptr
   */
  static const std::shared_ptr<AscGraph> GetNodeFusedAscGraph(const NodePtr &node);

  /**
   *该函数用于获取指定节点的融合子图（fused_compute_graph）
   *
   * @param node 需要获取属性的节点指针
   * @return 返回 ComputeGraphPtr属性指针，如果属性不存在则返回 nullptr
   */
  static ComputeGraphPtr GetNodeFusedComputeGraph(const NodePtr &node);

  /**
   * 该函数用于生成节点的输出映射表（node_output_map），排除在 link_map 中已映射的输出索引。
   * 主要逻辑包括：
   * 1. 遍历节点的输出节点数量（node_out_node_size）。
   * 2. 对于每个输出索引，检查是否存在于 link_map 中。
   *    2.1 如果存在，则跳过该输出索引。
   *    2.2 如果不存在，则将该输出索引添加到 node_output_map 中。
   *
   * @param link_map 输出节点的链接映射表，包含已映射的输出索引对
   * @param node_out_node_size 节点的输出节点数量
   * @param node_output_map 输出参数，用于存储生成的输出映射表
   */
  static void GetOutputMap(const std::vector<std::pair<int32_t, int32_t>> &link_map, const uint32_t node_out_node_size,
                           std::vector<int32_t> &node_output_map);

  /**
   * 该函数判断指定的节点（node）是否是需要后端融合的节点。
   * 它会检查节点的自动融合属性（AutoFuseAttrs），如果属性存在且融合图（fused_graph）不为空，
   * 则返回 true，表示该节点需要进行后端融合。
   *
   * @param node 需要判断的节点指针
   * @return 返回布尔值，true 表示节点需要后端融合，false 表示不需要
   */
  static bool IsBackendFuseNode(const NodePtr &node);

  /**
   * 该函数用于更新子图的 fused_subgraph_outputs 属性，确保融合后的节点输出与子图的输出节点对应，简化对子图输出的管理。
   * 主要逻辑包括：
   * 1. 如果节点类型不是 kAscBackendType，则无需更新子图输出属性。
   * 2. 如果 fused_subgraph_outputs 为空，则根据节点的输出锚点及其连接的节点，填充 fused_subgraph_outputs。
   * 3. 如果 fused_subgraph_outputs 不为空，则检查子图的输出节点是否需要更新，删除重复的输出节点。
   *
   * @param subgraph 需要更新的子图指针
   * @param node 需要处理的节点指针
   * @return 返回 SUCCESS 表示更新成功，否则返回错误状态
   */
  static Status UpdateSubgraphOutputAttr(const ComputeGraphPtr &subgraph, const NodePtr &node);

  /**
   * 该函数用于更新FusedAscBackend子图的NetOutput节点的输入anchor与外部FusedAscBackend节点的输出保持一致
   * 主要逻辑包括：
   * 在FusedAscBackend子图中根据NetOutput节点输入anchor的输出peer_out_anchor判断其是否为同一个
   * 对同一个peer_out_anchor输出的进行去重后更新NetOutput节点，保证其与FusedAscBackend节点的输出一致
   *
   * @param node 需要处理的节点指针
   * @return 返回 SUCCESS 表示更新成功，否则返回错误状态
   */
  static Status UpdateFusedAscBackendNetOutput(const NodePtr &node);

  /**
   * 该函数用于更新子图的输出节点列表，删除指定的输出节点索引。
   * 主要逻辑包括：
   * 1. 根据 subgraph_link_map 中的索引，确定需要删除的输出节点位置。
   * 2. 删除 outputs 中对应位置的节点。
   *
   * @param outputs 子图的输出节点列表
   * @param subgraph_link_map 子图输出节点的链接映射关系，包含需要删除的节点索引
   * @return 返回 SUCCESS 表示更新成功，否则返回错误状态
   */
  static Status UpdateSubGraphOutput(std::vector<ge::NodePtr> &outputs,
                                     const std::vector<std::pair<int32_t, int32_t>> &subgraph_link_map);

  /**
   * 该函数用于为新创建的节点创建输入描述属性，并从 node1 和 node2 中复制相应的输入描述信息。
   * 主要逻辑包括：
   * 1. 从 node1 的输入描述中复制属性到新节点的输入描述中。
   * 2. 从 node2 的输入描述中复制属性到新节点的输入描述中。
   * 3. 确保新节点的输入描述与 node1 和 node2 的输入描述一致。
   *
   * @param new_node 新创建的节点指针
   * @param node1 第一个源节点指针
   * @param node2 第二个源节点指针
   * @param node1_input_map node1 输入描述的映射关系
   * @param node2_input_map node2 输入描述的映射关系
   * @return 返回 SUCCESS 表示创建成功，否则返回错误状态
   */
  static Status CreateNewNodeInputDescAttr(const NodePtr &new_node, const NodePtr &node1, const NodePtr &node2,
                                           const std::vector<int32_t> &node1_input_map,
                                           const std::vector<int32_t> &node2_input_map);

  /**
   * 该函数用于为新创建的节点创建输出描述属性，并从 node1 和 node2 中复制相应的输出描述信息。
   * 主要逻辑包括：
   * 1. 从 node1 的输出描述中复制属性到新节点的输出描述中。
   * 2. 从 node2 的输出描述中复制属性到新节点的输出描述中。
   * 3. 确保新节点的输出描述与 node1 和 node2 的输出描述一致。
   *
   * @param new_node 新创建的节点指针
   * @param node1 第一个源节点指针
   * @param node2 第二个源节点指针
   * @param node1_output_map node1 输出描述的映射关系
   * @param node2_output_map node2 输出描述的映射关系
   * @return 返回 SUCCESS 表示创建成功，否则返回错误状态
   */
  static Status CreateNewNodeOutputDescAttr(const NodePtr &new_node, const NodePtr &node1, const NodePtr &node2,
                                            const std::vector<int32_t> &node1_output_map,
                                            const std::vector<int32_t> &node2_output_map);

  /**
   * 该函数尝试移除两个节点之间的控制边（control edges）。
   *
   * @param node1 第一个节点的指针，表示控制边的源节点
   * @param node2 第二个节点的指针，表示控制边的目标节点
   * @return 返回 SUCCESS 表示成功移除控制边，否则返回错误状态
   */
  static Status TryRemoveNodesCtrEdges(const NodePtr &node1, const NodePtr &node2);

  /**
   * 该函数用于打印指定节点的子图信息。
   * 主要逻辑包括：
   * 1. 检查日志级别。
   * 2. 根据节点的类型执行不同的逻辑：
   *    2.1 如果节点类型是
   * kAscBackendType，则获取节点的融合子图，并打印子图的详细信息（包括节点名称、类型、调度轴信息、张量属性等）。 2.2
   * 如果节点类型是 kFusedAscBackendType，则递归调用 DumpAscGraph 打印子图中的所有节点。
   *
   * @param node 需要打印子图信息的节点指针
   * @return 返回 SUCCESS 表示成功，否则返回错误状态
   */
  static Status DumpAscGraph(const NodePtr &node);

  /**
   * 该函数用于获取指定节点融合子图的轴组信息（Axis Group）。
   *
   * @param node 需要获取轴组信息的节点指针
   * @param axes_group 输出参数，用于存储获取到的轴组信息
   * @param axis_map 输入参数，表示轴映射关系，用于轴组信息的转换
   * @return 返回 SUCCESS 表示成功获取轴组信息，否则返回错误状态
   */
  static Status GetAscGraphAxisGroup(const NodePtr &node, optimize::autoschedule::AxisGroup &axes_group,
                                     const AxisPairSet &axis_map);

  /**
   * 该函数用于合并两个轴组axes_order_信息，用于补齐调用后端判断的信息
   *
   * @param group1 第一个轴组信息
   * @param group2 第二个轴组信息
   * @return 返回 true 表示轴组可以合并，否则返回 false
   */
  static void MergeAxesOrder(optimize::autoschedule::AxisGroup &group1, optimize::autoschedule::AxisGroup &group2);

  /**
   * 该函数用于判断两个轴组是否可以合并，并生成合并后的轴组信息。
   *
   * @param group1 第一个轴组信息
   * @param group2 第二个轴组信息
   * @param merged_axes_group 输出参数，用于存储合并后的轴组信息
   * @return 返回 true 表示轴组可以合并，否则返回 false
   */
  static bool IsCanMergeAxisGroup(optimize::autoschedule::AxisGroup &group1, optimize::autoschedule::AxisGroup &group2,
                                  optimize::autoschedule::AxisGroup &merged_axes_group);

  /**
   * 该函数用于检查两个轴集合之间的顺序子集关系。
   * 主要逻辑包括：
   * 1. 遍历两个轴集合，判断第一个集合是否是第二个集合的顺序子集。
   * 2. 如果第一个集合是第二个集合的顺序子集，则返回 true。
   * 3. 否则，继续判断第二个集合是否是第一个集合的循序子集。
   * 4. 如果第二个集合是第一个集合的顺序子集，则返回 true。
   * 5. 如果两个集合都不是对方的循序子集，则返回 false。
   *
   * @param axis1 第一个轴集合
   * @param axis2 第二个轴集合
   * @return 返回 true 表示其中一个集合是另一个集合的顺序子集，否则返回 false
   */
  static bool CheckAxisSubsetRelation(const std::vector<int64_t> &axis1, const std::vector<int64_t> &axis2);

  /**
   * 该函数用于尝试合并两个数据节点，并判断是否可以成功合并。
   * 主要逻辑包括：
   * 1. 获取两个节点的操作描述和输出张量属性。
   * 2. 检查两个节点的输出张量属性是否完全一致：
   *    2.1 如果一致，则返回 true，表示可以合并。
   * 3. 如果两个节点中有一个是 view 节点，则尝试将 view 节点的属性转移到另一个节点上。
   * 4. 判断两个节点是否可以融合：
   *    4.1 如果一个节点是 view 节点，另一个不是，则返回 true。
   *    4.2 如果两个节点不匹配，则返回 false。
   *
   * @param node1 第一个ascbackend节点指针
   * @param node2 第二个ascbackend节点指针
   * @param data_node1 第一个数据节点指针
   * @param data_node2 第二个数据节点指针
   * @return 返回 true 表示两个节点可以合并，否则返回 false
   */
  static bool TryAscDataNodeMerge(const NodePtr &node1, const NodePtr &node2, const NodePtr &data_node1,
                                  const NodePtr &data_node2);

  /**
   * 该函数用于根据轴映射关系转换轴集合。
   * 主要逻辑包括：
   * 1. 遍历轴集合中的每个轴 ID。
   * 2. 根据轴映射关系查找对应的轴 ID。
   * 3. 如果找到映射关系且需要刷新轴 ID，则更新轴集合中的轴 ID。
   *
   * @param node_map 轴映射关系集合
   * @param base_line_axis 需要转换的轴集合
   * @param need_flash 是否需要刷新轴 ID
   * @return 返回 SUCCESS 表示转换成功，否则返回 FAILED
   */
  static Status ConvertAxis(const AxisPairSet &node_map, std::vector<int64_t> &base_line_axis, bool need_flash = true);

  /**
   * 该函数用于根据轴映射关系转换单个轴 ID。
   * 主要逻辑包括：
   * 1. 将单个轴 ID 封装为轴集合。
   * 2. 调用 `ConvertAxis` 函数转换轴集合。
   * 3. 如果转换成功，则更新传入的轴 ID。
   *
   * @param node_map 轴映射关系集合
   * @param axis_id 需要转换的单个轴 ID
   * @param need_flash 是否需要刷新轴 ID
   * @return 返回 SUCCESS 表示转换成功，否则返回 FAILED
   */
  static Status ConvertAxis(const AxisPairSet &node_map, int64_t &axis_id, bool need_flash = true);

  /**
   * 该函数用于反推view算子。
   * 它根据两个节点的输出属性，反推出 broadcast_info 和 transpose_info。
   *
   * @param data_node 数据节点指针，表示一个操作节点。
   * @param load_node 加载节点指针，表示另一个操作节点。
   * @param attr_info.broadcast_info 用于存储需要广播的轴信息的向量。
   * @param attr_info.transpose_info 用于存储需要转置的轴对信息的向量。
   * @param attr_info.slice_info 待扩展开发。
   * @param just_broadcast 判断是否只需要处理broadcast。
   * @return 如果操作成功，返回 SUCCESS；否则返回错误状态。
   */
  static Status BackSteppingViewOp(const NodePtr &data_node, const NodePtr &load_node, ViewOpAttrInfo &attr_info,
                                   bool just_broadcast = false);

  /**
   * 该函数用于把一个load的反推结果应用到另一个load上。
   * 它会根据 broadcast_info 和 transpose_info 修改节点的输出属性。
   *
   * @param load_node 当前ascgraph的输入load。
   * @param attr_info.broadcast_info 用于存储需要广播的轴信息的向量。
   * @param attr_info.transpose_info 用于存储需要转置的轴对信息的向量。
   * @param attr_info.slice_info 待扩展。
   * @return 如果操作成功，返回 SUCCESS；否则返回错误状态。
   */
  static Status FusedApplyViewOp(const NodePtr &data_node, const NodePtr &load_node,
                                 ViewOpAttrInfo &attr_info, const NodePtr &node2);

  /**
   * 该函数用于将一个节点的输出 Tensor 信息转移到另一个节点。
   * 它会复制源节点的输出 repeats 和 strides 到目标节点。
   *
   * @param src_node 源节点指针，表示要复制的节点。
   * @param node 目标节点指针，表示要接收信息的节点。
   * @return 如果操作成功，返回 SUCCESS；否则返回错误状态。
   */
  static Status TransferOutputTensorToOtherNode(const NodePtr &src_node, const NodePtr &node,
                                                const bool is_vertical = true,
                                                const std::vector<int64_t> &broadcast_info = std::vector<int64_t>());

  /**
   * 该函数用于获取数据节点的下一个节点。
   *
   * @param data 数据节点的指针，表示当前节点的位置
   * @return 返回下一个节点的指针，如果获取失败则返回 nullptr
   */
  static NodePtr GetDataNextNode(const NodePtr &data);

  /**
   * 该函数处理View 操作的情况，主要逻辑是根据前序 Load 节点是否包含 View 操作，决定是否需要反推 View 操作并应用到 Data
   * 节点上。
   *
   * @param node1 前序ascbackend节点。
   * @param node2 后序ascbackend节点。
   * @param store_peer_node 存储节点的指针，表示前序节点。
   * @param data_node 当前 Data 节点的指针，表示需要处理的节点。
   * @param load_node 当前 Load 节点的指针，表示需要处理的节点。
   * @return 返回 SUCCESS 表示处理成功，否则返回错误状态。
   */
  static Status ProcessViewOps(const NodePtr &node1, const NodePtr &node2, const NodePtr &store_peer_node,
                               const NodePtr &data_node, const NodePtr &load_node);

  /**
   * 添加ascgraph的输入输出nodes
   *
   * @param asc_graph asc graph。
   * @return 返回 SUCCESS 表示处理成功，否则返回错误状态。
   */
  static Status AddInputOutputNodesForAscGraph(const ComputeGraphPtr &asc_graph);

  /**
   * 获取graph axis字符串
   *
   * @param axis asc graph 轴信息。
   * @return 返回轴信息字符串。
   */
  static std::string AscAxisToStr(const std::vector<AxisPtr> &axis);

  /**
   * 该函数用于扩展的融合策略判断是否满足融合条件。
   *
   * @param node1 第一个数据节点指针
   * @param node2 第二个数据节点指针
   * @param max_fusion_node_input_size 融合策略设置的输入个数限制
   * @return 返回 true 表示两个节点可以融合，否则返回 false
   */
  static bool CanFuseByStrategy(const NodePtr &node1, const NodePtr &node2, uint32_t &max_fusion_node_input_size);

  /**
   * 该函数用于判断是否能循环合并。
   *
   * @param node1 第一个数据节点指针
   * @param node2 第二个数据节点指针
   * @return 返回 true 表示两个节点可以循环合并，否则返回 false
   */
  static bool CanMergeLoopByStrategy(const NodePtr &node1, const NodePtr &node2);

  /**
   * 该函数用于获取输出节点属性。
   * 主要逻辑包括：
   * 1. node1有子图需要获取子图内对应节点的属性，如子图内concat作为子图输出的，就返回concat节点属性。
   *
   * @param node1 第一个数据节点指针
   * @param node2 第二个数据节点指针
   * @return 返回 输出节点属性
   */
  static AutoFuseAttrs *GetAscGraphOutputAttr(const NodePtr &node1, const NodePtr &node2);

  /**
   * 该函数用于获取融合后子图的输入节点及其对应的输入锚点索引。它会根据指定的节点和索引，
   * 找到对应的子图输入节点，并返回该节点的输出锚点的对等输入锚点所属的节点，同时更新输入锚点索引。
   *
   * @param node 融合后子图中的节点指针，表示需要获取输入节点的节点
   * @param index 输入锚点的索引，表示需要获取的输入锚点在子图输入节点中的位置
   * @param in_anchor_idx 用于存储对等输入锚点的索引的引用
   * @return 返回指定索引的子图输入节点的输出锚点的对等输入锚点所属的节点，如果获取失败则返回 nullptr
   */
  static NodePtr GetFusedAscBackendInputNode(const NodePtr &node, const int32_t index, int32_t &in_anchor_idx);

  /**
   * 该函数用于获取输出节点类型。
   * 主要逻辑包括：
   * 1. 合并两个节点的类型返回。
   *
   * @param node1 第一个数据节点指针
   * @param node2 第二个数据节点指针
   * @return 返回 节点类型
   */
  static uint64_t GetAllFuseType(const NodePtr &node1, const NodePtr &node2);

  /**
   * 该函数用于判断两个节点是不是垂直融合。
   * 主要逻辑包括：
   * 1. 如果两个节点是输入输出关系就返回true，否则返回false。
   *
   * @param node1 第一个数据节点指针
   * @param node2 第二个数据节点指针
   * @return 返回 是否垂直融合
   */
  static bool IsVertical(const NodePtr &node1, const NodePtr &node2);

  /**
   * 该函数用于重置节点的fused_subgraph_outputs
   * 主要逻辑包括：
   * 1. 对节点的原始fused_subgraph_outputs进行去重
   * 2. 去重之后重新获取output并设置fused_subgraph_outputs属性
   * @param node 数据节点指针
   * @return 返回 是否成功重置节点的fused_subgraph_outputs属性
   */
  static Status ResetFusedSubgraphOutputsAttr(const NodePtr &node);

  static Status MinSwapCount(const std::vector<int64_t> &in_axis, const std::vector<int64_t> &out_axis,
                             int64_t &swap_count, std::vector<std::pair<int64_t, int64_t>> &swaps);
  static Status MinSwapsToSortDesc(const vector<int64_t>& arr, std::vector<std::pair<int64_t, int64_t>> &swaps);
  static Status ApplySwaps(std::vector<int64_t> &axis, std::vector<ge::Expression> &repeats,
                           std::vector<ge::Expression> &strides, std::vector<int64_t> &sched_axis,
                           const std::vector<std::pair<int64_t, int64_t>> &swaps);
  static Status ApplySwap(std::vector<int64_t> &axis, std::vector<ge::Expression> &repeats,
                          std::vector<ge::Expression> &strides, std::vector<int64_t> &sched_axis,
                          const std::pair<int64_t, int64_t> &swap);
  static Status ApplySwaps(TensorAttrInfo &temp_data_attr, const std::vector<std::pair<int64_t, int64_t>> &swaps);
  static Status PostProBackSteppingViewOp(AscGraph &asc_graph, const NodePtr &cur_node, ViewOpAttrInfo &attr_info,
                                          bool is_back_broadcast);
  static Status TuningSubgraphBeforeMerge(const NodePtr &node1, const NodePtr &node2, const ComputeGraphPtr &graph1,
                                          const ComputeGraphPtr &graph2, const NodeFuseInfo &fuse_info);
  static Status GetPreNodeAndAnchor(const NodePtr &node, const int32_t index, NodePtr &peer_node,
                                    InDataAnchorPtr &in_anchor);
  static Status GetPreAscNode(const NodePtr &parent_node, const InDataAnchorPtr &peer_in_anchor, NodePtr &asc_node);
  static Status GetViewOpNextNodeByLoad(const NodePtr &load_node, NodePtr &finded_node);
  static Status UpdateBroadcastInfoToLoad(AscGraph &asc_graph);
  static Status FusedBackSteppingViewOp(const NodePtr &data_node, const NodePtr &load_node, ViewOpAttrInfo &attr_info,
                                        const bool is_merge_check);
  static Status FusedBackSteppingViewOpBroadcast(TensorAttrInfo &temp_graph_attr, TensorAttrInfo &temp_load_attr,
                                                 ViewOpAttrInfo &attr_info);
  static Status GetPreAscNodeAttrs(const NodePtr &parent_node, const InDataAnchorPtr &peer_in_anchor,
                                   std::vector<int64_t> &axis, std::vector<ge::Expression> &repeats);
  static Status GetSubgraphOutputIndex(const NodePtr &parent_node, const InDataAnchorPtr &peer_in_anchor,
                                       uint32_t &anchor_index, uint32_t &node_index);
  static Status ProcessAscgraphAfterMerge(const NodePtr &new_node);
  static Status GetNodeTensorAttrInfo(const NodePtr &node, TensorAttrInfo &tensor_attr);
  static Status GetGraphAttrInfo(const AscGraph &asc_graph, TensorAttrInfo &current_node_attr);
  static Status SwapGraphAxis(const std::vector<std::pair<int64_t, int64_t>> &transpose_info,
                              std::vector<AxisPtr> &axis);
  static bool SliceHasSameLoad(const NodePtr &node1, const NodePtr &node2,
                               std::vector<std::pair<int32_t, int32_t>> &same_input_map_);
  static bool IsSameBroadCastInfo(std::vector<ViewOpAttrInfo> &attr_infos1, std::vector<ViewOpAttrInfo> &attr_infos2);
  static bool IsNodeAllInputsAreSimplestLoad(const NodePtr &node);
  static bool IsEq(const Expression &e1, const Expression &e2);
  static bool IsEqOne(const Expression &e1);
  static bool IsEqZero(const Expression &e1);
  static Status UpdateContinueStrides(const std::vector<ge::Expression> &repeats,
                                      std::vector<ge::Expression> &strides);

  /**
   * 该函数用于判断指定节点的输入是否为最简单的 Load 操作。它会遍历输入节点的上游节点，
   * 检查是否存在简单的 Load 操作，并确保这些操作没有引入额外的视图操作。
   *
   * @param peer_node 输入节点的指针（输入参数），表示需要检查的节点。
   * @param in_anchor 输入节点的输入数据锚点指针（输入参数），表示需要检查的输入端。
   * @return 如果输入节点的输入是最简单的 Load 操作，则返回 true；否则返回 false。
   */
  static bool AscNodeInputIsSimplestLoad(const NodePtr &peer_node, const InDataAnchorPtr &in_anchor,
                                         std::vector<ViewOpAttrInfo> &attr_infos);

  /**
   * 该函数用于获取指定节点的前置 AscBackend 节点及其对应的输入锚点。
   * 它首先通过融合节点的输入锚点找到子图的输出索引，然后根据索引获取网络输出节点及其输入锚点，
   * 最后通过输出锚点的对等输出锚点获取前置的 AscBackend 节点。
   *
   * @param node 当前节点的指针，表示需要获取前置节点的节点
   * @param fused_in_anchor 融合节点的输入锚点指针，用于定位子图的输出索引
   * @param asc_node 用于存储前置 AscBackend 节点的引用，函数执行成功后将更新此参数
   * @param netoutput_in_anchor 用于存储网络输出节点输入锚点的引用，函数执行成功后将更新此参数
   * @return 如果函数执行成功，返回 SUCCESS；否则返回相应的错误状态
   */
  static Status GetPreAscBackendNodeAndAnchor(const NodePtr &node, const NodePtr &peer_node,
                                              const InDataAnchorPtr &fused_in_anchor, NodePtr &asc_node,
                                              InDataAnchorPtr &netoutput_in_anchor);

  /**
   * 该函数用于检查当前节点的输入是否为最简单的Load操作。
   * 如果节点的类型是kAscBackendType或kFusedAscBackendType，并且其输入节点是最简单的Load操作，则返回true。
   * 否则返回false。
   *
   * @param node 要检查的节点指针
   * @param index 输入锚点的索引
   * @return 如果输入节点是最简单的Load操作，则返回true；否则返回false
   */
  static bool CurNodeInputIsSimplestLoad(const NodePtr &node, const int32_t index,
                                         std::vector<ViewOpAttrInfo> &attr_infos,
                                         const bool is_condition_with_node_type = true);

  /**
   * 该函数用于判断指定节点的前置节点的输入是否为纯粹的load操作。
   *
   * @param node 当前节点的指针
   * @param index 当前节点的输入锚点的索引
   * @return 如果前置节点的输入是最简形式的加载操作，则返回 true；否则返回 false
   */
  static bool PreNodeInputIsSimplestLoad(const NodePtr &node, const int32_t index, std::vector<ViewOpAttrInfo> &attr_infos);

  /*
   * 该函数用于将origin_node中的AscNodeAttr和AscTensorAttr进行备份，将其依次放在backup_node_attr_and_tensor_attr中
   * 如果备份成功返回SUCCESS，否则返回FAILED。
   *
   * @param origin_node 需要备份的节点
   * @param backup_node_attr_and_tensor_attr 保存备份的属性的结构
   * @return 如果备份成功返回SUCCESS，否则返回FAILED
   */
  static Status BackupNodeAscTensorAttrAndAscNodeAttr(const NodePtr &origin_node,
                                                      NodeAttrPair &backup_node_attr_and_tensor_attr);

  /*
   * 该函数用于将backup_node_attr_and_tensor_attr中备份的两个属性还原到load_node中。
   * 如果还原成功返回SUCCESS，失败返回FAILED。
   *
   * @param load_node 需要恢复属性的节点
   * @param backup_node_attr_and_tensor_attr 保存备份的属性的结构
   * @return 如果恢复成功返回SUCCESS，否则返回FAILED
   */
  static Status RecoverNodeAscTensorAttrAndAscNodeAttr(const NodePtr &load_node,
                                                       NodeAttrPair &backup_node_attr_and_tensor_attr);

  /**
   * 该函数用于扩展的融合策略判断是否满足融合条件。
   *
   * @param node1 第一个数据节点指针
   * @param node2 第二个数据节点指针
   * @param node1_map node1轴映射集合
   * @param node2_map node2轴映射集合
   * @param node_fuse_info node1和node2的融合信息
   * @return 返回 true 表示两个节点可以融合，否则返回 false
   */
  static bool CheckSameSchedAxis(const NodePtr &node1, const NodePtr &node2, const AxisPairSet &node1_map,
                                 const AxisPairSet &node2_map, const NodeFuseInfo &node_fuse_info);

  /*
   * 该函数用于获取node的循环轴信息
   *
   * @param node 节点指针
   * @param input_index node节点输入的index
   * @param node_map node的轴映射集合
   * @param axis_id_list node的循环轴列表
   * @return 获取node的循环轴信息成功返回 SUCCESS ，否则返回 FAILED
   */
  static Status GetScheduleAxisInfo(const NodePtr &node, int32_t input_index, const AxisPairSet &node_map,
                                    AxisIdAndSizePair &axis_id_and_size_pair_list);

  /*
   * 该函数用于判断两个节点是否有水平连接
   *
   * @param node1 第一个节点指针
   * @param node2 第一个节点指针
   * @return 如果node1和node2存在水平连接则返回true，否则返回false
   */
  static bool IsHorizontal(const NodePtr &node1, const NodePtr &node2);

  // 单图Dump
  static Status DumpGraph(const std::string &graph_name, const std::string &path, const std::string &suffix);

  // 图融合关系
  static Status AddMergeGraphMap(const std::string &new_node, const std::string &node1, const std::string &node2,
                                 const std::string &merged_graph);
  static Status AddSubGraphMergeGraphMap(const std::string &new_node, const std::string &node1,
                                         const std::string &node2);

  // 单图缓存
  static Status CacheGraph(const NodePtr &node);
  static Status CacheGraph(const std::string &graph_name, const ComputeGraphPtr &graph);

  // 缓存当前图名
  static Status CacheCurrentGraphName(const std::string &graph_name1, const std::string &graph_name2);
  static Status CacheCurrentGraphName(const std::string &graph_name1, const std::string &graph_name2,
                                      const std::string &origin_graph_name);

  // Dump当前正在做融合的图及子图
  static Status DumpCurrentGraphAndSubgraphs();

  // Dump当前图及子图
  static Status DumpGraphAndSubgraphs(const std::vector<std::string> &target_graphs, const std::string &path);

  static bool IsCubeAscNode(const NodePtr &asc_node);

  // 获取AscBackend节点对应Ascgraph中除data, load, store, output节点之外的节点数
  static size_t GetComputeNodeNumInAscgraph(const NodePtr &node);

  // 当pointwise节点有多个输入的时候判断这些输入是否都来自于同一个节点
  static bool IsAllInputFromSameNode(const NodePtr &node);

  // 此函数用于处理Mul有一个输入是scalar的场景，这个时候Mul节点只有一个输入anchor
  static bool HasScalarInAscgraph(const NodePtr &node);
  // 判断node的ascgraph是否有某些节点type
  static bool HasTypesInAscgraph(const NodePtr &node, const std::vector<std::string> &target_types);
  // 判断node的ascgraph是否除了data load store output只有某些节点type
  static bool OnlyHasTypesInAscgraph(const NodePtr &node, const std::vector<std::string> &target_types);
  static void SetReduceOriginalAxisInfo(AutofuseInnerAttrs &attr_new, const AutofuseInnerAttrs &attr1,
                                        const AutofuseInnerAttrs &attr2);

 private:
  static Status BackSteppingViewOpBroadcast(TensorAttrInfo &temp_data_attr, TensorAttrInfo &temp_load_attr,
                                            ViewOpAttrInfo &attr_info);
  static Status FusedBackSteppingViewOpTranspose(TensorAttrInfo &temp_graph_attr, TensorAttrInfo &temp_load_attr,
                                          ViewOpAttrInfo &attr_info);
  static Status BackSteppingViewOpTranspose(const TensorAttrInfo &temp_data_attr, ViewOpAttrInfo &attr_info);
  static Status BackSteppingViewOpSlice(TensorAttrInfo &temp_data_attr, TensorAttrInfo &temp_load_attr,
                                        const NodePtr &load_node, ViewOpAttrInfo &attr_info, bool is_fuse,
                                        const bool is_merge_check);
  static Status FusedApplyViewOpBroadcast(const AscNodeAttr *node_attr, AscTensorAttr *output_attr,
                                          ViewOpAttrInfo &attr_info);
  static Status ApplySwaps(const NodePtr &node, const std::vector<std::pair<int64_t, int64_t>> &swaps);
  static Status FusedApplyViewOpTranspose(const NodePtr &data_node, const NodePtr &load_node, ViewOpAttrInfo &attr_info);
  static Status FusedApplyViewOpSlice(AscTensorAttr *output_attr, const NodePtr &data_node, const NodePtr &load_node,
                                      ViewOpAttrInfo &attr_info, const NodePtr &node2);
  static Status BackSteppingViewOpPro(TensorAttrInfo &temp_data_attr, TensorAttrInfo &temp_load_attr,
                                      ViewOpAttrInfo &attr_info, bool just_broadcast = false);
  static Status FusedApplyViewOpPro(const AscNodeAttr *node_attr, AscTensorAttr *output_attr, const NodePtr &data_node,
                                    const NodePtr &load_node, ViewOpAttrInfo &attr_info, const NodePtr &node2);
  static Status PostProBackSteppingViewOpTranspose(TensorAttrInfo &temp_graph_attr, TensorAttrInfo &temp_load_attr,
                                                   ViewOpAttrInfo &attr_info, const std::string &cur_node_type);
  static Status PostProBackSteppingViewOpPro(TensorAttrInfo &temp_graph_attr, TensorAttrInfo &temp_load_attr,
                                             ViewOpAttrInfo &attr_info, bool is_back_broadcast,
                                             const std::string &cur_node_type);
  static bool ViewOpMerge(const NodePtr &data_node1, const NodePtr &data_node2, const NodePtr &load_node1,
                          const NodePtr &load_node2, bool is_simplest_load1, bool is_simplest_load2);
  static Status GetPreStoreNode(const NodePtr &node, const int32_t index, NodePtr &store_node);
  static Status ApplyTransposeOp(const ComputeGraphPtr &graph,
                                 const std::vector<std::pair<int64_t, int64_t>> &transpose_info);
  static Status RemoveTransposeOp(const ComputeGraphPtr &graph);
  static Status GetTransposeInfo(NodePtr &node, std::vector<std::pair<int64_t, int64_t>> &transpose_info);
  static Status UpdateTransposeBeforeMerge(const NodePtr &node2, const ComputeGraphPtr &graph1,
                                           const ComputeGraphPtr &graph2, const NodeFuseInfo &fuse_info);
  static Status UpdateBroadcastBeforeMerge(const NodePtr &node1, const NodePtr &node2);
  static bool IsSameAttrInfoInVector(std::vector<ViewOpAttrInfo> &attr_infos);
};
}  // namespace ge

#endif  // AUTOFUSE_CAN_FUSE_BACKEND_BACKEND_UTILS_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_ASCEND_GRAPH_PARSER_H_
#define PARSER_ASCEND_GRAPH_PARSER_H_

#include "tuning_space.h"
#include "graph/compute_graph.h"

namespace att {
struct ScheduleAttr {
  std::vector<ge::AxisPtr> sched_axis_info; // sched axis info
  std::vector<int64_t> block_out_dim_info; // block outer axis ids
  int64_t exe_order;
  int64_t loop_axis_id;
  ge::ExecuteCondition exec_condition{ge::ExecuteCondition::kNoCache};
};

class AscendGraphParser {
public:
  explicit AscendGraphParser(TuningSpacePtr tuning_space) : tuning_space_(tuning_space) {}
  virtual ~AscendGraphParser() = default;

  ge::Status GraphParser(const ge::AscGraph &graph);

private:
  // 获取所有轴对应的原始轴id
  ge::Status ParserOriginAxis(const ge::AscGraph &graph);

  // 根据图中node解析调度信息
  ge::Status ParserSchedInfo(const ge::AscGraph &graph);

  // 创建求解器需要的轴信息
  ge::Status ParseInputTensor(const ge::AscNodePtr &ge_node, const NodeInfo &node_info, size_t in_id, TensorPtr &tensor);
  ge::Status AddSubAxisInfo(ge::AxisPtr &axis_info);
  ge::Status CreateSubAxisInfo(const ge::AscGraph &graph);

  // 把ascir 轴转换为 att轴信息
  void ParserSubAxis(const ge::AxisPtr &axis, SubAxisPtr &sub_axis_ptr) const;

  // 构建轴之间的关系，轴的父轴和原始轴
  void MakeSubAxisRelation(void);

  // 解析tensor内存信息
  ge::Status ConstructQueueContainer(const ge::AscTensorAttr &ascir_tensor_info);
  ge::Status ConstructBufferContainer(const ge::AscTensorAttr &ascir_tensor_info);
  ge::Status ConstructGlobalContainer(const ge::AscTensorAttr &ascir_tensor_info);
  ge::Status ParseTensorMemInfo(const ge::AscTensorAttr &ascir_tensor_info, std::string &node_type,
                           const TensorPtr &tensor);

  void ParseTensorOrigIdx(TensorPtr &tensor) const;

  // 计算tensor大小
  ge::Status ParseTensorDims(TensorPtr &tensor, ge::AscTensorAttr &tensor_attr);

  // 获取tensor的大小
  ge::Status GetTensorAxes(TensorPtr &tensor, ge::AscTensorAttr &tensor_attr);

  // 获取tensor属性信息
  ge::Status GetTensorAttrs(const ge::AscNodePtr &node, const TensorPtr &tensor,
                        size_t id, bool input);

  // 设置遍历轴优先级和默认值
  ge::Status SetAxisPriority(const ge::AscGraph &graph);

  // 解析Workspace大小
  ge::Status ParseWorkspaceNode(const ge::AscNodePtr &ge_node);

  // 解析node output tensors
  ge::Status ParserNodeOutputInfos(const ge::AscNodePtr &ge_node, const ge::AscGraph &graph,
                               NodeInfo &node_info);

  void UpdateTensorLocType(const ge::AscNodePtr &ge_node, size_t &in_id, TensorPtr &tensor) const;

  // 解析node input tensors
  ge::Status ParserNodeInputInfos(const ge::AscNodePtr &ge_node, const ge::AscGraph &graph,
                              NodeInfo &node_info);
  // 解析block outer信息
  ge::Status ParserBlockDimInfo();

  // 更新共存tensor信息，设置tensor对应的container
  void UpdateContainer(ContainerPtr &container, const int32_t new_id);

  // 获取节点归属于哪些Data节点
  ge::Status GetNodeFromData(const ge::AscNodePtr &ge_node, NodeInfo &node_info);

  // 转换node信息到tuning space
  ge::Status ConvertNodeInfos(const ge::AscNodePtr &ge_node, const ScheduleAttr &attrs, const ge::AscGraph &graph,
                              const bool use_cache_flag);

  // 把graph信息解析为tuning space
  ge::Status ConvertToTuningSpace(const ge::AscGraph &graph);

  // 更新tensor信息到tuning space
  void AssembleTensorInfos();

  // 解析ascir graph可选属性信息
  void ParserOptionalInfos(const ge::AscGraph &graph) const;

  // 校验轴id是否有效
  ge::Status CheckAxisIdValid(const int64_t axis_id);

  ge::Status CheckAxisIdValid(std::vector<int64_t> &axis_ids);

  void SaveTmpBufferInfos(const std::string &node_name, std::map<int64_t, Expr> &max_tmp_buffers_map,
                          std::vector<ge::TmpBuffer> &tmp_buffers) const;
  void SetContinuesStrides(TensorPtr &tensor, ge::AscTensorAttr &tensor_attr) const;

  // 打印tuning space信息
  std::string TuningSpacePrint(const SubAxis &sub_axis) const;

  std::string TuningSpacePrint(const Tensor &tensor) const;

  std::string TuningSpacePrint(const Container &container) const;

  std::string TuningSpacePrint(const NodeInfo &node_info) const;

  std::string TuningSpacePrint() const;

  // 非正式方案，正式方案需要由Schedule计算预留大小，待正式方案合入后删除
  ge::Status CalculateReservedUbSize(const ge::AscGraph &graph);

  // 检测Reduce/Broadcast分核Store冲突场景
  ge::Status CheckReduceBroadcastSplitStoreConflict();

  // 辅助方法：检查并标记轴是否为 Reduce 分核轴
  bool CheckAndMarkReduceSplitAxis(SubAxis *axis, const std::set<std::string> &reduce_axis_orig_names);

  // 辅助方法：检查并标记轴是否为 Broadcast 分核轴
  bool CheckAndMarkBroadcastSplitAxis(SubAxis *axis, const std::set<std::string> &broadcast_axis_orig_names);

private:
  std::map<std::string, TensorPtr> tensor_info_; // 记录所有tensor信息，目前未用到
  std::map<size_t, SubAxisPtr> sub_axes_info_; // 轴id到att处理过的轴信息, sub axis ptr owner
  std::map<size_t, ge::AscNodePtr> topo_order_node_; // <算子执行拓扑序，执行node>
  std::map<int64_t, ge::AxisPtr> axes_info_; // 记录图上的所有轴信息
  std::map<int32_t, ContainerPtr> queue_containers_; // 图上的queue信息
  std::map<int32_t, ContainerPtr> buf_containers_; // 图上的buf信息
  std::map<HardwareDef, ContainerPtr> global_containers_; // 图上的global信息
  std::unordered_map<int64_t, std::vector<int64_t>> orig_axes_info_; // 获取所有轴对应的原始轴id
  std::unordered_map<int64_t, std::vector<int64_t>> parent_axes_info_; // 获取所有轴对应父轴id
  std::unordered_map<int64_t, int64_t> orig_to_first_vec_id_; // 一个原始轴出现在第一个vectorized轴id（稀疏场景）
  std::map<ContainerPtr, std::map<int32_t, std::vector<TensorPtr>>> combined_tensors_; // 一个container里面的tensor是否共存
  std::map<ge::AscNodePtr, ScheduleAttr> graph_sched_info_; // node的调度信息
  TuningSpacePtr tuning_space_;
};
} // namespace att

#endif // PARSER_ASCEND_GRAPH_PARSER_H_

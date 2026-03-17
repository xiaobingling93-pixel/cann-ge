/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arg_list_reorder.h"

namespace att {
namespace {
constexpr size_t kOrderIdStart = 1;
const std::vector<std::string> kDefaultNodeWhiteList = {
    "Data", "Output", "Constant", "Workspace", "TbufData"
};
}

// 初始化arglist的优先级连边图
// 初始化排序原则：父轴的优先级必须大于子轴，举例说明:
// z0t->z0tt, z0t是z0tt的父轴，存在约束z0tt<=z0t，如果子轴优先级高于父轴，
// 轴排序算法可能会求解失败(子轴过大导致父轴无切分空间，容易超UB)
// 因此这是一个功能性的问题，即使子轴是reduce/broadcast/innermost轴，其优先级依然不能高于父轴(elementwise)
ge::Status ArgListReorder::InitArgListPriorityGraph(const vector<AttAxisPtr> &arg_list) {
  for (size_t i = 0; i < arg_list.size(); i++) {
    auto arg = arg_list[i];
    for (const auto &from_axis : arg->from_axis) {
      GE_ASSERT_TRUE(axis_name_2_id_map_.find(from_axis->name) != axis_name_2_id_map_.end(),
                     "from axis cannot be found in arg[%s]", arg->name.c_str());
      size_t from_axis_id = axis_name_2_id_map_[from_axis->name];
      GE_ASSERT_TRUE(graph_->AddEdge(from_axis_id + kOrderIdStart, i + kOrderIdStart),
                     "add edge from axis[%s] to arg[%s] failed", from_axis->name.c_str(), arg->name.c_str());
      GELOGI("add edge from axis[%s] to arg[%s] successfully", from_axis->name.c_str(), arg->name.c_str());
    }
  }
  return ge::SUCCESS;
}

bool ArgListReorder::HandleProperty(const SubAxis *dim, att::ArgListReorder::AxisProperty property, bool is_reduce,
                                    bool is_broadcast) {
  if ((property == AxisProperty::kReduce && is_reduce) || (property == AxisProperty::kBroadcast && is_broadcast)) {
    if (property == AxisProperty::kReduce) {
      reduce_map_.insert({dim->name, true});
    } else {
      broadcast_map_.insert({dim->name, true});
    }
    return true;
  }
  return false;
}

// 判断是否是reduce轴(不存在keep_dims=false(输出吞轴)，schedule会补轴)，规则如下：
// 1.node的输入tensor中某根轴的repeat不为1，输出tensor对应轴的repeat为1，那么该轴为reduce轴
// 2.node的输入tensor中某根轴的stride不为0，输出tensor对应轴的stride为0，那么该轴为reduce轴
bool ArgListReorder::CheckAxisProperty(const SubAxis *dim, const Expr &repeat, const Expr &stride,
                                       const std::vector<TensorPtr> &output_tensors, AxisProperty property) {
  for (const auto &tensor : output_tensors) {
    for (size_t i = 0; i < tensor->dim_info.size(); i++) {
      auto dim_info = tensor->dim_info[i];
      auto output_repeat = tensor->repeat[i];
      auto output_stride = tensor->stride[i];
      if (dim_info->name != dim->name) {
        continue;
      }
      bool is_reduce = (repeat != ge::sym::kSymbolOne && output_repeat == ge::sym::kSymbolOne) ||
                       (stride != ge::sym::kSymbolZero && output_stride == ge::sym::kSymbolZero);
      bool is_broadcast = (output_repeat != ge::sym::kSymbolOne && repeat == ge::sym::kSymbolOne) ||
                          (output_stride != ge::sym::kSymbolZero && stride == ge::sym::kSymbolZero);
      GELOGD("Begin to handle property is_reduce[%d], is_broadcast[%d], property[%u] of dim[%s]", is_reduce,
             is_broadcast, property, dim->name.c_str());
      if (HandleProperty(dim, property, is_reduce, is_broadcast)) {
        return true;
      }
    }
  }
  return false;
}

// 判断是否是broadcast轴(不存在隐式broadcast(输入吞轴)，schedule会补轴)，规则如下：
// 1.node的输出tensor中某根轴的repeat不为1，输入tensor对应轴的repeat为1，那么该轴为broadcast轴
// 2.node的输出tensor中某根轴的stride不为0，输入tensor对应轴的stride为0，那么该轴为broadcast轴
bool ArgListReorder::CheckBroadcast(const SubAxis *dim, const Expr &repeat, const Expr &stride,
                                    const std::vector<TensorPtr> &output_tensors) {
  return CheckAxisProperty(dim, repeat, stride, output_tensors, AxisProperty::kBroadcast);
}

bool ArgListReorder::CheckReduce(const SubAxis *dim, const Expr &repeat, const Expr &stride,
                                 const std::vector<TensorPtr> &output_tensors) {
  return CheckAxisProperty(dim, repeat, stride, output_tensors, AxisProperty::kReduce);
}

bool ArgListReorder::IsReduceAxisBlockSplit(const std::vector<SubAxisPtr> &all_axes,
                                            const std::set<std::string> &reduce_axis_ori_axes_set) const {
  for (const auto &axis : all_axes) {
    if (!axis->is_bind_multi_core) {
      continue;
    }
    GELOGI("axis[%s] bind multi core", axis->name.c_str());
    for (auto &orig_axis : axis->orig_axis_name) {
      GELOGI("bind multi core axis[%s] has orig_axis: [%s]", axis->name.c_str(), orig_axis.c_str());
      if (reduce_axis_ori_axes_set.find(orig_axis) != reduce_axis_ori_axes_set.end()) {
        return true;
      }
    }
  }
  return false;
}

void ArgListReorder::SaveReduceAxisOrig(const SubAxis *reduce_axis, std::set<std::string> &reduce_axis_ori_axes_set) const {
  for (SubAxis *ori_axis : reduce_axis->orig_axis) {
    GELOGI("reduce axis ori set add: [%s]", ori_axis->name.c_str());
    reduce_axis_ori_axes_set.insert(ori_axis->name);
  }
}

void ArgListReorder::RecordSpecialArgs(const NodeInfo &node, const TensorPtr &tensor, size_t id,
                                       const std::vector<TensorPtr> &output_tensors,
                                       std::set<std::string> &reduce_axis_ori_axes_set) {
  auto dim = tensor->dim_info[id];
  auto repeat = tensor->repeat[id];
  auto stride = tensor->stride[id];
  if (CheckReduce(dim, repeat, stride, output_tensors)) {
    GELOGD("find reduce axis %s from node %s", dim->name.c_str(), node.name.c_str());
    SaveReduceAxisOrig(dim, reduce_axis_ori_axes_set);
    reduce_map_.insert({dim->name, true});
  }
  if (CheckBroadcast(dim, repeat, stride, output_tensors)) {
    GELOGD("find broadcast axis %s from node %s", dim->name.c_str(), node.name.c_str());
    broadcast_map_.insert({dim->name, true});
  }
  if (dim->is_node_innerest_dim) {
    GELOGD("find innermost dim axis %s from node %s", dim->name.c_str(), node.name.c_str());
    innermost_dim_map_.insert({dim->name, true});
  }
}

// 查找特殊轴，目前仅支持reduce/broadcast和最内轴，后续如果有其余特殊轴在这里扩展
void ArgListReorder::FindSpecialArgs() {
  for (const auto &node : tuning_space_->node_infos) {
    if (std::find(kDefaultNodeWhiteList.begin(), kDefaultNodeWhiteList.end(), node.node_type) !=
        kDefaultNodeWhiteList.end()) {
      continue;
    }
    
    const auto &input_tensors = node.inputs;
    const auto &output_tensors = node.outputs;
    std::set<std::string> reduce_axis_ori_axes_set;
    
    for (const auto &tensor : input_tensors) {
      for (size_t i = 0; i < tensor->dim_info.size(); i++) {
        RecordSpecialArgs(node, tensor, i, output_tensors, reduce_axis_ori_axes_set);
      }
      auto asc_node = node.node_ptr;
      auto graph = tuning_space_->asc_graph;
      std::string last_dim_name;
      if (AttUtils::IsLoadStoreNode(asc_node.get()) && (asc_node != nullptr) && (graph != nullptr) &&
          (AttUtils::GetLastTileSplitAxisName(*asc_node, *graph, last_dim_name))) {
        load_store_inner_most_dims_.insert(last_dim_name);
        GELOGD("[DFX]Found Tile split axis %s in load/store node", last_dim_name.c_str());
      }
    }
    tiling_R_ = tiling_R_ || IsReduceAxisBlockSplit(tuning_space_->sub_axes, reduce_axis_ori_axes_set);
  }
}

// from_axes里的每根轴连边到to_axes里的每根轴
// 举例，假设from_axes里的轴都是reduce轴，to_axes里的轴都是非reduce轴，那么连边表示reduce轴优先级高于非reduce轴
ge::Status ArgListReorder::AddEdgeGroups(const std::vector<std::string> &from_axes_group,
                                         const std::vector<std::string> &to_axes_group) {
  for (const auto &from_axis : from_axes_group) {
    GE_ASSERT_TRUE(axis_name_2_id_map_.find(from_axis) != axis_name_2_id_map_.end(),
                   "from axis[%s] cannot be found in arg", from_axis.c_str());
    size_t from_axis_id = axis_name_2_id_map_[from_axis];
    
    for (const auto &to_axis : to_axes_group) {
      GE_ASSERT_TRUE(axis_name_2_id_map_.find(to_axis) != axis_name_2_id_map_.end(),
                     "to axis[%s] cannot be found in arg", to_axis.c_str());
      size_t to_axis_id = axis_name_2_id_map_[to_axis];
      GE_ASSERT_TRUE(graph_->AddEdge(from_axis_id + kOrderIdStart, to_axis_id + kOrderIdStart),
                     "add edge from axis[%s] to axis[%s] failed", from_axis.c_str(), to_axis.c_str());
      GELOGI("add edge from axis[%s] to axis[%s] successfully", from_axis.c_str(), to_axis.c_str());
    }
  }
  return ge::SUCCESS;
}

// 构建ArgList的优先级的连边图
// 优先级遵循的原则（依次排序）
// 1.父轴高于子轴
// 2.reduce轴高于非reduce轴
// 3.broadcast轴高于非broadcast轴（和reduce轴优先级孰高孰低可能存在争议，暂定reduce更高）
// 4.最内轴高于非最内轴
// 举例说明
// 假设[z0,z1].merge-> z0z1, z0z1.split->[z0z1T, z0z1t], z2.split->[z2T, z2t]
// 其中z0z1t是reduce轴，且z2t是最内轴，优先级图如下：
// z0  z1
//  \  /
//  z0z1
//   |
//  z0z1t  z2
//   \     /
//     z2t
// 拓扑排序结果为[z0, z1, z0z1, z0z1t, z2, z2t]
// 满足规则1,2,3,4
// 这里有一个冲突点，z0z1t是reduce轴，z2t是非reduce轴以及最内轴，但是连边的时候优先连的是reduce轴
// 即z0z1t->z2t, 在连最内轴的group到非最内轴的group的时候，会尝试将z2t->z0z1t，这里似乎冲突，但是图数据结构中
// 连边操作会判环，如果z0z1t->z2t, z2t->z0z1t会成环，因此不会连z2t->z0z1t，这样也是遵循reduce轴优先级高于最内轴这条原则
ArgListReorder::AxisCategories ArgListReorder::CategorizeAxesByProperty(const vector<AttAxisPtr> &arg_list) {
  AxisCategories categories;
  for (const auto &arg : arg_list) {
    if (reduce_map_.find(arg->name) != reduce_map_.end()) {
      categories.reduce_arg_names.push_back(arg->name);
    } else {
      categories.non_reduce_arg_names.push_back(arg->name);
    }
    
    if (broadcast_map_.find(arg->name) != broadcast_map_.end()) {
      categories.broadcast_arg_names.push_back(arg->name);
    } else {
      categories.non_broadcast_arg_names.push_back(arg->name);
    }
    
    if (innermost_dim_map_.find(arg->name) != innermost_dim_map_.end()) {
      categories.innermost_dim_arg_names.push_back(arg->name);
    } else {
      categories.non_innermost_dim_arg_names.push_back(arg->name);
    }
  }
  return categories;
}

ge::Status ArgListReorder::ApplyPriorityRules(bool tiling_R, const AxisCategories &categories) {
  if (!tiling_R) {
    GE_ASSERT_SUCCESS(AddEdgeGroups(categories.reduce_arg_names, categories.non_reduce_arg_names),
                      "add edge from reduce to non reduce failed");
    GE_ASSERT_SUCCESS(AddEdgeGroups(categories.broadcast_arg_names, categories.non_broadcast_arg_names),
                      "add edge from broadcast to non broadcast failed");
    GE_ASSERT_SUCCESS(AddEdgeGroups(categories.innermost_dim_arg_names, categories.non_innermost_dim_arg_names),
                      "add edge from innermost dim to non innermost dim failed");
  } else {
    GE_ASSERT_SUCCESS(AddEdgeGroups(categories.innermost_dim_arg_names, categories.non_innermost_dim_arg_names),
                      "add edge from innermost dim to non innermost dim failed");
    GE_ASSERT_SUCCESS(AddEdgeGroups(categories.broadcast_arg_names, categories.non_broadcast_arg_names),
                      "add edge from broadcast to non broadcast failed");
    GE_ASSERT_SUCCESS(AddEdgeGroups(categories.reduce_arg_names, categories.non_reduce_arg_names),
                      "add edge from reduce to non reduce failed");
  }
  return ge::SUCCESS;
}

// 构建ArgList的优先级的连边图
// 优先级遵循的原则（依次排序）
// 1.父轴高于子轴
// 2.reduce轴高于非reduce轴
// 3.broadcast轴高于非broadcast轴（和reduce轴优先级孰高孰低可能存在争议，暂定reduce更高）
// 4.最内轴高于非最内轴
// 举例说明
// 假设[z0,z1].merge-> z0z1, z0z1.split->[z0z1T, z0z1t], z2.split->[z2T, z2t]
// 其中z0z1t是reduce轴，且z2t是最内轴，优先级图如下：
// z0  z1
//  \  /
//  z0z1
//   |
//  z0z1t  z2
//   \     /
//     z2t
// 拓扑排序结果为[z0, z1, z0z1, z0z1t, z2, z2t]
// 满足规则1,2,3,4
// 这里有一个冲突点，z0z1t是reduce轴，z2t是非reduce轴以及最内轴，但是连边的时候优先连的是reduce轴
// 即z0z1t->z2t, 在连最内轴的group到非最内轴的group的时候，会尝试将z2t->z0z1t，这里似乎冲突，但是图数据结构中
// 连边操作会判环，如果z0z1t->z2t, z2t->z0z1t会成环，因此不会连z2t->z0z1t，这样也是遵循reduce轴优先级高于最内轴这条原则
ge::Status ArgListReorder::BuildArgListPriorityGraph(const vector<AttAxisPtr> &arg_list, bool tiling_R) {
  GE_ASSERT_SUCCESS(InitArgListPriorityGraph(arg_list), "init arg list graph failed");
  FindSpecialArgs();
  return ApplyPriorityRules(tiling_R, CategorizeAxesByProperty(arg_list));
}

void ArgListReorder::MakeSureLoadStoreInnerestSameOrder(const std::vector<AttAxisPtr> &arg_list) const {
  // 处理Load/Store的同等切分优先级
  size_t min_order = SIZE_MAX;
  std::vector<AttAxisPtr> args_to_make_same_order;
  std::string args_name;
  for (const auto &arg : arg_list) {
    const bool is_load_store = (load_store_inner_most_dims_.find(arg->name) != load_store_inner_most_dims_.cend());
    GELOGD("[DFX]arg name %s, axis_pos %d, order %d, bind_multicore %d, is_load_store %d", arg->name.c_str(),
           arg->axis_pos, arg->order, arg->bind_multicore, is_load_store);
    if (is_load_store && (!arg->bind_multicore && (arg->axis_pos == AxisPosition::INNER))) {
      args_to_make_same_order.emplace_back(arg);
      if (min_order > arg->order) {
        min_order = arg->order;
      }
    }
  }
  for (auto &arg : args_to_make_same_order) {
    GELOGD("[DFX]Set arg %s order to %zu", arg->name.c_str(), min_order);
    arg->order = min_order;
  }
}

std::vector<AttAxisPtr> ArgListReorder::GetNewArgList(const std::vector<size_t> &topo_order,
                                                      const std::vector<AttAxisPtr> &arg_list) const {
  std::vector<AttAxisPtr> new_arg_list;
  for (auto &order_id : topo_order) {
    if ((order_id >= kOrderIdStart) && (order_id <= arg_list.size() + kOrderIdStart - 1)) {
      arg_list[order_id - kOrderIdStart]->order = order_id;
      new_arg_list.emplace_back(arg_list[order_id - kOrderIdStart]);
    }
  }

  GE_ASSERT_TRUE(new_arg_list.size() == arg_list.size(),
                 "arg list size is not equal after reorder, new_arg_list size[%zu] vs arg_list size[%zu]",
                 new_arg_list.size(), arg_list.size());
  MakeSureLoadStoreInnerestSameOrder(new_arg_list);
  GELOGI("After reorder, Arglist size is [%zu]", new_arg_list.size());
  for (size_t i = 0; i < new_arg_list.size(); i++) {
    auto &arg = new_arg_list[i];
    GELOGI("arg[%s], id is [%zu], order is [%zu]", arg->name.c_str(), i, arg->order);
  }
  return new_arg_list;
}

// 排序的入口函数
ge::Status ArgListReorder::SortArgList(vector<AttAxisPtr> &arg_list, vector<AttAxisPtr> &tiling_R_arg_list) {
  GE_ASSERT_TRUE(!arg_list.empty(), "arg list is empty");
  
  graph_ = ge::MakeShared<ArgPriorityGraph>(arg_list.size());
  GE_ASSERT_NOTNULL(graph_, "Create graph failed");
  GELOGI("Before reorder ArgList:");
  for (size_t i = 0; i < arg_list.size(); i++) {
    auto &arg = arg_list[i];
    axis_name_2_id_map_[arg->name] = i;
    GELOGI("arg[%s], id is [%zu]", arg->name.c_str(), i);
  }
  // 构建最终优先级连边图
  GE_ASSERT_SUCCESS(BuildArgListPriorityGraph(arg_list, false), "build arg list graph failed");
  std::vector<AttAxisPtr> new_arg_list = GetNewArgList(graph_->TopologicalSort(), arg_list);
  
  if (tiling_R_) {
    GELOGI("ReduceAxis BlockSplit, another reorder ArgList:");
    graph_ = ge::MakeShared<ArgPriorityGraph>(arg_list.size());
    GE_ASSERT_NOTNULL(graph_, "Create graph_ failed.");
    GE_ASSERT_SUCCESS(BuildArgListPriorityGraph(arg_list, true), "build tiling_R arg list graph failed");
    std::vector<AttAxisPtr> tiling_R_new_arg_list = GetNewArgList(graph_->TopologicalSort(), arg_list);
    tiling_R_arg_list = tiling_R_new_arg_list;
  }
  
  arg_list = new_arg_list;
  return ge::SUCCESS;
}
}  // namespace att

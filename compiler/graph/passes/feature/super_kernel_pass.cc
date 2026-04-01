/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/feature/super_kernel_pass.h"
#include <set>
#include <memory>
#include <string>

#include "framework/common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/omg_util/omg_util.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "runtime/mem.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/passes/pass_utils.h"
#include "graph/ge_context.h"
#include "common/checker.h"
#include "exe_graph/lowering/data_dependent_interpreter.h"

namespace ge {
namespace {
  const std::string scope_check_key= "strict-scope-check";
  constexpr size_t default_pair_size = 2U;
  const std::set<std::string> scope_check_valid_value{"bypass", "abort"};
  const std::string super_scope_key = "_super_kernel_scope";
  bool IsSendNode(const NodePtr node) {
    auto type = node->GetType();
    return ((type == SEND) || (type == SENDNOTIFY) || (type == "SendMem"));
  }
  bool IsRcvNode(const NodePtr node) {
    auto type = node->GetType();
    return ((type == RECV) || (type == RECVNOTIFY) || (type == "RecvMem"));
  }
  bool IsSendRcvNode(const NodePtr node) {
    return (IsSendNode(node) || IsRcvNode(node));
  }

  bool IsHcomOpSupportSk(const OpDesc* op_desc) {
    bool is_hccl_support_sk = false;
    (void)AttrUtils::GetBool(op_desc, "_hccl", is_hccl_support_sk);
    return is_hccl_support_sk;
  }
  bool IsIgnoreType(const NodePtr node) {
    return (IsSendRcvNode(node) ||
           (node->GetType() == ge::DATA) ||
           (node->GetType() == ge::VARIABLE) ||
           (node->GetType() == ge::CONSTANT) ||
           (node->GetType() == ge::CONSTANTOP) ||
           (node->GetType() == ge::CONSTPLACEHOLDER));
  }
}

Status SuperKernelPass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  GELOGI("SuperKernelPass start graph is [%s]", graph->GetName().c_str());
  auto root_graph = GraphUtils::FindRootGraph(graph);
  GE_ASSERT_NOTNULL(root_graph);
  if (root_graph->GetGraphUnknownFlag()) {
    GELOGI("SuperKernelPass only support static graph[%s]", root_graph->GetName().c_str());
    return SUCCESS;
  }
  // 先找出所有的scope的节点
  auto all_nodes = graph->GetAllNodes();
  std::vector<NodePtr> send_rcv_nodes;
  for (auto &node : all_nodes) {
    GE_ASSERT_NOTNULL(node);
    auto op_desc = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    const int64_t stream_id = op_desc->GetStreamId();
    ori_stream_ordered_nodes_[stream_id].emplace_back(node);
    size_t cur_pos = ori_stream_ordered_nodes_[stream_id].size() - 1;
    if (IsSendRcvNode(node)) {
      send_rcv_nodes.emplace_back(node);
    }
    if (IsIgnoreType(node)) {
      // 此处对于data,cost这类非计算节点，上层可能直接with scope会带入sk属性，对于这种节点内部直接忽略。
      // 所以此处直接删除_super_kernel_scope，避免对后续逻辑产生干扰。
      op_desc->DelAttr(super_scope_key);
      continue;
    }
    std::string super_scope_name;
    (void)AttrUtils::GetStr(op_desc, super_scope_key, super_scope_name);
    if (super_scope_name.empty()) {
      continue;
    }
    int64_t support = 0;
    (void)AttrUtils::GetInt(op_desc, "supportSuperKernel", support);
    /*
      背景：SIMT算子不支持取消融合，在GE处规避，待GE/SK check方案合入后删除
      临时规避方案：SIMT算子中添加local_memory_size属性，GE SK通过该属性识别SIMT算子并跳过
      正式方案：提供SK的融合前检查，方案设计中
    */
    int64_t local_memory_size = 0;
    (void)AttrUtils::GetInt(op_desc, "local_memory_size", local_memory_size);
    const bool is_simt_op = local_memory_size > 0;
    std::string super_kernel_options;
    (void)AttrUtils::GetStr(op_desc, ATTR_NAME_SUPER_KERNEL_OPTIONS, super_kernel_options);
    std::string check_val;
    GE_ASSERT_SUCCESS(ParseSuperKernelOptions(super_scope_name, super_kernel_options, check_val));
    const bool no_support_sk_fusion = ((support != 1) || (is_simt_op)) && !IsHcomOpSupportSk(op_desc);
    // can not delete _super_kernel_scope attr when strict-scope-check is not empty,
    // because strict-scope-check logic need this attr to verify
    const bool need_del_attr = no_support_sk_fusion && check_val.empty();
    // 此处不支持融合的算子直接删除属性是为了自动拆分时便于处理
    // 特别是CMO场景下，前段标记会把CMO和计算算子一起打上融合标记，CMO内部是单独分流的，但不支持融合
    // 如果此处不删除属性，会造成后续自动拆分时把CMO的拓扑id作为断开点，使得原本可以融合一个的sk（通过sk和外部的同步机制保证顺序）拆分成多个
    if (need_del_attr) {
      op_desc->DelAttr(super_scope_key);
      std::string unsupported_reason = no_support_sk_fusion ?
           "can not fusion, maybe it is tbe or tik operator, please replace to ascendc operator and specify super_kernel_scope" :
           "can not fusion, please check your super_kernel_scope";
      GEEVENT("find super kernel sub op %s(%s) %s, local_memory_size %ld, super_scope_name %s, stream_id %ld, "
          "cur_pos %zu, topo id %ld, so delete attr",
          op_desc->GetNamePtr(), op_desc->GetTypePtr(), unsupported_reason.c_str(), local_memory_size, super_scope_name.c_str(), stream_id,
          cur_pos, op_desc->GetId());
      ori_super_nodes_delete_id_[super_scope_name][stream_id].emplace_back(op_desc->GetId());
      continue;
    }
    GELOGI("find super kernel sub op %s(%s), super_scope_name %s, stream_id %ld, cur_pos %zu, super_kernel_options %s",
           op_desc->GetNamePtr(), op_desc->GetTypePtr(), super_scope_name.c_str(), stream_id,
           cur_pos, super_kernel_options.c_str());
    ori_super_nodes_[super_scope_name].emplace_back(node);
    ori_super_nodes_id_[super_scope_name][stream_id].emplace_back(cur_pos);
  }

  // 对每个scope进行校验，是否能融合，执行序列不连续的都不融合
  GE_ASSERT_SUCCESS(SelectFusionScope());
  if (ori_super_nodes_.empty()) {
    GELOGI("graph [%s] has no super kernel scope", graph->GetName().c_str());
    return SUCCESS;
  }
  // avoid mem event_id is repeated
  uint32_t cur_event_id = static_cast<uint32_t>(INT32_MAX / 2);
  // 对每个scope进行合并
  for (auto &super_ops_it : ori_super_nodes_) {
    SuperKernelScope kernel_scope;
    GE_ASSERT_SUCCESS(kernel_scope.Init(super_ops_it.first, super_ops_it.second, cur_event_id));
    GE_ASSERT_SUCCESS(kernel_scope.MergeSuperKernelsToSubgraph());
    cur_event_id += kernel_scope.GetScopeEventSize();
  }
  GE_ASSERT_SUCCESS(RefreshAllNodesTopoId(root_graph));
  GE_DUMP(graph, "AfterSkGraph");
  return ge::SUCCESS;
}

Status SuperKernelPass::ParseSuperKernelOptions(const std::string &super_kernel_scope,
                                                const std::string &super_kernel_options,
                                                std::string &check_val) {
  // key : super_kernel_options, value: a_opt=xxx:b_opt=xxx:c_opt=xxx
  const auto kernel_options = StringUtils::Split(super_kernel_options, ':');
  for (const auto &kernel_option : kernel_options) {
    const auto option_pair = StringUtils::Split(kernel_option, '=');
    if ((option_pair.size() == default_pair_size) && (option_pair[0] == scope_check_key)) {
      check_val = option_pair[1];
      const auto it = scope_check_valid_value.find(check_val);
      GE_ASSERT_TRUE(it != scope_check_valid_value.end(), "options %s, value %s is invalid, only support bypass or abort",
                     scope_check_key.c_str(), check_val.c_str());
      super_kernel_scope_options_[super_kernel_scope] = check_val;
    }
  }
  return SUCCESS;
}

Status SuperKernelPass::RefreshAllNodesTopoId(ge::ComputeGraphPtr root_graph) const {
  int64_t topo_id_refresh = 0;
  for (auto &seen_node : root_graph->GetAllNodesPtr()) {
    GE_ASSERT_NOTNULL(seen_node);
    auto seen_op_desc = seen_node->GetOpDesc();
    GE_ASSERT_NOTNULL(seen_op_desc);
    seen_op_desc->SetId(topo_id_refresh);
    GELOGD("set op %s(%s) id %ld", seen_op_desc->GetNamePtr(), seen_op_desc->GetTypePtr(), topo_id_refresh);
    ++topo_id_refresh;

    if (seen_op_desc->GetType() == "SuperKernel") {
      ComputeGraphPtr sk_sub_graph = nullptr;
      sk_sub_graph = seen_op_desc->TryGetExtAttr("_sk_sub_graph", sk_sub_graph);
      GE_ASSERT_NOTNULL(sk_sub_graph);
      for (auto &sub_seen_node : sk_sub_graph->GetAllNodesPtr()) {
        GE_ASSERT_NOTNULL(sub_seen_node);
        auto sub_seen_op_desc = sub_seen_node->GetOpDesc();
        GE_ASSERT_NOTNULL(sub_seen_op_desc);
        sub_seen_op_desc->SetId(topo_id_refresh);
        GELOGD("set sk sub op %s(%s) id %ld",
               sub_seen_op_desc->GetNamePtr(), sub_seen_op_desc->GetTypePtr(), topo_id_refresh);
        ++topo_id_refresh;
      }
    }
  }
  return SUCCESS;
}

/*
 * 假设名为scope的融合图拓扑id如下，其中打x的是检测出来断开的点id，同一竖线代表一条流：
 *        1         2
 *        5(x)      3
 *        6         4(x)
 *        8(x)      7
 *        10        9
 *  得到需要断开点的数组 : 0,4,5,8,11，其中0和11是额外补位的，便于处理
 *  自动拆分逻辑：在两个断开点之间的打融合标记的就放在一起融合，得出结果如下
 *  scope_split_1(1,2,3)  scope_split_2(6,7) scope_split_3(9,10)
 */

Status SuperKernelPass::AutomaticSplitScope(const std::set<std::string> &no_fusion_scope,
                                            std::map<std::string, std::vector<int64_t>> &scope_cut_id) {
  for (const auto &scope : no_fusion_scope) {
    const auto it = scope_cut_id.find(scope);
    if (it == scope_cut_id.end() || it->second.empty()) {
      continue;
    }

    GE_ASSERT_TRUE(ori_super_nodes_.find(scope) != ori_super_nodes_.end());
    auto &cut_id = scope_cut_id[scope];
    auto &sub_nodes= ori_super_nodes_[scope];
    GE_ASSERT_TRUE(!sub_nodes.empty());
    cut_id.insert(cut_id.begin(), (sub_nodes[0]->GetOpDesc()->GetId() - 1));
    cut_id.emplace_back(sub_nodes[sub_nodes.size() - 1]->GetOpDesc()->GetId() + 1);
    std::sort(cut_id.begin(), cut_id.end());
    for (size_t i = 0; i < (cut_id.size() - 1); ++i) {
      const int64_t begin_id = cut_id[i];
      const int64_t end_id = cut_id[i + 1];
      GE_ASSERT_TRUE((end_id >= begin_id), "%ld vs %ld", begin_id, end_id);
      GELOGI("try to judge scope %s cut id form %ld to %ld", scope.c_str(), begin_id, end_id);
      const std::string new_scope_name = scope + "_split_" + to_string(begin_id) + "_" + to_string(end_id);
      for (auto &sub_node : sub_nodes) {
        const int64_t cur_id = sub_node->GetOpDesc()->GetId();
        if ((cur_id > begin_id) && (cur_id < end_id)) {
          std::string super_scope_name;
          if (AttrUtils::GetStr(sub_node->GetOpDesc(), super_scope_key, super_scope_name)) {
            GE_ASSERT_TRUE(AttrUtils::SetStr(sub_node->GetOpDesc(), super_scope_key, new_scope_name));
            ori_super_nodes_[new_scope_name].emplace_back(sub_node);
            GELOGI("refresh node %s scope from %s to %s, id %ld, cut id from %ld to %ld", sub_node->GetNamePtr(),
                   scope.c_str(), new_scope_name.c_str(), cur_id, begin_id, end_id);
          }
        }
      }
    }
  }
  return SUCCESS;
}

Status SuperKernelPass::SelectFusionScope() {
  std::set<std::string> no_fusion_scope;
  std::map<std::string, std::vector<int64_t>> scope_cut_id;
  for (auto &super_ops_it : ori_super_nodes_id_) {
    GE_ASSERT_TRUE(!super_ops_it.second.empty());
    auto cur_scope_str = super_ops_it.first;

    // 校验执行序列
    auto &streams_id_nodes = super_ops_it.second;
    for (auto &stream_id_nodes : streams_id_nodes) {
      int64_t cur_stream_id = stream_id_nodes.first;
      // add delete attr node to cut id
      const auto &delete_scope_ids = ori_super_nodes_delete_id_[cur_scope_str][cur_stream_id];
      auto &single_scope_cut_ids = scope_cut_id[cur_scope_str];
      single_scope_cut_ids.insert(single_scope_cut_ids.end(), delete_scope_ids.begin(), delete_scope_ids.end());
      GE_ASSERT_TRUE(!stream_id_nodes.second.empty());
      size_t begin_id = stream_id_nodes.second[0];
      size_t cur_sub_nodes_size = stream_id_nodes.second.size();
      auto end_id = stream_id_nodes.second[cur_sub_nodes_size - 1];
      GELOGI("start to verify %s from %zu to %zu, cur_stream_id %ld, scope cut ids size is %zu",
             cur_scope_str.c_str(), begin_id, end_id, cur_stream_id, single_scope_cut_ids.size());
      for (size_t i = begin_id; i <= end_id; ++i) {
        auto it_ordered = ori_stream_ordered_nodes_.find(cur_stream_id);
        GE_ASSERT_TRUE(it_ordered != ori_stream_ordered_nodes_.end());
        auto cur_node = it_ordered->second.at(i);
        GE_ASSERT_NOTNULL(cur_node);
        if (IsIgnoreType(cur_node)) {
          continue;
        }
        std::string super_scope_name;
        (void)AttrUtils::GetStr(cur_node->GetOpDesc(), super_scope_key, super_scope_name);
        int64_t support = 0;
        (void)AttrUtils::GetInt(cur_node->GetOpDesc(), "supportSuperKernel", support);
        /*
          背景：SIMT算子不支持取消融合，在GE处规避，待GE/SK check方案合入后删除
          临时规避方案：SIMT算子中添加local_memory_size属性，GE SK通过该属性识别SIMT算子并跳过
          正式方案：提供SK的融合前检查，方案设计中
        */
        int64_t local_memory_size = 0;
        (void)AttrUtils::GetInt(cur_node->GetOpDesc(), "local_memory_size", local_memory_size);
        const bool is_simt_op = local_memory_size > 0;
        const bool no_support_sk_fusion = ((support != 1) || (is_simt_op)) && !IsHcomOpSupportSk(cur_node->GetOpDescBarePtr());
        std::string unsupported_reason = no_support_sk_fusion ?
            "can not fusion, maybe it is tbe or tik operator, please replace to ascendc operator and specify super_kernel_scope" :
            "can not fusion, please check your super_kernel_scope";
        const bool need_cut = (super_scope_name != cur_scope_str) || no_support_sk_fusion;
        if (need_cut) {
          const auto check_val = super_kernel_scope_options_[cur_scope_str];
          GEEVENT("node %s %s, local_memory_size %ld. stream id %ld index %zu, target scope %s, topo id is %ld.",
                  cur_node->GetNamePtr(), unsupported_reason.c_str(), local_memory_size, cur_stream_id, i,
                  cur_scope_str.c_str(), cur_node->GetOpDesc()->GetId());
          GE_ASSERT_TRUE((check_val != "abort"), "In abort check scene, node %s %s, target scope %s",
              cur_node->GetNamePtr(), unsupported_reason.c_str(), cur_scope_str.c_str());
          if (check_val == "bypass") {
            GELOGW("current node %s %s, target scope %s",
                   cur_node->GetNamePtr(), unsupported_reason.c_str(), cur_scope_str.c_str());
          } else {
            scope_cut_id[cur_scope_str].emplace_back(cur_node->GetOpDesc()->GetId());
          }
          no_fusion_scope.insert(cur_scope_str);
          continue;
        }
      }
    }
    if (!scope_cut_id[cur_scope_str].empty()) {
      no_fusion_scope.insert(cur_scope_str);
    }
  }
  GE_ASSERT_SUCCESS(AutomaticSplitScope(no_fusion_scope, scope_cut_id));

  // 删除不能融合的
  for (const auto &scope : no_fusion_scope) {
    (void)ori_super_nodes_.erase(scope);
    (void)ori_super_nodes_id_.erase(scope);
  }
  return SUCCESS;
}

Status SuperKernelScope::Init(const std::string &name, const std::vector<NodePtr> &sk_nodes, uint32_t event_begin_id) {
  super_nodes_ = sk_nodes;
  event_begin_id_ = event_begin_id;
  for (const auto &sub_node : super_nodes_) {
    super_nodes_set_.insert(sub_node->GetName());
  }
  super_scope_name_ = name;
  origin_graph_ = super_nodes_[0]->GetOwnerComputeGraph();
  GELOGI("start to process scope %s, event begin id is %u, sk sub nodes size %zu",
         super_scope_name_.c_str(), event_begin_id_, sk_nodes.size());
  GE_ASSERT_NOTNULL(origin_graph_);
  auto all_nodes = origin_graph_->GetAllNodes();
  std::vector<NodePtr> send_rcv_nodes;
  std::map<int64_t, std::vector<NodePtr>> stream_super_nodes;
  for (auto &node : all_nodes) {
    GE_ASSERT_NOTNULL(node);
    auto op_desc = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    const int64_t stream_id = op_desc->GetStreamId();
    stream_ordered_nodes_[stream_id].emplace_back(node);
    size_t cur_pos = stream_ordered_nodes_[stream_id].size() - 1;
    std::string super_scope_name;
    (void)AttrUtils::GetStr(op_desc, super_scope_key, super_scope_name);
    if (super_scope_name == super_scope_name_) {
      GELOGI("find current super kernel sub op %s(%s), super_scope_name %s, stream_id %ld, cur_pos %zu",
             op_desc->GetNamePtr(), op_desc->GetTypePtr(), super_scope_name.c_str(), stream_id, cur_pos);
      super_nodes_id_[stream_id].emplace_back(cur_pos);
      stream_super_nodes[stream_id].emplace_back(node);
    }
    if (IsSendRcvNode(node)) {
      send_rcv_nodes.emplace_back(node);
    }
    if (op_desc->GetType() == "SuperKernel") {
      ComputeGraphPtr sk_sub_graph = nullptr;
      sk_sub_graph = op_desc->TryGetExtAttr("_sk_sub_graph", sk_sub_graph);
      GE_ASSERT_NOTNULL(sk_sub_graph);
      for (auto &sub_seen_node : sk_sub_graph->GetAllNodes()) {
        GE_ASSERT_NOTNULL(sub_seen_node);
        if (IsSendRcvNode(sub_seen_node)) {
          send_rcv_nodes.emplace_back(sub_seen_node);
        }
      }
    }
  }
  SelectSkStreamId(stream_super_nodes);
  GE_ASSERT_SUCCESS(UpdateWholeSendRcvMap(send_rcv_nodes));
  return SUCCESS;
}

void SuperKernelScope::SelectSkStreamId(const std::map<int64_t, std::vector<NodePtr>> &stream_super_nodes) {
  scope_stream_id_ = super_nodes_[0]->GetOpDesc()->GetStreamId();
  // if sk first sub node is hcom node, sk will execute on hccl stream id, it leads to many event, so exclude this scene
  if (IsHcomOpSupportSk(super_nodes_[0]->GetOpDescBarePtr())) {
    for (auto &ele : stream_super_nodes) {
      if (!IsHcomOpSupportSk(ele.second[0]->GetOpDescBarePtr())) {
        scope_stream_id_ = ele.first;
        GELOGI("select stream id %ld from op %s for super scope %s", scope_stream_id_, ele.second[0]->GetNamePtr(),
               super_scope_name_.c_str());
        break;
      }
    }
  }
  GELOGI("select stream id %ld for super scope %s", scope_stream_id_, super_scope_name_.c_str());
  return;
}

Status SuperKernelScope::RecordSendInfo(const NodePtr &send_node) {
  uint32_t event_id = 0;
  std::string event_key = (send_node->GetType() == SENDNOTIFY) ? SEND_ATTR_NOTIFY_ID : SEND_ATTR_EVENT_ID;
  GE_ASSERT_TRUE(AttrUtils::GetInt(send_node->GetOpDesc(), event_key, event_id),
                 "%s can not get event id", send_node->GetNamePtr());
  GELOGI("start to record send info from %s, event id %u", send_node->GetNamePtr(), event_id);
  auto in_control_anchor = send_node->GetInControlAnchor();
  GE_ASSERT_NOTNULL(in_control_anchor);
  auto out_control_anchors = in_control_anchor->GetPeerOutControlAnchors();
  GE_ASSERT_TRUE(out_control_anchors.size() == 1);
  GE_ASSERT_NOTNULL(out_control_anchors.at(0));
  auto src_node = out_control_anchors.at(0)->GetOwnerNode();
  GE_ASSERT_NOTNULL(src_node);
  const auto send_stream_id = send_node->GetOpDescBarePtr()->GetStreamId();
  send_nodes_map_[src_node].emplace_back(std::make_pair(nullptr, event_id));
  event_nodes_list_[event_id].src_node = src_node;
  event_nodes_list_[event_id].src_node_name = src_node->GetName();
  event_nodes_list_[event_id].send_node = send_node;
  event_nodes_list_[event_id].send_node_name = send_node->GetName();
  event_nodes_list_[event_id].event_id = event_id;
  event_nodes_list_[event_id].send_stream_id = send_stream_id;
  GELOGI("record send info from %s, event id %u, src_node %s, event_id %u, send_stream_id %ld",
         send_node->GetNamePtr(), event_id, src_node->GetNamePtr(), event_id, send_stream_id);
  return SUCCESS;
}

Status SuperKernelScope::RecordRcvInfo(const NodePtr &rcv_node) {
  uint32_t event_id = 0;
  std::string event_key = (rcv_node->GetType() == RECVNOTIFY) ? RECV_ATTR_NOTIFY_ID : RECV_ATTR_EVENT_ID;
  GE_ASSERT_TRUE(AttrUtils::GetInt(rcv_node->GetOpDesc(), event_key, event_id),
                 "%s can not get event id", rcv_node->GetNamePtr());
  GELOGI("start to record recv info from %s, event id %u", rcv_node->GetNamePtr(), event_id);
  auto out_control_anchor = rcv_node->GetOutControlAnchor();
  GE_ASSERT_NOTNULL(out_control_anchor);
  auto in_control_anchors = out_control_anchor->GetPeerInControlAnchors();
  GE_ASSERT_TRUE(in_control_anchors.size() == 1);
  GE_ASSERT_NOTNULL(in_control_anchors.at(0));
  auto dst_node = in_control_anchors.at(0)->GetOwnerNode();
  GE_ASSERT_NOTNULL(dst_node);
  const auto rcv_stream_id = rcv_node->GetOpDescBarePtr()->GetStreamId();
  rcv_nodes_map_[dst_node].emplace_back(std::make_pair(nullptr, event_id));
  event_nodes_list_[event_id].dst_node = dst_node;
  event_nodes_list_[event_id].dst_node_name = dst_node->GetName();
  event_nodes_list_[event_id].rcv_node = rcv_node;
  event_nodes_list_[event_id].rcv_node_name = rcv_node->GetName();
  event_nodes_list_[event_id].event_id = event_id;
  event_nodes_list_[event_id].rcv_stream_id = rcv_stream_id;
  GELOGI("record recv info from %s, event id %u, dst_node %s, event_id %u, rcv_stream_id is %ld",
         rcv_node->GetNamePtr(), event_id, dst_node->GetNamePtr(), event_id, rcv_stream_id);
  return SUCCESS;
}

Status SuperKernelScope::UpdateWholeSendRcvMap(const std::vector<NodePtr> &send_rcv_nodes) {
  for (const auto &ele : send_rcv_nodes) {
    if (IsSendNode(ele)) {
      GE_ASSERT_SUCCESS(RecordSendInfo(ele));
    }
    if (IsRcvNode(ele)) {
      GE_ASSERT_SUCCESS(RecordRcvInfo(ele));
    }
  }
  for (auto &send_ele : send_nodes_map_) {
    for (auto &send_pair : send_ele.second) {
      const uint32_t event_id = send_pair.second;
      auto it = event_nodes_list_.find(event_id);
      GE_ASSERT_TRUE(it != event_nodes_list_.end(), "%u can not find in event map for src_node %s",
                     event_id, send_ele.first->GetNamePtr());
      GE_ASSERT_NOTNULL(it->second.src_node, "event id %u src_node can not be null for src_node %s",
                        event_id, send_ele.first->GetNamePtr());
      GE_ASSERT_NOTNULL(it->second.send_node, "event id %u send_node can not be null for src_node %s",
                        event_id, send_ele.first->GetNamePtr());
      GE_ASSERT_NOTNULL(it->second.dst_node, "event id %u dst_node can not be null for src_node %s",
                        event_id, send_ele.first->GetNamePtr());
      GE_ASSERT_NOTNULL(it->second.rcv_node, "event id %u rcv_node can not be null for src_node %s",
                        event_id, send_ele.first->GetNamePtr());
      send_pair.first = it->second.dst_node;
      GELOGI("get src_node %s, event id %u src_node %s, send_node %s, rcv_node %s, dst_node %s",
             send_ele.first->GetNamePtr(), event_id, it->second.src_node->GetNamePtr(),
             it->second.send_node->GetNamePtr(), it->second.rcv_node->GetNamePtr(),
             it->second.dst_node->GetNamePtr());
    }
  }
  for (auto &rcv_ele : rcv_nodes_map_) {
    for (auto &rcv_pair : rcv_ele.second) {
      const uint32_t event_id = rcv_pair.second;
      auto it = event_nodes_list_.find(event_id);
      GE_ASSERT_TRUE(it != event_nodes_list_.end(), "%u can not find in event map for dst_node %s",
                     event_id, rcv_ele.first->GetNamePtr());
      GE_ASSERT_NOTNULL(it->second.src_node, "event id %u src_node can not be null for dst_node %s",
                        event_id, rcv_ele.first->GetNamePtr());
      GE_ASSERT_NOTNULL(it->second.send_node, "event id %u send_node can not be null for dst_node %s",
                        event_id, rcv_ele.first->GetNamePtr());
      GE_ASSERT_NOTNULL(it->second.dst_node, "event id %u dst_node can not be null for dst_node %s",
                        event_id, rcv_ele.first->GetNamePtr());
      GE_ASSERT_NOTNULL(it->second.rcv_node, "event id %u rcv_node can not be null for dst_node %s",
                        event_id, rcv_ele.first->GetNamePtr());
      rcv_pair.first = it->second.src_node;
      GELOGI("get dst_node %s, event id %u src_node %s, send_node %s, rcv_node %s, dst_node %s",
             rcv_ele.first->GetNamePtr(), event_id, it->second.src_node->GetNamePtr(),
             it->second.send_node->GetNamePtr(), it->second.rcv_node->GetNamePtr(),
             it->second.dst_node->GetNamePtr());
    }
  }
  return SUCCESS;
}

Status SuperKernelScope::GetSuperNodesIoInfo() {
  // 得到所有节点的输入输出连边关系
  for (const auto &node : super_nodes_) {
    NodeIoInfo cur_node_info;
    cur_node_info.node_name = node->GetName();
    cur_node_info.cur_node = node;
    // 获取输入信息
    for (const auto &in_data_anchor : node->GetAllInDataAnchorsPtr()) {
      InputNodeInfo cur_input_node_info;
      GE_ASSERT_NOTNULL(in_data_anchor);
      cur_input_node_info.in_data_anchor_id = in_data_anchor->GetIdx();
      auto input_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
      if (input_out_data_anchor == nullptr) {
        continue;
      }
      cur_input_node_info.out_data_anchor_id = input_out_data_anchor->GetIdx();
      cur_input_node_info.cur_node = input_out_data_anchor->GetOwnerNode();
      GE_ASSERT_NOTNULL(cur_input_node_info.cur_node);
      cur_input_node_info.node_name = input_out_data_anchor->GetOwnerNode()->GetName();

      GELOGI("super node %s input %zu link node %s output %zu",
        node->GetNamePtr(), cur_input_node_info.in_data_anchor_id,
        cur_input_node_info.node_name.c_str(), cur_input_node_info.out_data_anchor_id);
      cur_node_info.input_nodes_info.emplace_back(cur_input_node_info);
    }

    for (const auto &out_data_anchor : node->GetAllOutDataAnchorsPtr()) {
      std::vector<OutputNodeInfo> cur_output_nodes_info;
      GE_ASSERT_NOTNULL(out_data_anchor);
      for (const auto &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        if (peer_in_anchor == nullptr) {
          continue;
        }
        OutputNodeInfo cur_out_node_info;
        cur_out_node_info.out_data_anchor_id = out_data_anchor->GetIdx();
        cur_out_node_info.in_data_anchor_id = peer_in_anchor->GetIdx();
        auto out_node = peer_in_anchor->GetOwnerNode();
        GE_ASSERT_NOTNULL(out_node);
        cur_out_node_info.node_name = out_node->GetName();
        cur_out_node_info.cur_node = out_node;
        cur_output_nodes_info.emplace_back(cur_out_node_info);
        GELOGI("super node %s output %zu link node %s input %zu",
          node->GetNamePtr(), cur_out_node_info.out_data_anchor_id,
          cur_out_node_info.node_name.c_str(), cur_out_node_info.in_data_anchor_id);
      }
      cur_node_info.output_nodes_info.emplace_back(cur_output_nodes_info);
    }
    GE_ASSERT_SUCCESS(GetSuperNodesEventInfo(node, cur_node_info));
    super_nodes_info_[node->GetName()] = cur_node_info;
  }
  GE_ASSERT_SUCCESS(GetSkBoundaryEventInfo());
  return SUCCESS;
}

Status SuperKernelScope::GetSkBoundaryEventInfo() {
  for (auto &it_stream_ele : super_nodes_id_) {
    const int64_t cur_stream_id = it_stream_ele.first;
    if (cur_stream_id == scope_stream_id_) {
      continue;
    }
    GELOGI("find stream %ld should add event info", cur_stream_id);
    auto &stream_sub_nodes_id = it_stream_ele.second;
    GE_ASSERT_TRUE(!stream_sub_nodes_id.empty());
    auto begin_id = stream_sub_nodes_id[0];
    auto end_id = stream_sub_nodes_id[stream_sub_nodes_id.size() - 1];
    auto it_stream_ordered_nodes = stream_ordered_nodes_.find(cur_stream_id);
    GE_ASSERT_TRUE(it_stream_ordered_nodes != stream_ordered_nodes_.end());
    auto &tmp_stream_ordered_nodes = it_stream_ordered_nodes->second;
    GE_ASSERT_TRUE(tmp_stream_ordered_nodes.size() > begin_id);
    GE_ASSERT_TRUE(tmp_stream_ordered_nodes.size() > end_id);
    first_other_stm_sub_nodes_.insert(tmp_stream_ordered_nodes[begin_id]->GetName());
    GELOGI("find stream %ld should add event info, begin_id is %zu name %s,end_id is %zu, name %s", cur_stream_id, begin_id,
           tmp_stream_ordered_nodes[begin_id]->GetNamePtr(), end_id, tmp_stream_ordered_nodes[end_id]->GetNamePtr());
    {
      NodePtr src_node;
      NodePtr dst_node;
      for (size_t i = (begin_id - 1); i < begin_id; --i) {
        if (!IsSendRcvNode(tmp_stream_ordered_nodes[i])) {
          src_node = tmp_stream_ordered_nodes[i];
          break;
        }
      }
      for (size_t i = begin_id; (i <= end_id); ++i) {
        if (!IsSendRcvNode(tmp_stream_ordered_nodes[i])) {
          dst_node = tmp_stream_ordered_nodes[i];
          break;
        }
      }
      if ((src_node != nullptr) && (dst_node != nullptr)) {
        super_nodes_info_[dst_node->GetName()].rcv_nodes_info.push_back(
            {src_node, src_node->GetName(), nullptr, "",
            dst_node, dst_node->GetName(), nullptr, "", 0, cur_stream_id, scope_stream_id_});
        GELOGI("find stream id %lu top add event to src_node %s, dst_node %s",
               cur_stream_id, src_node->GetNamePtr(), dst_node->GetNamePtr());
      }
    }

    {
      NodePtr src_node;
      NodePtr dst_node;
      for (size_t i = end_id; ((i >= begin_id) && (i <= end_id)); --i) {
        if (!IsSendRcvNode(tmp_stream_ordered_nodes[i])) {
          src_node = tmp_stream_ordered_nodes[i];
          break;
        }
      }
      for (size_t i = end_id + 1; i < tmp_stream_ordered_nodes.size(); ++i) {
        if (!IsSendRcvNode(tmp_stream_ordered_nodes[i])) {
          dst_node = tmp_stream_ordered_nodes[i];
          break;
        }
      }
      if ((src_node != nullptr) && (dst_node != nullptr)) {
        super_nodes_info_[src_node->GetName()].send_nodes_info.push_back(
            {src_node, src_node->GetName(), nullptr, "",
            dst_node, dst_node->GetName(), nullptr, "", 0, scope_stream_id_, cur_stream_id});
        GELOGI("find stream id %lu bottom add event to src_node %s, dst_node %s",
               cur_stream_id, src_node->GetNamePtr(), dst_node->GetNamePtr());
      }
    }
  }
  return SUCCESS;
}

Status SuperKernelScope::GetSuperNodesEventInfo(const NodePtr &cur_node, NodeIoInfo &cur_node_info) {
  auto it_send = send_nodes_map_.find(cur_node);
  GELOGI("start to GetSuperNodesEventInfo %s", cur_node->GetNamePtr());
  if (it_send != send_nodes_map_.end()) {
    GELOGI("start to record send info %s", cur_node->GetNamePtr());
    std::vector<EventNodeInfo> send_nodes_info;
    for (auto &send_ele : it_send->second) {
      uint32_t event_id = send_ele.second;
      auto it_event = event_nodes_list_.find(event_id);
      GE_ASSERT_TRUE(it_event != event_nodes_list_.end());
      send_nodes_info.emplace_back(it_event->second);
      GELOGI("find src_node %s, event id %u src_node %s, send_node %s, rcv_node %s, dst_node %s",
             cur_node->GetNamePtr(), event_id, it_event->second.src_node->GetNamePtr(),
             it_event->second.send_node->GetNamePtr(), it_event->second.rcv_node->GetNamePtr(),
             it_event->second.dst_node->GetNamePtr());
    }
    cur_node_info.send_nodes_info = send_nodes_info;
  }
  auto it_rcv = rcv_nodes_map_.find(cur_node);
  if (it_rcv != rcv_nodes_map_.end()) {
    GELOGI("start to record rcv info %s", cur_node->GetNamePtr());
    std::vector<EventNodeInfo> rcv_nodes_info;
    for (auto &rcv_ele : it_rcv->second) {
      uint32_t event_id = rcv_ele.second;
      auto it_event = event_nodes_list_.find(event_id);
      GE_ASSERT_TRUE(it_event != event_nodes_list_.end());
      rcv_nodes_info.emplace_back(it_event->second);
      GELOGI("find dst_node %s, event id %u src_node %s, send_node %s, rcv_node %s, dst_node %s",
             cur_node->GetNamePtr(), event_id, it_event->second.src_node->GetNamePtr(),
             it_event->second.send_node->GetNamePtr(), it_event->second.rcv_node->GetNamePtr(),
             it_event->second.dst_node->GetNamePtr());
    }
    cur_node_info.rcv_nodes_info = rcv_nodes_info;
  }
  return SUCCESS;
}

Status SuperKernelScope::GetSkNodeLinkInfo() {
  std::set<std::string> merge_node_input_set;
  std::set<std::string> merge_node_output_set;
  for (auto &super_node_info_it : super_nodes_info_) {
    for (const auto &input_node_info : super_node_info_it.second.input_nodes_info) {
      if (super_nodes_set_.find(input_node_info.node_name) == super_nodes_set_.end()) {
        merge_node_input_vec_.emplace_back(
            std::make_pair(input_node_info.cur_node, input_node_info.out_data_anchor_id));
        merge_node_input_set.insert(input_node_info.cur_node->GetName());
        GELOGI("find node %s output %zu should link merge node",
          input_node_info.cur_node->GetNamePtr(), input_node_info.out_data_anchor_id);
      }
    }
    for (const auto &output_nodes_info : super_node_info_it.second.output_nodes_info) {
      std::vector<std::pair<NodePtr, size_t>> tmp_out_node_info;
      for (const auto &output_node_info : output_nodes_info) {
        if (super_nodes_set_.find(output_node_info.node_name) == super_nodes_set_.end()) {
          tmp_out_node_info.emplace_back(std::make_pair(output_node_info.cur_node, output_node_info.in_data_anchor_id));
          merge_node_output_set.insert(output_node_info.cur_node->GetName());
          GELOGI("find merge node should link node %s input %zu, current linked node %s",
            output_node_info.cur_node->GetNamePtr(), output_node_info.in_data_anchor_id,
            super_node_info_it.second.node_name.c_str());
        }
      }
      if (tmp_out_node_info.empty()) {
        continue;
      }
      merge_node_output_vec_.emplace_back(tmp_out_node_info);
      size_t parent_index = merge_node_output_vec_.size() - 1;
      size_t out_data_anchor_id = output_nodes_info[0].out_data_anchor_id;
      GELOGI("find node %s is sub_graph output node out_index %zu, parent_index %zu",
              super_node_info_it.second.node_name.c_str(), out_data_anchor_id, parent_index);
      out_nodes_order_[super_node_info_it.second.node_name].emplace_back(std::make_pair(parent_index, out_data_anchor_id));
    }
  }
  return SUCCESS;
}

NodePtr SuperKernelScope::ConstructSkNode() {
  std::string super_node_name = "sk_" + super_scope_name_ + "_start_" + super_nodes_[0]->GetName() + "_end_" +
                                super_nodes_[super_nodes_.size() - 1]->GetName();
  OpDescBuilder op_builder(super_node_name, "SuperKernel");
  uint32_t input_num = merge_node_input_vec_.size();
  op_builder.AddDynamicInput("args", input_num);
  uint32_t output_num = merge_node_output_vec_.size();
  op_builder.AddDynamicOutput("output", output_num);
  auto super_op_desc = op_builder.Build();
  GE_ASSERT_NOTNULL(super_op_desc);
  // set some aicore op attr
  super_op_desc->SetOpKernelLibName("AIcoreEngine");
  int64_t imply_type = 6;
  AttrUtils::SetInt(super_op_desc, "_fe_imply_type", imply_type);
  AttrUtils::SetInt(super_op_desc, "imply_type", 1);
  super_op_desc->SetId(super_nodes_[0]->GetOpDesc()->GetId());
  // should set super kernel stream id as this pass is after stream allocator
  super_op_desc->SetStreamId(scope_stream_id_);
  // 需要使用insertnode，插在super_ops第一个后面，super_ops后边会删除，这样保证superkernel节点在图中的顺序
  return origin_graph_->InsertNode(super_nodes_[0], super_op_desc);
}

Status SuperKernelScope::ConstructSubgraph() {
  auto super_op_desc = super_node_->GetOpDesc();
  static size_t name_index = 0;
  std::string sub_graph_name = "super_kernel_sub_graph" + to_string(name_index);
  sub_graph_ = MakeShared<ComputeGraph>(sub_graph_name);
  GE_ASSERT_NOTNULL(sub_graph_);
  sub_graph_->SetParentNode(super_node_);
  GELOGI("construct super kernel sub graph %s successfully", sub_graph_->GetName().c_str());
  return SUCCESS;
}

Status SuperKernelScope::LinkSkInputNode() {
  // 连接父节点输入并更新newNodeMap_
  auto super_op_desc = super_node_->GetOpDesc();
  std::vector<int64_t> v_input_offset;
  std::vector<bool> is_input_const;
  std::vector<int64_t> v_memory_type;
  for (size_t id = 0; id < merge_node_input_vec_.size(); ++id) {
    auto in_node = merge_node_input_vec_[id].first;
    auto output_idx = merge_node_input_vec_[id].second;
    auto in_node_output_tensordesc = in_node->GetOpDesc()->GetOutputDesc(output_idx);
    super_op_desc->UpdateInputDesc(id, in_node_output_tensordesc);
    GELOGI("link in node %s %zu to node %s %zu",
           in_node->GetNamePtr(), output_idx, super_node_->GetNamePtr(), id);
    GE_ASSERT_SUCCESS(GraphUtils::AddEdge(in_node->GetOutDataAnchor(output_idx), super_node_->GetInDataAnchor(id)));
    std::string data_node_name = sub_graph_->GetName() + "_innerdata_" + to_string(id);
    auto data_op_desc = MakeShared<OpDesc>(data_node_name, ge::DATA);
    GE_ASSERT_NOTNULL(data_op_desc);
    GE_ASSERT_SUCCESS(data_op_desc->AddInputDesc(in_node_output_tensordesc));
    GE_ASSERT_SUCCESS(data_op_desc->AddOutputDesc(in_node_output_tensordesc));
    GE_ASSERT_TRUE(AttrUtils::SetInt(data_op_desc, ATTR_NAME_PARENT_NODE_INDEX, id));
    auto inner_data_node = sub_graph_->AddNode(data_op_desc);
    GE_ASSERT_NOTNULL(inner_data_node);
    // outside in node may have multiple outputs, lead to multiple data nodes in subgraph, so key add _out_id
    newNodeMap_[in_node->GetName() + "_out_" + to_string(output_idx)] = inner_data_node;
    bool is_const = PassUtils::IsConstant(in_node);
    is_input_const.emplace_back(is_const);
    const std::vector<int64_t> v_output_offset = in_node->GetOpDesc()->GetOutputOffset();
    std::vector<int64_t> out_memory_type;
    (void)AttrUtils::GetListInt(in_node->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, out_memory_type);
    if (output_idx < v_output_offset.size()) {
      v_input_offset.emplace_back(v_output_offset[output_idx]);
    }
    if (output_idx < out_memory_type.size()) {
      v_memory_type.emplace_back(out_memory_type[output_idx]);
    }
    GELOGI("mapping origin node %s inner node %s, id %zu, output offset %zu, mem_type %zu",
           in_node->GetNamePtr(), inner_data_node->GetNamePtr(), output_idx,
           v_output_offset.size(), out_memory_type.size());
  }
  super_op_desc->SetInputOffset(v_input_offset);
  super_op_desc->SetIsInputConst(is_input_const);
  if (!v_memory_type.empty()) {
    GE_ASSERT_TRUE(AttrUtils::SetListInt(super_op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type));
  }
  GELOGI("sk %s has input offset %zu, mem_type_list %zu, is_input_const %zu", super_op_desc->GetNamePtr(),
         v_input_offset.size(), v_memory_type.size(), is_input_const.size());
  return SUCCESS;
}

Status SuperKernelScope::LinkSkOutputNode(NodePtr &inner_netoutput_node) {
  std::vector<int64_t> v_output_offset;
  std::vector<int64_t> v_memory_type;
  for (size_t i = 0; i < merge_node_output_vec_.size(); ++i) {
    for (auto &out_lin_node_index : merge_node_output_vec_[i]) {
      auto dst_node = out_lin_node_index.first;
      auto dst_in_data_anchor_id = out_lin_node_index.second;
      super_node_->GetOpDesc()->UpdateOutputDesc(i, dst_node->GetOpDesc()->GetInputDesc(dst_in_data_anchor_id));
      GELOGI("link input node %s %zu to node %s %zu",
             super_node_->GetNamePtr(), i, dst_node->GetNamePtr(), dst_in_data_anchor_id);
      GE_ASSERT_SUCCESS(GraphUtils::AddEdge(super_node_->GetOutDataAnchor(i),
                                            dst_node->GetInDataAnchor(dst_in_data_anchor_id)));
      inner_netoutput_node->GetOpDesc()->UpdateInputDesc(i, dst_node->GetOpDesc()->GetInputDesc(dst_in_data_anchor_id));
      newNodeMap_[dst_node->GetName() + "_in_" + to_string(dst_in_data_anchor_id)] = inner_netoutput_node;
      GELOGI("mapping origin node %s inner node %s", dst_node->GetNamePtr(), inner_netoutput_node->GetNamePtr());
      const std::vector<int64_t> v_input_offset = dst_node->GetOpDesc()->GetInputOffset();
      size_t real_input_cnt = 0UL;
      std::vector<int64_t> in_memory_type;
      (void)AttrUtils::GetListInt(dst_node->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, in_memory_type);
      for (size_t j = 0U; j < dst_node->GetOpDesc()->GetAllInputsSize(); ++j) {
        const GeTensorDescPtr tensor_desc = dst_node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(j));
        if (tensor_desc != nullptr) {
          GELOGI("dst node %s input offset %zu, mem type %zu, id %zu, real_input_cnt %zu, anchor id %zu",
                 dst_node->GetNamePtr(), v_input_offset.size(), in_memory_type.size(), j,
                 real_input_cnt, dst_in_data_anchor_id);
          if (real_input_cnt < v_input_offset.size() && (dst_in_data_anchor_id == j)) {
            v_output_offset.emplace_back(v_input_offset[real_input_cnt]);
          }
          if (real_input_cnt < in_memory_type.size() && (dst_in_data_anchor_id == j)) {
            v_memory_type.emplace_back(in_memory_type[real_input_cnt]);
          }
          real_input_cnt++;
        }
      }
    }
  }
  super_node_->GetOpDesc()->SetOutputOffset(v_output_offset);
  if (!v_memory_type.empty()) {
    GE_ASSERT_TRUE(AttrUtils::SetListInt(super_node_->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type));
  }
  GELOGI("sk %s has output offset %zu, mem_type_list %zu", super_node_->GetNamePtr(),
         v_output_offset.size(), v_memory_type.size());
  return SUCCESS;
}

Status SuperKernelScope::InsertSrcNode2SendMem(const NodePtr &src_node, const uint32_t event_id,
                                               const int64_t send_stream_id, const bool is_mem_event) {
  GE_ASSERT_NOTNULL(src_node);
  GE_CHECK_NOTNULL(src_node->GetOpDesc());
  std::string send_node_name = "sk_send_" + std::to_string(event_id);
  std::string send_type = is_mem_event ? "SendMem" : SEND;
  OpDescPtr op_desc_ptr = MakeShared<OpDesc>(send_node_name, send_type);
  GE_CHECK_NOTNULL(op_desc_ptr);

  op_desc_ptr->SetStreamId(send_stream_id);
  op_desc_ptr->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");
  GE_ASSERT_TRUE(AttrUtils::SetInt(op_desc_ptr, SEND_ATTR_EVENT_ID, event_id));
  (void)AttrUtils::SetListStr(op_desc_ptr, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES,
                              std::move(std::vector<std::string>()));
  auto graph = src_node->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(graph);
  NodePtr send_node = graph->InsertNode(src_node, op_desc_ptr);
  GE_CHECK_NOTNULL(send_node);
  GE_CHECK_NOTNULL(send_node->GetInControlAnchor());
  GE_ASSERT_SUCCESS(GraphUtils::AddEdge(src_node->GetOutControlAnchor(), send_node->GetInControlAnchor()),
                    "Add edge from node %s to node %s failed", src_node->GetNamePtr(), send_node->GetNamePtr());
  GE_ASSERT_SUCCESS(RefreshSendList(src_node, event_id, false));
  GELOGI("Insert send event node: %s event id %u after node: %s with stream: %ld",
         send_node->GetNamePtr(), event_id, src_node->GetNamePtr(), send_stream_id);

  return SUCCESS;
}

Status SuperKernelScope::RefreshSendList(const NodePtr src_node, const uint32_t event_id, const bool just_refresh) {
  std::vector<uint32_t> sk_send_event_ids;
  std::vector<uint32_t> sk_send_event_ids_newest;
  (void)AttrUtils::GetListInt(src_node->GetOpDesc(), "_sk_send_event_ids", sk_send_event_ids);
  for (const auto &ele : sk_send_event_ids) {
    if (delete_event_id_set_.find(ele) != delete_event_id_set_.end()) {
      GELOGI("event id %u is delete, no need to insert to _sk_send_event_ids", ele);
      continue;
    }
    sk_send_event_ids_newest.emplace_back(ele);
  }
  if (!just_refresh) {
    sk_send_event_ids_newest.emplace_back(event_id);
  }
  GE_ASSERT_TRUE(AttrUtils::SetListInt(src_node->GetOpDesc(), "_sk_send_event_ids", sk_send_event_ids_newest));
  GELOGI("src node: %s event id %u sk_send_event_ids_newest size is %zu",
         src_node->GetNamePtr(), event_id, sk_send_event_ids_newest.size());
  return SUCCESS;
}

Status SuperKernelScope::RefreshRcvList(const NodePtr dst_node, const uint32_t event_id, const bool just_refresh) {
  std::vector<uint32_t> sk_rcv_event_ids;
  std::vector<uint32_t> sk_rcv_event_ids_newest;
  (void)AttrUtils::GetListInt(dst_node->GetOpDesc(), "_sk_rcv_event_ids", sk_rcv_event_ids);
  for (const auto &ele : sk_rcv_event_ids) {
    if (delete_event_id_set_.find(ele) != delete_event_id_set_.end()) {
      GELOGI("event id %u is delete, no need to insert to _sk_send_event_ids", ele);
      continue;
    }
    sk_rcv_event_ids_newest.emplace_back(ele);
  }
  if (!just_refresh) {
    sk_rcv_event_ids_newest.emplace_back(event_id);
  }
  GE_ASSERT_TRUE(AttrUtils::SetListInt(dst_node->GetOpDesc(), "_sk_rcv_event_ids", sk_rcv_event_ids_newest));
  GELOGI("src dst_node: %s event id %u sk_rcv_event_ids_newest size is %zu",
         dst_node->GetNamePtr(), event_id, sk_rcv_event_ids_newest.size());
  return SUCCESS;
}

Status SuperKernelScope::InsertRecvMem2DstNode(const NodePtr &dst_node, const uint32_t event_id,
                                               const int64_t rcv_stream_id, const bool is_mem_event) {
  GE_ASSERT_NOTNULL(dst_node);
  GE_CHECK_NOTNULL(dst_node->GetOpDesc());
  std::string rcv_node_name = "sk_rcv_" + std::to_string(event_id);
  std::string rcv_type = is_mem_event ? "RecvMem" : RECV;
  OpDescPtr op_desc_ptr = MakeShared<OpDesc>(rcv_node_name, rcv_type);
  GE_CHECK_NOTNULL(op_desc_ptr);

  op_desc_ptr->SetStreamId(rcv_stream_id);
  op_desc_ptr->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");
  GE_ASSERT_TRUE(AttrUtils::SetInt(op_desc_ptr, RECV_ATTR_EVENT_ID, event_id));
  (void)AttrUtils::SetListStr(op_desc_ptr, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES,
                              std::move(std::vector<std::string>()));
  auto graph = dst_node->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(graph);
  NodePtr recv_node = graph->InsertNodeBefore(dst_node, op_desc_ptr);
  GE_CHECK_NOTNULL(recv_node);
  GE_CHECK_NOTNULL(recv_node->GetOutControlAnchor());
  GE_ASSERT_SUCCESS(GraphUtils::AddEdge(recv_node->GetOutControlAnchor(), dst_node->GetInControlAnchor()),
                    "Add edge from node %s to node %s failed", recv_node->GetNamePtr(), dst_node->GetNamePtr());
  GE_ASSERT_SUCCESS(RefreshRcvList(dst_node, event_id, false));
  GEEVENT("Insert recv event node %s event id %u before node: %s with stream %ld",
         recv_node->GetNamePtr(), event_id, dst_node->GetNamePtr(), rcv_stream_id);
  return SUCCESS;
}

Status SuperKernelScope::LinkEventNode() {
  std::map<std::string, std::set<std::string>> has_insert_send_rcv_nodes_set;
  for (auto &sub_info : super_nodes_info_) {
    // refresh sub node send and rcv list here directly
    GE_ASSERT_SUCCESS(RefreshSendList(sub_info.second.cur_node, 0, true));
    GE_ASSERT_SUCCESS(RefreshRcvList(sub_info.second.cur_node, 0, true));
    for (auto &send_event_info : sub_info.second.send_nodes_info) {
      GELOGI("start to process %s send link, origin event_id is %u",
             sub_info.second.cur_node->GetNamePtr(), send_event_info.event_id);
      send_event_info.send_node = nullptr;
      send_event_info.rcv_node = nullptr;

      auto it_new = newNodeMap_.find(send_event_info.src_node_name);
      GE_ASSERT_TRUE(it_new != newNodeMap_.end(), "%s is not in map", send_event_info.src_node_name.c_str());
      send_event_info.src_node = it_new->second;
      // only inner sub node need to be update
      if (super_nodes_set_.find(send_event_info.dst_node_name) != super_nodes_set_.end()) {
        it_new = newNodeMap_.find(send_event_info.dst_node_name);
        GE_ASSERT_TRUE(it_new != newNodeMap_.end(), "%s is not in map", send_event_info.dst_node_name.c_str());
        send_event_info.dst_node = it_new->second;
      }
      NodePtr real_src_node = send_event_info.src_node;
      bool is_last_send_node = send_event_info.src_node->GetName() == super_nodes_[super_nodes_.size() - 1]->GetName();
      if (is_last_send_node) {
        GELOGI("current node %s is sk last sub node, so sk parent node %s replace to send ",
               send_event_info.src_node->GetNamePtr(), super_node_->GetNamePtr());
        real_src_node = super_node_;
      }
      bool use_normal_event = is_last_send_node && (send_event_info.dst_node->GetOwnerComputeGraph() == origin_graph_) &&
                          !reuse_event_id_set_.empty();

      auto it_pair = has_insert_send_rcv_nodes_set[real_src_node->GetName()].
                     insert(send_event_info.dst_node->GetName());
      GELOGI("process %s event link src %s, dst %s, is_first_insert %d, reuse_event_id_set_ size %zu",
             sub_info.second.cur_node->GetNamePtr(), real_src_node->GetNamePtr(),
             send_event_info.dst_node->GetNamePtr(), it_pair.second, reuse_event_id_set_.size());
      if (it_pair.second) {
        send_event_info.event_id = event_begin_id_ + event_num_;
        if (use_normal_event) {
          send_event_info.event_id = *reuse_event_id_set_.begin();
          reuse_event_id_set_.erase(reuse_event_id_set_.begin());
          GELOGI("process %s event link src %s, dst %s, event_id %u, reuse_event_id_set_ size %zu, send stream id %ld",
                 sub_info.second.cur_node->GetNamePtr(), real_src_node->GetNamePtr(),
                 send_event_info.dst_node->GetNamePtr(), send_event_info.event_id, reuse_event_id_set_.size(),
                 send_event_info.send_stream_id);
        }

        GE_ASSERT_SUCCESS(RefreshSendList(send_event_info.src_node, 0, true));
        GE_ASSERT_SUCCESS(InsertSrcNode2SendMem(real_src_node, send_event_info.event_id,
                                                scope_stream_id_, !use_normal_event));
        GE_ASSERT_SUCCESS(InsertRecvMem2DstNode(send_event_info.dst_node, send_event_info.event_id,
                                                send_event_info.rcv_stream_id, !use_normal_event));
        ++event_num_;
      }
    }
    for (auto &rcv_event_info : sub_info.second.rcv_nodes_info) {
      GELOGI("start to process %s rcv link, origin event_id is %u",
             sub_info.second.cur_node->GetNamePtr(), rcv_event_info.event_id);
      rcv_event_info.send_node = nullptr;
      rcv_event_info.rcv_node = nullptr;

      auto it_new = newNodeMap_.find(rcv_event_info.dst_node_name);
      GE_ASSERT_TRUE(it_new != newNodeMap_.end(), "%s is not in map", rcv_event_info.dst_node_name.c_str());
      rcv_event_info.dst_node = it_new->second;
      // only inner sub node need to be update
      if (super_nodes_set_.find(rcv_event_info.src_node_name) != super_nodes_set_.end()) {
        it_new = newNodeMap_.find(rcv_event_info.src_node_name);
        GE_ASSERT_TRUE(it_new != newNodeMap_.end(), "%s is not in map", rcv_event_info.src_node_name.c_str());
        rcv_event_info.src_node = it_new->second;
      }

      NodePtr real_dst_node = rcv_event_info.dst_node;
      // special process
      bool is_support = false;
      GE_ASSERT_NOTNULL(real_dst_node);
      auto op_desc = real_dst_node->GetOpDesc();
      GE_ASSERT_NOTNULL(op_desc);
      gert::OpImplSpaceRegistryV2Array space_registry_array;
      GE_ASSERT_TRUE(static_cast<size_t>(op_desc->GetOppImplVersion()) < space_registry_array.size());
      space_registry_array.at(static_cast<size_t>(op_desc->GetOppImplVersion())) =
          gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
      gert::DataDependentInterpreter ddi(op_desc, space_registry_array);
      (void)ddi.IsSupportTilingDependPlacement(static_cast<uint32_t>(gert::TilingPlacement::TILING_ON_AICPU),
                                               is_support);
      const bool is_tiling_stream = (rcv_event_info.rcv_stream_id != real_dst_node->GetOpDesc()->GetStreamId()) && is_support;
      bool is_first_rcv_node = rcv_event_info.dst_node->GetName() == super_nodes_[0]->GetName();
      auto it = first_other_stm_sub_nodes_.find(rcv_event_info.dst_node->GetName());
      if (it != first_other_stm_sub_nodes_.end() &&
          super_nodes_set_.find(rcv_event_info.src_node->GetName()) == super_nodes_set_.end() &&
          ((rcv_event_info.rcv_stream_id == scope_stream_id_) ||
           (rcv_event_info.rcv_stream_id == rcv_event_info.dst_node->GetOpDesc()->GetStreamId()))) {
        is_first_rcv_node = true;
        GELOGI("current rcv pair if first rcv node, src %s, dst %s, send stream id %ld, rcv stream id %ld",
               rcv_event_info.src_node->GetNamePtr(), rcv_event_info.dst_node->GetNamePtr(),
               rcv_event_info.send_stream_id, rcv_event_info.rcv_stream_id);
      }
      if (is_first_rcv_node || is_tiling_stream) {
        GEEVENT("current node %s is sk first sub node, or tiling stream %d, so sk parent node %s replace to receive ",
               real_dst_node->GetNamePtr(), is_tiling_stream, super_node_->GetNamePtr());
        real_dst_node = super_node_;
      }
      bool use_normal_event = (is_first_rcv_node || is_tiling_stream) &&
                              (rcv_event_info.src_node->GetOwnerComputeGraph() == origin_graph_) &&
                              !reuse_event_id_set_.empty();
      auto it_pair = has_insert_send_rcv_nodes_set[rcv_event_info.src_node->GetName()].
                     insert(real_dst_node->GetName());
      GELOGI("process %s event link src %s, dst %s, is_first_insert %d, reuse_event_id_set_ size %zu",
             sub_info.second.cur_node->GetNamePtr(), rcv_event_info.src_node->GetNamePtr(),
             real_dst_node->GetNamePtr(), it_pair.second, reuse_event_id_set_.size());
      if (it_pair.second) {
        rcv_event_info.event_id = event_begin_id_ + event_num_;
        if (use_normal_event) {
          rcv_event_info.event_id = *reuse_event_id_set_.begin();
          reuse_event_id_set_.erase(reuse_event_id_set_.begin());
          GELOGI("process %s event link src %s, dst %s, event_id %u, reuse_event_id_set_ size %zu, rcv stream id is %ld",
                 sub_info.second.cur_node->GetNamePtr(), rcv_event_info.src_node->GetNamePtr(),
                 real_dst_node->GetNamePtr(), rcv_event_info.event_id, reuse_event_id_set_.size(),
                 rcv_event_info.rcv_stream_id);
        }
        GE_ASSERT_SUCCESS(InsertSrcNode2SendMem(rcv_event_info.src_node, rcv_event_info.event_id,
                                                rcv_event_info.send_stream_id, !use_normal_event));
        GE_ASSERT_SUCCESS(RefreshRcvList(rcv_event_info.dst_node, 0, true));
        int64_t last_rcv_stream_id = is_tiling_stream ? rcv_event_info.rcv_stream_id : scope_stream_id_;
        GE_ASSERT_SUCCESS(InsertRecvMem2DstNode(real_dst_node, rcv_event_info.event_id,
                                                last_rcv_stream_id, !use_normal_event));
        ++event_num_;
      }
    }
  }
  return SUCCESS;
}

Status SuperKernelScope::LinkInnerNodes(NodePtr &inner_netoutput_node) {
  for (auto &super_node_info_it : super_nodes_info_) {
    auto it = newNodeMap_.find(super_node_info_it.second.node_name);
    GE_ASSERT_TRUE(it != newNodeMap_.end());
    super_node_info_it.second.cur_node = it->second;
    for (auto &input_node_info : super_node_info_it.second.input_nodes_info) {
      // outside in_node key is name + _out_id
      std::string in_node_name = super_nodes_set_.find(input_node_info.node_name) != super_nodes_set_.end() ?
          input_node_info.node_name :
          (input_node_info.node_name + "_out_" + to_string(input_node_info.out_data_anchor_id));
      it = newNodeMap_.find(in_node_name);
      GE_ASSERT_TRUE(it != newNodeMap_.end());
      input_node_info.cur_node = it->second;
      if (input_node_info.cur_node->GetType() == ge::DATA) {
        input_node_info.out_data_anchor_id = 0;
      }
      GELOGI("link node %s output %zu to node %s input %zu",
             input_node_info.cur_node->GetNamePtr(), input_node_info.out_data_anchor_id,
             super_node_info_it.second.cur_node->GetNamePtr(), input_node_info.in_data_anchor_id);
      GE_ASSERT_SUCCESS(GraphUtils::AddEdge(
          input_node_info.cur_node->GetOutDataAnchor(input_node_info.out_data_anchor_id),
          super_node_info_it.second.cur_node->GetInDataAnchor(input_node_info.in_data_anchor_id)));
    }
  }

  // 连接子图内部netoutput节点
  for (const auto &ele : out_nodes_order_) {
    auto it = newNodeMap_.find(ele.first);
    GE_ASSERT_TRUE(it != newNodeMap_.end());
    auto &out_nodes = ele.second;
    for (const auto &out_node : out_nodes) {
      size_t parent_id = out_node.first;
      size_t out_data_anchor_id = out_node.second;
      GELOGI("link node %s output %zu to node %s input %zu", it->second->GetNamePtr(), out_data_anchor_id,
             inner_netoutput_node->GetNamePtr(), parent_id);
      GE_ASSERT_SUCCESS(GraphUtils::AddEdge(it->second->GetOutDataAnchor(out_data_anchor_id),
                                            inner_netoutput_node->GetInDataAnchor(parent_id)));
      GE_ASSERT_TRUE(AttrUtils::SetInt(inner_netoutput_node->GetOpDesc()->MutableInputDesc(parent_id),
                                       ATTR_NAME_PARENT_NODE_INDEX, parent_id));
    }
  }
  return SUCCESS;
}

NodePtr SuperKernelScope::ConstructInnerOutputNode() {
  std::string netoutput_node_name = sub_graph_->GetName() + "_inner_netoutput";
  auto netoutput_op_desc = MakeShared<OpDesc>(netoutput_node_name, ge::NETOUTPUT);
  for (size_t i = 0; i < merge_node_output_vec_.size(); ++i) {
    netoutput_op_desc->AddInputDesc(GeTensorDesc());
  }
  // 必须构造好inputdesc占位才能addnode生成node，否则会没有输入anchor
  return sub_graph_->AddNode(netoutput_op_desc);
}

Status SuperKernelScope::UnlinkSrcSendLink(const NodePtr send_node) const {
  auto in_ctrl_anchor = send_node->GetInControlAnchor();
  GE_ASSERT_NOTNULL(in_ctrl_anchor);
  auto out_ctrl_anchors = in_ctrl_anchor->GetPeerOutControlAnchors();
  GELOGI("UnlinkSrcSendLink %s, out_ctrl_anchors size is %zu", send_node->GetNamePtr(), out_ctrl_anchors.size());
  for (auto &out_ctrl_anchor : out_ctrl_anchors) {
    if (out_ctrl_anchor != nullptr) {
      GELOGI("unlink src_node %s, dst_node %s ctrl edge",
             out_ctrl_anchor->GetOwnerNode()->GetNamePtr(), send_node->GetNamePtr());
      (void)GraphUtils::RemoveEdge(out_ctrl_anchor, in_ctrl_anchor);
    }
  }
  return SUCCESS;
}

Status SuperKernelScope::UnlinkRcvDstLink(const NodePtr rcv_node) const {
  auto out_ctrl_anchor = rcv_node->GetOutControlAnchor();
  GE_ASSERT_NOTNULL(out_ctrl_anchor);
  auto in_control_anchors = out_ctrl_anchor->GetPeerInControlAnchors();
  GELOGI("UnlinkRcvDstLink %s, in_control_anchors size is %zu", rcv_node->GetNamePtr(), in_control_anchors.size());
  for (auto &in_ctrl_anchor : in_control_anchors) {
    if (in_ctrl_anchor != nullptr) {
      GELOGI("unlink src_node %s, dst_node %s ctrl edge",
             rcv_node->GetNamePtr(), in_ctrl_anchor->GetOwnerNode()->GetNamePtr());
      (void)GraphUtils::RemoveEdge(out_ctrl_anchor, in_ctrl_anchor);
    }
  }
  return SUCCESS;
}

Status SuperKernelScope::UnlinkSyncLink(EventNodeInfo &event_node_info) {
  GELOGI("src_node %s, unlink send_node %s, rcv_node %s, dst_node %s, unlink event_id %u",
         event_node_info.src_node_name.c_str(), event_node_info.send_node_name.c_str(),
         event_node_info.rcv_node_name.c_str(), event_node_info.dst_node_name.c_str(), event_node_info.event_id);
  // unlink src->send rcv->dst ctrl edge
  delete_event_id_set_.insert(event_node_info.event_id);
  if (event_node_info.send_node->GetType() == SEND) {
    reuse_event_id_set_.insert(event_node_info.event_id);
  }
  GE_ASSERT_SUCCESS(UnlinkSrcSendLink(event_node_info.send_node));
  GraphUtils::RemoveNodeWithoutRelink(event_node_info.send_node->GetOwnerComputeGraph(), event_node_info.send_node);
  GE_ASSERT_SUCCESS(UnlinkRcvDstLink(event_node_info.rcv_node));
  GraphUtils::RemoveNodeWithoutRelink(event_node_info.rcv_node->GetOwnerComputeGraph(), event_node_info.rcv_node);

  // unlink src and dst node ctrl edge
  auto &dst_node = event_node_info.dst_node;
  GE_ASSERT_NOTNULL(dst_node);
  auto in_ctrl_anchor = dst_node->GetInControlAnchor();
  GE_ASSERT_NOTNULL(in_ctrl_anchor);
  auto out_ctrl_anchors = in_ctrl_anchor->GetPeerOutControlAnchors();
  for (auto &out_ctrl_anchor : out_ctrl_anchors) {
    if ((out_ctrl_anchor != nullptr) && (out_ctrl_anchor->GetOwnerNode() == event_node_info.src_node)) {
      GELOGI("unlink src_node %s, dst_node %s ctrl edge",
             event_node_info.src_node->GetNamePtr(), event_node_info.dst_node->GetNamePtr());
      (void)GraphUtils::RemoveEdge(out_ctrl_anchor, in_ctrl_anchor);
    }
  }
  return SUCCESS;
}

Status SuperKernelScope::UnlinkSkNodes() {
  for (const auto &super_sub_node : super_nodes_) {
    GELOGI("start to process sub_node link %s", super_sub_node->GetNamePtr());
    // unlink send rcv nodes, if not in sub nodes, delete edge and node.
    for (auto &send_event_info : super_nodes_info_[super_sub_node->GetName()].send_nodes_info) {
      bool no_send_rcv_node = ((send_event_info.send_node == nullptr) || (send_event_info.rcv_node == nullptr));
      if (no_send_rcv_node) {
        continue;
      }
      GE_ASSERT_SUCCESS(UnlinkSyncLink(send_event_info));
    }
    for (auto &rcv_event_info : super_nodes_info_[super_sub_node->GetName()].rcv_nodes_info) {
      bool no_send_rcv_node = ((rcv_event_info.send_node == nullptr) || (rcv_event_info.rcv_node == nullptr));
      if (no_send_rcv_node) {
        continue;
      }
      GE_ASSERT_SUCCESS(UnlinkSyncLink(rcv_event_info));
    }
    NodeUtils::UnlinkAll(*super_sub_node);
    GraphUtils::RemoveNodeWithoutRelink(super_sub_node->GetOwnerComputeGraph(), super_sub_node);
    GELOGI("unlink super_sub_node %s", super_sub_node->GetNamePtr());
    auto inner_super_node = sub_graph_->AddNode(super_sub_node->GetOpDesc());
    GE_ASSERT_TRUE(AttrUtils::SetStr(super_sub_node->GetOpDesc(), "sk_parent_node_name", super_node_->GetName()));
    GE_ASSERT_NOTNULL(inner_super_node);
    newNodeMap_[inner_super_node->GetName()] = inner_super_node;
  }
  return SUCCESS;
}

Status SuperKernelScope::MergeSuperKernelsToSubgraph() {
  // 得到所有节点的输入输出连边关系
  GE_ASSERT_SUCCESS(GetSuperNodesIoInfo());

  // 获取父节点的需要连接的连边关系
  GE_ASSERT_SUCCESS(GetSkNodeLinkInfo());

  // 构造SuperKernel节点和子图
  super_node_ = ConstructSkNode();
  GE_ASSERT_NOTNULL(super_node_);
  GE_ASSERT_SUCCESS(ConstructSubgraph());

  // 主图断边并生成子图内节点
  GE_ASSERT_SUCCESS(UnlinkSkNodes());

  GE_ASSERT_SUCCESS(LinkSkInputNode());
  auto inner_netoutput_node = ConstructInnerOutputNode();
  GE_ASSERT_NOTNULL(inner_netoutput_node);
  GE_ASSERT_SUCCESS(LinkSkOutputNode(inner_netoutput_node));
  GE_ASSERT_SUCCESS(LinkInnerNodes(inner_netoutput_node));
  GE_ASSERT_SUCCESS(LinkEventNode());

  GE_DUMP(sub_graph_, sub_graph_->GetName());
  super_node_->GetOpDesc()->SetExtAttr("_sk_sub_graph", sub_graph_);
  return SUCCESS;
}

size_t SuperKernelScope::GetScopeEventSize() const {
  return event_num_;
}
}  // namespace ge

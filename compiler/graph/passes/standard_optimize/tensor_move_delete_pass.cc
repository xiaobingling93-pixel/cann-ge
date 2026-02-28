/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "tensor_move_delete_pass.h"

#include <stack>
#include <unordered_set>
#include <queue>

#include "graph/utils/graph_utils.h"
#include "graph/utils/attr_utils.h"
#include "common/checker.h"
#include "ge_local_context.h"
#include "node_utils.h"
#include "op_type_utils.h"
#include "common/omg_util.h"
#include "framework/common/util.h"

namespace ge {
namespace {
struct TensorMoveDeleteContext {
  NodePtr tensor_move;
  std::vector<std::pair<NodePtr, OutDataAnchorPtr>> path_to_source_node;
};
using DeleteRule = std::function<bool(TensorMoveDeleteContext &)>;

bool IsTensorMove(const NodePtr &node) {
  return node->GetType() == TENSORMOVE;
}

/**
 * @brief 判断源节点是否为特殊的数据源节点（可能在其他图中用到）
 * 主要包含以下几类：
 * 1. 变量类 (VARIABLE, VARIABLEV2, REFDATA)
 * 2. 常量类 (CONSTANT, CONSTANTOP, CONSTPLACEHOLDER)
 * 3. 显示引用类 (具有 REF_VAR_SRC_VAR_NAME 属性，指向源变量的节点)
 */
bool IsSourceNodeSpecial(const NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  const auto &node_type = node->GetType();
  if (OpTypeUtils::IsVariableNode(node_type) || OpTypeUtils::IsVarLikeNode(node_type) || OpTypeUtils::IsConstNode(node_type)) {
    GELOGI("Node %s of type %s is special node", node->GetName().c_str(), node_type.c_str());
    return true;
  }

  string ref_origin_name;
  if (AttrUtils::GetStr(node->GetOpDesc(), ge::REF_VAR_SRC_VAR_NAME, ref_origin_name)) {
    GELOGI("Node %s of type %s is special node because it has ref var src name: %s",
           node->GetName().c_str(), node_type.c_str(), ref_origin_name.c_str());
    return true;
  }
  return false;
}

/**
 * @brief 处理子图 DATA 节点的跳出逻辑，将追踪从子图内部切换到定位到父节点对应输入
 * * 查找当前 Data 节点所属的Wrapper Node，将 Wrapper 节点和前驱节点加入路径，并更新追踪锚点与状态标志。
 *
 * @param cur_node        [IN]  当前遇到的子图 DATA 节点
 * @param source_path     [OUT] 路径记录容器，Wrapper节点对应输出锚点会被加入此集合
 * @param cur_in_anchor   [OUT] 下一轮迭代的输入锚点，将被更新为父节点对应输入锚点
 * @return Status         SUCCESS: 成功跳出子图并定位到父节点
 *                        FAILED:  获取父节点失败
 */
Status JumpOutFromSubDataToTraceSource(const NodePtr &cur_node, std::vector<std::pair<NodePtr, OutDataAnchorPtr>> &source_path,
                                       InDataAnchorPtr &cur_in_anchor) {
  // 获取 Wrapper 节点的输入锚点 (也就是这一层子图的“入口”)
  const auto parent_in_anchor = NodeUtils::GetParentInDataAnchor(cur_node);
  GE_ASSERT_NOTNULL(parent_in_anchor, "Get parent input anchor failed for DATA node: %s", cur_node->GetName().c_str());

  const auto parent_node = parent_in_anchor->GetOwnerNode();

  // 将 Wrapper 节点加入路径, 记录 Wrapper 节点和它的第0个输出，主要用于标识路径
  source_path.emplace_back(parent_node, parent_node->GetOutDataAnchor(0));
  cur_in_anchor = parent_in_anchor;

  GELOGI("Jump out of subgraph from DATA %s to parent node %s input index %d.",
         cur_node->GetName().c_str(), parent_node->GetName().c_str(), cur_in_anchor->GetIdx());
  return SUCCESS;
}

/**
 * @brief 根据 ATTR_NAME_PARENT_NODE_INDEX 属性找到映射到当前 PCall 输出端口的输入分支，并将追踪状态更新为子图内部的生产者节点
 *
 * @param cur_node        [IN/OUT] 当前节点。输入时为 PartitionedCall 节点，成功后更新为子图内部的生产者节点
 * @param cur_out_idx     [IN/OUT] 当前输出索引。输入时为 PCall 的输出索引，成功后更新为子图内部节点的输出索引
 * @param source_path     [OUT]    路径记录容器，找到的子图内部节点和对应输出锚点会被加入此路径
 * @param cur_in_anchor   [OUT]    当前输入锚点。将被更新为子图内部节点的输入锚点，在主循环中继续向上回溯
 * @return Status         SUCCESS: 成功找到映射并进入子图
 *                        FAILED:  子图 NetOutput 中无法找到对应当前索引的映射关系
 */
Status JumpInPartitionedCallToTraceSource(const NodePtr &cur_node, int32_t &cur_out_idx,
                                          std::vector<std::pair<NodePtr, OutDataAnchorPtr>> &source_path,
                                          InDataAnchorPtr &cur_in_anchor) {
  const auto sub_graph = NodeUtils::GetSubgraph(*cur_node, 0U);
  GE_ASSERT_NOTNULL(sub_graph);

  const auto sub_graph_netoutput = sub_graph->GetOrUpdateNetOutputNode();
  GE_CHECK_NOTNULL(sub_graph_netoutput);

  for (const auto &in_data_anchor : sub_graph_netoutput->GetAllInDataAnchorsPtr()) {
    int32_t ref_o = -1;
    auto in_desc = sub_graph_netoutput->GetOpDesc()->GetInputDesc(static_cast<uint32_t>(in_data_anchor->GetIdx()));
    GE_ASSERT_TRUE(AttrUtils::GetInt(in_desc, ATTR_NAME_PARENT_NODE_INDEX, ref_o),
                   "Subgraph NetOutput node %s input index %d has no parent node index attr.",
                   sub_graph_netoutput->GetName().c_str(), in_data_anchor->GetIdx());

    if (ref_o == cur_out_idx) {
      // 将cur_in_anchor更新为子图netoutput对应的输入锚点
      auto in_data_anchor_idx = in_data_anchor->GetIdx();
      cur_in_anchor = sub_graph_netoutput->GetInDataAnchor(in_data_anchor_idx);
      // 下一轮将遍历到sub netoutput的前驱，因此这里将sub netoutput和输出锚点加入集合
      source_path.emplace_back(sub_graph_netoutput, in_data_anchor->GetPeerOutAnchor());

      GELOGI("Jump into subgraph %s, from PartitionedCall node %s to sub netoutput node %s index %d",
             sub_graph->GetName().c_str(), cur_node->GetName().c_str(), sub_graph_netoutput->GetName().c_str(), in_data_anchor_idx);
      return SUCCESS;
    }
  }
  GELOGE(FAILED, "Cannot find corresponding sub netoutput and parent node index mapping in subgraph %s for PartitionedCall node %s output index %d",
         sub_graph->GetName().c_str(), cur_node->GetName().c_str(), cur_out_idx);
  return FAILED;
}

void LogTraceRealSourcePath(const NodePtr &start_node, int32_t index,
                            const std::vector<std::pair<NodePtr, OutDataAnchorPtr>> &source_path) {
  if (start_node == nullptr) {
    return;
  }

  std::stringstream ss;
  // 反向遍历：从源头开始打印
  for (auto it = source_path.rbegin(); it != source_path.rend(); ++it) {
    const auto &node = it->first;
    const auto &anchor = it->second;
    if (node != nullptr && anchor != nullptr) {
      ss << node->GetName() << "(out:" << anchor->GetIdx() << ")-->";
    }
  }
  // 最后追加终点（即当前开始回溯的节点）的入边信息
  ss << "(in:" << index << ")" << start_node->GetName();

  GELOGI("Trace reach real source: %s", ss.str().c_str());
}

/**
 * @brief 从指定节点的输入端口出发，逆向回溯数据流，寻找生成该数据的真正源头节点
 *
 * 核心处理逻辑：
 *
 * 1. 遇到控制流算子终止
 * 若回溯路径上遇到 IF, WHILE, CASE 等控制流算子，视为追踪边界。
 * 此时停止追踪，返回 SUCCESS，并将 source_anchor_index 置为 kInvalidIndex。
 *
 * 2. 子图跳出
 * 若遇到子图内的 DATA 节点（且非根图），说明数据来自父图。
 * 函数会自动跳转到父图中对应的 Wrapper 节点（如 IF 算子）的前驱节点继续回溯。
 *
 * 3. 子图钻入
 * 若遇到 PARTITIONEDCALL 节点，说明数据产生于子图内部。
 * 函数会解析子图结构，钻入子图内部，找到 NetOutput 对应分支的真实生产者。
 *
 * 4. RefOp 透传 / TensorMove
 * 若节点属于RefOp，且存在输入输出的复用关系，
 * 函数会自动跳过该节点，继续沿其复用的输入端口向上回溯。
 *
 * jump_to_prev说明
 * true (默认): 标准回溯模式: 当前 cur_in_anchor 指向的是本层节点的输入，去寻找上游节点作为 cur_node。
 * false: 通常发生在从子图 DATA 节点跳出到父图时，此时 cur_in_anchor 已经被手动更新为父图节点的输入锚点。
 *
 * @param start_node          [IN]  回溯的起始节点。
 * @param index               [IN]  起始节点的输入锚点索引
 * @param source_path         [OUT] 路径记录容器。函数会将从起点到源头过程中经过的所有节点和对应输出锚点加到此列表中（不包含当前TenosrMove节点）
 *
 * @return Status
 * - SUCCESS: 追踪流程正常结束（无论是否找到有效源头，包括遇到控制流算子终止的情况）。
 * - FAILED:  追踪过程中发生错误（如断图、子图映射关系丢失、无法获取锚点等）。
 */
Status TraceRealSourceNode(const NodePtr &start_node, int32_t index, std::vector<std::pair<NodePtr, OutDataAnchorPtr>> &source_path) {
  InDataAnchorPtr cur_in_anchor = start_node->GetInDataAnchor(index);
  while (cur_in_anchor != nullptr) {
    NodePtr cur_node;
    int32_t cur_out_idx;
    OutDataAnchorPtr peer_out_anchor;

    peer_out_anchor = cur_in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      GELOGI("Input anchor index %d of node %s has no peer output anchor.", cur_in_anchor->GetIdx(), start_node->GetName().c_str());
      return SUCCESS;
    }
    cur_node = peer_out_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(cur_node);
    cur_out_idx = peer_out_anchor->GetIdx();
    source_path.emplace_back(cur_node, peer_out_anchor);

    // 1. 遇到控制流算子，停止追踪
    if (NodeUtils::IsMultiBranchControlFlowOp(cur_node)) {
      GELOGI("Stop tracing real source for node %s as multi branch control node %s (type: %s) is encountered.",
             start_node->GetName().c_str(), cur_node->GetName().c_str(), cur_node->GetType().c_str());
      return SUCCESS;
    }
    // 2. 处理跨子图跳出 (DATA)
    if (cur_node->GetType() == DATA && !NodeUtils::IsNodeInRootGraph(cur_node)) {
      GE_ASSERT_SUCCESS(JumpOutFromSubDataToTraceSource(cur_node, source_path, cur_in_anchor));
      continue;
    }
    // 3. 处理钻入子图 (PARTITIONEDCALL)
    if (cur_node->GetType() == PARTITIONEDCALL) {
      GE_ASSERT_SUCCESS(JumpInPartitionedCallToTraceSource(cur_node, cur_out_idx, source_path, cur_in_anchor));
      continue;
    }
    // 4. RefOp透传逻辑
    if (peer_out_anchor != nullptr) {
      int32_t reuse_in_idx = -1;
      if (GraphUtils::IsRefFromInput(peer_out_anchor, reuse_in_idx)) {
        cur_in_anchor = cur_node->GetInDataAnchor(reuse_in_idx);
        continue;
      }
    }
    // 到达终点，打印路径信息，期望格式: Data(out:0)-->RefOp(out:0)-->(in:0)TensorMove
    LogTraceRealSourcePath(start_node, index, source_path);
    return SUCCESS;
  }
  return FAILED;
}

/**
 * @brief 检查RefOp是否有其他输出复用了同一个输入
 * @param node 当前检查的节点
 * @param current_out_anchor 当前路径正在追踪的输出 Anchor
 * @param tensor_move_name TensorMove 节点名称
 * @return true 存在其他输出复用了同一个输入
 * @return false 不存在其他输出复用了同一个输入
 */
bool HasMultipleOutputsSharingSameInput(const NodePtr &node, const OutDataAnchorPtr &current_out_anchor, const std::string &tensor_move_name) {
  int32_t reuse_in_index = -1;
  // 1. 如果当前 Anchor 本身不是引用输入，直接返回false
  if (!GraphUtils::IsRefFromInput(current_out_anchor, reuse_in_index)) {
    return false;
  }

  // 2. 遍历该节点所有其他输出 Anchor
  for (const auto &tmp_out_anchor : node->GetAllOutDataAnchors()) {
    // 跳过当前正在追踪的这个端口
    if (tmp_out_anchor->GetIdx() == current_out_anchor->GetIdx()) {
      continue;
    }

    // 3. 检查旁路是否引用了同一个输入，且该旁路有下游节点连接
    int32_t tmp_reuse_in_index = -1;
    if (GraphUtils::IsRefFromInput(tmp_out_anchor, tmp_reuse_in_index) &&
        (tmp_reuse_in_index == reuse_in_index) && !tmp_out_anchor->GetPeerInDataAnchors().empty()) {
      GELOGI("Node %s(type %s) has multiple outputs (Out:%d and Out:%d) sharing input:%d, "
             "and both are connected. Cannot delete tensor move %s.",
             node->GetName().c_str(), node->GetType().c_str(),
             current_out_anchor->GetIdx(), tmp_out_anchor->GetIdx(),
             reuse_in_index, tensor_move_name.c_str());
      return true;
    }
  }

  return false;
}

/**
 * @brief 检查从TensorMove节点回溯到源节点的路径上，是否所有节点都仅有单一输出
 * @param tensor_move_node 当前的 TensorMove (数据拷贝) 节点
 * @param path_to_source_node 从 TensorMove 回溯到源节点的路径列表，存放节点和对应输出锚点
 * @return true 路径上所有节点都只有单一输出
 * @return false 路径上存在被多处引用的节点
 */
bool IsSourceNodeWithSinglePath(const NodePtr &tensor_move_node, const std::vector<std::pair<NodePtr, OutDataAnchorPtr>> &path_to_source_node) {
  GE_ASSERT_TRUE(!path_to_source_node.empty());

  for (const auto &pairs : path_to_source_node) {
    const auto &node = pairs.first;
    const auto &out_data_anchor = pairs.second;
    GE_ASSERT_NOTNULL(node);
    GE_ASSERT_NOTNULL(out_data_anchor);

    // 多分支控制流算子
    if (NodeUtils::IsMultiBranchControlFlowOp(node)) {
      GELOGI("Node %s type %s is multi branch control flow op, cannot delete tensor move %s.", node->GetName().c_str(),
             node->GetType().c_str(), tensor_move_node->GetName().c_str());
      return false;
    }

    // 多个输出和当前out_data_anchor都引用一个in_data_anchor
    if (HasMultipleOutputsSharingSameInput(node, out_data_anchor, tensor_move_node->GetName())) {
      return false;
    }

    // 单输出多引用
    if (out_data_anchor->GetPeerInDataAnchors().size() > 1U) {
      GELOGI("Out data anchor %d of node %s(type %s) has multiple peer intput data anchors, cannot delete tensor move %s.",
             out_data_anchor->GetIdx(), node->GetName().c_str(), node->GetType().c_str(), tensor_move_node->GetName().c_str());
      return false;
    }
  }
  GELOGI("All nodes in the path from node %s to source node %s have single output.",
         tensor_move_node->GetName().c_str(), path_to_source_node.back().first->GetName().c_str());
  return true;
}

/**
 * @brief 检查当前源输入节点索引是否在允许的内存复用配置中
 * @param src_out_idx 源节点（通常为 Data 节点）的输出索引，表示第几个输入
 * @param node_name 当前处理的节点名称（用于日志记录）
 * @return 如果该输入索引存在于全局配置中则返回 true，否则返回 false
 */
bool IsMemoryReuseAllowed(int32_t src_out_idx, const std::string &node_name) {
  // Check ge.exec.outputReuseInputMemIndexes first
  std::string output_reuse_input_config;
  if (GetThreadLocalContext().GetOption(OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES, output_reuse_input_config) == SUCCESS) {
    std::vector<std::pair<size_t, size_t>> io_same_addr_pairs;
    ParseOutputReuseInputMemIndexes(output_reuse_input_config, io_same_addr_pairs);

    for (const auto &pair : io_same_addr_pairs) {
      if (pair.first == static_cast<size_t>(src_out_idx)) {
        GELOGI("Memory reuse check passed: input index %d found in ge.exec.outputReuseInputMemIndexes config %s for node %s.",
               src_out_idx, output_reuse_input_config.c_str(), node_name.c_str());
        return true;
      }
    }
  }

  // Check ge.exec.inputReuseMemIndexes
  std::string input_reuse_config;
  if (GetThreadLocalContext().GetOption(OPTION_INPUT_REUSE_MEM_INDEXES, input_reuse_config) == SUCCESS) {
    if (input_reuse_config.empty()) {
      GELOGI("Memory reuse check skipped: ge.exec.inputReuseMemIndexes is empty.");
      return false;
    }
    std::vector<std::string> input_indexes;
    SplitStringByComma(input_reuse_config, input_indexes);
    for (const auto &index_str : input_indexes) {
      const int32_t index = std::stoi(index_str);
      if (index == src_out_idx) {
        GELOGI("Memory reuse check passed: input index %d found in ge.exec.inputReuseMemIndexes config %s for node %s.",
                src_out_idx, input_reuse_config.c_str(), node_name.c_str());
        return true;
      }
    }
  }

  GELOGI("Memory reuse check not pass: input index %d not found in any reuse config, node %s cannot be deleted.",
         src_out_idx, node_name.c_str());
  return false;
}

bool HasReservedAttr(const NodePtr &node) {
  bool reserved = false;
  AttrUtils::GetBool(node->GetOpDesc(),
                     ATTR_NAME_CANNOT_BE_DELETED, reserved);
  if (reserved) {
    GELOGI("TensorMove %s reserved by attr %s.", node->GetName().c_str(), ATTR_NAME_CANNOT_BE_DELETED.c_str());
    return true;
  }
  if (node->GetOpDesc()->HasAttr(ge::ATTR_NO_NEED_CONSTANT_FOLDING)) {
    GELOGI("TensorMove %s reserved by attr %s.", node->GetName().c_str(), ATTR_NO_NEED_CONSTANT_FOLDING.c_str());
    return true;
  }
  return false;
}

DeleteRule CheckPathToSourceNodeValid = [](const TensorMoveDeleteContext &ctx) {
  const auto &path_pairs = ctx.path_to_source_node;
  if (path_pairs.empty()) {
    GELOGI("TensorMove %s has empty path to source node.", ctx.tensor_move->GetName().c_str());
    return false;
  }

  const auto &source_node = path_pairs.back().first;
  if (NodeUtils::IsMultiBranchControlFlowOp(source_node)) {
    GELOGI("Node %s in source path of TensorMove %s is multi branch control node, cannot delete.",
           source_node->GetName().c_str(), ctx.tensor_move->GetName().c_str());
    return false;
  }

  if (IsSourceNodeSpecial(source_node)) {
    GELOGD("Source node %s of TensorMove %s is special node, cannot delete.",
           source_node->GetName().c_str(), ctx.tensor_move->GetName().c_str());
    return false;
  }
  return true;
};

DeleteRule CheckSourceNodeReuse = [](const TensorMoveDeleteContext &ctx) {
  const auto src_node = ctx.path_to_source_node.back().first;
  // 源输入是普通计算节点，则无需后续校验
  if (!OpTypeUtils::IsDataNode(src_node->GetType())) {
    return true;
  }

  uint64_t input_index;
  if (!AttrUtils::GetInt(src_node->GetOpDesc(), ATTR_NAME_INDEX, input_index)) {
    GELOGI("Node %s(type %s) does not have attr %s, cannot delete.", src_node->GetName().c_str(), src_node->GetType().c_str(), ATTR_NAME_INDEX.c_str());
    return false;
  }

  // 校验内存复用配置
  if (!IsMemoryReuseAllowed(static_cast<int32_t>(input_index), ctx.tensor_move->GetName())) {
    return false;
  }

  return true;
};

DeleteRule CheckSinglePath = [](const TensorMoveDeleteContext &ctx) {
  return IsSourceNodeWithSinglePath(ctx.tensor_move, ctx.path_to_source_node);
};
}

Status TensorMoveDeletePass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());

  if (!IsTensorMove(node) || HasReservedAttr(node)) {
    return SUCCESS;
  }

  TensorMoveDeleteContext ctx{node, {}};

  GE_ASSERT_SUCCESS(TraceRealSourceNode(ctx.tensor_move, 0, ctx.path_to_source_node));
  GE_ASSERT(!ctx.path_to_source_node.empty());
  GE_ASSERT_NOTNULL(ctx.path_to_source_node.back().first);

  std::vector<DeleteRule> rules = {
    CheckPathToSourceNodeValid,
    CheckSourceNodeReuse,
    CheckSinglePath
  };

  for (const auto &rule : rules) {
    if (!rule(ctx)) {
      GELOGD("Node %s(type %s) can not be deleted.", node->GetName().c_str(), node->GetType().c_str());
      return SUCCESS;
    }
  }

  GE_ASSERT_SUCCESS(IsolateAndDeleteNode(node, {0}));
  GELOGI("Node %s(type %s) deleted due to redundant copy.", node->GetName().c_str(), node->GetType().c_str());
  return SUCCESS;
}

REG_PASS_OPTION("TensorMoveDeletePass").LEVELS(OoLevel::kO3);
}  // namespace ge

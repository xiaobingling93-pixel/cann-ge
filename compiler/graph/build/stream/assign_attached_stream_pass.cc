/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "assign_attached_stream_pass.h"

#include "graph/debug/ge_attr_define.h"

namespace {
// 如果force_reuse为true则不拼主流id
std::string CalcuAttachedStreamReuseKey(const std::string &usage_name, const std::string &reuse_key, bool force_reuse,
                                        const ge::OpDescPtr &op_desc) {
  std::string new_reuse_key = std::string().append(reuse_key).append(usage_name);
  if (!force_reuse) {
    new_reuse_key.append("_stream_").append(std::to_string(op_desc->GetStreamId()));
  }
  return new_reuse_key;
}
}
namespace ge {
Status AssignAttachedStreamPass::Run(ComputeGraphPtr graph, const vector<SubgraphPtr> &subgraphs,
                                     LogicalStreamPass::Context &context) {
  (void)subgraphs;
  return Run(graph, context.next_stream);
}

Status AssignAttachedStreamPass::Run(ComputeGraphPtr graph, int64_t &stream_num) {
  GE_ASSERT_NOTNULL(graph);
  GELOGI("Start assign attached stream for graph %s with subgraph num: %zu", graph->GetName().c_str(),
         graph->GetAllSubgraphs().size());
  GE_ASSERT_SUCCESS(AttachedResourceAssignHelper::ClassifyNodesByGroup(
      graph, CheckAndGetAttachedStreamInfo, CheckAndGetAttachedStreamInfoV2, groups_2_nodes_));
  if (groups_2_nodes_.empty()) {
    return SUCCESS;
  }
  GELOGI("Start assign attached stream for graph %s with group num: %zu", graph->GetName().c_str(),
         groups_2_nodes_.size());
  for (const auto &group_2_nodes : groups_2_nodes_) {
    const auto &group_name = group_2_nodes.first;
    const auto &nodes_in_this_group = group_2_nodes.second;
    GE_ASSERT_SUCCESS(AttachedResourceAssignHelper::AssignAttachedResource(nodes_in_this_group, SetAttachedStream,
                                                                           SetAttachedStreamV2, stream_num));
    GELOGI(
        "Assign attached stream successfully for nodes with same group name {%s} in graph %s, total stream number "
        "now is:%ld ",
        group_name.c_str(), graph->GetName().c_str(), stream_num);
  }
  return SUCCESS;
}

// todo:待mc2整改后和GetAttachedStreamInfoByListNamedAttrs归一
Status AssignAttachedStreamPass::GetAttachedStreamInfoByNamedAttrs(
    const OpDescPtr &op_desc, std::vector<AttachedStreamInfo> &attached_stream_info) {
  NamedAttrs attached_stream_info_from_attr;
  if (!(AttrUtils::GetNamedAttrs(op_desc, ATTR_NAME_ATTACHED_STREAM_INFO, attached_stream_info_from_attr))) {
    return SUCCESS;
  }
  AttachedStreamInfo cur_attached_stream_info;
  // 校验是否有设置策略
  GE_ASSERT_TRUE(AttrUtils::GetStr(attached_stream_info_from_attr, ATTR_NAME_ATTACHED_STREAM_POLICY,
                                   cur_attached_stream_info.attached_policy));
  // 当前仅仅支持group级别的分配策略
  GE_ASSERT_EQ(cur_attached_stream_info.attached_policy, GROUP_POLICY);
  // 校验是否有设置group名称
  GE_ASSERT_TRUE(AttrUtils::GetStr(op_desc, GROUP_POLICY, cur_attached_stream_info.attached_group_name));
  // 校验是否有设置从流名称
  GE_ASSERT_TRUE(AttrUtils::GetStr(attached_stream_info_from_attr, ATTR_NAME_ATTACHED_STREAM_KEY,
                                   cur_attached_stream_info.attached_reuse_key));
  GELOGD("Op [%s %s] get [%s] successfully.", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
         cur_attached_stream_info.ToString("stream").c_str());

  attached_stream_info.emplace_back(cur_attached_stream_info);
  // ATTR_NAME_ATTACHED_STREAM_INFO属性的生命周期到这里应该就结束了，为了减少OM体积，删除他
  (void)op_desc->DelAttr(ATTR_NAME_ATTACHED_STREAM_INFO);
  return SUCCESS;
}

Status AssignAttachedStreamPass::GetAttachedStreamInfoByListNamedAttrs(
    const OpDescPtr &op_desc, std::vector<AttachedStreamInfo> &attached_stream_info) {
  std::vector<NamedAttrs> attached_stream_info_from_attr;
  if (!(AttrUtils::GetListNamedAttrs(op_desc, ATTR_NAME_ATTACHED_STREAM_INFO, attached_stream_info_from_attr))) {
    return SUCCESS;
  }

  for (auto &attr : attached_stream_info_from_attr) {
    AttachedStreamInfo cur_attached_stream_info;
    cur_attached_stream_info.attached_policy = GROUP_POLICY;
    cur_attached_stream_info.attached_group_name = DEFAULT_STREAM_INFO_GROUP;
    GE_ASSERT_TRUE(AttrUtils::GetStr(attr, ATTR_NAME_ATTACHED_STREAM_KEY,
                                     cur_attached_stream_info.attached_reuse_key));
    // todo:根据和主流id做组合key，但是mc2可能会存在问题，归一时需关注
    cur_attached_stream_info.attached_reuse_key.append("_stream_")
                                               .append(std::to_string(op_desc->GetStreamId()));
    GELOGD("Op [%s %s] get [%s] successfully.", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
            cur_attached_stream_info.ToString("stream").c_str());
    attached_stream_info.emplace_back(cur_attached_stream_info);
  }
  return SUCCESS;
}

Status AssignAttachedStreamPass::CheckAndGetAttachedStreamInfo(const OpDescPtr &op_desc,
                                                               std::vector<AttachedStreamInfo> &attached_stream_info) {
  GE_ASSERT_SUCCESS(GetAttachedStreamInfoByNamedAttrs(op_desc, attached_stream_info));
  GE_ASSERT_SUCCESS(GetAttachedStreamInfoByListNamedAttrs(op_desc, attached_stream_info));
  return SUCCESS;
}

Status AssignAttachedStreamPass::SetAttachedStream(const OpDescPtr &op_desc, const uint32_t stream_num,
                                                   int64_t &stream_id) {
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(stream_num == 1U, "multi attached streams is currently not supported.");
  op_desc->SetAttachedStreamIds({stream_id});
  GELOGI("Assign attached stream [%ld] for node {%s %s} with main stream [%ld] successfully", stream_id,
         op_desc->GetNamePtr(), op_desc->GetTypePtr(), op_desc->GetStreamId());
  return SUCCESS;
}

Status AssignAttachedStreamPass::CheckAndGetAttachedStreamInfoV2(
    const OpDescPtr &op_desc, std::vector<AttachedStreamInfoV2> &attached_stream_info) {
  // 后续全部归一到此流程
  std::vector<NamedAttrs> attached_stream_info_list_from_attr;
  GELOGD("Try Parser attached stream info list, op %s [%s]", op_desc->GetName().c_str(), op_desc->GetType().c_str());
  if (!AttrUtils::GetListNamedAttrs(op_desc, ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                    attached_stream_info_list_from_attr)) {
    return SUCCESS;
  }

  for (auto &attr : attached_stream_info_list_from_attr) {
    AttachedStreamInfoV2 cur_attached_stream_info;
    // name 必填，reuse_key选填，不填则按照name复用
    GE_ASSERT_TRUE(AttrUtils::GetStr(attr, ATTR_NAME_ATTACHED_RESOURCE_NAME, cur_attached_stream_info.name));
    (void) AttrUtils::GetStr(attr, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, cur_attached_stream_info.reuse_key);
    (void) AttrUtils::GetBool(attr, ATTR_NAME_ATTACHED_RESOURCE_FORCE_REUSE, cur_attached_stream_info.force_reuse);
    // 当主流不一致时，从流不复用，保证最大并发，如果流不够，最终依赖流复用解决
    // 当force_reuse为true时，表示跨主流复用从流
    cur_attached_stream_info.reuse_key =
        CalcuAttachedStreamReuseKey(cur_attached_stream_info.name, cur_attached_stream_info.reuse_key,
                                    cur_attached_stream_info.force_reuse, op_desc);
    (void) AttrUtils::GetBool(attr, ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, cur_attached_stream_info.required);
    GELOGD("Op [%s %s] get [%s] successfully.", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
           cur_attached_stream_info.ToString("stream").c_str());
    attached_stream_info.emplace_back(cur_attached_stream_info);
  }
  // ATTR_NAME_ATTACHED_STREAM_INFO_LIST 生命周期并未结束
  return SUCCESS;
}

Status AssignAttachedStreamPass::SetAttachedStreamV2(const OpDescPtr &op_desc, const std::string &reuse_key,
                                                     int64_t &stream_id) {
  GE_ASSERT_NOTNULL(op_desc);
  std::vector<NamedAttrs> attached_stream_info_list_from_attr;
  GELOGD("Try set attached stream id %d with reuse_key %s, op %s [%s]", stream_id, reuse_key.c_str(),
         op_desc->GetName().c_str(), op_desc->GetType().c_str());
  if (!AttrUtils::GetListNamedAttrs(op_desc, ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                    attached_stream_info_list_from_attr)) {
    return ge::SUCCESS;
  }

  for (auto &attr : attached_stream_info_list_from_attr) {
    std::string attr_name;
    std::string attr_reuse_key;
    bool force_reuse = false;
    // name 必填，reuse_key选填，不填则按照name复用
    GE_ASSERT_TRUE(AttrUtils::GetStr(attr, ATTR_NAME_ATTACHED_RESOURCE_NAME, attr_name));
    (void) AttrUtils::GetStr(attr, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, attr_reuse_key);
    (void) AttrUtils::GetBool(attr, ATTR_NAME_ATTACHED_RESOURCE_FORCE_REUSE, force_reuse);
    GELOGI("Try set stream id, reuse key is %s, reuse_key_from_attr is %s, force_reuse is %d", reuse_key.c_str(),
            attr_reuse_key.c_str(), static_cast<int32_t>(force_reuse));
    const std::string calc_reuse_key = CalcuAttachedStreamReuseKey(attr_name, attr_reuse_key, force_reuse, op_desc);
    if (calc_reuse_key != reuse_key) {
      continue;
    }
    GE_ASSERT_TRUE(AttrUtils::SetInt(attr, ATTR_NAME_ATTACHED_RESOURCE_ID, stream_id));
    GE_ASSERT_TRUE(AttrUtils::SetBool(attr, ATTR_NAME_ATTACHED_RESOURCE_IS_VALID, true));
    GELOGI("Assign attached stream [%ld] for node {%s %s} with main stream [%ld] and reuse_key [%s] successfully",
           stream_id, op_desc->GetNamePtr(), op_desc->GetTypePtr(), op_desc->GetStreamId(), reuse_key.c_str());
  }
  GE_ASSERT_TRUE(
      AttrUtils::SetListNamedAttrs(op_desc, ATTR_NAME_ATTACHED_STREAM_INFO_LIST, attached_stream_info_list_from_attr));

  return SUCCESS;
}
}  // namespace ge

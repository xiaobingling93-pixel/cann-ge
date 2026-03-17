/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "autofuse_utils.h"
#include <queue>
#include <numeric>
#include <google/protobuf/text_format.h>
#include "nlohmann/json.hpp"
#include "autofuse_attrs.h"
#include "utils/graph_utils.h"
#include "common/ge_common/ge_types.h"
#include "utils/node_utils.h"
#include "can_fuse/backend/backend_utils.h"
#include "graph/detail/model_serialize_imp.h"
#include "attribute_group/attr_group_shape_env.h"
#include "graph/utils/type_utils.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "mmpa/mmpa_api.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "ascend_graph_code_dumper.h"
#include "graph/ge_context.h"
#include "post_process/post_process_util.h"

namespace ge {
namespace {
const std::string kComputeGraph = "compute_graph";
const std::string kOutputSymbolShape = "output_symbol_shape";
const std::string kSymbolSourceInfo = "symbol_source_info";
const int64_t kDynamicType = 2;
// dump graph related
constexpr int32_t kBaseOfIntegerValue = 10;
const char_t *const kNpuCollectPath = "NPU_COLLECT_PATH";
const char_t *const kDumpGraphPath = "DUMP_GRAPH_PATH";
const char_t *const kDumpGEGraph = "DUMP_GE_GRAPH";
const char_t *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";

enum class DumpGraphLevel:int64_t {
  kDumpGraphLevel1 = 1,
};
} // namespace

bool IsStrNotNum(const std::string &val) {
  return std::any_of(val.begin(), val.end(), [](char c) { return !isdigit(c); });
}

bool NeedDumpGraphByWhitelist(const std::string &env_val, const std::string &suffix) {
  const auto &whitelist_names = StringUtils::Split(env_val, '|');
  return std::any_of(whitelist_names.begin(), whitelist_names.end(), [&](const std::string &name) {
    return suffix.find(name) != std::string::npos;
  });
}

bool NoNeedDumpGraphBySuffix(const std::string &suffix) {
  char_t dump_graph_level_str[MMPA_MAX_PATH] = {'\0'};
  const int32_t res = mmGetEnv(kDumpGraphLevel, &(dump_graph_level_str[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  if (res != EN_OK) {
    return false;
  }

  const std::string env_val(dump_graph_level_str);
  if (IsStrNotNum(env_val)) {
    return !NeedDumpGraphByWhitelist(env_val, suffix);
  }

  const int64_t dump_graph_level = std::strtol(dump_graph_level_str, nullptr, kBaseOfIntegerValue);
  if (dump_graph_level == static_cast<int64_t>(DumpGraphLevel::kDumpGraphLevel1)) {
    return false;
  }

  return true;
}

bool IsNoNeedDump() {
  char dump_ge_graph[MMPA_MAX_PATH] = {'\0'};
  const int32_t res = mmGetEnv(kDumpGEGraph, &(dump_ge_graph[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  if (res != EN_OK) {
    return true;
  }
  auto dump_level = (dump_ge_graph[0U] != '\0') ? std::strtol(&(dump_ge_graph[0U]), nullptr, kBaseOfIntegerValue)
                                                : static_cast<int64_t>(ge::DumpLevel::NO_DUMP);
  if ((dump_level == static_cast<int64_t>(ge::DumpLevel::NO_DUMP)) ||
      (dump_level >= static_cast<int64_t>(ge::DumpLevel::DUMP_LEVEL_END))) {
    return true;
  }
  return false;
}

bool IsDumpGraphLevel1() {
  char dump_graph_level[MMPA_MAX_PATH] = {'\0'};
  const int32_t res = mmGetEnv(kDumpGraphLevel, &(dump_graph_level[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  if (res != EN_OK) {
    return false;
  }
  auto dump_level = (dump_graph_level[0U] != '\0') ? std::strtol(&(dump_graph_level[0U]), nullptr, kBaseOfIntegerValue)
                                                : static_cast<int64_t>(ge::DumpLevel::NO_DUMP);
  if (dump_level == static_cast<int64_t>(ge::DumpLevel::DUMP_ALL)) {
    return true;
  }
  return false;
}

std::stringstream GetDumpGraphPrefixAndCreateDir(const std::string &module_name) {
  std::stringstream stream_file_name;
  char_t npu_collect_path[MMPA_MAX_PATH] = { '\0' };
  INT32 res = mmGetEnv(kNpuCollectPath, &(npu_collect_path[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  if (res == EN_OK) {
    const std::string base_path_str(npu_collect_path);
    stream_file_name << base_path_str << "/extra-info/graph/" << mmGetPid() << "_" << GetContext().DeviceId() << "/";
  } else {
    char_t dump_graph_path[MMPA_MAX_PATH] = { '\0' };
    res = mmGetEnv(kDumpGraphPath, &(dump_graph_path[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
    if (res == EN_OK) {
      const std::string dump_graph_path_str(dump_graph_path);
      const std::string tmp_path_dir = dump_graph_path_str.empty() ? "" : dump_graph_path_str + "/";
      stream_file_name << tmp_path_dir;
      stream_file_name << "pid_" << mmGetPid() << "_" << "deviceid_" << GetContext().DeviceId() << "/";
    } else {
      stream_file_name << "./";
      std::string ascend_work_path;
      (void)GetAscendWorkPath(ascend_work_path);
      if (!ascend_work_path.empty()) {
        stream_file_name.str("");
        stream_file_name << (ascend_work_path + "/");
      }
    }
  }
  stream_file_name << module_name.c_str();
  if (mmAccess2(stream_file_name.str().c_str(), M_F_OK) != EN_OK) {
    if (CreateDir(stream_file_name.str()) != 0) {
      GELOGW("[DumpGraph][CreateDir] Create dump graph dir failed, path:%s.", stream_file_name.str().c_str());
      stream_file_name.str("");
      stream_file_name << "./";
    }
  }
  return stream_file_name;
}

void AutofuseUtils::DumpGraphToOnnx(const ge::ComputeGraph &compute_graph, const std::string &module_name,
                                    const std::string &suffix) {
  if (IsNoNeedDump() || NoNeedDumpGraphBySuffix(suffix)) {
    return;
  }

  const std::stringstream prefix = GetDumpGraphPrefixAndCreateDir(module_name);
  ge::GraphUtils::DumpGrphToOnnx(compute_graph, prefix.str(), suffix);
}

void AutofuseUtils::DumpGEGraph(const ge::ComputeGraphPtr &graph, const std::string &module_name,
                                const std::string &suffix) {
  if (IsNoNeedDump() || NoNeedDumpGraphBySuffix(suffix)) {
    return;
  }

  const std::stringstream prefix = GetDumpGraphPrefixAndCreateDir(module_name);
  ge::GraphUtils::DumpGEGrph(graph, prefix.str(), suffix);
}

void AutofuseUtils::DumpGEGraphLevel1(const ge::ComputeGraphPtr &graph, const std::string &module_name,
                                      const std::string &suffix) {
  if (IsNoNeedDump() || NoNeedDumpGraphBySuffix(suffix)) {
    return;
  }
  if (!IsDumpGraphLevel1()) {
    return;
  }

  const std::stringstream prefix = GetDumpGraphPrefixAndCreateDir(module_name);
  ge::GraphUtils::DumpGEGrph(graph, prefix.str(), suffix);
}

void AutofuseUtils::DumpGraphToOnnxLevel1(const ge::ComputeGraph &compute_graph, const std::string &module_name,
                                          const std::string &suffix) {
  if (IsNoNeedDump() || NoNeedDumpGraphBySuffix(suffix)) {
    return;
  }

  if (!IsDumpGraphLevel1()) {
    return;
  }

  const std::stringstream prefix = GetDumpGraphPrefixAndCreateDir(module_name);
  ge::GraphUtils::DumpGrphToOnnx(compute_graph, prefix.str(), suffix);
}

thread_local int64_t AutofuseUtils::number = 0;

int64_t AutofuseUtils::GenUniqueNumber() {
  return number++;
}

void AutofuseUtils::ClearUniqueNumber() {
  number = 0;
}

Status AutofuseUtils::AddOperatorPrototypeAttrs(const OpDescPtr &op_desc) {
  std::vector<std::string> input_name_list;
  std::vector<int64_t> input_type_list;
  std::vector<std::string> output_name_list;
  std::vector<int64_t> output_type_list;

  // 初始化输入名称和类型列表
  input_name_list.assign(op_desc->GetAllInputsSize(), "inputs");
  input_type_list.assign(op_desc->GetAllInputsSize(), kDynamicType);
  output_name_list.assign(op_desc->GetAllOutputsDescSize(), "outputs");
  output_type_list.assign(op_desc->GetAllOutputsDescSize(), kDynamicType);

  // 设置属性
  GE_ASSERT_TRUE(AttrUtils::SetListStr(op_desc, "_input_name_list", input_name_list));
  GE_ASSERT_TRUE(AttrUtils::SetListInt(op_desc, "_input_para_type_list", input_type_list));
  GE_ASSERT_TRUE(AttrUtils::SetListStr(op_desc, "_output_name_list", output_name_list));
  GE_ASSERT_TRUE(AttrUtils::SetListInt(op_desc, "_output_para_type_list", output_type_list));
  return SUCCESS;
}

NodePtr AutofuseUtils::ConvertAscBackendNodeToAscGraphNode(const ComputeGraphPtr compute_graph, const NodePtr &node) {
  auto asc_op = OperatorFactory::CreateOperator(node->GetName().c_str(), "AscGraph");
  asc_op.BreakConnect();
  asc_op.DynamicInputRegister("inputs", node->GetAllInDataAnchorsSize());
  asc_op.DynamicOutputRegister("outputs", node->GetAllOutDataAnchorsSize());
  auto asc_graph_op_desc = OpDescUtils::GetOpDescFromOperator(asc_op);
  GE_ASSERT_NOTNULL(asc_graph_op_desc);
  GE_ASSERT_SUCCESS(AddOperatorPrototypeAttrs(asc_graph_op_desc));
  auto new_node = compute_graph->AddNode(asc_graph_op_desc);
  GE_ASSERT_NOTNULL(new_node);
  GE_ASSERT_SUCCESS(new_node->SetOwnerComputeGraph(compute_graph));
  auto asc_graph = BackendUtils::GetNodeFusedAscGraph(node);
  GE_ASSERT_NOTNULL(asc_graph);
  std::string asc_graph_str;
  GE_ASSERT_SUCCESS(AscGraphUtils::SerializeToReadable(*asc_graph, asc_graph_str));
  GE_ASSERT_TRUE(AttrUtils::SetStr(asc_graph_op_desc, "ascgraph", asc_graph_str));
  std::vector<int32_t> node1_output_map(node->GetAllOutDataAnchorsSize());
  const std::vector<int32_t> node2_output_map(0);
  std::iota(node1_output_map.begin(), node1_output_map.end(), 0);
  GE_ASSERT_SUCCESS(
      BackendUtils::CreateNewNodeOutputDescAttr(new_node, node, nullptr, node1_output_map, node2_output_map));
  std::vector<int32_t> node1_input_map(node->GetAllInDataAnchorsSize());
  const std::vector<int32_t> node2_input_map(0);
  std::iota(node1_input_map.begin(), node1_input_map.end(), 0);
  GE_ASSERT_SUCCESS(
      BackendUtils::CreateNewNodeInputDescAttr(new_node, node, nullptr, node1_input_map, node2_input_map));
  return new_node;
}

// 从Node获取GraphID的版本
Status AutofuseUtils::CreateComputeGraphWithGraphID(const ge::NodePtr &node, const std::string &graph_name,
                                                    ComputeGraphPtr &compute_graph) {
  compute_graph = ComGraphMakeShared<ComputeGraph>(graph_name.c_str());
  GE_ASSERT_NOTNULL(compute_graph);
  GE_ASSERT_NOTNULL(node->GetOwnerComputeGraph());
  compute_graph->SetGraphID(node->GetOwnerComputeGraph()->GetGraphID());
  return SUCCESS;
}

// 从Graph获取GraphID的版本
Status AutofuseUtils::CreateComputeGraphWithGraphID(const ComputeGraphPtr &graph, const std::string &graph_name,
                                                    ComputeGraphPtr &compute_graph) {
  compute_graph = ComGraphMakeShared<ComputeGraph>(graph_name.c_str());
  GE_ASSERT_NOTNULL(compute_graph);
  GE_ASSERT_NOTNULL(graph);
  compute_graph->SetGraphID(graph->GetGraphID());
  return SUCCESS;
}

Status AutofuseUtils::SerilizeAscBackend(ge::Node *node_ptr, std::string &output, bool isHash) {
  NodePtr node(node_ptr, [](ge::Node *) {});
  ComputeGraphPtr compute_graph;
  // 给ascbackend node创建子图
  GE_ASSERT_SUCCESS(CreateComputeGraphWithGraphID(node, node->GetName(), compute_graph));

  if ((node->GetType() == kAscBackendType) || (node->GetType() == kAscBackendNoKernelType)) {
    auto new_node = ConvertAscBackendNodeToAscGraphNode(compute_graph, node);
    GE_ASSERT_NOTNULL(new_node);
    std::vector<std::pair<ge::NodePtr, int32_t>> pre_nodes;
    GE_ASSERT_SUCCESS(BackendUtils::CreateSubGraphInput(compute_graph, new_node, node->GetAllInDataAnchorsSize(),
                                                        pre_nodes, false));
    std::vector<uint32_t> node_output_index;
    GE_ASSERT_SUCCESS(GetNodeOutputIndex(node, node_output_index));
    GE_ASSERT_SUCCESS(
        BackendUtils::CreateSubGraphOutput(compute_graph, new_node, node->GetOutDataNodesSize(), node_output_index));
    if (isHash) {
      GE_ASSERT_SUCCESS(RenameInputAndOutputForGraph(compute_graph, new_node));
    }
  } else if (node->GetType() == kFusedAscBackendType) {
    auto fused_compute_graph = BackendUtils::GetNodeFusedComputeGraph(node);
    GE_ASSERT_NOTNULL(fused_compute_graph);
    std::unordered_map<std::string, NodePtr> all_new_nodes;
    for (const auto &asc_graph_node : fused_compute_graph->GetDirectNode()) {
      GE_ASSERT_NOTNULL(asc_graph_node);
      const auto &op_desc = asc_graph_node->GetOpDesc();
      NodePtr new_node;
      if (asc_graph_node->GetType() == kAscBackendType) {
        new_node = ConvertAscBackendNodeToAscGraphNode(compute_graph, asc_graph_node);
        GE_ASSERT_NOTNULL(new_node);
      } else {
        auto new_op_desc = ge::OpDescUtils::GetOpDescFromOperator(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
        GE_ASSERT_NOTNULL(new_op_desc);
        new_node = compute_graph->AddNode(new_op_desc);
      }
      GE_ASSERT_NOTNULL(new_node);
      all_new_nodes[new_node->GetName()] = new_node;
    }
    for (const auto &src_node : fused_compute_graph->GetDirectNode()) {
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RelinkGraphEdges(src_node, "", all_new_nodes));
    }
  } else {
    GELOGD("node %s(%s) no need serialize.", node->GetNamePtr(), node->GetType().c_str());
    return SUCCESS;
  }
  if (!isHash) {
    AutofuseUtils::DumpGraphToOnnx(*compute_graph, kPostProcessDir, compute_graph->GetName() + "_serialize");
    compute_graph->Dump();
  }
  // 序列化并打包compute graph, output symbol shape and symbol source info
  GE_ASSERT_SUCCESS(SerializeAndPackComputeGraph(compute_graph, node, output, isHash));
  return SUCCESS;
}

Status AutofuseUtils::SerializeAndPackComputeGraph(const ComputeGraphPtr &compute_graph,
                                                   const NodePtr &node, std::string &output, bool isHash) {
  nlohmann::json symbol_info;
  auto root_graph = GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());
  GE_ASSERT_NOTNULL(root_graph);
  const auto shape_env_attr = root_graph->GetAttrsGroup<ShapeEnvAttr>();
  GE_ASSERT_NOTNULL(shape_env_attr);
  auto sub_graph_env_attr = compute_graph->GetOrCreateAttrsGroup<ge::ShapeEnvAttr>();
  GE_ASSERT_NOTNULL(sub_graph_env_attr);
  *sub_graph_env_attr = *shape_env_attr;
  if (isHash) {
    // hash场景删除symbolic_to_value及value_to_symbol字符
    sub_graph_env_attr->ClearSymbolValueInfo();
  }
  // 序列化compute graph
  proto::GraphDef graph_proto;
  const ModelSerializeImp serialize_imp;
  GE_ASSERT_TRUE(serialize_imp.SerializeGraph(compute_graph, &graph_proto, false));
  std::string compute_graph_str;
  GE_ASSERT_TRUE(google::protobuf::TextFormat::PrintToString(graph_proto, &compute_graph_str),
                 "SerializeToReadable failed.");

  nlohmann::json output_attr_array = nlohmann::json::array();
  const auto node_op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(node_op_desc);
  for (const auto &output_desc : node_op_desc->GetAllOutputsDescPtr()) {
    const auto node_attr = output_desc->GetAttrsGroup<ge::SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(node_attr);
    std::vector<std::string> ori_symbols;
    for (const auto &ori_symbol : node_attr->symbolic_tensor.GetOriginSymbolShape().GetDims()) {
      ori_symbols.emplace_back(ori_symbol.Serialize().get());
    }
    output_attr_array.push_back(ori_symbols);
  }

  for (const auto &sym_2_src : shape_env_attr->GetAllSym2Src()) {
    std::string sym_str(sym_2_src.first.Serialize().get());
    symbol_info[sym_str.c_str()] = sym_2_src.second->GetSourceStr();
    GELOGD("Serial symbol_to_source, symbol: %s, source: %s", sym_str.c_str(), sym_2_src.second->GetSourceStr().c_str());
  }

  nlohmann::json json_obj;
  json_obj[kComputeGraph] = compute_graph_str;
  json_obj[kOutputSymbolShape] = output_attr_array.dump();
  json_obj[kSymbolSourceInfo] = symbol_info.dump();
  output = json_obj.dump();
  return SUCCESS;
}

Status AutofuseUtils::GetNodeOutputIndex(const NodePtr &node, std::vector<uint32_t> &node_output_index) {
  const auto node_out_node_size = node->GetOutDataNodesSize();
  node_output_index.resize(node_out_node_size);
  auto index = 0U;
  for (auto node_output = 0U; node_output < node->GetAllOutDataAnchorsSize(); node_output++) {
    const auto node_out_anchor = node->GetOutDataAnchor(static_cast<int32_t>(node_output));
    GE_ASSERT_NOTNULL(node_out_anchor);
    const auto out_size = node_out_anchor->GetPeerInDataAnchors().size();
    for (size_t i = 0U; i < out_size; i++) {
      GE_ASSERT_TRUE(index < node_out_node_size, "size %zu VS size %zu.", index, node_out_node_size);
      node_output_index[index++] = node_output;
    }
  }
  return SUCCESS;
}

Status ReplaceNodeNameWithType(NodePtr &node, const ComputeGraphPtr &graph, const CounterPtr &counter) {
  (void)graph;
  std::string new_name;
  std::string delimiter = "/";
  new_name += node->GetOwnerComputeGraph()->GetName() + delimiter;
  // 分别用节点类型替换节点名
  new_name += node->GetType();
  // 保证名字唯一
  if (counter) {
    new_name += "_" + std::to_string(counter->NextId());
  }
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  op_desc->SetName(new_name);
  auto asc_node = std::dynamic_pointer_cast<AscNode>(node);
  if (asc_node) {
    // 如果是AscNode，同时修改属性名
    asc_node->attr.name = new_name;
  }
  return SUCCESS;
}

Status CompleteSymbolicAttrForCopyGraph(NodePtr &node, const ComputeGraphPtr &graph, int32_t node_index) {
  // 当前先手动补充符号化属性
  const auto node_op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(node_op_desc);
  auto origin_node = graph->GetAllNodes().at(node_index);
  GE_ASSERT_NOTNULL(origin_node);
  const auto origin_op_desc = origin_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(origin_op_desc);
  for (size_t index = 0U; index < node_op_desc->GetAllOutputsDescPtr().size(); ++index) {
    auto output_desc = node_op_desc->MutableOutputDesc(index);
    auto node_attr = output_desc->GetAttrsGroup<ge::SymbolicDescAttr>();
    if (node_attr != nullptr) {
      continue;
    }
    GE_ASSERT_TRUE(index < origin_op_desc->GetAllOutputsDescSize(), "index %d is over origin node output desc size %zu",
                   index, origin_op_desc->GetAllOutputsDescSize());
    auto &origin_output_desc = origin_op_desc->GetOutputDesc(index);
    const auto origin_attr = origin_output_desc.GetAttrsGroup<ge::SymbolicDescAttr>();
    auto dst_attr = output_desc->GetOrCreateAttrsGroup<SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(dst_attr);
    dst_attr->symbolic_tensor.MutableOriginSymbolShape() = origin_attr->symbolic_tensor.GetOriginSymbolShape();
    if (origin_attr->symbolic_tensor.GetSymbolicValue() != nullptr) {
      auto symbolic_value = std::make_unique<std::vector<Expression>>(*origin_attr->symbolic_tensor.GetSymbolicValue());
      if (symbolic_value != nullptr) {
        dst_attr->symbolic_tensor.SetSymbolicValue(std::move(symbolic_value));
      }
    }
  }
  return SUCCESS;
}

Status ReplaceNodeNameWithTypeForAscGraph(NodePtr &node, const ComputeGraphPtr &graph, const CounterPtr &counter,
                                          int32_t index) {
  auto origin_node = graph->GetAllNodes().at(index);
  GE_ASSERT_NOTNULL(origin_node);
  auto asc_graph = BackendUtils::GetNodeFusedAscGraph(origin_node);
  GE_ASSERT_NOTNULL(asc_graph);
  std::string temp_graph_shared_name = "HashCopyAscGraph";
  auto temp_graph_shared = ComGraphMakeShared<AscGraph>(temp_graph_shared_name.c_str());
  GE_ASSERT_NOTNULL(temp_graph_shared);
  temp_graph_shared->CopyFrom(*asc_graph);
  GE_ASSERT_NOTNULL(origin_node->GetOpDesc());
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  node->GetOpDesc()->CopyAttrsFrom(*(origin_node->GetOpDesc()));
  auto fuse_attrs = BackendUtils::GetNodeAutoFuseAttr(node);
  GE_ASSERT_NOTNULL(fuse_attrs);
  fuse_attrs->SetAscGraph(temp_graph_shared, fuse_attrs->GetFuseType());
  for (NodePtr node_in_ascgraph : temp_graph_shared->GetAllNodes()) {
    // 修改子图上节点名字
    GE_ASSERT_SUCCESS(ReplaceNodeNameWithType(node_in_ascgraph, AscGraphUtils::GetComputeGraph(*asc_graph), counter));
  }
  return SUCCESS;
}

Status AutofuseUtils::CopyGraphAndRenameNode(const ComputeGraphPtr &graph, ComputeGraphPtr &copy_graph,
                                             const CounterPtr &counter) {
  // 拷贝原图
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::CopyComputeGraph(graph, copy_graph));
  copy_graph->CopyAttrsFrom(*graph);
  int32_t index = -1;
  for (auto &node : copy_graph->GetAllNodes()) {
    index++;
    if ((node->GetType() == kAscBackendType) || (node->GetType() == kAscBackendNoKernelType)) {
      // AscGraph内部节点使用新的counter计数保证生成node场景复用
      auto new_counter = ComGraphMakeShared<DefaultCounter>();
      GE_ASSERT_NOTNULL(new_counter);
      CounterPtr new_counter_ptr = new_counter.get();
      GE_ASSERT_SUCCESS(ReplaceNodeNameWithTypeForAscGraph(node, graph, new_counter_ptr, index));
      GE_ASSERT_SUCCESS(CompleteSymbolicAttrForCopyGraph(node, graph, index));
    } else if (node->GetType() == kFusedAscBackendType) {
      // AscGraph内部节点使用新的counter计数保证生成node场景复用
      auto new_counter = ComGraphMakeShared<DefaultCounter>();
      GE_ASSERT_NOTNULL(new_counter);
      CounterPtr new_counter_ptr = new_counter.get();
      auto origin_node = graph->GetAllNodes().at(index);
      GE_ASSERT_NOTNULL(origin_node);
      auto fused_compute_graph = BackendUtils::GetNodeFusedComputeGraph(origin_node);
      GE_ASSERT_NOTNULL(fused_compute_graph);
      GE_ASSERT_NOTNULL(node->GetOpDesc());
      GE_ASSERT_NOTNULL(origin_node->GetOpDesc());
      node->GetOpDesc()->CopyAttrsFrom(*(origin_node->GetOpDesc()));
      GE_ASSERT_SUCCESS(CompleteSymbolicAttrForCopyGraph(node, graph, index));
      std::string copy_fused_graph_name = "HashCopyFusedAscGraph";
      ComputeGraphPtr copy_fused_graph;
      GE_ASSERT_SUCCESS(CreateComputeGraphWithGraphID(node, copy_fused_graph_name, copy_fused_graph));
      GE_ASSERT_SUCCESS(AutofuseUtils::CopyGraphAndRenameNode(fused_compute_graph, copy_fused_graph, new_counter_ptr));
      auto fuse_attrs = BackendUtils::GetNodeAutoFuseAttr(node);
      GE_ASSERT_NOTNULL(fuse_attrs);
      fuse_attrs->SetFuseComputeGraph(copy_fused_graph);
    }
    GE_ASSERT_SUCCESS(ReplaceNodeNameWithType(node, graph, counter));
  }
  return SUCCESS;
}

Status AutofuseUtils::RenameInputAndOutputForGraph(ComputeGraphPtr &graph, const NodePtr &node) {
  /**
   * 找到图中输入节点和NetOutput节点，分别进行改名
   */
  // AscGraph内部节点使用新的counter计数保证生成node场景复用
  auto new_counter = ComGraphMakeShared<DefaultCounter>();
  GE_ASSERT_NOTNULL(new_counter);
  CounterPtr new_counter_ptr = new_counter.get();
  std::vector<NodePtr> peer_out_nodes;
  GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNodes(node, peer_out_nodes));
  for (auto &data_node : peer_out_nodes) {
    GE_ASSERT_SUCCESS(ReplaceNodeNameWithType(data_node, graph, new_counter_ptr));
  }
  auto netoutput_node = graph->GetOrUpdateNetOutputNode();
  GE_ASSERT_NOTNULL(netoutput_node);
  GE_ASSERT_SUCCESS(ReplaceNodeNameWithType(netoutput_node, graph, new_counter_ptr));
  return SUCCESS;
}

bool AutofuseUtils::IsUbScalar(const std::vector<ge::Expression> &repeats) {
  return std::all_of(repeats.begin(), repeats.end(), [](const ge::Expression &repeat) {
    return ge::SymbolicUtils::StaticCheckEq(repeat, ge::sym::kSymbolOne) == ge::TriBool::kTrue;
  });
}

bool AutofuseUtils::IsSplitType(const std::string &node_type) {
  return std::find(SPLIT_TYPES.begin(), SPLIT_TYPES.end(), node_type) != SPLIT_TYPES.end();
}

Status AutofuseUtils::DelOneNodeInGraph(const ComputeGraphPtr &graph, const NodePtr &node) {
// 把node前后的节点连接起来
  const auto in_data_anchor = node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(in_data_anchor);
  const auto out_data_anchor = node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(out_data_anchor);
  const auto src_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src_anchor);
  GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::RemoveEdge(src_anchor, in_data_anchor));
  for (const auto &dst_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    GE_ASSERT_NOTNULL(dst_anchor);
    GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::RemoveEdge(out_data_anchor, dst_anchor));
    GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::AddEdge(src_anchor, dst_anchor));
  }

// 删除del node和边
  GELOGD("Remove node: %s(%s) from graph:%s.", node->GetName().c_str(), node->GetType().c_str(),
         graph->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveJustNode(graph, node),
                          "[Remove][JustNode] failed, graph:%s, node:%s.", graph->GetName().c_str(),
                          node->GetName().c_str());
  NodeUtils::UnlinkAll(*node);
  return SUCCESS;
}

// 检测是否存在[a,b,c]->[a*b,c]的情况, 从第一个未找到shape开始，到shape大小相同为止，划定下标范围。
bool AutofuseUtils::CheckAndMulDetect(const std::vector<Expression> &long_dims,
                                      const std::vector<Expression> &short_dims, size_t &sort_idx,
                                      std::vector<size_t> &mul_idx) {
  size_t dims_idx = 0U;
  mul_idx.clear();
  Expression total = Symbol(1);
  for (size_t i = 0U; i < short_dims.size(); ++i) {
    while (dims_idx < long_dims.size()) {
      if (((i < short_dims.size()) && (SymbolicUtils::StaticCheckEq(short_dims[i], long_dims[dims_idx]) == ge::TriBool::kTrue)) &&
          (SymbolicUtils::StaticCheckEq(total, Symbol(1)) == ge::TriBool::kTrue ||
           SymbolicUtils::StaticCheckEq(total, short_dims[sort_idx]) == ge::TriBool::kTrue)) {
        ++dims_idx;
        break;
      }
      if (mul_idx.empty()) {
        sort_idx = i;
        ++i;
      } else if (((dims_idx > 0U) && (mul_idx[mul_idx.size() - 1] != dims_idx - 1)) ||
                 SymbolicUtils::StaticCheckEq(long_dims[dims_idx], Symbol(1)) == ge::TriBool::kTrue) {
        return false;
      }
      mul_idx.push_back(dims_idx);
      total = total * long_dims[dims_idx];
      auto chk = SymbolicUtils::StaticCheckGt(total, short_dims[sort_idx]);
      auto chk_is_one = SymbolicUtils::StaticCheckEq(Symbol(1), long_dims[dims_idx]);
      if (chk == ge::TriBool::kTrue || chk == ge::TriBool::kUnknown || chk_is_one == ge::TriBool::kTrue) {
        return false;
      }

      ++dims_idx;
    }
  }
  if (dims_idx < long_dims.size()) {
    return false;
  }

  return SymbolicUtils::StaticCheckEq(total, short_dims[sort_idx]) == ge::TriBool::kTrue;
}

bool AutofuseUtils::IsCubeNodeType(const NodePtr &node) {
  static const std::unordered_map<std::string, std::string> kCubeTypeList = {
      {kMatMul, kMatMul},
      {kMatMulBias, kMatMulBias},
      {kMatMulOffset, kMatMulOffset},
      {kMatMulOffsetBias, kMatMulOffsetBias},
      {kBatchMatMul, kBatchMatMul},
      {kBatchMatMulBias, kBatchMatMulBias},
      {kBatchMatMulOffset, kBatchMatMulOffset},
      {kBatchMatMulOffsetBias, kBatchMatMulOffsetBias}};

  auto node_type = node->GetType();
  auto it = kCubeTypeList.find(node_type);
  return (it != kCubeTypeList.end());
}

graphStatus AutofuseUtils::GetListIntFromInput(const NodePtr &node, std::vector<int64_t> &value_vec,
                                               const std::string &input) {
  GE_ASSERT_NOTNULL(node);
  const auto op = ge::OpDescUtils::CreateOperatorFromNode(node);
  ge::Tensor val_tensor;
  if (op.GetInputConstData(input.c_str(), val_tensor) != ge::SUCCESS) {
    GELOGI("Force skip lowering node %s %s as failed to get tensor", node->GetNamePtr(), node->GetTypePtr());
    return GRAPH_FAILED;
  }
  const auto dims = val_tensor.GetTensorDesc().GetShape().GetDims();
  GE_ASSERT_TRUE(dims.size() <= 1U, "Input tensor must be a scalar or vector");
  const auto num_dim = dims.empty() ? 1L : dims[0];
  GE_ASSERT_TRUE(num_dim >= 0L, "Input tensor must be positive");
  GE_ASSERT_NOTNULL(val_tensor.GetData());
  const ge::DataType dtype = val_tensor.GetTensorDesc().GetDataType();
  for (int64_t idx = 0L; idx < num_dim; ++idx) {
    if (dtype == ge::DT_INT32) {
      int32_t tensor_val = *reinterpret_cast<const int32_t *>(val_tensor.GetData() + idx * sizeof(int32_t));
      value_vec.emplace_back(tensor_val);
    } else if (dtype == ge::DT_INT64) {
      int64_t tensor_val = *reinterpret_cast<const int64_t *>(val_tensor.GetData() + idx * sizeof(int64_t));
      value_vec.emplace_back(tensor_val);
    } else {
      GELOGW("Input tensor must be int32 or int64");
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus AutofuseUtils::GetListIntFromAttr(const NodePtr &node, std::vector<int64_t> &value_vec,
                                              const std::string &attr_name) {
  int64_t attr_val = 0;
  GE_ASSERT_NOTNULL(node);
  if (AttrUtils::GetInt(node->GetOpDesc(), attr_name, attr_val)) {
    value_vec.emplace_back(attr_val);
    return GRAPH_SUCCESS;
  }
  if (AttrUtils::GetListInt(node->GetOpDesc(), attr_name, value_vec)) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

graphStatus AutofuseUtils::GetListIntByInputOrAttr(const NodePtr &node, std::vector<int64_t> &value_vec,
                                                   const std::string &input, const std::string &attr) {
  GE_ASSERT_NOTNULL(node);
  if (GetListIntFromAttr(node, value_vec, attr) == GRAPH_SUCCESS) {
    return GRAPH_SUCCESS;
  }
  return GetListIntFromInput(node, value_vec, input);
}

static std::vector<std::string> view_type = {"ExpandDims", "Reshape", "Squeeze", "Unsqueeze"};
std::vector<const ge::Node *> AutofuseUtils::GetComputeOps(const std::vector<const ge::Node *> &nodes) {
  std::vector<const ge::Node *> compute_ops;
  for (auto &node : nodes) {
    GELOGD("check %s(%s) ComputeOps", node->GetType().c_str(), node->GetName().c_str());
    if (find(view_type.begin(), view_type.end(), node->GetType()) == view_type.end()) {
      compute_ops.emplace_back(node);
    }
  }
  return compute_ops;
}

}  // namespace ge
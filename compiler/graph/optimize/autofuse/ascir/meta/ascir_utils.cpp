/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ascir_utils.h"

#include <sstream>
#include <fstream>
#include <iomanip>
#include <queue>
#include "graph/utils/type_utils.h"
#include "graph_utils.h"
#include "mmpa/mmpa_api.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "ascend_graph_code_dumper.h"
#include "asc_graph_dumper_context.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "common_utils.h"
#include "ascir_ops_utils.h"

namespace {
constexpr int32_t DUMP_ID_WIDTH = 8;
constexpr int32_t kBaseOfIntegerValue = 10;
const char *const kDumpGEGraph = "DUMP_GE_GRAPH";
const char *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";
enum class DumpGraphLevel : int64_t {
  kDumpAllGraph = 1,
  kDumpInWhiteList = 2,
};

bool GetConcatDim(const ge::AscNode &node, size_t &concat_dim) {
  auto node_inputs = node.inputs;
  auto node_outputs = node.outputs;
  const auto &input_repeats = node_inputs[0].attr.repeats;
  const auto &output_repeats = node_outputs[0].attr.repeats;
  GE_CHK_BOOL_RET_SPECIAL_STATUS(input_repeats.size() != output_repeats.size(), false,
                                 "[%s] input_repeats.size() = %zu, mismatches output_repeats.size() = %zu",
                                 node.GetNamePtr(), input_repeats.size(), output_repeats.size());
  for (size_t i = 0U; i < input_repeats.size(); ++i) {
    if (ge::SymbolicUtils::StaticCheckEq(input_repeats[i], output_repeats[i]) != ge::TriBool::kTrue) {
      concat_dim = i;
      return true;
    }
  }
  return false;
}

bool IsStoreWithoutStride(const ge::AscNode &node) {
  std::set<ge::Node *> visited_nodes;
  std::queue<ge::NodePtr> next_nodes;
  std::vector<ge::AscNodePtr> store_nodes;
  for (const auto &out_data_node : node.GetOutDataNodes()) {
    if (visited_nodes.emplace(out_data_node.get()).second) {
      next_nodes.push(out_data_node);
    }
  }
  while (!next_nodes.empty()) {
    auto &top = next_nodes.front();
    auto asc_node = std::dynamic_pointer_cast<ge::AscNode>(top);
    GE_ASSERT_NOTNULL(asc_node);
    if (asc_node->attr.api.compute_type == ge::ComputeType::kComputeStore) {
      store_nodes.emplace_back(asc_node);
    } else {
      for (const auto &out_node : top->GetOutDataNodes()) {
        if (visited_nodes.emplace(out_node.get()).second) {
          next_nodes.emplace(out_node);
        }
      }
    }
    next_nodes.pop();
  }

  for (auto &peer_node : store_nodes) {
    const auto &output_tensor_attr = peer_node->outputs[0].attr;
    GELOGI("%s output_repeat = %s, output_stride = %s", peer_node->GetNamePtr(),
           ge::ToString(output_tensor_attr.repeats).c_str(), ge::ToString(output_tensor_attr.strides).c_str());
    size_t concat_dim;
    GE_WARN_ASSERT(GetConcatDim(node, concat_dim));
    GE_WARN_ASSERT((concat_dim > 0) && (concat_dim < output_tensor_attr.repeats.size()),
                   "concat_dim output range, concat_dim = %zu, repeats = %s, ",
                   concat_dim,
                   ge::ToString(output_tensor_attr.repeats).c_str());
    ge::Expression elt_num = output_tensor_attr.repeats[concat_dim];
    for (size_t i = concat_dim + 1UL; i < output_tensor_attr.repeats.size(); ++i) {
      elt_num = elt_num * output_tensor_attr.repeats[i];
    }
    if (ge::SymbolicUtils::StaticCheckEq(elt_num, output_tensor_attr.strides[concat_dim - 1U]) != ge::TriBool::kTrue) {
      return false;
    }
  }
  return true;
}

bool GetConcatDimAndColSizes(const ge::AscNode &node,
                             size_t &concat_dim,
                             std::vector<int64_t> &src_col_sizes,
                             int64_t &dst_col_size) {
  concat_dim = std::numeric_limits<size_t>::max();
  GE_WARN_ASSERT(GetConcatDim(node, concat_dim));

  auto node_inputs = node.inputs;
  auto node_outputs = node.outputs;
  const auto &input_repeats = node_inputs[0].attr.repeats;
  const auto &output_repeats = node_outputs[0].attr.repeats;
  GE_WARN_ASSERT(concat_dim < input_repeats.size());
  int64_t concat_dim_stride = 1;
  for (size_t i = concat_dim + 1; i < input_repeats.size(); ++i) {
    const auto &dim_size_expr = input_repeats[i];
    GE_CHK_BOOL_RET_SPECIAL_STATUS((!dim_size_expr.IsConstExpr()), false,
                                   "[%s] dynamic dim after concat dim, inputs = %s, outputs = %s", node.GetNamePtr(),
                                   ge::ToString(input_repeats).c_str(), ge::ToString(output_repeats).c_str());
    int64_t dim_size = -1;
    (void) dim_size_expr.GetConstValue(dim_size);
    concat_dim_stride *= dim_size;
  }
  GELOGD("[%s] inputs = %s, output = %s, concat_dim = %u, concat_dim_stride = %ld",
         node.GetNamePtr(), ge::ToString(input_repeats).c_str(), ge::ToString(output_repeats).c_str(),
         concat_dim, concat_dim_stride);
  for (uint32_t i = 0U; i < node_inputs.Size(); ++i) {
    const auto &dim_size_expr = node_inputs[i].attr.repeats[concat_dim];
    GE_CHK_BOOL_RET_SPECIAL_STATUS((!dim_size_expr.IsConstExpr()), false,
                                   "[%s] input[%u] concat dim = %s, not a static dim", node.GetNamePtr(), i,
                                   dim_size_expr.Str().get());
    int64_t dim_size = -1;
    (void) dim_size_expr.GetConstValue(dim_size);
    src_col_sizes.emplace_back(dim_size * concat_dim_stride);
  }

  const auto &output_dim_size_expr = node_outputs[0].attr.repeats[concat_dim];
  GE_CHK_BOOL_RET_SPECIAL_STATUS((!output_dim_size_expr.IsConstExpr()), false,
                                 "[%s] output concat dim = %s, not a static dim", node.GetNamePtr(),
                                 output_dim_size_expr.Str().get());
  int64_t output_dim_size = -1L;
  (void) output_dim_size_expr.GetConstValue(output_dim_size);
  dst_col_size = output_dim_size * concat_dim_stride;
  GELOGD("src_col_sizes = %s, dst_col_size = %ld", ge::ToString(src_col_sizes).c_str(), dst_col_size);
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
      (dump_level >= static_cast<int64_t>(ge::DumpLevel::DUMP_WITH_OUT_DESC))) {
    return true;
  }
  return false;
}

bool CheckNeedDumpGraphBySuffix(const std::string &suffix) {
  char dump_graph_level_str[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv(kDumpGraphLevel, &(dump_graph_level_str[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));

  auto dump_level = (dump_graph_level_str[0U] != '\0')
                        ? std::strtol(&(dump_graph_level_str[0U]), nullptr, kBaseOfIntegerValue)
                        : static_cast<int64_t>(DumpGraphLevel::kDumpInWhiteList);
  if (dump_level == static_cast<int64_t>(DumpGraphLevel::kDumpAllGraph)) {
    return true;
  } else if (dump_level == static_cast<int64_t>(DumpGraphLevel::kDumpInWhiteList)) {
    static std::set<std::string> white_lists = {"AutoFuseBeforeOptimize", "AutoFuseAfterOptimize"};
    return std::any_of(white_lists.begin(), white_lists.end(), [&suffix](const std::string &white_list) {
      return suffix.find(white_list) != std::string::npos;
    });
  }
  return false;
}
}  // namespace

namespace ascir::utils {
static uint64_t kDumpGraphIndex = 0UL;
static std::string DtypeToStr(ge::DataType dtype) {
  const char *kTypeName[] = {
      [ge::DT_FLOAT] = "float32", [ge::DT_FLOAT16] = "float16", [ge::DT_INT8] = "int8_t",
      [ge::DT_INT32] = "int32_t", [ge::DT_UINT8] = "uint8_t",   "",
      [ge::DT_INT16] = "int16_t", [ge::DT_UINT16] = "uint16_t", [ge::DT_UINT32] = "uint32_t",
      [ge::DT_INT64] = "int64_t", [ge::DT_UINT64] = "uint64_t",
  };

  if (dtype >= sizeof(kTypeName) / sizeof(kTypeName[0])) {
    return ge::TypeUtils::DataTypeToSerialString(dtype);
  }
  return kTypeName[dtype];
};

std::stringstream GetDumpGraphPrefixAndCreateDir() {
  std::stringstream stream_file_name;
  char dump_graph_path[MMPA_MAX_PATH] = {'\0'};
  auto res = mmGetEnv("DUMP_GRAPH_PATH", &(dump_graph_path[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  if (res == EN_OK) {
    const std::string dump_graph_path_str(dump_graph_path);
    stream_file_name << (dump_graph_path_str.empty() ? "" : dump_graph_path_str + "/");
  } else {
    stream_file_name << "./";
    std::string ascend_work_path;
    (void)ge::GetAscendWorkPath(ascend_work_path);
    if (!ascend_work_path.empty()) {
      stream_file_name.str("");
      stream_file_name << (ascend_work_path + "/");
    }
  }
  stream_file_name << "ascgen_dump_pid_";
  stream_file_name << mmGetPid() << "/";
  if (mmAccess2(stream_file_name.str().c_str(), M_F_OK) != EN_OK) {
    if (ge::CreateDir(stream_file_name.str()) != 0) {
      GELOGW("[DumpGraph][CreateDir] Create dump graph dir failed, path:%s", stream_file_name.str().c_str());
      stream_file_name.str("");
      stream_file_name << "./";
    }
  }
  return stream_file_name;
}

static std::string ExecConditionToStr(ge::ExecuteCondition condition) {
  static const std::map<ge::ExecuteCondition, std::string> kTypeToStr = {
    {ge::ExecuteCondition::kNoCache, "no_cache"},
    {ge::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis, "cache_block_split_fused_brc_axis"},
    {ge::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis, "cache_block_split_origin_brc_axis"},
    {ge::ExecuteCondition::kConditionInvalid, "invalid"}};

  auto it = kTypeToStr.find(condition);
  if (it != kTypeToStr.end()) {
    return it->second;
  }
  return "unknown";
}

static std::string ComputeUnitToStr(ge::ComputeUnit compute_unit) {
  const char *kTypeName[] = {
      [static_cast<int32_t>(ge::ComputeUnit::kUnitNone)] = "None",
      [static_cast<int32_t>(ge::ComputeUnit::kUnitMTE1)] = "MTE1",
      [static_cast<int32_t>(ge::ComputeUnit::kUnitMTE2)] = "MTE2",
      [static_cast<int32_t>(ge::ComputeUnit::kUnitMTE3)] = "MTE3",
      [static_cast<int32_t>(ge::ComputeUnit::kUnitScalar)] = "Scalar",
      [static_cast<int32_t>(ge::ComputeUnit::kUnitVector)] = "Vector",
      [static_cast<int32_t>(ge::ComputeUnit::kUnitCube)] = "Cube",
  };

  if (static_cast<size_t>(compute_unit) >= sizeof(kTypeName) / sizeof(kTypeName[0])) {
    return "unknown";
  }
  return kTypeName[static_cast<size_t>(compute_unit)];
}

static std::string MemHardwareToStr(ge::MemHardware mem_hardware) {
  const char *kTypeName[] = {
      [static_cast<int32_t>(ge::MemHardware::kMemHardwareGM)] = "GM",
      [static_cast<int32_t>(ge::MemHardware::kMemHardwareUB)] = "UB",
  };

  if (static_cast<size_t>(mem_hardware) >= sizeof(kTypeName) / sizeof(kTypeName[0])) {
    return "unknown";
  }
  return kTypeName[static_cast<size_t>(mem_hardware)];
}

static std::string PositionToStr(ge::Position position) {
  static const std::map<ge::Position, std::string> gPositionToStr = {
      {ge::Position::kPositionGM, "TPosition::GM"},
      {ge::Position::kPositionVecIn, "TPosition::VECIN"},
      {ge::Position::kPositionVecCalc, "TPosition::VECCALC"},
      {ge::Position::kPositionVecOut, "TPosition::VECOUT"}};
  auto it = gPositionToStr.find(position);
  if (it != gPositionToStr.end()) {
    return it->second;
  }
  return "unknown";
}

static std::map<ge::AxisId, std::string> GetAxisIdToName(const std::vector<ge::AxisPtr> &axes) {
  std::map<ge::AxisId, std::string> id_to_name;
  for (const auto &iter : axes) {
    id_to_name[iter->id] = iter->name;
  }
  return id_to_name;
}

static std::stringstream &GraphNameStr(std::stringstream &ss, const ascir::Graph &graph) {
  ss << "Graph: " << graph.GetName() << std::endl;
  return ss;
}

static std::stringstream &GraphSizeStr(std::stringstream &ss, const ascir::Graph &graph) {
  ss << "Sizes:" << std::endl;
  auto all_size_var = graph.GetAllSizeVar();
  for (const auto &size_var : all_size_var) {
    if (size_var->expr.GetExprType() == ge::ExprType::kExprVariable) {
      ss << "  " << size_var->expr.Str().get() << ": VAR" << std::endl;
    } else if (size_var->expr.GetExprType() <= ge::ExprType::kExprConstantRation) {
      int64_t val;
      size_var->expr.GetConstValue(val);
      ss << "  " << size_var->expr.Str().get() << ": CONST(" << val << ")" << std::endl;
    } else {
      //
    }
  }

  return ss;
}

static std::stringstream &GraphAxisStr(std::stringstream &ss, const ascir::Graph &graph) {
  static const std::map<ge::Axis::Type, std::string> kTypeToStr = {
      {ge::Axis::Type::kAxisTypeOriginal, "ORIGINAL"},   {ge::Axis::Type::kAxisTypeBlockOuter, "BLOCK_OUT"},
      {ge::Axis::Type::kAxisTypeBlockInner, "BLOCK_IN"}, {ge::Axis::Type::kAxisTypeTileOuter, "TILE_OUT"},
      {ge::Axis::Type::kAxisTypeTileInner, "TILE_IN"},   {ge::Axis::Type::kAxisTypeMerged, "MERGED"},
  };

  ss << "Axis:" << std::endl;
  auto all_axis = graph.GetAllAxis();
  std::map<ge::AxisId, std::string> axis_id_to_name = GetAxisIdToName(all_axis);
  for (auto &axis : all_axis) {
    ss << "  " << axis->name << "(" << axis->id << ") : ";
    auto iter = kTypeToStr.find(axis->type);
    if (iter != kTypeToStr.end()) {
      ss << iter->second;
    } else {
      ss << "UNKNOWN";
    }
    ss <<", size:"<< ge::SymbolicUtils::ToString(axis->size) << ", ";

    if (!axis->from.empty()) {
      ss << ", from: {";
      for (auto from_axis : axis->from) {
        ss << axis_id_to_name[from_axis] << ", ";
      }
      ss << "}";
    }

    ss << std::endl;
  }
  return ss;
}

static std::stringstream &NodeAttrStr(std::stringstream &ss, const ascir::Graph &graph, ascir::NodeView &node,
                                      bool verbose = false) {
  auto &ir_attr = node->GetOpDesc()->GetAttrsGroup<ge::AscNodeAttr>()->ir_attr;
  if (ir_attr != nullptr) {
    ascendc_ir::proto::AscIrAttrDef asc_ir_attr_def;
    (void)ir_attr->Serialize(asc_ir_attr_def);
    if (!asc_ir_attr_def.attr().empty()) {
      ss << "    .ir_attr =  {";
      for (const auto &pair : asc_ir_attr_def.attr()) {
        ss << "." << pair.first << " = " << pair.second.ShortDebugString();
      }
      ss << "}" << std::endl;
    }
  }

  auto all_axis = graph.GetAllAxis();
  std::map<ge::AxisId, std::string> axis_id_to_name = GetAxisIdToName(all_axis);

  bool is_buf = (node->attr.api.type == ge::ApiType::kAPITypeBuffer);

  if (!is_buf) {
    ss << "    .axis = "
       << "{";
    for (auto axis_id : node->attr.sched.axis) {
      ss << axis_id_to_name[axis_id] << ", ";
    }
    ss << "}" << std::endl;
  }

  // Node sched exec_condition (只显示非默认值)
  if (node->attr.sched.exec_condition != ge::ExecuteCondition::kNoCache) {
    ss << "    .exec_condition = " << ExecConditionToStr(node->attr.sched.exec_condition) << std::endl;
  }

  if (verbose && !is_buf) {
    const auto loop_axis = node->attr.sched.loop_axis;
    if ((loop_axis >= 0) && (loop_axis < static_cast<int64_t>(all_axis.size()))) {
      ss << "    .loop_axis = " << axis_id_to_name[loop_axis] << std::endl;
    } else {
      ss << "    .loop_axis = " << std::to_string(loop_axis) << std::endl;
    }
  }

  if (verbose && !is_buf) {
    ss << "    .api.unit = " << ComputeUnitToStr(node->attr.api.unit) << std::endl;
    const auto &tmp_buffers = node->attr.tmp_buffers;
    if (!tmp_buffers.empty()) {
      ss << "    .tmp_buf = {";
      for (size_t i = 0; i < tmp_buffers.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << "{buf_id=" << tmp_buffers[i].id << ", size=" << ge::SymbolicUtils::ToString(tmp_buffers[i].buf_desc.size) << "}";
      }
      ss << "}" << std::endl;
    }
  }

  return ss;
}

static std::vector<std::string> CollectInputNames(const ascir::Graph &graph, const ge::AscNodePtr &node) {
  (void)graph;
  std::vector<std::string> input_names;

  for (uint32_t index = 0U; index < node->GetAllInDataAnchorsSize(); index++) {
    auto in_anchor = node->GetInDataAnchor(static_cast<int32_t>(index));
    if (in_anchor == nullptr) {
      input_names.push_back("nil");
      continue;
    }
    auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      input_names.push_back("nil");
    } else {
      auto peer_name = peer_out_anchor->GetOwnerNode()->GetName();
      input_names.push_back(peer_name + ".y");
    }
  }
  return input_names;
}

static std::stringstream &NodeInputStr(std::stringstream &ss, const std::vector<std::string> &input_names) {
  bool all_nil = true;
  for (const auto &name : input_names) {
    if (name != "nil") {
      all_nil = false;
      break;
    }
  }
  if (all_nil) {
    return ss;
  }

  if (input_names.size() == 1) {
    ss << "    .x = " << input_names[0] << std::endl;
  } else {
    ss << "    .x = {";
    for (size_t i = 0; i < input_names.size(); ++i) {
      if (i > 0) ss << ", ";
      if (input_names[i] != "nil") {
        ss << input_names[i];
      }
    }
    ss << "}" << std::endl;
  }

  return ss;
}

// 输出形状信息 (axis, repeats, strides)
static std::stringstream &OutputShapeStr(std::stringstream &ss, const ascir::Graph &graph,
                                         const ge::AscTensorAttr &output_attr) {
  // 输出 axis 列表
  if (!output_attr.axis.empty()) {
    auto all_axis = graph.GetAllAxis();
    std::map<ge::AxisId, std::string> axis_id_to_name = GetAxisIdToName(all_axis);
    ss << "        .axis = {";
    for (size_t i = 0; i < output_attr.axis.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << axis_id_to_name[output_attr.axis[i]];
    }
    ss << "}" << std::endl;
  }

  // 输出 repeats
  if (!output_attr.repeats.empty()) {
    ss << "        .repeats = (";
    for (size_t i = 0; i < output_attr.repeats.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << ge::SymbolicUtils::ToString(output_attr.repeats[i]);
    }
    ss << ")" << std::endl;
  }

  // 输出 strides
  if (!output_attr.strides.empty()) {
    ss << "        .strides = (";
    for (size_t i = 0; i < output_attr.strides.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << ge::SymbolicUtils::ToString(output_attr.strides[i]);
    }
    ss << ")" << std::endl;
  }

  return ss;
}

// 输出 vectorized 信息
static std::stringstream &OutputVectorizedStr(std::stringstream &ss, const ascir::Graph &graph,
                                              const ge::AscTensorAttr &output_attr) {
  if (!output_attr.vectorized_axis.empty()) {
    auto all_axis = graph.GetAllAxis();
    std::map<ge::AxisId, std::string> axis_id_to_name = GetAxisIdToName(all_axis);

    ss << "        .vectorized = {";
    for (size_t i = 0; i < output_attr.vectorized_axis.size(); ++i) {
      if (i > 0) ss << ", ";
      auto axis_name = axis_id_to_name[output_attr.vectorized_axis[i]];
      ss << axis_name << ":";
      if (i < output_attr.vectorized_strides.size()) {
        ss << ge::SymbolicUtils::ToString(output_attr.vectorized_strides[i]);
      }
    }
    ss << "}" << std::endl;
  }
  return ss;
}

// 输出 Queue 类型的内存信息
static std::stringstream &OutputQueueMemStr(std::stringstream &ss, const ge::AscTensorAttr &output_attr,
                                            const std::string &pos_str) {
  const auto &que = output_attr.que;
  ss << MemHardwareToStr(output_attr.mem.hardware) << "[";
  if (output_attr.mem.tensor_id != ge::kIdNone) {
    ss << "tensor_id=" << output_attr.mem.tensor_id << ", ";
  }
  ss << "que_id=" << que.id;
  if (output_attr.mem.reuse_id >= 0) {
    ss << ", reuse_id=" << output_attr.mem.reuse_id;
  }
  ss << ", depth=" << que.depth << ", pos=" << pos_str << "]";
  return ss;
}

// 输出 Buffer 类型的内存信息
static std::stringstream &OutputBufferMemStr(std::stringstream &ss, const ge::AscTensorAttr &output_attr,
                                             const std::string &pos_str) {
  ss << MemHardwareToStr(output_attr.mem.hardware) << "[";
  if (output_attr.mem.tensor_id != ge::kIdNone) {
    ss << "tensor_id=" << output_attr.mem.tensor_id << ", ";
  }
  ss << "buf_id=" << output_attr.buf.id;
  if (output_attr.mem.reuse_id >= 0) {
    ss << ", reuse_id=" << output_attr.mem.reuse_id;
  }
  ss << ", pos=" << pos_str << "]";
  return ss;
}

// 输出普通类型的内存信息
static std::stringstream &OutputNormalMemStr(std::stringstream &ss, const ge::AscTensorAttr &output_attr,
                                             const std::string &pos_str) {
  ss << MemHardwareToStr(output_attr.mem.hardware);
  if (output_attr.mem.tensor_id != ge::kIdNone) {
    ss << "[tensor_id=" << output_attr.mem.tensor_id << ", pos=" << pos_str << "]";
  } else {
    ss << "[pos=" << pos_str << "]";
  }
  return ss;
}

// 输出内存信息
static std::stringstream &OutputMemStr(std::stringstream &ss, const ge::AscTensorAttr &output_attr, bool verbose) {
  if (!verbose && (output_attr.mem.alloc_type != ge::AllocType::kAllocTypeQueue) &&
      (output_attr.mem.alloc_type != ge::AllocType::kAllocTypeBuffer)) {
    return ss;
  }

  ss << "        .mem = ";
  // 获取 position 字符串
  std::string pos_str;
  switch (output_attr.mem.position) {
    case ge::Position::kPositionVecIn:
      pos_str = "VECIN";
      break;
    case ge::Position::kPositionVecCalc:
      pos_str = "VECCALC";
      break;
    case ge::Position::kPositionVecOut:
      pos_str = "VECOUT";
      break;
    case ge::Position::kPositionGM:
      pos_str = "GM";
      break;
    default:
      pos_str = PositionToStr(output_attr.mem.position);
      break;
  }

  if (output_attr.mem.alloc_type == ge::AllocType::kAllocTypeQueue) {
    OutputQueueMemStr(ss, output_attr, pos_str);
  } else if (output_attr.mem.alloc_type == ge::AllocType::kAllocTypeBuffer) {
    OutputBufferMemStr(ss, output_attr, pos_str);
  } else {
    OutputNormalMemStr(ss, output_attr, pos_str);
  }
  ss << std::endl;
  return ss;
}

// 输出节点信息
static std::stringstream &NodeOutputStr(std::stringstream &ss, const ascir::Graph &graph, ge::AscNode &node,
                                        ge::AscTensorAttr &output_attr, size_t output_idx, bool verbose) {
  auto output_name = node.GetOpDesc()->GetOutputNameByIndex(output_idx);
  auto dtype = node.GetOpDesc()->GetOutputDesc(output_idx).GetDataType();

  ss << "    ." << output_name << ": " << DtypeToStr(dtype) << std::endl;

  OutputShapeStr(ss, graph, output_attr);
  OutputVectorizedStr(ss, graph, output_attr);
  OutputMemStr(ss, output_attr, verbose);

  return ss;
}

static void DumpGraphText(const Graph &graph, const string &suffix, const uint32_t graph_id, const bool verbose) {
  static std::stringstream prefix = GetDumpGraphPrefixAndCreateDir();
  std::stringstream ss;
  ss << prefix.str();
  auto dump_asc_graph = DebugStr(graph, verbose);
  ss << "ascgraph_" << std::setw(DUMP_ID_WIDTH) << std::setfill('0') << kDumpGraphIndex;
  ss << "_" << graph.GetName() << "_" << suffix << "_" << graph_id << ".txt";
  std::ofstream f_stream(ss.str());
  if (f_stream.is_open()) {
    f_stream << dump_asc_graph << std::endl;
    f_stream.close();
  }
}

void DumpComputeGraph(const ge::ComputeGraphPtr &compute_graph, const std::string &suffix, bool always_dump) {
  bool no_need_dump = IsNoNeedDump();
  if (!always_dump && (no_need_dump || (compute_graph == nullptr) || !CheckNeedDumpGraphBySuffix(suffix))) {
    return;
  }
  static std::stringstream prefix = GetDumpGraphPrefixAndCreateDir();
  std::stringstream gss;
  gss << compute_graph->GetName() << "_" << suffix;
  ge::GraphUtils::DumpGrphToOnnx(*compute_graph, prefix.str(), gss.str());
}

void DumpGraph(const ascir::Graph &graph, const std::string &suffix, const uint32_t graph_id, const bool verbose) {
  if (IsNoNeedDump()) {
    // 环境变量没开启时, 捕获图对象为异常退出时维测服务
    AscGraphDumperContext::GetThreadLocalCtx().AddWatchGraph(suffix, graph);
    return;
  }

  DumpGraphText(graph, suffix, graph_id, verbose);
  std::vector<ge::AscGraph> subgraphs;
  (void)graph.GetAllSubGraphs(subgraphs);
  for (auto &subgraph : subgraphs) {
    DumpGraphText(subgraph, "_Subgraph_", graph_id, verbose);
  }

  // dump onnx
  const auto compute_graph = ge::AscGraphUtils::GetComputeGraph(graph);
  DumpComputeGraph(compute_graph, suffix);
  ++kDumpGraphIndex;
}

void AlwaysDumpGraph(const Graph &graph, const string &suffix, const uint32_t graph_id, const bool verbose) {
  // dump txt
  DumpGraphText(graph, suffix, graph_id, verbose);
  std::vector<ge::AscGraph> subgraphs;
  (void)graph.GetAllSubGraphs(subgraphs);
  for (auto &subgraph : subgraphs) {
    DumpGraphText(subgraph, "_Subgraph_", graph_id, verbose);
  }

  // dump onnx
  const auto compute_graph = ge::AscGraphUtils::GetComputeGraph(graph);
  DumpComputeGraph(compute_graph, suffix, true);
  ++kDumpGraphIndex;
}

void DumpImplGraphs(const std::vector<ascir::Graph> &graphs, const std::string &suffix) {
  if (IsNoNeedDump()) {
    return;
  }
  for (size_t i = 0UL; i < graphs.size(); ++i) {
    DumpGraph(graphs[i], std::to_string(i) + "_" + suffix, i, true);
  }
}

void DumpPyCode(const ge::AscGraph &graph) {
  if (IsNoNeedDump()) {
    return;
  }
  static std::stringstream prefix = GetDumpGraphPrefixAndCreateDir();
  std::stringstream ss;
  ss << prefix.str();
  ss << graph.GetName() << "_" << std::to_string(kDumpGraphIndex) << "_graph_builder.py";
  ge::ascir::PythonCodeDumper dumper;
  (void)dumper.Dump(graph, ss.str());
}

std::string IdentifierToStr(ascir::Identifier id) {
  if (id == ge::kIdNone) {
    return "nil";
  } else {
    return std::to_string(id);
  }
}

std::string DebugStr(const ascir::Graph &graph, bool verbose) {
  std::stringstream ss;

  GraphNameStr(ss, graph);
  GraphSizeStr(ss, graph);
  GraphAxisStr(ss, graph);

  ss << "Nodes:" << std::endl;
  size_t idx = 0UL;
  for (auto node : graph.GetAllNodes()) {
    // Node name and exec_order
    ss << "  " << node->GetName() << ": " << node->GetType() << " (" << idx++ << ")" << std::endl;

    NodeAttrStr(ss, graph, node, verbose);

    // Node inputs
    auto input_names = CollectInputNames(graph, node);
    NodeInputStr(ss, input_names);

    // Node outputs
    for (size_t i = 0UL; i < node->outputs().size(); i++) {
      NodeOutputStr(ss, graph, *node, node->outputs[i].attr, i, verbose);
    }
  }

  return ss.str();
}

std::string DebugHintGraphStr(const ascir::HintGraph &graph) {
  return DebugStr(graph, false);
}

std::string DebugImplGraphStr(const ascir::ImplGraph &graph) {
  return DebugStr(graph, true);
}

void DumpScheduleResult(const ascir::FusedScheduledResult &fused_scheduled_result, const std::string &suffix,
                        uint32_t graph_id, bool verbose) {
  for (const auto &results : fused_scheduled_result.node_idx_to_scheduled_results) {
    for (const auto &result : results) {
      for (const auto &schedule_group : result.schedule_groups) {
        for (const auto &impl_graph : schedule_group.impl_graphs) {
          ascir::utils::DumpGraph(impl_graph, suffix, graph_id, verbose);
        }
      }
    }
  }
}

bool UseSmallTailConcatApi(const ge::AscNode &node, bool *output_need_align) {
  bool force_small_tail = false;
  (void)ge::AttrUtils::GetBool(node.GetOpDesc(), "_concat_small_tail", force_small_tail);
  GE_CHK_BOOL_RET_SPECIAL_STATUS(force_small_tail, true, "[%s] marked use small tail kernel", node.GetNamePtr());
  auto node_inputs = node.inputs;
  GE_CHK_BOOL_RET_SPECIAL_STATUS(node_inputs.Size() <= 1U,
                                 false, "input num = %u, do not use small tail concat api", node_inputs.Size());
  const auto dtype_size = GetSizeByDataType(node_inputs[0].attr.dtype);
  GE_WARN_ASSERT(dtype_size > 0); // 其实下一判断可以确保非0, 然而静态检查识别不了
  GE_CHK_BOOL_RET_SPECIAL_STATUS(dtype_size != sizeof(uint16_t) && dtype_size != sizeof(uint32_t), false,
                                 "[%s] only support dtype size = 2 or 4, but = %d", node.GetNamePtr(), dtype_size);
  const int32_t kAlignSize = 32 / dtype_size;
  std::vector<int64_t> src_col_sizes;
  int64_t dst_col_size = 0;
  size_t concat_dim;
  GE_CHK_BOOL_RET_SPECIAL_STATUS(!GetConcatDimAndColSizes(node, concat_dim, src_col_sizes, dst_col_size),
                                 false, "do not use small tail concat api");
  size_t aligned_cnt = 0;
  int64_t gcd = 16;
  for (const auto src_col_size : src_col_sizes) {
    aligned_cnt += (src_col_size % kAlignSize == 0) ? 1 : 0;
    gcd = ascgen_utils::Gcd(gcd, src_col_size);
  }
  // 全对齐, 使用全对齐的api性能更好
  GE_CHK_BOOL_RET_SPECIAL_STATUS(aligned_cnt == node_inputs.Size(), false,
                                 "[%s] inputs is all aligned", node.GetNamePtr());
  constexpr int64_t kSrcMaxSrcColSize = 64;
  constexpr uint32_t kMaxDstColSize = 96;  // 最大96K的tmp buffer, 只支持96
  for (size_t i = 0U; i < src_col_sizes.size(); ++i) {
    GE_CHK_BOOL_RET_SPECIAL_STATUS((gcd == 0) || (src_col_sizes[i] / gcd > kSrcMaxSrcColSize), false,
                                   "[%s] input[%zu] col_size = %ld, gcd = %ld, not a small dim",
                                   node.GetNamePtr(), i, src_col_sizes[i], gcd);
  }
  GE_CHK_BOOL_RET_SPECIAL_STATUS((gcd == 0) || (dst_col_size / gcd) > kMaxDstColSize, false,
                                 "[%s] output col_size = %ld, gcd = %ld, not a small dim", node.GetNamePtr(),
                                 dst_col_size, gcd);
  if (!IsStoreWithoutStride(node)) {
    bool concat_last_dim = concat_dim == (node_inputs[0].attr.repeats.size() - 1UL);
    GE_CHK_BOOL_RET_SPECIAL_STATUS((!concat_last_dim) && (dst_col_size % kAlignSize != 0),
                                   false,
                                   "[%s] Concat on non tail dim, output is not aligned and need store with strides",
                                   node.GetNamePtr());
    if (output_need_align != nullptr) {
      *output_need_align = (dst_col_size % kAlignSize != 0);
    }
  }
  GELOGI("[%s] will use small tail kernel", node.GetNamePtr());
  return true;
}

bool IsConcatAllInputsAligned(const ge::AscNode &node) {
  constexpr int32_t kAlignment = 32;
  auto node_inputs = node.inputs;
  auto concat_dim = std::numeric_limits<size_t>::max();
  GE_WARN_ASSERT(GetConcatDim(node, concat_dim));
  GE_WARN_ASSERT(concat_dim < node_inputs[0].attr.repeats.size());

  const auto dtype_size = GetSizeByDataType(node_inputs[0].attr.dtype);
  GE_WARN_ASSERT(dtype_size > 0);
  for (size_t i = 0UL; i < node_inputs.Size(); ++i) {
    const auto &input_repeats = node_inputs[i].attr.repeats;
    ge::Expression size = ge::Symbol(dtype_size);
    for (size_t j = concat_dim; j < input_repeats.size(); ++j) {
      size = size * input_repeats[j];
    }
    if (ge::SymbolicUtils::StaticCheckEq(ge::sym::Mod(size, ge::Symbol(kAlignment)), ge::ops::Zero) !=
        ge::TriBool::kTrue) {
      GELOGI("input[%zu] size = %s, is not aligned", i, ge::SymbolicUtils::ToString(size).c_str());
      return false;
    }
  }
  GELOGI("[%s] All inputs is aligned", node.GetNamePtr());
  return true;
}
}  // namespace ascir::utils

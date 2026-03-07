/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/model_builder.h"
#include <securectype.h>
#include <iostream>
#include <set>
#include <unordered_map>
#include "mmpa/mmpa_api.h"
#include "common/dump/dump_manager.h"
#include "graph/build/stream/dynamic_stream_allocator.h"
#include "graph/build/stream_graph_optimizer.h"
#include "common/omg_util/omg_util.h"
#include "common/compile_profiling/ge_trace_wrapper.h"
#include "graph/ge_context.h"
#include "graph/optimize/params.h"
#include "graph/unfold/graph_unfolder.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "api/gelib/gelib.h"
#include "framework/memory/memory_assigner.h"
#include "framework/omg/version.h"
#include "framework/common/types.h"
#include "graph/passes/memory_conflict/set_input_output_offset_pass.h"
#include "graph/build/memory/block_mem_assigner.h"
#include "common/helper/model_parser_base.h"
#include "framework/common/helper/model_helper.h"
#include "common/proto_util/proto_util.h"
#include "common/checker.h"
#include "exec_runtime/execution_runtime_utils.h"
#include "graph/utils/op_type_utils.h"
#include "common/math/math_util.h"
#include "graph/passes/pass_manager.h"
#include "base/err_msg.h"
#include "ge/ge_api_types.h"

namespace {
const uint32_t kWeightsStartOffset = 512;
const int32_t kWrongIndex = -2;
const int32_t kInvalidIndexNum = -1;
constexpr size_t kAlignBytes = 32U;

const std::set<std::string> adjust_layer_type_ = {ge::CONVOLUTION};
constexpr const ge::char_t *kVectorCore = "VectorCore";
constexpr const ge::char_t *kCoreType = "ge.engineType";
constexpr const ge::char_t *kEnableL1Fusion = "ge.l1Fusion";
constexpr const ge::char_t *kAttrEntrySymbolOfElf = "_kernelname";

bool IsGeLocalOp(const ge::ConstOpDescPtr &op_desc) {
  auto type = op_desc->GetType();
  if ((type == ge::CONSTANTOP) || (type == ge::CONSTANT)) {
    // const op just has one output
    ge::GeTensorDesc output_desc = op_desc->GetOutputDesc(0);
    return !(output_desc.GetDataType() == ge::DT_STRING);
  }
  const std::set<std::string> ge_local_set = {ge::STREAMMERGE, ge::MEMCPYASYNC, ge::STREAMACTIVE,  ge::STREAMSWITCH,
                                              ge::VARIABLE,    ge::NOOP,        ge::CONSTANT,      ge::ENTER,
                                              ge::REFENTER,    ge::LOOPCOND,    ge::NEXTITERATION, ge::FILECONSTANT,
                                              ge::EXIT,        ge::REFEXIT,     ge::MERGE,         ge::MEMCPYADDRASYNC,
                                              ge::REFNEXTITERATION};
  return (ge_local_set.find(type) != ge_local_set.end());
}

ge::Status SaveSoftSyncOpWeightByDependNames(const ge::NodePtr &node, const std::vector<std::string> &depend_names) {
  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  for (const auto &depend_name : depend_names) {
    const int32_t input_idx = op_desc->GetInputIndexByName(depend_name);
    if (input_idx == kInvalidIndexNum) {
      GELOGW("Can not find soft sync op[%s]'s input of name: %s.", op_desc->GetName().c_str(), depend_name.c_str());
      continue;
    }
    const auto in_data_anchor = node->GetInDataAnchor(input_idx);
    GE_CHECK_NOTNULL(in_data_anchor);
    const auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_data_anchor);
    auto in_data_node = peer_out_data_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(in_data_node);
    if (in_data_node->GetType() == ge::DATA) {
      const auto in_desc = in_data_node->GetOpDesc();
      GE_CHECK_NOTNULL(in_desc);
      if (!in_desc->HasAttr(ge::ATTR_NAME_PARENT_NODE_INDEX)) {
        GELOGW("Soft sync op[%s]'s input of name: %s is not const.", op_desc->GetName().c_str(), depend_name.c_str());
        continue;
      }
    }
    GE_ASSERT_SUCCESS(ge::NodeUtils::GetInNodeCrossPartionedCallNode(node, in_data_anchor->GetIdx(), in_data_node));
    GE_CHECK_NOTNULL(in_data_node);
    if (ge::kConstOpTypes.count(in_data_node->GetType()) == 0U) {
      GELOGW("Soft sync op[%s]'s input of name: %s is not const.", op_desc->GetName().c_str(), depend_name.c_str());
      continue;
    }
    const auto const_desc = in_data_node->GetOpDesc();
    GE_CHECK_NOTNULL(const_desc);
    ge::ConstGeTensorPtr weight = nullptr;
    GE_ASSERT_TRUE(ge::AttrUtils::GetTensor(const_desc, ge::ATTR_NAME_WEIGHTS, weight));
    GE_CHECK_NOTNULL(weight);
    auto input_desc = op_desc->MutableInputDesc(depend_name);
    GE_CHECK_NOTNULL(input_desc);
    GE_ASSERT_TRUE(ge::AttrUtils::SetTensor(input_desc, ge::ATTR_NAME_VALUE, weight));
    GELOGD("Save weight to soft sync op[%s]'s input of name: %s.", op_desc->GetName().c_str(), depend_name.c_str());
  }
  return ge::SUCCESS;
}
}  // namespace

namespace ge {
ModelBuilder::ModelBuilder(uint64_t session_id, ComputeGraphPtr compute_graph,
                           const Graph2SubGraphInfoList &subgraphs,
                           const std::map<std::string, int32_t> &stream_max_parallel_num,
                           bool hcom_parallel, int32_t mode)
    : session_id_(session_id),
      weight_offset_(kWeightsStartOffset),
      compute_graph_(std::move(compute_graph)),
      subgraphs_(subgraphs),
      stream_allocator_(compute_graph_, subgraphs_),
      stream_num_(0),
      notify_num_(0),
      event_num_(0),
      label_num_(0),
      stream_max_parallel_num_(stream_max_parallel_num),
      hcom_parallel_(hcom_parallel),
      build_mode_(mode),
      max_mem_offset_(0),
      host_max_mem_offset_(kMemoryHostFeatureMapLogicBase),
      host_svm_max_mem_offset_(kMemoryHostSVMFeatureMapLogicBase),
      p2p_mem_offset_(0),
      zero_copy_mem_size_(0),
      platform_type_(0),
      is_loop_graph_(false),
      is_l1_fusion_enable_(false),
      has_assigned_var_mem(false) {}

ModelBuilder::~ModelBuilder() {}

Status ModelBuilder::CalcOutputSize(const ge::NodePtr &n) const {
  GE_CHECK_NOTNULL(n);
  auto node_op_desc = n->GetOpDesc();
  GE_CHECK_NOTNULL(node_op_desc);
  uint32_t index = 0;
  for (const auto &output_desc_ptr : node_op_desc->GetAllOutputsDescPtr()) {
    GeTensorDesc &desc_temp = *output_desc_ptr;

    uint32_t dim_num = static_cast<uint32_t>(desc_temp.GetShape().GetDimNum());
    GE_IF_BOOL_EXEC(dim_num > DIM_DEFAULT_SIZE, TensorUtils::SetRealDimCnt(desc_temp, dim_num));
    // calculate tensor size
    int64_t size_temp = 0;
    graphStatus graph_status = TensorUtils::GetTensorMemorySizeInBytes(desc_temp, size_temp);
    if (graph_status != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Get tensor size in bytes failed for op:%s(%s) index:%u",
                        node_op_desc->GetName().c_str(), node_op_desc->GetType().c_str(), index);
      GELOGE(graph_status, "[Get][TensorMemorySize] In Bytes failed for op:%s(%s) index:%u",
             node_op_desc->GetName().c_str(), node_op_desc->GetType().c_str(), index);
      return FAILED;
    }
    TensorUtils::SetSize(desc_temp, size_temp);
    GELOGD("Update output desc, dim_size: %u, mem_size: %ld, format: %s, type: %s, node name:%s", dim_num, size_temp,
           TypeUtils::FormatToSerialString(desc_temp.GetFormat()).c_str(),
           TypeUtils::DataTypeToSerialString(desc_temp.GetDataType()).c_str(), node_op_desc->GetName().c_str());
    index++;
  }

  return SUCCESS;
}

bool ModelBuilder::SetInputConst(const OpDescPtr &op_desc, const NodePtr &src_node, size_t index,
                                 std::vector<bool> &is_input_const) const {
  GELOGI("SetIsInputConst const: %s, source node: %s", op_desc->GetName().c_str(), src_node->GetName().c_str());
  for (size_t i = is_input_const.size(); i <= index; ++i) {
    is_input_const.push_back(false);
  }
  is_input_const[index] = true;

  std::vector<GeTensorPtr> weights = OpDescUtils::MutableWeights(src_node);
  if (weights.empty()) {
    GELOGW("SetInputIsConst weights is empty, node: %s", src_node->GetName().c_str());
    return false;
  }
  GeTensorPtr weight = weights[0];
  GE_IF_BOOL_EXEC(weight == nullptr, return true);
  GeTensorDesc &tensor_desc = weight->MutableTensorDesc();
  int64_t data_offset = 0;
  if (TensorUtils::GetDataOffset(tensor_desc, data_offset) != GRAPH_SUCCESS) {
    GELOGW("Get Offset from weight failed");
    return false;
  }
  auto input_tensor = op_desc->MutableInputDesc(static_cast<uint32_t>(index));
  if (input_tensor == nullptr) {
    GELOGW("Get input_tensor failed");
    return false;
  }
  TensorUtils::SetDataOffset(*input_tensor, data_offset);
  return true;
}

void ModelBuilder::SetInputIsConst(const ge::NodePtr &n) const {
  auto node_op_desc = n->GetOpDesc();
  GE_CHECK_NOTNULL_JUST_RETURN(node_op_desc);

  // must set all true input_const to false
  std::vector<bool> is_input_const(n->GetAllInDataAnchorsSize(), false);

  std::string const_type;
  auto in_data_anchors = n->GetAllInDataAnchors();
  for (size_t index = 0; index < in_data_anchors.size(); index++) {
    auto in_data_anchor = in_data_anchors.at(index);
    const auto &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
    const auto &src_node = peer_out_anchor->GetOwnerNode();
    if ((!NodeUtils::GetConstOpType(src_node, const_type)) ||
        gert::GraphUnfolder::IsDataNotNeedRefConst(src_node)) {
      continue;
    }

    if (const_type == CONSTANT) {
      if (!SetInputConst(node_op_desc, src_node, index, is_input_const)) {
        return;
      }
    } else {
      if ((index < is_input_const.size()) && is_input_const[index]) {
        is_input_const[index] = false;
      }
    }
  }

  GELOGD("Update opdesc:%s InputConst:%s", node_op_desc->GetName().c_str(),
         ToString(is_input_const).c_str());
  node_op_desc->SetIsInputConst(is_input_const);
}

void ModelBuilder::ReuseWeightMem(const size_t output_size, GeTensorPtr &weight,
                                  bool &find_same_const, size_t &current_mem_offset) {
  if (ExecutionRuntimeUtils::IsHeterogeneous()) {
    // helper is not supported due to some bug
    weight_offset_need_feeded_.insert(current_mem_offset);
    return;
  }
  const auto it = reuse_weight_map_.find(output_size);
  if (it == reuse_weight_map_.end()) {
    GELOGD("can not find same size %zu", output_size);
    std::vector<std::pair<void *, size_t>> tmp_weight_info;
    tmp_weight_info.emplace_back(std::make_pair(static_cast<void *>(weight->MutableData().data()),
        current_mem_offset));
    reuse_weight_map_.insert({output_size, tmp_weight_info});
    weight_offset_need_feeded_.insert(current_mem_offset);
  } else {
    auto &weights_info = it->second;
    for (auto &weight_info : weights_info) {
      if (memcmp(reinterpret_cast<void *>(weight->MutableData().data()),
          reinterpret_cast<void *>(weight_info.first), output_size) == 0) {
        current_mem_offset = weight_info.second;
        find_same_const = true;
        break;
      }
    }
    if (!find_same_const) {
      GELOGD("Can not find same const value, size is %zu", output_size);
      weights_info.emplace_back(std::make_pair(static_cast<void *>(weight->MutableData().data()),
          current_mem_offset));
      weight_offset_need_feeded_.insert(current_mem_offset);
    }
  }
}

Status ModelBuilder::AdjustConstWeightSize(const ge::NodePtr &node, size_t &mem_offset) {
  GE_CHECK_NOTNULL(node);
  if (node->GetType() == CONSTANT) {
    std::vector<GeTensorPtr> weights = OpDescUtils::MutableWeights(node);
    if (weights.empty()) {
      REPORT_INNER_ERR_MSG("E19999", "Check weights size of node %s(%s) is empty",
                         node->GetName().c_str(), node->GetType().c_str());
      GELOGE(FAILED, "[Check][Param] weights size of node %s is empty", node->GetName().c_str());
      return FAILED;
    }
    GeTensorPtr weight = weights[0];
    if (weight == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Check weight of node %s(%s) is nullptr",
                         node->GetName().c_str(), node->GetType().c_str());
      GELOGE(FAILED, "[Check][Param] weights[0] is null, node:%s.", node->GetName().c_str());
      return FAILED;
    }
    GeTensorDesc &tensor_desc = weight->MutableTensorDesc();
    size_t output_size = weight->GetData().size();
    size_t current_mem_offset = mem_offset;
    bool find_same_const = false;
    ReuseWeightMem(output_size, weight, find_same_const, current_mem_offset);
    TensorUtils::SetDataOffset(tensor_desc, current_mem_offset);
    GELOGD("Node: %s, weight size: %zu, current_mem_offset: %zu",
        node->GetName().c_str(), output_size, current_mem_offset);
    if (!find_same_const) {
      mem_offset += output_size;
    }
  }
  return SUCCESS;
}

Status ModelBuilder::SetInputOutputDesc() {
  Status ret;
  for (const ge::NodePtr &n : compute_graph_->GetNodes(compute_graph_->GetGraphUnknownFlag())) {
    auto node_op_desc = n->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, continue);
    const auto& type = node_op_desc->GetType();
    if (!is_loop_graph_ && (type == LOOPCOND)) {
      is_loop_graph_ = true;
    }
    // if user set input node format ND, the expected node for data and netoutput format is ND in
    // final graph.
    bool set_nd = (compute_graph_->GetParentGraph() == nullptr) &&
        (GetLocalOmgContext().format == domi::DOMI_TENSOR_ND) &&
        (!node_op_desc->HasAttr("_is_single_op")) &&
        (OpTypeUtils::IsDataNode(type) || (type == NETOUTPUT));
    if (set_nd) {
      auto inputDescsPtr = node_op_desc->GetAllInputsDescPtr();
      auto outputDescsPtr = node_op_desc->GetAllOutputsDescPtr();
      ge::Format format = ge::FORMAT_ND;
      for (auto &inputDescPtr : inputDescsPtr) {
        GE_CHECK_NOTNULL(inputDescPtr);
        if (AttrUtils::HasAttr(*inputDescPtr, ATTR_NAME_ORIGIN_FORMAT_IS_SET)) {
          continue;
        }

        inputDescPtr->SetFormat(format);
        inputDescPtr->SetOriginFormat(format);
      }
      for (auto &outputDescPtr : outputDescsPtr) {
        GE_CHECK_NOTNULL(outputDescPtr);
        if (AttrUtils::HasAttr(*outputDescPtr, ATTR_NAME_ORIGIN_FORMAT_IS_SET)) {
          continue;
        }

        outputDescPtr->SetFormat(format);
        outputDescPtr->SetOriginFormat(format);
      }
    }
    if (OpTypeUtils::IsDataNode(type)) {
      GELOGD("Data node: %s.", n->GetName().c_str());
      continue;
    }

    GE_IF_BOOL_EXEC((n->GetInNodesSize() == 0U) && (n->GetOutNodesSize() == 0U), continue;);
    SetInputIsConst(n);
    bool is_unknow = false;
    (void)NodeUtils::GetNodeUnknownShapeStatus(*n, is_unknow);
    if ((IsGeLocalOp(n->GetOpDesc())) && (!is_unknow)) {
      GE_CHK_STATUS_RET(CalcOutputSize(n), "[Calc][OutputSize] failed, node:%s", n->GetName().c_str());
    }
    ret = AdjustConstWeightSize(n, weight_offset_);
    GE_CHK_STATUS_RET(ret, "[Adjust][ConstWeightSize] failed, node:%s", n->GetName().c_str());

    GE_IF_BOOL_EXEC(((weight_offset_ > 0) && (weight_offset_ % MEM_ALIGN_SIZE != 0)),
                    weight_offset_ = (weight_offset_ + MEM_ALIGN_SIZE - 1) / MEM_ALIGN_SIZE * MEM_ALIGN_SIZE);
  }
  // 图编译后期，在 CalcOutputSize里面对opdesc上面的输出size进行了padding 32操作，但是实际weight申请内存时并没有做padding32操作
  // 导致在加载期，会存在拷贝越界情况。实际修改我们padding 32后做了512对齐，目的是确保下一个子图起始地址是512对齐的，不然copy性能会变差
  GELOGD("Before alignment processing, weight_offset_ is %zu", weight_offset_);
  if (weight_offset_ > 0U) {
    GE_CHK_STATUS_RET(CheckSizeTAddOverflow(weight_offset_, (kAlignBytes + MEM_ALIGN_SIZE - 1)),
                      "32-aligned weights overflow, weight_offset_ is %zu", weight_offset_);
    weight_offset_ = (weight_offset_ + kAlignBytes + MEM_ALIGN_SIZE - 1) / MEM_ALIGN_SIZE * MEM_ALIGN_SIZE;
  }
  GELOGD("After add 32 and then do 512 alignment, weight_offset_ is %zu", weight_offset_);
  GE_CHK_STATUS_RET(compute_graph_->TopologicalSorting(),
                    "[Call][TopologicalSorting] failed, graph:%s",
                    compute_graph_->GetName().c_str());
  return SUCCESS;
}

void ModelBuilder::AddNodeInputProperty() const {
  for (const ge::NodePtr &node : compute_graph_->GetNodes(compute_graph_->GetGraphUnknownFlag())) {
    auto node_op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, GELOGW("node_op_desc is nullptr!"); return);
    std::vector<std::string> src_name_list;
    src_name_list.reserve(node->GetInNodesSize());
    std::vector<int64_t> src_index_list;
    src_index_list.reserve(node->GetInNodesSize());

    for (const auto in_data_anchor : node->GetAllInDataAnchorsPtr()) {
      const auto& peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
      GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
      GE_IF_BOOL_EXEC(node_op_desc->HasAttr(MERGE_PRENODE_FLAG), continue);

      const auto src_node = peer_out_anchor->GetOwnerNodeBarePtr();
      src_name_list.emplace_back(src_node->GetName());
      src_index_list.emplace_back(peer_out_anchor->GetIdx());
    }
    auto in_control_anchor = node->GetInControlAnchor();
    if (in_control_anchor != nullptr) {
      std::string src_name_temp;
      for (const auto out_control_anchor : in_control_anchor->GetPeerOutControlAnchorsPtr()) {
        const auto src_node = out_control_anchor->GetOwnerNodeBarePtr();
        src_name_temp += src_name_temp.empty() ? src_node->GetName() : ":" + src_node->GetName();
      }
      GE_IF_BOOL_EXEC(!src_name_temp.empty(), src_name_list.emplace_back(src_name_temp);)
    }
    node_op_desc->SetSrcName(src_name_list);
    node_op_desc->SetSrcIndex(src_index_list);
  }

  for (const ge::NodePtr &node : compute_graph_->GetNodes(compute_graph_->GetGraphUnknownFlag())) {
    const auto& node_op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, GELOGW("node_op_desc is nullptr!"); return);
    GE_IF_BOOL_EXEC(node_op_desc->GetType() == NETOUTPUT, continue);
    auto out_control_anchor = node->GetOutControlAnchor();
    GE_IF_BOOL_EXEC(out_control_anchor == nullptr, GELOGW("out_control_anchor is nullptr!"); return);
    std::vector<std::string> dst_name_list;
    dst_name_list.reserve(node->GetOutNodesSize());
    std::vector<int64_t> dst_index_list;
    dst_index_list.reserve(node->GetOutNodesSize());
    std::string dst_name_temp;
    for (const auto in_control_anchor : out_control_anchor->GetPeerInControlAnchorsPtr()) {
      const auto dst_node = in_control_anchor->GetOwnerNodeBarePtr(); // dst_node must not be null
      dst_name_temp += dst_name_temp.empty() ? dst_node->GetName() : ":" + dst_node->GetName();
    }
    GE_IF_BOOL_EXEC(!dst_name_temp.empty(), dst_name_list.emplace_back(dst_name_temp));

    GE_IF_BOOL_EXEC(!out_control_anchor->GetPeerInControlAnchorsPtr().empty(),
                    dst_index_list.emplace_back(kInvalidIndexNum));

    for (const auto out_data_anchor : node->GetAllOutDataAnchorsPtr()) {
      GE_IF_BOOL_EXEC(node_op_desc->HasAttr(MERGE_PRENODE_FLAG), break);
      dst_name_temp = "";
      int64_t dst_index = kWrongIndex;  // assign an impossible value to dst_index.
      for (const auto in_data_anchor : out_data_anchor->GetPeerInDataAnchorsPtr()) {
        GE_IF_BOOL_EXEC(in_data_anchor == nullptr, GELOGW("in_data_anchor is nullptr!"); return);
        const auto dst_node = in_data_anchor->GetOwnerNodeBarePtr(); // dst_node must not be null
        dst_name_temp += dst_name_temp.empty() ? dst_node->GetName() : ":" + dst_node->GetName();
        dst_index = in_data_anchor->GetIdx();
      }
      GE_IF_BOOL_EXEC(dst_index != kWrongIndex, dst_index_list.emplace_back(dst_index));  // not found
      GE_IF_BOOL_EXEC(!dst_name_temp.empty(), dst_name_list.emplace_back(dst_name_temp));
    }
    node_op_desc->SetDstName(dst_name_list);
    node_op_desc->SetDstIndex(dst_index_list);
  }
}

Status ModelBuilder::AdjustInputTensorFlag() const {
  for (const ge::NodePtr &n : compute_graph_->GetNodes(compute_graph_->GetGraphUnknownFlag())) {
    if (OpTypeUtils::IsDataNode(n->GetType())) {
      GELOGD("Data node: %s.", n->GetName().c_str());
      for (const auto &anchor : n->GetAllOutDataAnchors()) {
        for (const auto &in_anchors : anchor->GetPeerInDataAnchors()) {
          GE_IF_BOOL_EXEC(in_anchors == nullptr, continue);
          auto owner_node_op_desc = in_anchors->GetOwnerNodeBarePtr()->GetOpDesc();
          GE_IF_BOOL_EXEC(owner_node_op_desc == nullptr, continue);
          const auto& input_desc = owner_node_op_desc->MutableInputDesc(in_anchors->GetIdx());
          if (input_desc == nullptr) {
            continue;
          }
          ge::TensorUtils::SetInputTensor(*input_desc, true);
        }
      }
    }
  }
  return SUCCESS;
}
Status ModelBuilder::InitL1FusionOption() {
  std::string buffer_optimize = "off_optimize";
  graphStatus ret = ge::GetContext().GetOption(BUFFER_OPTIMIZE, buffer_optimize);
  if (ret == GRAPH_SUCCESS) {
    bool off_superkernel = true; // 默认关闭
    (void)AttrUtils::GetBool(compute_graph_, ATTR_NAME_OFF_SUPERKERNEL_ATTR, off_superkernel);
    // l1fusion只有小海思才会使能，l1 fusion依赖superkernel使能进行绑核
    // 如果l1fusion的代码要使能，sgat组件会同步设置ATTR_NAME_OFF_SUPERKERNEL_ATTR为false来使能superkernel
    is_l1_fusion_enable_ = ((buffer_optimize == "l1_optimize") && (!off_superkernel));
    GELOGI("Compute graph %s the value of %s is %s, superkernel flag %d.", compute_graph_->GetName().c_str(),
           BUFFER_OPTIMIZE.c_str(), buffer_optimize.c_str(), is_l1_fusion_enable_);
  } else {
    GELOGW("The value of %s is empty.", kEnableL1Fusion);
    return SUCCESS;
  }

  if (is_l1_fusion_enable_) {
    std::string virtual_type = "0";
    ret = ge::GetContext().GetOption(VIRTUAL_TYPE, virtual_type);
    if ((ret == GRAPH_SUCCESS) && (virtual_type == "1")) {
        std::string situation = "L1_fusion is not supported in the virtual instance scenario";
        REPORT_PREDEFINED_ERR_MSG("E13024", std::vector<const char_t *>({"value", "env", "situation"}),
                                   std::vector<const char_t *>({virtual_type.c_str(), "VIRTUAL_TYPE", situation.c_str()}));
        GELOGE(FAILED, "BuildModelDef fail because l1fusion enable and virtual type is %s.", virtual_type.c_str());
        return FAILED;
    }
    GELOGW("Get virtual type ret %d , the type is %s.", ret, virtual_type.c_str());
  }
  GELOGI("Compute graph %s, l1fusion is %d.", compute_graph_->GetName().c_str(), is_l1_fusion_enable_);
  return SUCCESS;
}

Status ModelBuilder::BuildModelDef(ge::Model &model) {
  GE_ASSERT_SUCCESS(BuildModelDefForMem(model), "[Build][ModelDef] Part one failed!");
  GE_ASSERT_SUCCESS(BuildModelDefForStream(model), "[Build][ModelDef] Part two failed!");
  return SUCCESS;
}

Status ModelBuilder::BuildModelDefForMem(ge::Model &model) {
  ClearOriginalFormat();

  max_mem_offset_ = mem_type_to_mem_offset_[RT_MEMORY_HBM];
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, ATTR_MODEL_MEMORY_SIZE, max_mem_offset_),
                   REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s in model failed", ATTR_MODEL_MEMORY_SIZE.c_str());
                   GELOGE(FAILED, "[Set][Attr] %s in model failed", ATTR_MODEL_MEMORY_SIZE.c_str());
                   return FAILED);
  auto mem_type_session_scope = (kSessionScopeMemory | RT_MEMORY_HBM);
  size_t session_scope_mem_offset = 0;
  std::map<uint64_t, size_t>::const_iterator it = mem_type_to_mem_offset_.find(mem_type_session_scope);
  if (it != mem_type_to_mem_offset_.cend()) {
    session_scope_mem_offset = it->second;
  }
  if (mem_type_to_mem_offset_.find(RT_MEMORY_P2P_DDR) != mem_type_to_mem_offset_.cend()) {
    p2p_mem_offset_ = mem_type_to_mem_offset_[RT_MEMORY_P2P_DDR];
  }
  if (mem_type_to_mem_offset_.find(RT_MEMORY_HOST) != mem_type_to_mem_offset_.cend()) {
    host_max_mem_offset_ = mem_type_to_mem_offset_[RT_MEMORY_HOST];
  }
  if (mem_type_to_mem_offset_.find(RT_MEMORY_HOST_SVM) != mem_type_to_mem_offset_.cend()) {
    host_svm_max_mem_offset_ = mem_type_to_mem_offset_[RT_MEMORY_HOST_SVM];
  }

  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, ATTR_MODEL_SESSION_SCOPE_MEMORY_SIZE, session_scope_mem_offset),
                   REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s in model failed",
                                      ATTR_MODEL_SESSION_SCOPE_MEMORY_SIZE.c_str());
  GELOGE(FAILED, "SetInt of ATTR_NAME_SESSION_SCOPE_MEMORY_SIZE failed.");
  return FAILED);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, MODEL_ATTR_TASK_GEN_HOST_BASE_ADDR, kMemoryHostFeatureMapLogicBase),
     REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s in model failed", MODEL_ATTR_TASK_GEN_HOST_BASE_ADDR.c_str());
     GELOGE(FAILED, "[Set][Attr] %s in model failed", MODEL_ATTR_TASK_GEN_HOST_BASE_ADDR.c_str());
     return FAILED);
  GE_CHECK_GE(host_max_mem_offset_, kMemoryHostFeatureMapLogicBase);
  const auto host_memory_size = host_max_mem_offset_ - kMemoryHostFeatureMapLogicBase;
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, MODEL_ATTR_HOST_MEMORY_SIZE, host_memory_size),
     REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s in model failed", MODEL_ATTR_HOST_MEMORY_SIZE.c_str());
     GELOGE(FAILED, "[Set][Attr] %s in model failed", MODEL_ATTR_HOST_MEMORY_SIZE.c_str());
     return FAILED);
  GE_CHK_BOOL_EXEC(
     ge::AttrUtils::SetInt(&model, MODEL_ATTR_TASK_GEN_HOST_SVM_BASE_ADDR, kMemoryHostSVMFeatureMapLogicBase),
     REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s in model failed", MODEL_ATTR_TASK_GEN_HOST_SVM_BASE_ADDR.c_str());
     GELOGE(FAILED, "[Set][Attr] %s in model failed", MODEL_ATTR_TASK_GEN_HOST_SVM_BASE_ADDR.c_str());
     return FAILED);
  GE_CHECK_GE(host_svm_max_mem_offset_, kMemoryHostSVMFeatureMapLogicBase);
  const auto host_svm_memory_size = host_svm_max_mem_offset_ - kMemoryHostSVMFeatureMapLogicBase;
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, MODEL_ATTR_HOST_SVM_SIZE, host_svm_memory_size),
                   REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s in model failed", MODEL_ATTR_HOST_SVM_SIZE.c_str());
                   GELOGE(FAILED, "[Set][Attr] %s in model failed", MODEL_ATTR_HOST_SVM_SIZE.c_str());
                   return FAILED);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, ATTR_MODEL_P2P_MEMORY_SIZE, p2p_mem_offset_),
                   REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s in model failed", ATTR_MODEL_P2P_MEMORY_SIZE.c_str());
                   GELOGE(FAILED, "[Set][Attr] %s in model failed", ATTR_MODEL_P2P_MEMORY_SIZE.c_str());
                   return FAILED);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, ATTR_MODEL_WEIGHT_SIZE, weight_offset_),
                   REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s in model failed", ATTR_MODEL_WEIGHT_SIZE.c_str());
                   GELOGE(FAILED, "[Set][Attr] %s in model failed", ATTR_MODEL_WEIGHT_SIZE.c_str());
                   return FAILED);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, ATTR_MODEL_NOTIFY_NUM, notify_num_),
                   REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s in model failed", ATTR_MODEL_NOTIFY_NUM.c_str());
                       GELOGE(FAILED, "[Set][Attr] %s in model failed", ATTR_MODEL_NOTIFY_NUM.c_str());
                       return FAILED);
  GE_ASSERT_TRUE(ge::AttrUtils::SetListInt(&model, ATTR_MODEL_NOTIFY_TYPES, notify_types_));
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, ATTR_MODEL_EVENT_NUM, event_num_),
                   REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s in model failed", ATTR_MODEL_EVENT_NUM.c_str());
                       GELOGE(FAILED, "[Set][Attr] %s in model failed", ATTR_MODEL_EVENT_NUM.c_str());
                       return FAILED);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, ATTR_MODEL_LABEL_NUM, label_num_),
                   REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s in model failed", ATTR_MODEL_LABEL_NUM.c_str());
                   GELOGE(FAILED, "[Set][Attr] %s in model failed", ATTR_MODEL_LABEL_NUM.c_str());
                   return FAILED);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, zero_copy_mem_size_),
                   REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s in model failed",
                                      ATTR_MODEL_ZERO_COPY_MEMORY_SIZE.c_str());
                   GELOGE(FAILED, "[Set][Attr] %s in model failed.", ATTR_MODEL_ZERO_COPY_MEMORY_SIZE.c_str());
                   return FAILED);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListStr(&model, ATTR_MODEL_OUT_NODES_NAME, GetLocalOmgContext().net_out_nodes),
                   REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s in model failed", ATTR_MODEL_OUT_NODES_NAME.c_str());
                   GELOGE(FAILED, "[Set][Str] %s in model failed.", ATTR_MODEL_OUT_NODES_NAME.c_str());
                   return FAILED);
  (void) ge::AttrUtils::SetListListInt(&model, ATTR_MODEL_SUB_MEMORY_INFO, sub_mem_offsets_);

  // Set output reuse input memory indexes from option
  std::string output_reuse_input_mem_indexes;
  if (ge::GetContext().GetOption(OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES, output_reuse_input_mem_indexes) == SUCCESS) {
    if (!output_reuse_input_mem_indexes.empty()) {
      if (!ge::AttrUtils::SetStr(&model, ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, output_reuse_input_mem_indexes)) {
        REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s in model failed", ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES.c_str());
        GELOGE(FAILED, "[Set][Str] %s in model failed", ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES.c_str());
        return FAILED;
      }
      GELOGI("Set attr output_reuse_input_mem_indexes to model, value is %s.", output_reuse_input_mem_indexes.c_str());
    }
  }

  GELOGI(
      "For model, max_mem_offset: %zu, host_max_mem_offset: %zu, host_svm_max_mem_offset: %zu, p2p_mem_size: %zu, "
      "zero_copy_mem_size: %zu, "
      "session_scope_mem_size: %zu",
      max_mem_offset_, host_max_mem_offset_, host_svm_max_mem_offset_, p2p_mem_offset_, zero_copy_mem_size_,
      session_scope_mem_offset);
  std::string fp_ceiling_mode;
  if (ge::GetContext().GetOption("ge.fpCeilingMode", fp_ceiling_mode) == SUCCESS) {
    if (!ge::AttrUtils::SetStr(&model, ATTR_FP_CEILING_MODE, fp_ceiling_mode)) {
      REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s in model failed", ATTR_FP_CEILING_MODE.c_str());
      GELOGE(FAILED, "[Set][Str] %s in model failed", ATTR_FP_CEILING_MODE.c_str());
      return FAILED;
    }
    GELOGI("Set attr ATTR_FP_CEILING_MODE to model, value is %s.", fp_ceiling_mode.c_str());
  }

  std::string ge_core_type;
  Status ret = ge::GetContext().GetOption(kCoreType, ge_core_type);
  if (ret != SUCCESS) {
    GELOGW("get the option CORE_TYPE fail, set it to default value VECTOR_ENGINE");
  }
  int64_t core_type = (ge_core_type == kVectorCore) ? 1 : 0;
  GELOGI("core_type: %ld", core_type);
  if (!ge::AttrUtils::SetInt(&model, ATTR_MODEL_CORE_TYPE, core_type)) {
    REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s in model failed", ATTR_MODEL_CORE_TYPE.c_str());
    GELOGE(FAILED, "[Set][Attr] %s in model failed", ATTR_MODEL_CORE_TYPE.c_str());
  }
  GE_CHK_STATUS_RET_NOLOG(InitL1FusionOption());

  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetBool(&model, ATTR_NAME_SWITCH_FOR_L1_FUSION, is_l1_fusion_enable_),
                   REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s in model failed", ATTR_NAME_SWITCH_FOR_L1_FUSION.c_str());
                   GELOGE(FAILED, "[Set][Attr] %s in model failed.", ATTR_NAME_SWITCH_FOR_L1_FUSION.c_str());
                   return FAILED);

  model.SetName(compute_graph_->GetName());
  model.SetGraph(compute_graph_);
  GELOGI("weight_offset_: %zu event_num: %ld notify_num: %ld.", weight_offset_, event_num_, notify_num_);

  if (Params::Instance() == nullptr) {
    return FAILED;
  }

  platform_type_ = Params::Instance()->GetTarget_8bit();
  return SUCCESS;
}

void ModelBuilder::ClearOriginalFormat() const {
  for (const ge::NodePtr &n : compute_graph_->GetNodes(compute_graph_->GetGraphUnknownFlag())) {
    auto node_op_desc = n->GetOpDesc();
    if (node_op_desc != nullptr) {
      if (node_op_desc->HasAttr(ATTR_NAME_FORMAT)) {
        if (node_op_desc->DelAttr(ATTR_NAME_FORMAT) != SUCCESS) {
          GELOGW("DelAttr ATTR_NAME_FORMAT failed.");
        }
      }

      GE_IF_BOOL_EXEC(
        node_op_desc->HasAttr(ATTR_NAME_INFERRED_FORMAT),
        if (node_op_desc->DelAttr(ATTR_NAME_INFERRED_FORMAT) != SUCCESS) {
          GELOGW("DelAttr ATTR_NAME_INFERRED_FORMAT failed.");
        });

      GE_IF_BOOL_EXEC(
        node_op_desc->HasAttr(ATTR_NAME_PRED_PERMUTE_DELETED),
        if (node_op_desc->DelAttr(ATTR_NAME_PRED_PERMUTE_DELETED) != SUCCESS) {
          GELOGW("DelAttr ATTR_NAME_PRED_PERMUTE_DELETED failed.");
        });

      GE_IF_BOOL_EXEC(
        node_op_desc->HasAttr(ATTR_NAME_IGNORE_PRED_FORMAT),
        if (node_op_desc->DelAttr(ATTR_NAME_IGNORE_PRED_FORMAT) != SUCCESS) {
          GELOGW("DelAttr ATTR_NAME_IGNORE_PRED_FORMAT failed.");
        });
    }
  }
}

Status ModelBuilder::MergeWeights() {
  if (weight_offset_ == 0) {
    return SUCCESS;
  }

  ge::Buffer buffer(weight_offset_);
  weight_buffer_ = buffer;
  auto base_addr = weight_buffer_.GetData();

  for (const ge::NodePtr &node : compute_graph_->GetNodes(compute_graph_->GetGraphUnknownFlag())) {
    auto op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(op_desc == nullptr, continue);
    if (node->GetType() != CONSTANT) {
      continue;
    }

    // Get const op weight pointer
    ge::GeTensorPtr weight = nullptr;
    // If MutableTensor failed, weight is nullptr.
    (void)ge::AttrUtils::MutableTensor(op_desc, ATTR_NAME_WEIGHTS, weight);
    if (weight == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Can't get const weight in op:%s(%s)",
                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(FAILED, "[Call][MutableTensor] Can't get const op weight, name:%s", node->GetName().c_str());
      return FAILED;
    }

    // Get const op weight offset
    int64_t offset = 0;
    if (ge::TensorUtils::GetDataOffset(weight->GetTensorDesc(), offset) != SUCCESS) {
      GELOGW("Can't get const op offset, name: %s", node->GetName().c_str());
      continue;  // continue to merge if can not get offset
    }

    // Get const op weight data
    auto weight_data = weight->MutableData();

    // copy const op weight data to buffer
    GELOGI("Move to buffer, name: %s offset: %ld size: %zu", node->GetName().c_str(), offset, weight_data.size());
    ge::TensorUtils::SetWeightSize(weight->MutableTensorDesc(), static_cast<int64_t>(weight_data.size()));
    if ((offset == 0) || (weight_data.size() == 0)) {
      GELOGI("Size or offset is 0. size: %lu offset: %ld", weight_data.size(), offset);
      continue;
    }
    if (weight_offset_need_feeded_.find(static_cast<size_t>(offset)) == weight_offset_need_feeded_.end()) {
      GELOGI("name %s weight offset mem %ld has been feeded, no need feeded again", node->GetName().c_str(), offset);
      weight->ClearData();
      continue;
    }
    if (weight_data.data() != nullptr) {
      GE_IF_BOOL_EXEC(base_addr == nullptr,
                      REPORT_INNER_ERR_MSG("E19999", "Check weight in op:%s(%s) is nullptr",
                                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
                      GELOGE(FAILED, "[Check][Param] weight in op:%s(%s) is nullptr",
                             op_desc->GetName().c_str(), op_desc->GetType().c_str());
                      return FAILED);
      if (weight_offset_ - offset < weight_data.size()) {
        REPORT_INNER_ERR_MSG("E19999", "left weight size not enough for op:%s(%s) left_size:%zu, weight_size:%zu",
                           op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                           weight_offset_ - offset, weight_data.size());
        GELOGE(FAILED, "[Check][Param] left weight size not enough for op:%s(%s). left_size:%lu, weight_size:%lu",
               op_desc->GetName().c_str(), op_desc->GetType().c_str(), weight_offset_ - offset, weight_data.size());
        return FAILED;
      }
      uintptr_t dst_ptr = reinterpret_cast<uintptr_t>(base_addr) + offset;
      uintptr_t src_ptr = reinterpret_cast<uintptr_t>(weight_data.data());
      size_t left_size = weight_data.size();
      while (left_size > SECUREC_MEM_MAX_LEN) {
        auto err = memcpy_s(reinterpret_cast<void *>(dst_ptr), SECUREC_MEM_MAX_LEN, reinterpret_cast<void *>(src_ptr),
                            SECUREC_MEM_MAX_LEN);
        GE_ASSERT_EOK(err, "mem copy failed. err_ret:%d, dst_ptr:%lx, dst_size:%lu, src_ptr:%lx, src_size:%lu", err,
                      dst_ptr, SECUREC_MEM_MAX_LEN, src_ptr, SECUREC_MEM_MAX_LEN);
        left_size -= SECUREC_MEM_MAX_LEN;
        dst_ptr = dst_ptr + SECUREC_MEM_MAX_LEN;
        src_ptr = src_ptr + SECUREC_MEM_MAX_LEN;
      }
      auto err = memcpy_s(reinterpret_cast<void *>(dst_ptr), left_size, reinterpret_cast<void *>(src_ptr), left_size);
      GE_ASSERT_EOK(err, "mem copy failed. err ret:%d, dst_ptr:%lx, dst_size:%lu, src_ptr:%lx, src_size:%lu,", err,
                    dst_ptr, SECUREC_MEM_MAX_LEN, src_ptr, SECUREC_MEM_MAX_LEN);
      weight_offset_need_feeded_.erase(static_cast<size_t>(offset));
    }
    weight->ClearData();
  }

  return SUCCESS;
}

Status ModelBuilder::SavaAtomicWorkspace(const OpDescPtr &op_desc) const {
  auto workspace_info = op_desc->TryGetExtAttr(
      EXT_ATTR_ATOMIC_WORKSPACE_INFO, std::map<std::string, std::map<int64_t, int64_t>>{});
  if (workspace_info.empty()) {
    return SUCCESS;
  }
  GeAttrValue::NAMED_ATTRS workspaces;
  for (const auto &iter : workspace_info) {
    const std::string &op_name = iter.first;
    const auto &index_offset_map = iter.second;
    std::vector<int64_t> value;
    for (const auto &iter2 : index_offset_map) {
      value.emplace_back(iter2.first);
      value.emplace_back(iter2.second);
    }
    workspaces.SetAttr(op_name, GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>(value));
  }
  (void)AttrUtils::SetNamedAttrs(op_desc, EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspaces);
  return SUCCESS;
}

Status ModelBuilder::SaveAtomicTBEKernel(const OpDescPtr &op_desc) {
  ge::NodePtr atomic_clean_node = nullptr;
  atomic_clean_node = op_desc->TryGetExtAttr("memset_node_ptr", atomic_clean_node);
  if (atomic_clean_node == nullptr) {
    return SUCCESS;
  }

  ge::OpDescPtr atomic_op_desc = atomic_clean_node->GetOpDesc();
  GE_CHECK_NOTNULL(atomic_op_desc);
  TBEKernelPtr tbe_kernel = atomic_op_desc->TryGetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, TBEKernelPtr());
  if (tbe_kernel == nullptr) {
    std::string kernel_name;
    Buffer kernel_buffer;
    (void) AttrUtils::GetStr(atomic_op_desc, ATTR_NAME_TBE_KERNEL_NAME, kernel_name);
    (void) AttrUtils::GetBytes(atomic_op_desc, ATTR_NAME_TBE_KERNEL_BUFFER, kernel_buffer);
    if (!kernel_name.empty() && (kernel_buffer.GetSize() > 0)) {
      GE_CHECK_NOTNULL(kernel_buffer.GetData());
      std::vector<char> data(kernel_buffer.GetData(), kernel_buffer.GetData() + kernel_buffer.GetSize());
      tbe_kernel = MakeShared<OpKernelBin>(kernel_name, std::move(data));
      GE_CHECK_NOTNULL(tbe_kernel);
      GELOGI("Node [%s][%s] start recovery extra attr %s from %s", atomic_op_desc->GetName().c_str(),
             atomic_op_desc->GetType().c_str(), ge::OP_EXTATTR_NAME_TBE_KERNEL, ATTR_NAME_TBE_KERNEL_NAME.c_str());
      if (!(atomic_op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel))) {
        std::string error = "Node" + FmtToStr(atomic_op_desc->GetName()) + "set extra tbeKernel attr failed";
        GE_ERRORLOG_AND_ERRORMSG(ge::FAILED, error.c_str());
        return ge::FAILED;
      }
    }
  }
  if (tbe_kernel == nullptr) {
    GELOGD("Atomic_clean_node doesn't have tbe_kernel.");
    return SUCCESS;
  }
  auto atomic_kernel_key = kAtomicPrefix + std::string(ge::OP_EXTATTR_NAME_TBE_KERNEL);
  if (!(op_desc->SetExtAttr(atomic_kernel_key, tbe_kernel))) {
    std::string error = "Node" + FmtToStr(atomic_op_desc->GetName()) + "set extra tbeKernel attr failed";
    GE_ERRORLOG_AND_ERRORMSG(ge::FAILED, error.c_str());
    return ge::FAILED;
  }

  tbe_kernel_store_.AddTBEKernel(tbe_kernel);
  GELOGD("Atomic_clean_node tbe_kernel_name %s!", tbe_kernel->GetName().c_str());
  (void) AttrUtils::SetStr(op_desc, ATOMIC_ATTR_TBE_KERNEL_NAME, tbe_kernel->GetName());

  std::string kernel_name;
  (void) AttrUtils::GetStr(atomic_op_desc, atomic_op_desc->GetName() + "_kernelname", kernel_name);
  // Compat for compiler changes: remove prefix (node name) of attr name for symbol of kernel elf
  if (kernel_name.empty()) {
    (void) AttrUtils::GetStr(atomic_op_desc, kAttrEntrySymbolOfElf, kernel_name);
  }
  (void) AttrUtils::SetStr(op_desc, op_desc->GetName() + "_atomic_kernelname", kernel_name);
  std::string kernel_name_for_atomic = kAtomicPrefix + op_desc->GetName() + "_kernelname";
  (void) AttrUtils::SetStr(op_desc, kernel_name_for_atomic, kernel_name);
  GELOGI("op %s set attr name %s", op_desc->GetName().c_str(), kernel_name_for_atomic.c_str());

  std::string meta_data;
  (void) AttrUtils::GetStr(atomic_op_desc, TVM_ATTR_NAME_METADATA, meta_data);
  (void) AttrUtils::SetStr(op_desc, ATOMIC_ATTR_TVM_METADATA, meta_data);
  std::string meta_data_for_atomic = kAtomicPrefix + TVM_ATTR_NAME_METADATA;
  (void) AttrUtils::SetStr(op_desc, meta_data_for_atomic, meta_data);
  GELOGI("op %s set attr name %s", op_desc->GetName().c_str(), meta_data_for_atomic.c_str());

  std::string json_string;
  (void) AttrUtils::GetStr(atomic_op_desc, TVM_ATTR_NAME_MAGIC, json_string);
  (void) AttrUtils::SetStr(op_desc, ATOMIC_ATTR_TVM_MAGIC, json_string);
  std::string json_string_for_atomic = kAtomicPrefix + TVM_ATTR_NAME_MAGIC;
  (void) AttrUtils::SetStr(op_desc, json_string_for_atomic, json_string);
  GELOGI("op %s set attr name %s", op_desc->GetName().c_str(), json_string_for_atomic.c_str());
  return SUCCESS;
}

Status ModelBuilder::SaveNormalTBEKernel(const OpDescPtr &op_desc) {
  TBEKernelPtr tbe_kernel = op_desc->TryGetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, TBEKernelPtr());
  if (tbe_kernel == nullptr) {
    tbe_kernel = CreateOpTBEKernel(op_desc, "");
  }
  if (tbe_kernel == nullptr) {
    return SUCCESS; // Not TBE node.
  }
  (void)AttrUtils::SetStr(op_desc, "_kernelname", tbe_kernel->GetName());
  tbe_kernel_store_.AddTBEKernel(tbe_kernel);

  // Compat for compiler changes: remove prefix (node name) of attr name for symbol of kernel elf
  std::string symbol_of_elf;
  if (AttrUtils::GetStr(op_desc, kAttrEntrySymbolOfElf, symbol_of_elf) && !symbol_of_elf.empty()) {
    GE_ASSERT_TRUE(AttrUtils::SetStr(op_desc, op_desc->GetName() + kAttrEntrySymbolOfElf, symbol_of_elf));
  }
  return SUCCESS;
}

Status ModelBuilder::SaveCustAiCpuKernel(const OpDescPtr &op_desc, std::set<std::string> &aicpu_name_set) {
  const auto cust_aicpu_kernel = op_desc->TryGetExtAttr(OP_EXTATTR_CUSTAICPU_KERNEL, CustAICPUKernelPtr());
  if (cust_aicpu_kernel == nullptr) {
    return SUCCESS; // Not cust aicpu node.
  }

  if (aicpu_name_set.count(cust_aicpu_kernel->GetName()) > 0) {
    REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char_t *>({"value", "parameter", "reason"}),
                       std::vector<const char_t *>({cust_aicpu_kernel->GetName().c_str(), op_desc->GetName().c_str(),
                                                "Parameter aicpu_kernel_name must be unique."}));
    GELOGE(FAILED, "[Check][Param] aicpu_kernel name %s can't be the same, judge for op:%s(%s)",
           cust_aicpu_kernel->GetName().c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }
  aicpu_name_set.insert(cust_aicpu_kernel->GetName());
  cust_aicpu_kernel_store_.AddCustAICPUKernel(cust_aicpu_kernel);
  GELOGI("Add cust aicpu kernel bin %s", cust_aicpu_kernel->GetName().c_str());
  return SUCCESS;
}

Status ModelBuilder::SaveFftsPlusTBEKernel(const OpDescPtr &op_desc) {
  const auto thread_tbe_kernel =
      op_desc->TryGetExtAttr(OP_EXTATTR_NAME_THREAD_TBE_KERNEL, std::vector<OpKernelBinPtr>{});
  for (size_t i = 0UL; i < thread_tbe_kernel.size(); ++i) {
    tbe_kernel_store_.AddTBEKernel(thread_tbe_kernel[i]);
  }

  const auto SaveMixTBE = [&op_desc, this](const std::string &prefix, const std::string &core_type) {
    TBEKernelPtr tbe_kernel = op_desc->TryGetExtAttr(prefix + OP_EXTATTR_NAME_TBE_KERNEL, TBEKernelPtr());
    if (tbe_kernel == nullptr) {
      tbe_kernel = CreateOpTBEKernel(op_desc, prefix);
    }
    GE_CHECK_NOTNULL(tbe_kernel);
    tbe_kernel_store_.AddTBEKernel(tbe_kernel);
    GELOGD("Add %s kernel bin, Op(%s:%s)", core_type.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return SUCCESS;
  };

  std::vector<std::string> names_prefix;
  (void)AttrUtils::GetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
  if (!names_prefix.empty()) {
    std::string core_type;
    (void)AttrUtils::GetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
    for (const auto &prefix : names_prefix) {
      GE_CHK_STATUS_RET_NOLOG(SaveMixTBE(prefix, core_type));
    }
  }

  return SUCCESS;
}

TBEKernelPtr ModelBuilder::CreateOpTBEKernel(const OpDescPtr &op_desc, const std::string &prefix_kernel_name) const {
  std::string kernel_name;
  Buffer kernel_buffer;
  TBEKernelPtr tbe_kernel = nullptr;
  (void)AttrUtils::GetStr(op_desc, prefix_kernel_name + ATTR_NAME_TBE_KERNEL_NAME, kernel_name);
  (void)AttrUtils::GetBytes(op_desc, prefix_kernel_name + ATTR_NAME_TBE_KERNEL_BUFFER, kernel_buffer);
  if (!kernel_name.empty() && (kernel_buffer.GetSize() > 0U)) {
    if (kernel_buffer.GetData() == nullptr) {
      GELOGW("kernel data of op:%s(%s) is nullptr", op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return nullptr;
    }
    std::vector<char> data(kernel_buffer.GetData(), kernel_buffer.GetData() + kernel_buffer.GetSize());
    tbe_kernel = MakeShared<OpKernelBin>(kernel_name, std::move(data));
    GE_CHK_BOOL_EXEC(tbe_kernel != nullptr, return nullptr, "[Create][TBEKernel] failed");
    const std::string ext_attr_name = prefix_kernel_name + OP_EXTATTR_NAME_TBE_KERNEL;
    if (!op_desc->SetExtAttr(ext_attr_name, tbe_kernel)) {
      GELOGW("set ext attr:%s for op:%s(%s) failed", ext_attr_name.c_str(),
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return nullptr;
    }
  }
  return tbe_kernel;
}

Status ModelBuilder::SaveDataToModel(ge::Model &model, ge::GeModel &ge_model) {
  // Add weight
  ge_model.SetWeight(weight_buffer_);

  // Add TBE Kernels and custom aicpu op bin
  std::set<std::string> aicpu_name_set;
  std::set<std::string> aicpu_op_types;
  std::set<std::string> aicpu_tf_op_types;
  for (const auto &node : compute_graph_->GetNodes(compute_graph_->GetGraphUnknownFlag())) {
    const auto node_op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, continue);
    // check aicpu op type
    CollectCheckAicpuAttr(node_op_desc, aicpu_op_types, aicpu_tf_op_types);
    GE_CHK_STATUS_RET_NOLOG(SaveNormalTBEKernel(node_op_desc));
    GE_CHK_STATUS_RET_NOLOG(SaveCustAiCpuKernel(node_op_desc, aicpu_name_set));
    GE_CHK_STATUS_RET_NOLOG(SaveFftsPlusTBEKernel(node_op_desc));
    GE_CHK_STATUS_RET(SaveAtomicTBEKernel(node_op_desc),
                      "[Save][TBEKernel] Node[%s] type[%s] save atomic tbekernel failed!",
                      node_op_desc->GetName().c_str(), node_op_desc->GetType().c_str());
    GE_CHK_STATUS_RET(SavaAtomicWorkspace(node_op_desc),
                      "[Save][TBEKernel] Node[%s] type[%s] save atomic work space failed!",
                      node_op_desc->GetName().c_str(), node_op_desc->GetType().c_str());

    if ((!compute_graph_->GetGraphUnknownFlag()) || (node_op_desc->GetType() != PARTITIONEDCALL)) {
      continue;
    }
    // For dynamic FFTS-Plus subgraph node.
    if (node_op_desc->HasAttr(ATTR_NAME_FFTS_SUB_GRAPH) || node_op_desc->HasAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH)) {
      const auto sgt_graph = compute_graph_->GetSubgraph(node_op_desc->GetSubgraphInstanceName(0U));
      GE_IF_BOOL_EXEC(sgt_graph == nullptr, continue);
      for (const auto &sgt_node : sgt_graph->GetAllNodes()) {
        const auto sgt_op_desc = sgt_node->GetOpDesc();
        GE_IF_BOOL_EXEC(sgt_op_desc == nullptr, continue);
        GE_CHK_STATUS_RET_NOLOG(SaveNormalTBEKernel(sgt_op_desc));
        GE_CHK_STATUS_RET_NOLOG(SaveCustAiCpuKernel(sgt_op_desc, aicpu_name_set));
        GE_CHK_STATUS_RET_NOLOG(SaveFftsPlusTBEKernel(sgt_op_desc));
        GE_CHK_STATUS_RET_NOLOG(SaveAtomicTBEKernel(sgt_op_desc));
        GE_CHK_STATUS_RET_NOLOG(SavaAtomicWorkspace(sgt_op_desc));
      }
    }
  }

  SetModelCheckAicpuAttr(model, aicpu_op_types, aicpu_tf_op_types);

  if (!tbe_kernel_store_.Build()) {
    GELOGE(FAILED, "[Call][Build] TBE Kernels store build failed!");
    return FAILED;
  }
  if (!cust_aicpu_kernel_store_.Build()) {
    GELOGE(FAILED, "[Call][Build] custom AICPU kernels store build failed!");
    return FAILED;
  }
  ge_model.SetTBEKernelStore(tbe_kernel_store_);
  ge_model.SetCustAICPUKernelStore(cust_aicpu_kernel_store_);
  DelNodeRepeatSaveAttr();

  // Add task
  Buffer task_def_bytes;
  if (!AttrUtils::GetZeroCopyBytes(model, MODEL_ATTR_TASKS, task_def_bytes)) {
    REPORT_INNER_ERR_MSG("E19999", "Get attr:%s in model failed", MODEL_ATTR_TASKS.c_str());
    GELOGE(INTERNAL_ERROR, "[Get][Attr] %s in model failed", MODEL_ATTR_TASKS.c_str());
    return INTERNAL_ERROR;
  }
  int32_t byte_size = static_cast<int32_t>(task_def_bytes.GetSize());
  std::shared_ptr<domi::ModelTaskDef> task = ge::MakeShared<domi::ModelTaskDef>();
  GE_CHECK_NOTNULL(task);
  GE_CHK_BOOL_EXEC(ReadProtoFromArray(task_def_bytes.GetData(), byte_size, task.get()), return INTERNAL_ERROR,
                   "[Read][Proto] From Array failed.");
  ge_model.SetModelTaskDef(task);

  // Add graph
  ge_model.SetName(model.GetName());
  ge_model.SetGraph(model.GetGraph());
  ge_model.SetVersion(model.GetVersion());
  ge_model.SetPlatformVersion(model.GetPlatformVersion());
  ge_model.SetPlatformType(platform_type_);
  ge_model.SetAttrMap(model.MutableAttrMap());

  if (IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    const ModelPtr model_for_print = ge::MakeShared<ge::Model>(ge_model.GetName(), ge_model.GetPlatformVersion());
    GE_CHECK_NOTNULL(model_for_print);
    model_for_print->SetGraph(model.GetGraph());
    model_for_print->SetVersion(ge_model.GetVersion());
    model_for_print->SetAttr(ge_model.MutableAttrMap());
    ge::Buffer model_buff;
    (void)model_for_print->Save(model_buff);
    const size_t model_buff_size = model_buff.GetSize();
    const size_t weight_size = ge_model.GetWeightSize();
    const size_t tbe_kernelstore_size = ge_model.GetTBEKernelStore().DataSize();
    const size_t aicpu_kernelstore_size = ge_model.GetCustAICPUKernelStore().DataSize();
    const size_t task_size = ge_model.GetModelTaskDefPtr()->ByteSizeLong();
    const size_t total_size = model_buff_size + weight_size + tbe_kernelstore_size + aicpu_kernelstore_size + task_size;
    GELOGD(
        "Print model total size:%zu, model def size:%zu, weight data size:%zu, "
        "tbe kernel size:%zu, cust aicpu kernel size:%zu, task info size:%zu",
        total_size, model_buff_size, weight_size, tbe_kernelstore_size, aicpu_kernelstore_size, task_size);
  }
  return SUCCESS;
}

void ModelBuilder::DelNodeRepeatSaveAttr() {
  for (const ge::NodePtr &n : compute_graph_->GetNodes(compute_graph_->GetGraphUnknownFlag())) {
    auto node_op_desc = n->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, continue);
    (void)node_op_desc->DelAttr(ATTR_NAME_TBE_KERNEL_BUFFER);
    (void)node_op_desc->DelAttr(ATTR_NAME_TBE_KERNEL_NAME);
    // Load need kernel_name attr as key, so just remove kernel_buffer, which is more larger
    std::vector<std::string> names_prefix;
    (void)AttrUtils::GetListStr(node_op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
    for (const auto &prefix : names_prefix) {
      (void)node_op_desc->DelAttr(prefix + ATTR_NAME_TBE_KERNEL_BUFFER);
    }
  }
}

void ModelBuilder::SetModelVersion(ge::Model &model) const {
  // set framework_version TO model
  std::string framework_version;
  uint32_t counter = 0;
  Status frame_rt = PlatformVersionManager::GetPlatformVersion(framework_version);
  GE_IF_BOOL_EXEC((frame_rt == SUCCESS),
                  std::string model_framework_version = framework_version + "." + std::to_string(counter);
                  model.SetPlatformVersion(model_framework_version););

  // set IR Version TO model
  model.SetVersion(static_cast<uint32_t>(OM_PROTO_VERSION));
}

Status ModelBuilder::PreBuildModel() {
  if ((compute_graph_ == nullptr) || !(compute_graph_->IsValid())) {
    REPORT_INNER_ERR_MSG("E19999", "Check compute_graph no valid");
    GELOGE(FAILED, "[Check][Param] Graph_ is not valid.");
    return FAILED;
  }

  GE_CHK_STATUS_RET(SetInputOutputDesc(),
                    "[Set][InputOutputDesc] Failed! graph:%s", compute_graph_->GetName().c_str());

  AddNodeInputProperty();

  return SUCCESS;
}

Status ModelBuilder::RefreshRealStream(
    std::unordered_map<int64_t, std::vector<domi::TaskDef>> &node_id_2_node_tasks) {
  GE_ASSERT_SUCCESS(
      stream_allocator_.SplitStreamAndRefreshTaskDef(node_id_2_node_tasks, stream_num_, event_num_, notify_num_),
      "SplitStreamAndRefreshTaskDef failed, graph:%s", compute_graph_->GetName().c_str());
  huge_streams_ = stream_allocator_.GetHugeStreams();
  return SUCCESS;
}

Status ModelBuilder::BuildModelForGetTask(ge::Model &model) {
  GE_CHK_STATUS_RET(AdjustInputTensorFlag(), "[Adjust][InputTensorFlag] failed! graph:%s",
                    compute_graph_->GetName().c_str());

  // Assign logical streams.
  GE_TRACE_START(AssignLogicalStreams);
  GE_ASSERT_SUCCESS(stream_allocator_.AssignLogicalStreams(stream_max_parallel_num_, hcom_parallel_),
                    "[Assign][LogicalStreams] failed. graph:%s", compute_graph_->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(AssignLogicalStreams, "GraphBuilder::AssignLogicalStreams");

  // Assign functional op labels.
  auto root_graph = GraphUtils::FindRootGraph(compute_graph_);
  (void)AttrUtils::GetInt(*root_graph, ATTR_MODEL_LABEL_NUM, label_num_);

  GE_TRACE_START(AssignMemory);
  MemoryAssigner mem_assigner(compute_graph_);
  GE_CHK_STATUS_RET(mem_assigner.AssignMemory(mem_type_to_mem_offset_, zero_copy_mem_size_, GetHasAssignedVarMemFlag()),
                    "[Assign][Memory] Failed! graph:%s", compute_graph_->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(AssignMemory, "GraphBuilder::AssignMemory");
  sub_mem_offsets_ = mem_assigner.GetSubMemOffsets();

  GE_TRACE_START(SetInputOutputOffset);
  PassManager io_offset_pass_manager;
  GE_CHK_STATUS_RET(
      io_offset_pass_manager.AddPass("SetInputOutputOffsetPass", new (std::nothrow) SetInputOutputOffsetPass));
  GE_CHK_STATUS_RET(io_offset_pass_manager.Run(compute_graph_), "[Set][InputOutputOffset] failed. graph:%s",
                    compute_graph_->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(SetInputOutputOffset, "SetInputOutputOffsetPass::Run");

  // Compile single op in graph build stage
  GE_TRACE_START(CompileSingleOp);
  GE_CHK_STATUS_RET(CompileSingleOp(), "[Compile][SingleOp] fail. graph:%s", compute_graph_->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(CompileSingleOp, "GraphBuilder::CompileSingleOp");

  // insert event notify nodes by logical stream id.
  GE_TRACE_START(InsertSyncNodesByLogicStream);
  GE_ASSERT_SUCCESS(stream_allocator_.InsertSyncNodesByLogicStream(stream_num_, event_num_, notify_num_),
                    "[Refresh][RealStream] failed. graph:%s", compute_graph_->GetName().c_str());
  notify_types_ = stream_allocator_.GetNotifyTypes();
  GE_ASSERT_EQ(static_cast<int64_t>(notify_types_.size()), notify_num_);
  GE_COMPILE_TRACE_TIMESTAMP_END(InsertSyncNodesByLogicStream, "GraphBuilder::InsertSyncNodesByLogicStream");
  GE_TRACE_START(OptimizeStreamedWholeGraph);
  StreamGraphOptimizer stream_graph_optimizer;
  GE_CHK_STATUS_RET(stream_graph_optimizer.OptimizeStreamedWholeGraph(compute_graph_),
                    "[Optimize][StreamedWholeGraph] fail. graph:%s", compute_graph_->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(OptimizeStreamedWholeGraph, "GraphBuilder::OptimizeStreamedWholeGraph");

  GE_CHK_STATUS_RET(SaveSoftSyncOpWeight(), "[Save][Weights] Failed! graph:%s", compute_graph_->GetName().c_str());

  GE_TRACE_START(MergeWeights);
  GE_CHK_STATUS_RET(MergeWeights(), "[Merge][Weights] Failed! graph:%s", compute_graph_->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(MergeWeights, "GraphBuilder::MergeWeights");

  GE_TRACE_START(BuildModelDefForMem);
  GE_ASSERT_SUCCESS(BuildModelDefForMem(model), "[Build][ModelDef] Part one failed! graph:%s",
                    compute_graph_->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(BuildModelDefForMem, "GraphBuilder::BuildModelDefForMem");

  SetModelVersion(model);

  return SUCCESS;
}

Status ModelBuilder::BuildModelDefForStream(ge::Model &model) {
  GE_ASSERT_TRUE(ge::AttrUtils::SetInt(&model, ATTR_MODEL_STREAM_NUM, stream_num_), "[Set][Attr] %s in model failed",
                 ATTR_MODEL_STREAM_NUM.c_str());
  GE_ASSERT_TRUE(ge::AttrUtils::SetInt(&model, ATTR_MODEL_NOTIFY_NUM, notify_num_), "[Set][Attr] %s in model failed",
                 ATTR_MODEL_NOTIFY_NUM.c_str());
  GE_ASSERT_TRUE(ge::AttrUtils::SetListInt(&model, ATTR_MODEL_NOTIFY_TYPES, notify_types_));
  GE_ASSERT_TRUE(ge::AttrUtils::SetInt(&model, ATTR_MODEL_EVENT_NUM, event_num_), "[Set][Attr] %s in model failed",
                 ATTR_MODEL_EVENT_NUM.c_str());
  GE_ASSERT_TRUE(ge::AttrUtils::SetListInt(&model, ATTR_MODEL_HUGE_STREAM_LIST, huge_streams_),
                 "[Set][Attr] %s in model failed", ATTR_MODEL_HUGE_STREAM_LIST.c_str());
  const auto graph = model.GetGraph();
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_TRUE(ge::AttrUtils::SetStr(graph, "_split_logic_stream_2_origin_logic_stream",
                                       StreamUtils::TransMapToStr(stream_allocator_.GetSplitStreamToLogicStream())));
  GELOGI("build model def about stream, stream num: %ld, event_num: %ld, notify_num: %ld", stream_num_, event_num_,
         notify_num_);
  return SUCCESS;
}

Status ModelBuilder::SaveSoftSyncOpWeight() const {
  GELOGD("Start to recover soft sync op's weight of graph: %s.", compute_graph_->GetName().c_str());
  for (const auto &node : compute_graph_->GetAllNodes()) {
    const auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    bool is_soft_sync = false;
    if ((!ge::AttrUtils::GetBool(op_desc, ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, is_soft_sync)) || (!is_soft_sync)) {
      continue;
    }
    const auto depend_names = op_desc->GetOpInferDepends();
    if (depend_names.empty()) {
      continue;
    }
    GE_ASSERT_SUCCESS(SaveSoftSyncOpWeightByDependNames(node, depend_names));
  }
  GELOGD("Finished to recover soft sync op's weight of graph: %s.", compute_graph_->GetName().c_str());
  return SUCCESS;
}

Status ModelBuilder::BuildModelForGetDynShapeTask(ge::Model &model_def) {
  GE_TRACE_START(BuildModelDef);
  GE_CHK_STATUS_RET(BuildModelDef(model_def), "[Build][ModelDef] failed!");
  GE_COMPILE_TRACE_TIMESTAMP_END(BuildModelDef, "GraphBuilder::BuildModelDef");
  SetModelVersion(model_def);
  return SUCCESS;
}

ge::Buffer ModelBuilder::GetWeightBuffer() const { return weight_buffer_; }
Status ModelBuilder::CompileSingleOp() const {
  GELOGD("Begin to compile single op.");
  // Create ge instance
  std::shared_ptr<GELib> instance = ge::GELib::GetInstance();
  if ((instance == nullptr) || !instance->InitFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "Check GELib instance not init before");
    GELOGE(ge::GE_CLI_GE_NOT_INITIALIZED, "[Check][Param] CompileSingleOp failed.");
    return ge::GE_CLI_GE_NOT_INITIALIZED;
  }

  GE_TIMESTAMP_CALLNUM_START(BatchCompileOp);
  std::unordered_map<std::string, std::vector<ge::NodePtr>> node_vector_map;
  for (auto &node : compute_graph_->GetNodes(compute_graph_->GetGraphUnknownFlag())) {
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }

    //  Graph build stage only supports the individual compilation of atomic clean operator
    if (NodeUtils::IsLikeAtomicClean(node)) {
      std::string kernel_lib_name = op_desc->GetOpKernelLibName();
      if (kernel_lib_name.empty()) {
        // Reset op kernel lib
        (void)instance->DNNEngineManagerObj().GetDNNEngineName(node);
        kernel_lib_name = op_desc->GetOpKernelLibName();
        if (kernel_lib_name.empty()) {
          REPORT_INNER_ERR_MSG("E19999", "Check kernel lib name empty of op:%s(%s)",
                             node->GetName().c_str(), node->GetType().c_str());
          GELOGE(ge::INTERNAL_ERROR, "[Get][Name] of node:%s(%s) kernel lib failed.", node->GetName().c_str(),
                 node->GetType().c_str());
          return ge::INTERNAL_ERROR;
        }
      }
      GELOGI("Begin to compile single op, lib is %s, op name is %s, op type is %s.", kernel_lib_name.c_str(),
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      OpsKernelInfoStorePtr kernel_info = instance->OpsKernelManagerObj().GetOpsKernelInfoStore(kernel_lib_name);
      if (kernel_info != nullptr) {
        node_vector_map[kernel_lib_name].emplace_back(node);
      } else {
        REPORT_INNER_ERR_MSG("E19999", "Get ops kernel info store failed for op:%s(%s), op_kernel_name:%s,",
                           node->GetName().c_str(), node->GetType().c_str(), kernel_lib_name.c_str());
        GELOGE(ge::GE_GRAPH_PARAM_NULLPTR, "[Get][OpsKernelInfoStore] for op %s failed", node->GetName().c_str());
        return ge::GE_GRAPH_PARAM_NULLPTR;
      }
    }
  }
  for (auto &it : node_vector_map) {
    auto &kernel_lib_name = it.first;
    auto &node_vector = it.second;
    OpsKernelInfoStorePtr kernel_info = instance->OpsKernelManagerObj().GetOpsKernelInfoStore(kernel_lib_name);
    GE_CHECK_NOTNULL(kernel_info);
    GE_TIMESTAMP_RESTART(BatchCompileOp);
    auto ret = kernel_info->CompileOp(node_vector);
    GELOGI("[GEPERFTRACE] The node size of compile op of %s is %zu", kernel_lib_name.c_str(), node_vector.size());
    GE_TIMESTAMP_ADD(BatchCompileOp);
    if (ret != ge::SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Batch compile op failed, kernel lib name, node size:%zu,",
                        node_vector.size());
      GELOGE(ret, "[Compile][Op] failed, kernel lib name is %s", kernel_lib_name.c_str());
      return ret;
    }
  }
  GE_TIMESTAMP_CALLNUM_END(BatchCompileOp, "GraphBuild::CompileOp");
  return ge::SUCCESS;
}

void ModelBuilder::CollectCheckAicpuAttr(const OpDescPtr &op_desc, std::set<std::string> &aicpu_op_types,
                                         std::set<std::string> &aicpu_tf_op_types) const {
  std::string aicpu_optype;
  bool has_attr_check_cpu = ge::AttrUtils::GetStr(op_desc, "needCheckCpu", aicpu_optype);
  std::vector<std::string> tf_optypes;
  bool has_attr_check_tf = ge::AttrUtils::GetListStr(op_desc, "needCheckTf", tf_optypes);
  if (has_attr_check_cpu && !aicpu_optype.empty()) {
    aicpu_op_types.insert(aicpu_optype);
  }

  if (has_attr_check_tf && !tf_optypes.empty()) {
    aicpu_tf_op_types.insert(tf_optypes.cbegin(), tf_optypes.cend());
  }

  return;
}

void ModelBuilder::SetModelCheckAicpuAttr(ge::Model &model, std::set<std::string> &aicpu_op_types,
                                          std::set<std::string> &aicpu_tf_op_types) const {
  std::vector<std::string> aicpu_optype_list;
  std::vector<std::string> aicpu_tf_optype_list;
  if (ge::AttrUtils::GetListStr(&model, "needCheckCpu", aicpu_optype_list)) {
    GELOGI("Already have aicpu optype size: %zu", aicpu_optype_list.size());
    aicpu_op_types.insert(aicpu_optype_list.cbegin(), aicpu_optype_list.cend());
  }

  if (ge::AttrUtils::GetListStr(&model, "needCheckTf", aicpu_tf_optype_list)) {
    GELOGI("Already have aicpu tf optype size: %zu", aicpu_tf_optype_list.size());
    aicpu_tf_op_types.insert(aicpu_tf_optype_list.cbegin(), aicpu_tf_optype_list.cend());
  }

  // reset list with set
  aicpu_optype_list.assign(aicpu_op_types.begin(), aicpu_op_types.end());
  aicpu_tf_optype_list.assign(aicpu_tf_op_types.begin(), aicpu_tf_op_types.end());
  GELOGI("Check Aicpu op types ComputeGraph: %s aicpu_op_types: %zu, aicpu_optype_list: %zu, aicpu_tf_op_types: %zu, "
         "aicpu_tf_optype_list:%zu.",
         compute_graph_->GetName().c_str(), aicpu_op_types.size(), aicpu_optype_list.size(), aicpu_tf_op_types.size(),
         aicpu_tf_optype_list.size());
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListStr(&model, "needCheckCpu", aicpu_optype_list), return,
                   "[Set][Attr] needCheckCpu fail.");

  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListStr(&model, "needCheckTf", aicpu_tf_optype_list), return,
                   "[Set][Attr] needCheckTf fail.");
  return;
}

Status ModelBuilder::BuildModelForEvaluate(ModelDataInfo &model) const {
  // Assign logical streams.
  StreamAllocator stream_allocator(compute_graph_, subgraphs_);
  GE_TRACE_START(AssignLogicalStreams);
  GE_CHK_STATUS_RET(stream_allocator.AssignLogicalStreams(stream_max_parallel_num_, hcom_parallel_),
                    "[Assign][LogicalStreams] failed. graph:%s", compute_graph_->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(AssignLogicalStreams, "GraphBuilder::AssignLogicalStreams");

  GE_TRACE_START(AssignMemory);
  MemoryAssigner mem_assigner(compute_graph_);
  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  GE_CHK_STATUS_RET(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size),
                    "[Assign][Memory] Failed! graph:%s", compute_graph_->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(AssignMemory, "GraphBuilder::AssignMemory");
  size_t graph_memory_size = 0;
  for (const auto &memory_size : mem_type_to_mem_offset) {
    graph_memory_size += memory_size.second;
  }
  model.SetGraphMemorySize(graph_memory_size);
  const auto &var_manager = ge::VarManager::Instance(compute_graph_->GetSessionID());
  GE_ASSERT_NOTNULL(var_manager);
  model.SetVarMemorySize(var_manager->GetVarMemSize(RT_MEMORY_HBM));
  return SUCCESS;
}

Status ModelBuilder::AssignStreamForDynamicShapeGraph(ComputeGraphPtr &compute_graph) {
  if (compute_graph->GetParentGraph() != nullptr) {
    return SUCCESS;
  }

  if (!StreamUtils::EnableDynamicShapeMultiStream()) {
    return SUCCESS;
  }

  if (StreamUtils::EnableCvParallel()) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({ge::GetContext().GetReadableName("ge.autoMultistreamParallelMode").c_str(), "cv",
                                   "Dynamic multi-stream and CV parallel could not both enabled."}));
    GELOGE(FAILED, "dynamic multi stream and cv parallel could not both enable");
    return FAILED;
  }

  if (GraphUtils::IsSingleOpScene(compute_graph)) {
    return SUCCESS;
  }

  // to make sure topo id is accurate, to sorting before assign stream
  GE_CHK_STATUS_RET(compute_graph->TopologicalSorting(), "[Call][TopologicalSorting] failed, graph:%s",
                    compute_graph->GetName().c_str());

  const auto dynamic_stream_allocator = MakeShared<DynamicStreamAllocator>();
  GE_CHECK_NOTNULL(dynamic_stream_allocator);
  GE_ASSERT_SUCCESS(dynamic_stream_allocator->AssignStreamsForDynamicShapeGraph(compute_graph, subgraphs_));

  stream_num_ = dynamic_stream_allocator->GetStreamNum();
  event_num_ = dynamic_stream_allocator->GetEventNum();

  return SUCCESS;
}
}  // namespace ge

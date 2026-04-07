/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engines/local_engine/ops_kernel_store/ge_local_ops_kernel_builder.h"
#include <memory>
#include "common/checker.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_type_utils.h"
#include "engines/local_engine/ops_kernel_store/op/op_factory.h"
#include "engines/local_engine/common/constant/constant.h"
#include "register/ops_kernel_builder_registry.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "framework/common/op/ge_op_utils.h"
#include "base/err_msg.h"

namespace ge {
namespace ge_local {
REGISTER_OPS_KERNEL_BUILDER(kGeLocalOpKernelLibName, GeLocalOpsKernelBuilder);

namespace {
const char *const kConstantOpType = "Constant";
const char *const kFileConstantOpType = "FileConstant";
const char *const kConstOpType = "Const";
const std::unordered_set<std::string> kDependComputeOps = {"StackPop"};
const int64_t kMemAlignSize = 32;
int64_t AlignOutputMemSize(const int64_t mem_size) {
  const int64_t padding_size = TensorUtils::GetPaddingSize();
  GE_ASSERT_TRUE(mem_size < std::numeric_limits<int64_t>::max() - kMemAlignSize);
  return (mem_size + kMemAlignSize - 1) / kMemAlignSize * kMemAlignSize + padding_size;
}
using CalcOpParamCall = std::function<graphStatus(const Node &node)>;
std::map<std::string, CalcOpParamCall> calc_op_param_call = {
  {PHONYCONCAT, GeLocalOpsKernelBuilderCalcOpParam::CalcPhonyConcatNodeOffset},
  {PHONYSPLIT, GeLocalOpsKernelBuilderCalcOpParam::CalcPhonySplitNodeOffset},
  {"Bitcast", GeLocalOpsKernelBuilderCalcOpParam::CalcNodeOffsetByReuseInput},
  {"Flatten", GeLocalOpsKernelBuilderCalcOpParam::CalcNodeOffsetByReuseInput},
  {"FlattenV2", GeLocalOpsKernelBuilderCalcOpParam::CalcNodeOffsetByReuseInput},
  {"ExpandDims", GeLocalOpsKernelBuilderCalcOpParam::CalcNodeOffsetByReuseInput},
  {"ReFormat", GeLocalOpsKernelBuilderCalcOpParam::CalcNodeOffsetByReuseInput},
  {"Squeeze", GeLocalOpsKernelBuilderCalcOpParam::CalcNodeOffsetByReuseInput},
  {"Unsqueeze", GeLocalOpsKernelBuilderCalcOpParam::CalcNodeOffsetByReuseInput},
  {"SqueezeV2", GeLocalOpsKernelBuilderCalcOpParam::CalcNodeOffsetByReuseInput},
  {"UnsqueezeV2", GeLocalOpsKernelBuilderCalcOpParam::CalcNodeOffsetByReuseInput},
  {"SqueezeV3", GeLocalOpsKernelBuilderCalcOpParam::CalcNodeOffsetByReuseInput},
  {"UnsqueezeV3", GeLocalOpsKernelBuilderCalcOpParam::CalcNodeOffsetByReuseInput}
};
}  // namespace

GeLocalOpsKernelBuilder::~GeLocalOpsKernelBuilder() {
  GELOGI("GeLocalOpsKernelBuilder destroyed");
}

Status GeLocalOpsKernelBuilder::Initialize(const std::map<std::string, std::string> &options) {
  (void)options;
  return SUCCESS;
}

Status GeLocalOpsKernelBuilder::Finalize() {
  return SUCCESS;
}

graphStatus GeLocalOpsKernelBuilder::CalcMemSizeByNodeType(OpDescPtr &op_desc, GeTensorDesc &output_tensor,
                                                           int64_t &output_mem_size, const std::string &node_type) {
  GeShape output_shape = output_tensor.GetShape();
  Format format = output_tensor.GetFormat();
  DataType data_type = output_tensor.GetDataType();
  if ((data_type == DT_STRING) && ((node_type == kConstantOpType) || (node_type == kConstOpType))) {
    return OpUtils::GetConstantStrMemSize(op_desc, output_mem_size);
  } else if ((node_type == kFileConstantOpType) && AttrUtils::GetInt(op_desc, ATTR_NAME_LENGTH, output_mem_size)) {
    GELOGD("node: %s get length attr success. node type is FileConstant and data type is DT_STRING, size: %lld",
           op_desc->GetNamePtr(), output_mem_size);
    return GRAPH_SUCCESS;
  }
  if (OpTypeUtils::IsDataNode(node_type)) {
    return TensorUtils::GetTensorMemorySizeInBytes(output_tensor, output_mem_size);
  }
  bool is_no_tiling = false;
  (void)AttrUtils::GetBool(output_tensor, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, is_no_tiling);
  if (is_no_tiling) {
    return TensorUtils::CalcTensorMemSizeForNoTiling(output_tensor, format, data_type, output_mem_size);
  }
  GE_ASSERT_SUCCESS(TensorUtils::CalcTensorMemSize(output_shape, format, data_type, output_mem_size));
  if ((node_type == PHONYCONCAT) || (node_type == PARTITIONEDCALL)) {
    output_mem_size = AlignOutputMemSize(output_mem_size);
  }
  return SUCCESS;
}

graphStatus GeLocalOpsKernelBuilder::SetSizeForStringInput(Node &node, GeTensorDesc &output_tensor,
                                                           size_t output_index) {
  const std::string node_type = node.GetType();
  DataType data_type = output_tensor.GetDataType();
  if (!OpTypeUtils::IsDataNode(node_type) || (data_type != DT_STRING)) {
    return GRAPH_SUCCESS;
  }

  const auto &anchor = node.GetOutDataAnchor(output_index);
  if ((anchor == nullptr) || (anchor->GetFirstPeerAnchor() == nullptr) ||
      (anchor->GetFirstPeerAnchor()->GetOwnerNode() == nullptr)) {
    GELOGI("Param ge_node has no anchor, check invalid.");
    return GRAPH_SUCCESS;
  }

  std::vector<int64_t> op_max_size;
  const auto &node_desc = anchor->GetFirstPeerAnchor()->GetOwnerNode()->GetOpDesc();
  const auto anchor_idx = static_cast<size_t>(anchor->GetFirstPeerAnchor()->GetIdx());
  if (ge::AttrUtils::GetListInt(node_desc, "_op_max_size", op_max_size)) {
    if (op_max_size.size() <= anchor_idx) {
      REPORT_INNER_ERR_MSG("E19999", "anchor index invalid, anchor_idx:%zu, total:%zu", anchor_idx, op_max_size.size());
      GELOGE(FAILED, "[MaxSize] anchor index invalid, anchor_idx:%zu, total:%zu", anchor_idx, op_max_size.size());
      return FAILED;
    }
    GELOGI("set max size attr to node [%s] output tensor.", node.GetName().c_str());
    TensorUtils::SetSize(output_tensor, op_max_size[anchor_idx]);
  }
  return GRAPH_SUCCESS;
}

Status GeLocalOpsKernelBuilder::CalcOpRunningParam(Node &node) {
  GELOGD("[%s] CalcOpRunningParam In.", node.GetName().c_str());
  OpDescPtr op_desc = node.GetOpDesc();
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "param node has no opdesc, check invalid.");
    GELOGE(FAILED, "[Get][OpDesc] CalcOpRunningParam failed, as op desc is null");
    return FAILED;
  }

  bool is_no_tiling = false;
  (void)AttrUtils::GetBool(op_desc, ATTR_NAME_OP_NO_TILING, is_no_tiling);
  bool is_shape_unknown = false;
  if (NodeUtils::GetNodeUnknownShapeStatus(node, is_shape_unknown) == GRAPH_SUCCESS) {
    if (is_shape_unknown && !is_no_tiling) {
      GELOGI("op:%s is unknown shape, does not need to calc output size.", node.GetName().c_str());
      return SUCCESS;
    }
  }

  const std::string node_name = node.GetName();
  const std::string node_type = node.GetType();
  size_t output_size = op_desc->GetOutputsSize();
  GELOGD("Calc op[%s:%s] running param, output size=%zu.", node_name.c_str(), node_type.c_str(), output_size);

  for (size_t i = 0; i < output_size; ++i) {
    GeTensorDesc output_tensor = op_desc->GetOutputDesc(static_cast<uint32_t>(i));
    Format format = output_tensor.GetFormat();
    DataType data_type = output_tensor.GetDataType();

    int64_t mem_size = 0;
    graphStatus graph_status = TensorUtils::GetSize(output_tensor, mem_size);
    // If mem size has been set, no need reset.
    if ((graph_status == GRAPH_SUCCESS) && (mem_size > 0) && (data_type != DT_STRING)) {
      GELOGD("Op[%s:%s] out[%zu] mem size has been set, no need calc again, format=%s, data_type=%s, mem_size=%ld.",
             node_name.c_str(), node_type.c_str(), i, TypeUtils::FormatToSerialString(format).c_str(),
             TypeUtils::DataTypeToSerialString(data_type).c_str(), mem_size);
      continue;
    }

    int64_t output_mem_size = 0;
    graph_status = CalcMemSizeByNodeType(op_desc, output_tensor, output_mem_size, node_type);
    if (graph_status != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "calc op[%s:%s] out[%zu] mem size failed, format=%s, data_type=%s, error=%u.",
                        node_name.c_str(), node_type.c_str(), i, TypeUtils::FormatToSerialString(format).c_str(),
                        TypeUtils::DataTypeToSerialString(data_type).c_str(), graph_status);
      GELOGE(FAILED, "[Calc][MemSize] for op[%s:%s] out[%zu] failed, format=%s, data_type=%s, error=%u.",
             node_name.c_str(), node_type.c_str(), i, TypeUtils::FormatToSerialString(format).c_str(),
             TypeUtils::DataTypeToSerialString(data_type).c_str(), graph_status);
      return FAILED;
    }

    if (output_mem_size < 0) {
      REPORT_INNER_ERR_MSG("E19999", "Calc op[%s:%s] out[%zu] mem size is negative(not support),"
                         " format=%s, data_type=%s, mem_size=%ld.",
                         node_name.c_str(), node_type.c_str(), i, TypeUtils::FormatToSerialString(format).c_str(),
                         TypeUtils::DataTypeToSerialString(data_type).c_str(), output_mem_size);
      GELOGE(FAILED, "[Calc][MemSize] op[%s:%s] out[%zu] mem size is negative(not support),"
             " format=%s, data_type=%s, mem_size=%ld.",
             node_name.c_str(), node_type.c_str(), i, TypeUtils::FormatToSerialString(format).c_str(),
             TypeUtils::DataTypeToSerialString(data_type).c_str(), output_mem_size);
      return FAILED;
    }
    GELOGD("Calc op[%s:%s] out[%zu] mem size is %ld, format=%s, data_type=%s.", node_name.c_str(), node_type.c_str(), i,
           output_mem_size, TypeUtils::FormatToSerialString(format).c_str(),
           TypeUtils::DataTypeToSerialString(data_type).c_str());

    TensorUtils::SetSize(output_tensor, output_mem_size);

    // check string input add _op_max_size attr
    if (SetSizeForStringInput(node, output_tensor, i) != GRAPH_SUCCESS) {
      return FAILED;
    };

    graph_status = op_desc->UpdateOutputDesc(static_cast<uint32_t>(i), output_tensor);
    if (graph_status != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Update op[%s:%s] out[%zu] desc failed, format=%s, data_type=%s, error=%u.", node_name.c_str(),
             node_type.c_str(), i, TypeUtils::FormatToSerialString(format).c_str(),
             TypeUtils::DataTypeToSerialString(data_type).c_str(), graph_status);
      return FAILED;
    }
  }

  if (calc_op_param_call.find(node_type) != calc_op_param_call.end()) {
    GE_ASSERT_SUCCESS(calc_op_param_call[node_type](node),
                      "[Call]calc_op_param_call faild, node name: %s, node type: %s.",
                      node_name.c_str(), node_type.c_str());
  }

  GELOGD("Calc op[%s:%s] running param success.", node_name.c_str(), node_type.c_str());
  return SUCCESS;
}

Status GeLocalOpsKernelBuilder::GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) {
  if (kDependComputeOps.count(node.GetType()) != 0UL) {
    (void)AttrUtils::SetInt(node.GetOpDesc(), ATTR_NAME_UNKNOWN_SHAPE_TYPE, DEPEND_COMPUTE);
    GELOGI("Op:%s set unknown shape type DEPEND_COMPUTE.", node.GetName().c_str());
    return SUCCESS;
  }
  bool is_shape_unknown = false;
  if (NodeUtils::GetNodeUnknownShapeStatus(node, is_shape_unknown) == GRAPH_SUCCESS) {
    if (is_shape_unknown) {
      (void) ge::AttrUtils::SetBool(node.GetOpDesc(), ge::ATTR_NAME_NOTASK, true);
      GELOGI("op:%s is unknown shape, does not need to generate task",
             node.GetName().c_str());
      return SUCCESS;
    }
  }

  std::string name = node.GetName();
  std::string type = node.GetType();
  GELOGD("Ge local generate task for node:%s(%s) begin, tasks.size()=%zu.", name.c_str(), type.c_str(), tasks.size());

  auto op = OpFactory::Instance().CreateOp(node, context);
  if (op == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "create op for node:%s(%s) failed.", name.c_str(), type.c_str());
    GELOGE(FAILED, "[Create][Op] for node:%s(%s) failed.", name.c_str(), type.c_str());
    return FAILED;
  }

  Status ret = op->Run();
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) op run failed.", name.c_str(), type.c_str());
    GELOGE(ret, "[Call][Run] for Node:%s(%s) op failed.", name.c_str(), type.c_str());
    return ret;
  }
  GELOGD("Ge local generate task for node:%s(%s) end, tasks.size()=%zu.", name.c_str(), type.c_str(), tasks.size());
  return ret;
}
}  // namespace ge_local
}  // namespace ge

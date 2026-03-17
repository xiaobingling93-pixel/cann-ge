/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_UTILS_AUTOFUSE_UTILS_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_UTILS_AUTOFUSE_UTILS_H_
#include <sstream>
#include "ge_common/ge_api_types.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/utils/op_type_utils.h"
#include "graph/node.h"
#include "graph/operator_reg.h"
#include "autofuse_frame/autofuse_frames.h"
#include "common/checker.h"
#include "common/platform_context.h"
#include "graph/compute_graph.h"
#include "ascir_ops.h"

namespace ge {
const std::string kLoweringDir = "lowering";
const std::string kCanFuseDir = "canfuse";
const std::string kPostProcessDir = "postprocess";
const std::string kCanFuseOrigin = "CanFuseOrigin";
const std::string kGeFallBack = "ge_fallback";
const std::string kCompleteAscIoIndex = "complete_asc_io_index";

const std::string AF_SPLIT = "Split";
const std::string AF_SPLITD = "SplitD";
const std::string AF_SPLITV = "SplitV";
const std::string AF_SPLITVD = "SplitVD";
const std::set<string> SPLIT_TYPES{AF_SPLIT, AF_SPLITD, AF_SPLITVD, AF_SPLITV};
const std::string kMatMul = "MatMul";
const std::string kMatMulBias = "MatMulBias";
const std::string kMatMulOffset = "MatMulOffset";
const std::string kMatMulOffsetBias = "MatMulOffsetBias";
const std::string kBatchMatMul = "BatchMatMul";
const std::string kBatchMatMulBias = "BatchMatMulBias";
const std::string kBatchMatMulOffset = "BatchMatMulOffset";
const std::string kBatchMatMulOffsetBias = "BatchMatMulOffsetBias";

class DefaultCounter : public Counter {
public:
  DefaultCounter() = default;
  ~DefaultCounter() override = default;
  int64_t NextId() override { return id_++;};
private:
  int64_t id_ = 0;
};

class AutofuseUtils {
 public:
  static int64_t GenUniqueNumber();
  static void ClearUniqueNumber();

  template <typename Container>
  static std::string VectorPairToStr(const Container &vec) {
    std::ostringstream oss;
    oss << "[";
    auto i = 0U;
    for (auto &pair : vec) {
      oss << "(" << pair.first << ", " << pair.second << ")";
      if (i < vec.size() - 1U) {
        oss << ", ";
      }
      i++;
    }
    oss << "]";
    return oss.str();
  }

  template <typename T>
  static std::string VectorToStr(const std::vector<T> &vec) {
    std::string result = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      if constexpr (std::is_same<T, ge::Expression>::value) {
        result += (vec[i].Str().get());
      } else if constexpr (std::is_same<T, ge::NodePtr>::value || std::is_same<T, ge::OpDescPtr>::value) {
        result += (vec[i]->GetNamePtr());
        result += " ";
        result += (vec[i]->GetTypePtr());
      } else {
        result += std::to_string(vec[i]);
      }
      if (i < vec.size() - 1) {
        result += ", ";
      }
    }
    result += "]";
    return result;
  }

  template<typename T>
  static std::string VectorToStr(const std::vector<T> *vec) {
    if (vec == nullptr) {
      return "nullptr";
    }
    return VectorToStr(*vec);
  }

  template <typename T>
  static std::string SetToStr(const std::set<T> &s) {
    std::ostringstream oss;
    oss << "[";
    for (auto it = s.begin(); it != s.end(); ++it) {
      oss << *it;
      if (std::next(it) != s.end()) {
        oss << ", ";
      }
    }
    oss << "]";
    return oss.str();
  }

  // 获取 npu_arch 并调用 InferDataType
  template <typename OpType>
  static Status CallAscirInferDataType(const std::vector<DataType> &input_dtypes,
                                       std::vector<DataType> &expect_output_dtypes) {
    std::string npu_arch;
    GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetCurrentPlatformString(npu_arch));
    return OpType::InferDataType(input_dtypes, expect_output_dtypes, npu_arch);
  }

  // 获取 npu_arch 并调用 CommonInferDtype
  static Status CallAscirCommonInferDtype(const std::string &op_type,
                                          const std::vector<DataType> &input_dtypes,
                                          std::vector<DataType> &expect_output_dtypes) {
    std::string npu_arch;
    GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetCurrentPlatformString(npu_arch));
    return ge::ascir::CommonInferDtype(op_type, input_dtypes, expect_output_dtypes, npu_arch);
  }

  static bool IsAutoFuseNode(const ge::OpDescPtr &op_desc) {
    // op_desc外部保证非空
    return OpTypeUtils::IsAutofuseNode(op_desc);
  }
  static Status CreateComputeGraphWithGraphID(const ge::NodePtr &node, const std::string &graph_name,
                                              ComputeGraphPtr &compute_graph);

  static Status CreateComputeGraphWithGraphID(const ComputeGraphPtr &graph, const std::string &graph_name,
                                              ComputeGraphPtr &compute_graph);

  static Status SerilizeAscBackend(Node *node_ptr, std::string &output, bool isHash = false);

  static Status CopyGraphAndRenameNode(const ComputeGraphPtr &graph, ComputeGraphPtr &copy_graph,
                                       const CounterPtr &counter);

  static Status AddOperatorPrototypeAttrs(const OpDescPtr &op_desc);

  static void DumpGraphToOnnx(const ge::ComputeGraph &compute_graph, const std::string &module_name,
                              const std::string &suffix);

  static void DumpGEGraph(const ge::ComputeGraphPtr &graph, const std::string &module_name, const std::string &suffix);

  static void DumpGEGraphLevel1(const ge::ComputeGraphPtr &graph, const std::string &module_name,
                                const std::string &suffix);

  static void DumpGraphToOnnxLevel1(const ge::ComputeGraph &compute_graph, const std::string &module_name,
                                    const std::string &suffix);

  static bool IsUbScalar(const std::vector<ge::Expression> &repeats);

  static bool IsSplitType(const std::string &node_type);

  static Status DelOneNodeInGraph(const ComputeGraphPtr &graph, const NodePtr &node);

  static bool CheckAndMulDetect(const std::vector<Expression> &long_dims, const std::vector<Expression> &short_dims,
                                size_t &sort_idx, std::vector<size_t> &mul_idx);
  static bool IsCubeNodeType(const NodePtr &node);
  static graphStatus GetListIntFromInput(const NodePtr &node, std::vector<int64_t> &value_vec,
                                         const std::string &input = "");

  static graphStatus GetListIntFromAttr(const NodePtr &node, std::vector<int64_t> &value_vec,
                                        const std::string &attr_name = "");

  static graphStatus GetListIntByInputOrAttr(const NodePtr &node, std::vector<int64_t> &value_vec,
                                             const std::string &input = "", const std::string &attr = "");

  static std::vector<const ge::Node *> GetComputeOps(const std::vector<const ge::Node *> &nodes);

  /**
  * @brief 处理node_name的核心接口
  * 支持两种格式的node_name处理：
  * 1. autofuse_fused_数字_xxx 格式（包含concat分割）
  * 2. autofuse_数字_xxx 格式（直接统计）
  * 支持多次调用的累加统计
  * @param node_name 输入的节点名称，格式：autofuse_xx_数字_type1_type2_type3...
  * @return 处理后的节点名称（去重统计后）
  */
  static std::string SimplifyNodeName(const std::string &node_name);

 private:
  static NodePtr ConvertAscBackendNodeToAscGraphNode(const ComputeGraphPtr compute_graph, const NodePtr &node);

  static thread_local int64_t number;

  static Status SerializeAndPackComputeGraph(const ComputeGraphPtr &compute_graph, const NodePtr &node,
                                             std::string &output, bool isHash = false);

  static Status GetNodeOutputIndex(const NodePtr &node, std::vector<uint32_t> &node_output_index);

  static Status RenameInputAndOutputForGraph(ComputeGraphPtr &graph, const NodePtr &node);
};
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_UTILS_AUTOFUSE_UTILS_H_

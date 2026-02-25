/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __COMMON_UTILS_H__
#define __COMMON_UTILS_H__

#include <string>
#include "graph/symbolizer/symbolic.h"
#include "external/ge_common/ge_api_types.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/ascendc_ir/ascir_registry.h"
#include "ascir.h"
#include "schedule_result.h"
#include "symbolizer/symbolic_utils.h"

namespace ascgen_utils {
  const string INVALID_TILING = "invalid_tiling";
  constexpr int32_t NOT_SUPPORT_BRC_INLINE = -1;
  const std::string kKernelTaskTypeAIVOnly = "KERNEL_TYPE_AIV_ONLY";
  const std::string kKernelTaskTypeMixAIVOneZero = "KERNEL_TYPE_MIX_AIV_1_0";

  const std::string kMatMul = "MatMul";
  const std::string kMatMulBias = "MatMulBias";
  const std::string kMatMulOffset = "MatMulOffset";
  const std::string kMatMulOffsetBias = "MatMulOffsetBias";
  const std::string kBatchMatMul = "BatchMatMul";
  const std::string kBatchMatMulBias = "BatchMatMulBias";
  const std::string kBatchMatMulOffset = "BatchMatMulOffset";
  const std::string kBatchMatMulOffsetBias = "BatchMatMulOffsetBias";

  struct MatMulAttr {
    int64_t transpose_x1{0};
    int64_t transpose_x2{0};
    int64_t offset_x{0};
    int64_t enable_hf32{0};
    int64_t adj_x1{0};
    int64_t adj_x2{0};
    int64_t has_relu{0};
    bool is_batch{false};
    bool is_bias{false};
    bool is_offset_w{false};
    std::string output_dtype;
    std::string input_dtype;
    std::string m;
    std::string n;
    std::string k;
  };

#define GET_MATMUL_ATTRS(Node, AttrType, AttrData) \
    auto mm_attr = node->attr.ir_attr->DownCastTo<ge::ascir_op::AttrType::Asc##AttrType##IrAttrDef>(); \
    GE_ASSERT_NOTNULL(mm_attr); \
    GE_ASSERT_SUCCESS(mm_attr->GetHas_relu(AttrData.has_relu)); \
    GE_ASSERT_SUCCESS(mm_attr->GetEnable_hf32(AttrData.enable_hf32)); \
    GE_ASSERT_SUCCESS(mm_attr->GetOffset_x(AttrData.offset_x)); \
    GE_ASSERT_SUCCESS(mm_attr->GetTranspose_x1(AttrData.transpose_x1)); \
    GE_ASSERT_SUCCESS(mm_attr->GetTranspose_x2(AttrData.transpose_x2));

#define GET_BATCH_MATMUL_ATTRS(Node, AttrType, AttrData) \
    auto mm_attr = node->attr.ir_attr->DownCastTo<ge::ascir_op::AttrType::Asc##AttrType##IrAttrDef>(); \
    GE_ASSERT_NOTNULL(mm_attr); \
    GE_ASSERT_SUCCESS(mm_attr->GetHas_relu(AttrData.has_relu)); \
    GE_ASSERT_SUCCESS(mm_attr->GetEnable_hf32(AttrData.enable_hf32)); \
    GE_ASSERT_SUCCESS(mm_attr->GetOffset_x(AttrData.offset_x)); \
    GE_ASSERT_SUCCESS(mm_attr->GetAdj_x1(AttrData.adj_x1)); \
    GE_ASSERT_SUCCESS(mm_attr->GetAdj_x2(AttrData.adj_x2));

  struct MergeBrcAxisParams {
    const std::vector<ge::Expression> &repeats;
    const std::vector<ge::Expression> &strides;
    std::vector<ge::Expression> repeats_no_one;
    std::vector<ge::Expression> strides_no_one;
    std::vector<bool> is_axis_brc;
    std::vector<ge::Expression> merge_repeats;

    MergeBrcAxisParams(const std::vector<ge::Expression> &repeats, const std::vector<ge::Expression> &strides)
      : repeats(repeats), strides(strides) {
      repeats_no_one.reserve(repeats.size());
      strides_no_one.reserve(strides.size());
      is_axis_brc.reserve(repeats.size());
      merge_repeats.reserve(repeats.size());
    }
  };

  std::string CamelToLowerSneak(const std::string &str);
  std::string SubStringReplace(std::string &ori, const std::string &from, const std::string &to);
  std::string GenValidName(const std::string& t_name);
  bool GetRealPath(const std::string& file_path, std::string& real_file_path);
  bool IsEmptyTensorSence(const ascir::FusedScheduledResult& fused_schedule_result);

  template <typename T>
  static std::string VectorToStr(const std::vector<T> &vec) {
    std::string result = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      if constexpr (std::is_same<T, ge::Expression>::value) {
        result += (vec[i].Str().get());
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
  ge::Expression GetTensorSize(const ge::AscTensor &tensor);
  ge::Expression CalculateOneWorkspaceSize(const ge::AscNodePtr &workspace_nodes);
  ge::Expression CalculateWorkspaceSize(const std::vector<ge::AscNodePtr> &workspace_nodes);
  ge::Expression CalcExtraTmpBufForAscGraph(const ascir::ImplGraph &graph);
  std::vector<ascir::TensorId> GetWorkspaceTensorIdListInOneScheduleResult(const ascir::FusedScheduledResult& fused_schedule_result);

  ge::Status GetApiTilingTypeName(const ascir::NodeView& node, std::string& type_name);
  ge::Status GetApiTilingFieldName(const ascir::NodeView& node, std::string& field_name);
  void GetApiExtractDupSet(const ascir::ImplGraph &graph,
                         std::set<std::pair<std::string, std::string>> &pre_api_extract_dup,
                         uint32_t& total_blk_num);
  int32_t CalcReservedTmpBufSizeForAscGraph(const ascir::ImplGraph &graph);
  void GetApiReservedBlockNum(const ascir::ImplGraph &graph, uint32_t& total_blk_num);
  bool IsScalarNextNodeSupportBlkTensor(const ge::AscNodePtr &node);
  bool IsUbScalarLoad(const ge::AscNodePtr &node);
  bool IsStaticSchedResult(const ascir::FusedScheduledResult& fused_schedule_result);
  ge::Status ScalarValuePreProcess(const std::string& ori_value,
                                 const std::string& dtype,
                                 std::string& after_pre_pro_value);
  void MergeBrcAxisRepeats(const std::vector<ge::Expression> &input0_repeats,  // 输入0的vector_repeats, 带广播
                           const std::vector<ge::Expression> &input1_repeats,  // 输入1的vector_repeats, 不带广播
                           const std::vector<ge::Expression> &input1_strides,  // 输入1的vector_strides
                           std::vector<ge::Expression> &i0_merge_repeats,     // 输入0，合并后的vector_repeats
                           std::vector<ge::Expression> &i1_merge_repeats);     // 输入1，合并后的vector_repeats
  void MergeBrcAxisRepeats(MergeBrcAxisParams &in0, MergeBrcAxisParams &in1);

  bool IsGeneralizeBrcInlineScene(const ge::AscNodePtr &node);
  bool IsGeneralizeBrcInlineScene(const ge::AscNodePtr &node, std::vector<uint8_t> &input_idx_2_brc_inline);
  bool IsGeneralizeBrcInlineScene(const ge::AscNodePtr &node, const ge::AscTensor &input0,
                                  const ge::AscTensor &input1, std::vector<uint8_t> &input_idx_2_brc_inline);
  bool IsSupportBlkTensorInput(const ge::AscNodePtr &next_node);
  int32_t GetBrcInlineIndex(const ge::AscNodePtr &node);
  bool IsLinkToBrdcst(const ascir::NodeView &node, const std::set<std::string> &brc_types);

  std::string FormatExpression(const std::string &expression);
  template<typename T>
  T Gcd(T a, T b) {
    while (b != 0) {
      T tmp = b;
      b = a % b;
      a = tmp;
    }
    return a;
  }

  std::unique_ptr<ge::ascir::AscIrAtt> GetAscIrAttImpl(const string &ascir_type);
  std::unique_ptr<ge::ascir::AscIrCodegen> GetAscIrCodegenImpl(const string &ascir_type);

  bool IsScalarInput(const std::vector<ge::Expression> &repeats);
  bool IsNodeSupportsVectorFunction(const ge::AscNodePtr &node);
  bool IsNodeSupportsScalarInput(const ge::AscNodePtr &node, const std::vector<bool> &is_scalar_list);
  bool IsNodeSupportsAllScalar(const ge::AscNodePtr &node);
  bool IsNodeSupportsScalarIfExchangeInputs(const ge::AscNodePtr &node, const std::vector<bool> &is_scalar_list);
  bool IsNodeSupportsInplace(const ge::AscNodePtr &node);

  bool IsNodeSupportsBrcInline(const ge::AscNodePtr &node);
  bool IsNodeContainsBrcInline(const ge::AscNodePtr &node);

  inline bool ExpressEq(const ge::Expression &e1, const ge::Expression &e2) {
    return ge::SymbolicUtils::StaticCheckEq(e1, e2) == ge::TriBool::kTrue;
  }

  ge::ExecuteCondition GetNodeExecCondition(const ge::NodePtr &node);
  bool IsNodeCacheable(const ge::NodePtr &node);
  bool IsSingleGroup(const ascir::FusedScheduledResult &fused_schedule_result);
  bool CanUseTilingKey(const ascir::FusedScheduledResult &fused_schedule_result);
  bool IsJustCubeFixpip(const ascir::FusedScheduledResult &fused_schedule_result);
  bool IsCubeFusedScheduled(const ascir::FusedScheduledResult &fused_schedule_result);
  bool IsCubeUBFusedScheduled(const ascir::FusedScheduledResult &fused_schedule_result);
  bool IsCubeCommonFusedScheduled(const ascir::FusedScheduledResult &fused_schedule_result);
  bool HasCubeUBFusedScheduled(const ascir::FusedScheduledResult &fused_schedule_result);
  bool HasCubeCommonFusedScheduled(const ascir::FusedScheduledResult &fused_schedule_result);
  bool IsCubeType(const ascir::ImplGraph &impl_graph);
  bool IsCubeTypeWithBatch(const ascir::ImplGraph &impl_graph);
  bool IsCubeTypeWithBias(const ascir::ImplGraph &impl_graph);
  bool IsCubeTypeWithOffsetW(const ascir::ImplGraph &impl_graph);
  ge::Status ParseMatmulAttr(const ascir::NodeView &node, MatMulAttr &mm_attr_data);
  bool IsCubeGroupType(const ascir::ScheduleGroup &sched_group);
  bool IsSatetyResultType(const ascir::ScheduledResult &sched_result);
  ge::Status GetMutmulOutputTypeSize(const ascir::NodeView &node, uint32_t &length);
  ge::Status GetMutmulInputNum(const ascir::NodeView &node, uint32_t &num);
  ge::Status CreateCVFusionResult(ascir::FusedScheduledResult &elemwise_schedule_result);
  ge::Status CreateCVFusionCommonResult(ascir::FusedScheduledResult &elemwise_schedule_result);
  bool IsCVFusionUBGraph(const ascir::ImplGraph &impl_graph, ascir::CubeTemplateType cv_fusion_type);
  ge::Status FilterCVFusionUBResult(ascir::FusedScheduledResult &ub_schedule_result);
  ge::Status FilterCVFusionCommonResult(ascir::FusedScheduledResult &common_schedule_result);
}  // namespace ascgen_utils

#endif

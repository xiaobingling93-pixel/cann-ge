/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_SHARE_GRAPH_H
#define INC_SHARE_GRAPH_H
#include "graph/compute_graph.h"
#include "ascendc_ir.h"

namespace ascir {
struct ShareGraph {
  static ge::ComputeGraphPtr LoadLog2StoreFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr ModFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadLShiftStoreFusedGraph(size_t dims_size, ge::DataType in_dtype, ge::DataType out_dtype);
  static ge::ComputeGraphPtr AddAbsFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr SubAbsFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr SubTransposeAbsFusedGraph(size_t dims_size, vector<size_t> perms);
  static ge::ComputeGraphPtr ScalarInfAddFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr ScalarDivInfFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr AddGeluFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr CompareFusedGraph(size_t dims_size, bool is_second_input_tensor, ge::DataType dtype, std::string mode);
  static ge::ComputeGraphPtr AddNegFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadToStoreAndAbsFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadUnalignPadFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadNeedLoopModeFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr BrcInlineFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadWhereStoreFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadWhereX2X3IsUbscalarStoreFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadWhereX2IsUbscalarStoreFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadWhereX3IsUbscalarStoreFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadLogicalNotStoreFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadLogicalNotStoreFusedGraph(size_t dims_size, ge::DataType dt_in, ge::DataType dt_out);
  static ge::ComputeGraphPtr AddRsqrtFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadBitwiseAndStoreFusedGraph(size_t dims_size, ge::DataType in_dtype, ge::DataType out_dtype);
  static ge::ComputeGraphPtr ContinuesBrcFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr ScalarBrcFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadBrcFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr CastCastFusedGraph(size_t dims_size, ge::DataType in_dtype, ge::DataType out_dtype);
  static ge::ComputeGraphPtr CastCastNanFusedGraph(size_t dims_size, ge::DataType in_dtype, ge::DataType out_dtype);
  static ge::ComputeGraphPtr CastCastIsFiniteFusedGraph(size_t dims_size, ge::DataType in_dtype, ge::DataType out_dtype);
  static ge::ComputeGraphPtr CastCastReciprocalFusedGraph(size_t dims_size, ge::DataType in_dtype, ge::DataType out_dtype);
  static ge::ComputeGraphPtr LoadLeakyReluStoreFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadSigmoidStoreFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadErfStoreFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr AddExp2FusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr AddFloorFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr AddFloorBf16FusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr AbsFmaFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr AbsFmaBf16FusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr AddExpBf16FusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr FloordivAbsFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadTanhStoreFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr AddAbsScalarFusedGraph(size_t dims_size, ge::DataType dtype);
  static ge::ComputeGraphPtr AbsBrcAddFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr UbScalarBrcAbsAddFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr BrcReduceFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr FloorDivMulLessEqualSelectFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr AxpyAbsFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr AxpyAbsHalfFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr AxpyAddFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr TailBrcTailReduceFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadPowAllInputIsScalarStoreFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr AddAbsFusedConstGraph(size_t dims_size, std::vector<int> dims);
  static ge::ComputeGraphPtr SubTransposeAbsFusedConstGraph(size_t dims_size, vector<size_t> perms, std::vector<int> dims);
  static ge::ComputeGraphPtr LoadLogicalOrStoreFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadLogicalAndStoreFusedGraph(size_t dims_size);
  static void ConcatAscGraph(ge::AscGraph &graph,
                             const std::vector<std::string> &dim_sizes,
                             bool align = false);
  static ge::ComputeGraphPtr AbsClipFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadGatherAbsStore(int64_t gather_axis, ge::DataType data_type);
  static ge::ComputeGraphPtr LoadGatherTailAbsStore(int64_t gather_axis, ge::DataType data_type);
  static ge::ComputeGraphPtr LoadGatherOneAxisAbsStore(int64_t gather_axis, ge::DataType data_type);
  static ge::ComputeGraphPtr MatMulFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr GatherReduceStore(int64_t gather_axis, ge::DataType data_type);
  static ge::ComputeGraphPtr LoadWhereReduceStoreFusedGraph(size_t dims_size, bool x2_scalar, bool x3_scalar);
  static ge::ComputeGraphPtr LoadCompareStoreFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadCompareCastSumStoreFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadMatmulElewiseBrcFusedGraph();
  static ge::ComputeGraphPtr LoadMatmulCompareScalarFusedGraph();
  static ge::ComputeGraphPtr DivAbsFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr BF16AddFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr BF16NddmaAddFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr AbsBf16FusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr AbsUint8FusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr ErfBf16FusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadBitwiseNotStoreFusedGraph(size_t dims_size, ge::DataType in_dtype,
                                                           ge::DataType out_dtype);
  static ge::ComputeGraphPtr LoadBitwiseOrStoreFusedGraph(size_t dims_size, ge::DataType in_dtype,
                                                           ge::DataType out_dtype);
  static ge::ComputeGraphPtr LoadBitwiseXorStoreFusedGraph(size_t dims_size, ge::DataType in_dtype,
                                                           ge::DataType out_dtype);
  static ge::ComputeGraphPtr CeilBf16FusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr CosBf16FusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr AtanhBf16FusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr CoshBf16FusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr DigammaBf16FusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr ErfcBf16FusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr BF16SinFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr BF16SqrtFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr BF16RsqrtFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr BF16SigmoidFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr LoadCompareScalarWhereFusedGraph();
  static ge::ComputeGraphPtr LoadCompareWhereFusedGraph();
  static ge::ComputeGraphPtr BinaryApiScalarFusedGraph();
  static ge::ComputeGraphPtr AcosFloatFusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr AcosBf16FusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr AcoshBf16FusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr AsinBf16FusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr AsinhBf16FusedGraph(size_t dims_size);
  static ge::ComputeGraphPtr AtanBf16FusedGraph(size_t dims_size);
};
}  // namespace ascir
#endif

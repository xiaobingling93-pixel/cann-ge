/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AUTOFUSE_CONCAT_API_CALL_H__
#define __AUTOFUSE_CONCAT_API_CALL_H__

#include "codegen_kernel.h"
#include "symbolizer/symbolic_utils.h"

namespace codegen {
class ConcatApiCall : public ApiCall {
 public:
  using ApiCall::Generate;
  explicit ConcatApiCall(const std::string &api_name) : ApiCall(api_name) {}
  ~ConcatApiCall() override = default;
  Status Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                  const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                  std::string &result) const override;
 protected:
  struct ConcatTiling {
    uint32_t gcd = 1U;
    uint32_t tmp_buf_size = 0U;
    uint32_t dst_col_size = 0U;
    uint32_t dst_row_num_unit = 0U;
    uint32_t max_repeat_times = 0U;
    uint32_t max_element_num = 0U;
    uint32_t max_orig_row_num = 0U;
    uint32_t per_repeat_size = 0U;
    uint32_t first_copy_repeat_times = 0U;  // for diff dim
    uint32_t last_trans_repeat_times = 0U;  // for diff dim
    bool any_padded = false;
    bool all_static = true;
    ge::TriBool all_inputs_shape_equal = ge::TriBool::kUnknown;
    bool can_use_gather = false;
    ge::Expression dst_col_size_expr;
    ge::Expression dst_row_stride;
    std::vector<ge::Expression> src_col_size_exprs;
    std::vector<ge::Expression> src_col_actual_size_exprs;
    std::vector<ge::Expression> src_non_zero_strides;
    std::vector<ge::Expression> src_row_strides;
    std::vector<ge::Expression> last_dim_size_exprs;
    std::vector<int64_t> src_col_sizes;
    std::vector<ge::Expression> dst_offsets;
    std::vector<uint32_t> src_loop_strides;
    std::vector<uint32_t> src_buffer_offsets;
    std::vector<bool> is_padded;
    std::vector<uint32_t> second_last_dim_strides;
    std::vector<uint32_t> gather_mask_dim_sizes;
    ge::Expression total_rows_expr;
    uint32_t data_type_size = 0;
  };

  Status ParseAttr(const ascir::NodeView &node) override;
  static Status ParseConcatDim(const Tensor &x0, const Tensor &y, size_t &concat_dim);

  static Status InitializeTiling(size_t concat_dim,
                                 const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                 const Tensor &y,
                                 ConcatTiling &tiling);
  static bool IsAllAligned(ConcatApiCall::ConcatTiling &tiling,
                           const std::vector<ge::Expression> &col_size_exprs);
  static Status GenerateForAllAligned(const vector<std::reference_wrapper<const Tensor>> &inputs,
                                      const Tensor &y,
                                      const ConcatApiCall::ConcatTiling &tiling,
                                      const Tiler &tiler,
                                      std::stringstream &ss);

 private:
  static void GenConcatParams(const vector<std::reference_wrapper<const Tensor>> &inputs,
                              const Tensor &y,
                              const Tiler &tiler,
                              const std::string &dtype_name,
                              std::stringstream &ss);
  static void GenConcatTilingForAllAligned(const ConcatTiling &tiling,
                                           const Tiler &tiler,
                                           std::stringstream &ss);
  Status CalcTiling(size_t concat_dim,
                    const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                    ConcatTiling &tiling) const;
  static Status CalcTilingForInputs(const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                    size_t block_size,
                                    ConcatTiling &concat_tiling);
  static void DefineConcatTiling(const ConcatTiling &tiling, std::stringstream &ss);
  static void DefineConcatShape(const ConcatTiling &tiling, const Tiler &tiler, std::stringstream &ss);
  static Status DefineConcatContext(const ConcatTiling &tiling,
                                    const std::string &dtype_name,
                                    const Tiler &tiler,
                                    std::stringstream &ss);
  static void DefineInputList(const ConcatTiling &tiling,
                              const std::string &dtype_name,
                              const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                              std::stringstream &ss);
  static void GenSrcTensors(const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                            const std::string &dtype_name,
                            std::stringstream &ss);

  bool use_concat_small_tail_api_ = false;
  ascir::NodeView node_ = nullptr;
};
}  // namespace codegen

#endif  // __AUTOFUSE_CONCAT_API_CALL_H__

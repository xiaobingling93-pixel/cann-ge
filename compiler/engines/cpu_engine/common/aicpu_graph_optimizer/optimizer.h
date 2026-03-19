/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_OPTIMIZER_H_
#define AICPU_OPTIMIZER_H_

#include "aicpu_graph_optimizer.h"
#include "factory/factory.h"
#include "graph/option/optimization_option_info.h"
#include "register/optimization_option_registry.h"

namespace aicpu {
class Optimizer {
 public:
  /**
   * constructor
   * @param void
   */
  Optimizer() : op_check_mode_(false) {}

  /**
   * destructor
   * @param
   */
  virtual ~Optimizer() = default;

  /**
   * init optimizer
   * @return status whether this operation success
   */
  virtual ge::Status Initialize() { return ge::SUCCESS; }

  virtual ge::Status Finalize() { return ge::SUCCESS; }

  /**
   * Optimize original graph, using in graph preparation stage
   * @param graph Computation graph
   * @return status whether this operation success
   */
  virtual ge::Status OptimizeOriginalGraph(
      [[maybe_unused]] ge::ComputeGraph &graph,
      [[maybe_unused]] const std::map<std::string, OpFullInfo> &all_op_info) const {
    return ge::SUCCESS;
  }

  /**
   * Optimize original graph, using in graph preparation stage
   * @param graph Computation graph
   * @return status whether this operation success
   */
  virtual ge::Status OptimizeOriginalGraphJudgeInsert(
      const ge::ComputeGraph &graph,
      const std::map<std::string, OpFullInfo> &all_op_info) const;

  /**
   * Optimize fused graph, using to fuse op
   * @param graph Computation graph
   * @param all_op_info store ops information
   * @return status whether this operation success
   */
  virtual ge::Status OptimizeFusedGraph(
      [[maybe_unused]] ge::ComputeGraph &graph,
      [[maybe_unused]] const std::map<std::string, OpFullInfo> &all_op_info) const {
    return ge::SUCCESS;
  }

  virtual void SetCustUserInfos([[maybe_unused]] map<std::string, std::string> info) {
    return;
  }

 protected:
  /**
   * Get framework op original type
   * @param op_desc_ptr OpDesc ptr
   * @param op_type Op type
   * @return status whether this operation success
   */
  ge::Status GetFrameworkOpType(const ge::OpDescPtr &op_desc_ptr,
                                std::string &op_type) const;

  /**
   * Check whether need check op type in device
   */
  void InitOpCheckMode();

  void GetTfOpFussionOoLevel();

 private:
  /**
   * update the format and shape of each TensorDesc for it
   * @param OpInfo op info
   * @param OpDescPtr OpDescPtr
   * @return status whether this operation success
   */
  ge::Status UpdateInputFormatAndShape(const OpFullInfo &op_info,
                                       const ge::OpDescPtr &op_desc_ptr) const;

  /**
   * update the format and shape of each TensorDesc for it
   * @param OpInfo op info
   * @param OpDescPtr OpDescPtr
   * @return status whether this operation success
   */
  ge::Status UpdateOutputFormatAndShape(const OpFullInfo &op_info,
                                        const ge::OpDescPtr &op_desc_ptr) const;

  /**
   * Get format corresponding to format_name from formats map
   * @param formats All format info
   * @param format_name Format name
   * @param format Format
   */
  void GetFormat(const std::map<std::string, std::string> &formats,
                 const std::string &format_name, ge::Format &format) const;

  /**
   * Update format for Tensordesc
   * @param tensor_desc Tensor Desc
   * @param src_format Src format
   * @param dst_format Dst format
   */
  void UpdateTensorDesc(ge::GeTensorDesc &tensor_desc,
                        const ge::Format &src_format,
                        const ge::Format &dst_format) const;
                        
  ge::Status InitSlicePattern(const string &slice, const ge::NodePtr &node) const;
  
  // Copy prohibit
  Optimizer(const Optimizer &) = delete;
  // Copy prohibit
  Optimizer &operator=(const Optimizer &) = delete;
  // Move prohibit
  Optimizer(Optimizer &&) = delete;
  // Move prohibit
  Optimizer &operator=(Optimizer &&) = delete;

 protected:
  bool op_check_mode_;
  bool is_tf_op_fussion_oo_enable_ = true;
};

#define FACTORY_GRAPH_OPTIMIZER Factory<Optimizer>

#define FACTORY_GRAPH_OPTIMIZER_CLASS_KEY(CLASS, KEY) \
  FACTORY_GRAPH_OPTIMIZER::Register<CLASS> __##CLASS(KEY);
}  // namespace aicpu
#endif  // AICPU_OPTIMIZER_H_

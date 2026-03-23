/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_CAFFE_CAFFE_RESHAPE_PARSER_H_
#define PARSER_CAFFE_CAFFE_RESHAPE_PARSER_H_

#include "parser/caffe/caffe_op_parser.h"

namespace ge {
class PARSER_FUNC_VISIBILITY CaffeReshapeParser : public CaffeOpParser {
 public:
  /**
   * @ingroup domi_omg
   * @brief parse params of the operation
   * @param [in] op_src params to be parsed
   * @param [out] op_dest params after parsing
   * @return SUCCESS parse successfully
   * @return FAILED parse failed
   */
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op) override;

  /**
   * @ingroup domi_omg
   * @brief parse weight of the operation
   * @param [in] op_src params to be parsed
   * @param [out] op_dest params after parsing
   * @return SUCCESS parse successfully
   * @return FAILED parse failed
   * @author
   */
  Status ParseWeights(const Message *op_src, const ge::OpDescPtr &op) const;
  using CaffeOpParser::ParseWeights;

  /**
   * @ingroup domi_omg
   * @brief add const input node
   * @param [in] node to add const input
   * @param [out] node after add const input
   * @return SUCCESS add const input successfully
   * @return FAILED add const input failed
   * @author
   */
  Status AddConstInput(ge::NodePtr &node) override;
};
}  // namespace ge

#endif  // PARSER_CAFFE_CAFFE_RESHAPE_PARSER_H_

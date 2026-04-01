/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_TENSORFLOW_TENSORFLOW_CUSTOM_OP_PARSER_H_
#define PARSER_TENSORFLOW_TENSORFLOW_CUSTOM_OP_PARSER_H_

#include <string>
#include <unordered_map>
#include "common/pre_checker.h"
#include "proto/tensorflow/graph.pb.h"
#include "proto/tensorflow/node_def.pb.h"

namespace ge {

class TensorFlowCustomOpParser {
public:
  TensorFlowCustomOpParser() {}
  ~TensorFlowCustomOpParser() {}

  static Status ConstructRegOpString(const domi::tensorflow::OpDef &opdef, std::string &reg_op);
  static Status ConstructRegCustomOpString(const domi::tensorflow::OpDef &opdef, const domi::tensorflow::NodeDef &node_def,
                                    std::string &reg_op_custom_string);
  static Status CompileCustomOpFiles(const std::string &custom_op_cc_path, const std::string &output_so_path);
  static Status RegisteredTfaOps();
  static Status LoadCustomOpsLibrary(const std::string &so_path);
  static Status BuildCustomOpStrings(
      const std::unordered_map<std::string, const domi::tensorflow::NodeDef *> &custom_nodes_map, std::string &all_reg_op_strings);
  static Status WriteTextFile(const std::string &file_path, const std::string &content);
  static Status WriteWrapperCc(const std::string &cc_path);
  static Status DeleteTmpDirectoryContents(const std::string &out_dir);
  static Status ParseCustomOp(const std::unordered_map<std::string, const domi::tensorflow::NodeDef *> &custom_nodes_map);
};
}  // namespace ge
#endif  // PARSER_TENSORFLOW_TENSORFLOW_CUSTOM_OP_PARSER_H_

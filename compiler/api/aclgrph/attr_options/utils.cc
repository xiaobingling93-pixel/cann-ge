/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "api/aclgrph/attr_options/attr_options.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/debug/ge_log.h"
#include "common/omg_util/omg_util.h"
namespace ge {
  namespace {
  const std::string CFG_PRE_OPTYPE = "OpType::";
}
bool IsOriginalOpFind(const OpDescPtr &op_desc, const std::string &op_name) {
  std::vector<std::string> original_op_names;
  if (!AttrUtils::GetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_op_names)) {
    return false;
  }

  for (auto &origin_name : original_op_names) {
    if (origin_name == op_name) {
      return true;
    }
  }

  return false;
}

bool IsOpTypeEqual(const ge::NodePtr &node, const std::string &op_type) {
  if (op_type != node->GetOpDesc()->GetType()) {
    return false;
  }
  std::string origin_type;
  auto ret = GetOriginalType(node, origin_type);
  if (ret != SUCCESS) {
    GELOGW("[Get][OriginalType] from op:%s failed.", node->GetName().c_str());
    return false;
  }
  if (op_type != origin_type) {
    return false;
  }
  return true;
}

bool IsContainOpType(const std::string &cfg_line, std::string &op_type) {
  op_type = cfg_line;
  size_t pos = op_type.find(CFG_PRE_OPTYPE);
  if (pos != std::string::npos) {
    if (pos == 0) {
      op_type = cfg_line.substr(CFG_PRE_OPTYPE.length());
      return true;
    } else {
      GELOGW("[Check][Param] %s must be at zero pos of %s", CFG_PRE_OPTYPE.c_str(), cfg_line.c_str());
    }
    return false;
  }
  GELOGW("[Check][Param] %s not contain optype", cfg_line.c_str());
  return false;
}
}  // namespace ge

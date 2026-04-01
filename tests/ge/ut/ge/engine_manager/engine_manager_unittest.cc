/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "macro_utils/dt_public_scope.h"
#include "api/gelib/gelib.h"
#include "framework/omg/ge_init.h"
#include "engines/manager/engine_manager/dnnengine_manager.h"
#include "graph/op_desc.h"
#include "graph/debug/ge_attr_define.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
class UtestEngineManager : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestEngineManager, get_engine_name) {
  DNNEngineManager engine_manager;
  std::vector<OpInfo> op_infos;
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", "test");
  OpsKernelManager ops_kernel_manager;
  bool is_op_specified_engine = false;

  AttrUtils::SetStr(op_desc, ATTR_NAME_OP_SPECIFIED_ENGINE_NAME, "VectorEngine");
  AttrUtils::SetStr(op_desc, ATTR_NAME_OP_SPECIFIED_KERNEL_LIB_NAME, "VectorEngine");

  engine_manager.GetOpInfos(op_infos, op_desc, is_op_specified_engine);
  EXPECT_EQ(is_op_specified_engine, true);
  EXPECT_EQ(op_infos.front().engine, "VectorEngine");
  EXPECT_EQ(op_infos.front().opKernelLib, "VectorEngine");
}
}  // namespace ge

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
#include "compute_graph.h"
#include "framework/common/types.h"
#include "utils/graph_utils.h"
#include "common/debug/ge_log.h"

#include "dflow/base/model/endpoint.h"

namespace ge {
class EndpointTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(EndpointTest, SetGetAttr_QueueNode_Success) {
  Endpoint queue_def("queue_def_name", EndpointType::kQueue);
  auto queue_node_utils = QueueNodeUtils(queue_def).SetEnqueuePolicy("FIFO").SetNodeAction(kQueueActionControl);

  EXPECT_EQ(queue_node_utils.GetDepth(), 128L);
  EXPECT_EQ(queue_node_utils.GetEnqueuePolicy(), "FIFO");
  EXPECT_EQ(queue_node_utils.GetIsControl(), true);

  queue_node_utils.SetDepth(3L);
  EXPECT_EQ(queue_node_utils.GetDepth(), 3L);
}
}  // namespace ge

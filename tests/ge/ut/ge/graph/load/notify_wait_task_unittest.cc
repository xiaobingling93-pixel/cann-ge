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
#include "graph/load/model_manager/task_info/rts/notify_wait_task_info.h"
#include "graph/load/model_manager/davinci_model.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "hcom/hcom_topo_info.h"
#include "register/hidden_inputs_func_registry.h"

using namespace std;

namespace ge {
class UtestNotifyWaitTask : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

// test Init_NotifyWaitTaskInfo
TEST_F(UtestNotifyWaitTask, init_and_distribute_notify_wait_task_info) {
  domi::TaskDef task_def;
  task_def.set_stream_id(0);
  task_def.set_notify_id(0);

  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };

  rtNotify_t notify = nullptr;
  aclrtCreateNotify(&notify, 0U);
  model.notify_list_ = {notify};

  model.op_list_[0] = CreateOpDesc("op_name", "op_type");

  NotifyWaitTaskInfo task_info;
  EXPECT_EQ(task_info.Init(task_def, &model), SUCCESS);
  EXPECT_EQ(task_info.davinci_model_->GetOpByIndex(task_info.op_index_)->GetName(), "op_name");
  domi::GetContext().is_online_model = true;
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  EXPECT_TRUE(task_info.IsSupportReDistribute());
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  domi::GetContext().is_online_model = false;
}

TEST_F(UtestNotifyWaitTask, init_and_distribute_engine_notify_wait_task_info) {
  domi::TaskDef task_def;
  task_def.set_stream_id(0);
  task_def.set_notify_id(UINT32_MAX);

  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = {stream};
  const auto &op_desc = CreateOpDesc("op_name", "op_type");
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, "group", "hccl_world_group"));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, "tp_group", "test_group"));
  op_desc->SetAttachedStreamIds({0});
  model.op_list_[op_desc->GetId()] = op_desc;
  task_def.set_id(op_desc->GetId());

  REG_HIDDEN_INPUTS_FUNC(
      ge::HiddenInputsType::HCOM, [](const ge::OpDescPtr &op_desc, std::vector<void *> &addr) -> ge::graphStatus {
    HcomTopoInfo::TopoInfo topo_info;
    topo_info.rank_size = 8;
    topo_info.notify_handle = reinterpret_cast<void *>(0x800);
    EXPECT_EQ(HcomTopoInfo::Instance().SetGroupTopoInfo("hccl_world_group", topo_info), GRAPH_SUCCESS);
    EXPECT_EQ(HcomTopoInfo::Instance().SetGroupTopoInfo("test_group", topo_info), GRAPH_SUCCESS);
    return ge::GRAPH_SUCCESS;
  });

  NotifyWaitTaskInfo task_info;
  EXPECT_EQ(task_info.Init(task_def, &model), SUCCESS);
  EXPECT_EQ(task_info.davinci_model_->GetOpByIndex(task_info.op_index_)->GetName(), "op_name");
  EXPECT_EQ(task_info.Distribute(), SUCCESS);

  task_def.set_private_def("tp_group");
  EXPECT_EQ(task_info.Init(task_def, &model), SUCCESS);
  EXPECT_EQ(task_info.davinci_model_->GetOpByIndex(task_info.op_index_)->GetName(), "op_name");
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
}
}  // namespace ge

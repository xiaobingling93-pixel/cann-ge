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
#include "framework/common/helper/om_file_helper.h"
#include "common/helper/file_saver.h"
#include "common/math/math_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/util.h"
#include "graph/def_types.h"
#include "macro_utils/dt_public_unscope.h"

#include "proto/task.pb.h"

using namespace std;

namespace ge {
class UtestOmFileHelper : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UtestOmFileHelper, Normal)
{
  EXPECT_NO_THROW(OmFileLoadHelper loader);
  EXPECT_NO_THROW(OmFileSaveHelper saver);
}

TEST_F(UtestOmFileHelper, LoadInit)
{
  OmFileLoadHelper loader;
  ModelData md;
  EXPECT_EQ(loader.Init(md), PARAM_INVALID);
  ModelFileHeader header;
  md.model_data = &header;
  md.model_len = sizeof(ModelFileHeader) + sizeof(ModelPartitionTable) + 10;
  header.length = md.model_len - sizeof(ModelFileHeader);
  header.magic = MODEL_FILE_MAGIC_NUM;
  EXPECT_EQ(loader.Init(md), ACL_ERROR_GE_PARAM_INVALID);
}

TEST_F(UtestOmFileHelper, GetModelPartition)
{
  OmFileLoadHelper loader;
  loader.is_inited_ = true;
  ModelPartitionType type = ModelPartitionType::MODEL_DEF;
  ModelPartition partition;
  size_t model_index = 10;
  EXPECT_EQ(loader.GetModelPartition(type, partition, model_index), PARAM_INVALID);
  loader.is_inited_ = false;
  EXPECT_EQ(loader.GetModelPartition(type, partition, model_index), PARAM_INVALID);
  loader.is_inited_ = true;
  model_index = 0;
  type = TASK_INFO;
  ASSERT_TRUE(loader.model_contexts_.empty());
  EXPECT_EQ(loader.GetModelPartition(type, partition, model_index), PARAM_INVALID);
  ASSERT_TRUE(loader.model_contexts_.empty());

  loader.model_contexts_.emplace_back(OmFileContext{});
  EXPECT_EQ(loader.GetModelPartition(CUST_AICPU_KERNELS, partition, model_index), SUCCESS);
}

TEST_F(UtestOmFileHelper, GetModelPartitionNoIndex)
{
  OmFileLoadHelper loader;
  ModelPartitionType type = ModelPartitionType::MODEL_DEF;
  ModelPartition partition;
  loader.is_inited_ = false;
  EXPECT_EQ(loader.GetModelPartition(type, partition), PARAM_INVALID);
  loader.is_inited_ = true;
  type = TASK_INFO;
  ASSERT_TRUE(loader.model_contexts_.empty());
  EXPECT_EQ(loader.GetModelPartition(type, partition, 0U), PARAM_INVALID);
  ASSERT_TRUE(loader.model_contexts_.empty());
}

TEST_F(UtestOmFileHelper, LoadModelPartitionTable)
{
  OmFileLoadHelper loader;
  uint8_t* model_data = nullptr;
  uint64_t model_data_size = 0;
  size_t mem_offset = 0U;
  EXPECT_EQ(loader.LoadModelPartitionTable(model_data, model_data_size, 0U, mem_offset, nullptr), ACL_ERROR_GE_EXEC_MODEL_ADDR_INVALID);
}

TEST_F(UtestOmFileHelper, LoadModelPartitionTableWithNum)
{
  OmFileLoadHelper loader;
  uint8_t* model_data = nullptr;
  uint64_t model_data_size = 0;
  uint32_t model_num = 0;
  EXPECT_EQ(loader.LoadModelPartitionTable(model_data, model_data_size, model_num, nullptr), PARAM_INVALID);
}

TEST_F(UtestOmFileHelper, SaverGetPartitionTable1)
{
  OmFileSaveHelper saver;
  OmFileContext oc;
  oc.partition_datas_.push_back(ModelPartition());
  saver.model_contexts_.push_back(oc);
  size_t cur_ctx_index = 0;
  EXPECT_NE(saver.GetPartitionTable(cur_ctx_index), nullptr);
}

TEST_F(UtestOmFileHelper, SaverGetPartitionTable2)
{
  OmFileSaveHelper saver;
  OmFileContext oc;
  oc.partition_datas_.push_back(ModelPartition());
  saver.model_contexts_.push_back(oc);
  size_t cur_ctx_index = 0;
  const bool is_partition_align = true;
  const uint32_t align_bytes = 8U; // 8字节对齐
  EXPECT_NE(saver.GetPartitionTable(cur_ctx_index, is_partition_align, align_bytes), nullptr);
}

TEST_F(UtestOmFileHelper, SaveModel)
{
  OmFileSaveHelper saver;
  OmFileContext oc;
  oc.model_data_len_ = 1024;
  saver.model_contexts_.push_back(oc);
  const char_t *const output_file = "root.model";
  ModelBufferData model;
  const bool is_offline = true;
  EXPECT_NE(saver.SaveModel(output_file, model, is_offline), SUCCESS);
}

TEST_F(UtestOmFileHelper, SaveModel2)
{
  OmFileSaveHelper saver;
  OmFileContext oc;
  oc.model_data_len_ = 1024;
  saver.model_contexts_.push_back(oc);
  const char_t *const output_file = "root.model";
  ModelBufferData model;
  const bool is_offline = true;
  const bool is_partition_align = true;
  const uint32_t align_bytes = 8U; // 8字节对齐
  EXPECT_NE(saver.SaveModel(output_file, model, is_offline, is_partition_align, align_bytes), SUCCESS);
}

TEST_F(UtestOmFileHelper, AddPartition)
{
  OmFileSaveHelper saver;
  ModelPartition partition;
  EXPECT_EQ(saver.AddPartition(partition), SUCCESS);
  ASSERT_FALSE(saver.model_contexts_.empty());
  saver.model_contexts_[0U].model_data_len_ = 4000000000U;
  partition.size = 2000000000U;
  EXPECT_EQ(saver.AddPartition(partition), SUCCESS);
}

TEST_F(UtestOmFileHelper, TestInvalidPartitionNumber) {
  std::vector<ModelPartitionType> valid0{MODEL_DEF};
  EXPECT_EQ(OmFileLoadHelper::CheckPartitionTableNum(valid0.size()), true);

  std::vector<ModelPartitionType> valid1{MODEL_DEF, WEIGHTS_DATA,       TASK_INFO,  TBE_KERNELS,
                                         SO_BINS,   CUST_AICPU_KERNELS, TILING_DATA};
  EXPECT_EQ(OmFileLoadHelper::CheckPartitionTableNum(valid1.size()), true);
}

}  // namespace ge

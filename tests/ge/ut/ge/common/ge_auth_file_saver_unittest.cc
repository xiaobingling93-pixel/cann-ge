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
#include "common/helper/file_saver.h"
#include "graph/passes/graph_builder_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/ops_stub.h"
#include "macro_utils/dt_public_unscope.h"

#include "framework/common/ge_inner_error_codes.h"
namespace ge {
class UTEST_file_saver : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UTEST_file_saver, OpenFile_success) {
  int32_t fd;
  EXPECT_EQ(FileSaver::OpenFile(fd, "/"), FAILED);
  EXPECT_EQ(FileSaver::OpenFile(fd, "./test.txt"), SUCCESS);
  
  char_t *path = new char_t[MMPA_MAX_PATH + 1];
  memset(path, 'a', MMPA_MAX_PATH);
  path[MMPA_MAX_PATH] = 0;
  EXPECT_EQ(FileSaver::CheckPathValid(path), FAILED);
  delete[] path;
}



TEST_F(UTEST_file_saver, save_model_data_to_buff_success) {
  ModelFileHeader file_header;
  std::vector<char> data;
  data.resize(sizeof(ModelPartitionTable) + sizeof(ModelPartitionMemInfo), 0);
  ModelPartitionTable *partition_table = reinterpret_cast<ModelPartitionTable*>(data.data());
  partition_table->num = 1;
  size_t buff_size = SECUREC_MEM_MAX_LEN + 1;
  partition_table->partition[0] = { MODEL_DEF, 0, buff_size };
  std::vector<ModelPartitionTable *> partition_tables;
  partition_tables.push_back(partition_table);
  auto buff = reinterpret_cast<uint8_t *>(malloc(buff_size));
  EXPECT_NE(buff, nullptr);
  struct ge::ModelPartition model_partition;
  model_partition.type = MODEL_DEF;
  model_partition.data = buff;
  model_partition.size = buff_size;
  std::vector<ModelPartition> model_partitions = { model_partition };
  std::vector<std::vector<ModelPartition>> all_partition_datas = { model_partitions };
  ge::ModelBufferData model;

  Status ret = FileSaver::SaveToBuffWithFileHeader(file_header, partition_tables, all_partition_datas, model);
  EXPECT_EQ(ret, ge::SUCCESS);

  free(buff);
  buff = nullptr;
  model_partition.data = nullptr;
}

TEST_F(UTEST_file_saver, save_model_data_to_buff2_success) {
  ModelFileHeader file_header;
  std::vector<char> data;
  data.resize(sizeof(ModelPartitionTable) + sizeof(ModelPartitionMemInfo), 0);
  ModelPartitionTable *partition_table = reinterpret_cast<ModelPartitionTable*>(data.data());
  partition_table->num = 1;
  partition_table->partition[0] = { MODEL_DEF, 0, 12 };
  auto buff = reinterpret_cast<uint8_t *>(malloc(12));
  struct ge::ModelPartition model_partition;
  model_partition.type = MODEL_DEF;
  model_partition.data = buff;
  model_partition.size = 12;
  std::vector<ModelPartition> model_partitions = { model_partition };
  ge::ModelBufferData model;
  Status ret = FileSaver::SaveToBuffWithFileHeader(file_header, *partition_table, model_partitions, model);
  EXPECT_EQ(ret, ge::SUCCESS);
  free(buff);
  buff = nullptr;
  model_partition.data = nullptr;
}

TEST_F(UTEST_file_saver, SaveToFile1_success) {
  std::string file_path("");
  std::string model_data_str = "123456789";
  ge::ModelData modelData;
  modelData.model_data = &model_data_str;
  modelData.model_len = model_data_str.size();

  ModelFileHeader *model_file_header = reinterpret_cast<ModelFileHeader *>(modelData.model_data);
  EXPECT_EQ(FileSaver::SaveToFile(file_path, modelData, model_file_header), FAILED);
  file_path = "./test.om";
  EXPECT_EQ(FileSaver::SaveToFile(file_path, modelData, model_file_header), SUCCESS);
  system("rm -rf ./test.om");
}

TEST_F(UTEST_file_saver, SaveToFile2_success) {
  std::string model_data_str = "123456789";
  ge::ModelData modelData;
  modelData.model_data = &model_data_str;
  modelData.model_len = model_data_str.size();
  EXPECT_EQ(FileSaver::SaveToFile("", &model_data_str, model_data_str.size()), FAILED);
  EXPECT_EQ(FileSaver::SaveToFile("./test.om", &model_data_str, model_data_str.size()), SUCCESS);
  system("rm -rf ./test.om");
}

TEST_F(UTEST_file_saver, SaveToFile3_success) {
  std::vector<ModelPartition> partition_datas;
  std::vector<char> data;
  data.resize(sizeof(ModelPartitionTable) + sizeof(ModelPartitionMemInfo), 0);
  ModelPartitionTable *partition_table = reinterpret_cast<ModelPartitionTable*>(data.data());
  partition_table->num = 1;
  partition_table->partition[0] = { MODEL_DEF, 0, 12 };
  ModelFileHeader model_header ;//= reinterpret_cast<ModelFileHeader *>(modelData.model_data);
  EXPECT_EQ(FileSaver::SaveToFile("./test.om", model_header, *partition_table, partition_datas), FAILED);

  auto buff = reinterpret_cast<uint8_t *>(malloc(12));
  struct ge::ModelPartition model_partition;
  model_partition.type = MODEL_DEF;
  model_partition.data = buff;
  model_partition.size = 12;
  std::vector<ModelPartition> model_partitions = { model_partition };
  EXPECT_EQ(FileSaver::SaveToFile("./test.om", model_header, *partition_table, model_partitions), SUCCESS);
  free(buff);
  buff = nullptr;
  model_partition.data = nullptr;
  system("rm -rf ./test.om");
}

TEST_F(UTEST_file_saver, SaveToFile4_success) {
  std::vector<std::vector<ModelPartition>> all_partition_datas;
  std::vector<char> data;
  data.resize(sizeof(ModelPartitionTable) + sizeof(ModelPartitionMemInfo), 0);
  ModelPartitionTable *partition_table = reinterpret_cast<ModelPartitionTable*>(data.data());
  partition_table->num = 1;
  partition_table->partition[0] = { MODEL_DEF, 0, 12 };
  std::vector<ModelPartitionTable *> partition_tables;
  partition_tables.push_back(partition_table);
  ModelFileHeader model_header;
  EXPECT_EQ(FileSaver::SaveToFile("./test.om", model_header, partition_tables, all_partition_datas), FAILED);

  auto buff = reinterpret_cast<uint8_t *>(malloc(12));
  struct ge::ModelPartition model_partition;
  model_partition.type = MODEL_DEF;
  model_partition.data = buff;
  model_partition.size = 12;
  std::vector<ModelPartition> model_partitions = { model_partition };
  all_partition_datas.push_back(model_partitions);
  EXPECT_EQ(FileSaver::SaveToFile("./test.om", model_header, partition_tables, all_partition_datas), SUCCESS);
  free(buff);
  buff = nullptr;
  model_partition.data = nullptr;
  system("rm -rf ./test.om");
}

TEST_F(UTEST_file_saver, SaveWithAlignFill1_success) {
  std::string file_path("./test.om");
  int32_t fd = 0;
  Status ret = FileSaver::OpenFile(fd, file_path);
  if (ret == FAILED) {
    fd = 0;
  }
  uint32_t size = 0;
  uint32_t align_bytes = 32U;
  EXPECT_EQ(FileSaver::SaveWithAlignFill(size, align_bytes, fd), SUCCESS);
  size = 1;
  EXPECT_EQ(FileSaver::SaveWithAlignFill(size, align_bytes, fd), SUCCESS);
  system("rm -rf ./test.om");
}

}  // namespace ge

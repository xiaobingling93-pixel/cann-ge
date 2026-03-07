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
#include <dirent.h>
#include <sys/stat.h>
#include <string>
#include <unistd.h>

namespace {
bool IsDirectory(const std::string &path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

bool IsRegularFile(const std::string &path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

void CountFilesAndDirs(const std::string &path, int &max_cnt) {
  int file_count = 0;
  int dir_count = 0;

  DIR *dir = opendir(path.c_str());
  if (dir == nullptr) {
    return;
  }

  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }
    std::string entry_path = path + "/" + entry->d_name;
    if (IsDirectory(entry_path)) {
      dir_count++;
      // 递归调用以检查子目录
      CountFilesAndDirs(entry_path, max_cnt);
    } else if (IsRegularFile(entry_path)) {
      file_count++;
    }
  }
  closedir(dir);

  const auto count = file_count + dir_count;
  if (count > 50) {
    throw std::runtime_error("Directory " + path +
                             " exceeds the limit of 50 files and directories number "
                             ":" +
                             std::to_string(count));
  }
  if (count > max_cnt) {
    max_cnt = count;
    std::cout << path << "/ with files and direct sub directories number is " << count << ", update max_cnt."
              << std::endl;
  }
}
}  // namespace
namespace SC {
/*
 * SC要求每个目录的直属子目录和文件之和不超过50
 * 因此添加这个ut来看护代码仓目录
 */
TEST(FileCount, CheckFileCount) {
  int max_cnt = 0;
  // 待rts_engine整改后取消注释
  // std::string dir0(std::string(TOP_DIR).append("/compiler"));
  // EXPECT_NO_THROW(CountFilesAndDirs(dir0, max_cnt));

  std::string dir1(std::string(TOP_DIR).append("/inc"));
  EXPECT_NO_THROW(CountFilesAndDirs(dir1, max_cnt));

  std::string dir2(std::string(TOP_DIR).append("/parser"));
  EXPECT_NO_THROW(CountFilesAndDirs(dir2, max_cnt));

  std::string dir3(std::string(TOP_DIR).append("/api/session"));
  EXPECT_NO_THROW(CountFilesAndDirs(dir3, max_cnt));

  std::string dir4(std::string(TOP_DIR).append("/runtime"));
  EXPECT_NO_THROW(CountFilesAndDirs(dir4, max_cnt));

  std::string dir5(std::string(TOP_DIR).append("/api/python"));
  EXPECT_NO_THROW(CountFilesAndDirs(dir5, max_cnt));

  std::string dir6(std::string(TOP_DIR).append("/base/common"));
  EXPECT_NO_THROW(CountFilesAndDirs(dir6, max_cnt));

  std::cout << "max_cnt is " << max_cnt << std::endl;
}
}  // namespace SC

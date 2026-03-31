/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "file_utils.h"
#include "common/helper/om2/zip_archive.h"
#include <gtest/gtest.h>
#include <fstream>
#include <unordered_set>
#include "common/env_path.h"
#include "mmpa/mmpa_api.h"

namespace ge {

class ZipArchiveUt : public ::testing::Test {
 public:
  void SetUp() override {
    const ::testing::TestInfo *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    test_case_name = test_info->test_case_name();  // ZipArchiveUt
    test_work_dir = EnvPath().GetOrCreateCaseTmpPath(test_case_name);
  }
  void TearDown() override {
    EnvPath().RemoveRfCaseTmpPath(test_case_name);
  }
  static void CreateTestZipArchive(const std::string &archive_path,
                                   const std::vector<std::pair<std::string, std::string>> &entries,
                                   const bool compress = true) {
    zipFile zf = zipOpen64(archive_path.c_str(), 0);
    ASSERT_NE(zf, nullptr);
    int32_t compress_flag = compress ? Z_DEFAULT_COMPRESSION : Z_NO_COMPRESSION;
    int32_t method = compress ? Z_DEFLATED : Z_BINARY;

    for (const auto &entry : entries) {
      const std::string &file_name = entry.first;
      const std::string &content = entry.second;

      zip_fileinfo zi;
      memset_s(&zi, sizeof(zi), 0, sizeof(zi));

      auto ret =
          zipOpenNewFileInZip64(zf, file_name.c_str(), &zi, nullptr, 0, nullptr, 0, nullptr, method, compress_flag, 1);

      if (ret != ZIP_OK) {
        (void)zipClose(zf, nullptr);
        return;
      }

      if (!content.empty()) {
        ret = zipWriteInFileInZip(zf, content.data(), static_cast<unsigned>(content.size()));
        if (ret != ZIP_OK) {
          (void)zipCloseFileInZip(zf);
          (void)zipClose(zf, nullptr);
          return;
        }
      }

      (void)zipCloseFileInZip(zf);
    }

    (void)zipClose(zf, nullptr);
  }

  static std::vector<uint8_t> ReadFileToVector(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
      return {};
    }
    file.seekg(0, std::ios::end);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size < 0) {
      return {};
    }
    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
      return {};
    }

    return buffer;
  }

  std::string CreateTempFile(const std::string &file_name, const size_t file_size = 100) {
    const std::string full_path = PathUtils::Join({test_work_dir, file_name});
    std::ofstream ofs(full_path, std::ios::out | std::ios::binary);
    if (!ofs.is_open()) {
      return {};
    }

    constexpr size_t kBlockSize = 4096;
    const std::string block(kBlockSize, 'a');
    size_t written = 0;
    while (written < file_size) {
      const size_t to_write = std::min(kBlockSize, file_size - written);
      ofs.write(block.data(), to_write);
      written += to_write;
    }
    ofs.close();
    return full_path;
  }

  void CheckExtractedFiles(const std::string &zipfile_path,
                           const std::unordered_set<std::string> &expected_entries) const {
    const auto file_buf = ReadFileToVector(zipfile_path);
    RAIIZipArchive unzip_file(file_buf.data(), file_buf.size());
    ASSERT_TRUE(unzip_file.IsGood());
    const auto file_names = unzip_file.ListFiles();
    ASSERT_EQ(expected_entries.size(), file_names.size());
    for (const auto &entry : expected_entries) {
      const auto extract_file_name = PathUtils::Join({test_work_dir, kZipFileBaseName, entry});
      ASSERT_TRUE(unzip_file.ExtractToFile(PathUtils::Join({kZipFileBaseName, entry}), test_work_dir));
      ASSERT_EQ(mmAccess2(extract_file_name.c_str(), M_F_OK), EN_OK);
    }
  }

 public:
  std::string test_case_name;
  std::string test_work_dir;
  const std::string kZipFileBaseName = "fake_test";
};

TEST_F(ZipArchiveUt, TestRaiiZipArchive_Fail_InvalidFileOrData) {
  RAIIZipArchive unzip_file_invalid_data(nullptr, 0);
  EXPECT_EQ(unzip_file_invalid_data.IsGood(), false);
}

TEST_F(ZipArchiveUt, TestRaiiZipArchive_Ok_DecompressArchive) {
  const std::string archive_path = PathUtils::Join({test_work_dir, "__test.zip"});
  const std::vector<std::pair<std::string, std::string>> entries = {
      {"example/demo1.txt", "Hello from demo1!\nThis is example."},
      {"doc/doc1.txt", "Document 1 inside zip.\n-- EOF --"},
  };
  CreateTestZipArchive(archive_path, entries);
  // 测试从内存读取与解压功能
  {
    const auto zip_file_buf = ReadFileToVector(archive_path);
    RAIIZipArchive archive(zip_file_buf.data(), zip_file_buf.size());
    EXPECT_EQ(archive.IsGood(), true);
    const auto file_names = archive.ListFiles();
    const std::string extract_path = PathUtils::Join({test_work_dir, "temp_extract"});
    ASSERT_EQ(CreateDir(extract_path), 0);
    ASSERT_EQ(file_names.size(), 2);
    for (const auto &file_name : file_names) {
      ASSERT_TRUE(archive.ExtractToFile(file_name, extract_path));
      ASSERT_EQ(mmAccess2(PathUtils::Join({extract_path, file_name}).c_str(), M_F_OK), EN_OK);
    }
  }
}

TEST_F(ZipArchiveUt, TestRaiiZipArchive_Ok_ExtractToMem) {
  const std::string archive_path = PathUtils::Join({test_work_dir, "__test.zip"});
  std::string data_str1 = "1234test_zip_archive";
  const std::vector<std::pair<std::string, std::string>> entries = {
      {"example/demo1.txt", data_str1},
  };
  CreateTestZipArchive(archive_path, entries);
  // 测试从内存读取与解压功能
  {
    const auto file_buf = ReadFileToVector(archive_path);
    RAIIZipArchive archive(file_buf.data(), file_buf.size());
    EXPECT_EQ(archive.IsGood(), true);
    const auto file_names = archive.ListFiles();
    ASSERT_EQ(file_names.size(), 1);
    for (const auto &file_name : file_names) {
      size_t buff_size = 0UL;
      const auto buff_data = archive.ExtractToMem(file_name, buff_size);
      ASSERT_NE(buff_data, nullptr);
      EXPECT_EQ(buff_size, data_str1.size());
      ASSERT_TRUE(std::memcmp(buff_data.get(), data_str1.data(), buff_size) == 0);
    }
  }
}

TEST_F(ZipArchiveUt, TestRaiiZipArchive_Ok_ExtractToMemNoCompression) {
  const std::string archive_path = PathUtils::Join({test_work_dir, "__test.zip"});
  std::string data_str1(123456, 'c');
  const std::vector<std::pair<std::string, std::string>> entries = {
      {"example1/demo1.txt", data_str1},
      {"example2/demo2.txt", data_str1},
  };
  CreateTestZipArchive(archive_path, entries);
  // 测试从内存读取与解压功能
  {
    const auto file_buf = ReadFileToVector(archive_path);
    RAIIZipArchive archive(file_buf.data(), file_buf.size());
    EXPECT_EQ(archive.IsGood(), true);
    const auto file_names = archive.ListFiles();
    ASSERT_EQ(file_names.size(), 2);
    for (const auto &file_name : file_names) {
      size_t buff_size = 0UL;
      const auto buff_data = archive.ExtractToMem(file_name, buff_size);
      ASSERT_NE(buff_data, nullptr);
      EXPECT_EQ(buff_size, data_str1.size());
      ASSERT_TRUE(std::memcmp(buff_data.get(), data_str1.data(), buff_size) == 0);
    }
  }
}

TEST_F(ZipArchiveUt, TestRaiiZipArchive_Ok_ExtractToMemNoCompressionLargeFile) {
  const std::string archive_path = PathUtils::Join({test_work_dir, "__test.zip"});
  constexpr size_t file_size = static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1;
  {
    ZipArchiveWriter zip_writer(archive_path);
    const std::string file_path = CreateTempFile("fake_test.txt", file_size);
    ASSERT_TRUE(zip_writer.IsMemFileOpened());
    EXPECT_TRUE(zip_writer.WriteFile("fake_test.txt", file_path, false));
    ASSERT_TRUE(zip_writer.SaveModelDataToFile());
    ASSERT_FALSE(zip_writer.IsMemFileOpened());
  }
  // 测试从内存读取与解压功能
  {
    const auto file_buf = ReadFileToVector(archive_path);
    RAIIZipArchive archive(file_buf.data(), file_buf.size());
    EXPECT_EQ(archive.IsGood(), true);
    const auto file_names = archive.ListFiles();
    ASSERT_EQ(file_names.size(), 1);
    for (const auto &file_name : file_names) {
      size_t buff_size = 0UL;
      const auto buff_data = archive.ExtractToMem(file_name, buff_size);
      ASSERT_NE(buff_data, nullptr);
      EXPECT_EQ(buff_size, file_size);
    }
  }
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Fail_InvalidArchiveName) {
  ZipArchiveWriter zip_writer("");
  EXPECT_FALSE(zip_writer.IsMemFileOpened());
  ZipArchiveWriter zip_writer2(".");
  EXPECT_FALSE(zip_writer.IsMemFileOpened());
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Fail_InvaidFileOrDataBuffIsNull) {
  const auto zipfile_path = PathUtils::Join({test_work_dir, "invalid_case.zip"});
  ZipArchiveWriter zip_writer(zipfile_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  ASSERT_FALSE(zip_writer.WriteFile("data/fake_config.json", "fake_config.json"));
  ASSERT_FALSE(zip_writer.WriteFile("data/fake_config.json", ""));
  ASSERT_FALSE(zip_writer.WriteFile("", ""));
  ASSERT_FALSE(zip_writer.WriteBytes("data/fake_data.bin", nullptr, 123));
  ASSERT_FALSE(zip_writer.WriteBytes("data/fake_data.bin", zipfile_path.data(), 0));
  ASSERT_FALSE(zip_writer.WriteBytes("", zipfile_path.data(), 123));
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Fail_StateAfterFinalization) {
  const auto zipfile_path = PathUtils::Join({test_work_dir, "invalid_case.zip"});
  ZipArchiveWriter zip_writer(zipfile_path);
  EXPECT_TRUE(zip_writer.IsMemFileOpened());
  EXPECT_TRUE(zip_writer.SaveModelDataToFile());
  EXPECT_FALSE(zip_writer.IsMemFileOpened());
  EXPECT_FALSE(zip_writer.WriteFile("test.txt", CreateTempFile("fake_test.txt")));
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Ok_WriteBytesAndFileSucc) {
  const std::string zipfile_name = kZipFileBaseName + ".zip";
  const auto zipfile_path = PathUtils::Join({test_work_dir, zipfile_name});
  const std::string buffer = "123-abc-TestZipArchiveWriter_Ok_WriteBytesSucc";
  const std::string arc_name = "ok/file1.txt";
  const std::string file_path = CreateTempFile("fake_test.txt");
  const std::string arc_name2 = "ok/ok/file2.txt";

  ZipArchiveWriter zip_writer(zipfile_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  EXPECT_TRUE(zip_writer.WriteBytes(arc_name, buffer.data(), buffer.size()));
  EXPECT_TRUE(zip_writer.WriteFile(arc_name2, file_path));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());
  ASSERT_FALSE(zip_writer.IsMemFileOpened());

  // 解压并校验内容
  CheckExtractedFiles(zipfile_path, {arc_name, arc_name2});
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Ok_RepeatedAddSameFile) {
  const std::string zipfile_name = kZipFileBaseName + ".zip";
  const auto zipfile_path = PathUtils::Join({test_work_dir, zipfile_name});
  const std::string buffer = "123-abc-TestZipArchiveWriter_Ok_WriteBytesSucc";
  const std::string arc_name = "ok/file1.txt";
  const std::string file_path = CreateTempFile("fake_test.txt");
  const std::string arc_name2 = "ok/ok/file2.txt";

  // 重复添加相同文件
  ZipArchiveWriter zip_writer(zipfile_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  EXPECT_TRUE(zip_writer.WriteBytes(arc_name, buffer.data(), buffer.size()));
  EXPECT_TRUE(zip_writer.WriteBytes(arc_name, buffer.data(), buffer.size()));
  EXPECT_TRUE(zip_writer.WriteFile(arc_name2, file_path));
  EXPECT_TRUE(zip_writer.WriteFile(arc_name2, file_path));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());
  ASSERT_FALSE(zip_writer.IsMemFileOpened());

  // 解压并校验内容
  CheckExtractedFiles(zipfile_path, {arc_name, arc_name2});
}
}  // namespace ge
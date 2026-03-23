/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_COMMON_HELPER_ZIP_ARCHIVE
#define BASE_COMMON_HELPER_ZIP_ARCHIVE
#include <string>
#include <vector>
#include <fstream>
#include <unordered_set>

#include "common/ge_common/ge_types.h"
#include "zip_mem_ioapi.h"
#include "minizip/zip.h"
#include "minizip/unzip.h"

namespace ge {
class RAIIZipArchive {
 public:
  /**
   * Constructs a RAIIZipArchive object using a ZIP archive that already
   * resides in memory.
   * @param data Pointer to the beginning of the ZIP data in memory.
   * @param length Size of the ZIP data in bytes.
   */
  RAIIZipArchive(const uint8_t *data, const size_t length);

  ~RAIIZipArchive();
  bool IsGood() const {
    return zip_handle_ != nullptr;
  }
  /**
   * Lists all regular files in the ZIP archive, excluding directories.
   * @return Vector containing relative paths of all files in the archive.
   */
  std::vector<std::string> ListFiles() const;
  /**
   * Extracts a single file from the ZIP archive to the specified directory.
   * @param entry_name Filename (relative path) within the ZIP archive.
   * @param output_dir Output directory. Will be created if it does not exist.
   * @return true if extraction succeeded, false otherwise.
   */
  bool ExtractToFile(const std::string &entry_name, const std::string &output_dir) const;
  UniqueByteBuffer ExtractToMem(const std::string &entry_name, size_t &buff_size) const;

 private:
  UniqueByteBuffer FastReadRawDataToMem(const std::string &entry_name, const size_t pos_in_central_dir,
                                        const size_t buff_size) const;

 private:
  MemoryFileReadonly mem_file_{};
  unzFile zip_handle_ = nullptr;
};

class ZipArchiveWriter {
 public:
  explicit ZipArchiveWriter(const std::string &archive_path);
  ~ZipArchiveWriter();
  bool WriteFile(const std::string &entry_name, const std::string &src_file_path, const bool compress = true);
  /**
   * Write a memory buffer as a file entry into the opened zip archive.
   * @param entry_name  Name of the file inside the zip archive.
   * @param data      Pointer to the buffer to write. Must not be null.
   * @param data_size Size in bytes of the buffer. Extremely large (>4GB) buffers are supported.
   * @param compress  Whether to compress the data. If false, the compression method is STORED.
   * @return true on success, false if any error occurs.
   */
  bool WriteBytes(const std::string &entry_name, const void *data, const size_t data_size, const bool compress = true);
  bool SaveModelDataToFile();
  bool IsMemFileOpened() const {
    return (zip_handle_ != nullptr) && (mem_file_.buffer != nullptr);
  }

 private:
  bool InitArchive();
  bool WriteEndOfFile();

 private:
  std::string archive_path_;
  std::string base_name_;
  zipFile zip_handle_{};
  MemoryFile mem_file_{};
  std::unordered_set<std::string> files_written_;
};
}  // namespace ge

#endif
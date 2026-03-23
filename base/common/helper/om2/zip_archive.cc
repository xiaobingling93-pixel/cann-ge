/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/helper/om2/zip_archive.h"

#include <fstream>
#include <array>
#include <vector>

#include "framework/common/debug/log.h"
#include "common/checker.h"
#include "common/scope_guard.h"
#include "mmpa/mmpa_api.h"
#include "graph_metadef/graph/utils/file_utils.h"

namespace ge {
namespace {
constexpr uint32_t kMaxFileNameLength = 4096U;  // same as UNZ_MAXFILENAMEINZIP
constexpr int64_t kBufSize = 16384UL;           // same as UNZ_BUFSIZE
constexpr uint32_t kMaxWriteSize = std::numeric_limits<uint32_t>::max();
std::string ParentDirectory(const std::string &filepath) {
  size_t last_slash = filepath.find_last_of('/');
  if (last_slash != std::string::npos) {
    return filepath.substr(0, last_slash);
  }
  return "";
}

bool MakeSureDirExists(const std::string &file_path) {
  const auto parent_dir = ParentDirectory(file_path);
  GE_ASSERT(!parent_dir.empty());
  if (mmAccess2(parent_dir.c_str(), M_F_OK) != EN_OK) {
    GE_ASSERT_TRUE(CreateDir(parent_dir) == 0);
  }
  return true;
}

std::string GetBaseName(const std::string &path) {
  if (path.empty()) {
    return "";
  }

  const auto pos_slash = path.find_last_of('/');
  std::string file_name = (pos_slash == std::string::npos) ? path : path.substr(pos_slash + 1);
  if (file_name.empty()) {
    return "";
  }

  const auto pos_dot = file_name.find_last_of('.');
  if (pos_dot == std::string::npos || pos_dot == 0) return file_name;

  return file_name.substr(0, pos_dot);
}

// The following functions are used to support random access to files in a zip.
// Central directory file header (CD)
constexpr uint32_t kCdHeaderFixedSize = 46U;
constexpr uint32_t kCdCompressedSizeOffset = 20U;
constexpr uint32_t kCdUncompressedSizeOffset = 24U;
constexpr uint32_t kCdFileNameLenOffset = 28U;
constexpr uint32_t kCdExtraLenOffset = 30U;
constexpr uint32_t kCdLocalFileHeaderOffset = 42U;
constexpr uint32_t kCdHeaderMagicNum = 0x02014b50;
constexpr uint16_t kCdExtraMagicNum = 0x0001;
// Local file header (LF)
constexpr uint16_t kLfHeaderFixedSize = 30U;
constexpr uint32_t kLfNameLenOffset = 26U;
constexpr uint32_t kLfExtraLenOffset = 28U;
constexpr uint32_t kLfHeaderMagicNum = 0x04034b50;
// Byte/bit size definitions
constexpr size_t kBitsPerByte = 8;
constexpr size_t kBytesPerUint16 = sizeof(uint16_t);
constexpr size_t kBytesPerUint32 = sizeof(uint32_t);
constexpr size_t kBytesPerUint64 = sizeof(uint64_t);
constexpr size_t kBitsPerUint16 = kBitsPerByte * kBytesPerUint16;
constexpr size_t kBitsPerUint32 = kBitsPerByte * kBytesPerUint32;

struct ZipEntryInfo {
  // Size of compressed data in bytes (actual length stored in file)
  uint64_t compressed_size;
  // Size of data after decompression in bytes
  uint64_t uncompressed_size;
  // File offset to the Local File Header for an entry
  uint64_t local_file_header_offset;
};

uint16_t ReadLE16(const uint8_t *p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << kBitsPerByte);
}

uint32_t ReadLE32(const uint8_t *p) {
  return static_cast<uint32_t>(ReadLE16(p)) | (static_cast<uint32_t>(ReadLE16(p + kBytesPerUint16)) << kBitsPerUint16);
}

uint64_t ReadLE64(const uint8_t *p) {
  return static_cast<uint64_t>(ReadLE32(p)) | (static_cast<uint64_t>(ReadLE32(p + kBytesPerUint32)) << kBitsPerUint32);
}

bool ParseCentralDirEntry(const MemoryFileReadonly &buffer, const uint64_t pos_in_central_dir, ZipEntryInfo &entry_info) {
  GE_ASSERT_TRUE(pos_in_central_dir + kCdHeaderFixedSize <= buffer.length, "Invalid central directory position");
  const uint8_t *entry_buff = buffer.buffer + pos_in_central_dir;

  const uint32_t magic_num = ReadLE32(entry_buff);
  GE_ASSERT_TRUE(magic_num == kCdHeaderMagicNum, "Invalid central directory file header, magic = %u", magic_num);

  entry_info.compressed_size = ReadLE32(entry_buff + kCdCompressedSizeOffset);
  entry_info.uncompressed_size = ReadLE32(entry_buff + kCdUncompressedSizeOffset);

  const uint16_t name_len = ReadLE16(entry_buff + kCdFileNameLenOffset);
  const uint16_t extra_len = ReadLE16(entry_buff + kCdExtraLenOffset);
  entry_info.local_file_header_offset = ReadLE32(entry_buff + kCdLocalFileHeaderOffset);

  if (entry_info.compressed_size == MAXU32 || entry_info.uncompressed_size == MAXU32 ||
      entry_info.local_file_header_offset == MAXU32) {
    GE_ASSERT_TRUE(pos_in_central_dir + kCdHeaderFixedSize + name_len + extra_len <= buffer.length,
                   "Invalid extra field position");

    const uint8_t *extra_buf = entry_buff + kCdHeaderFixedSize + name_len;
    size_t offset = 0;
    while (offset + kBytesPerUint32 <= extra_len) {
      const uint16_t header_id = ReadLE16(extra_buf + offset);
      const uint16_t data_size = ReadLE16(extra_buf + offset + kBytesPerUint16);
      offset += kBytesPerUint32;

      if (header_id == kCdExtraMagicNum) {
        if ((entry_info.uncompressed_size == MAXU32) && (offset + kBytesPerUint64 <= extra_len)) {
          entry_info.uncompressed_size = ReadLE64(extra_buf + offset);
          offset += kBytesPerUint64;
        }
        if ((entry_info.compressed_size == MAXU32) && (offset + kBytesPerUint64 <= extra_len)) {
          entry_info.compressed_size = ReadLE64(extra_buf + offset);
          offset += kBytesPerUint64;
        }
        if ((entry_info.local_file_header_offset == MAXU32) && (offset + kBytesPerUint64 <= extra_len)) {
          entry_info.local_file_header_offset = ReadLE64(extra_buf + offset);
          offset += kBytesPerUint64;
        }
        // Skip reading disk start number
        break;
      } else {
        offset += data_size;
      }
    }
  }
  return true;
}

bool LocateFileDataOffset(const MemoryFileReadonly &buffer, const uint64_t local_file_header_offset,
                          uint64_t &raw_data_offset) {
  GE_ASSERT_TRUE(local_file_header_offset + kLfHeaderFixedSize <= buffer.length, "Invalid local file header position");
  const uint8_t *header_buff = buffer.buffer + local_file_header_offset;

  const uint32_t header_magic_num = ReadLE32(header_buff);
  GE_ASSERT_TRUE(header_magic_num == kLfHeaderMagicNum, "Invalid local file header, magic = %u", header_magic_num);

  const int64_t name_len = ReadLE16(header_buff + kLfNameLenOffset);
  const int64_t extra_len = ReadLE16(header_buff + kLfExtraLenOffset);
  raw_data_offset = local_file_header_offset + kLfHeaderFixedSize + name_len + extra_len;
  GE_ASSERT_TRUE(raw_data_offset <= buffer.length, "Invalid file data offset");

  return true;
}
}  // namespace
RAIIZipArchive::RAIIZipArchive(const uint8_t *data, const size_t length) : mem_file_{data, length, 0} {
  if (mem_file_.buffer == nullptr || mem_file_.length == 0) {
    GELOGE(FAILED, "Invalid zip archive data, data is [%p] and size is [%zu]", mem_file_.buffer, mem_file_.length);
    return;
  }

  zlib_filefunc64_def file_funcs;
  FillMemFileFuncReadonly(&file_funcs, &mem_file_);
  // file_funcs 是局部变量，unzOpen2_64内部会拷贝函数指针
  zip_handle_ = unzOpen2_64(nullptr, &file_funcs);
  if (zip_handle_ == nullptr) {
    GELOGE(FAILED, "Failed to open ZIP file from memory");
  }
}

RAIIZipArchive::~RAIIZipArchive() {
  if (zip_handle_ != nullptr) {
    unzClose(zip_handle_);
  }
}

std::vector<std::string> RAIIZipArchive::ListFiles() const {
  GE_ASSERT_NOTNULL(zip_handle_, "Invalid status of archive");

  std::vector<std::string> file_list;
  auto uz_ret = unzGoToFirstFile(zip_handle_);
  GE_ASSERT_TRUE(uz_ret == UNZ_OK, "Failed to go to the first file in the archive, ret = %d", uz_ret);

  do {
    std::vector<char_t> name_buff(kMaxFileNameLength, '\0');
    uz_ret = unzGetCurrentFileInfo64(zip_handle_, nullptr, name_buff.data(), name_buff.size(), nullptr, 0, nullptr, 0);
    GE_ASSERT_TRUE(uz_ret == UNZ_OK, "Failed to get the current file information, ret = %d", uz_ret);
    // ignore directory
    const std::string file_name(name_buff.data());
    if (!file_name.empty() && file_name.back() != '/') {
      file_list.emplace_back(file_name);
    }
    uz_ret = unzGoToNextFile(zip_handle_);
  } while (uz_ret == UNZ_OK);

  GE_ASSERT_TRUE(uz_ret == UNZ_END_OF_LIST_OF_FILE, "unzGoToNextFile failed, ret=%d", uz_ret);

  return file_list;
}

bool RAIIZipArchive::ExtractToFile(const std::string &entry_name, const std::string &output_dir) const {
  GE_ASSERT_NOTNULL(zip_handle_, "Invalid status of archive");
  GE_ASSERT_TRUE(!output_dir.empty(), "The name of output directory is empty");

  auto uz_ret = unzLocateFile(zip_handle_, entry_name.c_str(), 0);
  GE_ASSERT_TRUE(uz_ret == UNZ_OK, "Failed to locate file [%s], ret = %d", entry_name.c_str(), uz_ret);

  uz_ret = unzOpenCurrentFile(zip_handle_);
  GE_ASSERT_TRUE(uz_ret == UNZ_OK, "Failed to open file [%s], ret = %d", entry_name.c_str(), uz_ret);
  // Close the file in zip opened with unzOpenCurrentFile
  GE_MAKE_GUARD(zipfile_guard, [this]() { (void)unzCloseCurrentFile(zip_handle_); });

  auto output_path = output_dir + "/" + entry_name;
  GE_ASSERT_TRUE(MakeSureDirExists(output_path));
  std::ofstream ofs(output_path, std::ios::binary);
  GE_ASSERT_TRUE(ofs.is_open(), "Failed to open file [%s]", output_path.c_str());

  std::vector<char> buffer(kBufSize);
  size_t total_read = 0;
  int32_t bytes_read = 0;

  while ((bytes_read = unzReadCurrentFile(zip_handle_, buffer.data(), buffer.size())) > 0) {
    ofs.write(buffer.data(), bytes_read);
    total_read += bytes_read;
  }
  // verify here whether the file has been read correctly to EOF
  GE_ASSERT_TRUE(bytes_read == 0, "Failed to read file [%s], ret = %d", entry_name.c_str(), bytes_read);
  GELOGI("Successfully extract file [%s], total_read = %d bytes", entry_name.c_str(), total_read);

  return true;
}

UniqueByteBuffer RAIIZipArchive::ExtractToMem(const std::string &entry_name, size_t &buff_size) const {
  GE_ASSERT_NOTNULL(zip_handle_, "Invalid status of archive");

  auto uz_ret = unzLocateFile(zip_handle_, entry_name.c_str(), 0);
  GE_ASSERT_TRUE(uz_ret == UNZ_OK, "Failed to locate file [%s], ret = %d", entry_name.c_str(), uz_ret);

  uz_ret = unzOpenCurrentFile(zip_handle_);
  GE_ASSERT_TRUE(uz_ret == UNZ_OK, "Failed to open file [%s], ret = %d", entry_name.c_str(), uz_ret);
  // Close the file in zip opened with unzOpenCurrentFile
  GE_MAKE_GUARD(zipfile_guard, [this]() { (void)unzCloseCurrentFile(zip_handle_); });

  unz_file_info64 file_info{};
  uz_ret = unzGetCurrentFileInfo64(zip_handle_, &file_info, nullptr, 0, nullptr, 0, nullptr, 0);
  GE_ASSERT_TRUE(uz_ret == UNZ_OK, "Failed to get the current file information, ret = %d", uz_ret);
  GE_ASSERT_TRUE(file_info.uncompressed_size > 0);
  buff_size = file_info.uncompressed_size;

  if (file_info.compression_method == Z_NO_COMPRESSION) {
    unz64_file_pos file_pos;
    uz_ret = unzGetFilePos64(zip_handle_, &file_pos);
    GE_ASSERT_TRUE(uz_ret == UNZ_OK, "Failed to get the file pos, ret = %d", uz_ret);
    return FastReadRawDataToMem(entry_name, file_pos.pos_in_zip_directory, buff_size);
  }

  auto buffer = UniqueByteBuffer(new (std::nothrow) uint8_t[buff_size], ConditionalDeleter{true});
  GE_ASSERT_NOTNULL(buffer, "Failed to allocate buffer, size = %zu", buff_size);
  size_t total_read = 0;
  int32_t bytes_read = 0;
  do {
    const uint32_t remaining = static_cast<uint32_t>(
        std::min<size_t>(buff_size - total_read, static_cast<size_t>(std::numeric_limits<int32_t>::max())));
    bytes_read = unzReadCurrentFile(zip_handle_, buffer.get() + total_read, remaining);

    GE_ASSERT_TRUE(bytes_read >= 0, "Failed to read file [%s], ret = %d", entry_name.c_str(), bytes_read);
    total_read += static_cast<size_t>(bytes_read);
  } while (bytes_read > 0);  // EOF when bytes_read == 0

  GE_ASSERT_TRUE(total_read == buff_size, "Failed to extract file [%s], expected = %zu bytes, actual = %zu bytes",
                 entry_name.c_str(), buff_size, total_read);
  GELOGI("Successfully extract file [%s], total_read = %d bytes", entry_name.c_str(), total_read);

  return buffer;
}

UniqueByteBuffer RAIIZipArchive::FastReadRawDataToMem(const std::string &entry_name, const size_t pos_in_central_dir,
                                                   const size_t buff_size) const {
  GELOGI("Begin to read raw data of entry [%s]", entry_name.c_str());

  ZipEntryInfo entry_info{};
  GE_ASSERT_TRUE(ParseCentralDirEntry(mem_file_, pos_in_central_dir, entry_info));
  GE_ASSERT_TRUE(entry_info.compressed_size == entry_info.uncompressed_size,
                 "uncompressed_size and compressed_size must be equal when loading raw data");
  GE_ASSERT_TRUE(buff_size == entry_info.uncompressed_size, "buff_size is %zu, but uncompressed_size is %zu", buff_size,
                 entry_info.uncompressed_size);

  uint64_t raw_data_offset = 0UL;
  GE_ASSERT_TRUE(LocateFileDataOffset(mem_file_, entry_info.local_file_header_offset, raw_data_offset));
  GE_ASSERT_TRUE(raw_data_offset + entry_info.uncompressed_size <= mem_file_.length);
  GELOGI("Successfully get raw data of entry [%s], offset = %zu, size = %zu", entry_name.c_str(), raw_data_offset,
         buff_size);

  return UniqueByteBuffer(const_cast<uint8_t *>(mem_file_.buffer + raw_data_offset), ConditionalDeleter{false});
}

ZipArchiveWriter::ZipArchiveWriter(const std::string &archive_path)
    : archive_path_(archive_path), base_name_(GetBaseName(archive_path)) {
  if (!InitArchive()) {
    GELOGE(FAILED, "Failed to initialize archive [%s]", archive_path.c_str());
  }
}

ZipArchiveWriter::~ZipArchiveWriter() {
  (void)WriteEndOfFile();
  if (mem_file_.buffer != nullptr) {
    std::free(mem_file_.buffer);
    mem_file_ = {};
  }
}

bool ZipArchiveWriter::WriteFile(const std::string &entry_name, const std::string &src_file_path, const bool compress) {
  GE_ASSERT_TRUE(IsMemFileOpened(), "Invalid status of archive [%s]", archive_path_.c_str());
  GE_ASSERT_TRUE(!entry_name.empty(), "Entry name cannot be empty");
  const std::string arc_name_with_prefix = base_name_ + "/" + entry_name;
  if (files_written_.count(arc_name_with_prefix) != 0) {
    GELOGW("Duplicate file entry '%s' detected and ignored, src_file_name [%s]", arc_name_with_prefix.c_str(),
           src_file_path.c_str());
    return true;
  }

  std::ifstream fin(src_file_path, std::ios::binary);
  GE_ASSERT_TRUE(fin.is_open(), "Failed to open file [%s]", src_file_path.c_str());
  fin.seekg(0, std::ios::end);
  const auto file_size = fin.tellg();
  fin.seekg(0, std::ios::beg);

  zip_fileinfo file_info{};
  const auto compression_method = compress ? Z_DEFLATED : Z_BINARY;
  const auto compression_level = compress ? Z_DEFAULT_COMPRESSION : Z_NO_COMPRESSION;
  // open file in zip in zip64 mode (the last param = 1 enables zip64)
  auto ret = zipOpenNewFileInZip64(zip_handle_, arc_name_with_prefix.c_str(), &file_info, nullptr, 0, nullptr, 0,
                                   nullptr, compression_method, compression_level, 1);
  GE_ASSERT_TRUE(ret == ZIP_OK, "Failed to open zip entry [%s], ret = %d", arc_name_with_prefix.c_str(), ret);
  GE_MAKE_GUARD(close_file_in_zip, [this]() { (void)zipCloseFileInZip(zip_handle_); });

  std::vector<char_t> buffer(kBufSize);
  auto remaining = static_cast<int64_t>(file_size);
  while (remaining > 0) {
    const int64_t chunk = remaining > kBufSize ? kBufSize : remaining;
    fin.read(buffer.data(), chunk);
    auto read_bytes = fin.gcount();
    GE_ASSERT_TRUE(read_bytes == chunk, "Failed to read from file [%s], expected = %zu bytes, bytes_read = %zu bytes",
                   src_file_path.c_str(), chunk, read_bytes);

    ret = zipWriteInFileInZip(zip_handle_, buffer.data(), static_cast<unsigned>(chunk));
    GE_ASSERT_TRUE(ret == ZIP_OK, "zipWriteInFileInZip failed for [%s], ret = %d, bytes_left = %zu bytes",
                   arc_name_with_prefix.c_str(), ret, remaining);
    remaining -= chunk;
  }
  GELOGI("Successfully written [%s] to archive, file_size = %zu bytes", src_file_path.c_str(), file_size);
  files_written_.insert(arc_name_with_prefix);
  return true;
}

bool ZipArchiveWriter::WriteBytes(const std::string &entry_name, const void *data, const size_t data_size,
                                  const bool compress) {
  GE_ASSERT_TRUE(IsMemFileOpened(), "Invalid status of archive [%s]", archive_path_.c_str());
  GE_ASSERT_TRUE(!entry_name.empty(), "Entry name cannot be empty");
  GE_ASSERT_NOTNULL(data, "Pointer data is null, arc_name is [%s]", entry_name.c_str());
  GE_ASSERT_TRUE(data_size > 0, "Data size must be greater than zero, arc_name is [%s]", entry_name.c_str());
  const std::string arc_name_with_prefix = base_name_ + "/" + entry_name;
  if (files_written_.count(arc_name_with_prefix) != 0) {
    // File with the same name has already been written (minizip disallows duplicates)
    GELOGW("Duplicate file entry '%s' detected and ignored", arc_name_with_prefix.c_str());
    return true;
  }

  zip_fileinfo file_info{};
  const auto compression_method = compress ? Z_DEFLATED : Z_BINARY;
  const auto compression_level = compress ? Z_DEFAULT_COMPRESSION : Z_NO_COMPRESSION;
  // open file in zip in zip64 mode (the last param = 1 enables zip64)
  auto ret = zipOpenNewFileInZip64(zip_handle_, arc_name_with_prefix.c_str(), &file_info, nullptr, 0, nullptr, 0,
                                   nullptr, compression_method, compression_level, 1);
  GE_ASSERT_TRUE(ret == ZIP_OK, "Failed to open file [%s] in zip, ret = %d", arc_name_with_prefix.c_str(), ret);
  GE_MAKE_GUARD(close_file_in_zip, [this]() { (void)zipCloseFileInZip(zip_handle_); });

  auto pdata = static_cast<const uint8_t *>(data);
  size_t remaining = data_size;
  while (remaining > 0) {
    const auto chunk = remaining > kMaxWriteSize ? kMaxWriteSize : static_cast<uint32_t>(remaining);

    ret = zipWriteInFileInZip(zip_handle_, pdata, chunk);
    GE_ASSERT_TRUE(ret == ZIP_OK, "zipWriteInFileInZip failed for [%s], ret = %d, bytes_left = %zu bytes",
                   arc_name_with_prefix.c_str(), ret, remaining);

    pdata += chunk;
    remaining -= chunk;
  }

  GELOGI("Successfully written [%s] to archive, data_size = %zu bytes", entry_name.c_str(), data_size);
  files_written_.insert(arc_name_with_prefix);
  return true;
}

bool ZipArchiveWriter::WriteEndOfFile() {
  if (IsMemFileOpened()) {
    GE_MAKE_GUARD(zip_close_guard, [this]() { zip_handle_ = nullptr; });
    return zipClose(zip_handle_, nullptr) == ZIP_OK;
  }
  return true;
}

bool ZipArchiveWriter::SaveModelDataToFile() {
  GE_ASSERT_TRUE(WriteEndOfFile());
  GE_ASSERT_SUCCESS(SaveBinToFile(reinterpret_cast<const char *>(mem_file_.buffer), mem_file_.length, archive_path_));
  return true;
}

bool ZipArchiveWriter::InitArchive() {
  GE_ASSERT_TRUE(!base_name_.empty(), "The base_name_ of is empty");
  // file_funcs 是局部变量，zipOpen2_64内部会拷贝函数指针
  zlib_filefunc64_def file_funcs;
  FillMemFileFuncWithBuffer(&file_funcs, &mem_file_);
  zip_handle_ = zipOpen2_64(archive_path_.c_str(), APPEND_STATUS_CREATE, nullptr, &file_funcs);
  GE_ASSERT_NOTNULL(zip_handle_, "Failed to open archive [%s]", archive_path_.c_str());
  return true;
}

}  // namespace ge
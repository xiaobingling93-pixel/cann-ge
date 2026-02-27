/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/helper/file_saver.h"

#include <securec.h>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <iostream>
#include <array>
#include "common/checker.h"
#include "common/math/math_util.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/screen_printer.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/util.h"
#include "graph/def_types.h"
#include "base/err_msg.h"

namespace ge {
namespace {
constexpr int32_t kFileOpSuccess = 0;
constexpr size_t kMaxErrStrLen = 128U;

static bool CopyModelBuffer(void *dst_addr, const std::size_t dst_len,
                            const void *src_addr, const std::size_t src_len) {
  if ((dst_addr == nullptr) || (src_addr == nullptr)) {
    GELOGE(FAILED, "CopyModelBuffer input param is null.");
    return false;
  }
  GE_ASSERT_TRUE(src_len <= dst_len, "Invalid size as src len:%zu is larger than dst len:%zu.", src_len, dst_len);
  std::size_t remain_size = src_len;
  while (remain_size > SECUREC_MEM_MAX_LEN) {
    if (memcpy_s(dst_addr, SECUREC_MEM_MAX_LEN, src_addr, SECUREC_MEM_MAX_LEN) != EOK) {
      GELOGE(FAILED, "CopyModelBuffer memcpy_s failed.");
      return false;
    }
    remain_size -= SECUREC_MEM_MAX_LEN;
    src_addr = ValueToPtr(PtrToValue(src_addr) + SECUREC_MEM_MAX_LEN);
    dst_addr = ValueToPtr(PtrToValue(dst_addr) + SECUREC_MEM_MAX_LEN);
  }
  if ((remain_size != 0U) && (memcpy_s(dst_addr, remain_size, src_addr, remain_size) != EOK)) {
    GELOGE(FAILED, "CopyModelBuffer memcpy_s remain size failed.");
    return false;
  }
  return true;
}
}  //  namespace

bool FileSaver::host_platform_param_initialized_ = false;
Status FileSaver::OpenFile(int32_t &fd, const std::string &file_path, const bool append) {
  if (CheckPathValid(file_path) != SUCCESS) {
    GELOGE(FAILED, "[Check][FilePath]Check output file failed, file_path:%s.",
           file_path.c_str());
    return FAILED;
  }

  std::array<char_t, MMPA_MAX_PATH> real_path = {};
  GE_IF_BOOL_EXEC(mmRealPath(file_path.c_str(), &real_path[0], MMPA_MAX_PATH) != EN_OK,
                  GELOGI("File %s does not exist, it will be created.", file_path.c_str()));
  // Open file
  constexpr mmMode_t mode = static_cast<mmMode_t>(static_cast<uint32_t>(M_IRUSR) | static_cast<uint32_t>(M_IWUSR));
  uint32_t open_flag = static_cast<uint32_t>(M_RDWR) | static_cast<uint32_t>(M_CREAT);
  if (append) {
    open_flag |= static_cast<uint32_t>(O_APPEND);
  } else {
    open_flag |= static_cast<uint32_t>(O_TRUNC);
  }

  fd = mmOpen2(&real_path[0], static_cast<int32_t>(open_flag), mode);
  if ((fd == EN_INVALID_PARAM) || (fd == EN_ERROR)) {
    // -1: Failed to open file; - 2: Illegal parameter
    std::array<char_t, kMaxErrStrLen + 1U> err_buf = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrStrLen);
    std::string reason = "[Errno " + std::to_string(mmGetErrorCode()) + "] " + err_msg + ".";
    GELOGE(FAILED, "[Open][File]Failed. errno:%d, errmsg:%s", fd, err_msg);
    (void)REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"value", "parameter", "reason"}),
                                    std::vector<const char *>({&real_path[0], "file parameter", reason.c_str()}));
    return FAILED;
  }
  return SUCCESS;
}

Status FileSaver::WriteData(const void * const data, uint64_t size, const int32_t fd) {
  if ((size == 0U) || (data == nullptr)) {
    return PARAM_INVALID;
  }
  int64_t write_count;
  constexpr uint64_t kMaxWriteSize = 1 * 1024 * 1024 * 1024UL; // 1G
  auto seek = PtrToPtr<void, uint8_t>(const_cast<void *>(data));
  while (size > 0U) {
    const uint64_t expect_write_size = std::min(size, kMaxWriteSize);
    write_count = mmWrite(fd, reinterpret_cast<void *>(seek), static_cast<uint32_t>(expect_write_size));
    GE_ASSERT_TRUE(((write_count != EN_INVALID_PARAM) && (write_count != EN_ERROR)),
        "Write data failed, errno: %lld", write_count);
    seek = PtrAdd<uint8_t>(seek, static_cast<size_t>(size), write_count);
    GE_ASSERT_TRUE(size >= static_cast<uint64_t>(write_count),
        "Write data failed, errno: %lld, size: %u", write_count, size);
    size -= write_count;
  }

  return SUCCESS;
}

Status FileSaver::SaveWithFileHeader(const std::string &file_path, const ModelFileHeader &file_header,
                                     const void * const data, const uint64_t len) {
  if ((data == nullptr) || (len == 0)) {
    GELOGE(FAILED, "[Check][Param]Failed, model_data is null or the "
           "length[%" PRIu64 "] is less than 1.", len);
    REPORT_INNER_ERR_MSG("E19999", "Save file failed, model_data is null or the "
                       "length:%" PRIu64 " is less than 1.", len);
    return FAILED;
  }

  // Open file
  int32_t fd = 0;
  if (OpenFile(fd, file_path) != SUCCESS) {
    GELOGE(FAILED, "OpenFile FAILED");
    return FAILED;
  }

  Status ret = SUCCESS;
  do {
    // Write file header
    GE_CHK_BOOL_EXEC(WriteData(static_cast<const void *>(&file_header), sizeof(ModelFileHeader), fd) == SUCCESS,
                     ret = FAILED;
                     break, "WriteData FAILED");
    // write data
    GE_CHK_BOOL_EXEC(WriteData(data, len, fd) == SUCCESS, ret = FAILED, "WriteData FAILED");
  } while (false);
  // Close file
  if (mmClose(fd) != 0) {  // mmClose 0: success
    char_t err_buf[kMaxErrStrLen + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrStrLen);
    GELOGE(FAILED, "[Close][File]Failed, error_code:%u errmsg:%s", ret, err_msg);
    REPORT_INNER_ERR_MSG("E19999", "Close file failed, error_code:%u errmsg:%s",
                       ret, err_msg);
    ret = FAILED;
  }
  return ret;
}

Status FileSaver::SaveWithAlignFill(uint32_t size, uint32_t align_bytes, const int32_t fd) {
  const size_t padding_size = MemSizeAlign(static_cast<size_t>(size), align_bytes) - static_cast<size_t>(size);
  if (padding_size > 0U) {
    GELOGI("%u bytes need to be padded for alignment, raw size:%u, align bytes:%u", padding_size, size, align_bytes);
    auto buff = ge::MakeUnique<uint8_t[]>(padding_size);
    const errno_t err = memset_s(buff.get(), padding_size, 0, padding_size);
    GE_ASSERT_EOK(err, "memset_s err, error_code %d", err);
    GE_ASSERT_TRUE(WriteData(static_cast<const void *>(buff.get()), padding_size, fd) == SUCCESS, "write data failed");
  }
  return SUCCESS;
}

Status FileSaver::SaveWithFileHeader(const std::string &file_path, const ModelFileHeader &file_header,
                                     const ModelPartitionTable &model_partition_table,
                                     const std::vector<ModelPartition> &partition_datas) {
  GE_CHK_BOOL_RET_STATUS((!partition_datas.empty()) && (model_partition_table.num != 0U)
      && (model_partition_table.num == partition_datas.size()), FAILED,
      "Invalid param:partition data size is (%zu), model_partition_table.num is (%u).",
      partition_datas.size(),
      model_partition_table.num);
  // Open file
  int32_t fd = 0;
  if (OpenFile(fd, file_path) != SUCCESS) {
    return FAILED;
  }
  Status ret = SUCCESS;
  do {
    // Write file header
    if (WriteData(static_cast<const void *>(&file_header), sizeof(ModelFileHeader), fd) != SUCCESS) {
      ret = FAILED;
      break;
    }

    // Write model partition table
    const uint64_t table_size = SizeOfModelPartitionTable(model_partition_table);
    if (WriteData(static_cast<const void *>(&model_partition_table), table_size, fd) != SUCCESS) {
      ret = FAILED;
      break;
    }

    // Write partition data
    for (const auto &partitionData : partition_datas) {
      GELOGI("GC:size[%zu]", partitionData.size);
      if (WriteData(static_cast<const void *>(partitionData.data), partitionData.size, fd) != SUCCESS) {
        ret = FAILED;
        break;
      }
    }
  } while (false);
  // Close file
  if (mmClose(fd) != EN_OK) {
    char_t err_buf[kMaxErrStrLen + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrStrLen);
    REPORT_INNER_ERR_MSG("E19999", "Close file failed, error_code:%u errmsg:%s", ret, err_msg);
    ret = FAILED;
  }
  return ret;
}

Status FileSaver::SaveToBuffWithFileHeader(const ModelFileHeader &file_header,
                                           ModelPartitionTable &model_partition_table,
                                           const std::vector<ModelPartition> &partition_datas,
                                           ge::ModelBufferData &model) {
  const std::vector<ModelPartitionTable *> model_partition_tables = { &model_partition_table };
  const std::vector<std::vector<ModelPartition>> all_partition_datas = { partition_datas };
  return SaveToBuffWithFileHeader(file_header, model_partition_tables, all_partition_datas, model);
}

Status FileSaver::SaveToBuffWithFileHeader(const ModelFileHeader &file_header,
                                           const std::vector<ModelPartitionTable *> &model_partition_tables,
                                           const std::vector<std::vector<ModelPartition>> &all_partition_datas,
                                           ge::ModelBufferData &model) {
  GE_CHK_BOOL_RET_STATUS(model_partition_tables.size() == all_partition_datas.size(),
                         PARAM_INVALID,
                         "Model table size %zu does not match partition size %zu.",
                         model_partition_tables.size(), all_partition_datas.size());
  for (size_t index = 0U; index < model_partition_tables.size(); ++index) {
    auto &cur_partiton_data = all_partition_datas[index];
    auto &cur_model_partition_table = *model_partition_tables[index];
    GE_CHK_BOOL_RET_STATUS((!cur_partiton_data.empty()) && (cur_model_partition_table.num != 0U)
                           && (cur_model_partition_table.num == cur_partiton_data.size()), FAILED,
                           "Invalid param: partition data size is (%zu), model_partition_table.num is (%u).",
                           cur_partiton_data.size(), cur_model_partition_table.num);
  }

  constexpr uint64_t model_header_size = sizeof(ModelFileHeader);
  uint64_t total_size = model_header_size;
  for (size_t index = 0U; index < model_partition_tables.size(); ++index) {
    auto &cur_model_partition_table = *model_partition_tables[index];
    FMK_UINT64_ADDCHECK(total_size, SizeOfModelPartitionTable(cur_model_partition_table));
    total_size += SizeOfModelPartitionTable(cur_model_partition_table);
    auto &cur_partition_data = all_partition_datas[index];
    for (const auto &partition_data : cur_partition_data) {
      const auto ret = ge::CheckUint64AddOverflow(total_size, static_cast<uint64_t>(partition_data.size));
      GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED, "Add uint64 overflow!");
      total_size += static_cast<uint64_t>(partition_data.size);
    }
  }
  // save to buff
  auto buff_data = MakeUnique<uint8_t[]>(total_size);
  GE_CHK_BOOL_RET_STATUS(buff_data != nullptr, FAILED, "Malloc failed!");
  GE_PRINT_DYNAMIC_MEMORY(malloc, "File buffer.", total_size);
  model.data.reset(buff_data.release(), std::default_delete<uint8_t[]>());
  model.length = total_size;
  uint64_t left_space = total_size;
  uint8_t *buff = model.data.get();
  auto ret_mem = CopyModelBuffer(buff,
                                 left_space,
                                 static_cast<void *>(const_cast<ModelFileHeader *>(&file_header)),
                                 model_header_size);
  GE_CHK_BOOL_RET_STATUS(ret_mem, FAILED, "CopyModelBuffer failed!");
  buff += model_header_size;
  left_space -= model_header_size;

  for (size_t index = 0U; index < model_partition_tables.size(); ++index) {
    auto &cur_tabel = *model_partition_tables[index];
    const uint64_t table_size = SizeOfModelPartitionTable(cur_tabel);
    ret_mem = CopyModelBuffer(buff, left_space, reinterpret_cast<void *>(&cur_tabel), table_size);
    GE_CHK_BOOL_RET_STATUS(ret_mem, FAILED, "CopyModelBuffer failed!");
    buff += table_size;
    left_space -= table_size;
    auto &cur_partition_data = all_partition_datas[index];
    for (const auto &partition_data : cur_partition_data) {
      ret_mem = CopyModelBuffer(buff,
                                left_space,
                                static_cast<void *>(const_cast<uint8_t *>(partition_data.data)),
                                static_cast<uint64_t>(partition_data.size));
      GE_CHK_BOOL_RET_STATUS(ret_mem, FAILED, "CopyModelBuffer failed!");
      buff += partition_data.size;
      left_space -= partition_data.size;
    }
  }

  return SUCCESS;
}

Status FileSaver::CheckPathValid(const std::string &file_path) {
  // Determine file path length
  if (file_path.size() >= static_cast<size_t>(MMPA_MAX_PATH)) {
    GELOGE(FAILED, "[Check][FilePath]Failed, file path's length:%zu >= mmpa_max_path:%d",
           file_path.size(), MMPA_MAX_PATH);
    std::string max_path_str = std::to_string(MMPA_MAX_PATH);
    (void)REPORT_PREDEFINED_ERR_MSG(
        "E13002", 
        std::vector<const char *>({"filepath", "size"}),
        std::vector<const char *>({file_path.c_str(), max_path_str.c_str()})
    );
    return FAILED;
  }

  // Find the last separator
  int32_t path_split_pos = static_cast<int32_t>(file_path.size() - 1U);
  for (; path_split_pos >= 0; path_split_pos--) {
    if ((file_path[static_cast<size_t>(path_split_pos)] == '\\') ||
        (file_path[static_cast<size_t>(path_split_pos)] == '/')) {
      break;
    }
  }

  if (path_split_pos == 0) {
    return SUCCESS;
  }

  // If there is a path before the file name, create the path
  if (path_split_pos != -1) {
    if (CreateDirectory(std::string(file_path).substr(0U, static_cast<size_t>(path_split_pos))) != kFileOpSuccess) {
      GELOGE(FAILED, "[Create][Directory]Failed, file path:%s.", file_path.c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

Status FileSaver::SaveToFile(const std::string &file_path, const ge::ModelData &model,
                             const ModelFileHeader *const model_file_header) {
  if (file_path.empty()) {
    GELOGE(FAILED, "[Save][File]Incorrect input param, file_path is empty");
    (void)REPORT_PREDEFINED_ERR_MSG(
          "E10059", std::vector<const char *>({"stage", "reason"}),
          std::vector<const char *>({"SaveToFile", "Input parameter file_path is empty"}));
    return FAILED;
  }

  if ((model.model_data == nullptr) || (model.model_len == 0U)) {
    GELOGE(FAILED, "[Save][File]Incorrect input param, model_data is nullptr or model_len is 0");
    REPORT_INNER_ERR_MSG("E19999", "Save file failed, at least one of the "
                       "input parameters(model_data, model_len) is incorrect");
    return FAILED;
  }

  ModelFileHeader file_header;

  bool copy_header_ret = false;
  GE_IF_BOOL_EXEC(model_file_header != nullptr, copy_header_ret =
                  CopyModelBuffer(&file_header, sizeof(ModelFileHeader), model_file_header, sizeof(ModelFileHeader)));
  GE_CHK_BOOL_RET_STATUS(copy_header_ret, FAILED, "Copy ModelFileHeader failed, CopyModelBuffer return: %d",
                         static_cast<int32_t>(copy_header_ret));

  file_header.model_length = model.model_len;

  const Status ret = SaveWithFileHeader(file_path, file_header, model.model_data, model.model_len);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Save][File]Failed, file_path:%s, file_header_len:%lu, error_code:%u.",
           file_path.c_str(), file_header.model_length, ret);
    return FAILED;
  }

  return SUCCESS;
}

Status FileSaver::SaveToFile(const std::string &file_path, const ModelFileHeader &model_file_header,
                             const ModelPartitionTable &model_partition_table,
                             const std::vector<ModelPartition> &partition_datas) {
  const Status ret = SaveWithFileHeader(file_path, model_file_header, model_partition_table, partition_datas);
  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED, "save file failed, file_path:%s, file header len:%" PRIu64 ".",
                         file_path.c_str(), model_file_header.model_length);
  return SUCCESS;
}

Status FileSaver::SaveToFile(const std::string &file_path, const ModelFileHeader &file_header,
                             const std::vector<ModelPartitionTable *> &model_partition_tables,
                             const std::vector<std::vector<ModelPartition>> &all_partition_datas,
                             const bool is_partition_align,
                             const uint32_t align_bytes) {
  const Status ret = SaveWithFileHeader(
      file_path, file_header, model_partition_tables, all_partition_datas, is_partition_align, align_bytes);
  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED, "save file failed, file_path:%s, file header len:%" PRIu64 ".",
                         file_path.c_str(), file_header.model_length);
  if (file_header.need_check_os_cpu_info == static_cast<uint8_t>(OsCpuInfoCheckTyep::NO_CHECK)) {
    FileSaver::PrintModelSaveLog();
  }
  return SUCCESS;
}

Status FileSaver::SaveWithFileHeader(const std::string &file_path, const ModelFileHeader &file_header,
                                     const std::vector<ModelPartitionTable *> &model_partition_tables,
                                     const std::vector<std::vector<ModelPartition>> &all_partition_datas,
                                     const bool is_partition_align,
                                     const uint32_t align_bytes) {
  GE_CHK_BOOL_EXEC(model_partition_tables.size() == all_partition_datas.size(),
                   return PARAM_INVALID,
                   "model table size %zu does not match partition size %zu",
                   model_partition_tables.size(), all_partition_datas.size());
  for (size_t index = 0U; index < model_partition_tables.size(); ++index) {
    auto &cur_partiton_data = all_partition_datas[index];
    auto &cur_model_partition_table = *model_partition_tables[index];
    GE_CHK_BOOL_RET_STATUS((!cur_partiton_data.empty()) && (cur_model_partition_table.num != 0U)
                           && (cur_model_partition_table.num == cur_partiton_data.size()), FAILED,
                           "Invalid param:partition data size is (%zu), model_partition_table.num is (%u).",
                           cur_partiton_data.size(), cur_model_partition_table.num);
  }

  // Open file
  int32_t fd = 0;
  Status ret = OpenFile(fd, file_path);
  if (ret != SUCCESS) {
    return FAILED;
  }

  do {
    // Write file header
    if (WriteData(static_cast<const void *>(&file_header), sizeof(ModelFileHeader), fd) != SUCCESS) {
      ret = FAILED;
      break;
    }
    for (size_t index = 0U; index < model_partition_tables.size(); ++index) {
      // Write model partition table
      auto &cur_table = *model_partition_tables[index];
      const uint64_t table_size = SizeOfModelPartitionTable(cur_table);
      GELOGI("table_size[%u]", table_size);
      for (const auto& part : cur_table.partition) {
        GELOGI("partition type:%u, offset:%u, size:%u", part.type,
                                                        part.mem_offset,
                                                        part.mem_size);
      }

      if (WriteData(static_cast<const void *>(&cur_table), table_size, fd) != SUCCESS ||
          (is_partition_align && (SaveWithAlignFill(table_size, align_bytes, fd) != SUCCESS))) {
        ret = FAILED;
        break;
      }

      // Write partition data
      auto &cur_partition_datas = all_partition_datas[index];
      for (const auto &partition_data : cur_partition_datas) {
        GELOGI("part_size[%zu]", partition_data.size);
        if (WriteData(static_cast<const void *>(partition_data.data), partition_data.size, fd) != SUCCESS ||
            (is_partition_align && (SaveWithAlignFill(partition_data.size, align_bytes, fd) != SUCCESS))) {
          ret = FAILED;
          break;
        }
      }
    }
  } while (false);
  // Close file
  if (mmClose(fd) != 0) {  // mmClose 0: success
    std::array<char_t, kMaxErrStrLen + 1U> err_buf = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrStrLen);
    GELOGE(FAILED, "[Close][File]Failed, error_code:%u errmsg:%s", ret, err_msg);
    REPORT_INNER_ERR_MSG("E19999", "Close file failed, error_code:%u errmsg:%s",
                      ret, err_msg);
    ret = FAILED;
  }
  return ret;
}

Status FileSaver::SaveToFile(const std::string &file_path, const void *const data, const uint64_t len,
                             const bool append) {
  if ((data == nullptr) || (len <= 0)) {
    GELOGE(FAILED, "[Check][Param]Failed, model_data is null or the "
           "length[%lu] is less than 1.", len);
    REPORT_INNER_ERR_MSG("E19999", "Save file failed, the model_data is null or "
                       "its length:%" PRIu64 " is less than 1.", len);
    return FAILED;
  }

  // Open file
  int32_t fd = 0;
  if (OpenFile(fd, file_path, append) != SUCCESS) {
    GELOGE(FAILED, "OpenFile FAILED");
    return FAILED;
  }

  Status ret = SUCCESS;

  // write data
  GE_CHK_BOOL_EXEC(WriteData(data, len, fd) == SUCCESS, ret = FAILED, "WriteData FAILED");

  // Close file
  if (mmClose(fd) != 0) {  // mmClose 0: success
    std::array<char_t, kMaxErrStrLen + 1U> err_buf = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrStrLen);
    GELOGE(FAILED, "[Close][File]Failed, error_code:%u errmsg:%s", ret, err_msg);
    REPORT_INNER_ERR_MSG("E19999", "Close file failed, error_code:%u errmsg:%s",
                      ret, err_msg);
    ret = FAILED;
  }
  return ret;
}

void FileSaver::PrintModelSaveLog() {
  if (!host_platform_param_initialized_) {
    return;
  }
  SCREEN_LOG("This model is irrelevant to the host platform, parameters about host os and host cpu are ignored.");
  return;
}
}  //  namespace ge

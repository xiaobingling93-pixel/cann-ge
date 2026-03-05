/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <cstring>
#include "zip_mem_ioapi.h"
#include "mmpa/mmpa_api.h"
#include "common/math/math_util.h"
#include "graph_metadef/graph/utils/math_util.h"
#include "common/debug/ge_log.h"

static voidpf ZCALLBACK MemOpenFileFuncWithBuffer(voidpf opaque, const void *filename, int mode);
static uLong ZCALLBACK MemReadFileFunc(voidpf opaque, voidpf stream, void *buf, uLong size);
static uLong ZCALLBACK MemWriteFileFunc(voidpf opaque, voidpf stream, const void *buf, uLong size);
static ZPOS64_T ZCALLBACK MemTell64FileFunc(voidpf opaque, voidpf stream);
static long ZCALLBACK MemSeek64FileFunc(voidpf opaque, voidpf stream, ZPOS64_T offset, int origin);
static int ZCALLBACK MemCloseFileFunc(voidpf opaque, voidpf stream);
static int ZCALLBACK MemErrorFileFunc(voidpf opaque, voidpf stream);

static voidpf ZCALLBACK MemOpenFileFuncReadonly(voidpf opaque, const void *filename, int mode);
static uLong ZCALLBACK MemReadFileFuncReadonly(voidpf opaque, voidpf stream, void *buf, uLong size);
static uLong ZCALLBACK MemWriteFileFuncReadonly(voidpf opaque, voidpf stream, const void *buf, uLong size);
static ZPOS64_T ZCALLBACK MemTell64FileFuncReadonly(voidpf opaque, voidpf stream);
static long ZCALLBACK MemSeek64FileFuncReadonly(voidpf opaque, voidpf stream, ZPOS64_T offset, int origin);
static int ZCALLBACK MemCloseFileFuncReadonly(voidpf opaque, voidpf stream);
static int ZCALLBACK MemErrorFileFuncReadonly(voidpf opaque, voidpf stream);

constexpr uint64_t kMemInitialCapacity = 64 * 1024;
constexpr int kMemGrowFactor = 2;

static int MemGrow(MemoryFile *mem_file, const uint64_t new_size) {
  if (new_size <= mem_file->capacity) {
    return MEM_ZIP_OK;
  }

  uint64_t new_capacity = mem_file->capacity;
  if (new_capacity == 0) {
    new_capacity = kMemInitialCapacity;
  }

  while (new_capacity < new_size) {
    if (ge::CheckUint64MulOverflow(new_capacity, kMemGrowFactor) != ge::SUCCESS) {
      GELOGE(ge::FAILED,
             "[MEMZIP] Memory file capacity overflow: current_capacity[%lu bytes], grow_factor[%d], target_size[%zu "
             "bytes]",
             new_capacity, kMemGrowFactor, new_size);
      return MEM_ZIP_ERROR;
    }
    new_capacity *= kMemGrowFactor;
  }

  const auto new_buffer = static_cast<uint8_t *>(std::malloc(new_capacity));
  if (new_buffer == nullptr) {
    GELOGE(ge::FAILED,
           "[MEMZIP] Failed to allocate memory: target_capacity[%lu bytes], error_msg[%s]",
           new_capacity, strerror(errno));
    return MEM_ZIP_ERROR;
  }

  if ((mem_file->buffer != nullptr) && (mem_file->length > 0)) {
    const auto ret = ge::GeMemcpy(new_buffer, new_capacity, mem_file->buffer, mem_file->length);
    if (ret != ge::SUCCESS) {
      GELOGE(ge::FAILED,
             "[MEMZIP] Failed to copy memory: dest_ptr[%p], dest_max[%lu], src_ptr[%p], src_size[%lu], ret=%d",
             new_buffer, new_capacity, mem_file->buffer, mem_file->capacity, ret);
      std::free(new_buffer);
      return MEM_ZIP_ERROR;
    }
  }

  std::free(mem_file->buffer);
  mem_file->buffer = new_buffer;
  mem_file->capacity = new_capacity;
  return MEM_ZIP_OK;
}

static voidpf ZCALLBACK MemOpenFileFuncWithBuffer(voidpf opaque, const void *filename, int mode) {
  (void)filename;

  const auto mem_file = static_cast<MemoryFile *>(opaque);
  if (mem_file == nullptr) {
    return nullptr;
  }

  mem_file->buffer = nullptr;
  mem_file->length = 0;
  mem_file->capacity = 0;
  mem_file->position = 0;
  mem_file->error = MEM_ZIP_OK;
  mem_file->grow_mode = 1;
  mem_file->release_from_outside = 1;

  if (mode & ZLIB_FILEFUNC_MODE_CREATE) {
    if (MemGrow(mem_file, kMemInitialCapacity) != MEM_ZIP_OK) {
      GELOGE(ge::FAILED, "[MEMZIP] Failed to allocate initial capacity[%zu bytes]", kMemInitialCapacity);
      return nullptr;
    }
  }

  return mem_file;
}

static voidpf ZCALLBACK MemOpenFileFuncReadonly(voidpf opaque, const void *filename, int mode) {
  (void)filename;
  (void)mode;

  const auto mem_file_readonly = static_cast<MemoryFileReadonly *>(opaque);
  if (mem_file_readonly == nullptr) {
    GELOGE(ge::FAILED, "[MEMZIP] Opaque pointer is null. Cannot initialize memory file.");
    return nullptr;
  }

  mem_file_readonly->position = 0;

  return mem_file_readonly;
}

static uLong ZCALLBACK MemReadFileFunc(voidpf opaque, voidpf stream, void *buf, uLong size) {
  const auto mem_file = static_cast<MemoryFile *>(stream);
  (void)opaque;

  if (mem_file == nullptr || mem_file->error != MEM_ZIP_OK) {
    return 0;
  }

  uLong bytes_to_read = size;
  if (mem_file->position + bytes_to_read > mem_file->length) {
    bytes_to_read = mem_file->length - mem_file->position;
  }

  if (bytes_to_read > 0) {
    const auto ret =
        ge::GeMemcpy(static_cast<uint8_t *>(buf), size, mem_file->buffer + mem_file->position, bytes_to_read);
    if (ret != ge::SUCCESS) {
      GELOGE(ge::FAILED,
             "[MEMZIP] Failed to copy, ret=%d: dest_ptr[%p], dest_max[%zu], src_base_ptr[%p], src_position[%zu], "
             "src_size[%zu]",
             ret, buf, size, mem_file->buffer, mem_file->position, bytes_to_read);
      return 0;
    }
    mem_file->position += bytes_to_read;
  }

  return bytes_to_read;
}

static uLong ZCALLBACK MemReadFileFuncReadonly(voidpf opaque, voidpf stream, void *buf, uLong size) {
  const auto mem_file = static_cast<MemoryFileReadonly *>(stream);
  (void)opaque;

  if (mem_file == nullptr) {
    return 0;
  }

  uLong bytes_to_read = size;
  if (mem_file->position + bytes_to_read > mem_file->length) {
    bytes_to_read = mem_file->length - mem_file->position;
  }

  if (bytes_to_read > 0) {
    const auto ret =
        ge::GeMemcpy(static_cast<uint8_t *>(buf), size, mem_file->buffer + mem_file->position, bytes_to_read);
    if (ret != ge::SUCCESS) {
      GELOGE(ge::FAILED,
             "[MEMZIP] Failed to copy, ret=%d: dest_ptr[%p], dest_max[%zu], src_base_ptr[%p], src_position[%zu], "
             "src_size[%zu]",
             ret, buf, size, mem_file->buffer, mem_file->position, bytes_to_read);
      return 0;
    }
    mem_file->position += bytes_to_read;
  }

  return bytes_to_read;
}

static uLong ZCALLBACK MemWriteFileFunc(voidpf opaque, voidpf stream, const void *buf, uLong size) {
  const auto mem_file = static_cast<MemoryFile *>(stream);
  (void)opaque;

  if (mem_file == nullptr || mem_file->error != MEM_ZIP_OK) {
    GELOGE(ge::FAILED, "[MEMZIP] Opaque pointer is null or file is invalid");
    return 0;
  }

  if (size == 0) {
    return 0;
  }

  uint64_t new_size = mem_file->position + size;
  if (new_size > mem_file->capacity) {
    if (MemGrow(mem_file, new_size) != MEM_ZIP_OK) {
      GELOGE(ge::FAILED, "[MEMZIP] Failed to expand memory capacity[%zu bytes]", new_size);
      mem_file->error = MEM_ZIP_ERROR;
      return 0;
    }
  }
  // 这里需要注意下destMax不能超2G
  const auto ret = memcpy_s(mem_file->buffer + mem_file->position, size, buf, size);
  if (ret != EOK) {
    GELOGE(ge::FAILED,
           "[MEMZIP] Failed to copy, ret=%d: dest_base_ptr[%p], dest_position[%zu], dest_max[%zu], src_ptr[%p], "
           "src_size[%zu]",
           ret, mem_file->buffer, mem_file->position, mem_file->capacity - mem_file->position, buf, size);
    return 0;
  }

  mem_file->position += size;

  if (mem_file->position > mem_file->length) {
    mem_file->length = mem_file->position;
  }

  return size;
}

static uLong ZCALLBACK MemWriteFileFuncReadonly(voidpf opaque, voidpf stream, const void *buf, uLong size) {
  (void)opaque;
  (void)stream;
  (void)buf;
  (void)size;
  return 0;
}

static ZPOS64_T ZCALLBACK MemTell64FileFunc(voidpf opaque, voidpf stream) {
  const auto mem_file = static_cast<MemoryFile *>(stream);
  (void)opaque;

  if (mem_file == nullptr) {
    return static_cast<ZPOS64_T>(-1);
  }

  return mem_file->position;
}

static ZPOS64_T ZCALLBACK MemTell64FileFuncReadonly(voidpf opaque, voidpf stream) {
  const auto mem_file = static_cast<MemoryFileReadonly *>(stream);
  (void)opaque;

  if (mem_file == nullptr) {
    return static_cast<ZPOS64_T>(-1);
  }

  return mem_file->position;
}

static long ZCALLBACK MemSeek64FileFunc(voidpf opaque, voidpf stream, ZPOS64_T offset, int origin) {
  const auto mem_file = static_cast<MemoryFile *>(stream);
  uint64_t new_position;
  (void)opaque;

  if (mem_file == nullptr) {
    GELOGE(ge::FAILED, "[MEMZIP] Get invalid memory file");
    return MEM_ZIP_ERROR;
  }

  switch (origin) {
    case ZLIB_FILEFUNC_SEEK_CUR:
      new_position = mem_file->position + offset;
      break;
    case ZLIB_FILEFUNC_SEEK_END:
      new_position = mem_file->length + offset;
      break;
    case ZLIB_FILEFUNC_SEEK_SET:
      new_position = offset;
      break;
    default:
      return MEM_ZIP_ERROR;
  }

  if (new_position > mem_file->length) {
    GELOGE(ge::FAILED, "[MEMZIP] Failed to seek, expected new position=%zu, file length=%zu", new_position,
           mem_file->length);
    return MEM_ZIP_ERROR;
  }

  mem_file->position = new_position;
  return MEM_ZIP_OK;
}

static long ZCALLBACK MemSeek64FileFuncReadonly(voidpf opaque, voidpf stream, ZPOS64_T offset, int origin) {
  const auto mem_file = static_cast<MemoryFileReadonly *>(stream);
  uint64_t new_position;
  (void)opaque;

  if (mem_file == nullptr) {
    return MEM_ZIP_ERROR;
  }

  switch (origin) {
    case ZLIB_FILEFUNC_SEEK_CUR:
      new_position = mem_file->position + offset;
      break;
    case ZLIB_FILEFUNC_SEEK_END:
      new_position = mem_file->length + offset;
      break;
    case ZLIB_FILEFUNC_SEEK_SET:
      new_position = offset;
      break;
    default:
      return MEM_ZIP_ERROR;
  }

  if (new_position > mem_file->length) {
    return MEM_ZIP_ERROR;
  }

  mem_file->position = new_position;
  return MEM_ZIP_OK;
}

static int ZCALLBACK MemCloseFileFunc(voidpf opaque, voidpf stream) {
  const auto mem_file = static_cast<MemoryFile *>(stream);
  (void)opaque;

  if (mem_file == nullptr) {
    return MEM_ZIP_ERROR;
  }

  if ((mem_file->buffer != nullptr) && (mem_file->release_from_outside == 0)) {
    // 默认buffer内存由用户释放，释放后重置mem_file
    std::free(mem_file->buffer);
    *mem_file = {};
  }

  return MEM_ZIP_OK;
}

static int ZCALLBACK MemCloseFileFuncReadonly(voidpf opaque, voidpf stream) {
  (void)opaque;
  (void)stream;
  return MEM_ZIP_OK;
}

static int ZCALLBACK MemErrorFileFunc(voidpf opaque, voidpf stream) {
  const auto mem_file = static_cast<MemoryFile *>(stream);
  (void)opaque;

  if (mem_file == nullptr) {
    return MEM_ZIP_ERROR;
  }

  return mem_file->error;
}

static int ZCALLBACK MemErrorFileFuncReadonly(voidpf opaque, voidpf stream) {
  const auto mem_file = static_cast<MemoryFileReadonly *>(stream);
  (void)opaque;

  if (mem_file == nullptr) {
    return MEM_ZIP_ERROR;
  }

  return MEM_ZIP_OK;
}

void FillMemFileFuncWithBuffer(zlib_filefunc64_def *file_func_def, MemoryFile *mem_file) {
  file_func_def->zopen64_file = MemOpenFileFuncWithBuffer;
  file_func_def->zread_file = MemReadFileFunc;
  file_func_def->zwrite_file = MemWriteFileFunc;
  file_func_def->ztell64_file = MemTell64FileFunc;
  file_func_def->zseek64_file = MemSeek64FileFunc;
  file_func_def->zclose_file = MemCloseFileFunc;
  file_func_def->zerror_file = MemErrorFileFunc;
  file_func_def->opaque = mem_file;
}

void FillMemFileFuncReadonly(zlib_filefunc64_def *file_func_def, MemoryFileReadonly *mem_file) {
  file_func_def->zopen64_file = MemOpenFileFuncReadonly;
  file_func_def->zread_file = MemReadFileFuncReadonly;
  file_func_def->zwrite_file = MemWriteFileFuncReadonly;
  file_func_def->ztell64_file = MemTell64FileFuncReadonly;
  file_func_def->zseek64_file = MemSeek64FileFuncReadonly;
  file_func_def->zclose_file = MemCloseFileFuncReadonly;
  file_func_def->zerror_file = MemErrorFileFuncReadonly;
  file_func_def->opaque = mem_file;
}

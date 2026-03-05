/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_COMMON_HELPER_ZIP_MEM_IOAPI_H
#define BASE_COMMON_HELPER_ZIP_MEM_IOAPI_H

#include <cstdint>
#include "minizip/ioapi.h"

#define MEM_ZIP_OK (0)
#define MEM_ZIP_ERROR (-1)

struct MemoryFile {
  uint8_t *buffer;         // 可写缓冲区
  uint64_t length;         // 可写缓冲区内容实际长度
  uint64_t capacity;       // 可写缓冲区的容量，即最多可写入的字节数
  uint64_t position;       // 当前位置
  int error;               // 错误标记
  int grow_mode;           // 缓冲区是否可扩容
  int release_from_outside; // 是否由外部释放buffer, 0表示zipClose时释放buffer, 1表示内存由外部释放
};

struct MemoryFileReadonly {
  const uint8_t *buffer;  // 只读缓冲区, 内存由外部管理
  uint64_t length;        // 只读缓冲区内容实际长度
  uint64_t position;      // 当前位置
};

void FillMemFileFuncWithBuffer(zlib_filefunc64_def *file_func_def, MemoryFile *mem_file);
void FillMemFileFuncReadonly(zlib_filefunc64_def *file_func_def, MemoryFileReadonly *mem_file);
#endif  // BASE_COMMON_HELPER_ZIP_MEM_IOAPI_H

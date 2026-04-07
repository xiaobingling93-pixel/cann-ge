/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dlog_pub.h"
#include <stdio.h>
#include <fstream>
#include <string.h>
#include <stdarg.h>
#include <map>

#define MSG_LENGTH_STUB   (1024)
#define SET_MOUDLE_ID_MAP_NAME(x) { #x, x}

#ifdef __cplusplus
#ifndef LOG_CPP
extern "C" {
#endif
#endif // __cplusplus

#ifndef LINUX
#define LINUX 0
#endif

#ifndef OS_TYPE
#define OS_TYPE 0
#endif

#if (OS_TYPE == LINUX)
#define PATH_SLOG "/usr/slog/slog"
#else
#define PATH_SLOG "C:\\Program Files\\Huawei\\HiAI Foundation\\log"
#endif

typedef struct _dcode_stub {
  const char *c_name;
  int c_val;
} DCODE_STUB;

int CheckLogLevel(int moduleId, int level)
{
    return level >= DLOG_ERROR;
}

const std::map<int, std::string> LOG_LEVEL_STR_MAP {
        {DLOG_DEBUG, "DEBUG"},
        {DLOG_INFO, "INFO"},
        {DLOG_WARN, "WARN"},
        {DLOG_ERROR, "ERROR"},
};

void DlogRecord(int moduleId, int level, const char *fmt, ...) {
  if(moduleId < 0 || moduleId >= INVLID_MOUDLE_ID){
      return;
  }
  if (!CheckLogLevel(moduleId, level)) {
      return;
  }
  int len;
  char msg[MSG_LENGTH_STUB] = {0};
  std::string level_str = "NULL";
  auto iter = LOG_LEVEL_STR_MAP.find(level);
  if (iter != LOG_LEVEL_STR_MAP.end()) {
    level_str = iter->second;
  }
  snprintf(msg, MSG_LENGTH_STUB,"[FE] [%s] ", level_str.c_str());
  va_list ap;

  va_start(ap,fmt);
  len = strlen(msg);
  vsnprintf(msg + len, MSG_LENGTH_STUB - len, fmt, ap);
  va_end(ap);

  len = strlen(msg);
  if (len < MSG_LENGTH_STUB - 1 && msg[len - 1] != '\n') {
    msg[len] = '\n';
    msg[len + 1] = '\0';
  }
  printf("%s",msg);
  return;
}
#ifdef __cplusplus
#ifndef LOG_CPP
}
#endif
#endif // __cplusplus
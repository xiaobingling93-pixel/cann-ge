/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "plog.h"
#include "slog_stub.h"

void dav_log(int module_id, const char *fmt, ...) {}

void DlogRecord(int moduleId, int level, const char *fmt, ...) {
  va_list valist;
  va_start(valist, fmt);
  if (moduleId & RUN_LOG_MASK) {
    ge::SlogStub::GetInstance()->Log(moduleId & (~RUN_LOG_MASK), DLOG_INFO, fmt, valist);
  } else {
    ge::SlogStub::GetInstance()->Log(moduleId, level, fmt, valist);
  }
  va_end(valist);
}

void DlogErrorInner(int module_id, const char *fmt, ...) {
  va_list valist;
  va_start(valist, fmt);
  ge::SlogStub::GetInstance()->Log(module_id, DLOG_ERROR, fmt, valist);
  va_end(valist);
}

void DlogWarnInner(int module_id, const char *fmt, ...) {
  va_list valist;
  va_start(valist, fmt);
  ge::SlogStub::GetInstance()->Log(module_id, DLOG_WARN, fmt, valist);
  va_end(valist);
}

void DlogInfoInner(int module_id, const char *fmt, ...) {
  va_list valist;
  va_start(valist, fmt);
  if (module_id & RUN_LOG_MASK) {
    ge::SlogStub::GetInstance()->Log(module_id & (~RUN_LOG_MASK), DLOG_INFO, fmt, valist);
  } else {
    ge::SlogStub::GetInstance()->Log(module_id, DLOG_INFO, fmt, valist);
  }
  va_end(valist);
}

void DlogDebugInner(int module_id, const char *fmt, ...) {
  va_list valist;
  va_start(valist, fmt);
  ge::SlogStub::GetInstance()->Log(module_id, DLOG_DEBUG, fmt, valist);
  va_end(valist);
}

void DlogEventInner(int module_id, const char *fmt, ...) {
  va_list valist;
  va_start(valist, fmt);
  ge::SlogStub::GetInstance()->Log((module_id & (~RUN_LOG_MASK)), DLOG_INFO, fmt, valist);
  va_end(valist);
}

void DlogInner(int module_id, int level, const char *fmt, ...) {
  dav_log(module_id, fmt);
}

int dlog_setlevel(int module_id, int level, int enable_event) {
  auto log_level = getenv("ASCEND_GLOBAL_LOG_LEVEL");
  // 设置环境变量时忽略用例代码里的设置
  if (log_level == nullptr) {
    ge::SlogStub::GetInstance()->SetLevel(level);
    ge::SlogStub::GetInstance()->SetEventLevel(enable_event);
  }
  return 0;
}

int dlog_getlevel(int module_id, int *enable_event) {
  return ge::SlogStub::GetInstance()->GetLevel();
}

int CheckLogLevel(int moduleId, int log_level_check) {
  if (moduleId & RUN_LOG_MASK) {
    return 1;
  }
  if (moduleId == GE) {
    return log_level_check >= ge::SlogStub::GetInstance()->GetGeLevel();
  }
  return log_level_check >= dlog_getlevel(moduleId, nullptr);
}

/**
* @ingroup plog
* @brief DlogReportInitialize: init log in service process before all device setting.
* @return: 0: SUCCEED, others: FAILED
*/
int DlogReportInitialize() {
  return 0;
}

/**
* @ingroup plog
* @brief DlogReportFinalize: release log resource in service process after all device reset.
* @return: 0: SUCCEED, others: FAILED
*/
int DlogReportFinalize() {
  return 0;
}

int DlogSetAttr(LogAttr logAttr) {
  return 0;
}

void DlogReportStop(int devId) {}

int DlogReportStart(int devId, int mode) {
  return 0;
}
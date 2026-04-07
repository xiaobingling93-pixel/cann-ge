/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "trace/atrace_pub.h"
#include <map>
#include <iostream>

namespace {
int64_t trace_id = 0;
std::map<std::string, int64_t> trace_handle_map;
int64_t event_id = 0;
std::map<std::string, int64_t> event_handle_map;
}
/**
 * @brief       Create trace handle.
 * @param [in]  tracerType:    trace type
 * @param [in]  objName:       object name
 * @return      atrace handle
 */
TraHandle AtraceCreate(TracerType tracerType, const char *objName) {
  trace_id++;
  trace_handle_map.emplace(objName, trace_id);
  return static_cast<intptr_t>(trace_id);
}

/**
 * @brief       Create trace handle.
 * @param [in]  tracerType:    trace type
 * @param [in]  objName:       object name
 * @param [in]  attr:          object attribute
 * @return      atrace handle
 */
TraHandle AtraceCreateWithAttr(TracerType tracerType, const char *objName, const TraceAttr *attr) {
  trace_id++;
  trace_handle_map.emplace(objName, trace_id);
  return static_cast<intptr_t>(trace_id);
}

/**
 * @brief       Get trace handle
 * @param [in]  tracerType:    trace type
 * @param [in]  objName:       object name
 * @return      atrace handle
 */
TraHandle AtraceGetHandle(TracerType tracerType, const char *objName) {
  auto iter = trace_handle_map.find(objName);
  if (iter == trace_handle_map.end()) {
    return -1;
  }
  return static_cast<intptr_t>(iter->second);
}

/**
 * @brief       Submite trace info
 * @param [in]  handle:    trace handle
 * @param [in]  buffer:    trace info buffer
 * @param [in]  bufSize:   size of buffer
 * @return      TraStatus
 */
TraStatus AtraceSubmit(TraHandle handle, const void *buffer, uint32_t bufSize) {
  const char* msg = reinterpret_cast<const char*>(buffer);
  std::string msg_str(msg, bufSize);
  std::cout << msg_str << std::endl;
  return TRACE_SUCCESS;
}

/**
 * @brief       Destroy trace handle
 * @param [in]  handle:    trace handle
 * @return      NA
 */
void AtraceDestroy(TraHandle handle) {
  for (auto iter = trace_handle_map.begin(); iter != trace_handle_map.end(); iter++) {
    if (iter->second == static_cast<int64_t>(handle)) {
      trace_handle_map.erase(iter);
      break;
    }
  }
}

/**
 * @brief       Save trace info for all handle of tracerType
 * @param [in]  tracerType:    trace type to be saved
 * @param [in]  syncFlag:      synchronize or asynchronizing
 * @return      TraStatus
 */
TraStatus AtraceSave(TracerType tracerType, bool syncFlag) {
  return TRACE_SUCCESS;
}

void *AtraceStructEntryListInit(void) {
  return nullptr;
}
void AtraceStructEntryName(TraceStructEntry *entry, const char *name) {}
void AtraceStructItemSet(TraceStructEntry *entry, const char *name, uint8_t type, uint8_t mode,
                         uint8_t bytes, uint64_t length) {}
void AtraceStructEntryExit(TraceStructEntry *entry) {}

/**
 * @brief       create thread to recv device trace log
 * @param [in]  devId:         device id
 * @return      TraStatus
 */
TraStatus AtraceReportStart(int32_t devId) {
  return TRACE_SUCCESS;
}

/**
* @brief       stop thread to recv device trace log
* @param [in]  devId:         device id
*/
void AtraceReportStop(int32_t devId) {}

/**
 * @brief       Create trace event.
 * @param [in]  eventName:     event name
 * @return      event handle
 */
TraEventHandle AtraceEventCreate(const char *eventName) {
  event_id++;
  event_handle_map.emplace(eventName, event_id);
  return static_cast<intptr_t>(event_id);
}

/**
 * @brief       Get event handle
 * @param [in]  eventName:     event name
 * @return      event handle
 */
TraEventHandle AtraceEventGetHandle(const char *eventName) {
  auto iter = event_handle_map.find(eventName);
  if (iter == event_handle_map.end()) {
    return static_cast<intptr_t>(-1);
  }
  return static_cast<intptr_t>(iter->second);
}

/**
 * @brief       Destroy event handle
 * @param [in]  eventHandle:    event handle
 * @return      NA
 */
void AtraceEventDestroy(TraEventHandle eventHandle) {
  for (auto iter = event_handle_map.begin(); iter != event_handle_map.end(); iter++) {
    if (eventHandle == static_cast<intptr_t>(iter->second)) {
      event_handle_map.erase(iter);
      break;
    }
  }
}

/**
 * @brief       Bind event handle with trace handle
 * @param [in]  eventHandle:    event handle
 * @param [in]  handle:         trace handle
 * @return      TraStatus
 */
TraStatus AtraceEventBindTrace(TraEventHandle eventHandle, TraHandle handle) {
  return TRACE_SUCCESS;
}

/**
 * @brief       Set event attr
 * @param [in]  eventHandle:    event handle
 * @param [in]  attr:           event attribute
 * @return      TraStatus
 */
TraStatus AtraceEventSetAttr(TraEventHandle eventHandle, const TraceEventAttr *attr) {
  return TRACE_SUCCESS;
}

/**
 * @brief       Report event and save the bound trace log to disk
 * @param [in]  eventHandle:    event handle
 * @return      TraStatus
 */
TraStatus AtraceEventReportSync(TraEventHandle eventHandle) {
  return TRACE_SUCCESS;
}
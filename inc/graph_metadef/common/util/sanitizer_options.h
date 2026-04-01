/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_UTILS_SANITIZERS_SANITIZER_OPTIONS_H_
#define COMMON_UTILS_SANITIZERS_SANITIZER_OPTIONS_H_

// active if -fsanitize=address
#if defined(__SANITIZE_ADDRESS__)
#include "sanitizer/lsan_interface.h"
/*  如果业务代码中存在已知的内存泄漏的代码块, 并且允许这部分内存泄漏存在, 可在代码块首尾添加开关,
 *  控制地址消毒器关闭与开启.
 *  开关仅在蓝区CI场景可用，仅在当前thread有效.
 *  DT_DETECT_LEAKS_OFF();
 *  // code block with memory leak
 *  DT_DETECT_LEAKS_ON();
 */
#define DT_DETECT_LEAKS_OFF() \
  do {                        \
    __lsan_disable();         \
  } while (0)
#define DT_DETECT_LEAKS_ON()  \
  do {                        \
    __lsan_enable();          \
  } while (0)
#define DT_DO_DETECT_LEAKS()  \
  do {                        \
    __lsan_do_leak_check();   \
  } while (0)
#else
#define DT_DETECT_LEAKS_OFF() \
  do {                        \
  } while (0)
#define DT_DETECT_LEAKS_ON()  \
  do {                        \
  } while (0)
#define DT_DO_DETECT_LEAKS()  \
  do {                        \
  } while (0)
#endif

#define DT_ALLOW_LEAKS_GUARD(name) ::ge::LeaksGuarder leaks_guard_for_##name

namespace ge {
class LeaksGuarder {
 public:
  LeaksGuarder(const LeaksGuarder &) = delete;
  LeaksGuarder &operator=(const LeaksGuarder &) = delete;

  LeaksGuarder() {
    DT_DETECT_LEAKS_OFF();
  }

  ~LeaksGuarder() {
    DT_DETECT_LEAKS_ON();
  }
};

}  // namespace ge

#endif  // COMMON_UTILS_SANITIZERS_SANITIZER_OPTIONS_H_

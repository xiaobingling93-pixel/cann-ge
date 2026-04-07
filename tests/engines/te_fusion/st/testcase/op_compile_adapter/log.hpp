/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <stdio.h>
#include "securec.h"
#include "cce/dnn.h"
#include <cmath>

/**
  * @ingroup cce
  * @brief log level fatal
  */
#define CC_LOG_FATAL                   ("FATAL")

/**
  * @ingroup cce
  * @brief log level error
  */
#define CC_LOG_ERROR                   ("ERROR")

/**
  * @ingroup cce
  * @brief log level warning
  */
#define CC_LOG_WARNING                 ("WARNING")

/**
  * @ingroup cce
  * @brief log level info
  */
#define CC_LOG_INFO                    ("INFO")

/**
  * @ingroup cce
  * @brief log level debug
  */
#define CC_LOG_DEBUG                   ("DEBUG")

#define TVM_LOG(level, format, ...) \
    do {fprintf(stderr, "[%s] [%s] [%s:%d] " format "\n", \
                level, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__);}while(0);

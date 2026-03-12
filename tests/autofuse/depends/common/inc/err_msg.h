/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STUB_ERR_MSG_H_
#define STUB_ERR_MSG_H_

// Stub version of err_msg.h for UT/ST testing
// REPORT_INNER_ERR_MSG is a macro used in error reporting, defined as no-op in test environment

#ifdef STUB_LOG
#define REPORT_INNER_ERR_MSG(err_code, fmt, ...) do {} while (0)
#else
#define REPORT_INNER_ERR_MSG(err_code, fmt, ...) printf("[ERR_MSG]" fmt "\n", ##__VA_ARGS__)
#endif

#endif  // STUB_ERR_MSG_H_

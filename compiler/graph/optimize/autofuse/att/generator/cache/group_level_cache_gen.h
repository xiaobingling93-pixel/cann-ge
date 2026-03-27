/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_GROUP_LEVEL_CACHE_GEN_H_
#define ATT_GROUP_LEVEL_CACHE_GEN_H_

#include "tiling_cache_code_gen.h"
#include "base/model_info.h"
#include "external/ge_common/ge_api_types.h"

namespace att {
namespace cache {
/**
 * @brief Group级缓存代码生成器
 * 负责生成Group间缓存相关代码（复用现有逻辑）
 */
class GroupLevelCacheGen : public TilingCacheCodeGen {
public:
  GroupLevelCacheGen() = default;
  ~GroupLevelCacheGen() override = default;

  /**
   * @brief 生成FixedSizeHashMap模板类定义
   * @param code_printer 代码打印器
   * @return ge::Status
   */
  ge::Status GenFixedSizeHashMapDef(ge::CodePrinter &code_printer) override;

  /**
   * @brief 生成Group级缓存类型定义
   * @param code_printer 代码打印器
   * @param input_vars_size 输入变量大小
   * @param cache_capacity 缓存容量
   * @return ge::Status
   */
  ge::Status GenGroupCacheTypes(ge::CodePrinter &code_printer,
                                size_t cache_capacity);

  /**
   * @brief 生成Group级缓存函数定义
   * @param code_printer 代码打印器
   * @param tiling_data_type_name TilingData类型名称
   * @return ge::Status
   */
  ge::Status GenGroupCacheFunctions(ge::CodePrinter &code_printer,
                                    const std::string &tiling_data_type_name);
};
} // namespace cache
} // namespace att

#endif // ATT_GROUP_LEVEL_CACHE_GEN_H_
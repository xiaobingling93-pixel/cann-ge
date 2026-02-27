/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_TILING_CACHE_CODE_GEN_H_
#define ATT_TILING_CACHE_CODE_GEN_H_

#include <string>
#include "base/base_types.h"
#include "generator_config.h"
#include "external/ge_common/ge_api_types.h"
#include "common/code_printer.h"

namespace att {
namespace cache {

/**
 * @brief Tiling缓存代码生成器基类
 * 负责生成FixedSizeHashMap模板类和通用缓存函数
 */
class TilingCacheCodeGen {
public:
  TilingCacheCodeGen() = default;
  virtual ~TilingCacheCodeGen() = default;

  /**
   * @brief 生成FixedSizeHashMap模板类定义
   * @param code_printer 代码打印器
   * @return ge::Status
   */
  virtual ge::Status GenFixedSizeHashMapDef(ge::CodePrinter& code_printer) = 0;

  /**
   * @brief 生成常量定义代码
   * @param code_printer 代码打印器
   * @param input_vars_size 输入变量大小
   */
  static void GenConstantDefs(ge::CodePrinter& code_printer, size_t input_vars_size);

protected:
  /**
   * @brief 生成HashMap模板代码
   * @return std::string 生成的代码字符串
   */
  static std::string GenHashMapTemplate();

  /**
   * @brief 生成HashMap类结构（私有部分）
   * @return std::string 生成的代码字符串
   */
  static std::string GenHashMapClassStructure();

  /**
   * @brief 生成HashMap构造函数
   * @return std::string 生成的代码字符串
   */
  static std::string GenHashMapConstructor();

  /**
   * @brief 生成HashMap公共方法（Find/Insert/Erase/Clear等）
   * @return std::string 生成的代码字符串
   */
  static std::string GenHashMapPublicMethods();

  /**
   * @brief 生成HashMap的Find方法
   * @return std::string 生成的代码字符串
   */
  static std::string GenFindMethod();

  /**
   * @brief 生成HashMap的Insert方法
   * @return std::string 生成的代码字符串
   */
  static std::string GenInsertMethod();

  /**
   * @brief 生成HashMap的Erase方法
   * @return std::string 生成的代码字符串
   */
  static std::string GenEraseMethod();

  /**
   * @brief 生成HashMap的Clear和Size方法
   * @return std::string 生成的代码字符串
   */
  static std::string GenClearAndSizeMethods();

  /**
   * @brief 生成Hash函数代码
   * @return std::string 生成的代码字符串
   */
  static std::string GenHashFunction();

  /**
   * @brief 生成FindIndex函数代码
   * @return std::string 生成的代码字符串
   */
  static std::string GenFindIndexFunction();
};

} // namespace cache
} // namespace att

#endif // ATT_TILING_CACHE_CODE_GEN_H_

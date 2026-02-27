/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_GENERATOR_MANAGER_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_GENERATOR_MANAGER_H_

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include "gen_esb_options.h"
#include "generator_interface.h"
#include "utils_generator.h"

namespace ge {
namespace es {

/**
 * 代码生成器管理器
 * 统一管理所有代码生成器
 */
class GeneratorManager {
 public:
  // 添加生成器
  void AddGenerator(std::unique_ptr<ICodeGenerator> generator);

  // 生成所有算子的代码
  void GenAllOps(const std::vector<OpDescPtr> &all_ops,
                 std::unordered_map<std::string, std::vector<std::string>> &unsupported_reasons_to_ops,
                 std::vector<std::string> &exclude_ops,
                 size_t &supported_num);

  // 生成聚合文件头部
  void GenAggregateHeaders();

  // 生成聚合文件
  void GenAggregateFiles(const std::string &output_dir);

  // 生成单个算子文件
  void GenPerOpFiles(const std::string &output_dir);

  // 获取所有生成器
  const std::vector<std::unique_ptr<ICodeGenerator>> &GetGenerators() const {
    return generators_;
  }

  // 获取Utils生成器, 目前仅生成一个es_elog.h
  const std::unique_ptr<UtilsGenerator> &GetUtilGenerator() const {
    return utils_generator_;
  }

  // 配置UtilsGenerator
  void SetUtilsGenerator(std::unique_ptr<UtilsGenerator> generator);

  // 生成Utils文件
  void GenUtils(const std::string &output_dir);

 private:
  // 收集已生成的算子类型（排序）
  std::vector<std::string> CollectGeneratedOpTypes() const;

  // 收集已生需要生成的utils名字
  std::vector<std::string> CollectGeneratedUtilNames() const;

  // 重置所有生成器的失败状态
  void ResetAllGeneratorsOnFailure(const OpDescPtr &op);

  // 处理生成异常
  void HandleGenerationException(const std::exception &e, const OpDescPtr &op,
                                 std::unordered_map<std::string, std::vector<std::string>> &unsupported_reasons_to_ops);

  std::vector<std::unique_ptr<ICodeGenerator>> generators_;
  std::unique_ptr<UtilsGenerator> utils_generator_;
};
void GeneratePerOpFiles(const std::string &output_dir, GeneratorManager &manager);
void GenEsImpl(const GenEsbOptions &options);
void Gen(GeneratorManager &manager, std::vector<std::string> &exclude_ops);
void GenUnsupportedOpsInfo(GeneratorManager &manager,
                           const std::unordered_map<std::string, std::vector<std::string>> &unsupported_reasons_to_ops);
void GenerateAggregateFiles(const std::string &output_dir, GeneratorManager &manager);
void GeneratePerUtilFiles(const std::string &output_dir, GeneratorManager &manager);
void ProcessOutputDirectory(std::string &output_dir);
void ExecuteCodeGeneration(const std::string &output_dir, GeneratorManager &manager,
                           std::vector<std::string> &exclude_ops);
void GenAllOps(GeneratorManager &manager,
               std::unordered_map<std::string, std::vector<std::string>> &unsupported_reasons_to_ops,
               std::vector<std::string> &exclude_ops, size_t &supported_num);
void DisplayGenerationParameters(const GenEsbOptions &options);
std::unique_ptr<GeneratorManager> InitializeGenerators(const std::string &module_name, const std::string &guard_prefix);
std::unique_ptr<GeneratorManager> CreateGenerators(const std::string &module_name, const std::string &guard_prefix);
void GenerateAggregateHeaders(GeneratorManager &manager);
std::vector<std::string> ParseExcludeGenOps(const std::string &exclude_ops_str);
void GenerateHistoryRegistry(const std::string &output_dir, const std::string &release_version,
                            const std::string &release_date, const std::string &branch_name);
std::vector<OpDescPtr> CollectAndSortAllOps();
}  // namespace es
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_GENERATOR_MANAGER_H_

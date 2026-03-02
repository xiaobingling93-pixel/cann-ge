/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <vector>
#include "graph/utils/file_utils.h"
#include "mmpa/mmpa_api.h"

#include "ge_ir_collector.h"
#include "history/history_registry_writer.h"
#include "utils.h"
#include "common/ge_common/string_util.h"
#include "generator_manager.h"
#include "py_generator.h"
#include "c_generator.h"
#include "cpp_generator.h"
#include "es_codegen_default_value.h"

namespace ge {
namespace es {

/**
 * 创建并配置所有代码生成器
 * @param module_name 模块名
 * @param guard_prefix 保护宏前缀
 * @param history_registry 历史原型结构化数据目录
 * @param release_version 当前版本号
 * @return 配置好的生成器管理器
 */
std::unique_ptr<GeneratorManager> CreateGenerators(const std::string &module_name, const std::string &guard_prefix,
                                                  const std::string &history_registry, const std::string &release_version) {
  // 外部统一捕获make_unique失败等异常
  auto manager = std::make_unique<GeneratorManager>();

  // 创建C生成器
  auto c_gen = std::make_unique<CGenerator>();
  c_gen->SetModuleName(module_name);
  c_gen->SetHGuardPrefix(guard_prefix);
  manager->AddGenerator(std::move(c_gen));

  // 创建C++生成器
  auto cpp_gen = std::make_unique<CppGenerator>();
  cpp_gen->SetModuleName(module_name);
  cpp_gen->SetHGuardPrefix(guard_prefix);
  cpp_gen->SetHistoryRegistry(history_registry);
  cpp_gen->SetReleaseVersion(release_version);
  manager->AddGenerator(std::move(cpp_gen));

  // 创建Python生成器
  auto py_gen = std::make_unique<PyGenerator>();
  py_gen->SetModuleName(module_name);
  manager->AddGenerator(std::move(py_gen));

  // 创建Utils生成器
  auto util_gen = std::make_unique<UtilsGenerator>();
  manager->SetUtilsGenerator(std::move(util_gen));

  return manager;
}

std::vector<OpDescPtr> CollectAndSortAllOps() {
  auto env_opp_path = std::getenv("ASCEND_OPP_PATH");
  std::cout << "ASCEND_OPP_PATH is: " << env_opp_path << std::endl;

  auto all_ops = GeIrCollector::CollectAndCreateAllOps();
  std::cout << "Get ops num: " << all_ops.size() << std::endl;

  std::sort(all_ops.begin(), all_ops.end(),
            [](const OpDescPtr &a, const OpDescPtr &b) { return a->GetType() < b->GetType(); });
  return all_ops;
}

/**
 * 生成所有算子的代码
 * @param manager 生成器管理器
 * @param unsupported_reasons_to_ops 不支持的算子及原因
 * @param supported_num 支持的算子数量
 */
void GenAllOps(GeneratorManager &manager,
               std::unordered_map<std::string, std::vector<std::string>> &unsupported_reasons_to_ops,
               std::vector<std::string> &exclude_ops,
               size_t &supported_num) {
  auto all_ops = CollectAndSortAllOps();
  if (all_ops.empty()) {
    std::cerr << "ERROR: no ops to generate, please check the ops proto path constructed from env ASCEND_OPP_PATH:"
              << std::getenv("ASCEND_OPP_PATH") << std::endl;
    return;
  }
  manager.GenAllOps(all_ops, unsupported_reasons_to_ops, exclude_ops, supported_num);
}

/**
 * 生成不支持的算子信息到所有生成器
 * @param manager 生成器管理器
 * @param unsupported_reasons_to_ops 不支持的算子及原因
 */
void GenUnsupportedOpsInfo(
    GeneratorManager &manager,
    const std::unordered_map<std::string, std::vector<std::string>> &unsupported_reasons_to_ops) {
  for (auto &generator : manager.GetGenerators()) {
    auto &content_stream = generator->GetAggregateContentStream();
    content_stream << std::endl << generator->GetCommentStart() << " Unsupported ops and reasons: " << std::endl;

    for (const auto &iter : unsupported_reasons_to_ops) {
      content_stream << generator->GetCommentLinePrefix() << iter.first << "(" << iter.second.size() << "): [";
      for (const auto &op : iter.second) {
        content_stream << op << ", ";
      }
      content_stream << ']' << std::endl;
    }
    content_stream << generator->GetCommentEnd() << std::endl;
  }
}

/**
 * 生成代码的主要流程
 * @param manager 生成器管理器
 */
void Gen(GeneratorManager &manager, std::vector<std::string> &exclude_ops) {
  std::unordered_map<std::string, std::vector<std::string>> unsupported_reasons_to_ops;
  size_t supported_num = 0U;

  GenAllOps(manager, unsupported_reasons_to_ops, exclude_ops, supported_num);

  std::cout << "Generate done, generated num " << supported_num << ", unsupported: " << std::endl;

  // 输出不支持的算子信息到控制台，使用ANSI支持醒目颜色输出
  for (const auto &iter : unsupported_reasons_to_ops) {
    std::cout << "\033[1;33m" << iter.first << "(" << iter.second.size() << "): [";
    bool first = true;
    for (const auto &op : iter.second) {
      if (!first) {
        std::cout << ", ";
      }
      std::cout << op;
      first = false;
    }
    std::cout << "]\033[0m" << std::endl;
  }

  // 输出不支持的算子信息到所有生成器的注释中
  GenUnsupportedOpsInfo(manager, unsupported_reasons_to_ops);
}

/**
 * 生成每个op的单独文件
 * @param output_dir 输出目录
 * @param manager 生成器管理器
 */
void GeneratePerOpFiles(const std::string &output_dir, GeneratorManager &manager) {
  manager.GenPerOpFiles(output_dir);
}

/**
 * 生成聚合头文件
 * @param output_dir 输出目录
 * @param manager 生成器管理器
 */
void GenerateAggregateFiles(const std::string &output_dir, GeneratorManager &manager) {
  manager.GenAggregateFiles(output_dir);

  std::cout << "Generated aggregate files:" << std::endl;
  for (const auto &generator : manager.GetGenerators()) {
    std::cout << "  " << generator->GetGeneratorName() << ": " << output_dir << generator->GetAggregateFileName()
              << std::endl;
  }
}

void GeneratePerUtilFiles(const std::string &output_dir, GeneratorManager &manager) {
  manager.GenUtils(output_dir);

  std::cout << "Generated util files:" << std::endl;
  std::cout << "  " << manager.GetUtilGenerator()->GetGeneratorName() << ":" << std::endl;
  for (const auto &util_name: manager.GetUtilGenerator()->GetUtilFileNames()) {
    std::cout << "    " << output_dir<< util_name << std::endl;
  }
}

/**
 * 初始化代码生成器
 * @param module_name 模块名
 * @param guard_prefix 保护前缀
 * @param history_registry 历史原型结构化数据目录
 * @param release_version 当前版本号
 * @return 初始化后的生成器管理器
 */
std::unique_ptr<GeneratorManager> InitializeGenerators(const std::string &module_name,
                                                       const std::string &guard_prefix,
                                                       const std::string &history_registry,
                                                       const std::string &release_version) {
  return CreateGenerators(module_name, guard_prefix, history_registry, release_version);
}

/**
 * 生成聚合头文件头部
 * @param manager 生成器管理器
 */
void GenerateAggregateHeaders(GeneratorManager &manager) {
  manager.GenAggregateHeaders();
}

/**
 * 处理输出目录路径并创建目录（如果不存在）
 * @param output_dir 输出参数，输出目录路径
 */
void ProcessOutputDirectory(std::string &output_dir) {
  // 确保输出目录以路径分隔符结尾
  if (output_dir.back() != '/' && output_dir.back() != '\\') {
    output_dir += "/";
  }

  // 检查输出目录是否存在，如果不存在则创建
  if (mmAccess(output_dir.c_str()) != EN_OK) {
    if (ge::CreateDir(output_dir) != 0) {
      std::cerr << "Error creating output directory: " << output_dir << std::endl;
      output_dir = ge::es::kEsCodeGenDefaultOutputDir;
    } else {
      std::cout << "Created output directory: " << output_dir << std::endl;
    }
  }

  std::cout << "Output directory: " << output_dir << std::endl;
}

/**
 * 执行完整的代码生成流程
 * @param output_dir 输出目录
 * @param manager 生成器管理器
 */
void ExecuteCodeGeneration(const std::string &output_dir, GeneratorManager &manager, std::vector<std::string> &exclude_ops) {
  // 生成所有算子的代码
  Gen(manager, exclude_ops);

  // 生成每个算子的单独文件
  GeneratePerOpFiles(output_dir, manager);

  // 生成聚合文件
  GenerateAggregateFiles(output_dir, manager);

  // 生成utils文件， 目前仅es_log.h
  GeneratePerUtilFiles(output_dir, manager);

  std::cout << "Generated files in: " << output_dir << std::endl;
}

/**
 * 显示 gen_esb 参数信息
 * @param options gen_esb 参数
 */
void DisplayGenerationParameters(const GenEsbOptions &options) {
  if (options.mode == ge::es::kEsExtractHistoryMode) {
    if (!options.release_version.empty()) {
      std::cout << "Release version: " << options.release_version << std::endl;
    }
    if (!options.release_date.empty()) {
      std::cout << "Release date: " << options.release_date << std::endl;
    }
    if (!options.branch_name.empty()) {
      std::cout << "Branch name: " << options.branch_name << std::endl;
    }
    return;
  }

  std::cout << "Module name: " << options.module_name << std::endl;
  if (!options.h_guard_prefix.empty()) {
    std::cout << "Header guard prefix: " << options.h_guard_prefix << std::endl;
  }
  if (!options.exclude_ops.empty()) {
    std::cout << "Exclude ops: " << options.exclude_ops << std::endl;
  }
  if (!options.history_registry.empty()) {
    std::cout << "History registry: " << options.history_registry << std::endl;
  }
  if (!options.release_version.empty()) {
    std::cout << "Release version: " << options.release_version << std::endl;
  }
}

std::vector<std::string> ParseExcludeGenOps(const std::string &exclude_ops_str) {
  std::string tmp_str = exclude_ops_str;
  (void)ge::StringUtils::Trim(tmp_str);
  std::vector<std::string> exclude_vec = ge::StringUtils::Split(tmp_str, ',');
  return exclude_vec;
}

/**
 * 生成历史原型结构化数据
 * @param output_dir 输出目录
 * @param release_version 发布版本号
 * @param release_date 发布日期
 * @param branch_name 分支名
 */
void GenerateHistoryRegistry(const std::string &output_dir, const std::string &release_version,
                            const std::string &release_date, const std::string &branch_name) {
  auto all_ops = CollectAndSortAllOps();
  if (all_ops.empty()) {
    std::cerr << "ERROR: no ops to generate, please check the ops proto path constructed from env ASCEND_OPP_PATH:"
              << std::getenv("ASCEND_OPP_PATH") << std::endl;
    return;
  }

  ge::es::history::HistoryRegistryWriter::WriteRegistry(output_dir, release_version, release_date, branch_name, all_ops);
}

/**
 * 生成代码
 * @param output_dir 输出目录
 * @param options 生成代码参数
 */
void GenerateCode(const std::string &output_dir, const GenEsbOptions &options) {
  // 1. 初始化代码生成器
  auto manager = InitializeGenerators(options.module_name, options.h_guard_prefix,
                                                               options.history_registry, options.release_version);

  // 2. 生成聚合文件头部
  GenerateAggregateHeaders(*manager);

  auto exclude_ops = ParseExcludeGenOps(options.exclude_ops);

  // 3. 执行完整的代码生成流程
  ExecuteCodeGeneration(output_dir, *manager, exclude_ops);
}

void GenEsImpl(const GenEsbOptions &options) {
  // 1. 处理输出目录路径和创建目录
  std::string processed_output_dir = options.output_dir;
  ProcessOutputDirectory(processed_output_dir);

  // 2. 显示生成参数信息
  DisplayGenerationParameters(options);

  // 3. 执行 ES 生成任务
  if (options.mode == ge::es::kEsExtractHistoryMode) {
    GenerateHistoryRegistry(processed_output_dir, options.release_version, options.release_date, options.branch_name);
  } else {
    GenerateCode(processed_output_dir, options);
  }
}
}  // namespace es
}  // namespace ge

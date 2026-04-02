/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "om2_codegen.h"
#include <fstream>
#include <sstream>
#include "common/helper/om2/om2_utils.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "program_generator.h"
#include "om2_code_printer.h"
#include "common/om2/codegen/emitter/cpp_emitter.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "mmpa/mmpa_api.h"


namespace ge {
namespace {
constexpr int32_t kCppFileExtLen = 4;

Status WriteProgramToFiles(const Program &program, const std::string &ws_dir, const std::string &model_name,
                              std::vector<std::string> &output_file_paths) {
  CppEmitter emitter;
  Om2CodePrinter code_printer(model_name, ws_dir);

  for (size_t i = 0U; i < static_cast<size_t>(GeneratedFileIndex::kEnd); i++) {
    if (program[i].empty()) {
      GELOGI("[OM2] Model %s program part %d has no content.", model_name.c_str(), i);
      continue;
    }
    for (const auto *node : program[i]) {
      if (node != nullptr) {
        std::string code_content;
        GE_ASSERT_SUCCESS(node->Accept(emitter, code_content), "[OM2] Program %zu code generation failed.", i);
        code_printer.AddLine(static_cast<GeneratedFileIndex>(i), code_content);
      }
    }
  }

  GE_ASSERT_SUCCESS(code_printer.WriteFiles(ws_dir));
  code_printer.GetOutputFilePaths(output_file_paths);
  GELOGI("[OM2] Model %s has finished the codegen process.", model_name.c_str());

  return SUCCESS;
}

}  // namespace

Om2Codegen::~Om2Codegen() {
  CleanOm2WorkDir();
}

void Om2Codegen::CleanOm2WorkDir() const {
  if (mmAccess2(ws_dir_.c_str(), M_F_OK) != EN_OK) {
    return;
  }

  // 临时环境变量，调试用
  const char *keep_ws = std::getenv("OM2_KEEP_WORKSPACE");
  if (keep_ws != nullptr && std::strcmp(keep_ws, "1") == 0) {
    GELOGI("[OM2] Keep workspace directory (OM2_KEEP_WORKSPACE=1): %s", ws_dir_.c_str());
    return;
  }

  (void)Om2Utils::RmOm2WorkspaceDir(ws_dir_);
}

Status Om2Codegen::Om2CodegenAndCompile(const ge::GeModelPtr &ge_model, vector<std::string> &output_file_paths) {
  ProgramGenerator generator;
  GE_ASSERT_SUCCESS(generator.Init(ge_model));
  GE_ASSERT_SUCCESS(Om2Utils::CreateOm2WorkspaceDir(ws_dir_));
  const std::string runtime_ws_dir = ws_dir_ + "runtime/";
  GE_ASSERT_SUCCESS(ge::CreateDir(runtime_ws_dir), "Failed to create om2 work dir: [%s]", runtime_ws_dir.c_str());

  Program program;
  GE_ASSERT_SUCCESS(generator.GenerateKernelRegSource(program));
  GE_ASSERT_SUCCESS(generator.GenerateInterfaceHeader(program));
  GE_ASSERT_SUCCESS(generator.GenerateResourcesSource(program));
  GE_ASSERT_SUCCESS(generator.GenerateLoadAndRunSource(program));
  GE_ASSERT_SUCCESS(generator.GenerateArgsManagerSource(program));
  GE_ASSERT_SUCCESS(generator.GenerateMakeFile(program));

  std::vector<std::string> all_file_paths;
  GE_ASSERT_SUCCESS(WriteProgramToFiles(program, runtime_ws_dir, ge_model->GetName(), all_file_paths));

  std::vector<std::string> cpp_file_paths;
  for (const auto &path : all_file_paths) {
    if (!path.empty() && path.size() >= kCppFileExtLen &&
        path.compare(path.size() - kCppFileExtLen, kCppFileExtLen, ".cpp") == 0) {
      cpp_file_paths.push_back(path);
    }
  }

  const std::string so_file_path = runtime_ws_dir + "lib" + ge_model->GetName() + "_om2.so";
  GE_ASSERT_SUCCESS(Om2Utils::CompileGeneratedCppToSo(cpp_file_paths, so_file_path, false));
  GELOGI("[OM2] Model %s has finished generating source code files and compiling to the shared library.",
         ge_model->GetName().c_str());
  GE_ASSERT_TRUE(mmAccess2(so_file_path.c_str(), M_F_OK) == EN_OK);
  output_file_paths.insert(output_file_paths.end(), all_file_paths.begin(), all_file_paths.end());
  output_file_paths.push_back(so_file_path);
  return SUCCESS;
}
}  // namespace ge

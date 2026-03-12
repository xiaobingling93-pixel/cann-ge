/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_CODEGEN_PROGRAM_GENERATOR_H_
#define AIR_CXX_BASE_COMMON_OM2_CODEGEN_PROGRAM_GENERATOR_H_

#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "om2_codegen.h"
#include "common/model/ge_model.h"
#include "common/om2/codegen/code_generator_factory.h"
#include "common/om2/codegen/task_code_generator/task_code_generator.h"
#include "common/om2/codegen/ast/ast_context.h"
#include "framework/common/ge_inner_error_codes.h"

namespace ge {
#define EMIT_CODE(ss, code) (ss << code << '\n')

class ProgramGenerator {
 public:
  Status Init(const GeModelPtr &ge_model);
  Status GenerateInterfaceHeader(Program &program);
  Status GenerateResourcesSource(Program &program);
  Status GenerateKernelRegSource(Program &program);
  Status GenerateArgsManagerSource(Program &program);
  Status GenerateLoadAndRunSource(Program &program);
  Status GenerateMakeFile(Program &program);

 private:
  struct LoadTaskParams {
    std::vector<AstNode *> *distribution_code = nullptr;
    std::vector<AstNode *> *dist_impl_code = nullptr;
    std::unordered_set<ModelTaskType> *model_task_types = nullptr;
    std::unordered_map<int64_t, OpInputEdges> *op_id_to_input_edges = nullptr;
    std::unordered_map<int64_t, std::string> *weight_offset_to_varname = nullptr;
    std::vector<AstNode *> *const_input_ast_nodes = nullptr;
  };

  Status PrepareGraphData();
  Status InitRuntimeParams();
  Status CreateTaskCodeGenerators();
  OpDescPtr FindOpDescByIndex(int64_t op_index) const;
  Status ProcessLoadTask(size_t task_index, domi::TaskDef &task_def, LoadTaskParams &params);

  void GenKernelRegConsts(Program &program);
  void GenKernelRegCommonFuncs(Program &program);
  void GenKernelRegFuncsImpl(Program &program);
  Status GenRegisterKernelsImpl(Program &program);
  Status GenAicpuArgsCommon(Program &program);
  Status GenConstInputs(std::vector<AstNode *> &const_input_ast_nodes, const OpDescPtr &op_desc,
                        std::unordered_map<int64_t, std::string> &weight_offset_to_varname);
  Status GenArgsTableImpl(std::vector<AstNode *> &ast_nodes);
  void GenInterfaceHeaderMacros(Program &program);
  void GenInterfaceHeaderCommonFunc(Program &program);
  Status GenInterfaceHeaderCommonPart(Program &program);
  void GenInterfaceHeaderOm2ArgsTableClass(Program &program);
  void GenInterfaceHeaderOm2ModelClass(Program &program);
  void GenInterfaceHeaderExternalApi(Program &program);
  void GenOm2ModelConstructor(Program &program);
  void GenOm2ModelDestructor(Program &program);
  Status GenInitResourcesImpl(Program &program);
  void GenReleaseResourcesImpl(Program &program);
  Status GenLoadImpl(std::vector<AstNode *> &load_impl_code, std::vector<AstNode *> &dist_impl_code);
  Status GenRunImpl(std::vector<AstNode *> &load_impl_code);
  void GenExternalApiImpl(Program &program);

  GeModelPtr ge_model_;
  Om2RuntimeParam runtime_param_{};
  std::unordered_map<int64_t, OpDescPtr> op_list_;
  std::vector<TaskCodeGeneratorPtr> task_code_generator_list_;
  std::unordered_map<std::string, uint32_t> func_handle_indices_;
  AstContext ast_ctx_;
  ArgsInfo args_info_{};
  uint32_t aicpu_task_num_ = 0U;
  uint64_t args_table_index_ = 0U;
  std::set<int64_t> model_io_offsets_;
};
}  // namespace ge

#endif  // AIR_CXX_BASE_COMMON_OM2_CODEGEN_PROGRAM_GENERATOR_H_

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "program_generator.h"
#include <cinttypes>
#include <sstream>
#include <set>
#include <unordered_set>
#include "common/om2/codegen/ast/ast_nodes.h"
#include "common/helper/om2/om2_utils.h"
#include "common/om2/codegen/om2_codegen_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/node_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/tensor_utils.h"

namespace ge {

void WriteInterfaceHeader(Program &program, AstNode *node) {
  program[static_cast<size_t>(GeneratedFileIndex::kInterfaceHeaderFile)].push_back(node);
}

void WriteResourcesSource(Program &program, AstNode *node) {
  program[static_cast<size_t>(GeneratedFileIndex::kResourcesFile)].push_back(node);
}

void WriteKernelRegSource(Program &program, AstNode *node) {
  program[static_cast<size_t>(GeneratedFileIndex::kKernelRegistryFile)].push_back(node);
}

void WriteLoadAndRunSource(Program &program, AstNode *node) {
  program[static_cast<size_t>(GeneratedFileIndex::kLoadingAndRunningFile)].push_back(node);
}

void WriteCmakeLists(Program &program, AstNode *node) {
  program[static_cast<size_t>(GeneratedFileIndex::kCMakeListsFile)].push_back(node);
}

void WriteArgManagerSource(Program &program, AstNode *node) {
  program[static_cast<size_t>(GeneratedFileIndex::kArgsManagerFile)].push_back(node);
}

Status GetModelInputNumber(const GeModelPtr &ge_model, size_t &input_num) {
  auto compute_graph = ge_model->GetGraph();
  GE_ASSERT_NOTNULL(compute_graph);
  uint32_t input_count = 0U;
  for (const auto &node : compute_graph->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    if (OpTypeUtils::IsDataNode(op_desc->GetType())) {
      input_count++;
    }
  }
  input_num = input_count;
  return ge::GRAPH_SUCCESS;
}

Status GetModelOutputNumber(const GeModelPtr &ge_model, size_t &output_num) {
  auto compute_graph = ge_model->GetGraph();
  GE_ASSERT_NOTNULL(compute_graph);
  uint32_t output_count = 0U;
  for (const auto &node : compute_graph->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    if (OpTypeUtils::IsGraphOutputNode(op_desc->GetType())) {
      output_count += op_desc->GetAllInputsSize();
    }
  }
  output_num = output_count;
  return ge::GRAPH_SUCCESS;
}

Status GetStreamFlags(const GeModelPtr &ge_model, std::vector<std::string> &stream_flags_list) {
  auto compute_graph = ge_model->GetGraph();
  GE_ASSERT_NOTNULL(compute_graph);
  std::vector<int64_t> huge_stream_list;
  (void)AttrUtils::GetListInt(ge_model, ATTR_MODEL_HUGE_STREAM_LIST, huge_stream_list);
  const std::set<int64_t> huge_streams(huge_stream_list.begin(), huge_stream_list.end());
  for (uint32_t i = 0U; i < stream_flags_list.size(); ++i) {
    stream_flags_list[i] = "ACL_STREAM_PERSISTENT";
    if (huge_streams.count(static_cast<int64_t>(i)) > 0) {
      GELOGI("[OM2] Stream %u is huge stream.", i);
      stream_flags_list[i] += " | ACL_STREAM_HUGE";
    }
  }
  return ge::GRAPH_SUCCESS;
}

void ProgramGenerator::GenKernelRegConsts(Program &program) {
  std::string code = R"(constexpr uint32_t kMaxJsonFileLen = 512U;
struct BinaryBuffer {
  std::unique_ptr<uint8_t[]> data;
  size_t size = 0;
};
struct AicoreRegisterInfo {
  uint32_t magic;
  const char *kernel_name;
  std::string file;
};
struct AicpuRegisterInfo {
  const char *op_type;
  const char *so_name;
  const char *kernel_name;
  const char *op_kernel_lib;
};
struct CustAicpuRegisterInfo {
  const char *kernel_name;
  const char *func_name;
  const char *kernel_file;
};)";
  WriteKernelRegSource(program, RAW_CODE_STMT(ast_ctx_, code));
}

void ProgramGenerator::GenKernelRegCommonFuncs(Program &program) {
  std::stringstream code;
  EMIT_CODE(code, R"(BinaryBuffer ReadBinaryFileToBuffer(const std::string &file_path) {
  BinaryBuffer result;
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return result;
  }
  std::streamsize file_size = file.tellg();
  if (file_size <= 0) {
    return result;
  }
  result.size = static_cast<size_t>(file_size);
  result.data = std::make_unique<uint8_t[]>(result.size);
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char *>(result.data.get()), result.size);
  if (!file.good()) {
    file.close();
    result.data.reset();
    result.size = 0;
  }
  return result;
})");
  EMIT_CODE(code, R"(
aclError GenerateJsonFile(const AicpuRegisterInfo &register_info, std::string &json_path) {
  using namespace std::chrono;
  int64_t cur_timestamp = duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
  json_path = "/tmp/temp_ops_info_" + std::to_string(cur_timestamp) + ".json";)");
  EMIT_CODE(code, "  std::string json_data_format = R\"(");
  EMIT_CODE(code, R"({
    "%s":{
        "opInfo":{
            "opKernelLib":"%s",
            "kernelSo":"%s",
            "functionName":"%s"
        }
    }
})");
  EMIT_CODE(code, (")\";"));
  EMIT_CODE(code, R"(  char json_data[kMaxJsonFileLen];
  std::string op_kernel_lib = register_info.op_kernel_lib;
  std::string so_name = register_info.so_name;
  std::string kernel_name = register_info.kernel_name;
  std::string op_type = register_info.op_type;
  auto ret = snprintf_s(json_data, kMaxJsonFileLen, kMaxJsonFileLen - 1U, json_data_format.c_str(),
                        register_info.op_type, register_info.op_kernel_lib, register_info.so_name, register_info.kernel_name);
  OM2_CHK_TRUE(ret >= 0);
  std::ofstream ofs(json_path.c_str(), std::ios::trunc);
  OM2_CHK_TRUE(ofs);
  ofs << json_data;
  return ACL_SUCCESS;
})");
  WriteKernelRegSource(program, RAW_CODE_STMT(ast_ctx_, code.str()));
}

void ProgramGenerator::GenKernelRegFuncsImpl(Program &program) {
  std::string code = R"(void AssembleAicpuLoadOptions(aclrtBinaryLoadOptions &load_options, int32_t cpu_kernel_mode) {
  aclrtBinaryLoadOption option;
  load_options.numOpt = 1;
  load_options.options = &option;
  option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
  option.value.cpuKernelMode = cpu_kernel_mode;
}

aclError RegisterAicoreKernel(aclrtBinHandle &bin_handle, aclrtFuncHandle &func_handle, const AicoreRegisterInfo &register_info, std::unordered_map<std::string, BinDataInfo> &bin_info_map) {
  auto &bin_info = bin_info_map[register_info.file];
  aclrtBinaryLoadOptions load_options;
  aclrtBinaryLoadOption option;
  load_options.numOpt = 1;
  load_options.options = &option;
  option.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
  option.value.magic = register_info.magic;
  OM2_CHK_STATUS(aclrtBinaryLoadFromData(bin_info.data, bin_info.size, &load_options, &bin_handle));
  OM2_CHK_STATUS(aclrtBinaryGetFunction(bin_handle, register_info.kernel_name, &func_handle));
  return ACL_SUCCESS;
}

aclError RegisterAicpuKernel(aclrtBinHandle &bin_handle, aclrtFuncHandle &func_handle, const AicpuRegisterInfo &register_info) {
  std::string json_path;
  OM2_CHK_STATUS(GenerateJsonFile(register_info, json_path));
  OM2_MAKE_GUARD(json_guard, [&json_path]() {
    (void)std::remove(json_path.c_str());
  });
  OM2_CHK_TRUE(!json_path.empty());
  aclrtBinaryLoadOptions load_options;
  aclrtBinaryLoadOption option;
  load_options.numOpt = 1;
  load_options.options = &option;
  option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
  option.value.cpuKernelMode = 0;
  OM2_CHK_STATUS(aclrtBinaryLoadFromFile(json_path.c_str(), &load_options, &bin_handle));
  OM2_CHK_STATUS(aclrtBinaryGetFunction(bin_handle, register_info.op_type, &func_handle));
  return ACL_SUCCESS;
}

aclError RegisterCustAicpuKernel(aclrtBinHandle &bin_handle, aclrtFuncHandle &func_handle, const CustAicpuRegisterInfo &register_info) {
  const auto &kernel_buf = ReadBinaryFileToBuffer(register_info.kernel_file);
  OM2_CHK_TRUE((kernel_buf.size > 0) && (kernel_buf.data != nullptr));
  aclrtBinaryLoadOptions load_options;
  AssembleAicpuLoadOptions(load_options, 2);
  OM2_CHK_STATUS(aclrtBinaryLoadFromData(kernel_buf.data.get(), kernel_buf.size, &load_options, &bin_handle));
  OM2_CHK_STATUS(aclrtRegisterCpuFunc(bin_handle, register_info.func_name, register_info.kernel_name, &func_handle));
  return ACL_SUCCESS;
})";
  WriteKernelRegSource(program, RAW_CODE_STMT(ast_ctx_, code));
}

Status GenBinRegisterCode(AstContext &ast_ctx, std::vector<AstNode *> &nodes, const OpDescPtr &op_desc,
                          const domi::TaskDef &task_def,
                          std::unordered_map<std::string, uint32_t> &func_handle_indices) {
  const auto &kernel_name = task_def.kernel().kernel_name();
  auto task_type = static_cast<ModelTaskType>(task_def.type());
  auto kernel_type = Om2CodegenUtils::IsAllKernel(task_type) ? task_def.kernel_with_handle().context().kernel_type()
                                                      : task_def.kernel().context().kernel_type();
  std::stringstream code_stream;
  if (Om2CodegenUtils::IsAllKernel(task_type) ||
      Om2CodegenUtils::IsAICoreKernel(static_cast<ccKernelType>(kernel_type))) {
    if (func_handle_indices.find(kernel_name) != func_handle_indices.end()) {
      return SUCCESS;
    }
    std::string magic;
    GE_CHK_STATUS_RET(Om2CodegenUtils::GetMagic(op_desc, magic), "Magic value is invalid.");
    auto kernel_file_name = Om2CodegenUtils::GetKernelNameWithExtension(kernel_name);
    uint32_t func_handle_index = func_handle_indices.size();

    code_stream << "  OM2_CHK_STATUS(RegisterAicoreKernel(bin_handles_[" << func_handle_index << "], func_handles_["
                << func_handle_index << "], {" << magic << ", \"" << kernel_name << "\", \"" << kernel_file_name
                << "\"}, bin_info_map_));";
    func_handle_indices[kernel_name] = func_handle_index;
    GELOGI(
        "[OM2] Aicore binary kernel registry code is generated, (kernel_name=%s, kernel_file_name=%s, magic=%s, "
        "func_handle_index=%d)",
        kernel_name.c_str(), kernel_file_name.c_str(), magic.c_str(), func_handle_index);
  } else if (static_cast<ccKernelType>(kernel_type) == ge::ccKernelType::AI_CPU) {
    // 支持内置Aicpu算子
    uint32_t func_handle_index = func_handle_indices.size();
    std::string op_type = op_desc->GetType();
    std::string so_name = task_def.kernel().so_name();
    std::string kernel_name = task_def.kernel().kernel_name();
    std::string op_kernel_lib = "AICPUKernel";
    std::string aicpu_kernel_sign = op_type + kernel_name;
    if (func_handle_indices.find(aicpu_kernel_sign) != func_handle_indices.end()) {
      return SUCCESS;
    }

    code_stream << "  OM2_CHK_STATUS(RegisterAicpuKernel(bin_handles_[" << func_handle_index << "], func_handles_["
                << func_handle_index << "], {\"" << op_type << "\", \"" << so_name << "\", \"" << kernel_name
                << "\", \"" << op_kernel_lib << "\"}));";
    
    func_handle_indices[aicpu_kernel_sign] = func_handle_index;
    GELOGI(
        "[OM2] Aicpu binary kernel registry code is generated, (op_type=%s, so_name=%s, kernel_name=%s, "
        "op_kernel_lib=%s, func_handle_index=%d)",
        op_type.c_str(), so_name.c_str(), kernel_name.c_str(), op_kernel_lib.c_str(), func_handle_index);
  } else {
    REPORT_INNER_ERR_MSG("E19999", "Unsupported task type %d", static_cast<int32_t>(task_type));
    GELOGE(FAILED, "[OM2] Unsupported task type %d, task def %s", static_cast<int32_t>(task_type),
             task_def.ShortDebugString().c_str());
    return FAILED;
  }
  nodes.push_back(RawCodeStmt::Create(ast_ctx, {{RawCodeStmt::LANG_CPP, code_stream.str()}}));
  GELOGD("[OM2] GenBinRegisterCode finished.");
  return SUCCESS;
}

Status ProgramGenerator::GenRegisterKernelsImpl(Program &program) {
  const auto &model_task_def = ge_model_->GetModelTaskDefPtr();
  GE_ASSERT_NOTNULL(model_task_def);
  const size_t task_size = static_cast<size_t>(model_task_def->task_size());
  std::vector<AstNode *> ast_nodes;
  for (size_t i = 0UL; i < task_size; ++i) {
    domi::TaskDef *const task_def = model_task_def->mutable_task(static_cast<int32_t>(i));
    auto &task_code_generator = task_code_generator_list_.at(i);
    GE_ASSERT_NOTNULL(task_code_generator, "Task code generator from type %d and task index %zu has not been created.",
                      task_def->type(), i);
    auto op_index = task_code_generator->ParseOpIndex(*task_def);
    if (op_index != kInvalidOpIndex &&
        Om2CodegenUtils::RequireBinaryKernel(static_cast<ModelTaskType>(task_def->type()))) {
      const OpDescPtr op_desc = FindOpDescByIndex(op_index);
      GE_ASSERT_NOTNULL(op_desc, "[OM2] Failed to find op_desc by op_index %" PRId64 ".", op_index);
      GE_ASSERT_SUCCESS(GenBinRegisterCode(ast_ctx_, ast_nodes, op_desc, *task_def, func_handle_indices_));
    }
  }
  runtime_param_.kernel_bin_num = func_handle_indices_.size();
  WriteKernelRegSource(program, RAW_CODE_STMT(ast_ctx_, "aclError Om2Model::RegisterKernels() {"));
  program[static_cast<size_t>(GeneratedFileIndex::kKernelRegistryFile)].insert(
      program[static_cast<size_t>(GeneratedFileIndex::kKernelRegistryFile)].end(), ast_nodes.begin(), ast_nodes.end());
  WriteKernelRegSource(program, RAW_CODE_STMT(ast_ctx_, "  return ACL_SUCCESS;"));
  WriteKernelRegSource(program, RAW_CODE_STMT(ast_ctx_, "}"));
  GELOGD("[OM2] Om2Model::RegisterKernels() func is successfully generated.");
  return SUCCESS;
}

Status ProgramGenerator::GenAicpuArgsCommon(Program &program) {
  std::string getHostArgsFunc = R"(constexpr uint32_t kAicpuArgsExtInfoAddrOffset = 12U;
constexpr uint32_t kAicpuArgsio_addr_offset = 20U;

aclError UpdateExtInfoSession(uint8_t *extInfo, size_t session_info_offset, uint64_t *session_id, uint64_t *kernel_id) {
  AicpuSessionInfo *session_info = reinterpret_cast<AicpuSessionInfo *>(&(extInfo[session_info_offset]));
  session_info->sessionId = *session_id;
  session_info->kernelId = *kernel_id;
  session_info->sessFlag = true;
  (*kernel_id)++;
  return ACL_SUCCESS;
}
aclError AssembleAicpuExtInfo(uint8_t *ext_info, size_t ext_info_len, int32_t session_info_offset, uint64_t *session_id, uint64_t *kernel_id, std::vector<void *> &dev_ext_info_mem_ptrs, size_t index) {
  std::unique_ptr<uint8_t[]> tmp_ext_info = std::make_unique<uint8_t[]>(ext_info_len);
  memcpy_s(tmp_ext_info.get(), ext_info_len, ext_info, ext_info_len);
  if (session_info_offset != -1) {
    OM2_CHK_STATUS(UpdateExtInfoSession(tmp_ext_info.get(), session_info_offset, session_id, kernel_id));
  }
  void *dev_ptr = nullptr;
  OM2_CHK_STATUS(aclrtMallocAlign32(&(dev_ptr), ext_info_len, ACL_MEM_MALLOC_HUGE_FIRST));
  OM2_CHK_STATUS(aclrtMemcpy(dev_ptr, ext_info_len, tmp_ext_info.get(), ext_info_len, ACL_MEMCPY_HOST_TO_DEVICE));
  dev_ext_info_mem_ptrs[index] = dev_ptr;
  return ACL_SUCCESS;
}
aclError AssembleAicpuArgs(uint8_t *args, size_t args_len, void *ext_info_addr, size_t ext_info_len, std::vector<uint64_t> &io_addr, void *target_args_ptr) {
  std::unique_ptr<uint8_t[]> tmp_args = std::make_unique<uint8_t[]>(args_len);
  memcpy_s(tmp_args.get(), args_len, args, args_len);
  const auto aicpu_param_head = reinterpret_cast<AicpuParamHead*>(tmp_args.get());
  aicpu_param_head->extInfoLength = static_cast<uint32_t>(ext_info_len);
  uint64_t ext_info_addr_value = reinterpret_cast<uint64_t>(ext_info_addr);
  memcpy_s(tmp_args.get() + kAicpuArgsExtInfoAddrOffset, sizeof(uint64_t), &(ext_info_addr_value), sizeof(uint64_t));
  size_t addrs_size = sizeof(uint64_t) * io_addr.size();
  memcpy_s(tmp_args.get() + kAicpuArgsio_addr_offset, addrs_size, io_addr.data(), addrs_size);
  memcpy_s(target_args_ptr, args_len, tmp_args.get(), args_len);
  return ACL_SUCCESS;
}
aclError AicpuKernelTaskDistribute(const std::vector<uint8_t>& args, ArgsInfo *args_info, aclrtFuncHandle func_handle,
                              uint32_t block_dim, aclrtStream stream, aclrtLaunchKernelCfg *config) {
  OM2_CHK_NOTNULL(args_info);
  OM2_CHK_STATUS(memcpy_s(args_info->host_addr, args_info->size, args.data(), args.size()));
  OM2_CHK_STATUS(
    aclrtLaunchKernelV2(
      func_handle,
      block_dim,
      args_info->dev_addr,
      args_info->size,
      config,
      stream
    )
  );
  return ACL_SUCCESS;
}  
)";
  WriteLoadAndRunSource(program, RAW_CODE_STMT(ast_ctx_, getHostArgsFunc));
  return SUCCESS;
}

Status ProgramGenerator::GenConstInputs(std::vector<AstNode *> &const_input_ast_nodes, const OpDescPtr &op_desc,
                                        std::unordered_map<int64_t, std::string> &weight_offset_to_varname) {
  const vector_bit_t &v_is_input_const = op_desc->GetIsInputConst();
  for (size_t i = 0U; i < op_desc->GetAllInputsSize(); ++i) {
    if ((i < v_is_input_const.size()) && v_is_input_const[i]) {
      const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
      if (tensor_desc == nullptr) {
        continue;
      }
      // 使用与 GenInputAddrCode 相同的方式获取 data_offset
      int64_t data_offset = 0;
      GE_CHK_STATUS(TensorUtils::GetDataOffset(*tensor_desc, data_offset));

      if (weight_offset_to_varname.find(data_offset) != weight_offset_to_varname.end()) {
        continue;
      }
      std::string input_ptr_name = "const_" + std::to_string(weight_offset_to_varname.size());
      const_input_ast_nodes.push_back(RAW_CODE_STMT(ast_ctx_, "  auto " + input_ptr_name + " = GET_ADDR(total_weight_mem_ptr_, " +
        std::to_string(data_offset) + ");"));
      weight_offset_to_varname[data_offset] = input_ptr_name;
    }
  }
  return SUCCESS;
}

Status BuildOpInputEdges(const ComputeGraphPtr &compute_graph,
                         std::unordered_map<int64_t, OpInputEdges> &op_id_to_input_edges) {
  GE_ASSERT_NOTNULL(compute_graph);
  for (const auto &node : compute_graph->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    int64_t op_id = op_desc->GetId();
    OpInputEdges edges;
    const size_t all_inputs_size = op_desc->GetAllInputsSize();
    edges.input_op_ids.resize(all_inputs_size, kInvalidOpId);
    edges.input_anchor_indices.resize(all_inputs_size, kInvalidAnchorIndex);
    edges.output_var_names.resize(op_desc->GetOutputsSize());
    op_id_to_input_edges[op_id] = edges;
  }
  for (const auto &node : compute_graph->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    int64_t op_id = op_desc->GetId();
    const auto &in_anchors = node->GetAllInDataAnchorsPtr();
    for (size_t i = 0; i < in_anchors.size(); ++i) {
      InDataAnchor *in_anchor = in_anchors[i];
      if (in_anchor == nullptr) {
        continue;
      }
      OutDataAnchorPtr peer_out_anchor = in_anchor->GetPeerOutAnchor();
      if (peer_out_anchor == nullptr) {
        continue;
      }
      NodePtr src_node = peer_out_anchor->GetOwnerNode();
      const auto &src_op_desc = src_node->GetOpDesc();
      if (src_op_desc == nullptr) {
        continue;
      }
      int64_t src_op_id = src_op_desc->GetId();
      int32_t src_anchor_idx = peer_out_anchor->GetIdx();
      auto it = op_id_to_input_edges.find(op_id);
      GE_ASSERT_TRUE(it != op_id_to_input_edges.end(), "[OM2] Op id %" PRId64 " not found in mapping", op_id);
      GE_ASSERT_TRUE(i < it->second.input_op_ids.size(),
                     "[OM2] Input index %zu out of range for op_id %" PRId64, i, op_id);
      it->second.input_op_ids[i] = src_op_id;
      it->second.input_anchor_indices[i] = src_anchor_idx;
    }
  }
  return SUCCESS;
}

Status ProgramGenerator::GenArgsTableImpl(std::vector<AstNode *> &ast_nodes) {
  std::string InitHead =
      R"(aclError Om2ArgsTable::Init() {
  args_size_ = )" + std::to_string(args_info_.host_args_len) + R"(;
  host_args_.clear();)";
  ast_nodes.push_back(RAW_CODE_STMT(ast_ctx_, InitHead));
  ast_nodes.push_back(RAW_CODE_STMT(ast_ctx_, "  host_args_.resize(args_size_);"));
  ast_nodes.push_back(RAW_CODE_STMT(ast_ctx_,
    "  OM2_CHK_STATUS(aclrtMalloc(&dev_args_, args_size_, ACL_MEM_MALLOC_HUGE_FIRST));"));
  
  std::stringstream code_stream;
  EMIT_CODE(code_stream, "  args_info_ = {");
  for (size_t i = 0UL; i < args_info_.args_offset.size(); ++i) {
    EMIT_CODE(code_stream,
              "    {GetHostArgAddr(" + std::to_string(args_info_.args_offset[i]) + "), GetDevArgAddr(" + std::
              to_string(args_info_.args_offset[i]) + "), "
              + std::to_string(args_info_.args_sizes[i]) + "},");
  }
  EMIT_CODE(code_stream, "  };");

  EMIT_CODE(code_stream, "  iow_args_addrs_ = {");
  for (size_t i = 0UL; i < args_info_.io_addr_offset.size(); ++i) {
    EMIT_CODE(code_stream, "    GetHostArgAddr(" + std::to_string(args_info_.io_addr_offset[i]) +"),");
  }
  EMIT_CODE(code_stream, "  };");
  EMIT_CODE(code_stream, "  return ACL_SUCCESS;");
  EMIT_CODE(code_stream, "}");
  ast_nodes.push_back(RAW_CODE_STMT(ast_ctx_, code_stream.str()));

  // GetHostArgs
  std::string getHostArgsFunc = R"(Om2ArgsTable::~Om2ArgsTable() {
}

ArgsInfo *Om2ArgsTable::GetArgsInfo(size_t index) {
  if (index >= args_info_.size()) {
    return nullptr;
  }
  return &args_info_[index];
}

void *Om2ArgsTable::GetDevArgAddr(size_t offset) {
  if (offset >= args_size_) {
    return nullptr;
  }
  return GET_ADDR(dev_args_, offset);
}

void *Om2ArgsTable::GetHostArgAddr(size_t offset) {
  if (offset >= args_size_) {
    return nullptr;
  }
  return GET_ADDR(host_args_.data(), offset);
}

aclError Om2ArgsTable::CopyArgsToDevice() {
  OM2_CHK_STATUS(aclrtMemcpy(dev_args_, args_size_, host_args_.data(), args_size_, ACL_MEMCPY_HOST_TO_DEVICE));
  return ACL_SUCCESS;
}
)";
  ast_nodes.push_back(RAW_CODE_STMT(ast_ctx_, getHostArgsFunc));
  return SUCCESS;
}

Status ProgramGenerator::Init(const GeModelPtr &model) {
  ge_model_ = model;
  runtime_param_ = {};
  op_list_.clear();
  task_code_generator_list_.clear();
  func_handle_indices_.clear();
  args_info_ = {};
  aicpu_task_num_ = 0U;
  model_io_offsets_.clear();

  GE_ASSERT_NOTNULL(ge_model_);
  GE_ASSERT_SUCCESS(PrepareGraphData());
  GE_ASSERT_SUCCESS(InitRuntimeParams());
  GE_ASSERT_SUCCESS(CreateTaskCodeGenerators());
  return SUCCESS;
}

Status ProgramGenerator::PrepareGraphData() {
  const auto compute_graph = ge_model_->GetGraph();
  GE_ASSERT_NOTNULL(compute_graph);

  for (const auto &node : compute_graph->GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    GE_ASSERT_TRUE(!Om2CodegenUtils::IsUnsupportedNodeType(op_desc->GetType()),
                   "[OM2] Unsupported node type %s, op_name=%s", op_desc->GetType().c_str(),
                   op_desc->GetName().c_str());
    op_list_[op_desc->GetId()] = op_desc;
  }
  return SUCCESS;
}

Status ProgramGenerator::InitRuntimeParams() {
  (void)AttrUtils::GetInt(ge_model_, ATTR_MODEL_MEMORY_SIZE, runtime_param_.mem_size);
  (void)AttrUtils::GetInt(ge_model_, ATTR_MODEL_WEIGHT_SIZE, runtime_param_.weight_size);
  (void)AttrUtils::GetInt(ge_model_, ATTR_MODEL_STREAM_NUM, runtime_param_.stream_num);
  (void)AttrUtils::GetInt(ge_model_, ATTR_MODEL_NOTIFY_NUM, runtime_param_.notify_num);
  (void)AttrUtils::GetInt(ge_model_, ATTR_MODEL_EVENT_NUM, runtime_param_.event_num);
  (void)AttrUtils::GetInt(ge_model_, ATTR_MODEL_LABEL_NUM, runtime_param_.label_num);
  return SUCCESS;
}

Status ProgramGenerator::CreateTaskCodeGenerators() {
  const auto &model_task_def = ge_model_->GetModelTaskDefPtr();
  GE_ASSERT_NOTNULL(model_task_def);
  const size_t task_size = static_cast<size_t>(model_task_def->task_size());
  task_code_generator_list_.resize(task_size);
  for (size_t i = 0UL; i < task_size; ++i) {
    domi::TaskDef *const task_def = model_task_def->mutable_task(static_cast<int32_t>(i));
    auto task_type = static_cast<ModelTaskType>(task_def->type());
    auto &task_code_generator = task_code_generator_list_.at(i);
    task_code_generator = TaskCodeGeneratorFactory::Instance().Create(task_type);
    if (!Om2CodegenUtils::IsSupportedTask(task_type)) {
      const auto op_index =
          (task_code_generator == nullptr) ? kInvalidOpIndex : task_code_generator->ParseOpIndex(*task_def);
      if (task_code_generator == nullptr || op_index == kInvalidOpIndex) {
        REPORT_INNER_ERR_MSG("E19999", "Unsupported task type %d", static_cast<int32_t>(task_type));
        GELOGE(FAILED, "[OM2] Unsupported task type %d, task def %s", static_cast<int32_t>(task_type),
               task_def->ShortDebugString().c_str());
      } else {
        const auto op_desc = FindOpDescByIndex(op_index);
        if (op_desc == nullptr) {
          REPORT_INNER_ERR_MSG("E19999", "Unsupported task type %d, op_index=%" PRId64,
                               static_cast<int32_t>(task_type), op_index);
          GELOGE(FAILED, "[OM2] Unsupported task type %d, op_index=%" PRId64 ", task def %s",
                 static_cast<int32_t>(task_type), op_index, task_def->ShortDebugString().c_str());
        } else {
          REPORT_INNER_ERR_MSG("E19999", "Unsupported task type %d for op %s, op_index=%" PRId64,
                               static_cast<int32_t>(task_type), op_desc->GetName().c_str(), op_index);
          GELOGE(FAILED, "[OM2] Unsupported task type %d for op %s, op type %s, op_index=%" PRId64 ", task def %s",
                 static_cast<int32_t>(task_type), op_desc->GetName().c_str(), op_desc->GetTypePtr(), op_index,
                 task_def->ShortDebugString().c_str());
        }
      }
      return FAILED;
    }
    GE_ASSERT_NOTNULL(task_code_generator, "Failed to create task code generator from type %d, task index %zu",
                      task_def->type(), i);
    GELOGD("[OM2] TaskCodeGenerator for type %d is successfully created.", task_type);
  }
  return SUCCESS;
}

OpDescPtr ProgramGenerator::FindOpDescByIndex(int64_t op_index) const {
  const auto it = op_list_.find(op_index);
  return (it == op_list_.end()) ? nullptr : it->second;
}

void ProgramGenerator::GenInterfaceHeaderMacros(Program &program) {
  std::string macros = R"(#define OM2_CHK_STATUS(expr, ...)            \
do {                                       \
  const aclError _chk_status = (expr);     \
  if (_chk_status != ACL_SUCCESS) {        \
    return _chk_status;                    \
  }                                        \
} while (false)

#define OM2_CHK_NOTNULL(ptr, ...)            \
do {                                       \
  if ((ptr) == nullptr) {                  \
    return ACL_ERROR_FAILURE;              \
  }                                        \
} while (false)

#define OM2_CHK_TRUE(expr, ...)              \
do {                                       \
  if (!(expr)) {                           \
    return ACL_ERROR_FAILURE;              \
  }                                        \
} while (false)

#define GET_ADDR(mem_ptr, offset)   \
(reinterpret_cast<void *>(                 \
  reinterpret_cast<uintptr_t>(mem_ptr) +   \
  static_cast<uintptr_t>(offset)))

#define OM2_MAKE_GUARD(var, callback) const ::om2::ScopeGuard const_guard_##var(callback)
)";
  WriteInterfaceHeader(program, RAW_CODE_STMT(ast_ctx_, macros));
}

void ProgramGenerator::GenInterfaceHeaderCommonFunc(Program &program) {
  std::string funcs = R"(template<typename T>
inline T *PtrAdd(T *const ptr, const size_t max_buf_len, const size_t idx) {
  if ((ptr != nullptr) && (idx < max_buf_len)) {
    return reinterpret_cast<T *>(ptr + idx);
  }
  return nullptr;
}
template<typename TI, typename TO>
inline TO *PtrToPtr(TI *const ptr) {
  return reinterpret_cast<TO *>(ptr);
}
inline uint64_t PtrToValue(const void *const ptr) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
}
inline void *ValueToPtr(const uint64_t value) {
  return reinterpret_cast<void *>(static_cast<uintptr_t>(value));
}

template<typename... Args>
inline std::vector<uint64_t> FlattenHostArgs(Args&&... args) {
  std::vector<uint64_t> buf;
  auto append_arg = [&](auto&& arg) {
    using T = std::decay_t<decltype(arg)>;
    if constexpr (std::is_pointer_v<T>) {
      buf.push_back(reinterpret_cast<uint64_t>(arg));
    } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
      for (auto d : arg) buf.push_back(static_cast<uint64_t>(d));
    } else if constexpr (std::is_integral_v<T>) {
      buf.push_back(static_cast<uint64_t>(arg));
    } else {
      static_assert(sizeof(T) == 0, "Unsupported type in FlattenHostArgs");
    }
  };
  (append_arg(std::forward<Args>(args)), ...);
  return buf;
}
)";
  WriteInterfaceHeader(program, RAW_CODE_STMT(ast_ctx_, funcs));
}

Status ProgramGenerator::GenInterfaceHeaderCommonPart(Program &program) {
  size_t input_num = 0U;
  GE_ASSERT_SUCCESS(GetModelInputNumber(ge_model_, input_num));
  size_t output_num = 0U;
  GE_ASSERT_SUCCESS(GetModelOutputNumber(ge_model_, output_num));
  std::string content = R"(constexpr int32_t INPUT_NUM = )" + std::to_string(input_num) + R"(;
constexpr int32_t OUTPUT_NUM = )" +
                        std::to_string(output_num) + R"(;
typedef void *Om2ModelHandle;
typedef void *GeTensorHandle;

struct BinDataInfo {
  const void *data;
  size_t size;
};
struct AicpuParamHead {
  uint32_t length;
  uint32_t ioAddrNum;
  uint32_t extInfoLength;
  uint64_t extInfoAddr;
};
struct AicpuSessionInfo {
  uint64_t sessionId;
  uint64_t kernelId;
  bool sessFlag;
};
struct ArgsInfo {
  void *host_addr;
  void *dev_addr;
  size_t size;
};

class ScopeGuard {
 public:
  // Noncopyable
  ScopeGuard(const ScopeGuard &) = delete;
  ScopeGuard &operator=(const ScopeGuard &) = delete;

  explicit ScopeGuard(const std::function<void()> &on_exit_scope)
      : on_exit_scope_(on_exit_scope) {}

  ~ScopeGuard() {
    if (on_exit_scope_) {
      try {
        on_exit_scope_();
      } catch (std::bad_function_call &) {
      } catch (...) {
      }
    }
  }

 private:
  std::function<void()> on_exit_scope_;
};)";
  WriteInterfaceHeader(program, RAW_CODE_STMT(ast_ctx_, content));
  return SUCCESS;
}

void ProgramGenerator::GenInterfaceHeaderOm2ArgsTableClass(Program &program) {
  std::string om2ArgsTableStr = R"(class Om2ArgsTable {
 public:
  Om2ArgsTable() = default;
  ~Om2ArgsTable();
  aclError Init();
  ArgsInfo *GetArgsInfo(size_t index);
  void *GetDevArgAddr(size_t offset);
  void *GetHostArgAddr(size_t offset);
  aclError CopyArgsToDevice();
 private:
  int64_t args_size_;
  std::vector<ArgsInfo> args_info_;
  std::vector<uint8_t> host_args_;
  void *dev_args_;
  std::vector<void *> iow_args_addrs_;
};)";
  WriteInterfaceHeader(program, RAW_CODE_STMT(ast_ctx_, om2ArgsTableStr));
}

void ProgramGenerator::GenInterfaceHeaderOm2ModelClass(Program &program) {
  std::stringstream code;
  EMIT_CODE(code, R"(
class Om2Model {
 public:
  Om2Model(const char **bin_files, const void **bin_data, size_t *bin_size, size_t bin_num, void *host_weight_mem_ptr, uint64_t *session_id);
  ~Om2Model();
  aclError InitResources();
  aclError RegisterKernels();
  aclError Load();
  aclError Run(size_t input_count, void **input_data, size_t output_count, void **output_data);
  aclError RunAsync(aclrtStream &exe_stream, size_t input_count, void **input_data, size_t output_count, void **output_data);
  aclError ReleaseResources();
 private:
  void *host_weight_mem_ptr_;
  aclmdlRI model_handle_;)");
  if (runtime_param_.kernel_bin_num > 0U) {
    EMIT_CODE(code, "  std::vector<aclrtBinHandle> bin_handles_;");
    EMIT_CODE(code, "  std::vector<aclrtFuncHandle> func_handles_;");
  }
  if (runtime_param_.stream_num > 0U) {
    EMIT_CODE(code, "  std::vector<aclrtStream> stream_list_;");
  }
  if (runtime_param_.notify_num > 0U) {
    EMIT_CODE(code, "  std::vector<aclrtNotify> notify_list_;");
  }
  if (runtime_param_.event_num > 0U) {
    EMIT_CODE(code, "  std::vector<aclrtEvent> event_list_;");
  }
  if (runtime_param_.label_num > 0U) {
    EMIT_CODE(code, "  std::vector<aclrtLabel> label_list_;");
    EMIT_CODE(code, "  aclrtLabelList aclrt_label_list_;");
  }
  EMIT_CODE(code, R"(  void *total_dev_mem_ptr_;
  void *total_weight_mem_ptr_;
  bool is_stream_list_bind_;
  std::unordered_map<std::string, BinDataInfo> bin_info_map_;
  Om2ArgsTable args_table_;
  uint64_t *session_id_;
  uint64_t kernel_id_;
  std::vector<void *> dev_ext_info_mem_ptrs_;
};)");
  WriteInterfaceHeader(program, RAW_CODE_STMT(ast_ctx_, code.str()));
  GELOGD("[OM2] Om2Model class definition is generated.");
}

void ProgramGenerator::GenInterfaceHeaderExternalApi(Program &program) {
  std::string content = R"(#ifdef __cplusplus
extern "C" {
#endif

aclError Om2ModelCreate(om2::Om2ModelHandle* model_handle, const char** bin_files, const void ** bin_data, size_t *bin_size, int bin_num, void *host_weight_mem_ptr, uint64_t *session_id);
aclError Om2ModelRunAsync(om2::Om2ModelHandle* model_handle, aclrtStream stream, int input_count, void **input_data, int output_count, void **output_data);
aclError Om2ModelRun(om2::Om2ModelHandle* model_handle, int input_count, void **input_data, int output_count, void **output_data);
aclError Om2ModelDestroy(om2::Om2ModelHandle* model_handle);

#ifdef __cplusplus
}
#endif
)";
  WriteInterfaceHeader(program, RAW_CODE_STMT(ast_ctx_, content));
}

void ProgramGenerator::GenOm2ModelConstructor(Program &program) {
  std::stringstream code;
  EMIT_CODE(
      code,
      R"(Om2Model::Om2Model(const char **bin_files, const void **bin_data, size_t *bin_size, size_t bin_num, void *host_weight_mem_ptr, uint64_t *session_id)
  : host_weight_mem_ptr_(host_weight_mem_ptr), session_id_(session_id), kernel_id_(0) {
  for (size_t i = 0; i < bin_num; ++i) {
    bin_info_map_[std::string(bin_files[i])] = BinDataInfo{bin_data[i], bin_size[i]};
  })");
  if (runtime_param_.kernel_bin_num > 0U) {
    EMIT_CODE(code, "  bin_handles_.resize(" + std::to_string(runtime_param_.kernel_bin_num) + ");");
    EMIT_CODE(code, "  func_handles_.resize(" + std::to_string(runtime_param_.kernel_bin_num) + ");");
  }
  if (runtime_param_.stream_num > 0U) {
    EMIT_CODE(code, "  stream_list_.resize(" + std::to_string(runtime_param_.stream_num) + ");");
  }
  if (runtime_param_.notify_num > 0U) {
    EMIT_CODE(code, "  notify_list_.resize(" + std::to_string(runtime_param_.notify_num) + ");");
  }
  if (runtime_param_.event_num > 0U) {
    EMIT_CODE(code, "  event_list_.resize(" + std::to_string(runtime_param_.event_num) + ");");
  }
  if (runtime_param_.label_num > 0U) {
    EMIT_CODE(code, "  label_list_.resize(" + std::to_string(runtime_param_.label_num) + ");");
  }
  EMIT_CODE(code, "}");
  WriteResourcesSource(program, RAW_CODE_STMT(ast_ctx_, code.str()));
  GELOGD("[OM2] Om2Model constructor implementation is generated.");
}

void ProgramGenerator::GenOm2ModelDestructor(Program &program) {
  std::string content = R"(Om2Model::~Om2Model() {
  (void)ReleaseResources();
})";
  WriteResourcesSource(program, RAW_CODE_STMT(ast_ctx_, content));
  GELOGD("[OM2] Om2Model destructor implementation is generated.");
}

Status ProgramGenerator::GenInitResourcesImpl(Program &program) {
  auto model_task_def = ge_model_->GetModelTaskDefPtr();
  GE_ASSERT_NOTNULL(model_task_def);

  std::stringstream code;
  EMIT_CODE(code, R"(aclError Om2Model::InitResources() {
  // 1. 创建 model
  OM2_CHK_STATUS(aclmdlRIBuildBegin(&model_handle_, 0));

  // 2. 申请内存
  OM2_CHK_STATUS(aclrtMallocAlign32(&total_dev_mem_ptr_, )" +
                      std::to_string(model_task_def->memory_size()) + R"(, ACL_MEM_MALLOC_HUGE_FIRST));
  OM2_CHK_STATUS(aclrtMallocAlign32(&total_weight_mem_ptr_, )" +
                      std::to_string(model_task_def->weight_size()) + R"(, ACL_MEM_MALLOC_HUGE_FIRST));

  // 3. 下沉权重
  OM2_CHK_STATUS(aclrtMemcpy(total_weight_mem_ptr_, )" +
                      std::to_string(model_task_def->weight_size()) + R"(, host_weight_mem_ptr_, )" +
                      std::to_string(model_task_def->weight_size()) + R"(, ACL_MEMCPY_HOST_TO_DEVICE));
)");
  EMIT_CODE(code, "  // 4. 创建其他资源");
  if (runtime_param_.stream_num > 0U) {
    EMIT_CODE(code, "  // 创建下沉Stream并绑定模型");
    std::vector<std::string> stream_flags_list(runtime_param_.stream_num);
    GE_ASSERT_SUCCESS(GetStreamFlags(ge_model_, stream_flags_list));
    std::stringstream create_and_bind_stream_content;
    for (uint32_t i = 0U; i < runtime_param_.stream_num; ++i) {
      const std::string &flag_var_name = "stream" + std::to_string(i) + "_flag";
      create_and_bind_stream_content << "  uint32_t " << flag_var_name << " = " << stream_flags_list[i] << ";"
                                     << std::endl;
      create_and_bind_stream_content << "  OM2_CHK_STATUS(aclrtCreateStreamWithConfig(&stream_list_[" << i << "], 0, "
                                     << flag_var_name << "));" << std::endl;
      create_and_bind_stream_content << "  OM2_CHK_STATUS(aclmdlRIBindStream(model_handle_, stream_list_[" << i
                                     << "], ACL_MODEL_STREAM_FLAG_HEAD));" << std::endl;
    }
    EMIT_CODE(code, create_and_bind_stream_content.str());
    EMIT_CODE(code, "  is_stream_list_bind_ = true;");
  }
  if (runtime_param_.notify_num > 0U) {
    EMIT_CODE(code, "  // 创建Notify");
    EMIT_CODE(code, "  for (size_t i = 0; i < " + std::to_string(runtime_param_.notify_num) + "; ++i) {");
    EMIT_CODE(code, "    OM2_CHK_STATUS(aclrtCreateNotify(&notify_list_[i], ACL_NOTIFY_DEVICE_USE_ONLY));");
    EMIT_CODE(code, "  }");
  }
  if (runtime_param_.event_num > 0U) {
    EMIT_CODE(code, "  // 创建Event");
    EMIT_CODE(code, "  for (size_t i = 0; i < " + std::to_string(runtime_param_.event_num) + "; ++i) {");
    EMIT_CODE(code, "    OM2_CHK_STATUS(aclrtCreateEventWithFlag(&event_list_[i], ACL_EVENT_SYNC));");
    EMIT_CODE(code, "  }");
  }
  if (runtime_param_.label_num > 0U) {
    EMIT_CODE(code, "  // 创建Label");
    EMIT_CODE(code, "  for (size_t i = 0; i < " + std::to_string(runtime_param_.label_num) + "; ++i) {");
    EMIT_CODE(code, "    OM2_CHK_STATUS(aclrtCreateLabel(&label_list_[i]));");
    EMIT_CODE(code, "  }");
    EMIT_CODE(code,
              "  OM2_CHK_STATUS(aclrtCreateLabelList(label_list_.data(), label_list_.size(), &aclrt_label_list_));");
  }
  EMIT_CODE(code, "  args_table_.Init();");
  EMIT_CODE(code, "  return ACL_SUCCESS;");
  EMIT_CODE(code, "}");
  WriteResourcesSource(program, RAW_CODE_STMT(ast_ctx_, code.str()));
  GELOGD(
      "[OM2] Om2Model::InitResources() implementation is generated, resource summary: "
      "kernel_bins=%u, streams=%u, notifies=%u, events=%u, labels=%u",
      runtime_param_.kernel_bin_num, runtime_param_.stream_num, runtime_param_.notify_num,
      runtime_param_.event_num, runtime_param_.label_num);
  return SUCCESS;
}

void ProgramGenerator::GenReleaseResourcesImpl(Program &program) {
  std::stringstream code;
  EMIT_CODE(code, "aclError Om2Model::ReleaseResources() {");
  if (runtime_param_.label_num > 0U) {
    EMIT_CODE(code, R"(
  for (auto label : label_list_) {
    if (label != nullptr) {
      OM2_CHK_STATUS(aclrtDestroyLabel(label));
    }
  }
  OM2_CHK_STATUS(aclrtDestroyLabelList(aclrt_label_list_));)");
  }
  if (runtime_param_.event_num > 0U) {
    EMIT_CODE(code, R"(
  for (auto event : event_list_) {
    OM2_CHK_STATUS(aclrtDestroyEvent(event));
  })");
  }
  if (runtime_param_.notify_num > 0U) {
    EMIT_CODE(code, R"(
  for (auto notify : notify_list_) {
    OM2_CHK_STATUS(aclrtDestroyNotify(notify));
  })");
  }
  if (runtime_param_.stream_num > 0U) {
    EMIT_CODE(code, R"(
  if (is_stream_list_bind_) {
    for (auto stream : stream_list_) {
      OM2_CHK_STATUS(aclmdlRIUnbindStream(model_handle_, stream));
    }
  }
  for (auto stream : stream_list_) {
    OM2_CHK_STATUS(aclrtDestroyStream(stream));
  })");
  }
  if (runtime_param_.kernel_bin_num > 0U) {
    EMIT_CODE(code, R"(
  for (auto bin_handle : bin_handles_) {
    OM2_CHK_STATUS(aclrtBinaryUnLoad(bin_handle));
  })");
  }
  EMIT_CODE(code, R"(
  OM2_CHK_STATUS(aclmdlRIDestroy(model_handle_));
  OM2_CHK_STATUS(aclrtFree(total_dev_mem_ptr_));
  OM2_CHK_STATUS(aclrtFree(total_weight_mem_ptr_));
  for (int i = 0; i < dev_ext_info_mem_ptrs_.size(); i++) {
    if (dev_ext_info_mem_ptrs_[i] != nullptr) {
      OM2_CHK_STATUS(aclrtFree(dev_ext_info_mem_ptrs_[i]));
    }
  }
  return ACL_SUCCESS;
})");
  WriteResourcesSource(program, RAW_CODE_STMT(ast_ctx_, code.str()));
  GELOGD("[OM2] Om2Model::ReleaseResources() implementation is generated.");
}

Status ProgramGenerator::GenLoadImpl(std::vector<AstNode *> &load_impl_code,
                                     std::vector<AstNode *> &dist_impl_code) {
  args_table_index_ = 0U;
  load_impl_code.push_back(RAW_CODE_STMT(ast_ctx_, "aclError Om2Model::Load() {"));
  const ComputeGraphPtr compute_graph = ge_model_->GetGraph();
  const auto &model_task_def = ge_model_->GetModelTaskDefPtr();
  GE_ASSERT_NOTNULL(model_task_def);
  const size_t task_size = static_cast<size_t>(model_task_def->task_size());
  std::vector<AstNode *> distribution_code;
  std::unordered_set<ModelTaskType> model_task_types;

  std::unordered_map<int64_t, OpInputEdges> op_id_to_input_edges;
  GE_ASSERT_SUCCESS(BuildOpInputEdges(compute_graph, op_id_to_input_edges));
  std::unordered_map<int64_t, std::string> weight_offset_to_varname;
  std::vector<AstNode *> const_input_ast_nodes;
  for (size_t i = 0UL; i < task_size; ++i) {
    domi::TaskDef *const task_def = model_task_def->mutable_task(static_cast<int32_t>(i));
    auto task_type = static_cast<ModelTaskType>(task_def->type());
    auto &task_code_generator = task_code_generator_list_.at(i);
    GE_ASSERT_NOTNULL(task_code_generator, "Task code generator from type %d and task index %zu has not been created.",
                      task_def->type(), i);
    const auto op_index = task_code_generator->ParseOpIndex(*task_def);
    if (model_task_types.find(task_type) == model_task_types.end()) {
      TaskDistributionImplContext task_codegen_ctx = {.ast_ctx = ast_ctx_, .nodes = dist_impl_code};
      GE_ASSERT_SUCCESS(task_code_generator->GenDistributionImplCode(task_codegen_ctx));
      model_task_types.insert(task_type);
    }
    if (op_index == kInvalidOpIndex) {
      TaskDistributionContext task_dist_ctx = {
        ast_ctx_, distribution_code, nullptr,
        *task_def, op_index, func_handle_indices_, op_id_to_input_edges, weight_offset_to_varname,
        runtime_param_, aicpu_task_num_, args_info_, args_table_index_, model_io_offsets_
      };
      GE_ASSERT_SUCCESS(task_code_generator->GenTaskDistributionCode(task_dist_ctx));
    } else {
      const OpDescPtr op_desc = FindOpDescByIndex(op_index);
      GE_ASSERT_NOTNULL(op_desc, "[OM2] Failed to find op_desc by op_index %" PRId64 ".", op_index);
      GE_ASSERT_SUCCESS(GenConstInputs(const_input_ast_nodes, op_desc, weight_offset_to_varname));
      TaskDistributionContext task_dist_ctx = {
        ast_ctx_, distribution_code, op_desc,
        *task_def, op_index, func_handle_indices_, op_id_to_input_edges, weight_offset_to_varname,
        runtime_param_, aicpu_task_num_, args_info_, args_table_index_, model_io_offsets_
      };
      auto kernel_type = Om2CodegenUtils::IsAllKernel(task_type)
                             ? task_def->kernel_with_handle().context().kernel_type()
                             : task_def->kernel().context().kernel_type();
      if (static_cast<ccKernelType>(kernel_type) == ge::ccKernelType::AI_CPU) {
        aicpu_task_num_++;
      }
      GE_ASSERT_SUCCESS(task_code_generator->GenTaskDistributionCode(task_dist_ctx));
    }
    GELOGI("[OM2] Task launch code is generated, op_index=%" PRId64 ", task_type=%d", op_index, task_type);
  }
  load_impl_code.push_back(RAW_CODE_STMT(ast_ctx_, "  dev_ext_info_mem_ptrs_.resize(" + std::to_string(aicpu_task_num_) + ");"));
  load_impl_code.insert(load_impl_code.end(), const_input_ast_nodes.begin(), const_input_ast_nodes.end());
  load_impl_code.insert(load_impl_code.end(), distribution_code.begin(), distribution_code.end());
  std::stringstream code;
  EMIT_CODE(code, "  OM2_CHK_STATUS(aclmdlRIBuildEnd(model_handle_, nullptr));");
  EMIT_CODE(code, "  return ACL_SUCCESS;");
  EMIT_CODE(code, "}");
  load_impl_code.push_back(RAW_CODE_STMT(ast_ctx_, code.str()));
  GELOGD("[OM2] Om2Model::Load() implementation is generated.");
  return SUCCESS;
}

Status ProgramGenerator::GenRunImpl(std::vector<AstNode *> &load_impl_code) {
  std::stringstream input_memcpy_tensor_data_content;
  std::stringstream output_memcpy_tensor_data_content;
  std::stringstream input_memcpy_tensor_data_async_content;
  std::stringstream output_memcpy_tensor_data_async_content;
  auto compute_graph = ge_model_->GetGraph();
  GE_ASSERT_NOTNULL(compute_graph);
  uint32_t input_data_index = 0U;
  uint32_t output_data_index = 0U;
  for (const auto &node : compute_graph->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    if (OpTypeUtils::IsDataNode(op_desc->GetType())) {
      uint32_t tmp_index = input_data_index;
      if (AttrUtils::GetInt(op_desc, ATTR_NAME_INDEX, tmp_index)) {
        GELOGD("[OM2] Get new index %u, old %u", tmp_index, input_data_index);
      }
      auto output_offsets = op_desc->GetOutputOffset();
      GE_ASSERT_TRUE(!output_offsets.empty());
      model_io_offsets_.emplace(output_offsets[0]);
      const auto &addr_var_name = "dev_input" + std::to_string(input_data_index) + "_ptr";
      input_memcpy_tensor_data_content << "  auto " << addr_var_name << " = GET_ADDR(total_dev_mem_ptr_, "
                                       << output_offsets[0] << ");" << std::endl;
      const auto &tensor_var_name = "input_data_" + std::to_string(input_data_index) + "_tensor";
      input_memcpy_tensor_data_content << "  auto " << tensor_var_name
                                       << " = reinterpret_cast<gert::Tensor *>(input_data[" << tmp_index << "]);"
                                       << std::endl;
      input_memcpy_tensor_data_content << "  OM2_CHK_STATUS(aclrtMemcpy(" << addr_var_name << ", "
                                       << tensor_var_name
                                       << "->GetSize(), " << tensor_var_name << "->GetAddr(), "
                                       << tensor_var_name << "->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE));"
                                       << std::endl;
      input_memcpy_tensor_data_async_content << "  auto " << addr_var_name << " = GET_ADDR(total_dev_mem_ptr_, "
                                             << output_offsets[0] << ");" << std::endl;
      input_memcpy_tensor_data_async_content << "  auto " << tensor_var_name
                                             << " = reinterpret_cast<gert::Tensor *>(input_data[" << tmp_index << "]);"
                                             << std::endl;
      input_memcpy_tensor_data_async_content << "  OM2_CHK_STATUS(aclrtMemcpyAsync(" << addr_var_name << ", "
                                             << tensor_var_name
                                             << "->GetSize(), " << tensor_var_name << "->GetAddr(), "
                                             << tensor_var_name << "->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE, exe_stream));"
                                             << std::endl;
      input_data_index++;
    } else if (OpTypeUtils::IsGraphOutputNode(op_desc->GetType())) {  // 当前考虑静态，模型中只有一个NETOUTPUT
      auto input_offsets = op_desc->GetInputOffset();
      for (size_t i = 0U; i < op_desc->GetAllInputsSize(); ++i) {
        const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
        GE_IF_BOOL_EXEC(tensor_desc == nullptr,
                        GELOGD("[OM2] Op: %s, Index: %zu, has no input", op_desc->GetName().c_str(), i);
                        continue);
        model_io_offsets_.emplace(input_offsets[i]);
        const auto &addr_var_name = "dev_output" + std::to_string(output_data_index) + "_ptr";
        output_memcpy_tensor_data_content << "  auto " << addr_var_name << " = GET_ADDR(total_dev_mem_ptr_, "
                                          << input_offsets[i] << ");" << std::endl;
        const auto &tensor_var_name = "output_data_" + std::to_string(output_data_index) + "_tensor";
        output_memcpy_tensor_data_content << "  auto " << tensor_var_name
                                          << " = reinterpret_cast<gert::Tensor *>(output_data[" << output_data_index
                                          << "]);"
                                          << std::endl;
        output_memcpy_tensor_data_content << "  OM2_CHK_STATUS(aclrtMemcpy("
                                          << tensor_var_name << "->GetAddr(), " << tensor_var_name
                                          << "->GetSize(), " << addr_var_name << ", " << tensor_var_name
                                          << "->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE));" << std::endl;
        output_memcpy_tensor_data_async_content << "  auto " << addr_var_name
                                                << " = GET_ADDR(total_dev_mem_ptr_, " << input_offsets[i] << ");"
                                                << std::endl;
        output_memcpy_tensor_data_async_content << "  auto " << tensor_var_name
                                                << " = reinterpret_cast<gert::Tensor *>(output_data["
                                                << output_data_index
                                                << "]);"
                                                << std::endl;
        output_memcpy_tensor_data_async_content << "  OM2_CHK_STATUS(aclrtMemcpyAsync("
                                                << tensor_var_name << "->GetAddr(), "
                                                << tensor_var_name
                                                << "->GetSize(), " << addr_var_name << ", "
                                                << tensor_var_name
                                                << "->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE, exe_stream));"
                                                << std::endl;
        output_data_index++;
      }
    }
  }

  std::string content = R"(aclError Om2Model::RunAsync(
  aclrtStream &exe_stream,
  size_t input_count,
  void **input_data,
  size_t output_count,
  void **output_data) {
  if (input_count != om2::INPUT_NUM || output_count != om2::OUTPUT_NUM) {
    return ACL_ERROR_FAILURE;
  }
)" + input_memcpy_tensor_data_async_content.str() +
                        R"(
  OM2_CHK_STATUS(args_table_.CopyArgsToDevice());
  OM2_CHK_STATUS(aclmdlRIExecuteAsync(model_handle_, exe_stream));
)" + output_memcpy_tensor_data_async_content.str() +
                        R"(
  return ACL_SUCCESS;
}

aclError Om2Model::Run(
  size_t input_count,
  void **input_data,
  size_t output_count,
  void **output_data) {
  if (input_count != om2::INPUT_NUM || output_count != om2::OUTPUT_NUM) {
    return ACL_ERROR_FAILURE;
  }
)" + input_memcpy_tensor_data_content.str() +
                        R"(
  OM2_CHK_STATUS(args_table_.CopyArgsToDevice());
  OM2_CHK_STATUS(aclmdlRIExecute(model_handle_, -1));
)" + output_memcpy_tensor_data_content.str() +
                        R"(
  return ACL_SUCCESS;
}
)";
  load_impl_code.push_back(RAW_CODE_STMT(ast_ctx_, content));
  GELOGD("[OM2] Om2Model::Run() and Om2Model::RunAsync() implementation is generated.");
  return SUCCESS;
}

void ProgramGenerator::GenExternalApiImpl(Program &program) {
  std::string content =
      R"(aclError Om2ModelCreate(om2::Om2ModelHandle *model_handle, const char **bin_files, const void **bin_data, size_t *bin_size, int bin_num, void *host_weight_mem_ptr, uint64_t *session_id) {
  if (*model_handle != nullptr) {
    return ACL_ERROR_FAILURE;
  }
  auto *obj = new om2::Om2Model(bin_files, bin_data, bin_size, bin_num, host_weight_mem_ptr, session_id);
  if (obj == nullptr) {
    return ACL_ERROR_FAILURE;
  }
  auto ret = obj->InitResources();
  if (ret != ACL_SUCCESS) {
    delete obj;
    return ret;
  }
  ret = obj->RegisterKernels();
  if (ret != ACL_SUCCESS) {
    delete obj;
    return ret;
  }
  ret = obj->Load();
  if (ret != ACL_SUCCESS) {
    delete obj;
    return ret;
  }
  *model_handle = reinterpret_cast<om2::Om2ModelHandle>(obj);
  return ACL_SUCCESS;
}

aclError Om2ModelRunAsync(om2::Om2ModelHandle* model_handle, aclrtStream stream, int input_count, void **input_data, int output_count, void **output_data) {
  return static_cast<om2::Om2Model*>(*model_handle)->RunAsync(stream, input_count, input_data, output_count, output_data);
}

aclError Om2ModelRun(om2::Om2ModelHandle* model_handle, int input_count, void **input_data, int output_count, void **output_data) {
  return static_cast<om2::Om2Model*>(*model_handle)->Run(input_count, input_data, output_count, output_data);
}

aclError Om2ModelDestroy(om2::Om2ModelHandle* model_handle) {
  delete static_cast<om2::Om2Model*>(*model_handle);
  return ACL_SUCCESS;
})";
  WriteLoadAndRunSource(program, RAW_CODE_STMT(ast_ctx_, content));
}

Status ProgramGenerator::GenerateInterfaceHeader(Program &program) {
  std::string include_header = R"(#include <iostream>
#include <cstddef>
#include <ctime>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <functional>
#include <cstdint>
#include <type_traits>

#include "securec.h"
#include "acl/acl.h"
#include "acl/acl_base.h"
#include "exe_graph/runtime/tensor.h"
)";
  WriteInterfaceHeader(program, RAW_CODE_STMT(ast_ctx_, include_header));
  GenInterfaceHeaderMacros(program);
  GenInterfaceHeaderCommonFunc(program);
  WriteInterfaceHeader(program, RAW_CODE_STMT(ast_ctx_, "namespace om2 {"));
  GE_ASSERT_SUCCESS(GenInterfaceHeaderCommonPart(program));
  GenInterfaceHeaderOm2ArgsTableClass(program);
  GenInterfaceHeaderOm2ModelClass(program);
  WriteInterfaceHeader(program, RAW_CODE_STMT(ast_ctx_, "} // namespace om2"));
  GenInterfaceHeaderExternalApi(program);
  return SUCCESS;
}

Status ProgramGenerator::GenerateResourcesSource(Program &program) {
  WriteResourcesSource(program, RAW_CODE_STMT(ast_ctx_, "#include \"" + ge_model_->GetName() + "_interface.h\"\n"));
  WriteResourcesSource(program, RAW_CODE_STMT(ast_ctx_, ("namespace om2 {")));
  GenOm2ModelConstructor(program);
  GenOm2ModelDestructor(program);
  GE_ASSERT_SUCCESS(GenInitResourcesImpl(program));
  GenReleaseResourcesImpl(program);
  WriteResourcesSource(program, RAW_CODE_STMT(ast_ctx_, ("} // namespace om2")));
  GELOGD("[OM2] Interface header file code is generated.");
  return SUCCESS;
}

Status ProgramGenerator::GenerateArgsManagerSource(Program &program) {
  WriteArgManagerSource(program, RAW_CODE_STMT(ast_ctx_, "#include \"" + ge_model_->GetName() + "_interface.h\"\n"));
  WriteArgManagerSource(program, RAW_CODE_STMT(ast_ctx_, ("namespace om2 {")));
  std::vector<AstNode *> args_table_impl_ast_nodes;
  GE_ASSERT_SUCCESS(GenArgsTableImpl(args_table_impl_ast_nodes));
  program[static_cast<size_t>(GeneratedFileIndex::kArgsManagerFile)].insert(
      program[static_cast<size_t>(GeneratedFileIndex::kArgsManagerFile)].end(), args_table_impl_ast_nodes.begin(),
      args_table_impl_ast_nodes.end());
  WriteArgManagerSource(program, RAW_CODE_STMT(ast_ctx_, ("} // namespace om2")));
  GELOGD("[OM2] Args Manager source file code is generated.");
  return SUCCESS;
}

Status ProgramGenerator::GenerateKernelRegSource(Program &program) {
  WriteKernelRegSource(program, RAW_CODE_STMT(ast_ctx_, "#include \"" + ge_model_->GetName() + "_interface.h\"\n"));
  WriteKernelRegSource(program, RAW_CODE_STMT(ast_ctx_, ("namespace om2 {")));
  WriteKernelRegSource(program, RAW_CODE_STMT(ast_ctx_, ("namespace {")));
  GenKernelRegConsts(program);
  GenKernelRegCommonFuncs(program);
  GenKernelRegFuncsImpl(program);
  WriteKernelRegSource(program, RAW_CODE_STMT(ast_ctx_, ("} // namespace")));
  GE_ASSERT_SUCCESS(GenRegisterKernelsImpl(program));
  WriteKernelRegSource(program, RAW_CODE_STMT(ast_ctx_, ("} // namespace om2")));
  GELOGD("[OM2] Kernel Reg source file code is generated.");
  return SUCCESS;
}

Status ProgramGenerator::GenerateLoadAndRunSource(Program &program) {
  WriteLoadAndRunSource(program, RAW_CODE_STMT(ast_ctx_, "#include \"rt_external_kernel.h\""));
  WriteLoadAndRunSource(program, RAW_CODE_STMT(ast_ctx_, "#include \"" + ge_model_->GetName() + "_interface.h\"\n"));
  WriteLoadAndRunSource(program, RAW_CODE_STMT(ast_ctx_, "namespace om2 {"));
  WriteLoadAndRunSource(program, RAW_CODE_STMT(ast_ctx_, ("namespace {")));
  WriteLoadAndRunSource(program, RAW_CODE_STMT(ast_ctx_, "constexpr const size_t max_launch_cfg_num = 8UL;"));
  WriteLoadAndRunSource(program, RAW_CODE_STMT(ast_ctx_, "constexpr int64_t kDImEndFlag = std::numeric_limits<int64_t>::min();"));
  std::vector<AstNode *> run_impl_code;
  std::vector<AstNode *> load_impl_code;
  std::vector<AstNode *> dist_impl_code;
  GE_ASSERT_SUCCESS(GenRunImpl(run_impl_code));
  GE_ASSERT_SUCCESS(GenLoadImpl(load_impl_code, dist_impl_code));
  program[static_cast<size_t>(GeneratedFileIndex::kLoadingAndRunningFile)].insert(
      program[static_cast<size_t>(GeneratedFileIndex::kLoadingAndRunningFile)].end(), dist_impl_code.begin(),
      dist_impl_code.end());
  if (aicpu_task_num_ > 0UL) {
    GE_ASSERT_SUCCESS(GenAicpuArgsCommon(program));
  }
  WriteLoadAndRunSource(program, RAW_CODE_STMT(ast_ctx_, ("} // namespace")));
  program[static_cast<size_t>(GeneratedFileIndex::kLoadingAndRunningFile)].insert(
      program[static_cast<size_t>(GeneratedFileIndex::kLoadingAndRunningFile)].end(), load_impl_code.begin(),
      load_impl_code.end());
  program[static_cast<size_t>(GeneratedFileIndex::kLoadingAndRunningFile)].insert(
      program[static_cast<size_t>(GeneratedFileIndex::kLoadingAndRunningFile)].end(), run_impl_code.begin(),
      run_impl_code.end());
  WriteLoadAndRunSource(program, RAW_CODE_STMT(ast_ctx_, ("} // namespace om2")));
  GenExternalApiImpl(program);
  GELOGD("[OM2] Load and run source file code is generated.");
  return SUCCESS;
}

Status ProgramGenerator::GenerateMakeFile(Program &program) {
  const std::string model_name = ge_model_->GetName();
  const std::string lib_name = model_name + "_om2";
  std::string cmakelists_content = R"(CANN_ROOT ?= $(ASCEND_HOME_PATH)
USE_STUB_LIB ?= 0

CXX := g++
TARGET := lib)" + lib_name + R"(.so
SRC_FILES := )" + model_name + R"(_resources.cpp )" + model_name + R"(_kernel_reg.cpp )" + model_name +
                                     R"(_load_and_run.cpp )" + model_name + R"(_args_manager.cpp

CXXFLAGS := -std=c++17 -O2 -fPIC \
  -I$(CANN_ROOT)/include \
  -I$(CANN_ROOT)/pkg_inc \
  -I$(CANN_ROOT)/pkg_inc/runtime \
  -I$(CANN_ROOT)/pkg_inc/runtime/runtime \
  -I$(CANN_ROOT)/pkg_inc/profiling \
  -I$(CURDIR)/include

ifeq ($(USE_STUB_LIB),1)
LIB_PATH := $(CANN_ROOT)/devlib
else
LIB_PATH := $(CANN_ROOT)/lib64
endif

LDFLAGS := -shared -L$(LIB_PATH) -Wl,--no-as-needed -lacl_rt -Wl,--as-needed

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC_FILES)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
)";
  WriteCmakeLists(program, RAW_CODE_STMT(ast_ctx_, cmakelists_content));
  GELOGD("[OM2] Makefile code is generated.");
  return SUCCESS;
}
}  // namespace ge

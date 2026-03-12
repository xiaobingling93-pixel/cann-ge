/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPENSOURCEGE_AIR_TESTS_ST_STUBS_UTILS_TASKDEF_BUILDER_H_
#define OPENSOURCEGE_AIR_TESTS_ST_STUBS_UTILS_TASKDEF_BUILDER_H_

#include "proto/task.pb.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "aicpu_task_struct.h"
#include "hybrid/node_executor/aicpu/aicpu_ext_info_handler.h"
#include "framework/common/taskdown_common.h"
#include "register/op_tiling_info.h"
#include "graph/debug/ge_attr_define.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace ge {
class ExtInfoBuilder {
 public:
  explicit ExtInfoBuilder(std::vector<uint8_t> &buffer) : buffer_(buffer) {
  }
  template<typename T>
  ExtInfoBuilder &AddExtInfo(int32_t type, const T &value) {
    AddExtInfoHeader(type, sizeof(value));
    AddExtInfoValue(value);
    return *this;
  }

  ExtInfoBuilder &AddExtInfoHeader(int32_t type, size_t value_size) {
    auto current_size = buffer_.size();
    auto header_size = sizeof(hybrid::AicpuExtInfo);
    buffer_.resize(current_size + header_size);
    auto *ext_info = reinterpret_cast<hybrid::AicpuExtInfo *>(buffer_.data() + current_size);
    ext_info->infoType = type;
    ext_info->infoLen = value_size;
    return *this;
  }

  template<typename T>
  ExtInfoBuilder &AddExtInfoValue(const T &value) {
    auto current_size = buffer_.size();
    auto ext_info_size = sizeof(value);
    buffer_.resize(current_size + ext_info_size);
    auto *dst_addr = buffer_.data() + current_size;
    memcpy_s(dst_addr, sizeof(value), &value, sizeof(value));
    return *this;
  }

  ExtInfoBuilder &AddUnknownShapeType(int32_t type) {
    return AddExtInfo(aicpu::FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE, type);
  }

  ExtInfoBuilder &AddShapeAndType(size_t num, int32_t type) {
    AddExtInfoHeader(type, sizeof(hybrid::AicpuShapeAndType) * num);
    hybrid::AicpuShapeAndType shape_and_type;
    shape_and_type.type = 0;
    for (size_t i = 0; i < num; ++i) {
      AddExtInfoValue(shape_and_type);
    }
    return *this;
  }

 private:
  std::vector<uint8_t> &buffer_;
};

class AicpuTaskDefBuilder {
 public:
  struct AicpuTaskStruct {
    aicpu::AicpuParamHead head;
    uint64_t io_addrp[6];
  }__attribute__((packed));

  AicpuTaskDefBuilder(const Node &node) : node_(node) {
  }

  domi::TaskDef BuildHostCpuTask(int unknown_shape_type) {
    auto task_def = BuildAicpuTask(unknown_shape_type);
    task_def.mutable_kernel()->mutable_context()->set_kernel_type(8);  // HOST_CPU
    node_.GetOpDesc()->SetOpKernelLibName("DNN_VM_HOST_CPU_OP_STORE");
    return task_def;
  }

  domi::TaskDef BuildTfTask(int unknown_shape_type, uint32_t topic_type_flag = 0) {
    auto op_desc = node_.GetOpDesc();
    AttrUtils::SetInt(op_desc, ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type);
    op_desc->SetOpKernelLibName("aicpu_tf_kernel");
    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL_EX));
    domi::KernelExDef *kernel_ex_def = task_def.mutable_kernel_ex();
    kernel_ex_def->set_kernel_ext_info_size(12);
    kernel_ex_def->set_op_index(op_desc->GetId());

    std::vector<uint8_t> buffer;
    BuildAiCpuExtInfo(unknown_shape_type, *op_desc, buffer, topic_type_flag);
    kernel_ex_def->set_kernel_ext_info(buffer.data(), buffer.size());
    kernel_ex_def->set_kernel_ext_info_size(buffer.size());
    return task_def;
  }

  domi::TaskDef BuildAicpuTask(int unknown_shape_type, uint32_t topic_type_flag = 0) {
    auto op_desc = node_.GetOpDesc();
    domi::TaskDef task_def;
    AttrUtils::SetInt(op_desc, ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type);
    op_desc->SetOpKernelLibName("aicpu_ascend_kernel");
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    domi::KernelDef *kernel_def = task_def.mutable_kernel();

    AicpuTaskStruct args{};
    args.head.length = sizeof(args);
    args.head.ioAddrNum = op_desc->GetInputsSize() + op_desc->GetOutputsSize();
    kernel_def->set_args(reinterpret_cast<const char *>(&args), args.head.length);
    kernel_def->set_args_size(args.head.length);

    std::vector<uint8_t> buffer;
    BuildAiCpuExtInfo(unknown_shape_type, *op_desc, buffer, topic_type_flag);
    kernel_def->set_kernel_ext_info(buffer.data(), buffer.size());
    kernel_def->set_kernel_ext_info_size(buffer.size());
    domi::KernelContext *context = kernel_def->mutable_context();

    context->set_kernel_type(6);    // ccKernelType::AI_CPU
    context->set_op_index(op_desc->GetId());
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
    return task_def;
  }

  static void BuildAiCpuExtInfo(int32_t type, OpDesc &op_desc, std::vector<uint8_t> &buffer, uint32_t topic_type = 0) {
    uint64_t ext_bitmap = 0U;   // FWK_ADPT_EXT_BITMAP
    uint32_t update_addr = 0U;  // FWK_ADPT_EXT_UPDATE_ADDR
    ExtInfoBuilder(buffer)
        .AddUnknownShapeType(type)
        .AddShapeAndType(op_desc.GetInputsSize(), aicpu::FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE)
        .AddShapeAndType(op_desc.GetOutputsSize(), aicpu::FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE)
        .AddExtInfo(aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT, hybrid::AsyncWaitInfo{})
        .AddExtInfo(aicpu::FWKAdapter::FWK_ADPT_EXT_WORKSPACE_INFO, hybrid::WorkSpaceInfo{})
        .AddExtInfo(aicpu::FWKAdapter::FWK_ADPT_EXT_SESSION_INFO, hybrid::AicpuSessionInfo{})
        .AddExtInfo(aicpu::FWKAdapter::FWK_ADPT_EXT_BITMAP, ext_bitmap)
        .AddExtInfo(aicpu::FWKAdapter::FWK_ADPT_EXT_UPDATE_ADDR, update_addr)
        .AddExtInfo(aicpu::FWKAdapter::FWK_ADPT_EXT_TOPIC_TYPE, topic_type);
  }

 private:
  const Node &node_;
};

class MixL2TaskDefBuilder {
 public:
  explicit MixL2TaskDefBuilder(const Node &node) : node_(node) {}

  domi::TaskDef BuildTask(bool is_dynamic = false, bool is_single_kernel = false) {
    auto op_desc = node_.GetOpDesc();
    (void)AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_AIC");
    op_desc->SetOpKernelLibName("AIcoreEngine");

    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));

    domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
    ffts_plus_task_def->set_op_index(op_desc->GetId());
    ffts_plus_task_def->set_addr_size(20);

    domi::FftsPlusCtxDef *ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
    ctx_def->set_context_type(RT_CTX_TYPE_MIX_AIC);

    domi::FftsPlusMixAicAivCtxDef *mixctx_def = ctx_def->mutable_mix_aic_aiv_ctx();
    mixctx_def->set_thread_id(0);
    mixctx_def->set_thread_dim(1);

    if (is_single_kernel) {
      std::vector<std::string> name_prefix = {"_mix_enhanced"};
      AttrUtils::SetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, name_prefix);
      mixctx_def->add_kernel_name("_mix_enhanced");
      std::vector<char> mix_aic_bin(64, '\0');
      TBEKernelPtr mix_aic_kernel = MakeShared<ge::OpKernelBin>("_mix_aictbeKernel", std::move(mix_aic_bin));
      op_desc->SetExtAttr(std::string("_mix_enhanced") + OP_EXTATTR_NAME_TBE_KERNEL, mix_aic_kernel);
      AttrUtils::SetStr(op_desc, std::string("_mix_enhanced") + ATTR_NAME_TBE_KERNEL_NAME, mix_aic_kernel->GetName());
    } else {
      std::vector<std::string> name_prefix = {"_mix_aic", "_mix_aiv"};
      AttrUtils::SetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, name_prefix);
      mixctx_def->add_kernel_name("_mix_aic");
      mixctx_def->add_kernel_name("_mix_aiv");
      std::vector<char> mix_aic_bin(64, '\0');
      TBEKernelPtr mix_aic_kernel = MakeShared<ge::OpKernelBin>("_mix_aictbeKernel", std::move(mix_aic_bin));
      op_desc->SetExtAttr(std::string("_mix_aic") + OP_EXTATTR_NAME_TBE_KERNEL, mix_aic_kernel);
      AttrUtils::SetStr(op_desc, std::string("_mix_aic") + ATTR_NAME_TBE_KERNEL_NAME, mix_aic_kernel->GetName());

      std::vector<char> mix_aiv_bin(64, '\0');
      TBEKernelPtr mix_aiv_kernel = MakeShared<ge::OpKernelBin>("_mix_aivtbeKernel", std::move(mix_aiv_bin));
      op_desc->SetExtAttr(std::string("_mix_aiv") + OP_EXTATTR_NAME_TBE_KERNEL, mix_aiv_kernel);
      AttrUtils::SetStr(op_desc, std::string("_mix_aiv") + ATTR_NAME_TBE_KERNEL_NAME, mix_aiv_kernel->GetName());

      string attr_kernel_name_aic = "_mix_aic" + op_desc->GetName() + "_kernelname";
      (void)AttrUtils::SetStr(op_desc, attr_kernel_name_aic, "_mix_aic");

      string attr_kernel_name_aiv = "_mix_aiv" + op_desc->GetName() + "_kernelname";
      (void)AttrUtils::SetStr(op_desc, attr_kernel_name_aiv, "_mix_aiv");

      mixctx_def->add_task_addr(std::numeric_limits<uint64_t>::max());           // value0
      mixctx_def->add_task_addr(std::numeric_limits<uint64_t>::max());           // value1
      mixctx_def->add_task_addr(std::numeric_limits<uint64_t>::max());           // tilling
    }
    std::vector<uint8_t> args(100, 0);

    if (is_dynamic) {
      AttrUtils::SetBool(op_desc, "support_dynamicshape", true);
      AttrUtils::SetInt(op_desc, "op_para_size", 512);
      AttrUtils::SetStr(op_desc, "compile_info_key", "ddd");
      AttrUtils::SetStr(op_desc, "compile_info_json", "{}");
      AttrUtils::SetStr(op_desc, "_mix_aic_kernel_list_first_name", "aic");
      AttrUtils::SetStr(op_desc, "_mix_aiv_kernel_list_first_name", "aiv");
      std::vector<int64_t> workspace{1024};
       op_desc->SetWorkspaceBytes(workspace);
    }
    return task_def;
  }

 private:
  const Node &node_;
};


class AiCoreTaskDefBuilder {
 public:
  explicit AiCoreTaskDefBuilder(const Node &node) : node_(node) {}

  domi::TaskDef BuildTask(bool is_dynamic = false) {
    auto op_desc = node_.GetOpDesc();
    op_desc->SetOpKernelLibName("AIcoreEngine");

    AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, op_desc->GetName() + "_fake_kernel_bin_id");
    std::vector<char> atomic_kernel(128);
    std::vector<char> relu_kernel(256);
    op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, std::make_shared<OpKernelBin>("relu_xxx", std::move(relu_kernel)));

    std::vector<uint8_t> args(64, 0);
    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    auto kernel_info = task_def.mutable_kernel();
    kernel_info->set_args(args.data(), args.size());
    kernel_info->set_args_size(64);
    kernel_info->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    kernel_info->set_kernel_name(node_.GetName());
    kernel_info->set_block_dim(1);
    uint16_t args_offset[2] = {0};
    kernel_info->mutable_context()->set_args_offset(args_offset, 2 * sizeof(uint16_t));
    kernel_info->mutable_context()->set_op_index(node_.GetOpDesc()->GetId());

    if (is_dynamic) {
      AttrUtils::SetBool(op_desc, "support_dynamicshape", true);
      AttrUtils::SetInt(op_desc, "op_para_size", 512);
      AttrUtils::SetStr(op_desc, "compile_info_key", "ddd");
      AttrUtils::SetStr(op_desc, "compile_info_json", "{}");
    }
    return task_def;
  }

  domi::TaskDef BuildAtomicAddrCleanTask() {
    auto op_desc = node_.GetOpDesc();
    AttrUtils::SetInt(op_desc, "atomic_op_para_size", 256);
    AttrUtils::SetStr(op_desc, "_atomic_compile_info_key", "ddd");
    if (!op_desc->HasAttr("_atomic_compile_info_json")) {
      AttrUtils::SetStr(op_desc, "_atomic_compile_info_json", "{}");
    }
    std::vector<char> atomic_kernel(128);
    op_desc->SetExtAttr(EXT_ATTR_ATOMIC_TBE_KERNEL,
                        std::make_shared<OpKernelBin>("atomic_xxx", std::move(atomic_kernel)));

    vector<int64_t> atomic_output_index = {0};
    AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
    auto task_def = BuildTask(false);
    return task_def;
  }

  domi::TaskDef BuildTaskWithHandle() {
    auto op_desc = node_.GetOpDesc();
    op_desc->SetOpKernelLibName("AIcoreEngine");

    AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    std::vector<char> atomic_kernel(128);
    std::vector<char> relu_kernel(256);
    std::string bin_name = op_desc->GetName() + "_tvm";
    op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, std::make_shared<OpKernelBin>(bin_name, std::move(relu_kernel)));
    AttrUtils::SetInt(op_desc, ATTR_NAME_IMPLY_TYPE, static_cast<uint32_t>(domi::ImplyType::TVM));
    AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, bin_name);

    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
    auto *kernel_with_handle = task_def.mutable_kernel_with_handle();
    kernel_with_handle->set_original_kernel_key("");
    kernel_with_handle->set_node_info("");
    kernel_with_handle->set_block_dim(32);
    kernel_with_handle->set_args_size(64);
    string args(64, '1');
    kernel_with_handle->set_args(args.data(), 64);
    domi::KernelContext *context = kernel_with_handle->mutable_context();
    context->set_op_index(op_desc->GetId());
    context->set_kernel_type(2);    // ccKernelType::TE
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));

    AttrUtils::SetBool(op_desc, "support_dynamicshape", true);
    AttrUtils::SetInt(op_desc, "op_para_size", 512);

    AttrUtils::SetStr(op_desc, "compile_info_key", "ddd");
    AttrUtils::SetStr(op_desc, "compile_info_json", "{}");
    return task_def;
  }

 private:
  const Node &node_;
};

}  // namespace ge
#endif //OPENSOURCEGE_AIR_TESTS_ST_STUBS_UTILS_TASKDEF_BUILDER_H_

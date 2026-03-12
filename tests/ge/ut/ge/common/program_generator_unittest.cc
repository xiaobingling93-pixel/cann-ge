/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "common/om2/codegen/program_generator.h"
#include "common/om2/codegen/emitter/cpp_emitter.h"
#include "framework/common/taskdown_common.h"
#include "ge_runtime_stub/include/common/share_graph.h"
#include "ge_runtime_stub/include/faker/ge_model_builder.h"
#include "ge_runtime_stub/include/faker/aicore_taskdef_faker.h"
#include "ge_runtime_stub/include/faker/aicpu_taskdef_faker.h"
#include "graph/debug/ge_attr_define.h"

#include <cinttypes>
#include <securec.h>
#include "faker/task_def_faker.h"
#include "aicpu_engine_struct.h"
#include "aicpu_task_struct.h"
#include "engine/aicpu/kernel/aicpu_ext_info_handle.h"

namespace ge {
namespace {
using AicpuShapeAndType = aicpu::FWKAdapter::ShapeAndType;
using AicpuExtInfo = aicpu::FWKAdapter::ExtInfo;
using AsyncWaitInfo = aicpu::FWKAdapter::AsyncWait;
using WorkSpaceInfo = aicpu::FWKAdapter::WorkSpaceInfo;
using AicpuSessionInfo = SessionInfo;
struct AicpuTaskArgs {
  aicpu::AicpuParamHead head;
  uint64_t io_addrp[6];
} __attribute__((packed));
void AppendShape(aicpu::FWKAdapter::FWKTaskExtInfoType type, size_t shape_num, std::string &out) {
  size_t len = sizeof(AicpuShapeAndType) * shape_num + sizeof(AicpuExtInfo);
  vector<char> vec(len, 0);
  AicpuExtInfo *aicpu_ext_info = reinterpret_cast<AicpuExtInfo *>(vec.data());
  aicpu_ext_info->infoType = type;
  aicpu_ext_info->infoLen = sizeof(AicpuShapeAndType) * shape_num;
  AicpuShapeAndType input_shape_and_types[shape_num] = {};
  for (auto m = 0U; m < shape_num; m++) {
    input_shape_and_types[m].dims[0] = 5;
  }
  memcpy_s(aicpu_ext_info->infoMsg, sizeof(AicpuShapeAndType) * shape_num,
           reinterpret_cast<void *>(input_shape_and_types), sizeof(AicpuShapeAndType) * shape_num);

  std::string s(vec.data(), len);
  out.append(s);
}

void AppendSessionInfo(std::string &out) {
  size_t len = sizeof(AicpuSessionInfo) + sizeof(AicpuExtInfo);
  vector<char> vec(len, 0);
  AicpuExtInfo *aicpu_ext_info = reinterpret_cast<AicpuExtInfo *>(vec.data());
  aicpu_ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_SESSION_INFO;
  aicpu_ext_info->infoLen = sizeof(AicpuSessionInfo);
  AicpuSessionInfo session = {};
  *(ge::PtrToPtr<char, AicpuSessionInfo>(aicpu_ext_info->infoMsg)) = session;
  std::string s(vec.data(), len);
  out.append(s);
}

void AppendBitMap(std::string &out) {
  size_t len = sizeof(uint64_t) + sizeof(AicpuExtInfo);
  vector<char> vec(len, 0);
  AicpuExtInfo *aicpu_ext_info = reinterpret_cast<AicpuExtInfo *>(vec.data());
  aicpu_ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_BITMAP;
  aicpu_ext_info->infoLen = sizeof(uint64_t);
  *(ge::PtrToPtr<char, uint64_t>(aicpu_ext_info->infoMsg)) = 1;
  std::string s(vec.data(), len);
  out.append(s);
}

void AppendUpdateAddr(std::string &out) {
  size_t len = sizeof(uint32_t) + sizeof(AicpuExtInfo);
  vector<char> vec(len, 0);
  AicpuExtInfo *aicpu_ext_info = reinterpret_cast<AicpuExtInfo *>(vec.data());
  aicpu_ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_UPDATE_ADDR;
  aicpu_ext_info->infoLen = sizeof(uint32_t);
  *(ge::PtrToPtr<char, uint32_t>(aicpu_ext_info->infoMsg)) = 1;
  std::string s(vec.data(), len);
  out.append(s);
}

void AppendTopicType(std::string &out) {
  size_t len = sizeof(int32_t) + sizeof(AicpuExtInfo);
  vector<char> vec(len, 0);
  AicpuExtInfo *aicpu_ext_info = reinterpret_cast<AicpuExtInfo *>(vec.data());
  aicpu_ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_TOPIC_TYPE;
  aicpu_ext_info->infoLen = sizeof(int32_t);
  *(ge::PtrToPtr<char, int32_t>(aicpu_ext_info->infoMsg)) = 1;
  std::string s(vec.data(), len);
  out.append(s);
}

std::string GetFakeExtInfo() {
  std::string ext_info;
  AppendShape(aicpu::FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE, 2, ext_info);
  AppendShape(aicpu::FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE, 1, ext_info);
  AppendSessionInfo(ext_info);
  AppendBitMap(ext_info);
  AppendUpdateAddr(ext_info);
  AppendTopicType(ext_info);
  return ext_info;
}

GeRootModelPtr CreateGeRootModelWithAicoreOp() {
  auto graph = gert::ShareGraph::AicoreStaticGraph();
  graph->TopologicalSorting();
  gert::GeModelBuilder builder(graph);
  auto ge_root_model =
      builder
          .AddTaskDef("Add",
                      gert::AiCoreTaskDefFaker("add_stub").ArgsFormat("{i_instance0*}{i_instance1*}{o_instance0*}{ws0*}"))
          .FakeTbeBin({"Add"})
          .BuildGeRootModel();
  auto &compute_graph = ge_root_model->GetRootGraph();

  compute_graph->SetGraphUnknownFlag(false);
  for (const auto &node : compute_graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      return nullptr;
    }
    if ((op_desc->GetType() == DATA)) {
      op_desc->SetOutputOffset({1024});
    } else if (op_desc->GetType() == NETOUTPUT) {
      op_desc->SetInputOffset({1024});
    } else {
      op_desc->AppendIrInput("x1", kIrInputRequired);
      op_desc->AppendIrInput("x2", kIrInputRequired);
      op_desc->AppendIrOutput("y", kIrOutputRequired);
      op_desc->SetInputOffset(std::vector<int64_t>(op_desc->GetInputsSize(), 1024));
      op_desc->SetOutputOffset(std::vector<int64_t>(op_desc->GetOutputsSize(), 1024));
      op_desc->SetWorkspaceBytes(std::vector<int64_t>(1, 64));
      op_desc->SetWorkspace(std::vector<int64_t>(1, 0));
    }
  }

  const auto ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  std::vector<uint64_t> weights_value(64, 1024);
  const size_t weight_size = weights_value.size() * sizeof(uint64_t);
  ge_model->SetWeight(Buffer::CopyFrom(reinterpret_cast<uint8_t *>(weights_value.data()), weight_size));

  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2048);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  std::cout << ge_model->GetModelTaskDefPtr()->DebugString() << std::endl;

  return ge_root_model;
}

GeRootModelPtr CreateGeRootModelWithAicoreOp2() {
  auto graph = gert::ShareGraph::AicoreStaticGraph();
  graph->TopologicalSorting();
  for (const auto &node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if ((op_desc != nullptr) && (op_desc->GetType() == "Add")) {
      op_desc->AppendIrInput("x1", kIrInputRequired);
      op_desc->AppendIrInput("x2", kIrInputRequired);
      op_desc->AppendIrOutput("y", kIrOutputRequired);
    }
  }
  gert::GeModelBuilder builder(graph);
  auto ge_root_model =
      builder
          .AddTaskDef("Add",
                      gert::AiCoreTaskDefFaker("add_stub").ArgsFormat("{i0*}{i1*}{o0*}{ws0*}"))
          .FakeTbeBin({"Add"})
          .BuildGeRootModel();
  auto &compute_graph = ge_root_model->GetRootGraph();

  compute_graph->SetGraphUnknownFlag(false);
  for (const auto &node : compute_graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      return nullptr;
    }
    if ((op_desc->GetType() == DATA)) {
      op_desc->SetOutputOffset({1024});
    } else if (op_desc->GetType() == NETOUTPUT) {
      op_desc->SetInputOffset({1024});
    } else {
      op_desc->SetInputOffset(std::vector<int64_t>(op_desc->GetInputsSize(), 1024));
      op_desc->SetOutputOffset(std::vector<int64_t>(op_desc->GetOutputsSize(), 1024));
      op_desc->SetWorkspaceBytes(std::vector<int64_t>(1, 64));
      op_desc->SetWorkspace(std::vector<int64_t>(1, 0));
    }
  }

  const auto ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  std::vector<uint64_t> weights_value(64, 1024);
  const size_t weight_size = weights_value.size() * sizeof(uint64_t);
  ge_model->SetWeight(Buffer::CopyFrom(reinterpret_cast<uint8_t *>(weights_value.data()), weight_size));

  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2048);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  std::cout << ge_model->GetModelTaskDefPtr()->DebugString() << std::endl;

  return ge_root_model;
}

GeRootModelPtr CreateGeRootModelWithAicpuOp() {
  auto graph = gert::ShareGraph::Aicpu4thGraph();
  graph->TopologicalSorting();
  gert::GeModelBuilder builder(graph);
  gert::AiCpuCCTaskDefFaker aicpu_task_def_faker;
  auto ge_root_model = builder.AddTaskDef("add1", aicpu_task_def_faker.SetNeedMemcpy(false))
                              .AddTaskDef("add2", aicpu_task_def_faker.SetNeedMemcpy(false))
                              .BuildGeRootModel();
  auto &compute_graph = ge_root_model->GetRootGraph();

  compute_graph->SetGraphUnknownFlag(false);
  for (const auto &node : compute_graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      return nullptr;
    }
    if ((op_desc->GetType() == DATA)) {
      op_desc->SetOutputOffset({1024});
    } else if (op_desc->GetType() == NETOUTPUT) {
      op_desc->SetInputOffset({1024});
    } else {
      op_desc->SetInputOffset(std::vector<int64_t>(op_desc->GetInputsSize(), 1024));
      op_desc->SetOutputOffset(std::vector<int64_t>(op_desc->GetOutputsSize(), 1024));
    }
  }

  const auto ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  auto *model_task_def = ge_model->GetModelTaskDefPtr().get();
  if (model_task_def != nullptr) {
    for (int32_t i = 0; i < model_task_def->task_size(); ++i) {
      auto *task_def = model_task_def->mutable_task(i);
      if ((task_def != nullptr) && task_def->has_kernel()) {
        task_def->mutable_kernel()->mutable_context()->set_kernel_type(6U);
        auto ext_info = GetFakeExtInfo();
        auto kernel_def = task_def->mutable_kernel();
        AicpuTaskArgs args = {};
        args.head.length = sizeof(args);
        args.head.ioAddrNum = 3;
        kernel_def->set_args(reinterpret_cast<const char *>(&args), args.head.length);
        kernel_def->set_args_size(args.head.length);
        kernel_def->set_kernel_ext_info(ext_info.c_str(), ext_info.size());
        kernel_def->set_kernel_ext_info_size(ext_info.size());
      }
    }
  }
  std::vector<uint64_t> weights_value(64, 1024);
  const size_t weight_size = weights_value.size() * sizeof(uint64_t);
  ge_model->SetWeight(Buffer::CopyFrom(reinterpret_cast<uint8_t *>(weights_value.data()), weight_size));

  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2048);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  std::cout << ge_model->GetModelTaskDefPtr()->DebugString() << std::endl;

  return ge_root_model;
}

GeRootModelPtr CreateGeRootModelWithAicoreOpOfDynamicIo() {
  auto graph = std::make_shared<ComputeGraph>("g1");
  GeTensorDesc tensor_desc(GeShape({1, 1, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  auto data_x_desc = std::make_shared<OpDesc>("data_x", DATA);
  (void)data_x_desc->AddOutputDesc(tensor_desc);
  auto data_x = graph->AddNode(data_x_desc);
  auto data_dx_desc = std::make_shared<OpDesc>("data_dx", DATA);
  (void)data_dx_desc->AddOutputDesc(tensor_desc);
  auto data_dx = graph->AddNode(data_dx_desc);
  auto op_desc = std::make_shared<OpDesc>("add1", "Add");
  (void)op_desc->AddInputDesc("x", tensor_desc);
  (void)op_desc->AddDynamicInputDesc("dx", 1);
  (void)op_desc->AddDynamicOutputDesc("dy", 1);
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrInput("dx", kIrInputDynamic);
  op_desc->AppendIrOutput("dy", kIrOutputDynamic);
  if (op_desc->GetInputsSize() > 1U) {
    (void)op_desc->UpdateInputDesc(1U, tensor_desc);
  }
  if (op_desc->GetOutputsSize() > 0U) {
    (void)op_desc->UpdateOutputDesc(0U, tensor_desc);
  }
  auto dynamic_node = graph->AddNode(op_desc);
  auto netoutput_desc = std::make_shared<OpDesc>("netoutput", NETOUTPUT);
  (void)netoutput_desc->AddInputDesc(tensor_desc);
  auto netoutput = graph->AddNode(netoutput_desc);
  if ((data_x == nullptr) || (data_dx == nullptr) || (dynamic_node == nullptr) || (netoutput == nullptr)) {
    return nullptr;
  }
  GraphUtils::AddEdge(data_x->GetOutDataAnchor(0), dynamic_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(data_dx->GetOutDataAnchor(0), dynamic_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(dynamic_node->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));
  graph->TopologicalSorting();
  gert::GeModelBuilder builder(graph);
  auto ge_root_model =
      builder
          .AddTaskDef("Add",
                      gert::AiCoreTaskDefFaker("add_stub").ArgsFormat("{i_desc0}{i_desc1}{o_desc0}"))
          .FakeTbeBin({"Add"})
          .BuildGeRootModel();
  auto &compute_graph = ge_root_model->GetRootGraph();

  compute_graph->SetGraphUnknownFlag(false);
  for (const auto &node : compute_graph->GetDirectNode()) {
    op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      return nullptr;
    }
    if ((op_desc->GetType() == DATA)) {
      op_desc->SetOutputOffset({1024});
    } else if (op_desc->GetType() == NETOUTPUT) {
      op_desc->SetInputOffset({1024});
    } else {
      op_desc->SetInputOffset(std::vector<int64_t>(op_desc->GetInputsSize(), 1024));
      op_desc->SetOutputOffset(std::vector<int64_t>(op_desc->GetOutputsSize(), 1024));
    }
  }

  const auto ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  std::vector<uint64_t> weights_value(64, 1024);
  const size_t weight_size = weights_value.size() * sizeof(uint64_t);
  ge_model->SetWeight(Buffer::CopyFrom(reinterpret_cast<uint8_t *>(weights_value.data()), weight_size));

  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2048);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  std::cout << ge_model->GetModelTaskDefPtr()->DebugString() << std::endl;

  return ge_root_model;
}

void EmitCodeFromNodes(const std::vector<AstNode *> &nodes, std::stringstream &output) {
  CppEmitter emitter;
  for (const auto *node : nodes) {
    if (node != nullptr) {
      std::string code_content;
      ASSERT_EQ(node->Accept(emitter, code_content), SUCCESS);
      output << code_content << '\n';
    }
  }
}

Status BuildProgram(ProgramGenerator &generator, Program &program, GeRootModelPtr &ge_root_model) {
  const auto &name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
  GE_ASSERT_TRUE(!name_to_ge_model.empty(), "[OM2] No subgraphs found in ge_root_model");
  GE_ASSERT_SUCCESS(generator.Init(name_to_ge_model.begin()->second));
  return SUCCESS;
}
}  // namespace

class ProgramGeneratorUt : public testing::Test {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(ProgramGeneratorUt, GenerateMakefile_Ok) {
  ProgramGenerator generator;
  Program program;
  GeRootModelPtr ge_root_model = CreateGeRootModelWithAicoreOp();
  ASSERT_EQ(BuildProgram(generator, program, ge_root_model), SUCCESS);
  ASSERT_EQ(generator.GenerateMakeFile(program), SUCCESS);

  auto &makefile_nodes = program[static_cast<size_t>(GeneratedFileIndex::kCMakeListsFile)];
  ASSERT_FALSE(makefile_nodes.empty());

  std::stringstream output;
  EmitCodeFromNodes(makefile_nodes, output);

  const std::string expected = R"(CANN_ROOT ?= $(ASCEND_HOME_PATH)
USE_STUB_LIB ?= 0

CXX := g++
TARGET := libg1_om2.so
SRC_FILES := g1_resources.cpp g1_kernel_reg.cpp g1_load_and_run.cpp g1_args_manager.cpp

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
  ASSERT_EQ(output.str(), expected + "\n");
}

TEST_F(ProgramGeneratorUt, GenerateResourcesSource_Ok) {
  ProgramGenerator generator;
  Program program;
    GeRootModelPtr ge_root_model = CreateGeRootModelWithAicoreOp();
    ASSERT_EQ(BuildProgram(generator, program, ge_root_model), SUCCESS);
  ASSERT_EQ(generator.GenerateResourcesSource(program), SUCCESS);

  auto &resources_nodes = program[static_cast<size_t>(GeneratedFileIndex::kResourcesFile)];
  ASSERT_FALSE(resources_nodes.empty());

  std::stringstream output;
  EmitCodeFromNodes(resources_nodes, output);
  const std::string expected = R"(#include "g1_interface.h"

namespace om2 {
Om2Model::Om2Model(const char **bin_files, const void **bin_data, size_t *bin_size, size_t bin_num, void *host_weight_mem_ptr, uint64_t *session_id)
  : host_weight_mem_ptr_(host_weight_mem_ptr), session_id_(session_id), kernel_id_(0) {
  for (size_t i = 0; i < bin_num; ++i) {
    bin_info_map_[std::string(bin_files[i])] = BinDataInfo{bin_data[i], bin_size[i]};
  }
  stream_list_.resize(1);
}

Om2Model::~Om2Model() {
  (void)ReleaseResources();
}
aclError Om2Model::InitResources() {
  // 1. 创建 model
  OM2_CHK_STATUS(aclmdlRIBuildBegin(&model_handle_, 0));

  // 2. 申请内存
  OM2_CHK_STATUS(aclrtMallocAlign32(&total_dev_mem_ptr_, 0, ACL_MEM_MALLOC_HUGE_FIRST));
  OM2_CHK_STATUS(aclrtMallocAlign32(&total_weight_mem_ptr_, 0, ACL_MEM_MALLOC_HUGE_FIRST));

  // 3. 下沉权重
  OM2_CHK_STATUS(aclrtMemcpy(total_weight_mem_ptr_, 0, host_weight_mem_ptr_, 0, ACL_MEMCPY_HOST_TO_DEVICE));

  // 4. 创建其他资源
  // 创建下沉Stream并绑定模型
  uint32_t stream0_flag = ACL_STREAM_PERSISTENT;
  OM2_CHK_STATUS(aclrtCreateStreamWithConfig(&stream_list_[0], 0, stream0_flag));
  OM2_CHK_STATUS(aclmdlRIBindStream(model_handle_, stream_list_[0], ACL_MODEL_STREAM_FLAG_HEAD));
  is_stream_list_bind_ = true;
  args_table_.Init();
  return ACL_SUCCESS;
}

aclError Om2Model::ReleaseResources() {

  if (is_stream_list_bind_) {
    for (auto stream : stream_list_) {
      OM2_CHK_STATUS(aclmdlRIUnbindStream(model_handle_, stream));
    }
  }
  for (auto stream : stream_list_) {
    OM2_CHK_STATUS(aclrtDestroyStream(stream));
  }

  OM2_CHK_STATUS(aclmdlRIDestroy(model_handle_));
  OM2_CHK_STATUS(aclrtFree(total_dev_mem_ptr_));
  OM2_CHK_STATUS(aclrtFree(total_weight_mem_ptr_));
  for (int i = 0; i < dev_ext_info_mem_ptrs_.size(); i++) {
    if (dev_ext_info_mem_ptrs_[i] != nullptr) {
      OM2_CHK_STATUS(aclrtFree(dev_ext_info_mem_ptrs_[i]));
    }
  }
  return ACL_SUCCESS;
}

} // namespace om2)";
  ASSERT_EQ(output.str(), expected + "\n");
}

TEST_F(ProgramGeneratorUt, GenerateInterfaceHeader_Ok) {
  ProgramGenerator generator;
  Program program;
    GeRootModelPtr ge_root_model = CreateGeRootModelWithAicoreOp();
    ASSERT_EQ(BuildProgram(generator, program, ge_root_model), SUCCESS);
  ASSERT_EQ(generator.GenerateInterfaceHeader(program), SUCCESS);

  auto &header_nodes = program[static_cast<size_t>(GeneratedFileIndex::kInterfaceHeaderFile)];
  ASSERT_FALSE(header_nodes.empty());

  std::stringstream output;
  EmitCodeFromNodes(header_nodes, output);
  const std::string expected = R"(#include <iostream>
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

#define OM2_CHK_STATUS(expr, ...)            \
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

template<typename T>
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

namespace om2 {
constexpr int32_t INPUT_NUM = 2;
constexpr int32_t OUTPUT_NUM = 1;
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
};
class Om2ArgsTable {
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
};

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
  aclmdlRI model_handle_;
  std::vector<aclrtStream> stream_list_;
  void *total_dev_mem_ptr_;
  void *total_weight_mem_ptr_;
  bool is_stream_list_bind_;
  std::unordered_map<std::string, BinDataInfo> bin_info_map_;
  Om2ArgsTable args_table_;
  uint64_t *session_id_;
  uint64_t kernel_id_;
  std::vector<void *> dev_ext_info_mem_ptrs_;
};

} // namespace om2
#ifdef __cplusplus
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
  ASSERT_EQ(output.str(), expected + "\n");
}

TEST_F(ProgramGeneratorUt, GenerateLoadAndRunSource_Ok) {
  ProgramGenerator generator;
  Program program;
  GeRootModelPtr ge_root_model = CreateGeRootModelWithAicoreOp();
  ASSERT_EQ(BuildProgram(generator, program, ge_root_model), SUCCESS);
  ASSERT_EQ(generator.GenerateKernelRegSource(program), SUCCESS);
  ASSERT_EQ(generator.GenerateLoadAndRunSource(program), SUCCESS);
  ASSERT_EQ(generator.GenerateArgsManagerSource(program), SUCCESS);

  auto &kernel_reg_nodes = program[static_cast<size_t>(GeneratedFileIndex::kKernelRegistryFile)];
  ASSERT_FALSE(kernel_reg_nodes.empty());
  std::stringstream kernel_reg_output;
  EmitCodeFromNodes(kernel_reg_nodes, kernel_reg_output);
  const std::string kernel_reg_expected = R"OM2(#include "g1_interface.h"

namespace om2 {
namespace {
constexpr uint32_t kMaxJsonFileLen = 512U;
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
};
BinaryBuffer ReadBinaryFileToBuffer(const std::string &file_path) {
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
}

aclError GenerateJsonFile(const AicpuRegisterInfo &register_info, std::string &json_path) {
  using namespace std::chrono;
  int64_t cur_timestamp = duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
  json_path = "/tmp/temp_ops_info_" + std::to_string(cur_timestamp) + ".json";
  std::string json_data_format = R"(
{
    "%s":{
        "opInfo":{
            "opKernelLib":"%s",
            "kernelSo":"%s",
            "functionName":"%s"
        }
    }
}
)";
  char json_data[kMaxJsonFileLen];
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
}

void AssembleAicpuLoadOptions(aclrtBinaryLoadOptions &load_options, int32_t cpu_kernel_mode) {
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
}
} // namespace
aclError Om2Model::RegisterKernels() {
  OM2_CHK_STATUS(RegisterAicoreKernel(bin_handles_[0], func_handles_[0], {ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE, "name", "name.o"}, bin_info_map_));
  return ACL_SUCCESS;
}
} // namespace om2
)OM2";
  ASSERT_EQ(kernel_reg_output.str(), kernel_reg_expected);

  auto &args_manager_nodes = program[static_cast<size_t>(GeneratedFileIndex::kArgsManagerFile)];
  ASSERT_FALSE(args_manager_nodes.empty());
  std::stringstream args_manager_output;
  EmitCodeFromNodes(args_manager_nodes, args_manager_output);
  const std::string args_manager_expected = R"(#include "g1_interface.h"

namespace om2 {
aclError Om2ArgsTable::Init() {
  args_size_ = 168;
  host_args_.clear();
  host_args_.resize(args_size_);
  OM2_CHK_STATUS(aclrtMalloc(&dev_args_, args_size_, ACL_MEM_MALLOC_HUGE_FIRST));
  args_info_ = {
    {GetHostArgAddr(0), GetDevArgAddr(0), 168},
  };
  iow_args_addrs_ = {
    GetHostArgAddr(0),
    GetHostArgAddr(8),
    GetHostArgAddr(16),
  };
  return ACL_SUCCESS;
}

Om2ArgsTable::~Om2ArgsTable() {
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

} // namespace om2
)";
  ASSERT_EQ(args_manager_output.str(), args_manager_expected);

  auto &load_and_run_nodes = program[static_cast<size_t>(GeneratedFileIndex::kLoadingAndRunningFile)];
  ASSERT_FALSE(load_and_run_nodes.empty());

  std::stringstream output;
  EmitCodeFromNodes(load_and_run_nodes, output);
  const std::string expected = R"(#include "rt_external_kernel.h"
#include "g1_interface.h"

namespace om2 {
namespace {
constexpr const size_t max_launch_cfg_num = 8UL;
constexpr int64_t kDImEndFlag = std::numeric_limits<int64_t>::min();
struct LaunchKernelCfgHolder {
  aclrtLaunchKernelCfg cfg{};
  aclrtLaunchKernelAttr attrs[max_launch_cfg_num];
};

struct LaunchKernelConfig {
  uint8_t schedule_mode{0U};
  aclrtEngineType engine_type{ACL_RT_ENGINE_TYPE_AIC};
  uint32_t block_dim_offset{0U};
  bool is_block_task_prefetch{false};
  bool is_data_dump{false};
  uint16_t time_out{0U};
  uint32_t local_memory_size{0U};
};

void AssembleLaunchConfig(LaunchKernelCfgHolder &holder, const LaunchKernelConfig &launch_config) {
  size_t actual_cfg_num = 0UL;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
  holder.attrs[actual_cfg_num].value.schemMode = launch_config.schedule_mode;
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
  holder.attrs[actual_cfg_num].value.engineType = launch_config.engine_type;
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_BLOCKDIM_OFFSET;
  holder.attrs[actual_cfg_num].value.blockDimOffset = launch_config.block_dim_offset;
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_BLOCK_TASK_PREFETCH;
  holder.attrs[actual_cfg_num].value.isBlockTaskPrefetch =
     static_cast<uint8_t>(launch_config.is_block_task_prefetch);
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_DATA_DUMP;
  holder.attrs[actual_cfg_num].value.isDataDump = static_cast<uint8_t>(launch_config.is_data_dump);
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
  holder.attrs[actual_cfg_num].value.timeout = launch_config.time_out;
  actual_cfg_num++;
  holder.cfg.attrs = &holder.attrs[0];
  holder.cfg.numAttrs = actual_cfg_num;
}

aclError KernelTaskDistribute(const std::vector<uint64_t>& io_addrs, ArgsInfo *args_info, aclrtFuncHandle func_handle,
                              uint32_t block_dim, aclrtStream stream, aclrtLaunchKernelCfg *config) {
  OM2_CHK_NOTNULL(args_info);
  OM2_CHK_STATUS(memcpy_s(args_info->host_addr, args_info->size, io_addrs.data(), io_addrs.size() * sizeof(uint64_t)));
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

} // namespace
aclError Om2Model::Load() {
  dev_ext_info_mem_ptrs_.resize(0);
  // ============================= add1 ===============================
  auto op2_input0 = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto op2_input1 = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto op2_output0 = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto op2_ws0 = GET_ADDR(total_dev_mem_ptr_, 0);
  LaunchKernelCfgHolder op2_cfg_holder;
  AssembleLaunchConfig(op2_cfg_holder, {0U, ACL_RT_ENGINE_TYPE_AIC, 0U, false, false, 0U});
  OM2_CHK_STATUS((KernelTaskDistribute(FlattenHostArgs(op2_input0, op2_input1, op2_output0, op2_ws0), args_table_.GetArgsInfo(0), func_handles_[0], 8, stream_list_[0], &op2_cfg_holder.cfg)));

  OM2_CHK_STATUS(aclmdlRIBuildEnd(model_handle_, nullptr));
  return ACL_SUCCESS;
}

aclError Om2Model::RunAsync(
  aclrtStream &exe_stream,
  size_t input_count,
  void **input_data,
  size_t output_count,
  void **output_data) {
  if (input_count != om2::INPUT_NUM || output_count != om2::OUTPUT_NUM) {
    return ACL_ERROR_FAILURE;
  }
  auto dev_input0_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto input_data_0_tensor = reinterpret_cast<gert::Tensor *>(input_data[0]);
  OM2_CHK_STATUS(aclrtMemcpyAsync(dev_input0_ptr, input_data_0_tensor->GetSize(), input_data_0_tensor->GetAddr(), input_data_0_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE, exe_stream));
  auto dev_input1_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto input_data_1_tensor = reinterpret_cast<gert::Tensor *>(input_data[1]);
  OM2_CHK_STATUS(aclrtMemcpyAsync(dev_input1_ptr, input_data_1_tensor->GetSize(), input_data_1_tensor->GetAddr(), input_data_1_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE, exe_stream));

  OM2_CHK_STATUS(args_table_.CopyArgsToDevice());
  OM2_CHK_STATUS(aclmdlRIExecuteAsync(model_handle_, exe_stream));
  auto dev_output0_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto output_data_0_tensor = reinterpret_cast<gert::Tensor *>(output_data[0]);
  OM2_CHK_STATUS(aclrtMemcpyAsync(output_data_0_tensor->GetAddr(), output_data_0_tensor->GetSize(), dev_output0_ptr, output_data_0_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE, exe_stream));

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
  auto dev_input0_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto input_data_0_tensor = reinterpret_cast<gert::Tensor *>(input_data[0]);
  OM2_CHK_STATUS(aclrtMemcpy(dev_input0_ptr, input_data_0_tensor->GetSize(), input_data_0_tensor->GetAddr(), input_data_0_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE));
  auto dev_input1_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto input_data_1_tensor = reinterpret_cast<gert::Tensor *>(input_data[1]);
  OM2_CHK_STATUS(aclrtMemcpy(dev_input1_ptr, input_data_1_tensor->GetSize(), input_data_1_tensor->GetAddr(), input_data_1_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE));

  OM2_CHK_STATUS(args_table_.CopyArgsToDevice());
  OM2_CHK_STATUS(aclmdlRIExecute(model_handle_, -1));
  auto dev_output0_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto output_data_0_tensor = reinterpret_cast<gert::Tensor *>(output_data[0]);
  OM2_CHK_STATUS(aclrtMemcpy(output_data_0_tensor->GetAddr(), output_data_0_tensor->GetSize(), dev_output0_ptr, output_data_0_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE));

  return ACL_SUCCESS;
}

} // namespace om2
aclError Om2ModelCreate(om2::Om2ModelHandle *model_handle, const char **bin_files, const void **bin_data, size_t *bin_size, int bin_num, void *host_weight_mem_ptr, uint64_t *session_id) {
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
  ASSERT_EQ(output.str(), expected + "\n");
}

TEST_F(ProgramGeneratorUt, GenerateLoadAndRunSource2_Ok) {
  ProgramGenerator generator;
  Program program;
  GeRootModelPtr ge_root_model = CreateGeRootModelWithAicoreOp2();
  ASSERT_EQ(BuildProgram(generator, program, ge_root_model), SUCCESS);
  ASSERT_EQ(generator.GenerateKernelRegSource(program), SUCCESS);
  ASSERT_EQ(generator.GenerateLoadAndRunSource(program), SUCCESS);

  auto &load_and_run_nodes = program[static_cast<size_t>(GeneratedFileIndex::kLoadingAndRunningFile)];
  ASSERT_FALSE(load_and_run_nodes.empty());

  std::stringstream output;
  EmitCodeFromNodes(load_and_run_nodes, output);
  const std::string expected = R"(#include "rt_external_kernel.h"
#include "g1_interface.h"

namespace om2 {
namespace {
constexpr const size_t max_launch_cfg_num = 8UL;
constexpr int64_t kDImEndFlag = std::numeric_limits<int64_t>::min();
struct LaunchKernelCfgHolder {
  aclrtLaunchKernelCfg cfg{};
  aclrtLaunchKernelAttr attrs[max_launch_cfg_num];
};

struct LaunchKernelConfig {
  uint8_t schedule_mode{0U};
  aclrtEngineType engine_type{ACL_RT_ENGINE_TYPE_AIC};
  uint32_t block_dim_offset{0U};
  bool is_block_task_prefetch{false};
  bool is_data_dump{false};
  uint16_t time_out{0U};
  uint32_t local_memory_size{0U};
};

void AssembleLaunchConfig(LaunchKernelCfgHolder &holder, const LaunchKernelConfig &launch_config) {
  size_t actual_cfg_num = 0UL;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
  holder.attrs[actual_cfg_num].value.schemMode = launch_config.schedule_mode;
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
  holder.attrs[actual_cfg_num].value.engineType = launch_config.engine_type;
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_BLOCKDIM_OFFSET;
  holder.attrs[actual_cfg_num].value.blockDimOffset = launch_config.block_dim_offset;
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_BLOCK_TASK_PREFETCH;
  holder.attrs[actual_cfg_num].value.isBlockTaskPrefetch =
     static_cast<uint8_t>(launch_config.is_block_task_prefetch);
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_DATA_DUMP;
  holder.attrs[actual_cfg_num].value.isDataDump = static_cast<uint8_t>(launch_config.is_data_dump);
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
  holder.attrs[actual_cfg_num].value.timeout = launch_config.time_out;
  actual_cfg_num++;
  holder.cfg.attrs = &holder.attrs[0];
  holder.cfg.numAttrs = actual_cfg_num;
}

aclError KernelTaskDistribute(const std::vector<uint64_t>& io_addrs, ArgsInfo *args_info, aclrtFuncHandle func_handle,
                              uint32_t block_dim, aclrtStream stream, aclrtLaunchKernelCfg *config) {
  OM2_CHK_NOTNULL(args_info);
  OM2_CHK_STATUS(memcpy_s(args_info->host_addr, args_info->size, io_addrs.data(), io_addrs.size() * sizeof(uint64_t)));
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

} // namespace
aclError Om2Model::Load() {
  dev_ext_info_mem_ptrs_.resize(0);
  // ============================= add1 ===============================
  auto op2_input0 = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto op2_input1 = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto op2_output0 = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto op2_ws0 = GET_ADDR(total_dev_mem_ptr_, 0);
  LaunchKernelCfgHolder op2_cfg_holder;
  AssembleLaunchConfig(op2_cfg_holder, {0U, ACL_RT_ENGINE_TYPE_AIC, 0U, false, false, 0U});
  OM2_CHK_STATUS((KernelTaskDistribute(FlattenHostArgs(op2_input0, op2_input1, op2_output0, op2_ws0), args_table_.GetArgsInfo(0), func_handles_[0], 8, stream_list_[0], &op2_cfg_holder.cfg)));

  OM2_CHK_STATUS(aclmdlRIBuildEnd(model_handle_, nullptr));
  return ACL_SUCCESS;
}

aclError Om2Model::RunAsync(
  aclrtStream &exe_stream,
  size_t input_count,
  void **input_data,
  size_t output_count,
  void **output_data) {
  if (input_count != om2::INPUT_NUM || output_count != om2::OUTPUT_NUM) {
    return ACL_ERROR_FAILURE;
  }
  auto dev_input0_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto input_data_0_tensor = reinterpret_cast<gert::Tensor *>(input_data[0]);
  OM2_CHK_STATUS(aclrtMemcpyAsync(dev_input0_ptr, input_data_0_tensor->GetSize(), input_data_0_tensor->GetAddr(), input_data_0_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE, exe_stream));
  auto dev_input1_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto input_data_1_tensor = reinterpret_cast<gert::Tensor *>(input_data[1]);
  OM2_CHK_STATUS(aclrtMemcpyAsync(dev_input1_ptr, input_data_1_tensor->GetSize(), input_data_1_tensor->GetAddr(), input_data_1_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE, exe_stream));

  OM2_CHK_STATUS(args_table_.CopyArgsToDevice());
  OM2_CHK_STATUS(aclmdlRIExecuteAsync(model_handle_, exe_stream));
  auto dev_output0_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto output_data_0_tensor = reinterpret_cast<gert::Tensor *>(output_data[0]);
  OM2_CHK_STATUS(aclrtMemcpyAsync(output_data_0_tensor->GetAddr(), output_data_0_tensor->GetSize(), dev_output0_ptr, output_data_0_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE, exe_stream));

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
  auto dev_input0_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto input_data_0_tensor = reinterpret_cast<gert::Tensor *>(input_data[0]);
  OM2_CHK_STATUS(aclrtMemcpy(dev_input0_ptr, input_data_0_tensor->GetSize(), input_data_0_tensor->GetAddr(), input_data_0_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE));
  auto dev_input1_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto input_data_1_tensor = reinterpret_cast<gert::Tensor *>(input_data[1]);
  OM2_CHK_STATUS(aclrtMemcpy(dev_input1_ptr, input_data_1_tensor->GetSize(), input_data_1_tensor->GetAddr(), input_data_1_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE));

  OM2_CHK_STATUS(args_table_.CopyArgsToDevice());
  OM2_CHK_STATUS(aclmdlRIExecute(model_handle_, -1));
  auto dev_output0_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto output_data_0_tensor = reinterpret_cast<gert::Tensor *>(output_data[0]);
  OM2_CHK_STATUS(aclrtMemcpy(output_data_0_tensor->GetAddr(), output_data_0_tensor->GetSize(), dev_output0_ptr, output_data_0_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE));

  return ACL_SUCCESS;
}

} // namespace om2
aclError Om2ModelCreate(om2::Om2ModelHandle *model_handle, const char **bin_files, const void **bin_data, size_t *bin_size, int bin_num, void *host_weight_mem_ptr, uint64_t *session_id) {
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
  ASSERT_EQ(output.str(), expected + "\n");
}

TEST_F(ProgramGeneratorUt, GenerateLoadAndRunSourceForAicpu_Ok) {
  ProgramGenerator generator;
  Program program;
  GeRootModelPtr ge_root_model = CreateGeRootModelWithAicpuOp();
  ASSERT_EQ(BuildProgram(generator, program, ge_root_model), SUCCESS);
  ASSERT_EQ(generator.GenerateKernelRegSource(program), SUCCESS);
  ASSERT_EQ(generator.GenerateLoadAndRunSource(program), SUCCESS);

  auto &load_and_run_nodes = program[static_cast<size_t>(GeneratedFileIndex::kLoadingAndRunningFile)];
  ASSERT_FALSE(load_and_run_nodes.empty());

  std::stringstream output;
  EmitCodeFromNodes(load_and_run_nodes, output);
  const std::string expected = R"(#include "rt_external_kernel.h"
#include "g1_interface.h"

namespace om2 {
namespace {
constexpr const size_t max_launch_cfg_num = 8UL;
constexpr int64_t kDImEndFlag = std::numeric_limits<int64_t>::min();
struct LaunchKernelCfgHolder {
  aclrtLaunchKernelCfg cfg{};
  aclrtLaunchKernelAttr attrs[max_launch_cfg_num];
};

struct LaunchKernelConfig {
  uint8_t schedule_mode{0U};
  aclrtEngineType engine_type{ACL_RT_ENGINE_TYPE_AIC};
  uint32_t block_dim_offset{0U};
  bool is_block_task_prefetch{false};
  bool is_data_dump{false};
  uint16_t time_out{0U};
  uint32_t local_memory_size{0U};
};

void AssembleLaunchConfig(LaunchKernelCfgHolder &holder, const LaunchKernelConfig &launch_config) {
  size_t actual_cfg_num = 0UL;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
  holder.attrs[actual_cfg_num].value.schemMode = launch_config.schedule_mode;
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
  holder.attrs[actual_cfg_num].value.engineType = launch_config.engine_type;
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_BLOCKDIM_OFFSET;
  holder.attrs[actual_cfg_num].value.blockDimOffset = launch_config.block_dim_offset;
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_BLOCK_TASK_PREFETCH;
  holder.attrs[actual_cfg_num].value.isBlockTaskPrefetch =
     static_cast<uint8_t>(launch_config.is_block_task_prefetch);
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_DATA_DUMP;
  holder.attrs[actual_cfg_num].value.isDataDump = static_cast<uint8_t>(launch_config.is_data_dump);
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
  holder.attrs[actual_cfg_num].value.timeout = launch_config.time_out;
  actual_cfg_num++;
  holder.cfg.attrs = &holder.attrs[0];
  holder.cfg.numAttrs = actual_cfg_num;
}

aclError KernelTaskDistribute(const std::vector<uint64_t>& io_addrs, ArgsInfo *args_info, aclrtFuncHandle func_handle,
                              uint32_t block_dim, aclrtStream stream, aclrtLaunchKernelCfg *config) {
  OM2_CHK_NOTNULL(args_info);
  OM2_CHK_STATUS(memcpy_s(args_info->host_addr, args_info->size, io_addrs.data(), io_addrs.size() * sizeof(uint64_t)));
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

constexpr uint32_t kAicpuArgsExtInfoAddrOffset = 12U;
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

} // namespace
aclError Om2Model::Load() {
  dev_ext_info_mem_ptrs_.resize(2);
  // ============================= add1 ===============================
  auto op2_input0 = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto op2_input1 = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto op2_output0 = GET_ADDR(total_dev_mem_ptr_, 1024);
  std::vector<uint64_t> op2_iow_addr = FlattenHostArgs(op2_input0, op2_input1, op2_output0);
  const char* op2_args_str = "\104\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000";
  const char* op2_ext_info_str = "\001\000\000\000\210\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\002\000\000\000\104\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\005\000\000\000\021\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\006\000\000\000\010\000\000\000\001\000\000\000\000\000\000\000\003\000\000\000\004\000\000\000\001\000\000\000\007\000\000\000\004\000\000\000\001\000\000\000";
  std::vector<uint8_t> op2_args(68);
  AssembleAicpuExtInfo(reinterpret_cast<uint8_t*>(const_cast<char*>(op2_ext_info_str)), 285, 228, session_id_, &kernel_id_, dev_ext_info_mem_ptrs_, 0);
  AssembleAicpuArgs(reinterpret_cast<uint8_t*>(const_cast<char*>(op2_args_str)), 68, dev_ext_info_mem_ptrs_[0], 285, op2_iow_addr, op2_args.data());
  LaunchKernelCfgHolder op2_cfg_holder;
  AssembleLaunchConfig(op2_cfg_holder, {0U, ACL_RT_ENGINE_TYPE_AIC, 0U, false, false, 0U});
  OM2_CHK_STATUS((AicpuKernelTaskDistribute(op2_args, args_table_.GetArgsInfo(0), func_handles_[0], 8, stream_list_[0], &op2_cfg_holder.cfg)));

  // ============================= add2 ===============================
  auto op3_output0 = GET_ADDR(total_dev_mem_ptr_, 1024);
  std::vector<uint64_t> op3_iow_addr = FlattenHostArgs(op2_input0, op2_output0, op3_output0);
  const char* op3_args_str = "\104\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000";
  const char* op3_ext_info_str = "\001\000\000\000\210\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\002\000\000\000\104\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\005\000\000\000\021\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\006\000\000\000\010\000\000\000\001\000\000\000\000\000\000\000\003\000\000\000\004\000\000\000\001\000\000\000\007\000\000\000\004\000\000\000\001\000\000\000";
  std::vector<uint8_t> op3_args(68);
  AssembleAicpuExtInfo(reinterpret_cast<uint8_t*>(const_cast<char*>(op3_ext_info_str)), 285, 228, session_id_, &kernel_id_, dev_ext_info_mem_ptrs_, 1);
  AssembleAicpuArgs(reinterpret_cast<uint8_t*>(const_cast<char*>(op3_args_str)), 68, dev_ext_info_mem_ptrs_[1], 285, op3_iow_addr, op3_args.data());
  LaunchKernelCfgHolder op3_cfg_holder;
  AssembleLaunchConfig(op3_cfg_holder, {0U, ACL_RT_ENGINE_TYPE_AIC, 0U, false, false, 0U});
  OM2_CHK_STATUS((AicpuKernelTaskDistribute(op3_args, args_table_.GetArgsInfo(1), func_handles_[0], 8, stream_list_[0], &op3_cfg_holder.cfg)));

  OM2_CHK_STATUS(aclmdlRIBuildEnd(model_handle_, nullptr));
  return ACL_SUCCESS;
}

aclError Om2Model::RunAsync(
  aclrtStream &exe_stream,
  size_t input_count,
  void **input_data,
  size_t output_count,
  void **output_data) {
  if (input_count != om2::INPUT_NUM || output_count != om2::OUTPUT_NUM) {
    return ACL_ERROR_FAILURE;
  }
  auto dev_input0_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto input_data_0_tensor = reinterpret_cast<gert::Tensor *>(input_data[0]);
  OM2_CHK_STATUS(aclrtMemcpyAsync(dev_input0_ptr, input_data_0_tensor->GetSize(), input_data_0_tensor->GetAddr(), input_data_0_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE, exe_stream));
  auto dev_input1_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto input_data_1_tensor = reinterpret_cast<gert::Tensor *>(input_data[1]);
  OM2_CHK_STATUS(aclrtMemcpyAsync(dev_input1_ptr, input_data_1_tensor->GetSize(), input_data_1_tensor->GetAddr(), input_data_1_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE, exe_stream));

  OM2_CHK_STATUS(args_table_.CopyArgsToDevice());
  OM2_CHK_STATUS(aclmdlRIExecuteAsync(model_handle_, exe_stream));
  auto dev_output0_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto output_data_0_tensor = reinterpret_cast<gert::Tensor *>(output_data[0]);
  OM2_CHK_STATUS(aclrtMemcpyAsync(output_data_0_tensor->GetAddr(), output_data_0_tensor->GetSize(), dev_output0_ptr, output_data_0_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE, exe_stream));

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
  auto dev_input0_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto input_data_0_tensor = reinterpret_cast<gert::Tensor *>(input_data[0]);
  OM2_CHK_STATUS(aclrtMemcpy(dev_input0_ptr, input_data_0_tensor->GetSize(), input_data_0_tensor->GetAddr(), input_data_0_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE));
  auto dev_input1_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto input_data_1_tensor = reinterpret_cast<gert::Tensor *>(input_data[1]);
  OM2_CHK_STATUS(aclrtMemcpy(dev_input1_ptr, input_data_1_tensor->GetSize(), input_data_1_tensor->GetAddr(), input_data_1_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE));

  OM2_CHK_STATUS(args_table_.CopyArgsToDevice());
  OM2_CHK_STATUS(aclmdlRIExecute(model_handle_, -1));
  auto dev_output0_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto output_data_0_tensor = reinterpret_cast<gert::Tensor *>(output_data[0]);
  OM2_CHK_STATUS(aclrtMemcpy(output_data_0_tensor->GetAddr(), output_data_0_tensor->GetSize(), dev_output0_ptr, output_data_0_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE));

  return ACL_SUCCESS;
}

} // namespace om2
aclError Om2ModelCreate(om2::Om2ModelHandle *model_handle, const char **bin_files, const void **bin_data, size_t *bin_size, int bin_num, void *host_weight_mem_ptr, uint64_t *session_id) {
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
  ASSERT_EQ(output.str(), expected + "\n");
}


TEST_F(ProgramGeneratorUt, GenerateLoadAndRunSourceForDynamicIo_Ok) {
  ProgramGenerator generator;
  Program program;
  GeRootModelPtr ge_root_model = CreateGeRootModelWithAicoreOpOfDynamicIo();
  ASSERT_EQ(BuildProgram(generator, program, ge_root_model), SUCCESS);
  ASSERT_EQ(generator.GenerateKernelRegSource(program), SUCCESS);
  ASSERT_EQ(generator.GenerateLoadAndRunSource(program), SUCCESS);

  auto &load_and_run_nodes = program[static_cast<size_t>(GeneratedFileIndex::kLoadingAndRunningFile)];
  ASSERT_FALSE(load_and_run_nodes.empty());

  std::stringstream output;
  EmitCodeFromNodes(load_and_run_nodes, output);
  const std::string expected = R"(#include "rt_external_kernel.h"
#include "g1_interface.h"

namespace om2 {
namespace {
constexpr const size_t max_launch_cfg_num = 8UL;
constexpr int64_t kDImEndFlag = std::numeric_limits<int64_t>::min();
struct LaunchKernelCfgHolder {
  aclrtLaunchKernelCfg cfg{};
  aclrtLaunchKernelAttr attrs[max_launch_cfg_num];
};

struct LaunchKernelConfig {
  uint8_t schedule_mode{0U};
  aclrtEngineType engine_type{ACL_RT_ENGINE_TYPE_AIC};
  uint32_t block_dim_offset{0U};
  bool is_block_task_prefetch{false};
  bool is_data_dump{false};
  uint16_t time_out{0U};
  uint32_t local_memory_size{0U};
};

void AssembleLaunchConfig(LaunchKernelCfgHolder &holder, const LaunchKernelConfig &launch_config) {
  size_t actual_cfg_num = 0UL;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
  holder.attrs[actual_cfg_num].value.schemMode = launch_config.schedule_mode;
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
  holder.attrs[actual_cfg_num].value.engineType = launch_config.engine_type;
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_BLOCKDIM_OFFSET;
  holder.attrs[actual_cfg_num].value.blockDimOffset = launch_config.block_dim_offset;
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_BLOCK_TASK_PREFETCH;
  holder.attrs[actual_cfg_num].value.isBlockTaskPrefetch =
     static_cast<uint8_t>(launch_config.is_block_task_prefetch);
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_DATA_DUMP;
  holder.attrs[actual_cfg_num].value.isDataDump = static_cast<uint8_t>(launch_config.is_data_dump);
  actual_cfg_num++;
  holder.attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
  holder.attrs[actual_cfg_num].value.timeout = launch_config.time_out;
  actual_cfg_num++;
  holder.cfg.attrs = &holder.attrs[0];
  holder.cfg.numAttrs = actual_cfg_num;
}

aclError KernelTaskDistribute(const std::vector<uint64_t>& io_addrs, ArgsInfo *args_info, aclrtFuncHandle func_handle,
                              uint32_t block_dim, aclrtStream stream, aclrtLaunchKernelCfg *config) {
  OM2_CHK_NOTNULL(args_info);
  OM2_CHK_STATUS(memcpy_s(args_info->host_addr, args_info->size, io_addrs.data(), io_addrs.size() * sizeof(uint64_t)));
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

} // namespace
aclError Om2Model::Load() {
  dev_ext_info_mem_ptrs_.resize(0);
  // ============================= add1 ===============================
  auto op2_io_desc0 = args_table_.GetDevArgAddr(0);
  OM2_CHK_NOTNULL(op2_io_desc0);
  auto op2_io_desc1 = args_table_.GetDevArgAddr(56);
  OM2_CHK_NOTNULL(op2_io_desc1);
  auto op2_io_desc2 = args_table_.GetDevArgAddr(112);
  OM2_CHK_NOTNULL(op2_io_desc2);
  std::vector<int64_t> op2_shape_info0 = {48, 4294967300, 1, 1, 224, 224};
  auto op2_input0 = GET_ADDR(total_dev_mem_ptr_, 1024);
  std::vector<int64_t> op2_shape_info1 = {48, 4294967300, 1, 1, 224, 224};
  auto op2_input1 = GET_ADDR(total_dev_mem_ptr_, 1024);
  std::vector<int64_t> op2_shape_info2 = {48, 4294967300, 1, 1, 224, 224};
  auto op2_output0 = GET_ADDR(total_dev_mem_ptr_, 1024);
  LaunchKernelCfgHolder op2_cfg_holder;
  AssembleLaunchConfig(op2_cfg_holder, {0U, ACL_RT_ENGINE_TYPE_AIC, 0U, false, false, 0U});
  OM2_CHK_STATUS((KernelTaskDistribute(FlattenHostArgs(op2_io_desc0, op2_io_desc1, op2_io_desc2, op2_shape_info0, op2_input0, op2_shape_info1, op2_input1, op2_shape_info2, op2_output0), args_table_.GetArgsInfo(0), func_handles_[0], 8, stream_list_[0], &op2_cfg_holder.cfg)));

  OM2_CHK_STATUS(aclmdlRIBuildEnd(model_handle_, nullptr));
  return ACL_SUCCESS;
}

aclError Om2Model::RunAsync(
  aclrtStream &exe_stream,
  size_t input_count,
  void **input_data,
  size_t output_count,
  void **output_data) {
  if (input_count != om2::INPUT_NUM || output_count != om2::OUTPUT_NUM) {
    return ACL_ERROR_FAILURE;
  }
  auto dev_input0_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto input_data_0_tensor = reinterpret_cast<gert::Tensor *>(input_data[0]);
  OM2_CHK_STATUS(aclrtMemcpyAsync(dev_input0_ptr, input_data_0_tensor->GetSize(), input_data_0_tensor->GetAddr(), input_data_0_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE, exe_stream));
  auto dev_input1_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto input_data_1_tensor = reinterpret_cast<gert::Tensor *>(input_data[1]);
  OM2_CHK_STATUS(aclrtMemcpyAsync(dev_input1_ptr, input_data_1_tensor->GetSize(), input_data_1_tensor->GetAddr(), input_data_1_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE, exe_stream));

  OM2_CHK_STATUS(args_table_.CopyArgsToDevice());
  OM2_CHK_STATUS(aclmdlRIExecuteAsync(model_handle_, exe_stream));
  auto dev_output0_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto output_data_0_tensor = reinterpret_cast<gert::Tensor *>(output_data[0]);
  OM2_CHK_STATUS(aclrtMemcpyAsync(output_data_0_tensor->GetAddr(), output_data_0_tensor->GetSize(), dev_output0_ptr, output_data_0_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE, exe_stream));

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
  auto dev_input0_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto input_data_0_tensor = reinterpret_cast<gert::Tensor *>(input_data[0]);
  OM2_CHK_STATUS(aclrtMemcpy(dev_input0_ptr, input_data_0_tensor->GetSize(), input_data_0_tensor->GetAddr(), input_data_0_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE));
  auto dev_input1_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto input_data_1_tensor = reinterpret_cast<gert::Tensor *>(input_data[1]);
  OM2_CHK_STATUS(aclrtMemcpy(dev_input1_ptr, input_data_1_tensor->GetSize(), input_data_1_tensor->GetAddr(), input_data_1_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE));

  OM2_CHK_STATUS(args_table_.CopyArgsToDevice());
  OM2_CHK_STATUS(aclmdlRIExecute(model_handle_, -1));
  auto dev_output0_ptr = GET_ADDR(total_dev_mem_ptr_, 1024);
  auto output_data_0_tensor = reinterpret_cast<gert::Tensor *>(output_data[0]);
  OM2_CHK_STATUS(aclrtMemcpy(output_data_0_tensor->GetAddr(), output_data_0_tensor->GetSize(), dev_output0_ptr, output_data_0_tensor->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE));

  return ACL_SUCCESS;
}

} // namespace om2
aclError Om2ModelCreate(om2::Om2ModelHandle *model_handle, const char **bin_files, const void **bin_data, size_t *bin_size, int bin_num, void *host_weight_mem_ptr, uint64_t *session_id) {
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
  ASSERT_EQ(output.str(), expected + "\n");
}
} // namespace ge

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "depends/profiler/src/profiling_test_util.h"
#include "common/global_variables/diagnose_switch.h"

#include "macro_utils/dt_public_scope.h"
#include "ge_running_env/fake_ops_kernel_builder.h"
#include "engines/manager/opskernel_manager/ops_kernel_builder_manager.h"
#include "framework/executor/ge_executor.h"
#include "init_ge.h"
#include "ge_running_env/path_utils.h"
#include "api/atc/main_impl.h"
#include "runtime/rt.h"
#include "framework/executor/ge_executor.h"
#include "framework/generator/ge_generator.h"
#include "single_op/single_op.h"
#include "utils/model_data_builder.h"
#include "single_op/task/tbe_task_builder.h"
#include "utils/tensor_descs.h"
#include "utils/data_buffers.h"
#include "utils/mock_runtime.h"
#include "register/op_tiling_registry.h"
#include "graph/debug/ge_attr_define.h"
#include "hybrid/node_executor/ge_local/ge_local_node_executor.h"
#include "graph/manager/mem_manager.h"
#include "utils/bench_env.h"
#include "utils/taskdef_builder.h"
#include "hybrid/model/hybrid_model_builder.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "aicpu_task_struct.h"
#include "hybrid/node_executor/aicpu/aicpu_ext_info_handler.h"
#include "depends/runtime/src/runtime_stub.h"
#include "graph/operator_reg.h"
#include "single_op/single_op_manager.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "register/op_impl_registry.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "depends/op_stub/op_impl/dynamicatomicaddrclean/dynamic_atomic_addr_clean_impl.h"
#include "macro_utils/dt_public_unscope.h"

#include "ge_running_env/fake_op.h"
#include "utils/mock_ops_kernel_builder.h"
#include "base/common/helper/model_parser_base.h"
#include "framework/common/helper/model_helper.h"
#include "framework/runtime/subscriber/global_dumper.h"
#include "framework/ge_runtime_stub/include/stub/gert_runtime_stub.h"
#include "framework/ge_runtime_stub/include/faker/fake_allocator.h"
#include "depends/profiler/src/dump_stub.h"
#include "common/error_tracking/error_tracking.h"
#include "session/inner_session.h"
#include "common/dump/dump_manager.h"
#include "common/opskernel/ops_kernel_info_types.h"

USING_FAKE_NS
namespace ge {
constexpr int64_t kMemtypeHostCompileIndependent = 2;
REG_OP(FakeOpForSingleOp)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
        .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
        .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
        .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
        .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
        .ATTR(transpose_x1, Bool, false)
        .ATTR(transpose_x2, Bool, false)
        .ATTR(offset_x, Int, 0)
        .OP_END_FACTORY_REG(FakeOpForSingleOp);

namespace {
using GenerateTaskFun = std::function<Status(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks)>;
constexpr size_t kHostMemInputIndex = 0U;
void ResetFlags() {
}
bool IsFile(const std::string &filename) {
  struct stat buffer;
  return (stat(filename.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

bool IsDirectory(const std::string &file_folder) {
  struct stat buffer;
  return (stat(file_folder.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

// remove dir and files inside
int RemoveFileAndDirectory(const std::string &path) {
  int result = 0;
  DIR *p_dir;
  struct dirent *p_dirent;
  if (IsDirectory(path)) {
    if ((p_dir = opendir(path.c_str())) == nullptr) {
      return -1;
    }

    while ((p_dirent = readdir(p_dir)) != nullptr) {
      std::string file_name = path + "/" + p_dirent->d_name;
      // It is a directory
      if (IsDirectory(file_name) && (0 != strcmp(p_dirent->d_name, ".")) && (0 != strcmp(p_dirent->d_name, ".."))) {
        result = RemoveFileAndDirectory(file_name);
        if (result < 0) {
          return result;
        }
      }
        // It is a file
      else if ((0 != strcmp(p_dirent->d_name, ".")) && (0 != strcmp(p_dirent->d_name, ".."))) {
        result = remove(file_name.c_str());
        if (result < 0) {
          return result;
        }
      }
    }
    closedir(p_dir);
    result = rmdir(path.c_str());
  } else if (IsFile(path)) {
    result = remove(path.c_str());
  }
  return result;
}

bool IsLogValid(std::vector<gert::OneLog> logs, const std::string &expected_log) {
  for (auto &log :logs) {
    std::string content = log.content;
    if (content.find(expected_log) != std::string::npos) {
      return true;
    }
  }
  return false;
}

Status CompileSingleOpByAtc(const std::string &json_path) {
  auto test_dir = PathJoin(GetRunPath().c_str(), "st_run_data/single_op_output");
  Mkdir(test_dir.c_str());
  std::string json_path_arg = "--singleop=" + json_path;
  std::string output_arg = "--output=" + test_dir;
  const char *argv[] = {"atc",
                        "--soc_version=Ascend310",
                        "--log=error",
                        json_path_arg.c_str(),
                        output_arg.c_str(),
  };
  auto ret = main_impl(sizeof(argv) / sizeof(argv[0]), (char **) argv);
  ResetFlags();
  return ret;
}

Status GenerateTaskForAicpu(const Node &node, RunContext &run_context, std::vector<domi::TaskDef> &tasks) {
  if (node.GetType() != NEG) {
    return SUCCESS;
  }

  tasks.emplace_back(AicpuTaskDefBuilder(node).BuildAicpuTask(0));
  return SUCCESS;
}

Status GenerateTaskForTfKernel(const Node &node, RunContext &run_context, std::vector<domi::TaskDef> &tasks) {
  if (node.GetType() != NEG) {
    return SUCCESS;
  }

  tasks.emplace_back(AicpuTaskDefBuilder(node).BuildTfTask(0));
  return SUCCESS;
}

Status GenerateTaskForTfDependCompute(const Node &node, RunContext &run_context, std::vector<domi::TaskDef> &tasks) {
  if (node.GetType() != NEG) {
    return SUCCESS;
  }

  tasks.emplace_back(AicpuTaskDefBuilder(node).BuildTfTask(4));
  tasks.emplace_back(AicpuTaskDefBuilder(node).BuildTfTask(0));
  AttrUtils::SetInt(node.GetOpDesc(), ATTR_NAME_UNKNOWN_SHAPE_TYPE, 4);
  return SUCCESS;
}

Status GenerateTaskForAiCpuDependCompute(const Node &node, RunContext &run_context, std::vector<domi::TaskDef> &tasks) {
  if (node.GetType() != NEG) {
    return SUCCESS;
  }

  tasks.emplace_back(AicpuTaskDefBuilder(node).BuildAicpuTask(4));
  tasks.emplace_back(AicpuTaskDefBuilder(node).BuildAicpuTask(0));
  AttrUtils::SetInt(node.GetOpDesc(), ATTR_NAME_UNKNOWN_SHAPE_TYPE, 4);
  return SUCCESS;
}

Status GenerateTaskForAiCore(const Node &node,
                             RunContext &run_context,
                             std::vector<domi::TaskDef> &tasks) {
  if (node.GetType() != RELU) {
    return SUCCESS;
  }

  tasks.emplace_back(AiCoreTaskDefBuilder(node).BuildTask());
  return SUCCESS;
}

Status GenerateTaskForMixAiCore(const Node &node, RunContext &run_context, std::vector<domi::TaskDef> &tasks) {
  if (node.GetType() != RELU) {
    return SUCCESS;
  }
  auto aicore_task = AiCoreTaskDefBuilder(node).BuildTask();
  auto kernel_info = aicore_task.mutable_kernel();
  kernel_info->_impl_.context_->set_args_format("{ffts_addr}{i_instance0}{o_instance0}");
  tasks.emplace_back(aicore_task);
  return SUCCESS;
}

Status GenerateTaskForDynamicAiCore(const Node &node,
                                    RunContext &run_context,
                                    std::vector<domi::TaskDef> &tasks) {
  if (node.GetType() != RELU) {
    return SUCCESS;
  }

  tasks.emplace_back(AiCoreTaskDefBuilder(node).BuildTask(true));
  return SUCCESS;
}

Status GenerateTaskForDynamicAiCoreV2(const Node &node,
                                      RunContext &run_context,
                                      std::vector<domi::TaskDef> &tasks) {
  if (node.GetType() != RELU) {
    return SUCCESS;
  }

  tasks.emplace_back(AiCoreTaskDefBuilder(node).BuildTaskWithHandle());
  return SUCCESS;
}

Status GenerateTaskForDynamicAiCoreWithAtomicAddrClean(const Node &node,
                                                       RunContext &run_context,
                                                       std::vector<domi::TaskDef> &tasks) {
  if (node.GetType() != RELU) {
    return SUCCESS;
  }

  tasks.emplace_back(AiCoreTaskDefBuilder(node).BuildAtomicAddrCleanTask());
  tasks.emplace_back(AiCoreTaskDefBuilder(node).BuildTask(true));
  return SUCCESS;
}

Status GenerateTaskForMixl2(const Node &node,
                            RunContext &run_context,
                            std::vector<domi::TaskDef> &tasks) {
  if (node.GetType() != PRELU) {
    return SUCCESS;
  }

  tasks.emplace_back(MixL2TaskDefBuilder(node).BuildTask(false));
  return SUCCESS;
}

Status GenerateTaskForMemcpyAsync(const Node &node,
                                  RunContext &run_context,
                                  std::vector<domi::TaskDef> &tasks) {
  if (node.GetType() != MEMCPYASYNC) {
    return SUCCESS;
  }
  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
  auto kernel_def = task_def.mutable_memcpy_async();
  kernel_def->set_op_index(node.GetOpDesc()->GetId());
  kernel_def->set_count(64);
  kernel_def->set_dst_max(64);
  kernel_def->set_src(0);
  kernel_def->set_dst(64);
  kernel_def->set_kind(RT_MEMCPY_ADDR_DEVICE_TO_DEVICE);
  tasks.emplace_back(task_def);
  return SUCCESS;
}

Status GenerateTaskForNpuClearFloatStatus(const Node &node, RunContext &run_context,
                                          std::vector<domi::TaskDef> &tasks) {
  if (node.GetType() != NPUCLEARFLOATSTATUS) {
    return SUCCESS;
  }
  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NPU_CLEAR_FLOAT_STATUS));
  auto kernel_def = task_def.mutable_npu_clear_float_status();
  kernel_def->set_mode(0);
  kernel_def->set_op_index(1);
  tasks.emplace_back(task_def);
  return SUCCESS;
}

Status GenerateTaskForNpuGetFloatStatus(const Node &node, RunContext &run_context,
                                        std::vector<domi::TaskDef> &tasks) {
  if (node.GetType() != NPUGETFLOATSTATUS) {
    return SUCCESS;
  }
  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NPU_GET_FLOAT_STATUS));
  auto kernel_def = task_def.mutable_npu_get_float_status();
  kernel_def->set_mode(0);
  kernel_def->set_output_addr(0);
  kernel_def->set_output_size(8);
  tasks.emplace_back(task_def);
  return SUCCESS;
}

Status GenerateTaskForNpuClearFloatDebugStatus(const Node &node, RunContext &run_context,
                                               std::vector<domi::TaskDef> &tasks) {
  if (node.GetType() != NPUCLEARFLOATDEBUGSTATUS) {
    return SUCCESS;
  }
  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NPU_CLEAR_DEBUG_FLOAT_STATUS));
  auto kernel_def = task_def.mutable_npu_clear_float_debug_status();
  kernel_def->set_mode(0);
  kernel_def->set_op_index(1);
  tasks.emplace_back(task_def);
  return SUCCESS;
}

Status GenerateTaskForNpuGetFloatDebugStatus(const Node &node, RunContext &run_context,
                                             std::vector<domi::TaskDef> &tasks) {
  if (node.GetType() != NPUGETFLOATDEBUGSTATUS) {
    return SUCCESS;
  }
  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NPU_GET_DEBUG_FLOAT_STATUS));
  auto kernel_def = task_def.mutable_npu_get_float_debug_status();
  kernel_def->set_mode(0);
  kernel_def->set_output_addr(0);
  kernel_def->set_output_size(8);
  tasks.emplace_back(task_def);
  return SUCCESS;
}

Status GenerateTaskForDsa(const Node &node, RunContext &run_context,
                          std::vector<domi::TaskDef> &tasks) {
  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_DSA));
  auto dsa_task = task_def.mutable_dsa_task();
  dsa_task->set_op_index(node.GetOpDesc()->GetId());
  dsa_task->set_start(1);
  dsa_task->set_sqe_type(1);
  dsa_task->set_distribution_type(1);
  dsa_task->set_data_type(1);
  dsa_task->set_alg_type(1);
  dsa_task->set_input_vld(1);
  dsa_task->set_input_value_addr_flag(1);
  if (node.GetInNodesSize() == 5U) {
    node.GetOpDesc()->SetWorkspace({0});
    node.GetOpDesc()->SetWorkspaceBytes({0});
    dsa_task->set_input1_value_or_ptr(0);
    dsa_task->set_seed_value_or_ptr(0);
    dsa_task->set_random_count_value_or_ptr(0);
  } else {
    node.GetOpDesc()->SetWorkspace({0,0});
    node.GetOpDesc()->SetWorkspaceBytes({0,0});
    dsa_task->set_input1_value_or_ptr(1);
    dsa_task->set_seed_value_or_ptr(1);
    dsa_task->set_random_count_value_or_ptr(1);
    domi::DSATaskArgsDef *dsa_task_args = dsa_task->mutable_args();
    dsa_task_args->set_seed_value_or_addr("5");
    dsa_task_args->set_random_count_value_or_addr("6");
    dsa_task_args->set_input1_value_or_addr("1");
    dsa_task_args->set_input2_value_or_addr("2");
  }
  tasks.emplace_back(task_def);
  return SUCCESS;
}
}  // namespace

class SingleOpTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    GEFinalize();
    ReInitGe();
    BenchEnv::Init();
    optiling::OpTilingFuncV2 tilingfun = [](const ge::Operator &op,
                                            const optiling::OpCompileInfoV2 &compile_info,
                                            optiling::OpRunInfoV2 &run_info) -> bool {
      run_info.SetWorkspaces({1024});
      return true;
    };

    optiling::OpTilingRegistryInterf_V2(RELU, tilingfun);
    REGISTER_OP_TILING_UNIQ_V2(ReLU, tilingfun, 1);
  }

  void SetUp() {
    env_.InstallDefault();
    rtStreamCreate(&stream_, 0);
    gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(0);
  }

  void TearDown() {
    MockRuntime::SetInstance(nullptr);
    GeExecutor::ReleaseSingleOpResource(stream_);
    GeExecutor::ReleaseResource();
    rtStreamDestroy(stream_);
    GEFinalize();
    ReInitGe(); // the main_impl will call GEFinalize, so re-init after call it

  }

  uint64_t model_id = 0;
  rtStream_t stream_ = nullptr;
  std::vector<std::unique_ptr<uint8_t[]>> input_buffers_;
  std::vector<std::unique_ptr<uint8_t[]>> output_buffers_;
  std::unique_ptr<FakeAllocator> allocator_;
  GeRunningEnvFaker env_;

  static Status BuildSingleOp(OpDescPtr &op_desc, ModelBufferData &model_buffer) {
    vector<GeTensor> inputs;
    vector<GeTensor> outputs;
    std::vector<GeTensorDesc> input_desc;
    std::vector<GeTensorDesc> output_desc;
    for (const auto &tensor_desc : op_desc->GetAllInputsDescPtr()) {
      inputs.emplace_back(GeTensor(*tensor_desc));
      input_desc.emplace_back(*tensor_desc);
    }
    for (const auto &tensor_desc : op_desc->GetAllOutputsDescPtr()) {
      outputs.emplace_back(GeTensor(*tensor_desc));
      output_desc.emplace_back(*tensor_desc);
    }
    GeGenerator generator;
    generator.Initialize({});
    auto ret = generator.BuildSingleOpModel(op_desc, inputs, outputs, ENGINE_SYS, false, model_buffer);
    return ret;
  }

  Status RunStaticTestCast(OpDescPtr &op_desc, ge::Allocator *allocator = nullptr) {
    ModelBufferData model_buffer;
    EXPECT_EQ(BuildSingleOp(op_desc, model_buffer), SUCCESS);
    ModelData model_data;
    model_data.model_data = model_buffer.data.get();
    model_data.model_len = model_buffer.length;
    SingleOp *single_op = nullptr;
    GeExecutor ge_executor;
    if (allocator != nullptr) {
      ge_executor.SetAllocator(stream_, allocator);
    }
    EXPECT_EQ(ge_executor.LoadSingleOpV2("aicore_op", model_data, stream_, &single_op, model_id++), SUCCESS);

    std::vector<DataBuffer> inputs;
    std::vector<DataBuffer> outputs;
    std::vector<GeTensorDesc> input_desc;
    std::vector<GeTensorDesc> output_desc;
    CreateInputsAndOutputs(op_desc, input_desc, inputs, output_desc, outputs);
    return GeExecutor::ExecuteAsync(single_op, inputs, outputs);
  }

  Status RunStaticTestCast2(OpDescPtr &op_desc, ge::Allocator *allocator = nullptr) {
    ModelBufferData model_buffer;
    EXPECT_EQ(BuildSingleOp(op_desc, model_buffer), SUCCESS);
    ModelData model_data;
    model_data.model_data = model_buffer.data.get();
    model_data.model_len = model_buffer.length;
    SingleOp *single_op = nullptr;
    GeExecutor ge_executor;
    if (allocator != nullptr) {
      ge_executor.SetAllocator(stream_, allocator);
    }
    EXPECT_EQ(ge_executor.LoadSingleOpV2("aicore_op", model_data, stream_, &single_op, model_id++), SUCCESS);

    std::vector<DataBuffer> inputs;
    std::vector<DataBuffer> outputs;
    std::vector<GeTensorDesc> input_desc;
    std::vector<GeTensorDesc> output_desc;
    CreateInputsAndOutputs(op_desc, input_desc, inputs, output_desc, outputs);
    EXPECT_EQ(GeExecutor::ExecuteAsync(single_op, inputs, outputs), SUCCESS);
    return GeExecutor::ExecuteAsync(single_op, inputs, outputs);
  }

  Status RunDynamicAicpu(OpDescPtr &op_desc) {
    ModelBufferData model_buffer;
    EXPECT_EQ(BuildSingleOp(op_desc, model_buffer), SUCCESS);

    std::vector<std::pair<int64_t, int64_t>> actual_origin_shape_range;
    for (size_t i = 0; i < op_desc->GetAllInputsDesc().size(); i++) {
      EXPECT_EQ(op_desc->GetInputDesc(i).GetShape().IsUnknownShape(), true);
      op_desc->GetInputDesc(i).GetOriginShapeRange(actual_origin_shape_range);
      EXPECT_TRUE(actual_origin_shape_range.empty());
    }

    for (size_t i = 0; i < op_desc->GetAllOutputsDesc().size(); i++) {
      EXPECT_EQ(op_desc->GetOutputDesc(i).GetShape().IsUnknownShape(), true);
      op_desc->GetOutputDesc(i).GetOriginShapeRange(actual_origin_shape_range);
      EXPECT_TRUE(actual_origin_shape_range.empty());
    }
    return SUCCESS;
  }

  Status LoadDynamicSingleOp(OpDescPtr &op_desc, DynamicSingleOp **single_op) {
    ModelBufferData model_buffer;
    EXPECT_EQ(BuildSingleOp(op_desc, model_buffer), SUCCESS);

    ModelData model_data;
    model_data.model_data = model_buffer.data.get();
    model_data.model_len = model_buffer.length;
    return GeExecutor::LoadDynamicSingleOpV2("dyn_op", model_data, stream_, single_op, model_id++);
  }

  void FillDataForHostMemInput(const GeTensorDescPtr &tensor_desc, DataBuffer &data_buffer) {
    int64_t mem_type = 0;
    AttrUtils::GetInt(tensor_desc, ge::ATTR_NAME_PLACEMENT, mem_type);
    if (mem_type == kMemtypeHostCompileIndependent) {
      data_buffer.placement = kHostMemType;
      uint64_t *data_ptr = PtrToPtr<void, uint64_t>(data_buffer.data);
      for (size_t i = 0; i < data_buffer.length / sizeof(uint64_t); i++) {
        data_ptr[i] = kHostMemInputValue;
      }
    }
  }

  void CreateInputsAndOutputs(OpDescPtr &op_desc,
                              std::vector<GeTensorDesc> &input_desc,
                              std::vector<DataBuffer> &inputs,
                              std::vector<GeTensorDesc> &output_desc,
                              std::vector<DataBuffer> &outputs) {
    for (const auto &tensor_desc : op_desc->GetAllInputsDescPtr()) {
      bool is_const = false;
      AttrUtils::GetBool(tensor_desc, CONST_ATTR_NAME_INPUT, is_const);
      if (is_const) {
        continue;
      }
      input_desc.emplace_back(*tensor_desc);
      int64_t tensor_size = -1;
      TensorUtils::GetTensorSizeInBytes(*tensor_desc, tensor_size);
      EXPECT_GE(tensor_size, 0);
      input_buffers_.emplace_back(MakeUnique<uint8_t[]>(tensor_size));
      DataBuffer data_buffer;
      data_buffer.data = input_buffers_.back().get();
      data_buffer.length = tensor_size;
      FillDataForHostMemInput(tensor_desc, data_buffer);
      inputs.emplace_back(data_buffer);
    }
    for (const auto &tensor_desc : op_desc->GetAllOutputsDescPtr()) {
      output_desc.emplace_back(*tensor_desc);
      int64_t tensor_size = -1;
      TensorUtils::GetTensorSizeInBytes(*tensor_desc, tensor_size);
      EXPECT_GE(tensor_size, 0);
      output_buffers_.emplace_back(MakeUnique<uint8_t[]>(tensor_size));
      DataBuffer data_buffer;
      data_buffer.data = output_buffers_.back().get();
      data_buffer.length = tensor_size;
      outputs.emplace_back(data_buffer);
    }
  }

  Status RunDynamicTestCast(OpDescPtr &op_desc) {
    DynamicSingleOp *single_op = nullptr;
    EXPECT_EQ(LoadDynamicSingleOp(op_desc, &single_op), SUCCESS);

    std::vector<DataBuffer> inputs;
    std::vector<DataBuffer> outputs;
    std::vector<GeTensorDesc> input_desc;
    std::vector<GeTensorDesc> output_desc;
    CreateInputsAndOutputs(op_desc, input_desc, inputs, output_desc, outputs);
    return GeExecutor::ExecuteAsync(single_op, input_desc, inputs, output_desc, outputs);
  }

  OpDescPtr CreateOp(const std::string &op_type) {
    GeShape shape({2, 8});
    GeTensorDesc tensor_desc(shape);
    auto op_desc = std::make_shared<OpDesc>(op_type, op_type);
    op_desc->AddInputDesc("x", tensor_desc);
    op_desc->AddOutputDesc("y", tensor_desc);
    return op_desc;
  }

  OpDescPtr CreateAicpuOp(const std::string &op_type) {
    GeShape shape({2, 8});
    GeTensorDesc tensor_desc(shape);
    auto op_desc = std::make_shared<OpDesc>(op_type, op_type);
    op_desc->AddInputDesc("x", tensor_desc);
    op_desc->AddOutputDesc("y", tensor_desc);
    op_desc->SetOpEngineName("DNN_VM_AICPU_ASCEND");
    op_desc->SetOpKernelLibName("aicpu_tf_kernel");
    (void)AttrUtils::SetBool(op_desc, "_AllShape", true);
    return op_desc;
  }

  OpDescPtr CreateOpWithHostMemInput(const std::string &op_type) {
    auto op_desc = std::make_shared<OpDesc>(op_type, op_type);
    // input 0
    GeShape shape0({4});
    GeTensorDesc tensor_desc0(shape0, FORMAT_ND, DT_UINT64);
    AttrUtils::SetInt(tensor_desc0, ge::ATTR_NAME_PLACEMENT, kMemtypeHostCompileIndependent);
    op_desc->AddInputDesc("x0", tensor_desc0);

    GeShape shape1({1});
    GeTensorDesc tensor_desc1(shape1, FORMAT_ND, DT_UINT64);
    AttrUtils::SetInt(tensor_desc1, ge::ATTR_NAME_PLACEMENT, kMemtypeHostCompileIndependent);
    op_desc->AddInputDesc("x1", tensor_desc1);

    GeShape shape2({2, 8});
    GeTensorDesc tensor_desc2(shape2, FORMAT_ND, DT_UINT64);
    op_desc->AddOutputDesc("y", tensor_desc2);
    return op_desc;
  }
};

TEST_F(SingleOpTest, TestStaticAicpu) {
  MockForGenerateTask("AicpuLib", GenerateTaskForAicpu);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpu);
  auto op_desc = CreateOp(NEG);
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
}

TEST_F(SingleOpTest, TestDynamicAicpu) {
  MockForGenerateTask("AicpuLib", GenerateTaskForAicpu);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpu);
  env_.Reset()
      .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeEngine(kEngineNameAiCpu).KernelInfoStore(kEngineNameAiCpu))
      .Install(FakeEngine(kEngineNameAiCpuTf).KernelInfoStore(kEngineNameAiCpuTf))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(NEG).InfoStoreAndBuilder("DNN_VM_AICPU_ASCEND"));
  auto op_desc = CreateAicpuOp(NEG);
  ModelBufferData model_buffer;
  EXPECT_EQ(RunDynamicAicpu(op_desc), SUCCESS);
}

TEST_F(SingleOpTest, TestStaticTfKernel) {
  MockForGenerateTask("AicpuLib", GenerateTaskForTfKernel);
  MockForGenerateTask("aicpu_tf_kernel", GenerateTaskForTfKernel);
  auto op_desc = CreateOp(NEG);
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
}

TEST_F(SingleOpTest, TestStaticTfKernelFailedByEngineSelect) {
  MockForGenerateTask("AicpuLib", GenerateTaskForTfKernel);
  MockForGenerateTask("aicpu_tf_kernel", GenerateTaskForTfKernel);
  auto op_desc = CreateOp(NEG);
  AttrUtils::SetStr(op_desc, "_exclude_engines", "AiCpu");
  ModelBufferData model_buffer;
  EXPECT_NE(BuildSingleOp(op_desc, model_buffer), SUCCESS);
  GetThreadLocalContext().SetGraphOption({});  // option clear
}

//  static single op, aicore host mem input
TEST_F(SingleOpTest, TestStaticAicoreHostMem) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  auto runtime_stub = MockForKernelLaunchWithHostMemInput();
  auto op_desc = CreateOpWithHostMemInput(RELU);
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
}

TEST_F(SingleOpTest, TestStaticMixAicoreHostMem) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForMixAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForMixAiCore);
  auto runtime_stub = MockForKernelLaunchWithHostMemInput();
  auto op_desc = CreateOpWithHostMemInput(RELU);
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
}

//  static single op, aicpu host mem input
TEST_F(SingleOpTest, TestStaticAicpuKernelHostMem) {
  MockForGenerateTask("AicpuLib", GenerateTaskForAicpu);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpu);
  auto runtime_stub = MockForKernelLaunchWithHostMemInput();
  auto op_desc = CreateOpWithHostMemInput(NEG);
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
}

//  static single op, aicpu host mem input
TEST_F(SingleOpTest, TestStaticTfAicpuKernelHostMem) {
  MockForGenerateTask("AicpuLib", GenerateTaskForTfKernel);
  MockForGenerateTask("aicpu_tf_kernel", GenerateTaskForTfKernel);
  auto runtime_stub = MockForKernelLaunchWithHostMemInput();
  auto op_desc = CreateOpWithHostMemInput(NEG);
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
}

// dynamic single op, aicore host mem input
TEST_F(SingleOpTest, TestDynamicAicoreWithHostMem) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForDynamicAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForDynamicAiCore);
  auto runtime_stub = MockForKernelLaunchWithHostMemInput();
  auto op_desc = CreateOpWithHostMemInput(RELU);
  EXPECT_EQ(RunDynamicTestCast(op_desc), SUCCESS);
}

// dynamic single op, tf aicpu host mem input
TEST_F(SingleOpTest, TestDynamicTfAicpuWithHostMem) {
  MockForGenerateTask("AicpuLib", GenerateTaskForTfDependCompute);
  MockForGenerateTask("aicpu_tf_kernel", GenerateTaskForTfDependCompute);
  auto runtime_stub = MockForKernelLaunchWithHostMemInput();
  auto op_desc = CreateOpWithHostMemInput(NEG);
  EXPECT_EQ(RunDynamicTestCast(op_desc), SUCCESS);
}

// dynamic single op, aicpu host mem input
TEST_F(SingleOpTest, TestDynamicAicpuWithHostMem) {
  MockForGenerateTask("AicpuLib", GenerateTaskForAiCpuDependCompute);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAiCpuDependCompute);
  auto runtime_stub = MockForKernelLaunchWithHostMemInput();
  auto op_desc = CreateOpWithHostMemInput(NEG);
  EXPECT_EQ(RunDynamicTestCast(op_desc), SUCCESS);
}

TEST_F(SingleOpTest, test_delete_resource) {
  int32_t fake_stream = 0;
  uint64_t op_id = 999;
  auto stream = reinterpret_cast<rtStream_t>(fake_stream);
  auto &instance = SingleOpManager::GetInstance();
  uintptr_t resource_id = 0U;
  ASSERT_EQ(instance.GetResourceId(stream, resource_id), SUCCESS);
  auto res = instance.GetResource(resource_id, stream);
  ASSERT_NE(res, nullptr);
  auto new_op = MakeUnique<SingleOp>(res, &res->stream_mu_, res->stream_);
  res->op_map_[op_id] = std::move(new_op);
  auto new_dynamic_op = MakeUnique<DynamicSingleOp>(&res->tensor_pool_, res->resource_id_, &res->stream_mu_, res->stream_);
  res->dynamic_op_map_[op_id] = std::move(new_dynamic_op);
  ASSERT_EQ(ge::GeExecutor::UnloadSingleOp(op_id), SUCCESS);
  ASSERT_EQ(ge::GeExecutor::UnloadDynamicSingleOp(op_id), SUCCESS);
  ASSERT_EQ(instance.ReleaseResource(stream), SUCCESS);
}

UINT32 StubTiling(gert::TilingContext *) {
  return ge::GRAPH_SUCCESS;
}

UINT32 StubTilingParse(gert::KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

void* CompileInfoCreator() {
  auto tmp =  ge::MakeUnique<char>();
  return tmp.get();
}


TEST_F(SingleOpTest, TestStaticAiCore) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  auto op_desc = CreateOp(RELU);
  AttrUtils::SetBool(op_desc, ge::ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 512);
  AttrUtils::SetBool(op_desc, "support_dynamicshape", true);

  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl(RELU);
  funcs->tiling = StubTiling;
  funcs->tiling_parse = StubTilingParse;
  funcs->compile_info_creator = CompileInfoCreator;

  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
}

TEST_F(SingleOpTest, TestStaticAiCoreErrorTracking) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  auto op_desc1 = CreateOp(RELU);
  AttrUtils::SetBool(op_desc1, ge::ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  AttrUtils::SetStr(op_desc1, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc1, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc1, "op_para_size", 512);
  AttrUtils::SetBool(op_desc1, "support_dynamicshape", true);
  AttrUtils::SetListStr(op_desc1, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, {"op1", "op2", "op3"});

  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl(RELU);
  funcs->tiling = StubTiling;
  funcs->tiling_parse = StubTilingParse;
  funcs->compile_info_creator = CompileInfoCreator;

  EXPECT_EQ(RunStaticTestCast(op_desc1), SUCCESS);
  rtExceptionInfo rt_exception_info;
  rt_exception_info.streamid = 0;
  rt_exception_info.taskid = 0;
  ErrorTrackingCallback(&rt_exception_info);
}

/**
 * 用例描述：静态shape单算子执行时上报cann profiling的host sch数据
 *
 * 预置条件：
 * 1. 构造静态单算子执行用例
 *
 * 测试步骤：
 * 1. 使能cann host profiling
 * 2. 构造静态单算子执行器并执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 校验profiling上报的host sch data中有正确的op type和op name的hash值
 */
TEST_F(SingleOpTest, ProfilingStaticSingleOp_RecortHostSchData) {
  ge::diagnoseSwitch::EnableCannHostProfiling();
  MockForGenerateTask("AiCoreLib", GenerateTaskForAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  auto op_desc = CreateOp(RELU);
  AttrUtils::SetBool(op_desc, ge::ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 512);
  AttrUtils::SetBool(op_desc, "support_dynamicshape", true);

  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl(RELU);
  funcs->tiling = StubTiling;
  funcs->tiling_parse = StubTilingParse;
  funcs->compile_info_creator = CompileInfoCreator;

  std::hash<std::string> hs;
  auto func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len) {
    struct MsprofHashData *hash_data;
    ReporterData *reporter_data;
    MsprofGeProfHostSchData *host_sch_data;
    char *str;
    switch (type) {
      case MSPROF_REPORTER_REPORT:
        reporter_data = reinterpret_cast<ReporterData *>(data);
        host_sch_data = reinterpret_cast<MsprofGeProfHostSchData *>(reporter_data->data);
        EXPECT_EQ(host_sch_data->element, hs("ReLU"));
        EXPECT_EQ(*reinterpret_cast<uint64_t *>(host_sch_data->reserve), hs("ReLU"));
        break;
      case MSPROF_REPORTER_HASH:
        hash_data = reinterpret_cast<struct MsprofHashData *>(data);
        str = reinterpret_cast<char *>(hash_data->data);
        hash_data->hashId = hs(str);
        break;
      default:
        break;
    }
    return 0;
  };
  ProfilingTestUtil::Instance().SetProfFunc(func);
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
  ge::diagnoseSwitch::DisableProfiling();
}

TEST_F(SingleOpTest, TestDynamicAiCore) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForDynamicAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForDynamicAiCore);
  auto op_desc = CreateOp(RELU);
  EXPECT_EQ(RunDynamicTestCast(op_desc), SUCCESS);
}

TEST_F(SingleOpTest, TestDynamicAiCoreV2) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForDynamicAiCoreV2);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForDynamicAiCoreV2);
  auto op_desc = CreateOp(RELU);
  EXPECT_EQ(RunDynamicTestCast(op_desc), SUCCESS);
}

TEST_F(SingleOpTest, TestDynamicAiCoreWithAtomicAddrClean) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForDynamicAiCoreWithAtomicAddrClean);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForDynamicAiCoreWithAtomicAddrClean);
  auto op_desc = CreateOp(RELU);
  EXPECT_EQ(RunDynamicTestCast(op_desc), SUCCESS);
}

UINT32 StubTilingMixST(gert::TilingContext *context) {
  context->SetNeedAtomic(false);
  context->SetTilingKey(666U);
  context->SetBlockDim(666U);
  auto tiling_data = context->GetTilingData<uint64_t>();
  *tiling_data = 100;
  return ge::GRAPH_SUCCESS;
}

UINT32 StubTilingParseMixST(gert::KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

void* CompileInfoCreatorMixST() {
  auto tmp =  ge::MakeUnique<char>();
  return tmp.get();
}

TEST_F(SingleOpTest, TestMixl2StaticOp) {
  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl(PRELU);
  funcs->tiling = StubTilingMixST;
  funcs->tiling_parse = StubTilingParseMixST;
  funcs->compile_info_creator = CompileInfoCreatorMixST;

  MockForGenerateTask("mix_l2", GenerateTaskForMixl2);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForMixl2);
  auto op_desc = CreateOp(PRELU);
  AttrUtils::SetBool(op_desc, ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 512);
  std::shared_ptr<ge::OpKernelBin> kernel_bin = std::make_shared<ge::OpKernelBin>("bin_name", std::vector<char>());
  void *stub_func = ValueToPtr(1234U);
  KernelBinRegistry::GetInstance().AddKernel("0/_tvmbin",
  std::unique_ptr<KernelHolder>(new KernelHolder((const char_t*)stub_func, kernel_bin)));

  ge::diagnoseSwitch::EnableProfiling({gert::ProfilingType::kTaskTime, gert::ProfilingType::kDevice});
  // EXPECT_NE(RunStaticTestCast(op_desc), SUCCESS);
  AttrUtils::SetStr(op_desc, ATTR_NAME_ALIAS_ENGINE_NAME, "mix_l2");
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
  ge::diagnoseSwitch::DisableProfiling();
}


TEST_F(SingleOpTest, TestMixl2StaticOp_WithL0ExceptionDump) {
  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kLiteExceptionDump}));

  gert::GertRuntimeStub runtime_stub;
  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl(PRELU);
  funcs->tiling = StubTilingMixST;
  funcs->tiling_parse = StubTilingParseMixST;
  funcs->compile_info_creator = CompileInfoCreatorMixST;

  MockForGenerateTask("mix_l2", GenerateTaskForMixl2);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForMixl2);
  auto op_desc = CreateOp(PRELU);
  AttrUtils::SetBool(op_desc, ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 512);
  std::shared_ptr<ge::OpKernelBin> kernel_bin = std::make_shared<ge::OpKernelBin>("bin_name", std::vector<char>());
  void *stub_func = ValueToPtr(1234U);
  KernelBinRegistry::GetInstance().AddKernel("0/_tvmbin",
  std::unique_ptr<KernelHolder>(new KernelHolder((const char_t*)stub_func, kernel_bin)));

  ge::diagnoseSwitch::EnableProfiling({gert::ProfilingType::kTaskTime, gert::ProfilingType::kDevice});
  AttrUtils::SetStr(op_desc, ATTR_NAME_ALIAS_ENGINE_NAME, "mix_l2");

  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);

  // check size info
  uint64_t *size_info = (uint64_t *)runtime_stub.GetRtsRuntimeStub().GetLiteEceptionArgs()[0];
  EXPECT_EQ(*(size_info + 1), 6U);  // all_len, index 2~7
  EXPECT_EQ(*(size_info + 2), 0U);  // ctxid, 0
  EXPECT_EQ(*(size_info + 3), 24U); // args table size, input1(8byte) | output1(8byte) | tilling(8byte)
  EXPECT_EQ(*(size_info + 4), 3U);  // size num, input1(1)| output1(1)|tilling(1)
  EXPECT_EQ(*(size_info + 5), 64U);  // input1 size, 64
  EXPECT_EQ(*(size_info + 6), 64U);  // output1 size, 74
  EXPECT_EQ(*(size_info + 7), 0x0300000000000008UL);  // tiling size, high32 bit: size type, low 32bit: data size

  ge::diagnoseSwitch::DisableProfiling();
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0UL);
  runtime_stub.Clear();
}

/**
 * 用例描述：ffts plus + soft sync的exception dump
 *
 * 预置条件：NA
 *
 * 测试步骤：
 * 1. 构造算力切分的mixl2算子
 * 2. 加载执行
 *
 * 预期结果：
 * 1、成功保存到该软同步算子的tiling 信息
 */
TEST_F(SingleOpTest, TestMixl2StaticOp_SaveExceptionDumpInfo_WithExceptionDumpOn) {
  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl(PRELU);
  funcs->tiling = StubTilingMixST;
  funcs->tiling_parse = StubTilingParseMixST;
  funcs->compile_info_creator = CompileInfoCreatorMixST;

  MockForGenerateTask("mix_l2", GenerateTaskForMixl2);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForMixl2);
  auto op_desc = CreateOp(PRELU);
  AttrUtils::SetBool(op_desc, ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 512);
  std::shared_ptr<ge::OpKernelBin> kernel_bin = std::make_shared<ge::OpKernelBin>("bin_name", std::vector<char>());
  void *stub_func = ValueToPtr(1234U);
  KernelBinRegistry::GetInstance().AddKernel("0/_tvmbin",
                                             std::unique_ptr<KernelHolder>(new KernelHolder((const char_t*)stub_func, kernel_bin)));

  // EXPECT_NE(RunStaticTestCast(op_desc), SUCCESS);
  AttrUtils::SetStr(op_desc, ATTR_NAME_ALIAS_ENGINE_NAME, "mix_l2");
  ge::diagnoseSwitch::EnableExceptionDump();
  gert::GlobalDumper::GetInstance()->MutableExceptionDumper()->Clear();
  ge::DumpStub::GetInstance().ClearOpInfos();

  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);

  const auto &op_info = DumpStub::GetInstance().GetOpInfos()[0];
  std::string tiling_key = AdxGetTilingKey(op_info);
  EXPECT_EQ(tiling_key, "666");
  ge::DumpStub::GetInstance().ClearOpInfos();
  ge::diagnoseSwitch::DisableDumper();
}


TEST_F(SingleOpTest, TestMixl2StaticOpHostMem) {
  MockForGenerateTask("mix_l2", GenerateTaskForMixl2);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForMixl2);
  // auto runtime_stub = MockForKernelLaunchWithHostMemInput();
  auto op_desc = CreateOpWithHostMemInput(PRELU);
  AttrUtils::SetStr(op_desc, ATTR_NAME_ALIAS_ENGINE_NAME, "mix_l2");
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
}

TEST_F(SingleOpTest, TestMemcpyAsync) {
  MockForGenerateTask("RTSLib", GenerateTaskForMemcpyAsync);
  auto op_desc = CreateOp(MEMCPYASYNC);
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
}

TEST_F(SingleOpTest, TestNpuClearFloatStatus) {
  MockForGenerateTask("RTSLib", GenerateTaskForNpuClearFloatStatus);
  auto op_desc = CreateOp(NPUCLEARFLOATSTATUS);
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
}

TEST_F(SingleOpTest, TestNpuGetFloatStatus) {
  MockForGenerateTask("RTSLib", GenerateTaskForNpuGetFloatStatus);
  auto op_desc = CreateOp(NPUGETFLOATSTATUS);
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
}

TEST_F(SingleOpTest, TestNpuClearFloatDebugStatus) {
  auto ge_env = GeRunningEnvFaker();
  auto infer_fun = [](Operator &op) -> graphStatus {
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    *op_desc->MutableOutputDesc(0) = *op_desc->GetInputDescPtr(0);
    return GRAPH_SUCCESS;
  };
  ge_env.Reset()
      .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("DNN_VM_RTS_OP_STORE"))
      .Install(FakeOp(NPUCLEARFLOATDEBUGSTATUS).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE").InferShape(infer_fun))
      .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"));
  MockForGenerateTask("DNN_VM_RTS_OP_STORE", GenerateTaskForNpuClearFloatDebugStatus);
  auto op_desc = CreateOp(NPUCLEARFLOATDEBUGSTATUS);
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
}

TEST_F(SingleOpTest, TestNpuGetFloatDebugStatus) {
  auto ge_env = GeRunningEnvFaker();
  auto infer_fun = [](Operator &op) -> graphStatus {
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    *op_desc->MutableOutputDesc(0) = *op_desc->GetInputDescPtr(0);
    return GRAPH_SUCCESS;
  };
  ge_env.Reset()
      .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("DNN_VM_RTS_OP_STORE"))
      .Install(FakeOp(NPUGETFLOATDEBUGSTATUS).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE").InferShape(infer_fun))
      .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"));

  MockForGenerateTask("DNN_VM_RTS_OP_STORE", GenerateTaskForNpuGetFloatDebugStatus);
  auto op_desc = CreateOp(NPUGETFLOATDEBUGSTATUS);
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
}

TEST_F(SingleOpTest, TestDsa) {
  MockForGenerateTask("AicpuLib", GenerateTaskForDsa);

  auto op_desc = std::make_shared<OpDesc>("DSA", NEG);
  GeTensorDesc tensor(GeShape({8}), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc(tensor);
  op_desc->AddInputDesc(tensor);
  op_desc->AddInputDesc(tensor);
  op_desc->AddInputDesc(tensor);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);

  auto op_desc1 = std::make_shared<OpDesc>("DSA1", NEG);
  op_desc1->AddInputDesc(tensor);
  op_desc1->AddInputDesc(tensor);
  op_desc1->AddInputDesc(tensor);
  op_desc1->AddInputDesc(tensor);
  op_desc1->AddOutputDesc(tensor);
  EXPECT_EQ(RunStaticTestCast(op_desc1), SUCCESS);
}

TEST_F(SingleOpTest, TestStaticAiCoreWithDump) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);

  GeExecutor ge_executor;
  ge::DumpConfig dump_cfg;
  dump_cfg.dump_path = "./dump/";
  dump_cfg.dump_mode = "all";
  dump_cfg.dump_status = "on";
  dump_cfg.dump_op_switch = "on";
  EXPECT_EQ(ge_executor.SetDump(dump_cfg), SUCCESS);

  auto op_desc = CreateOp(RELU);
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);

  dump_cfg.dump_status = "off";
  dump_cfg.dump_op_switch = "off";
  EXPECT_EQ(ge_executor.SetDump(dump_cfg), SUCCESS);
}

TEST_F(SingleOpTest, TestDynamicWithDump) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForDynamicAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForDynamicAiCore);

  GeExecutor ge_executor;
  ge::DumpConfig dump_cfg;
  dump_cfg.dump_path = "./dump/";
  dump_cfg.dump_mode = "all";
  dump_cfg.dump_status = "on";
  dump_cfg.dump_op_switch = "on";
  EXPECT_EQ(ge_executor.SetDump(dump_cfg), SUCCESS);

  auto op_desc = CreateOp(RELU);
  EXPECT_EQ(RunDynamicTestCast(op_desc), SUCCESS);

  dump_cfg.dump_status = "off";
  dump_cfg.dump_op_switch = "off";
  dump_cfg.dump_debug = "off";
  EXPECT_EQ(ge_executor.SetDump(dump_cfg), SUCCESS);
}

TEST_F(SingleOpTest, TestDumpOff) {
  DumpManager::GetInstance().callback_map_.clear();
  GeExecutor ge_executor;
  ge::DumpConfig dump_cfg;
  dump_cfg.dump_path = "./dump/";
  dump_cfg.dump_mode = "all";
  dump_cfg.dump_status = "on";
  dump_cfg.dump_op_switch = "on";
  EXPECT_EQ(ge_executor.SetDump(dump_cfg), SUCCESS);

  dump_cfg.dump_status = "off";
  dump_cfg.dump_debug = "off";
  EXPECT_EQ(ge_executor.SetDump(dump_cfg), SUCCESS);
}

TEST_F(SingleOpTest, TestMultipleSingleOp) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);

  {
    auto op_desc = CreateOp(RELU);
    EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
  }

  {
    auto op_desc = CreateOp(RELU);
    EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
  }
}

TEST_F(SingleOpTest, TestSingleOpWithConstant) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);

  auto op_desc = CreateOp(RELU);
  auto tensor = MakeShared<GeTensor>(*op_desc->MutableInputDesc(0));
  AttrUtils::SetBool(op_desc->MutableInputDesc(0), CONST_ATTR_NAME_INPUT, true);
  AttrUtils::SetTensor(op_desc->MutableInputDesc(0), ATTR_NAME_WEIGHTS, tensor);
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
}

TEST_F(SingleOpTest, TestTfDependCompute) {
  MockForGenerateTask("AicpuLib", GenerateTaskForTfDependCompute);
  MockForGenerateTask("aicpu_tf_kernel", GenerateTaskForTfDependCompute);

  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtKernelLaunchEx).WillRepeatedly(testing::Invoke(MockRtKernelLaunchEx));
  EXPECT_CALL(*runtime_stub, rtMemcpy).WillRepeatedly(testing::Invoke(MockRtMemcpy));

  auto op_desc = CreateOp(NEG);
  AttrUtils::SetInt(op_desc, ATTR_NAME_UNKNOWN_SHAPE_TYPE, static_cast<int64_t>(DEPEND_COMPUTE));

  EXPECT_EQ(RunDynamicTestCast(op_desc), SUCCESS);
}

TEST_F(SingleOpTest, TestTfDependRange) {
  auto fun = [](const Node &node, RunContext &run_context, std::vector<domi::TaskDef> &tasks) -> Status {
    if (node.GetType() != NEG) {
      return SUCCESS;
    }

    tasks.emplace_back(AicpuTaskDefBuilder(node).BuildTfTask(3));
    return SUCCESS;
  };
  MockForGenerateTask("AicpuLib", fun);
  MockForGenerateTask("aicpu_tf_kernel", fun);


  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtKernelLaunchEx).WillRepeatedly(testing::Invoke(MockRtKernelLaunchEx));
  EXPECT_CALL(*runtime_stub, rtMemcpy).WillRepeatedly(testing::Invoke(MockRtMemcpy));

  auto op_desc = CreateOp(NEG);
  AttrUtils::SetInt(op_desc, ATTR_NAME_UNKNOWN_SHAPE_TYPE, static_cast<int64_t>(DEPEND_SHAPE_RANGE));

  EXPECT_EQ(RunDynamicTestCast(op_desc), SUCCESS);
}

TEST_F(SingleOpTest, TestAiCpuDependCompute) {
  MockForGenerateTask("AicpuLib", GenerateTaskForAiCpuDependCompute);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAiCpuDependCompute);

  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtKernelLaunchEx).WillRepeatedly(testing::Invoke(MockRtKernelLaunchEx));
  EXPECT_CALL(*runtime_stub, rtMemcpy).WillRepeatedly(testing::Invoke(MockRtMemcpy));

  auto op_desc = CreateOp(NEG);
  AttrUtils::SetInt(op_desc, ATTR_NAME_UNKNOWN_SHAPE_TYPE, static_cast<int64_t>(DEPEND_COMPUTE));

  EXPECT_EQ(RunDynamicTestCast(op_desc), SUCCESS);
}

TEST_F(SingleOpTest, TestCompileDynSingleOp) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  GeShape shape({-2});
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginShape(shape);
  auto op_desc = std::make_shared<OpDesc>("add", "add");
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);

  vector<GeTensor> inputs;
  vector<GeTensor> outputs;
  inputs.emplace_back(GeTensor(tensor_desc));
  outputs.emplace_back(GeTensor(tensor_desc));

  GeGenerator generator;
  generator.Initialize({});
  ModelBufferData model_buffer;
  EXPECT_NE(generator.BuildSingleOpModel(op_desc, inputs, outputs, ENGINE_SYS, false, model_buffer), ge::SUCCESS);
}

TEST_F(SingleOpTest, TestSingleOpSetPrecisionMode_Failed_WhenValueIsInvalid) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);

  GeGenerator generator;
  std::map<std::string, std::string> options = {std::make_pair(PRECISION_MODE, "111")};
  ASSERT_NE(generator.Initialize(options), SUCCESS);
  generator.Finalize();
}

TEST_F(SingleOpTest, TestSingleOpRunFlagMode_Failed_WhenValueIsInvalid) {
  std::map<std::string, std::string> options_invalidParam;
  options_invalidParam.emplace(ge::RUN_FLAG, "2");
  GeGenerator generator;
  ASSERT_NE(generator.Initialize(options_invalidParam), SUCCESS);
}

TEST_F(SingleOpTest, TestSingleOpStreamNum_Failed_WhenValueIsInvalid) {
  std::map<std::string, std::string> options_invalidParam;
  GeGenerator generator;

  options_invalidParam.emplace(ge::STREAM_NUM, "a");
  ASSERT_NE(generator.Initialize(options_invalidParam), SUCCESS);
}

TEST_F(SingleOpTest, TestSingleOpStreamMaxParallelNum_Failed_WhenValueIsInvalid) {
  std::map<std::string, std::string> options_invalidParam;
  GeGenerator generator;

  options_invalidParam.emplace("ge.streamMaxParallelNum", "AIcoreEngine,8");
  ASSERT_NE(generator.Initialize(options_invalidParam), SUCCESS);
  options_invalidParam.erase("ge.streamMaxParallelNum");
  options_invalidParam.emplace("ge.streamMaxParallelNum", "AIcoreEngine:0");
  ASSERT_NE(generator.Initialize(options_invalidParam), SUCCESS);
  options_invalidParam.erase("ge.streamMaxParallelNum");
  options_invalidParam.emplace("ge.streamMaxParallelNum", "AIcoreEngine:a");
  ASSERT_NE(generator.Initialize(options_invalidParam), SUCCESS);
  options_invalidParam.erase("ge.streamMaxParallelNum");
  options_invalidParam.emplace("ge.streamMaxParallelNum", "AIcoreEngine:");
  ASSERT_NE(generator.Initialize(options_invalidParam), SUCCESS);
}

TEST_F(SingleOpTest, TestSingleOpEgeineName_Failed_WhenValueIsInvalid) {
  std::map<std::string, std::string> options_invalidParam;
  GeGenerator generator;

  options_invalidParam.emplace("ge.streamMaxParallelNum", ":");
  ASSERT_NE(generator.Initialize(options_invalidParam), SUCCESS);
}

TEST_F(SingleOpTest, TestAiCpuDependRange) {
  auto fun = [](const Node &node, RunContext &run_context, std::vector<domi::TaskDef> &tasks) -> Status {
    if (node.GetType() != NEG) {
      return SUCCESS;
    }

    tasks.emplace_back(AicpuTaskDefBuilder(node).BuildAicpuTask(3));
    return SUCCESS;
  };
  MockForGenerateTask("AicpuLib", fun);
  MockForGenerateTask("aicpu_ascend_kernel", fun);

  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtKernelLaunchEx).WillRepeatedly(testing::Invoke(MockRtKernelLaunchEx));
  EXPECT_CALL(*runtime_stub, rtMemcpy).WillRepeatedly(testing::Invoke(MockRtMemcpy));

  auto op_desc = CreateOp(NEG);
  AttrUtils::SetInt(op_desc, ATTR_NAME_UNKNOWN_SHAPE_TYPE, static_cast<int64_t>(DEPEND_SHAPE_RANGE));
  AttrUtils::SetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, true);
  EXPECT_EQ(RunDynamicTestCast(op_desc), SUCCESS);
}

TEST_F(SingleOpTest, TestCompileStaticSingleOp) {
  EXPECT_EQ(CompileSingleOpByAtc("st_run_data/json/single_op/add_op.json"), SUCCESS);
  GEFinalize();
  ReInitGe();
  EXPECT_EQ(CompileSingleOpByAtc("st_run_data/json/single_op/matmul_op.json"), SUCCESS);
}

TEST_F(SingleOpTest, TestCompileDynamicSingleOp) {
  EXPECT_EQ(CompileSingleOpByAtc("st_run_data/json/single_op/dynamic_ops.json"), SUCCESS);
}

TEST_F(SingleOpTest, TestInvalidOpType) {
  EXPECT_NE(CompileSingleOpByAtc("st_run_data/json/single_op/exception/invalid_op.json"), SUCCESS);
}

TEST_F(SingleOpTest, TestInvalidAttrType) {
  EXPECT_NE(CompileSingleOpByAtc("st_run_data/json/single_op/exception/invalid_attr_type.json"), SUCCESS);
}

TEST_F(SingleOpTest, TestInvalidAttrName) {
  EXPECT_NE(CompileSingleOpByAtc("st_run_data/json/single_op/exception/invalid_attr_name.json"), SUCCESS);
}

TEST_F(SingleOpTest, TestInvalidAttrValue) {
  EXPECT_NE(CompileSingleOpByAtc("st_run_data/json/single_op/exception/invalid_attr_value.json"), SUCCESS);
}

TEST_F(SingleOpTest, TestInvalidInputDesc) {
  EXPECT_NE(CompileSingleOpByAtc("st_run_data/json/single_op/exception/invalid_input_desc.json"), SUCCESS);
}

TEST_F(SingleOpTest, TestInvalidInputDescDtypeAndFormat) {
  EXPECT_NE(CompileSingleOpByAtc("st_run_data/json/single_op/exception/invalid_input_desc_dtype_and_format.json"), SUCCESS);
}

TEST_F(SingleOpTest, TestInvalidOutputDesc) {
  EXPECT_NE(CompileSingleOpByAtc("st_run_data/json/single_op/exception/invalid_output_desc.json"), SUCCESS);
}

TEST_F(SingleOpTest, TestInvalidOutputDescDtype) {
  EXPECT_NE(CompileSingleOpByAtc("st_run_data/json/single_op/exception/invalid_output_desc_dtype.json"), SUCCESS);
}

TEST_F(SingleOpTest, TestInvalidOutputDescFormat) {
  EXPECT_NE(CompileSingleOpByAtc("st_run_data/json/single_op/exception/invalid_output_desc_format.json"), SUCCESS);
}

TEST_F(SingleOpTest, TestInvalidJsonPath) {
  EXPECT_NE(CompileSingleOpByAtc("st_run_data/json/single_op/exception/_.json"), SUCCESS);
}

TEST_F(SingleOpTest, TestInvalidJsonContent) {
  EXPECT_NE(CompileSingleOpByAtc("st_run_data/json/single_op/exception/broken.json"), SUCCESS);
}

TEST_F(SingleOpTest, TestInvalidInputNum) {
  EXPECT_NE(CompileSingleOpByAtc("st_run_data/json/single_op/exception/invalid_input_num_mismatch.json"), SUCCESS);
}

TEST_F(SingleOpTest, TestInvalidOutputNum) {
  EXPECT_NE(CompileSingleOpByAtc("st_run_data/json/single_op/exception/invalid_output_num_mismatch.json"), SUCCESS);
}

TEST_F(SingleOpTest, TestInvalidUnknownRankShape) {
  EXPECT_NE(CompileSingleOpByAtc("st_run_data/json/single_op/exception/invalid_unknown_rank_shape.json"), SUCCESS);
}

TEST_F(SingleOpTest, TestInvalidUnknownRankWithRange) {
  EXPECT_NE(CompileSingleOpByAtc("st_run_data/json/single_op/exception/invalid_unknown_rank_with_range.json"), SUCCESS);
}

TEST_F(SingleOpTest, TestInvalidUnknownShapeRange) {
  EXPECT_NE(CompileSingleOpByAtc("st_run_data/json/single_op/exception/invalid_unknown_shape_range_mismatch.json"), SUCCESS);
}

TEST_F(SingleOpTest, TestInvalidUnknownShapeNumRange) {
  EXPECT_NE(CompileSingleOpByAtc("st_run_data/json/single_op/exception/invalid_unknown_shape_num_range_mismatch.json"), SUCCESS);
}

TEST_F(SingleOpTest, TestOpManyAttrs) {
  EXPECT_EQ(CompileSingleOpByAtc("st_run_data/json/single_op/op_many_attrs.json"), SUCCESS);
}

TEST_F(SingleOpTest, TestOpUnregstAttrs) {
  EXPECT_NE(CompileSingleOpByAtc("st_run_data/json/single_op/op_unregst_oppath_attrs.json"), SUCCESS);
}

TEST_F(SingleOpTest, TestLongNameSingleOp) {
  EXPECT_EQ(CompileSingleOpByAtc("st_run_data/json/single_op/long_name.json"), SUCCESS);
}

/**
 * 用例描述：so进om后，解除执行时原型依赖，单算子模式在线流程 + 加载符合预期
 *
 * 预置条件：
 * 编译时注册部分算子原型
 *
 * 测试步骤：
 * 1. 构造Conv2D的OpDesc对象
 * 2. 编译、生成ModelBbufferData
 * 3. 加载解析ModelBbufferData
 * 4. 校验结果
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 加载后包含预期属性
 */
TEST_F(SingleOpTest, RecoverIrDefinition_singlop_online) {
  BenchEnv::Init();
  GeShape shape({-2});
  GeTensorDesc tensor_desc(shape);
  auto op_desc = std::make_shared<OpDesc>("conv2d", "Conv2D");
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);

  vector<GeTensor> inputs;
  vector<GeTensor> outputs;
  inputs.emplace_back(GeTensor(tensor_desc));
  outputs.emplace_back(GeTensor(tensor_desc));

  GeGenerator generator;
  generator.Initialize({});
  ModelBufferData model_buffer;
  EXPECT_EQ(generator.BuildSingleOpModel(op_desc, inputs, outputs, ENGINE_AICORE, false, model_buffer), SUCCESS);

  ModelData model_data;
  model_data.model_data = model_buffer.data.get();
  model_data.model_len = model_buffer.length;
  ModelHelper model_helper;
  EXPECT_EQ(model_helper.LoadRootModel(model_data), GRAPH_SUCCESS);
  auto root_model = model_helper.GetGeRootModel();
  EXPECT_NE(root_model, nullptr);
  auto root_graph = root_model->GetRootGraph();
  EXPECT_NE(root_graph, nullptr);
  NodePtr conv2d = nullptr;
  for (const auto &node : root_graph->GetAllNodes()){
    if (node->GetName() == "conv2d") {
      conv2d = node;
      break;
    }
  }
  EXPECT_NE(conv2d, nullptr);
  auto ir_inputs = conv2d->GetOpDesc()->GetIrInputs();
  auto ir_attr_names = conv2d->GetOpDesc()->GetIrAttrNames();
  const std::vector<std::string> target_ir_input = {"x", "filter", "bias", "offset_w"};
  const std::vector<std::string> target_ir_attr_name = {"strides", "pads", "dilations", "groups", "data_format", "offset_x"};
  EXPECT_EQ(ir_inputs.size(), target_ir_input.size());
  for (size_t i = 0U; i < ir_inputs.size(); ++i) {
    EXPECT_EQ(ir_inputs[i].first, target_ir_input[i]);
  }
  EXPECT_EQ(ir_attr_names.size(), target_ir_attr_name.size());
  for (size_t i = 0U; i < ir_attr_names.size(); ++i) {
    EXPECT_EQ(ir_attr_names[i], target_ir_attr_name[i]);
  }
}

/**
 * 用例描述：静态shape单算子执行时开启exception dump，并手动触发回调
 *
 * 测试步骤：
 * 1. 构造SinlgeOp执行
 * 2、开启exception dump
 * 3. 调用回调接口
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 生成dump文件
 */
TEST_F(SingleOpTest, StaticAicore_SaveExceptionDumpInfo_EnableExceptionDump) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  auto op_desc = CreateOp(RELU);
  AttrUtils::SetBool(op_desc, ge::ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 512);
  AttrUtils::SetBool(op_desc, "support_dynamicshape", true);

  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl(RELU);
  funcs->tiling = StubTiling;
  funcs->tiling_parse = StubTilingParse;
  funcs->compile_info_creator = CompileInfoCreator;
  gert::GlobalDumper::GetInstance()->MutableExceptionDumper()->Clear();
  gert::GlobalDumper::GetInstance()->ClearInnerExceptionDumpers();
  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kExceptionDump}));
  setenv("NPU_COLLECT_PATH_EXE", "dump", true);
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
  unsetenv("NPU_COLLECT_PATH_EXE");
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
}

/**
 * 用例描述：静态shape单算子加载并执行，不设置外置allocator
 *
 * 测试步骤：
 * 1. 构造SinlgeOp
 * 2. load & execute
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 有streamresource分配内存日志
 */
TEST_F(SingleOpTest, TestStaticAiCore_LoadAndExecuteOk_WithoutExternalAllocator) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  auto op_desc = CreateOp(RELU);
  AttrUtils::SetBool(op_desc, ge::ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, false);
  AttrUtils::SetStr(op_desc, ATTR_NAME_SGT_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 512);
  AttrUtils::SetBool(op_desc, "support_dynamicshape", true);
  op_desc->SetWorkspace({1});
  op_desc->SetWorkspaceBytes({1256});
  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl(RELU);
  funcs->tiling = StubTiling;
  funcs->tiling_parse = StubTilingParse;
  funcs->compile_info_creator = CompileInfoCreator;
  gert::GertRuntimeStub runtime_stub;
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
  auto logs = runtime_stub.GetSlogStub().GetLogs();
  EXPECT_TRUE(IsLogValid(logs, "func=AllocatorMalloc, size=1536"));
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

/**
 * 用例描述：静态shape单算子加载并执行多次，不设置外置allocator
 *
 * 测试步骤：
 * 1. 构造SinlgeOp
 * 2. load & execute
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 有streamresource分配内存日志
 */
TEST_F(SingleOpTest, TestStaticAiCore_LoadAndExecuteMultipleTimesOk_WithoutExternalAllocator) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  auto op_desc = CreateOp(RELU);
  AttrUtils::SetBool(op_desc, ge::ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, false);
  AttrUtils::SetStr(op_desc, ATTR_NAME_SGT_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 512);
  AttrUtils::SetBool(op_desc, "support_dynamicshape", true);
  op_desc->SetWorkspace({1});
  op_desc->SetWorkspaceBytes({1256});
  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl(RELU);
  funcs->tiling = StubTiling;
  funcs->tiling_parse = StubTilingParse;
  funcs->compile_info_creator = CompileInfoCreator;
  gert::GertRuntimeStub runtime_stub;
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  EXPECT_EQ(RunStaticTestCast2(op_desc), SUCCESS);
  auto logs = runtime_stub.GetSlogStub().GetLogs();
  EXPECT_TRUE(IsLogValid(logs, "func=AllocatorMalloc, size=1536"));
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

/**
* 用例描述：静态shape单算子加载并执行，不设置外置allocator
*
* 测试步骤：
* 1. 构造SinlgeOp
* 2. rt接口打桩，返回失败
* 3. load & execute
*
* 预期结果：
* 1. 执行失败
*/
TEST_F(SingleOpTest, TestStaticAiCore_LoadAndExecuteFailed_WithoutExternalAllocator) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  auto op_desc = CreateOp(RELU);
  AttrUtils::SetBool(op_desc, ge::ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, false);
  AttrUtils::SetStr(op_desc, ATTR_NAME_SGT_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 512);
  AttrUtils::SetBool(op_desc, "support_dynamicshape", true);
  op_desc->SetWorkspace({1});
  op_desc->SetWorkspaceBytes({1256});
  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl(RELU);
  funcs->tiling = StubTiling;
  funcs->tiling_parse = StubTilingParse;
  funcs->compile_info_creator = CompileInfoCreator;

  mmSetEnv("CONSTANT_FOLDING_PASS_2", "mock_fail", 1);
  EXPECT_NE(RunStaticTestCast(op_desc), SUCCESS);
  unsetenv("CONSTANT_FOLDING_PASS_2");
}

/**
 * 用例描述：静态shape单算子加载并执行，设置外置allocator
 *
 * 测试步骤：
 * 1. 构造SinlgeOp
 * 2、设置外置allocator
 * 3. load & execute
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 有外置allocator分配内存日志
 */
TEST_F(SingleOpTest, TestStaticAiCore_LoadAndExecuteOk_WithExternalAllocator) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  auto op_desc = CreateOp(RELU);
  AttrUtils::SetBool(op_desc, ge::ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, false);
  AttrUtils::SetStr(op_desc, ATTR_NAME_SGT_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 512);
  AttrUtils::SetBool(op_desc, "support_dynamicshape", true);
  op_desc->SetWorkspace({1});
  op_desc->SetWorkspaceBytes({1256});
  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl(RELU);
  funcs->tiling = StubTiling;
  funcs->tiling_parse = StubTilingParse;
  funcs->compile_info_creator = CompileInfoCreator;

  auto allocator = MakeUnique<FakeAllocator>();
  gert::GertRuntimeStub runtime_stub;
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  EXPECT_EQ(RunStaticTestCast(op_desc, allocator.get()), SUCCESS);
  auto logs = runtime_stub.GetSlogStub().GetLogs();
  EXPECT_TRUE(IsLogValid(logs, "func=AllocatorMalloc, size=1536"));
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(SingleOpTest, TestStaticAiCore_L0Exception_Ok) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  auto op_desc = CreateOp(RELU);
  AttrUtils::SetBool(op_desc, ge::ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 512);
  AttrUtils::SetBool(op_desc, "support_dynamicshape", true);

  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl(RELU);
  funcs->tiling = StubTiling;
  funcs->tiling_parse = StubTilingParse;
  funcs->compile_info_creator = CompileInfoCreator;

  gert::GlobalDumper::GetInstance()->MutableExceptionDumper()->Clear();
  gert::GlobalDumper::GetInstance()->ClearInnerExceptionDumpers();
  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kLiteExceptionDump}));
  DumpStub::GetInstance().Clear();
  gert::GertRuntimeStub runtime_stub;
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
  EXPECT_EQ(DumpStub::GetInstance().GetUnits().size(), 4);
  EXPECT_EQ(runtime_stub.GetRtsRuntimeStub().GetLiteEceptionArgs()[0],
            reinterpret_cast<uintptr_t>(DumpStub::GetInstance().GetByIndex(3)));
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
}

TEST_F(SingleOpTest, TestStaticMixAiCore_L0Exception_Ok) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForMixAiCore);
      MockForGenerateTask("AIcoreEngine", GenerateTaskForMixAiCore);
  auto op_desc = CreateOp(RELU);
  AttrUtils::SetBool(op_desc, ge::ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 512);
  AttrUtils::SetBool(op_desc, "support_dynamicshape", true);

  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl(RELU);
  funcs->tiling = StubTiling;
  funcs->tiling_parse = StubTilingParse;
  funcs->compile_info_creator = CompileInfoCreator;

  gert::GlobalDumper::GetInstance()->MutableExceptionDumper()->Clear();
  gert::GlobalDumper::GetInstance()->ClearInnerExceptionDumpers();
  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kLiteExceptionDump}));
  DumpStub::GetInstance().Clear();
  gert::GertRuntimeStub runtime_stub;
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
  EXPECT_EQ(DumpStub::GetInstance().GetUnits().size(), 5);
  EXPECT_EQ(runtime_stub.GetRtsRuntimeStub().GetLiteEceptionArgs()[0],
            reinterpret_cast<uintptr_t>(DumpStub::GetInstance().GetByIndex(4)));
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
}

TEST_F(SingleOpTest, TestStaticAiCore_Exception_Ok) {
  MockForGenerateTask("AiCoreLib", GenerateTaskForAiCore);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  auto op_desc = CreateOp(RELU);
  AttrUtils::SetBool(op_desc, ge::ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 512);
  AttrUtils::SetBool(op_desc, "support_dynamicshape", true);
  op_desc->SetWorkspaceBytes({0});

  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl(RELU);
  funcs->tiling = StubTiling;
  funcs->tiling_parse = StubTilingParse;
  funcs->compile_info_creator = CompileInfoCreator;

  gert::GlobalDumper::GetInstance()->MutableExceptionDumper()->Clear();
  gert::GlobalDumper::GetInstance()->ClearInnerExceptionDumpers();
  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kExceptionDump}));
  DumpStub::GetInstance().Clear();
  gert::GertRuntimeStub runtime_stub;
  EXPECT_EQ(RunStaticTestCast(op_desc), SUCCESS);
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
}

TEST_F(SingleOpTest, MemcpyAsyncTask_GetHostArgsAndSize) {
  MemcpyAsyncTask task{};
  uintptr_t args = 0;
  size_t args_size = 0;
  EXPECT_NO_THROW(task.GetHostArgsAndSize(args, args_size));
}

TEST_F(SingleOpTest, test_dynamic_mixl2_launch_failed) {
  auto graph = make_shared<ComputeGraph>("graph");
  auto op_desc = make_shared<OpDesc>("Add", "Add");
  GeTensorDesc desc;
  op_desc->AddInputDesc(desc);
  op_desc->AddOutputDesc(desc);
  auto node = graph->AddNode(op_desc);
  MixL2OpTask task(node);
  ge::DataBuffer data_buffer;
  vector<GeTensorDesc> input_desc;
  vector<DataBuffer> input_buffers = {data_buffer};
  vector<GeTensorDesc> output_desc;
  vector<DataBuffer> output_buffers = {data_buffer};
  task.op_desc_ = op_desc;
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  task.op_ = std::move(std::unique_ptr<Operator>(new (std::nothrow) Operator(op)));
  task.node_ = node;
  rtStream_t stream = nullptr;
  EXPECT_NE(task.LaunchKernel(input_desc, input_buffers, output_desc, output_buffers, stream), SUCCESS);
}

TEST_F(SingleOpTest, dump_op_debug_on_and_set_op_switch) {
  DumpConfig dump_config;
  dump_config.dump_debug = "on";
  dump_config.dump_status = "on";
  auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(0).IsSingleOpNeedDump(), false);
  DumpManager::GetInstance().SetDumpOpSwitch(0, "on");
  EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(0).IsSingleOpNeedDump(), true);

  DumpProperties dp;
  std::map<std::string, std::string> options {{OPTION_EXEC_ENABLE_DUMP_DEBUG, "1"},
                                              {OPTION_EXEC_DUMP_PATH, "/tmp/"},
                                              {OPTION_EXEC_DUMP_DEBUG_MODE, "aicore_overflow"}};
  GetThreadLocalContext().SetGlobalOption(options);
  Status st = dp.InitByOptions();
  EXPECT_EQ(st, SUCCESS);
  std::map<std::string, std::string> option = {};
  uint64_t session_id = 1;
  InnerSession inner_session(session_id, option);
  inner_session.AddDumpProperties(dp);
  EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(1).IsSingleOpNeedDump(), false);
  DumpManager::GetInstance().SetDumpOpSwitch(1, "on");
  EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(1).IsSingleOpNeedDump(), true);
}
}  // namespace ge

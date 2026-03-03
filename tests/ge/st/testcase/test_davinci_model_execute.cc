/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "init_ge.h"
#include "utils/bench_env.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "hcom/hcom_topo_info.h"
#include "base/err_mgr.h"

#include "macro_utils/dt_public_scope.h"
#include "ge/ut/ge/test_tools_task_info.h"
#include "hybrid/hybrid_davinci_model.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "ge/ge_api.h"
#include "framework/executor/ge_executor.h"
#include "framework/common/types.h"
#include "graph/execute/model_executor.h"
#include "runtime/subscriber/global_dumper.h"
#include "graph/utils/attr_utils.h"
#include "graph/ge_context.h"
#include "graph/graph.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/mem_manager.h"
#include "common/profiling/profiling_manager.h"
#include "graph/load/model_manager/model_utils.h"
#include "common/dump/dump_manager.h"
#include "common/dump/dump_utils.h"
#include "graph/load/model_manager/model_manager.h"
#include "opskernel_executor/ops_kernel_executor_manager.h"
#include "exec_runtime/execution_runtime_utils.h"
#include "faker/space_registry_faker.h"
#include "register/op_impl_registry.h"
#include "graph/load/model_manager/task_info/rts/memcpy_async_task_info.h"
#include "faker/aicpu_ext_info_faker.h"
#include "register/op_tiling_info.h"
#include "utils/mock_runtime.h"
#include "common/share_graph.h"
#include "graph/args_format_desc.h"
#include "depends/profiler/src/dump_stub.h"
#include "macro_utils/dt_public_unscope.h"
#include "common/error_tracking/error_tracking.h"
#include "rt_error_codes.h"
#include "framework/ge_runtime_stub/include/common/dump_checker.h"
#include "common/helper/model_parser_base.h"
#include "common/global_variables/diagnose_switch.h"
#include "hcom/hcom_topo_info.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/load/model_manager/task_info/fe/kernel_task_info.h"
#include "graph/load/model_manager/task_info/hccl/hccl_util.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "common/env_path.h"
#include "graph_metadef/depends/checker/tensor_check_utils.h"

#include <hccl/hccl_types.h>
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "utils/mock_ops_kernel_builder.h"
#include "common/tbe_handle_store/tbe_handle_store.h"

using namespace std;
using namespace testing;
using namespace gert;

namespace {
  HcclResult InitializeHeterogeneousRuntime(const std::string &group, void *tilingData, void *ccuTaskGroup) {
    return HCCL_SUCCESS;
  }
}
namespace ge {
namespace{
void MockGenerateTask() {
  auto aicore_func = [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    auto op_desc = node.GetOpDesc();
    op_desc->SetOpKernelLibName("AIcoreEngine");
    ge::AttrUtils::SetStr(op_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_KERNEL_BIN_ID, op_desc->GetName() + "_fake_id");
    const char tbeBin[] = "tbe_bin";
    vector<char> buffer(tbeBin, tbeBin + strlen(tbeBin));
    ge::OpKernelBinPtr tbeKernelPtr = std::make_shared<ge::OpKernelBin>("test_tvm", std::move(buffer));
    op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbeKernelPtr);
    size_t arg_size = 100;
    std::vector<uint8_t> args(arg_size, 0);
    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    auto kernel_info = task_def.mutable_kernel();
    kernel_info->set_args(args.data(), args.size());
    kernel_info->set_args_size(arg_size);
    kernel_info->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    kernel_info->set_kernel_name(node.GetName());
    kernel_info->set_block_dim(1);
    uint16_t args_offset[2] = {0};
    kernel_info->mutable_context()->set_args_offset(args_offset, 2 * sizeof(uint16_t));
    kernel_info->mutable_context()->set_op_index(node.GetOpDesc()->GetId());

    tasks.emplace_back(task_def);
    return SUCCESS;
  };

  auto rts_func = [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    return SUCCESS;
  };

  MockForGenerateTask("AIcoreEngine", aicore_func);
  MockForGenerateTask("AiCoreLib", aicore_func);
  MockForGenerateTask("RTSLib", rts_func);
}
}
void SetGeModelAttrs(const GeModelPtr &ge_model, bool set_sub_mem_infos = false) {
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120));
  ge::AttrUtils::SetInt(ge_model, ge::ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 5120);

  if (set_sub_mem_infos) {
    std::vector<std::vector<int64_t>> sub_memory_infos;
    sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 5120});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 5120, 5120});
    (void)AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);
  }
}
class DavinciModelTest : public testing::Test {
 protected:
  void SetUp() override {
    VarManagerPool::Instance().Destory();
    char runtime2_env[MMPA_MAX_PATH] = {'0'};
    mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
    ReInitGe();
    BenchEnv::Init();
    actual_info_type.clear();
    (void)mmGetEnv("ASCEND_OPP_PATH", old_path_env_, MMPA_MAX_PATH);
    gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry();
    MockGenerateTask();
  }
  void TearDown() override {
    VarManagerPool::Instance().Destory();
    MockRuntime::Reset();
    actual_info_type.clear();
    char runtime2_env[MMPA_MAX_PATH] = {'1'};
    mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
    if (strlen(old_path_env_) > 0U) {
      mmSetEnv("ASCEND_OPP_PATH", old_path_env_, 1);
    }
  }

 private:
  char old_path_env_[MMPA_MAX_PATH] = {'\0'};
};

extern void BuildSampleGraph(ComputeGraphPtr &graph, uint32_t &mem_offset);
extern void BuildHcclSampleGraph(ComputeGraphPtr &graph, uint32_t &mem_offset);
extern void BuildGraphModel(ComputeGraphPtr &graph, GeModelPtr &ge_model, uint32_t mem_offset);
extern void BuildHcclSampleGraphWithQos(ComputeGraphPtr &graph, uint32_t &mem_offset);
namespace {
const char_t *const kEnvName = "ASCEND_OPP_PATH";
const char_t *const kBuiltIn = "built-in";
const char_t *const kVendors = "vendors";
const char_t *const kOpMasterDeviceLib = "/op_impl/ai_core/tbe/op_master_device/lib/";

static void BuildSampleNoTilingGraph(ComputeGraphPtr &graph, uint32_t &mem_offset) {
  BuildSampleGraph(graph, mem_offset);
  {
    const auto &node = graph->FindNode("_arg_0");
    EXPECT_NE(node, nullptr);
    GeTensorDesc input_desc(GeShape({2, 4, 8, 2}), FORMAT_FRACTAL_Z, DT_FLOAT);
    auto tensor1 = node->GetOpDesc()->MutableOutputDesc(0);
    (void)AttrUtils::SetBool(tensor1, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
  }
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

static ge::OpDescPtr CreateOpDesc(string name = "", string type = "") {
  auto op_desc = std::make_shared<ge::OpDesc>(name, type);
  op_desc->SetStreamId(0);
  op_desc->SetId(0);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetInputOffset({100, 200});
  op_desc->SetOutputOffset({100, 200});
  return op_desc;
}

static OpDescPtr CreateOpDesc(const std::string &name,
                              const std::string &type,
                              uint32_t input_num,
                              uint32_t output_num) {
  GeTensorDesc int32_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
  OpDescPtr op_desc = shared_ptr<OpDesc>(new(std::nothrow) OpDesc(name, type));
  if (op_desc == nullptr) {
    return nullptr;
  }
  for (uint32_t i = 0; i < input_num; i++) {
    op_desc->AddInputDesc(int32_tensor);
  }
  for (uint32_t i = 0; i < output_num; i++) {
    op_desc->AddOutputDesc(int32_tensor);
  }
  return op_desc;
}

Status BuildGraphNode(GraphId graph_id, GraphNodePtr &graph_node, GeRootModelPtr &ge_root_model, GeModelPtr &ge_model) {
  uint32_t mem_offset = 0;
  ComputeGraphPtr graph;
  BuildSampleGraph(graph, mem_offset);
  EXPECT_NE(graph, nullptr);

  BuildGraphModel(graph, ge_model, mem_offset);
  EXPECT_NE(ge_model, nullptr);

  ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(graph);
  ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

  graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  return SUCCESS;
}

UINT32 StubTiling(gert::TilingContext *context) {
  context->SetNeedAtomic(false);
  context->SetTilingKey(666U);
  context->SetBlockDim(666U);
  size_t *workspace_size = context->GetWorkspaceSizes(1);
  *workspace_size = 64U;
  return ge::GRAPH_SUCCESS;
}

UINT32 StubTilingParse(gert::KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

struct AddCompileInfo {
  int64_t a;
  int64_t b;
};

void* CompileInfoCreator() {
  return new AddCompileInfo;
}

void ConstructOpMasterDeviceSo(const std::string &opp_path, const size_t built_in_num, const size_t cust_num,
                               const bool &is_cust_same,
                               std::vector<std::pair<ccKernelType, const std::string>> &kernel_type_so_names) {
  for (size_t i = 0UL; i < built_in_num; ++i) {
    std::string inner_op_master = opp_path + kBuiltIn + kOpMasterDeviceLib;
    system(("mkdir -p " + inner_op_master).c_str());
    inner_op_master += std::to_string(i) + "-Ascend-V7.6-libopmaster.so";
    system(("touch " + inner_op_master).c_str());
    system(("echo 'Ascend-V7.6-libopmaster' > " + inner_op_master).c_str());
    kernel_type_so_names.emplace_back(ccKernelType::AI_CPU, inner_op_master);
  }

  std::string vendor_names = "vendor=";
  for (size_t i = 0UL; i < cust_num; ++i) {
    std::string vendor_name = "cust-" + std::to_string(i);
    std::string inner_op_master = opp_path + kVendors + "/" + vendor_name + kOpMasterDeviceLib;
    system(("mkdir -p " + inner_op_master).c_str());
    inner_op_master += "libcust_opmaster.so";
    system(("touch " + inner_op_master).c_str());
    if (is_cust_same) {
      system(("echo 'Ascend-V7.6-libopmaster' > " + inner_op_master).c_str());
    } else {
      system(("echo " + std::to_string(i) + " > " + inner_op_master).c_str());
    }
    vendor_names.append(vendor_name + ",");
    kernel_type_so_names.emplace_back(ccKernelType::CUST_AI_CPU, inner_op_master);
  }

  std::string vendor_config = opp_path + kVendors + "/config.ini";
  system(("touch " + vendor_config).c_str());
  system(("echo " + vendor_names + " > " + vendor_config).c_str());
}

void ConstructTilingSinkGeModel(const std::vector<std::pair<ccKernelType, const std::string>> &kernel_type_so_names,
                                GeModelPtr &ge_model, ComputeGraphPtr &root_graph, bool has_args_format = false) {
  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto ifa_0 = OP_CFG("Ifa_Test");
    auto ifa_1 = OP_CFG("Ifa_Test");
    auto ifa_2 = OP_CFG("Ifa_Test");
    CHAIN(NODE("_arg_0", data_0)
              ->EDGE(0, 0)
              ->NODE("ifa_0", ifa_0)
              ->EDGE(0, 0)
              ->NODE("ifa_1", ifa_1)
              ->EDGE(0, 0)
              ->NODE("ifa_2", ifa_2)
              ->EDGE(0, 0)
              ->NODE("output", NETOUTPUT));
  };

  root_graph = ToComputeGraph(g1);
  EXPECT_NE(root_graph, nullptr);

  // Data
  GeTensorDesc tensor_desc(GeShape({4}), FORMAT_ND, DT_FLOAT);
  int64_t offset = 64L;
  const auto &data = root_graph->FindNode("_arg_0" );
  EXPECT_NE(data, nullptr);
  data->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
  data->GetOpDesc()->SetOutputOffset({offset});
  data->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");

  // NetOutput
  const auto &out_node = root_graph->FindNode("output");
  EXPECT_NE(out_node, nullptr);
  out_node->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
  out_node->GetOpDesc()->SetInputOffset({offset});
  out_node->GetOpDesc()->SetSrcName({"ifa_2"});
  out_node->GetOpDesc()->SetSrcIndex({0});
  out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("tbeKernel", std::move(test_bin));
  size_t aicpu_args_size = 128UL;
  const std::vector<uint8_t> args_info(aicpu_args_size, 0);

  if (has_args_format) {
    auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto funcs = space_registry->CreateOrGetOpImpl("Ifa_Test");;
    funcs->tiling = StubTiling;
    funcs->tiling_parse = StubTilingParse;
    funcs->compile_info_creator = CompileInfoCreator;
    funcs->compile_info_deleter = nullptr;
    EXPECT_EQ(funcs->SetTilingInputDataDependency(0), GRAPH_SUCCESS);
  }

  for (auto i = 0; i <= 2; ++i) {
    const auto &ifa = root_graph->FindNode("ifa_" + std::to_string(i));
    EXPECT_NE(data, nullptr);
    ifa->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
    ifa->GetOpDesc()->SetOutputOffset({offset + i * offset});
    (void)AttrUtils::SetStr(ifa->GetOpDesc(), ATTR_NAME_TBE_KERNEL_NAME, test_kernel->GetName());
    ifa->GetOpDesc()->SetExtAttr(test_kernel->GetName(), test_kernel);
    ifa->GetOpDesc()->AppendIrInput("query", IrInputType::kIrInputRequired);


    // aicpu kernel
    auto &aicpu_task = *model_task_def->add_task();
    aicpu_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL));
    auto aicpu_kernel = aicpu_task.mutable_kernel();
    aicpu_kernel->set_so_name(kernel_type_so_names[i].second);
    domi::KernelContext &aicpu_context = *aicpu_kernel->mutable_context();
    aicpu_context.set_kernel_type(static_cast<int32_t>(kernel_type_so_names[i].first));
    aicpu_context.set_op_id(ifa->GetOpDesc()->GetId());
    aicpu_context.set_op_index(ifa->GetOpDesc()->GetId());
    aicpu_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
    aicpu_kernel->set_args_size(aicpu_args_size);
    if (has_args_format) {
      aicpu_context.set_args_format("{tiling_context}{*op_type}{tiling_context.tiling_key}{tiling_context.block_dim}");
    }
  }

  ge_model = MakeShared<GeModel>();
  auto &kernel_store = ge_model->GetTBEKernelStore();
  kernel_store.AddTBEKernel(test_kernel);

  ge_model->SetGraph(root_graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 2));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
}
}  // namespace

TEST_F(DavinciModelTest, hccl_dump) {
  uint32_t mem_offset = 0U;
  ComputeGraphPtr graph;
  BuildHcclSampleGraph(graph, mem_offset);
  EXPECT_NE(graph, nullptr);

  GeModelPtr ge_model;
  BuildGraphModel(graph, ge_model, mem_offset);
  EXPECT_NE(ge_model, nullptr);

  std::vector<gert::Tensor> inputs(4);
  TensorCheckUtils::ConstructGertTensor(inputs[0], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[1], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[2], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[3], {1}, DT_INT64, FORMAT_ND);

  std::vector<uint32_t> model_ids;

  DumpProperties dump_properties;
  dump_properties.SetDumpMode("all");
  dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
  DumpManager::GetInstance().RemoveDumpProperties(graph->GetSessionID());
  DumpManager::GetInstance().AddDumpProperties(graph->GetSessionID(), dump_properties);

  setenv(kEnvGeuseStaticMemory.c_str(), "1", 1);
  {
    // Test LoadModelOnline: RunAsyncListener
    const auto ge_root_model = MakeShared<GeRootModel>();
    ge_root_model->Initialize(graph);
    const auto graph_node = MakeShared<GraphNode>(graph->GetGraphID());
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->IncreaseLoadCount();

    // Callback for execute.
    std::mutex run_mutex;
    std::condition_variable model_run_cv;
    Status run_status = FAILED;
    const auto callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
      std::unique_lock<std::mutex> lock(run_mutex);
      run_status = status;
      model_run_cv.notify_one();
    };

    // RunArgsV2 of Graph.
    GEThreadLocalContext context;
    context.SetGraphOption(
        {{OPTION_EXEC_DYNAMIC_EXECUTE_MODE, "lazy_recompile"}, {OPTION_EXEC_ENABLE_COPY_OUTPUT_ADDR, "1"}});
    error_message::ErrorManagerContext error_context;
    graph_node->Lock();
    std::shared_ptr<RunArgs> arg;
    arg = std::make_shared<RunArgs>();
    ASSERT_TRUE(arg != nullptr);
    arg->graph_node = graph_node;
    arg->graph_id = graph->GetGraphID();
    arg->session_id = graph->GetSessionID();
    arg->error_context = error_context;
    arg->input_tensor = std::move(inputs);
    arg->context = context;
    arg->callback = callback;
    // Load and execute.
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.PushRunArgs(arg), SUCCESS);

    // Wait for execute.
    std::unique_lock<std::mutex> lock(run_mutex);
    EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
    EXPECT_EQ(run_status, SUCCESS);
    model_ids.emplace_back(ge_root_model->GetModelId());

    // Unload model of graph
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph->GetGraphID()), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  DumpManager::GetInstance().RemoveDumpProperties(graph->GetSessionID());
  unsetenv(kEnvGeuseStaticMemory.c_str());
}

TEST_F(DavinciModelTest, hccl_dump_on_watcher_model) {
  uint32_t mem_offset = 0U;
  ComputeGraphPtr graph;
  BuildHcclSampleGraph(graph, mem_offset);
  EXPECT_NE(graph, nullptr);

  GeModelPtr ge_model;
  BuildGraphModel(graph, ge_model, mem_offset);
  EXPECT_NE(ge_model, nullptr);

  std::vector<gert::Tensor> inputs(4);
  TensorCheckUtils::ConstructGertTensor(inputs[0], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[1], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[2], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[3], {1}, DT_INT64, FORMAT_ND);

  std::vector<uint32_t> model_ids;

  const uint64_t session_id = graph->GetSessionID();
  DumpProperties dump_properties;
  dump_properties.SetDumpMode("output");
  dump_properties.AddPropertyValue(DUMP_LAYER_OP_MODEL, {"HcomAllreduce"});
  dump_properties.AddPropertyValue(DUMP_WATCHER_MODEL, {"cond/add","add_n"});
  DumpManager::GetInstance().RemoveDumpProperties(session_id);
  DumpManager::GetInstance().AddDumpProperties(session_id, dump_properties);

  setenv(kEnvGeuseStaticMemory.c_str(), "1", 1);
  {
    // Test LoadModelOnline: RunAsyncListener
    const auto ge_root_model = MakeShared<GeRootModel>();
    ge_root_model->Initialize(graph);
    const auto graph_node = MakeShared<GraphNode>(graph->GetGraphID());
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->IncreaseLoadCount();

    // Callback for execute.
    std::mutex run_mutex;
    std::condition_variable model_run_cv;
    Status run_status = FAILED;
    const auto callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
      std::unique_lock<std::mutex> lock(run_mutex);
      run_status = status;
      model_run_cv.notify_one();
    };

    // RunArgsV2 of Graph.
    GEThreadLocalContext context;
    context.SetGraphOption(
        {{OPTION_EXEC_DYNAMIC_EXECUTE_MODE, "lazy_recompile"}, {OPTION_EXEC_ENABLE_COPY_OUTPUT_ADDR, "1"}});
    error_message::ErrorManagerContext error_context;
    graph_node->Lock();
    std::shared_ptr<RunArgs> arg;
    arg = std::make_shared<RunArgs>();
    ASSERT_TRUE(arg != nullptr);
    arg->graph_node = graph_node;
    arg->graph_id = graph->GetGraphID();
    arg->session_id = graph->GetSessionID();
    arg->error_context = error_context;
    arg->input_tensor = std::move(inputs);
    arg->context = context;
    arg->callback = callback;
    // Load and execute.
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.PushRunArgs(arg), SUCCESS);

    // Wait for execute.
    std::unique_lock<std::mutex> lock(run_mutex);
    EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
    EXPECT_EQ(run_status, SUCCESS);
    model_ids.emplace_back(ge_root_model->GetModelId());

    // Unload model of graph
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph->GetGraphID()), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  DumpManager::GetInstance().RemoveDumpProperties(session_id);
  unsetenv(kEnvGeuseStaticMemory.c_str());
}

TEST_F(DavinciModelTest, sdma_dump) {
  uint32_t mem_offset = 0U;
  ComputeGraphPtr graph;
  BuildHcclSampleGraph(graph, mem_offset);
  EXPECT_NE(graph, nullptr);

  GeModelPtr ge_model;
  BuildGraphModel(graph, ge_model, mem_offset);
  EXPECT_NE(ge_model, nullptr);

  std::vector<gert::Tensor> inputs(4);
  TensorCheckUtils::ConstructGertTensor(inputs[0], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[1], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[2], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[3], {1}, DT_INT64, FORMAT_ND);

  std::vector<uint32_t> model_ids;

  DumpProperties dump_properties;
  dump_properties.is_train_op_debug_ = true;
  DumpManager::GetInstance().AddDumpProperties(graph->GetSessionID(), dump_properties);

  setenv(kEnvGeuseStaticMemory.c_str(), "1", 1);
  {
    // Test LoadModelOnline: RunAsyncListener
    const auto ge_root_model = MakeShared<GeRootModel>();
    ge_root_model->Initialize(graph);
    const auto graph_node = MakeShared<GraphNode>(graph->GetGraphID());
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->IncreaseLoadCount();

    // Callback for execute.
    std::mutex run_mutex;
    std::condition_variable model_run_cv;
    Status run_status = FAILED;
    const auto callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
      std::unique_lock<std::mutex> lock(run_mutex);
      run_status = status;
      model_run_cv.notify_one();
    };

    // RunArgsV2 of Graph.
    GEThreadLocalContext context;
    context.SetGraphOption(
        {{OPTION_EXEC_DYNAMIC_EXECUTE_MODE, "lazy_recompile"}, {OPTION_EXEC_ENABLE_COPY_OUTPUT_ADDR, "1"}});
    error_message::ErrorManagerContext error_context;
    graph_node->Lock();
    std::shared_ptr<RunArgs> arg;
    arg = std::make_shared<RunArgs>();
    ASSERT_TRUE(arg != nullptr);
    arg->graph_node = graph_node;
    arg->graph_id = graph->GetGraphID();
    arg->session_id = graph->GetSessionID();
    arg->error_context = error_context;
    arg->input_tensor = std::move(inputs);
    arg->context = context;
    arg->callback = callback;
    // Load and execute.
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.Initialize({{"ge.variableMemoryMaxSize", "12800"}}, graph->GetSessionID()), SUCCESS);
    EXPECT_EQ(model_executor.PushRunArgs(arg), SUCCESS);

    // Wait for execute.
    std::unique_lock<std::mutex> lock(run_mutex);
    EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
    EXPECT_EQ(run_status, SUCCESS);
    model_ids.emplace_back(ge_root_model->GetModelId());

    // Unload model of graph
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph->GetGraphID()), SUCCESS);
    RunAsyncCallbackV2 callback2 = nullptr;
    model_executor.ReturnError(callback2, FAILED, "test return error");
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  DumpManager::GetInstance().RemoveDumpProperties(graph->GetSessionID());
  unsetenv(kEnvGeuseStaticMemory.c_str());
}

TEST_F(DavinciModelTest, sample_davinci_model_static_memory_no_tiling) {
  uint32_t mem_offset = 0U;
  ComputeGraphPtr graph;
  BuildSampleNoTilingGraph(graph, mem_offset);
  EXPECT_NE(graph, nullptr);

  GeModelPtr ge_model;
  BuildGraphModel(graph, ge_model, mem_offset);
  EXPECT_NE(ge_model, nullptr);

  std::vector<gert::Tensor> inputs(4);
  TensorCheckUtils::ConstructGertTensor(inputs[0], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[1], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[2], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[3], {1}, DT_INT64, FORMAT_ND);
  std::vector<uint32_t> model_ids;
  setenv(kEnvGeuseStaticMemory.c_str(), "4", 1);
  {
    // Test LoadModelOnline: RunAsyncListener
    const auto ge_root_model = MakeShared<GeRootModel>();
    ge_root_model->Initialize(graph);
    const auto graph_node = MakeShared<GraphNode>(graph->GetGraphID());
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetAsync(false);
    graph_node->IncreaseLoadCount();

    // Callback for execute.
    std::mutex run_mutex;
    std::condition_variable model_run_cv;
    Status run_status = FAILED;
    std::vector<gert::Tensor> run_outputs;
    const auto callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
      std::unique_lock<std::mutex> lock(run_mutex);
      run_status = status;
      run_outputs.swap(outputs);
      model_run_cv.notify_one();
    };

    // RunArgsV2 of Graph.
    GEThreadLocalContext context;
    context.SetGraphOption(
        {{OPTION_EXEC_DYNAMIC_EXECUTE_MODE, "lazy_recompile"}, {OPTION_EXEC_ENABLE_COPY_OUTPUT_ADDR, "1"}});
    error_message::ErrorManagerContext error_context;
    graph_node->Lock();
    std::shared_ptr<RunArgs> arg;
    arg = std::make_shared<RunArgs>();
    ASSERT_TRUE(arg != nullptr);
    arg->graph_node = graph_node;
    arg->graph_id = graph->GetGraphID();
    arg->session_id = graph->GetSessionID();
    arg->error_context = error_context;
    arg->input_tensor = std::move(inputs);
    arg->context = context;
    arg->callback = callback;
    // Load and execute.
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.Initialize({{"ge.variableMemoryMaxSize", "12800"}}, graph->GetSessionID()), SUCCESS);
    EXPECT_EQ(model_executor.PushRunArgs(arg), SUCCESS);

    // Wait for execute.
    std::unique_lock<std::mutex> lock(run_mutex);
    EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
    EXPECT_EQ(run_status, SUCCESS);
    EXPECT_EQ(run_outputs.size(), 1U);
    model_ids.emplace_back(ge_root_model->GetModelId());

    // Unload model of graph
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph->GetGraphID()), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  unsetenv(kEnvGeuseStaticMemory.c_str());
}

static void BuildAddGraph(ComputeGraphPtr &graph, const std::string &file_constant_name, const bool is_unknown) {
  std::vector<int64_t> shape = {2, 2, 2, 2};
  auto file_const_op = OP_CFG(FILECONSTANT).Attr("shape", shape).Attr("dtype", DT_FLOAT).Attr("file_id", "vector_search_bucker_value_bin");

  int64_t dims_size = 1;
  vector<int64_t> data_vec = {2, 2, 2, 2};
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<float> data_value_vec(dims_size, 1);
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_FLOAT);
  GeTensorPtr data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                  data_value_vec.size() * sizeof(float));
  std::cout << "davinci_model_execute_with_file_constant" << data_value_vec.size() << std::endl;
  auto const_op = OP_CFG(CONSTANT).Weight(data_tensor);
  auto add = OP_CFG(ADD).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  auto output = OP_CFG(NETOUTPUT);
  DEF_GRAPH(g1) {
    CHAIN(NODE(file_constant_name, file_const_op)->EDGE(0, 0)->NODE("add", add));
    CHAIN(NODE("const_op", const_op)->EDGE(0, 1)->NODE("add", add));
    CHAIN(NODE("add", add)->EDGE(0, 0)->NODE(NODE_NAME_NET_OUTPUT, output));
  };

  graph = ToComputeGraph(g1);
  graph->SetGraphUnknownFlag(is_unknown);
  uint32_t mem_offset = 0;
  SetUnknownOpKernel(graph, mem_offset, true);
}

void BuildAddGraphModel(ComputeGraphPtr &graph, GeModelPtr &ge_model) {
  TBEKernelStore tbe_kernel_store;
  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  EXPECT_TRUE(AttrUtils::SetInt(graph, "globalworkspace_type", 1));
  EXPECT_TRUE(AttrUtils::SetInt(graph, "globalworkspace_size", 1));
  InitKernelTaskDef_TE(graph, *model_task_def, "add", tbe_kernel_store);

  InitEventTaskDef(graph, *model_task_def);
  InitFusionTaskDef(graph, *model_task_def);
  InitEndGraphDef(graph, *model_task_def, NODE_NAME_NET_OUTPUT);

  InitProfilerTaskDef(graph, *model_task_def);

  const size_t logic_var_base = VarManager::Instance(graph->GetSessionID())->GetVarMemLogicBase();
  ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 3));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_TASK_GEN_VAR_ADDR, logic_var_base));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, logic_var_base));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120));

  std::vector<uint64_t> weights_value(100, 1024);
  size_t weight_size = weights_value.size() * sizeof(uint64_t);
  ge_model->SetWeight(Buffer::CopyFrom((uint8_t *)weights_value.data(), weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size));

  EXPECT_TRUE(tbe_kernel_store.Build());
  ge_model->SetTBEKernelStore(tbe_kernel_store);
}

TEST_F(DavinciModelTest, davinci_model_execute_with_file_constant) {
  ComputeGraphPtr graph;
  BuildAddGraph(graph, "file_constant_1", false);
  {
    size_t file_const_size = 64;
    float *float_buf = (float *)malloc(file_const_size);
    if (float_buf == nullptr) {
      return;
    }
    std::ofstream out1("test_copy_one_weight.bin", std::ios::binary);
    if (!out1.is_open()) {
      free(float_buf);
      return;
    }
    out1.write((char *)float_buf, file_const_size);
    out1.close();
    free(float_buf);
  }
  GeModelPtr ge_model;
  BuildAddGraphModel(graph, ge_model);
  EXPECT_NE(ge_model, nullptr);

  {
    // Test LoadModelOnline
    auto runtime_stub = MockForKernelLaunchExFailed();
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    ge_root_model->Initialize(graph);
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);

    std::map<std::string, std::string> config;
    std::string a = "ge.exec.value_bins";
    std::string b = "{\"value_bins\":[{\"value_bin_id\":\"vector_search_bucker_value_bin\", \"value_bin_file\":\"./test_copy_one_weight.bin\"}]}";
    config.insert(std::pair<std::string, std::string>(a,b));
    GetThreadLocalContext().SetGraphOption(config);

    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
    ModelManager::GetInstance().sess_id_to_device_ids_[0] = {0};
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.Initialize({{"ge.variableMemoryMaxSize", "12800"}}, 0), SUCCESS);
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
    runtime_stub->Reset();
  }
  (void)remove("test_copy_one_weight.bin");
}

static void BuildAddGraph_ConstPlaceHolder(ComputeGraphPtr &graph) {
  vector<int64_t > shape({1, 2, 3});
  DataType data_type = DT_FLOAT;
  auto const_place_holder = OP_CFG(CONSTPLACEHOLDER).Attr("origin_shape", shape).Attr("storage_shape", shape).\
                            Attr("dtype", data_type).Attr("size", 24L).Attr("placement", 1L).Attr("addr", 20000L);

  int64_t dims_size = 1;
  vector<int64_t> data_vec = {1, 2, 3};
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<float> data_value_vec(dims_size, 1);
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_FLOAT);
  GeTensorPtr data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                       data_value_vec.size() * sizeof(float));

  auto const_op = OP_CFG(CONSTANT).Weight(data_tensor);
  auto add = OP_CFG(ADD).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  auto output = OP_CFG(NETOUTPUT);
  DEF_GRAPH(g1) {
                    CHAIN(NODE("constplaceholdertest", const_place_holder)->EDGE(0, 0)->NODE("add", add));
                    CHAIN(NODE("const_op", const_op)->EDGE(0, 1)->NODE("add", add));
                    CHAIN(NODE("add", add)->EDGE(0, 0)->NODE(NODE_NAME_NET_OUTPUT, output));
                };

  graph = ToComputeGraph(g1);
  uint32_t mem_offset = 0;
  SetUnknownOpKernel(graph, mem_offset, true);
}

TEST_F(DavinciModelTest, davinci_model_execute_with_const_placeholder) {
    ComputeGraphPtr graph;
    BuildAddGraph_ConstPlaceHolder(graph);
    GeModelPtr ge_model;
    BuildAddGraphModel(graph, ge_model);
    EXPECT_NE(ge_model, nullptr);
    {
        // Test LoadModelOnline
        auto runtime_stub = MockForKernelLaunchExFailed();
        GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
        ge_root_model->Initialize(graph);
        ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

        GraphId graph_id = 1001;
        GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
        graph_node->SetGeRootModel(ge_root_model);
        graph_node->SetLoadFlag(true);
        graph_node->SetAsync(true);

        ModelExecutor model_executor;
        EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
        ModelManager::GetInstance().sess_id_to_device_ids_[0] = {0};
        model_executor.StartRunThread();
        EXPECT_EQ(model_executor.Initialize({{"ge.variableMemoryMaxSize", "12800"}}, 0), SUCCESS);
        EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
        EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
        EXPECT_EQ(model_executor.Finalize(), SUCCESS);
        runtime_stub->Reset();
    }
}

TEST_F(DavinciModelTest, davinci_model_execute_with_file_constant_failed) {
  ComputeGraphPtr graph;
  BuildAddGraph(graph, "file_constant_1", false);
  GeModelPtr ge_model;
  BuildAddGraphModel(graph, ge_model);
  EXPECT_NE(ge_model, nullptr);

  {
    auto runtime_stub = MockForKernelLaunchExFailed();
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(graph);
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);

    std::map<std::string, std::string> config;
    config.insert(std::pair<std::string, std::string>("ge.exec.value_bins", "{\"value_bins\":"));
    GetThreadLocalContext().SetGraphOption(config);

    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
    ModelManager::GetInstance().sess_id_to_device_ids_[0] = {0};
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.Initialize({{"ge.variableMemoryMaxSize", "12800"}}, 0), SUCCESS);
    EXPECT_NE(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
    runtime_stub->Reset();
  }
}

TEST_F(DavinciModelTest, command_profiling_get_hybrid_model) {
  uint32_t model_id = 999;
  Command cmd;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  ge_root_model->Initialize(graph);
  auto hybrid_model_ptr = ge::hybrid::HybridDavinciModel::Create(ge_root_model);
  auto shared_model = std::shared_ptr<hybrid::HybridDavinciModel>(hybrid_model_ptr.release());
  shared_model->SetDeviceId(0);
  cmd.cmd_params.push_back("modelId");
  cmd.cmd_params.push_back(to_string(model_id));
  EXPECT_EQ(ModelManager::GetInstance().HandleProfModelSubscribeCommand(cmd), FAILED);
  ModelManager::GetInstance().InsertModel(model_id, shared_model);
  EXPECT_EQ(ModelManager::GetInstance().HandleProfModelSubscribeCommand(cmd), SUCCESS);
  EXPECT_EQ(ModelManager::GetInstance().HandleProfModelUnsubscribeCommand(cmd), SUCCESS);
  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(model_id), SUCCESS);
}

TEST_F(DavinciModelTest, unknown_shape_execute_with_file_constant_host) {
  ComputeGraphPtr graph;
  BuildAddGraph(graph, "file_constant_2", true);
  {
    size_t file_const_size = 64;
    float *float_buf = (float *)malloc(file_const_size);
    if (float_buf == nullptr) {
      return;
    }
    std::ofstream out1("test_copy_one_weight.bin", std::ios::binary);
    if (!out1.is_open()) {
      free(float_buf);
      return;
    }
    out1.write((char *)float_buf, file_const_size);
    out1.close();
    free(float_buf);
  }
  GeModelPtr ge_model;
  BuildAddGraphModel(graph, ge_model);
  EXPECT_NE(ge_model, nullptr);

  // Test LoadModelOnline
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(graph);
  ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);

  std::map<std::string, std::string> config;
  std::string a = "ge.exec.value_bins";
  std::string b =
      "{\"value_bins\":[{\"value_bin_id\":\"vector_search_bucker_value_bin\", "
      "\"value_bin_file\":\"./test_copy_one_weight.bin\"}]}";
  config.insert(std::pair<std::string, std::string>(a, b));
  config["ge.exec.placement"] = "HOST";
  GetThreadLocalContext().SetGraphOption(config);

  auto hybrid_model = hybrid::HybridDavinciModel::Create(ge_root_model);
  ASSERT_NE(hybrid_model, nullptr);
  ASSERT_EQ(hybrid_model->Init(), SUCCESS);

  (void)remove("test_copy_one_weight.bin");
}

TEST_F(DavinciModelTest, unknown_shape_execute_with_file_constant) {
  ComputeGraphPtr graph;
  BuildAddGraph(graph, "file_constant_33333", true);
  {
    size_t file_const_size = 64;
    float *float_buf = (float *)malloc(file_const_size);
    if (float_buf == nullptr) {
      return;
    }
    std::ofstream out1("test_copy_one_weight.bin", std::ios::binary);
    if (!out1.is_open()) {
      free(float_buf);
      return;
    }
    out1.write((char *)float_buf, file_const_size);
    out1.close();
    free(float_buf);
  }
  GeModelPtr ge_model;
  BuildAddGraphModel(graph, ge_model);
  EXPECT_NE(ge_model, nullptr);

  // Test LoadModelOnline
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(graph);
  ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);

  map<std::string, std::string> options{{"ge.variableMemoryMaxSize", "12800"}};
  GetThreadLocalContext().SetSessionOption(options);
  std::map<std::string, std::string> config;
  std::string a = "ge.exec.value_bins";
  std::string b =
      "{\"value_bins\":[{\"value_bin_id\":\"vector_search_bucker_value_bin\", "
      "\"value_bin_file\":\"./test_copy_one_weight.bin\"}]}";
  config.insert(std::pair<std::string, std::string>(a, b));
  GetThreadLocalContext().SetGraphOption(config);
  auto hybrid_model = hybrid::HybridDavinciModel::Create(ge_root_model);
  ASSERT_NE(hybrid_model, nullptr);
  ASSERT_EQ(hybrid_model->Init(), SUCCESS);
  (void)VarManager::Instance(graph->GetSessionID())->FreeVarMemory();
  (void)remove("test_copy_one_weight.bin");
}

TEST_F(DavinciModelTest, davinci_model_execute_no_tiling_with_sub_mem) {
  const auto SetUnknownOpKernelForNoTiling = [](const ComputeGraph::Vistor<NodePtr> &all_nodes) {
    GeTensorDesc tensor0(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor0, 64);
    AttrUtils::SetBool(tensor0, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor0, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 0);
    std::vector<int64_t> max_shape_list = {1, 10, 224, 224};
    AttrUtils::SetListInt(tensor0, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    GeTensorDesc tensor1(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor1, 64);
    AttrUtils::SetBool(tensor1, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor1, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1024);
    AttrUtils::SetListInt(tensor1, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      if (op_desc->GetType() == DATA) {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateOutputDesc(0, tensor0);
        op_desc->SetOutputOffset({2048});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else if (op_desc->GetType() == ADD) {
        op_desc->SetOpKernelLibName("AIcoreEngine");
        op_desc->UpdateInputDesc(0, tensor0);
        op_desc->UpdateOutputDesc(0, tensor1);
        op_desc->SetInputOffset({0, 2048});
        op_desc->SetOutputOffset({2112});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateInputDesc(0, tensor1);
        op_desc->SetInputOffset({2112});
        op_desc->SetSrcName( { "add" } );
        op_desc->SetSrcIndex({ 0 });
      }
    }
  };

  auto add = OP_CFG(ADD).Attr(ATTR_NAME_OP_NO_TILING, true).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  auto data = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto output = OP_CFG(NETOUTPUT).Attr(ATTR_NAME_OP_NO_TILING, true);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", data)->EDGE(0, 0)->NODE("add", add));
    CHAIN(NODE("add", add)->EDGE(0, 0)->NODE("output", output));
  };

  auto graph = ToComputeGraph(g1);
  SetUnknownOpKernelForNoTiling(graph->GetDirectNode());
  EXPECT_NE(graph, nullptr);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  TBEKernelStore tbe_kernel_store;
  InitKernelTaskDef_TE(graph, *model_task_def, "add", tbe_kernel_store);
  InitEventTaskDef(graph, *model_task_def);
  InitFusionTaskDef(graph, *model_task_def);
  InitEndGraphDef(graph, *model_task_def, "output");
  InitProfilerTaskDef(graph, *model_task_def);


  DumpProperties dump_properties;
  dump_properties.SetDumpMode("all");
  dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
  DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  SetGeModelAttrs(ge_model, true);
  EXPECT_NE(ge_model, nullptr);

  {
    // Test LoadModelOnline
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(graph);
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);

    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }

  // Serialization GeModel to memory.
  ModelHelper model_helper;
  model_helper.SetSaveMode(false);  // Save to buffer.
  ModelBufferData model_buffer;
  EXPECT_TRUE(tbe_kernel_store.Build());
  ge_model->SetTBEKernelStore(tbe_kernel_store);
  EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "file_name_prefix", model_buffer), SUCCESS);
  const ModelData model_data{model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", ""};

  // Test LoadModelWithQ
  {
    const std::vector<uint32_t> input_queue_ids{ 1001U };
    const std::vector<uint32_t> output_queue_ids{ 1002U };
    ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
    uint32_t model_id = 0;
    GeExecutor ge_executor;
    EXPECT_EQ(ge_executor.LoadModelWithQ(model_id, model_data, input_queue_ids, output_queue_ids), SUCCESS);
    EXPECT_EQ(ge_executor.UnloadModel(model_id), SUCCESS);
    ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  }

  // Test LoadModelWithQueueParam
  {
    const std::vector<uint32_t> input_queue_ids{ 1001U };
    QueueAttrs in_queue_0 = {.queue_id = 1001U, .device_type = NPU, .device_id = 0};
    const std::vector<uint32_t> output_queue_ids{ 1002U };
    QueueAttrs out_queue_0 = {.queue_id = 1002U, .device_type = NPU, .device_id = 0};
    uint32_t model_id = 0;
    ModelQueueParam model_queue_param{};
    model_queue_param.input_queues = input_queue_ids;
    model_queue_param.output_queues = output_queue_ids;
    model_queue_param.input_queues_attrs = {in_queue_0};
    model_queue_param.output_queues_attrs = {out_queue_0};
    ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
    GeExecutor ge_executor;
    ge_executor.is_inited_ = false;
    EXPECT_EQ(ge_executor.LoadModelWithQueueParam(model_id, model_data, model_queue_param), ACL_ERROR_GE_EXEC_NOT_INIT);
    ge_executor.is_inited_ = true;
    EXPECT_EQ(ge_executor.LoadModelWithQueueParam(model_id, model_data, model_queue_param), SUCCESS);
    EXPECT_EQ(ge_executor.UnloadModel(model_id), SUCCESS);
    ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  }

  // Test LoadModelWithQ dummy
  {
    const std::vector<uint32_t> input_queue_ids{ 1001U };
    const std::vector<uint32_t> output_queue_ids{ UINT32_MAX };
    ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
    uint32_t model_id = 0;
    GeExecutor ge_executor;
    EXPECT_EQ(ge_executor.LoadModelWithQ(model_id, model_data, input_queue_ids, output_queue_ids), SUCCESS);
    EXPECT_EQ(ge_executor.UnloadModel(model_id), SUCCESS);
    ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  }

  // Test LoadModelWithQ batch dequeue
  {
    auto data_node = graph->FindFirstNodeMatchType(DATA);
    EXPECT_NE(data_node, nullptr);
    NamedAttrs align_attr;
    AttrUtils::SetInt(align_attr, ATTR_NAME_INPUTS_ALIGN_OFFSET, 0);
    AttrUtils::SetInt(align_attr, ATTR_NAME_INPUTS_ALIGN_INTERVAL, 1);
    AttrUtils::SetNamedAttrs(data_node->GetOpDesc(), ATTR_NAME_INPUTS_ALIGN_ATTR, align_attr);

    ModelHelper model_helper2;
    model_helper2.SetSaveMode(false);  // Save to buffer.
    ModelBufferData model_buffer2;
    EXPECT_EQ(model_helper2.SaveToOmModel(ge_model, "file_name_prefix", model_buffer2), SUCCESS);
    const ModelData model_data2{model_buffer2.data.get(), static_cast<uint32_t>(model_buffer2.length), 0, "", ""};

    const std::vector<uint32_t> input_queue_ids{ 1001U };
    const std::vector<uint32_t> output_queue_ids{ 1002U };
    ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
    uint32_t model_id = 0;
    GeExecutor ge_executor;
    EXPECT_EQ(ge_executor.LoadModelWithQ(model_id, model_data2, input_queue_ids, output_queue_ids), SUCCESS);
    EXPECT_EQ(ge_executor.UnloadModel(model_id), SUCCESS);
    ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  }

  // Test LoadModelWithQueueParam
  {
    ModelHelper model_helper3;
    EXPECT_EQ(model_helper3.LoadModel(model_data), SUCCESS);
    const std::vector<uint32_t> input_queue_ids{ 1001U };
    QueueAttrs in_queue_0 = {.queue_id = 1001U, .device_type = NPU, .device_id = 0};
    const std::vector<uint32_t> output_queue_ids{ 1002U };
    QueueAttrs out_queue_0 = {.queue_id = 1002U, .device_type = NPU, .device_id = 0};
    ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
    uint32_t model_id = 0;
    ModelQueueParam model_queue_param{};
    model_queue_param.group_total_count = 1;
    model_queue_param.group_index = 0;
    model_queue_param.input_queues = input_queue_ids;
    model_queue_param.output_queues = output_queue_ids;
    model_queue_param.input_queues_attrs = {in_queue_0};
    model_queue_param.output_queues_attrs = {out_queue_0};
    model_queue_param.is_dynamic_sched = true;
    model_queue_param.need_report_status = true;
    DumpProperties dump_properties;
    dump_properties.enable_dump_ = "1";
    dump_properties.dump_step_ = "0|2-4|6";
    DumpManager::GetInstance().RemoveDumpProperties(0);
    DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
    EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(0).IsDumpOpen(), true);

    GeExecutor ge_executor;
    EXPECT_EQ(ge_executor.LoadModelWithQ(model_id, model_helper3.GetGeRootModel(), model_queue_param), SUCCESS);
    EXPECT_EQ(ge_executor.UnloadModel(model_id), SUCCESS);
    ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  }
}

TEST_F(DavinciModelTest, davinci_model_execute_with_input_align_attrs) {
  const auto SetUnknownOpKernelForNoTiling = [](const ComputeGraph::Vistor<NodePtr> &all_nodes) {
    GeTensorDesc tensor0(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor0, 64);
    AttrUtils::SetBool(tensor0, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor0, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 0);
    std::vector<int64_t> max_shape_list = {1, 10, 224, 224};
    AttrUtils::SetListInt(tensor0, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    GeTensorDesc tensor1(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor1, 64);
    AttrUtils::SetBool(tensor1, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor1, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 0);
    AttrUtils::SetListInt(tensor1, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    GeTensorDesc tensor2(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor2, 64);
    AttrUtils::SetBool(tensor2, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor2, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1024);
    AttrUtils::SetListInt(tensor2, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      if (op_desc->GetType() == DATA && op_desc->GetName() == "data0") {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateOutputDesc(0, tensor0);
        op_desc->SetOutputOffset({2048});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else if (op_desc->GetType() == DATA && op_desc->GetName() == "data1") {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateOutputDesc(0, tensor1);
        op_desc->SetOutputOffset({2048});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      }  else if (op_desc->GetType() == ADD) {
        op_desc->SetOpKernelLibName("AIcoreEngine");
        op_desc->UpdateInputDesc(0, tensor0);
        op_desc->UpdateInputDesc(1, tensor1);
        op_desc->UpdateOutputDesc(0, tensor2);
        op_desc->SetInputOffset({0, 2048});
        op_desc->SetInputOffset({1, 2048});
        op_desc->SetOutputOffset({2112});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateInputDesc(0, tensor2);
        op_desc->SetInputOffset({2112});
        op_desc->SetSrcName( { "add" } );
        op_desc->SetSrcIndex({ 0 });
      }
    }
  };

  auto add = OP_CFG(ADD).Attr(ATTR_NAME_OP_NO_TILING, true).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  auto data0 = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true).Attr(ATTR_NAME_INDEX, 0);
  auto data1 = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true).Attr(ATTR_NAME_INDEX, 1);
  auto output = OP_CFG(NETOUTPUT).Attr(ATTR_NAME_OP_NO_TILING, true);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("add", add));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("add", add));
    CHAIN(NODE("add", add)->EDGE(0, 0)->NODE("output", output));
  };

  auto graph = ToComputeGraph(g1);
  SetUnknownOpKernelForNoTiling(graph->GetDirectNode());
  EXPECT_NE(graph, nullptr);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  TBEKernelStore tbe_kernel_store;
  InitKernelTaskDef_TE(graph, *model_task_def, "add", tbe_kernel_store);
  InitEventTaskDef(graph, *model_task_def);
  InitFusionTaskDef(graph, *model_task_def);
  InitEndGraphDef(graph, *model_task_def, "output");
  InitProfilerTaskDef(graph, *model_task_def);


  DumpProperties dump_properties;
  dump_properties.SetDumpMode("all");
  dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
  DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  SetGeModelAttrs(ge_model, true);
  EXPECT_NE(ge_model, nullptr);
  // Serialization GeModel to memory.
  ModelHelper model_helper;
  model_helper.SetSaveMode(false);  // Save to buffer.
  ModelBufferData model_buffer;
  EXPECT_TRUE(tbe_kernel_store.Build());
  ge_model->SetTBEKernelStore(tbe_kernel_store);
  EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "file_name_prefix", model_buffer), SUCCESS);
  const ModelData model_data{model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", ""};
  // Test LoadModelWithQueueParam
  {
    ModelHelper model_helper3;
    EXPECT_EQ(model_helper3.LoadModel(model_data), SUCCESS);
    const std::vector<uint32_t> input_queue_ids{ 1001U, 1003U };
    QueueAttrs in_queue_0 = {.queue_id = 1001U, .device_type = NPU, .device_id = 0};
    QueueAttrs in_queue_1 = {.queue_id = 1003U, .device_type = NPU, .device_id = 0};
    const std::vector<uint32_t> output_queue_ids{ 1002U };
    QueueAttrs out_queue_0 = {.queue_id = 1002U, .device_type = NPU, .device_id = 0};
    ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
    uint32_t model_id = 0;
    ModelQueueParam model_queue_param{};
    model_queue_param.group_total_count = 1;
    model_queue_param.group_index = 0;
    model_queue_param.input_queues = input_queue_ids;
    model_queue_param.output_queues = output_queue_ids;
    model_queue_param.input_queues_attrs = {in_queue_0, in_queue_1};
    model_queue_param.output_queues_attrs = {out_queue_0};
    model_queue_param.is_dynamic_sched = true;
    model_queue_param.need_report_status = true;
    model_queue_param.input_align_attrs = {.align_max_cache_num = 4,
                                           .align_timeout = 200,
                                           .drop_when_not_align = true};
    DumpProperties dump_properties;
    dump_properties.enable_dump_ = "1";
    dump_properties.dump_step_ = "0|2-4|6";
    DumpManager::GetInstance().RemoveDumpProperties(0);
    DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
    EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(0).IsDumpOpen(), true);

    GeExecutor ge_executor;
    EXPECT_EQ(ge_executor.LoadModelWithQ(model_id, model_helper3.GetGeRootModel(), model_queue_param), PARAM_INVALID);
    ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  }
}

TEST_F(DavinciModelTest, davinci_model_execute_datadump_on_watcher_model_success) {
  auto dump_checker_stub = std::make_shared<ge::DumpCheckRuntimeStub>();
  ge::RuntimeStub::SetInstance(dump_checker_stub);
  auto &model_mgr = ModelManager::GetInstance();
  model_mgr.model_map_.clear();
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 10086), SUCCESS); // fixed sessionid
  model_executor.StartRunThread();
  GetContext().SetSessionId(10086);

  GraphId graph_id = 1;
  GraphNodePtr graph_node;
  GeRootModelPtr ge_root_model;
  GeModelPtr ge_model;
  (void)BuildGraphNode(graph_id, graph_node, ge_root_model, ge_model);

  DumpProperties dump_properties;
  dump_properties.SetDumpMode("output");
  dump_properties.AddPropertyValue(DUMP_LAYER_OP_MODEL, {"Less", "cond/pow"});
  dump_properties.AddPropertyValue(DUMP_WATCHER_MODEL, {"cond/add","add_n"});
  DumpManager::GetInstance().RemoveDumpProperties(10086);
  DumpManager::GetInstance().AddDumpProperties(10086, dump_properties);

  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 32));
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_mgr.model_map_.size(), 1);
  EXPECT_EQ(dump_checker_stub->GetDumpChecker().GetOpMappingInfoTaskSize("Less_To_add_n"), 7UL);

  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  DumpManager::GetInstance().RemoveDumpProperties(10086);
  GetThreadLocalContext().SetGraphOption({});
  ge::RuntimeStub::Reset();
}

TEST_F(DavinciModelTest, davinci_model_execute_dumpok) {
  const auto SetUnknownOpKernelForNoTiling = [](const ComputeGraph::Vistor<NodePtr> &all_nodes) {
    GeTensorDesc tensor0(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor0, 64);
    AttrUtils::SetBool(tensor0, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor0, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 0);
    std::vector<int64_t> max_shape_list = {1, 10, 224, 224};
    AttrUtils::SetListInt(tensor0, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    GeTensorDesc tensor1(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor1, 64);
    AttrUtils::SetBool(tensor1, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor1, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1024);
    AttrUtils::SetListInt(tensor1, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      if (op_desc->GetType() == DATA) {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateOutputDesc(0, tensor0);
        op_desc->SetOutputOffset({2048});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else if (op_desc->GetType() == ADD) {
        op_desc->SetOpKernelLibName("aicpu_ascend_kernel");//aicpu_ascend_kernel AIcoreEngine
        op_desc->UpdateInputDesc(0, tensor0);
        op_desc->UpdateOutputDesc(0, tensor1);
        op_desc->SetInputOffset({0, 2048});
        op_desc->SetOutputOffset({2112});
        op_desc->SetWorkspace(vector<int64_t>{32});
        std::vector<int64_t> tvm_workspace_memory_type = {ge::AicpuWorkSpaceType::CUST_LOG};
        ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_AICPU_WORKSPACE_TYPE, tvm_workspace_memory_type);
        op_desc->SetWorkspaceBytes(vector<int64_t>{32});
      } else if (op_desc->GetType() == HCOMREDUCE) {
        op_desc->SetOpKernelLibName("ops_kernel_info_hccl");
        op_desc->UpdateInputDesc(0, tensor0);
        op_desc->UpdateOutputDesc(0, tensor1);
        int32_t root_id = 0;
        ge::AttrUtils::SetInt(op_desc, HCOM_ATTR_ROOT_RANK, root_id);
        op_desc->SetInputOffset({2048});
        op_desc->SetOutputOffset({2048});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateInputDesc(0, tensor1);
        op_desc->SetInputOffset({2112});
        op_desc->SetSrcName( { "add" } );
        op_desc->SetSrcIndex({ 0 });
      }
    }
  };

  auto add = OP_CFG(ADD).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto hcom_reduce = OP_CFG(HCOMREDUCE).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto data1 = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto data2 = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto output = OP_CFG(NETOUTPUT).Attr(ATTR_NAME_OP_NO_TILING, true);
  DEF_GRAPH(g1) {
          CHAIN(NODE("data", data1)->EDGE(0, 0)->NODE("add", add));
          CHAIN(NODE("data", data2)->EDGE(0, 0)->NODE("hcom_reduce", hcom_reduce));
          CHAIN(NODE("hcom_reduce", hcom_reduce)->EDGE(0, 1)->NODE("add", add));
          CHAIN(NODE("add", add)->EDGE(0, 0)->NODE("output", output));
  };

  auto graph = ToComputeGraph(g1);
  SetUnknownOpKernelForNoTiling(graph->GetAllNodes());
  EXPECT_NE(graph, nullptr);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  TBEKernelStore tbe_kernel_store;
  CustAICPUKernelStore cpu_kernel_store;
  InitKernelTaskDef_CUST_CPU(graph, *model_task_def, "add", cpu_kernel_store);
  InitHcclTaskDef(graph, *model_task_def, "hcom_reduce", "HcomBroadcast");
  InitEventTaskDef(graph, *model_task_def);
  InitFusionTaskDef(graph, *model_task_def);
  InitEndGraphDef(graph, *model_task_def, "output");
  InitProfilerTaskDef(graph, *model_task_def);


  DumpProperties dump_properties;
  dump_properties.SetDumpMode("all");
  dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
  DumpManager::GetInstance().RemoveDumpProperties(0);
  DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  SetGeModelAttrs(ge_model);
  EXPECT_NE(ge_model, nullptr);

  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(graph);
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);;
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);

    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
}


TEST_F(DavinciModelTest, davinci_model_execute_dumpok_with_op_range) {
  const auto SetUnknownOpKernelForNoTiling = [](const ComputeGraph::Vistor<NodePtr> &all_nodes) {
    GeTensorDesc tensor0(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor0, 64);
    AttrUtils::SetBool(tensor0, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor0, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 0);
    std::vector<int64_t> max_shape_list = {1, 10, 224, 224};
    AttrUtils::SetListInt(tensor0, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    GeTensorDesc tensor1(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor1, 64);
    AttrUtils::SetBool(tensor1, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor1, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1024);
    AttrUtils::SetListInt(tensor1, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      if (op_desc->GetType() == DATA) {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateOutputDesc(0, tensor0);
        op_desc->SetOutputOffset({2048});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else if (op_desc->GetType() == ADD) {
        op_desc->SetOpKernelLibName("aicpu_ascend_kernel");//aicpu_ascend_kernel AIcoreEngine
        op_desc->UpdateInputDesc(0, tensor0);
        op_desc->UpdateOutputDesc(0, tensor1);
        op_desc->SetInputOffset({0, 2048});
        op_desc->SetOutputOffset({2112});
        op_desc->SetWorkspace(vector<int64_t>{32});
        std::vector<int64_t> tvm_workspace_memory_type = {ge::AicpuWorkSpaceType::CUST_LOG};
        ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_AICPU_WORKSPACE_TYPE, tvm_workspace_memory_type);
        op_desc->SetWorkspaceBytes(vector<int64_t>{32});
      } else if (op_desc->GetType() == HCOMREDUCE) {
        op_desc->SetOpKernelLibName("ops_kernel_info_hccl");
        op_desc->UpdateInputDesc(0, tensor0);
        op_desc->UpdateOutputDesc(0, tensor1);
        int32_t root_id = 0;
        ge::AttrUtils::SetInt(op_desc, HCOM_ATTR_ROOT_RANK, root_id);
        op_desc->SetInputOffset({2048});
        op_desc->SetOutputOffset({2048});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateInputDesc(0, tensor1);
        op_desc->SetInputOffset({2112});
        op_desc->SetSrcName( { "add" } );
        op_desc->SetSrcIndex({ 0 });
      }
    }
  };

  auto add = OP_CFG(ADD).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto hcom_reduce = OP_CFG(HCOMREDUCE).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto data1 = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto data2 = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto output = OP_CFG(NETOUTPUT).Attr(ATTR_NAME_OP_NO_TILING, true);
  DEF_GRAPH(g1) {
          CHAIN(NODE("data", data1)->EDGE(0, 0)->NODE("add", add));
          CHAIN(NODE("data", data2)->EDGE(0, 0)->NODE("hcom_reduce", hcom_reduce));
          CHAIN(NODE("hcom_reduce", hcom_reduce)->EDGE(0, 1)->NODE("add", add));
          CHAIN(NODE("add", add)->EDGE(0, 0)->NODE("output", output));
  };

  auto graph = ToComputeGraph(g1);
  SetUnknownOpKernelForNoTiling(graph->GetAllNodes());
  EXPECT_NE(graph, nullptr);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  TBEKernelStore tbe_kernel_store;
  CustAICPUKernelStore cpu_kernel_store;
  InitKernelTaskDef_CUST_CPU(graph, *model_task_def, "add", cpu_kernel_store);
  InitHcclTaskDef(graph, *model_task_def, "hcom_reduce", "HcomBroadcast");
  InitEventTaskDef(graph, *model_task_def);
  InitFusionTaskDef(graph, *model_task_def);
  InitEndGraphDef(graph, *model_task_def, "output");
  InitProfilerTaskDef(graph, *model_task_def);


  const uint64_t session_id = graph->GetSessionID();
  DumpProperties dump_properties;
  dump_properties.SetDumpMode("all");
  // model.om_name_ = "test";
  std::vector<std::pair<std::string, std::string>> op_ranges = {{"add", "add"}, {"hcom_reduce","add"}};
  dump_properties.SetOpDumpRange("", op_ranges);
  dump_properties.SetOpDumpRange("test", op_ranges);
  DumpManager::GetInstance().RemoveDumpProperties(session_id);
  DumpManager::GetInstance().AddDumpProperties(session_id, dump_properties);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  SetGeModelAttrs(ge_model);
  EXPECT_NE(ge_model, nullptr);
  ge_model->SetName("test");
  GetContext().SetSessionId(session_id);
  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    ge_root_model->Initialize(graph);
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);

    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
}

TEST_F(DavinciModelTest, davinci_model_execute_exception_dumpok) {
  const auto SetUnknownOpKernelForNoTiling = [](const ComputeGraph::Vistor<NodePtr> &all_nodes) {
    GeTensorDesc tensor0(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor0, 64);
    AttrUtils::SetBool(tensor0, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor0, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 0);
    std::vector<int64_t> max_shape_list = {1, 10, 224, 224};
    AttrUtils::SetListInt(tensor0, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    GeTensorDesc tensor1(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor1, 64);
    AttrUtils::SetBool(tensor1, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor1, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1024);
    AttrUtils::SetListInt(tensor1, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      if (op_desc->GetType() == DATA) {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateOutputDesc(0, tensor0);
        op_desc->SetOutputOffset({2048});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else if (op_desc->GetType() == ADD) {
        op_desc->SetOpKernelLibName("aicpu_ascend_kernel");//aicpu_ascend_kernel AIcoreEngine
        op_desc->UpdateInputDesc(0, tensor0);
        op_desc->UpdateOutputDesc(0, tensor1);
        op_desc->SetInputOffset({0, 2048});
        op_desc->SetOutputOffset({2112});
        op_desc->SetWorkspace(vector<int64_t>{32});
        std::vector<int64_t> tvm_workspace_memory_type = {ge::AicpuWorkSpaceType::CUST_LOG};
        ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_AICPU_WORKSPACE_TYPE, tvm_workspace_memory_type);
        op_desc->SetWorkspaceBytes(vector<int64_t>{32});
      } else {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateInputDesc(0, tensor1);
        op_desc->SetInputOffset({2112});
        op_desc->SetSrcName( { "add" } );
        op_desc->SetSrcIndex({ 0 });
      }
    }
  };

  setenv("NPU_COLLECT_PATH_EXE", "dump", true);
  gert::GlobalDumper::GetInstance()->ClearInnerExceptionDumpers();
  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kExceptionDump}));
  auto add = OP_CFG(ADD).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto data = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto output = OP_CFG(NETOUTPUT).Attr(ATTR_NAME_OP_NO_TILING, true);
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", data)->EDGE(0, 0)->NODE("add", add));
                  CHAIN(NODE("add", add)->EDGE(0, 0)->NODE("output", output));
                };

  auto graph = ToComputeGraph(g1);
  SetUnknownOpKernelForNoTiling(graph->GetAllNodes());
  EXPECT_NE(graph, nullptr);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  TBEKernelStore tbe_kernel_store;
  CustAICPUKernelStore cpu_kernel_store;
  std::string kernel_ext_info = gert::GetFakeExtInfo();
  InitKernelTaskDef_CUST_CPU(graph, *model_task_def, "add", cpu_kernel_store, kernel_ext_info);
  InitEventTaskDef(graph, *model_task_def);
  InitFusionTaskDef(graph, *model_task_def);
  InitEndGraphDef(graph, *model_task_def, "output");
  InitProfilerTaskDef(graph, *model_task_def);


  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  SetGeModelAttrs(ge_model);
  EXPECT_NE(ge_model, nullptr);

  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    ge_root_model->Initialize(graph);
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(false);

    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  unsetenv("NPU_COLLECT_PATH_EXE");
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
}

TEST_F(DavinciModelTest, davinci_model_execute_control_output) {
  std::vector<int64_t> shape = {16};
  DEF_GRAPH(assert_graph) {
    auto assert = OP_CFG(ASSERT).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF").TensorDesc(FORMAT_ND, DT_INT32, shape).InCnt(1).Build("assert");
    auto data1 =
        OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data1");
    data1->SetOutputOffset({0});
    assert->SetInputOffset({0});

    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(assert));
  };

  auto graph = ToComputeGraph(assert_graph);
  EXPECT_NE(graph, nullptr);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  TBEKernelStore tbe_kernel_store;
  InitKernelTaskDef_TE(graph, *model_task_def, "assert", tbe_kernel_store);
  InitEventTaskDef(graph, *model_task_def);
  InitFusionTaskDef(graph, *model_task_def);


  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120));
  EXPECT_NE(ge_model, nullptr);

  // Serialization GeModel to memory.
  ModelHelper model_helper;
  model_helper.SetSaveMode(false);  // Save to buffer.
  ModelBufferData model_buffer;
  EXPECT_TRUE(tbe_kernel_store.Build());
  ge_model->SetTBEKernelStore(tbe_kernel_store);
  EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "file_name_prefix", model_buffer), SUCCESS);
  const ModelData model_data{model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", ""};

  // Test LoadModelWithQ
  {
    const std::vector<uint32_t> input_queue_ids{1001U};
    const std::vector<uint32_t> output_queue_ids{1002U};
    ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
    uint32_t model_id = 0;
    GeExecutor ge_executor;
    EXPECT_EQ(ge_executor.LoadModelWithQ(model_id, model_data, input_queue_ids, output_queue_ids), SUCCESS);
    EXPECT_EQ(ge_executor.UnloadModel(model_id), SUCCESS);
    ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  }
}

UINT32 StubTiling1(gert::TilingContext *context) {
  context->SetNeedAtomic(false);
  context->SetTilingKey(666U);
  context->SetBlockDim(666U);
  size_t *workspace_size = context->GetWorkspaceSizes(1);
  *workspace_size = 66U;
  return ge::GRAPH_SUCCESS;
}

UINT32 StubTilingParse1(gert::KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

void* CompileInfoCreator1() {
  return new AddCompileInfo;
}

void TestDavinciModelExecuteWithSoftSyncOp(bool is_exception_dump_enabled = false) {
  std::vector<int64_t> shape = {16};
  DEF_GRAPH(add_graph) {
    auto add = OP_CFG(ADD)
                   .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
                   .TensorDesc(FORMAT_ND, DT_INT32, shape)
                   .InCnt(2)
                   .Build("add");
    auto data1 =
        OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data1");
    data1->SetOutputOffset({0});

    int32_t data_value_vec1[2] = {2, 4};
    GeTensorDesc data_tensor_desc(GeShape({2}), FORMAT_ND, DT_INT32);
    TensorUtils::SetDataOffset(data_tensor_desc, 16);
    TensorUtils::SetWeightSize(data_tensor_desc, 16);
    TensorUtils::SetSize(data_tensor_desc, 16);
    GeTensorPtr data_tensor1 = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec1, 2 * sizeof(int32_t));
    auto const1 = OP_CFG(CONSTANT).Weight(data_tensor1);

    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(add)->NODE("noop", NOOP));
    CHAIN(NODE("const_op", const1)->EDGE(0, 1)->NODE(add));
  };

  auto graph = ToComputeGraph(add_graph);
  EXPECT_NE(graph, nullptr);

  auto add = graph->FindNode("add");
  EXPECT_NE(add, nullptr);
  auto op_desc = add->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);
  const char tbeBin[] = "tbe_bin";
  vector<char> buffer(tbeBin, tbeBin + strlen(tbeBin));
  OpKernelBinPtr tbeKernelPtr = std::make_shared<OpKernelBin>("test_tvm", std::move(buffer));
  AttrUtils::SetStr(op_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetStr(op_desc, ge::ATTR_NAME_KERNEL_BIN_ID, "_add_node_1_fake_id");
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbeKernelPtr);
  AttrUtils::SetBool(op_desc, ge::ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  // tiling_data
  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("hahahaha");
  // 离线无法存储，最终会丢失
  op_desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);

  AttrUtils::SetBool(op_desc, "globalworkspace_type", true);
  AttrUtils::SetBool(op_desc, "support_dynamicshape", false);
  AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  std::vector<std::string> depends = {"axes"};
  AttrUtils::SetListStr(op_desc, "_op_infer_depends", depends);

  op_desc->SetInputOffset({0, 64});
  op_desc->SetOutputOffset({128});
  TensorUtils::SetSize(*op_desc->MutableInputDesc(0), 32);
  TensorUtils::SetSize(*op_desc->MutableInputDesc(1), 32);
  TensorUtils::SetSize(*op_desc->MutableOutputDesc(0), 32);
  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl(ADD);
  funcs->tiling = StubTiling1;
  funcs->tiling_parse = StubTilingParse1;
  funcs->compile_info_creator = CompileInfoCreator1;

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  TBEKernelStore tbe_kernel_store;

  InitKernelWithHandleTaskDef_TE(graph, *model_task_def, "add", tbe_kernel_store);
  InitEventTaskDef(graph, *model_task_def);
  InitFusionTaskDef(graph, *model_task_def);


  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  SetGeModelAttrs(ge_model);
  EXPECT_NE(ge_model, nullptr);

  // Serialization GeModel to memory.
  ModelHelper model_helper;
  model_helper.SetSaveMode(false);  // Save to buffer.
  ModelBufferData model_buffer;
  EXPECT_TRUE(tbe_kernel_store.Build());
  ge_model->SetTBEKernelStore(tbe_kernel_store);
  EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "file_name_prefix", model_buffer), SUCCESS);
  const ModelData model_data{model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", ""};

  // Test LoadModelWithQ
  {
    const std::vector<uint32_t> input_queue_ids{1001U};
    const std::vector<uint32_t> output_queue_ids{1002U};
    ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
    uint32_t model_id = 0;
    GeExecutor ge_executor;
    EXPECT_EQ(ge_executor.LoadModelWithQ(model_id, model_data, input_queue_ids, output_queue_ids), SUCCESS);
    if (is_exception_dump_enabled) {
      auto exception_dumper = *gert::GlobalDumper::GetInstance()->GetInnerExceptionDumpers().begin();
      EXPECT_EQ(exception_dumper->op_desc_info_.size(), 1);
      EXPECT_EQ(exception_dumper->op_desc_info_[0].tiling_data, "");
    }
    EXPECT_EQ(ge_executor.UnloadModel(model_id), SUCCESS);
    ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  }
}

TEST_F(DavinciModelTest, davinci_model_execute_soft_sync_op) {
  TestDavinciModelExecuteWithSoftSyncOp();
}

/**
 * 用例描述：软同步算子保存tiling data 和 arg table
 *
 * 预置条件：NA
 *
 * 测试步骤：
 * 1. 构造带软同步算子的静态图
 * 2. 加载执行
 *
 * 预期结果：
 * 1、保存到该软同步算子的tiling data 和arg table
 */
TEST_F(DavinciModelTest, DavinciModelExecute_SaveExceptionDump_WithSoftSyncOp) {
  ge::diagnoseSwitch::EnableExceptionDump();
  gert::GlobalDumper::GetInstance()->ClearInnerExceptionDumpers();
  TestDavinciModelExecuteWithSoftSyncOp(true);
  ge::diagnoseSwitch::DisableDumper();
}

TEST_F(DavinciModelTest, davinci_model_execute_static_shape_reuse_binary) {
  std::vector<int64_t> shape = {16};
  DEF_GRAPH(relu_graph) {
    auto relu = OP_CFG(RELU).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF").TensorDesc(FORMAT_ND, DT_INT32, shape)
        .InCnt(1).Build("relu");
    auto data1 =
        OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data1");
    data1->SetOutputOffset({0});

    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(relu)->NODE("noop", NOOP));
  };

  auto graph = ToComputeGraph(relu_graph);
  EXPECT_NE(graph, nullptr);

  auto relu = graph->FindNode("relu");
  EXPECT_NE(relu, nullptr);
  auto op_desc = relu->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);
  AttrUtils::SetBool(op_desc, "support_dynamicshape", false);
  AttrUtils::SetStr(op_desc, ATTR_NAME_SGT_CUBE_VECTOR_CORE_TYPE, "AiCore");
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "_relu_fake_id");
  const char kernel_bin[] = "kernel_bin";
  vector<char> buffer(kernel_bin, kernel_bin + strlen(kernel_bin));
  ge::OpKernelBinPtr kernel_bin_ptr = std::make_shared<ge::OpKernelBin>("test", std::move(buffer));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, kernel_bin_ptr);
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, ATTR_NAME_MAX_TILING_SIZE, 2);

  op_desc->SetInputOffset({0});
  op_desc->SetOutputOffset({128});
  TensorUtils::SetSize(*op_desc->MutableInputDesc(0), 32);
  TensorUtils::SetSize(*op_desc->MutableOutputDesc(0), 32);

  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("666");
  run_info->SetWorkspaces({1, 2, 3, 4, 5});
  op_desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  TBEKernelStore tbe_kernel_store;

  InitKernelWithHandleTaskDef_TE(graph, *model_task_def, "relu", tbe_kernel_store);
  InitEventTaskDef(graph, *model_task_def);
  InitFusionTaskDef(graph, *model_task_def);


  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  SetGeModelAttrs(ge_model);
  EXPECT_NE(ge_model, nullptr);

  // Test LoadModelOnline
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(graph);
  ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  auto runtime_stub = MockForKernelLaunchForStaticShapeUseBinary();
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  runtime_stub->Reset();
}

TEST_F(DavinciModelTest, davinci_model_execute_static_shape_ifa_memcheck) {
  auto graph = ShareGraph::IFASingleGraph();
  EXPECT_NE(graph, nullptr);
  auto ifa_node = graph->FindNode("IncreFlashAttention");
  EXPECT_NE(ifa_node, nullptr);
  auto ifa_op = ifa_node->GetOpDesc();
  EXPECT_NE(ifa_op, nullptr);
  AttrUtils::SetBool(ifa_op, "support_dynamicshape", false);
  AttrUtils::SetStr(ifa_op, ATTR_NAME_SGT_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(ifa_op, "compile_info_json", json_str);
  AttrUtils::SetStr(ifa_op, "op_unique_key", "ifa_key");
  AttrUtils::SetInt(ifa_op, ATTR_NAME_MAX_TILING_SIZE, 1000);
  AttrUtils::SetBool(ifa_op, "_memcheck", true);
  AttrUtils::SetInt(ifa_op, "ori_op_para_size", 24);

  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("6");
  run_info->SetWorkspaces({1, 2, 3});
  ifa_op->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);
  ge::ArgsFormatDesc args_format;
  size_t arg_id = 0;
  args_format.Append(ge::AddrType::INPUT, arg_id++);
  args_format.Append(ge::AddrType::INPUT_DESC, arg_id++, true);
  args_format.Append(ge::AddrType::INPUT_DESC, arg_id++, true);
  for (size_t i = 0; i < 12UL; i++) {
    args_format.Append(ge::AddrType::INPUT, arg_id++);
  }
  args_format.Append(ge::AddrType::OUTPUT, 0);
  auto args_format_str = args_format.ToString();

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  TBEKernelStore tbe_kernel_store;

  InitKernelWithHandleTaskDef_TE(graph, *model_task_def, "IncreFlashAttention", tbe_kernel_store);
  InitEventTaskDef(graph, *model_task_def);
  InitFusionTaskDef(graph, *model_task_def);


  auto &task_def = *(model_task_def->mutable_task(0));
  auto &kernel_def = *task_def.mutable_kernel_with_handle();
  auto &context = *kernel_def.mutable_context();
  context.set_args_format(args_format_str);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  SetGeModelAttrs(ge_model);
  EXPECT_NE(ge_model, nullptr);

  // Test LoadModelOnline
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(graph);
  ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  auto runtime_stub = MockForKernelLaunchForStaticShapeUseBinary();
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  runtime_stub->Reset();
}


TEST_F(DavinciModelTest, davinci_model_execute_static_shape_ifa_memcheck_args_list_is_null) {
  auto graph = ShareGraph::IFASingleGraph();
  EXPECT_NE(graph, nullptr);
  auto ifa_node = graph->FindNode("IncreFlashAttention");
  EXPECT_NE(ifa_node, nullptr);
  auto ifa_op = ifa_node->GetOpDesc();
  EXPECT_NE(ifa_op, nullptr);
  AttrUtils::SetBool(ifa_op, "support_dynamicshape", false);
  AttrUtils::SetStr(ifa_op, ATTR_NAME_SGT_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(ifa_op, "compile_info_json", json_str);
  AttrUtils::SetStr(ifa_op, "op_unique_key", "ifa_key");
  AttrUtils::SetInt(ifa_op, ATTR_NAME_MAX_TILING_SIZE, 1000);
  AttrUtils::SetBool(ifa_op, "_memcheck", true);
  AttrUtils::SetInt(ifa_op, "ori_op_para_size", 24);

  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("6");
  ifa_op->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);
  ge::ArgsFormatDesc args_format;
  args_format.Append(ge::AddrType::FFTS_ADDR, 0);
  auto args_format_str = args_format.ToString();

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  TBEKernelStore tbe_kernel_store;

  InitKernelWithHandleTaskDef_TE(graph, *model_task_def, "IncreFlashAttention", tbe_kernel_store);
  InitEventTaskDef(graph, *model_task_def);
  InitFusionTaskDef(graph, *model_task_def);


  auto &task_def = *(model_task_def->mutable_task(0));
  auto &kernel_def = *task_def.mutable_kernel_with_handle();
  auto &context = *kernel_def.mutable_context();
  context.set_args_format(args_format_str);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  SetGeModelAttrs(ge_model);
  EXPECT_NE(ge_model, nullptr);

  // Test LoadModelOnline
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(graph);
  ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  auto runtime_stub = MockForKernelLaunchForStaticShapeUseBinary();
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  runtime_stub->Reset();
}

TEST_F(DavinciModelTest, davinci_model_execute_static_shape_batch_memcheck) {
  auto graph = ShareGraph::BatchSingleGraph();
  EXPECT_NE(graph, nullptr);
  auto batch_node = graph->FindNode("Batch");
  EXPECT_NE(batch_node, nullptr);
  auto batch_op = batch_node->GetOpDesc();
  EXPECT_NE(batch_op, nullptr);
  AttrUtils::SetBool(batch_op, "support_dynamicshape", false);
  AttrUtils::SetStr(batch_op, ATTR_NAME_SGT_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(batch_op, "compile_info_json", json_str);
  AttrUtils::SetInt(batch_op, ATTR_NAME_MAX_TILING_SIZE, 1000);
  AttrUtils::SetBool(batch_op, "_memcheck", true);
  (void)ge::AttrUtils::SetStr(batch_op, "dynamicParamMode", "folded_with_desc");
  (void)ge::AttrUtils::SetStr(batch_op, "optionalInputMode", "gen_placeholder");
  ge::ArgsFormatDesc args_format;
  args_format.Append(ge::AddrType::INPUT_DESC, 0, true);
  args_format.Append(ge::AddrType::OUTPUT_DESC, 0, true);
  args_format.Append(ge::AddrType::OUTPUT, 1);
  args_format.Append(ge::AddrType::OUTPUT, 2);
  auto args_format_str = args_format.ToString();

  batch_op->SetInputOffset({0, 128, 256, 384});
  batch_op->SetOutputOffset({512, 640, 768, 896});

  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("6");
  run_info->SetWorkspaces({1, 2, 3});
  batch_op->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  TBEKernelStore tbe_kernel_store;

  InitKernelWithHandleTaskDef_TE(graph, *model_task_def, "Batch", tbe_kernel_store);
  InitEventTaskDef(graph, *model_task_def);
  InitFusionTaskDef(graph, *model_task_def);


  auto &task_def = *(model_task_def->mutable_task(0));
  auto &kernel_def = *task_def.mutable_kernel_with_handle();
  auto &context = *kernel_def.mutable_context();
  context.set_args_format(args_format_str);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  SetGeModelAttrs(ge_model);
  EXPECT_NE(ge_model, nullptr);

  // Test LoadModelOnline
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(graph);
  ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  auto runtime_stub = MockForKernelLaunchForStaticShapeUseBinary();
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  runtime_stub->Reset();
}

TEST_F(DavinciModelTest, davinci_model_execute_static_shape_batch_memcheck_no_argsformat) {
  auto graph = ShareGraph::BatchSingleGraph();
  EXPECT_NE(graph, nullptr);
  auto batch_node = graph->FindNode("Batch");
  EXPECT_NE(batch_node, nullptr);
  auto batch_op = batch_node->GetOpDesc();
  EXPECT_NE(batch_op, nullptr);
  AttrUtils::SetBool(batch_op, "support_dynamicshape", false);
  AttrUtils::SetStr(batch_op, ATTR_NAME_SGT_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(batch_op, "compile_info_json", json_str);
  AttrUtils::SetInt(batch_op, ATTR_NAME_MAX_TILING_SIZE, 1000);
  AttrUtils::SetBool(batch_op, "_memcheck", true);
  (void)ge::AttrUtils::SetStr(batch_op, "dynamicParamMode", "folded_with_desc");
  (void)ge::AttrUtils::SetStr(batch_op, "optionalInputMode", "gen_placeholder");
  ge::ArgsFormatDesc args_format;
  auto args_format_str = args_format.ToString();

  batch_op->SetInputOffset({0, 128, 256, 384});
  batch_op->SetOutputOffset({512, 640, 768, 896});

  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("6");
  run_info->SetWorkspaces({1, 2, 3});
  batch_op->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  TBEKernelStore tbe_kernel_store;

  InitKernelWithHandleTaskDef_TE(graph, *model_task_def, "Batch", tbe_kernel_store);
  InitEventTaskDef(graph, *model_task_def);
  InitFusionTaskDef(graph, *model_task_def);


  auto &task_def = *(model_task_def->mutable_task(0));
  auto &kernel_def = *task_def.mutable_kernel_with_handle();
  auto &context = *kernel_def.mutable_context();
  context.set_args_format(args_format_str);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  SetGeModelAttrs(ge_model);
  EXPECT_NE(ge_model, nullptr);

  // Test LoadModelOnline
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(graph);
  ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  auto runtime_stub = MockForKernelLaunchForStaticShapeUseBinary();
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  runtime_stub->Reset();
}

TEST_F(DavinciModelTest, davinci_model_execute_with_attached_vector_core) {
  std::vector<int64_t> shape = {16};
  DEF_GRAPH(relu_graph) {
                          auto relu = OP_CFG(RELU).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF").TensorDesc(FORMAT_ND, DT_INT32, shape)
                              .InCnt(1).Build("relu");
                          auto data1 =
                              OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, shape).InCnt(1).OutCnt(1).Build("data1");
                          data1->SetOutputOffset({0});

                          CHAIN(NODE(data1)->EDGE(0, 0)->NODE(relu)->NODE("noop", NOOP));
                        };

  auto graph = ToComputeGraph(relu_graph);
  EXPECT_NE(graph, nullptr);

  auto relu = graph->FindNode("relu");
  EXPECT_NE(relu, nullptr);
  auto op_desc = relu->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "_relu_fake_id");
  const char kernel_bin[] = "kernel_bin";
  vector<char> buffer(kernel_bin, kernel_bin + strlen(kernel_bin));
  ge::OpKernelBinPtr kernel_bin_ptr = std::make_shared<ge::OpKernelBin>("test", std::move(buffer));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, kernel_bin_ptr);
  AttrUtils::SetBool(op_desc, "support_dynamicshape", false);
  AttrUtils::SetStr(op_desc, ATTR_NAME_SGT_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, ATTR_NAME_MAX_TILING_SIZE, 2);

  op_desc->SetInputOffset({0});
  op_desc->SetOutputOffset({128});
  TensorUtils::SetSize(*op_desc->MutableInputDesc(0), 32);
  TensorUtils::SetSize(*op_desc->MutableOutputDesc(0), 32);

  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("666");
  run_info->SetWorkspaces({1, 2, 3, 4, 5});
  op_desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  TBEKernelStore tbe_kernel_store;

  InitKernelWithHandleTaskDef_Attached(graph, *model_task_def, "relu", tbe_kernel_store);
  InitEventTaskDef(graph, *model_task_def);
  InitFusionTaskDef(graph, *model_task_def);


  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  SetGeModelAttrs(ge_model);
  EXPECT_NE(ge_model, nullptr);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 2));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, "_vector_streams", 1));
  // Test LoadModelOnline
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(graph);
  ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  auto runtime_stub = MockForKernelLaunchForStaticShapeUseBinary();

  // profiling model subscribe on
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  runtime_stub->Reset();
}

TEST_F(DavinciModelTest, davinci_model_execute_atomic_clean_task) {
  const auto SetUnknownOpKernelForNoTiling = [](const ComputeGraph::Vistor<NodePtr> &all_nodes) {
    GeTensorDesc tensor0(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor0, 64);
    AttrUtils::SetBool(tensor0, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor0, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 0);
    std::vector<int64_t> max_shape_list = {1, 10, 224, 224};
    AttrUtils::SetListInt(tensor0, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    GeTensorDesc tensor1(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor1, 64);
    AttrUtils::SetBool(tensor1, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor1, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1024);
    AttrUtils::SetListInt(tensor1, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      if (op_desc->GetType() == DATA) {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateOutputDesc(0, tensor0);
        op_desc->SetOutputOffset({2048});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else if (op_desc->GetType() == ADD) {
        op_desc->SetOpKernelLibName("AIcoreEngine");
        op_desc->UpdateInputDesc(0, tensor0);
        op_desc->UpdateOutputDesc(0, tensor1);
        op_desc->SetInputOffset({0, 2048});
        op_desc->SetOutputOffset({2112});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateInputDesc(0, tensor1);
        op_desc->SetInputOffset({2112});
        op_desc->SetSrcName( { "add" } );
        op_desc->SetSrcIndex({ 0 });
      }
    }
  };

  auto add = OP_CFG(ADD).Attr(ATTR_NAME_OP_NO_TILING, true).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  auto data = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto output = OP_CFG(NETOUTPUT).Attr(ATTR_NAME_OP_NO_TILING, true);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", data)->EDGE(0, 0)->NODE("add", add));
    CHAIN(NODE("add", add)->EDGE(0, 0)->NODE("output", output));
  };

  auto graph = ToComputeGraph(g1);
  SetUnknownOpKernelForNoTiling(graph->GetDirectNode());
  EXPECT_NE(graph, nullptr);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  TBEKernelStore tbe_kernel_store;
  InitKernelTaskDef_Atomic(graph, *model_task_def, "add", tbe_kernel_store);
  InitEventTaskDef(graph, *model_task_def);
  InitFusionTaskDef(graph, *model_task_def);
  InitEndGraphDef(graph, *model_task_def, "output");
  InitProfilerTaskDef(graph, *model_task_def);

  DumpProperties dump_properties;
  dump_properties.SetDumpMode("all");
  dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
  DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  SetGeModelAttrs(ge_model);
  EXPECT_NE(ge_model, nullptr);

  {
    // Test LoadModelOnline
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    ge_root_model->Initialize(graph);
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);

    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }

  // Serialization GeModel to memory.
  ModelHelper model_helper;
  model_helper.SetSaveMode(false);  // Save to buffer.
  ModelBufferData model_buffer;
  EXPECT_TRUE(tbe_kernel_store.Build());
  ge_model->SetTBEKernelStore(tbe_kernel_store);
  EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "file_name_prefix", model_buffer), SUCCESS);
  const ModelData model_data{model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", ""};

  // Test LoadModelWithQ
  {
    const std::vector<uint32_t> input_queue_ids{ 1001U };
    const std::vector<uint32_t> output_queue_ids{ 1002U };
    ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
    uint32_t model_id = 0;
    GeExecutor ge_executor;
    EXPECT_EQ(ge_executor.LoadModelWithQ(model_id, model_data, input_queue_ids, output_queue_ids), SUCCESS);
    EXPECT_EQ(ge_executor.UnloadModel(model_id), SUCCESS);
    ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  }
}

TEST_F(DavinciModelTest, davinci_model_execute_with_aicpu_deploy_host) {
  auto n_queue_id = OP_CFG(CONSTANT);
  DEF_GRAPH(g1) {
    CHAIN(NODE("input", n_queue_id)->EDGE(0, 0)->NODE("aicpu_stub", SLICE)->EDGE(0, 0)->NODE("output", NETOUTPUT));
    CHAIN(NODE("input1", n_queue_id)->EDGE(0, 0)->NODE("aicore_aicpu_stub", SLICE)->EDGE(0, 1)->NODE("output"));
  };
  const auto graph = ToComputeGraph(g1);
  EXPECT_NE(graph, nullptr);
  const auto node = graph->FindNode("aicpu_stub");
  // set host mem
  node->GetOpDesc()->SetInputOffset({kMemoryHostSVMFeatureMapLogicBase + 0U});
  node->GetOpDesc()->SetOutputOffset({kMemoryHostSVMFeatureMapLogicBase + 1024U});
  EXPECT_TRUE(ge::AttrUtils::SetListInt(node->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, {RT_MEMORY_HOST_SVM}));
  EXPECT_TRUE(ge::AttrUtils::SetListInt(node->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, {RT_MEMORY_HOST_SVM}));
  const auto output = graph->FindNode("output");
  output->GetOpDesc()->SetInputOffset({0, 512});
  output->GetOpDesc()->SetSrcName({"input0", "input1"});
  output->GetOpDesc()->SetSrcIndex({0, 1});
  const auto aicore_aicpu_stub = graph->FindNode("aicore_aicpu_stub");
  aicore_aicpu_stub->GetOpDesc()->SetInputOffset({0});
  aicore_aicpu_stub->GetOpDesc()->SetOutputOffset({kMemoryHostSVMFeatureMapLogicBase + 0U});

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  std::string kernel_ext_info = std::string{7, 0, 0, 0, 4, 0, 0, 0, 2U << 4, 0, 0, 0};  // RT_KERNEL_HOST_ONLY (0x20)
  InitKernelExTaskDef(graph, *model_task_def, "aicpu_stub", kernel_ext_info);
  InitKernelTaskDef_AI_CPU(graph, *model_task_def, "aicore_aicpu_stub", kernel_ext_info);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_HOST_SVM_SIZE, 128));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_HOST_SVM_BASE_ADDR, kMemoryHostSVMFeatureMapLogicBase));

  EXPECT_NE(ge_model, nullptr);

  {
    // Test LoadModelOnline
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    ge_root_model->Initialize(graph);
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);

    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
}

TEST_F(DavinciModelTest, davinci_model_execute_with_aicpu_queue) {
  std::vector<NamedAttrs> resources(2);
  NamedAttrs &queue_resource = resources[0];
  AttrUtils::SetStr(queue_resource, "resource_type", "RES_QUEUE");
  AttrUtils::SetStr(queue_resource, "queue_name", "some_queue");
  AttrUtils::SetInt(queue_resource, "queue_depth", 2);
  AttrUtils::SetInt(queue_resource, "queue_id_idx", 0);

  NamedAttrs &channel_resource = resources[1];
  AttrUtils::SetStr(channel_resource, "resource_type", "RES_CHANNEL");

  auto n_queue_id = OP_CFG(CONSTANT);
  NamedAttrs op_resource;
  auto n_batch_dequeue = OP_CFG("BatchDequeue").Attr("_resource_list", resources);
  DEF_GRAPH(g1) {
    CHAIN(NODE("queue_id", n_queue_id)->EDGE(0, 0)->NODE("batch_dequeue", n_batch_dequeue));
  };
  const auto graph = ToComputeGraph(g1);
  EXPECT_NE(graph, nullptr);
  const auto node = graph->FindNode("batch_dequeue");
  node->GetOpDesc()->SetInputOffset({0});

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  InitKernelTaskDef_AI_CPU(graph, *model_task_def, "batch_dequeue");

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  SetGeModelAttrs(ge_model);
  EXPECT_NE(ge_model, nullptr);

  {
    // Test LoadModelOnline
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    ge_root_model->Initialize(graph);
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);

    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
}

TEST_F(DavinciModelTest, sample_davinci_model_execute_reuse_zero_copy_memory) {
  uint32_t mem_offset = 0;
  ComputeGraphPtr graph;
  BuildSampleGraph(graph, mem_offset);
  EXPECT_NE(graph, nullptr);

  GeModelPtr ge_model;
  BuildGraphModel(graph, ge_model, mem_offset);
  EXPECT_NE(ge_model, nullptr);

  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(graph);
  ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  GetThreadLocalContext().SetGraphOption(
    std::map<std::string, std::string>{{OPTION_EXEC_REUSE_ZERO_COPY_MEMORY, "1"}, {"ge.exec.hostInputIndexes", "0"}});
  GetThreadLocalContext().SetSessionOption(std::map<std::string, std::string>{{"ge.exec.overflow", "1"}});
  GetThreadLocalContext().SetSessionOption(std::map<std::string, std::string>{{"ge.graphLevelSat", "1"}});

  const static std::string kEnabled = "1";
  std::string reuse_zero_copy_memory;
  GetContext().GetOption(OPTION_EXEC_REUSE_ZERO_COPY_MEMORY, reuse_zero_copy_memory);
  EXPECT_EQ(reuse_zero_copy_memory, kEnabled);

  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.Initialize({{"ge.variableMemoryMaxSize", "5368709120"}}, graph->GetSessionID()), SUCCESS);
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  // model_executor session_id is 0, graph session_id is 10086
  (void)VarManagerPool::Instance().GetVarManager(graph->GetSessionID())->FreeVarMemory();

  GetThreadLocalContext().SetGraphOption(
    std::map<std::string, std::string>{{OPTION_EXEC_REUSE_ZERO_COPY_MEMORY, ""}, {"ge.exec.hostInputIndexes", ""}});
}

TEST_F(DavinciModelTest, sample_davinci_model_execute_cmo_offset_invalid) {
  uint32_t mem_offset = 0;
  ComputeGraphPtr graph;
  BuildSampleGraph(graph, mem_offset);
  EXPECT_NE(graph, nullptr);

  GeModelPtr ge_model;
  BuildGraphModel(graph, ge_model, mem_offset);
  EXPECT_NE(ge_model, nullptr);

  auto cmo = graph->FindNode("cmo1");
  EXPECT_NE(cmo, nullptr);
  AttrUtils::SetInt(cmo->GetOpDesc(), "offset", 200000000);

  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(graph);
  ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  GetThreadLocalContext().SetGraphOption(std::map<std::string, std::string>{{OPTION_EXEC_REUSE_ZERO_COPY_MEMORY, "1"}});
  GetThreadLocalContext().SetSessionOption(std::map<std::string, std::string>{{"ge.exec.overflow", "1"}});
  GetThreadLocalContext().SetSessionOption(std::map<std::string, std::string>{{"ge.graphLevelSat", "1"}});

  const static std::string kEnabled = "1";
  std::string reuse_zero_copy_memory;
  GetContext().GetOption(OPTION_EXEC_REUSE_ZERO_COPY_MEMORY, reuse_zero_copy_memory);
  EXPECT_EQ(reuse_zero_copy_memory, kEnabled);

  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.Initialize({{"ge.variableMemoryMaxSize", "5368709120"}}, graph->GetSessionID()), SUCCESS);
  EXPECT_NE(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  // model_executor session_id is 0, graph session_id is 10086
  (void)VarManagerPool::Instance().GetVarManager(graph->GetSessionID())->FreeVarMemory();

  GetThreadLocalContext().SetGraphOption(std::map<std::string, std::string>{{OPTION_EXEC_REUSE_ZERO_COPY_MEMORY, ""}});
}

// davinci model reuse memory
TEST_F(DavinciModelTest, load_model_with_extending_static_memory) {
  // set and check option
  std::map<std::string, std::string> options_map;
  options_map[STATIC_MEMORY_POLICY] = "2";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_TRUE(ModelUtils::IsGeUseExtendSizeMemory());

  // first model, malloc memory
  GraphId graph_id1 = 1001;
  GraphNodePtr graph_node1;
  GeRootModelPtr flow_root_model1;
  GeModelPtr ge_model1;
  (void)BuildGraphNode(graph_id1, graph_node1, flow_root_model1, ge_model1);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model1, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0));
  ModelExecutor model_executor;
  const uint64_t session_id = flow_root_model1->GetRootGraph()->GetSessionID();
  EXPECT_EQ(model_executor.Initialize({}, session_id), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.Initialize({{"ge.variableMemoryMaxSize", "5368709120"}}, session_id), SUCCESS);
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model1, graph_node1), SUCCESS);

  // second model, reuse memory of model1
  options_map[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "1";
  GetThreadLocalContext().SetGraphOption(options_map);
  GraphId graph_id2 = 1002;
  GraphNodePtr graph_node2;
  GeRootModelPtr flow_root_model2;
  GeModelPtr ge_model2;
  // if previous VarManager is removed, memory leak occurs when VarResource is destroyed
  // VarManagerPool::Instance().RemoveVarManager(session_id);
  (void)BuildGraphNode(graph_id2, graph_node2, flow_root_model2, ge_model2);

  EXPECT_TRUE(AttrUtils::SetInt(ge_model2, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0));
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model2, graph_node2), SUCCESS);
  auto &model_mgr = ModelManager::GetInstance();
  // Check the number of models loaded
  EXPECT_EQ(model_mgr.model_map_.size(), 2);
  // Check, two davinci model use the same memory base;
  EXPECT_EQ(model_mgr.model_map_.begin()->second->mem_base_, model_mgr.model_map_.rbegin()->second->mem_base_);
  model_mgr.model_map_.begin()->second->is_async_mode_ = true;

  // third model, not unload all model, molloc new memory
  GraphId graph_id3 = 1003;
  GraphNodePtr graph_node3;
  GeRootModelPtr flow_root_model3;
  GeModelPtr ge_model3;
  (void)VarManagerPool::Instance().GetVarManager(session_id)->FreeVarMemory();
  VarManagerPool::Instance().RemoveVarManager(session_id);
  (void)BuildGraphNode(graph_id3, graph_node3, flow_root_model3, ge_model3);
  VarManager::Instance(session_id)->SetMemManager(&MemManager::Instance());
  EXPECT_EQ(VarManager::Instance(session_id)->GetVarMemoryAddr(0, RT_MEMORY_HBM, 0), nullptr);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model3, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0));
  int64_t mem_offset = 0;
  AttrUtils::GetInt(ge_model3, ATTR_MODEL_MEMORY_SIZE, mem_offset);
  mem_offset += 64;
  EXPECT_TRUE(AttrUtils::SetInt(ge_model3, ATTR_MODEL_MEMORY_SIZE, mem_offset));
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model3, graph_node3), SUCCESS);
  // Check the number of models loaded
  EXPECT_EQ(model_mgr.model_map_.size(), 3);

  EXPECT_EQ(model_executor.UnloadGraph(flow_root_model1, graph_id1), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(flow_root_model2, graph_id2), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(flow_root_model3, graph_id3), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  GetThreadLocalContext().SetGraphOption({});
}

// davinci model reuse memory
TEST_F(DavinciModelTest, load_model_with_no_extending_static_memory) {
  // set and check option
  std::map<std::string, std::string> options_map;
  options_map[STATIC_MEMORY_POLICY] = "2";
  options_map[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "1";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_TRUE(ModelUtils::IsGeUseExtendSizeMemory());

  // first model, malloc memory
  GraphId graph_id1 = 1001;
  GraphNodePtr graph_node1;
  GeRootModelPtr flow_root_model1;
  GeModelPtr ge_model1;
  (void)BuildGraphNode(graph_id1, graph_node1, flow_root_model1, ge_model1);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model1, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 128));
  ModelExecutor model_executor;
  const uint64_t session_id = flow_root_model1->GetRootGraph()->GetSessionID();
  EXPECT_EQ(model_executor.Initialize({}, session_id), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.Initialize({{"ge.variableMemoryMaxSize", "5368709120"}}, session_id), SUCCESS);
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model1, graph_node1), SUCCESS);

  // second model, reuse memory of model1
  GraphId graph_id2 = 1002;
  GraphNodePtr graph_node2;
  GeRootModelPtr flow_root_model2;
  GeModelPtr ge_model2;
  (void)BuildGraphNode(graph_id2, graph_node2, flow_root_model2, ge_model2);

  EXPECT_TRUE(AttrUtils::SetInt(ge_model2, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 128));
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model2, graph_node2), SUCCESS);
  auto &model_mgr = ModelManager::GetInstance();
  // Check the number of models loaded
  EXPECT_EQ(model_mgr.model_map_.size(), 2);
  // Check, two davinci model use the same memory base;
  EXPECT_EQ(model_mgr.model_map_.begin()->second->mem_base_, model_mgr.model_map_.rbegin()->second->mem_base_);
  model_mgr.model_map_.begin()->second->is_async_mode_ = true;

  // Check the number of models loaded
  EXPECT_EQ(model_mgr.model_map_.size(), 2);

  EXPECT_EQ(model_executor.UnloadGraph(flow_root_model1, graph_id1), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(flow_root_model2, graph_id2), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  GetThreadLocalContext().SetGraphOption({});
}

TEST_F(DavinciModelTest, heterogeneous_load_model_with_extending_static_memory) {
  // set and check option
  std::map<std::string, std::string> options_map;
  options_map[STATIC_MEMORY_POLICY] = "2";
  options_map[OPTION_EXEC_MODEL_DEPLOY_MODE] = "SPMD";
  options_map[OPTION_EXEC_MODEL_DEPLOY_DEVICELIST] = "[0]";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_TRUE(ModelUtils::IsGeUseExtendSizeMemory());

  // first model, malloc memory
  GraphId graph_id1 = 1001;
  GraphNodePtr graph_node1;
  GeRootModelPtr flow_root_model1;
  GeModelPtr ge_model1;
  (void)BuildGraphNode(graph_id1, graph_node1, flow_root_model1, ge_model1);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model1, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0));
  ModelExecutor model_executor;
  const uint64_t session_id = flow_root_model1->GetRootGraph()->GetSessionID();
  EXPECT_EQ(model_executor.Initialize({}, session_id), SUCCESS);
  model_executor.StartRunThread();
  // Heterogeneous load
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model1, graph_node1), SUCCESS);

  // second model, reuse memory of model1
  options_map[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "1";
  GetThreadLocalContext().SetGraphOption(options_map);
  GraphId graph_id2 = 1002;
  GraphNodePtr graph_node2;
  GeRootModelPtr flow_root_model2;
  GeModelPtr ge_model2;
  // VarManagerPool::Instance().RemoveVarManager(session_id);

  (void)BuildGraphNode(graph_id2, graph_node2, flow_root_model2, ge_model2);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model2, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0));
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model2, graph_node2), SUCCESS);
  auto &model_mgr = ModelManager::GetInstance();
  // Check the number of models loaded
  EXPECT_EQ(model_mgr.model_map_.size(), 2);
  // Check, two davinci model use the same memory base;
  EXPECT_EQ(model_mgr.model_map_.begin()->second->mem_base_, model_mgr.model_map_.rbegin()->second->mem_base_);
  model_mgr.model_map_.begin()->second->is_async_mode_ = true;

  // third model, not unload all model, molloc new memory
  GraphId graph_id3 = 1003;
  GraphNodePtr graph_node3;
  GeRootModelPtr flow_root_model3;
  GeModelPtr ge_model3;
  VarManager::Instance(session_id)->FreeVarMemory();
  VarManagerPool::Instance().RemoveVarManager(session_id);
  VarManager::Instance(session_id)->SetMemManager(&MemManager::Instance());
  EXPECT_EQ(VarManager::Instance(session_id)->GetVarMemoryAddr(0, RT_MEMORY_HBM, 0), nullptr);
  (void)BuildGraphNode(graph_id3, graph_node3, flow_root_model3, ge_model3);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model3, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0));
  int64_t mem_offset = 0;
  AttrUtils::GetInt(ge_model3, ATTR_MODEL_MEMORY_SIZE, mem_offset);
  mem_offset += 64;
  EXPECT_TRUE(AttrUtils::SetInt(ge_model3, ATTR_MODEL_MEMORY_SIZE, mem_offset));
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model3, graph_node3), SUCCESS);
  // Check the number of models loaded
  EXPECT_EQ(model_mgr.model_map_.size(), 3);

  EXPECT_EQ(model_executor.UnloadGraph(flow_root_model1, graph_id1), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(flow_root_model2, graph_id2), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(flow_root_model3, graph_id3), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  GetThreadLocalContext().SetGraphOption({});
}

TEST_F(DavinciModelTest, davinci_model_execute_no_tiling_without_q) {
  const auto SetUnknownOpKernelForNoTiling = [](const ComputeGraph::Vistor<NodePtr> &all_nodes) {
    GeTensorDesc tensor0(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor0, 64);
    AttrUtils::SetBool(tensor0, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor0, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 0);
    std::vector<int64_t> max_shape_list = {1, 10, 224, 224};
    AttrUtils::SetListInt(tensor0, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    GeTensorDesc tensor1(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor1, 64);
    AttrUtils::SetBool(tensor1, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor1, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1024);
    AttrUtils::SetListInt(tensor1, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    REG_HIDDEN_INPUTS_FUNC(
        ge::HiddenInputsType::HCOM, [](const ge::OpDescPtr &op_desc, std::vector<void *> &addr) -> ge::graphStatus {
          HcomTopoInfo::TopoInfo topo_info;
          topo_info.rank_size = 8;
          topo_info.notify_handle = reinterpret_cast<void *>(0x800);
          EXPECT_EQ(HcomTopoInfo::Instance().SetGroupTopoInfo("hccl_world_group", topo_info), GRAPH_SUCCESS);
          EXPECT_EQ(HcomTopoInfo::Instance().SetGroupTopoInfo("hccl_world_group_2", topo_info), GRAPH_SUCCESS);
          return ge::GRAPH_SUCCESS;
        });

    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      AttrUtils::SetStr(op_desc, "group", "hccl_world_group");
      AttrUtils::SetStr(op_desc, "test_group", "hccl_world_group_2");
      op_desc->SetAttachedStreamIds({0});
      if (op_desc->GetType() == DATA) {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateOutputDesc(0, tensor0);
        op_desc->SetOutputOffset({2048});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else if (op_desc->GetType() == ADD) {
        op_desc->SetOpKernelLibName("AIcoreEngine");
        op_desc->UpdateInputDesc(0, tensor0);
        op_desc->UpdateOutputDesc(0, tensor1);
        op_desc->SetInputOffset({0, 2048});
        op_desc->SetOutputOffset({2112});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateInputDesc(0, tensor1);
        op_desc->SetInputOffset({2112});
        op_desc->SetSrcName( { "add" } );
        op_desc->SetSrcIndex({ 0 });
      }
    }
  };
  int32_t data_value_vec1[2] = {0, 0};
  GeTensorDesc data_tensor_desc(GeShape({2}), FORMAT_ND, DT_INT32);
  GeTensorPtr data_tensor1 = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec1, 2 * sizeof(int32_t));
  auto const1 = OP_CFG(CONSTANT).Weight(data_tensor1).Attr(ATTR_NAME_OP_NO_TILING, true);

  auto add = OP_CFG(ADD).Attr(ATTR_NAME_OP_NO_TILING, true).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  auto output = OP_CFG(NETOUTPUT).Attr(ATTR_NAME_OP_NO_TILING, true);
  DEF_GRAPH(g1) {
      CHAIN(NODE("const1", const1)->EDGE(0, 0)->NODE("add", add));
      CHAIN(NODE("data", DATA)->EDGE(0, 1)->NODE("add", add)->EDGE(0, 0)->NODE("output", output));
  };

  auto graph = ToComputeGraph(g1);
  SetUnknownOpKernelForNoTiling(graph->GetDirectNode());
  EXPECT_NE(graph, nullptr);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  TBEKernelStore tbe_kernel_store;
  InitKernelTaskDef_TE(graph, *model_task_def, "add", tbe_kernel_store);
  InitEventTaskDef(graph, *model_task_def);
  InitNotifyTaskDef(graph, *model_task_def, UINT32_MAX, "test_group");
  InitNotifyWaitTaskDef(graph, *model_task_def, UINT32_MAX, "");
  InitFusionTaskDef(graph, *model_task_def);
  InitEndGraphDef(graph, *model_task_def, "output");
  InitProfilerTaskDef(graph, *model_task_def);


  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_NOTIFY_NUM, 2));
  EXPECT_TRUE(AttrUtils::SetListInt(ge_model, ATTR_MODEL_NOTIFY_TYPES, {0, 1}));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120));
  EXPECT_NE(ge_model, nullptr);
  HcomTopoInfo::TopoInfo topo_info;
  topo_info.rank_size = 8;
  topo_info.notify_handle = reinterpret_cast<void *>(0x800);
  topo_info.local_window_size = 209715200;
  EXPECT_EQ(HcomTopoInfo::Instance().SetGroupTopoInfo("hccl_world_group", topo_info), GRAPH_SUCCESS);
  uint64_t local_window_size = 0;
  EXPECT_EQ(HcomTopoInfo::Instance().GetGroupLocalWindowSize("hccl_world_group", local_window_size), GRAPH_SUCCESS);
  EXPECT_EQ(local_window_size, static_cast<uint64_t>(209715200));
  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    ge_root_model->Initialize(graph);
    GeExecutor ge_executor;
    uint32_t model_id = 0;
    EXPECT_NE(ge_executor.LoadModelWithoutQ(model_id, ge_root_model), SUCCESS);
  }

  {
    // Test LoadModelOnline
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    ge_root_model->Initialize(graph);
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);

    GeExecutor ge_executor;
    uint32_t model_id = 0;
    EXPECT_NE(ge_executor.LoadModelWithoutQ(model_id, ge_root_model), SUCCESS);
    EXPECT_EQ(ge_executor.UnloadModel(model_id), SUCCESS);
  }
}

TEST_F(DavinciModelTest, sdma_dump_with_qos) {
  uint32_t mem_offset = 0U;
  ComputeGraphPtr graph;
  BuildHcclSampleGraphWithQos(graph, mem_offset);
  EXPECT_NE(graph, nullptr);

  GeModelPtr ge_model;
  BuildGraphModel(graph, ge_model, mem_offset);
  EXPECT_NE(ge_model, nullptr);

  std::vector<gert::Tensor> input_tensors(4);
  TensorCheckUtils::ConstructGertTensor(input_tensors[0], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(input_tensors[1], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(input_tensors[2], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(input_tensors[3], {1}, DT_INT64, FORMAT_ND);

  std::vector<uint32_t> model_ids;

  DumpProperties dump_properties;
  dump_properties.is_train_op_debug_ = true;
  DumpManager::GetInstance().AddDumpProperties(graph->GetSessionID(), dump_properties);

  setenv(kEnvGeuseStaticMemory.c_str(), "1", 1);
  {
    // Test LoadModelOnline: RunAsyncListener
    const auto ge_root_model = MakeShared<GeRootModel>();
    const auto graph_node = MakeShared<GraphNode>(graph->GetGraphID());
    ge_root_model->Initialize(graph);
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->IncreaseLoadCount();

    // Callback for execute.
    std::mutex run_mutex;
    std::condition_variable model_run_cv;
    Status run_status = FAILED;
    const auto callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
      std::unique_lock<std::mutex> lock(run_mutex);
      run_status = status;
      model_run_cv.notify_one();
    };

    // RunArgsV2 of Graph.
    GEThreadLocalContext context;
    context.SetGraphOption(
        {{OPTION_EXEC_DYNAMIC_EXECUTE_MODE, "lazy_recompile"}, {OPTION_EXEC_ENABLE_COPY_OUTPUT_ADDR, "1"}});
    error_message::ErrorManagerContext error_context;
    graph_node->Lock();
    std::shared_ptr<RunArgs> arg;
    arg = std::make_shared<RunArgs>();
    ASSERT_TRUE(arg != nullptr);
    arg->graph_node = graph_node;
    arg->graph_id = graph->GetGraphID();
    arg->session_id = graph->GetSessionID();
    arg->error_context = error_context;
    arg->input_tensor = std::move(input_tensors);
    arg->context = context;
    arg->callback = callback;
    // Load and execute.
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.Initialize({{"ge.variableMemoryMaxSize", "12800"}}, graph->GetSessionID()), SUCCESS);
    EXPECT_EQ(model_executor.PushRunArgs(arg), SUCCESS);

    // Wait for execute.
    std::unique_lock<std::mutex> lock(run_mutex);
    EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
    EXPECT_EQ(run_status, SUCCESS);
    model_ids.emplace_back(ge_root_model->GetModelId());

    // Unload model of graph
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph->GetGraphID()), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  DumpManager::GetInstance().RemoveDumpProperties(graph->GetSessionID());
  unsetenv(kEnvGeuseStaticMemory.c_str());
}

TEST_F(DavinciModelTest, davinci_model_with_non_zero_cpy_inpouts) {
  const auto SetUnknownOpKernelForNoTiling = [](const ComputeGraph::Vistor<NodePtr> &all_nodes) {
    GeTensorDesc tensor0(GeShape({1, 1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor0, 64);
    AttrUtils::SetBool(tensor0, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor0, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 0);
    std::vector<int64_t> max_shape_list = {1, 10, 224, 224};
    AttrUtils::SetListInt(tensor0, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    GeTensorDesc tensor1(GeShape({1, 1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor1, 64);
    AttrUtils::SetBool(tensor1, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor1, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1024);
    AttrUtils::SetListInt(tensor1, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      if (op_desc->GetType() == DATA) {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateOutputDesc(0, tensor0);
        op_desc->SetOutputOffset({2048});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else if (op_desc->GetType() == ADD) {
        op_desc->SetOpKernelLibName("AIcoreEngine");
        op_desc->UpdateInputDesc(0, tensor0);
        op_desc->UpdateOutputDesc(0, tensor1);
        op_desc->SetInputOffset({0, 2048});
        op_desc->SetOutputOffset({2112});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateInputDesc(0, tensor1);
        op_desc->SetInputOffset({2112});
        op_desc->SetSrcName( { "add" } );
        op_desc->SetSrcIndex({ 0 });
      }
    }
  };

  auto add = OP_CFG(ADD).Attr(ATTR_NAME_OP_NO_TILING, true).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  auto data = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto output = OP_CFG(NETOUTPUT).Attr(ATTR_NAME_OP_NO_TILING, true);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", data)->EDGE(0, 0)->NODE("add", add));
    CHAIN(NODE("add", add)->EDGE(0, 0)->NODE("output", output));
  };
  auto graph = ToComputeGraph(g1);
  SetUnknownOpKernelForNoTiling(graph->GetDirectNode());
  EXPECT_NE(graph, nullptr);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  TBEKernelStore tbe_kernel_store;
  InitKernelTaskDef_TE(graph, *model_task_def, "add", tbe_kernel_store);
  InitEventTaskDef(graph, *model_task_def);
  InitFusionTaskDef(graph, *model_task_def);
  InitEndGraphDef(graph, *model_task_def, "output");
  InitProfilerTaskDef(graph, *model_task_def);


  DumpProperties dump_properties;
  dump_properties.SetDumpMode("all");
  dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
  DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  SetGeModelAttrs(ge_model);
  EXPECT_NE(ge_model, nullptr);

  {
    // Test LoadModelOnline
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(graph);
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);

    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  GetThreadLocalContext().SetGraphOption(std::map<std::string, std::string>{{OPTION_EXEC_REUSE_ZERO_COPY_MEMORY, "1"}});
  // Serialization GeModel to memory.
  ModelHelper model_helper;
  model_helper.SetSaveMode(false);  // Save to buffer.
  ModelBufferData model_buffer;
  EXPECT_TRUE(tbe_kernel_store.Build());
  ge_model->SetTBEKernelStore(tbe_kernel_store);
  EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "file_name_prefix", model_buffer), SUCCESS);
  const ModelData model_data{model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", ""};
  // Test LoadModelWithQueueParam
  {
    ModelHelper model_helper3;
    EXPECT_EQ(model_helper3.LoadModel(model_data), SUCCESS);
    const std::vector<uint32_t> input_queue_ids{ 1001U };
    QueueAttrs in_queue_0 = {.queue_id = 1001U, .device_type = NPU, .device_id = 0};
    const std::vector<uint32_t> output_queue_ids{ 1002U };
    QueueAttrs out_queue_0 = {.queue_id = 1002U, .device_type = NPU, .device_id = 0};
    ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
    uint32_t model_id = 0;
    ModelQueueParam model_queue_param{};
    model_queue_param.group_total_count = 1;
    model_queue_param.group_index = 0;
    model_queue_param.input_queues = input_queue_ids;
    model_queue_param.output_queues = output_queue_ids;
    model_queue_param.input_queues_attrs = {in_queue_0};
    model_queue_param.output_queues_attrs = {out_queue_0};

    GeExecutor ge_executor;
    EXPECT_EQ(ge_executor.LoadModelWithQ(model_id, model_helper3.GetGeRootModel(), model_queue_param), SUCCESS);
    EXPECT_EQ(ge_executor.UnloadModel(model_id), SUCCESS);
    ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  }
}

TEST_F(DavinciModelTest, davinci_model_error_tracking_test) {
  const auto SetUnknownOpKernelForNoTiling = [](const ComputeGraph::Vistor<NodePtr> &all_nodes) {
    GeTensorDesc tensor0(GeShape({1, 1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor0, 64);
    AttrUtils::SetBool(tensor0, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor0, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 0);
    std::vector<int64_t> max_shape_list = {1, 10, 224, 224};
    AttrUtils::SetListInt(tensor0, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    GeTensorDesc tensor1(GeShape({1, 1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor1, 64);
    AttrUtils::SetBool(tensor1, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor1, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1024);
    AttrUtils::SetListInt(tensor1, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      if (op_desc->GetType() == DATA) {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateOutputDesc(0, tensor0);
        op_desc->SetOutputOffset({2048});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else if (op_desc->GetType() == ADD) {
        op_desc->SetOpKernelLibName("AIcoreEngine");
        op_desc->UpdateInputDesc(0, tensor0);
        op_desc->UpdateOutputDesc(0, tensor1);
        op_desc->SetInputOffset({0, 2048});
        op_desc->SetOutputOffset({2112});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
        AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, {"op1", "op2", "op3"});
      } else {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateInputDesc(0, tensor1);
        op_desc->SetInputOffset({2112});
        op_desc->SetSrcName( { "add" } );
        op_desc->SetSrcIndex({ 0 });
      }
    }
  };

  auto add = OP_CFG(ADD).Attr(ATTR_NAME_OP_NO_TILING, true).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  auto data = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto output = OP_CFG(NETOUTPUT).Attr(ATTR_NAME_OP_NO_TILING, true);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", data)->EDGE(0, 0)->NODE("add", add));
    CHAIN(NODE("add", add)->EDGE(0, 0)->NODE("output", output));
  };
  auto graph = ToComputeGraph(g1);
  SetUnknownOpKernelForNoTiling(graph->GetDirectNode());
  EXPECT_NE(graph, nullptr);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  TBEKernelStore tbe_kernel_store;
  InitKernelTaskDef_TE(graph, *model_task_def, "add", tbe_kernel_store);
  InitEventTaskDef(graph, *model_task_def);
  InitFusionTaskDef(graph, *model_task_def);
  InitEndGraphDef(graph, *model_task_def, "output");
  InitProfilerTaskDef(graph, *model_task_def);


  DumpProperties dump_properties;
  dump_properties.SetDumpMode("all");
  dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
  DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  SetGeModelAttrs(ge_model);
  EXPECT_NE(ge_model, nullptr);

  {
    // Test LoadModelOnline
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    ge_root_model->Initialize(graph);
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);

    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    ErrorTrackingOpInfo op_info;
    EXPECT_TRUE(ErrorTracking::GetInstance().GetGraphTaskOpdescInfo(0, 0, op_info));
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_FALSE(ErrorTracking::GetInstance().GetGraphTaskOpdescInfo(0, 0, op_info));
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  GetThreadLocalContext().SetGraphOption(std::map<std::string, std::string>{{OPTION_EXEC_REUSE_ZERO_COPY_MEMORY, "1"}});
  // Serialization GeModel to memory.
  ModelHelper model_helper;
  model_helper.SetSaveMode(false);  // Save to buffer.
  ModelBufferData model_buffer;
  EXPECT_TRUE(tbe_kernel_store.Build());
  ge_model->SetTBEKernelStore(tbe_kernel_store);
  EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "file_name_prefix", model_buffer), SUCCESS);
  const ModelData model_data{model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", ""};
  // Test LoadModelWithQueueParam
  {
    ModelHelper model_helper3;
    EXPECT_EQ(model_helper3.LoadModel(model_data), SUCCESS);
    const std::vector<uint32_t> input_queue_ids{ 1001U };
    QueueAttrs in_queue_0 = {.queue_id = 1001U, .device_type = NPU, .device_id = 0};
    const std::vector<uint32_t> output_queue_ids{ 1002U };
    QueueAttrs out_queue_0 = {.queue_id = 1002U, .device_type = NPU, .device_id = 0};
    ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
    uint32_t model_id = 0;
    ModelQueueParam model_queue_param{};
    model_queue_param.group_total_count = 1;
    model_queue_param.group_index = 0;
    model_queue_param.input_queues = input_queue_ids;
    model_queue_param.output_queues = output_queue_ids;
    model_queue_param.input_queues_attrs = {in_queue_0};
    model_queue_param.output_queues_attrs = {out_queue_0};

    GeExecutor ge_executor;
    EXPECT_EQ(ge_executor.LoadModelWithQ(model_id, model_helper3.GetGeRootModel(), model_queue_param), SUCCESS);
    rtExceptionInfo rt_exception_info;
    rt_exception_info.streamid = 0;
    rt_exception_info.taskid = 0;
    ErrorTrackingCallback(&rt_exception_info);
    rt_exception_info.retcode = ACL_ERROR_RT_AICORE_OVER_FLOW;
    ErrorTrackingCallback(&rt_exception_info);
    EXPECT_EQ(ge_executor.UnloadModel(model_id), SUCCESS);
    ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  }
}

TEST_F(DavinciModelTest, update_task_id_success) {
  auto add_op = std::make_shared<OpDesc>("Add", "Add");
  uint32_t model_id = 0;
  uint32_t stream_id = 1;
  uint32_t old_task_id = 100;
  uint32_t new_task_id = 200;

  ErrorTracking::GetInstance().SaveGraphTaskOpdescInfo(add_op, old_task_id, stream_id, model_id);

  auto &task_map = ErrorTracking::GetInstance().graph_task_to_op_info_[model_id];
  TaskKey old_key(old_task_id, stream_id);
  EXPECT_NE(task_map.find(old_key), task_map.end());
  EXPECT_EQ(task_map[old_key].op_name, "Add");

  ErrorTracking::GetInstance().UpdateTaskId(old_task_id, new_task_id, stream_id, model_id);

  TaskKey new_key(new_task_id, stream_id);
  EXPECT_EQ(task_map.find(old_key), task_map.end());
  EXPECT_NE(task_map.find(new_key), task_map.end());
  EXPECT_EQ(task_map[new_key].op_name, "Add");
}

TEST_F(DavinciModelTest, update_task_id_not_found) {
  uint32_t model_id = 0;
  uint32_t stream_id = 1;
  uint32_t old_task_id = 999;
  uint32_t new_task_id = 200;

  auto &task_map = ErrorTracking::GetInstance().graph_task_to_op_info_[model_id];
  size_t initial_size = task_map.size();

  ErrorTracking::GetInstance().UpdateTaskId(old_task_id, new_task_id, stream_id, model_id);

  EXPECT_EQ(task_map.size(), initial_size);
}

TEST_F(DavinciModelTest, update_task_id_model_not_found) {
  uint32_t model_id = 0;
  uint32_t null_model_id = 999;
  uint32_t stream_id = 1;
  uint32_t old_task_id = 999;
  uint32_t new_task_id = 200;

  auto &task_map = ErrorTracking::GetInstance().graph_task_to_op_info_[model_id];
  size_t initial_size = task_map.size();


  ErrorTracking::GetInstance().UpdateTaskId(old_task_id, new_task_id, stream_id, null_model_id);

  EXPECT_EQ(task_map.size(), initial_size);
}

TEST_F(DavinciModelTest, memcpy_async_task_success) {
  DavinciModel model(0, nullptr);
  model.runtime_param_.logic_mem_base = 0x08003000;
  model.runtime_param_.logic_var_base = 0x08008000;
  model.runtime_param_.mem_size = 0x5000;
  model.runtime_param_.mem_base = 0x12345678;
  model.runtime_param_.var_size = 0x1000;
  model.runtime_param_.fileconstant_addr_mapping[0x08008000] = 0x08008000;

  const auto op_desc = std::make_shared<OpDesc>("memcpyasync", MEMCPYASYNC);
  model.op_list_[0] = op_desc;

  domi::TaskDef task_def;
  domi::MemcpyAsyncDef *memcpy_async = task_def.mutable_memcpy_async();
  memcpy_async->set_src(0x08008000);
  memcpy_async->set_dst(0x08003000);
  memcpy_async->set_dst_max(512);
  memcpy_async->set_count(1);
  memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
  memcpy_async->set_op_index(0);

  MemcpyAsyncTaskInfo memcpy_async_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(memcpy_async_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  EXPECT_EQ(task_run_param.parsed_input_addrs.size(), 1U);
  EXPECT_EQ(task_run_param.parsed_output_addrs.size(), 1U);
  EXPECT_EQ(task_run_param.parsed_input_addrs[0].logic_addr, 0x08008000);
  EXPECT_EQ(task_run_param.parsed_output_addrs[0].logic_addr, 0x12345678);
  EXPECT_EQ(task_run_param.parsed_input_addrs[0].memory_type, kConstantMemType);
  EXPECT_EQ(task_run_param.parsed_output_addrs[0].memory_type, kFmMemType);
  int64_t op_index = memcpy_async_task_info.ParseOpIndex(task_def);
  EXPECT_EQ(op_index, 0);
  model.runtime_param_.fileconstant_addr_mapping.clear();
}

TEST_F(DavinciModelTest, memcpy_async_with_p2p_task_success) {
  DavinciModel model(0, nullptr);
  model.runtime_param_.logic_mem_base = 0x08003000;
  model.runtime_param_.logic_var_base = 0x18008000;
  model.runtime_param_.mem_size = 0x5000;
  model.runtime_param_.mem_base = 0x12345678;
  model.runtime_param_.var_size = 0x1000;
  model.runtime_param_.fileconstant_addr_mapping[0x18008000] = 0x18008000;
  MemInfo p2p_info;
  p2p_info.memory_type = RT_MEMORY_P2P_DDR;
  p2p_info.logic_memory_base = 0UL;
  p2p_info.memory_base = reinterpret_cast<uint8_t *>(0x70000000);
  p2p_info.memory_size = 0x2000;
  model.runtime_param_.memory_infos[RT_MEMORY_P2P_DDR] = p2p_info;
  const auto op_desc = std::make_shared<OpDesc>("memcpyasync", MEMCPYASYNC);
  model.op_list_[0] = op_desc;

  domi::TaskDef task_def;
  domi::MemcpyAsyncDef *memcpy_async = task_def.mutable_memcpy_async();
  memcpy_async->set_src(0x08008000);
  memcpy_async->set_dst(0x08009000);
  memcpy_async->set_dst_max(512);
  memcpy_async->set_count(1);
  memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
  memcpy_async->set_op_index(0);

  MemcpyAsyncTaskInfo memcpy_async_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(memcpy_async_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  EXPECT_EQ(task_run_param.parsed_input_addrs.size(), 1U);
  EXPECT_EQ(task_run_param.parsed_output_addrs.size(), 1U);
  EXPECT_EQ(task_run_param.parsed_input_addrs[0].logic_addr, 0x70000000);
  EXPECT_EQ(task_run_param.parsed_output_addrs[0].logic_addr, 0x70001000);
  EXPECT_EQ(task_run_param.parsed_input_addrs[0].memory_type, RT_MEMORY_P2P_DDR);
  EXPECT_EQ(task_run_param.parsed_output_addrs[0].memory_type, RT_MEMORY_P2P_DDR);
  int64_t op_index = memcpy_async_task_info.ParseOpIndex(task_def);
  EXPECT_EQ(op_index, 0);
  model.runtime_param_.memory_infos.clear();
  model.runtime_param_.fileconstant_addr_mapping.clear();
}

TEST_F(DavinciModelTest, GetEventIdForBlockingAicpuOp_fail) {
  DavinciModel model(0, nullptr);
  GeModelPtr ge_model = MakeShared<GeModel>();
  model.Assign(ge_model);

  OpDescPtr op_desc = CreateOpDesc("data", "Data");
  rtStream_t stream = (rtStream_t)0;
  uint32_t event_id = 0;

  model.stream_2_event_[stream] = (rtEvent_t)2;
  g_runtime_stub_mock = "rtGetEventID";
  EXPECT_NE(model.GetEventIdForBlockingAicpuOp(op_desc, stream, event_id), SUCCESS);

  model.stream_2_event_.clear();
  EXPECT_NE(model.GetEventIdForBlockingAicpuOp(op_desc, stream, event_id), SUCCESS);

  g_runtime_stub_mock = "rtEventCreateWithFlag";
  EXPECT_EQ(model.GetEventIdForBlockingAicpuOp(op_desc, stream, event_id), SUCCESS); //??
}

/**
 * 用例描述：加载model时，设备上实际可用的stream数量小于model需要的值，已加载的model数量为1，
 *          已加载model占用的stream资源 + 设备可用的stream资源 = model需要的stream值，加载model成功
 * 预置条件：
 * 1.流创建及流销毁的桩函数中增加对g_free_stream_num的维护，
 *   g_free_stream_num初始值为2048，流创建时g_free_stream_num减1，流销毁时g_free_stream_num加1
 * 测试步骤
 * 1.构造单个计算图1，ATTR_MODEL_STREAM_NUM设置为32，并加载该子图
 * 2.构造单个计算图2，ATTR_MODEL_STREAM_NUM设置为设备上当前可用的stream数量 + 32
 * 预期结果
 * 1.model1 unload，model2 加载成功
 */
TEST_F(DavinciModelTest, davinci_model_load_check_and_release_stream_resource_success) {
  auto &model_mgr = ModelManager::GetInstance();
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 10086), SUCCESS); // fixed sessionid
  model_executor.StartRunThread();

  GraphId graph_id_1 = 1;
  GraphNodePtr graph_node_1;
  GeRootModelPtr flow_root_model_1;
  GeModelPtr ge_model_1;
  (void)BuildGraphNode(graph_id_1, graph_node_1, flow_root_model_1, ge_model_1);

  EXPECT_TRUE(AttrUtils::SetInt(ge_model_1, ATTR_MODEL_STREAM_NUM, 32));
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model_1, graph_node_1), SUCCESS);
  EXPECT_EQ(model_mgr.model_map_.size(), 1);
  model_mgr.model_map_.begin()->second->is_async_mode_ = true;

  GraphId graph_id_2 = 2;
  GraphNodePtr graph_node_2;
  GeRootModelPtr flow_root_model_2;
  GeModelPtr ge_model_2;
  (void)BuildGraphNode(graph_id_2, graph_node_2, flow_root_model_2, ge_model_2);

  uint32_t stream_num_dev_avail;
  (void)rtGetAvailStreamNum(RT_NORMAL_STREAM, &stream_num_dev_avail);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model_2, ATTR_MODEL_STREAM_NUM, stream_num_dev_avail + 32));
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model_2, graph_node_2), SUCCESS);
  EXPECT_EQ(model_mgr.model_map_.size(), 1);


  EXPECT_EQ(model_executor.UnloadGraph(flow_root_model_2, graph_id_2), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  GetThreadLocalContext().SetGraphOption({});
}

/**
 * 用例描述：加载model时，设备上实际可用的stream数量小于model需要的值，已加载的model数量为1，
 *          已加载model占用的stream资源 + 设备可用的stream资源 < model需要的stream值，加载模型失败
 * 预置条件：
 * 1.流创建及流销毁的桩函数中增加对g_free_stream_num的维护，
 *   g_free_stream_num初始值为2048，流创建时g_free_stream_num减1，流销毁时g_free_stream_num加1
 * 测试步骤
 * 1.构造单个计算图1，ATTR_MODEL_STREAM_NUM设置为32，并加载该子图
 * 2.构造单个计算图2，ATTR_MODEL_STREAM_NUM设置为设备上当前可用的stream数量 + 33
 * 预期结果
 * 1.model1 unload，model2 加载失败
 */
TEST_F(DavinciModelTest, davinci_model_load_check_and_release_stream_resource_failed) {
  auto &model_mgr = ModelManager::GetInstance();
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 10086), SUCCESS); // fixed sessionid
  model_executor.StartRunThread();

  GraphId graph_id_1 = 1;
  GraphNodePtr graph_node_1;
  GeRootModelPtr flow_root_model_1;
  GeModelPtr ge_model_1;
  (void)BuildGraphNode(graph_id_1, graph_node_1, flow_root_model_1, ge_model_1);

  EXPECT_TRUE(AttrUtils::SetInt(ge_model_1, ATTR_MODEL_STREAM_NUM, 32));
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model_1, graph_node_1), SUCCESS);
  EXPECT_EQ(model_mgr.model_map_.size(), 1);
  model_mgr.model_map_.begin()->second->is_async_mode_ = true;

  GraphId graph_id_2 = 2;
  GraphNodePtr graph_node_2;
  GeRootModelPtr flow_root_model_2;
  GeModelPtr ge_model_2;
  (void)BuildGraphNode(graph_id_2, graph_node_2, flow_root_model_2, ge_model_2);

  uint32_t stream_num_dev_avail;
  (void)rtGetAvailStreamNum(RT_NORMAL_STREAM, &stream_num_dev_avail);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model_2, ATTR_MODEL_STREAM_NUM, stream_num_dev_avail + 33));
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model_2, graph_node_2), FAILED);
  EXPECT_EQ(model_mgr.model_map_.size(), 0);

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  GetThreadLocalContext().SetGraphOption({});
}

/**
 * 用例描述：加载model时，设备上实际可用的event数量小于model需要的值，已加载的model数量为1，
 *          已加载model占用的event资源 + 设备可用的event资源 = model需要的event值，加载模型成功
 * 预置条件：
 * 1.event创建及event销毁的桩函数中增加对g_free_event_num的维护，
 *   g_free_event_num初始值为2048，event创建时g_free_event_num减1，event销毁时g_free_event_num加1
 * 测试步骤
 * 1.构造单个计算图1，ATTR_MODEL_EVENT_NUM设置为32，并加载该子图
 * 2.构造单个计算图2，ATTR_MODEL_EVENT_NUM设置为设备上当前可用的event数量 + 32
 * 预期结果
 * 1.model1 unload，model2 加载成功
 */
TEST_F(DavinciModelTest, davinci_model_load_check_and_release_event_resource_success) {
  auto &model_mgr = ModelManager::GetInstance();
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 10086), SUCCESS); // fixed sessionid
  model_executor.StartRunThread();

  GraphId graph_id_1 = 1;
  GraphNodePtr graph_node_1;
  GeRootModelPtr flow_root_model_1;
  GeModelPtr ge_model_1;
  (void)BuildGraphNode(graph_id_1, graph_node_1, flow_root_model_1, ge_model_1);

  // 增加kfc event
  auto compute_graph = ge_model_1->GetGraph();
  const auto &no = compute_graph->FindNode("cond/pow");
  AttrUtils::SetListStr(no->GetOpDesc(), "_hccl_group_id_list", {"group0", "group1"});
  const auto &ni = compute_graph->FindNode("cond/sub");
  AttrUtils::SetListStr(ni->GetOpDesc(), "_hccl_group_id_list", {"group1", "group2"});
  EXPECT_EQ(HcomTopoInfo::Instance().SetGroupOrderedStream(0, "group0", (void*)1), GRAPH_SUCCESS);
  EXPECT_EQ(HcomTopoInfo::Instance().SetGroupOrderedStream(0, "group1", (void*)2), GRAPH_SUCCESS);
  EXPECT_EQ(HcomTopoInfo::Instance().SetGroupOrderedStream(0, "group2", (void*)3), GRAPH_SUCCESS);

  EXPECT_TRUE(AttrUtils::SetInt(ge_model_1, ATTR_MODEL_EVENT_NUM, 32));
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model_1, graph_node_1), SUCCESS);
  EXPECT_EQ(model_mgr.model_map_.size(), 1);
  model_mgr.model_map_.begin()->second->is_async_mode_ = true;

  GraphId graph_id_2 = 2;
  GraphNodePtr graph_node_2;
  GeRootModelPtr flow_root_model_2;
  GeModelPtr ge_model_2;
  (void)BuildGraphNode(graph_id_2, graph_node_2, flow_root_model_2, ge_model_2);

  uint32_t event_num_dev_avail;
  (void)rtGetAvailEventNum(&event_num_dev_avail);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model_2, ATTR_MODEL_EVENT_NUM, event_num_dev_avail + 32 + 2)); // 32：model event 2：kfc stream event
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model_2, graph_node_2), SUCCESS);
  EXPECT_EQ(model_mgr.model_map_.size(), 1);

  EXPECT_EQ(model_executor.UnloadGraph(flow_root_model_2, graph_id_2), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  GetThreadLocalContext().SetGraphOption({});
  HcomTopoInfo::Instance().UnsetGroupOrderedStream(0, "group0");
  HcomTopoInfo::Instance().UnsetGroupOrderedStream(0, "group1");
  HcomTopoInfo::Instance().UnsetGroupOrderedStream(0, "group2");
}

/**
 * 用例描述：加载model过程中，设备上实际可用的event数量小于model需要的值，已加载的model数量为1，
 *          已加载model占用的event资源 + 设备可用的event资源 < model需要的event值, 加载模型失败
 * 预置条件：
 * 1.event创建及event销毁的桩函数中增加对g_free_event_num的维护，
 *   g_free_event_num初始值为2048，event创建时g_free_event_num减1，event销毁时g_free_event_num加1
 * 测试步骤
 * 1.构造单个计算图1，ATTR_MODEL_EVENT_NUM设置为32，并加载该子图
 * 2.构造单个计算图2，ATTR_MODEL_EVENT_NUM设置为设备上当前可用的event数量 + 33
 * 预期结果
 * 1.model1 unload，model2 加载失败
 */
TEST_F(DavinciModelTest, davinci_model_load_check_and_release_event_resource_failed) {
  auto &model_mgr = ModelManager::GetInstance();
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 10086), SUCCESS); // fixed sessionid
  model_executor.StartRunThread();

  GraphId graph_id_1 = 1;
  GraphNodePtr graph_node_1;
  GeRootModelPtr flow_root_model_1;
  GeModelPtr ge_model_1;
  (void)BuildGraphNode(graph_id_1, graph_node_1, flow_root_model_1, ge_model_1);

  EXPECT_TRUE(AttrUtils::SetInt(ge_model_1, ATTR_MODEL_EVENT_NUM, 32));
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model_1, graph_node_1), SUCCESS);
  EXPECT_EQ(model_mgr.model_map_.size(), 1);
  model_mgr.model_map_.begin()->second->is_async_mode_ = true;

  GraphId graph_id_2 = 2;
  GraphNodePtr graph_node_2;
  GeRootModelPtr flow_root_model_2;
  GeModelPtr ge_model_2;
  (void)BuildGraphNode(graph_id_2, graph_node_2, flow_root_model_2, ge_model_2);

  uint32_t event_num_dev_avail;
  (void)rtGetAvailEventNum(&event_num_dev_avail);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model_2, ATTR_MODEL_EVENT_NUM, event_num_dev_avail + 33));
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model_2, graph_node_2), FAILED);
  EXPECT_EQ(model_mgr.model_map_.size(), 0);

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  GetThreadLocalContext().SetGraphOption({});
}

/**
 * 用例描述：加载model过程中，设备上实际可用的stream数量小于model需要的值，已加载的model数量大于1，
 *          已加载model占用的stream资源 + 设备可用的stream资源 = model需要的stream值, unload已加载model后，加载model成功
 * 预置条件：
 * 1.流创建及流销毁的桩函数中增加对g_free_stream_num的维护，
 *   g_free_stream_num初始值为2048，流创建时g_free_stream_num减1，流销毁时g_free_stream_num加1
 * 测试步骤
 * 1.构造单个计算图1，ATTR_MODEL_STREAM_NUM设置为32，并加载该子图
 * 2.构造单个计算图2，ATTR_MODEL_STREAM_NUM设置为32，并加载该子图
 * 3.构造单个计算图3，ATTR_MODEL_STREAM_NUM设置为设备上当前可用的stream数量 + 32 + 32
 *
 * 预期结果
 * 1.model1 unload，model2 unload，model3加载成功
 */
TEST_F(DavinciModelTest, davinci_model_load_check_and_release_multi_model_stream_resource_success) {
  auto &model_mgr = ModelManager::GetInstance();
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 10086), SUCCESS); // fixed sessionid
  model_executor.StartRunThread();

  GraphId graph_id_1 = 1;
  GraphNodePtr graph_node_1;
  GeRootModelPtr flow_root_model_1;
  GeModelPtr ge_model_1;
  (void)BuildGraphNode(graph_id_1, graph_node_1, flow_root_model_1, ge_model_1);

  EXPECT_TRUE(AttrUtils::SetInt(ge_model_1, ATTR_MODEL_STREAM_NUM, 32));
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model_1, graph_node_1), SUCCESS);
  EXPECT_EQ(model_mgr.model_map_.size(), 1);
  model_mgr.model_map_.begin()->second->is_async_mode_ = true;

  GraphId graph_id_2 = 2;
  GraphNodePtr graph_node_2;
  GeRootModelPtr flow_root_model_2;
  GeModelPtr ge_model_2;
  (void)BuildGraphNode(graph_id_2, graph_node_2, flow_root_model_2, ge_model_2);

  EXPECT_TRUE(AttrUtils::SetInt(ge_model_2, ATTR_MODEL_STREAM_NUM, 32));
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model_2, graph_node_2), SUCCESS);
  EXPECT_EQ(model_mgr.model_map_.size(), 2);

  GraphId graph_id_3 = 3;
  GraphNodePtr graph_node_3;
  GeRootModelPtr flow_root_model_3;
  GeModelPtr ge_model_3;
  (void)BuildGraphNode(graph_id_3, graph_node_3, flow_root_model_3, ge_model_3);

  uint32_t stream_num_dev_avail;
  (void)rtGetAvailStreamNum(RT_NORMAL_STREAM, &stream_num_dev_avail);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model_3, ATTR_MODEL_STREAM_NUM, stream_num_dev_avail + 32 + 32));
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model_3, graph_node_3), SUCCESS);
  EXPECT_EQ(model_mgr.model_map_.size(), 1);

  EXPECT_EQ(model_executor.UnloadGraph(flow_root_model_3, graph_id_3), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  GetThreadLocalContext().SetGraphOption({});
}

/**
 * 用例描述：加载model时，设备上实际可用的stream数量小于model需要的值，已加载的model数量为1，
 *          已加载model占用的tream资源 + 设备可用的stream资源 = model需要的stream值, model可以加载成功
 *          重新加载已经unload的model，可以加载成功
 * 预置条件：
 * 1.流创建及流销毁的桩函数中增加对g_free_stream_num的维护，
 *   g_free_stream_num初始值为2048，流创建时g_free_stream_num减1，流销毁时g_free_stream_num加1
 * 测试步骤
 * 1.构造单个计算图1，ATTR_MODEL_STREAM_NUM设置为32，并加载该子图
 * 2.构造单个计算图3，ATTR_MODEL_STREAM_NUM设置为设备上当前可用的stream数量 + 32，并加载该子图
 * 3.重新加载计算图1
 *
 * 预期结果
 * 1.model1 unload，model2 加载成功，model2 unload，model 1加载成功
 */
TEST_F(DavinciModelTest, davinci_model_load_check_and_release_model_stream_resource_and_reload_model_success) {
  auto &model_mgr = ModelManager::GetInstance();
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 10086), SUCCESS); // fixed sessionid
  model_executor.StartRunThread();

  GraphId graph_id_1 = 1;
  GraphNodePtr graph_node_1;
  GeRootModelPtr flow_root_model_1;
  GeModelPtr ge_model_1;
  (void)BuildGraphNode(graph_id_1, graph_node_1, flow_root_model_1, ge_model_1);

  EXPECT_TRUE(AttrUtils::SetInt(ge_model_1, ATTR_MODEL_STREAM_NUM, 32));
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model_1, graph_node_1), SUCCESS);
  EXPECT_EQ(model_mgr.model_map_.size(), 1);
  model_mgr.model_map_.begin()->second->is_async_mode_ = true;

  GraphId graph_id_2 = 2;
  GraphNodePtr graph_node_2;
  GeRootModelPtr flow_root_model_2;
  GeModelPtr ge_model_2;
  (void)BuildGraphNode(graph_id_2, graph_node_2, flow_root_model_2, ge_model_2);

  uint32_t stream_num_dev_avail;
  (void)rtGetAvailStreamNum(RT_NORMAL_STREAM, &stream_num_dev_avail);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model_2, ATTR_MODEL_STREAM_NUM, stream_num_dev_avail + 32));
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model_2, graph_node_2), SUCCESS);
  EXPECT_EQ(model_mgr.model_map_.size(), 1);

  EXPECT_EQ(model_executor.LoadGraph(flow_root_model_1, graph_node_1), SUCCESS);
  EXPECT_EQ(model_mgr.model_map_.size(), 1);

  EXPECT_EQ(model_executor.UnloadGraph(flow_root_model_1, graph_id_1), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  GetThreadLocalContext().SetGraphOption({});
}

/**
 * 用例描述：加载model时，设备上实际可用的stream数量小于model需要的值，已加载的model数量为1，
 *          已加载model占用的stream资源 + 设备可用的stream资源 = model需要的stream值,
 *          已加载的model包含hccl任务，模型卸载失败，model加载失败
 * 预置条件：
 * 1.流创建及流销毁的桩函数中增加对g_free_stream_num的维护，
 *   g_free_stream_num初始值为2048，流创建时g_free_stream_num减1，流销毁时g_free_stream_num加1
 * 测试步骤
 * 1.构造单个计算图1，ATTR_MODEL_STREAM_NUM设置为32，并加载该子图计算图1包含hccl task
 * 2.构造单个计算图2，ATTR_MODEL_STREAM_NUM设置为设备上当前可用的stream数量 + 32，并加载该子图
 *
 * 预期结果
 * 1.model1 unload 失败，model2 加载失败
 */
TEST_F(DavinciModelTest, davinci_model_load_check_and_release_model_stream_resource_which_has_hccl_task_failed) {
  auto &model_mgr = ModelManager::GetInstance();
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 10086), SUCCESS); // fixed sessionid
  model_executor.StartRunThread();

 // 构造包含hccl task的model1
  GraphId graph_id_1 = 1;
  GraphNodePtr graph_node_1;
  GeModelPtr ge_model_1;

  uint32_t mem_offset = 0;
  ComputeGraphPtr graph;
  BuildHcclSampleGraph(graph, mem_offset);
  EXPECT_NE(graph, nullptr);

  BuildGraphModel(graph, ge_model_1, mem_offset);
  EXPECT_NE(ge_model_1, nullptr);

  auto ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(graph);
  ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model_1);

  graph_node_1 = MakeShared<GraphNode>(graph_id_1);
  graph_node_1->SetGeRootModel(ge_root_model);
  graph_node_1->SetLoadFlag(true);
  graph_node_1->SetAsync(true);

  EXPECT_TRUE(AttrUtils::SetInt(ge_model_1, ATTR_MODEL_STREAM_NUM, 32));
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node_1), SUCCESS);
  EXPECT_EQ(model_mgr.model_map_.size(), 1);
  model_mgr.model_map_.begin()->second->is_async_mode_ = true;

  // 构造model2
  GraphId graph_id_2 = 2;
  GraphNodePtr graph_node_2;
  GeRootModelPtr flow_root_model_2;
  GeModelPtr ge_model_2;
  (void)BuildGraphNode(graph_id_2, graph_node_2, flow_root_model_2, ge_model_2);

  uint32_t stream_num_dev_avail;
  (void)rtGetAvailStreamNum(RT_NORMAL_STREAM, &stream_num_dev_avail);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model_2, ATTR_MODEL_STREAM_NUM, stream_num_dev_avail + 32));
  EXPECT_EQ(model_executor.LoadGraph(flow_root_model_2, graph_node_2), FAILED);
  EXPECT_EQ(model_mgr.model_map_.size(), 1);

  EXPECT_EQ(model_executor.ReleaseMemory(ge_root_model, graph_node_1), false);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id_1), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  GetThreadLocalContext().SetGraphOption({});
}

TEST_F(DavinciModelTest, davinci_model_execute_hcom_continuous_input) {
  const auto SetOpKernelForNoTiling = [](const ComputeGraph::Vistor<NodePtr> &all_nodes) {
    GeTensorDesc tensor0(GeShape({1, 1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor0, 64);
    AttrUtils::SetBool(tensor0, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor0, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 0);
    std::vector<int64_t> max_shape_list = {1, 10, 224, 224};
    AttrUtils::SetListInt(tensor0, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    GeTensorDesc tensor1(GeShape({1, 1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor1, 64);
    AttrUtils::SetBool(tensor1, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor1, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1024);
    AttrUtils::SetListInt(tensor1, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      if (op_desc->GetType() == DATA) {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateOutputDesc(0, tensor0);
        op_desc->SetOutputOffset({2048});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else if (op_desc->GetType() == HCOMREDUCESCATTER) {
        op_desc->SetOpKernelLibName("ops_kernel_info_hccl");
        op_desc->UpdateInputDesc(0, tensor0);
        op_desc->UpdateOutputDesc(0, tensor1);
        (void) AttrUtils::SetInt(op_desc, HCOM_ATTR_RANK_SIZE, 8);
        (void) AttrUtils::SetStr(op_desc, HCOM_ATTR_REDUCE_TYPE, "sum");
        (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_CONTINUOUS_INPUT, true);
        std::vector<NamedAttrs> attached_stream_infos;
        NamedAttrs attrs0;
        NamedAttrs attrs1;
        attached_stream_infos.push_back(attrs0);
        attached_stream_infos.push_back(attrs1);
        AttrUtils::SetListNamedAttrs(op_desc, ATTR_NAME_ATTACHED_STREAM_INFO_LIST, attached_stream_infos);
        op_desc->SetAttachedStreamIds({-1, 0});
        int32_t root_id = 0;
        ge::AttrUtils::SetInt(op_desc, HCOM_ATTR_ROOT_RANK, root_id);
        op_desc->SetInputOffset({2048});
        op_desc->SetOutputOffset({2048});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else if (op_desc->GetType() == HCOMALLREDUCE) {
        op_desc->SetOpKernelLibName("ops_kernel_info_hccl");
        op_desc->UpdateInputDesc(0, tensor0);
        op_desc->UpdateOutputDesc(0, tensor1);
        (void) AttrUtils::SetStr(op_desc, HCOM_ATTR_REDUCE_TYPE, "sum");
        (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_CONTINUOUS_INPUT, true);
        AttrUtils::SetBool(op_desc, "_is_unknown_shape", true);
        int32_t root_id = 0;
        ge::AttrUtils::SetInt(op_desc, HCOM_ATTR_ROOT_RANK, root_id);
        op_desc->SetInputOffset({2048});
        op_desc->SetOutputOffset({2048});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else if (op_desc->GetType() == HCOMALLGATHER) {
        op_desc->SetOpKernelLibName("ops_kernel_info_hccl");
        op_desc->UpdateInputDesc(0, tensor0);
        op_desc->UpdateInputDesc(1, tensor0);
        op_desc->UpdateOutputDesc(0, tensor1);
        (void) AttrUtils::SetStr(op_desc, HCOM_ATTR_REDUCE_TYPE, "sum");
        (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_CONTINUOUS_INPUT, true);
        int32_t root_id = 0;
        ge::AttrUtils::SetInt(op_desc, HCOM_ATTR_ROOT_RANK, root_id);
        op_desc->SetInputOffset({2048, 4096});
        op_desc->SetOutputOffset({2048});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateInputDesc(0, tensor1);
        op_desc->SetInputOffset({2112});
        op_desc->SetSrcName( { "add" } );
        op_desc->SetSrcIndex({ 0 });
      }
    }
  };

  auto hcom_allgather = OP_CFG(HCOMALLGATHER).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto hcom_reducescatter = OP_CFG(HCOMREDUCESCATTER).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto hcom_allreduce = OP_CFG(HCOMALLREDUCE).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto data1 = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto data2 = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto output = OP_CFG(NETOUTPUT).Attr(ATTR_NAME_OP_NO_TILING, true);
  DEF_GRAPH(g1) {
          CHAIN(NODE("data", data1)->EDGE(0, 0)->NODE("hcom_allgather", hcom_allgather));
          CHAIN(NODE("data", data2)->EDGE(0, 1)->NODE("hcom_allgather", hcom_allgather));
          CHAIN(NODE("hcom_allgather", hcom_allgather)->EDGE(0, 0)->NODE("hcom_allreduce", hcom_allreduce));
          CHAIN(NODE("hcom_allreduce", hcom_allreduce)->EDGE(0, 0)->NODE("hcom_reducescatter", hcom_reducescatter));
          CHAIN(NODE("hcom_reducescatter", hcom_reducescatter)->EDGE(0, 0)->NODE("output", output));
  };

  auto graph = ToComputeGraph(g1);
  SetOpKernelForNoTiling(graph->GetAllNodes());
  EXPECT_NE(graph, nullptr);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  TBEKernelStore tbe_kernel_store;

  // init hccl
  InitHcclTaskDef(graph, *model_task_def, "hcom_allgather", "HcomAllGather");
  InitHcclTaskDef(graph, *model_task_def, "hcom_reducescatter", "HcomReduceScatter");
  InitHcclTaskDef(graph, *model_task_def, "hcom_allreduce", "HcomAllReduce");
  InitEventTaskDef(graph, *model_task_def);
  InitFusionTaskDef(graph, *model_task_def);
  InitEndGraphDef(graph, *model_task_def, "output");
  InitProfilerTaskDef(graph, *model_task_def);


  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  SetGeModelAttrs(ge_model);
  EXPECT_NE(ge_model, nullptr);

  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    ge_root_model->Initialize(graph);
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);

    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
}

TEST_F(DavinciModelTest, DavinciModelExecute_LiteException_Ok) {
  const auto SetUnknownOpKernelForNoTiling = [](const ComputeGraph::Vistor<NodePtr> &all_nodes) {
    GeTensorDesc tensor0(GeShape({1, 1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor0, 64);
    AttrUtils::SetBool(tensor0, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor0, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 0);
    std::vector<int64_t> max_shape_list = {1, 10, 224, 224};
    AttrUtils::SetListInt(tensor0, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    GeTensorDesc tensor1(GeShape({1, 1, 224, 224}), FORMAT_NCHW, DT_INT64);
    TensorUtils::SetSize(tensor1, 64);
    AttrUtils::SetBool(tensor1, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor1, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1024);
    AttrUtils::SetListInt(tensor1, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);
    const std::string kOpDfxOptions = "_op_dfx_options";
    const std::string kOpDfxAssert = "assert";
    std::vector<std::string> dfx_opts{kOpDfxAssert};
    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      if (op_desc->GetType() == DATA) {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateOutputDesc(0, tensor0);
        op_desc->SetOutputOffset({2048});
        op_desc->SetWorkspace({1});
        op_desc->SetWorkspaceBytes({4});
      } else if (op_desc->GetType() == ADD) {
        ge::AttrUtils::SetListStr(op_desc, kOpDfxOptions, dfx_opts);
        op_desc->SetOpKernelLibName("AIcoreEngine");
        op_desc->UpdateInputDesc(0, tensor0);
        op_desc->UpdateOutputDesc(0, tensor1);
        op_desc->SetInputOffset({0, 2048});
        op_desc->SetOutputOffset({2112});
        op_desc->SetWorkspace({1});
        op_desc->SetWorkspaceBytes({4});
      } else {
        op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
        op_desc->UpdateInputDesc(0, tensor1);
        op_desc->SetInputOffset({2112});
        op_desc->SetSrcName( { "add" } );
        op_desc->SetSrcIndex({ 0 });
      }
    }
  };

  auto add = OP_CFG(ADD).Attr(ATTR_NAME_OP_NO_TILING, true).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  auto data = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto output = OP_CFG(NETOUTPUT).Attr(ATTR_NAME_OP_NO_TILING, true);
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", data)->EDGE(0, 0)->NODE("add", add));
                  CHAIN(NODE("add", add)->EDGE(0, 0)->NODE("output", output));
                };
  auto graph = ToComputeGraph(g1);
  SetUnknownOpKernelForNoTiling(graph->GetDirectNode());
  EXPECT_NE(graph, nullptr);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  TBEKernelStore tbe_kernel_store;
  InitKernelTaskDef_TE(graph, *model_task_def, "add", tbe_kernel_store);
  InitEventTaskDef(graph, *model_task_def);
  InitFusionTaskDef(graph, *model_task_def);
  InitEndGraphDef(graph, *model_task_def, "output");
  InitProfilerTaskDef(graph, *model_task_def);


  DumpManager::GetInstance().Init({{"ge.exec.enable_exception_dump", "2"}});
  DumpProperties dump_properties;
  dump_properties.SetDumpMode("all");
  dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
  DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  SetGeModelAttrs(ge_model);
  EXPECT_NE(ge_model, nullptr);
  DumpStub::GetInstance().Clear();
  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kLiteExceptionDump}));
  {
    // Test LoadModelOnline
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    ge_root_model->Initialize(graph);
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);

    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  GetThreadLocalContext().SetGraphOption(std::map<std::string, std::string>{{OPTION_EXEC_REUSE_ZERO_COPY_MEMORY, "1"}});
  // Serialization GeModel to memory.
  ModelHelper model_helper;
  model_helper.SetSaveMode(false);  // Save to buffer.
  ModelBufferData model_buffer;
  EXPECT_TRUE(tbe_kernel_store.Build());
  ge_model->SetTBEKernelStore(tbe_kernel_store);
  EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "file_name_prefix", model_buffer), SUCCESS);
  const ModelData model_data{model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", ""};
  // Test LoadModelWithQueueParam
  {
    ModelHelper model_helper3;
    EXPECT_EQ(model_helper3.LoadModel(model_data), SUCCESS);
    QueueAttrs inputQueue1 = {1001U, 0, 0, 0U};
    QueueAttrs outputQueue1 = {1002U, 0, 0, 0U};
    std::vector<QueueAttrs> input_queue_attrs;
    input_queue_attrs.emplace_back(inputQueue1);
    std::vector<QueueAttrs> output_queue_attrs;
    output_queue_attrs.emplace_back(outputQueue1);
    ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
    uint32_t model_id = 0;
    ModelQueueParam model_queue_param{};
    model_queue_param.group_total_count = 1;
    model_queue_param.group_index = 0;
    model_queue_param.input_queues_attrs = input_queue_attrs;
    model_queue_param.output_queues_attrs = output_queue_attrs;

    GeExecutor ge_executor;
    EXPECT_EQ(ge_executor.LoadModelWithQ(model_id, model_helper3.GetGeRootModel(), model_queue_param), SUCCESS);
    EXPECT_EQ(ge_executor.UnloadModel(model_id), SUCCESS);
    ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  }
  EXPECT_EQ(ge::DumpStub::GetInstance().GetUnits().size(), 12);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetUnits()[11][0], 12);
  EXPECT_EQ(ge::DumpStub::GetInstance().GetUnits()[11][1], 4);
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpManager::GetInstance().Finalize();
}

TEST_F(DavinciModelTest, sample_davinci_model_end_sequence) {
  uint32_t mem_offset = 0U;
  ComputeGraphPtr graph;
  BuildSampleGraph(graph, mem_offset);
  EXPECT_NE(graph, nullptr);

  GeModelPtr ge_model;
  BuildGraphModel(graph, ge_model, mem_offset);
  EXPECT_NE(ge_model, nullptr);

  std::vector<gert::Tensor> input_tensors(4);
  TensorCheckUtils::ConstructGertTensor(input_tensors[0], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(input_tensors[1], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(input_tensors[2], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(input_tensors[3], {1}, DT_INT64, FORMAT_ND);

  std::vector<uint32_t> model_ids;
  setenv(kEnvGeuseStaticMemory.c_str(), "1", 1);
  {
    // Test LoadModelOnline: RunAsyncListener
    const auto ge_root_model = MakeShared<GeRootModel>();
    ge_root_model->Initialize(graph);
    const auto graph_node = MakeShared<GraphNode>(graph->GetGraphID());
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->IncreaseLoadCount();

    // Callback for execute.
    std::mutex run_mutex;
    std::condition_variable model_run_cv;
    Status run_status = FAILED;
    const auto callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
      std::unique_lock<std::mutex> lock(run_mutex);
      run_status = status;
      model_run_cv.notify_one();
    };

    // RunArgsV2 of Graph.
    GEThreadLocalContext context;
    context.SetGraphOption(
        {{OPTION_EXEC_DYNAMIC_EXECUTE_MODE, "lazy_recompile"}, {OPTION_EXEC_ENABLE_COPY_OUTPUT_ADDR, "1"}});
    error_message::ErrorManagerContext error_context;
    graph_node->Lock();
    std::shared_ptr<RunArgs> arg;
    arg = std::make_shared<RunArgs>();
    ASSERT_TRUE(arg != nullptr);
    arg->graph_node = graph_node;
    arg->graph_id = graph->GetGraphID();
    arg->session_id = graph->GetSessionID();
    arg->error_context = error_context;
    arg->input_tensor = std::move(input_tensors);
    arg->context = context;
    arg->callback = callback;
    // Load and execute.
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.Initialize({{"ge.variableMemoryMaxSize", "12800"}}, graph->GetSessionID()), SUCCESS);
    const char_t *const kEnvRecordPath = "WITH_TIMEOUT_END_OF_SEQUENCE";
    char_t record_path[MMPA_MAX_PATH] = "end";
    mmSetEnv(kEnvRecordPath, &record_path[0U], MMPA_MAX_PATH);
    EXPECT_EQ(model_executor.PushRunArgs(arg), SUCCESS);

    // Wait for execute.
    std::unique_lock<std::mutex> lock(run_mutex);
    EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
    EXPECT_EQ(run_status, END_OF_SEQUENCE);
    unsetenv(kEnvRecordPath);
    model_ids.emplace_back(ge_root_model->GetModelId());

    // Unload model of graph
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph->GetGraphID()), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  unsetenv(kEnvGeuseStaticMemory.c_str());
}

TEST_F(DavinciModelTest, super_kernel_graph_load_and_success) {
  auto hcom_hidden_funcs = [](const ge::OpDescPtr &op_desc, std::vector<void *> &addrs) {
    addrs.push_back(reinterpret_cast<void *>(0xf1));
    return ge::GRAPH_SUCCESS;
  };
  REG_HIDDEN_INPUTS_FUNC(HiddenInputsType::HCOM, hcom_hidden_funcs);

  DEF_GRAPH(g0) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);  // sub node 1 input 1
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);  // sub node 1 input 2
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);  // sub node 1 input 3
    auto data_3 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 3);  // sub node 1 input 4
    auto data_4 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 4);  // sub node 1 input 5

    auto data_5 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 5);  // sub node 2 input 1
    auto data_6 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 6);  // sub node 2 input 2
    auto data_7 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 7);  // sub node 2 input 3

    auto node_1 = OP_CFG("node_1")
        .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
        .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
        .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    auto node_2 = OP_CFG("HcomAllReduce")
        .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
        .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
        .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");

    CHAIN(NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("node_1", node_1));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("node_1", node_1));
    CHAIN(NODE("_arg_2", data_2)->EDGE(0, 2)->NODE("node_1", node_1));
    CHAIN(NODE("_arg_3", data_3)->EDGE(0, 3)->NODE("node_1", node_1));
    CHAIN(NODE("_arg_4", data_4)->EDGE(0, 4)->NODE("node_1", node_1));
    CHAIN(NODE("_arg_5", data_5)->EDGE(0, 0)->NODE("node_2", node_2));
    CHAIN(NODE("_arg_6", data_6)->EDGE(0, 1)->NODE("node_2", node_2));
    CHAIN(NODE("_arg_7", data_6)->EDGE(0, 3)->NODE("node_2", node_2));
    CHAIN(NODE("node_1", node_1)->EDGE(0, 2)->NODE("node_2", node_2));
    CHAIN(NODE("node_2", node_2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("node_2", node_2)->EDGE(1, 1)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("node_2", node_2)->EDGE(2, 2)->NODE("Node_Output", NETOUTPUT));
  };

  auto sub_graph = ToComputeGraph(g0);
  EXPECT_NE(sub_graph, nullptr);

  {
    // 设置data的shape和offset
    for (auto i = 0; i < 8; ++i) {
      GeTensorDesc output_tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
      const auto &data = sub_graph->FindNode("_arg_" + std::to_string(i));
      EXPECT_NE(data, nullptr);
      data->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
      data->GetOpDesc()->SetOutputOffset({1000 + i * 1000});
      data->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");

      // 设置父节点属性
      AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, i);
    }

    // 设置输出的shape和offset
    const auto &out_node = sub_graph->FindNode("Node_Output");
    EXPECT_NE(out_node, nullptr);
    out_node->GetOpDesc()->SetSrcName({"node_2", "node_2", "node_3"});
    out_node->GetOpDesc()->SetSrcIndex({0, 1, 2});
    out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
    GeTensorDesc input_desc(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
    out_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
    out_node->GetOpDesc()->UpdateInputDesc(1, input_desc);
    out_node->GetOpDesc()->UpdateInputDesc(2, input_desc);
    out_node->GetOpDesc()->SetInputOffset({41000, 42000, 43000});
    AttrUtils::SetInt(out_node->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
    AttrUtils::SetInt(out_node->GetOpDesc()->MutableInputDesc(1), ATTR_NAME_PARENT_NODE_INDEX, 1);
    AttrUtils::SetInt(out_node->GetOpDesc()->MutableInputDesc(2), ATTR_NAME_PARENT_NODE_INDEX, 2);

    GeShape shape({4, 4, 4, 4});
    GeTensorDesc desc(shape);
    GeShape scalar_shape;
    GeTensorDesc scalar_desc(scalar_shape);

    // 初始化sub node1
    auto skt_sub_node_1 = sub_graph->FindNode("node_1");
    EXPECT_NE(skt_sub_node_1, nullptr);

    const auto op_desc_1 = skt_sub_node_1->GetOpDescBarePtr();
    op_desc_1->AddDynamicInputDescByIndex("x", 2, 0);
    op_desc_1->UpdateInputDesc(0, desc);
    op_desc_1->UpdateInputDesc(1, desc);
    op_desc_1->AddDynamicInputDescByIndex("y", 3, 2);
    op_desc_1->UpdateInputDesc(2, scalar_desc);
    op_desc_1->UpdateInputDesc(3, scalar_desc);
    op_desc_1->UpdateInputDesc(4, scalar_desc);
    op_desc_1->UpdateOutputDesc(0, desc);

    op_desc_1->SetInputOffset({11000, 12000, 13000, 14000, 15000});
    op_desc_1->SetOutputOffset({21000});
    op_desc_1->SetWorkspace({7000});
    op_desc_1->SetWorkspaceBytes({512});

    op_desc_1->MutableAllInputName() = {{"x0", 0}, {"x1", 1}, {"y0", 2}, {"y1", 3}, {"y2", 4}};
    op_desc_1->MutableAllOutputName() = {{"z", 0}};

    op_desc_1->AppendIrInput("x", IrInputType::kIrInputDynamic);
    op_desc_1->AppendIrInput("y", IrInputType::kIrInputDynamic);
    op_desc_1->AppendIrOutput("z", IrOutputType::kIrOutputRequired);

    op_desc_1->SetId(19);

    // 初始化subnode2
    auto skt_sub_node_2 = sub_graph->FindNode("node_2");
    EXPECT_NE(skt_sub_node_2, nullptr);

    const auto op_desc_2 = skt_sub_node_2->GetOpDescBarePtr();
    op_desc_2->AddDynamicInputDescByIndex("x", 2, 0);
    op_desc_2->UpdateInputDesc(0, desc);
    op_desc_2->UpdateInputDesc(1, desc);

    op_desc_2->SetSrcName({"node_1", "node_1"});
    op_desc_2->SetSrcIndex({0}); // 内部连边的配置么？
    op_desc_2->UpdateInputDesc(2, desc);

    op_desc_2->AddDynamicInputDescByIndex("z", 1, 3);
    op_desc_2->UpdateInputDesc(3, desc);
    op_desc_2->UpdateOutputDesc(0, desc);
    op_desc_2->UpdateOutputDesc(1, desc);
    op_desc_2->UpdateOutputDesc(2, desc);

    op_desc_2->SetInputOffset({31000, 32000, 21000, 33000});
    op_desc_2->SetOutputOffset({41000, 42000, 43000});
    op_desc_2->SetWorkspace({8000});
    op_desc_2->SetWorkspaceBytes({512});

    op_desc_2->MutableAllInputName() = {{"x0", 0}, {"x1", 1}, {"y0", 2}, {"z0", 3}};
    op_desc_2->MutableAllOutputName() = {{"j0", 0}, {"j1", 1}, {"k0", 2}};

    op_desc_2->AppendIrInput("x", IrInputType::kIrInputDynamic);
    op_desc_2->AppendIrInput("y", IrInputType::kIrInputRequired);
    op_desc_2->AppendIrInput("z", IrInputType::kIrInputDynamic);
    op_desc_2->AppendIrOutput("j", IrOutputType::kIrOutputDynamic);
    op_desc_2->AppendIrOutput("k", IrOutputType::kIrOutputDynamic);

    op_desc_2->SetId(20);
    AttrUtils::SetInt(op_desc_2, ATTR_NAME_ATTACHED_STREAM_ID, 0);
    AttrUtils::SetInt(op_desc_2, "op_para_size", 256);
    AttrUtils::SetStr(op_desc_2, "_op_aicore_num", "2");
    AttrUtils::SetStr(op_desc_2, "_op_vectorcore_num", "5");

    gert::SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
    auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto funcs = space_registry->CreateOrGetOpImpl("HcomAllReduce");
    funcs->tiling = StubTiling;
    funcs->tiling_parse = StubTilingParse;
    funcs->compile_info_creator = CompileInfoCreator;
    funcs->compile_info_deleter = nullptr;
    EXPECT_EQ(funcs->SetTilingInputDataDependency(0), GRAPH_SUCCESS);
  }

  // 构造super kernel所在的图
  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);
    auto data_3 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 3);
    auto data_4 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 4);
    auto data_5 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 5);
    auto data_6 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 6);
    auto data_7 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 6);

    auto skt_node = OP_CFG("super_kernel")
        .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
        .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
        .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");

    CHAIN(NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("super_kernel", skt_node));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("super_kernel", skt_node));
    CHAIN(NODE("_arg_2", data_2)->EDGE(0, 2)->NODE("super_kernel", skt_node));
    CHAIN(NODE("_arg_3", data_3)->EDGE(0, 3)->NODE("super_kernel", skt_node));
    CHAIN(NODE("_arg_4", data_4)->EDGE(0, 4)->NODE("super_kernel", skt_node));
    CHAIN(NODE("_arg_5", data_5)->EDGE(0, 5)->NODE("super_kernel", skt_node));
    CHAIN(NODE("_arg_6", data_6)->EDGE(0, 6)->NODE("super_kernel", skt_node));
    CHAIN(NODE("_arg_7", data_7)->EDGE(0, 7)->NODE("super_kernel", skt_node));
    CHAIN(NODE("super_kernel", skt_node)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("super_kernel", skt_node)->EDGE(1, 1)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("super_kernel", skt_node)->EDGE(2, 2)->NODE("Node_Output", NETOUTPUT));
  };

  auto root_graph = ToComputeGraph(g1);
  EXPECT_NE(root_graph, nullptr);
  {
    // 设置data的shape和offset
    for (auto i = 0; i < 8; ++i) {
      GeTensorDesc output_tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
      const auto &data = root_graph->FindNode("_arg_" + std::to_string(i));
      EXPECT_NE(data, nullptr);
      data->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
      data->GetOpDesc()->SetOutputOffset({1000 + i * 1000});
      data->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
    }

    // 设置输出的shape和offset
    const auto &out_node = root_graph->FindNode("Node_Output");
    EXPECT_NE(out_node, nullptr);
    out_node->GetOpDesc()->SetSrcName({"super_kernel", "super_kernel", "super_kernel"});
    out_node->GetOpDesc()->SetSrcIndex({0, 1, 2});
    out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
    GeTensorDesc input_desc(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
    out_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
    out_node->GetOpDesc()->UpdateInputDesc(1, input_desc);
    out_node->GetOpDesc()->UpdateInputDesc(2, input_desc);
    out_node->GetOpDesc()->SetInputOffset({41000, 42000, 43000});

    // 设置super_kernel
    auto skt_node = root_graph->FindNode("super_kernel");
    EXPECT_NE(skt_node, nullptr);

    GeShape shape({4, 4, 4, 4});
    GeTensorDesc desc(shape);
    const auto op_desc = skt_node->GetOpDescBarePtr();
    op_desc->UpdateInputDesc(0, desc);
    op_desc->UpdateInputDesc(1, desc);
    op_desc->UpdateInputDesc(2, desc);
    op_desc->UpdateInputDesc(3, desc);
    op_desc->UpdateInputDesc(4, desc);
    op_desc->UpdateInputDesc(5, desc);
    op_desc->UpdateInputDesc(6, desc);
    op_desc->UpdateInputDesc(7, desc);
    op_desc->UpdateOutputDesc(0, desc);
    op_desc->UpdateOutputDesc(1, desc);
    op_desc->UpdateOutputDesc(2, desc);

    op_desc->SetInputOffset({11000, 12000, 13000, 14000, 15000, 31000, 32000, 33000});
    op_desc->SetOutputOffset({41000, 42000, 43000});
    op_desc->SetId(19);
    int32_t run_mode = static_cast<uint32_t>(domi::ImplyType::TVM);
    EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_IMPLY_TYPE, run_mode));
    std::vector<char> kernel_bin(64, '\0');
    TBEKernelPtr kernel_handle = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
    EXPECT_TRUE(op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, kernel_handle));
    EXPECT_TRUE(AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", op_desc->GetName()));
    AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "_fake_super_kernel_bin_id");
  }

  auto super_node = root_graph->FindNode("super_kernel");
  const auto op_desc = super_node->GetOpDescBarePtr();
  op_desc->SetId(21);
  op_desc->SetWorkspace({9000});
  op_desc->SetWorkspaceBytes({512});

  // 将子图填入父节点属性
  super_node->GetOpDesc()->SetExtAttr("_sk_sub_graph", sub_graph);

  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("_aivtbeKernel_test", std::move(test_bin));
  (void)AttrUtils::SetStr(super_node->GetOpDesc(), ATTR_NAME_TBE_KERNEL_NAME, test_kernel->GetName());

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  auto &skt_task = *model_task_def->add_task();

  skt_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_SUPER_KERNEL));
  skt_task.set_stream_id(0);

  auto aicore_kernel = skt_task.mutable_kernel();
  domi::KernelContext &aicore_context = *aicore_kernel->mutable_context();
  aicore_context.set_kernel_type(static_cast<int32_t>(ge::ccKernelType::TE));
  aicore_context.set_op_id(super_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_op_index(super_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_args_format("{skn19ffts_addr}{skn19i_desc0}{skn19i_desc1}{skn19o0}{skn19ws0}{skn20ffts_addr}{skn20event_addr123*}"
                                 "{skn20i_desc0}{skn20i1}{skn20i_instance3}{skn20o_desc0}{skn20o_instance2}{skn20ws*}{skn20hi.hcom0*}"
                                 "{skn20tiling_context}{skn20tiling_context.tiling_data}{skn20*op_type}{skn20tiling_context.tiling_key}{skn20tiling_context.block_dim}{ws0}{overflow_addr}");
  aicore_context.set_args_count(14);
  uint16_t args_offset = 0;
  aicore_context.set_args_offset(&args_offset, sizeof(uint16_t));
  size_t args_size = 128UL;
  const std::vector<uint8_t> args_info(args_size, 0);
  aicore_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
  aicore_kernel->set_args_size(args_size);

  auto &event_mem_record_task = *model_task_def->add_task();
  event_mem_record_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEM_EVENT_RECORD));
  event_mem_record_task.set_stream_id(0);
  event_mem_record_task.set_event_id(1);
  auto event_ex = event_mem_record_task.mutable_event_ex();
  event_ex->set_op_index(super_node->GetOpDescBarePtr()->GetId());

  auto &event_mem_wait_task = *model_task_def->add_task();
  event_mem_wait_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEM_EVENT_WAIT));
  event_mem_wait_task.set_stream_id(0);
  event_mem_wait_task.set_event_id(1);
  auto event_wait_ex = event_mem_record_task.mutable_event_ex();
  event_wait_ex->set_op_index(super_node->GetOpDescBarePtr()->GetId());

  GeModelPtr ge_model = MakeShared<GeModel>();
  auto &kernel_store = ge_model->GetTBEKernelStore();
  kernel_store.AddTBEKernel(test_kernel);

  ge_model->SetGraph(root_graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 50480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
  EXPECT_NE(ge_model, nullptr);

  // davinci model的流程
  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
    ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, root_graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();

    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node, nullptr), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  RuntimeStub::Reset();
}

TEST_F(DavinciModelTest, ifa_aicore_with_tiling_sink_graph_load_and_success) {
  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);  // query
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);  // k0
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);  // k1
    auto data_3 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 3);  // value0
    auto data_4 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 4);  // value1
    auto data_5 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 5);  // attention_mask
    auto ifa = OP_CFG("IncreFlashAttention_T")
                   .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                   .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
                   .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
                   .Attr("_kernel_list_first_name", true);
    CHAIN(NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("ifa", ifa)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_2", data_2)->EDGE(0, 2)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_3", data_3)->EDGE(0, 3)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_4", data_4)->EDGE(0, 4)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_5", data_5)->EDGE(0, 6)->NODE("ifa", ifa));
    CHAIN(NODE("ifa", ifa)->EDGE(1, 1)->NODE("Node_Output", NETOUTPUT));
  };

  auto root_graph = ToComputeGraph(g1);
  EXPECT_NE(root_graph, nullptr);

  for (auto i = 0; i <= 5; ++i) {
    GeTensorDesc output_tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
    const auto &data = root_graph->FindNode("_arg_" + std::to_string(i));
    EXPECT_NE(data, nullptr);
    data->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
    data->GetOpDesc()->SetOutputOffset({1000 + i * 1000});
    data->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  }

  const auto &out_node = root_graph->FindNode("Node_Output");
  EXPECT_NE(out_node, nullptr);
  out_node->GetOpDesc()->SetSrcName({"ifa", "ifa"});
  out_node->GetOpDesc()->SetSrcIndex({0, 1});
  out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  GeTensorDesc input_desc(GeShape({1, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
  out_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  out_node->GetOpDesc()->UpdateInputDesc(1, input_desc);
  out_node->GetOpDesc()->SetInputOffset({10000, 11000});
  auto ifa_node = root_graph->FindNode("ifa");
  EXPECT_NE(ifa_node, nullptr);
  GeShape shape({4, 4, 4, 4});
  GeTensorDesc desc(shape);
  GeShape scalar_shape;
  GeTensorDesc scalar_desc(scalar_shape);
  const auto op_desc = ifa_node->GetOpDescBarePtr();

  op_desc->UpdateInputDesc(0, desc);
  op_desc->AddDynamicInputDescByIndex("key", 2, 1);
  op_desc->UpdateInputDesc(1, desc);
  op_desc->UpdateInputDesc(2, desc);
  op_desc->AddDynamicInputDescByIndex("value", 2, 3);
  op_desc->UpdateInputDesc(3, scalar_desc);
  op_desc->UpdateInputDesc(4, scalar_desc);
  op_desc->UpdateInputDesc("attention_mask", desc);
  AttrUtils::SetInt(op_desc, ATTR_NAME_ATTACHED_STREAM_ID, 0);

  op_desc->MutableAllInputName() = {{"query", 0},  {"k0", 1},     {"k1", 2},
                                    {"value0", 3}, {"value1", 4}, {"attention_mask", 5}};
  op_desc->MutableAllOutputName() = {{"attention_out0", 0}, {"attention_out1", 1}};

  op_desc->AppendIrInput("query", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("k", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("value", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("padding_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("attention_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("seq_lens", IrInputType::kIrInputOptional);
  op_desc->AppendIrOutput("attention_out", IrOutputType::kIrOutputDynamic);

  op_desc->SetInputOffset({1000, 2000, 3000, 4000, 5000, 6000});
  op_desc->SetOutputOffset({5000, 6000});
  op_desc->SetWorkspace({7000});
  op_desc->SetWorkspaceBytes({512});
  (void)ge::AttrUtils::SetBool(op_desc, ATTR_NAME_DYNAMIC_TILING_DEPEND_OP, true);

  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl("IncreFlashAttention_T");
  funcs->tiling = nullptr;
  funcs->tiling_parse = nullptr;
  funcs->compile_info_creator = nullptr;
  funcs->compile_info_deleter = nullptr;
  EXPECT_EQ(funcs->SetTilingInputDataDependency(5), GRAPH_SUCCESS);

  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("tbeKernel", std::move(test_bin));
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATTR_NAME_TBE_KERNEL_NAME, test_kernel->GetName());
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATTR_NAME_KERNEL_BIN_ID, "_fake_ifa_kernel_bin_id");
  op_desc->SetExtAttr(test_kernel->GetName(), test_kernel);
  op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  // aicpu kernel
  auto &aicpu_task = *model_task_def->add_task();
  aicpu_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  auto aicpu_kernel = aicpu_task.mutable_kernel();
  domi::KernelContext &aicpu_context = *aicpu_kernel->mutable_context();
  aicpu_context.set_kernel_type(static_cast<int32_t>(ccKernelType::AI_CPU_KFC));
  aicpu_context.set_op_id(ifa_node->GetOpDesc()->GetId());
  aicpu_context.set_op_index(ifa_node->GetOpDesc()->GetId());
  aicpu_context.set_args_format("{tiling_context}{*op_type}{tiling_context.block_dim}{tiling_context.tiling_key}");
  aicpu_context.set_args_count(10);
  aicpu_kernel->set_so_name("libmc2_aicpu.so");
  aicpu_kernel->set_kernel_name("mc2_aicpu");
  size_t aicpu_args_size = 128UL;
  const std::vector<uint8_t> args_info(aicpu_args_size, 0);
  aicpu_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
  aicpu_kernel->set_args_size(aicpu_args_size);
  // aicore_kenrel
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  task_def.set_stream_id(0);
  auto aicore_kernel = task_def.mutable_kernel_with_handle();
  domi::KernelContext &aicore_context = *aicore_kernel->mutable_context();
  aicore_context.set_kernel_type(static_cast<int32_t>(ccKernelType::TE));
  aicore_context.set_op_id(ifa_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_op_index(ifa_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_args_format("{i0}{i_desc1}{i_desc2}{i4}{o_desc0}{ws0}{ws*}{tiling_context.tiling_data}");
  aicore_context.set_args_count(9);
  aicore_kernel->set_args_size(256);
  string args(256, '1');
  aicore_kernel->set_args(args.data(), args.size());
  uint16_t args_offset[9] = {0};
  aicore_context.set_args_offset(args_offset, 9 * sizeof(uint16_t));

  GeModelPtr ge_model = MakeShared<GeModel>();
  auto &kernel_store = ge_model->GetTBEKernelStore();
  kernel_store.AddTBEKernel(test_kernel);
  // update_pc_kenrel
  auto &rts_task_def = *model_task_def->add_task();
  rts_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_UPDATE));
  rts_task_def.mutable_update_pc_task()->set_stream_id(1);
  rts_task_def.mutable_update_pc_task()->set_op_index(ifa_node->GetOpDescBarePtr()->GetId());
  // nop_kernel
  auto &rts_nop_task_def = *model_task_def->add_task();
  rts_nop_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOP));
  rts_nop_task_def.set_stream_id(0);

  ge_model->SetGraph(root_graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 2));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
  EXPECT_NE(ge_model, nullptr);
  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
    ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, root_graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node, nullptr), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  RuntimeStub::Reset();
}

TEST_F(DavinciModelTest, ifa_aicore_with_tiling_sink_graph_load_and_launch_cust_platform_from_op_master_success) {
  constexpr const char_t *kAscendHomePath = "ASCEND_HOME_PATH";
  char old_path[MMPA_MAX_PATH] = {0};
  (void)mmGetEnv(kAscendHomePath, old_path, MMPA_MAX_PATH);
  std::string ascend_home_path("/test/ascend_path");
  mmSetEnv(kAscendHomePath, ascend_home_path.c_str(), 1);

  std::string opp_path = __FILE__;
  std::vector<std::pair<ccKernelType, const std::string>> kernel_type_so_names;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "/test_tmp/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  ConstructOpMasterDeviceSo(opp_path, 0, 1, true, kernel_type_so_names);

  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);  // query
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);  // k0
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);  // k1
    auto data_3 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 3);  // value0
    auto data_4 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 4);  // value1
    auto data_5 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 5);  // attention_mask
    auto ifa = OP_CFG("IncreFlashAttention_T")
                   .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                   .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
                   .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
                   .Attr("_kernel_list_first_name", true);
    CHAIN(NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("ifa", ifa)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_2", data_2)->EDGE(0, 2)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_3", data_3)->EDGE(0, 3)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_4", data_4)->EDGE(0, 4)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_5", data_5)->EDGE(0, 6)->NODE("ifa", ifa));
    CHAIN(NODE("ifa", ifa)->EDGE(1, 1)->NODE("Node_Output", NETOUTPUT));
  };

  auto root_graph = ToComputeGraph(g1);
  EXPECT_NE(root_graph, nullptr);

  for (auto i = 0; i <= 5; ++i) {
    GeTensorDesc output_tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
    const auto &data = root_graph->FindNode("_arg_" + std::to_string(i));
    EXPECT_NE(data, nullptr);
    data->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
    data->GetOpDesc()->SetOutputOffset({1000 + i * 1000});
    data->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  }

  const auto &out_node = root_graph->FindNode("Node_Output");
  EXPECT_NE(out_node, nullptr);
  out_node->GetOpDesc()->SetSrcName({"ifa", "ifa"});
  out_node->GetOpDesc()->SetSrcIndex({0, 1});
  out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  GeTensorDesc input_desc(GeShape({1, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
  out_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  out_node->GetOpDesc()->UpdateInputDesc(1, input_desc);
  out_node->GetOpDesc()->SetInputOffset({10000, 11000});
  auto ifa_node = root_graph->FindNode("ifa");
  EXPECT_NE(ifa_node, nullptr);
  GeShape shape({4, 4, 4, 4});
  GeTensorDesc desc(shape);
  GeShape scalar_shape;
  GeTensorDesc scalar_desc(scalar_shape);
  const auto op_desc = ifa_node->GetOpDescBarePtr();

  op_desc->UpdateInputDesc(0, desc);
  op_desc->AddDynamicInputDescByIndex("key", 2, 1);
  op_desc->UpdateInputDesc(1, desc);
  op_desc->UpdateInputDesc(2, desc);
  op_desc->AddDynamicInputDescByIndex("value", 2, 3);
  op_desc->UpdateInputDesc(3, scalar_desc);
  op_desc->UpdateInputDesc(4, scalar_desc);
  op_desc->UpdateInputDesc("attention_mask", desc);
  AttrUtils::SetInt(op_desc, ATTR_NAME_ATTACHED_STREAM_ID, 0);

  op_desc->MutableAllInputName() = {{"query", 0},  {"k0", 1},     {"k1", 2},
                                    {"value0", 3}, {"value1", 4}, {"attention_mask", 5}};
  op_desc->MutableAllOutputName() = {{"attention_out0", 0}, {"attention_out1", 1}};

  op_desc->AppendIrInput("query", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("k", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("value", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("padding_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("attention_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("seq_lens", IrInputType::kIrInputOptional);
  op_desc->AppendIrOutput("attention_out", IrOutputType::kIrOutputDynamic);

  op_desc->SetInputOffset({1000, 2000, 3000, 4000, 5000, 6000});
  op_desc->SetOutputOffset({5000, 6000});
  op_desc->SetWorkspace({7000});
  op_desc->SetWorkspaceBytes({512});
  (void)ge::AttrUtils::SetBool(op_desc, ATTR_NAME_DYNAMIC_TILING_DEPEND_OP, true);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry(true);
  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl("IncreFlashAttention_T");
  funcs->tiling = nullptr;
  funcs->tiling_parse = nullptr;
  funcs->compile_info_creator = nullptr;
  funcs->compile_info_deleter = nullptr;
  EXPECT_EQ(funcs->SetTilingInputDataDependency(5), GRAPH_SUCCESS);

  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("tbeKernel", std::move(test_bin));
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATTR_NAME_TBE_KERNEL_NAME, test_kernel->GetName());
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATTR_NAME_KERNEL_BIN_ID, "_fake_ifa_kernel_bin_id");
  op_desc->SetExtAttr(test_kernel->GetName(), test_kernel);
  op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  // aicpu kernel
  auto &aicpu_task = *model_task_def->add_task();
  aicpu_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL));
  auto aicpu_kernel = aicpu_task.mutable_kernel();
  domi::KernelContext &aicpu_context = *aicpu_kernel->mutable_context();
  aicpu_context.set_kernel_type(static_cast<int32_t>(kernel_type_so_names[0].first));
  aicpu_context.set_op_id(ifa_node->GetOpDesc()->GetId());
  aicpu_context.set_op_index(ifa_node->GetOpDesc()->GetId());
  aicpu_context.set_args_format("{tiling_context}{*op_type}{tiling_context.block_dim}{tiling_context.tiling_key}");
  aicpu_context.set_args_count(10);
  aicpu_kernel->set_so_name(kernel_type_so_names[0].second);
  aicpu_kernel->set_kernel_name("mc2_aicpu");
  size_t aicpu_args_size = 128UL;
  const std::vector<uint8_t> args_info(aicpu_args_size, 0);
  aicpu_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
  aicpu_kernel->set_args_size(aicpu_args_size);
  // aicore_kenrel
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  task_def.set_stream_id(0);
  auto aicore_kernel = task_def.mutable_kernel_with_handle();
  domi::KernelContext &aicore_context = *aicore_kernel->mutable_context();
  aicore_context.set_kernel_type(static_cast<int32_t>(ccKernelType::TE));
  aicore_context.set_op_id(ifa_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_op_index(ifa_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_args_format("{i0}{i_desc1}{i_desc2}{i4}{o_desc0}{ws0}{ws*}{tiling_context.tiling_data}{event_addr123*}");
  aicore_context.set_args_count(10);
  aicore_kernel->set_args_size(264);
  string args(256, '1');
  aicore_kernel->set_args(args.data(), args.size());
  uint16_t args_offset[9] = {0};
  aicore_context.set_args_offset(args_offset, 9 * sizeof(uint16_t));

  GeModelPtr ge_model = MakeShared<GeModel>();
  auto &kernel_store = ge_model->GetTBEKernelStore();
  kernel_store.AddTBEKernel(test_kernel);
  // update_pc_kenrel
  auto &rts_task_def = *model_task_def->add_task();
  rts_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_UPDATE));
  rts_task_def.mutable_update_pc_task()->set_stream_id(1);
  rts_task_def.mutable_update_pc_task()->set_op_index(ifa_node->GetOpDescBarePtr()->GetId());
  // nop_kernel
  auto &rts_nop_task_def = *model_task_def->add_task();
  rts_nop_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOP));
  rts_nop_task_def.set_stream_id(0);

  ge_model->SetGraph(root_graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 2));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
  EXPECT_NE(ge_model, nullptr);
  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
    ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, root_graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node, nullptr), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  RuntimeStub::Reset();
  unsetenv(kEnvName);
  system(("rm -rf " + opp_path).c_str());
  MemManager::Instance().MemInstance(RT_MEMORY_HBM).ReleaseResource();

  mmSetEnv(kAscendHomePath, old_path, 1);
}

TEST_F(DavinciModelTest, ifa_aicore_with_tiling_sink_graph_load_and_launch_cust_platform_from_platform_so_success) {
  constexpr const char_t *kAscendHomePath = "ASCEND_HOME_PATH";
  char old_path[MMPA_MAX_PATH] = {0};
  (void)mmGetEnv(kAscendHomePath, old_path, MMPA_MAX_PATH);
  auto work_path = EnvPath().GetAirBasePath();
  mmSetEnv(kAscendHomePath, (work_path + "/build_st/tests/depends/op_stub/so_stub").c_str(), 1);

  std::string opp_path = __FILE__;
  std::vector<std::pair<ccKernelType, const std::string>> kernel_type_so_names;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "/test_tmp/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  ConstructOpMasterDeviceSo(opp_path, 0, 1, true, kernel_type_so_names);

  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);  // query
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);  // k0
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);  // k1
    auto data_3 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 3);  // value0
    auto data_4 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 4);  // value1
    auto data_5 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 5);  // attention_mask
    auto ifa = OP_CFG("IncreFlashAttention_T")
                   .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                   .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
                   .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
                   .Attr("_kernel_list_first_name", true);
    CHAIN(NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("ifa", ifa)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_2", data_2)->EDGE(0, 2)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_3", data_3)->EDGE(0, 3)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_4", data_4)->EDGE(0, 4)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_5", data_5)->EDGE(0, 6)->NODE("ifa", ifa));
    CHAIN(NODE("ifa", ifa)->EDGE(1, 1)->NODE("Node_Output", NETOUTPUT));
  };

  auto root_graph = ToComputeGraph(g1);
  EXPECT_NE(root_graph, nullptr);

  for (auto i = 0; i <= 5; ++i) {
    GeTensorDesc output_tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
    const auto &data = root_graph->FindNode("_arg_" + std::to_string(i));
    EXPECT_NE(data, nullptr);
    data->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
    data->GetOpDesc()->SetOutputOffset({1000 + i * 1000});
    data->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  }

  const auto &out_node = root_graph->FindNode("Node_Output");
  EXPECT_NE(out_node, nullptr);
  out_node->GetOpDesc()->SetSrcName({"ifa", "ifa"});
  out_node->GetOpDesc()->SetSrcIndex({0, 1});
  out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  GeTensorDesc input_desc(GeShape({1, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
  out_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  out_node->GetOpDesc()->UpdateInputDesc(1, input_desc);
  out_node->GetOpDesc()->SetInputOffset({10000, 11000});
  auto ifa_node = root_graph->FindNode("ifa");
  EXPECT_NE(ifa_node, nullptr);
  GeShape shape({4, 4, 4, 4});
  GeTensorDesc desc(shape);
  GeShape scalar_shape;
  GeTensorDesc scalar_desc(scalar_shape);
  const auto op_desc = ifa_node->GetOpDescBarePtr();

  op_desc->UpdateInputDesc(0, desc);
  op_desc->AddDynamicInputDescByIndex("key", 2, 1);
  op_desc->UpdateInputDesc(1, desc);
  op_desc->UpdateInputDesc(2, desc);
  op_desc->AddDynamicInputDescByIndex("value", 2, 3);
  op_desc->UpdateInputDesc(3, scalar_desc);
  op_desc->UpdateInputDesc(4, scalar_desc);
  op_desc->UpdateInputDesc("attention_mask", desc);
  AttrUtils::SetInt(op_desc, ATTR_NAME_ATTACHED_STREAM_ID, 0);

  op_desc->MutableAllInputName() = {{"query", 0},  {"k0", 1},     {"k1", 2},
                                    {"value0", 3}, {"value1", 4}, {"attention_mask", 5}};
  op_desc->MutableAllOutputName() = {{"attention_out0", 0}, {"attention_out1", 1}};

  op_desc->AppendIrInput("query", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("k", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("value", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("padding_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("attention_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("seq_lens", IrInputType::kIrInputOptional);
  op_desc->AppendIrOutput("attention_out", IrOutputType::kIrOutputDynamic);

  op_desc->SetInputOffset({1000, 2000, 3000, 4000, 5000, 6000});
  op_desc->SetOutputOffset({5000, 6000});
  op_desc->SetWorkspace({7000});
  op_desc->SetWorkspaceBytes({512});
  (void)ge::AttrUtils::SetBool(op_desc, ATTR_NAME_DYNAMIC_TILING_DEPEND_OP, true);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry(true);
  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl("IncreFlashAttention_T");
  funcs->tiling = nullptr;
  funcs->tiling_parse = nullptr;
  funcs->compile_info_creator = nullptr;
  funcs->compile_info_deleter = nullptr;
  EXPECT_EQ(funcs->SetTilingInputDataDependency(5), GRAPH_SUCCESS);

  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("tbeKernel", std::move(test_bin));
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATTR_NAME_TBE_KERNEL_NAME, test_kernel->GetName());
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATTR_NAME_KERNEL_BIN_ID, "_fake_ifa_kernel_bin_id");
  op_desc->SetExtAttr(test_kernel->GetName(), test_kernel);
  op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  // aicpu kernel
  auto &aicpu_task = *model_task_def->add_task();
  aicpu_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL));
  auto aicpu_kernel = aicpu_task.mutable_kernel();
  domi::KernelContext &aicpu_context = *aicpu_kernel->mutable_context();
  aicpu_context.set_kernel_type(static_cast<int32_t>(kernel_type_so_names[0].first));
  aicpu_context.set_op_id(ifa_node->GetOpDesc()->GetId());
  aicpu_context.set_op_index(ifa_node->GetOpDesc()->GetId());
  aicpu_context.set_args_format("{tiling_context}{*op_type}{tiling_context.block_dim}{tiling_context.tiling_key}");
  aicpu_context.set_args_count(10);
  aicpu_kernel->set_so_name(kernel_type_so_names[0].second);
  aicpu_kernel->set_kernel_name("mc2_aicpu");
  size_t aicpu_args_size = 128UL;
  const std::vector<uint8_t> args_info(aicpu_args_size, 0);
  aicpu_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
  aicpu_kernel->set_args_size(aicpu_args_size);
  // aicore_kenrel
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  task_def.set_stream_id(0);
  auto aicore_kernel = task_def.mutable_kernel_with_handle();
  domi::KernelContext &aicore_context = *aicore_kernel->mutable_context();
  aicore_context.set_kernel_type(static_cast<int32_t>(ccKernelType::TE));
  aicore_context.set_op_id(ifa_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_op_index(ifa_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_args_format("{i0}{i_desc1}{i_desc2}{i4}{o_desc0}{ws0}{ws*}{tiling_context.tiling_data}{event_addr123*}");
  aicore_context.set_args_count(10);
  aicore_kernel->set_args_size(264);
  string args(256, '1');
  aicore_kernel->set_args(args.data(), args.size());
  uint16_t args_offset[9] = {0};
  aicore_context.set_args_offset(args_offset, 9 * sizeof(uint16_t));

  GeModelPtr ge_model = MakeShared<GeModel>();
  auto &kernel_store = ge_model->GetTBEKernelStore();
  kernel_store.AddTBEKernel(test_kernel);
  // update_pc_kenrel
  auto &rts_task_def = *model_task_def->add_task();
  rts_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_UPDATE));
  rts_task_def.mutable_update_pc_task()->set_stream_id(1);
  rts_task_def.mutable_update_pc_task()->set_op_index(ifa_node->GetOpDescBarePtr()->GetId());
  // nop_kernel
  auto &rts_nop_task_def = *model_task_def->add_task();
  rts_nop_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOP));
  rts_nop_task_def.set_stream_id(0);

  ge_model->SetGraph(root_graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 2));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
  EXPECT_NE(ge_model, nullptr);
  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
    ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, root_graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node, nullptr), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  RuntimeStub::Reset();
  unsetenv(kEnvName);
  system(("rm -rf " + opp_path).c_str());
  MemManager::Instance().MemInstance(RT_MEMORY_HBM).ReleaseResource();

  mmSetEnv(kAscendHomePath, old_path, 1);
}

TEST_F(DavinciModelTest, ifa_aicore_with_tiling_sink_graph_load_and_success_with_instance_format) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry();
  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);  // query
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);  // k0
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);  // k1
    auto data_3 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 3);  // value0
    auto data_4 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 4);  // value1
    auto data_5 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 5);  // attention_mask
    auto ifa = OP_CFG("IncreFlashAttention_T")
                   .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                   .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
                   .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
                   .Attr(ATTR_NAME_KERNEL_BIN_ID, "_ifa_fake_id")
                   .Attr("_kernel_list_first_name", true);
    CHAIN(NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("ifa", ifa)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_2", data_2)->EDGE(0, 2)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_3", data_3)->EDGE(0, 3)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_4", data_4)->EDGE(0, 4)->NODE("ifa", ifa));
    CHAIN(NODE("_arg_5", data_5)->EDGE(0, 6)->NODE("ifa", ifa));
    CHAIN(NODE("ifa", ifa)->EDGE(1, 1)->NODE("Node_Output", NETOUTPUT));
  };

  auto root_graph = ToComputeGraph(g1);
  EXPECT_NE(root_graph, nullptr);

  for (auto i = 0; i <= 5; ++i) {
    GeTensorDesc output_tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
    const auto &data = root_graph->FindNode("_arg_" + std::to_string(i));
    EXPECT_NE(data, nullptr);
    data->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
    data->GetOpDesc()->SetOutputOffset({1000 + i * 1000});
    data->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  }

  const auto &out_node = root_graph->FindNode("Node_Output");
  EXPECT_NE(out_node, nullptr);
  out_node->GetOpDesc()->SetSrcName({"ifa", "ifa"});
  out_node->GetOpDesc()->SetSrcIndex({0, 1});
  out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  GeTensorDesc input_desc(GeShape({1, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
  out_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  out_node->GetOpDesc()->UpdateInputDesc(1, input_desc);
  out_node->GetOpDesc()->SetInputOffset({10000, 11000});
  auto ifa_node = root_graph->FindNode("ifa");
  EXPECT_NE(ifa_node, nullptr);
  GeShape shape({4, 4, 4, 4});
  GeTensorDesc desc(shape);
  GeShape scalar_shape;
  GeTensorDesc scalar_desc(scalar_shape);
  const auto op_desc = ifa_node->GetOpDescBarePtr();

  op_desc->UpdateInputDesc(0, desc);
  op_desc->AddDynamicInputDescByIndex("key", 2, 1);
  op_desc->UpdateInputDesc(1, desc);
  op_desc->UpdateInputDesc(2, desc);
  op_desc->AddDynamicInputDescByIndex("value", 2, 3);
  op_desc->UpdateInputDesc(3, scalar_desc);
  op_desc->UpdateInputDesc(4, scalar_desc);
  op_desc->UpdateInputDesc("attention_mask", desc);
  AttrUtils::SetInt(op_desc, ATTR_NAME_ATTACHED_STREAM_ID, 0);

  op_desc->MutableAllInputName() = {{"query", 0},  {"k0", 1},     {"k1", 2},
                                    {"value0", 3}, {"value1", 4}, {"attention_mask", 5}};
  op_desc->MutableAllOutputName() = {{"attention_out0", 0}, {"attention_out1", 1}};

  op_desc->AppendIrInput("query", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("k", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("value", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("padding_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("attention_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("seq_lens", IrInputType::kIrInputOptional);
  op_desc->AppendIrOutput("attention_out", IrOutputType::kIrOutputDynamic);

  op_desc->SetInputOffset({1000, 2000, 3000, 4000, 5000, 6000});
  op_desc->SetOutputOffset({5000, 6000});
  op_desc->SetWorkspace({7000});
  op_desc->SetWorkspaceBytes({512});
  (void)ge::AttrUtils::SetBool(op_desc, ATTR_NAME_DYNAMIC_TILING_DEPEND_OP, true);

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry(true);
  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl("IncreFlashAttention_T");
  funcs->tiling = nullptr;
  funcs->tiling_parse = nullptr;
  funcs->compile_info_creator = nullptr;
  funcs->compile_info_deleter = nullptr;
  EXPECT_EQ(funcs->SetTilingInputDataDependency(5), GRAPH_SUCCESS);

  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("tbeKernel", std::move(test_bin));
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATTR_NAME_TBE_KERNEL_NAME, test_kernel->GetName());
  (void)AttrUtils::SetStr(ifa_node->GetOpDesc(), ATTR_NAME_KERNEL_BIN_ID, "_ifa_fake_id");
  op_desc->SetExtAttr(test_kernel->GetName(), test_kernel);
  op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  // aicpu kernel
  auto &aicpu_task = *model_task_def->add_task();
  aicpu_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  auto aicpu_kernel = aicpu_task.mutable_kernel();
  domi::KernelContext &aicpu_context = *aicpu_kernel->mutable_context();
  aicpu_context.set_kernel_type(static_cast<int32_t>(ccKernelType::AI_CPU_KFC));
  aicpu_context.set_op_id(ifa_node->GetOpDesc()->GetId());
  aicpu_context.set_op_index(ifa_node->GetOpDesc()->GetId());
  aicpu_context.set_args_format("{tiling_context}{*op_type}{tiling_context.block_dim}{tiling_context.tiling_key}");
  aicpu_context.set_args_count(10);
  aicpu_kernel->set_so_name("libmc2_aicpu.so");
  aicpu_kernel->set_kernel_name("mc2_aicpu");
  size_t aicpu_args_size = 128UL;
  const std::vector<uint8_t> args_info(aicpu_args_size, 0);
  aicpu_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
  aicpu_kernel->set_args_size(aicpu_args_size);
  // aicore_kenrel
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  task_def.set_stream_id(0);
  auto aicore_kernel = task_def.mutable_kernel_with_handle();
  domi::KernelContext &aicore_context = *aicore_kernel->mutable_context();
  aicore_context.set_kernel_type(static_cast<int32_t>(ccKernelType::TE));
  aicore_context.set_op_id(ifa_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_op_index(ifa_node->GetOpDescBarePtr()->GetId());
  aicore_context.set_args_format("{i_instance0}{i_instance1}{i_instance2}{i_desc2}{}{o_instance0}{o_instance1}{ws0}{tiling_context.tiling_data}");
  aicore_context.set_args_count(9);
  aicore_kernel->set_args_size(256);
  string args(256, '1');
  aicore_kernel->set_args(args.data(), args.size());
  uint16_t args_offset[9] = {0};
  aicore_context.set_args_offset(args_offset, 9 * sizeof(uint16_t));

  GeModelPtr ge_model = MakeShared<GeModel>();
  auto &kernel_store = ge_model->GetTBEKernelStore();
  kernel_store.AddTBEKernel(test_kernel);
  // update_pc_kenrel
  auto &rts_task_def = *model_task_def->add_task();
  rts_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_UPDATE));
  rts_task_def.mutable_update_pc_task()->set_stream_id(1);
  rts_task_def.mutable_update_pc_task()->set_op_index(ifa_node->GetOpDescBarePtr()->GetId());
  // nop_kernel
  auto &rts_nop_task_def = *model_task_def->add_task();
  rts_nop_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOP));
  rts_nop_task_def.set_stream_id(0);

  ge_model->SetGraph(root_graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 2));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
  EXPECT_NE(ge_model, nullptr);
  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
    ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, root_graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node, nullptr), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  RuntimeStub::Reset();
}


TEST_F(DavinciModelTest, sample_davinci_model_execute_fail) {
  uint32_t mem_offset = 0U;
  ComputeGraphPtr graph;
  BuildSampleGraph(graph, mem_offset);
  EXPECT_NE(graph, nullptr);

  GeModelPtr ge_model;
  BuildGraphModel(graph, ge_model, mem_offset);
  EXPECT_NE(ge_model, nullptr);

  std::vector<gert::Tensor> input_tensors(4);
  TensorCheckUtils::ConstructGertTensor(input_tensors[0], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(input_tensors[1], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(input_tensors[2], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(input_tensors[3], {1}, DT_INT64, FORMAT_ND);

  std::vector<uint32_t> model_ids;
  setenv(kEnvGeuseStaticMemory.c_str(), "1", 1);
  {
    // Test LoadModelOnline: RunAsyncListener
    const auto ge_root_model = MakeShared<GeRootModel>();
    ge_root_model->Initialize(graph);
    const auto graph_node = MakeShared<GraphNode>(graph->GetGraphID());
    ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->IncreaseLoadCount();

    // Callback for execute.
    std::mutex run_mutex;
    std::condition_variable model_run_cv;
    Status run_status = FAILED;
    const auto callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
      std::unique_lock<std::mutex> lock(run_mutex);
      run_status = status;
      model_run_cv.notify_one();
    };

    // RunArgsV2 of Graph.
    GEThreadLocalContext context;
    context.SetGraphOption(
        {{OPTION_EXEC_DYNAMIC_EXECUTE_MODE, "lazy_recompile"}, {OPTION_EXEC_ENABLE_COPY_OUTPUT_ADDR, "1"}});
    error_message::ErrorManagerContext error_context;
    graph_node->Lock();
    std::shared_ptr<RunArgs> arg;
    arg = std::make_shared<RunArgs>();
    ASSERT_TRUE(arg != nullptr);
    arg->graph_node = graph_node;
    arg->graph_id = graph->GetGraphID();
    arg->session_id = graph->GetSessionID();
    arg->error_context = error_context;
    arg->input_tensor = std::move(input_tensors);
    arg->context = context;
    arg->callback = callback;
    // Load and execute.
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.Initialize({{"ge.variableMemoryMaxSize", "12800"}}, graph->GetSessionID()), SUCCESS);
    const char_t *const kEnvRecordPath = "CONSTANT_FOLDING_PASS_8";
    char_t record_path[MMPA_MAX_PATH] = "mock_fail";
    mmSetEnv(kEnvRecordPath, &record_path[0U], MMPA_MAX_PATH);
    EXPECT_EQ(model_executor.PushRunArgs(arg), SUCCESS);

    // Wait for execute.
    std::unique_lock<std::mutex> lock(run_mutex);
    EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
    EXPECT_EQ(run_status, INTERNAL_ERROR);
    unsetenv(kEnvRecordPath);
    model_ids.emplace_back(ge_root_model->GetModelId());

    // Unload model of graph
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph->GetGraphID()), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  unsetenv(kEnvGeuseStaticMemory.c_str());
}

TEST_F(DavinciModelTest, init_space_registry_with_upgraded_so) {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph;
  BuildAddGraph(graph, "file_constant_1", false);
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(graph);

  std::vector<OpSoBinPtr> kernels;
  std::string so_name("libopsproto_rt.so");
  std::string vendor_name("/opp_latest/built-in/op_proto/lib");
  auto so_bin = std::unique_ptr<char[]>(new (std::nothrow) char[so_name.length()]);
  (void) memcpy_s(so_bin.get(), so_name.length(), so_name.data(), so_name.length());
  auto op_so_bin = std::make_shared<OpSoBin>(so_name, vendor_name, std::move(so_bin), so_name.length());
  kernels.emplace_back(op_so_bin);

  std::string cust_so_name("libcust_op_master.so");
  std::string cust_vendor_name("/custom/op_impl/ai_core/tbe/op_tiling/lib");
  auto cust_so_bin = std::unique_ptr<char[]>(new (std::nothrow) char[cust_so_name.length()]);
  (void)memcpy_s(cust_so_bin.get(), cust_so_name.length(), cust_so_name.data(), cust_so_name.length());
  auto cust_op_so_bin = std::make_shared<OpSoBin>(cust_so_name, cust_vendor_name,
                                                  std::move(cust_so_bin), cust_so_name.length());
  kernels.emplace_back(cust_op_so_bin);

  ge_root_model->op_so_store_.kernels_ = std::move(kernels);
  (void) model.InitSpaceRegistry(ge_root_model);
  // EXPECT_NE(model.InitSpaceRegistry(ge_root_model), SUCCESS);
  EXPECT_NE(model.GetSpaceRegistries(), nullptr);
}

TEST_F(DavinciModelTest, CalculateUpdateModelParamTiling_test) {
  DavinciModel model(0, nullptr);

  ModelArgsManager mam(&model);
  UpdateModelParamTilingData tiling{0};
  uint32_t active_base_len = 512 * 8;
  uint32_t index_len = 32 * 1024;
  uint32_t block_dim{0};
  mam.CalculateUpdateModelParamTiling(active_base_len, index_len, block_dim, tiling);
  EXPECT_EQ(tiling.totalActiveBaseTblCnt, 1024);
  EXPECT_EQ(tiling.blockCnt, 2048);
  EXPECT_EQ(tiling.tileCnt, 2048);
  EXPECT_EQ(tiling.tailCnt, 2048);
  EXPECT_EQ(tiling.lastTailCnt, 2048);
  EXPECT_EQ(tiling.tileNum, 1);
  EXPECT_EQ(tiling.lastTileNum, 1);
  EXPECT_EQ(block_dim, 4);
  index_len = 128 * 1024;
  mam.CalculateUpdateModelParamTiling(active_base_len, index_len, block_dim,tiling);
  EXPECT_EQ(tiling.totalActiveBaseTblCnt, 1024);
  EXPECT_EQ(tiling.blockCnt, 2368);
  EXPECT_EQ(tiling.tileCnt, 2368);
  EXPECT_EQ(tiling.tailCnt, 2368);
  EXPECT_EQ(tiling.lastTailCnt, 1984);
  EXPECT_EQ(tiling.tileNum, 1);
  EXPECT_EQ(tiling.lastTileNum, 1);
  EXPECT_EQ(block_dim, 14);
};

TEST_F(DavinciModelTest, TilingSink_From_OppPackage_Success) {
  std::string opp_path = __FILE__;
  std::vector<std::pair<ccKernelType, const std::string>> kernel_type_so_names;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "/test_tmp/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  ConstructOpMasterDeviceSo(opp_path, 1, 2, true, kernel_type_so_names);

  GeModelPtr  ge_model = nullptr;
  ComputeGraphPtr root_graph = nullptr;
  ConstructTilingSinkGeModel(kernel_type_so_names, ge_model, root_graph);
  EXPECT_NE(ge_model, nullptr);
  EXPECT_NE(root_graph, nullptr);
  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
    ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);
    EXPECT_EQ(ge_root_model->CheckAndSetNeedSoInOM(), SUCCESS);
    EXPECT_EQ(ge_root_model->GetSoInOmFlag(), 0x4000);

    auto &model_mgr = ModelManager::GetInstance();
    model_mgr.builtin_aicpu_so_.clear();
    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, root_graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node, nullptr), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
    EXPECT_EQ(model_mgr.builtin_aicpu_so_.size(), 1UL);
  }
  RuntimeStub::Reset();
  unsetenv(kEnvName);
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(DavinciModelTest, TilingSink_From_Model_Success) {
  std::string opp_path = __FILE__;
  std::vector<std::pair<ccKernelType, const std::string>> kernel_type_so_names;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "/test_tmp/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  ConstructOpMasterDeviceSo(opp_path, 1, 2, true, kernel_type_so_names);

  GeModelPtr  ge_model = nullptr;
  ComputeGraphPtr root_graph = nullptr;
  ConstructTilingSinkGeModel(kernel_type_so_names, ge_model, root_graph);
  EXPECT_NE(ge_model, nullptr);
  EXPECT_NE(root_graph, nullptr);

  const auto &dumps = DumpManager::GetInstance().GetDumpPropertiesMap();
  for (const auto &item : dumps) {
    DumpManager::GetInstance().RemoveDumpProperties(item.first);
  }

  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
    ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);
    EXPECT_EQ(ge_root_model->CheckAndSetNeedSoInOM(), SUCCESS);
    EXPECT_EQ(ge_root_model->GetSoInOmFlag(), 0x4000);

    ModelBufferData model;
    ModelHelper model_helper;
    model_helper.SetSaveMode(true);
    std::string output_file = opp_path + "/output.om";
    EXPECT_EQ(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, false), SUCCESS);

    ge::ModelParserBase base;
    ge::ModelData model_data;
    EXPECT_EQ(base.LoadFromFile(output_file.c_str(), 0, model_data), SUCCESS);

    auto &model_mgr = ModelManager::GetInstance();
    model_mgr.builtin_aicpu_so_.clear();
    uint32_t model_id = 0;
    GeExecutor ge_executor;
    EXPECT_EQ(ge_executor.LoadModelFromData(model_id, model_data, nullptr, 0U, nullptr, 0U), SUCCESS);
    EXPECT_EQ(ge_executor.UnloadModel(model_id), SUCCESS);
    if (model_data.model_data != nullptr) {
      delete[] reinterpret_cast<char_t *>(model_data.model_data);
    }
    EXPECT_EQ(model_mgr.builtin_aicpu_so_.size(), 1UL);
  }
  RuntimeStub::Reset();
  unsetenv(kEnvName);
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(DavinciModelTest, TilingSink_From_Model_Failed) {
  std::string opp_path = __FILE__;
  std::vector<std::pair<ccKernelType, const std::string>> kernel_type_so_names;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "/test_tmp/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  ConstructOpMasterDeviceSo(opp_path, 1, 2, true, kernel_type_so_names);

  GeModelPtr  ge_model = nullptr;
  ComputeGraphPtr root_graph = nullptr;
  ConstructTilingSinkGeModel(kernel_type_so_names, ge_model, root_graph, true);
  EXPECT_NE(ge_model, nullptr);
  EXPECT_NE(root_graph, nullptr);

  const auto &dumps = DumpManager::GetInstance().GetDumpPropertiesMap();
  for (const auto &item : dumps) {
    DumpManager::GetInstance().RemoveDumpProperties(item.first);
  }

  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
    ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);
    EXPECT_EQ(ge_root_model->CheckAndSetNeedSoInOM(), SUCCESS);
    EXPECT_EQ(ge_root_model->GetSoInOmFlag(), 0x4000);

    ModelBufferData model;
    ModelHelper model_helper;
    model_helper.SetSaveMode(true);
    std::string output_file = opp_path + "/output.om";
    EXPECT_EQ(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, false), SUCCESS);

    ge::ModelParserBase base;
    ge::ModelData model_data;
    EXPECT_EQ(base.LoadFromFile(output_file.c_str(), 0, model_data), SUCCESS);

    auto &model_mgr = ModelManager::GetInstance();
    model_mgr.builtin_aicpu_so_.clear();
    model_mgr.cust_aicpu_so_.clear();
    uint32_t model_id = 0;
    GeExecutor ge_executor;
    EXPECT_NE(ge_executor.LoadModelFromData(model_id, model_data, nullptr, 0U, nullptr, 0U), SUCCESS);
    if (model_data.model_data != nullptr) {
      delete[] reinterpret_cast<char_t *>(model_data.model_data);
    }
    EXPECT_EQ(model_mgr.builtin_aicpu_so_.size(), 0UL);
    EXPECT_EQ(model_mgr.cust_aicpu_so_.size(), 0UL);
  }
  RuntimeStub::Reset();
  unsetenv(kEnvName);
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(DavinciModelTest, FileConstant_Success_UserSetDeviceMem) {
  std::string opp_path = __FILE__;
  std::vector<std::pair<ccKernelType, const std::string>> kernel_type_so_names;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "/test_tmp/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  ConstructOpMasterDeviceSo(opp_path, 1, 2, true, kernel_type_so_names);

  GeModelPtr  ge_model = nullptr;
  ComputeGraphPtr root_graph = nullptr;
  ConstructTilingSinkGeModel(kernel_type_so_names, ge_model, root_graph, true);
  EXPECT_NE(ge_model, nullptr);
  EXPECT_NE(root_graph, nullptr);

  const auto &dumps = DumpManager::GetInstance().GetDumpPropertiesMap();
  for (const auto &item : dumps) {
    DumpManager::GetInstance().RemoveDumpProperties(item.first);
  }

  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
    ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);
    EXPECT_EQ(ge_root_model->CheckAndSetNeedSoInOM(), SUCCESS);
    EXPECT_EQ(ge_root_model->GetSoInOmFlag(), 0x4000);

    ModelBufferData model;
    ModelHelper model_helper;
    model_helper.SetSaveMode(true);
    std::string output_file = opp_path + "/output.om";
    EXPECT_EQ(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, false), SUCCESS);

    ge::ModelParserBase base;
    ge::ModelData model_data;
    EXPECT_EQ(base.LoadFromFile(output_file.c_str(), 0, model_data), SUCCESS);

    auto &model_mgr = ModelManager::GetInstance();
    model_mgr.builtin_aicpu_so_.clear();
    model_mgr.cust_aicpu_so_.clear();
    uint32_t model_id = 0;
    GeExecutor ge_executor;
    EXPECT_NE(ge_executor.LoadModelFromData(model_id, model_data, nullptr, 0U, nullptr, 0U), SUCCESS);
    if (model_data.model_data != nullptr) {
      delete[] reinterpret_cast<char_t *>(model_data.model_data);
    }
    EXPECT_EQ(model_mgr.builtin_aicpu_so_.size(), 0UL);
    EXPECT_EQ(model_mgr.cust_aicpu_so_.size(), 0UL);
  }
  RuntimeStub::Reset();
  unsetenv(kEnvName);
  system(("rm -rf " + opp_path).c_str());
}

void StubExceptionFunc(aclrtExceptionInfo *exception_info, void *reserved) {
  (void)exception_info;
  (void)reserved;
}

TEST_F(DavinciModelTest, init_mc2_cust_aicpu_with_tilefwk_hiddeninput_success) {
  auto tilefwk_hidden_funcs = [](const ge::OpDescPtr &op_desc, std::vector<void *> &addrs) {
    addrs.push_back(reinterpret_cast<void *>(0xf1));
    addrs.push_back(reinterpret_cast<void *>(0xf2));
    return ge::GRAPH_SUCCESS;
  };

  REG_HIDDEN_INPUTS_FUNC(HiddenInputsType::TILEFWK, tilefwk_hidden_funcs);

  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_size = 10000UL;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  const auto op_desc = CreateOpDesc("mc2", "MatmulAllGather", 3, 3);
  EXPECT_NE(op_desc, nullptr);
  model.op_list_[op_desc->GetId()] = op_desc;
  const auto operator_info = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
  model.operator_list_[op_desc->GetId()] = operator_info;
  // ir_def
  op_desc->MutableAllInputName() = {{"x1", 0}, {"bias", 1}};
  op_desc->MutableAllOutputName() = {{"y", 0}, {"z", 1}, {"gather_out", 2}};
  op_desc->AppendIrInput("x1", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("x2", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("bias", IrInputType::kIrInputRequired);
  op_desc->AppendIrOutput("y", IrOutputType::kIrOutputRequired);
  op_desc->AppendIrOutput("z", IrOutputType::kIrOutputRequired);
  op_desc->AppendIrOutput("gather_out", IrOutputType::kIrOutputRequired);

  op_desc->MutableInputDesc(1) = nullptr;
  AttrUtils::SetInt(op_desc, ATTR_NAME_ATTACHED_STREAM_ID, 0);
  AttrUtils::SetInt(op_desc, RECV_ATTR_NOTIFY_ID, 0);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  // add aicpu task
  op_desc->SetInputOffset({1000, 3000});
  op_desc->SetOutputOffset({5000, 5100, 6000});
  op_desc->SetWorkspace({7000});
  op_desc->SetWorkspaceBytes({512});
  // tiling_data
  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("hahahaha");
  op_desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);
  gert::SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl("MatmulAllGather");
  funcs->exception_func = StubExceptionFunc;

  auto &aicpu_task = *model_task_def->add_task();
  aicpu_task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  auto aicpu_kernel = aicpu_task.mutable_kernel();
  domi::KernelContext &aicpu_context = *aicpu_kernel->mutable_context();
  aicpu_context.set_kernel_type(static_cast<int32_t>(ccKernelType::AI_CPU_KFC));
  aicpu_context.set_op_id(op_desc->GetId());
  aicpu_context.set_op_index(op_desc->GetId());
  aicpu_context.set_args_format("{#123}{i0}{i1}{}{o0}{}{o2}{hi.tilefwk0*}{hi.tilefwk1*}{ws*}{overflow_addr}{ws0}{t}");
  aicpu_context.set_args_count(12);
  aicpu_kernel->set_so_name("libmc2_aicpu.so");
  aicpu_kernel->set_kernel_name("mc2_aicpu");
  size_t aicpu_args_size = 128UL;
  const std::vector<uint8_t> args_info(aicpu_args_size, 0);
  aicpu_kernel->set_args(args_info.data(), args_info.size() * sizeof(uint8_t));
  aicpu_kernel->set_args_size(aicpu_args_size);

  // aicpu kernel
  std::vector<char> kernel_bin(128, '0');
  const auto aicpu_bin = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
  model.ge_model_ = MakeShared<GeModel>();
  model.ge_model_->cust_aicpu_kernal_store_.AddKernel(aicpu_bin);

  rtStream_t stream1 = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream1, 0, 0, 0);
  rtStream_t stream2 = nullptr;
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream2, 0, 0, 0);
  model.stream_list_ = {stream1, stream2};

  rtNotify_t rt_notify = nullptr;
  rtNotifyCreate(0, &rt_notify);
  model.notify_list_ = {rt_notify};

  KernelTaskInfo kernel_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(kernel_task_info.ParseTaskRunParam(aicpu_task, &model, task_run_param), SUCCESS);
  PisToArgs args;
  uint8_t device_args[1024];
  args[0].dev_addr = reinterpret_cast<uint64_t>(device_args);
  uint8_t host_data[2048] = {0};
  args[0].len = 2048;
  args[0].host_addr = host_data;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};
  EXPECT_EQ(kernel_task_info.Init(aicpu_task, &model, args, persistant_workspace, iow_addrs), SUCCESS);
}

TEST_F(DavinciModelTest, mc2_with_fusion_task_graph_load_and_success) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  auto hcom_hidden_funcs = [](const ge::OpDescPtr &op_desc, std::vector<void *> &addrs) {
    addrs.push_back(reinterpret_cast<void *>(0xf1));
    return ge::GRAPH_SUCCESS;
  };
  REG_HIDDEN_INPUTS_FUNC(HiddenInputsType::HCOM, hcom_hidden_funcs);

  HcclDllHcomMgr mgr = HcclDllHcomMgr::GetInstance();
  HcclDllHcomMgr::GetInstance().hccl_HcomGetCcuTaskInfo_func = &InitializeHeterogeneousRuntime;

  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);  // x1
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);  // bias
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);  // k0
    auto data_3 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 3);  // k1
    auto data_4 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 4);  // x2
    auto mc2 = OP_CFG("mc2_fusion")
                   .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                   .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
                   .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
                   .Attr(ATTR_NAME_KERNEL_BIN_ID, "fake_mc2_bin_id")
                   .Attr("_kernel_list_first_name", true);
    CHAIN(NODE("_arg_0", data_0)->EDGE(0, 0)->NODE("mc2", mc2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("mc2", mc2));
    CHAIN(NODE("_arg_2", data_2)->EDGE(0, 2)->NODE("mc2", mc2));
    CHAIN(NODE("_arg_3", data_3)->EDGE(0, 3)->NODE("mc2", mc2));
    CHAIN(NODE("_arg_4", data_4)->EDGE(0, 4)->NODE("mc2", mc2));
    CHAIN(NODE("mc2", mc2)->EDGE(1, 1)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("mc2", mc2)->EDGE(2, 2)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("mc2", mc2)->EDGE(3, 3)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("mc2", mc2)->EDGE(4, 4)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("mc2", mc2)->EDGE(5, 5)->NODE("Node_Output", NETOUTPUT));
  };

  auto root_graph = ToComputeGraph(g1);
  EXPECT_NE(root_graph, nullptr);

  for (auto i = 0; i <= 4; ++i) {
    GeTensorDesc output_tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
    const auto &data = root_graph->FindNode("_arg_" + std::to_string(i));
    EXPECT_NE(data, nullptr);
    data->GetOpDesc()->UpdateOutputDesc(0, output_tensor);
    data->GetOpDesc()->SetOutputOffset({1000 + i * 1000});
    data->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  }

  const auto &out_node = root_graph->FindNode("Node_Output");
  EXPECT_NE(out_node, nullptr);
  out_node->GetOpDesc()->SetSrcName({"mc2", "mc2", "mc2", "mc2", "mc2", "mc2"});
  out_node->GetOpDesc()->SetSrcIndex({0, 1, 2, 3, 4, 5});
  out_node->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  GeTensorDesc input_desc(GeShape({1, 4, 4, 4}), FORMAT_ND, DT_FLOAT);
  out_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  out_node->GetOpDesc()->UpdateInputDesc(1, input_desc);
  out_node->GetOpDesc()->SetInputOffset({10000, 11000, 12000, 13000, 14000, 15000});
  auto mc2_node = root_graph->FindNode("mc2");
  EXPECT_NE(mc2_node, nullptr);
  GeShape shape({4, 4, 4, 4});
  GeTensorDesc desc(shape);
  GeShape scalar_shape;
  GeTensorDesc scalar_desc(scalar_shape);
  const auto op_desc = mc2_node->GetOpDescBarePtr();

  op_desc->UpdateInputDesc(0, desc);
  op_desc->UpdateInputDesc(1, desc);
  op_desc->AddDynamicInputDescByIndex("k", 2, 2);
  op_desc->UpdateInputDesc(2, desc);
  op_desc->UpdateInputDesc(3, desc);
  op_desc->UpdateInputDesc(4, desc);
  AttrUtils::SetInt(op_desc, ATTR_NAME_ATTACHED_STREAM_ID, 0);

  op_desc->MutableAllInputName() = {{"x1", 0}, {"bias", 1}, {"k0", 2}, {"k1", 3}, {"a", 4}};
  op_desc->MutableAllOutputName() = {{"y", 0}, {"gather_out", 1}, {"z0", 2}, {"z1", 3}, {"m", 4}, {"n", 5}};

  // ir的顺序为添加顺序
  op_desc->AppendIrInput("x1", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("x2", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("bias", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("k", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("a", IrInputType::kIrInputRequired);
  op_desc->AppendIrOutput("y", IrOutputType::kIrOutputRequired);
  op_desc->AppendIrOutput("gather_out", IrOutputType::kIrOutputRequired);
  op_desc->AppendIrOutput("z", IrOutputType::kIrOutputDynamic);
  op_desc->AppendIrOutput("m", IrOutputType::kIrOutputRequired);
  op_desc->AppendIrOutput("n", IrOutputType::kIrOutputRequired);

  op_desc->MutableInputDesc(1) = nullptr;
  AttrUtils::SetInt(op_desc, ATTR_NAME_ATTACHED_STREAM_ID, 0);
  AttrUtils::SetInt(op_desc, RECV_ATTR_NOTIFY_ID, 0);

  auto output_desc_ptr = op_desc->MutableOutputDesc(5);
  (void)ge::AttrUtils::SetInt(output_desc_ptr, ge::ATTR_NAME_MEMORY_SIZE_CALC_TYPE,
                              static_cast<int64_t>(ge::MemorySizeCalcType::ALWAYS_EMPTY));

  op_desc->SetInputOffset({1000, 2000, 3000, 4000, 5000});
  op_desc->SetOutputOffset({10000, 11000, 12000, 13000, 14000, 15000});
  op_desc->SetWorkspace({20000});
  op_desc->SetWorkspaceBytes({512});
  (void)AttrUtils::SetInt(op_desc, GLOBALWORKSPACE_TYPE, 1);
  (void)AttrUtils::SetStr(op_desc, HCOM_ATTR_GROUP, "test");

  gert::SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl("mc2_fusion");
  funcs->tiling = nullptr;
  funcs->tiling_parse = nullptr;
  funcs->compile_info_creator = nullptr;
  funcs->compile_info_deleter = nullptr;
  EXPECT_EQ(funcs->SetTilingInputDataDependency(5), GRAPH_SUCCESS);

  std::vector<char> test_bin(64, '\0');
  std::string kernel_handle_name = "fake_mc2_bin_id_static_bin";
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("fake_mc2_bin_id_static_bin", std::move(test_bin));
  (void)AttrUtils::SetStr(mc2_node->GetOpDesc(), ATTR_NAME_TBE_KERNEL_NAME, "fake_mc2_bin_id_static_bin");
  op_desc->SetExtAttr(test_kernel->GetName(), test_kernel);
  op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);
  TBEHandleStore::GetInstance().StoreTBEHandle(kernel_handle_name, nullptr, nullptr);


  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("11111111");
  run_info->SetTilingKey(0x1234);
  run_info->AddWorkspace(512);
  op_desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);
  AttrUtils::SetBool(op_desc, "_memcheck", true);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  auto &fusion_task = *model_task_def->add_task();
  fusion_task.set_type(static_cast<int32_t>(ge::ModelTaskType::MODEL_TASK_FUSION_KERNEL));
  auto fusion = fusion_task.mutable_fusion_task();
  fusion->set_args_format("{i0}{i2}{}{i_desc3}{i_instance4}{o0}{o1}{o_desc2}{o_instance4}{}{hi.hcom0*}{ws*}{overflow_addr}{ws0}{t}{#123}");
  fusion->set_op_index(op_desc->GetId());
  fusion->set_kfc_args_format_offset(12);

  // 1.1 AICORE 子任务
  auto* sub1 = fusion->add_fusion_sub_task_info();
  sub1->set_type(domi::FusionSubTaskInfo::AICORE);
  auto* aicore = sub1->mutable_task()->mutable_aicore_fusion_task_info();

  // 1.1.1 KernelContext
  auto* ctx = aicore->mutable_context();
  ctx->set_kernel_type(11);

  // 1.1.2 args 二进制
  aicore->set_is_all_kernel(true);

  // 1.1.3 LaunchConfig → LaunchAttribute
  auto* cfg = aicore->mutable_config();
  auto* attr = cfg->add_launch_attribute();
  attr->set_id(domi::LaunchAttribute::BLOCKDIM);
  attr->mutable_value()->set_block_dim(256);

  attr = cfg->add_launch_attribute();
  attr->set_id(domi::LaunchAttribute::BLOCKDIM_OFFSET);
  attr->mutable_value()->set_block_dim_offset(1);

  attr = cfg->add_launch_attribute();
  attr->set_id(domi::LaunchAttribute::SCHEMMODE);
  attr->mutable_value()->set_schem_model(1);

  // 1.2 CCU 子任务
  auto* sub3 = fusion->add_fusion_sub_task_info();
  sub3->set_type(domi::FusionSubTaskInfo::CCU);
  sub3->mutable_task()->mutable_ccu_task_group()->add_group("group");

  GeModelPtr ge_model = MakeShared<GeModel>();
  auto &kernel_store = ge_model->GetTBEKernelStore();
  kernel_store.AddTBEKernel(test_kernel);

  ge_model->SetGraph(root_graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 2));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
  EXPECT_NE(ge_model, nullptr);
  {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
    ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, root_graph->GetSessionID()), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node, nullptr), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  RuntimeStub::Reset();
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

/**
 * ST场景: CheckIoReuseAddrs - 端到端验证地址相同时校验通过
 *
 * 说明：
 * - 设置 ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES = "0,0"
 * - 执行时输入输出使用相同地址
 *
 * 预期：
 * - CheckIoReuseAddrs 返回 SUCCESS
 */
TEST_F(DavinciModelTest, CheckIoReuseAddrs_SameAddress_Success) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("test_io_reuse");
  GeTensorDesc tensor(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  // Data 节点
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA, 1, 1);
    op_desc->UpdateInputDesc(0, tensor);
    op_desc->UpdateOutputDesc(0, tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    graph->AddNode(op_desc);
  }

  // NetOutput 节点
  {
    OpDescPtr op_desc = CreateOpDesc("netoutput", NETOUTPUT, 1, 0);
    op_desc->UpdateInputDesc(0, tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetSrcName({"data"});
    op_desc->SetSrcIndex({0});
    graph->AddNode(op_desc);
  }

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  SetGeModelAttrs(ge_model);

  // 设置地址复用属性
  AttrUtils::SetStr(ge_model, ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0");

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  DavinciModel model(0, nullptr);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  // 使用相同地址作为输入和输出
  void *shared_addr = reinterpret_cast<void *>(0x10000);
  std::vector<DataBuffer> input_blobs = {{shared_addr, 512, false}};
  std::vector<DataBuffer> output_blobs = {{shared_addr, 512, false}};

  std::vector<GeTensor> empty_tensors;
  EXPECT_EQ(model.CheckIoReuseAddrs(input_blobs, output_blobs, empty_tensors, empty_tensors), SUCCESS);

  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

/**
 * ST场景: CheckIoReuseAddrs - 端到端验证地址不同时校验失败
 *
 * 说明：
 * - 设置 ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES = "0,0"
 * - 执行时输入输出使用不同地址
 *
 * 预期：
 * - CheckIoReuseAddrs 返回 FAILED
 */
TEST_F(DavinciModelTest, CheckIoReuseAddrs_DifferentAddress_Fail) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("test_io_reuse_fail");
  GeTensorDesc tensor(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  // Data 节点
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA, 1, 1);
    op_desc->UpdateInputDesc(0, tensor);
    op_desc->UpdateOutputDesc(0, tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    graph->AddNode(op_desc);
  }

  // NetOutput 节点
  {
    OpDescPtr op_desc = CreateOpDesc("netoutput", NETOUTPUT, 1, 0);
    op_desc->UpdateInputDesc(0, tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetSrcName({"data"});
    op_desc->SetSrcIndex({0});
    graph->AddNode(op_desc);
  }

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  SetGeModelAttrs(ge_model);

  // 设置地址复用属性
  AttrUtils::SetStr(ge_model, ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0");

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  DavinciModel model(0, nullptr);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  // 使用不同地址作为输入和输出
  std::vector<DataBuffer> input_blobs = {{reinterpret_cast<void *>(0x10000), 512, false}};
  std::vector<DataBuffer> output_blobs = {{reinterpret_cast<void *>(0x20000), 512, false}};

  std::vector<GeTensor> empty_tensors;
  EXPECT_NE(model.CheckIoReuseAddrs(input_blobs, output_blobs, empty_tensors, empty_tensors), SUCCESS);

  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

/**
 * ST场景: CheckIoReuseAddrs - 多输入多输出复用验证
 *
 * 说明：
 * - 设置 ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES = "0,0|1,1"
 * - 验证多对索引的地址校验
 *
 * 预期：
 * - 地址匹配时返回 SUCCESS
 * - 部分地址不匹配时返回 FAILED
 */
TEST_F(DavinciModelTest, CheckIoReuseAddrs_MultipleInputsOutputs) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("test_multi_io_reuse");
  GeTensorDesc tensor(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  // Data0 节点
  {
    OpDescPtr op_desc = CreateOpDesc("data0", DATA, 1, 1);
    op_desc->UpdateInputDesc(0, tensor);
    op_desc->UpdateOutputDesc(0, tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    graph->AddNode(op_desc);
  }

  // Data1 节点
  {
    OpDescPtr op_desc = CreateOpDesc("data1", DATA, 1, 1);
    op_desc->UpdateInputDesc(0, tensor);
    op_desc->UpdateOutputDesc(0, tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    graph->AddNode(op_desc);
  }

  // NetOutput 节点
  {
    OpDescPtr op_desc = CreateOpDesc("netoutput", NETOUTPUT, 2, 0);
    op_desc->UpdateInputDesc(0, tensor);
    op_desc->UpdateInputDesc(1, tensor);
    op_desc->SetInputOffset({1024, 1536});
    op_desc->SetSrcName({"data0", "data1"});
    op_desc->SetSrcIndex({0, 0});
    graph->AddNode(op_desc);
  }

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  SetGeModelAttrs(ge_model);

  // 设置多对地址复用属性
  AttrUtils::SetStr(ge_model, ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0|1,1");

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  DavinciModel model(0, nullptr);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  // 测试1：地址全部匹配
  void *addr0 = reinterpret_cast<void *>(0x10000);
  void *addr1 = reinterpret_cast<void *>(0x20000);

  std::vector<DataBuffer> input_blobs = {{addr0, 512, false}, {addr1, 512, false}};
  std::vector<DataBuffer> output_blobs = {{addr0, 512, false}, {addr1, 512, false}};

  std::vector<GeTensor> empty_tensors;
  EXPECT_EQ(model.CheckIoReuseAddrs(input_blobs, output_blobs, empty_tensors, empty_tensors), SUCCESS);

  // 测试2：部分地址不匹配
  void *addr2 = reinterpret_cast<void *>(0x30000);
  output_blobs[1].data = addr2;  // output[1] 使用不同地址

  EXPECT_NE(model.CheckIoReuseAddrs(input_blobs, output_blobs, empty_tensors, empty_tensors), SUCCESS);

  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

/**
 * - 输入输出均使用 GeTensor
 * - 使用同一个 GeTensor 对象放入输入输出列表 (模拟地址复用)
 *
 * 预期：SUCCESS
 */
TEST_F(DavinciModelTest, CheckIoReuseAddrs_GeTensor_SameAddress_Success) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("test_ge_tensor");
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  SetGeModelAttrs(ge_model);
  AttrUtils::SetStr(ge_model, ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0");

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  DavinciModel model(0, nullptr);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  GeTensor tensor;
  std::vector<uint8_t> dummy_data(128, 0xAB);
  tensor.SetData(dummy_data); // 内部分配内存

  std::vector<GeTensor> input_tensors = {tensor};
  std::vector<GeTensor> output_tensors = {tensor};
  std::vector<DataBuffer> empty_blobs;

  EXPECT_EQ(model.CheckIoReuseAddrs(empty_blobs, empty_blobs, input_tensors, output_tensors), SUCCESS);

  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

/**
 * - 设置 ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES = "0,0"
 * - Input 使用 GeTensor A
 * - Output 使用 GeTensor B (拥有独立的内存地址)
 *
 * 预期：
 * - CheckIoReuseAddrs 返回 FAILED (不等于 SUCCESS)
 */
TEST_F(DavinciModelTest, CheckIoReuseAddrs_GeTensor_DiffAddress_Fail) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("test_ge_tensor_fail");
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  SetGeModelAttrs(ge_model);

  AttrUtils::SetStr(ge_model, ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0");

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  DavinciModel model(0, nullptr);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  GeTensor input_tensor;
  std::vector<uint8_t> data_in(128, 0xAA);
  input_tensor.SetData(data_in); // 分配内存地址 A

  GeTensor output_tensor;
  std::vector<uint8_t> data_out(128, 0xBB);
  output_tensor.SetData(data_out); // 分配内存地址 B

  std::vector<GeTensor> input_tensors = {input_tensor};
  std::vector<GeTensor> output_tensors = {output_tensor};
  std::vector<DataBuffer> empty_blobs;

  EXPECT_NE(model.CheckIoReuseAddrs(empty_blobs, empty_blobs, input_tensors, output_tensors), SUCCESS);

  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

/**
 * - 使用 gert::Tensor 重载
 * - 手动设置不同的物理地址
 * - 注意 gert::Tensor 不可拷贝，需直接构造 vector
 *
 * 预期：FAILED
 */
TEST_F(DavinciModelTest, CheckIoReuseAddrs_GertTensor_DiffAddress_Fail) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("test_gert");
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  SetGeModelAttrs(ge_model);
  AttrUtils::SetStr(ge_model, ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0");

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  DavinciModel model(0, nullptr);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  std::vector<gert::Tensor> input_tensors(1);
  input_tensors[0].MutableTensorData().SetAddr(reinterpret_cast<void*>(0xAAAA), nullptr);

  std::vector<gert::Tensor> output_tensors(1);
  output_tensors[0].MutableTensorData().SetAddr(reinterpret_cast<void*>(0xBBBB), nullptr); // 地址不同

  std::vector<DataBuffer> empty_blobs;

  EXPECT_NE(model.CheckIoReuseAddrs(empty_blobs, empty_blobs, input_tensors, output_tensors), SUCCESS);

  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

/**
 * ST场景: CheckIoReuseAddrs - gert::Tensor 场景验证 (地址一致成功)
 *
 * 说明：
 * - 手动设置相同的物理地址
 *
 * 预期：SUCCESS
 */
TEST_F(DavinciModelTest, CheckIoReuseAddrs_GertTensor_SameAddress_Success) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);

  // 1. 初始化
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("test_gert_success");
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  SetGeModelAttrs(ge_model);
  AttrUtils::SetStr(ge_model, ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0");

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  DavinciModel model(0, nullptr);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  // 2. 构造 gert::Tensor
  void* shared_addr = reinterpret_cast<void*>(0xCCCC);

  std::vector<gert::Tensor> input_tensors(1);
  input_tensors[0].MutableTensorData().SetAddr(shared_addr, nullptr);

  std::vector<gert::Tensor> output_tensors(1);
  output_tensors[0].MutableTensorData().SetAddr(shared_addr, nullptr); // 地址一致

  std::vector<DataBuffer> empty_blobs;

  // 3. 执行校验
  EXPECT_EQ(model.CheckIoReuseAddrs(empty_blobs, empty_blobs, input_tensors, output_tensors), SUCCESS);

  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}
} // namespace ge


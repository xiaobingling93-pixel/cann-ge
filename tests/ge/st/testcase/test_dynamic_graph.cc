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
#include <condition_variable>
#include <mutex>
#include <future>
#include "nlohmann/json.hpp"
#include "graph/ge_context.h"
#include "macro_utils/dt_public_scope.h"
#include "base/registry/op_impl_space_registry_v2.h"

#include "graph/load/model_manager/model_manager.h"
#include "host_cpu_engine/host_cpu_engine.h"
#include "ge_running_env/fake_ops_kernel_builder.h"
#include "engines/manager/opskernel_manager/ops_kernel_builder_manager.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "graph/bin_cache/node_compile_cache_module.h"
#include "dflow/compiler/pne/process_node_engine_manager.h"
#include "register/op_tiling_registry.h"
#include "lowering/graph_converter.h"
#include "runtime/model_v2_executor.h"
#include "stub/gert_runtime_stub.h"
#include "api/gelib/gelib.h"
#include "graph/build/stream/stream_utils.h"
#include "macro_utils/dt_public_unscope.h"

#include "graph/operator_reg.h"
#include "graph/ge_attr_value.h"
#include "common/dump/dump_manager.h"
#include "register/op_tiling_registry.h"
#include "framework/executor/ge_executor.h"
#include "ge_running_env/fake_op.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge/ge_api.h"
#include "ge/ge_api_v2.h"
#include "session/session_manager.h"
#include "graph/utils/tensor_adapter.h"
#include "init_ge.h"
#include "common/memory/tensor_trans_utils.h"
#include "utils/bench_env.h"
#include "utils/mock_ops_kernel_builder.h"
#include "utils/taskdef_builder.h"
#include "utils/graph_factory.h"
#include "depends/runtime/src/runtime_stub.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "ge_graph_dsl/assert/check_utils.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "graph/ge_context.h"
#include "graph/bin_cache/op_binary_cache.h"
#include "dflow/compiler/pne/npu/npu_process_node_engine.h"
#include "dflow/compiler/pne/cpu/cpu_process_node_engine.h"

#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/ge_runtime_stub/include/common/share_graph.h"
#include "register/node_converter_registry.h"
#include "depends/op_stub/op_impl/less_important_op_impl.h"
#include "framework/ge_runtime_stub/include/faker/ge_model_builder.h"
#include "register/op_impl_registry.h"
#include "mmpa/mmpa_api.h"
#include "faker/space_registry_faker.h"
#include "faker/global_data_faker.h"
#include "framework/ge_runtime_stub/include/stub/gert_runtime_stub.h"
#include "depends/profiler/src/profiling_test_util.h"
#include "graph/manager/host_mem_manager.h"
#include "register/register_custom_pass.h"

namespace ge {
namespace {
const char *const kEnvName = "ASCEND_OPP_PATH";
const string kOpsProto = "libopsproto_rt2.0.so";
const string kOpMaster = "libopmaster_rt2.0.so";
const string kInner = "built-in";
const string kx86OpsProtoPath = "/op_proto/lib/linux/x86_64/";
const string kx86OpMasterPath = "/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/";
const string kaarch64OpsProtoPath = "/op_proto/lib/linux/aarch64/";
const string kaarch64OpMasterPath = "/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/";

void *mock_host_cpu_handle = (void *) 0x12345678;
optiling::OpRunInfoV2 tiling_run_info_;
bool tiling_result_ = true;

struct DummyCompileInfo {
  int64_t a;
  int64_t b;
  std::vector<int64_t> c;
};

template<typename T, typename std::enable_if<(!std::is_array<T>::value), int>::type = 0>
static void *CreateCompileInfo() {
  return new T();
}
template<typename T>
static void DeleteCompileInfo(void *const obj) {
  delete reinterpret_cast<T *>(obj);
}

auto infer_fun = [](Operator &op) -> graphStatus {
      auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
      *op_desc->MutableOutputDesc(0) = *op_desc->GetInputDescPtr(0);
      return GRAPH_SUCCESS;
    };

uint32_t RunHostCpuForAssign(void *args) {
  auto *arg_base = reinterpret_cast<uint8_t *>(args);
  auto io_addrs = reinterpret_cast<uintptr_t *>(arg_base + sizeof(aicpu::AicpuParamHead));
  auto *input_1 = reinterpret_cast<int32_t *>(io_addrs[1]);
  auto *output = reinterpret_cast<int32_t *>(io_addrs[2]);
  *output = *input_1;
  return 0;
}

class MockRuntime : public RuntimeStub {
 public:
  MOCK_METHOD7(rtKernelLaunchWithFlagV2, int32_t(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo,
      rtSmDesc_t *smDesc, rtStream_t stream, uint32_t flag, const rtTaskCfgInfo_t *cfgInfo));
};

class MockMemcpy : public RuntimeStub {
 public:
  MOCK_METHOD5(rtMemcpy, int32_t(void * , uint64_t, const void *, uint64_t, rtMemcpyKind_t));
};

class MockMmpa : public MmpaStubApiGe {
 public:
  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "RunHostCpuKernel") {
      return (void *) &RunHostCpuForAssign;
    }
    return dlsym(handle, func_name);
  }

  int32_t DlClose(void *handle) override {
    return 0;
  }
};

class MockMalloc : public RuntimeStub {
 public:
  rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) override {
    total_malloc_size += size;

    if (total_malloc_size > 4000) {
      return -1;
    }

    *dev_ptr = new uint8_t[size];
    memset_s(*dev_ptr, size, 0, size);

    return RT_ERROR_NONE;
  }

  rtError_t rtFree(void *dev_ptr) override {
    total_malloc_size = 0;
    delete[](uint8_t *) dev_ptr;
    return RT_ERROR_NONE;
  }

 private:
  uint64_t total_malloc_size = 0;
};

struct GeneralizedShapeInfo
{
  GeShape shape;
  std::vector<std::pair<int64_t, int64_t>> shape_range;
};

struct FakeFuzzCompilerOpsKernelInfoStore : public FakeOpsKernelInfoStore {
 FakeFuzzCompilerOpsKernelInfoStore(const std::string &kernel_lib_name) : FakeOpsKernelInfoStore(kernel_lib_name) {}
  uint32_t GetNodeFuzzCompileCount(const std::string &node_name) {
    return node_name_2_comile_hits_[node_name];
  }
  Status Initialize(const std::map<std::string, std::string> &options) override {
    return SUCCESS;
  }
  Status Finalize() override {
    return SUCCESS;
  }
  bool CheckSupported(const OpDescPtr &op_desc, std::string &reason) const override {
    return true;
  }
  bool CheckSupported(const NodePtr &node, std::string &reason, CheckSupportFlag &flag) const override {
    if (node->GetName() == "not_support_dynamic_shape") {
      flag = CheckSupportFlag::kNotSupportDynamicShape;
      return false;
    }
    return true;
  }
  void GetAllOpsKernelInfo(std::map<std::string, ge::OpInfo> &infos) const override {}

  // fuzz compile interface
  Status FuzzCompileOp(std::vector<ge::NodePtr> &node_vec) override {
    if (AttrUtils::HasAttr(node_vec[0]->GetOpDesc(), "_original_fusion_graph")) {
      ++node_name_2_comile_hits_[node_vec[0]->GetName()];
      // if node is ub fusion node, fuzz compile return failed, to switch origin graph execution
      return FAILED;
    }
    for (auto &node : node_vec) {
      // set compile info on node
      ge::AttrUtils::SetStr(node->GetOpDesc(), "compile_info_key", "op_compile_info_key");
      ge::AttrUtils::SetStr(node->GetOpDesc(), "compile_info_json", "op_compile_info_json");
      ++node_name_2_comile_hits_[node->GetName()];
    }
    return SUCCESS;
  }
  private:
   std::map<std::string, uint32_t> node_name_2_comile_hits_;
};

class FakeFuzzCompileOptimizer : public FakeGraphOptimizer {
  public:
  void SetGeneralizedInfoToNode(const std::string &node_name, const GeneralizedShapeInfo &shape_info) {
    node_2_shape_info_[node_name] = shape_info;
  }

  // simulate fuzz compile
  Status OptimizeGraphPrepare(ComputeGraph &graph) override {
    std::string build_mode;
    if (ge::GetContext().GetOption("ge.shape_generalized_build_mode", build_mode) != ge::GRAPH_SUCCESS) {
      return SUCCESS;
    }

    if (build_mode != "shape_generalized") {
      return SUCCESS;
    }
    // set generlized shape to nodes on graph, current only support graph without subgraph
    for (const auto &node : graph.GetDirectNode()) {
      const auto node_name = node->GetName();
      auto iter = node_2_shape_info_.find(node_name);
      if (iter == node_2_shape_info_.end()) {
        // stub data is wrong. break the process
        continue;
      }
      auto shape_info = iter->second;
      for (size_t i = 0; i < node->GetAllInDataAnchorsSize(); ++i) {
        auto input_desc = node->GetOpDesc()->MutableInputDesc(i);
        input_desc->SetShape(shape_info.shape);
        input_desc->SetOriginShape(shape_info.shape);
        input_desc->SetShapeRange(shape_info.shape_range);
        input_desc->SetOriginShapeRange(shape_info.shape_range);
      }
      for (size_t i = 0; i < node->GetAllOutDataAnchorsSize(); ++i) {
        auto output_desc = node->GetOpDesc()->MutableOutputDesc(i);
        output_desc->SetShape(shape_info.shape);
        output_desc->SetOriginShape(shape_info.shape);
        output_desc->SetShapeRange(shape_info.shape_range);
        output_desc->SetOriginShapeRange(shape_info.shape_range);
      }
    }
    return SUCCESS;
  }

  private:
  std::map<string, GeneralizedShapeInfo> node_2_shape_info_;
};

void FakeFuzzCompileEngine() {
  auto fuzz_compile_optimzer = MakeShared<FakeFuzzCompileOptimizer>();
  GeneralizedShapeInfo shape_info;
  shape_info.shape = GeShape({2,-1,-1,2});
  shape_info.shape_range = {{2,2},{1,20},{1,20},{2,2}};
  fuzz_compile_optimzer->SetGeneralizedInfoToNode("data1", shape_info);
  fuzz_compile_optimzer->SetGeneralizedInfoToNode("data2", shape_info);
  fuzz_compile_optimzer->SetGeneralizedInfoToNode("conv2d", shape_info);
  auto fuzz_compile_ops_kernel_store = MakeShared<FakeFuzzCompilerOpsKernelInfoStore>("AIcoreEngine");
  GeRunningEnvFaker().Reset()
        .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeEngine(kEngineNameAiCpu).KernelInfoStore(kEngineNameAiCpu))
        .Install(FakeEngine(kEngineNameAiCpuTf).KernelInfoStore(kEngineNameAiCpuTf))
        .Install(FakeEngine("DNN_VM_HOST_CPU").KernelInfoStore("DNN_VM_HOST_CPU_OP_STORE"))
        .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("DNN_VM_RTS_OP_STORE"))
        .Install(FakeEngine("AIcoreEngine")
                     .KernelInfoStore(fuzz_compile_ops_kernel_store)
                     .GraphOptimizer("FuzzOptimizer", fuzz_compile_optimzer))
        .Install(FakeOp(CONV2D).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_fun))
        .Install(FakeOp(RELU).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(VARIABLE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"));
}

void FakeFuzzCompileEngineForUbFusion() {
  auto fuzz_compile_optimzer = MakeShared<FakeFuzzCompileOptimizer>();
  GeneralizedShapeInfo shape_info;
  shape_info.shape = GeShape({2,-1,-1,2});
  shape_info.shape_range = {{2,2},{1,20},{1,20},{2,2}};
  fuzz_compile_optimzer->SetGeneralizedInfoToNode("data1", shape_info);
  fuzz_compile_optimzer->SetGeneralizedInfoToNode("conv2d", shape_info);
  fuzz_compile_optimzer->SetGeneralizedInfoToNode("relu", shape_info);
  fuzz_compile_optimzer->SetGeneralizedInfoToNode("netoutput_sub", shape_info);
  fuzz_compile_optimzer->SetGeneralizedInfoToNode("data_a", shape_info);
  fuzz_compile_optimzer->SetGeneralizedInfoToNode("fused_conv2d", shape_info);
  fuzz_compile_optimzer->SetGeneralizedInfoToNode("netoutput", shape_info);
  auto fuzz_compile_ops_kernel_store = MakeShared<FakeFuzzCompilerOpsKernelInfoStore>("AIcoreEngine");
  GeRunningEnvFaker().Reset()
        .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
        .Install(FakeEngine(kEngineNameAiCpu).KernelInfoStore(kEngineNameAiCpu))
        .Install(FakeEngine(kEngineNameAiCpuTf).KernelInfoStore(kEngineNameAiCpuTf))
        .Install(FakeEngine("DNN_VM_HOST_CPU").KernelInfoStore("DNN_VM_HOST_CPU_OP_STORE"))
        .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("DNN_VM_RTS_OP_STORE"))
        .Install(FakeEngine("AIcoreEngine")
                     .KernelInfoStore(fuzz_compile_ops_kernel_store)
                     .GraphOptimizer("FuzzOptimizer", fuzz_compile_optimzer))
        .Install(FakeOp(CONV2D).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_fun))
        .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(RELU).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_fun))
        .Install(FakeOp(VARIABLE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp("_RetVal").InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"));
}

Graph BuildFuzzCompileUnknownRankGraph() {
  std::vector<int64_t> shape = {-2};  // NCHW

  auto data1 = OP_CFG(DATA)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .Build("data1");

  vector<int64_t> test_int64_list_attr = {1,2,3};
  auto conv2d = OP_CFG(CONV2D)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Attr("data_format", "NHWC")  // attr on operator
        .Attr("dilations", test_int64_list_attr)
        .Attr("groups", (int32_t)1)
        .Attr("offset_x", (int32_t)1)
        .Build("conv2d");
  conv2d->SetOpEngineName("AIcoreEngine");
  conv2d->SetOpKernelLibName("AIcoreEngine");  // fake op can not do that?

  auto netoutput = OP_CFG(NETOUTPUT)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Build("netoutput");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->NODE(conv2d)->NODE(netoutput));
  };
  return ToGeGraph(g1);
}

Graph BuildFuzzCompileOriginGraphWithUBfusion() {
   std::vector<int64_t> shape = {2,2,3,2};  // NCHW
   std::vector<int64_t> unknown_shape = {2,2,-1,2};  // NCHW

  auto data1 = OP_CFG(DATA)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, unknown_shape)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
        .Attr("OwnerGraphIsUnknown", true)
        .Build("data1");

  vector<int64_t> test_int64_list_attr = {1,2,3};
  vector<int32_t> test_int32_list_attr = {1,2,3};
  vector<uint32_t> test_uint32_list_attr = {1,2,3};
  auto conv2d = OP_CFG(CONV2D)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, unknown_shape)
        .InCnt(1)
        .OutCnt(1)
        .Attr("string_attr", "test")
        .Attr("int32_attr", (int32_t)1)
        .Attr("uint32_attr", (uint32_t)1)
        .Attr("test_int64_list_attr", test_int64_list_attr)
        .Attr("test_int32_list_attr", test_int32_list_attr)
        .Attr("test_uint32_list_attr", test_uint32_list_attr)
        .Attr("data_format", "NHWC")  // attr on operator
        .Attr("dilations", test_int64_list_attr)
        .Attr("groups", (int32_t)1)
        .Attr("offset_x", (int32_t)1)
        .Build("conv2d");
  conv2d->SetOpEngineName("AIcoreEngine");
  conv2d->SetOpKernelLibName("AIcoreEngine");  // fake op can not do that?

  auto relu = OP_CFG(RELU)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, unknown_shape)
        .InCnt(1)
        .OutCnt(1)
        .Build("relu");
  relu->SetOpEngineName("AIcoreEngine");
  relu->SetOpKernelLibName("AIcoreEngine");  // fake op can not do that? // fe should insure kernel lib name

  auto netoutput_sub = OP_CFG("_RetVal")
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, unknown_shape)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
        .Build("netoutput_sub");

  DEF_GRAPH(fuse_origin_graph) {
    CHAIN(NODE(data1)->NODE(conv2d)->NODE(relu)->NODE(netoutput_sub));
  };

  auto data_a = OP_CFG(DATA)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .Build("data_a");

  auto conv2d_fused = OP_CFG(CONV2D)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Attr("data_format", "NHWC")  // attr on operator
        .Attr("dilations", test_int64_list_attr)
        .Attr("groups", (int32_t)1)
        .Attr("offset_x", (int32_t)1)
        .Attr("_original_fusion_graph", fuse_origin_graph)
        .Build("conv2d_fused");
  conv2d_fused->SetOpEngineName("AIcoreEngine");
  conv2d_fused->SetOpKernelLibName("AIcoreEngine");  // fake op can not do that?

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_a)->NODE(conv2d_fused)->NODE("netoutput", NETOUTPUT));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto conv2d_fused_node = compute_graph->FindNode("conv2d_fused");
  auto fused_graph = ToGeGraph(fuse_origin_graph);
  auto fused_compute_graph = GraphUtilsEx::GetComputeGraph(fused_graph);
  fused_compute_graph->SetGraphUnknownFlag(true);
  auto netoutput_sub_node = fused_compute_graph->FindNode("netoutput_sub");
  AttrUtils::SetGraph(conv2d_fused_node->GetOpDesc(), "_original_fusion_graph", fused_compute_graph);
  return graph;
}

void InitGeLib() {
  map<string, string> options;
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);
  auto instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);

  SchedulerConf scheduler_conf;
  scheduler_conf.name = "aaaaa";

  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->name = "DNN_VM_GE_LOCAL";
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->id = "DNN_VM_GE_LOCAL";
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->independent = false;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->attach = true;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->skip_assign_stream = true;

  scheduler_conf.cal_engines["AIcoreEngine"] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines["AIcoreEngine"]->name = "AIcoreEngine";
  scheduler_conf.cal_engines["AIcoreEngine"]->id = "AIcoreEngine";
  scheduler_conf.cal_engines["AIcoreEngine"]->independent = false;
  scheduler_conf.cal_engines["AIcoreEngine"]->attach = false;
  scheduler_conf.cal_engines["AIcoreEngine"]->skip_assign_stream = false;

  scheduler_conf.cal_engines["DNN_VM_RTS"] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines["DNN_VM_RTS"]->name = "DNN_VM_RTS";
  scheduler_conf.cal_engines["DNN_VM_RTS"]->id = "DNN_VM_RTS";
  scheduler_conf.cal_engines["DNN_VM_RTS"]->independent = false;
  scheduler_conf.cal_engines["DNN_VM_RTS"]->attach = true;
  scheduler_conf.cal_engines["DNN_VM_RTS"]->skip_assign_stream = false;

  scheduler_conf.cal_engines["aicpu_ascend_kernel"] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines["aicpu_ascend_kernel"]->name = "aicpu_ascend_kernel";
  scheduler_conf.cal_engines["aicpu_ascend_kernel"]->id = "aicpu_ascend_kernel";
  scheduler_conf.cal_engines["aicpu_ascend_kernel"]->independent = false;
  scheduler_conf.cal_engines["aicpu_ascend_kernel"]->attach = true;
  scheduler_conf.cal_engines["aicpu_ascend_kernel"]->skip_assign_stream = false;

  scheduler_conf.cal_engines["DNN_VM_AICPU"] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->name = "DNN_VM_AICPU";
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->id = "DNN_VM_AICPU";
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->independent = false;
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->attach = true;
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->skip_assign_stream = false;

  instance_ptr->DNNEngineManagerObj().schedulers_["aaaaa"] = scheduler_conf;
}
}
struct V4CompileInfo : public optiling::CompileInfoBase {
  int64_t a;
  int64_t b;
  int64_t c;
};
bool V4TilingSuccess(const ge::Operator &, const optiling::CompileInfoPtr, optiling::OpRunInfoV2 &) {
  return true;
}
optiling::CompileInfoPtr V4TilingParse(const ge::Operator &, const ge::AscendString &) {
  return std::make_shared<V4CompileInfo>();
}

class DynamicGraphTest : public testing::Test {
 public:
  static void SetUpTestCase() {
    ModelManager::GetInstance().max_model_id_ = 1024;
    GEFinalize();
    ReInitGe();
    BenchEnv::Init();
    // clear engine priority info before registering fake engines
    StreamUtils::engine_priority_.clear();
  }

  static void TearDownTestCase() {
    StreamUtils::engine_priority_.clear();
  }

  void SetUp() {
    InitGeLib();
    char runtime2_env[MMPA_MAX_PATH] = {'0'};
    mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
    std::string opp_path = "./";
    std::string opp_version = "version.info";
    setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);
    (void) system(("touch " + opp_version).c_str());
    (void) system(("echo 'Version=3.20.T100.0.B356' > " + opp_version).c_str());

    std::string path_vendors = opp_path + "vendors";
    std::string path_config = path_vendors + "/config.ini";
    (void) system(("mkdir -p " + path_vendors).c_str());
    (void) system(("echo 'load_priority=customize' > " + path_config).c_str());

    std::string inner_x86_proto_path = opp_path + kInner + kx86OpsProtoPath;
    (void)system(("mkdir -p " + inner_x86_proto_path).c_str());
    inner_x86_proto_path += kOpsProto;
    (void) system(("touch " + inner_x86_proto_path).c_str());
    (void) system(("echo 'ops proto x86 ' > " + inner_x86_proto_path).c_str());

    std::string inner_aarch64_proto_path = opp_path + kInner + kaarch64OpsProtoPath;
    (void) system(("mkdir -p " + inner_aarch64_proto_path).c_str());
    inner_aarch64_proto_path += kOpsProto;
    (void) system(("touch " + inner_aarch64_proto_path).c_str());
    (void) system(("echo 'ops proto aarch64 ' > " + inner_aarch64_proto_path).c_str());

    std::string inner_x86_tiling_path = opp_path + kInner + kx86OpMasterPath;
    (void) system(("mkdir -p " + inner_x86_tiling_path).c_str());
    inner_x86_tiling_path += kOpMaster;
    (void) system(("touch " + inner_x86_tiling_path).c_str());
    (void) system(("echo 'op tiling_x86 ' > " + inner_x86_tiling_path).c_str());

    std::string inner_aarch64_tiling_path = opp_path + kInner + kaarch64OpMasterPath;
    (void) system(("mkdir -p " + inner_aarch64_tiling_path).c_str());
    inner_aarch64_tiling_path += kOpMaster;
    (void) system(("touch " + inner_aarch64_tiling_path).c_str());
    (void) system(("echo 'op tiling aarch_64 ' > " + inner_aarch64_tiling_path).c_str());

    tiling_result_ = true;
    tiling_run_info_ = optiling::OpRunInfoV2{};

    auto unique_infer_fun = [](Operator &op) -> graphStatus {
      auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
      auto output_desc = op_desc->MutableOutputDesc(0);
      output_desc->SetShape(GeShape({-1}));
      output_desc->SetShapeRange({{1, 16}});

      op_desc->MutableOutputDesc(1)->SetDataType(DT_INT32);
      op_desc->MutableOutputDesc(1)->SetShape({});
      return GRAPH_SUCCESS;
    };

    auto type2_infer = [](Operator &op) -> graphStatus {
      auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
      op_desc->SetOpInferDepends({"__input0"});
      Tensor tensor;
      auto output_desc = op_desc->MutableOutputDesc(0);
      if (op.GetInputConstData("__input0", tensor) == GRAPH_SUCCESS) {
        output_desc->SetShape(GeShape({4}));
      } else {
        output_desc->SetShape(GeShape({-1}));
        output_desc->SetShapeRange({{1, 16}});
      }
      return GRAPH_SUCCESS;
    };
    auto ge_dev = GeRunningEnvFaker();
    ge_dev.Reset()
        .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
        .Install(FakeEngine(kEngineNameAiCpu).KernelInfoStore(kEngineNameAiCpu))
        .Install(FakeEngine(kEngineNameAiCpuTf).KernelInfoStore(kEngineNameAiCpuTf))
        .Install(FakeEngine("DNN_VM_HOST_CPU").KernelInfoStore("DNN_VM_HOST_CPU_OP_STORE"))
        .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("DNN_VM_RTS_OP_STORE"))
        .Install(FakeEngine("DNN_VM_AICPU").KernelInfoStore("DNN_VM_AICPU"))
        .Install(FakeOp("MyAdd").InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_fun))
        .Install(FakeOp(RELU).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_fun))
        .Install(FakeOp(CONV2D).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_fun))
        .Install(FakeOp(MUL).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_fun))
        .Install(FakeOp(CAST).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_fun))
        .Install(FakeOp("MyNeg").InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_fun))
        .Install(FakeOp(NEG).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_fun))
        .Install(FakeOp(TOPKV2).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_fun))
        .Install(FakeOp("FakeType2Op").InfoStoreAndBuilder("AIcoreEngine").InferShape(type2_infer))
        .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(VARIABLE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(SHAPE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(IDENTITY).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE").InferShape(infer_fun))
        .Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(IF).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(CASE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(WHILE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp("Unique").InfoStoreAndBuilder("aicpu_ascend_kernel").InferShape(unique_infer_fun))
        .Install(FakeOp("UniqueV2").InfoStoreAndBuilder("AIcoreEngine").InferShape(unique_infer_fun))
        .Install(FakeOp("FakeUnique").InfoStoreAndBuilder("DNN_VM_AICPU").InferShape(unique_infer_fun))
        .Install(FakeOp(GETNEXT).InfoStoreAndBuilder("DNN_VM_AICPU"))
        .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(ASSIGN).InfoStoreAndBuilder("DNN_VM_HOST_CPU_OP_STORE"))
        .Install(FakeOp(SUB).InfoStoreAndBuilder("DNN_VM_HOST_CPU_OP_STORE"))
        .Install(FakeOp("FakeOpHostCpu").InfoStoreAndBuilder("DNN_VM_HOST_CPU_OP_STORE"))
        .Install(FakeOp("FakeOpNpu").InfoStoreAndBuilder("AIcoreEngine"))
        .Install(FakeOp(ADD).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_fun))
        .Install(FakeOp(LESS).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_fun))
        .Install(FakeOp(ENTER).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"))
        .Install(FakeOp(MERGE).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"))
        .Install(FakeOp(SWITCH).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"))
        .Install(FakeOp(LOOPCOND).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"))
        .Install(FakeOp(STREAMSWITCH).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"))
        .Install(FakeOp(STREAMMERGE).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"))
        .Install(FakeOp(STREAMACTIVE).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"))
        .Install(FakeOp(NEXTITERATION).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"))
	.Install(FakeOp(NPUGETFLOATSTATUS).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE").InferShape(infer_fun))
        .Install(FakeOp(NPUCLEARFLOATSTATUS).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE").InferShape(infer_fun))
        .Install(FakeOp(EXIT).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"));

    optiling::OpTilingFuncV2 tilingfun = [](const ge::Operator &op,
                                            const optiling::OpCompileInfoV2 &compile_info,
                                            optiling::OpRunInfoV2 &run_info) -> bool {
      run_info.SetWorkspaces({1024});
      return true;
    };
    graphStatus (*tiling_parse_func_rt2)(::gert::TilingParseContext *)  =
        [](gert::TilingParseContext *parse_context) -> graphStatus { return GRAPH_SUCCESS; };

    gert::OpImplKernelRegistry::TilingKernelFunc tilingfun_rt2 =
        [](gert::TilingContext *tiling_context) -> graphStatus {
      size_t *workspace_size = tiling_context->GetWorkspaceSizes(1);
      *workspace_size=1024;
      return ge::GRAPH_SUCCESS;
    };

    optiling::OpTilingFuncV2 mock_tiling_func = [&](const ge::Operator &op,
                                                       const optiling::OpCompileInfoV2 &compile_info,
                                                       optiling::OpRunInfoV2 &run_info) -> bool {
      run_info = tiling_run_info_;
      return tiling_result_;
    };

    optiling::OpTilingRegistryInterf_V2(RELU, tilingfun);
    REGISTER_OP_TILING_UNIQ_V2(ReLU, tilingfun, 1);

    // optiling::OpTilingRegistryInterf_V2("FakeType2Op", tilingfun);
    // REGISTER_OP_TILING_UNIQ_V2(FakeType2Op, tilingfun, 1);
    // 修改为rt2的tiling 桩
    auto funcs = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl("FakeType2Op");
    funcs->tiling = tilingfun_rt2;
    funcs->tiling_parse = reinterpret_cast<gert::KernelRegistry::KernelFunc>(tiling_parse_func_rt2);
    funcs->compile_info_creator = CreateCompileInfo<DummyCompileInfo>;
    funcs->compile_info_deleter = DeleteCompileInfo<DummyCompileInfo>;

    optiling::OpTilingRegistryInterf_V2("MyAdd", mock_tiling_func);
    REGISTER_OP_TILING_UNIQ_V2(MyAdd, mock_tiling_func, 1);

    REGISTER_OP_TILING_UNIQ_V4(MyNeg, V4TilingSuccess, V4TilingParse, 1);

    optiling::OpTilingRegistryInterf_V2(LESS, mock_tiling_func);
    REGISTER_OP_TILING_UNIQ_V2(Less, mock_tiling_func, 1);

    optiling::OpTilingRegistryInterf_V2(MUL, mock_tiling_func);
    REGISTER_OP_TILING_UNIQ_V2(Mul, mock_tiling_func, 1);

    optiling::OpTilingRegistryInterf_V2(CONV2D, tilingfun);
    REGISTER_OP_TILING_UNIQ_V2(Conv2D, tilingfun, 1);
  }
  void TearDown() {
    MmpaStub::GetInstance().Reset();
    MockRuntime::Reset();
    GEFinalize();
    ReInitGe();
    char runtime2_env[MMPA_MAX_PATH] = {'1'};
    mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
    (void) system("rm -rf ./version.info");
    (void) system("rm -rf ./vendors");
    (void) system("rm -rf ./built-in");
    unsetenv("ASCEND_OPP_PATH");
    unsetenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM");
  }
};

extern void OfflineModelCommand(GeExecutor &ge_executor, const uint32_t model_id);

namespace {
Tensor CreateTensor(const TensorDesc &tensor_desc) {
  int64_t tensor_size = -1;
  TensorUtils::GetTensorSizeInBytes(TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc), tensor_size);
  std::vector<uint8_t> tensor_buffer(tensor_size);
  Tensor tensor(tensor_desc);
  tensor.SetData(std::move(tensor_buffer));
  return tensor;
}

Tensor CreateTensor(const std::vector<int64_t> &dims, Format format = FORMAT_ND, DataType data_type = DT_FLOAT) {
  return CreateTensor(TensorDesc(Shape(dims), format, data_type));
}

std::vector<Tensor> CreateInputTensors(const Graph &graph) {
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  std::map<int64_t, GeTensorDescPtr> tensor_descs;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == DATA) {
      int64_t index = 0;
      AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_INDEX, index);
      tensor_descs[index] = node->GetOpDesc()->MutableOutputDesc(0);
    }
  }

  std::vector<Tensor> tensors(tensor_descs.size());
  for (const auto &it : tensor_descs) {
    tensors[it.first] = CreateTensor(TensorAdapter::GeTensorDesc2TensorDesc(*it.second));
  }

  return tensors;
}

Status GenerateTaskForAiCore(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) {
  tasks.emplace_back(AiCoreTaskDefBuilder(node).BuildAtomicAddrCleanTask());
  tasks.emplace_back(AiCoreTaskDefBuilder(node).BuildTask(true));
  return SUCCESS;
}

Status GenerateTaskForTaskWithHandle(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) {
  tasks.emplace_back(AiCoreTaskDefBuilder(node).BuildTaskWithHandle());
  return SUCCESS;
}

Status GenerateTaskForStaticAicore(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) {
  tasks.emplace_back(AiCoreTaskDefBuilder(node).BuildTask());
  return SUCCESS;
}

Status GenerateTaskForHostCpu(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) {
  tasks.emplace_back(AicpuTaskDefBuilder(node).BuildHostCpuTask(0));
  return SUCCESS;
}

Status GenerateTaskForAicpuDependRange(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) {
  tasks.emplace_back(AicpuTaskDefBuilder(node).BuildAicpuTask(3));
  return SUCCESS;
}

Status SkipGenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) {
  (void)tasks;
  return SUCCESS;
}

Graph BuildDynamicInputGraph() {
  DEF_GRAPH(dynamic_graph) {
    auto data_0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1});

    auto data_1 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1});

    auto add = OP_CFG("MyAdd")
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1});

    CHAIN(NODE("_arg_0", data_0)->NODE("add", add)->NODE("Node_Output", net_output));
    CHAIN(NODE("_arg_1", data_1)->NODE("add", add));
  };

  return ToGeGraph(dynamic_graph);
}

Graph BuildDynamicInputGraphForRtV2() {
  DEF_GRAPH(graph_def) {
    auto data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto shape_op = OP_CFG(SHAPE)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {1});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("data", data)
              ->NODE("shape", shape_op)
              ->NODE("Node_Output", net_output));
  };
  return ToGeGraph(graph_def);
}

Graph BuildControlOpIfGraph() {
  DEF_GRAPH(then_branch) {
    auto data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("then_arg_0", data)->NODE("then_Node_Output", net_output));
  };

  DEF_GRAPH(else_branch) {
    auto data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    auto unique_op = OP_CFG("Unique")
        .InCnt(1)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("else_arg_0", data)->NODE("Unique", unique_op)->NODE("else_Node_Output", net_output));
  };

  auto then_graph = ToComputeGraph(then_branch);
  auto else_graph = ToComputeGraph(else_branch);

  DEF_GRAPH(if_graph) {
    auto pred_data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto value_data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto if_op = OP_CFG(IF)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16})
        .Build("if");

    if_op->MutableOutputDesc(0)->SetShape(GeShape({-1}));
    if_op->RegisterSubgraphIrName("then_branch", SubgraphType::kStatic);
    if_op->RegisterSubgraphIrName("else_branch", SubgraphType::kStatic);
    if_op->AddSubgraphName(then_graph->GetName());
    if_op->SetSubgraphInstanceName(0, then_graph->GetName());
    if_op->AddSubgraphName(else_graph->GetName());
    if_op->SetSubgraphInstanceName(1, else_graph->GetName());

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    CHAIN(NODE("arg_pred", pred_data)->NODE(if_op)->NODE("Node_Output", net_output));
    CHAIN(NODE("arg_value", value_data)->NODE(if_op));
  };

  auto root_graph = ToComputeGraph(if_graph);
  auto if_node = root_graph->FindFirstNodeMatchType(IF);
  EXPECT_TRUE(if_node != nullptr);
  then_graph->SetParentNode(if_node);
  then_graph->SetParentGraph(root_graph);
  else_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(root_graph);
  root_graph->AddSubgraph(then_graph);
  root_graph->AddSubgraph(else_graph);
  return GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
}

Graph BuildType2AndGeLocal() {
  DEF_GRAPH(graph_def) {
    auto var = OP_CFG(VARIABLE)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op = OP_CFG("FakeType2Op")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16}).Build("fake_type2_op");

    auto shape_op = OP_CFG(SHAPE)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    auto cast_op = OP_CFG(CAST)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("var", var)
              ->NODE(fake_type2_op)
              ->NODE("shape", shape_op)
              ->NODE("cast", cast_op)
              ->NODE("Node_Output", net_output));
  };
  return ToGeGraph(graph_def);
}

void TestRuntimeV2Compile(Graph &graph) {
  char runtime2_env[MMPA_MAX_PATH] = {'1'};
  mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  std::map<string, string> build_options;
  ModelBufferData model_buffer_data{};
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  runtime2_env[0] = {'0'};
  mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
}

void LoadDynamicOfflineGraph(Graph &graph, uint32_t &model_id) {
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  std::map<AscendString, AscendString> init_options;
  init_options[ge::OPTION_HOST_ENV_OS] = "linux";
  init_options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  std::map<string, string> build_options;
  ModelBufferData model_buffer_data{};
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);

  GeExecutor ge_executor;
  ge_executor.Initialize();

  ModelData model_data;
  model_data.model_data = model_buffer_data.data.get();
  model_data.model_len = model_buffer_data.length;
  EXPECT_EQ(ge_executor.LoadModelFromData(model_id, model_data, nullptr, 0, nullptr, 0), SUCCESS);
  aclgrphBuildFinalize();
}

void BuildAndExecDynamicOfflineModel() {
  auto graph = BuildDynamicInputGraph();
  uint32_t model_id = 0;
  LoadDynamicOfflineGraph(graph, model_id);

  GeExecutor ge_executor;
  ge_executor.Initialize({});
  GeShape shape({8, 16});
  GeTensorDesc tensor_desc(shape);
  std::vector<GeTensorDesc> input_desc{tensor_desc, tensor_desc};
  std::vector<GeTensorDesc> output_desc;
  uint8_t buffer[8 * 16 * 4];
  RunModelData input_data;
  input_data.blobs.emplace_back(DataBuffer(buffer, sizeof(buffer)));
  input_data.blobs.emplace_back(DataBuffer(buffer, sizeof(buffer)));
  RunModelData output_data;
  output_data.blobs.emplace_back(DataBuffer(buffer, sizeof(buffer)));
  EXPECT_EQ(ge_executor.ExecModel(model_id, nullptr, input_data, input_desc, output_data, output_desc), SUCCESS);

  OfflineModelCommand(ge_executor, model_id);
  ge_executor.UnloadModel(model_id);
  ge_executor.Finalize();
}

Status RunGraphAsync(Session &session,
                     uint32_t graph_id,
                     const std::vector<Tensor> &inputs,
                     std::vector<Tensor> &outputs) {
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  Status ret = SUCCESS;
  RunAsyncCallback callback = [&](Status status, std::vector<ge::Tensor> &output_tensors) {
    std::unique_lock<std::mutex> lk(mu);
    ret = status;
    outputs = output_tensors;
    done = true;
    cv.notify_all();
  };

  auto run_ret = session.RunGraphAsync(graph_id, inputs, callback);
  if (run_ret != SUCCESS) {
    return run_ret;
  }

  std::unique_lock<std::mutex> lk(mu);
  if (!cv.wait_for(lk, std::chrono::seconds(15), [&]() { return done; })) {
    // TODO timeout occasionally
    return SUCCESS;
  }
  return ret;
}
Status RunGraphAsync(GeSession &session,
                     uint32_t graph_id,
                     const std::vector<gert::Tensor> &inputs,
                     std::vector<gert::Tensor> &outputs) {
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  Status ret = SUCCESS;
  RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &output_tensors) {
    std::unique_lock<std::mutex> lk(mu);
    ret = status;
    outputs = std::move(output_tensors);
    done = true;
    cv.notify_all();
  };

  auto run_ret = session.RunGraphAsync(graph_id, inputs, callback);
  if (run_ret != SUCCESS) {
    return run_ret;
  }

  std::unique_lock<std::mutex> lk(mu);
  if (!cv.wait_for(lk, std::chrono::seconds(15), [&]() { return done; })) {
    // TODO timeout occasionally
    return SUCCESS;
  }
  return ret;
}
void ExecuteDynamicOnlineGraph(Graph &graph,
                               const std::map<std::string, std::string> &session_options = {},
                               const std::map<std::string, std::string> &graph_options = {},
                               bool is_train = false) {
  auto options = session_options;
  if (is_train) {
    options[OPTION_GRAPH_RUN_MODE] = "1";  // train
  }
  options[VARIABLE_MEMORY_MAX_SIZE] = "5000000";
  DumpProperties dump_properties;
  dump_properties.SetDumpStatus("on");
  dump_properties.SetDumpMode("all");
  dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
  DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);
  std::vector<Tensor> inputs = CreateInputTensors(graph);
  std::vector<Tensor> outputs;
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  session.RemoveGraph(graph_id);
}

void EXPECT_ExecuteDynamicOnlineInfer(Graph &graph,
                                      const std::map<std::string, std::string> &session_options = {},
                                      const std::map<std::string, std::string> &graph_options = {}) {
  return ExecuteDynamicOnlineGraph(graph, session_options, graph_options, false);
}

void EXPECT_ExecuteDynamicOnlineTrain(Graph &graph,
                                      const std::map<std::string, std::string> &session_options = {},
                                      const std::map<std::string, std::string> &graph_options = {}) {
  return ExecuteDynamicOnlineGraph(graph, session_options, graph_options, true);
}

void BuildAndExecDynamicOnlineModel() {
  DEF_GRAPH(graph_def) {
    auto var = OP_CFG(VARIABLE)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto unique_op = OP_CFG("Unique")
        .InCnt(1)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    CHAIN(NODE("var", var)->NODE("unique", unique_op)->NODE("Node_Output", net_output));
  };

  auto graph = ToGeGraph(graph_def);
  ExecuteDynamicOnlineGraph(graph);
}

void BuildAndExecDynamicOnlineModelExp(Status status) {
  DEF_GRAPH(graph_def) {
    auto var = OP_CFG(VARIABLE)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto unique_op = OP_CFG("Unique")
        .InCnt(1)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    CHAIN(NODE("var", var)->NODE("unique", unique_op)->NODE("Node_Output", net_output));
  };

  auto graph = ToGeGraph(graph_def);
  std::map<std::string, std::string> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "5000000";

  Session session(options);
  DumpManager::GetInstance().RemoveDumpProperties(session.GetSessionId());
  DumpProperties dump_properties;
  dump_properties.SetDumpStatus("on");
  dump_properties.SetDumpMode("all");
  dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
  dump_properties.is_train_op_debug_ = true;
  DumpManager::GetInstance().AddDumpProperties(session.GetSessionId(), dump_properties);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);
  std::vector<Tensor> inputs = CreateInputTensors(graph);
  std::vector<Tensor> outputs;
  if (status == SUCCESS) {
    EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  } else {
    EXPECT_NE(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  }
  session.RemoveGraph(graph_id);
}

void ExecDynamicOfflineModel(GeExecutor &ge_executor, uint32_t model_id) {
  GeShape shape({8, 16});
  GeTensorDesc tensor_desc(shape);
  std::vector<GeTensorDesc> input_desc{tensor_desc, tensor_desc};
  std::vector<GeTensorDesc> output_desc;
  uint8_t buffer[8 * 16 * 4];
  RunModelData input_data;
  input_data.blobs.emplace_back(DataBuffer(buffer, sizeof(buffer)));
  input_data.blobs.emplace_back(DataBuffer(buffer, sizeof(buffer)));
  RunModelData output_data;
  output_data.blobs.emplace_back(DataBuffer(buffer, sizeof(buffer)));
  EXPECT_EQ(ge_executor.ExecModel(model_id, nullptr, input_data, input_desc, output_data, output_desc), SUCCESS);
}

void BuildDynamicGraph(Graph &graph, ModelBufferData &model_buffer_data) {
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  std::map<AscendString, AscendString> init_options;
  EXPECT_EQ(aclgrphBuildInitialize(init_options), SUCCESS);

  std::map<string, string> build_options;
  EXPECT_EQ(aclgrphBuildModel(graph, build_options, model_buffer_data), SUCCESS);
  aclgrphBuildFinalize();
}

void LoadDynamicGraph(ModelBufferData &model_buffer_data, uint32_t &model_id) {
  GeExecutor ge_executor;
  ge_executor.Initialize({});
  ModelData model_data;
  model_data.model_data = model_buffer_data.data.get();
  model_data.model_len = model_buffer_data.length;
  EXPECT_EQ(ge_executor.LoadModelFromData(model_id, model_data, nullptr, 0, nullptr, 0), SUCCESS);
}
namespace {
  Status DynamicGraphWithAicpuPass(const ConstGraphPtr &graph, StreamPassContext &context) {
    AscendString graph_name;
    graph->GetName(graph_name);
    if (graph_name != "DynamicGraphWithAicpu") {
      return SUCCESS;
    }
    std::cout << "before current max stream id is "<< context.GetCurrMaxStreamId()<< std::endl;
    for (auto n : graph->GetDirectNode()) {
      AscendString name;
      n.GetName(name);
      if (name != "unique") {
        continue;
      }
      context.SetStreamId(n, context.AllocateNextStreamId());
    }
    std::cout << "after current max stream id is "<< context.GetCurrMaxStreamId()<< std::endl;
    return SUCCESS;
  }
} // namespace

/**
 *
  ┌────────┐  (0,0)   ┌─────────┐  (0,0)   ┌────────┐  (0,0)   ┌─────────────┐  (0,1)   ┌─────────┐
  │ _arg_0 │ ───────> │   neg   │ ───────> │ unique │ ───────> │ Node_Output │ <─────── │ const_0 │
  └────────┘          └─────────┘          └────────┘          └─────────────┘          └─────────┘
                        ∧                                        ∧
                        │ (0,1)                                  │ (0,2)
                        │                                        │
                      ┌─────────┐                              ┌─────────────┐
                      │ getnext │                              │   const_1   │
                      └─────────┘                              └─────────────┘
*/
Graph BuildDynamicGraphWithAicpu() {
  GeTensorDesc tensor_desc(GeShape{});
  GeTensor tensor(tensor_desc);

  GeTensorDesc tensor_desc_1(GeShape(std::vector<int64_t>({1})), FORMAT_ND, DT_STRING);
  std::vector<uint8_t> string_buffer(24, 0);
  GeTensor tensor_1(tensor_desc_1);
  tensor_1.SetData(std::move(string_buffer));

  DEF_GRAPH(dynamic_graph) {
    auto var_0 = OP_CFG(VARIABLE)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto const_0 = OP_CFG(CONSTANTOP)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .Attr(ATTR_NAME_WEIGHTS, tensor)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {});

    auto const_1 = OP_CFG(CONSTANTOP)
        .OutCnt(1)
        .Attr(ATTR_NAME_WEIGHTS, tensor_1)
        .Attr(ATTR_NAME_FORCE_UNKNOWN_SHAPE, true)
        .TensorDesc(FORMAT_ND, DT_STRING, {});

    auto neg = OP_CFG(NEG)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16})
        .Attr(ATTR_NAME_STREAM_LABEL, "aaa");

    auto unique_op = OP_CFG("FakeUnique")
        .InCnt(1)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16})
        .Attr(ATTR_NAME_INSERT_BP_PROFILILNG_TASK, true)
        .Attr(ATTR_NAME_INSERT_FP_PROFILILNG_TASK, true)
        .Attr(ATTR_NAME_STREAM_LABEL, "aaa");

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1});

    CHAIN(NODE("_arg_0", var_0)->EDGE(0, 0)->NODE("neg", neg)->NODE("unique", unique_op)->NODE("Node_Output", net_output));
    CHAIN(NODE("getnext", GETNEXT)->EDGE(0, 1)->NODE("neg"));
    CHAIN(NODE("const_0", const_0)->NODE("Node_Output", net_output));
    CHAIN(NODE("const_1", const_1)->NODE("Node_Output", net_output));
  };

  return ToGeGraph(dynamic_graph);
}
}  // namespace

TEST_F(DynamicGraphTest, TestDynamicOfflineModel_aicore_with_atomic_output) {
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  BuildAndExecDynamicOfflineModel();
}

TEST_F(DynamicGraphTest, TestDynamicOfflineModel_multi_thread) {
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);
  auto malloc_mock = std::make_shared<MockMalloc>();
  RuntimeStub::SetInstance(malloc_mock);

  auto graph = BuildDynamicInputGraph();
  ModelBufferData model_buffer_data{};
  BuildDynamicGraph(graph, model_buffer_data);

  uint32_t model_id = 0;
  uint32_t model_id2 = 0;
  LoadDynamicGraph(model_buffer_data, model_id);
  LoadDynamicGraph(model_buffer_data, model_id2);

  GeExecutor ge_executor;
  ge_executor.Initialize({});
  auto future1 = std::async(std::launch::async, [&ge_executor, model_id] () {
      ExecDynamicOfflineModel(ge_executor, model_id);
  });
  auto future2 = std::async(std::launch::async, [&ge_executor, model_id2] () {
      ExecDynamicOfflineModel(ge_executor, model_id2);
  });
  future1.wait();
  future2.wait();
  future1.get();
  future2.get();

  ge_executor.UnloadModel(model_id);
  ge_executor.UnloadModel(model_id2);
  ge_executor.Finalize();
}

TEST_F(DynamicGraphTest, TestGetMemAndWeightSizeFromDynamicOM) {
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);

  auto graph = BuildDynamicInputGraph();
  ModelBufferData model_buffer_data{};
  BuildDynamicGraph(graph, model_buffer_data);

  uint32_t model_id = 0;
  LoadDynamicGraph(model_buffer_data, model_id);

  GeExecutor ge_executor;
  ge_executor.Initialize({});
  size_t mem_size = 0U;
  size_t weight_size = 0U;
  ModelData model_data;
  model_data.model_data = model_buffer_data.data.get();
  model_data.model_len = model_buffer_data.length;
  EXPECT_EQ(ge_executor.GetMemAndWeightSize(model_data.model_data, model_data.model_len, mem_size, weight_size),
            SUCCESS);
  EXPECT_EQ(weight_size, 1024);
  EXPECT_TRUE(mem_size == 0U);
  ExecDynamicOfflineModel(ge_executor, model_id);
  ge_executor.Finalize();
}

TEST_F(DynamicGraphTest, TestUnloadModelAfterFinalize) {
  MockForGenerateTask("AIcoreEngine", GenerateTaskForAiCore);

  auto graph = BuildDynamicInputGraph();
  ModelBufferData model_buffer_data{};
  BuildDynamicGraph(graph, model_buffer_data);

  uint32_t model_id = 0;
  LoadDynamicGraph(model_buffer_data, model_id);

  GeExecutor ge_executor;
  ge_executor.Initialize({});
  ExecDynamicOfflineModel(ge_executor, model_id);
  ge_executor.Finalize();
  ge_executor.UnloadModel(model_id);
}

TEST_F(DynamicGraphTest, TestDynamicOfflineModel_aicore_with_atomic_workspace) {
  auto func = [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    auto op_desc = node.GetOpDesc();
    auto ret = GenerateTaskForAiCore(node, context, tasks);
    GeAttrValue::NAMED_ATTRS workspaces;
    GeAttrValue::NamedAttrs workspaces_attrs;
    vector<int> dim_types;
    dim_types.push_back(0);
    dim_types.push_back(1);
    AttrUtils::SetListInt(workspaces_attrs, op_desc->GetName(), dim_types);
    AttrUtils::SetNamedAttrs(op_desc, EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspaces_attrs);
    return ret;
  };
  MockForGenerateTask("AIcoreEngine", func);
  tiling_run_info_.SetWorkspaces({256});
  BuildAndExecDynamicOfflineModel();
}

TEST_F(DynamicGraphTest, TestDynamicOfflineModel_aicore_with_atomic_workspace_on_rt2_tiling) {
  auto func = [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    auto op_desc = node.GetOpDesc();
    auto ret = GenerateTaskForAiCore(node, context, tasks);
    GeAttrValue::NAMED_ATTRS workspaces;
    GeAttrValue::NamedAttrs workspaces_attrs;
    vector<int> dim_types;
    dim_types.push_back(0);
    dim_types.push_back(1);
    AttrUtils::SetListInt(workspaces_attrs, op_desc->GetName(), dim_types);
    AttrUtils::SetListInt(op_desc, "tbe_op_atomic_dtypes", {0});
    AttrUtils::SetNamedAttrs(op_desc, EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspaces_attrs);
    return ret;
  };
  MockForGenerateTask("AIcoreEngine", func);
  tiling_run_info_.SetWorkspaces({256});

  gert::KernelRegistry::KernelFunc tiling_parse_func_rt2 =
      [](gert::KernelContext *parse_context) -> graphStatus { return GRAPH_SUCCESS; };

  typedef void* (*CreateCompileInfo)();
  typedef void (*DeleteCompileInfo)(void *obj);
  CreateCompileInfo create_compile_info = []() -> void *{
    return new int64_t();
  };
  DeleteCompileInfo delete_compile_info = [](void *obj) -> void {
    if (obj != nullptr) {
      delete (int64_t *)obj;
      obj = nullptr;
    }
  };

  gert::OpImplKernelRegistry::TilingKernelFunc tilingfun_need_atomic_rt2 =
      [](gert::TilingContext *tiling_context) -> graphStatus {
    size_t *workspace_size = tiling_context->GetWorkspaceSizes(1);
    *workspace_size = 1024;
    tiling_context->SetNeedAtomic(true);
    return ge::GRAPH_SUCCESS;
  };

  // remove rt1 tiling
  auto my_add_rt1_tiling = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().at("MyAdd");
  optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().erase("MyAdd");
  // remove rt1 atomic clean tiling
  // add rt2 tiling
  auto funcs = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl("MyAdd");
  funcs->tiling = tilingfun_need_atomic_rt2;
  funcs->tiling_parse = tiling_parse_func_rt2;
  funcs->compile_info_creator = create_compile_info;
  funcs->compile_info_deleter = delete_compile_info;
  auto atomic_op_impl = const_cast<gert::OpImplKernelRegistry::OpImplFunctionsV2 *>(
      gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MemSet"));
  ASSERT_NE(atomic_op_impl, nullptr);
  auto atomic_tiling = atomic_op_impl->tiling;
  auto atomic_tiling_parse = atomic_op_impl->tiling_parse;
  auto atomic_compile_info_creator = atomic_op_impl->compile_info_creator;
  auto atomic_ompile_info_deleter = atomic_op_impl->compile_info_deleter;

  atomic_op_impl->tiling = tilingfun_need_atomic_rt2;
  atomic_op_impl->tiling_parse = tiling_parse_func_rt2;
  atomic_op_impl->compile_info_creator = create_compile_info;
  atomic_op_impl->compile_info_deleter = delete_compile_info;

  BuildAndExecDynamicOfflineModel();
  // recover rt1 tiling
  optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().insert({"MyAdd", my_add_rt1_tiling});
  atomic_op_impl->tiling = atomic_tiling;
  atomic_op_impl->tiling_parse = atomic_tiling_parse;
  atomic_op_impl->compile_info_creator = atomic_compile_info_creator;
  atomic_op_impl->compile_info_deleter = atomic_ompile_info_deleter;
}

TEST_F(DynamicGraphTest, TestDynamicOfflineModel_aicore_task_with_handle) {
  MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
  BuildAndExecDynamicOfflineModel();
}

TEST_F(DynamicGraphTest, TestDynamicOfflineModel_aicore_fallback_to_aicpu) {
  auto func = [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    auto op_desc = node.GetOpDesc();
    GenerateTaskForAiCore(node, context, tasks);
    tasks.emplace_back(AicpuTaskDefBuilder(node).BuildAicpuTask(0));
    AttrUtils::SetBool(op_desc, "partially_supported", true);
    op_desc->SetOpKernelLibName("AIcoreEngine");
    return SUCCESS;
  };
  MockForGenerateTask("AIcoreEngine", func);
  tiling_result_ = false;
  BuildAndExecDynamicOfflineModel();
}

TEST_F(DynamicGraphTest, TestDynamicOnlineInfer) {
  MockForGenerateTask("AIcoreEngine", GenerateTaskForStaticAicore);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);
  DEF_GRAPH(dynamic_graph) {
    auto data_0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto neg = OP_CFG(NEG)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto unique_op = OP_CFG("Unique")
        .InCnt(1)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1});

    CHAIN(NODE("_arg_0", data_0)->NODE("neg", neg)->NODE("unique", unique_op)->NODE("Node_Output", net_output));
  };

  Graph graph = ToGeGraph(dynamic_graph);

  std::map<std::string, std::string> dump_options;
  dump_options.emplace(OPTION_EXEC_ENABLE_DUMP, "1");
  dump_options.emplace(OPTION_EXEC_ENABLE_DUMP_DEBUG, "0");
  dump_options.emplace(OPTION_EXEC_DUMP_PATH, "./");
  dump_options.emplace(OPTION_EXEC_DUMP_STEP, "0|5|10-20");
  dump_options.emplace(OPTION_EXEC_DUMP_MODE, "all");

  std::map<std::string, std::string> graph_options;
  graph_options["ge.outputMaxSize"] = "64";
  EXPECT_ExecuteDynamicOnlineInfer(graph, dump_options, graph_options);
}

TEST_F(DynamicGraphTest, TestDynamicOnlineInferWithType3Aicore) {
  MockForGenerateTask("AIcoreEngine", GenerateTaskForStaticAicore);
  DEF_GRAPH(dynamic_graph) {
    auto data_0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto unique_op = OP_CFG("UniqueV2")
        .InCnt(1)
        .OutCnt(2)
        .Attr(ATTR_NAME_UNKNOWN_SHAPE_TYPE, 3)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16})
        .Build("unique");

    auto func = [](const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo,
                   rtSmDesc_t *smDesc, rtStream_t stream, uint32_t flag, const rtTaskCfgInfo_t *cfgInfo) -> int {
      uintptr_t shape_buffer_addr = reinterpret_cast<uintptr_t *>(argsInfo->args)[3];
      auto shape_buffer = reinterpret_cast<uint32_t *>(shape_buffer_addr);
      shape_buffer[0] = 1;  // 1-dim
      shape_buffer[1] = 8;  // [8]
      return RT_ERROR_NONE;
    };
    auto runtime_stub = std::make_shared<MockRuntime>();
    RuntimeStub::SetInstance(runtime_stub);
    EXPECT_CALL(*runtime_stub, rtKernelLaunchWithFlagV2).WillRepeatedly(testing::Invoke(func));

    unique_op->MutableOutputDesc(0)->SetShape(GeShape({-1}));
    unique_op->MutableOutputDesc(1)->SetDataType(DT_INT32);
    unique_op->MutableOutputDesc(1)->SetShape({});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1});

    CHAIN(NODE("_arg_0", data_0)->NODE(unique_op)->NODE("Node_Output", net_output));
  };

  Graph graph = ToGeGraph(dynamic_graph);
  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);
  EXPECT_ExecuteDynamicOnlineInfer(graph);
}

TEST_F(DynamicGraphTest, TestDynamicOnlineTraining) {
  setenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM", "1", 0);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForStaticAicore);
  MockForGenerateTask("DNN_VM_AICPU", GenerateTaskForAicpuDependRange);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);
  GeTensorDesc tensor_desc(GeShape{});
  GeTensor tensor(tensor_desc);

  GeTensorDesc tensor_desc_1(GeShape(std::vector<int64_t>({1})), FORMAT_ND, DT_STRING);
  std::vector<uint8_t> string_buffer(24, 0);
  GeTensor tensor_1(tensor_desc_1);
  tensor_1.SetData(std::move(string_buffer));

  std::map<std::string, std::string> options;
  Graph graph = BuildDynamicGraphWithAicpu();
  EXPECT_ExecuteDynamicOnlineTrain(graph, {}, options);
  unsetenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM");
}
/** 动态shape模型，支持自定义流pass接入修改unique的stream id，使其单独分流*/
TEST_F(DynamicGraphTest, TestDynamicOnlineTraining_WithCustomPass) {
  setenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM", "1", 0);
  REGISTER_CUSTOM_PASS("DynamicGraphWithAicpuPass")
   .CustomAllocateStreamPassFn(DynamicGraphWithAicpuPass)
   .Stage(CustomPassStage::kAfterAssignLogicStream);

  MockForGenerateTask("AIcoreEngine", GenerateTaskForStaticAicore);
  MockForGenerateTask("DNN_VM_AICPU", GenerateTaskForAicpuDependRange);
  GeTensorDesc tensor_desc(GeShape{});
  GeTensor tensor(tensor_desc);

  GeTensorDesc tensor_desc_1(GeShape(std::vector<int64_t>({1})), FORMAT_ND, DT_STRING);
  std::vector<uint8_t> string_buffer(24, 0);
  GeTensor tensor_1(tensor_desc_1);
  tensor_1.SetData(std::move(string_buffer));

  std::map<std::string, std::string> options;
  Graph graph = BuildDynamicGraphWithAicpu();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->SetName("DynamicGraphWithAicpu");
  DUMP_GRAPH_WHEN("AfterAssignResource");
  EXPECT_ExecuteDynamicOnlineTrain(graph, {}, options);
  unsetenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM");
  CHECK_GRAPH(AfterAssignResource) {
    auto unique = graph->FindNode("unique");
    EXPECT_NE(unique, nullptr);
    EXPECT_EQ(unique->GetOpDesc()->GetStreamId(), 3);
    std::vector<int64_t> receve_id;
    std::vector<int64_t> send_id;
    AttrUtils::GetListInt(unique->GetOpDesc(), ATTR_NAME_RECV_EVENT_IDS, receve_id);
    AttrUtils::GetListInt(unique->GetOpDesc(), ATTR_NAME_SEND_EVENT_IDS, send_id);
    EXPECT_EQ(send_id.size(), 1);
    EXPECT_EQ(receve_id.size(), 1);
  };
}

TEST_F(DynamicGraphTest, TestDynamicOnlineTraining_invalid_ac_parallel_enable) {
  setenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM", "1", 0);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForStaticAicore);
  MockForGenerateTask("DNN_VM_AICPU", GenerateTaskForAicpuDependRange);

  std::map<std::string, std::string> options;
  options[AC_PARALLEL_ENABLE] = "-1";
  Graph graph = BuildDynamicGraphWithAicpu();
  DumpProperties dump_properties;
  dump_properties.SetDumpStatus("on");
  dump_properties.SetDumpMode("all");
  dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
  DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
  std::map<std::string, std::string> session_options = {};
  session_options[OPTION_GRAPH_RUN_MODE] = "1";  // train
  session_options[VARIABLE_MEMORY_MAX_SIZE] = "5000000";
  Session session(session_options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  std::vector<Tensor> inputs = CreateInputTensors(graph);
  std::vector<Tensor> outputs;
  EXPECT_NE(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  session.RemoveGraph(graph_id);
  unsetenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM");
}

TEST_F(DynamicGraphTest, TestDynamicOnlineTraining_ac_parallel_enable) {
  setenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM", "1", 0);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForStaticAicore);
  MockForGenerateTask("DNN_VM_AICPU", GenerateTaskForAicpuDependRange);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);
  std::map<std::string, std::string> options;
  options[AC_PARALLEL_ENABLE] = "1";
  Graph graph = BuildDynamicGraphWithAicpu();
  EXPECT_ExecuteDynamicOnlineTrain(graph, {}, options);
  unsetenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM");
}

TEST_F(DynamicGraphTest, TestDynamicOnlineTraining_Ok_EnableAcParallelWithSingleStream) {
  setenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM", "0", 0);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForStaticAicore);
  MockForGenerateTask("DNN_VM_AICPU", GenerateTaskForAicpuDependRange);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  std::map<std::string, std::string> options;
  options[AC_PARALLEL_ENABLE] = "1";
  Graph graph = BuildDynamicGraphWithAicpu();
  EXPECT_ExecuteDynamicOnlineTrain(graph, {}, options);
  EXPECT_EQ(runtime_stub.GetSlogStub().FindLog(DLOG_INFO, "Enable multi-stream in dynamic graph"), -1);
  unsetenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM");
}

TEST_F(DynamicGraphTest, TestDynamicOnlineTrainingWithNpuGetFloatStatus) {
  MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
  GeTensorDesc tensor_desc(GeShape{});
  GeTensor tensor(tensor_desc);

  GeTensorDesc tensor_desc_1(GeShape(std::vector<int64_t>({1})), FORMAT_ND, DT_STRING);
  std::vector<uint8_t> string_buffer(24, 0);
  GeTensor tensor_1(tensor_desc_1);
  tensor_1.SetData(std::move(string_buffer));

  DEF_GRAPH(dynamic_graph) {
    auto data_0 = OP_CFG(DATA)
	    .InCnt(1)
	    .OutCnt(1)
	    .Attr(ATTR_NAME_INDEX, 0)
	    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1});

    auto npu_get_float_status = OP_CFG("NPUGetFloatStatus")
                                    .InCnt(1)
                                    .OutCnt(1)
                                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1})
                                    .Build("npu_get_float_status");

    auto npu_clear_float_status = OP_CFG("NPUClearFloatStatus")
                                    .InCnt(1)
                                    .OutCnt(1)
                                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1})
                                    .Build("npu_clear_float_status");

    auto add = OP_CFG("MyAdd").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1});

    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1});

    CHAIN(NODE("_arg_0", data_0)->NODE(npu_get_float_status)->NODE(npu_clear_float_status)->NODE("add", add)->NODE("Node_Output", net_output));
  };

  Graph graph = ToGeGraph(dynamic_graph);
  std::map<std::string, std::string> options = {};
  options[OPTION_GRAPH_RUN_MODE] = "1";  // train
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  std::map<AscendString, AscendString> graph_options;
  graph_options[OPTION_EXEC_DYNAMIC_EXECUTE_MODE] = "dynamic_execute";
  graph_options[OPTION_EXEC_DATA_INPUTS_SHAPE_RANGE] = "[1~20,1~30]";

  DumpProperties dump_properties;
  dump_properties.SetDumpStatus("on");
  dump_properties.SetDumpMode("all");
  dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
  DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);
  std::vector<Tensor> inputs;
  inputs.emplace_back(CreateTensor({2, 16}));

  std::vector<Tensor> outputs;
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  session.RemoveGraph(graph_id);
}

TEST_F(DynamicGraphTest, TestDynamicTraining_String_Type) {
  DEF_GRAPH(dynamic_graph) {
    auto data_0 = OP_CFG(DATA)
    .InCnt(1)
    .OutCnt(1)
    .Attr(ATTR_NAME_INDEX, 0)
    .Attr(ATTR_NAME_FORCE_UNKNOWN_SHAPE, true)
    .TensorDesc(FORMAT_ND, DT_STRING, {1});

    auto net_output = OP_CFG(NETOUTPUT)
    .InCnt(1)
    .OutCnt(1)
    .TensorDesc(FORMAT_ND, DT_STRING, {1});

    CHAIN(NODE("_arg_0", data_0)->NODE("Node_Output", net_output));
  };

 Graph graph = ToGeGraph(dynamic_graph);
 EXPECT_ExecuteDynamicOnlineTrain(graph);
}

TEST_F(DynamicGraphTest, TestControlOp_If) {
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);

  auto graph1 = BuildControlOpIfGraph();
  TestRuntimeV2Compile(graph1);
  auto graph2 = BuildControlOpIfGraph();
  char runtime2_env[MMPA_MAX_PATH] = {'1'};
  mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  EXPECT_ExecuteDynamicOnlineTrain(graph2);
}

TEST_F(DynamicGraphTest, TestControlOp_While) {
  char runtime2_env[MMPA_MAX_PATH] = {'1'};
  mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForStaticAicore);

  DEF_GRAPH(cond) {
    auto cond_data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {});
    CHAIN(NODE("cond_data", cond_data)->NODE("cond_Node_Output", net_output));
  };

  GeTensor zero_tensor(GeTensorDesc(GeShape(std::vector<int64_t>{}), FORMAT_ND, DT_INT32));
  zero_tensor.SetData(std::vector<uint8_t>{0, 0, 0, 0});
  DEF_GRAPH(body) {
    auto cond_data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {});

    auto value_data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto const_data = OP_CFG(CONSTANT)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .Attr(ATTR_NAME_WEIGHTS, zero_tensor);

    auto mul = OP_CFG(MUL)
        .InCnt(2)
        .OutCnt(1)
        .Attr("op_para_size", 1)
        .TensorDesc(FORMAT_ND, DT_INT32, {});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(2)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16})
        .Build("body_Node_Output");

    net_output->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({})));
    net_output->MutableOutputDesc(0)->SetDataType(DT_INT32);
    CHAIN(NODE("body_arg_0", cond_data)->NODE("mul", mul)->NODE(net_output));
    CHAIN(NODE("one_tensor", const_data)->NODE("mul", mul));
    CHAIN(NODE("value_data", value_data)->NODE(net_output));
  };

  auto cond_graph = ToComputeGraph(cond);
  auto body_graph = ToComputeGraph(body);

  DEF_GRAPH(while_graph) {
    auto cond_data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {});

    auto value_data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto unique_op = OP_CFG("Unique")
        .InCnt(1)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto while_op = OP_CFG(WHILE)
        .InCnt(2)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1})
        .Build("while_op");

    while_op->MutableInputDesc(0)->SetShape(GeShape(std::vector<int64_t>({})));
    while_op->MutableInputDesc(0)->SetDataType(DT_INT32);
    while_op->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({})));
    while_op->MutableOutputDesc(0)->SetDataType(DT_INT32);
    while_op->RegisterSubgraphIrName("cond", SubgraphType::kStatic);
    while_op->RegisterSubgraphIrName("body", SubgraphType::kStatic);

    while_op->AddSubgraphName(cond_graph->GetName());
    while_op->SetSubgraphInstanceName(0, cond_graph->GetName());
    while_op->AddSubgraphName(body_graph->GetName());
    while_op->SetSubgraphInstanceName(1, body_graph->GetName());

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    CHAIN(NODE("arg_cond", cond_data)->NODE(while_op));
    CHAIN(NODE("arg_value", value_data)->NODE("unique", unique_op)->NODE(while_op)->EDGE(1, 0)->NODE("Node_Output",
                                                                                                     net_output));
  };

  auto root_graph = ToComputeGraph(while_graph);
  auto while_node = root_graph->FindFirstNodeMatchType(WHILE);
  EXPECT_TRUE(while_node != nullptr);
  cond_graph->SetParentNode(while_node);
  cond_graph->SetParentGraph(root_graph);
  body_graph->SetParentNode(while_node);
  body_graph->SetParentGraph(root_graph);
  root_graph->AddSubgraph(cond_graph);
  root_graph->AddSubgraph(body_graph);

  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);

  auto mul_kernel = [](const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo,
                       rtSmDesc_t *smDesc, rtStream_t stream, uint32_t flag, const rtTaskCfgInfo_t *cfgInfo) -> int {
    auto io_addrs = reinterpret_cast<uintptr_t *>(argsInfo->args);
    auto *input_0 = reinterpret_cast<int32_t *>(io_addrs[0]);
    auto *input_1 = reinterpret_cast<int32_t *>(io_addrs[1]);
    auto *output = reinterpret_cast<int32_t *>(io_addrs[2]);
    *output = *input_0 * *input_1;
    return RT_ERROR_NONE;
  };
  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtKernelLaunchWithFlagV2).WillRepeatedly(testing::Invoke(mul_kernel));

  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);

  Shape shape_cond(std::vector<int64_t>{});
  Tensor cond_tensor(TensorDesc(shape_cond, FORMAT_ND, DT_INT32));
  int32_t value = 1;
  cond_tensor.SetData((uint8_t *) &value, sizeof(value));

  uint8_t value_buffer[16 * 4];
  Shape shape_value(std::vector<int64_t>({16}));
  Tensor value_tensor(TensorDesc(shape_value, FORMAT_ND, DT_FLOAT));
  value_tensor.SetData(value_buffer, sizeof(value_buffer));

  std::vector<Tensor> inputs{cond_tensor, value_tensor};
  std::vector<Tensor> outputs;
  // cond->body->cond
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  value = 0;
  cond_tensor.SetData((uint8_t *) &value, sizeof(value));
  // cond
  inputs = {cond_tensor, value_tensor};
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  session.RemoveGraph(graph_id);
}

TEST_F(DynamicGraphTest, TestHostCpu) {
  auto cpu_engine = std::make_shared<CPUProcessNodeEngine>();
  EXPECT_NE(cpu_engine, nullptr);
  ProcessNodeEngineManager::GetInstance().init_flag_ = false;
  auto fn = []()->::ge::ProcessNodeEngine * { return new (std::nothrow) ge::CPUProcessNodeEngine(); };
  EXPECT_EQ(ProcessNodeEngineManager::GetInstance().RegisterEngine("HOST_CPU", cpu_engine, fn), SUCCESS);
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  MockForGenerateTask("DNN_VM_HOST_CPU_OP_STORE", GenerateTaskForHostCpu);
  GeTensorDesc tensor_desc(GeShape{});
  GeTensor tensor(tensor_desc);
  int32_t value = 666;
  tensor.SetData((uint8_t *) &value, sizeof(value));
  DEF_GRAPH(host_cpu_graph) {
    auto var_0 = OP_CFG(VARIABLE)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_VARIABLE_PLACEMENT, "host")
        .TensorDesc(FORMAT_ND, DT_INT32, {});

    auto const_0 = OP_CFG(CONSTANTOP)
        .OutCnt(1)
        .Attr(ATTR_NAME_WEIGHTS, tensor)
        .Attr(ATTR_VARIABLE_PLACEMENT, "host")
        .TensorDesc(FORMAT_ND, DT_INT32, {});

    auto assign = OP_CFG(ASSIGN)
        .InCnt(2)
        .OutCnt(1)
        .Attr(ATTR_VARIABLE_PLACEMENT, "host")
        .TensorDesc(FORMAT_ND, DT_INT32, {});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {});

    CHAIN(NODE("_arg_0", var_0)->NODE("assign", assign)->NODE("Node_Output", net_output));
    CHAIN(NODE("const_0", const_0)->NODE("assign", assign));
  };

  Graph graph = ToGeGraph(host_cpu_graph);
  std::map<AscendString, AscendString> options;
  options[OPTION_GRAPH_RUN_MODE] = "1";  // train
  options["ge.exec.placement"] = "HOST";
  Session session(options);
  GraphId graph_id = 1;

  SharedMemInfo shm_info("_arg_0", 64);
  ASSERT_EQ(HostMemManager::Instance().Initialize(), SUCCESS);
  ASSERT_EQ(HostMemManager::Instance().MallocHostSharedMemory(shm_info), SUCCESS);
  SharedMemInfo shm_info_malloced;
  ASSERT_TRUE(HostMemManager::Instance().QueryVarMemInfo("_arg_0", shm_info_malloced));

  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);

  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  HostCpuEngine::GetInstance().constant_folding_handle_ = mock_host_cpu_handle;
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  session.RemoveGraph(graph_id);

  EXPECT_EQ(outputs.size(), 1);
  auto output = reinterpret_cast<int32_t *>(outputs[0].GetData())[0];
  EXPECT_EQ(output, value);
}

TEST_F(DynamicGraphTest, TestDtResourceHostCpu) {
  auto cpu_engine = std::make_shared<CPUProcessNodeEngine>();
  EXPECT_NE(cpu_engine, nullptr);
  ProcessNodeEngineManager::GetInstance().init_flag_ = false;
  auto fn = []()->::ge::ProcessNodeEngine * { return new (std::nothrow) ge::CPUProcessNodeEngine(); };
  EXPECT_EQ(ProcessNodeEngineManager::GetInstance().RegisterEngine("HOST_CPU", cpu_engine, fn), SUCCESS);
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  MockForGenerateTask("DNN_VM_HOST_CPU_OP_STORE", GenerateTaskForHostCpu);
  GeTensorDesc tensor_desc(GeShape{});
  GeTensor tensor(tensor_desc);
  int32_t value = 666;
  tensor.SetData((uint8_t *) &value, sizeof(value));
  DEF_GRAPH(host_cpu_graph) {
    auto var_0 = OP_CFG(VARIABLE)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_VARIABLE_PLACEMENT, "host")
        .TensorDesc(FORMAT_ND, DT_RESOURCE, {});

    auto const_0 = OP_CFG(CONSTANTOP)
        .OutCnt(1)
        .Attr(ATTR_NAME_WEIGHTS, tensor)
        .Attr(ATTR_VARIABLE_PLACEMENT, "host")
        .TensorDesc(FORMAT_ND, DT_INT32, {});

    auto assign = OP_CFG(ASSIGN)
        .InCnt(2)
        .OutCnt(1)
        .Attr(ATTR_VARIABLE_PLACEMENT, "host")
        .TensorDesc(FORMAT_ND, DT_INT32, {});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {});

    CHAIN(NODE("_arg_0", var_0)->NODE("assign", assign)->NODE("Node_Output", net_output));
    CHAIN(NODE("const_0", const_0)->NODE("assign", assign));
  };

  SharedMemInfo shm_info("_arg_0", 64);
  ASSERT_EQ(HostMemManager::Instance().Initialize(), SUCCESS);
  ASSERT_EQ(HostMemManager::Instance().MallocHostSharedMemory(shm_info), SUCCESS);
  SharedMemInfo shm_info_malloced;
  ASSERT_TRUE(HostMemManager::Instance().QueryVarMemInfo("_arg_0", shm_info_malloced));

  Graph graph = ToGeGraph(host_cpu_graph);
  std::map<AscendString, AscendString> options;
  options[OPTION_GRAPH_RUN_MODE] = "1";  // train
  options["ge.exec.placement"] = "HOST";
  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);

  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  HostCpuEngine::GetInstance().constant_folding_handle_ = mock_host_cpu_handle;
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  session.RemoveGraph(graph_id);

  EXPECT_EQ(outputs.size(), 1);
  auto output = reinterpret_cast<int32_t *>(outputs[0].GetData())[0];
  EXPECT_EQ(output, value);
}

TEST_F(DynamicGraphTest, TestCaseOpAndPartitionedCallExecutor) {
  char runtime2_env[MMPA_MAX_PATH] = {'1'};
  mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
  DEF_GRAPH(partitioned_call) {
    auto cond_data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto neg = OP_CFG("MyNeg")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    CHAIN(NODE("partitioned_call_data", cond_data)->NODE("neg", neg)->NODE("partitioned_call_Node_Output",
                                                                           net_output));
  };
  auto sub_graph = ToComputeGraph(partitioned_call);

  DEF_GRAPH(branch_0) {
    auto data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    CHAIN(NODE("branch_0_arg_0", data)->NODE("branch_0_Node_Output", net_output));
  };

  DEF_GRAPH(branch_1) {
    auto data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto partitioned_call_op = OP_CFG(PARTITIONEDCALL)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1})
        .Build("partitioned_call_op");

    partitioned_call_op->RegisterSubgraphIrName("f", SubgraphType::kStatic);
    partitioned_call_op->AddSubgraphName(sub_graph->GetName());
    partitioned_call_op->SetSubgraphInstanceName(0, sub_graph->GetName());

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    CHAIN(NODE("branch_1_arg_0", data)->NODE(partitioned_call_op)->NODE("branch_1_Node_Output", net_output));
  };

  auto sub_graph_b0 = ToComputeGraph(branch_0);
  auto sub_graph_b1 = ToComputeGraph(branch_1);

  auto partitioned_call_node = sub_graph_b1->FindFirstNodeMatchType(PARTITIONEDCALL);
  EXPECT_TRUE(partitioned_call_node != nullptr);
  sub_graph->SetParentNode(partitioned_call_node);
  sub_graph->SetParentGraph(sub_graph_b1);

  DEF_GRAPH(case_graph) {
    auto arg_branch_index = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {});

    auto arg_value = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto unique_op = OP_CFG("Unique")
        .InCnt(1)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto case_op = OP_CFG(CASE)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1})
        .Build("case_op");

    case_op->RegisterSubgraphIrName("branches", SubgraphType::kDynamic);
    case_op->AddSubgraphName(sub_graph_b0->GetName());
    case_op->SetSubgraphInstanceName(0, sub_graph_b0->GetName());
    case_op->AddSubgraphName(sub_graph_b1->GetName());
    case_op->SetSubgraphInstanceName(1, sub_graph_b1->GetName());

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    CHAIN(NODE("arg_branch_index", arg_branch_index)->NODE(case_op)->NODE("Node_Output", net_output));
    CHAIN(NODE("arg_value", arg_value)->NODE("unique_op", unique_op)->NODE(case_op));
  };

  auto root_graph = ToComputeGraph(case_graph);
  auto case_node = root_graph->FindFirstNodeMatchType(CASE);
  EXPECT_TRUE(case_node != nullptr);
  sub_graph_b0->SetParentNode(case_node);
  sub_graph_b0->SetParentGraph(root_graph);
  sub_graph_b1->SetParentNode(case_node);
  sub_graph_b1->SetParentGraph(root_graph);
  EXPECT_EQ(root_graph->AddSubgraph(sub_graph_b0), GRAPH_SUCCESS);
  EXPECT_EQ(root_graph->AddSubgraph(sub_graph_b1), GRAPH_SUCCESS);
  EXPECT_EQ(root_graph->AddSubgraph(sub_graph), GRAPH_SUCCESS);

  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);

  std::map<AscendString, AscendString> options;
  options[OPTION_GRAPH_RUN_MODE] = "1";  // train
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);

  Shape shape_index(std::vector<int64_t>({}));
  int32_t branch = 0;
  TensorDesc tensor_desc_index(shape_index);
  Tensor input_0(tensor_desc_index);
  input_0.SetData((uint8_t *) &branch, sizeof(branch));

  Shape shape_value(std::vector<int64_t>({16}));
  TensorDesc tensor_desc_value(shape_value);
  Tensor input_1(tensor_desc_value);
  uint8_t buffer[16 * sizeof(float)];
  input_1.SetData((uint8_t *) &buffer, sizeof(buffer));

  // taking branch 0
  std::vector<Tensor> inputs{input_0, input_1};
  std::vector<Tensor> outputs;
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);

  // taking branch 1
  branch = 1;
  inputs[0].SetData((uint8_t *) &branch, sizeof(branch));
  outputs.clear();
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  // cover muting workspace count
  tiling_run_info_.SetWorkspaces({16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16});
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  session.RemoveGraph(graph_id);
}

TEST_F(DynamicGraphTest, TestSubGraphHostCpuNode) {
  char runtime2_env[MMPA_MAX_PATH] = {'1'};
  mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
  DEF_GRAPH(partitioned_call) {
    auto cond_data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto neg = OP_CFG("MyNeg")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("partitioned_call_data", cond_data)->NODE("neg", neg)->NODE("partitioned_call_Node_Output",
                                                                           net_output));
  };
  auto sub_graph = ToComputeGraph(partitioned_call);

  DEF_GRAPH(branch_1) {
    auto data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto partitioned_call_op = OP_CFG(PARTITIONEDCALL)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16})
        .Build("partitioned_call_op");

    partitioned_call_op->RegisterSubgraphIrName("f", SubgraphType::kStatic);
    partitioned_call_op->AddSubgraphName(sub_graph->GetName());
    partitioned_call_op->SetSubgraphInstanceName(0, sub_graph->GetName());

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("branch_0_arg_0", data)->NODE(partitioned_call_op)->NODE("branch_0_Node_Output", net_output));
  };

  auto sub_graph_b1 = ToComputeGraph(branch_1);

  auto partitioned_call_node = sub_graph_b1->FindFirstNodeMatchType(PARTITIONEDCALL);
  EXPECT_TRUE(partitioned_call_node != nullptr);
  sub_graph->SetParentNode(partitioned_call_node);
  sub_graph->SetParentGraph(sub_graph_b1);
auto myneg_op = sub_graph->FindFirstNodeMatchType("MyNeg");
  myneg_op->GetOpDesc()->SetOpEngineName("DNN_VM_HOST_CPU");

  DEF_GRAPH(graph1) {
    auto arg_branch_index = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto arg_value = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto unique_op = OP_CFG("Unique")
        .InCnt(1)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto partitioned_call_op2 = OP_CFG(PARTITIONEDCALL)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16})
        .Build("partitioned_call_op2");

    partitioned_call_op2->AddSubgraphName(sub_graph_b1->GetName());
    partitioned_call_op2->SetSubgraphInstanceName(0, sub_graph_b1->GetName());

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    CHAIN(NODE("arg_branch_index", arg_branch_index)->NODE(partitioned_call_op2)->NODE("Node_Output", net_output));
    CHAIN(NODE("arg_value", arg_value)->NODE("unique_op", unique_op)->NODE(partitioned_call_op2));
  };

  auto root_graph = ToComputeGraph(graph1);
  auto node_partitioned_call_op2 = root_graph->FindFirstNodeMatchType(PARTITIONEDCALL);
  EXPECT_TRUE(node_partitioned_call_op2 != nullptr);
  sub_graph_b1->SetParentNode(node_partitioned_call_op2);
  sub_graph_b1->SetParentGraph(root_graph);
  EXPECT_EQ(root_graph->AddSubgraph(sub_graph_b1), GRAPH_SUCCESS);
  EXPECT_EQ(root_graph->AddSubgraph(sub_graph), GRAPH_SUCCESS);

  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  std::map<AscendString, AscendString> options;
  options[OPTION_GRAPH_RUN_MODE] = "1";  // train
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);

  Shape shape_index(std::vector<int64_t>({16}));
  TensorDesc tensor_desc_index(shape_index);
  Tensor input_0(tensor_desc_index);
  uint8_t buffer0[16 * sizeof(float)];
  input_0.SetData((uint8_t *) &buffer0, sizeof(buffer0));

  Shape shape_value(std::vector<int64_t>({16}));
  TensorDesc tensor_desc_value(shape_value);
  Tensor input_1(tensor_desc_value);
  uint8_t buffer[16 * sizeof(float)];
  input_1.SetData((uint8_t *) &buffer, sizeof(buffer));

  std::vector<Tensor> inputs{input_0, input_1};
  std::vector<Tensor> outputs;
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  EXPECT_EQ(sub_graph->GetGraphUnknownFlag(), true);
}

TEST_F(DynamicGraphTest, TestSubGraphForceUnKnownShape) {
  char runtime2_env[MMPA_MAX_PATH] = {'1'};
  mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
  DEF_GRAPH(partitioned_call) {
    auto cond_data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto neg = OP_CFG("MyNeg")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("partitioned_call_data", cond_data)->NODE("neg", neg)->NODE("partitioned_call_Node_Output",
                                                                           net_output));
  };
  auto sub_graph = ToComputeGraph(partitioned_call);

  DEF_GRAPH(branch_1) {
    auto data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto partitioned_call_op = OP_CFG(PARTITIONEDCALL)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16})
        .Build("partitioned_call_op");

    partitioned_call_op->RegisterSubgraphIrName("f", SubgraphType::kStatic);
    partitioned_call_op->AddSubgraphName(sub_graph->GetName());
    partitioned_call_op->SetSubgraphInstanceName(0, sub_graph->GetName());

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("branch_0_arg_0", data)->NODE(partitioned_call_op)->NODE("branch_0_Node_Output", net_output));
  };

  auto sub_graph_b1 = ToComputeGraph(branch_1);

  auto partitioned_call_node = sub_graph_b1->FindFirstNodeMatchType(PARTITIONEDCALL);
  EXPECT_TRUE(partitioned_call_node != nullptr);
  sub_graph->SetParentNode(partitioned_call_node);
  sub_graph->SetParentGraph(sub_graph_b1);
  auto myneg_op = sub_graph->FindFirstNodeMatchType("MyNeg");
  AttrUtils::SetBool(myneg_op->GetOpDesc(), ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);

  DEF_GRAPH(graph1) {
    auto arg_branch_index = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto arg_value = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto unique_op = OP_CFG("Unique")
        .InCnt(1)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto partitioned_call_op2 = OP_CFG(PARTITIONEDCALL)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16})
        .Build("partitioned_call_op2");

    partitioned_call_op2->AddSubgraphName(sub_graph_b1->GetName());
    partitioned_call_op2->SetSubgraphInstanceName(0, sub_graph_b1->GetName());

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    CHAIN(NODE("arg_branch_index", arg_branch_index)->NODE(partitioned_call_op2)->NODE("Node_Output", net_output));
    CHAIN(NODE("arg_value", arg_value)->NODE("unique_op", unique_op)->NODE(partitioned_call_op2));
  };

  auto root_graph = ToComputeGraph(graph1);
  auto node_partitioned_call_op2 = root_graph->FindFirstNodeMatchType(PARTITIONEDCALL);
  EXPECT_TRUE(node_partitioned_call_op2 != nullptr);
  sub_graph_b1->SetParentNode(node_partitioned_call_op2);
  sub_graph_b1->SetParentGraph(root_graph);
  EXPECT_EQ(root_graph->AddSubgraph(sub_graph_b1), GRAPH_SUCCESS);
  EXPECT_EQ(root_graph->AddSubgraph(sub_graph), GRAPH_SUCCESS);

  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  std::map<AscendString, AscendString> options;
  options[OPTION_GRAPH_RUN_MODE] = "1";  // train
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
  GraphId graph_id = 1;
  std::map<AscendString, AscendString> graph_option;
  options[OPTION_STATIC_MODEL_OPS_LOWER_LIMIT] = "4";
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);

  Shape shape_index(std::vector<int64_t>({16}));
  TensorDesc tensor_desc_index(shape_index);
  Tensor input_0(tensor_desc_index);
  uint8_t buffer0[16 * sizeof(float)];
  input_0.SetData((uint8_t *) &buffer0, sizeof(buffer0));

  Shape shape_value(std::vector<int64_t>({16}));
  TensorDesc tensor_desc_value(shape_value);
  Tensor input_1(tensor_desc_value);
  uint8_t buffer[16 * sizeof(float)];
  input_1.SetData((uint8_t *) &buffer, sizeof(buffer));

  std::vector<Tensor> inputs{input_0, input_1};
  std::vector<Tensor> outputs;
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  EXPECT_EQ(sub_graph->GetGraphUnknownFlag(), true);
}

TEST_F(DynamicGraphTest, TestSingleOpWithSubGraph) {
  char runtime2_env[MMPA_MAX_PATH] = {'1'};
  mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
  DEF_GRAPH(partitioned_call) {
    auto cond_data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto neg = OP_CFG("MyNeg")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("partitioned_call_data", cond_data)->NODE("neg", neg)->NODE("partitioned_call_Node_Output",
                                                                           net_output));
  };
  auto sub_graph = ToComputeGraph(partitioned_call);

  DEF_GRAPH(branch_1) {
    auto data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto partitioned_call_op = OP_CFG(PARTITIONEDCALL)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16})
        .Build("partitioned_call_op");

    partitioned_call_op->RegisterSubgraphIrName("f", SubgraphType::kStatic);
    partitioned_call_op->AddSubgraphName(sub_graph->GetName());
    partitioned_call_op->SetSubgraphInstanceName(0, sub_graph->GetName());

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("branch_0_arg_0", data)->NODE(partitioned_call_op)->NODE("branch_0_Node_Output", net_output));
  };

  auto sub_graph_b1 = ToComputeGraph(branch_1);

  auto partitioned_call_node = sub_graph_b1->FindFirstNodeMatchType(PARTITIONEDCALL);
  EXPECT_TRUE(partitioned_call_node != nullptr);
  sub_graph->SetParentNode(partitioned_call_node);
  sub_graph->SetParentGraph(sub_graph_b1);
  auto myneg_op = sub_graph->FindFirstNodeMatchType("MyNeg");

  DEF_GRAPH(graph1) {
    auto arg_branch_index = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    auto arg_value = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto unique_op = OP_CFG("Unique")
        .InCnt(1)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto partitioned_call_op2 = OP_CFG(PARTITIONEDCALL)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16})
        .Build("partitioned_call_op2");

    partitioned_call_op2->AddSubgraphName(sub_graph_b1->GetName());
    partitioned_call_op2->SetSubgraphInstanceName(0, sub_graph_b1->GetName());

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    CHAIN(NODE("arg_branch_index", arg_branch_index)->NODE(partitioned_call_op2)->NODE("Node_Output", net_output));
    CHAIN(NODE("arg_value", arg_value)->NODE("unique_op", unique_op)->NODE(partitioned_call_op2));
  };

  auto root_graph = ToComputeGraph(graph1);
  (void)AttrUtils::SetBool(root_graph, ATTR_SINGLE_OP_SCENE, true);
  auto node_partitioned_call_op2 = root_graph->FindFirstNodeMatchType(PARTITIONEDCALL);
  EXPECT_TRUE(node_partitioned_call_op2 != nullptr);
  sub_graph_b1->SetParentNode(node_partitioned_call_op2);
  sub_graph_b1->SetParentGraph(root_graph);
  EXPECT_EQ(root_graph->AddSubgraph(sub_graph_b1), GRAPH_SUCCESS);
  EXPECT_EQ(root_graph->AddSubgraph(sub_graph), GRAPH_SUCCESS);

  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  std::map<AscendString, AscendString> options;
  options[OPTION_GRAPH_RUN_MODE] = "1";  // train
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);

  Shape shape_index(std::vector<int64_t>({16}));
  TensorDesc tensor_desc_index(shape_index);
  Tensor input_0(tensor_desc_index);
  uint8_t buffer0[16 * sizeof(float)];
  input_0.SetData((uint8_t *) &buffer0, sizeof(buffer0));

  Shape shape_value(std::vector<int64_t>({16}));
  TensorDesc tensor_desc_value(shape_value);
  Tensor input_1(tensor_desc_value);
  uint8_t buffer[16 * sizeof(float)];
  input_1.SetData((uint8_t *) &buffer, sizeof(buffer));

  std::vector<Tensor> inputs{input_0, input_1};
  std::vector<Tensor> outputs;
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  EXPECT_EQ(sub_graph->GetGraphUnknownFlag(), true);
  EXPECT_EQ(sub_graph_b1->GetGraphUnknownFlag(), true);
}

TEST_F(DynamicGraphTest, TestAicpuKernels) {
  auto mock_memcpy = [](void *dst, uint64_t dest_max, const void *src, uint64_t count, rtMemcpyKind_t kind) -> int {
    if (count == 0) {
      return RT_ERROR_NONE;
    }
    if (count == sizeof(aicpu::FWKAdapter::ResultSummary) && kind == RT_MEMCPY_DEVICE_TO_HOST) {
      aicpu::FWKAdapter::ResultSummary summary{};
      summary.shape_data_size = 8;
      summary.raw_data_size = 4;
      return memcpy_s(dst, dest_max, &summary, count);
    } else {
      return memcpy_s(dst, dest_max, src, count);
    }
  };
  auto runtime_stub = std::make_shared<MockMemcpy>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtMemcpy).WillRepeatedly(testing::Invoke(mock_memcpy));

  auto generate_aicpu_type_4_kernels =
      [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
        tasks.emplace_back(AicpuTaskDefBuilder(node).BuildAicpuTask(4));
        tasks.emplace_back(AicpuTaskDefBuilder(node).BuildAicpuTask(0));
        AttrUtils::SetInt(node.GetOpDesc(), ATTR_NAME_UNKNOWN_SHAPE_TYPE, 4);
        return SUCCESS;
      };
  MockForGenerateTask("aicpu_ascend_kernel", generate_aicpu_type_4_kernels);
  BuildAndExecDynamicOnlineModel();

  auto generate_tf_type_4_kernels =
      [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
        tasks.emplace_back(AicpuTaskDefBuilder(node).BuildTfTask(4));
        tasks.emplace_back(AicpuTaskDefBuilder(node).BuildTfTask(0));
        AttrUtils::SetInt(node.GetOpDesc(), ATTR_NAME_UNKNOWN_SHAPE_TYPE, 4);
        return SUCCESS;
      };
  MockForGenerateTask("aicpu_tf_kernel", generate_tf_type_4_kernels);
  BuildAndExecDynamicOnlineModel();

  auto generate_aicpu_type_3_kernel =
      [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
        tasks.emplace_back(AicpuTaskDefBuilder(node).BuildAicpuTask(3));
        return SUCCESS;
      };
  MockForGenerateTask("aicpu_ascend_kernel", generate_aicpu_type_3_kernel);
  BuildAndExecDynamicOnlineModel();

  auto generate_tf_type_3_kernel =
      [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
        tasks.emplace_back(AicpuTaskDefBuilder(node).BuildTfTask(3));
        return SUCCESS;
      };
  MockForGenerateTask("aicpu_tf_kernel", generate_tf_type_3_kernel);
  BuildAndExecDynamicOnlineModel();

  auto generate_aicpu_type_3_kernel_blocking =
      [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
        tasks.emplace_back(AicpuTaskDefBuilder(node).BuildAicpuTask(3));
        AttrUtils::SetBool(node.GetOpDesc(), ATTR_NAME_IS_BLOCKING_OP, true);
        AttrUtils::SetBool(node.GetOpDesc(), ATTR_NAME_IS_BLOCKING_OP, true);
        AttrUtils::SetInt(node.GetOpDesc(), ATTR_NAME_BLOCKDIM_INDEX, 1);
        return SUCCESS;
      };
  MockForGenerateTask("aicpu_ascend_kernel", generate_aicpu_type_3_kernel_blocking);
  BuildAndExecDynamicOnlineModel();

  auto generate_tf_type_3_kernel_blocking =
      [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
        tasks.emplace_back(AicpuTaskDefBuilder(node).BuildTfTask(3));
        AttrUtils::SetBool(node.GetOpDesc(), ATTR_NAME_IS_BLOCKING_OP, true);
        AttrUtils::SetBool(node.GetOpDesc(), ATTR_NAME_IS_BLOCKING_OP, true);
        AttrUtils::SetInt(node.GetOpDesc(), ATTR_NAME_BLOCKDIM_INDEX, 1);
        return SUCCESS;
      };
  MockForGenerateTask("aicpu_tf_kernel", generate_tf_type_3_kernel_blocking);
  BuildAndExecDynamicOnlineModel();

  const char_t *const kEnvOverFlowPath = "ACL_ERROR_RT_OVER_FLOW_ST";
  char_t over_flow_path[MMPA_MAX_PATH] = "over_st_flow";
  mmSetEnv(kEnvOverFlowPath, &over_flow_path[0U], MMPA_MAX_PATH);
  auto generate_tf_type_3_kernel_check_overflow =
      [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
        tasks.emplace_back(AicpuTaskDefBuilder(node).BuildTfTask(3));
        return SUCCESS;
      };
  MockForGenerateTask("aicpu_tf_kernel", generate_tf_type_3_kernel_check_overflow);
  BuildAndExecDynamicOnlineModelExp(SUCCESS);
  unsetenv(kEnvOverFlowPath);

  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(DynamicGraphTest, TesthostAicpuKernels) {
  const uint32_t topic_type_flag = RT_KERNEL_HOST_ONLY;
  auto mock_memcpy = [](void *dst, uint64_t dest_max, const void *src, uint64_t count, rtMemcpyKind_t kind) -> int {
    if (count == 0) {
      return RT_ERROR_NONE;
    }
    if (count == sizeof(aicpu::FWKAdapter::ResultSummary) && kind == RT_MEMCPY_HOST_TO_HOST) {
      aicpu::FWKAdapter::ResultSummary summary{};
      summary.shape_data_size = 8;
      summary.raw_data_size = 4;
      return memcpy_s(dst, dest_max, &summary, count);
    } else {
      return memcpy_s(dst, dest_max, src, count);
    }
  };
  auto runtime_stub = std::make_shared<MockMemcpy>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtMemcpy).WillRepeatedly(testing::Invoke(mock_memcpy));

  auto generate_aicpu_type_4_kernels = [](const Node &node, RunContext &context,
                                          std::vector<domi::TaskDef> &tasks) -> Status {
    tasks.emplace_back(AicpuTaskDefBuilder(node).BuildAicpuTask(4, topic_type_flag));
    tasks.emplace_back(AicpuTaskDefBuilder(node).BuildAicpuTask(0, topic_type_flag));
    AttrUtils::SetInt(node.GetOpDesc(), ATTR_NAME_UNKNOWN_SHAPE_TYPE, 4);
    return SUCCESS;
  };
  MockForGenerateTask("aicpu_ascend_kernel", generate_aicpu_type_4_kernels);
  BuildAndExecDynamicOnlineModel();

  auto generate_tf_type_4_kernels = [](const Node &node, RunContext &context,
                                       std::vector<domi::TaskDef> &tasks) -> Status {
    tasks.emplace_back(AicpuTaskDefBuilder(node).BuildTfTask(4, topic_type_flag));
    tasks.emplace_back(AicpuTaskDefBuilder(node).BuildTfTask(0, topic_type_flag));
    AttrUtils::SetInt(node.GetOpDesc(), ATTR_NAME_UNKNOWN_SHAPE_TYPE, 4);
    return SUCCESS;
  };
  MockForGenerateTask("aicpu_ascend_kernel", generate_tf_type_4_kernels);
  MockForGenerateTask("aicpu_tf_kernel", generate_tf_type_4_kernels);
  BuildAndExecDynamicOnlineModel();

  auto generate_aicpu_type_3_kernel = [](const Node &node, RunContext &context,
                                         std::vector<domi::TaskDef> &tasks) -> Status {
    tasks.emplace_back(AicpuTaskDefBuilder(node).BuildAicpuTask(3, topic_type_flag));
    return SUCCESS;
  };
  MockForGenerateTask("aicpu_ascend_kernel", generate_aicpu_type_3_kernel);
  BuildAndExecDynamicOnlineModel();

  auto generate_tf_type_3_kernel = [](const Node &node, RunContext &context,
                                      std::vector<domi::TaskDef> &tasks) -> Status {
    tasks.emplace_back(AicpuTaskDefBuilder(node).BuildTfTask(3, topic_type_flag));
    return SUCCESS;
  };
  MockForGenerateTask("aicpu_ascend_kernel", generate_tf_type_3_kernel);
  MockForGenerateTask("aicpu_tf_kernel", generate_tf_type_3_kernel);
  BuildAndExecDynamicOnlineModel();

  auto generate_aicpu_type_3_kernel_blocking = [](const Node &node, RunContext &context,
                                                  std::vector<domi::TaskDef> &tasks) -> Status {
    tasks.emplace_back(AicpuTaskDefBuilder(node).BuildAicpuTask(3, topic_type_flag));
    AttrUtils::SetBool(node.GetOpDesc(), ATTR_NAME_IS_BLOCKING_OP, true);
    AttrUtils::SetBool(node.GetOpDesc(), ATTR_NAME_IS_BLOCKING_OP, true);
    AttrUtils::SetInt(node.GetOpDesc(), ATTR_NAME_BLOCKDIM_INDEX, 1);
    return SUCCESS;
  };
  MockForGenerateTask("aicpu_ascend_kernel", generate_aicpu_type_3_kernel_blocking);
  BuildAndExecDynamicOnlineModel();

  auto generate_tf_type_3_kernel_blocking = [](const Node &node, RunContext &context,
                                               std::vector<domi::TaskDef> &tasks) -> Status {
    tasks.emplace_back(AicpuTaskDefBuilder(node).BuildTfTask(3, topic_type_flag));
    AttrUtils::SetBool(node.GetOpDesc(), ATTR_NAME_IS_BLOCKING_OP, true);
    AttrUtils::SetBool(node.GetOpDesc(), ATTR_NAME_IS_BLOCKING_OP, true);
    AttrUtils::SetInt(node.GetOpDesc(), ATTR_NAME_BLOCKDIM_INDEX, 1);
    return SUCCESS;
  };
  MockForGenerateTask("aicpu_ascend_kernel", generate_tf_type_3_kernel_blocking);
  MockForGenerateTask("aicpu_tf_kernel", generate_tf_type_3_kernel_blocking);
  BuildAndExecDynamicOnlineModel();
}

TEST_F(DynamicGraphTest, TestType2AndGeLocal) {
  auto graph1 = BuildType2AndGeLocal();
  TestRuntimeV2Compile(graph1);

  DEF_GRAPH(graph_def) {
    auto var = OP_CFG(VARIABLE)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op = OP_CFG("FakeType2Op")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16}).Build("fake_type2_op");

    auto shape_op = OP_CFG(SHAPE)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {1});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("var", var)
              ->NODE(fake_type2_op)
              ->NODE("shape", shape_op)
              ->NODE("Node_Output", net_output));
  };

  auto graph = ToGeGraph(graph_def);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
  EXPECT_ExecuteDynamicOnlineTrain(graph);
}

TEST_F(DynamicGraphTest, TestInferShapeForSubgraph) {
  DEF_GRAPH(fused_subgraph) {
    auto data_0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto fake_type2_op = OP_CFG("FakeType2Op")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16})
        .Build("fake_type2_op");

    fake_type2_op->SetOpInferDepends({"__input0"});
    fake_type2_op->SetOpEngineName("AIcoreEngine");
    fake_type2_op->SetOpKernelLibName("AIcoreEngine");  // fake op can not do that?

    auto ret_val = OP_CFG("_RetVal")
        .InCnt(1)
        .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("_arg_0", data_0)->NODE(fake_type2_op)->NODE("ret_val", ret_val));
  };

  DEF_GRAPH(dynamic_graph) {
    auto data_0 = OP_CFG(VARIABLE)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op = OP_CFG("FakeType2Op")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("_arg_0", data_0)->NODE("fused_op", fake_type2_op)->NODE("Node_Output", net_output));
  };

  auto org_graph = ToComputeGraph(fused_subgraph);
  auto root_graph = ToComputeGraph(dynamic_graph);
  auto add_node = root_graph->FindNode("fused_op");
  EXPECT_TRUE(add_node != nullptr);
  AttrUtils::SetGraph(add_node->GetOpDesc(), "_original_fusion_graph", org_graph);
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
  EXPECT_ExecuteDynamicOnlineTrain(graph);
}

TEST_F(DynamicGraphTest, TestDynamicInput) {
  MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
  auto recover_ir_graph = BuildDynamicInputGraph();
  std::map<AscendString, AscendString> options;
  options[OPTION_GRAPH_RUN_MODE] = "1";  // train
  options[OPTION_EXEC_ENABLE_DUMP_DEBUG] = "1";
  options[OPTION_EXEC_DUMP_PATH] = "./";
  options[OPTION_EXEC_DUMP_DEBUG_MODE] = "aicore_overflow"; // OP_DEBUG_ATOMIC /  OP_DEBUG_ALL
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  options["ge.compile_dynamic_mode"] = "1";
  std::map<AscendString, AscendString> graph_options;

  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, recover_ir_graph, graph_options), SUCCESS);

  std::vector<Tensor> inputs;
  inputs.emplace_back(CreateTensor({2, 16}));
  inputs.emplace_back(CreateTensor({2, 16}));
  inputs.emplace_back(CreateTensor({3, 12, 5, 5}));
  std::vector<Tensor> outputs;
  setenv("HYBRID_PROFILING_LEVEL", "1", 1);
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    ComputeGraphPtr compute_graph = GraphUtilsEx::GetComputeGraph(recover_ir_graph);
    auto arg_node = compute_graph->FindNode("_arg_0");
    EXPECT_NE(arg_node, nullptr);
    auto ir_inputs = arg_node->GetOpDesc()->GetIrInputs();
    auto ir_attr_names = arg_node->GetOpDesc()->GetIrAttrNames();
    const std::vector<std::string> target_ir_input;
    const std::vector<std::string> target_ir_attr_name;
    EXPECT_EQ(ir_inputs.size(), target_ir_input.size());
    for (size_t i = 0U; i < ir_inputs.size(); ++i) {
      EXPECT_EQ(ir_inputs[i].first, target_ir_input[i]);
    }
    EXPECT_EQ(ir_attr_names.size(), target_ir_attr_name.size());
    for (size_t i = 0U; i < ir_attr_names.size(); ++i) {
      EXPECT_EQ(ir_attr_names[i], target_ir_attr_name[i]);
    }
  };

  EXPECT_EQ(RunGraphAsync(session, graph_id, inputs, outputs), SUCCESS);
  unsetenv("HYBRID_PROFILING_LEVEL");
  session.RemoveGraph(graph_id);
}

TEST_F(DynamicGraphTest, TestDynamicInput_fail) {
  MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
  auto recover_ir_graph = BuildDynamicInputGraph();
  std::map<AscendString, AscendString> options;
  options[OPTION_GRAPH_RUN_MODE] = "1";  // train
  options[OPTION_EXEC_ENABLE_DUMP_DEBUG] = "1";
  options[OPTION_EXEC_DUMP_PATH] = "./";
  options[OPTION_EXEC_DUMP_DEBUG_MODE] = "aicore_overflow"; // OP_DEBUG_ATOMIC /  OP_DEBUG_ALL
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  options[JIT_COMPILE.c_str()] = "1";
  std::map<AscendString, AscendString> graph_options;

  graph_options[OPTION_EXEC_DYNAMIC_EXECUTE_MODE] = "dynamic_execute";
  graph_options[OPTION_EXEC_DATA_INPUTS_SHAPE_RANGE] = "[10~10],[1~10]";
  graph_options[OPTION_EXEC_ENABLE_EXCEPTION_DUMP] = "1";
  graph_options[JIT_COMPILE.c_str()] = "1";

  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, recover_ir_graph, graph_options), SUCCESS);

  std::vector<Tensor> inputs;
  inputs.emplace_back(CreateTensor({3, 12, 5, 5}));
  inputs.emplace_back(CreateTensor({2, 16}));
  std::vector<Tensor> outputs;
  setenv("HYBRID_PROFILING_LEVEL", "1", 1);
  EXPECT_NE(session.BuildGraph(graph_id, inputs), SUCCESS);
  EXPECT_NE(RunGraphAsync(session, graph_id, inputs, outputs), SUCCESS);
  unsetenv("HYBRID_PROFILING_LEVEL");

  std::vector<Tensor> inputs1;
  inputs1.emplace_back(CreateTensor({12}));
  inputs1.emplace_back(CreateTensor({12}));
  std::vector<Tensor> outputs1;
  setenv("HYBRID_PROFILING_LEVEL", "1", 1);
  EXPECT_NE(session.BuildGraph(graph_id, inputs1), SUCCESS);
  EXPECT_NE(RunGraphAsync(session, graph_id, inputs1, outputs1), SUCCESS);
  unsetenv("HYBRID_PROFILING_LEVEL");
  session.RemoveGraph(graph_id);

  std::map<AscendString, AscendString> graph_options_second;
  graph_options_second[OPTION_EXEC_DYNAMIC_EXECUTE_MODE] = "dynamic_execute";
  graph_options_second[OPTION_EXEC_DATA_INPUTS_SHAPE_RANGE] = "[1~10],[1~10]";
  graph_options_second[OPTION_EXEC_ENABLE_EXCEPTION_DUMP] = "1";
  graph_options[JIT_COMPILE.c_str()] = "1";
  GraphId graph_id_second = 2;
  auto recover_ir_graph_second = BuildDynamicInputGraph();
  EXPECT_EQ(session.AddGraph(graph_id_second, recover_ir_graph_second, graph_options_second), SUCCESS);
  std::vector<Tensor> inputs_second;
  inputs_second.emplace_back(CreateTensor({12}));
  inputs_second.emplace_back(CreateTensor({12}));
  std::vector<Tensor> outputs_second;
  setenv("HYBRID_PROFILING_LEVEL", "1", 1);
  EXPECT_NE(session.BuildGraph(graph_id_second, inputs_second), SUCCESS);
  EXPECT_NE(RunGraphAsync(session, graph_id_second, inputs_second, outputs_second), SUCCESS);
  unsetenv("HYBRID_PROFILING_LEVEL");
  session.RemoveGraph(graph_id_second);
}

TEST_F(DynamicGraphTest, TestDynamicInputWithKnownNodeFirstStrategy) {
  OpsKernelManager::GetInstance().enable_ffts_flag_ = true;
  MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
  auto recover_ir_graph = BuildDynamicInputGraph();
  std::map<AscendString, AscendString> options;
  options[OPTION_GRAPH_RUN_MODE] = "1";  // train
  options[OPTION_EXEC_ENABLE_DUMP_DEBUG] = "1";
  options[OPTION_EXEC_DUMP_PATH] = "./";
  options[OPTION_EXEC_DUMP_DEBUG_MODE] = "aicore_overflow"; // OP_DEBUG_ATOMIC /  OP_DEBUG_ALL
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  std::map<AscendString, AscendString> graph_options;

  graph_options[OPTION_EXEC_DYNAMIC_EXECUTE_MODE] = "dynamic_execute";
  graph_options[OPTION_EXEC_DATA_INPUTS_SHAPE_RANGE] = "[1~20,1~30],[1~20,1~30]";
  graph_options[OPTION_EXEC_ENABLE_EXCEPTION_DUMP] = "1";

  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, recover_ir_graph, graph_options), SUCCESS);

  std::vector<Tensor> inputs;
  inputs.emplace_back(CreateTensor({2, 16}));
  inputs.emplace_back(CreateTensor({2, 16}));
  std::vector<Tensor> outputs;
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    ComputeGraphPtr compute_graph = GraphUtilsEx::GetComputeGraph(recover_ir_graph);
    auto arg_node = compute_graph->FindNode("_arg_0");
    EXPECT_NE(arg_node, nullptr);
    auto ir_inputs = arg_node->GetOpDesc()->GetIrInputs();
    auto ir_attr_names = arg_node->GetOpDesc()->GetIrAttrNames();
    const std::vector<std::string> target_ir_input;
    const std::vector<std::string> target_ir_attr_name;
    EXPECT_EQ(ir_inputs.size(), target_ir_input.size());
    for (size_t i = 0U; i < ir_inputs.size(); ++i) {
      EXPECT_EQ(ir_inputs[i].first, target_ir_input[i]);
    }
    EXPECT_EQ(ir_attr_names.size(), target_ir_attr_name.size());
    for (size_t i = 0U; i < ir_attr_names.size(); ++i) {
      EXPECT_EQ(ir_attr_names[i], target_ir_attr_name[i]);
    }
  };
  auto ret = RunGraphAsync(session, graph_id, inputs, outputs);
  OpsKernelManager::GetInstance().enable_ffts_flag_ = false;
  EXPECT_EQ(ret, SUCCESS);
  session.RemoveGraph(graph_id);
}

TEST_F(DynamicGraphTest, TestRunGraphAsyncRuntime2) {
  char runtime2_env[MMPA_MAX_PATH] = {'1'};
  mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
  auto graph = BuildDynamicInputGraphForRtV2();
  std::map<AscendString, AscendString> options;
  options[OPTION_EXEC_ENABLE_DUMP_DEBUG] = "1";
  options[OPTION_EXEC_DUMP_PATH] = "./";
  options[OPTION_EXEC_DUMP_DEBUG_MODE] = "aicore_overflow"; // OP_DEBUG_ATOMIC /  OP_DEBUG_ALL
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  std::map<AscendString, AscendString> graph_options;

  graph_options[OPTION_EXEC_DYNAMIC_EXECUTE_MODE] = "dynamic_execute";

  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  std::vector<Tensor> inputs;
  inputs.emplace_back(CreateTensor({2, 16}));
  std::vector<Tensor> outputs;
  setenv("HYBRID_PROFILING_LEVEL", "1", 1);
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
  EXPECT_EQ(RunGraphAsync(session, graph_id, inputs, outputs), SUCCESS);
  unsetenv("HYBRID_PROFILING_LEVEL");
  session.RemoveGraph(graph_id);
  runtime2_env[0] = {'0'};
  mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
}

TEST_F(DynamicGraphTest, TestRunGraphRuntime2) {
  char runtime2_env[MMPA_MAX_PATH] = {'1'};
  mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
  auto graph = BuildDynamicInputGraphForRtV2();
  std::map<AscendString, AscendString> options;
  options[OPTION_EXEC_ENABLE_DUMP_DEBUG] = "1";
  options[OPTION_EXEC_DUMP_PATH] = "./";
  options[OPTION_EXEC_DUMP_DEBUG_MODE] = "aicore_overflow"; // OP_DEBUG_ATOMIC /  OP_DEBUG_ALL
  std::map<AscendString, AscendString> graph_options;

  graph_options[OPTION_EXEC_DYNAMIC_EXECUTE_MODE] = "dynamic_execute";

  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  std::vector<Tensor> inputs;
  inputs.emplace_back(CreateTensor({2, 16}));
  std::vector<Tensor> outputs;
  setenv("HYBRID_PROFILING_LEVEL", "1", 1);
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  unsetenv("HYBRID_PROFILING_LEVEL");
  session.RemoveGraph(graph_id);
  runtime2_env[0] = {'0'};
  mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
}

TEST_F(DynamicGraphTest, TestLazyRecompile) {
  DEF_GRAPH(graph_def) {
    auto var = OP_CFG(VARIABLE)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op = OP_CFG("FakeType2Op")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});


    auto shape_op = OP_CFG(SHAPE)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {1});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("var", var)
              ->NODE("fake_type2_op", fake_type2_op)
              ->NODE("shape", shape_op)
              ->NODE("Node_Output", net_output));
  };

  auto graph = ToGeGraph(graph_def);
  MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
  std::map<std::string, std::string> options;
  options[OPTION_GRAPH_RUN_MODE] = "1";  // train
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);

  std::map<AscendString, AscendString> graph_options;
  graph_options[OPTION_EXEC_DYNAMIC_EXECUTE_MODE] = "lazy_recompile";
  graph_options[OPTION_EXEC_ENABLE_COPY_OUTPUT_ADDR] = "1";
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);
  std::vector<Tensor> inputs = CreateInputTensors(graph);
  std::vector<Tensor> outputs;
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  session.RemoveGraph(graph_id);
}

TEST_F(DynamicGraphTest, TestOptimizeDependenciesForConstantInputs) {
  MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
  // 1. const in known-shaped subgraph (after partitioning)
  DEF_GRAPH(graph_def) {
    auto const_op = OP_CFG(CONSTANTOP)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_WEIGHTS, TensorAdapter::AsGeTensor(CreateTensor({16})))
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto fake_type2_op = OP_CFG("FakeType2Op")
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_FORCE_UNKNOWN_SHAPE, true)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("const_op", const_op)->NODE("fake_type2_op", fake_type2_op)->NODE("Node_Output", net_output));
  };
  auto graph = ToGeGraph(graph_def);
  EXPECT_ExecuteDynamicOnlineTrain(graph);

  // 2. const in root graph
  DEF_GRAPH(graph_def2) {
    auto const_op = OP_CFG(CONSTANTOP)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_WEIGHTS, TensorAdapter::AsGeTensor(CreateTensor({16})))
        .Attr(ATTR_NAME_FORCE_UNKNOWN_SHAPE, true)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto fake_type2_op = OP_CFG("FakeType2Op")
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_FORCE_UNKNOWN_SHAPE, true)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    CHAIN(NODE("const_op", const_op)->NODE("fake_type2_op", fake_type2_op)->NODE("Node_Output", net_output));
  };
  graph = ToGeGraph(graph_def2);
  EXPECT_ExecuteDynamicOnlineTrain(graph);
}

TEST_F(DynamicGraphTest, BasicV1LoopDynamicExecSucc) {
  MockForGenerateTask("AIcoreEngine", GenerateTaskForStaticAicore);
  auto graph = GraphFactory::BuildV1LoopGraph1();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->SetGraphUnknownFlag(true);
  for (const auto &node : compute_graph->GetAllNodes()) {
    if (node->GetType() == DATA || node->GetType() == CONSTANT) {
      continue;
    }
    (void)AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
  }
  const bool tiling_result_backup = tiling_result_;
  tiling_result_ = true;
  EXPECT_ExecuteDynamicOnlineTrain(graph);
  tiling_result_ = tiling_result_backup;
}

/**
 * 测试思路
 * 准备工作：
 * 1、构造动态shape图，图中conv2d算子shape为{-2}
 * 2、不打桩gentask，即跳过gentask
 * 3、为FE的执行时泛化编译接口FuzzComile打桩
 *
 * 执行用例：
 * 1、session配置graph option, 开启OPTION_EXEC_DYNAMIC_EXECUTE_MODE为dynamic_execute，不配置shape range。表示开启模糊编译
 *    开启ge.shape_generalized_build_mode为shape_generalized，模拟adapter在训练时开启模糊编译传入的option
 * 2、session add准备工作中构造的动态图
 * 3、执行RunGraphAsync，给定input shape.  [2,2,100,2]
 * 4、执行成功
 *    校验FuzzComile函数被调用了1次，表示-2的conv2d算子发生了在线编译.
 *
 */
TEST_F(DynamicGraphTest, TestFuzzCompileUnknownRankLoadWithOutKernel) {
  FakeFuzzCompileEngine();
  MockOnceForOnceSkipGenerateTask("AIcoreEngine", SkipGenerateTask, GenerateTaskForTaskWithHandle);
  auto graph = BuildFuzzCompileUnknownRankGraph();
  std::map<AscendString, AscendString> options;
  options[OPTION_GRAPH_RUN_MODE] = "1";  // train
  options[OPTION_EXEC_ENABLE_DUMP_DEBUG] = "1";
  options[OPTION_EXEC_DUMP_PATH] = "./";
  options[OPTION_EXEC_DUMP_DEBUG_MODE] = "aicore_overflow"; // OP_DEBUG_ATOMIC /  OP_DEBUG_ALL
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";

  std::map<AscendString, AscendString> graph_options;
  graph_options[OPTION_EXEC_DYNAMIC_EXECUTE_MODE] = "dynamic_execute";
  graph_options["ge.shape_generalized_build_mode"] = "shape_generalized";

  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  std::vector<Tensor> inputs;
  inputs.emplace_back(CreateTensor({2,2,100,2}));
  std::vector<Tensor> outputs;
  EXPECT_EQ(RunGraphAsync(session, graph_id, inputs, outputs), SUCCESS);

  // check fuzz_compile called count when execute
  OpsKernelManager &kernel_manager = OpsKernelManager::GetInstance();
  auto iter = kernel_manager.GetAllOpsKernelInfoStores().find("AIcoreEngine");
  auto fuzz_compile_store = dynamic_cast<FakeFuzzCompilerOpsKernelInfoStore *>(iter->second.get());
  auto fuzz_compile_counts = fuzz_compile_store->GetNodeFuzzCompileCount("conv2d");
  EXPECT_EQ(fuzz_compile_counts, 1); // input shape is out of fuzz range, so conv2d will fuzz_compile once
  session.RemoveGraph(graph_id);
}

/**
 * 测试思路
 * 准备工作：
 * 1、构造动态shape图，图中conv2d算子shape为{-2}
 * 2、不打桩gentask，即跳过gentask
 * 3、为FE的执行时泛化编译接口FuzzComile打桩
 *
 * 执行用例：
 * 1、session配置graph option, 开启OPTION_EXEC_DYNAMIC_EXECUTE_MODE为dynamic_execute，不配置shape range。表示开启模糊编译
 *    开启ge.shape_generalized_build_mode为shape_generalized，模拟adapter在训练时开启模糊编译传入的option
 * 2、session add准备工作中构造的动态图
 * 3、执行RunGraphAsync，给定input shape.  [2,2,100,2]
 * 4、执行成功
 *    校验FuzzComile函数被调用了1次，表示-2的conv2d算子发生了在线编译.
 *
 */
TEST_F(DynamicGraphTest, TestFuzzCompileUnknownRankLoadWithOutKernel_GertTensor) {
  FakeFuzzCompileEngine();
  MockOnceForOnceSkipGenerateTask("AIcoreEngine", SkipGenerateTask, GenerateTaskForTaskWithHandle);
  auto graph = BuildFuzzCompileUnknownRankGraph();
  std::map<AscendString, AscendString> options;
  options[OPTION_GRAPH_RUN_MODE] = "1";  // train
  options[OPTION_EXEC_ENABLE_DUMP_DEBUG] = "1";
  options[OPTION_EXEC_DUMP_PATH] = "./";
  options[OPTION_EXEC_DUMP_DEBUG_MODE] = "aicore_overflow"; // OP_DEBUG_ATOMIC /  OP_DEBUG_ALL
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";

  std::map<AscendString, AscendString> graph_options;
  graph_options[OPTION_EXEC_DYNAMIC_EXECUTE_MODE] = "dynamic_execute";
  graph_options["ge.shape_generalized_build_mode"] = "shape_generalized";

  GeSession session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  std::vector<Tensor> inputs;
  inputs.emplace_back(CreateTensor({2,2,100,2}));
  std::vector<gert::Tensor> gert_inputs;
  TensorTransUtils::Tensors2GertTensors(inputs, gert_inputs);

  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(RunGraphAsync(session, graph_id, gert_inputs, outputs), SUCCESS);

  // check fuzz_compile called count when execute
  OpsKernelManager &kernel_manager = OpsKernelManager::GetInstance();
  auto iter = kernel_manager.GetAllOpsKernelInfoStores().find("AIcoreEngine");
  auto fuzz_compile_store = dynamic_cast<FakeFuzzCompilerOpsKernelInfoStore *>(iter->second.get());
  auto fuzz_compile_counts = fuzz_compile_store->GetNodeFuzzCompileCount("conv2d");
  EXPECT_EQ(fuzz_compile_counts, 1); // input shape is out of fuzz range, so conv2d will fuzz_compile once
  session.RemoveGraph(graph_id);
}

/**
 * 测试思路
 * 准备工作：
 * 1、构造静态shape图，图中conv2d算子shape为{2,2,3,2}
 * 2、为FE的fuzz函数打桩，将图中算子shape泛化为[2,2,-1,2], range 泛化为[[2,2],[2,2,],[1,20],[2,2]]
 * 3、为FE的子图优化函数打桩，将图中算子(conv2d+relu)融合为1个ub融合算子
 *    将融合前的图作为属性打在ub融合算子上。
 * 4、为FE的执行时泛化编译接口FuzzComile打桩，泛化编译ub融合算子泛化失败
 *
 * 执行用例：
 * 1、session配置graph option, 开启OPTION_EXEC_DYNAMIC_EXECUTE_MODE为dynamic_execute，不配置shape range。表示开启模糊编译
 *    开启ge.shape_generalized_build_mode为shape_generalized，模拟adapter在训练时开启模糊编译传入的option
 * 2、session add准备工作中构造的静态图
 * 3、执行buildgraph。此时会触发fe的fuzz函数桩。
 *    校验图中conv2d算子的shape
 *    校验图中conv2d+relu融合为一个conv2d算子，且带融合算子属性
 * 4、执行RunGraphAsync，给定input超出输入shape range.  [2,2,100,2]
 * 5、校验FuzzComile函数被调用了1次，表示发生kernel miss，并触发了执行时编译.
 *
 */
TEST_F(DynamicGraphTest, TestFuzzCompileUBfusionExecuteSwitchToOriginGraphExecution) {
  FakeFuzzCompileEngineForUbFusion();
  MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
  auto graph_with_ubfusion = BuildFuzzCompileOriginGraphWithUBfusion();
  std::map<AscendString, AscendString> options;
  options[OPTION_GRAPH_RUN_MODE] = "1";  // train
  options[OPTION_EXEC_ENABLE_DUMP_DEBUG] = "1";
  options[OPTION_EXEC_DUMP_PATH] = "./";
  options[OPTION_EXEC_DUMP_DEBUG_MODE] = "aicore_overflow"; // OP_DEBUG_ATOMIC /  OP_DEBUG_ALL
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";

  std::map<AscendString, AscendString> graph_options;
  graph_options[OPTION_EXEC_DYNAMIC_EXECUTE_MODE] = "dynamic_execute";
  graph_options["ge.shape_generalized_build_mode"] = "shape_generalized";

  Session session(options);
  GraphId graph_id = 2;
  EXPECT_EQ(session.AddGraph(graph_id, graph_with_ubfusion, graph_options), SUCCESS);

  std::vector<Tensor> inputs;
  inputs.emplace_back(CreateTensor({2,20,300,2})); // range out of fuzz result
  inputs.emplace_back(CreateTensor({2,20,300,2}));
  std::vector<Tensor> outputs;

  EXPECT_EQ(RunGraphAsync(session, graph_id, inputs, outputs), SUCCESS);

  // check fuzz_compile called count when execute
  OpsKernelManager &kernel_manager = OpsKernelManager::GetInstance();
  auto iter = kernel_manager.GetAllOpsKernelInfoStores().find("AIcoreEngine");
  auto fuzz_compile_store = dynamic_cast<FakeFuzzCompilerOpsKernelInfoStore *>(iter->second.get());
  auto fuzz_compile_counts = fuzz_compile_store->GetNodeFuzzCompileCount("conv2d_fused");
  EXPECT_EQ(fuzz_compile_counts, 1); // input shape is out of fuzz range, so conv2d_fused will fuzz_compile once
  auto conv2d_in_sub_fuzz_compile_counts = fuzz_compile_store->GetNodeFuzzCompileCount("conv2d");
  EXPECT_EQ(conv2d_in_sub_fuzz_compile_counts, 1);
  auto relu_in_sub_fuzz_compile_counts = fuzz_compile_store->GetNodeFuzzCompileCount("relu");
  EXPECT_EQ(relu_in_sub_fuzz_compile_counts, 1); //fuse node fuzz failed, switch to origin graph execution, so relu will fuzz once
  session.RemoveGraph(graph_id);
}

const static std::vector<int64_t> val_list_int;
const static std::vector<bool> val_list_bool;
const static std::vector<float> val_list_float;
const static std::vector<std::string> val_list_str;
const static std::vector<DataType> val_list_dt;
const static NamedAttrs var_name_attr;

REG_OP(TestAllAttr)
    .INPUT(data, TensorType::ALL())
    .OPTIONAL_INPUT(option_input, TensorType::ALL())
    .OUTPUT(out, TensorType::ALL())
    .ATTR(test_str, String, "")
    .ATTR(test_int, Int, 0)
    .ATTR(test_bool, Bool, false)
    .ATTR(test_float, Float, 0.0)
    .ATTR(test_dt, Type, DT_FLOAT)
    .ATTR(test_list_string, ListString, val_list_str)
    .ATTR(test_list_int, ListInt, val_list_int)
    .ATTR(test_list_bool, ListBool, val_list_bool)
    .ATTR(test_list_float, ListFloat, val_list_float)
    .ATTR(test_list_dt, ListType, val_list_dt)
    .ATTR(test_name_attr, NamedAttrs, var_name_attr)
    .OP_END_FACTORY_REG(TestAllAttr)

TEST_F(DynamicGraphTest, TestCcmAllAttr) {
  std::vector<int64_t> shape{1, 2};
  auto graph = std::make_shared<ComputeGraph>("fake_graph");
  auto tensor_desc = std::make_shared<GeTensorDesc>(GeShape(shape));
  auto op_desc = std::make_shared<OpDesc>("TestAllAttr", "TestAllAttr");
  ASSERT_NE(op_desc, nullptr);
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddOptionalInputDesc("option_input", GeTensorDesc(GeShape(), FORMAT_RESERVED, DT_UNDEFINED));
  op_desc->AddOutputDesc(tensor_desc->Clone());
  ASSERT_EQ(op_desc->GetAllInputsSize(), 2);
  ASSERT_EQ(op_desc->GetInputsSize(), 1);
  ASSERT_NE(op_desc->MutableInputDesc(0), nullptr);
  ASSERT_EQ(op_desc->MutableInputDesc(1), nullptr);
  AttrUtils::SetInt(op_desc, "test_int", 0);
  AttrUtils::SetStr(op_desc, "test_str", "");
  AttrUtils::SetBool(op_desc, "test_bool", false);
  AttrUtils::SetFloat(op_desc, "test_float", 0.0);
  AttrUtils::SetDataType(op_desc, "test_dt", DT_FLOAT);
  std::vector<DataType> val_list_dt{DT_FLOAT};
  AttrUtils::SetListDataType(op_desc, "test_list_dt", val_list_dt);
  std::vector<bool> val_list_bool{true};
  AttrUtils::SetListBool(op_desc, "test_list_bool", val_list_bool);
  std::vector<int64_t> val_list_int{1,2};
  AttrUtils::SetListInt(op_desc, "test_list_int", val_list_int);
  std::vector<float> val_list_float{1.0, 2.0};
  AttrUtils::SetListFloat(op_desc, "test_list_float", val_list_float);
  std::vector<std::string> val_list_string{"1", "2"};
  AttrUtils::SetListStr(op_desc, "test_list_string", val_list_string);
  std::vector<std::vector<int64_t>> val_list_list_int{{1,2}};
  AttrUtils::SetListListInt(op_desc, "test_list_list_int", val_list_list_int);
  NamedAttrs name_attr;
  AttrUtils::SetNamedAttrs(op_desc, "test_name_attr", name_attr);
  NodeCompileCacheItem item;
  NodeCompileCacheModule ccm;
  auto node = graph->AddNode(op_desc);
  ASSERT_NE(node, nullptr);
  auto add_item = ccm.AddCompileCache(node, item);
  ASSERT_NE(add_item, nullptr);
  auto find_item = ccm.FindCompileCache(node);
  ASSERT_NE(find_item, nullptr);
  ASSERT_EQ(add_item->GetCacheItemId(), find_item->GetCacheItemId());
}

TEST_F(DynamicGraphTest, TestNotSupportDynamicShape) {
  FakeFuzzCompileEngine();

  DEF_GRAPH(dynamic_graph) {
    auto data_0 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto relu = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {16})
                  .Build("not_support_dynamic_shape");

    auto unique_op = OP_CFG("Unique").InCnt(1).OutCnt(2).TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1});

    CHAIN(NODE("_arg_0", data_0)->NODE(relu)->NODE("unique", unique_op)
          ->NODE("Node_Output", net_output));
  };

  Graph graph = ToGeGraph(dynamic_graph);

  std::map<std::string, std::string> options;
  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);
  std::vector<Tensor> inputs = CreateInputTensors(graph);
  std::vector<Tensor> outputs;
  EXPECT_NE(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
}

UINT32 StubTiling4ST(gert::TilingContext *context) {
  context->SetNeedAtomic(false);
  context->SetTilingKey(666U);
  context->SetBlockDim(666U);
  auto tiling_context = reinterpret_cast<gert::TilingContext *>(context);
  auto tiling_data = tiling_context->GetTilingData<uint64_t>();
  *tiling_data = 100;
  return ge::GRAPH_SUCCESS;
}

UINT32 StubTilingParse4ST(gert::KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

void* CompileInfoCreator4ST() {
  auto tmp =  ge::MakeUnique<char>();
  return tmp.get();
}

TEST_F(DynamicGraphTest, load_KnownSubgraph_WithSoftsyncop_ExecuteSuccess) {
  auto graph = gert::ShareGraph::BuildWithKnownSubgraph();
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() != "Relu") {
      continue;
    }
    const auto op_desc = node->GetOpDesc();
    AttrUtils::SetBool(op_desc, ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
    AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
    std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
    AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
    AttrUtils::SetInt(op_desc, "op_para_size", 512);
    op_desc->SetIsInputConst({false});
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({128});
    AttrUtils::SetInt(op_desc, ATTR_NAME_IMPLY_TYPE, static_cast<uint32_t>(domi::ImplyType::TVM));
    TensorUtils::SetSize(*op_desc->MutableInputDesc(0), 32);
    TensorUtils::SetSize(*op_desc->MutableOutputDesc(0), 32);
    std::vector<char> kernelBin;
    TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("00_0_kernel", std::move(kernelBin));
    AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "00_0_kernel");
    op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
    AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  }
  graph->TopologicalSorting();
  auto root_model = gert::GeModelBuilder(graph).BuildGeRootModel();
  auto faker = gert::GlobalDataFaker(root_model);
  auto global_data = faker.FakeWithHandleAiCore("Relu", false).Build();

  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl("Relu");
  funcs->tiling = StubTiling4ST;
  funcs->tiling_parse = StubTilingParse4ST;
  funcs->compile_info_creator = CompileInfoCreator4ST;
  funcs->compile_info_deleter = nullptr;

  gert::ModelDescHolder model_desc_holder;
  model_desc_holder.SetSpaceRegistry(space_registry);
  auto graph_convert = gert::GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  auto model_executor = gert::ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
}

/**
 * 用例描述：davinci model上报的加载信息的model id非无效值
 *
 * 预置条件：NA
 *
 * 测试步骤：
 * 1. 构造带静态图执行器
 * 2. 打开profiling开关
 * 3. 加载执行
 *
 * 预期结果：
 * 1. 收到model load event的信息，且model id不为uint32_max
 */
TEST_F(DynamicGraphTest, ProfilingReport_ReportValidModelId_OnModelLoad) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  ModelManager::GetInstance().cust_aicpu_so_.clear();
  ProfilingProperties::Instance().SetLoadProfiling(true);
  auto graph = gert::ShareGraph::BuildWithKnownSubgraph();
  gert::GertRuntimeStub fakeRuntime;
  graph->TopologicalSorting();
  auto root_model = gert::GeModelBuilder(graph).BuildGeRootModel();
  auto faker = gert::GlobalDataFaker(root_model);
  auto global_data = faker.FakeWithoutHandleAiCore("Conv2d", false).Build();
  graph->SetGraphID(999);
  gert::ModelDescHolder model_desc_holder;
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = gert::GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  auto model_executor = gert::ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  ge::ProfilingProperties::Instance().SetLoadProfiling(true);
  gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::ProfilingType>(
          {gert::ProfilingType::kTaskTime, gert::ProfilingType::kDevice}));
  auto check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len) -> int32_t {
    if (type == ge::InfoType::kEvent) {
      auto prof_event = reinterpret_cast<MsprofEvent *>(data);
      if (prof_event->type == static_cast<uint32_t>(gert::GeProfInfoType::kModelLoad)){
        EXPECT_NE(prof_event->itemId, std::numeric_limits<uint32_t>::max());
      }
    }
    if (type == ge::InfoType::kInfo) {
      auto prof_info = reinterpret_cast<MsprofAdditionalInfo *>(data);
      if (prof_info->type == MSPROF_REPORT_MODEL_GRAPH_ID_MAP_TYPE) {
        auto map_data = reinterpret_cast<MsprofGraphIdInfo *>(prof_info->data);
        EXPECT_EQ(map_data->graphId, 999);
      }
    }
    return 0;
  };
  ge::ProfilingTestUtil::Instance().SetProfFunc(check_func);
  domi::GetContext().is_online_model = true;
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(0);
  model_executor->UnLoad();
}

TEST_F(DynamicGraphTest, TestDynamicInput_size_fail) {
  auto recover_ir_graph = BuildDynamicInputGraph();
  std::map<AscendString, AscendString> options;
  std::map<AscendString, AscendString> graph_options;

  graph_options[OPTION_EXEC_DYNAMIC_EXECUTE_MODE] = "dynamic_execute";
  graph_options[OPTION_EXEC_DATA_INPUTS_SHAPE_RANGE] = "[1~20,1~30]";

  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, recover_ir_graph, graph_options), SUCCESS);

  std::vector<Tensor> inputs;
  inputs.emplace_back(CreateTensor({2, 16}));
  inputs.emplace_back(CreateTensor({2, 16}));
  gert::GertRuntimeStub stub;
  stub.GetSlogStub().SetLevel(DLOG_DEBUG);
  stub.GetSlogStub().Clear();
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), FAILED);
  EXPECT_EQ(stub.GetSlogStub().FindLog(DLOG_ERROR, "Dynamic input shape range size "), 0);
  EXPECT_NE(stub.GetSlogStub().FindLog(DLOG_DEBUG, "Option ge.exec.dataInputsShapeRange's readable name"), -1);
  stub.GetSlogStub().SetLevel(DLOG_ERROR);
}

TEST_F(DynamicGraphTest, TestDynamicInput_single_stream_succ) {
  setenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM", "1", 0);
  auto recover_ir_graph = BuildDynamicInputGraph();
  std::map<AscendString, AscendString> options;
  std::map<AscendString, AscendString> graph_options;

  graph_options[ENABLE_SINGLE_STREAM] = "true";
  options[JIT_COMPILE.c_str()] = "1";

  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, recover_ir_graph, graph_options), SUCCESS);

  std::vector<Tensor> inputs;
  inputs.emplace_back(CreateTensor({2, 16}));
  inputs.emplace_back(CreateTensor({2, 16}));
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
  unsetenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM");
}
}  // namespace ge

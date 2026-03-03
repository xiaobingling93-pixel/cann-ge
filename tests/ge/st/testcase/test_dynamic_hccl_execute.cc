/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// To test the execution of dynamic data flow ops (Stack series)

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "macro_utils/dt_public_scope.h"
#include "ge_running_env/fake_ops_kernel_builder.h"
#include "engines/manager/opskernel_manager/ops_kernel_builder_manager.h"
#include "macro_utils/dt_public_unscope.h"

#include <dlfcn.h>
#include "register/op_tiling_registry.h"
#include "hccl/base.h"
#include "hccl/hccl_types.h"
#include "framework/executor/ge_executor.h"
#include "graph/utils/graph_utils.h"
#include "ge_running_env/fake_op.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "common/plugin/ge_make_unique_util.h"
#include "ge/ge_api.h"
#include "session/session_manager.h"
#include "init_ge.h"
#include "utils/bench_env.h"
#include "utils/taskdef_builder.h"
#include "utils/mock_ops_kernel_builder.h"
#include "depends/runtime/src/runtime_stub.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "graph/manager/mem_manager.h"

using namespace std;
using namespace testing;

namespace ge {
namespace {
void *mock_handle = nullptr;

HcclResult HcomExecEnqueueOperation(HcomOperation opInfo, std::function<void(HcclResult status)> callback) {
  callback(HCCL_SUCCESS);
  return HCCL_SUCCESS;
}

HcclResult HcomExecEnqueueRemoteOperation(HcomRemoteOperation opInfo,
                                          std::function<void(HcclResult status)> callback) {
  callback(HCCL_SUCCESS);
  return HCCL_SUCCESS;
}

HcclResult HcomExecEnqueueRemoteAccess(const std::string& remoteAccessType,
                                       const std::vector<HcomRemoteAccessAddrInfo>& addrInfos,
                                       std::function<void(HcclResult status)> callback) {
  callback(HCCL_SUCCESS);
  return HCCL_SUCCESS;
}

HcclResult HcomExecEnqueueAllToAllV(HcomAllToAllVParams params, std::function<void(HcclResult status)> callback) {
  callback(HCCL_SUCCESS);
  return HCCL_SUCCESS;
}

HcclResult HcomExecEnqueueAllToAllVC(HcomAllToAllVCParams params, std::function<void(HcclResult status)> callback) {
  callback(HCCL_SUCCESS);
  return HCCL_SUCCESS;
}

HcclResult HcomExecEnqueueGatherAllToAllV(HcomGatherAllToAllVParams params,
                                          std::function<void(HcclResult status)> callback) {
  callback(HCCL_SUCCESS);
  return HCCL_SUCCESS;
}

HcclResult HcomExecInitialize() {
  return HCCL_SUCCESS;
}

HcclResult HcomExecFinalize() {
  return HCCL_SUCCESS;
}

class MockMmpa : public MmpaStubApiGe {
 public:
  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "HcomExecEnqueueOperation") {
      return (void *)&HcomExecEnqueueOperation;
    } else if (std::string(func_name) == "HcomExecEnqueueRemoteAccess") {
      return (void *)&HcomExecEnqueueRemoteAccess;
    } else if (std::string(func_name) == "HcomExecEnqueueAllToAllV") {
      return (void *)&HcomExecEnqueueAllToAllV;
    } else if (std::string(func_name) == "HcomExecEnqueueAllToAllVC") {
      return (void *)&HcomExecEnqueueAllToAllVC;
    } else if (std::string(func_name) == "HcomExecEnqueueGatherAllToAllV") {
      return (void *)&HcomExecEnqueueGatherAllToAllV;
    } else if (std::string(func_name) == "HcomExecInitialize") {
      return (void *)&HcomExecInitialize;
    } else if (std::string(func_name) == "HcomExecFinalize") {
      return (void *)&HcomExecFinalize;
    } else if (std::string(func_name) == "HcomExecEnqueueRemoteOperation") {
      return (void *)&HcomExecEnqueueRemoteOperation;
    }
    return dlsym(handle, func_name);
  }
  int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
    (void)realpath(path, realPath);
    return EN_OK;
  }
  void *DlOpen(const char *fileName, int32_t mode) override {
    return (void *)mock_handle;
  }
  int32_t DlClose(void *handle) override {
    return 0L;
  }
};

Status GenerateTaskForStaticAicore(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) {
  tasks.emplace_back(AiCoreTaskDefBuilder(node).BuildTask());
  return SUCCESS;
}

Status GenerateTaskForAicpuDependRange(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) {
  tasks.emplace_back(AicpuTaskDefBuilder(node).BuildAicpuTask(3));
  return SUCCESS;
}

}
class DynamicHcclTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    GEFinalize();
    ReInitGe();
    BenchEnv::Init();
  }
  void SetUp() {
    char runtime2_env[MMPA_MAX_PATH] = {'0'};
    mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
    auto infer_fun = [](Operator &op) -> graphStatus {
      auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
      *op_desc->MutableOutputDesc(0) = *op_desc->GetInputDescPtr(0);
      return GRAPH_SUCCESS;
    };
    auto infer_depend1_fun = [](Operator &op) -> graphStatus {
      auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
      *op_desc->MutableOutputDesc(0) = *op_desc->GetInputDescPtr(0);
      op_desc->SetOpInferDepends({"remote", "local", "local_offset"});
      return GRAPH_SUCCESS;
    };
    auto infer_depend2_fun = [](Operator &op) -> graphStatus {
      auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
      *op_desc->MutableOutputDesc(0) = *op_desc->GetInputDescPtr(0);
      op_desc->SetOpInferDepends({"remote"});
      return GRAPH_SUCCESS;
    };

    auto unique_infer_fun = [](Operator &op) -> graphStatus {
      auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
      auto output_desc = op_desc->MutableOutputDesc(0);
      output_desc->SetShape(GeShape({-1}));
      output_desc->SetShapeRange({{1, 16}});
      return GRAPH_SUCCESS;
    };

    GeRunningEnvFaker()
        .Reset()
        .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
        .Install(FakeEngine("DNN_HCCL").KernelInfoStore("DNN_HCCL"))
        .Install(FakeEngine("DNN_HCCL").KernelInfoStore(kEngineNameHccl))
        .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("DNN_VM_RTS"))
        .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("DNN_VM_RTS_OP_STORE"))
        .Install(FakeEngine(kEngineNameAiCpu).KernelInfoStore(kEngineNameAiCpu))
        .Install(FakeOp(STREAMACTIVE).InfoStoreAndBuilder("DNN_VM_RTS").InferShape(infer_fun))
        .Install(FakeOp(SEND).InfoStoreAndBuilder("DNN_VM_RTS").InferShape(infer_fun))
        .Install(FakeOp(SENDNOTIFY).InfoStoreAndBuilder("DNN_VM_RTS").InferShape(infer_fun))
        .Install(FakeOp(RECV).InfoStoreAndBuilder("DNN_VM_RTS").InferShape(infer_fun))
        .Install(FakeOp(RECVNOTIFY).InfoStoreAndBuilder("DNN_VM_RTS").InferShape(infer_fun))
        .Install(FakeOp(IDENTITY).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE").InferShape(infer_fun))
        .Install(FakeOp(ADD).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_fun))
        .Install(FakeOp(NEG).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_fun))
        .Install(FakeOp(HCOMALLREDUCE).InfoStoreAndBuilder(kEngineNameHccl).InferShape(infer_fun))
        .Install(FakeOp(HCOMREMOTEWRITE).Inputs({"remote", "local", "local_offset"})
                                                .InfoStoreAndBuilder(kEngineNameHccl).InferShape(infer_depend1_fun))
        .Install(FakeOp(HCOMREMOTEREFREAD).Inputs({"remote"})
                                                  .InfoStoreAndBuilder(kEngineNameHccl).InferShape(infer_depend2_fun))
        .Install(FakeOp(HCOMREMOTEREAD).Inputs({"remote"})
                     .InfoStoreAndBuilder(kEngineNameHccl).InferShape(infer_depend2_fun))
        .Install(FakeOp(HCOMGATHERALLTOALLV).InfoStoreAndBuilder(kEngineNameHccl).InferShape(infer_fun))
        .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(infer_fun))
        .Install(FakeOp(VARIABLE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(infer_fun))
        .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(HCOMALLTOALLV).InfoStoreAndBuilder(kEngineNameHccl).InferShape(infer_fun))
        .Install(FakeOp(HCOMALLTOALLVC).InfoStoreAndBuilder(kEngineNameHccl).InferShape(infer_fun))
        .Install(FakeOp(IF).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp("Unique").InfoStoreAndBuilder("aicpu_ascend_kernel").InferShape(unique_infer_fun))
        .Install(FakeOp("UniqueV2").InfoStoreAndBuilder("AIcoreEngine").InferShape(unique_infer_fun))
        .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"));
  }
  void TearDown() {
    MmpaStub::GetInstance().Reset();
    GEFinalize();
    ReInitGe();
    char runtime2_env[MMPA_MAX_PATH] = {'1'};
    mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  }
};

/*******************************************************************************

******************************************************************************/
static void BuildHcclAllReduceGraph(Graph &ge_graph, uint32_t &mem_offset) {
  GeTensorDesc tensor_desc(GeShape{});
  GeTensor tensor(tensor_desc);
  DEF_GRAPH(g0) {
    auto data0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto add = OP_CFG(ADD)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto hcom_allreduce = OP_CFG(HCOMALLREDUCE)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16})
        .Attr(ATTR_NAME_FORCE_UNKNOWN_SHAPE, true)
        .Attr(HCOM_ATTR_REDUCE_TYPE, "min");

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {});

    CHAIN(NODE("_arg_0", data0)->NODE("add0", add)->NODE("allreduce", hcom_allreduce)
              ->NODE("Node_Output", net_output));
//    CHAIN(NODE("const_0", const_0)->NODE("Node_Output", net_output));
  };
  ComputeGraphPtr graph = ToComputeGraph(g0);
  ge_graph = ToGeGraph(g0);
}

/*******************************************************************************

******************************************************************************/
static void BuildHcclRefReadGraph(Graph &ge_graph, uint32_t &mem_offset) {
  GeTensorDesc tensor_desc(GeShape{});
  GeTensor tensor(tensor_desc);
  uint64_t data_raw[3] = {10, 11, 12};
  std::vector<uint8_t> data(sizeof(data_raw));
  std::vector<int64_t> shape{3};
  (void)memcpy_s(data.data(), sizeof(data_raw), data_raw, sizeof(data_raw));
  tensor.MutableTensorDesc().SetShape(GeShape(shape));
  tensor.SetData(data);
  DEF_GRAPH(g0) {
    auto const0 = OP_CFG(CONSTANTOP)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_WEIGHTS, tensor)
        .Attr(ATTR_NAME_FORCE_UNKNOWN_SHAPE, true)
        .TensorDesc(FORMAT_ND, DT_UINT64, {3});

    auto hcom_remote_refread = OP_CFG(HCOMREMOTEREFREAD)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_UINT64, {3})
        .Attr(ATTR_NAME_FORCE_UNKNOWN_SHAPE, true)
        .Attr(HCOM_ATTR_REDUCE_TYPE, "min");

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_UINT64, {3});

    CHAIN(NODE("_arg_0", const0)->NODE("refread", hcom_remote_refread)->NODE("Node_Output", net_output));
  };
  ComputeGraphPtr graph = ToComputeGraph(g0);
  ge_graph = ToGeGraph(g0);
}

static void BuildHcclReadGraph(Graph &ge_graph, uint32_t &mem_offset) {
  GeTensorDesc tensor_desc(GeShape{});
  GeTensor tensor(tensor_desc);
  uint64_t data_raw[] = {10, 11, 25, 10, 11, 25, 10, 11, 25};
  std::vector<uint8_t> data(sizeof(data_raw));
  std::vector<int64_t> shape{3, 3};
  (void)memcpy_s(data.data(), sizeof(data_raw), data_raw, sizeof(data_raw));
  tensor.MutableTensorDesc().SetShape(GeShape(shape));
  tensor.SetData(data);
  DEF_GRAPH(g0) {
    auto const0 = OP_CFG(CONSTANTOP)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_WEIGHTS, tensor)
        .Attr(ATTR_NAME_FORCE_UNKNOWN_SHAPE, true)
        .TensorDesc(FORMAT_ND, DT_UINT64, {3, 3});

    auto hcom_remote_read = OP_CFG(HCOMREMOTEREAD)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_UINT64, {3, 3})
        .Attr(ATTR_NAME_FORCE_UNKNOWN_SHAPE, true)
        .Attr(HCOM_ATTR_REDUCE_TYPE, "min");

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_UINT64, {3, 3});

    CHAIN(NODE("_arg_0", const0)->NODE("refread", hcom_remote_read)->NODE("Node_Output", net_output));
  };
  ComputeGraphPtr graph = ToComputeGraph(g0);
  ge_graph = ToGeGraph(g0);
}

static void BuildHcclGatherAlltoAllGraph(Graph &ge_graph, uint32_t &mem_offset) {
  GeTensorDesc tensor_desc(GeShape{});
  GeTensor tensor(tensor_desc);
  DEF_GRAPH(g0) {
    auto data0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto add1 = OP_CFG(ADD)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    auto add2 = OP_CFG(ADD)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    auto add3 = OP_CFG(ADD)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    auto add4 = OP_CFG(ADD)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    auto add5 = OP_CFG(ADD)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto hcom_all_to_allv = OP_CFG(HCOMGATHERALLTOALLV)
        .InCnt(5)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16})
        .Attr(ATTR_NAME_FORCE_UNKNOWN_SHAPE, true)
        .Attr(HCOM_ATTR_REDUCE_TYPE, "min");

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {});

    CHAIN(NODE("_arg_0", data0)->NODE("add1", add1)->NODE("all_to_all", hcom_all_to_allv)->NODE("Node_Output", net_output));
    CHAIN(NODE("_arg_0", data0)->NODE("add2", add2)->NODE("all_to_all", hcom_all_to_allv)->NODE("Node_Output", net_output));
    CHAIN(NODE("_arg_0", data0)->NODE("add3", add3)->NODE("all_to_all", hcom_all_to_allv));
    CHAIN(NODE("_arg_0", data0)->NODE("add4", add4)->NODE("all_to_all", hcom_all_to_allv));
    CHAIN(NODE("_arg_0", data0)->NODE("add5", add5)->NODE("all_to_all", hcom_all_to_allv));
  };
  ComputeGraphPtr graph = ToComputeGraph(g0);
  ge_graph = ToGeGraph(g0);
}

/*******************************************************************************

******************************************************************************/
static void BuildHcclAlltoAllGraph(Graph &ge_graph, uint32_t &mem_offset) {
  GeTensorDesc tensor_desc(GeShape{});
  GeTensor tensor(tensor_desc);
  DEF_GRAPH(g0) {
    auto data0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto add1 = OP_CFG(ADD)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    auto add2 = OP_CFG(ADD)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    auto add3 = OP_CFG(ADD)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    auto add4 = OP_CFG(ADD)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    auto add5 = OP_CFG(ADD)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto hcom_all_to_allv = OP_CFG(HCOMALLTOALLV)
        .InCnt(5)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16})
        .Attr(ATTR_NAME_FORCE_UNKNOWN_SHAPE, true)
        .Attr(HCOM_ATTR_REDUCE_TYPE, "min");

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {});

    CHAIN(NODE("_arg_0", data0)->NODE("add1", add1)->NODE("all_to_all", hcom_all_to_allv)->NODE("Node_Output", net_output));
    CHAIN(NODE("_arg_0", data0)->NODE("add2", add2)->NODE("all_to_all", hcom_all_to_allv));
    CHAIN(NODE("_arg_0", data0)->NODE("add3", add3)->NODE("all_to_all", hcom_all_to_allv));
    CHAIN(NODE("_arg_0", data0)->NODE("add4", add4)->NODE("all_to_all", hcom_all_to_allv));
    CHAIN(NODE("_arg_0", data0)->NODE("add5", add5)->NODE("all_to_all", hcom_all_to_allv));
  };
  ComputeGraphPtr graph = ToComputeGraph(g0);
  ge_graph = ToGeGraph(g0);
}

/*******************************************************************************

******************************************************************************/
static void BuildHcclAlltoAllVCGraph(Graph &ge_graph, uint32_t &mem_offset) {
  GeTensorDesc tensor_desc(GeShape{});
  GeTensor tensor(tensor_desc);
  DEF_GRAPH(g0) {
    auto data0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto add1 = OP_CFG(ADD)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    auto add2 = OP_CFG(ADD)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto hcom_all_to_allv = OP_CFG(HCOMALLTOALLVC)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16})
        .Attr(ATTR_NAME_FORCE_UNKNOWN_SHAPE, true)
        .Attr(HCOM_ATTR_REDUCE_TYPE, "min");

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {});

    CHAIN(NODE("_arg_0", data0)->NODE("add1", add1)->NODE("all_to_all", hcom_all_to_allv)->NODE("Node_Output", net_output));
    CHAIN(NODE("_arg_0", data0)->NODE("add2", add2)->NODE("all_to_all", hcom_all_to_allv));
  };
  ComputeGraphPtr graph = ToComputeGraph(g0);
  ge_graph = ToGeGraph(g0);
}

/*******************************************************************************

┌────────┐  (0,0)   ┌──────┐  (0,0)   ┌──────────────┐  (0,0)   ┌─────────┐  (0,0)   ┌───────────┐  (0,0)   ┌───────────┐  (0,0)   ┌─────────────┐
│ _arg_0 │ ───────> │ add0 │ ───────> │ remote_write │ ───────> │ refread │ ───────> │ alltoallv │ ───────> │ allreduce │ ───────> │ Node_Output │
└────────┘          └──────┘          └──────────────┘          └─────────┘          └───────────┘          └───────────┘          └─────────────┘

******************************************************************************/
static void BuildHcclRemoteWriteGraph(Graph &ge_graph, uint32_t &mem_offset) {
  GeTensorDesc tensor_desc(GeShape{});
  GeTensor tensor(tensor_desc);

  uint64_t data_raw[3] = {10, 11, 12};
  std::vector<uint8_t> data(sizeof(data_raw));
  std::vector<int64_t> shape{3};
  (void)memcpy_s(data.data(), sizeof(data_raw), data_raw, sizeof(data_raw));
  tensor.MutableTensorDesc().SetShape(GeShape(shape));
  tensor.SetData(data);
  DEF_GRAPH(g0) {
    auto const0 = OP_CFG(CONSTANTOP)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_WEIGHTS, tensor)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_UINT64, {3});

    auto const1 = OP_CFG(CONSTANTOP)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_WEIGHTS, tensor)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_UINT64, {3});

    auto const2 = OP_CFG(CONSTANTOP)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_WEIGHTS, tensor)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_UINT64, {3});

    auto hcom_remote_write = OP_CFG(HCOMREMOTEWRITE)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_UINT64, {3})
        .Attr(ATTR_NAME_FORCE_UNKNOWN_SHAPE, true)
        .Attr(HCOM_ATTR_REDUCE_TYPE, "min");

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_UINT64, {3});

    CHAIN(NODE("_arg_0", const0)->NODE("remote_write", hcom_remote_write)->NODE("Node_Output", net_output));
    CHAIN(NODE("_arg_1", const1)->NODE("remote_write", hcom_remote_write)->NODE("Node_Output", net_output));
    CHAIN(NODE("_arg_2", const2)->NODE("remote_write", hcom_remote_write)->NODE("Node_Output", net_output));
  };
  ComputeGraphPtr graph = ToComputeGraph(g0);
  ge_graph = ToGeGraph(g0);
}

TEST_F(DynamicHcclTest, TestDynamicOnlineTrainingRemoteWrite) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  mock_handle = (void *)0xffffffff;
  MockForGenerateTask("AIcoreEngine", GenerateTaskForStaticAicore);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);
  uint32_t mem_offset = 0;
  Graph graph;
  BuildHcclRemoteWriteGraph(graph, mem_offset);

  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);

  Shape shape({3});
  uint8_t buffer[3 * 8];
  TensorDesc tensor_desc(shape);
  Tensor input_0(tensor_desc);
  input_0.SetData(buffer, sizeof(buffer));

  std::vector<Tensor> inputs{input_0};
  std::vector<Tensor> outputs;
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  session.RemoveGraph(graph_id);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(DynamicHcclTest, TestDynamicOnlineTrainingRefRead) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  mock_handle = (void *)0xffffffff;
  MockForGenerateTask("AIcoreEngine", GenerateTaskForStaticAicore);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);
  uint32_t mem_offset = 0;
  Graph graph;
  BuildHcclRefReadGraph(graph, mem_offset);

  std::map<AscendString, AscendString> options;
//  options[OPTION_GRAPH_RUN_MODE] = "1";  // train
  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);

  Shape shape({3});
  uint8_t buffer[3 * 8] = {55,55,55,55};
  TensorDesc tensor_desc(shape);
  Tensor input_0(tensor_desc);
  input_0.SetData(buffer, sizeof(buffer));

  std::vector<Tensor> inputs{input_0};
  std::vector<Tensor> outputs;
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  session.RemoveGraph(graph_id);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}


TEST_F(DynamicHcclTest, TestDynamicOnlineTrainingRead) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).InitMemory(1024);
  EXPECT_NE(MemManager::Instance().GetMemoryBase(RT_MEMORY_RDMA_HBM, "rdma", 0U), nullptr);
  auto mem = MemManager::Instance().GetRdmaPoolMemory(RT_MEMORY_HBM, 1024U, 0U);
  EXPECT_NE(mem, nullptr);
  MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).Free(mem);
  mock_handle = (void *)0xffffffff;
  MockForGenerateTask("AIcoreEngine", GenerateTaskForStaticAicore);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);
  uint32_t mem_offset = 0;
  Graph graph;
  BuildHcclReadGraph(graph, mem_offset);

  std::map<AscendString, AscendString> options;
  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);

  Shape shape({3, 3});
  uint8_t buffer[9 * 8] = {55,55,55,55};
  TensorDesc tensor_desc(shape);
  Tensor input_0(tensor_desc);
  input_0.SetData(buffer, sizeof(buffer));

  std::vector<Tensor> inputs{input_0};
  std::vector<Tensor> outputs;
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  session.RemoveGraph(graph_id);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(DynamicHcclTest, TestDynamicOnlineTrainingAlltoAll) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  mock_handle = (void *)0xffffffff;
  MockForGenerateTask("AIcoreEngine", GenerateTaskForStaticAicore);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);
  uint32_t mem_offset = 0;
  Graph graph;
  BuildHcclAlltoAllGraph(graph, mem_offset);

  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);

  Shape shape({16});
  uint8_t buffer[16 * 4];
  TensorDesc tensor_desc(shape);
  Tensor input_0(tensor_desc);
  Tensor input_1(tensor_desc);
  Tensor input_2(tensor_desc);
  Tensor input_3(tensor_desc);
  Tensor input_4(tensor_desc);
  input_0.SetData(buffer, sizeof(buffer));
  input_1.SetData(buffer, sizeof(buffer));
  input_2.SetData(buffer, sizeof(buffer));
  input_3.SetData(buffer, sizeof(buffer));
  input_4.SetData(buffer, sizeof(buffer));

  std::vector<Tensor> inputs{input_0, input_1, input_2, input_3, input_4};
  std::vector<Tensor> outputs;
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  session.RemoveGraph(graph_id);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(DynamicHcclTest, TestDynamicOnlineTrainingGatherAlltoAll) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  mock_handle = (void *)0xffffffff;
  MockForGenerateTask("AIcoreEngine", GenerateTaskForStaticAicore);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);
  uint32_t mem_offset = 0;
  Graph graph;
  BuildHcclGatherAlltoAllGraph(graph, mem_offset);

  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);

  Shape shape({16});
  uint8_t buffer[16 * 4];
  TensorDesc tensor_desc(shape);
  Tensor input_0(tensor_desc);
  Tensor input_1(tensor_desc);
  Tensor input_2(tensor_desc);
  Tensor input_3(tensor_desc);
  Tensor input_4(tensor_desc);
  input_0.SetData(buffer, sizeof(buffer));
  input_1.SetData(buffer, sizeof(buffer));
  input_2.SetData(buffer, sizeof(buffer));
  input_3.SetData(buffer, sizeof(buffer));
  input_4.SetData(buffer, sizeof(buffer));

  std::vector<Tensor> inputs{input_0, input_1, input_2, input_3, input_4};
  std::vector<Tensor> outputs;
  outputs.resize(2);
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  session.RemoveGraph(graph_id);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(DynamicHcclTest, TestDynamicOnlineTrainingAlltoAllVC) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  mock_handle = (void *)0xffffffff;
  MockForGenerateTask("AIcoreEngine", GenerateTaskForStaticAicore);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);
  uint32_t mem_offset = 0;
  Graph graph;
  BuildHcclAlltoAllVCGraph(graph, mem_offset);

  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);

  Shape shape({16});
  uint8_t buffer[16 * 4];
  TensorDesc tensor_desc(shape);
  Tensor input_0(tensor_desc);
  Tensor input_1(tensor_desc);
  input_0.SetData(buffer, sizeof(buffer));
  input_1.SetData(buffer, sizeof(buffer));

  std::vector<Tensor> inputs{input_0, input_1};
  std::vector<Tensor> outputs;
  outputs.resize(1);
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  session.RemoveGraph(graph_id);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(DynamicHcclTest, TestDynamicOnlineTrainingAllReduce) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  mock_handle = (void *)0xffffffff;
  MockForGenerateTask("AIcoreEngine", GenerateTaskForStaticAicore);
  MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);
  uint32_t mem_offset = 0;
  Graph graph;
  BuildHcclAllReduceGraph(graph, mem_offset);

  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);

  Shape shape({16});
  uint8_t buffer[16 * 4];
  TensorDesc tensor_desc(shape);
  Tensor input_0(tensor_desc);
  Tensor input_1(tensor_desc);
  Tensor input_2(tensor_desc);
  input_0.SetData(buffer, sizeof(buffer));
  input_1.SetData(buffer, sizeof(buffer));
  input_2.SetData(buffer, sizeof(buffer));

  std::vector<Tensor> inputs{input_0, input_1, input_2};
  std::vector<Tensor> outputs;
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  session.RemoveGraph(graph_id);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}
} // namespace ge


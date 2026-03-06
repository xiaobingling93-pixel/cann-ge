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
#include <string.h>

#include "macro_utils/dt_public_scope.h"

#include "common/profiling/profiling_manager.h"
#include "common/profiling/profiling_properties.h"
#include "common/helper/om_file_helper.h"
#include "common/op/ge_op_utils.h"
#include "hybrid/model/hybrid_model_builder.h"
#include "graph/load/graph_loader.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/load/model_manager/data_inputer.h"
#include "graph/ge_context.h"
#include "graph/ops_stub.h"
#include "graph/manager/graph_manager.h"
#include "graph/passes/graph_builder_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "depends/runtime/src/runtime_stub.h"
#include "depends/ascendcl/src/ascendcl_stub.h"
#include "common/dump/dump_manager.h"
#include "common/share_graph.h"
#include "base/common/model/external_allocator_manager.h"
#include "runtime/v1/graph/manager/mem_manager.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "common/env_path.h"

using namespace std;
using namespace testing;
extern std::string g_runtime_stub_mock;

namespace ge {
namespace {
const char_t *const kEnvName = "ASCEND_OPP_PATH";
const char_t *const kBuiltIn = "built-in";
const char_t *const kVendors = "vendors";
const char_t *const kOpMasterDeviceLib = "/op_impl/ai_core/tbe/op_master_device/lib/";
const std::string ENC_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
class MockRuntime : public RuntimeStub {
 public:
  rtError_t rtKernelLaunchEx(void *args, uint32_t args_size, uint32_t flags, rtStream_t stream) override {
    return -1;
  }

  rtError_t rtsLaunchKernelWithDevArgs(rtFuncHandle funcHandle, uint32_t blockDim, rtStream_t stm,
    rtKernelLaunchCfg_t *cfg, const void *args, uint32_t argsSize, void *reserve) {
    return -1;
  }

  rtError_t rtsLaunchKernelWithHostArgs(rtFuncHandle funcHandle, uint32_t blockDim,
      rtStream_t stm, rtKernelLaunchCfg_t * cfg, void * hostArgs, uint32_t argsSize,
      rtPlaceHolderInfo_t * placeHolderArray, uint32_t placeHolderNum) {
    return -1;
  }

  MOCK_METHOD3(rtStreamCreateWithFlags, rtError_t(rtStream_t *, int32_t, uint32_t));
};

int32_t g_call_stream_create_times = 0;
rtError_t MockrtStreamCreateWithFlags(rtStream_t *stream, int32_t priority, uint32_t flags) {
  ++g_call_stream_create_times;
  return 0;
}

class ExternalAllocatorUtStub : public Allocator {
 public:
  MemBlock *Malloc(size_t size) override {
    block_ = new (std::nothrow) MemBlock(*this, &mem, size);
    return block_;
  }
  MemBlock *MallocAdvise(size_t size, void *addr) override {
    block_ = new (std::nothrow) MemBlock(*this, &mem, size);
    advise_cnt++;
    return block_;
  }
  void Free(MemBlock *block) override {
    delete block;
    if (block == block_) {
      block_ = nullptr;
    }
  }
  MemBlock *GetBlockAddr() {
    return block_;
  }
  uint64_t GetAdviseCnt() {
    return advise_cnt;
  }
 private:
  uint64_t mem = 0;
  MemBlock *block_{nullptr};
  uint64_t advise_cnt = 0U;
};

static GeRootModelPtr ConstructGeRootModel(
    const std::vector<std::pair<ccKernelType, const std::string>> &kernel_type_so_names) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_root_model->subgraph_instance_name_to_model_["graph"] = ge_model;
  ge_model->SetGraph(graph);

  domi::ModelTaskDef model_task_def;
  std::shared_ptr<domi::ModelTaskDef> model_task_def_ptr = make_shared<domi::ModelTaskDef>(model_task_def);
  for (const auto &item : kernel_type_so_names) {
    domi::TaskDef *task_def = model_task_def_ptr->add_task();
    ge_model->SetModelTaskDef(model_task_def_ptr);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL));

    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_so_name(item.second);
    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_kernel_type(static_cast<uint32_t>(item.first));
    context->set_op_index(1);
  }
  return ge_root_model;
}

static void ConstructOpMasterDeviceSo(const std::string &opp_path, const size_t built_in_num, const size_t cust_num,
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
}

class UtestModelManagerModelManager : public testing::Test {
 protected:
  void SetUp() {
    RTS_STUB_SETUP();
    auto acl_runtime = std::make_shared<AclRuntimeStub>();
    ge::AclRuntimeStub::SetInstance(acl_runtime);
  }
  void TearDown() override {
    ge::AclRuntimeStub::Reset();
    unsetenv("NPU_COLLECT_PATH");
    unsetenv("NPU_COLLECT_PATH_EXE");
    EXPECT_TRUE(ModelManager::GetInstance().model_map_.empty());
    EXPECT_TRUE(ModelManager::GetInstance().hybrid_model_map_.empty());
    remove("valid_path");
    RTS_STUB_TEARDOWN();
  }
};

void CreateGraph(Graph &graph) {
  TensorDesc desc(ge::Shape({1, 3, 224, 224}));
  uint32_t size = desc.GetShape().GetShapeSize();
  desc.SetSize(size);
  auto data = op::Data("Data").set_attr_index(0);
  data.update_input_desc_data(desc);
  data.update_output_desc_out(desc);

  auto flatten = op::Flatten("Flatten").set_input_x(data, data.name_out_out());

  std::vector<Operator> inputs{data};
  std::vector<Operator> outputs{flatten};
  std::vector<Operator> targets{flatten};
  // Graph graph("test_graph");
  graph.SetInputs(inputs).SetOutputs(outputs).SetTargets(targets);
}

void GenUnencryptModelData(ModelData &data) {
  const int model_len = 10;
  data.model_len = sizeof(ModelFileHeader) + model_len;
  data.model_data = new uint8_t[data.model_len];
  memset(data.model_data, 0, data.model_len);

  ModelFileHeader *header = (ModelFileHeader *) data.model_data;
  header->magic = MODEL_FILE_MAGIC_NUM;
  header->version = MODEL_VERSION;
  header->is_encrypt = ModelEncryptType::UNENCRYPTED;
  header->length = model_len;
  header->is_checksum = ModelCheckType::CHECK;
  header->model_num = 1U;
}
struct PartitionInfo {
  size_t partition_num = PARTITION_SIZE;
  size_t weight_size = 0U;
};
// 只会定制化设置Weight Partition的size，之后的size均为0
void SetPartitionTables(ModelData &data, const std::vector<PartitionInfo> &partition_infos,
                        const uint64_t model_len = 1024, const uint32_t model_num = 1U) {
  ASSERT_EQ(partition_infos.size(), model_num);

  uint8_t *model_data = reinterpret_cast<uint8_t *>(data.model_data);
  uint64_t mem_offset = sizeof(ModelFileHeader);
  for (uint32_t i = 0; i < model_num; ++i) {

    TinyModelPartitionTable *tiny_partition_table = reinterpret_cast<TinyModelPartitionTable *>(model_data + mem_offset);
    tiny_partition_table->num = partition_infos[i].partition_num;
    mem_offset += sizeof(TinyModelPartitionTable) + sizeof(TinyModelPartitionMemInfo) * tiny_partition_table->num;

    Model model;
    ComputeGraphPtr graph = make_shared<ComputeGraph>("default");
    model.SetGraph(graph);
    model.SetVersion(123);

    GeModelPtr ge_model = MakeShared<GeModel>();
    ge::AttrUtils::SetStr(ge_model, ge::ATTR_MODEL_OPP_VERSION, "3.20.T100.0.B356");
    model.SetAttr(ge_model->MutableAttrMap());

    Buffer buffer;
    model.Save(buffer);
    EXPECT_TRUE(mem_offset + buffer.GetSize() < model_len);
    memcpy(model_data + mem_offset, buffer.GetData(), buffer.GetSize());

    for (size_t j = 0; j < partition_infos[i].partition_num; ++j) {
      auto &partition_info = tiny_partition_table->partition[j];
      auto type = (ModelPartitionType)(ModelPartitionType::MODEL_DEF + j);
      partition_info.type = type;
      if (type == ModelPartitionType::MODEL_DEF) {
        partition_info.mem_size = buffer.GetSize();
      } else if (type == ModelPartitionType::WEIGHTS_DATA) {
        partition_info.mem_size = partition_infos[i].weight_size;
      } else {
        partition_info.mem_size = 0U;
      }
      mem_offset += partition_info.mem_size;
    }
  }
  EXPECT_TRUE(mem_offset < model_len);
  ModelFileHeader *header = new(data.model_data) ModelFileHeader;
  header->length = mem_offset - sizeof(ModelFileHeader);
  header->model_num = model_num;
  header->version = ge::MODEL_VERSION;
  data.model_len = mem_offset;
}

void LoadFlexibleModelData(ModelData &data, const std::vector<PartitionInfo> &partition_infos,
                           const uint64_t model_len = 512, const uint32_t model_num = 1U) {
  data.model_len = model_len;
  data.model_data = new uint8_t[data.model_len];

  SetPartitionTables(data, partition_infos, model_len, model_num);
}

void LoadStandardModelData(ModelData &data, const uint64_t model_len = 1024, const uint32_t model_num = 1U) {
  data.model_len = model_len;
  data.model_data = new uint8_t[data.model_len];
  std::vector<PartitionInfo> partition_infos(model_num);
  SetPartitionTables(data, partition_infos, model_len, model_num);
}

class DModelListener : public ModelListener {
 public:
  DModelListener() {};
  uint32_t OnComputeDone(uint32_t model_id, uint32_t data_index,
                         uint32_t resultCode, std::vector<gert::Tensor> &outputs) { return 0; }
};

class StubExecutor : public Executor {
 public:
  Status LoadGraph(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node,
                   const rtStream_t stream = nullptr) override {
    return SUCCESS;
  }

  Status UnloadGraph(const GeRootModelPtr &ge_root_model, const uint32_t graph_id) override {
    return SUCCESS;
  }

  Status PushRunArgs(const std::shared_ptr<RunArgs> &args) override {
    return SUCCESS;
  }

   Status RunGraph(const GraphNodePtr &graph_node, const GraphId graph_id,
                   const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs) override {
     return SUCCESS;
   }

  Status RunGraphWithStream(const GraphNodePtr &graph_node, const GraphId graph_id, const rtStream_t stream,
                            const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs) override {
    return SUCCESS;
  }

  Status ExecuteGraphWithStream(const GraphNodePtr &graph_node, const GraphId graph_id,
                               rtStream_t const stream, const std::vector<gert::Tensor> &inputs,
                               std::vector<gert::Tensor> &outputs) override {
    return SUCCESS;
  }

  Status UpdateFeatureMemoryBase(const GraphNodePtr &graph_node, const uintptr_t mem_base, const size_t size) override {
    (void)graph_node;
    mem_base_ = mem_base;
    mem_base_size_ = size;
    return SUCCESS;
  }

  Status PaRemapped(const GraphNodePtr &graph_node, const uint64_t va, const uint64_t new_pa,
                    const uint64_t len, std::vector<std::pair<uint64_t, uint64_t>> &cross_ranges) override {
    return SUCCESS;
  }
  uintptr_t mem_base_;
  size_t mem_base_size_;
};

TEST_F(UtestModelManagerModelManager, case_is_need_hybrid_load) {
  ModelManager mm;
  ComputeGraphPtr root_graph = std::make_shared<ComputeGraph>("graph");
  ge::GeRootModel model;
  EXPECT_EQ(mm.IsNeedHybridLoad(model), false);
  model.SetRootGraph(root_graph);
  EXPECT_EQ(mm.IsNeedHybridLoad(model), false);
}

TEST_F(UtestModelManagerModelManager, case_load_incorrect_param) {
  ModelManager mm;
  uint32_t model_id = 0;
  ModelData data;
  // Load allow listener is null
  const ModelParam param;
  EXPECT_EQ(mm.LoadModelOffline(data, param, model_id), ACL_ERROR_GE_PARAM_INVALID);
}

TEST_F(UtestModelManagerModelManager, case_load_model_len_too_short) {
  ModelManager mm;
  ModelData data;
  data.model_len = 10;
  data.model_data = (void *) &data;
  uint32_t model_id = 1;
  const ModelParam param;
  EXPECT_EQ(mm.LoadModelOffline(data, param, model_id), ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);
  data.model_data = nullptr;
}

TEST_F(UtestModelManagerModelManager, case_load_model_len_not_match) {
  ModelManager mm;
  ModelData data;
  GenUnencryptModelData(data);
  data.model_len = sizeof(ModelFileHeader) + 1;
  uint32_t model_id = 1;
  const ModelParam param;
  EXPECT_EQ(mm.LoadModelOffline(data, param, model_id), ACL_ERROR_GE_PARAM_INVALID);
  delete[](uint8_t *) data.model_data;
}

TEST_F(UtestModelManagerModelManager, case_load_model_encypt_not_match) {
  ModelManager mm;
  ModelData data;
  GenUnencryptModelData(data);
  data.key = ENC_KEY;
  uint32_t model_id = 1;
  const ModelParam param;
  EXPECT_EQ(mm.LoadModelOffline(data, param, model_id), ACL_ERROR_GE_PARAM_INVALID);
  delete[](uint8_t *) data.model_data;
}

TEST_F(UtestModelManagerModelManager, case_load_model_encypt_type_unsupported) {
  ModelManager mm;
  ModelData data;
  GenUnencryptModelData(data);
  ModelFileHeader *header = (ModelFileHeader *) data.model_data;
  header->is_encrypt = 255;
  uint32_t model_id = 1;
  // Error for: LoadModelPartitionTable: Invalid partition_table->num:0
  const ModelParam param;
  EXPECT_EQ(mm.LoadModelOffline(data, param, model_id), ACL_ERROR_GE_PARAM_INVALID);
  delete[](uint8_t *) data.model_data;
}

TEST_F(UtestModelManagerModelManager, case_load_model_data_success) {
  ModelData data;
  LoadStandardModelData(data);
  ModelFileHeader &header = *static_cast<ModelFileHeader *>(data.model_data);
  EXPECT_EQ(header.model_num, 1U);

  {
    ModelManager mm;
    uint32_t model_id = std::numeric_limits<uint32_t>::max();
    const ModelParam param;
    EXPECT_EQ(mm.LoadModelOffline(data, param, model_id), SUCCESS);
  }

  delete [] (uint8_t *)data.model_data;
}

TEST_F(UtestModelManagerModelManager, case_load_model_data_with_rtsession_success) {
  ModelData data;
  LoadStandardModelData(data);
  ModelFileHeader &header = *static_cast<ModelFileHeader *>(data.model_data);
  EXPECT_EQ(header.model_num, 1U);

  {
    gert::RtSession rt_session;
    rt_session.session_id_ = 889900UL;
    ModelManager mm;
    uint32_t model_id = std::numeric_limits<uint32_t>::max();
    const ModelParam param;
    EXPECT_EQ(mm.LoadModelOffline(data, param, model_id, &rt_session), SUCCESS);
  }

  delete [] (uint8_t *)data.model_data;
}


TEST_F(UtestModelManagerModelManager, load_unknown_shape_model_data_success) {
  ModelData data;
  LoadStandardModelData(data, 1024U, 2U);
  ModelFileHeader &header = *static_cast<ModelFileHeader *>(data.model_data);
  EXPECT_EQ(header.model_num, 2U);

  {
    std::string opp_path = "./";
    std::string opp_version = "version.info";
    setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);
    system(("touch " + opp_version).c_str());
    system(("echo 'Version=3.20.T100.0.B356' > " + opp_version).c_str());

    ModelManager mm;
    uint32_t model_id = std::numeric_limits<uint32_t>::max();
    const ModelParam param;
    EXPECT_EQ(mm.LoadModelOffline(data, param, model_id), SUCCESS);

    system(("rm -f " + opp_version).c_str());
    unsetenv("ASCEND_OPP_PATH");
  }

  {
    OmFileLoadHelper loader;
    const uint64_t model_data_size = data.model_len - sizeof(ModelFileHeader);
    uint8_t *const model_data = static_cast<uint8_t *>(data.model_data) + sizeof(ModelFileHeader);
    size_t mem_offset = 0U;
    EXPECT_EQ(loader.LoadModelPartitionTable(model_data, model_data_size, 1U, mem_offset), FAILED);
    EXPECT_TRUE(loader.model_contexts_.empty());
    EXPECT_EQ(loader.LoadModelPartitionTable(model_data, model_data_size, 1U), FAILED);
    EXPECT_FALSE(loader.model_contexts_.empty());
  }

  {
    OmFileLoadHelper loader;
    const uint64_t model_data_size = data.model_len - sizeof(ModelFileHeader);
    uint8_t *const model_data = static_cast<uint8_t *>(data.model_data) + sizeof(ModelFileHeader);
    size_t mem_offset = 0U;
    EXPECT_EQ(loader.LoadModelPartitionTable(model_data, model_data_size, 1U, mem_offset), FAILED);
    EXPECT_TRUE(loader.model_contexts_.empty());
    EXPECT_EQ(loader.LoadModelPartitionTable(model_data, model_data_size, 1U), FAILED);
    EXPECT_FALSE(loader.model_contexts_.empty());
  }

  delete [] (uint8_t *)data.model_data;
}

TEST_F(UtestModelManagerModelManager, test_launch_kernel_cust_aicpu_resource_id_not_found) {
  ModelManager mm;
  uintptr_t resource_id = 2;
  std::vector<char> kernel_bin(256);
  auto &cust_resource = mm.cust_aicpu_so_[resource_id];
  auto tbe_kernel = std::shared_ptr<OpKernelBin>(new OpKernelBin("deleteCustOp", std::move(kernel_bin)));
  cust_resource["deleteCustOp"] = {tbe_kernel, 2UL};
  EXPECT_FALSE(mm.cust_aicpu_so_.empty());
  EXPECT_EQ(mm.LaunchKernelCustAicpuSo("deleteCustOp"), SUCCESS);
}

TEST_F(UtestModelManagerModelManager, test_launch_kernel_cust_aicpu) {
  ModelManager mm;

  // cust_aicpu_so_ is empty.
  EXPECT_EQ(mm.LaunchKernelCustAicpuSo("empty_cust_aicpu"), SUCCESS);

  // deleteCustOp after Launch will deleted.
  uintptr_t resource_id = 1;    // for rtCtxGetCurrent stub
  std::vector<char> kernel_bin(256);
  auto &cust_resource_001 = mm.cust_aicpu_so_[resource_id];
  auto tbe_kernel = std::shared_ptr<OpKernelBin>(new OpKernelBin("deleteCustOp", std::move(kernel_bin)));
  cust_resource_001["deleteCustOp"] = {tbe_kernel, 2UL};

  EXPECT_FALSE(mm.cust_aicpu_so_.empty());
  EXPECT_EQ(mm.LaunchKernelCustAicpuSo("deleteCustOp"), SUCCESS);
  EXPECT_EQ(mm.LaunchKernelCustAicpuSo("deleteCustOp"), SUCCESS);
  EXPECT_EQ(mm.cust_aicpu_so_[resource_id]["deleteCustOp"].launch_count, 0UL);
}

TEST_F(UtestModelManagerModelManager, test_cust_aicpu_repeated_launch) {
  ModelManager mm;
  // deleteCustOp after Launch will deleted.
  uintptr_t resource_id = 1;    // for rtCtxGetCurrent stub
  std::vector<char> kernel_bin(256);
  auto &cust_resource_001 = mm.cust_aicpu_so_[resource_id];
  auto tbe_kernel = std::shared_ptr<OpKernelBin>(new OpKernelBin("op", std::move(kernel_bin)));
  cust_resource_001["op"] = {tbe_kernel, 0UL};

  EXPECT_EQ(mm.cust_aicpu_so_[resource_id].size(), 1UL);
  EXPECT_EQ(mm.LaunchKernelCustAicpuSo("batchLoadsoFrombuf"), SUCCESS);
  EXPECT_EQ(mm.cust_aicpu_so_[resource_id]["op"].launch_count, 1UL);

  auto tbe_kernel1 = std::shared_ptr<OpKernelBin>(new OpKernelBin("op1", std::move(kernel_bin)));
  cust_resource_001["op1"] = {tbe_kernel1, 0UL};
  EXPECT_EQ(mm.LaunchKernelCustAicpuSo("batchLoadsoFrombuf"), SUCCESS);
  EXPECT_EQ(mm.cust_aicpu_so_[resource_id].size(), 2UL);
  EXPECT_EQ(mm.cust_aicpu_so_[resource_id]["op"].launch_count, 2UL);
  EXPECT_EQ(mm.cust_aicpu_so_[resource_id]["op1"].launch_count, 1UL);

  EXPECT_EQ(mm.LaunchKernelCustAicpuSo("deleteCustOp"), SUCCESS);
  EXPECT_EQ(mm.cust_aicpu_so_[resource_id]["op"].launch_count, 1UL);
  EXPECT_EQ(mm.cust_aicpu_so_[resource_id]["op1"].launch_count, 0UL);
  EXPECT_EQ(mm.LaunchKernelCustAicpuSo("deleteCustOp"), SUCCESS);
  EXPECT_EQ(mm.cust_aicpu_so_[resource_id]["op"].launch_count, 0UL);
  EXPECT_EQ(mm.cust_aicpu_so_[resource_id]["op1"].launch_count, 0UL);
}

TEST_F(UtestModelManagerModelManager, test_launch_custom_aicpu_platform_infos_from_op_master) {
  constexpr const char_t *kAscendHomePath = "ASCEND_HOME_PATH";
  char old_path[MMPA_MAX_PATH] = {0};
  (void)mmGetEnv(kAscendHomePath, old_path, MMPA_MAX_PATH);
  std::string ascend_home_path("/test/ascend_path");
  mmSetEnv(kAscendHomePath, ascend_home_path.c_str(), 1);

  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  auto &mm = ModelManager::GetInstance();
  DavinciModel model(0, nullptr);
  ModelHelper model_helper;
  model_helper.HandleDeviceInfo(model.platform_infos_);
  EXPECT_EQ(model.LaunchCustPlatformInfos(), SUCCESS);

  // ===首次load阶段申请内存，相同核数配置根据cust_platform_infos_addr_to_launch_复用缓存===
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr op_desc_ptr_0 = std::make_shared<ge::OpDesc>();
  op_desc_ptr_0->SetName("Variable0");
  ge::NodePtr variable_node_0 = graph->AddNode(op_desc_ptr_0);
  void *cust_platform_infos_addr_global_0 = nullptr;
  EXPECT_EQ(model.LoadCustPlatformInfos(cust_platform_infos_addr_global_0, variable_node_0), SUCCESS);

  ge::OpDescPtr op_desc_ptr_1 = std::make_shared<ge::OpDesc>();
  op_desc_ptr_1->SetName("Variable1");
  AttrUtils::SetStr(op_desc_ptr_1, "_op_aicore_num", "2");
  AttrUtils::SetStr(op_desc_ptr_1, "_op_vectorcore_num", "4");
  ge::NodePtr variable_node_1 = graph->AddNode(op_desc_ptr_1);
  void *cust_platform_infos_addr_2_4_1 = nullptr;
  EXPECT_EQ(model.LoadCustPlatformInfos(cust_platform_infos_addr_2_4_1, variable_node_1), SUCCESS);
  ge::OpDescPtr op_desc_ptr2 = std::make_shared<ge::OpDesc>();
  op_desc_ptr2->SetName("Variable2");
  AttrUtils::SetStr(op_desc_ptr2, "_op_aicore_num", "2");
  AttrUtils::SetStr(op_desc_ptr2, "_op_vectorcore_num", "4");
  ge::NodePtr variable_node2 = graph->AddNode(op_desc_ptr2);
  void *cust_platform_infos_addr_2_4_2 = nullptr;
  EXPECT_EQ(model.LoadCustPlatformInfos(cust_platform_infos_addr_2_4_2, variable_node2), SUCCESS);
  EXPECT_EQ(cust_platform_infos_addr_2_4_1, cust_platform_infos_addr_2_4_2);

  ge::OpDescPtr op_desc_ptr6 = std::make_shared<ge::OpDesc>();
  op_desc_ptr6->SetName("Variable6");
  AttrUtils::SetStr(op_desc_ptr6, "_op_aicore_num", "4");
  AttrUtils::SetStr(op_desc_ptr6, "_op_vectorcore_num", "8");
  ge::NodePtr variable_node6 = graph->AddNode(op_desc_ptr6);
  void *cust_platform_infos_addr_4_8_1 = nullptr;
  EXPECT_EQ(model.LoadCustPlatformInfos(cust_platform_infos_addr_4_8_1, variable_node6), SUCCESS);

  EXPECT_NE(cust_platform_infos_addr_global_0, cust_platform_infos_addr_2_4_1);
  EXPECT_NE(cust_platform_infos_addr_2_4_1, cust_platform_infos_addr_4_8_1);
  EXPECT_NE(cust_platform_infos_addr_4_8_1, cust_platform_infos_addr_global_0);

  uintptr_t resource_id = 1;    // for rtCtxGetCurrent stub.
  std::string so_name = "b.so";
  std::vector<char> kernel_bin(256);
  auto &cust_resource_001 = mm.cust_aicpu_so_[resource_id];
  auto tbe_kernel1 = std::shared_ptr<OpKernelBin>(new OpKernelBin("a.so", std::move(kernel_bin)));
  cust_resource_001["a.so"] = {tbe_kernel1, 0UL};
  auto tbe_kernel2 = std::shared_ptr<OpKernelBin>(new OpKernelBin(so_name, std::move(kernel_bin)));
  cust_resource_001[so_name] = {tbe_kernel2, 1UL};
  EXPECT_EQ(mm.GetPlatformInfosSoName(so_name), SUCCESS);
  EXPECT_EQ(mm.cust_aicpu_so_[resource_id].size(), 2UL);
  std::unique_ptr<char[]> so_bin = std::unique_ptr<char[]>(new (std::nothrow) char[so_name.length()]);
  (void) memcpy_s(so_bin.get(), so_name.length(), so_name.data(), so_name.length());
  OpSoBinPtr op_so_bin = std::make_shared<OpSoBin>(so_name, "", std::move(so_bin), so_name.length());
  mm.cust_op_master_so_names_to_bin_[so_name] = op_so_bin;
  EXPECT_EQ(mm.cust_op_master_so_names_to_bin_.size(), 1UL);
  rtBinHandle bin_handle = (void *)0x12000;
  mm.SetPlatformBinHandle(bin_handle);
  EXPECT_EQ(model.LaunchCustPlatformInfos(), SUCCESS);

  // ===launch过再申请内存，相同核数配置根据cust_platform_infos_addr_to_launch_复用缓存===
  ge::OpDescPtr op_desc_ptr3 = std::make_shared<ge::OpDesc>();
  op_desc_ptr3->SetName("Variable3");
  ge::NodePtr variable_node3 = graph->AddNode(op_desc_ptr3);
  void *cust_platform_infos_addr_global_1 = nullptr;
  EXPECT_EQ(model.LoadCustPlatformInfos(cust_platform_infos_addr_global_1, variable_node3), SUCCESS);  // 命中cust_platform_infos_addr_缓存

  ge::OpDescPtr op_desc_ptr4 = std::make_shared<ge::OpDesc>();
  op_desc_ptr4->SetName("Variable4");
  AttrUtils::SetStr(op_desc_ptr4, "_op_aicore_num", "2");
  AttrUtils::SetStr(op_desc_ptr4, "_op_vectorcore_num", "4");
  ge::NodePtr variable_node4 = graph->AddNode(op_desc_ptr4);
  void *cust_platform_infos_addr_2_4_3 = nullptr;
  EXPECT_EQ(model.LoadCustPlatformInfos(cust_platform_infos_addr_2_4_3, variable_node4), SUCCESS);  // 命中cust_platform_infos_addr_缓存

  ge::OpDescPtr op_desc_ptr5 = std::make_shared<ge::OpDesc>();
  op_desc_ptr5->SetName("Variable5");
  AttrUtils::SetStr(op_desc_ptr5, "_op_aicore_num", "4");
  AttrUtils::SetStr(op_desc_ptr5, "_op_vectorcore_num", "8");
  ge::NodePtr variable_node5 = graph->AddNode(op_desc_ptr5);
  void *cust_platform_infos_addr_4_8_2 = nullptr;
  EXPECT_EQ(model.LoadCustPlatformInfos(cust_platform_infos_addr_4_8_2, variable_node5), SUCCESS);  // 命中cust_platform_infos_addr_缓存

  ge::OpDescPtr op_desc_ptr7 = std::make_shared<ge::OpDesc>();
  op_desc_ptr7->SetName("Variable6");
  AttrUtils::SetStr(op_desc_ptr7, "_op_aicore_num", "2");
  ge::NodePtr variable_node7 = graph->AddNode(op_desc_ptr7);
  void *cust_platform_infos_addr_2_null_1 = nullptr;
  EXPECT_EQ(model.LoadCustPlatformInfos(cust_platform_infos_addr_2_null_1, variable_node7), SUCCESS);  // 命中cust_platform_infos_addr_缓存

  EXPECT_EQ(cust_platform_infos_addr_global_0, cust_platform_infos_addr_global_1);
  EXPECT_EQ(cust_platform_infos_addr_2_4_1, cust_platform_infos_addr_2_4_3);
  EXPECT_EQ(cust_platform_infos_addr_4_8_1, cust_platform_infos_addr_4_8_2);

  EXPECT_NE(cust_platform_infos_addr_global_1, cust_platform_infos_addr_2_4_3);
  EXPECT_NE(cust_platform_infos_addr_global_1, cust_platform_infos_addr_4_8_2);
  EXPECT_NE(cust_platform_infos_addr_2_4_3, cust_platform_infos_addr_4_8_2);

  EXPECT_NE(cust_platform_infos_addr_2_null_1, cust_platform_infos_addr_global_1);
  EXPECT_NE(cust_platform_infos_addr_2_null_1, cust_platform_infos_addr_2_4_3);
  EXPECT_NE(cust_platform_infos_addr_2_null_1, cust_platform_infos_addr_4_8_2);

  EXPECT_EQ(model.LaunchCustPlatformInfos(), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
  MemManager::Instance().MemInstance(RT_MEMORY_HBM).ReleaseResource();

  mmSetEnv(kAscendHomePath, old_path, 1);
}

TEST_F(UtestModelManagerModelManager, test_launch_custom_aicpu_platform_infos_from_platform_so) {
  constexpr const char_t *kAscendHomePath = "ASCEND_HOME_PATH";
  char old_path[MMPA_MAX_PATH] = {0};
  (void)mmGetEnv(kAscendHomePath, old_path, MMPA_MAX_PATH);
  auto work_path = EnvPath().GetAirBasePath();
  mmSetEnv(kAscendHomePath, (work_path + "/build_ut/tests/depends/op_stub/so_stub").c_str(), 1);

  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  auto &mm = ModelManager::GetInstance();
  DavinciModel model(0, nullptr);
  ModelHelper model_helper;
  model_helper.HandleDeviceInfo(model.platform_infos_);
  EXPECT_EQ(model.LaunchCustPlatformInfos(), SUCCESS);

  // ===首次load阶段申请内存，相同核数配置根据cust_platform_infos_addr_to_launch_复用缓存===
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr op_desc_ptr_0 = std::make_shared<ge::OpDesc>();
  op_desc_ptr_0->SetName("Variable0");
  ge::NodePtr variable_node_0 = graph->AddNode(op_desc_ptr_0);
  void *cust_platform_infos_addr_global_0 = nullptr;
  EXPECT_EQ(model.LoadCustPlatformInfos(cust_platform_infos_addr_global_0, variable_node_0), SUCCESS);

  ge::OpDescPtr op_desc_ptr_1 = std::make_shared<ge::OpDesc>();
  op_desc_ptr_1->SetName("Variable1");
  AttrUtils::SetStr(op_desc_ptr_1, "_op_aicore_num", "2");
  AttrUtils::SetStr(op_desc_ptr_1, "_op_vectorcore_num", "4");
  ge::NodePtr variable_node_1 = graph->AddNode(op_desc_ptr_1);
  void *cust_platform_infos_addr_2_4_1 = nullptr;
  EXPECT_EQ(model.LoadCustPlatformInfos(cust_platform_infos_addr_2_4_1, variable_node_1), SUCCESS);
  ge::OpDescPtr op_desc_ptr2 = std::make_shared<ge::OpDesc>();
  op_desc_ptr2->SetName("Variable2");
  AttrUtils::SetStr(op_desc_ptr2, "_op_aicore_num", "2");
  AttrUtils::SetStr(op_desc_ptr2, "_op_vectorcore_num", "4");
  ge::NodePtr variable_node2 = graph->AddNode(op_desc_ptr2);
  void *cust_platform_infos_addr_2_4_2 = nullptr;
  EXPECT_EQ(model.LoadCustPlatformInfos(cust_platform_infos_addr_2_4_2, variable_node2), SUCCESS);
  EXPECT_EQ(cust_platform_infos_addr_2_4_1, cust_platform_infos_addr_2_4_2);

  ge::OpDescPtr op_desc_ptr6 = std::make_shared<ge::OpDesc>();
  op_desc_ptr6->SetName("Variable6");
  AttrUtils::SetStr(op_desc_ptr6, "_op_aicore_num", "4");
  AttrUtils::SetStr(op_desc_ptr6, "_op_vectorcore_num", "8");
  ge::NodePtr variable_node6 = graph->AddNode(op_desc_ptr6);
  void *cust_platform_infos_addr_4_8_1 = nullptr;
  EXPECT_EQ(model.LoadCustPlatformInfos(cust_platform_infos_addr_4_8_1, variable_node6), SUCCESS);

  EXPECT_NE(cust_platform_infos_addr_global_0, cust_platform_infos_addr_2_4_1);
  EXPECT_NE(cust_platform_infos_addr_2_4_1, cust_platform_infos_addr_4_8_1);
  EXPECT_NE(cust_platform_infos_addr_4_8_1, cust_platform_infos_addr_global_0);

  uintptr_t resource_id = 1;    // for rtCtxGetCurrent stub.
  std::string so_name = "b.so";
  std::vector<char> kernel_bin(256);
  auto &cust_resource_001 = mm.cust_aicpu_so_[resource_id];
  auto tbe_kernel1 = std::shared_ptr<OpKernelBin>(new OpKernelBin("a.so", std::move(kernel_bin)));
  cust_resource_001["a.so"] = {tbe_kernel1, 0UL};
  auto tbe_kernel2 = std::shared_ptr<OpKernelBin>(new OpKernelBin(so_name, std::move(kernel_bin)));
  cust_resource_001[so_name] = {tbe_kernel2, 1UL};
  EXPECT_EQ(mm.GetPlatformInfosSoName(so_name), SUCCESS);
  EXPECT_EQ(mm.cust_aicpu_so_[resource_id].size(), 2UL);
  std::unique_ptr<char[]> so_bin = std::unique_ptr<char[]>(new (std::nothrow) char[so_name.length()]);
  (void) memcpy_s(so_bin.get(), so_name.length(), so_name.data(), so_name.length());
  OpSoBinPtr op_so_bin = std::make_shared<OpSoBin>(so_name, "", std::move(so_bin), so_name.length());
  mm.cust_op_master_so_names_to_bin_[so_name] = op_so_bin;
  EXPECT_EQ(mm.cust_op_master_so_names_to_bin_.size(), 1UL);
  rtBinHandle bin_handle = (void *)0x12000;
  mm.SetPlatformBinHandle(bin_handle);
  EXPECT_EQ(model.LaunchCustPlatformInfos(), SUCCESS);

  // ===launch过再申请内存，相同核数配置根据cust_platform_infos_addr_to_launch_复用缓存===
  ge::OpDescPtr op_desc_ptr3 = std::make_shared<ge::OpDesc>();
  op_desc_ptr3->SetName("Variable3");
  ge::NodePtr variable_node3 = graph->AddNode(op_desc_ptr3);
  void *cust_platform_infos_addr_global_1 = nullptr;
  EXPECT_EQ(model.LoadCustPlatformInfos(cust_platform_infos_addr_global_1, variable_node3), SUCCESS);  // 命中cust_platform_infos_addr_缓存

  ge::OpDescPtr op_desc_ptr4 = std::make_shared<ge::OpDesc>();
  op_desc_ptr4->SetName("Variable4");
  AttrUtils::SetStr(op_desc_ptr4, "_op_aicore_num", "2");
  AttrUtils::SetStr(op_desc_ptr4, "_op_vectorcore_num", "4");
  ge::NodePtr variable_node4 = graph->AddNode(op_desc_ptr4);
  void *cust_platform_infos_addr_2_4_3 = nullptr;
  EXPECT_EQ(model.LoadCustPlatformInfos(cust_platform_infos_addr_2_4_3, variable_node4), SUCCESS);  // 命中cust_platform_infos_addr_缓存

  ge::OpDescPtr op_desc_ptr5 = std::make_shared<ge::OpDesc>();
  op_desc_ptr5->SetName("Variable5");
  AttrUtils::SetStr(op_desc_ptr5, "_op_aicore_num", "4");
  AttrUtils::SetStr(op_desc_ptr5, "_op_vectorcore_num", "8");
  ge::NodePtr variable_node5 = graph->AddNode(op_desc_ptr5);
  void *cust_platform_infos_addr_4_8_2 = nullptr;
  EXPECT_EQ(model.LoadCustPlatformInfos(cust_platform_infos_addr_4_8_2, variable_node5), SUCCESS);  // 命中cust_platform_infos_addr_缓存

  ge::OpDescPtr op_desc_ptr7 = std::make_shared<ge::OpDesc>();
  op_desc_ptr7->SetName("Variable6");
  AttrUtils::SetStr(op_desc_ptr7, "_op_aicore_num", "2");
  ge::NodePtr variable_node7 = graph->AddNode(op_desc_ptr7);
  void *cust_platform_infos_addr_2_null_1 = nullptr;
  EXPECT_EQ(model.LoadCustPlatformInfos(cust_platform_infos_addr_2_null_1, variable_node7), SUCCESS);  // 命中cust_platform_infos_addr_缓存

  EXPECT_EQ(cust_platform_infos_addr_global_0, cust_platform_infos_addr_global_1);
  EXPECT_EQ(cust_platform_infos_addr_2_4_1, cust_platform_infos_addr_2_4_3);
  EXPECT_EQ(cust_platform_infos_addr_4_8_1, cust_platform_infos_addr_4_8_2);

  EXPECT_NE(cust_platform_infos_addr_global_1, cust_platform_infos_addr_2_4_3);
  EXPECT_NE(cust_platform_infos_addr_global_1, cust_platform_infos_addr_4_8_2);
  EXPECT_NE(cust_platform_infos_addr_2_4_3, cust_platform_infos_addr_4_8_2);

  EXPECT_NE(cust_platform_infos_addr_2_null_1, cust_platform_infos_addr_global_1);
  EXPECT_NE(cust_platform_infos_addr_2_null_1, cust_platform_infos_addr_2_4_3);
  EXPECT_NE(cust_platform_infos_addr_2_null_1, cust_platform_infos_addr_4_8_2);

  EXPECT_EQ(model.LaunchCustPlatformInfos(), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
  MemManager::Instance().MemInstance(RT_MEMORY_HBM).ReleaseResource();

  mmSetEnv(kAscendHomePath, old_path, 1);
}

TEST_F(UtestModelManagerModelManager, test_launch_kernel_builtin_aicpu) {
  ModelManager mm;

  // builtin_aicpu_so_ is empty.
  uint32_t resource_id = 1;
  EXPECT_EQ(mm.LaunchKernelBuiltinAicpuSo("empty_builtin_aicpu", resource_id), SUCCESS);

  std::vector<char> kernel_bin(256);
  auto &builtin_resource_001 = mm.builtin_aicpu_so_[resource_id];

  auto tbe_kernel = std::shared_ptr<OpKernelBin>(new OpKernelBin("so", std::move(kernel_bin)));
  builtin_resource_001["so"] = {tbe_kernel, 2UL};

  EXPECT_FALSE(mm.builtin_aicpu_so_.empty());
  EXPECT_EQ(mm.LaunchKernelBuiltinAicpuSo("xxxx", 2), SUCCESS);
  EXPECT_EQ(mm.LaunchKernelBuiltinAicpuSo("xxxx", resource_id), SUCCESS);
  EXPECT_EQ(mm.LaunchKernelBuiltinAicpuSo("RunAicpuPreprocessUnloadSoLaunch", resource_id), SUCCESS);
  EXPECT_EQ(mm.LaunchKernelBuiltinAicpuSo("RunAicpuPreprocessUnloadSoLaunch", resource_id), SUCCESS);
  EXPECT_EQ(mm.builtin_aicpu_so_[resource_id]["so"].launch_count, 0UL);
}

TEST_F(UtestModelManagerModelManager, test_builtin_aicpu_repeated_launch) {
  ModelManager mm;

  uintptr_t resource_id = 1;
  std::vector<char> kernel_bin(256);
  auto &builtin_resource_001 = mm.builtin_aicpu_so_[resource_id];
  auto tbe_kernel = std::shared_ptr<OpKernelBin>(new OpKernelBin("Ascendxxxx-v7.5-libopmaster.so",
                                                 std::move(kernel_bin)));
  builtin_resource_001["so"] = {tbe_kernel, 0UL};

  EXPECT_EQ(mm.builtin_aicpu_so_[resource_id].size(), 1UL);
  EXPECT_EQ(mm.LaunchKernelBuiltinAicpuSo("RunAicpuPreprocessLoadSoLaunch", resource_id), SUCCESS);
  EXPECT_EQ(mm.builtin_aicpu_so_[resource_id]["so"].launch_count, 1UL);

  auto tbe_kernel1 = std::shared_ptr<OpKernelBin>(new OpKernelBin("so1", std::move(kernel_bin)));
  builtin_resource_001["so1"] = {tbe_kernel1, 0UL};
  EXPECT_EQ(mm.LaunchKernelBuiltinAicpuSo("RunAicpuPreprocessLoadSoLaunch", resource_id), SUCCESS);
  EXPECT_EQ(mm.builtin_aicpu_so_[resource_id].size(), 2UL);
  EXPECT_EQ(mm.builtin_aicpu_so_[resource_id]["so"].launch_count, 2UL);
  EXPECT_EQ(mm.builtin_aicpu_so_[resource_id]["so1"].launch_count, 1UL);

  EXPECT_EQ(mm.LaunchKernelBuiltinAicpuSo("RunAicpuPreprocessUnloadSoLaunch", resource_id), SUCCESS);
  EXPECT_EQ(mm.builtin_aicpu_so_[resource_id]["so"].launch_count, 1UL);
  EXPECT_EQ(mm.builtin_aicpu_so_[resource_id]["so1"].launch_count, 0UL);
  EXPECT_EQ(mm.LaunchKernelBuiltinAicpuSo("RunAicpuPreprocessUnloadSoLaunch", resource_id), SUCCESS);
  EXPECT_EQ(mm.builtin_aicpu_so_[resource_id]["so"].launch_count, 0UL);
  EXPECT_EQ(mm.builtin_aicpu_so_[resource_id]["so1"].launch_count, 0UL);
}

shared_ptr<ModelListener> listerner(new DModelListener());
TEST_F(UtestModelManagerModelManager, test_load_model_online) {
  ModelManager mm;
  uint32_t model_id = 1;
  uint32_t device_id = 0;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  GraphNodePtr graph_node = MakeShared<GraphNode>(0);

  ge::ProfilingProperties::Instance().SetSubscribeInfo(0, model_id, true);
  EXPECT_EQ(mm.LoadModelOnline(model_id, ge_root_model, graph_node, device_id), PARAM_INVALID);  // GeModel is null

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  const auto graph_node2 = MakeShared<GraphNode>(graph->GetGraphID());
  ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
  ge_root_model->SetIsSpecificStream(true); // For not start DavinciModel thread.
  graph_node2->SetGeRootModel(ge_root_model);
  graph_node2->SetLoadFlag(true);
  graph_node2->SetAsync(true);
  graph_node2->GetFrozenInputIndex().insert(0);
  uint64_t mem = 0UL;
  ge_root_model->MutableFixedFeatureMemory().insert({RT_MEMORY_HBM, {RT_MEMORY_HBM, &mem, sizeof(mem), true, false, false, 0U, nullptr}});
  EXPECT_EQ(mm.LoadModelOnline(model_id, ge_root_model, graph_node2, device_id), SUCCESS);
  ge::ProfilingProperties::Instance().CleanSubscribeInfo();
}

TEST_F(UtestModelManagerModelManager, command_profiling) {
  ModelManager manager;
  uint32_t model_id = 1;
  Command cmd;
  auto model = std::make_shared<DavinciModel>(1, listerner);
  model->SetId(model_id);
  cmd.cmd_params.push_back("modelId");
  cmd.cmd_params.push_back(to_string(model_id));

  ge::ProfilingProperties::Instance().SetSubscribeInfo(0, model_id, true);
  EXPECT_EQ(manager.HandleProfModelUnsubscribeCommand(cmd), FAILED);
  ge::ProfilingProperties::Instance().CleanSubscribeInfo();
}

TEST_F(UtestModelManagerModelManager, command_profiling_get_hybrid_model) {
  uint32_t model_id = 999;
  Command cmd;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  auto hybrid_model_ptr = ge::hybrid::HybridDavinciModel::Create(ge_root_model);
  auto shared_model = std::shared_ptr<hybrid::HybridDavinciModel>(hybrid_model_ptr.release());
  shared_model->SetDeviceId(0);
  cmd.cmd_params.push_back("modelId");
  cmd.cmd_params.push_back(to_string(model_id));
  EXPECT_EQ(ModelManager::GetInstance().HandleProfModelSubscribeCommand(cmd), FAILED);
  ModelManager::GetInstance().InsertModel(model_id, shared_model);
  ProfilingManager::Instance().SetGraphIdToModelMap(1, model_id);
  EXPECT_EQ(ModelManager::GetInstance().HandleProfModelSubscribeCommand(cmd), SUCCESS);
  EXPECT_EQ(ModelManager::GetInstance().HandleProfModelUnsubscribeCommand(cmd), SUCCESS);
  EXPECT_EQ(ModelManager::GetInstance().Unload(model_id), SUCCESS);
  EXPECT_TRUE(ProfilingManager::Instance().model_id_map_.find(1) == ProfilingManager::Instance().model_id_map_.end());
}

TEST_F(UtestModelManagerModelManager, test_get_aipp_info_and_type) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  auto hybrid_model_ptr = ge::hybrid::HybridDavinciModel::Create(ge_root_model);
  auto shared_model = std::shared_ptr<hybrid::HybridDavinciModel>(hybrid_model_ptr.release());
  ModelManager model_manager;
  const uint32_t model_id = 1;
  model_manager.hybrid_model_map_[model_id] = shared_model;

  uint32_t index = 0;
  AippConfigInfo aipp_info;
  auto ret = model_manager.GetAippInfo(model_id, index, aipp_info);
  EXPECT_EQ(ret, ACL_ERROR_GE_AIPP_NOT_EXIST);

  InputAippType aipp_type;
  size_t aipp_index;
  ret = model_manager.GetAippType(model_id, index, aipp_type, aipp_index);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestModelManagerModelManager, test_execute_model) {
  auto davinci_model = std::make_shared<DavinciModel>(0, nullptr);
  davinci_model->Init();
  ModelManager mm;
  const uint32_t model_id = 1;
  bool async_mode = true;
  rtStream_t stream = nullptr;
  vector<GeTensor> input_tensor;
  vector<GeTensor> output_tensor;
  davinci_model->session_id_ = 1;
  davinci_model->model_id_ = 1;
  davinci_model->sub_model_id_ = 1;

  mm.model_map_[model_id] = davinci_model;
  EXPECT_TRUE(mm.GetModel(model_id) != nullptr);

  mm.ExecuteModel(model_id, stream, async_mode, input_tensor, output_tensor);
}

TEST_F(UtestModelManagerModelManager, test_execute_model1) {
  auto hybrid_model = std::make_shared<hybrid::HybridDavinciModel>();
  ModelManager mm;
  const uint32_t model_id = 1;
  bool async_mode = true;
  rtStream_t stream = nullptr;
  GeTensorDesc td(GeShape(), FORMAT_ND, DT_FLOAT);
  GeTensor geTensor(td);
  vector<GeTensor> input_tensor;
  input_tensor.emplace_back(geTensor);
  vector<GeTensor> output_tensor;

  mm.hybrid_model_map_[model_id] = hybrid_model;
  EXPECT_TRUE(mm.GetHybridModel(model_id) != nullptr);

  mm.ExecuteModel(model_id, stream, async_mode, input_tensor, output_tensor);
}

TEST_F(UtestModelManagerModelManager, test_execute_model1_davincimodel) {
  auto shared_model = MakeShared<DavinciModel>(0, nullptr);
  uint32_t davinci_model_id = 0U;
  shared_model->SetDeviceId(0);
  ModelManager::GetInstance().InsertModel(davinci_model_id, shared_model);
  bool async_mode = true;
  rtStream_t stream = nullptr;
  GeTensorDesc td(GeShape(), FORMAT_ND, DT_FLOAT);
  GeTensor geTensor(td);
  std::vector<gert::Tensor> gert_input_tensor;
  std::vector<gert::Tensor> gert_output_tensor;
  EXPECT_TRUE(ModelManager::GetInstance().GetModel(davinci_model_id) != nullptr);
  
  shared_model->args_manager_.AllocKernelLaunchArgsHostMem(shared_model->GetLogicalMemAllocation().size());
  auto ret = ModelManager::GetInstance().ExecuteModelWithStream(davinci_model_id, stream, async_mode, gert_input_tensor, gert_output_tensor);
  EXPECT_EQ(ret, SUCCESS);
  ModelManager::GetInstance().model_map_.clear();
}

TEST_F(UtestModelManagerModelManager, TestLoadModelWithQueueFromDataFailed) {
  uint32_t model_id = 1;
  ModelManager model_manager;
  ModelData data;
  LoadStandardModelData(data);
  std::unique_ptr<uint8_t[]> auto_delete(reinterpret_cast<uint8_t *>(data.model_data));

  // failed: input & output queue are all empty
  ASSERT_EQ(model_manager.LoadModelWithQ(model_id, data, {}), ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID);
  ASSERT_EQ(model_manager.model_map_.count(model_id), 0);
}

TEST_F(UtestModelManagerModelManager, TestLoadModelWithInputOutputFailed) {
  uint32_t model_id = 1;
  ModelManager model_manager;
  ModelData data;
  LoadStandardModelData(data);
  std::unique_ptr<uint8_t[]> auto_delete(reinterpret_cast<uint8_t *>(data.model_data));

  // failed: input & output queue are all empty
  std::vector<uint32_t> input_queue_ids(1);
  input_queue_ids[0] = 0;
  std::vector<uint32_t> output_queue_ids(1);
  input_queue_ids[0] = 0;
  ASSERT_EQ(model_manager.LoadModelWithQ(model_id, data, {input_queue_ids, output_queue_ids, {}}),
    INTERNAL_ERROR);
}

TEST_F(UtestModelManagerModelManager, TestLoadModelWithQueueParamFromGeRootModelFailed) {
  auto root_graph = std::make_shared<ComputeGraph>("root-graph");
  auto root_model = std::make_shared<GeRootModel>();
  root_model->Initialize(root_graph);
  uint32_t model_id = 1;
  ModelManager model_manager;
  ModelQueueParam model_queue_param{};
  model_queue_param.input_queues = {};
  model_queue_param.output_queues = {};
  model_queue_param.input_queues_attrs = {};
  model_queue_param.output_queues_attrs = {};
  model_queue_param.group_policy = 0;
  model_queue_param.is_dynamic_sched = true;
  model_queue_param.need_report_status = true;
  // Failed to get GeModel
  ASSERT_EQ(model_manager.LoadModelWithQueueParam(model_id, root_model, model_queue_param, 0, false), INTERNAL_ERROR);

  // GeModel is null
  root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), nullptr);
  ASSERT_EQ(model_manager.LoadModelWithQueueParam(model_id, root_model, model_queue_param, 0, false), PARAM_INVALID);  // GeModel is null

  root_model->subgraph_instance_name_to_model_[root_graph->GetName()] = std::make_shared<GeModel>();
  // incorrect queue size
  ASSERT_EQ(model_manager.LoadModelWithQueueParam(model_id, root_model, model_queue_param, 0, false), ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID);
  ASSERT_EQ(model_manager.model_map_.count(model_id), 0);

  // init davinci model failed, model's graph is nullptr
  std::vector<QueueAttrs> output_queues_attrs(1);
  output_queues_attrs[0].queue_id = 1;
  model_queue_param.output_queues_attrs = output_queues_attrs;
  ASSERT_EQ(model_manager.LoadModelWithQueueParam(model_id, root_model, model_queue_param, 0, false), INTERNAL_ERROR);
}

TEST_F(UtestModelManagerModelManager, TestLoadModelWithQueueParamNeedUpdateSessionId) {
  auto root_graph = std::make_shared<ComputeGraph>("root-graph");
  auto root_model = std::make_shared<GeRootModel>();
  root_model->Initialize(root_graph);
  uint32_t model_id = 1;
  ModelManager model_manager;
  ModelQueueParam model_queue_param{};
  model_queue_param.input_queues = {};
  model_queue_param.output_queues = {};
  model_queue_param.input_queues_attrs = {};
  model_queue_param.output_queues_attrs = {};
  model_queue_param.group_policy = 0;
  model_queue_param.is_dynamic_sched = true;
  model_queue_param.need_report_status = true;
  root_model->subgraph_instance_name_to_model_[root_graph->GetName()] = std::make_shared<GeModel>();
  // test with need_update_session_id true
  ASSERT_EQ(model_manager.LoadModelWithQueueParam(model_id, root_model, model_queue_param, 0, true), ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID);
  ASSERT_EQ(model_manager.model_map_.count(model_id), 0);
}

TEST_F(UtestModelManagerModelManager, Cal_follow_stream_sum) {
  std::multimap<int64_t, uint64_t> hccl_stream_map = {{1, 10}, {1, 20}, {2, 10}, {2, 5}};
  uint64_t result = ModelUtils::CalFollowStreamSum(hccl_stream_map);
  EXPECT_EQ(result, 30);
}

TEST_F(UtestModelManagerModelManager, record_ts_snapshot_success) {
  const std::string kTriggerFile = "exec_record_trigger";
  const char_t * const kEnvRecordPath = "NPU_COLLECT_PATH";
  const std::string kRecordFilePrefix = "exec_record_";

  // 设置环境变量
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  mmSetEnv(kEnvRecordPath, &npu_collect_path[0U], 1);

  // 创建trigger文件
  const std::string trigger_file = std::string(&npu_collect_path[0U]) + "/" + kTriggerFile;
  auto trigger_fd = mmOpen(trigger_file.c_str(), M_WRONLY | M_CREAT);
  mmClose(trigger_fd);

  ModelManager *mm = new ModelManager();
  mmSleep(1000U);

  std::string record_file = std::string(&npu_collect_path[0U]) + "/" + kRecordFilePrefix + std::to_string(mmGetPid());
  const auto record_fd = mmOpen(record_file.c_str(), M_RDONLY);
  EXPECT_TRUE(record_fd >= 0);
  mmClose(record_fd);

  delete mm;
  trigger_fd = mmOpen(trigger_file.c_str(), M_RDONLY);
  EXPECT_TRUE(trigger_fd < 0);

  // 清理环境变量
  mmSetEnv(kEnvRecordPath, "", 1);
  // 清理record file
  mmUnlink(record_file.c_str());
}

TEST_F(UtestModelManagerModelManager, record_ts_snapshot_fail) {
  const std::string kTriggerFile = "exec_record_trigger";
  const char_t * const kEnvRecordPath = "NPU_COLLECT_PATH";
  const std::string kRecordFilePrefix = "exec_record_";

  // 设置环境变量
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvRecordPath, fail_collect_path.c_str(), 1);

  // 创建trigger文件
  const std::string trigger_file = fail_collect_path + "/" + kTriggerFile;
  auto trigger_fd = mmOpen(trigger_file.c_str(), M_WRONLY | M_CREAT);
  mmClose(trigger_fd);

  ModelManager *mm = new ModelManager();
  mmSleep(1000U);

  std::string record_file = fail_collect_path + "/" + kRecordFilePrefix + std::to_string(mmGetPid());
  const auto record_fd = mmOpen(record_file.c_str(), M_RDONLY);
  EXPECT_TRUE(record_fd < 0);
  mmClose(record_fd);
  delete mm;
  trigger_fd = mmOpen(trigger_file.c_str(), M_RDONLY);
  EXPECT_TRUE(trigger_fd < 0);

  // 清理环境变量
  mmSetEnv(kEnvRecordPath, "", 1);
}

TEST_F(UtestModelManagerModelManager, DestroyAicpuSessionForInfer) {
  ModelManager mm;
  uint32_t model_id = 0;
  auto ret = mm.DestroyAicpuSessionForInfer(model_id);
  EXPECT_EQ(ret, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
}

TEST_F(UtestModelManagerModelManager, DestroyAicpuSessionForInferSuccess) {
  ModelManager mm;
  uint32_t model_id = 0;
  GetContext().SetCtxDeviceId(0);
  const auto davinci_model = MakeShared<DavinciModel>(0, nullptr);
  EXPECT_NE(davinci_model, nullptr);
  davinci_model->SetId(model_id);
  davinci_model->SetDeviceId(0);
  davinci_model->session_id_ = 0;
  mm.model_map_[model_id] = davinci_model;
  mm.sess_id_to_device_ids_[0] = {0};
  auto ret = mm.DestroyAicpuSessionForInfer(model_id);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestModelManagerModelManager, DestroyAicpuSessionMultiDeviceSuccess) {
  ModelManager mm;
  GetContext().SetCtxDeviceId(0);
  mm.sess_id_to_device_ids_[0] = {0};
  g_runtime_stub_mock = "rtCtxGetCurrent";
  mm.DestroyAicpuSession(0, false);
  EXPECT_EQ(mm.sess_id_to_device_ids_.size(), 0);
  g_runtime_stub_mock = "";
}

TEST_F(UtestModelManagerModelManager, DestroyAicpuSessionMultiDeviceFailed) {
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());
  ModelManager mm;
  GetContext().SetCtxDeviceId(0);
  mm.sess_id_to_device_ids_[0] = {0};
  mm.DestroyAicpuSession(0, false);
  EXPECT_EQ(mm.sess_id_to_device_ids_.size(), 1);
  RuntimeStub::Reset();
}

TEST_F(UtestModelManagerModelManager, DestroyAicpuSessionWithSocketClosed) {
  ModelManager mm;
  GetContext().SetCtxDeviceId(0);
  mm.sess_id_to_device_ids_[0] = {0};
  g_runtime_stub_mock = "rtCtxGetCurrent";
  mm.SetSocketCloseStatus(true);
  mm.DestroyAicpuSession(0, false);
  EXPECT_NE(mm.sess_id_to_device_ids_.size(), 0);
  g_runtime_stub_mock = "";
}

TEST_F(UtestModelManagerModelManager, DestroyAicpuSessionWithSingleDevice) {
  ModelManager mm;
  GetContext().SetCtxDeviceId(0);
  mm.sess_id_to_device_ids_[0] = {0};
  g_runtime_stub_mock = "rtCtxGetCurrent";
  mm.DestroyAicpuSession(0, true, 1);
  EXPECT_NE(mm.sess_id_to_device_ids_.size(), 0);
  g_runtime_stub_mock = "";
}

TEST_F(UtestModelManagerModelManager, SetDynamicSize) {
  ModelManager mm;
  uint32_t model_id = 0;
  std::vector<uint64_t> batch_num;
  int32_t dynamic_type = static_cast<int32_t>(FIXED);
  auto ret = mm.SetDynamicSize(model_id, batch_num, dynamic_type);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestModelManagerModelManager, DoLoadHybridModelOnline) {
  ModelManager mm;
  uint32_t model_id = 0;
  ModelData model_data;
  model_data.om_name = "om_name";
  model_data.om_path = "";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetRootGraph(graph);
  hybrid::HybridModel model(ge_root_model);
  model.root_graph_ = graph;

  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtStreamCreateWithFlags).WillRepeatedly(testing::Invoke(MockrtStreamCreateWithFlags));

  rtStream_t stream = (void *)0x1;
  auto ret = mm.DoLoadHybridModelOnline(model_id, model_data, 0U, ge_root_model, listerner, stream);
  EXPECT_EQ(ret, ge::PARAM_INVALID);

  // use owner stream
  EXPECT_EQ(g_call_stream_create_times, 0);
  RuntimeStub::Reset();
}

TEST_F(UtestModelManagerModelManager, HandleCommand_Invalid) {
  ModelManager mm;
  uint32_t model_id = 0;
  Command cmd;
  cmd.cmd_params.push_back("modelId");
  cmd.cmd_params.push_back(to_string(model_id));

  auto ret = mm.HandleProfModelSubscribeCommand(cmd);
  EXPECT_EQ(ret, FAILED);

  ret = mm.HandleProfFinalizeCommand(cmd);
  EXPECT_EQ(ret, SUCCESS);

  ret = mm.HandleProfStartCommand(cmd);
  EXPECT_EQ(ret, SUCCESS);

  ret = mm.HandleProfStopCommand(cmd);
  EXPECT_EQ(ret, SUCCESS);

  ret = mm.HandleDumpCommand(cmd);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestModelManagerModelManager, GetSomething_Invalid) {
  ModelManager mm;
  uint32_t model_id = 0;

  std::vector<InputOutputDescInfo> input_desc;
  std::vector<InputOutputDescInfo> output_desc;
  std::vector<uint32_t> inputFormats;
  std::vector<uint32_t> outputFormats;
  bool new_model_desc = false;
  auto ret = mm.GetInputOutputDescInfo(model_id, input_desc, output_desc, inputFormats, outputFormats, new_model_desc);
  EXPECT_EQ(ret, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);

  std::vector<std::vector<int64_t>> batch_info;
  int32_t dynamic_type = static_cast<int32_t>(FIXED);
  ret = mm.GetDynamicBatchInfo(model_id, batch_info, dynamic_type);
  EXPECT_EQ(ret, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);

  ret = mm.GetCombinedDynamicDims(model_id, batch_info);
  EXPECT_EQ(ret, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);

  std::vector<std::string> user_input_shape_order;
  ret = mm.GetUserDesignateShapeOrder(model_id, user_input_shape_order);
  EXPECT_EQ(ret, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);

  std::vector<int64_t> batch_info2;
  ret = mm.GetCurrentShape(model_id, batch_info2, dynamic_type);
  EXPECT_EQ(ret, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);

  std::vector<std::string> dynamic_output_shape_info;
  ret = mm.GetOutputShapeInfo(model_id, dynamic_output_shape_info);
  EXPECT_EQ(ret, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
}

TEST_F(UtestModelManagerModelManager, test_execute_model2) {
  auto hybrid_model = std::make_shared<hybrid::HybridDavinciModel>();
  ModelManager mm;
  const uint32_t model_id = 1;
  rtStream_t stream = nullptr;
  bool async_mode = true;
  InputData input_data;
  OutputData output_data;
  std::vector<GeTensorDesc> input_desc;
  std::vector<GeTensorDesc> output_desc;

  mm.hybrid_model_map_[model_id] = hybrid_model;
  EXPECT_TRUE(mm.GetHybridModel(model_id) != nullptr);
  const std::vector<GeTensor> input_tensor = {};
  const std::vector<GeTensor> output_tensor = {};

  mm.ExecuteModel(model_id, stream, async_mode, input_data, input_desc, output_data,
                  output_desc, input_tensor, output_tensor);

  std::vector<gert::Tensor> gert_input_tensor;
  std::vector<gert::Tensor> gert_output_tensor;
  gert_input_tensor.resize(1);
  gert_output_tensor.resize(1);
  GraphNodePtr graph_node = MakeShared<GraphNode>(0);
  mm.ExecuteModelWithStreamAsync(model_id, graph_node, gert_input_tensor,
    gert_output_tensor, stream);
}

TEST_F(UtestModelManagerModelManager, test_ExternalAllocatorMalloc) {
  auto external_allocator = MakeShared<ExternalAllocatorUtStub>();
  auto hybrid_model = std::make_shared<hybrid::HybridDavinciModel>();
  GraphId graph_id = 1;
  ModelManager mm;
  const uint32_t model_id = 1;
  rtStream_t stream = (void*)0x111;
  ExternalAllocatorManager::SetExternalAllocator(stream, external_allocator);
  EXPECT_EQ(ExternalAllocatorManager::GetExternalAllocator(stream), external_allocator);

  GraphNodePtr graph_node = MakeShared<GraphNode>(0);
  graph_node->SetFeatureBaseRefreshable(true);
  graph_node->SetAppRefreshConstMemoryFlag();
  EXPECT_EQ(mm.ExternalAllocatorMalloc(graph_id, model_id, graph_node, stream), SUCCESS);
  /* No memory allocated for refreshable node. */
  EXPECT_EQ(graph_node->GetFeatureMemBlock(), nullptr);

  GraphNodePtr graph_node2 = MakeShared<GraphNode>(0);
  graph_node2->SetFeatureBaseRefreshable(false);
  graph_node2->SetAppRefreshConstMemoryFlag();
  EXPECT_EQ(mm.ExternalAllocatorMalloc(graph_id, model_id, graph_node2, stream), SUCCESS);
  EXPECT_EQ(graph_node2->GetFeatureMemBlock(), external_allocator->GetBlockAddr());
  FreeFeatureMemory(graph_node2);
  EXPECT_EQ(external_allocator->GetBlockAddr(), nullptr);
}

TEST_F(UtestModelManagerModelManager, LoadClearAicpuSo_Invalid) {
  ModelManager mm;
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});

  std::string so_name = "abc.so";
  bool loaded = false;

  auto ret = mm.LoadCustAicpuSo(op_desc, so_name, loaded);
  EXPECT_EQ(ret, SUCCESS);

  ret = mm.ClearAicpuSo();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestModelManagerModelManager, LoadCustAicpuSoAndUpdateSoName_Success) {
  std::string uniq_so_name = "1_vendors_test_libcust_opmaster.so";
  std::string so_name;
  so_name.append("/opp/vendors/test/").append(kOpMasterDeviceLib).append("libcust_opmaster.so");
  std::unique_ptr<char[]> so_bin = std::unique_ptr<char[]>(new (std::nothrow) char[so_name.length()]);
  (void) memcpy_s(so_bin.get(), so_name.length(), so_name.data(), so_name.length());
  OpSoBinPtr op_so_bin = std::make_shared<OpSoBin>(so_name, "", std::move(so_bin), so_name.length());
  ModelManager mm;
  mm.cust_op_master_so_names_to_unique_name_[uniq_so_name] = uniq_so_name;
  mm.cust_op_master_so_names_to_bin_[uniq_so_name] = op_so_bin;
  mm.cust_aicpu_so_.clear();
  EXPECT_EQ(mm.LoadCustAicpuSoAndUpdateSoName(1, so_name), SUCCESS);
  EXPECT_EQ(mm.cust_aicpu_so_.size(), 1UL);
  EXPECT_EQ(so_name, uniq_so_name);
}

TEST_F(UtestModelManagerModelManager, LoadBuiltinAicpuSoAndUpdateSoName_Success) {
  std::string uniq_so_name = "libopmaster.so";
  std::string so_name;
  so_name.append("/opp/built-in/").append(kOpMasterDeviceLib).append(uniq_so_name);
  std::unique_ptr<char[]> so_bin = std::unique_ptr<char[]>(new (std::nothrow) char[so_name.length()]);
  (void) memcpy_s(so_bin.get(), so_name.length(), so_name.data(), so_name.length());
  OpSoBinPtr op_so_bin = std::make_shared<OpSoBin>(so_name, "", std::move(so_bin), so_name.length());
  ModelManager mm;
  mm.built_in_op_master_so_names_to_bin_[uniq_so_name] = op_so_bin;
  mm.builtin_aicpu_so_.clear();
  EXPECT_EQ(mm.LoadBuiltinAicpuSoAndUpdateSoName(1, so_name), SUCCESS);
  EXPECT_EQ(mm.builtin_aicpu_so_.size(), 1UL);
  EXPECT_EQ(so_name, uniq_so_name);
}

TEST_F(UtestModelManagerModelManager, testGetModelMemAndWeightSize) {
  ModelManager mm;
  ModelData data;
  LoadStandardModelData(data, 1024U, 1U);
  size_t mem_size;
  size_t weight_size;

  auto ret = mm.GetModelMemAndWeightSize(data, mem_size, weight_size);
  delete [] (uint8_t *)data.model_data;
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestModelManagerModelManager, test_DynamicModel_GetModelMemAndWeightSize_Success) {
  ModelData data;
  std::vector<PartitionInfo> partition_infos = {{5U, 10U}, {5U, 5U}};
  LoadFlexibleModelData(data, partition_infos, 1024U, partition_infos.size());

  size_t mem_size = 0U;
  size_t weight_size = 0U;
  auto ret = ModelManager::GetModelMemAndWeightSize(data, mem_size, weight_size);
  delete [] (uint8_t *)data.model_data;
  EXPECT_TRUE(mem_size == 0U);
  EXPECT_EQ(weight_size, 15U);
  EXPECT_EQ(ret, SUCCESS);
}

// partition数量为0，om加载失败
TEST_F(UtestModelManagerModelManager, test_DynamicModel_GetModelMemAndWeightSize_FailedWithInvalidPartitionNum1) {
  ModelData data;
  std::vector<PartitionInfo> partition_infos = {{0U, 10U}};
  LoadFlexibleModelData(data, partition_infos, 1024U, partition_infos.size());

  size_t mem_size = 0U;
  size_t weight_size = 0U;
  auto ret = ModelManager::GetModelMemAndWeightSize(data, mem_size, weight_size);
  delete [] (uint8_t *)data.model_data;
  EXPECT_EQ(ret, ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);
}
// partition数量为1，为仿真模式的模型，从加载的om_laod_helper中读取失败
TEST_F(UtestModelManagerModelManager, test_SimulatedStaticModel_Failed) {
  ModelData data;
  std::vector<PartitionInfo> partition_infos = {{1U, 5U}};
  LoadFlexibleModelData(data, partition_infos, 1024U, partition_infos.size());

  size_t mem_size = 0U;
  size_t weight_size = 0U;
  auto ret = ModelManager::GetModelMemAndWeightSize(data, mem_size, weight_size);
  delete [] (uint8_t *)data.model_data;
  EXPECT_EQ(ret, ACL_ERROR_GE_PARAM_INVALID);
}

TEST_F(UtestModelManagerModelManager, testGetOrigInputInfo) {
  ModelManager mm;
  const uint32_t model_id = 1;
  uint32_t index = 0;
  OriginInputInfo orig_input_info;

  auto ret = mm.GetOrigInputInfo(model_id, index, orig_input_info);
  EXPECT_EQ(ret, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
}

TEST_F(UtestModelManagerModelManager, testGetOpDescInfo) {
  ModelManager mm;
  uint32_t device_id = 1;
  uint32_t stream_id = 1;
  uint32_t task_id = 1;
  OpDescInfo op_desc_info;

  auto ret = mm.GetOpDescInfo(device_id, stream_id, task_id, op_desc_info);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestModelManagerModelManager, KernelLaunchEx_Invalid) {
  ModelManager mm;
  aicpu::FWKAdapter::FWKOperateType op_type = aicpu::FWKAdapter::FWKOperateType::FWK_ADPT_KERNEL_DESTROY;
  uint64_t session_id = 0;
  uint32_t model_id = 0;
  uint32_t sub_model_id = 0;
  std::vector<uint64_t> aicpu_kernel = {1, 2, 3};
  mm.model_aicpu_kernel_.insert({string("0_0_0"), aicpu_kernel});

  auto ret = mm.KernelLaunchEx(op_type, session_id, model_id, sub_model_id);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestModelManagerModelManager, DestroyAicpuKernel_Invalid) {
  ModelManager mm;
  uint64_t session_id = 0;
  uint32_t model_id = 0;
  uint32_t sub_model_id = 0;
  std::vector<uint64_t> aicpu_kernel = {1, 2, 3};
  mm.model_aicpu_kernel_.insert({string("0_0_0"), aicpu_kernel});

  auto ret = mm.DestroyAicpuKernel(session_id, model_id, sub_model_id);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestModelManagerModelManager, GetModelByCmd_Invalid) {
  uint32_t model_id = std::numeric_limits<uint32_t>::max();
  Command command;
  EXPECT_EQ(ModelManager::GetModelIdByCmd(command, model_id), PARAM_INVALID);
  EXPECT_TRUE(model_id == std::numeric_limits<uint32_t>::max());

  command.cmd_params.push_back(string("not model Id"));
  EXPECT_EQ(ModelManager::GetModelIdByCmd(command, model_id), PARAM_INVALID);
  EXPECT_TRUE(model_id == std::numeric_limits<uint32_t>::max());
}

TEST_F(UtestModelManagerModelManager, testParserPara) {
  ModelManager mm;
  Command command;
  command.cmd_params.push_back(ge::DUMP_STATUS);
  command.cmd_params.push_back(string("value"));

  auto ret = mm.HandleDumpCommand(command);
  EXPECT_EQ(ret, SUCCESS);
 }

TEST_F(UtestModelManagerModelManager, HandleProfStartCommand_Invalid) {
  ModelManager mm;
  ModelData data;
  LoadStandardModelData(data);
  uint32_t model_id = std::numeric_limits<uint32_t>::max();
  const ModelParam param;
  mm.LoadModelOffline(data, param, model_id);

  Command cmd;
  unsetenv("GE_PROFILING_TO_STD_OUT");

  auto ret = mm.HandleProfStartCommand(cmd);
  EXPECT_EQ(ret, PARAM_INVALID);
  ret = mm.HandleProfStopCommand(cmd);
  EXPECT_EQ(ret, PARAM_INVALID);

  cmd.cmd_params.push_back(string("modelId"));
  cmd.cmd_params.push_back(to_string(model_id));
  cmd.module_index = 0;
  ret = mm.HandleProfStartCommand(cmd);
  EXPECT_EQ(ret, FAILED);
  ret = mm.HandleProfStopCommand(cmd);
  EXPECT_EQ(ret, FAILED);

  for(uint32_t i = 0; i < 1001; i++) {
    cmd.cmd_params.push_back(string("m"));
    cmd.cmd_params.push_back(to_string(model_id));
  }
  ret = mm.HandleProfStartCommand(cmd);
  EXPECT_EQ(ret, PARAM_INVALID);
  ret = mm.HandleProfStopCommand(cmd);
  EXPECT_EQ(ret, PARAM_INVALID);

  EXPECT_EQ(mm.DeleteModel(model_id), SUCCESS);
  setenv("GE_PROFILING_TO_STD_OUT", "1", 1);
  delete[] (uint8_t *)data.model_data;
}

TEST_F(UtestModelManagerModelManager, TestLoadModelWithoutQ) {
  auto root_graph = std::make_shared<ComputeGraph>("root-graph");
  auto root_model = std::make_shared<GeRootModel>();
  root_model->Initialize(root_graph);
  uint32_t model_id = 1;
  ModelManager model_manager;
  // Failed to get GeModel
  AttrUtils::SetBool(root_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
  ASSERT_EQ(model_manager.LoadModelWithoutQ(model_id, root_model, 0), UNSUPPORTED);
  AttrUtils::SetBool(root_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, false);


  ASSERT_EQ(model_manager.LoadModelWithoutQ(model_id, root_model, 0), INTERNAL_ERROR);

  // GeModel is null
  root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), nullptr);
  ASSERT_EQ(model_manager.LoadModelWithoutQ(model_id, root_model, 0), PARAM_INVALID);  // GeModel is null
  root_model->subgraph_instance_name_to_model_[root_graph->GetName()] = std::make_shared<GeModel>();

  // init davinci model failed, model's graph is nullptr
  ASSERT_EQ(model_manager.LoadModelWithoutQ(model_id, root_model, 0), INTERNAL_ERROR);
}

TEST_F(UtestModelManagerModelManager, register_for_dump) {
  DumpConfig dump_config;
  dump_config.dump_path = "/test";
  dump_config.dump_mode = "all";
  dump_config.dump_status = "on";
  dump_config.dump_op_switch = "on";
  auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(ModelManager::GetInstance().SetCallBackFuncForDumpManager(), SUCCESS);
  EXPECT_EQ(ModelManager::GetInstance().SetCallBackFuncForDumpManager(), SUCCESS);
}

TEST_F(UtestModelManagerModelManager, load_and_unload_task_for_davinci_model) {
  auto shared_model = MakeShared<DavinciModel>(0, nullptr);
  uint32_t davinci_model_id = 0U;
  shared_model->SetDeviceId(0);
  ModelManager::GetInstance().InsertModel(davinci_model_id, shared_model);
  const auto dump_properties = DumpManager::GetInstance().GetDumpProperties(0);
  EXPECT_EQ(ModelManager::GetInstance().LoadTaskForDavinciModel(dump_properties), SUCCESS);
  RTS_STUB_RETURN_VALUE(rtGetDevice, rtError_t, 0x78000001);
  EXPECT_EQ(ModelManager::GetInstance().UnloadTaskForDavinciModel(dump_properties), SUCCESS);
  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(davinci_model_id), SUCCESS);
  DumpManager::GetInstance().RemoveDumpProperties(0);
}

TEST_F(UtestModelManagerModelManager, over_flow_dump_flag_check) {
  ModelData data;
  LoadStandardModelData(data);
  ModelFileHeader &header = *static_cast<ModelFileHeader *>(data.model_data);
  EXPECT_EQ(header.model_num, 1U);

  DumpProperties dump_properties;
  dump_properties.InitInferOpDebug();
  DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
  ModelManager mm;
  uint32_t model_id = 0;
  const ModelParam param;
  EXPECT_EQ(mm.LoadModelOffline(data, param, model_id), SUCCESS);
  EXPECT_TRUE(mm.GetModel(model_id)->GetDumpProperties().IsOpDebugOpen());

  delete [] (uint8_t *)data.model_data;
}
ge::ComputeGraphPtr CreateGraphWithConstOutput() {
  ge::ut::GraphBuilder builder("graph");
  auto data = builder.AddNode("data1", "Data", 1, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, netoutput, 0);
  data->GetOpDesc()->SetOutputOffset({1});
  netoutput->GetOpDesc()->SetInputOffset({1});
  return builder.GetGraph();
}

void CreateSummaryCompiledModel(GraphNodePtr &graph_node, GeModelPtr &ge_model, bool has_p2p = true) {
  auto compute_graph = CreateGraphWithConstOutput();
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);
  ge_root_model->SetModelId(1U);

  AttrUtils::SetStr(compute_graph, "_split_logic_stream_2_origin_logic_stream", "");

  GraphId graph_id = 1;
  graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetComputeGraph(compute_graph);

  AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, 512);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 1024);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 2);
  if (has_p2p) {
    AttrUtils::SetInt(ge_model, ATTR_MODEL_P2P_MEMORY_SIZE, 1024);
  }

  uint64_t mem = 0UL;
  std::vector<std::vector<int64_t>> sub_mem_infos;
  std::vector<int64_t> sub_mem_offset;
  sub_mem_offset.emplace_back(0x2U);// mem_type RT_MEMORY_HBM 0x2U
  sub_mem_offset.emplace_back((int64_t)(&mem));// mem_offset_base
  sub_mem_offset.emplace_back(sizeof(mem)); // mem_size
  sub_mem_offset.emplace_back(1UL); // is_fixed_addr_prior
  sub_mem_infos.emplace_back(sub_mem_offset);
  AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_mem_infos);

  std::map<std::string, std::string> graph_options;
  graph_options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestModelManagerModelManager, MallocConstMemory_has_been_set) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  uint64_t mem = 0;
  graph_node->SetConstMemoryBase(&mem, sizeof(mem));

  EXPECT_EQ(MallocConstMemory(graph_id, graph_node, external_allocator), SUCCESS);
}

TEST_F(UtestModelManagerModelManager, MallocOutputsMemory_AppRefreshable) {
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  graph_node->SetAppRefreshFeatureMemoryFlag();
  std::vector<gert::Tensor> outputs_empty;
  EXPECT_EQ(MallocOutputsMemory(graph_id, graph_node, nullptr, outputs_empty), SUCCESS);
}

TEST_F(UtestModelManagerModelManager, MallocConstMemory_AppRefreshable) {
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  GeModelPtr ge_model;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  graph_node->SetAppRefreshConstMemoryFlag();
  EXPECT_EQ(MallocConstMemory(graph_id, graph_node, nullptr), SUCCESS);
}

TEST_F(UtestModelManagerModelManager, MallocConstMemory_success) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  graph_manager.graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  uint64_t mem = 0;
  graph_node->SetConstMemoryBase(nullptr, sizeof(mem));

  EXPECT_EQ(MallocConstMemory(graph_id, graph_node, external_allocator), SUCCESS);
  MemBlock *mem_block = dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetBlockAddr();
  InputMemoryBaseInfo mem_base = graph_node->GetConstMemoryBase();
  EXPECT_EQ(mem_block->GetAddr(), mem_base.first);
  EXPECT_EQ(mem_block->GetSize(), mem_base.second);
  graph_node->SetLoadFlag(true);
  graph_manager.RemoveGraph(graph_id);
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetBlockAddr(), nullptr);
}

TEST_F(UtestModelManagerModelManager, MallocFeatureMemory_fixed_mem_has_been_set) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  StubExecutor executor;
  graph_manager.executor_ = &executor;

  auto davinci_model = std::make_shared<DavinciModel>(0, nullptr);
  davinci_model->Init();
  const uint32_t model_id = 1;
  davinci_model->session_id_ = 1;
  davinci_model->model_id_ = 1;
  davinci_model->sub_model_id_ = 1;
  ModelManager::GetInstance().InsertModel(model_id, davinci_model);
  EXPECT_TRUE(ModelManager::GetInstance().GetModel(model_id) != nullptr);

  uint64_t mem = 0UL;
  EXPECT_EQ(graph_manager.SetFixedFeatureMemoryBase(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, &mem, sizeof(mem)), SUCCESS);

  EXPECT_EQ(MallocFeatureMemory(graph_id, model_id, graph_node, external_allocator), SUCCESS);
  dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetBlockAddr()->Free();
  ModelManager::GetInstance().model_map_.clear();
}

// TEST_F(UtestModelManagerModelManager, MallocFeatureMemory_invalid) {
//   GraphManager graph_manager;
//   GeModelPtr ge_model;
//   GraphNodePtr graph_node;
//   GraphId graph_id = 1;
//   CreateSummaryCompiledModel(graph_node, ge_model);
//   graph_manager.AddGraphNode(graph_id, graph_node);
//   graph_node->SetBuildFlag(true);
//   graph_node->SetCompiledFlag(true);
//   StubExecutor executor;
//   graph_manager.executor_ = &executor;

//   auto davinci_model = std::make_shared<DavinciModel>(0, nullptr);
//   davinci_model->Init();
//   const uint32_t model_id = 1;
//   davinci_model->session_id_ = 1;
//   davinci_model->model_id_ = 1;
//   davinci_model->sub_model_id_ = 1;
//   ModelManager::GetInstance().InsertModel(model_id, davinci_model);
//   EXPECT_TRUE(ModelManager::GetInstance().GetModel(model_id) != nullptr);

//   graph_node->SetFeatureBaseRefreshable(true);
//   EXPECT_EQ(MallocFeatureMemory(graph_id, model_id, graph_node, nullptr), SUCCESS);
//   ModelManager::GetInstance().model_map_.clear();
// }

// TEST_F(UtestModelManagerModelManager, MallocFeatureMemory_ExternalMallocFeatureMemorySize) {
//   std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
//   GraphManager graph_manager;
//   GeModelPtr ge_model;
//   GraphNodePtr graph_node;
//   GraphId graph_id = 1;
//   CreateSummaryCompiledModel(graph_node, ge_model);
//   graph_manager.AddGraphNode(graph_id, graph_node);
//   graph_node->SetBuildFlag(true);
//   graph_node->SetCompiledFlag(true);
//   StubExecutor executor;
//   graph_manager.executor_ = &executor;

//   auto davinci_model = std::make_shared<DavinciModel>(0, nullptr);
//   davinci_model->Init();
//   const uint32_t model_id = 1;
//   davinci_model->session_id_ = 1;
//   davinci_model->model_id_ = 1;
//   davinci_model->sub_model_id_ = 1;
//   ModelManager::GetInstance().InsertModel(model_id, davinci_model);
//   EXPECT_TRUE(ModelManager::GetInstance().GetModel(model_id) != nullptr);

//   EXPECT_EQ(MallocFeatureMemory(graph_id, model_id, graph_node, external_allocator), SUCCESS);
//   const auto mem_block = graph_node->GetFeatureMemBlock();
//   ASSERT_NE(mem_block, nullptr);
//   size_t feature_mem_size = graph_node->GetConstMemoryBase().second;
//   size_t fixed_feature_mem_size = 0U;
//   CompiledGraphSummaryPtr summary = nullptr;
//   graph_manager.GetCompiledGraphSummary(graph_id, summary);
//   EXPECT_EQ(summary->GetFixedFeatureMemorySize(fixed_feature_mem_size), SUCCESS);

//   // 用户没有设置fix内存，ge会兜底申请fixed内存，所以这里会将fixed长度抛出去
//   EXPECT_EQ(mem_block->GetSize(), feature_mem_size);
//   dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetBlockAddr()->Free();
//   ModelManager::GetInstance().model_map_.clear();
// }

TEST_F(UtestModelManagerModelManager, MallocFeatureMemory_has_been_set) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  StubExecutor executor;
  graph_manager.executor_ = &executor;

  auto davinci_model = std::make_shared<DavinciModel>(0, nullptr);
  davinci_model->Init();
  const uint32_t model_id = 1;
  davinci_model->session_id_ = 1;
  davinci_model->model_id_ = 1;
  davinci_model->sub_model_id_ = 1;
  ModelManager::GetInstance().InsertModel(model_id, davinci_model);
  EXPECT_TRUE(ModelManager::GetInstance().GetModel(model_id) != nullptr);

  EXPECT_EQ(MallocFeatureMemory(graph_id, model_id, graph_node, external_allocator), SUCCESS);
  dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetBlockAddr()->Free();
  ModelManager::GetInstance().model_map_.clear();
}

TEST_F(UtestModelManagerModelManager, MallocFeatureMemory_success) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  StubExecutor executor;
  graph_manager.executor_ = &executor;

  auto davinci_model = std::make_shared<DavinciModel>(0, nullptr);
  davinci_model->Init();
  const uint32_t model_id = 1;
  davinci_model->session_id_ = 1;
  davinci_model->model_id_ = 1;
  davinci_model->sub_model_id_ = 1;
  ModelManager::GetInstance().InsertModel(model_id, davinci_model);
  EXPECT_TRUE(ModelManager::GetInstance().GetModel(model_id) != nullptr);

  EXPECT_EQ(MallocFeatureMemory(graph_id, model_id, graph_node, external_allocator), SUCCESS);
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetBlockAddr(), nullptr);
  MemBlock *mem_block = dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetBlockAddr();
  InputMemoryBaseInfo mem_base = graph_node->GetFeatureMemoryBase();
  EXPECT_EQ(mem_block->GetAddr(), mem_base.first);
  EXPECT_EQ(mem_block->GetSize(), mem_base.second);
  mem_block->Free();
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetBlockAddr(), nullptr);

  EXPECT_EQ(MallocFeatureMemory(graph_id, model_id, graph_node, external_allocator), SUCCESS);
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetBlockAddr(), nullptr);
  mem_block = dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetBlockAddr();
  mem_base = graph_node->GetFeatureMemoryBase();
  EXPECT_EQ(mem_block->GetAddr(), mem_base.first);
  EXPECT_EQ(mem_block->GetSize(), mem_base.second);
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetAdviseCnt(), 1U);
  mem_block->Free();
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetBlockAddr(), nullptr);
  ModelManager::GetInstance().model_map_.clear();
}

TEST_F(UtestModelManagerModelManager, MallocFeatureMemory_refresh) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  std::map<std::string, std::string> graph_options;
  graph_options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "0");
  GetThreadLocalContext().SetGraphOption(graph_options);

  auto davinci_model = std::make_shared<DavinciModel>(0, nullptr);
  davinci_model->Init();
  const uint32_t model_id = 1;
  davinci_model->session_id_ = 1;
  davinci_model->model_id_ = 1;
  davinci_model->sub_model_id_ = 1;
  ModelManager::GetInstance().InsertModel(model_id, davinci_model);
  EXPECT_TRUE(ModelManager::GetInstance().GetModel(model_id) != nullptr);

  EXPECT_EQ(MallocFeatureMemory(graph_id, model_id, graph_node, external_allocator), SUCCESS);
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetBlockAddr(), nullptr);
  MemBlock *mem_block = dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetBlockAddr();
  InputMemoryBaseInfo mem_base = graph_node->GetFeatureMemoryBase();
  EXPECT_EQ(mem_block->GetAddr(), mem_base.first);
  EXPECT_EQ(mem_block->GetSize(), mem_base.second);

  graph_node->SetLoadFlag(true);
  EXPECT_EQ(MallocFeatureMemory(graph_id, model_id, graph_node, external_allocator), SUCCESS);
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetBlockAddr(), nullptr);
  mem_block = dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetBlockAddr();
  mem_base = graph_node->GetFeatureMemoryBase();
  EXPECT_EQ(mem_block->GetAddr(), mem_base.first);
  EXPECT_EQ(mem_block->GetSize(), mem_base.second);
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetAdviseCnt(), 0U);
  mem_block->Free();
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetBlockAddr(), nullptr);
  ModelManager::GetInstance().model_map_.clear();
}

TEST_F(UtestModelManagerModelManager, FreeFeatureMemory_success) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  StubExecutor executor;
  graph_manager.executor_ = &executor;

  auto davinci_model = std::make_shared<DavinciModel>(0, nullptr);
  davinci_model->Init();
  const uint32_t model_id = 1;
  davinci_model->session_id_ = 1;
  davinci_model->model_id_ = 1;
  davinci_model->sub_model_id_ = 1;
  ModelManager::GetInstance().InsertModel(model_id, davinci_model);
  EXPECT_TRUE(ModelManager::GetInstance().GetModel(model_id) != nullptr);

  FreeFeatureMemory(graph_node);
  EXPECT_EQ(MallocFeatureMemory(graph_id, model_id, graph_node, external_allocator), SUCCESS);
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetBlockAddr(), nullptr);
  MemBlock *mem_block = dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetBlockAddr();
  InputMemoryBaseInfo mem_base = graph_node->GetFeatureMemoryBase();
  EXPECT_EQ(mem_block->GetAddr(), mem_base.first);
  EXPECT_EQ(mem_block->GetSize(), mem_base.second);
  FreeFeatureMemory(graph_node);
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetBlockAddr(), nullptr);
  ModelManager::GetInstance().model_map_.clear();
}

TEST_F(UtestModelManagerModelManager, CalcTensorSizeByShape_overflow) {
  GraphManager graph_manager;
  std::vector<int64_t> shape = {-20};
  GeShape ge_shape(shape);
  size_t ret_tensor_size;

  EXPECT_NE(CalcTensorSizeByShape(ge_shape, ge::DT_STRING, ret_tensor_size), SUCCESS);
}

TEST_F(UtestModelManagerModelManager, MallocFeatureMemory_with_FeatureMemoryBase) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  std::map<std::string, std::string> graph_options;
  graph_options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "0");
  GetThreadLocalContext().SetGraphOption(graph_options);

  auto davinci_model = std::make_shared<DavinciModel>(0, nullptr);
  davinci_model->Init();
  const uint32_t model_id = 1;
  davinci_model->session_id_ = 1;
  davinci_model->model_id_ = 1;
  davinci_model->sub_model_id_ = 1;
  ModelManager::GetInstance().InsertModel(model_id, davinci_model);
  EXPECT_TRUE(ModelManager::GetInstance().GetModel(model_id) != nullptr);

  std::vector<uint8_t> mem(1024, 0);
  EXPECT_EQ(graph_manager.UpdateRefreshableFeatureMemoryBase(graph_id, mem.data(), 1024), SUCCESS);
  EXPECT_EQ(MallocFeatureMemory(graph_id, model_id, graph_node, external_allocator), SUCCESS);

  //  graph is load
  graph_node->SetLoadFlag(true);
  graph_node->SetFeatureBaseRefreshable(false);
  EXPECT_EQ(MallocFeatureMemory(graph_id, model_id, graph_node, external_allocator), SUCCESS);
  ModelManager::GetInstance().model_map_.clear();
}


TEST_F(UtestModelManagerModelManager, test_execute_model3) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  auto hybrid_model = std::make_shared<hybrid::HybridDavinciModel>();
  GraphNodePtr graph_node = MakeShared<GraphNode>(0);
  ModelManager mm;
  const uint32_t model_id = 1;
  rtStream_t stream = (void*)0x111;
  ExternalAllocatorManager::SetExternalAllocator(stream, external_allocator);
  InputData input_data;
  OutputData output_data;
  std::vector<GeTensorDesc> input_desc;
  std::vector<GeTensorDesc> output_desc;

  const std::vector<GeTensor> input_tensor = {};
  std::vector<GeTensor> output_tensor = {};
  auto compute_graph = CreateGraphWithConstOutput();
  graph_node->SetComputeGraph(compute_graph);
  graph_node->SetFeatureBaseRefreshable(true);
  graph_node->SetAppRefreshFeatureMemoryFlag();

  EXPECT_NE(mm.ExecuteModelWithStreamAsync(model_id, graph_node, input_tensor,
    output_tensor, stream), SUCCESS);

  std::vector<gert::Tensor> gert_input_tensor;
  std::vector<gert::Tensor> gert_output_tensor;
  gert_input_tensor.resize(1);
  gert_output_tensor.resize(1);

  EXPECT_NE(mm.ExecuteModelWithStreamAsync(model_id, graph_node, gert_input_tensor,
    gert_output_tensor, stream), SUCCESS);
}

TEST_F(UtestModelManagerModelManager, CalcTensorSizeByShape_string) {
  GraphManager graph_manager;
  std::vector<int64_t> shape = {1, 1, 224, 224};
  GeShape ge_shape(shape);
  size_t ret_tensor_size;


  EXPECT_EQ(CalcTensorSizeByShape(ge_shape, ge::DT_STRING, ret_tensor_size), SUCCESS);
}

TEST_F(UtestModelManagerModelManager, CalcTensorSizeByShape_float) {
  GraphManager graph_manager;
  std::vector<int64_t> shape = {1, 1, 224, 224};
  GeShape ge_shape(shape);
  size_t ret_tensor_size;

  EXPECT_EQ(CalcTensorSizeByShape(ge_shape, ge::DT_FLOAT, ret_tensor_size), SUCCESS);
  EXPECT_EQ(ret_tensor_size, 200736);
}

TEST_F(UtestModelManagerModelManager, CalcTensorSizeByShape_failed) {
  GraphManager graph_manager;
  std::vector<int64_t> shape = {-1};
  GeShape ge_shape(shape);
  size_t ret_tensor_size;

  EXPECT_NE(CalcTensorSizeByShape(ge_shape, ge::DT_FLOAT, ret_tensor_size), SUCCESS);
}

TEST_F(UtestModelManagerModelManager, MallocOutputsMemory_success) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  auto compute_graph = gert::ShareGraph::AicoreStaticGraph();
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  graph_node->SetComputeGraph(compute_graph);
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  std::vector<GeTensor> outputs;
  GeTensor output;
  output.ClearData();
  outputs.emplace_back(std::move(output));

  EXPECT_EQ(MallocOutputsMemory(graph_id, graph_node, external_allocator, outputs), SUCCESS);
  EXPECT_NE(outputs[0].GetData().GetSize(), 0);

  uint8_t mem = 0;
  const AlignedPtr::Deleter deleter = [](const uint8_t *const ptr){ (void)ptr; };
  outputs[0].SetData(static_cast<uint8_t *>(&mem), 10, deleter);
  EXPECT_EQ(MallocOutputsMemory(graph_id, graph_node, external_allocator, outputs), SUCCESS);
  EXPECT_EQ(outputs[0].GetData().GetSize(), 10);
}

TEST_F(UtestModelManagerModelManager, MallocOutputsMemory_outputs_empty_success) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  auto compute_graph = gert::ShareGraph::AicoreStaticGraph();
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  graph_node->SetComputeGraph(compute_graph);
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  std::vector<GeTensor> outputs;

  EXPECT_EQ(MallocOutputsMemory(graph_id, graph_node, external_allocator, outputs), SUCCESS);
  EXPECT_NE(outputs[0].GetData().GetSize(), 0);
}

TEST_F(UtestModelManagerModelManager, MallocOutputsMemory_success_with_gertTensor) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  auto compute_graph = gert::ShareGraph::AicoreStaticGraph();
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  graph_node->SetComputeGraph(compute_graph);
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  std::vector<gert::Tensor> outputs;
  outputs.resize(1);
  std::vector<uint8_t> output_data_1(96, 0xFF);
  outputs[0] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                             {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                             gert::kOnDeviceHbm,                                // placement
                             ge::DT_FLOAT,                              // data type
                             (void *) output_data_1.data()};

  EXPECT_EQ(MallocOutputsMemory(graph_id, graph_node, external_allocator, outputs), SUCCESS);
  EXPECT_EQ(outputs[0].GetSize(), 96);
  std::vector<gert::Tensor> outputs_empty;
  outputs_empty.resize(1);
  EXPECT_EQ(MallocOutputsMemory(graph_id, graph_node, external_allocator, outputs_empty), SUCCESS);
  EXPECT_EQ(outputs[0].GetSize(), 96);
  graph_node->SetAppRefreshFeatureMemoryFlag();
  EXPECT_EQ(MallocOutputsMemory(graph_id, graph_node, external_allocator, outputs_empty), SUCCESS);
}

TEST_F(UtestModelManagerModelManager, MallocOutputsMemory_outputs_empty_success_with_gertTensor) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  auto compute_graph = gert::ShareGraph::AicoreStaticGraph();
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  graph_node->SetComputeGraph(compute_graph);
  StubExecutor executor;
  graph_manager.executor_ = &executor;
  std::vector<gert::Tensor> outputs = {};

  EXPECT_EQ(MallocOutputsMemory(graph_id, graph_node, external_allocator, outputs), SUCCESS);
  EXPECT_NE(outputs[0].GetSize(), 0);
}

TEST_F(UtestModelManagerModelManager, test_CreateGertTensor) {
  unique_ptr<uint8_t[]> data_buf(new(std::nothrow) uint8_t[512]);
  gert::Tensor input_tensor = {{{1, 1, 1, 128}, {1, 1, 1, 128}},                // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              gert::kOnDeviceHbm,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *) data_buf.get()};
  GeTensorDescPtr tensor_desc = make_shared<GeTensorDesc>(GeShape({1, 16, 16, 3}));
  GraphManager graph_manager;
  tensor_desc->SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc->SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc->SetDataType(ge::DT_FLOAT16);
  tensor_desc->SetOriginDataType(ge::DT_FLOAT);
  tensor_desc->SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  EXPECT_EQ(CreateGertTensor(tensor_desc, input_tensor), SUCCESS);
}

TEST_F(UtestModelManagerModelManager, HandleCommand_Prof) {
  ModelManager mm;
  Command cmd;
  cmd.cmd_params.push_back("heterogeneous_host");
  cmd.cmd_params.push_back("1");
  unsetenv("GE_PROFILING_TO_STD_OUT");

  auto ret = mm.HandleProfStartCommand(cmd);
  EXPECT_EQ(ret, SUCCESS);

  ret = mm.HandleProfStopCommand(cmd);
  EXPECT_EQ(ret, SUCCESS);
  setenv("GE_PROFILING_TO_STD_OUT", "1", 1);
}

TEST_F(UtestModelManagerModelManager, InitOpMasterDeviceSo_FromOm_Success) {
  std::string opp_path = __FILE__;
  std::vector<std::pair<ccKernelType, const std::string>> kernel_type_so_names;

  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "/test_tmp/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  ConstructOpMasterDeviceSo(opp_path, 1, 2, true, kernel_type_so_names);

  ModelBufferData model;
  ModelHelper model_helper;
  model_helper.SetSaveMode(true);
  GeRootModelPtr ge_root_model = ConstructGeRootModel(kernel_type_so_names);

  EXPECT_EQ(ge_root_model->CheckAndSetNeedSoInOM(), SUCCESS);
  EXPECT_EQ(ge_root_model->GetSoInOmFlag(), 0x4000);

  std::string output_file = opp_path + "/output.om";
  EXPECT_EQ(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, false), SUCCESS);

  auto &model_mgr = ModelManager::GetInstance();
  model_mgr.built_in_op_master_so_names_to_bin_.clear();
  model_mgr.cust_op_master_so_names_to_bin_.clear();
  ge::ModelParserBase base;
  ge::ModelData model_data;
  EXPECT_EQ(base.LoadFromFile(output_file.c_str(), -1, model_data), SUCCESS);
  uint32_t model_id = 0;
  const ModelParam param;
  EXPECT_NE(ModelManager::GetInstance().LoadModelOffline(model_data, param, model_id), SUCCESS);
  if (model_data.model_data != nullptr) {
    delete[] reinterpret_cast<char_t *>(model_data.model_data);
  }
  EXPECT_EQ(model_mgr.built_in_op_master_so_names_to_bin_.size(), 1UL);
  EXPECT_EQ(model_mgr.cust_op_master_so_names_to_bin_.size(), 1UL);
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestModelManagerModelManager, InitOpMasterDeviceSo_FromPackage_Success) {
  std::string opp_path = __FILE__;
  std::vector<std::pair<ccKernelType, const std::string>> kernel_type_so_names;

  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "/test_tmp/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  ConstructOpMasterDeviceSo(opp_path, 1, 2, true, kernel_type_so_names);

  ModelBufferData model;
  GeRootModelPtr ge_root_model = ConstructGeRootModel(kernel_type_so_names);

  EXPECT_EQ(ge_root_model->CheckAndSetNeedSoInOM(), SUCCESS);
  EXPECT_EQ(ge_root_model->GetSoInOmFlag(), 0x4000);

  std::string output_file = opp_path + "/output.om";
  ModelHelper model_helper;
  model_helper.SetSaveMode(false);
  EXPECT_EQ(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, false), SUCCESS);

  auto &model_mgr = ModelManager::GetInstance();
  model_mgr.built_in_op_master_so_names_to_bin_.clear();
  model_mgr.cust_op_master_so_names_to_bin_.clear();
  ge::ModelData model_data;
  model_data.model_data = PtrToPtr<uint8_t, void>(model.data.get());
  model_data.model_len = model.length;
  uint32_t model_id = 0;
  const ModelParam param;
  EXPECT_NE(ModelManager::GetInstance().LoadModelOffline(model_data, param, model_id), SUCCESS);
  EXPECT_EQ(model_mgr.built_in_op_master_so_names_to_bin_.size(), 1UL);
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestModelManagerModelManager, MallocWeightsMem) {
  std::map<std::string, std::string> options;
  options[GRAPH_MAX_PARALLEL_MODEL_NUM] = "8";
  GetThreadLocalContext().SetSessionOption(options);
  uint32_t dev_id = 1;
  size_t mem_size = 100;
  const std::string weights_mem_id = "1_1_graph";
  auto mem1 = ModelManager::MallocWeightsMem(weights_mem_id, dev_id, mem_size);
  auto mem2 = ModelManager::MallocWeightsMem(weights_mem_id, dev_id, mem_size);
  EXPECT_NE(mem1, nullptr);
  EXPECT_TRUE(mem1 == mem2);
  auto iter1 = ModelManager::weights_mem_ids_to_addr_info_.find(weights_mem_id);
  EXPECT_NE(iter1, ModelManager::weights_mem_ids_to_addr_info_.end());

  EXPECT_EQ(ModelManager::FreeWeightsMem(weights_mem_id, dev_id, mem1), SUCCESS);
  EXPECT_EQ(ModelManager::FreeWeightsMem(weights_mem_id, dev_id, mem2), SUCCESS);
  auto iter2 = ModelManager::weights_mem_ids_to_addr_info_.find(weights_mem_id);
  EXPECT_EQ(iter2, ModelManager::weights_mem_ids_to_addr_info_.end());
}
}  // namespace ge

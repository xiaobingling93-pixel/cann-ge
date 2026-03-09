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
#include <memory>

#include "macro_utils/dt_public_scope.h"
#include "graph/manager/graph_var_manager.h"
#include "common/file_constant_utils/file_constant_utils.h"
#include "graph/manager/mem_manager.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "graph/utils/constant_utils.h"
#include "graph/utils/tensor_utils.h"
#include "framework/common/types.h"
#include "graph/compute_graph.h"
#include "depends/runtime/src/runtime_stub.h"
#include "graph/manager/active_memory_allocator.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"

namespace ge {
namespace {
GeTensorPtr CreateTensor(const std::vector<uint8_t> &value) {
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<int64_t> shape{(int64_t)value.size()};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  return tensor;
}

OpDescPtr CreateOpDesc(const std::string &name, const GeTensorPtr &tensor) {
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, CONSTANTOP);
  op_desc->AddOutputDesc(tensor->MutableTensorDesc());
  ConstantUtils::SetWeight(op_desc, 0, tensor);

  return op_desc;
}

OpDescPtr CreateFileConstOpDesc(const std::string &name, const GeTensorPtr &tensor, const std::string &file_path) {
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, FILECONSTANT);
  op_desc->AddOutputDesc(tensor->MutableTensorDesc());
  int64_t offset = 0U;
  auto length = static_cast<int64_t>(tensor->GetData().GetSize());
  FileConstantUtils::SetFileConstantPath(op_desc, file_path, offset, length);
  return op_desc;
}
}
class UtestGraphVarManagerTest : public testing::Test {
 protected:
  void SetUp() {
    VarManagerPool::Instance().Destory();
  }
  void TearDown() {
    VarManagerPool::Instance().Destory();
  }
};

TEST_F(UtestGraphVarManagerTest, test_set_memory_malloc_size_no_related_option) {
  const map<string, string> options{};
  EXPECT_EQ(VarManager::Instance(0)->SetMemoryMallocSize(options, 1024UL * 1024UL * 1024UL), SUCCESS);
  EXPECT_EQ(VarManager::Instance(0)->graph_mem_max_size_, floor(1024UL * 1024UL * 1024UL * (26.0f / 32.0f)));
  EXPECT_EQ(VarManager::Instance(0)->var_mem_max_size_, floor(1024UL * 1024UL * 1024UL * (5.0f / 32.0f)));
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);
}

TEST_F(UtestGraphVarManagerTest, test_set_memory_malloc_size_with_user_specify_graph_mem_max_size) {
  const map<string, string> options{{"ge.graphMemoryMaxSize", "536870912"}};
  Status ret = VarManager::Instance(0)->SetMemoryMallocSize(options, 1024UL * 1024UL * 1024UL);
  EXPECT_EQ(VarManager::Instance(0)->graph_mem_max_size_, floor(1024UL * 1024UL * 1024UL * (26.0f / 32.0f)));
  EXPECT_EQ(VarManager::Instance(0)->var_mem_max_size_, floor(1024UL * 1024UL * 1024UL * (5.0f / 32.0f)));
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphVarManagerTest, test_set_memory_malloc_size_with_user_specify_var_mem_max_size) {
  const map<string, string> options{{"ge.variableMemoryMaxSize", "536870912"}};
  Status ret = VarManager::Instance(0)->SetMemoryMallocSize(options, 1024UL * 1024UL * 1024UL);
  EXPECT_EQ(VarManager::Instance(0)->graph_mem_max_size_, floor(1024UL * 1024UL * 1024UL * (26.0f / 32.0f)));
  EXPECT_EQ(VarManager::Instance(0)->var_mem_max_size_, floor(1024UL * 1024UL * 1024UL * (5.0f / 32.0f)));
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphVarManagerTest, test_mem_manager_not_set) {
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);

  EXPECT_EQ(VarManager::Instance(0)->GetVarMemoryAddr(nullptr, RT_MEMORY_RDMA_HBM), nullptr);

  GeTensorDesc tensor_desc;
  EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("global_step", nullptr, tensor_desc, RT_MEMORY_RDMA_HBM), INTERNAL_ERROR);
  
  EXPECT_EQ(VarManager::Instance(0)->GetVarMemoryAddr(nullptr, RT_MEMORY_RDMA_HBM), nullptr);
  EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("global_step", nullptr, tensor_desc, RT_MEMORY_RDMA_HBM), INTERNAL_ERROR);
}

TEST_F(UtestGraphVarManagerTest, test_with_mem_manager) {
  const std::vector<rtMemType_t> memory_types({RT_MEMORY_HBM, RT_MEMORY_P2P_DDR});
  EXPECT_EQ(MemManager::Instance().Initialize(memory_types), SUCCESS);
  VarManager::Instance(0)->SetMemManager(&MemManager::Instance());
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);

  uint8_t logic_addr = 0;
  EXPECT_EQ(VarManager::Instance(0)->GetVarMemoryAddr(&logic_addr, RT_MEMORY_RDMA_HBM), &logic_addr);
  EXPECT_EQ(VarManager::Instance(0)->GetVarMemoryAddr(&logic_addr, RT_MEMORY_HBM), nullptr);

  // RdmaPoolAllocator block_bin_ not found.
  GeTensorDesc tensor_desc;
  EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("global_step", nullptr, tensor_desc, RT_MEMORY_RDMA_HBM), INTERNAL_ERROR);
}

TEST_F(UtestGraphVarManagerTest, test_var_manager_addr_and_free) {
  const std::vector<rtMemType_t> memory_types({RT_MEMORY_HBM, RT_MEMORY_P2P_DDR});
  EXPECT_EQ(MemManager::Instance().Initialize(memory_types), SUCCESS);
  VarManager::Instance(5)->SetMemManager(&MemManager::Instance());
  EXPECT_EQ(VarManager::Instance(5)->Init(0, 0, 0, 0), SUCCESS);

  const map<string, string> options{{"ge.graphMemoryMaxSize", "536870912"}};
  Status ret = VarManager::Instance(5)->Init(static_cast<uint32_t>(SessionVersion::MINI_VERSION), 1, 0, 0x5a5a);
  EXPECT_EQ(ret, SUCCESS);
  
  ret = VarManager::Instance(5)->SetMemoryMallocSize(options, 1024UL * 1024UL * 1024UL);
  EXPECT_EQ(ret, SUCCESS);
  std::vector<int64_t> s = {1,2,3,4};
  GeShape shape(s);
  GeTensorDesc tensor_desc(shape);
  TensorUtils::SetSize(tensor_desc, shape.GetShapeSize());
  std::string str = "global_step";
  ret = VarManager::Instance(5)->AssignVarMem(str, nullptr, tensor_desc, RT_MEMORY_HBM);
  EXPECT_EQ(ret, SUCCESS);

  int64_t logic_value = 34359738368;
  VarManager::Instance(5)->var_resource_->UpdateDevVarMgrInfo(0);
  ret = VarManager::Instance(5)->var_resource_->SetVarMgrDevAddr(0, logic_value, nullptr);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_NE(VarManager::Instance(5)->GetVarMemoryAddr(PtrToPtr<void, uint8_t>(ValueToPtr(34359738368)), RT_MEMORY_HBM), nullptr);
  EXPECT_EQ(VarManager::Instance(5)->FreeVarMemory(), SUCCESS);
  ret = VarManager::Instance(5)->var_resource_->SetVarMgrDevAddr(0, logic_value, reinterpret_cast<uint8_t *>(&logic_value));
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_NE(VarManager::Instance(5)->GetVarMemoryAddr(PtrToPtr<void, uint8_t>(ValueToPtr(34359738368)),
      RT_MEMORY_HBM), nullptr);
  logic_value = 0;
  EXPECT_EQ(VarManager::Instance(5)->GetVarMemoryAddr(reinterpret_cast<uint8_t *>(&logic_value), RT_MEMORY_HBM), nullptr);
  VarManager::Instance(5)->var_resource_->device_id_to_var_dev_addr_mgr_map_.clear();
}

TEST_F(UtestGraphVarManagerTest, Malloc1GHugePageFailed_Return) {
  class MockRuntime : public ge::RuntimeStub {
   public:
    rtError_t rtMallocPhysical(rtDrvMemHandle* handle, size_t size, rtDrvMemProp_t* prop, uint64_t flags) {
      pg_type_local = prop->pg_type;
      ++call_count;
      size_local = size;
      if (prop->pg_type == 2) {
        return -1;
      }
      *handle = (rtDrvMemHandle) new uint8_t[8];
      return 0;
    }
    rtError_t rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t *free, size_t *total) {
      *free = 64UL * 1024U * 1024U * 1024U;
      *total = 64UL * 1024U * 1024U * 1024U;
      return 0;
    }
    rtError_t rtGetRtCapability(rtFeatureType_t featureType, int32_t featureInfo, int64_t *value) {
      *value = 0;
      return 0;
    }
    uint32_t call_count = 0U;
    uint32_t pg_type_local = 0U;
    uint32_t size_local = 0;
  };
  auto old_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  std::map<std::string, std::string> options;
  options.insert(pair<string, string>(OPTION_VARIABLE_USE_1G_HUGE_PAGE, "1"));
  ge::GetThreadLocalContext().SetSessionOption(options);

  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);
  const std::vector<rtMemType_t> memory_types({RT_MEMORY_HBM, RT_MEMORY_P2P_DDR});
  EXPECT_EQ(MemManager::Instance().Initialize(memory_types), SUCCESS);
  VarManager::Instance(5)->SetMemManager(&MemManager::Instance());
  EXPECT_EQ(VarManager::Instance(5)->Init(0, 0, 0, 0), SUCCESS);

  options.insert(pair<string, string>("ge.graphMemoryMaxSize", "536870912"));
  Status ret = VarManager::Instance(5)->Init(static_cast<uint32_t>(SessionVersion::MINI_VERSION), 1, 0, 0x5a5a);
  EXPECT_EQ(ret, SUCCESS);
  auto mem_allocator1 =
      SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(0, 1, RT_MEMORY_HBM, kDrv1GPageSize);
  ASSERT_NE(mem_allocator1, nullptr);
  VarManager::Instance(5)->InitExpandableMemoryAllocator(mem_allocator1);

  
  ret = VarManager::Instance(5)->SetMemoryMallocSize(options, 1024UL * 1024UL * 1024UL);
  EXPECT_EQ(ret, SUCCESS);
  std::vector<int64_t> s = {1,2,3,4};
  GeShape shape(s);
  GeTensorDesc tensor_desc(shape);
  TensorUtils::SetSize(tensor_desc, shape.GetShapeSize());
  std::string str = "global_step";
  ret = VarManager::Instance(5)->AssignVarMem(str, nullptr, tensor_desc, RT_MEMORY_HBM);
  EXPECT_EQ(ret, SUCCESS);

  int64_t logic_value = 34359738368;
  VarManager::Instance(5)->var_resource_->UpdateDevVarMgrInfo(0);
  ret = VarManager::Instance(5)->var_resource_->SetVarMgrDevAddr(0, logic_value, nullptr);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(VarManager::Instance(5)->GetVarMemoryAddr(PtrToPtr<void, uint8_t>(ValueToPtr(34359738368)), RT_MEMORY_HBM), nullptr);
  EXPECT_EQ(VarManager::Instance(5)->FreeVarMemory(), SUCCESS);
  VarManager::Instance(5)->var_resource_->device_id_to_var_dev_addr_mgr_map_.clear();
  ge::RuntimeStub::Reset();
  ge::GetThreadLocalContext().SetSessionOption(old_options);
}

TEST_F(UtestGraphVarManagerTest, test_var_manager_serial_deserial) {
  const map<string, string> options{};
  Status ret = VarManager::Instance(1)->Init(static_cast<uint32_t>(SessionVersion::MINI_VERSION), 1, 0, 0x5a5a);
  EXPECT_EQ(ret, SUCCESS);
  
  ret = VarManager::Instance(1)->SetMemoryMallocSize(options, 1024UL * 1024UL * 1024UL);
  EXPECT_EQ(ret, SUCCESS);
  size_t graph_mem_max_size = VarManager::Instance(1)->graph_mem_max_size_;
  size_t var_mem_max_size = VarManager::Instance(1)->var_mem_max_size_;
  size_t var_mem_logic_base = VarManager::Instance(1)->var_mem_logic_base_;
  size_t use_max_mem_size = VarManager::Instance(1)->use_max_mem_size_;
  std::vector<int64_t> s = {1,2,3,4};
  GeShape shape(s);
  GeTensorDesc tensor_desc(shape);
  TensorUtils::SetSize(tensor_desc, shape.GetShapeSize());
  std::string str = "global_step";
  ret = VarManager::Instance(1)->AssignVarMem(str, nullptr, tensor_desc, RT_MEMORY_HBM);
  EXPECT_EQ(ret, SUCCESS);
  TransNodeInfo trans_node_info;
  VarTransRoad fusion_road;
  fusion_road.emplace_back(trans_node_info);
  VarManager::Instance(1)->SetTransRoad(str, fusion_road);

  VarBroadCastInfo broadcast_info;
  broadcast_info.var_name = "test";
  VarManager::Instance(1)->SaveBroadCastInfo(0, broadcast_info);

  deployer::VarManagerInfo info;
  ret = VarManager::Instance(1)->VarManagerToSerial(1, info);
  EXPECT_EQ(ret, SUCCESS);
  auto session_id = info.session_id();
  EXPECT_EQ(session_id, 1);
  info.set_session_id(2);
  ret = VarManager::Instance(2)->VarManagerToDeserial(2, info);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(VarManager::Instance(2)->graph_mem_max_size_, floor(1024UL * 1024UL * 1024UL * (26.0f / 32.0f)));
  EXPECT_EQ(VarManager::Instance(2)->var_mem_max_size_, floor(1024UL * 1024UL * 1024UL * (5.0f / 32.0f)));
  EXPECT_EQ(VarManager::Instance(2)->version_, SessionVersion::MINI_VERSION);
  EXPECT_EQ(VarManager::Instance(2)->job_id_, 0x5a5a);
  EXPECT_EQ(VarManager::Instance(2)->graph_mem_max_size_, graph_mem_max_size);
  EXPECT_EQ(VarManager::Instance(2)->var_mem_max_size_, var_mem_max_size);
  EXPECT_EQ(VarManager::Instance(2)->var_mem_logic_base_, var_mem_logic_base);
  EXPECT_EQ(VarManager::Instance(2)->use_max_mem_size_, use_max_mem_size);
  EXPECT_EQ(VarManager::Instance(2)->var_resource_->session_id_, 2);

  EXPECT_EQ(VarManager::Instance(2)->var_resource_->var_offset_map_.size(), 1);
  EXPECT_EQ(VarManager::Instance(2)->var_resource_->var_addr_mgr_map_.size(), 1);
  EXPECT_EQ(VarManager::Instance(2)->var_resource_->cur_var_tensor_desc_map_.size(), 1);

  EXPECT_EQ(VarManager::Instance(2)->var_resource_->IsVarExist(str, tensor_desc), true);
  EXPECT_EQ(VarManager::Instance(2)->mem_resource_map_.size(), 1);
  auto resource_src = VarManager::Instance(1)->mem_resource_map_[RT_MEMORY_HBM];
  auto resource = VarManager::Instance(2)->mem_resource_map_[RT_MEMORY_HBM];
  EXPECT_EQ(resource->var_mem_size_, 1536);
  EXPECT_EQ(resource->var_mem_size_, resource_src->var_mem_size_);

  ret = VarManager::Instance(2)->AssignVarMem("Hello_variable", nullptr, tensor_desc, RT_MEMORY_HBM);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphVarManagerTest, var_address_op_fail) {
  uint64_t session_id = 1;
  rtMemType_t memory_type = RT_MEMORY_HBM;
  Status retStatus;
  std::vector<int64_t> s = {1,2,3,4};
  GeShape shape(s);
  GeTensorDesc tensor_desc(shape);
  TensorUtils::SetSize(tensor_desc, shape.GetShapeSize());

  uint8_t *dev_ptr = nullptr;
  VarResource tmpVarRes(22);
  retStatus = tmpVarRes.GetVarAddr("", tensor_desc, &dev_ptr, memory_type);
  ASSERT_EQ(retStatus, FAILED);
  retStatus = tmpVarRes.GetVarAddr("", tensor_desc, nullptr, memory_type);
  ASSERT_EQ(retStatus, FAILED);

  VarManager tmpVarMng(22);
  retStatus = tmpVarMng.SetVarAddr("", tensor_desc, dev_ptr, memory_type, nullptr);
  EXPECT_EQ(retStatus, ge::INTERNAL_ERROR);
  retStatus = tmpVarMng.GetVarAddr("", tensor_desc, dev_ptr, memory_type);
  EXPECT_EQ(retStatus, ge::INTERNAL_ERROR);
#if 0
  std::shared_ptr<MemResource> memResPtr = std::make_shared<MemResource>();
  memResPtr.reset();
  tmpVarMng.mem_resource_map_[memory_type] = memResPtr;
  retInt = tmpVarMng.GetVarMemSize(memory_type);
  EXPECT_EQ(retInt, 0);
#endif
  const map<string, string> options{};
  Status ret = VarManager::Instance(session_id)->Init(0U, session_id, 0, 0x5a5a);
  EXPECT_EQ(ret, SUCCESS);
  ret = VarManager::Instance(session_id)->SetMemoryMallocSize(options, 1024UL * 1024UL * 1024UL);
  EXPECT_EQ(ret, SUCCESS);

  std::string var_name = "global_step";
  ret = VarManager::Instance(session_id)->AssignVarMem(var_name, nullptr, tensor_desc, memory_type);
  EXPECT_EQ(ret, SUCCESS);

  retStatus = VarManager::Instance(session_id)->var_resource_->SaveVarAddr(var_name,
                                                                           tensor_desc,
                                                                           dev_ptr,
                                                                           memory_type, nullptr);
  EXPECT_EQ(retStatus, FAILED);

}

TEST_F(UtestGraphVarManagerTest, renew_current_var_desc) {
  VarResource tmpVarRes(1);

  std::vector<int64_t> s = {1,2,3,4};
  GeShape shape(s);
  GeTensorDesc tensor_desc(shape);
  // var not exist
  Status retStatus = tmpVarRes.RenewCurVarDesc("some_var", tensor_desc);
  ASSERT_EQ(retStatus, SUCCESS);
  retStatus = tmpVarRes.GetCurVarDesc("some_var", tensor_desc);
  ASSERT_EQ(retStatus, FAILED);

  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Add", ADD);
  tmpVarRes.SetVarAddr("some_var", tensor_desc, nullptr, RT_MEMORY_RDMA_HBM, op_desc);
  retStatus = tmpVarRes.RenewCurVarDesc("some_var", tensor_desc);
  ASSERT_EQ(retStatus, SUCCESS);
  retStatus = tmpVarRes.GetCurVarDesc("some_var", tensor_desc);
  ASSERT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestGraphVarManagerTest, VarManager_WithNo_varResource) {
  Status retStatus;
  bool retBool;
  rtMemType_t retMemType; 
  VarTransRoad *retVarTransRoad;
  VarManager::Instance(1)->var_resource_.reset();

  ge::GeTensorDesc tensor_desc;
  VarManager::Instance(1)->SetVarIsReady(std::string("a"), tensor_desc, 0U);
  retBool = VarManager::Instance(1)->IsVarReady(std::string("a"), tensor_desc, 0U);
  EXPECT_EQ(retBool, false);

  retBool = VarManager::Instance(1)->IsVarExist(std::string("a"), tensor_desc);
  EXPECT_EQ(retBool, false);

  deployer::VarManagerInfo info;
  retStatus = VarManager::Instance(1)->VarManagerToSerial(1, info);
  EXPECT_EQ(retStatus, INTERNAL_ERROR);

  VarBroadCastInfo broad_cast_info;
  retStatus = VarManager::Instance(1)->SaveBroadCastInfo(0, broad_cast_info);
  EXPECT_EQ(retStatus, INTERNAL_ERROR);

  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Add", ADD);
  retStatus = VarManager::Instance(1)->RenewCurVarDesc(std::string("a"), op_desc);
  EXPECT_EQ(retStatus, INTERNAL_ERROR);

  retStatus = VarManager::Instance(1)->RenewCurVarDesc(std::string("a"), tensor_desc);
  EXPECT_EQ(retStatus, INTERNAL_ERROR);

  retStatus = VarManager::Instance(1)->RecordStagedVarDesc(0, std::string("a"), tensor_desc);
  EXPECT_EQ(retStatus, INTERNAL_ERROR);

  deployer::VarDescInfo desc_info;
  retStatus = VarManager::Instance(1)->VarDescInfoToSerial(0, desc_info);
  EXPECT_EQ(retStatus, INTERNAL_ERROR);

  auto ret = VarManager::Instance(1)->GetStagedVarDescs(0);
  EXPECT_EQ(ret.size(), 0);

  auto set_ret = VarManager::Instance(1)->GetChangedVarNames(0);
  EXPECT_EQ(set_ret.size(), 0);

  retMemType = VarManager::Instance(1)->GetVarMemType(0);
  EXPECT_EQ(retMemType, RT_MEMORY_RESERVED);

  VarTransRoad trans_road;
  retStatus = VarManager::Instance(1)->SetTransRoad(std::string("a"), trans_road);
  EXPECT_EQ(retStatus, INTERNAL_ERROR);

  retVarTransRoad = VarManager::Instance(1)->GetTransRoad(std::string("a"));
  EXPECT_EQ(retVarTransRoad, nullptr);

  uint32_t graph_id = 0;
  retStatus = VarManager::Instance(1)->GetChangedGraphId(std::string("a"), graph_id);
  EXPECT_EQ(retStatus, INTERNAL_ERROR);
  VarManager::Instance(1)->RemoveChangedGraphId(std::string("a"));

  retStatus = VarManager::Instance(1)->SetAllocatedGraphId(std::string("a"), 0);
  EXPECT_EQ(retStatus, INTERNAL_ERROR);

  retStatus = VarManager::Instance(1)->GetAllocatedGraphId(std::string("a"), graph_id);
  EXPECT_EQ(retStatus, INTERNAL_ERROR);

  std::map<std::string, GeTensorDesc> all_variables;
  retStatus = VarManager::Instance(1)->GetAllVariables(all_variables);
  EXPECT_EQ(retStatus, INTERNAL_ERROR);

  retBool = VarManager::Instance(1)->HasSharedVarMemBetweenBatch();
  EXPECT_EQ(retBool, false);
}

TEST_F(UtestGraphVarManagerTest, aoe_const_mem_reuse_succ) {
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);

  auto tensor1 = CreateTensor({1, 2, 3});
  auto const1 = CreateOpDesc("const1", tensor1);
  EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("const1", const1, tensor1->MutableTensorDesc(), RT_MEMORY_HBM), SUCCESS);
  auto const2 = CreateOpDesc("const2", tensor1);
  EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("const2", const2, tensor1->MutableTensorDesc(), RT_MEMORY_HBM), SUCCESS);
  auto tensor2 = CreateTensor({1, 2, 3, 4, 5});
  auto const3 = CreateOpDesc("const3", tensor2);
  EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("const3", const3, tensor2->MutableTensorDesc(), RT_MEMORY_HBM), SUCCESS);

  EXPECT_EQ(VarManager::Instance(0)->var_resource_->GetVarMgrInfo(0, 0), nullptr);
  EXPECT_EQ(VarManager::Instance(0)->var_resource_->SetVarMgrDevAddr(0, 0, nullptr), INTERNAL_ERROR);
  VarManager::Instance(0)->var_resource_->UpdateDevVarMgrInfo(0);
  EXPECT_NE(VarManager::Instance(0)->var_resource_->GetVarMgrInfo(0, 34359738368), nullptr);
  EXPECT_EQ(VarManager::Instance(0)->var_resource_->GetVarMgrInfo(0, 0), nullptr);
  EXPECT_NE(VarManager::Instance(0)->var_resource_->SetVarMgrDevAddr(0, 0, nullptr), SUCCESS);
  EXPECT_EQ(VarManager::Instance(0)->var_resource_->SetVarMgrDevAddr(0, 34359738368, nullptr), SUCCESS);
}

TEST_F(UtestGraphVarManagerTest, fileconstant_mem_reuse_succ) {
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);

  auto tensor1 = CreateTensor({1, 2, 3});
  std::string file_path1 = "tmp_weight/12345/weight.bin";
  auto file_const1 = CreateFileConstOpDesc("file_const1", tensor1, file_path1);
  EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("file_const1", file_const1, tensor1->MutableTensorDesc(), RT_MEMORY_HBM), SUCCESS);
  auto file_const2 = CreateFileConstOpDesc("file_const2", tensor1, file_path1);
  EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("file_const2", file_const2, tensor1->MutableTensorDesc(), RT_MEMORY_HBM), SUCCESS);
  std::string file_path2 = "tmp_weight/12345/weight1.bin";
  auto file_const3 = CreateFileConstOpDesc("const3", tensor1, file_path2);
  EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("file_const3", file_const3, tensor1->MutableTensorDesc(), RT_MEMORY_HBM), SUCCESS);
  // clear file constant reuse key
  VarManager::Instance(0)->var_resource_->var_addr_mgr_map_.clear();
  auto file_const4 = CreateFileConstOpDesc("file_const4", tensor1, file_path1);
  EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("file_const4", file_const1, tensor1->MutableTensorDesc(), RT_MEMORY_HBM), SUCCESS);
}

TEST_F(UtestGraphVarManagerTest, get_reuse_addr_failed_with_no_aligned_ptr) {
  std::vector<uint8_t> value = {1, 2, 3};
  ge::GeTensorPtr tensor1 = std::make_shared<GeTensor>();
  std::vector<int64_t> shape{(int64_t)value.size()};
  tensor1->MutableTensorDesc().SetShape(GeShape(shape));
  tensor1->MutableTensorDesc().SetDataType(DT_UINT8);

  auto const1 = CreateOpDesc("const1", tensor1);
  uint8_t *mem_offset = nullptr;
  rtMemType_t memory_type = RT_MEMORY_HBM;
  VarResource var(0);
  EXPECT_EQ(var.GetReuseAddr(const1, &mem_offset, memory_type), ge::FAILED);
}

TEST_F(UtestGraphVarManagerTest, evaluate_graph_resource_mode) {
  std::map<std::string, std::string> options;
  size_t total_mem_size = 32UL * 1024UL * 1024UL * 1024UL;

  options.emplace(EVALUATE_GRAPH_RESOURCE_MODE, "1");
  ge::GetThreadLocalContext().SetGraphOption(options);
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);
  EXPECT_EQ(VarManager::Instance(0)->GetVarMemMaxSize(true), std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(VarManager::Instance(0)->GetGraphMemoryMaxSize(true), std::numeric_limits<uint64_t>::max());

  options.clear();
  options.emplace(EVALUATE_GRAPH_RESOURCE_MODE, "0");
  ge::GetThreadLocalContext().SetGraphOption(options);
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);
  EXPECT_EQ(VarManager::Instance(0)->GetVarMemMaxSize(), kMemoryVarManagerMallocSize);
  EXPECT_EQ(VarManager::Instance(0)->GetGraphMemoryMaxSize(),
            floor(static_cast<float64_t>(total_mem_size) * kGraphMemoryManagerMallocRatio));

  options.clear();
  ge::GetThreadLocalContext().SetGraphOption(options);
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);
  EXPECT_EQ(VarManager::Instance(0)->GetVarMemMaxSize(), kMemoryVarManagerMallocSize);
  EXPECT_EQ(VarManager::Instance(0)->GetGraphMemoryMaxSize(),
            floor(static_cast<float64_t>(total_mem_size) * kGraphMemoryManagerMallocRatio));
}

TEST_F(UtestGraphVarManagerTest, multi_batch_trans_road) {
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);
  std::string batch_name = "var1_ascend_mbatch_batch_0";
  VarManager::Instance(0)->SetBatchVariablesKeyName(batch_name, "var1");
  TransNodeInfo trans_node_info;
  trans_node_info.node_type = "TransData";
  VarTransRoad var_trans_road{trans_node_info};
  VarManager::Instance(0)->SetTransRoad("var1", var_trans_road);
  auto get_road = VarManager::Instance(0)->GetTransRoad("var1_ascend_mbatch_batch_0");
  EXPECT_EQ(get_road->front().node_type, "TransData");
}

TEST_F(UtestGraphVarManagerTest, const_place_holder_set_addr_success) {
  const std::vector<rtMemType_t> memory_types({RT_MEMORY_HBM, RT_MEMORY_P2P_DDR});
  EXPECT_EQ(MemManager::Instance().Initialize(memory_types), SUCCESS);
  VarManager::Instance(0)->SetMemManager(&MemManager::Instance());
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);
  auto tensor1 = CreateTensor({1, 2, 3});
  ge::TensorUtils::SetSize(tensor1->MutableTensorDesc(), 24L);
  auto constplaceholder_op_desc1 = CreateOpDesc("ConstPlaceHolder", tensor1);
  constplaceholder_op_desc1->SetType("ConstPlaceHolder");
  vector<int64_t > shape({1, 2, 3});
  ge::AttrUtils::SetListInt(constplaceholder_op_desc1, "origin_shape", shape);
  ge::AttrUtils::SetListInt(constplaceholder_op_desc1, "storage_shape", shape);
  DataType data_type = DT_FLOAT;
  ge::AttrUtils::SetDataType(constplaceholder_op_desc1, "dtype", data_type); // float
  int64_t data_length = 24L;
  ge::AttrUtils::SetInt(constplaceholder_op_desc1, "size", data_length);
  int64_t placement = 1L;
  ge::AttrUtils::SetInt(constplaceholder_op_desc1, "placement", placement); // device
  int64_t device_addr = 20000;
  ge::AttrUtils::SetInt(constplaceholder_op_desc1, "addr", device_addr);
  uint64_t logic_address = VarManager::Instance(0)->GetVarMemLogicBase();
  EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("constplacehoulder_test1", constplaceholder_op_desc1,
                                                  tensor1->MutableTensorDesc(), RT_MEMORY_HBM), SUCCESS);
  EXPECT_EQ(VarManager::Instance(0)->GetVarMemoryAddr(reinterpret_cast<uint8_t *>(logic_address), RT_MEMORY_HBM),
            reinterpret_cast<void *>(device_addr));
}

TEST_F(UtestGraphVarManagerTest, test_check_and_set_var_loaded) {
  auto tensor1 = CreateTensor({1, 2, 3});
  auto const1 = CreateOpDesc("const1", tensor1);
  EXPECT_EQ(VarManager::Instance(0)->CheckAndSetVarLoaded(const1, 0), false);
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);
  EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("const1", const1, tensor1->MutableTensorDesc(), RT_MEMORY_HBM), SUCCESS);
  uint8_t *dev_ptr = nullptr;
  EXPECT_EQ(VarManager::Instance(0)->GetVarAddr(const1->GetName(), tensor1->GetTensorDesc(), dev_ptr), SUCCESS);
  std::vector<int64_t> output_list{static_cast<int64_t>(PtrToValue(dev_ptr))};
  const1->SetOutputOffset(output_list);
  EXPECT_EQ(VarManager::Instance(0)->CheckAndSetVarLoaded(const1, 0), false);
  // Load again
  EXPECT_EQ(VarManager::Instance(0)->CheckAndSetVarLoaded(const1, 0), true);
}

TEST_F(UtestGraphVarManagerTest, test_var_manager_restore_var_mem) {
  const std::vector<rtMemType_t> memory_types({RT_MEMORY_HBM, RT_MEMORY_P2P_DDR});
  EXPECT_EQ(MemManager::Instance().Initialize(memory_types), SUCCESS);
  uint32_t session_id = 9887U;
  auto var_manager = VarManager::Instance(session_id);
  var_manager->SetMemManager(&MemManager::Instance());
  EXPECT_EQ(var_manager->Init(0, session_id, 0, 0), SUCCESS);

  const map<string, string> options{{"ge.graphMemoryMaxSize", "536870912"}};
  EXPECT_EQ(var_manager->SetMemoryMallocSize(options, 1024UL * 1024UL * 1024UL), SUCCESS);

  OpDescPtr desc = std::make_shared<OpDesc>("tmp_var", VARIABLE);
  std::vector<int64_t> s = {1, 2, 3, 4};
  GeShape shape(s);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetFormat(FORMAT_NCHW);
  TensorUtils::SetSize(tensor_desc, shape.GetShapeSize());
  desc->AddOutputDesc(tensor_desc);
  desc->SetOutputOffset({137438953472U});
  EXPECT_EQ(var_manager->RestoreVarMem("tmp_var", desc, tensor_desc, RT_MEMORY_HBM), SUCCESS);

  uint8_t *var_logic{nullptr};
  EXPECT_EQ(var_manager->GetVarAddr("tmp_var", tensor_desc, var_logic), SUCCESS);
  int64_t logic_value = 137438953472U;
  EXPECT_EQ(var_manager->GetVarMemoryAddr(reinterpret_cast<uint8_t *>(&logic_value), RT_MEMORY_HBM), nullptr);

  GeTensorDesc tensor_desc_nd(shape);
  tensor_desc_nd.SetFormat(FORMAT_ND);
  TensorUtils::SetSize(tensor_desc_nd, shape.GetShapeSize());
  EXPECT_EQ(var_manager->RestoreVarMem("tmp_var", desc, tensor_desc_nd, RT_MEMORY_HBM), SUCCESS);
  uint8_t *var_logic_nd{nullptr};
  EXPECT_EQ(var_manager->GetVarAddr("tmp_var", tensor_desc, var_logic_nd), SUCCESS);
  EXPECT_EQ(var_logic, var_logic_nd);

  std::vector<int64_t> s2 = {16, 2, 3, 4};
  GeShape shape2(s2);
  GeTensorDesc tensor_desc2(shape2);
  tensor_desc2.SetDataType(DT_COMPLEX64);
  TensorUtils::SetSize(tensor_desc2, shape2.GetShapeSize());
  EXPECT_NE(var_manager->RestoreVarMem("tmp_var", desc, tensor_desc2, RT_MEMORY_HBM), SUCCESS);
  var_manager->Destory();
}

TEST_F(UtestGraphVarManagerTest, test_init_var_if_has_init_value_match) {
    // Initialize VarManager
    const std::vector<rtMemType_t> memory_types({RT_MEMORY_HBM, RT_MEMORY_P2P_DDR});
    EXPECT_EQ(MemManager::Instance().Initialize(memory_types), SUCCESS);

    VarManager::Instance(0)->SetMemManager(&MemManager::Instance());
    EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);

    // Create variable tensor with placement device
    std::vector<int64_t> var_shape{1, 1, 1, 1, 10}; 
    GeShape shape(var_shape);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetDataType(DT_FLOAT);
    tensor_desc.SetFormat(FORMAT_NCHW);
    TensorUtils::SetSize(tensor_desc, 10 * sizeof(float)); 

    // Create init_value tensor with matching format and type
    std::vector<float> init_data(10, 1.0f); 
    auto init_tensor = std::make_shared<GeTensor>();
    GeTensorDesc init_desc(GeShape(var_shape), FORMAT_NCHW, DT_FLOAT);
    init_tensor->SetData(reinterpret_cast<uint8_t*>(init_data.data()), init_data.size() * sizeof(float));
    init_tensor->MutableTensorDesc() = init_desc;

    // Set init_value attribute
    EXPECT_TRUE(ge::AttrUtils::SetTensor(&tensor_desc, ATTR_NAME_INIT_VALUE, init_tensor));

    // Create OpDesc
    OpDescPtr op_desc = std::make_shared<OpDesc>("test_var_match", VARIABLE);
    op_desc->AddOutputDesc(tensor_desc);

    // Assign variable memory
    std::string var_name = "test_var_match";
    Status status = VarManager::Instance(0)->AssignVarMem(var_name, op_desc, tensor_desc, RT_MEMORY_HBM);
    EXPECT_EQ(status, SUCCESS);

    VarManager::Instance(0)->var_resource_->UpdateDevVarMgrInfo(0);

    // Get the variable logical address
    uint8_t* logic_addr = nullptr;
    status = VarManager::Instance(0)->GetVarAddr(var_name, tensor_desc, logic_addr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_NE(logic_addr, nullptr);

    // Get GetVarMemoryAddr to trigger InitVarIfHasInitValue
    uint8_t *dev_addr = VarManager::Instance(0)->GetVarMemoryAddr(var_name, logic_addr, RT_MEMORY_HBM, 0);
    EXPECT_NE(dev_addr, nullptr);

    // Cast dev_addr to float pointer to access the data
    float *device_data = reinterpret_cast<float *>(dev_addr);

    // Compare the data in device memory with the init_data
    for (size_t i = 0; i < init_data.size(); ++i) {
      EXPECT_FLOAT_EQ(device_data[i], init_data[i]);
    }
}

TEST_F(UtestGraphVarManagerTest, test_external_var) {
  // Initialize VarManager
  const std::vector<rtMemType_t> memory_types({RT_MEMORY_HBM, RT_MEMORY_P2P_DDR});
  EXPECT_EQ(MemManager::Instance().Initialize(memory_types), SUCCESS);

  VarManager::Instance(0)->SetMemManager(&MemManager::Instance());
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);

  // Create variable tensor with placement device
  std::vector<int64_t> var_shape{1, 1, 1, 1, 10};
  GeShape shape(var_shape);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetFormat(FORMAT_NCHW);
  TensorUtils::SetSize(tensor_desc, 10 * sizeof(float));

  // Create OpDesc
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_var_match", VARIABLE);
  op_desc->AddOutputDesc(tensor_desc);

  // Assign variable memory
  std::string var_name = "test_var_match";
  Status status = VarManager::Instance(0)->AssignVarMem(var_name, op_desc, tensor_desc, RT_MEMORY_HBM);
  EXPECT_EQ(status, SUCCESS);

  VarManager::Instance(0)->var_resource_->UpdateDevVarMgrInfo(0);
  uint8_t* logic_addr = nullptr;
  status = VarManager::Instance(0)->GetVarAddr(var_name, tensor_desc, logic_addr);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_NE(logic_addr, nullptr);
  const size_t total_var_size = VarManager::Instance(0)->GetVarMemSize(RT_MEMORY_HBM);
  void *external_var_addr = nullptr;
  rtMalloc(&external_var_addr, total_var_size, RT_MEMORY_DEFAULT, GE_MODULE_NAME_U16);
  EXPECT_NE(external_var_addr, nullptr);
  VarManager::Instance(0)->SetExternalVar(external_var_addr, total_var_size);

  uint8_t *dev_addr = VarManager::Instance(0)->GetVarMemoryAddr(var_name, logic_addr, RT_MEMORY_HBM, 0);
  EXPECT_NE(dev_addr, nullptr);
  const size_t real_offset = PtrToValue(logic_addr) - VarManager::Instance(0)->GetVarMemLogicBase();
  EXPECT_EQ(dev_addr, (PtrToPtr<void, uint8_t>(external_var_addr) + real_offset));
  rtFree(external_var_addr);
  VarManager::Instance(0)->SetExternalVar(nullptr, 0);
}

TEST_F(UtestGraphVarManagerTest, test_init_var_if_has_init_value_format_mismatch) {
  // Initialize VarManager 
  const std::vector<rtMemType_t> memory_types({RT_MEMORY_HBM, RT_MEMORY_P2P_DDR});
  EXPECT_EQ(MemManager::Instance().Initialize(memory_types), SUCCESS);

  VarManager::Instance(0)->SetMemManager(&MemManager::Instance());
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);

  // Create variable tensor with placement device
  std::vector<int64_t> var_shape{8, 1, 64, 64, 16};
  GeShape shape(var_shape);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetFormat(FORMAT_NCHW);  // Variable uses NCHW format

  // Create init_value tensor with format NHWC (mismatched)
  std::vector<float> init_data(8 * 1 * 64 * 64 * 16, 1.0f);
  auto init_tensor = std::make_shared<GeTensor>();
  GeTensorDesc init_desc(GeShape(var_shape), FORMAT_NHWC, DT_FLOAT);  // Mismatched format
  init_tensor->SetData(reinterpret_cast<uint8_t*>(init_data.data()), init_data.size() * sizeof(float));
  init_tensor->MutableTensorDesc() = init_desc;

  // Set init_value attribute
  EXPECT_TRUE(ge::AttrUtils::SetTensor(&tensor_desc, ATTR_NAME_INIT_VALUE, init_tensor));

  // Create OpDesc
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_var_format_mismatch", VARIABLE);
  op_desc->AddOutputDesc(tensor_desc);

  // Assign variable memory
  std::string var_name = "test_var_format_mismatch";
  Status status = VarManager::Instance(0)->AssignVarMem(var_name, op_desc, tensor_desc, RT_MEMORY_HBM);
  EXPECT_EQ(status, SUCCESS);

  VarManager::Instance(0)->var_resource_->UpdateDevVarMgrInfo(0);

  uint8_t* logic_addr = nullptr;
  status = VarManager::Instance(0)->GetVarAddr(var_name, tensor_desc, logic_addr);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_NE(logic_addr, nullptr);

  uint8_t* dev_addr = VarManager::Instance(0)->GetVarMemoryAddr(var_name, logic_addr, RT_MEMORY_HBM, 0);

  EXPECT_EQ(dev_addr, nullptr);  
}

TEST_F(UtestGraphVarManagerTest, test_init_var_if_has_init_value_shape_mismatch) {

  const std::vector<rtMemType_t> memory_types({RT_MEMORY_HBM, RT_MEMORY_P2P_DDR});
  EXPECT_EQ(MemManager::Instance().Initialize(memory_types), SUCCESS);

  VarManager::Instance(0)->SetMemManager(&MemManager::Instance());
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);

  std::vector<int64_t> var_shape{8, 1, 64, 64, 16};
  GeShape shape(var_shape);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetFormat(FORMAT_NCHW);

  std::vector<int64_t> init_shape{8, 1, 32, 32, 16};  
  std::vector<float> init_data(8 * 1 * 32 * 32 * 16, 1.0f);
  auto init_tensor = std::make_shared<GeTensor>();
  GeTensorDesc init_desc(GeShape(init_shape), FORMAT_NCHW, DT_FLOAT);
  init_tensor->SetData(reinterpret_cast<uint8_t*>(init_data.data()), init_data.size() * sizeof(float));
  init_tensor->MutableTensorDesc() = init_desc;

  EXPECT_TRUE(ge::AttrUtils::SetTensor(&tensor_desc, ATTR_NAME_INIT_VALUE, init_tensor));

  OpDescPtr op_desc = std::make_shared<OpDesc>("test_var_shape_mismatch", VARIABLE);
  op_desc->AddOutputDesc(tensor_desc);

  std::string var_name = "test_var_shape_mismatch";
  Status status = VarManager::Instance(0)->AssignVarMem(var_name, op_desc, tensor_desc, RT_MEMORY_HBM);
  EXPECT_EQ(status, SUCCESS);

  VarManager::Instance(0)->var_resource_->UpdateDevVarMgrInfo(0);

  uint8_t* logic_addr = nullptr;
  status = VarManager::Instance(0)->GetVarAddr(var_name, tensor_desc, logic_addr);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_NE(logic_addr, nullptr);

  uint8_t* dev_addr = VarManager::Instance(0)->GetVarMemoryAddr(var_name, logic_addr, RT_MEMORY_HBM, 0);

  EXPECT_EQ(dev_addr, nullptr);
}

TEST_F(UtestGraphVarManagerTest, test_init_var_if_has_init_value_dtype_mismatch) {

  const std::vector<rtMemType_t> memory_types({RT_MEMORY_HBM, RT_MEMORY_P2P_DDR});
  EXPECT_EQ(MemManager::Instance().Initialize(memory_types), SUCCESS);

  VarManager::Instance(0)->SetMemManager(&MemManager::Instance());
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);

  std::vector<int64_t> var_shape{8, 1, 64, 64, 16};
  GeShape shape(var_shape);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetDataType(DT_FLOAT);  
  tensor_desc.SetFormat(FORMAT_NCHW);

  std::vector<float> init_data(8 * 1 * 64 * 64 * 16, 1.0f);
  auto init_tensor = std::make_shared<GeTensor>();
  GeTensorDesc init_desc(GeShape(var_shape), FORMAT_NCHW, DT_INT32);  
  init_tensor->SetData(reinterpret_cast<uint8_t*>(init_data.data()), init_data.size() * sizeof(float));
  init_tensor->MutableTensorDesc() = init_desc;

  EXPECT_TRUE(ge::AttrUtils::SetTensor(&tensor_desc, ATTR_NAME_INIT_VALUE, init_tensor));

  OpDescPtr op_desc = std::make_shared<OpDesc>("test_var_dtype_mismatch", VARIABLE);
  op_desc->AddOutputDesc(tensor_desc);

  std::string var_name = "test_var_dtype_mismatch";
  Status status = VarManager::Instance(0)->AssignVarMem(var_name, op_desc, tensor_desc, RT_MEMORY_HBM);
  EXPECT_EQ(status, SUCCESS);

  VarManager::Instance(0)->var_resource_->UpdateDevVarMgrInfo(0);

  uint8_t* logic_addr = nullptr;
  status = VarManager::Instance(0)->GetVarAddr(var_name, tensor_desc, logic_addr);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_NE(logic_addr, nullptr);

  uint8_t* dev_addr = VarManager::Instance(0)->GetVarMemoryAddr(var_name, logic_addr, RT_MEMORY_HBM, 0);

  EXPECT_EQ(dev_addr, nullptr);
}

TEST_F(UtestGraphVarManagerTest, test_init_var_if_has_init_value_size_exceeds) {

  const std::vector<rtMemType_t> memory_types({RT_MEMORY_HBM, RT_MEMORY_P2P_DDR});
  EXPECT_EQ(MemManager::Instance().Initialize(memory_types), SUCCESS);

  VarManager::Instance(0)->SetMemManager(&MemManager::Instance());
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);

  std::vector<int64_t> var_shape{8, 1, 64, 64, 16};
  GeShape shape(var_shape);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetFormat(FORMAT_NCHW);
  TensorUtils::SetSize(tensor_desc, 10 * sizeof(float)); 

  std::vector<float> oversized_data(10000000, 1.0f);  
  auto init_tensor = std::make_shared<GeTensor>();
  GeTensorDesc init_desc(GeShape(var_shape), FORMAT_NCHW, DT_FLOAT);
  init_tensor->SetData(reinterpret_cast<uint8_t*>(oversized_data.data()), oversized_data.size() * sizeof(float));
  init_tensor->MutableTensorDesc() = init_desc;

  EXPECT_TRUE(ge::AttrUtils::SetTensor(&tensor_desc, ATTR_NAME_INIT_VALUE, init_tensor));

  OpDescPtr op_desc = std::make_shared<OpDesc>("test_var_size_exceeds", VARIABLE);
  op_desc->AddOutputDesc(tensor_desc);

  std::string var_name = "test_var_size_exceeds";
  Status status = VarManager::Instance(0)->AssignVarMem(var_name, op_desc, tensor_desc, RT_MEMORY_HBM);
  EXPECT_EQ(status, SUCCESS);

  VarManager::Instance(0)->var_resource_->UpdateDevVarMgrInfo(0);

  uint8_t* logic_addr = nullptr;
  status = VarManager::Instance(0)->GetVarAddr(var_name, tensor_desc, logic_addr);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_NE(logic_addr, nullptr);

  uint8_t* dev_addr = VarManager::Instance(0)->GetVarMemoryAddr(var_name, logic_addr, RT_MEMORY_HBM, 0);

  EXPECT_EQ(dev_addr, nullptr);
}
} // namespace ge

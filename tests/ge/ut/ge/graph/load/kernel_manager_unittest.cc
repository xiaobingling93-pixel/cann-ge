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
#include "graph/load/model_manager/kernel/model_kernel_handles_manager.h"
#include "common/kernel_handles_manager/kernel_handle_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/op_kernel_bin.h"
#include "framework/common/types.h"
#include "stub/runtime_stub_for_kernel_v2.h"
#include "graph/load/model_manager/model_manager.h"

#define private public
#include "common/kernel_handles_manager/aicore_kernel_handles_manager.h"
#include "common/kernel_handles_manager/aicpu_kernel_handles_manager.h"
#include "common/kernel_handles_manager/cust_aicpu_kernel_handles_manager.h"
#undef private

namespace ge {
class KernelManagerUtest : public testing::Test {
 protected:
  void SetUp() {
    auto mock_runtime = std::make_shared<gert::RuntimeStubForKernelV2>();
    ge::RuntimeStub::Install(mock_runtime.get());
  }
  void TearDown() {
    RuntimeStub::UnInstall(nullptr);
  }
};

TEST_F(KernelManagerUtest, test_aicore_kernel_register) {
  ModelKernelHandlesManager kernel_manager;
  auto aicore_kernel_handle = kernel_manager.GetKernelHandle(KernelHandleType::kAicore);
  EXPECT_NE(aicore_kernel_handle, nullptr);
  const char relu_bin[] = "tbe_relu_kernel_bin_001";
  vector<char> buffer(relu_bin, relu_bin + strlen(relu_bin));
  OpKernelBinPtr relu_tbe_kernel_ptr = std::make_shared<OpKernelBin>("te_relu_123456", std::move(buffer));
  AicoreRegisterInfo aicore_register_info = {"te_relu_123456", RT_DEV_BINARY_MAGIC_ELF_AIVEC, relu_tbe_kernel_ptr};
  KernelRegisterInfo register_info = aicore_register_info;
  std::string aicore_key = aicore_kernel_handle->GenerateKey(register_info);
  EXPECT_EQ(aicore_key, "te_relu_123456_AicoreKernel");
  EXPECT_NE(aicore_kernel_handle->GetOrRegisterKernel(register_info, aicore_key), nullptr);
  EXPECT_EQ(aicore_kernel_handle->local_refer_cnt_.size(), 1);
  EXPECT_EQ(aicore_kernel_handle->local_refer_cnt_[aicore_key], 1);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_.size(), 1);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_[aicore_key].refer_cnt, 1);
  EXPECT_NE(aicore_kernel_handle->GetOrRegisterKernel(register_info, aicore_key), nullptr);
  EXPECT_EQ(aicore_kernel_handle->local_refer_cnt_.size(), 1);
  EXPECT_EQ(aicore_kernel_handle->local_refer_cnt_[aicore_key], 2);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_.size(), 1);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_[aicore_key].refer_cnt, 2);

  // 不同模型中的handle注册
  ModelKernelHandlesManager kernel_manager_2;
  auto aicore_kernel_handle_2 = kernel_manager_2.GetKernelHandle(KernelHandleType::kAicore);
  std::string aicore_key_2 = aicore_kernel_handle->GenerateKey(register_info);
  EXPECT_EQ(aicore_key_2, aicore_key);
  EXPECT_NE(aicore_kernel_handle_2->GetOrRegisterKernel(register_info, aicore_key_2), nullptr);
  EXPECT_EQ(aicore_kernel_handle_2->local_refer_cnt_.size(), 1);
  EXPECT_EQ(aicore_kernel_handle_2->local_refer_cnt_[aicore_key], 1);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_.size(), 1);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_[aicore_key].refer_cnt, 3);
  auto bin_handle = aicore_kernel_handle_2->GetOrRegisterKernel(register_info, aicore_key);
  EXPECT_EQ(aicore_kernel_handle_2->local_refer_cnt_.size(), 1);
  EXPECT_EQ(aicore_kernel_handle_2->local_refer_cnt_[aicore_key], 2);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_.size(), 1);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_[aicore_key].refer_cnt, 4);
  EXPECT_NE(bin_handle, nullptr);
  auto func_handle = KernelHandleUtils::GetFuncHandle(bin_handle, "te_relu_123456");
  EXPECT_NE(func_handle, nullptr);
  // 卸载一个模型
  EXPECT_EQ(kernel_manager.ClearAllHandle(), SUCCESS);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_.size(), 1);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_[aicore_key].refer_cnt, 2);
  EXPECT_EQ(kernel_manager_2.ClearAllHandle(), SUCCESS);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_.size(), 0);
}

TEST_F(KernelManagerUtest, test_aicore_kernel_with_atomic_register_and_get_func) {
  ModelKernelHandlesManager kernel_manager;
  auto aicore_kernel_handle = kernel_manager.GetKernelHandle(KernelHandleType::kAicore);
  EXPECT_NE(aicore_kernel_handle, nullptr);
  const char relu_bin[] = "tbe_relu_kernel_bin_001";
  vector<char> buffer(relu_bin, relu_bin + strlen(relu_bin));
  OpKernelBinPtr relu_tbe_kernel_ptr = std::make_shared<OpKernelBin>("te_relu_123456", std::move(buffer));
  AicoreRegisterInfo aicore_register_info = {"te_relu_123456", RT_DEV_BINARY_MAGIC_ELF_AICUBE, relu_tbe_kernel_ptr};
  KernelRegisterInfo aic_register_info = aicore_register_info;

  const char memset_bin[] = "memset_kernel_bin_001";
  vector<char> memset_buffer(memset_bin, memset_bin + strlen(memset_bin));
  OpKernelBinPtr memset_tbe_kernel_ptr = std::make_shared<OpKernelBin>("te_memset_123456", std::move(memset_buffer));
  AicoreRegisterInfo memset_aicore_register_info =
      {"te_memset_123456", RT_DEV_BINARY_MAGIC_ELF_AIVEC, memset_tbe_kernel_ptr};
  KernelRegisterInfo memset_register_info = memset_aicore_register_info;

  std::string aicore_key = aicore_kernel_handle->GenerateKey(aic_register_info);
  EXPECT_EQ(aicore_key, "te_relu_123456_AicoreKernel");
  EXPECT_NE(aicore_kernel_handle->GetOrRegisterKernel(aic_register_info, aicore_key), nullptr);
  EXPECT_EQ(aicore_kernel_handle->local_refer_cnt_.size(), 1);
  EXPECT_EQ(aicore_kernel_handle->local_refer_cnt_[aicore_key], 1);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_.size(), 1);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_[aicore_key].refer_cnt, 1);
  std::string atomic_key = aicore_kernel_handle->GenerateKey(memset_register_info);
  EXPECT_EQ(atomic_key, "te_memset_123456_AicoreKernel");
  EXPECT_NE(aicore_kernel_handle->GetOrRegisterKernel(memset_register_info, atomic_key), nullptr);
  EXPECT_EQ(aicore_kernel_handle->local_refer_cnt_.size(), 2);
  EXPECT_EQ(aicore_kernel_handle->local_refer_cnt_[aicore_key], 1);
  EXPECT_EQ(aicore_kernel_handle->local_refer_cnt_[atomic_key], 1);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_.size(), 2);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_[aicore_key].refer_cnt, 1);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_[atomic_key].refer_cnt, 1);
  auto aicore_bin_handle = aicore_kernel_handle->GetOrRegisterKernel(aic_register_info, aicore_key);
  EXPECT_NE(aicore_bin_handle, nullptr);
  EXPECT_EQ(aicore_kernel_handle->local_refer_cnt_.size(), 2);
  EXPECT_EQ(aicore_kernel_handle->local_refer_cnt_[aicore_key], 2);
  EXPECT_EQ(aicore_kernel_handle->local_refer_cnt_[atomic_key], 1);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_.size(), 2);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_[aicore_key].refer_cnt, 2);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_[atomic_key].refer_cnt, 1);
  auto aicore_func_handle = KernelHandleUtils::GetFuncHandle(aicore_bin_handle, 0);
  EXPECT_NE(aicore_func_handle, nullptr);
  auto atomic_bin_handle = aicore_kernel_handle->GetOrRegisterKernel(memset_register_info, atomic_key);
  EXPECT_NE(atomic_bin_handle, nullptr);
  EXPECT_EQ(aicore_kernel_handle->local_refer_cnt_.size(), 2);
  EXPECT_EQ(aicore_kernel_handle->local_refer_cnt_[aicore_key], 2);
  EXPECT_EQ(aicore_kernel_handle->local_refer_cnt_[atomic_key], 2);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_.size(), 2);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_[aicore_key].refer_cnt, 2);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_[atomic_key].refer_cnt, 2);
  auto atomic_func_handle = KernelHandleUtils::GetFuncHandle(atomic_bin_handle, "te_memset_123456");
  EXPECT_NE(atomic_func_handle, nullptr);
  EXPECT_EQ(kernel_manager.ClearAllHandle(), SUCCESS);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_.size(), 0);
}

TEST_F(KernelManagerUtest, test_aicpu_kernel_register_and_get_func) {
  ModelKernelHandlesManager kernel_manager;
  auto aicpu_kernel_handle = kernel_manager.GetKernelHandle(KernelHandleType::kAicpu);
  EXPECT_NE(aicpu_kernel_handle, nullptr);

  AicpuRegisterInfo cpu_aicpu_register_info = {"Relu", "libcpu_kernels.so", "RunCpuKernels", "AICPUKernels"};
  KernelRegisterInfo cpu_register_info = cpu_aicpu_register_info;

  AicpuRegisterInfo tf_aicpu_register_info = {"Add", "libtf_kernels.so", "TfOperatorAPI", "TFKernels"};
  KernelRegisterInfo tf_register_info = tf_aicpu_register_info;

  AicpuRegisterInfo kfc_aicpu_register_info = {"MC2", "libccl_kernels.so", "RunAicpuKfcSrvLaunch", "KFCKernels"};
  KernelRegisterInfo kfc_register_info = kfc_aicpu_register_info;

  std::string aicpu_key = aicpu_kernel_handle->GenerateKey(cpu_register_info);
  EXPECT_EQ(aicpu_key, "Relu_libcpu_kernels.so_AicpuKernel");
  std::string tf_key = aicpu_kernel_handle->GenerateKey(tf_register_info);
  EXPECT_EQ(tf_key, "Add_libtf_kernels.so_AicpuKernel");
  std::string kfc_key = aicpu_kernel_handle->GenerateKey(kfc_register_info);
  EXPECT_EQ(kfc_key, "MC2_libccl_kernels.so_AicpuKernel");

  EXPECT_NE(aicpu_kernel_handle->GetOrRegisterKernel(cpu_register_info, aicpu_key), nullptr);
  EXPECT_NE(aicpu_kernel_handle->GetOrRegisterKernel(tf_register_info, tf_key), nullptr);
  EXPECT_NE(aicpu_kernel_handle->GetOrRegisterKernel(kfc_register_info, kfc_key), nullptr);
  EXPECT_EQ(aicpu_kernel_handle->local_refer_cnt_.size(), 3);
  EXPECT_EQ(aicpu_kernel_handle->local_refer_cnt_[aicpu_key], 1);
  EXPECT_EQ(aicpu_kernel_handle->local_refer_cnt_[tf_key], 1);
  EXPECT_EQ(aicpu_kernel_handle->local_refer_cnt_[kfc_key], 1);

  auto aicpu_bin_handle = aicpu_kernel_handle->GetOrRegisterKernel(cpu_register_info, aicpu_key);
  EXPECT_NE(aicpu_bin_handle, nullptr);
  auto aicpu_func_handle = KernelHandleUtils::GetFuncHandle(aicpu_bin_handle, "Relu");
  EXPECT_NE(aicpu_func_handle, nullptr);
  auto tf_bin_handle = aicpu_kernel_handle->GetOrRegisterKernel(tf_register_info, tf_key);
  EXPECT_NE(tf_bin_handle, nullptr);
  auto tf_func_handle = KernelHandleUtils::GetFuncHandle(tf_bin_handle, "Add");
  EXPECT_NE(tf_func_handle, nullptr);
  auto kfc_bin_handle = aicpu_kernel_handle->GetOrRegisterKernel(kfc_register_info, kfc_key);
  EXPECT_NE(kfc_bin_handle, nullptr);
  auto kfc_func_handle = KernelHandleUtils::GetFuncHandle(kfc_bin_handle, "MC2");
  EXPECT_NE(kfc_func_handle, nullptr);

  EXPECT_EQ(kernel_manager.ClearAllHandle(), SUCCESS);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_.size(), 0);
}

TEST_F(KernelManagerUtest, test_cust_aicpu_kernel_register_and_get_func) {
  ModelKernelHandlesManager kernel_manager;
  auto cust_aicpu_kernel_handle = kernel_manager.GetKernelHandle(KernelHandleType::kCustAicpu);
  EXPECT_NE(cust_aicpu_kernel_handle, nullptr);

  const char relu_bin[] = "cust_aicpu_relu_kernel_bin_001";
  vector<char> buffer(relu_bin, relu_bin + strlen(relu_bin));
  const auto kernel_handle = std::make_shared<OpKernelBin>("Relu", std::move(buffer));
  CustAicpuRegisterInfo cust_aicpu_register_info0 = {kernel_handle};
  KernelRegisterInfo cust_cpu_register_info0 = cust_aicpu_register_info0;

  CustAicpuRegisterInfo cust_aicpu_register_info1 = {kernel_handle};
  KernelRegisterInfo cust_cpu_register_info1 = cust_aicpu_register_info1;

  vector<char> buffer2(relu_bin, relu_bin + strlen(relu_bin));
  const auto kernel_handle1 = std::make_shared<OpKernelBin>("Relu", std::move(buffer2));
  CustAicpuRegisterInfo cust_aicpu_register_info2 = {kernel_handle1};
  KernelRegisterInfo cust_cpu_register_info2 = cust_aicpu_register_info2;

  const char add_bin[] = "cust_aicpu_add_kernel_bin_001";
  vector<char> buffer3(add_bin, add_bin + strlen(add_bin));
  const auto kernel_handle2 = std::make_shared<OpKernelBin>("Add", std::move(buffer));
  CustAicpuRegisterInfo cust_aicpu_register_info3 = {kernel_handle2};
  KernelRegisterInfo cust_cpu_register_info3 = cust_aicpu_register_info3;

  std::string cust_aicpu_key0 = cust_aicpu_kernel_handle->GenerateKey(cust_cpu_register_info0);
  std::string cust_aicpu_key1 = cust_aicpu_kernel_handle->GenerateKey(cust_cpu_register_info1);
  EXPECT_EQ(cust_aicpu_key0, cust_aicpu_key1);
  std::string cust_aicpu_key2 = cust_aicpu_kernel_handle->GenerateKey(cust_cpu_register_info2);
  EXPECT_EQ(cust_aicpu_key1, cust_aicpu_key2);
  std::string cust_aicpu_key3 = cust_aicpu_kernel_handle->GenerateKey(cust_cpu_register_info3);
  EXPECT_NE(cust_aicpu_key3, cust_aicpu_key0);

  EXPECT_NE(cust_aicpu_kernel_handle->GetOrRegisterKernel(cust_cpu_register_info0, cust_aicpu_key0), nullptr);
  EXPECT_NE(cust_aicpu_kernel_handle->GetOrRegisterKernel(cust_cpu_register_info1, cust_aicpu_key1), nullptr);
  EXPECT_NE(cust_aicpu_kernel_handle->GetOrRegisterKernel(cust_cpu_register_info2, cust_aicpu_key2), nullptr);
  EXPECT_NE(cust_aicpu_kernel_handle->GetOrRegisterKernel(cust_cpu_register_info3, cust_aicpu_key3), nullptr);
  EXPECT_EQ(cust_aicpu_kernel_handle->local_refer_cnt_.size(), 2);
  EXPECT_EQ(cust_aicpu_kernel_handle->local_refer_cnt_[cust_aicpu_key0], 3);
  EXPECT_EQ(cust_aicpu_kernel_handle->local_refer_cnt_[cust_aicpu_key3], 1);

  auto cust_aicpu_bin_handle0 = cust_aicpu_kernel_handle->GetOrRegisterKernel(cust_cpu_register_info0, cust_aicpu_key0);
  EXPECT_NE(cust_aicpu_bin_handle0, nullptr);
  auto cust_aicpu_func_handle0 = KernelHandleUtils::GetCustAicpuFuncHandle(cust_aicpu_bin_handle0,
      "Relu", "RunCpuKernels");
  EXPECT_NE(cust_aicpu_func_handle0, nullptr);
  auto cust_aicpu_bin_handle1 = cust_aicpu_kernel_handle->GetOrRegisterKernel(cust_cpu_register_info1, cust_aicpu_key1);
  EXPECT_NE(cust_aicpu_bin_handle1, nullptr);
  auto cust_aicpu_func_handle1 = KernelHandleUtils::GetCustAicpuFuncHandle(cust_aicpu_bin_handle1,
      "Relu", "RunCpuKernels");
  EXPECT_NE(cust_aicpu_func_handle1, nullptr);
  auto cust_aicpu_bin_handle2 = cust_aicpu_kernel_handle->GetOrRegisterKernel(cust_cpu_register_info2, cust_aicpu_key2);
  EXPECT_NE(cust_aicpu_bin_handle2, nullptr);
  auto cust_aicpu_func_handle2 = KernelHandleUtils::GetCustAicpuFuncHandle(cust_aicpu_bin_handle2,
      "Relu", "RunCpuKernels");
  EXPECT_NE(cust_aicpu_func_handle2, nullptr);
  auto cust_aicpu_bin_handle3 = cust_aicpu_kernel_handle->GetOrRegisterKernel(cust_cpu_register_info3, cust_aicpu_key3);
  EXPECT_NE(cust_aicpu_bin_handle3, nullptr);
  auto cust_aicpu_func_handle3 = KernelHandleUtils::GetCustAicpuFuncHandle(cust_aicpu_bin_handle3,
      "Add", "RunCpuKernels");
  EXPECT_NE(cust_aicpu_func_handle3, nullptr);
  EXPECT_EQ(kernel_manager.ClearAllHandle(), SUCCESS);
  EXPECT_EQ(KernelHandlesManager::global_bin_store_.size(), 0);
}

TEST_F(KernelManagerUtest, test_launch_kernel) {
  LaunchKernelParam launch_param;
  LaunchKernelConfig launch_config;
  launch_config.local_memory_size = 10;
  launch_config.block_dim_offset  = 20;
  launch_param.launch_config = launch_config;
  launch_param.block_dim = 32;
  launch_param.stream = (void *)0x1200;
  launch_param.args = (void *)0x1300;
  launch_param.args_size = 128;
  rtFuncHandle func_handle = (void *)0x1400;
  EXPECT_EQ(KernelHandleUtils::LaunchKernel(func_handle, launch_param), SUCCESS);
  launch_param.is_host_args = true;
  EXPECT_EQ(KernelHandleUtils::LaunchKernel(func_handle, launch_param), SUCCESS);
}
}
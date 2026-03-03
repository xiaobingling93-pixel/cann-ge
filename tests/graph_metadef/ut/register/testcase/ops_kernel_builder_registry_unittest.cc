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

#include "register/ops_kernel_builder_registry.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
class UtestOpsKernelBuilderRegistry : public testing::Test { 
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestOpsKernelBuilderRegistry, GetAllKernelBuildersTest) {
    ge::OpsKernelBuilderRegistry ops_registry;
    OpsKernelBuilderPtr opsptr = std::shared_ptr<OpsKernelBuilder>();
    ops_registry.kernel_builders_.insert(pair<std::string, OpsKernelBuilderPtr>("ops1", opsptr));
    std::map<std::string, OpsKernelBuilderPtr> ops_map;
    ops_map = ops_registry.GetAll();
    EXPECT_EQ(ops_map.size(), 1);
    EXPECT_EQ(ops_map["ops1"], opsptr);
}

TEST_F(UtestOpsKernelBuilderRegistry, RegisterTest) {
    ge::OpsKernelBuilderRegistry ops_registry;
    std::string name = "register1";
    OpsKernelBuilderPtr instance = std::shared_ptr<OpsKernelBuilder>();
    ops_registry.Register(name, instance);
    std::map<std::string, OpsKernelBuilderPtr> kernel_builders_;
    kernel_builders_ = ops_registry.GetAll();
    EXPECT_EQ(kernel_builders_.size(), 1);
    EXPECT_EQ(kernel_builders_["register1"], instance);
}

TEST_F(UtestOpsKernelBuilderRegistry, UnregisterTest) {
    ge::OpsKernelBuilderRegistry ops_registry;
    std::string name1 = "register1";
    OpsKernelBuilderPtr opsPtr1 = std::shared_ptr<OpsKernelBuilder>();
    ops_registry.Register(name1, opsPtr1);

    std::string name2 = "register2";
    OpsKernelBuilderPtr opsPtr2 = std::shared_ptr<OpsKernelBuilder>();
    ops_registry.Register(name2, opsPtr2);

    std::map<std::string, OpsKernelBuilderPtr> ops_map;
    ops_map = ops_registry.GetAll();
    EXPECT_EQ(ops_map.size(), 2);
    EXPECT_EQ(ops_map["register1"], opsPtr1);
    EXPECT_EQ(ops_map["register2"], opsPtr2);

    ops_registry.Unregister("register1");
    ops_map = ops_registry.GetAll();
    EXPECT_EQ(ops_map.size(), 1);
    EXPECT_EQ(ops_map.count("register1"), 0);
}

TEST_F(UtestOpsKernelBuilderRegistry, UnregisterAllTest) {
    ge::OpsKernelBuilderRegistry ops_registry;
    std::string name1 = "register1";
    OpsKernelBuilderPtr opsPtr1 = std::shared_ptr<OpsKernelBuilder>();
    ops_registry.Register(name1, opsPtr1);

    std::string name2 = "register2";
    OpsKernelBuilderPtr opsPtr2 = std::shared_ptr<OpsKernelBuilder>();
    ops_registry.Register(name2, opsPtr2);

    std::map<std::string, OpsKernelBuilderPtr> ops_map;
    ops_map = ops_registry.GetAll();
    EXPECT_EQ(ops_map.size(), 2);
    EXPECT_EQ(ops_map["register1"], opsPtr1);
    EXPECT_EQ(ops_map["register2"], opsPtr2);

    ops_registry.UnregisterAll();
    ops_map = ops_registry.GetAll();
    EXPECT_EQ(ops_map.size(), 0);
    EXPECT_EQ(ops_map.count("register1"), 0);
    EXPECT_EQ(ops_map.count("register2"), 0);
}

TEST_F(UtestOpsKernelBuilderRegistry, OpsKernelBuilderRegistrarTest) {
    std::string name = "register";
    ge::OpsKernelBuilderRegistrar::CreateFn fn = nullptr;
    OpsKernelBuilderRegistrar ops_rar(name, fn);
    std::map<std::string, OpsKernelBuilderPtr> ops_map;
    ops_map = OpsKernelBuilderRegistry::GetInstance().GetAll();
    EXPECT_EQ(ops_map.size(), 1);
}

} // namespace ge

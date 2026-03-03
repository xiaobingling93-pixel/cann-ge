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
#include <climits>
#include <fstream>

#include "register/register_custom_pass.h"
#include "register/custom_pass_context_impl.h"
#include "framework/common/debug/ge_log.h"
#include "register/custom_pass_helper.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "graph/ge_local_context.h"

namespace ge {
namespace {
  const char *const kEnvName = "ASCEND_OPP_PATH";
}
class UtestRegisterPass : public testing::Test { 
 protected:
  void SetUp() {}
  void TearDown() {}

  void CreateSharedLibrary(const std::string &path) {
    std::ofstream ofs(path + ".cpp");
    ofs << R"(
      #include <iostream>
      extern "C" void hello() {
        std::cout << "Hello, world!" << std::endl;
      }
    )";
    ofs.close();
    std::string cmd = "g++ -shared -fPIC -o " + path + ".so " + path + ".cpp";
    system(cmd.c_str());
    std::remove((path + ".cpp").c_str());
  }

  static Status MyCustomPass(ge::GraphPtr &graph, CustomPassContext &context) {
    if (graph->GetName() == "test") {
      context.SetErrorMessage("graph name is invalid");
      return FAILED;
    }
    return SUCCESS;
  }

  static Status FooConstGraphCustomPass(const ConstGraphPtr &graph, CustomPassContext &context) {
    if (graph->GetName() == "error_graph") {
      context.SetErrorMessage("graph name is invalid");
      return FAILED;
    }
    return SUCCESS;
  }
};

TEST_F(UtestRegisterPass, GetPassNameTest) {
  ge::PassRegistrationData pass_data("registry");
  std::string name = pass_data.GetPassName();
  EXPECT_EQ(name, "registry");

  pass_data.impl_ = nullptr;
  name = pass_data.GetPassName();
  EXPECT_EQ(name, "");
}

TEST_F(UtestRegisterPass, CustomPassFnTest) {
  CustomPassFunc custom_pass_fn = nullptr;
  ge::PassRegistrationData pass_data("registry");
  pass_data.CustomPassFn(custom_pass_fn);
  auto ret = pass_data.GetCustomPassFn();
  EXPECT_EQ(ret, nullptr);

  custom_pass_fn = std::function<Status(ge::GraphPtr &, CustomPassContext &)>();
  pass_data.impl_ = nullptr;
  pass_data.CustomPassFn(custom_pass_fn);
  ret = pass_data.GetCustomPassFn();
  EXPECT_EQ(ret, nullptr);
}

TEST_F(UtestRegisterPass, CustomPassHelperRunTest) {
  PassRegistrationData pass_data("registry");
  ge::PassReceiver pass_receiver(pass_data);
  CustomPassHelper cust_helper;
  auto graph = std::make_shared<Graph>("test");
  auto custom_pass_context = CustomPassContext();

  bool ret = cust_helper.Run(graph, custom_pass_context);
  EXPECT_EQ(ret, SUCCESS);

  // not register pass func
  PassRegistrationData pass_data2("registry2");
  cust_helper.registration_datas_.emplace_back(pass_data2);
  auto graph2 = std::make_shared<Graph>("test2");
  ret = cust_helper.Run(graph2, custom_pass_context);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestRegisterPass, CustomPassHelperRunTest_Failed) {
  CustomPassHelper cust_helper;
  auto custom_pass_context = CustomPassContext();

  PassRegistrationData pass_data2("registry2");
  pass_data2.CustomPassFn(MyCustomPass);
  cust_helper.registration_datas_.emplace_back(pass_data2);
  auto graph = std::make_shared<Graph>("test");
  auto ret = cust_helper.Run(graph, custom_pass_context);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(UtestRegisterPass, CustomPassHelperRunTest_Success) {
  CustomPassHelper cust_helper;
  auto custom_pass_context = CustomPassContext();

  PassRegistrationData pass_data2("registry2");
  pass_data2.CustomPassFn(MyCustomPass);
  cust_helper.registration_datas_.emplace_back(pass_data2);
  auto graph = std::make_shared<Graph>("test2");
  auto ret = cust_helper.Run(graph, custom_pass_context);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestRegisterPass, LoadCustomPassLibsTest_Failed) {
  CustomPassHelper cust_helper;
  ge::Status status = cust_helper.Load();
  EXPECT_EQ(status, ge::SUCCESS);
  status = cust_helper.Unload();
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestRegisterPass, LoadCustomPassLibsTest_Failed_Invalid_Lib) {
  std::string path = __FILE__;
  path = path.substr(0, path.rfind("/") + 1) + "opp";
  mmSetEnv(kEnvName, path.c_str(), 1);
  system(("mkdir -p " + path).c_str());

  std::string custom_path = path + "/vendors/1/custom_fusion_passes";
  system(("mkdir -p " + custom_path).c_str());
  system(("touch " + custom_path + "/concat_pass.so").c_str());
  system(("touch " + custom_path + "/tile_pass.so").c_str());
  system(("touch " + custom_path + "/add_pass.so").c_str());

  CustomPassHelper cust_helper;
  ge::Status status = cust_helper.Load();
  EXPECT_EQ(status, ge::FAILED);
  status = cust_helper.Unload();
  EXPECT_EQ(status, ge::SUCCESS);

  system(("rm -rf " + path).c_str());
}

TEST_F(UtestRegisterPass, LoadCustomPassLibsTest_MissingDependencies) {
  std::string path = __FILE__;
  path = path.substr(0, path.rfind("/") + 1) + "opp";
  mmSetEnv(kEnvName, path.c_str(), 1);
  system(("mkdir -p " + path).c_str());

  std::string custom_path = path + "/vendors/1/custom_fusion_passes";
  system(("mkdir -p " + custom_path).c_str());

  // Create a shared library that depends on a dummy library
  std::ofstream dummy_lib(custom_path + "/libdummy.cpp");
  dummy_lib << R"(
    #include <iostream>
    extern "C" void dummy() {
      std::cout << "Dummy function" << std::endl;
    }
  )";
  dummy_lib.close();
  std::string dummy_cmd = "g++ -shared -fPIC -o " + custom_path + "/libdummy.so " + custom_path + "/libdummy.cpp";
  system(dummy_cmd.c_str());
  std::remove((custom_path + "/libdummy.cpp").c_str());

  // Create the main shared library that depends on the dummy library
  std::ofstream main_lib(custom_path + "/libcustom_pass.cpp");
  main_lib << R"(
    #include <iostream>
    extern void dummy();
    extern "C" void hello() {
      dummy();
      std::cout << "Hello, world!" << std::endl;
    }
  )";
  main_lib.close();
  std::string main_cmd = "g++ -shared -fPIC -o " + custom_path + "/libcustom_pass.so " + custom_path + "/libcustom_pass.cpp -L" + custom_path + " -ldummy";
  system(main_cmd.c_str());
  std::remove((custom_path + "/libcustom_pass.cpp").c_str());

  // Ensure the shared library is created
  struct stat buffer;
  ASSERT_EQ(stat((custom_path + "/libcustom_pass.so").c_str(), &buffer), 0);

  // Remove the dummy library to simulate missing dependency
  system(("rm " + custom_path + "/libdummy.so").c_str());

  // Call the function under test
  CustomPassHelper cust_helper;
  ge::Status status = cust_helper.Load();
  EXPECT_EQ(status, ge::FAILED);

  system(("rm -rf " + path).c_str());
}

TEST_F(UtestRegisterPass, LoadCustomPassLibsTest_Success) {
  std::string path = __FILE__;
  path = path.substr(0, path.rfind("/") + 1) + "opp";
  mmSetEnv(kEnvName, path.c_str(), 1);
  system(("mkdir -p " + path).c_str());

  std::string custom_path = path + "/vendors/1/custom_fusion_passes/add";
  system(("mkdir -p " + custom_path).c_str());

  CreateSharedLibrary(custom_path);

  // Call the function under test
  CustomPassHelper cust_helper;
  ge::Status status = cust_helper.Load();
  EXPECT_EQ(status, ge::SUCCESS);
  status = cust_helper.Unload();
  EXPECT_EQ(status, ge::SUCCESS);

  system(("rm -rf " + path).c_str());
}

TEST_F(UtestRegisterPass, CustomPassStage_Success) {
  PassRegistrationData pass_reg_data("custom_pass");
  pass_reg_data.Stage(CustomPassStage::kAfterInferShape);
  EXPECT_EQ(pass_reg_data.GetStage(), CustomPassStage::kAfterInferShape);
}

TEST_F(UtestRegisterPass, CustomPassStage_AndRun_Success) {
  PassRegistrationData pass_reg_data("custom_pass");
  pass_reg_data.CustomPassFn(MyCustomPass).Stage(CustomPassStage::kAfterInferShape);
  CustomPassHelper::Instance().Unload();
  CustomPassHelper::Instance().Insert(pass_reg_data);
  auto graph = std::make_shared<Graph>("test2");
  auto custom_pass_context = CustomPassContext();

  EXPECT_EQ(CustomPassHelper::Instance().Run(graph, custom_pass_context), SUCCESS);
  EXPECT_EQ(pass_reg_data.GetStage(), CustomPassStage::kAfterInferShape);
}

TEST_F(UtestRegisterPass, CustomPassStage_Failed) {
  PassRegistrationData pass_reg_data;
  pass_reg_data.Stage(CustomPassStage::kAfterInferShape);
  EXPECT_EQ(pass_reg_data.GetStage(), CustomPassStage::kInvalid);
}

TEST_F(UtestRegisterPass, ConstGraphCustomPass_AndRun_SUCCESS) {
  PassRegistrationData pass_reg_data("custom_pass");
  pass_reg_data.CustomPassFn(FooConstGraphCustomPass).Stage(CustomPassStage::kAfterAssignLogicStream);
  CustomPassHelper::Instance().Unload();
  CustomPassHelper::Instance().Insert(pass_reg_data);
  auto graph = std::make_shared<Graph>("test2");
  auto custom_pass_context = CustomPassContext();

  EXPECT_NE(CustomPassHelper::Instance().Run(graph, custom_pass_context, CustomPassStage::kAfterAssignLogicStream), SUCCESS);
  EXPECT_EQ(pass_reg_data.GetStage(), CustomPassStage::kAfterAssignLogicStream);
}

TEST_F(UtestRegisterPass, ConstGraphCustomPass_AndRun_Failed_RegisterWrongFunc) {
  PassRegistrationData pass_reg_data("custom_pass");
  // wrong func in kAfterAssignLogicStream stage
  pass_reg_data.CustomPassFn(MyCustomPass).Stage(CustomPassStage::kAfterAssignLogicStream);
  CustomPassHelper::Instance().Unload();
  CustomPassHelper::Instance().Insert(pass_reg_data);
  auto graph = std::make_shared<Graph>("test2");
  auto custom_pass_context = CustomPassContext();

  EXPECT_NE(CustomPassHelper::Instance().Run(graph, custom_pass_context, CustomPassStage::kAfterAssignLogicStream), SUCCESS);
  EXPECT_EQ(pass_reg_data.GetStage(), CustomPassStage::kAfterAssignLogicStream);
}

TEST_F(UtestRegisterPass, ConstGraphCustomPass_AndRun_Failed_FuncReturnError) {
  PassRegistrationData pass_reg_data("custom_pass");
  pass_reg_data.CustomPassFn(FooConstGraphCustomPass).Stage(CustomPassStage::kAfterAssignLogicStream);
  CustomPassHelper::Instance().Unload();
  CustomPassHelper::Instance().Insert(pass_reg_data);
  auto graph = std::make_shared<Graph>("error_graph");
  auto custom_pass_context = CustomPassContext();
  
  EXPECT_NE(CustomPassHelper::Instance().Run(graph, custom_pass_context, CustomPassStage::kAfterAssignLogicStream), SUCCESS);
  EXPECT_EQ(pass_reg_data.GetStage(), CustomPassStage::kAfterAssignLogicStream);
}

TEST_F(UtestRegisterPass, ConstGraph_AfterBuiltinFusionCustomPass_AndRun_SUCCESS) {
  PassRegistrationData pass_reg_data("custom_pass");
  pass_reg_data.CustomPassFn(FooConstGraphCustomPass).Stage(CustomPassStage::kAfterBuiltinFusionPass);
  CustomPassHelper::Instance().Unload();
  CustomPassHelper::Instance().Insert(pass_reg_data);
  auto graph = std::make_shared<Graph>("test2");
  auto custom_pass_context = CustomPassContext();

  EXPECT_EQ(CustomPassHelper::Instance().Run(graph, custom_pass_context, CustomPassStage::kAfterBuiltinFusionPass), SUCCESS);
  EXPECT_EQ(pass_reg_data.GetStage(), CustomPassStage::kAfterBuiltinFusionPass);
}

TEST_F(UtestRegisterPass, ConstGraph_AfterBuiltinFusionCustomPass_AndRun_Failed_FuncReturnError) {
  PassRegistrationData pass_reg_data("custom_pass");
  pass_reg_data.CustomPassFn(FooConstGraphCustomPass).Stage(CustomPassStage::kAfterBuiltinFusionPass);
  CustomPassHelper::Instance().Unload();
  CustomPassHelper::Instance().Insert(pass_reg_data);
  auto graph = std::make_shared<Graph>("error_graph");
  auto custom_pass_context = CustomPassContext();

  EXPECT_NE(CustomPassHelper::Instance().Run(graph, custom_pass_context, CustomPassStage::kAfterBuiltinFusionPass), SUCCESS);
  EXPECT_EQ(pass_reg_data.GetStage(), CustomPassStage::kAfterBuiltinFusionPass);
}

TEST_F(UtestRegisterPass, CustomPassContext_GetOptionValue) {
  std::map<std::string, std::string> options_map = {{ge::OPTION_GRAPH_RUN_MODE, "train"}};
  auto option_bak = GetThreadLocalContext().GetAllGraphOptions();
  GetThreadLocalContext().SetGraphOption(options_map);
  auto custom_pass = CustomPassContext();
  AscendString graph_run_mode;
  auto ret = custom_pass.GetOptionValue(ge::OPTION_GRAPH_RUN_MODE, graph_run_mode);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_STREQ(graph_run_mode.GetString(), "train");
  GetThreadLocalContext().SetGraphOption(option_bak);
}

TEST_F(UtestRegisterPass, CustomPassContext_GetOptionValue_Failed) {
  auto custom_pass = CustomPassContext();
  AscendString graph_run_mode;
  auto ret = custom_pass.GetOptionValue("not_exist", graph_run_mode);
  EXPECT_NE(ret, GRAPH_SUCCESS);
  EXPECT_STREQ(graph_run_mode.GetString(), "");
}

TEST_F(UtestRegisterPass, CustomPassContext_SetPassName_GetPassName) {
  auto custom_pass = CustomPassContext();
  string pass_name = "TestPassName";
  custom_pass.SetPassName(pass_name.c_str());
  EXPECT_EQ(custom_pass.GetPassName().GetString(), pass_name);
}

TEST_F(UtestRegisterPass, CustomPassContext_GetPassName_without_SetPassName) {
  auto custom_pass = CustomPassContext();
  EXPECT_STREQ(custom_pass.GetPassName().GetString(), "");
}
} // namespace ge

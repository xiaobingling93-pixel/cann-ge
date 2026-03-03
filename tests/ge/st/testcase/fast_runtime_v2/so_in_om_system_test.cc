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
#include "faker/fake_value.h"
#include "base/registry/opp_package_utils.h"
#include "faker/space_registry_faker.h"
#include "common/share_graph.h"
#include "lowering/graph_converter.h"
#include "faker/global_data_faker.h"
#include "faker/model_data_faker.h"
#include "runtime/dev.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "stub/gert_runtime_stub.h"
#include "op_impl/less_important_op_impl.h"
#include "graph/utils/op_desc_utils.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "ge/ge_api_types.h"
#include "graph/ge_local_context.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "common/env_path.h"
#include "graph_metadef/common/ge_common/util.h"

using namespace ge;

namespace gert {
namespace {
const char *const kEnvName = "ASCEND_OPP_PATH";
const char *const kHomeEnvName = "HOME";
const string kOpsProto = "libopsproto_rt2.0.so";
const string kOpMaster = "libopmaster_rt2.0.so";
const string kInner = "built-in";
const string kx86OpsProtoPath = "/op_proto/lib/linux/x86_64/";
const string kx86OpMasterPath = "/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/";
const string kaarch64OpsProtoPath = "/op_proto/lib/linux/aarch64/";
std::map<std::string, std::string> options {{ge::OPTION_HOST_ENV_CPU, "x86_64"}, {ge::OPTION_HOST_ENV_OS, "linux"}};
void FreeModelData(ge::ModelData &model_data) {
  delete[] static_cast<ge::char_t *>(model_data.model_data);
  model_data.model_data = nullptr;
}

bool HaveExpectLog(std::vector<OneLog> &logs, std::string expect_log) {
  for (auto onelog : logs) {
    std::string content = onelog.content;
    if (content.find(expect_log) != std::string::npos) {
      return true;
    }
  }
  return false;
}

void CreateOpmasterSo1EnvInfoFunc(std::string opp_path, bool env_initialized = false) {
  if (env_initialized) {
    return;
  }
  gert::CreateOpmasterSoEnvInfoFunc(std::move(opp_path));
}

void CreateOpmasterSo2EnvInfoFunc(std::string opp_path, bool env_initialized = false) {
  if (env_initialized) {
    return;
  }
  auto base_path = EnvPath().GetAirBasePath();
  GELOGD("base_path :%s", base_path.c_str());
  std::string command = "find " + base_path + " -name " + "libgert_op_impl2.so";
  char retmsg[1024];
  (void)SuperSystem(command.c_str(), retmsg, sizeof(retmsg));
  std::string op_impl_path = retmsg;
  GELOGD("op_impl_path :%s", op_impl_path.c_str());

  opp_path += "/test/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize' > " + path_config).c_str());

  std::string inner_x86_op_master_path = opp_path + kInner + kx86OpMasterPath;
  GELOGD("inner_x86_op_master_path:%s", inner_x86_op_master_path.c_str());
  system(("mkdir -p " + inner_x86_op_master_path).c_str());
  std::string opmaster_rt2_path = inner_x86_op_master_path + kOpMaster;
  command = "cp " + op_impl_path + " " + opmaster_rt2_path;
  GELOGD("command: %s", command.c_str());
  system(command.c_str());

  std::string inner_x86_ops_proto_path = opp_path + kInner + kx86OpsProtoPath;
  GELOGD("inner_x86_ops_proto_path:%s", inner_x86_ops_proto_path.c_str());
  system(("mkdir -p " + inner_x86_ops_proto_path).c_str());
  std::string opsproto_rt2_path = inner_x86_ops_proto_path + kOpsProto;
  command = "cp " + op_impl_path + " " + opsproto_rt2_path;
  GELOGD("command: %s", command.c_str());
  system(command.c_str());
}

void CreateDlSymFaildEnvInfoFunc(std::string opp_path, bool env_initialized = false) {
  auto base_path = EnvPath().GetAirBasePath();
  GELOGD("base_path :%s", base_path.c_str());
  std::string command = "find " + base_path + " -name " + "libprofapi.so";
  char retmsg[1024];
  (void)SuperSystem(command.c_str(), retmsg, sizeof(retmsg));
  std::string op_impl_path = retmsg;
  GELOGD("op_impl_path :%s", op_impl_path.c_str());

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/confiLoadStreamExecutorFromModelDatag.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize' > " + path_config).c_str());

  std::string inner_x86_op_master_path = opp_path + kInner + kx86OpMasterPath;
  GELOGD("inner_x86_op_master_path:%s", inner_x86_op_master_path.c_str());
  system(("mkdir -p " + inner_x86_op_master_path).c_str());
  std::string opmaster_rt2_path = inner_x86_op_master_path + kOpMaster;
  command = "cp " + op_impl_path + " " + opmaster_rt2_path;
  GELOGD("command: %s", command.c_str());
  system(command.c_str());

  std::string inner_x86_ops_proto_path = opp_path + kInner + kx86OpsProtoPath;
  GELOGD("inner_x86_ops_proto_path:%s", inner_x86_ops_proto_path.c_str());
  system(("mkdir -p " + inner_x86_ops_proto_path).c_str());
  std::string opsproto_rt2_path = inner_x86_ops_proto_path + kOpsProto;
  command = "cp " + op_impl_path + " " + opsproto_rt2_path;
  GELOGD("command: %s", command.c_str());
  system(command.c_str());
}

int32_t GetOmSoFilesNumFromDisk() {
  char path_env[MMPA_MAX_PATH] = {0};
  int32_t ret = mmGetEnv(kHomeEnvName, path_env, MMPA_MAX_PATH);
  if ((ret != EN_OK) || (strlen(path_env) == 0)) {
    return -1;
  }
  std::string file_path = ge::RealPath(path_env);
  if (file_path.empty()) {
    return -1;
  }
  std::string om_exe_data = file_path + '/' + ".ascend_temp/.om_exe_data/";
  std::string command = "ls " + om_exe_data + " | wc -l";
  std::cout << command << std::endl;
  char retMsg[1024];
  (void)SuperSystem(command.c_str(), retMsg, sizeof(retMsg));
  std::string file_num(retMsg);
  int num = std::stoi(file_num);
  return num;
}

void RmOmSoFilesNumFromDisk() {
  char path_env[MMPA_MAX_PATH] = {0};
  int32_t ret = mmGetEnv(kHomeEnvName, path_env, MMPA_MAX_PATH);
  if ((ret != EN_OK) || (strlen(path_env) == 0)) {
    return;
  }
  std::string file_path = ge::RealPath(path_env);
  if (file_path.empty()) {
    return;
  }
  std::string om_exe_data = file_path + '/' + ".ascend_temp/.om_exe_data/";
  std::string command = "rm -rf " + om_exe_data;
  std::cout << command << std::endl;
  char retMsg[1024];
  (void)SuperSystem(command.c_str(), retMsg, sizeof(retMsg));
}

void CreateBuiltInAndCustomizeOppSoFunc(std::string opp_path, bool env_initialized = false) {
  if (env_initialized) {
    return;
  }
  CreateVendorsOppSo(std::move(opp_path), "customize");
}

void CreateMultiCustomizeOppSoFunc(std::string opp_path, bool env_initialized = false) {
  if (env_initialized) {
    return;
  }
  CreateVendorsOppSo(std::move(opp_path), "customize", "mdc");
}

class MockMmpa : public MmpaStubApiGe {
 public:
  INT32 mmMkdir(const CHAR *lp_path_name, mmMode_t mode) override {
    INT32 t_mode = mode;
    INT32 ret = EN_OK;

    if (NULL == lp_path_name) {
      syslog(LOG_ERR, "The input path is null.\r\n");
      return EN_INVALID_PARAM;
    }

    if (t_mode < MMPA_ZERO) {
      syslog(LOG_ERR, "The input mode is wrong.\r\n");
      return EN_INVALID_PARAM;
    }

    if (std::string(lp_path_name).find("tmp_home") != std::string::npos) {
      return EN_ERROR;
    }

    ret = mkdir(lp_path_name, mode);
    if (EN_OK != ret) {
      syslog(LOG_ERR, "Failed to create the directory, the ret is %s.\r\n", strerror(errno));
      return EN_ERROR;
    }
    return EN_OK;
  }
};
}

class SoInOmST : public testing::Test {
 protected:
  void SetUp() override {
    rtSetDevice(0);
    MmpaStub::GetInstance().Reset();
    ge::GetThreadLocalContext().SetGlobalOption(options);
    // creates a temporary opp directory
    gert::CreateVersionInfo();
    RmOmSoFilesNumFromDisk();
  }

  void TearDown() override {
    gert::DestroyVersionInfo();
    // remove the temporary opp directory
    gert::DestroyTempOppPath();
  }
};
const std::string DynamicAtomicStubName = "DynamicAtomicBin";

/**
 * 用例描述：default so 加载后om执行
 *
 * 预置条件：
 * 1. 安装算子包，om中不携带so
 *
 * 测试步骤：
 * 1. ST环境部署算子包
 * 2. 加载算子包
 * 3. 构造模型
 * 3. 模型加载、执行
 *
 * 预期结果：
 * 1. 模型执行成功
 * 2. 模型卸载后内存中持有default so
 */

TEST_F(SoInOmST, RunPackageSoLoad_0001) {
  {
    auto paths = gert::CreateSceneInfo();
    auto path = paths[0];
    system(("realpath " + path).c_str());
    gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry();

    auto graph = ShareGraph::BuildSingleNodeGraph();
    ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
    graph->TopologicalSorting();
    auto ge_root_model = GeModelBuilder(graph)
        .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
        .FakeTbeBin({"Add"})
        .BuildGeRootModel();

    auto model_data_holder = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
    ge::graphStatus error_code = ge::GRAPH_FAILED;
    auto stream_executor = LoadStreamExecutorFromModelData(model_data_holder.Get(), error_code);
    ASSERT_NE(stream_executor, nullptr);
    ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

    auto model_executor = stream_executor->GetOrCreateLoaded(stream, {stream, nullptr});
    ASSERT_NE(model_executor, nullptr);

    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                      outputs.size()),
              ge::GRAPH_SUCCESS);

    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                      outputs.size()),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
    system(("rm -f " + path).c_str());
  }
}

/**
 * 用例描述：default so 加载失败
 *
 * 预置条件：
 * 1. 安装算子包，om中不携带so
 * 2. 算子包环境变量配置一个不存在的路劲
 *
 * 测试步骤：
 * 1. ST环境部署算子包
 * 2. 加载算子包
 * 3. 构造模型
 * 3. 模型加载、执行
 *
 * 预期结果：
 * 1. defalut so 加载so失败，waring日志
 * 2. 模型执行失败
 */
#if 0
TEST_F(SoInOmST, RunPackageSoLoad_0002) {
  UnLoadDefaultSpaceRegistry();
  GertRuntimeStub stub;
  stub.GetSlogStub().Clear();
  {
    auto paths = CreateSceneInfo();
    auto scene_info_path = paths[0];
    auto opp_path = paths[1];
    opp_path = "./xxxx";
    mmSetEnv(kEnvName, opp_path.c_str(), 1);
    CreateOpmasterSo1EnvInfoFunc(opp_path);
    gert::OppPackageUtils::LoadAllOppPackage();
    system(("rm -f " + scene_info_path).c_str());

    auto graph = ShareGraph::BuildSingleNodeGraph();
    ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
    graph->TopologicalSorting();
    auto ge_root_model = GeModelBuilder(graph)
        .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
        .FakeTbeBin({"Add"})
        .BuildGeRootModel();

    auto model_data_holder = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
    ge::graphStatus error_code = ge::GRAPH_FAILED;
    auto stream_executor = LoadStreamExecutorFromModelData(model_data_holder.Get(), error_code);
    ASSERT_EQ(stream_executor, nullptr);
    ASSERT_NE(error_code, ge::GRAPH_SUCCESS);
  }

  std::string expect_log = "Get path with op_tiling path [./xxxx/op_impl/built-in/ai_core/tbe/op_tiling/lib/linux/x86_64/] failed";
  auto logs = stub.GetSlogStub().GetLogs(DLOG_WARN);
  EXPECT_EQ(HaveExpectLog(logs, expect_log), true);
  LoadDefaultSpaceRegistry();
}
#endif

/**
 * 用例描述：om携带so加载后om执行
 *
 * 预置条件：
 * 1. 不安装算子包
 *
 * 测试步骤：
 * 1. 构造ge_root_model
 * 2. 设置so打包进om的环境信息
 * 3. 构造model_data
 * 4. 模型加载、执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. ModelV2Executor 析构后，home目录下没有存盘的so
 * 3. ModelV2Executor 析构后，内存中不持有so
 */

TEST_F(SoInOmST, OmSoLoad_0001) {
  UnLoadDefaultSpaceRegistry();
  auto paths = CreateSceneInfo();
  auto scene_info_path = paths[0];
  auto opp_path = paths[1];

  {
    auto graph = ShareGraph::BuildSingleNodeGraph();
    ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
    graph->TopologicalSorting();
    auto ge_root_model = GeModelBuilder(graph)
                             .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                             .FakeTbeBin({"Add"})
                             .BuildGeRootModel();

    auto model_data = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShapeSoInOmFile(CreateOpmasterSo1EnvInfoFunc, opp_path);
    GE_MAKE_GUARD(release_model_data, [&model_data] {
      if (model_data.model_data != nullptr) {
        FreeModelData(model_data);
      }
    });

    ge::graphStatus error_code = ge::GRAPH_FAILED;
    auto stream_executor = LoadStreamExecutorFromModelData(model_data, error_code);
    ASSERT_NE(stream_executor, nullptr);
    ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

    auto model_executor = stream_executor->GetOrCreateLoaded(stream, {stream, nullptr});
    ASSERT_NE(model_executor, nullptr);

    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                      outputs.size()),
              ge::GRAPH_SUCCESS);

    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                      outputs.size()),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }

  ASSERT_EQ(GetOmSoFilesNumFromDisk(), 0);
  std::cout << "scene_info_path:" << scene_info_path << std::endl;
  system(("rm -f " + scene_info_path).c_str());
  // LoadDefaultSpaceRegistry();
}

/**
 * 用例描述：om携带so加载失败
 *
 * 预置条件：
 * 1. 不安装算子包
 * 2. 磁盘没有写权限
 *
 * 测试步骤：
 * 1. 构造ge_root_model
 * 2. 设置so打包进om的环境信息
 * 3. 构造model_data
 * 4. 模型加载
 *
 * 预期结果：
 * 1. so 落盘失败
 * 2. 模型加载失败
 */
TEST_F(SoInOmST, OmSoLoad_0002) {
  UnLoadDefaultSpaceRegistry();
  GertRuntimeStub stub;
  stub.GetSlogStub().SetLevel(DLOG_WARN);

  auto paths = CreateSceneInfo();
  auto scene_info_path = paths[0];
  auto opp_path = paths[1];

  auto graph = ShareGraph::BuildSingleNodeGraph();
  ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  graph->TopologicalSorting();
  auto ge_root_model = GeModelBuilder(graph)
                           .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .FakeTbeBin({"Add"})
                           .BuildGeRootModel();

  auto model_data =
      ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShapeSoInOmFile(CreateOpmasterSo1EnvInfoFunc, opp_path);
  GE_MAKE_GUARD(release_model_data, [&model_data] {
    if (model_data.model_data != nullptr) {
      FreeModelData(model_data);
    }
  });

  char old_path_env[MMPA_MAX_PATH] = {0};
  mmGetEnv(kHomeEnvName, old_path_env, MMPA_MAX_PATH);

  std::string home_path = "./tmp_home";
  system(("mkdir -p " + home_path).c_str());
  mmSetEnv(kHomeEnvName, home_path.c_str(), 1);

  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  ge::graphStatus error_code = ge::GRAPH_FAILED;
  auto stream_executor = LoadStreamExecutorFromModelData(model_data, error_code);
  ge::MmpaStub::GetInstance().Reset();
  ASSERT_EQ(stream_executor, nullptr);
  ASSERT_NE(error_code, ge::GRAPH_SUCCESS);
  ASSERT_EQ(GetOmSoFilesNumFromDisk(), 0);
  system(("rm -rf " + home_path).c_str());

  std::string expect_log = "Make sure the directory exists and writable";
  auto logs = stub.GetSlogStub().GetLogs(DLOG_WARN);
  EXPECT_EQ(HaveExpectLog(logs, expect_log), true);
  system(("rm -f " + scene_info_path).c_str());
  mmSetEnv(kHomeEnvName, old_path_env, 1);
  LoadDefaultSpaceRegistry();
}

/**
 * 用例描述：om携带so后om执行失败
 *
 * 预置条件：
 * 1. 不安装算子包
 * 2. 环境的os类型和om携带的os类型不匹配
 *
 * 测试步骤：
 * 1. 构造ge_root_model
 * 2. 设置so打包进om的环境信息
 * 3. 构造model_data
 * 4. 模型加载
 *
 * 预期结果：
 * 1. 模型加载失败， 打印os_type 校验失败
 */
TEST_F(SoInOmST, OmSoLoad_0003) {
  UnLoadDefaultSpaceRegistry();
  GertRuntimeStub stub;
  stub.GetSlogStub().Clear();

  std::string model_path = gert::GetTempOppBasePath();

  std::string opp_path = model_path + "opp/";
  system(("mkdir -p " + opp_path).c_str());
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  std::string scene_info_path = opp_path + "scene.info";
  system(("touch " + scene_info_path).c_str());
  system(("echo 'os=windows' > " + scene_info_path).c_str());
  system(("echo 'arch=x86_64' >> " + scene_info_path).c_str());

  auto graph = ShareGraph::BuildSingleNodeGraph();
  ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  graph->TopologicalSorting();
  auto ge_root_model = GeModelBuilder(graph)
      .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
      .FakeTbeBin({"Add"})
      .BuildGeRootModel();

  auto model_data =
      ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShapeSoInOmFile(CreateOpmasterSo1EnvInfoFunc, opp_path);
  GE_MAKE_GUARD(release_model_data, [&model_data] {
    if (model_data.model_data != nullptr) {
      FreeModelData(model_data);
    }
  });

  ge::graphStatus error_code = ge::GRAPH_FAILED;
  auto stream_executor = LoadStreamExecutorFromModelData(model_data, error_code);
  ASSERT_EQ(stream_executor, nullptr);
  ASSERT_NE(error_code, ge::GRAPH_SUCCESS);
  ASSERT_EQ(GetOmSoFilesNumFromDisk(), 0);
  system(("rm -f " + scene_info_path).c_str());

  std::string expect_log = "The os/cpu type of the model does not match the current system";
  auto logs = stub.GetSlogStub().GetLogs(DLOG_ERROR);
  EXPECT_EQ(HaveExpectLog(logs, expect_log), true);
  LoadDefaultSpaceRegistry();
}

/**
 * 用例描述：om携带so后om执行失败
 *
 * 预置条件：
 * 1. 不安装算子包
 * 2. 环境的cpu类型和om携带的cpu类型不匹配
 *
 * 测试步骤：
 * 1. 构造ge_root_model
 * 2. 设置so打包进om的环境信息
 * 3. 构造model_data
 * 4. 模型加载
 *
 * 预期结果：
 * 1. 模型加载失败， 打印cpu_type 校验失败
 */
TEST_F(SoInOmST, OmSoLoad_0004) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry();
  GertRuntimeStub stub;
  stub.GetSlogStub().Clear();

  std::string model_path = gert::GetTempOppBasePath();

  std::string opp_path = model_path + "opp/";
  system(("mkdir -p " + opp_path).c_str());
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  std::string scene_info_path = opp_path + "scene.info";
  system(("touch " + scene_info_path).c_str());
  system(("echo 'os=windows' > " + scene_info_path).c_str());
  system(("echo 'arch=aarch64' >> " + scene_info_path).c_str());

  auto graph = ShareGraph::BuildSingleNodeGraph();
  ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  graph->TopologicalSorting();
  auto ge_root_model = GeModelBuilder(graph)
      .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
      .FakeTbeBin({"Add"})
      .BuildGeRootModel();

  auto model_data =
      ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShapeSoInOmFile(CreateOpmasterSo1EnvInfoFunc, opp_path);
  GE_MAKE_GUARD(release_model_data, [&model_data] {
    if (model_data.model_data != nullptr) {
      FreeModelData(model_data);
    }
  });

  ge::graphStatus error_code = ge::GRAPH_FAILED;
  auto stream_executor = LoadStreamExecutorFromModelData(model_data, error_code);
  ASSERT_EQ(stream_executor, nullptr);
  ASSERT_NE(error_code, ge::GRAPH_SUCCESS);
  ASSERT_EQ(GetOmSoFilesNumFromDisk(), 0);
  system(("rm -f " + scene_info_path).c_str());
  std::string expect_log = "The os/cpu type of the model does not match the current system";
  auto logs = stub.GetSlogStub().GetLogs(DLOG_ERROR);
  EXPECT_EQ(HaveExpectLog(logs, expect_log), true);
}

/**
 * 用例描述：om携带so后om执行失败
 *
 * 预置条件：
 * 1. 不安装算子包
 * 2. 设置slsym失败的环境信息
 *
 * 测试步骤：
 * 1. 构造ge_root_model
 * 2. 设置so打包进om的环境信息
 * 3. 构造model_data
 * 4. 模型加载
 *
 * 预期结果：
 * 1. 模型加载失败，打印获取registed op num functions failed
 */
TEST_F(SoInOmST, OmSoLoad_0005) {
  GertRuntimeStub stub;
  stub.GetSlogStub().SetLevel(DLOG_WARN);
  stub.GetSlogStub().Clear();
  auto paths = CreateSceneInfo();
  auto scene_info_path = paths[0];
  auto opp_path = paths[1];

  auto graph = ShareGraph::BuildSingleNodeGraph();
  ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  graph->TopologicalSorting();
  auto ge_root_model = GeModelBuilder(graph)
      .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
      .FakeTbeBin({"Add"})
      .BuildGeRootModel();

  auto model_data =
      ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShapeSoInOmFile(CreateDlSymFaildEnvInfoFunc, opp_path);
  GE_MAKE_GUARD(release_model_data, [&model_data] {
    if (model_data.model_data != nullptr) {
      FreeModelData(model_data);
    }
  });

  ge::graphStatus error_code = ge::GRAPH_FAILED;
  auto stream_executor = LoadStreamExecutorFromModelData(model_data, error_code);
  ASSERT_NE(stream_executor, nullptr);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);
  ASSERT_EQ(GetOmSoFilesNumFromDisk(), 0);

  std::string expect_log = "Get registered op num functions failed";
  auto logs = stub.GetSlogStub().GetLogs(DLOG_WARN);
  EXPECT_EQ(HaveExpectLog(logs, expect_log), true);
  system(("rm -f " + scene_info_path).c_str());
}

/**
 * 用例描述：default 和om的不同版本so加载后om执行
 *
 * 预置条件：
 * 1. 安装算子包
 *
 * 测试步骤：
 * 1. 构造ge_root_model
 * 2. 设置so打包进om的环境信息，打包的so和算子包使用不同的so
 * 3. 构造model_data
 * 4. 模型加载、执行
 *
 * 预期结果：
 * 1. 执行成功，使用om携带so的infershape和tiling函数
 * 2. ModelV2Executor 析构后，home目录下没有存盘的so
 * 3. ModelV2Executor 析构后，内存中只持有default so
 */
TEST_F(SoInOmST, MultiSoLoad_0001) {
  UnLoadDefaultSpaceRegistry();
  auto paths = CreateSceneInfo();
  auto scene_info_path = paths[0];
  auto opp_path = paths[1];
  CreateOpmasterSo1EnvInfoFunc(opp_path);

  gert::OppPackageUtils::LoadAllOppPackage();

  {
    auto graph = ShareGraph::BuildSingleNodeGraph();
    ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
    graph->TopologicalSorting();
    auto ge_root_model = GeModelBuilder(graph)
                             .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                             .FakeTbeBin({"Add"})
                             .BuildGeRootModel();

    auto model_data =
        ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShapeSoInOmFile(CreateOpmasterSo2EnvInfoFunc, opp_path);
    GE_MAKE_GUARD(release_model_data, [&model_data] {
      if (model_data.model_data != nullptr) {
        FreeModelData(model_data);
      }
    });

    ge::graphStatus error_code = ge::GRAPH_FAILED;
    auto stream_executor = LoadStreamExecutorFromModelData(model_data, error_code);
    ASSERT_NE(stream_executor, nullptr);
    ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

    auto model_executor = stream_executor->GetOrCreateLoaded(stream, {stream, nullptr});
    ASSERT_NE(model_executor, nullptr);

    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                      outputs.size()),
              ge::GRAPH_SUCCESS);

    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                      outputs.size()),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }

  ASSERT_EQ(GetOmSoFilesNumFromDisk(), 0);
  system(("rm -f " + scene_info_path).c_str());
  UnLoadDefaultSpaceRegistry();
  LoadDefaultSpaceRegistry();
}

/**
 * 用例描述：default 和om的相同版本so加载后om执行
 *
 * 预置条件：
 * 1. 安装算子包
 *
 * 测试步骤：
 * 1. 构造ge_root_model
 * 2. 设置so打包进om的环境信息，打包的so和算子包使用相同的so
 * 3. 构造model_data
 * 4. 模型加载、执行
 *
 * 预期结果：
 * 1. 执行成功，使用om携带so的infershape和tiling函数
 * 2. ModelV2Executor 析构后，home目录下没有存盘的so
 * 3. ModelV2Executor 析构后，内存中只持有default so
 */
#if 0
TEST_F(SoInOmST, MultiSoLoad_0002) {
  UnLoadDefaultSpaceRegistry();
  GertRuntimeStub stub;
  stub.GetSlogStub().Clear();
  stub.GetSlogStub().SetLevel(DLOG_INFO);

  auto paths = CreateSceneInfo();
  auto scene_info_path = paths[0];
  auto opp_path = paths[1];
  CreateOpmasterSo2EnvInfoFunc(opp_path);

  gert::OppPackageUtils::LoadAllOppPackage();

  {
    auto graph = ShareGraph::BuildSingleNodeGraph();
    ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
    graph->TopologicalSorting();
    auto ge_root_model = GeModelBuilder(graph)
        .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
        .FakeTbeBin({"Add"})
        .BuildGeRootModel();

    auto model_data =
        ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShapeSoInOmFile(CreateOpmasterSo2EnvInfoFunc, opp_path, true);
    GE_MAKE_GUARD(release_model_data, [&model_data] {
      if (model_data.model_data != nullptr) {
        FreeModelData(model_data);
      }
    });

    ge::graphStatus error_code = ge::GRAPH_FAILED;
    auto stream_executor = LoadStreamExecutorFromModelData(model_data, error_code);
    ASSERT_NE(stream_executor, nullptr);
    ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

    auto model_executor = stream_executor->GetOrCreateLoaded(stream, {stream, nullptr});
    ASSERT_NE(model_executor, nullptr);

    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                      outputs.size()),
              ge::GRAPH_SUCCESS);

    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                      outputs.size()),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }

  ASSERT_EQ(GetOmSoFilesNumFromDisk(), 0);

  std::string expect_log = "so has been loaded";
  auto logs = stub.GetSlogStub().GetLogs(DLOG_INFO);
  EXPECT_EQ(HaveExpectLog(logs, expect_log), true);
  system(("rm -f " + scene_info_path).c_str());
  UnLoadDefaultSpaceRegistry();
  LoadDefaultSpaceRegistry();
}
#endif
/**
 * 用例描述：携带不同版本so的多om并行加载执行
 *
 * 预置条件：
 * 1. 不安装算子包
 *
 * 测试步骤：
 * 1. 构造两个模型分别打包不同版本so
 * 2. 模型1加载、执行
 * 3. 模型2加载、执行
 * 4. 模型2卸载、model_executor2 析构
 * 5. 模型1卸载、model_executor1 析构
 *
 * 预期结果：
 * 1. 执行成功
 * 2. model_executor2 析构后对应so关闭，模型1对应so未被关闭
 * 3. model_executor1 析构后，对应so关闭
 */

TEST_F(SoInOmST, ParallelOmLoad_0001) {
  UnLoadDefaultSpaceRegistry();
  ge::GetThreadLocalContext().SetGlobalOption(options);
  auto paths = CreateSceneInfo();
  auto scene_info_path = paths[0];
  auto opp_path = paths[1];

  {
    // 模型1加载和执行
    auto graph1 = ShareGraph::BuildSingleNodeGraph();
    ge::AttrUtils::SetBool(graph1, ge::ATTR_SINGLE_OP_SCENE, true);
    graph1->TopologicalSorting();
    auto ge_root_model1 = GeModelBuilder(graph1)
        .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
        .FakeTbeBin({"Add"})
        .BuildGeRootModel();

    auto model_data1 =
        ModelDataFaker().GeRootModel(ge_root_model1).BuildUnknownShapeSoInOmFile(CreateOpmasterSo1EnvInfoFunc, opp_path);
    GE_MAKE_GUARD(release_model_data, [&model_data1] {
      if (model_data1.model_data != nullptr) {
        FreeModelData(model_data1);
      }
    });
    ge::graphStatus error_code = ge::GRAPH_FAILED;
    auto stream_executor1 = LoadStreamExecutorFromModelData(model_data1, error_code);
    ASSERT_NE(stream_executor1, nullptr);
    ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

    rtStream_t stream1;
    ASSERT_EQ(rtStreamCreate(&stream1, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

    auto model_executor1 = stream_executor1->GetOrCreateLoaded(stream1, {stream1, nullptr});
    ASSERT_NE(model_executor1, nullptr);

    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream1));

    ASSERT_EQ(model_executor1->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                       outputs.size()),
              ge::GRAPH_SUCCESS);

    // 模型2加载和执行
    {
      auto graph2 = ShareGraph::BuildSingleNodeGraph();
      ge::AttrUtils::SetBool(graph2, ge::ATTR_SINGLE_OP_SCENE, true);
      graph2->TopologicalSorting();
      auto ge_root_model2 = GeModelBuilder(graph2)
          .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
          .FakeTbeBin({"Add"})
          .BuildGeRootModel();

      auto model_data2 =
          ModelDataFaker().GeRootModel(ge_root_model2).BuildUnknownShapeSoInOmFile(CreateOpmasterSo2EnvInfoFunc, opp_path);
      GE_MAKE_GUARD(release_model_data2, [&model_data2] {
        if (model_data2.model_data != nullptr) {
          FreeModelData(model_data2);
        }
      });

      auto stream_executor2 = LoadStreamExecutorFromModelData(model_data2, error_code);
      ASSERT_NE(stream_executor2, nullptr);
      ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

      rtStream_t stream2;
      ASSERT_EQ(rtStreamCreate(&stream2, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

      auto model_executor2 = stream_executor2->GetOrCreateLoaded(stream2, {stream2, nullptr});
      ASSERT_NE(model_executor2, nullptr);

      auto outputs2 = FakeTensors({2048}, 1);
      auto inputs2 = FakeTensors({2048}, 2);

      auto i3_2 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream2));

      ASSERT_EQ(model_executor2->Execute({i3_2.value}, inputs2.GetTensorList(), inputs2.size(),
                                         outputs2.GetTensorList(), outputs2.size()),
                ge::GRAPH_SUCCESS);
      // 模型2卸载
      ASSERT_EQ(model_executor2->UnLoad(), ge::GRAPH_SUCCESS);
      rtStreamDestroy(stream2);
    }

    // 模型1卸载
    ASSERT_EQ(model_executor1->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream1);
  }

  system(("rm -f " + scene_info_path).c_str());
  LoadDefaultSpaceRegistry();
}

/**
 * 用例描述：携带相同版本so的多om并行加载执行
 *
 * 预置条件：
 * 1. 不安装算子包
 *
 * 测试步骤：
 * 1. 构造两个模型分别打包相同版本so
 * 2. 模型1加载、执行
 * 3. 模型2加载、执行
 * 4. 模型1、模型2卸载、model_executor1、model_executor2析构
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 相同so只加载一次
 * 3. model_executor1 和 model_executor2 析构后不在持有so
 */

TEST_F(SoInOmST, ParallelOmLoad_0002) {
  UnLoadDefaultSpaceRegistry();
  ge::GetThreadLocalContext().SetGlobalOption(options);
  auto paths = CreateSceneInfo();
  auto scene_info_path = paths[0];
  auto opp_path = paths[1];
  {
    // 模型1加载和执行
    auto graph1 = ShareGraph::BuildSingleNodeGraph();
    ge::AttrUtils::SetBool(graph1, ge::ATTR_SINGLE_OP_SCENE, true);
    graph1->TopologicalSorting();
    auto ge_root_model1 = GeModelBuilder(graph1)
                              .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                              .FakeTbeBin({"Add"})
                              .BuildGeRootModel();

    auto model_data1 =
        ModelDataFaker().GeRootModel(ge_root_model1).BuildUnknownShapeSoInOmFile(CreateOpmasterSo1EnvInfoFunc, opp_path);
    GE_MAKE_GUARD(release_model_data, [&model_data1] {
      if (model_data1.model_data != nullptr) {
        FreeModelData(model_data1);
      }
    });
    ge::graphStatus error_code = ge::GRAPH_FAILED;
    auto stream_executor1 = LoadStreamExecutorFromModelData(model_data1, error_code);
    ASSERT_NE(stream_executor1, nullptr);
    ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

    rtStream_t stream1;
    ASSERT_EQ(rtStreamCreate(&stream1, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

    auto model_executor1 = stream_executor1->GetOrCreateLoaded(stream1, {stream1, nullptr});
    ASSERT_NE(model_executor1, nullptr);

    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream1));

    ASSERT_EQ(model_executor1->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                       outputs.size()),
              ge::GRAPH_SUCCESS);

    // 模型2加载和执行
    auto graph2 = ShareGraph::BuildSingleNodeGraph();
    ge::AttrUtils::SetBool(graph2, ge::ATTR_SINGLE_OP_SCENE, true);
    graph2->TopologicalSorting();
    auto ge_root_model2 = GeModelBuilder(graph2)
                              .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                              .FakeTbeBin({"Add"})
                              .BuildGeRootModel();

    auto model_data2 =
        ModelDataFaker().GeRootModel(ge_root_model2).BuildUnknownShapeSoInOmFile(CreateOpmasterSo1EnvInfoFunc, opp_path, true);
    GE_MAKE_GUARD(release_model_data2, [&model_data2] {
      if (model_data2.model_data != nullptr) {
        FreeModelData(model_data2);
      }
    });

    auto stream_executor2 = LoadStreamExecutorFromModelData(model_data2, error_code);
    ASSERT_NE(stream_executor2, nullptr);
    ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

    rtStream_t stream2;
    ASSERT_EQ(rtStreamCreate(&stream2, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

    auto model_executor2 = stream_executor2->GetOrCreateLoaded(stream2, {stream2, nullptr});
    ASSERT_NE(model_executor2, nullptr);

    auto outputs2 = FakeTensors({2048}, 1);
    auto inputs2 = FakeTensors({2048}, 2);

    auto i3_2 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream2));

    ASSERT_EQ(model_executor2->Execute({i3_2.value}, inputs2.GetTensorList(), inputs2.size(), outputs2.GetTensorList(),
                                       outputs2.size()),
              ge::GRAPH_SUCCESS);

    ASSERT_EQ(model_executor1->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream1);
    ASSERT_EQ(model_executor2->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream2);
  }
  system(("rm -f " + scene_info_path).c_str());
  LoadDefaultSpaceRegistry();
}

/**
 * 用例描述：携带相同版本so的多om并行加载执行
 *
 * 预置条件：
 * 1. 不安装算子包
 *
 * 测试步骤：
 * 1. 构造两个模型分别打包相同版本so
 * 2. 模型1加载、执行
 * 3. 模型2加载、执行
 * 4. 模型1卸载、model_executor1 析构
 * 5. 模型2卸载、model_executor2 析构
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 相同so只加载一次
 * 3. model_executor1 析构后内存中仍持有一个so
 * 4. model_executor2 析构后内存中不再持有so
 */
TEST_F(SoInOmST, ParallelOmLoad_0003) {
  UnLoadDefaultSpaceRegistry();
  auto paths = CreateSceneInfo();
  auto scene_info_path = paths[0];
  auto opp_path = paths[1];

  {
    ge::GetThreadLocalContext().SetGlobalOption(options);
    // 模型1加载和执行
    auto graph1 = ShareGraph::BuildSingleNodeGraph();
    ge::AttrUtils::SetBool(graph1, ge::ATTR_SINGLE_OP_SCENE, true);
    graph1->TopologicalSorting();
    auto ge_root_model1 = GeModelBuilder(graph1)
        .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
        .FakeTbeBin({"Add"})
        .BuildGeRootModel();

    auto model_data1 =
        ModelDataFaker().GeRootModel(ge_root_model1).BuildUnknownShapeSoInOmFile(CreateOpmasterSo1EnvInfoFunc, opp_path);
    GE_MAKE_GUARD(release_model_data, [&model_data1] {
      if (model_data1.model_data != nullptr) {
        FreeModelData(model_data1);
      }
    });
    ge::graphStatus error_code = ge::GRAPH_FAILED;
    auto stream_executor1 = LoadStreamExecutorFromModelData(model_data1, error_code);
    ASSERT_NE(stream_executor1, nullptr);
    ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

    rtStream_t stream1;
    ASSERT_EQ(rtStreamCreate(&stream1, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

    auto model_executor1 = stream_executor1->GetOrCreateLoaded(stream1, {stream1, nullptr});
    ASSERT_NE(model_executor1, nullptr);

    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream1));

    ASSERT_EQ(model_executor1->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                       outputs.size()),
              ge::GRAPH_SUCCESS);

    // 模型2加载和执行
    {
      auto graph2 = ShareGraph::BuildSingleNodeGraph();
      ge::AttrUtils::SetBool(graph2, ge::ATTR_SINGLE_OP_SCENE, true);
      graph2->TopologicalSorting();
      auto ge_root_model2 = GeModelBuilder(graph2)
                                .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                                .FakeTbeBin({"Add"})
                                .BuildGeRootModel();

      auto model_data2 =
          ModelDataFaker().GeRootModel(ge_root_model2).BuildUnknownShapeSoInOmFile(CreateOpmasterSo1EnvInfoFunc, opp_path, true);
      GE_MAKE_GUARD(release_model_data2, [&model_data2] {
        if (model_data2.model_data != nullptr) {
          FreeModelData(model_data2);
        }
      });

      auto stream_executor2 = LoadStreamExecutorFromModelData(model_data2, error_code);
      ASSERT_NE(stream_executor2, nullptr);
      ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

      rtStream_t stream2;
      ASSERT_EQ(rtStreamCreate(&stream2, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

      auto model_executor2 = stream_executor2->GetOrCreateLoaded(stream2, {stream2, nullptr});
      ASSERT_NE(model_executor2, nullptr);

      auto outputs2 = FakeTensors({2048}, 1);
      auto inputs2 = FakeTensors({2048}, 2);

      auto i3_2 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream2));

      ASSERT_EQ(model_executor2->Execute({i3_2.value}, inputs2.GetTensorList(), inputs2.size(),
                                         outputs2.GetTensorList(), outputs2.size()),
                ge::GRAPH_SUCCESS);
      // 模型2卸载
      ASSERT_EQ(model_executor2->UnLoad(), ge::GRAPH_SUCCESS);
      rtStreamDestroy(stream2);
    }

    // 模型1卸载
    ASSERT_EQ(model_executor1->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream1);
  }
  system(("rm -f " + scene_info_path).c_str());
  LoadDefaultSpaceRegistry();
}

/**
 * 用例描述：携带相同版本so的多om并行加载执行，且环境上有相同版本so算子包
 *
 * 预置条件：
 * 1. 安装算子包
 *
 * 测试步骤：
 * 1. 加载算子包
 * 1. 构造两个模型分别打包相同版本so（该so和算子包相同）
 * 2. 模型1加载、执行
 * 3. 模型2加载、执行
 * 4. 模型2卸载、model_executor2 析构
 * 5. 模型1卸载、model_executor1 析构
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 相同so只加载一次
 * 3. model_executor2 析构后内存中仍持有一个so
 * 4. model_executor1 析构后内存中仍持有一个so
 */
#if 0
TEST_F(SoInOmST, ParallelOmLoad_0004) {
  UnLoadDefaultSpaceRegistry();
  auto paths = CreateSceneInfo();
  auto scene_info_path = paths[0];
  auto opp_path = paths[1];
  CreateOpmasterSo1EnvInfoFunc(opp_path);
  gert::OppPackageUtils::LoadAllOppPackage();

  {
    // 模型1加载和执行
    auto graph1 = ShareGraph::BuildSingleNodeGraph();
    ge::AttrUtils::SetBool(graph1, ge::ATTR_SINGLE_OP_SCENE, true);
    graph1->TopologicalSorting();
    auto ge_root_model1 = GeModelBuilder(graph1)
                              .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                              .FakeTbeBin({"Add"})
                              .BuildGeRootModel();
    auto model_data1 =
        ModelDataFaker().GeRootModel(ge_root_model1).BuildUnknownShapeSoInOmFile(CreateOpmasterSo1EnvInfoFunc, opp_path, true);
    GE_MAKE_GUARD(release_model_data, [&model_data1] {
      if (model_data1.model_data != nullptr) {
        FreeModelData(model_data1);
      }
    });
    ge::graphStatus error_code = ge::GRAPH_FAILED;
    auto stream_executor1 = LoadStreamExecutorFromModelData(model_data1, error_code);
    ASSERT_NE(stream_executor1, nullptr);
    ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

    rtStream_t stream1;
    ASSERT_EQ(rtStreamCreate(&stream1, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

    auto model_executor1 = stream_executor1->GetOrCreateLoaded(stream1, {stream1, nullptr});
    ASSERT_NE(model_executor1, nullptr);

    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream1));

    ASSERT_EQ(model_executor1->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                       outputs.size()),
              ge::GRAPH_SUCCESS);

    // 模型2加载和执行
    {
      auto graph2 = ShareGraph::BuildSingleNodeGraph();
      ge::AttrUtils::SetBool(graph2, ge::ATTR_SINGLE_OP_SCENE, true);
      graph2->TopologicalSorting();
      auto ge_root_model2 = GeModelBuilder(graph2)
                                .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                                .FakeTbeBin({"Add"})
                                .BuildGeRootModel();

      auto model_data2 =
          ModelDataFaker().GeRootModel(ge_root_model2).BuildUnknownShapeSoInOmFile(CreateOpmasterSo1EnvInfoFunc, opp_path, true);
      GE_MAKE_GUARD(release_model_data2, [&model_data2] {
        if (model_data2.model_data != nullptr) {
          FreeModelData(model_data2);
        }
      });

      auto stream_executor2 = LoadStreamExecutorFromModelData(model_data2, error_code);
      ASSERT_NE(stream_executor2, nullptr);
      ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

      rtStream_t stream2;
      ASSERT_EQ(rtStreamCreate(&stream2, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

      auto model_executor2 = stream_executor2->GetOrCreateLoaded(stream2, {stream2, nullptr});
      ASSERT_NE(model_executor2, nullptr);

      auto outputs2 = FakeTensors({2048}, 1);
      auto inputs2 = FakeTensors({2048}, 2);

      auto i3_2 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream2));

      ASSERT_EQ(model_executor2->Execute({i3_2.value}, inputs2.GetTensorList(), inputs2.size(),
                                         outputs2.GetTensorList(), outputs2.size()),
                ge::GRAPH_SUCCESS);
      // 模型2卸载
      ASSERT_EQ(model_executor2->UnLoad(), ge::GRAPH_SUCCESS);
      rtStreamDestroy(stream2);
    }

    // 模型1卸载
    ASSERT_EQ(model_executor1->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream1);
  }
  system(("rm -f " + scene_info_path).c_str());
  UnLoadDefaultSpaceRegistry();
  LoadDefaultSpaceRegistry();
}
#endif
/**
 * 用例描述：携带相同版本so的多om并行加载执行，环境上有不同版本so算子包
 *
 * 预置条件：
 * 1. 安装算子包
 *
 * 测试步骤：
 * 1. 加载算子包
 * 1. 构造两个模型分别打包相同版本so（该so和算子包不相同）
 * 2. 模型1加载、执行
 * 3. 模型2加载、执行
 * 4. 模型2卸载、model_executor2 析构
 * 5. 模型1卸载、model_executor1 析构
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 内存中持有两个so，一个是算子包so，另一个是om中携带的so
 * 3. model_executor2 析构后内存中持有两个so
 * 4. model_executor1 析构后内存中仍持有一个so
 */
TEST_F(SoInOmST, ParallelOmLoad_0005) {
  UnLoadDefaultSpaceRegistry();
  auto paths = CreateSceneInfo();
  auto scene_info_path = paths[0];
  auto opp_path = paths[1];
  CreateOpmasterSo2EnvInfoFunc(opp_path);
  gert::OppPackageUtils::LoadAllOppPackage();

  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  {
    ge::GetThreadLocalContext().SetGlobalOption(options);
    // 模型1加载和执行
    auto graph1 = ShareGraph::BuildSingleNodeGraph();
    ge::AttrUtils::SetBool(graph1, ge::ATTR_SINGLE_OP_SCENE, true);
    graph1->TopologicalSorting();
    auto ge_root_model1 = GeModelBuilder(graph1)
        .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
        .FakeTbeBin({"Add"})
        .BuildGeRootModel();

    auto model_data1 =
        ModelDataFaker().GeRootModel(ge_root_model1).BuildUnknownShapeSoInOmFile(CreateOpmasterSo1EnvInfoFunc, opp_path);
    GE_MAKE_GUARD(release_model_data, [&model_data1] {
      if (model_data1.model_data != nullptr) {
        FreeModelData(model_data1);
      }
    });
    ge::graphStatus error_code = ge::GRAPH_FAILED;
    auto stream_executor1 = LoadStreamExecutorFromModelData(model_data1, error_code);
    ASSERT_NE(stream_executor1, nullptr);
    ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

    rtStream_t stream1;
    ASSERT_EQ(rtStreamCreate(&stream1, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

    auto model_executor1 = stream_executor1->GetOrCreateLoaded(stream1, {stream1, nullptr});
    ASSERT_NE(model_executor1, nullptr);

    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream1));

    ASSERT_EQ(model_executor1->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                       outputs.size()),
              ge::GRAPH_SUCCESS);

    // 模型2加载和执行
    {
      auto graph2 = ShareGraph::BuildSingleNodeGraph();
      ge::AttrUtils::SetBool(graph2, ge::ATTR_SINGLE_OP_SCENE, true);
      graph2->TopologicalSorting();
      auto ge_root_model2 = GeModelBuilder(graph2)
          .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
          .FakeTbeBin({"Add"})
          .BuildGeRootModel();

      auto model_data2 =
          ModelDataFaker().GeRootModel(ge_root_model2).BuildUnknownShapeSoInOmFile(CreateOpmasterSo1EnvInfoFunc, opp_path, true);
      GE_MAKE_GUARD(release_model_data2, [&model_data2] {
        if (model_data2.model_data != nullptr) {
          FreeModelData(model_data2);
        }
      });

      auto stream_executor2 = LoadStreamExecutorFromModelData(model_data2, error_code);
      ASSERT_NE(stream_executor2, nullptr);
      ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

      rtStream_t stream2;
      ASSERT_EQ(rtStreamCreate(&stream2, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

      auto model_executor2 = stream_executor2->GetOrCreateLoaded(stream2, {stream2, nullptr});
      ASSERT_NE(model_executor2, nullptr);

      auto outputs2 = FakeTensors({2048}, 1);
      auto inputs2 = FakeTensors({2048}, 2);

      auto i3_2 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream2));

      ASSERT_EQ(model_executor2->Execute({i3_2.value}, inputs2.GetTensorList(), inputs2.size(),
                                         outputs2.GetTensorList(), outputs2.size()),
                ge::GRAPH_SUCCESS);
      // 模型2卸载
      ASSERT_EQ(model_executor2->UnLoad(), ge::GRAPH_SUCCESS);
      rtStreamDestroy(stream2);
    }

    // 模型1卸载
    ASSERT_EQ(model_executor1->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream1);
  }
  system(("rm -f " + scene_info_path).c_str());
}

/**
 * 用例描述：om中内置和自定义不同版本的so加载后om执行
 *
 * 预置条件：
 * 1. 安装算子包
 *
 * 测试步骤：
 * 1. 构造ge_root_model
 * 2. 设置so打包进om的环境信息，打包的内置和自定义so使用不同的so
 * 3. 构造model_data
 * 4. 模型加载、执行
 *
 * 预期结果：
 * 1. ModelV2Executor 析构后，home目录下没有存盘的so
 * 2. ModelV2Executor 析构后，内存中只持有default so
 */
TEST_F(SoInOmST, BuiltInAndCustomizeSoLoad) {
  UnLoadDefaultSpaceRegistry();
  auto paths = CreateSceneInfo();
  auto scene_info_path = paths[0];
  auto opp_path = paths[1];
  auto model_path = paths[2];
  std::string runtime_path = model_path + "/opp/runtime";
  auto runtime_verson_info_path = runtime_path + "/version.info";

  auto graph = ShareGraph::BuildSingleNodeGraph();
  ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  graph->TopologicalSorting();
  auto ge_root_model = GeModelBuilder(graph)
                           .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .FakeTbeBin({"Add"})
                           .BuildGeRootModel();

  auto model_data = ModelDataFaker()
                        .GeRootModel(ge_root_model)
                        .BuildUnknownShapeSoInOmFile(CreateBuiltInAndCustomizeOppSoFunc, opp_path);

  system(("mkdir -p " + runtime_path).c_str());
  system(("echo 'required_opp_abi_version=>=6.4,<=6.4' > " + runtime_verson_info_path).c_str());

  GE_MAKE_GUARD(release_model_data, [&model_data] {
    if (model_data.model_data != nullptr) {
      FreeModelData(model_data);
    }
  });

  ge::graphStatus error_code = ge::GRAPH_FAILED;
  auto stream_executor = LoadStreamExecutorFromModelData(model_data, error_code);
  ASSERT_NE(stream_executor, nullptr);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

  auto model_executor = stream_executor->GetOrCreateLoaded(stream, {stream, nullptr});
  ASSERT_NE(model_executor, nullptr);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);

  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);

  ASSERT_EQ(GetOmSoFilesNumFromDisk(), 0);
  system(("rm -rf " + scene_info_path).c_str());
  UnLoadDefaultSpaceRegistry();
  LoadDefaultSpaceRegistry();
}

/**
 * 用例描述：om中自定义不同版本的so加载后om执行
 *
 * 预置条件：
 * 1. 安装算子包
 *
 * 测试步骤：
 * 1. 构造ge_root_model
 * 2. 设置so打包进om的环境信息，打包的自定义so使用不同的so
 * 3. 构造model_data
 * 4. 模型加载、执行
 *
 * 预期结果：
 * 1. ModelV2Executor 析构后，home目录下没有存盘的so
 * 2. ModelV2Executor 析构后，内存中只持有default so
 */
TEST_F(SoInOmST, MultiCustomizeSoLoad) {
  UnLoadDefaultSpaceRegistry();
  auto paths = CreateSceneInfo();
  auto scene_info_path = paths[0];
  auto opp_path = paths[1];

  auto graph = ShareGraph::BuildSingleNodeGraph();
  ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  graph->TopologicalSorting();
  auto ge_root_model = GeModelBuilder(graph)
                           .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .FakeTbeBin({"Add"})
                           .BuildGeRootModel();

  auto model_data =
      ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShapeSoInOmFile(CreateMultiCustomizeOppSoFunc, opp_path);
  GE_MAKE_GUARD(release_model_data, [&model_data] {
    if (model_data.model_data != nullptr) {
      FreeModelData(model_data);
    }
  });

  ge::graphStatus error_code = ge::GRAPH_FAILED;
  auto stream_executor = LoadStreamExecutorFromModelData(model_data, error_code);
  ASSERT_NE(stream_executor, nullptr);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

  auto model_executor = stream_executor->GetOrCreateLoaded(stream, {stream, nullptr});
  ASSERT_NE(model_executor, nullptr);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);

  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);

  ASSERT_EQ(GetOmSoFilesNumFromDisk(), 0);
  system(("rm -rf " + scene_info_path).c_str());
  UnLoadDefaultSpaceRegistry();
  LoadDefaultSpaceRegistry();
}
}  // namespace gert

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/path_utils.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "faker/space_registry_faker.h"
#include "mmpa/mmpa_api.h"
#include "base/registry/opp_package_utils.h"
#include "graph_metadef/common/plugin/plugin_manager.h"

namespace gert {
namespace {
const char *const kEnvName = "ASCEND_OPP_PATH";
const char *const kAscendHomePath = "ASCEND_HOME_PATH";
const std::string kInner = "built-in";
const std::string kVendor = "vendors";
const std::string kx86OpsProtoPath = "/op_proto/lib/linux/x86_64/";
const std::string kx86OpMasterPath = "/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/";
void *handle = nullptr;

void CopyStubSoToOppPath(const std::string &path_stub_so, const std::string &opp_path, const std::string &suffix) {
  std::string inner_x86_opmaster_path = opp_path + suffix + kx86OpMasterPath;
  GELOGD("inner_x86_opmaster_path:%s", inner_x86_opmaster_path.c_str());
  system(("mkdir -p " + inner_x86_opmaster_path).c_str());
  std::string opmaster_rt2_path = inner_x86_opmaster_path + "libopmaster_rt2.0.so";
  auto command = "cp " + path_stub_so + " " + opmaster_rt2_path;
  GELOGD("command: %s", command.c_str());
  system(command.c_str());

  std::string inner_x86_opsproto_path = opp_path + suffix + kx86OpsProtoPath;
  GELOGD("inner_x86_opsproto_path:%s", inner_x86_opsproto_path.c_str());
  system(("mkdir -p " + inner_x86_opsproto_path).c_str());
  std::string opsproto_rt2_path = inner_x86_opsproto_path + "libopsproto_rt2.0.so";
  command = "cp " + path_stub_so + " " + opsproto_rt2_path;
  GELOGD("command: %s", command.c_str());
  system(command.c_str());
}
}

int SuperSystem(const char *cmd, char *retmsg, int msg_len) {
  FILE *fp;
    GELOGD("current cmd %s", cmd);
  int res = -1;
  if (cmd == NULL || retmsg == NULL || msg_len < 0) {
    GELOGD("Err: Fuc:%s system paramer invalid! ", __func__);
    return 1;
  }
  if ((fp = popen(cmd, "r")) == NULL) {
    perror("popen");
    GELOGD("Err: Fuc:%s popen error: %s ", __func__, strerror(errno));
    return 2;
  } else {
    memset(retmsg, 0, msg_len);
    while (fgets(retmsg, msg_len - 1, fp))
    {
      GELOGD("Fuc: %s fgets buf is %s ", __func__, retmsg);
    }
    if ((res = pclose(fp)) == -1) {
      GELOGD("Fuc:%s close popen file pointer fp error! ", __func__);
      return 3;
    }
    if (strlen(retmsg) != 0) {
      retmsg[strlen(retmsg) - 1] = '\0';
    }
    return 0;
  }
}

void UnLoadDefaultSpaceRegistry() {
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
}

std::string GetTempOppBasePath() {
  std::string model_path = ge::GetModelPath();
  GELOGD("Current lib path is:%s", model_path.c_str());
  model_path = model_path.substr(0, model_path.rfind('/'));
  model_path = model_path.substr(0, model_path.rfind('/'));
  model_path = model_path.substr(0, model_path.rfind('/') + 1);

  return model_path;
}

void DestroyTempOppPath() {
  std::string model_path = GetTempOppBasePath();
  std::string command = "rm -rf " + model_path + "opp/";
  std::cout << "DestroyTempOppPath:" << command << std::endl;
  GELOGD("command: %s", command.c_str());
  system(command.c_str());
}

std::vector<std::string> CreateSceneInfo() {
  std::string model_path = GetTempOppBasePath();
  GELOGD("Current model path is:%s", model_path.c_str());

  std::string opp_path = model_path + "opp/";
  system(("mkdir -p " + opp_path).c_str());
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  std::string scene_info_path = opp_path + "scene.info";
  system(("touch " + scene_info_path).c_str());
  system(("echo 'os=linux' > " + scene_info_path).c_str());
  system(("echo 'arch=x86_64' >> " + scene_info_path).c_str());

  return {scene_info_path, opp_path, model_path};
}

void CreateVersionInfo() {
  std::string model_path = GetTempOppBasePath();
  GELOGD("Current model path is:%s", model_path.c_str());

  std::string opp_path = model_path + "opp/";
  system(("mkdir -p " + opp_path).c_str());
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  std::string version_info_path = opp_path + "version.info";
  system(("touch " + version_info_path).c_str());
  system(("echo 'Version=6.4.T5.0.B121' > " + version_info_path).c_str());
}

void DestroyVersionInfo() {
  std::string model_path = GetTempOppBasePath();
  GELOGD("Current model path is:%s", model_path.c_str());

  std::string opp_path = model_path + "opp/";
  std::string version_info_path = opp_path + "version.info";
  system(("rm -rf " + version_info_path).c_str());
}

void CreateOpmasterSoEnvInfoFunc(std::string opp_path) {
  std::string cmake_binary_path = CMAKE_BINARY_DIR;
  auto op_impl_path = cmake_binary_path + "/tests/depends/op_stub/libgert_op_impl.so";
  GELOGD("op_impl_path :%s", op_impl_path.c_str());

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/confiLoadStreamExecutorFromModelDatag.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize' > " + path_config).c_str());

  std::string inner_x86_opmaster_path = opp_path + kInner + kx86OpMasterPath;
  GELOGD("inner_x86_opmaster_path:%s", inner_x86_opmaster_path.c_str());
  system(("mkdir -p " + inner_x86_opmaster_path).c_str());
  std::string opmaster_rt2_path = inner_x86_opmaster_path + "libopmaster_rt2.0.so";
  auto command = "cp " + op_impl_path + " " + opmaster_rt2_path;
  GELOGD("command: %s", command.c_str());
  system(command.c_str());

  std::string inner_x86_opsproto_path = opp_path + kInner + kx86OpsProtoPath;
  GELOGD("inner_x86_opsproto_path:%s", inner_x86_opsproto_path.c_str());
  system(("mkdir -p " + inner_x86_opsproto_path).c_str());
  std::string opsproto_rt2_path = inner_x86_opsproto_path + "libopsproto_rt2.0.so";
  command = "cp " + op_impl_path + " " + opsproto_rt2_path;
  GELOGD("command: %s", command.c_str());
  system(command.c_str());
}

void CloseHandle(void *&handle) {
  if (handle != nullptr) {
    GELOGI("start close handle, handle[%p].", handle);
    if (mmDlclose(handle) != 0) {
      const ge::char_t *error = mmDlerror();
      error = (error == nullptr) ? "" : error;
      GELOGE(ge::FAILED, "[Close][Handle] failed, reason:%s", error);
      return;
    }
  }
  handle = nullptr;
}

SpaceRegistryFaker::~SpaceRegistryFaker() {
  CloseHandle(handle);
}

OpImplSpaceRegistryV2Ptr SpaceRegistryFaker::Build() {
  auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
   std::string cmake_binary_path = CMAKE_BINARY_DIR;
   auto op_impl_path = cmake_binary_path + "/tests/depends/op_stub/libgert_op_impl2.so";

   space_registry->AddSoToRegistry(
      gert::OppSoDesc(std::vector<ge::AscendString>{ge::AscendString(op_impl_path.c_str())}, "libgert_op_impl2.so"));
  return space_registry;
}

std::shared_ptr<OpImplSpaceRegistryV2Array> SpaceRegistryFaker::BuildRegistryArray() {
  auto space_registry_array = ge::MakeShared<OpImplSpaceRegistryV2Array>();
  if (space_registry_array == nullptr) {
    return nullptr;
  }
  space_registry_array->at(static_cast<size_t>(gert::OppImplVersionTag::kOpp)) = Build();
  space_registry_array->at(static_cast<size_t>(gert::OppImplVersionTag::kOppKernel)) = Build();
  return space_registry_array;
}

// 有InferSymbolShape的场景会创建DefaultSpaceRegistry，如果后面再重新创建DefaultSpaceRegistry会导致InferSymbolShape丢失
// 这里使用原来的DefaultSpaceRegistry
std::shared_ptr<OpImplSpaceRegistryV2> GetMainSpaceRegistry() {
  auto space_registry = std::make_shared<OpImplSpaceRegistryV2>();
  if (space_registry == nullptr) {
    GELOGE(ge::FAILED, "Create space registry failed!");
    return nullptr;
  }
  space_registry->AddSoToRegistry(
      gert::OppSoDesc(std::vector<ge::AscendString>{ge::AscendString("main_exe")}, "main_exe"));

  GELOGI("space_registry:%p", (void *)space_registry.get());
  return space_registry;
}

OpImplSpaceRegistryV2Ptr SpaceRegistryFaker::BuildMainSpace() {
  return GetMainSpaceRegistry();
}

std::shared_ptr<OpImplSpaceRegistryV2Array> SpaceRegistryFaker::BuildMainSpaceRegistryArray() {
  auto space_registry_array = ge::MakeShared<OpImplSpaceRegistryV2Array>();
  if (space_registry_array == nullptr) {
    return nullptr;
  }
  space_registry_array->at(static_cast<size_t>(gert::OppImplVersionTag::kOpp)) = GetMainSpaceRegistry();
  space_registry_array->at(static_cast<size_t>(gert::OppImplVersionTag::kOppKernel)) = GetMainSpaceRegistry();
  return space_registry_array;
}

void SpaceRegistryFaker::RestorePreRegisteredSpaceRegistry() {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  if (space_registry == nullptr) {
    return;
  }

  std::string cmake_binary_path = CMAKE_BINARY_DIR;
  auto op_impl_path = cmake_binary_path + "/tests/depends/op_stub/libgert_op_impl2.so";
  space_registry->AddSoToRegistry(
      gert::OppSoDesc(std::vector<ge::AscendString>{ge::AscendString(op_impl_path.c_str())}, "libgert_op_impl2.so"));
}

void SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry() {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  if (space_registry == nullptr) {
    space_registry = std::make_shared<OpImplSpaceRegistryV2>();
    if (space_registry == nullptr) {
      GELOGE(ge::FAILED, "Create space registry failed!");
      return;
    }
  }

  space_registry->AddSoToRegistry(
      gert::OppSoDesc(std::vector<ge::AscendString>{ge::AscendString("main_exe")}, "main_exe"));

  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);
}

void SpaceRegistryFaker::CreateDefaultSpaceRegistry(bool only_main_space) {
  // 重新创建避免用例间影响
  auto  space_registry = GetMainSpaceRegistry();
  if (space_registry == nullptr) {
    GELOGE(ge::FAILED, "Create space registry failed!");
    return;
  }

  if (!only_main_space) {
    std::string cmake_binary_path = CMAKE_BINARY_DIR;
    auto op_impl_path = cmake_binary_path + "/tests/depends/op_stub/libgert_op_impl.so";
    space_registry->AddSoToRegistry(
        gert::OppSoDesc(std::vector<ge::AscendString>{ge::AscendString(op_impl_path.c_str())}, "libgert_op_impl.so"));
  }

  GELOGI("space_registry:%p", (void*)space_registry.get());
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);
}

void SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl() {
  auto space_registry = std::make_shared<OpImplSpaceRegistryV2>();
  if (space_registry == nullptr) {
    GELOGE(ge::FAILED, "Create space registry failed!");
    return;
  }

  std::string cmake_binary_path = CMAKE_BINARY_DIR;
  auto op_impl_path = cmake_binary_path + "/tests/depends/op_stub/libgert_op_impl.so";
  space_registry->AddSoToRegistry(
      gert::OppSoDesc(std::vector<ge::AscendString>{ge::AscendString(op_impl_path.c_str())}, "libgert_op_impl.so"));

  GELOGI("space_registry:%p", (void *)space_registry.get());
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);
}

void SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2(bool only_main_space) {
  // 重新创建避免用例间影响
  auto space_registry = GetMainSpaceRegistry();
  if (space_registry == nullptr) {
    GELOGE(ge::FAILED, "Create space registry failed!");
    return;
  }

  if (!only_main_space) {
    std::string cmake_binary_path = CMAKE_BINARY_DIR;
    auto op_impl_path = cmake_binary_path + "/tests/depends/op_stub/libgert_op_impl2.so";
    space_registry->AddSoToRegistry(
        gert::OppSoDesc(std::vector<ge::AscendString>{ge::AscendString(op_impl_path.c_str())}, "libgert_op_impl2.so"));
  }
  GELOGI("space_registry:%p", (void*)space_registry.get());
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);
}

void LoadMainSpaceRegistry() {
  auto space_registry = GetMainSpaceRegistry();
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);
}

// 有注册到主进程空间的impl，也有注册到libgert_op_impl.so里的impl
void LoadDefaultSpaceRegistry() {
  LoadMainSpaceRegistry();
  auto paths = CreateSceneInfo();
  auto scene_info_path = paths[0];
  auto opp_path = paths[1];
  CreateOpmasterSoEnvInfoFunc(opp_path);
  gert::OppPackageUtils::LoadAllOppPackage();
  system(("rm -f " + scene_info_path).c_str());
}

void CreateVendorsOppSo(const std::string &opp_path, const std::string &customize_1, const std::string &customize_2) {
  std::string cmake_binary_path = CMAKE_BINARY_DIR;
  auto op_impl_path = cmake_binary_path + "/tests/depends/op_stub/libgert_op_impl.so";
  GELOGD("op_impl_path :%s", op_impl_path.c_str());
  // High priority
  auto op_impl_path1 = cmake_binary_path + "/tests/depends/op_stub/libgert_op_impl2.so";
  GELOGD("op_impl_path :%s", op_impl_path1.c_str());

  std::string path_vendors = opp_path + kVendor;
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());

  CopyStubSoToOppPath(op_impl_path, opp_path, kInner);
  system(("echo 'Version=6.4.T5.0.B121' > " + opp_path + "/version.info").c_str());

  std::string load_priority;
  if (!customize_1.empty()) {
    load_priority += customize_1;
    CopyStubSoToOppPath(op_impl_path1, opp_path, (kVendor + "/" + customize_1));
    system(("echo 'compiler_version=6.4.T5.0.B121' > " + opp_path + kVendor + "/" + customize_1 + "/version.info").c_str());
  }
  if (!customize_2.empty()) {
    load_priority += "," + customize_2;
    CopyStubSoToOppPath(op_impl_path, opp_path, (kVendor + "/" + customize_2));
    system(("echo 'compiler_version=6.5.T5.0.B121' > " + opp_path + kVendor + "/" + customize_2 + "/version.info").c_str());
  }
  GELOGD("load_priority is:%s", load_priority.c_str());
  system(("echo 'load_priority=" + load_priority + "' > " + path_config).c_str());
}

void CreateBuiltInSplitAndUpgradedSo(std::vector<std::string> &paths) {
  auto path_infos = CreateSceneInfo();
  auto opp_path = path_infos[1U];

  std::string proto_path = opp_path + "built-in/op_proto/lib/linux/x86_64";
  system(("mkdir -p " + proto_path).c_str());
  proto_path += "/a_rt.so";
  paths.emplace_back(proto_path);
  system(("touch " + proto_path).c_str());
  system(("echo 'ops proto 123' > " + proto_path).c_str());

  std::string tiling_path = opp_path + "built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64";
  system(("mkdir -p " + tiling_path).c_str());
  tiling_path += "/b_rt.so";
  paths.emplace_back(tiling_path);
  system(("touch " + tiling_path).c_str());
  system(("echo 'op tiling 456' > " + tiling_path).c_str());

  std::string home_path = opp_path.substr(0, opp_path.rfind("/opp/"));
  mmSetEnv(kAscendHomePath, home_path.c_str(), 1);
  std::string opp_latest_path = home_path + "/opp_latest/";
  system(("mkdir -p " + opp_latest_path).c_str());
  paths.emplace_back(opp_latest_path);
  system(("cp -rf " + opp_path + "built-in " + opp_latest_path).c_str());
}

void CreateBuiltInSubPkgSo(std::vector<std::string> &paths) {
  auto path_infos = CreateSceneInfo();
  auto opp_path = path_infos[1U];

  std::string proto_path = opp_path + "built-in/op_proto/lib/linux/x86_64";
  system(("mkdir -p " + proto_path).c_str());
  proto_path += "/a_rt.so";
  paths.emplace_back(proto_path);
  system(("touch " + proto_path).c_str());
  system(("echo 'ops proto 123' > " + proto_path).c_str());

  std::string tiling_path = opp_path + "built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64";
  system(("mkdir -p " + tiling_path).c_str());
  tiling_path += "/b_rt.so";
  paths.emplace_back(tiling_path);
  system(("touch " + tiling_path).c_str());
  system(("echo 'op tiling 456' > " + tiling_path).c_str());

  proto_path = opp_path + "built-in/op_graph/lib/linux/x86_64";
  system(("mkdir -p " + proto_path).c_str());
  proto_path += "/libopgraph_math.so";
  paths.emplace_back(proto_path);
  system(("touch " + proto_path).c_str());
  system(("echo 'ops proto 123' > " + proto_path).c_str());

  tiling_path = opp_path + "built-in/op_impl/ai_core/tbe/op_host/lib/linux/x86_64";
  system(("mkdir -p " + tiling_path).c_str());
  tiling_path += "/libophost_math.so";
  paths.emplace_back(tiling_path);
  system(("touch " + tiling_path).c_str());
  system(("echo 'op tiling 456' > " + tiling_path).c_str());

  std::string home_path = opp_path.substr(0, opp_path.rfind("/opp/"));
  mmSetEnv(kAscendHomePath, home_path.c_str(), 1);
  std::string opp_latest_path = home_path + "/opp_latest/";
  system(("mkdir -p " + opp_latest_path).c_str());
  paths.emplace_back(opp_latest_path);
  system(("cp -rf " + opp_path + "built-in " + opp_latest_path).c_str());
}
} // gert

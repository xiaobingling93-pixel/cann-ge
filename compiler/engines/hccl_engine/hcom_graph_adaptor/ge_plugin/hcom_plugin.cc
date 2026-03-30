/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcom_plugin.h"
#include <sstream>
#include "ge/ge_api_types.h"            // ge对内options
#include "framework/common/ge_types.h"  // ge对外options
#include "hccl/hcom.h"
#include "hccl/hccl_rank_graph.h"
#include "auto_tuning_plugin.h"
#include "graph/tuning_utils.h"
#include "hcom/hcom_topo_info.h"
#include "mmpa/mmpa_api.h"
#include "offline_build_config_parse.h"
#include "graph/ge_local_context.h"
#include "hcom_op_utils.h"
#include "op_hcom_comm.h"
#include "hcom_log.h"
#include "hcom_executor.h"

namespace hccl {
HcomPlugin::HcomPlugin()
    : initializeCount_(0), hcomOpsKernelInfoStoreInfoPtr_(nullptr), hcomGraphOptimizerPtr_(nullptr) {}

HcomPlugin::~HcomPlugin() {}

HcomPlugin &HcomPlugin::Instance() {
  static HcomPlugin hm;
  /*
   * GE在进程异常退出析构GE内部对象时，会调用HcomPlugin的Finalize接口，触发ge::HcomTopoInfo单例的接口调用。
   * 但是此时ge::HcomTopoInfo单例可能已经析构，导致接口调用时发生进程崩溃。
   * 此处添加ge::HcomTopoInfo的构造函数调用，避免ge::HcomTopoInfo单例对象提前析构
   */
  (void)ge::HcomTopoInfo::Instance();
  return hm;
}

ge::Status HcomPlugin::Initialize(const std::map<string, string> &options) {
  if (initializeCount_ == 0) {
    HCCL_INFO("Initialize start.");
    initializeCount_++;
    HCCL_INFO("initializeCount_ : %d", initializeCount_);
  } else {
    initializeCount_++;
    HCCL_INFO("initializeCount_ : %d", initializeCount_);
    return ge::SUCCESS;
  }
  HcomInitConfig comConfig;
  CHK_PRT_RET(ConfigHcclAlgo(options, &comConfig), HCCL_ERROR("[Initialize][Plugin]ConfigHcclAlgo failed."),
              ge::INTERNAL_ERROR);
  CHK_PRT_RET(ConfigHcclExecTimeOut(options, &comConfig),
              HCCL_ERROR("[Initialize][Plugin]ConfigHcclExecTimeOut failed."), ge::INTERNAL_ERROR);
  CHK_PRT_RET(ConfigHcclDeterministic(options, &comConfig),
              HCCL_ERROR("[Initialize][Plugin]ConfigHcclDeterministic failed."), ge::INTERNAL_ERROR);

  HcomTopoInfoRegCallback(HcomSetGroupToTopoInfo, HcomUnsetGroupToTopoInfo);
  ge::Status geRet = InitializeHcom(options, &comConfig);
  CHK_PRT_RET((geRet != ge::SUCCESS), HCCL_ERROR("[Initialize][Plugin]Initialize Hcom failed"), geRet);
  EXECEPTION_CATCH((hcomOpsKernelInfoStoreInfoPtr_ = std::make_shared<hccl::HcomOpsKernelInfoStore>()),
                   return ge::INTERNAL_ERROR);
#ifndef HCOM_EXECUTOR
  EXECEPTION_CATCH((hcomFusionOptimizerPtr_ = std::make_shared<hccl::HcomFusionOptimizer>()),
                   return ge::INTERNAL_ERROR);
  EXECEPTION_CATCH((hcomGraphOptimizerPtr_ = std::make_shared<hccl::HcomGraphOptimizer>()), return ge::INTERNAL_ERROR);
#endif
  HCCL_INFO("hccl ops plugin init success.");
  return ge::SUCCESS;
}

ge::Status HcomPlugin::Finalize() {
  initializeCount_--;
  if (initializeCount_ == 0) {
    HCCL_INFO("Finalize start.");
  } else {
    return ge::SUCCESS;
  }
  HcclResult ret;
  hcomFusionOptimizerPtr_ = nullptr;
  hcomOpsKernelInfoStoreInfoPtr_ = nullptr;
  hcomGraphOptimizerPtr_ = nullptr;
  ret = HcomDestroy();
  if (ret != HCCL_SUCCESS) {
    HCCL_ERROR("[Finalize][Plugin]errNo[0x%016llx] Finalize: HcomDestroy failed.", HCOM_ERROR_CODE(ret));
    return ge::INTERNAL_ERROR;
  } else {
    HCCL_INFO("Finalize: HcomDestroy success.");
    return ge::SUCCESS;
  }
}

void HcomPlugin::GetOpsKernelInfoStores(map<string, OpsKernelInfoStorePtr> &opKernInfos) {
  HCCL_INFO("get hccl kernel info store start.");
  if (hcomOpsKernelInfoStoreInfoPtr_ != nullptr) {
    opKernInfos.insert(std::make_pair(HCCL_OPS_LIB_NAME, hcomOpsKernelInfoStoreInfoPtr_));
  } else {
    HCCL_ERROR("[Get][OpsKernelget hcom ops kernel info stores ptr failed for nullptr.");
  }
  HCCL_INFO("get hccl kernel info store finished.");
  return;
}

void HcomPlugin::GetOpsKernelInfoPtr(HcomOpsKernelInfoStorePtr &opsKernelInfoStoreInfoPtr) {
  HCCL_INFO("get hccl kernel info ptr start.");
  if (hcomOpsKernelInfoStoreInfoPtr_ != nullptr) {
    opsKernelInfoStoreInfoPtr = hcomOpsKernelInfoStoreInfoPtr_;
  } else {
    HCCL_ERROR("[Get][OpsKernelInfoPtr]get hcom ops kernel info ptr failed for nullptr.");
  }
  HCCL_INFO("get hccl kernel info ptr finished.");
  return;
}

void HcomPlugin::GetGraphOptimizerObjs([[maybe_unused]] map<string, GraphOptimizerPtr> &graphOptimizers) {
  HCCL_INFO("get hccl graph optimizer objs start.");
#ifndef HCOM_EXECUTOR
  if (hcomGraphOptimizerPtr_ != nullptr) {
    graphOptimizers.insert(std::make_pair(HCCL_GRAPH_OPTIMIZER_NAME, hcomGraphOptimizerPtr_));
  } else {
    HCCL_ERROR("[Get][GraphOptimizerObjs]get hcom graph optimizer objs failed for nullptr.");
  }

  if (hcomFusionOptimizerPtr_ != nullptr) {
    graphOptimizers.insert(std::make_pair(HCCL_FUSION_OPTIMIZER_NAME, hcomFusionOptimizerPtr_));
  } else {
    HCCL_ERROR("[Get][GraphOptimizerObjs]get hcom fusion optimizer objs failed for nullptr.");
  }
#endif
  HCCL_INFO("get hccl graph optimizer objs end.");
  return;
}

bool GetMasterInfo(const std::map<string, string> &options, string &masterIp, string &masterPort,
                   string &masterDeviceId) {
  auto iter_ip = options.find(ge::OPTION_EXEC_CM_CHIEF_IP);
  auto iter_Port = options.find(ge::OPTION_EXEC_CM_CHIEF_PORT);
  auto it_DeviceId = options.find(ge::OPTION_EXEC_CM_CHIEF_DEVICE);
  if (iter_ip == options.end() || iter_Port == options.end() || it_DeviceId == options.end()) {
    HCCL_INFO("[masterInfo]get Master Info:ip or port or deviceId not set.");
    return false;
  }
  masterIp = iter_ip->second;
  masterPort = iter_Port->second;
  masterDeviceId = it_DeviceId->second;
  HCCL_INFO("[masterInfo]get Master Info ip[%s], port[%s], rankSize[%s]", masterIp.c_str(), masterPort.c_str(),
            masterDeviceId.c_str());
  return true;
}

bool GetRankInfo(const std::map<string, string> &options, string &rankIp, string &rankSize) {
  auto iter_ip = options.find(ge::OPTION_EXEC_CM_WORKER_IP);
  auto iter_size = options.find(ge::OPTION_EXEC_CM_WORKER_SIZE);
  if (iter_size == options.end()) {  // workIP 若用户未配置，则选择IF_IP或者host网卡进行检索
    HCCL_INFO("[masterInfo]get Rank Info: worker size not set.");
    return false;
  }
  rankSize = iter_size->second;
  if (iter_ip == options.end()) {
    HCCL_INFO("[masterInfo]get rank Info ip[not set], sizes[%s], ", rankSize.c_str());
  } else {
    rankIp = iter_ip->second;
    HCCL_INFO("[masterInfo]get rank Info ip[%s], sizes[%s], ", rankIp.c_str(), rankSize.c_str());
  }

  return true;
}

ge::Status HcomPlugin::ProfilingModeParser(const std::map<string, string> &options) {
  HcclResult ret = HCCL_SUCCESS;
  auto iter = options.find(ge::OPTION_EXEC_PROFILING_MODE);
  if (iter != options.end()) {
    HCCL_INFO("Initialize: (OPTION_EXEC_PROFILING_MODE[%s])", iter->second.c_str());
    if (iter->second == "1") {
      iter = options.find(ge::OPTION_EXEC_PROFILING_OPTIONS);
      if (iter != options.end()) {
        HCCL_INFO("Initialize: (OPTION_EXEC_PROFILING_OPTIONS[%s])", iter->second.c_str());
        if (iter->second.find("task_trace") != iter->second.npos) {
          ret = HcomSetProfilingMode(HcomProfilingMode::PROFILING_OPEN, iter->second.c_str());
          CHK_PRT_RET(ret != HCCL_SUCCESS,
                      HCCL_ERROR("[Init][HcomPlugin]errNo[0x%016llx] Initialize: enable"
                                 "profiling mode failed.",
                                 HCOM_ERROR_CODE(ret)),
                      ge::INTERNAL_ERROR);
          HCCL_INFO("option task_trace is setted. ");
        } else {
          HCCL_WARNING("profiling options profiling task_trace is not found.");
        }
      } else {
        HCCL_INFO("option profiling mode is false. ");
      }
    }
  } else {
    HCCL_WARNING("option profiling mode is not found.");
  }
  return ge::SUCCESS;
}

ge::Status HcomPlugin::InitializeHcom(const std::map<string, string> &options, HcomInitConfig *comConfig) {
  HcclResult ret = HCCL_SUCCESS;
  ge::Status geRet = ProfilingModeParser(options);
  CHK_PRT_RET(geRet != ge::SUCCESS,
              HCCL_ERROR("[Init][HcomPlugin]errNo[0x%016llx] Initialize: ProfilingModeParser "
                         "failed.",
                         HCOM_ERROR_CODE(geRet)),
              ge::INTERNAL_ERROR);
  auto iter = options.find(ge::OPTION_EXEC_RANK_TABLE_FILE);
  if (iter == options.end()) {
    std::string masterIp;
    std::string masterPort;
    std::string masterDeviceId;
    std::string rankSize;
    std::string rankIp;

    bool masterInfoConfiged = GetMasterInfo(options, masterIp, masterPort, masterDeviceId);
    bool rankInfoConfiged = GetRankInfo(options, rankIp, rankSize);
    if (masterInfoConfiged && rankInfoConfiged) {
      ret = HcomInitByMasterInfo(masterIp.c_str(), masterPort.c_str(), masterDeviceId.c_str(), rankSize.c_str(),
                                 rankIp.c_str(), comConfig);
      CHK_PRT_RET(ret != HCCL_SUCCESS,
                  HCCL_ERROR("[Init][HcomPlugin]errNo[0x%016llx] Initialize: HcomInitByMasterInfo failed.",
                             HCOM_ERROR_CODE(ret)),
                  ge::INTERNAL_ERROR);
    } else {
      HCCL_INFO("InitializePlugin: Init without ranktable and masterInfo, please check your mode.");
      return ge::SUCCESS;
    }
  } else {
    char *mmSysGetEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ESCLUSTER_CONFIG_PATH, mmSysGetEnvValue);
    std::string esClusterConfig = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
    if (esClusterConfig != "EmptyString") {
      HCCL_INFO("InitializePlugin: ESCLUSTER_CONFIG_PATH exists.");
      return ge::SUCCESS;
    }
    std::string rankTable = iter->second;
    // 优先使用RANK_ID; RANK_ID没设置再读POD_NAME;
    std::string identify;
    iter = options.find(ge::OPTION_EXEC_RANK_ID);
    if (iter != options.end()) {
      identify = iter->second;
    } else {
      iter = options.find(ge::OPTION_EXEC_POD_NAME);
      if (iter != options.end()) {
        identify = iter->second;
      } else {
        HCCL_ERROR("[Init][HcomPlugin]Initialize failed, not set RANK_ID or POD_NAME");
        return ge::INTERNAL_ERROR;
      }
    }
    HCCL_INFO("initialize hccl by rank table[%s], identify[%s]", rankTable.c_str(), identify.c_str());

    std::string rankTableM;
    ret = HcomLoadRanktableFile(rankTable, rankTableM);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[GetRanktable] rankTablePath[%s]"
                           "load rankTable error.",
                           rankTable.c_str()),
                ge::INTERNAL_ERROR);
    ret = HcomInitByString(rankTableM.c_str(), identify.c_str(), WorkMode::HCCL_MODE_NORMAL, comConfig);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][HcomPlugin]errNo[0x%016llx] Initialize: HcomInitByString failed.", HCOM_ERROR_CODE(ret)),
        ge::INTERNAL_ERROR);
  }
  return ge::SUCCESS;
}

HcclResult HcomPlugin::ConfigHcclExecTimeOut(const std::map<string, string> &options, HcomInitConfig *comConfig) {
  auto iterOption = options.find("ge.exec.hcclExecuteTimeOut");
  if (iterOption == options.end()) {
    HCCL_INFO("ParserHcclExecTimeOut: there is no key named \"%s\" in the options.", "HCCL_execTimeOut");
    return HCCL_SUCCESS;
  }

  if (iterOption->second.empty()) {
    HCCL_WARNING("ParserHcclExecTimeOut: key[ge.exec.hcclExecuteTimeOut] has no value, use default setting");
    return HCCL_SUCCESS;
  }

  comConfig->execTimeOut = const_cast<char *>(iterOption->second.c_str());
  HCCL_INFO("ParserHcclExecTimeOut: key[ge.exec.hcclExecuteTimeOut] value[%s]", comConfig->execTimeOut);
  return HCCL_SUCCESS;
}

HcclResult HcomPlugin::ConfigHcclDeterministic(const std::map<string, string> &options, HcomInitConfig *comConfig) {
  auto iter = options.find(ge::DETERMINISTIC);
  if (iter != options.end()) {
    HCCL_INFO("[Init][HcomPlugin]: (DETERMINISTIC[%s])", iter->second.c_str());
    char *mmSysGetEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_DETERMINISTIC, mmSysGetEnvValue);
    std::string hcclDeterministicEnv = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
    if (hcclDeterministicEnv == "EmptyString") {
      if (iter->second.empty()) {
        HCCL_WARNING("ParserHcclDeterministicDesc: key[ge.deterministic] has no value, use default setting");
        return HCCL_SUCCESS;
      } else if (iter->second == "2") {
        comConfig->deterministic = static_cast<u8>(DETERMINISTIC_STRICT);
        HCCL_INFO("ParserHcclDeterministicDesc: key[ge.deterministic] config: DETERMINISTIC is strict");
      } else if (iter->second == "1") {
        comConfig->deterministic = static_cast<u8>(DETERMINISTIC_ENABLE);
        HCCL_INFO("ParserHcclDeterministicDesc: key[ge.deterministic] config: DETERMINISTIC is true");
      } else if (iter->second == "0") {
        comConfig->deterministic = static_cast<u8>(DETERMINISTIC_DISABLE);
        HCCL_INFO("ParserHcclDeterministicDesc: key[ge.deterministic] config: DETERMINISTIC is false");
      }
    } else {
      HCCL_WARNING(
          "ParserHcclDeterministicDesc: key[ge.deterministic] has been set by"
          "HCCL_DETERMINISTIC Env, so will not be reset again");
      return HCCL_SUCCESS;
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomPlugin::ConfigHcclAlgo(const std::map<string, string> &options, HcomInitConfig *comConfig) {
  auto iterOption = options.find("HCCL_algorithm");
  if (iterOption == options.end()) {
    HCCL_INFO("ParserHcclAlgoDesc: there is no key named \"%s\" in the options.", "HCCL_algorithm");
    return HCCL_SUCCESS;
  }

  if (iterOption->second.empty()) {
    HCCL_WARNING("ParserHcclAlgoDesc: key[HCCL_algorithm] has no value, use default setting");
    return HCCL_SUCCESS;
  }

  comConfig->algo = const_cast<char *>(iterOption->second.c_str());
  HCCL_INFO("ParserHcclExecTimeOut: key[HCCL_algorithm] value[%s]", comConfig->algo);
  return HCCL_SUCCESS;
}

HcclResult HcomPlugin::HcomSetGroupToTopoInfo(const char *group, uint32_t rankSize) {
  if (group == nullptr) {
    HCCL_ERROR("[Set][GroupTopoInfo]SetGroupTopoInfo group is null");
    return HCCL_E_PTR;
  }
  HCCL_INFO("[Set][GroupTopoInfo]group[%s] rankSize[%u].", group, rankSize);
  HcclResult ret;
  ge::HcomTopoInfo::TopoInfo topoInfo;
  topoInfo.rank_size = rankSize;

  HcclComm commHandle;
  ret = HcomGetCommHandleByGroup(group, &commHandle);
  if (ret != HCCL_SUCCESS) {
    return ret;
  }

  u32 gRankSize;
  CommTopo topoType;
  u32 netLayerNum = 0;
  u32 *netLayer = nullptr;

  ret = HcclRankGraphGetLayers(commHandle, &netLayer, &netLayerNum);
  if (ret != HCCL_SUCCESS) return ret;
  for (u32 i = 0; i < netLayerNum; i++) {
    ret = HcclRankGraphGetRankSizeByLayer(commHandle, netLayer[i], &gRankSize);
    if (ret != HCCL_SUCCESS) return ret;
    ret = HcclRankGraphGetTopoTypeByLayer(commHandle, netLayer[i], &topoType);
    if (ret != HCCL_SUCCESS) return ret;
    topoInfo.topo_level_descs[netLayer[i]].comm_sets = static_cast<uint32_t>(topoType);
    topoInfo.topo_level_descs[netLayer[i]].rank_size = gRankSize;
  }
  uint64_t localWindowSize;
  CHK_RET(HcomGetCommCCLBufferSize(group, localWindowSize));
  topoInfo.local_window_size = localWindowSize;
  HCCL_RUN_INFO("[Set][GroupTopoInfo]localWindowSize[%lu]", localWindowSize);

  uint32_t retInstance = ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group, topoInfo);
  if (retInstance != ge::GRAPH_SUCCESS) {
    HCCL_ERROR("[Set][GroupTopoInfo]errNo[0x%016llx] SetGroupTopoInfo error, group[%s], rankSize[%u].", retInstance,
               group, topoInfo.rank_size);
    return HCCL_E_NOT_FOUND;
  }
  HCCL_INFO("[Set][GroupTopoInfo]SetGroupTopoInfo group set success");
  return HCCL_SUCCESS;
}

void HcomPlugin::HcomUnsetGroupToTopoInfo(const char *group) {
  if (group == nullptr) {
    HCCL_ERROR("[Unset][GroupTopoInfo]UnsetGroupTopoInfo group is null");
    return;
  }
  HCCL_INFO("[Unset][GroupTopoInfo]group[%s].", group);
  ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group);
}
}  // namespace hccl
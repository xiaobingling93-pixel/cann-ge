/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_hcom_comm.h"
#include "graph/ge_local_context.h"
#include "hcom_ops_kernel_info_store.h"
#include "framework/common/ge_types.h"  // ge对外options
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "hccl/hcom.h"
#include "mmpa/mmpa_api.h"
#include <pthread.h>

namespace hccl {

HcclResult HcomInitialize() {
  bool isInited = false;
  HcomGetInitStatus(&isInited);
  if (isInited) {
    return HCCL_SUCCESS;
  }

  std::string rankTableAddrStr;
  std::string rankTableLenStr;
  ge::graphStatus geRet = ge::GetThreadLocalContext().GetOption(ge::OPTION_EXEC_RANK_TABLE_ADDR, rankTableAddrStr);
  CHK_PRT_RET((geRet != ge::GRAPH_SUCCESS),
              HCCL_WARNING("[HcomOpsKernelInfoStore][InitHcom]OPTION_EXEC_RANK_TABLE is not found."), HCCL_SUCCESS);
  geRet = ge::GetThreadLocalContext().GetOption(ge::OPTION_EXEC_RANK_TABLE_LEN, rankTableLenStr);
  CHK_PRT_RET((geRet != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[HcomOpsKernelInfoStore][InitHcom]Failed to get OPTION_EXEC_RANK_TABLE_LEN."),
              HCCL_E_INTERNAL);

  u64 rankTableAddr;
  u32 rankTableLen;
  CHK_RET(SalStrToULonglong(rankTableAddrStr, HCCL_BASE_DECIMAL, rankTableAddr));
  CHK_RET(SalStrToULong(rankTableLenStr, HCCL_BASE_DECIMAL, rankTableLen));
  CHK_PRT_RET((rankTableLen > RANK_TABLE_MAX_LEN) || (rankTableLen == 0),
              HCCL_ERROR("[HcomOpsKernelInfoStore][InitHcom]errNo[0x%016llx] rankTable file is invalid, len is %u",
                         HCOM_ERROR_CODE(HCCL_E_PARA), rankTableLen),
              HCCL_E_PARA);
  std::string rankTable(reinterpret_cast<char *>(rankTableAddr), rankTableLen);

  std::string rankId;
  geRet = ge::GetThreadLocalContext().GetOption(ge::OPTION_EXEC_RANK_ID, rankId);
  CHK_PRT_RET((geRet != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[HcomOpsKernelInfoStore][InitHcom]Failed to get OPTION_EXEC_RANK_ID."), HCCL_E_INTERNAL);

  CHK_RET(HcomInitByString(rankTable.c_str(), rankId.c_str()));
  return HCCL_SUCCESS;
}

HcclResult InitGroup() {
  std::string groupConf;
  nlohmann::json groupListConf;
  ge::graphStatus geRet = ge::GetThreadLocalContext().GetOption(ge::OPTION_EXEC_HCOM_GROUPLIST, groupConf);
  CHK_PRT_RET((geRet != ge::GRAPH_SUCCESS),
              HCCL_WARNING("[HcomOpsKernelInfoStore][InitHcom]OPTION_EXEC_HCOM_GROUPLIST is not found."), HCCL_SUCCESS);
  HCCL_DEBUG("groupList:%s", groupConf.c_str());
  CHK_RET(SalParseInformation(groupListConf, groupConf));
  std::vector<nlohmann::json> groupList = groupListConf.get<std::vector<nlohmann::json>>();
  for (auto &groupInfo : groupList) {
    HCCL_DEBUG("groupInfo:%s", groupInfo.dump().c_str());
    std::vector<u32> ranks = groupInfo["group_rank_list"];
    std::string groupName = groupInfo["group_name"];
    HCCL_DEBUG("groupName:%s", groupName.c_str());

    u32 curRank = 0;
    CHK_RET(HcomGetRankId(HCCL_WORLD_GROUP, &curRank));
    if (!HcomFindGroup(groupName.c_str()) && find(ranks.begin(), ranks.end(), curRank) != ranks.end()) {
      if (strncmp(groupName.c_str(), HCCL_WORLD_GROUP, sizeof(HCCL_WORLD_GROUP)) == 0) {
        HCCL_WARNING("[HcomOpsKernelInfoStore][InitHcom]cur groupname is HCCL_WORLD_GROUP.");
        continue;
      }
      CHK_RET(HcomCreateGroup(groupName.c_str(), ranks.size(), ranks.data()));
    }
  }
  return HCCL_SUCCESS;
}

HcclResult GetOpDescIntAttr(const ge::OpDesc &op, const string &attr, s32 &output) {
  if (ge::AttrUtils::HasAttr(op, attr)) {
    ge::AttrUtils::GetInt(op, attr, output);
  } else {
    return HCCL_E_NOT_FOUND;
  }
  return HCCL_SUCCESS;
}

HcclResult GetOpDescStrAttr(const ge::OpDesc &op, const string &attr, string &output) {
  if (ge::AttrUtils::HasAttr(op, attr)) {
    ge::AttrUtils::GetStr(op, attr, output);
  } else {
    return HCCL_E_NOT_FOUND;
  }
  return HCCL_SUCCESS;
}

HcclResult SetWorkspaceResource(const std::string &tag, const char *group, std::vector<rtStream_t> stream, void *memPtr,
                                u64 maxSize) {
  rtStream_t *streamPtr = static_cast<rtStream_t *>(stream.data());
  s32 streamNum = stream.size();
  CHK_RET(HcomSetWorkspaceResource(tag.c_str(), group, streamPtr, streamNum, memPtr, maxSize));
  return HCCL_SUCCESS;
}

HcclResult GetCCLBufferAvailableSize(u64 &size) {
  // HCCL校验过该环境变量，这里直接读取
  char *mmSysGetEnvValue = nullptr;
  MM_SYS_GET_ENV(MM_ENV_HCCL_BUFFSIZE, mmSysGetEnvValue);
  std::string hcclBufferSize = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
  u32 cclBufferSize = HCCL_CCL_COMM_DEFAULT_BUFFER_SIZE;
  if (hcclBufferSize != "EmptyString") {
    CHK_RET(SalStrToULong(hcclBufferSize.c_str(), HCCL_BASE_DECIMAL, cclBufferSize));
  }
  size =
      static_cast<u64>(cclBufferSize * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE) - CCL_COMM_INBUFFER_UNALIGNED_RESERVE_SIZE;
  return HCCL_SUCCESS;
}

constexpr u32 HCCL_RANKTABLE_TIMEOUT_S = (30 * 60);  // 读取ranktable json文件超时时间30 * 60s

bool CheckFilePath(const std::string &filePath, const std::string &fileType) {
  HCCL_DEBUG("file_path:%s,file_type:%s", filePath.c_str(), fileType.c_str());

  /* 如果file_path是file_type类型的文件，则file_path是以file_type结尾的 */
  return filePath.find(fileType) + fileType.length() == filePath.length();
}

HcclResult HcomLoadRanktableFile(const std::string &rankTablePath, std::string &rankTableM) {
  HcclResult ret;
  if (rankTablePath.empty()) {
    REPORT_PREDEFINED_ERR_MSG(
        "EI0004", std::vector<const char *>({"error_reason", "ranktable_path"}),
        std::vector<const char *>({"Ranktable json file length is zero.", rankTablePath.c_str()}));
    HCCL_ERROR("[Load][File] json file length is zero");
    return HCCL_E_PARA;
  }

  /* 如果file_path是file_type类型的文件，则file_path是以file_type结尾的 */
  std::string fileType = ".json";
  if (!CheckFilePath(rankTablePath, fileType)) {
    REPORT_PREDEFINED_ERR_MSG("EI0004", std::vector<const char *>({"error_reason", "ranktable_path"}),
                              std::vector<const char *>({"Ranktable file name is invalid.", rankTablePath.c_str()}));
    HCCL_ERROR("[Load][File] path %s is not a valid %s file", rankTablePath.c_str(), fileType.c_str());
    return HCCL_E_PARA;
  }

  // 打开该文件前，判断该文件路径是否有效 规范
  std::string realFilePath;
  ret = HcomGetRanktableRealPath(rankTablePath.c_str(), realFilePath);
  CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Load][File] get file[%s] real path error", rankTablePath.c_str()),
              HCCL_E_PARA);

  const std::chrono::seconds TIMEOUT(HCCL_RANKTABLE_TIMEOUT_S);
  auto startTime = std::chrono::steady_clock::now();
  HCCL_INFO("waiting for ranktable load complete...");
  nlohmann::json fileContent;  // json文件的内容

  // 实验室场景下总是completed，cloud场景初始填入initlizing，kube补充完整后成为completed状态
  do {
    if ((std::chrono::steady_clock::now() - startTime) >= TIMEOUT) {
      HCCL_ERROR("[Load][File] Load ranktable file[%s] timeout[%lld]s", realFilePath.c_str(), TIMEOUT);
      return HCCL_E_TIMEOUT;
    }
    // 读取文件内容
    ret = ReadFile(realFilePath, fileContent);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Load][File]read file[%s] error", realFilePath.c_str()), HCCL_E_PARA);

    std::string status = "";
    CHK_RET(GetJsonProperty(fileContent, "status", status));
    if (status == "completed") {
      break;
    }

    SalSleep(1);  // 每间隔一段时间去检测文件是否ready
  } while (true);

  rankTableM = fileContent.dump();
  HCCL_INFO("ranktable is ready");

  return HCCL_SUCCESS;
}

HcclResult ReadFile(const std::string &readFile, nlohmann::json &fileContent) {
  // 已只读方式打开该文件
  std::ifstream infile(readFile.c_str(), std::ifstream::in);
  if (!infile) {
    HCCL_ERROR("[Read][File] open file %s failed", readFile.c_str());
    return HCCL_E_INTERNAL;
  } else {
    fileContent.clear();
    try {
      infile >> fileContent;  // 将文件内容读取到json对象内
    } catch (...) {
      REPORT_PREDEFINED_ERR_MSG("EI0004", std::vector<const char *>({"error_reason", "ranktable_path"}),
                                std::vector<const char *>({"Invalid ranktable format.", readFile.c_str()}));
      HCCL_ERROR("[Read][File] load file[%s] to json fail. please check json file!", readFile.c_str());
      infile.close();
      return HCCL_E_INTERNAL;
    }
  }
  infile.close();
  return HCCL_SUCCESS;
}

HcclResult HcomGetRanktableRealPath(const char *rankTable, std::string &realFilePath) {
  CHK_PTR_NULL(rankTable);

  u32 rankTablePathLen = strnlen(rankTable, RANK_TABLE_MAX_LEN + 1);
  if (rankTablePathLen == (RANK_TABLE_MAX_LEN + 1) || rankTablePathLen == 0) {
    REPORT_PREDEFINED_ERR_MSG("EI0004", std::vector<const char *>({"error_reason", "ranktable_path"}),
                              std::vector<const char *>({"Ranktable file name is invalid.", rankTable}));
    HCCL_ERROR("[Get][RanktableRealPath]errNo[0x%016llx] rankTable file name is invalid, len is %u",
               HCOM_ERROR_CODE(HCCL_E_PARA), rankTablePathLen);
    return HCCL_E_PARA;
  }
  // 校验文件是否存在
  char realFile[PATH_MAX] = {0};
  if (realpath(rankTable, realFile) == nullptr) {
    REPORT_PREDEFINED_ERR_MSG("EI0004", std::vector<const char *>({"error_reason", "ranktable_path"}),
                              std::vector<const char *>({"Ranktable path is not a valid real path.", rankTable}));
    HCCL_ERROR("[Get][RanktableRealPath]errNo[0x%016llx] path %s is not a valid real path",
               HCOM_ERROR_CODE(HCCL_E_PARA), rankTable);
    return HCCL_E_PARA;
  }
  realFilePath = std::string(realFile);
  return HCCL_SUCCESS;
}

void SalSleep(u32 sec) {
  /* sleep()可能会因为进程收到信号(比如alarm)而提前返回EINTR, 后续优化  */
  s32 iRet = sleep(sec);
  if (iRet != 0) {
    HCCL_WARNING("Sleep: sleep failed[%d]: %s [%d]", iRet, strerror(errno), errno);
  }
}

HcclResult GetJsonProperty(const nlohmann::json &obj, const char *propName, std::string &propValue) {
  /* 查找json对象中是否有该属性, 不存在的属性不能直接访问 */
  if (obj.find(propName) == obj.end()) {
    HCCL_WARNING("json object has no property called %s", propName);
    return HCCL_E_PARA;
  }
  propValue = obj[propName];
  CHK_PRT_RET(propValue.size() == 0, HCCL_ERROR("[Get][JsonProperty]get property[%s] size is zero", propName),
              HCCL_E_PARA);
  return HCCL_SUCCESS;
}

HcclResult GetInCCLbuffer(const char *group, void *&buffer, u64 &size) {
  CHK_RET(HcomGetInCCLbuffer(group, &buffer, &size));
  return HCCL_SUCCESS;
}

HcclResult GetOutCCLbuffer(const char *group, void *&buffer, u64 &size) {
  CHK_RET(HcomGetOutCCLbuffer(group, &buffer, &size));
  return HCCL_SUCCESS;
}

bool IsSocVersion910(std::string socVersion) {
  return socVersion == "Ascend910" || socVersion == "Ascend910A" || socVersion == "Ascend910B" ||
         socVersion == "Ascend910ProA" || socVersion == "Ascend910ProB" || socVersion == "Ascend910PremiumA";
}

bool IsSocVersion910B(std::string socVersion) {
  return socVersion == "Ascend910B1" || socVersion == "Ascend910B2" || socVersion == "Ascend910B2C" ||
         socVersion == "Ascend910B3" || socVersion == "Ascend910B4" || socVersion == "Ascend910B4-1";
}

bool IsSocVersion91093(std::string socVersion) {
  return socVersion == "Ascend910_9391" || socVersion == "Ascend910_9381" || socVersion == "Ascend910_9392" ||
         socVersion == "Ascend910_9382" || socVersion == "Ascend910_9372" || socVersion == "Ascend910_9362";
}

void SetThreadName(const std::string &threadStr) {
  // 线程名应限制在15个字符内，防止被截断
  s32 sRet = pthread_setname_np(pthread_self(), threadStr.c_str());
  CHK_PRT_CONT(sRet != 0, HCCL_WARNING("err[%d] link[%s] nameSet failed.", sRet, threadStr.c_str()));
}

// 字符串转换成整型
HcclResult SalStrToInt(const std::string str, int base, s32 &val) {
  try {
    val = std::stoi(str, nullptr, base);
  } catch (std::invalid_argument &) {
    HCCL_ERROR("[Transform][StrToInt]strtoi invalid argument, str[%s] base[%d] val[%d]", str.c_str(), base, val);
    return HCCL_E_PARA;
  } catch (std::out_of_range &) {
    HCCL_ERROR("[Transform][StrToInt]strtoi out of range, str[%s] base[%d] val[%d]", str.c_str(), base, val);
    return HCCL_E_PARA;
  } catch (...) {
    HCCL_ERROR("[Transform][StrToInt]strtoi catch error, str[%s] base[%d] val[%d]", str.c_str(), base, val);
    return HCCL_E_PARA;
  }
  return HCCL_SUCCESS;
}

// 字串符转换成无符号整型
HcclResult SalStrToULong(const std::string str, int base, u32 &val) {
  try {
    u64 tmp = std::stoull(str, nullptr, base);
    if (tmp > INVALID_UINT) {
      HCCL_ERROR("[Transform][StrToULong]stoul out of range, str[%s] base[%d] val[%llu]", str.c_str(), base, tmp);
      return HCCL_E_PARA;
    } else {
      val = static_cast<u32>(tmp);
    }
  } catch (std::invalid_argument &) {
    HCCL_ERROR("[Transform][StrToULong]stoull invalid argument, str[%s] base[%d] val[%u]", str.c_str(), base, val);
    return HCCL_E_PARA;
  } catch (std::out_of_range &) {
    HCCL_ERROR("[Transform][StrToULong]stoull out of range, str[%s] base[%d] val[%u]", str.c_str(), base, val);
    return HCCL_E_PARA;
  } catch (...) {
    HCCL_ERROR("[Transform][StrToULong]stoull catch error, str[%s] base[%d] val[%u]", str.c_str(), base, val);
    return HCCL_E_PARA;
  }
  return HCCL_SUCCESS;
}

// 字串符转换成无符号长整型
HcclResult SalStrToULonglong(const std::string str, int base, u64 &val) {
  try {
    val = std::stoull(str, nullptr, base);
  } catch (std::invalid_argument &) {
    HCCL_ERROR("[Transform][StrToULonglong]stoull invalid argument, str[%s] base[%d] val[%llu]", str.c_str(), base,
               val);
    return HCCL_E_PARA;
  } catch (std::out_of_range &) {
    HCCL_ERROR("[Transform][StrToULonglong]stoull out of range, str[%s] base[%d] val[%llu]", str.c_str(), base, val);
    return HCCL_E_PARA;
  } catch (...) {
    HCCL_ERROR("[Transform][StrToULonglong]stoull catch error, str[%s] base[%d] val[%llu]", str.c_str(), base, val);
    return HCCL_E_PARA;
  }
  return HCCL_SUCCESS;
}

HcclResult SalGetDataTypeSize(HcclDataType dataType, u32 &dataTypeSize) {
  if ((dataType >= HCCL_DATA_TYPE_INT8) && (dataType < HCCL_DATA_TYPE_RESERVED)) {
    dataTypeSize = SIZE_TABLE[dataType];
  } else {
    HCCL_ERROR("[Get][DataTypeSize]errNo[0x%016llx] get date size failed. dataType[%s] is invalid.",
               HCOM_ERROR_CODE(HCCL_E_PARA), GetDataTypeEnumStr(dataType).c_str());
    return HCCL_E_PARA;
  }
  return HCCL_SUCCESS;
}

HcclResult SalParseInformation(nlohmann::json &parseInformation, const std::string &information) {
  try {
    parseInformation = nlohmann::json::parse(information);
  } catch (...) {
    HCCL_ERROR(
        "[Parse][Information] errNo[0x%016llx] load allocated resource to json fail. "
        "please check json input!",
        HCOM_ERROR_CODE(HCCL_E_PARA));
    return HCCL_E_PARA;
  }
  return HCCL_SUCCESS;
}
}  // namespace hccl
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include "adx_datadump_server.h"

#include "dump_stub.h"
#include "adump_pub.h"
#include "adump_api.h"
#include "runtime/base.h"

namespace ge {
DumpStub &DumpStub::GetInstance() {
  static DumpStub dump_stub;
  return dump_stub;
}

void DumpStub::AddOpInfo(const Adx::OperatorInfoV2 &info) {
  std::lock_guard<std::mutex> lk(mu_);
  dump_op_infos_.emplace_back(info);
  dump_op_tensors_.emplace_back(std::vector<gert::Tensor>());
  auto &tensors = dump_op_tensors_.back();
  auto &op_info = dump_op_infos_.back();
  tensors.resize(op_info.tensorInfos.size());

  for (size_t i = 0; i < op_info.tensorInfos.size(); ++i) {
    auto &tensor = tensors[i];
    auto &tensor_info = op_info.tensorInfos[i];
    tensor.SetDataType(static_cast<ge::DataType>(tensor_info.dataType));
    tensor.SetPlacement(static_cast<gert::TensorPlacement>(tensor_info.placement));
    tensor.MutableTensorData().SetAddr(tensor_info.tensorAddr, nullptr);
    tensor.MutableTensorData().SetSize(tensor_info.tensorSize);
  }
}

int32_t DumpStub::SetDumpConfig(Adx::DumpType dump_type, const Adx::DumpConfig &dump_config) {
  std::lock_guard<std::mutex> lk(mu_);
  const auto ret = ge::DumpStub::GetInstance().GetFuncRet("SetDumpConfig", 0);
  if (ret != 0) {
    return ret;
  }

  // remove
  if (dump_config.dumpStatus != "on") {
    if (dump_configs_.find(dump_type) != dump_configs_.end()) {
      dump_configs_.erase(dump_type);
    }
    return 0;
  }

  dump_configs_[dump_type] = dump_config;
  return 0;
}

std::string AdxGetAdditionalInfo(const Adx::OperatorInfoV2 &info, const std::string key) {
  const auto &iter = info.additionalInfo.find(key);
  if (iter != info.additionalInfo.end()) {
    return iter->second;
  }
  return "";
}

std::string AdxGetTilingKey(const Adx::OperatorInfoV2 &info) {
  return AdxGetAdditionalInfo(info, Adx::DUMP_ADDITIONAL_TILING_KEY);
}

bool AdxGetArgsInfo(const Adx::OperatorInfoV2 &info, void *&addr, uint64_t &length) {
  if (info.deviceInfos.size() == 0) {
    return false;
  }

  for (const auto &device_info : info.deviceInfos) {
    if (device_info.name == Adx::DEVICE_INFO_NAME_ARGS) {
      addr = device_info.addr;
      length = device_info.length;
      return true;
    }
  }
  return false;
}

bool AdxGetWorkspaceInfo(const Adx::OperatorInfoV2 &info, uint32_t index, void *&addr, uint64_t &length) {
  uint32_t i = 0;
  for (const auto &tensorInfo : info.tensorInfos) {
    if (tensorInfo.type == Adx::TensorType::WORKSPACE) {
      if (i == index) {
        addr = tensorInfo.tensorAddr;
        length = tensorInfo.tensorSize;
        return true;
      }
      ++i;
    }
  }
  return false;
}
}  // namespace ge

int AdxDataDumpServerUnInit() {
  return 0;
}

int AdxDataDumpServerInit() {
  return 0;
}

namespace Adx {
void *AdumpGetSizeInfoAddr(uint32_t space, uint32_t &atomicIndex) {
  (void)space;
  return ge::DumpStub::GetInstance().Get(space, atomicIndex);
}

int32_t AdumpSetDumpConfig(DumpType dumpType, const DumpConfig &dumpConfig) {
  return ge::DumpStub::GetInstance().SetDumpConfig(dumpType, dumpConfig);
}

void AdumpPrintWorkSpace(const void *work_addr, const size_t work_size, rtStream_t stream, const char *opType) {
  (void)work_addr;
  (void)work_size;
  (void)stream;
  (void)opType;
  return;
}

void *AdumpGetDFXInfoAddrForStatic(uint32_t space, uint64_t &atomicIndex) {
  (void)space;
  return ge::DumpStub::GetInstance().GetStatic(space, atomicIndex);
}

void *AdumpGetDFXInfoAddrForDynamic(uint32_t space, uint64_t &atomicIndex) {
  (void)space;
  return ge::DumpStub::GetInstance().GetDynamic(space, atomicIndex);
}

uint64_t AdumpGetDumpSwitch(const Adx::DumpType dumpType) {
  (void)dumpType;
  return ge::DumpStub::GetInstance().GetEnableFlag();
}

int32_t AdumpAddExceptionOperatorInfoV2(const OperatorInfoV2 &opInfo) {
  ge::DumpStub::GetInstance().AddOpInfo(opInfo);
  return ge::DumpStub::GetInstance().GetFuncRet("AdumpAddExceptionOperatorInfoV2", 0);
}

int32_t AdumpDelExceptionOperatorInfo(uint32_t deviceId, uint32_t streamId) {
  bool is_found = ge::DumpStub::GetInstance().DelOpInfo(deviceId, streamId);
  return is_found ? ge::DumpStub::GetInstance().GetFuncRet("AdumpDelExceptionOperatorInfo", 0) : -1;
}

int32_t AdumpRegisterCallback(uint32_t module_id, AdumpCallback enable_callback, AdumpCallback disable_callback) {
    ge::DumpStub::GetInstance().RecordCall("AdumpRegisterCallback", module_id);

    return ge::DumpStub::GetInstance().GetFuncRet("AdumpRegisterCallback", 0);
}
}  // namespace Adx

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <climits>
#include "communication.h"
#include "topology.h"
#include "hccl/hcom.h"
#include "cluster.h"

using namespace std;

struct ModelPara {
  string serverModel;
  string productForm;
  int devNumInServer;
  int serverNum;
  int serverPortNum;
};
string GetModelPara(string workPath, string str) {
  char charbuf[MAX_BUF_SIZE];
  size_t found;
  string ret;
  char realWorkPath[PATH_MAX + 1] = {0};  // lint !e813

  if ((workPath.size() > PATH_MAX) || (realpath(workPath.c_str(), realWorkPath) == nullptr)) {
    HCCL_WARNING("path_len[%u] > [%d](PATH_MAX) or workPath[%s] is invalid, err[%d]", workPath.size(), PATH_MAX,
                 workPath.c_str(), errno);
    return ret;
  }
  std::ifstream fileStream(realWorkPath);
  CHK_PRT_RET(!fileStream.is_open(), HCCL_WARNING("WARNING: open cluster file fail, using default para"), ret);
  while (!fileStream.eof()) {
    fileStream.getline(charbuf, MAX_BUF_SIZE);
    string buffer(charbuf);
    if (buffer.find(str) != string::npos) {
      found = buffer.find(':');
      if (found != string::npos) {
        ret = buffer.substr(found + 1);
        fileStream.close();
        return ret;
      }
    }
  }
  fileStream.close();
  return ret;
}

void InitDefaultTopoInfo(struct TopoInfo &topoFirst, struct TopoInfo &topoSecond) {
  topoFirst.topoType = "ring";
  topoFirst.algorithm = "ring";
  topoFirst.topoStackNum = DEFAULT_TOPOFIRST_STACK_NUM;
  topoFirst.gpuNum = DEFAULT_TOPOFIRST_GPU_NUM;
  topoFirst.bwPerGpu = DEFAULT_TOPOFIRST_BW_PER_GPU;
  topoFirst.bwPortPerGpu = DEFAULT_TOPOFIRST_BW_PORT_PER_GPU;
  topoFirst.syncCostPerPort = DEFAULT_TOPOFIRST_SYNC_COST_PER_PORT;
  topoFirst.costSync = DEFAULT_TOPOFIRST_COST_SYNC;
  topoFirst.deviceMemory = DEFAULT_TOPOFIRST_DEVICE_MEMORY;
  topoFirst.bwComputation = DEFAULT_TOPOFIRST_BW_COMPUTATION;
  topoFirst.fixedJitter = DEFAULT_TOPOFIRST_FIXED_JITTER;

  topoSecond.topoType = "star";
  topoSecond.algorithm = "H-D";
  topoSecond.topoStackNum = DEFAULT_TOPOSECOND_STACK_NUM;
  topoSecond.gpuNum = DEFAULT_TOPOSECOND_GPU_NUM;
  topoSecond.bwPerGpu = DEFAULT_TOPOSECOND_BW_PER_GPU;
  topoSecond.bwPortPerGpu = DEFAULT_TOPOSECOND_BW_PORT_PER_GPU;
  topoSecond.syncCostPerPort = DEFAULT_TOPOSECOND_SYNC_COST_PER_PORT;
  topoSecond.costSync = DEFAULT_TOPOSECOND_COST_SYNC;
  topoSecond.deviceMemory = DEFAULT_TOPOSECOND_DEVICE_MEMORY;
  topoSecond.bwComputation = DEFAULT_TOPOSECOND_BW_COMPUTATION;
  topoSecond.fixedJitter = DEFAULT_TOPOSECOND_FIXED_JITTER;
}

void InitModelPara(struct ModelPara &modelPara) {
  HcclResult ret;
  u32 serverNum;
  u32 deviceNumPerServer;
  u32 deviceNumPerAggregation;
  ret = HcomGetServerNumAndDeviceNumPerServer(&serverNum, &deviceNumPerServer, &deviceNumPerAggregation);
  if (ret != HCCL_SUCCESS) {
    HCCL_ERROR("[TUNE]Get server and device num failed.");
  }

  if (deviceNumPerServer != deviceNumPerAggregation) {
    serverNum = serverNum * (deviceNumPerServer / deviceNumPerAggregation);
  }

  HCCL_INFO("[Get][HierarchicalPolicy] Number of CommOuters is [%u], Number of devices in a CommOuter is [%u]",
            serverNum, deviceNumPerAggregation);

  modelPara.serverModel = "G560 V5";

  modelPara.productForm = "module";

  modelPara.devNumInServer = deviceNumPerAggregation;

  modelPara.serverNum = serverNum;

  modelPara.serverPortNum = DEFAULT_SERVER_PORT_NUM;
}

void GetTopoInfoFromFile(vector<TopoInfo> &toposInfo) {
  struct ModelPara modelPara;
  struct TopoInfo topoFirst;
  struct TopoInfo topoSecond;

  InitModelPara(modelPara);
  InitDefaultTopoInfo(topoFirst, topoSecond);

  char *algo = nullptr;
  HcclResult ret = HcomGetAlgorithm(0, &algo);
  topoFirst.algorithm = algo;
  if (topoFirst.algorithm != "ring" && topoFirst.algorithm != "mesh") {
    HCCL_WARNING("[GDAT][GetTopoInfoFromFile] No matching Algo found in first level, use ring");
    topoFirst.algorithm = "ring";  // 匹配不到支持的算法，暂时使用ring算法默认
  }
  ret = HcomGetAlgorithm(1, &algo);
  topoFirst.algorithm = algo;
  if (topoSecond.algorithm != "ring" && topoSecond.algorithm != "H-D") {
    HCCL_WARNING("[GDAT][GetTopoInfoFromFile] No matching Algo found in second level, use H-D");
    topoFirst.algorithm = "H-D";  // 匹配不到支持的算法，暂时使用H-D算法默认
  }
  ret = HcomGetBandWidthPerNPU(0, &topoFirst.bwPerGpu);
  if (ret != HCCL_SUCCESS) {
    topoFirst.bwPerGpu = BW_PER_GPU_TEN;
  }
  ret = HcomGetBandWidthPerNPU(1, &topoSecond.bwPerGpu);
  if (ret != HCCL_SUCCESS) {
    topoSecond.bwPerGpu = BW_PER_GPU_TEN;
  }
  HCCL_INFO("[Get][TopoInfoFromFile] algorithm0[%s], algorithm1[%s], bandwidth0[%.2f], bandwidth1[%.2f]",
            topoFirst.algorithm.c_str(), topoSecond.algorithm.c_str(), topoFirst.bwPerGpu, topoSecond.bwPerGpu);

  topoFirst.topoStackNum = 1;
  topoSecond.topoStackNum = modelPara.serverPortNum;
  topoFirst.gpuNum = modelPara.devNumInServer;
  topoSecond.gpuNum = modelPara.serverNum;

  if (modelPara.productForm.find("module") != string::npos) {
    topoFirst.bwPortPerGpu = BW_PORT_PER_GPU;
    topoSecond.bwPortPerGpu = 1;
  } else if (modelPara.productForm.find("card") != string::npos) {
    topoFirst.bwPortPerGpu = 1;
    topoSecond.bwPortPerGpu = 1;
  }

  toposInfo.push_back(topoFirst);
  if (modelPara.serverNum > 1) {
    toposInfo.push_back(topoSecond);
  }

  return;
}

Cluster::Cluster([[maybe_unused]] std::string workPath, int gpuNum, float fixedJetter) {
  vector<TopoInfo> topoInfo;
  Topology topo;

  gpuNum_ = gpuNum;
  mFixedJetter_ = fixedJetter;
  mDeviceMemory_ = DEFAULT_TOPOFIRST_DEVICE_MEMORY;
  GetTopoInfoFromFile(topoInfo);
  for (size_t i = 0; i < topoInfo.size(); i++) {
    if (topoInfo[i].topoType == "ring") {
      topo = Ring(topoInfo[i]);
    } else if (topoInfo[i].topoType == "mesh") {
      topo = Mesh(topoInfo[i]);
    } else if (topoInfo[i].topoType == "2D-torus") {
      topo = Torus2D(topoInfo[i]);
    } else if (topoInfo[i].topoType == "star") {
      topo = Star(topoInfo[i]);
    } else if (topoInfo[i].topoType == "tree") {
      topo = Tree(topoInfo[i]);
    }

    mTopoList_.push_back(topo);
    mTopoNumList_.push_back(topoInfo[i].topoStackNum);
    mDeviceMemory_ = topoInfo[i].deviceMemory;
  }
}

Cluster::~Cluster() {}

float Cluster::CalculateCost(const Communication &op, float size, float divisor) {
  float cost = 0.0;
  float inputSize = size;
  float tmp = 0.0;
  vector<int> slices;

  for (size_t i = 0; i < this->mTopoList_.size(); i++) {
    if (op.mName_ == "Allreduce" || op.mName_ == "Reducescatter") {
      inputSize = inputSize / this->mTopoNumList_[i];
    }
    if (i < this->mTopoList_.size() - 1) {
      slices.clear();
      if (this->mTopoNumList_[i + 1] < this->mTopoList_[i].gpuNum_ && this->mTopoList_[i + 1].gpuNum_ > 1) {
        for (int j = 0; j < this->mTopoNumList_[i + 1]; j++) {
          slices.push_back(inputSize / this->mTopoNumList_[i + 1]);
        }
        for (int j = 0; j < (this->mTopoList_[i].gpuNum_ - this->mTopoNumList_[i + 1]); j++) {
          slices.push_back(0);
        }
      }
      tmp = static_cast<float>(this->mTopoList_[i].CalculateCost(op, inputSize, slices, divisor));
      cost += tmp;
    } else {
      tmp = static_cast<float>(this->mTopoList_[i].CalculateCost(op, inputSize, slices, divisor));
      cost += tmp;
    }
    if (op.mName_ == "Allgather") {
      inputSize = inputSize * this->mTopoList_[i].gpuNum_;
    }
  }

  return cost;
}

float Cluster::CalculateCostWithJetter(const Communication &op, float size, float fixedJetter, float divisor) {
  float jetter = this->mFixedJetter_;

  if (fabs(size - 0) < EPSILON) {
    return 0;
  }
  if (fabs(fixedJetter - 0) > EPSILON) {
    jetter = fixedJetter;
  }

  return CalculateCost(op, size, divisor) + jetter;
}

float Cluster::CalculateStartUpCost(const Communication &op, float size, float divisor) {
  float cost = 0.0;
  float inputSize = size;
  float tmp = 0.0;
  vector<int> slices;

  for (size_t i = 0; i < this->mTopoList_.size(); i++) {
    if (op.mName_ == "Allreduce" || op.mName_ == "Reducescatter") {
      inputSize = inputSize / this->mTopoNumList_[i];
    }
    if (i < this->mTopoList_.size() - 1) {
      slices.clear();
      if (this->mTopoNumList_[i + 1] < this->mTopoList_[i].gpuNum_ && this->mTopoList_[i + 1].gpuNum_ > 1) {
        for (int j = 0; j < this->mTopoNumList_[i + 1]; j++) {
          slices.push_back(inputSize / this->mTopoNumList_[i + 1]);
        }
        for (int j = 0; j < (this->mTopoList_[i].gpuNum_ - this->mTopoNumList_[i + 1]); j++) {
          slices.push_back(0);
        }
      }
      tmp = this->mTopoList_[i].CalculateStartUpCost(op, inputSize, slices, divisor);
      cost += tmp;
    } else {
      tmp = this->mTopoList_[i].CalculateStartUpCost(op, inputSize, slices, divisor);
      cost += tmp;
    }
    if (op.mName_ == "Allgather") {
      inputSize = inputSize * this->mTopoList_[i].gpuNum_;
    }
  }

  return cost;
}
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iomanip>
#include "layers.h"
#include "cluster.h"
#include "evaluator.h"
#include "hccl/base.h"
#include "model.h"

using namespace std;

namespace {
const int MAX_SPLIT_NUM = 8;
const int PRINT_NUM = 2;
const float US_TO_S = 1000000.0;
const float KB_TO_M = 1024.0;
const float TOTAL_PERC = 100.00;  // 将size转换为百分比
const float TOTAL_HUND = 100.00;  // 用于保留两位有效数字
}  // namespace
vector<float> GetCostInfo(std::vector<uint64_t> &graInfo) {
  vector<float> layerCost;

  layerCost.push_back(0.0);
  for (size_t i = 1; i < graInfo.size(); i++) {
    layerCost.push_back(static_cast<float>(graInfo[i] / US_TO_S));
  }
  return layerCost;
}

vector<float> GetSizeInfo(std::vector<uint64_t> &graInfo) {
  vector<float> layerSizeParam;

  for (size_t i = 0; i < graInfo.size(); i++) {
    layerSizeParam.push_back(graInfo[i] / KB_TO_M / KB_TO_M);
  }
  return layerSizeParam;
}

vector<Layer> GetBpsInfo(vector<float> layerCost, vector<float> layerSizeParam) {
  string layerName = "";
  vector<Layer> bps;

  for (size_t i = 0; i < layerCost.size(); i++) {
    Layer layer(layerName, layerSizeParam[i], layerCost[i]);
    bps.push_back(layer);
  }

  return bps;
}

vector<float> CalculateSizeRatio(const vector<float> &sliceSize, float totalSize) {
  vector<float> sliceRatio;
  float usedSizeRatio = 0.0f;
  float curSizeRatio = 0.0f;
  int sliceNum = sliceSize.size() - 1;
  for (int i = 0; i < sliceNum; i++) {
    curSizeRatio = floor(TOTAL_HUND * (TOTAL_PERC * sliceSize[i] / totalSize)) / TOTAL_HUND;
    curSizeRatio = ((TOTAL_PERC - (usedSizeRatio + curSizeRatio)) < 1e-6) ? (TOTAL_PERC - usedSizeRatio) : curSizeRatio;
    usedSizeRatio += curSizeRatio;
    sliceRatio.push_back(curSizeRatio);
  }
  sliceRatio.push_back(TOTAL_PERC - usedSizeRatio);
  return sliceRatio;
}

Model::Model(std::vector<uint64_t> &graInfoCost, std::vector<uint64_t> &graInfoSize, int batchs, u64 tensorLimit) {
  mBatchs_ = batchs;
  mName_ = "";
  vector<float> layerCost = GetCostInfo(graInfoCost);
  vector<float> layerSizeParam = GetSizeInfo(graInfoSize);

  mBps_ = GetBpsInfo(layerCost, layerSizeParam);
  mLimit_ = static_cast<float>(tensorLimit);
}

Model::~Model() {}

float Model::CalculateParamSize(vector<Layer> layers, int start, int end) {
  float size = 0.0;

  if ((end == -1) || (end >= static_cast<int>(layers.size()))) {
    for (size_t i = start; i < layers.size(); i++) {
      size += layers[i].mParamSize_;
    }
  } else {
    for (int i = start; i < end; i++) {
      size += layers[i].mParamSize_;
    }
  }

  return size;
}

float Model::CalculateCost(vector<Layer> layers, int start, int end) {
  float cost = 0.0;

  if ((end == -1) || (end >= static_cast<int>(layers.size()))) {
    for (size_t i = start; i < layers.size(); i++) {
      cost += layers[i].mCalTime_;
    }
  } else {
    for (int i = start; i < end; i++) {
      cost += layers[i].mCalTime_;
    }
  }

  return cost;
}

int Model::CalculateNextSlice(vector<Layer> &layers, int sliceStart, float costCommunication, float startUpCommTime) {
  float tmpCalTime = 0.0;
  float size = 0.0;

  for (size_t i = sliceStart; i < layers.size(); i++) {
    tmpCalTime += layers[i].mCalTime_;
    size += layers[i].mParamSize_;

    if ((tmpCalTime > costCommunication + startUpCommTime) || (size > mLimit_)) {
      return i;
    }
  }
  return layers.size();
}

Model::SliceMeth Model::CalculateTrail(Cluster &cluster, const Communication &op, vector<Layer> &layersOriginal,
                                       int firstSliceEnd, float &trailCostNew) {
  SliceMeth slice;
  vector<int> slicesNew;
  vector<float> sliceSize;
  bool sliceLastFound = false;
  int sliceStart;
  int sliceEnd;

  float communication_start = this->CalculateCost(layersOriginal, 0, firstSliceEnd);
  float size = this->CalculateParamSize(layersOriginal, 0, firstSliceEnd);
  float costCommunication = cluster.CalculateCostWithJetter(op, size);
  float startUpCommTime = cluster.CalculateStartUpCost(op);

  slicesNew.push_back(firstSliceEnd);
  sliceSize.push_back(size);
  sliceStart = firstSliceEnd;
  trailCostNew = costCommunication + communication_start;
  while (!sliceLastFound) {
    sliceEnd = CalculateNextSlice(layersOriginal, sliceStart, costCommunication, startUpCommTime);
    if (sliceEnd == sliceStart) {
      trailCostNew += layersOriginal[sliceStart].mCalTime_ - costCommunication;
      sliceEnd++;
    }
    size = this->CalculateParamSize(layersOriginal, sliceStart, sliceEnd);
    costCommunication = cluster.CalculateCostWithJetter(op, size);
    slicesNew.push_back(sliceEnd);
    sliceSize.push_back(size);

    if (sliceEnd == static_cast<int>(layersOriginal.size())) {
      trailCostNew += costCommunication;
      sliceLastFound = true;
      break;
    } else {
      sliceStart = sliceEnd;
    }
    trailCostNew += costCommunication;
  }
  trailCostNew -= this->CalculateCost(layersOriginal);
  slice.sliceRatio = slicesNew;
  slice.sliceSize = sliceSize;

  return slice;
}

vector<int> Model::GradientSlicing(Cluster &cluster, const Communication &op, [[maybe_unused]] int batchSize) {
  float trailCost = 0.0;
  vector<int> slices;
  vector<float> sliceBySize;
  vector<float> sliceRatio;
  SliceMeth sliceMeth;
  float size;
  vector<Layer> layersOriginal;
  int firstSliceEnd;
  float trailCostNew = 0.0;
  stringstream sliceBySizeMethed;

  layersOriginal = this->mBps_;
  size = CalculateParamSize(layersOriginal);
  trailCost = cluster.CalculateCostWithJetter(op, size);
  slices.push_back(layersOriginal.size());
  for (size_t i = 1; i < layersOriginal.size(); i++) {
    firstSliceEnd = static_cast<int>(i);
    size = this->CalculateParamSize(layersOriginal, 0, firstSliceEnd);
    if (size > mLimit_) {
      break;
    }
    sliceMeth = CalculateTrail(cluster, op, layersOriginal, firstSliceEnd, trailCostNew);
    if (trailCostNew < trailCost && sliceMeth.sliceRatio.size() <= MAX_SPLIT_NUM) {
      slices = sliceMeth.sliceRatio;
      trailCost = trailCostNew;
      sliceBySize = sliceMeth.sliceSize;
    }
  }
  sliceRatio = CalculateSizeRatio(sliceBySize, size);
  sliceBySizeMethed << setiosflags(ios::fixed) << setprecision(PRINT_NUM);
  for (auto iter = sliceRatio.begin(); iter < sliceRatio.end() - 1; iter++) {
    sliceBySizeMethed << *iter << ", ";
  }
  sliceBySizeMethed << *(sliceRatio.end() - 1);
  HCCL_RUN_INFO(
      "Calculate GradientSlicing: last trail cost = %f; \
        the proportion of size in each segment is [%s].",
      trailCost, (sliceBySizeMethed.str()).c_str());

  return slices;
}
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "communication.h"
#include "topology.h"

using namespace std;
Topology::Topology() {
  algorithm_ = "ring";
  gpuNum_ = DEFAULT_TOPOFIRST_GPU_NUM;
  portNum_ = DEFAULT_TOPOFIRST_BW_PORT_PER_GPU;
  bw_ = DEFAULT_TOPOFIRST_BW_PER_GPU * portNum_;
  bwComputation_ = DEFAULT_TOPOFIRST_BW_COMPUTATION;
  topoType_ = "ring";
  syncCostPerXfer_ = DEFAULT_TOPOFIRST_COST_SYNC;
  syncCostPerPort_ = DEFAULT_TOPOFIRST_SYNC_COST_PER_PORT;
  syncCostFixed_ = SYNC_COST_FIXED;
  fixedJitter_ = DEFAULT_TOPOFIRST_FIXED_JITTER;
}

Topology::Topology(struct TopoInfo &topoInfo) {
  algorithm_ = topoInfo.algorithm;
  gpuNum_ = topoInfo.gpuNum;
  portNum_ = topoInfo.bwPortPerGpu;
  bw_ = topoInfo.bwPerGpu * portNum_;
  bwComputation_ = topoInfo.bwComputation;
  topoType_ = topoInfo.topoType;
  syncCostPerXfer_ = topoInfo.costSync;
  syncCostPerPort_ = topoInfo.syncCostPerPort;
  syncCostFixed_ = SYNC_COST_FIXED;
  fixedJitter_ = topoInfo.fixedJitter;
}

Topology::~Topology() {}

float Topology::CalculateJitter() const {
  return fixedJitter_;
}

float Topology::CalculateCost(const Communication &op, float size, vector<int> &slices, int divisor) const {
  float xferCost;
  float computeCost;
  float syncCost;
  float bubbleCost;
  float fixedCost;
  float cost;
  float jitter;

  if (this->gpuNum_ == 1) {  // lint !e1542
    return 0.0;
  }

  xferCost = CalculateXferCost(op, size, slices);
  if (divisor != 0) {
    xferCost /= divisor;
  }
  computeCost = CalculateComputeCost(op, size, slices);
  syncCost = CalculateSyncCost(op, size, slices);
  bubbleCost = CalculateBubbleCost(op, size, slices);
  fixedCost = CalculateFixedCost(op, size, slices);
  cost = xferCost + computeCost + syncCost + bubbleCost + fixedCost;
  jitter = CalculateJitter();

  cost += jitter;
  return cost;
}

float Topology::CalculateStartUpCost(const Communication &op, float size, vector<int> &slices,
                                     [[maybe_unused]] int divisor) const {
  float syncCost;
  float fixedCost;
  float cost;

  if (this->gpuNum_ == 1) {
    return 0.0;
  }

  syncCost = CalculateSyncCost(op, size, slices);
  fixedCost = CalculateFixedCost(op, size, slices);
  cost = syncCost + fixedCost;

  return cost;
}

float Topology::CalculateXferCost(const Communication &op, float size, vector<int> &slices) const {
  float cost = 0.0;
  float xferPercentage = 0.0;

  if (this->gpuNum_ > 1) {
    xferPercentage = op.CalculateXferPercentage(this->gpuNum_, this->algorithm_, slices);
    cost = size * xferPercentage / this->bw_;
  }
  return cost;
}

float Topology::CalculateComputeCost(const Communication &op, float size, vector<int> &slices) const {
  float cost = 0.0;
  float computePercentage = 0.0;

  if (abs(this->bwComputation_) > 1.0e-9) {  // lint !e1542
    computePercentage = op.CalculateComputePercentage(this->gpuNum_, this->algorithm_, slices);
    cost = size * computePercentage / this->bwComputation_;
  }

  return cost;
}

float Topology::CalculateSyncCost(const Communication &op, [[maybe_unused]] float size, vector<int> &slices) const {
  float cost = 0.0;
  float xferNum = 0.0;
  float bubbleNum = 0.0;
  float totalNum = 0.0;

  if (this->gpuNum_ > 1) {
    xferNum = op.CalculateXferFrequency(this->gpuNum_, this->algorithm_, slices);
    bubbleNum = op.CalculateXferBubbles(this->gpuNum_, this->algorithm_, slices);
    totalNum = xferNum + bubbleNum;
    cost = totalNum * this->syncCostPerXfer_ + totalNum * this->syncCostPerPort_;  // lint !e1542
  }

  return cost;
}

float Topology::CalculateBubbleCost(const Communication &op, float size, vector<int> &slices) const {
  float cost = 0.0;
  float xferNum = 0.0;
  float bubbleNum = 0.0;
  float xferPercentage = 0.0;
  float xferCost = 0.0;

  if (this->gpuNum_ > 1) {
    xferNum = op.CalculateXferFrequency(this->gpuNum_, this->algorithm_, slices);
    bubbleNum = op.CalculateXferBubbles(this->gpuNum_, this->algorithm_, slices);
    xferPercentage = op.CalculateXferPercentage(this->gpuNum_, this->algorithm_, slices);

    if (xferNum > 0.0) {
      xferCost = (size * xferPercentage / this->bw_);
      cost = xferCost * bubbleNum;
    }
  }

  return cost;
}

float Topology::CalculateFixedCost(const Communication &op, [[maybe_unused]] float size, vector<int> &slices) const {
  float cost = 0.0;
  float xferNum;

  if (this->gpuNum_ > 1) {
    xferNum = op.CalculateXferFrequency(this->gpuNum_, this->algorithm_, slices);
    cost += xferNum * this->syncCostFixed_ * 5;               // lint !e1542
    cost += (this->portNum_ - 1) * this->syncCostFixed_ * 4;  // lint !e1542
  }

  return cost;
}

Mesh::Mesh(struct TopoInfo &topo_info) : Topology(topo_info) {}

Mesh::~Mesh() {}

float Mesh::CalculateComputeCost(const Communication &op, float size, vector<int> &slices) const {
  float cost = 0.0;
  float computePercentage = 0.0;

  if (abs(this->bwComputation_) > 1.0e-9) {
    computePercentage = op.CalculateComputePercentage(this->gpuNum_, this->algorithm_, slices);
    cost = size * computePercentage / this->bwComputation_;
  }

  return cost;
}

Tree::Tree(struct TopoInfo &topo_info) : Topology(topo_info) {}

Tree::~Tree() {}

Torus2D::Torus2D(struct TopoInfo &topo_info) : Topology(topo_info) {}

Torus2D::~Torus2D() {}

Ring::Ring(struct TopoInfo &topo_info) : Topology(topo_info) {}

Ring::~Ring() {}

Star::Star(struct TopoInfo &topo_info) : Topology(topo_info) {}

Star::~Star() {}
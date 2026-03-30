/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/weight_compress_flag/weight_compress_judge.h"
#include "common/configuration.h"
#include "common/util/op_info_util.h"
#include "graph/utils/op_desc_utils.h"
#include "compiler/graph/common/compress/inc/compress.h"
#include "common/fe_thread_pool.h"
#include "common/math_util.h"
#include "common/fe_inner_attr_define.h"

namespace fe {
const uint64_t kRecursiveMax = 100;
uint64_t kRecursiveCnt = 0;
static const size_t FRACTAL_SIZE_MIN = 512;
static const size_t FRACTAL_SIZE_MAX_TIMES = 64;
static const size_t ENGINE_NUMBER_DC = 4;
static const size_t MAX_RATIOS_DC = 64;
static const size_t COMPRESS_CHANNLES_DC = 2;
static const std::set<std::string> kSupportTwoCompressTypesOpType = {};
const std::string kWeightCompressJudgePrefix = "judge_3_";

Status WeightCompressJudge::CompressTypeJudge(ge::OptimizeUtility *const optimize_utility, ge::ComputeGraph &graph) {
  if (!Configuration::Instance(AI_CORE_NAME).IsEnableCompressWeight()) {
    return SUCCESS;
  }
  FE_LOGD("Starting judgment of weight compression type.");
  for (auto &node : graph.GetDirectNode()) {
    if (kCubeCompressOpList.count(node->GetType()) == 0) {
      continue;
    }
    ge::NodePtr switch_node = GetSpecificNode(node, 1, SWITCH);
    FE_CHECK_NOTNULL(switch_node);
    ge::NodePtr host_node = GetSpecificNode(switch_node, 1, kWeightCompressHost);
    FE_CHECK_NOTNULL(host_node);
    FE_CHECK_NOTNULL(optimize_utility);

    // 1. do const foling for nodes between weight node and host node
    if (PreConstFolding(optimize_utility, host_node) != ge::SUCCESS) {
      return FAILED;
    }

    // 2. compare compress type
    ge::NodePtr weight_node = GetSpecificNode(host_node, 0, CONSTANT);
    FE_CHECK_NOTNULL(weight_node);
    WeightCompressType weight_compress_type = WeightCompressType::DISABLE_COMPRESS;
    bool is_support_two_compress_types = (IsSupportTwoCompressTypes(node->GetType()));
    weight_compress_type = CompareCompressType(weight_node, is_support_two_compress_types);
    FE_LOGD("Node [%s, %s]: The weight compression type is %ld.", node->GetNamePtr(), node->GetTypePtr(),
            static_cast<int64_t>(weight_compress_type));

    // 3. set attr to cube_compress node and host node
    if (weight_compress_type != WeightCompressType::DISABLE_COMPRESS) {
      (void)ge::AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_WEIGHT_COMPRESS_TYPE,
          static_cast<int64_t>(weight_compress_type));
      (void)ge::AttrUtils::SetInt(host_node->GetOpDesc(), ATTR_NAME_WEIGHT_COMPRESS_TYPE,
          static_cast<int64_t>(weight_compress_type));
    }
    // 4. do const folding for host node
    FE_LOGD("Node [%s, %s]: begins to perform constant folding.", host_node->GetName().c_str(), host_node->GetType().c_str());
    if (optimize_utility->ConstantFolding(host_node) != ge::SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][WtCmpsJudge][CpmsTypeJudge] Failed to perform constant folding for node [%s, %s].",
                      host_node->GetName().c_str(), host_node->GetType().c_str());
      return FAILED;
    }
  }
  FeGraphUtils::DumpGraphAndOnnx(graph, "OptimizeOriginalGraph_FeCompressTypeJudgeAfter");
  return SUCCESS;
}

Status WeightCompressJudge::PreConstFolding(ge::OptimizeUtility *const optimize_utility, const ge::NodePtr &host_node) {
  if (host_node->GetInNodes().empty()) {
    REPORT_FE_ERROR("[GraphOpt][WtCmpsJudge][PreConstFolding] Failed to get input nodes for node [%s, %s].",
                    host_node->GetName().c_str(), host_node->GetType().c_str());
    return FAILED;
  }
  ge::InDataAnchorPtr host_in_node_anchor = host_node->GetInDataAnchor(0);
  FE_CHECK_NOTNULL(host_in_node_anchor);
  auto host_peer_out_anchor =  host_in_node_anchor->GetPeerOutAnchor();
  FE_CHECK_NOTNULL(host_peer_out_anchor);
  ge::NodePtr host_in_node = host_peer_out_anchor->GetOwnerNode();
  FE_CHECK_NOTNULL(host_in_node);

  std::vector<ge::NodePtr> const_folding_nodes = {};
  kRecursiveCnt = 0;
  if (!GetConstFoldingNodes(host_in_node, const_folding_nodes)) {
    REPORT_FE_ERROR("[GraphOpt][WtCmpsJudge][PreConstFolding] Failed to obtain constant folding nodes between weight node"
                    " and node[%s, %s].", host_node->GetName().c_str(), host_node->GetType().c_str());
    return FAILED;
  }
  for (auto iter = const_folding_nodes.rbegin(); iter != const_folding_nodes.rend(); ++iter) {
    ge::NodePtr node = *iter;
    FE_LOGD("Node [%s, %s]: begins to perform constant folding.", node->GetName().c_str(), node->GetType().c_str());
    if (optimize_utility->ConstantFolding(node) != ge::SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][WtCmpsJudge][PreConstFolding] Failed to perform constant folding for node [%s, %s].",
                      node->GetName().c_str(), node->GetType().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

ge::NodePtr WeightCompressJudge::GetSpecificNode(const ge::NodePtr &node, const int32_t &idx,
                                                 const std::string &op_type) {
  if (idx > static_cast<int32_t>(node->GetAllInDataAnchorsSize())) {
    return nullptr;
  }
  ge::InDataAnchorPtr in_node_anchor = node->GetInDataAnchor(idx);
  if (in_node_anchor == nullptr ||  in_node_anchor->GetPeerOutAnchor() == nullptr ||
      in_node_anchor->GetPeerOutAnchor()->GetOwnerNode() == nullptr) {
    return nullptr;
  }
  ge::NodePtr in_node =  in_node_anchor->GetPeerOutAnchor()->GetOwnerNode();
  if (in_node->GetType() != op_type) {
    return nullptr;
  }
  return in_node;
}

bool WeightCompressJudge::GetConstFoldingNodes(const ge::NodePtr &node,
                                               std::vector<ge::NodePtr> &const_folding_nodes) {
  if (kRecursiveCnt == kRecursiveMax) {
    FE_LOGD("Recursive calls have reached the maximum number of %lu.", kRecursiveMax);
    return false;
  }
  kRecursiveCnt++;
  if (node->GetType() == CONSTANT && ge::AttrUtils::HasAttr(node->GetOpDesc(), ge::ATTR_NAME_WEIGHTS)) {
    return true;
  }
  if (kConstFoldingOpType.count(node->GetType()) == 0) {
    FE_LOGD("Node [%s, %s]: It cannot be folded.", node->GetName().c_str(), node->GetType().c_str());
    return false;
  }
  const_folding_nodes.emplace_back(node);
  for (auto &in_node : node->GetInNodes()) {
    if (!GetConstFoldingNodes(in_node, const_folding_nodes)) {
      return false;
    }
  }
  return true;
}

bool WeightCompressJudge::IsSupportTwoCompressTypes(const std::string &op_type) {
  if (kSupportTwoCompressTypesOpType.count(op_type) == 0) {
    return false;
  }
  FE_LOGD("Node [%s]: It supports two compression types.", op_type.c_str());
  return true;
}

WeightCompressType WeightCompressJudge::CompareCompressType(const ge::NodePtr &weight_node,
                                                            const bool &is_support_two_compress_types) {
  std::vector<ge::GeTensorPtr> weights = ge::OpDescUtils::MutableWeights(weight_node);
  if (weights.empty()) {
    FE_LOGW("Node [%s, %s]: the weight tensor vector is empty.", weight_node->GetName().c_str(),
            weight_node->GetType().c_str());
    return WeightCompressType::DISABLE_COMPRESS;
  }
  ge::GeTensorPtr weight_tensor = weights[0];
  if (weight_tensor == nullptr) {
    FE_LOGW("Node[%s, %s]: the weight tensor is nullptr.", weight_node->GetNamePtr(), weight_node->GetTypePtr());
    return WeightCompressType::DISABLE_COMPRESS;
  }
  char *weight_data = const_cast<char *>(reinterpret_cast<const char *>((weight_tensor->GetData().data())));
  if (weight_data == nullptr) {
    FE_LOGW("Node[%s, %s]: the weight data pointer is nullptr.", weight_node->GetNamePtr(), weight_node->GetTypePtr());
    return WeightCompressType::DISABLE_COMPRESS;
  }

  size_t weight_size = weight_tensor->GetData().size();
  FE_LOGD("Node [%s, %s]: the weight size is %zu.", weight_node->GetNamePtr(), weight_node->GetTypePtr(), weight_size);
  if (weight_size == 0 || weight_size % FRACTAL_SIZE_MIN != 0) {
    FE_LOGW("Node [%s, %s]: The weight data is either empty or not a multiple of 512.", weight_node->GetNamePtr(),
            weight_node->GetTypePtr());
    return WeightCompressType::DISABLE_COMPRESS;
  }

  float low_sparse_compress_ratio = 0.0;
  float high_sparse_compress_ratio = 0.0;
  if (!is_support_two_compress_types) {
    low_sparse_compress_ratio = DoCompressWeights(weight_data, weight_size, WeightCompressType::LOW_SPARSE_COMPRESS);
  } else {
    fe::ThreadPool executor(kWeightCompressJudgePrefix + fe::GetCurThreadIdStr(),
        static_cast<uint32_t>(kWeightCompressTypes.size()));
    std::vector<std::future<float>> vector_future;
    for (auto compress_type : kWeightCompressTypes) {
      std::future<float> f = executor.commit(DoCompressWeights, weight_data, weight_size, compress_type);
      if (!f.valid()) {
        FE_LOGW("[Call][Commit] failed, Future is invalid, node name:%s", weight_node->GetName().c_str());
        return WeightCompressType::DISABLE_COMPRESS;
      }
      vector_future.emplace_back(std::move(f));
    }
    if (vector_future.size() != kWeightCompressTypes.size()) {
      FE_LOGW("Multi-thread do compress weights for node[%s, %s] failed.", weight_node->GetName().c_str(),
              weight_node->GetType().c_str());
      return WeightCompressType::DISABLE_COMPRESS;
    }
    try {
      low_sparse_compress_ratio = vector_future[0].get();
      high_sparse_compress_ratio = vector_future[1].get();
    } catch (const std::exception &exp) {
      FE_LOGE("Node[%s, %s]: An exception occurred while running DoCompressWeights, with the error message: [%s].",
              weight_node->GetNamePtr(), weight_node->GetTypePtr(), exp.what());
      return WeightCompressType::DISABLE_COMPRESS;
    }
  }
  return GetFinalCompressType(low_sparse_compress_ratio, high_sparse_compress_ratio);
}

float WeightCompressJudge::DoCompressWeights(char* input, const size_t &input_size,
                                             const WeightCompressType &compress_type) {
  float compress_ratio = 0.0;
  CompressConfig compress_config;
  compress_config.inputSize = input_size;
  compress_config.engineNum = ENGINE_NUMBER_DC;
  compress_config.maxRatio = MAX_RATIOS_DC;
  compress_config.channel = COMPRESS_CHANNLES_DC;
  compress_config.fractalSize = ComputeFractalSize(input_size);
  compress_config.isTight = true;
  compress_config.init_offset = 0;
  compress_config.compressType = static_cast<int32_t>(compress_type);

  unique_ptr<char[]> indexs(new (std::nothrow) char[input_size]());
  if (indexs.get() == nullptr) {
    return compress_ratio;
  }
  unique_ptr<char[]> output(new (std::nothrow) char[input_size]());
  if (output.get() == nullptr) {
    return compress_ratio;
  }
  FE_LOGD("The fractal size is %zu bytes.", compress_config.fractalSize);

  size_t compress_size = 0;
  if (CompressWeights(input, compress_config, indexs.get(), output.get(), compress_size) != RET_SUCCESS) {
    return compress_ratio;
  }
  return static_cast<float>(compress_size) / input_size;
}


size_t WeightCompressJudge::ComputeFractalSize(const size_t &weight_size) {
  size_t fractal_size = FRACTAL_SIZE_MIN;
  for (size_t i = 1; i <= FRACTAL_SIZE_MAX_TIMES; ++i) {
    size_t fract_size = i * FRACTAL_SIZE_MIN;
    if (weight_size % fract_size == 0) {
      fractal_size = fract_size;
    }
  }
  return fractal_size;
}

WeightCompressType WeightCompressJudge::GetFinalCompressType(const float &low_sparse_compress_ratio,
                                                             const float &high_sparse_compress_ratio) {
  FE_LOGD("The low sparse compression ratio is %f, and the high sparse compression ratio is %f.",
          low_sparse_compress_ratio, high_sparse_compress_ratio);

  bool is_low_sparse_compress_ratio_smaller = (low_sparse_compress_ratio < high_sparse_compress_ratio ||
                                               FloatEqual(low_sparse_compress_ratio - high_sparse_compress_ratio, 0));
  bool is_low_sparse_compress_ratio_meet_threshold = IsMeetCompressRatioThreshold(low_sparse_compress_ratio);
  bool is_high_sparse_compress_ratio_meet_threshold = IsMeetCompressRatioThreshold(high_sparse_compress_ratio);

  if (is_low_sparse_compress_ratio_meet_threshold) {
    if (!is_high_sparse_compress_ratio_meet_threshold || is_low_sparse_compress_ratio_smaller) {
      return WeightCompressType::LOW_SPARSE_COMPRESS;
    }
  }
  if (is_high_sparse_compress_ratio_meet_threshold) {
    if (!is_low_sparse_compress_ratio_meet_threshold || !is_low_sparse_compress_ratio_smaller) {
      return WeightCompressType::HIGH_SPARSE_COMPRESS;
    }
  }
  return WeightCompressType::DISABLE_COMPRESS;
}

bool WeightCompressJudge::IsMeetCompressRatioThreshold(const float &compress_ratio) {
  float compress_ratio_threshold = Configuration::Instance(AI_CORE_NAME).GetAICoreCompressRatio();
  bool res = compress_ratio > FLT_EPSILON &&
             (compress_ratio < compress_ratio_threshold || FloatEqual(compress_ratio - compress_ratio_threshold, 0));
  if (!res) {
    FE_LOGD("The compression ratio %f does not meet the threshold %f.", compress_ratio, compress_ratio_threshold);
  }
  return res;
}
}

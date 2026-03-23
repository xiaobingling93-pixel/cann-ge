/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_PNE_MODEL_FLOW_MODEL_OM_SAVER_H_
#define BASE_PNE_MODEL_FLOW_MODEL_OM_SAVER_H_

#include "dflow/inc/data_flow/model/flow_model.h"
#include "google/protobuf/message.h"
#include "framework/common/helper/om_file_helper.h"
#include "graph/buffer.h"

namespace ge {

class FlowModelOmSaver {
 public:
  explicit FlowModelOmSaver(const FlowModelPtr &flow_model) : flow_model_(flow_model) {}
  ~FlowModelOmSaver() = default;
  // split om data dir is not empty in cache function. split_om_data_base_dir = ./cache_dir/graph_key
  Status SaveToOm(const std::string &output_file, const std::string &split_om_data_base_dir = "");
  Status SaveToModelData(ModelBufferData &model_buff);

 private:
  Status AddModelDefPartition();
  Status AddFlowModelPartition();
  Status AddFlowSubModelPartitions(const std::string &split_om_data_base_dir = "");
  Status UpdateModelHeader();
  Status AddPartition(const google::protobuf::Message &partition_msg, ModelPartitionType partition_type);
  Status AddPartition(Buffer &buffer, ModelPartitionType partition_type);
  Status SaveFlowModelToFile(const std::string &output_file);
  Status SaveFlowModelToDataBuffer(ModelBufferData &model_buff);

  /**
   * @brief fix non standard graph load failed.
   * flow model is seperate by partitionCall, graph output node and subgraph is incorrect.
   * now just remove output nodes and subgraphs.
   * @param graph graph.
   */
  static void FixNonStandardGraph(const ComputeGraphPtr &graph);
  const FlowModelPtr flow_model_;
  OmFileSaveHelper om_file_save_helper_;
  // used for cache partition buffer before save to file.
  std::vector<Buffer> buffers_;
};
}  // namespace ge
#endif  // BASE_PNE_MODEL_FLOW_MODEL_OM_SAVER_H_
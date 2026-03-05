/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DFLOW_INC_EXTERNAL_DFLOW_API_H_
#define DFLOW_INC_EXTERNAL_DFLOW_API_H_

#include <map>
#include <memory>
#include <vector>

#include "ge_common/ge_api_error_codes.h"
#include "ge_common/ge_api_types.h"
#include "ge/ge_data_flow_api.h"
#include "flow_graph/data_flow.h"

namespace ge {
class DFlowSessionImpl;

namespace dflow {
GE_FUNC_VISIBILITY Status DFlowInitialize(const std::map<AscendString, AscendString> &options);
GE_FUNC_VISIBILITY Status DFlowFinalize();

class GE_FUNC_VISIBILITY DFlowSession {
 public:
  explicit DFlowSession(const std::map<AscendString, AscendString> &options);
  ~DFlowSession();

  /**
   * @brief add a flow graph with a specific graphId and graphOptions
   * @param [in] graph_id graph id
   * @param [in] graph the graph
   * @param [in] options graph options
   * @return Status result of function
   */
  Status AddGraph(uint32_t graph_id, const FlowGraph &graph,
                  const std::map<AscendString, AscendString> &options = {});

  /**
   * @brief remove a graph of the session with specific session id
   * @param [in] graph_id graph id
   * @return Status result of function
   */
  Status RemoveGraph(uint32_t graph_id);

  /**
   * @brief build graph. The build graph interface includes compile graph and load graph.
   * @param [in] graph_id graph id
   * @param [in] inputs input data
   * @return Status result of function
  */
  Status BuildGraph(uint32_t graph_id, const std::vector<Tensor> &inputs);

  /**
   * @brief get session id
   * @return id of session
   */
  uint64_t GetSessionId() const;
  /**
   * @brief Feed input data to graph.
   * @param [in] graph_id graph id
   * @param [in] inputs input data
   * @param [in] info intput data flow flag
   * @param [in] timeout data feed timeout(ms), -1 means never timeout
   * @return Status result of function
   */
  Status FeedDataFlowGraph(uint32_t graph_id, const std::vector<Tensor> &inputs, const DataFlowInfo &info,
                           int32_t timeout);
  /**
   * @brief Feed input data to graph.
   * @param [in] graph_id graph id
   * @param [in] indexes fetch output data order(index cannot be duplicated)
   * @param [in] inputs input data
   * @param [in] info intput data flow flag
   * @param [in] timeout data feed timeout(ms), -1 means never timeout
   * @return Status result of function
   */
  Status FeedDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                           const std::vector<Tensor> &inputs, const DataFlowInfo &info, int32_t timeout);

  /**
   * @brief Feed input data to graph.
   * @param [in] graph_id graph id
   * @param [in] inputs input data
   * @param [in] timeout data feed timeout(ms), -1 means never timeout
   * @return Status result of function
   */
  Status FeedDataFlowGraph(uint32_t graph_id, const std::vector<FlowMsgPtr> &inputs, int32_t timeout);

  /**
   * @brief Feed input data to graph.
   * @param [in] graph_id graph id
   * @param [in] indexes fetch output data order(index cannot be duplicated)
   * @param [in] inputs input data
   * @param [in] timeout data feed timeout(ms), -1 means never timeout
   * @return Status result of function
   */
  Status FeedDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                           const std::vector<FlowMsgPtr> &inputs, int32_t timeout);
  /**
   * @brief Feed input data to graph.
   * @param [in] graph_id graph id
   * @param [in] raw_data_list A list containing one or multiple RawData objects.
   *            All RawData elements in this list will be automatically combined as one input.
   * @param [in] index feed input index
   * @param [in] info intput data flow flag
   * @param [in] timeout data feed timeout(ms), -1 means never timeout
   * @return Status result of function
   */
  Status FeedRawData(uint32_t graph_id, const std::vector<RawData> &raw_data_list, uint32_t index,
                     const DataFlowInfo &info, int32_t timeout);

  /**
   * @brief Fetch graph output data in order.
   * @param [in] graph_id graph id
   * @param [out] outputs output data
   * @param [in] timeout data fetch timeout(ms), -1 means never timeout
   * @return Status result of function
   */
  Status FetchDataFlowGraph(uint32_t graph_id, std::vector<FlowMsgPtr> &outputs, int32_t timeout);

  /**
   * @brief Fetch graph output data in order.
   * @param [in] graph_id graph id
   * @param [out] outputs output data
   * @param [out] info output data flow flag
   * @param [in] timeout data fetch timeout(ms), -1 means never timeout
   * @return Status result of function
   */
  Status FetchDataFlowGraph(uint32_t graph_id, std::vector<Tensor> &outputs, DataFlowInfo &info,
                            int32_t timeout);
  /**
   * @brief Fetch graph output data in order.
   * @param [in] graph_id graph id
   * @param [in] indexes fetch output data order(index cannot be duplicated)
   * @param [out] outputs output data
   * @param [out] info output data flow flag
   * @param [in] timeout data fetch timeout(ms), -1 means never timeout
   * @return Status result of function
   */
  Status FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                            std::vector<Tensor> &outputs, DataFlowInfo &info, int32_t timeout);

  /**
   * @brief Fetch graph output data in order.
   * @param [in] graph_id graph id
   * @param [in] indexes fetch output data order(index cannot be duplicated)
   * @param [out] outputs output data
   * @param [in] timeout data fetch timeout(ms), -1 means never timeout
   * @return Status result of function
   */
  Status FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                            std::vector<FlowMsgPtr> &outputs, int32_t timeout);

 private:
  std::shared_ptr<DFlowSessionImpl> dflow_session_impl_;
};
} // namespace dflow
} // namespace ge
#endif  // DFLOW_INC_EXTERNAL_DFLOW_API_H_
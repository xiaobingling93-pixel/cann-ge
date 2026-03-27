/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PREPROCESS_GRAPH_PREPROCESS_H_
#define GE_GRAPH_PREPROCESS_GRAPH_PREPROCESS_H_
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "framework/common/debug/log.h"
#include "common/helper/model_parser_base.h"
#include "framework/common/string_util.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/compute_graph.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/manager/util/graph_rebuild_state_ctrl.h"
#include "graph/model.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/resource_context_mgr.h"

namespace ge {
class GraphPrepare {
 public:
  GraphPrepare();
  virtual ~GraphPrepare();
  GraphPrepare(const GraphPrepare &in) = delete;
  GraphPrepare &operator=(const GraphPrepare &in) = delete;
  Status PrepareInit(const GraphNodePtr &graph_node, uint64_t session_id = 0,
                     GraphRebuildStateCtrl *graph_rebuild_state_ctrl = nullptr,
                     ResourceContextMgr *resource_context_mgr = nullptr);
  Status NormalizeGraph(const ComputeGraphPtr &compute_graph,
                        const std::map<std::string, std::string> &options,
                        const std::vector<GeTensor> &user_input);
  Status PrepareDynShape();
  Status RecordAIPPInfo(const ge::ComputeGraphPtr &compute_graph) const;
  Status PrepareRunningFormatRefiner();
  void SetOptions(const GraphManagerOptions &options);
  Status GenerateInfershapeGraph(ConstGraphPtr graph);
  Status SwitchOpOptimize(ComputeGraphPtr &compute_graph) const;
  void SetGraphNormalized(const bool graph_normalized);
  // some functions required session_id on compute_graph, which set when graph_prepare init.
  // so every func which invoke InferShapeForPreprocess need keep position after graph_prepare init.
  static Status InferShapeForPreprocess(ComputeGraphPtr &compute_graph, GraphRebuildStateCtrl *rebuild_ctrl,
                                        ResourceContextMgr *resource_mgr);
  Status CheckAippInsert();
 private:
  // Remove magic attributes (should be added by compile) to prevent compilation result injection.
  void RemoveMagicCompiledAttrs() const;
  Status CheckGraphAndUpdateOriginShape() const;
  Status Init(const ge::Graph &graph, uint64_t session_id = 0,
              GraphRebuildStateCtrl *graph_rebuild_state_ctrl = nullptr,
              ResourceContextMgr *resource_context_mgr = nullptr);
  Status CheckRefInputNode(const NodePtr &node, const std::string &input_name,
                           const std::set<NodePtr> &ref_nodes) const;
  Status CheckRefOp();
  Status AdjustDataOpOutput(const NodePtr &node) const;
  Status CheckInternalFormat(const NodePtr &input_node, const GeTensorDesc &desc) const;
  Status UpdateDataInputOutputDesc(int64_t index, const OpDescPtr &op, GeTensorDesc &desc) const;
  Status UpdateInput(const std::vector<GeTensor> &user_input, const std::map<std::string, std::string> &graph_option);
  Status CheckAndUpdateInput(const std::vector<GeTensor> &user_input,
                             const std::map<std::string, std::string> &graph_option);
  Status CheckConstOp() const;
  Status CheckTensorIsValid(const NodePtr &node, int64_t shape_size, size_t data_size,
                            size_t dim_num, DataType data_type) const;
  Status VerifyConstOp(const NodePtr &node) const;
  Status CheckUserInput(const std::vector<GeTensor> &user_input);
  Status UpdateDataNetOutputByStorageFormat() const;
  Status UpdateDataByStorageFormat(const NodePtr &data_node) const;
  Status UpdateConstPlaceHolderByStorageFormat(const NodePtr &node) const;
  Status PrepareOptimize();
  Status TryDoAipp();
  Status UpdateVariableFormats(const ComputeGraphPtr &graph) const;
  Status FormatAndShapeProcess();
  Status SaveOriginalGraphToOmModel() const;
  Status ProcessNetOutput() const;
  Status UpdateInputOutputByOptions();
  Status CtrlFlowPreProcess() const;
  Status ProcessAippNodesDataFormat();

  bool IsTansDataOpData(const ge::NodePtr &var_node) const;

  Status GraphEquivalentTransformation();
  Status CopyVarIntoSubgraph() const;
  void TypeConversionOfConstant() const;
  bool IsDynamicDims(const NodePtr &input_node) const;
  Status UpdateUninitializedOriginShape(const NodePtr &input_node) const;
  Status RunCustomPass() const;
  Status InferFormatStage2() const;
  ge::ComputeGraphPtr compute_graph_;
  GraphManagerOptions options_;
  uint64_t session_id_ = 0;
  GraphRebuildStateCtrl *graph_rebuild_state_ctrl_ = nullptr;
  ResourceContextMgr *resource_context_mgr_ = nullptr;
  bool graph_normalized_ = false;
  bool aipp_checked_ = false;
};
}  // namespace ge
#endif  // GE_GRAPH_PREPROCESS_GRAPH_PREPROCESS_H_

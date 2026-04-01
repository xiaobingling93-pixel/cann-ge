/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_GRAPH_LOADER_H_
#define GE_GRAPH_LOAD_GRAPH_LOADER_H_

#include "base/err_msg.h"

#include <memory>
#include <string>
#include <vector>

#include "framework/common/debug/log.h"
#include "framework/common/fmk_types.h"
#include "framework/common/ge_types.h"
#include "graph/compute_graph.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/model.h"
#include "runtime/mem.h"
#include "base/err_mgr.h"
#include "framework/common/ge_model_inout_types.h"
#include "acl/acl_rt.h"

namespace ge {
class GraphLoader {
 public:
  GraphLoader() = default;

  virtual ~GraphLoader() = default;

  GraphLoader(const GraphLoader &in) = delete;

  GraphLoader &operator=(const GraphLoader &in) & = delete;

  static Status UnloadModel(const uint32_t model_id);

  static Status LoadDataFromFile(const std::string &path, const int32_t priority, ModelData &model_data);

  static Status LoadModelFromData(const ModelData &model_data, const ModelParam &model_param, uint32_t &model_id);

  static Status LoadModelWithQ(uint32_t &model_id, const ModelData &model_data, const ModelQueueArg &arg);

  static Status LoadModelWithQueueParam(uint32_t &model_id, const ModelData &model_data,
                                        const ModelQueueParam &model_queue_param);

  static Status LoadModelWithQueueParam(uint32_t &model_id, const GeRootModelPtr &root_model,
                                        const ModelQueueParam &model_queue_param,
                                        const bool need_update_session_id = true);

  static Status LoadModelWithoutQ(uint32_t &model_id, const GeRootModelPtr &root_model);

  static Status ExecuteModel(const uint32_t model_id, aclrtStream const stream, const bool async_mode,
                             const InputData &input_data, const std::vector<GeTensorDesc> &input_desc,
                             OutputData &output_data, std::vector<GeTensorDesc> &output_desc);

  static Status LoadModelOnline(uint32_t &model_id, const GeRootModelPtr &ge_root_model,
                                const GraphNodePtr &graph_node, const uint32_t device_id,
                                const error_message::ErrorManagerContext &error_context,
                                const aclrtStream stream = nullptr);

  static Status GetModelDescInfoFromMem(const ModelData &model_data, ModelInOutInfo &info);

  static Status GetRuntimeModelId(const uint32_t model_id, uint32_t &model_runtime_id);
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_GRAPH_LOADER_H_

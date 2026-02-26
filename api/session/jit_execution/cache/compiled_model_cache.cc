/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <checker.h>
#include <chrono>
#include "compiled_model_cache.h"
#include "compiler/graph/build/model_cache.h"
#include "api/session/jit_execution/utils/guarded_execution_point_util.h"
#include "graph/ge_local_context.h"
#include "graph_metadef/graph/utils/file_utils.h"

namespace ge {
CompiledModelCache::CompiledModelCache(uint32_t user_graph_id, CompileContext &context, GraphManager &graph_manager) : user_graph_id_(user_graph_id),
       compile_context_(context), graph_manager_(graph_manager)	{
  user_graph_key_ = ModelCache::GetGraphKeyFromContext();
  if (user_graph_key_.empty()) {
    GELOGI("The user_graph_key is not set in the options, cmc will not restore or save cache.");
  }

  const std::string root_dir_origin = ModelCache::GetCacheDirFromContext();
  if (root_dir_origin.empty()) { // if user does not set cache_dir in the option, cmc will not generate cache_dir
    GELOGI("The cache_dir is not set in the options, cmc will not restore or save cache.");
  } else {
    root_dir_ = root_dir_origin + "/" + kCompiledModelCacheDirName + "/";
    CreateDirectory(root_dir_); // create the cache_dir
  }
  GELOGI("Init complied model cache success, user_graph_id[%u].", user_graph_id_);
}

Status CompiledModelCache::GetGuardedExecutionPointGraphKey(const GuardedExecutionPoint *gep, std::string &gep_graph_key) {
  return eo_util_.GetGuardedExecutionPointGraphKey(gep, gep_graph_key);
}

Status CompiledModelCache::CreateKeyOptionForGuardedExecutionPoint(const GuardedExecutionPoint *gep, std::map<std::string, std::string> &options) {
  return eo_util_.CreateKeyOptionForGuardedExecutionPoint(root_dir_, user_graph_key_, gep, options);
}

Status CompiledModelCache::RestoreCache(ExecutionOrder &order) {
  if (user_graph_key_.empty() || root_dir_.empty()) {
    GELOGI("user_graph_key_ = %s and root_dir_ = %s, either is not valid. Will skip RestoreCache.",
      user_graph_key_.c_str(), root_dir_.c_str());
    return SUCCESS; // user may not set user_graph_key_ or cache_dir, then skip restoration
  }

  GE_WARN_ASSERT_GRAPH_SUCCESS(eo_util_.RestoreExecutionOrder(root_dir_, user_graph_key_, order),
		  "Failed to restore execution order.");
  GELOGI("Compiled model cache restoration success, user_graph_id[%u].", user_graph_id_);
  return SUCCESS;
}

Status CompiledModelCache::SaveCache(ExecutionOrder &order) {
  if (user_graph_key_.empty() || root_dir_.empty()) {
    GELOGI("user_graph_key_ = %s and root_dir_ = %s, cmc will not save caches.",
      user_graph_key_.c_str(), root_dir_.c_str());
    return SUCCESS;
  }

  GE_CHK_STATUS_RET(eo_util_.SaveExecutionOrder(root_dir_, user_graph_key_, user_graph_id_, order),
		  "Failed to save execution order.");
  GELOGI("Compiled model cache store success.");
  return SUCCESS;
}
} // namespace ge
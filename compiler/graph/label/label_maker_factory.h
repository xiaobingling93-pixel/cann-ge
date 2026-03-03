/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_LABEL_MAKER_FACTORY_H_
#define GE_GRAPH_PASSES_LABEL_MAKER_FACTORY_H_

#include <map>
#include <string>
#include <memory>
#include <functional>

#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
class LabelMaker;
using LabelMakerPtr = std::shared_ptr<LabelMaker>;

class LabelMakerFactory {
 public:
  // TaskManagerCreator function def
  using LabelCreatorFun = std::function<LabelMakerPtr(const ComputeGraphPtr &, const NodePtr &)>;

  static LabelMakerFactory &Instance() {
    static LabelMakerFactory instance;
    return instance;
  }

  LabelMakerPtr Create(const std::string &node_type, const ComputeGraphPtr &graph, const NodePtr &node) {
    auto it = creator_map_.find(node_type);
    if (it == creator_map_.end()) {
      GELOGW("Cannot find node type %s in map.", node_type.c_str());
      return nullptr;
    }

    return it->second(graph, node);
  }

  // LabelInfo registerar
  class Registerar {
   public:
    Registerar(const std::string &node_type, const LabelCreatorFun func) noexcept {
      LabelMakerFactory::Instance().RegisterCreator(node_type, func);
    }

    ~Registerar() = default;
  };

 private:
  LabelMakerFactory() = default;
  ~LabelMakerFactory() = default;

  // register creator, this function will call in the constructor
  void RegisterCreator(const std::string &node_type, const LabelCreatorFun func) {
    auto it = creator_map_.find(node_type);
    if (it != creator_map_.end()) {
      GELOGD("LabelMarkFactory::RegisterCreator: %s creator already exist", node_type.c_str());
      return;
    }

    creator_map_[node_type] = func;
  }

  std::map<std::string, LabelCreatorFun> creator_map_;
};

#define REGISTER_LABEL_MAKER(type, clazz)                                                         \
  LabelMakerPtr Creator_##type##_Label_Maker(const ComputeGraphPtr &graph, const NodePtr &node) { \
    std::shared_ptr<clazz> maker = nullptr;                                                       \
    maker = MakeShared<clazz>(graph, node);                                                       \
    return maker;                                                                                 \
  }                                                                                               \
  LabelMakerFactory::Registerar g_##type##_Label_Maker_Creator(type, Creator_##type##_Label_Maker)
}  // namespace ge
#endif  // GE_GRAPH_PASSES_LABEL_MAKER_FACTORY_H_
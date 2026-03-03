/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "omg/parser/parser_factory.h"
#include "framework/common/debug/ge_log.h"
#include "common/op_registration_tbe.h"
#include "base/err_msg.h"

namespace domi {
FMK_FUNC_HOST_VISIBILITY WeightsParserFactory *WeightsParserFactory::Instance() {
  static WeightsParserFactory instance;
  return &instance;
}

std::shared_ptr<WeightsParser> WeightsParserFactory::CreateWeightsParser(const domi::FrameworkType type) {
  std::map<domi::FrameworkType, WEIGHTS_PARSER_CREATOR_FUN>::const_iterator iter = creator_map_.find(type);
  if (iter != creator_map_.end()) {
    return iter->second();
  }
  REPORT_INNER_ERR_MSG("E19999", "param type invalid, Not supported Type: %d", type);
  GELOGE(FAILED, "[Check][Param]WeightsParserFactory::CreateWeightsParser: Not supported Type: %d", type);
  return nullptr;
}

FMK_FUNC_HOST_VISIBILITY void WeightsParserFactory::RegisterCreator(const domi::FrameworkType type,
                                                                    WEIGHTS_PARSER_CREATOR_FUN fun) {
  std::map<domi::FrameworkType, WEIGHTS_PARSER_CREATOR_FUN>::const_iterator iter = creator_map_.find(type);
  if (iter != creator_map_.end()) {
    GELOGW("WeightsParserFactory::RegisterCreator: %d creator already exist", type);
    return;
  }

  creator_map_[type] = fun;
}

WeightsParserFactory::~WeightsParserFactory() {
  creator_map_.clear();
}

FMK_FUNC_HOST_VISIBILITY ModelParserFactory *ModelParserFactory::Instance() {
  static ModelParserFactory instance;
  return &instance;
}

std::shared_ptr<ModelParser> ModelParserFactory::CreateModelParser(const domi::FrameworkType type) {
  std::map<domi::FrameworkType, MODEL_PARSER_CREATOR_FUN>::const_iterator iter = creator_map_.find(type);
  if (iter != creator_map_.end()) {
    return iter->second();
  }
  REPORT_INNER_ERR_MSG("E19999", "param type invalid, Not supported Type: %d", type);
  GELOGE(FAILED, "[Check][Param]ModelParserFactory::CreateModelParser: Not supported Type: %d", type);
  return nullptr;
}

FMK_FUNC_HOST_VISIBILITY void ModelParserFactory::RegisterCreator(const domi::FrameworkType type,
                                                                  MODEL_PARSER_CREATOR_FUN fun) {
  std::map<domi::FrameworkType, MODEL_PARSER_CREATOR_FUN>::const_iterator iter = creator_map_.find(type);
  if (iter != creator_map_.end()) {
    GELOGW("ModelParserFactory::RegisterCreator: %d creator already exist", type);
    return;
  }

  creator_map_[type] = fun;
}

ModelParserFactory::~ModelParserFactory() {
  creator_map_.clear();
}

FMK_FUNC_HOST_VISIBILITY OpRegTbeParserFactory *OpRegTbeParserFactory::Instance() {
  static OpRegTbeParserFactory instance;
  return &instance;
}

void OpRegTbeParserFactory::Finalize(const domi::OpRegistrationData &reg_data) {
  (void)ge::OpRegistrationTbe::Instance()->Finalize(reg_data);
}
}  // namespace domi

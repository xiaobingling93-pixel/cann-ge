/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/checker.h"
#include "framework/omg/parser/model_parser.h"
#include "framework/omg/parser/parser_api.h"
#include "framework/omg/parser/parser_factory.h"
#include "framework/omg/parser/parser_inner_ctx.h"

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY _declspec(dllexport)
#else
#define PARSER_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define PARSER_FUNC_VISIBILITY
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

PARSER_FUNC_VISIBILITY
ge::Status GeApiWrapper_ParseProtoWithSubgraph(const std::vector<ge::AscendString> &partitioned_serialized,
                                               const std::map<ge::AscendString, ge::AscendString> &const_value_map,
                                               domi::GetGraphCallbackV3 callback,
                                               ge::ComputeGraphPtr &graph) {
  std::shared_ptr<domi::ModelParser> model_parser =
    domi::ModelParserFactory::Instance()->CreateModelParser(domi::FrameworkType::TENSORFLOW);
  GE_ASSERT_NOTNULL(model_parser, "create model parser failed!");
  return model_parser->ParseProtoWithSubgraph(partitioned_serialized, const_value_map, callback, graph);
}

PARSER_FUNC_VISIBILITY
ge::Status GeApiWrapper_GetGeDataTypeByTFType(const uint32_t type, ge::DataType &data_type) {
  std::shared_ptr<domi::ModelParser> model_parser =
    domi::ModelParserFactory::Instance()->CreateModelParser(domi::FrameworkType::TENSORFLOW);
  data_type = ge::DT_UNDEFINED;
  GE_ASSERT_NOTNULL(model_parser, "create model parser failed!");
  data_type = model_parser->ConvertToGeDataType(type);
  return ge::SUCCESS;
}

PARSER_FUNC_VISIBILITY
ge::Status GeApiWrapper_ParserFinalize() {
  return ge::ParserFinalize();
}

PARSER_FUNC_VISIBILITY
ge::Status GeApiWrapper_ParserInitialize(const std::map<ge::AscendString, ge::AscendString>& options) {
  return ge::ParserInitialize(options);
}

PARSER_FUNC_VISIBILITY
void GeApiWrapper_SetDomiFormatFromParserContext() {
  domi::GetContext().format = ge::GetParserContext().format;
}

#ifdef __cplusplus
}
#endif

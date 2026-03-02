/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SAMPLES_ES_SHOWCASE_MAKE_SIMPLE_H_
#define SAMPLES_ES_SHOWCASE_MAKE_SIMPLE_H_
#include <memory>
#include "graph/graph.h"
namespace es_showcase {
  std::unique_ptr<ge::Graph> MakeSubGraphByEs();
  void MakeSubGraphByEsAndDump();
  std::unique_ptr<ge::Graph> MakeSubSoftMaxGraphByEs();
  void MakeSubSoftMaxGraphByEsAndDump();
  int MakeAddSubMulDivGraphByEsAndRun();

  std::unique_ptr<ge::Graph> MakeTransformerNzGraphByEs();
  void MakeTransformerNzGraphByEsAndDump();
  int MakeTransformerNzGraphByEsAndRun();
}
#endif //SAMPLES_ES_SHOWCASE_MAKE_SIMPLE_H_
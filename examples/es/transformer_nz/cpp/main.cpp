/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "src/es_showcase.h" // es构图方式
#include <iostream>
#include <map>
#include <functional>
#include "ge/ge_api.h"
#include "graph/graph.h"
#include "graph/tensor.h"
#include "utils.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    es_showcase::MakeTransformerNzGraphByEsAndDump();
    return 0;
  }
  std::string command = argv[1];
  if (command == "run") {
    std::map<ge::AscendString, ge::AscendString> config = {
      {"ge.exec.deviceId", "0"},
      {"ge.graphRunMode", "0"}
    };
    auto ret = ge::GEInitialize(config);
    if (ret != ge::SUCCESS) {
      std::cerr << "GE 初始化失败\n";
      return -1;
    }
    int result = -1;
    if (es_showcase::MakeTransformerNzGraphByEsAndRun() == 0 ) {
      result = 0;
    }
    ge::GEFinalize();
    std::cout << "执行结束" << std::endl;
    return result;
  } else if (command == "dump") {
    es_showcase::MakeTransformerNzGraphByEsAndDump();
    return 0;
  } else {
    std::cout << "错误: 未知命令 '" << command << "'" << std::endl;
    return -1;
  }
}

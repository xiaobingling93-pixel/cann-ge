/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "es_showcase.h"
#include "utils.h"
#include <memory>
#include "ge/ge_api.h"
using namespace ge;
using namespace ge::es;
namespace {
es::EsTensorHolder MakeAddGraph(es::EsTensorHolder input1, es::EsTensorHolder input2) {
  // 操作符重载
  return input1 + input2;
}
}
namespace es_showcase {
int RunGraphAdd(ge::Graph &graph, const std::vector<ge::Tensor> &inputs,
             const std::string &output_prefix) {
  std::map<ge::AscendString, ge::AscendString> options;
  auto *s = new (std::nothrow) ge::Session(options);
  if (s == nullptr) {
    std::cout << "Global session not ready" << std::endl;
    return -1;
  }
  static uint32_t next =0;
  const uint32_t graph_id = next++;
  auto ret = s->AddGraph(graph_id, graph);
  if (ret != ge::SUCCESS) {
    std::cout << "AddGraph failed" << std::endl;
    return -1;
  }
  std::vector<ge::Tensor> outputs;
  ret = s->RunGraph(graph_id, inputs, outputs);
  if (ret != ge::SUCCESS) {
    std::cout << "RunGraph failed" << std::endl;
    (void)s->RemoveGraph(graph_id);
    return -1;
  }
  (void)s->RemoveGraph(graph_id);
  ge::Utils::PrintTensorsToFile(outputs, output_prefix);
  delete s;
  return 0;
}
std::unique_ptr<ge::Graph> MakeAddGraphByEs() {
  // 1、创建图构建器
  auto graph_builder = std::make_unique<EsGraphBuilder>("MakeAddGraph");
  // 默认数据类型是float, 格式为ND的标量节点, 命名为input_x
  // 2、创建输入节点
  auto input1 = graph_builder->CreateInput(0, "data0", ge::DT_INT32, FORMAT_ND, {});
  auto input2 = graph_builder->CreateInput(1, "data1", ge::DT_INT32, FORMAT_ND, {});
  auto result = MakeAddGraph(input1, input2);
  // 3、设置输出
  (void) graph_builder->SetOutput(result, 0);
  // 4、构建图
  return graph_builder->BuildAndReset();
}
void MakeAddGraphByEsAndDump() {
  std::unique_ptr<ge::Graph> graph = MakeAddGraphByEs();
  graph->DumpToFile(ge::Graph::DumpFormat::kOnnx, ge::AscendString("make_add_graph"));
}
int MakeAddGraphByEsAndRun() {
  std::unique_ptr<ge::Graph> graph = MakeAddGraphByEs();
  std::vector<ge::Tensor> inputs;
  inputs.push_back(*ge::Utils::StubTensor<int32_t>({10}, {}));
  inputs.push_back(*ge::Utils::StubTensor<int32_t>({20}, {}));
  return RunGraphAdd(*graph, inputs, "Add");
}
}

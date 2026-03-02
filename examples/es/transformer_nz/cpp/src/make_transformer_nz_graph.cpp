/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "es_showcase.h"
#include "es_all_ops.h"
#include "utils.h"
#include <memory>
#include "ge/ge_api.h"
#include <random>
#include "graph/tensor.h"

using namespace ge;
using namespace ge::es;
namespace es_showcase {
int RunGraph(ge::Graph &graph, const std::vector<ge::Tensor> &inputs,
             const std::string &output_prefix) {
  ge::Utils::PrintTensorsToFile(inputs, "input");
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
    delete s;
    return -1;
  }
  std::vector<ge::Tensor> outputs;
  ret = s->RunGraph(graph_id, inputs, outputs);
  if (ret != ge::SUCCESS) {
    std::cout << "RunGraph failed" << std::endl;
    (void)s->RemoveGraph(graph_id);
    delete s;
    return -1;
  }
  (void)s->RemoveGraph(graph_id);
  ge::Utils::PrintTensorsToFile(outputs, output_prefix);
  delete s;
  return 0;
}

std::unique_ptr<ge::Graph> MakeTransformerNzGraphByEs() {
  // 1、创建图构建器
  auto builder = std::make_unique<EsGraphBuilder>("MakeTransformerNzSubGraph");

  // 2、创建依赖源节点
  auto [input1, input2, input3] = builder->CreateInputs<3>();
  input1.SetFormat(FORMAT_FRACTAL_NZ);
  input2.SetFormat(FORMAT_FRACTAL_NZ);
  input3.SetFormat(FORMAT_FRACTAL_NZ);

  auto matmul_result1 = MatMul(
    Cast(input1, ge::DT_FLOAT),
    Transpose(Cast(input2, ge::DT_FLOAT), std::vector<int64_t>{1, 0}));
  auto sigmoid_result1 = Sigmoid(matmul_result1);
  auto reshape_result2 = Reshape(sigmoid_result1, std::vector<int64_t>{-1, 256});
  auto add_result1 = reshape_result2 + Cast(input3, ge::DT_FLOAT);
  auto [values1, indices1] = TopKV2(add_result1, builder->CreateScalar(2), true, -1, true, 3);
  auto reducesum_result1 = ReduceSum(values1, std::vector<int64_t>{-1});
  auto [values2, indices2] = TopKV2(reducesum_result1, builder->CreateScalar(4), false, -1, true, 3);
  auto cast_result2 = Cast(indices2, ge::DT_INT64);
  auto scatterelements_result1 = ScatterElements(
    ZerosLike(reducesum_result1),
    cast_result2,
    Fill(ge::es::Shape(cast_result2), Cast(builder->CreateScalar(1.0f), ge::DT_FLOAT))
  );
  auto identity_result1 = Identity(
    BroadcastTo(Unsqueeze(scatterelements_result1, std::vector<int64_t>{-1}), std::vector<int64_t>{256, 256})
  );
  auto maskedfill_result1 = MaskedFill(
    add_result1,
    LogicalNot(Cast(Reshape(identity_result1, std::vector<int64_t>{256, 256}), ge::DT_BOOL)),
    builder->CreateScalar(0.0f)
  );
  auto [values3, indices3] = TopKV2(maskedfill_result1, builder->CreateScalar(4), false, -1, true, 3);
  // 3、设置输出节点
  auto cast_result3 = Cast(indices3, ge::DT_INT64);
  auto gatherelements_result1 = GatherElements(sigmoid_result1, cast_result3, 1);
  auto realdiv_result1 = RealDiv(gatherelements_result1, builder->CreateScalar(1e-6f));
  // 4、构建图
  return builder->BuildAndReset({cast_result3, Cast(realdiv_result1 * builder->CreateScalar(2.5f), ge::DT_FLOAT)});
}

void MakeTransformerNzGraphByEsAndDump() {
  std::unique_ptr<ge::Graph> graph = MakeTransformerNzGraphByEs();
  graph->DumpToFile(ge::Graph::DumpFormat::kOnnx, ge::AscendString("make_transformer_nz_graph"));
}

std::shared_ptr<ge::Tensor> CreateNzTensor(const std::vector<float> &data, const std::vector<int64_t> &shape) {
  auto tensor = ge::Utils::StubTensor<float>(data, shape);
  if (tensor == nullptr) {
    return nullptr;
  }
  ge::TensorDesc tensor_desc = tensor->GetTensorDesc();
  tensor_desc.SetFormat(FORMAT_FRACTAL_NZ);
  tensor->SetTensorDesc(tensor_desc);
  return tensor;
}

int MakeTransformerNzGraphByEsAndRun() {
  std::unique_ptr<ge::Graph> graph = MakeTransformerNzGraphByEs();
  std::vector<ge::Tensor> inputs;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> input1_data(256 * 7168);
  for (auto &val : input1_data) {
    val = dist(gen);
  }
  std::vector<float> input2_data(256 * 7168);
  for (auto &val : input2_data) {
    val = dist(gen);
  }
  std::vector<float> input3_data(1 * 256);
  for (auto &val : input3_data) {
    val = dist(gen);
  }
  inputs.push_back(*CreateNzTensor(input1_data, {256, 7168}));
  inputs.push_back(*CreateNzTensor(input2_data, {256, 7168}));
  inputs.push_back(*CreateNzTensor(input3_data, {1, 256}));
  return RunGraph(*graph, inputs, "Transformer_nz");
}
}

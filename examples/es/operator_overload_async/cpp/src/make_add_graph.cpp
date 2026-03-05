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
#include "acl/acl.h"
#include "ge/ge_api.h"
using namespace ge;
using namespace ge::es;
namespace {
constexpr int32_t kRunDeviceId = 0;

es::EsTensorHolder MakeAddGraph(es::EsTensorHolder input1, es::EsTensorHolder input2) {
  // 操作符重载
  return input1 + input2;
}

int RunGraphAddWithSession(ge::Session &session, ge::Graph &graph, const std::vector<ge::Tensor> &inputs,
                           std::vector<ge::Tensor> &outputs, const std::string &output_prefix) {
  static uint32_t next = 0;
  const uint32_t graph_id = next++;
  auto ret = session.AddGraph(graph_id, graph);
  if (ret != ge::SUCCESS) {
    std::cout << "AddGraph failed" << std::endl;
    return -1;
  }

  ret = session.CompileGraph(graph_id);
  if (ret != ge::SUCCESS) {
    std::cout << "CompileGraph failed, ret=" << ret << std::endl;
    (void)session.RemoveGraph(graph_id);
    return -1;
  }

  const aclError set_device_before_stream_ret = aclrtSetDevice(kRunDeviceId);
  if (set_device_before_stream_ret != ACL_SUCCESS) {
    std::cout << "aclrtSetDevice failed before CreateStream, error code: "
              << set_device_before_stream_ret << std::endl;
    (void)session.RemoveGraph(graph_id);
    return -1;
  }
  aclrtStream stream = nullptr;
  const aclError create_stream_ret = aclrtCreateStream(&stream);
  if (create_stream_ret != ACL_SUCCESS) {
    std::cout << "aclrtCreateStream failed, error code: " << create_stream_ret << std::endl;
    (void)session.RemoveGraph(graph_id);
    return -1;
  }

  const aclError set_device_before_run_ret = aclrtSetDevice(kRunDeviceId);
  if (set_device_before_run_ret != ACL_SUCCESS) {
    std::cout << "aclrtSetDevice failed before RunGraphWithStreamAsync, error code: "
              << set_device_before_run_ret << std::endl;
    (void)aclrtDestroyStream(stream);
    (void)session.RemoveGraph(graph_id);
    return -1;
  }
  ret = session.RunGraphWithStreamAsync(graph_id, static_cast<void *>(stream), inputs, outputs);
  if (ret != ge::SUCCESS) {
    std::cout << "RunGraphWithStreamAsync failed" << std::endl;
    (void)aclrtSetDevice(kRunDeviceId);
    (void)aclrtDestroyStream(stream);
    (void)session.RemoveGraph(graph_id);
    return -1;
  }

  const aclError set_device_before_sync_ret = aclrtSetDevice(kRunDeviceId);
  if (set_device_before_sync_ret != ACL_SUCCESS) {
    std::cout << "aclrtSetDevice failed before aclrtSynchronizeStream, error code: "
              << set_device_before_sync_ret << std::endl;
    (void)aclrtDestroyStream(stream);
    (void)session.RemoveGraph(graph_id);
    return -1;
  }
  const aclError sync_ret = aclrtSynchronizeStream(stream);
  if (sync_ret != ACL_SUCCESS) {
    std::cout << "aclrtSynchronizeStream failed, error code: " << sync_ret << std::endl;
    (void)aclrtSetDevice(kRunDeviceId);
    (void)aclrtDestroyStream(stream);
    (void)session.RemoveGraph(graph_id);
    return -1;
  }
  const aclError set_device_before_destroy_ret = aclrtSetDevice(kRunDeviceId);
  if (set_device_before_destroy_ret != ACL_SUCCESS) {
    std::cout << "aclrtSetDevice failed before aclrtDestroyStream, error code: "
              << set_device_before_destroy_ret << std::endl;
    (void)session.RemoveGraph(graph_id);
    return -1;
  }
  (void)aclrtDestroyStream(stream);
  (void)session.RemoveGraph(graph_id);

  const aclError set_device_before_copy_ret = aclrtSetDevice(kRunDeviceId);
  if (set_device_before_copy_ret != ACL_SUCCESS) {
    std::cout << "aclrtSetDevice failed before CopyDeviceOutputsToHost, error code: "
              << set_device_before_copy_ret << std::endl;
    return -1;
  }
  std::vector<ge::Tensor> host_outputs;
  if (!ge::Utils::CopyDeviceOutputsToHost(outputs, host_outputs)) {
    std::cout << "CopyDeviceOutputsToHost failed" << std::endl;
    return -1;
  }
  ge::Utils::PrintTensorsToFile(host_outputs, output_prefix);
  return 0;
}
}
namespace es_showcase {
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
  std::vector<ge::Tensor> outputs;
  std::map<ge::AscendString, ge::AscendString> options;
  ge::Session session(options);
  ge::Tensor input0;
  ge::Tensor input1;
  ge::Tensor output0;
  if (!ge::Utils::CreateDeviceInputTensor<int32_t>({10}, {}, ge::DT_INT32, input0) ||
      !ge::Utils::CreateDeviceInputTensor<int32_t>({20}, {}, ge::DT_INT32, input1) ||
      !ge::Utils::CreateDeviceOutputTensor({}, ge::DT_INT32, output0)) {
    std::cout << "Create device tensor for copy mode failed" << std::endl;
    return -1;
  }
  inputs.push_back(input0);
  inputs.push_back(input1);
  outputs.push_back(output0);
  return RunGraphAddWithSession(session, *graph, inputs, outputs, "AddCopy");
}
}

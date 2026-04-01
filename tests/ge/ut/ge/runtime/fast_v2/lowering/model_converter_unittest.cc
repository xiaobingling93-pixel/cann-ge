/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "common/bg_test.h"
#include "common/share_graph.h"
#include "common/helper/model_helper.h"
#include "runtime/dev.h"
#include "faker/ge_model_builder.h"
#include "faker/kernel_run_context_facker.h"
#include "faker/aicore_taskdef_faker.h"
#include "faker/space_registry_faker.h"
#include "graph/debug/ge_attr_define.h"
#include "lowering/exe_graph_attrs.h"
#include "runtime/continuous_buffer.h"
#include "runtime/model_desc.h"
#include "lowering/model_converter.h"

using namespace ge;
namespace gert {
namespace {
std::unique_ptr<uint8_t[]> ReadBufferFromAttr(const ge::ExecuteGraph *const exe_graph, const char *attr_name) {
  ge::Buffer attr_buffer;
  if (!ge::AttrUtils::GetZeroCopyBytes(exe_graph, attr_name, attr_buffer)) {
    GELOGE(ge::PARAM_INVALID, "Failed to get buffer %s from root graph", attr_name);
    return nullptr;
  }
  std::unique_ptr<uint8_t[]> buffer_data = ge::MakeUnique<uint8_t[]>(attr_buffer.GetSize());
  if (buffer_data == nullptr) {
    return nullptr;
  }
  size_t buffer_size = attr_buffer.GetSize();
  size_t temp_size = 0UL;
  while (temp_size < attr_buffer.GetSize()) {
    size_t copy_size = (buffer_size > SECUREC_MEM_MAX_LEN) ? SECUREC_MEM_MAX_LEN : buffer_size;
    if (memcpy_s(buffer_data.get() + temp_size, copy_size, attr_buffer.GetData() + temp_size, copy_size) != EOK) {
      return nullptr;
    }
    temp_size += copy_size;
    buffer_size -= copy_size;
  }
  return buffer_data;
}
}
extern ge::ExecuteGraphPtr LoadExecuteGraphFromModelFile(const ge::char_t *const model_path, ge::graphStatus &error_code);
class ModelConverterUT : public bg::BgTest {
 protected:
  void SetUp() override {
    bg::BgTest::SetUp();
    rtSetDevice(0);
  }
  void TearDown() override {
    Test::TearDown();
    while (bg::ValueHolder::PopGraphFrame() != nullptr) {
    }
  }
};

TEST_F(ModelConverterUT, LoadExecuteGraphFromModelFile) {
  gert::CreateVersionInfo();
  auto graph = ShareGraph::BuildSingleNodeGraph();
  ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  graph->TopologicalSorting();
  auto ge_root_model = GeModelBuilder(graph)
      .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
      .FakeTbeBin({"Add"})
      .BuildGeRootModel();
  auto name_2_sub_models = ge_root_model->GetSubgraphInstanceNameToModel();
  EXPECT_EQ(name_2_sub_models.size(), 1);
  auto root_model = name_2_sub_models[graph->GetName()];
  EXPECT_NE(root_model, nullptr);
  ge::AttrUtils::SetInt(root_model, ge::ATTR_MODEL_STREAM_NUM, 2);
  ge::AttrUtils::SetInt(root_model, ge::ATTR_MODEL_EVENT_NUM, 0);
  ge::AttrUtils::SetInt(root_model, "_attached_stream_num", 1);

  ModelHelper model_save_helper;
  std::string om_path = "./test.om";
  ModelBufferData model;
  auto ret = model_save_helper.SaveToOmRootModel(ge_root_model, om_path, model, true);
  ASSERT_EQ(ret, ge::SUCCESS);
  auto exe_graph = LoadExecuteGraphFromModelFile(om_path.c_str(), ret);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);

  auto buffer_data = ReadBufferFromAttr(exe_graph.get(), kModelDesc);
  EXPECT_NE(buffer_data, nullptr);

  auto model_desc_buffer = reinterpret_cast<ContinuousBuffer *>(buffer_data.get());
  auto model_desc = model_desc_buffer->Get<ModelDesc>(model_desc_buffer->GetNum() - 1);
  EXPECT_NE(model_desc, nullptr);
  EXPECT_EQ(model_desc->GetReusableStreamNum(), 1);
  EXPECT_EQ(model_desc->GetReusableEventNum(), 0);

  std::string command = std::string("rm -rf ").append(om_path);
  system(command.c_str());
  gert::DestroyVersionInfo();
}

TEST_F(ModelConverterUT, LoadModelDescFromRootModel) {
  gert::CreateVersionInfo();
  auto graph = ShareGraph::BuildTwoAddNodeGraph();
  graph->TopologicalSorting();
  auto ge_root_model = GeModelBuilder(graph)
      .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
      .FakeTbeBin({"Add"})
      .BuildGeRootModel();
  auto name_2_sub_models = ge_root_model->GetSubgraphInstanceNameToModel();
  EXPECT_EQ(name_2_sub_models.size(), 1);
  auto root_model = name_2_sub_models[graph->GetName()];
  EXPECT_NE(root_model, nullptr);
  ge::AttrUtils::SetInt(root_model, ge::ATTR_MODEL_STREAM_NUM, 3);
  ge::AttrUtils::SetInt(root_model, ge::ATTR_MODEL_EVENT_NUM, 3);
  ge::AttrUtils::SetInt(root_model, ge::ATTR_MODEL_NOTIFY_NUM, 2);
  ge::AttrUtils::SetInt(root_model, "_attached_stream_num", 1);
  ge::AttrUtils::SetListStr(root_model, ge::ATTR_MODEL_OUT_NODES_NAME, {"Add:0"});

  ModelConverter model_converter;
  auto exe_graph = model_converter.ConvertGeModelToExecuteGraph(ge_root_model);

  auto buffer_data = ReadBufferFromAttr(exe_graph.get(), kModelDesc);
  EXPECT_NE(buffer_data, nullptr);
  auto model_desc_buffer = reinterpret_cast<ContinuousBuffer *>(buffer_data.get());
  auto model_desc = model_desc_buffer->Get<ModelDesc>(model_desc_buffer->GetNum() - 1);
  EXPECT_NE(model_desc, nullptr);
  EXPECT_EQ(model_desc->GetReusableStreamNum(), 2);
  EXPECT_EQ(model_desc->GetReusableEventNum(), 3);
  EXPECT_EQ(model_desc->GetReusableNotifyNum(), 2);
  EXPECT_EQ(model_desc->GetAttachedStreamNum(), 1);
  EXPECT_NE(model_desc->GetOutputDesc(0), nullptr);
  EXPECT_NE(model_desc->GetOutputDesc(0)->GetName(), nullptr);

  gert::DestroyVersionInfo();
}

TEST_F(ModelConverterUT, ConvertWithRollBackSingleStreamForStreamNotEnough) {
  setenv("MOCK_AVAIL_STREAM_NUM", "1", 0); // only has 1 stream
  gert::CreateVersionInfo();
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamTwoNodeGraph(stream_num, event_num);
  graph->TopologicalSorting();
  ASSERT_EQ(stream_num, 2);

  GeModelBuilder builder(graph);
  auto ge_root_model = builder.SetRootModelStreamNum(stream_num)
      .SetRootModelEventNum(event_num)
      .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
      .FakeTbeBin({"Add"})
      .AddTaskDef("Relu", AiCoreTaskDefFaker("AddStubBin").WithHandle())
      .FakeTbeBin({"Relu"})
      .BuildGeRootModel();

  ModelConverter model_converter;
  LoweringOption lowering_opt;
  lowering_opt.enable_single_stream = true;
  StreamAllocator stream_allocator;
  EventAllocator event_allocator;
  NotifyAllocator notify_allocator;
  ModelConverter::Args args{lowering_opt, &stream_allocator,
                            &event_allocator, &notify_allocator, nullptr};
  auto exe_graph = model_converter.ConvertGeModelToExecuteGraph(ge_root_model, args);

  auto buffer_data = ReadBufferFromAttr(exe_graph.get(), kModelDesc);
  EXPECT_NE(buffer_data, nullptr);
  auto model_desc_buffer = reinterpret_cast<ContinuousBuffer *>(buffer_data.get());
  auto model_desc = model_desc_buffer->Get<ModelDesc>(model_desc_buffer->GetNum() - 1);
  EXPECT_NE(model_desc, nullptr);
  EXPECT_EQ(model_desc->GetReusableStreamNum(), 1);
  EXPECT_EQ(model_desc->GetReusableEventNum(), 0);

  gert::DestroyVersionInfo();
  unsetenv("MOCK_AVAIL_STREAM_NUM");
}
TEST_F(ModelConverterUT, ConvertWithRollBackSingleStreamForRtsInterfaceReturnFail) {
  // invalid stream num, aclrtGetStreamAvailableNum will return fail
  setenv("MOCK_AVAIL_STREAM_NUM", "a", 0);
  gert::CreateVersionInfo();
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamTwoNodeGraph(stream_num, event_num);
  graph->TopologicalSorting();
  ASSERT_EQ(stream_num, 2);

  GeModelBuilder builder(graph);
  auto ge_root_model = builder.SetRootModelStreamNum(stream_num)
      .SetRootModelEventNum(event_num)
      .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
      .FakeTbeBin({"Add"})
      .AddTaskDef("Relu", AiCoreTaskDefFaker("AddStubBin").WithHandle())
      .FakeTbeBin({"Relu"})
      .BuildGeRootModel();

  ModelConverter model_converter;
  LoweringOption lowering_opt;
  lowering_opt.enable_single_stream = true;
  StreamAllocator stream_allocator;
  EventAllocator event_allocator;
  NotifyAllocator notify_allocator;
  ModelConverter::Args args{lowering_opt, &stream_allocator,
                            &event_allocator, &notify_allocator, nullptr};
  auto exe_graph = model_converter.ConvertGeModelToExecuteGraph(ge_root_model, args);

  auto buffer_data = ReadBufferFromAttr(exe_graph.get(), kModelDesc);
  EXPECT_NE(buffer_data, nullptr);
  auto model_desc_buffer = reinterpret_cast<ContinuousBuffer *>(buffer_data.get());
  auto model_desc = model_desc_buffer->Get<ModelDesc>(model_desc_buffer->GetNum() - 1);
  EXPECT_NE(model_desc, nullptr);
  EXPECT_EQ(model_desc->GetReusableStreamNum(), 1);
  EXPECT_EQ(model_desc->GetReusableEventNum(), 0);

  gert::DestroyVersionInfo();
  unsetenv("MOCK_AVAIL_STREAM_NUM");
}

TEST_F(ModelConverterUT, ConvertWithRollBackSingleStreamWithStaticSubModelForStreamNotEnough) {
  setenv("MOCK_AVAIL_STREAM_NUM", "4", 0); // only has 1 stream
  gert::CreateVersionInfo();
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamGraphDynamicAndStaticGraph(stream_num, event_num);
  graph->TopologicalSorting();
  ASSERT_EQ(stream_num, 5);

  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("TransData", AiCoreTaskDefFaker("TransDataStubName"))
      .AddTaskDef("Relu", AiCoreTaskDefFaker("ReluStubName"))
      .SetRootModelStreamNum(stream_num)
      .SetRootModelEventNum(event_num)
      .BuildGeRootModel();

  ModelConverter model_converter;
  LoweringOption lowering_opt;
  StreamAllocator stream_allocator;
  EventAllocator event_allocator;
  NotifyAllocator notify_allocator;
  ModelConverter::Args args{lowering_opt, &stream_allocator,
                            &event_allocator, &notify_allocator, nullptr};
  auto exe_graph = model_converter.ConvertGeModelToExecuteGraph(ge_root_model, args);

  auto buffer_data = ReadBufferFromAttr(exe_graph.get(), kModelDesc);
  EXPECT_NE(buffer_data, nullptr);
  auto model_desc_buffer = reinterpret_cast<ContinuousBuffer *>(buffer_data.get());
  auto model_desc = model_desc_buffer->Get<ModelDesc>(model_desc_buffer->GetNum() - 1);
  EXPECT_NE(model_desc, nullptr);
  EXPECT_EQ(model_desc->GetReusableStreamNum(), 1);
  EXPECT_EQ(model_desc->GetReusableEventNum(), 0);

  gert::DestroyVersionInfo();
  unsetenv("MOCK_AVAIL_STREAM_NUM");
}
}  // namespace gert

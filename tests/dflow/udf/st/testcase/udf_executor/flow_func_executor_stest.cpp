/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <list>
#include <condition_variable>
#include "gtest/gtest.h"
#include "utils/udf_utils.h"
#include "stub/udf_stub.h"
#include "common/data_utils.h"
#include "common/scope_guard.h"
#include "flow_func/logger/flow_func_logger_manager.h"
#include "flow_func/flow_func_manager.h"
#include "flow_func/flow_func_processor.h"
#define private public
#include "execute/flow_func_executor.h"
#include "model/flow_func_model.h"
#include "mockcpp/mockcpp.hpp"
#undef private
#include "config/global_config.h"
#include "dlog_pub.h"
#include "flow_func/flow_func_config_manager.h"
#include "toolchain/dump/udf_dump_manager.h"
#include "utils/udf_test_helper.h"

namespace FlowFunc {
namespace {
constexpr uint64_t kWaitInMsPerTime = 10;
class DefaultFlowMsg : public FlowMsg {
  FlowFunc::MsgType GetMsgType() const override {
    return FlowFunc::MsgType::MSG_TYPE_TENSOR_DATA;
  }

  FlowFunc::Tensor *GetTensor() const override {
    return nullptr;
  }

  int32_t GetRetCode() const override {
    return 0;
  }

  void SetRetCode(int32_t ret_code) override {}

  void SetStartTime(uint64_t start_time) override {}

  uint64_t GetStartTime() const override {
    return 0;
  }

  void SetEndTime(uint64_t end_time) override {}

  uint64_t GetEndTime() const override {
    return 0;
  }

  void SetFlowFlags(uint32_t flags) override {}

  uint32_t GetFlowFlags() const override {
    return 0;
  }

  void SetRouteLabel(uint32_t route_label) override {}
};

class DefaultMetaRunContext : public MetaRunContext {
 public:
  std::shared_ptr<FlowMsg> AllocTensorMsg(const std::vector<int64_t> &shape, TensorDataType data_type) override {
    return std::shared_ptr<FlowMsg>(nullptr);
  }

  std::shared_ptr<FlowMsg> AllocEmptyDataMsg(MsgType msg_type) override {
    return std::shared_ptr<FlowMsg>(nullptr);
  }

  int32_t SetOutput(uint32_t out_idx, std::shared_ptr<FlowMsg> out_msg) override {
    return 0;
  }

  int32_t RunFlowModel(const char *model_key, const std::vector<std::shared_ptr<FlowMsg>> &input_msgs,
                       std::vector<std::shared_ptr<FlowMsg>> &output_msgs, int32_t timeout) override {
    return 0;
  }

  int32_t GetUserData(void *data, size_t size, size_t offset = 0U) const override {
    return 0;
  }

  int32_t SetOutput(uint32_t out_idx, std::shared_ptr<FlowMsg> out_msg, const OutOptions &options) override {
    return 0;
  }

  int32_t SetMultiOutputs(uint32_t out_idx, const std::vector<std::shared_ptr<FlowMsg>> &out_msgs,
                          const OutOptions &options) override {
    return 0;
  }
};

class DefaultMetaParams : public MetaParams {
 public:
  const char *GetName() const override {
    return nullptr;
  }

  std::shared_ptr<const AttrValue> GetAttr(const char *attr_name) const override {
    return std::shared_ptr<const AttrValue>(nullptr);
  }

  size_t GetInputNum() const override {
    return 0;
  }

  size_t GetOutputNum() const override {
    return 0;
  }

  const char *GetWorkPath() const override {
    return nullptr;
  }

  int32_t GetRunningDeviceId() const override {
    return 0;
  }
};
}  // namespace
class FlowFuncExecutorSTest : public testing::Test {
protected:
  static void SetUpTestSuite() {
    FlowFuncConfigManager::SetConfig(
        std::shared_ptr<FlowFuncConfig>(&GlobalConfig::Instance(), [](FlowFuncConfig *) {}));
  }
  virtual void SetUp() {
    ClearStubEschedEvents();
    CreateModelDir();
    req_queue_id = CreateQueue();
    rsp_queue_id = CreateQueue();
    status_queue_id = CreateQueue();
    GlobalConfig::Instance().SetMessageQueueIds(req_queue_id, rsp_queue_id);
  }

  virtual void TearDown() {
    DeleteModelDir();
    for (auto qid : record_queue_ids) {
      DestroyQueue(qid);
    }
    record_queue_ids.clear();
    DestroyQueue(req_queue_id);
    DestroyQueue(rsp_queue_id);
    DestroyQueue(status_queue_id);
    GlobalMockObject::verify();
    GlobalConfig::Instance().SetMessageQueueIds(UINT32_MAX, UINT32_MAX);
  }

  std::string CreateOnlyOneBatchModel(uint32_t &input_qid, uint32_t &output_qid,
                                      const std::map<std::string, std::string> &attrs = {}, bool reshape = false,
                                      const std::vector<BufferConfigItem> &buf_cfg = {}) {
    ff::udf::UdfModelDef model_def;
    CreateUdfModel(model_def, "FlowFuncSt", __FILE__, {});
    auto udf_def = model_def.mutable_udf_def(0);

    auto proto_attrs = udf_def->mutable_attrs();
    ff::udf::AttrValue value;
    value.set_type((uint32_t)TensorDataType::DT_INT64);
    proto_attrs->insert({"out_type", value});
    ff::udf::AttrValue bool_attr;
    bool_attr.set_b(true);
    proto_attrs->insert({"need_re_init_attr", bool_attr});
    proto_attrs->insert({"_balance_scatter", bool_attr});
    if (reshape) {
      proto_attrs->insert({"_test_reshape", bool_attr});
    }
    ff::udf::AttrValue cpu_num_value;
    cpu_num_value.set_i(2);
    proto_attrs->insert({std::string("__cpu_num"), cpu_num_value});
    for (auto &cfg : buf_cfg) {
      auto cfg_proto = udf_def->add_user_buf_cfg();
      cfg_proto->set_total_size(cfg.total_size);
      cfg_proto->set_blk_size(cfg.blk_size);
      cfg_proto->set_max_buf_size(cfg.max_buf_size);
      cfg_proto->set_page_type(cfg.page_type);
    }
    auto proto_path = WriteProto(model_def, "FlowFuncSt.pb");
    ff::deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_req;
    input_qid = CreateQueue();
    record_queue_ids.emplace_back(input_qid);
    output_qid = CreateQueue();
    record_queue_ids.emplace_back(output_qid);
    AddModelToBatchModel(batch_load_model_req, proto_path, {input_qid}, {output_qid}, attrs);
    return WriteProto(batch_load_model_req, "batchModels.pb");
  }

  std::string CreateModelWithDummyOutput(uint32_t &input_qid, uint32_t &output_qid,
                                         const std::map<std::string, std::string> &attrs = {}) {
    ff::udf::UdfModelDef model_def;
    CreateUdfModel(model_def, "FlowFuncStWithDummy", __FILE__, {});
    auto udf_def = model_def.mutable_udf_def(0);

    auto proto_attrs = udf_def->mutable_attrs();
    ff::udf::AttrValue value;
    value.set_type((uint32_t)TensorDataType::DT_INT64);
    proto_attrs->insert({"out_type", value});
    ff::udf::AttrValue bool_attr;
    bool_attr.set_b(true);
    proto_attrs->insert({"need_re_init_attr", bool_attr});
    proto_attrs->insert({"_balance_gather", bool_attr});
    ff::udf::AttrValue cpu_num_value;
    cpu_num_value.set_i(2);
    proto_attrs->insert({std::string("__cpu_num"), cpu_num_value});
    auto proto_path = WriteProto(model_def, "FlowFuncSt.pb");
    ff::deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_req;
    input_qid = CreateQueue();
    record_queue_ids.emplace_back(input_qid);
    output_qid = CreateQueue();
    record_queue_ids.emplace_back(output_qid);
    AddModelToBatchModel(batch_load_model_req, proto_path, {input_qid}, {output_qid, UINT32_MAX}, attrs);
    return WriteProto(batch_load_model_req, "batchModels.pb");
  }

  void SetQueueAttr(::ff::deployer::ExecutorRequest_QueueAttrs *queue_attr, uint32_t qid,
                    int32_t device_type = Common::kDeviceTypeCpu, int32_t device_id = 0) {
    queue_attr->set_queue_id(qid);
    queue_attr->set_device_type(device_type);
    queue_attr->set_device_id(device_id);
  }

  ff::deployer::ExecutorRequest_ModelQueuesAttrs CreateModelQueues(bool link_model_out_to_in, int32_t device_type) {
    ff::deployer::ExecutorRequest_ModelQueuesAttrs model_queues_attrs;
    auto input_qid1 = CreateQueue();
    auto input_qid2 = CreateQueue();
    auto output_qid1 = CreateQueue();
    auto output_qid2 = CreateQueue();
    record_queue_ids.emplace_back(input_qid1);
    record_queue_ids.emplace_back(input_qid2);
    record_queue_ids.emplace_back(output_qid1);
    record_queue_ids.emplace_back(output_qid2);
    if (link_model_out_to_in) {
      LinkQueueInTest(output_qid1, input_qid1);
      LinkQueueInTest(output_qid2, input_qid2);
    }

    for (auto input_qid : {input_qid1, input_qid2}) {
      auto input_queue_qttr = model_queues_attrs.add_input_queues_attrs();
      SetQueueAttr(input_queue_qttr, input_qid, device_type);
    }

    for (auto output_qid : {output_qid1, output_qid2}) {
      auto output_queue_attr = model_queues_attrs.add_output_queues_attrs();
      SetQueueAttr(output_queue_attr, output_qid, device_type);
    }
    return model_queues_attrs;
  }

  std::string CreateModelWithInvoke(std::vector<uint32_t> &input_queues, std::vector<uint32_t> &output_queues,
                                    const std::string &model_key, const std::string &scope) {
    int32_t device_type = Common::kDeviceTypeCpu;
    int64_t timeout = 1000;
    ff::udf::UdfModelDef model_def;
    std::map<AscendString, ff::udf::AttrValue> attr_map;
    ff::udf::AttrValue timeout_attr;
    timeout_attr.set_i(timeout);
    attr_map["_TEST_ATTR_TIMEOUT"] = timeout_attr;
    ff::udf::AttrValue invoke_scope_attr;
    invoke_scope_attr.set_s(scope);
    attr_map["_dflow_data_flow_scope"] = invoke_scope_attr;
    ff::udf::AttrValue visible_attr2;
    visible_attr2.mutable_array()->add_s("scope/model_key");
    attr_map["_dflow_data_flow_invoked_scopes"] = visible_attr2;
    CreateUdfModel(model_def, "ST_FlowFuncCallDataFlow", __FILE__, attr_map);

    auto proto_path = WriteProto(model_def, "st_call_df.pb");
    ff::deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_req;

    for (size_t idx = 0; idx < 1; ++idx) {
      auto input_qid = CreateQueue();
      auto output_qid = CreateQueue();
      input_queues.emplace_back(input_qid);
      output_queues.emplace_back(output_qid);
    }

    AddModelToBatchModel(batch_load_model_req, proto_path, input_queues, output_queues, {}, device_type);
    auto invoked_model_queues_attrs = batch_load_model_req.mutable_models(0)->mutable_invoked_model_queues_attrs();
    (*invoked_model_queues_attrs)[model_key] = CreateModelQueues(true, device_type);

    return WriteProto(batch_load_model_req, "invoked_model.pb");
  }

 protected:
  std::vector<uint32_t> record_queue_ids;
  uint32_t req_queue_id;
  uint32_t rsp_queue_id;
  uint32_t status_queue_id;
};

TEST_F(FlowFuncExecutorSTest, basic_test) {
  uint32_t input_qid = 0;
  uint32_t output_qid = 0;
  std::map<std::string, std::string> attrs;
  attrs["_eschedProcessPriority"] = "0";
  attrs["_eschedEventPriority"] = "2";
  std::string batch_model_path = CreateOnlyOneBatchModel(input_qid, output_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  EXPECT_EQ(batch_models[0]->model_esched_process_priority_, 0);
  EXPECT_EQ(batch_models[0]->model_esched_event_priority_, 2);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(executor.cpu_num_, 3U);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ScopeGuard executor_guard([&executor]() {
    executor.Stop(true);
    executor.WaitForStop();
    executor.Destroy();
  });

  std::vector<int64_t> shape = {1, 2, 3, 4};
  float float_value = 123.1;
  DataEnqueue(input_qid, shape, TensorDataType::DT_FLOAT, float_value);
  void *out_mbuf_ptr = nullptr;

  constexpr uint64_t kMaxWaitInMs = 60 * 1000UL;
  uint64_t wait_in_ms = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    auto drv_ret = halQueueDeQueue(0, output_qid, &out_mbuf_ptr);
    if (drv_ret == DRV_ERROR_NONE) {
      break;
    } else if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
      wait_in_ms += kWaitInMsPerTime;
      continue;
    } else {
      ADD_FAILURE() << "drv_ret=" << drv_ret;
      break;
    }
  }
  ASSERT_NE(out_mbuf_ptr, nullptr) << "wait_in_ms=" << wait_in_ms;
  Mbuf *out_mbuf = (Mbuf *)out_mbuf_ptr;
  std::vector<int64_t> expect_output(CalcElementCnt(shape), (int64_t)float_value);
  CheckMbufData(out_mbuf, shape, TensorDataType::DT_INT64, expect_output.data(), expect_output.size());
  halMbufFree(out_mbuf);
}

TEST_F(FlowFuncExecutorSTest, basic_test_with_valid_buf_cfg) {
  uint32_t input_qid = 0;
  uint32_t output_qid = 0;
  std::map<std::string, std::string> attrs;
  attrs["_eschedProcessPriority"] = "0";
  attrs["_eschedEventPriority"] = "2";
  BufferConfigItem cfg0 = {2 * 1024 * 1024, 256, 8 * 1024, "normal"};
  BufferConfigItem cfg1 = {32 * 1024 * 1024, 8 * 1024, 8 * 1024 * 1024, "normal"};
  BufferConfigItem cfg2 = {2 * 1024 * 1024, 256, 8 * 1024, "huge"};
  BufferConfigItem cfg3 = {52 * 1024 * 1024, 8 * 1024, 50 * 1024 * 1024, "huge"};
  BufferConfigItem cfg4 = {66 * 1024 * 1024, 8 * 1024, 64 * 1024 * 1024, "huge"};
  std::string batch_model_path =
      CreateOnlyOneBatchModel(input_qid, output_qid, attrs, false, {cfg0, cfg1, cfg2, cfg3, cfg4});
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  EXPECT_EQ(batch_models[0]->model_esched_process_priority_, 0);
  EXPECT_EQ(batch_models[0]->model_esched_event_priority_, 2);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(executor.cpu_num_, 3U);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ScopeGuard executor_guard([&executor]() {
    executor.Stop(true);
    executor.WaitForStop();
    executor.Destroy();
  });

  std::vector<int64_t> shape = {1, 2, 3, 4};
  float float_value = 123.1;
  DataEnqueue(input_qid, shape, TensorDataType::DT_FLOAT, float_value);
  void *out_mbuf_ptr = nullptr;

  constexpr uint64_t kMaxWaitInMs = 60 * 1000UL;
  uint64_t wait_in_ms = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    auto drv_ret = halQueueDeQueue(0, output_qid, &out_mbuf_ptr);
    if (drv_ret == DRV_ERROR_NONE) {
      break;
    } else if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
      wait_in_ms += kWaitInMsPerTime;
      continue;
    } else {
      ADD_FAILURE() << "drv_ret=" << drv_ret;
      break;
    }
  }
  ASSERT_NE(out_mbuf_ptr, nullptr) << "wait_in_ms=" << wait_in_ms;
  Mbuf *out_mbuf = (Mbuf *)out_mbuf_ptr;
  std::vector<int64_t> expect_output(CalcElementCnt(shape), (int64_t)float_value);
  CheckMbufData(out_mbuf, shape, TensorDataType::DT_INT64, expect_output.data(), expect_output.size());
  halMbufFree(out_mbuf);
}

TEST_F(FlowFuncExecutorSTest, basic_test_with_tensor_list) {
  uint32_t input_qid = 0;
  uint32_t output_qid = 0;
  std::map<std::string, std::string> attrs;
  attrs["_eschedProcessPriority"] = "0";
  attrs["_eschedEventPriority"] = "2";
  std::string batch_model_path = CreateOnlyOneBatchModel(input_qid, output_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  EXPECT_EQ(batch_models[0]->model_esched_process_priority_, 0);
  EXPECT_EQ(batch_models[0]->model_esched_event_priority_, 2);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(executor.cpu_num_, 3U);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ScopeGuard executor_guard([&executor]() {
    executor.Stop(true);
    executor.WaitForStop();
    executor.Destroy();
  });

  std::vector<int64_t> shape = {1, 2, 3, 4};
  float float_value = 123.1;

  MbufHeadMsg head_msg{0};
  head_msg.flags = 1;
  head_msg.msg_type = static_cast<uint16_t>(MsgType::MSG_TYPE_TENSOR_LIST);
  uint64_t size = 64;
  uint8_t input_data[size];
  DataEnqueue(input_qid, {static_cast<int64_t>(size)}, TensorDataType::DT_FLOAT, head_msg, input_data);

  void *out_mbuf_ptr = nullptr;
  constexpr uint64_t kMaxWaitInMs = 60 * 1000UL;
  uint64_t wait_in_ms = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    auto drv_ret = halQueueDeQueue(0, output_qid, &out_mbuf_ptr);
    if (drv_ret == DRV_ERROR_NONE) {
      break;
    } else if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
      wait_in_ms += kWaitInMsPerTime;
      continue;
    } else {
      ADD_FAILURE() << "drv_ret=" << drv_ret;
      break;
    }
  }
  Mbuf *out_mbuf = (Mbuf *)out_mbuf_ptr;
  halMbufFree(out_mbuf);
}

TEST_F(FlowFuncExecutorSTest, ReshapeTest) {
  uint32_t input_qid = 0;
  uint32_t output_qid = 0;
  std::map<std::string, std::string> attrs;
  std::string batch_model_path = CreateOnlyOneBatchModel(input_qid, output_qid, attrs, true);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ScopeGuard executor_guard([&executor]() {
    executor.Stop(true);
    executor.WaitForStop();
    executor.Destroy();
  });

  std::vector<int64_t> shape = {1, 2, 3, 4};
  float float_value = 123.1;
  DataEnqueue(input_qid, shape, TensorDataType::DT_FLOAT, float_value);
  void *out_mbuf_ptr = nullptr;

  constexpr uint64_t kMaxWaitInMs = 60 * 1000UL;
  uint64_t wait_in_ms = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    auto drv_ret = halQueueDeQueue(0, output_qid, &out_mbuf_ptr);
    if (drv_ret == DRV_ERROR_NONE) {
      break;
    } else if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
      wait_in_ms += kWaitInMsPerTime;
      continue;
    } else {
      ADD_FAILURE() << "drv_ret=" << drv_ret;
      break;
    }
  }
  ASSERT_NE(out_mbuf_ptr, nullptr) << "wait_in_ms=" << wait_in_ms;
  Mbuf *out_mbuf = (Mbuf *)out_mbuf_ptr;

  std::vector<int64_t> expect_shape = {CalcElementCnt(shape)};
  std::vector<int64_t> expect_output(CalcElementCnt(shape), (int64_t)float_value);
  CheckMbufData(out_mbuf, expect_shape, TensorDataType::DT_INT64, expect_output.data(), expect_output.size());
  halMbufFree(out_mbuf);
}

TEST_F(FlowFuncExecutorSTest, basic_test_dummy_q) {
  uint32_t input_qid = 0;
  uint32_t output_qid = 0;
  std::map<std::string, std::string> attrs;
  attrs["_eschedProcessPriority"] = "0";
  attrs["_eschedEventPriority"] = "2";
  std::string batch_model_path = CreateModelWithDummyOutput(input_qid, output_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  EXPECT_EQ(batch_models[0]->model_esched_process_priority_, 0);
  EXPECT_EQ(batch_models[0]->model_esched_event_priority_, 2);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(executor.cpu_num_, 3U);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ScopeGuard executor_guard([&executor]() {
    executor.Stop(true);
    executor.WaitForStop();
    executor.Destroy();
  });

  std::vector<int64_t> shape = {1, 2, 3, 4};
  float float_value = 123.1;
  DataEnqueue(input_qid, shape, TensorDataType::DT_FLOAT, float_value);
  void *out_mbuf_ptr = nullptr;
  constexpr uint64_t kMaxWaitInMs = 60 * 1000UL;
  uint64_t wait_in_ms = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    auto drv_ret = halQueueDeQueue(0, output_qid, &out_mbuf_ptr);
    if (drv_ret == DRV_ERROR_NONE) {
      break;
    } else if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
      wait_in_ms += kWaitInMsPerTime;
      continue;
    } else {
      ADD_FAILURE() << "drv_ret=" << drv_ret;
      break;
    }
  }
  ASSERT_NE(out_mbuf_ptr, nullptr) << "wait_in_ms=" << wait_in_ms;
  Mbuf *out_mbuf = (Mbuf *)out_mbuf_ptr;
  std::vector<int64_t> expect_output(CalcElementCnt(shape), (int64_t)float_value);
  CheckMbufData(out_mbuf, shape, TensorDataType::DT_INT64, expect_output.data(), expect_output.size());
  halMbufFree(out_mbuf);
}

TEST_F(FlowFuncExecutorSTest, data_size_over_tensor_size) {
  uint32_t input_qid = 0;
  uint32_t output_qid = 0;
  std::string batch_model_path = CreateOnlyOneBatchModel(input_qid, output_qid);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ScopeGuard executor_guard([&executor]() {
    executor.Stop(true);
    executor.WaitForStop();
    executor.Destroy();
  });

  std::vector<int64_t> shape = {1, 2, 3, 4};
  int64_t data_size = CalcDataSize(shape, TensorDataType::DT_FLOAT);
  Mbuf *mbuf = nullptr;
  auto drv_ret = halMbufAllocEx(sizeof(RuntimeTensorDesc) + data_size + 1, 0, 0, 0, &mbuf);
  EXPECT_EQ(drv_ret, DRV_ERROR_NONE);
  SetMbufTensorDesc(mbuf, shape, TensorDataType::DT_FLOAT);
  drv_ret = halQueueEnQueue(0, input_qid, mbuf);
  EXPECT_EQ(drv_ret, DRV_ERROR_NONE);
  void *out_mbuf_ptr = nullptr;
  constexpr uint64_t kMaxWaitInMs = 1000UL;
  uint64_t wait_in_ms = 0;
  while (wait_in_ms <= kMaxWaitInMs) {
    auto drv_ret = halQueueDeQueue(0, output_qid, &out_mbuf_ptr);
    if (drv_ret == DRV_ERROR_NONE) {
      break;
    } else if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
      wait_in_ms += kWaitInMsPerTime;
      continue;
    } else {
      ADD_FAILURE() << "drv_ret=" << drv_ret;
      break;
    }
  }
  ASSERT_NE(out_mbuf_ptr, nullptr) << "wait_in_ms=" << wait_in_ms;
  ;
  Mbuf *out_mbuf = (Mbuf *)out_mbuf_ptr;
  halMbufFree(out_mbuf);
}

TEST_F(FlowFuncExecutorSTest, data_size_less_than_tensor_size) {
  uint32_t input_qid = 0;
  uint32_t output_qid = 0;
  std::string batch_model_path = CreateOnlyOneBatchModel(input_qid, output_qid);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ScopeGuard executor_guard([&executor]() {
    executor.Stop(true);
    executor.WaitForStop();
    executor.Destroy();
  });
  std::vector<int64_t> shape = {1, 2, 3, 4};
  int64_t data_size = CalcDataSize(shape, TensorDataType::DT_FLOAT);
  Mbuf *mbuf = nullptr;
  auto drv_ret = halMbufAllocEx(sizeof(RuntimeTensorDesc) + data_size - 1, 0, 0, 0, &mbuf);
  EXPECT_EQ(drv_ret, DRV_ERROR_NONE);
  SetMbufTensorDesc(mbuf, shape, TensorDataType::DT_FLOAT);
  drv_ret = halQueueEnQueue(0, input_qid, mbuf);
  EXPECT_EQ(drv_ret, DRV_ERROR_NONE);
  void *out_mbuf_ptr = nullptr;
  constexpr uint64_t kMaxWaitInMs = 1000UL;
  uint64_t wait_in_ms = 0;
  while (wait_in_ms <= kMaxWaitInMs) {
    auto drv_ret = halQueueDeQueue(0, output_qid, &out_mbuf_ptr);
    if (drv_ret == DRV_ERROR_NONE) {
      break;
    } else if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
      wait_in_ms += kWaitInMsPerTime;
      continue;
    } else {
      ADD_FAILURE() << "drv_ret=" << drv_ret;
      break;
    }
  }
  ASSERT_NE(out_mbuf_ptr, nullptr);
  Mbuf *out_mbuf = (Mbuf *)out_mbuf_ptr;
  halMbufFree(out_mbuf);
}

TEST_F(FlowFuncExecutorSTest, shape_negative) {
  uint32_t input_qid = 0;
  uint32_t output_qid = 0;
  std::string batch_model_path = CreateOnlyOneBatchModel(input_qid, output_qid);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ScopeGuard executor_guard([&executor]() {
    executor.Stop(true);
    executor.WaitForStop();
    executor.Destroy();
  });

  std::vector<int64_t> shape = {1, -2, 3, 4};
  Mbuf *mbuf = nullptr;
  auto drv_ret = halMbufAllocEx(sizeof(RuntimeTensorDesc) + 100, 0, 0, 0, &mbuf);
  EXPECT_EQ(drv_ret, DRV_ERROR_NONE);
  SetMbufTensorDesc(mbuf, shape, TensorDataType::DT_FLOAT);
  drv_ret = halQueueEnQueue(0, input_qid, mbuf);
  EXPECT_EQ(drv_ret, DRV_ERROR_NONE);
  void *out_mbuf_ptr = nullptr;

  constexpr uint64_t kMaxWaitInMs = 1000UL;
  uint64_t wait_in_ms = 0;
  while (wait_in_ms <= kMaxWaitInMs) {
    auto drv_ret = halQueueDeQueue(0, output_qid, &out_mbuf_ptr);
    if (drv_ret == DRV_ERROR_NONE) {
      break;
    } else if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
      wait_in_ms += kWaitInMsPerTime;
      continue;
    } else {
      ADD_FAILURE() << "drv_ret=" << drv_ret;
      break;
    }
  }
  ASSERT_EQ(out_mbuf_ptr, nullptr);
}

TEST_F(FlowFuncExecutorSTest, alloc_tensor_success) {
  auto tensor = FlowBufferFactory::AllocTensor({1, 2}, TensorDataType::DT_INT32);
  EXPECT_NE(tensor, nullptr);
  auto tensor_data = tensor->GetData();
  EXPECT_NE(tensor_data, nullptr);
  auto tensor_shape = tensor->GetShape();
  EXPECT_EQ(tensor_shape[0], 1);
  EXPECT_EQ(tensor_shape[1], 2);
}

TEST_F(FlowFuncExecutorSTest, alloc_tensor_failed) {
  MOCKER(halMbufGetPrivInfo).stubs().will(returnValue((int)DRV_ERROR_PARA_ERROR));
  auto tensor = FlowBufferFactory::AllocTensor({1, 2}, TensorDataType::DT_INT32);
  EXPECT_EQ(tensor, nullptr);
}

TEST_F(FlowFuncExecutorSTest, mbug_flow_msg_data_label_test) {
  auto mbuf_flow_msg = MbufFlowMsg::AllocRawDataMsg(1, 0);
  EXPECT_NE(mbuf_flow_msg, nullptr);
  uint32_t data_label = 1000;
  mbuf_flow_msg->SetDataLabel(data_label);
  std::string debug_str = mbuf_flow_msg->DebugString();
  EXPECT_TRUE(debug_str.find("data_label=") != std::string::npos);
}

TEST_F(FlowFuncExecutorSTest, mbug_flow_msg_tensor_data_size) {
  MbufHead head{};
  auto mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg({2, 3}, TensorDataType::DT_INT32, 0, head);
  EXPECT_NE(mbuf_flow_msg, nullptr);
  uint32_t data_label = 1000;
  mbuf_flow_msg->SetDataLabel(data_label);
  std::string debug_str = mbuf_flow_msg->DebugString();
  EXPECT_TRUE(debug_str.find("data_size=24") != std::string::npos);
}

TEST_F(FlowFuncExecutorSTest, mbug_flow_msg_tensor_list) {
  MbufHead head{};
  auto mbuf_flow_msg = MbufFlowMsg::AllocTensorListMsg(
      {{2, 3}, {2, 2}}, {TensorDataType::DT_INT32, TensorDataType::DT_INT32}, 0, head, 64);
  EXPECT_NE(mbuf_flow_msg, nullptr);
  auto tensor_list = mbuf_flow_msg->GetTensorList();
  EXPECT_EQ(tensor_list.size(), 2);
  EXPECT_EQ(tensor_list[0]->GetDataBufferSize(), 64);
  EXPECT_EQ(tensor_list[0]->GetDataSize(), 24);
  EXPECT_EQ(tensor_list[0]->GetElementCnt(), 6);
  EXPECT_EQ(tensor_list[1]->GetDataBufferSize(), 64);
  EXPECT_EQ(tensor_list[1]->GetDataSize(), 16);
  EXPECT_EQ(tensor_list[1]->GetElementCnt(), 4);
}

TEST_F(FlowFuncExecutorSTest, raw_data_alloc_param_check) {
  auto mbuf_flow_msg = MbufFlowMsg::AllocRawDataMsg(-1, 0);
  EXPECT_EQ(mbuf_flow_msg, nullptr);
}

TEST_F(FlowFuncExecutorSTest, raw_data_alloc_alloc_failed) {
  MOCKER(halMbufAllocEx).stubs().will(returnValue((int)DRV_ERROR_INNER_ERR));
  auto mbuf_flow_msg = MbufFlowMsg::AllocRawDataMsg(100, 0);
  EXPECT_EQ(mbuf_flow_msg, nullptr);
}

TEST_F(FlowFuncExecutorSTest, init_drv_failed) {
  MOCKER(halQueueInit).stubs().will(returnValue(3));
  uint32_t input_qid = 0;
  uint32_t output_qid = 0;
  std::map<std::string, std::string> attrs;
  attrs["_eschedProcessPriority"] = "0";
  attrs["_eschedEventPriority"] = "2";
  std::string batch_model_path = CreateOnlyOneBatchModel(input_qid, output_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  EXPECT_EQ(batch_models[0]->model_esched_process_priority_, 0);
  EXPECT_EQ(batch_models[0]->model_esched_event_priority_, 2);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_ERR_QUEUE_ERROR);
}

TEST_F(FlowFuncExecutorSTest, full_to_not_full) {
  uint32_t input_qid = 0;
  uint32_t output_qid = 0;
  std::string batch_model_path = CreateOnlyOneBatchModel(input_qid, output_qid);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ScopeGuard executor_guard([&executor]() {
    executor.Stop(true);
    executor.WaitForStop();
    executor.Destroy();
  });

  float float_value = 123.1;
  for (int64_t i = 0; i < UDF_ST_QUEUE_MAX_DEPTH * 2; ++i) {
    DataEnqueue(input_qid, {i}, TensorDataType::DT_FLOAT, float_value);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  for (int64_t i = 0; i < UDF_ST_QUEUE_MAX_DEPTH * 2; ++i) {
    void *out_mbuf_ptr = nullptr;
    constexpr uint64_t kMaxWaitInMs = 120 * 1000UL;
    uint64_t wait_in_ms = 0;
    while (wait_in_ms < kMaxWaitInMs) {
      auto drv_ret = halQueueDeQueue(0, output_qid, &out_mbuf_ptr);
      if (drv_ret == DRV_ERROR_NONE) {
        break;
      } else if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
        wait_in_ms += kWaitInMsPerTime;
        continue;
      } else {
        ADD_FAILURE() << "drv_ret=" << drv_ret;
        break;
      }
    }
    ASSERT_NE(out_mbuf_ptr, nullptr) << "wait_in_ms=" << wait_in_ms;
    ;
    Mbuf *out_mbuf = (Mbuf *)out_mbuf_ptr;
    std::vector<int64_t> expect_output(CalcElementCnt({i}), (int64_t)float_value);
    CheckMbufData(out_mbuf, {i}, TensorDataType::DT_INT64, expect_output.data(), expect_output.size());
    halMbufFree(out_mbuf);
  }
}

TEST_F(FlowFuncExecutorSTest, Start_Failed) {
  uint32_t input_qid = 0;
  uint32_t output_qid = 0;
  std::string batch_model_path = CreateOnlyOneBatchModel(input_qid, output_qid);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  MOCKER(halEschedSubmitEvent).stubs().will(returnValue(DRV_ERROR_NO_SUBSCRIBE_THREAD));
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_ERR_DRV_ERROR);
  executor.Stop(true);
  executor.WaitForStop();
}

TEST_F(FlowFuncExecutorSTest, default_get_trans_id) {
  DefaultFlowMsg default_flow_msg;
  EXPECT_EQ(default_flow_msg.GetTransactionId(), UINT64_MAX);
}
TEST_F(FlowFuncExecutorSTest, default_get_tensor_list) {
  DefaultFlowMsg default_flow_msg;
  EXPECT_EQ(default_flow_msg.GetTensorList().empty(), true);
}

TEST_F(FlowFuncExecutorSTest, default_get_raw_data) {
  DefaultFlowMsg default_flow_msg;
  default_flow_msg.SetMsgType(MsgType::MSG_TYPE_RAW_MSG);
  void *data;
  uint64_t len = 0;
  EXPECT_EQ(default_flow_msg.GetRawData(data, len), FLOW_FUNC_ERR_NOT_SUPPORT);
}

TEST_F(FlowFuncExecutorSTest, default_run_context_test) {
  DefaultMetaRunContext default_context;
  // void method, for cover
  auto ret = default_context.AllocRawDataMsg(1, 1);
  EXPECT_EQ(ret, nullptr);
  DefaultMetaParams default_params;
  auto ins_id = default_params.GetRunningInstanceId();
  EXPECT_EQ(ins_id, -1);
  auto ins_num = default_params.GetRunningInstanceNum();
  EXPECT_EQ(ins_num, -1);
}

TEST_F(FlowFuncExecutorSTest, basic_test_with_exception_msg) {
  uint32_t input_qid = 0;
  uint32_t output_qid = 0;
  std::map<std::string, std::string> attrs;
  attrs["_eschedProcessPriority"] = "0";
  attrs["_eschedEventPriority"] = "2";
  std::string batch_model_path = CreateOnlyOneBatchModel(input_qid, output_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  batch_models[0]->model_param_.exception_catch = true;
  batch_models[0]->input_align_attrs_.align_max_cache_num = 2;
  batch_models[0]->input_align_attrs_.align_timeout = 20;
  batch_models[0]->input_align_attrs_.drop_when_not_align = false;

  EXPECT_EQ(batch_models.size(), 1);
  EXPECT_EQ(batch_models[0]->model_esched_process_priority_, 0);
  EXPECT_EQ(batch_models[0]->model_esched_event_priority_, 2);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  executor.status_output_queue_map_[0] = {status_queue_id};
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  void *rsp_mbuff = nullptr;
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ScopeGuard executor_guard([&executor]() {
    executor.Stop(true);
    executor.WaitForStop();
    executor.Destroy();
  });

  std::vector<int64_t> shape = {1, 2, 3, 4};
  float float_value = 123.1;

  const uint32_t priv_size = 256;
  uint8_t priv_info[priv_size];
  uint64_t transid = 100;
  MbufHeadMsg *head_msg = reinterpret_cast<MbufHeadMsg *>(priv_info + priv_size - sizeof(MbufHeadMsg));
  head_msg->trans_id = transid;
  ff::deployer::ExecutorRequest executor_request;
  auto exp_notify = executor_request.mutable_exception_request();
  auto exp_request = exp_notify->mutable_exception_notify();
  ASSERT_NE(exp_request, nullptr);
  exp_request->set_type(1);
  exp_request->set_exception_code(-1);
  exp_request->set_trans_id(transid);
  exp_request->set_scope("");
  exp_request->set_user_context_id(101);
  exp_request->set_exception_context(&priv_info[0], priv_size);

  EnqueueControlMsg(req_queue_id, executor_request);

  constexpr uint64_t kMaxWaitInMs = 60 * 1000UL;
  uint64_t wait_in_ms = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    auto drv_ret = halQueueDeQueue(0, rsp_queue_id, &rsp_mbuff);
    if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
      wait_in_ms += kWaitInMsPerTime;
      continue;
    } else if (drv_ret == DRV_ERROR_NONE) {
      ASSERT_NE(rsp_mbuff, nullptr) << "wait_in_ms=" << wait_in_ms;
      void *data_ptr = nullptr;
      EXPECT_EQ(halMbufGetBuffAddr(rsp_mbuff, &data_ptr), DRV_ERROR_NONE);
      uint64_t data_len = 0UL;
      EXPECT_EQ(halMbufGetBuffSize(rsp_mbuff, &data_len), DRV_ERROR_NONE);
      google::protobuf::io::ArrayInputStream stream(data_ptr, static_cast<int32_t>(data_len));
      ff::deployer::ExecutorResponse response;
      EXPECT_TRUE(response.ParseFromZeroCopyStream(&stream));
      EXPECT_EQ(response.error_code(), 0);
      EXPECT_EQ(response.error_message(), "Execute exception message success.");
      halMbufFree(rsp_mbuff);
      break;
    } else {
      ADD_FAILURE() << "drv_ret=" << drv_ret;
      break;
    }
  }
  exp_request->set_type(0);
  transid = 101;
  head_msg->trans_id = transid;
  exp_request->set_trans_id(transid);
  exp_request->set_exception_context(&priv_info[0], priv_size);
  EnqueueControlMsg(req_queue_id, executor_request);
  wait_in_ms = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    auto drv_ret = halQueueDeQueue(0, rsp_queue_id, &rsp_mbuff);
    if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
      wait_in_ms += kWaitInMsPerTime;
      continue;
    } else if (drv_ret == DRV_ERROR_NONE) {
      ASSERT_NE(rsp_mbuff, nullptr) << "wait_second=" << wait_in_ms;
      void *data_ptr = nullptr;
      EXPECT_EQ(halMbufGetBuffAddr(rsp_mbuff, &data_ptr), DRV_ERROR_NONE);
      uint64_t data_len = 0UL;
      EXPECT_EQ(halMbufGetBuffSize(rsp_mbuff, &data_len), DRV_ERROR_NONE);
      google::protobuf::io::ArrayInputStream stream(data_ptr, static_cast<int32_t>(data_len));
      ff::deployer::ExecutorResponse response;
      EXPECT_TRUE(response.ParseFromZeroCopyStream(&stream));
      EXPECT_EQ(response.error_code(), 0);
      EXPECT_EQ(response.error_message(), "Execute exception message success.");
      halMbufFree(rsp_mbuff);
      break;
    } else {
      ADD_FAILURE() << "drv_ret=" << drv_ret;
      break;
    }
  }
  DataEnqueue(input_qid, shape, TensorDataType::DT_FLOAT, float_value);
  DataEnqueue(input_qid, shape, TensorDataType::DT_FLOAT, float_value);
  void *out_mbuf_ptr = nullptr;
  while (true) {
    auto drv_ret = halQueueDeQueue(0, output_qid, &out_mbuf_ptr);
    if (drv_ret == DRV_ERROR_NONE) {
      ASSERT_NE(out_mbuf_ptr, nullptr);
      Mbuf *out_mbuf = (Mbuf *)out_mbuf_ptr;
      std::vector<int64_t> expect_output(CalcElementCnt(shape), (int64_t)float_value);
      CheckMbufData(out_mbuf, shape, TensorDataType::DT_INT64, expect_output.data(), expect_output.size());
      halMbufFree(out_mbuf);
      continue;
    } else if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
      break;
    } else {
      ADD_FAILURE() << "drv_ret=" << drv_ret;
      break;
    }
  }
}

TEST_F(FlowFuncExecutorSTest, basic_test_with_suspend_msg) {
  uint32_t input_qid = 0;
  uint32_t output_qid = 0;
  std::map<std::string, std::string> attrs;
  attrs["_eschedProcessPriority"] = "0";
  attrs["_eschedEventPriority"] = "2";
  std::string batch_model_path = CreateOnlyOneBatchModel(input_qid, output_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  batch_models[0]->input_align_attrs_.align_max_cache_num = 2;
  batch_models[0]->input_align_attrs_.align_timeout = 2000;
  batch_models[0]->input_align_attrs_.drop_when_not_align = false;
  EXPECT_EQ(batch_models.size(), 1);
  EXPECT_EQ(batch_models[0]->model_esched_process_priority_, 0);
  EXPECT_EQ(batch_models[0]->model_esched_event_priority_, 2);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  void *rsp_mbuff = nullptr;
  EXPECT_EQ(executor.cpu_num_, 3U);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ScopeGuard executor_guard([&executor]() {
    executor.Stop(true);
    executor.WaitForStop();
    executor.Destroy();
  });

  std::vector<int64_t> shape = {1, 2, 3, 4};
  float float_value = 123.1;
  DataEnqueue(input_qid, shape, TensorDataType::DT_FLOAT, float_value);
  DataEnqueue(input_qid, shape, TensorDataType::DT_FLOAT, float_value);

  ff::deployer::ExecutorRequest executor_request;
  auto control_msg = executor_request.mutable_clear_model_message();
  ASSERT_NE(control_msg, nullptr);
  control_msg->set_clear_msg_type(1);
  control_msg->set_model_id(1);
  EnqueueControlMsg(req_queue_id, executor_request);

  constexpr uint64_t kMaxWaitInMs = 60 * 1000UL;
  uint64_t wait_in_ms = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    auto drv_ret = halQueueDeQueue(0, rsp_queue_id, &rsp_mbuff);
    if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
      wait_in_ms += kWaitInMsPerTime;
      continue;
    } else if (drv_ret == DRV_ERROR_NONE) {
      ASSERT_NE(rsp_mbuff, nullptr) << "wait_in_ms=" << wait_in_ms;
      void *data_ptr = nullptr;
      EXPECT_EQ(halMbufGetBuffAddr(rsp_mbuff, &data_ptr), DRV_ERROR_NONE);
      uint64_t data_len = 0UL;
      EXPECT_EQ(halMbufGetBuffSize(rsp_mbuff, &data_len), DRV_ERROR_NONE);
      google::protobuf::io::ArrayInputStream stream(data_ptr, static_cast<int32_t>(data_len));
      ff::deployer::ExecutorResponse response;
      EXPECT_TRUE(response.ParseFromZeroCopyStream(&stream));
      EXPECT_EQ(response.error_code(), 0);
      EXPECT_EQ(response.error_message(), "Execute suspend success.");
      halMbufFree(rsp_mbuff);
      break;
    } else {
      ADD_FAILURE() << "drv_ret=" << drv_ret;
      break;
    }
  }
  void *out_mbuf_ptr = nullptr;
  while (true) {
    auto drv_ret = halQueueDeQueue(0, output_qid, &out_mbuf_ptr);
    if (drv_ret == DRV_ERROR_NONE) {
      ASSERT_NE(out_mbuf_ptr, nullptr);
      Mbuf *out_mbuf = (Mbuf *)out_mbuf_ptr;
      std::vector<int64_t> expect_output(CalcElementCnt(shape), (int64_t)float_value);
      CheckMbufData(out_mbuf, shape, TensorDataType::DT_INT64, expect_output.data(), expect_output.size());
      halMbufFree(out_mbuf);
      continue;
    } else if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
      break;
    } else {
      ADD_FAILURE() << "drv_ret=" << drv_ret;
      break;
    }
  }
}

TEST_F(FlowFuncExecutorSTest, basic_test_with_suspend_and_recover_msg) {
  uint32_t input_qid = 0;
  uint32_t output_qid = 0;
  std::map<std::string, std::string> attrs;
  attrs["_eschedProcessPriority"] = "0";
  attrs["_eschedEventPriority"] = "2";
  std::string batch_model_path = CreateOnlyOneBatchModel(input_qid, output_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  EXPECT_EQ(batch_models[0]->model_esched_process_priority_, 0);
  EXPECT_EQ(batch_models[0]->model_esched_event_priority_, 2);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  void *rsp_mbuff = nullptr;
  EXPECT_EQ(executor.cpu_num_, 3U);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ScopeGuard executor_guard([&executor]() {
    executor.Stop(true);
    executor.WaitForStop();
    executor.Destroy();
  });

  std::vector<int64_t> shape = {1, 2, 3, 4};
  float float_value = 123.1;
  DataEnqueue(input_qid, shape, TensorDataType::DT_FLOAT, float_value);
  DataEnqueue(input_qid, shape, TensorDataType::DT_FLOAT, float_value);

  ff::deployer::ExecutorRequest executor_request;
  auto control_msg = executor_request.mutable_clear_model_message();
  ASSERT_NE(control_msg, nullptr);
  control_msg->set_clear_msg_type(1);
  control_msg->set_model_id(1);
  EnqueueControlMsg(req_queue_id, executor_request);
  constexpr uint64_t kMaxWaitInMs = 60 * 1000UL;
  uint64_t wait_in_ms = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    auto drv_ret = halQueueDeQueue(0, rsp_queue_id, &rsp_mbuff);
    if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
      wait_in_ms += kWaitInMsPerTime;
      continue;
    } else if (drv_ret == DRV_ERROR_NONE) {
      ASSERT_NE(rsp_mbuff, nullptr) << "wait_in_ms=" << wait_in_ms;
      void *data_ptr = nullptr;
      EXPECT_EQ(halMbufGetBuffAddr(rsp_mbuff, &data_ptr), DRV_ERROR_NONE);
      uint64_t data_len = 0UL;
      EXPECT_EQ(halMbufGetBuffSize(rsp_mbuff, &data_len), DRV_ERROR_NONE);
      google::protobuf::io::ArrayInputStream stream(data_ptr, static_cast<int32_t>(data_len));
      ff::deployer::ExecutorResponse response;
      EXPECT_TRUE(response.ParseFromZeroCopyStream(&stream));
      EXPECT_EQ(response.error_code(), 0);
      EXPECT_EQ(response.error_message(), "Execute suspend success.");
      halMbufFree(rsp_mbuff);
      break;
    } else {
      ADD_FAILURE() << "drv_ret=" << drv_ret;
      break;
    }
  }
  DataEnqueue(input_qid, shape, TensorDataType::DT_FLOAT, float_value);
  DataEnqueue(input_qid, shape, TensorDataType::DT_FLOAT, float_value);

  control_msg->set_clear_msg_type(2);
  control_msg->set_model_id(1);
  EnqueueControlMsg(req_queue_id, executor_request);
  wait_in_ms = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    auto drv_ret = halQueueDeQueue(0, rsp_queue_id, &rsp_mbuff);
    if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
      wait_in_ms += kWaitInMsPerTime;
      continue;
    } else if (drv_ret == DRV_ERROR_NONE) {
      ASSERT_NE(rsp_mbuff, nullptr) << "wait_in_ms=" << wait_in_ms;
      void *data_ptr = nullptr;
      EXPECT_EQ(halMbufGetBuffAddr(rsp_mbuff, &data_ptr), DRV_ERROR_NONE);
      uint64_t data_len = 0UL;
      EXPECT_EQ(halMbufGetBuffSize(rsp_mbuff, &data_len), DRV_ERROR_NONE);
      google::protobuf::io::ArrayInputStream stream(data_ptr, static_cast<int32_t>(data_len));
      ff::deployer::ExecutorResponse response;
      EXPECT_TRUE(response.ParseFromZeroCopyStream(&stream));
      EXPECT_EQ(response.error_code(), 0);
      EXPECT_EQ(response.error_message(), "Execute recover success.");
      halMbufFree(rsp_mbuff);
      break;
    } else {
      ADD_FAILURE() << "drv_ret=" << drv_ret;
      break;
    }
  }
  DataEnqueue(input_qid, shape, TensorDataType::DT_FLOAT, float_value);
  void *out_mbuf_ptr = nullptr;
  while (true) {
    auto drv_ret = halQueueDeQueue(0, output_qid, &out_mbuf_ptr);
    if (drv_ret == DRV_ERROR_NONE) {
      ASSERT_NE(out_mbuf_ptr, nullptr);
      Mbuf *out_mbuf = (Mbuf *)out_mbuf_ptr;
      std::vector<int64_t> expect_output(CalcElementCnt(shape), (int64_t)float_value);
      CheckMbufData(out_mbuf, shape, TensorDataType::DT_INT64, expect_output.data(), expect_output.size());
      halMbufFree(out_mbuf);
      continue;
    } else if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
      break;
    } else {
      ADD_FAILURE() << "drv_ret=" << drv_ret;
      break;
    }
  }
}

TEST_F(FlowFuncExecutorSTest, basic_test_with_dump) {
  UdfDumpManager::Instance().SetHostPid(9999);
  UdfDumpManager::Instance().EnableDump();
  UdfDumpManager::Instance().SetDumpPath("./");
  UdfDumpManager::Instance().SetDeviceId(0);
  EXPECT_EQ(UdfDumpManager::Instance().Init(), FLOW_FUNC_SUCCESS);
  uint32_t input_qid = 0;
  uint32_t output_qid = 0;
  std::map<std::string, std::string> attrs;
  attrs["_eschedProcessPriority"] = "0";
  attrs["_eschedEventPriority"] = "2";
  std::string batch_model_path = CreateOnlyOneBatchModel(input_qid, output_qid, attrs);
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  EXPECT_EQ(batch_models[0]->model_esched_process_priority_, 0);
  EXPECT_EQ(batch_models[0]->model_esched_event_priority_, 2);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(executor.cpu_num_, 3U);
  ret = executor.Start();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ScopeGuard executor_guard([&executor]() {
    executor.Stop(true);
    executor.WaitForStop();
    executor.Destroy();
  });

  std::vector<int64_t> shape = {1, 2, 3, 4};
  float float_value = 123.1;
  DataEnqueue(input_qid, shape, TensorDataType::DT_FLOAT, float_value);
  void *out_mbuf_ptr = nullptr;

  constexpr uint64_t kMaxWaitInMs = 60 * 1000UL;
  uint64_t wait_in_ms = 0;
  while (wait_in_ms < kMaxWaitInMs) {
    auto drv_ret = halQueueDeQueue(0, output_qid, &out_mbuf_ptr);
    if (drv_ret == DRV_ERROR_NONE) {
      break;
    } else if (drv_ret == DRV_ERROR_QUEUE_EMPTY) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kWaitInMsPerTime));
      wait_in_ms += kWaitInMsPerTime;
      continue;
    } else {
      ADD_FAILURE() << "drv_ret=" << drv_ret;
      break;
    }
  }
  ASSERT_NE(out_mbuf_ptr, nullptr) << "wait_second=" << wait_in_ms;
  Mbuf *out_mbuf = (Mbuf *)out_mbuf_ptr;
  std::vector<int64_t> expect_output(CalcElementCnt(shape), (int64_t)float_value);
  CheckMbufData(out_mbuf, shape, TensorDataType::DT_INT64, expect_output.data(), expect_output.size());
  halMbufFree(out_mbuf);
  UdfDumpManager::Instance().ClearDumpInfo();
}

TEST_F(FlowFuncExecutorSTest, FlowFuncExecutor_Start_Failed) {
  FlowFuncExecutor executor;
  GlobalConfig::Instance().SetRunOnAiCpu(true);
  MOCKER(halGetDeviceInfo).stubs().will(returnValue(DRV_ERROR_NOT_SUPPORT)).then(returnValue(DRV_ERROR_NONE));
  auto ret = executor.Start();
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
  GlobalConfig::Instance().SetRunOnAiCpu(false);

  MOCKER_CPP(&FlowFuncManager::Init).stubs().will(returnValue(FLOW_FUNC_FAILED)).then(returnValue(0));
  ret = executor.Start();
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);

  MOCKER_CPP(&FlowFuncLoggerManager::Init).stubs().will(returnValue(FLOW_FUNC_FAILED));
  ret = executor.Start();
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
  executor.Stop(true);
  executor.WaitForStop();
  executor.Destroy();
}

TEST_F(FlowFuncExecutorSTest, ThreadLoop_SubscribeWaitEventMask_failed) {
  MOCKER(halEschedSubscribeEvent)
      .stubs()
      .will(returnValue(DRV_ERROR_NONE))
      .then(returnValue(DRV_ERROR_NONE))
      .then(returnValue(DRV_ERROR_INNER_ERR));
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.ThreadLoop(0);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorSTest, ProcessProcessorInitEvent_Failed) {
  MOCKER(halQueueAttach).stubs().will(returnValue(DRV_ERROR_INNER_ERR));
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.dev_output_queue_map_[0][0U] = false;
  struct event_info event {};
  executor.ProcessProcessorInitEvent(event, 0);
  EXPECT_FALSE(executor.running_);

  FlowFuncExecutor executor1;
  executor1.running_ = true;
  executor1.status_output_queue_map_[0] = {0U};
  executor1.ProcessProcessorInitEvent(event, 0);
  EXPECT_FALSE(executor1.running_);

  FlowFuncExecutor executor2;
  executor2.running_ = true;
  executor2.dev_input_queue_map_[0][0U] = false;
  executor2.ProcessProcessorInitEvent(event, 0);
  EXPECT_FALSE(executor2.running_);
}

TEST_F(FlowFuncExecutorSTest, ThreadLoop_SubscribeInvokeModelEvent_failed) {
  MOCKER(halEschedSubscribeEvent).stubs().will(returnValue(DRV_ERROR_INNER_ERR));
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.ThreadLoop(0);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorSTest, ThreadLoop_InvokeModel_halEschedWaitEventSubscribe_failed) {
  MOCKER(halEschedWaitEvent).stubs().will(returnValue(DRV_ERROR_INNER_ERR));
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.ThreadLoop(0);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorSTest, ThreadLoop_SubscribeFlowMsgQueueEvent_failed) {
  MOCKER(halEschedSubscribeEvent).stubs().will(returnValue(DRV_ERROR_NONE)).then(returnValue(DRV_ERROR_INNER_ERR));
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.ThreadLoop(0);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorSTest, ThreadLoop_FlowMsgQueue_halEschedWaitEventSubscribe_failed) {
  MOCKER(halEschedWaitEvent).stubs().will(returnValue(DRV_ERROR_NONE)).then(returnValue(DRV_ERROR_INNER_ERR));
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.ThreadLoop(0);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorSTest, ThreadLoop_halEschedSubscribeEvent_failed) {
  MOCKER(halEschedWaitEvent)
      .stubs()
      .will(returnValue(DRV_ERROR_NONE))
      .then(returnValue(DRV_ERROR_NONE))
      .then(returnValue(DRV_ERROR_INNER_ERR));
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.ThreadLoop(0);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorSTest, ThreadLoop_seccomp_failed) {
  MOCKER_CPP(&FlowFuncThreadPool::ThreadSecureCompute).stubs().will(returnValue(FLOW_FUNC_FAILED));
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.ThreadLoop(0);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorSTest, ProcessSingleFlowFuncInitEvent_failed) {
  MOCKER_CPP(&FlowFuncProcessor::InitFlowFunc).stubs().will(returnValue(FLOW_FUNC_FAILED)).then(returnValue(0));
  MOCKER_CPP(&FlowFuncExecutor::ScheduleFlowFunc).stubs().will(returnValue(FLOW_FUNC_FAILED));
  std::shared_ptr<FlowFuncParams> param(new FlowFuncParams("test", 1U, 0, 0, 0));
  param->SetNeedReportStatusFlag(true);
  param->SetInstanceName("test@0_1_1@0@0");
  QueueDevInfo queue_dev_info = {0, 0, 0, 0, false};
  param->SetStatusOutputQueue(queue_dev_info);
  auto ret = param->Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowFuncProcessor> processor(new FlowFuncProcessor(
      param, "test", UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({1}), {1U}));
  FlowFuncExecutor executor;
  executor.running_ = true;
  executor.func_processors_.emplace_back(processor);
  struct event_info event {};
  event.comm.subevent_id = 0U;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdFlowFuncReportStatus);
  // InitFlowFunc failed
  executor.ProcessSingleFlowFuncInitEvent(event, 0U);
  EXPECT_FALSE(executor.running_);
  executor.running_ = true;
  // ScheduleFlowFunc failed
  executor.ProcessSingleFlowFuncInitEvent(event, 0U);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorSTest, ProcessRaiseExceptionEvent_failed) {
  FlowFuncExecutor executor;
  event_info event;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdProcessorInit);
  event.comm.subevent_id = 0;
  uint64_t *trans_id = reinterpret_cast<uint64_t *>(event.priv.msg);
  *trans_id = 1;
  event.priv.msg_len = sizeof(uint64_t);
  executor.running_ = true;
  executor.ProcessRaiseExceptionEvent(event, 0);
  EXPECT_EQ(executor.running_, false);

  executor.running_ = true;
  std::shared_ptr<FlowFuncParams> param(new FlowFuncParams("test", 1U, 0, 0, 0));
  param->SetNeedReportStatusFlag(true);
  param->SetInstanceName("test@0_1_1@0@0");
  QueueDevInfo queue_dev_info = {0, 0, 0, 0, false};
  param->SetStatusOutputQueue(queue_dev_info);
  auto ret = param->Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowFuncProcessor> processor(new FlowFuncProcessor(
      param, "test", UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({1}), {1U}));
  executor.func_processors_.emplace_back(processor);
  event.priv.msg_len = sizeof(uint32_t);
  executor.ProcessRaiseExceptionEvent(event, 0);
  EXPECT_EQ(executor.running_, false);
}

TEST_F(FlowFuncExecutorSTest, ProcessReportStatusEvent_failed) {
  MOCKER_CPP(&FlowFuncExecutor::ReportStatus).stubs().will(returnValue(FLOW_FUNC_FAILED));
  FlowFuncExecutor executor;
  executor.running_ = true;
  struct event_info event {};
  event.comm.subevent_id = 0U;
  event.comm.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdFlowFuncReportStatus);
  executor.ProcessReportStatusEvent(event, 0U);
  EXPECT_FALSE(executor.running_);

  executor.running_ = true;
  std::shared_ptr<FlowFuncParams> param(new FlowFuncParams("test", 1U, 0, 0, 0));
  param->SetNeedReportStatusFlag(true);
  param->SetInstanceName("test@0_1_1@0@0");
  QueueDevInfo queue_dev_info = {0, 0, 0, 0, false};
  param->SetStatusOutputQueue(queue_dev_info);
  auto ret = param->Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::shared_ptr<FlowFuncProcessor> processor(new FlowFuncProcessor(
      param, "test", UdfTestHelper::CreateQueueDevInfos({1}), UdfTestHelper::CreateQueueDevInfos({1}), {1U}));
  executor.func_processors_.emplace_back(processor);
  executor.ProcessReportStatusEvent(event, 0U);
  EXPECT_FALSE(executor.running_);
}

TEST_F(FlowFuncExecutorSTest, basic_test_invoked_model_success) {
  std::vector<uint32_t> input_queues;
  std::vector<uint32_t> output_queues;
  std::string batch_model_path = CreateModelWithInvoke(input_queues, output_queues, "scope/model_key", "scope/");
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(FlowFuncExecutorSTest, basic_test_invoked_model_error) {
  std::vector<uint32_t> input_queues;
  std::vector<uint32_t> output_queues;
  std::string batch_model_path = CreateModelWithInvoke(input_queues, output_queues, "model_key", "scope/");
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  EXPECT_EQ(ret, FLOW_FUNC_FAILED);
}

TEST_F(FlowFuncExecutorSTest, basic_invoke_model_test_with_exception_msg) {
  std::vector<uint32_t> input_queues;
  std::vector<uint32_t> output_queues;
  std::string batch_model_path = CreateModelWithInvoke(input_queues, output_queues, "model_key", "");
  auto batch_models = FlowFuncModel::ParseModels(batch_model_path);
  batch_models[0]->model_param_.exception_catch = true;
  batch_models[0]->input_align_attrs_.align_max_cache_num = 2;
  batch_models[0]->input_align_attrs_.align_timeout = 20;
  batch_models[0]->input_align_attrs_.drop_when_not_align = false;

  EXPECT_EQ(batch_models.size(), 1);
  FlowFuncExecutor executor;
  auto ret = executor.Init(batch_models);
  executor.status_output_queue_map_[0] = {status_queue_id};
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  const uint32_t priv_size = 256;
  uint8_t priv_info[priv_size];
  ff::deployer::ExecutorRequest_DataflowExceptionNotify exp_msg;
  exp_msg.set_type(0);
  exp_msg.set_trans_id(100);
  exp_msg.set_scope("scope/");
  exp_msg.set_user_context_id(101);
  exp_msg.set_exception_context(&priv_info[0], priv_size);
  exp_msg.set_exception_code(-1);
  ret = executor.ProcessExceptionMsg(exp_msg);
  exp_msg.set_type(1);
  ret = executor.ProcessExceptionMsg(exp_msg);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
}
}  // namespace FlowFunc

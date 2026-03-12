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
#include <gmock/gmock.h>
#include <vector>
#include "graph/utils/graph_utils.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "mmpa/mmpa_api.h"
#include "framework/common/helper/model_helper.h"

#include "macro_utils/dt_public_scope.h"
#include "common/model/ge_root_model.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "macro_utils/dt_public_unscope.h"
#include "common/op_tiling/op_tiling_rt2.h"
#include "tests/ge/ut/ge/graph/passes/graph_builder_utils.h"
#include "base/graph/manager/graph_manager_utils.h"
#include "graph/ge_local_context.h"
#include "compiler/graph/manager/graph_manager.h"
using namespace std;
using namespace testing;

namespace ge {
namespace {
const char *const kEnvName = "ASCEND_OPP_PATH";
const string kOpsProto = "libopsproto_rt2.0.so";
const string kOpMaster = "libopmaster_rt2.0.so";
const string kInner = "built-in";
const string kOpsProtoPath = "/op_proto/lib/linux/x86_64/";
const string kOpMasterPath = "/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/";
}

class UtestGeRootModel : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestGeRootModel, LoadBinDataSuccess) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1);
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize' > " + path_config).c_str());

  std::string inner_proto_path = opp_path + kInner + kOpsProtoPath;
  system(("mkdir -p " + inner_proto_path).c_str());
  inner_proto_path += kOpsProto;
  system(("touch " + inner_proto_path).c_str());
  system(("echo 'ops proto:123 ' > " + inner_proto_path).c_str());

  std::string inner_tiling_path = opp_path + kInner + kOpMasterPath;
  system(("mkdir -p " + inner_tiling_path).c_str());
  inner_tiling_path += kOpMaster;
  system(("touch " + inner_tiling_path).c_str());
  system(("echo 'op tiling:456 ' > " + inner_tiling_path).c_str());

  ModelHelper model_helper;
  string cpu_info = "x86_64";
  string os_info = "linux";
  auto ret = model_helper.GetSoBinData(cpu_info, os_info);
  EXPECT_EQ(ret, SUCCESS);
  system(("rm -rf " + path_vendors).c_str());
}

TEST_F(UtestGeRootModel, CheckSoInDynamicSuccsess) {
  auto root_graph = std::make_shared<ComputeGraph>("root-graph");
  auto root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(root_model->Initialize(root_graph), SUCCESS);

  AttrUtils::SetBool(root_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);

  EXPECT_EQ(root_model->CheckAndSetNeedSoInOM(), SUCCESS);
  EXPECT_EQ(root_model->GetSoInOmFlag(), 0x8000);
}

TEST_F(UtestGeRootModel, CheckSoInStaticSuccsess) {
  auto root_graph = std::make_shared<ComputeGraph>("root-graph");
  auto root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(root_model->Initialize(root_graph), SUCCESS);

  OpDescPtr dy_op = std::make_shared<OpDesc>("padv4", "PadV4");
  vector<int64_t> dims = {1, 2, 3, 4};
  vector<int64_t> dims2 = {1, 1, 3, 4, 16};
  GeShape shape(dims);
  GeShape shape2(dims2);
  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NC1HWC0);
  in_desc1.SetOriginFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  in_desc1.SetOriginDataType(DT_FLOAT);
  in_desc1.SetShape(shape2);
  in_desc1.SetOriginShape(shape);
  dy_op->AddInputDesc("x", in_desc1);
  dy_op->AddOutputDesc("Y", in_desc1);
  NodePtr dy_node = root_graph->AddNode(dy_op);
  (void)ge::AttrUtils::SetBool(dy_op, "_static_to_dynamic_softsync_op", true);
  EXPECT_EQ(root_model->CheckAndSetNeedSoInOM(), SUCCESS);
  EXPECT_EQ(root_model->GetSoInOmFlag(), 0x8000);

  (void)ge::AttrUtils::SetBool(dy_op, "_static_to_dynamic_softsync_op", false);
  // tiling_data
  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("hahahaha");
  dy_op->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);
  EXPECT_EQ(root_model->CheckAndSetNeedSoInOM(), SUCCESS);
  EXPECT_EQ(root_model->GetSoInOmFlag(), 0x8000);
}

TEST_F(UtestGeRootModel, CheckSoInSuccsessRetFalse) {
  auto root_graph = std::make_shared<ComputeGraph>("root-graph");
  auto root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(root_model->Initialize(root_graph), SUCCESS);

  OpDescPtr dy_op = std::make_shared<OpDesc>("padv4", "PadV4");
  vector<int64_t> dims = {1, 2, 3, 4};
  vector<int64_t> dims2 = {1, 1, 3, 4, 16};
  GeShape shape(dims);
  GeShape shape2(dims2);
  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NC1HWC0);
  in_desc1.SetOriginFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  in_desc1.SetOriginDataType(DT_FLOAT);
  in_desc1.SetShape(shape2);
  in_desc1.SetOriginShape(shape);
  dy_op->AddInputDesc("x", in_desc1);
  dy_op->AddOutputDesc("Y", in_desc1);
  NodePtr dy_node = root_graph->AddNode(dy_op);

  EXPECT_EQ(root_model->CheckAndSetNeedSoInOM(), false);
}

TEST_F(UtestGeRootModel, IsNeedMallocFixedFeatureMemByType_ReturnTrue_IfNotSetFixedMem) {
  GeRootModel ge_root_model;
  EXPECT_TRUE(ge_root_model.IsNeedMallocFixedFeatureMemByType(RT_MEMORY_HBM));
}

TEST_F(UtestGeRootModel, IsNeedMallocFixedFeatureMemByType_ReturnTrue_NotSetHbmMemType) {
  GeRootModel ge_root_model;
  EXPECT_TRUE(ge_root_model.IsNeedMallocFixedFeatureMemByType(RT_MEMORY_HBM));
  ge_root_model.MutableFixedFeatureMemory().insert({RT_MEMORY_HBM, {RT_MEMORY_HBM, nullptr, 0U, true, false, false, 0U, nullptr}});
  EXPECT_TRUE(ge_root_model.IsNeedMallocFixedFeatureMemByType(RT_MEMORY_P2P_DDR));
}

TEST_F(UtestGeRootModel, IsNeedMallocFixedFeatureMemByType_ReturnFalse_IfSetFixedAddr) {
  GeRootModel ge_root_model;
  ge_root_model.MutableFixedFeatureMemory().insert({RT_MEMORY_HBM, {RT_MEMORY_HBM, (void *)0x123, 0U, true, false, false, 0U, nullptr}});
  EXPECT_FALSE(ge_root_model.IsNeedMallocFixedFeatureMemByType(RT_MEMORY_HBM));
  EXPECT_TRUE(ge_root_model.IsNeedMallocFixedFeatureMemByType(RT_MEMORY_P2P_DDR));
}

TEST_F(UtestGeRootModel, IsNeedMallocFixedFeatureMemByType_ReturnFalse_IfUserSetNullFixedAddr) {
  GeRootModel ge_root_model;
  ge_root_model.MutableFixedFeatureMemory().insert({RT_MEMORY_HBM, {RT_MEMORY_HBM, nullptr, 0U, true, false, false, 0U, nullptr}});
  EXPECT_FALSE(ge_root_model.IsNeedMallocFixedFeatureMemByType(RT_MEMORY_HBM));

  ge_root_model.MutableFixedFeatureMemory().insert({RT_MEMORY_P2P_DDR, {RT_MEMORY_P2P_DDR, nullptr, 0U, true, false, false, 0U, nullptr}});
  EXPECT_FALSE(ge_root_model.IsNeedMallocFixedFeatureMemByType(RT_MEMORY_P2P_DDR));
}

TEST_F(UtestGeRootModel, IsNeedMallocFixedFeatureMemByType_ReturnFalseByDefault) {
  GeRootModel ge_root_model;
  ge_root_model.MutableFixedFeatureMemory().insert({RT_MEMORY_HBM, {RT_MEMORY_HBM, nullptr, 0U, false, true, false, 0U, nullptr}});
  EXPECT_FALSE(ge_root_model.IsNeedMallocFixedFeatureMemByType(RT_MEMORY_HBM));
}

TEST_F(UtestGeRootModel, IsNeedMallocFixedFeatureMemByType_ReturnTrue_IsGeUseExtendSizeMemory) {
  GeRootModel ge_root_model;
  ge_root_model.MutableFixedFeatureMemory().insert({RT_MEMORY_HBM, {RT_MEMORY_HBM, nullptr, 0U, false, true, false, 0U, nullptr}});
  mmSetEnv("GE_USE_STATIC_MEMORY", "2", 1);
  EXPECT_TRUE(ge_root_model.IsNeedMallocFixedFeatureMemByType(RT_MEMORY_HBM));
  mmSetEnv("GE_USE_STATIC_MEMORY", "0", 1);
}

TEST_F(UtestGeRootModel, ModifySubgraphPtr) {
  auto root_graph = std::make_shared<ComputeGraph>("root-graph");
  for (int i = 0; i < 4; ++i) {
    auto sub = std::make_shared<ComputeGraph>("sub_" + std::to_string(i));
    root_graph->AddSubGraph(sub);
  }
  auto root_model = std::make_shared<GeRootModel>();
  auto ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(root_graph);
  ge_model->SetName(root_graph->GetName());
  root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

  EXPECT_EQ(root_model->Initialize(root_graph), SUCCESS);

  for (int i = 0; i < 4; ++i) {
    auto sub_model = std::make_shared<GeModel>();
    auto sub = std::make_shared<ComputeGraph>("sub_" + std::to_string(i));
    sub_model->SetGraph(sub);
    sub_model->SetName(sub->GetName());
    root_model->SetSubgraphInstanceNameToModel(sub->GetName(), sub_model);
  }
  EXPECT_EQ(root_model->ModifyOwnerGraphForSubModels(), SUCCESS);

  for (const auto &iter : root_model->GetSubgraphInstanceNameToModel()) {
    auto sub_graph = root_graph->GetSubgraph(iter.first);
    if (sub_graph != nullptr) {
      EXPECT_EQ(sub_graph.get(), iter.second->GetGraph().get());
    }
  }
}

TEST_F(UtestGeRootModel, IsNeedMallocFixedFeatureMemByType_ReturnFalse_IfUnknow) {
  GeRootModel ge_root_model;
  auto root_graph = std::make_shared<ComputeGraph>("root-graph");
  (void)AttrUtils::SetBool(root_graph, ATTR_NAME_GRAPH_UNKNOWN_FLAG, true);
  ge_root_model.SetRootGraph(root_graph);
  EXPECT_FALSE(ge_root_model.IsNeedMallocFixedFeatureMem());
}
}

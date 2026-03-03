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
#include <fstream>

#include "macro_utils/dt_public_scope.h"
#include "engines/manager/engine_manager/dnnengine_manager.h"
#include "nlohmann/json.hpp"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "framework/engine/dnnengine.h"
#include "graph/op_desc.h"
#include "graph/node.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "graph/ge_local_context.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;

namespace ge {

class SubDNNEngine : public DNNEngine{
public:
    SubDNNEngine(){init_flag_ = false;};
    virtual ~SubDNNEngine() = default;
    Status Initialize(const std::map<std::string, std::string> &options) const{
    	if (init_flag_){
    		return SUCCESS;
    	}
        return FAILED;
    }

    void GetAttributes(DNNEngineAttribute &attr) const{
    	attr.runtime_type = RuntimeType::DEVICE;
        return;
    }
    bool init_flag_;
};


class SubOpsKernelInfoStore2 : public OpsKernelInfoStore{
public:
    SubOpsKernelInfoStore2(){check_flag_ = true;}
    virtual Status Initialize(const std::map<std::string, std::string> &options);
    virtual Status Finalize();
    virtual void GetAllOpsKernelInfo(std::map<std::string, OpInfo> &infos) const;
    virtual bool CheckSupported(const OpDescPtr &opDescPtr, std::string &un_supported_reason) const;
    virtual bool CheckSupported(const NodePtr &node, std::string &un_supported_reason,
                                CheckSupportFlag &flag) const;
    bool check_flag_;
};

Status SubOpsKernelInfoStore2::Initialize(const std::map<std::string, std::string> &options){
    return FAILED;
}

Status SubOpsKernelInfoStore2::Finalize(){
    return SUCCESS;
}

void SubOpsKernelInfoStore2::GetAllOpsKernelInfo(std::map<std::string, OpInfo> &infos) const{
    infos["opinfo"] = OpInfo();
    return;
}

bool SubOpsKernelInfoStore2::CheckSupported(const OpDescPtr &opDescPtr, std::string &un_supported_reason) const{
    return check_flag_;
}

bool SubOpsKernelInfoStore2::CheckSupported(const NodePtr &node, std::string &un_supported_reason,
                                            CheckSupportFlag &flag) const {
  if (node->GetName() == "not_support_dynamic_shape") {
    flag = CheckSupportFlag::kNotSupportDynamicShape;
    return false;
  }
  return CheckSupported(node->GetOpDesc(), un_supported_reason);
}

class UtestDnnengineManager : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  	DNNEngineManager::GetInstance().engines_map_.clear();
  	OpsKernelManager::GetInstance().ops_kernel_store_.clear();
  	OpsKernelManager::GetInstance().composite_engines_.clear();
  }
 public:
  NodePtr UtAddNode(ComputeGraphPtr &graph, std::string name, std::string type, int in_cnt, int out_cnt){
    auto tensor_desc = std::make_shared<GeTensorDesc>();
    std::vector<int64_t> shape = {1, 1, 224, 224};
    tensor_desc->SetShape(GeShape(shape));
    tensor_desc->SetFormat(FORMAT_NCHW);
    tensor_desc->SetDataType(DT_FLOAT);
    tensor_desc->SetOriginFormat(FORMAT_NCHW);
    tensor_desc->SetOriginShape(GeShape(shape));
    tensor_desc->SetOriginDataType(DT_FLOAT);
    auto op_desc = std::make_shared<OpDesc>(name, type);
    for (int i = 0; i < in_cnt; ++i) {
        op_desc->AddInputDesc(tensor_desc->Clone());
    }
    for (int i = 0; i < out_cnt; ++i) {
        op_desc->AddOutputDesc(tensor_desc->Clone());
    }
    return graph->AddNode(op_desc);
  }
};

TEST_F(UtestDnnengineManager, Normal) {
    auto &instance = DNNEngineManager::GetInstance();
    std::map<std::string, std::string> options;
    instance.engines_map_["engine"] = std::make_shared<DNNEngine>();
    instance.engines_map_["null"] = nullptr;
    EXPECT_NE(instance.Initialize(options), SUCCESS);
    instance.init_flag_ = true;
    EXPECT_EQ(instance.Finalize(), SUCCESS);
    instance.engines_map_.clear();
    instance.engines_map_["sub"] = std::make_shared<SubDNNEngine>();
    EXPECT_EQ(instance.Initialize(options), FAILED);
    instance.engines_map_["sub"]->engine_attribute_.runtime_type = RuntimeType::DEVICE;
    auto p = std::dynamic_pointer_cast<SubDNNEngine>(instance.engines_map_["sub"]);
    p->init_flag_ = true;
    EXPECT_EQ(instance.Initialize(options), GE_ENG_MEMTYPE_ERROR);
}

TEST_F(UtestDnnengineManager, GetEngine) {
    auto &instance = DNNEngineManager::GetInstance();
    EXPECT_EQ(instance.GetEngine("name"), nullptr);
    instance.engines_map_["engine"] = std::make_shared<DNNEngine>();
    EXPECT_NE(instance.GetEngine("engine"), nullptr);
}

TEST_F(UtestDnnengineManager, GetCompositeEngineName) {
    auto &instance = DNNEngineManager::GetInstance();
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
    auto node = UtAddNode(graph, "data1", "DATA", 0, 1);
    EXPECT_EQ(instance.GetCompositeEngineName(node, 20), "");
}

TEST_F(UtestDnnengineManager, GetCompositeEngine) {
    auto &instance = DNNEngineManager::GetInstance();
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
    auto node1 = UtAddNode(graph, "data1", "DATA", 1, 1);
    auto node2 = UtAddNode(graph, "data2", "DATA", 1, 1);
    auto node3 = UtAddNode(graph, "data3", "DATA", 1, 1);
    auto node4 = UtAddNode(graph, "data4", "DATA", 1, 1);
    EXPECT_EQ(instance.GetCompositeEngine(node1), "");
    EXPECT_EQ(instance.GetCompositeEngine(node1, 1), "");
    EXPECT_EQ(instance.GetCompositeEngine(graph, 1), "");
    node1->GetOpDesc()->SetAttr("_no_task", AnyValue::CreateFrom<int>(1));
    EXPECT_EQ(instance.GetCompositeEngine(node1), "");
    EXPECT_EQ(node1->GetInDataAnchor(0)->LinkFrom(node2->GetOutDataAnchor(0)), GRAPH_SUCCESS);
    EXPECT_EQ(instance.GetCompositeEngine(node1), "");
    auto od = node3->GetOpDesc();
    od.reset();
    EXPECT_EQ(instance.GetCompositeEngine(node3, 1), "");
    auto st = std::set<std::string>();
    st.insert("atomic_engine_name");
    OpsKernelManager::GetInstance().composite_engines_["com"] = st;
    EXPECT_EQ(instance.GetCompositeEngine(node4, 1), "");
}

TEST_F(UtestDnnengineManager, GetHostCpuEngineName) {
    auto &instance = DNNEngineManager::GetInstance();
    std::vector<OpInfo> op_infos;
    op_infos.push_back(OpInfo());
    OpDescPtr odp = std::make_shared<OpDesc>("name", "type");
    OpInfo matched_op_info;
    EXPECT_EQ(instance.GetHostCpuEngineName(op_infos, odp, matched_op_info), "");
    EXPECT_EQ(matched_op_info.opKernelLib, "");
    auto oi = OpInfo();
    oi.engine = "DNN_VM_HOST_CPU";
    oi.opKernelLib = "DNN_VM_HOST_CPU_OP_STORE";
    op_infos.push_back(oi);
    EXPECT_EQ(instance.GetHostCpuEngineName(op_infos, odp, matched_op_info), "DNN_VM_HOST_CPU");
    EXPECT_EQ(matched_op_info.opKernelLib, "DNN_VM_HOST_CPU_OP_STORE");
}

TEST_F(UtestDnnengineManager, ReadJsonFile) {
    auto &instance = DNNEngineManager::GetInstance();
    EXPECT_EQ(instance.ReadJsonFile("", nullptr), FAILED);
    EXPECT_EQ(instance.ReadJsonFile("/tmp", nullptr), FAILED);
}

TEST_F(UtestDnnengineManager, ParserJsonFile) {
    auto &instance = DNNEngineManager::GetInstance();
    //std::string json_file_path = "plugin/nnengine/ge_config/engine_conf.json";
    std::string path = GetModelPath();
    path.append("plugin/nnengine/ge_config/");
    std::string path_copy = path;
    path_copy.append("engine_conf_backup.json");
    string cmd = "mkdir -p " + path;
    system(cmd.data());
    path.append("engine_conf.json");
    // back up origin engine_conf.json
    std::string backup_cmd = "cp " + path + " "+ path_copy;
    system(backup_cmd.data());

    std::ofstream ofs(path.c_str(), std::ios::out);
    ofs << "{}";
    ofs.close();
    EXPECT_EQ(instance.ParserJsonFile(), FAILED);
    std::ofstream ofs1(path.c_str(), std::ios::out);
    ofs1 << "{\"schedule_units\":1";
    ofs1.close();
    EXPECT_EQ(instance.ParserJsonFile(), FAILED);
    std::ofstream ofs2(path.c_str(), std::ios::out);
    ofs2 << "{\"schedule_units\":[{\"cal_engines22\":}]";
    ofs2.close();
    EXPECT_EQ(instance.ParserJsonFile(), FAILED);
    std::ofstream ofs3(path.c_str(), std::ios::out);
    ofs3 << "{\"schedule_units\":[{\"cal_engines\":\"cal\"}]";
    ofs3.close();
    EXPECT_EQ(instance.ParserJsonFile(), FAILED);
    std::ofstream ofs4(path.c_str(), std::ios::out);
    ofs4 << "{\"schedule_units\":[{\"cal_engines\":\"cal\", \"id\":\"\"}]";
    ofs4.close();
    EXPECT_EQ(instance.ParserJsonFile(), FAILED);

    // recover origin engine_conf.json
    std::string recover_cmd1 = "rm -rf " + path + ";mv " + path_copy + " " +path;
    system(recover_cmd1.data());
}

TEST_F(UtestDnnengineManager, CheckJsonFile) {
    auto &instance = DNNEngineManager::GetInstance();
    instance.engines_map_["engine1"] = std::make_shared<DNNEngine>();
    instance.engines_map_["engine2"] = std::make_shared<DNNEngine>();
    instance.schedulers_["sch"] = SchedulerConf();
    instance.schedulers_["sch"].cal_engines["engine1"] = std::make_shared<EngineConf>();
    instance.schedulers_["sch"].cal_engines["engine2"] = std::make_shared<EngineConf>();
    EXPECT_EQ(instance.CheckJsonFile(), SUCCESS);
}

TEST_F(UtestDnnengineManager, InitAtomicCompositeMapping) {
    auto &instance = DNNEngineManager::GetInstance();
    instance.InitAtomicCompositeMapping();
    auto st = std::set<std::string>();
    st.insert("atomic_engine_name");
    OpsKernelManager::GetInstance().composite_engines_["com"] = st;
    EXPECT_NO_THROW(instance.InitAtomicCompositeMapping());
}

TEST_F(UtestDnnengineManager, IsEngineRegistered) {
    auto &instance = DNNEngineManager::GetInstance();
    instance.engines_map_["engine"] = std::make_shared<DNNEngine>();
    EXPECT_EQ(instance.IsEngineRegistered("engine"), true);
}

TEST_F(UtestDnnengineManager, GetOpInfos) {
    auto &instance = DNNEngineManager::GetInstance();
    std::vector<OpInfo> op_infos;
    auto op_desc = std::make_shared<OpDesc>("name", "type");
    AttrUtils::SetStr(op_desc, "_specified_engine_name", "engine_name");
    AttrUtils::SetStr(op_desc, "_specified_kernel_lib_name", "kernel_name");
    bool is_op_specified_engine = true;
    instance.GetOpInfos(op_infos, op_desc, is_op_specified_engine);
    EXPECT_NE(op_infos.size(), 0);
}

TEST_F(UtestDnnengineManager, GetExcludeEngines) {
  auto &instance = DNNEngineManager::GetInstance();
  std::map<std::string, std::string> options =
    {{EXCLUDE_ENGINES, " AiCore |AiVector|   Dsa| FakeEngine|FftsPlus"},
     {ge::CORE_TYPE, "FakeEngine"}};
  GetThreadLocalContext().SetGraphOption(options);

  std::set<std::string> exclude_engines;
  instance.GetExcludeEngines(exclude_engines);
  EXPECT_EQ(exclude_engines.size(), 4);
  bool flag = exclude_engines.find("AIcoreEngine") == exclude_engines.end();
  EXPECT_FALSE(flag);
  flag = exclude_engines.find("VectorEngine") == exclude_engines.end();
  EXPECT_FALSE(flag);
  flag = exclude_engines.find("DSAEngine") == exclude_engines.end();
  EXPECT_FALSE(flag);
  flag = exclude_engines.find("ffts_plus") == exclude_engines.end();
  EXPECT_FALSE(flag);

  GetThreadLocalContext().SetGraphOption({});
  exclude_engines.clear();
  instance.GetExcludeEngines(exclude_engines);
  EXPECT_EQ(exclude_engines.size(), 1);

}

TEST_F(UtestDnnengineManager, GetDNNEngineName) {
    auto &instance = DNNEngineManager::GetInstance();
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
    auto node = UtAddNode(graph, "data1", "DATA", 0, 1);
    AttrUtils::SetStr(node->GetOpDesc(), "_specified_engine_name", "engine_name");
    AttrUtils::SetStr(node->GetOpDesc(), "_specified_kernel_lib_name", "kernel_name");
    EXPECT_EQ(instance.GetDNNEngineName(node), "");
    auto &okm = OpsKernelManager::GetInstance();
    std::map<std::string, std::string> options;
    okm.ops_kernel_store_["kernel_name"] = std::make_shared<SubOpsKernelInfoStore2>();
    EXPECT_EQ(instance.GetDNNEngineName(node), "engine_name");
    okm.ops_kernel_store_.clear();
    auto p = std::make_shared<SubOpsKernelInfoStore2>();
    p->check_flag_ = false;
    okm.ops_kernel_store_["kernel_name"] = p;
    EXPECT_EQ(instance.GetDNNEngineName(node), "");
}

TEST_F(UtestDnnengineManager, GetDNNEngineName_not_support_dynamic_shape) {
  auto &instance = DNNEngineManager::GetInstance();
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "not_support_dynamic_shape", "DATA", 0, 1);
  AttrUtils::SetStr(node->GetOpDesc(), "_specified_engine_name", "engine_name");
  AttrUtils::SetStr(node->GetOpDesc(), "_specified_kernel_lib_name", "kernel_name");
  EXPECT_EQ(instance.GetDNNEngineName(node), "");
  auto &okm = OpsKernelManager::GetInstance();
  std::map<std::string, std::string> options;
  okm.ops_kernel_store_["kernel_name"] = std::make_shared<SubOpsKernelInfoStore2>();
  EXPECT_EQ(instance.GetDNNEngineName(node), "");
}
} // namespace ge

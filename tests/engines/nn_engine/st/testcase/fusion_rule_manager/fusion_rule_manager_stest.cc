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
#include <nlohmann/json.hpp>
#include "fe_llt_utils.h"

#define protected public
#define private public
#include "common/constants_define.h"
#include "common/platform_utils.h"
#include "ops_store/ops_kernel_manager.h"
#include "fusion_rule_manager/fusion_rule_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "fusion_rule_manager/fusion_rule_parser/fusion_rule_parser_utils.h"
#include "common/fe_log.h"
#include "common/configuration.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#undef private
#undef protected

using namespace std;
using namespace fe;
using namespace ge;

class fusion_rule_manager_stest : public testing::Test
{
protected:
    void SetUp()
    {
        ops_kernel_info_store_ptr_ = std::make_shared<FEOpsKernelInfoStore>();
        std::map<std::string, std::string> options;
        ops_kernel_info_store_ptr_ = make_shared<fe::FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
        FEOpsStoreInfo tbe_custom {
                6,
                "tbe-custom",
                EN_IMPL_HW_TBE,
                GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_rule_manager/tbe_builtin_info",
                ""};
        vector<FEOpsStoreInfo> store_info;
        store_info.emplace_back(tbe_custom);
        Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
        OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

        OpsKernelManager::Instance(AI_CORE_NAME).Initialize();
        ops_kernel_info_store_ptr_->Initialize(options);
        FusionRuleParserUtils::Instance()->SetEngineName(AI_CORE_NAME);
        PlatformUtils::Instance().soc_version_ = "Ascend910B1";
        PlatformUtils::Instance().short_soc_version_ = "Ascend910B";
    }

    void TearDown()
    {
    }
    FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr_;
};

TEST_F(fusion_rule_manager_stest, get_fusion_rules_by_rule_type_failed_not_init)
{
    auto frm = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    RuleType rule_type;
    rule_type = RuleType::CUSTOM_GRAPH_RULE;
    vector<FusionRulePatternPtr> out_rule_vector = {};
    Status ret = frm->GetFusionRulesByRuleType(rule_type, out_rule_vector);
    EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(fusion_rule_manager_stest, fusion_rule_manager_01)
{
    std::cout << "=======================================================================" << std::endl;
    string file_path = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_rule_manager/test.json";
    auto frm = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    vector<FusionRulePatternPtr> out_rule_vector = {};
    Status ret = frm->LoadFusionRule(file_path, out_rule_vector);
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(fusion_rule_manager_stest, fusion_rule_manager_02)
{
    std::cout << "=======================================================================" << std::endl;
    string file_path = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_rule_manager/non_exist.json";
    auto frm = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    vector<FusionRulePatternPtr> out_rule_vector = {};
    Status ret = frm->LoadFusionRule(file_path, out_rule_vector);
    EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(fusion_rule_manager_stest, fusion_rule_manager_03)
{
    std::cout << "=======================================================================" << std::endl;
    string file_path = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_rule_manager/testnull.json";
    auto frm = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    vector<FusionRulePatternPtr> out_rule_vector = {};
    Status ret = frm->LoadFusionRule(file_path, out_rule_vector);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}

TEST_F(fusion_rule_manager_stest, fusion_rule_manager_04)
{
    std::cout << "=======================================================================" << std::endl;
    string file_path = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_rule_manager/testwrong.json";
    auto frm = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    vector<FusionRulePatternPtr> out_rule_vector = {};
    Status ret = frm->LoadFusionRule(file_path, out_rule_vector);
    EXPECT_EQ(fe::ILLEGAL_RULE, ret);
}

TEST_F(fusion_rule_manager_stest, attr_assignment_expression_succ_01)
{
    std::cout << "=======================================================================" << std::endl;
    nlohmann::json json_object = { {"attr", "node1.attr1"}, {"value",{"node1.attr1", 18.2}}, {"expr", "="}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseJson(json_object);
    EXPECT_EQ(fe::SUCCESS, ret);
}
TEST_F(fusion_rule_manager_stest, attr_assignment_expression_fail_01)
{
    std::cout << "=======================================================================" << std::endl;
    nlohmann::json json_object = { {"no_attr", "node1.attr1"}, {"value",{"node1.attr1", 18.2}}, {"expr", "="}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseJson(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_stest, attr_assignment_expression_fail_02)
{
    std::cout << "=======================================================================" << std::endl;
    nlohmann::json json_object = { {"attr", {"attr1"}}, {"value",{"node1.attr1", 18.2}}, {"expr", "="}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseJson(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_stest, attr_assignment_expression_fail_03)
{
    std::cout << "=======================================================================" << std::endl;
    nlohmann::json json_object = { {"attr", ""}, {"value",{"node1.attr1", 18.2}}, {"expr", "="}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseJson(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_stest, attr_assignment_expression_fail_04)
{
    std::cout << "=======================================================================" << std::endl;
    nlohmann::json json_object = { {"attr", "attr1"}, {"value",{"node1.attr1", 18.2}}, {"expr", "="}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseJson(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_stest, attr_assignment_expression_fail_05)
{
    std::cout << "=======================================================================" << std::endl;
    nlohmann::json json_object = { {"attr", "node1.attr1"}, {"value",{"node1.attr1", 18.2}}, {"expr", {"="}}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseJson(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_stest, attr_assignment_expression_fail_06)
{
    std::cout << "=======================================================================" << std::endl;
    nlohmann::json json_object = { {"attr", "node1.attr1"}, {"value",{"node1.attr1", 18.2}}, {"expr", "-"}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseJson(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_stest, ParseToJsonAttr_succ_01)
{
    std::cout << "=======================================================================" << std::endl;
    nlohmann::json json_object = { {"attr", "node1.attr1"}, {"value",{"node1.attr1", 18.2}}, {"expr", "="}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseToJsonAttr(json_object);
    EXPECT_EQ(fe::SUCCESS, ret);
}
TEST_F(fusion_rule_manager_stest, ParseToJsonAttr_fail_01)
{
    std::cout << "=======================================================================" << std::endl;
    nlohmann::json json_object = { {"no_attr", "node1.attr1"}, {"value",{"node1.attr1", 18.2}}, {"expr", "="}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseToJsonAttr(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_stest, ParseToJsonAttr_fail_02)
{
    std::cout << "=======================================================================" << std::endl;
    nlohmann::json json_object = { {"attr", {"attr1"}}, {"value",{"node1.attr1", 18.2}}, {"expr", "="}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseToJsonAttr(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_stest, ParseToJsonAttr_fail_03)
{
    std::cout << "=======================================================================" << std::endl;
    nlohmann::json json_object = { {"attr", ""}, {"value",{"node1.attr1", 18.2}}, {"expr", "="}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseToJsonAttr(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_stest, ParseToJsonAttr_fail_04)
{
    std::cout << "=======================================================================" << std::endl;
    nlohmann::json json_object = { {"attr", "attr1"}, {"value",{"node1.attr1", 18.2}}, {"expr", "="}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseToJsonAttr(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_stest, ParseToJsonAttr_fail_05)
{
    std::cout << "=======================================================================" << std::endl;
    nlohmann::json json_object = { {"attr", "node1.attr1"}, {"value",{"node1.attr1", 18.2}}, {"expr", {"="}}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseToJsonAttr(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_stest, ParseToJsonAttr_fail_06)
{
    std::cout << "=======================================================================" << std::endl;
    nlohmann::json json_object = { {"attr", "node1.attr1"}, {"value",{"node1.attr1", 18.2}}, {"expr", "-"}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseToJsonAttr(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_stest, ParseToJsonEdge_succ_01)
{
    std::cout << "=======================================================================" << std::endl;
    nlohmann::json json_object = {{"src", "input1"}, {"dst", "node1"}};
    auto fusion_rule_edge = std::make_shared<FusionRuleJsonEdge>();
    Status ret = fusion_rule_edge->ParseToJsonEdge(json_object);
    EXPECT_EQ(fe::SUCCESS, ret);
}
TEST_F(fusion_rule_manager_stest, ParseToJsonNode_succ_01)
{
    std::cout << "=======================================================================" << std::endl;
    nlohmann::json json_object = {{"name", "Conv"}, {"type", {"min"}}};
    FusionRuleParserUtils::Instance()->SetEngineName(AI_CORE_NAME);
    auto fusion_rule_node = std::make_shared<FusionRuleJsonNode>();
    Status ret = fusion_rule_node->ParseToJsonNode(json_object);
    EXPECT_EQ(fe::SUCCESS, ret);
}
TEST_F(fusion_rule_manager_stest, ParseToJsonOuter_succ_01)
{
    std::cout << "=======================================================================" << std::endl;
    nlohmann::json json_object = {{"name", "out"}, {"src", "node1"}};
    auto fusion_rule_out = std::make_shared<FusionRuleJsonOuter>();
    Status ret = fusion_rule_out->ParseToJsonOuter(json_object);
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(fusion_rule_manager_stest, initialize_success)
{
    auto frm = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    frm->init_flag_ = true;
    Status ret = frm->Initialize(fe::AI_CORE_NAME);
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(fusion_rule_manager_stest, finalize_success_not_init)
{
    auto frm = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    Status ret = frm->Finalize();
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(fusion_rule_manager_stest, finalize_success_init)
{
    auto frm = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    frm->init_flag_ = true;
    Status ret = frm->Finalize();
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(fusion_rule_manager_stest, get_fusion_rules_by_rule_type_failed_not_init1)
{
    auto frm = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    RuleType rule_type;
    rule_type = RuleType::CUSTOM_GRAPH_RULE;
    vector<FusionRulePatternPtr> out_rule_vector = {};
    Status ret = frm->GetFusionRulesByRuleType(rule_type, out_rule_vector);
    EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(fusion_rule_manager_stest, get_fusion_rules_by_rule_type_failed_not_init2)
{
    auto frm = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    frm->init_flag_ = false;
    RuleType rule_type;
    std::string rule_name = "rule_name";
    Status ret = frm->RunGraphFusionRuleByType(*graph, rule_type, rule_name);
    EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(fusion_rule_manager_stest, get_str_from_attr_alue_fail1)
{
    ge::GeAttrValue attr_value_float = GeAttrValue::CreateFrom<int64_t>(1);
    auto attr_value = fe::GetStrOfFloat(attr_value_float);
    EXPECT_EQ(attr_value, "");
}
TEST_F(fusion_rule_manager_stest, get_str_from_attr_alue_fail2)
{
    ge::GeAttrValue attr_value_float = GeAttrValue::CreateFrom<float>(1.1);
    auto attr_value = fe::GetStrOfInt(attr_value_float);
    EXPECT_EQ(attr_value, "");
}
TEST_F(fusion_rule_manager_stest, get_str_from_attr_alue_fail3)
{
    ge::GeAttrValue attr_value_float = GeAttrValue::CreateFrom<int64_t>(1);
    auto attr_value = fe::GetStrOfBool(attr_value_float);
    EXPECT_EQ(attr_value, "");
}
TEST_F(fusion_rule_manager_stest, get_str_from_attr_alue_fail4)
{
    ge::GeAttrValue attr_value_float = GeAttrValue::CreateFrom<float>(1.1);
    auto attr_value = fe::GetStrOfListInt(attr_value_float);
    EXPECT_EQ(attr_value, "");
}
TEST_F(fusion_rule_manager_stest, get_str_from_attr_alue_fail5)
{
    ge::GeAttrValue attr_value_float = GeAttrValue::CreateFrom<int64_t>(1);
    auto attr_value = fe::GetStrOfListFloat(attr_value_float);
    EXPECT_EQ(attr_value, "");
}
TEST_F(fusion_rule_manager_stest, get_str_from_attr_alue_fail6)
{
    ge::GeAttrValue attr_value_float = GeAttrValue::CreateFrom<int64_t>(1);
    auto attr_value = fe::GetStrOfListBool(attr_value_float);
    EXPECT_EQ(attr_value, "");
}

TEST_F(fusion_rule_manager_stest, test_init_graph_rules_aicore)
{
    std::string engine_name = fe::AI_CORE_NAME;
    const std::string current_dir = "./";
    const std::string aicore_json_path = current_dir + "plugin/opskernel/fusion_rules/ai_core/";
    std::cout << "aicore_json_path is " << aicore_json_path << std::endl;
    CreateDir(aicore_json_path);
    const std::string fileName = aicore_json_path + "built_in_graph_rules.json";
    std::string ori_json_path = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_rule_manager/test.json";
    std::ifstream ifs(ori_json_path);
    if (!ifs.is_open()) {
        printf("open json[%s] failed, %s", ori_json_path.c_str(), strerror(errno));
    }
    ASSERT_TRUE(ifs.is_open());
    nlohmann::json ori_json_value;
    ifs >> ori_json_value;
    CreateFileAndFillContent(fileName, ori_json_value, true);
    Configuration::Instance(engine_name).lib_path_ = current_dir;
    std::cout << Configuration::Instance(engine_name).content_map_[CONFIG_KEY_CUSTOM_PASS_FILE] <<std::endl;
    std::cout << Configuration::Instance(engine_name).content_map_[CONFIG_KEY_BUILTIN_PASS_FILE] <<std::endl;
    std::cout << Configuration::Instance(engine_name).content_map_[CONFIG_KEY_COMPILER_PASS_FILE] <<std::endl;
    std::cout << Configuration::Instance(engine_name).content_map_[CONFIG_KEY_GRAPH_FILE] <<std::endl;
    std::cout << Configuration::Instance(engine_name).content_map_[CONFIG_KEY_COMPILER_GRAPH_FILE] <<std::endl;

    Configuration::Instance(engine_name).content_map_[CONFIG_KEY_GRAPH_FILE] = "lib64/plugin/opskernel/fusion_rules/ai_core/built_in_graph_rules.json";
    FEOpsKernelInfoStorePtr aicore_ops_kernel_info_store_ptr_  = make_shared<fe::FEOpsKernelInfoStore>(engine_name);
    auto frm = std::make_shared<FusionRuleManager>(aicore_ops_kernel_info_store_ptr_);
    Status ret = frm->InitGraphRules(engine_name);
    EXPECT_EQ(ret, fe::SUCCESS);
    system(("rm -rf " + current_dir + "plugin").c_str());
}

TEST_F(fusion_rule_manager_stest, test_init_graph_rules_vectorcore) {
    std::string engine_name = fe::VECTOR_CORE_NAME;
    const std::string current_dir = "./";
    const std::string vectorcore_json_path = current_dir + "plugin/opskernel/fusion_rules/vector_core/";
    std::cout << "vectorcore_json_path is " << vectorcore_json_path << std::endl;
    CreateDir(vectorcore_json_path);
    const std::string vector_file_name = vectorcore_json_path + "built_in_graph_rules.json";

    nlohmann::json ori_vector_json_value;
    std::vector<int> null_vector;
    ori_vector_json_value["Rules"] = null_vector;
    CreateFileAndFillContent(vector_file_name, ori_vector_json_value, true);
    Configuration::Instance(engine_name).lib_path_ = current_dir;
    FEOpsKernelInfoStorePtr vectorcore_ops_kernel_info_store_ptr_;
    vectorcore_ops_kernel_info_store_ptr_ = make_shared<fe::FEOpsKernelInfoStore>(engine_name);
    auto frm = std::make_shared<FusionRuleManager>(vectorcore_ops_kernel_info_store_ptr_);
    Configuration::Instance(engine_name).content_map_["fusionrulemgr.vectorcore.compiler.graphfilepath"] =
                   "plugin/opskernel/fusion_rules/vector_core/built_in_graph_rules.json";
    Status ret = frm->InitGraphRules(engine_name);
    EXPECT_EQ(ret, fe::SUCCESS);
    system(("rm -rf " + current_dir + "plugin").c_str());
}
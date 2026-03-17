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

class fusion_rule_manager_uttest : public testing::Test
{
protected:
    void SetUp()
    {
        std::map<std::string, std::string> options;
        ops_kernel_info_store_ptr_ = make_shared<fe::FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
        FEOpsStoreInfo tbe_custom {
                6,
                "tbe-custom",
                EN_IMPL_HW_TBE,
                GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_rule_manager/tbe_builtin_info",
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
FusionRuleAnchorPtr CreateAnchor(int anchor_idx,
                                    const string &anchor_name,
                                    FusionRuleNodePtr owner_node,
                                    const vector<FusionRuleAnchorPtr> &peer_anchors)
{
    FusionRuleAnchorPtr anchor = make_shared<fe::FusionRuleAnchor>();
    anchor->anchor_idx_ = anchor_idx;
    anchor->anchor_name_ = anchor_name;
    anchor->owner_node_ = owner_node;
    for (size_t i = 0; i < peer_anchors.size(); ++i) {
        anchor->peer_anchors_.emplace_back(peer_anchors[i]);
    }
    for (size_t i = 0; i < anchor->peer_anchors_.size(); ++i) {
        auto peer_anchor = anchor->peer_anchors_[i].lock();
        peer_anchor->peer_anchors_.emplace_back(anchor);
    }
    return anchor;
}
FusionRuleNodePtr CreateFusionRuleNode(const string &node_name,
                                        const vector<string> &node_types,
                                        vector<int> inputs_anchor_indxs,
                                        vector<int> output_anchor_indexs,
                                        const map<string, FusionRuleAttrValuePtr> &attributes)
{
    FusionRuleNodePtr node = make_shared<fe::FusionRuleNode>();
    node->node_name_ = node_name;
    node->node_type_ = node_types;
    for (size_t i = 0; i < inputs_anchor_indxs.size(); ++i) {
        int index = inputs_anchor_indxs[i];
        string anchor_name = node_name + "_input_" + to_string(index);
        auto input_anchor = CreateAnchor(index, anchor_name, node, {});
        node->input_data_anchors_.push_back(input_anchor);
    }
    for (size_t i = 0; i < output_anchor_indexs.size(); ++i) {
        int index = output_anchor_indexs[i];
        string anchor_name = node_name + "_output_" + to_string(index);
        auto output_anchor = CreateAnchor(index, anchor_name, node, {});
        node->output_data_anchors_.push_back(output_anchor);
    }
    node->attributes_ = attributes;
    return node;
}

TEST_F(fusion_rule_manager_uttest, get_fusion_rules_by_rule_type_failed_not_init)
{
    auto frm = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    RuleType rule_type;
    rule_type = RuleType::CUSTOM_GRAPH_RULE;
    vector<FusionRulePatternPtr> out_rule_vector = {};
    Status ret = frm->GetFusionRulesByRuleType(rule_type, out_rule_vector);
    EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(fusion_rule_manager_uttest, fusion_rule_manager_01)
{
    string file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_rule_manager/test.json";
    auto frm = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    vector<FusionRulePatternPtr> out_rule_vector = {};
    Status ret = frm->LoadFusionRule(file_path, out_rule_vector);
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(fusion_rule_manager_uttest, fusion_rule_manager_02)
{
  std::cout << "=======================================================================" << std::endl;
  string file_path = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_rule_manager/non_exist.json";
  auto frm = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  vector<FusionRulePatternPtr> out_rule_vector = {};
  Status ret = frm->LoadFusionRule(file_path, out_rule_vector);
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(fusion_rule_manager_uttest, fusion_rule_manager_03)
{
  std::cout << "=======================================================================" << std::endl;
  string file_path = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_rule_manager/testnull.json";
  auto frm = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  vector<FusionRulePatternPtr> out_rule_vector = {};
  Status ret = frm->LoadFusionRule(file_path, out_rule_vector);
  EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}

TEST_F(fusion_rule_manager_uttest, fusion_rule_manager_04)
{
  std::cout << "=======================================================================" << std::endl;
  string file_path = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_rule_manager/testwrong.json";
  auto frm = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  vector<FusionRulePatternPtr> out_rule_vector = {};
  Status ret = frm->LoadFusionRule(file_path, out_rule_vector);
  EXPECT_EQ(fe::ILLEGAL_RULE, ret);
}

TEST_F(fusion_rule_manager_uttest, attr_assignment_expression_succ_01)
{
    nlohmann::json json_object = { {"attr", "node1.attr1"}, {"value",{"node1.attr1", 18.2}}, {"expr", "="}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseJson(json_object);
    EXPECT_EQ(fe::SUCCESS, ret);
}
TEST_F(fusion_rule_manager_uttest, attr_assignment_expression_fail_01)
{
    nlohmann::json json_object = { {"no_attr", "node1.attr1"}, {"value",{"node1.attr1", 18.2}}, {"expr", "="}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseJson(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_uttest, attr_assignment_expression_fail_02)
{
    nlohmann::json json_object = { {"attr", {"attr1"}}, {"value",{"node1.attr1", 18.2}}, {"expr", "="}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseJson(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_uttest, attr_assignment_expression_fail_03)
{
    nlohmann::json json_object = { {"attr", ""}, {"value",{"node1.attr1", 18.2}}, {"expr", "="}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseJson(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_uttest, attr_assignment_expression_fail_04)
{
    nlohmann::json json_object = { {"attr", "attr1"}, {"value",{"node1.attr1", 18.2}}, {"expr", "="}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseJson(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_uttest, attr_assignment_expression_fail_05)
{
    nlohmann::json json_object = { {"attr", "node1.attr1"}, {"value",{"node1.attr1", 18.2}}, {"expr", {"="}}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseJson(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_uttest, attr_assignment_expression_fail_06)
{
    nlohmann::json json_object = { {"attr", "node1.attr1"}, {"value",{"node1.attr1", 18.2}}, {"expr", "-"}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseJson(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_uttest, attr_assignment_expression_fail_07)
{
    nlohmann::json json_object = "json_object";
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseJson(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_uttest, ParseToJsonAttr_succ_01)
{
    nlohmann::json json_object = { {"attr", "node1.attr1"}, {"value",{"node1.attr1", 18.2}}, {"expr", "="}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseToJsonAttr(json_object);
    EXPECT_EQ(fe::SUCCESS, ret);
}
TEST_F(fusion_rule_manager_uttest, ParseToJsonAttr_fail_01)
{
    nlohmann::json json_object = { {"no_attr", "node1.attr1"}, {"value",{"node1.attr1", 18.2}}, {"expr", "="}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseToJsonAttr(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_uttest, ParseToJsonAttr_fail_02)
{
    nlohmann::json json_object = { {"attr", {"attr1"}}, {"value",{"node1.attr1", 18.2}}, {"expr", "="}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseToJsonAttr(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_uttest, ParseToJsonAttr_fail_03)
{
    nlohmann::json json_object = { {"attr", ""}, {"value",{"node1.attr1", 18.2}}, {"expr", "="}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseToJsonAttr(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_uttest, ParseToJsonAttr_fail_04)
{
    nlohmann::json json_object = { {"attr", "attr1"}, {"value",{"node1.attr1", 18.2}}, {"expr", "="}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseToJsonAttr(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_uttest, ParseToJsonAttr_fail_05)
{
    nlohmann::json json_object = { {"attr", "node1.attr1"}, {"value",{"node1.attr1", 18.2}}, {"expr", {"="}}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseToJsonAttr(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_uttest, ParseToJsonAttr_fail_06)
{
    nlohmann::json json_object = { {"attr", "node1.attr1"}, {"value",{"node1.attr1", 18.2}}, {"expr", "-"}};
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    Status ret = attr_assign->ParseToJsonAttr(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_uttest, ParseToJsonAttr_fail_07)
{
    nlohmann::json json_object = { 1, 2, 3, 4, 5 };
    auto attr_assign = std::make_shared<AttrAssignmentExpression>();
    vector<string> value;
    FusionRuleAttrValuePtr attr_value = make_shared<FusionRuleAttrValue>();
    Status ret = attr_assign->GetStrAndConvert(json_object, value, attr_value);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_uttest, ParseToJsonEdge_succ_01)
{
    nlohmann::json json_object = {{"src", "input1"}, {"dst", "node1"}};
    auto fusion_rule_edge = std::make_shared<FusionRuleJsonEdge>();
    Status ret = fusion_rule_edge->ParseToJsonEdge(json_object);
    EXPECT_EQ(fe::SUCCESS, ret);
}
TEST_F(fusion_rule_manager_uttest, ParseToJsonNode_succ_01)
{
    nlohmann::json json_object = {{"name", "Conv"}, {"type", {"min"}}};
    FusionRuleParserUtils::Instance()->SetEngineName(AI_CORE_NAME);
    auto fusion_rule_node = std::make_shared<FusionRuleJsonNode>();
    Status ret = fusion_rule_node->ParseToJsonNode(json_object);
    EXPECT_EQ(fe::SUCCESS, ret);
}
TEST_F(fusion_rule_manager_uttest, ParseToJsonNode_failed1)
{
    FusionRuleParserUtils::Instance()->SetEngineName(AI_CORE_NAME);
    auto fusion_rule_node = std::make_shared<FusionRuleJsonNode>();
    Status ret = fusion_rule_node->ParseToJsonNode("");
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_uttest, ParseToJsonNode_failed2)
{
    nlohmann::json json_object = {{"name", 1}, {"type", {"min"}}};
    FusionRuleParserUtils::Instance()->SetEngineName(AI_CORE_NAME);
    auto fusion_rule_node = std::make_shared<FusionRuleJsonNode>();
    Status ret = fusion_rule_node->ParseToJsonNode(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_uttest, ParseToJsonNode_failed3)
{
    nlohmann::json json_object = {{"name", "Conv"}, {"type", {""}}};
    FusionRuleParserUtils::Instance()->SetEngineName(AI_CORE_NAME);
    auto fusion_rule_node = std::make_shared<FusionRuleJsonNode>();
    Status ret = fusion_rule_node->ParseToJsonNode(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_uttest, ParseToJsonNode_failed4)
{
    nlohmann::json json_object = {{"name", "Conv"}, {"type", {}}};
    FusionRuleParserUtils::Instance()->SetEngineName(AI_CORE_NAME);
    auto fusion_rule_node = std::make_shared<FusionRuleJsonNode>();
    Status ret = fusion_rule_node->ParseToJsonNode(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_uttest, ParseToJsonNode_failed5)
{
    nlohmann::json json_object = {{"name", "Conv"}, {"type", {"min","max"}}};
    FusionRuleParserUtils::Instance()->SetEngineName(AI_CORE_NAME);
    auto fusion_rule_node = std::make_shared<FusionRuleJsonNode>();
    Status ret = fusion_rule_node->ParseToJsonNode(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_uttest, ParseToJsonNode_failed6)
{
    nlohmann::json json_object = {{"Not_support_key", "Conv"}, {"type", {"min","max"}}};
    FusionRuleParserUtils::Instance()->SetEngineName(AI_CORE_NAME);
    auto fusion_rule_node = std::make_shared<FusionRuleJsonNode>();
    Status ret = fusion_rule_node->ParseToJsonNode(json_object);
    EXPECT_EQ(fe::ILLEGAL_JSON, ret);
}
TEST_F(fusion_rule_manager_uttest, ParseToJsonOuter_succ_01)
{
    nlohmann::json json_object = {{"name", "out"}, {"src", "node1"}};
    auto fusion_rule_out = std::make_shared<FusionRuleJsonOuter>();
    Status ret = fusion_rule_out->ParseToJsonOuter(json_object);
    EXPECT_EQ(fe::SUCCESS, ret);
}
TEST_F(fusion_rule_manager_uttest, convert_to_attr_value_success)
{
    AttrAssignmentExpression test_object;
    std::map<string, std::vector<string>> node_map;
    FusionRuleAttr tmp_attr;
    tmp_attr.attr_name = "keep_dims";
    tmp_attr.node_name = "node";
    std::vector<string> vec{"formatAgnosticOp", "conv"};
    node_map.insert(pair<string, std::vector<string>>("node", vec));
    FusionRuleParserUtils::Instance()->SetEngineName(AI_CORE_NAME);

    test_object.attr_ = tmp_attr;
    test_object.value_ = make_shared<FusionRuleAttrValue>();
    test_object.tmp_value_ = {"1.1", "1.2"};
    test_object.parse_to_attr_success_ = true;
    Status ret = test_object.ConvertToAttrValue(test_object.tmp_value_, ge::GeAttrValue::VT_LIST_FLOAT, test_object.value_);
    EXPECT_EQ(ret, fe::SUCCESS);

}
TEST_F(fusion_rule_manager_uttest, convert_to_attr_value_failed)
{
    AttrAssignmentExpression test_object;
    std::map<string, std::vector<string>> node_map;
    FusionRuleAttr tmp_attr;
    tmp_attr.attr_name = "keep_dims";
    tmp_attr.node_name = "node";
    std::vector<string> vec{"formatAgnosticOp", "conv"};
    node_map.insert(pair<string, std::vector<string>>("node", vec));
    FusionRuleParserUtils::Instance()->SetEngineName(AI_CORE_NAME);

    test_object.attr_ = tmp_attr;
    test_object.value_ = make_shared<FusionRuleAttrValue>();
    test_object.tmp_value_ = {"1.1"};
    test_object.parse_to_attr_success_ = true;
    Status ret = test_object.ConvertToAttrValue(test_object.tmp_value_, ge::GeAttrValue::VT_BYTES, test_object.value_);
    EXPECT_EQ(ret, fe::ILLEGAL_JSON);

}
TEST_F(fusion_rule_manager_uttest, parse_to_json_attr_failed)
{
    AttrAssignmentExpression test_object;
    Status ret = test_object.ParseToJsonAttr("");
    EXPECT_EQ(ret, fe::ILLEGAL_JSON);
}
TEST_F(fusion_rule_manager_uttest, parse_to_attr_value_failed)
{
    AttrAssignmentExpression test_object;
    std::map<string, std::vector<string>> node_map;
    ge::GeAttrValue::ValueType tmp_value_type;
    FusionRuleAttr tmp_attr;
    tmp_attr.attr_name = "attr";
    tmp_attr.node_name = "node";
    std::vector<string> vec{"conv", "StridedRead"};
    node_map.insert(pair<string, std::vector<string>>("node", vec));
    FusionRuleParserUtils::Instance()->SetEngineName(AI_CORE_NAME);

    Status ret = test_object.ParseToAttrValue(node_map);
    EXPECT_EQ(ret, fe::ILLEGAL_JSON);
}
TEST_F(fusion_rule_manager_uttest, parse_to_json_pattern_failed)
{
    FusionRuleJsonPattern test_object;
    Status ret = test_object.ParseToJsonPattern("");
    EXPECT_EQ(ret, fe::ILLEGAL_JSON);
}
TEST_F(fusion_rule_manager_uttest, load_json_failed)
{
    FusionRuleJsonPattern test_object;
    Status ret = test_object.ParseToJsonPattern("");
    EXPECT_EQ(ret, fe::ILLEGAL_JSON);
}
TEST_F(fusion_rule_manager_uttest, parse_to_origin_graph_failed)
{
    FusionRuleJsonPattern test_object;
    FusionRuleJsonGraphPtr origin_graph = make_shared<FusionRuleJsonGraph>();
    Status ret = test_object.ParseToOriginGraph("", origin_graph);
    EXPECT_EQ(ret, fe::ILLEGAL_JSON);
}
TEST_F(fusion_rule_manager_uttest, parse_to_fusion_graph_failed)
{
    FusionRuleJsonPattern test_object;
    FusionRuleJsonGraphPtr fusion_graph = make_shared<FusionRuleJsonGraph>();
    Status ret = test_object.ParseToFusionGraph("", fusion_graph);
    EXPECT_EQ(ret, fe::ILLEGAL_JSON);
}
TEST_F(fusion_rule_manager_uttest, parse_json_failed)
{
    FusionRuleJsonNode test_object;
    Status ret = test_object.ParseJson("");
    EXPECT_EQ(ret, fe::ILLEGAL_JSON);
}
TEST_F(fusion_rule_manager_uttest, parse_json_failed1)
{
    FusionRuleJsonNode test_object;
    nlohmann::json json_object = {{"name", 1}, {"type", {"min"}}};
    Status ret = test_object.ParseJson(json_object);
    EXPECT_EQ(ret, fe::ILLEGAL_JSON);
}
TEST_F(fusion_rule_manager_uttest, parse_json_failed2)
{
    FusionRuleJsonNode test_object;
    nlohmann::json json_object = {{"name", "Conv"}, {"type", 1}};
    Status ret = test_object.ParseJson(json_object);
    EXPECT_EQ(ret, fe::ILLEGAL_JSON);
}
TEST_F(fusion_rule_manager_uttest, parse_json_failed3)
{
    FusionRuleJsonNode test_object;
    nlohmann::json json_object = {{"name", "Conv"}, {"type", {""}}};
    Status ret = test_object.ParseJson(json_object);
    EXPECT_EQ(ret, fe::ILLEGAL_JSON);
}
TEST_F(fusion_rule_manager_uttest, parse_json_failed4)
{
    FusionRuleJsonNode test_object;
    nlohmann::json json_object = {{"name", "Conv"}, {"type", {"min","max"}}};
    Status ret = test_object.ParseJson(json_object);
    EXPECT_EQ(ret, fe::ILLEGAL_JSON);
}
TEST_F(fusion_rule_manager_uttest, parse_json_failed5)
{
    FusionRuleJsonNode test_object;
    nlohmann::json json_object = {{"name", "Conv"}, {"type", {}}};
    Status ret = test_object.ParseJson(json_object);
    EXPECT_EQ(ret, fe::ILLEGAL_JSON);
}
TEST_F(fusion_rule_manager_uttest, parse_json_failed6)
{
    FusionRuleJsonNode test_object;
    nlohmann::json json_object = {{"Not_support_key", "Conv"}, {"type", {"min"}}};
    Status ret = test_object.ParseJson(json_object);
    EXPECT_EQ(ret, fe::ILLEGAL_JSON);
}
TEST_F(fusion_rule_manager_uttest, add_input_anchor_failed)
{
    FusionRuleNodeConstructor test_object;
    map<string, FusionRuleAttrValuePtr> attributes;
    FusionRuleAttrValuePtr attrvalueptr = make_shared<FusionRuleAttrValue>();
    attributes.insert(pair<string, FusionRuleAttrValuePtr> ("attr1", attrvalueptr));
    FusionRuleNodePtr test_node = CreateFusionRuleNode("test_node", {"Test"}, {}, {0}, attributes);
    Status ret = test_object.AddAttr(test_node, "attr1", attrvalueptr);
    EXPECT_EQ(ret, fe::ILLEGAL_RULE);
}
TEST_F(fusion_rule_manager_uttest, check_node_validity_failed)
{
    FusionRuleNodeConstructor test_object;
    FusionRuleNodePtr test_node = CreateFusionRuleNode("test_node", {"Test"}, {}, {}, {});
    Status ret = test_object.CheckNodeValidity(test_node);
    EXPECT_EQ(ret, fe::ILLEGAL_RULE);
}
TEST_F(fusion_rule_manager_uttest, check_node_validity_failed1)
{
    FusionRuleNodeConstructor test_object;
    FusionRuleNodePtr test_node = CreateFusionRuleNode("test_node", {"Test"}, {}, {}, {});
    Status ret = test_object.CheckNodeValidity(test_node);
    EXPECT_EQ(ret, fe::ILLEGAL_RULE);
}
TEST_F(fusion_rule_manager_uttest, check_node_validity_failed2)
{
    FusionRuleNodeConstructor test_object;
    FusionRuleNodePtr node = make_shared<fe::FusionRuleNode>();
    vector<string> node_types = {"node_types"};
    node->node_name_ = "node_name";
    node->node_type_ = node_types;
    for (size_t i = 0; i < 2; ++i) {
        int index = 0;
        string anchor_name = "_input_" + to_string(index);
        auto input_anchor = CreateAnchor(index, anchor_name, node, {});
        node->input_data_anchors_.push_back(input_anchor);
    }
    for (size_t i = 0; i < 2; ++i) {
        int index = i;
        string anchor_name = "_output_" + to_string(index);
        auto output_anchor = CreateAnchor(index, anchor_name, node, {});
        node->output_data_anchors_.push_back(output_anchor);
    }
    node->attributes_ = {};
    Status ret = test_object.CheckNodeValidity(node);
    EXPECT_EQ(ret, fe::ILLEGAL_RULE);
}
TEST_F(fusion_rule_manager_uttest, check_node_validity_failed3)
{
    FusionRuleNodeConstructor test_object;
    FusionRuleNodePtr node = make_shared<fe::FusionRuleNode>();
    vector<string> node_types = {"node_types"};
    node->node_name_ = "node_name";
    node->node_type_ = node_types;
    for (size_t i = 0; i < 2; ++i) {
        int index = i;
        string anchor_name = "_input_" + to_string(index);
        auto input_anchor = CreateAnchor(index, anchor_name, node, {});
        node->input_data_anchors_.push_back(input_anchor);
    }
    for (size_t i = 0; i < 2; ++i) {
        int index = 0;
        string anchor_name = "_output_" + to_string(index);
        auto output_anchor = CreateAnchor(index, anchor_name, node, {});
        node->output_data_anchors_.push_back(output_anchor);
    }
    node->attributes_ = {};
    Status ret = test_object.CheckNodeValidity(node);
    EXPECT_EQ(ret, fe::ILLEGAL_RULE);
}
TEST_F(fusion_rule_manager_uttest, construct_failed)
{
    FusionRuleAnchorConstructor test_object;
    FusionRuleAnchorPtr anchor = make_shared<FusionRuleAnchor>();
    Status ret = test_object.Construct(anchor, -3, "name");
    EXPECT_EQ(ret, fe::ILLEGAL_RULE);
}
TEST_F(fusion_rule_manager_uttest, get_str_from_attr_alue_succ1)
{
    ge::GeAttrValue attr_value_float = GeAttrValue::CreateFrom<float>(1.1);
    auto attr_value = fe::GetStrFromAttrValue(attr_value_float);
    EXPECT_EQ(attr_value, "1.100000");
    ge::GeAttrValue attr_value_datatype = GeAttrValue::CreateFrom<ge::DataType>(ge::DataType::DT_INT4);
    attr_value = fe::GetStrFromAttrValue(attr_value_datatype);
    EXPECT_EQ(attr_value, "");
}
TEST_F(fusion_rule_manager_uttest, get_str_from_attr_alue_fail1)
{
    ge::GeAttrValue attr_value_float = GeAttrValue::CreateFrom<int64_t>(1);
    auto attr_value = fe::GetStrOfFloat(attr_value_float);
    EXPECT_EQ(attr_value, "");
}
TEST_F(fusion_rule_manager_uttest, get_str_from_attr_alue_fail2)
{
    ge::GeAttrValue attr_value_float = GeAttrValue::CreateFrom<float>(1.1);
    auto attr_value = fe::GetStrOfInt(attr_value_float);
    EXPECT_EQ(attr_value, "");
}
TEST_F(fusion_rule_manager_uttest, get_str_from_attr_alue_fail3)
{
    ge::GeAttrValue attr_value_float = GeAttrValue::CreateFrom<int64_t>(1);
    auto attr_value = fe::GetStrOfBool(attr_value_float);
    EXPECT_EQ(attr_value, "");
}
TEST_F(fusion_rule_manager_uttest, get_str_from_attr_alue_fail4)
{
    ge::GeAttrValue attr_value_float = GeAttrValue::CreateFrom<float>(1.1);
    auto attr_value = fe::GetStrOfListInt(attr_value_float);
    EXPECT_EQ(attr_value, "");
}
TEST_F(fusion_rule_manager_uttest, get_str_from_attr_alue_fail5)
{
    ge::GeAttrValue attr_value_float = GeAttrValue::CreateFrom<int64_t>(1);
    auto attr_value = fe::GetStrOfListFloat(attr_value_float);
    EXPECT_EQ(attr_value, "");
}
TEST_F(fusion_rule_manager_uttest, get_str_from_attr_alue_fail6)
{
    ge::GeAttrValue attr_value_float = GeAttrValue::CreateFrom<int64_t>(1);
    auto attr_value = fe::GetStrOfListBool(attr_value_float);
    EXPECT_EQ(attr_value, "");
}
TEST_F(fusion_rule_manager_uttest, check_fusion_rule_pattern_failed)
{
    FusionRulePatternConstructor fusionrule_pattern;
    FusionRulePatternPtr pattern = make_shared<FusionRulePattern>();
    FusionRuleNodePtr test_node = CreateFusionRuleNode("test_node", {"Test"}, {}, {}, {});
    pattern->input_info_.push_back(test_node);
    Status ret = fusionrule_pattern.CheckFusionRulePattern(pattern);
    EXPECT_EQ(ret, fe::ILLEGAL_RULE);
}
TEST_F(fusion_rule_manager_uttest, check_fusion_rule_pattern_failed1)
{
    FusionRulePatternConstructor fusionrule_pattern;
    FusionRulePatternPtr pattern = make_shared<FusionRulePattern>();
    FusionRuleNodePtr test_node = CreateFusionRuleNode("test_node", {"Test"}, {}, {}, {});
    pattern->output_info_.push_back(test_node);
    Status ret = fusionrule_pattern.CheckFusionRulePattern(pattern);
    EXPECT_EQ(ret, fe::ILLEGAL_RULE);
}

TEST_F(fusion_rule_manager_uttest, test_init_graph_rules_aicore)
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

    Configuration::Instance(engine_name).content_map_[CONFIG_KEY_GRAPH_FILE] = "lib64/plugin/opskernel/fusion_rules/ai_core/built_in_graph_rules.json";
    FEOpsKernelInfoStorePtr aicore_ops_kernel_info_store_ptr_  = make_shared<fe::FEOpsKernelInfoStore>(engine_name);
    auto frm = std::make_shared<FusionRuleManager>(aicore_ops_kernel_info_store_ptr_);
    Status ret = frm->InitGraphRules(engine_name);
    EXPECT_EQ(ret, fe::SUCCESS);
    system(("rm -rf " + current_dir + "plugin").c_str());
}

TEST_F(fusion_rule_manager_uttest, test_init_graph_rules_vectorcore) {
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
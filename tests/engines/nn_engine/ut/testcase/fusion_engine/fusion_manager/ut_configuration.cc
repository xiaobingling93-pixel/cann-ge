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
#include <iostream>
#include <stdlib.h>
#include <list>
#include "common/comm_error_codes.h"
#include "fe_llt_utils.h"
#define protected public
#define private public
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "common/fe_context_utils.h"
#include "common/util/json_util.h"
#include "common/util/constants.h"
#include "common/config_parser/op_impl_mode_config_parser.h"
#include "common/config_parser/op_cust_dtypes_config_parser.h"
#include "common/config_parser/modify_mixlist_config_parser.h"
#include "common/config_parser/op_debug_config_parser.h"
#include "platform/platform_info.h"
#include "common/platform_utils.h"
#include "mmpa/src/mmpa_stub.h"
#include "graph/ge_context.h"
#include "ge/ge_api_types.h"
#include "common/aicore_util_constants.h"
#include "fusion_config_manager/fusion_config_parser.h"
#include "graph/ge_local_context.h"
#include "register/optimization_option_registry.h"
#undef private
#undef protected
#include "graph/ge_local_context.h"
using namespace std;
using namespace fe;

class configuration_ut: public testing::Test
{
protected:
  static void SetUpTestCase() {
    std::string soc_version = "Ascend910B1";
    PlatformInfoManager::Instance().opti_compilation_info_.soc_version = soc_version;
    PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion(soc_version);
    PlatformUtils::Instance().soc_version_ = soc_version;
    PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::IsaArchVersion)] = static_cast<int64_t>(ISAArchVersion::EN_ISA_ARCH_V300);
  }
  void SetUp() {
    map<string, string> options;
    options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
    options.emplace(ge::MODIFY_MIXLIST, GetCodeDir() + "/tests/engines/nn_engine/config/mix_list/op_mix_list1.json");
    ge::GetThreadLocalContext().SetGraphOption(options);
    Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
    config.is_init_ = false;
    string soc_version = "Ascend910B1";
    PlatformUtils::Instance().soc_version_ = soc_version;
    PlatformUtils::Instance().short_soc_version_ = "Ascend910B";
    config.Initialize(options);
    if (config.impl_mode_parser_ == nullptr) {
      config.impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
    }
    if (config.cust_dtypes_parser_ == nullptr) {
      config.cust_dtypes_parser_ = std::make_shared<OpCustDtypesConfigParser>();
    }
    if (config.mix_list_parser_ == nullptr) {
      config.mix_list_parser_ = std::make_shared<ModifyMixlistConfigParser>();
      config.mix_list_parser_->InitializeFromOptions(options);
    }
  }

  void TearDown() {
    map<string, string> options;
    options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
    ge::GetThreadLocalContext().SetGraphOption(options);
    Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
    config.is_init_ = false;
    string soc_version = "Ascend910B1";
    PlatformUtils::Instance().soc_version_ = soc_version;
    config.Initialize(options);
  }
// AUTO GEN PLEASE DO NOT MODIFY IT
};

string ConfGetAscendPath() {
  const char *ascend_custom_path_ptr = std::getenv("ASCEND_INSTALL_PATH");
  string ascend_path;
  if (ascend_custom_path_ptr != nullptr) {
    ascend_path = fe::RealPath(string(ascend_custom_path_ptr));
  } else {
    const char *ascend_home_path_ptr = std::getenv("ASCEND_HOME");
    if (ascend_home_path_ptr != nullptr) {
      ascend_path = fe::RealPath(string(ascend_home_path_ptr));
    } else {
      ascend_path = "/mnt/d/Ascend";
    }
  }
  return ascend_path;
}

TEST_F(configuration_ut, init_and_finalize)
{
  Configuration config(fe::AI_CORE_NAME);
  map<string, string> options;
  options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
  ge::GetThreadLocalContext().SetGraphOption(options);
  string soc_version = "Ascend910B1";
  PlatformUtils::Instance().soc_version_ = soc_version;
  Status status = config.Initialize(options);
  EXPECT_EQ(status, SUCCESS);
  status = config.Initialize(options);
  EXPECT_EQ(status, SUCCESS);
  status = config.Finalize();
  EXPECT_EQ(status, SUCCESS);
  status = config.Finalize();
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(configuration_ut, get_boolvalue)
{
    bool bool_value = Configuration::Instance(fe::AI_CORE_NAME).GetBoolValue("needl2fusion", false);
    ASSERT_FALSE(bool_value);
}

TEST_F(configuration_ut, get_context_string)
{
  map<string, string> options;
  options.emplace("AAA", "111");
  options.emplace("BBB", "000");
  options.emplace("CCC", "");
  ge::GetThreadLocalContext().SetGraphOption(options);
  string str_value = FEContextUtils::GetGeContextValue("AAA");
  EXPECT_EQ(str_value, "111");

  str_value = FEContextUtils::GetGeContextValue("BBB");
  EXPECT_EQ(str_value, "000");

  str_value = FEContextUtils::GetGeContextValue("CCC");
  EXPECT_EQ(str_value, "");

  str_value = FEContextUtils::GetGeContextValue("DDD");
  EXPECT_EQ(str_value, "");
}

TEST_F(configuration_ut, get_boolvalue_abnormal)
{
    Configuration::Instance(fe::AI_CORE_NAME).content_map_.emplace("needl2fusion2", "!@##$");
    bool bool_value = Configuration::Instance(fe::AI_CORE_NAME).GetBoolValue("needl2fusion2", false);
    ASSERT_FALSE(bool_value);
}

TEST_F(configuration_ut, loadconfigfile_success)
{
    Configuration config(fe::AI_CORE_NAME);
    map<string, string> options;
    options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
    string soc_version = "Ascend910B";
    PlatformUtils::Instance().soc_version_ = soc_version;
    config.Initialize(options);
    config.content_map_.clear();
    Status status = config.LoadConfigFile();
    EXPECT_EQ(status, SUCCESS);
}

TEST_F(configuration_ut, AssembleOpsStoreInfoVector_success)
{
    Configuration config(fe::AI_CORE_NAME);
    map<string, string> options;
    string soc_version = "Ascend910B";
    PlatformUtils::Instance().soc_version_ = soc_version;
    options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
    config.Initialize(options);
    Status status = config.AssembleOpsStoreInfoVector();
    EXPECT_EQ(status, SUCCESS);
}

TEST_F(configuration_ut, AssembleOpsStoreInfoVector_success1)
{
    Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
    PlatformUtils::Instance().soc_version_ = "Ascend310";
    PlatformUtils::Instance().short_soc_version_ = "Ascend310";
    config.content_map_.clear();
    config.content_map_.emplace("op.store.tbe-builtin", "2|6|/tests/engines/nn_engine/config/fe_config|/tests/engines/nn_engine/config/fe_config|true|true");
    config.ascend_ops_path_ = GetCurpath() + "../../../../../..";
    Status status = config.AssembleOpsStoreInfoVector();
    EXPECT_EQ(status, SUCCESS);
    FEOpsStoreInfo op_store_info;
    config.GetOpStoreInfoByImplType(EN_IMPL_HW_TBE, op_store_info);
    string sub_path = "ascend310";
    int32_t pos = op_store_info.cfg_file_path.rfind(sub_path);
    EXPECT_EQ(pos, op_store_info.cfg_file_path.length()-sub_path.length());
}

TEST_F(configuration_ut, AssembleOpsStoreInfoVector_success3)
{
    Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
    PlatformUtils::Instance().soc_version_ = "Ascend910A";
    PlatformUtils::Instance().short_soc_version_ = "Ascend910";
    config.content_map_.clear();
    config.content_map_.emplace("op.store.tbe-builtin", "2|6|/tests/engines/nn_engine/config/fe_config|/tests/engines/nn_engine/config/fe_config|true|true");
    config.ascend_ops_path_ = GetCurpath() + "../../../../../..";
    Status status = config.AssembleOpsStoreInfoVector();
    EXPECT_EQ(status, SUCCESS);
    FEOpsStoreInfo op_store_info;
    config.GetOpStoreInfoByImplType(EN_IMPL_HW_TBE, op_store_info);
    string sub_path = "ascend910";
    int32_t pos = op_store_info.cfg_file_path.rfind(sub_path);
    EXPECT_EQ(pos, op_store_info.cfg_file_path.length()-sub_path.length());
}

TEST_F(configuration_ut, AssembleOpsStoreInfoVector_success4)
{
    Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
    PlatformUtils::Instance().soc_version_ = "Ascend910B";
    PlatformUtils::Instance().short_soc_version_ = "Ascend910";
    config.content_map_.clear();
    config.content_map_.emplace("op.store.tbe-builtin", "2|6|/tests/engines/nn_engine/config/fe_config|/tests/engines/nn_engine/config/fe_config|true|true");
    config.ascend_ops_path_ = GetCurpath() + "../../../../../..";
    Status status = config.AssembleOpsStoreInfoVector();
    EXPECT_EQ(status, SUCCESS);
    FEOpsStoreInfo op_store_info;
    config.GetOpStoreInfoByImplType(EN_IMPL_HW_TBE, op_store_info);
    string sub_path = "ascend910";
    int32_t pos = op_store_info.cfg_file_path.rfind(sub_path);
    EXPECT_EQ(pos, op_store_info.cfg_file_path.length()-sub_path.length());
}

TEST_F(configuration_ut, AssembleOpsStoreInfoVector_failed1)
{
    Configuration config(fe::AI_CORE_NAME);
    map<string, string> options;
    string soc_version = "Ascend910B";
    PlatformUtils::Instance().soc_version_ = soc_version;
    options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
    config.Initialize(options);
    config.content_map_.clear();
    config.content_map_.emplace("op.store.cce-custom", "0|0|./config/fe_config/cce_custom_opinfo|./config/fe_config/cce_custom_opinfo|false|false");
    config.content_map_.emplace("op.store.", "1|1|./config/fe_config/tik_custom_opinfo|./config/fe_config/cce_custom_opinfo|false|false");

    Status status = config.AssembleOpsStoreInfoVector();
    EXPECT_EQ(status, OPSTORE_NAME_EMPTY);
}

TEST_F(configuration_ut, AssembleOpsStoreInfoVector_failed2)
{
    Configuration config(fe::AI_CORE_NAME);
    map<string, string> options;
    string soc_version = "Ascend910B";
    PlatformUtils::Instance().soc_version_ = soc_version;
    options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
    config.Initialize(options);
    config.content_map_.clear();
    config.content_map_.emplace("op.store.cce-custom", "0|0|./config/fe_config/cce_custom_opinfo|./config/fe_config/cce_custom_opinfo|false|false");
    config.content_map_.emplace("op.store.tik-custom", "");

    Status status = config.AssembleOpsStoreInfoVector();
    EXPECT_EQ(status, OPSTORE_VALUE_EMPTY);
}

TEST_F(configuration_ut, AssembleOpsStoreInfoVector_failed3)
{
    Configuration config(fe::AI_CORE_NAME);
    map<string, string> options;
    string soc_version = "Ascend910B";
    PlatformUtils::Instance().soc_version_ = soc_version;
    options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
    config.Initialize(options);
    config.content_map_.clear();
    config.content_map_.emplace("op.store.cce-custom", "0|0|./config/fe_config/cce_custom_opinfo|./config/fe_config/cce_custom_opinfo|false|false");
    config.content_map_.emplace("op.store.tik-custom", "0|0|./config/fe_config/cce_custom_opinfo|false|false");

    Status status = config.AssembleOpsStoreInfoVector();
    EXPECT_EQ(status, OPSTORE_VALUE_ITEM_SIZE_INCORRECT);
}

TEST_F(configuration_ut, AssembleOpsStoreInfoVector_failed4)
{
    Configuration config(fe::AI_CORE_NAME);
    map<string, string> options;
    string soc_version = "Ascend910B";
    PlatformUtils::Instance().soc_version_ = soc_version;
    options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
    config.Initialize(options);
    config.content_map_.clear();
    config.content_map_.emplace("op.store.cce-custom", "0|0|./config/fe_config/cce_custom_opinfo|./config/fe_config/cce_custom_opinfo|false|false");
    config.content_map_.emplace("op.store.tik-custom", "1|1|./config/fe_config/tik_custom_opinfo|x|qwer|false|false|false");

    Status status = config.AssembleOpsStoreInfoVector();
    EXPECT_EQ(status, OPSTORE_VALUE_ITEM_SIZE_INCORRECT);
}

TEST_F(configuration_ut, AssembleOpsStoreInfoVector_failed5)
{
    Configuration config(fe::AI_CORE_NAME);
    map<string, string> options;
    string soc_version = "Ascend910B";
    PlatformUtils::Instance().soc_version_ = soc_version;
    options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
    config.Initialize(options);
    config.content_map_.clear();
    config.content_map_.emplace("op.store.cce-custom", "0|0|./config/fe_config/cce_custom_opinfo|./config/fe_config/cce_custom_opinfo|false|false");
    config.content_map_.emplace("op.store.tik-custom", "1|1|./config/fe_config/tik_custom_opinfo||false|false");

    Status status = config.AssembleOpsStoreInfoVector();
    EXPECT_EQ(status, OPSTORE_VALUE_ITEM_EMPTY);
}

TEST_F(configuration_ut, AssembleOpsStoreInfoVector_failed6)
{
    Configuration config(fe::AI_CORE_NAME);
    map<string, string> options;
    string soc_version = "Ascend910B";
    PlatformUtils::Instance().soc_version_ = soc_version;
    options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
    config.Initialize(options);
    config.content_map_.clear();
    config.content_map_.emplace("op.store.cce-custom", "0|0|./config/fe_config/cce_custom_opinfo|./config/fe_config/cce_custom_opinfo|false|false");
    config.content_map_.emplace("op.store.tik-custom", "1|1||./config/fe_config/tik_custom_opinfo|false|false");

    Status status = config.AssembleOpsStoreInfoVector();
    EXPECT_EQ(status, OPSTORE_VALUE_ITEM_EMPTY);
}

TEST_F(configuration_ut, AssembleOpsStoreInfoVector_failed7)
{
    Configuration config(fe::AI_CORE_NAME);
    map<string, string> options;
    string soc_version = "Ascend910B";
    PlatformUtils::Instance().soc_version_ = soc_version;
    options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
    config.Initialize(options);
    config.content_map_.clear();

    Status status = config.AssembleOpsStoreInfoVector();
    EXPECT_EQ(status, OPSTORE_EMPTY);
}

TEST_F(configuration_ut, AssembleOpsStoreInfoVector_failed8)
{
    Configuration config(fe::AI_CORE_NAME);
    map<string, string> options;
    string soc_version = "Ascend910B";
    PlatformUtils::Instance().soc_version_ = soc_version;
    options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
    config.Initialize(options);
    config.content_map_.clear();
    config.content_map_.emplace("op.store.cce-custom", "0|0|./config/fe_config/cce_custom_opinfo|./config/fe_config/cce_custom_opinfo|false|false");
    config.content_map_.emplace("op.store.tik-custom", "x|1|./config/fe_config/tik_custom_opinfo|./config/fe_config/cce_custom_opinfo|false|false");

    Status status = config.AssembleOpsStoreInfoVector();
    EXPECT_EQ(status, OPSTORE_PRIORITY_INVALID);
}

TEST_F(configuration_ut, AssembleOpsStoreInfoVector_failed9)
{
    Configuration config(fe::AI_CORE_NAME);
    map<string, string> options;
    string soc_version = "Ascend910B";
    PlatformUtils::Instance().soc_version_ = soc_version;
    options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
    config.Initialize(options);
    config.content_map_.clear();
    config.content_map_.emplace("op.store.cce-custom", "0|0|./config/fe_config/cce_custom_opinfo|./config/fe_config/cce_custom_opinfo|false|false");
    config.content_map_.emplace("op.store.tik-custom", "1|c|./config/fe_config/tik_custom_opinfo|./config/fe_config/cce_custom_opinfo|false|false");

    Status status = config.AssembleOpsStoreInfoVector();
    EXPECT_EQ(status, OPSTORE_OPIMPLTYPE_INVALID);
}

TEST_F(configuration_ut, AssembleOpsStoreInfoVector_failed10)
{
    Configuration config(fe::AI_CORE_NAME);
    map<string, string> options;
    string soc_version = "Ascend910B";
    PlatformUtils::Instance().soc_version_ = soc_version;
    options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
    config.Initialize(options);
    config.content_map_.clear();
    config.content_map_.emplace("op.store.cce-custom", "0|0|./config/fe_config/cce_custom_opinfo|./config/fe_config/cce_custom_opinfo|false|false");
    config.content_map_.emplace("op.store.tik-custom", "1|1099511627776|./config/fe_config/tik_custom_opinfo|./config/fe_config/cce_custom_opinfo|false|false");
    cout << "AssembleOpsStoreInfoVector_failed10 TEST:" << endl;
    Status status = config.AssembleOpsStoreInfoVector();
    EXPECT_EQ(status, OPSTORE_OPIMPLTYPE_INVALID);
}

TEST_F(configuration_ut, AssembleOpsStoreInfoVector_failed11)
{
    Configuration config(fe::AI_CORE_NAME);
    map<string, string> options;
    string soc_version = "Ascend910B";
    PlatformUtils::Instance().soc_version_ = soc_version;
    options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
    config.Initialize(options);
    config.content_map_.clear();
    config.content_map_.emplace("op.store.cce-custom", "0|0|./config/fe_config/cce_custom_opinfo|./config/fe_config/cce_custom_opinfo|false|false");
    config.content_map_.emplace("op.store.tik-custom", "1|-1|./config/fe_config/tik_custom_opinfo|./config/fe_config/cce_custom_opinfo|false|false");

    Status status = config.AssembleOpsStoreInfoVector();
    EXPECT_EQ(status, OPSTORE_OPIMPLTYPE_INVALID);
}

TEST_F(configuration_ut, AssembleOpsStoreInfoVector_failed12)
{
    Configuration config(fe::AI_CORE_NAME);
    map<string, string> options;
    string soc_version = "Ascend910B";
    PlatformUtils::Instance().soc_version_ = soc_version;
    options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
    config.Initialize(options);
    config.content_map_.clear();
    config.content_map_.emplace("op.store.cce-custom", "0|0|./config/fe_config/cce_custom_opinfo|./config/fe_config/cce_custom_opinfo|false|false");
    config.content_map_.emplace("op.store.tik-custom", "1|2|./config/fe_config/tik_custom_opinfo|./config/fe_config/cce_custom_opinfo|false|false");
    config.content_map_.emplace("op.store.tdk-custom", "2|2|./config/fe_config/tik_custom_opinfo|./config/fe_config/cce_custom_opinfo|false|false");

    Status status = config.AssembleOpsStoreInfoVector();
    EXPECT_EQ(status, OPSTORE_OPIMPLTYPE_REPEAT);
}

TEST_F(configuration_ut, AssembleOpsStoreInfoVector_failed13)
{
    Configuration config(fe::AI_CORE_NAME);
    map<string, string> options;
    string soc_version = "Ascend910B";
    PlatformUtils::Instance().soc_version_ = soc_version;
    options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
    config.Initialize(options);
    config.content_map_.clear();
    config.content_map_.emplace("op.store.cce-custom", "0|0|./config/fe_config/cce_custom_opinfo|./config/fe_config/cce_custom_opinfo|false|false");
    config.content_map_.emplace("op.store.tik-custom", "0|2|./config/fe_config/tik_custom_opinfo|./config/fe_config/cce_custom_opinfo|false|false");

    Status status = config.AssembleOpsStoreInfoVector();
    EXPECT_EQ(status, OPSTORE_PRIORITY_INVALID);
}

TEST_F(configuration_ut, get_opsstoreinfo)
{
    Configuration::Instance(fe::AI_CORE_NAME).is_init_ = false;
    FEOpsStoreInfo tbe_builtin {
            6,
            "tbe-builtin",
            EN_IMPL_HW_TBE,
            GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/format_selector/fe_config/tbe_dynamic_opinfo",
            ""
    };
    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(tbe_builtin);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    vector<FEOpsStoreInfo> ops_store_info_vec = Configuration::Instance(fe::AI_CORE_NAME).GetOpsStoreInfo();
    EXPECT_EQ(ops_store_info_vec.size(), 1);

    for (FEOpsStoreInfo &op_store_info : ops_store_info_vec)
    {
        if (op_store_info.op_impl_type == EN_IMPL_HW_TBE)
        {
            EXPECT_EQ(op_store_info.fe_ops_store_name, "tbe-builtin");
        }
        if (op_store_info.op_impl_type == EN_IMPL_VECTOR_CORE_HW_TBE)
        {
            EXPECT_EQ(op_store_info.fe_ops_store_name, "vectorcore-tbe-builtin");
            EXPECT_EQ(op_store_info.need_pre_compile, true);
            EXPECT_EQ(op_store_info.need_compile, false);
        }
        if (op_store_info.op_impl_type == EN_IMPL_VECTOR_CORE_CUSTOM_TBE)
        {
            EXPECT_EQ(op_store_info.fe_ops_store_name, "vectorcore-tbe-builtin");
            EXPECT_EQ(op_store_info.need_pre_compile, true);
            EXPECT_EQ(op_store_info.need_compile, true);
        }
        if (op_store_info.op_impl_type == EN_IMPL_PLUGIN_TBE)
        {
            EXPECT_EQ(op_store_info.need_pre_compile, true);
            EXPECT_EQ(op_store_info.need_compile, false);
        }
    }
}

TEST_F(configuration_ut, getstringvalue_success)
{
    string stringvalue;
    Status status = Configuration::Instance(fe::AI_CORE_NAME).GetStringValue("l2cache.datavisitdist.threshold", stringvalue);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(stringvalue, "5");
}

TEST_F(configuration_ut, getstringvalue_failed)
{
    string stringvalue;
    Status status = Configuration::Instance(fe::AI_CORE_NAME).GetStringValue("fusionrulemgr.xxx", stringvalue);
    EXPECT_EQ(status, FAILED);
}

TEST_F(configuration_ut, GetOpStoreInfoByImplType_success)
{
    FEOpsStoreInfo tbe_builtin {
            6,
            "tbe-custom",
            EN_IMPL_CUSTOM_TBE,
            GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/format_selector/fe_config/tbe_dynamic_opinfo",
            ""
    };
    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(tbe_builtin);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    FEOpsStoreInfo op_store_info;
    Status status = Configuration::Instance(fe::AI_CORE_NAME).GetOpStoreInfoByImplType(EN_IMPL_CUSTOM_TBE, op_store_info);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(op_store_info.fe_ops_store_name, "tbe-custom");
}

TEST_F(configuration_ut, GetOpStoreInfoByImplType_failed)
{
    FEOpsStoreInfo op_store_info;
    Status status = Configuration::Instance(fe::AI_CORE_NAME).GetOpStoreInfoByImplType(EN_RESERVED, op_store_info);
    EXPECT_EQ(status, FAILED);
}

TEST_F(configuration_ut, getgraphfilepath)
{
    string graph_file_path;
    Status status = Configuration::Instance(fe::AI_CORE_NAME).GetGraphFilePath(graph_file_path);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_NE(graph_file_path, "");
}

TEST_F(configuration_ut, getcustomfilepath)
{
    string custom_file_path;
    Status status = Configuration::Instance(fe::AI_CORE_NAME).GetCustomFilePath(custom_file_path);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(custom_file_path, "");
}

TEST_F(configuration_ut, dsagetgraphfilepath)
{
    string graph_file_path;
    Status status = Configuration::Instance(fe::kDsaCoreName).GetGraphFilePath(graph_file_path);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(graph_file_path, "");
}

TEST_F(configuration_ut, dsagetcustomfilepath)
{
    string custom_file_path;
    Status status = Configuration::Instance(fe::kDsaCoreName).GetCustomFilePath(custom_file_path);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(custom_file_path, "");
}

TEST_F(configuration_ut, dsagetcustompassfilepath)
{
    string graph_file_path;
    Status status = Configuration::Instance(fe::kDsaCoreName).GetCustomPassFilePath(graph_file_path);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(graph_file_path, "");
}

TEST_F(configuration_ut, dsagetbuiltinpassfilepath)
{
    string custom_file_path;
    Status status = Configuration::Instance(fe::kDsaCoreName).GetBuiltinPassFilePath(custom_file_path);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(custom_file_path, "");
}

TEST_F(configuration_ut, get_opsstoreinfo_vectorcore)
{
  Configuration config(fe::VECTOR_CORE_NAME);
  map<string, string> options;
  string soc_version = "Ascend910B";
  PlatformUtils::Instance().soc_version_ = soc_version;
  options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
  config.Initialize(options);
  vector<FEOpsStoreInfo> ops_store_info_vec = config.GetOpsStoreInfo();
  EXPECT_EQ(ops_store_info_vec.size(), 1);

  for (FEOpsStoreInfo &op_store_info : ops_store_info_vec)
  {
    if (op_store_info.op_impl_type == EN_IMPL_VECTOR_CORE_HW_TBE)
    {
      EXPECT_EQ(op_store_info.fe_ops_store_name, "vector-core-tbe-builtin");
      EXPECT_EQ(op_store_info.need_pre_compile, true);
      EXPECT_EQ(op_store_info.need_compile, true);
    }
    if (op_store_info.op_impl_type == EN_IMPL_VECTOR_CORE_CUSTOM_TBE)
    {
      EXPECT_EQ(op_store_info.fe_ops_store_name, "vector-core-tbe-custom");
      EXPECT_EQ(op_store_info.need_pre_compile, true);
      EXPECT_EQ(op_store_info.need_compile, true);
    }
  }
}

TEST_F(configuration_ut, getgraphfilepath_vectorcore)
{
  string graph_file_path;
  Configuration config(fe::VECTOR_CORE_NAME);
  map<string, string> options;
  string soc_version = "Ascend910B";
  PlatformUtils::Instance().soc_version_ = soc_version;
  options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
  config.Initialize(options);
  Status status = config.GetGraphFilePath(graph_file_path);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_NE(graph_file_path, "");
}

TEST_F(configuration_ut, getcustomfilepath_vectorcore)
{
  string custom_file_path;
  Configuration config(fe::VECTOR_CORE_NAME);
  map<string, string> options;
  string soc_version = "Ascend910B";
  PlatformUtils::Instance().soc_version_ = soc_version;
  options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
  config.Initialize(options);
  Status status = config.GetCustomFilePath(custom_file_path);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(custom_file_path, "");
}

TEST_F(configuration_ut, init_op_precision_mode_case1)
{
  Configuration config(AI_CORE_NAME);
  config.impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
  map<string, string> options;
  // unix file format
  options.emplace("ge.exec.op_precision_mode",
                  GetCodeDir() + "/tests/engines/nn_engine/config/op_precision_mode_1.ini");
  Status status = config.impl_mode_parser_->InitializeFromOptions(options);
  EXPECT_EQ(status, SUCCESS);
  std::string str;
  EXPECT_EQ(config.GetOpImplMode("", "ccc", str), true);
  EXPECT_EQ(str, "high_performance");
  EXPECT_EQ(config.GetOpImplMode("", "ddd", str), true);
  EXPECT_EQ(str, "high_precision");
  EXPECT_EQ(config.GetOpImplMode("op_name1", "", str), true);
  EXPECT_EQ(str, "high_precision");
  EXPECT_EQ(config.GetOpImplMode("op_name2", "", str), false);

  ge::GetThreadLocalContext().SetGraphOption(options);
  status = config.impl_mode_parser_->InitializeFromContext();
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(config.GetOpImplMode("", "ccc", str), true);
  EXPECT_EQ(str, "high_performance");
  EXPECT_EQ(config.GetOpImplMode("", "ddd", str), true);
  EXPECT_EQ(str, "high_precision");
  EXPECT_EQ(config.GetOpImplMode("op_name1", "", str), true);
  EXPECT_EQ(str, "high_precision");
  EXPECT_EQ(config.GetOpImplMode("op_name2", "", str), false);
}

TEST_F(configuration_ut, init_op_precision_mode_case2)
{
  Configuration config(AI_CORE_NAME);
  config.impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
  map<string, string> options;
  // empty file
  options.emplace("ge.exec.op_precision_mode",
                  GetCodeDir() + "/tests/engines/nn_engine/config/op_precision_mode_2.ini");
  Status status = config.impl_mode_parser_->InitializeFromOptions(options);
  EXPECT_EQ(status, SUCCESS);
  std::string str;
  EXPECT_EQ(config.GetOpImplMode("op_name", "ccc", str), false);
  EXPECT_EQ(config.GetOpImplMode("op_name", "ddd", str), false);
}

TEST_F(configuration_ut, init_op_precision_mode_case3)
{
  Configuration config(AI_CORE_NAME);
  config.impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
  map<string, string> options;
  // dos file format
  options.emplace("ge.exec.op_precision_mode",
                  GetCodeDir() + "/tests/engines/nn_engine/config/op_precision_mode_3.ini");
  Status status = config.impl_mode_parser_->InitializeFromOptions(options);
  EXPECT_EQ(status, SUCCESS);
  std::string str;
  EXPECT_EQ(config.GetOpImplMode("", "ccc", str), true);
  EXPECT_EQ(str, "high_precision");
  EXPECT_EQ(config.GetOpImplMode("", "ddd", str), false);
  EXPECT_EQ(config.GetOpImplMode("op_name1", "", str), true);
  EXPECT_EQ(str, "high_precision");
  EXPECT_EQ(config.GetOpImplMode("op_name2", "", str), true);
  EXPECT_EQ(str, "high_performance");

  options["ge.exec.op_precision_mode"] = GetCodeDir() + "/tests/engines/nn_engine/config/op_precision_mode_1.ini";
  ge::GetThreadLocalContext().SetGraphOption(options);
  status = config.impl_mode_parser_->InitializeFromContext();
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(config.GetOpImplMode("", "ccc", str), true);
  EXPECT_EQ(str, "high_performance");
  EXPECT_EQ(config.GetOpImplMode("", "ddd", str), true);
  EXPECT_EQ(str, "high_precision");
  EXPECT_EQ(config.GetOpImplMode("op_name1", "", str), true);
  EXPECT_EQ(str, "high_precision");
  EXPECT_EQ(config.GetOpImplMode("op_name2", "", str), false);
}

TEST_F(configuration_ut, init_op_precision_mode_case4)
{
  Configuration config(AI_CORE_NAME);
  config.impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
  map<string, string> options;
  options["ge.exec.op_precision_mode"] = GetCodeDir() + "/tests/engines/nn_engine/config/op_precision_mode_0.ini";
  Status status = config.impl_mode_parser_->InitializeFromOptions(options);
  EXPECT_EQ(status, FAILED);

  options["ge.exec.op_precision_mode"] = GetCodeDir() + "/tests/engines/nn_engine/config/op_precision_mode_1.ini";
  ge::GetThreadLocalContext().SetGraphOption(options);
  status = config.impl_mode_parser_->InitializeFromContext();
  EXPECT_EQ(status, SUCCESS);

  status = config.impl_mode_parser_->InitializeFromOptions(options);
  EXPECT_EQ(status, SUCCESS);

  options["ge.exec.op_precision_mode"] = GetCodeDir() + "/tests/engines/nn_engine/config/op_precision_mode_0.ini";
  ge::GetThreadLocalContext().SetGraphOption(options);
  status = config.impl_mode_parser_->InitializeFromContext();
  EXPECT_EQ(status, FAILED);
}

TEST_F(configuration_ut, init_op_select_impl_mode_for_all_case1)
{
  Configuration config(AI_CORE_NAME);
  config.ascend_ops_path_ = GetCodeDir();
  config.impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
  map<string, string> options;
  options.emplace("ge.opSelectImplmode", "high_precision_for_all");
  Status status = config.impl_mode_parser_->InitializeFromOptions(options);
  EXPECT_EQ(status, SUCCESS);
  std::string str;
  EXPECT_EQ(config.GetOpImplMode("op_name", "ccc", str), false);
  EXPECT_EQ(config.GetOpImplMode("op_name", "ddd", str), false);
}

TEST_F(configuration_ut, init_op_select_impl_mode_for_all_case2)
{
  Configuration config(AI_CORE_NAME);
  config.ascend_ops_path_ = GetCodeDir();
  config.impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
  map<string, string> options;
  options.emplace("ge.opSelectImplmode", "high_performance_for_all");
  Status status = config.impl_mode_parser_->InitializeFromOptions(options);
  EXPECT_EQ(status, SUCCESS);
  std::string str;
  EXPECT_EQ(config.GetOpImplMode("op_name", "ccc", str), false);
  EXPECT_EQ(config.GetOpImplMode("op_name", "ddd", str), false);
}

TEST_F(configuration_ut, init_op_select_impl_mode_for_all_case3)
{
  Configuration config(AI_CORE_NAME);
  config.ascend_ops_path_ = GetCodeDir() + "/tests/engines/nn_engine/config/";
  config.impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
  map<string, string> options;
  options.emplace("ge.opSelectImplmode", "high_precision_for_all");
  Status status = config.impl_mode_parser_->InitializeFromOptions(options);
  EXPECT_EQ(status, SUCCESS);
  std::string str;
  EXPECT_EQ(config.GetOpImplMode("op_name", "Add", str), true);
  EXPECT_EQ(str, "high_precision");
  EXPECT_EQ(config.GetOpImplMode("op_name", "Mul", str), true);
  EXPECT_EQ(str, "high_precision");
  EXPECT_EQ(config.GetOpImplMode("op_name", "Relu", str), true);
  EXPECT_EQ(str, "high_precision");
}

TEST_F(configuration_ut, init_op_select_impl_mode_for_all_case4)
{
  Configuration config(AI_CORE_NAME);
  config.ascend_ops_path_ = GetCodeDir() + "/tests/engines/nn_engine/config/";
  config.impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
  map<string, string> options;
  options.emplace("ge.opSelectImplmode", "high_performance_for_all");
  Status status = config.impl_mode_parser_->InitializeFromOptions(options);
  EXPECT_EQ(status, SUCCESS);
  std::string str;
  EXPECT_EQ(config.GetOpImplMode("op_name", "Add", str), true);
  EXPECT_EQ(str, "high_performance");
  EXPECT_EQ(config.GetOpImplMode("op_name", "Mul", str), true);
  EXPECT_EQ(str, "high_performance");
  EXPECT_EQ(config.GetOpImplMode("op_name", "Relu", str), true);
  EXPECT_EQ(str, "high_performance");
}

TEST_F(configuration_ut, init_op_select_impl_mode_case1)
{
  Configuration config(AI_CORE_NAME);
  config.ascend_ops_path_ = GetCodeDir();
  config.impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
  map<string, string> options;
  options.emplace("ge.opSelectImplmode", "high_precision");
  Status status = config.impl_mode_parser_->InitializeFromOptions(options);
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(configuration_ut, init_op_select_impl_mode_case2)
{
  Configuration config(AI_CORE_NAME);
  config.ascend_ops_path_ = GetCodeDir();
  config.impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
  map<string, string> options;
  options.emplace("ge.opSelectImplmode", "high_performance");
  Status status = config.impl_mode_parser_->InitializeFromOptions(options);
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(configuration_ut, init_op_select_impl_mode_case3)
{
  Configuration config(AI_CORE_NAME);
  config.ascend_ops_path_ = GetCodeDir() + "/tests/engines/nn_engine/config/";
  config.impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
  map<string, string> options;
  options.emplace("ge.opSelectImplmode", "high_precision");
  options.emplace("ge.optypelistForImplmode", "");
  Status status = config.impl_mode_parser_->InitializeFromOptions(options);
  EXPECT_EQ(status, SUCCESS);
  std::string str;
  EXPECT_EQ(config.GetOpImplMode("op_name", "Add", str), true);
  EXPECT_EQ(str, "high_performance");
  EXPECT_EQ(config.GetOpImplMode("op_name", "Mul", str), true);
  EXPECT_EQ(str, "high_precision");
  EXPECT_EQ(config.GetOpImplMode("op_name", "Relu", str), true);
  EXPECT_EQ(str, "high_precision");
}

TEST_F(configuration_ut, init_op_select_impl_mode_case4)
{
  Configuration config(AI_CORE_NAME);
  config.ascend_ops_path_ = GetCodeDir() + "/tests/engines/nn_engine/config/";
  config.impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
  map<string, string> options;
  options.emplace("ge.opSelectImplmode", "high_performance");
  options.emplace("ge.optypelistForImplmode", "   ");
  Status status = config.impl_mode_parser_->InitializeFromOptions(options);
  EXPECT_EQ(status, SUCCESS);
  std::string str;
  EXPECT_EQ(config.GetOpImplMode("op_name", "Add", str), true);
  EXPECT_EQ(str, "high_performance");
  EXPECT_EQ(config.GetOpImplMode("op_name", "Mul", str), true);
  EXPECT_EQ(str, "high_performance");
  EXPECT_EQ(config.GetOpImplMode("op_name", "Relu", str), true);
  EXPECT_EQ(str, "high_precision");
}

TEST_F(configuration_ut, init_op_select_impl_mode_case5)
{
  Configuration config(AI_CORE_NAME);
  config.ascend_ops_path_ = GetCodeDir() + "/tests/engines/nn_engine/config/";
  config.impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
  map<string, string> options;
  options.emplace("ge.opSelectImplmode", "high_precision");
  options.emplace("ge.optypelistForImplmode", "Relu,Add,Mul");
  Status status = config.impl_mode_parser_->InitializeFromOptions(options);
  EXPECT_EQ(status, SUCCESS);
  std::string str;
  EXPECT_EQ(config.GetOpImplMode("op_name", "Add", str), true);
  EXPECT_EQ(str, "high_precision");
  EXPECT_EQ(config.GetOpImplMode("op_name", "Mul", str), true);
  EXPECT_EQ(str, "high_precision");
  EXPECT_EQ(config.GetOpImplMode("op_name", "Relu", str), true);
  EXPECT_EQ(str, "high_precision");
}

TEST_F(configuration_ut, init_op_select_impl_mode_case6)
{
  Configuration config(AI_CORE_NAME);
  config.ascend_ops_path_ = GetCodeDir() + "/tests/engines/nn_engine/config/";
  config.impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
  map<string, string> options;
  options.emplace("ge.opSelectImplmode", "high_performance");
  options.emplace("ge.optypelistForImplmode", "  Relu ,Add , Mul ");
  Status status = config.impl_mode_parser_->InitializeFromOptions(options);
  EXPECT_EQ(status, SUCCESS);
  std::string str;
  EXPECT_EQ(config.GetOpImplMode("op_name", "Add", str), true);
  EXPECT_EQ(str, "high_performance");
  EXPECT_EQ(config.GetOpImplMode("op_name", "Mul", str), true);
  EXPECT_EQ(str, "high_performance");
  EXPECT_EQ(config.GetOpImplMode("op_name", "Relu", str), true);
  EXPECT_EQ(str, "high_performance");
}

TEST_F(configuration_ut, init_allow_hf32_empty)
{
  Configuration config(AI_CORE_NAME);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::AllowHf32)] = 1;
  config.ascend_ops_path_ = GetCodeDir() + "/tests/engines/nn_engine/config/";
  config.impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
  map<string, string> options;
  options.emplace("ge.opSelectImplmode", "high_performance");
  options.emplace("ge.optypelistForImplmode", "  Relu ,Add , Mul ");
  Status status = config.impl_mode_parser_->InitializeFromOptions(options);
  EXPECT_EQ(status, SUCCESS);
  std::string str;
  EXPECT_EQ(config.GetOpImplMode("op_name", "Add", str), true);
  EXPECT_EQ(str, "high_performance");
  EXPECT_EQ(config.GetOpImplMode("op_name", "Mul", str), true);
  EXPECT_EQ(str, "high_performance");
  EXPECT_EQ(config.GetOpImplMode("op_name", "Relu", str), true);
  EXPECT_EQ(str, "high_performance");
  EXPECT_EQ(config.GetOpImplMode("op_name", "Conv2D", str), true);
  EXPECT_EQ(str, "enable_hi_float_32_execution");
  EXPECT_EQ(config.GetOpImplMode("op_name", "MatMulV2", str), true);
  EXPECT_EQ(str, "enable_float_32_execution");
}

TEST_F(configuration_ut, init_allow_hf32_10)
{
  Configuration config(AI_CORE_NAME);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::AllowHf32)] = 1;
  config.ascend_ops_path_ = GetCodeDir() + "/tests/engines/nn_engine/config/";
  config.impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
  map<string, string> options;
  options.emplace("ge.exec.allow_hf32", "10");
  options.emplace("ge.opSelectImplmode", "high_performance");
  options.emplace("ge.optypelistForImplmode", "  Relu ,Add , Mul ");
  Status status = config.impl_mode_parser_->InitializeFromOptions(options);
  EXPECT_EQ(status, SUCCESS);
  std::string str;
  EXPECT_EQ(config.GetOpImplMode("op_name", "Add", str), true);
  EXPECT_EQ(str, "high_performance");
  EXPECT_EQ(config.GetOpImplMode("op_name", "Mul", str), true);
  EXPECT_EQ(str, "high_performance");
  EXPECT_EQ(config.GetOpImplMode("op_name", "Relu", str), true);
  EXPECT_EQ(str, "high_performance");
  EXPECT_EQ(config.GetOpImplMode("op_name", "Conv2D", str), true);
  EXPECT_EQ(str, "enable_hi_float_32_execution");
  EXPECT_EQ(config.GetOpImplMode("op_name", "MatMulV2", str), true);
  EXPECT_EQ(str, "enable_float_32_execution");
}

TEST_F(configuration_ut, init_allow_hf32_01)
{
  Configuration config(AI_CORE_NAME);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::AllowHf32)] = 1;
  config.ascend_ops_path_ = GetCodeDir() + "/tests/engines/nn_engine/config/";
  config.impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
  map<string, string> options;
  options.emplace("ge.exec.allow_hf32", "01");
  options.emplace("ge.opSelectImplmode", "high_performance");
  options.emplace("ge.optypelistForImplmode", "  Relu ,Add , Mul ");
  Status status = config.impl_mode_parser_->InitializeFromOptions(options);
  EXPECT_EQ(status, SUCCESS);
  std::string str;
  EXPECT_EQ(config.GetOpImplMode("op_name", "Add", str), true);
  EXPECT_EQ(str, "high_performance");
  EXPECT_EQ(config.GetOpImplMode("op_name", "Mul", str), true);
  EXPECT_EQ(str, "high_performance");
  EXPECT_EQ(config.GetOpImplMode("op_name", "Relu", str), true);
  EXPECT_EQ(str, "high_performance");
  EXPECT_EQ(config.GetOpImplMode("op_name", "Conv2D", str), true);
  EXPECT_EQ(str, "enable_float_32_execution");
  EXPECT_EQ(config.GetOpImplMode("op_name", "MatMulV2", str), true);
  EXPECT_EQ(str, "enable_hi_float_32_execution");
}

TEST_F(configuration_ut, init_allow_hf32_all_empty)
{
  Configuration config(AI_CORE_NAME);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::AllowHf32)] = 1;
  config.ascend_ops_path_ = GetCodeDir() + "/tests/engines/nn_engine/config/";
  config.impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
  map<string, string> options;
  Status status = config.impl_mode_parser_->InitializeFromOptions(options);
  EXPECT_EQ(status, SUCCESS);
  std::string str;
  EXPECT_EQ(config.GetOpImplMode("op_name", "Conv2D", str), true);
  EXPECT_EQ(str, "enable_hi_float_32_execution");
  EXPECT_EQ(config.GetOpImplMode("op_name", "MatMulV2", str), true);
  EXPECT_EQ(str, "enable_float_32_execution");
}

TEST_F(configuration_ut, init_allow_hf32_all_invalid)
{
  Configuration config(AI_CORE_NAME);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::AllowHf32)] = 1;
  config.ascend_ops_path_ = GetCodeDir() + "/tests/engines/nn_engine/config/";
  config.impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
  map<string, string> options;
  options.emplace("ge.exec.allow_hf32", "0111");
  Status status = config.impl_mode_parser_->InitializeFromOptions(options);
  EXPECT_EQ(status, FAILED);
}

TEST_F(configuration_ut, init_compress_ratio)
{
  Configuration config(fe::AI_CORE_NAME);
  map<string, string> options;
  string soc_version = "Ascend910B";
  PlatformUtils::Instance().soc_version_ = soc_version;
  options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
  config.Initialize(options);
  config.content_map_.clear();
  Status status = config.LoadConfigFile();
  config.InitCompressRatio();
  auto compress_ratios = config.GetCompressRatios();
  for (auto compress_ratio : compress_ratios) {
    cout << "core num:" << compress_ratio.first << ", ratio:" << compress_ratio.second << endl;
  }
  EXPECT_EQ(compress_ratios.size(), 6);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_FLOAT_EQ(config.GetAICoreCompressRatio(), 0.8);
}

TEST_F(configuration_ut, init_compress_ratio_1)
{
  Configuration config(fe::AI_CORE_NAME);
  map<string, string> options;
  string soc_version = "Ascend910B";
  PlatformUtils::Instance().soc_version_ = soc_version;
  options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
  options.insert(std::make_pair("ge.customizeDtypes", GetCodeDir() + "/tests/engines/nn_engine/ut/stub/custom.cfg"));
  config.Initialize(options);
  config.content_map_.clear();
  config.compress_ratios_.clear();
  config.content_map_.emplace(std::make_pair("multi_core.compress_ratio", "2:0.8:0.88|%:0.7|10:%%|30:0.8"));
  config.InitCompressRatio();
  auto compress_ratios = config.GetCompressRatios();
  for (auto compress_ratio : compress_ratios) {
    cout << "core num:" << compress_ratio.first << ", ratio:" << compress_ratio.second << endl;
  }
  EXPECT_EQ(compress_ratios.size(), 1);
  EXPECT_FLOAT_EQ(config.GetAICoreCompressRatio(), 0.8);
}

TEST_F(configuration_ut, OpCustomizeDtype_001)
{
  Configuration config(fe::AI_CORE_NAME);
  config.cust_dtypes_parser_ = std::make_shared<OpCustDtypesConfigParser>();
  map<string, string> options;
  string soc_version = "Ascend910B";
  options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
  options.insert(std::make_pair("ge.customizeDtypes", GetCodeDir() + "/tests/engines/nn_engine/ut/stub/custom.cfg"));
  ge::GetThreadLocalContext().SetGraphOption(options);
  config.config_parser_map_vec_[static_cast<size_t>(CONFIG_PARSER_PARAM::CustDtypes)]
      .emplace(GetCodeDir() + "/tests/engines/nn_engine/ut/stub/custom.cfg", config.cust_dtypes_parser_);
  Status bres = config.cust_dtypes_parser_->InitializeFromOptions(options);
  EXPECT_EQ(bres, SUCCESS);
  OpCustomizeDtype custom_dtype_0, custom_dtype_1, custom_dtype_2, custom_dtype_3;
  (void)config.GetCustomizeDtypeByOpName("Matmul_name", custom_dtype_0);
  (void)config.GetCustomizeDtypeByOpType("MatMul", custom_dtype_1);
  (void)config.GetCustomizeDtypeByOpName("Test_name", custom_dtype_2);
  (void)config.GetCustomizeDtypeByOpName("Test_name_2", custom_dtype_3);
  EXPECT_EQ(custom_dtype_0.input_dtypes[3], custom_dtype_1.input_dtypes[0]);
  EXPECT_EQ(custom_dtype_0.input_dtypes[1], custom_dtype_1.output_dtypes[0]);
  EXPECT_EQ(custom_dtype_2.input_dtypes[0], custom_dtype_3.output_dtypes[0]);
}

TEST_F(configuration_ut, OpCustomizeDtype_002)
{
  Configuration config(fe::AI_CORE_NAME);
  config.cust_dtypes_parser_ = std::make_shared<OpCustDtypesConfigParser>();
  map<string, string> options;
  string soc_version = "Ascend910B";
  options.insert(std::make_pair("ge.customizeDtypes", GetCodeDir() + "/tests/engines/nn_engine/ut/stub/XXX.cfg"));
  Status bres = config.cust_dtypes_parser_->InitializeFromOptions(options);
  EXPECT_EQ(bres, FAILED);
  options.insert(std::make_pair("ge.customizeDtypes", GetCodeDir() + "/tests/engines/nn_engine/ut/stub/custom_failed.cfg"));
  bres = config.cust_dtypes_parser_->InitializeFromOptions(options);
  EXPECT_EQ(bres, false);
  options.insert(std::make_pair("ge.customizeDtypes", GetCodeDir() + "/tests/engines/nn_engine/ut/stub/custom_fail2.cfg"));
  bres = config.cust_dtypes_parser_->InitializeFromOptions(options);
  EXPECT_EQ(bres, false);
}

TEST_F(configuration_ut, OpCustomizeDtype_003)
{
  Configuration config(fe::AI_CORE_NAME);
  map<string, string> options;
  std::string tmp = "";
  std::string str = "gecustomizeDtypes";
  string soc_version = "Ascend910B";
  PlatformUtils::Instance().soc_version_ = soc_version;
  options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
  options.insert(std::make_pair("ge.customizeDtypes", GetCodeDir() + "/tests/engines/nn_engine/ut/stub/asdad"));
  EXPECT_EQ(config.Initialize(options), fe::FAILED);
}

TEST_F(configuration_ut, OpCustomizeDtype_004)
{
  Configuration config(fe::AI_CORE_NAME);
  config.cust_dtypes_parser_ = std::make_shared<OpCustDtypesConfigParser>();
  map<string, string> options;
  string soc_version = "Ascend910B";
  options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
  options.insert(std::make_pair("ge.customizeDtypes", GetCodeDir() + "/tests/engines/nn_engine/ut/stub/custom_fail2.cfg"));
  Status bres = config.cust_dtypes_parser_->InitializeFromOptions(options);
  EXPECT_EQ(bres, fe::FAILED);
}

TEST_F(configuration_ut, InitOpPrecisionMode_fail)
{
  Configuration config(fe::AI_CORE_NAME);
  config.impl_mode_parser_ = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
  map<string, string> options;
  options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
  options.insert(std::make_pair("ge.opSelectImplmode", "invalid"));
  Status ret = config.impl_mode_parser_->InitializeFromOptions(options);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(configuration_ut, InitOpPrecisionMode_fail_002)
{
  Configuration config(fe::AI_CORE_NAME);
  OpImplModeConfigParserPtr impl_mode_ptr = std::make_shared<OpImplModeConfigParser>(config.ascend_ops_path_);
  Status ret = impl_mode_ptr->InitOpPrecisionMode("", "", "");
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(configuration_ut, OppNewPath_001)
{
  Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
  config.ascend_ops_path_ = GetCodeDir() + "/tests/engines/nn_engine/config/new_opp/";
  config.content_map_.clear();
  config.op_binary_path_map_.clear();
  config.op_store_priority_count_ = 0;
  Status ret = config.LoadOppConfigFile();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(configuration_ut, OppNewPath_002)
{
  Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
  config.ascend_ops_path_ = GetCodeDir() + "/tests/engines/nn_engine/config/new_opp/";
  config.content_map_.clear();
  config.op_binary_path_map_.clear();
  config.op_store_priority_count_ = 0;
  Status ret = config.LoadOppConfigFile();
  EXPECT_EQ(ret, SUCCESS);
  // 3 custom + 8 builtin
  EXPECT_EQ(config.content_map_.size(), 3);
  EXPECT_EQ(config.op_binary_path_map_.size(), 2);
  ret = config.AssembleOpsStoreInfoVector();
  // 1 builtin + 2custom
  EXPECT_EQ(ret, SUCCESS);

  EXPECT_EQ(config.GetBinaryPathMap().size(), 2);

  auto it = config.content_map_.find("op.store.tbe-custom1");
  bool ret1 = (it != config.content_map_.end() && !it->second.empty());
  EXPECT_EQ(ret1, true);
  int64_t tmp = 1;
  int64_t impl = (tmp << 32) | 2;
  std::string str_impl = to_string(impl);
  cout << "str_impl:" << str_impl << endl;
  size_t pos = it->second.find(str_impl);
  EXPECT_NE(pos, std::string::npos);
}

TEST_F(configuration_ut, OppNewPath_003)
{
  Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
  config.ascend_ops_path_ = GetCodeDir() + "/tests/engines/nn_engine/config/new_opp/";
  config.content_map_.clear();
  config.op_binary_path_map_.clear();
  config.op_store_priority_count_ = 0;

  EXPECT_EQ(config.GetBinaryPathMap().size(), 0);
  int64_t imply_value = (2 << 32) | 2;
  int64_t ret_type =  GetMainImplType<int64_t>(imply_value);
  EXPECT_EQ(ret_type, 2);
}

TEST_F(configuration_ut, OppNewPath_004)
{
  Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
  config.ascend_ops_path_ = GetCodeDir() + "/tests/engines/nn_engine/config/none_opp/";
  config.content_map_.clear();
  config.op_binary_path_map_.clear();
  config.op_store_priority_count_ = 0;
  config.LoadOppConfigFile();
  EXPECT_EQ(config.content_map_.size(), 0);
}

TEST_F(configuration_ut, OppNewPath_005_InvalidConfig)
{
  Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
  config.ascend_ops_path_ = GetCodeDir() + "/tests/engines/nn_engine/config/err_opp/";
  config.content_map_.clear();
  config.op_binary_path_map_.clear();
  config.op_store_priority_count_ = 0;
  config.LoadOppConfigFile();
  EXPECT_EQ(config.content_map_.size(), 0);
}

TEST_F(configuration_ut, parse_memory_check_001)
{
  Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
  std::map<std::string, std::string> options1 = {{kOpDebugConfig, GetCodeDir() + "/tests/engines/nn_engine/config/fe_config/op_debug_config1.ini"}};
  config.op_debug_config_parse_->InitializeFromOptions(options1);
  EXPECT_EQ(config.GetMemoryCheckSwitch(), false);
  std::map<std::string, std::string> options2 = {{kOpDebugConfig, " "}};
  config.op_debug_config_parse_->InitializeFromOptions(options2);
  EXPECT_EQ(config.GetMemoryCheckSwitch(), false);
  std::map<std::string, std::string> options3 = {{kOpDebugConfig, GetCodeDir() + "/tests/engines/nn_engine/config/fe_config/op_debug_config2.ini"}};
  config.op_debug_config_parse_->InitializeFromOptions(options3);
  EXPECT_EQ(config.GetMemoryCheckSwitch(), true);
  std::map<std::string, std::string> options4 = {{kOpDebugConfig, GetCodeDir() + "/tests/engines/nn_engine/config/fe_config/op_debug_config3.ini"}};
  bool ret = config.op_debug_config_parse_->InitializeFromOptions(options4);
  EXPECT_EQ(ret, false);
  std::map<std::string, std::string> options5 = {{kOpDebugConfig, GetCodeDir() + "/tests/engines/nn_engine/config/fe_config/op_debug_config4.ini"}};
  ret = config.op_debug_config_parse_->InitializeFromOptions(options5);
  EXPECT_EQ(ret, false);
  std::map<std::string, std::string> options6 = {{kOpDebugConfig, "abc"}};
  ret = config.InitializeConfigParser(options6);
  EXPECT_EQ(ret, true);
}

TEST_F(configuration_ut, parse_first_layer_Test)
{
  Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
  std::map<std::string, std::string> options1 = {{ge::COMPRESSION_OPTIMIZE_CONF, GetCodeDir() + "/tests/engines/nn_engine/config/fe_config/first_config.ini"}};
  bool ret = config.InitFirstLayerQuantization(options1);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(config.enable_first_layer_quantization_, true);
  std::map<std::string, std::string> options2 = {{ge::COMPRESSION_OPTIMIZE_CONF, GetCodeDir() + "/tests/engines/nn_engine/config/fe_config/first_config1.ini"}};
  ret = config.InitFirstLayerQuantization(options2);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(config.enable_first_layer_quantization_, false);
  std::map<std::string, std::string> options3 = {{ge::COMPRESSION_OPTIMIZE_CONF, GetCodeDir() + "/tests/engines/nn_engine/config/fe_config/first_config2.ini"}};
  ret = config.InitFirstLayerQuantization(options3);
  EXPECT_EQ(ret, false);
  std::map<std::string, std::string> options4 = {{ge::COMPRESSION_OPTIMIZE_CONF, GetCodeDir() + "/tests/engines/nn_engine/config/fe_config/first_config3.ini"}};
  ret = config.InitFirstLayerQuantization(options4);
  EXPECT_EQ(ret, false);
  std::map<std::string, std::string> options5 = {{ge::COMPRESSION_OPTIMIZE_CONF, GetCodeDir() + "/tests/engines/nn_engine/config/fe_config/first_config4.ini"}};
  ret = config.InitFirstLayerQuantization(options5);
  EXPECT_EQ(ret, false);
}

TEST_F(configuration_ut, init_env_param_case1) {
  unsetenv("OP_DYNAMIC_COMPILE_STATIC");
  unsetenv("ENABLE_NETWORK_ANALYSIS_DEBUG");
  unsetenv("ENABLE_ACLNN");
  unsetenv("NPU_COLLECT_PATH");
  mmSetEnv("NPU_COLLECT_PATH", "", 1);
  Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
  std::array<string, static_cast<size_t>(ENV_STR_PARAM::EnvStrParamBottom)>
      env_str_param_vec_bac = config.env_str_param_vec_;
  Status status = config.InitLibPath();
  EXPECT_EQ(status, SUCCESS);

  config.InitParamFromEnv();
  status = config.InitAscendOpsPath();
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(config.IsDynamicImplFirst(), true);
  EXPECT_EQ(config.IsEnableNetworkAnalysis(), false);
  EXPECT_EQ(config.ascend_ops_path_.empty(), false);
  EXPECT_EQ(config.IsEnableAclnn(), false);
  EXPECT_EQ(config.env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::NpuCollectPath)], "");

  mmSetEnv("OP_DYNAMIC_COMPILE_STATIC", "False", 1);
  mmSetEnv("ENABLE_NETWORK_ANALYSIS_DEBUG", "0", 1);
  mmSetEnv("ENABLE_NETWORK_ANALYSIS_DEBUG", "0", 1);
  mmSetEnv("ENABLE_ACLNN", "TTT", 1);
  mmSetEnv("NPU_COLLECT_PATH", "123", 1);
  config.InitParamFromEnv();
  status = config.InitAscendOpsPath();
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(config.IsDynamicImplFirst(), false);
  EXPECT_EQ(config.IsEnableNetworkAnalysis(), false);
  EXPECT_EQ(config.IsEnableAclnn(), false);
  EXPECT_EQ(config.env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::NpuCollectPath)], "123");

  mmSetEnv("OP_DYNAMIC_COMPILE_STATIC", "faLse", 1);
  mmSetEnv("ENABLE_NETWORK_ANALYSIS_DEBUG", "1", 1);
  mmSetEnv("ENABLE_ACLNN", "faLse", 1);
  config.InitParamFromEnv();
  EXPECT_EQ(config.IsDynamicImplFirst(), false);
  EXPECT_EQ(config.IsEnableNetworkAnalysis(), true);
  EXPECT_EQ(config.IsEnableAclnn(), false);

  mmSetEnv("OP_DYNAMIC_COMPILE_STATIC", "TTT", 1);
  mmSetEnv("ENABLE_ACLNN", "tRUe", 1);
  config.InitParamFromEnv();
  EXPECT_EQ(config.IsDynamicImplFirst(), true);
  EXPECT_EQ(config.IsEnableAclnn(), true);

  mmSetEnv("OP_DYNAMIC_COMPILE_STATIC", "True", 1);
  mmSetEnv("ENABLE_ACLNN", "True", 1);
  config.InitParamFromEnv();
  EXPECT_EQ(config.IsDynamicImplFirst(), true);
  EXPECT_EQ(config.IsEnableAclnn(), true);

  unsetenv("OP_DYNAMIC_COMPILE_STATIC");
  unsetenv("ENABLE_NETWORK_ANALYSIS_DEBUG");
  unsetenv("ENABLE_ACLNN");
  unsetenv("NPU_COLLECT_PATH");
  config.env_str_param_vec_ = env_str_param_vec_bac;
  config.InitParamFromEnv();
  EXPECT_EQ(config.IsDynamicImplFirst(), true);
  EXPECT_EQ(config.IsEnableAclnn(), false);
}

TEST_F(configuration_ut, load_binary_config_file)
{
  Configuration config(fe::AI_CORE_NAME);
  map<string, string> options;
  options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
  ge::GetThreadLocalContext().SetGraphOption(options);
  string soc_version = "Ascend310";
  PlatformUtils::Instance().soc_version_ = soc_version;
  Status status = config.Initialize(options);
  std::string path = GetCurpath() + "../../../../../../tests/engines/nn_engine/config";
  config.ascend_ops_path_ = path;
  config.InitBinaryConfigFilePath();
  string str = config.GetBinaryConfigFilePath();
  char resoved_path[PATH_MAX] = {0x00};
  realpath(path.c_str(), resoved_path);
  path = resoved_path;

  EXPECT_EQ(str, path + "/built-in/op_impl/ai_core/tbe/kernel/config/Ascend310/op_info_config.json");
}

TEST_F(configuration_ut, init_options_context_case1) {
  Configuration &config = Configuration::Instance(fe::AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend910B";
  config.hardware_info_map_.clear();
  config.config_str_param_vec_.fill("");
  map<string, string> options;
  Status ret = config.InitConfigParamFromOptions(options);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(config.IsEnableSmallChannel(), false);
  EXPECT_EQ(config.GetJitCompileCfg(), JitCompileCfg::CFG_TRUE);
  EXPECT_EQ(config.IsEnableCompressWeight(), false);
  EXPECT_EQ(config.IsEnableSparseMatrixWeight(), false);
  EXPECT_EQ(config.IsEnableReuseMemory(), true);
  EXPECT_EQ(config.IsEnableVirtualType(), false);
  EXPECT_EQ(config.GetBufferOptimize(), EN_UNKNOWN_OPTIMIZE);
  EXPECT_EQ(config.EnableL2Fusion(), true);
  EXPECT_EQ(config.IsQuantDumpable(), false);
  EXPECT_EQ(config.IsEnableQuantBiasOptimize(), true);

  EXPECT_EQ(config.GetHardwareInfo().size(), 0);
  EXPECT_EQ(config.GetLicenseFusionStr(), "ALL");
  EXPECT_EQ(config.GetLicenseFusionDetailInfo().size(), 0);

  EXPECT_EQ(FEContextUtils::GetFusionSwitchFilePath(), "");

  map<string, string> context_map;
  ge::GetThreadLocalContext().SetGraphOption(context_map);
  ret = config.InitConfigParamFromContext();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(config.IsEnableSmallChannel(), false);
  EXPECT_EQ(config.GetJitCompileCfg(), JitCompileCfg::CFG_TRUE);
  EXPECT_EQ(config.IsEnableCompressWeight(), false);
  EXPECT_EQ(config.IsEnableSparseMatrixWeight(), false);
  EXPECT_EQ(config.IsEnableReuseMemory(), true);
  EXPECT_EQ(config.IsEnableVirtualType(), false);
  EXPECT_EQ(config.GetBufferOptimize(), EN_UNKNOWN_OPTIMIZE);
  EXPECT_EQ(config.EnableL2Fusion(), true);
  EXPECT_EQ(config.IsQuantDumpable(), false);
  EXPECT_EQ(config.IsEnableQuantBiasOptimize(), true);

  EXPECT_EQ(config.GetHardwareInfo().size(), 0);
  EXPECT_EQ(config.GetLicenseFusionStr(), "ALL");
  EXPECT_EQ(config.GetLicenseFusionDetailInfo().size(), 0);

  EXPECT_EQ(FEContextUtils::GetFusionSwitchFilePath(), "");
}

TEST_F(configuration_ut, init_options_context_case2) {
  Configuration &config = Configuration::Instance(fe::AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend910B";
  map<string, string> options;
  options.emplace(ge::ENABLE_SMALL_CHANNEL, "1");
  options.emplace(ge::JIT_COMPILE, "0");
  options.emplace(ge::ENABLE_COMPRESS_WEIGHT, "1");
  options.emplace(ge::ENABLE_SPARSE_MATRIX_WEIGHT, "1");
  options.emplace(ge::OPTION_EXEC_DISABLE_REUSED_MEMORY, "1");
  options.emplace(ge::BUFFER_OPTIMIZE, "l2_optimize");
  options.emplace(ge::VIRTUAL_TYPE, "1");
  options.emplace(ge::QUANT_DUMPABLE, "1");
  options.emplace(ge::FUSION_SWITCH_FILE, "abc");
  options.emplace("opt_module.fe", "3:4");
  options.emplace("ge.hardwareInfo", "ai_core_cnt:16;vector_core_cnt:10");
  options.emplace("ge.experiment.quant_bias_optimize", "1");
  PlatformUtils::Instance().vir_type_list_ = {};
  Status ret = config.InitConfigParamFromOptions(options);
  EXPECT_EQ(ret, FAILED);
  PlatformUtils::Instance().vir_type_list_ = {2,4,8,16};
  ret = config.InitConfigParamFromOptions(options);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(config.IsEnableSmallChannel(), true);
  EXPECT_EQ(config.GetJitCompileCfg(), JitCompileCfg::CFG_FALSE);
  EXPECT_EQ(config.IsEnableCompressWeight(), true);
  EXPECT_EQ(config.IsEnableSparseMatrixWeight(), true);
  EXPECT_EQ(config.IsEnableReuseMemory(), false);
  EXPECT_EQ(config.IsEnableVirtualType(), true);
  EXPECT_EQ(config.GetBufferOptimize(), EN_L2_OPTIMIZE);
  EXPECT_EQ(config.EnableL2Fusion(), true);
  EXPECT_EQ(config.IsQuantDumpable(), true);
  EXPECT_EQ(config.IsEnableQuantBiasOptimize(), true);

  std::map<string, string> hard_ware_info_1 = config.GetHardwareInfo();
  EXPECT_EQ(hard_ware_info_1.size(), 2);
  EXPECT_EQ(hard_ware_info_1["ai_core_cnt"], "16");
  EXPECT_EQ(hard_ware_info_1["vector_core_cnt"], "10");
  EXPECT_EQ(config.GetLicenseFusionStr(), "3:4");
  EXPECT_EQ(config.GetLicenseFusionDetailInfo().size(), 2);
  std::set<std::string> ret_vec_1 = {"MatMulBiasAddFusionPass", "OneHotFusionPass"};
  EXPECT_EQ(config.GetLicenseFusionDetailInfo(), ret_vec_1);

  EXPECT_EQ(FEContextUtils::GetFusionSwitchFilePath(), "");

  map<string, string> context_map;
  ge::GetThreadLocalContext().SetGraphOption(context_map);
  ret = config.InitConfigParamFromContext();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(config.IsEnableSmallChannel(), true);
  EXPECT_EQ(config.GetJitCompileCfg(), JitCompileCfg::CFG_FALSE);
  EXPECT_EQ(config.IsEnableCompressWeight(), true);
  EXPECT_EQ(config.IsEnableSparseMatrixWeight(), true);
  EXPECT_EQ(config.IsEnableReuseMemory(), false);
  EXPECT_EQ(config.IsEnableVirtualType(), true);
  EXPECT_EQ(config.GetBufferOptimize(), EN_L2_OPTIMIZE);
  EXPECT_EQ(config.EnableL2Fusion(), true);
  EXPECT_EQ(config.IsQuantDumpable(), true);
  EXPECT_EQ(config.IsEnableQuantBiasOptimize(), true);

  std::map<string, string> hard_ware_info_2 = config.GetHardwareInfo();
  EXPECT_EQ(hard_ware_info_2.size(), 2);
  EXPECT_EQ(hard_ware_info_2["ai_core_cnt"], "16");
  EXPECT_EQ(hard_ware_info_2["vector_core_cnt"], "10");
  EXPECT_EQ(config.GetLicenseFusionStr(), "3:4");
  EXPECT_EQ(config.GetLicenseFusionDetailInfo().size(), 2);
  std::set<std::string> ret_vec_2 = {"MatMulBiasAddFusionPass", "OneHotFusionPass"};
  EXPECT_EQ(config.GetLicenseFusionDetailInfo(), ret_vec_2);

  EXPECT_EQ(FEContextUtils::GetFusionSwitchFilePath(), "");
}

TEST_F(configuration_ut, init_options_context_case3) {
  Configuration &config = Configuration::Instance(fe::AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend910B";
  PlatformUtils::Instance().vir_type_list_ = {2,4,8,16};
  config.hardware_info_map_.clear();
  config.config_str_param_vec_.fill("");
  map<string, string> options;
  Status ret = config.InitConfigParamFromOptions(options);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(config.IsEnableSmallChannel(), false);
  EXPECT_EQ(config.GetJitCompileCfg(), JitCompileCfg::CFG_TRUE);
  EXPECT_EQ(config.IsEnableCompressWeight(), false);
  EXPECT_EQ(config.IsEnableSparseMatrixWeight(), false);
  EXPECT_EQ(config.IsEnableReuseMemory(), true);
  EXPECT_EQ(config.IsEnableVirtualType(), false);
  EXPECT_EQ(config.GetBufferOptimize(), EN_UNKNOWN_OPTIMIZE);
  EXPECT_EQ(config.EnableL2Fusion(), true);
  EXPECT_EQ(config.IsQuantDumpable(), false);
  EXPECT_EQ(config.IsEnableQuantBiasOptimize(), true);

  EXPECT_EQ(config.GetHardwareInfo().size(), 0);
  EXPECT_EQ(config.GetLicenseFusionStr(), "ALL");
  EXPECT_EQ(config.GetLicenseFusionDetailInfo().size(), 0);

  map<string, string> context_map;
  context_map.emplace(ge::ENABLE_SMALL_CHANNEL, "1");
  context_map.emplace(ge::JIT_COMPILE, "0");
  context_map.emplace(ge::ENABLE_COMPRESS_WEIGHT, "1");
  context_map.emplace(ge::ENABLE_SPARSE_MATRIX_WEIGHT, "1");
  context_map.emplace(ge::OPTION_EXEC_DISABLE_REUSED_MEMORY, "1");
  context_map.emplace(ge::BUFFER_OPTIMIZE, "l2_optimize");
  context_map.emplace(ge::VIRTUAL_TYPE, "1");
  context_map.emplace(ge::QUANT_DUMPABLE, "1");
  context_map.emplace(ge::FUSION_SWITCH_FILE, "abc");
  context_map.emplace("opt_module.fe", "3:4");
  context_map.emplace("ge.hardwareInfo", "ai_core_cnt:16;vector_core_cnt:10");
  context_map.emplace("ge.experiment.quant_bias_optimize", "0");
  ge::GetThreadLocalContext().SetGraphOption(context_map);
  ret = config.InitConfigParamFromContext();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(config.IsEnableSmallChannel(), true);
  EXPECT_EQ(config.GetJitCompileCfg(), JitCompileCfg::CFG_FALSE);
  EXPECT_EQ(config.IsEnableCompressWeight(), true);
  EXPECT_EQ(config.IsEnableSparseMatrixWeight(), true);
  EXPECT_EQ(config.IsEnableReuseMemory(), false);
  EXPECT_EQ(config.IsEnableVirtualType(), true);
  EXPECT_EQ(config.GetBufferOptimize(), EN_L2_OPTIMIZE);
  EXPECT_EQ(config.EnableL2Fusion(), true);
  EXPECT_EQ(config.IsQuantDumpable(), true);
  EXPECT_EQ(config.IsEnableQuantBiasOptimize(), false);

  std::map<string, string> hard_ware_info_2 = config.GetHardwareInfo();
  EXPECT_EQ(hard_ware_info_2.size(), 2);
  EXPECT_EQ(hard_ware_info_2["ai_core_cnt"], "16");
  EXPECT_EQ(hard_ware_info_2["vector_core_cnt"], "10");
  EXPECT_EQ(config.GetLicenseFusionStr(), "3:4");
  EXPECT_EQ(config.GetLicenseFusionDetailInfo().size(), 2);
  std::set<std::string> ret_vec_2 = {"MatMulBiasAddFusionPass", "OneHotFusionPass"};
  EXPECT_EQ(config.GetLicenseFusionDetailInfo(), ret_vec_2);

  EXPECT_EQ(FEContextUtils::GetFusionSwitchFilePath(), "abc");
}

TEST_F(configuration_ut, init_options_context_case4) {
  Configuration &config = Configuration::Instance(fe::AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend910B";
  PlatformUtils::Instance().vir_type_list_ = {2,4,8,16};
  map<string, string> options;
  options.emplace(ge::ENABLE_SMALL_CHANNEL, "1");
  options.emplace(ge::JIT_COMPILE, "0");
  options.emplace(ge::ENABLE_COMPRESS_WEIGHT, "1");
  options.emplace(ge::ENABLE_SPARSE_MATRIX_WEIGHT, "1");
  options.emplace(ge::OPTION_EXEC_DISABLE_REUSED_MEMORY, "0");
  options.emplace(ge::BUFFER_OPTIMIZE, "off_optimize");
  options.emplace(ge::VIRTUAL_TYPE, "1");
  options.emplace(ge::QUANT_DUMPABLE, "1");
  options.emplace(ge::FUSION_SWITCH_FILE, "abc");
  options.emplace("opt_module.fe", "3:4");
  options.emplace("ge.hardwareInfo", "ai_core_cnt:16;vector_core_cnt:10");
  options.emplace("ge.experiment.quant_bias_optimize", "0");
  Status ret = config.InitConfigParamFromOptions(options);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(config.IsEnableSmallChannel(), true);
  EXPECT_EQ(config.GetJitCompileCfg(), JitCompileCfg::CFG_FALSE);
  EXPECT_EQ(config.IsEnableCompressWeight(), true);
  EXPECT_EQ(config.IsEnableSparseMatrixWeight(), true);
  EXPECT_EQ(config.IsEnableReuseMemory(), true);
  EXPECT_EQ(config.IsEnableVirtualType(), true);
  EXPECT_EQ(config.GetBufferOptimize(), EN_OFF_OPTIMIZE);
  EXPECT_EQ(config.EnableL2Fusion(), false);
  EXPECT_EQ(config.IsQuantDumpable(), true);
  EXPECT_EQ(config.IsEnableQuantBiasOptimize(), false);

  std::map<string, string> hard_ware_info_1 = config.GetHardwareInfo();
  EXPECT_EQ(hard_ware_info_1.size(), 2);
  EXPECT_EQ(hard_ware_info_1["ai_core_cnt"], "16");
  EXPECT_EQ(hard_ware_info_1["vector_core_cnt"], "10");
  EXPECT_EQ(config.GetLicenseFusionStr(), "3:4");
  EXPECT_EQ(config.GetLicenseFusionDetailInfo().size(), 2);
  std::set<std::string> ret_vec_1 = {"MatMulBiasAddFusionPass", "OneHotFusionPass"};
  EXPECT_EQ(config.GetLicenseFusionDetailInfo(), ret_vec_1);

  EXPECT_EQ(FEContextUtils::GetFusionSwitchFilePath(), "");

  PlatformUtils::Instance().soc_version_ = "Ascend910B";
  map<string, string> context_map;
  context_map.emplace(ge::OPTION_EXEC_DISABLE_REUSED_MEMORY, "1");
  context_map.emplace(ge::BUFFER_OPTIMIZE, "l1_optimize");
  context_map.emplace(ge::VIRTUAL_TYPE, "0");
  context_map.emplace(ge::QUANT_DUMPABLE, "0");
  context_map.emplace(ge::FUSION_SWITCH_FILE, "bcd");
  context_map.emplace("opt_module.fe", "");
  context_map.emplace("ge.hardwareInfo", "ai_core_cnt:32;vector_core_cnt:5");
  context_map.emplace("ge.experiment.quant_bias_optimize", "1");
  ge::GetThreadLocalContext().SetGraphOption(context_map);
  ret = config.InitConfigParamFromContext();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(config.IsEnableSmallChannel(), true);
  EXPECT_EQ(config.GetJitCompileCfg(), JitCompileCfg::CFG_FALSE);
  EXPECT_EQ(config.IsEnableCompressWeight(), true);
  EXPECT_EQ(config.IsEnableSparseMatrixWeight(), true);
  EXPECT_EQ(config.IsEnableReuseMemory(), false);
  EXPECT_EQ(config.IsEnableVirtualType(), false);
  EXPECT_EQ(config.GetBufferOptimize(), EN_L1_OPTIMIZE);
  EXPECT_EQ(config.EnableL2Fusion(), true);
  EXPECT_EQ(config.IsQuantDumpable(), false);
  EXPECT_EQ(config.IsEnableQuantBiasOptimize(), true);

  std::map<string, string> hard_ware_info_2 = config.GetHardwareInfo();
  EXPECT_EQ(hard_ware_info_2.size(), 2);
  EXPECT_EQ(hard_ware_info_2["ai_core_cnt"], "32");
  EXPECT_EQ(hard_ware_info_2["vector_core_cnt"], "5");
  EXPECT_EQ(config.GetLicenseFusionStr(), "");
  EXPECT_EQ(config.GetLicenseFusionDetailInfo().size(), 0);

  EXPECT_EQ(FEContextUtils::GetFusionSwitchFilePath(), "bcd");
}

TEST_F(configuration_ut, init_options_context_case5) {
  Configuration &config = Configuration::Instance(fe::AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend910B";
  map<string, string> options;
  options.emplace(ge::ENABLE_SMALL_CHANNEL, "12#$");
  Status ret = config.InitConfigParamFromOptions(options);
  EXPECT_EQ(ret, SUCCESS);

  ge::GetThreadLocalContext().SetGraphOption(options);
  ret = config.InitConfigParamFromContext();
  EXPECT_EQ(ret, FAILED);
}

TEST_F(configuration_ut, init_options_context_case6) {
  Configuration &config = Configuration::Instance(fe::AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend910B";
  map<string, string> options;
  options.emplace("ge.experiment.quant_bias_optimize", "xxx");
  Status ret = config.InitConfigParamFromOptions(options);
  EXPECT_EQ(ret, SUCCESS);

  ge::GetThreadLocalContext().SetGraphOption(options);
  ret = config.InitConfigParamFromContext();
  EXPECT_EQ(ret, FAILED);
}

TEST_F(configuration_ut, refresh_parameters_01) {
  map<string, string> options;
  options.emplace(ge::OP_SELECT_IMPL_MODE, "high_precision");
  options.emplace(ge::OPTYPELIST_FOR_IMPLMODE, "Add, Mul");
  ge::GetThreadLocalContext().SetGraphOption(options);
  Status ret = Configuration::Instance(fe::AI_CORE_NAME).RefreshParameters();
  EXPECT_EQ(ret, SUCCESS);
  std::string op_precision_mode;
  EXPECT_EQ(Configuration::Instance(fe::AI_CORE_NAME).GetOpImplMode("op_name", "Sub", op_precision_mode), false);

  EXPECT_EQ(Configuration::Instance(fe::AI_CORE_NAME).GetOpImplMode("op_name", "Add", op_precision_mode), true);
  EXPECT_EQ(op_precision_mode, "high_precision");
  EXPECT_EQ(Configuration::Instance(fe::AI_CORE_NAME).GetOpImplMode("op_name", "Add", op_precision_mode), true);
  EXPECT_EQ(op_precision_mode, "high_precision");
}

TEST_F(configuration_ut, refresh_parameters_02) {
  PlatformUtils::Instance().soc_version_ = "Ascend910B";
  map<string, string> options;
  options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_BF16);
  options.emplace(ge::BUFFER_OPTIMIZE, "l1_optimize");
  ge::GetThreadLocalContext().SetGraphOption(options);
  Status ret = Configuration::Instance(fe::AI_CORE_NAME).RefreshParameters();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(Configuration::Instance(fe::AI_CORE_NAME).GetBufferOptimize(), EN_L1_OPTIMIZE);
  EXPECT_EQ(Configuration::Instance(fe::AI_CORE_NAME).EnableL2Fusion(), true);
}

TEST_F(configuration_ut, refresh_parameters_04) {
  map<string, string> options;
  options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_BF16);
  options.emplace("opt_module.fe", "3:4");
  ge::GetThreadLocalContext().SetGraphOption(options);
  Status ret = Configuration::Instance(fe::AI_CORE_NAME).RefreshParameters();
  EXPECT_EQ(ret, SUCCESS);
  std::set<std::string> ret_vec = {"MatMulBiasAddFusionPass", "OneHotFusionPass"};
  EXPECT_EQ(Configuration::Instance(fe::AI_CORE_NAME).license_fusion_detail_value_, ret_vec);
}

TEST_F(configuration_ut, refresh_parameters_05) {
  map<string, string> options;
  options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_BF16);
  options.emplace(ge::ENABLE_SMALL_CHANNEL, "1");
  ge::GetThreadLocalContext().SetGraphOption(options);
  Status ret = Configuration::Instance(fe::AI_CORE_NAME).RefreshParameters();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(Configuration::Instance(fe::AI_CORE_NAME).IsEnableSmallChannel(), true);
  options[ge::ENABLE_SMALL_CHANNEL] = "0";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ret = Configuration::Instance(fe::AI_CORE_NAME).RefreshParameters();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(Configuration::Instance(fe::AI_CORE_NAME).IsEnableSmallChannel(), false);
}

TEST_F(configuration_ut, refresh_parameters_06) {
  map<string, string> options;
  options.emplace(ge::OP_DEBUG_OPTION, "oom, -g");
  ge::GetThreadLocalContext().SetGraphOption(options);
  Status ret = Configuration::Instance(fe::AI_CORE_NAME).InitDebugConfigParam();
  EXPECT_EQ(ret, SUCCESS);
  string op_debug_cfg_str = Configuration::Instance(fe::AI_CORE_NAME).GetOpDebugConfig();
  EXPECT_EQ(op_debug_cfg_str, "oom,-g");
}

TEST_F(configuration_ut, modify_mixlist_case1) {
  Configuration &config = Configuration::Instance(AI_CORE_NAME);
  EXPECT_EQ(config.GetPrecisionPolicy("A", GRAY), GRAY);
  EXPECT_EQ(config.GetPrecisionPolicy("B", GRAY), GRAY);
  EXPECT_EQ(config.GetPrecisionPolicy("C", GRAY), GRAY);
  EXPECT_EQ(config.GetPrecisionPolicy("D", GRAY), GRAY);
  EXPECT_EQ(config.GetPrecisionPolicy("E", GRAY), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("F", GRAY), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("G", GRAY), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("H", GRAY), GRAY);
  EXPECT_EQ(config.GetPrecisionPolicy("I", GRAY), WHITE);
  EXPECT_EQ(config.GetPrecisionPolicy("J", GRAY), WHITE);
  EXPECT_EQ(config.GetPrecisionPolicy("K", GRAY), WHITE);
}

TEST_F(configuration_ut, modify_mixlist_case2) {
  Configuration &config = Configuration::Instance(AI_CORE_NAME);
  EXPECT_EQ(config.GetPrecisionPolicy("A", BLACK), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("B", BLACK), GRAY);
  EXPECT_EQ(config.GetPrecisionPolicy("C", BLACK), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("D", BLACK), GRAY);
  EXPECT_EQ(config.GetPrecisionPolicy("E", BLACK), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("F", BLACK), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("G", BLACK), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("H", BLACK), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("I", BLACK), WHITE);
  EXPECT_EQ(config.GetPrecisionPolicy("J", BLACK), WHITE);
  EXPECT_EQ(config.GetPrecisionPolicy("K", BLACK), WHITE);
}

TEST_F(configuration_ut, modify_mixlist_case3) {
  Configuration &config = Configuration::Instance(AI_CORE_NAME);
  EXPECT_EQ(config.GetPrecisionPolicy("A", WHITE), WHITE);
  EXPECT_EQ(config.GetPrecisionPolicy("B", WHITE), WHITE);
  EXPECT_EQ(config.GetPrecisionPolicy("C", WHITE), GRAY);
  EXPECT_EQ(config.GetPrecisionPolicy("D", WHITE), WHITE);
  EXPECT_EQ(config.GetPrecisionPolicy("E", WHITE), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("F", WHITE), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("G", WHITE), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("H", WHITE), GRAY);
  EXPECT_EQ(config.GetPrecisionPolicy("I", WHITE), WHITE);
  EXPECT_EQ(config.GetPrecisionPolicy("J", WHITE), WHITE);
  EXPECT_EQ(config.GetPrecisionPolicy("K", WHITE), WHITE);
}

TEST_F(configuration_ut, modify_mixlist_case4) {
  Configuration &config = Configuration::Instance(AI_CORE_NAME);
  map<string, string> options;
  options.emplace(ge::MODIFY_MIXLIST, GetCodeDir() + "/tests/engines/nn_engine/config/mix_list/op_mix_list2.json");
  ge::GetThreadLocalContext().SetGraphOption(options);
  string session_graph_id;
  Status ret = config.RefreshParameters();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(config.GetPrecisionPolicy("A", GRAY), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("A", BLACK), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("A", WHITE), BLACK);

  options[ge::MODIFY_MIXLIST] = GetCodeDir() + "/tests/engines/nn_engine/config/mix_list/op_mix_list3.json";
  ge::GetThreadLocalContext().SetGraphOption(options);
  session_graph_id = "0_1";
  ret = config.RefreshParameters();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(config.GetPrecisionPolicy("A", GRAY), WHITE);
  EXPECT_EQ(config.GetPrecisionPolicy("A", BLACK), WHITE);
  EXPECT_EQ(config.GetPrecisionPolicy("A", WHITE), WHITE);

  options[ge::MODIFY_MIXLIST] = GetCodeDir() + "/tests/engines/nn_engine/config/mix_list/op_mix_list1.json";
  ge::GetThreadLocalContext().SetGraphOption(options);
  session_graph_id.clear();
  ret = config.RefreshParameters();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(config.GetPrecisionPolicy("A", GRAY), GRAY);
  EXPECT_EQ(config.GetPrecisionPolicy("A", BLACK), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("A", WHITE), WHITE);

  options[ge::MODIFY_MIXLIST] = GetCodeDir() + "/tests/engines/nn_engine/config/mix_list/op_mix_list2.json";
  ge::GetThreadLocalContext().SetGraphOption(options);
  session_graph_id = "0_2";
  ret = config.RefreshParameters();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(config.GetPrecisionPolicy("A", GRAY), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("A", BLACK), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("A", WHITE), BLACK);

  options[ge::MODIFY_MIXLIST] = GetCodeDir() + "/tests/engines/nn_engine/config/mix_list/op_mix_list_invalid.json";
  ge::GetThreadLocalContext().SetGraphOption(options);
  session_graph_id = "0_2";
  ret = config.RefreshParameters();
  EXPECT_EQ(ret, FAILED);
}

TEST_F(configuration_ut, modify_mixlist_case5) {
  Configuration &config = Configuration::Instance(AI_CORE_NAME);
  if (config.mix_list_parser_ == nullptr) {
    config.mix_list_parser_ = std::make_shared<ModifyMixlistConfigParser>();
  }
  map<string, string> options;
  Status ret = config.mix_list_parser_->InitializeFromOptions(options);
  EXPECT_EQ(ret, SUCCESS);

  options.emplace(ge::MODIFY_MIXLIST, GetCodeDir() + "/tests/engines/nn_engine/config/mix_list/op_mix_list4.json");
  ge::GetThreadLocalContext().SetGraphOption(options);
  string session_graph_id;
  ret = config.RefreshParameters();
  EXPECT_EQ(ret, FAILED);

  options[ge::MODIFY_MIXLIST] = GetCodeDir() + "/tests/engines/nn_engine/config/mix_list/op_mix_list4.json";
  ge::GetThreadLocalContext().SetGraphOption(options);
  session_graph_id = "0_1";
  ret = config.RefreshParameters();
  EXPECT_EQ(ret, FAILED);
}

TEST_F(configuration_ut, modify_mixlist_case6) {
  Configuration &config = Configuration::Instance(AI_CORE_NAME);
  if (config.mix_list_parser_ == nullptr) {
    config.mix_list_parser_ = std::make_shared<ModifyMixlistConfigParser>();
  }
  map<string, string> options;
  Status ret = config.mix_list_parser_->InitializeFromOptions(options);
  EXPECT_EQ(ret, SUCCESS);

  options.emplace(ge::MODIFY_MIXLIST, GetCodeDir() + "/tests/engines/nn_engine/config/mix_list/op_mix_list5.json");
  ge::GetThreadLocalContext().SetGraphOption(options);
  string session_graph_id;
  ret = config.RefreshParameters();
  EXPECT_EQ(ret, FAILED);

  options[ge::MODIFY_MIXLIST] = GetCodeDir() + "/tests/engines/nn_engine/config/mix_list/op_mix_list5.json";
  ge::GetThreadLocalContext().SetGraphOption(options);
  session_graph_id = "0_1";
  ret = config.RefreshParameters();
  EXPECT_EQ(ret, FAILED);
}

TEST_F(configuration_ut, custom_opp_path_001) {
  Configuration config(fe::AI_CORE_NAME);
  std::string path = GetCurpath() + "../../../../../../tests/engines/nn_engine/config/customize/:";
  mmSetEnv("ASCEND_CUSTOM_OPP_PATH", path.c_str(), MMPA_MAX_PATH);

  map<string, string> options;
  options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
  ge::GetThreadLocalContext().SetGraphOption(options);
  string soc_version = "Ascend910B";
  PlatformUtils::Instance().soc_version_ = soc_version;
  Status status = config.Initialize(options);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(config.ops_store_info_vector_.size(), 2);
  status = config.Finalize();
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(path, config.env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::AscendCustomOppPath)]);
  unsetenv("ASCEND_CUSTOM_OPP_PATH");
}

TEST_F(configuration_ut, precision_mode_case1) {
  ge::GetThreadLocalContext().graph_options_.clear();
  ge::GetThreadLocalContext().session_options_.clear();
  ge::GetThreadLocalContext().global_options_.clear();
  map<string, string> options;
  ge::GetThreadLocalContext().SetGraphOption(options);
  string precision_mode = FEContextUtils::GetPrecisionMode();
  EXPECT_EQ(precision_mode.empty(), true);
  Status ret = FEContextUtils::GetPrecisionMode(precision_mode);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(precision_mode, "allow_fp32_to_fp16");

  options.emplace(ge::PRECISION_MODE, "None");
  ge::GetThreadLocalContext().SetGraphOption(options);
  precision_mode = FEContextUtils::GetPrecisionMode();
  EXPECT_EQ(precision_mode, "None");
  ret = FEContextUtils::GetPrecisionMode(precision_mode);
  EXPECT_EQ(ret, FAILED);

  options[ge::PRECISION_MODE] = "force_fp32";
  ge::GetThreadLocalContext().SetGraphOption(options);
  precision_mode = FEContextUtils::GetPrecisionMode();
  EXPECT_EQ(precision_mode, "force_fp32");
  ret = FEContextUtils::GetPrecisionMode(precision_mode);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(precision_mode, "force_fp32");

  options[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION_BF16;
  ge::GetThreadLocalContext().SetGraphOption(options);
  precision_mode = FEContextUtils::GetPrecisionMode();
  EXPECT_EQ(precision_mode, ALLOW_MIX_PRECISION_BF16);
  ret = FEContextUtils::GetPrecisionMode(precision_mode);
  EXPECT_EQ(ret, SUCCESS);

  options[ge::PRECISION_MODE] = "None";
  ge::GetThreadLocalContext().SetGraphOption(options);
  fe::PrecisionMode precision_mode_enum = fe::PrecisionMode::ENUM_ALLOW_FP32_TO_BF16;
  ret = FEContextUtils::GetPrecisionMode(precision_mode_enum);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(configuration_ut, precision_mode_hif8_case) {
  ge::GetThreadLocalContext().graph_options_.clear();
  ge::GetThreadLocalContext().session_options_.clear();
  ge::GetThreadLocalContext().global_options_.clear();
  map<string, string> options;

  options[ge::PRECISION_MODE_V2] = "cube_hif8";
  ge::GetThreadLocalContext().SetGraphOption(options);
  string precision_mode = FEContextUtils::GetPrecisionMode();
  EXPECT_EQ(precision_mode, "cube_hif8");
  Status ret = FEContextUtils::GetPrecisionMode(precision_mode);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(precision_mode, "cube_hif8");

  options[ge::PRECISION_MODE_V2] = "mixed_hif8";
  ge::GetThreadLocalContext().SetGraphOption(options);
  precision_mode = FEContextUtils::GetPrecisionMode();
  EXPECT_EQ(precision_mode, "mixed_hif8");
  ret = FEContextUtils::GetPrecisionMode(precision_mode);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(precision_mode, "mixed_hif8");

  options[ge::PRECISION_MODE] = "force_fp16";
  options[ge::PRECISION_MODE_V2] = "cube_hif8";
  ge::GetThreadLocalContext().SetGraphOption(options);
  precision_mode = FEContextUtils::GetPrecisionMode();
  EXPECT_EQ(precision_mode, "force_fp16");
  ret = FEContextUtils::GetPrecisionMode(precision_mode);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(precision_mode, "force_fp16");
}

TEST_F(configuration_ut, session_graph_config_params_01) {
  Configuration config(AI_CORE_NAME);
  map<string, string> options;
  string session_graph_id = "0_1";
  options.emplace(ge::OPTION_EXEC_DISABLE_REUSED_MEMORY, "1");
  options.emplace(ge::CUSTOMIZE_DTYPES, GetCodeDir() + "/tests/engines/nn_engine/ut/stub/custom.cfg");
  options.emplace(ge::MODIFY_MIXLIST, GetCodeDir() + "/tests/engines/nn_engine/config/mix_list/op_mix_list1.json");
  config.is_init_ = false;
  Status ret = config.Initialize(options);
  ge::GetThreadLocalContext().SetGraphOption(options);
  ret = config.RefreshParameters();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(config.IsEnableReuseMemory(), 0);
  OpCustomizeDtype custom_dtype_0, custom_dtype_1, custom_dtype_2, custom_dtype_3;
  (void)config.GetCustomizeDtypeByOpName("Matmul_name", custom_dtype_0);
  (void)config.GetCustomizeDtypeByOpType("MatMul", custom_dtype_1);
  (void)config.GetCustomizeDtypeByOpName("Test_name", custom_dtype_2);
  (void)config.GetCustomizeDtypeByOpName("Test_name_2", custom_dtype_3);
  EXPECT_EQ(custom_dtype_0.input_dtypes[1], ge::DataType::DT_FLOAT);
  EXPECT_EQ(custom_dtype_0.output_dtypes[0], ge::DataType::DT_FLOAT);
  EXPECT_EQ(custom_dtype_1.input_dtypes[0], ge::DataType::DT_FLOAT16);
  EXPECT_EQ(custom_dtype_1.output_dtypes[0], ge::DataType::DT_FLOAT);
  EXPECT_EQ(custom_dtype_2.input_dtypes[0], ge::DataType::DT_FLOAT);
  EXPECT_EQ(custom_dtype_3.output_dtypes[1], ge::DataType::DT_UNDEFINED);
  EXPECT_EQ(config.GetPrecisionPolicy("A", GRAY), GRAY);
  EXPECT_EQ(config.GetPrecisionPolicy("A", BLACK), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("A", WHITE), WHITE);

  session_graph_id = "0_2";
  options[ge::OPTION_EXEC_DISABLE_REUSED_MEMORY] = "0";
  options[ge::CUSTOMIZE_DTYPES] = GetCodeDir() + "/tests/engines/nn_engine/ut/stub/custom2.cfg";
  options[ge::MODIFY_MIXLIST] = GetCodeDir() + "/tests/engines/nn_engine/config/mix_list/op_mix_list2.json";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ret = config.RefreshParameters();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(config.IsEnableReuseMemory(), 1);
  (void)config.GetCustomizeDtypeByOpName("Matmul_name", custom_dtype_0);
  (void)config.GetCustomizeDtypeByOpType("MatMul", custom_dtype_1);
  (void)config.GetCustomizeDtypeByOpName("Test_name", custom_dtype_2);
  (void)config.GetCustomizeDtypeByOpName("Test_name_2", custom_dtype_3);
  EXPECT_EQ(custom_dtype_0.input_dtypes[1], ge::DataType::DT_FLOAT16);
  EXPECT_EQ(custom_dtype_0.output_dtypes[0], ge::DataType::DT_FLOAT16);
  EXPECT_EQ(custom_dtype_1.input_dtypes[0], ge::DataType::DT_FLOAT);
  EXPECT_EQ(custom_dtype_1.output_dtypes[0], ge::DataType::DT_FLOAT16);
  EXPECT_EQ(custom_dtype_2.input_dtypes[0], ge::DataType::DT_FLOAT);
  EXPECT_EQ(custom_dtype_3.output_dtypes[1], ge::DataType::DT_UNDEFINED);
  EXPECT_EQ(config.GetPrecisionPolicy("A", GRAY), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("A", BLACK), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("A", WHITE), BLACK);


  session_graph_id = "0_3";
  options[ge::OPTION_EXEC_DISABLE_REUSED_MEMORY] = "1";
  options[ge::CUSTOMIZE_DTYPES] = GetCodeDir() + "/tests/engines/nn_engine/ut/stub/custom.cfg";
  options[ge::MODIFY_MIXLIST] = GetCodeDir() + "/tests/engines/nn_engine/config/mix_list/op_mix_list2.json";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ret = config.RefreshParameters();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(config.IsEnableReuseMemory(), 0);
  (void)config.GetCustomizeDtypeByOpName("Matmul_name", custom_dtype_0);
  (void)config.GetCustomizeDtypeByOpType("MatMul", custom_dtype_1);
  (void)config.GetCustomizeDtypeByOpName("Test_name", custom_dtype_2);
  (void)config.GetCustomizeDtypeByOpName("Test_name_2", custom_dtype_3);
  EXPECT_EQ(custom_dtype_0.input_dtypes[1], ge::DataType::DT_FLOAT);
  EXPECT_EQ(custom_dtype_0.output_dtypes[0], ge::DataType::DT_FLOAT);
  EXPECT_EQ(custom_dtype_1.input_dtypes[0], ge::DataType::DT_FLOAT16);
  EXPECT_EQ(custom_dtype_1.output_dtypes[0], ge::DataType::DT_FLOAT);
  EXPECT_EQ(custom_dtype_2.input_dtypes[0], ge::DataType::DT_FLOAT);
  EXPECT_EQ(custom_dtype_3.output_dtypes[1], ge::DataType::DT_UNDEFINED);
  EXPECT_EQ(config.GetPrecisionPolicy("A", GRAY), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("A", BLACK), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("A", WHITE), BLACK);
}

TEST_F(configuration_ut, session_graph_config_params_02) {
  Configuration config(AI_CORE_NAME);
  map<string, string> options;
  string session_graph_id = "0_1";
  options.emplace(ge::OPTION_EXEC_DISABLE_REUSED_MEMORY, "");
  config.is_init_ = false;
  Status ret = config.Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  options.emplace(ge::CUSTOMIZE_DTYPES, "");
  config.is_init_ = false;
  ret = config.Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  options.emplace(ge::MODIFY_MIXLIST, "");
  config.is_init_ = false;
  ret = config.Initialize(options);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(configuration_ut, session_graph_config_params_03) {
  Configuration config(AI_CORE_NAME);
  map<string, string> options;
  string session_graph_id = "0_1";
  options.emplace(ge::OPTION_EXEC_DISABLE_REUSED_MEMORY, "3");
  config.is_init_ = false;
  Status ret = config.Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  options.emplace(ge::CUSTOMIZE_DTYPES, "xxx");
  config.is_init_ = false;
  ret = config.Initialize(options);
  EXPECT_EQ(ret, FAILED);

  options.emplace(ge::MODIFY_MIXLIST, "xxx");
  config.is_init_ = false;
  ret = config.Initialize(options);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(configuration_ut, session_graph_config_params_04) {
  Configuration config(AI_CORE_NAME);
  map<string, string> options;
  string session_graph_id = "0_1";
  options.emplace(ge::OPTION_EXEC_DISABLE_REUSED_MEMORY, "1");
  options.emplace(ge::CUSTOMIZE_DTYPES, GetCodeDir() + "/tests/engines/nn_engine/ut/stub/custom.cfg");
  options.emplace(ge::MODIFY_MIXLIST, GetCodeDir() + "/tests/engines/nn_engine/config/mix_list/op_mix_list1.json");
  config.is_init_ = false;
  Status ret = config.Initialize(options);
  ge::GetThreadLocalContext().SetGraphOption(options);
  ret = config.RefreshParameters();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(config.IsEnableReuseMemory(), 0);
  OpCustomizeDtype custom_dtype_0, custom_dtype_1, custom_dtype_2, custom_dtype_3;
  (void)config.GetCustomizeDtypeByOpName("Matmul_name", custom_dtype_0);
  (void)config.GetCustomizeDtypeByOpType("MatMul", custom_dtype_1);
  (void)config.GetCustomizeDtypeByOpName("Test_name", custom_dtype_2);
  (void)config.GetCustomizeDtypeByOpName("Test_name_2", custom_dtype_3);
  EXPECT_EQ(custom_dtype_0.input_dtypes[1], ge::DataType::DT_FLOAT);
  EXPECT_EQ(custom_dtype_0.output_dtypes[0], ge::DataType::DT_FLOAT);
  EXPECT_EQ(custom_dtype_1.input_dtypes[0], ge::DataType::DT_FLOAT16);
  EXPECT_EQ(custom_dtype_1.output_dtypes[0], ge::DataType::DT_FLOAT);
  EXPECT_EQ(custom_dtype_2.input_dtypes[0], ge::DataType::DT_FLOAT);
  EXPECT_EQ(custom_dtype_3.output_dtypes[1], ge::DataType::DT_UNDEFINED);
  EXPECT_EQ(config.GetPrecisionPolicy("A", GRAY), GRAY);
  EXPECT_EQ(config.GetPrecisionPolicy("A", BLACK), BLACK);
  EXPECT_EQ(config.GetPrecisionPolicy("A", WHITE), WHITE);

  session_graph_id = "0_2";
  options[ge::OPTION_EXEC_DISABLE_REUSED_MEMORY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ret = config.RefreshParameters();
  EXPECT_EQ(ret, FAILED);

  options[ge::CUSTOMIZE_DTYPES] = GetCodeDir() + "/tests/engines/nn_engine/ut/stub/custom2.cfg";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ret = config.RefreshParameters();
  EXPECT_EQ(ret, FAILED);

  options[ge::CUSTOMIZE_DTYPES] = "";
  options[ge::MODIFY_MIXLIST] = GetCodeDir() + "/tests/engines/nn_engine/config/mix_list/op_mix_list2.json";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ret = config.RefreshParameters();
  EXPECT_EQ(ret, FAILED);

  config.SetEnableSmallChannel(true);
  ge::GetThreadLocalContext().GetOo().working_opt_names_to_value_[fe::kComLevelO1Opt] = fe::kStrFalse;
  config.RefreshSmallChannel();
  EXPECT_EQ(config.IsEnableSmallChannel(), false);
  ge::GetThreadLocalContext().GetOo().working_opt_names_to_value_.clear();
}

TEST_F(configuration_ut, get_time_interval_config) {
  EXPECT_EQ(Configuration::Instance(AI_CORE_NAME).GetCompileTaskTraceTimeInterval(), 300);
  EXPECT_EQ(Configuration::Instance(AI_CORE_NAME).GetCompileTaskTraceTimeConstThreshold(), 300);
}

TEST_F(configuration_ut, get_export_compile_stat) {
  Configuration config(AI_CORE_NAME);
  config.is_init_ = false;
  std::map<string, string> options;
  Status ret = config.Initialize(options);
  ge::GetThreadLocalContext().SetGraphOption(options);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(config.GetExportCompileStat(), ExportCompileStatType::AFTER_EXEC_COMPLITE);

  options.emplace("ge.exportCompileStat", "0");
  ge::GetThreadLocalContext().SetGraphOption(options);
  EXPECT_EQ(config.GetExportCompileStat(), ExportCompileStatType::NONE);

  options["ge.exportCompileStat"] = "1";
  ge::GetThreadLocalContext().SetGraphOption(options);
  EXPECT_EQ(config.GetExportCompileStat(), ExportCompileStatType::AFTER_EXEC_COMPLITE);

  options["ge.exportCompileStat"] = "2";
  ge::GetThreadLocalContext().SetGraphOption(options);
  EXPECT_EQ(config.GetExportCompileStat(), ExportCompileStatType::AFTER_COMPILE_COMPLITE);

  options["ge.exportCompileStat"] = "3";
  ge::GetThreadLocalContext().SetGraphOption(options);
  config.is_init_ = false;
  ret = config.Initialize(options);
  EXPECT_EQ(ret, SUCCESS);
  ret = config.RefreshParameters();
  EXPECT_EQ(ret, FAILED);
}

TEST_F(configuration_ut, get_export_compile_stat_support_session_graph) {
  Configuration config(AI_CORE_NAME);
  map<string, string> options;
  string session_graph_id = "0_1";
  options.emplace("ge.exportCompileStat", "0");
  ge::GetThreadLocalContext().SetGraphOption(options);
  config.is_init_ = false;
  Status ret = config.Initialize(options);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(config.GetExportCompileStat(), ExportCompileStatType::NONE);

  session_graph_id = "0_2";
  options["ge.exportCompileStat"] = "1";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ret = config.RefreshParameters();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(config.GetExportCompileStat(), ExportCompileStatType::AFTER_EXEC_COMPLITE);
}

TEST_F(configuration_ut, test_get_opp_latest_path) {
  Configuration config(AI_CORE_NAME);
  std::string opp_path;
  int32_t err = 0;
  std::string home_path = "";
  MM_SYS_SET_ENV(MM_ENV_ASCEND_HOME_PATH, home_path.c_str(), 1, err);
  Status ret = config.GetOppLatestPath(opp_path);
  EXPECT_EQ(ret, FAILED);
  home_path = "  ";
  MM_SYS_SET_ENV(MM_ENV_ASCEND_HOME_PATH, home_path.c_str(), 1, err);
  ret = config.GetOppLatestPath(opp_path);
  EXPECT_EQ(ret, FAILED);
  std::string path_env = "/home/jenkins/Ascend/ascend-toolkit/latest";
  MM_SYS_SET_ENV(MM_ENV_ASCEND_HOME_PATH, path_env.c_str(), 1, err);
}
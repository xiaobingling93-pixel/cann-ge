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
#include <string>
#include <memory>
#include <map>
#include <utility>
#include "fe_llt_utils.h"
#include "common/util/op_info_util.h"
#include "common/aicore_util_types.h"
#define private public
#define protected public
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "ops_store/ops_kernel_manager.h"
#include "common/platform_utils.h"
#include "graph/ge_tensor.h"
#include "graph/op_kernel_bin.h"
#include "graph/ge_local_context.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph_optimizer/op_setter/op_setter.h"
#include "fusion_manager/fusion_manager.h"
#include "ops_kernel_store/sub_ops_store.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "format_selector/manager/format_dtype_querier.h"
#include "common/platform_utils.h"
#include "common/config_parser/modify_mixlist_config_parser.h"
#include "adapter/common/op_store_adapter_manager.h"

using namespace std;
using namespace testing;
using namespace fe;
using namespace te;

using fe::FEOpsKernelInfoStore;
using fe::SubOpsStore;
using ge::GeTensorDesc;
using ge::GeShape;
using ge::AttrUtils;
using ge::Format;
using ge::DataType;
using ge::ConstGeTensorDescPtr;
using ge::GeTensorDescPtr;
using ge::OpDescPtr;
using ge::OpDesc;
using fe::InputOrOutputInfoPtr ;
using ge::GeAttrValue;
using std::vector;
using std::map;
using namespace ge;

using OpSetterPtr = std::shared_ptr<OpSetter>;
using FormatDtypeQuerierPtr = std::shared_ptr<FormatDtypeQuerier>;

enum TestIter {
    TEST_SUCCESS = 0,
    TEST_HAVE_ALL,        // have one "all" type for attr check
    TEST_ATTR_NOT_FOUND,  // can not found attr ATTR_NAME_STR in OpDesc
    TEST_NOT_SUPPORT_DATA_TYPE,  // exit not support ValueType
    TEST_CHECK_FAILED,    // have one not match iter (ATTR_NAME_FLOAT)
    TEST_INT,
    TEST_FLOAT,
    TEST_BOOL,
    TEST_STR,
    TEST_LIST_INT,
    TEST_LIST_FLOAT,
    TEST_LIST_BOOL,
    TEST_LIST_STR,
    TEST_LACK_OF_ATTR_INT
};

extern bool teGeneralize(const te::TbeOpInfo &op_info, const te::TE_GENERALIZE_TYPE &general_type,
                         const ge::NodePtr &node);
extern bool checkIsRegistered(const te::TbeOpInfo &op_info, bool &val);

namespace {
const string ATTR_NAME_INT = "transposX";
const string ATTR_NAME_FLOAT = "transposY";
const string ATTR_NAME_STR = "attrStr";
const string ATTR_NAME_BOOL = "attrBool";
const string ATTR_NAME_LIST_INT = "attrListInt";
const string ATTR_NAME_LIST_FLOAT = "attrListFloat";
const string ATTR_NAME_LIST_STR = "attrListStr";
const string ATTR_NAME_LIST_BOOL = "attrListBool";
const string ATTR_NAME_DEFAULT = "attr_name_default";

te::LX_QUERY_STATUS GetOpInfoStubTestImplJudgeSt(const te::TbeOpInfo &a, std::string &b) {
  return te::LX_QUERY_SUCC;
};

bool PreBuildTbeOpStubTestImplJudgeSt(te::TbeOpInfo &a, uint64_t b, uint64_t c) {
  a.SetPattern("Opaque");
  return true;
};

bool CheckTbeSupportedReasonRange(TbeOpInfo& opinfo, CheckSupportedInfo &result) {
  result.reason = "The shape is not support now";
  return true;
}

bool CheckTbeSupportedOtherReason(TbeOpInfo& opinfo, CheckSupportedInfo &result) {
  result.reason = "other";
  return true;
}

bool GetOpSpecificInfoStub(const TbeOpInfo &tbeOpInfo, std::string &opSpecificInfo) {
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore", "promoteType": "x1,x2"})";
  opSpecificInfo = json_str;
  return true;
}
}

class OptimizeUtilityStubST: public ge::OptimizeUtility {
 public:
  OptimizeUtilityStubST() {}
  virtual ~OptimizeUtilityStubST() override {}

  ge::Status InferShape(ComputeGraph &compute_graph) override{
    return ge::SUCCESS;
  }

  ge::Status InferShape(const ComputeGraphPtr &compute_graph) override {
    return ge::SUCCESS;
  }
};

class STEST_OP_KERNEL_INFO_STORE : public testing::Test{
  protected:
    static void SetUpTestCase() {
        cout << "STEST_OP_KERNEL_INFO_STORE SetUP" << endl;
    }
    static void TearDownTestCase() {
        cout << "STEST_OP_KERNEL_INFO_STORE TearDown" << endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp(){
        op_desc_ptr = make_shared<ge::OpDesc>();
        input0_desc_ptr = make_shared<ge::GeTensorDesc>();
        input1_desc_ptr = make_shared<ge::GeTensorDesc>();
        input2_desc_ptr = make_shared<ge::GeTensorDesc>();
        output0_desc_ptr = make_shared<ge::GeTensorDesc>();
        std::map<std::string, std::string> options;

        tbe_adapter_ptr_ = std::dynamic_pointer_cast<TbeOpStoreAdapter>(OpStoreAdapterManager::Instance(AI_CORE_NAME).GetOpStoreAdapter(EN_IMPL_HW_TBE));
        tbe_adapter_ptr_->GetOpInfo = GetOpInfoStubTestImplJudgeSt;
        tbe_adapter_ptr_->PreBuildTbeOp = PreBuildTbeOpStubTestImplJudgeSt;

        sub_ops_store_ptr = make_shared<fe::SubOpsStore>(fe::AI_CORE_NAME);
        FEOpsStoreInfo tbe_custom {
          2,
          "tbe-custom",
          EN_IMPL_CUSTOM_TBE,
          GetCodeDir() + "/tests/engines/nn_engine/st/testcase/ops_kernel_store/fe_config/tbe_custom_opinfo",
          ""};

        sub_ops_store_ptr->SetSubStoreInfo(tbe_custom);
        Status stu = sub_ops_store_ptr->InitializeSubStore();
        EXPECT_EQ(fe::SUCCESS, stu);
        sub_ops_store_ptr->init_flag_ = true;

        sub_ops_store_ptr_cce = make_shared<fe::SubOpsStore>(fe::AI_CORE_NAME);
        FEOpsStoreInfo cce_custom {
        1,
        "cce-custom",
        EN_IMPL_HW_CONSTANT_CCE,
        GetCodeDir() + "/tests/engines/nn_engine/st/testcase/ops_kernel_store/fe_config/cce_custom_opinfo",
        ""};
        sub_ops_store_ptr_cce->SetSubStoreInfo(cce_custom);
        stu = sub_ops_store_ptr_cce->InitializeSubStore();
        EXPECT_EQ(fe::SUCCESS, stu);
        sub_ops_store_ptr_cce->init_flag_ = true;
        Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = {tbe_custom, cce_custom};
        if (Configuration::Instance(fe::AI_CORE_NAME).mix_list_parser_ == nullptr) {
          Configuration::Instance(fe::AI_CORE_NAME).mix_list_parser_ = make_shared<ModifyMixlistConfigParser>();
        }
        mixlist_parser_ptr_ =
                std::dynamic_pointer_cast<ModifyMixlistConfigParser>(Configuration::Instance(fe::AI_CORE_NAME).mix_list_parser_);

        OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

        fe_ops_kernel_info_store_ptr = make_shared<fe::FEOpsKernelInfoStore>(AI_CORE_NAME);
        Configuration::Instance(fe::AI_CORE_NAME).content_map_["check_subformat.enable"] = "true";
        fe_ops_kernel_info_store_ptr->Initialize(options);
        fe_ops_store_ptr = make_shared<fe::FEOpsKernelInfoStore>();
        fe_ops_store_ptr->init_flag_ = true;
        fe_ops_store_ptr->map_all_sub_store_info_.emplace(std::make_pair("cce-custom", sub_ops_store_ptr_cce));
        fe_ops_store_ptr->map_all_sub_store_info_.emplace(std::make_pair("tbe-custom", sub_ops_store_ptr));
        fe_ops_store_ptr->op_kernel_store_type_ = "FEOpsStore";
        op_desc_ptr->SetName("tbe_conv");
        ge::OpDescUtilsEx::SetType(op_desc_ptr, "conv");
        ge::DataType set_dtype = ge::DT_FLOAT16;
        ge::Format set_format = ge::FORMAT_ND;
        std::vector<int64_t> shape_vec{256,256,512};
        ge::GeShape shape_desc = GeShape(shape_vec);

        input0_desc_ptr->SetDataType(set_dtype);
        input0_desc_ptr->SetFormat(set_format);
        input0_desc_ptr->SetShape(shape_desc);
        op_desc_ptr->AddInputDesc("x", input0_desc_ptr->Clone());

        std::vector<int64_t> shape_vec1{256,256,512};
        ge::GeShape shape_desc1 = GeShape(shape_vec1);
        input1_desc_ptr->SetDataType(set_dtype);
        input1_desc_ptr->SetFormat(set_format);
        input1_desc_ptr->SetShape(shape_desc1);
        op_desc_ptr->AddInputDesc("y", input1_desc_ptr->Clone());

        std::vector<int64_t> shape_vec2{256,256,512};
        ge::GeShape shape_desc2 = GeShape(shape_vec2);
        input2_desc_ptr->SetDataType(set_dtype);
        input2_desc_ptr->SetFormat(set_format);
        input2_desc_ptr->SetShape(shape_desc2);
        op_desc_ptr->AddInputDesc("x1", input2_desc_ptr->Clone());

        output0_desc_ptr->SetDataType(set_dtype);
        output0_desc_ptr->SetFormat(set_format);
        op_desc_ptr->AddOutputDesc("z", output0_desc_ptr->Clone());

        format_dtype_querier_ptr_ = std::make_shared<FormatDtypeQuerier>(AI_CORE_NAME);
        cout << "A ops kernel info store stest set up" << endl;

    }
    virtual void TearDown(){
        cout << "A ops kernel info store stest is tearing down" << endl;
        sub_ops_store_ptr->FinalizeSubStore();
        sub_ops_store_ptr_cce->FinalizeSubStore();
        fe_ops_store_ptr->Finalize();
        fe_ops_kernel_info_store_ptr->Finalize();
       // c_fe_ops_kernel_info_store_ptr.reset();

    }
    void set_op_desc_default_value (OpDescPtr &op_desc_ptr_t)
    {
        op_desc_ptr_t->SetName("tbe_conv");
        ge::OpDescUtilsEx::SetType(op_desc_ptr_t, "conv");
    }

    void SetOpDescPtrAttrValue(TestIter test_iter, OpDescPtr desc_ptr)
    {
        if (test_iter == TEST_INT) {
            AttrUtils::SetInt(desc_ptr, ATTR_NAME_INT, 10);
        }else if (test_iter != TEST_LACK_OF_ATTR_INT) {
            AttrUtils::SetInt(desc_ptr, ATTR_NAME_INT, 1);
        }
        if (test_iter == TEST_FLOAT) {
            AttrUtils::SetFloat(desc_ptr, ATTR_NAME_FLOAT, 22.0);
        }else{
            AttrUtils::SetFloat(desc_ptr, ATTR_NAME_FLOAT, 2.0);
        }
        if (test_iter == TEST_BOOL) {
            AttrUtils::SetBool(desc_ptr, ATTR_NAME_BOOL, true);
        }else{
            AttrUtils::SetBool(desc_ptr, ATTR_NAME_BOOL, false);
        }
        if (test_iter == TEST_STR) {
            AttrUtils::SetStr(desc_ptr, ATTR_NAME_STR, "not_exist");
        }else{
            AttrUtils::SetStr(desc_ptr, ATTR_NAME_STR, "abc");
        }
        if (test_iter == TEST_LIST_INT) {
            AttrUtils::SetListInt(desc_ptr, ATTR_NAME_LIST_INT, {6,7,8});
        }else{
            AttrUtils::SetListInt(desc_ptr, ATTR_NAME_LIST_INT, {1,2,3});
        }
        if (test_iter == TEST_LIST_FLOAT) {
            AttrUtils::SetListFloat(desc_ptr, ATTR_NAME_LIST_FLOAT, {6.0, 7.0, 8.0});
        }else{
            AttrUtils::SetListFloat(desc_ptr, ATTR_NAME_LIST_FLOAT, {1.0, 2.0, 3.0});
        }
        if (test_iter == TEST_LIST_BOOL) {
            AttrUtils::SetListBool(desc_ptr, ATTR_NAME_LIST_BOOL, {true,false,true});
        }else{
            AttrUtils::SetListBool(desc_ptr, ATTR_NAME_LIST_BOOL, {true,true,true});
        }
        if (test_iter == TEST_LIST_STR) {
            AttrUtils::SetListStr(desc_ptr, ATTR_NAME_LIST_STR, {"aa", "bb", "cc"});
        }else{
            AttrUtils::SetListStr(desc_ptr, ATTR_NAME_LIST_STR, {"a", "b", "c"});
        }

    }
public:
    shared_ptr<fe::SubOpsStore> sub_ops_store_ptr;
    shared_ptr<fe::SubOpsStore>  sub_ops_store_ptr_cce;
    shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_store_ptr;
    shared_ptr<ge::GeTensorDesc> input0_desc_ptr;
    shared_ptr<ge::GeTensorDesc> input1_desc_ptr;
    shared_ptr<ge::GeTensorDesc> input2_desc_ptr;
    shared_ptr<ge::GeTensorDesc> output0_desc_ptr;
    shared_ptr<ge::OpDesc> op_desc_ptr;
    TbeOpStoreAdapterPtr tbe_adapter_ptr_;
    FormatDtypeQuerierPtr format_dtype_querier_ptr_;
    shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr;
    ModifyMixlistConfigParserPtr mixlist_parser_ptr_;
};


void CreateConvSt(ge::NodePtr &node, string op_type) {
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>();
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> input1_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> input2_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> output0_desc_ptr = make_shared<ge::GeTensorDesc>();
  op_desc_ptr_t->SetName("tbe_conv");
  ge::OpDescUtilsEx::SetType(op_desc_ptr_t, op_type);
  int64_t int_value = 1;
  float float_value = 2.0;
  bool bool_value = false;
  string str_value = "abc";
  vector<int64_t> int_vec{1, 2, 3};
  vector<int64_t> rint_vec;
  vector<float> float_vec{4.0, 5.0, 6.0};
  vector<float> rfloat_vec;
  vector<bool> bool_vec{false, true, true};
  vector<bool> rbool_vec;
  std::vector<string> str_vec{"a", "b", "c"};
  AttrUtils::SetInt(op_desc_ptr_t, "transposX", int_value);
  AttrUtils::SetFloat(op_desc_ptr_t, "transposY", float_value);
  AttrUtils::SetBool(op_desc_ptr_t, "attrBool", bool_value);
  AttrUtils::SetStr(op_desc_ptr_t, "attrStr", str_value);
  AttrUtils::SetListInt(op_desc_ptr_t, "attrListInt", int_vec);
  AttrUtils::SetListFloat(op_desc_ptr_t, "attrListFloat", float_vec);
  AttrUtils::SetListBool(op_desc_ptr_t, "attrListBool", bool_vec);
  AttrUtils::SetListStr(op_desc_ptr_t, "attrListStr", str_vec);

  ge::DataType set_dtype = ge::DT_FLOAT16;
  std::vector<int64_t> shape_vec{256, 256, 512};
  ge::GeShape shape_desc = GeShape(shape_vec);

  input0_desc_ptr->SetDataType(set_dtype);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetFormat(ge::FORMAT_NCHW);
  input0_desc_ptr->SetOriginFormat(ge::FORMAT_NCHW);
  op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

  std::vector<int64_t> shape_vec1{256, 256, 512};
  ge::GeShape shape_desc1 = GeShape(shape_vec1);
  input1_desc_ptr->SetDataType(set_dtype);
  input1_desc_ptr->SetShape(shape_desc1);
  input1_desc_ptr->SetFormat(ge::FORMAT_NCHW);
  input1_desc_ptr->SetOriginFormat(ge::FORMAT_NCHW);
  op_desc_ptr_t->AddInputDesc("y", input1_desc_ptr->Clone());

  std::vector<int64_t> shape_vec2{256, 256, 512};
  ge::GeShape shape_desc2 = GeShape(shape_vec2);
  input2_desc_ptr->SetDataType(set_dtype);
  input2_desc_ptr->SetShape(shape_desc2);
  input2_desc_ptr->SetFormat(ge::FORMAT_NCHW);
  input2_desc_ptr->SetOriginFormat(ge::FORMAT_NCHW);
  op_desc_ptr_t->AddInputDesc("h", input2_desc_ptr->Clone());

  ge::DataType set_dtype2 = ge::DT_FLOAT;
  output0_desc_ptr->SetDataType(set_dtype2);
  output0_desc_ptr->SetShape(shape_desc);
  output0_desc_ptr->SetFormat(ge::FORMAT_NCHW);
  output0_desc_ptr->SetOriginFormat(ge::FORMAT_NCHW);
  op_desc_ptr_t->AddOutputDesc("z", output0_desc_ptr->Clone());
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  node = graph->AddNode(op_desc_ptr_t);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, impl_judge_1) {
  ge::NodePtr test_node;
  CreateConvSt(test_node, "conv");

  OpImplTypeJudge impl_judge("AiCoreEngine", fe_ops_store_ptr);
  Status result = impl_judge.JudgeByNode(test_node);
  EXPECT_EQ(result, fe::SUCCESS);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, impl_judge_2) {
  ge::NodePtr test_node;
  CreateConvSt(test_node, "conv_dynamic");

  OpImplTypeJudge impl_judge("AiCoreEngine", fe_ops_store_ptr);
  Status result = impl_judge.JudgeByNode(test_node);

  EXPECT_EQ(result, fe::FAILED);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, impl_judge_3) {
  ge::NodePtr test_node;
  CreateConvSt(test_node, "conv");

  OpImplTypeJudge impl_judge("AiCoreEngine", fe_ops_store_ptr);
  AttrUtils::SetBool(test_node->GetOpDesc(), kAttrName64BytesFlag, true);
  Status result = impl_judge.JudgeByNode(test_node);
  EXPECT_EQ(result, fe::SUCCESS);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, initialize_succ){
    shared_ptr<SubOpsStore> sub_ops_store_ptr = make_shared<SubOpsStore>(fe::AI_CORE_NAME);
    FEOpsStoreInfo tbe_custom {
          2,
          "tbe-custom",
          EN_IMPL_CUSTOM_TBE,
          GetCodeDir() + "/tests/engines/nn_engine/st/testcase/ops_kernel_store/fe_config/tbe_custom_opinfo",
          ""};
    sub_ops_store_ptr->SetSubStoreInfo(tbe_custom);
    Status ret = sub_ops_store_ptr->InitializeSubStore();
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, initialize_twice){
    shared_ptr<SubOpsStore> sub_ops_store_ptr = make_shared<SubOpsStore>(fe::AI_CORE_NAME);
    map<string, string> options;
    FEOpsStoreInfo tbe_custom {
          2,
          "tbe-custom",
          EN_IMPL_CUSTOM_TBE,
          GetCodeDir() + "/tests/engines/nn_engine/st/testcase/ops_kernel_store/fe_config/tbe_custom_opinfo",
          ""};
    sub_ops_store_ptr->SetSubStoreInfo(tbe_custom);
    Status ret1 = sub_ops_store_ptr->InitializeSubStore();
    Status ret2 = sub_ops_store_ptr->InitializeSubStore();
    EXPECT_EQ(fe::SUCCESS, ret1);
    EXPECT_EQ(fe::SUCCESS, ret2);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, get_all_ops_kernel_info_succ){
    shared_ptr<map<string, ge::OpInfo>> infos = make_shared<map<string, ge::OpInfo>>();
    fe_ops_store_ptr->GetAllOpsKernelInfo(*(infos.get()));
    EXPECT_NE(false, infos->size());
    infos.reset();
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, get_one_op_kernel_info_ptr)
{
    string op_type = "conv";
    string op_not_exist = "relu";
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", op_type);
    OpKernelInfoPtr op_kernel_info_ptr1 = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", op_not_exist);
    EXPECT_EQ(nullptr, op_kernel_info_ptr1);
    EXPECT_NE(nullptr, op_kernel_info_ptr);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, get_high_prio_op_kernel_info_ptr)
{
    string op_type = "conv";
    string op_not_exist = "relu";

    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetHighPrioOpKernelInfoPtr(op_type);
    OpKernelInfoPtr op_kernel_info_ptr1 = OpsKernelManager::Instance(AI_CORE_NAME).GetHighPrioOpKernelInfoPtr(op_not_exist);

    EXPECT_NE(nullptr, op_kernel_info_ptr);
    if(op_kernel_info_ptr != nullptr){
        EXPECT_EQ("conv", op_kernel_info_ptr->GetOpType());
    }

    EXPECT_EQ(nullptr, op_kernel_info_ptr1);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, check_attr_supported_succ)
{
    shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>();
    set_op_desc_default_value(op_desc_ptr_t);
    SetOpDescPtrAttrValue(TEST_SUCCESS, op_desc_ptr_t);
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "conv");
    EXPECT_NE(nullptr, op_kernel_info_ptr);
    std::string reason;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(op_desc_ptr_t);
    bool ret = sub_ops_store_ptr->CheckAttrSupport(test_node, *(op_kernel_info_ptr.get()), reason);
    EXPECT_EQ(true, ret);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, check_attr_supported)
{
    shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>();
    set_op_desc_default_value(op_desc_ptr_t);
    SetOpDescPtrAttrValue(TEST_SUCCESS, op_desc_ptr_t);
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "conv");
    EXPECT_NE(nullptr, op_kernel_info_ptr);
    std::string reason;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(op_desc_ptr_t);
    bool ret = sub_ops_store_ptr->CheckAttrSupport(test_node, *(op_kernel_info_ptr.get()), reason);
    EXPECT_EQ(true, ret);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, check_attr_supported_lack_of_attr)
{
    shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>();
    set_op_desc_default_value(op_desc_ptr_t);
    /* Do not initialize attr int for this op desc */
    SetOpDescPtrAttrValue(TEST_LACK_OF_ATTR_INT, op_desc_ptr_t);
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("cce-custom", "conv");
    EXPECT_NE(nullptr, op_kernel_info_ptr);

    if(op_kernel_info_ptr != nullptr) {
        std::string reason;
        ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
        ge::NodePtr test_node = graph->AddNode(op_desc_ptr_t);
        bool ret = sub_ops_store_ptr->CheckAttrSupport(test_node, *(op_kernel_info_ptr.get()), reason);
        EXPECT_EQ(false, ret);
    }
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, check_attr_int_false)
{
    shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>();
    set_op_desc_default_value(op_desc_ptr_t);
    /* Set int value as 10, which is not supported  */
    SetOpDescPtrAttrValue(TEST_INT, op_desc_ptr_t);
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("cce-custom", "conv");
    EXPECT_NE(op_kernel_info_ptr, nullptr);

    if(op_kernel_info_ptr != nullptr) {
        std::string reason;

ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
        ge::NodePtr test_node = graph->AddNode(op_desc_ptr_t);
        bool ret = sub_ops_store_ptr->CheckAttrSupport(test_node, *(op_kernel_info_ptr.get()), reason);
        EXPECT_EQ(false, ret);
    }
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, check_attr_float_false)
{
    shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>();
    set_op_desc_default_value(op_desc_ptr_t);
    /* Set float value as 22.0, which is not supported  */
    SetOpDescPtrAttrValue(TEST_FLOAT, op_desc_ptr_t);

    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("cce-custom", "conv");
    EXPECT_NE(op_kernel_info_ptr, nullptr);
    if(op_kernel_info_ptr != nullptr) {
        std::string reason;
        ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
        ge::NodePtr test_node = graph->AddNode(op_desc_ptr_t);
        bool ret = sub_ops_store_ptr->CheckAttrSupport(test_node, *(op_kernel_info_ptr.get()), reason);
        EXPECT_EQ(false, ret);
    }
}
TEST_F(STEST_OP_KERNEL_INFO_STORE, check_attr_str_false)
{
    shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>();
    set_op_desc_default_value(op_desc_ptr_t);
    /* Set string value as "not exist", which is not supported  */
    SetOpDescPtrAttrValue(TEST_STR, op_desc_ptr_t);

    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("cce-custom", "conv");
    EXPECT_NE(op_kernel_info_ptr, nullptr);
    if(op_kernel_info_ptr != nullptr) {
        std::string reason;
        ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
        ge::NodePtr test_node = graph->AddNode(op_desc_ptr_t);
        bool ret = sub_ops_store_ptr->CheckAttrSupport(test_node, *(op_kernel_info_ptr.get()), reason);
        EXPECT_EQ(false, ret);
    }
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, check_attr_bool_false)
{
    shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>();
    set_op_desc_default_value(op_desc_ptr_t);
    /* Set bool value as true, which is not supported  */
    SetOpDescPtrAttrValue(TEST_BOOL, op_desc_ptr_t);
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("cce-custom", "conv");
    EXPECT_NE(nullptr, op_kernel_info_ptr);
    std::string reason;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(op_desc_ptr_t);
    bool ret = sub_ops_store_ptr->CheckAttrSupport(test_node, *(op_kernel_info_ptr.get()), reason);
    EXPECT_EQ(false, ret);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, check_attr_supported_list_bool_false)
{
    shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>();
    set_op_desc_default_value(op_desc_ptr_t);
    /* Set list bool value as [true, false, true], which is not supported  */
    SetOpDescPtrAttrValue(TEST_LIST_BOOL, op_desc_ptr_t);

    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("cce-custom", "conv");
    EXPECT_NE(nullptr, op_kernel_info_ptr);
    std::string reason;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(op_desc_ptr_t);
    bool ret = sub_ops_store_ptr->CheckAttrSupport(test_node, *(op_kernel_info_ptr.get()), reason);
    EXPECT_EQ(false, ret);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, check_attr_supported_list_int_false)
{
    shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>();
    set_op_desc_default_value(op_desc_ptr_t);
    /* Set list int value as [6, 7, 8], which is not supported  */
    SetOpDescPtrAttrValue(TEST_LIST_INT, op_desc_ptr_t);
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("cce-custom", "conv");
    EXPECT_NE(nullptr, op_kernel_info_ptr);
    std::string reason;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(op_desc_ptr_t);
    bool ret = sub_ops_store_ptr->CheckAttrSupport(test_node, *(op_kernel_info_ptr.get()), reason);
    EXPECT_EQ(false, ret);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, check_attr_list_float_false)
{
    shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>();
    set_op_desc_default_value(op_desc_ptr_t);
    /* Set list float value as [6.0, 7.0, 8.0], which is not supported  */
    SetOpDescPtrAttrValue(TEST_LIST_FLOAT, op_desc_ptr_t);
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("cce-custom", "conv");
    EXPECT_NE(nullptr, op_kernel_info_ptr);
    std::string reason;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(op_desc_ptr_t);
    bool ret = sub_ops_store_ptr->CheckAttrSupport(test_node, *(op_kernel_info_ptr.get()), reason);
    EXPECT_EQ(false, ret);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, check_attr_list_str_false)
{

    shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>();
    set_op_desc_default_value(op_desc_ptr_t);
    /* Set list string value as ["aa", "bb", "cc"], which is not supported  */
    SetOpDescPtrAttrValue(TEST_LIST_STR, op_desc_ptr_t);
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "conv");
    EXPECT_NE(nullptr, op_kernel_info_ptr);
    std::string reason;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(op_desc_ptr_t);
    bool ret = sub_ops_store_ptr->CheckAttrSupport(test_node, *(op_kernel_info_ptr.get()), reason);
    EXPECT_EQ(false, ret);
}
TEST_F(STEST_OP_KERNEL_INFO_STORE, check_supported_succ)
{
    shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>();
    shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
    shared_ptr<ge::GeTensorDesc>  input1_desc_ptr = make_shared<ge::GeTensorDesc>();
    shared_ptr<ge::GeTensorDesc>  input2_desc_ptr = make_shared<ge::GeTensorDesc>();
    shared_ptr<ge::GeTensorDesc>  output0_desc_ptr = make_shared<ge::GeTensorDesc>();
    op_desc_ptr_t->SetName("tbe_conv");
    ge::OpDescUtilsEx::SetType(op_desc_ptr_t, "conv");
    int64_t int_value = 1;
    float float_value = 2.0;
    bool bool_value = false;
    string str_value = "abc";
    vector<int64_t> int_vec{1, 2, 3};
    vector<int64_t> rint_vec;
    vector<float> float_vec{4.0, 5.0, 6.0};
    vector<float> rfloat_vec;
    vector<bool> bool_vec{false, true, true};
    vector<bool> rbool_vec;
    std::vector<string> str_vec{"a", "b", "c"};
    AttrUtils::SetInt(op_desc_ptr_t, "transposX", int_value);
    AttrUtils::SetFloat(op_desc_ptr_t, "transposY", float_value);
    AttrUtils::SetBool(op_desc_ptr_t,"attrBool", bool_value);
    AttrUtils::SetStr(op_desc_ptr_t,"attrStr", str_value);
    AttrUtils::SetListInt(op_desc_ptr_t, "attrListInt", int_vec);
    AttrUtils::SetListFloat(op_desc_ptr_t, "attrListFloat", float_vec);
    AttrUtils::SetListBool(op_desc_ptr_t, "attrListBool", bool_vec);
    AttrUtils::SetListStr(op_desc_ptr_t, "attrListStr", str_vec);

    ge::DataType set_dtype = ge::DT_FLOAT16;
    std::vector<int64_t> shape_vec{256,256,512};
    ge::GeShape shape_desc = GeShape(shape_vec);

    input0_desc_ptr->SetDataType(set_dtype);
    input0_desc_ptr->SetShape(shape_desc);
    input0_desc_ptr->SetFormat(ge::FORMAT_NCHW);
    input0_desc_ptr->SetOriginFormat(ge::FORMAT_NCHW);
    op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

    std::vector<int64_t> shape_vec1{256,256,512};
    ge::GeShape shape_desc1 = GeShape(shape_vec1);
    input1_desc_ptr->SetDataType(set_dtype);
    input1_desc_ptr->SetShape(shape_desc1);
    input1_desc_ptr->SetFormat(ge::FORMAT_NCHW);
    input1_desc_ptr->SetOriginFormat(ge::FORMAT_NCHW);
    op_desc_ptr_t->AddInputDesc("y", input1_desc_ptr->Clone());

    std::vector<int64_t> shape_vec2{256,256,512};
    ge::GeShape shape_desc2 = GeShape(shape_vec2);
    input2_desc_ptr->SetDataType(set_dtype);
    input2_desc_ptr->SetShape(shape_desc2);
    input2_desc_ptr->SetFormat(ge::FORMAT_NCHW);
    input2_desc_ptr->SetOriginFormat(ge::FORMAT_NCHW);
    op_desc_ptr_t->AddInputDesc("x1", input2_desc_ptr->Clone());

    output0_desc_ptr->SetDataType(set_dtype);
    output0_desc_ptr->SetShape(shape_desc);
    output0_desc_ptr->SetFormat(ge::FORMAT_NCHW);
    output0_desc_ptr->SetOriginFormat(ge::FORMAT_NCHW);
    op_desc_ptr_t->AddOutputDesc("z", output0_desc_ptr->Clone());

    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("cce-custom", "conv");
    EXPECT_NE(nullptr, op_kernel_info_ptr);
    std::string reason;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(op_desc_ptr_t);
    bool ret = sub_ops_store_ptr->CheckAttrSupport(test_node, *(op_kernel_info_ptr.get()), reason);
    string un_supported_reason;
    bool ret2 = fe_ops_store_ptr->CheckSupported(op_desc_ptr_t, un_supported_reason);
    ge::ComputeGraphPtr graph_hgl = std::make_shared<ge::ComputeGraph>("Node");
    ge::NodePtr node = graph_hgl->AddNode(op_desc_ptr_t);
    ge::CheckSupportFlag flag = ge::CheckSupportFlag::kDefault;
    (void)fe_ops_kernel_info_store_ptr->CheckSupported(node, un_supported_reason, flag);
    EXPECT_EQ(flag, ge::CheckSupportFlag::kDefault);
    EXPECT_EQ(true, ret);
    EXPECT_EQ(true, ret2);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, check_supported_shape_fail)
{
    shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>();
    shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
    shared_ptr<ge::GeTensorDesc>  input1_desc_ptr = make_shared<ge::GeTensorDesc>();
    shared_ptr<ge::GeTensorDesc>  input2_desc_ptr = make_shared<ge::GeTensorDesc>();
    shared_ptr<ge::GeTensorDesc>  output0_desc_ptr = make_shared<ge::GeTensorDesc>();
    op_desc_ptr_t->SetName("tbe_conv");
    ge::OpDescUtilsEx::SetType(op_desc_ptr_t, "conv");
    int64_t int_value = 1;
    float float_value = 2.0;
    bool bool_value = false;
    string str_value = "abc";
    vector<int64_t> int_vec{1, 2, 3};
    vector<int64_t> rint_vec;
    vector<float> float_vec{4.0, 5.0, 6.0};
    vector<float> rfloat_vec;
    vector<bool> bool_vec{false, true, true};
    vector<bool> rbool_vec;
    std::vector<string> str_vec{"a", "b", "c"};
    AttrUtils::SetInt(op_desc_ptr_t, "transposX", int_value);
    AttrUtils::SetFloat(op_desc_ptr_t, "transposY", float_value);
    AttrUtils::SetBool(op_desc_ptr_t,"attrBool", bool_value);
    AttrUtils::SetStr(op_desc_ptr_t,"attrStr", str_value);
    AttrUtils::SetListInt(op_desc_ptr_t, "attrListInt", int_vec);
    AttrUtils::SetListFloat(op_desc_ptr_t, "attrListFloat", float_vec);
    AttrUtils::SetListBool(op_desc_ptr_t, "attrListBool", bool_vec);
    AttrUtils::SetListStr(op_desc_ptr_t, "attrListStr", str_vec);

    ge::DataType set_dtype = ge::DT_FLOAT16;
    std::vector<int64_t> shape_vec{256,256,512};
    ge::GeShape shape_desc = GeShape(shape_vec);

    input0_desc_ptr->SetDataType(set_dtype);
    input0_desc_ptr->SetShape(shape_desc);
    op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

    std::vector<int64_t> shape_vec1{256,-1,512};
    ge::GeShape shape_desc1 = GeShape(shape_vec1);
    input1_desc_ptr->SetDataType(set_dtype);
    input1_desc_ptr->SetShape(shape_desc1);
    op_desc_ptr_t->AddInputDesc("y", input1_desc_ptr->Clone());

    std::vector<int64_t> shape_vec2{256,256,512};
    ge::GeShape shape_desc2 = GeShape(shape_vec2);
    input2_desc_ptr->SetDataType(set_dtype);
    input2_desc_ptr->SetShape(shape_desc2);
    op_desc_ptr_t->AddInputDesc("x1", input2_desc_ptr->Clone());

    output0_desc_ptr->SetDataType(set_dtype);
    output0_desc_ptr->SetShape(shape_desc);
    op_desc_ptr_t->AddOutputDesc("z", output0_desc_ptr->Clone());

    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("cce-custom", "conv");
    EXPECT_NE(nullptr, op_kernel_info_ptr);

    string un_supported_reason;
    bool ret1 = fe_ops_store_ptr->CheckSupported(op_desc_ptr_t, un_supported_reason);
    EXPECT_EQ(false, ret1);
    un_supported_reason = "";
    bool ret2 = fe_ops_store_ptr->CheckSupported(op_desc_ptr_t, un_supported_reason);
    EXPECT_EQ(false, ret2);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, check_not_support_dyn_soc)
{
    shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>();
    shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
    shared_ptr<ge::GeTensorDesc> output0_desc_ptr = make_shared<ge::GeTensorDesc>();
    op_desc_ptr_t->SetName("tbe_conv");
    ge::OpDescUtilsEx::SetType(op_desc_ptr_t, "DynamicRank");

    ge::DataType set_dtype = ge::DT_INT32;
    std::vector<int64_t> shape_vec{-2};
    ge::GeShape shape_desc = GeShape(shape_vec);

    input0_desc_ptr->SetDataType(set_dtype);
    input0_desc_ptr->SetShape(shape_desc);
    op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

    output0_desc_ptr->SetDataType(set_dtype);
    output0_desc_ptr->SetShape(shape_desc);
    op_desc_ptr_t->AddOutputDesc("y", output0_desc_ptr->Clone());
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "UnknownShape");
    SubOpsStorePtr sub_ops_store_ptr = std::make_shared<SubOpsStore>(AI_CORE_NAME);
    std::string un_supported_reason;
    bool ret = fe_ops_kernel_info_store_ptr->CheckSupported(op_desc_ptr_t, un_supported_reason);
    EXPECT_EQ(false, ret);
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("Node");
    ge::NodePtr node = graph->AddNode(op_desc_ptr_t);
    ge::CheckSupportFlag flag = ge::CheckSupportFlag::kDefault;
    string soc_version_bk = PlatformUtils::Instance().soc_version_;
    PlatformUtils::Instance().soc_version_ = "Ascend310";
    (void)fe_ops_kernel_info_store_ptr->CheckSupported(node, un_supported_reason, flag);
    EXPECT_NE(flag, ge::CheckSupportFlag::kNotSupportDynamicShape);
    PlatformUtils::Instance().soc_version_ = soc_version_bk;
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, check_accuracy_supported_succ)
{
  shared_ptr<FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr = make_shared<FEOpsKernelInfoStore>();

  std::map<std::string, std::string> options;
  fe_ops_kernel_info_store_ptr = make_shared<fe::FEOpsKernelInfoStore>();
  FEOpsStoreInfo tbe_custom {
      2,
      "tbe-custom",
      EN_IMPL_CUSTOM_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
      ""};

  vector<FEOpsStoreInfo> store_info;
  store_info.emplace_back(tbe_custom);
  Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
  OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

  fe_ops_kernel_info_store_ptr->Initialize(options);
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>();
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc>  input1_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc>  input2_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc>  output0_desc_ptr = make_shared<ge::GeTensorDesc>();
  op_desc_ptr_t->SetName("tbe_conv");
  ge::OpDescUtilsEx::SetType(op_desc_ptr_t, "conv");
  int64_t int_value = 1;
  float float_value = 2.0;
  bool bool_value = false;
  string str_value = "abc";
  vector<int64_t> int_vec{1, 2, 3};
  vector<int64_t> rint_vec;
  vector<float> float_vec{4.0, 5.0, 6.0};
  vector<float> rfloat_vec;
  vector<bool> bool_vec{false, true, true};
  vector<bool> rbool_vec;
  std::vector<string> str_vec{"a", "b", "c"};
  AttrUtils::SetInt(op_desc_ptr_t, "transposX", int_value);
  AttrUtils::SetFloat(op_desc_ptr_t, "transposY", float_value);
  AttrUtils::SetBool(op_desc_ptr_t,"attrBool", bool_value);
  AttrUtils::SetStr(op_desc_ptr_t,"attrStr", str_value);
  AttrUtils::SetListInt(op_desc_ptr_t, "attrListInt", int_vec);
  AttrUtils::SetListFloat(op_desc_ptr_t, "attrListFloat", float_vec);
  AttrUtils::SetListBool(op_desc_ptr_t, "attrListBool", bool_vec);
  AttrUtils::SetListStr(op_desc_ptr_t, "attrListStr", str_vec);

  ge::DataType set_dtype = ge::DT_FLOAT16;
  std::vector<int64_t> shape_vec{256,256,512};
  ge::GeShape shape_desc = GeShape(shape_vec);

  input0_desc_ptr->SetDataType(set_dtype);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetFormat(ge::FORMAT_NCHW);
  input0_desc_ptr->SetOriginFormat(ge::FORMAT_NCHW);
  op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

  std::vector<int64_t> shape_vec1{256,256,512};
  ge::GeShape shape_desc1 = GeShape(shape_vec1);
  input1_desc_ptr->SetDataType(set_dtype);
  input1_desc_ptr->SetShape(shape_desc1);
  input1_desc_ptr->SetFormat(ge::FORMAT_NCHW);
  input1_desc_ptr->SetOriginFormat(ge::FORMAT_NCHW);
  op_desc_ptr_t->AddInputDesc("y", input1_desc_ptr->Clone());

  std::vector<int64_t> shape_vec2{256,256,512};
  ge::GeShape shape_desc2 = GeShape(shape_vec2);
  input2_desc_ptr->SetDataType(set_dtype);
  input2_desc_ptr->SetShape(shape_desc2);
  input2_desc_ptr->SetFormat(ge::FORMAT_NCHW);
  input2_desc_ptr->SetOriginFormat(ge::FORMAT_NCHW);
  op_desc_ptr_t->AddInputDesc("x1", input2_desc_ptr->Clone());

  ge::DataType set_dtype2 = ge::DT_FLOAT;
  output0_desc_ptr->SetDataType(set_dtype2);
  output0_desc_ptr->SetShape(shape_desc);
  output0_desc_ptr->SetFormat(ge::FORMAT_NCHW);
  output0_desc_ptr->SetOriginFormat(ge::FORMAT_NCHW);
  op_desc_ptr_t->AddOutputDesc("z", output0_desc_ptr->Clone());

  OpKernelInfoPtr op_kernel_info_ptr;
  SubOpsStorePtr sub_ops_store_ptr = fe_ops_kernel_info_store_ptr->map_all_sub_store_info_["tbe-custom"];
  string un_supported_reason;
  op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "conv");
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr test_node = graph->AddNode(op_desc_ptr_t);
  bool ret1 = sub_ops_store_ptr->CheckAttrSupport(test_node, *(op_kernel_info_ptr.get()), un_supported_reason);
  bool ret2 = fe_ops_kernel_info_store_ptr->CheckAccuracySupported(op_desc_ptr_t, un_supported_reason);
  int64_t is_check_supported = 1;
  is_check_supported = (is_check_supported << 63);
  ge::AttrUtils::SetInt(op_desc_ptr_t, IS_CHECK_SUPPORTED, is_check_supported);
  bool ret3 = fe_ops_kernel_info_store_ptr->CheckAccuracySupported(op_desc_ptr_t, un_supported_reason);
  EXPECT_EQ(true, ret1);
  EXPECT_EQ(true, ret2);
  EXPECT_EQ(false, ret3);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, check_accuracy_supported_for_cast)
{
  FEOpsStoreInfo tbe_custom {
      2,
      "tbe-custom",
      EN_IMPL_CUSTOM_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
      ""};
  vector<FEOpsStoreInfo> store_info;
  store_info.emplace_back(tbe_custom);
  Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
  OpsKernelManager::Instance(AI_CORE_NAME).Finalize();
  shared_ptr<FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr = make_shared<FEOpsKernelInfoStore>();
  std::map<std::string, std::string> options;
  fe_ops_kernel_info_store_ptr->Initialize(options);

  ge::Format format_5dh_16 = static_cast<ge::Format>(ge::GetFormatFromC0(ge::FORMAT_NC1HWC0, 5));
  ge::Format format_5dh_8 = static_cast<ge::Format>(ge::GetFormatFromC0(ge::FORMAT_NC1HWC0, 4));
  std::vector<int64_t> dims_nchw{10,20,15,15};
  std::vector<int64_t> dims_5hd{10,2,15,15,16};
  ge::GeShape shape_nchw(dims_nchw);
  ge::GeShape shape_5hd(dims_5hd);
  ge::GeTensorDesc tensor_desc(shape_5hd, format_5dh_16, ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetOriginShape(shape_nchw);
  ge::OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("cast", "Cast");
  op_desc_ptr->AddInputDesc("x", tensor_desc);
  op_desc_ptr->AddOutputDesc("y", tensor_desc);

  string un_supported_reason;
  bool ret = fe_ops_kernel_info_store_ptr->CheckAccuracySupported(op_desc_ptr, un_supported_reason);
  EXPECT_EQ(ret, true);

  op_desc_ptr->MutableOutputDesc(0)->SetFormat(format_5dh_8);
  ret = fe_ops_kernel_info_store_ptr->CheckAccuracySupported(op_desc_ptr, un_supported_reason);
  EXPECT_EQ(ret, true);

  op_desc_ptr->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
  ret = fe_ops_kernel_info_store_ptr->CheckAccuracySupported(op_desc_ptr, un_supported_reason);
  EXPECT_EQ(ret, false);

}

TEST_F(STEST_OP_KERNEL_INFO_STORE, check_accuracy_supported_failed1)
{
  shared_ptr<FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr = make_shared<FEOpsKernelInfoStore>();
  std::map<std::string, std::string> options;
  fe_ops_kernel_info_store_ptr = make_shared<fe::FEOpsKernelInfoStore>();
  FEOpsStoreInfo tbe_custom {
      2,
      "tbe-custom",
      EN_IMPL_CUSTOM_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
      ""};

  vector<FEOpsStoreInfo> store_info;
  store_info.emplace_back(tbe_custom);
  Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
  OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

  fe_ops_kernel_info_store_ptr->Initialize(options);
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>();
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc>  input1_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc>  input2_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc>  output0_desc_ptr = make_shared<ge::GeTensorDesc>();
  op_desc_ptr_t->SetName("tbe_conv");
  ge::OpDescUtilsEx::SetType(op_desc_ptr_t, "conv");
  int64_t int_value = 1;
  float float_value = 2.0;
  bool bool_value = false;
  string str_value = "abc";
  vector<int64_t> int_vec{1, 2, 3};
  vector<int64_t> rint_vec;
  vector<float> float_vec{4.0, 5.0, 6.0};
  vector<float> rfloat_vec;
  vector<bool> bool_vec{false, true, true};
  vector<bool> rbool_vec;
  std::vector<string> str_vec{"a", "b", "c"};
  AttrUtils::SetInt(op_desc_ptr_t, "transposX", int_value);
  AttrUtils::SetFloat(op_desc_ptr_t, "transposY", float_value);
  AttrUtils::SetBool(op_desc_ptr_t,"attrBool", bool_value);
  AttrUtils::SetStr(op_desc_ptr_t,"attrStr", str_value);
  AttrUtils::SetListInt(op_desc_ptr_t, "attrListInt", int_vec);
  AttrUtils::SetListFloat(op_desc_ptr_t, "attrListFloat", float_vec);
  AttrUtils::SetListBool(op_desc_ptr_t, "attrListBool", bool_vec);
  AttrUtils::SetListStr(op_desc_ptr_t, "attrListStr", str_vec);

  ge::DataType set_dtype = ge::DT_FLOAT16;
  std::vector<int64_t> shape_vec{256,256,512};
  ge::GeShape shape_desc = GeShape(shape_vec);

  input0_desc_ptr->SetDataType(set_dtype);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetFormat(ge::FORMAT_NCHW);
  input0_desc_ptr->SetOriginFormat(ge::FORMAT_NCHW);
  op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

  std::vector<int64_t> shape_vec1{256,256,512};
  ge::GeShape shape_desc1 = GeShape(shape_vec1);
  input1_desc_ptr->SetDataType(set_dtype);
  input1_desc_ptr->SetShape(shape_desc1);
  input1_desc_ptr->SetFormat(ge::FORMAT_NCHW);
  input1_desc_ptr->SetOriginFormat(ge::FORMAT_NCHW);
  op_desc_ptr_t->AddInputDesc("y", input1_desc_ptr->Clone());

  std::vector<int64_t> shape_vec2{256,256,512};
  ge::GeShape shape_desc2 = GeShape(shape_vec2);
  input2_desc_ptr->SetDataType(set_dtype);
  input2_desc_ptr->SetShape(shape_desc2);
  input2_desc_ptr->SetFormat(ge::FORMAT_NCHW);
  input2_desc_ptr->SetOriginFormat(ge::FORMAT_NCHW);
  op_desc_ptr_t->AddInputDesc("x1", input2_desc_ptr->Clone());

  output0_desc_ptr->SetDataType(set_dtype);
  output0_desc_ptr->SetShape(shape_desc);
  output0_desc_ptr->SetFormat(ge::FORMAT_NCHW);
  output0_desc_ptr->SetOriginFormat(ge::FORMAT_NCHW);
  op_desc_ptr_t->AddOutputDesc("z", output0_desc_ptr->Clone());

  OpKernelInfoPtr op_kernel_info_ptr;
  string un_supported_reason;

  SubOpsStorePtr sub_ops_store_ptr = fe_ops_kernel_info_store_ptr->map_all_sub_store_info_["tbe-custom"];
  op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "conv");
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr test_node = graph->AddNode(op_desc_ptr_t);
  bool ret1 = sub_ops_store_ptr->CheckAttrSupport(test_node, *(op_kernel_info_ptr.get()), un_supported_reason);
  bool ret2 = fe_ops_kernel_info_store_ptr->CheckAccuracySupported(op_desc_ptr_t, un_supported_reason);
  int64_t is_check_supported = 1;
  is_check_supported = (is_check_supported << 63);
  ge::AttrUtils::SetInt(op_desc_ptr_t, IS_CHECK_SUPPORTED, is_check_supported);
  bool ret3 = fe_ops_kernel_info_store_ptr->CheckAccuracySupported(op_desc_ptr_t, un_supported_reason);
  EXPECT_EQ(true, ret1);
  EXPECT_EQ(false, ret2);
  EXPECT_EQ(false, ret3);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, check_dtype_false)
{
    shared_ptr<ge::GeTensorDesc> input_ptr = make_shared<ge::GeTensorDesc>();
    shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>();
    set_op_desc_default_value(op_desc_ptr_t);
    /* Set list int value as [6, 7, 8], which is not supported  */
    SetOpDescPtrAttrValue(TEST_SUCCESS, op_desc_ptr_t);

    ge::DataType set_dtype = ge::DT_UINT64;
    ge::Format set_format = ge::FORMAT_ND;
    std::vector<int64_t> shape_vec{256,256,512};
    ge::GeShape shape_desc = GeShape(shape_vec);

    input_ptr->SetDataType(set_dtype);
    input_ptr->SetFormat(set_format);
    input_ptr->SetShape(shape_desc);
    op_desc_ptr_t->AddInputDesc("x", input_ptr->Clone());
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("cce-custom", "conv");
    EXPECT_NE(nullptr, op_kernel_info_ptr);
    InputOrOutputInfoPtr input_info_ptr;
    op_kernel_info_ptr->GetInputInfoByName("x", input_info_ptr);

    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(op_desc_ptr_t);
    FormatDtypeInfo format_dtype_info;
    Status result = format_dtype_querier_ptr_->GetSupportFormatAndDtype(op_kernel_info_ptr, test_node,
                                                                 false, format_dtype_info);
    EXPECT_EQ(fe::SUCCESS, result);
    SupportedFormatAndDtype check_info(op_kernel_info_ptr, "");
    bool ret = sub_ops_store_ptr->CheckDtypeSupported(test_node, input_ptr, input_info_ptr,
                                                      format_dtype_info.data_type_map.at(input_info_ptr->GetUniqueName()),
                                                      check_info);
    EXPECT_EQ(fe::SUCCESS, ret);
    check_info.promote_flag = true;
    check_info.is_input = true;
    check_info.cur_idx = 0;
    check_info.promote_target_type.emplace_back(ge::DT_FLOAT);
    vector<int> tmp_vec = {0, 1};
    check_info.promote_input_list.emplace_back(tmp_vec);
    vector<ge::DataType> op_support_data_types = {ge::DT_FLOAT16, ge::DT_FLOAT};
    ret = fe::SubOpsStore::CheckPromoteTypeSupport(op_support_data_types, check_info);
    EXPECT_EQ(true, ret);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, feed_promote_info){
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("mul_0", "Mul_Custom");
  ge::NodePtr test_node = graph->AddNode(op_desc_ptr);

  ge::Format format_5dh_16 = static_cast<ge::Format>(ge::GetFormatFromC0(ge::FORMAT_NC1HWC0, 5));
  std::vector<int64_t> dims_nchw{10,20,15,15};
  std::vector<int64_t> dims_5hd{10,2,15,15,16};
  ge::GeShape shape_nchw(dims_nchw);
  ge::GeShape shape_5hd(dims_5hd);
  ge::GeTensorDesc tensor_desc(shape_5hd, format_5dh_16, ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetOriginShape(shape_nchw);
  ge::GeTensorDesc tensor_desc1 = tensor_desc;
  tensor_desc1.SetDataType(ge::DT_INT32);
  tensor_desc.SetDataType(ge::DT_FLOAT);

  op_desc_ptr->AddInputDesc("x1", tensor_desc);
  op_desc_ptr->AddInputDesc("x2", tensor_desc1);
  op_desc_ptr->AddOutputDesc("y", tensor_desc);
  OpKernelInfoPtr op_kernel_info_ptr =
      OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "Mul_Custom");
  SupportedFormatAndDtype info(op_kernel_info_ptr, "");
  SubOpsStorePtr sub_ops_store_ptr = std::make_shared<SubOpsStore>(AI_CORE_NAME);
  sub_ops_store_ptr->FeedPromoteInfo(test_node, info);
  EXPECT_EQ(true, info.promote_flag);
  sub_ops_store_ptr->FeedPromoteInfo(test_node, info);

  ge::OpDescPtr op_desc_ptr_1 = make_shared<ge::OpDesc>("mul_2", "Mul_Promote");
  op_desc_ptr_1->AddInputDesc("x1", tensor_desc);
  op_desc_ptr_1->AddInputDesc("x2", tensor_desc1);
  op_desc_ptr_1->AddOutputDesc("y", tensor_desc);
  ge::NodePtr test_node_1 = graph->AddNode(op_desc_ptr_1);
  tbe_adapter_ptr_->GetOpSpecificInfo = GetOpSpecificInfoStub;
  OpKernelInfoPtr op_kernel_info_ptr_1=
      OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "Mul_Promote");
  SupportedFormatAndDtype info_1(op_kernel_info_ptr_1, "");
  sub_ops_store_ptr->FeedPromoteInfo(test_node_1, info_1);
  info_1.op_kernel_info_ptr = nullptr;
  sub_ops_store_ptr->FeedPromoteInfo(test_node_1, info_1);

  ge::OpDescPtr op_desc_ptr_2 = make_shared<ge::OpDesc>("mul", "Mul_Dynamic");
  op_desc_ptr_2->AddInputDesc("x1", tensor_desc);
  op_desc_ptr_2->AddInputDesc("x2", tensor_desc1);
  op_desc_ptr_2->AddOutputDesc("y", tensor_desc);
  ge::NodePtr test_node_2 = graph->AddNode(op_desc_ptr_2);
  tbe_adapter_ptr_->GetOpSpecificInfo = GetOpSpecificInfoStub;
  OpKernelInfoPtr op_kernel_info_ptr_2=
      OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "Mul_Dynamic");
  SupportedFormatAndDtype info_2(op_kernel_info_ptr_2, "");
  sub_ops_store_ptr->FeedPromoteInfo(test_node_2, info_2);
  std::string promote_str;
  tbe_adapter_ptr_->GetDynamicPromoteType(test_node_2, op_kernel_info_ptr_2, promote_str);
  EXPECT_EQ(promote_str, "x1,x2");

  TileFwkOpInfo::Instance().SetTileFwkOpFlag("Mul_Dynamic", true);
  ge::OpDescPtr op_desc_ptr_3 = make_shared<ge::OpDesc>("mul_3", "Mul_Dynamic");
  op_desc_ptr_3->AddInputDesc("x1", tensor_desc);
  op_desc_ptr_3->AddInputDesc("x2", tensor_desc1);
  op_desc_ptr_3->AddOutputDesc("y", tensor_desc);
  ge::NodePtr test_node_3 = graph->AddNode(op_desc_ptr_3);
  OpKernelInfoPtr op_kernel_info_ptr_3=
      OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "Mul_Dynamic");
  SupportedFormatAndDtype info_3(op_kernel_info_ptr_3, "");
  EXPECT_EQ(tbe_adapter_ptr_->IsNeedSkipOpJudge(test_node_3, op_kernel_info_ptr_3), true);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, finalize_succ){
    Status ret = fe_ops_store_ptr->Finalize();
    EXPECT_EQ(fe::SUCCESS, ret);
}

bool CheckTbeSupportedStub1(TbeOpInfo& opinfo, CheckSupportedInfo &result) {
  result.isSupported = opinfo.IsDynamicImpl() ? te::FULLY_SUPPORTED : te::NOT_SUPPORTED;
  return true;
}

bool CheckTbeSupportedStub2(TbeOpInfo& opinfo, CheckSupportedInfo &result) {
  result.isSupported = opinfo.IsDynamicImpl() ? te::NOT_SUPPORTED : te::FULLY_SUPPORTED;
  return true;
}

bool SelectOpFormatStub(const te::TbeOpInfo &tbeOpInfo, std::string &opDtypeFormat) {
  if (tbeOpInfo.IsDynamicImpl()) {
    opDtypeFormat = "{\"input0\":{\"name\":\"x\", \"dtype\":\"float16\", \"format\":\"NCHW\"},"
                    "\"output0\":{\"name\":\"y\", \"dtype\":\"float16\", \"format\":\"NCHW\"}}";
  } else {
    opDtypeFormat = "{\"input0\":{\"name\":\"x\", \"dtype\":\"float16\", \"format\":\"FRACTAL_NZ\"},"
                    "\"output0\":{\"name\":\"y\", \"dtype\":\"float16\", \"format\":\"FRACTAL_NZ\"}}";
  }
  return true;
}

static void CreateSpacesizeTwoOpGraph(ComputeGraphPtr graph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "conv");
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "conv");

  // add descriptor
  vector<int64_t> dims = {1,2,3,4};
  GeShape shape(dims);

  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", in_desc1);

  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_HWCN);
  out_desc1.SetDataType(DT_FLOAT16);
  relu_op->AddOutputDesc("y", out_desc1);

  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_FRACTAL_Z);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc2);

  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NHWC);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc2);

  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr relu_node = graph->AddNode(relu_op);

  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, check_format_nd_success)
{
    shared_ptr<ge::GeTensorDesc> input_ptr = make_shared<ge::GeTensorDesc>();
    shared_ptr<ge::OpDesc> test_op_desc_ptr = make_shared<ge::OpDesc>();
    SetOpDescPtrAttrValue(TEST_SUCCESS,test_op_desc_ptr);
    ge::DataType set_dtype = ge::DT_UINT64;
    ge::Format set_format = ge::FORMAT_ND;
    std::vector<int64_t> shape_vec{256,256,512};
    ge::GeShape shape_desc = GeShape(shape_vec);

    input_ptr->SetDataType(set_dtype);
    input_ptr->SetOriginFormat(set_format);
    input_ptr->SetFormat(set_format);
    input_ptr->SetShape(shape_desc);
    test_op_desc_ptr->AddInputDesc("x", input_ptr->Clone());

    SubOpsStorePtr sub_ops_store_ptr = fe_ops_store_ptr->map_all_sub_store_info_["tbe-custom"];
    OpKernelInfoPtr op_kernel_info_ptr1 = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "conv");
    InputOrOutputInfoPtr input_info_ptr1;
    op_kernel_info_ptr1->GetInputInfoByName("x", input_info_ptr1);

    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::NodePtr test_node = graph->AddNode(test_op_desc_ptr);
    FormatDtypeInfo format_dtype_info1;
    Status get_format_dtype_status = format_dtype_querier_ptr_->GetSupportFormatAndDtype(op_kernel_info_ptr1, test_node,
                                                                                   false, format_dtype_info1);
    EXPECT_EQ(fe::SUCCESS, get_format_dtype_status);
    bool ret1 = sub_ops_store_ptr->CheckFormatSupported(test_node, input_ptr, input_info_ptr1,
            format_dtype_info1.format_map.at(input_info_ptr1->GetUniqueName()));
    EXPECT_EQ(false, ret1);


    OpKernelInfoPtr op_kernel_info_ptr2 = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "K");
    InputOrOutputInfoPtr input_info_ptr2;
    op_kernel_info_ptr2->GetInputInfoByName("x", input_info_ptr2);
    FormatDtypeInfo format_dtype_info2;
    get_format_dtype_status = format_dtype_querier_ptr_->GetSupportFormatAndDtype(op_kernel_info_ptr2, test_node,
        false, format_dtype_info2);
    EXPECT_EQ(fe::SUCCESS, get_format_dtype_status);

    bool ret2 = sub_ops_store_ptr->CheckFormatSupported(test_node,input_ptr,input_info_ptr2,
            format_dtype_info2.format_map.at(input_info_ptr2->GetUniqueName()));
    EXPECT_EQ(true, ret2);
    std::map<string, vector<ge::Format>> format_map;
    Status status = format_dtype_querier_ptr_->GetAllSupportFormat(op_kernel_info_ptr2, test_node, format_map);
    EXPECT_EQ(status, fe::SUCCESS);
}


TEST_F(STEST_OP_KERNEL_INFO_STORE, set_cut_info_01)
{
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr op = std::make_shared<ge::OpDesc>("op", "Relu");
  ge::OpDescPtr op1 = std::make_shared<ge::OpDesc>("op1", "Relu");
  GeShape shape = ge::GeShape({1,2,1,1,1});
  Format format = FORMAT_NHWC;
  DataType dt = DT_FLOAT;
  ge::GeTensorDesc tensor_desc(shape, format, dt);
  op->AddInputDesc(tensor_desc);
  op->AddOutputDesc(tensor_desc);
  op1->AddInputDesc(tensor_desc);
  op1->AddOutputDesc(tensor_desc);

  string relu_slice_info =
      "{\"_op_slice_info\":{\"l1FusionEnable\":0,\"minTbeL1Space\":0,\"reduceMaps\":"
      "[],\"splitMaps\":[{\"inputList\":[{\"axis\":[0],\"headOverLap\":[],\"idx\":0,\"tailOverLap\":"
      "[]}],\"outputList\":[{\"axis\":[0],\"idx\":0}]},{\"inputList\":[{\"axis\":[1],\"headOverLap\":"
      "[],\"idx\":0,\"tailOverLap\":[]}],\"outputList\":[{\"axis\":[1],\"idx\":0}]},{\"inputList\":[{\"axis\":"
      "[2],\"headOverLap\":[],\"idx\":0,\"tailOverLap\":[]}],\"outputList\":[{\"axis\":[2],\"idx\":0}]},"
      "{\"inputList\":[{\"axis\":[3],\"headOverLap\":[],\"idx\":0,\"tailOverLap\":[]}],\"outputList\":[{\"axis\":"
      "[3],\"idx\":0}]},{\"inputList\":[{\"axis\":[4],\"headOverLap\":[],\"idx\":0,\"tailOverLap\":"
      "[]}],\"outputList\":[{\"axis\":[4],\"idx\":0}]}]}}";
  ge::AttrUtils::SetStr(op, "_op_slice_info", relu_slice_info);
  auto node = graph->AddNode(op);
  auto node1 = graph->AddNode(op1);
  fe_ops_kernel_info_store_ptr->SetCutSupportedInfo(node);
  fe_ops_kernel_info_store_ptr->SetCutSupportedInfo(node1);
  auto input = node->GetOpDesc()->MutableInputDesc(0);
  auto input1 = node1->GetOpDesc()->MutableInputDesc(0);

  vector<vector<int64_t>> current_stgy;
  vector<vector<int64_t>> current_stgy1;
  vector<vector<int64_t>> current_stgy_expect = {
      {1, 0, 0, 0, 0},
      {0, 1, 0, 0, 0},
      {0, 0, 1, 0, 0},
      {0, 0, 0, 1, 0},
      {0, 0, 0, 0, 1}
  };
  (void)ge::AttrUtils::GetListListInt(input, "_cut_info", current_stgy);
  (void)ge::AttrUtils::GetListListInt(input1, "_cut_info", current_stgy1);
  ASSERT_EQ(current_stgy.size(), current_stgy_expect.size());
  ASSERT_EQ(current_stgy1.size(), 0);
  EXPECT_EQ(current_stgy[0], current_stgy_expect[0]);
  EXPECT_EQ(current_stgy[1], current_stgy_expect[1]);
  EXPECT_EQ(current_stgy[2], current_stgy_expect[2]);
  EXPECT_EQ(current_stgy[3], current_stgy_expect[3]);
  EXPECT_EQ(current_stgy[4], current_stgy_expect[4]);

  current_stgy = {};
  auto output = node->GetOpDesc()->MutableOutputDesc(0);
  auto output1 = node1->GetOpDesc()->MutableOutputDesc(0);
  (void)ge::AttrUtils::GetListListInt(output, "_cut_info", current_stgy);
  (void)ge::AttrUtils::GetListListInt(output1, "_cut_info", current_stgy1);
  ASSERT_EQ(current_stgy.size(), current_stgy_expect.size());
  ASSERT_EQ(current_stgy1.size(), 0);
  EXPECT_EQ(current_stgy[0], current_stgy_expect[0]);
  EXPECT_EQ(current_stgy[1], current_stgy_expect[1]);
  EXPECT_EQ(current_stgy[2], current_stgy_expect[2]);
  EXPECT_EQ(current_stgy[3], current_stgy_expect[3]);
  EXPECT_EQ(current_stgy[4], current_stgy_expect[4]);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, precompile_01)
{
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>("test", "DynamicCompileStatic");
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> output0_desc_ptr = make_shared<ge::GeTensorDesc>();

  ge::DataType set_dtype = ge::DT_BOOL;
  ge::Format set_format = ge::FORMAT_NCHW;
  std::vector<int64_t> shape_vec{4, 16, 100, 100};
  ge::GeShape shape_desc = GeShape(shape_vec);

  input0_desc_ptr->SetDataType(set_dtype);
  input0_desc_ptr->SetFormat(set_format);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetOriginDataType(set_dtype);
  input0_desc_ptr->SetOriginFormat(set_format);
  input0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

  output0_desc_ptr->SetDataType(set_dtype);
  output0_desc_ptr->SetFormat(set_format);
  output0_desc_ptr->SetShape(shape_desc);
  output0_desc_ptr->SetOriginDataType(set_dtype);
  output0_desc_ptr->SetOriginFormat(set_format);
  output0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddOutputDesc("y", output0_desc_ptr->Clone());
  OpKernelInfoPtr op_kernel_info_ptr =
      OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "DynamicCompileStatic");
  SubOpsStorePtr sub_ops_store_ptr = std::make_shared<SubOpsStore>(AI_CORE_NAME);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
  auto node = graph->AddNode(op_desc_ptr_t);
  shared_ptr<ge::OpDesc> data1 = make_shared<ge::OpDesc>("data1", DATA);
  (void)ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  auto node_data = graph->AddNode(data1);
  TbeInfoAssembler t;
  t.Initialize();
  te::TbeOpInfo info("test",
                     "test1",
                     "DynamicCompileStatic",
                     fe::AI_CORE_NAME);
  (void)ge::AttrUtils::SetBool(op_desc_ptr_t, ge::ATTR_NAME_DISABLE_ATTACHED_RESOURCE, true);
  Status ret = t.AssembleTbeInfo(node.get(), op_kernel_info_ptr, info, fe::AI_CORE_NAME);
  EXPECT_EQ(info.GetVectorCoreType(), te::VectorCoreType::DISABLE);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, autofuse01)
{
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>("test", "AscBackend");
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> output0_desc_ptr = make_shared<ge::GeTensorDesc>();

  ge::DataType set_dtype = ge::DT_BOOL;
  ge::Format set_format = ge::FORMAT_NCHW;
  std::vector<int64_t> shape_vec{4, 16, 100, 100};
  ge::GeShape shape_desc = GeShape(shape_vec);

  input0_desc_ptr->SetDataType(set_dtype);
  input0_desc_ptr->SetFormat(set_format);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetOriginDataType(set_dtype);
  input0_desc_ptr->SetOriginFormat(set_format);
  input0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

  output0_desc_ptr->SetDataType(set_dtype);
  output0_desc_ptr->SetFormat(set_format);
  output0_desc_ptr->SetShape(shape_desc);
  output0_desc_ptr->SetOriginDataType(set_dtype);
  output0_desc_ptr->SetOriginFormat(set_format);
  output0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddOutputDesc("y", output0_desc_ptr->Clone());

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
  auto node = graph->AddNode(op_desc_ptr_t);
  shared_ptr<ge::OpDesc> data1 = make_shared<ge::OpDesc>("data1", DATA);
  (void)ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  auto node_data = graph->AddNode(data1);
  TbeInfoAssembler t;
  t.Initialize();
  te::TbeOpInfo info("test",
                     "test1",
                     "AscBackend",
                     fe::AI_CORE_NAME);
  (void)ge::AttrUtils::SetBool(op_desc_ptr_t, ge::ATTR_NAME_DISABLE_ATTACHED_RESOURCE, true);
  Status ret = t.AssembleAutoFuseTbeInfo(node.get(), info);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, precompile_02)
{
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>("test", "DynamicCompileStatic");
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> output0_desc_ptr = make_shared<ge::GeTensorDesc>();

  ge::DataType set_input_dtype = ge::DT_BOOL;
  ge::DataType set_output_dtype = ge::DT_VARIANT;
  ge::Format set_format = ge::FORMAT_NCHW;
  std::vector<int64_t> shape_vec{4, 16, 100, 100};
  ge::GeShape shape_desc = GeShape(shape_vec);

  input0_desc_ptr->SetDataType(set_input_dtype);
  input0_desc_ptr->SetFormat(set_format);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetOriginDataType(set_input_dtype);
  input0_desc_ptr->SetOriginFormat(set_format);
  input0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

  output0_desc_ptr->SetDataType(set_output_dtype);
  output0_desc_ptr->SetFormat(set_format);
  output0_desc_ptr->SetShape(shape_desc);
  output0_desc_ptr->SetOriginDataType(set_output_dtype);
  output0_desc_ptr->SetOriginFormat(set_format);
  output0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddOutputDesc("y", output0_desc_ptr->Clone());

  te::TbeOpTensor output_tensor;
  TensorDescAndIndex tensor_info = {output0_desc_ptr, "y", 0, 0, false};

  Status ret = CreateTbeTensor(*op_desc_ptr_t.get(), tensor_info, output_tensor);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, precompile_03)
{
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>("test", "DynamicCompileStatic");
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> output0_desc_ptr = make_shared<ge::GeTensorDesc>();

  ge::DataType set_dtype = ge::DT_BOOL;
  ge::Format set_format = ge::FORMAT_NCHW;
  std::vector<int64_t> shape_vec{4, 16, 100, 100};
  ge::GeShape shape_desc = GeShape(shape_vec);

  input0_desc_ptr->SetDataType(set_dtype);
  input0_desc_ptr->SetFormat(set_format);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetOriginDataType(set_dtype);
  input0_desc_ptr->SetOriginFormat(set_format);
  input0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

  output0_desc_ptr->SetDataType(set_dtype);
  output0_desc_ptr->SetFormat(set_format);
  output0_desc_ptr->SetShape(shape_desc);
  output0_desc_ptr->SetOriginDataType(set_dtype);
  output0_desc_ptr->SetOriginFormat(set_format);
  output0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddOutputDesc("x", output0_desc_ptr->Clone());
  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "InPlaceOp");
  SubOpsStorePtr sub_ops_store_ptr = std::make_shared<SubOpsStore>(AI_CORE_NAME);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
  auto node = graph->AddNode(op_desc_ptr_t);
  TbeInfoAssembler t;
  t.Initialize();
  te::TbeOpInfo info("test",
                     "test1",
                     "DynamicCompileStatic",
                     fe::AI_CORE_NAME);
  Status ret = t.AssembleTbeInfo(node.get(), op_kernel_info_ptr, info, fe::AI_CORE_NAME);

  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, outputInplaceAbility)
{
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>("test", "DynamicCompileStatic");
  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "InPlaceOp");
  std::vector<std::vector<int64_t>> output_inplace = {{0,0},{0,1}};
  op_kernel_info_ptr->SetOutputIplaceInfo(output_inplace);
  TbeInfoAssembler t;
  t.Initialize();
  t.SetOutputInplaceAttr(op_desc_ptr_t, op_kernel_info_ptr);
  std::vector<std::vector<int64_t>> inplace;
  ge::AttrUtils::GetListListInt(op_desc_ptr_t, kAttrOutputInplaceAbility, inplace);
  EXPECT_EQ(inplace.empty(), true);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, parseOutputInplace)
{
  string ouputInplaceAbility = "{0,1}, {1,2}";
  std::vector<std::vector<int64_t>> output_inplace;
  fe::StringUtils::TransStringToListListInt(ouputInplaceAbility, output_inplace);
  std::vector<std::vector<int64_t>> golden = {{0,1}, {1,2}};
  EXPECT_EQ(output_inplace, golden);
  
  ouputInplaceAbility = "{{0,1}, {1,2}";
  std::vector<std::vector<int64_t>> golden1={};
  fe::StringUtils::TransStringToListListInt(ouputInplaceAbility, output_inplace);
  EXPECT_EQ(output_inplace, golden1);
  
  ouputInplaceAbility = "{{0,1,1}, {1,2}";
  fe::StringUtils::TransStringToListListInt(ouputInplaceAbility, output_inplace);
  EXPECT_EQ(output_inplace, golden1);
  
  ouputInplaceAbility = "{{0,1}, {1}}";
  fe::StringUtils::TransStringToListListInt(ouputInplaceAbility, output_inplace);
  EXPECT_EQ(output_inplace, golden1);
  
  ouputInplaceAbility = "{{0,1}, {1, 1}}}";
  fe::StringUtils::TransStringToListListInt(ouputInplaceAbility, output_inplace);
  EXPECT_EQ(output_inplace, golden1);
  
  ouputInplaceAbility = "{{0,1}, {11, 1}}";
  std::vector<std::vector<int64_t>> golden2={{0,1}, {11, 1}};
  fe::StringUtils::TransStringToListListInt(ouputInplaceAbility, output_inplace);
  EXPECT_EQ(output_inplace, golden2);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, precompile_04)
{
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>("test", "DynamicCompileStatic");
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> output0_desc_ptr = make_shared<ge::GeTensorDesc>();

  ge::DataType set_dtype = ge::DT_BOOL;
  ge::Format set_format = ge::FORMAT_NCHW;
  std::vector<int64_t> shape_vec{4, 16, 100, 100};
  ge::GeShape shape_desc = GeShape(shape_vec);

  input0_desc_ptr->SetDataType(set_dtype);
  input0_desc_ptr->SetFormat(set_format);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetOriginDataType(set_dtype);
  input0_desc_ptr->SetOriginFormat(set_format);
  input0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

  output0_desc_ptr->SetDataType(set_dtype);
  output0_desc_ptr->SetFormat(set_format);
  output0_desc_ptr->SetShape(shape_desc);
  output0_desc_ptr->SetOriginDataType(set_dtype);
  output0_desc_ptr->SetOriginFormat(set_format);
  output0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddOutputDesc("x", output0_desc_ptr->Clone());
  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "InPlaceOp");
  SubOpsStorePtr sub_ops_store_ptr = std::make_shared<SubOpsStore>(AI_CORE_NAME);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
  auto node = graph->AddNode(op_desc_ptr_t);
  TbeInfoAssembler t;
  t.Initialize();
  te::TbeOpInfo info("test",
                     "test1",
                     "DynamicCompileStatic",
                     fe::AI_CORE_NAME);
  (void)ge::AttrUtils::SetStr(op_desc_ptr_t, kRelationReusedParam, "");
  Status ret = t.AssembleTbeInfo(node.get(), op_kernel_info_ptr, info, fe::AI_CORE_NAME);

  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, precompile_ub_fusion)
{
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>("test", "DynamicCompileStatic");
  uint32_t ub_size = 2048;
  op_desc_ptr_t->SetExtAttr(ATTR_NAME_UB_FUSION_SPACE_SIZE, ub_size);
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> output0_desc_ptr = make_shared<ge::GeTensorDesc>();

  ge::DataType set_dtype = ge::DT_BOOL;
  ge::Format set_format = ge::FORMAT_NCHW;
  std::vector<int64_t> shape_vec{4, 16, 100, 100};
  ge::GeShape shape_desc = GeShape(shape_vec);

  input0_desc_ptr->SetDataType(set_dtype);
  input0_desc_ptr->SetFormat(set_format);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetOriginDataType(set_dtype);
  input0_desc_ptr->SetOriginFormat(set_format);
  input0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

  output0_desc_ptr->SetDataType(set_dtype);
  output0_desc_ptr->SetFormat(set_format);
  output0_desc_ptr->SetShape(shape_desc);
  output0_desc_ptr->SetOriginDataType(set_dtype);
  output0_desc_ptr->SetOriginFormat(set_format);
  output0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddOutputDesc("x", output0_desc_ptr->Clone());
  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "InPlaceOp");
  SubOpsStorePtr sub_ops_store_ptr = std::make_shared<SubOpsStore>(AI_CORE_NAME);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
  auto node = graph->AddNode(op_desc_ptr_t);
  TbeInfoAssembler t;
  t.Initialize();
  te::TbeOpInfo info("test",
                     "test1",
                     "DynamicCompileStatic",
                     fe::AI_CORE_NAME);
  (void)ge::AttrUtils::SetStr(op_desc_ptr_t, kRelationReusedParam, "");
  Status ret = t.AssembleTbeInfo(node.get(), op_kernel_info_ptr, info, fe::AI_CORE_NAME);

  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, precompile_05)
{
  std::string soc_version = PlatformUtils::Instance().GetSocVersion();
  PlatformUtils::Instance().soc_version_ = "Ascend035";
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::SpecifiedMemBase)] = 1;
  ge::OpDescPtr op_desc_vd = make_shared<ge::OpDesc>("ddr_base_prop", "DdrBaseProp");
  ge::OpDescPtr op_desc_const = make_shared<ge::OpDesc>("const0", "Const");
  ge::OpDescPtr op_desc_pld1 = make_shared<ge::OpDesc>("PlaceHolder1", "PlaceHolder");
  ge::OpDescPtr op_desc_pld2 = make_shared<ge::OpDesc>("PlaceHolder2", "PlaceHolder");
  ge::OpDescPtr op_desc_pld3 = make_shared<ge::OpDesc>("PlaceHolder3", "PlaceHolder");
  ge::OpDescPtr op_desc_data = make_shared<ge::OpDesc>("data0", "Data");
  ge::OpDescPtr op_desc_const1 = make_shared<ge::OpDesc>("const1", "Const");
  ge::OpDescPtr op_desc_k = make_shared<ge::OpDesc>("k", "K");
  ge::OpDescPtr op_desc_end = make_shared<ge::OpDesc>("end", "End");
  ge::OpDescPtr op_desc_end2 = make_shared<ge::OpDesc>("end2", "End");
  ge::OpDescPtr op_desc_netoutput = make_shared<ge::OpDesc>("Net_Output", "NetOutput");

  std::vector<int64_t> shape_vec = {4, 2, 10, 10, 16};
  ge::GeShape shape_desc(shape_vec);
  ge::GeTensorDesc tensor_desc(shape_desc, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetOriginFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetOriginShape(shape_desc);

  ge::GeTensorDesc tensor_desc1(shape_desc, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc1.SetOriginFormat(ge::FORMAT_NC1HWC0);
  tensor_desc1.SetOriginShape(shape_desc);
  ge::AttrUtils::SetInt(tensor_desc1, ge::ATTR_NAME_TENSOR_MEMORY_SCOPE, 2);

  op_desc_vd->AddInputDesc("a", tensor_desc);
  op_desc_vd->AddInputDesc("b", tensor_desc);
  op_desc_vd->AddInputDesc("c", tensor_desc);
  op_desc_vd->AddInputDesc("d", tensor_desc);
  op_desc_vd->AddInputDesc("e", tensor_desc1);
  op_desc_vd->AddOutputDesc("zz", tensor_desc);
  op_desc_vd->AddOutputDesc("zzz", tensor_desc1);

  op_desc_data->AddOutputDesc(tensor_desc);
  op_desc_pld1->AddOutputDesc(tensor_desc);
  op_desc_pld2->AddOutputDesc(tensor_desc);
  op_desc_pld3->AddOutputDesc(tensor_desc);
  op_desc_const->AddOutputDesc(tensor_desc);
  op_desc_const1->AddOutputDesc(tensor_desc1);
  op_desc_k->AddInputDesc(tensor_desc);
  op_desc_k->AddOutputDesc(tensor_desc);
  op_desc_end->AddInputDesc(tensor_desc);
  op_desc_end2->AddInputDesc(tensor_desc);
  op_desc_netoutput->AddInputDesc(tensor_desc);

  ge::AttrUtils::SetStr(op_desc_pld2, "parentOpType", "Data");
  ge::AttrUtils::SetStr(op_desc_pld3, "parentOpType", "Data");
  ge::AttrUtils::SetStr(op_desc_end, "parentOpType", "NetOutput");

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::NodePtr node_vd = graph->AddNode(op_desc_vd);
  ge::NodePtr node_data = graph->AddNode(op_desc_data);
  ge::NodePtr node_pld1 = graph->AddNode(op_desc_pld1);
  ge::NodePtr node_pld2 = graph->AddNode(op_desc_pld2);
  ge::NodePtr node_pld3 = graph->AddNode(op_desc_pld3);
  ge::NodePtr node_const = graph->AddNode(op_desc_const);
  ge::NodePtr node_const1 = graph->AddNode(op_desc_const1);
  ge::NodePtr node_k = graph->AddNode(op_desc_k);
  ge::NodePtr node_end = graph->AddNode(op_desc_end);
  ge::NodePtr node_end2 = graph->AddNode(op_desc_end2);
  ge::NodePtr node_endout = graph->AddNode(op_desc_netoutput);
  op_desc_pld3->SetExtAttr("parentNode", node_data);
  op_desc_end->SetExtAttr("parentNode", node_endout);

  ge::GraphUtils::AddEdge(node_pld1->GetOutDataAnchor(0), node_vd->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_const->GetOutDataAnchor(0), node_vd->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(node_pld2->GetOutDataAnchor(0), node_vd->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(node_pld3->GetOutDataAnchor(0), node_vd->GetInDataAnchor(3));
  ge::GraphUtils::AddEdge(node_const1->GetOutDataAnchor(0), node_vd->GetInDataAnchor(4));
  ge::GraphUtils::AddEdge(node_vd->GetOutDataAnchor(0), node_k->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_vd->GetOutDataAnchor(0), node_end->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_vd->GetOutDataAnchor(1), node_end2->GetInDataAnchor(0));

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "DdrBaseProp");
  SubOpsStorePtr sub_ops_store_ptr = std::make_shared<SubOpsStore>(AI_CORE_NAME);

  TbeInfoAssembler t;
  t.Initialize();
  te::TbeOpInfo tbe_op_info("test", "test1", "DdrBaseProp", fe::AI_CORE_NAME);
  Status ret = t.AssembleTbeInfo(node_vd.get(), op_kernel_info_ptr, tbe_op_info, fe::AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = soc_version;
  std::vector<te::TbeOpParam> inputs;
  tbe_op_info.GetInputs(inputs);
  std::vector<te::TbeOpParam> outputs;
  tbe_op_info.GetOutputs(outputs);
  for (size_t i = 0; i < inputs.size(); i++) {
    te::TbeOpParam op_param = inputs[i];
    std::vector<te::TbeOpTensor> tensors;
    op_param.GetTensors(tensors);
    ASSERT_EQ(tensors.size(), 1);
    if (i == 0) {
      EXPECT_EQ(tensors[0].GetDdrBaseProp(), te::DdrBaseType::WORKSPACE);
    }
    if (i == 1) {
      EXPECT_EQ(tensors[0].GetDdrBaseProp(), te::DdrBaseType::WEIGHT);
    }
    if (i == 2) {
      EXPECT_EQ(tensors[0].GetDdrBaseProp(), te::DdrBaseType::NET_EDGE);
    }
    if (i == 3) {
      EXPECT_EQ(tensors[0].GetDdrBaseProp(), te::DdrBaseType::NET_EDGE);
    }
    if (i == 4) {
      EXPECT_EQ(tensors[0].GetDdrBaseProp(), te::DdrBaseType::NET_EDGE);
    }
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    te::TbeOpParam op_param = outputs[i];
    std::vector<te::TbeOpTensor> tensors;
    op_param.GetTensors(tensors);
    ASSERT_EQ(tensors.size(), 1);
    if (i == 0) {
      EXPECT_EQ(tensors[0].GetDdrBaseProp(), te::DdrBaseType::NET_EDGE);
    }
    if (i == 1) {
      EXPECT_EQ(tensors[0].GetDdrBaseProp(), te::DdrBaseType::NET_EDGE);
    }
  }
  EXPECT_EQ(ret, fe::SUCCESS);
  OpKernelInfoPtr op_kernel_info_ptr_k = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "K");
  te::TbeOpInfo tbe_op_info_k("test2", "test2", "K", fe::AI_CORE_NAME);
  ret = t.AssembleTbeInfo(node_k.get(), op_kernel_info_ptr_k, tbe_op_info_k, fe::AI_CORE_NAME);
  std::vector<te::TbeOpParam> k_inputs;
  tbe_op_info_k.GetInputs(k_inputs);
  std::vector<te::TbeOpParam> k_outputs;
  tbe_op_info_k.GetOutputs(k_outputs);
  for (size_t i = 0; i < k_inputs.size(); i++) {
    te::TbeOpParam op_param = k_inputs[i];
    std::vector<te::TbeOpTensor> tensors;
    op_param.GetTensors(tensors);
    ASSERT_EQ(tensors.size(), 1);
    if (i == 0) {
      EXPECT_EQ(tensors[0].GetDdrBaseProp(), te::DdrBaseType::NET_EDGE);
    }
  }
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, dynamic_compile_static_1)
{
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>("DynamicCompileStatic_op_1", "DynamicCompileStatic");
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> output0_desc_ptr = make_shared<ge::GeTensorDesc>();

  ge::DataType set_dtype = ge::DT_FLOAT16;
  ge::Format set_format = ge::FORMAT_NCHW;
  std::vector<int64_t> shape_vec{4, 16, 100, 100};
  ge::GeShape shape_desc = GeShape(shape_vec);

  input0_desc_ptr->SetDataType(set_dtype);
  input0_desc_ptr->SetFormat(set_format);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetOriginDataType(set_dtype);
  input0_desc_ptr->SetOriginFormat(set_format);
  input0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

  output0_desc_ptr->SetDataType(set_dtype);
  output0_desc_ptr->SetFormat(set_format);
  output0_desc_ptr->SetShape(shape_desc);
  output0_desc_ptr->SetOriginDataType(set_dtype);
  output0_desc_ptr->SetOriginFormat(set_format);
  output0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddOutputDesc("y", output0_desc_ptr->Clone());
  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "UnknownShape");
  SubOpsStorePtr sub_ops_store_ptr = std::make_shared<SubOpsStore>(AI_CORE_NAME);
  std::string un_supported_reason;
  bool ret = fe_ops_store_ptr->CheckSupported(op_desc_ptr_t, un_supported_reason);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(ge::AttrUtils::HasAttr(op_desc_ptr_t, ATTR_NAME_IS_OP_DYNAMIC_IMPL), true);
  EXPECT_EQ(IsOpDynamicImpl(op_desc_ptr_t), true);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, dynamic_compile_static_2)
{
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>("DynamicCompileStatic_op_1", "DynamicCompileStatic");
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> output0_desc_ptr = make_shared<ge::GeTensorDesc>();

  ge::DataType set_dtype = ge::DT_FLOAT16;
  ge::Format set_format = ge::FORMAT_ND;
  std::vector<int64_t> shape_vec{4, 16, 100, 100};
  ge::GeShape shape_desc = GeShape(shape_vec);

  input0_desc_ptr->SetDataType(set_dtype);
  input0_desc_ptr->SetFormat(set_format);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetOriginDataType(set_dtype);
  input0_desc_ptr->SetOriginFormat(set_format);
  input0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

  output0_desc_ptr->SetDataType(set_dtype);
  output0_desc_ptr->SetFormat(set_format);
  output0_desc_ptr->SetShape(shape_desc);
  output0_desc_ptr->SetOriginDataType(set_dtype);
  output0_desc_ptr->SetOriginFormat(set_format);
  output0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddOutputDesc("y", output0_desc_ptr->Clone());
  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "UnknownShape");
  SubOpsStorePtr sub_ops_store_ptr = std::make_shared<SubOpsStore>(AI_CORE_NAME);
  std::string un_supported_reason;
  bool ret = fe_ops_store_ptr->CheckSupported(op_desc_ptr_t, un_supported_reason);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(ge::AttrUtils::HasAttr(op_desc_ptr_t, ATTR_NAME_IS_OP_DYNAMIC_IMPL), true);
  EXPECT_EQ(IsOpDynamicImpl(op_desc_ptr_t), false);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, dynamic_compile_static_3)
{
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>("DynamicCompileStatic_op_1", "DynamicCompileStatic");
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> output0_desc_ptr = make_shared<ge::GeTensorDesc>();

  ge::DataType set_dtype = ge::DT_INT32;
  ge::Format set_format = ge::FORMAT_ND;
  std::vector<int64_t> shape_vec{4, 16, 100, 100};
  ge::GeShape shape_desc = GeShape(shape_vec);

  input0_desc_ptr->SetDataType(set_dtype);
  input0_desc_ptr->SetFormat(set_format);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetOriginDataType(set_dtype);
  input0_desc_ptr->SetOriginFormat(set_format);
  input0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

  output0_desc_ptr->SetDataType(set_dtype);
  output0_desc_ptr->SetFormat(set_format);
  output0_desc_ptr->SetShape(shape_desc);
  output0_desc_ptr->SetOriginDataType(set_dtype);
  output0_desc_ptr->SetOriginFormat(set_format);
  output0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddOutputDesc("y", output0_desc_ptr->Clone());
  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "UnknownShape");
  SubOpsStorePtr sub_ops_store_ptr = std::make_shared<SubOpsStore>(AI_CORE_NAME);
  std::string un_supported_reason;
  bool ret = fe_ops_store_ptr->CheckSupported(op_desc_ptr_t, un_supported_reason);
  EXPECT_EQ(ret, false);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, dynamic_compile_static_4)
{
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>("DynamicCompileStatic_op_1", "DynamicCompileStatic1");
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> output0_desc_ptr = make_shared<ge::GeTensorDesc>();

  ge::DataType set_dtype = ge::DT_FLOAT16;
  ge::Format set_format = ge::FORMAT_NCHW;
  std::vector<int64_t> shape_vec{4, 16, 100, 100};
  ge::GeShape shape_desc = GeShape(shape_vec);

  input0_desc_ptr->SetDataType(set_dtype);
  input0_desc_ptr->SetFormat(set_format);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetOriginDataType(set_dtype);
  input0_desc_ptr->SetOriginFormat(set_format);
  input0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

  output0_desc_ptr->SetDataType(set_dtype);
  output0_desc_ptr->SetFormat(set_format);
  output0_desc_ptr->SetShape(shape_desc);
  output0_desc_ptr->SetOriginDataType(set_dtype);
  output0_desc_ptr->SetOriginFormat(set_format);
  output0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddOutputDesc("y", output0_desc_ptr->Clone());
  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "UnknownShape");
  SubOpsStorePtr sub_ops_store_ptr = std::make_shared<SubOpsStore>(AI_CORE_NAME);
  std::string un_supported_reason;
  tbe_adapter_ptr_->CheckTbeSupported = CheckTbeSupportedStub1;
  bool ret = fe_ops_store_ptr->CheckSupported(op_desc_ptr_t, un_supported_reason);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(ge::AttrUtils::HasAttr(op_desc_ptr_t, ATTR_NAME_IS_OP_DYNAMIC_IMPL), true);
  EXPECT_EQ(IsOpDynamicImpl(op_desc_ptr_t), true);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, dynamic_compile_static_5)
{
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>("DynamicCompileStatic_op_1", "DynamicCompileStatic1");
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> output0_desc_ptr = make_shared<ge::GeTensorDesc>();

  ge::DataType set_dtype = ge::DT_FLOAT16;
  ge::Format set_format = ge::FORMAT_NCHW;
  std::vector<int64_t> shape_vec{4, 16, 100, 100};
  ge::GeShape shape_desc = GeShape(shape_vec);

  input0_desc_ptr->SetDataType(set_dtype);
  input0_desc_ptr->SetFormat(set_format);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetOriginDataType(set_dtype);
  input0_desc_ptr->SetOriginFormat(set_format);
  input0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

  output0_desc_ptr->SetDataType(set_dtype);
  output0_desc_ptr->SetFormat(set_format);
  output0_desc_ptr->SetShape(shape_desc);
  output0_desc_ptr->SetOriginDataType(set_dtype);
  output0_desc_ptr->SetOriginFormat(set_format);
  output0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddOutputDesc("y", output0_desc_ptr->Clone());
  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "UnknownShape");
  SubOpsStorePtr sub_ops_store_ptr = std::make_shared<SubOpsStore>(AI_CORE_NAME);
  std::string un_supported_reason;
  tbe_adapter_ptr_->CheckTbeSupported = CheckTbeSupportedStub2;
  bool ret = fe_ops_store_ptr->CheckSupported(op_desc_ptr_t, un_supported_reason);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(ge::AttrUtils::HasAttr(op_desc_ptr_t, ATTR_NAME_IS_OP_DYNAMIC_IMPL), true);
  EXPECT_EQ(IsOpDynamicImpl(op_desc_ptr_t), false);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, dynamic_compile_static_6)
{
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>("DynamicCompileStatic_op_1", "DynamicCompileStatic2");
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> output0_desc_ptr = make_shared<ge::GeTensorDesc>();

  ge::DataType set_dtype = ge::DT_FLOAT16;
  ge::Format set_format = ge::FORMAT_NCHW;
  std::vector<int64_t> shape_vec{4, 16, 100, 100};
  ge::GeShape shape_desc = GeShape(shape_vec);

  input0_desc_ptr->SetDataType(set_dtype);
  input0_desc_ptr->SetFormat(set_format);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetOriginDataType(set_dtype);
  input0_desc_ptr->SetOriginFormat(set_format);
  input0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

  output0_desc_ptr->SetDataType(set_dtype);
  output0_desc_ptr->SetFormat(set_format);
  output0_desc_ptr->SetShape(shape_desc);
  output0_desc_ptr->SetOriginDataType(set_dtype);
  output0_desc_ptr->SetOriginFormat(set_format);
  output0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddOutputDesc("y", output0_desc_ptr->Clone());
  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "UnknownShape");
  SubOpsStorePtr sub_ops_store_ptr = std::make_shared<SubOpsStore>(AI_CORE_NAME);
  std::string un_supported_reason;
  tbe_adapter_ptr_->SelectTbeOpFormat = SelectOpFormatStub;
  bool ret = fe_ops_store_ptr->CheckSupported(op_desc_ptr_t, un_supported_reason);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(ge::AttrUtils::HasAttr(op_desc_ptr_t, ATTR_NAME_IS_OP_DYNAMIC_IMPL), true);
  EXPECT_EQ(IsOpDynamicImpl(op_desc_ptr_t), true);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, dynamic_compile_static_7)
{
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>("DynamicCompileStatic_op_1", "DynamicCompileStatic2");
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> output0_desc_ptr = make_shared<ge::GeTensorDesc>();

  ge::DataType set_dtype = ge::DT_FLOAT16;
  ge::Format set_format = ge::FORMAT_ND;
  std::vector<int64_t> shape_vec{4, 16, 100, 100};
  ge::GeShape shape_desc = GeShape(shape_vec);

  input0_desc_ptr->SetDataType(set_dtype);
  input0_desc_ptr->SetFormat(set_format);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetOriginDataType(set_dtype);
  input0_desc_ptr->SetOriginFormat(set_format);
  input0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

  output0_desc_ptr->SetDataType(set_dtype);
  output0_desc_ptr->SetFormat(set_format);
  output0_desc_ptr->SetShape(shape_desc);
  output0_desc_ptr->SetOriginDataType(set_dtype);
  output0_desc_ptr->SetOriginFormat(set_format);
  output0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddOutputDesc("y", output0_desc_ptr->Clone());
  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("tbe-custom", "UnknownShape");
  SubOpsStorePtr sub_ops_store_ptr = std::make_shared<SubOpsStore>(AI_CORE_NAME);
  std::string un_supported_reason;
  tbe_adapter_ptr_->SelectTbeOpFormat = SelectOpFormatStub;
  bool ret = fe_ops_store_ptr->CheckSupported(op_desc_ptr_t, un_supported_reason);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(ge::AttrUtils::HasAttr(op_desc_ptr_t, ATTR_NAME_IS_OP_DYNAMIC_IMPL), true);
  EXPECT_EQ(IsOpDynamicImpl(op_desc_ptr_t), false);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, reshape_1)
{
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>("reshape_1", "Reshape");
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> output0_desc_ptr = make_shared<ge::GeTensorDesc>();

  ge::DataType set_dtype = ge::DT_FLOAT16;
  ge::Format set_format = ge::FORMAT_NCHW;
  std::vector<int64_t> shape_vec{4, 16, 100, 100};
  ge::GeShape shape_desc = GeShape(shape_vec);

  input0_desc_ptr->SetDataType(set_dtype);
  input0_desc_ptr->SetFormat(set_format);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetOriginDataType(set_dtype);
  input0_desc_ptr->SetOriginFormat(set_format);
  input0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddInputDesc("a", input0_desc_ptr->Clone());
  op_desc_ptr_t->AddInputDesc("b", input0_desc_ptr->Clone());
  op_desc_ptr_t->AddInputDesc("c", input0_desc_ptr->Clone());

  output0_desc_ptr->SetDataType(set_dtype);
  output0_desc_ptr->SetFormat(set_format);
  output0_desc_ptr->SetShape(shape_desc);
  output0_desc_ptr->SetOriginDataType(set_dtype);
  output0_desc_ptr->SetOriginFormat(set_format);
  output0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddOutputDesc("z", output0_desc_ptr->Clone());

  std::string un_supported_reason;
  bool ret = fe_ops_store_ptr->CheckSupported(op_desc_ptr_t, un_supported_reason);
  EXPECT_EQ(ret, false);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, reshape_2)
{
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>("reshape_2", "Reshape");
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> output0_desc_ptr = make_shared<ge::GeTensorDesc>();

  ge::DataType set_dtype = ge::DT_FLOAT16;
  ge::Format set_format = ge::FORMAT_NCHW;
  std::vector<int64_t> shape_vec{4, 16, 100, 100};
  ge::GeShape shape_desc = GeShape(shape_vec);

  input0_desc_ptr->SetDataType(set_dtype);
  input0_desc_ptr->SetFormat(set_format);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetOriginDataType(set_dtype);
  input0_desc_ptr->SetOriginFormat(set_format);
  input0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());
  op_desc_ptr_t->AddInputDesc("z", input0_desc_ptr->Clone());

  output0_desc_ptr->SetDataType(set_dtype);
  output0_desc_ptr->SetFormat(set_format);
  output0_desc_ptr->SetShape(shape_desc);
  output0_desc_ptr->SetOriginDataType(set_dtype);
  output0_desc_ptr->SetOriginFormat(set_format);
  output0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddOutputDesc("d", output0_desc_ptr->Clone());
  op_desc_ptr_t->AddOutputDesc("e", output0_desc_ptr->Clone());

  std::string un_supported_reason;
  bool ret = fe_ops_store_ptr->CheckSupported(op_desc_ptr_t, un_supported_reason);
  EXPECT_EQ(ret, false);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, check_dtype_by_allow_fp32_to_fp16_suc)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
  shared_ptr<ge::GeTensorDesc> input_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> output_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>("matmul", "MatMulV10");
  /* Set list int value as [6, 7, 8], which is not supported  */
  SetOpDescPtrAttrValue(TEST_SUCCESS, op_desc_ptr_t);

  ge::DataType set_dtype = ge::DT_FLOAT;
  ge::Format set_format = ge::FORMAT_ND;
  std::vector<int64_t> shape_vec{256,256,512};
  ge::GeShape shape_desc = GeShape(shape_vec);

  vector<int64_t> tensorShape = {1,1,3,1};
  GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, ge::DT_FLOAT);
  input_ptr->SetDataType(set_dtype);
  input_ptr->SetFormat(set_format);
  input_ptr->SetShape(shape_desc);
  output_ptr->SetDataType(set_dtype);
  output_ptr->SetFormat(set_format);
  output_ptr->SetShape(shape_desc);
  op_desc_ptr_t->AddInputDesc("x1", tensor1);
  op_desc_ptr_t->AddOutputDesc("y", tensor1);
  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("cce-custom", "MatMulV10");
  EXPECT_NE(nullptr, op_kernel_info_ptr);
  InputOrOutputInfoPtr input_info_ptr;
  InputOrOutputInfoPtr output_info_ptr;
  op_kernel_info_ptr->GetInputInfoByName("x1", input_info_ptr);
  op_kernel_info_ptr->GetOutputInfoByName("y", output_info_ptr);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr test_node = graph->AddNode(op_desc_ptr_t);
  FormatDtypeInfo format_dtype_info;
  Status get_format_dtype_status = format_dtype_querier_ptr_->GetSupportFormatAndDtype(op_kernel_info_ptr,
          test_node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, get_format_dtype_status);
  for (auto dtype : format_dtype_info.data_type_map.at(input_info_ptr->GetUniqueName())) {
    std::cout << "current dtype is " << dtype << std::endl;
    std::cout << "input_info_ptr uniqname is " << input_info_ptr->GetUniqueName() << std::endl;
  }
  ge::GeTensorDescPtr input_desc = op_desc_ptr_t->MutableInputDesc(0);
  ge::GeTensorDescPtr output_desc = op_desc_ptr_t->MutableOutputDesc(0);

  SupportedFormatAndDtype check_info(op_kernel_info_ptr, "");
  bool ret = sub_ops_store_ptr->CheckDtypeSupported(test_node, input_desc, input_info_ptr,
                                                    format_dtype_info.data_type_map.at(input_info_ptr->GetUniqueName()),
                                                    check_info);
  ret = sub_ops_store_ptr->CheckDtypeSupported(test_node, output_desc, output_info_ptr,
                                               format_dtype_info.data_type_map.at(output_info_ptr->GetUniqueName()),
                                               check_info);
  EXPECT_EQ(ret, true);

  bool has_need_update_dtype_flag = false;
  has_need_update_dtype_flag = ge::AttrUtils::GetBool(input_desc,
                              NEED_UPDATE_DTYPE_WHEN_OP_CHECKSUPPORT, has_need_update_dtype_flag);
  EXPECT_EQ(true, has_need_update_dtype_flag);
  has_need_update_dtype_flag = ge::AttrUtils::GetBool(output_desc,
                              NEED_UPDATE_DTYPE_WHEN_OP_CHECKSUPPORT, has_need_update_dtype_flag);
  EXPECT_EQ(true, has_need_update_dtype_flag);
  for (const auto &tensor : op_desc_ptr_t->GetAllInputsDesc()) {
    bool flag = ge::AttrUtils::GetBool(tensor,
                              NEED_UPDATE_DTYPE_WHEN_OP_CHECKSUPPORT, flag);
    if (flag) {
      std::cout << "get NEED_UPDATE_DTYPE_WHEN_OP_CHECKSUPPORT" << std::endl;
    } else {
      std::cout << "not get NEED_UPDATE_DTYPE_WHEN_OP_CHECKSUPPORT" << std::endl;
    }
  }
  std::pair<std::vector<size_t>, std::vector<size_t>> in_out_changed_idx_vec;
  tbe_adapter_ptr_->UpdateTensorByMixPrecisionMode(test_node, op_kernel_info_ptr, in_out_changed_idx_vec);
  bool all_dtype_fp16_flag = false;
  EXPECT_EQ(ge::DT_FLOAT16, op_desc_ptr_t->MutableInputDesc("x1")->GetDataType());
  EXPECT_EQ(ge::DT_FLOAT16, op_desc_ptr_t->MutableOutputDesc("y")->GetDataType());
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, get_all_atomic_clean_node_suc)
{
  ge::OpDescPtr conv_op = std::make_shared<ge::OpDesc>("Conv", "conv");
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr node_ptr = graph->AddNode(conv_op);
  vector<ge::NodePtr> atomic_node_vec;
  Status ret = fe_ops_store_ptr->GetAllAtomicCleanNode(node_ptr, atomic_node_vec);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, compile_op_get_tvm_json_info_failed)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("data", "Data");
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_0->AddOutputDesc(out_desc);
  NodePtr node_0 = graph->AddNode(op_desc_0);

  ScopeNodeIdMap fusion_nodes_map;
  std::vector<ge::Node*> fusion_nodes;
  fusion_nodes.push_back(node_0.get());
  fusion_nodes_map.emplace(std::make_pair(1, fusion_nodes));
  CompileResultMap compile_ret_map;
  Status ret = fe_ops_store_ptr->CompileOpGetTvmJsonInfo(fusion_nodes_map, compile_ret_map);
  EXPECT_EQ(ret, OP_COMPILER_CHECK_FALSE_FAILED);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, pre_compile_and_compile_success)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_0->AddOutputDesc(out_desc);
  std::unordered_map<OpStoreAdapterPtr, vector<PreCompileNodePara>> node_map;
  NodePtr node_0 = graph->AddNode(op_desc_0);
  ScopeNodeIdMap fusion_node_map;
  Status ret = fe_ops_store_ptr->PreCompileAndCompile(node_map, node_0, fusion_node_map, false);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, compile_single_op_failed)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_0->AddOutputDesc(out_desc);
  NodePtr node_0 = graph->AddNode(op_desc_0);
  Status ret = fe_ops_store_ptr->CompileSingleOp(node_0);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, update_op_info_store_success)
{
  FEOpsStoreInfo tbe_custom {
  2,
  "tbe-custom",
  EN_IMPL_CUSTOM_TBE,
  GetCodeDir() + "/tests/engines/nn_engine/st/testcase/ops_kernel_store/fe_config/tbe_custom_opinfo",
  ""};
  shared_ptr<fe::SubOpInfoStore> sub_ops_kernel_ptr = std::make_shared<fe::SubOpInfoStore>(tbe_custom);
  std::map<std::string, OpContent> op_content_map;
  OpContent op_content;
  EXPECT_EQ(sub_ops_kernel_ptr->GetOpContentByOpType("conv", op_content), fe::FAILED);
  sub_ops_kernel_ptr->op_content_map_.emplace(std::make_pair("conv", op_content));
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, finalize_opkernel_info_test) {
  OpKernelInfoPtr op_kernel_info = std::make_shared<OpKernelInfo>("FrameworkOp");
  InputOrOutputInfoPtr input_info_ptr = nullptr;
  op_kernel_info->input_infos_.push_back(input_info_ptr);
  InputOrOutputInfoPtr output_info_ptr = std::make_shared<InputOrOutputInfo>("y");
  op_kernel_info->output_infos_.push_back(output_info_ptr);
  OpKernelInfoConstructor op_kernel_info_constructor;
  Status ret = op_kernel_info_constructor.FinalizeOpKernelInfo(op_kernel_info);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, finalize_opkernel_info_test1) {
  OpKernelInfoPtr op_kernel_info = std::make_shared<OpKernelInfo>("FrameworkOp");
  // InputOrOutputInfoPtr input_info_ptr = std::make_shared<InputOrOutputInfoPtr>("x");
  // op_kernel_info->input_infos_.push_back(input_info_ptr);
  InputOrOutputInfoPtr output_info_ptr = nullptr;
  op_kernel_info->output_infos_.push_back(output_info_ptr);
  OpKernelInfoConstructor op_kernel_info_constructor;
  Status ret = op_kernel_info_constructor.FinalizeOpKernelInfo(op_kernel_info);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, compile_and_set_kernel_name_for_memset){
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    OpDescPtr op_desc_0 = std::make_shared<OpDesc>("data", "MatMul");
    vector<int64_t> dim(4, 4);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);
    op_desc_0->AddOutputDesc(out_desc);
    ge::NodePtr test_node1 = graph->AddNode(op_desc_0);
    ge::NodePtr test_node2 = graph->AddNode(op_desc_0);
    std::string core_type = "vector_core";
    ge::AttrUtils::SetStr(test_node1->GetOpDesc(), "_cube_vector_core_type", core_type);
    ge::AttrUtils::SetStr(test_node1->GetOpDesc(), "compile_info_json", "compile_info_json");
    ge::AttrUtils::SetStr(test_node1->GetOpDesc(), "compile_info_key", "compile_info_key");
    ge::AttrUtils::SetInt(test_node1->GetOpDesc(), "globalworkspace_size", 8);
    ge::AttrUtils::SetInt(test_node1->GetOpDesc(), "globalworkspace_type", 0);
    test_node2->GetOpDesc()->SetExtAttr(fe::ATTR_NAME_MEMSET_NODE, test_node1);
    test_node2->GetOpDesc()->SetExtAttr(ATTR_NAME_SUPPORT_DYNAMIC_SHAPE, false);
    std::string kernel_name = "te_matmul";
    ge::AttrUtils::SetStr(test_node2->GetOpDesc(), "_kernelname", kernel_name);
    ge::AttrUtils::SetStr(test_node2->GetOpDesc(), ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    ge::AttrUtils::SetStr(test_node2->GetOpDesc(), "_unregst_oppath", "../../abs.py");
    std::vector<uint32_t> tmp_output_index {1, 1, 2, 1};
    ge::AttrUtils::SetListInt(test_node2->GetOpDesc(), TBE_OP_ATOMIC_OUTPUT_INDEX, tmp_output_index);
    const char tbe_bin[] = "tbe_bin";
    std::vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<OpKernelBin>(test_node2->GetName(), std::move(buffer));
    test_node2->GetOpDesc()->SetExtAttr("tbeKernel", tbe_kernel_ptr);
    std::vector<ge::NodePtr> node_vec;
    std::vector<ge::NodePtr> atomic_memset_nodes;
    node_vec.push_back(test_node2);
    atomic_memset_nodes.push_back(test_node2);
    shared_ptr<FEOpsKernelInfoStore> ops_store = make_shared<FEOpsKernelInfoStore>();
    Status ret = ops_store->CompileAndSetKernelNameForMemSet(node_vec, atomic_memset_nodes);
    if ((op_desc_0->HasAttr(op_desc_0->GetName() + "_atomic_kernelname")) &&
        (op_desc_0->HasAttr(ge::ATTR_NAME_TBE_KERNEL_NAME_FOR_LOAD))) {
    ret = fe::FAILED;
    }
    EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, set_workspace_info_for_memset){
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("data", "Data");
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_0->AddOutputDesc(out_desc);
  NodePtr node_0 = graph->AddNode(op_desc_0);
  vector<uint32_t> output_index = {1};
  (void)ge::AttrUtils::SetListInt(op_desc_0, TBE_OP_ATOMIC_OUTPUT_INDEX, output_index);
  std::map<int64_t, int64_t> workspace_info{{0, 20}};
  std::map<std::string, std::map<int64_t, int64_t>> sub_node_workspace_info;
  sub_node_workspace_info.insert(std::make_pair(op_desc_0->GetName(), workspace_info));
  op_desc_0->SetExtAttr(ge::EXT_ATTR_ATOMIC_WORKSPACE_INFO, sub_node_workspace_info);

  std::vector<int64_t> dtype_list = {0, 2};
  vector<int64_t> init_value_int64_list = {1};
  vector<float> init_value_float_list = {1.1};
  (void)ge::AttrUtils::SetListInt(op_desc_0, TBE_OP_ATOMIC_DTYPES, dtype_list);
  (void)ge::AttrUtils::SetListInt(op_desc_0, TBE_OP_ATOMIC_INT64_VALUES, init_value_int64_list);
  (void)ge::AttrUtils::SetListFloat(op_desc_0, TBE_OP_ATOMIC_FLOAT_VALUES, init_value_float_list);
  (void)ge::AttrUtils::SetBool(op_desc_0, kStaticToDynamicSoftSyncOp, true);
  vector<ge::NodePtr> node_vec = {node_0};
  Status ret = fe_ops_kernel_info_store_ptr->CompileMemSet(node_vec);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, fuzzy_compile_op_fail) {
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    OpDescPtr dy_op = std::make_shared<OpDesc>("dynamicShape", "DynamicShape");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    vector<int64_t> dims = {1, 2, 3, 4};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetOriginFormat(FORMAT_NCHW);
    in_desc1.SetOriginShape(shape);
    in_desc1.SetDataType(DT_FLOAT16);
    dy_op->AddInputDesc("x", in_desc1);
    data->AddOutputDesc("x", in_desc1);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetOriginFormat(FORMAT_HWCN);
    out_desc1.SetOriginShape(shape);
    out_desc1.SetDataType(DT_FLOAT16);
    dy_op->AddOutputDesc("z", out_desc1);

    ge::AttrUtils::SetStr(dy_op, ge::ATTR_NAME_UNREGST_OPPATH, "./impl/abc");
    ge::AttrUtils::SetInt(dy_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_CUSTOM_TBE));
    ge::AttrUtils::SetStr(dy_op, "relu_kernelname", "relu_build");
    ge::AttrUtils::SetBool(dy_op, ATTR_NAME_SUPPORT_DYNAMIC_SHAPE, false);
    NodePtr relu_node = graph->AddNode(dy_op);
    NodePtr data_node = graph->AddNode(data);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));

    tbe_adapter_ptr_->CheckIsTbeGeneralizeFuncRegistered = checkIsRegistered;
    tbe_adapter_ptr_->TeGeneralize = teGeneralize;
    std::vector<ge::NodePtr> node_vec{};
    node_vec.push_back(relu_node);
    Status ret = fe_ops_kernel_info_store_ptr->FuzzCompileOp(node_vec);
    EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, fuzzy_compile_fusionop_success) {
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    OpDescPtr dy_op = std::make_shared<OpDesc>("DynamicShape", "DynamicShape");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    vector<int64_t> dims = {1, 2, 3, 4};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetOriginFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    in_desc1.SetOriginShape(shape);
    dy_op->AddInputDesc("x", in_desc1);
    data->AddInputDesc("x", in_desc1);
    data->AddOutputDesc("y", in_desc1);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetOriginFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    out_desc1.SetOriginShape(shape);
    dy_op->AddOutputDesc("z", out_desc1);

    NodePtr dy_node = graph->AddNode(dy_op);
    NodePtr data_node = graph->AddNode(data);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), dy_node->GetInDataAnchor(0));
    ge::AttrUtils::SetInt(data, ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    ge::ComputeGraphPtr fusion_graph = std::make_shared<ge::ComputeGraph>("fusionop");
    OpDescPtr test_op = std::make_shared<OpDesc>("test", "test");
    test_op->AddInputDesc("x", in_desc1);
    test_op->AddOutputDesc("y", out_desc1);
    NodePtr test_node = fusion_graph->AddNode(test_op);
    AttrUtils::SetGraph(test_node->GetOpDesc(), "_original_fusion_graph", graph);
    ge::AttrUtils::SetBool(dy_op, ATTR_NAME_SUPPORT_DYNAMIC_SHAPE, false);

    OptimizeUtilityStubST *optimize_utility_stub = new OptimizeUtilityStubST();
    fe_ops_kernel_info_store_ptr->SetGeneralizeRelatedParam(optimize_utility_stub, nullptr);
    tbe_adapter_ptr_->engine_name_ = fe::AI_CORE_NAME;
    tbe_adapter_ptr_->CheckIsTbeGeneralizeFuncRegistered = checkIsRegistered;
    tbe_adapter_ptr_->TeGeneralize = teGeneralize;
    std::vector<ge::NodePtr> node_vec{};
    node_vec.push_back(test_node);
    Status ret = fe_ops_kernel_info_store_ptr->FuzzCompileOp(node_vec);
    EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, update_diff_shape_change) {
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
    vector<int64_t> dims = {-1, -1, -1, 4};
    GeShape shape(dims);
    GeTensorDesc in_desc;
    in_desc.SetShape(shape);
    in_desc.SetOriginShape(shape);
    in_desc.SetFormat(FORMAT_ND_RNN_BIAS);
    in_desc.SetOriginFormat(FORMAT_ND);
    in_desc.SetDataType(DT_FLOAT16);
    in_desc.SetOriginShapeRange({{2,3},{1,3},{4,15},{4,4}});
    relu_op->AddInputDesc("x", in_desc);
    NodePtr relu_node = graph->AddNode(relu_op);
    fe_ops_kernel_info_store_ptr->UpdateNodeShapeAndRange(relu_node);

    vector<int64_t> res_dims = {-1, -1, -1, 64};
    std::vector<std::pair<int64_t, int64_t>> ori_shape_range;
    std::vector<std::pair<int64_t, int64_t>> res_shape_range = {{2, 3}, {1, 3}, {4, 15}, {64, 64}};
    EXPECT_EQ(relu_node->GetOpDesc()->MutableInputDesc(0)->GetShape().GetDims(), res_dims);
    relu_node->GetOpDesc()->MutableInputDesc(0)->GetShapeRange(ori_shape_range);
    EXPECT_EQ(ori_shape_range, res_shape_range);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, dim_shape_check_support_success) {
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    OpDescPtr dy_op = std::make_shared<OpDesc>("dynamicShape", "DynamicShape");
    vector<int64_t> dims = {-2};
    GeShape shape(dims);
    GeTensorDesc desc;
    desc.SetShape(shape);
    dy_op->AddInputDesc("x", desc);
    dy_op->AddOutputDesc("y", desc);
    NodePtr relu_node = graph->AddNode(dy_op);
    std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
    graph_options["ge.shape_generalized_build_mode"] = "shape_generalized";
    GetThreadLocalContext().SetGraphOption(graph_options);

    tbe_adapter_ptr_->engine_name_ = fe::AI_CORE_NAME;
    tbe_adapter_ptr_->CheckTbeSupported = CheckTbeSupportedReasonRange;

    CheckSupportMode check_mode = CheckSupportMode::DTYPE_FORMAT_MODE;
    OpImplType impl_type = EN_IMPL_HW_TBE;
    CheckSupportParam check_param;
    bool res = fe_ops_kernel_info_store_ptr->CheckSupportedByOpsStore(relu_node, check_mode,
                                                                      check_param, impl_type);
    EXPECT_TRUE(res);
    EXPECT_TRUE(AttrUtils::HasAttr(relu_node->GetOpDesc(), kOpShapeOrRangeUnsupport));
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, dim_shape_check_support_fail) {
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    OpDescPtr dy_op = std::make_shared<OpDesc>("dynamicShape", "DynamicShape");
    vector<int64_t> dims = {-2};
    GeShape shape(dims);
    GeTensorDesc desc;
    desc.SetShape(shape);
    dy_op->AddInputDesc("x", desc);
    dy_op->AddOutputDesc("y", desc);
    NodePtr relu_node = graph->AddNode(dy_op);
    std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
    graph_options["ge.shape_generalized_build_mode"] = "shape_generalized";
    GetThreadLocalContext().SetGraphOption(graph_options);

    tbe_adapter_ptr_->engine_name_ = fe::AI_CORE_NAME;
    tbe_adapter_ptr_->CheckTbeSupported = CheckTbeSupportedOtherReason;

    CheckSupportMode check_mode = CheckSupportMode::DTYPE_FORMAT_MODE;
    OpImplType impl_type = EN_IMPL_HW_TBE;
    CheckSupportParam check_param;
    bool res = fe_ops_kernel_info_store_ptr->CheckSupportedByOpsStore(relu_node, check_mode,
                                                                      check_param, impl_type);
    EXPECT_FALSE(res);
    EXPECT_FALSE(AttrUtils::HasAttr(relu_node->GetOpDesc(), kOpShapeOrRangeUnsupport));
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, dynamic_shape_check_support_success) {
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    OpDescPtr dy_op = std::make_shared<OpDesc>("dynamicShape", "DynamicShape");
    vector<int64_t> dims = {-1};
    GeShape shape(dims);
    GeTensorDesc desc;
    desc.SetShape(shape);
    dy_op->AddInputDesc("x", desc);
    dy_op->AddOutputDesc("y", desc);
    NodePtr relu_node = graph->AddNode(dy_op);

    tbe_adapter_ptr_->engine_name_ = fe::AI_CORE_NAME;
    tbe_adapter_ptr_->CheckTbeSupported = CheckTbeSupportedReasonRange;

    CheckSupportMode check_mode = CheckSupportMode::DTYPE_FORMAT_MODE;
    OpImplType impl_type = EN_IMPL_HW_TBE;
    CheckSupportParam check_param;
    bool res = fe_ops_kernel_info_store_ptr->CheckSupportedByOpsStore(relu_node, check_mode,
                                                                      check_param, impl_type);
    EXPECT_TRUE(res);
    EXPECT_TRUE(AttrUtils::HasAttr(relu_node->GetOpDesc(), kOpShapeOrRangeUnsupport));
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, GetDynamicCustomOpStoreInfoByNode_fail1) {
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    OpDescPtr dy_op = std::make_shared<OpDesc>("dynamicShape", "DynamicShape");
    vector<int64_t> dims = {-1};
    GeShape shape(dims);
    GeTensorDesc desc;
    desc.SetShape(shape);
    dy_op->AddInputDesc("x", desc);
    dy_op->AddOutputDesc("y", desc);
    ge::AttrUtils::SetStr(dy_op, "_custom_op_impl_config_path", "test");
    NodePtr relu_node = graph->AddNode(dy_op);
    vector<std::string> json_files = {"test"};
    SubOpInfoStorePtr op_info_ptr = nullptr;
    Status res = fe_ops_kernel_info_store_ptr->GetDynamicCustomOpStoreInfoByNode(relu_node, json_files, op_info_ptr);
    EXPECT_EQ(res, fe::FAILED);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, GetDynamicCustomOpStoreInfoByNode_fail2) {
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    OpDescPtr dy_op = std::make_shared<OpDesc>("dynamicShape", "NANOP");
    vector<int64_t> dims = {-1};
    GeShape shape(dims);
    GeTensorDesc desc;
    desc.SetShape(shape);
    dy_op->AddInputDesc("x", desc);
    dy_op->AddOutputDesc("y", desc);
    NodePtr relu_node = graph->AddNode(dy_op);
    vector<std::string> json_files = {"test"};
    SubOpInfoStorePtr op_info_ptr = nullptr;
    Status res = fe_ops_kernel_info_store_ptr->GetDynamicCustomOpStoreInfoByNode(relu_node, json_files, op_info_ptr);
    EXPECT_EQ(res, fe::FAILED);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, get_node_support_info_01) {
  vector<int64_t> dims(4, 2);
  GeShape shape(dims);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetOriginShape(shape);

  OpDescPtr op_desc = std::make_shared<OpDesc>("ValueDepend", "ValueDepend");
  op_desc->AddInputDesc("a", tensor_desc);
  op_desc->AddOutputDesc("zz", tensor_desc);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node = graph->AddNode(op_desc);
  ge::Operator op = OpDescUtils::CreateOperatorFromNode(node);
  ge::OperatorPtr op_ptr = std::make_shared<ge::Operator>(op);

  string support_info;
  bool ret = fe_ops_kernel_info_store_ptr->GetNodeSupportInfo(op_ptr, support_info);
  EXPECT_EQ(ret, false);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, get_node_support_info_02) {
  vector<int64_t> dims(4, 2);
  GeShape shape(dims);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetOriginShape(shape);

  OpDescPtr op_desc = std::make_shared<OpDesc>("ValueDepend", "ValueDepend");
  op_desc->AddInputDesc("a", tensor_desc);
  op_desc->AddOutputDesc("zz", tensor_desc);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node = graph->AddNode(op_desc);
  vector<uint16_t> data_vec(16, 10);
  GeTensorPtr weight_ptr =
          std::make_shared<ge::GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), 16 * sizeof(uint16_t));
  ge::OpDescPtr const_op = ge::OpDescUtils::CreateConstOp(weight_ptr);
  NodePtr const_node = graph->AddNode(const_op);
  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));

  ge::Operator op = OpDescUtils::CreateOperatorFromNode(node);
  ge::OperatorPtr op_ptr = std::make_shared<ge::Operator>(op);

  string support_info;
  bool ret = fe_ops_kernel_info_store_ptr->GetNodeSupportInfo(op_ptr, support_info);
  EXPECT_EQ(ret, true);
  try {
    nlohmann::json op_info_json = nlohmann::json::parse(support_info);
    EXPECT_EQ(op_info_json["is_dynamic_impl"], "false");
    EXPECT_EQ(op_info_json["input0"]["name"], "a");
    EXPECT_EQ(op_info_json["input0"]["param_type"], "required");
    EXPECT_EQ(op_info_json["input0"]["format"], "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0");
    EXPECT_EQ(op_info_json["output0"]["name"], "zz");
    EXPECT_EQ(op_info_json["output0"]["dtype"], "DT_UINT8,DT_INT8,DT_DOUBLE,DT_FLOAT,DT_FLOAT16");
  } catch (nlohmann::json::parse_error &ex) {
    EXPECT_TRUE(false);
  }
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, get_node_support_info_03) {
  vector<int64_t> dims(4, 2);
  GeShape shape(dims);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetOriginShape(shape);

  OpDescPtr op_desc = std::make_shared<OpDesc>("pad", "PadV31");
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node = graph->AddNode(op_desc);
  ge::Operator op = OpDescUtils::CreateOperatorFromNode(node);
  ge::OperatorPtr op_ptr = std::make_shared<ge::Operator>(op);

  string support_info;
  bool ret = fe_ops_kernel_info_store_ptr->GetNodeSupportInfo(op_ptr, support_info);
  EXPECT_EQ(ret, true);
  try {
    nlohmann::json op_info_json = nlohmann::json::parse(support_info);
    EXPECT_EQ(op_info_json["is_dynamic_impl"], "true");
    EXPECT_EQ(op_info_json["input0"]["name"], "x");
    EXPECT_EQ(op_info_json["input0"]["param_type"], "required");
    EXPECT_EQ(op_info_json["input0"]["format"], "NCHW,NC1HWC0,NHWC,ND,HWCN,CHWN,NDHWC,DHWCN,DHWNC");
    EXPECT_EQ(op_info_json["output0"]["name"], "y");
    EXPECT_EQ(op_info_json["output0"]["dtype"], "DT_FLOAT16,DT_FLOAT16,DT_FLOAT16,DT_FLOAT16,DT_FLOAT16,DT_FLOAT16,DT_FLOAT16,DT_FLOAT16,DT_FLOAT16");
  } catch (nlohmann::json::parse_error &ex) {
    EXPECT_TRUE(false);
  }
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, FeedOutputTensorAtomicAttr) {
  ffts::ThreadSliceMapPtr slice_info_ptr = std::make_shared<ffts::ThreadSliceMap>();
  slice_info_ptr->atomic_types = {ffts::AtomicType::ADD};
  te::TbeOpTensor output_tensor = te::TbeOpTensor("x", {3, 3}, "float", "NCHW");
  TbeInfoAssembler t;
  t.FeedOutputTensorAtomicAttr(slice_info_ptr, output_tensor, 0);
  std::string atomic_type;
  output_tensor.GetAtomicType(atomic_type);
  EXPECT_EQ(atomic_type, "add");
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, softsync_op_checksupport01) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr dy_op = std::make_shared<OpDesc>("padv3", "PadV3");
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
  NodePtr dy_node = graph->AddNode(dy_op);

  string un_supported_reason;
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::VirtualType)] = 1;
  bool res = fe_ops_store_ptr->CheckSupported(dy_node, un_supported_reason);
  EXPECT_EQ(res, true);
  bool support_dyn_shape = false;
  bool stc_to_dyn_softsync = false;
  bool is_dyn_impl = false;
  (void)ge::AttrUtils::GetBool(dy_op, ATTR_NAME_SUPPORT_DYNAMIC_SHAPE, support_dyn_shape);
  (void)ge::AttrUtils::GetBool(dy_op, kStaticToDynamicSoftSyncOp, stc_to_dyn_softsync);
  (void)ge::AttrUtils::GetBool(dy_op, ATTR_NAME_IS_OP_DYNAMIC_IMPL, is_dyn_impl);
  EXPECT_EQ(support_dyn_shape, true);
  EXPECT_EQ(stc_to_dyn_softsync, true);
  EXPECT_EQ(is_dyn_impl, true);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, GetOpKernelInfoPtr) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr dy_op = std::make_shared<OpDesc>("padv3", "PadV3");
  NodePtr dy_node = graph->AddNode(dy_op);
  OpKernelInfoPtr op_kernel_info;
  fe_ops_kernel_info_store_ptr->GetOpKernelInfoPtr(dy_node, op_kernel_info);
  bool bres = op_kernel_info == nullptr ? true : false;
  EXPECT_EQ(bres, false);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, SetAttrParamTypeList) {
  ge::OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr conv_node = graph->AddNode(conv_op);
  OpKernelInfoPtr op_kernel_info;
  fe_ops_kernel_info_store_ptr->GetOpKernelInfoPtr(conv_node, op_kernel_info);
  op_kernel_info->input_infos_.clear();
  InputOrOutputInfoPtr input_info_ptr = std::make_shared<InputOrOutputInfo>("x");
  input_info_ptr->op_param_type_ = fe::REQUIRED;
  op_kernel_info->input_infos_.emplace_back(input_info_ptr);
  InputOrOutputInfoPtr input_info_ptr1 = std::make_shared<InputOrOutputInfo>("filter");
  input_info_ptr1->op_param_type_ = fe::REQUIRED;
  op_kernel_info->input_infos_.emplace_back(input_info_ptr1);
  InputOrOutputInfoPtr input_info_ptr2 = std::make_shared<InputOrOutputInfo>("bias");
  input_info_ptr2->op_param_type_ = fe::OPTIONAL;
  op_kernel_info->input_infos_.emplace_back(input_info_ptr2);
  InputOrOutputInfoPtr input_info_ptr3 = std::make_shared<InputOrOutputInfo>("weight");
  input_info_ptr3->op_param_type_ = fe::DYNAMIC;
  op_kernel_info->input_infos_.emplace_back(input_info_ptr3);

  op_kernel_info->output_infos_.clear();
  op_kernel_info->output_infos_.emplace_back(input_info_ptr3);
  op_kernel_info->output_infos_.emplace_back(input_info_ptr2);
  op_kernel_info->output_infos_.emplace_back(input_info_ptr1);

  SupportedFormatAndDtype info(op_kernel_info, "");
  info.input_index_name_map[0] = "x";
  info.input_index_name_map[1] = "filter";
  info.input_index_name_map[2] = "bias";
  info.input_index_name_map[3] = "weight";

  info.output_index_name_map[1] = "weight";
  info.output_index_name_map[2] = "bias";
  info.output_index_name_map[3] = "filter";

  SubOpsStorePtr sub_ops_store_ptr = std::make_shared<SubOpsStore>(AI_CORE_NAME);
  sub_ops_store_ptr->SetAttrParamTypeList(conv_op, op_kernel_info, info);
  std::vector<uint32_t> input_type_list;
  (void)ge::AttrUtils::GetListInt(conv_op, kInputParaTypeList, input_type_list);
  std::vector<uint32_t> input_type_list_bk = {0, 0, 1, 2};
  std::vector<uint32_t> output_type_list;
  (void)ge::AttrUtils::GetListInt(conv_op, kOutputParaTypeList, output_type_list);
  std::vector<uint32_t> output_type_list_bk = {2, 1, 0};
  EXPECT_EQ(output_type_list, output_type_list_bk);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, compile_op_tiling){
  shared_ptr<ge::OpDesc> op_desc_ptr = make_shared<ge::OpDesc>("Add", "Add");
  op_desc_ptr->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  op_desc_ptr->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  op_desc_ptr->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  ge::AttrUtils::SetStr(op_desc_ptr, "compile_info_json", json_str);
  ge::AttrUtils::SetStr(op_desc_ptr, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr test_node = graph->AddNode(op_desc_ptr);
  Status ret = fe_ops_kernel_info_store_ptr->CompileOpTiling(test_node);
  EXPECT_EQ(fe::SUCCESS, ret);

  shared_ptr<ge::OpDesc> op_memset = make_shared<ge::OpDesc>("memset", "MemSet");
  shared_ptr<ge::OpDesc> op_reduce = make_shared<ge::OpDesc>("reduce", "ReduceSumD");
  ge::AttrUtils::SetStr(op_memset, "compile_info_json", json_str);
  ge::AttrUtils::SetStr(op_reduce, "compile_info_json", json_str);
  ge::AttrUtils::SetStr(op_memset, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  ge::AttrUtils::SetStr(op_reduce, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  ge::AttrUtils::SetBool(op_reduce, ATTR_NAME_UNKNOWN_SHAPE, true);
  op_reduce->AddInputDesc(ge::GeTensorDesc(ge::GeShape({-1, -1}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::NodePtr memset_node = graph->AddNode(op_memset);
  ge::NodePtr reduce_node = graph->AddNode(op_reduce);
  ge::GraphUtils::AddEdge(memset_node->GetOutControlAnchor(), reduce_node->GetInControlAnchor());
  ret = fe_ops_kernel_info_store_ptr->CompileOpTiling(memset_node);
  EXPECT_EQ(fe::SUCCESS, ret);
  ret = fe_ops_kernel_info_store_ptr->CompileOpTiling(reduce_node);
  EXPECT_EQ(fe::SUCCESS, ret);

  PlatformUtils::Instance().vir_type_list_ = {2,4,8,16};
  shared_ptr<ge::OpDesc> op_sync = make_shared<ge::OpDesc>("BiasAddGrad", "BiasAddGrad");
  ge::AttrUtils::SetStr(op_sync, "compile_info_json", json_str);
  ge::AttrUtils::SetStr(op_sync, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  ge::AttrUtils::SetBool(op_sync, kStaticToDynamicSoftSyncOp, true);
  ge::NodePtr sync_node = graph->AddNode(op_sync);
  ge::GraphUtils::AddEdge(reduce_node->GetOutControlAnchor(), sync_node->GetInControlAnchor());
  ret = fe_ops_kernel_info_store_ptr->CompileOpTiling(sync_node);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, compile_mix_tiling){
  shared_ptr<ge::OpDesc> op_desc_ptr = make_shared<ge::OpDesc>("Add", "Add");
  op_desc_ptr->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  op_desc_ptr->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  op_desc_ptr->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  ge::AttrUtils::SetStr(op_desc_ptr, "compile_info_json", json_str);
  ge::AttrUtils::SetStr(op_desc_ptr, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_AIC");
  ge::AttrUtils::SetBool(op_desc_ptr, fe::kMixDynamicRatio, true);
  ge::GeAttrValue::NAMED_ATTRS tiling_with_ratio;
  std::vector<std::string> tiling_key_vec = {"0", "1", "2"};
  std::vector<int64_t> c_ratio_vec = {0,0,2};
  std::vector<int64_t> v_ratio_vec = {1,1,1};
  tiling_with_ratio.SetAttr("mix_tiling_key", ge::GeAttrValue::CreateFrom<std::vector<std::string>>(tiling_key_vec));
  tiling_with_ratio.SetAttr("mix_tiling_c_ratio", ge::GeAttrValue::CreateFrom<std::vector<int64_t>>(c_ratio_vec));
  tiling_with_ratio.SetAttr("mix_tiling_v_ratio", ge::GeAttrValue::CreateFrom<std::vector<int64_t>>(v_ratio_vec));
  ge::AttrUtils::SetNamedAttrs(op_desc_ptr, "mix_tiling_with_ratio_attr", tiling_with_ratio);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr test_node = graph->AddNode(op_desc_ptr);
  Status ret = fe_ops_kernel_info_store_ptr->CompileOpTiling(test_node);
  EXPECT_EQ(fe::SUCCESS, ret);

  c_ratio_vec = {1,0,2};
  v_ratio_vec = {0,1,1};
  tiling_with_ratio.SetAttr("mix_tiling_c_ratio", ge::GeAttrValue::CreateFrom<std::vector<int64_t>>(c_ratio_vec));
  tiling_with_ratio.SetAttr("mix_tiling_v_ratio", ge::GeAttrValue::CreateFrom<std::vector<int64_t>>(v_ratio_vec));
  ge::AttrUtils::SetNamedAttrs(op_desc_ptr, "mix_tiling_with_ratio_attr", tiling_with_ratio);
  graph = std::make_shared<ge::ComputeGraph>("test");
  test_node = graph->AddNode(op_desc_ptr);
  ret = fe_ops_kernel_info_store_ptr->CompileOpTiling(test_node);
  EXPECT_EQ(fe::SUCCESS, ret);

  c_ratio_vec = {2,0,2};
  v_ratio_vec = {1,1,1};
  tiling_with_ratio.SetAttr("mix_tiling_c_ratio", ge::GeAttrValue::CreateFrom<std::vector<int64_t>>(c_ratio_vec));
  tiling_with_ratio.SetAttr("mix_tiling_v_ratio", ge::GeAttrValue::CreateFrom<std::vector<int64_t>>(v_ratio_vec));
  ge::AttrUtils::SetNamedAttrs(op_desc_ptr, "mix_tiling_with_ratio_attr", tiling_with_ratio);
  graph = std::make_shared<ge::ComputeGraph>("test");
  test_node = graph->AddNode(op_desc_ptr);
  ret = fe_ops_kernel_info_store_ptr->CompileOpTiling(test_node);
  EXPECT_EQ(fe::SUCCESS, ret);

  c_ratio_vec = {1,0,2};
  v_ratio_vec = {2,1,1};
  tiling_with_ratio.SetAttr("mix_tiling_c_ratio", ge::GeAttrValue::CreateFrom<std::vector<int64_t>>(c_ratio_vec));
  tiling_with_ratio.SetAttr("mix_tiling_v_ratio", ge::GeAttrValue::CreateFrom<std::vector<int64_t>>(v_ratio_vec));
  ge::AttrUtils::SetNamedAttrs(op_desc_ptr, "mix_tiling_with_ratio_attr", tiling_with_ratio);
  (void)ge::AttrUtils::SetBool(op_desc_ptr, kMixIsAiv, true);
  (void)ge::AttrUtils::SetStr(op_desc_ptr, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX");
  graph = std::make_shared<ge::ComputeGraph>("test");
  test_node = graph->AddNode(op_desc_ptr);
  ret = fe_ops_kernel_info_store_ptr->CompileOpTiling(test_node);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, compile_vector_core_mix_tiling){
  shared_ptr<ge::OpDesc> op_desc_ptr = make_shared<ge::OpDesc>("Add", "Add");
  op_desc_ptr->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  op_desc_ptr->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  op_desc_ptr->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  ge::AttrUtils::SetStr(op_desc_ptr, "compile_info_json", json_str);
  ge::AttrUtils::SetStr(op_desc_ptr, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_VECTOR_CORE");
  ge::AttrUtils::SetBool(op_desc_ptr, fe::kMixDynamicRatio, true);
  ge::GeAttrValue::NAMED_ATTRS tiling_with_ratio;
  std::vector<std::string> tiling_key_vec = {"0", "1", "2"};
  std::vector<int64_t> c_ratio_vec = {1,1,1};
  std::vector<int64_t> v_ratio_vec = {1,1,1};
  tiling_with_ratio.SetAttr("mix_tiling_key", ge::GeAttrValue::CreateFrom<std::vector<std::string>>(tiling_key_vec));
  tiling_with_ratio.SetAttr("mix_tiling_c_ratio", ge::GeAttrValue::CreateFrom<std::vector<int64_t>>(c_ratio_vec));
  tiling_with_ratio.SetAttr("mix_tiling_v_ratio", ge::GeAttrValue::CreateFrom<std::vector<int64_t>>(v_ratio_vec));
  ge::AttrUtils::SetNamedAttrs(op_desc_ptr, "mix_tiling_with_ratio_attr", tiling_with_ratio);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr test_node = graph->AddNode(op_desc_ptr);
  ge::AttrUtils::SetStr(op_desc_ptr, kTilingRemoveDuplicates, "0");
  Status ret = fe_ops_kernel_info_store_ptr->CompileOpTiling(test_node);
  EXPECT_EQ(fe::SUCCESS, ret);
  std::string res;
  ge::AttrUtils::GetStr(op_desc_ptr, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, res);
  EXPECT_EQ(res, "MIX_VECTOR_CORE");
  ret = fe_ops_kernel_info_store_ptr->CompileOpTiling(test_node);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, allow_fp32_to_bf16_support_check_failed)
{
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr dy_op = std::make_shared<OpDesc>("padv3", "PadV3");
  vector<int64_t> dims = {1, 2, 3, 4};
  vector<int64_t> dims2 = {1, 1, 3, 4, 16};
  GeShape shape(dims);
  GeShape shape2(dims2);
  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NC1HWC0);
  in_desc1.SetOriginFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT);
  in_desc1.SetOriginDataType(DT_FLOAT);
  in_desc1.SetShape(shape2);
  in_desc1.SetOriginShape(shape);
  dy_op->AddInputDesc("x", in_desc1);
  dy_op->AddOutputDesc("Y", in_desc1);
  NodePtr dy_node = graph->AddNode(dy_op);
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B2");
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_BF16;
  string un_supported_reason;
  bool res = fe_ops_store_ptr->CheckSupported(dy_node, un_supported_reason);
  EXPECT_EQ(res, false);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, allow_fp32_to_bf16_support_check_success)
{
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr dy_op = std::make_shared<OpDesc>("PadBF16", "PadBF16");
  vector<int64_t> dims = {1, 2, 3, 4};
  vector<int64_t> dims2 = {1, 1, 3, 4, 16};
  GeShape shape(dims);
  GeShape shape2(dims2);
  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NC1HWC0);
  in_desc1.SetOriginFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT);
  in_desc1.SetOriginDataType(DT_FLOAT);
  in_desc1.SetShape(shape2);
  in_desc1.SetOriginShape(shape);
  dy_op->AddInputDesc("x", in_desc1);
  dy_op->AddOutputDesc("Y", in_desc1);
  NodePtr dy_node = graph->AddNode(dy_op);
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B2");
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_BF16;
  string un_supported_reason;
  bool res = fe_ops_store_ptr->CheckSupported(dy_node, un_supported_reason);
  EXPECT_EQ(res, true);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, subformat_support_check_suc) {
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["check_subformat.enable"] = "true";
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr dy_op = std::make_shared<OpDesc>("padv32", "PadV32");
  vector<int64_t> dims = {1, 2, 3, 4};
  vector<int64_t> dims2 = {1, 1, 3, 4, 16};
  GeShape shape(dims);
  GeShape shape2(dims2);
  GeTensorDesc in_desc1(shape);
  GeTensorDesc in_desc2(shape);
  in_desc1.SetFormat(FORMAT_NC1HWC0);
  in_desc1.SetOriginFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  in_desc1.SetOriginDataType(DT_FLOAT);
  in_desc1.SetShape(shape2);
  in_desc1.SetOriginShape(shape);
  in_desc2.SetFormat(ge::FORMAT_FRACTAL_Z_3D);
  in_desc2.SetOriginFormat(FORMAT_NCHW);
  in_desc2.SetDataType(DT_FLOAT16);
  in_desc2.SetOriginDataType(DT_FLOAT);
  in_desc2.SetShape(shape2);
  in_desc2.SetOriginShape(shape);
  dy_op->AddInputDesc("x", in_desc1);
  dy_op->AddInputDesc("y", in_desc2);
  dy_op->AddOutputDesc("z", in_desc2);
  ge::AttrUtils::SetInt(dy_op, ATTR_NAME_GROUPS, 2);
  NodePtr dy_node = graph->AddNode(dy_op);

  string un_supported_reason;
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::VirtualType)] = 1;
  bool res = fe_ops_store_ptr->CheckSupported(dy_node, un_supported_reason);
  EXPECT_EQ(res, true);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, subformat_support_check_suc2) {
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["check_subformat.enable"] = "true";
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr dy_op = std::make_shared<OpDesc>("padv33", "PadV33");
  vector<int64_t> dims = {1, 2, 3, 4};
  vector<int64_t> dims2 = {1, 1, 3, 4, 16};
  GeShape shape(dims);
  GeShape shape2(dims2);
  GeTensorDesc in_desc1(shape);
  GeTensorDesc in_desc2(shape);
  in_desc1.SetFormat(FORMAT_NC1HWC0);
  in_desc1.SetOriginFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  in_desc1.SetOriginDataType(DT_FLOAT);
  in_desc1.SetShape(shape2);
  in_desc1.SetOriginShape(shape);
  in_desc2.SetFormat(ge::FORMAT_FRACTAL_Z_3D);
  in_desc2.SetOriginFormat(FORMAT_NCHW);
  in_desc2.SetDataType(DT_FLOAT16);
  in_desc2.SetOriginDataType(DT_FLOAT);
  in_desc2.SetShape(shape2);
  in_desc2.SetOriginShape(shape);
  dy_op->AddInputDesc("x", in_desc1);
  dy_op->AddInputDesc("y", in_desc2);
  dy_op->AddOutputDesc("z", in_desc2);
  ge::AttrUtils::SetInt(dy_op, ATTR_NAME_GROUPS, 2);
  NodePtr dy_node = graph->AddNode(dy_op);

  string un_supported_reason;
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::VirtualType)] = 1;
  bool res = fe_ops_store_ptr->CheckSupported(dy_node, un_supported_reason);
  EXPECT_EQ(res, true);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, subformat_support_check_fail) {
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["check_subformat.enable"] = "true";
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr dy_op = std::make_shared<OpDesc>("padv34", "PadV34");
  vector<int64_t> dims = {1, 2, 3, 4};
  vector<int64_t> dims2 = {1, 1, 3, 4, 16};
  GeShape shape(dims);
  GeShape shape2(dims2);
  GeTensorDesc in_desc1(shape);
  GeTensorDesc in_desc2(shape);
  in_desc1.SetFormat(FORMAT_NC1HWC0);
  in_desc1.SetOriginFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  in_desc1.SetOriginDataType(DT_FLOAT);
  in_desc1.SetShape(shape2);
  in_desc1.SetOriginShape(shape);
  in_desc2.SetFormat(ge::FORMAT_FRACTAL_Z_3D);
  in_desc2.SetOriginFormat(FORMAT_FRACTAL_Z_3D);
  in_desc2.SetDataType(DT_FLOAT16);
  in_desc2.SetOriginDataType(DT_FLOAT);
  in_desc2.SetShape(shape2);
  in_desc2.SetOriginShape(shape);
  dy_op->AddInputDesc("x", in_desc1);
  dy_op->AddInputDesc("y", in_desc2);
  dy_op->AddOutputDesc("z", in_desc2);
  ge::AttrUtils::SetInt(dy_op, ATTR_NAME_GROUPS, 2);
  NodePtr dy_node = graph->AddNode(dy_op);

  string un_supported_reason;
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::VirtualType)] = 1;
  bool res = fe_ops_store_ptr->CheckSupported(dy_node, un_supported_reason);
  EXPECT_EQ(res, false);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, subformat_support_check_fail2) {
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["check_subformat.enable"] = "true";
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr dy_op = std::make_shared<OpDesc>("padv33", "PadV33");
  vector<int64_t> dims = {1, 2, 3, 4};
  vector<int64_t> dims2 = {1, 1, 3, 4, 16};
  GeShape shape(dims);
  GeShape shape2(dims2);
  GeTensorDesc in_desc1(shape);
  GeTensorDesc in_desc2(shape);
  in_desc1.SetFormat(FORMAT_NC1HWC0);
  in_desc1.SetOriginFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  in_desc1.SetOriginDataType(DT_FLOAT);
  in_desc1.SetShape(shape2);
  in_desc1.SetOriginShape(shape);
  in_desc2.SetFormat(ge::FORMAT_FRACTAL_Z_3D);
  in_desc2.SetOriginFormat(FORMAT_NCHW);
  in_desc2.SetDataType(DT_FLOAT16);
  in_desc2.SetOriginDataType(DT_FLOAT);
  in_desc2.SetShape(shape2);
  in_desc2.SetOriginShape(shape);
  dy_op->AddInputDesc("x", in_desc1);
  dy_op->AddInputDesc("y", in_desc2);
  dy_op->AddOutputDesc("z", in_desc2);
  ge::AttrUtils::SetInt(dy_op, ATTR_NAME_GROUPS, 65536);
  NodePtr dy_node = graph->AddNode(dy_op);

  string un_supported_reason;
  bool res = fe_ops_store_ptr->CheckSupported(dy_node, un_supported_reason);
  EXPECT_EQ(res, false);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, check_dtype_by_mix_bf16_suc)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION_BF16;
  shared_ptr<ge::GeTensorDesc> input_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> output_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>("maxpool", "MaxPool_0");
  /* Set list int value as [6, 7, 8], which is not supported  */
  SetOpDescPtrAttrValue(TEST_SUCCESS, op_desc_ptr_t);

  ge::DataType set_dtype = ge::DT_FLOAT;
  ge::Format set_format = ge::FORMAT_ND;
  std::vector<int64_t> shape_vec{256,256,512};
  ge::GeShape shape_desc = GeShape(shape_vec);

  vector<int64_t> tensorShape = {1,1,3,1};
  GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, ge::DT_FLOAT);
  input_ptr->SetDataType(set_dtype);
  input_ptr->SetFormat(set_format);
  input_ptr->SetShape(shape_desc);
  output_ptr->SetDataType(set_dtype);
  output_ptr->SetFormat(set_format);
  output_ptr->SetShape(shape_desc);
  op_desc_ptr_t->AddInputDesc("x", tensor1);
  op_desc_ptr_t->AddOutputDesc("y", tensor1);
  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("cce-custom", "MaxPool_0");
  EXPECT_NE(nullptr, op_kernel_info_ptr);
  InputOrOutputInfoPtr input_info_ptr;
  InputOrOutputInfoPtr output_info_ptr;
  op_kernel_info_ptr->GetInputInfoByName("x", input_info_ptr);
  op_kernel_info_ptr->GetOutputInfoByName("y", output_info_ptr);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr test_node = graph->AddNode(op_desc_ptr_t);
  FormatDtypeInfo format_dtype_info;
  Status get_format_dtype_status = format_dtype_querier_ptr_->GetSupportFormatAndDtype(op_kernel_info_ptr,
          test_node, false, format_dtype_info);
  EXPECT_EQ(fe::SUCCESS, get_format_dtype_status);
  ge::GeTensorDescPtr input_desc = op_desc_ptr_t->MutableInputDesc(0);
  ge::GeTensorDescPtr output_desc = op_desc_ptr_t->MutableOutputDesc(0);

  SupportedFormatAndDtype check_info(op_kernel_info_ptr, "");
  bool ret = sub_ops_store_ptr->CheckDtypeSupported(test_node, input_desc, input_info_ptr,
                                                    format_dtype_info.data_type_map.at(input_info_ptr->GetUniqueName()),
                                                    check_info);
  ret = sub_ops_store_ptr->CheckDtypeSupported(test_node, output_desc, output_info_ptr,
                                               format_dtype_info.data_type_map.at(output_info_ptr->GetUniqueName()),
                                               check_info);
  EXPECT_EQ(ret, true);

  bool has_need_update_dtype_flag = false;
  has_need_update_dtype_flag = ge::AttrUtils::GetBool(input_desc,
                              NEED_UPDATE_DTYPE_WHEN_OP_CHECKSUPPORT, has_need_update_dtype_flag);
  EXPECT_EQ(true, has_need_update_dtype_flag);
  has_need_update_dtype_flag = ge::AttrUtils::GetBool(output_desc,
                              NEED_UPDATE_DTYPE_WHEN_OP_CHECKSUPPORT, has_need_update_dtype_flag);
  EXPECT_EQ(true, has_need_update_dtype_flag);
  std::pair<std::vector<size_t>, std::vector<size_t>> in_out_changed_idx_vec;
  tbe_adapter_ptr_->UpdateTensorByMixPrecisionMode(test_node, op_kernel_info_ptr, in_out_changed_idx_vec);
  EXPECT_EQ(ge::DT_BF16, op_desc_ptr_t->MutableInputDesc("x")->GetDataType());
  EXPECT_EQ(ge::DT_FLOAT, op_desc_ptr_t->MutableOutputDesc("y")->GetDataType());
  OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
  op_setter_ptr->format_dtype_querier_ptr_ = format_dtype_querier_ptr_;
  bool bres = op_setter_ptr->SetAclnnAttr(test_node, 1, op_kernel_info_ptr, fe::PrecisionMode::ENUM_ALLOW_FP32_TO_FP16);
  EXPECT_EQ(bres, true);
  OpKernelInfoPtr op_kernel_info_ptr_1 =
      OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType("cce-custom", "MaxPool_fallback");
  bres = op_setter_ptr->SetAclnnAttr(test_node, 1, op_kernel_info_ptr_1, fe::PrecisionMode::ENUM_ALLOW_FP32_TO_FP16);
  EXPECT_EQ(bres, false);
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, test_set_hashed_extram_params)
{
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>("test", "AscBackend");
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> output0_desc_ptr = make_shared<ge::GeTensorDesc>();

  ge::DataType set_dtype = ge::DT_BOOL;
  ge::Format set_format = ge::FORMAT_NCHW;
  std::vector<int64_t> shape_vec{4, 16, 100, 100};
  ge::GeShape shape_desc = GeShape(shape_vec);

  input0_desc_ptr->SetDataType(set_dtype);
  input0_desc_ptr->SetFormat(set_format);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetOriginDataType(set_dtype);
  input0_desc_ptr->SetOriginFormat(set_format);
  input0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

  output0_desc_ptr->SetDataType(set_dtype);
  output0_desc_ptr->SetFormat(set_format);
  output0_desc_ptr->SetShape(shape_desc);
  output0_desc_ptr->SetOriginDataType(set_dtype);
  output0_desc_ptr->SetOriginFormat(set_format);
  output0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddOutputDesc("y", output0_desc_ptr->Clone());

  op_desc_ptr_t->SetExtAttr("_extra_param_builder",
    std::function<std::string()>([]() -> std::string {
    std::string output = "{name:Relu_1_out0_asc}";
    return output;
  }));

  std::string hashed_extra_param = "{name:Relu_1_out0_asc}{type:Data}";
  op_desc_ptr_t->SetExtAttr("_hashed_extra_param_builder", hashed_extra_param);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
  auto node = graph->AddNode(op_desc_ptr_t);
  shared_ptr<ge::OpDesc> data1 = make_shared<ge::OpDesc>("data1", DATA);
  (void)ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  auto node_data = graph->AddNode(data1);
  TbeInfoAssembler t;
  t.Initialize();
  te::TbeOpInfo info("test",
                     "test1",
                     "AscBackend",
                     fe::AI_CORE_NAME);
  (void)ge::AttrUtils::SetBool(op_desc_ptr_t, ge::ATTR_NAME_DISABLE_ATTACHED_RESOURCE, true);
  Status ret = t.AssembleAutoFuseTbeInfo(node.get(), info);
  EXPECT_EQ(ret, fe::SUCCESS);
  EXPECT_EQ(info.GetExtraParams(), "{name:Relu_1_out0_asc}");
  EXPECT_EQ(info.GetHashedExtraParams(), "{name:Relu_1_out0_asc}{type:Data}");
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, test_set_op_impl_switch)
{
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>("test", "AscBackend");
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> output0_desc_ptr = make_shared<ge::GeTensorDesc>();

  ge::DataType set_dtype = ge::DT_BOOL;
  ge::Format set_format = ge::FORMAT_NCHW;
  std::vector<int64_t> shape_vec{4, 16, 100, 100};
  ge::GeShape shape_desc = GeShape(shape_vec);

  input0_desc_ptr->SetDataType(set_dtype);
  input0_desc_ptr->SetFormat(set_format);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetOriginDataType(set_dtype);
  input0_desc_ptr->SetOriginFormat(set_format);
  input0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

  output0_desc_ptr->SetDataType(set_dtype);
  output0_desc_ptr->SetFormat(set_format);
  output0_desc_ptr->SetShape(shape_desc);
  output0_desc_ptr->SetOriginDataType(set_dtype);
  output0_desc_ptr->SetOriginFormat(set_format);
  output0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddOutputDesc("y", output0_desc_ptr->Clone());

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
  auto node = graph->AddNode(op_desc_ptr_t);
  TbeInfoAssembler t;
  t.Initialize();
  te::TbeOpInfo info("test",
                     "test1",
                     "AscBackend",
                     fe::AI_CORE_NAME);
  string op_impl_switch = "";
  (void)ge::AttrUtils::SetStr(op_desc_ptr_t, kAttrOpImplSwitchValue, "has_set");
  t.SetOpImplSwitch(node->GetOpDesc(), info);
  (void)ge::AttrUtils::GetStr(op_desc_ptr_t, kAttrOpImplSwitchValue, op_impl_switch);
  EXPECT_EQ(op_impl_switch, "has_set");
}

TEST_F(STEST_OP_KERNEL_INFO_STORE, test_set_op_custom_op_file_path)
{
  shared_ptr<ge::OpDesc> op_desc_ptr_t = make_shared<ge::OpDesc>("test", "AscBackend");
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  shared_ptr<ge::GeTensorDesc> output0_desc_ptr = make_shared<ge::GeTensorDesc>();

  ge::DataType set_dtype = ge::DT_BOOL;
  ge::Format set_format = ge::FORMAT_NCHW;
  std::vector<int64_t> shape_vec{4, 16, 100, 100};
  ge::GeShape shape_desc = GeShape(shape_vec);

  input0_desc_ptr->SetDataType(set_dtype);
  input0_desc_ptr->SetFormat(set_format);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetOriginDataType(set_dtype);
  input0_desc_ptr->SetOriginFormat(set_format);
  input0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddInputDesc("x", input0_desc_ptr->Clone());

  output0_desc_ptr->SetDataType(set_dtype);
  output0_desc_ptr->SetFormat(set_format);
  output0_desc_ptr->SetShape(shape_desc);
  output0_desc_ptr->SetOriginDataType(set_dtype);
  output0_desc_ptr->SetOriginFormat(set_format);
  output0_desc_ptr->SetOriginShape(shape_desc);
  op_desc_ptr_t->AddOutputDesc("y", output0_desc_ptr->Clone());

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
  auto node = graph->AddNode(op_desc_ptr_t);
  FEOpsStoreInfo ops_store;
  ops_store.fe_ops_store_name = "tbe-custom";
  ops_store.is_custom_store = true;
  ops_store.cfg_file_path = "/home/jenkins/Ascend/ascend-toolkit/latest/opp_vendors/customize/op_impl/ai_core/tbe/config/ascend910";
  shared_ptr<FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr = make_shared<FEOpsKernelInfoStore>();
  fe_ops_kernel_info_store_ptr->GetAndSetCustomOpFilePath(node, ops_store);
  std::string custom_op_file_path;
  (void)ge::AttrUtils::GetStr(node->GetOpDesc(), CUSTOM_OP_FILE_PATH, custom_op_file_path);
  EXPECT_EQ(custom_op_file_path, "/home/jenkins/Ascend/ascend-toolkit/latest/opp_vendors/customize");
  fe_ops_kernel_info_store_ptr->GetAndSetCustomOpFilePath(node, ops_store);

  (void)node->GetOpDesc()->DelAttr(CUSTOM_OP_FILE_PATH);
  FEOpsStoreInfo ops_store1;
  ops_store1.fe_ops_store_name = "tbe-custom1";
  ops_store1.is_custom_store = true;
  ops_store1.cfg_file_path = "/home/jenkins/Ascend/ascend-toolkit/latest/opp_vendors/customize";
  fe_ops_kernel_info_store_ptr->GetAndSetCustomOpFilePath(node, ops_store1);
  std::string custom_op_file_path1;
  (void)ge::AttrUtils::GetStr(node->GetOpDesc(), CUSTOM_OP_FILE_PATH, custom_op_file_path1);
  EXPECT_EQ(custom_op_file_path1, "");

  (void)node->GetOpDesc()->DelAttr(CUSTOM_OP_FILE_PATH);
  FEOpsStoreInfo ops_store2;
  ops_store2.fe_ops_store_name = "tbe-custom1";
  ops_store2.is_custom_store = true;
  ops_store2.cfg_file_path = "/home/jenkins/Ascend/ascend-toolkit/latest/opp_vendors/customize/op_impl/ai_core/tbe/config/ascend910";
  fe_ops_kernel_info_store_ptr->GetAndSetCustomOpFilePath(node, ops_store2);
  std::string custom_op_file_path2;
  (void)ge::AttrUtils::GetStr(node->GetOpDesc(), CUSTOM_OP_FILE_PATH, custom_op_file_path2);
  EXPECT_EQ(custom_op_file_path2, "/home/jenkins/Ascend/ascend-toolkit/latest/opp_vendors/customize");

  (void)node->GetOpDesc()->DelAttr(CUSTOM_OP_FILE_PATH);
  FEOpsStoreInfo ops_store3;
  ops_store3.fe_ops_store_name = "tbe-builtin";
  ops_store3.is_custom_store = false;
  ops_store3.cfg_file_path = "/home/jenkins/Ascend/ascend-toolkit/latest/opp_vendors/customize/op_impl/ai_core/tbe/config/ascend910";
  fe_ops_kernel_info_store_ptr->GetAndSetCustomOpFilePath(node, ops_store3);
  std::string custom_op_file_path3;
  (void)ge::AttrUtils::GetStr(node->GetOpDesc(), CUSTOM_OP_FILE_PATH, custom_op_file_path3);
  EXPECT_EQ(custom_op_file_path3, "");
}